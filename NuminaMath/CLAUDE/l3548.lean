import Mathlib

namespace lcm_24_150_l3548_354863

theorem lcm_24_150 : Nat.lcm 24 150 = 600 := by
  sorry

end lcm_24_150_l3548_354863


namespace min_b_for_real_roots_F_monotonic_iff_l3548_354806

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - Real.log x

-- Define the function F
def F (a : ℝ) (x : ℝ) : ℝ := f a x * Real.exp (-x)

-- Theorem for part 1
theorem min_b_for_real_roots (x : ℝ) :
  ∃ (b : ℝ), b ≥ 0 ∧ ∃ (x : ℝ), x > 0 ∧ f (-1) x = b / x ∧
  ∀ (b' : ℝ), b' < b → ¬∃ (x : ℝ), x > 0 ∧ f (-1) x = b' / x :=
sorry

-- Theorem for part 2
theorem F_monotonic_iff (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → F a x₁ < F a x₂) ∨
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → F a x₁ > F a x₂) ↔
  a ≤ 2 :=
sorry

end

end min_b_for_real_roots_F_monotonic_iff_l3548_354806


namespace total_hamburgers_bought_l3548_354817

/-- Calculates the total number of hamburgers bought given the conditions --/
theorem total_hamburgers_bought
  (total_spent : ℚ)
  (single_burger_cost : ℚ)
  (double_burger_cost : ℚ)
  (double_burgers_count : ℕ)
  (h1 : total_spent = 68.5)
  (h2 : single_burger_cost = 1)
  (h3 : double_burger_cost = 1.5)
  (h4 : double_burgers_count = 37) :
  ∃ (single_burgers_count : ℕ),
    single_burgers_count + double_burgers_count = 50 ∧
    total_spent = single_burger_cost * single_burgers_count + double_burger_cost * double_burgers_count :=
by
  sorry


end total_hamburgers_bought_l3548_354817


namespace ivan_work_and_charity_l3548_354849

/-- Represents Ivan Petrovich's daily work and financial situation --/
structure IvanPetrovich where
  workDays : ℕ -- number of working days per month
  sleepHours : ℕ -- hours of sleep per day
  workHours : ℝ -- hours of work per day
  hobbyRatio : ℝ -- ratio of hobby time to work time
  hourlyRate : ℝ -- rubles earned per hour of work
  rentalIncome : ℝ -- monthly rental income in rubles
  charityRatio : ℝ -- ratio of charity donation to rest hours
  monthlyExpenses : ℝ -- monthly expenses excluding charity in rubles

/-- Theorem stating Ivan Petrovich's work hours and charity donation --/
theorem ivan_work_and_charity 
  (ivan : IvanPetrovich)
  (h1 : ivan.workDays = 21)
  (h2 : ivan.sleepHours = 8)
  (h3 : ivan.hobbyRatio = 2)
  (h4 : ivan.hourlyRate = 3000)
  (h5 : ivan.rentalIncome = 14000)
  (h6 : ivan.charityRatio = 1/3)
  (h7 : ivan.monthlyExpenses = 70000)
  (h8 : 24 = ivan.sleepHours + ivan.workHours + ivan.hobbyRatio * ivan.workHours + (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)))
  (h9 : ivan.workDays * (ivan.hourlyRate * ivan.workHours + ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000) + ivan.rentalIncome = ivan.monthlyExpenses + ivan.workDays * ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000) :
  ivan.workHours = 2 ∧ ivan.workDays * ivan.charityRatio * (24 - ivan.sleepHours - ivan.workHours * (1 + ivan.hobbyRatio)) * 1000 = 70000 := by
  sorry

end ivan_work_and_charity_l3548_354849


namespace marbles_distribution_l3548_354861

/-- Given a total number of marbles and a number of boxes, calculate the number of marbles per box -/
def marblesPerBox (totalMarbles : ℕ) (numBoxes : ℕ) : ℕ :=
  totalMarbles / numBoxes

theorem marbles_distribution (totalMarbles : ℕ) (numBoxes : ℕ) 
  (h1 : totalMarbles = 18) (h2 : numBoxes = 3) :
  marblesPerBox totalMarbles numBoxes = 6 := by
  sorry

end marbles_distribution_l3548_354861


namespace polynomial_square_l3548_354870

theorem polynomial_square (a b : ℚ) : 
  (∃ p q : ℚ, ∀ x, x^4 + x^3 - x^2 + a*x + b = (x^2 + p*x + q)^2) → 
  b = 25/64 := by
sorry

end polynomial_square_l3548_354870


namespace function_inequality_l3548_354869

/-- Given a differentiable function f: ℝ → ℝ, if f(x) + f''(x) < 0 for all x,
    then f(1) < f(0)/e < f(-1)/(e^2) -/
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f)
    (hf'' : Differentiable ℝ (deriv (deriv f)))
    (h : ∀ x, f x + (deriv (deriv f)) x < 0) :
    f 1 < f 0 / Real.exp 1 ∧ f 0 / Real.exp 1 < f (-1) / (Real.exp 1)^2 := by
  sorry

end function_inequality_l3548_354869


namespace sphere_surface_area_increase_l3548_354822

theorem sphere_surface_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.1 * r
  let original_area := 4 * Real.pi * r^2
  let new_area := 4 * Real.pi * new_radius^2
  (new_area - original_area) / original_area = 0.21 := by
sorry

end sphere_surface_area_increase_l3548_354822


namespace smallest_value_S_l3548_354826

theorem smallest_value_S (a₁ a₂ a₃ b₁ b₂ b₃ c₁ c₂ c₃ d₁ d₂ d₃ : ℕ) : 
  ({a₁, a₂, a₃, b₁, b₂, b₃, c₁, c₂, c₃, d₁, d₂, d₃} = Finset.range 12) →
  (a₁ * a₂ * a₃ + b₁ * b₂ * b₃ + c₁ * c₂ * c₃ + d₁ * d₂ * d₃ ≥ 613) ∧
  (∃ (a₁' a₂' a₃' b₁' b₂' b₃' c₁' c₂' c₃' d₁' d₂' d₃' : ℕ),
    {a₁', a₂', a₃', b₁', b₂', b₃', c₁', c₂', c₃', d₁', d₂', d₃'} = Finset.range 12 ∧
    a₁' * a₂' * a₃' + b₁' * b₂' * b₃' + c₁' * c₂' * c₃' + d₁' * d₂' * d₃' = 613) :=
by sorry

end smallest_value_S_l3548_354826


namespace total_yards_run_l3548_354877

/-- Calculates the total yards run by three athletes given their individual performances -/
theorem total_yards_run (athlete1_yards athlete2_yards athlete3_avg_yards : ℕ) 
  (games : ℕ) (h1 : games = 4) (h2 : athlete1_yards = 18) (h3 : athlete2_yards = 22) 
  (h4 : athlete3_avg_yards = 11) : 
  athlete1_yards * games + athlete2_yards * games + athlete3_avg_yards * games = 204 :=
by sorry

end total_yards_run_l3548_354877


namespace chord_slope_of_ellipse_l3548_354876

/-- Given an ellipse and a chord bisected by a point, prove the slope of the chord -/
theorem chord_slope_of_ellipse (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 / 36 + y₁^2 / 9 = 1) →  -- Point (x₁, y₁) is on the ellipse
  (x₂^2 / 36 + y₂^2 / 9 = 1) →  -- Point (x₂, y₂) is on the ellipse
  ((x₁ + x₂) / 2 = 4) →         -- Midpoint x-coordinate is 4
  ((y₁ + y₂) / 2 = 2) →         -- Midpoint y-coordinate is 2
  (y₁ - y₂) / (x₁ - x₂) = -1/2  -- Slope of the chord is -1/2
:= by sorry

end chord_slope_of_ellipse_l3548_354876


namespace vector_colinearity_l3548_354874

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 0)
def c : ℝ × ℝ := (2, 1)

theorem vector_colinearity (k : ℝ) :
  (∃ t : ℝ, t ≠ 0 ∧ (k * a.1 + b.1, k * a.2 + b.2) = (t * c.1, t * c.2)) →
  k = -1 := by
  sorry

end vector_colinearity_l3548_354874


namespace function_inequality_implies_bound_l3548_354805

theorem function_inequality_implies_bound (a : ℝ) : 
  (∃ x : ℝ, 4 - x^2 ≥ |x - a| + a) → a ≤ 17/8 := by
  sorry

end function_inequality_implies_bound_l3548_354805


namespace square_grid_perimeter_l3548_354888

theorem square_grid_perimeter (total_area : ℝ) (h_area : total_area = 144) :
  let side_length := Real.sqrt (total_area / 4)
  let perimeter := 4 * (2 * side_length)
  perimeter = 48 := by
sorry

end square_grid_perimeter_l3548_354888


namespace total_fat_served_l3548_354851

/-- The amount of fat in ounces for each type of fish --/
def herring_fat : ℕ := 40
def eel_fat : ℕ := 20
def pike_fat : ℕ := eel_fat + 10
def salmon_fat : ℕ := 35
def halibut_fat : ℕ := 50

/-- The number of each type of fish served --/
def herring_count : ℕ := 40
def eel_count : ℕ := 30
def pike_count : ℕ := 25
def salmon_count : ℕ := 20
def halibut_count : ℕ := 15

/-- The total amount of fat served --/
def total_fat : ℕ := 
  herring_fat * herring_count +
  eel_fat * eel_count +
  pike_fat * pike_count +
  salmon_fat * salmon_count +
  halibut_fat * halibut_count

theorem total_fat_served : total_fat = 4400 := by
  sorry

end total_fat_served_l3548_354851


namespace division_problem_l3548_354897

theorem division_problem (a b c : ℚ) : 
  a = (2 : ℚ) / 3 * (b + c) →
  b = (6 : ℚ) / 9 * (a + c) →
  a = 200 →
  a + b + c = 500 := by
sorry

end division_problem_l3548_354897


namespace star_value_l3548_354854

/-- Custom operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

/-- Theorem: If a + b = 15 and a * b = 36, then a * b = 5/12 -/
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum : a + b = 15) (product : a * b = 36) : 
  star a b = 5 / 12 := by
  sorry

end star_value_l3548_354854


namespace square_sum_value_l3548_354856

theorem square_sum_value (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 15) : a^2 + b^2 = 39 := by
  sorry

end square_sum_value_l3548_354856


namespace one_count_greater_than_zero_count_l3548_354865

/-- Represents the sequence of concatenated decimal representations of numbers from 1 to n -/
def concatenatedSequence (n : ℕ) : List ℕ := sorry

/-- Counts the occurrences of a specific digit in the concatenated sequence -/
def digitCount (d : ℕ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the count of '1' is always greater than the count of '0' in the sequence -/
theorem one_count_greater_than_zero_count (n : ℕ) : digitCount 1 n > digitCount 0 n := by sorry

end one_count_greater_than_zero_count_l3548_354865


namespace harmonic_mean_of_2_3_6_l3548_354873

theorem harmonic_mean_of_2_3_6 (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 6) :
  3 / (1/a + 1/b + 1/c) = 3 := by
  sorry

end harmonic_mean_of_2_3_6_l3548_354873


namespace expand_expression_l3548_354862

theorem expand_expression (x : ℝ) : 20 * (3 * x + 4) = 60 * x + 80 := by
  sorry

end expand_expression_l3548_354862


namespace simplify_expression_1_simplify_expression_2_l3548_354844

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by
  sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) (h3 : x ≠ 1) :
  ((x^2 - 2*x) / (x^2 - 4*x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 := by
  sorry

end simplify_expression_1_simplify_expression_2_l3548_354844


namespace crayons_left_correct_l3548_354810

/-- Given an initial number of crayons and a number of crayons lost or given away,
    calculate the number of crayons left. -/
def crayons_left (initial : ℕ) (lost_or_given : ℕ) : ℕ :=
  initial - lost_or_given

/-- Theorem: The number of crayons left is equal to the initial number minus
    the number lost or given away. -/
theorem crayons_left_correct (initial : ℕ) (lost_or_given : ℕ) 
  (h : lost_or_given ≤ initial) : 
  crayons_left initial lost_or_given = initial - lost_or_given :=
by sorry

end crayons_left_correct_l3548_354810


namespace work_completion_time_l3548_354841

theorem work_completion_time (x_days y_days combined_days : ℝ) 
  (hx : x_days = 15)
  (hc : combined_days = 11.25)
  (h_combined : 1 / x_days + 1 / y_days = 1 / combined_days) :
  y_days = 45 := by sorry

end work_completion_time_l3548_354841


namespace painting_price_change_l3548_354820

/-- Calculates the final price of a painting after a series of value changes and currency depreciation. -/
def final_price_percentage (initial_increase : ℝ) (first_decrease : ℝ) (second_decrease : ℝ) 
  (discount : ℝ) (currency_depreciation : ℝ) : ℝ :=
  let year1 := 1 + initial_increase
  let year2 := year1 * (1 - first_decrease)
  let year3 := year2 * (1 - second_decrease)
  let discounted := year3 * (1 - discount)
  discounted * (1 + currency_depreciation)

/-- Theorem stating that the final price of the painting is 113.373% of the original price -/
theorem painting_price_change : 
  ∀ (ε : ℝ), ε > 0 → 
  |final_price_percentage 0.30 0.15 0.10 0.05 0.20 - 1.13373| < ε :=
sorry

end painting_price_change_l3548_354820


namespace product_congruence_l3548_354868

theorem product_congruence : 65 * 76 * 87 ≡ 5 [ZMOD 25] := by sorry

end product_congruence_l3548_354868


namespace convex_quad_probability_l3548_354814

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords -/
def total_chords : ℕ := n.choose 2

/-- The probability of forming a convex quadrilateral -/
def prob_convex_quad : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability of forming a convex quadrilateral -/
theorem convex_quad_probability : prob_convex_quad = 2 / 585 := by
  sorry

end convex_quad_probability_l3548_354814


namespace probability_three_fives_in_eight_rolls_l3548_354864

/-- A fair die has 6 sides -/
def die_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of times we want the specific outcome (5 in this case) to appear -/
def target_occurrences : ℕ := 3

/-- The probability of rolling exactly 3 fives in 8 rolls of a fair die -/
theorem probability_three_fives_in_eight_rolls :
  (Nat.choose num_rolls target_occurrences : ℚ) / (die_sides ^ num_rolls) = 56 / 1679616 := by
  sorry

end probability_three_fives_in_eight_rolls_l3548_354864


namespace inequality_solution_set_sqrt_sum_inequality_l3548_354840

-- Part I
theorem inequality_solution_set (x : ℝ) :
  (|x - 5| - |2*x + 3| ≥ 1) ↔ (-7 ≤ x ∧ x ≤ 1/3) := by sorry

-- Part II
theorem sqrt_sum_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1/2) :
  Real.sqrt a + Real.sqrt b ≤ 1 := by sorry

end inequality_solution_set_sqrt_sum_inequality_l3548_354840


namespace businesspeople_neither_coffee_nor_tea_l3548_354852

theorem businesspeople_neither_coffee_nor_tea 
  (total : ℕ) 
  (coffee : ℕ) 
  (tea : ℕ) 
  (both : ℕ) 
  (h1 : total = 35)
  (h2 : coffee = 18)
  (h3 : tea = 15)
  (h4 : both = 6) :
  total - (coffee + tea - both) = 8 := by
sorry

end businesspeople_neither_coffee_nor_tea_l3548_354852


namespace only_set_B_forms_triangle_l3548_354804

-- Define a function to check if three lengths can form a triangle
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem statement
theorem only_set_B_forms_triangle :
  ¬(can_form_triangle 1 2 3) ∧
  can_form_triangle 2 3 4 ∧
  ¬(can_form_triangle 3 4 9) ∧
  ¬(can_form_triangle 2 2 4) :=
sorry

end only_set_B_forms_triangle_l3548_354804


namespace prob_three_primes_six_dice_l3548_354898

/-- The probability of rolling a prime number on a 10-sided die -/
def prob_prime_10 : ℚ := 2 / 5

/-- The probability of not rolling a prime number on a 10-sided die -/
def prob_not_prime_10 : ℚ := 3 / 5

/-- The number of ways to choose 3 dice out of 6 -/
def choose_3_from_6 : ℕ := 20

theorem prob_three_primes_six_dice : 
  (choose_3_from_6 : ℚ) * prob_prime_10^3 * prob_not_prime_10^3 = 4320 / 15625 := by
  sorry

end prob_three_primes_six_dice_l3548_354898


namespace average_of_eleven_numbers_l3548_354825

theorem average_of_eleven_numbers
  (first_six_avg : Real)
  (last_six_avg : Real)
  (sixth_number : Real)
  (h1 : first_six_avg = 58)
  (h2 : last_six_avg = 65)
  (h3 : sixth_number = 78) :
  (6 * first_six_avg + 6 * last_six_avg - sixth_number) / 11 = 60 := by
  sorry

end average_of_eleven_numbers_l3548_354825


namespace triangle_proof_l3548_354879

/-- Given a triangle ABC with sides a, b, c opposite angles A, B, C respectively,
    and vectors m and n, prove that C = π/3 and if a^2 = 2b^2 + c^2, then tan(A) = -3√3 --/
theorem triangle_proof (a b c A B C : Real) (m n : Real × Real) :
  let m_x := 2 * Real.cos (C / 2)
  let m_y := -Real.sin C
  let n_x := Real.cos (C / 2)
  let n_y := 2 * Real.sin C
  m = (m_x, m_y) →
  n = (n_x, n_y) →
  m.1 * n.1 + m.2 * n.2 = 0 →  -- m ⊥ n
  (C = Real.pi / 3 ∧ (a^2 = 2*b^2 + c^2 → Real.tan A = -3 * Real.sqrt 3)) :=
by sorry

end triangle_proof_l3548_354879


namespace intersecting_lines_coefficient_sum_l3548_354800

/-- Two lines intersecting at a point implies a specific sum of their coefficients -/
theorem intersecting_lines_coefficient_sum 
  (m b : ℝ) 
  (h1 : 8 = m * 5 + 3) 
  (h2 : 8 = 4 * 5 + b) : 
  b + m = -11 := by sorry

end intersecting_lines_coefficient_sum_l3548_354800


namespace eight_x_plus_y_value_l3548_354892

theorem eight_x_plus_y_value (x y z : ℝ) 
  (eq1 : x + 2*y - 3*z = 7) 
  (eq2 : 2*x - y + 2*z = 6) : 
  8*x + y = 32 := by sorry

end eight_x_plus_y_value_l3548_354892


namespace expression_value_l3548_354867

theorem expression_value (a b c : ℤ) (h1 : a = 12) (h2 : b = 2) (h3 : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := by
  sorry

end expression_value_l3548_354867


namespace equation_value_l3548_354859

theorem equation_value (x y : ℝ) (eq1 : 2*x + y = 8) (eq2 : x + 2*y = 10) :
  8*x^2 + 10*x*y + 8*y^2 = 164 := by
  sorry

end equation_value_l3548_354859


namespace count_distinct_lines_l3548_354824

def S : Set ℕ := {0, 1, 2, 3}

def is_valid_line (a b : ℕ) : Prop := a ∈ S ∧ b ∈ S

def distinct_lines : ℕ := sorry

theorem count_distinct_lines :
  distinct_lines = 9 :=
sorry

end count_distinct_lines_l3548_354824


namespace geometric_sequence_ratio_count_l3548_354893

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_ratio_count
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 2 * a 8 = 36)
  (h_sum : a 3 + a 7 = 15) :
  ∃ (S : Finset ℝ), (∀ q ∈ S, ∃ (a : ℕ → ℝ), is_geometric_sequence a ∧ a 2 * a 8 = 36 ∧ a 3 + a 7 = 15) ∧ S.card = 4 :=
sorry

end geometric_sequence_ratio_count_l3548_354893


namespace daily_wage_calculation_l3548_354819

/-- Proves the daily wage for a worker given total days, idle days, total pay, and idle day deduction --/
theorem daily_wage_calculation (total_days idle_days : ℕ) (total_pay idle_day_deduction : ℚ) :
  total_days = 60 →
  idle_days = 40 →
  total_pay = 280 →
  idle_day_deduction = 3 →
  ∃ (daily_wage : ℚ), 
    daily_wage * (total_days - idle_days : ℚ) - idle_day_deduction * idle_days = total_pay ∧
    daily_wage = 20 := by
  sorry

end daily_wage_calculation_l3548_354819


namespace complex_equation_solution_l3548_354836

theorem complex_equation_solution (z : ℂ) : (1 - I) * z = 2 * I → z = -1 + I := by
  sorry

end complex_equation_solution_l3548_354836


namespace election_winner_votes_l3548_354872

theorem election_winner_votes 
  (total_votes : ℕ) 
  (winner_percentage : ℚ) 
  (vote_difference : ℕ) :
  winner_percentage = 54/100 →
  vote_difference = 288 →
  ⌊(winner_percentage : ℝ) * total_votes⌋ - ⌊((1 - winner_percentage) : ℝ) * total_votes⌋ = vote_difference →
  ⌊(winner_percentage : ℝ) * total_votes⌋ = 1944 :=
by sorry

end election_winner_votes_l3548_354872


namespace sum_of_cubes_l3548_354832

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 2) : a^3 + b^3 = 9 := by
  sorry

end sum_of_cubes_l3548_354832


namespace paper_cutting_impossibility_l3548_354842

theorem paper_cutting_impossibility : ¬ ∃ m : ℕ, 1 + 3 * m = 50 := by
  sorry

end paper_cutting_impossibility_l3548_354842


namespace min_sum_squares_roots_l3548_354871

/-- The sum of squares of the roots of x^2 - (m+1)x + (m-1) = 0 is minimized when m = 0 -/
theorem min_sum_squares_roots (m : ℝ) : 
  let f : ℝ → ℝ := λ m => m^2 + 3
  let sum_squares := f m
  ∀ k : ℝ, f k ≥ f 0 := by sorry

end min_sum_squares_roots_l3548_354871


namespace yellow_tint_percentage_l3548_354896

/-- Proves that adding 5 liters of yellow tint to a 30-liter mixture
    with 30% yellow tint results in a new mixture with 40% yellow tint -/
theorem yellow_tint_percentage
  (original_volume : ℝ)
  (original_yellow_percent : ℝ)
  (added_yellow : ℝ)
  (h1 : original_volume = 30)
  (h2 : original_yellow_percent = 30)
  (h3 : added_yellow = 5) :
  let original_yellow := original_volume * (original_yellow_percent / 100)
  let new_yellow := original_yellow + added_yellow
  let new_volume := original_volume + added_yellow
  new_yellow / new_volume * 100 = 40 := by
sorry

end yellow_tint_percentage_l3548_354896


namespace max_side_length_l3548_354894

/-- A triangle with three different integer side lengths and perimeter 30 -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  different : a ≠ b ∧ b ≠ c ∧ a ≠ c
  perimeter : a + b + c = 30

/-- The maximum length of any side in a triangle with perimeter 30 and different integer side lengths is 14 -/
theorem max_side_length (t : Triangle) : t.a ≤ 14 ∧ t.b ≤ 14 ∧ t.c ≤ 14 :=
sorry

end max_side_length_l3548_354894


namespace circle_area_is_one_l3548_354848

theorem circle_area_is_one (r : ℝ) (h : r > 0) :
  (4 * (1 / (2 * Real.pi * r)) = 2 * r) → (Real.pi * r^2 = 1) := by
  sorry

end circle_area_is_one_l3548_354848


namespace ternary_121_equals_16_l3548_354821

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The ternary representation of the number -/
def ternary_121 : List Nat := [1, 2, 1]

theorem ternary_121_equals_16 :
  ternary_to_decimal ternary_121 = 16 := by
  sorry

end ternary_121_equals_16_l3548_354821


namespace brian_initial_cards_l3548_354858

def initial_cards : ℕ := 76
def cards_taken : ℕ := 59
def cards_left : ℕ := 17

theorem brian_initial_cards : initial_cards = cards_taken + cards_left := by
  sorry

end brian_initial_cards_l3548_354858


namespace mixed_number_calculation_l3548_354885

theorem mixed_number_calculation : 
  26 * (2 + 4/7 - (3 + 1/3)) + (3 + 1/5 + 2 + 3/7) = -(14 + 223/735) := by
  sorry

end mixed_number_calculation_l3548_354885


namespace range_of_x_l3548_354890

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_x (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, |a + b| + |a - b| ≥ |a| * f x) →
  ∃ x, x ∈ Set.Icc 0 4 ∧ ∀ y, (∀ z, |a + b| + |a - b| ≥ |a| * f z) → y ∈ Set.Icc 0 4 :=
by sorry

end range_of_x_l3548_354890


namespace arithmetic_sequences_ratio_l3548_354853

/-- Two arithmetic sequences and their sums -/
structure ArithmeticSequences where
  a : ℕ → ℚ
  b : ℕ → ℚ
  S : ℕ → ℚ
  T : ℕ → ℚ

/-- The main theorem -/
theorem arithmetic_sequences_ratio 
  (seq : ArithmeticSequences)
  (h : ∀ n, seq.S n / seq.T n = (3 * n - 1) / (2 * n + 3)) :
  seq.a 7 / seq.b 7 = 38 / 29 := by
  sorry

end arithmetic_sequences_ratio_l3548_354853


namespace no_common_points_range_and_max_m_l3548_354886

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (a : ℝ) (x : ℝ) := a * x
noncomputable def h (x : ℝ) := exp x / x

theorem no_common_points_range_and_max_m :
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → f x ≠ g a x) ∧
  (∃ m : ℝ, ∀ x : ℝ, x > 1/2 → f x + m / x < h x) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 1/2 → f x + m / x < h x) → m ≤ 1) :=
sorry

end no_common_points_range_and_max_m_l3548_354886


namespace largest_n_for_factorization_l3548_354801

/-- 
Given a quadratic polynomial of the form 6x^2 + nx + 144, where n is an integer,
this theorem states that the largest value of n for which the polynomial 
can be factored as the product of two linear factors with integer coefficients is 865.
-/
theorem largest_n_for_factorization : 
  (∃ (n : ℤ), ∀ (A B : ℤ), 
    (6 * A = 6 ∧ A + 6 * B = n ∧ A * B = 144) → 
    (∀ (m : ℤ), (∃ (C D : ℤ), 6 * C = 6 ∧ C + 6 * D = m ∧ C * D = 144) → m ≤ n)) ∧
  (∃ (A B : ℤ), 6 * A = 6 ∧ A + 6 * B = 865 ∧ A * B = 144) := by
  sorry

end largest_n_for_factorization_l3548_354801


namespace spinner_probability_l3548_354831

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 5/12 → p_A + p_B + p_C + p_D = 1 → p_D = 0 := by
  sorry

end spinner_probability_l3548_354831


namespace tangent_line_implies_sum_l3548_354833

noncomputable section

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := (a * x - 1) * Real.log x + b

-- Define the derivative of f(x)
def f_derivative (a x : ℝ) : ℝ := a * Real.log x + (a * x - 1) / x

theorem tangent_line_implies_sum (a b : ℝ) : 
  (∀ x, f_derivative a x = f a b x) →  -- f_derivative is the derivative of f
  f_derivative a 1 = -a →              -- Slope condition at x = 1
  f a b 1 = -a + 1 →                   -- Point condition at x = 1
  a + b = 1 := by sorry

end

end tangent_line_implies_sum_l3548_354833


namespace smallest_prime_digit_sum_28_l3548_354827

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a natural number is prime -/
def is_prime (n : ℕ) : Prop := sorry

/-- Theorem: 1999 is the smallest prime number whose digits sum to 28 -/
theorem smallest_prime_digit_sum_28 :
  (∀ p : ℕ, p < 1999 → (is_prime p ∧ digit_sum p = 28) → False) ∧
  is_prime 1999 ∧ digit_sum 1999 = 28 := by sorry

end smallest_prime_digit_sum_28_l3548_354827


namespace smallest_y_value_l3548_354802

theorem smallest_y_value (y : ℝ) (h : y > 0) :
  (y / 7 + 2 / (7 * y) = 1 / 3) → y ≥ 2 / 3 :=
by
  sorry

end smallest_y_value_l3548_354802


namespace x_intercept_of_line_A_l3548_354846

/-- A line in the coordinate plane -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- The intersection point of two lines -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Theorem: The x-intercept of line A is 2 -/
theorem x_intercept_of_line_A (lineA lineB : Line) (intersection : IntersectionPoint) :
  lineA.slope = -1 →
  lineB.slope = 5 →
  lineB.yIntercept = -10 →
  intersection.x + intersection.y = 2 →
  lineA.yIntercept - lineA.slope * intersection.x = lineB.slope * intersection.x + lineB.yIntercept →
  lineA.yIntercept = 2 →
  -lineA.slope * 2 + lineA.yIntercept = 0 := by
  sorry

#check x_intercept_of_line_A

end x_intercept_of_line_A_l3548_354846


namespace sum_sqrt_inequality_l3548_354895

theorem sum_sqrt_inequality (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c ≥ 1) 
  (sum_eq : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end sum_sqrt_inequality_l3548_354895


namespace perpendicular_line_to_parallel_plane_l3548_354815

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define the non-coincidence property
variable (non_coincident_planes : Plane → Plane → Prop)
variable (non_coincident_lines : Line → Line → Prop)

-- Theorem statement
theorem perpendicular_line_to_parallel_plane
  (α β : Plane) (m n : Line)
  (h_non_coincident_planes : non_coincident_planes α β)
  (h_non_coincident_lines : non_coincident_lines m n)
  (h_parallel_lines : parallel_lines m n)
  (h_perp_n_α : perpendicular_line_plane n α)
  (h_parallel_planes : parallel_planes α β) :
  perpendicular_line_plane m β :=
sorry

end perpendicular_line_to_parallel_plane_l3548_354815


namespace two_std_dev_below_mean_l3548_354813

-- Define the normal distribution parameters
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5

-- Define the value we want to prove
def value : ℝ := 12.5

-- Theorem statement
theorem two_std_dev_below_mean : 
  mean - 2 * std_dev = value := by
  sorry

end two_std_dev_below_mean_l3548_354813


namespace book_cost_prices_correct_l3548_354889

/-- Represents the cost and quantity information for a type of book -/
structure BookType where
  cost_per_book : ℝ
  total_cost : ℝ
  quantity : ℝ

/-- Proves that given the conditions, the cost prices for book types A and B are correct -/
theorem book_cost_prices_correct (book_a book_b : BookType)
  (h1 : book_a.cost_per_book = book_b.cost_per_book + 15)
  (h2 : book_a.total_cost = 675)
  (h3 : book_b.total_cost = 450)
  (h4 : book_a.quantity = book_b.quantity)
  (h5 : book_a.quantity = book_a.total_cost / book_a.cost_per_book)
  (h6 : book_b.quantity = book_b.total_cost / book_b.cost_per_book) :
  book_a.cost_per_book = 45 ∧ book_b.cost_per_book = 30 := by
  sorry

#check book_cost_prices_correct

end book_cost_prices_correct_l3548_354889


namespace derivative_f_at_2_l3548_354878

noncomputable def f (x : ℝ) : ℝ := (1 - x) / x + Real.log x

theorem derivative_f_at_2 : 
  deriv f 2 = 1/4 := by sorry

end derivative_f_at_2_l3548_354878


namespace insufficient_apples_l3548_354811

def apples_picked : ℕ := 150
def num_children : ℕ := 4
def apples_per_child_per_day : ℕ := 12
def days_in_week : ℕ := 7
def apples_per_pie : ℕ := 12
def num_pies : ℕ := 2
def apples_per_salad : ℕ := 15
def salads_per_week : ℕ := 2
def apples_taken_by_sister : ℕ := 5

theorem insufficient_apples :
  apples_picked < 
    (num_children * apples_per_child_per_day * days_in_week) +
    (num_pies * apples_per_pie) +
    (apples_per_salad * salads_per_week) +
    apples_taken_by_sister := by
  sorry

end insufficient_apples_l3548_354811


namespace opposite_seven_eighteen_implies_twentytwo_l3548_354803

/-- Represents a circular arrangement of people -/
structure CircularArrangement where
  total : ℕ
  is_valid : total > 0

/-- Defines the property of two positions being opposite in a circular arrangement -/
def are_opposite (c : CircularArrangement) (p1 p2 : ℕ) : Prop :=
  p1 ≤ c.total ∧ p2 ≤ c.total ∧ (2 * p1 - 1) % c.total = (2 * p2 - 1) % c.total

/-- Theorem: In a circular arrangement where the 7th person is opposite the 18th, there are 22 people -/
theorem opposite_seven_eighteen_implies_twentytwo :
  ∀ c : CircularArrangement, are_opposite c 7 18 → c.total = 22 :=
sorry

end opposite_seven_eighteen_implies_twentytwo_l3548_354803


namespace special_function_at_three_l3548_354887

/-- A function satisfying f(2x + 1) = 2f(x) + 1 for all real x, and f(0) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 * x + 1) = 2 * f x + 1) ∧ f 0 = 2

/-- The value of f(3) for a special function f -/
theorem special_function_at_three (f : ℝ → ℝ) (h : special_function f) : f 3 = 11 := by
  sorry

end special_function_at_three_l3548_354887


namespace wang_heng_birth_date_l3548_354880

theorem wang_heng_birth_date :
  ∃! (year month : ℕ),
    1901 ≤ year ∧ year ≤ 2000 ∧
    1 ≤ month ∧ month ≤ 12 ∧
    (month * 2 + 5) * 50 + year - 250 = 2088 ∧
    year = 1988 ∧
    month = 1 := by
  sorry

end wang_heng_birth_date_l3548_354880


namespace gcd_876543_765432_l3548_354855

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 9 := by
  sorry

end gcd_876543_765432_l3548_354855


namespace root_sum_cube_theorem_l3548_354847

theorem root_sum_cube_theorem (a : ℝ) (x₁ x₂ x₃ : ℝ) : 
  (x₁^3 - 6*x₁^2 + a*x₁ + a = 0) →
  (x₂^3 - 6*x₂^2 + a*x₂ + a = 0) →
  (x₃^3 - 6*x₃^2 + a*x₃ + a = 0) →
  ((x₁ - 3)^3 + (x₂ - 3)^3 + (x₃ - 3)^3 = 0) →
  (a = 9) := by
sorry

end root_sum_cube_theorem_l3548_354847


namespace cube_root_bound_l3548_354829

theorem cube_root_bound (n : ℕ) (hn : n ≥ 2) :
  (n : ℝ) + 0.6 < (((n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ))^(1/3 : ℝ)) ∧
  (((n : ℝ)^3 + 2*(n : ℝ)^2 + (n : ℝ))^(1/3 : ℝ)) < (n : ℝ) + 0.7 :=
by sorry

end cube_root_bound_l3548_354829


namespace inequality_proof_l3548_354845

theorem inequality_proof (x₁ x₂ : ℝ) (h₁ : |x₁| ≤ 1) (h₂ : |x₂| ≤ 1) :
  Real.sqrt (1 - x₁^2) + Real.sqrt (1 - x₂^2) ≤ 2 * Real.sqrt (1 - ((x₁ + x₂) / 2)^2) := by
  sorry

end inequality_proof_l3548_354845


namespace fifth_root_monotone_l3548_354843

theorem fifth_root_monotone (x y : ℝ) (h : x < y) : (x^(1/5) : ℝ) < (y^(1/5) : ℝ) := by
  sorry

end fifth_root_monotone_l3548_354843


namespace divisibility_count_l3548_354891

theorem divisibility_count : ∃! n : ℕ, n > 0 ∧ n < 500 ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 7 ∣ n := by
  sorry

end divisibility_count_l3548_354891


namespace point_movement_l3548_354809

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Calculates the new position of a point after moving with a given velocity for a certain time -/
def move (p : Point2D) (v : Vector2D) (t : ℝ) : Point2D :=
  { x := p.x + v.x * t,
    y := p.y + v.y * t }

theorem point_movement :
  let initialPoint : Point2D := { x := -10, y := 10 }
  let velocity : Vector2D := { x := 4, y := -3 }
  let time : ℝ := 5
  let finalPoint : Point2D := move initialPoint velocity time
  finalPoint = { x := 10, y := -5 } := by sorry

end point_movement_l3548_354809


namespace sum_of_cubes_zero_l3548_354812

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_sum : a / (2 * (b - c)) + b / (2 * (c - a)) + c / (2 * (a - b)) = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
sorry

end sum_of_cubes_zero_l3548_354812


namespace donut_selection_count_l3548_354881

theorem donut_selection_count :
  let n : ℕ := 5  -- number of donuts to select
  let k : ℕ := 3  -- number of donut types
  Nat.choose (n + k - 1) (k - 1) = 21 :=
by sorry

end donut_selection_count_l3548_354881


namespace not_divisible_by_121_l3548_354816

theorem not_divisible_by_121 : ∀ n : ℤ, ¬(121 ∣ (n^2 + 2*n + 2014)) := by
  sorry

end not_divisible_by_121_l3548_354816


namespace max_value_expression_l3548_354818

theorem max_value_expression (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 3) (h5 : x = y) : 
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 9/4 :=
by sorry

end max_value_expression_l3548_354818


namespace geometric_sequence_ratio_l3548_354828

/-- Given a geometric sequence {a_n} where 3a_1, (1/2)a_5, and 2a_3 form an arithmetic sequence,
    prove that (a_9 + a_10) / (a_7 + a_8) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
    (h2 : (1/2) * a 5 = (3 * a 1 + 2 * a 3) / 2) :
  (a 9 + a 10) / (a 7 + a 8) = 3 := by
sorry

end geometric_sequence_ratio_l3548_354828


namespace peytons_children_l3548_354882

/-- The number of juice boxes each child uses per week -/
def juice_boxes_per_week : ℕ := 5

/-- The number of weeks in the school year -/
def school_year_weeks : ℕ := 25

/-- The total number of juice boxes needed for all children for the entire school year -/
def total_juice_boxes : ℕ := 375

/-- Peyton's number of children -/
def num_children : ℕ := total_juice_boxes / (juice_boxes_per_week * school_year_weeks)

theorem peytons_children :
  num_children = 3 :=
sorry

end peytons_children_l3548_354882


namespace rhombus_perimeter_l3548_354850

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  4 * side = 52 := by sorry

end rhombus_perimeter_l3548_354850


namespace arithmetic_sequence_sum_divisibility_l3548_354883

theorem arithmetic_sequence_sum_divisibility :
  ∀ (x c : ℕ+), 
  ∃ (k : ℕ+), 
  15 * k = 15 * (x + 7 * c) ∧ 
  ∀ (d : ℕ+), (∀ (y z : ℕ+), ∃ (m : ℕ+), d * m = 15 * (y + 7 * z)) → d ≤ 15 :=
sorry

end arithmetic_sequence_sum_divisibility_l3548_354883


namespace correct_systematic_sample_l3548_354830

def total_products : ℕ := 60
def sample_size : ℕ := 5

def systematic_sample (start : ℕ) : List ℕ :=
  List.range sample_size |>.map (λ i => start + i * (total_products / sample_size))

theorem correct_systematic_sample :
  systematic_sample 5 = [5, 17, 29, 41, 53] := by sorry

end correct_systematic_sample_l3548_354830


namespace janet_ride_count_l3548_354835

theorem janet_ride_count (roller_coaster_tickets : ℕ) (giant_slide_tickets : ℕ) 
  (roller_coaster_rides : ℕ) (total_tickets : ℕ) :
  roller_coaster_tickets = 5 →
  giant_slide_tickets = 3 →
  roller_coaster_rides = 7 →
  total_tickets = 47 →
  ∃ (giant_slide_rides : ℕ), 
    roller_coaster_tickets * roller_coaster_rides + giant_slide_tickets * giant_slide_rides = total_tickets ∧
    giant_slide_rides = 4 := by
  sorry

end janet_ride_count_l3548_354835


namespace final_selling_price_l3548_354823

/-- The final selling price of a commodity after markup and reduction -/
theorem final_selling_price (a : ℝ) : 
  let initial_markup : ℝ := 1.25
  let price_reduction : ℝ := 0.9
  a * initial_markup * price_reduction = 1.125 * a := by sorry

end final_selling_price_l3548_354823


namespace sum_of_squared_coefficients_l3548_354899

def polynomial (x : ℝ) : ℝ := 5 * (x^4 + 2*x^3 + 3*x^2 + 1)

theorem sum_of_squared_coefficients : 
  (5^2) + (10^2) + (15^2) + (5^2) = 375 := by sorry

end sum_of_squared_coefficients_l3548_354899


namespace solution_set_l3548_354807

theorem solution_set (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let S := {(x, y, z) : ℝ × ℝ × ℝ | 
    a * x + b * y = (x - y)^2 ∧
    b * y + c * z = (y - z)^2 ∧
    c * z + a * x = (z - x)^2}
  S = {(0, 0, 0), (a, 0, 0), (0, b, 0), (0, 0, c)} := by
sorry

end solution_set_l3548_354807


namespace tangent_angle_cosine_at_e_l3548_354875

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_angle_cosine_at_e :
  let θ := Real.arctan (deriv f e)
  Real.cos θ = Real.sqrt 5 / 5 := by
sorry

end tangent_angle_cosine_at_e_l3548_354875


namespace min_people_to_ask_l3548_354866

theorem min_people_to_ask (knights : ℕ) (civilians : ℕ) : 
  knights = 50 → civilians = 15 → 
  ∃ (n : ℕ), n > civilians ∧ n - civilians ≤ knights ∧ 
  ∀ (m : ℕ), m < n → (m - civilians ≤ knights → m ≤ civilians) :=
sorry

end min_people_to_ask_l3548_354866


namespace quadratic_equation_roots_l3548_354808

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + 4*x₁ = 0 ∧ x₂^2 + 4*x₂ = 0 := by
  sorry

end quadratic_equation_roots_l3548_354808


namespace quarter_percent_of_200_l3548_354839

theorem quarter_percent_of_200 : (1 / 4 : ℚ) / 100 * 200 = (1 / 2 : ℚ) := by sorry

#eval (1 / 4 : ℚ) / 100 * 200

end quarter_percent_of_200_l3548_354839


namespace squares_ending_in_nine_l3548_354834

theorem squares_ending_in_nine (x : ℤ) :
  (x ^ 2) % 10 = 9 ↔ ∃ a : ℤ, (x = 10 * a + 3 ∨ x = 10 * a + 7) :=
by sorry

end squares_ending_in_nine_l3548_354834


namespace product_of_sum_and_difference_l3548_354860

theorem product_of_sum_and_difference (x y : ℝ) : 
  x + y = 26 → x - y = 8 → x * y = 153 := by
sorry

end product_of_sum_and_difference_l3548_354860


namespace farm_field_theorem_l3548_354857

/-- Represents the farm field and ploughing scenario -/
structure FarmField where
  totalArea : ℕ
  plannedRate : ℕ
  actualRate : ℕ
  extraDays : ℕ

/-- Calculates the area left to plough given the farm field scenario -/
def areaLeftToPlough (f : FarmField) : ℕ :=
  f.totalArea - f.actualRate * (f.totalArea / f.plannedRate + f.extraDays)

/-- Theorem stating that under the given conditions, 40 hectares are left to plough -/
theorem farm_field_theorem (f : FarmField) 
  (h1 : f.totalArea = 3780)
  (h2 : f.plannedRate = 90)
  (h3 : f.actualRate = 85)
  (h4 : f.extraDays = 2) :
  areaLeftToPlough f = 40 := by
  sorry

#eval areaLeftToPlough { totalArea := 3780, plannedRate := 90, actualRate := 85, extraDays := 2 }

end farm_field_theorem_l3548_354857


namespace cube_volume_surface_area_l3548_354838

theorem cube_volume_surface_area (x : ℝ) :
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) →
  x = Real.sqrt 3 / 72 := by
sorry

end cube_volume_surface_area_l3548_354838


namespace tom_trip_cost_l3548_354884

/-- Calculates the total cost of Tom's trip to Barbados --/
def total_trip_cost (num_vaccines : ℕ) (vaccine_cost : ℚ) (doctor_visit_cost : ℚ) 
  (insurance_coverage : ℚ) (flight_cost : ℚ) (num_nights : ℕ) (lodging_cost_per_night : ℚ) 
  (transportation_cost : ℚ) (food_cost_per_day : ℚ) (exchange_rate : ℚ) 
  (conversion_fee_rate : ℚ) : ℚ :=
  let medical_cost := num_vaccines * vaccine_cost + doctor_visit_cost
  let out_of_pocket_medical := medical_cost * (1 - insurance_coverage)
  let local_expenses := (num_nights * lodging_cost_per_night + transportation_cost + 
    num_nights * food_cost_per_day)
  let conversion_fee := local_expenses * exchange_rate * conversion_fee_rate / exchange_rate
  out_of_pocket_medical + flight_cost + local_expenses + conversion_fee

/-- Theorem stating that the total cost of Tom's trip is $3060.10 --/
theorem tom_trip_cost : 
  total_trip_cost 10 45 250 0.8 1200 7 150 200 60 2 0.03 = 3060.1 := by
  sorry

#eval total_trip_cost 10 45 250 0.8 1200 7 150 200 60 2 0.03

end tom_trip_cost_l3548_354884


namespace statement_d_not_always_true_l3548_354837

/-- Two planes are different if they are not equal -/
def different_planes (α β : Plane) : Prop := α ≠ β

/-- Two lines are different if they are not equal -/
def different_lines (m n : Line) : Prop := m ≠ n

/-- A line is perpendicular to a plane -/
def line_perp_plane (l : Line) (p : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry

/-- Two lines are perpendicular -/
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

/-- Statement D is not always true -/
theorem statement_d_not_always_true 
  (α : Plane) (m n : Line) 
  (h1 : different_lines m n) 
  (h2 : lines_perpendicular m n) 
  (h3 : line_perp_plane m α) : 
  ¬ (line_parallel_plane n α) := sorry

end statement_d_not_always_true_l3548_354837
