import Mathlib

namespace product_squared_l547_54720

theorem product_squared (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by sorry

end product_squared_l547_54720


namespace binomial_distributions_l547_54756

/-- A random variable follows a binomial distribution if it represents the number of successes
    in a fixed number of independent Bernoulli trials with the same probability of success. -/
def IsBinomialDistribution (X : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ) (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧
    ∀ k, 0 ≤ k ∧ k ≤ n → X k = (n.choose k : ℝ) * p^k * (1-p)^(n-k)

/-- The probability mass function for the number of shots needed to hit the target for the first time -/
def GeometricDistribution (p : ℝ) (X : ℕ → ℝ) : Prop :=
  0 < p ∧ p ≤ 1 ∧ ∀ k, k > 0 → X k = (1-p)^(k-1) * p

/-- The distribution of computer virus infections -/
def VirusInfection (n : ℕ) (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

/-- The distribution of hitting a target in n shots -/
def TargetHits (n : ℕ) (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

/-- The distribution of cars refueling at a gas station -/
def CarRefueling (X : ℕ → ℝ) : Prop :=
  IsBinomialDistribution X

theorem binomial_distributions (n : ℕ) (p : ℝ) (X₁ X₂ X₃ X₄ : ℕ → ℝ) :
  VirusInfection n X₁ ∧
  GeometricDistribution p X₂ ∧
  TargetHits n X₃ ∧
  CarRefueling X₄ →
  IsBinomialDistribution X₁ ∧
  ¬IsBinomialDistribution X₂ ∧
  IsBinomialDistribution X₃ ∧
  IsBinomialDistribution X₄ :=
sorry

end binomial_distributions_l547_54756


namespace exam_score_percentage_l547_54783

theorem exam_score_percentage : 
  let score1 : ℕ := 42
  let score2 : ℕ := 33
  let total_score : ℕ := score1 + score2
  (score1 : ℚ) / (total_score : ℚ) * 100 = 56 := by
sorry

end exam_score_percentage_l547_54783


namespace function_inequality_implies_a_range_l547_54793

theorem function_inequality_implies_a_range (a : ℝ) 
  (h1 : 0 < a) (h2 : a ≤ 1) :
  (∀ (x₁ x₂ : ℝ), 1 ≤ x₁ ∧ x₁ ≤ Real.exp 1 ∧ 1 ≤ x₂ ∧ x₂ ≤ Real.exp 1 → 
    x₁ + a / x₁ ≥ x₂ - Real.log x₂) →
  Real.exp 1 - 2 ≤ a ∧ a ≤ 1 := by
  sorry

#check function_inequality_implies_a_range

end function_inequality_implies_a_range_l547_54793


namespace sum_of_negatives_l547_54758

theorem sum_of_negatives : (-3) + (-9) = -12 := by
  sorry

end sum_of_negatives_l547_54758


namespace damage_cost_calculation_l547_54755

def tire_cost (prices : List ℕ) (quantities : List ℕ) : ℕ :=
  List.sum (List.zipWith (· * ·) prices quantities)

def window_cost (prices : List ℕ) : ℕ :=
  List.sum prices

def fence_cost (plank_price : ℕ) (plank_quantity : ℕ) (labor_cost : ℕ) : ℕ :=
  plank_price * plank_quantity + labor_cost

theorem damage_cost_calculation (tire_prices : List ℕ) (tire_quantities : List ℕ)
    (window_prices : List ℕ) (paint_job_cost : ℕ)
    (fence_plank_price : ℕ) (fence_plank_quantity : ℕ) (fence_labor_cost : ℕ) :
    tire_prices = [230, 250, 280] →
    tire_quantities = [2, 2, 2] →
    window_prices = [700, 800, 900] →
    paint_job_cost = 1200 →
    fence_plank_price = 35 →
    fence_plank_quantity = 5 →
    fence_labor_cost = 150 →
    tire_cost tire_prices tire_quantities +
    window_cost window_prices +
    paint_job_cost +
    fence_cost fence_plank_price fence_plank_quantity fence_labor_cost = 5445 := by
  sorry

end damage_cost_calculation_l547_54755


namespace number_problem_l547_54781

theorem number_problem : ∃ n : ℕ, 
  let sum := 555 + 445
  let diff := 555 - 445
  let quotient := 2 * diff
  n / sum = quotient ∧ n % sum = 30 → n = 220030 :=
by
  sorry

end number_problem_l547_54781


namespace mikaela_hourly_rate_l547_54733

/-- Mikaela's tutoring earnings problem -/
theorem mikaela_hourly_rate :
  ∀ (hourly_rate : ℝ),
  let first_month_hours : ℝ := 35
  let second_month_hours : ℝ := first_month_hours + 5
  let total_hours : ℝ := first_month_hours + second_month_hours
  let total_earnings : ℝ := total_hours * hourly_rate
  let personal_needs_fraction : ℝ := 4/5
  let savings : ℝ := 150
  (personal_needs_fraction * total_earnings + savings = total_earnings) →
  hourly_rate = 10 := by
sorry


end mikaela_hourly_rate_l547_54733


namespace inscribed_square_side_length_l547_54711

/-- A right triangle with an inscribed square -/
structure RightTriangleWithSquare where
  -- Sides of the right triangle
  de : ℝ
  ef : ℝ
  df : ℝ
  -- The triangle is right-angled
  is_right : de^2 + ef^2 = df^2
  -- Side lengths are positive
  de_pos : de > 0
  ef_pos : ef > 0
  df_pos : df > 0
  -- The square is inscribed in the triangle
  square_inscribed : True

/-- The side length of the inscribed square -/
def square_side_length (t : RightTriangleWithSquare) : ℝ := sorry

/-- Theorem stating that for a right triangle with sides 6, 8, and 10, 
    the inscribed square has side length 120/37 -/
theorem inscribed_square_side_length :
  let t : RightTriangleWithSquare := {
    de := 6,
    ef := 8,
    df := 10,
    is_right := by norm_num,
    de_pos := by norm_num,
    ef_pos := by norm_num,
    df_pos := by norm_num,
    square_inscribed := trivial
  }
  square_side_length t = 120 / 37 := by sorry

end inscribed_square_side_length_l547_54711


namespace book_pages_maximum_l547_54774

theorem book_pages_maximum (pages : ℕ) : pages ≤ 208 :=
by
  have h1 : pages ≤ 13 * 16 := by sorry
  have h2 : pages ≤ 11 * 20 := by sorry
  sorry

#check book_pages_maximum

end book_pages_maximum_l547_54774


namespace sqrt_neg_four_squared_equals_four_l547_54780

theorem sqrt_neg_four_squared_equals_four : Real.sqrt ((-4)^2) = 4 := by
  sorry

end sqrt_neg_four_squared_equals_four_l547_54780


namespace milestone_solution_l547_54764

def milestone_problem (initial_number : ℕ) (second_number : ℕ) (third_number : ℕ) : Prop :=
  let a := initial_number / 10
  let b := initial_number % 10
  (initial_number = 10 * a + b) ∧
  (second_number = 10 * b + a) ∧
  (third_number = 100 * a + b) ∧
  (0 < a) ∧ (a < 10) ∧ (0 < b) ∧ (b < 10)

theorem milestone_solution :
  ∃ (initial_number second_number : ℕ),
    milestone_problem initial_number second_number 106 :=
  sorry

end milestone_solution_l547_54764


namespace red_ant_percentage_l547_54721

/-- Proves that the percentage of red ants in the population is 85%, given the specified conditions. -/
theorem red_ant_percentage (female_red_percentage : ℝ) (male_red_total_percentage : ℝ) :
  female_red_percentage = 45 →
  male_red_total_percentage = 46.75 →
  ∃ (red_percentage : ℝ),
    red_percentage = 85 ∧
    (100 - female_red_percentage) / 100 * red_percentage = male_red_total_percentage :=
by sorry

end red_ant_percentage_l547_54721


namespace diagonal_length_l547_54771

structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : sorry)
  (diagonal_bisects : sorry)
  (AB_eq_CD : dist A B = dist C D)
  (BC_eq_AD : dist B C = dist A D)
  (AB_length : dist A B = 5)
  (BC_length : dist B C = 3)

/-- The length of the diagonal AC in the given parallelogram is 5√2 -/
theorem diagonal_length (p : Parallelogram) : dist p.A p.C = 5 * Real.sqrt 2 := by
  sorry

end diagonal_length_l547_54771


namespace remainder_53_pow_10_mod_8_l547_54732

theorem remainder_53_pow_10_mod_8 : 53^10 % 8 = 1 := by
  sorry

end remainder_53_pow_10_mod_8_l547_54732


namespace fraction_equality_l547_54798

theorem fraction_equality (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
    (h3 : (4 * a + b) / (a - 4 * b) = 3) : 
  (a + 4 * b) / (4 * a - b) = 9 / 53 := by
  sorry

end fraction_equality_l547_54798


namespace least_whole_number_for_ratio_l547_54757

theorem least_whole_number_for_ratio : 
  ∃ x : ℕ, x > 0 ∧ 
    (∀ y : ℕ, y > 0 → y < x → (6 - y : ℚ) / (7 - y) ≥ 16 / 21) ∧
    (6 - x : ℚ) / (7 - x) < 16 / 21 :=
by
  use 3
  sorry

end least_whole_number_for_ratio_l547_54757


namespace no_perfect_squares_in_sequence_l547_54751

def x : ℕ → ℤ
  | 0 => 1
  | 1 => 3
  | (n + 2) => 6 * x (n + 1) - x n

theorem no_perfect_squares_in_sequence : ∀ n : ℕ, ¬∃ k : ℤ, x n = k ^ 2 := by
  sorry

end no_perfect_squares_in_sequence_l547_54751


namespace least_subtraction_for_divisibility_l547_54791

theorem least_subtraction_for_divisibility : 
  ∃! x : ℕ, x ≤ 953 ∧ (218791 - x) % 953 = 0 ∧ ∀ y : ℕ, y < x → (218791 - y) % 953 ≠ 0 :=
by sorry

end least_subtraction_for_divisibility_l547_54791


namespace solve_equation_l547_54734

theorem solve_equation : ∃! y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 5 * y) ∧ y = 250 / 7 := by
  sorry

end solve_equation_l547_54734


namespace prob_draw_club_is_one_fourth_l547_54747

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Represents the suits in a deck -/
inductive Suit
| Spades | Hearts | Diamonds | Clubs

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The total number of cards in the deck -/
def total_cards : ℕ := 52

/-- The probability of drawing a club from the top of a shuffled deck -/
def prob_draw_club (d : Deck) : ℚ :=
  cards_per_suit / total_cards

theorem prob_draw_club_is_one_fourth (d : Deck) :
  prob_draw_club d = 1 / 4 := by
  sorry

#check prob_draw_club_is_one_fourth

end prob_draw_club_is_one_fourth_l547_54747


namespace sqrt_two_irrational_l547_54704

-- Define what it means for a real number to be rational
def IsRational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = (a : ℝ) / (b : ℝ)

-- State the theorem
theorem sqrt_two_irrational : ¬ IsRational (Real.sqrt 2) := by
  sorry

end sqrt_two_irrational_l547_54704


namespace tangent_circle_center_slope_l547_54714

-- Define the circles u₁ and u₂
def u₁ (x y : ℝ) : Prop := x^2 + y^2 + 8*x - 20*y - 32 = 0
def u₂ (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 20*y + 128 = 0

-- Define the condition for a point (x, y) to be on the line y = bx
def on_line (x y b : ℝ) : Prop := y = b * x

-- Define the condition for a circle to be externally tangent to u₁
def externally_tangent_u₁ (x y r : ℝ) : Prop :=
  r + 12 = Real.sqrt ((x + 4)^2 + (y - 10)^2)

-- Define the condition for a circle to be internally tangent to u₂
def internally_tangent_u₂ (x y r : ℝ) : Prop :=
  8 - r = Real.sqrt ((x - 4)^2 + (y - 10)^2)

-- State the theorem
theorem tangent_circle_center_slope :
  ∃ n : ℝ, 
    (∀ b : ℝ, b > 0 → 
      (∃ x y r : ℝ, 
        on_line x y b ∧ 
        externally_tangent_u₁ x y r ∧ 
        internally_tangent_u₂ x y r) → 
      n ≤ b) ∧
    n^2 = 69/25 := by sorry

end tangent_circle_center_slope_l547_54714


namespace lines_dont_form_triangle_iff_l547_54797

/-- A line in 2D space represented by ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The three lines given in the problem -/
def line1 : Line := ⟨4, 1, 4⟩
def line2 (m : ℝ) : Line := ⟨m, 1, 0⟩
def line3 (m : ℝ) : Line := ⟨2, -3*m, 4⟩

/-- The condition for the lines not forming a triangle -/
def lines_dont_form_triangle (m : ℝ) : Prop :=
  are_parallel line1 (line2 m) ∨ 
  are_parallel line1 (line3 m) ∨ 
  are_parallel (line2 m) (line3 m)

theorem lines_dont_form_triangle_iff (m : ℝ) : 
  lines_dont_form_triangle m ↔ m = 4 ∨ m = -1/6 := by sorry

end lines_dont_form_triangle_iff_l547_54797


namespace train_length_l547_54712

theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 870)
  (h2 : platform_time = 39)
  (h3 : pole_time = 10) :
  let train_length := (platform_length * pole_time) / (platform_time - pole_time)
  train_length = 300 := by sorry

end train_length_l547_54712


namespace cupcakes_theorem_l547_54784

/-- The number of children sharing the cupcakes -/
def num_children : ℕ := 8

/-- The number of cupcakes each child gets when shared equally -/
def cupcakes_per_child : ℕ := 12

/-- The total number of cupcakes -/
def total_cupcakes : ℕ := num_children * cupcakes_per_child

theorem cupcakes_theorem : total_cupcakes = 96 := by
  sorry

end cupcakes_theorem_l547_54784


namespace arithmetic_sequence_problem_l547_54750

/-- An arithmetic sequence with its sum -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The problem statement -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 2)
  (h2 : seq.S 3 = 12) :
  seq.a 5 = 10 := by
  sorry


end arithmetic_sequence_problem_l547_54750


namespace effective_area_percentage_difference_l547_54735

/-- Calculates the effective area percentage difference between two circular fields -/
theorem effective_area_percentage_difference
  (r1 r2 : ℝ)  -- radii of the two fields
  (sqi1 sqi2 : ℝ)  -- soil quality indices
  (wa1 wa2 : ℝ)  -- water allocations
  (cyf1 cyf2 : ℝ)  -- crop yield factors
  (h_ratio : r2 = (10 / 4) * r1)  -- radius ratio condition
  (h_sqi1 : sqi1 = 0.8)
  (h_sqi2 : sqi2 = 1.2)
  (h_wa1 : wa1 = 15000)
  (h_wa2 : wa2 = 30000)
  (h_cyf1 : cyf1 = 1.5)
  (h_cyf2 : cyf2 = 2) :
  let ea1 := π * r1^2 * sqi1 * wa1 * cyf1
  let ea2 := π * r2^2 * sqi2 * wa2 * cyf2
  (ea2 - ea1) / ea1 * 100 = 1566.67 := by
  sorry

end effective_area_percentage_difference_l547_54735


namespace particular_number_problem_l547_54788

theorem particular_number_problem : ∃! x : ℚ, 2 * (67 - (x / 23)) = 102 := by sorry

end particular_number_problem_l547_54788


namespace exists_number_with_specific_digit_sum_l547_54753

/-- Sum of digits function -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of a number with specific digit sum properties -/
theorem exists_number_with_specific_digit_sum : 
  ∃ n : ℕ, n > 0 ∧ digit_sum n = 1000 ∧ digit_sum (n^2) = 1000^2 := by sorry

end exists_number_with_specific_digit_sum_l547_54753


namespace sqrt_20_in_terms_of_a_and_b_l547_54790

theorem sqrt_20_in_terms_of_a_and_b (a b : ℝ) (ha : a = Real.sqrt 2) (hb : b = Real.sqrt 10) :
  Real.sqrt 20 = a * b := by
  sorry

end sqrt_20_in_terms_of_a_and_b_l547_54790


namespace study_group_probability_l547_54717

/-- Given a study group where 70% of members are women and 40% of women are lawyers,
    the probability of randomly selecting a woman lawyer is 0.28. -/
theorem study_group_probability (total : ℕ) (women : ℕ) (women_lawyers : ℕ)
    (h1 : women = (70 : ℕ) * total / 100)
    (h2 : women_lawyers = (40 : ℕ) * women / 100) :
    (women_lawyers : ℚ) / total = 28 / 100 := by
  sorry

end study_group_probability_l547_54717


namespace vasya_lives_on_fifth_floor_l547_54739

/-- The floor on which Vasya lives -/
def vasya_floor (petya_steps : ℕ) (vasya_steps : ℕ) : ℕ :=
  let steps_per_floor := petya_steps / 2
  1 + vasya_steps / steps_per_floor

/-- Theorem stating that Vasya lives on the 5th floor -/
theorem vasya_lives_on_fifth_floor :
  vasya_floor 36 72 = 5 := by
  sorry

end vasya_lives_on_fifth_floor_l547_54739


namespace square_roots_to_N_l547_54738

theorem square_roots_to_N (m : ℝ) (N : ℝ) : 
  (3 * m - 4) ^ 2 = N ∧ (7 - 4 * m) ^ 2 = N → N = 25 := by
  sorry

end square_roots_to_N_l547_54738


namespace sandys_correct_sums_l547_54763

theorem sandys_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (correct_marks : ℕ) 
  (incorrect_marks : ℕ) 
  (h1 : total_sums = 30) 
  (h2 : total_marks = 55) 
  (h3 : correct_marks = 3) 
  (h4 : incorrect_marks = 2) : 
  ∃ (correct_sums : ℕ), 
    correct_sums * correct_marks - (total_sums - correct_sums) * incorrect_marks = total_marks ∧ 
    correct_sums = 23 := by
  sorry

end sandys_correct_sums_l547_54763


namespace retail_overhead_expenses_l547_54796

/-- A problem about calculating overhead expenses in retail --/
theorem retail_overhead_expenses 
  (purchase_price : ℝ) 
  (selling_price : ℝ) 
  (profit_percent : ℝ) 
  (h1 : purchase_price = 225)
  (h2 : selling_price = 300)
  (h3 : profit_percent = 25) :
  ∃ (overhead_expenses : ℝ),
    selling_price = (purchase_price + overhead_expenses) * (1 + profit_percent / 100) ∧
    overhead_expenses = 15 := by
  sorry

end retail_overhead_expenses_l547_54796


namespace gift_payment_l547_54770

theorem gift_payment (total : ℝ) (alice bob carlos : ℝ) : 
  total = 120 ∧ 
  alice = (1/3) * (bob + carlos) ∧ 
  bob = (1/4) * (alice + carlos) ∧ 
  total = alice + bob + carlos → 
  carlos = 72 := by
sorry

end gift_payment_l547_54770


namespace total_frogs_in_pond_l547_54761

def frogs_on_lilypads : ℕ := 5
def frogs_on_logs : ℕ := 3
def dozen : ℕ := 12
def baby_frogs_dozens : ℕ := 2

theorem total_frogs_in_pond : 
  frogs_on_lilypads + frogs_on_logs + baby_frogs_dozens * dozen = 32 := by
  sorry

end total_frogs_in_pond_l547_54761


namespace twenty_fifth_digit_is_zero_l547_54766

/-- The decimal representation of 1/13 -/
def decimal_1_13 : ℚ := 1 / 13

/-- The decimal representation of 1/11 -/
def decimal_1_11 : ℚ := 1 / 11

/-- The sum of the decimal representations of 1/13 and 1/11 -/
def sum_decimals : ℚ := decimal_1_13 + decimal_1_11

/-- The function that returns the nth digit after the decimal point of a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ := sorry

/-- Theorem: The 25th digit after the decimal point in the sum of 1/13 and 1/11 is 0 -/
theorem twenty_fifth_digit_is_zero : nth_digit_after_decimal sum_decimals 25 = 0 := by sorry

end twenty_fifth_digit_is_zero_l547_54766


namespace units_digit_sum_factorials_100_l547_54742

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_100 : sum_factorials 100 % 10 = 3 := by
  sorry

end units_digit_sum_factorials_100_l547_54742


namespace solution_set_a_neg_one_range_of_a_for_nonnegative_f_l547_54731

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * x

-- Part 1: Solution set when a = -1
theorem solution_set_a_neg_one :
  {x : ℝ | f (-1) x ≤ 0} = {x : ℝ | x ≤ -1/3} := by sorry

-- Part 2: Range of a for f(x) ≥ 0 when x ≥ -1
theorem range_of_a_for_nonnegative_f :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ -1 → f a x ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) := by sorry

end solution_set_a_neg_one_range_of_a_for_nonnegative_f_l547_54731


namespace dave_total_earnings_l547_54723

/-- Calculates daily earnings after tax -/
def dailyEarnings (hourlyWage : ℚ) (hoursWorked : ℚ) (unpaidBreak : ℚ) : ℚ :=
  let actualHours := hoursWorked - unpaidBreak
  let earningsBeforeTax := actualHours * hourlyWage
  let taxDeduction := earningsBeforeTax * (1 / 10)
  earningsBeforeTax - taxDeduction

/-- Represents Dave's total earnings for the week -/
def daveEarnings : ℚ :=
  dailyEarnings 6 6 (1/2) +  -- Monday
  dailyEarnings 7 2 (1/4) +  -- Tuesday
  dailyEarnings 9 3 0 +      -- Wednesday
  dailyEarnings 8 5 (1/2)    -- Thursday

theorem dave_total_earnings :
  daveEarnings = 9743 / 100 := by sorry

end dave_total_earnings_l547_54723


namespace base6_addition_l547_54767

-- Define a function to convert a base-6 number (represented as a list of digits) to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

-- Define a function to convert a natural number to its base-6 representation
def natToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
  aux n []

-- State the theorem
theorem base6_addition :
  base6ToNat [4, 5, 1, 2] + base6ToNat [2, 3, 4, 5, 3] = base6ToNat [3, 4, 4, 0, 5] := by
  sorry

end base6_addition_l547_54767


namespace sum_of_squares_not_prime_l547_54719

theorem sum_of_squares_not_prime (a b c d : ℕ+) (h : a * b = c * d) :
  ¬ Nat.Prime (a^2 + b^2 + c^2 + d^2) := by
  sorry

end sum_of_squares_not_prime_l547_54719


namespace number_puzzle_l547_54713

theorem number_puzzle (N : ℚ) : 
  (N / (4/5)) = ((4/5) * N + 18) → N = 40 := by
sorry

end number_puzzle_l547_54713


namespace root_product_sum_l547_54702

theorem root_product_sum (p q r : ℂ) : 
  (6 * p^3 - 9 * p^2 + 16 * p - 12 = 0) →
  (6 * q^3 - 9 * q^2 + 16 * q - 12 = 0) →
  (6 * r^3 - 9 * r^2 + 16 * r - 12 = 0) →
  p * q + p * r + q * r = 8/3 := by
sorry

end root_product_sum_l547_54702


namespace smallest_prime_divisor_of_sum_l547_54700

theorem smallest_prime_divisor_of_sum (n : ℕ) :
  (n = 3^15 + 11^13) → (Nat.minFac n = 2) := by
  sorry

end smallest_prime_divisor_of_sum_l547_54700


namespace ab_positive_necessary_not_sufficient_l547_54716

theorem ab_positive_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, a * b > 0 ∧ b / a + a / b ≤ 2) ∧
  (∀ a b : ℝ, b / a + a / b > 2 → a * b > 0) :=
sorry

end ab_positive_necessary_not_sufficient_l547_54716


namespace stratified_sampling_under_40_l547_54799

/-- Represents the total number of teachers -/
def total_teachers : ℕ := 400

/-- Represents the number of teachers under 40 -/
def teachers_under_40 : ℕ := 250

/-- Represents the total sample size -/
def sample_size : ℕ := 80

/-- Calculates the number of teachers under 40 in the sample -/
def sample_under_40 : ℕ := (teachers_under_40 * sample_size) / total_teachers

theorem stratified_sampling_under_40 :
  sample_under_40 = 50 :=
sorry

end stratified_sampling_under_40_l547_54799


namespace flag_making_problem_l547_54741

/-- The number of students in each group making flags -/
def students_per_group : ℕ := 10

/-- The total number of flags to be made -/
def total_flags : ℕ := 240

/-- The number of groups initially assigned to make flags -/
def initial_groups : ℕ := 3

/-- The number of groups after reassignment -/
def final_groups : ℕ := 2

/-- The additional number of flags each student has to make after reassignment -/
def additional_flags_per_student : ℕ := 4

theorem flag_making_problem :
  (total_flags / final_groups - total_flags / initial_groups) / students_per_group = additional_flags_per_student :=
by sorry

end flag_making_problem_l547_54741


namespace sixtieth_pair_is_five_seven_l547_54759

/-- Represents a pair of integers -/
structure IntPair :=
  (first : ℕ)
  (second : ℕ)

/-- The sequence of integer pairs -/
def pairSequence : ℕ → IntPair :=
  sorry

/-- The 60th pair in the sequence -/
def sixtiethPair : IntPair :=
  pairSequence 60

theorem sixtieth_pair_is_five_seven :
  sixtiethPair = IntPair.mk 5 7 := by
  sorry

end sixtieth_pair_is_five_seven_l547_54759


namespace bus_trip_speed_l547_54776

theorem bus_trip_speed (distance : ℝ) (speed_increase : ℝ) (time_decrease : ℝ) : 
  distance = 500 ∧ 
  speed_increase = 10 ∧ 
  time_decrease = 2 →
  ∃ (v : ℝ), v > 0 ∧ 
    distance / v - distance / (v + speed_increase) = time_decrease ∧ 
    v = 45.25 := by
  sorry

end bus_trip_speed_l547_54776


namespace coral_reading_pages_l547_54778

def pages_night1 : ℕ := 30

def pages_night2 : ℕ := 2 * pages_night1 - 2

def pages_night3 : ℕ := pages_night1 + pages_night2 + 3

def total_pages : ℕ := pages_night1 + pages_night2 + pages_night3

theorem coral_reading_pages : total_pages = 179 := by
  sorry

end coral_reading_pages_l547_54778


namespace smallest_multiplier_for_ten_zeros_l547_54743

theorem smallest_multiplier_for_ten_zeros (n : ℕ) : 
  (∀ m : ℕ, m < 78125000 → ¬(∃ k : ℕ, 128 * m = k * 10^10)) ∧ 
  (∃ k : ℕ, 128 * 78125000 = k * 10^10) := by
  sorry

end smallest_multiplier_for_ten_zeros_l547_54743


namespace tangent_line_at_origin_l547_54726

/-- Given a real number a and a function f(x) = x^3 + ax^2 + (a - 2)x where its derivative f'(x) is an even function,
    the equation of the tangent line to the curve y = f(x) at the origin is y = -2x. -/
theorem tangent_line_at_origin (a : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, f x = x^3 + a*x^2 + (a - 2)*x) →
  (∀ x, (deriv f) x = f' x) →
  (∀ x, f' x = f' (-x)) →
  (∀ x, x * (-2) = f x - f 0) :=
sorry

end tangent_line_at_origin_l547_54726


namespace stating_school_travel_time_l547_54706

/-- Represents the time in minutes to get from home to school -/
def time_to_school : ℕ := 12

/-- Represents the fraction of the way Kolya walks before realizing he forgot his book -/
def initial_fraction : ℚ := 1/4

/-- Represents the time in minutes Kolya arrives early if he doesn't go back -/
def early_time : ℕ := 5

/-- Represents the time in minutes Kolya arrives late if he goes back -/
def late_time : ℕ := 1

/-- 
Theorem stating that the time to get to school is 12 minutes, given the conditions of the problem.
-/
theorem school_travel_time :
  time_to_school = 12 ∧
  initial_fraction = 1/4 ∧
  early_time = 5 ∧
  late_time = 1 →
  time_to_school = 12 :=
by
  sorry


end stating_school_travel_time_l547_54706


namespace range_of_function_l547_54725

theorem range_of_function (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end range_of_function_l547_54725


namespace arithmetic_mean_of_numbers_l547_54787

def numbers : List ℝ := [17, 25, 38]

theorem arithmetic_mean_of_numbers :
  (numbers.sum / numbers.length : ℝ) = 80 / 3 := by
  sorry

end arithmetic_mean_of_numbers_l547_54787


namespace simplify_inverse_sum_product_l547_54701

theorem simplify_inverse_sum_product (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end simplify_inverse_sum_product_l547_54701


namespace ellipse_line_slope_product_l547_54718

/-- Given an ellipse C and a line l intersecting C at two points, 
    prove that the product of slopes of OM and l is -9 --/
theorem ellipse_line_slope_product (x₁ x₂ y₁ y₂ : ℝ) 
  (hC₁ : 9 * x₁^2 + y₁^2 = 1)
  (hC₂ : 9 * x₂^2 + y₂^2 = 1)
  (h_not_origin : x₁ ≠ 0 ∨ y₁ ≠ 0)
  (h_not_parallel : x₁ ≠ x₂ ∧ y₁ ≠ y₂) :
  let k_OM := (y₁ + y₂) / (x₁ + x₂)
  let k_l := (y₁ - y₂) / (x₁ - x₂)
  k_OM * k_l = -9 := by
  sorry

end ellipse_line_slope_product_l547_54718


namespace average_visitors_is_276_l547_54730

/-- Calculates the average number of visitors per day in a 30-day month starting on Sunday -/
def averageVisitorsPerDay (sundayVisitors : ℕ) (otherDayVisitors : ℕ) : ℚ :=
  let totalDays : ℕ := 30
  let sundays : ℕ := 4
  let otherDays : ℕ := totalDays - sundays
  let totalVisitors : ℕ := sundays * sundayVisitors + otherDays * otherDayVisitors
  (totalVisitors : ℚ) / totalDays

/-- Theorem: The average number of visitors per day is 276 -/
theorem average_visitors_is_276 :
  averageVisitorsPerDay 510 240 = 276 := by
  sorry


end average_visitors_is_276_l547_54730


namespace geometric_sequence_s6_l547_54709

/-- A geometric sequence with its sum -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum of the first n terms
  is_geometric : ∀ n : ℕ, n > 0 → a (n + 1) / a n = a 2 / a 1

/-- Theorem stating the result for S_6 given the conditions -/
theorem geometric_sequence_s6 (seq : GeometricSequence) 
  (h1 : seq.a 3 = 4)
  (h2 : seq.S 3 = 7) :
  seq.S 6 = 63 ∨ seq.S 6 = 133/27 := by
  sorry

end geometric_sequence_s6_l547_54709


namespace greatest_integer_solution_greatest_integer_value_minus_four_is_solution_minus_four_is_greatest_l547_54795

theorem greatest_integer_solution (x : ℤ) : (5 - 4 * x > 17) ↔ x < -3 :=
  sorry

theorem greatest_integer_value : ∀ x : ℤ, (5 - 4 * x > 17) → x ≤ -4 :=
  sorry

theorem minus_four_is_solution : 5 - 4 * (-4) > 17 :=
  sorry

theorem minus_four_is_greatest : ∀ x : ℤ, x > -4 → ¬(5 - 4 * x > 17) :=
  sorry

end greatest_integer_solution_greatest_integer_value_minus_four_is_solution_minus_four_is_greatest_l547_54795


namespace ipod_final_price_l547_54703

/-- Calculates the final price of an item after two discounts and a compound sales tax. -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_discount1 := original_price * (1 - discount1)
  let price_after_discount2 := price_after_discount1 * (1 - discount2)
  price_after_discount2 * (1 + tax_rate)

/-- Theorem stating that the final price of the iPod is approximately $77.08 -/
theorem ipod_final_price :
  ∃ ε > 0, |final_price 128 (7/20) 0.15 0.09 - 77.08| < ε :=
sorry

end ipod_final_price_l547_54703


namespace at_least_one_leq_neg_two_l547_54715

theorem at_least_one_leq_neg_two (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) : 
  (a + 1/b ≤ -2) ∨ (b + 1/c ≤ -2) ∨ (c + 1/a ≤ -2) := by
  sorry

end at_least_one_leq_neg_two_l547_54715


namespace days_from_thursday_l547_54727

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advance_days (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | m + 1 => next_day (advance_days start m)

theorem days_from_thursday :
  advance_days DayOfWeek.Thursday 53 = DayOfWeek.Monday := by
  sorry


end days_from_thursday_l547_54727


namespace order_relation_l547_54728

theorem order_relation (a b c : ℝ) : 
  a = (1 : ℝ) / 2023 →
  b = Real.exp (-(2022 : ℝ) / 2023) →
  c = Real.cos ((1 : ℝ) / 2023) / 2023 →
  b > a ∧ a > c := by
  sorry

end order_relation_l547_54728


namespace arithmetic_sequence_length_l547_54710

theorem arithmetic_sequence_length : 
  ∀ (a₁ n d : ℕ) (aₙ : ℕ), 
    a₁ = 3 → 
    d = 3 → 
    aₙ = 144 → 
    aₙ = a₁ + (n - 1) * d → 
    n = 48 := by
  sorry

end arithmetic_sequence_length_l547_54710


namespace triangle_problem_l547_54729

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 2 * Real.sqrt 3 →
  B = π / 3 ∧ 
  (∀ (a' c' : ℝ), a' * c' ≤ 12) ∧
  (∃ (a' c' : ℝ), a' * c' = 12) :=
by sorry

end triangle_problem_l547_54729


namespace is_cylinder_l547_54777

/-- Represents the shape of a view in an orthographic projection --/
inductive ViewShape
  | Circle
  | Rectangle

/-- Represents the three orthographic views of a solid --/
structure OrthographicViews where
  top : ViewShape
  front : ViewShape
  side : ViewShape

/-- Represents different types of solids --/
inductive Solid
  | Sphere
  | Cylinder
  | Cone
  | Cuboid

/-- Given the three orthographic views of a solid, determine if it is a cylinder --/
theorem is_cylinder (views : OrthographicViews) :
  views.top = ViewShape.Circle ∧ 
  views.front = ViewShape.Rectangle ∧ 
  views.side = ViewShape.Rectangle → 
  ∃ (s : Solid), s = Solid.Cylinder :=
sorry

end is_cylinder_l547_54777


namespace vector_magnitude_proof_l547_54736

/-- Given two planar vectors a and b, prove that the magnitude of (a - 2b) is 5. -/
theorem vector_magnitude_proof (a b : ℝ × ℝ) :
  a = (-2, 1) →
  b = (1, 2) →
  ‖a - 2 • b‖ = 5 := by
  sorry

end vector_magnitude_proof_l547_54736


namespace points_earned_proof_l547_54760

def video_game_points (total_enemies : ℕ) (points_per_enemy : ℕ) (enemies_not_destroyed : ℕ) : ℕ :=
  (total_enemies - enemies_not_destroyed) * points_per_enemy

theorem points_earned_proof :
  video_game_points 8 5 6 = 10 := by
sorry

end points_earned_proof_l547_54760


namespace athlete_heartbeats_l547_54765

/-- Calculates the total number of heartbeats during an athlete's activity --/
def totalHeartbeats (joggingHeartRate walkingHeartRate : ℕ) 
                    (walkingDuration : ℕ) 
                    (joggingDistance joggingPace : ℕ) : ℕ :=
  let joggingDuration := joggingDistance * joggingPace
  let joggingBeats := joggingDuration * joggingHeartRate
  let walkingBeats := walkingDuration * walkingHeartRate
  joggingBeats + walkingBeats

/-- Proves that the total number of heartbeats is 9900 given the specified conditions --/
theorem athlete_heartbeats :
  totalHeartbeats 120 90 30 10 6 = 9900 := by
  sorry

end athlete_heartbeats_l547_54765


namespace problem_solution_l547_54754

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 * y = 2)
  (h2 : y^2 * z = 4)
  (h3 : z^2 / x = 5) :
  x = 5^(1/7) := by
sorry

end problem_solution_l547_54754


namespace percentage_relationship_l547_54773

theorem percentage_relationship (x : ℝ) (p : ℝ) :
  x = 120 →
  5.76 = p * (0.4 * x) →
  p = 0.12 :=
by sorry

end percentage_relationship_l547_54773


namespace f_max_min_values_l547_54740

-- Define the function f(x) = |x-2| + |x-3| - |x-1|
def f (x : ℝ) : ℝ := |x - 2| + |x - 3| - |x - 1|

-- Define the condition that |x-2| + |x-3| is minimized
def is_minimized (x : ℝ) : Prop := 2 ≤ x ∧ x ≤ 3

-- Theorem statement
theorem f_max_min_values :
  (∃ (x : ℝ), is_minimized x) →
  (∃ (max min : ℝ), 
    (∀ (y : ℝ), is_minimized y → f y ≤ max) ∧
    (∃ (z : ℝ), is_minimized z ∧ f z = max) ∧
    (∀ (y : ℝ), is_minimized y → min ≤ f y) ∧
    (∃ (z : ℝ), is_minimized z ∧ f z = min) ∧
    max = 0 ∧ min = -1) :=
sorry

end f_max_min_values_l547_54740


namespace waitress_tips_l547_54775

theorem waitress_tips (salary : ℝ) (tips : ℝ) (h1 : salary > 0) (h2 : tips > 0) :
  tips / (salary + tips) = 1/3 → tips / salary = 1/2 :=
by sorry

end waitress_tips_l547_54775


namespace expected_allergies_in_sample_l547_54707

/-- The probability that an American suffers from allergies -/
def allergy_probability : ℚ := 1 / 5

/-- The size of the random sample -/
def sample_size : ℕ := 250

/-- The expected number of Americans with allergies in the sample -/
def expected_allergies : ℚ := allergy_probability * sample_size

theorem expected_allergies_in_sample :
  expected_allergies = 50 := by sorry

end expected_allergies_in_sample_l547_54707


namespace base_conversion_256_to_base_5_l547_54746

theorem base_conversion_256_to_base_5 :
  (2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0) = 256 := by
  sorry

end base_conversion_256_to_base_5_l547_54746


namespace sine_equality_l547_54745

/-- Given three nonzero real numbers a, b, c and three real angles α, β, γ,
    if a sin α + b sin β + c sin γ = 0 and a cos α + b cos β + c cos γ = 0,
    then sin(β - γ)/a = sin(γ - α)/b = sin(α - β)/c -/
theorem sine_equality (a b c α β γ : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a * Real.sin α + b * Real.sin β + c * Real.sin γ = 0)
  (h2 : a * Real.cos α + b * Real.cos β + c * Real.cos γ = 0) :
  Real.sin (β - γ) / a = Real.sin (γ - α) / b ∧ 
  Real.sin (γ - α) / b = Real.sin (α - β) / c :=
by sorry

end sine_equality_l547_54745


namespace duck_flying_days_l547_54789

/-- The number of days it takes a duck to fly south during winter -/
def days_south : ℕ := 40

/-- The number of days it takes a duck to fly north during summer -/
def days_north : ℕ := 2 * days_south

/-- The number of days it takes a duck to fly east during spring -/
def days_east : ℕ := 60

/-- The total number of days a duck flies during winter, summer, and spring -/
def total_flying_days : ℕ := days_south + days_north + days_east

/-- Theorem stating that the total number of days a duck flies during winter, summer, and spring is 180 -/
theorem duck_flying_days : total_flying_days = 180 := by
  sorry

end duck_flying_days_l547_54789


namespace place_value_comparison_l547_54749

def number : ℚ := 52648.2097

def tens_place_value : ℚ := 10
def tenths_place_value : ℚ := 0.1

theorem place_value_comparison : 
  tens_place_value / tenths_place_value = 100 := by sorry

end place_value_comparison_l547_54749


namespace pq_length_is_25_over_3_l547_54752

/-- Triangle DEF with given side lengths and a parallel segment PQ on DE -/
structure TriangleWithParallelSegment where
  /-- Length of side DE -/
  de : ℝ
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FD -/
  fd : ℝ
  /-- Length of segment PQ -/
  pq : ℝ
  /-- PQ is parallel to EF -/
  pq_parallel_ef : Bool
  /-- PQ is on DE -/
  pq_on_de : Bool
  /-- PQ is one-third of DE -/
  pq_is_third_of_de : pq = de / 3

/-- The length of PQ in the given triangle configuration is 25/3 -/
theorem pq_length_is_25_over_3 (t : TriangleWithParallelSegment)
  (h_de : t.de = 25)
  (h_ef : t.ef = 29)
  (h_fd : t.fd = 32)
  (h_pq_parallel : t.pq_parallel_ef = true)
  (h_pq_on_de : t.pq_on_de = true) :
  t.pq = 25 / 3 := by
  sorry

end pq_length_is_25_over_3_l547_54752


namespace ternary_decimal_conversion_decimal_base7_conversion_l547_54737

-- Define a function to convert from base 3 to base 10
def ternary_to_decimal (t : List Nat) : Nat :=
  t.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (t.length - 1 - i))) 0

-- Define a function to convert from base 10 to base 7
def decimal_to_base7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
  aux n []

theorem ternary_decimal_conversion :
  ternary_to_decimal [1, 0, 2, 1, 2] = 104 := by sorry

theorem decimal_base7_conversion :
  decimal_to_base7 1234 = [3, 4, 1, 2] := by sorry

end ternary_decimal_conversion_decimal_base7_conversion_l547_54737


namespace circle_center_l547_54786

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 8*x + y^2 + 4*y = 16

/-- The center of a circle given by its coordinates -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Theorem: The center of the circle with equation x^2 - 8x + y^2 + 4y = 16 is (4, -2) -/
theorem circle_center : 
  ∃ (c : CircleCenter), c.x = 4 ∧ c.y = -2 ∧ 
  ∀ (x y : ℝ), circle_equation x y ↔ (x - c.x)^2 + (y - c.y)^2 = 36 :=
sorry

end circle_center_l547_54786


namespace fiftieth_term_is_346_l547_54785

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem fiftieth_term_is_346 : 
  arithmetic_sequence 3 7 50 = 346 := by
sorry

end fiftieth_term_is_346_l547_54785


namespace selection_theorem_l547_54779

/-- The number of ways to select 4 students from 7 students (4 boys and 3 girls), 
    ensuring that the selection includes both boys and girls -/
def selection_ways : ℕ :=
  Nat.choose 7 4 - Nat.choose 4 4

theorem selection_theorem : selection_ways = 34 := by
  sorry

end selection_theorem_l547_54779


namespace square_mod_five_l547_54792

theorem square_mod_five (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end square_mod_five_l547_54792


namespace parabola_focus_l547_54708

/-- A parabola is defined by its equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The focus of a parabola is a point (h, k) -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -2x^2 + 5, its focus is (0, 9/2) -/
theorem parabola_focus (p : Parabola) (f : Focus) : 
  p.a = -2 ∧ p.b = 0 ∧ p.c = 5 → f.h = 0 ∧ f.k = 9/2 := by sorry

end parabola_focus_l547_54708


namespace candidate_score_approx_45_l547_54782

-- Define the maximum marks for Paper I
def max_marks : ℝ := 127.27

-- Define the passing percentage
def passing_percentage : ℝ := 0.55

-- Define the margin by which the candidate failed
def failing_margin : ℝ := 25

-- Define the candidate's score
def candidate_score : ℝ := max_marks * passing_percentage - failing_margin

-- Theorem to prove
theorem candidate_score_approx_45 : 
  ∃ ε > 0, abs (candidate_score - 45) < ε :=
sorry

end candidate_score_approx_45_l547_54782


namespace median_could_be_16_l547_54772

/-- Represents the age distribution of the school band --/
structure AgeDist where
  age13 : Nat
  age14 : Nat
  age15 : Nat
  age16 : Nat

/-- Calculates the total number of members in the band --/
def totalMembers (dist : AgeDist) : Nat :=
  dist.age13 + dist.age14 + dist.age15 + dist.age16

/-- Checks if a given age is the median of the distribution --/
def isMedian (dist : AgeDist) (age : Nat) : Prop :=
  let total := totalMembers dist
  let halfTotal := total / 2
  let countBelow := 
    if age == 13 then 0
    else if age == 14 then dist.age13
    else if age == 15 then dist.age13 + dist.age14
    else dist.age13 + dist.age14 + dist.age15
  countBelow < halfTotal ∧ countBelow + (if age == 16 then dist.age16 else 0) ≥ halfTotal

/-- The main theorem stating that 16 could be the median --/
theorem median_could_be_16 (dist : AgeDist) : 
  dist.age13 = 5 → dist.age14 = 7 → dist.age15 = 13 → ∃ n : Nat, isMedian { age13 := 5, age14 := 7, age15 := 13, age16 := n } 16 :=
sorry

end median_could_be_16_l547_54772


namespace air_inhaled_24_hours_l547_54768

/-- The volume of air inhaled in 24 hours given the breathing rate and volume per breath -/
theorem air_inhaled_24_hours 
  (breaths_per_minute : ℕ) 
  (air_per_breath : ℚ) 
  (h1 : breaths_per_minute = 17) 
  (h2 : air_per_breath = 5/9) : 
  (breaths_per_minute : ℚ) * air_per_breath * (24 * 60) = 13600 := by
  sorry

end air_inhaled_24_hours_l547_54768


namespace parking_garage_has_four_stories_l547_54769

/-- Represents a parking garage with the given specifications -/
structure ParkingGarage where
  spots_per_level : ℕ
  open_spots_level1 : ℕ
  open_spots_level2 : ℕ
  open_spots_level3 : ℕ
  open_spots_level4 : ℕ
  full_spots_total : ℕ

/-- Calculates the number of stories in the parking garage -/
def number_of_stories (garage : ParkingGarage) : ℕ :=
  (garage.full_spots_total + garage.open_spots_level1 + garage.open_spots_level2 +
   garage.open_spots_level3 + garage.open_spots_level4) / garage.spots_per_level

/-- Theorem stating that the parking garage has exactly 4 stories -/
theorem parking_garage_has_four_stories (garage : ParkingGarage) :
  garage.spots_per_level = 100 ∧
  garage.open_spots_level1 = 58 ∧
  garage.open_spots_level2 = garage.open_spots_level1 + 2 ∧
  garage.open_spots_level3 = garage.open_spots_level2 + 5 ∧
  garage.open_spots_level4 = 31 ∧
  garage.full_spots_total = 186 →
  number_of_stories garage = 4 := by
  sorry

end parking_garage_has_four_stories_l547_54769


namespace rectangle_perimeter_increase_l547_54748

theorem rectangle_perimeter_increase (l w : ℝ) (hl : l > 0) (hw : w > 0) :
  let initial_perimeter := 2 * (l + w)
  let new_perimeter := 2 * (1.1 * l + 1.1 * w)
  new_perimeter / initial_perimeter = 1.1 := by
sorry

end rectangle_perimeter_increase_l547_54748


namespace total_results_l547_54722

theorem total_results (avg_all : ℝ) (avg_first_six : ℝ) (avg_last_six : ℝ) (sixth_result : ℝ) : 
  avg_all = 52 → 
  avg_first_six = 49 → 
  avg_last_six = 52 → 
  sixth_result = 34 → 
  ∃ n : ℕ, n = 11 ∧ n * avg_all = (6 * avg_first_six + 6 * avg_last_six - sixth_result) :=
by
  sorry

#check total_results

end total_results_l547_54722


namespace petya_win_probability_l547_54724

/-- The "Heap of Stones" game -/
structure HeapOfStones where
  initialStones : Nat
  minTake : Nat
  maxTake : Nat

/-- A player in the game -/
inductive Player
  | Petya
  | Computer

/-- The state of the game -/
structure GameState where
  stones : Nat
  currentPlayer : Player

/-- The outcome of the game -/
inductive GameOutcome
  | PetyaWins
  | ComputerWins

/-- A strategy for playing the game -/
def Strategy := GameState → Nat

/-- The random strategy that Petya uses -/
def randomStrategy : Strategy := sorry

/-- The optimal strategy that the computer uses -/
def optimalStrategy : Strategy := sorry

/-- Play the game with given strategies -/
def playGame (petyaStrategy : Strategy) (computerStrategy : Strategy) : GameOutcome := sorry

/-- The probability of Petya winning -/
def petyaWinProbability : ℚ := sorry

/-- Main theorem: The probability of Petya winning is 1/256 -/
theorem petya_win_probability :
  let game : HeapOfStones := ⟨16, 1, 4⟩
  petyaWinProbability = 1 / 256 := by sorry

end petya_win_probability_l547_54724


namespace intersection_A_complement_B_l547_54762

-- Define the sets A and B
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {y | y ≥ 1}

-- State the theorem
theorem intersection_A_complement_B : A ∩ Bᶜ = Set.Ioo 0 1 := by sorry

end intersection_A_complement_B_l547_54762


namespace ten_trees_road_length_l547_54794

/-- The length of a road with trees planted at equal intervals --/
def road_length (num_trees : ℕ) (interval : ℕ) : ℕ :=
  (num_trees - 1) * interval

/-- Theorem: The length of a road with 10 trees planted at 10-meter intervals is 90 meters --/
theorem ten_trees_road_length :
  road_length 10 10 = 90 := by
  sorry

end ten_trees_road_length_l547_54794


namespace inverse_of_A_l547_54705

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem inverse_of_A :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by
  sorry

end inverse_of_A_l547_54705


namespace trains_meeting_point_l547_54744

/-- The speed of the Bombay Express in km/h -/
def bombay_speed : ℝ := 60

/-- The speed of the Rajdhani Express in km/h -/
def rajdhani_speed : ℝ := 80

/-- The time difference between the departures of the two trains in hours -/
def time_difference : ℝ := 2

/-- The meeting point of the two trains -/
def meeting_point : ℝ := 480

theorem trains_meeting_point :
  ∃ t : ℝ, t > 0 ∧ bombay_speed * (t + time_difference) = rajdhani_speed * t ∧
  rajdhani_speed * t = meeting_point := by sorry

end trains_meeting_point_l547_54744
