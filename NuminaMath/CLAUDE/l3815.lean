import Mathlib

namespace second_discount_percentage_l3815_381536

theorem second_discount_percentage (initial_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  initial_price = 600 →
  first_discount = 10 →
  final_price = 513 →
  ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 5 := by
  sorry

end second_discount_percentage_l3815_381536


namespace a4_range_l3815_381591

theorem a4_range (a₁ a₂ a₃ a₄ : ℝ) 
  (sum_zero : a₁ + a₂ + a₃ = 0)
  (quad_eq : a₁ * a₄^2 + a₂ * a₄ - a₂ = 0)
  (order : a₁ > a₂ ∧ a₂ > a₃) :
  -1/2 - Real.sqrt 5/2 < a₄ ∧ a₄ < -1/2 + Real.sqrt 5/2 := by
sorry

end a4_range_l3815_381591


namespace age_difference_constant_l3815_381599

theorem age_difference_constant (seokjin_initial_age mother_initial_age years_passed : ℕ) :
  mother_initial_age - seokjin_initial_age = 
  (mother_initial_age + years_passed) - (seokjin_initial_age + years_passed) :=
by sorry

end age_difference_constant_l3815_381599


namespace haley_tv_watching_time_l3815_381554

theorem haley_tv_watching_time (saturday_hours sunday_hours : ℕ) 
  (h1 : saturday_hours = 6) 
  (h2 : sunday_hours = 3) : 
  saturday_hours + sunday_hours = 9 := by
sorry

end haley_tv_watching_time_l3815_381554


namespace similar_triangles_height_l3815_381573

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = 15 ∧ h_large / h_small = Real.sqrt area_ratio :=
sorry

end similar_triangles_height_l3815_381573


namespace complex_power_four_l3815_381546

theorem complex_power_four : (1 + 2 * Complex.I) ^ 4 = -7 - 24 * Complex.I := by
  sorry

end complex_power_four_l3815_381546


namespace range_of_f_l3815_381548

def f (x : ℝ) : ℝ := |x - 3| - |x + 4|

theorem range_of_f :
  ∀ y ∈ Set.range f, -7 ≤ y ∧ y ≤ 7 ∧
  ∀ z, -7 ≤ z ∧ z ≤ 7 → ∃ x, f x = z :=
sorry

end range_of_f_l3815_381548


namespace parallel_vectors_m_value_l3815_381529

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -3)
  let b : ℝ × ℝ := (m, 6)
  parallel a b → m = -2 := by
  sorry

end parallel_vectors_m_value_l3815_381529


namespace mass_CaSO4_formed_l3815_381567

-- Define the molar masses
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.06
def molar_mass_O : ℝ := 16.00

-- Define the molar mass of CaSO₄
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Define the number of moles of Ca(OH)₂
def moles_CaOH2 : ℝ := 12

-- Theorem statement
theorem mass_CaSO4_formed (excess_H2SO4 : Prop) (neutralization_reaction : Prop) :
  moles_CaOH2 * molar_mass_CaSO4 = 1633.68 := by
  sorry


end mass_CaSO4_formed_l3815_381567


namespace polynomial_remainder_l3815_381504

theorem polynomial_remainder (x : ℂ) : 
  x^2 - x + 1 = 0 → (2*x^5 - x^4 + x^2 - 1)*(x^3 - 1) = 0 := by
  sorry

end polynomial_remainder_l3815_381504


namespace unique_solution_l3815_381510

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  x^3 - 3*x = 4 - y ∧
  2*y^3 - 6*y = 6 - z ∧
  3*z^3 - 9*z = 8 - x

-- Theorem statement
theorem unique_solution :
  ∀ x y z : ℝ, system x y z → (x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

end unique_solution_l3815_381510


namespace percent_less_u_than_y_l3815_381583

theorem percent_less_u_than_y 
  (w u y z : ℝ) 
  (hw : w = 0.60 * u) 
  (hz1 : z = 0.54 * y) 
  (hz2 : z = 1.50 * w) : 
  u = 0.60 * y := by sorry

end percent_less_u_than_y_l3815_381583


namespace adrianna_gum_purchase_l3815_381513

/-- Calculates the number of gum pieces bought at the store -/
def gum_bought_at_store (initial_gum : ℕ) (friends_given : ℕ) (gum_left : ℕ) : ℕ :=
  friends_given + gum_left - initial_gum

/-- Theorem: Given the initial conditions, prove that Adrianna bought 3 pieces of gum at the store -/
theorem adrianna_gum_purchase :
  gum_bought_at_store 10 11 2 = 3 := by
  sorry

end adrianna_gum_purchase_l3815_381513


namespace prob_12th_last_value_l3815_381537

/-- Probability of getting a different roll on a four-sided die -/
def p_different : ℚ := 3 / 4

/-- Probability of getting the same roll on a four-sided die -/
def p_same : ℚ := 1 / 4

/-- Number of rolls before the final roll -/
def n : ℕ := 11

/-- Probability of the 12th roll being the last roll -/
def prob_12th_last : ℚ := p_different ^ n * p_same

theorem prob_12th_last_value : 
  prob_12th_last = (3 ^ 10 : ℚ) / (4 ^ 11 : ℚ) := by sorry

end prob_12th_last_value_l3815_381537


namespace find_x_l3815_381594

theorem find_x : ∃ x : ℕ, 
  (∃ k : ℕ, x = 8 * k) ∧ 
  x^2 > 100 ∧ 
  x < 20 ∧ 
  x = 16 := by
  sorry

end find_x_l3815_381594


namespace function_satisfying_inequality_is_constant_two_l3815_381557

/-- A function satisfying the given inequality for all real x and y is constant and equal to 2. -/
theorem function_satisfying_inequality_is_constant_two 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, 2 * f x + 2 * f y - f x * f y ≥ 4) : 
  ∀ x : ℝ, f x = 2 := by
  sorry

end function_satisfying_inequality_is_constant_two_l3815_381557


namespace second_athlete_high_jump_l3815_381582

def athlete1_long_jump : ℝ := 26
def athlete1_triple_jump : ℝ := 30
def athlete1_high_jump : ℝ := 7

def athlete2_long_jump : ℝ := 24
def athlete2_triple_jump : ℝ := 34

def winner_average_jump : ℝ := 22

def number_of_jumps : ℕ := 3

theorem second_athlete_high_jump :
  let athlete1_total := athlete1_long_jump + athlete1_triple_jump + athlete1_high_jump
  let athlete1_average := athlete1_total / number_of_jumps
  let athlete2_total_before_high := athlete2_long_jump + athlete2_triple_jump
  let winner_total := winner_average_jump * number_of_jumps
  athlete1_average < winner_average_jump →
  winner_total - athlete2_total_before_high = 8 := by
sorry

end second_athlete_high_jump_l3815_381582


namespace triangle_third_side_length_l3815_381580

theorem triangle_third_side_length 
  (a b : ℝ) 
  (θ : ℝ) 
  (ha : a = 10) 
  (hb : b = 15) 
  (hθ : θ = Real.pi / 3) : 
  Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos θ)) = 5 * Real.sqrt 7 := by
sorry

end triangle_third_side_length_l3815_381580


namespace joan_seashells_left_l3815_381592

/-- The number of seashells Joan has left after giving some to Sam -/
def seashells_left (initial : ℕ) (given : ℕ) : ℕ :=
  initial - given

/-- Theorem stating that Joan has 27 seashells left -/
theorem joan_seashells_left : seashells_left 70 43 = 27 := by
  sorry

end joan_seashells_left_l3815_381592


namespace decaf_percentage_l3815_381578

/-- Calculates the percentage of decaffeinated coffee in the total stock -/
theorem decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_stock : ℝ)
  (additional_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 30)
  (h3 : additional_stock = 100)
  (h4 : additional_decaf_percent = 60) :
  let total_stock := initial_stock + additional_stock
  let total_decaf := initial_stock * (initial_decaf_percent / 100) +
                     additional_stock * (additional_decaf_percent / 100)
  total_decaf / total_stock * 100 = 36 := by
sorry


end decaf_percentage_l3815_381578


namespace smallest_a_value_l3815_381547

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ (x : ℤ), Real.sin (a * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) :
  a ≥ 17 ∧ ∃ (a₀ : ℝ), a₀ ≥ 0 ∧ a₀ < 17 ∧ 
    (∀ (x : ℤ), Real.sin (a₀ * (x : ℝ) + b) = Real.sin (17 * (x : ℝ))) → False :=
by sorry

end smallest_a_value_l3815_381547


namespace sum_of_a_and_c_l3815_381572

theorem sum_of_a_and_c (a b c d : ℝ) 
  (h1 : a * b + b * c + c * d + d * a = 40) 
  (h2 : b + d = 8) : 
  a + c = 5 := by
sorry

end sum_of_a_and_c_l3815_381572


namespace absolute_value_greater_than_two_l3815_381576

theorem absolute_value_greater_than_two (x : ℝ) : |x| > 2 ↔ x > 2 ∨ x < -2 := by
  sorry

end absolute_value_greater_than_two_l3815_381576


namespace base5_division_l3815_381511

/-- Converts a base 5 number to base 10 -/
def toBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base 10 number to base 5 -/
def toBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division (dividend : List Nat) (divisor : List Nat) :
  dividend = [2, 0, 1, 3] ∧ divisor = [3, 2] →
  toBase5 (toBase10 dividend / toBase10 divisor) = [0, 1, 1] := by
  sorry

end base5_division_l3815_381511


namespace environmental_law_support_l3815_381593

theorem environmental_law_support (men : ℕ) (women : ℕ) 
  (men_support_percent : ℚ) (women_support_percent : ℚ) 
  (h1 : men = 200) 
  (h2 : women = 800) 
  (h3 : men_support_percent = 75 / 100) 
  (h4 : women_support_percent = 65 / 100) : 
  (men_support_percent * men + women_support_percent * women) / (men + women) = 67 / 100 := by
  sorry

end environmental_law_support_l3815_381593


namespace line_intersects_circle_l3815_381575

-- Define the line equation
def line_equation (m : ℝ) (x : ℝ) : ℝ := m * x - 3

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 25

-- Theorem stating that any real slope m results in an intersection
theorem line_intersects_circle (m : ℝ) :
  ∃ x : ℝ, circle_equation x (line_equation m x) := by
  sorry

end line_intersects_circle_l3815_381575


namespace remainder_of_sum_divided_by_eight_l3815_381571

theorem remainder_of_sum_divided_by_eight :
  (2356789 + 211) % 8 = 0 := by
sorry

end remainder_of_sum_divided_by_eight_l3815_381571


namespace root_sum_reciprocal_plus_one_l3815_381584

theorem root_sum_reciprocal_plus_one (a b c : ℂ) : 
  (a^3 - a - 2 = 0) → (b^3 - b - 2 = 0) → (c^3 - c - 2 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 2) :=
by sorry

end root_sum_reciprocal_plus_one_l3815_381584


namespace largest_prime_factor_of_4851_l3815_381589

theorem largest_prime_factor_of_4851 : ∃ p : ℕ, Nat.Prime p ∧ p ∣ 4851 ∧ ∀ q : ℕ, Nat.Prime q → q ∣ 4851 → q ≤ p :=
by sorry

end largest_prime_factor_of_4851_l3815_381589


namespace median_is_39_l3815_381531

/-- Represents the score distribution of students --/
structure ScoreDistribution where
  scores : List Nat
  counts : List Nat
  total_students : Nat

/-- Calculates the median of a score distribution --/
def median (sd : ScoreDistribution) : Rat :=
  sorry

/-- The specific score distribution from the problem --/
def problem_distribution : ScoreDistribution :=
  { scores := [36, 37, 38, 39, 40],
    counts := [1, 2, 1, 4, 2],
    total_students := 10 }

/-- Theorem stating that the median of the given distribution is 39 --/
theorem median_is_39 : median problem_distribution = 39 := by
  sorry

end median_is_39_l3815_381531


namespace external_angle_c_l3815_381540

theorem external_angle_c (A B C : ℝ) : 
  A = 40 → B = 2 * A → A + B + C = 180 → 180 - C = 120 := by sorry

end external_angle_c_l3815_381540


namespace staircase_perimeter_l3815_381595

/-- A staircase-shaped region with right angles -/
structure StaircaseRegion where
  /-- The number of 1-foot sides in the staircase -/
  num_sides : ℕ
  /-- The area of the region in square feet -/
  area : ℝ
  /-- Assumption that the number of sides is 10 -/
  sides_eq_ten : num_sides = 10
  /-- Assumption that the area is 85 square feet -/
  area_eq_85 : area = 85

/-- Calculate the perimeter of a staircase region -/
def perimeter (r : StaircaseRegion) : ℝ := sorry

/-- Theorem stating that the perimeter of the given staircase region is 30.5 feet -/
theorem staircase_perimeter (r : StaircaseRegion) : perimeter r = 30.5 := by sorry

end staircase_perimeter_l3815_381595


namespace min_m_plus_n_l3815_381545

theorem min_m_plus_n (m n : ℕ+) (h : 108 * m = n ^ 3) : 
  ∀ (m' n' : ℕ+), 108 * m' = n' ^ 3 → m + n ≤ m' + n' := by
  sorry

end min_m_plus_n_l3815_381545


namespace problem_statement_l3815_381515

noncomputable def f₁ (a x : ℝ) : ℝ := Real.exp (abs (x - 2*a + 1))
noncomputable def f₂ (a x : ℝ) : ℝ := Real.exp (abs (x - a) + 1)
noncomputable def f (a x : ℝ) : ℝ := f₁ a x + f₂ a x
noncomputable def g (a x : ℝ) : ℝ := (f₁ a x + f₂ a x) / 2 - abs (f₁ a x - f₂ a x) / 2

theorem problem_statement :
  (∀ x ∈ Set.Icc 2 3, f 2 x ≥ 2 * Real.exp 1) ∧
  (∃ x ∈ Set.Icc 2 3, f 2 x = 2 * Real.exp 1) ∧
  (∀ a, (∀ x ≥ a, f₂ a x ≥ f₁ a x) ↔ 0 ≤ a ∧ a ≤ 2) ∧
  (∀ x ∈ Set.Icc 1 6, 
    g a x ≥ 
      (if 1 ≤ a ∧ a ≤ 7/2 then 1
      else if -2 ≤ a ∧ a ≤ 0 then Real.exp (2 - a)
      else if a < -2 ∨ (0 < a ∧ a < 1) then Real.exp (3 - 2*a)
      else if 7/2 < a ∧ a ≤ 6 then Real.exp 1
      else Real.exp (a - 5))) ∧
  (∃ x ∈ Set.Icc 1 6, 
    g a x = 
      (if 1 ≤ a ∧ a ≤ 7/2 then 1
      else if -2 ≤ a ∧ a ≤ 0 then Real.exp (2 - a)
      else if a < -2 ∨ (0 < a ∧ a < 1) then Real.exp (3 - 2*a)
      else if 7/2 < a ∧ a ≤ 6 then Real.exp 1
      else Real.exp (a - 5))) := by sorry

end problem_statement_l3815_381515


namespace mary_max_earnings_l3815_381534

/-- Calculates the maximum weekly earnings for a worker with the given conditions -/
def max_weekly_earnings (max_hours : ℕ) (regular_hours : ℕ) (regular_rate : ℚ) (overtime_rate_increase : ℚ) : ℚ :=
  let overtime_hours := max_hours - regular_hours
  let overtime_rate := regular_rate * (1 + overtime_rate_increase)
  regular_hours * regular_rate + overtime_hours * overtime_rate

/-- Mary's maximum weekly earnings under the given conditions -/
theorem mary_max_earnings :
  max_weekly_earnings 60 20 8 (1/4) = 560 := by
  sorry

#eval max_weekly_earnings 60 20 8 (1/4)

end mary_max_earnings_l3815_381534


namespace square_difference_fraction_l3815_381526

theorem square_difference_fraction (x y : ℚ) 
  (sum_eq : x + y = 8/15) 
  (diff_eq : x - y = 1/35) : 
  x^2 - y^2 = 1/75 := by
sorry

end square_difference_fraction_l3815_381526


namespace absolute_value_equation_solution_l3815_381564

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 5| = 3 * x - 2 :=
by
  -- The unique solution is x = 7/4
  use 7/4
  sorry

end absolute_value_equation_solution_l3815_381564


namespace complete_square_for_given_equation_l3815_381533

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the result of completing the square for a quadratic equation -/
structure CompletedSquareForm where
  r : ℝ
  s : ℝ

/-- Completes the square for a given quadratic equation -/
def completeSquare (eq : QuadraticEquation) : CompletedSquareForm :=
  sorry

theorem complete_square_for_given_equation :
  let eq := QuadraticEquation.mk 9 (-18) (-720)
  let result := completeSquare eq
  result.s = 81 := by sorry

end complete_square_for_given_equation_l3815_381533


namespace amount_after_two_years_l3815_381577

theorem amount_after_two_years 
  (present_value : ℝ) 
  (yearly_increase_rate : ℝ) 
  (h1 : present_value = 57600) 
  (h2 : yearly_increase_rate = 1/8) 
  (h3 : (present_value * (1 + yearly_increase_rate)^2 : ℝ) = 72900) : 
  (present_value * (1 + yearly_increase_rate)^2 : ℝ) = 72900 := by
  sorry

end amount_after_two_years_l3815_381577


namespace coefficient_of_x_squared_l3815_381552

def p (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 3 * x
def q (x : ℝ) : ℝ := 3 * x^2 - 4 * x - 5

theorem coefficient_of_x_squared :
  ∃ (a b c d e : ℝ), p x * q x = a * x^5 + b * x^4 + c * x^3 - 37 * x^2 + d * x + e :=
by sorry

end coefficient_of_x_squared_l3815_381552


namespace binomSum_not_div_five_l3815_381549

def binomSum (n : ℕ) : ℕ :=
  Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose (2 * n + 1) (2 * k + 1) * 2^(3 * k))

theorem binomSum_not_div_five (n : ℕ) : ¬(5 ∣ binomSum n) := by
  sorry

end binomSum_not_div_five_l3815_381549


namespace real_solutions_condition_l3815_381516

theorem real_solutions_condition (a : ℝ) :
  (∃ x y : ℝ, x + y^2 = a ∧ y + x^2 = a) ↔ a ≥ 3/4 := by
  sorry

end real_solutions_condition_l3815_381516


namespace base6_sum_is_6_l3815_381543

/-- Represents a single digit in base 6 -/
def Base6Digit := Fin 6

/-- The addition problem in base 6 -/
def base6_addition (X Y : Base6Digit) : Prop :=
  ∃ (carry : Nat),
    (3 * 6^2 + X.val * 6 + Y.val) + 24 = 
    6 * 6^2 + carry * 6 + X.val

/-- The main theorem to prove -/
theorem base6_sum_is_6 :
  ∀ X Y : Base6Digit,
    base6_addition X Y →
    (X.val : ℕ) + (Y.val : ℕ) = 6 := by sorry

end base6_sum_is_6_l3815_381543


namespace deck_size_l3815_381505

theorem deck_size (r b : ℕ) : 
  r ≠ 0 → 
  b ≠ 0 → 
  r / (r + b) = 1 / 4 → 
  r / (r + b + 6) = 1 / 6 → 
  r + b = 12 := by
sorry

end deck_size_l3815_381505


namespace custom_operation_calculation_l3815_381587

-- Define the custom operation *
def star (a b : ℕ) : ℕ := a + 2 * b

-- Theorem statement
theorem custom_operation_calculation :
  star (star (star 2 3) 4) 5 = 26 := by
  sorry

end custom_operation_calculation_l3815_381587


namespace sufficient_not_necessary_condition_l3815_381551

theorem sufficient_not_necessary_condition (a b c : ℝ) :
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2) :=
sorry

end sufficient_not_necessary_condition_l3815_381551


namespace triangle_third_side_length_l3815_381568

theorem triangle_third_side_length 
  (a b c : ℝ) 
  (angle_cos : ℝ) 
  (h1 : a = 4) 
  (h2 : b = 5) 
  (h3 : 2 * angle_cos^2 + 3 * angle_cos - 2 = 0) 
  (h4 : c^2 = a^2 + b^2 - 2*a*b*angle_cos) : 
  c = Real.sqrt 21 := by
sorry

end triangle_third_side_length_l3815_381568


namespace g_function_equality_l3815_381581

/-- Given that 4x^4 + 5x^2 - 2x + 7 + g(x) = 6x^3 - 4x^2 + 8x - 1,
    prove that g(x) = -4x^4 + 6x^3 - 9x^2 + 10x - 8 -/
theorem g_function_equality (x : ℝ) (g : ℝ → ℝ)
    (h : ∀ x, 4 * x^4 + 5 * x^2 - 2 * x + 7 + g x = 6 * x^3 - 4 * x^2 + 8 * x - 1) :
  g x = -4 * x^4 + 6 * x^3 - 9 * x^2 + 10 * x - 8 := by
  sorry

end g_function_equality_l3815_381581


namespace train_length_l3815_381514

/-- Given a train that crosses a tree in 100 seconds and takes 150 seconds to pass a platform 700 m long, prove that the length of the train is 1400 meters. -/
theorem train_length (tree_crossing_time platform_crossing_time platform_length : ℝ) 
  (h1 : tree_crossing_time = 100)
  (h2 : platform_crossing_time = 150)
  (h3 : platform_length = 700) : 
  ∃ train_length : ℝ, train_length = 1400 ∧ 
    train_length / tree_crossing_time = (train_length + platform_length) / platform_crossing_time :=
by
  sorry


end train_length_l3815_381514


namespace log_difference_negative_l3815_381558

theorem log_difference_negative (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  Real.log (b - a) < 0 := by
  sorry

end log_difference_negative_l3815_381558


namespace protege_zero_implies_two_and_five_l3815_381517

/-- A digit is a protégé of a natural number if it is the units digit of some divisor of that number. -/
def isProtege (d : Nat) (n : Nat) : Prop :=
  ∃ k : Nat, k ∣ n ∧ k % 10 = d

/-- Theorem: If 0 is a protégé of a natural number, then 2 and 5 are also protégés of that number. -/
theorem protege_zero_implies_two_and_five (n : Nat) :
  isProtege 0 n → isProtege 2 n ∧ isProtege 5 n := by
  sorry


end protege_zero_implies_two_and_five_l3815_381517


namespace min_pieces_for_special_l3815_381544

/-- Represents a piece of the pie -/
inductive PieceType
| Empty
| Fish
| Sausage
| Special

/-- Represents the 8x8 pie grid -/
def Pie := Fin 8 → Fin 8 → PieceType

/-- Checks if a 6x6 square in the pie has at least 2 fish pieces -/
def has_two_fish (p : Pie) (i j : Fin 8) : Prop :=
  ∃ (i1 j1 i2 j2 : Fin 8),
    i1 < i + 6 ∧ j1 < j + 6 ∧ i2 < i + 6 ∧ j2 < j + 6 ∧
    (i1 ≠ i2 ∨ j1 ≠ j2) ∧
    p i1 j1 = PieceType.Fish ∧ p i2 j2 = PieceType.Fish

/-- Checks if a 3x3 square in the pie has at most 1 sausage piece -/
def at_most_one_sausage (p : Pie) (i j : Fin 8) : Prop :=
  ∀ (i1 j1 i2 j2 : Fin 8),
    i1 < i + 3 → j1 < j + 3 → i2 < i + 3 → j2 < j + 3 →
    p i1 j1 = PieceType.Sausage → p i2 j2 = PieceType.Sausage →
    i1 = i2 ∧ j1 = j2

/-- Defines a valid pie configuration -/
def valid_pie (p : Pie) : Prop :=
  (∃ (i1 j1 i2 j2 i3 j3 : Fin 8),
     p i1 j1 = PieceType.Fish ∧ p i2 j2 = PieceType.Fish ∧ p i3 j3 = PieceType.Fish ∧
     (i1 ≠ i2 ∨ j1 ≠ j2) ∧ (i1 ≠ i3 ∨ j1 ≠ j3) ∧ (i2 ≠ i3 ∨ j2 ≠ j3)) ∧
  (∃ (i1 j1 i2 j2 : Fin 8),
     p i1 j1 = PieceType.Sausage ∧ p i2 j2 = PieceType.Sausage ∧
     (i1 ≠ i2 ∨ j1 ≠ j2)) ∧
  (∃! (i j : Fin 8), p i j = PieceType.Special) ∧
  (∀ (i j : Fin 8), has_two_fish p i j) ∧
  (∀ (i j : Fin 8), at_most_one_sausage p i j)

/-- Theorem: The minimum number of pieces to guarantee getting the special piece is 5 -/
theorem min_pieces_for_special (p : Pie) (h : valid_pie p) :
  ∀ (s : Finset (Fin 8 × Fin 8)),
    s.card < 5 → ∃ (i j : Fin 8), p i j = PieceType.Special ∧ (i, j) ∉ s :=
sorry

end min_pieces_for_special_l3815_381544


namespace optimal_price_l3815_381586

/-- Represents the daily sales profit function for an agricultural product. -/
def W (x : ℝ) : ℝ := -2 * x^2 + 120 * x - 1600

/-- Represents the daily sales quantity function for an agricultural product. -/
def y (x : ℝ) : ℝ := -2 * x + 80

/-- The cost price per kilogram of the agricultural product. -/
def cost_price : ℝ := 20

/-- The maximum allowed selling price per kilogram. -/
def max_price : ℝ := 30

/-- The desired daily sales profit. -/
def target_profit : ℝ := 150

/-- Theorem stating that a selling price of 25 achieves the target profit
    while satisfying the given conditions. -/
theorem optimal_price :
  W 25 = target_profit ∧
  25 ≤ max_price ∧
  y 25 > 0 :=
sorry

end optimal_price_l3815_381586


namespace cricket_run_rate_theorem_l3815_381565

/-- Represents a cricket game situation -/
structure CricketGame where
  totalOvers : ℕ
  firstPeriodOvers : ℕ
  firstPeriodRunRate : ℚ
  targetRuns : ℕ

/-- Calculates the required run rate for the remaining overs -/
def requiredRunRate (game : CricketGame) : ℚ :=
  let remainingOvers := game.totalOvers - game.firstPeriodOvers
  let runsScored := game.firstPeriodRunRate * game.firstPeriodOvers
  let runsNeeded := game.targetRuns - runsScored
  runsNeeded / remainingOvers

/-- Theorem stating the required run rate for the given game situation -/
theorem cricket_run_rate_theorem (game : CricketGame) 
  (h1 : game.totalOvers = 50)
  (h2 : game.firstPeriodOvers = 10)
  (h3 : game.firstPeriodRunRate = 21/5)  -- 4.2 as a fraction
  (h4 : game.targetRuns = 282) :
  requiredRunRate game = 6 := by
  sorry

end cricket_run_rate_theorem_l3815_381565


namespace function_identity_l3815_381527

theorem function_identity (f : ℝ → ℝ) :
  (∀ x, f x + 2 * f (3 - x) = x^2) →
  (∀ x, f x = (1/3) * x^2 - 4 * x + 6) :=
by sorry

end function_identity_l3815_381527


namespace percentage_of_defective_meters_l3815_381596

theorem percentage_of_defective_meters 
  (total_meters : ℕ) 
  (rejected_meters : ℕ) 
  (h1 : total_meters = 200) 
  (h2 : rejected_meters = 20) : 
  (rejected_meters : ℝ) / (total_meters : ℝ) * 100 = 10 := by
  sorry

end percentage_of_defective_meters_l3815_381596


namespace inequality_solution_l3815_381502

theorem inequality_solution (x : ℝ) :
  x ≠ 4 →
  (x * (x + 1) / (x - 4)^2 ≥ 15 ↔ x ∈ Set.Iic 3 ∪ Set.Ioo (40/7) 4 ∪ Set.Ioi 4) :=
by sorry

end inequality_solution_l3815_381502


namespace blue_jellybean_probability_blue_jellybean_probability_is_two_nineteenths_l3815_381535

/-- The probability of drawing 3 blue jellybeans in a row from a bag containing 10 red and 10 blue jellybeans, without replacement. -/
theorem blue_jellybean_probability : ℚ :=
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3

  let prob_first : ℚ := blue_jellybeans / total_jellybeans
  let prob_second : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1)
  let prob_third : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2)

  prob_first * prob_second * prob_third

/-- Proof that the probability of drawing 3 blue jellybeans in a row is 2/19. -/
theorem blue_jellybean_probability_is_two_nineteenths :
  blue_jellybean_probability = 2 / 19 := by
  sorry

end blue_jellybean_probability_blue_jellybean_probability_is_two_nineteenths_l3815_381535


namespace inequality_of_positive_reals_l3815_381561

theorem inequality_of_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^x * y^y * z^z ≥ (x*y*z)^((x+y+z)/3) := by
  sorry

end inequality_of_positive_reals_l3815_381561


namespace smallest_n_multiple_of_3_and_3n_multiple_of_5_l3815_381525

theorem smallest_n_multiple_of_3_and_3n_multiple_of_5 :
  ∃ n : ℕ, n > 0 ∧ 3 ∣ n ∧ 5 ∣ (3 * n) ∧
  ∀ m : ℕ, m > 0 → 3 ∣ m → 5 ∣ (3 * m) → n ≤ m :=
by
  -- The proof goes here
  sorry

#check smallest_n_multiple_of_3_and_3n_multiple_of_5

end smallest_n_multiple_of_3_and_3n_multiple_of_5_l3815_381525


namespace place_two_after_three_digit_number_l3815_381597

/-- Given a three-digit number with hundreds digit a, tens digit b, and units digit c,
    prove that placing the digit 2 after this number results in 1000a + 100b + 10c + 2 -/
theorem place_two_after_three_digit_number (a b c : ℕ) :
  let original := 100 * a + 10 * b + c
  10 * original + 2 = 1000 * a + 100 * b + 10 * c + 2 := by
  sorry

end place_two_after_three_digit_number_l3815_381597


namespace complex_equality_l3815_381563

theorem complex_equality (a b : ℝ) : (1 + Complex.I) + (2 - 3 * Complex.I) = a + b * Complex.I → a = 3 ∧ b = -2 := by
  sorry

end complex_equality_l3815_381563


namespace smallest_max_sum_l3815_381570

theorem smallest_max_sum (p q r s t : ℕ+) (h : p + q + r + s + t = 2025) :
  let N := max (p + q) (max (q + r) (max (r + s) (s + t)))
  ∀ m : ℕ, (∃ p' q' r' s' t' : ℕ+, p' + q' + r' + s' + t' = 2025 ∧ 
    max (p' + q') (max (q' + r') (max (r' + s') (s' + t'))) < m) → m > 676 := by
  sorry

end smallest_max_sum_l3815_381570


namespace product_sum_inequality_l3815_381588

theorem product_sum_inequality (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : a₁ < a₂) (h2 : b₁ < b₂) : 
  a₁ * b₁ + a₂ * b₂ > a₁ * b₂ + a₂ * b₁ := by
  sorry

end product_sum_inequality_l3815_381588


namespace clients_using_all_three_l3815_381585

def total_clients : ℕ := 180
def tv_clients : ℕ := 115
def radio_clients : ℕ := 110
def magazine_clients : ℕ := 130
def tv_and_magazine : ℕ := 85
def tv_and_radio : ℕ := 75
def radio_and_magazine : ℕ := 95

theorem clients_using_all_three :
  tv_clients + radio_clients + magazine_clients -
  tv_and_magazine - tv_and_radio - radio_and_magazine +
  (total_clients - (tv_clients + radio_clients + magazine_clients -
  tv_and_magazine - tv_and_radio - radio_and_magazine)) = 80 :=
by sorry

end clients_using_all_three_l3815_381585


namespace ribbon_distribution_l3815_381507

/-- Given total ribbon, number of gifts, and leftover ribbon, calculate ribbon per gift --/
def ribbon_per_gift (total_ribbon : ℕ) (num_gifts : ℕ) (leftover : ℕ) : ℚ :=
  (total_ribbon - leftover : ℚ) / num_gifts

theorem ribbon_distribution (total_ribbon num_gifts leftover : ℕ) 
  (h1 : total_ribbon = 18)
  (h2 : num_gifts = 6)
  (h3 : leftover = 6)
  (h4 : num_gifts > 0) :
  ribbon_per_gift total_ribbon num_gifts leftover = 2 := by
  sorry

end ribbon_distribution_l3815_381507


namespace equation_solution_l3815_381518

theorem equation_solution (x : ℝ) :
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
sorry

end equation_solution_l3815_381518


namespace nell_initial_cards_l3815_381590

/-- The number of baseball cards Nell initially had -/
def initial_cards : ℕ := sorry

/-- The number of cards Nell has at the end -/
def final_cards : ℕ := 154

/-- The number of cards Nell gave to Jeff -/
def cards_given_to_jeff : ℕ := 301

/-- The number of new cards Nell bought -/
def new_cards_bought : ℕ := 60

/-- The number of cards Nell traded away to Sam -/
def cards_traded_away : ℕ := 45

/-- The number of cards Nell received from Sam -/
def cards_received : ℕ := 30

/-- Theorem stating that Nell's initial number of baseball cards was 410 -/
theorem nell_initial_cards :
  initial_cards = 410 :=
by sorry

end nell_initial_cards_l3815_381590


namespace range_of_a_l3815_381542

theorem range_of_a (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3) 
  (square_condition : a^2 + 2*b^2 + 3*c^2 + 6*d^2 = 5) : 
  1 ≤ a ∧ a ≤ 2 := by sorry

end range_of_a_l3815_381542


namespace sallys_remaining_cards_l3815_381598

/-- Given Sally's initial number of baseball cards, the number of torn cards, 
    and the number of cards Sara bought, prove the number of cards Sally has now. -/
theorem sallys_remaining_cards (initial_cards torn_cards cards_bought : ℕ) :
  initial_cards = 39 →
  torn_cards = 9 →
  cards_bought = 24 →
  initial_cards - torn_cards - cards_bought = 6 := by
  sorry

end sallys_remaining_cards_l3815_381598


namespace bianca_points_l3815_381530

/-- Calculates the points earned for recycling cans given the total number of bags, 
    number of bags not recycled, and points per bag. -/
def points_earned (total_bags : ℕ) (bags_not_recycled : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - bags_not_recycled) * points_per_bag

/-- Proves that Bianca earned 45 points for recycling cans. -/
theorem bianca_points : points_earned 17 8 5 = 45 := by
  sorry

end bianca_points_l3815_381530


namespace tan_alpha_plus_pi_fourth_l3815_381562

theorem tan_alpha_plus_pi_fourth (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 7/23 := by
  sorry

end tan_alpha_plus_pi_fourth_l3815_381562


namespace first_two_satisfying_numbers_l3815_381500

def satisfiesConditions (n : ℕ) : Prop :=
  n % 7 = 3 ∧ n % 9 = 4

theorem first_two_satisfying_numbers :
  ∃ (a b : ℕ), a < b ∧
  satisfiesConditions a ∧
  satisfiesConditions b ∧
  (∀ (x : ℕ), x < a → ¬satisfiesConditions x) ∧
  (∀ (x : ℕ), a < x → x < b → ¬satisfiesConditions x) ∧
  a = 31 ∧ b = 94 := by
  sorry

end first_two_satisfying_numbers_l3815_381500


namespace odd_binomial_coefficients_count_l3815_381555

theorem odd_binomial_coefficients_count (n : ℕ) : 
  (Finset.sum (Finset.range (2^n)) (λ u => 
    (Finset.sum (Finset.range (u+1)) (λ v => 
      if Nat.choose u v % 2 = 1 then 1 else 0
    ))
  )) = 3^n := by sorry

end odd_binomial_coefficients_count_l3815_381555


namespace solution_set_inequality_l3815_381566

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici (1/2) :=
by sorry

end solution_set_inequality_l3815_381566


namespace purchase_combinations_eq_545_l3815_381523

/-- Represents the number of oreo flavors available -/
def oreo_flavors : ℕ := 6

/-- Represents the number of milk flavors available -/
def milk_flavors : ℕ := 4

/-- Represents the total number of products purchased -/
def total_products : ℕ := 3

/-- Represents the number of flavors Alpha can choose from (excluding chocolate) -/
def alpha_flavors : ℕ := oreo_flavors - 1 + milk_flavors

/-- Function to calculate the number of ways Alpha and Beta can purchase products -/
def purchase_combinations : ℕ := sorry

/-- Theorem stating the correct number of purchase combinations -/
theorem purchase_combinations_eq_545 : purchase_combinations = 545 := by sorry

end purchase_combinations_eq_545_l3815_381523


namespace cousins_age_sum_l3815_381506

theorem cousins_age_sum (ages : Fin 5 → ℕ) 
  (mean_condition : (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 10)
  (median_condition : ages 2 = 7)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 4 = 29 := by
sorry

end cousins_age_sum_l3815_381506


namespace base_conversion_185_to_113_l3815_381574

/-- Converts a base 13 number to base 10 --/
def base13ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 13^2 + tens * 13^1 + ones * 13^0

/-- Checks if a number is a valid base 13 digit --/
def isValidBase13Digit (d : Nat) : Prop :=
  d < 13

theorem base_conversion_185_to_113 :
  (∀ d, isValidBase13Digit d → d < 13) →
  base13ToBase10 1 1 3 = 185 :=
sorry

end base_conversion_185_to_113_l3815_381574


namespace distance_between_places_l3815_381532

/-- The distance between two places given speed changes and time differences --/
theorem distance_between_places (x : ℝ) (y : ℝ) : 
  ((x + 6) * (y - 5/60) = x * y) →
  ((x - 5) * (y + 6/60) = x * y) →
  x * y = 15 := by
sorry

end distance_between_places_l3815_381532


namespace circle_area_from_circumference_l3815_381528

/-- Given a circle with circumference 87.98229536926875 cm, its area is approximately 615.75164 square centimeters. -/
theorem circle_area_from_circumference : 
  let circumference : ℝ := 87.98229536926875
  let radius : ℝ := circumference / (2 * Real.pi)
  let area : ℝ := Real.pi * radius ^ 2
  ∃ ε > 0, abs (area - 615.75164) < ε :=
by
  sorry

end circle_area_from_circumference_l3815_381528


namespace circle_center_l3815_381508

/-- The center of the circle with equation x^2 + y^2 - x + 2y = 0 has coordinates (1/2, -1) -/
theorem circle_center (x y : ℝ) : 
  x^2 + y^2 - x + 2*y = 0 → (x - 1/2)^2 + (y + 1)^2 = 5/4 := by
sorry

end circle_center_l3815_381508


namespace value_of_lg_ta_ratio_l3815_381538

-- Define the necessary functions
noncomputable def sn (x : ℝ) : ℝ := Real.sin x
noncomputable def si (x : ℝ) : ℝ := Real.sin x
noncomputable def ta (x : ℝ) : ℝ := Real.tan x
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem value_of_lg_ta_ratio (α β : ℝ) 
  (h1 : sn (α + β) = 1/2) 
  (h2 : si (α - β) = 1/3) : 
  lg (5 * (ta α / ta β)) = 1 := by
  sorry

end value_of_lg_ta_ratio_l3815_381538


namespace xy_square_sum_l3815_381520

theorem xy_square_sum (x y : ℝ) (h1 : x + y = -2) (h2 : x * y = -3) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end xy_square_sum_l3815_381520


namespace ratio_difference_problem_l3815_381550

theorem ratio_difference_problem (A B : ℚ) : 
  A / B = 3 / 5 → B - A = 12 → A = 18 := by
  sorry

end ratio_difference_problem_l3815_381550


namespace cubic_polynomial_sum_of_coefficients_l3815_381522

theorem cubic_polynomial_sum_of_coefficients 
  (A B C : ℝ) (v : ℂ) :
  let Q : ℂ → ℂ := λ z ↦ z^3 + A*z^2 + B*z + C
  (∀ z : ℂ, Q z = 0 ↔ z = v - 2*I ∨ z = v + 7*I ∨ z = 3*v + 5) →
  A + B + C = Q 1 - 1 :=
by sorry

end cubic_polynomial_sum_of_coefficients_l3815_381522


namespace boric_acid_solution_percentage_l3815_381509

/-- Proves that the percentage of boric acid in the first solution must be 1% 
    to create a 3% boric acid solution under the given conditions -/
theorem boric_acid_solution_percentage 
  (total_volume : ℝ) 
  (final_concentration : ℝ) 
  (volume1 : ℝ) 
  (volume2 : ℝ) 
  (concentration2 : ℝ) 
  (h1 : total_volume = 30)
  (h2 : final_concentration = 0.03)
  (h3 : volume1 = 15)
  (h4 : volume2 = 15)
  (h5 : concentration2 = 0.05)
  (h6 : volume1 + volume2 = total_volume)
  : ∃ (concentration1 : ℝ), 
    concentration1 = 0.01 ∧ 
    concentration1 * volume1 + concentration2 * volume2 = final_concentration * total_volume :=
by sorry

end boric_acid_solution_percentage_l3815_381509


namespace infinite_series_sum_l3815_381541

theorem infinite_series_sum : 
  (∑' k : ℕ, (k : ℝ) / (3 : ℝ) ^ k) = (1 : ℝ) / 4 := by
  sorry

end infinite_series_sum_l3815_381541


namespace polygon_angle_theorem_l3815_381521

/-- 
Theorem: For a convex n-sided polygon with one interior angle x° and 
the sum of the remaining interior angles 2180°, x = 160° and n = 15.
-/
theorem polygon_angle_theorem (n : ℕ) (x : ℝ) 
  (h_convex : n ≥ 3)
  (h_sum : x + 2180 = 180 * (n - 2)) :
  x = 160 ∧ n = 15 := by
  sorry

end polygon_angle_theorem_l3815_381521


namespace red_chips_probability_l3815_381512

theorem red_chips_probability (total_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ) 
  (h1 : total_chips = red_chips + green_chips)
  (h2 : red_chips = 5)
  (h3 : green_chips = 3) :
  (Nat.choose (total_chips - 1) (green_chips - 1) : ℚ) / (Nat.choose total_chips green_chips) = 3/8 :=
sorry

end red_chips_probability_l3815_381512


namespace prob_king_or_queen_in_special_deck_l3815_381539

structure Deck :=
  (total_cards : ℕ)
  (num_ranks : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)

def probability_king_or_queen (d : Deck) : ℚ :=
  let kings_and_queens := d.num_suits * 2
  kings_and_queens / d.total_cards

theorem prob_king_or_queen_in_special_deck :
  let d : Deck := {
    total_cards := 60,
    num_ranks := 15,
    num_suits := 4,
    cards_per_suit := 15
  }
  probability_king_or_queen d = 2 / 15 := by sorry

end prob_king_or_queen_in_special_deck_l3815_381539


namespace ariel_age_quadruples_l3815_381503

/-- Proves that it takes 15 years for Ariel to be four times her current age -/
theorem ariel_age_quadruples (current_age : ℕ) (years_passed : ℕ) : current_age = 5 →
  current_age + years_passed = 4 * current_age →
  years_passed = 15 := by
  sorry

end ariel_age_quadruples_l3815_381503


namespace equation_solution_l3815_381559

theorem equation_solution : 
  ∃ x : ℝ, (x + 6) / (x - 3) = 4 ∧ x = 6 := by sorry

end equation_solution_l3815_381559


namespace greatest_power_of_two_factor_l3815_381560

theorem greatest_power_of_two_factor (n : ℕ) : 
  2^1200 ∣ (15^600 - 3^600) ∧ 
  ∀ k > 1200, ¬(2^k ∣ (15^600 - 3^600)) :=
sorry

end greatest_power_of_two_factor_l3815_381560


namespace adult_ticket_cost_l3815_381569

theorem adult_ticket_cost (student_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (adult_tickets : ℕ) (student_tickets : ℕ) :
  student_ticket_cost = 3 →
  total_tickets = 846 →
  total_revenue = 3846 →
  adult_tickets = 410 →
  student_tickets = 436 →
  ∃ (adult_ticket_cost : ℝ), adult_ticket_cost = 6.19 ∧
    adult_ticket_cost * adult_tickets + student_ticket_cost * student_tickets = total_revenue :=
by
  sorry

end adult_ticket_cost_l3815_381569


namespace positive_sum_and_product_equivalence_l3815_381579

theorem positive_sum_and_product_equivalence (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by sorry

end positive_sum_and_product_equivalence_l3815_381579


namespace elberta_has_35_5_l3815_381501

/-- The amount of money Granny Smith has -/
def granny_smith_amount : ℚ := 81

/-- The amount of money Anjou has -/
def anjou_amount : ℚ := granny_smith_amount / 4

/-- The amount of money Elberta has -/
def elberta_amount : ℚ := 2 * anjou_amount - 5

/-- Theorem stating that Elberta has $35.5 -/
theorem elberta_has_35_5 : elberta_amount = 35.5 := by
  sorry

end elberta_has_35_5_l3815_381501


namespace beavers_working_on_home_l3815_381553

/-- The number of beavers initially working on their home -/
def initial_beavers : ℕ := 2

/-- The number of beavers that went for a swim -/
def swimming_beavers : ℕ := 1

/-- The number of beavers still working on their home -/
def remaining_beavers : ℕ := initial_beavers - swimming_beavers

theorem beavers_working_on_home :
  remaining_beavers = 1 :=
by sorry

end beavers_working_on_home_l3815_381553


namespace cosine_angle_C_l3815_381524

/-- Given a triangle ABC with side lengths and angle relation, prove the cosine of angle C -/
theorem cosine_angle_C (A B C : ℝ) (BC AC : ℝ) (h1 : BC = 5) (h2 : AC = 4) 
  (h3 : Real.cos (A - B) = 7/8) : Real.cos C = 9/16 := by
  sorry

end cosine_angle_C_l3815_381524


namespace units_digit_E_1000_l3815_381519

def E (n : ℕ) : ℕ := 3^(3^n) + 1

theorem units_digit_E_1000 : E 1000 % 10 = 4 := by sorry

end units_digit_E_1000_l3815_381519


namespace cube_units_digits_eq_all_digits_l3815_381556

/-- The set of all single digits -/
def AllDigits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- The set of units digits of integral perfect cubes -/
def CubeUnitsDigits : Set Nat :=
  {d | ∃ n : Nat, d = (n^3) % 10}

/-- Theorem: The set of units digits of integral perfect cubes
    is equal to the set of all single digits -/
theorem cube_units_digits_eq_all_digits :
  CubeUnitsDigits = AllDigits := by sorry

end cube_units_digits_eq_all_digits_l3815_381556
