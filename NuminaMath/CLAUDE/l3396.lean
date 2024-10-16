import Mathlib

namespace NUMINAMATH_CALUDE_fish_catch_total_l3396_339618

def fish_problem (bass : ℕ) (trout : ℕ) (blue_gill : ℕ) : Prop :=
  (bass = 32) ∧
  (trout = bass / 4) ∧
  (blue_gill = 2 * bass) ∧
  (bass + trout + blue_gill = 104)

theorem fish_catch_total :
  ∀ (bass trout blue_gill : ℕ), fish_problem bass trout blue_gill :=
by
  sorry

end NUMINAMATH_CALUDE_fish_catch_total_l3396_339618


namespace NUMINAMATH_CALUDE_total_students_count_l3396_339631

def third_grade : ℕ := 19

def fourth_grade : ℕ := 2 * third_grade

def second_grade_boys : ℕ := 10
def second_grade_girls : ℕ := 19

def total_students : ℕ := third_grade + fourth_grade + second_grade_boys + second_grade_girls

theorem total_students_count : total_students = 86 := by sorry

end NUMINAMATH_CALUDE_total_students_count_l3396_339631


namespace NUMINAMATH_CALUDE_initial_stock_calculation_l3396_339675

/-- Given that 450 bags represent 75% of the initial stock, prove that the initial stock was 600 bags. -/
theorem initial_stock_calculation (sold : ℕ) (percentage_sold : ℚ) (h1 : sold = 450) (h2 : percentage_sold = 3/4) :
  (sold : ℚ) / percentage_sold = 600 := by
  sorry

end NUMINAMATH_CALUDE_initial_stock_calculation_l3396_339675


namespace NUMINAMATH_CALUDE_combined_salaries_l3396_339684

/-- Given the salary of E and the average salary of A, B, C, D, and E,
    calculate the combined salaries of A, B, C, and D. -/
theorem combined_salaries 
  (salary_E : ℕ) 
  (average_salary : ℕ) 
  (h1 : salary_E = 9000)
  (h2 : average_salary = 8600) :
  (5 * average_salary) - salary_E = 34000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l3396_339684


namespace NUMINAMATH_CALUDE_profit_percentage_l3396_339685

/-- Given that the cost price of 58 articles equals the selling price of 50 articles, 
    the percent profit is 16%. -/
theorem profit_percentage (C S : ℝ) (h : 58 * C = 50 * S) : 
  (S - C) / C * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_l3396_339685


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_neg_two_a_range_when_f_leq_g_on_interval_l3396_339666

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |2*x - 1| + |2*x + a|
def g (x : ℝ) : ℝ := x + 3

-- Part 1
theorem solution_set_when_a_is_neg_two :
  {x : ℝ | f (-2) x < g x} = Set.Ioo 0 2 := by sorry

-- Part 2
theorem a_range_when_f_leq_g_on_interval :
  ∀ a : ℝ, a > -1 →
  (∀ x ∈ Set.Icc (-a/2) (1/2), f a x ≤ g x) →
  a ∈ Set.Ioo (-1) (4/3) := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_neg_two_a_range_when_f_leq_g_on_interval_l3396_339666


namespace NUMINAMATH_CALUDE_coin_count_l3396_339652

theorem coin_count (total_amount : ℕ) (five_dollar_count : ℕ) : 
  total_amount = 125 →
  five_dollar_count = 15 →
  ∃ (two_dollar_count : ℕ), 
    two_dollar_count * 2 + five_dollar_count * 5 = total_amount ∧
    two_dollar_count + five_dollar_count = 40 :=
by sorry

end NUMINAMATH_CALUDE_coin_count_l3396_339652


namespace NUMINAMATH_CALUDE_smallest_n_exceeding_500000_l3396_339654

theorem smallest_n_exceeding_500000 : 
  (∀ k : ℕ, k < 10 → (3 : ℝ) ^ ((k * (k + 1) : ℝ) / 16) ≤ 500000) ∧ 
  (3 : ℝ) ^ ((10 * 11 : ℝ) / 16) > 500000 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_exceeding_500000_l3396_339654


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l3396_339606

theorem rectangle_area_diagonal_relation (l w d : ℝ) (h1 : l / w = 4 / 3) (h2 : l^2 + w^2 = d^2) :
  ∃ k : ℝ, l * w = k * d^2 ∧ k = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l3396_339606


namespace NUMINAMATH_CALUDE_five_month_practice_time_l3396_339602

/-- Calculates the total piano practice time over a given number of months. -/
def total_practice_time (weekly_hours : ℕ) (weeks_per_month : ℕ) (months : ℕ) : ℕ :=
  weekly_hours * weeks_per_month * months

/-- Theorem stating that practicing 4 hours per week for 5 months results in 80 hours of practice. -/
theorem five_month_practice_time :
  total_practice_time 4 4 5 = 80 := by
  sorry

#eval total_practice_time 4 4 5

end NUMINAMATH_CALUDE_five_month_practice_time_l3396_339602


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l3396_339614

theorem complex_exponential_sum (α β θ : ℝ) :
  Complex.exp (Complex.I * (α + θ)) + Complex.exp (Complex.I * (β + θ)) = (1/3 : ℂ) + (4/9 : ℂ) * Complex.I →
  Complex.exp (-Complex.I * (α + θ)) + Complex.exp (-Complex.I * (β + θ)) = (1/3 : ℂ) - (4/9 : ℂ) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l3396_339614


namespace NUMINAMATH_CALUDE_circle_chord_difference_equals_radius_l3396_339676

theorem circle_chord_difference_equals_radius (R : ℝ) (h : R > 0) :
  let chord_length (θ : ℝ) := 2 * R * Real.sin (θ / 2)
  chord_length (3 * π / 5) - chord_length (π / 5) = R :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_difference_equals_radius_l3396_339676


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3396_339610

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5*x + 6 < 0 ↔ 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3396_339610


namespace NUMINAMATH_CALUDE_unique_three_digit_even_with_digit_sum_26_l3396_339695

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a 3-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The set of 3-digit even numbers with digit sum 26 -/
def S : Set ℕ := {n : ℕ | is_three_digit n ∧ Even n ∧ digit_sum n = 26}

theorem unique_three_digit_even_with_digit_sum_26 : ∃! n, n ∈ S := by sorry

end NUMINAMATH_CALUDE_unique_three_digit_even_with_digit_sum_26_l3396_339695


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3396_339608

theorem not_sufficient_nor_necessary : 
  ¬(∀ x : ℝ, x < 0 → Real.log (x + 1) ≤ 0) ∧ 
  ¬(∀ x : ℝ, Real.log (x + 1) ≤ 0 → x < 0) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l3396_339608


namespace NUMINAMATH_CALUDE_gangster_undetected_conditions_l3396_339657

/-- Configuration of streets and houses -/
structure StreetConfig where
  a : ℝ  -- Side length of houses
  street_distance : ℝ  -- Distance between parallel streets
  house_gap : ℝ  -- Distance between neighboring houses
  police_interval : ℝ  -- Interval between police officers

/-- Movement parameters -/
structure MovementParams where
  police_speed : ℝ  -- Speed of police officers
  gangster_speed : ℝ  -- Speed of the gangster
  gangster_direction : Bool  -- True if moving towards police, False otherwise

/-- Predicate to check if the gangster remains undetected -/
def remains_undetected (config : StreetConfig) (params : MovementParams) : Prop :=
  (params.gangster_direction = true) ∧ 
  ((params.gangster_speed = 2 * params.police_speed) ∨ 
   (params.gangster_speed = params.police_speed / 2))

/-- Main theorem: Conditions for the gangster to remain undetected -/
theorem gangster_undetected_conditions 
  (config : StreetConfig) 
  (params : MovementParams) :
  config.street_distance = 3 * config.a ∧ 
  config.house_gap = 2 * config.a ∧
  config.police_interval = 9 * config.a ∧
  params.police_speed > 0 →
  remains_undetected config params ↔ 
  (params.gangster_direction = true ∧ 
   (params.gangster_speed = 2 * params.police_speed ∨ 
    params.gangster_speed = params.police_speed / 2)) :=
by sorry

end NUMINAMATH_CALUDE_gangster_undetected_conditions_l3396_339657


namespace NUMINAMATH_CALUDE_min_sqrt_difference_l3396_339624

theorem min_sqrt_difference (p : ℕ) (h_prime : Nat.Prime p) (h_odd : Odd p) :
  ∃ (m n : ℕ), 
    0 < m ∧ 0 < n ∧ m ≤ n ∧
    (∀ (a b : ℕ), 0 < a → 0 < b → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt m - Real.sqrt n ≤ 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    m = (p - 1) / 2 ∧ n = (p + 1) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_sqrt_difference_l3396_339624


namespace NUMINAMATH_CALUDE_polynomial_coefficient_difference_l3396_339672

theorem polynomial_coefficient_difference (m n : ℝ) : 
  (∀ x : ℝ, 3 * x * (x - 1) = m * x^2 + n * x) → m - n = 6 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_difference_l3396_339672


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_two_range_of_a_for_all_real_solution_l3396_339600

def f (a x : ℝ) : ℝ := a * x^2 + a * x - 1

theorem solution_set_when_a_is_two :
  let a := 2
  {x : ℝ | f a x < 0} = {x : ℝ | -(1 + Real.sqrt 3) / 2 < x ∧ x < (-1 + Real.sqrt 3) / 2} := by sorry

theorem range_of_a_for_all_real_solution :
  {a : ℝ | ∀ x, f a x < 0} = {a : ℝ | -4 < a ∧ a ≤ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_two_range_of_a_for_all_real_solution_l3396_339600


namespace NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_a_range_when_inequality_holds_l3396_339641

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for part (1)
theorem min_value_when_a_is_neg_three :
  ∃ (min : ℝ), min = 4 ∧ ∀ x, f (-3) x ≥ min :=
sorry

-- Theorem for part (2)
theorem a_range_when_inequality_holds :
  (∀ x, f a x ≤ 2*a + 2*|x - 1|) → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_neg_three_a_range_when_inequality_holds_l3396_339641


namespace NUMINAMATH_CALUDE_a3_value_l3396_339696

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) :=
  b ^ 2 = a * c

theorem a3_value (a : ℕ → ℝ) :
  arithmetic_sequence a 2 →
  geometric_sequence (a 1) (a 3) (a 4) →
  a 3 = -4 :=
by sorry

end NUMINAMATH_CALUDE_a3_value_l3396_339696


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3396_339693

/-- 
Given complex numbers a and b, this theorem states that a^2 = 2b ≠ 0 
if and only if the roots of the polynomial x^2 + ax + b form an isosceles 
right triangle on the complex plane with the right angle at the origin.
-/
theorem isosceles_right_triangle_roots 
  (a b : ℂ) : a^2 = 2*b ∧ b ≠ 0 ↔ 
  ∃ (x₁ x₂ : ℂ), x₁^2 + a*x₁ + b = 0 ∧ 
                 x₂^2 + a*x₂ + b = 0 ∧ 
                 x₁ ≠ x₂ ∧
                 (x₁ = Complex.I * x₂ ∨ x₂ = Complex.I * x₁) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_roots_l3396_339693


namespace NUMINAMATH_CALUDE_equation_solution_expression_simplification_l3396_339668

-- Part 1
theorem equation_solution :
  ∃ x : ℝ, (x / (2*x - 3) + 5 / (3 - 2*x) = 4) ∧ (x = 1) :=
sorry

-- Part 2
theorem expression_simplification (a : ℝ) (h : a ≠ 2 ∧ a ≠ -2) :
  (a - 2 - 4 / (a - 2)) / ((a - 4) / (a^2 - 4)) = a^2 + 2*a :=
sorry

end NUMINAMATH_CALUDE_equation_solution_expression_simplification_l3396_339668


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l3396_339632

theorem at_least_one_less_than_two (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a + b > 2) : 
  (1 + b) / a < 2 ∨ (1 + a) / b < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l3396_339632


namespace NUMINAMATH_CALUDE_slide_total_l3396_339635

theorem slide_total (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 22 → additional = 13 → total = initial + additional → total = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_total_l3396_339635


namespace NUMINAMATH_CALUDE_three_integer_solutions_l3396_339680

theorem three_integer_solutions (n : ℕ) (x₁ y₁ : ℤ) 
  (h : x₁^3 - 3*x₁*y₁^2 + y₁^3 = n) : 
  ∃ (x₂ y₂ x₃ y₃ : ℤ), 
    (x₂^3 - 3*x₂*y₂^2 + y₂^3 = n) ∧ 
    (x₃^3 - 3*x₃*y₃^2 + y₃^3 = n) ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ 
    (x₁ ≠ x₃ ∨ y₁ ≠ y₃) ∧ 
    (x₂ ≠ x₃ ∨ y₂ ≠ y₃) := by
  sorry

end NUMINAMATH_CALUDE_three_integer_solutions_l3396_339680


namespace NUMINAMATH_CALUDE_sequence_periodicity_l3396_339628

theorem sequence_periodicity (u : ℕ → ℝ) 
  (h : ∀ n : ℕ, u (n + 2) = |u (n + 1)| - u n) : 
  ∃ p : ℕ+, ∀ n : ℕ, u n = u (n + p) := by sorry

end NUMINAMATH_CALUDE_sequence_periodicity_l3396_339628


namespace NUMINAMATH_CALUDE_quality_difference_proof_l3396_339604

-- Define the data from the problem
def total_products : ℕ := 400
def machine_a_first_class : ℕ := 150
def machine_a_second_class : ℕ := 50
def machine_b_first_class : ℕ := 120
def machine_b_second_class : ℕ := 80

-- Define the K² formula
def k_squared (n a b c d : ℕ) : ℚ :=
  (n * (a * d - b * c)^2 : ℚ) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the theorem
theorem quality_difference_proof :
  (machine_a_first_class : ℚ) / (machine_a_first_class + machine_a_second_class) = 3/4 ∧
  (machine_b_first_class : ℚ) / (machine_b_first_class + machine_b_second_class) = 3/5 ∧
  k_squared total_products machine_a_first_class machine_a_second_class machine_b_first_class machine_b_second_class > 6635/1000 :=
by sorry

end NUMINAMATH_CALUDE_quality_difference_proof_l3396_339604


namespace NUMINAMATH_CALUDE_ice_water_masses_l3396_339674

/-- Proof of initial ice and water masses in a cylindrical vessel --/
theorem ice_water_masses
  (S : ℝ) (ρw ρi : ℝ) (hf Δh : ℝ)
  (h_S : S = 15)
  (h_ρw : ρw = 1)
  (h_ρi : ρi = 0.9)
  (h_hf : hf = 115)
  (h_Δh : Δh = 5) :
  ∃ (m_ice m_water : ℝ),
    m_ice = 675 ∧
    m_water = 1050 ∧
    m_ice / ρi - m_ice / ρw = S * Δh ∧
    m_water = ρw * S * hf - m_ice :=
by sorry

end NUMINAMATH_CALUDE_ice_water_masses_l3396_339674


namespace NUMINAMATH_CALUDE_sum_x_y_equals_eight_l3396_339601

theorem sum_x_y_equals_eight (x y : ℝ) 
  (h1 : |x| + x + y = 14)
  (h2 : x + |y| - y = 10)
  (h3 : |x| - |y| + x - y = 8) :
  x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_eight_l3396_339601


namespace NUMINAMATH_CALUDE_sum_base4_to_base10_l3396_339697

/-- Converts a base 4 number represented as a list of digits to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The sum of 2213₄, 2703₄, and 1531₄ in base 10 is 309 -/
theorem sum_base4_to_base10 :
  base4ToBase10 [3, 1, 2, 2] + base4ToBase10 [3, 0, 7, 2] + base4ToBase10 [1, 3, 5, 1] = 309 := by
  sorry

#eval base4ToBase10 [3, 1, 2, 2] + base4ToBase10 [3, 0, 7, 2] + base4ToBase10 [1, 3, 5, 1]

end NUMINAMATH_CALUDE_sum_base4_to_base10_l3396_339697


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3396_339698

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ ∃ r : ℝ, r > 0 ∧ ∀ k : ℕ, a (k + 1) = r * a k

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : is_positive_geometric_sequence a)
  (h_roots : a 4 * a 6 = 6 ∧ a 4 + a 6 = 5) :
  a 5 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3396_339698


namespace NUMINAMATH_CALUDE_one_nonnegative_solution_l3396_339690

theorem one_nonnegative_solution :
  ∃! (x : ℝ), x ≥ 0 ∧ x^2 + 6*x = 0 :=
by sorry

end NUMINAMATH_CALUDE_one_nonnegative_solution_l3396_339690


namespace NUMINAMATH_CALUDE_supplementary_angle_of_10_degrees_l3396_339609

def is_supplementary (a b : ℝ) : Prop :=
  (a + b) % 360 = 180

theorem supplementary_angle_of_10_degrees (k : ℤ) :
  is_supplementary 10 (k * 360 + 250) :=
sorry

end NUMINAMATH_CALUDE_supplementary_angle_of_10_degrees_l3396_339609


namespace NUMINAMATH_CALUDE_train_length_l3396_339659

theorem train_length (tunnel_length platform_length tunnel_time platform_time : ℝ) 
  (h1 : tunnel_length = 1200)
  (h2 : platform_length = 180)
  (h3 : tunnel_time = 45)
  (h4 : platform_time = 15)
  : ∃ train_length : ℝ, 
    train_length + tunnel_length = (train_length + platform_length) * (tunnel_time / platform_time) ∧
    train_length = 330 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3396_339659


namespace NUMINAMATH_CALUDE_team_selection_count_l3396_339694

theorem team_selection_count (total : ℕ) (veterans : ℕ) (new : ℕ) (team_size : ℕ) (max_veterans : ℕ) :
  total = veterans + new →
  total = 10 →
  veterans = 2 →
  new = 8 →
  team_size = 3 →
  max_veterans = 1 →
  Nat.choose (new - 1) team_size + veterans * Nat.choose (new - 1) (team_size - 1) = 77 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_count_l3396_339694


namespace NUMINAMATH_CALUDE_N2O3_molecular_weight_l3396_339673

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in g/mol -/
def N2O3_weight : ℝ := nitrogen_count * nitrogen_weight + oxygen_count * oxygen_weight

theorem N2O3_molecular_weight : N2O3_weight = 76.02 := by
  sorry

end NUMINAMATH_CALUDE_N2O3_molecular_weight_l3396_339673


namespace NUMINAMATH_CALUDE_work_done_by_force_l3396_339648

theorem work_done_by_force (F : ℝ → ℝ) (x₁ x₂ : ℝ) :
  (∀ x, F x = 1 + Real.exp x) →
  x₁ = 0 →
  x₂ = 1 →
  ∫ x in x₁..x₂, F x = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_work_done_by_force_l3396_339648


namespace NUMINAMATH_CALUDE_bankers_discount_l3396_339633

/-- Banker's discount calculation --/
theorem bankers_discount (bankers_gain : ℚ) (interest_rate : ℚ) (time : ℕ) : 
  bankers_gain = 270 → interest_rate = 12 / 100 → time = 3 → 
  let present_value := (bankers_gain * 100) / (interest_rate * time)
  let face_value := present_value + bankers_gain
  let bankers_discount := (face_value * interest_rate * time)
  bankers_discount = 36720 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bankers_discount_l3396_339633


namespace NUMINAMATH_CALUDE_worker_b_completion_time_l3396_339691

/-- Given workers A, B, and C, and their work rates, prove that B can complete the work alone in 5 days -/
theorem worker_b_completion_time 
  (total_work : ℝ) 
  (rate_a : ℝ) (rate_b : ℝ) (rate_c : ℝ) 
  (time_a : ℝ) (time_b : ℝ) (time_c : ℝ) (time_abc : ℝ) 
  (h1 : rate_a = total_work / time_a)
  (h2 : rate_b = total_work / time_b)
  (h3 : rate_c = total_work / time_c)
  (h4 : rate_a + rate_b + rate_c = total_work / time_abc)
  (h5 : time_a = 4)
  (h6 : time_c = 20)
  (h7 : time_abc = 2)
  (h8 : total_work > 0) :
  time_b = 5 := by
  sorry

end NUMINAMATH_CALUDE_worker_b_completion_time_l3396_339691


namespace NUMINAMATH_CALUDE_larger_part_of_90_l3396_339669

theorem larger_part_of_90 (x : ℝ) : 
  x + (90 - x) = 90 ∧ 
  0.4 * x = 0.3 * (90 - x) + 15 → 
  max x (90 - x) = 60 := by
sorry

end NUMINAMATH_CALUDE_larger_part_of_90_l3396_339669


namespace NUMINAMATH_CALUDE_min_value_expression_l3396_339637

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2) / (a*b + 2*b*c) ≥ 2 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3396_339637


namespace NUMINAMATH_CALUDE_ascending_order_l3396_339629

theorem ascending_order (x y : ℝ) (hx : x > 1) (hy : -1 < y ∧ y < 0) :
  y < -y ∧ -y < -x*y ∧ -x*y < x := by sorry

end NUMINAMATH_CALUDE_ascending_order_l3396_339629


namespace NUMINAMATH_CALUDE_masha_ate_ten_pies_l3396_339639

/-- Represents the eating rates of Masha and the bear -/
structure EatingRates where
  masha : ℝ
  bear : ℝ
  bear_faster : bear = 3 * masha

/-- Represents the distribution of food between Masha and the bear -/
structure FoodDistribution where
  total_pies : ℕ
  total_pies_positive : total_pies > 0
  masha_pies : ℕ
  bear_pies : ℕ
  pies_sum : masha_pies + bear_pies = total_pies
  equal_raspberries : ℝ  -- Represents the fact that they ate equal raspberries

/-- Theorem stating that Masha ate 10 pies given the problem conditions -/
theorem masha_ate_ten_pies (rates : EatingRates) (food : FoodDistribution) 
  (h_total_pies : food.total_pies = 40) :
  food.masha_pies = 10 := by
  sorry


end NUMINAMATH_CALUDE_masha_ate_ten_pies_l3396_339639


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_set_A_forms_triangle_l3396_339665

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_inequality_theorem (a b c : ℝ) :
  can_form_triangle a b c ↔ (a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

theorem set_A_forms_triangle :
  can_form_triangle 8 6 5 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_set_A_forms_triangle_l3396_339665


namespace NUMINAMATH_CALUDE_cistern_fill_time_l3396_339649

def fill_time_p : ℝ := 12
def fill_time_q : ℝ := 15
def initial_time : ℝ := 6

theorem cistern_fill_time : 
  let rate_p := 1 / fill_time_p
  let rate_q := 1 / fill_time_q
  let initial_fill := (rate_p + rate_q) * initial_time
  let remaining_fill := 1 - initial_fill
  remaining_fill / rate_q = 1.5 := by
sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l3396_339649


namespace NUMINAMATH_CALUDE_fraction_addition_l3396_339605

theorem fraction_addition (c : ℝ) : (5 + 5 * c) / 7 + 3 = (26 + 5 * c) / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3396_339605


namespace NUMINAMATH_CALUDE_max_value_of_s_l3396_339615

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 10) 
  (sum_products_eq : p*q + p*r + p*s + q*r + q*s + r*s = 20) : 
  s ≤ (5 + Real.sqrt 105) / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_s_l3396_339615


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3396_339677

theorem geometric_series_sum (c d : ℝ) (h : ∑' n, c / d^n = 3) :
  ∑' n, c / (c + 2*d)^n = (3*d - 3) / (5*d - 4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3396_339677


namespace NUMINAMATH_CALUDE_workshop_workers_l3396_339655

theorem workshop_workers (total_avg : ℝ) (tech_count : ℕ) (tech_avg : ℝ) (rest_avg : ℝ)
  (h1 : total_avg = 8000)
  (h2 : tech_count = 7)
  (h3 : tech_avg = 16000)
  (h4 : rest_avg = 6000) :
  ∃ (total_workers : ℕ),
    (total_workers : ℝ) * total_avg = 
      (tech_count : ℝ) * tech_avg + ((total_workers - tech_count) : ℝ) * rest_avg ∧
    total_workers = 35 :=
by sorry

end NUMINAMATH_CALUDE_workshop_workers_l3396_339655


namespace NUMINAMATH_CALUDE_expression_value_l3396_339646

theorem expression_value (x y : ℝ) (h : x - 3*y = 4) :
  (x - 3*y)^2 + 2*x - 6*y - 10 = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3396_339646


namespace NUMINAMATH_CALUDE_shopkeeper_loss_l3396_339670

/-- Represents the overall loss amount given stock worth and selling conditions --/
def overall_loss (stock_worth : ℝ) : ℝ :=
  let profit_part := 0.2 * stock_worth * 1.2
  let loss_part := 0.8 * stock_worth * 0.9
  stock_worth - (profit_part + loss_part)

/-- Theorem stating the overall loss for the given problem --/
theorem shopkeeper_loss : 
  overall_loss 12499.99 = 500 :=
by
  sorry

#eval overall_loss 12499.99

end NUMINAMATH_CALUDE_shopkeeper_loss_l3396_339670


namespace NUMINAMATH_CALUDE_inscribed_triangle_property_l3396_339687

-- Define the triangle XYZ
def Triangle (X Y Z : ℝ × ℝ) : Prop :=
  let xy := Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
  let yz := Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2)
  let xz := Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2)
  xy = 26 ∧ yz = 28 ∧ xz = 27

-- Define the inscribed triangle GHI
def InscribedTriangle (X Y Z G H I : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ,
    0 < t₁ ∧ t₁ < 1 ∧
    0 < t₂ ∧ t₂ < 1 ∧
    0 < t₃ ∧ t₃ < 1 ∧
    G = (t₁ * Y.1 + (1 - t₁) * Z.1, t₁ * Y.2 + (1 - t₁) * Z.2) ∧
    H = (t₂ * X.1 + (1 - t₂) * Z.1, t₂ * X.2 + (1 - t₂) * Z.2) ∧
    I = (t₃ * X.1 + (1 - t₃) * Y.1, t₃ * X.2 + (1 - t₃) * Y.2)

-- Define the equality of arcs
def ArcEqual (X Y Z G H I : ℝ × ℝ) : Prop :=
  let yi := Real.sqrt ((Y.1 - I.1)^2 + (Y.2 - I.2)^2)
  let gz := Real.sqrt ((G.1 - Z.1)^2 + (G.2 - Z.2)^2)
  let xi := Real.sqrt ((X.1 - I.1)^2 + (X.2 - I.2)^2)
  let hz := Real.sqrt ((H.1 - Z.1)^2 + (H.2 - Z.2)^2)
  let xh := Real.sqrt ((X.1 - H.1)^2 + (X.2 - H.2)^2)
  let gy := Real.sqrt ((G.1 - Y.1)^2 + (G.2 - Y.2)^2)
  yi = gz ∧ xi = hz ∧ xh = gy

theorem inscribed_triangle_property
  (X Y Z G H I : ℝ × ℝ)
  (h₁ : Triangle X Y Z)
  (h₂ : InscribedTriangle X Y Z G H I)
  (h₃ : ArcEqual X Y Z G H I) :
  let gy := Real.sqrt ((G.1 - Y.1)^2 + (G.2 - Y.2)^2)
  gy = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_property_l3396_339687


namespace NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3396_339607

def complex_number : ℂ := Complex.I * ((-2 : ℝ) + Complex.I)

theorem complex_number_in_third_quadrant :
  let z := complex_number
  (z.re < 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_third_quadrant_l3396_339607


namespace NUMINAMATH_CALUDE_dance_troupe_average_age_l3396_339638

theorem dance_troupe_average_age 
  (num_females : ℕ) 
  (num_males : ℕ) 
  (avg_age_females : ℚ) 
  (avg_age_males : ℚ) 
  (h1 : num_females = 12) 
  (h2 : num_males = 18) 
  (h3 : avg_age_females = 25) 
  (h4 : avg_age_males = 30) : 
  (num_females * avg_age_females + num_males * avg_age_males) / (num_females + num_males) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dance_troupe_average_age_l3396_339638


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l3396_339671

/-- The number of people in the group -/
def groupSize : ℕ := 6

/-- The number of ways to choose a President and Vice-President when A is not President -/
def waysWithoutA : ℕ := groupSize * (groupSize - 1)

/-- The number of ways to choose a President and Vice-President when A is President -/
def waysWithA : ℕ := 1 * (groupSize - 2)

/-- The total number of ways to choose a President and Vice-President -/
def totalWays : ℕ := waysWithoutA + waysWithA

theorem president_vice_president_selection :
  totalWays = 34 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l3396_339671


namespace NUMINAMATH_CALUDE_number_of_pears_number_of_pears_is_correct_l3396_339650

/-- The number of pears in a basket, given the following conditions:
  * There are 5 baskets in total
  * There are 58 fruits in total
  * One basket contains 18 mangoes
  * One basket contains 12 pawpaws
  * Two baskets contain the same number of kiwi and lemon respectively
  * There are 9 lemons
-/
theorem number_of_pears : ℕ :=
  let total_baskets : ℕ := 5
  let total_fruits : ℕ := 58
  let mangoes : ℕ := 18
  let pawpaws : ℕ := 12
  let lemons : ℕ := 9
  let kiwis : ℕ := lemons
  10

#check number_of_pears

theorem number_of_pears_is_correct : number_of_pears = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pears_number_of_pears_is_correct_l3396_339650


namespace NUMINAMATH_CALUDE_jerry_won_47_tickets_l3396_339656

/-- The number of tickets Jerry won later at the arcade -/
def tickets_won_later (initial_tickets : ℕ) (spent_tickets : ℕ) (final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Theorem: Jerry won 47 tickets later at the arcade -/
theorem jerry_won_47_tickets :
  tickets_won_later 4 2 49 = 47 := by
  sorry

end NUMINAMATH_CALUDE_jerry_won_47_tickets_l3396_339656


namespace NUMINAMATH_CALUDE_consecutive_integers_problem_l3396_339616

theorem consecutive_integers_problem (n : ℕ) (x : ℤ) : 
  n > 0 → 
  x + n - 1 = 23 → 
  (n : ℝ) * 20 = (n / 2 : ℝ) * (2 * x + n - 1) → 
  n = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_problem_l3396_339616


namespace NUMINAMATH_CALUDE_function_determination_l3396_339692

/-- Given a function f(x) = a^x + k, if f(1) = 3 and f(0) = 2, then f(x) = 2^x + 1 -/
theorem function_determination (a k : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a^x + k) 
  (h2 : f 1 = 3) 
  (h3 : f 0 = 2) : 
  ∀ x, f x = 2^x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l3396_339692


namespace NUMINAMATH_CALUDE_min_value_of_f_l3396_339679

/-- The quadratic function f(x) = 2(x-3)^2 + 2 -/
def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 2

/-- Theorem: The minimum value of f(x) = 2(x-3)^2 + 2 is 2 -/
theorem min_value_of_f :
  ∀ x : ℝ, f x ≥ 2 ∧ ∃ x₀ : ℝ, f x₀ = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3396_339679


namespace NUMINAMATH_CALUDE_simplify_fraction_l3396_339651

theorem simplify_fraction : (216 : ℚ) / 4536 = 1 / 21 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3396_339651


namespace NUMINAMATH_CALUDE_positive_expression_l3396_339661

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l3396_339661


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3396_339683

theorem sin_2theta_value (θ : Real) (h : Real.sin θ + Real.cos θ = 1/5) :
  Real.sin (2 * θ) = -24/25 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3396_339683


namespace NUMINAMATH_CALUDE_fencing_cost_theorem_l3396_339686

/-- The total cost of fencing a rectangular plot -/
def fencing_cost (length breadth cost_per_metre : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_metre

/-- Theorem: The total cost of fencing a rectangular plot with given dimensions -/
theorem fencing_cost_theorem (length breadth cost_per_metre : ℝ) 
  (h1 : length = 60)
  (h2 : breadth = length - 20)
  (h3 : cost_per_metre = 26.50) :
  fencing_cost length breadth cost_per_metre = 5300 := by
  sorry

#eval fencing_cost 60 40 26.50

end NUMINAMATH_CALUDE_fencing_cost_theorem_l3396_339686


namespace NUMINAMATH_CALUDE_max_integer_with_divisor_difference_twenty_four_satisfies_condition_l3396_339627

theorem max_integer_with_divisor_difference (n : ℕ) : 
  (∀ k : ℕ, k > 0 → k ≤ n / 2 → ∃ d₁ d₂ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ ∣ n ∧ d₂ ∣ n ∧ d₂ - d₁ = k) →
  n ≤ 24 :=
by sorry

theorem twenty_four_satisfies_condition : 
  ∀ k : ℕ, k > 0 → k ≤ 24 / 2 → ∃ d₁ d₂ : ℕ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ ∣ 24 ∧ d₂ ∣ 24 ∧ d₂ - d₁ = k :=
by sorry

end NUMINAMATH_CALUDE_max_integer_with_divisor_difference_twenty_four_satisfies_condition_l3396_339627


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3396_339622

theorem quadratic_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - 3*a = 0 ∧ x = -2) → 
  (∃ y : ℝ, y^2 - a*y - 3*a = 0 ∧ y = 6) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3396_339622


namespace NUMINAMATH_CALUDE_james_seed_planting_l3396_339603

/-- Calculates the percentage of seeds planted -/
def percentage_planted (original_trees : ℕ) (plants_per_tree : ℕ) (seeds_per_plant : ℕ) (new_trees : ℕ) : ℚ :=
  (new_trees : ℚ) / ((original_trees * plants_per_tree * seeds_per_plant) : ℚ) * 100

/-- Proves that the percentage of seeds planted is 60% given the problem conditions -/
theorem james_seed_planting :
  let original_trees : ℕ := 2
  let plants_per_tree : ℕ := 20
  let seeds_per_plant : ℕ := 1
  let new_trees : ℕ := 24
  percentage_planted original_trees plants_per_tree seeds_per_plant new_trees = 60 := by
  sorry

end NUMINAMATH_CALUDE_james_seed_planting_l3396_339603


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3396_339621

/-- The quadratic equation (2kx^2 + 7kx + 2) = 0 has equal roots when k = 16/49 -/
theorem equal_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + 7 * k * x + 2 = 0) → 
  (∃! r : ℝ, 2 * k * r^2 + 7 * k * r + 2 = 0) → 
  k = 16/49 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3396_339621


namespace NUMINAMATH_CALUDE_thabo_total_books_l3396_339619

/-- The number of books Thabo owns of each type and in total. -/
structure ThabosBooks where
  hardcover_nonfiction : ℕ
  paperback_nonfiction : ℕ
  paperback_fiction : ℕ
  total : ℕ

/-- The conditions of Thabo's book collection. -/
def thabo_book_conditions (books : ThabosBooks) : Prop :=
  books.hardcover_nonfiction = 30 ∧
  books.paperback_nonfiction = books.hardcover_nonfiction + 20 ∧
  books.paperback_fiction = 2 * books.paperback_nonfiction ∧
  books.total = books.hardcover_nonfiction + books.paperback_nonfiction + books.paperback_fiction

/-- Theorem stating that given the conditions, Thabo owns 180 books in total. -/
theorem thabo_total_books :
  ∀ books : ThabosBooks, thabo_book_conditions books → books.total = 180 := by
  sorry


end NUMINAMATH_CALUDE_thabo_total_books_l3396_339619


namespace NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l3396_339620

theorem product_xyz_equals_negative_one 
  (x y z : ℝ) 
  (h1 : x + 1 / y = 2) 
  (h2 : y + 1 / z = 2) : 
  x * y * z = -1 := by
sorry

end NUMINAMATH_CALUDE_product_xyz_equals_negative_one_l3396_339620


namespace NUMINAMATH_CALUDE_cow_chicken_problem_l3396_339636

theorem cow_chicken_problem (C H : ℕ) : 4*C + 2*H = 2*(C + H) + 10 → C = 5 :=
by sorry

end NUMINAMATH_CALUDE_cow_chicken_problem_l3396_339636


namespace NUMINAMATH_CALUDE_carolyn_stitching_rate_l3396_339613

/-- Represents the number of stitches required for a flower -/
def flower_stitches : ℕ := 60

/-- Represents the number of stitches required for a unicorn -/
def unicorn_stitches : ℕ := 180

/-- Represents the number of stitches required for Godzilla -/
def godzilla_stitches : ℕ := 800

/-- Represents the number of unicorns in the embroidery -/
def num_unicorns : ℕ := 3

/-- Represents the number of flowers in the embroidery -/
def num_flowers : ℕ := 50

/-- Represents the total time Carolyn spends embroidering (in minutes) -/
def total_time : ℕ := 1085

/-- Calculates Carolyn's stitching rate -/
def stitching_rate : ℚ :=
  (godzilla_stitches + num_unicorns * unicorn_stitches + num_flowers * flower_stitches) / total_time

theorem carolyn_stitching_rate :
  stitching_rate = 4 := by sorry

end NUMINAMATH_CALUDE_carolyn_stitching_rate_l3396_339613


namespace NUMINAMATH_CALUDE_taco_palace_bill_l3396_339660

theorem taco_palace_bill (mike_additional : ℝ) (john_additional : ℝ) 
  (h1 : mike_additional = 11.75)
  (h2 : john_additional = 5.25)
  (h3 : ∃ (taco_grande : ℝ), 
    taco_grande + mike_additional = 1.5 * (taco_grande + john_additional)) :
  ∃ (total_bill : ℝ), total_bill = 58.75 ∧ 
    total_bill = (taco_grande + mike_additional) + (taco_grande + john_additional) :=
by
  sorry

end NUMINAMATH_CALUDE_taco_palace_bill_l3396_339660


namespace NUMINAMATH_CALUDE_hotel_double_room_cost_l3396_339623

/-- Proves that the cost of each double room is $60 given the hotel booking information -/
theorem hotel_double_room_cost :
  let total_rooms : ℕ := 260
  let single_room_cost : ℕ := 35
  let total_revenue : ℕ := 14000
  let double_rooms : ℕ := 196
  let single_rooms : ℕ := total_rooms - double_rooms
  let double_room_cost : ℕ := (total_revenue - single_rooms * single_room_cost) / double_rooms
  double_room_cost = 60 := by
  sorry

#eval (14000 - (260 - 196) * 35) / 196  -- Should output 60

end NUMINAMATH_CALUDE_hotel_double_room_cost_l3396_339623


namespace NUMINAMATH_CALUDE_correct_number_l3396_339699

theorem correct_number (x : ℤ) (h1 : x - 152 = 346) : x + 152 = 650 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_l3396_339699


namespace NUMINAMATH_CALUDE_no_solution_for_lcm_gcd_equation_l3396_339645

theorem no_solution_for_lcm_gcd_equation : 
  ¬ ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.lcm n 60 = Nat.gcd n 60 + 200) ∧ 
    (Nat.Prime n) ∧ 
    (60 % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_lcm_gcd_equation_l3396_339645


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3396_339643

theorem quadratic_roots_problem (a b m p r : ℝ) : 
  (∀ x, x^2 - m*x + 4 = 0 ↔ x = a ∨ x = b) →
  (∀ x, x^2 - p*x + r = 0 ↔ x = a + 2/b ∨ x = b + 2/a) →
  r = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3396_339643


namespace NUMINAMATH_CALUDE_max_abs_z_given_distance_from_2i_l3396_339681

theorem max_abs_z_given_distance_from_2i (z : ℂ) : 
  Complex.abs (z - 2 * Complex.I) = 1 → Complex.abs z ≤ 3 ∧ ∃ w : ℂ, Complex.abs (w - 2 * Complex.I) = 1 ∧ Complex.abs w = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_z_given_distance_from_2i_l3396_339681


namespace NUMINAMATH_CALUDE_no_solution_for_modified_problem_l3396_339634

theorem no_solution_for_modified_problem (r : ℝ) : 
  ¬∃ (a h : ℝ), 
    (0 < r) ∧ 
    (0 < a) ∧ (a ≤ 2*r) ∧ 
    (0 < h) ∧ (h < 2*r) ∧ 
    (a + h = 2*Real.pi*r) := by
  sorry


end NUMINAMATH_CALUDE_no_solution_for_modified_problem_l3396_339634


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3396_339625

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α + π/4) = 3*Real.sqrt 2/5) : 
  Real.sin (2*α) = -11/25 := by
sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3396_339625


namespace NUMINAMATH_CALUDE_root_twice_other_iff_a_equals_four_l3396_339662

theorem root_twice_other_iff_a_equals_four (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - (2*a + 1)*x + a^2 + 2 = 0 ∧ 
    y^2 - (2*a + 1)*y + a^2 + 2 = 0 ∧ 
    y = 2*x) ↔ 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_root_twice_other_iff_a_equals_four_l3396_339662


namespace NUMINAMATH_CALUDE_largest_angle_in_18_sided_polygon_l3396_339678

theorem largest_angle_in_18_sided_polygon (n : ℕ) (sum_other_angles : ℝ) :
  n = 18 ∧ sum_other_angles = 2754 →
  (n - 2) * 180 - sum_other_angles = 126 :=
by sorry

end NUMINAMATH_CALUDE_largest_angle_in_18_sided_polygon_l3396_339678


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3396_339682

/-- Two points are symmetric about the y-axis if their x-coordinates are opposite and their y-coordinates are equal -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis a 3 4 b → (a + b)^2008 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3396_339682


namespace NUMINAMATH_CALUDE_gravel_path_cost_l3396_339640

/-- Calculates the cost of gravelling a path inside a rectangular plot. -/
theorem gravel_path_cost
  (plot_length : ℝ)
  (plot_width : ℝ)
  (path_width : ℝ)
  (cost_per_sqm : ℝ)
  (h1 : plot_length = 110)
  (h2 : plot_width = 65)
  (h3 : path_width = 2.5)
  (h4 : cost_per_sqm = 0.70) :
  let total_area := plot_length * plot_width
  let inner_length := plot_length - 2 * path_width
  let inner_width := plot_width - 2 * path_width
  let inner_area := inner_length * inner_width
  let path_area := total_area - inner_area
  path_area * cost_per_sqm = 595 :=
by sorry

end NUMINAMATH_CALUDE_gravel_path_cost_l3396_339640


namespace NUMINAMATH_CALUDE_min_width_rectangle_l3396_339626

/-- Given a rectangular area with length 20 ft longer than the width,
    and an area of at least 150 sq. ft, the minimum possible width is 10 ft. -/
theorem min_width_rectangle (w : ℝ) (h1 : w > 0) : 
  w * (w + 20) ≥ 150 → w ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_width_rectangle_l3396_339626


namespace NUMINAMATH_CALUDE_turnip_potato_ratio_l3396_339663

/-- Given a ratio of potatoes to turnips and a new amount of potatoes, 
    calculate the amount of turnips that maintains the same ratio -/
def calculate_turnips (potato_ratio : ℚ) (turnip_ratio : ℚ) (new_potato : ℚ) : ℚ :=
  (new_potato * turnip_ratio) / potato_ratio

/-- Prove that given the initial ratio of 5 cups of potatoes to 2 cups of turnips,
    the amount of turnips that can be mixed with 20 cups of potatoes while 
    maintaining the same ratio is 8 cups -/
theorem turnip_potato_ratio : 
  let initial_potato : ℚ := 5
  let initial_turnip : ℚ := 2
  let new_potato : ℚ := 20
  calculate_turnips initial_potato initial_turnip new_potato = 8 := by
  sorry

end NUMINAMATH_CALUDE_turnip_potato_ratio_l3396_339663


namespace NUMINAMATH_CALUDE_age_problem_l3396_339642

theorem age_problem (a b c : ℕ) : 
  (4 * a + b = 3 * c) →
  (3 * c^3 = 4 * a^3 + b^3) →
  (Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd a c = 1) →
  (a^2 + b^2 + c^2 = 35) :=
by sorry

end NUMINAMATH_CALUDE_age_problem_l3396_339642


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3396_339647

open Real

theorem necessary_but_not_sufficient 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) : 
  (a > b ∧ b > ℯ → a^b < b^a) ∧
  ¬(a^b < b^a → a > b ∧ b > ℯ) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l3396_339647


namespace NUMINAMATH_CALUDE_car_traveler_speed_ratio_l3396_339611

/-- Represents the bridge in the problem -/
structure Bridge where
  length : ℝ
  mk_pos : length > 0

/-- Represents the traveler in the problem -/
structure Traveler where
  speed : ℝ
  mk_pos : speed > 0

/-- Represents the car in the problem -/
structure Car where
  speed : ℝ
  mk_pos : speed > 0

/-- The main theorem stating the ratio of car speed to traveler speed -/
theorem car_traveler_speed_ratio (b : Bridge) (t : Traveler) (c : Car) :
  (t.speed * (4 / 9) * b.length / t.speed = c.speed * (4 / 9) * b.length / c.speed) →
  (t.speed * (5 / 9) * b.length / t.speed = b.length / c.speed) →
  c.speed / t.speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_car_traveler_speed_ratio_l3396_339611


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3396_339664

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 4) * (Real.sqrt 5 / Real.sqrt 6) * (Real.sqrt 8 / Real.sqrt 9) = Real.sqrt 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3396_339664


namespace NUMINAMATH_CALUDE_item_sale_ratio_l3396_339667

theorem item_sale_ratio (c x y : ℝ) (hx : x = 0.8 * c) (hy : y = 1.25 * c) : y / x = 25 / 16 := by
  sorry

end NUMINAMATH_CALUDE_item_sale_ratio_l3396_339667


namespace NUMINAMATH_CALUDE_right_triangles_on_circle_l3396_339617

theorem right_triangles_on_circle (n : ℕ) (h : n = 100) :
  ¬ (∃ (k : ℕ), k = 1000 ∧ k = (n / 2) * (n - 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangles_on_circle_l3396_339617


namespace NUMINAMATH_CALUDE_completing_square_solution_l3396_339612

theorem completing_square_solution (x : ℝ) : 
  (x^2 + 4*x + 1 = 0) ↔ ((x + 2)^2 = 3) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_solution_l3396_339612


namespace NUMINAMATH_CALUDE_ninth_term_is_zero_l3396_339689

/-- An arithmetic sequence with a₄ = 5 and a₅ = 4 -/
def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d ∧ a 4 = 5 ∧ a 5 = 4

theorem ninth_term_is_zero (a : ℕ → ℤ) (h : arithmeticSequence a) : a 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ninth_term_is_zero_l3396_339689


namespace NUMINAMATH_CALUDE_quadratic_point_relation_l3396_339653

/-- The quadratic function f(x) = x^2 + x + 1 -/
def f (x : ℝ) : ℝ := x^2 + x + 1

theorem quadratic_point_relation :
  let y₁ := f (-3)
  let y₂ := f 2
  let y₃ := f (1/2)
  y₃ < y₁ ∧ y₁ = y₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_point_relation_l3396_339653


namespace NUMINAMATH_CALUDE_four_person_four_office_assignment_l3396_339688

def number_of_assignments (n : ℕ) : ℕ := n.factorial

theorem four_person_four_office_assignment :
  number_of_assignments 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_four_person_four_office_assignment_l3396_339688


namespace NUMINAMATH_CALUDE_largest_negative_integer_negation_l3396_339658

theorem largest_negative_integer_negation (x : ℤ) : 
  (∀ y : ℤ, y < 0 → y ≤ x) ∧ x < 0 → -(-(-x)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_largest_negative_integer_negation_l3396_339658


namespace NUMINAMATH_CALUDE_randy_lunch_cost_l3396_339644

theorem randy_lunch_cost (initial_amount : ℝ) (remaining_amount : ℝ) (lunch_cost : ℝ) :
  initial_amount = 30 →
  remaining_amount = 15 →
  remaining_amount = initial_amount - lunch_cost - (1/4) * (initial_amount - lunch_cost) →
  lunch_cost = 10 := by
  sorry

end NUMINAMATH_CALUDE_randy_lunch_cost_l3396_339644


namespace NUMINAMATH_CALUDE_unique_m_existence_l3396_339630

theorem unique_m_existence : ∃! m : ℤ,
  50 ≤ m ∧ m ≤ 180 ∧
  m % 9 = 0 ∧
  m % 10 = 7 ∧
  m % 7 = 5 ∧
  m = 117 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_existence_l3396_339630
