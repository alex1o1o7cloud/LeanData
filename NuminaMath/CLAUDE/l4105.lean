import Mathlib

namespace NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l4105_410538

theorem max_sum_given_sum_of_squares_and_product (x y : ℝ) :
  x^2 + y^2 = 130 → xy = 45 → x + y ≤ 10 * Real.sqrt 2.2 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_given_sum_of_squares_and_product_l4105_410538


namespace NUMINAMATH_CALUDE_javier_children_count_l4105_410516

/-- The number of children in Javier's household -/
def num_children : ℕ := 
  let total_legs : ℕ := 22
  let javier_wife_legs : ℕ := 2 + 2
  let dog_legs : ℕ := 2 * 4
  let cat_legs : ℕ := 1 * 4
  let remaining_legs : ℕ := total_legs - (javier_wife_legs + dog_legs + cat_legs)
  remaining_legs / 2

theorem javier_children_count : num_children = 3 := by
  sorry

end NUMINAMATH_CALUDE_javier_children_count_l4105_410516


namespace NUMINAMATH_CALUDE_sin_cos_sum_zero_l4105_410501

theorem sin_cos_sum_zero : 
  Real.sin (35 * π / 6) + Real.cos (-11 * π / 3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_zero_l4105_410501


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l4105_410502

/-- Represents a four-digit number in the form 28a3 where a is a digit -/
def fourDigitNumber (a : ℕ) : ℕ := 2803 + 100 * a

/-- The denominator of the original fraction -/
def denominator : ℕ := 7276

/-- Theorem stating that 641 is the solution to the fraction equation -/
theorem fraction_equation_solution :
  ∃ (a : ℕ), a < 10 ∧ 
  (fourDigitNumber a - 641) * 7 = 2 * (denominator + 641) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l4105_410502


namespace NUMINAMATH_CALUDE_complex_division_l4105_410555

/-- Given complex numbers z₁ and z₂ corresponding to points (1, -1) and (-2, 1) in the complex plane,
    prove that z₂/z₁ = -3/2 - 1/2i. -/
theorem complex_division (z₁ z₂ : ℂ) (h₁ : z₁ = Complex.mk 1 (-1)) (h₂ : z₂ = Complex.mk (-2) 1) :
  z₂ / z₁ = Complex.mk (-3/2) (-1/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l4105_410555


namespace NUMINAMATH_CALUDE_average_marks_combined_l4105_410543

theorem average_marks_combined (n1 n2 : ℕ) (avg1 avg2 : ℚ) :
  n1 = 30 →
  n2 = 50 →
  avg1 = 40 →
  avg2 = 70 →
  (n1 * avg1 + n2 * avg2) / (n1 + n2 : ℚ) = 58.75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_combined_l4105_410543


namespace NUMINAMATH_CALUDE_hidden_primes_average_l4105_410563

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def card_sum (a b : ℕ) : ℕ := a + b

theorem hidden_primes_average (p₁ p₂ p₃ : ℕ) 
  (h₁ : is_prime p₁) (h₂ : is_prime p₂) (h₃ : is_prime p₃)
  (h₄ : card_sum p₁ 51 = card_sum p₂ 72)
  (h₅ : card_sum p₂ 72 = card_sum p₃ 43)
  (h₆ : p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃)
  (h₇ : p₁ ≠ 51 ∧ p₂ ≠ 72 ∧ p₃ ≠ 43) :
  (p₁ + p₂ + p₃) / 3 = 56 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l4105_410563


namespace NUMINAMATH_CALUDE_max_product_constraint_l4105_410582

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (hsum : 3 * a + 2 * b = 1) :
  a * b ≤ 1 / 24 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ a₀ * b₀ = 1 / 24 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l4105_410582


namespace NUMINAMATH_CALUDE_caitlin_bracelets_l4105_410522

/-- The number of bracelets Caitlin can make -/
def num_bracelets : ℕ := 11

/-- The total number of beads Caitlin has -/
def total_beads : ℕ := 528

/-- The number of large beads per bracelet -/
def large_beads_per_bracelet : ℕ := 12

/-- The ratio of small beads to large beads in each bracelet -/
def small_to_large_ratio : ℕ := 2

theorem caitlin_bracelets :
  (total_beads / 2) / (large_beads_per_bracelet * small_to_large_ratio) = num_bracelets :=
sorry

end NUMINAMATH_CALUDE_caitlin_bracelets_l4105_410522


namespace NUMINAMATH_CALUDE_hiking_problem_l4105_410578

/-- Hiking problem -/
theorem hiking_problem (R_up : ℝ) (R_down : ℝ) (T_up : ℝ) (T_down : ℝ) (D_down : ℝ) :
  R_up = 7 →
  R_down = 1.5 * R_up →
  T_up = T_down →
  D_down = 21 →
  T_up = 2 :=
by sorry

end NUMINAMATH_CALUDE_hiking_problem_l4105_410578


namespace NUMINAMATH_CALUDE_inequality_solution_set_l4105_410509

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l4105_410509


namespace NUMINAMATH_CALUDE_factor_calculation_l4105_410573

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 5 → 
  factor * (2 * initial_number + 15) = 75 → 
  factor = 3 := by sorry

end NUMINAMATH_CALUDE_factor_calculation_l4105_410573


namespace NUMINAMATH_CALUDE_solve_for_y_l4105_410542

theorem solve_for_y (x y : ℤ) (h1 : x^2 + 3*x + 6 = y - 2) (h2 : x = -5) : y = 18 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l4105_410542


namespace NUMINAMATH_CALUDE_integer_x_is_seven_l4105_410588

theorem integer_x_is_seven (x : ℤ) 
  (h1 : 3 < x ∧ x < 10)
  (h2 : 5 < x ∧ x < 18)
  (h3 : 9 > x ∧ x > 6)
  (h4 : 8 > x ∧ x > 0)
  (h5 : x + 1 < 9) :
  x = 7 := by
  sorry

end NUMINAMATH_CALUDE_integer_x_is_seven_l4105_410588


namespace NUMINAMATH_CALUDE_veranda_area_l4105_410505

/-- Given a rectangular room with length 17 m and width 12 m, surrounded by a veranda of width 2 m on all sides, the area of the veranda is 132 m². -/
theorem veranda_area (room_length : ℝ) (room_width : ℝ) (veranda_width : ℝ) :
  room_length = 17 →
  room_width = 12 →
  veranda_width = 2 →
  (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width = 132 :=
by sorry

end NUMINAMATH_CALUDE_veranda_area_l4105_410505


namespace NUMINAMATH_CALUDE_water_level_rise_l4105_410595

/-- Given a cube and a rectangular vessel, calculate the rise in water level when the cube is fully immersed. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) (h_cube : cube_edge = 12) 
    (h_length : vessel_length = 20) (h_width : vessel_width = 15) : 
    (cube_edge ^ 3) / (vessel_length * vessel_width) = 5.76 := by
  sorry

end NUMINAMATH_CALUDE_water_level_rise_l4105_410595


namespace NUMINAMATH_CALUDE_red_mailbox_houses_l4105_410541

/-- Proves the number of houses with red mailboxes given the total junk mail,
    total houses, houses with white mailboxes, and junk mail per house. -/
theorem red_mailbox_houses
  (total_junk_mail : ℕ)
  (total_houses : ℕ)
  (white_mailbox_houses : ℕ)
  (junk_mail_per_house : ℕ)
  (h1 : total_junk_mail = 48)
  (h2 : total_houses = 8)
  (h3 : white_mailbox_houses = 2)
  (h4 : junk_mail_per_house = 6)
  : total_houses - white_mailbox_houses = 6 := by
  sorry

#check red_mailbox_houses

end NUMINAMATH_CALUDE_red_mailbox_houses_l4105_410541


namespace NUMINAMATH_CALUDE_unique_three_digit_cube_divisible_by_8_and_9_l4105_410532

/-- A number is a three-digit number if it's between 100 and 999 inclusive -/
def IsThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- A number is a perfect cube if it's the cube of some integer -/
def IsPerfectCube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

/-- Main theorem: There exists exactly one three-digit number that is a perfect cube 
    and divisible by both 8 and 9 -/
theorem unique_three_digit_cube_divisible_by_8_and_9 : 
  ∃! n : ℕ, IsThreeDigit n ∧ IsPerfectCube n ∧ n % 8 = 0 ∧ n % 9 = 0 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_cube_divisible_by_8_and_9_l4105_410532


namespace NUMINAMATH_CALUDE_slope_of_line_l4105_410593

theorem slope_of_line (x y : ℝ) :
  4 * x - 7 * y = 28 → (y - (-4)) / (x - 0) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_slope_of_line_l4105_410593


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l4105_410580

theorem unique_x_with_three_prime_factors (x n : ℕ) : 
  x = 7^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 :=
by sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_factors_l4105_410580


namespace NUMINAMATH_CALUDE_triangles_in_decagon_l4105_410599

/-- The number of triangles that can be formed using the vertices of a regular decagon -/
def numTrianglesInDecagon : ℕ := 120

/-- A regular decagon has 10 vertices -/
def numVerticesInDecagon : ℕ := 10

theorem triangles_in_decagon :
  numTrianglesInDecagon = (numVerticesInDecagon.choose 3) := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_l4105_410599


namespace NUMINAMATH_CALUDE_commission_change_point_l4105_410590

/-- The sales amount where the commission rate changes -/
def X : ℝ := 10000

/-- The total sales amount -/
def total_sales : ℝ := 32500

/-- The amount remitted to the parent company -/
def remitted_amount : ℝ := 31100

/-- The commission rate for sales up to X -/
def commission_rate_low : ℝ := 0.05

/-- The commission rate for sales exceeding X -/
def commission_rate_high : ℝ := 0.04

theorem commission_change_point :
  X = 10000 ∧
  total_sales = 32500 ∧
  remitted_amount = 31100 ∧
  commission_rate_low = 0.05 ∧
  commission_rate_high = 0.04 ∧
  remitted_amount = total_sales - (commission_rate_low * X + commission_rate_high * (total_sales - X)) :=
by sorry

end NUMINAMATH_CALUDE_commission_change_point_l4105_410590


namespace NUMINAMATH_CALUDE_exam_pass_probability_l4105_410564

theorem exam_pass_probability (p_A p_B p_C : ℚ) 
  (h_A : p_A = 2/3) 
  (h_B : p_B = 3/4) 
  (h_C : p_C = 2/5) : 
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_exam_pass_probability_l4105_410564


namespace NUMINAMATH_CALUDE_greatest_n_for_2008_l4105_410507

-- Define the sum of digits function
def sum_of_digits (a : ℕ) : ℕ := sorry

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => sorry  -- Initial value, not specified in the problem
  | n + 1 => a n + sum_of_digits (a n)

-- Theorem statement
theorem greatest_n_for_2008 : (∃ n : ℕ, a n = 2008) ∧ (∀ m : ℕ, m > 6 → a m ≠ 2008) := by sorry

end NUMINAMATH_CALUDE_greatest_n_for_2008_l4105_410507


namespace NUMINAMATH_CALUDE_percentage_and_absolute_difference_l4105_410561

/-- Given two initial values and an annual percentage increase, 
    calculate the percentage difference and the absolute difference after 5 years. -/
theorem percentage_and_absolute_difference 
  (initial_value1 : ℝ) 
  (initial_value2 : ℝ) 
  (annual_increase : ℝ) 
  (h1 : initial_value1 = 0.60 * 5000) 
  (h2 : initial_value2 = 0.42 * 3000) :
  let difference := initial_value1 - initial_value2
  let percentage_difference := (difference / initial_value1) * 100
  let new_difference := difference * (1 + annual_increase / 100) ^ 5
  percentage_difference = 58 ∧ 
  new_difference = 1740 * (1 + annual_increase / 100) ^ 5 := by
sorry

end NUMINAMATH_CALUDE_percentage_and_absolute_difference_l4105_410561


namespace NUMINAMATH_CALUDE_barn_size_calculation_barn_size_is_1000_l4105_410553

/-- Given a property with a house and a barn, calculate the size of the barn. -/
theorem barn_size_calculation (price_per_sqft : ℝ) (house_size : ℝ) (total_value : ℝ) : ℝ :=
  let house_value := price_per_sqft * house_size
  let barn_value := total_value - house_value
  barn_value / price_per_sqft

/-- The size of the barn is 1000 square feet. -/
theorem barn_size_is_1000 :
  barn_size_calculation 98 2400 333200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_barn_size_calculation_barn_size_is_1000_l4105_410553


namespace NUMINAMATH_CALUDE_average_of_four_numbers_l4105_410584

theorem average_of_four_numbers (n : ℝ) :
  (3 + 16 + 33 + (n + 1)) / 4 = 20 → n = 27 := by
  sorry

end NUMINAMATH_CALUDE_average_of_four_numbers_l4105_410584


namespace NUMINAMATH_CALUDE_jane_ribbons_per_dress_l4105_410549

/-- The number of ribbons Jane adds to each dress --/
def ribbons_per_dress (dresses_first_week : ℕ) (dresses_second_week : ℕ) (total_ribbons : ℕ) : ℚ :=
  total_ribbons / (dresses_first_week + dresses_second_week)

/-- Theorem stating that Jane adds 2 ribbons to each dress --/
theorem jane_ribbons_per_dress :
  ribbons_per_dress (7 * 2) (2 * 3) 40 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jane_ribbons_per_dress_l4105_410549


namespace NUMINAMATH_CALUDE_equation_solution_l4105_410545

theorem equation_solution (y : ℚ) : 
  (1 : ℚ) / 3 + 1 / y = 7 / 9 → y = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4105_410545


namespace NUMINAMATH_CALUDE_complement_intersection_equals_d_l4105_410581

-- Define the universe
def U : Set Char := {'a', 'b', 'c', 'd', 'e'}

-- Define sets M and N
def M : Set Char := {'a', 'b', 'c'}
def N : Set Char := {'a', 'c', 'e'}

-- State the theorem
theorem complement_intersection_equals_d :
  (U \ M) ∩ (U \ N) = {'d'} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_d_l4105_410581


namespace NUMINAMATH_CALUDE_function_properties_l4105_410510

/-- Given two real numbers p1 and p2, where p1 ≠ p2, we define two functions f and g. -/
theorem function_properties (p1 p2 : ℝ) (h : p1 ≠ p2) :
  let f := fun x : ℝ => (3 : ℝ) ^ (|x - p1|)
  let g := fun x : ℝ => (3 : ℝ) ^ (|x - p2|)
  -- 1. f can be obtained by translating g
  (∃ k : ℝ, ∀ x : ℝ, f x = g (x + k)) ∧
  -- 2. f + g is symmetric about x = (p1 + p2) / 2
  (∀ x : ℝ, f x + g x = f (p1 + p2 - x) + g (p1 + p2 - x)) ∧
  -- 3. f - g is symmetric about the point ((p1 + p2) / 2, 0)
  (∀ x : ℝ, f x - g x = -(f (p1 + p2 - x) - g (p1 + p2 - x))) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l4105_410510


namespace NUMINAMATH_CALUDE_frequency_20_plus_l4105_410508

-- Define the sample size
def sample_size : ℕ := 35

-- Define the frequencies for each interval
def freq_5_10 : ℕ := 5
def freq_10_15 : ℕ := 12
def freq_15_20 : ℕ := 7
def freq_20_25 : ℕ := 5
def freq_25_30 : ℕ := 4
def freq_30_35 : ℕ := 2

-- Theorem to prove
theorem frequency_20_plus (h : freq_5_10 + freq_10_15 + freq_15_20 + freq_20_25 + freq_25_30 + freq_30_35 = sample_size) :
  (freq_20_25 + freq_25_30 + freq_30_35 : ℚ) / sample_size = 11 / 35 := by
  sorry

end NUMINAMATH_CALUDE_frequency_20_plus_l4105_410508


namespace NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l4105_410552

/-- Charlie's schedule cycle length -/
def charlie_cycle : Nat := 6

/-- Dana's schedule cycle length -/
def dana_cycle : Nat := 10

/-- Number of days in the period -/
def total_days : Nat := 1200

/-- Number of rest days in Charlie's cycle -/
def charlie_rest_days : Nat := 2

/-- Number of rest days in Dana's cycle -/
def dana_rest_days : Nat := 1

/-- Function to calculate the number of coinciding rest days -/
def coinciding_rest_days (charlie_cycle dana_cycle total_days : Nat) : Nat :=
  sorry

theorem coinciding_rest_days_theorem :
  coinciding_rest_days charlie_cycle dana_cycle total_days = 40 := by
  sorry

end NUMINAMATH_CALUDE_coinciding_rest_days_theorem_l4105_410552


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l4105_410571

theorem rectangle_measurement_error (L W : ℝ) (p : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.05 * L) * ((1 - p) * W) = (1 + 0.008) * (L * W) → p = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l4105_410571


namespace NUMINAMATH_CALUDE_weight_distribution_l4105_410594

theorem weight_distribution :
  ∀ x y z : ℕ,
  x + y + z = 100 →
  x + 10 * y + 50 * z = 500 →
  x = 60 ∧ y = 39 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_weight_distribution_l4105_410594


namespace NUMINAMATH_CALUDE_sin_equation_range_l4105_410566

theorem sin_equation_range : 
  let f : ℝ → ℝ := λ x => Real.sin x ^ 2 - 2 * Real.sin x
  ∃ (a_min a_max : ℝ), a_min = -1 ∧ a_max = 3 ∧
    (∀ a : ℝ, (∃ x : ℝ, f x = a) ↔ a_min ≤ a ∧ a ≤ a_max) :=
by sorry

end NUMINAMATH_CALUDE_sin_equation_range_l4105_410566


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_99999_l4105_410598

theorem largest_n_divisible_by_seven (n : ℕ) : n < 100000 →
  (10 * (n - 3)^5 - n^2 + 20 * n - 30) % 7 = 0 →
  n ≤ 99999 :=
by sorry

theorem largest_n_is_99999 :
  (10 * (99999 - 3)^5 - 99999^2 + 20 * 99999 - 30) % 7 = 0 ∧
  ∀ m : ℕ, m > 99999 → m < 100000 →
    (10 * (m - 3)^5 - m^2 + 20 * m - 30) % 7 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_99999_l4105_410598


namespace NUMINAMATH_CALUDE_expression_evaluation_l4105_410562

theorem expression_evaluation : 3 - (5 : ℝ)^(3-3) = 2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4105_410562


namespace NUMINAMATH_CALUDE_dylan_ice_cubes_l4105_410530

-- Define the number of ice cubes in Dylan's glass
def ice_cubes_in_glass : ℕ := sorry

-- Define the number of ice cubes in the pitcher
def ice_cubes_in_pitcher : ℕ := 2 * ice_cubes_in_glass

-- Define the total number of ice cubes used
def total_ice_cubes : ℕ := ice_cubes_in_glass + ice_cubes_in_pitcher

-- Define the capacity of one tray
def tray_capacity : ℕ := 12

-- Define the number of trays
def number_of_trays : ℕ := 2

-- Theorem to prove
theorem dylan_ice_cubes : 
  ice_cubes_in_glass = 8 ∧ 
  total_ice_cubes = number_of_trays * tray_capacity :=
sorry

end NUMINAMATH_CALUDE_dylan_ice_cubes_l4105_410530


namespace NUMINAMATH_CALUDE_tent_production_equation_l4105_410576

theorem tent_production_equation (x : ℝ) (h : x > 0) : 
  (7000 / x) - (7000 / (1.4 * x)) = 4 ↔ 
  ∃ (original_days actual_days : ℝ),
    original_days > 0 ∧ 
    actual_days > 0 ∧
    original_days = 7000 / x ∧ 
    actual_days = 7000 / (1.4 * x) ∧
    original_days - actual_days = 4 :=
by sorry

end NUMINAMATH_CALUDE_tent_production_equation_l4105_410576


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4105_410504

theorem min_value_reciprocal_sum (t q r₁ r₂ : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = r₁ ∨ x = r₂) →
  r₁ + r₂ = r₁^2 + r₂^2 →
  r₁^2 + r₂^2 = r₁^4 + r₂^4 →
  ∃ (min : ℝ), min = 2 ∧ ∀ (s t : ℝ), 
    (∀ x, x^2 - s*x + t = 0 ↔ x = r₁ ∨ x = r₂) →
    r₁ + r₂ = r₁^2 + r₂^2 →
    r₁^2 + r₂^2 = r₁^4 + r₂^4 →
    min ≤ 1/r₁^5 + 1/r₂^5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l4105_410504


namespace NUMINAMATH_CALUDE_min_value_a_l4105_410551

theorem min_value_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → 2 * a * Real.exp (2 * x) - Real.log x + Real.log a ≥ 0) →
  a ≥ 1 / (2 * Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l4105_410551


namespace NUMINAMATH_CALUDE_equation_solution_l4105_410528

theorem equation_solution (x : ℝ) : (x - 2) * (x - 3) = 0 ↔ x = 2 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4105_410528


namespace NUMINAMATH_CALUDE_two_part_journey_average_speed_l4105_410575

/-- Calculates the average speed for a two-part journey -/
theorem two_part_journey_average_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (second_part_speed : ℝ)
  (h1 : total_distance = 60)
  (h2 : first_part_distance = 12)
  (h3 : first_part_speed = 24)
  (h4 : second_part_speed = 48)
  : (total_distance / (first_part_distance / first_part_speed +
     (total_distance - first_part_distance) / second_part_speed)) = 40 := by
  sorry

#check two_part_journey_average_speed

end NUMINAMATH_CALUDE_two_part_journey_average_speed_l4105_410575


namespace NUMINAMATH_CALUDE_investment_ratio_equals_profit_ratio_l4105_410506

/-- Given two investors A and B with equal investment periods, 
    this theorem proves that their investment ratio is equal to their profit ratio. -/
theorem investment_ratio_equals_profit_ratio 
  (profit_A profit_B : ℕ) 
  (h1 : profit_A = 60000) 
  (h2 : profit_B = 6000) : 
  (profit_A : ℚ) / (profit_B : ℚ) = 10 / 1 := by
  sorry

#check investment_ratio_equals_profit_ratio

end NUMINAMATH_CALUDE_investment_ratio_equals_profit_ratio_l4105_410506


namespace NUMINAMATH_CALUDE_inequality_solution_l4105_410572

theorem inequality_solution : 
  ∃! a : ℝ, a > 0 ∧ ∀ x > 0, (2 * x - 2 * a + Real.log (x / a)) * (-2 * x^2 + a * x + 5) ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l4105_410572


namespace NUMINAMATH_CALUDE_exam_score_problem_l4105_410519

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 50)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 130) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 36 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l4105_410519


namespace NUMINAMATH_CALUDE_inequality_range_l4105_410550

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x - (a + 2) < 0) ↔ (-1 < a ∧ a ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l4105_410550


namespace NUMINAMATH_CALUDE_engine_system_theorems_l4105_410520

/-- Engine connecting rod and crank system -/
structure EngineSystem where
  a : ℝ  -- length of crank OA
  b : ℝ  -- length of connecting rod AP
  α : ℝ  -- angle AOP
  β : ℝ  -- angle APO
  x : ℝ  -- length of PQ

/-- Theorems about the engine connecting rod and crank system -/
theorem engine_system_theorems (sys : EngineSystem) :
  -- 1. Sine rule relation
  sys.a * Real.sin sys.α = sys.b * Real.sin sys.β ∧
  -- 2. Maximum value of sin β
  (∃ (max_sin_β : ℝ), max_sin_β = sys.a / sys.b ∧
    ∀ β', Real.sin β' ≤ max_sin_β) ∧
  -- 3. Relation for x
  sys.x = sys.a * (1 - Real.cos sys.α) + sys.b * (1 - Real.cos sys.β) :=
by sorry

end NUMINAMATH_CALUDE_engine_system_theorems_l4105_410520


namespace NUMINAMATH_CALUDE_bowling_team_weight_problem_l4105_410531

theorem bowling_team_weight_problem (original_players : ℕ) (original_avg_weight : ℝ)
  (new_players : ℕ) (new_avg_weight : ℝ) (known_new_player_weight : ℝ) :
  original_players = 7 →
  original_avg_weight = 121 →
  new_players = 2 →
  new_avg_weight = 113 →
  known_new_player_weight = 60 →
  ∃ x : ℝ,
    (original_players * original_avg_weight + known_new_player_weight + x) /
      (original_players + new_players) = new_avg_weight ∧
    x = 110 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_weight_problem_l4105_410531


namespace NUMINAMATH_CALUDE_basketball_game_difference_l4105_410557

/-- Given a ratio of boys to girls and the number of girls, 
    calculate the difference between the number of boys and girls -/
def boys_girls_difference (boys_ratio : ℕ) (girls_ratio : ℕ) (num_girls : ℕ) : ℕ :=
  let num_boys := (num_girls / girls_ratio) * boys_ratio
  num_boys - num_girls

/-- Theorem stating that with a ratio of 8:5 boys to girls and 30 girls, 
    there are 18 more boys than girls -/
theorem basketball_game_difference : boys_girls_difference 8 5 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_difference_l4105_410557


namespace NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l4105_410514

theorem prime_divisors_of_50_factorial (n : ℕ) : n = 50 → 
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card = 15 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l4105_410514


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l4105_410524

/-- Given a rectangle where the length is twice the width and the perimeter in inches
    equals the area in square inches, prove that the width is 3 inches and the length is 6 inches. -/
theorem rectangle_dimensions (w : ℝ) (h1 : w > 0) :
  (6 * w = 2 * w^2) → (w = 3 ∧ 2 * w = 6) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l4105_410524


namespace NUMINAMATH_CALUDE_pencils_per_pack_l4105_410548

theorem pencils_per_pack (num_packs : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  num_packs = 35 → num_rows = 70 → pencils_per_row = 2 → 
  (num_rows * pencils_per_row) / num_packs = 4 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_pack_l4105_410548


namespace NUMINAMATH_CALUDE_fashion_show_evening_wear_correct_evening_wear_count_l4105_410515

theorem fashion_show_evening_wear (num_models : ℕ) (bathing_suits_per_model : ℕ) 
  (runway_time : ℕ) (total_show_time : ℕ) : ℕ :=
  let total_bathing_suit_trips := num_models * bathing_suits_per_model
  let bathing_suit_time := total_bathing_suit_trips * runway_time
  let evening_wear_time := total_show_time - bathing_suit_time
  let evening_wear_trips := evening_wear_time / runway_time
  evening_wear_trips / num_models

theorem correct_evening_wear_count : 
  fashion_show_evening_wear 6 2 2 60 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fashion_show_evening_wear_correct_evening_wear_count_l4105_410515


namespace NUMINAMATH_CALUDE_sandcastle_heights_sum_l4105_410554

/-- Represents the height of a sandcastle in feet and fractions of a foot -/
structure SandcastleHeight where
  whole : ℕ
  numerator : ℕ
  denominator : ℕ

/-- Calculates the total height of four sandcastles -/
def total_height (janet : SandcastleHeight) (sister : SandcastleHeight) 
                 (tom : SandcastleHeight) (lucy : SandcastleHeight) : ℚ :=
  (janet.whole : ℚ) + (janet.numerator : ℚ) / (janet.denominator : ℚ) +
  (sister.whole : ℚ) + (sister.numerator : ℚ) / (sister.denominator : ℚ) +
  (tom.whole : ℚ) + (tom.numerator : ℚ) / (tom.denominator : ℚ) +
  (lucy.whole : ℚ) + (lucy.numerator : ℚ) / (lucy.denominator : ℚ)

theorem sandcastle_heights_sum :
  let janet := SandcastleHeight.mk 3 5 6
  let sister := SandcastleHeight.mk 2 7 12
  let tom := SandcastleHeight.mk 1 11 20
  let lucy := SandcastleHeight.mk 2 13 24
  total_height janet sister tom lucy = 10 + 61 / 120 := by sorry

end NUMINAMATH_CALUDE_sandcastle_heights_sum_l4105_410554


namespace NUMINAMATH_CALUDE_jeff_rental_duration_l4105_410534

/-- Represents the rental scenario for Jeff's apartment. -/
structure RentalScenario where
  initialRent : ℕ  -- Monthly rent for the first 3 years
  raisedRent : ℕ   -- Monthly rent after the raise
  initialYears : ℕ -- Number of years at the initial rent
  totalPaid : ℕ    -- Total amount paid over the entire rental period

/-- Calculates the total number of years Jeff rented the apartment. -/
def totalRentalYears (scenario : RentalScenario) : ℕ :=
  scenario.initialYears + 
  ((scenario.totalPaid - scenario.initialRent * scenario.initialYears * 12) / (scenario.raisedRent * 12))

/-- Theorem stating that Jeff rented the apartment for 5 years. -/
theorem jeff_rental_duration (scenario : RentalScenario) 
  (h1 : scenario.initialRent = 300)
  (h2 : scenario.raisedRent = 350)
  (h3 : scenario.initialYears = 3)
  (h4 : scenario.totalPaid = 19200) :
  totalRentalYears scenario = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeff_rental_duration_l4105_410534


namespace NUMINAMATH_CALUDE_no_three_color_solution_exists_seven_color_solution_l4105_410544

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an equilateral triangle
structure EqTriangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define a coloring function type
def ColoringFunction (n : ℕ) := ℝ × ℝ → Fin n

-- Define what it means for a circle to be contained in another circle
def containedIn (c1 c2 : Circle) : Prop :=
  (c1.radius + ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2).sqrt ≤ c2.radius)

-- Define what it means for a triangle to be inscribed in a circle
def inscribedIn (t : EqTriangle) (c : Circle) : Prop := sorry

-- Define what it means for a coloring to be good for a circle
def goodColoring (f : ColoringFunction n) (c : Circle) : Prop :=
  ∀ t : EqTriangle, inscribedIn t c → (f (t.vertices 0) ≠ f (t.vertices 1) ∧ 
                                       f (t.vertices 0) ≠ f (t.vertices 2) ∧ 
                                       f (t.vertices 1) ≠ f (t.vertices 2))

-- Main theorem for part 1
theorem no_three_color_solution :
  ¬ ∃ (f : ColoringFunction 3), 
    ∀ (c : Circle), 
      containedIn c (Circle.mk (0, 0) 2) → c.radius ≥ 1 → goodColoring f c :=
sorry

-- Main theorem for part 2
theorem exists_seven_color_solution :
  ∃ (g : ColoringFunction 7), 
    ∀ (c : Circle), 
      containedIn c (Circle.mk (0, 0) 2) → c.radius ≥ 1 → goodColoring g c :=
sorry

end NUMINAMATH_CALUDE_no_three_color_solution_exists_seven_color_solution_l4105_410544


namespace NUMINAMATH_CALUDE_fruit_store_total_weight_l4105_410533

theorem fruit_store_total_weight 
  (boxes_sold : ℕ) 
  (weight_per_box : ℕ) 
  (remaining_weight : ℕ) 
  (h1 : boxes_sold = 14)
  (h2 : weight_per_box = 30)
  (h3 : remaining_weight = 80) :
  boxes_sold * weight_per_box + remaining_weight = 500 := by
sorry

end NUMINAMATH_CALUDE_fruit_store_total_weight_l4105_410533


namespace NUMINAMATH_CALUDE_regular_ngon_minimal_l4105_410574

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define an n-gon
structure NGon where
  n : ℕ
  vertices : List (ℝ × ℝ)

-- Function to check if an n-gon is inscribed in a circle
def isInscribed (ngon : NGon) (circle : Circle) : Prop :=
  ngon.vertices.length = ngon.n ∧
  ∀ v ∈ ngon.vertices, (v.1 - circle.center.1)^2 + (v.2 - circle.center.2)^2 = circle.radius^2

-- Function to check if an n-gon is regular
def isRegular (ngon : NGon) : Prop :=
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    ∀ v ∈ ngon.vertices, (v.1 - center.1)^2 + (v.2 - center.2)^2 = radius^2

-- Function to calculate the area of an n-gon
noncomputable def area (ngon : NGon) : ℝ := sorry

-- Function to calculate the perimeter of an n-gon
noncomputable def perimeter (ngon : NGon) : ℝ := sorry

-- Theorem statement
theorem regular_ngon_minimal (circle : Circle) (n : ℕ) :
  ∀ (ngon : NGon), 
    ngon.n = n → 
    isInscribed ngon circle → 
    ∃ (regular_ngon : NGon), 
      regular_ngon.n = n ∧ 
      isInscribed regular_ngon circle ∧ 
      isRegular regular_ngon ∧ 
      area regular_ngon ≤ area ngon ∧ 
      perimeter regular_ngon ≤ perimeter ngon :=
by sorry

end NUMINAMATH_CALUDE_regular_ngon_minimal_l4105_410574


namespace NUMINAMATH_CALUDE_wire_division_l4105_410529

/-- Given a wire of length 28 cm divided into quarters, prove that each quarter is 7 cm long. -/
theorem wire_division (wire_length : ℝ) (h : wire_length = 28) :
  wire_length / 4 = 7 := by
sorry

end NUMINAMATH_CALUDE_wire_division_l4105_410529


namespace NUMINAMATH_CALUDE_x_n_prime_iff_n_eq_two_l4105_410587

/-- Definition of x_n as a number of the form 10101...1 with n ones -/
def x_n (n : ℕ) : ℕ := (10^(2*n) - 1) / 99

/-- Theorem stating that x_n is prime only when n = 2 -/
theorem x_n_prime_iff_n_eq_two :
  ∀ n : ℕ, Nat.Prime (x_n n) ↔ n = 2 :=
sorry

end NUMINAMATH_CALUDE_x_n_prime_iff_n_eq_two_l4105_410587


namespace NUMINAMATH_CALUDE_sample_represents_knowledge_l4105_410523

/-- Represents the population of teachers and students -/
def Population : ℕ := 1500

/-- Represents the sample size -/
def SampleSize : ℕ := 150

/-- Represents an individual in the population -/
structure Individual where
  id : ℕ
  isTeacher : Bool

/-- Represents the survey sample -/
structure Sample where
  individuals : Finset Individual
  size : ℕ
  h_size : size = SampleSize

/-- Represents the national security knowledge of an individual -/
def NationalSecurityKnowledge : Type := ℕ

/-- The theorem stating what the sample represents in the survey -/
theorem sample_represents_knowledge (sample : Sample) :
  ∃ (knowledge : Individual → NationalSecurityKnowledge),
    (∀ i ∈ sample.individuals, knowledge i ∈ Set.range knowledge) ∧
    (∀ i ∉ sample.individuals, knowledge i ∉ Set.range knowledge) :=
sorry

end NUMINAMATH_CALUDE_sample_represents_knowledge_l4105_410523


namespace NUMINAMATH_CALUDE_min_red_chips_is_76_l4105_410560

/-- Represents the number of chips of each color in the box -/
structure ChipCount where
  red : ℕ
  white : ℕ
  blue : ℕ

/-- Checks if the chip count satisfies the given conditions -/
def isValidChipCount (c : ChipCount) : Prop :=
  c.blue ≥ c.white / 3 ∧
  c.blue ≤ c.red / 4 ∧
  c.white + c.blue ≥ 75

/-- The minimum number of red chips that satisfies the conditions -/
def minRedChips : ℕ := 76

/-- Theorem stating that the minimum number of red chips is 76 -/
theorem min_red_chips_is_76 :
  ∀ c : ChipCount, isValidChipCount c → c.red ≥ minRedChips :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_is_76_l4105_410560


namespace NUMINAMATH_CALUDE_snack_distribution_l4105_410569

theorem snack_distribution (candies jellies students : ℕ) 
  (h1 : candies = 72)
  (h2 : jellies = 56)
  (h3 : students = 4) :
  (candies + jellies) / students = 32 :=
by sorry

end NUMINAMATH_CALUDE_snack_distribution_l4105_410569


namespace NUMINAMATH_CALUDE_min_sum_squares_l4105_410535

theorem min_sum_squares (x y : ℝ) :
  x^2 - y^2 + 6*x + 4*y + 5 = 0 →
  ∃ (min : ℝ), min = 0.5 ∧ ∀ (a b : ℝ), a^2 - b^2 + 6*a + 4*b + 5 = 0 → a^2 + b^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l4105_410535


namespace NUMINAMATH_CALUDE_distance_between_cities_l4105_410596

/-- The distance between two cities given the speeds of two travelers and their meeting point --/
theorem distance_between_cities (john_speed lewis_speed : ℝ) (meeting_point : ℝ) : 
  john_speed = 40 →
  lewis_speed = 60 →
  meeting_point = 160 →
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance + meeting_point) / lewis_speed = (distance - meeting_point) / john_speed ∧
    distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l4105_410596


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l4105_410556

theorem pie_crust_flour_calculation (total_flour : ℚ) (original_crusts new_crusts : ℕ) :
  total_flour > 0 →
  original_crusts > 0 →
  new_crusts > 0 →
  (total_flour / original_crusts) * new_crusts = total_flour →
  total_flour / new_crusts = 1 / 5 := by
  sorry

#check pie_crust_flour_calculation (5 : ℚ) 40 25

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l4105_410556


namespace NUMINAMATH_CALUDE_circle_no_intersection_with_axes_l4105_410591

theorem circle_no_intersection_with_axes (k : ℝ) :
  (k > 0) →
  (∀ x y : ℝ, x^2 + y^2 - 2*k*x + 2*y + 2 = 0 → (x ≠ 0 ∧ y ≠ 0)) →
  k > 1 ∧ k < Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_no_intersection_with_axes_l4105_410591


namespace NUMINAMATH_CALUDE_starting_lineup_with_twins_l4105_410536

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem starting_lineup_with_twins (total_players : ℕ) (lineup_size : ℕ) (twin_count : ℕ) :
  total_players = 12 →
  lineup_size = 5 →
  twin_count = 2 →
  choose (total_players - twin_count) (lineup_size - twin_count) = 120 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_with_twins_l4105_410536


namespace NUMINAMATH_CALUDE_polar_coordinate_equivalence_l4105_410592

def standard_polar_form (r : ℝ) (θ : ℝ) : Prop :=
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi

theorem polar_coordinate_equivalence :
  ∀ (r₁ r₂ θ₁ θ₂ : ℝ),
  r₁ = -3 ∧ θ₁ = 5 * Real.pi / 6 →
  r₂ = 3 ∧ θ₂ = 11 * Real.pi / 6 →
  standard_polar_form r₂ θ₂ →
  (r₁ * (Real.cos θ₁), r₁ * (Real.sin θ₁)) = (r₂ * (Real.cos θ₂), r₂ * (Real.sin θ₂)) :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinate_equivalence_l4105_410592


namespace NUMINAMATH_CALUDE_largest_corner_sum_l4105_410526

-- Define the face values of the cube
def face_values : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the property that opposite faces sum to 9
def opposite_sum_9 (faces : List ℕ) : Prop :=
  ∀ x ∈ faces, (9 - x) ∈ faces

-- Define a function to check if three numbers can be on adjacent faces
def can_be_adjacent (a b c : ℕ) : Prop :=
  a + b ≠ 9 ∧ b + c ≠ 9 ∧ a + c ≠ 9

-- Theorem statement
theorem largest_corner_sum :
  ∀ (cube : List ℕ),
  cube = face_values →
  opposite_sum_9 cube →
  (∃ (a b c : ℕ),
    a ∈ cube ∧ b ∈ cube ∧ c ∈ cube ∧
    can_be_adjacent a b c ∧
    (∀ (x y z : ℕ),
      x ∈ cube → y ∈ cube → z ∈ cube →
      can_be_adjacent x y z →
      x + y + z ≤ a + b + c)) →
  (∃ (a b c : ℕ),
    a ∈ cube ∧ b ∈ cube ∧ c ∈ cube ∧
    can_be_adjacent a b c ∧
    a + b + c = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_corner_sum_l4105_410526


namespace NUMINAMATH_CALUDE_algebraic_expressions_l4105_410577

theorem algebraic_expressions (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : (x - 2) * (y - 2) = -3) : 
  x * y = 3 ∧ x^2 + 4*x*y + y^2 = 31 ∧ x^2 + x*y + 5*y = 25 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expressions_l4105_410577


namespace NUMINAMATH_CALUDE_pencil_sharpener_time_l4105_410585

/-- Represents the time in minutes for which we're solving -/
def t : ℝ := 6

/-- Time (in seconds) for hand-crank sharpener to sharpen one pencil -/
def hand_crank_time : ℝ := 45

/-- Time (in seconds) for electric sharpener to sharpen one pencil -/
def electric_time : ℝ := 20

/-- The difference in number of pencils sharpened -/
def pencil_difference : ℕ := 10

theorem pencil_sharpener_time :
  (60 * t / electric_time) = (60 * t / hand_crank_time) + pencil_difference :=
sorry

end NUMINAMATH_CALUDE_pencil_sharpener_time_l4105_410585


namespace NUMINAMATH_CALUDE_line_x_intercept_l4105_410589

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def xIntercept (l : Line) : ℝ := sorry

/-- The theorem stating that the line through (6, 2) and (2, 6) has x-intercept at x = 8 -/
theorem line_x_intercept :
  let l : Line := { x₁ := 6, y₁ := 2, x₂ := 2, y₂ := 6 }
  xIntercept l = 8 := by sorry

end NUMINAMATH_CALUDE_line_x_intercept_l4105_410589


namespace NUMINAMATH_CALUDE_perfect_cubes_between_500_and_2000_l4105_410565

theorem perfect_cubes_between_500_and_2000 : 
  (Finset.filter (fun n => 500 ≤ n^3 ∧ n^3 ≤ 2000) (Finset.range 13)).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cubes_between_500_and_2000_l4105_410565


namespace NUMINAMATH_CALUDE_smallest_x_power_inequality_l4105_410568

theorem smallest_x_power_inequality : 
  ∃ x : ℕ, (∀ y : ℕ, 27^y > 3^24 → x ≤ y) ∧ 27^x > 3^24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_power_inequality_l4105_410568


namespace NUMINAMATH_CALUDE_expression_value_l4105_410537

theorem expression_value (x y : ℝ) (h : x^2 - 4*x + 4 + |y - 1| = 0) :
  (2*x - y)^2 - 2*(2*x - y)*(x + 2*y) + (x + 2*y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4105_410537


namespace NUMINAMATH_CALUDE_arithmetic_sequence_difference_l4105_410558

/-- An arithmetic sequence with the given property -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the common difference of the specific arithmetic sequence -/
theorem arithmetic_sequence_difference (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) (h2015 : a 2015 = a 2013 + 6) : 
    ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_difference_l4105_410558


namespace NUMINAMATH_CALUDE_sqrt_sixteen_times_sqrt_sixteen_equals_eight_l4105_410586

theorem sqrt_sixteen_times_sqrt_sixteen_equals_eight : Real.sqrt (16 * Real.sqrt 16) = 2^3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sixteen_times_sqrt_sixteen_equals_eight_l4105_410586


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l4105_410583

theorem cubic_equation_roots (p : ℝ) : 
  (∃ x y z : ℤ, x > 0 ∧ y > 0 ∧ z > 0 ∧
   (∀ t : ℝ, 5*t^3 - 5*(p+1)*t^2 + (71*p - 1)*t + 1 = 66*p ↔ t = x ∨ t = y ∨ t = z))
  ↔ p = 76 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l4105_410583


namespace NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l4105_410570

theorem termite_ridden_not_collapsing 
  (total_homes : ℕ) 
  (termite_ridden : ℕ) 
  (collapsing : ℕ) 
  (h1 : termite_ridden = total_homes / 3)
  (h2 : collapsing = (termite_ridden * 4) / 7) :
  (termite_ridden - collapsing : ℚ) / total_homes = 3 / 21 :=
by sorry

end NUMINAMATH_CALUDE_termite_ridden_not_collapsing_l4105_410570


namespace NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l4105_410521

theorem regular_polygon_exterior_angle (n : ℕ) (n_pos : 0 < n) :
  (360 : ℝ) / n = 72 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_exterior_angle_l4105_410521


namespace NUMINAMATH_CALUDE_banana_group_size_l4105_410512

theorem banana_group_size 
  (total_bananas : ℕ) 
  (num_groups : ℕ) 
  (h1 : total_bananas = 392) 
  (h2 : num_groups = 196) : 
  total_bananas / num_groups = 2 := by
sorry

end NUMINAMATH_CALUDE_banana_group_size_l4105_410512


namespace NUMINAMATH_CALUDE_stone_slab_area_l4105_410567

/-- Given 50 square stone slabs with a length of 120 cm each, 
    the total floor area covered is 72 square meters. -/
theorem stone_slab_area (n : ℕ) (length_cm : ℝ) (total_area_m2 : ℝ) : 
  n = 50 → 
  length_cm = 120 → 
  total_area_m2 = (n * (length_cm / 100)^2) → 
  total_area_m2 = 72 := by
sorry

end NUMINAMATH_CALUDE_stone_slab_area_l4105_410567


namespace NUMINAMATH_CALUDE_goals_theorem_l4105_410546

def goals_problem (bruce_goals michael_goals jack_goals sarah_goals : ℕ) : Prop :=
  bruce_goals = 4 ∧
  michael_goals = 2 * bruce_goals ∧
  jack_goals = bruce_goals - 1 ∧
  sarah_goals = jack_goals / 2 ∧
  michael_goals + jack_goals + sarah_goals = 12

theorem goals_theorem :
  ∃ (bruce_goals michael_goals jack_goals sarah_goals : ℕ),
    goals_problem bruce_goals michael_goals jack_goals sarah_goals :=
by
  sorry

end NUMINAMATH_CALUDE_goals_theorem_l4105_410546


namespace NUMINAMATH_CALUDE_candy_difference_example_l4105_410518

/-- Given a total number of candies and the number of strawberry candies,
    calculate the difference between grape and strawberry candies. -/
def candy_difference (total : ℕ) (strawberry : ℕ) : ℕ :=
  (total - strawberry) - strawberry

/-- Theorem stating that given 821 total candies and 267 strawberry candies,
    the difference between grape and strawberry candies is 287. -/
theorem candy_difference_example : candy_difference 821 267 = 287 := by
  sorry

end NUMINAMATH_CALUDE_candy_difference_example_l4105_410518


namespace NUMINAMATH_CALUDE_swim_time_ratio_l4105_410513

/-- The ratio of time taken to swim upstream vs downstream -/
theorem swim_time_ratio 
  (Vm : ℝ) 
  (Vs : ℝ) 
  (h1 : Vm = 5) 
  (h2 : Vs = 1.6666666666666667) : 
  (Vm + Vs) / (Vm - Vs) = 2 := by
  sorry

end NUMINAMATH_CALUDE_swim_time_ratio_l4105_410513


namespace NUMINAMATH_CALUDE_parabola_translation_l4105_410559

def original_parabola (x : ℝ) : ℝ := x^2 + 1

def transformed_parabola (x : ℝ) : ℝ := x^2 + 4*x + 5

def translation_distance : ℝ := 2

theorem parabola_translation :
  ∀ x : ℝ, transformed_parabola (x + translation_distance) = original_parabola x :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l4105_410559


namespace NUMINAMATH_CALUDE_no_prime_solution_l4105_410517

-- Define a function to convert a number from base p to base 10
def to_base_10 (digits : List Nat) (p : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * p ^ i) 0

-- Define the left-hand side of the equation
def lhs (p : Nat) : Nat :=
  to_base_10 [6, 0, 0, 2] p +
  to_base_10 [4, 0, 4] p +
  to_base_10 [5, 1, 2] p +
  to_base_10 [2, 2, 2] p +
  to_base_10 [9] p

-- Define the right-hand side of the equation
def rhs (p : Nat) : Nat :=
  to_base_10 [3, 3, 4] p +
  to_base_10 [2, 7, 5] p +
  to_base_10 [1, 2, 3] p

-- State the theorem
theorem no_prime_solution :
  ¬ ∃ p : Nat, Nat.Prime p ∧ lhs p = rhs p :=
sorry

end NUMINAMATH_CALUDE_no_prime_solution_l4105_410517


namespace NUMINAMATH_CALUDE_largest_good_number_all_greater_bad_smallest_bad_number_all_lesser_good_l4105_410539

/-- Definition of a good number -/
def is_good (M : ℕ) : Prop :=
  ∃ a b c d : ℤ, M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ a * d = b * c

/-- 576 is a good number -/
theorem largest_good_number : is_good 576 := by sorry

/-- All numbers greater than 576 are bad numbers -/
theorem all_greater_bad (M : ℕ) : M > 576 → ¬ is_good M := by sorry

/-- 443 is a bad number -/
theorem smallest_bad_number : ¬ is_good 443 := by sorry

/-- All numbers less than 443 are good numbers -/
theorem all_lesser_good (M : ℕ) : M < 443 → is_good M := by sorry

end NUMINAMATH_CALUDE_largest_good_number_all_greater_bad_smallest_bad_number_all_lesser_good_l4105_410539


namespace NUMINAMATH_CALUDE_victor_initial_books_l4105_410540

/-- The number of books Victor had initially -/
def initial_books : ℕ := sorry

/-- The number of books Victor bought during the book fair -/
def bought_books : ℕ := 3

/-- The total number of books Victor had after buying more -/
def total_books : ℕ := 12

/-- Theorem stating that Victor initially had 9 books -/
theorem victor_initial_books : 
  initial_books + bought_books = total_books → initial_books = 9 := by
  sorry

end NUMINAMATH_CALUDE_victor_initial_books_l4105_410540


namespace NUMINAMATH_CALUDE_complex_number_coordinates_l4105_410597

theorem complex_number_coordinates : 
  let z : ℂ := Complex.I * (2 - Complex.I)
  (z.re = 1 ∧ z.im = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_coordinates_l4105_410597


namespace NUMINAMATH_CALUDE_largest_positive_root_bound_l4105_410525

theorem largest_positive_root_bound (b c : ℝ) (hb : |b| ≤ 3) (hc : |c| ≤ 3) :
  let r := (3 + Real.sqrt 21) / 2
  ∀ x : ℝ, x > 0 → x^2 + b*x + c = 0 → x ≤ r :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_root_bound_l4105_410525


namespace NUMINAMATH_CALUDE_cos_difference_equals_half_l4105_410511

theorem cos_difference_equals_half : 
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.cos (66 * π / 180) * Real.cos (54 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_difference_equals_half_l4105_410511


namespace NUMINAMATH_CALUDE_vertical_motion_time_relation_l4105_410500

/-- Represents the vertical motion of a ball thrown upward and returning to its starting point. -/
structure VerticalMotion where
  V₀ : ℝ  -- Initial velocity
  g : ℝ   -- Gravitational acceleration
  t₁ : ℝ  -- Time to reach maximum height
  H : ℝ   -- Maximum height
  t : ℝ   -- Total time of motion

/-- The theorem stating the relationship between initial velocity, gravity, and total time of motion. -/
theorem vertical_motion_time_relation (motion : VerticalMotion)
  (h_positive_V₀ : 0 < motion.V₀)
  (h_positive_g : 0 < motion.g)
  (h_max_height : motion.H = (1/2) * motion.g * motion.t₁^2)
  (h_symmetry : motion.t = 2 * motion.t₁) :
  motion.t = 2 * motion.V₀ / motion.g :=
by sorry

end NUMINAMATH_CALUDE_vertical_motion_time_relation_l4105_410500


namespace NUMINAMATH_CALUDE_even_sum_and_sum_greater_20_count_l4105_410579

def IntSet := Finset (Nat)

def range_1_to_20 : IntSet := Finset.range 20

def even_sum_pairs (s : IntSet) : Nat :=
  (s.filter (λ x => x ≤ 20)).card

def sum_greater_20_pairs (s : IntSet) : Nat :=
  (s.filter (λ x => x ≤ 20)).card

theorem even_sum_and_sum_greater_20_count :
  (even_sum_pairs range_1_to_20 = 90) ∧
  (sum_greater_20_pairs range_1_to_20 = 100) := by
  sorry

end NUMINAMATH_CALUDE_even_sum_and_sum_greater_20_count_l4105_410579


namespace NUMINAMATH_CALUDE_sine_function_properties_l4105_410527

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem sine_function_properties (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π/2) 
  (h_max : f ω φ (π/4) = 1) 
  (h_min : f ω φ (7*π/12) = -1) 
  (h_period : ∃ T > 0, ∀ x, f ω φ (x + T) = f ω φ x) :
  ω = 3 ∧ 
  φ = -π/4 ∧ 
  ∀ k : ℤ, ∀ x ∈ Set.Icc (2*k*π/3 + π/4) (2*k*π/3 + 7*π/12), 
    ∀ y ∈ Set.Icc (2*k*π/3 + π/4) (2*k*π/3 + 7*π/12), 
      x ≤ y → f ω φ x ≥ f ω φ y :=
by sorry

end NUMINAMATH_CALUDE_sine_function_properties_l4105_410527


namespace NUMINAMATH_CALUDE_expression_simplification_l4105_410547

theorem expression_simplification (x : ℝ) : 
  3*x - 4*(2 + x^2) + 5*(3 - x) - 6*(1 - 2*x + x^2) = 10*x - 10*x^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l4105_410547


namespace NUMINAMATH_CALUDE_coefficient_without_x_is_70_l4105_410503

/-- The coefficient of the term without x in (xy - 1/x)^8 -/
def coefficientWithoutX : ℕ :=
  Nat.choose 8 4

/-- Theorem: The coefficient of the term without x in (xy - 1/x)^8 is 70 -/
theorem coefficient_without_x_is_70 : coefficientWithoutX = 70 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_without_x_is_70_l4105_410503
