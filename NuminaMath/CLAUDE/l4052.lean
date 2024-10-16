import Mathlib

namespace NUMINAMATH_CALUDE_max_figures_in_cube_l4052_405223

/-- The volume of a rectangular cuboid -/
def volume (length width height : ℕ) : ℕ := length * width * height

/-- The dimensions of the cube -/
def cube_dim : ℕ := 3

/-- The dimensions of the figure -/
def figure_dim : Vector ℕ 3 := ⟨[2, 2, 1], by simp⟩

/-- The maximum number of figures that can fit in the cube -/
def max_figures : ℕ := 6

theorem max_figures_in_cube :
  (volume cube_dim cube_dim cube_dim) ≥ max_figures * (volume figure_dim[0] figure_dim[1] figure_dim[2]) ∧
  ∀ n : ℕ, n > max_figures → (volume cube_dim cube_dim cube_dim) < n * (volume figure_dim[0] figure_dim[1] figure_dim[2]) :=
by sorry

end NUMINAMATH_CALUDE_max_figures_in_cube_l4052_405223


namespace NUMINAMATH_CALUDE_triangle_max_area_l4052_405254

/-- The maximum area of a triangle ABC where b = 3a and c = 2 -/
theorem triangle_max_area (a b c : ℝ) (h1 : b = 3 * a) (h2 : c = 2) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let area := (1/2) * b * c * Real.sin A
  ∀ x > 0, area ≤ Real.sqrt 2 / 2 ∧ 
  ∃ a₀ > 0, area = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l4052_405254


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l4052_405277

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((4 - 1)^2 + (x - 3)^2 : ℝ) = 5^2 → 
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l4052_405277


namespace NUMINAMATH_CALUDE_fraction_calculation_l4052_405259

theorem fraction_calculation : 
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l4052_405259


namespace NUMINAMATH_CALUDE_computer_store_discount_rate_l4052_405299

/-- Proves that the discount rate of the second store is approximately 0.87% given the conditions of the problem -/
theorem computer_store_discount_rate (price1 : ℝ) (discount1 : ℝ) (price2 : ℝ) (price_diff : ℝ) :
  price1 = 950 →
  discount1 = 0.06 →
  price2 = 920 →
  price_diff = 19 →
  let discounted_price1 := price1 * (1 - discount1)
  let discounted_price2 := discounted_price1 + price_diff
  let discount2 := (price2 - discounted_price2) / price2
  ∃ ε > 0, |discount2 - 0.0087| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_computer_store_discount_rate_l4052_405299


namespace NUMINAMATH_CALUDE_det_2x2_matrix_l4052_405209

/-- The determinant of a 2x2 matrix [[5, x], [-3, 4]] is 20 + 3x -/
theorem det_2x2_matrix (x : ℝ) : 
  Matrix.det !![5, x; -3, 4] = 20 + 3 * x := by
  sorry

end NUMINAMATH_CALUDE_det_2x2_matrix_l4052_405209


namespace NUMINAMATH_CALUDE_rental_miles_driven_l4052_405255

/-- Given rental information, calculate the number of miles driven -/
theorem rental_miles_driven (rental_fee : ℝ) (charge_per_mile : ℝ) (total_paid : ℝ) : 
  rental_fee = 20.99 →
  charge_per_mile = 0.25 →
  total_paid = 95.74 →
  (total_paid - rental_fee) / charge_per_mile = 299 :=
by
  sorry

end NUMINAMATH_CALUDE_rental_miles_driven_l4052_405255


namespace NUMINAMATH_CALUDE_only_vertical_angles_always_equal_l4052_405218

-- Define the types for lines and angles
def Line : Type := ℝ → ℝ → Prop
def Angle : Type := ℝ

-- Define the relationships between angles
def are_alternate_interior (a b : Angle) (l1 l2 : Line) : Prop := sorry
def are_consecutive_interior (a b : Angle) (l1 l2 : Line) : Prop := sorry
def are_vertical (a b : Angle) : Prop := sorry
def are_adjacent_supplementary (a b : Angle) : Prop := sorry
def are_corresponding (a b : Angle) (l1 l2 : Line) : Prop := sorry

-- Define the property of being supplementary
def are_supplementary (a b : Angle) : Prop := sorry

-- Theorem stating that only vertical angles are always equal
theorem only_vertical_angles_always_equal :
  ∀ (a b : Angle) (l1 l2 : Line),
    (are_vertical a b → a = b) ∧
    (are_alternate_interior a b l1 l2 → ¬(l1 = l2) → ¬(a = b)) ∧
    (are_consecutive_interior a b l1 l2 → ¬(l1 = l2) → ¬(are_supplementary a b)) ∧
    (are_adjacent_supplementary a b → ¬(a = b)) ∧
    (are_corresponding a b l1 l2 → ¬(l1 = l2) → ¬(a = b)) :=
by
  sorry

end NUMINAMATH_CALUDE_only_vertical_angles_always_equal_l4052_405218


namespace NUMINAMATH_CALUDE_last_digit_2016_octal_l4052_405237

def decimal_to_octal_last_digit (n : ℕ) : ℕ :=
  n % 8

theorem last_digit_2016_octal : decimal_to_octal_last_digit 2016 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_2016_octal_l4052_405237


namespace NUMINAMATH_CALUDE_unique_function_solution_l4052_405226

theorem unique_function_solution :
  ∃! f : ℝ → ℝ, (∀ x y : ℝ, f (x + f y - 1) = x + y) ∧ (∀ x : ℝ, f x = x + 1/2) :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l4052_405226


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l4052_405284

theorem divisible_by_eleven (n : ℕ) : 
  11 ∣ (6^(2*n) + 3^(n+2) + 3^n) := by
sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l4052_405284


namespace NUMINAMATH_CALUDE_denominator_numerator_difference_l4052_405232

/-- The repeating decimal 0.868686... -/
def F : ℚ := 86 / 99

/-- F expressed as a decimal is 0.868686... (infinitely repeating) -/
axiom F_decimal : F = 0.868686

theorem denominator_numerator_difference :
  (F.den : ℤ) - (F.num : ℤ) = 13 := by sorry

end NUMINAMATH_CALUDE_denominator_numerator_difference_l4052_405232


namespace NUMINAMATH_CALUDE_min_a_for_increasing_cubic_l4052_405291

/-- Given a function f(x) = x^3 + ax that is increasing on [1, +∞), 
    the minimum value of a is -3. -/
theorem min_a_for_increasing_cubic (a : ℝ) : 
  (∀ x ≥ 1, Monotone (fun x => x^3 + a*x)) → a ≥ -3 := by
  sorry

end NUMINAMATH_CALUDE_min_a_for_increasing_cubic_l4052_405291


namespace NUMINAMATH_CALUDE_tan_15_degrees_l4052_405271

theorem tan_15_degrees : Real.tan (15 * π / 180) = 2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_15_degrees_l4052_405271


namespace NUMINAMATH_CALUDE_chocolate_chip_cookie_price_l4052_405221

/-- The price of a box of chocolate chip cookies given the following conditions:
  * Total boxes sold: 1,585
  * Combined value of all boxes: $1,586.75
  * Plain cookies price: $0.75 each
  * Number of plain cookie boxes sold: 793.375
-/
theorem chocolate_chip_cookie_price :
  let total_boxes : ℝ := 1585
  let total_value : ℝ := 1586.75
  let plain_cookie_price : ℝ := 0.75
  let plain_cookie_boxes : ℝ := 793.375
  let chocolate_chip_boxes : ℝ := total_boxes - plain_cookie_boxes
  let chocolate_chip_price : ℝ := (total_value - (plain_cookie_price * plain_cookie_boxes)) / chocolate_chip_boxes
  chocolate_chip_price = 1.2525 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_cookie_price_l4052_405221


namespace NUMINAMATH_CALUDE_calculate_expression_l4052_405266

theorem calculate_expression : -1^2023 - (-2)^3 - (-2) * (-3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4052_405266


namespace NUMINAMATH_CALUDE_group_size_calculation_l4052_405278

theorem group_size_calculation (n : ℕ) 
  (h1 : n * (40 - 3) = n * 40 - 40 + 10) : n = 10 := by
  sorry

#check group_size_calculation

end NUMINAMATH_CALUDE_group_size_calculation_l4052_405278


namespace NUMINAMATH_CALUDE_second_price_increase_l4052_405234

/-- Given an initial price increase of 20% followed by a second price increase,
    if the total price increase is 38%, then the second price increase is 15%. -/
theorem second_price_increase (P : ℝ) (x : ℝ) 
  (h1 : P > 0)
  (h2 : 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_second_price_increase_l4052_405234


namespace NUMINAMATH_CALUDE_similarity_ratio_bounds_l4052_405212

theorem similarity_ratio_bounds (x y z p : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_pos_p : 0 < p)
  (h_similar : y = x * (z / y) ∧ z = y * (p / z)) :
  let k := z / y
  let φ := (1 + Real.sqrt 5) / 2
  φ⁻¹ < k ∧ k < φ := by
  sorry

end NUMINAMATH_CALUDE_similarity_ratio_bounds_l4052_405212


namespace NUMINAMATH_CALUDE_canMeasureFourLiters_l4052_405230

/-- Represents a container with a certain capacity -/
structure Container where
  capacity : ℕ
  current : ℕ
  h : current ≤ capacity

/-- Represents the state of the water measuring system -/
structure WaterSystem where
  small : Container
  large : Container

/-- Checks if the given state has 4 liters in the large container -/
def hasFourLiters (state : WaterSystem) : Prop :=
  state.large.current = 4

/-- Defines the possible operations on the water system -/
inductive Operation
  | FillSmall
  | FillLarge
  | EmptySmall
  | EmptyLarge
  | PourSmallToLarge
  | PourLargeToSmall

/-- Applies an operation to the water system -/
def applyOperation (op : Operation) (state : WaterSystem) : WaterSystem :=
  sorry

/-- Theorem stating that it's possible to measure 4 liters -/
theorem canMeasureFourLiters :
  ∃ (ops : List Operation),
    let initialState : WaterSystem := {
      small := { capacity := 3, current := 0, h := by simp },
      large := { capacity := 5, current := 0, h := by simp }
    }
    let finalState := ops.foldl (fun state op => applyOperation op state) initialState
    hasFourLiters finalState :=
  sorry

end NUMINAMATH_CALUDE_canMeasureFourLiters_l4052_405230


namespace NUMINAMATH_CALUDE_largest_valid_partition_l4052_405288

/-- Represents a partition of the set {1, 2, ..., m} into n subsets -/
def Partition (m : ℕ) (n : ℕ) := Fin n → Finset (Fin m)

/-- Checks if a partition satisfies the condition that the product of two different
    elements in the same subset is never a perfect square -/
def ValidPartition (p : Partition m n) : Prop :=
  ∀ i : Fin n, ∀ x y : Fin m, x ∈ p i → y ∈ p i → x ≠ y →
    ¬ ∃ z : ℕ, (x.val + 1) * (y.val + 1) = z * z

/-- The main theorem stating that n^2 + 2n is the largest m for which
    a valid partition exists -/
theorem largest_valid_partition (n : ℕ) (h : 0 < n) :
  (∃ p : Partition (n^2 + 2*n) n, ValidPartition p) ∧
  (∀ m : ℕ, m > n^2 + 2*n → ¬ ∃ p : Partition m n, ValidPartition p) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_partition_l4052_405288


namespace NUMINAMATH_CALUDE_managers_salary_l4052_405287

/-- Proves that given 24 employees with an average salary of Rs. 2400, 
    if adding a manager's salary increases the average by Rs. 100, 
    then the manager's salary is Rs. 4900. -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) : 
  num_employees = 24 → 
  avg_salary = 2400 → 
  salary_increase = 100 → 
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - 
   num_employees * avg_salary) = 4900 := by
  sorry

#check managers_salary

end NUMINAMATH_CALUDE_managers_salary_l4052_405287


namespace NUMINAMATH_CALUDE_unique_solution_condition_l4052_405228

theorem unique_solution_condition (b : ℝ) : 
  (∃! x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0) ↔ b < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l4052_405228


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_empty_solution_l4052_405279

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |2*x + 2|

-- Theorem for the solution set of f(x) < 3
theorem solution_set_f_less_than_3 :
  {x : ℝ | f x < 3} = {x : ℝ | -4/3 < x ∧ x < 0} := by sorry

-- Theorem for the range of a when f(x) < a has no solutions
theorem range_of_a_empty_solution :
  {a : ℝ | ∀ x, f x ≥ a} = {a : ℝ | a ≤ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_3_range_of_a_empty_solution_l4052_405279


namespace NUMINAMATH_CALUDE_coefficient_x_squared_is_70_l4052_405210

/-- The coefficient of x^2 in the expansion of (2+x)(1-2x)^5 -/
def coefficient_x_squared : ℤ :=
  2 * (Nat.choose 5 2) * (-2)^2 + (Nat.choose 5 1) * (-2)

/-- Theorem stating that the coefficient of x^2 in the expansion of (2+x)(1-2x)^5 is 70 -/
theorem coefficient_x_squared_is_70 : coefficient_x_squared = 70 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_is_70_l4052_405210


namespace NUMINAMATH_CALUDE_roller_derby_laces_l4052_405273

theorem roller_derby_laces (num_teams : ℕ) (members_per_team : ℕ) (skates_per_member : ℕ) (total_laces : ℕ) :
  num_teams = 4 →
  members_per_team = 10 →
  skates_per_member = 2 →
  total_laces = 240 →
  total_laces / (num_teams * members_per_team * skates_per_member) = 3 :=
by sorry

end NUMINAMATH_CALUDE_roller_derby_laces_l4052_405273


namespace NUMINAMATH_CALUDE_philatelist_stamps_problem_l4052_405249

theorem philatelist_stamps_problem :
  ∃! x : ℕ,
    x % 2 = 1 ∧
    x % 3 = 1 ∧
    x % 5 = 3 ∧
    x % 9 = 7 ∧
    150 < x ∧
    x ≤ 300 ∧
    x = 223 := by sorry

end NUMINAMATH_CALUDE_philatelist_stamps_problem_l4052_405249


namespace NUMINAMATH_CALUDE_arccos_one_half_l4052_405290

theorem arccos_one_half : Real.arccos (1 / 2) = π / 3 := by sorry

end NUMINAMATH_CALUDE_arccos_one_half_l4052_405290


namespace NUMINAMATH_CALUDE_enrollment_theorem_l4052_405202

-- Define the schools and their enrollments
def schools : Fin 4 → ℕ
| 0 => 1300  -- Varsity
| 1 => 1500  -- Northwest
| 2 => 1800  -- Central
| 3 => 1600  -- Greenbriar
| _ => 0     -- This case should never occur

-- Calculate the average enrollment
def average_enrollment : ℚ := (schools 0 + schools 1 + schools 2 + schools 3) / 4

-- Calculate the positive difference between a school's enrollment and the average
def positive_difference (i : Fin 4) : ℚ := |schools i - average_enrollment|

-- Theorem stating the average enrollment and positive differences
theorem enrollment_theorem :
  average_enrollment = 1550 ∧
  positive_difference 0 = 250 ∧
  positive_difference 1 = 50 ∧
  positive_difference 2 = 250 ∧
  positive_difference 3 = 50 :=
by sorry

end NUMINAMATH_CALUDE_enrollment_theorem_l4052_405202


namespace NUMINAMATH_CALUDE_total_price_two_corgis_is_2507_l4052_405203

/-- Calculates the total price for two Corgi dogs with given conditions -/
def total_price_two_corgis (cost : ℝ) (profit_percent : ℝ) (discount_percent : ℝ) (tax_percent : ℝ) (shipping_fee : ℝ) : ℝ :=
  let selling_price := cost * (1 + profit_percent)
  let total_before_discount := 2 * selling_price
  let discounted_price := total_before_discount * (1 - discount_percent)
  let price_with_tax := discounted_price * (1 + tax_percent)
  price_with_tax + shipping_fee

/-- Theorem stating the total price for two Corgi dogs is $2507 -/
theorem total_price_two_corgis_is_2507 :
  total_price_two_corgis 1000 0.30 0.10 0.05 50 = 2507 := by
  sorry

end NUMINAMATH_CALUDE_total_price_two_corgis_is_2507_l4052_405203


namespace NUMINAMATH_CALUDE_limit_of_a_l4052_405238

def a (n : ℕ) : ℚ := (3 * n - 1) / (5 * n + 1)

theorem limit_of_a : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 3/5| < ε := by sorry

end NUMINAMATH_CALUDE_limit_of_a_l4052_405238


namespace NUMINAMATH_CALUDE_foci_distance_of_problem_ellipse_l4052_405206

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

-- Define the conditions of the problem
def problem_ellipse : Ellipse := {
  center := (5, 2)
  semi_major_axis := 5
  semi_minor_axis := 2
}

-- Theorem statement
theorem foci_distance_of_problem_ellipse :
  let e := problem_ellipse
  let c := Real.sqrt (e.semi_major_axis ^ 2 - e.semi_minor_axis ^ 2)
  c = Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_foci_distance_of_problem_ellipse_l4052_405206


namespace NUMINAMATH_CALUDE_three_to_negative_x_is_exponential_l4052_405260

/-- Definition of an exponential function -/
def is_exponential_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x, f x = a^x

/-- The function y = 3^(-x) is an exponential function -/
theorem three_to_negative_x_is_exponential :
  is_exponential_function (fun x => 3^(-x)) :=
sorry

end NUMINAMATH_CALUDE_three_to_negative_x_is_exponential_l4052_405260


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l4052_405268

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- Marks in Physics, Chemistry, and Mathematics
  (h1 : P + C + M = P + 150) -- Total marks condition
  : (C + M) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l4052_405268


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_2023_l4052_405241

theorem simplify_and_evaluate (x : ℝ) : (x + 1)^2 - x * (x + 1) = x + 1 :=
  sorry

theorem evaluate_at_2023 : (2023 + 1)^2 - 2023 * (2023 + 1) = 2024 :=
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_evaluate_at_2023_l4052_405241


namespace NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l4052_405253

theorem existence_of_n_with_k_prime_factors (k m : ℕ) (hk : k > 0) (hm : m > 0) (hm_odd : Odd m) :
  ∃ n : ℕ, n > 0 ∧ (∃ (p : Finset ℕ), p.card ≥ k ∧ ∀ q ∈ p, Prime q ∧ q ∣ (m^n + n^m)) :=
sorry

end NUMINAMATH_CALUDE_existence_of_n_with_k_prime_factors_l4052_405253


namespace NUMINAMATH_CALUDE_probability_at_most_six_distinct_numbers_l4052_405227

theorem probability_at_most_six_distinct_numbers : 
  let n_dice : ℕ := 8
  let n_faces : ℕ := 6
  let total_outcomes : ℕ := n_faces ^ n_dice
  let favorable_outcomes : ℕ := 3628800
  (favorable_outcomes : ℚ) / total_outcomes = 45 / 52 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_most_six_distinct_numbers_l4052_405227


namespace NUMINAMATH_CALUDE_number_of_teams_in_league_l4052_405257

/-- The number of teams in the league -/
def n : ℕ := 20

/-- The number of games each team plays against every other team -/
def games_per_pair : ℕ := 4

/-- The total number of games played in the season -/
def total_games : ℕ := 760

/-- Theorem stating that n is the correct number of teams in the league -/
theorem number_of_teams_in_league :
  n * (n - 1) * games_per_pair / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_number_of_teams_in_league_l4052_405257


namespace NUMINAMATH_CALUDE_correct_evaluation_l4052_405256

/-- Evaluates an expression according to right-to-left rules -/
noncomputable def evaluate (a b c d e : ℝ) : ℝ :=
  a * (b^c - (d + e))

/-- Theorem stating that the evaluation is correct -/
theorem correct_evaluation (a b c d e : ℝ) :
  evaluate a b c d e = a * (b^c - (d + e)) := by sorry

end NUMINAMATH_CALUDE_correct_evaluation_l4052_405256


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_8191_l4052_405217

def greatest_prime_factor (n : ℕ) : ℕ := sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_greatest_prime_factor_8191 :
  sum_of_digits (greatest_prime_factor 8191) = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_factor_8191_l4052_405217


namespace NUMINAMATH_CALUDE_complex_equation_solution_l4052_405236

theorem complex_equation_solution (z : ℂ) : 
  (1 : ℂ) + Complex.I * Real.sqrt 3 = z * ((1 : ℂ) - Complex.I * Real.sqrt 3) → 
  z = -1/2 + Complex.I * (Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l4052_405236


namespace NUMINAMATH_CALUDE_product_ab_equals_six_l4052_405264

def A : Set ℝ := {-1, 3}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b = 0}

theorem product_ab_equals_six (a b : ℝ) (h : A = B a b) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_equals_six_l4052_405264


namespace NUMINAMATH_CALUDE_multiplication_digits_sum_l4052_405295

theorem multiplication_digits_sum (x y : Nat) : 
  x < 10 → y < 10 → (30 + x) * (10 * y + 4) = 136 → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_digits_sum_l4052_405295


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l4052_405200

theorem quadratic_roots_relation (p q : ℝ) : 
  (∃ a b : ℝ, 
    (2 * a^2 - 6 * a + 1 = 0) ∧ 
    (2 * b^2 - 6 * b + 1 = 0) ∧
    ((3 * a - 1)^2 + p * (3 * a - 1) + q = 0) ∧
    ((3 * b - 1)^2 + p * (3 * b - 1) + q = 0)) →
  q = -0.5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l4052_405200


namespace NUMINAMATH_CALUDE_negation_of_implication_l4052_405244

theorem negation_of_implication (a b : ℝ) : 
  ¬(ab = 0 → a = 0 ∨ b = 0) ↔ (ab ≠ 0 → a ≠ 0 ∧ b ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l4052_405244


namespace NUMINAMATH_CALUDE_recording_time_is_one_hour_l4052_405286

/-- Represents the recording interval in seconds -/
def recording_interval : ℕ := 5

/-- Represents the number of recorded instances -/
def recorded_instances : ℕ := 720

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem: The total recording time is 1 hour -/
theorem recording_time_is_one_hour : 
  (recording_interval * recorded_instances) / seconds_per_hour = 1 := by
  sorry

end NUMINAMATH_CALUDE_recording_time_is_one_hour_l4052_405286


namespace NUMINAMATH_CALUDE_perpendicular_lines_main_theorem_l4052_405222

/-- Two lines are perpendicular if their slopes multiply to -1 or if one of them is vertical --/
def perpendicular (m1 m2 : ℝ) : Prop :=
  m1 * m2 = -1 ∨ m1 = 0 ∨ m2 = 0

theorem perpendicular_lines (a : ℝ) :
  perpendicular (-a/2) (-1/(a*(a+1))) → a = -3/2 ∨ a = 0 := by
  sorry

/-- The main theorem stating the conditions for perpendicularity of the given lines --/
theorem main_theorem :
  ∀ a : ℝ, (∃ x y : ℝ, a*x + 2*y + 6 = 0 ∧ x + a*(a+1)*y + (a^2-1) = 0) →
  perpendicular (-a/2) (-1/(a*(a+1))) →
  a = -3/2 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_main_theorem_l4052_405222


namespace NUMINAMATH_CALUDE_decimal_2011_equals_base7_5602_l4052_405204

/-- Converts a base 10 number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else (n % 7) :: toBase7 (n / 7)

/-- Converts a list of digits in base 7 to a natural number in base 10 -/
def fromBase7 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 7 * acc + d) 0

theorem decimal_2011_equals_base7_5602 :
  fromBase7 [2, 0, 6, 5] = 2011 :=
by sorry

end NUMINAMATH_CALUDE_decimal_2011_equals_base7_5602_l4052_405204


namespace NUMINAMATH_CALUDE_highlighter_count_after_increase_l4052_405292

/-- Calculates the final number of highlighters after accounting for broken and borrowed ones, 
    and applying a 25% increase. -/
theorem highlighter_count_after_increase 
  (pink yellow blue green purple : ℕ)
  (broken_pink broken_yellow broken_blue : ℕ)
  (borrowed_green borrowed_purple : ℕ)
  (h1 : pink = 18)
  (h2 : yellow = 14)
  (h3 : blue = 11)
  (h4 : green = 8)
  (h5 : purple = 7)
  (h6 : broken_pink = 3)
  (h7 : broken_yellow = 2)
  (h8 : broken_blue = 1)
  (h9 : borrowed_green = 1)
  (h10 : borrowed_purple = 2) :
  let remaining := (pink - broken_pink) + (yellow - broken_yellow) + (blue - broken_blue) +
                   (green - borrowed_green) + (purple - borrowed_purple)
  let increase := (remaining * 25) / 100
  (remaining + increase) = 61 :=
by sorry

end NUMINAMATH_CALUDE_highlighter_count_after_increase_l4052_405292


namespace NUMINAMATH_CALUDE_min_value_weighted_sum_l4052_405251

theorem min_value_weighted_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : 2*x + 3*y + 4*z = 1) :
  (4/x) + (9/y) + (8/z) ≥ 81 ∧ 
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    2*x₀ + 3*y₀ + 4*z₀ = 1 ∧ (4/x₀) + (9/y₀) + (8/z₀) = 81 := by
  sorry

end NUMINAMATH_CALUDE_min_value_weighted_sum_l4052_405251


namespace NUMINAMATH_CALUDE_algorithm_properties_l4052_405261

-- Define the concept of an algorithm
def Algorithm : Type := Unit

-- Properties of algorithms
def yields_definite_result (a : Algorithm) : Prop := sorry
def multiple_solutions_exist : Prop := sorry
def terminates_in_finite_steps (a : Algorithm) : Prop := sorry

-- Theorem stating the correct properties of algorithms
theorem algorithm_properties :
  (∀ a : Algorithm, yields_definite_result a) ∧
  multiple_solutions_exist ∧
  (∀ a : Algorithm, terminates_in_finite_steps a) := by sorry

end NUMINAMATH_CALUDE_algorithm_properties_l4052_405261


namespace NUMINAMATH_CALUDE_modulus_of_z_l4052_405294

theorem modulus_of_z (z : ℂ) (h : (z - Complex.I) * Complex.I = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l4052_405294


namespace NUMINAMATH_CALUDE_bank_interest_calculation_l4052_405214

theorem bank_interest_calculation 
  (initial_deposit : ℝ) 
  (interest_rate : ℝ) 
  (years : ℕ) 
  (h1 : initial_deposit = 5600) 
  (h2 : interest_rate = 0.07) 
  (h3 : years = 2) : 
  initial_deposit + years * (initial_deposit * interest_rate) = 6384 :=
by
  sorry

end NUMINAMATH_CALUDE_bank_interest_calculation_l4052_405214


namespace NUMINAMATH_CALUDE_paige_pencils_at_home_l4052_405231

/-- The number of pencils Paige had in her backpack -/
def pencils_in_backpack : ℕ := 2

/-- The difference between the number of pencils at home and in the backpack -/
def pencil_difference : ℕ := 13

/-- The number of pencils Paige had at home -/
def pencils_at_home : ℕ := pencils_in_backpack + pencil_difference

theorem paige_pencils_at_home :
  pencils_at_home = 15 := by sorry

end NUMINAMATH_CALUDE_paige_pencils_at_home_l4052_405231


namespace NUMINAMATH_CALUDE_stamp_distribution_l4052_405219

theorem stamp_distribution (total : ℕ) (x y : ℕ) 
  (h1 : total = 70)
  (h2 : x = 4 * y + 5)
  (h3 : x + y = total) :
  x = 57 ∧ y = 13 := by
  sorry

end NUMINAMATH_CALUDE_stamp_distribution_l4052_405219


namespace NUMINAMATH_CALUDE_carol_achieves_target_average_l4052_405248

-- Define the inverse relationship between exercise time and test score
def inverse_relation (exercise_time : ℝ) (test_score : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ exercise_time * test_score = k

-- Define Carol's first test results
def first_test_exercise_time : ℝ := 45
def first_test_score : ℝ := 80

-- Define Carol's target average score
def target_average_score : ℝ := 85

-- Define Carol's exercise time for the second test
def second_test_exercise_time : ℝ := 40

-- Theorem to prove
theorem carol_achieves_target_average :
  inverse_relation first_test_exercise_time first_test_score →
  inverse_relation second_test_exercise_time ((2 * target_average_score * 2) - first_test_score) →
  (first_test_score + ((2 * target_average_score * 2) - first_test_score)) / 2 = target_average_score :=
by
  sorry

end NUMINAMATH_CALUDE_carol_achieves_target_average_l4052_405248


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l4052_405243

theorem largest_multiple_of_15_less_than_500 : 
  ∃ n : ℕ, n * 15 = 495 ∧ 
  495 < 500 ∧ 
  ∀ m : ℕ, m * 15 < 500 → m * 15 ≤ 495 := by
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_less_than_500_l4052_405243


namespace NUMINAMATH_CALUDE_cube_sum_of_roots_l4052_405283

theorem cube_sum_of_roots (a b c : ℂ) : 
  (5 * a^3 + 2003 * a + 3005 = 0) → 
  (5 * b^3 + 2003 * b + 3005 = 0) → 
  (5 * c^3 + 2003 * c + 3005 = 0) → 
  (a + b)^3 + (b + c)^3 + (c + a)^3 = 1803 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_of_roots_l4052_405283


namespace NUMINAMATH_CALUDE_sum_of_rotated_digits_l4052_405281

theorem sum_of_rotated_digits : 
  2345 + 3452 + 4523 + 5234 + 3245 + 2453 + 4532 + 5324 = 8888 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rotated_digits_l4052_405281


namespace NUMINAMATH_CALUDE_first_rewind_time_l4052_405216

theorem first_rewind_time (total_time second_rewind_time first_segment second_segment third_segment : ℕ) 
  (h1 : total_time = 120)
  (h2 : second_rewind_time = 15)
  (h3 : first_segment = 35)
  (h4 : second_segment = 45)
  (h5 : third_segment = 20) :
  total_time - (first_segment + second_segment + third_segment) - second_rewind_time = 5 := by
sorry

end NUMINAMATH_CALUDE_first_rewind_time_l4052_405216


namespace NUMINAMATH_CALUDE_michael_earnings_l4052_405285

/-- Michael's earnings from selling paintings --/
theorem michael_earnings (large_price small_price : ℕ) (large_quantity small_quantity : ℕ) :
  large_price = 100 →
  small_price = 80 →
  large_quantity = 5 →
  small_quantity = 8 →
  large_price * large_quantity + small_price * small_quantity = 1140 :=
by sorry

end NUMINAMATH_CALUDE_michael_earnings_l4052_405285


namespace NUMINAMATH_CALUDE_jesses_room_length_l4052_405208

theorem jesses_room_length (width : ℝ) (total_area : ℝ) (h1 : width = 8) (h2 : total_area = 96) :
  total_area / width = 12 := by
sorry

end NUMINAMATH_CALUDE_jesses_room_length_l4052_405208


namespace NUMINAMATH_CALUDE_max_integers_satisfying_inequalities_l4052_405280

theorem max_integers_satisfying_inequalities :
  (∃ x : ℕ, x = 7 ∧ 50 * x < 360 ∧ ∀ y : ℕ, 50 * y < 360 → y ≤ x) ∧
  (∃ y : ℕ, y = 4 ∧ 80 * y < 352 ∧ ∀ z : ℕ, 80 * z < 352 → z ≤ y) ∧
  (∃ z : ℕ, z = 6 ∧ 70 * z < 424 ∧ ∀ w : ℕ, 70 * w < 424 → w ≤ z) ∧
  (∃ w : ℕ, w = 4 ∧ 60 * w < 245 ∧ ∀ v : ℕ, 60 * v < 245 → v ≤ w) :=
by sorry

end NUMINAMATH_CALUDE_max_integers_satisfying_inequalities_l4052_405280


namespace NUMINAMATH_CALUDE_no_zero_points_for_exp_minus_x_l4052_405296

theorem no_zero_points_for_exp_minus_x :
  ∀ x : ℝ, x > 0 → ∃ ε : ℝ, ε > 0 ∧ Real.exp x - x > ε := by
  sorry

end NUMINAMATH_CALUDE_no_zero_points_for_exp_minus_x_l4052_405296


namespace NUMINAMATH_CALUDE_wall_width_calculation_l4052_405211

/-- Given a square mirror and a rectangular wall, if the mirror's area is exactly half the area of the wall,
    the side length of the mirror is 34 inches, and the length of the wall is 42.81481481481482 inches,
    then the width of the wall is 54 inches. -/
theorem wall_width_calculation (mirror_side : ℝ) (wall_length : ℝ) (wall_width : ℝ) :
  mirror_side = 34 →
  wall_length = 42.81481481481482 →
  mirror_side ^ 2 = (wall_length * wall_width) / 2 →
  wall_width = 54 := by
  sorry

end NUMINAMATH_CALUDE_wall_width_calculation_l4052_405211


namespace NUMINAMATH_CALUDE_a_values_l4052_405205

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (2 * x + a) ^ 3

-- Define the derivative of f
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 6 * (2 * x + a) ^ 2

-- Theorem statement
theorem a_values (a : ℝ) : f_derivative a 1 = 6 → a = -1 ∨ a = -3 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l4052_405205


namespace NUMINAMATH_CALUDE_trivia_team_groups_l4052_405276

theorem trivia_team_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) :
  total_students = 64 →
  not_picked = 36 →
  students_per_group = 7 →
  (total_students - not_picked) / students_per_group = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l4052_405276


namespace NUMINAMATH_CALUDE_simplify_expression_l4052_405224

theorem simplify_expression : 5 * (18 / 7) * (49 / -54) = -(245 / 9) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4052_405224


namespace NUMINAMATH_CALUDE_min_sum_of_product_1806_l4052_405213

theorem min_sum_of_product_1806 (x y z : ℕ+) (h : x * y * z = 1806) :
  ∃ (a b c : ℕ+), a * b * c = 1806 ∧ a + b + c ≤ x + y + z ∧ a + b + c = 72 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_1806_l4052_405213


namespace NUMINAMATH_CALUDE_star_operation_example_l4052_405220

def star_operation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem star_operation_example :
  let A : Set ℕ := {1,3,5,7}
  let B : Set ℕ := {2,3,5}
  star_operation A B = {1,7} := by
sorry

end NUMINAMATH_CALUDE_star_operation_example_l4052_405220


namespace NUMINAMATH_CALUDE_angle_sum_theorem_l4052_405201

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_theorem_l4052_405201


namespace NUMINAMATH_CALUDE_milk_container_problem_l4052_405297

theorem milk_container_problem (x : ℝ) : 
  (3 * x + 2 * 0.75 + 5 * 0.5 = 10) → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l4052_405297


namespace NUMINAMATH_CALUDE_part_one_part_two_l4052_405245

-- Define the quadratic functions p and q
def p (x : ℝ) : Prop := x^2 - 7*x + 10 < 0
def q (m x : ℝ) : Prop := x^2 - 4*m*x + 3*m^2 < 0

-- Theorem for part (1)
theorem part_one (m : ℝ) (h : m = 4) :
  (∀ x, p x ∧ q m x ↔ 4 < x ∧ x < 5) :=
sorry

-- Theorem for part (2)
theorem part_two :
  (∀ m, (∀ x, ¬(q m x) → ¬(p x)) ∧ (∃ x, ¬(p x) ∧ q m x)) ↔ (5/3 ≤ m ∧ m ≤ 2) :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l4052_405245


namespace NUMINAMATH_CALUDE_square_product_inequality_l4052_405289

theorem square_product_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 > a*b ∧ a*b > b^2 := by sorry

end NUMINAMATH_CALUDE_square_product_inequality_l4052_405289


namespace NUMINAMATH_CALUDE_min_volume_ratio_l4052_405263

/-- A spherical cap -/
structure SphericalCap where
  volume : ℝ

/-- A cylinder -/
structure Cylinder where
  volume : ℝ

/-- Configuration of a spherical cap and cylinder sharing a common inscribed sphere -/
structure Configuration where
  cap : SphericalCap
  cylinder : Cylinder
  bottom_faces_on_same_plane : Prop
  share_common_inscribed_sphere : Prop

/-- The minimum volume ratio theorem -/
theorem min_volume_ratio (config : Configuration) :
  ∃ (min_ratio : ℝ), min_ratio = 4/3 ∧
  ∀ (ratio : ℝ), ratio = config.cap.volume / config.cylinder.volume → min_ratio ≤ ratio :=
sorry

end NUMINAMATH_CALUDE_min_volume_ratio_l4052_405263


namespace NUMINAMATH_CALUDE_bakery_problem_l4052_405215

/-- Calculates the number of cookies remaining in the last bag -/
def cookies_in_last_bag (total_cookies : ℕ) (bag_capacity : ℕ) : ℕ :=
  total_cookies % bag_capacity

theorem bakery_problem (total_cookies : ℕ) (choc_chip : ℕ) (oatmeal : ℕ) (sugar : ℕ) 
  (bag_capacity : ℕ) (h1 : total_cookies = choc_chip + oatmeal + sugar) 
  (h2 : choc_chip = 154) (h3 : oatmeal = 86) (h4 : sugar = 52) (h5 : bag_capacity = 16) :
  (cookies_in_last_bag choc_chip bag_capacity = 10) ∧ 
  (cookies_in_last_bag oatmeal bag_capacity = 6) ∧ 
  (cookies_in_last_bag sugar bag_capacity = 4) := by
  sorry

#eval cookies_in_last_bag 154 16  -- Should output 10
#eval cookies_in_last_bag 86 16   -- Should output 6
#eval cookies_in_last_bag 52 16   -- Should output 4

end NUMINAMATH_CALUDE_bakery_problem_l4052_405215


namespace NUMINAMATH_CALUDE_four_Z_three_equals_37_l4052_405250

-- Define the Z operation
def Z (a b : ℕ) : ℕ := a^2 + a*b + b^2

-- Theorem to prove
theorem four_Z_three_equals_37 : Z 4 3 = 37 := by sorry

end NUMINAMATH_CALUDE_four_Z_three_equals_37_l4052_405250


namespace NUMINAMATH_CALUDE_binomial_sum_of_squares_l4052_405275

theorem binomial_sum_of_squares (a : ℝ) : 
  3 * a^4 + 1 = (a^2 + a)^2 + (a^2 - a)^2 + (a^2 - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_of_squares_l4052_405275


namespace NUMINAMATH_CALUDE_power_difference_equality_l4052_405269

theorem power_difference_equality : 5^(7+2) - 2^(5+3) = 1952869 := by sorry

end NUMINAMATH_CALUDE_power_difference_equality_l4052_405269


namespace NUMINAMATH_CALUDE_probability_two_positive_one_negative_l4052_405240

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1) * d

def first_10_terms (a₁ d : ℚ) : List ℚ :=
  List.map (arithmetic_sequence a₁ d) (List.range 10)

theorem probability_two_positive_one_negative
  (a₁ d : ℚ)
  (h₁ : arithmetic_sequence a₁ d 4 = 2)
  (h₂ : arithmetic_sequence a₁ d 7 = -4)
  : (((first_10_terms a₁ d).filter (λ x => x > 0)).length : ℚ) / 10 * 
    (((first_10_terms a₁ d).filter (λ x => x > 0)).length : ℚ) / 10 * 
    (((first_10_terms a₁ d).filter (λ x => x < 0)).length : ℚ) / 10 * 3 = 6 / 25 :=
by sorry

end NUMINAMATH_CALUDE_probability_two_positive_one_negative_l4052_405240


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l4052_405246

theorem sqrt_x_minus_one_meaningful (x : ℝ) :
  (∃ y : ℝ, y^2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_meaningful_l4052_405246


namespace NUMINAMATH_CALUDE_repeating_decimal_bounds_l4052_405242

/-- Represents a repeating decimal with an integer part and a repeating fractional part -/
structure RepeatingDecimal where
  integerPart : ℕ
  fractionalPart : List ℕ
  repeatingPart : List ℕ

/-- Converts a RepeatingDecimal to a real number -/
noncomputable def RepeatingDecimal.toReal (d : RepeatingDecimal) : ℝ :=
  sorry

/-- Generates all possible repeating decimals from a given decimal string -/
def generateRepeatingDecimals (s : String) : List RepeatingDecimal :=
  sorry

/-- Finds the maximum repeating decimal from a list of repeating decimals -/
noncomputable def findMaxRepeatingDecimal (decimals : List RepeatingDecimal) : RepeatingDecimal :=
  sorry

/-- Finds the minimum repeating decimal from a list of repeating decimals -/
noncomputable def findMinRepeatingDecimal (decimals : List RepeatingDecimal) : RepeatingDecimal :=
  sorry

theorem repeating_decimal_bounds :
  let decimals := generateRepeatingDecimals "0.20120415"
  let maxDecimal := findMaxRepeatingDecimal decimals
  let minDecimal := findMinRepeatingDecimal decimals
  maxDecimal = { integerPart := 0, fractionalPart := [2, 0, 1, 2, 0, 4, 1], repeatingPart := [5] } ∧
  minDecimal = { integerPart := 0, fractionalPart := [2], repeatingPart := [0, 1, 2, 0, 4, 1, 5] } :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_bounds_l4052_405242


namespace NUMINAMATH_CALUDE_fixed_distance_to_H_l4052_405207

/-- Given a parabola y^2 = 4x with origin O and moving points A and B on the parabola,
    such that OA ⊥ OB, and OH ⊥ AB where H is the foot of the perpendicular,
    prove that the point (2,0) has a fixed distance to H. -/
theorem fixed_distance_to_H (A B H : ℝ × ℝ) : 
  (∀ (y₁ y₂ : ℝ), A.2^2 = 4 * A.1 ∧ B.2^2 = 4 * B.1) →  -- A and B on parabola
  (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
  (∃ (m n : ℝ), H.1 = m * H.2 + n ∧ 
    A.1 = m * A.2 + n ∧ B.1 = m * B.2 + n) →  -- H on line AB
  (H.1 * 0 + H.2 * 1 = 0) →  -- OH ⊥ AB
  ∃ (r : ℝ), (H.1 - 2)^2 + H.2^2 = r^2 := by
    sorry

end NUMINAMATH_CALUDE_fixed_distance_to_H_l4052_405207


namespace NUMINAMATH_CALUDE_farm_acreage_difference_l4052_405239

theorem farm_acreage_difference (total_acres flax_acres : ℕ) 
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : flax_acres < total_acres - flax_acres) : 
  total_acres - flax_acres - flax_acres = 80 := by
sorry

end NUMINAMATH_CALUDE_farm_acreage_difference_l4052_405239


namespace NUMINAMATH_CALUDE_nonagon_intersection_points_l4052_405272

/-- A regular nonagon is a 9-sided regular polygon -/
def RegularNonagon : Type := Unit

/-- The number of distinct interior intersection points of diagonals in a regular nonagon -/
def intersectionPoints (n : RegularNonagon) : ℕ := sorry

/-- The number of ways to choose 4 vertices from 9 vertices -/
def chooseFromNine : ℕ := Nat.choose 9 4

/-- Theorem stating that the number of intersection points in a regular nonagon
    is equal to the number of ways to choose 4 vertices from 9 -/
theorem nonagon_intersection_points (n : RegularNonagon) :
  intersectionPoints n = chooseFromNine := by sorry

end NUMINAMATH_CALUDE_nonagon_intersection_points_l4052_405272


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l4052_405229

/-- A line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_m_value :
  ∀ m : ℝ,
  let l1 : Line := ⟨6, m, -1⟩
  let l2 : Line := ⟨2, -1, 1⟩
  parallel l1 l2 → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l4052_405229


namespace NUMINAMATH_CALUDE_lcm_gcd_product_12_75_l4052_405235

theorem lcm_gcd_product_12_75 : Nat.lcm 12 75 * Nat.gcd 12 75 = 900 := by sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_12_75_l4052_405235


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l4052_405258

universe u

def U : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Finset ℕ := {2, 4, 5}

theorem complement_of_A_in_U :
  (U \ A : Finset ℕ) = {1, 3, 6, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l4052_405258


namespace NUMINAMATH_CALUDE_fine_calculation_l4052_405293

/-- Calculates the fine for inappropriate items in the recycling bin -/
def calculate_fine (weeks : ℕ) (trash_bin_cost : ℚ) (recycling_bin_cost : ℚ) 
  (trash_bins : ℕ) (recycling_bins : ℕ) (discount_percent : ℚ) (total_bill : ℚ) : ℚ := 
  let weekly_cost := trash_bin_cost * trash_bins + recycling_bin_cost * recycling_bins
  let monthly_cost := weekly_cost * weeks
  let discount := discount_percent * monthly_cost
  let discounted_cost := monthly_cost - discount
  total_bill - discounted_cost

theorem fine_calculation :
  calculate_fine 4 10 5 2 1 (18/100) 102 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fine_calculation_l4052_405293


namespace NUMINAMATH_CALUDE_lcm_18_24_l4052_405225

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l4052_405225


namespace NUMINAMATH_CALUDE_absolute_value_plus_exponent_l4052_405274

theorem absolute_value_plus_exponent : |(-8 : ℤ)| + 3^(0 : ℕ) = 9 := by sorry

end NUMINAMATH_CALUDE_absolute_value_plus_exponent_l4052_405274


namespace NUMINAMATH_CALUDE_mean_score_all_students_l4052_405298

theorem mean_score_all_students
  (score_first : ℝ)
  (score_second : ℝ)
  (ratio_first_to_second : ℚ)
  (h1 : score_first = 90)
  (h2 : score_second = 75)
  (h3 : ratio_first_to_second = 2 / 3) :
  let total_students := (ratio_first_to_second + 1) * students_second
  let total_score := score_first * ratio_first_to_second * students_second + score_second * students_second
  total_score / total_students = 81 :=
by sorry

end NUMINAMATH_CALUDE_mean_score_all_students_l4052_405298


namespace NUMINAMATH_CALUDE_linear_combination_of_reals_with_rational_products_l4052_405265

theorem linear_combination_of_reals_with_rational_products 
  (a b c : ℝ) 
  (hab : ∃ (q : ℚ), a * b = q) 
  (hbc : ∃ (q : ℚ), b * c = q) 
  (hca : ∃ (q : ℚ), c * a = q) 
  (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  ∃ (x y z : ℤ), a * (x : ℝ) + b * (y : ℝ) + c * (z : ℝ) = 0 := by
sorry

end NUMINAMATH_CALUDE_linear_combination_of_reals_with_rational_products_l4052_405265


namespace NUMINAMATH_CALUDE_cost_price_of_ball_l4052_405270

/-- 
Given:
- The selling price of 13 balls is 720 Rs.
- The loss incurred is equal to the cost price of 5 balls.

Prove that the cost price of one ball is 90 Rs.
-/
theorem cost_price_of_ball (selling_price : ℕ) (num_balls : ℕ) (loss_balls : ℕ) :
  selling_price = 720 →
  num_balls = 13 →
  loss_balls = 5 →
  ∃ (cost_price : ℕ), 
    cost_price * num_balls - selling_price = cost_price * loss_balls ∧
    cost_price = 90 :=
by sorry

end NUMINAMATH_CALUDE_cost_price_of_ball_l4052_405270


namespace NUMINAMATH_CALUDE_shelter_cat_count_l4052_405252

/-- Proves that the total number of cats and kittens in the shelter is 280 --/
theorem shelter_cat_count : ∀ (adult_cats female_cats litters kittens_per_litter : ℕ),
  adult_cats = 120 →
  female_cats = 2 * adult_cats / 3 →
  litters = 2 * female_cats / 5 →
  kittens_per_litter = 5 →
  adult_cats + litters * kittens_per_litter = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_shelter_cat_count_l4052_405252


namespace NUMINAMATH_CALUDE_simplify_expression_l4052_405262

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - b^2 = 9*b^3 + 5*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4052_405262


namespace NUMINAMATH_CALUDE_angle_ABF_is_right_l4052_405267

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = (Real.sqrt 5 - 1) / 2

/-- The angle ABF in an ellipse, where A is the left vertex, 
    F is the right focus, and B is one endpoint of the minor axis -/
def angle_ABF (E : Ellipse) : ℝ := sorry

/-- Theorem: In an ellipse with the given properties, the angle ABF is 90° -/
theorem angle_ABF_is_right (E : Ellipse) : angle_ABF E = 90 := by sorry

end NUMINAMATH_CALUDE_angle_ABF_is_right_l4052_405267


namespace NUMINAMATH_CALUDE_no_eighteen_consecutive_good_l4052_405233

/-- A natural number is "good" if it has exactly two prime divisors -/
def isGood (n : ℕ) : Prop :=
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n.divisors = {1, p, q, n})

/-- Theorem: There do not exist 18 consecutive natural numbers that are all "good" -/
theorem no_eighteen_consecutive_good :
  ¬ ∃ k : ℕ, ∀ i : ℕ, i < 18 → isGood (k + i) := by
  sorry

end NUMINAMATH_CALUDE_no_eighteen_consecutive_good_l4052_405233


namespace NUMINAMATH_CALUDE_disease_gender_relation_expected_trial_cost_l4052_405247

-- Define the total number of patients
def total_patients : ℕ := 1800

-- Define the number of male and female patients
def male_patients : ℕ := 1200
def female_patients : ℕ := 600

-- Define the number of patients with type A disease
def male_type_a : ℕ := 800
def female_type_a : ℕ := 450

-- Define the χ² formula
def chi_square (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

-- Define the critical value for α = 0.001
def critical_value : ℚ := 10828 / 1000

-- Define the probability of producing antibodies
def antibody_prob : ℚ := 2 / 3

-- Define the cost per dose
def cost_per_dose : ℕ := 9

-- Define the number of doses per cycle
def doses_per_cycle : ℕ := 3

-- Theorem statements
theorem disease_gender_relation :
  chi_square male_type_a (male_patients - male_type_a) female_type_a (female_patients - female_type_a) > critical_value := by sorry

theorem expected_trial_cost :
  (20 : ℚ) / 27 * (3 * cost_per_dose) + 7 / 27 * (6 * cost_per_dose) = 34 := by sorry

end NUMINAMATH_CALUDE_disease_gender_relation_expected_trial_cost_l4052_405247


namespace NUMINAMATH_CALUDE_inverse_composition_equals_neg_sixteen_ninths_l4052_405282

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 7

-- Define the inverse function g⁻¹
noncomputable def g_inv (x : ℝ) : ℝ := (x - 7) / 3

-- Theorem statement
theorem inverse_composition_equals_neg_sixteen_ninths :
  g_inv (g_inv 12) = -16/9 :=
by sorry

end NUMINAMATH_CALUDE_inverse_composition_equals_neg_sixteen_ninths_l4052_405282
