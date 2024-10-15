import Mathlib

namespace NUMINAMATH_CALUDE_zoo_field_trip_vans_l2591_259160

/-- The number of vans needed for a field trip --/
def vans_needed (van_capacity : ℕ) (num_students : ℕ) (num_adults : ℕ) : ℕ :=
  (num_students + num_adults + van_capacity - 1) / van_capacity

/-- Theorem: The number of vans needed for the zoo field trip is 6 --/
theorem zoo_field_trip_vans : vans_needed 5 25 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_zoo_field_trip_vans_l2591_259160


namespace NUMINAMATH_CALUDE_adam_has_more_apple_difference_l2591_259186

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := 6

/-- Adam has more apples than Jackie -/
theorem adam_has_more : adam_apples > jackie_apples := by sorry

/-- The difference in apples between Adam and Jackie is 3 -/
theorem apple_difference : adam_apples - jackie_apples = 3 := by sorry

end NUMINAMATH_CALUDE_adam_has_more_apple_difference_l2591_259186


namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l2591_259141

theorem sum_of_roots_zero (p q a b c : ℝ) : 
  a ≠ b → b ≠ c → a ≠ c →
  a^3 + p*a + q = 0 →
  b^3 + p*b + q = 0 →
  c^3 + p*c + q = 0 →
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l2591_259141


namespace NUMINAMATH_CALUDE_new_person_weight_example_l2591_259195

/-- Calculates the weight of a new person given the initial number of persons,
    the average weight increase, and the weight of the replaced person. -/
def new_person_weight (initial_count : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * avg_increase

theorem new_person_weight_example :
  new_person_weight 7 6.2 76 = 119.4 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_example_l2591_259195


namespace NUMINAMATH_CALUDE_lawnmower_depreciation_l2591_259134

theorem lawnmower_depreciation (initial_value : ℝ) (first_depreciation_rate : ℝ) (second_depreciation_rate : ℝ) :
  initial_value = 100 →
  first_depreciation_rate = 0.25 →
  second_depreciation_rate = 0.20 →
  initial_value * (1 - first_depreciation_rate) * (1 - second_depreciation_rate) = 60 := by
sorry

end NUMINAMATH_CALUDE_lawnmower_depreciation_l2591_259134


namespace NUMINAMATH_CALUDE_driver_comparison_l2591_259136

theorem driver_comparison (d : ℝ) (h : d > 0) : d / 40 < 8 * d / 315 := by
  sorry

#check driver_comparison

end NUMINAMATH_CALUDE_driver_comparison_l2591_259136


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2591_259158

theorem gcd_lcm_product (a b : ℕ) (h1 : Nat.gcd a b = 12) (h2 : Nat.lcm a b = 168) :
  a * b = 2016 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2591_259158


namespace NUMINAMATH_CALUDE_reflection_line_sum_l2591_259163

/-- Given a reflection of point (0,1) across line y = mx + b to point (4,5), prove m + b = 4 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ 
    ((x - 0) * (x - 0) + (y - 1) * (y - 1)) / 4 = 
    ((x - 0) * (1 + y) / 2 - (y - 1) * (0 + x) / 2)^2 / ((x - 0)^2 + (y - 1)^2) ∧
    y = m * x + b) →
  m + b = 4 := by
sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l2591_259163


namespace NUMINAMATH_CALUDE_pennsylvania_quarters_l2591_259111

theorem pennsylvania_quarters (total : ℕ) (state_fraction : ℚ) (penn_fraction : ℚ) : 
  total = 35 → 
  state_fraction = 2 / 5 → 
  penn_fraction = 1 / 2 → 
  ⌊total * state_fraction * penn_fraction⌋ = 7 := by
  sorry

end NUMINAMATH_CALUDE_pennsylvania_quarters_l2591_259111


namespace NUMINAMATH_CALUDE_six_digit_repeating_divisible_by_11_l2591_259106

/-- A 6-digit integer where the first three digits and the last three digits
    form the same three-digit number in the same order is divisible by 11. -/
theorem six_digit_repeating_divisible_by_11 (N : ℕ) (a b c : ℕ) :
  N = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c →
  a < 10 → b < 10 → c < 10 →
  11 ∣ N :=
by sorry

end NUMINAMATH_CALUDE_six_digit_repeating_divisible_by_11_l2591_259106


namespace NUMINAMATH_CALUDE_combined_distance_is_1890_l2591_259145

/-- The combined swimming distance for Jamir, Sarah, and Julien for a week -/
def combined_swimming_distance (julien_distance : ℕ) : ℕ :=
  let sarah_distance := 2 * julien_distance
  let jamir_distance := sarah_distance + 20
  let days_in_week := 7
  (julien_distance + sarah_distance + jamir_distance) * days_in_week

/-- Theorem stating that the combined swimming distance for a week is 1890 meters -/
theorem combined_distance_is_1890 :
  combined_swimming_distance 50 = 1890 := by
  sorry

end NUMINAMATH_CALUDE_combined_distance_is_1890_l2591_259145


namespace NUMINAMATH_CALUDE_two_digit_sqrt_prob_l2591_259162

theorem two_digit_sqrt_prob : 
  let two_digit_numbers := Finset.Icc 10 99
  let satisfying_numbers := two_digit_numbers.filter (λ n => n.sqrt < 8)
  (satisfying_numbers.card : ℚ) / two_digit_numbers.card = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sqrt_prob_l2591_259162


namespace NUMINAMATH_CALUDE_rearranged_box_surface_area_l2591_259146

theorem rearranged_box_surface_area :
  let original_length : ℝ := 2
  let original_width : ℝ := 1
  let original_height : ℝ := 1
  let first_cut_height : ℝ := 1/4
  let second_cut_height : ℝ := 1/3
  let piece_A_height : ℝ := first_cut_height
  let piece_B_height : ℝ := second_cut_height
  let piece_C_height : ℝ := original_height - (piece_A_height + piece_B_height)
  let new_length : ℝ := original_width * 3
  let new_width : ℝ := original_length
  let new_height : ℝ := piece_A_height + piece_B_height + piece_C_height
  let top_bottom_area : ℝ := 2 * (new_length * new_width)
  let side_area : ℝ := 2 * (new_height * new_width)
  let front_back_area : ℝ := 2 * (new_length * new_height)
  let total_surface_area : ℝ := top_bottom_area + side_area + front_back_area
  total_surface_area = 12 := by
    sorry

end NUMINAMATH_CALUDE_rearranged_box_surface_area_l2591_259146


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l2591_259166

/-- A quadratic function with axis of symmetry at x = 9.5 -/
def p (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_symmetry (d e f : ℝ) :
  (∀ x, p d e f (9.5 + x) = p d e f (9.5 - x)) →  -- axis of symmetry at x = 9.5
  p d e f (-1) = 1 →  -- p(-1) = 1
  ∃ n : ℤ, p d e f 20 = n →  -- p(20) is an integer
  p d e f 20 = 1 := by  -- prove p(20) = 1
sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l2591_259166


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2591_259121

theorem solution_set_inequality (x : ℝ) : 
  (x ≠ -1 ∧ (2*x - 1)/(x + 1) ≤ 1) ↔ -1 < x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2591_259121


namespace NUMINAMATH_CALUDE_perpendicular_vectors_sum_l2591_259165

/-- Given two perpendicular vectors a and b in ℝ², prove their sum is (3, -1) -/
theorem perpendicular_vectors_sum (a b : ℝ × ℝ) :
  a.1 = x ∧ a.2 = 1 ∧ b = (1, -2) ∧ a.1 * b.1 + a.2 * b.2 = 0 →
  a + b = (3, -1) := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_sum_l2591_259165


namespace NUMINAMATH_CALUDE_triangle_side_length_l2591_259150

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  2 * b = a + c →
  B = π / 6 →
  (1 / 2) * a * c * Real.sin B = 3 / 2 →
  b = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2591_259150


namespace NUMINAMATH_CALUDE_sin_160_eq_sin_20_l2591_259176

theorem sin_160_eq_sin_20 : Real.sin (160 * π / 180) = Real.sin (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_160_eq_sin_20_l2591_259176


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l2591_259135

theorem quadratic_roots_property : ∀ x₁ x₂ : ℝ, 
  x₁^2 - 3*x₁ - 5 = 0 →
  x₂^2 - 3*x₂ - 5 = 0 →
  x₁ + x₂ - x₁ * x₂ = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l2591_259135


namespace NUMINAMATH_CALUDE_division_problem_l2591_259168

theorem division_problem (L S Q : ℕ) : 
  L - S = 2500 → 
  L = 2982 → 
  L = Q * S + 15 → 
  Q = 6 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2591_259168


namespace NUMINAMATH_CALUDE_pq_length_is_eight_l2591_259147

/-- A quadrilateral with three equal sides -/
structure ThreeEqualSidesQuadrilateral where
  -- The lengths of the four sides
  pq : ℝ
  qr : ℝ
  rs : ℝ
  sp : ℝ
  -- Three sides are equal
  three_equal : pq = qr ∧ pq = sp
  -- SR length is 16
  sr_length : rs = 16
  -- Perimeter is 40
  perimeter : pq + qr + rs + sp = 40

/-- The length of PQ in a ThreeEqualSidesQuadrilateral is 8 -/
theorem pq_length_is_eight (quad : ThreeEqualSidesQuadrilateral) : quad.pq = 8 :=
by sorry

end NUMINAMATH_CALUDE_pq_length_is_eight_l2591_259147


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2591_259105

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The sum of the first five terms of a sequence -/
def SumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5

/-- The sum of the squares of the first five terms of a sequence -/
def SumSquaresFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1^2 + a 2^2 + a 3^2 + a 4^2 + a 5^2

/-- The alternating sum of the first five terms of a sequence -/
def AlternatingSumFirstFive (a : ℕ → ℝ) : ℝ :=
  a 1 - a 2 + a 3 - a 4 + a 5

theorem arithmetic_geometric_sequence_property (a : ℕ → ℝ) :
  ArithmeticGeometricSequence a →
  SumFirstFive a = 3 →
  SumSquaresFirstFive a = 12 →
  AlternatingSumFirstFive a = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l2591_259105


namespace NUMINAMATH_CALUDE_columbus_discovery_year_l2591_259185

def is_15th_century (year : ℕ) : Prop := 1400 ≤ year ∧ year ≤ 1499

def sum_of_digits (year : ℕ) : ℕ :=
  (year / 1000) + ((year / 100) % 10) + ((year / 10) % 10) + (year % 10)

def tens_digit (year : ℕ) : ℕ := (year / 10) % 10

def units_digit (year : ℕ) : ℕ := year % 10

theorem columbus_discovery_year :
  ∃! year : ℕ,
    is_15th_century year ∧
    sum_of_digits year = 16 ∧
    tens_digit year / units_digit year = 4 ∧
    tens_digit year % units_digit year = 1 ∧
    year = 1492 :=
by
  sorry

end NUMINAMATH_CALUDE_columbus_discovery_year_l2591_259185


namespace NUMINAMATH_CALUDE_three_digit_difference_l2591_259107

theorem three_digit_difference (a b c : ℕ) (h1 : a ≥ 1) (h2 : a ≤ 9) (h3 : b ≥ 0) (h4 : b ≤ 9) (h5 : c ≥ 0) (h6 : c ≤ 9) (h7 : a = c + 2) :
  (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = 198 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_difference_l2591_259107


namespace NUMINAMATH_CALUDE_smallest_four_digit_non_divisor_l2591_259188

def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

def product_of_first_n (n : ℕ) : ℕ := Nat.factorial n

theorem smallest_four_digit_non_divisor :
  (∀ m : ℕ, 1000 ≤ m → m < 1005 → (product_of_first_n m)^2 % (sum_of_first_n m) = 0) ∧
  (product_of_first_n 1005)^2 % (sum_of_first_n 1005) ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_non_divisor_l2591_259188


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l2591_259137

def first_four_primes : List Nat := [2, 3, 5, 7]

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  let reciprocals := first_four_primes.map (λ x => (1 : ℚ) / x)
  (reciprocals.sum / reciprocals.length) = 247 / 840 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_reciprocals_of_first_four_primes_l2591_259137


namespace NUMINAMATH_CALUDE_circle_c_and_line_theorem_l2591_259189

/-- Circle C with given properties -/
structure CircleC where
  radius : ℝ
  center : ℝ × ℝ
  chord_length : ℝ
  center_below_x_axis : center.2 < 0
  center_on_y_eq_x : center.1 = center.2
  radius_eq_3 : radius = 3
  chord_eq_2root5 : chord_length = 2 * Real.sqrt 5

/-- Line with slope 1 -/
structure Line where
  b : ℝ
  equation : ℝ → ℝ
  slope_eq_1 : ∀ x, equation x = x + b

/-- Theorem about CircleC and related Line -/
theorem circle_c_and_line_theorem (c : CircleC) :
  (∃ x y, (x + 2)^2 + (y + 2)^2 = 9 ↔ (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) ∧
  (∃ l : Line, (l.b = 1 ∨ l.b = -1) ∧
    ∃ x₁ y₁ x₂ y₂, ((x₁ - c.center.1)^2 + (y₁ - c.center.2)^2 = c.radius^2 ∧
                    (x₂ - c.center.1)^2 + (y₂ - c.center.2)^2 = c.radius^2 ∧
                    y₁ = l.equation x₁ ∧ y₂ = l.equation x₂ ∧
                    x₁ * x₂ + y₁ * y₂ = 0)) :=
sorry

end NUMINAMATH_CALUDE_circle_c_and_line_theorem_l2591_259189


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2591_259197

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, n ≤ 999 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2591_259197


namespace NUMINAMATH_CALUDE_jacqueline_initial_plums_l2591_259169

/-- The number of plums Jacqueline had initially -/
def initial_plums : ℕ := 16

/-- The number of guavas Jacqueline had initially -/
def initial_guavas : ℕ := 18

/-- The number of apples Jacqueline had initially -/
def initial_apples : ℕ := 21

/-- The number of fruits Jacqueline gave away -/
def fruits_given_away : ℕ := 40

/-- The number of fruits Jacqueline had left -/
def fruits_left : ℕ := 15

/-- Theorem stating that the initial number of plums is 16 -/
theorem jacqueline_initial_plums :
  initial_plums = 16 ∧
  initial_plums + initial_guavas + initial_apples = fruits_given_away + fruits_left :=
by sorry

end NUMINAMATH_CALUDE_jacqueline_initial_plums_l2591_259169


namespace NUMINAMATH_CALUDE_age_sum_problem_l2591_259115

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ 
  a > c ∧ 
  c < 10 ∧ 
  a * b * c = 162 → 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l2591_259115


namespace NUMINAMATH_CALUDE_cosine_sum_problem_l2591_259174

theorem cosine_sum_problem (x y z : ℝ) : 
  x = Real.cos (π / 13) → 
  y = Real.cos (3 * π / 13) → 
  z = Real.cos (9 * π / 13) → 
  x * y + y * z + z * x = -1/4 := by
sorry

end NUMINAMATH_CALUDE_cosine_sum_problem_l2591_259174


namespace NUMINAMATH_CALUDE_no_equal_result_from_19_and_98_l2591_259161

/-- Represents the two possible operations: squaring or adding one -/
inductive Operation
  | square
  | addOne

/-- Applies the given operation to a number -/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.square => n * n
  | Operation.addOne => n + 1

/-- Applies a sequence of operations to a number -/
def applyOperations (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl applyOperation start

/-- Theorem stating that it's impossible to obtain the same number from 19 and 98
    using the same number of operations -/
theorem no_equal_result_from_19_and_98 :
  ¬ ∃ (ops1 ops2 : List Operation) (result : ℕ),
    ops1.length = ops2.length ∧
    applyOperations 19 ops1 = result ∧
    applyOperations 98 ops2 = result :=
  sorry


end NUMINAMATH_CALUDE_no_equal_result_from_19_and_98_l2591_259161


namespace NUMINAMATH_CALUDE_calculation_proof_l2591_259155

theorem calculation_proof :
  (1/5 - 2/3 - 3/10) * (-60) = 46 ∧
  (-1)^2024 + 24 / (-2)^3 - 15^2 * (1/15)^2 = -3 :=
by sorry

end NUMINAMATH_CALUDE_calculation_proof_l2591_259155


namespace NUMINAMATH_CALUDE_contractor_payment_l2591_259119

/-- Calculates the total amount a contractor receives given the contract terms and attendance. -/
def contractorPay (totalDays : ℕ) (payPerDay : ℚ) (finePerDay : ℚ) (absentDays : ℕ) : ℚ :=
  let workDays := totalDays - absentDays
  let totalPay := (workDays : ℚ) * payPerDay
  let totalFine := (absentDays : ℚ) * finePerDay
  totalPay - totalFine

/-- Proves that under the given conditions, the contractor receives Rs. 425. -/
theorem contractor_payment :
  contractorPay 30 25 7.50 10 = 425 := by
  sorry

end NUMINAMATH_CALUDE_contractor_payment_l2591_259119


namespace NUMINAMATH_CALUDE_coral_population_decline_l2591_259171

/-- The yearly decrease rate of the coral population -/
def decrease_rate : ℝ := 0.25

/-- The threshold below which we consider the population critically low -/
def critical_threshold : ℝ := 0.05

/-- The number of years it takes for the population to fall below the critical threshold -/
def years_to_critical : ℕ := 9

/-- The remaining population after n years -/
def population_after (n : ℕ) : ℝ := (1 - decrease_rate) ^ n

theorem coral_population_decline :
  population_after years_to_critical < critical_threshold :=
sorry

end NUMINAMATH_CALUDE_coral_population_decline_l2591_259171


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l2591_259132

/-- The area of a rhombus inscribed in a circle, which is in turn inscribed in a square -/
theorem rhombus_area_in_square (s : ℝ) (h : s = 16) : 
  let r := s / 2
  let d := s
  let rhombus_area := d * d / 2
  rhombus_area = 128 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l2591_259132


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l2591_259140

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def f_domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), (∀ x, f_domain x → f_inv (f x) = x) ∧ f_inv 3 = -1 :=
sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l2591_259140


namespace NUMINAMATH_CALUDE_equal_savings_time_l2591_259120

def sara_initial_savings : ℕ := 4100
def sara_weekly_savings : ℕ := 10
def jim_weekly_savings : ℕ := 15

theorem equal_savings_time : 
  ∃ w : ℕ, w = 820 ∧ 
  sara_initial_savings + sara_weekly_savings * w = jim_weekly_savings * w :=
by sorry

end NUMINAMATH_CALUDE_equal_savings_time_l2591_259120


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2591_259103

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : a ≥ b) (h5 : c^2 = a^2 + b^2) : 
  a + b / 2 > c ∧ c > 8 / 9 * (a + b / 2) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2591_259103


namespace NUMINAMATH_CALUDE_almeriense_polynomial_characterization_l2591_259196

/-- A polynomial is almeriense if it has the form x³ + ax² + bx + a
    and its three roots are positive real numbers in arithmetic progression. -/
def IsAlmeriense (p : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ),
    (∀ x, p x = x^3 + a*x^2 + b*x + a) ∧
    (∃ (r₁ r₂ r₃ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0 ∧
      r₂ - r₁ = r₃ - r₂ ∧
      (∀ x, p x = (x - r₁) * (x - r₂) * (x - r₃)))

theorem almeriense_polynomial_characterization :
  ∀ p : ℝ → ℝ,
    IsAlmeriense p →
    p (7/4) = 0 →
    ((∀ x, p x = x^3 - (21/4)*x^2 + (73/8)*x - 21/4) ∨
     (∀ x, p x = x^3 - (291/56)*x^2 + (14113/1568)*x - 291/56)) :=
by sorry

end NUMINAMATH_CALUDE_almeriense_polynomial_characterization_l2591_259196


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2591_259151

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^2 + 7 * x - 3) * (3 * x^3 + 4) = 
  15 * x^5 + 21 * x^4 - 9 * x^3 + 20 * x^2 + 28 * x - 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2591_259151


namespace NUMINAMATH_CALUDE_largest_solution_and_fraction_l2591_259172

theorem largest_solution_and_fraction (a b c d : ℤ) : 
  (∃ x : ℚ, (5 * x) / 6 + 1 = 3 / x ∧ 
             x = (a + b * Real.sqrt c) / d ∧ 
             ∀ y : ℚ, (5 * y) / 6 + 1 = 3 / y → y ≤ x) →
  a * c * d / b = -55 := by
  sorry

end NUMINAMATH_CALUDE_largest_solution_and_fraction_l2591_259172


namespace NUMINAMATH_CALUDE_eeshas_travel_time_l2591_259164

/-- Eesha's travel time problem -/
theorem eeshas_travel_time 
  (usual_time : ℝ) 
  (usual_speed : ℝ) 
  (late_start : ℝ) 
  (late_arrival : ℝ) 
  (speed_reduction : ℝ) 
  (h1 : late_start = 30) 
  (h2 : late_arrival = 50) 
  (h3 : speed_reduction = 0.25) 
  (h4 : usual_time / (usual_time + late_arrival) = (1 - speed_reduction)) :
  usual_time = 150 := by
sorry

end NUMINAMATH_CALUDE_eeshas_travel_time_l2591_259164


namespace NUMINAMATH_CALUDE_gcd_lcm_product_360_l2591_259102

theorem gcd_lcm_product_360 (x y : ℕ+) : 
  (Nat.gcd x y * Nat.lcm x y = 360) → 
  (∃ (s : Finset ℕ), s.card = 8 ∧ ∀ (d : ℕ), d ∈ s ↔ ∃ (a b : ℕ+), Nat.gcd a b * Nat.lcm a b = 360 ∧ Nat.gcd a b = d) :=
sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_360_l2591_259102


namespace NUMINAMATH_CALUDE_construct_triangle_from_symmetric_points_l2591_259113

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : Point := sorry

/-- Check if a triangle is acute-angled -/
def is_acute_angled (t : Triangle) : Prop := sorry

/-- The symmetric point of a given point with respect to a line segment -/
def symmetric_point (p : Point) (a : Point) (b : Point) : Point := sorry

/-- Theorem: Given three points that are symmetric to the orthocenter of an acute-angled triangle
    with respect to its sides, the triangle can be uniquely constructed -/
theorem construct_triangle_from_symmetric_points
  (A' B' C' : Point) :
  ∃! (t : Triangle),
    is_acute_angled t ∧
    A' = symmetric_point (orthocenter t) t.B t.C ∧
    B' = symmetric_point (orthocenter t) t.C t.A ∧
    C' = symmetric_point (orthocenter t) t.A t.B :=
sorry

end NUMINAMATH_CALUDE_construct_triangle_from_symmetric_points_l2591_259113


namespace NUMINAMATH_CALUDE_number_of_students_in_line_l2591_259181

/-- The number of students in a line with specific conditions -/
theorem number_of_students_in_line :
  ∀ (n : ℕ),
  (∃ (eunjung_position yoojung_position : ℕ),
    eunjung_position = 5 ∧
    yoojung_position = n ∧
    yoojung_position - eunjung_position = 9) →
  n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_students_in_line_l2591_259181


namespace NUMINAMATH_CALUDE_circle_area_radius_decrease_l2591_259156

theorem circle_area_radius_decrease (r : ℝ) (h : r > 0) :
  let A := π * r^2
  let A' := 0.64 * A
  let r' := Real.sqrt (A' / π)
  r' / r = 0.8 := by sorry

end NUMINAMATH_CALUDE_circle_area_radius_decrease_l2591_259156


namespace NUMINAMATH_CALUDE_n_has_9_digits_l2591_259199

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_minimal : ∀ m : ℕ, m > 0 → 30 ∣ m → (∃ k : ℕ, m^2 = k^3) → (∃ k : ℕ, m^3 = k^2) → n ≤ m

/-- The number of digits in n -/
def num_digits (x : ℕ) : ℕ := sorry

/-- Theorem stating that n has 9 digits -/
theorem n_has_9_digits : num_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_9_digits_l2591_259199


namespace NUMINAMATH_CALUDE_sum_always_positive_l2591_259108

-- Define a monotonically increasing odd function on ℝ
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Theorem statement
theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l2591_259108


namespace NUMINAMATH_CALUDE_parking_garage_problem_l2591_259175

theorem parking_garage_problem (first_level : ℕ) (second_level : ℕ) (third_level : ℕ) (fourth_level : ℕ) 
  (h1 : first_level = 90)
  (h2 : second_level = first_level + 8)
  (h3 : third_level = second_level + 12)
  (h4 : fourth_level = third_level - 9)
  (h5 : first_level + second_level + third_level + fourth_level - 299 = 100) : 
  ∃ (cars_parked : ℕ), cars_parked = 100 := by
sorry

end NUMINAMATH_CALUDE_parking_garage_problem_l2591_259175


namespace NUMINAMATH_CALUDE_quadratic_lower_bound_l2591_259138

theorem quadratic_lower_bound 
  (f : ℝ → ℝ) 
  (a b : ℤ) 
  (h1 : ∀ x, f x = x^2 + a*x + b) 
  (h2 : ∀ x, f x ≥ -9/10) : 
  ∀ x, f x ≥ -1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_lower_bound_l2591_259138


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l2591_259149

/-- Given two positive real numbers with sum 55, HCF 5, and LCM 120, 
    prove that the sum of their reciprocals is 11/120 -/
theorem sum_of_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (sum : a + b = 55) (hcf : Int.gcd (Int.floor a) (Int.floor b) = 5) 
  (lcm : Int.lcm (Int.floor a) (Int.floor b) = 120) : 
  1 / a + 1 / b = 11 / 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l2591_259149


namespace NUMINAMATH_CALUDE_octagon_side_length_l2591_259179

/-- The side length of a regular octagon with an area equal to the sum of the areas of three regular octagons with side lengths 3, 4, and 12 units is 13 units. -/
theorem octagon_side_length (a b c d : ℝ) : 
  a > 0 → b > 0 → c > 0 → d > 0 →
  (a = 3) → (b = 4) → (c = 12) →
  (d^2 = a^2 + b^2 + c^2) →
  d = 13 := by sorry

end NUMINAMATH_CALUDE_octagon_side_length_l2591_259179


namespace NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l2591_259177

theorem sqrt_eight_div_sqrt_two_equals_two : 
  Real.sqrt 8 / Real.sqrt 2 = 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_eight_div_sqrt_two_equals_two_l2591_259177


namespace NUMINAMATH_CALUDE_percentage_of_returned_books_l2591_259116

/-- Given a library's special collection with initial and final book counts,
    and the number of books loaned out, prove the percentage of returned books. -/
theorem percentage_of_returned_books
  (initial_books : ℕ)
  (final_books : ℕ)
  (loaned_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : final_books = 69)
  (h3 : loaned_books = 30) :
  (initial_books - final_books : ℚ) / loaned_books * 100 = 20 := by
  sorry

#check percentage_of_returned_books

end NUMINAMATH_CALUDE_percentage_of_returned_books_l2591_259116


namespace NUMINAMATH_CALUDE_percentage_increase_l2591_259101

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 350 → final = 525 → (final - initial) / initial * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l2591_259101


namespace NUMINAMATH_CALUDE_correct_calculation_l2591_259142

theorem correct_calculation (x : ℚ) (h : x + 7/5 = 81/20) : x - 7/5 = 25/20 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2591_259142


namespace NUMINAMATH_CALUDE_darnel_sprint_distance_l2591_259159

theorem darnel_sprint_distance (jogged_distance : Real) (additional_sprint : Real) :
  jogged_distance = 0.75 →
  additional_sprint = 0.125 →
  jogged_distance + additional_sprint = 0.875 := by
  sorry

end NUMINAMATH_CALUDE_darnel_sprint_distance_l2591_259159


namespace NUMINAMATH_CALUDE_abc_divisibility_problem_l2591_259173

theorem abc_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end NUMINAMATH_CALUDE_abc_divisibility_problem_l2591_259173


namespace NUMINAMATH_CALUDE_cube_of_sum_fractions_is_three_l2591_259154

theorem cube_of_sum_fractions_is_three (a b c : ℤ) 
  (h : (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a = 3) : 
  ∃ n : ℤ, a * b * c = n^3 := by
sorry

end NUMINAMATH_CALUDE_cube_of_sum_fractions_is_three_l2591_259154


namespace NUMINAMATH_CALUDE_det_skew_symmetric_nonneg_l2591_259125

/-- A 4x4 real matrix is skew-symmetric if its transpose is equal to its negation. -/
def isSkewSymmetric (A : Matrix (Fin 4) (Fin 4) ℝ) : Prop :=
  A.transpose = -A

/-- The determinant of a 4x4 real skew-symmetric matrix is non-negative. -/
theorem det_skew_symmetric_nonneg (A : Matrix (Fin 4) (Fin 4) ℝ) 
  (h : isSkewSymmetric A) : 0 ≤ A.det := by
  sorry

end NUMINAMATH_CALUDE_det_skew_symmetric_nonneg_l2591_259125


namespace NUMINAMATH_CALUDE_computer_multiplications_l2591_259126

theorem computer_multiplications (multiplications_per_second : ℕ) (hours : ℕ) : 
  multiplications_per_second = 15000 → 
  hours = 3 → 
  multiplications_per_second * (hours * 3600) = 162000000 := by
  sorry

end NUMINAMATH_CALUDE_computer_multiplications_l2591_259126


namespace NUMINAMATH_CALUDE_prob_sum_five_or_nine_l2591_259194

def fair_dice_roll : Finset ℕ := Finset.range 6

def sum_outcomes (n : ℕ) : Finset (ℕ × ℕ) :=
  (fair_dice_roll.product fair_dice_roll).filter (fun p => p.1 + p.2 + 2 = n)

def prob_sum (n : ℕ) : ℚ :=
  (sum_outcomes n).card / (fair_dice_roll.card * fair_dice_roll.card : ℕ)

theorem prob_sum_five_or_nine :
  prob_sum 5 = 1/9 ∧ prob_sum 9 = 1/9 :=
sorry

end NUMINAMATH_CALUDE_prob_sum_five_or_nine_l2591_259194


namespace NUMINAMATH_CALUDE_zoo_animal_ratio_l2591_259178

/-- Given the animal counts at San Diego Zoo, prove the ratio of bee-eaters to leopards -/
theorem zoo_animal_ratio :
  let total_animals : ℕ := 670
  let snakes : ℕ := 100
  let arctic_foxes : ℕ := 80
  let leopards : ℕ := 20
  let cheetahs : ℕ := snakes / 2
  let alligators : ℕ := 2 * (arctic_foxes + leopards)
  let bee_eaters : ℕ := total_animals - (snakes + arctic_foxes + leopards + cheetahs + alligators)
  (bee_eaters : ℚ) / leopards = 11 / 1 := by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_ratio_l2591_259178


namespace NUMINAMATH_CALUDE_min_weighings_for_extremes_l2591_259144

/-- Represents a coin with a weight -/
structure Coin where
  weight : ℕ

/-- Represents a weighing operation that compares two coins -/
def weighing (a b : Coin) : Bool :=
  a.weight > b.weight

theorem min_weighings_for_extremes (coins : List Coin) : 
  coins.length = 68 → (∃ n : ℕ, n = 100 ∧ 
    (∀ m : ℕ, m < n → ¬(∃ heaviest lightest : Coin, 
      heaviest ∈ coins ∧ lightest ∈ coins ∧
      (∀ c : Coin, c ∈ coins → c.weight ≤ heaviest.weight) ∧
      (∀ c : Coin, c ∈ coins → c.weight ≥ lightest.weight) ∧
      (heaviest ≠ lightest)))) :=
by
  sorry

end NUMINAMATH_CALUDE_min_weighings_for_extremes_l2591_259144


namespace NUMINAMATH_CALUDE_min_distance_from_point_on_unit_circle_l2591_259191

theorem min_distance_from_point_on_unit_circle (z : ℂ) (h : Complex.abs z = 1) :
  Complex.abs (z - (3 + 4 * Complex.I)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_from_point_on_unit_circle_l2591_259191


namespace NUMINAMATH_CALUDE_sum_of_possible_numbers_l2591_259148

/-- The original number from which we remove digits -/
def original_number : ℕ := 112277

/-- The set of all possible three-digit numbers obtained by removing three digits from the original number -/
def possible_numbers : Finset ℕ := {112, 117, 122, 127, 177, 227, 277}

/-- The theorem stating that the sum of all possible three-digit numbers is 1159 -/
theorem sum_of_possible_numbers : 
  (possible_numbers.sum id) = 1159 := by sorry

end NUMINAMATH_CALUDE_sum_of_possible_numbers_l2591_259148


namespace NUMINAMATH_CALUDE_equality_of_solution_sets_implies_sum_l2591_259198

theorem equality_of_solution_sets_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, |x - 2| > 1 ↔ x^2 + a*x + b > 0) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_solution_sets_implies_sum_l2591_259198


namespace NUMINAMATH_CALUDE_factorization_proof_l2591_259190

theorem factorization_proof :
  (∀ x : ℝ, 4 * x^2 - 36 = 4 * (x + 3) * (x - 3)) ∧
  (∀ x y : ℝ, x^3 - 2 * x^2 * y + x * y^2 = x * (x - y)^2) := by
sorry

end NUMINAMATH_CALUDE_factorization_proof_l2591_259190


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2591_259129

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2591_259129


namespace NUMINAMATH_CALUDE_max_value_problem_l2591_259183

theorem max_value_problem (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 2) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 2) : 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 := by
  sorry

end NUMINAMATH_CALUDE_max_value_problem_l2591_259183


namespace NUMINAMATH_CALUDE_positive_sum_reciprocal_inequality_l2591_259182

theorem positive_sum_reciprocal_inequality (p : ℝ) (hp : p > 0) :
  p + 1/p > 2 ↔ p ≠ 1 := by sorry

end NUMINAMATH_CALUDE_positive_sum_reciprocal_inequality_l2591_259182


namespace NUMINAMATH_CALUDE_common_measure_of_angles_l2591_259192

-- Define the angles and natural numbers
variable (α β : ℝ)
variable (m n : ℕ)

-- State the theorem
theorem common_measure_of_angles (h : α = β * (m / n)) :
  α / m = β / n ∧ 
  ∃ (k₁ k₂ : ℕ), α = k₁ * (α / m) ∧ β = k₂ * (β / n) :=
sorry

end NUMINAMATH_CALUDE_common_measure_of_angles_l2591_259192


namespace NUMINAMATH_CALUDE_frame_interior_edges_sum_l2591_259100

/-- Represents a rectangular picture frame -/
structure Frame where
  outer_length : ℝ
  outer_width : ℝ
  frame_width : ℝ

/-- Calculates the area of the frame -/
def frame_area (f : Frame) : ℝ :=
  f.outer_length * f.outer_width - (f.outer_length - 2 * f.frame_width) * (f.outer_width - 2 * f.frame_width)

/-- Calculates the sum of the lengths of the four interior edges of the frame -/
def interior_edges_sum (f : Frame) : ℝ :=
  2 * ((f.outer_length - 2 * f.frame_width) + (f.outer_width - 2 * f.frame_width))

/-- Theorem stating that for a frame with given conditions, the sum of interior edges is 8 inches -/
theorem frame_interior_edges_sum :
  ∀ (f : Frame),
    f.frame_width = 2 →
    f.outer_length = 8 →
    frame_area f = 32 →
    interior_edges_sum f = 8 := by
  sorry

end NUMINAMATH_CALUDE_frame_interior_edges_sum_l2591_259100


namespace NUMINAMATH_CALUDE_incorrect_transformation_l2591_259130

theorem incorrect_transformation (a b : ℝ) :
  ¬(∀ a b : ℝ, a = b → a / b = 1) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l2591_259130


namespace NUMINAMATH_CALUDE_cubic_equation_solution_product_l2591_259133

theorem cubic_equation_solution_product (d e f : ℝ) : 
  d^3 + 2*d^2 + 3*d - 5 = 0 ∧ 
  e^3 + 2*e^2 + 3*e - 5 = 0 ∧ 
  f^3 + 2*f^2 + 3*f - 5 = 0 → 
  (d - 1) * (e - 1) * (f - 1) = 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_product_l2591_259133


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2591_259117

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x + 3| < 1} = {x : ℝ | -4 < x ∧ x < -2} := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2591_259117


namespace NUMINAMATH_CALUDE_fraction_subtraction_problem_l2591_259123

theorem fraction_subtraction_problem : (1/2 : ℚ) + 5/6 - 2/3 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_problem_l2591_259123


namespace NUMINAMATH_CALUDE_max_students_distribution_l2591_259152

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 1001) (h2 : pencils = 910) : ℕ :=
  Nat.gcd pens pencils

#check max_students_distribution

end NUMINAMATH_CALUDE_max_students_distribution_l2591_259152


namespace NUMINAMATH_CALUDE_ice_skate_profit_maximization_l2591_259143

/-- Ice skate problem -/
theorem ice_skate_profit_maximization
  (cost_A cost_B : ℕ)  -- Cost prices of type A and B
  (sell_A sell_B : ℕ)  -- Selling prices of type A and B
  (total_pairs : ℕ)    -- Total number of pairs to purchase
  : cost_B = 2 * cost_A  -- Condition 1
  → 2 * cost_A + cost_B = 920  -- Condition 2
  → sell_A = 400  -- Condition 3
  → sell_B = 560  -- Condition 4
  → total_pairs = 50  -- Condition 5
  → (∀ x y : ℕ, x + y = total_pairs → x ≤ 2 * y)  -- Condition 6
  → ∃ (x y : ℕ),
      x + y = total_pairs ∧
      x = 33 ∧
      y = 17 ∧
      x * (sell_A - cost_A) + y * (sell_B - cost_B) = 6190 ∧
      ∀ (a b : ℕ), a + b = total_pairs →
        a * (sell_A - cost_A) + b * (sell_B - cost_B) ≤ 6190 :=
by sorry

end NUMINAMATH_CALUDE_ice_skate_profit_maximization_l2591_259143


namespace NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l2591_259104

theorem min_hypotenuse_right_triangle (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a + b = 10) 
  (h5 : c^2 = a^2 + b^2) : 
  c ≥ 5 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_hypotenuse_right_triangle_l2591_259104


namespace NUMINAMATH_CALUDE_race_head_start_l2591_259112

/-- Proves that the head start in a race is equal to the difference in distances covered by two runners with different speeds in a given time. -/
theorem race_head_start (cristina_speed nicky_speed : ℝ) (race_time : ℝ) 
  (h1 : cristina_speed > nicky_speed) 
  (h2 : cristina_speed = 4)
  (h3 : nicky_speed = 3)
  (h4 : race_time = 36) :
  cristina_speed * race_time - nicky_speed * race_time = 36 := by
  sorry

#check race_head_start

end NUMINAMATH_CALUDE_race_head_start_l2591_259112


namespace NUMINAMATH_CALUDE_parentheses_expression_l2591_259187

theorem parentheses_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ∃ z : ℝ, x * y * z = -x^3 * y^2 → z = -x^2 * y :=
sorry

end NUMINAMATH_CALUDE_parentheses_expression_l2591_259187


namespace NUMINAMATH_CALUDE_P_bounds_l2591_259180

/-- Represents the minimum number of transformations needed to convert
    any triangulation of a convex n-gon to any other triangulation. -/
def P (n : ℕ) : ℕ := sorry

/-- The main theorem about the bounds of P(n) -/
theorem P_bounds (n : ℕ) : 
  (n ≥ 3 → P n ≥ n - 3) ∧ 
  (n ≥ 3 → P n ≤ 2*n - 7) ∧ 
  (n ≥ 13 → P n ≤ 2*n - 10) := by
  sorry

end NUMINAMATH_CALUDE_P_bounds_l2591_259180


namespace NUMINAMATH_CALUDE_divisibility_property_l2591_259157

theorem divisibility_property (x y a b S : ℤ) 
  (sum_eq : x + y = S) 
  (masha_divisible : S ∣ (a * x + b * y)) : 
  S ∣ (b * x + a * y) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2591_259157


namespace NUMINAMATH_CALUDE_fraction_difference_l2591_259131

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : 1 / a - 1 / b = -2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_difference_l2591_259131


namespace NUMINAMATH_CALUDE_num_sequences_mod_1000_l2591_259167

/-- The number of increasing sequences of positive integers satisfying the given conditions -/
def num_sequences : ℕ := sorry

/-- The upper bound for the sequence elements -/
def upper_bound : ℕ := 1007

/-- The length of the sequences -/
def sequence_length : ℕ := 12

/-- Predicate to check if a sequence satisfies the given conditions -/
def valid_sequence (b : Fin sequence_length → ℕ) : Prop :=
  (∀ i j : Fin sequence_length, i ≤ j → b i ≤ b j) ∧
  (∀ i : Fin sequence_length, b i ≤ upper_bound) ∧
  (∀ i : Fin sequence_length, Even (b i - i.val))

theorem num_sequences_mod_1000 :
  num_sequences % 1000 = 508 := by sorry

end NUMINAMATH_CALUDE_num_sequences_mod_1000_l2591_259167


namespace NUMINAMATH_CALUDE_simplify_irrational_denominator_l2591_259193

theorem simplify_irrational_denominator :
  (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 2) = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_irrational_denominator_l2591_259193


namespace NUMINAMATH_CALUDE_diego_yearly_savings_l2591_259128

/-- Calculates the yearly savings given monthly deposit, monthly expenses, and number of months in a year. -/
def yearly_savings (monthly_deposit : ℕ) (monthly_expenses : ℕ) (months_in_year : ℕ) : ℕ :=
  (monthly_deposit - monthly_expenses) * months_in_year

/-- Theorem stating that Diego's yearly savings is $4,800 -/
theorem diego_yearly_savings :
  yearly_savings 5000 4600 12 = 4800 := by
  sorry

end NUMINAMATH_CALUDE_diego_yearly_savings_l2591_259128


namespace NUMINAMATH_CALUDE_set_equality_invariant_under_variable_renaming_l2591_259109

theorem set_equality_invariant_under_variable_renaming :
  {x : ℝ | x ≤ 1} = {t : ℝ | t ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_set_equality_invariant_under_variable_renaming_l2591_259109


namespace NUMINAMATH_CALUDE_difference_of_squares_example_l2591_259114

theorem difference_of_squares_example : (538 * 538) - (537 * 539) = 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_example_l2591_259114


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l2591_259184

theorem min_value_quadratic_form (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) :
  |5 * x^2 + 11 * x * y - 5 * y^2| ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l2591_259184


namespace NUMINAMATH_CALUDE_disc_price_calculation_l2591_259118

/-- The price of the other type of compact disc -/
def other_disc_price : ℝ := 10.50

theorem disc_price_calculation (total_discs : ℕ) (total_spent : ℝ) (known_price : ℝ) (known_quantity : ℕ) :
  total_discs = 10 →
  total_spent = 93 →
  known_price = 8.50 →
  known_quantity = 6 →
  other_disc_price = (total_spent - known_price * known_quantity) / (total_discs - known_quantity) :=
by
  sorry

#eval other_disc_price

end NUMINAMATH_CALUDE_disc_price_calculation_l2591_259118


namespace NUMINAMATH_CALUDE_x_convergence_to_sqrt2_l2591_259170

-- Define the sequence x_n
def x : ℕ → ℚ
| 0 => 1
| (n+1) => 1 + 1 / (2 + 1 / (x n))

-- Define the bound function
def bound (n : ℕ) : ℚ := 1 / 2^(2^n - 1)

-- State the theorem
theorem x_convergence_to_sqrt2 (n : ℕ) :
  |x n - Real.sqrt 2| < bound n :=
sorry

end NUMINAMATH_CALUDE_x_convergence_to_sqrt2_l2591_259170


namespace NUMINAMATH_CALUDE_car_original_price_l2591_259127

/-- 
Given a car sale scenario where:
1. A car is sold at a 10% loss to a friend
2. The friend sells it for Rs. 54000 with a 20% gain

This theorem proves that the original cost price of the car was Rs. 50000.
-/
theorem car_original_price : ℝ → Prop :=
  fun original_price =>
    let friend_buying_price := 0.9 * original_price
    let friend_selling_price := 54000
    (1.2 * friend_buying_price = friend_selling_price) →
    (original_price = 50000)

-- The proof is omitted
example : car_original_price 50000 := by sorry

end NUMINAMATH_CALUDE_car_original_price_l2591_259127


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2591_259139

theorem quadratic_factorization (y a b : ℤ) : 
  (3 * y^2 - 7 * y - 6 = (3 * y + a) * (y + b)) → (a - b = 5) := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2591_259139


namespace NUMINAMATH_CALUDE_agricultural_experiment_l2591_259122

theorem agricultural_experiment (seeds_second_plot : ℕ) : 
  (300 : ℝ) * 0.30 + seeds_second_plot * 0.35 = (300 + seeds_second_plot) * 0.32 →
  seeds_second_plot = 200 := by
sorry

end NUMINAMATH_CALUDE_agricultural_experiment_l2591_259122


namespace NUMINAMATH_CALUDE_function_value_at_two_l2591_259153

/-- Given a function f(x) = x^5 + px^3 + qx - 8 where f(-2) = 10, prove that f(2) = -26 -/
theorem function_value_at_two (p q : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^5 + p*x^3 + q*x - 8)
  (h2 : f (-2) = 10) : 
  f 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_two_l2591_259153


namespace NUMINAMATH_CALUDE_probability_one_of_each_l2591_259124

/-- The number of forks in the drawer -/
def num_forks : ℕ := 7

/-- The number of spoons in the drawer -/
def num_spoons : ℕ := 8

/-- The number of knives in the drawer -/
def num_knives : ℕ := 5

/-- The total number of pieces of silverware -/
def total_pieces : ℕ := num_forks + num_spoons + num_knives

/-- The number of pieces to be selected -/
def num_selected : ℕ := 3

/-- The probability of selecting one fork, one spoon, and one knife -/
theorem probability_one_of_each : 
  (num_forks * num_spoons * num_knives : ℚ) / (Nat.choose total_pieces num_selected) = 14 / 57 := by
  sorry

end NUMINAMATH_CALUDE_probability_one_of_each_l2591_259124


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2591_259110

-- Problem 1
theorem problem_1 : 2 * Real.sqrt 12 - 6 * Real.sqrt (1/3) + 3 * Real.sqrt 48 = 14 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (hx : x > 0) : 
  (2/3) * Real.sqrt (9*x) + 6 * Real.sqrt (x/4) - x * Real.sqrt (1/x) = 4 * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2591_259110
