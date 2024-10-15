import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solutions_l2202_220256

theorem inequality_solutions :
  (∀ x : ℝ, 2 + 3*x - 2*x^2 > 0 ↔ -1/2 < x ∧ x < 2) ∧
  (∀ x : ℝ, x*(3-x) ≤ x*(x+2) - 1 ↔ x ≤ -1/2 ∨ x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solutions_l2202_220256


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2202_220231

theorem absolute_value_inequality (b : ℝ) (h₁ : b > 0) :
  (∃ x : ℝ, |2*x - 8| + |2*x - 6| < b) → b > 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2202_220231


namespace NUMINAMATH_CALUDE_inverse_composition_l2202_220245

-- Define the function f
def f : ℕ → ℕ
| 3 => 10
| 4 => 17
| 5 => 26
| 6 => 37
| 7 => 50
| _ => 0  -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 10 => 3
| 17 => 4
| 26 => 5
| 37 => 6
| 50 => 7
| _ => 0  -- Default case for other inputs

-- Theorem statement
theorem inverse_composition :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_composition_l2202_220245


namespace NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l2202_220261

def arithmetic_sequence (a₁ a₂ a₃ : ℤ) : Prop :=
  a₂ - a₁ = a₃ - a₂

theorem second_term_of_arithmetic_sequence :
  ∀ y : ℤ, arithmetic_sequence (3^2) y (3^4) → y = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_of_arithmetic_sequence_l2202_220261


namespace NUMINAMATH_CALUDE_fourth_root_equation_implies_x_power_eight_zero_l2202_220215

theorem fourth_root_equation_implies_x_power_eight_zero (x : ℝ) :
  (((1 - x^4 : ℝ)^(1/4) + (1 + x^4 : ℝ)^(1/4)) = 2) → x^8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_equation_implies_x_power_eight_zero_l2202_220215


namespace NUMINAMATH_CALUDE_polynomial_remainder_l2202_220243

theorem polynomial_remainder (x : ℝ) : 
  (8 * x^3 - 20 * x^2 + 28 * x - 30) % (4 * x - 8) = 10 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l2202_220243


namespace NUMINAMATH_CALUDE_problem_solution_l2202_220272

theorem problem_solution (p_xavier p_yvonne p_zelda p_wendell : ℚ)
  (h_xavier : p_xavier = 1/4)
  (h_yvonne : p_yvonne = 1/3)
  (h_zelda : p_zelda = 5/8)
  (h_wendell : p_wendell = 1/2) :
  p_xavier * p_yvonne * (1 - p_zelda) * (1 - p_wendell) = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2202_220272


namespace NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2202_220263

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The number of distinct interior intersection points of diagonals in a regular decagon -/
def intersection_points (n : ℕ) : ℕ := Nat.choose n 4

theorem decagon_diagonal_intersections :
  intersection_points n = 210 :=
sorry

end NUMINAMATH_CALUDE_decagon_diagonal_intersections_l2202_220263


namespace NUMINAMATH_CALUDE_continuous_function_characterization_l2202_220251

theorem continuous_function_characterization
  (f : ℝ → ℝ)
  (hf_continuous : Continuous f)
  (hf_zero : f 0 = 0)
  (hf_ineq : ∀ x y : ℝ, f (x^2 - y^2) ≥ x * f x - y * f y) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
sorry

end NUMINAMATH_CALUDE_continuous_function_characterization_l2202_220251


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2202_220299

/-- Given two vectors OA and OB in 2D space, if they are perpendicular,
    then the second component of OB is 3/2. -/
theorem perpendicular_vectors_m_value (OA OB : ℝ × ℝ) :
  OA = (-1, 2) → OB.1 = 3 → OA.1 * OB.1 + OA.2 * OB.2 = 0 → OB.2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2202_220299


namespace NUMINAMATH_CALUDE_A_eq_set_zero_one_two_l2202_220205

def A : Set ℤ := {x | -1 < |x - 1| ∧ |x - 1| < 2}

theorem A_eq_set_zero_one_two : A = {0, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_eq_set_zero_one_two_l2202_220205


namespace NUMINAMATH_CALUDE_theresa_crayons_l2202_220207

/-- Theresa's initial number of crayons -/
def theresa_initial : ℕ := sorry

/-- Theresa's number of crayons after sharing -/
def theresa_after : ℕ := 19

/-- Janice's initial number of crayons -/
def janice_initial : ℕ := 12

/-- Number of crayons Janice shares with Nancy -/
def janice_shares : ℕ := 13

theorem theresa_crayons : theresa_initial = theresa_after := by sorry

end NUMINAMATH_CALUDE_theresa_crayons_l2202_220207


namespace NUMINAMATH_CALUDE_proposition_truth_l2202_220252

theorem proposition_truth : 
  -- Proposition A
  (∃ a b m : ℝ, a < b ∧ ¬(a * m^2 < b * m^2)) ∧
  -- Proposition B
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) ∧
  -- Proposition C
  (∃ x : ℝ, x^2 = 9 ∧ x ≠ 3) ∧
  -- Proposition D
  ((∀ x : ℝ, x > 1 → 1/x < 1) ∧ (∃ x : ℝ, 1/x < 1 ∧ ¬(x > 1))) :=
by sorry

end NUMINAMATH_CALUDE_proposition_truth_l2202_220252


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l2202_220294

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of the number we want to convert -/
def binary_number : List Bool := [true, true, false, false, true, true]

/-- Theorem stating that the binary number 110011 is equal to the decimal number 51 -/
theorem binary_110011_equals_51 :
  binary_to_decimal (binary_number.reverse) = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l2202_220294


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l2202_220217

theorem partial_fraction_decomposition :
  ∃ (A B C : ℝ), ∀ (x : ℝ), x ≠ 0 → x^2 + 1 ≠ 0 →
    (-x^2 + 3*x - 4) / (x^3 + x) = A / x + (B*x + C) / (x^2 + 1) ∧
    A = -4 ∧ B = 3 ∧ C = 3 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l2202_220217


namespace NUMINAMATH_CALUDE_single_digit_integer_equation_l2202_220268

theorem single_digit_integer_equation : ∃ (x a y z b : ℕ),
  (0 < x ∧ x < 10) ∧
  (0 < a ∧ a < 10) ∧
  (0 < y ∧ y < 10) ∧
  (0 < z ∧ z < 10) ∧
  (0 < b ∧ b < 10) ∧
  (x = a / 6) ∧
  (z = b / 6) ∧
  (y = (a + b) % 5) ∧
  (100 * x + 10 * y + z = 121) :=
by
  sorry

end NUMINAMATH_CALUDE_single_digit_integer_equation_l2202_220268


namespace NUMINAMATH_CALUDE_race_speed_ratio_l2202_220209

theorem race_speed_ratio (total_distance : ℝ) (head_start : ℝ) (speed_A : ℝ) (speed_B : ℝ) 
  (h1 : total_distance = 128)
  (h2 : head_start = 64)
  (h3 : total_distance / speed_A = (total_distance - head_start) / speed_B) :
  speed_A / speed_B = 2 := by
  sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l2202_220209


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2202_220202

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

/-- Definition of the major axis length -/
def major_axis_length (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: The length of the major axis of the ellipse is 6 -/
theorem ellipse_major_axis_length :
  major_axis_length is_ellipse = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2202_220202


namespace NUMINAMATH_CALUDE_triangle_area_on_grid_l2202_220221

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

theorem triangle_area_on_grid :
  let A : Point := { x := 0, y := 0 }
  let B : Point := { x := 2, y := 0 }
  let C : Point := { x := 2, y := 2.5 }
  triangleArea A B C = 2.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_on_grid_l2202_220221


namespace NUMINAMATH_CALUDE_reflection_of_point_l2202_220246

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The x-axis reflection of a point -/
def reflect_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Theorem: The x-axis reflection of point (-2, 3) is (-2, -3) -/
theorem reflection_of_point :
  let P : Point := { x := -2, y := 3 }
  reflect_x P = { x := -2, y := -3 } := by
sorry

end NUMINAMATH_CALUDE_reflection_of_point_l2202_220246


namespace NUMINAMATH_CALUDE_harris_flour_amount_l2202_220218

theorem harris_flour_amount (flour_per_cake : ℕ) (total_cakes : ℕ) (traci_flour : ℕ) :
  flour_per_cake = 100 →
  total_cakes = 9 →
  traci_flour = 500 →
  flour_per_cake * total_cakes - traci_flour = 400 := by
sorry

end NUMINAMATH_CALUDE_harris_flour_amount_l2202_220218


namespace NUMINAMATH_CALUDE_intersection_ratio_l2202_220244

-- Define the slopes and y-intercepts of the two lines
variable (k₁ k₂ : ℝ)

-- Define the condition that the lines intersect on the x-axis
def intersect_on_x_axis (k₁ k₂ : ℝ) : Prop :=
  ∃ x : ℝ, k₁ * x + 4 = 0 ∧ k₂ * x - 2 = 0

-- Theorem statement
theorem intersection_ratio (k₁ k₂ : ℝ) (h : intersect_on_x_axis k₁ k₂) (h₁ : k₁ ≠ 0) (h₂ : k₂ ≠ 0) :
  k₁ / k₂ = -2 :=
sorry

end NUMINAMATH_CALUDE_intersection_ratio_l2202_220244


namespace NUMINAMATH_CALUDE_change_received_l2202_220203

/-- The change received when buying gum and a protractor -/
theorem change_received (gum_cost protractor_cost amount_paid : ℕ) : 
  gum_cost = 350 → protractor_cost = 500 → amount_paid = 1000 → 
  amount_paid - (gum_cost + protractor_cost) = 150 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l2202_220203


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2202_220281

/-- Theorem: In a geometric sequence where a₅ = 4 and a₇ = 6, a₉ = 9 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + m) = a n * a m) →  -- geometric sequence property
  a 5 = 4 →
  a 7 = 6 →
  a 9 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2202_220281


namespace NUMINAMATH_CALUDE_nth_prime_power_bound_l2202_220234

/-- p_nth n returns the n-th prime number -/
def p_nth : ℕ → ℕ := sorry

/-- Theorem stating that for any positive integers n and k, 
    n is less than the k-th power of the 2k-th prime number -/
theorem nth_prime_power_bound (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : 
  n < (p_nth (2 * k)) ^ k := by sorry

end NUMINAMATH_CALUDE_nth_prime_power_bound_l2202_220234


namespace NUMINAMATH_CALUDE_custom_op_three_six_l2202_220230

/-- Custom operation @ for positive integers -/
def custom_op (a b : ℕ+) : ℚ :=
  (a.val ^ 2 * b.val : ℚ) / (a.val + b.val)

/-- Theorem stating that 3 @ 6 = 6 -/
theorem custom_op_three_six :
  custom_op 3 6 = 6 := by sorry

end NUMINAMATH_CALUDE_custom_op_three_six_l2202_220230


namespace NUMINAMATH_CALUDE_complex_arithmetic_equation_l2202_220293

theorem complex_arithmetic_equation : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equation_l2202_220293


namespace NUMINAMATH_CALUDE_yoki_cans_count_l2202_220280

def total_cans : ℕ := 85
def ladonna_cans : ℕ := 25
def prikya_cans : ℕ := 2 * ladonna_cans
def avi_initial_cans : ℕ := 8
def avi_remaining_cans : ℕ := avi_initial_cans / 2

theorem yoki_cans_count : 
  total_cans - (ladonna_cans + prikya_cans + avi_remaining_cans) = 6 := by
  sorry

end NUMINAMATH_CALUDE_yoki_cans_count_l2202_220280


namespace NUMINAMATH_CALUDE_total_bags_delivered_l2202_220298

-- Define the problem parameters
def bags_per_trip_light : ℕ := 15
def bags_per_trip_heavy : ℕ := 20
def total_days : ℕ := 7
def trips_per_day_light : ℕ := 25
def trips_per_day_heavy : ℕ := 18
def days_with_light_bags : ℕ := 3
def days_with_heavy_bags : ℕ := 4

-- Define the theorem
theorem total_bags_delivered : 
  (days_with_light_bags * trips_per_day_light * bags_per_trip_light) +
  (days_with_heavy_bags * trips_per_day_heavy * bags_per_trip_heavy) = 2565 :=
by sorry

end NUMINAMATH_CALUDE_total_bags_delivered_l2202_220298


namespace NUMINAMATH_CALUDE_marjs_wallet_problem_l2202_220275

/-- Prove that given the conditions in Marj's wallet problem, the value of each of the two bills is $20. -/
theorem marjs_wallet_problem (bill_value : ℚ) : 
  (2 * bill_value + 3 * 5 + 4.5 = 42 + 17.5) → bill_value = 20 := by
  sorry

end NUMINAMATH_CALUDE_marjs_wallet_problem_l2202_220275


namespace NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l2202_220284

theorem greatest_common_divisor_under_30 : ∃ (d : ℕ), d = 18 ∧ 
  d ∣ 450 ∧ d ∣ 90 ∧ d < 30 ∧ 
  ∀ (x : ℕ), x ∣ 450 ∧ x ∣ 90 ∧ x < 30 → x ≤ d :=
by sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_under_30_l2202_220284


namespace NUMINAMATH_CALUDE_expression_equals_2x_to_4th_l2202_220242

theorem expression_equals_2x_to_4th (x : ℝ) :
  let A := x^4 * x^4
  let B := x^4 + x^4
  let C := 2*x^2 + x^2
  let D := 2*x * x^4
  B = 2 * x^4 := by sorry

end NUMINAMATH_CALUDE_expression_equals_2x_to_4th_l2202_220242


namespace NUMINAMATH_CALUDE_jeans_sale_savings_l2202_220274

/-- Calculates the total savings when purchasing jeans with given prices and discounts -/
def total_savings (fox_price pony_price : ℚ) (fox_discount pony_discount : ℚ) 
  (fox_quantity pony_quantity : ℕ) : ℚ :=
  let regular_total := fox_price * fox_quantity + pony_price * pony_quantity
  let discounted_total := (fox_price * (1 - fox_discount)) * fox_quantity + 
                          (pony_price * (1 - pony_discount)) * pony_quantity
  regular_total - discounted_total

/-- Theorem stating that the total savings is $18 under the given conditions -/
theorem jeans_sale_savings :
  let fox_price : ℚ := 15
  let pony_price : ℚ := 18
  let fox_quantity : ℕ := 3
  let pony_quantity : ℕ := 2
  let pony_discount : ℚ := 1/2
  let fox_discount : ℚ := 1/2 - pony_discount
  total_savings fox_price pony_price fox_discount pony_discount fox_quantity pony_quantity = 18 :=
by sorry

end NUMINAMATH_CALUDE_jeans_sale_savings_l2202_220274


namespace NUMINAMATH_CALUDE_fourth_power_sum_sqrt_div_two_l2202_220286

theorem fourth_power_sum_sqrt_div_two :
  Real.sqrt (4^4 + 4^4 + 4^4) / 2 = 8 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_sqrt_div_two_l2202_220286


namespace NUMINAMATH_CALUDE_mixture_ratio_l2202_220273

theorem mixture_ratio (p q : ℝ) : 
  p + q = 20 →
  p / (q + 1) = 4 / 3 →
  p / q = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_mixture_ratio_l2202_220273


namespace NUMINAMATH_CALUDE_circle_and_distance_l2202_220282

-- Define points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop :=
  (P.1 + 1)^2 + P.2^2 = 2 * ((P.1 - 1)^2 + P.2^2)

-- Define the circle C
def C : Set (ℝ × ℝ) :=
  {P | (P.1 - 3)^2 + P.2^2 = 8}

-- Define the parabola
def parabola : Set (ℝ × ℝ) :=
  {P | P.2^2 = P.1}

theorem circle_and_distance :
  (∀ P, P_condition P → P ∈ C) ∧
  (∃ Q ∈ parabola, ∀ R ∈ parabola, 
    dist (3, 0) Q ≤ dist (3, 0) R ∧ 
    dist (3, 0) Q = Real.sqrt 11 / 2) :=
sorry

end NUMINAMATH_CALUDE_circle_and_distance_l2202_220282


namespace NUMINAMATH_CALUDE_sum_due_proof_l2202_220257

/-- Banker's discount (BD) is the simple interest on the face value (FV) of a bill for the unexpired time -/
def bankers_discount (face_value : ℝ) : ℝ := 288

/-- True discount (TD) is the simple interest on the present value (PV) of the bill for the unexpired time -/
def true_discount (face_value : ℝ) : ℝ := 240

/-- The relationship between banker's discount, true discount, and face value -/
def discount_relationship (face_value : ℝ) : Prop :=
  bankers_discount face_value = true_discount face_value + (true_discount face_value)^2 / face_value

theorem sum_due_proof : 
  ∃ (face_value : ℝ), face_value = 1200 ∧ discount_relationship face_value := by
  sorry

end NUMINAMATH_CALUDE_sum_due_proof_l2202_220257


namespace NUMINAMATH_CALUDE_books_about_sports_l2202_220237

theorem books_about_sports (total_books school_books : ℕ) : 
  total_books = 58 → school_books = 19 → total_books - school_books = 39 := by
  sorry

end NUMINAMATH_CALUDE_books_about_sports_l2202_220237


namespace NUMINAMATH_CALUDE_weight_difference_l2202_220277

theorem weight_difference (n : ℕ) (joe_weight : ℝ) (initial_avg : ℝ) (new_avg : ℝ) :
  joe_weight = 42 →
  initial_avg = 30 →
  new_avg = 31 →
  (n * initial_avg + joe_weight) / (n + 1) = new_avg →
  let total_weight := n * initial_avg + joe_weight
  let remaining_students := n - 1
  ∃ (x : ℝ), (total_weight - 2 * x) / remaining_students = initial_avg →
  |x - joe_weight| = 6 :=
by sorry

end NUMINAMATH_CALUDE_weight_difference_l2202_220277


namespace NUMINAMATH_CALUDE_meaningful_expression_l2202_220236

theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 5)) / x) ↔ (x ≥ -5 ∧ x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l2202_220236


namespace NUMINAMATH_CALUDE_apples_taken_per_basket_l2202_220250

theorem apples_taken_per_basket (initial_apples : ℕ) (num_baskets : ℕ) (apples_per_basket : ℕ) :
  initial_apples = 64 →
  num_baskets = 4 →
  apples_per_basket = 13 →
  ∃ (taken_per_basket : ℕ),
    taken_per_basket * num_baskets = initial_apples - (apples_per_basket * num_baskets) ∧
    taken_per_basket = 3 :=
by sorry

end NUMINAMATH_CALUDE_apples_taken_per_basket_l2202_220250


namespace NUMINAMATH_CALUDE_side_median_ratio_not_unique_l2202_220285

/-- A triangle in a 2D plane --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The ratio of a side to its corresponding median in a triangle --/
def sideMedianRatio (t : Triangle) : ℝ := 
  sorry

/-- Predicate to check if two triangles have the same shape (are similar) --/
def hasSameShape (t1 t2 : Triangle) : Prop := 
  sorry

/-- Theorem stating that the ratio of a side to its corresponding median 
    does not uniquely determine a triangle's shape --/
theorem side_median_ratio_not_unique : 
  ∃ t1 t2 : Triangle, 
    sideMedianRatio t1 = sideMedianRatio t2 ∧ 
    ¬(hasSameShape t1 t2) := by
  sorry

end NUMINAMATH_CALUDE_side_median_ratio_not_unique_l2202_220285


namespace NUMINAMATH_CALUDE_ellipse_area_lower_bound_l2202_220296

/-- Given a right-angled triangle with area t, where the endpoints of its hypotenuse
    lie at the foci of an ellipse and the third vertex lies on the ellipse,
    the area of the ellipse is at least √2πt. -/
theorem ellipse_area_lower_bound (t : ℝ) (a b c : ℝ) (h1 : 0 < t) (h2 : 0 < b) (h3 : b < a)
    (h4 : a^2 = b^2 + c^2) (h5 : t = b^2) : π * a * b ≥ Real.sqrt 2 * π * t :=
by sorry

end NUMINAMATH_CALUDE_ellipse_area_lower_bound_l2202_220296


namespace NUMINAMATH_CALUDE_circumscribed_isosceles_trapezoid_radius_l2202_220255

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedIsoscelesTrapezoid where
  /-- The angle at the base of the trapezoid -/
  baseAngle : ℝ
  /-- The length of the midline of the trapezoid -/
  midline : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ

/-- The theorem stating the relationship between the trapezoid's properties and the inscribed circle's radius -/
theorem circumscribed_isosceles_trapezoid_radius 
  (t : CircumscribedIsoscelesTrapezoid) 
  (h1 : t.baseAngle = 30 * π / 180)  -- 30 degrees in radians
  (h2 : t.midline = 10) : 
  t.radius = 2.5 := by
  sorry


end NUMINAMATH_CALUDE_circumscribed_isosceles_trapezoid_radius_l2202_220255


namespace NUMINAMATH_CALUDE_clock_divisibility_impossible_l2202_220224

theorem clock_divisibility_impossible (a b : ℕ) : 
  0 < a → a ≤ 12 → b < 60 → 
  ¬ (∃ k : ℕ, (120 * a + 2 * b) = k * (100 * a + b)) := by
  sorry

end NUMINAMATH_CALUDE_clock_divisibility_impossible_l2202_220224


namespace NUMINAMATH_CALUDE_smallest_rectangle_area_is_768_l2202_220232

/-- The side length of each square in centimeters -/
def square_side : ℝ := 8

/-- The number of squares in the height of the L-shape -/
def height_squares : ℕ := 3

/-- The number of squares in the width of the L-shape -/
def width_squares : ℕ := 4

/-- The height of the L-shape in centimeters -/
def l_shape_height : ℝ := square_side * height_squares

/-- The width of the L-shape in centimeters -/
def l_shape_width : ℝ := square_side * width_squares

/-- The smallest possible area of a rectangle that can completely contain the L-shape -/
def smallest_rectangle_area : ℝ := l_shape_height * l_shape_width

theorem smallest_rectangle_area_is_768 : smallest_rectangle_area = 768 := by
  sorry

end NUMINAMATH_CALUDE_smallest_rectangle_area_is_768_l2202_220232


namespace NUMINAMATH_CALUDE_green_face_prob_half_l2202_220283

/-- A cube with colored faces -/
structure ColoredCube where
  total_faces : ℕ
  green_faces : ℕ
  purple_faces : ℕ

/-- The probability of rolling a green face on a colored cube -/
def green_face_probability (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

/-- Theorem: The probability of rolling a green face on a cube with 3 green faces and 3 purple faces is 1/2 -/
theorem green_face_prob_half :
  let cube : ColoredCube := { total_faces := 6, green_faces := 3, purple_faces := 3 }
  green_face_probability cube = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_green_face_prob_half_l2202_220283


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2202_220291

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) =
  12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2202_220291


namespace NUMINAMATH_CALUDE_total_jog_time_two_weeks_l2202_220227

/-- The number of hours jogged daily -/
def daily_jog_hours : ℝ := 1.5

/-- The number of days in two weeks -/
def days_in_two_weeks : ℕ := 14

/-- Theorem: Total jogging time in two weeks -/
theorem total_jog_time_two_weeks : 
  daily_jog_hours * (days_in_two_weeks : ℝ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_jog_time_two_weeks_l2202_220227


namespace NUMINAMATH_CALUDE_fifteen_divides_Q_largest_divisor_fifteen_largest_divisor_l2202_220228

/-- The product of four consecutive positive odd integers -/
def Q (n : ℕ) : ℕ := (2*n - 3) * (2*n - 1) * (2*n + 1) * (2*n + 3)

/-- 15 divides Q for all n -/
theorem fifteen_divides_Q (n : ℕ) : 15 ∣ Q n :=
sorry

/-- For any integer k > 15, there exists an n such that k does not divide Q n -/
theorem largest_divisor (k : ℕ) (h : k > 15) : ∃ n : ℕ, ¬(k ∣ Q n) :=
sorry

/-- 15 is the largest integer that divides Q for all n -/
theorem fifteen_largest_divisor : ∀ k : ℕ, (∀ n : ℕ, k ∣ Q n) → k ≤ 15 :=
sorry

end NUMINAMATH_CALUDE_fifteen_divides_Q_largest_divisor_fifteen_largest_divisor_l2202_220228


namespace NUMINAMATH_CALUDE_line_plane_relationship_l2202_220219

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perp_line_line : Line → Line → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the contained relation (line contained in a plane)
variable (contained_line_plane : Line → Plane → Prop)

-- Theorem statement
theorem line_plane_relationship 
  (m n : Line) (α : Plane) 
  (h1 : perp_line_plane m α) 
  (h2 : perp_line_line m n) : 
  parallel_line_plane n α ∨ contained_line_plane n α :=
sorry

end NUMINAMATH_CALUDE_line_plane_relationship_l2202_220219


namespace NUMINAMATH_CALUDE_tan_sum_problem_l2202_220241

theorem tan_sum_problem (α β : Real) 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := by sorry

end NUMINAMATH_CALUDE_tan_sum_problem_l2202_220241


namespace NUMINAMATH_CALUDE_min_value_cos_squared_minus_sin_squared_l2202_220214

theorem min_value_cos_squared_minus_sin_squared :
  ∃ (m : ℝ), (∀ x, m ≤ (Real.cos (x/2))^2 - (Real.sin (x/2))^2) ∧ 
  (∃ x₀, m = (Real.cos (x₀/2))^2 - (Real.sin (x₀/2))^2) ∧
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cos_squared_minus_sin_squared_l2202_220214


namespace NUMINAMATH_CALUDE_steel_copper_weight_difference_l2202_220225

/-- Represents the weight of a metal bar in kilograms. -/
structure MetalBar where
  weight : ℝ

/-- The container with metal bars. -/
structure Container where
  steel : MetalBar
  tin : MetalBar
  copper : MetalBar
  count : ℕ
  totalWeight : ℝ

/-- Theorem stating the weight difference between steel and copper bars. -/
theorem steel_copper_weight_difference (c : Container) : 
  c.steel.weight - c.copper.weight = 20 :=
  by
  have h1 : c.steel.weight = 2 * c.tin.weight := sorry
  have h2 : c.copper.weight = 90 := sorry
  have h3 : c.count = 20 := sorry
  have h4 : c.totalWeight = 5100 := sorry
  have h5 : c.count * (c.steel.weight + c.tin.weight + c.copper.weight) = c.totalWeight := sorry
  sorry

#check steel_copper_weight_difference

end NUMINAMATH_CALUDE_steel_copper_weight_difference_l2202_220225


namespace NUMINAMATH_CALUDE_ab_squared_commutes_l2202_220216

theorem ab_squared_commutes (a b : ℝ) : a * b^2 - b^2 * a = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_squared_commutes_l2202_220216


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2202_220290

theorem sum_of_fractions_equals_seven : 
  let S := 1 / (4 - Real.sqrt 15) - 1 / (Real.sqrt 15 - Real.sqrt 14) + 
           1 / (Real.sqrt 14 - Real.sqrt 13) - 1 / (Real.sqrt 13 - Real.sqrt 12) + 
           1 / (Real.sqrt 12 - 3)
  S = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_seven_l2202_220290


namespace NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2202_220212

theorem greatest_integer_satisfying_inequality :
  ∀ y : ℤ, (3 * |y| + 6 < 24) → y ≤ 5 ∧ ∃ (z : ℤ), z > 5 ∧ ¬(3 * |z| + 6 < 24) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_satisfying_inequality_l2202_220212


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l2202_220278

/-- A quadratic function of the form y = x^2 + (1-m)x + 1 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + (1-m)*x + 1

/-- The derivative of the quadratic function -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + (1-m)

theorem quadratic_increasing_condition (m : ℝ) :
  (∀ x > 1, quadratic_derivative m x > 0) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l2202_220278


namespace NUMINAMATH_CALUDE_calvin_collection_total_l2202_220253

def insect_collection (roaches scorpions : ℕ) : ℕ :=
  let crickets := roaches / 2
  let caterpillars := 2 * scorpions
  roaches + scorpions + crickets + caterpillars

theorem calvin_collection_total :
  insect_collection 12 3 = 27 :=
by sorry

end NUMINAMATH_CALUDE_calvin_collection_total_l2202_220253


namespace NUMINAMATH_CALUDE_min_value_fraction_l2202_220265

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 2 → 1/a + 2/b ≤ 1/x + 2/y) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 2 ∧ 1/x + 2/y = (3 + 2 * Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2202_220265


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2202_220204

theorem cubic_roots_sum (k m : ℝ) : 
  (∃ a b c : ℕ+, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 - 8*x^2 + k*x - m = 0 ↔ (x = a ∨ x = b ∨ x = c))) →
  (k + m = 27 ∨ k + m = 31) :=
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2202_220204


namespace NUMINAMATH_CALUDE_paint_needed_for_smaller_statues_l2202_220248

/-- The height of the original statue in feet -/
def original_height : ℝ := 12

/-- The height of each smaller statue in feet -/
def smaller_height : ℝ := 2

/-- The number of smaller statues -/
def num_statues : ℕ := 720

/-- The amount of paint in pints needed for the original statue -/
def paint_for_original : ℝ := 1

/-- The amount of paint needed for all smaller statues -/
def paint_for_all_statues : ℝ := 20

theorem paint_needed_for_smaller_statues :
  (num_statues : ℝ) * paint_for_original * (smaller_height / original_height) ^ 2 = paint_for_all_statues :=
sorry

end NUMINAMATH_CALUDE_paint_needed_for_smaller_statues_l2202_220248


namespace NUMINAMATH_CALUDE_arc_length_sector_l2202_220297

/-- The arc length of a sector with radius π cm and central angle 2π/3 radians is 2π²/3 cm. -/
theorem arc_length_sector (r : Real) (θ : Real) (l : Real) :
  r = π → θ = 2 * π / 3 → l = θ * r → l = 2 * π^2 / 3 := by
  sorry

#check arc_length_sector

end NUMINAMATH_CALUDE_arc_length_sector_l2202_220297


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l2202_220249

theorem sum_of_first_n_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l2202_220249


namespace NUMINAMATH_CALUDE_abs_neg_five_halves_l2202_220211

theorem abs_neg_five_halves : |(-5 : ℚ) / 2| = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_halves_l2202_220211


namespace NUMINAMATH_CALUDE_circle_center_problem_circle_center_l2202_220220

/-- The equation of a circle in the form x² + y² + 2ax + 2by + c = 0 
    has center (-a, -b) -/
theorem circle_center (a b c : ℝ) :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 + 2*a*x + 2*b*y + c = 0
  let center := (-a, -b)
  ∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = a^2 + b^2 - c :=
by sorry

/-- The center of the circle x² + y² + 2x + 4y - 3 = 0 is (-1, -2) -/
theorem problem_circle_center :
  let circle_eq := fun (x y : ℝ) => x^2 + y^2 + 2*x + 4*y - 3 = 0
  let center := (-1, -2)
  ∀ x y, circle_eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = 8 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_problem_circle_center_l2202_220220


namespace NUMINAMATH_CALUDE_sphere_center_sum_l2202_220271

theorem sphere_center_sum (x y z : ℝ) :
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sphere_center_sum_l2202_220271


namespace NUMINAMATH_CALUDE_students_not_in_biology_l2202_220266

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) : 
  total_students = 880 → 
  biology_percentage = 30 / 100 →
  (total_students : ℚ) * (1 - biology_percentage) = 616 :=
by sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l2202_220266


namespace NUMINAMATH_CALUDE_least_common_meeting_time_l2202_220292

def prime_lap_times : List Nat := [2, 3, 5, 7, 11, 13, 17]

def is_divisible_by_at_least_four (n : Nat) : Bool :=
  (prime_lap_times.filter (fun p => n % p = 0)).length ≥ 4

theorem least_common_meeting_time :
  ∃ T : Nat, T > 0 ∧ is_divisible_by_at_least_four T ∧
  ∀ t : Nat, 0 < t ∧ t < T → ¬is_divisible_by_at_least_four t :=
by sorry

end NUMINAMATH_CALUDE_least_common_meeting_time_l2202_220292


namespace NUMINAMATH_CALUDE_max_value_theorem_l2202_220289

-- Define the function f
def f (a x : ℝ) : ℝ := a * x^2 - |x - a|

-- State the theorem
theorem max_value_theorem (a b : ℝ) :
  (-1 ≤ a) →
  (a ≤ 1) →
  (∀ x ∈ Set.Icc 1 3, f a x + b * x ≤ 0) →
  (a^2 + 3 * b ≤ 10) ∧ 
  (∃ a₀ b₀, (-1 ≤ a₀) ∧ (a₀ ≤ 1) ∧ 
   (∀ x ∈ Set.Icc 1 3, f a₀ x + b₀ * x ≤ 0) ∧ 
   (a₀^2 + 3 * b₀ = 10)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2202_220289


namespace NUMINAMATH_CALUDE_prob_greater_than_three_is_half_l2202_220213

/-- The probability of rolling a number greater than 3 on a standard six-sided die is 1/2. -/
theorem prob_greater_than_three_is_half : 
  let outcomes := Finset.range 6
  let favorable := {4, 5, 6}
  Finset.card favorable / Finset.card outcomes = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_greater_than_three_is_half_l2202_220213


namespace NUMINAMATH_CALUDE_sugar_mixture_theorem_l2202_220201

/-- Given a bowl of sugar with the following properties:
  * Initially contains 320 grams of pure white sugar
  * Mixture Y is formed by removing x grams of white sugar and adding x grams of brown sugar
  * In Mixture Y, the ratio of white sugar to brown sugar is w:b in lowest terms
  * Mixture Z is formed by removing x grams of Mixture Y and adding x grams of brown sugar
  * In Mixture Z, the ratio of white sugar to brown sugar is 49:15
  Prove that x + w + b = 48 -/
theorem sugar_mixture_theorem (x w b : ℕ) : 
  x > 0 ∧ x < 320 ∧ 
  (320 - x : ℚ) / x = w / b ∧ 
  (320 - x) * (320 - x) / (320 : ℚ) / ((2 * x - x^2 / 320 : ℚ)) = 49 / 15 →
  x + w + b = 48 :=
by sorry

end NUMINAMATH_CALUDE_sugar_mixture_theorem_l2202_220201


namespace NUMINAMATH_CALUDE_empty_cell_exists_l2202_220222

/-- Represents a 5x5 grid --/
def Grid := Fin 5 → Fin 5 → Bool

/-- A function that checks if two cells are adjacent --/
def adjacent (a b : Fin 5 × Fin 5) : Prop :=
  (a.1 = b.1 ∧ (a.2.val + 1 = b.2.val ∨ a.2.val = b.2.val + 1)) ∨
  (a.2 = b.2 ∧ (a.1.val + 1 = b.1.val ∨ a.1.val = b.1.val + 1))

/-- Represents the movement of bugs --/
def moves (before after : Grid) : Prop :=
  ∀ (i j : Fin 5), 
    before i j → ∃ (i' j' : Fin 5), adjacent (i, j) (i', j') ∧ after i' j'

/-- The main theorem --/
theorem empty_cell_exists (before after : Grid) 
  (h1 : ∀ (i j : Fin 5), before i j)
  (h2 : moves before after) : 
  ∃ (i j : Fin 5), ¬after i j :=
sorry

end NUMINAMATH_CALUDE_empty_cell_exists_l2202_220222


namespace NUMINAMATH_CALUDE_scientific_notation_15510000_l2202_220233

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h_coeff_range : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_15510000 :
  toScientificNotation 15510000 = ScientificNotation.mk 1.551 7 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_15510000_l2202_220233


namespace NUMINAMATH_CALUDE_sqrt_360_simplification_l2202_220235

theorem sqrt_360_simplification : Real.sqrt 360 = 6 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_360_simplification_l2202_220235


namespace NUMINAMATH_CALUDE_max_divisor_with_remainder_l2202_220260

theorem max_divisor_with_remainder (A B : ℕ) : 
  (24 = A * B + 4) → A ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_max_divisor_with_remainder_l2202_220260


namespace NUMINAMATH_CALUDE_sqrt_65_greater_than_8_l2202_220270

theorem sqrt_65_greater_than_8 : Real.sqrt 65 > 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_65_greater_than_8_l2202_220270


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2202_220206

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := fun x : ℝ => a^(x + 1) - 1
  f (-1) = 0 := by
sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l2202_220206


namespace NUMINAMATH_CALUDE_real_roots_condition_specific_roots_condition_l2202_220295

variable (m : ℝ)
variable (x₁ x₂ : ℝ)

-- Define the quadratic equation
def quadratic (x : ℝ) := x^2 - 6*x + (4*m + 1)

-- Theorem 1: For real roots, m ≤ 2
theorem real_roots_condition : (∃ x : ℝ, quadratic m x = 0) → m ≤ 2 := by sorry

-- Theorem 2: If x₁ and x₂ are roots and x₁² + x₂² = 26, then m = 1
theorem specific_roots_condition : 
  quadratic m x₁ = 0 → quadratic m x₂ = 0 → x₁^2 + x₂^2 = 26 → m = 1 := by sorry

end NUMINAMATH_CALUDE_real_roots_condition_specific_roots_condition_l2202_220295


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_3_and_5_l2202_220254

theorem greatest_two_digit_multiple_of_3_and_5 : 
  ∀ n : ℕ, n ≤ 99 → n ≥ 10 → n % 3 = 0 → n % 5 = 0 → n ≤ 90 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_3_and_5_l2202_220254


namespace NUMINAMATH_CALUDE_bouquets_to_buy_is_correct_l2202_220239

/-- Represents the number of roses in a bouquet Bill buys -/
def roses_per_bought_bouquet : ℕ := 7

/-- Represents the number of roses in a bouquet Bill sells -/
def roses_per_sold_bouquet : ℕ := 5

/-- Represents the price of a bouquet (both buying and selling) -/
def price_per_bouquet : ℕ := 20

/-- Represents the target profit -/
def target_profit : ℕ := 1000

/-- Calculates the number of bouquets Bill needs to buy to earn the target profit -/
def bouquets_to_buy : ℕ :=
  let bought_bouquets_per_operation := roses_per_sold_bouquet
  let sold_bouquets_per_operation := roses_per_bought_bouquet
  let profit_per_operation := sold_bouquets_per_operation * price_per_bouquet - bought_bouquets_per_operation * price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * bought_bouquets_per_operation

theorem bouquets_to_buy_is_correct :
  bouquets_to_buy = 125 := by sorry

end NUMINAMATH_CALUDE_bouquets_to_buy_is_correct_l2202_220239


namespace NUMINAMATH_CALUDE_ball_distribution_l2202_220240

/-- Represents the number of ways to distribute balls into boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to place 7 distinguishable balls into 3 boxes,
    where one box is red and the other two are indistinguishable -/
theorem ball_distribution : distribute_balls 7 3 = 64 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_l2202_220240


namespace NUMINAMATH_CALUDE_stating_solve_age_problem_l2202_220269

/-- Represents the age-related problem described in the question. -/
def AgeProblem (current_age : ℕ) (years_ago : ℕ) : Prop :=
  3 * (current_age + 3) - 3 * (current_age - years_ago) = current_age

/-- 
Theorem stating that given the person's current age of 18, 
the number of years ago referred to in their statement is 3.
-/
theorem solve_age_problem : 
  ∃ (years_ago : ℕ), AgeProblem 18 years_ago ∧ years_ago = 3 :=
sorry

end NUMINAMATH_CALUDE_stating_solve_age_problem_l2202_220269


namespace NUMINAMATH_CALUDE_carries_payment_l2202_220208

/-- Calculate Carrie's payment for clothes shopping --/
theorem carries_payment (shirt_quantity : ℕ) (pants_quantity : ℕ) (jacket_quantity : ℕ) 
  (skirt_quantity : ℕ) (shoes_quantity : ℕ) (shirt_price : ℚ) (pants_price : ℚ) 
  (jacket_price : ℚ) (skirt_price : ℚ) (shoes_price : ℚ) (shirt_discount : ℚ) 
  (jacket_discount : ℚ) (skirt_discount : ℚ) (mom_payment_ratio : ℚ) :
  shirt_quantity = 8 →
  pants_quantity = 4 →
  jacket_quantity = 4 →
  skirt_quantity = 3 →
  shoes_quantity = 2 →
  shirt_price = 12 →
  pants_price = 25 →
  jacket_price = 75 →
  skirt_price = 30 →
  shoes_price = 50 →
  shirt_discount = 0.2 →
  jacket_discount = 0.2 →
  skirt_discount = 0.1 →
  mom_payment_ratio = 2/3 →
  let total_cost := 
    (shirt_quantity : ℚ) * shirt_price * (1 - shirt_discount) +
    (pants_quantity : ℚ) * pants_price +
    (jacket_quantity : ℚ) * jacket_price * (1 - jacket_discount) +
    (skirt_quantity : ℚ) * skirt_price * (1 - skirt_discount) +
    (shoes_quantity : ℚ) * shoes_price
  (1 - mom_payment_ratio) * total_cost = 199.27 := by
  sorry

end NUMINAMATH_CALUDE_carries_payment_l2202_220208


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problems_l2202_220200

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_problems
  (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0) (h_arith : arithmetic_sequence a d)
  (h_sum : a 3 + a 6 + a 10 + a 13 = 32)
  (h_m : ∃ m : ℕ, a m = 8)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = (n : ℝ) * (a 1 + a n) / 2)
  (h_S3 : S 3 = 9)
  (h_S6 : S 6 = 36) :
  (∃ m : ℕ, a m = 8 ∧ m = 8) ∧
  (a 7 + a 8 + a 9 = 45) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problems_l2202_220200


namespace NUMINAMATH_CALUDE_P_in_M_l2202_220276

def P : Set Nat := {0, 1}

def M : Set (Set Nat) := {x | x ⊆ P}

theorem P_in_M : P ∈ M := by sorry

end NUMINAMATH_CALUDE_P_in_M_l2202_220276


namespace NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_y_l2202_220210

/-- The cost in cents for Plan X given y minutes of usage -/
def costX (y : ℕ) : ℚ := 15 * y

/-- The cost in cents for Plan Y given y minutes of usage -/
def costY (y : ℕ) : ℚ := 2500 + 8 * y

/-- Theorem stating that 358 is the minimum whole number of minutes for Plan Y to be cheaper -/
theorem min_minutes_for_cheaper_plan_y : 
  (∀ y : ℕ, y < 358 → costY y ≥ costX y) ∧ 
  costY 358 < costX 358 := by
  sorry

end NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_y_l2202_220210


namespace NUMINAMATH_CALUDE_only_setA_is_pythagorean_triple_l2202_220287

/-- Checks if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- The sets of numbers to check -/
def setA : List ℕ := [6, 8, 10]
def setB : List ℚ := [3/10, 4/10, 5/10]
def setC : List ℚ := [3/2, 4/2, 5/2]
def setD : List ℕ := [5, 11, 12]

theorem only_setA_is_pythagorean_triple :
  (∃ (a b c : ℕ), a ∈ setA ∧ b ∈ setA ∧ c ∈ setA ∧ isPythagoreanTriple a b c) ∧
  (¬∃ (a b c : ℚ), a ∈ setB ∧ b ∈ setB ∧ c ∈ setB ∧ a.num * a.num + b.num * b.num = c.num * c.num) ∧
  (¬∃ (a b c : ℚ), a ∈ setC ∧ b ∈ setC ∧ c ∈ setC ∧ a.num * a.num + b.num * b.num = c.num * c.num) ∧
  (¬∃ (a b c : ℕ), a ∈ setD ∧ b ∈ setD ∧ c ∈ setD ∧ isPythagoreanTriple a b c) :=
by sorry


end NUMINAMATH_CALUDE_only_setA_is_pythagorean_triple_l2202_220287


namespace NUMINAMATH_CALUDE_age_ratio_problem_l2202_220238

theorem age_ratio_problem (a b : ℕ) (h1 : 5 * b = 3 * a) (h2 : a - 4 = b + 4) :
  3 * (b - 4) = a + 4 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_problem_l2202_220238


namespace NUMINAMATH_CALUDE_small_box_tape_length_l2202_220264

theorem small_box_tape_length (large_seal : ℕ) (medium_seal : ℕ) (label_tape : ℕ)
  (large_count : ℕ) (medium_count : ℕ) (small_count : ℕ) (total_tape : ℕ)
  (h1 : large_seal = 4)
  (h2 : medium_seal = 2)
  (h3 : label_tape = 1)
  (h4 : large_count = 2)
  (h5 : medium_count = 8)
  (h6 : small_count = 5)
  (h7 : total_tape = 44)
  (h8 : total_tape = large_count * large_seal + medium_count * medium_seal + 
        small_count * label_tape + large_count * label_tape + 
        medium_count * label_tape + small_count * label_tape + 
        small_count * small_seal) :
  small_seal = 1 :=
by sorry

end NUMINAMATH_CALUDE_small_box_tape_length_l2202_220264


namespace NUMINAMATH_CALUDE_prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1_l2202_220262

theorem prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1 
  (p : ℕ) (n : ℕ) (h_prime : Nat.Prime p) (h_n_ge_2 : n ≥ 2) 
  (h_div : p ∣ (n^6 - 1)) : n > Real.sqrt p - 1 :=
sorry

end NUMINAMATH_CALUDE_prime_divides_n6_minus_1_implies_n_greater_than_sqrt_p_minus_1_l2202_220262


namespace NUMINAMATH_CALUDE_valid_assignments_l2202_220223

-- Define a type for statements
inductive Statement
| Assign1 : Statement  -- x←1, y←2, z←3
| Assign2 : Statement  -- S^2←4
| Assign3 : Statement  -- i←i+2
| Assign4 : Statement  -- x+1←x

-- Define a predicate for valid assignment statements
def is_valid_assignment (s : Statement) : Prop :=
  match s with
  | Statement.Assign1 => True
  | Statement.Assign2 => False
  | Statement.Assign3 => True
  | Statement.Assign4 => False

-- Theorem stating which statements are valid assignments
theorem valid_assignments :
  (is_valid_assignment Statement.Assign1) ∧
  (¬is_valid_assignment Statement.Assign2) ∧
  (is_valid_assignment Statement.Assign3) ∧
  (¬is_valid_assignment Statement.Assign4) := by
  sorry

end NUMINAMATH_CALUDE_valid_assignments_l2202_220223


namespace NUMINAMATH_CALUDE_average_salary_l2202_220226

/-- The average salary of 5 people with given salaries is 9000 --/
theorem average_salary (a b c d e : ℕ) 
  (ha : a = 8000) (hb : b = 5000) (hc : c = 16000) (hd : d = 7000) (he : e = 9000) : 
  (a + b + c + d + e) / 5 = 9000 := by
  sorry

end NUMINAMATH_CALUDE_average_salary_l2202_220226


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l2202_220288

theorem five_digit_multiple_of_nine :
  ∃ (n : ℕ), n = 56781 ∧ n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l2202_220288


namespace NUMINAMATH_CALUDE_min_total_cost_l2202_220229

/-- Represents the transportation problem with two warehouses and two construction sites -/
structure TransportationProblem where
  warehouseA_capacity : ℝ
  warehouseB_capacity : ℝ
  siteA_demand : ℝ
  siteB_demand : ℝ
  costA_to_A : ℝ
  costA_to_B : ℝ
  costB_to_A : ℝ
  costB_to_B : ℝ

/-- The specific transportation problem instance -/
def problem : TransportationProblem :=
  { warehouseA_capacity := 800
  , warehouseB_capacity := 1200
  , siteA_demand := 1300
  , siteB_demand := 700
  , costA_to_A := 12
  , costA_to_B := 15
  , costB_to_A := 10
  , costB_to_B := 18
  }

/-- The cost reduction from Warehouse A to Site A -/
def cost_reduction (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 6

/-- The amount transported from Warehouse A to Site A -/
def transport_amount (x : ℝ) : Prop := 100 ≤ x ∧ x ≤ 800

/-- The theorem stating the minimum total transportation cost after cost reduction -/
theorem min_total_cost (p : TransportationProblem) (a : ℝ) (x : ℝ) 
  (h1 : p = problem) (h2 : cost_reduction a) (h3 : transport_amount x) : 
  ∃ y : ℝ, y = 22400 ∧ ∀ z : ℝ, z ≥ y := by
  sorry

end NUMINAMATH_CALUDE_min_total_cost_l2202_220229


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2202_220259

/-- Represents a tile arrangement -/
structure TileArrangement where
  num_tiles : ℕ
  perimeter : ℕ

/-- Represents the process of adding tiles to an arrangement -/
def add_tiles (initial : TileArrangement) (added_tiles : ℕ) : TileArrangement :=
  { num_tiles := initial.num_tiles + added_tiles,
    perimeter := initial.perimeter }  -- Placeholder, actual calculation depends on arrangement

/-- The theorem to be proved -/
theorem perimeter_after_adding_tiles 
  (initial : TileArrangement) 
  (h1 : initial.num_tiles = 10) 
  (h2 : initial.perimeter = 20) :
  ∃ (final : TileArrangement), 
    final = add_tiles initial 2 ∧ 
    final.perimeter = 19 :=
sorry

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l2202_220259


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l2202_220258

/-- The common ratio of the geometric series 4/5 - 5/12 + 25/72 - ... is -25/48 -/
theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 4/5
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 25/72
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 1 → a₂ = r * a₁ ∧ a₃ = r * a₂) →
  r = -25/48 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l2202_220258


namespace NUMINAMATH_CALUDE_basketball_court_perimeter_l2202_220247

/-- The perimeter of a rectangular basketball court is 96 meters -/
theorem basketball_court_perimeter :
  ∀ (length width : ℝ),
  length = width + 14 →
  (length = 31 ∧ width = 17) →
  2 * (length + width) = 96 := by
  sorry

end NUMINAMATH_CALUDE_basketball_court_perimeter_l2202_220247


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2202_220279

def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

theorem perpendicular_vectors (m : ℝ) : 
  (a.1 * (a.1 + 2 * (b m).1) + a.2 * (a.2 + 2 * (b m).2) = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2202_220279


namespace NUMINAMATH_CALUDE_ratio_to_two_l2202_220267

theorem ratio_to_two (x : ℝ) : (x / 2 = 150 / 1) → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_ratio_to_two_l2202_220267
