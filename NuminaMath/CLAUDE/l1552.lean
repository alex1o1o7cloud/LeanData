import Mathlib

namespace NUMINAMATH_CALUDE_circle_point_range_l1552_155209

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1

-- Define points A and B
def point_A (m : ℝ) : ℝ × ℝ := (0, m)
def point_B (m : ℝ) : ℝ × ℝ := (0, -m)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C P.1 P.2 ∧ 
  ∃ (A B : ℝ × ℝ), A = point_A m ∧ B = point_B m ∧ 
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 → (∃ P : ℝ × ℝ, point_P_condition P m) → 1 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_circle_point_range_l1552_155209


namespace NUMINAMATH_CALUDE_sqrt_identity_l1552_155273

theorem sqrt_identity (t : ℝ) : 
  Real.sqrt (t^6 + t^4 + t^2) = |t| * Real.sqrt (t^4 + t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l1552_155273


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1552_155239

theorem point_not_in_second_quadrant (n : ℝ) : ¬(n + 1 < 0 ∧ 2*n - 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1552_155239


namespace NUMINAMATH_CALUDE_vertical_asymptote_at_neg_two_l1552_155256

/-- The function f(x) = (x^2 + 6x + 9) / (x + 2) has a vertical asymptote at x = -2 -/
theorem vertical_asymptote_at_neg_two :
  ∃ (f : ℝ → ℝ), 
    (∀ x ≠ -2, f x = (x^2 + 6*x + 9) / (x + 2)) ∧
    (∃ (L : ℝ → ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x + 2| ∧ |x + 2| < δ → |f x| > L ε)) :=
by sorry

end NUMINAMATH_CALUDE_vertical_asymptote_at_neg_two_l1552_155256


namespace NUMINAMATH_CALUDE_equation_solution_l1552_155263

theorem equation_solution :
  ∃ x : ℚ, x - 2 ≠ 0 ∧ (2 / (x - 2) = (1 + x) / (x - 2) + 1) ∧ x = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1552_155263


namespace NUMINAMATH_CALUDE_estimate_larger_than_actual_l1552_155250

theorem estimate_larger_than_actual (x y z : ℝ) 
  (h1 : x > y) (h2 : y > 0) (h3 : z > 0) : 
  (x + 2*z) - (y - 2*z) > x - y := by
  sorry

end NUMINAMATH_CALUDE_estimate_larger_than_actual_l1552_155250


namespace NUMINAMATH_CALUDE_indeterminate_product_l1552_155240

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the continuity of f on [-2, 2]
variable (hcont : ContinuousOn f (Set.Icc (-2) 2))

-- Define that f has at least one root in (-2, 2)
variable (hroot : ∃ x ∈ Set.Ioo (-2) 2, f x = 0)

-- Theorem statement
theorem indeterminate_product :
  ¬ (∀ (f : ℝ → ℝ) (hcont : ContinuousOn f (Set.Icc (-2) 2)) 
    (hroot : ∃ x ∈ Set.Ioo (-2) 2, f x = 0),
    (f (-2) * f 2 > 0) ∨ (f (-2) * f 2 < 0) ∨ (f (-2) * f 2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_indeterminate_product_l1552_155240


namespace NUMINAMATH_CALUDE_sum_two_longest_altitudes_l1552_155287

/-- A right triangle with sides 6, 8, and 10 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 6
  side_b : b = 8
  side_c : c = 10

/-- The sum of the lengths of the two longest altitudes in the given right triangle is 14 -/
theorem sum_two_longest_altitudes (t : RightTriangle) : 
  max t.a t.b + min t.a t.b = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_two_longest_altitudes_l1552_155287


namespace NUMINAMATH_CALUDE_intersecting_circles_sum_l1552_155264

/-- Given two intersecting circles with centers on the line x - y + c = 0 and
    intersection points A(1, 3) and B(m, -1), prove that m + c = -1 -/
theorem intersecting_circles_sum (m c : ℝ) : 
  (∃ (circle1 circle2 : Set (ℝ × ℝ)),
    (∃ (center1 center2 : ℝ × ℝ),
      center1 ∈ circle1 ∧ center2 ∈ circle2 ∧
      center1.1 - center1.2 + c = 0 ∧ center2.1 - center2.2 + c = 0) ∧
    (1, 3) ∈ circle1 ∩ circle2 ∧ (m, -1) ∈ circle1 ∩ circle2) →
  m + c = -1 := by
sorry

end NUMINAMATH_CALUDE_intersecting_circles_sum_l1552_155264


namespace NUMINAMATH_CALUDE_root_in_interval_l1552_155261

noncomputable def f (x : ℝ) := Real.log x + x - 2

theorem root_in_interval : ∃ (k : ℤ), ∃ (x₀ : ℝ),
  x₀ > 0 ∧ 
  f x₀ = 0 ∧
  x₀ > k ∧ 
  x₀ < k + 1 ∧
  k = 1 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1552_155261


namespace NUMINAMATH_CALUDE_a2_value_l1552_155280

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 2

-- Define the geometric sequence property for a_1, a_2, and a_5
def geometric_property (a : ℕ → ℝ) : Prop :=
  (a 2 / a 1) = (a 5 / a 2)

-- Theorem statement
theorem a2_value (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_geom : geometric_property a) : 
  a 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_a2_value_l1552_155280


namespace NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1552_155215

theorem rectangle_area_diagonal_relation :
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  length / width = 5 / 2 →
  2 * (length + width) = 56 →
  ∃ (d : ℝ),
  d^2 = length^2 + width^2 ∧
  length * width = (10/29) * d^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_diagonal_relation_l1552_155215


namespace NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l1552_155247

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 48) (h2 : b * 8 = a * 9) : 
  Nat.lcm a b = 432 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_ratio_numbers_l1552_155247


namespace NUMINAMATH_CALUDE_probability_second_new_given_first_new_l1552_155208

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := 10

/-- Represents the number of new balls initially in the box -/
def new_balls : ℕ := 6

/-- Represents the number of old balls initially in the box -/
def old_balls : ℕ := 4

/-- Theorem stating the probability of drawing a new ball on the second draw,
    given that the first ball drawn was new -/
theorem probability_second_new_given_first_new :
  (new_balls - 1) / (total_balls - 1) = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_probability_second_new_given_first_new_l1552_155208


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1552_155217

/-- Given a quadratic equation (m-2)x^2 + 3x + m^2 - 4 = 0 where x = 0 is a solution, prove that m = -2 -/
theorem quadratic_equation_solution (m : ℝ) : 
  ((m - 2) * 0^2 + 3 * 0 + m^2 - 4 = 0) → m = -2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1552_155217


namespace NUMINAMATH_CALUDE_complex_point_in_second_quadrant_l1552_155229

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem complex_point_in_second_quadrant (a : ℝ) (z : ℂ) 
  (h1 : z = a + Complex.I) 
  (h2 : Complex.abs z < Real.sqrt 2) : 
  is_in_second_quadrant (a - 1) 1 := by
  sorry

#check complex_point_in_second_quadrant

end NUMINAMATH_CALUDE_complex_point_in_second_quadrant_l1552_155229


namespace NUMINAMATH_CALUDE_tan_double_angle_special_case_l1552_155289

/-- Given a function f(x) = sin x + cos x with f'(x) = 3f(x), prove that tan 2x = -4/3 -/
theorem tan_double_angle_special_case (f : ℝ → ℝ) (h1 : ∀ x, f x = Real.sin x + Real.cos x) 
  (h2 : ∀ x, deriv f x = 3 * f x) : 
  ∀ x, Real.tan (2 * x) = -4/3 := by sorry

end NUMINAMATH_CALUDE_tan_double_angle_special_case_l1552_155289


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l1552_155218

/-- The number formed by concatenating digits a, 7, 1, 9 in that order -/
def number (a : ℕ) : ℕ := a * 1000 + 719

/-- The alternating sum of digits used in the divisibility rule for 11 -/
def alternating_sum (a : ℕ) : ℤ := a - 7 + 1 - 9

theorem divisible_by_eleven (a : ℕ) : 
  (0 ≤ a ∧ a ≤ 9) → (number a % 11 = 0 ↔ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l1552_155218


namespace NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1552_155266

theorem base_b_not_divisible_by_five (b : ℤ) (h : b ∈ ({3, 5, 7, 10, 12} : Set ℤ)) : 
  ¬ (5 ∣ ((b - 1)^2)) := by
sorry

end NUMINAMATH_CALUDE_base_b_not_divisible_by_five_l1552_155266


namespace NUMINAMATH_CALUDE_power_calculation_l1552_155282

theorem power_calculation : 16^10 * 8^12 / 4^28 = 2^20 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l1552_155282


namespace NUMINAMATH_CALUDE_exists_noninteger_zero_point_l1552_155212

/-- Definition of the polynomial p(x,y) -/
def p (b : Fin 12 → ℝ) (x y : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4

/-- The theorem stating the existence of a non-integer point (r,s) where p(r,s) = 0 -/
theorem exists_noninteger_zero_point :
  ∃ (r s : ℝ), ¬(∃ m n : ℤ, (r : ℝ) = m ∧ (s : ℝ) = n) ∧
    ∀ (b : Fin 12 → ℝ), 
      p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧ 
      p b 0 1 = 0 ∧ p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧ 
      p b 1 (-1) = 0 ∧ p b 2 2 = 0 ∧ p b (-1) (-1) = 0 →
      p b r s = 0 :=
sorry

end NUMINAMATH_CALUDE_exists_noninteger_zero_point_l1552_155212


namespace NUMINAMATH_CALUDE_function_value_comparison_l1552_155279

theorem function_value_comparison (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x = x^2 + 2*x*(deriv f 2)) : f (-1) > f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_comparison_l1552_155279


namespace NUMINAMATH_CALUDE_triangle_properties_l1552_155214

/-- An acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2

/-- The theorem stating the properties of the specific triangle -/
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.a = 2 * t.b * Real.sin t.A)
  (h2 : t.a = 3 * Real.sqrt 3)
  (h3 : t.c = 5) :
  t.B = π/6 ∧ t.b = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1552_155214


namespace NUMINAMATH_CALUDE_sandwich_counts_l1552_155253

def is_valid_sandwich_count (s : ℕ) : Prop :=
  ∃ (c : ℕ), 
    s + c = 7 ∧ 
    (100 * s + 75 * c) % 100 = 0

theorem sandwich_counts : 
  ∀ s : ℕ, is_valid_sandwich_count s ↔ (s = 3 ∨ s = 7) :=
by sorry

end NUMINAMATH_CALUDE_sandwich_counts_l1552_155253


namespace NUMINAMATH_CALUDE_parallelogram_area_36_24_l1552_155281

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 36 cm and height 24 cm is 864 cm² -/
theorem parallelogram_area_36_24 :
  parallelogram_area 36 24 = 864 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_36_24_l1552_155281


namespace NUMINAMATH_CALUDE_find_M_l1552_155290

theorem find_M : ∃ M : ℝ, (0.2 * M = 0.6 * 1230) ∧ (M = 3690) := by sorry

end NUMINAMATH_CALUDE_find_M_l1552_155290


namespace NUMINAMATH_CALUDE_square_diagonals_equal_l1552_155275

/-- A structure representing a parallelogram -/
structure Parallelogram :=
  (diagonals_equal : Bool)

/-- A structure representing a square, which is a special case of a parallelogram -/
structure Square extends Parallelogram

/-- Theorem stating that the diagonals of a parallelogram are equal -/
axiom parallelogram_diagonals_equal :
  ∀ (p : Parallelogram), p.diagonals_equal = true

/-- Theorem stating that a square is a parallelogram -/
axiom square_is_parallelogram :
  ∀ (s : Square), ∃ (p : Parallelogram), s = ⟨p⟩

/-- Theorem to prove: The diagonals of a square are equal -/
theorem square_diagonals_equal (s : Square) :
  s.diagonals_equal = true := by sorry

end NUMINAMATH_CALUDE_square_diagonals_equal_l1552_155275


namespace NUMINAMATH_CALUDE_shelly_friends_in_classes_l1552_155260

/-- The number of friends Shelly made in classes -/
def friends_in_classes : ℕ := sorry

/-- The number of friends Shelly made in after-school clubs -/
def friends_in_clubs : ℕ := sorry

/-- The amount of thread needed for each keychain in inches -/
def thread_per_keychain : ℕ := 12

/-- The total amount of thread needed in inches -/
def total_thread : ℕ := 108

/-- Theorem stating that Shelly made 6 friends in classes -/
theorem shelly_friends_in_classes : 
  friends_in_classes = 6 ∧
  friends_in_clubs = friends_in_classes / 2 ∧
  friends_in_classes * thread_per_keychain + friends_in_clubs * thread_per_keychain = total_thread :=
sorry

end NUMINAMATH_CALUDE_shelly_friends_in_classes_l1552_155260


namespace NUMINAMATH_CALUDE_calculate_principal_l1552_155246

/-- Given simple interest, time, and rate, calculate the principal amount -/
theorem calculate_principal (simple_interest : ℝ) (time : ℝ) (rate : ℝ) :
  simple_interest = 140 ∧ time = 2 ∧ rate = 17.5 →
  (simple_interest / (rate * time / 100) : ℝ) = 400 := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_l1552_155246


namespace NUMINAMATH_CALUDE_reciprocal_of_point_six_repeating_l1552_155288

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  d / (1 - (1/10))

theorem reciprocal_of_point_six_repeating :
  (repeating_decimal_to_fraction (6/10))⁻¹ = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_point_six_repeating_l1552_155288


namespace NUMINAMATH_CALUDE_expansion_coefficient_l1552_155206

/-- The coefficient of x^(3/2) in the expansion of (√x - a/x)^6 -/
def coefficient (a : ℝ) : ℝ := 6 * (-a)

theorem expansion_coefficient (a : ℝ) : coefficient a = 30 → a = -5 := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l1552_155206


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1552_155205

/-- Represents a seating arrangement around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 seats -/
def are_adjacent (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are across from each other on a round table with 12 seats -/
def are_across (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Checks if a seating arrangement is valid according to the problem constraints -/
def is_valid_arrangement (arr : SeatingArrangement) (couples : Fin 6 → Fin 12 × Fin 12) : Prop :=
  ∀ i j : Fin 12,
    (i ≠ j) →
    (¬are_adjacent (arr i) (arr j)) ∧
    (¬are_across (arr i) (arr j)) ∧
    (∀ k : Fin 6, (couples k).1 ≠ i ∨ (couples k).2 ≠ j)

/-- The main theorem stating the number of valid seating arrangements -/
theorem seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Fin 12 × Fin 12),
    (∀ arr ∈ arrangements, is_valid_arrangement arr couples) ∧
    arrangements.card = 1440 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1552_155205


namespace NUMINAMATH_CALUDE_commission_for_8000_l1552_155276

/-- Represents the commission structure of a bank -/
structure BankCommission where
  /-- Fixed fee for any withdrawal -/
  fixed_fee : ℝ
  /-- Proportional fee rate for withdrawal amount -/
  prop_rate : ℝ

/-- Calculates the commission for a given withdrawal amount -/
def calculate_commission (bc : BankCommission) (amount : ℝ) : ℝ :=
  bc.fixed_fee + bc.prop_rate * amount

theorem commission_for_8000 :
  ∀ (bc : BankCommission),
    calculate_commission bc 5000 = 110 →
    calculate_commission bc 11000 = 230 →
    calculate_commission bc 8000 = 170 := by
  sorry

end NUMINAMATH_CALUDE_commission_for_8000_l1552_155276


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1552_155274

theorem inequality_system_solution (x : ℝ) : 
  (7 - 2*(x + 1) ≥ 1 - 6*x ∧ (1 + 2*x) / 3 > x - 1) ↔ -1 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1552_155274


namespace NUMINAMATH_CALUDE_root_in_interval_l1552_155219

noncomputable def f (x : ℝ) := Real.exp x + 4 * x - 3

theorem root_in_interval :
  ∃! r : ℝ, r ∈ Set.Ioo (1/4 : ℝ) (1/2 : ℝ) ∧ f r = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l1552_155219


namespace NUMINAMATH_CALUDE_sum_of_digits_11_pow_2010_l1552_155242

/-- The sum of the tens digit and the units digit in the decimal representation of 11^2010 is 1. -/
theorem sum_of_digits_11_pow_2010 : ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 11^2010 % 100 = 10 * a + b ∧ a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_11_pow_2010_l1552_155242


namespace NUMINAMATH_CALUDE_polynomial_identity_l1552_155267

theorem polynomial_identity (p : ℝ → ℝ) 
  (h1 : ∀ x, p (x^2 + 1) = (p x)^2 + 1) 
  (h2 : p 0 = 0) : 
  ∀ x, p x = x := by sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1552_155267


namespace NUMINAMATH_CALUDE_triangle_inradius_l1552_155210

/-- Given a triangle with perimeter 32 cm and area 40 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : P = 32) 
  (h_area : A = 40) 
  (h_inradius : A = r * (P / 2)) : 
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1552_155210


namespace NUMINAMATH_CALUDE_revenue_equals_scientific_notation_l1552_155258

/-- Represents the total revenue in yuan -/
def total_revenue : ℝ := 998.64e9

/-- Represents the scientific notation of the total revenue -/
def scientific_notation : ℝ := 9.9864e11

/-- Theorem stating that the total revenue is equal to its scientific notation representation -/
theorem revenue_equals_scientific_notation : total_revenue = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_revenue_equals_scientific_notation_l1552_155258


namespace NUMINAMATH_CALUDE_sum_ends_with_1379_l1552_155223

theorem sum_ends_with_1379 (S : Finset ℕ) (h1 : S.card = 10000) 
  (h2 : ∀ n ∈ S, Odd n ∧ ¬(5 ∣ n)) : 
  ∃ T ⊆ S, (T.sum id) % 10000 = 1379 := by
sorry

end NUMINAMATH_CALUDE_sum_ends_with_1379_l1552_155223


namespace NUMINAMATH_CALUDE_pencil_distribution_l1552_155277

/-- Given a class with 8 students and 120 pencils, prove that when the pencils are divided equally,
    each student receives 15 pencils. -/
theorem pencil_distribution (num_students : ℕ) (num_pencils : ℕ) (pencils_per_student : ℕ) 
    (h1 : num_students = 8)
    (h2 : num_pencils = 120)
    (h3 : num_pencils = num_students * pencils_per_student) :
  pencils_per_student = 15 := by
  sorry

end NUMINAMATH_CALUDE_pencil_distribution_l1552_155277


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_one_l1552_155234

theorem fraction_zero_implies_a_equals_one (a : ℝ) : 
  (|a| - 1) / (a + 1) = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_one_l1552_155234


namespace NUMINAMATH_CALUDE_parabola_focus_point_slope_l1552_155228

/-- The slope of line AF for a parabola y² = 4x with focus F(1,0) and point A on the parabola -/
theorem parabola_focus_point_slope (A : ℝ × ℝ) : 
  A.1 > 0 → -- A is in the first quadrant
  A.2 > 0 →
  A.1 + 1 = 5 → -- distance from A to directrix x = -1 is 5
  A.2^2 = 4 * A.1 → -- A is on the parabola y² = 4x
  (A.2 - 0) / (A.1 - 1) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_parabola_focus_point_slope_l1552_155228


namespace NUMINAMATH_CALUDE_banana_pile_count_l1552_155249

/-- The total number of bananas in a pile after adding more bananas -/
def total_bananas (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: Given 2 initial bananas and 7 added bananas, the total is 9 -/
theorem banana_pile_count : total_bananas 2 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_banana_pile_count_l1552_155249


namespace NUMINAMATH_CALUDE_c2h5cl_formed_equals_c2h6_used_l1552_155295

-- Define the chemical reaction
structure Reaction where
  c2h6 : ℝ
  cl2 : ℝ
  c2h5cl : ℝ
  hcl : ℝ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.c2h6 = r.cl2 ∧ r.c2h6 = r.c2h5cl ∧ r.c2h6 = r.hcl

-- Theorem statement
theorem c2h5cl_formed_equals_c2h6_used 
  (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : r.c2h6 = 3) 
  (h3 : r.c2h5cl = 3) : 
  r.c2h5cl = r.c2h6 := by
  sorry


end NUMINAMATH_CALUDE_c2h5cl_formed_equals_c2h6_used_l1552_155295


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1552_155285

/-- The constant term in the expansion of y^3 * (x + 1/(x^2*y))^n if it exists -/
def constantTerm (n : ℕ+) : ℕ :=
  if n = 9 then 84 else 0

theorem constant_term_expansion (n : ℕ+) :
  (∃ k : ℕ, k ≠ 0 ∧ constantTerm n = k) →
  constantTerm n = 84 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1552_155285


namespace NUMINAMATH_CALUDE_max_students_is_eight_l1552_155200

/-- Represents the relation of two students knowing each other -/
def knows (n : ℕ) : (Fin n → Fin n → Prop) := sorry

/-- The property that in any group of 3 students, at least 2 know each other -/
def three_two_know (n : ℕ) (knows : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- The property that in any group of 4 students, at least 2 do not know each other -/
def four_two_dont_know (n : ℕ) (knows : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students_is_eight :
  (∃ (n : ℕ), n = 8 ∧
    three_two_know n (knows n) ∧
    four_two_dont_know n (knows n)) ∧
  (∀ (m : ℕ), m > 8 →
    ¬(three_two_know m (knows m) ∧
      four_two_dont_know m (knows m))) :=
by sorry

end NUMINAMATH_CALUDE_max_students_is_eight_l1552_155200


namespace NUMINAMATH_CALUDE_fourth_sunday_january_l1552_155299

-- Define the year N
def N : ℕ := sorry

-- Define the day of the week as an enumeration
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

-- Define a function to determine if a year is a leap year
def isLeapYear (year : ℕ) : Bool := sorry

-- Define a function to get the next day of the week
def nextDay (day : DayOfWeek) : DayOfWeek := sorry

-- Define a function to add days to a given day of the week
def addDays (start : DayOfWeek) (days : ℕ) : DayOfWeek := sorry

-- State the theorem
theorem fourth_sunday_january (h1 : 2000 < N ∧ N < 2100)
  (h2 : addDays DayOfWeek.Tuesday 364 = DayOfWeek.Tuesday)
  (h3 : addDays (nextDay (addDays DayOfWeek.Tuesday 364)) 730 = DayOfWeek.Friday)
  : addDays DayOfWeek.Saturday 22 = DayOfWeek.Sunday := by
  sorry

end NUMINAMATH_CALUDE_fourth_sunday_january_l1552_155299


namespace NUMINAMATH_CALUDE_equation_solution_l1552_155292

theorem equation_solution : ∀ x : ℝ, x^2 - 2*x - 3 = x + 7 → x = 5 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1552_155292


namespace NUMINAMATH_CALUDE_asian_games_touring_routes_l1552_155284

theorem asian_games_touring_routes :
  let total_cities : ℕ := 7
  let cities_to_visit : ℕ := 5
  let mandatory_cities : ℕ := 2
  let remaining_cities : ℕ := total_cities - mandatory_cities
  let cities_to_choose : ℕ := cities_to_visit - mandatory_cities
  let gaps : ℕ := cities_to_choose + 1

  (remaining_cities.factorial / (remaining_cities - cities_to_choose).factorial) *
  (gaps.choose mandatory_cities) = 600 :=
by sorry

end NUMINAMATH_CALUDE_asian_games_touring_routes_l1552_155284


namespace NUMINAMATH_CALUDE_no_integer_b_with_two_distinct_roots_l1552_155269

theorem no_integer_b_with_two_distinct_roots :
  ¬ ∃ (b : ℤ), ∃ (x y : ℤ), x ≠ y ∧
    x^4 + 4*x^3 + b*x^2 + 16*x + 8 = 0 ∧
    y^4 + 4*y^3 + b*y^2 + 16*y + 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_integer_b_with_two_distinct_roots_l1552_155269


namespace NUMINAMATH_CALUDE_polynomial_equation_solution_l1552_155245

theorem polynomial_equation_solution : 
  ∃ x : ℝ, ((x^3 * 0.76^3 - 0.008) / (x^2 * 0.76^2 + x * 0.76 * 0.2 + 0.04) = 0) ∧ 
  (abs (x - 0.262) < 0.001) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equation_solution_l1552_155245


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1552_155252

theorem arithmetic_mean_problem (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 4 → x = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1552_155252


namespace NUMINAMATH_CALUDE_g_behavior_at_infinity_l1552_155225

def g (x : ℝ) : ℝ := -3 * x^4 + 15 * x^2 - 10

theorem g_behavior_at_infinity :
  (∀ ε > 0, ∃ N > 0, ∀ x : ℝ, x > N → g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x : ℝ, x < -N → g x < -ε) :=
sorry

end NUMINAMATH_CALUDE_g_behavior_at_infinity_l1552_155225


namespace NUMINAMATH_CALUDE_car_truck_distance_difference_l1552_155294

theorem car_truck_distance_difference 
  (truck_distance : ℝ) 
  (truck_time : ℝ) 
  (car_time : ℝ) 
  (speed_difference : ℝ) 
  (h1 : truck_distance = 296)
  (h2 : truck_time = 8)
  (h3 : car_time = 5.5)
  (h4 : speed_difference = 18) : 
  let truck_speed := truck_distance / truck_time
  let car_speed := truck_speed + speed_difference
  let car_distance := car_speed * car_time
  car_distance - truck_distance = 6.5 := by
sorry

end NUMINAMATH_CALUDE_car_truck_distance_difference_l1552_155294


namespace NUMINAMATH_CALUDE_binding_cost_per_manuscript_l1552_155241

/-- Proves that the binding cost per manuscript is $5 given the specified conditions. -/
theorem binding_cost_per_manuscript
  (num_manuscripts : ℕ)
  (pages_per_manuscript : ℕ)
  (copy_cost_per_page : ℚ)
  (total_cost : ℚ)
  (h1 : num_manuscripts = 10)
  (h2 : pages_per_manuscript = 400)
  (h3 : copy_cost_per_page = 5 / 100)
  (h4 : total_cost = 250) :
  (total_cost - (num_manuscripts * pages_per_manuscript * copy_cost_per_page)) / num_manuscripts = 5 :=
by sorry

end NUMINAMATH_CALUDE_binding_cost_per_manuscript_l1552_155241


namespace NUMINAMATH_CALUDE_circle_center_sum_l1552_155286

/-- Given a circle with equation x^2 + y^2 = 4x - 12y - 8, 
    the sum of the coordinates of its center is -4. -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 12*y - 8) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 + 8) ∧ h + k = -4) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l1552_155286


namespace NUMINAMATH_CALUDE_cubic_polynomial_bound_l1552_155283

theorem cubic_polynomial_bound (p q r : ℝ) : 
  ∃ x : ℝ, x ∈ Set.Icc (-1 : ℝ) 1 ∧ |x^3 + p*x^2 + q*x + r| ≥ (1/4 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_bound_l1552_155283


namespace NUMINAMATH_CALUDE_wind_speed_calculation_l1552_155201

/-- Given a jet's flight conditions, prove the wind speed is 50 mph -/
theorem wind_speed_calculation (j w : ℝ) 
  (h1 : 2000 = (j + w) * 4)   -- Equation for flight with tailwind
  (h2 : 2000 = (j - w) * 5)   -- Equation for return flight against wind
  : w = 50 := by
  sorry

end NUMINAMATH_CALUDE_wind_speed_calculation_l1552_155201


namespace NUMINAMATH_CALUDE_intersection_volume_is_constant_l1552_155221

def cube_side_length : ℝ := 6

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def is_inside_cube (p : Point3D) : Prop :=
  0 < p.x ∧ p.x < cube_side_length ∧
  0 < p.y ∧ p.y < cube_side_length ∧
  0 < p.z ∧ p.z < cube_side_length

def intersection_volume (p : Point3D) : ℝ :=
  cube_side_length ^ 3 - cube_side_length ^ 2

theorem intersection_volume_is_constant (p : Point3D) (h : is_inside_cube p) :
  intersection_volume p = 180 := by sorry

end NUMINAMATH_CALUDE_intersection_volume_is_constant_l1552_155221


namespace NUMINAMATH_CALUDE_sofia_running_time_l1552_155236

/-- The time Sofia takes to complete 8 laps on a track with given conditions -/
theorem sofia_running_time (laps : ℕ) (track_length : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ)
  (h1 : laps = 8)
  (h2 : track_length = 300)
  (h3 : first_half_speed = 5)
  (h4 : second_half_speed = 6) :
  let time_per_lap := track_length / (2 * first_half_speed) + track_length / (2 * second_half_speed)
  let total_time := laps * time_per_lap
  total_time = 440 := by sorry

#eval (7 * 60 + 20 : ℕ) -- Evaluates to 440, confirming 7 minutes and 20 seconds

end NUMINAMATH_CALUDE_sofia_running_time_l1552_155236


namespace NUMINAMATH_CALUDE_balloon_permutations_l1552_155220

def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2)

theorem balloon_permutations :
  balloon_arrangements = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l1552_155220


namespace NUMINAMATH_CALUDE_number_of_small_boxes_l1552_155268

/-- Given a large box containing small boxes of chocolates, this theorem proves
    the number of small boxes given the total number of chocolates and
    the number of chocolates per small box. -/
theorem number_of_small_boxes
  (total_chocolates : ℕ)
  (chocolates_per_box : ℕ)
  (h1 : total_chocolates = 400)
  (h2 : chocolates_per_box = 25)
  (h3 : total_chocolates % chocolates_per_box = 0) :
  total_chocolates / chocolates_per_box = 16 := by
  sorry

#check number_of_small_boxes

end NUMINAMATH_CALUDE_number_of_small_boxes_l1552_155268


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l1552_155271

theorem sqrt_sum_equality (x : ℝ) :
  Real.sqrt (x^2 - 2*x + 4) + Real.sqrt (x^2 + 2*x + 4) =
  Real.sqrt ((x-1)^2 + 3) + Real.sqrt ((x+1)^2 + 3) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l1552_155271


namespace NUMINAMATH_CALUDE_i_to_2016_l1552_155227

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_to_2016 : i^2016 = 1 := by
  sorry

end NUMINAMATH_CALUDE_i_to_2016_l1552_155227


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1552_155235

/-- Given a rectangular field with perimeter 120 meters and length three times the width,
    prove that its area is 675 square meters. -/
theorem rectangular_field_area (l w : ℝ) : 
  (2 * l + 2 * w = 120) → 
  (l = 3 * w) → 
  (l * w = 675) := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1552_155235


namespace NUMINAMATH_CALUDE_equation_solution_l1552_155204

theorem equation_solution (x : ℝ) : 
  (Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 4) ↔ 
  (x = 15 + 4 * Real.sqrt 11 ∨ x = 15 - 4 * Real.sqrt 11) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1552_155204


namespace NUMINAMATH_CALUDE_expression_evaluation_l1552_155222

theorem expression_evaluation :
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3^2 = 197 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1552_155222


namespace NUMINAMATH_CALUDE_field_distance_l1552_155291

theorem field_distance (D : ℝ) (mary edna lucy : ℝ) : 
  mary = (3/8) * D →
  edna = (2/3) * mary →
  lucy = (5/6) * edna →
  lucy + 4 = mary →
  D = 24 := by
sorry

end NUMINAMATH_CALUDE_field_distance_l1552_155291


namespace NUMINAMATH_CALUDE_chocolate_milk_ounces_l1552_155224

/-- The number of ounces of milk in each glass of chocolate milk. -/
def milk_per_glass : ℚ := 13/2

/-- The number of ounces of chocolate syrup in each glass of chocolate milk. -/
def syrup_per_glass : ℚ := 3/2

/-- The total number of ounces in each glass of chocolate milk. -/
def total_per_glass : ℚ := milk_per_glass + syrup_per_glass

/-- Theorem stating that each glass of chocolate milk contains 8 ounces. -/
theorem chocolate_milk_ounces : total_per_glass = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_ounces_l1552_155224


namespace NUMINAMATH_CALUDE_fractional_unit_problem_l1552_155207

def fractional_unit (n : ℕ) (d : ℕ) : ℚ := 1 / d

def smallest_prime : ℕ := 2

theorem fractional_unit_problem (n d : ℕ) (h : n = 13 ∧ d = 5) :
  let x := fractional_unit n d
  x = 1/5 ∧ n * x - 3 * x = smallest_prime := by sorry

end NUMINAMATH_CALUDE_fractional_unit_problem_l1552_155207


namespace NUMINAMATH_CALUDE_system_solution_l1552_155297

theorem system_solution (x y z : ℝ) : 
  x^4 + y^2 + 4 = 5*y*z ∧
  y^4 + z^2 + 4 = 5*z*x ∧
  z^4 + x^2 + 4 = 5*x*y →
  (x = y ∧ y = z ∧ (x = Real.sqrt 2 ∨ x = -Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1552_155297


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1552_155226

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 2 + a 5 + a 8 + a 11 = 48 →
  a 6 + a 7 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1552_155226


namespace NUMINAMATH_CALUDE_group_size_calculation_l1552_155203

theorem group_size_calculation (n : ℕ) : 
  (15 * n + 35) / (n + 1) = 17 → n = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_calculation_l1552_155203


namespace NUMINAMATH_CALUDE_sum_and_multiply_l1552_155231

theorem sum_and_multiply : (57.6 + 1.4) * 3 = 177 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_multiply_l1552_155231


namespace NUMINAMATH_CALUDE_f_nonnegative_and_a_range_f_unique_zero_l1552_155255

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem f_nonnegative_and_a_range (a : ℝ) (h : a > 0) :
  (∀ x > 0, f a x ≥ 0) ∧ a ≥ 1 := by sorry

theorem f_unique_zero (a : ℝ) (h : 0 < a ∧ a ≤ 2/3) :
  ∃! x, x > -a ∧ f a x = 0 := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_and_a_range_f_unique_zero_l1552_155255


namespace NUMINAMATH_CALUDE_work_completion_time_l1552_155202

/-- The time it takes for worker A to complete the work alone -/
def time_A : ℝ := 12

/-- The time it takes for workers A and B to complete the work together -/
def time_AB : ℝ := 7.2

/-- The time it takes for worker B to complete the work alone -/
def time_B : ℝ := 18

/-- Theorem stating that given the time for A and the time for A and B together,
    we can prove that B takes 18 days to complete the work alone -/
theorem work_completion_time :
  (1 / time_A + 1 / time_B = 1 / time_AB) → time_B = 18 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l1552_155202


namespace NUMINAMATH_CALUDE_total_triangles_is_nine_l1552_155257

/-- Represents a triangular grid with a specific number of rows and triangles per row. -/
structure TriangularGrid where
  rows : Nat
  triangles_per_row : Nat → Nat
  row_count_correct : rows = 3
  top_row_correct : triangles_per_row 0 = 3
  second_row_correct : triangles_per_row 1 = 2
  bottom_row_correct : triangles_per_row 2 = 1

/-- Calculates the total number of triangles in the grid, including larger triangles formed by combining smaller ones. -/
def totalTriangles (grid : TriangularGrid) : Nat :=
  sorry

/-- Theorem stating that the total number of triangles in the specified triangular grid is 9. -/
theorem total_triangles_is_nine (grid : TriangularGrid) : totalTriangles grid = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_triangles_is_nine_l1552_155257


namespace NUMINAMATH_CALUDE_polynomial_expansion_properties_l1552_155298

theorem polynomial_expansion_properties (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ = 1) ∧
  (|a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| = 243) ∧
  (a₁ + a₃ + a₅ = 122) ∧
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_properties_l1552_155298


namespace NUMINAMATH_CALUDE_vector_norm_condition_l1552_155244

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given non-zero vectors a and b, a = -2b is a sufficient but not necessary condition
    for |a| - |b| = |a + b| --/
theorem vector_norm_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a = -2 • b → ‖a‖ - ‖b‖ = ‖a + b‖) ∧
  ¬(‖a‖ - ‖b‖ = ‖a + b‖ → a = -2 • b) :=
sorry

end NUMINAMATH_CALUDE_vector_norm_condition_l1552_155244


namespace NUMINAMATH_CALUDE_age_difference_l1552_155293

theorem age_difference (C D m : ℕ) : 
  C = D + m →                    -- Chris is m years older than Daniel
  C - 1 = 3 * (D - 1) →          -- Last year Chris was 3 times as old as Daniel
  C * D = 72 →                   -- This year, the product of their ages is 72
  m = 9 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l1552_155293


namespace NUMINAMATH_CALUDE_intersection_sum_l1552_155259

/-- Given two lines that intersect at (4,3), prove that a + b = 7/4 -/
theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (3/4) * y + a ↔ y = (3/4) * x + b) → 
  (4 = (3/4) * 3 + a ∧ 3 = (3/4) * 4 + b) →
  a + b = 7/4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l1552_155259


namespace NUMINAMATH_CALUDE_trip_price_calculation_egypt_trip_price_l1552_155272

theorem trip_price_calculation (num_people : ℕ) (discount_per_person : ℕ) (total_cost_after_discount : ℕ) : ℕ :=
  let total_discount := num_people * discount_per_person
  let total_cost_before_discount := total_cost_after_discount + total_discount
  let original_price_per_person := total_cost_before_discount / num_people
  original_price_per_person

theorem egypt_trip_price : 
  trip_price_calculation 2 14 266 = 147 := by
  sorry

end NUMINAMATH_CALUDE_trip_price_calculation_egypt_trip_price_l1552_155272


namespace NUMINAMATH_CALUDE_equation_solutions_l1552_155243

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 4 ↔ x = 4 ∨ x = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1552_155243


namespace NUMINAMATH_CALUDE_base6_arithmetic_l1552_155278

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 * 6^3 + d2 * 6^2 + d3 * 6 + d4

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ :=
  let d1 := n / (6^3)
  let d2 := (n / (6^2)) % 6
  let d3 := (n / 6) % 6
  let d4 := n % 6
  d1 * 1000 + d2 * 100 + d3 * 10 + d4

/-- The main theorem to prove --/
theorem base6_arithmetic : 
  base10ToBase6 (base6ToBase10 4512 - base6ToBase10 2324 + base6ToBase10 1432) = 4020 := by
  sorry

end NUMINAMATH_CALUDE_base6_arithmetic_l1552_155278


namespace NUMINAMATH_CALUDE_exactly_one_integer_solution_l1552_155270

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the property for (3n+i)^6 to be an integer
def is_integer_power (n : ℤ) : Prop :=
  ∃ m : ℤ, (3 * n + i : ℂ)^6 = m

-- Theorem statement
theorem exactly_one_integer_solution :
  ∃! n : ℤ, is_integer_power n :=
sorry

end NUMINAMATH_CALUDE_exactly_one_integer_solution_l1552_155270


namespace NUMINAMATH_CALUDE_sequence_term_l1552_155248

theorem sequence_term (a : ℕ → ℝ) (h : ∀ n, a n = Real.sqrt (3 * n - 1)) :
  a 7 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_l1552_155248


namespace NUMINAMATH_CALUDE_hyperbola_a_value_l1552_155254

-- Define the hyperbola equation
def hyperbola_eq (x y a : ℝ) : Prop := x^2 / (a + 3) - y^2 / 3 = 1

-- Define the eccentricity
def eccentricity : ℝ := 2

-- Theorem statement
theorem hyperbola_a_value :
  ∃ (a : ℝ), (∀ (x y : ℝ), hyperbola_eq x y a) ∧ 
  (eccentricity = 2) → a = -2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_a_value_l1552_155254


namespace NUMINAMATH_CALUDE_calculate_expression_l1552_155296

theorem calculate_expression : (-3)^0 + Real.sqrt 8 + (-3)^2 - 4 * (Real.sqrt 2 / 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1552_155296


namespace NUMINAMATH_CALUDE_chime_2400_date_l1552_155232

/-- Represents a date in the year 2004 --/
structure Date2004 where
  month : Nat
  day : Nat

/-- Represents a time of day --/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the number of chimes for a given hour --/
def chimesForHour (hour : Nat) : Nat :=
  3 * (hour % 12 + if hour % 12 = 0 then 12 else 0)

/-- Calculates the total chimes from the start time to midnight --/
def chimesToMidnight (startTime : Time) : Nat :=
  sorry

/-- Calculates the total chimes for a full day --/
def chimesPerDay : Nat :=
  258

/-- Determines the date when the nth chime occurs --/
def dateOfNthChime (n : Nat) (startDate : Date2004) (startTime : Time) : Date2004 :=
  sorry

theorem chime_2400_date :
  dateOfNthChime 2400 ⟨2, 28⟩ ⟨17, 45⟩ = ⟨3, 7⟩ := by sorry

end NUMINAMATH_CALUDE_chime_2400_date_l1552_155232


namespace NUMINAMATH_CALUDE_number_exceeding_percentage_l1552_155238

theorem number_exceeding_percentage : 
  ∃ (x : ℝ), x = 200 ∧ x = 0.25 * x + 150 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeding_percentage_l1552_155238


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1552_155237

/-- Given a geometric sequence {a_n} with specific conditions, prove that a_1 = -1/2 -/
theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) 
  (h_product : a 2 * a 5 * a 8 = -8) 
  (h_sum : a 1 + a 2 + a 3 = a 2 + 3 * a 1) : 
  a 1 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1552_155237


namespace NUMINAMATH_CALUDE_arithmetic_sequence_500th_term_l1552_155233

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_500th_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p 9 2 = 9)
  (h2 : arithmetic_sequence p 9 3 = 3*p - q^3)
  (h3 : arithmetic_sequence p 9 4 = 3*p + q^3) :
  arithmetic_sequence p 9 500 = 2005 - 2 * Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_500th_term_l1552_155233


namespace NUMINAMATH_CALUDE_circle_problem_l1552_155213

-- Define the circles and points
def largeCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 100}
def smallCircle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 36}
def P : ℝ × ℝ := (6, 8)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem circle_problem (k : ℝ) 
  (h1 : P ∈ largeCircle) 
  (h2 : S k ∈ smallCircle) 
  (h3 : (10 : ℝ) - (6 : ℝ) = 4) : 
  k = 6 := by sorry

end NUMINAMATH_CALUDE_circle_problem_l1552_155213


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1552_155216

/-- Circle C: x^2 + y^2 - 2x + 2y - 4 = 0 -/
def Circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - 4 = 0

/-- Line l: y = x + b with slope 1 -/
def Line (x y b : ℝ) : Prop := y = x + b

/-- Intersection points of Circle and Line -/
def Intersection (b : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ, 
  x₁ ≠ x₂ ∧ Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ b ∧ Line x₂ y₂ b

/-- The circle with diameter AB passes through the origin -/
def CircleThroughOrigin (b : ℝ) : Prop := ∃ x₁ y₁ x₂ y₂ : ℝ,
  Circle x₁ y₁ ∧ Circle x₂ y₂ ∧ Line x₁ y₁ b ∧ Line x₂ y₂ b ∧
  x₁*x₂ + y₁*y₂ = 0

theorem circle_line_intersection :
  (∀ b : ℝ, Intersection b ↔ -3-3*Real.sqrt 2 < b ∧ b < -3+3*Real.sqrt 2) ∧
  (∃! b₁ b₂ : ℝ, b₁ ≠ b₂ ∧ CircleThroughOrigin b₁ ∧ CircleThroughOrigin b₂ ∧
    b₁ = 1 ∧ b₂ = -4) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1552_155216


namespace NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l1552_155251

/-- A line passing through (1, 2) with equal X and Y intercepts has equation 2x - y = 0 or x + y - 3 = 0 -/
theorem line_through_point_equal_intercepts :
  ∀ (a b c : ℝ),
    (a ≠ 0 ∧ b ≠ 0) →
    (a * 1 + b * 2 + c = 0) →  -- Line passes through (1, 2)
    ((-c/a) = (-c/b)) →        -- Equal X and Y intercepts
    ((a = 2 ∧ b = -1 ∧ c = 0) ∨ (a = 1 ∧ b = 1 ∧ c = -3)) := by
  sorry


end NUMINAMATH_CALUDE_line_through_point_equal_intercepts_l1552_155251


namespace NUMINAMATH_CALUDE_sector_to_cone_l1552_155211

/-- Represents a cone formed from a circular sector -/
structure SectorCone where
  sector_radius : ℝ
  sector_angle : ℝ
  base_radius : ℝ
  slant_height : ℝ

/-- Theorem: A 270° sector of a circle with radius 12 forms a cone with base radius 9 and slant height 12 -/
theorem sector_to_cone :
  ∀ (cone : SectorCone),
    cone.sector_radius = 12 ∧
    cone.sector_angle = 270 ∧
    cone.slant_height = cone.sector_radius →
    cone.base_radius = 9 ∧
    cone.slant_height = 12 := by
  sorry


end NUMINAMATH_CALUDE_sector_to_cone_l1552_155211


namespace NUMINAMATH_CALUDE_max_product_l1552_155265

def digits : Finset Nat := {3, 5, 6, 8, 9}

def isValidPair (a b c d e : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧ e ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def threeDigitNum (a b c : Nat) : Nat := 100 * a + 10 * b + c
def twoDigitNum (d e : Nat) : Nat := 10 * d + e

def product (a b c d e : Nat) : Nat :=
  threeDigitNum a b c * twoDigitNum d e

theorem max_product :
  ∀ a b c d e,
    isValidPair a b c d e →
    product a b c d e ≤ product 8 5 9 6 3 :=
by sorry

end NUMINAMATH_CALUDE_max_product_l1552_155265


namespace NUMINAMATH_CALUDE_num_technicians_correct_l1552_155230

/-- The number of technicians in a workshop with given conditions. -/
def num_technicians : ℕ :=
  let total_workers : ℕ := 42
  let avg_salary_all : ℕ := 8000
  let avg_salary_technicians : ℕ := 18000
  let avg_salary_rest : ℕ := 6000
  7

/-- Theorem stating that the number of technicians is correct given the workshop conditions. -/
theorem num_technicians_correct :
  let total_workers : ℕ := 42
  let avg_salary_all : ℕ := 8000
  let avg_salary_technicians : ℕ := 18000
  let avg_salary_rest : ℕ := 6000
  let num_technicians := num_technicians
  let num_rest := total_workers - num_technicians
  (num_technicians * avg_salary_technicians + num_rest * avg_salary_rest) / total_workers = avg_salary_all :=
by
  sorry

#eval num_technicians

end NUMINAMATH_CALUDE_num_technicians_correct_l1552_155230


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_l1552_155262

/-- Represents a trapezoid with sides AB, BC, CD, and DA -/
structure Trapezoid where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ

/-- The perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := t.AB + t.BC + t.CD + t.DA

/-- Theorem: Perimeter of a specific trapezoid ABCD -/
theorem trapezoid_perimeter (x y : ℝ) (hx : x ≠ 0) :
  ∃ (ABCD : Trapezoid),
    ABCD.AB = 2 * x ∧
    ABCD.CD = 4 * x ∧
    ABCD.BC = y ∧
    ABCD.DA = 2 * y ∧
    perimeter ABCD = 6 * x + 3 * y := by
  sorry

#check trapezoid_perimeter

end NUMINAMATH_CALUDE_trapezoid_perimeter_l1552_155262
