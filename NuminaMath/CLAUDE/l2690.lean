import Mathlib

namespace NUMINAMATH_CALUDE_ice_cream_bars_per_friend_l2690_269005

/-- Proves that given the conditions of the ice cream problem, each friend wants to eat 2 bars. -/
theorem ice_cream_bars_per_friend 
  (box_cost : ℚ) 
  (bars_per_box : ℕ) 
  (num_friends : ℕ) 
  (cost_per_person : ℚ) 
  (h1 : box_cost = 15/2)
  (h2 : bars_per_box = 3)
  (h3 : num_friends = 6)
  (h4 : cost_per_person = 5) : 
  (num_friends * cost_per_person / box_cost * bars_per_box) / num_friends = 2 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_bars_per_friend_l2690_269005


namespace NUMINAMATH_CALUDE_jacket_cost_is_30_l2690_269042

/-- Represents the cost of clothing items in a discount store. -/
structure ClothingCost where
  sweater : ℝ
  jacket : ℝ

/-- Represents a shipment of clothing items. -/
structure Shipment where
  sweaters : ℕ
  jackets : ℕ
  totalCost : ℝ

/-- The conditions of the problem. -/
def problemConditions (cost : ClothingCost) : Prop :=
  ∃ (shipment1 shipment2 : Shipment),
    shipment1.sweaters = 10 ∧
    shipment1.jackets = 20 ∧
    shipment1.totalCost = 800 ∧
    shipment2.sweaters = 5 ∧
    shipment2.jackets = 15 ∧
    shipment2.totalCost = 550 ∧
    shipment1.totalCost = cost.sweater * shipment1.sweaters + cost.jacket * shipment1.jackets ∧
    shipment2.totalCost = cost.sweater * shipment2.sweaters + cost.jacket * shipment2.jackets

/-- The main theorem stating that under the given conditions, the cost of a jacket is $30. -/
theorem jacket_cost_is_30 :
  ∀ (cost : ClothingCost), problemConditions cost → cost.jacket = 30 := by
  sorry


end NUMINAMATH_CALUDE_jacket_cost_is_30_l2690_269042


namespace NUMINAMATH_CALUDE_range_of_a_l2690_269079

/-- The function f(x) = x^3 + ax^2 - a^2x -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 - a^2*x

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x - a^2

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f_deriv a x ≤ 0) ∧
  (∀ x ∈ Set.Ioi 2, f_deriv a x ≥ 0) →
  a ∈ Set.Icc (-2) (-1) ∪ Set.Icc 3 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2690_269079


namespace NUMINAMATH_CALUDE_event_relationship_l2690_269016

-- Define the critical value
def critical_value : ℝ := 6.635

-- Define the confidence level
def confidence_level : ℝ := 0.99

-- Define the relationship between K^2 and the confidence level
theorem event_relationship (K : ℝ) :
  K^2 > critical_value → confidence_level = 0.99 := by
  sorry

#check event_relationship

end NUMINAMATH_CALUDE_event_relationship_l2690_269016


namespace NUMINAMATH_CALUDE_calculation_proof_l2690_269082

theorem calculation_proof :
  (0.001)^(-1/3) + 27^(2/3) + (1/4)^(-1/2) - (1/9)^(-3/2) = -6 ∧
  1/2 * Real.log 25 / Real.log 10 + Real.log 2 / Real.log 10 - Real.log (Real.sqrt 0.1) / Real.log 10 - 
    (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2690_269082


namespace NUMINAMATH_CALUDE_part_one_part_two_l2690_269020

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Define propositions p and q
def p (x : ℝ) (a : ℝ) : Prop := x ∈ A a
def q (x : ℝ) : Prop := x ∈ B

-- Theorem for part I
theorem part_one (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : A a ∪ B = Set.univ) : a = 2 := by
  sorry

-- Theorem for part II
theorem part_two (a : ℝ) (h : ∀ x, p x a → q x) : a ≤ 0 ∨ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2690_269020


namespace NUMINAMATH_CALUDE_f_properties_l2690_269006

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x) + 1

theorem f_properties :
  (∀ x : ℝ, f (π/12 + x) = f (π/12 - x)) ∧
  (¬ ∀ x ∈ Set.Ioo (5*π/12) (11*π/12), ∀ y ∈ Set.Ioo (5*π/12) (11*π/12), x < y → f x > f y) ∧
  (∀ x : ℝ, f (π/3 + x) = f (π/3 - x)) ∧
  (∀ x : ℝ, f x ≤ 3) ∧ (∃ x : ℝ, f x = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2690_269006


namespace NUMINAMATH_CALUDE_unique_valid_number_l2690_269011

def is_valid_number (n : ℕ) : Prop :=
  let x := n / 100
  let y := (n / 10) % 10
  let z := n % 10
  (100 * x + 10 * z + y = 64 * x + 8 * z + y) ∧ 
  (100 * y + 10 * x + z = 36 * y + 6 * x + z - 16) ∧
  (100 * z + 10 * y + x = 16 * z + 4 * y + x + 18)

theorem unique_valid_number : 
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ is_valid_number n :=
by sorry

end NUMINAMATH_CALUDE_unique_valid_number_l2690_269011


namespace NUMINAMATH_CALUDE_range_of_a_for_fourth_quadrant_l2690_269057

/-- A point in the fourth quadrant has a positive x-coordinate and a negative y-coordinate -/
def is_in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The x-coordinate of point P -/
def x_coord (a : ℝ) : ℝ := a + 1

/-- The y-coordinate of point P -/
def y_coord (a : ℝ) : ℝ := 2 * a - 3

/-- The theorem stating the range of a for point P to be in the fourth quadrant -/
theorem range_of_a_for_fourth_quadrant :
  ∀ a : ℝ, is_in_fourth_quadrant (x_coord a) (y_coord a) ↔ -1 < a ∧ a < 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_fourth_quadrant_l2690_269057


namespace NUMINAMATH_CALUDE_a_power_b_is_one_fourth_l2690_269034

theorem a_power_b_is_one_fourth (a b : ℝ) (h : (a + b)^2 + |b + 2| = 0) : a^b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_a_power_b_is_one_fourth_l2690_269034


namespace NUMINAMATH_CALUDE_cake_sugar_calculation_l2690_269084

/-- The amount of sugar stored in the house (in pounds) -/
def sugar_stored : ℕ := 287

/-- The amount of additional sugar needed (in pounds) -/
def sugar_additional : ℕ := 163

/-- The total amount of sugar needed for the cake (in pounds) -/
def total_sugar_needed : ℕ := sugar_stored + sugar_additional

theorem cake_sugar_calculation :
  total_sugar_needed = 450 :=
by sorry

end NUMINAMATH_CALUDE_cake_sugar_calculation_l2690_269084


namespace NUMINAMATH_CALUDE_count_numbers_with_2_between_200_and_499_l2690_269040

def count_numbers_with_digit_2 (lower_bound upper_bound : ℕ) : ℕ :=
  sorry

theorem count_numbers_with_2_between_200_and_499 :
  count_numbers_with_digit_2 200 499 = 138 := by
  sorry

end NUMINAMATH_CALUDE_count_numbers_with_2_between_200_and_499_l2690_269040


namespace NUMINAMATH_CALUDE_equation_solution_l2690_269001

theorem equation_solution (k : ℤ) : 
  (∃ x : ℤ, Real.sqrt (39 - 6 * Real.sqrt 12) + Real.sqrt (k * x * (k * x + Real.sqrt 12) + 3) = 2 * k) ↔ 
  (k = 3 ∨ k = 6) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2690_269001


namespace NUMINAMATH_CALUDE_b_over_c_equals_27_l2690_269065

-- Define the coefficients of the quadratic equations
variable (a b c : ℝ)

-- Define the roots of the second equation
variable (s₁ s₂ : ℝ)

-- Assumptions
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom c_nonzero : c ≠ 0

-- Define the relationships between roots and coefficients
axiom vieta_sum : c = -(s₁ + s₂)
axiom vieta_product : a = s₁ * s₂

-- Define the relationship between the roots of the two equations
axiom root_relationship : a = -(3*s₁ + 3*s₂) ∧ b = 9*s₁*s₂

-- Theorem to prove
theorem b_over_c_equals_27 : b / c = 27 := by
  sorry

end NUMINAMATH_CALUDE_b_over_c_equals_27_l2690_269065


namespace NUMINAMATH_CALUDE_johnny_hourly_wage_l2690_269007

/-- Johnny's hourly wage calculation -/
theorem johnny_hourly_wage :
  let hours_worked : ℝ := 6
  let total_earnings : ℝ := 28.5
  let hourly_wage := total_earnings / hours_worked
  hourly_wage = 4.75 := by
sorry

end NUMINAMATH_CALUDE_johnny_hourly_wage_l2690_269007


namespace NUMINAMATH_CALUDE_binomial_coefficient_n_n_l2690_269056

theorem binomial_coefficient_n_n (n : ℕ) : Nat.choose n n = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_n_n_l2690_269056


namespace NUMINAMATH_CALUDE_isosceles_triangulation_condition_l2690_269037

/-- A regular convex polygon with n sides -/
structure RegularConvexPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- A triangulation of a polygon -/
structure Triangulation (P : RegularConvexPolygon) where
  isosceles : Bool

/-- Theorem: If a regular convex polygon with n sides has a triangulation
    consisting of only isosceles triangles, then n can be written as 2^(a+1) + 2^b
    for some non-negative integers a and b -/
theorem isosceles_triangulation_condition (P : RegularConvexPolygon)
  (T : Triangulation P) (h : T.isosceles = true) :
  ∃ (a b : ℕ), P.n = 2^(a+1) + 2^b :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangulation_condition_l2690_269037


namespace NUMINAMATH_CALUDE_min_distance_to_line_l2690_269049

open Complex

theorem min_distance_to_line (z : ℂ) (h : abs (z - 1) = abs (z + 2*I)) :
  ∃ (min_val : ℝ), min_val = (9 * Real.sqrt 5) / 10 ∧
  ∀ (w : ℂ), abs (w - 1) = abs (w + 2*I) → abs (w - 1 - I) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l2690_269049


namespace NUMINAMATH_CALUDE_inverse_inequality_for_negatives_l2690_269073

theorem inverse_inequality_for_negatives (a b : ℝ) : 0 > a → a > b → (1 / a) < (1 / b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_for_negatives_l2690_269073


namespace NUMINAMATH_CALUDE_negation_of_forall_inequality_l2690_269003

theorem negation_of_forall_inequality :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_inequality_l2690_269003


namespace NUMINAMATH_CALUDE_second_race_length_l2690_269033

/-- Given a 100 m race where A beats B by 10 m and C by 13 m, and another race where B beats C by 6 m,
    prove that the length of the second race is 180 meters. -/
theorem second_race_length (vA vB vC : ℝ) (t : ℝ) (h1 : vA * t = 100)
                            (h2 : vB * t = 90) (h3 : vC * t = 87) : 
  ∃ (L : ℝ), L / vB = (L - 6) / vC ∧ L = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_race_length_l2690_269033


namespace NUMINAMATH_CALUDE_triangle_properties_l2690_269068

noncomputable section

/-- Triangle ABC with area S and sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  S : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.S = (3/2) * t.b * t.c * Real.cos t.A)
  (h2 : t.C = π/4)
  (h3 : t.S = 24) :
  Real.cos t.B = Real.sqrt 5 / 5 ∧ t.b = 8 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l2690_269068


namespace NUMINAMATH_CALUDE_functional_equation_l2690_269059

theorem functional_equation (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x - y) = f x * f y) : 
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_l2690_269059


namespace NUMINAMATH_CALUDE_estimated_y_value_l2690_269080

/-- Linear regression equation -/
def linear_regression (x : ℝ) : ℝ := 0.50 * x - 0.81

theorem estimated_y_value (x : ℝ) (h : x = 25) : linear_regression x = 11.69 := by
  sorry

end NUMINAMATH_CALUDE_estimated_y_value_l2690_269080


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2690_269085

/-- Given a hyperbola x^2 - y^2/b^2 = 1 with b > 1, the angle θ between its asymptotes is not 2arctan(b) -/
theorem hyperbola_asymptote_angle (b : ℝ) (h : b > 1) :
  let θ := Real.pi - 2 * Real.arctan b
  θ ≠ 2 * Real.arctan b :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l2690_269085


namespace NUMINAMATH_CALUDE_scaling_transforms_line_l2690_269048

-- Define the original line
def original_line (x y : ℝ) : Prop := x - 2*y = 2

-- Define the transformed line
def transformed_line (x' y' : ℝ) : Prop := 2*x' - y' = 4

-- Define the scaling transformation
def scaling_transformation (x y x' y' : ℝ) : Prop := x' = x ∧ y' = 4*y

-- Theorem statement
theorem scaling_transforms_line :
  ∀ (x y x' y' : ℝ),
    original_line x y →
    scaling_transformation x y x' y' →
    transformed_line x' y' := by
  sorry

end NUMINAMATH_CALUDE_scaling_transforms_line_l2690_269048


namespace NUMINAMATH_CALUDE_total_different_groups_l2690_269028

-- Define the number of marbles of each color
def red_marbles : ℕ := 1
def green_marbles : ℕ := 1
def blue_marbles : ℕ := 1
def yellow_marbles : ℕ := 3

-- Define the total number of distinct colors
def distinct_colors : ℕ := 4

-- Define the function to calculate the number of different groups
def different_groups : ℕ :=
  -- Groups with two yellow marbles
  1 +
  -- Groups with two different colors
  (distinct_colors.choose 2)

-- Theorem statement
theorem total_different_groups :
  different_groups = 7 :=
sorry

end NUMINAMATH_CALUDE_total_different_groups_l2690_269028


namespace NUMINAMATH_CALUDE_object_crosses_x_axis_l2690_269029

/-- The position vector of an object moving in two dimensions -/
def position_vector (t : ℝ) : ℝ × ℝ :=
  (4 * t^2 - 9, 2 * t - 5)

/-- The time when the object crosses the x-axis -/
def crossing_time : ℝ := 2.5

/-- Theorem: The object crosses the x-axis at t = 2.5 seconds -/
theorem object_crosses_x_axis :
  (position_vector crossing_time).2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_object_crosses_x_axis_l2690_269029


namespace NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2690_269009

theorem quadratic_inequality_equivalence (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k-2)*x - k + 8 ≥ 0) ↔ 
  (k ≥ -2 * Real.sqrt 7 ∧ k ≤ 2 * Real.sqrt 7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_equivalence_l2690_269009


namespace NUMINAMATH_CALUDE_xy_equals_zero_l2690_269066

theorem xy_equals_zero (x y : ℝ) (h1 : x + y = 4) (h2 : x^3 - y^3 = 64) : x * y = 0 := by
  sorry

end NUMINAMATH_CALUDE_xy_equals_zero_l2690_269066


namespace NUMINAMATH_CALUDE_geometric_sequence_inequality_l2690_269052

/-- A geometric sequence with positive terms and common ratio not equal to 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h1 : ∀ n, a n > 0
  h2 : q ≠ 1
  h3 : ∀ n, a (n + 1) = q * a n

/-- The sum of the first and eighth terms is greater than the sum of the fourth and fifth terms -/
theorem geometric_sequence_inequality (seq : GeometricSequence) :
  seq.a 1 + seq.a 8 > seq.a 4 + seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_inequality_l2690_269052


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l2690_269088

/-- An increasing arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, d > 0 ∧ ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b → b 4 * b 5 = 21 → b 3 * b 6 = -779 ∨ b 3 * b 6 = -11 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l2690_269088


namespace NUMINAMATH_CALUDE_quadratic_function_k_range_l2690_269051

/-- Given a quadratic function f(x) = 4x^2 - kx - 8 with no maximum or minimum at (5, 20),
    prove that the range of k is k ≤ 40 or k ≥ 160. -/
theorem quadratic_function_k_range (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 4 * x^2 - k * x - 8
  (∀ x, f x ≠ f 5 ∨ (∃ y, y ≠ 5 ∧ f y = f 5)) →
  f 5 = 20 →
  k ≤ 40 ∨ k ≥ 160 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_k_range_l2690_269051


namespace NUMINAMATH_CALUDE_envelope_area_l2690_269019

/-- The area of a rectangular envelope with width and height both 6 inches is 36 square inches. -/
theorem envelope_area (width height : ℝ) (h1 : width = 6) (h2 : height = 6) :
  width * height = 36 := by
  sorry

end NUMINAMATH_CALUDE_envelope_area_l2690_269019


namespace NUMINAMATH_CALUDE_intersection_point_l2690_269030

theorem intersection_point (x y : ℝ) : 
  y = 4 * x - 32 ∧ y = -6 * x + 8 → (x, y) = (4, -16) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l2690_269030


namespace NUMINAMATH_CALUDE_product_divisibility_l2690_269026

def die_numbers : Finset ℕ := Finset.range 8

theorem product_divisibility (visible : Finset ℕ) 
  (h1 : visible ⊆ die_numbers) 
  (h2 : visible.card = 6) : 
  96 ∣ visible.prod id :=
sorry

end NUMINAMATH_CALUDE_product_divisibility_l2690_269026


namespace NUMINAMATH_CALUDE_abs_product_zero_implies_one_equal_one_l2690_269014

theorem abs_product_zero_implies_one_equal_one (a b : ℝ) :
  |a - 1| * |b - 1| = 0 → a = 1 ∨ b = 1 := by
sorry

end NUMINAMATH_CALUDE_abs_product_zero_implies_one_equal_one_l2690_269014


namespace NUMINAMATH_CALUDE_area_between_circles_l2690_269062

theorem area_between_circles (r_small : ℝ) (r_large : ℝ) : 
  r_small = 2 →
  r_large = 5 * r_small →
  (π * r_large^2 - π * r_small^2) = 96 * π := by
sorry

end NUMINAMATH_CALUDE_area_between_circles_l2690_269062


namespace NUMINAMATH_CALUDE_ancient_chinese_problem_correct_l2690_269091

/-- Represents the system of equations for the ancient Chinese math problem --/
def ancient_chinese_problem (x y : ℤ) : Prop :=
  (y = 8*x - 3) ∧ (y = 7*x + 4)

/-- Theorem stating that the system of equations correctly represents the problem --/
theorem ancient_chinese_problem_correct (x y : ℤ) :
  (ancient_chinese_problem x y) ↔
  (x ≥ 0) ∧  -- number of people is non-negative
  (y ≥ 0) ∧  -- price is non-negative
  (8*x - y = 3) ∧  -- excess of 3 coins when each contributes 8
  (y - 7*x = 4)    -- shortage of 4 coins when each contributes 7
  := by sorry

end NUMINAMATH_CALUDE_ancient_chinese_problem_correct_l2690_269091


namespace NUMINAMATH_CALUDE_simplify_expression_l2690_269075

theorem simplify_expression : ∃ (a b c : ℕ+), 
  (Real.sqrt 3 + 2 / Real.sqrt 3 + Real.sqrt 11 + 3 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    Real.sqrt 3 + 2 / Real.sqrt 3 + Real.sqrt 11 + 3 / Real.sqrt 11 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c' →
    c ≤ c') ∧
  a = 39 ∧ b = 42 ∧ c = 33 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l2690_269075


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l2690_269050

theorem largest_prime_factor_of_12321 : ∃ p : ℕ, p.Prime ∧ p ∣ 12321 ∧ ∀ q : ℕ, q.Prime → q ∣ 12321 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_12321_l2690_269050


namespace NUMINAMATH_CALUDE_marble_arrangement_count_l2690_269017

def total_arrangements (n : ℕ) : ℕ := Nat.factorial n

def adjacent_arrangements (n : ℕ) : ℕ := 2 * Nat.factorial (n - 1)

theorem marble_arrangement_count :
  total_arrangements 5 - adjacent_arrangements 5 = 72 := by
  sorry

end NUMINAMATH_CALUDE_marble_arrangement_count_l2690_269017


namespace NUMINAMATH_CALUDE_inverse_proposition_false_l2690_269090

theorem inverse_proposition_false : ∃ a b : ℝ, (abs a = abs b) ∧ (a ≠ b) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proposition_false_l2690_269090


namespace NUMINAMATH_CALUDE_ratio_difference_l2690_269039

theorem ratio_difference (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / y = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_difference_l2690_269039


namespace NUMINAMATH_CALUDE_solution_difference_l2690_269095

theorem solution_difference (r s : ℝ) : 
  (∀ x : ℝ, (5 * x - 15) / (x^2 + 3 * x - 18) = x + 3 → x = r ∨ x = s) →
  r ≠ s →
  r > s →
  r - s = Real.sqrt 29 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l2690_269095


namespace NUMINAMATH_CALUDE_find_k_l2690_269004

def vector_a (k : ℝ) : ℝ × ℝ := (k, 3)
def vector_b : ℝ × ℝ := (1, 4)
def vector_c : ℝ × ℝ := (2, 1)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem find_k : ∃ k : ℝ, 
  perpendicular ((2 * (vector_a k).1 - 3 * vector_b.1, 2 * (vector_a k).2 - 3 * vector_b.2)) vector_c ∧ 
  k = 3 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l2690_269004


namespace NUMINAMATH_CALUDE_tournament_prize_interval_l2690_269018

def total_prize : ℕ := 4800
def first_place_prize : ℕ := 2000

def prize_interval (x : ℕ) : Prop :=
  first_place_prize + (first_place_prize - x) + (first_place_prize - 2*x) = total_prize

theorem tournament_prize_interval : ∃ (x : ℕ), prize_interval x ∧ x = 400 := by
  sorry

end NUMINAMATH_CALUDE_tournament_prize_interval_l2690_269018


namespace NUMINAMATH_CALUDE_wedding_attendance_percentage_l2690_269076

theorem wedding_attendance_percentage 
  (total_invitations : ℕ) 
  (rsvp_rate : ℚ)
  (thank_you_cards : ℕ) 
  (no_gift_attendees : ℕ) :
  total_invitations = 200 →
  rsvp_rate = 9/10 →
  thank_you_cards = 134 →
  no_gift_attendees = 10 →
  (thank_you_cards + no_gift_attendees) / (total_invitations * rsvp_rate) = 4/5 := by
sorry

#eval (134 + 10) / (200 * (9/10)) -- This should evaluate to 4/5

end NUMINAMATH_CALUDE_wedding_attendance_percentage_l2690_269076


namespace NUMINAMATH_CALUDE_range_of_a_l2690_269087

-- Define the set of real numbers a that satisfy the condition
def A : Set ℝ := {a : ℝ | ∀ x : ℝ, a * x^2 + a * x + 1 > 0}

-- Theorem stating that A is equal to the interval [0, 4)
theorem range_of_a : A = Set.Icc 0 (4 : ℝ) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l2690_269087


namespace NUMINAMATH_CALUDE_root_difference_product_l2690_269025

theorem root_difference_product (p q a b c : ℝ) : 
  (a^2 + p*a + 1 = 0) → 
  (b^2 + p*b + 1 = 0) → 
  (b^2 + q*b + 2 = 0) → 
  (c^2 + q*c + 2 = 0) → 
  (b - a) * (b - c) = p*q - 6 := by
sorry

end NUMINAMATH_CALUDE_root_difference_product_l2690_269025


namespace NUMINAMATH_CALUDE_matrix_cube_l2690_269044

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube :
  A ^ 3 = !![3, -6; 6, -3] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_l2690_269044


namespace NUMINAMATH_CALUDE_car_repair_cost_l2690_269036

/-- Calculates the total cost for a car repair given the mechanic's hourly rate,
    hours worked per day, number of days worked, and cost of parts. -/
theorem car_repair_cost (hourly_rate : ℕ) (hours_per_day : ℕ) (days_worked : ℕ) (parts_cost : ℕ) :
  hourly_rate = 60 →
  hours_per_day = 8 →
  days_worked = 14 →
  parts_cost = 2500 →
  hourly_rate * hours_per_day * days_worked + parts_cost = 9220 := by
  sorry

#check car_repair_cost

end NUMINAMATH_CALUDE_car_repair_cost_l2690_269036


namespace NUMINAMATH_CALUDE_floor_paving_cost_l2690_269067

-- Define the room dimensions
def room_length : ℝ := 6
def room_width : ℝ := 4.75

-- Define the cost per square meter
def cost_per_sqm : ℝ := 900

-- Define the function to calculate the area of a rectangle
def area (length width : ℝ) : ℝ := length * width

-- Define the function to calculate the total cost
def total_cost (length width cost_per_sqm : ℝ) : ℝ :=
  area length width * cost_per_sqm

-- State the theorem
theorem floor_paving_cost :
  total_cost room_length room_width cost_per_sqm = 25650 := by sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l2690_269067


namespace NUMINAMATH_CALUDE_no_snow_probability_l2690_269070

theorem no_snow_probability (p : ℚ) : 
  p = 2/3 → (1 - p)^4 = 1/81 := by sorry

end NUMINAMATH_CALUDE_no_snow_probability_l2690_269070


namespace NUMINAMATH_CALUDE_money_redistribution_theorem_l2690_269015

/-- Represents the money redistribution problem among boys and girls -/
theorem money_redistribution_theorem 
  (num_boys : ℕ) 
  (num_girls : ℕ) 
  (boy_initial : ℕ) 
  (girl_initial : ℕ) : 
  num_boys = 9 → 
  num_girls = 3 → 
  boy_initial = 12 → 
  girl_initial = 36 → 
  ∃ (boy_gives girl_gives final_amount : ℕ), 
    (∀ (b : ℕ), b < num_boys → 
      boy_initial - num_girls * boy_gives + num_girls * girl_gives = final_amount) ∧
    (∀ (g : ℕ), g < num_girls → 
      girl_initial - num_boys * girl_gives + num_boys * boy_gives = final_amount) := by
  sorry

end NUMINAMATH_CALUDE_money_redistribution_theorem_l2690_269015


namespace NUMINAMATH_CALUDE_sum_difference_theorem_l2690_269038

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def jo_sum (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def lisa_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => round_to_nearest_five (i + 1))

theorem sum_difference_theorem :
  jo_sum 60 - lisa_sum 60 = 240 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_theorem_l2690_269038


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2690_269096

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2690_269096


namespace NUMINAMATH_CALUDE_pool_tiles_l2690_269043

theorem pool_tiles (blue_tiles : ℕ) (red_tiles : ℕ) (additional_tiles : ℕ) : 
  blue_tiles = 48 → red_tiles = 32 → additional_tiles = 20 →
  blue_tiles + red_tiles + additional_tiles = 100 := by
  sorry

end NUMINAMATH_CALUDE_pool_tiles_l2690_269043


namespace NUMINAMATH_CALUDE_remaining_money_l2690_269012

def initial_amount : ℕ := 53
def toy_car_cost : ℕ := 11
def toy_car_quantity : ℕ := 2
def scarf_cost : ℕ := 10
def beanie_cost : ℕ := 14

theorem remaining_money :
  initial_amount - (toy_car_cost * toy_car_quantity + scarf_cost + beanie_cost) = 7 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l2690_269012


namespace NUMINAMATH_CALUDE_q_equals_sixteen_l2690_269081

/-- The polynomial with four distinct real roots in geometric progression -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ := x^4 + p*x^3 + q*x^2 - 18*x + 16

/-- The roots of the polynomial form a geometric progression -/
def roots_in_geometric_progression (p q : ℝ) : Prop :=
  ∃ (a r : ℝ), a ≠ 0 ∧ r ≠ 1 ∧
    (polynomial p q a = 0) ∧
    (polynomial p q (a*r) = 0) ∧
    (polynomial p q (a*r^2) = 0) ∧
    (polynomial p q (a*r^3) = 0)

/-- The theorem stating that q equals 16 for the given conditions -/
theorem q_equals_sixteen (p q : ℝ) :
  roots_in_geometric_progression p q → q = 16 := by
  sorry

end NUMINAMATH_CALUDE_q_equals_sixteen_l2690_269081


namespace NUMINAMATH_CALUDE_fruit_card_probability_l2690_269023

theorem fruit_card_probability (total_cards : ℕ) (fruit_cards : ℕ) 
  (h1 : total_cards = 6)
  (h2 : fruit_cards = 2) :
  (fruit_cards : ℚ) / total_cards = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_card_probability_l2690_269023


namespace NUMINAMATH_CALUDE_sqrt_inequality_not_arithmetic_sequence_l2690_269053

-- Statement 1
theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by sorry

-- Statement 2
theorem not_arithmetic_sequence : 
  ¬ ∃ (d k : ℝ), (k = 1 ∧ k + d = Real.sqrt 2 ∧ k + 2*d = 3) := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_not_arithmetic_sequence_l2690_269053


namespace NUMINAMATH_CALUDE_midpoint_intersection_l2690_269074

/-- Given a line segment from (1,3) to (5,11), if the line x + y = b
    intersects this segment at its midpoint, then b = 10. -/
theorem midpoint_intersection (b : ℝ) : 
  (∃ (x y : ℝ), x + y = b ∧ 
   x = (1 + 5) / 2 ∧ 
   y = (3 + 11) / 2) → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_midpoint_intersection_l2690_269074


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_l2690_269032

/-- The line x + y - 2 = 0 -/
def line (x y : ℝ) : Prop := x + y - 2 = 0

/-- The curve x^2 + y^2 - 12x - 12y + 54 = 0 -/
def curve (x y : ℝ) : Prop := x^2 + y^2 - 12*x - 12*y + 54 = 0

/-- The circle with center (6, 6) and radius 3√2 -/
def small_circle (x y : ℝ) : Prop := (x - 6)^2 + (y - 6)^2 = (3 * Real.sqrt 2)^2

/-- A circle is tangent to the line and curve if it touches them at exactly one point each -/
def is_tangent (circle line curve : ℝ → ℝ → Prop) : Prop := sorry

/-- A circle has the smallest radius if no other circle with a smaller radius is tangent to both the line and curve -/
def has_smallest_radius (circle line curve : ℝ → ℝ → Prop) : Prop := sorry

theorem smallest_tangent_circle :
  is_tangent small_circle line curve ∧ has_smallest_radius small_circle line curve := by sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_l2690_269032


namespace NUMINAMATH_CALUDE_expenditure_ratio_l2690_269072

/-- Represents a person with income and expenditure -/
structure Person where
  income : ℕ
  expenditure : ℕ

/-- The problem setup -/
def problem_setup : Prop := ∃ (p1 p2 : Person),
  -- The ratio of incomes is 5:4
  p1.income * 4 = p2.income * 5 ∧
  -- Each person saves 2200
  p1.income - p1.expenditure = 2200 ∧
  p2.income - p2.expenditure = 2200 ∧
  -- P1's income is 5500
  p1.income = 5500

/-- The theorem to prove -/
theorem expenditure_ratio (h : problem_setup) :
  ∃ (p1 p2 : Person), p1.expenditure * 2 = p2.expenditure * 3 :=
sorry

end NUMINAMATH_CALUDE_expenditure_ratio_l2690_269072


namespace NUMINAMATH_CALUDE_janet_stuffies_l2690_269099

theorem janet_stuffies (x : ℚ) : 
  let total := x
  let kept := (3 / 7) * total
  let distributed := total - kept
  let ratio_sum := 3 + 4 + 2 + 1 + 5
  let janet_part := 1
  (janet_part / ratio_sum) * distributed = (4 * x) / 105 := by
sorry

end NUMINAMATH_CALUDE_janet_stuffies_l2690_269099


namespace NUMINAMATH_CALUDE_square_perimeter_increase_l2690_269060

theorem square_perimeter_increase (x : ℝ) (h : x > 0) :
  let original_side := x / 4
  let new_perimeter := 4 * x
  let new_side := new_perimeter / 4
  new_side / original_side = 4 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_increase_l2690_269060


namespace NUMINAMATH_CALUDE_whistlers_count_l2690_269061

/-- The number of whistlers in each of Koby's boxes -/
def whistlers_per_box : ℕ := sorry

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

theorem whistlers_count : whistlers_per_box = 5 := by
  have h1 : total_fireworks = koby_boxes * (koby_sparklers_per_box + whistlers_per_box) + cherie_sparklers + cherie_whistlers := by sorry
  sorry

end NUMINAMATH_CALUDE_whistlers_count_l2690_269061


namespace NUMINAMATH_CALUDE_kristin_reading_time_l2690_269058

/-- Given that Peter reads one book in 18 hours and can read three times as fast as Kristin,
    prove that Kristin will take 540 hours to read half of her 20 books. -/
theorem kristin_reading_time 
  (peter_time : ℝ) 
  (peter_speed : ℝ) 
  (kristin_books : ℕ) 
  (h1 : peter_time = 18) 
  (h2 : peter_speed = 3) 
  (h3 : kristin_books = 20) : 
  (kristin_books / 2 : ℝ) * (peter_time * peter_speed) = 540 := by
  sorry

end NUMINAMATH_CALUDE_kristin_reading_time_l2690_269058


namespace NUMINAMATH_CALUDE_max_sum_of_digits_12hour_clock_l2690_269013

/-- Represents a time in 12-hour format -/
structure Time12Hour where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours ≥ 1 ∧ hours ≤ 12
  minutes_valid : minutes ≤ 59
  seconds_valid : seconds ≤ 59

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given Time12Hour -/
def sumOfTimeDigits (t : Time12Hour) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The maximum sum of digits on a 12-hour format digital clock -/
theorem max_sum_of_digits_12hour_clock :
  ∃ (t : Time12Hour), ∀ (t' : Time12Hour), sumOfTimeDigits t ≥ sumOfTimeDigits t' ∧ sumOfTimeDigits t = 37 :=
sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_12hour_clock_l2690_269013


namespace NUMINAMATH_CALUDE_point_B_value_l2690_269027

def point_A : ℝ := -1

def distance_AB : ℝ := 4

theorem point_B_value : 
  ∃ (B : ℝ), (B = 3 ∨ B = -5) ∧ |B - point_A| = distance_AB :=
by sorry

end NUMINAMATH_CALUDE_point_B_value_l2690_269027


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l2690_269094

/-- Given a line l₁: 3x - 6y = 9 and a point P(-2, 4), 
    prove that the line l₂: y = -2x is perpendicular to l₁ and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) : 
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 3 * x - 6 * y = 9
  let l₂ : ℝ → ℝ := λ x ↦ -2 * x
  let P : ℝ × ℝ := (-2, 4)
  (∀ x y, l₁ x y ↔ y = 1/2 * x - 3/2) ∧  -- l₁ in slope-intercept form
  (l₂ P.1 = P.2) ∧                      -- l₂ passes through P
  ((-2) * (1/2) = -1)                   -- l₁ and l₂ are perpendicular
  :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l2690_269094


namespace NUMINAMATH_CALUDE_three_digit_powers_of_two_l2690_269031

theorem three_digit_powers_of_two (n : ℕ) : 
  (∃ m : ℕ, 100 ≤ 2^m ∧ 2^m ≤ 999) ↔ (n = 7 ∨ n = 8 ∨ n = 9) :=
by sorry

end NUMINAMATH_CALUDE_three_digit_powers_of_two_l2690_269031


namespace NUMINAMATH_CALUDE_wage_cut_and_raise_l2690_269089

theorem wage_cut_and_raise (original_wage : ℝ) (h : original_wage > 0) :
  let reduced_wage := 0.8 * original_wage
  let raised_wage := reduced_wage * 1.25
  raised_wage = original_wage :=
by sorry

end NUMINAMATH_CALUDE_wage_cut_and_raise_l2690_269089


namespace NUMINAMATH_CALUDE_coffee_cups_total_l2690_269055

theorem coffee_cups_total (sandra_cups marcie_cups : ℕ) 
  (h1 : sandra_cups = 6) 
  (h2 : marcie_cups = 2) : 
  sandra_cups + marcie_cups = 8 := by
sorry

end NUMINAMATH_CALUDE_coffee_cups_total_l2690_269055


namespace NUMINAMATH_CALUDE_difference_of_squares_of_odd_numbers_divisible_by_eight_l2690_269021

theorem difference_of_squares_of_odd_numbers_divisible_by_eight (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 2 * k + 1) (hb : ∃ k : ℤ, b = 2 * k + 1) : 
  ∃ m : ℤ, a^2 - b^2 = 8 * m :=
sorry

end NUMINAMATH_CALUDE_difference_of_squares_of_odd_numbers_divisible_by_eight_l2690_269021


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2690_269045

theorem quadratic_equation_solution :
  ∃ x : ℝ, 3 * x^2 - 6 * x + 3 = 0 ∧ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2690_269045


namespace NUMINAMATH_CALUDE_percentage_of_men_in_company_l2690_269083

/-- The percentage of men in a company, given attendance rates at a company picnic -/
theorem percentage_of_men_in_company : 
  ∀ (M : ℝ), 
  (M ≥ 0) →  -- M is non-negative
  (M ≤ 1) →  -- M is at most 1
  (0.20 * M + 0.40 * (1 - M) = 0.33) →  -- Picnic attendance equation
  (M = 0.35) :=  -- Conclusion: 35% of employees are men
by sorry

end NUMINAMATH_CALUDE_percentage_of_men_in_company_l2690_269083


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2690_269069

/-- Given a group of 8 persons, if replacing one person with a new person weighing 94 kg
    increases the average weight by 3 kg, then the weight of the replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (original_count : ℕ)
  (weight_increase : ℝ)
  (new_person_weight : ℝ)
  (h1 : original_count = 8)
  (h2 : weight_increase = 3)
  (h3 : new_person_weight = 94)
  : ℝ :=
by
  sorry

#check weight_of_replaced_person

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2690_269069


namespace NUMINAMATH_CALUDE_yellow_face_probability_l2690_269093

-- Define the die
def die_sides : ℕ := 8
def yellow_faces : ℕ := 3

-- Define the probability function
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Theorem statement
theorem yellow_face_probability : 
  probability yellow_faces die_sides = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l2690_269093


namespace NUMINAMATH_CALUDE_digits_of_product_l2690_269022

theorem digits_of_product : ∃ n : ℕ, n = 3^4 * 6^8 ∧ (Nat.log 10 n).succ = 9 := by sorry

end NUMINAMATH_CALUDE_digits_of_product_l2690_269022


namespace NUMINAMATH_CALUDE_angle_C_indeterminate_l2690_269098

/-- Represents a quadrilateral with angles A, B, C, and D -/
structure Quadrilateral where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ
  angleD : ℝ
  sum_360 : angleA + angleB + angleC + angleD = 360

/-- Theorem stating that ∠C cannot be determined in a quadrilateral ABCD 
    where ∠A = 80° and ∠B = 100° without information about ∠D -/
theorem angle_C_indeterminate (q : Quadrilateral) 
    (hA : q.angleA = 80) (hB : q.angleB = 100) :
  ∀ (x : ℝ), 0 < x ∧ x < 180 → 
  ∃ (q' : Quadrilateral), q'.angleA = q.angleA ∧ q'.angleB = q.angleB ∧ q'.angleC = x :=
sorry

end NUMINAMATH_CALUDE_angle_C_indeterminate_l2690_269098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2690_269046

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a_n where a_1 + 3a_8 + a_15 = 120, 
    prove that 2a_6 - a_4 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
    (h_arith : ArithmeticSequence a) 
    (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
    2 * a 6 - a 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2690_269046


namespace NUMINAMATH_CALUDE_otimes_inequality_range_l2690_269008

/-- Custom operation ⊗ on real numbers -/
def otimes (x y : ℝ) : ℝ := x * (1 - y)

/-- Theorem stating the range of 'a' for which the inequality holds for all real x -/
theorem otimes_inequality_range (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + a) < 1) ↔ a ∈ Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_otimes_inequality_range_l2690_269008


namespace NUMINAMATH_CALUDE_prob_product_one_four_dice_l2690_269041

/-- The number of sides on a standard die -/
def dieSides : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def probSingleDie : ℚ := 1 / dieSides

/-- The number of dice rolled -/
def numDice : ℕ := 4

/-- The probability of rolling all ones on multiple dice -/
def probAllOnes : ℚ := probSingleDie ^ numDice

theorem prob_product_one_four_dice :
  probAllOnes = 1 / 1296 := by sorry

end NUMINAMATH_CALUDE_prob_product_one_four_dice_l2690_269041


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l2690_269077

theorem largest_n_with_unique_k : ∃ (n : ℕ), n > 0 ∧ n ≤ 136 ∧
  (∃! (k : ℤ), (9 : ℚ)/17 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/15) ∧
  (∀ (m : ℕ), m > n → ¬(∃! (k : ℤ), (9 : ℚ)/17 < (m : ℚ)/(m + k) ∧ (m : ℚ)/(m + k) < 8/15)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l2690_269077


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2690_269002

theorem arithmetic_sequence_first_term
  (a d : ℚ)  -- First term and common difference
  (sum_60 : ℚ → ℚ → ℕ → ℚ)  -- Function to calculate sum of n terms
  (h1 : sum_60 a d 60 = 240)  -- Sum of first 60 terms
  (h2 : sum_60 (a + 60 * d) d 60 = 3240)  -- Sum of next 60 terms
  : a = -247 / 12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l2690_269002


namespace NUMINAMATH_CALUDE_optimal_betting_strategy_l2690_269078

def num_boxes : ℕ := 100

-- The maximum factor for exactly one blue cube
def max_factor_one_blue (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / n

-- The maximum factor for at least two blue cubes
def max_factor_two_plus_blue (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / ((2 ^ n : ℚ) - (n + 1 : ℚ))

theorem optimal_betting_strategy :
  (max_factor_one_blue num_boxes = (2 ^ 98 : ℚ) / 25) ∧
  (max_factor_two_plus_blue num_boxes = (2 ^ 100 : ℚ) / ((2 ^ 100 : ℚ) - 101)) :=
by sorry

end NUMINAMATH_CALUDE_optimal_betting_strategy_l2690_269078


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l2690_269054

def batting_average (innings : ℕ) (total_runs : ℕ) : ℚ :=
  total_runs / innings

def revised_average (innings : ℕ) (total_runs : ℕ) (not_out : ℕ) : ℚ :=
  total_runs / (innings - not_out)

theorem batsman_average_theorem (total_runs_11 : ℕ) (innings : ℕ) (last_score : ℕ) (not_out : ℕ) :
  innings = 12 →
  last_score = 92 →
  not_out = 3 →
  batting_average innings (total_runs_11 + last_score) - batting_average (innings - 1) total_runs_11 = 2 →
  batting_average innings (total_runs_11 + last_score) = 70 ∧
  revised_average innings (total_runs_11 + last_score) not_out = 93.33 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_theorem_l2690_269054


namespace NUMINAMATH_CALUDE_no_solution_to_inequalities_l2690_269047

theorem no_solution_to_inequalities :
  ¬∃ x : ℝ, (3 * x - 2 < (x + 2)^2) ∧ ((x + 2)^2 < 8 * x - 5) := by
sorry

end NUMINAMATH_CALUDE_no_solution_to_inequalities_l2690_269047


namespace NUMINAMATH_CALUDE_sandy_spending_percentage_l2690_269035

def initial_amount : ℝ := 300
def remaining_amount : ℝ := 210

theorem sandy_spending_percentage :
  (initial_amount - remaining_amount) / initial_amount * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sandy_spending_percentage_l2690_269035


namespace NUMINAMATH_CALUDE_horner_v3_value_hex_210_to_decimal_l2690_269000

-- Define the polynomial and Horner's method
def f (x : ℤ) : ℤ := 3*x^6 + 5*x^5 + 6*x^4 + 79*x^3 - 8*x^2 + 35*x + 12

def horner_step (v : ℤ) (x : ℤ) (a : ℤ) : ℤ := v * x + a

def horner_v3 (x : ℤ) : ℤ :=
  let v0 := 3
  let v1 := horner_step v0 x 5
  let v2 := horner_step v1 x 6
  horner_step v2 x 79

-- Theorem for the first part of the problem
theorem horner_v3_value : horner_v3 (-4) = -57 := by sorry

-- Define hexadecimal to decimal conversion
def hex_to_decimal (d2 d1 d0 : ℕ) : ℕ := d2 * 6^2 + d1 * 6^1 + d0 * 6^0

-- Theorem for the second part of the problem
theorem hex_210_to_decimal : hex_to_decimal 2 1 0 = 78 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_hex_210_to_decimal_l2690_269000


namespace NUMINAMATH_CALUDE_expression_is_negative_l2690_269092

theorem expression_is_negative : 
  Real.sqrt (25 * Real.sqrt 7 - 27 * Real.sqrt 6) - Real.sqrt (17 * Real.sqrt 5 - 38) < 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_is_negative_l2690_269092


namespace NUMINAMATH_CALUDE_circle_circumference_l2690_269086

/-- Given two circles with equal areas, where half the radius of one circle is 4.5,
    prove that the circumference of the other circle is 18π. -/
theorem circle_circumference (x y : ℝ) (harea : π * x^2 = π * y^2) (hy : y / 2 = 4.5) :
  2 * π * x = 18 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_l2690_269086


namespace NUMINAMATH_CALUDE_negative_300_equals_60_l2690_269064

/-- Two angles have the same terminal side if they differ by a multiple of 360° -/
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + 360 * k

/-- Prove that -300° and 60° have the same terminal side -/
theorem negative_300_equals_60 : same_terminal_side (-300) 60 := by
  sorry

end NUMINAMATH_CALUDE_negative_300_equals_60_l2690_269064


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2690_269071

/-- The sum of the repeating decimals 0.3̄ and 0.6̄ is equal to 1. -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), (∀ n : ℕ, x * 10^n - ⌊x * 10^n⌋ = 0.3) ∧ 
                (∀ n : ℕ, y * 10^n - ⌊y * 10^n⌋ = 0.6) ∧ 
                x + y = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2690_269071


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2690_269010

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2690_269010


namespace NUMINAMATH_CALUDE_congruence_solution_extension_l2690_269063

theorem congruence_solution_extension 
  (p : ℕ) (n a : ℕ) (h_prime : Nat.Prime p) 
  (h_n : ¬ p ∣ n) (h_a : ¬ p ∣ a) 
  (h_base : ∃ x : ℕ, x^n ≡ a [MOD p]) :
  ∀ r : ℕ, ∃ y : ℕ, y^n ≡ a [MOD p^r] :=
by sorry

end NUMINAMATH_CALUDE_congruence_solution_extension_l2690_269063


namespace NUMINAMATH_CALUDE_min_value_theorem_l2690_269097

/-- The function f(x) defined as |x+a| + |x+3a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x + 3*a|

/-- The theorem stating the minimum value of 1/m^2 + n^2 given conditions -/
theorem min_value_theorem (a m n : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), f a y ≥ f a x) →  -- minimum of f(x) exists
  (∀ (x : ℝ), f a x ≥ 2) →  -- minimum value of f(x) is 2
  (a - m) * (a + m) = 4 / n^2 →  -- given condition
  (∃ (k : ℝ), ∀ (p q : ℝ), 1 / p^2 + q^2 ≥ k ∧ (1 / m^2 + n^2 = k)) →  -- minimum of 1/m^2 + n^2 exists
  (1 / m^2 + n^2 = 9)  -- conclusion: minimum value is 9
:= by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2690_269097


namespace NUMINAMATH_CALUDE_beverly_bottle_caps_l2690_269024

/-- The number of bottle caps in Beverly's collection -/
def total_bottle_caps (small_box_caps : ℕ) (large_box_caps : ℕ) 
                      (small_boxes : ℕ) (large_boxes : ℕ) 
                      (individual_caps : ℕ) : ℕ :=
  small_box_caps * small_boxes + large_box_caps * large_boxes + individual_caps

/-- Theorem stating the total number of bottle caps in Beverly's collection -/
theorem beverly_bottle_caps : 
  total_bottle_caps 35 75 7 3 23 = 493 := by
  sorry

end NUMINAMATH_CALUDE_beverly_bottle_caps_l2690_269024
