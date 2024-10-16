import Mathlib

namespace NUMINAMATH_CALUDE_line_system_properties_l2392_239273

-- Define the line system M
def line_system (θ : ℝ) (x y : ℝ) : Prop :=
  x * Real.cos θ + y * Real.sin θ = 1

-- Define the region enclosed by the lines
def enclosed_region (p : ℝ × ℝ) : Prop :=
  ∃ θ, line_system θ p.1 p.2

-- Theorem statement
theorem line_system_properties :
  -- 1. The area of the region enclosed by the lines is π
  (∃ A : Set (ℝ × ℝ), (∀ p, p ∈ A ↔ enclosed_region p) ∧ MeasureTheory.volume A = π) ∧
  -- 2. Not all lines in the system are parallel
  (∃ θ₁ θ₂, θ₁ ≠ θ₂ ∧ ¬ (∀ x y, line_system θ₁ x y ↔ line_system θ₂ x y)) ∧
  -- 3. Not all lines in the system pass through a fixed point
  (¬ ∃ p : ℝ × ℝ, ∀ θ, line_system θ p.1 p.2) ∧
  -- 4. For any integer n ≥ 3, there exists a regular n-gon with edges on the lines of the system
  (∀ n : ℕ, n ≥ 3 → ∃ vertices : Fin n → ℝ × ℝ,
    (∀ i : Fin n, ∃ θ, line_system θ (vertices i).1 (vertices i).2) ∧
    (∀ i j : Fin n, (vertices i).1^2 + (vertices i).2^2 = (vertices j).1^2 + (vertices j).2^2) ∧
    (∀ i j : Fin n, i ≠ j → (vertices i).1 ≠ (vertices j).1 ∨ (vertices i).2 ≠ (vertices j).2)) :=
by sorry


end NUMINAMATH_CALUDE_line_system_properties_l2392_239273


namespace NUMINAMATH_CALUDE_circle_bisection_l2392_239203

-- Define the two circles
def circle1 (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = b^2 + 1
def circle2 (x y : ℝ) : Prop := (x + 1)^2 + (y + 1)^2 = 4

-- Define the bisection condition
def bisects (a b : ℝ) : Prop := 
  ∀ x y : ℝ, circle1 a b x y → circle2 x y → 
    ∃ x' y' : ℝ, x' ≠ x ∧ y' ≠ y ∧ circle1 a b x' y' ∧ circle2 x' y'

-- State the theorem
theorem circle_bisection (a b : ℝ) :
  bisects a b → a^2 + 2*a + 2*b + 5 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_bisection_l2392_239203


namespace NUMINAMATH_CALUDE_unique_lcm_triple_l2392_239227

theorem unique_lcm_triple : ∃! (x y z : ℕ+), 
  (Nat.lcm x.val y.val = 108) ∧ 
  (Nat.lcm x.val z.val = 400) ∧ 
  (Nat.lcm y.val z.val = 450) := by
  sorry

end NUMINAMATH_CALUDE_unique_lcm_triple_l2392_239227


namespace NUMINAMATH_CALUDE_ear_muffs_before_december_count_l2392_239298

/-- The number of ear muffs bought before December -/
def ear_muffs_before_december (total : ℕ) (during_december : ℕ) : ℕ :=
  total - during_december

/-- Theorem stating that the number of ear muffs bought before December is 1346 -/
theorem ear_muffs_before_december_count :
  ear_muffs_before_december 7790 6444 = 1346 := by
  sorry

end NUMINAMATH_CALUDE_ear_muffs_before_december_count_l2392_239298


namespace NUMINAMATH_CALUDE_complex_product_equals_43_l2392_239242

theorem complex_product_equals_43 (x : ℂ) (h : x = Complex.exp (2 * Real.pi * Complex.I / 7)) :
  (2*x + x^2) * (2*x^2 + x^4) * (2*x^3 + x^6) * (2*x^4 + x^8) * (2*x^5 + x^10) * (2*x^6 + x^12) = 43 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_43_l2392_239242


namespace NUMINAMATH_CALUDE_discriminant_neither_sufficient_nor_necessary_l2392_239284

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f(x) = ax^2 + bx + c is always above the x-axis -/
def always_above_x_axis (a b c : ℝ) : Prop :=
  ∀ x, quadratic_function a b c x > 0

/-- The discriminant condition b^2 - 4ac < 0 -/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

/-- Theorem stating that the discriminant condition is neither sufficient nor necessary 
    for the quadratic function to always be above the x-axis -/
theorem discriminant_neither_sufficient_nor_necessary :
  ¬(∀ a b c : ℝ, discriminant_condition a b c → always_above_x_axis a b c) ∧
  ¬(∀ a b c : ℝ, always_above_x_axis a b c → discriminant_condition a b c) :=
sorry

end NUMINAMATH_CALUDE_discriminant_neither_sufficient_nor_necessary_l2392_239284


namespace NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l2392_239241

theorem smallest_nonfactor_product_of_48 :
  ∃ (u v : ℕ), 
    u ≠ v ∧ 
    u > 0 ∧ 
    v > 0 ∧ 
    48 % u = 0 ∧ 
    48 % v = 0 ∧ 
    48 % (u * v) ≠ 0 ∧
    u * v = 18 ∧
    (∀ (x y : ℕ), x ≠ y → x > 0 → y > 0 → 48 % x = 0 → 48 % y = 0 → 48 % (x * y) ≠ 0 → x * y ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_nonfactor_product_of_48_l2392_239241


namespace NUMINAMATH_CALUDE_cube_cut_surface_area_l2392_239261

/-- Represents a piece of the cube -/
structure Piece where
  height : ℝ

/-- Represents the solid formed by rearranging the cube pieces -/
structure Solid where
  pieces : List Piece

/-- Calculates the surface area of the solid -/
def surfaceArea (s : Solid) : ℝ :=
  sorry

theorem cube_cut_surface_area :
  let cube_volume : ℝ := 1
  let cut1 : ℝ := 1/2
  let cut2 : ℝ := 1/3
  let cut3 : ℝ := 1/17
  let piece_A : Piece := ⟨cut1⟩
  let piece_B : Piece := ⟨cut2⟩
  let piece_C : Piece := ⟨cut3⟩
  let piece_D : Piece := ⟨1 - (cut1 + cut2 + cut3)⟩
  let solid : Solid := ⟨[piece_A, piece_B, piece_C, piece_D]⟩
  surfaceArea solid = 11 :=
sorry

end NUMINAMATH_CALUDE_cube_cut_surface_area_l2392_239261


namespace NUMINAMATH_CALUDE_equation_solution_l2392_239232

theorem equation_solution : ∃! x : ℝ, (1 / (x + 11) + 1 / (x + 5) = 1 / (x + 12) + 1 / (x + 4)) ∧ x = -8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2392_239232


namespace NUMINAMATH_CALUDE_expression_evaluation_l2392_239269

theorem expression_evaluation (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) :
  ((2*a + b)^2 - (2*a + b)*(2*a - b)) / (-1/2 * b) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2392_239269


namespace NUMINAMATH_CALUDE_dropped_class_hours_l2392_239262

/-- Calculates the remaining class hours after dropping a class -/
def remaining_class_hours (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

/-- Theorem: Given 4 classes of 2 hours each, dropping 1 class results in 6 hours of classes -/
theorem dropped_class_hours : remaining_class_hours 4 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_dropped_class_hours_l2392_239262


namespace NUMINAMATH_CALUDE_bobs_final_salary_l2392_239202

/-- Calculates the final salary after two raises and a pay cut -/
def final_salary (initial_salary : ℝ) (first_raise : ℝ) (second_raise : ℝ) (pay_cut : ℝ) : ℝ :=
  let salary_after_first_raise := initial_salary * (1 + first_raise)
  let salary_after_second_raise := salary_after_first_raise * (1 + second_raise)
  salary_after_second_raise * (1 - pay_cut)

/-- Theorem stating that Bob's final salary is $2541 -/
theorem bobs_final_salary :
  final_salary 3000 0.1 0.1 0.3 = 2541 := by
  sorry

end NUMINAMATH_CALUDE_bobs_final_salary_l2392_239202


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l2392_239283

/-- A parabola with directrix x = 1 has the standard equation y² = -4x -/
theorem parabola_standard_equation (x y : ℝ) :
  (∃ (p : ℝ), p / 2 = 1 ∧ y^2 = -2 * p * x) → y^2 = -4 * x := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l2392_239283


namespace NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l2392_239258

/-- Represents the miles run by Bill and Julia on Saturday and Sunday -/
structure WeekendRun where
  billSat : ℕ
  billSun : ℕ
  juliaSat : ℕ
  juliaSun : ℕ

/-- The conditions of the problem -/
def weekend_run_conditions (run : WeekendRun) : Prop :=
  run.billSun > run.billSat ∧
  run.juliaSat = 0 ∧
  run.juliaSun = 2 * run.billSun ∧
  run.billSat + run.billSun + run.juliaSat + run.juliaSun = 28 ∧
  run.billSun = 8

/-- The theorem to prove -/
theorem bill_sunday_saturday_difference (run : WeekendRun) 
  (h : weekend_run_conditions run) : 
  run.billSun - run.billSat = 4 := by
sorry

end NUMINAMATH_CALUDE_bill_sunday_saturday_difference_l2392_239258


namespace NUMINAMATH_CALUDE_max_revenue_at_18_75_l2392_239247

/-- The revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 4 * p)

/-- The theorem stating that 18.75 maximizes the revenue function --/
theorem max_revenue_at_18_75 :
  ∀ p : ℝ, p ≤ 30 → R p ≤ R 18.75 := by
  sorry

#check max_revenue_at_18_75

end NUMINAMATH_CALUDE_max_revenue_at_18_75_l2392_239247


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_l2392_239257

/-- Given a hyperbola C with equation x²/m - y² = 1 (m > 0) and asymptote √3x + my = 0,
    prove that the focal length of C is 4. -/
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / m - y^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | Real.sqrt 3 * x + m * y = 0}
  ∃ (a b c : ℝ), a^2 = m ∧ b^2 = m ∧ c^2 = a^2 + b^2 ∧ 2 * c = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_l2392_239257


namespace NUMINAMATH_CALUDE_max_profit_theorem_l2392_239238

/-- Represents the daily production and profit of an eco-friendly bag factory --/
structure BagFactory where
  totalBags : ℕ
  costA : ℚ
  sellA : ℚ
  costB : ℚ
  sellB : ℚ
  maxInvestment : ℚ

/-- Calculates the profit function for the bag factory --/
def profitFunction (factory : BagFactory) (x : ℚ) : ℚ :=
  (factory.sellA - factory.costA) * x + (factory.sellB - factory.costB) * (factory.totalBags - x)

/-- Theorem stating the maximum profit of the bag factory --/
theorem max_profit_theorem (factory : BagFactory) 
    (h1 : factory.totalBags = 4500)
    (h2 : factory.costA = 2)
    (h3 : factory.sellA = 2.3)
    (h4 : factory.costB = 3)
    (h5 : factory.sellB = 3.5)
    (h6 : factory.maxInvestment = 10000) :
    ∃ x : ℚ, x ≥ 0 ∧ x ≤ factory.totalBags ∧
    factory.costA * x + factory.costB * (factory.totalBags - x) ≤ factory.maxInvestment ∧
    ∀ y : ℚ, y ≥ 0 → y ≤ factory.totalBags →
    factory.costA * y + factory.costB * (factory.totalBags - y) ≤ factory.maxInvestment →
    profitFunction factory x ≥ profitFunction factory y ∧
    profitFunction factory x = 1550 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_theorem_l2392_239238


namespace NUMINAMATH_CALUDE_snake_count_l2392_239221

theorem snake_count (breeding_balls : Nat) (snake_pairs : Nat) (total_snakes : Nat) :
  breeding_balls = 3 →
  snake_pairs = 6 →
  total_snakes = 36 →
  ∃ snakes_per_ball : Nat, snakes_per_ball * breeding_balls + snake_pairs * 2 = total_snakes ∧ snakes_per_ball = 8 := by
  sorry

end NUMINAMATH_CALUDE_snake_count_l2392_239221


namespace NUMINAMATH_CALUDE_sqrt_sum_over_sqrt_l2392_239268

theorem sqrt_sum_over_sqrt (a b c : ℝ) (ha : a = 112) (hb : b = 567) (hc : c = 175) :
  (Real.sqrt a + Real.sqrt b) / Real.sqrt c = 13 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_over_sqrt_l2392_239268


namespace NUMINAMATH_CALUDE_crackers_sales_total_l2392_239255

theorem crackers_sales_total (friday_sales : ℕ) 
  (h1 : friday_sales = 30) 
  (h2 : ∃ saturday_sales : ℕ, saturday_sales = 2 * friday_sales) 
  (h3 : ∃ sunday_sales : ℕ, sunday_sales = saturday_sales - 15) : 
  friday_sales + 2 * friday_sales + (2 * friday_sales - 15) = 135 := by
  sorry

end NUMINAMATH_CALUDE_crackers_sales_total_l2392_239255


namespace NUMINAMATH_CALUDE_magazines_per_box_l2392_239279

theorem magazines_per_box (total_magazines : ℕ) (num_boxes : ℕ) (magazines_per_box : ℕ) : 
  total_magazines = 63 → num_boxes = 7 → total_magazines = num_boxes * magazines_per_box → magazines_per_box = 9 := by
  sorry

end NUMINAMATH_CALUDE_magazines_per_box_l2392_239279


namespace NUMINAMATH_CALUDE_subset_implies_a_value_l2392_239211

def A (a : ℝ) : Set ℝ := {1, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2 - a}

theorem subset_implies_a_value (a : ℝ) (h : B a ⊆ A a) : a = -1 ∨ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_value_l2392_239211


namespace NUMINAMATH_CALUDE_tangencyTriangleAreaTheorem_l2392_239294

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents a triangle formed by the points of tangency of three circles -/
structure TangencyTriangle where
  c1 : Circle
  c2 : Circle
  c3 : Circle

/-- The area of the triangle formed by the points of tangency of three mutually externally tangent circles -/
def tangencyTriangleArea (t : TangencyTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the triangle formed by the points of tangency
    of three mutually externally tangent circles with radii 1, 3, and 5 is 5/3 -/
theorem tangencyTriangleAreaTheorem :
  let c1 : Circle := { radius := 1 }
  let c2 : Circle := { radius := 3 }
  let c3 : Circle := { radius := 5 }
  let t : TangencyTriangle := { c1 := c1, c2 := c2, c3 := c3 }
  tangencyTriangleArea t = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_tangencyTriangleAreaTheorem_l2392_239294


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2392_239240

structure Quadrilateral :=
  (P Q R S : ℝ × ℝ)
  (PQ QR RS SP : ℝ)
  (PR : ℤ)

theorem quadrilateral_diagonal_length 
  (quad : Quadrilateral) 
  (h1 : quad.PQ = 7)
  (h2 : quad.QR = 15)
  (h3 : quad.RS = 7)
  (h4 : quad.SP = 8) :
  9 ≤ quad.PR ∧ quad.PR ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2392_239240


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2392_239210

theorem rationalize_denominator : 
  (35 : ℝ) / Real.sqrt 15 = (7 / 3) * Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2392_239210


namespace NUMINAMATH_CALUDE_root_condition_implies_k_value_l2392_239299

theorem root_condition_implies_k_value (a b c k : ℝ) :
  (∃ x₁ x₂ : ℝ, 
    (x₁^2 - (b+1)*x₁) / ((a+1)*x₁ - c) = (k-2)/(k+2) ∧
    (x₂^2 - (b+1)*x₂) / ((a+1)*x₂ - c) = (k-2)/(k+2) ∧
    x₁ = -x₂ ∧ x₁ ≠ 0) →
  k = (-2*(b-a))/(b+a+2) :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_k_value_l2392_239299


namespace NUMINAMATH_CALUDE_equation_has_one_integral_root_l2392_239224

theorem equation_has_one_integral_root :
  ∃! (x : ℤ), x - 5 / (x - 4 : ℚ) = 2 - 5 / (x - 4 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_has_one_integral_root_l2392_239224


namespace NUMINAMATH_CALUDE_product_is_even_l2392_239239

def pi_digits : Finset ℕ := {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4}

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem product_is_even (a : Fin 24 → ℕ) (h : ∀ i, a i ∈ pi_digits) :
  is_even ((a 0 - a 1) * (a 2 - a 3) * (a 4 - a 5) * (a 6 - a 7) * (a 8 - a 9) * (a 10 - a 11) *
           (a 12 - a 13) * (a 14 - a 15) * (a 16 - a 17) * (a 18 - a 19) * (a 20 - a 21) * (a 22 - a 23)) :=
by sorry

end NUMINAMATH_CALUDE_product_is_even_l2392_239239


namespace NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2392_239245

-- Define the properties
def is_in_second_quadrant (α : Real) : Prop := 90 < α ∧ α ≤ 180
def is_obtuse_angle (α : Real) : Prop := 90 < α ∧ α < 180

-- Theorem statement
theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α, is_obtuse_angle α → is_in_second_quadrant α) ∧
  (∃ α, is_in_second_quadrant α ∧ ¬is_obtuse_angle α) :=
sorry

end NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l2392_239245


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2392_239244

theorem polynomial_simplification (x : ℝ) :
  (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1 = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2392_239244


namespace NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l2392_239253

/-- Calculates the cost per serving of an apple pie --/
theorem apple_pie_cost_per_serving 
  (num_servings : ℕ)
  (apple_pounds : ℝ)
  (apple_cost_per_pound : ℝ)
  (crust_cost : ℝ)
  (lemon_cost : ℝ)
  (butter_cost : ℝ)
  (h1 : num_servings = 8)
  (h2 : apple_pounds = 2)
  (h3 : apple_cost_per_pound = 2)
  (h4 : crust_cost = 2)
  (h5 : lemon_cost = 0.5)
  (h6 : butter_cost = 1.5) :
  (apple_pounds * apple_cost_per_pound + crust_cost + lemon_cost + butter_cost) / num_servings = 1 :=
by sorry

end NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l2392_239253


namespace NUMINAMATH_CALUDE_intersection_to_left_focus_distance_l2392_239209

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the intersection point in the first quadrant
def intersection_point (x y : ℝ) : Prop :=
  ellipse x y ∧ parabola x y ∧ x > 0 ∧ y > 0

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)

-- Theorem statement
theorem intersection_to_left_focus_distance :
  ∀ x y : ℝ, intersection_point x y →
  Real.sqrt ((x - left_focus.1)^2 + (y - left_focus.2)^2) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_to_left_focus_distance_l2392_239209


namespace NUMINAMATH_CALUDE_monotonic_quadratic_l2392_239200

/-- The function f(x) = ax² + 2x - 3 is monotonically increasing on (-∞, 4) iff -1/4 ≤ a ≤ 0 -/
theorem monotonic_quadratic (a : ℝ) :
  (∀ x y : ℝ, x < y → x < 4 → y < 4 → a * x^2 + 2 * x - 3 < a * y^2 + 2 * y - 3) ↔
  -1/4 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_l2392_239200


namespace NUMINAMATH_CALUDE_roses_ian_kept_l2392_239267

/-- The number of roses Ian initially had -/
def initial_roses : ℝ := 25.5

/-- The number of roses Ian gave to his mother -/
def roses_to_mother : ℝ := 7.8

/-- The number of roses Ian gave to his grandmother -/
def roses_to_grandmother : ℝ := 11.2

/-- The number of roses Ian gave to his sister -/
def roses_to_sister : ℝ := 4.3

/-- The theorem states that the number of roses Ian kept is 2.2 -/
theorem roses_ian_kept : 
  initial_roses - (roses_to_mother + roses_to_grandmother + roses_to_sister) = 2.2 := by
  sorry

end NUMINAMATH_CALUDE_roses_ian_kept_l2392_239267


namespace NUMINAMATH_CALUDE_a_16_value_l2392_239285

def sequence_a : ℕ → ℚ
  | 0 => 2
  | (n + 1) => (1 + sequence_a n) / (1 - sequence_a n)

theorem a_16_value : sequence_a 16 = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_a_16_value_l2392_239285


namespace NUMINAMATH_CALUDE_valid_book_pairs_18_4_l2392_239222

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of different pairs of books that can be chosen from a collection of books,
    given the total number of books and the number of books in a series,
    with the restriction that two books from the series cannot be chosen together. -/
def validBookPairs (totalBooks seriesBooks : ℕ) : ℕ :=
  choose totalBooks 2 - choose seriesBooks 2

theorem valid_book_pairs_18_4 :
  validBookPairs 18 4 = 147 := by sorry

end NUMINAMATH_CALUDE_valid_book_pairs_18_4_l2392_239222


namespace NUMINAMATH_CALUDE_even_function_sum_l2392_239230

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  a + b = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_even_function_sum_l2392_239230


namespace NUMINAMATH_CALUDE_circle_condition_l2392_239223

-- Define the equation of the curve
def curve_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 + 4*m*x - 2*y + 5*m = 0

-- Define the condition for m
def m_condition (m : ℝ) : Prop :=
  m < 1/4 ∨ m > 1

-- Theorem statement
theorem circle_condition (m : ℝ) :
  (∃ h k r, ∀ x y, curve_equation x y m ↔ (x - h)^2 + (y - k)^2 = r^2) ↔ m_condition m :=
sorry

end NUMINAMATH_CALUDE_circle_condition_l2392_239223


namespace NUMINAMATH_CALUDE_initial_number_of_girls_l2392_239266

/-- The initial number of girls -/
def n : ℕ := sorry

/-- The initial average weight of the girls -/
def A : ℝ := sorry

/-- The weight of the replaced girl -/
def replaced_weight : ℝ := 40

/-- The weight of the new girl -/
def new_weight : ℝ := 80

/-- The increase in average weight -/
def avg_increase : ℝ := 2

theorem initial_number_of_girls :
  (n : ℝ) * (A + avg_increase) - n * A = new_weight - replaced_weight →
  n = 20 := by sorry

end NUMINAMATH_CALUDE_initial_number_of_girls_l2392_239266


namespace NUMINAMATH_CALUDE_min_value_theorem_l2392_239281

def arithmeticSequence (a : ℕ → ℝ) : Prop := 
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem min_value_theorem (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →
  arithmeticSequence a →
  a 7 = a 6 + 2 * a 5 →
  Real.sqrt (a m * a n) = 4 * a 1 →
  (∃ min : ℝ, min = 3/2 ∧ ∀ p q : ℕ, 1/p + 4/q ≥ min) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2392_239281


namespace NUMINAMATH_CALUDE_mean_equality_problem_l2392_239277

theorem mean_equality_problem (y : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_problem_l2392_239277


namespace NUMINAMATH_CALUDE_part_one_part_two_l2392_239297

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 4 < 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + 4*m*x - 5*m^2 < 0}

-- Part 1: Prove that when B = {x | -5 < x < 1}, m = 1
theorem part_one : 
  (B 1 = {x | -5 < x ∧ x < 1}) → 1 = 1 := by sorry

-- Part 2: Prove that when A ⊆ B, m ≤ -1 or m ≥ 4
theorem part_two (m : ℝ) : 
  A ⊆ B m → m ≤ -1 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l2392_239297


namespace NUMINAMATH_CALUDE_bs_sequence_bounded_iff_f_null_l2392_239248

def is_bs_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n = |a (n + 1) - a (n + 2)|

def is_bounded_sequence (a : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |a n| ≤ M

def f (a : ℕ → ℝ) (n k : ℕ) : ℝ :=
  a n * a k * (a n - a k)

theorem bs_sequence_bounded_iff_f_null (a : ℕ → ℝ) :
  is_bs_sequence a →
  (is_bounded_sequence a ↔ ∀ n k : ℕ, f a n k = 0) :=
sorry

end NUMINAMATH_CALUDE_bs_sequence_bounded_iff_f_null_l2392_239248


namespace NUMINAMATH_CALUDE_field_trip_passengers_l2392_239293

/-- The number of passengers a single bus can transport -/
def passengers_per_bus : ℕ := 48

/-- The number of buses needed for the field trip -/
def buses_needed : ℕ := 26

/-- The total number of passengers (students and teachers) going on the field trip -/
def total_passengers : ℕ := passengers_per_bus * buses_needed

theorem field_trip_passengers :
  total_passengers = 1248 :=
sorry

end NUMINAMATH_CALUDE_field_trip_passengers_l2392_239293


namespace NUMINAMATH_CALUDE_discontinuity_coincidence_l2392_239218

-- Define the functions f, g, and h
variable (f g h : ℝ → ℝ)

-- Define the conditions
variable (hf_diff : Differentiable ℝ f)
variable (hg_mono : Monotone g)
variable (hh_mono : Monotone h)
variable (hf_deriv : ∀ x, deriv f x = f x + g x + h x)

-- State the theorem
theorem discontinuity_coincidence :
  ∀ x : ℝ, ¬(ContinuousAt g x) ↔ ¬(ContinuousAt h x) := by
  sorry

end NUMINAMATH_CALUDE_discontinuity_coincidence_l2392_239218


namespace NUMINAMATH_CALUDE_abc_inequality_l2392_239212

noncomputable def a : ℝ := 2 / Real.log 2
noncomputable def b : ℝ := Real.exp 2 / (4 - Real.log 4)
noncomputable def c : ℝ := 2 * Real.sqrt (Real.exp 1)

theorem abc_inequality : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l2392_239212


namespace NUMINAMATH_CALUDE_shorter_train_length_l2392_239280

-- Define the speeds of the trains in km/hr
def speed1 : ℝ := 60
def speed2 : ℝ := 40

-- Define the length of the longer train in meters
def longerTrainLength : ℝ := 180

-- Define the time taken to cross in seconds
def timeToCross : ℝ := 11.519078473722104

-- Theorem statement
theorem shorter_train_length :
  let relativeSpeed : ℝ := (speed1 + speed2) * (1000 / 3600)
  let totalDistance : ℝ := relativeSpeed * timeToCross
  let shorterTrainLength : ℝ := totalDistance - longerTrainLength
  shorterTrainLength = 140 :=
by sorry

end NUMINAMATH_CALUDE_shorter_train_length_l2392_239280


namespace NUMINAMATH_CALUDE_solutions_cubic_equation_l2392_239287

theorem solutions_cubic_equation :
  {x : ℝ | x^3 - 4*x = 0} = {0, -2, 2} := by sorry

end NUMINAMATH_CALUDE_solutions_cubic_equation_l2392_239287


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2392_239272

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 10 - 5 * Complex.I) :
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2392_239272


namespace NUMINAMATH_CALUDE_dandelion_puffs_count_l2392_239286

/-- The number of dandelion puffs Caleb originally picked -/
def original_puffs : ℕ := 40

/-- The number of puffs given to mom -/
def mom_puffs : ℕ := 3

/-- The number of puffs given to sister -/
def sister_puffs : ℕ := 3

/-- The number of puffs given to grandmother -/
def grandmother_puffs : ℕ := 5

/-- The number of puffs given to dog -/
def dog_puffs : ℕ := 2

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of puffs each friend received -/
def puffs_per_friend : ℕ := 9

theorem dandelion_puffs_count :
  original_puffs = mom_puffs + sister_puffs + grandmother_puffs + dog_puffs + num_friends * puffs_per_friend :=
by sorry

end NUMINAMATH_CALUDE_dandelion_puffs_count_l2392_239286


namespace NUMINAMATH_CALUDE_circle_probability_l2392_239235

def total_figures : ℕ := 10
def triangle_count : ℕ := 4
def circle_count : ℕ := 3
def square_count : ℕ := 3

theorem circle_probability : 
  (circle_count : ℚ) / total_figures = 3 / 10 := by sorry

end NUMINAMATH_CALUDE_circle_probability_l2392_239235


namespace NUMINAMATH_CALUDE_worker_problem_l2392_239291

theorem worker_problem (time_B time_together : ℝ) 
  (h1 : time_B = 10)
  (h2 : time_together = 4.444444444444445)
  (h3 : 1 / time_A + 1 / time_B = 1 / time_together) :
  time_A = 8 :=
sorry

end NUMINAMATH_CALUDE_worker_problem_l2392_239291


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_c_l2392_239292

/-- Given a quadratic equation x^2 - 6x + c = 0 with one root being 2,
    prove that the other root is 4 and the value of c is 8. -/
theorem quadratic_equation_roots_and_c (c : ℝ) :
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ x = 2) →
  (∃ y : ℝ, y^2 - 6*y + c = 0 ∧ y = 4 ∧ c = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_c_l2392_239292


namespace NUMINAMATH_CALUDE_circular_permutation_sum_l2392_239295

def CircularPermutation (xs : List ℕ) : Prop :=
  xs.length = 6 ∧ xs.toFinset = {1, 2, 3, 4, 6}

def CircularProduct (xs : List ℕ) : ℕ :=
  (List.zip xs (xs.rotate 1)).map (λ (a, b) => a * b) |>.sum

def MaxCircularProduct : ℕ := sorry

def MaxCircularProductPermutations : ℕ := sorry

theorem circular_permutation_sum :
  MaxCircularProduct + MaxCircularProductPermutations = 96 := by sorry

end NUMINAMATH_CALUDE_circular_permutation_sum_l2392_239295


namespace NUMINAMATH_CALUDE_percentage_of_female_cows_l2392_239205

theorem percentage_of_female_cows (total_cows : ℕ) (pregnant_cows : ℕ) 
  (h1 : total_cows = 44)
  (h2 : pregnant_cows = 11)
  (h3 : pregnant_cows = (female_cows / 2 : ℚ)) :
  (female_cows : ℚ) / total_cows * 100 = 50 :=
by
  sorry

#check percentage_of_female_cows

end NUMINAMATH_CALUDE_percentage_of_female_cows_l2392_239205


namespace NUMINAMATH_CALUDE_regular_polygon_not_unique_by_circumradius_triangle_not_unique_by_circumradius_l2392_239256

/-- A regular polygon -/
structure RegularPolygon where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The radius of the circumscribed circle -/
  circumRadius : ℝ
  /-- Assertion that the number of sides is at least 3 -/
  sidesGe3 : sides ≥ 3

/-- Theorem stating that a regular polygon is not uniquely determined by its circumradius -/
theorem regular_polygon_not_unique_by_circumradius :
  ∃ (p q : RegularPolygon), p.circumRadius = q.circumRadius ∧ p.sides ≠ q.sides :=
sorry

/-- Corollary specifically for triangles -/
theorem triangle_not_unique_by_circumradius :
  ∃ (t : RegularPolygon) (p : RegularPolygon), 
    t.sides = 3 ∧ p.sides ≠ 3 ∧ t.circumRadius = p.circumRadius :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_not_unique_by_circumradius_triangle_not_unique_by_circumradius_l2392_239256


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2392_239236

/-- Given an arithmetic sequence with first term 6, last term 154, and common difference 4,
    prove that the number of terms is 38. -/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ),
    a = 6 →
    d = 4 →
    last = 154 →
    last = a + (n - 1) * d →
    n = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2392_239236


namespace NUMINAMATH_CALUDE_inscribed_cylinder_height_l2392_239270

theorem inscribed_cylinder_height (r_hemisphere r_cylinder : ℝ) (h_hemisphere : r_hemisphere = 7) (h_cylinder : r_cylinder = 3) :
  let h_cylinder := Real.sqrt (r_hemisphere ^ 2 - r_cylinder ^ 2)
  h_cylinder = Real.sqrt 40 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cylinder_height_l2392_239270


namespace NUMINAMATH_CALUDE_tim_dozens_of_golf_balls_l2392_239234

def total_golf_balls : ℕ := 156
def balls_per_dozen : ℕ := 12

theorem tim_dozens_of_golf_balls : 
  total_golf_balls / balls_per_dozen = 13 := by sorry

end NUMINAMATH_CALUDE_tim_dozens_of_golf_balls_l2392_239234


namespace NUMINAMATH_CALUDE_quadratic_integer_root_l2392_239206

/-- The quadratic equation kx^2 - 2(3k - 1)x + 9k - 1 = 0 has at least one integer root
    if and only if k is -3 or -7. -/
theorem quadratic_integer_root (k : ℤ) : 
  (∃ x : ℤ, k * x^2 - 2*(3*k - 1)*x + 9*k - 1 = 0) ↔ (k = -3 ∨ k = -7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_integer_root_l2392_239206


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l2392_239278

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem f_derivative_at_one : 
  (deriv f) 1 = 2 * Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l2392_239278


namespace NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2392_239259

theorem halfway_between_one_eighth_and_one_third :
  (1 / 8 : ℚ) / 2 + (1 / 3 : ℚ) / 2 = 11 / 48 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_one_eighth_and_one_third_l2392_239259


namespace NUMINAMATH_CALUDE_max_plus_min_of_f_l2392_239246

noncomputable def f (x : ℝ) : ℝ := 1 + x / (x^2 + 1)

theorem max_plus_min_of_f : 
  ∃ (M N : ℝ), (∀ x, f x ≤ M) ∧ (∃ x, f x = M) ∧ 
                (∀ x, N ≤ f x) ∧ (∃ x, f x = N) ∧ 
                (M + N = 2) :=
sorry

end NUMINAMATH_CALUDE_max_plus_min_of_f_l2392_239246


namespace NUMINAMATH_CALUDE_total_inflation_time_is_900_l2392_239208

/-- The time in minutes it takes to inflate one soccer ball -/
def inflation_time : ℕ := 20

/-- The number of soccer balls Alexia inflates -/
def alexia_balls : ℕ := 20

/-- The number of additional balls Ermias inflates compared to Alexia -/
def ermias_additional_balls : ℕ := 5

/-- The total number of balls Ermias inflates -/
def ermias_balls : ℕ := alexia_balls + ermias_additional_balls

/-- The total time taken by Alexia and Ermias to inflate all soccer balls -/
def total_inflation_time : ℕ := inflation_time * (alexia_balls + ermias_balls)

theorem total_inflation_time_is_900 : total_inflation_time = 900 := by
  sorry

end NUMINAMATH_CALUDE_total_inflation_time_is_900_l2392_239208


namespace NUMINAMATH_CALUDE_marla_horse_purchase_time_l2392_239217

/-- Represents the exchange rates and Marla's scavenging abilities in the post-apocalyptic wasteland -/
structure WastelandEconomy where
  lizard_to_caps : ℕ
  lizards_to_water : ℕ
  water_to_lizards : ℕ
  horse_to_water : ℕ
  daily_scavenge : ℕ
  nightly_cost : ℕ

/-- Calculates the number of days it takes Marla to collect enough bottle caps to buy a horse -/
def days_to_buy_horse (e : WastelandEconomy) : ℕ :=
  let caps_per_lizard := e.lizard_to_caps
  let water_per_horse := e.horse_to_water
  let lizards_per_horse := (water_per_horse * e.water_to_lizards) / e.lizards_to_water
  let caps_per_horse := lizards_per_horse * caps_per_lizard
  let daily_savings := e.daily_scavenge - e.nightly_cost
  caps_per_horse / daily_savings

/-- Theorem stating that it takes Marla 24 days to collect enough bottle caps to buy a horse -/
theorem marla_horse_purchase_time :
  days_to_buy_horse {
    lizard_to_caps := 8,
    lizards_to_water := 3,
    water_to_lizards := 5,
    horse_to_water := 80,
    daily_scavenge := 20,
    nightly_cost := 4
  } = 24 := by
  sorry

end NUMINAMATH_CALUDE_marla_horse_purchase_time_l2392_239217


namespace NUMINAMATH_CALUDE_complement_A_eq_l2392_239271

/-- The universal set U -/
def U : Set Int := {-2, -1, 1, 3, 5}

/-- The set A -/
def A : Set Int := {-1, 3}

/-- The complement of A with respect to U -/
def complement_A : Set Int := {x | x ∈ U ∧ x ∉ A}

theorem complement_A_eq : complement_A = {-2, 1, 5} := by sorry

end NUMINAMATH_CALUDE_complement_A_eq_l2392_239271


namespace NUMINAMATH_CALUDE_visits_neither_country_l2392_239207

/-- Given a group of people and information about their visits to Iceland and Norway,
    calculate the number of people who have visited neither country. -/
theorem visits_neither_country
  (total : ℕ)
  (visited_iceland : ℕ)
  (visited_norway : ℕ)
  (visited_both : ℕ)
  (h_total : total = 90)
  (h_iceland : visited_iceland = 55)
  (h_norway : visited_norway = 33)
  (h_both : visited_both = 51) :
  total - (visited_iceland + visited_norway - visited_both) = 53 := by
  sorry

#check visits_neither_country

end NUMINAMATH_CALUDE_visits_neither_country_l2392_239207


namespace NUMINAMATH_CALUDE_trailing_zeros_count_l2392_239215

/-- The number of trailing zeros in (10¹² - 25)² is 12 -/
theorem trailing_zeros_count : ∃ n : ℕ, n > 0 ∧ (10^12 - 25)^2 = n * 10^12 ∧ n % 10 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeros_count_l2392_239215


namespace NUMINAMATH_CALUDE_betty_age_l2392_239289

/-- Given the ages of Albert, Mary, and Betty, prove Betty's age -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 8) :
  betty = 4 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l2392_239289


namespace NUMINAMATH_CALUDE_triangle_bisector_inequality_l2392_239260

/-- Given a triangle ABC with side lengths a, b, c, semiperimeter p, circumradius R,
    inradius r, and angle bisector lengths l_a, l_b, l_c, prove that
    l_a * l_b + l_b * l_c + l_c * l_a ≤ p * √(3r² + 12Rr) -/
theorem triangle_bisector_inequality
  (a b c : ℝ)
  (p : ℝ)
  (R r : ℝ)
  (l_a l_b l_c : ℝ)
  (h_p : p = (a + b + c) / 2)
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_R : R > 0)
  (h_r : r > 0)
  (h_l_a : l_a > 0)
  (h_l_b : l_b > 0)
  (h_l_c : l_c > 0) :
  l_a * l_b + l_b * l_c + l_c * l_a ≤ p * Real.sqrt (3 * r^2 + 12 * R * r) :=
by sorry

end NUMINAMATH_CALUDE_triangle_bisector_inequality_l2392_239260


namespace NUMINAMATH_CALUDE_sabrina_video_votes_l2392_239276

theorem sabrina_video_votes (total_votes : ℕ) (upvotes downvotes : ℕ) (score : ℤ) : 
  upvotes = (3 * total_votes) / 4 →
  downvotes = total_votes / 4 →
  score = 150 →
  (upvotes : ℤ) - (downvotes : ℤ) = score →
  total_votes = 300 := by
sorry

end NUMINAMATH_CALUDE_sabrina_video_votes_l2392_239276


namespace NUMINAMATH_CALUDE_min_even_integers_l2392_239243

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 → 
  a + b + c + d = 50 → 
  a + b + c + d + e + f = 70 → 
  ∃ (evens : Finset ℤ), evens ⊆ {a, b, c, d, e, f} ∧ 
    (∀ x ∈ evens, Even x) ∧ 
    evens.card = 2 ∧ 
    (∀ (other_evens : Finset ℤ), other_evens ⊆ {a, b, c, d, e, f} → 
      (∀ x ∈ other_evens, Even x) → other_evens.card ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l2392_239243


namespace NUMINAMATH_CALUDE_final_student_count_l2392_239249

theorem final_student_count (initial_students leaving_students new_students : ℕ) :
  initial_students = 11 →
  leaving_students = 6 →
  new_students = 42 →
  initial_students - leaving_students + new_students = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l2392_239249


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l2392_239233

theorem cube_root_equation_solution :
  ∃ (a b c : ℕ+),
    (2 * (7^(1/3) + 6^(1/3))^(1/2) : ℝ) = a^(1/3) - b^(1/3) + c^(1/3) ∧
    a + b + c = 42 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l2392_239233


namespace NUMINAMATH_CALUDE_profit_percent_when_cost_is_40_percent_of_selling_price_l2392_239228

theorem profit_percent_when_cost_is_40_percent_of_selling_price :
  ∀ (selling_price : ℝ), selling_price > 0 →
  let cost_price := 0.4 * selling_price
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  profit_percent = 150 := by
sorry

end NUMINAMATH_CALUDE_profit_percent_when_cost_is_40_percent_of_selling_price_l2392_239228


namespace NUMINAMATH_CALUDE_john_payment_first_year_l2392_239237

/- Define the family members -/
inductive FamilyMember
| John
| Wife
| Son
| Daughter

/- Define whether a family member is extended or not -/
def isExtended : FamilyMember → Bool
  | FamilyMember.Wife => true
  | _ => false

/- Define the initial membership fee -/
def initialMembershipFee : ℕ := 4000

/- Define the monthly cost for each family member -/
def monthlyCost : FamilyMember → ℕ
  | FamilyMember.John => 1000
  | FamilyMember.Wife => 1200
  | FamilyMember.Son => 800
  | FamilyMember.Daughter => 900

/- Define the membership fee discount rate for extended family members -/
def membershipDiscountRate : ℚ := 1/5

/- Define the monthly fee discount rate for extended family members -/
def monthlyDiscountRate : ℚ := 1/10

/- Define the number of months in a year -/
def monthsInYear : ℕ := 12

/- Define John's payment fraction -/
def johnPaymentFraction : ℚ := 1/2

/- Theorem statement -/
theorem john_payment_first_year :
  let totalCost := (FamilyMember.John :: FamilyMember.Wife :: FamilyMember.Son :: FamilyMember.Daughter :: []).foldl
    (fun acc member =>
      let membershipFee := if isExtended member then initialMembershipFee * (1 - membershipDiscountRate) else initialMembershipFee
      let monthlyFee := if isExtended member then monthlyCost member * (1 - monthlyDiscountRate) else monthlyCost member
      acc + membershipFee + monthlyFee * monthsInYear)
    0
  johnPaymentFraction * totalCost = 30280 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_first_year_l2392_239237


namespace NUMINAMATH_CALUDE_collinear_vectors_y_value_l2392_239288

theorem collinear_vectors_y_value (y : ℝ) : 
  let a : Fin 2 → ℝ := ![(-3), 1]
  let b : Fin 2 → ℝ := ![6, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) → y = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_y_value_l2392_239288


namespace NUMINAMATH_CALUDE_complex_calculation_l2392_239265

theorem complex_calculation (c d : ℂ) (hc : c = 3 + 2*I) (hd : d = 2 - 3*I) :
  3*c + 4*d = 17 - 6*I :=
by sorry

end NUMINAMATH_CALUDE_complex_calculation_l2392_239265


namespace NUMINAMATH_CALUDE_flower_arrangement_count_l2392_239290

/-- The number of ways to choose k items from n items -/
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of ways to arrange k items in k distinct positions -/
def arrangements (k : ℕ) : ℕ := Nat.factorial k

theorem flower_arrangement_count :
  let n : ℕ := 5  -- Total number of flower types
  let k : ℕ := 2  -- Number of flowers to pick
  (combinations n k) * (arrangements k) = 20 := by
  sorry

#eval (combinations 5 2) * (arrangements 2)  -- Should output 20

end NUMINAMATH_CALUDE_flower_arrangement_count_l2392_239290


namespace NUMINAMATH_CALUDE_no_solution_equation_l2392_239214

theorem no_solution_equation (x : ℝ) : 
  (4 * x - 1) / 6 - (5 * x - 2/3) / 10 + (9 - x/2) / 3 ≠ 101/20 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l2392_239214


namespace NUMINAMATH_CALUDE_function_properties_l2392_239251

/-- The function f(x) = ax² + bx + 1 -/
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g(x) = f(x) - kx -/
def g (a b k x : ℝ) : ℝ := f a b x - k * x

theorem function_properties (a b k : ℝ) :
  (∀ x, f a b x ≥ 0) ∧  -- Range of f(x) is [0, +∞)
  (f a b (-1) = 0) ∧    -- f(x) has a zero point at x = -1
  (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) -- g(x) is monotonic on [-2, 2]
  →
  (f a b = fun x ↦ x^2 + 2*x + 1) ∧  -- f(x) = x² + 2x + 1
  (k ≥ 6 ∨ k ≤ -2)                   -- Range of k
  := by sorry

end NUMINAMATH_CALUDE_function_properties_l2392_239251


namespace NUMINAMATH_CALUDE_passenger_disembark_ways_l2392_239264

theorem passenger_disembark_ways (n : ℕ) (s : ℕ) (h1 : n = 10) (h2 : s = 5) :
  s^n = 5^10 := by
  sorry

end NUMINAMATH_CALUDE_passenger_disembark_ways_l2392_239264


namespace NUMINAMATH_CALUDE_square_formation_theorem_l2392_239252

def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

def can_form_square (n : ℕ) : Bool :=
  sum_of_naturals n % 4 = 0

def min_breaks_to_square (n : ℕ) : ℕ :=
  if can_form_square n then 0
  else
    let total := sum_of_naturals n
    let target := (total + 3) / 4 * 4
    (target - total + 1) / 2

theorem square_formation_theorem :
  (min_breaks_to_square 12 = 2) ∧ (can_form_square 15 = true) := by
  sorry

end NUMINAMATH_CALUDE_square_formation_theorem_l2392_239252


namespace NUMINAMATH_CALUDE_line_translation_l2392_239275

/-- Given a line with equation y = -2x, prove that translating it upward by 1 unit results in the equation y = -2x + 1 -/
theorem line_translation (x y : ℝ) :
  (y = -2 * x) →  -- Original line equation
  (∃ (y' : ℝ), y' = y + 1 ∧ y' = -2 * x + 1) -- Translated line equation
  := by sorry

end NUMINAMATH_CALUDE_line_translation_l2392_239275


namespace NUMINAMATH_CALUDE_tom_has_sixteen_robots_l2392_239225

/-- The number of animal robots Michael has -/
def michael_robots : ℕ := 8

/-- The number of animal robots Tom has -/
def tom_robots : ℕ := 2 * michael_robots

/-- Theorem stating that Tom has 16 animal robots -/
theorem tom_has_sixteen_robots : tom_robots = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_has_sixteen_robots_l2392_239225


namespace NUMINAMATH_CALUDE_false_conjunction_implication_l2392_239226

theorem false_conjunction_implication : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_false_conjunction_implication_l2392_239226


namespace NUMINAMATH_CALUDE_square_difference_l2392_239204

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 64) (h2 : x * y = 10) :
  (x - y)^2 = 24 := by sorry

end NUMINAMATH_CALUDE_square_difference_l2392_239204


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2392_239231

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (k : ℕ), k ≤ 4 ∧ (9671 - k) % 5 = 0 ∧ ∀ (m : ℕ), m < k → (9671 - m) % 5 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2392_239231


namespace NUMINAMATH_CALUDE_f_properties_l2392_239216

noncomputable def f (x : ℝ) : ℝ := Real.log (abs x) / Real.log 2

theorem f_properties :
  (∀ x ≠ 0, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2392_239216


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2392_239250

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

theorem quadratic_minimum :
  ∃ (x_min : ℝ), f x_min = 0 ∧ ∀ (x : ℝ), f x ≥ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2392_239250


namespace NUMINAMATH_CALUDE_min_quadratic_expression_l2392_239254

theorem min_quadratic_expression :
  ∃ (x : ℝ), ∀ (y : ℝ), 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_min_quadratic_expression_l2392_239254


namespace NUMINAMATH_CALUDE_inequality_theorem_l2392_239296

open Set

-- Define the interval (0,+∞)
def openPositiveReals : Set ℝ := {x : ℝ | x > 0}

-- Define the properties of functions f and g
def hasContinuousDerivative (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ Differentiable ℝ f

-- Define the inequality condition
def satisfiesInequality (f g : ℝ → ℝ) : Prop :=
  ∀ x ∈ openPositiveReals, f x > x * (deriv f x) - x^2 * (deriv g x)

-- Theorem statement
theorem inequality_theorem (f g : ℝ → ℝ) 
  (hf : hasContinuousDerivative f) (hg : hasContinuousDerivative g)
  (h_ineq : satisfiesInequality f g) :
  2 * g 2 + 2 * f 1 > f 2 + 2 * g 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2392_239296


namespace NUMINAMATH_CALUDE_library_visitors_average_l2392_239201

/-- Calculates the average number of visitors per day for a month in a library --/
def averageVisitorsPerDay (
  daysInMonth : ℕ)
  (sundayVisitors : ℕ)
  (regularDayVisitors : ℕ)
  (publicHolidays : ℕ)
  (specialEvents : ℕ) : ℚ :=
  let sundayCount := (daysInMonth + 6) / 7
  let regularDays := daysInMonth - sundayCount - publicHolidays - specialEvents
  let totalVisitors := 
    sundayCount * sundayVisitors +
    regularDays * regularDayVisitors +
    publicHolidays * (2 * regularDayVisitors) +
    specialEvents * (3 * regularDayVisitors)
  (totalVisitors : ℚ) / daysInMonth

theorem library_visitors_average :
  averageVisitorsPerDay 30 510 240 2 1 = 308 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l2392_239201


namespace NUMINAMATH_CALUDE_child_ticket_price_l2392_239220

theorem child_ticket_price (total_revenue : ℕ) (adult_price : ℕ) (total_tickets : ℕ) (child_tickets : ℕ) :
  total_revenue = 104 →
  adult_price = 6 →
  total_tickets = 21 →
  child_tickets = 11 →
  ∃ (child_price : ℕ), child_price * child_tickets + adult_price * (total_tickets - child_tickets) = total_revenue ∧ child_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_price_l2392_239220


namespace NUMINAMATH_CALUDE_no_common_real_solution_l2392_239263

theorem no_common_real_solution :
  ¬∃ (x y : ℝ), x^2 + y^2 + 8 = 0 ∧ x^2 - 5*y + 20 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_real_solution_l2392_239263


namespace NUMINAMATH_CALUDE_divisibility_by_3_and_2_l2392_239229

theorem divisibility_by_3_and_2 (n : ℕ) : 
  (3 ∣ n) → (2 ∣ n) → (6 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_3_and_2_l2392_239229


namespace NUMINAMATH_CALUDE_camp_gender_ratio_l2392_239213

theorem camp_gender_ratio (total : ℕ) (boys_added : ℕ) (girls_percent : ℝ) : 
  total = 100 → 
  boys_added = 100 → 
  girls_percent = 5 → 
  (total : ℝ) * girls_percent / 100 = (total - ((total + boys_added) * girls_percent / 100)) → 
  (100 : ℝ) * (total - ((total + boys_added) * girls_percent / 100)) / total = 90 :=
by sorry

end NUMINAMATH_CALUDE_camp_gender_ratio_l2392_239213


namespace NUMINAMATH_CALUDE_pastry_count_consistency_l2392_239219

/-- Represents the number of pastries in different states --/
structure Pastries where
  initial : ℕ
  sold : ℕ
  remaining : ℕ

/-- The problem statement --/
theorem pastry_count_consistency (p : Pastries) 
  (h1 : p.initial = 148)
  (h2 : p.sold = 103)
  (h3 : p.remaining = 45) :
  p.initial = p.sold + p.remaining := by
  sorry

end NUMINAMATH_CALUDE_pastry_count_consistency_l2392_239219


namespace NUMINAMATH_CALUDE_arianna_daily_chores_l2392_239282

def hours_in_day : ℕ := 24
def work_hours : ℕ := 6
def sleep_hours : ℕ := 13

theorem arianna_daily_chores : 
  hours_in_day - (work_hours + sleep_hours) = 5 := by
  sorry

end NUMINAMATH_CALUDE_arianna_daily_chores_l2392_239282


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l2392_239274

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- Define the constants a, b, and c
def a : ℝ := sorry
def b : ℝ := sorry
def c : ℝ := sorry

-- State the theorem
theorem function_equality_implies_sum (x : ℝ) :
  (∀ x, f (x + 4) = 2 * x^2 + 8 * x + 10) ∧
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 4 := by sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l2392_239274
