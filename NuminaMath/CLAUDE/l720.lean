import Mathlib

namespace NUMINAMATH_CALUDE_remainder_theorem_l720_72085

/-- The polynomial P(x) = 5x^4 - 13x^3 + 3x^2 - x + 15 -/
def P (x : ℝ) : ℝ := 5*x^4 - 13*x^3 + 3*x^2 - x + 15

/-- The divisor polynomial d(x) = 3x - 9 -/
def d (x : ℝ) : ℝ := 3*x - 9

/-- Theorem stating that the remainder when P(x) is divided by d(x) is 93 -/
theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, P x = d x * q x + 93 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l720_72085


namespace NUMINAMATH_CALUDE_triangle_inequality_from_seven_numbers_l720_72062

theorem triangle_inequality_from_seven_numbers
  (a : Fin 7 → ℝ)
  (h : ∀ i, 1 < a i ∧ a i < 13) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    a i + a j > a k ∧
    a j + a k > a i ∧
    a k + a i > a j :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_seven_numbers_l720_72062


namespace NUMINAMATH_CALUDE_set_membership_implies_value_l720_72029

theorem set_membership_implies_value (a : ℝ) : 
  3 ∈ ({1, a, a - 2} : Set ℝ) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_set_membership_implies_value_l720_72029


namespace NUMINAMATH_CALUDE_ratio_problem_l720_72052

theorem ratio_problem (a b : ℝ) (h1 : a / b = 150 / 1) (h2 : a = 300) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l720_72052


namespace NUMINAMATH_CALUDE_intersection_when_m_3_union_equals_A_iff_l720_72065

-- Define sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x + m^2 - 1 < 0}

-- Theorem for part 1
theorem intersection_when_m_3 : 
  A ∩ B 3 = {x | 2 < x ∧ x < 3} := by sorry

-- Theorem for part 2
theorem union_equals_A_iff (m : ℝ) : 
  A ∪ B m = A ↔ 0 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_3_union_equals_A_iff_l720_72065


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l720_72038

/-- Represents a solid formed by unit cubes --/
structure CubeSolid where
  base_layer : Nat
  second_layer : Nat
  third_layer : Nat
  top_layer : Nat

/-- Calculates the surface area of a CubeSolid --/
def surface_area (solid : CubeSolid) : Nat :=
  sorry

/-- The specific solid described in the problem --/
def problem_solid : CubeSolid :=
  { base_layer := 4
  , second_layer := 4
  , third_layer := 3
  , top_layer := 1 }

/-- Theorem stating that the surface area of the problem_solid is 28 --/
theorem problem_solid_surface_area :
  surface_area problem_solid = 28 :=
sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l720_72038


namespace NUMINAMATH_CALUDE_capital_growth_l720_72008

def capital_sequence : ℕ → ℝ
  | 0 => 60
  | n + 1 => 1.5 * capital_sequence n - 15

theorem capital_growth (n : ℕ) :
  -- a₁ = 60
  capital_sequence 0 = 60 ∧
  -- {aₙ - 3} forms a geometric sequence
  (∀ k : ℕ, capital_sequence (k + 1) - 3 = 1.5 * (capital_sequence k - 3)) ∧
  -- By the end of 2026 (6 years from 2021), the remaining capital will exceed 210 million yuan
  ∃ m : ℕ, m ≤ 6 ∧ capital_sequence m > 210 :=
by sorry

end NUMINAMATH_CALUDE_capital_growth_l720_72008


namespace NUMINAMATH_CALUDE_largest_product_bound_l720_72080

theorem largest_product_bound (a : Fin 1985 → ℕ) 
  (h_perm : Function.Bijective a) 
  (h_range : ∀ i, a i ∈ Finset.range 1986) : 
  (Finset.range 1985).sup (λ k => (k + 1) * a k) ≥ 993^2 := by
  sorry

end NUMINAMATH_CALUDE_largest_product_bound_l720_72080


namespace NUMINAMATH_CALUDE_most_advantageous_order_l720_72027

-- Define the probabilities
variable (p₁ p₂ p₃ : ℝ)

-- Define the conditions
variable (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1)
variable (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1)
variable (h₃ : 0 ≤ p₃ ∧ p₃ ≤ 1)
variable (h₄ : p₃ < p₁)
variable (h₅ : p₁ < p₂)

-- Define the probability of winning two games in a row with p₂ as the second opponent
def prob_p₂_second := p₂ * (p₁ + p₃ - p₁ * p₃)

-- Define the probability of winning two games in a row with p₁ as the second opponent
def prob_p₁_second := p₁ * (p₂ + p₃ - p₂ * p₃)

-- The theorem to prove
theorem most_advantageous_order :
  prob_p₂_second p₁ p₂ p₃ > prob_p₁_second p₁ p₂ p₃ :=
sorry

end NUMINAMATH_CALUDE_most_advantageous_order_l720_72027


namespace NUMINAMATH_CALUDE_container_emptying_l720_72064

/-- Represents the state of the three containers -/
structure ContainerState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a valid transfer between containers -/
inductive Transfer : ContainerState → ContainerState → Prop where
  | ab {s t : ContainerState} : t.a = s.a + s.a ∧ t.b = s.b - s.a ∧ t.c = s.c → Transfer s t
  | ac {s t : ContainerState} : t.a = s.a + s.a ∧ t.b = s.b ∧ t.c = s.c - s.a → Transfer s t
  | ba {s t : ContainerState} : t.a = s.a - s.b ∧ t.b = s.b + s.b ∧ t.c = s.c → Transfer s t
  | bc {s t : ContainerState} : t.a = s.a ∧ t.b = s.b + s.b ∧ t.c = s.c - s.b → Transfer s t
  | ca {s t : ContainerState} : t.a = s.a - s.c ∧ t.b = s.b ∧ t.c = s.c + s.c → Transfer s t
  | cb {s t : ContainerState} : t.a = s.a ∧ t.b = s.b - s.c ∧ t.c = s.c + s.c → Transfer s t

/-- A sequence of transfers -/
def TransferSeq : ContainerState → ContainerState → Prop :=
  Relation.ReflTransGen Transfer

/-- The main theorem stating that it's always possible to empty a container -/
theorem container_emptying (initial : ContainerState) : 
  ∃ (final : ContainerState), TransferSeq initial final ∧ (final.a = 0 ∨ final.b = 0 ∨ final.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_container_emptying_l720_72064


namespace NUMINAMATH_CALUDE_sortable_configurations_after_three_passes_l720_72011

/-- The number of sortable book configurations after three passes -/
def sortableConfigurations (n : ℕ) : ℕ :=
  6 * 4^(n - 3)

/-- Theorem stating the number of sortable configurations for n ≥ 3 books after three passes -/
theorem sortable_configurations_after_three_passes (n : ℕ) (h : n ≥ 3) :
  sortableConfigurations n = 6 * 4^(n - 3) := by
  sorry

end NUMINAMATH_CALUDE_sortable_configurations_after_three_passes_l720_72011


namespace NUMINAMATH_CALUDE_inequality_proof_l720_72087

theorem inequality_proof (x a b : ℝ) (h1 : x < a) (h2 : a < 0) (h3 : b = -a) :
  x^2 > b^2 ∧ b^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l720_72087


namespace NUMINAMATH_CALUDE_tom_catches_jerry_l720_72026

/-- The time it takes for Tom to catch Jerry in the given scenario --/
def catch_time : ℝ → Prop := λ t =>
  let rectangle_width : ℝ := 15
  let rectangle_length : ℝ := 30
  let tom_speed : ℝ := 5
  let jerry_speed : ℝ := 3
  16 * t^2 - 45 * Real.sqrt 2 * t - 225 = 0

theorem tom_catches_jerry : ∃ t : ℝ, catch_time t := by sorry

end NUMINAMATH_CALUDE_tom_catches_jerry_l720_72026


namespace NUMINAMATH_CALUDE_expression_value_l720_72021

theorem expression_value (a b : ℝ) (h1 : a = Real.sqrt 3 - Real.sqrt 2) (h2 : b = Real.sqrt 3 + Real.sqrt 2) :
  a^2 + 3*a*b + b^2 - a + b = 13 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l720_72021


namespace NUMINAMATH_CALUDE_terminal_side_of_half_angle_l720_72083

def is_in_third_quadrant (α : Real) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

def is_in_second_or_fourth_quadrant (α : Real) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α ∧ α < n * 360 + 135) ∨
  (∃ n : ℤ, n * 360 + 270 < α ∧ α < n * 360 + 315)

theorem terminal_side_of_half_angle (α : Real) :
  is_in_third_quadrant α → is_in_second_or_fourth_quadrant (α / 2) :=
by sorry

end NUMINAMATH_CALUDE_terminal_side_of_half_angle_l720_72083


namespace NUMINAMATH_CALUDE_paved_road_time_l720_72017

/-- Calculates the time spent on a paved road given total trip distance,
    dirt road travel time and speed, and speed difference between paved and dirt roads. -/
theorem paved_road_time (total_distance : ℝ) (dirt_time : ℝ) (dirt_speed : ℝ) (speed_diff : ℝ) :
  total_distance = 200 →
  dirt_time = 3 →
  dirt_speed = 32 →
  speed_diff = 20 →
  (total_distance - dirt_time * dirt_speed) / (dirt_speed + speed_diff) = 2 := by
  sorry

#check paved_road_time

end NUMINAMATH_CALUDE_paved_road_time_l720_72017


namespace NUMINAMATH_CALUDE_expression_simplification_l720_72048

theorem expression_simplification (x : ℝ) : 2*x + 3*x^2 + 1 - (6 - 2*x - 3*x^2) = 6*x^2 + 4*x - 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l720_72048


namespace NUMINAMATH_CALUDE_largest_inscribed_rectangle_area_l720_72067

theorem largest_inscribed_rectangle_area (r : ℝ) (h : r = 6) :
  let d := 2 * r
  let s := d / Real.sqrt 2
  s * s = 72 := by sorry

end NUMINAMATH_CALUDE_largest_inscribed_rectangle_area_l720_72067


namespace NUMINAMATH_CALUDE_spade_calculation_l720_72002

-- Define the ♠ operation
def spade (x y : ℝ) : ℝ := (x + 2*y)^2 * (x - y)

-- State the theorem
theorem spade_calculation : spade 3 (spade 2 3) = 1046875 := by
  sorry

end NUMINAMATH_CALUDE_spade_calculation_l720_72002


namespace NUMINAMATH_CALUDE_nuts_in_masons_car_l720_72068

/-- The number of busy squirrels -/
def busy_squirrels : ℕ := 2

/-- The number of nuts stored by each busy squirrel per day -/
def busy_nuts_per_day : ℕ := 30

/-- The number of days busy squirrels have been storing nuts -/
def busy_days : ℕ := 35

/-- The number of slightly lazy squirrels -/
def lazy_squirrels : ℕ := 3

/-- The number of nuts stored by each slightly lazy squirrel per day -/
def lazy_nuts_per_day : ℕ := 20

/-- The number of days slightly lazy squirrels have been storing nuts -/
def lazy_days : ℕ := 40

/-- The number of extremely sleepy squirrels -/
def sleepy_squirrels : ℕ := 1

/-- The number of nuts stored by the extremely sleepy squirrel per day -/
def sleepy_nuts_per_day : ℕ := 10

/-- The number of days the extremely sleepy squirrel has been storing nuts -/
def sleepy_days : ℕ := 45

/-- The total number of nuts in Mason's car -/
def total_nuts : ℕ := busy_squirrels * busy_nuts_per_day * busy_days +
                      lazy_squirrels * lazy_nuts_per_day * lazy_days +
                      sleepy_squirrels * sleepy_nuts_per_day * sleepy_days

theorem nuts_in_masons_car : total_nuts = 4950 := by
  sorry

end NUMINAMATH_CALUDE_nuts_in_masons_car_l720_72068


namespace NUMINAMATH_CALUDE_number_problem_l720_72072

theorem number_problem : ∃ x : ℚ, 34 + 3 * x = 49 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l720_72072


namespace NUMINAMATH_CALUDE_x_eq_x_squared_is_quadratic_l720_72091

/-- A quadratic equation in terms of x is an equation that can be written in the form ax^2 + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function f(x) = x^2 - x represents the equation x = x^2 -/
def f (x : ℝ) : ℝ := x^2 - x

/-- Theorem: The equation x = x^2 is a quadratic equation in terms of x -/
theorem x_eq_x_squared_is_quadratic : is_quadratic_equation f := by
  sorry


end NUMINAMATH_CALUDE_x_eq_x_squared_is_quadratic_l720_72091


namespace NUMINAMATH_CALUDE_alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta_l720_72035

theorem alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta :
  ∃ (α β : Real),
    0 < α ∧ α < π/2 ∧
    0 < β ∧ β < π/2 ∧
    (
      (α > β ∧ ¬(Real.sin α > Real.sin β)) ∧
      (Real.sin α > Real.sin β ∧ ¬(α > β))
    ) :=
by sorry

end NUMINAMATH_CALUDE_alpha_gt_beta_not_sufficient_nor_necessary_for_sin_alpha_gt_sin_beta_l720_72035


namespace NUMINAMATH_CALUDE_ellipse_and_tangent_line_l720_72012

/-- Given an ellipse and a line passing through its vertex and focus, 
    prove the standard equation of the ellipse and its tangent line equation. -/
theorem ellipse_and_tangent_line 
  (a b : ℝ) 
  (h_ab : a > b ∧ b > 0) 
  (ellipse : ℝ → ℝ → Prop) 
  (line : ℝ → ℝ → Prop) 
  (h_ellipse : ellipse = λ x y => x^2/a^2 + y^2/b^2 = 1)
  (h_line : line = λ x y => Real.sqrt 6 * x + 2 * y - 2 * Real.sqrt 6 = 0)
  (h_vertex_focus : ∃ (E F : ℝ × ℝ), 
    ellipse E.1 E.2 ∧ 
    ellipse F.1 F.2 ∧ 
    line E.1 E.2 ∧ 
    line F.1 F.2 ∧ 
    (E.1 = 0 ∧ E.2 = Real.sqrt 6) ∧ 
    (F.1 = 2 ∧ F.2 = 0)) :
  (∀ x y, ellipse x y ↔ x^2/10 + y^2/6 = 1) ∧
  (∀ x y, (Real.sqrt 5 / 10) * x + (Real.sqrt 3 / 6) * y = 1 →
    (x = Real.sqrt 5 ∧ y = Real.sqrt 3) ∨
    ¬(ellipse x y)) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_and_tangent_line_l720_72012


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l720_72040

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

/-- The distance between two points in a 2D plane. -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The area of a quadrilateral given its four vertices. -/
def quadrilateralArea (A B C D : ℝ × ℝ) : ℝ := sorry

theorem cyclic_quadrilateral_area 
  (A B C D : ℝ × ℝ) 
  (h_cyclic : CyclicQuadrilateral A B C D)
  (h_AB : distance A B = 2)
  (h_BC : distance B C = 6)
  (h_CD : distance C D = 4)
  (h_DA : distance D A = 4) :
  quadrilateralArea A B C D = 8 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_area_l720_72040


namespace NUMINAMATH_CALUDE_bing_dwen_dwen_prices_l720_72049

theorem bing_dwen_dwen_prices 
  (total_budget : ℝ) 
  (budget_A : ℝ) 
  (price_difference : ℝ) 
  (quantity_ratio : ℝ) :
  total_budget = 1700 →
  budget_A = 800 →
  price_difference = 25 →
  quantity_ratio = 3 →
  ∃ (price_B : ℝ) (price_A : ℝ),
    price_B = 15 ∧
    price_A = 40 ∧
    price_A = price_B + price_difference ∧
    (total_budget - budget_A) / price_B = quantity_ratio * (budget_A / price_A) := by
  sorry

#check bing_dwen_dwen_prices

end NUMINAMATH_CALUDE_bing_dwen_dwen_prices_l720_72049


namespace NUMINAMATH_CALUDE_polynomial_equality_l720_72042

theorem polynomial_equality (m : ℝ) : (2 * m^2 + 3 * m - 4) + (m^2 - 2 * m + 3) = 3 * m^2 + m - 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l720_72042


namespace NUMINAMATH_CALUDE_fencemaker_problem_l720_72001

theorem fencemaker_problem (length width : ℝ) : 
  width = 40 → 
  length * width = 200 → 
  2 * length + width = 50 := by
sorry

end NUMINAMATH_CALUDE_fencemaker_problem_l720_72001


namespace NUMINAMATH_CALUDE_tangent_line_at_origin_monotonically_decreasing_when_m_positive_extremum_values_iff_m_negative_l720_72004

noncomputable section

variables (m : ℝ) (x : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m^2 * x) / (x^2 - m)

theorem tangent_line_at_origin (h : m = 1) :
  ∃ (k : ℝ), ∀ (x y : ℝ), y = f m x → (x = 0 ∧ y = 0) → x + y = 0 :=
sorry

theorem monotonically_decreasing_when_m_positive (h : m > 0) :
  ∀ (x₁ x₂ : ℝ), x₁ < x₂ → f m x₁ > f m x₂ :=
sorry

theorem extremum_values_iff_m_negative :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f m x₁ = f m x₂ ∧ 
   (∀ (x : ℝ), f m x ≤ f m x₁ ∨ f m x ≤ f m x₂)) ↔ m < 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_origin_monotonically_decreasing_when_m_positive_extremum_values_iff_m_negative_l720_72004


namespace NUMINAMATH_CALUDE_travis_has_10000_apples_l720_72059

/-- The number of apples Travis has given the conditions of the problem -/
def travis_apples (apples_per_box : ℕ) (price_per_box : ℕ) (total_revenue : ℕ) : ℕ :=
  (total_revenue / price_per_box) * apples_per_box

/-- Theorem stating that Travis has 10000 apples given the problem conditions -/
theorem travis_has_10000_apples :
  travis_apples 50 35 7000 = 10000 := by
  sorry

end NUMINAMATH_CALUDE_travis_has_10000_apples_l720_72059


namespace NUMINAMATH_CALUDE_value_of_y_l720_72022

theorem value_of_y (x y : ℝ) (h1 : x^(2*y) = 16) (h2 : x = 16) : y = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l720_72022


namespace NUMINAMATH_CALUDE_key_lime_juice_yield_l720_72028

def recipe_amount : ℚ := 1/4
def tablespoons_per_cup : ℕ := 16
def key_limes_needed : ℕ := 8

theorem key_lime_juice_yield : 
  let doubled_amount : ℚ := 2 * recipe_amount
  let total_tablespoons : ℚ := doubled_amount * tablespoons_per_cup
  let juice_per_lime : ℚ := total_tablespoons / key_limes_needed
  juice_per_lime = 1 := by sorry

end NUMINAMATH_CALUDE_key_lime_juice_yield_l720_72028


namespace NUMINAMATH_CALUDE_x_equals_ten_l720_72031

/-- A structure representing the number pyramid --/
structure NumberPyramid where
  row1_left : ℕ
  row1_right : ℕ
  row2_left : ℕ
  row2_middle : ℕ → ℕ
  row2_right : ℕ → ℕ
  row3_left : ℕ → ℕ
  row3_right : ℕ → ℕ
  row4 : ℕ → ℕ

/-- The theorem stating that x must be 10 given the conditions --/
theorem x_equals_ten (pyramid : NumberPyramid) 
  (h1 : pyramid.row1_left = 11)
  (h2 : pyramid.row1_right = 49)
  (h3 : pyramid.row2_left = 11)
  (h4 : ∀ x, pyramid.row2_middle x = 6 + x)
  (h5 : ∀ x, pyramid.row2_right x = x + 7)
  (h6 : ∀ x, pyramid.row3_left x = pyramid.row2_left + pyramid.row2_middle x)
  (h7 : ∀ x, pyramid.row3_right x = pyramid.row2_middle x + pyramid.row2_right x)
  (h8 : ∀ x, pyramid.row4 x = pyramid.row3_left x + pyramid.row3_right x)
  (h9 : pyramid.row4 10 = 60) :
  ∃ x, x = 10 ∧ pyramid.row4 x = 60 :=
sorry

end NUMINAMATH_CALUDE_x_equals_ten_l720_72031


namespace NUMINAMATH_CALUDE_polynomial_factorization_l720_72033

theorem polynomial_factorization (a b : ℝ) : 
  (∀ x, x^2 - 3*x + a = (x - 5) * (x - b)) → (a = -10 ∧ b = -2) := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l720_72033


namespace NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l720_72044

-- Problem 1
theorem simplify_expression (a b : ℝ) :
  4 * a^2 + 3 * b^2 + 2 * a * b - 3 * a^2 - 3 * b * a - a^2 = a^2 - a * b + 3 * b^2 := by
  sorry

-- Problem 2
theorem simplify_and_evaluate (x : ℝ) (h : x = -3) :
  3 * x - 4 * x^2 + 7 - 3 * x + 2 * x^2 + 1 = -10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_simplify_and_evaluate_l720_72044


namespace NUMINAMATH_CALUDE_two_special_birth_years_l720_72000

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem two_special_birth_years :
  ∃ (y1 y2 : ℕ),
    y1 ≠ y2 ∧
    y1 ≥ 1900 ∧ y1 ≤ 2021 ∧
    y2 ≥ 1900 ∧ y2 ≤ 2021 ∧
    2021 - y1 = sum_of_digits y1 ∧
    2021 - y2 = sum_of_digits y2 ∧
    2022 - y1 = 8 ∧
    2022 - y2 = 26 :=
by sorry

end NUMINAMATH_CALUDE_two_special_birth_years_l720_72000


namespace NUMINAMATH_CALUDE_polynomial_factorization_sum_l720_72066

theorem polynomial_factorization_sum (a b c : ℝ) : 
  (∀ x, x^2 + 17*x + 52 = (x + a) * (x + b)) →
  (∀ x, x^2 + 7*x - 60 = (x + b) * (x - c)) →
  a + b + c = 27 := by
sorry

end NUMINAMATH_CALUDE_polynomial_factorization_sum_l720_72066


namespace NUMINAMATH_CALUDE_warehouse_weight_limit_l720_72024

theorem warehouse_weight_limit (P : ℕ) (certain_weight : ℝ) : 
  (P : ℝ) * 0.3 < 75 ∧ 
  (P : ℝ) * 0.2 = 48 ∧ 
  (P : ℝ) * 0.8 ≥ certain_weight ∧ 
  24 ≥ certain_weight ∧ 24 < 75 →
  certain_weight = 75 := by
sorry

end NUMINAMATH_CALUDE_warehouse_weight_limit_l720_72024


namespace NUMINAMATH_CALUDE_max_t_value_max_t_is_negative_one_l720_72078

open Real

noncomputable def f (x : ℝ) : ℝ := log x / (x + 1)

theorem max_t_value (t : ℝ) :
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t / x > log x / (x - 1)) →
  t ≤ -1 :=
by sorry

theorem max_t_is_negative_one :
  ∃ t : ℝ, t = -1 ∧
  (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t / x > log x / (x - 1)) ∧
  (∀ t' : ℝ, (∀ x : ℝ, x > 0 ∧ x ≠ 1 → f x - t' / x > log x / (x - 1)) → t' ≤ t) :=
by sorry

end NUMINAMATH_CALUDE_max_t_value_max_t_is_negative_one_l720_72078


namespace NUMINAMATH_CALUDE_zoo_visitors_l720_72047

/-- Given the number of visitors on Friday and the ratio of Saturday visitors to Friday visitors,
    prove that the number of visitors on Saturday is equal to the product of the Friday visitors and the ratio. -/
theorem zoo_visitors (friday_visitors : ℕ) (saturday_ratio : ℕ) :
  let saturday_visitors := friday_visitors * saturday_ratio
  saturday_visitors = friday_visitors * saturday_ratio :=
by sorry

/-- Example with the given values -/
example : 
  let friday_visitors : ℕ := 1250
  let saturday_ratio : ℕ := 3
  let saturday_visitors := friday_visitors * saturday_ratio
  saturday_visitors = 3750 :=
by sorry

end NUMINAMATH_CALUDE_zoo_visitors_l720_72047


namespace NUMINAMATH_CALUDE_max_product_value_l720_72089

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the theorem
theorem max_product_value :
  (∀ x, -7 ≤ f x ∧ f x ≤ 4) →
  (∀ x, -3 ≤ g x ∧ g x ≤ 2) →
  (∃ x, f x * g x = 21) ∧
  (∀ x, f x * g x ≤ 21) :=
sorry

end NUMINAMATH_CALUDE_max_product_value_l720_72089


namespace NUMINAMATH_CALUDE_expansion_properties_l720_72010

open Real Nat

/-- Represents the expansion of (1 + 2∛x)^n -/
def expansion (n : ℕ) (x : ℝ) := (1 + 2 * x^(1/3))^n

/-- Coefficient of the r-th term in the expansion -/
def coefficient (n r : ℕ) : ℝ := 2^r * choose n r

/-- Condition for the coefficient relation -/
def coefficient_condition (n : ℕ) : Prop :=
  ∃ r, coefficient n r = 2 * coefficient n (r-1) ∧
       coefficient n r = 5/6 * coefficient n (r+1)

/-- Sum of all coefficients in the expansion -/
def sum_coefficients (n : ℕ) : ℝ := 3^n

/-- Sum of all binomial coefficients -/
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

/-- Rational terms in the expansion -/
def rational_terms (n : ℕ) : List (ℝ × ℕ) :=
  [(1, 0), (560, 1), (448, 2), (2016, 3)]

theorem expansion_properties (n : ℕ) :
  coefficient_condition n →
  n = 7 ∧
  sum_coefficients n = 2187 ∧
  sum_binomial_coefficients n = 128 ∧
  rational_terms n = [(1, 0), (560, 1), (448, 2), (2016, 3)] :=
by sorry

end NUMINAMATH_CALUDE_expansion_properties_l720_72010


namespace NUMINAMATH_CALUDE_inequality_proof_l720_72009

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l720_72009


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l720_72076

-- Define the curve
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  f a (-1) = a + 2 →  -- The curve passes through the point (-1, a+2)
  f' a (-1) = 8 →     -- The slope of the tangent line at x = -1 is 8
  a = -6 :=           -- Then a must equal -6
by
  sorry


end NUMINAMATH_CALUDE_tangent_slope_implies_a_l720_72076


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_x_axis_l720_72092

/-- The curve represented by the equation x²/(sin θ + 3) + y²/(sin θ - 2) = 1 -/
def curve (x y θ : ℝ) : Prop :=
  x^2 / (Real.sin θ + 3) + y^2 / (Real.sin θ - 2) = 1

/-- The curve is a hyperbola with foci on the x-axis -/
theorem curve_is_hyperbola_with_foci_on_x_axis :
  ∀ x y θ, curve x y θ → 
    (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2/a^2 - y^2/b^2 = 1) ∧ 
    (∃ c : ℝ, c > 0 ∧ (∃ f₁ f₂ : ℝ × ℝ, f₁.1 = c ∧ f₁.2 = 0 ∧ f₂.1 = -c ∧ f₂.2 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_x_axis_l720_72092


namespace NUMINAMATH_CALUDE_arrangements_count_l720_72057

/-- Represents the number of acts in the show -/
def total_acts : ℕ := 6

/-- Represents the possible positions for Act A -/
def act_a_positions : Finset ℕ := {1, 2}

/-- Represents the possible positions for Act B -/
def act_b_positions : Finset ℕ := {2, 3, 4, 5}

/-- Represents the position of Act C -/
def act_c_position : ℕ := total_acts

/-- A function that calculates the number of arrangements -/
def count_arrangements : ℕ := sorry

/-- The theorem stating that the number of arrangements is 42 -/
theorem arrangements_count : count_arrangements = 42 := by sorry

end NUMINAMATH_CALUDE_arrangements_count_l720_72057


namespace NUMINAMATH_CALUDE_set_difference_proof_l720_72054

def I : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

theorem set_difference_proof : I \ A = {4} := by sorry

end NUMINAMATH_CALUDE_set_difference_proof_l720_72054


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l720_72088

/-- A geometric sequence with specific properties -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧
  a 2 * a 3 = 5 ∧
  a 5 * a 6 = 10

/-- Theorem stating the property of the 8th and 9th terms -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 8 * a 9 = 20 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l720_72088


namespace NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_smallest_prime_l720_72023

-- Define a function to get the nth smallest prime number
def nthSmallestPrime (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fourth_power_of_square_of_fourth_smallest_prime :
  (nthSmallestPrime 4)^2^4 = 5764801 := by sorry

end NUMINAMATH_CALUDE_fourth_power_of_square_of_fourth_smallest_prime_l720_72023


namespace NUMINAMATH_CALUDE_inequality_addition_l720_72018

theorem inequality_addition (x y : ℝ) (h : x < y) : x + 5 < y + 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_addition_l720_72018


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l720_72030

/-- A line passing through a point and perpendicular to a given line segment --/
def perpendicular_line (A B : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 ∧ P ≠ A}

/-- The equation of a line in the form ax + by + c = 0 --/
def line_equation (a b c : ℝ) : Set (ℝ × ℝ) :=
  {P | a * P.1 + b * P.2 + c = 0}

theorem perpendicular_line_equation :
  perpendicular_line (3, 4) (-3, 2) = line_equation 3 1 (-13) := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l720_72030


namespace NUMINAMATH_CALUDE_conor_weekly_vegetables_l720_72097

/-- Represents the number of vegetables Conor can chop in a day -/
structure DailyVegetables where
  eggplants : ℕ
  carrots : ℕ
  potatoes : ℕ
  onions : ℕ
  zucchinis : ℕ

/-- Calculates the total number of vegetables chopped in a day -/
def DailyVegetables.total (d : DailyVegetables) : ℕ :=
  d.eggplants + d.carrots + d.potatoes + d.onions + d.zucchinis

/-- Conor's chopping rate from Monday to Wednesday -/
def earlyWeekRate : DailyVegetables :=
  { eggplants := 12
    carrots := 9
    potatoes := 8
    onions := 15
    zucchinis := 7 }

/-- Conor's chopping rate from Thursday to Saturday -/
def lateWeekRate : DailyVegetables :=
  { eggplants := 7
    carrots := 5
    potatoes := 4
    onions := 10
    zucchinis := 4 }

/-- The number of days Conor works in the early part of the week -/
def earlyWeekDays : ℕ := 3

/-- The number of days Conor works in the late part of the week -/
def lateWeekDays : ℕ := 3

/-- Theorem: Conor can chop 243 vegetables in a week -/
theorem conor_weekly_vegetables : 
  earlyWeekDays * earlyWeekRate.total + lateWeekDays * lateWeekRate.total = 243 := by
  sorry

end NUMINAMATH_CALUDE_conor_weekly_vegetables_l720_72097


namespace NUMINAMATH_CALUDE_sum_of_right_angles_l720_72061

/-- A rectangle has 4 right angles -/
def rectangle_right_angles : ℕ := 4

/-- A square has 4 right angles -/
def square_right_angles : ℕ := 4

/-- The sum of right angles in a rectangle and a square -/
def total_right_angles : ℕ := rectangle_right_angles + square_right_angles

theorem sum_of_right_angles : total_right_angles = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_right_angles_l720_72061


namespace NUMINAMATH_CALUDE_min_value_theorem_l720_72094

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + r

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℝ) : Prop :=
  arithmetic_sequence a ∧
  (∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n + r) ∧
  a 2018 = a 2017 + 2 * a 2016 ∧
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2

-- State the theorem
theorem min_value_theorem (a : ℕ → ℝ) :
  problem_conditions a →
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ 4/m + 1/n ≥ 5/3 ∧
  (∀ k l : ℕ, k > 0 → l > 0 → 4/k + 1/l ≥ 4/m + 1/n) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l720_72094


namespace NUMINAMATH_CALUDE_tan_alpha_value_l720_72074

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l720_72074


namespace NUMINAMATH_CALUDE_linear_equation_solve_l720_72079

theorem linear_equation_solve (x y : ℝ) :
  2 * x - 7 * y = 5 → y = (2 * x - 5) / 7 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solve_l720_72079


namespace NUMINAMATH_CALUDE_bricklayer_electrician_problem_l720_72096

theorem bricklayer_electrician_problem :
  ∀ (bricklayer_rate electrician_rate total_pay bricklayer_hours : ℝ),
    bricklayer_rate = 12 →
    electrician_rate = 16 →
    total_pay = 1350 →
    bricklayer_hours = 67.5 →
    ∃ (electrician_hours : ℝ),
      electrician_hours = (total_pay - bricklayer_rate * bricklayer_hours) / electrician_rate ∧
      bricklayer_hours + electrician_hours = 101.25 :=
by sorry

end NUMINAMATH_CALUDE_bricklayer_electrician_problem_l720_72096


namespace NUMINAMATH_CALUDE_kabadi_players_l720_72032

theorem kabadi_players (kho_kho_only : ℕ) (both : ℕ) (total : ℕ) :
  kho_kho_only = 35 →
  both = 5 →
  total = 45 →
  ∃ kabadi : ℕ, kabadi = 15 ∧ total = kabadi + kho_kho_only - both :=
by sorry

end NUMINAMATH_CALUDE_kabadi_players_l720_72032


namespace NUMINAMATH_CALUDE_joan_money_found_l720_72075

def total_money (dimes_jacket : ℕ) (dimes_shorts : ℕ) (nickels_shorts : ℕ) 
  (quarters_jeans : ℕ) (pennies_jeans : ℕ) (nickels_backpack : ℕ) (pennies_backpack : ℕ) : ℚ :=
  (dimes_jacket + dimes_shorts) * (10 : ℚ) / 100 +
  (nickels_shorts + nickels_backpack) * (5 : ℚ) / 100 +
  quarters_jeans * (25 : ℚ) / 100 +
  (pennies_jeans + pennies_backpack) * (1 : ℚ) / 100

theorem joan_money_found :
  total_money 15 4 7 12 2 8 23 = (590 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_joan_money_found_l720_72075


namespace NUMINAMATH_CALUDE_arc_problem_l720_72019

theorem arc_problem (X Y Z : ℝ × ℝ) (d : ℝ) : 
  X.1 = 0 ∧ X.2 = 0 ∧  -- Assume X is at origin
  Y.1 = 15 ∧ Y.2 = 0 ∧  -- Assume Y is on x-axis
  Z.1^2 + Z.2^2 = (3 + d)^2 ∧  -- XZ = 3 + d
  (Z.1 - 15)^2 + Z.2^2 = (12 + d)^2 →  -- YZ = 12 + d
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_arc_problem_l720_72019


namespace NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l720_72050

theorem second_smallest_hot_dog_packs : ∃ (n : ℕ), n > 0 ∧
  (12 * n) % 8 = 6 ∧
  (∀ (m : ℕ), m > 0 ∧ m < n → (12 * m) % 8 ≠ 6) ∧
  (∃ (k : ℕ), k > 0 ∧ k < n ∧ (12 * k) % 8 = 6) ∧
  n = 7 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_hot_dog_packs_l720_72050


namespace NUMINAMATH_CALUDE_eric_erasers_l720_72025

/-- Given that Eric shares his erasers among 99 friends and each friend gets 94 erasers,
    prove that Eric has 9306 erasers in total. -/
theorem eric_erasers (num_friends : ℕ) (erasers_per_friend : ℕ) 
    (h1 : num_friends = 99) (h2 : erasers_per_friend = 94) : 
    num_friends * erasers_per_friend = 9306 := by
  sorry

end NUMINAMATH_CALUDE_eric_erasers_l720_72025


namespace NUMINAMATH_CALUDE_intersection_distance_l720_72082

/-- The distance between the intersection points of a line and a circle --/
theorem intersection_distance (x y : ℝ) : 
  (x - y + 1 = 0) → -- Line equation
  (x^2 + (y-2)^2 = 4) → -- Circle equation
  ∃ A B : ℝ × ℝ, -- Two intersection points
    A ≠ B ∧
    (A.1 - A.2 + 1 = 0) ∧ (A.1^2 + (A.2-2)^2 = 4) ∧ -- A satisfies both equations
    (B.1 - B.2 + 1 = 0) ∧ (B.1^2 + (B.2-2)^2 = 4) ∧ -- B satisfies both equations
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 14 -- Distance between A and B is √14
  := by sorry

end NUMINAMATH_CALUDE_intersection_distance_l720_72082


namespace NUMINAMATH_CALUDE_n_squared_divisible_by_144_l720_72099

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∀ w : ℕ+, w ∣ n → w ≤ 12) : 144 ∣ n^2 := by
  sorry

end NUMINAMATH_CALUDE_n_squared_divisible_by_144_l720_72099


namespace NUMINAMATH_CALUDE_square_in_base_b_l720_72036

/-- Represents a number in base b with digits d₂d₁d₀ --/
def base_b_number (b : ℕ) (d₂ d₁ d₀ : ℕ) : ℕ := d₂ * b^2 + d₁ * b + d₀

/-- The number 144 in base b --/
def number_144_b (b : ℕ) : ℕ := base_b_number b 1 4 4

theorem square_in_base_b (b : ℕ) (h : b > 4) :
  ∃ (n : ℕ), number_144_b b = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_in_base_b_l720_72036


namespace NUMINAMATH_CALUDE_expand_and_simplify_l720_72053

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l720_72053


namespace NUMINAMATH_CALUDE_parabola_sum_coefficients_l720_72046

/-- A parabola with equation x = ay^2 + by + c, vertex at (-3, 2), and passing through (-1, 0) -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ
  vertex_condition : -3 = a * 2^2 + b * 2 + c
  point_condition : -1 = a * 0^2 + b * 0 + c

/-- The sum of coefficients a, b, and c for the given parabola is -7/2 -/
theorem parabola_sum_coefficients (p : Parabola) : p.a + p.b + p.c = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_coefficients_l720_72046


namespace NUMINAMATH_CALUDE_polynomial_product_l720_72056

-- Define the polynomials f, g, and h
def f (x : ℝ) : ℝ := x^4 - x^3 - 1
def g (x : ℝ) : ℝ := x^8 - x^6 - 2*x^4 + 1
def h (x : ℝ) : ℝ := x^4 + x^3 - 1

-- State the theorem
theorem polynomial_product :
  (∀ x, g x = f x * h x) → (∀ x, h x = x^4 + x^3 - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_l720_72056


namespace NUMINAMATH_CALUDE_complex_subtraction_equality_l720_72060

theorem complex_subtraction_equality : ((1 - 1) - 1) - ((1 - (1 - 1))) = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_equality_l720_72060


namespace NUMINAMATH_CALUDE_ice_cream_cost_l720_72006

theorem ice_cream_cost (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) (quarters : ℕ) 
  (family_members : ℕ) (remaining_cents : ℕ) :
  pennies = 123 →
  nickels = 85 →
  dimes = 35 →
  quarters = 26 →
  family_members = 5 →
  remaining_cents = 48 →
  let total_cents := pennies + nickels * 5 + dimes * 10 + quarters * 25
  let spent_cents := total_cents - remaining_cents
  let cost_per_scoop := spent_cents / family_members
  cost_per_scoop = 300 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_cost_l720_72006


namespace NUMINAMATH_CALUDE_temperature_conversion_l720_72081

theorem temperature_conversion (t k some_number : ℝ) :
  t = 5 / 9 * (k - some_number) →
  t = 105 →
  k = 221 →
  some_number = 32 := by
sorry

end NUMINAMATH_CALUDE_temperature_conversion_l720_72081


namespace NUMINAMATH_CALUDE_building_height_is_270_l720_72093

/-- Calculates the height of a building with specified floor heights -/
def building_height (total_stories : ℕ) (first_half_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := (total_stories / 2) * first_half_height
  let second_half := (total_stories / 2) * (first_half_height + height_increase)
  first_half + second_half

/-- Proves that the height of the specified building is 270 feet -/
theorem building_height_is_270 :
  building_height 20 12 3 = 270 := by
  sorry

#eval building_height 20 12 3

end NUMINAMATH_CALUDE_building_height_is_270_l720_72093


namespace NUMINAMATH_CALUDE_cookie_ratio_l720_72020

/-- Proves that the ratio of cookies baked by Jake to Clementine is 2:1 given the problem conditions -/
theorem cookie_ratio (clementine jake tory : ℕ) (total_revenue : ℕ) : 
  clementine = 72 →
  tory = (jake + clementine) / 2 →
  total_revenue = 648 →
  2 * (clementine + jake + tory) = total_revenue →
  jake = 2 * clementine :=
by sorry

end NUMINAMATH_CALUDE_cookie_ratio_l720_72020


namespace NUMINAMATH_CALUDE_tangent_line_at_point_l720_72090

-- Define the curve
def curve (x y : ℝ) : Prop := x^3 - y = 0

-- Define the point of tangency
def point : ℝ × ℝ := (-2, -8)

-- Define the proposed tangent line equation
def tangent_line (x y : ℝ) : Prop := 12*x - y + 16 = 0

-- Theorem statement
theorem tangent_line_at_point :
  ∀ x y : ℝ,
  curve x y →
  (x, y) = point →
  tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_l720_72090


namespace NUMINAMATH_CALUDE_equation_solution_l720_72003

theorem equation_solution : ∃ x : ℝ, 45 - (x - (37 - (15 - 19))) = 58 ∧ x = 28 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l720_72003


namespace NUMINAMATH_CALUDE_d_is_zero_l720_72069

def d (n m : ℕ) : ℚ :=
  if m = 0 ∨ m = n then 0
  else if 0 < m ∧ m < n then
    (m * d (n-1) m + (2*n - m) * d (n-1) (m-1)) / m
  else 0

theorem d_is_zero (n m : ℕ) (h : m ≤ n) : d n m = 0 := by
  sorry

end NUMINAMATH_CALUDE_d_is_zero_l720_72069


namespace NUMINAMATH_CALUDE_power_product_simplification_l720_72039

theorem power_product_simplification (a : ℝ) : (36 * a^9)^4 * (63 * a^9)^4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l720_72039


namespace NUMINAMATH_CALUDE_max_payment_is_31_l720_72086

/-- Represents a four-digit number of the form 20** -/
def FourDigitNumber := { n : ℕ // 2000 ≤ n ∧ n ≤ 2099 }

/-- Calculates the payment for a given divisor -/
def payment (d : ℕ) : ℕ :=
  match d with
  | 1 => 1
  | 3 => 3
  | 5 => 5
  | 7 => 7
  | 9 => 9
  | 11 => 11
  | _ => 0

/-- Calculates the total payment for a number based on its divisibility -/
def totalPayment (n : FourDigitNumber) : ℕ :=
  (payment 1) +
  (if n.val % 3 = 0 then payment 3 else 0) +
  (if n.val % 5 = 0 then payment 5 else 0) +
  (if n.val % 7 = 0 then payment 7 else 0) +
  (if n.val % 9 = 0 then payment 9 else 0) +
  (if n.val % 11 = 0 then payment 11 else 0)

theorem max_payment_is_31 :
  ∃ (n : FourDigitNumber), totalPayment n = 31 ∧
  ∀ (m : FourDigitNumber), totalPayment m ≤ 31 :=
sorry

end NUMINAMATH_CALUDE_max_payment_is_31_l720_72086


namespace NUMINAMATH_CALUDE_david_zachary_pushup_difference_l720_72041

/-- Given that David did 62 push-ups and Zachary did 47 push-ups,
    prove that David did 15 more push-ups than Zachary. -/
theorem david_zachary_pushup_difference :
  let david_pushups : ℕ := 62
  let zachary_pushups : ℕ := 47
  david_pushups - zachary_pushups = 15 := by
  sorry

end NUMINAMATH_CALUDE_david_zachary_pushup_difference_l720_72041


namespace NUMINAMATH_CALUDE_band_repertoire_size_l720_72095

def prove_band_repertoire (first_set second_set encore third_and_fourth_avg : ℕ) : Prop :=
  let total_songs := first_set + second_set + encore + 2 * third_and_fourth_avg
  total_songs = 30

theorem band_repertoire_size :
  prove_band_repertoire 5 7 2 8 := by
  sorry

end NUMINAMATH_CALUDE_band_repertoire_size_l720_72095


namespace NUMINAMATH_CALUDE_range_of_a_l720_72007

/-- Given propositions p and q, where p is a necessary but not sufficient condition for q,
    prove that the range of real number a is [-1/2, 1]. -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, x^2 - ax - 2*a^2 < 0 → x^2 - 2*x - 3 < 0) ∧ 
  (∃ x, x^2 - 2*x - 3 < 0 ∧ x^2 - ax - 2*a^2 ≥ 0) →
  -1/2 ≤ a ∧ a ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l720_72007


namespace NUMINAMATH_CALUDE_carls_watermelon_profit_l720_72045

/-- Calculates the profit of a watermelon seller -/
def watermelon_profit (initial_count : ℕ) (final_count : ℕ) (price_per_melon : ℕ) : ℕ :=
  (initial_count - final_count) * price_per_melon

/-- Theorem: Carl's watermelon profit -/
theorem carls_watermelon_profit :
  watermelon_profit 53 18 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_carls_watermelon_profit_l720_72045


namespace NUMINAMATH_CALUDE_fraction_product_simplification_l720_72005

theorem fraction_product_simplification :
  (240 : ℚ) / 20 * 6 / 180 * 10 / 4 = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_product_simplification_l720_72005


namespace NUMINAMATH_CALUDE_peter_has_320_dollars_l720_72070

-- Define the friends' money amounts
def john_money : ℝ := 160
def peter_money : ℝ := 2 * john_money
def quincy_money : ℝ := peter_money + 20
def andrew_money : ℝ := 1.15 * quincy_money

-- Define the total money and expenses
def total_money : ℝ := john_money + peter_money + quincy_money + andrew_money
def item_cost : ℝ := 1200
def money_left : ℝ := 11

-- Theorem to prove
theorem peter_has_320_dollars :
  peter_money = 320 ∧
  john_money + peter_money + quincy_money + andrew_money = item_cost + money_left :=
by sorry

end NUMINAMATH_CALUDE_peter_has_320_dollars_l720_72070


namespace NUMINAMATH_CALUDE_washing_machine_loads_l720_72084

/-- Calculate the minimum number of loads required to wash a given number of items with a fixed machine capacity -/
def minimum_loads (total_items : ℕ) (machine_capacity : ℕ) : ℕ :=
  (total_items + machine_capacity - 1) / machine_capacity

/-- The washing machine capacity -/
def machine_capacity : ℕ := 12

/-- The total number of items to wash -/
def total_items : ℕ := 19 + 8 + 15 + 10

theorem washing_machine_loads :
  minimum_loads total_items machine_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_washing_machine_loads_l720_72084


namespace NUMINAMATH_CALUDE_vector_at_t_3_l720_72034

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  -- The vector on the line at any given t
  vector : ℝ → ℝ × ℝ × ℝ

/-- Theorem: Given a parameterized line with specific points, 
    we can determine the vector at t = 3 -/
theorem vector_at_t_3 
  (line : ParametricLine)
  (h1 : line.vector (-1) = (1, 3, 8))
  (h2 : line.vector 2 = (0, -2, -4)) :
  line.vector 3 = (-1/3, -11/3, -8) := by
  sorry

end NUMINAMATH_CALUDE_vector_at_t_3_l720_72034


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l720_72014

def A : Set ℝ := {x | x + 1 > 0}
def B : Set ℝ := {x | x^2 - x < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l720_72014


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l720_72071

theorem sum_of_roots_quadratic (a b : ℝ) : 
  (a^2 - a - 2 = 0) → (b^2 - b - 2 = 0) → (a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l720_72071


namespace NUMINAMATH_CALUDE_white_balls_count_l720_72058

theorem white_balls_count (x : ℕ) : 
  (3 : ℚ) / (3 + x) = 1/5 → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l720_72058


namespace NUMINAMATH_CALUDE_mad_hatter_winning_condition_l720_72077

/-- Represents the fraction of voters for each candidate and undecided voters -/
structure VoterFractions where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ
  undecided : ℝ

/-- Represents the additional fraction of undecided voters each candidate receives -/
structure UndecidedAllocation where
  mad_hatter : ℝ
  march_hare : ℝ
  dormouse : ℝ

/-- The minimum fraction of undecided voters the Mad Hatter needs to secure -/
def minimum_fraction_for_mad_hatter (v : VoterFractions) : ℝ :=
  0.7

theorem mad_hatter_winning_condition 
  (v : VoterFractions)
  (h1 : v.mad_hatter = 0.2)
  (h2 : v.march_hare = 0.25)
  (h3 : v.dormouse = 0.3)
  (h4 : v.undecided = 1 - (v.mad_hatter + v.march_hare + v.dormouse))
  (h5 : v.mad_hatter + v.march_hare + v.dormouse + v.undecided = 1) :
  ∀ (u : UndecidedAllocation),
    (u.mad_hatter + u.march_hare + u.dormouse = 1) →
    (u.mad_hatter ≥ minimum_fraction_for_mad_hatter v) →
    (v.mad_hatter + v.undecided * u.mad_hatter ≥ v.march_hare + v.undecided * u.march_hare) ∧
    (v.mad_hatter + v.undecided * u.mad_hatter ≥ v.dormouse + v.undecided * u.dormouse) :=
sorry

end NUMINAMATH_CALUDE_mad_hatter_winning_condition_l720_72077


namespace NUMINAMATH_CALUDE_smallest_five_digit_mod_13_l720_72037

theorem smallest_five_digit_mod_13 : 
  ∀ n : ℕ, n ≥ 10000 ∧ n ≡ 11 [MOD 13] → n ≥ 10009 :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_mod_13_l720_72037


namespace NUMINAMATH_CALUDE_average_age_combined_l720_72016

-- Define the groups and their properties
def num_fifth_graders : ℕ := 40
def avg_age_fifth_graders : ℚ := 12
def num_parents : ℕ := 60
def avg_age_parents : ℚ := 35
def num_teachers : ℕ := 10
def avg_age_teachers : ℚ := 45

-- Define the theorem
theorem average_age_combined :
  let total_people := num_fifth_graders + num_parents + num_teachers
  let total_age := num_fifth_graders * avg_age_fifth_graders +
                   num_parents * avg_age_parents +
                   num_teachers * avg_age_teachers
  total_age / total_people = 27.5454545 := by
  sorry


end NUMINAMATH_CALUDE_average_age_combined_l720_72016


namespace NUMINAMATH_CALUDE_golden_state_team_total_points_l720_72013

def golden_state_team_points : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun draymond curry kelly durant klay =>
    draymond = 12 ∧
    curry = 2 * draymond ∧
    kelly = 9 ∧
    durant = 2 * kelly ∧
    klay = draymond / 2 ∧
    draymond + curry + kelly + durant + klay = 69

theorem golden_state_team_total_points :
  ∃ (draymond curry kelly durant klay : ℕ),
    golden_state_team_points draymond curry kelly durant klay :=
by
  sorry

end NUMINAMATH_CALUDE_golden_state_team_total_points_l720_72013


namespace NUMINAMATH_CALUDE_polynomial_negative_roots_l720_72043

theorem polynomial_negative_roots (q : ℝ) (hq : q > 1/2) :
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^4 + q*x₁^3 + 3*x₁^2 + q*x₁ + 9 = 0 ∧
  x₂^4 + q*x₂^3 + 3*x₂^2 + q*x₂ + 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_negative_roots_l720_72043


namespace NUMINAMATH_CALUDE_jasons_books_l720_72073

theorem jasons_books (keith_books : ℕ) (total_books : ℕ) (h1 : keith_books = 20) (h2 : total_books = 41) :
  total_books - keith_books = 21 := by
sorry

end NUMINAMATH_CALUDE_jasons_books_l720_72073


namespace NUMINAMATH_CALUDE_prime_pairs_congruence_l720_72015

theorem prime_pairs_congruence (p q : ℕ) : 
  Prime p ∧ Prime q →
  (∀ x : ℤ, x^(3*p*q) ≡ x [ZMOD (3*p*q)]) →
  ((p = 11 ∧ q = 17) ∨ (p = 17 ∧ q = 11)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_congruence_l720_72015


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l720_72063

/-- Given a geometric sequence {a_n} where a₂ = 2 and a₆ = 8, 
    prove that a₃ * a₄ * a₅ = 64 -/
theorem geometric_sequence_product (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_a2 : a 2 = 2) 
  (h_a6 : a 6 = 8) : 
  a 3 * a 4 * a 5 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l720_72063


namespace NUMINAMATH_CALUDE_squares_below_specific_line_l720_72051

/-- Represents a line in 2D space --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Counts the number of unit squares below a line in the first quadrant --/
def countSquaresBelowLine (l : Line) : ℕ :=
  sorry

/-- The specific line 10x + 210y = 2100 --/
def specificLine : Line := { a := 10, b := 210, c := 2100 }

theorem squares_below_specific_line :
  countSquaresBelowLine specificLine = 941 :=
sorry

end NUMINAMATH_CALUDE_squares_below_specific_line_l720_72051


namespace NUMINAMATH_CALUDE_y_exceeds_x_by_100_percent_l720_72055

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : 
  y = 2 * x := by
sorry

end NUMINAMATH_CALUDE_y_exceeds_x_by_100_percent_l720_72055


namespace NUMINAMATH_CALUDE_sqrt_54_times_sqrt_one_third_l720_72098

theorem sqrt_54_times_sqrt_one_third : Real.sqrt 54 * Real.sqrt (1/3) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_54_times_sqrt_one_third_l720_72098
