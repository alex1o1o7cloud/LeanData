import Mathlib

namespace NUMINAMATH_CALUDE_saras_team_games_l514_51489

/-- The total number of games played by Sara's high school basketball team -/
def total_games (won_games defeated_games : ℕ) : ℕ :=
  won_games + defeated_games

/-- Theorem stating that for Sara's team, the total number of games
    is equal to the sum of won games and defeated games -/
theorem saras_team_games :
  total_games 12 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_saras_team_games_l514_51489


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l514_51435

/-- Given a tetrahedron with vertices A₁, A₂, A₃, A₄ in ℝ³ -/
def A₁ : ℝ × ℝ × ℝ := (-1, -5, 2)
def A₂ : ℝ × ℝ × ℝ := (-6, 0, -3)
def A₃ : ℝ × ℝ × ℝ := (3, 6, -3)
def A₄ : ℝ × ℝ × ℝ := (-10, 6, 7)

/-- Calculate the volume of the tetrahedron -/
def tetrahedron_volume (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedron_height (A₁ A₂ A₃ A₄ : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem tetrahedron_properties :
  tetrahedron_volume A₁ A₂ A₃ A₄ = 190 ∧
  tetrahedron_height A₁ A₂ A₃ A₄ = 2 * Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l514_51435


namespace NUMINAMATH_CALUDE_equal_milk_water_ratio_l514_51413

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- The ratio of two quantities -/
def ratio (a b : ℚ) : ℚ := a / b

/-- Mixture p with milk to water ratio 5:4 -/
def mixture_p : Mixture := { milk := 5, water := 4 }

/-- Mixture q with milk to water ratio 2:7 -/
def mixture_q : Mixture := { milk := 2, water := 7 }

/-- Combine two mixtures in a given ratio -/
def combine_mixtures (m1 m2 : Mixture) (r : ℚ) : Mixture :=
  { milk := m1.milk * r + m2.milk,
    water := m1.water * r + m2.water }

/-- Theorem stating that mixing p and q in ratio 5:1 results in equal milk and water -/
theorem equal_milk_water_ratio :
  let result := combine_mixtures mixture_p mixture_q (5/1)
  ratio result.milk result.water = 1 := by sorry

end NUMINAMATH_CALUDE_equal_milk_water_ratio_l514_51413


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l514_51404

/-- Calculates the net rate of pay for a driver given specific conditions --/
theorem driver_net_pay_rate (hours : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (pay_per_mile : ℝ) (gas_price : ℝ) :
  hours = 3 →
  speed = 50 →
  fuel_efficiency = 25 →
  pay_per_mile = 0.60 →
  gas_price = 2.50 →
  let distance := hours * speed
  let gas_used := distance / fuel_efficiency
  let earnings := distance * pay_per_mile
  let gas_cost := gas_used * gas_price
  let net_earnings := earnings - gas_cost
  net_earnings / hours = 25 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l514_51404


namespace NUMINAMATH_CALUDE_trigonometric_identity_l514_51442

theorem trigonometric_identity (α : Real) : 
  (2 * Real.tan (π / 4 - α)) / (1 - Real.tan (π / 4 - α)^2) * 
  (Real.sin α * Real.cos α) / (Real.cos α^2 - Real.sin α^2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l514_51442


namespace NUMINAMATH_CALUDE_pencil_difference_l514_51417

/-- The number of pencils each person has -/
structure PencilCounts where
  candy : ℕ
  caleb : ℕ
  calen : ℕ

/-- The conditions of the problem -/
def problem_conditions (p : PencilCounts) : Prop :=
  p.candy = 9 ∧
  p.calen = p.caleb + 5 ∧
  p.caleb < 2 * p.candy ∧
  p.calen - 10 = 10

/-- The theorem to be proved -/
theorem pencil_difference (p : PencilCounts) 
  (h : problem_conditions p) : 2 * p.candy - p.caleb = 3 := by
  sorry


end NUMINAMATH_CALUDE_pencil_difference_l514_51417


namespace NUMINAMATH_CALUDE_common_chord_theorem_l514_51490

-- Define the circles
def C1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 3)^2 = 9
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 11 = 0

-- Define the line equation
def common_chord_line (x y : ℝ) : Prop := 3*x - 4*y + 6 = 0

-- Theorem statement
theorem common_chord_theorem :
  ∃ (x y : ℝ), 
    (C1 x y ∧ C2 x y) → 
    (common_chord_line x y ∧ 
     ∃ (x1 y1 x2 y2 : ℝ), 
       C1 x1 y1 ∧ C2 x1 y1 ∧ 
       C1 x2 y2 ∧ C2 x2 y2 ∧ 
       common_chord_line x1 y1 ∧ 
       common_chord_line x2 y2 ∧ 
       ((x2 - x1)^2 + (y2 - y1)^2)^(1/2) = 24/5) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_theorem_l514_51490


namespace NUMINAMATH_CALUDE_apple_cost_price_l514_51476

/-- Proves that given a selling price of 15 and a loss of 1/6th of the cost price, the cost price of the apple is 18. -/
theorem apple_cost_price (selling_price : ℚ) (loss_fraction : ℚ) : 
  selling_price = 15 ∧ loss_fraction = 1/6 → 
  ∃ (cost_price : ℚ), cost_price = 18 ∧ selling_price = cost_price * (1 - loss_fraction) :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_price_l514_51476


namespace NUMINAMATH_CALUDE_dilation_result_l514_51452

/-- Dilation of a complex number -/
def dilation (c k z : ℂ) : ℂ := c + k * (z - c)

theorem dilation_result :
  let c : ℂ := 1 - 3*I
  let k : ℂ := 3
  let z : ℂ := -2 + I
  dilation c k z = -8 + 9*I := by sorry

end NUMINAMATH_CALUDE_dilation_result_l514_51452


namespace NUMINAMATH_CALUDE_expression_equals_two_l514_51405

theorem expression_equals_two (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Real.sqrt (a * b / 2) + Real.sqrt 8) / Real.sqrt ((a * b + 16) / 8 + Real.sqrt (a * b)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_two_l514_51405


namespace NUMINAMATH_CALUDE_direct_proportion_only_f3_l514_51480

/-- A function f: ℝ → ℝ is a direct proportion function if there exists a constant k such that f(x) = k * x for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- Function 1: f(x) = 3x - 4 -/
def f1 : ℝ → ℝ := λ x ↦ 3 * x - 4

/-- Function 2: f(x) = -2x + 1 -/
def f2 : ℝ → ℝ := λ x ↦ -2 * x + 1

/-- Function 3: f(x) = 3x -/
def f3 : ℝ → ℝ := λ x ↦ 3 * x

/-- Function 4: f(x) = 4 -/
def f4 : ℝ → ℝ := λ _ ↦ 4

theorem direct_proportion_only_f3 :
  ¬ is_direct_proportion f1 ∧
  ¬ is_direct_proportion f2 ∧
  is_direct_proportion f3 ∧
  ¬ is_direct_proportion f4 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_only_f3_l514_51480


namespace NUMINAMATH_CALUDE_mollys_age_problem_l514_51402

theorem mollys_age_problem (current_age : ℕ) (years_ahead : ℕ) (multiplier : ℕ) : 
  current_age = 12 →
  years_ahead = 18 →
  multiplier = 5 →
  ∃ (years_ago : ℕ), current_age + years_ahead = multiplier * (current_age - years_ago) ∧ years_ago = 6 := by
  sorry

end NUMINAMATH_CALUDE_mollys_age_problem_l514_51402


namespace NUMINAMATH_CALUDE_inequality_equivalence_l514_51428

theorem inequality_equivalence (x y : ℝ) : 
  (y - 2*x < Real.sqrt (4*x^2 - 4*x + 1)) ↔ 
  ((x < 1/2 ∧ y < 1) ∨ (x ≥ 1/2 ∧ y < 4*x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l514_51428


namespace NUMINAMATH_CALUDE_triangle_ratio_l514_51473

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l514_51473


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l514_51447

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 7*x - 30 = 0 ∧ (∀ y : ℝ, y^2 + 7*y - 30 = 0 → y ≥ x) → x = -10 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l514_51447


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l514_51433

-- Define the arithmetic sequence a_n
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the geometric sequence b_n
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem arithmetic_geometric_sequence_property
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_geometric : geometric_sequence b)
  (h_a_sum : a 1001 + a 1015 = Real.pi)
  (h_b_prod : b 6 * b 9 = 2) :
  Real.tan ((a 1 + a 2015) / (1 + b 7 * b 8)) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_property_l514_51433


namespace NUMINAMATH_CALUDE_ellipse_intersection_midpoints_line_slope_l514_51438

/-- Definition of the ellipse -/
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

/-- Definition of the parallel lines -/
def parallel_line (x y m : ℝ) : Prop := y = (1/4) * x + m

/-- Definition of a point being the midpoint of two other points -/
def is_midpoint (x y x1 y1 x2 y2 : ℝ) : Prop :=
  x = (x1 + x2) / 2 ∧ y = (y1 + y2) / 2

/-- The main theorem -/
theorem ellipse_intersection_midpoints_line_slope :
  ∀ (l : ℝ → ℝ),
  (∀ x y m x1 y1 x2 y2 : ℝ,
    ellipse x1 y1 ∧ ellipse x2 y2 ∧
    parallel_line x1 y1 m ∧ parallel_line x2 y2 m ∧
    is_midpoint x y x1 y1 x2 y2 →
    y = l x) →
  ∃ k, ∀ x, l x = -2 * x + k :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_midpoints_line_slope_l514_51438


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l514_51498

theorem mean_of_added_numbers (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 12 →
  original_list.sum / original_list.length = 75 →
  (original_list.sum + x + y + z) / (original_list.length + 3) = 90 →
  (x + y + z) / 3 = 150 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l514_51498


namespace NUMINAMATH_CALUDE_unit_conversions_l514_51425

/-- Conversion factor from cubic decimeters to cubic meters -/
def cubic_dm_to_m : ℚ := 1 / 1000

/-- Conversion factor from seconds to minutes -/
def sec_to_min : ℚ := 1 / 60

/-- Conversion factor from minutes to hours -/
def min_to_hour : ℚ := 1 / 60

/-- Conversion factor from square centimeters to square decimeters -/
def sq_cm_to_sq_dm : ℚ := 1 / 100

/-- Conversion factor from milliliters to liters -/
def ml_to_l : ℚ := 1 / 1000

/-- Theorem stating the correctness of unit conversions -/
theorem unit_conversions :
  (35 * cubic_dm_to_m = 7 / 200) ∧
  (53 * sec_to_min = 53 / 60) ∧
  (5 * min_to_hour = 1 / 12) ∧
  (1 * sq_cm_to_sq_dm = 1 / 100) ∧
  (450 * ml_to_l = 9 / 20) := by
  sorry

end NUMINAMATH_CALUDE_unit_conversions_l514_51425


namespace NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l514_51493

theorem exponent_of_five_in_30_factorial : 
  ∃ k : ℕ, (30 : ℕ).factorial = 5^7 * k ∧ k % 5 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_exponent_of_five_in_30_factorial_l514_51493


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l514_51419

theorem simplify_nested_roots (a : ℝ) (ha : a > 0) :
  (((a^16)^(1/5))^(1/4))^6 * (((a^16)^(1/4))^(1/5))^6 = a^(48/5) :=
sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l514_51419


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l514_51432

/-- An equilateral triangle ABC divided into 9 smaller equilateral triangles -/
structure TriangleABC where
  /-- The side length of the large equilateral triangle ABC -/
  side : ℝ
  /-- The side length of each smaller equilateral triangle -/
  small_side : ℝ
  /-- The number of smaller triangles that make up triangle ABC -/
  num_small_triangles : ℕ
  /-- The number of smaller triangles that are half shaded -/
  num_half_shaded : ℕ
  /-- Condition: The large triangle is divided into 9 smaller triangles -/
  h_num_small : num_small_triangles = 9
  /-- Condition: Two smaller triangles are half shaded -/
  h_num_half : num_half_shaded = 2
  /-- Condition: The side length of the large triangle is 3 times the small triangle -/
  h_side : side = 3 * small_side

/-- The shaded area is 2/9 of the total area of triangle ABC -/
theorem shaded_area_fraction (t : TriangleABC) : 
  (t.num_half_shaded : ℝ) / 2 / t.num_small_triangles = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l514_51432


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l514_51420

theorem a_equals_one_sufficient_not_necessary :
  (∃ a : ℝ, a = 1 → a^2 = 1) ∧ 
  (∃ a : ℝ, a^2 = 1 ∧ a ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l514_51420


namespace NUMINAMATH_CALUDE_product_a2_a6_l514_51457

def S (n : ℕ) : ℕ := 2^n - 1

def a (n : ℕ) : ℕ := S n - S (n-1)

theorem product_a2_a6 : a 2 * a 6 = 64 := by
  sorry

end NUMINAMATH_CALUDE_product_a2_a6_l514_51457


namespace NUMINAMATH_CALUDE_fourth_term_coefficient_of_binomial_expansion_l514_51466

theorem fourth_term_coefficient_of_binomial_expansion :
  let n : ℕ := 7
  let k : ℕ := 3
  let coef : ℕ := n.choose k * 2^k
  coef = 280 := by sorry

end NUMINAMATH_CALUDE_fourth_term_coefficient_of_binomial_expansion_l514_51466


namespace NUMINAMATH_CALUDE_four_line_corresponding_angles_l514_51488

/-- Represents a line in a plane -/
structure Line

/-- Represents an intersection point of two lines -/
structure IntersectionPoint

/-- Represents a pair of corresponding angles -/
structure CorrespondingAnglePair

/-- A configuration of four lines intersecting pairwise -/
structure FourLineConfiguration where
  lines : Fin 4 → Line
  intersections : Fin 6 → IntersectionPoint
  no_triple_intersection : ∀ (i j k : Fin 4), i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ∃ (p q : IntersectionPoint), p ≠ q ∧ 
    (p ∈ (Set.range intersections) ∧ q ∈ (Set.range intersections))

/-- The number of corresponding angle pairs in a four-line configuration -/
def num_corresponding_angles (config : FourLineConfiguration) : ℕ :=
  48

/-- Theorem stating that a four-line configuration has 48 pairs of corresponding angles -/
theorem four_line_corresponding_angles (config : FourLineConfiguration) :
  num_corresponding_angles config = 48 := by sorry

end NUMINAMATH_CALUDE_four_line_corresponding_angles_l514_51488


namespace NUMINAMATH_CALUDE_g_four_to_four_l514_51450

/-- Given two functions f and g satisfying certain conditions, prove that [g(4)]^4 = 16 -/
theorem g_four_to_four (f g : ℝ → ℝ) 
  (h1 : ∀ x ≥ 1, f (g x) = x^2)
  (h2 : ∀ x ≥ 1, g (f x) = x^4)
  (h3 : g 16 = 16) : 
  (g 4)^4 = 16 := by
sorry

end NUMINAMATH_CALUDE_g_four_to_four_l514_51450


namespace NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l514_51478

theorem hyperbola_midpoint_existence :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (x₁^2 - y₁^2/9 = 1) ∧
    (x₂^2 - y₂^2/9 = 1) ∧
    ((x₁ + x₂)/2 = -1) ∧
    ((y₁ + y₂)/2 = -4) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_midpoint_existence_l514_51478


namespace NUMINAMATH_CALUDE_max_area_four_squares_l514_51475

/-- The maximum area covered by 4 squares with side length 2 when arranged to form a larger square -/
theorem max_area_four_squares (n : ℕ) (side_length : ℝ) (h1 : n = 4) (h2 : side_length = 2) :
  n * side_length^2 - (n - 1) = 13 :=
sorry

end NUMINAMATH_CALUDE_max_area_four_squares_l514_51475


namespace NUMINAMATH_CALUDE_scores_mode_is_80_l514_51481

def scores : List Nat := [70, 80, 100, 60, 80, 70, 90, 50, 80, 70, 80, 70, 90, 80, 90, 80, 70, 90, 60, 80]

def mode (l : List α) [DecidableEq α] : Option α :=
  l.foldl (fun acc x =>
    match acc with
    | none => some x
    | some y => if l.count x > l.count y then some x else some y
  ) none

theorem scores_mode_is_80 : mode scores = some 80 := by
  sorry

end NUMINAMATH_CALUDE_scores_mode_is_80_l514_51481


namespace NUMINAMATH_CALUDE_curve_C_symmetry_l514_51441

/-- The curve C is defined by the equation x^2*y + x*y^2 = 1 --/
def C (x y : ℝ) : Prop := x^2*y + x*y^2 = 1

/-- A point (x, y) is symmetric to (a, b) with respect to the line y=x --/
def symmetric_y_eq_x (x y a b : ℝ) : Prop := x = b ∧ y = a

theorem curve_C_symmetry :
  (∀ x y : ℝ, C x y → C y x) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C x (-y)) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-x) y) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-x) (-y)) ∧ 
  (∃ x y : ℝ, C x y ∧ ¬C (-y) (-x)) :=
sorry

end NUMINAMATH_CALUDE_curve_C_symmetry_l514_51441


namespace NUMINAMATH_CALUDE_triangle_problem_l514_51459

noncomputable section

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Vector m in the problem -/
def m (t : Triangle) : ℝ × ℝ := (Real.cos t.A, Real.sin t.A)

/-- Vector n in the problem -/
def n (t : Triangle) : ℝ × ℝ := (Real.cos t.A, -Real.sin t.A)

/-- Dot product of vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- The theorem to be proved -/
theorem triangle_problem (t : Triangle) 
    (h_acute : 0 < t.A ∧ t.A < π / 2)
    (h_dot : dot_product (m t) (n t) = 1 / 2)
    (h_a : t.a = Real.sqrt 5) :
  t.A = π / 6 ∧ 
  Real.arccos (dot_product (m t) (n t) / (Real.sqrt ((m t).1^2 + (m t).2^2) * Real.sqrt ((n t).1^2 + (n t).2^2))) = π / 3 ∧
  (let max_area := (10 + 5 * Real.sqrt 3) / 4
   ∀ b c, t.b = b → t.c = c → 1 / 2 * b * c * Real.sin t.A ≤ max_area) :=
sorry

end NUMINAMATH_CALUDE_triangle_problem_l514_51459


namespace NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l514_51415

/-- Given a parabola y = ax^2 + bx, prove that after reflecting about the y-axis
    and translating one parabola 4 units right and the other 4 units left,
    the sum of the resulting parabolas' equations is y = 2ax^2 - 8b. -/
theorem parabola_reflection_translation_sum (a b : ℝ) :
  let f (x : ℝ) := a * x^2 + b * (x - 4)
  let g (x : ℝ) := a * x^2 - b * (x + 4)
  ∀ x, (f + g) x = 2 * a * x^2 - 8 * b :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l514_51415


namespace NUMINAMATH_CALUDE_smallest_consecutive_number_l514_51429

theorem smallest_consecutive_number (x : ℕ) : 
  (∃ (a b c d : ℕ), x + a + b + c + d = 225 ∧ 
   a = x + 1 ∧ b = x + 2 ∧ c = x + 3 ∧ d = x + 4 ∧
   ∃ (k : ℕ), x = 7 * k) → 
  x = 42 := by
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_number_l514_51429


namespace NUMINAMATH_CALUDE_expensive_candy_price_l514_51494

/-- Given a mixture of two types of candy, prove the price of the more expensive candy. -/
theorem expensive_candy_price
  (total_weight : ℝ)
  (mixture_price : ℝ)
  (cheap_price : ℝ)
  (cheap_weight : ℝ)
  (h1 : total_weight = 80)
  (h2 : mixture_price = 2.20)
  (h3 : cheap_price = 2)
  (h4 : cheap_weight = 64) :
  ∃ (expensive_price : ℝ), expensive_price = 3 := by
sorry

end NUMINAMATH_CALUDE_expensive_candy_price_l514_51494


namespace NUMINAMATH_CALUDE_total_items_l514_51424

def num_children : ℕ := 12
def pencils_per_child : ℕ := 5
def erasers_per_child : ℕ := 3
def skittles_per_child : ℕ := 13
def crayons_per_child : ℕ := 7

theorem total_items :
  num_children * pencils_per_child = 60 ∧
  num_children * erasers_per_child = 36 ∧
  num_children * skittles_per_child = 156 ∧
  num_children * crayons_per_child = 84 := by
  sorry

end NUMINAMATH_CALUDE_total_items_l514_51424


namespace NUMINAMATH_CALUDE_ali_baba_max_coins_l514_51437

/-- Represents the state of the coin distribution game -/
structure GameState :=
  (piles : List Nat)
  (total_coins : Nat)

/-- Represents a move in the game -/
structure Move :=
  (chosen_piles : List Nat)
  (coins_removed : List Nat)

/-- Ali Baba's strategy -/
def aliBabaStrategy (state : GameState) : Move :=
  sorry

/-- Thief's strategy -/
def thiefStrategy (state : GameState) (move : Move) : List Nat :=
  sorry

/-- Simulate one round of the game -/
def playRound (state : GameState) : GameState :=
  sorry

/-- Check if the game should end -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Calculate Ali Baba's final score -/
def calculateScore (state : GameState) : Nat :=
  sorry

/-- Main theorem: Ali Baba can secure at most 72 coins -/
theorem ali_baba_max_coins :
  ∀ (initial_state : GameState),
    initial_state.total_coins = 100 ∧ 
    initial_state.piles.length = 10 ∧ 
    (∀ pile ∈ initial_state.piles, pile = 10) →
    calculateScore (playRound initial_state) ≤ 72 :=
  sorry

end NUMINAMATH_CALUDE_ali_baba_max_coins_l514_51437


namespace NUMINAMATH_CALUDE_sisters_age_when_kolya_was_her_current_age_l514_51410

/- Define the current ages of the brother, sister, and Kolya -/
variable (x y k : ℕ)

/- Define the time differences -/
variable (t₁ t₂ : ℕ)

/- First condition: When Kolya was as old as they both are now, the sister was as old as the brother is now -/
axiom condition1 : k - t₁ = x + y ∧ y - t₁ = x

/- Second condition: When Kolya was as old as the sister is now, the sister's age was to be determined -/
axiom condition2 : k - t₂ = y

/- The theorem to prove -/
theorem sisters_age_when_kolya_was_her_current_age : y - t₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_sisters_age_when_kolya_was_her_current_age_l514_51410


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l514_51444

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (digits : List Bool) : Nat :=
  digits.foldl (fun acc d => 2 * acc + if d then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

theorem binary_multiplication_theorem :
  let a := [true, false, true, true, false, true, true]  -- 1101101₂
  let b := [true, true, false, true]                     -- 1011₂
  let product := [true, true, true, true, false, false, true, false, false, false, true]  -- 10001001111₂
  binary_to_nat a * binary_to_nat b = binary_to_nat product := by
  sorry

#eval binary_to_nat [true, false, true, true, false, true, true]  -- Should output 109
#eval binary_to_nat [true, true, false, true]  -- Should output 11
#eval binary_to_nat [true, true, true, true, false, false, true, false, false, false, true]  -- Should output 1103

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l514_51444


namespace NUMINAMATH_CALUDE_pollen_grain_diameter_scientific_notation_l514_51477

theorem pollen_grain_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.0000065 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 6.5 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_pollen_grain_diameter_scientific_notation_l514_51477


namespace NUMINAMATH_CALUDE_odd_power_minus_self_div_24_l514_51439

theorem odd_power_minus_self_div_24 (n : ℕ) (h : Odd n) (h' : n > 0) :
  ∃ k : ℤ, (n^n : ℤ) - n = 24 * k := by
  sorry

end NUMINAMATH_CALUDE_odd_power_minus_self_div_24_l514_51439


namespace NUMINAMATH_CALUDE_no_real_solutions_l514_51455

theorem no_real_solutions (k : ℝ) : 
  (∀ x : ℝ, x^2 ≠ 5*x + k) ↔ k < -25/4 := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l514_51455


namespace NUMINAMATH_CALUDE_largest_N_equals_n_l514_51451

theorem largest_N_equals_n (n : ℕ) (hn : n ≥ 2) :
  ∃ N : ℕ, N > 0 ∧
  (∀ M : ℕ, M > N →
    ¬∃ (a : ℕ → ℝ), a 0 + a 1 = -1 / n ∧
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ M - 1 →
      (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) ∧
  (∃ (a : ℕ → ℝ), a 0 + a 1 = -1 / n ∧
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ N - 1 →
      (a k + a (k - 1)) * (a k + a (k + 1)) = a (k - 1) - a (k + 1)) ∧
  N = n :=
sorry

end NUMINAMATH_CALUDE_largest_N_equals_n_l514_51451


namespace NUMINAMATH_CALUDE_special_triangle_properties_l514_51499

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Properties
  angle_sum : A + B + C = π
  side_angle_relation : (3 * b - c) * Real.cos A - a * Real.cos C = 0
  side_a_value : a = 2 * Real.sqrt 3
  area : 1 / 2 * b * c * Real.sin A = 3 * Real.sqrt 2
  angle_product : Real.sin B * Real.sin C = 2 / 3

/-- Theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  Real.cos t.A = 1 / 3 ∧
  t.b = 3 ∧ t.c = 3 ∧
  Real.tan t.A + Real.tan t.B + Real.tan t.C = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l514_51499


namespace NUMINAMATH_CALUDE_unique_solution_condition_l514_51454

theorem unique_solution_condition (a b c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l514_51454


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l514_51443

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary
  a / b = 3 / 2 →  -- The ratio of the angles is 3:2
  b = 36 :=  -- The smaller angle is 36°
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l514_51443


namespace NUMINAMATH_CALUDE_no_integer_solution_l514_51440

theorem no_integer_solution : ∀ x y : ℤ, x^2 + 3*x*y - 2*y^2 ≠ 122 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l514_51440


namespace NUMINAMATH_CALUDE_problem_solution_l514_51426

theorem problem_solution : ∃ x : ℝ, 0.75 * x = x / 3 + 110 ∧ x = 264 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l514_51426


namespace NUMINAMATH_CALUDE_commission_calculation_l514_51497

def base_salary : ℚ := 370
def past_incomes : List ℚ := [406, 413, 420, 436, 395]
def desired_average : ℚ := 500
def total_weeks : ℕ := 7
def past_weeks : ℕ := 5

theorem commission_calculation (base_salary : ℚ) (past_incomes : List ℚ) 
  (desired_average : ℚ) (total_weeks : ℕ) (past_weeks : ℕ) :
  (desired_average * total_weeks - past_incomes.sum - base_salary * total_weeks) / (total_weeks - past_weeks) = 345 :=
by sorry

end NUMINAMATH_CALUDE_commission_calculation_l514_51497


namespace NUMINAMATH_CALUDE_point_guard_footage_l514_51430

/-- Represents the number of seconds in a minute -/
def seconds_per_minute : ℕ := 60

/-- Represents the number of players on the basketball team -/
def num_players : ℕ := 5

/-- Represents the average number of minutes each player should get in the highlight film -/
def avg_minutes_per_player : ℕ := 2

/-- Represents the total seconds of footage for the shooting guard, small forward, power forward, and center -/
def other_players_footage : ℕ := 470

/-- Theorem stating that the point guard's footage is 130 seconds -/
theorem point_guard_footage : 
  (num_players * avg_minutes_per_player * seconds_per_minute) - other_players_footage = 130 := by
sorry

end NUMINAMATH_CALUDE_point_guard_footage_l514_51430


namespace NUMINAMATH_CALUDE_no_extreme_points_iff_m_in_range_l514_51423

/-- A cubic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + x + 2023

/-- The derivative of f with respect to x -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + 1

/-- Predicate for f having no extreme points -/
def has_no_extreme_points (m : ℝ) : Prop :=
  ∀ x : ℝ, f_derivative m x ≠ 0 ∨ 
    (∀ y : ℝ, y < x → f_derivative m y > 0) ∧ 
    (∀ y : ℝ, y > x → f_derivative m y > 0)

theorem no_extreme_points_iff_m_in_range :
  ∀ m : ℝ, has_no_extreme_points m ↔ -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_no_extreme_points_iff_m_in_range_l514_51423


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l514_51474

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) (h : r = 3 / 2) : 
  (π * r^2) / 2 = 9 * π / 8 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l514_51474


namespace NUMINAMATH_CALUDE_harmonic_series_inequality_l514_51464

/-- The harmonic series function -/
def f (n : ℕ) : ℚ := (Finset.range n).sum (fun i => 1 / (i + 1 : ℚ))

/-- The main theorem: f(2^n) > (n+2)/2 for all n ≥ 1 -/
theorem harmonic_series_inequality (n : ℕ) (h : n ≥ 1) : f (2^n) > (n + 2 : ℚ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_series_inequality_l514_51464


namespace NUMINAMATH_CALUDE_population_equal_in_14_years_l514_51468

/-- The number of years it takes for two villages' populations to be equal -/
def years_to_equal_population (initial_x initial_y decline_rate_x growth_rate_y : ℕ) : ℕ :=
  (initial_x - initial_y) / (decline_rate_x + growth_rate_y)

/-- Theorem stating that it takes 14 years for the populations to be equal -/
theorem population_equal_in_14_years :
  years_to_equal_population 70000 42000 1200 800 = 14 := by
  sorry

#eval years_to_equal_population 70000 42000 1200 800

end NUMINAMATH_CALUDE_population_equal_in_14_years_l514_51468


namespace NUMINAMATH_CALUDE_perpendicular_lines_relationship_l514_51449

-- Define a type for lines in 3D space
def Line3D := ℝ × ℝ × ℝ → Prop

-- Define perpendicularity of lines
def perpendicular (l₁ l₂ : Line3D) : Prop := sorry

-- Define parallel lines
def parallel (l₁ l₂ : Line3D) : Prop := sorry

-- Define skew lines
def skew (l₁ l₂ : Line3D) : Prop := sorry

theorem perpendicular_lines_relationship (a b c : Line3D) 
  (h1 : perpendicular a b) (h2 : perpendicular b c) :
  ¬ (parallel a c ∨ perpendicular a c ∨ skew a c) → False := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_relationship_l514_51449


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l514_51418

theorem smallest_number_satisfying_conditions : ∃! x : ℕ, 
  (∀ y : ℕ, y < x → ¬((y + 3).ModEq 0 7 ∧ (y - 5).ModEq 0 8)) ∧
  (x + 3).ModEq 0 7 ∧ 
  (x - 5).ModEq 0 8 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l514_51418


namespace NUMINAMATH_CALUDE_sequence_identity_l514_51422

def StrictlyIncreasing (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem sequence_identity (a : ℕ → ℕ) 
  (h_increasing : StrictlyIncreasing a)
  (h_upper_bound : ∀ n : ℕ, a n ≤ n + 2020)
  (h_divisibility : ∀ n : ℕ, (a (n + 1)) ∣ (n^3 * (a n) - 1)) :
  ∀ n : ℕ, a n = n :=
sorry

end NUMINAMATH_CALUDE_sequence_identity_l514_51422


namespace NUMINAMATH_CALUDE_distinctFourDigitNumbers_eq_360_l514_51483

/-- The number of distinct four-digit numbers that can be formed using the digits 1, 2, 3, 4, 5,
    where exactly one digit repeats once. -/
def distinctFourDigitNumbers : ℕ :=
  let digits : Fin 5 := 5
  let positionsForRepeatedDigit : ℕ := Nat.choose 4 2
  let remainingDigitChoices : ℕ := 4 * 3
  digits * positionsForRepeatedDigit * remainingDigitChoices

/-- Theorem stating that the number of distinct four-digit numbers under the given conditions is 360. -/
theorem distinctFourDigitNumbers_eq_360 : distinctFourDigitNumbers = 360 := by
  sorry

end NUMINAMATH_CALUDE_distinctFourDigitNumbers_eq_360_l514_51483


namespace NUMINAMATH_CALUDE_negative_two_in_M_l514_51421

def M : Set ℝ := {x | x^2 - 4 = 0}

theorem negative_two_in_M : -2 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_negative_two_in_M_l514_51421


namespace NUMINAMATH_CALUDE_largest_n_for_binomial_equality_l514_51486

theorem largest_n_for_binomial_equality : 
  (∃ n : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n)) ∧ 
  (∀ m : ℕ, (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m) → m ≤ 6) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_binomial_equality_l514_51486


namespace NUMINAMATH_CALUDE_pizza_slices_theorem_l514_51465

/-- Given a number of pizzas and slices per pizza, calculate the total number of slices -/
def total_slices (num_pizzas : ℕ) (slices_per_pizza : ℕ) : ℕ :=
  num_pizzas * slices_per_pizza

/-- Theorem: With 14 pizzas and 2 slices per pizza, the total number of slices is 28 -/
theorem pizza_slices_theorem : total_slices 14 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_theorem_l514_51465


namespace NUMINAMATH_CALUDE_function_condition_implies_a_range_l514_51434

/-- Given a function f and a positive real number a, proves that if the given condition holds, then a ≥ 1 -/
theorem function_condition_implies_a_range (a : ℝ) (h_a : a > 0) :
  (∀ (x : ℝ), x > 0 → ∃ (f : ℝ → ℝ), f x = a * Real.log x + (1/2) * x^2) →
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → 
    (a * Real.log x₁ + (1/2) * x₁^2 - (a * Real.log x₂ + (1/2) * x₂^2)) / (x₁ - x₂) ≥ 2) →
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_condition_implies_a_range_l514_51434


namespace NUMINAMATH_CALUDE_square_difference_l514_51460

theorem square_difference (n : ℕ) (h : n = 50) : n^2 - (n-1)^2 = 2*n - 1 := by
  sorry

#check square_difference

end NUMINAMATH_CALUDE_square_difference_l514_51460


namespace NUMINAMATH_CALUDE_parabola_intersection_fixed_points_l514_51467

/-- The parabola y^2 = 2px -/
def Parabola (p : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.2^2 = 2 * p * xy.1}

/-- The fixed point A -/
def A (t : ℝ) : ℝ × ℝ := (t, 0)

/-- The line x = -t -/
def VerticalLine (t : ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | xy.1 = -t}

/-- The circle with diameter MN -/
def CircleMN (M N : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {xy : ℝ × ℝ | (xy.1 - (M.1 + N.1) / 2)^2 + (xy.2 - (M.2 + N.2) / 2)^2 = 
    ((N.1 - M.1)^2 + (N.2 - M.2)^2) / 4}

theorem parabola_intersection_fixed_points (p t : ℝ) (hp : p > 0) (ht : t > 0) :
  ∀ (B C M N : ℝ × ℝ),
    B ∈ Parabola p → C ∈ Parabola p →
    (∃ (k : ℝ), B.1 = k * B.2 + t ∧ C.1 = k * C.2 + t) →
    M ∈ VerticalLine t → N ∈ VerticalLine t →
    (∃ (r : ℝ), M.2 = r * M.1 ∧ B.2 = r * B.1) →
    (∃ (s : ℝ), N.2 = s * N.1 ∧ C.2 = s * C.1) →
    ((-t - Real.sqrt (2 * p * t), 0) ∈ CircleMN M N) ∧
    ((-t + Real.sqrt (2 * p * t), 0) ∈ CircleMN M N) := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_fixed_points_l514_51467


namespace NUMINAMATH_CALUDE_audio_channel_bandwidth_l514_51406

/-- Represents the parameters for an audio channel --/
structure AudioChannelParams where
  session_duration : ℕ  -- in minutes
  sampling_rate : ℕ     -- in Hz
  bit_depth : ℕ         -- in bits
  metadata_size : ℕ     -- in bytes
  metadata_per : ℕ      -- in kilobits of audio
  is_stereo : Bool

/-- Calculates the required bandwidth for an audio channel --/
def calculate_bandwidth (params : AudioChannelParams) : ℝ :=
  sorry

/-- Theorem stating the required bandwidth for the given audio channel parameters --/
theorem audio_channel_bandwidth 
  (params : AudioChannelParams)
  (h1 : params.session_duration = 51)
  (h2 : params.sampling_rate = 63)
  (h3 : params.bit_depth = 17)
  (h4 : params.metadata_size = 47)
  (h5 : params.metadata_per = 5)
  (h6 : params.is_stereo = true) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (calculate_bandwidth params - 2.25) < ε :=
sorry

end NUMINAMATH_CALUDE_audio_channel_bandwidth_l514_51406


namespace NUMINAMATH_CALUDE_three_million_squared_l514_51456

theorem three_million_squared :
  (3000000 : ℕ) * 3000000 = 9000000000000 := by
  sorry

end NUMINAMATH_CALUDE_three_million_squared_l514_51456


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l514_51409

/-- The value of p for which a circle (x-1)^2 + y^2 = 4 is tangent to the directrix of a parabola y^2 = 2px -/
theorem circle_tangent_to_parabola_directrix (p : ℝ) : 
  p > 0 → 
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = 4 ∧ y^2 = 2*p*x) →
  (∀ (x y : ℝ), (x - 1)^2 + y^2 = 4 → x ≥ -p/2) →
  (∃ (x y : ℝ), (x - 1)^2 + y^2 = 4 ∧ x = -p/2) →
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l514_51409


namespace NUMINAMATH_CALUDE_average_study_time_difference_l514_51412

-- Define the list of daily differences
def daily_differences : List Int := [15, -5, 25, 0, -15, 10, 20]

-- Define the number of days
def num_days : Nat := 7

-- Theorem to prove
theorem average_study_time_difference :
  (daily_differences.sum : ℚ) / num_days = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l514_51412


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l514_51461

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, x^2 + 6*x*y + 9*y^2 ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l514_51461


namespace NUMINAMATH_CALUDE_laura_charge_account_l514_51401

/-- Represents the simple interest calculation for a charge account -/
def simple_interest_charge_account (principal : ℝ) (interest_rate : ℝ) (time : ℝ) (total_owed : ℝ) : Prop :=
  total_owed = principal + (principal * interest_rate * time)

theorem laura_charge_account :
  ∀ (principal : ℝ),
    simple_interest_charge_account principal 0.05 1 36.75 →
    principal = 35 := by
  sorry

end NUMINAMATH_CALUDE_laura_charge_account_l514_51401


namespace NUMINAMATH_CALUDE_total_cleaner_needed_l514_51400

def cleaner_per_dog : ℕ := 6
def cleaner_per_cat : ℕ := 4
def cleaner_per_rabbit : ℕ := 1

def num_dogs : ℕ := 6
def num_cats : ℕ := 3
def num_rabbits : ℕ := 1

theorem total_cleaner_needed :
  cleaner_per_dog * num_dogs + cleaner_per_cat * num_cats + cleaner_per_rabbit * num_rabbits = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cleaner_needed_l514_51400


namespace NUMINAMATH_CALUDE_double_plus_five_l514_51416

theorem double_plus_five (x : ℝ) (h : x = 6) : 2 * x + 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_double_plus_five_l514_51416


namespace NUMINAMATH_CALUDE_trig_identity_l514_51403

theorem trig_identity (x : Real) (h : Real.sin (x + π/6) = 1/4) :
  Real.sin ((5*π)/6 - x) + (Real.cos ((π/3) - x))^2 = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l514_51403


namespace NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_l514_51487

theorem two_numbers_with_sum_and_gcd : ∃ (a b : ℕ), a + b = 168 ∧ Nat.gcd a b = 24 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_sum_and_gcd_l514_51487


namespace NUMINAMATH_CALUDE_pencil_discount_l514_51484

theorem pencil_discount (original_cost final_price : ℝ) 
  (h1 : original_cost = 4)
  (h2 : final_price = 3.37) : 
  original_cost - final_price = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_pencil_discount_l514_51484


namespace NUMINAMATH_CALUDE_todd_money_left_l514_51463

def initial_amount : ℕ := 20
def candy_bars : ℕ := 4
def cost_per_bar : ℕ := 2

theorem todd_money_left : 
  initial_amount - (candy_bars * cost_per_bar) = 12 := by sorry

end NUMINAMATH_CALUDE_todd_money_left_l514_51463


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l514_51448

theorem absolute_value_inequality (x : ℝ) :
  |((x^2 - 5*x + 4) / 3)| < 1 ↔ (5 - Real.sqrt 21) / 2 < x ∧ x < (5 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l514_51448


namespace NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l514_51479

theorem largest_two_digit_prime_factor_of_binomial : 
  ∃ (p : ℕ), p.Prime ∧ 10 ≤ p ∧ p < 100 ∧ p ∣ Nat.choose 300 150 ∧
  ∀ (q : ℕ), q.Prime → 10 ≤ q → q < 100 → q ∣ Nat.choose 300 150 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_prime_factor_of_binomial_l514_51479


namespace NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l514_51458

theorem smallest_common_multiple_9_6 : ∃ n : ℕ+, (∀ m : ℕ+, 9 ∣ m ∧ 6 ∣ m → n ≤ m) ∧ 9 ∣ n ∧ 6 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_9_6_l514_51458


namespace NUMINAMATH_CALUDE_unsold_books_percentage_l514_51470

/-- Calculates the percentage of unsold books in a bookshop -/
theorem unsold_books_percentage 
  (initial_stock : ℕ) 
  (monday_sales tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ) : 
  initial_stock = 700 →
  monday_sales = 50 →
  tuesday_sales = 82 →
  wednesday_sales = 60 →
  thursday_sales = 48 →
  friday_sales = 40 →
  (initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_percentage_l514_51470


namespace NUMINAMATH_CALUDE_line_segment_ratio_l514_51453

/-- Given seven points O, A, B, C, D, E, F on a straight line, with P on CD,
    prove that OP = (3a + 2d) / 5 when AP:PD = 2:3 and BP:PC = 3:4 -/
theorem line_segment_ratio (a b c d e f : ℝ) :
  let O : ℝ := 0
  let A : ℝ := a
  let B : ℝ := b
  let C : ℝ := c
  let D : ℝ := d
  let E : ℝ := e
  let F : ℝ := f
  ∀ P : ℝ,
    c ≤ P ∧ P ≤ d →
    (A - P) / (P - D) = 2 / 3 →
    (B - P) / (P - C) = 3 / 4 →
    P = (3 * a + 2 * d) / 5 :=
by sorry

end NUMINAMATH_CALUDE_line_segment_ratio_l514_51453


namespace NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l514_51496

theorem unique_solution_for_diophantine_equation :
  ∃! (m : ℕ+) (p q : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ 
    2^(m:ℕ) * p^2 + 1 = q^5 ∧
    m = 1 ∧ p = 11 ∧ q = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_diophantine_equation_l514_51496


namespace NUMINAMATH_CALUDE_power_multiplication_simplification_l514_51485

theorem power_multiplication_simplification (x : ℝ) : (x^5 * x^3) * x^2 = x^10 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_simplification_l514_51485


namespace NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l514_51469

theorem sqrt_15_minus_1_range : 2 < Real.sqrt 15 - 1 ∧ Real.sqrt 15 - 1 < 3 := by
  have h1 : 9 < 15 := by sorry
  have h2 : 15 < 16 := by sorry
  sorry

end NUMINAMATH_CALUDE_sqrt_15_minus_1_range_l514_51469


namespace NUMINAMATH_CALUDE_milk_level_lowering_l514_51491

/-- Proves that lowering the milk level by 6 inches in a 50 feet by 25 feet rectangular box removes 4687.5 gallons of milk, given that 1 cubic foot equals 7.5 gallons. -/
theorem milk_level_lowering (box_length : Real) (box_width : Real) (gallons_removed : Real) (cubic_foot_to_gallon : Real) (inches_lowered : Real) : 
  box_length = 50 ∧ 
  box_width = 25 ∧ 
  gallons_removed = 4687.5 ∧ 
  cubic_foot_to_gallon = 7.5 ∧
  inches_lowered = 6 → 
  gallons_removed = (box_length * box_width * (inches_lowered / 12)) * cubic_foot_to_gallon :=
by sorry

end NUMINAMATH_CALUDE_milk_level_lowering_l514_51491


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l514_51445

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 210 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l514_51445


namespace NUMINAMATH_CALUDE_paperback_ratio_l514_51407

/-- Represents the number and types of books Thabo owns -/
structure BookCollection where
  total : ℕ
  paperback_fiction : ℕ
  paperback_nonfiction : ℕ
  hardcover_nonfiction : ℕ

/-- The properties of Thabo's book collection -/
def thabos_books : BookCollection where
  total := 220
  paperback_fiction := 120
  paperback_nonfiction := 60
  hardcover_nonfiction := 40

/-- Theorem stating the ratio of paperback fiction to paperback nonfiction books -/
theorem paperback_ratio (b : BookCollection) 
  (h1 : b.total = 220)
  (h2 : b.paperback_fiction + b.paperback_nonfiction + b.hardcover_nonfiction = b.total)
  (h3 : b.paperback_nonfiction = b.hardcover_nonfiction + 20)
  (h4 : b.hardcover_nonfiction = 40) :
  b.paperback_fiction / b.paperback_nonfiction = 2 := by
  sorry

#check paperback_ratio thabos_books

end NUMINAMATH_CALUDE_paperback_ratio_l514_51407


namespace NUMINAMATH_CALUDE_misha_earnings_l514_51431

theorem misha_earnings (current_amount target_amount : ℕ) 
  (h1 : current_amount = 34) 
  (h2 : target_amount = 47) : 
  target_amount - current_amount = 13 := by
  sorry

end NUMINAMATH_CALUDE_misha_earnings_l514_51431


namespace NUMINAMATH_CALUDE_product_96_104_l514_51408

theorem product_96_104 : 96 * 104 = 9984 := by
  sorry

end NUMINAMATH_CALUDE_product_96_104_l514_51408


namespace NUMINAMATH_CALUDE_f_geq_g_l514_51427

/-- Given positive real numbers a, b, c, and a real number α, 
    we define functions f and g as follows:
    f(α) = abc(a^α + b^α + c^α)
    g(α) = a^(α+2)(b+c-a) + b^(α+2)(a-b+c) + c^(α+2)(a+b-c)
    This theorem states that f(α) ≥ g(α) for all real α. -/
theorem f_geq_g (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let f := fun (α : ℝ) ↦ a * b * c * (a^α + b^α + c^α)
  let g := fun (α : ℝ) ↦ a^(α+2)*(b+c-a) + b^(α+2)*(a-b+c) + c^(α+2)*(a+b-c)
  ∀ α, f α ≥ g α :=
by sorry

end NUMINAMATH_CALUDE_f_geq_g_l514_51427


namespace NUMINAMATH_CALUDE_batsman_average_l514_51436

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) :
  total_innings = 25 →
  last_innings_score = 175 →
  average_increase = 6 →
  (∃ (previous_average : ℕ),
    (previous_average * (total_innings - 1) + last_innings_score) / total_innings =
    previous_average + average_increase) →
  (((total_innings - 1) * ((last_innings_score / average_increase) - total_innings) +
    last_innings_score) / total_innings) = 31 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_l514_51436


namespace NUMINAMATH_CALUDE_forest_leaves_count_l514_51495

theorem forest_leaves_count :
  let trees_in_forest : ℕ := 20
  let main_branches_per_tree : ℕ := 15
  let sub_branches_per_main : ℕ := 25
  let tertiary_branches_per_sub : ℕ := 30
  let leaves_per_sub : ℕ := 75
  let leaves_per_tertiary : ℕ := 45

  let total_leaves : ℕ := 
    trees_in_forest * 
    (main_branches_per_tree * sub_branches_per_main * leaves_per_sub +
     main_branches_per_tree * sub_branches_per_main * tertiary_branches_per_sub * leaves_per_tertiary)

  total_leaves = 10687500 := by
sorry

end NUMINAMATH_CALUDE_forest_leaves_count_l514_51495


namespace NUMINAMATH_CALUDE_highest_point_parabola_l514_51446

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := -2 * x^2 + 28 * x + 418

/-- The x-coordinate of the vertex of the parabola -/
def vertex_x : ℝ := 7

/-- The y-coordinate of the vertex of the parabola -/
def vertex_y : ℝ := parabola vertex_x

theorem highest_point_parabola :
  ∀ x : ℝ, parabola x ≤ vertex_y :=
by sorry

end NUMINAMATH_CALUDE_highest_point_parabola_l514_51446


namespace NUMINAMATH_CALUDE_log_equation_proof_l514_51414

theorem log_equation_proof : -2 * Real.log 10 / Real.log 5 - Real.log 0.25 / Real.log 5 + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_proof_l514_51414


namespace NUMINAMATH_CALUDE_square_perimeter_square_perimeter_holds_l514_51471

/-- The perimeter of a square with side length 7 meters is 28 meters. -/
theorem square_perimeter : ℝ → Prop :=
  fun side_length =>
    side_length = 7 → 4 * side_length = 28

/-- The theorem holds for the given side length. -/
theorem square_perimeter_holds : square_perimeter 7 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_square_perimeter_holds_l514_51471


namespace NUMINAMATH_CALUDE_rotated_logarithm_function_l514_51492

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the rotation transformation
def rotate_counterclockwise_pi_over_2 (x y : ℝ) : ℝ × ℝ := (y, -x)

-- State the theorem
theorem rotated_logarithm_function (f : ℝ → ℝ) :
  (∀ x, rotate_counterclockwise_pi_over_2 (f x) x = (lg (x + 1), x)) →
  (∀ x, f x = 10^(-x) - 1) :=
by sorry

end NUMINAMATH_CALUDE_rotated_logarithm_function_l514_51492


namespace NUMINAMATH_CALUDE_f_is_odd_l514_51462

-- Define the property of being an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the property of a function not being identically zero
def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

-- Main theorem
theorem f_is_odd 
  (f : ℝ → ℝ) 
  (h1 : IsEven (fun x => (x^3 - 2*x) * f x))
  (h2 : NotIdenticallyZero f) : 
  IsOdd f := by
  sorry


end NUMINAMATH_CALUDE_f_is_odd_l514_51462


namespace NUMINAMATH_CALUDE_trigonometric_values_signs_l514_51472

theorem trigonometric_values_signs :
  (∃ x, x = Real.sin (-1000 * π / 180) ∧ x > 0) ∧
  (∃ y, y = Real.cos (-2200 * π / 180) ∧ y > 0) ∧
  (∃ z, z = Real.tan (-10) ∧ z < 0) ∧
  (∃ w, w = (Real.sin (7 * π / 10) * Real.cos π) / Real.tan (17 * π / 9) ∧ w > 0) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_values_signs_l514_51472


namespace NUMINAMATH_CALUDE_parabola_tangent_to_circle_l514_51482

/-- Given a parabola and a circle, if the parabola's axis is tangent to the circle, 
    then the parameter p of the parabola equals 2. -/
theorem parabola_tangent_to_circle (p : ℝ) (h1 : p > 0) : 
  (∀ x y : ℝ, y^2 = 2*p*x) →  -- parabola equation
  (∀ x y : ℝ, x^2 + y^2 - 8*x - 9 = 0) →  -- circle equation
  (∃ x : ℝ, x = -p/2 ∧ (x-4)^2 = 25) →  -- parabola's axis is tangent to the circle
  p = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_circle_l514_51482


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l514_51411

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- all terms are positive
  (∃ d, ∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  a 1 + a 2 = 1 →
  a 3 + a 4 = 4 →
  a 4 + a 5 = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l514_51411
