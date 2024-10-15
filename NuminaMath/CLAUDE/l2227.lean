import Mathlib

namespace NUMINAMATH_CALUDE_x_power_y_value_l2227_222715

theorem x_power_y_value (x y : ℝ) (h : |x + 2*y| + (y - 3)^2 = 0) : x^y = -216 := by
  sorry

end NUMINAMATH_CALUDE_x_power_y_value_l2227_222715


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2227_222712

theorem cubic_root_sum (p q r : ℝ) : 
  (p^3 - 2*p - 2 = 0) → 
  (q^3 - 2*q - 2 = 0) → 
  (r^3 - 2*r - 2 = 0) → 
  p*(q - r)^2 + q*(r - p)^2 + r*(p - q)^2 = -24 := by sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2227_222712


namespace NUMINAMATH_CALUDE_power_tower_mod_1000_l2227_222736

theorem power_tower_mod_1000 : 5^(5^(5^5)) ≡ 625 [ZMOD 1000] := by sorry

end NUMINAMATH_CALUDE_power_tower_mod_1000_l2227_222736


namespace NUMINAMATH_CALUDE_strongest_correlation_l2227_222741

-- Define the correlation coefficients
def r₁ : ℝ := 0
def r₂ : ℝ := -0.95
def r₃ : ℝ := 0.89  -- We use the absolute value directly as it's given
def r₄ : ℝ := 0.75

-- Theorem stating that r₂ has the largest absolute value
theorem strongest_correlation :
  abs r₂ > abs r₁ ∧ abs r₂ > abs r₃ ∧ abs r₂ > abs r₄ := by
  sorry


end NUMINAMATH_CALUDE_strongest_correlation_l2227_222741


namespace NUMINAMATH_CALUDE_expression_equals_five_l2227_222714

theorem expression_equals_five :
  (π + Real.sqrt 3) ^ 0 + (-2) ^ 2 + |(-1/2)| - Real.sin (30 * π / 180) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_five_l2227_222714


namespace NUMINAMATH_CALUDE_triangle_arctan_sum_l2227_222755

/-- Given a triangle ABC with sides a, b, c and angles α, β, γ in arithmetic progression
    with the smallest angle α = π/6, prove that arctan(a/(c+b)) + arctan(b/(c+a)) = π/4 -/
theorem triangle_arctan_sum (a b c : ℝ) (α β γ : ℝ) :
  α = π/6 →
  β = α + (γ - α)/2 →
  γ = α + 2*(γ - α)/2 →
  α + β + γ = π →
  a^2 + b^2 = c^2 →
  Real.arctan (a/(c+b)) + Real.arctan (b/(c+a)) = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_arctan_sum_l2227_222755


namespace NUMINAMATH_CALUDE_combine_numbers_to_24_l2227_222766

theorem combine_numbers_to_24 : (10 * 10 - 4) / 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_combine_numbers_to_24_l2227_222766


namespace NUMINAMATH_CALUDE_nathan_basketball_games_l2227_222701

/-- Calculates the number of basketball games played given the number of air hockey games,
    the cost per game, and the total tokens used. -/
def basketball_games (air_hockey_games : ℕ) (cost_per_game : ℕ) (total_tokens : ℕ) : ℕ :=
  (total_tokens - air_hockey_games * cost_per_game) / cost_per_game

/-- Proves that Nathan played 4 basketball games given the problem conditions. -/
theorem nathan_basketball_games :
  basketball_games 2 3 18 = 4 := by
  sorry

#eval basketball_games 2 3 18

end NUMINAMATH_CALUDE_nathan_basketball_games_l2227_222701


namespace NUMINAMATH_CALUDE_green_block_weight_l2227_222780

/-- The weight of the yellow block in pounds -/
def yellow_weight : ℝ := 0.6

/-- The difference in weight between the yellow and green blocks in pounds -/
def weight_difference : ℝ := 0.2

/-- The weight of the green block in pounds -/
def green_weight : ℝ := yellow_weight - weight_difference

theorem green_block_weight : green_weight = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_green_block_weight_l2227_222780


namespace NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_l2227_222726

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 4|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f_geq_2 :
  {x : ℝ | f x ≥ 2} = {x : ℝ | x ≤ -7} ∪ {x : ℝ | x ≥ 5/3} := by sorry

-- Theorem for the minimum value of f(x)
theorem min_value_f :
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_geq_2_min_value_f_l2227_222726


namespace NUMINAMATH_CALUDE_probability_for_specific_cube_l2227_222784

/-- Represents a cube with painted faces -/
structure PaintedCube where
  side_length : ℕ
  total_cubes : ℕ
  full_face_painted : ℕ
  half_face_painted : ℕ

/-- Calculates the probability of selecting one cube with exactly one painted face
    and one cube with no painted faces when two cubes are randomly selected -/
def probability_one_painted_one_unpainted (cube : PaintedCube) : ℚ :=
  let one_face_painted := cube.full_face_painted - cube.half_face_painted
  let no_face_painted := cube.total_cubes - cube.full_face_painted - cube.half_face_painted
  let total_combinations := (cube.total_cubes * (cube.total_cubes - 1)) / 2
  let favorable_outcomes := one_face_painted * no_face_painted
  favorable_outcomes / total_combinations

/-- The main theorem stating the probability for the specific cube configuration -/
theorem probability_for_specific_cube : 
  let cube := PaintedCube.mk 5 125 25 12
  probability_one_painted_one_unpainted cube = 44 / 155 := by
  sorry

end NUMINAMATH_CALUDE_probability_for_specific_cube_l2227_222784


namespace NUMINAMATH_CALUDE_concave_hexagon_guard_theorem_l2227_222762

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A concave hexagon represented by its vertices -/
structure ConcaveHexagon where
  vertices : Fin 6 → Point
  is_concave : Bool

/-- Represents visibility between two points -/
def visible (p1 p2 : Point) (h : ConcaveHexagon) : Prop :=
  sorry

/-- A guard's position -/
structure Guard where
  position : Point

/-- Checks if a point is visible to at least one guard -/
def visible_to_guards (p : Point) (guards : List Guard) (h : ConcaveHexagon) : Prop :=
  ∃ g ∈ guards, visible g.position p h

theorem concave_hexagon_guard_theorem (h : ConcaveHexagon) :
  ∃ (guards : List Guard), guards.length ≤ 2 ∧
    ∀ (p : Point), (∃ i : Fin 6, p = h.vertices i) → visible_to_guards p guards h :=
  sorry

end NUMINAMATH_CALUDE_concave_hexagon_guard_theorem_l2227_222762


namespace NUMINAMATH_CALUDE_max_composite_sum_l2227_222716

/-- A positive integer is composite if it has a factor other than 1 and itself. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

/-- The sum of a list of natural numbers. -/
def ListSum (L : List ℕ) : ℕ :=
  L.foldl (· + ·) 0

/-- A list of natural numbers is a valid decomposition if all its elements are composite
    and their sum is 2013. -/
def IsValidDecomposition (L : List ℕ) : Prop :=
  (∀ n ∈ L, IsComposite n) ∧ ListSum L = 2013

theorem max_composite_sum :
  (∃ L : List ℕ, IsValidDecomposition L ∧ L.length = 502) ∧
  (∀ L : List ℕ, IsValidDecomposition L → L.length ≤ 502) := by
  sorry

end NUMINAMATH_CALUDE_max_composite_sum_l2227_222716


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l2227_222795

/-- The equation x^2 + x - m = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of "If m > 0, then the equation x^2 + x - m = 0 has real roots" 
    is equivalent to "If the equation x^2 + x - m = 0 does not have real roots, then m ≤ 0" -/
theorem contrapositive_equivalence : 
  (¬(has_real_roots m) → m ≤ 0) ↔ (m > 0 → has_real_roots m) :=
sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l2227_222795


namespace NUMINAMATH_CALUDE_system_solution_l2227_222725

def solution_set : Set (ℂ × ℂ × ℂ) :=
  {(0, 0, 0), (2/3, -1/3, -1/3), (1/3, (-1+Complex.I*Real.sqrt 3)/6, (-1-Complex.I*Real.sqrt 3)/6),
   (1/3, (-1-Complex.I*Real.sqrt 3)/6, (-1+Complex.I*Real.sqrt 3)/6), (1, 0, 0), (1/3, 1/3, 1/3),
   (2/3, (1+Complex.I*Real.sqrt 3)/6, (1-Complex.I*Real.sqrt 3)/6),
   (2/3, (1-Complex.I*Real.sqrt 3)/6, (1+Complex.I*Real.sqrt 3)/6)}

theorem system_solution (x y z : ℂ) :
  (x^2 + 2*y*z = x ∧ y^2 + 2*z*x = z ∧ z^2 + 2*x*y = y) ↔ (x, y, z) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l2227_222725


namespace NUMINAMATH_CALUDE_cosine_product_sqrt_value_l2227_222713

theorem cosine_product_sqrt_value :
  Real.sqrt ((3 - Real.cos (π / 9) ^ 2) * (3 - Real.cos (2 * π / 9) ^ 2) * (3 - Real.cos (4 * π / 9) ^ 2)) = 9 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cosine_product_sqrt_value_l2227_222713


namespace NUMINAMATH_CALUDE_problem_statement_l2227_222746

def prop_p (m : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, 2 * x - 2 ≥ m^2 - 3 * m

def prop_q (m : ℝ) (a : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (-1) 1, m ≤ a * x

theorem problem_statement (m : ℝ) :
  (prop_p m → m ∈ Set.Icc 1 2) ∧
  (¬(prop_p m ∧ prop_q m 1) ∧ (prop_p m ∨ prop_q m 1) →
    m < 1 ∨ (1 < m ∧ m ≤ 2)) := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2227_222746


namespace NUMINAMATH_CALUDE_number_problem_l2227_222711

theorem number_problem (x : ℚ) : x - (3/5) * x = 64 → x = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2227_222711


namespace NUMINAMATH_CALUDE_absolute_value_inequality_implies_range_l2227_222709

theorem absolute_value_inequality_implies_range (a : ℝ) :
  (∀ x : ℝ, |2*x - 1| + |x + 2| ≥ a^2 + (1/2)*a + 2) →
  -1 ≤ a ∧ a ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_implies_range_l2227_222709


namespace NUMINAMATH_CALUDE_money_distribution_correctness_l2227_222756

def bag_distribution : List Nat := [1, 2, 4, 8, 16, 32, 64, 128, 256, 489]

def sum_subset (l : List Nat) (subset : List Bool) : Nat :=
  (l.zip subset).foldl (λ acc (x, b) => acc + if b then x else 0) 0

theorem money_distribution_correctness :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 1000 →
    ∃ subset : List Bool, subset.length = 10 ∧ sum_subset bag_distribution subset = n :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_correctness_l2227_222756


namespace NUMINAMATH_CALUDE_a_cubed_congruent_implies_a_sixth_congruent_l2227_222750

theorem a_cubed_congruent_implies_a_sixth_congruent (n : ℕ+) (a : ℤ) 
  (h : a^3 ≡ 1 [ZMOD n]) : a^6 ≡ 1 [ZMOD n] := by
  sorry

end NUMINAMATH_CALUDE_a_cubed_congruent_implies_a_sixth_congruent_l2227_222750


namespace NUMINAMATH_CALUDE_smallest_area_ellipse_l2227_222754

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive_a : 0 < a
  h_positive_b : 0 < b

/-- Checks if an ellipse contains a circle with center (h, 0) and radius 2 -/
def Ellipse.contains_circle (e : Ellipse) (h : ℝ) : Prop :=
  ∀ x y : ℝ, (x - h)^2 + y^2 = 4 → x^2 / e.a^2 + y^2 / e.b^2 ≤ 1

/-- The theorem stating the smallest possible area of the ellipse -/
theorem smallest_area_ellipse (e : Ellipse) 
  (h_contains_circle1 : e.contains_circle 2)
  (h_contains_circle2 : e.contains_circle (-2)) :
  ∃ k : ℝ, k = Real.sqrt 3 ∧ 
    ∀ e' : Ellipse, e'.contains_circle 2 → e'.contains_circle (-2) → 
      π * e'.a * e'.b ≥ k * π :=
sorry

end NUMINAMATH_CALUDE_smallest_area_ellipse_l2227_222754


namespace NUMINAMATH_CALUDE_apple_baskets_l2227_222733

/-- 
Given two baskets A and B with apples, prove that:
1. If the total amount of apples in both baskets is 75 kg
2. And after transferring 5 kg from A to B, A has 7 kg more than B
Then the original amounts in A and B were 46 kg and 29 kg, respectively
-/
theorem apple_baskets (a b : ℕ) : 
  a + b = 75 → 
  (a - 5) = (b + 5) + 7 → 
  (a = 46 ∧ b = 29) := by
sorry

end NUMINAMATH_CALUDE_apple_baskets_l2227_222733


namespace NUMINAMATH_CALUDE_cube_intersection_figures_l2227_222751

-- Define the set of possible plane figures
inductive PlaneFigure
| EquilateralTriangle
| Trapezoid
| RightAngledTriangle
| Rectangle

-- Define the set of plane figures that can be obtained from cube intersection
def CubeIntersectionFigures : Set PlaneFigure :=
  {PlaneFigure.EquilateralTriangle, PlaneFigure.Trapezoid, PlaneFigure.Rectangle}

-- Theorem statement
theorem cube_intersection_figures :
  CubeIntersectionFigures = {PlaneFigure.EquilateralTriangle, PlaneFigure.Trapezoid, PlaneFigure.Rectangle} :=
by sorry

end NUMINAMATH_CALUDE_cube_intersection_figures_l2227_222751


namespace NUMINAMATH_CALUDE_exam_scores_l2227_222758

theorem exam_scores (total_students : Nat) (high_scorers : Nat) (high_score : Nat) 
  (rest_average : Nat) (class_average : Nat) 
  (h1 : total_students = 25)
  (h2 : high_scorers = 3)
  (h3 : high_score = 95)
  (h4 : rest_average = 45)
  (h5 : class_average = 42) : 
  ∃ zero_scorers : Nat, 
    (zero_scorers + high_scorers + (total_students - zero_scorers - high_scorers)) = total_students ∧
    (high_scorers * high_score + (total_students - zero_scorers - high_scorers) * rest_average) 
      = (total_students * class_average) ∧
    zero_scorers = 5 := by
  sorry

end NUMINAMATH_CALUDE_exam_scores_l2227_222758


namespace NUMINAMATH_CALUDE_articles_sold_l2227_222765

theorem articles_sold (cost_price : ℝ) (h : cost_price > 0) : 
  ∃ (N : ℕ), (20 : ℝ) * cost_price = N * (2 * cost_price) ∧ N = 10 :=
by sorry

end NUMINAMATH_CALUDE_articles_sold_l2227_222765


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2227_222723

theorem smallest_integer_with_remainder_one : ∃ m : ℕ, 
  (m > 1) ∧ 
  (m % 5 = 1) ∧ 
  (m % 7 = 1) ∧ 
  (m % 3 = 1) ∧ 
  (∀ n : ℕ, n > 1 → n % 5 = 1 → n % 7 = 1 → n % 3 = 1 → m ≤ n) ∧
  (m = 106) := by
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_l2227_222723


namespace NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l2227_222789

-- Define the function f
def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

-- Theorem 1: Solution set of f(x) ≥ 0
theorem solution_set_f_nonnegative :
  {x : ℝ | f x ≥ 0} = Set.Iic (-3/5) ∪ Set.Ici 1 := by sorry

-- Theorem 2: Range of a given the condition
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) →
  a ∈ Set.Iic (-3) ∪ Set.Ici 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_nonnegative_range_of_a_l2227_222789


namespace NUMINAMATH_CALUDE_james_travel_time_l2227_222760

-- Define the parameters
def driving_speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

-- Define the theorem
theorem james_travel_time :
  (distance / driving_speed) + stop_time = 7 :=
by sorry

end NUMINAMATH_CALUDE_james_travel_time_l2227_222760


namespace NUMINAMATH_CALUDE_correct_operation_l2227_222707

theorem correct_operation (x y : ℝ) : 5 * x * y - 4 * x * y = x * y := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2227_222707


namespace NUMINAMATH_CALUDE_quadrilateral_area_theorem_l2227_222786

-- Define a structure for a point in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to calculate the signed area of a triangle
def signedArea (A B C : Point) : ℝ := sorry

-- Define the theorem
theorem quadrilateral_area_theorem 
  (A B C D O K L : Point) 
  (h1 : K.x = (A.x + C.x) / 2 ∧ K.y = (A.y + C.y) / 2)  -- K is midpoint of AC
  (h2 : L.x = (B.x + D.x) / 2 ∧ L.y = (B.y + D.y) / 2)  -- L is midpoint of BD
  : (signedArea A O B) + (signedArea C O D) - 
    ((signedArea B O C) - (signedArea D O A)) = 
    4 * (signedArea K O L) := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_theorem_l2227_222786


namespace NUMINAMATH_CALUDE_birds_on_fence_l2227_222785

theorem birds_on_fence (initial_birds landing_birds : ℕ) :
  initial_birds = 12 →
  landing_birds = 8 →
  initial_birds + landing_birds = 20 := by
sorry

end NUMINAMATH_CALUDE_birds_on_fence_l2227_222785


namespace NUMINAMATH_CALUDE_new_earnings_after_raise_l2227_222770

-- Define the original weekly earnings
def original_earnings : ℚ := 50

-- Define the percentage increase
def percentage_increase : ℚ := 50 / 100

-- Theorem to prove
theorem new_earnings_after_raise :
  original_earnings * (1 + percentage_increase) = 75 := by
  sorry

end NUMINAMATH_CALUDE_new_earnings_after_raise_l2227_222770


namespace NUMINAMATH_CALUDE_pages_per_chapter_l2227_222798

theorem pages_per_chapter 
  (total_chapters : ℕ) 
  (total_pages : ℕ) 
  (h1 : total_chapters = 31) 
  (h2 : total_pages = 1891) :
  total_pages / total_chapters = 61 := by
sorry

end NUMINAMATH_CALUDE_pages_per_chapter_l2227_222798


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l2227_222772

-- Define the complex number z
def z (m : ℝ) : ℂ := (1 + Complex.I) * (m - 2 * Complex.I)

-- Define the condition for a complex number to be in the first quadrant
def in_first_quadrant (w : ℂ) : Prop := 0 < w.re ∧ 0 < w.im

-- Theorem statement
theorem z_in_first_quadrant_iff_m_gt_two (m : ℝ) :
  in_first_quadrant (z m) ↔ m > 2 := by sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_iff_m_gt_two_l2227_222772


namespace NUMINAMATH_CALUDE_bicycle_wheels_l2227_222724

/-- Proves that each bicycle has 2 wheels given the conditions of the problem -/
theorem bicycle_wheels :
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 4
  let num_unicycles : ℕ := 7
  let tricycle_wheels : ℕ := 3
  let unicycle_wheels : ℕ := 1
  let total_wheels : ℕ := 25
  ∃ (bicycle_wheels : ℕ),
    bicycle_wheels * num_bicycles +
    tricycle_wheels * num_tricycles +
    unicycle_wheels * num_unicycles = total_wheels ∧
    bicycle_wheels = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l2227_222724


namespace NUMINAMATH_CALUDE_original_sandbox_capacity_l2227_222767

/-- Given a rectangular sandbox, this theorem proves that if a new sandbox with twice the dimensions
    has a capacity of 80 cubic feet, then the original sandbox has a capacity of 10 cubic feet. -/
theorem original_sandbox_capacity
  (length width height : ℝ)
  (new_sandbox_capacity : ℝ → ℝ → ℝ → ℝ)
  (h_new_sandbox : new_sandbox_capacity (2 * length) (2 * width) (2 * height) = 80) :
  length * width * height = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_sandbox_capacity_l2227_222767


namespace NUMINAMATH_CALUDE_sin_six_arcsin_one_third_l2227_222744

theorem sin_six_arcsin_one_third :
  Real.sin (6 * Real.arcsin (1/3)) = 191 * Real.sqrt 2 / 729 := by
  sorry

end NUMINAMATH_CALUDE_sin_six_arcsin_one_third_l2227_222744


namespace NUMINAMATH_CALUDE_sector_angle_unchanged_l2227_222783

theorem sector_angle_unchanged 
  (r₁ r₂ : ℝ) 
  (s₁ s₂ : ℝ) 
  (θ₁ θ₂ : ℝ) 
  (h_positive : r₁ > 0 ∧ r₂ > 0)
  (h_radius : r₂ = 2 * r₁)
  (h_arc : s₂ = 2 * s₁)
  (h_angle₁ : s₁ = r₁ * θ₁)
  (h_angle₂ : s₂ = r₂ * θ₂) :
  θ₂ = θ₁ := by
sorry

end NUMINAMATH_CALUDE_sector_angle_unchanged_l2227_222783


namespace NUMINAMATH_CALUDE_min_value_theorem_l2227_222763

theorem min_value_theorem (x y k : ℝ) 
  (hx : x > k) (hy : y > k) (hk : k > 1) :
  ∃ (m : ℝ), m = 8 * k ∧ 
  ∀ (a b : ℝ), a > k → b > k → 
  (a^2 / (b - k) + b^2 / (a - k)) ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2227_222763


namespace NUMINAMATH_CALUDE_complex_solution_l2227_222720

theorem complex_solution (z : ℂ) (h : (2 + Complex.I) * z = 3 + 4 * Complex.I) :
  z = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_solution_l2227_222720


namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l2227_222739

/-- Given a cubic function f(x) = ax³ + bx² passing through (-1, 2) with slope -3 at x = -1,
    prove that f(x) = x³ + 3x² -/
theorem cubic_function_uniqueness (a b : ℝ) : 
  let f := fun (x : ℝ) ↦ a * x^3 + b * x^2
  let f' := fun (x : ℝ) ↦ 3 * a * x^2 + 2 * b * x
  (f (-1) = 2) → (f' (-1) = -3) → (a = 1 ∧ b = 3) := by sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l2227_222739


namespace NUMINAMATH_CALUDE_dave_remaining_candy_l2227_222706

/-- The number of chocolate candy boxes Dave bought -/
def total_boxes : ℕ := 12

/-- The number of boxes Dave gave to his little brother -/
def given_boxes : ℕ := 5

/-- The number of candy pieces in each box -/
def pieces_per_box : ℕ := 3

/-- The number of candy pieces Dave still has -/
def remaining_pieces : ℕ := (total_boxes - given_boxes) * pieces_per_box

theorem dave_remaining_candy : remaining_pieces = 21 := by
  sorry

end NUMINAMATH_CALUDE_dave_remaining_candy_l2227_222706


namespace NUMINAMATH_CALUDE_emma_bank_account_l2227_222730

/-- Calculates the final amount in a bank account after a withdrawal and deposit -/
def final_amount (initial_savings withdrawal : ℕ) : ℕ :=
  let remaining := initial_savings - withdrawal
  let deposit := 2 * withdrawal
  remaining + deposit

/-- Proves that given the specific conditions, the final amount is $290 -/
theorem emma_bank_account : final_amount 230 60 = 290 := by
  sorry

end NUMINAMATH_CALUDE_emma_bank_account_l2227_222730


namespace NUMINAMATH_CALUDE_golden_ratio_and_relations_l2227_222757

theorem golden_ratio_and_relations :
  -- Part 1: Golden Ratio
  (∃ x : ℝ, x > 0 ∧ x^2 + x - 1 = 0 ∧ x = (-1 + Real.sqrt 5) / 2) ∧
  -- Part 2: Relation between a and b
  (∀ m a b : ℝ, a^2 + m*a = 1 → b^2 - 2*m*b = 4 → b ≠ -2*a → a*b = 2) ∧
  -- Part 3: Relation between p, q, and n
  (∀ n p q : ℝ, p ≠ q → p^2 + n*p - 1 = q → q^2 + n*q - 1 = p → p*q - n = 0) :=
by sorry

end NUMINAMATH_CALUDE_golden_ratio_and_relations_l2227_222757


namespace NUMINAMATH_CALUDE_permutation_fraction_equality_l2227_222792

def A (n m : ℕ) : ℚ := (Nat.factorial n) / (Nat.factorial (n - m))

theorem permutation_fraction_equality : 
  (2 * A 8 5 + 7 * A 8 4) / (A 8 8 - A 9 5) = 1 / 15 := by sorry

end NUMINAMATH_CALUDE_permutation_fraction_equality_l2227_222792


namespace NUMINAMATH_CALUDE_election_votes_calculation_l2227_222776

theorem election_votes_calculation (total_votes : ℕ) : 
  (75 : ℚ) / 100 * ((100 : ℚ) - 15) / 100 * total_votes = 357000 → 
  total_votes = 560000 := by
sorry

end NUMINAMATH_CALUDE_election_votes_calculation_l2227_222776


namespace NUMINAMATH_CALUDE_solve_scooter_price_l2227_222731

def scooter_price_problem (upfront_percentage : ℚ) (upfront_amount : ℚ) (num_installments : ℕ) : Prop :=
  let total_price : ℚ := upfront_amount / upfront_percentage * 100
  let remaining_amount : ℚ := total_price * (1 - upfront_percentage)
  let installment_amount : ℚ := remaining_amount / num_installments
  (upfront_percentage = 20/100) ∧ 
  (upfront_amount = 240) ∧ 
  (num_installments = 12) ∧
  (total_price = 1200) ∧ 
  (installment_amount = 80)

theorem solve_scooter_price : 
  ∃ (upfront_percentage : ℚ) (upfront_amount : ℚ) (num_installments : ℕ),
    scooter_price_problem upfront_percentage upfront_amount num_installments :=
by
  sorry

end NUMINAMATH_CALUDE_solve_scooter_price_l2227_222731


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l2227_222738

/-- The number of integer solutions to the given system of equations -/
def num_solutions : ℕ := 2

/-- The system of equations -/
def system (x y z : ℤ) : Prop :=
  x^2 - 4*x*y + 3*y^2 - z^2 = 40 ∧
  -x^2 + 4*y*z + 3*z^2 = 47 ∧
  x^2 + 2*x*y + 9*z^2 = 110

/-- Theorem stating that there are exactly 2 solutions to the system -/
theorem exactly_two_solutions :
  (∃! (solutions : Finset (ℤ × ℤ × ℤ)), solutions.card = num_solutions ∧
    ∀ (x y z : ℤ), (x, y, z) ∈ solutions ↔ system x y z) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l2227_222738


namespace NUMINAMATH_CALUDE_system_implies_quadratic_l2227_222764

theorem system_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y - 2 = 0) ∧ (3 * x + 2 * y - 6 = 0) →
  y^2 - 13 * y + 26 = 0 := by
  sorry

end NUMINAMATH_CALUDE_system_implies_quadratic_l2227_222764


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2227_222799

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (x + 12) = 10) ∧ (x = 88) := by
sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2227_222799


namespace NUMINAMATH_CALUDE_second_month_interest_l2227_222777

/-- Calculates the interest charged in the second month for a loan with monthly compound interest. -/
theorem second_month_interest
  (initial_loan : ℝ)
  (monthly_interest_rate : ℝ)
  (h1 : initial_loan = 200)
  (h2 : monthly_interest_rate = 0.1) :
  let first_month_total := initial_loan * (1 + monthly_interest_rate)
  let second_month_interest := first_month_total * monthly_interest_rate
  second_month_interest = 22 := by
sorry

end NUMINAMATH_CALUDE_second_month_interest_l2227_222777


namespace NUMINAMATH_CALUDE_champion_is_C_l2227_222743

-- Define the contestants
inductive Contestant : Type
  | A | B | C | D | E

-- Define the predictions
def father_prediction (c : Contestant) : Prop :=
  c = Contestant.A ∨ c = Contestant.C

def mother_prediction (c : Contestant) : Prop :=
  c ≠ Contestant.B ∧ c ≠ Contestant.C

def child_prediction (c : Contestant) : Prop :=
  c = Contestant.D ∨ c = Contestant.E

-- Define the condition that only one prediction is correct
def only_one_correct (c : Contestant) : Prop :=
  (father_prediction c ∧ ¬mother_prediction c ∧ ¬child_prediction c) ∨
  (¬father_prediction c ∧ mother_prediction c ∧ ¬child_prediction c) ∨
  (¬father_prediction c ∧ ¬mother_prediction c ∧ child_prediction c)

-- Theorem statement
theorem champion_is_C :
  ∃ (c : Contestant), only_one_correct c → c = Contestant.C :=
sorry

end NUMINAMATH_CALUDE_champion_is_C_l2227_222743


namespace NUMINAMATH_CALUDE_parallelogram_vertices_parabola_parallel_intersection_l2227_222778

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if four points form a parallelogram -/
def isParallelogram (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x = p4.x - p3.x ∧ p2.y - p1.y = p4.y - p3.y) ∨
  (p3.x - p1.x = p4.x - p2.x ∧ p3.y - p1.y = p4.y - p2.y)

/-- The parabola equation x = y^2 -/
def onParabola (p : Point) : Prop :=
  p.x = p.y^2

/-- Theorem about parallelogram vertices -/
theorem parallelogram_vertices :
  ∀ p : Point,
  isParallelogram ⟨0, 0⟩ ⟨1, 1⟩ ⟨1, 0⟩ p →
  (p = ⟨0, 1⟩ ∨ p = ⟨0, -1⟩ ∨ p = ⟨2, 1⟩) :=
sorry

/-- Theorem about parallel lines intersecting parabola -/
theorem parabola_parallel_intersection (a : ℝ) :
  a ≠ 0 → a ≠ 1 → a ≠ -1 →
  ∀ v : Point,
  onParabola ⟨0, 0⟩ ∧ onParabola ⟨1, 1⟩ ∧ onParabola ⟨a^2, a⟩ ∧ onParabola v →
  (∃ l1 l2 : ℝ → ℝ, l1 0 = 0 ∧ l1 1 = 1 ∧ l1 (a^2) = a ∧ l1 v.x = v.y ∧
               l2 0 = 0 ∧ l2 1 = 1 ∧ l2 (a^2) = a ∧ l2 v.x = v.y ∧
               ∀ x, l1 x - l2 x = (l1 1 - l2 1)) →
  (v = ⟨4, a⟩ ∨ v = ⟨4, -a⟩) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_vertices_parabola_parallel_intersection_l2227_222778


namespace NUMINAMATH_CALUDE_grid_walk_probability_l2227_222700

def grid_walk (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * (grid_walk (x-1) y + grid_walk x (y-1) + grid_walk (x-1) (y-1))

theorem grid_walk_probability :
  ∃ (m n : ℕ), 
    m > 0 ∧ 
    n > 0 ∧ 
    ¬(3 ∣ m) ∧ 
    grid_walk 5 5 = m / (3^n : ℚ) ∧ 
    m + n = 1186 :=
sorry

end NUMINAMATH_CALUDE_grid_walk_probability_l2227_222700


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2227_222742

theorem triangle_perimeter (a b c : ℝ) (ha : a = 10) (hb : b = 15) (hc : c = 19) :
  a + b + c = 44 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2227_222742


namespace NUMINAMATH_CALUDE_turnip_potato_ratio_l2227_222752

theorem turnip_potato_ratio (total_potatoes : ℝ) (total_turnips : ℝ) (base_potatoes : ℝ) 
  (h1 : total_potatoes = 20)
  (h2 : total_turnips = 8)
  (h3 : base_potatoes = 5) :
  (base_potatoes / total_potatoes) * total_turnips = 2 := by
  sorry

end NUMINAMATH_CALUDE_turnip_potato_ratio_l2227_222752


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l2227_222791

-- Define the cost of mangoes and oranges
def mango_cost : ℝ := 0.60
def orange_cost : ℝ := 0.40

-- Define the weight of mangoes and oranges Kelly buys
def mango_weight : ℝ := 5
def orange_weight : ℝ := 5

-- Define the discount percentage
def discount_rate : ℝ := 0.10

-- Define the function to calculate the total cost after discount
def total_cost_after_discount (m_cost o_cost m_weight o_weight disc_rate : ℝ) : ℝ :=
  let total_cost := (m_cost * 2 * m_weight) + (o_cost * 4 * o_weight)
  total_cost * (1 - disc_rate)

-- Theorem statement
theorem fruit_purchase_cost :
  total_cost_after_discount mango_cost orange_cost mango_weight orange_weight discount_rate = 12.60 := by
  sorry

end NUMINAMATH_CALUDE_fruit_purchase_cost_l2227_222791


namespace NUMINAMATH_CALUDE_inequality_proof_l2227_222768

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  a * b > a * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2227_222768


namespace NUMINAMATH_CALUDE_board_cutting_theorem_l2227_222722

def is_valid_board_size (n : ℕ) : Prop :=
  ∃ m : ℕ, n * n = 5 * m ∧ n > 5

theorem board_cutting_theorem (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n * n = m + 4 * m) ↔ is_valid_board_size n :=
sorry

end NUMINAMATH_CALUDE_board_cutting_theorem_l2227_222722


namespace NUMINAMATH_CALUDE_largest_stamp_collection_l2227_222788

theorem largest_stamp_collection (n : ℕ) (friends : ℕ) (extra : ℕ) : 
  friends = 15 →
  extra = 5 →
  n < 150 →
  n % friends = extra →
  ∀ m, m < 150 → m % friends = extra → m ≤ n →
  n = 140 :=
sorry

end NUMINAMATH_CALUDE_largest_stamp_collection_l2227_222788


namespace NUMINAMATH_CALUDE_bill_amount_correct_l2227_222728

/-- The amount of the bill in dollars -/
def bill_amount : ℝ := 26

/-- The percentage a bad tipper tips -/
def bad_tip_percent : ℝ := 0.05

/-- The percentage a good tipper tips -/
def good_tip_percent : ℝ := 0.20

/-- The difference between a good tip and a bad tip in dollars -/
def tip_difference : ℝ := 3.90

theorem bill_amount_correct : 
  (good_tip_percent - bad_tip_percent) * bill_amount = tip_difference := by
  sorry

end NUMINAMATH_CALUDE_bill_amount_correct_l2227_222728


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l2227_222719

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop := a₁ * b₂ = a₂ * b₁

/-- The first line: x + my + 6 = 0 -/
def line1 (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0

/-- The second line: 3x + (m - 2)y + 2m = 0 -/
def line2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + (m - 2) * y + 2 * m = 0

theorem parallel_lines_m_value :
  ∀ m : ℝ, (are_parallel 1 m 3 (m - 2)) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l2227_222719


namespace NUMINAMATH_CALUDE_bus_average_speed_l2227_222734

/-- The average speed of a bus traveling three equal-length sections of a road -/
theorem bus_average_speed (a : ℝ) (h : a > 0) : 
  let v1 : ℝ := 50  -- speed of first section in km/h
  let v2 : ℝ := 30  -- speed of second section in km/h
  let v3 : ℝ := 70  -- speed of third section in km/h
  let total_distance : ℝ := 3 * a  -- total distance traveled
  let total_time : ℝ := a / v1 + a / v2 + a / v3  -- total time taken
  let average_speed : ℝ := total_distance / total_time
  ∃ (ε : ℝ), ε > 0 ∧ |average_speed - 44| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_bus_average_speed_l2227_222734


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l2227_222747

/-- The area of a circle with circumference 31.4 meters is 246.49/π square meters -/
theorem circle_area_from_circumference :
  let circumference : ℝ := 31.4
  let radius : ℝ := circumference / (2 * Real.pi)
  let area : ℝ := Real.pi * radius^2
  area = 246.49 / Real.pi := by sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l2227_222747


namespace NUMINAMATH_CALUDE_missing_fraction_sum_l2227_222748

theorem missing_fraction_sum (x : ℚ) : 
  (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (-2/15 : ℚ) + (3/5 : ℚ) = (8/60 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_missing_fraction_sum_l2227_222748


namespace NUMINAMATH_CALUDE_checkerboard_corner_sum_l2227_222737

theorem checkerboard_corner_sum : 
  let n : ℕ := 8  -- size of the checkerboard
  let total_squares : ℕ := n * n
  let top_left : ℕ := 1
  let top_right : ℕ := n
  let bottom_left : ℕ := total_squares - n + 1
  let bottom_right : ℕ := total_squares
  top_left + top_right + bottom_left + bottom_right = 130 :=
by sorry

end NUMINAMATH_CALUDE_checkerboard_corner_sum_l2227_222737


namespace NUMINAMATH_CALUDE_smallest_student_count_l2227_222774

/-- Represents the number of students in each grade --/
structure GradeCount where
  eighth : ℕ
  seventh : ℕ
  sixth : ℕ

/-- Checks if the given grade counts satisfy the required ratios --/
def satisfiesRatios (gc : GradeCount) : Prop :=
  gc.eighth * 4 = gc.seventh * 7 ∧ gc.seventh * 9 = gc.sixth * 10

/-- Theorem stating the smallest possible total number of students --/
theorem smallest_student_count :
  ∃ (gc : GradeCount), satisfiesRatios gc ∧
    gc.eighth + gc.seventh + gc.sixth = 73 ∧
    (∀ (gc' : GradeCount), satisfiesRatios gc' →
      gc'.eighth + gc'.seventh + gc'.sixth ≥ 73) :=
by sorry

end NUMINAMATH_CALUDE_smallest_student_count_l2227_222774


namespace NUMINAMATH_CALUDE_factorization_problems_l2227_222775

theorem factorization_problems :
  (∀ x y : ℝ, 2*x^2*y - 8*x*y + 8*y = 2*y*(x-2)^2) ∧
  (∀ a : ℝ, 18*a^2 - 50 = 2*(3*a+5)*(3*a-5)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l2227_222775


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l2227_222796

theorem sum_of_four_consecutive_even_integers :
  ¬ (∃ m : ℤ, 56 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 20 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 108 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 88 = 4*m + 12 ∧ Even m) ∧
  (∃ m : ℤ, 200 = 4*m + 12 ∧ Even m) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_even_integers_l2227_222796


namespace NUMINAMATH_CALUDE_larger_number_proof_l2227_222708

theorem larger_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 3) (h3 : x^2 - y^2 = 39) : x = 8 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2227_222708


namespace NUMINAMATH_CALUDE_f_two_eq_zero_iff_r_eq_neg_38_l2227_222729

/-- The function f(x) as defined in the problem -/
def f (x r : ℝ) : ℝ := 2 * x^4 + x^3 + x^2 - 3 * x + r

/-- Theorem stating that f(2) = 0 if and only if r = -38 -/
theorem f_two_eq_zero_iff_r_eq_neg_38 : ∀ r : ℝ, f 2 r = 0 ↔ r = -38 := by sorry

end NUMINAMATH_CALUDE_f_two_eq_zero_iff_r_eq_neg_38_l2227_222729


namespace NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l2227_222790

/-- Given a paper with a certain number of pages and a number of days to complete it,
    calculate the number of pages that need to be written per day to finish on time. -/
def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  total_pages / days

/-- Theorem stating that for a 63-page paper due in 3 days,
    21 pages need to be written per day to finish on time. -/
theorem stacy_paper_pages_per_day :
  pages_per_day 63 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_stacy_paper_pages_per_day_l2227_222790


namespace NUMINAMATH_CALUDE_find_m_l2227_222721

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

-- Define the complement of A in U
def C_UA (m : ℕ) : Set ℕ := U \ A m

-- Theorem statement
theorem find_m : ∃ m : ℕ, C_UA m = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2227_222721


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2227_222779

-- Define the initial amount
def initial_amount : ℚ := 6160

-- Define the interest rates
def interest_rate_year1 : ℚ := 10 / 100
def interest_rate_year2 : ℚ := 12 / 100

-- Define the function to calculate the amount after one year
def amount_after_one_year (principal : ℚ) (rate : ℚ) : ℚ :=
  principal * (1 + rate)

-- Define the function to calculate the final amount after two years
def final_amount : ℚ :=
  amount_after_one_year (amount_after_one_year initial_amount interest_rate_year1) interest_rate_year2

-- State the theorem
theorem compound_interest_calculation :
  final_amount = 7589.12 := by sorry

end NUMINAMATH_CALUDE_compound_interest_calculation_l2227_222779


namespace NUMINAMATH_CALUDE_hall_area_l2227_222717

/-- The area of a rectangular hall with given length and breadth relationship -/
theorem hall_area (length breadth : ℝ) : 
  length = 30 ∧ length = breadth + 5 → length * breadth = 750 := by
  sorry

end NUMINAMATH_CALUDE_hall_area_l2227_222717


namespace NUMINAMATH_CALUDE_mixed_groups_count_l2227_222727

/-- Represents the chess club structure and game results -/
structure ChessClub where
  total_children : ℕ
  total_groups : ℕ
  children_per_group : ℕ
  boy_vs_boy_games : ℕ
  girl_vs_girl_games : ℕ

/-- Calculates the number of mixed groups in the chess club -/
def mixed_groups (club : ChessClub) : ℕ :=
  let total_games := club.total_groups * (club.children_per_group.choose 2)
  let mixed_games := total_games - club.boy_vs_boy_games - club.girl_vs_girl_games
  mixed_games / 2

/-- Theorem stating that the number of mixed groups is 23 -/
theorem mixed_groups_count (club : ChessClub) 
  (h1 : club.total_children = 90)
  (h2 : club.total_groups = 30)
  (h3 : club.children_per_group = 3)
  (h4 : club.boy_vs_boy_games = 30)
  (h5 : club.girl_vs_girl_games = 14) :
  mixed_groups club = 23 := by
  sorry

#eval mixed_groups ⟨90, 30, 3, 30, 14⟩

end NUMINAMATH_CALUDE_mixed_groups_count_l2227_222727


namespace NUMINAMATH_CALUDE_painted_cube_equality_l2227_222771

/-- Represents a cube with edge length n and two opposite faces painted. -/
structure PaintedCube where
  n : ℕ
  h_n_gt_3 : n > 3

/-- The number of unit cubes with exactly one face painted black. -/
def one_face_painted (cube : PaintedCube) : ℕ :=
  2 * (cube.n - 2)^2

/-- The number of unit cubes with exactly two faces painted black. -/
def two_faces_painted (cube : PaintedCube) : ℕ :=
  4 * (cube.n - 2)

/-- Theorem stating that the number of unit cubes with one face painted
    equals the number of unit cubes with two faces painted iff n = 4. -/
theorem painted_cube_equality (cube : PaintedCube) :
  one_face_painted cube = two_faces_painted cube ↔ cube.n = 4 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_equality_l2227_222771


namespace NUMINAMATH_CALUDE_deposit_percentage_l2227_222745

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 50 →
  remaining = 950 →
  (deposit / (deposit + remaining)) * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_deposit_percentage_l2227_222745


namespace NUMINAMATH_CALUDE_store_coloring_books_l2227_222781

theorem store_coloring_books 
  (sold : ℕ) 
  (shelves : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : sold = 33) 
  (h2 : shelves = 9) 
  (h3 : books_per_shelf = 6) : 
  sold + shelves * books_per_shelf = 87 := by
  sorry

end NUMINAMATH_CALUDE_store_coloring_books_l2227_222781


namespace NUMINAMATH_CALUDE_tree_height_difference_l2227_222759

/-- The height difference between two trees -/
theorem tree_height_difference (maple_height spruce_height : ℚ) 
  (h_maple : maple_height = 10 + 1/4)
  (h_spruce : spruce_height = 14 + 1/2) :
  spruce_height - maple_height = 19 + 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_difference_l2227_222759


namespace NUMINAMATH_CALUDE_finite_quadruples_factorial_sum_l2227_222787

theorem finite_quadruples_factorial_sum : 
  ∃ (S : Finset (ℕ × ℕ × ℕ × ℕ)), 
    ∀ (a b c n : ℕ), 
      0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < n → 
      (n.factorial = a^(n-1) + b^(n-1) + c^(n-1)) → 
      (a, b, c, n) ∈ S := by
sorry

end NUMINAMATH_CALUDE_finite_quadruples_factorial_sum_l2227_222787


namespace NUMINAMATH_CALUDE_green_to_yellow_ratio_is_two_to_one_l2227_222749

/-- Represents the number of fish of each color in an aquarium -/
structure FishCounts where
  total : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ
  other : ℕ

/-- Calculates the ratio of green fish to yellow fish -/
def greenToYellowRatio (fc : FishCounts) : ℚ :=
  fc.green / fc.yellow

/-- Theorem: The ratio of green fish to yellow fish is 2:1 given the conditions -/
theorem green_to_yellow_ratio_is_two_to_one (fc : FishCounts)
  (h1 : fc.total = 42)
  (h2 : fc.yellow = 12)
  (h3 : fc.blue = fc.yellow / 2)
  (h4 : fc.total = fc.yellow + fc.blue + fc.green + fc.other) :
  greenToYellowRatio fc = 2 := by
  sorry

#eval greenToYellowRatio { total := 42, yellow := 12, blue := 6, green := 24, other := 0 }

end NUMINAMATH_CALUDE_green_to_yellow_ratio_is_two_to_one_l2227_222749


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l2227_222703

theorem sum_of_roots_eq_fourteen : ∀ x₁ x₂ : ℝ, 
  (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 → x₁ + x₂ = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l2227_222703


namespace NUMINAMATH_CALUDE_five_b_value_l2227_222797

theorem five_b_value (a b : ℚ) (h1 : 3 * a + 4 * b = 4) (h2 : b - 3 = a) : 5 * b = 65 / 7 := by
  sorry

end NUMINAMATH_CALUDE_five_b_value_l2227_222797


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2227_222793

theorem quadratic_equation_real_roots (k : ℝ) : 
  k > 0 → ∃ x : ℝ, x^2 + 2*x - k = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_l2227_222793


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2227_222732

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where 
    2(a_1 + a_3 + a_5) + 3(a_8 + a_10) = 36, prove that a_6 = 3 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : isArithmeticSequence a)
  (h_sum : 2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36) :
  a 6 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2227_222732


namespace NUMINAMATH_CALUDE_small_cube_volume_ratio_l2227_222761

/-- Given a larger cube composed of smaller cubes, this theorem proves
    the relationship between the volumes of the larger cube and each smaller cube. -/
theorem small_cube_volume_ratio (V_L V_S : ℝ) (h : V_L > 0) (h_cube : V_L = 125 * V_S) :
  V_S = V_L / 125 := by
  sorry

end NUMINAMATH_CALUDE_small_cube_volume_ratio_l2227_222761


namespace NUMINAMATH_CALUDE_red_card_value_is_three_l2227_222735

/-- The value of a red card in credits -/
def red_card_value : ℕ := sorry

/-- The value of a blue card in credits -/
def blue_card_value : ℕ := 5

/-- The total number of cards needed to play a game -/
def total_cards : ℕ := 20

/-- The total number of credits available to buy cards -/
def total_credits : ℕ := 84

/-- The number of red cards used when playing -/
def red_cards_used : ℕ := 8

theorem red_card_value_is_three :
  red_card_value = 3 :=
by sorry

end NUMINAMATH_CALUDE_red_card_value_is_three_l2227_222735


namespace NUMINAMATH_CALUDE_new_girl_weight_l2227_222705

/-- Given a group of 8 girls, if replacing one girl weighing 70 kg with a new girl
    increases the average weight by 3 kg, then the weight of the new girl is 94 kg. -/
theorem new_girl_weight (W : ℝ) (new_weight : ℝ) : 
  (W / 8 + 3) * 8 = W - 70 + new_weight →
  new_weight = 94 := by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l2227_222705


namespace NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l2227_222773

theorem sum_of_extreme_prime_factors_of_1365 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1365 ∧ largest ∣ 1365 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1365 → p ≥ smallest) ∧
    smallest + largest = 16 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_extreme_prime_factors_of_1365_l2227_222773


namespace NUMINAMATH_CALUDE_musicians_count_l2227_222702

theorem musicians_count : ∃! n : ℕ, 
  80 < n ∧ n < 130 ∧ 
  n % 4 = 3 ∧ 
  n % 5 = 3 ∧ 
  n % 6 = 3 ∧ 
  n = 123 := by
sorry

end NUMINAMATH_CALUDE_musicians_count_l2227_222702


namespace NUMINAMATH_CALUDE_integral_roots_system_l2227_222753

theorem integral_roots_system : ∃! (x y z : ℕ),
  (z^x = y^(3*x)) ∧
  (2^z = 8 * 4^x) ∧
  (x + y + z = 18) ∧
  x = 6 ∧ y = 2 ∧ z = 15 := by sorry

end NUMINAMATH_CALUDE_integral_roots_system_l2227_222753


namespace NUMINAMATH_CALUDE_dans_helmet_craters_l2227_222740

theorem dans_helmet_craters :
  ∀ (d D R r : ℕ),
  D = d + 10 →                   -- Dan's helmet has 10 more craters than Daniel's
  R = D + d + 15 →               -- Rin's helmet has 15 more craters than Dan's and Daniel's combined
  r = 2 * R - 10 →               -- Rina's helmet has double the number of craters in Rin's minus 10
  R = 75 →                       -- Rin's helmet has 75 craters
  d + D + R + r = 540 →          -- Total craters on all helmets is 540
  Even d ∧ Even D ∧ Even R ∧ Even r →  -- Number of craters in each helmet is even
  D = 168 :=
by sorry

end NUMINAMATH_CALUDE_dans_helmet_craters_l2227_222740


namespace NUMINAMATH_CALUDE_farmer_adds_eight_pigs_l2227_222769

/-- Represents the number of animals on a farm -/
structure FarmAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- The initial number of animals on the farm -/
def initial : FarmAnimals := { cows := 2, pigs := 3, goats := 6 }

/-- The number of animals to be added -/
structure AddedAnimals where
  cows : ℕ
  pigs : ℕ
  goats : ℕ

/-- The final total number of animals -/
def finalTotal : ℕ := 21

/-- The number of cows and goats to be added -/
def knownAdditions : AddedAnimals := { cows := 3, pigs := 0, goats := 2 }

/-- Calculates the total number of animals -/
def totalAnimals (farm : FarmAnimals) : ℕ :=
  farm.cows + farm.pigs + farm.goats

/-- Theorem: The farmer plans to add 8 pigs -/
theorem farmer_adds_eight_pigs :
  ∃ (added : AddedAnimals),
    added.cows = knownAdditions.cows ∧
    added.goats = knownAdditions.goats ∧
    totalAnimals { cows := initial.cows + added.cows,
                   pigs := initial.pigs + added.pigs,
                   goats := initial.goats + added.goats } = finalTotal ∧
    added.pigs = 8 := by
  sorry

end NUMINAMATH_CALUDE_farmer_adds_eight_pigs_l2227_222769


namespace NUMINAMATH_CALUDE_empty_can_weight_l2227_222794

/-- Given a can that weighs 34 kg when full of milk and 17.5 kg when half-full, 
    prove that the empty can weighs 1 kg. -/
theorem empty_can_weight (full_weight half_weight : ℝ) 
  (h_full : full_weight = 34)
  (h_half : half_weight = 17.5) : 
  ∃ (empty_weight milk_weight : ℝ),
    empty_weight + milk_weight = full_weight ∧
    empty_weight + milk_weight / 2 = half_weight ∧
    empty_weight = 1 := by
  sorry

end NUMINAMATH_CALUDE_empty_can_weight_l2227_222794


namespace NUMINAMATH_CALUDE_teal_color_perception_l2227_222710

theorem teal_color_perception (total : ℕ) (kinda_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : total = 150)
  (h2 : kinda_blue = 90)
  (h3 : both = 45)
  (h4 : neither = 25) :
  ∃ kinda_green : ℕ, kinda_green = 80 ∧ 
  kinda_green = total - (kinda_blue - both) - neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_perception_l2227_222710


namespace NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_180_l2227_222718

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def prime_factors (n : ℕ) : Set ℕ := {p : ℕ | is_prime p ∧ p ∣ n}

theorem sum_two_smallest_prime_factors_of_180 :
  ∃ (p q : ℕ), p ∈ prime_factors 180 ∧ q ∈ prime_factors 180 ∧
  p < q ∧
  (∀ r ∈ prime_factors 180, r ≠ p → r ≥ q) ∧
  p + q = 5 :=
sorry

end NUMINAMATH_CALUDE_sum_two_smallest_prime_factors_of_180_l2227_222718


namespace NUMINAMATH_CALUDE_factor_expression_l2227_222782

theorem factor_expression (x : ℝ) : x^2 * (x + 3) + 3 * (x + 3) = (x^2 + 3) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l2227_222782


namespace NUMINAMATH_CALUDE_base_13_conversion_l2227_222704

/-- Represents a digit in base 13 -/
inductive Base13Digit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C

/-- Converts a natural number to its value in base 13 -/
def toBase13Value (d : Base13Digit) : ℕ :=
  match d with
  | Base13Digit.D0 => 0
  | Base13Digit.D1 => 1
  | Base13Digit.D2 => 2
  | Base13Digit.D3 => 3
  | Base13Digit.D4 => 4
  | Base13Digit.D5 => 5
  | Base13Digit.D6 => 6
  | Base13Digit.D7 => 7
  | Base13Digit.D8 => 8
  | Base13Digit.D9 => 9
  | Base13Digit.A => 10
  | Base13Digit.B => 11
  | Base13Digit.C => 12

/-- Represents a two-digit number in base 13 -/
structure Base13Number :=
  (msb : Base13Digit)
  (lsb : Base13Digit)

/-- Converts a Base13Number to its decimal (base 10) value -/
def toDecimal (n : Base13Number) : ℕ :=
  13 * (toBase13Value n.msb) + (toBase13Value n.lsb)

theorem base_13_conversion :
  toDecimal (Base13Number.mk Base13Digit.C Base13Digit.D0) = 156 := by
  sorry

end NUMINAMATH_CALUDE_base_13_conversion_l2227_222704
