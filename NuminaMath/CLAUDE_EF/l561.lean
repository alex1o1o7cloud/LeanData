import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l561_56129

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define the line l
def line_l (k x y : ℝ) : Prop := k * x + y + 1 = 0

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem circle_and_line_intersection :
  -- Given conditions
  (circle_C (3/2) (Real.sqrt 3 / 2)) →
  (∀ k : ℝ, ∃ x1 y1 x2 y2 : ℝ,
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l k x1 y1 ∧ line_l k x2 y2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 2) →
  -- Conclusions
  ((∀ x y : ℝ, circle_C x y ↔ (x - 2)^2 + y^2 = 1) ∧
   (∃ k1 k2 : ℝ, k1 = -1 ∧ k2 = -1/7 ∧
    (∀ k : ℝ, (∃ x1 y1 x2 y2 : ℝ,
      circle_C x1 y1 ∧ circle_C x2 y2 ∧
      line_l k x1 y1 ∧ line_l k x2 y2 ∧
      distance x1 y1 x2 y2 = Real.sqrt 2) ↔ (k = k1 ∨ k = k2))))
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_intersection_l561_56129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_dot_product_l561_56199

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 2)^2 + (y - 2)^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (10, 0)

-- Define the line l
def line_l (k x y : ℝ) : Prop := k * x - y - 10 * k = 0

-- Theorem for the trajectory equation of midpoint Q
theorem trajectory_equation :
  ∀ x y : ℝ,
  (∃ x₀ y₀ : ℝ, circleC x₀ y₀ ∧ x = (x₀ + 10) / 2 ∧ y = y₀ / 2) →
  (x - 6)^2 + (y - 1)^2 = 4 :=
by
  sorry

-- Theorem for the dot product of AM and AN
theorem dot_product :
  ∀ k x₁ y₁ x₂ y₂ : ℝ,
  circleC x₁ y₁ ∧ circleC x₂ y₂ ∧
  line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
  x₁ ≠ x₂ →
  (x₁ - 10) * (x₂ - 10) + y₁ * y₂ = 48 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_dot_product_l561_56199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l561_56183

theorem problem_solution (t : ℝ) (a b c : ℕ) 
  (h1 : (1 + Real.sin t) * (1 - Real.cos t) = 1)
  (h2 : (1 - Real.sin t) * (1 + Real.cos t) = (a : ℝ) / b - Real.sqrt (c : ℝ))
  (h3 : Nat.Coprime a b)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0) : 
  c + a + b = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l561_56183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l561_56160

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (3 * Real.cos θ, Real.sin θ)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 4 / (Real.cos θ - Real.sin θ)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- State the theorem
theorem min_distance_point :
  ∃ (θ₁ : ℝ), ∀ (θ₂ : ℝ),
    distance (C₁ θ₁) (C₂ θ₂) ≤ distance (C₁ (9 * Real.sqrt 10 / 10)) (C₂ θ₂) :=
by
  sorry

#check min_distance_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_point_l561_56160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_black_cell_l561_56150

/-- Represents a chessboard with black and white cells -/
structure Chessboard :=
  (size : ℕ)
  (black_cells : Finset (ℕ × ℕ))

/-- Represents a 2x2 square on the chessboard -/
structure Square :=
  (top_left : ℕ × ℕ)

/-- Recolors a 2x2 square on the chessboard -/
def recolor (board : Chessboard) (square : Square) : Chessboard :=
  sorry

/-- Applies a list of recoloring operations to a chessboard -/
def applyRecolors : Chessboard → List Square → Chessboard
  | board, [] => board
  | board, square::rest => applyRecolors (recolor board square) rest

/-- Theorem stating that it's impossible to have exactly one black cell after any number of recoloring operations -/
theorem no_single_black_cell (initial_board : Chessboard) :
  ∀ (operations : List Square), (applyRecolors initial_board operations).black_cells.card ≠ 1 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_single_black_cell_l561_56150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l561_56159

/-- Circle C with center at origin and radius 4 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 16

/-- Line l passing through (1,2) with angle of inclination π/6 -/
def line_l (x y : ℝ) : Prop := ∃ t : ℝ, x = 1 + (Real.sqrt 3 / 2) * t ∧ y = 2 + (1 / 2) * t

/-- Point P on the line l -/
def point_P : ℝ × ℝ := (1, 2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem intersection_product :
  ∀ A B : ℝ × ℝ,
  circle_C A.1 A.2 → circle_C B.1 B.2 →
  line_l A.1 A.2 → line_l B.1 B.2 →
  A ≠ B →
  (distance point_P A) * (distance point_P B) = 11 := by
  sorry

#check intersection_product

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l561_56159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_figure_maximal_distance_support_lines_l561_56162

-- Define a convex figure
def ConvexFigure (Φ : Set (ℝ × ℝ)) : Prop := sorry

-- Define a point in a set
def PointInSet (p : ℝ × ℝ) (S : Set (ℝ × ℝ)) : Prop := p ∈ S

-- Define the distance between two points
noncomputable def Distance (p q : ℝ × ℝ) : ℝ := sorry

-- Define a maximal distance
def MaximalDistance (d : ℝ) (Φ : Set (ℝ × ℝ)) : Prop :=
  ∀ p q, p ∈ Φ → q ∈ Φ → Distance p q ≤ d

-- Define a line perpendicular to a segment passing through a point
def PerpendicularLine (p q r : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Define a support line
def SupportLine (l : Set (ℝ × ℝ)) (Φ : Set (ℝ × ℝ)) : Prop := sorry

theorem convex_figure_maximal_distance_support_lines
  (Φ : Set (ℝ × ℝ)) (A B : ℝ × ℝ) (d : ℝ)
  (h_convex : ConvexFigure Φ)
  (h_A_in_Φ : PointInSet A Φ)
  (h_B_in_Φ : PointInSet B Φ)
  (h_maximal : MaximalDistance d Φ)
  (h_d_AB : Distance A B = d) :
  SupportLine (PerpendicularLine A B A) Φ ∧
  SupportLine (PerpendicularLine A B B) Φ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_figure_maximal_distance_support_lines_l561_56162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l561_56157

/-- The distance from a point to a plane defined by three points -/
noncomputable def distanceToPlane (M₀ M₁ M₂ M₃ : ℝ × ℝ × ℝ) : ℝ :=
  let (x₀, y₀, z₀) := M₀
  let (x₁, y₁, z₁) := M₁
  let (x₂, y₂, z₂) := M₂
  let (x₃, y₃, z₃) := M₃
  let A := (y₂ - y₁) * (z₃ - z₁) - (z₂ - z₁) * (y₃ - y₁)
  let B := (z₂ - z₁) * (x₃ - x₁) - (x₂ - x₁) * (z₃ - z₁)
  let C := (x₂ - x₁) * (y₃ - y₁) - (y₂ - y₁) * (x₃ - x₁)
  let D := -A * x₁ - B * y₁ - C * z₁
  abs (A * x₀ + B * y₀ + C * z₀ + D) / Real.sqrt (A * A + B * B + C * C)

/-- The main theorem to prove -/
theorem distance_to_specific_plane :
  distanceToPlane (5, -4, 5) (1, 3, 6) (2, 2, 1) (-1, 0, 1) = 2 * Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_specific_plane_l561_56157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l561_56197

theorem shaded_areas_equality (φ : Real) (h1 : 0 < φ) (h2 : φ < π / 2) :
  (∃ r : Real, r > 0 ∧ 
    (r^2 * Real.tan φ) / 2 - (φ * r^2) / 2 = (φ * r^2) / 2) ↔ 
  Real.tan φ = 2 * φ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_areas_equality_l561_56197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equivalence_l561_56125

open Real Polynomial

-- Define the set of x values
def X_set : Set ℝ := {x : ℝ | |x| ≤ 1}

-- Define the condition for P
def satisfies_condition (P : ℝ[X]) : Prop :=
  ∀ x ∈ X_set, P.eval (x * Real.sqrt 2) = P.eval (x + Real.sqrt (1 - x^2))

-- Define the final form of P
def final_form (P : ℝ[X]) : Prop :=
  ∃ U : ℝ[X], P = U.comp (X^8 - 4*X^6 + 5*X^4 - 2*X^2 + (1/4 : ℝ[X]))

-- The main theorem
theorem polynomial_equivalence :
  ∀ P : ℝ[X], satisfies_condition P ↔ final_form P :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_equivalence_l561_56125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l561_56171

/-- Represents a parabola with equation y^2 = 4px where p > 0 -/
structure Parabola where
  p : ℝ
  p_pos : p > 0

/-- A point on the parabola -/
structure PointOnParabola (parabola : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4 * parabola.p * x

/-- The focus of the parabola -/
noncomputable def focus (parabola : Parabola) : ℝ × ℝ := (parabola.p / 2, 0)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: If a point P(8, a) on the parabola is at distance 10 from the focus,
    then the distance from the focus to the directrix is 4 -/
theorem parabola_focus_directrix_distance (parabola : Parabola)
  (P : PointOnParabola parabola) (h1 : P.x = 8)
  (h2 : distance (P.x, P.y) (focus parabola) = 10) :
  parabola.p = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_directrix_distance_l561_56171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_QC_l561_56152

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x + 2) + 2

-- Define points P, A, B, and C
def P : ℝ × ℝ := (-2, 2)
def A : ℝ × ℝ → Prop := λ p ↦ parabola p.1 p.2 ∧ ∃ k, line_l k p.1 p.2
def B : ℝ × ℝ → Prop := λ p ↦ parabola p.1 p.2 ∧ ∃ k, line_l k p.1 p.2
def C : ℝ × ℝ := (4, 0)

-- Define the vector relation for PA and PB
def vector_relation_PA_PB (a b : ℝ × ℝ) (lambda : ℝ) : Prop :=
  (a.1 - P.1, a.2 - P.2) = lambda • (b.1 - P.1, b.2 - P.2)

-- Define the vector relation for QA and QB
def vector_relation_QA_QB (q a b : ℝ × ℝ) (lambda : ℝ) : Prop :=
  (a.1 - q.1, a.2 - q.2) = (-lambda) • (b.1 - q.1, b.2 - q.2)

-- Define the theorem
theorem min_distance_QC :
  ∀ (a b q : ℝ × ℝ) (k lambda : ℝ),
  A a → B b → a ≠ b →
  line_l k a.1 a.2 → line_l k b.1 b.2 →
  vector_relation_PA_PB a b lambda →
  vector_relation_QA_QB q a b lambda →
  (∃ (d : ℝ), ∀ (q' : ℝ × ℝ), 
    vector_relation_QA_QB q' a b lambda → 
    d ≤ Real.sqrt ((q'.1 - C.1)^2 + (q'.2 - C.2)^2)) →
  ∃ (q_min : ℝ × ℝ), 
    vector_relation_QA_QB q_min a b lambda ∧
    Real.sqrt ((q_min.1 - C.1)^2 + (q_min.2 - C.2)^2) = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_QC_l561_56152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l561_56198

theorem cos_alpha_value (α : ℝ) 
  (h1 : Real.sin (Real.pi - α) = -2 * Real.sqrt 2 / 3)
  (h2 : α > Real.pi ∧ α < 3 * Real.pi / 2) :
  Real.cos α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l561_56198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_fraction_l561_56185

/-- A positive integer containing 11235 as a contiguous substring in its decimal representation -/
def ContainsSubstring (N : ℕ+) : Prop :=
  ∃ a b : ℕ, N.val = a * 100000 + 11235 * 10^(Nat.log b 10) + b

/-- The theorem statement -/
theorem min_value_fraction (N : ℕ+) (k : ℕ+) 
  (h1 : ContainsSubstring N) 
  (h2 : (10 : ℕ)^k.val > N.val) :
  (∃ m : ℕ+, ∀ N' k' : ℕ+, 
    ContainsSubstring N' → (10 : ℕ)^k'.val > N'.val → 
    ((10 : ℕ)^k'.val - 1) / Nat.gcd N'.val ((10 : ℕ)^k'.val - 1) ≥ m.val) ∧
  ((10 : ℕ)^748 - 1) / 11235 = 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_fraction_l561_56185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l561_56102

theorem exponential_equation_solution (x : ℝ) : (3 : ℝ)^4 * (3 : ℝ)^x = 81 → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l561_56102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l561_56165

/-- 
Given a diver's descent rate and the depth of a target, 
calculate the time it takes to reach the target depth.
-/
noncomputable def time_to_reach_depth (descent_rate : ℝ) (target_depth : ℝ) : ℝ :=
  target_depth / descent_rate

/-- 
Theorem: A diver descending at 80 feet per minute will reach 
a depth of 4000 feet in 50 minutes.
-/
theorem diver_descent_time : 
  time_to_reach_depth 80 4000 = 50 := by
  -- Unfold the definition of time_to_reach_depth
  unfold time_to_reach_depth
  -- Perform the division
  norm_num
  -- QED

-- Remove the #eval statement as it's not computable
-- #eval time_to_reach_depth 80 4000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diver_descent_time_l561_56165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l561_56144

/-- A right triangle with sides a, b, c, where c is the hypotenuse -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_right : a^2 + b^2 = c^2
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c

/-- An interior point P of the triangle with perpendicular distances to the sides -/
structure InteriorPoint (t : RightTriangle) where
  PD : ℝ
  PE : ℝ
  PF : ℝ
  h_interior : 0 < PD ∧ 0 < PE ∧ 0 < PF

/-- The sum of perpendicular distances from P to the sides -/
def sum_distances (t : RightTriangle) (p : InteriorPoint t) : ℝ :=
  p.PD + p.PE + p.PF

/-- The altitudes of the right triangle -/
noncomputable def altitudes (t : RightTriangle) : ℝ × ℝ × ℝ :=
  (t.b, t.a, t.a * t.b / t.c)

theorem max_sum_distances (t : RightTriangle) :
  ∃ (max : ℝ), max = (2/3) * (altitudes t).1 + (2/3) * (altitudes t).2.1 + (2/3) * (altitudes t).2.2 ∧
  ∀ (p : InteriorPoint t), sum_distances t p ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l561_56144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l561_56107

theorem simplify_expression : (-1/16 : ℝ)^(-3/4 : ℝ) = -8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_l561_56107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l561_56140

noncomputable section

-- Define the square ABCD
def A : ℝ × ℝ := (0, 3)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (3, 0)
def D : ℝ × ℝ := (3, 3)

-- Define point E (midpoint of AB)
def E : ℝ × ℝ := (0, 1.5)

-- Define point F (one-third from B to C)
def F : ℝ × ℝ := (1, 0)

-- Define lines AF and DE
def line_AF (x : ℝ) : ℝ := -1.5 * x + 3
def line_DE (x : ℝ) : ℝ := -0.5 * x + 1.5

-- Define point I (intersection of AF and DE)
def I : ℝ × ℝ := (3/5, 12/5)

-- Define point H (intersection of BD and AF)
def H : ℝ × ℝ := (3, 3)

-- Define the area of a quadrilateral using the Shoelace formula
def quadrilateralArea (p1 p2 p3 p4 : ℝ × ℝ) : ℝ :=
  0.5 * abs (p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p4.2 + p4.1 * p1.2 -
             p2.1 * p1.2 - p3.1 * p2.2 - p4.1 * p3.2 - p1.1 * p4.2)

theorem area_of_BEIH :
  quadrilateralArea B E I H = 27/20 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_BEIH_l561_56140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l561_56177

noncomputable def f (x : ℝ) : ℝ := x^2 + x + 1

theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y : ℝ,
    (y = m * x + b) ↔ (y - f 1 = (deriv f) 1 * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l561_56177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_meat_zongzi_boxes_l561_56172

/-- The cost of meat zongzi and red date zongzi boxes -/
def CostZongzi (meat_boxes : ℕ) (date_boxes : ℕ) (meat_price : ℕ) (date_price : ℕ) : ℕ :=
  meat_boxes * meat_price + date_boxes * date_price

/-- The maximum number of red date zongzi boxes Xuanxuan can buy more than meat zongzi boxes -/
def MaxExtraDateBoxes : ℕ := 6

/-- Xuanxuan's budget -/
def Budget : ℕ := 1000

/-- Theorem: The maximum number of meat zongzi boxes Xuanxuan can buy is 12 -/
theorem max_meat_zongzi_boxes :
  ∃ (meat_price date_price : ℕ),
    CostZongzi 4 5 meat_price date_price = 220 ∧
    CostZongzi 5 10 meat_price date_price = 350 ∧
    (∀ n : ℕ,
      CostZongzi n (n + MaxExtraDateBoxes) meat_price date_price ≤ Budget →
      n ≤ 12) ∧
    CostZongzi 12 (12 + MaxExtraDateBoxes) meat_price date_price ≤ Budget :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_meat_zongzi_boxes_l561_56172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_g_zeros_l561_56111

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x + a / x + Real.log x

-- Define the function g(x)
def g (a : ℝ) (x : ℝ) : ℝ := 1 + 1 / x - a / x^2 - x

-- Theorem for the minimum value of f(x) when a = 2
theorem f_minimum_value :
  ∃ (x_min : ℝ), x_min > 0 ∧ 
  (∀ (x : ℝ), x > 0 → f 2 x_min ≤ f 2 x) ∧
  f 2 x_min = 3 :=
sorry

-- Theorem for the number of zeros of g(x)
theorem g_zeros (a : ℝ) :
  (a > 1 → ∀ x, x > 0 → g a x ≠ 0) ∧
  ((a = 1 ∨ a ≤ 0) → ∃! x, x > 0 ∧ g a x = 0) ∧
  (0 < a ∧ a < 1 → ∃ x₁ x₂, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ g a x₁ = 0 ∧ g a x₂ = 0) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_value_g_zeros_l561_56111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_expression_l561_56141

theorem units_digit_of_expression : ∃ n : ℤ, 
  (17 + Real.sqrt 251)^25 - (17 - Real.sqrt 251)^25 + 2*(17 + Real.sqrt 251)^91 = 10 * n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_expression_l561_56141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_identification_l561_56173

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem quadratic_radical_identification :
  is_quadratic_radical (Real.sqrt 3) ∧
  ¬is_quadratic_radical (3 ^ (1/3 : ℝ)) ∧
  ¬is_quadratic_radical (Real.sqrt (-4^2)) ∧
  ¬is_quadratic_radical (Real.sqrt (-5)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_radical_identification_l561_56173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_plus_cos_2x_l561_56178

noncomputable def f (x : ℝ) := Real.sin x + Real.cos (2 * x)

theorem period_of_sin_plus_cos_2x :
  ∃! T : ℝ, T > 0 ∧ 
  (∀ x : ℝ, f (x + T) = f x) ∧ 
  (∀ S : ℝ, S > 0 → (∀ x : ℝ, f (x + S) = f x) → T ≤ S) ∧
  T = π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_sin_plus_cos_2x_l561_56178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_graph_shifted_function_continuous_l561_56191

-- Define a continuous function g on an interval [a, b]
variable (g : ℝ → ℝ) (a b : ℝ) (h : ContinuousOn g (Set.Icc a b))

-- Define the horizontal shift
variable (c : ℝ)

-- State the theorem
theorem horizontal_shift_graph (x : ℝ) :
  x ∈ Set.Icc (a + c) (b + c) ↔ (x - c) ∈ Set.Icc a b :=
by
  sorry

-- State that the shifted function is continuous on the shifted interval
theorem shifted_function_continuous :
  ContinuousOn (fun x ↦ g (x - c)) (Set.Icc (a + c) (b + c)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_graph_shifted_function_continuous_l561_56191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_fast_horse_speed_equation_represents_speed_relationship_l561_56176

/-- Represents the speed of a horse in miles per day -/
noncomputable def speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- The correct equation for the specified time x days -/
theorem correct_equation (x : ℝ) : 
  x > 3 → (speed 900 (x + 1)) * 2 = speed 900 (x - 3) := by
  sorry

/-- The speed of the fast horse is twice the speed of the slow horse -/
theorem fast_horse_speed (x : ℝ) : 
  x > 3 → speed 900 (x - 3) = 2 * speed 900 (x + 1) := by
  sorry

/-- The equation correctly represents the relationship between the speeds of the horses -/
theorem equation_represents_speed_relationship (x : ℝ) : 
  x > 3 → (900 / (x + 1)) * 2 = 900 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_equation_fast_horse_speed_equation_represents_speed_relationship_l561_56176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_on_circle_l561_56134

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - m)^2 = 4

-- Define the condition for the product of y-intercepts
def intercept_product (x y : ℝ) : Prop :=
  (y / (x + 1)) * (-5 * y / (x - 5)) = 5

-- Main theorem
theorem unique_point_on_circle (m : ℝ) :
  (∃! p : ℝ × ℝ, circle_M m p.1 p.2 ∧ intercept_product p.1 p.2) →
  m = Real.sqrt 21 ∨ m = -Real.sqrt 21 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_point_on_circle_l561_56134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_c_squared_plus_c_continued_fraction_l561_56182

def continued_fraction (c : ℕ) : ℕ → ℝ 
| 0 => c
| n + 1 => if n % 2 = 0 then 2 else 2 * c

theorem sqrt_c_squared_plus_c_continued_fraction (c : ℕ) :
  Real.sqrt (c^2 + c : ℝ) = c + 1 / (continued_fraction c 0 + 1 / (continued_fraction c 1 + 1 / (continued_fraction c 2 + 1 / (continued_fraction c 3)))) := by
  sorry

#eval continued_fraction 3 0
#eval continued_fraction 3 1
#eval continued_fraction 3 2
#eval continued_fraction 3 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_c_squared_plus_c_continued_fraction_l561_56182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_condition_l561_56151

theorem sin_cos_sum_condition (α : ℝ) 
  (h1 : Real.sin α + Real.cos α = 1/5) 
  (h2 : 0 ≤ α) 
  (h3 : α ≤ π) : 
  Real.sqrt 2 * Real.sin (2 * α - π/4) = -17/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_sum_condition_l561_56151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l561_56137

/-- Represents the cost structure for apples -/
structure AppleCost where
  l : ℚ  -- Cost per kg for the first few kgs
  q : ℚ  -- Cost per kg for additional kgs
  x : ℚ  -- Number of kgs at price l

/-- Calculates the total cost for a given number of kilograms -/
def totalCost (ac : AppleCost) (kg : ℚ) : ℚ :=
  if kg ≤ ac.x then ac.l * kg else ac.l * ac.x + ac.q * (kg - ac.x)

theorem apple_cost_problem (ac : AppleCost) : 
  (totalCost ac 33 = 333) →
  (totalCost ac 36 = 366) →
  (totalCost ac 15 = 150) →
  ac.x = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_problem_l561_56137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_eq_M_l561_56170

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | |p.1 * p.2| = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arctan (1 / p.2) = Real.pi / 2}

-- Theorem statement
theorem M_union_N_eq_M : M ∪ N = M := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_eq_M_l561_56170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_vectors_l561_56100

def a : ℝ × ℝ := (1, -1)
def b : ℝ → ℝ × ℝ := fun t ↦ (-2, t)

theorem cos_angle_vectors :
  ∀ t : ℝ,
  (a.1 * (a.1 - (b t).1) + a.2 * (a.2 - (b t).2) = 0) →
  (a.1 * (b t).1 + a.2 * (b t).2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt ((b t).1^2 + (b t).2^2)) = Real.sqrt 10 / 10 := by
  sorry

#check cos_angle_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_vectors_l561_56100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_initial_money_correct_l561_56190

/-- The amount of money Brian left his house with -/
noncomputable def brian_initial_money : ℚ := 50

/-- The cost of a bag of dozen apples -/
def apple_bag_cost : ℚ := 14

/-- The amount Brian spent on kiwis -/
def kiwi_cost : ℚ := 10

/-- The amount Brian spent on bananas -/
def banana_cost : ℚ := kiwi_cost / 2

/-- The maximum number of apples Brian can buy -/
def max_apples : ℕ := 24

/-- The subway fare for one way -/
def subway_fare : ℚ := 7/2

/-- Theorem stating that the amount of money Brian left his house with is correct -/
theorem brian_initial_money_correct :
  brian_initial_money = 
    kiwi_cost + banana_cost + 
    (apple_bag_cost * (max_apples / 12 : ℚ)) + 
    (2 * subway_fare) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_initial_money_correct_l561_56190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_round_trip_percentage_l561_56104

noncomputable def round_trip_percentage (distance_to_center : ℝ) (return_trip_factor : ℝ) (return_trip_completed : ℝ) : ℝ :=
  let return_distance := distance_to_center * return_trip_factor
  let total_distance := distance_to_center + return_distance
  let distance_traveled := distance_to_center + (return_distance * return_trip_completed)
  (distance_traveled / total_distance) * 100

theorem technician_round_trip_percentage :
  let distance_to_center : ℝ := 200
  let return_trip_factor : ℝ := 1.1
  let return_trip_completed : ℝ := 0.4
  abs (round_trip_percentage distance_to_center return_trip_factor return_trip_completed - 68.57) < 0.01 := by
  sorry

-- Use #eval only for computable functions
def round_trip_percentage_approx (distance_to_center : Float) (return_trip_factor : Float) (return_trip_completed : Float) : Float :=
  let return_distance := distance_to_center * return_trip_factor
  let total_distance := distance_to_center + return_distance
  let distance_traveled := distance_to_center + (return_distance * return_trip_completed)
  (distance_traveled / total_distance) * 100

#eval round_trip_percentage_approx 200 1.1 0.4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_technician_round_trip_percentage_l561_56104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l561_56167

/-- The radius of the circle formed by points with spherical coordinates (ρ, θ, φ) = (2, θ, π/4) -/
noncomputable def circle_radius : ℝ := Real.sqrt 2

/-- Theorem: The radius of the circle formed by points with spherical coordinates (ρ, θ, φ) = (2, θ, π/4) is √2 -/
theorem circle_radius_is_sqrt_2 :
  let r : ℝ → ℝ → ℝ → ℝ := λ ρ θ φ => Real.sqrt ((ρ * Real.sin φ * Real.cos θ)^2 + (ρ * Real.sin φ * Real.sin θ)^2)
  ∀ θ, r 2 θ (π/4) = circle_radius := by
  sorry

#check circle_radius_is_sqrt_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_is_sqrt_2_l561_56167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_division_l561_56118

theorem symmetric_complex_division (z₁ z₂ : ℂ) : 
  z₁ = 1 - 2*I →
  z₂ = -(z₁.re) + z₁.im * I →
  (z₂ / z₁).im = -(4/5) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_complex_division_l561_56118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_functions_exist_for_case1_functions_exist_for_case2_l561_56169

-- Part 1
theorem no_functions_exist_for_case1 :
  ¬ (∃ f g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3) :=
by sorry

-- Part 2
theorem functions_exist_for_case2 :
  ∃ f g : ℝ → ℝ, ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_functions_exist_for_case1_functions_exist_for_case2_l561_56169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_dimensions_l561_56124

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

/-- Checks if at least one dimension is even -/
def hasEvenDimension (d : BoxDimensions) : Prop :=
  d.length % 2 = 0 ∨ d.width % 2 = 0 ∨ d.height % 2 = 0

/-- Calculates the volume of the box -/
def volume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the sum of dimensions -/
def sumDimensions (d : BoxDimensions) : ℕ :=
  d.length + d.width + d.height

/-- The main theorem -/
theorem min_sum_dimensions :
  ∃ (d : BoxDimensions),
    hasEvenDimension d ∧
    volume d = 1806 ∧
    (∀ (d' : BoxDimensions),
      hasEvenDimension d' → volume d' = 1806 →
      sumDimensions d ≤ sumDimensions d') ∧
    sumDimensions d = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_dimensions_l561_56124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_is_even_f₂_is_neither_f₃_is_even_f₄_is_neither_f₅_is_odd_l561_56149

-- Define the functions
noncomputable def f₁ (x : ℝ) : ℝ := Real.sin x / x
noncomputable def f₂ (x : ℝ) : ℝ := x + Real.cos x
noncomputable def f₃ (x : ℝ) : ℝ := x^2 + Real.cos x
noncomputable def f₄ (x : ℝ) : ℝ := Real.log x / Real.log 10
noncomputable def f₅ (x : ℝ) : ℝ := x + Real.sin x

-- Define even and odd functions
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorems
theorem f₁_is_even : IsEven f₁ := by sorry

theorem f₂_is_neither : ¬IsEven f₂ ∧ ¬IsOdd f₂ := by sorry

theorem f₃_is_even : IsEven f₃ := by sorry

theorem f₄_is_neither : ¬IsEven f₄ ∧ ¬IsOdd f₄ := by sorry

theorem f₅_is_odd : IsOdd f₅ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f₁_is_even_f₂_is_neither_f₃_is_even_f₄_is_neither_f₅_is_odd_l561_56149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_droplets_distance_l561_56148

/-- The acceleration due to gravity in mm/s² -/
noncomputable def g : ℝ := 9800

/-- The height of the cliff in millimeters -/
def cliff_height : ℝ := 300000

/-- The distance the first droplet has fallen when the second starts falling, in millimeters -/
def initial_distance : ℝ := 0.001

/-- The time it takes for the first droplet to reach the bottom of the cliff -/
noncomputable def fall_time : ℝ := Real.sqrt (2 * cliff_height / g)

/-- The distance between the two droplets when the first reaches the bottom -/
noncomputable def distance_between_droplets : ℝ := 
  2 * Real.sqrt (cliff_height * initial_distance) - initial_distance

theorem water_droplets_distance : 
  ∃ ε > 0, |distance_between_droplets - 34.6| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_droplets_distance_l561_56148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_speed_l561_56187

/-- Given a car trip with the following parameters:
  * total_time: The total duration of the trip in hours
  * total_avg_speed: The average speed for the entire trip in mph
  * first_part_time: The duration of the first part of the trip in hours
  * first_part_speed: The average speed for the first part of the trip in mph

  This theorem proves that the average speed for the remaining part of the trip
  is equal to the calculated remaining_speed. -/
theorem car_trip_speed 
  (total_time : ℝ) 
  (total_avg_speed : ℝ) 
  (first_part_time : ℝ) 
  (first_part_speed : ℝ) 
  (h1 : total_time = 8) 
  (h2 : total_avg_speed = 65) 
  (h3 : first_part_time = 4) 
  (h4 : first_part_speed = 50) :
  let remaining_time := total_time - first_part_time
  let total_distance := total_time * total_avg_speed
  let first_part_distance := first_part_time * first_part_speed
  let remaining_distance := total_distance - first_part_distance
  let remaining_speed := remaining_distance / remaining_time
  remaining_speed = 80 := by
  sorry

#check car_trip_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_trip_speed_l561_56187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_triangle_area_l561_56158

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of the triangle -/
noncomputable def area (t : Triangle) : ℝ := 
  (1/2) * t.b * t.c * Real.sin t.A

/-- The dot product of vectors AB and AC -/
noncomputable def dotProduct (t : Triangle) : ℝ := 
  t.b * t.c * Real.cos t.A

theorem triangle_property (t : Triangle) :
  area t = dotProduct t → Real.tan t.A = 2 := by
  sorry

theorem triangle_area (t : Triangle) :
  Real.tan t.A = 2 → t.B = π/4 → t.c = 3 → area t = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_triangle_area_l561_56158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l561_56110

-- Define the hyperbola
def hyperbola (m : ℝ) (x y : ℝ) : Prop :=
  x^2 / m^2 - y^2 = 1

-- Define the focal length
def focal_length (c : ℝ) : ℝ := 2 * c

-- Define the asymptote equation
def asymptote (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x ∨ y = -k * x

-- Theorem statement
theorem hyperbola_asymptotes (m : ℝ) :
  (∃ c, focal_length c = 4 ∧ c^2 = m^2 + 1) →
  (∀ x y, hyperbola m x y ↔ asymptote (Real.sqrt 3 / 3) x y) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l561_56110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_theorem_l561_56181

/-- Represents the dual-fuel car system -/
structure DualFuelCar where
  gasoline_efficiency : ℝ  -- km per liter
  diesel_efficiency : ℝ    -- km per liter
  initial_gasoline : ℝ     -- gallons
  initial_diesel : ℝ       -- gallons
  used_gasoline : ℝ        -- gallons
  used_diesel : ℝ          -- gallons
  travel_time : ℝ          -- hours

/-- Conversion factors -/
noncomputable def gallon_to_liter : ℝ := 3.78541
noncomputable def km_to_mile : ℝ := 1 / 1.60934

/-- Calculate the speed of the car in miles per hour -/
noncomputable def calculate_speed (car : DualFuelCar) : ℝ :=
  let gasoline_distance := car.used_gasoline * gallon_to_liter * car.gasoline_efficiency
  let diesel_distance := car.used_diesel * gallon_to_liter * car.diesel_efficiency
  let total_distance_km := gasoline_distance + diesel_distance
  let total_distance_miles := total_distance_km * km_to_mile
  total_distance_miles / car.travel_time

/-- Theorem stating that the car's speed is approximately 119.95 miles per hour -/
theorem car_speed_theorem (car : DualFuelCar)
  (h1 : car.gasoline_efficiency = 40)
  (h2 : car.diesel_efficiency = 55)
  (h3 : car.initial_gasoline = 5.9)
  (h4 : car.initial_diesel = 4.1)
  (h5 : car.used_gasoline = 3.9)
  (h6 : car.used_diesel = 2.45)
  (h7 : car.travel_time = 5.7) :
  ∃ ε > 0, |calculate_speed car - 119.95| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_theorem_l561_56181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_always_bal_l561_56114

-- Define the types of beings
inductive Being
| Human
| Zombie

-- Define the possible meanings of "bal"
inductive BalMeaning
| Yes
| No

-- Define the possible answers
inductive Answer
| Bal
| Yes

-- Function to determine if a being is human
def isHuman (b : Being) : Prop :=
  match b with
  | Being.Human => True
  | Being.Zombie => False

-- Function to determine the correct answer for "Are you human?"
def correctAnswerHuman (b : Being) : Prop :=
  isHuman b

-- Function to determine if a being tells the truth
def tellsTruth (b : Being) : Prop :=
  match b with
  | Being.Human => True
  | Being.Zombie => False

-- Function to determine the answer given by a being
noncomputable def answerGiven (b : Being) (m : BalMeaning) (p : Prop) : Answer :=
  if (tellsTruth b = p) then
    match m with
    | BalMeaning.Yes => Answer.Bal
    | BalMeaning.No => Answer.Yes
  else
    match m with
    | BalMeaning.Yes => Answer.Yes
    | BalMeaning.No => Answer.Bal

-- Theorem: Asking "Is 'bal' the correct answer to 'Are you human?'" always results in "bal"
theorem always_bal (b : Being) (m : BalMeaning) :
  answerGiven b m (m = BalMeaning.Yes ↔ correctAnswerHuman b) = Answer.Bal := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_always_bal_l561_56114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l561_56131

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + 1 / x

-- Define the function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x

-- Theorem statement
theorem function_properties (a : ℝ) :
  -- Part 1: Monotonicity of f when a = 1
  (a = 1 →
    (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  
  -- Part 2: Conditions for tangent line through origin
  (∃ x₀ > 0, f a x₀ = 0 ∧ (deriv (f a)) x₀ = f a x₀ / x₀ ↔ a ≤ 0 ∨ a ≥ 2) ∧
  
  -- Part 3: Range of a for the given condition on g(x)
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ (deriv (g a)) x₁ = 0 ∧ (deriv (g a)) x₂ = 0 ∧
    (g a x₁ - g a x₂) / (x₁ - x₂) ≤ (2 * Real.exp 1 / (Real.exp 2 - 1)) * a - 2 →
    a ≥ Real.exp 1 + 1 / Real.exp 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l561_56131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_negative_beta_l561_56108

/-- A function representing direct proportionality between α and β -/
def direct_prop (c : ℝ) : ℝ → ℝ := λ β => c * β

theorem alpha_value_for_negative_beta 
  (c : ℝ) -- Constant of proportionality
  (h1 : direct_prop c 10 = 5) -- Given condition: α = 5 when β = 10
  : direct_prop c (-20) = -10 := by
  sorry

#check alpha_value_for_negative_beta

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_for_negative_beta_l561_56108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_amount_is_800_l561_56166

/-- Calculates the commission for a given sale amount -/
noncomputable def commission (s : ℝ) : ℝ :=
  if s ≤ 500 then 0.20 * s
  else 0.20 * 500 + 0.25 * (s - 500)

/-- Theorem stating that if the commission is 21.875% of the sale amount,
    then the sale amount is $800 -/
theorem sale_amount_is_800 (s : ℝ) (h : commission s = 0.21875 * s) :
  s = 800 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sale_amount_is_800_l561_56166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l561_56142

-- Define the function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a*x + b) / (1 + x^2)

-- Main theorem
theorem function_properties
  (a b : ℝ)
  (h1 : ∀ x ∈ Set.Ioo (-1 : ℝ) 1, f a b (-x) = -(f a b x))
  (h2 : f a b (1/2) = 2/5)
  (h3 : StrictMono (f a b))
  : (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f a b x = x / (1 + x^2)) ∧
    (Set.Ioo (0 : ℝ) (1/2) = {x | x ∈ Set.Ioo (-1 : ℝ) 1 ∧ f a b (x-1) + f a b x < 0}) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l561_56142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l561_56186

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x * log x + a * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := |exp x - a| + a^2 / 2

-- State the theorem
theorem find_a_value :
  ∃ a : ℝ,
    (∀ x ∈ Set.Ioo 0 (exp 1), Monotone (f a)) ∧
    (∀ x ∈ Set.Icc 0 (log 3), g a x ≤ g a 0) ∧
    (g a 0 - (a^2 / 2) = 3 / 2) ∧
    a = 5 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_value_l561_56186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_go_kart_rides_l561_56195

-- Define the variables and constants
def go_kart_rides : ℕ → ℕ := λ x => x
def bumper_car_rides : ℕ := 4
def go_kart_cost : ℕ := 4
def bumper_car_cost : ℕ := 5
def total_tickets : ℕ := 24

-- State the theorem
theorem paula_go_kart_rides : 
  ∃ (g : ℕ), go_kart_rides g = g ∧ 
  go_kart_cost * g + bumper_car_cost * bumper_car_rides = total_tickets ∧
  g = 1 := by
  -- Prove the theorem
  use 1
  constructor
  · rfl
  constructor
  · rfl
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_go_kart_rides_l561_56195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_monotone_iff_k_positive_l561_56146

/-- A function f is monotonically increasing on ℝ if for all x₁ < x₂, f(x₁) < f(x₂) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ < f x₂

/-- The exponential function with parameter k -/
noncomputable def ExpFunction (k : ℝ) : ℝ → ℝ := fun x ↦ Real.exp (k * x)

/-- Theorem: The exponential function e^(kx) is monotonically increasing on ℝ 
    if and only if k is in the open interval (0, +∞) -/
theorem exp_monotone_iff_k_positive :
  ∀ k : ℝ, MonotonicallyIncreasing (ExpFunction k) ↔ k > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exp_monotone_iff_k_positive_l561_56146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l561_56138

/-- Given two vectors a and b, where a = (3/2, sin α) and b = (sin α, 1/6),
    if a is parallel to b, then α = 30° (π/6 radians). -/
theorem parallel_vectors_angle (α : ℝ) :
  let a : Fin 2 → ℝ := ![3/2, Real.sin α]
  let b : Fin 2 → ℝ := ![Real.sin α, 1/6]
  (∃ (k : ℝ), a = k • b) →
  α = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_angle_l561_56138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_trajectory_dot_product_range_triangle_area_l561_56193

noncomputable section

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Definition of the companion point -/
def companion_point (a b x₀ y₀ x y : ℝ) : Prop :=
  x = x₀ / a ∧ y = y₀ / b

/-- The trajectory of the companion point is a circle -/
theorem companion_trajectory (a b : ℝ) :
  ∀ x₀ y₀ x y : ℝ, ellipse a b x₀ y₀ → companion_point a b x₀ y₀ x y → x^2 + y^2 = 1 := by sorry

/-- The range of OM · ON when a = 2 and b = √3 -/
theorem dot_product_range :
  ∀ x₀ y₀ x y : ℝ, 
    ellipse 2 (Real.sqrt 3) x₀ y₀ → companion_point 2 (Real.sqrt 3) x₀ y₀ x y → 
    Real.sqrt 3 ≤ (x₀ * x + y₀ * y) ∧ (x₀ * x + y₀ * y) ≤ 2 := by sorry

/-- The area of triangle OAB is √3 under specific conditions -/
theorem triangle_area (x₁ y₁ x₂ y₂ : ℝ) :
  ellipse 2 (Real.sqrt 3) x₁ y₁ → 
  ellipse 2 (Real.sqrt 3) x₂ y₂ → 
  (∃ k m : ℝ, (y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m) ∨ (x₁ = m ∧ x₂ = m)) →
  (x₁ / 2)^2 + (y₁ / Real.sqrt 3)^2 + (x₂ / 2)^2 + (y₂ / Real.sqrt 3)^2 = 
    ((x₁ / 2 - x₂ / 2)^2 + (y₁ / Real.sqrt 3 - y₂ / Real.sqrt 3)^2) / 2 →
  (1/2) * Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) * 
    (abs (y₁ - y₂) * abs x₁ + abs (x₂ - x₁) * abs y₁) / 
    Real.sqrt ((y₁ - y₂)^2 + (x₂ - x₁)^2) = Real.sqrt 3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_companion_trajectory_dot_product_range_triangle_area_l561_56193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l561_56130

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x m : ℝ) : ℝ := (1/2)^x - m

-- State the theorem
theorem range_of_m (m : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1) 3, ∃ x₂ ∈ Set.Icc 0 2, f x₁ ≥ g x₂ m) → 
  m ≥ 1/4 := by
  sorry

-- Note: Set.Icc a b represents the closed interval [a, b]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l561_56130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_x5y3_l561_56147

theorem maximize_x5y3 (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 45) :
  x^5 * y^3 ≤ (225/8)^5 * (135/8)^3 ∧
  (x^5 * y^3 = (225/8)^5 * (135/8)^3 ↔ x = 225/8 ∧ y = 135/8) := by
  sorry

#check maximize_x5y3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_maximize_x5y3_l561_56147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_specific_interest_rate_l561_56139

/-- Calculates the interest rate given loan details -/
theorem interest_rate_calculation (loan_amount interest_paid : ℝ) 
  (h1 : loan_amount > 0)
  (h2 : interest_paid > 0) :
  let rate := Real.sqrt ((interest_paid * 100) / loan_amount)
  rate * rate * loan_amount / 100 = interest_paid :=
by
  sorry

/-- Specific case for the given problem -/
theorem specific_interest_rate :
  let loan_amount : ℝ := 45000
  let interest_paid : ℝ := 12500
  let rate := Real.sqrt ((interest_paid * 100) / loan_amount)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |rate - 5.27| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_specific_interest_rate_l561_56139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_properties_l561_56105

noncomputable def F (a b : ℝ) : ℝ := (a + b - abs (a - b)) / 2

noncomputable def G (x : ℝ) : ℝ := F (Real.sin x) (Real.cos x)

theorem G_properties :
  (∀ x, G x ∈ Set.Icc (-Real.sqrt 2 / 2) 1) ∧
  (∀ x, G x < 0 ↔ ∃ k : ℤ, 2 * k * Real.pi + Real.pi / 2 < x ∧ x < 2 * (k + 1) * Real.pi) ∧
  (∀ x, G x = 1 ↔ (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 2) ∨ (∃ k : ℤ, x = 2 * k * Real.pi)) ∧
  (∃ d₁ d₂ : ℝ, d₁ > 0 ∧ d₂ > 0 ∧ d₁ = 4 * d₂ ∧
    (∀ x y z : ℝ, Real.pi / 4 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 9 * Real.pi / 4 →
      (IsLocalMax G x ∧ IsLocalMax G z ∧ (∀ w, x < w ∧ w < z → ¬IsLocalMax G w)) →
      (IsLocalMin G y ∧ (∀ w, x < w ∧ w < z → w = y ∨ ¬IsLocalMin G w)) →
      z - x = d₁ ∧ y - x = d₂)) ∧
  (∀ x, G (5 * Real.pi / 4 - x) = G (5 * Real.pi / 4 + x)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_properties_l561_56105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_length_is_sqrt_15_l561_56132

/-- A rectangular solid with edge lengths a, b, and c. -/
structure RectangularSolid where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The total surface area of a rectangular solid. -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.a * r.b + r.b * r.c + r.a * r.c)

/-- The total length of all edges of a rectangular solid. -/
def totalEdgeLength (r : RectangularSolid) : ℝ :=
  4 * (r.a + r.b + r.c)

/-- The length of an interior diagonal of a rectangular solid. -/
noncomputable def interiorDiagonalLength (r : RectangularSolid) : ℝ :=
  Real.sqrt (r.a^2 + r.b^2 + r.c^2)

/-- Theorem: If the surface area is 34 and total edge length is 28,
    then the interior diagonal length is √15. -/
theorem interior_diagonal_length_is_sqrt_15 (r : RectangularSolid) 
    (h1 : surfaceArea r = 34) (h2 : totalEdgeLength r = 28) : 
    interiorDiagonalLength r = Real.sqrt 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_length_is_sqrt_15_l561_56132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_integral_l561_56101

open BigOperators

theorem binomial_sum_integral (n : ℕ) :
  (∑ k in Finset.range (n + 1), (1 / (k + 1 : ℝ)) * (n.choose k) * (1 / 3 : ℝ) ^ (k + 1)) =
  (1 / (n + 1 : ℝ)) * ((4 / 3 : ℝ) ^ (n + 1) - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_integral_l561_56101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_natural_number_inequality_l561_56143

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x + 1 / x

-- Theorem 1: f has a unique zero
theorem f_unique_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 := by sorry

-- Theorem 2: Inequality for natural numbers
theorem natural_number_inequality (n : ℕ) (hn : n > 0) :
  Real.log ((n + 1 : ℝ) / n) < 1 / Real.sqrt (n^2 + n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unique_zero_natural_number_inequality_l561_56143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l561_56113

noncomputable def f (x : ℝ) := Real.sin (x / 2) + Real.sqrt 3 * Real.cos (x / 2)

theorem f_properties :
  ∃ (S : Set ℝ) (T : ℝ → ℝ),
    (∀ x ∈ S, ∃ k : ℤ, x = 4 * k * Real.pi + Real.pi / 3) ∧
    (∀ x ∈ S, ∀ y : ℝ, f y ≤ f x) ∧
    (∀ x : ℝ, T (f x) = Real.sin x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l561_56113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_max_distance_l561_56154

/-- Proves the relationship between circle radius and maximum distance to a line -/
theorem circle_line_max_distance (r : ℝ) : 
  let circle_center : ℝ × ℝ := (-Real.sqrt 2 / 2, -Real.sqrt 2 / 2)
  let line_eq := fun (x y : ℝ) ↦ x + y = 1
  let max_distance := 3
  let distance_to_line := Real.sqrt 2 + 1 / Real.sqrt 2
  r > 0 →
  (∃ (p : ℝ × ℝ), (p.1 - circle_center.1)^2 + (p.2 - circle_center.2)^2 = r^2 ∧ 
    abs (p.1 + p.2 - 1) / Real.sqrt 2 = max_distance) ↔ 
  r = 2 - Real.sqrt 2 / 2 := by
  sorry

#check circle_line_max_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_max_distance_l561_56154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walking_rate_l561_56145

/-- Calculates the rate of travel given distance and time -/
noncomputable def calculate_rate (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Proves that for a distance of 7 miles and a time of 1.25 hours, the rate is 5.6 miles per hour -/
theorem jack_walking_rate : calculate_rate 7 1.25 = 5.6 := by
  -- Unfold the definition of calculate_rate
  unfold calculate_rate
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jack_walking_rate_l561_56145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_regular_polygons_with_interior_angle_ratio_5_3_l561_56112

/-- Interior angle of a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := 180 - (360 / n)

/-- The proposition that no two regular polygons with unit length sides have interior angles in the ratio 5:3 -/
theorem no_regular_polygons_with_interior_angle_ratio_5_3 :
  ∀ r k : ℕ, r > 2 → k > 2 →
  (interior_angle r) / (interior_angle k) ≠ 5 / 3 :=
by
  sorry

#check no_regular_polygons_with_interior_angle_ratio_5_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_regular_polygons_with_interior_angle_ratio_5_3_l561_56112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l561_56192

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 100 + y^2 / 81 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y fx fy : ℝ) : ℝ :=
  Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- Theorem statement
theorem ellipse_foci_distance 
  (x y : ℝ) 
  (h_ellipse : is_on_ellipse x y) 
  (f1x f1y f2x f2y : ℝ) 
  (h_focus1 : distance_to_focus x y f1x f1y = 6) :
  distance_to_focus x y f2x f2y = 14 := by
  sorry

#check ellipse_foci_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l561_56192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l561_56135

/-- Curve C₁ -/
noncomputable def C₁ : Set (ℝ × ℝ) :=
  {p | ∃ θ : ℝ, p.1 = 2 * Real.cos θ ∧ p.2 = Real.sqrt 3 * Real.sin θ}

/-- Curve C₂ -/
def C₂ : Set (ℝ × ℝ) :=
  {p | p.1^2 + p.2^2 = 4}

/-- Upper vertex of C₁ -/
noncomputable def M : ℝ × ℝ := (2, Real.sqrt 3)

/-- Lower vertex of C₁ -/
noncomputable def N : ℝ × ℝ := (2, -Real.sqrt 3)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Maximum value of |PM| + |PN| is 2√7 -/
theorem max_sum_distances :
  ∃ (max : ℝ), max = 2 * Real.sqrt 7 ∧
  ∀ P ∈ C₂, distance P M + distance P N ≤ max := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_distances_l561_56135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_m_value_l561_56103

def a : ℝ × ℝ := (2, -1)
def b (m : ℝ) : ℝ × ℝ := (-1, m)
def c : ℝ × ℝ := (-1, 2)

theorem vector_parallel_implies_m_value (m : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (a.1 + (b m).1, a.2 + (b m).2) = (k * c.1, k * c.2)) →
  m = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_parallel_implies_m_value_l561_56103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_l561_56163

-- Define the points
def A₁ : ℝ × ℝ := (1, 0)
def A₂ : ℝ × ℝ := (-2, 0)
def A₃ : ℝ × ℝ := (-1, 0)

-- Define the moving point M
def M : ℝ × ℝ → Prop := λ p => 
  let (x, y) := p
  (Real.sqrt ((x - A₁.1)^2 + (y - A₁.2)^2)) / (Real.sqrt ((x - A₂.1)^2 + (y - A₂.2)^2)) = Real.sqrt 2 / 2

-- Define the circle on which N moves
def circleN : ℝ × ℝ → Prop := λ p =>
  let (x, y) := p
  (x - 3)^2 + y^2 = 4

-- Define the theorem
theorem apollonian_circle :
  (∀ p, M p ↔ let (x, y) := p; x^2 + y^2 - 8*x - 2 = 0) ∧
  (∃ A₄ : ℝ × ℝ, A₄ = (2, 0) ∧
    ∀ N : ℝ × ℝ, circleN N → 
      let (x, y) := N
      (Real.sqrt ((x - A₃.1)^2 + (y - A₃.2)^2)) / (Real.sqrt ((x - A₄.1)^2 + (y - A₄.2)^2)) = 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apollonian_circle_l561_56163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_243_equals_3_to_m_l561_56116

theorem cube_root_243_equals_3_to_m (m : ℝ) : (243 : ℝ) ^ (1/3 : ℝ) = 3 ^ m ↔ m = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_243_equals_3_to_m_l561_56116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_in_cents_l561_56161

/-- Represents the cost of items in euros -/
structure CostInEuros where
  rice : ℚ
  eggs : ℚ
  kerosene : ℚ
  apples : ℚ
  cheese : ℚ

/-- Calculates the total cost in euros -/
def totalCostEuros (c : CostInEuros) : ℚ :=
  c.rice + c.eggs + c.kerosene + c.apples + c.cheese

/-- Converts euros to dollars based on the exchange rate -/
def eurosToDollars (euros : ℚ) (exchangeRate : ℚ) : ℚ :=
  euros / exchangeRate

/-- Converts dollars to cents -/
def dollarsToCents (dollars : ℚ) : ℕ :=
  (dollars * 100).floor.toNat

/-- The main theorem to prove -/
theorem total_cost_in_cents 
  (riceCost : ℚ)
  (dozenEggsCost : ℚ)
  (keroseneHalfLiterCost : ℚ)
  (fiveApplesCost : ℚ)
  (cheeseHalfKiloCost : ℚ)
  (exchangeRateIncreasePerHour : ℚ)
  (h1 : riceCost = 33/100)
  (h2 : dozenEggsCost = 2 * riceCost)
  (h3 : keroseneHalfLiterCost = 8 * (dozenEggsCost / 12))
  (h4 : fiveApplesCost = cheeseHalfKiloCost)
  (h5 : cheeseHalfKiloCost = 2 * riceCost)
  (h6 : exchangeRateIncreasePerHour = 5/100) :
  let costs := CostInEuros.mk 
    riceCost
    (dozenEggsCost / 2)
    (keroseneHalfLiterCost / 2)
    (fiveApplesCost * 3/5)
    (cheeseHalfKiloCost * 2/5)
  let totalEuros := totalCostEuros costs
  let exchangeRate := 1 + 2 * exchangeRateIncreasePerHour
  let totalDollars := eurosToDollars totalEuros exchangeRate
  let totalCents := dollarsToCents totalDollars
  totalCents = 140 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_in_cents_l561_56161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_mod_500_l561_56115

open BigOperators

def T : ℤ := ∑ n in Finset.range 1005, (-1)^n * (Nat.choose 3006 (3*n))

theorem T_mod_500 : T ≡ 18 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_mod_500_l561_56115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l561_56120

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * (Real.cos x)^2 + Real.sin (2 * x) - Real.sqrt 3

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  f (-π/6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l561_56120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_increasing_condition_log_inequality_l561_56133

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 - x) / (a * x) + log x

-- Theorem 1: Tangent line equation
theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ m b : ℝ, ∀ x, m * x + b = 0 ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f a (1 + h) - (f a 1 + m * h)| ≤ ε * |h|) :=
by sorry

-- Theorem 2: Condition for increasing function
theorem increasing_condition (a : ℝ) :
  (∀ x ≥ 2, ∀ y > x, f a y > f a x) ↔ a ≥ (1/2) :=
by sorry

-- Theorem 3: Logarithm inequality
theorem log_inequality (n : ℕ) (h : n > 1) :
  log (n : ℝ) > (Finset.range (n - 1)).sum (λ i ↦ 1 / ((i + 2) : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_increasing_condition_log_inequality_l561_56133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_weight_set_l561_56188

/-- A type representing a set of weights -/
def WeightSet := List Nat

/-- Function to check if a set of weights can measure all masses from 1 to n -/
def can_measure_all (weights : WeightSet) (n : Nat) : Prop :=
  ∀ m : Nat, m ≤ n → ∃ (left right : List Nat), 
    left ⊆ weights ∧ right ⊆ weights ∧ left.sum = m + right.sum

/-- The minimum number of weights needed to measure all masses from 1 to 100 -/
def min_weights : Nat := 5

/-- The optimal set of weights -/
def optimal_weights : WeightSet := [1, 3, 9, 27, 81]

/-- Theorem stating that the optimal weight set is correct and minimal -/
theorem optimal_weight_set :
  can_measure_all optimal_weights 100 ∧
  ∀ (weights : WeightSet), can_measure_all weights 100 → weights.length ≥ min_weights := by
  sorry

#check optimal_weight_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_weight_set_l561_56188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_doubles_when_side_and_angle_double_l561_56121

theorem triangle_area_doubles_when_side_and_angle_double
  (a b c : ℝ) (α β γ : ℝ) (A : ℝ) :
  a > 0 → b > 0 → c > 0 →
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  A = (1/2) * a * b * Real.sin α →
  (2 * a) * b * Real.sin (2 * α) / 2 = 4 * A := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_doubles_when_side_and_angle_double_l561_56121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_olympic_correct_smallest_odd_olympic_correct_l561_56180

def lambda (n : ℕ+) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1^2 - p.2^2 = n) (Finset.product (Finset.range n) (Finset.range n))).card

def is_olympic (n : ℕ+) : Prop := lambda n = 2021

def smallest_olympic : ℕ+ := 2^48 * 3^42 * 5

def smallest_odd_olympic : ℕ+ := 3^46 * 5^42 * 7

theorem smallest_olympic_correct :
  is_olympic smallest_olympic ∧
  ∀ m : ℕ+, m < smallest_olympic → ¬is_olympic m :=
sorry

theorem smallest_odd_olympic_correct :
  is_olympic smallest_odd_olympic ∧
  Odd smallest_odd_olympic.val ∧
  ∀ m : ℕ+, Odd m.val → m < smallest_odd_olympic → ¬is_olympic m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_olympic_correct_smallest_odd_olympic_correct_l561_56180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_correct_l561_56136

noncomputable section

/-- The distance from a point (x, y) to (3, 0) -/
def distToFocus (x y : ℝ) : ℝ := Real.sqrt ((x - 3)^2 + y^2)

/-- The distance from a point (x, y) to the line x = -2 -/
def distToLine (x y : ℝ) : ℝ := |x + 2|

/-- The equation of the locus of points -/
def locusEquation (x y : ℝ) : Prop := y^2 = 12 * x

theorem locus_equation_correct :
  ∀ x y : ℝ, locusEquation x y ↔ distToFocus x y = distToLine x y + 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_equation_correct_l561_56136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_output_is_45_l561_56179

/-- Represents the production of cogs on an assembly line -/
structure CogProduction where
  initial_rate : ℚ
  initial_order : ℚ
  increased_rate : ℚ
  second_batch : ℚ

/-- Calculates the overall average output of cogs per hour -/
def average_output (prod : CogProduction) : ℚ :=
  let initial_time := prod.initial_order / prod.initial_rate
  let second_time := prod.second_batch / prod.increased_rate
  let total_time := initial_time + second_time
  let total_cogs := prod.initial_order + prod.second_batch
  total_cogs / total_time

/-- Theorem stating that the average output is 45 cogs per hour for the given production parameters -/
theorem average_output_is_45 (prod : CogProduction) 
  (h1 : prod.initial_rate = 36)
  (h2 : prod.initial_order = 60)
  (h3 : prod.increased_rate = 60)
  (h4 : prod.second_batch = 60) :
  average_output prod = 45 := by
  sorry

#eval average_output { initial_rate := 36, initial_order := 60, increased_rate := 60, second_batch := 60 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_output_is_45_l561_56179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l561_56153

/-- A triangle with integral side lengths and a specific angle relationship -/
structure SpecialTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  angle_relation : ℝ  -- Placeholder for the angle relation

/-- The perimeter of a triangle -/
def perimeter (t : SpecialTriangle) : ℕ := t.a + t.b + t.c

/-- Theorem stating the minimum perimeter of the special triangle -/
theorem min_perimeter_special_triangle :
  ∀ (t : SpecialTriangle), perimeter t ≥ 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perimeter_special_triangle_l561_56153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l561_56189

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 7)
  (h_S3 : a 0 + a 1 + a 2 = 21) :
  (a 1 / a 0 = 1) ∨ (a 1 / a 0 = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l561_56189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bound_l561_56126

/-- Given a quadratic function f(x) = x^2 + bx + c, if the maximum difference
    between any two function values in [-1, 1] is at most 4, then b is in [-2, 2]. -/
theorem quadratic_function_bound (b c : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 →
    |((x₁^2 + b*x₁ + c) - (x₂^2 + b*x₂ + c))| ≤ 4) →
  b ∈ Set.Icc (-2) 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_bound_l561_56126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l561_56123

/-- The area of a trapezoid with height x, base lengths 2x+1 and 3x-1 is 5x^2/2 -/
theorem trapezoid_area (x : ℝ) :
  let height := x
  let base1 := 2*x + 1
  let base2 := 3*x - 1
  (height * (base1 + base2) / 2) = 5*x^2 / 2 := by
  simp [mul_add, mul_sub, mul_one, add_sub]
  ring

#check trapezoid_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_l561_56123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l561_56127

/-- Complex number z as a function of real number m -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 + 3*m - 28)

/-- Predicate for z being in the fourth quadrant -/
def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

/-- Predicate for z being on the negative half of the x-axis -/
def on_negative_x_axis (z : ℂ) : Prop := z.re < 0 ∧ z.im = 0

/-- Predicate for z being in the upper half-plane (including the real axis) -/
def in_upper_half_plane (z : ℂ) : Prop := z.im ≥ 0

theorem complex_number_properties (m : ℝ) :
  (in_fourth_quadrant (z m) ↔ -7 < m ∧ m < 3) ∧
  (on_negative_x_axis (z m) ↔ m = 4) ∧
  (in_upper_half_plane (z m) ↔ m ≥ 4 ∨ m ≤ -7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_properties_l561_56127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_approximation_l561_56168

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval
def interval : Set ℝ := Set.Icc 0 1

-- State the properties of f
axiom f_sign_change : f 0.625 < 0 ∧ f 0.75 > 0 ∧ f 0.6875 < 0

-- Define the precision
def precision : ℝ := 0.1

-- Define the approximate solution
def approx_solution : ℝ := 0.7

-- Theorem statement
theorem bisection_method_approximation :
  ∃ (x : ℝ), x ∈ interval ∧ 
  f x = 0 ∧ 
  |x - approx_solution| ≤ precision := by
  sorry

#check bisection_method_approximation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_method_approximation_l561_56168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_negative_roots_l561_56106

-- Define the interval
def interval : Set ℝ := Set.Icc 0 5

-- Define the condition for two negative roots
def hasTwoNegativeRoots (p : ℝ) : Prop :=
  (2/3 < p ∧ p ≤ 1) ∨ (p ≥ 2)

-- Define the measure of the set satisfying the condition
noncomputable def favorableSetMeasure : ℝ := (1 - 2/3) + (5 - 2)

-- Define the total measure of the interval
def totalMeasure : ℝ := 5 - 0

-- State the theorem
theorem probability_of_two_negative_roots :
  (favorableSetMeasure / totalMeasure) = 2/3 := by
  -- The proof is omitted and replaced with sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_two_negative_roots_l561_56106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l561_56194

/-- Oblique coordinate system -/
structure ObliqueCoordinateSystem where
  angle : ℝ
  angle_eq : angle = 2 * Real.pi / 3

/-- Point in oblique coordinates -/
structure ObliquePoint (sys : ObliqueCoordinateSystem) where
  x : ℝ
  y : ℝ

/-- Distance between two points in oblique coordinates -/
def oblique_distance (sys : ObliqueCoordinateSystem) (p1 p2 : ObliquePoint sys) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 - (p1.x - p2.x) * (p1.y - p2.y)

/-- Circle equation in oblique coordinates -/
def is_on_circle (sys : ObliqueCoordinateSystem) (center : ObliquePoint sys) (radius : ℝ) (p : ObliquePoint sys) : Prop :=
  oblique_distance sys center p = radius^2

theorem circle_equation (sys : ObliqueCoordinateSystem) 
  (center : ObliquePoint sys) 
  (hc : center.x = 2 ∧ center.y = 3) 
  (radius : ℝ) 
  (hr : radius = 2) 
  (p : ObliquePoint sys) :
  is_on_circle sys center radius p ↔ p.x^2 + p.y^2 - p.x * p.y - p.x - 4 * p.y + 3 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_equation_l561_56194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_l561_56175

def sequenceList : List ℕ := [2, 6, 12, 20, 42]

def increasing_difference (list : List ℕ) : Prop :=
  ∀ i : ℕ, i + 3 < list.length →
    (list[i+2]! - list[i+1]!) - (list[i+1]! - list[i]!) = 2

theorem missing_number (x : ℕ) :
  increasing_difference sequenceList →
  x - 20 = 42 - x →
  x = 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_missing_number_l561_56175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_and_collinearity_l561_56174

-- Define the points and sequences
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def O : ℝ × ℝ := (0, 0)

-- Define the sequences a_n and b_n
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

-- Define P_n
def P (n : ℕ) : ℝ × ℝ := (a n * A.1 + b n * B.1, a n * A.2 + b n * B.2)

-- State the theorem
theorem point_coordinates_and_collinearity :
  -- Conditions
  (∀ n m : ℕ, n ≠ m → P n ≠ P m) →  -- Distinct points
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) →  -- Arithmetic sequence
  (∃ q : ℝ, ∀ n : ℕ, b (n + 1) / b n = q) →  -- Geometric sequence
  (P 1 - A = 2 * (B - P 1)) →  -- Condition for P1
  -- Conclusions
  (P 1 = (1/3, 2/3)) ∧  -- Coordinates of P1
  (∃ d q : ℝ, (d = 0 ∨ q = 1) → ∀ n m : ℕ, (P n).1 = (P m).1 ∨ (P n).2 = (P m).2) -- Collinearity condition
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_and_collinearity_l561_56174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equation_one_root_l561_56128

/-- Given non-zero real numbers a, b, c, d, the determinant equation has exactly one real root (x = 0) -/
theorem determinant_equation_one_root (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃! x : ℝ, Matrix.det
    !![x, c + d, -b;
      -c, x, a + d;
      b, -a, x] = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_equation_one_root_l561_56128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rhombuses_in_triangle_l561_56164

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  sideLength : ℕ

/-- Represents a rhombus composed of smaller triangles -/
structure Rhombus where
  numTriangles : ℕ

/-- The main triangle divided into smaller triangles -/
def mainTriangle : EquilateralTriangle :=
  { sideLength := 10 }

/-- The smaller triangles that make up the main triangle -/
def smallerTriangles : EquilateralTriangle :=
  { sideLength := 1 }

/-- The total number of smaller triangles in the main triangle -/
def totalSmallerTriangles : ℕ := 100

/-- The type of rhombus we're looking for -/
def targetRhombus : Rhombus :=
  { numTriangles := 8 }

/-- Function to calculate the number of rhombuses -/
def number_of_rhombuses (main : EquilateralTriangle) (small : EquilateralTriangle) 
                        (total : ℕ) (target : Rhombus) : ℕ :=
  -- Placeholder implementation
  84

/-- The theorem to be proved -/
theorem count_rhombuses_in_triangle :
  ∃ (n : ℕ), n = 84 ∧
    n = number_of_rhombuses mainTriangle smallerTriangles totalSmallerTriangles targetRhombus :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_rhombuses_in_triangle_l561_56164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_three_l561_56119

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x^3 - x^2 - 4*x + 4
  else x^3 - 7*x^2 + 16*x - 12

-- State the theorem
theorem sum_of_roots_is_three : 
  (∀ x : ℝ, f x = f (2 - x)) → 
  (∀ x : ℝ, x ≤ 1 → f x = x^3 - x^2 - 4*x + 4) → 
  ∃ r₁ r₂ r₃ : ℝ, 
    (f r₁ = 0 ∧ f r₂ = 0 ∧ f r₃ = 0) ∧ 
    (∀ x : ℝ, f x = 0 → x = r₁ ∨ x = r₂ ∨ x = r₃) ∧
    r₁ + r₂ + r₃ = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_three_l561_56119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_P_subset_Q_l561_56117

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | |x - a| < 4}
def Q : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Theorem statement
theorem range_of_a_for_P_subset_Q :
  {a : ℝ | ∀ x, x ∈ P a → x ∈ Q} = Set.Icc (-1) 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_P_subset_Q_l561_56117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_equal_and_real_l561_56184

-- Define the quadratic equation
noncomputable def quadratic_equation (x c : ℝ) : Prop :=
  3 * x^2 - 6 * x * Real.sqrt 3 + c = 0

-- Define the discriminant
noncomputable def discriminant (c : ℝ) : ℝ :=
  (-6 * Real.sqrt 3)^2 - 4 * 3 * c

-- Theorem statement
theorem roots_equal_and_real (c : ℝ) :
  discriminant c = 0 →
  ∃ x : ℝ, quadratic_equation x c ∧
    ∀ y : ℝ, quadratic_equation y c → y = x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_equal_and_real_l561_56184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l561_56155

/-- A point in the coordinate plane with integer coordinates -/
structure Point where
  x : Int
  y : Int

/-- A triangle formed by three points -/
structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

/-- Check if a point is within the 6x6 grid -/
def inGrid (p : Point) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 6 ∧ 1 ≤ p.y ∧ p.y ≤ 6

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

/-- Check if a triangle has positive area -/
def hasPositiveArea (t : Triangle) : Prop :=
  inGrid t.p1 ∧ inGrid t.p2 ∧ inGrid t.p3 ∧ ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with positive area in the 6x6 grid -/
def validTriangles : Set Triangle :=
  {t : Triangle | hasPositiveArea t}

/-- Assume the set of valid triangles is finite -/
instance : Fintype validTriangles := sorry

/-- The main theorem stating the number of valid triangles -/
theorem count_valid_triangles : Fintype.card validTriangles = 6778 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_triangles_l561_56155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_congruent_1_mod_500_l561_56196

open BigOperators

def S : ℤ := ∑ n in Finset.range 334, (-1)^n * (Nat.choose 1000 (3*n))

theorem S_congruent_1_mod_500 : S ≡ 1 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_congruent_1_mod_500_l561_56196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_theorem_specific_resistors_theorem_l561_56122

/-- The combined resistance of resistors in parallel -/
noncomputable def combined_resistance (x y z w : ℝ) : ℝ :=
  1 / (1/x + 1/y + 1/z + 1/w)

theorem parallel_resistors_theorem (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  combined_resistance x y z w = (x * y * z * w) / (y * z * w + x * z * w + x * y * w + x * y * z) :=
by sorry

theorem specific_resistors_theorem :
  combined_resistance 5 7 3 9 = 315 / 248 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_resistors_theorem_specific_resistors_theorem_l561_56122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_luke_bus_time_to_work_l561_56109

/-- Luke's travel time to work by bus in minutes -/
noncomputable def L : ℝ := Real.pi -- Using an arbitrary real number as a placeholder

/-- Paula's travel time to work by bus in minutes -/
noncomputable def paula_to_work : ℝ := (3/5) * L

/-- Luke's travel time home by bike in minutes -/
noncomputable def luke_home_bike : ℝ := 5 * L

/-- Paula's travel time home by bus in minutes -/
noncomputable def paula_home_bus : ℝ := paula_to_work

/-- Total travel time for both Luke and Paula in minutes -/
def total_time : ℝ := 504

theorem luke_bus_time_to_work : 
  L + paula_to_work + luke_home_bike + paula_home_bus = total_time → L = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_luke_bus_time_to_work_l561_56109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_cross_not_foldable_l561_56156

/-- Represents a pattern of squares -/
inductive Pattern
| ExtendedCross
| LShape

/-- Represents a square -/
structure Square

/-- Represents a cube -/
structure Cube where
  faces : Fin 6 → Square
  congruent : ∀ i j, faces i = faces j

/-- Represents a folding operation -/
def fold (p : Pattern) : Option Cube :=
  sorry

/-- Theorem: An extended cross pattern cannot be folded into a cube -/
theorem extended_cross_not_foldable : 
  fold Pattern.ExtendedCross = none := by
  sorry

#check extended_cross_not_foldable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_cross_not_foldable_l561_56156
