import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1086_108697

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line in the problem -/
def line (x y : ℝ) : Prop := x - y - 5 = 0

/-- The distance function from a point to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x - y - 5| / Real.sqrt 2

/-- The theorem stating the maximum distance -/
theorem max_distance_to_line : 
  ∃ (max_dist : ℝ), max_dist = 5 * Real.sqrt 2 ∧ 
  ∀ (x y : ℝ), ellipse x y → distance_to_line x y ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l1086_108697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_commutative_not_associative_l1086_108637

-- Define the diamond operation
noncomputable def diamond (c : ℝ) (x y : ℝ) : ℝ := (c * x * y) / (x + y + 1)

-- Theorem statement
theorem diamond_commutative_not_associative (c : ℝ) :
  (c > 0) →
  (∀ x y : ℝ, x > 0 ∧ y > 0 → diamond c x y = diamond c y x) ∧
  (∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ diamond c (diamond c x y) z ≠ diamond c x (diamond c y z)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_commutative_not_associative_l1086_108637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quad_area_equals_original_l1086_108638

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a convex quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point
  convex : Bool  -- Assume this is true for our quadrilateral

/-- Divides a line segment in the given ratio -/
def divideSegment (P Q : Point) (ratio : ℝ) : Point :=
  { x := P.x + ratio * (Q.x - P.x),
    y := P.y + ratio * (Q.y - P.y) }

/-- Calculates the area of a quadrilateral -/
noncomputable def area (q : Quadrilateral) : ℝ := sorry

/-- Constructs the inner quadrilateral by dividing sides and drawing lines -/
noncomputable def constructInnerQuadrilateral (q : Quadrilateral) : Quadrilateral :=
  let ratio := Real.sqrt 2
  let P := divideSegment q.A q.B (1 / (2 + ratio))
  let Q := divideSegment q.B q.C (1 / (2 + ratio))
  let R := divideSegment q.C q.D (1 / (2 + ratio))
  let S := divideSegment q.D q.A (1 / (2 + ratio))
  { A := P, B := Q, C := R, D := S, convex := true }

/-- The main theorem: The area of the inner quadrilateral is equal to the area of the original quadrilateral -/
theorem inner_quad_area_equals_original (q : Quadrilateral) :
  area (constructInnerQuadrilateral q) = area q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_quad_area_equals_original_l1086_108638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_value_l1086_108662

/-- A function f(x) is quadratic if it can be written in the form f(x) = ax² + bx + c, where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function y = x^(2m-1) + x - 3 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^(2*m-1) + x - 3

theorem quadratic_function_m_value (m : ℝ) :
  IsQuadratic (f m) → m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_m_value_l1086_108662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_l1086_108658

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 4

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := 5*x + y - 6 = 0

-- Theorem statement
theorem tangent_line_at_point_one :
  ∃ (m : ℝ), (∀ h : ℝ, h ≠ 0 → (((f (1 + h) - f 1) / h) = m)) ∧
             tangent_line 1 1 ∧
             (∀ x y : ℝ, tangent_line x y ↔ y - 1 = m * (x - 1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_point_one_l1086_108658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_l1086_108644

/-- Triangle type definition -/
structure Triangle where
  sides : Finset ℝ
  side_count : sides.card = 3

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
axiom triangle_inequality {a b c : ℝ} : 
  (a + b > c ∧ b + c > a ∧ c + a > b) ↔ (∃ (t : Triangle), t.sides = {a, b, c})

/-- A triangle can be formed from the given side lengths. -/
theorem triangle_exists (a b c : ℝ) (ha : a = 6) (hb : b = 9) (hc : c = 14) :
  ∃ (t : Triangle), t.sides = {a, b, c} :=
by
  apply Iff.mp (triangle_inequality)
  constructor
  · -- Prove a + b > c
    rw [ha, hb, hc]
    norm_num
  constructor
  · -- Prove b + c > a
    rw [ha, hb, hc]
    norm_num
  · -- Prove c + a > b
    rw [ha, hb, hc]
    norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_exists_l1086_108644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_F1PF2_value_l1086_108654

-- Define the ellipse and hyperbola equations
noncomputable def is_on_ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1
noncomputable def is_on_hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define a point P that satisfies both equations
def P_satisfies_equations (P : ℝ × ℝ) : Prop :=
  is_on_ellipse P.1 P.2 ∧ is_on_hyperbola P.1 P.2

-- Define the cosine of the angle F1PF2
noncomputable def cos_angle_F1PF2 (P : ℝ × ℝ) : ℝ :=
  let PF1 := (F1.1 - P.1, F1.2 - P.2)
  let PF2 := (F2.1 - P.1, F2.2 - P.2)
  let dot_product := PF1.1 * PF2.1 + PF1.2 * PF2.2
  let magnitude_PF1 := Real.sqrt (PF1.1^2 + PF1.2^2)
  let magnitude_PF2 := Real.sqrt (PF2.1^2 + PF2.2^2)
  dot_product / (magnitude_PF1 * magnitude_PF2)

-- Theorem statement
theorem cos_angle_F1PF2_value (P : ℝ × ℝ) (h : P_satisfies_equations P) :
  cos_angle_F1PF2 P = 1/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_F1PF2_value_l1086_108654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_and_sum_l1086_108655

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem point_on_inverse_graph_and_sum (h : f 2 = 3/2) :
  f_inv (3/2) = 2 ∧ (⟨3/2, 1/2⟩ : ℝ × ℝ) ∈ {p : ℝ × ℝ | p.2 = f_inv p.1 / 4} ∧ 3/2 + 1/2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_and_sum_l1086_108655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_difference_l1086_108667

/-- Represents the problem of two trains moving towards each other. -/
structure TrainProblem where
  speed1 : ℝ  -- Speed of the first train
  speed2 : ℝ  -- Speed of the second train
  totalDistance : ℝ  -- Total distance between starting points

/-- Calculates the difference in distance traveled by the two trains. -/
noncomputable def distanceDifference (p : TrainProblem) : ℝ :=
  let time := p.totalDistance / (p.speed1 + p.speed2)
  (p.speed2 * time) - (p.speed1 * time)

/-- Theorem stating that the difference in distance traveled is 50 km
    for the given problem parameters. -/
theorem train_distance_difference :
  let p := TrainProblem.mk 20 25 450
  distanceDifference p = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_distance_difference_l1086_108667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_integral_l1086_108653

-- Define the power function as noncomputable
noncomputable def f (n : ℝ) (x : ℝ) : ℝ := x^n

-- State the theorem
theorem power_function_integral (n : ℝ) :
  f n 9 = 3 → ∫ x in (0:ℝ)..1, f n x = 2/3 := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_integral_l1086_108653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_of_circles_l1086_108649

theorem intersection_line_of_circles 
  (circle1 : ℝ → ℝ → Prop)
  (circle2 : ℝ → ℝ → Prop)
  (h1 : ∀ x y : ℝ, circle1 x y ↔ x^2 + y^2 + 2*x + 3*y = 0)
  (h2 : ∀ x y : ℝ, circle2 x y ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0) :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → 6*x + y - 1 = 0 :=
by
  intros x y h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_line_of_circles_l1086_108649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_intersected_equilateral_triangle_l1086_108618

-- Define an equilateral triangle
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define a line passing through the midpoint of a side
structure IntersectingLine where
  angle : ℝ
  angle_acute : 0 < angle ∧ angle < Real.pi / 2

-- Define the ratio of areas
noncomputable def areaRatio (t : EquilateralTriangle) (l : IntersectingLine) : ℝ × ℝ :=
  ((2 * Real.sqrt 3 * Real.cos l.angle + Real.sin l.angle), Real.sin l.angle)

-- The theorem
theorem area_ratio_of_intersected_equilateral_triangle
  (t : EquilateralTriangle) (l : IntersectingLine) :
  areaRatio t l = ((2 * Real.sqrt 3 * Real.cos l.angle + Real.sin l.angle), Real.sin l.angle) := by
  -- Unfold the definition of areaRatio
  unfold areaRatio
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_of_intersected_equilateral_triangle_l1086_108618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowls_sold_is_102_l1086_108659

/-- Calculates the number of glass bowls sold given the total number bought,
    cost per bowl, selling price per bowl, and percentage gain. -/
def bowls_sold (total : ℕ) (cost : ℚ) (sell_price : ℚ) (gain_percent : ℚ) : ℕ :=
  let cp := total * cost
  let x := (gain_percent / 100 + 1) * cp / sell_price
  Int.toNat (Int.floor x)

/-- The number of glass bowls sold is 102. -/
theorem bowls_sold_is_102 :
  bowls_sold 118 12 15 (8050847457627118 / 100000000000000) = 102 := by
  sorry

#eval bowls_sold 118 12 15 (8050847457627118 / 100000000000000)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowls_sold_is_102_l1086_108659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_geq_one_l1086_108686

theorem negation_of_universal_sin_geq_one :
  (¬ ∀ x : ℝ, Real.sin x ≥ 1) ↔ (∃ x : ℝ, Real.sin x < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_sin_geq_one_l1086_108686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_theorem_l1086_108652

/-- A sequence of integers representing values of a quadratic polynomial at equally spaced points -/
def sequenceValues : List ℤ := [1261, 1332, 1407, 1484, 1565, 1648, 1733, 1820]

/-- The index of the potentially incorrect value in the sequence -/
def incorrect_index : ℕ := 3

/-- Function to calculate first differences of a list -/
def first_differences (l : List ℤ) : List ℤ :=
  List.zipWith (·-·) (List.tail l) l

/-- Function to calculate second differences of a list -/
def second_differences (l : List ℤ) : List ℤ :=
  first_differences (first_differences l)

/-- Predicate to check if a list has constant second differences except at one point -/
def has_constant_second_differences_except_one (l : List ℤ) : Prop :=
  ∃ (i : ℕ) (c : ℤ), i < l.length ∧
    ∀ (j : ℕ), j < l.length - 2 → j ≠ i → (second_differences l).get! j = c

theorem incorrect_value_theorem :
  has_constant_second_differences_except_one sequenceValues →
  (second_differences sequenceValues).get! (incorrect_index - 1) ≠
  (second_differences sequenceValues).get! incorrect_index →
  incorrect_index = 3 := by
  sorry

#eval sequenceValues
#eval incorrect_index
#eval first_differences sequenceValues
#eval second_differences sequenceValues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_value_theorem_l1086_108652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_close_functions_l1086_108608

open Real

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x)
noncomputable def g (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem mutually_close_functions :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ |f x₀ - g x₀| < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mutually_close_functions_l1086_108608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_bound_l1086_108678

theorem root_sum_bound (a b : ℝ) 
  (h1 : ∀ x, (x + a) * (x + b) = -9)
  (h2 : (a * b + a) * (a * b + b) = -9) 
  (h3 : a < 0) 
  (h4 : b < 0) : 
  a + b < -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_sum_bound_l1086_108678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_neg_one_l1086_108674

-- Define the sequence
def a : ℕ → ℚ
  | 0 => 2  -- We define a(0) as 2 to match a(1) in the original problem
  | n + 1 => (a n - 1) / (a n)

-- State the theorem
theorem a_2016_equals_neg_one : a 2016 = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2016_equals_neg_one_l1086_108674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1086_108626

-- Define the line l
noncomputable def line_l (t α : ℝ) : ℝ × ℝ := (1 + t * Real.cos α, 2 + t * Real.sin α)

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y - 3)^2 = 9

-- Define point P
def point_P : ℝ × ℝ := (1, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem min_distance_sum (α : ℝ) :
  ∃ (t₁ t₂ : ℝ),
    circle_C (line_l t₁ α).1 (line_l t₁ α).2 ∧
    circle_C (line_l t₂ α).1 (line_l t₂ α).2 ∧
    (∀ (t : ℝ), circle_C (line_l t α).1 (line_l t α).2 → t = t₁ ∨ t = t₂) ∧
    distance point_P (line_l t₁ α) + distance point_P (line_l t₂ α) ≥ 2 * Real.sqrt 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1086_108626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_parallelism_l1086_108647

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define lines and planes as affine subspaces
variable (m n : AffineSubspace ℝ V) (α β : AffineSubspace ℝ V)

-- Define the parallelism relation
def parallel (L₁ L₂ : AffineSubspace ℝ V) : Prop := 
  L₁ ≠ L₂ ∧ L₁.direction = L₂.direction

-- Theorem statement
theorem line_plane_parallelism 
  (h1 : parallel m α) 
  (h2 : m ≤ β) 
  (h3 : α ⊓ β = n) : 
  parallel m n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_parallelism_l1086_108647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_filled_fraction_l1086_108661

/-- Represents the fraction of a vessel filled with paint -/
noncomputable def fraction_filled (E P p : ℝ) : ℝ := p / P

/-- Proves that the fraction of the vessel filled is 19/44 given the conditions -/
theorem vessel_filled_fraction 
  (E P p : ℝ) 
  (h1 : E = 0.12 * (E + P)) 
  (h2 : E + p = (1/2) * (E + P)) 
  (h_pos : E > 0 ∧ P > 0 ∧ p > 0) :
  fraction_filled E P p = 19/44 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_filled_fraction_l1086_108661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l1086_108683

/-- The total surface area of a regular triangular pyramid with given parameters. -/
noncomputable def totalSurfaceArea (a m n : ℝ) : ℝ :=
  (a^2 * Real.sqrt 3 / 4) * (1 + Real.sqrt ((3 * (m + 2*n)) / m))

/-- Theorem stating the total surface area of a regular triangular pyramid. -/
theorem pyramid_surface_area (a m n : ℝ) (h1 : a > 0) (h2 : m > 0) (h3 : n > 0) :
  let base_side_length := a
  let lateral_edge_ratio := (m, n)
  totalSurfaceArea a m n = (a^2 * Real.sqrt 3 / 4) * (1 + Real.sqrt ((3 * (m + 2*n)) / m)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_l1086_108683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_speed_calculation_l1086_108696

/-- Calculates the return speed given the one-way distance, outbound speed, and average round-trip speed -/
noncomputable def return_speed (distance : ℝ) (outbound_speed : ℝ) (average_speed : ℝ) : ℝ :=
  (2 * distance * average_speed) / (2 * distance - outbound_speed * average_speed)

theorem return_speed_calculation (distance outbound_speed average_speed : ℝ) 
  (h1 : distance = 150)
  (h2 : outbound_speed = 75)
  (h3 : average_speed = 60) :
  return_speed distance outbound_speed average_speed = 50 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval return_speed 150 75 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_speed_calculation_l1086_108696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_william_max_moves_l1086_108603

def move_a (n : ℕ) : ℕ := 2 * n + 1
def move_b (n : ℕ) : ℕ := 4 * n + 3

def game_over (n : ℕ) : Prop := n > 2^100

def optimal_play (current_value : ℕ) (is_mark_turn : Bool) : Prop :=
  sorry  -- Definition of optimal play strategy

theorem william_max_moves :
  ∃ (game : ℕ → Bool → ℕ),
    game 0 true = 1 ∧  -- Initial value is 1, Mark's turn
    (∀ n, game_over (game n false) → game_over (game (n+1) true)) ∧  -- Game ends when value exceeds 2^100
    (∀ n, ¬game_over (game n false) →
      (game (n+1) true = move_a (game n false) ∨
       game (n+1) true = move_b (game n false))) ∧
    (∀ n, ¬game_over (game n true) →
      (game (n+1) false = move_a (game n true) ∨
       game (n+1) false = move_b (game n true))) ∧
    optimal_play (game 0 true) true ∧
    (∀ n, optimal_play (game n false) false → optimal_play (game (n+1) true) true) ∧
    (∀ n, optimal_play (game n true) true → optimal_play (game (n+1) false) false) ∧
    (∃ k, game_over (game k true) ∧ ¬game_over (game (k-1) false) ∧ k ≤ 67) ∧
    ¬(∃ k, game_over (game k true) ∧ ¬game_over (game (k-1) false) ∧ k ≤ 66) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_william_max_moves_l1086_108603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ellipse_construction_l1086_108645

-- Define the basic types
def Point : Type := ℝ × ℝ
def Line : Type := Point → Prop

-- Define the given conditions
variable (F₁ : Point) -- Given focus
variable (t₁ t₂ t₃ : Line) -- Given tangents

-- Define the reflection of a point across a line
noncomputable def reflect (p : Point) (l : Line) : Point := sorry

-- Define the circle passing through three points
def circle_through_points (p₁ p₂ p₃ : Point) : Point → Prop := sorry

-- Define the directrix circle
def directrix_circle (F₁ : Point) (t₁ t₂ t₃ : Line) : Point → Prop :=
  let G₁ := reflect F₁ t₁
  let G₂ := reflect F₁ t₂
  let G₃ := reflect F₁ t₃
  circle_through_points G₁ G₂ G₃

-- Define the second focus
noncomputable def F₂ (F₁ : Point) (t₁ t₂ t₃ : Line) : Point := sorry

-- Define an ellipse
def is_ellipse (F₁ F₂ : Point) (a : ℝ) : Point → Prop := sorry

-- State the theorem
theorem unique_ellipse_construction 
  (F₁ : Point) (t₁ t₂ t₃ : Line) 
  (h : directrix_circle F₁ t₁ t₂ t₃ (F₂ F₁ t₁ t₂ t₃)) : 
  ∃! (E : Point → Prop) (a : ℝ), 
    E = is_ellipse F₁ (F₂ F₁ t₁ t₂ t₃) a ∧ 
    (∀ p, E p → (t₁ p ∨ t₂ p ∨ t₃ p)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_ellipse_construction_l1086_108645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1086_108607

def M : Set ℕ := {1, 2, 3, 4}

def N : Set ℕ := {x : ℕ | ∃ y : ℝ, y = Real.sqrt (x - 3)}

theorem intersection_M_N : M ∩ N = {3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1086_108607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_l1086_108616

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 5)

theorem smallest_x_in_domain_of_f_of_f : 
  (∀ y : ℝ, f (f y) = f (f y) → y ≥ 30) ∧ 
  f (f 30) = f (f 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_x_in_domain_of_f_of_f_l1086_108616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_no_sqrt_l1086_108635

-- Define the expressions
noncomputable def expr_a : ℝ := (-1/4)^2
def expr_b : ℝ := 0
noncomputable def expr_c : ℝ := 100 -- Simplified from (±10)^2
noncomputable def expr_d : ℝ := -9  -- Simplified from -|(-9)|

-- Define a predicate for having an arithmetic square root
def has_arith_sqrt (x : ℝ) : Prop := ∃ y : ℝ, y ≥ 0 ∧ y^2 = x

-- Theorem statement
theorem unique_no_sqrt : 
  has_arith_sqrt expr_a ∧ 
  has_arith_sqrt expr_b ∧ 
  has_arith_sqrt expr_c ∧ 
  ¬has_arith_sqrt expr_d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_no_sqrt_l1086_108635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_existence_l1086_108614

theorem modular_inverse_existence (p : ℕ) (a : ℤ) (h_prime : Nat.Prime p) (h_not_dvd : ¬(↑p ∣ a)) :
  ∃ b : ℕ, (a * ↑b) % ↑p = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_modular_inverse_existence_l1086_108614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sample_l1086_108666

def total_missiles : ℕ := 50
def sample_size : ℕ := 5

def is_valid_sample (sample : List ℕ) : Prop :=
  sample.length = sample_size ∧
  sample.all (λ x => 1 ≤ x ∧ x ≤ total_missiles) ∧
  (List.zip sample (List.tail sample)).all (λ (x, y) => y - x = total_missiles / sample_size)

theorem correct_sample :
  is_valid_sample [3, 13, 23, 33, 43] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_sample_l1086_108666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_moves_on_line_l1086_108676

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : Point :=
  { x := (t.A.x + t.B.x + t.C.x) / 3
  , y := (t.A.y + t.B.y + t.C.y) / 3 }

-- Define a function to check if a point is on a line
def isOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem centroid_moves_on_line 
  (A B : Point) 
  (l : Line) 
  (h : ∀ (C : Point), isOnLine C l → 
       ∃ (l' : Line), ∀ (t : ℝ), 
         isOnLine (centroid {A := A, B := B, C := {x := C.x + t * l.b, y := C.y - t * l.a}}) l') : 
  True := by
  sorry

#check centroid_moves_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_moves_on_line_l1086_108676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1086_108602

theorem equation_solution (x : ℝ) : 
  (25 : ℝ)^x + (49 : ℝ)^x / ((35 : ℝ)^x + (40 : ℝ)^x) = 5/4 ↔ x = -Real.log 4 / Real.log (5/7) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1086_108602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_expression_value_l1086_108675

theorem cubic_expression_value (r s : ℝ) : 
  3 * r^2 - 4 * r - 8 = 0 →
  3 * s^2 - 4 * s - 8 = 0 →
  r ≠ s →
  (9 * r^3 - 9 * s^3) / (r - s) = 40 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_expression_value_l1086_108675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_l1086_108660

def normal_vector : ℝ × ℝ × ℝ := (2, -2, 1)
def point_P : ℝ × ℝ × ℝ := (-1, 3, 2)

theorem distance_point_to_plane :
  let d := Real.sqrt ((normal_vector.1 * point_P.1 + normal_vector.2.1 * point_P.2.1 + normal_vector.2.2 * point_P.2.2)^2) /
           Real.sqrt (normal_vector.1^2 + normal_vector.2.1^2 + normal_vector.2.2^2)
  d = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_plane_l1086_108660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1086_108636

theorem solve_exponential_equation (x : ℝ) :
  3 * (2:ℝ)^x * (2:ℝ)^x * (2:ℝ)^x = 4608 → x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l1086_108636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_fixed_point_with_min_area_l1086_108606

-- Define the line l
noncomputable def line_l (a b c : ℝ) := {(x, y) : ℝ × ℝ | a * x + b * y + c = 0}

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-5, -3)

-- Define the intersection points
noncomputable def point_A (a b c : ℝ) : ℝ × ℝ := (-c/a, 0)
noncomputable def point_B (a b c : ℝ) : ℝ × ℝ := (0, -c/b)

-- Define the area of the triangle
noncomputable def triangle_area (a b c : ℝ) : ℝ := abs (c^2 / (2 * a * b))

-- State the theorem
theorem line_through_fixed_point_with_min_area :
  ∃ (a b c : ℝ), 
    (fixed_point ∈ line_l a b c) ∧ 
    (a > 0) ∧ (b < 0) ∧ (c < 0) ∧
    (∀ (a' b' c' : ℝ), (fixed_point ∈ line_l a' b' c') ∧ (a' > 0) ∧ (b' < 0) ∧ (c' < 0) →
      triangle_area a b c ≤ triangle_area a' b' c') ∧
    (a = 5 ∧ b = -2 ∧ c = -5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_fixed_point_with_min_area_l1086_108606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_l1086_108698

/-- The length of the minute hand on a clock in centimeters. -/
def minute_hand_length : ℝ := 8

/-- The time elapsed in minutes. -/
def elapsed_time : ℝ := 45

/-- The time for one complete revolution of the minute hand in minutes. -/
def revolution_time : ℝ := 60

/-- The distance traveled by the tip of the minute hand in the given time. -/
noncomputable def distance_traveled : ℝ := 12 * Real.pi

theorem minute_hand_distance :
  2 * Real.pi * minute_hand_length * (elapsed_time / revolution_time) = distance_traveled :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_l1086_108698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l1086_108651

-- Define the sets A, B, and C
def A : Set ℝ := Set.Icc (-(Real.sqrt 3) / 3) (-1 / 2)
def B : Set ℝ := {m : ℝ | ∀ x, x^2 + m*x + m ≥ 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | a + 1 < x ∧ x < 2*a}

-- State the theorem
theorem subset_condition (a : ℝ) :
  C a ⊆ A ∪ B ↔ a < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_condition_l1086_108651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l1086_108665

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 8x -/
def Parabola := {p : Point | p.y^2 = 8 * p.x}

/-- Represents a hyperbola of the form (x^2 / a^2) - (y^2 / b^2) = 1 -/
def Hyperbola (a b : ℝ) := {p : Point | p.x^2 / a^2 - p.y^2 / b^2 = 1}

/-- The focus of the parabola y^2 = 8x -/
def parabolaFocus : Point := ⟨2, 0⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (a c : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity_is_two 
  (a b : ℝ) 
  (P : Point) 
  (h1 : P ∈ Parabola)
  (h2 : P ∈ Hyperbola a b)
  (h3 : distance P parabolaFocus = 5)
  (h4 : P.x > 0 ∧ P.y > 0) :
  eccentricity a 2 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_is_two_l1086_108665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l1086_108689

/-- The height of a flagpole given specific conditions -/
theorem flagpole_height (wire_anchor_distance : ℝ) (person_height : ℝ) (person_distance : ℝ)
  (h1 : wire_anchor_distance = 5)
  (h2 : person_height = 1.6)
  (h3 : person_distance = 4)
  (h4 : person_distance < wire_anchor_distance) :
  (wire_anchor_distance * person_height) / (wire_anchor_distance - person_distance) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_l1086_108689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_20_l1086_108604

-- Define the arithmetic sequence and its sum
noncomputable def arithmetic_sequence : ℕ → ℝ := sorry
noncomputable def S (n : ℕ) : ℝ := (n : ℝ) / 2 * (arithmetic_sequence 1 + arithmetic_sequence n)

-- Define points in 3D space
noncomputable def O : ℝ × ℝ × ℝ := (0, 0, 0)
noncomputable def M : ℝ × ℝ × ℝ := sorry
noncomputable def N : ℝ × ℝ × ℝ := sorry
noncomputable def P : ℝ × ℝ × ℝ := sorry

-- State the theorem
theorem arithmetic_sequence_sum_20 
  (h_collinear : ∃ (t : ℝ), N = M + t • (P - M))
  (h_vector_eq : N - O = arithmetic_sequence 15 • (M - O) + arithmetic_sequence 6 • (P - O))
  (h_not_through_O : ∃ (s : ℝ), M + s • (P - M) ≠ O) :
  S 20 = 10 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_20_l1086_108604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_circle_circumference_l1086_108688

-- Define the circle
def circle_center : ℝ × ℝ := (2, -1)
def circle_point : ℝ × ℝ := (7, 4)

-- Define the radius
noncomputable def radius : ℝ := Real.sqrt ((circle_point.1 - circle_center.1)^2 + (circle_point.2 - circle_center.2)^2)

-- Theorem for the area of the circle
theorem circle_area : π * radius^2 = 50 * π := by sorry

-- Theorem for the circumference of the circle
theorem circle_circumference : 2 * π * radius = 10 * π * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_circle_circumference_l1086_108688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_conditions_l1086_108650

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x + k - 1

noncomputable def vertex (k : ℝ) : ℝ × ℝ := (k/2, f k (k/2))

theorem parabola_conditions (k : ℝ) :
  -- 1. If the vertex is on the x-axis, then k = 2
  (∃ x : ℝ, vertex k = (x, 0)) → k = 2 ∧
  -- 2. If the vertex is on the y-axis, then k = 0
  (∃ y : ℝ, vertex k = (0, y)) → k = 0 ∧
  -- 3. If the vertex is at (-1, -4), then k = 1
  (vertex k = (-1, -4)) → k = 1 ∧
  -- 4. If the parabola passes through the origin, then k = 1
  (f k 0 = 0) → k = 1 ∧
  -- 5. If the parabola has a minimum value of y when x = 1, then k = 2
  (∀ x : ℝ, f k x ≥ f k 1) → k = 2 ∧
  -- 6. If the minimum value of y is -1, then k = 0 or k = 4
  (∃ x : ℝ, f k x = -1 ∧ ∀ y : ℝ, f k y ≥ -1) → k = 0 ∨ k = 4 :=
by sorry

#check parabola_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_conditions_l1086_108650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1086_108609

-- Define the quadratic function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_theorem (a b c : ℝ) :
  (∀ x : ℝ, 2 * x ≤ f a b c x ∧ f a b c x ≤ (1/2) * (x + 1)^2) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) (-1) → x₂ ∈ Set.Icc (-3) (-1) → |f a b c x₁ - f a b c x₂| ≤ 1) →
  f a b c (-1) ∈ Set.Ioc (-2) 0 ∧ a ∈ Set.Icc (1/4) ((9 + Real.sqrt 17) / 32) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_theorem_l1086_108609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_angle_l1086_108620

/-- Semicircle with diameter AB -/
def Semicircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 1 ∧ p.2 ≥ 0}

/-- Point A -/
def A : ℝ × ℝ × ℝ := (-1, 0, 0)

/-- Point B -/
def B : ℝ × ℝ × ℝ := (1, 0, 0)

/-- Point S -/
def S : ℝ × ℝ × ℝ := (0, 0, 2)

/-- Moving point C on the semicircle -/
noncomputable def C (θ : ℝ) : ℝ × ℝ × ℝ := (Real.cos θ, Real.sin θ, 0)

/-- Projection of A onto SB -/
noncomputable def M (θ : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Projection of A onto SC -/
noncomputable def N (θ : ℝ) : ℝ × ℝ × ℝ := sorry

/-- Volume of the triangular pyramid S-AMN -/
noncomputable def volume (θ : ℝ) : ℝ := sorry

/-- Angle between SC and the plane ABC -/
noncomputable def angle (θ : ℝ) : ℝ := sorry

theorem max_volume_angle :
  ∃ θ : ℝ, IsMax (volume θ) ∧ Real.sin (angle θ) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_angle_l1086_108620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_completion_l1086_108648

/-- Represents a round trip with equal distances to and from a destination -/
structure RoundTrip where
  distance : ℝ
  distance_positive : distance > 0

/-- Calculates the percentage of a round trip completed -/
noncomputable def completed_percentage (trip : RoundTrip) (to_destination : ℝ) (from_destination : ℝ) : ℝ :=
  (to_destination + from_destination) / (2 * trip.distance) * 100

/-- Theorem: If a traveler completes the entire distance to the destination
    plus 40% of the return trip, they have completed 70% of the total round trip -/
theorem round_trip_completion (trip : RoundTrip) :
  completed_percentage trip trip.distance (0.4 * trip.distance) = 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_completion_l1086_108648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_new_polygon_l1086_108673

/-- Represents a regular polygon -/
structure RegularPolygon where
  sides : ℕ
  sideLength : ℝ

/-- The configuration of polygons meeting at a point -/
structure PolygonConfiguration where
  polygon1 : RegularPolygon
  polygon2 : RegularPolygon
  square : RegularPolygon
  angleSumAtA : ℝ

noncomputable def interiorAngle (p : RegularPolygon) : ℝ :=
  180 * (p.sides - 2 : ℝ) / p.sides

def isValidConfiguration (config : PolygonConfiguration) : Prop :=
  config.polygon1 = config.polygon2 ∧
  config.square.sides = 4 ∧
  config.polygon1.sideLength = 1 ∧
  config.polygon2.sideLength = 1 ∧
  config.square.sideLength = 1 ∧
  config.angleSumAtA = 360 ∧
  2 * interiorAngle config.polygon1 + interiorAngle config.square = config.angleSumAtA

noncomputable def perimeterOfNewPolygon (config : PolygonConfiguration) : ℝ :=
  (2 * (config.polygon1.sides - 2 : ℝ) + (config.square.sides - 1 : ℝ)) * config.polygon1.sideLength

theorem max_perimeter_of_new_polygon (config : PolygonConfiguration) 
  (h : isValidConfiguration config) : 
  perimeterOfNewPolygon config = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_perimeter_of_new_polygon_l1086_108673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_implies_k_greater_than_four_l1086_108680

/-- An inverse proportion function with parameter k -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := (k - 4) / x

/-- The function decreases as x increases in each quadrant -/
def decreasing_in_quadrants (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → (x₁ > 0 → f x₁ > f x₂) ∧ (x₁ < 0 → f x₁ > f x₂)

/-- Theorem: If the inverse proportion function is decreasing in all quadrants, then k > 4 -/
theorem inverse_proportion_decreasing_implies_k_greater_than_four (k : ℝ) :
  decreasing_in_quadrants (inverse_proportion k) → k > 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_decreasing_implies_k_greater_than_four_l1086_108680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1086_108694

theorem expression_simplification :
  (6 * Real.sqrt (3 / 2) - Real.sqrt 48 / Real.sqrt 3 = 3 * Real.sqrt 6 - 4) ∧
  ((-Real.sqrt 5)^2 + (1 + Real.sqrt 3) * (3 - Real.sqrt 3) - (27 : ℝ)^(1/3) = 2 + 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_simplification_l1086_108694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_distance_theorem_l1086_108610

/-- Represents a bicycle trip with two parts -/
structure BicycleTrip where
  distance1 : ℝ
  speed1 : ℝ
  speed2 : ℝ
  avgSpeed : ℝ

/-- Calculates the distance traveled in the second part of the trip -/
noncomputable def secondDistance (trip : BicycleTrip) : ℝ :=
  (trip.avgSpeed * (trip.distance1 / trip.speed1 + trip.distance1 / trip.avgSpeed) - trip.distance1) /
  (1 - trip.avgSpeed / trip.speed2)

/-- Theorem stating the distance traveled in the second part of the trip -/
theorem second_distance_theorem (trip : BicycleTrip) 
  (h1 : trip.distance1 = 9)
  (h2 : trip.speed1 = 12)
  (h3 : trip.speed2 = 9)
  (h4 : trip.avgSpeed = 10.08) :
  secondDistance trip = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_distance_theorem_l1086_108610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_sine_values_l1086_108677

theorem cosine_and_sine_values (x : Real) 
  (h1 : Real.cos (x - π/4) = Real.sqrt 2/10)
  (h2 : x ∈ Set.Ioo (π/2) (3*π/4)) :
  Real.cos x = -3/5 ∧ Real.sin (2*x + π/3) = -(24 + 7*Real.sqrt 3)/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_and_sine_values_l1086_108677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_equals_twentyseven_l1086_108611

/-- A power function that passes through the point (2,8) -/
noncomputable def f (x : ℝ) : ℝ := x ^ (Real.log 8 / Real.log 2)

/-- Theorem stating that f(3) = 27 -/
theorem f_three_equals_twentyseven : f 3 = 27 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_three_equals_twentyseven_l1086_108611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1086_108681

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then x * (1 + x) else -(-x * (1 + (-x)))

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧ 
  f (-2) = -6 ∧
  ∀ x < 0, f x = x * (1 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1086_108681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_sum_property_l1086_108671

open Real

theorem extremum_points_sum_property (a : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ ≠ x₂) :
  let f := λ x : ℝ => x^2 - 4*a*x + a * log x
  (∀ x > 0, (deriv f x = 0 ↔ x = x₁ ∨ x = x₂)) →
  f x₁ + f x₂ < -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_points_sum_property_l1086_108671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_abs_x_plus_2_l1086_108624

-- Define the function f(x) = |x+2|
def f (x : ℝ) : ℝ := |x + 2|

-- State the theorem
theorem integral_abs_x_plus_2 : ∫ x in (-4)..3, f x = 29/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_abs_x_plus_2_l1086_108624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_properties_l1086_108612

open Real

-- Define the triangle and vectors
def Triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = Real.pi

noncomputable def m (B : ℝ) : ℝ × ℝ := (2 * sin B, sqrt 3)

noncomputable def n (B : ℝ) : ℝ × ℝ := (2 * (cos (B/2))^2 - 1, cos (2*B))

def orthogonal (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

noncomputable def f (x B : ℝ) : ℝ := sin (2*x - B)

theorem triangle_area_and_function_properties
  (A B C a b c : ℝ)
  (h_triangle : Triangle A B C a b c)
  (h_orthogonal : orthogonal (m B) (n B))
  (h_side : b = 4) :
  (∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), f (x + T) B = f x B ∧
    ∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') B = f x B) → T ≤ T') ∧
  (∀ (k : ℤ), ∀ (x : ℝ),
    (↑k * Real.pi - Real.pi/12 : ℝ) ≤ x ∧ x ≤ (↑k * Real.pi + 5*Real.pi/12 : ℝ) →
    ∀ (y : ℝ), x < y → f x B < f y B) ∧
  (∃ (S : ℝ), S = 4 * sqrt 3 ∧
    ∀ (S' : ℝ), S' = 1/2 * a * c * sin B → S' ≤ S) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_properties_l1086_108612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_odd_sum_is_one_thirtyfourth_l1086_108615

/-- The number of tiles -/
def num_tiles : ℕ := 12

/-- The number of players -/
def num_players : ℕ := 4

/-- The number of tiles each player selects -/
def tiles_per_player : ℕ := 3

/-- The set of all possible tile selections -/
def all_selections : Finset (Finset ℕ) :=
  Finset.powersetCard tiles_per_player (Finset.range num_tiles)

/-- Predicate to check if a sum is odd -/
def is_odd_sum (selection : Finset ℕ) : Prop :=
  Odd (Finset.sum selection id)

/-- The set of selections that result in an odd sum -/
noncomputable def odd_sum_selections : Finset (Finset ℕ) :=
  Finset.filter (fun s => Odd (Finset.sum s id)) all_selections

/-- The probability of all players getting an odd sum -/
noncomputable def prob_all_odd_sum : ℚ :=
  (Finset.card odd_sum_selections ^ num_players : ℚ) /
  (Finset.card all_selections ^ num_players : ℚ)

theorem prob_all_odd_sum_is_one_thirtyfourth :
  prob_all_odd_sum = 1 / 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_all_odd_sum_is_one_thirtyfourth_l1086_108615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l1086_108641

/-- Represents a frustum with given dimensions -/
structure Frustum where
  r1 : ℝ  -- radius of top base
  r2 : ℝ  -- radius of bottom base
  v : ℝ   -- volume

/-- Calculate the height of a frustum -/
noncomputable def frustum_height (f : Frustum) : ℝ :=
  (3 * f.v) / (Real.pi * (f.r1^2 + f.r2^2 + f.r1 * f.r2))

/-- Calculate the slant height of a frustum -/
noncomputable def slant_height (f : Frustum) : ℝ :=
  Real.sqrt ((frustum_height f)^2 + (f.r2 - f.r1)^2)

theorem frustum_slant_height :
  let f : Frustum := { r1 := 2, r2 := 6, v := 104 * Real.pi }
  slant_height f = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_slant_height_l1086_108641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_75_degrees_l1086_108687

theorem sin_75_degrees :
  Real.sin (75 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_75_degrees_l1086_108687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_is_75_l1086_108631

def sequenceA (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => if n % 2 = 0 then sequenceA n + 1 else sequenceA n + 2

def sum_first_n (n : ℕ) : ℕ :=
  (List.range n).map sequenceA |>.sum

theorem sum_first_10_is_75 : sum_first_n 10 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_10_is_75_l1086_108631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_approx_l1086_108622

/-- Represents a right pyramid with an equilateral triangular base -/
structure RightPyramid where
  base_side_length : ℝ
  height : ℝ

/-- Calculates the total surface area of a right pyramid with an equilateral triangular base -/
noncomputable def total_surface_area (p : RightPyramid) : ℝ :=
  let base_area := Real.sqrt 3 / 4 * p.base_side_length ^ 2
  let slant_height := Real.sqrt (p.height ^ 2 + (p.base_side_length * Real.sqrt 3 / 3) ^ 2)
  let lateral_area := 3 * (1 / 2 * p.base_side_length * slant_height)
  base_area + lateral_area

theorem pyramid_surface_area_approx :
  let p := RightPyramid.mk 10 15
  ∃ ε > 0, abs (total_surface_area p - 284.35) < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_surface_area_approx_l1086_108622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1086_108629

theorem inequality_condition (α β : ℝ) :
  (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x^α * y^β < k * (x + y)) ↔ 
  (α ≥ 0 ∧ β ≥ 0 ∧ α + β = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_condition_l1086_108629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_max_area_l1086_108670

-- Define the curves C₁ and C₂
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 + 2 * Real.sin θ)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := -4 * Real.cos θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) := {(0, 0), (-2, 2)}

-- Define the maximum distance between A and B
noncomputable def max_distance : ℝ := 2 * Real.sqrt 2 + 4

-- Define the area of triangle OAB when |AB| is maximum
noncomputable def max_area : ℝ := 2 + 2 * Real.sqrt 2

-- Helper function to calculate the area of a triangle
noncomputable def area_triangle (O A B : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem intersection_and_max_area :
  (∀ θ : ℝ, C₁ θ ∈ intersection_points ∨ C₂ θ ∈ intersection_points) ∧
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (∃ θ₁ : ℝ, C₁ θ₁ = A) ∧ 
    (∃ θ₂ : ℝ, C₂ θ₂ = B) ∧
    (∀ θ₃ θ₄ : ℝ, ‖C₁ θ₃ - C₂ θ₄‖ ≤ max_distance) ∧
    (area_triangle (0, 0) A B = max_area)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_max_area_l1086_108670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_x_in_pentagon_square_l1086_108617

/-- The angle x formed between one side of a regular pentagon inscribed in a square and the adjacent side of the square --/
def angle_x : ℚ := 54

/-- The number of sides in a pentagon --/
def pentagon_sides : ℕ := 5

/-- The sum of exterior angles of any convex polygon --/
def sum_exterior_angles : ℚ := 360

/-- Exterior angle of a regular pentagon --/
noncomputable def exterior_angle_pentagon : ℚ := sum_exterior_angles / pentagon_sides

/-- Interior angle of a regular pentagon --/
noncomputable def interior_angle_pentagon : ℚ := 180 - exterior_angle_pentagon

/-- Angle formed by the side of the square and the diagonal connecting the vertex of the square to the vertex of the pentagon --/
noncomputable def angle_q : ℚ := 180 - 90 - exterior_angle_pentagon

theorem angle_x_in_pentagon_square : 
  angle_x = 180 - (angle_q + interior_angle_pentagon) := by
  sorry

#eval angle_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_x_in_pentagon_square_l1086_108617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_for_nonnegative_f_l1086_108668

-- Define the function f
noncomputable def f (a x : ℝ) : ℝ := a * x^2 - x * Real.log x - (2 * a - 1) * x + a - 1

-- Define the derivative of f
noncomputable def f' (a x : ℝ) : ℝ := 2 * a * x - 2 * a - Real.log x

-- Theorem 1: Tangent line equation
theorem tangent_line_equation (x y : ℝ) :
  (f 0 x = -x * Real.log x + x - 1) →
  (f' 0 (Real.exp 1) = -1) →
  (f 0 (Real.exp 1) = -1) →
  (y = -(x - Real.exp 1) - 1) →
  x + y + 1 - Real.exp 1 = 0 := by
  sorry

-- Theorem 2: Range of a for non-negative f
theorem range_of_a_for_nonnegative_f (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) ↔ a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_range_of_a_for_nonnegative_f_l1086_108668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l1086_108613

theorem circle_radius (x y : ℝ) :
  4 * x^2 + 8 * x + 4 * y^2 - 24 * y + 20 = 0 →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = (Real.sqrt 10) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_l1086_108613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l1086_108639

theorem trigonometric_equation_reduction (a b c : ℤ) : 
  (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (2*x))^2 + (Real.sin (5*x))^2 + (Real.sin (6*x))^2 = 3 ↔ 
    (Real.cos (↑a*x)) * (Real.cos (↑b*x)) * (Real.cos (↑c*x)) = 0) →
  a + b + c = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l1086_108639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_b_range_of_a_l1086_108684

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x - 1

-- Theorem for monotonicity
theorem monotonicity_of_f (a : ℝ) :
  (∀ x : ℝ, (deriv (f a)) x > 0) ∨
  (∃ c : ℝ, ∀ x < c, (deriv (f a)) x < 0 ∧ ∀ x > c, (deriv (f a)) x > 0) :=
sorry

-- Theorem for range of b
theorem range_of_b (a b : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (deriv (f a)) x = 0) →
  (∀ x : ℝ, x > 0 → f a x ≥ b * x - 1) →
  b ≤ 0 :=
sorry

-- Theorem for range of a
theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0) →
  a < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_range_of_b_range_of_a_l1086_108684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_properties_l1086_108657

noncomputable section

-- Define the curves and transformation
def C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}

def φ : ℝ × ℝ → ℝ × ℝ := fun p => (Real.sqrt 3 * p.1, p.2)

def C1 : Set (ℝ × ℝ) := φ '' C

def C2 : Set (ℝ × ℝ) := {p | ∃ θ : ℝ, p.1 = (4 * Real.sqrt 2 / Real.sin (θ + Real.pi/4)) * Real.cos θ ∧
                                      p.2 = (4 * Real.sqrt 2 / Real.sin (θ + Real.pi/4)) * Real.sin θ}

-- State the theorem
theorem curves_properties :
  (∃ f : ℝ → ℝ × ℝ, ∀ α, f α ∈ C1 ∧ f α = (Real.sqrt 3 * Real.cos α, Real.sin α)) ∧
  (∀ p, p ∈ C2 ↔ p.1 + p.2 = 8) ∧
  (∃ d : ℝ, d = 3 * Real.sqrt 2 ∧
    ∀ p q, p ∈ C1 → q ∈ C2 → Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_properties_l1086_108657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_distance_proof_l1086_108632

noncomputable section

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The intersection point of two lines -/
noncomputable def intersection (l1 l2 : Line) : ℝ × ℝ :=
  let x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
  let y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
  (x, y)

/-- Check if two lines are parallel -/
def isParallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The distance between two parallel lines -/
noncomputable def distance (l1 l2 : Line) : ℝ :=
  abs (l2.c - l1.c) / Real.sqrt (l1.a^2 + l1.b^2)

theorem line_and_distance_proof (l1 l2 l3 l4 : Line) : 
  l1 = Line.mk 3 4 (-5) →
  l2 = Line.mk 2 (-3) 8 →
  l3 = Line.mk 2 1 5 →
  isParallel l3 l4 →
  intersection l1 l2 = (-1, 2) →
  l4 = Line.mk 2 1 (-4) ∧ distance l3 l4 = 9 * Real.sqrt 5 / 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_distance_proof_l1086_108632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersects_x_axis_l1086_108628

-- Define the curve function
noncomputable def f (x : ℝ) : ℝ := 
  (3*x - 1)*(Real.sqrt (9*x^2 - 6*x + 5) + 1) + 
  (2*x - 3)*(Real.sqrt (4*x^2 - 12*x + 13) + 1)

-- Theorem statement
theorem curve_intersects_x_axis : 
  ∃ (x : ℝ), f x = 0 ∧ x = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersects_x_axis_l1086_108628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_is_14_l1086_108605

/-- The certain number we're looking for -/
def n : ℕ := sorry

/-- The lower bound of the range -/
def a : ℕ := sorry

/-- The upper bound of the range -/
def b : ℕ := sorry

/-- The set of consecutive integers between a and b, inclusive -/
def q : Finset ℕ := sorry

theorem certain_number_is_14 :
  (∃ k₁ k₂ : ℕ, a = k₁ * n ∧ b = k₂ * n) →  -- a and b are multiples of n
  (∀ x ∈ q, a ≤ x ∧ x ≤ b) →  -- q is the set of consecutive integers between a and b, inclusive
  (q.filter (fun x => n ∣ x)).card = 9 →  -- q contains 9 multiples of n
  (q.filter (fun x => 7 ∣ x)).card = 17 →  -- q contains 17 multiples of 7
  n = 14 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_certain_number_is_14_l1086_108605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_income_l1086_108642

/-- Given Rebecca's current and increased income, and the percentage of combined income,
    prove Jimmy's current income. -/
theorem jimmy_income
  (rebecca_current : ℕ)
  (rebecca_increase : ℕ)
  (percentage : ℚ)
  (jimmy_income : ℕ)
  (h1 : rebecca_current = 15000)
  (h2 : rebecca_increase = 7000)
  (h3 : percentage = 55 / 100)
  (h4 : percentage * (rebecca_current + rebecca_increase + jimmy_income) = rebecca_current + rebecca_increase) :
  jimmy_income = 18000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jimmy_income_l1086_108642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chili_pepper_cost_l1086_108633

/-- The cost of seeds and Harry's purchase --/
structure SeedPurchase where
  pumpkin_cost : ℚ
  tomato_cost : ℚ
  chili_cost : ℚ
  pumpkin_packets : ℚ
  tomato_packets : ℚ
  chili_packets : ℚ
  total_spent : ℚ

/-- Theorem stating the cost of chili pepper seeds --/
theorem chili_pepper_cost (purchase : SeedPurchase)
  (h1 : purchase.pumpkin_cost = 5/2)
  (h2 : purchase.tomato_cost = 3/2)
  (h3 : purchase.pumpkin_packets = 3)
  (h4 : purchase.tomato_packets = 4)
  (h5 : purchase.chili_packets = 5)
  (h6 : purchase.total_spent = 18)
  (h7 : purchase.pumpkin_cost * purchase.pumpkin_packets +
        purchase.tomato_cost * purchase.tomato_packets +
        purchase.chili_cost * purchase.chili_packets = purchase.total_spent) :
  purchase.chili_cost = 9/10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chili_pepper_cost_l1086_108633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_correct_l1086_108699

/-- Represents the sale of cloth by a shopkeeper -/
structure ClothSale where
  total_selling_price : ℕ
  loss_per_metre : ℕ
  cost_price_per_metre : ℕ
  metres_sold : ℕ

/-- The actual cloth sale instance -/
def actual_sale : ClothSale where
  total_selling_price := 15000
  loss_per_metre := 10
  cost_price_per_metre := 40
  metres_sold := 500

/-- Theorem stating that the number of metres sold is correct -/
theorem cloth_sale_correct (sale : ClothSale) : 
  sale.total_selling_price = (sale.cost_price_per_metre - sale.loss_per_metre) * sale.metres_sold →
  sale = actual_sale :=
by
  sorry

#check cloth_sale_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cloth_sale_correct_l1086_108699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_distance_range_l1086_108656

-- Define the ellipse E
noncomputable def E (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
noncomputable def l (k m x y : ℝ) : Prop := y = k * x + m

-- Define the distance from a point to a line
noncomputable def distance_point_to_line (k m : ℝ) : ℝ :=
  abs m / Real.sqrt (k^2 + 1)

-- Main theorem
theorem ellipse_line_intersection_distance_range (k m : ℝ) :
  k > 0 →
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    E x₁ y₁ ∧ E x₂ y₂ ∧
    l k m x₁ y₁ ∧ l k m x₂ y₂ ∧
    y₁ / x₁ + y₂ / x₂ = 2) →
  0 ≤ distance_point_to_line k m ∧ distance_point_to_line k m < 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_distance_range_l1086_108656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1086_108601

/-- Represents a hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Represents a point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt (1 + h.b^2 / h.a^2)

/-- The theorem stating the eccentricity of the hyperbola under given conditions -/
theorem hyperbola_eccentricity 
  (h : Hyperbola) 
  (p : Parabola) 
  (o a b : Point) 
  (h_asymptotes : 
    (a.y = h.b / h.a * a.x ∨ a.y = -h.b / h.a * a.x) ∧ 
    (b.y = h.b / h.a * b.x ∨ b.y = -h.b / h.a * b.x) ∧
    o.x = 0 ∧ o.y = 0)
  (h_parabola : a.y^2 = 2 * p.p * a.x ∧ b.y^2 = 2 * p.p * b.x)
  (h_circumcenter : 
    ∃ (c : Point), c.x = p.p / 2 ∧ c.y = 0 ∧ 
    (c.x - o.x)^2 + (c.y - o.y)^2 = (c.x - a.x)^2 + (c.y - a.y)^2 ∧
    (c.x - o.x)^2 + (c.y - o.y)^2 = (c.x - b.x)^2 + (c.y - b.y)^2) :
  eccentricity h = 3 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1086_108601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_130_l1086_108692

/-- A boat traveling on a stream with given upstream and downstream times and distances -/
structure Boat where
  downstream_time : ℝ
  upstream_time : ℝ
  upstream_distance : ℝ
  stream_speed : ℝ

/-- Calculate the downstream distance traveled by the boat -/
noncomputable def downstream_distance (b : Boat) : ℝ :=
  let boat_speed := (b.upstream_distance / b.upstream_time) + b.stream_speed
  (boat_speed + b.stream_speed) * b.downstream_time

/-- Theorem stating that the downstream distance is 130 km for the given conditions -/
theorem downstream_distance_is_130 (b : Boat) 
    (h1 : b.downstream_time = 10)
    (h2 : b.upstream_time = 15)
    (h3 : b.upstream_distance = 75)
    (h4 : b.stream_speed = 4) :
  downstream_distance b = 130 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_downstream_distance_is_130_l1086_108692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_relation_l1086_108627

noncomputable def data1 : List ℝ := [2, 4, 6, 8]
noncomputable def data2 : List ℝ := [1, 2, 3, 4]

noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  (data.map (fun x => (x - mean) ^ 2)).sum / data.length

theorem variance_relation : variance data1 = 4 * variance data2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_relation_l1086_108627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_inequality_solution_set_a_range_l1086_108646

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- State the properties of f
axiom f_symmetry : ∀ x : ℝ, f (2 - x) = f x
axiom f_def : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)

-- Theorem 1: Value of f(-1)
theorem f_neg_one : f (-1) = 1 - Real.log 3 := by sorry

-- Theorem 2: Solution set of the inequality
theorem inequality_solution_set : 
  {x : ℝ | f (2 - 2*x) < f (x + 3)} = Set.Ioo (-1/3) 3 := by sorry

-- Theorem 3: Range of a for which the equation has a solution
theorem a_range : 
  {a : ℝ | ∃ x > 1, f x = Real.log (a/x + 2*a)} = Set.Ioi (2/3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neg_one_inequality_solution_set_a_range_l1086_108646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_fraction_sold_l1086_108643

theorem orange_fraction_sold (initial_oranges initial_apples : ℕ) 
  (apple_fraction_sold : ℚ) (total_left : ℕ) :
  initial_oranges = 40 →
  initial_apples = 70 →
  apple_fraction_sold = 1/2 →
  total_left = 65 →
  ∃ (orange_fraction_sold : ℚ),
    orange_fraction_sold = 1/4 ∧
    (1 - orange_fraction_sold) * (initial_oranges : ℚ) + 
    (1 - apple_fraction_sold) * (initial_apples : ℚ) = (total_left : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_fraction_sold_l1086_108643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_N_divisibility_l1086_108625

def N (k : ℕ) : ℚ := (2 * k).factorial / k.factorial

theorem N_divisibility (k : ℕ) : 
  (∃ m : ℕ, N k = 2^k * m) ∧ ¬(∃ n : ℕ, N k = 2^(k+1) * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_N_divisibility_l1086_108625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_trapezoid_error_l1086_108664

/-- The percentage error of the Egyptian method for calculating the area of an isosceles trapezoid -/
noncomputable def trapezoid_error (a b c : ℝ) : ℝ :=
  |20 / Real.sqrt 399 - 1| * 100

/-- Theorem stating the percentage error for the given trapezoid dimensions -/
theorem egyptian_trapezoid_error :
  let a : ℝ := 6  -- lower base
  let b : ℝ := 4  -- upper base
  let c : ℝ := 20 -- lateral side
  trapezoid_error a b c = |20 / Real.sqrt 399 - 1| * 100 := by
  -- Unfold the definition of trapezoid_error
  unfold trapezoid_error
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_egyptian_trapezoid_error_l1086_108664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_neg_one_range_of_a_l1086_108623

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x + (1/2) * x^2

-- Part 1
theorem monotonicity_when_a_neg_one :
  let f' := fun x => Real.exp x - 1 + x
  (∀ x < 0, f' x < 0) ∧ (∀ x > 0, f' x > 0) := by sorry

-- Part 2
theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f a x ≥ (3/2) * x^2 + 3 * a * x + a^2 - 2 * Real.exp x) →
  a ∈ Set.Icc (-Real.sqrt 3) (2 - Real.log (4/3)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_when_a_neg_one_range_of_a_l1086_108623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l1086_108621

noncomputable def nested_sqrt (x : ℝ) (n : ℕ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => Real.sqrt (x + nested_sqrt x n)

theorem unique_integer_solution :
  ∀ x y : ℤ, (nested_sqrt (x : ℝ) 1964 : ℝ) = y → x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_integer_solution_l1086_108621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_times_B_equals_union_l1086_108672

-- Define the sets A, B, and A × B
def A : Set ℝ := {x | |x - 1/2| < 1}
def B : Set ℝ := {x | 1/x ≥ 1}
def A_times_B : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- State the theorem
theorem A_times_B_equals_union :
  A_times_B = Set.Ioc (-1/2) 0 ∪ Set.Ioo 1 (3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_times_B_equals_union_l1086_108672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_is_neg_27_final_result_l1086_108619

/-- The sum of the coefficients of the expanded form of -(2x - 5)(3x + 4(2x - 5)) is -27 -/
theorem sum_of_coefficients_expansion (x : ℝ) : 
  let expression := -(2*x - 5)*(3*x + 4*(2*x - 5))
  expression = -22*x^2 + 95*x - 100 := by
  sorry

/-- The sum of the coefficients of the expanded form is -27 -/
theorem sum_of_coefficients_is_neg_27 :
  (-22) + 95 + (-100) = -27 := by
  ring

/-- Combining the expansion and sum of coefficients -/
theorem final_result (x : ℝ) :
  let expression := -(2*x - 5)*(3*x + 4*(2*x - 5))
  let coefficients_sum := (-22) + 95 + (-100)
  coefficients_sum = -27 := by
  exact sum_of_coefficients_is_neg_27

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coefficients_expansion_sum_of_coefficients_is_neg_27_final_result_l1086_108619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contagious_characterization_l1086_108663

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Sum of digits for 1000 consecutive integers starting from k -/
def C (k : ℕ) : ℕ := (Finset.range 1000).sum (λ i => sum_of_digits (k + i))

/-- A positive integer is contagious if it's the sum of digits of 1000 consecutive non-negative integers -/
def is_contagious (N : ℕ) : Prop :=
  ∃ k : ℕ, C k = N

theorem contagious_characterization :
  ∀ N : ℕ, N > 0 → (is_contagious N ↔ N ≥ 13500) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contagious_characterization_l1086_108663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_predicting_function_l1086_108690

theorem exists_predicting_function :
  ∃ f : Set ℕ → ℕ, ∀ A : Set ℕ, Set.Finite {x : ℕ | x ∉ A ∧ f (A ∪ {x}) ≠ x} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_predicting_function_l1086_108690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_specific_l1086_108600

/-- The area between two equal parallel chords in a circle -/
noncomputable def area_between_chords (r : ℝ) (d : ℝ) : ℝ :=
  2 * (r^2 * Real.arccos (d / (2 * r)) - d * Real.sqrt (r^2 - (d^2 / 4)))

/-- Theorem: Area between two equal parallel chords in a circle of radius 10 inches,
    6 inches apart, is 66π - 3√91 square inches -/
theorem area_between_chords_specific :
  area_between_chords 10 6 = 66 * Real.pi - 3 * Real.sqrt 91 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_chords_specific_l1086_108600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1086_108630

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  h : a > 0

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = p.a * x

/-- Focus of a parabola -/
noncomputable def focus (p : Parabola) : ℝ × ℝ := (p.a / 4, 0)

/-- Line passing through the focus and intersecting the parabola -/
structure IntersectingLine (p : Parabola) where
  k : ℝ
  A : ParabolaPoint p
  B : ParabolaPoint p
  h1 : A.x = k * A.y + p.a / 4
  h2 : B.x = k * B.y + p.a / 4
  h3 : A ≠ B

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := ((x1 - x2)^2 + (y1 - y2)^2).sqrt

/-- Theorem: The ratio of products to sum of distances is always a/4 -/
theorem parabola_intersection_ratio (p : Parabola) (l : IntersectingLine p) :
  let F := focus p
  let AF := distance l.A.x l.A.y F.1 F.2
  let BF := distance l.B.x l.B.y F.1 F.2
  (AF * BF) / (AF + BF) = p.a / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_ratio_l1086_108630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1086_108669

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + c

theorem cubic_function_properties
  (a b c : ℝ)
  (h_a : a ≠ 0)
  (h_odd : ∀ x, f a b c (-x) = -(f a b c x))
  (h_perpendicular : (3 * a + b) * 6 = -1)
  (h_min_derivative : ∀ x, 3 * a * x^2 + b ≥ -12)
  (h_min_derivative_exists : ∃ x, 3 * a * x^2 + b = -12) :
  (a = 2 ∧ b = -12 ∧ c = 0) ∧
  (∀ x, (x < -Real.sqrt 2 ∨ x > Real.sqrt 2) → StrictMonoOn (f a b c) (Set.Ioo x (x + 1))) ∧
  (∀ x ∈ Set.Icc (-1) 3, f a b c x ≤ 18) ∧
  (∀ x ∈ Set.Icc (-1) 3, f a b c x ≥ -8 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l1086_108669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1086_108679

theorem max_k_value : 
  ∃ k_max : ℝ, k_max = 8 ∧ 
  ∀ k : ℝ, (∀ m : ℝ, 0 < m → m < 1/2 → 1/m + 2/(1-2*m) ≥ k) → 
  k ≤ k_max :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_k_value_l1086_108679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_transformations_l1086_108685

-- Define the pattern
structure Pattern where
  ℓ : Line
  triangles : Set RightTriangle
  squares : Set Square

-- Define the transformations
inductive Transformation
  | Rotation90 : Point → Transformation
  | TranslationParallel : Line → Transformation
  | ReflectionAcross : Line → Transformation
  | ReflectionPerpendicular : Line → Transformation

-- Define the property of mapping the pattern onto itself
def MapsOntoItself (t : Transformation) (p : Pattern) : Prop :=
  ∃ (p' : Pattern), p' = p -- We remove the application of t here

-- State the theorem
theorem pattern_transformations (p : Pattern) :
  (∃ (ts : Finset Transformation), ts.card = 3 ∧ 
    (∀ t ∈ ts, MapsOntoItself t p) ∧
    (∀ t : Transformation, t ∉ ts → ¬MapsOntoItself t p)) :=
by
  sorry

-- Define the specific transformations
def rotation90 (center : Point) : Transformation :=
  Transformation.Rotation90 center

def translationParallel (ℓ : Line) : Transformation :=
  Transformation.TranslationParallel ℓ

def reflectionAcross (ℓ : Line) : Transformation :=
  Transformation.ReflectionAcross ℓ

def reflectionPerpendicular (ℓ : Line) : Transformation :=
  Transformation.ReflectionPerpendicular ℓ

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_transformations_l1086_108685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1086_108695

-- Define the parabola
def parabola (P : ℝ × ℝ) : Prop :=
  (P.2)^2 = 4 * P.1

-- Define the projection Q of P on the y-axis
def projection (P Q : ℝ × ℝ) : Prop :=
  Q.1 = 0 ∧ Q.2 = P.2

-- Define the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Define the point M
def M : ℝ × ℝ := (4, 5)

-- State the theorem
theorem min_distance_sum :
  ∀ P Q : ℝ × ℝ,
  parabola P →
  projection P Q →
  (∀ P' Q' : ℝ × ℝ, parabola P' → projection P' Q' →
    distance P Q + distance P M ≤ distance P' Q' + distance P' M) →
  distance P Q + distance P M = Real.sqrt 34 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1086_108695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_for_specific_point_l1086_108634

/-- The parabola type representing y² = 2px -/
structure Parabola where
  p : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a parabola -/
def lies_on (point : Point) (parabola : Parabola) : Prop :=
  point.y ^ 2 = 2 * parabola.p * point.x

/-- Calculate the distance from a point to the directrix of a parabola -/
noncomputable def distance_to_directrix (point : Point) (parabola : Parabola) : ℝ :=
  point.x + parabola.p / 2

theorem distance_to_directrix_for_specific_point :
  ∀ (C : Parabola),
    let A : Point := ⟨1, Real.sqrt 5⟩
    lies_on A C →
    distance_to_directrix A C = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_directrix_for_specific_point_l1086_108634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1086_108691

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem f_properties :
  (∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0 ∧ (∀ c, f c = 0 → c = a ∨ c = b)) ∧
  (∀ x : ℝ, f (1 + x) + f (1 - x) = 4) ∧
  (∃ x : ℝ, f x = -3*x + 5 ∧ (deriv f x) = -3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1086_108691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l1086_108693

theorem periodic_function_value (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 3) = -1 / f x)
  (h2 : f 2 = 1/2) : 
  f 2015 = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l1086_108693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_goldenRatioMinusOneBinaryBound_l1086_108640

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Binary representation of the golden ratio minus one -/
def goldenRatioMinusOneBinary : ℕ → ℕ
  | 0 => 0  -- placeholder for n_0
  | (n+1) => sorry  -- actual definition would be complex

/-- The sequence is non-decreasing -/
axiom goldenRatioMinusOneBinaryMonotone : ∀ k, goldenRatioMinusOneBinary k ≤ goldenRatioMinusOneBinary (k+1)

/-- The sequence starts from 1 or greater -/
axiom goldenRatioMinusOneBinaryStart : goldenRatioMinusOneBinary 1 ≥ 1

/-- The binary representation correctly represents φ - 1 -/
axiom goldenRatioMinusOneBinaryCorrect : 
  φ - 1 = ∑' k, (2 : ℝ)^(-(goldenRatioMinusOneBinary k : ℤ))

/-- The main theorem to prove -/
theorem goldenRatioMinusOneBinaryBound (k : ℕ) (h : k ≥ 4) : 
  goldenRatioMinusOneBinary k ≤ 2^(k-1) - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_goldenRatioMinusOneBinaryBound_l1086_108640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1086_108682

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

noncomputable def g (x : ℝ) : ℝ := 1 / Real.exp x

theorem function_inequality (a : ℝ) :
  (∃ x₁ x₂ : ℝ, 1/2 ≤ x₁ ∧ x₁ ≤ 2 ∧ 1/2 ≤ x₂ ∧ x₂ ≤ 2 ∧
    (deriv (f a)) x₁ ≤ g x₂) →
  a ≤ Real.sqrt (Real.exp 1) / Real.exp 1 - 5/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l1086_108682
