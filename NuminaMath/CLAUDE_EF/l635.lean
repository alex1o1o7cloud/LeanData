import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_quadrilateral_existence_l635_63586

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  radius_pos : radius > 0

/-- A point on a circle --/
def PointOnCircle (c : Circle) : Type := { p : ℝ × ℝ // (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2 }

/-- A tangential quadrilateral is a quadrilateral inscribed in a circle where the sum of one pair of opposite sides equals the sum of the other pair --/
def IsTangentialQuadrilateral (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  ∃ (c : Circle), (∃ (p1' : PointOnCircle c), p1 = p1'.val) ∧
                  (∃ (p2' : PointOnCircle c), p2 = p2'.val) ∧
                  (∃ (p3' : PointOnCircle c), p3 = p3'.val) ∧
                  (∃ (p4' : PointOnCircle c), p4 = p4'.val) ∧
  (‖p1 - p2‖ + ‖p3 - p4‖ = ‖p2 - p3‖ + ‖p4 - p1‖)

/-- Given three distinct points on a circle, there exists a fourth point on the circle such that the four points form a tangential quadrilateral --/
theorem tangential_quadrilateral_existence (c : Circle) (p1 p2 p3 : PointOnCircle c)
  (h_distinct : p1 ≠ p2 ∧ p2 ≠ p3 ∧ p3 ≠ p1) :
  ∃ (p4 : PointOnCircle c), IsTangentialQuadrilateral p1.val p2.val p3.val p4.val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangential_quadrilateral_existence_l635_63586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_managers_is_47_l635_63510

/-- Represents the number of managers in a department -/
def managers : ℕ := 9

/-- Represents the ratio of managers to non-managers -/
def ratio : ℚ := 7 / 37

/-- Calculates the maximum number of non-managers given the number of managers and the ratio constraint -/
def max_non_managers (m : ℕ) (r : ℚ) : ℕ :=
  Int.toNat (Int.floor ((m : ℚ) / r))

/-- Theorem stating that the maximum number of non-managers is 47 -/
theorem max_non_managers_is_47 :
  max_non_managers managers ratio = 47 := by
  sorry

#eval max_non_managers managers ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_non_managers_is_47_l635_63510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_of_g_l635_63541

/-- Given a polynomial g(x) that satisfies g(x+1) - g(x) = 6x^3 + 4 for all real x,
    its leading coefficient is 1.5 -/
theorem leading_coefficient_of_g (g : ℝ → ℝ) :
  (∀ x : ℝ, g (x + 1) - g x = 6 * x^3 + 4) →
  ∃ (h : Polynomial ℝ),
    (∀ x : ℝ, g x = (h.eval x : ℝ)) ∧
    h.leadingCoeff = 1.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_leading_coefficient_of_g_l635_63541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_intersection_range_l635_63557

/-- Given that f(x) = x^2 + e^x - k*e^(-x) is an even function, 
    prove that y = f(x) has common points with g(x) = x^2 + a 
    if and only if a ∈ [2, +∞) -/
theorem even_function_intersection_range : 
  (∀ x : ℝ, (fun x : ℝ => x^2 + Real.exp x - Real.exp (-x)) = 
            (fun x : ℝ => ((-x)^2 + Real.exp (-x) - Real.exp x))) → 
  (∀ a : ℝ, (∃ x : ℝ, x^2 + Real.exp x - Real.exp (-x) = x^2 + a) ↔ 
            a ∈ Set.Ici (2 : ℝ)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_intersection_range_l635_63557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_a_2016_l635_63509

/-- A geometric sequence {b_n} -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- Sequence a_n defined recursively -/
def a (b : ℕ → ℝ) : ℕ → ℝ
  | 0 => 1
  | n + 1 => a b n * b (n + 1)

theorem ln_a_2016 (b : ℕ → ℝ) 
    (h_geometric : geometric_sequence b)
    (h_b_1008 : b 1008 = Real.exp 1) :
  Real.log (a b 2016) = 2015 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_a_2016_l635_63509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inverse_square_sum_constant_l635_63556

/-- An ellipse in the xy-plane -/
structure Ellipse where
  /-- Parametric equation for x-coordinate -/
  x : ℝ → ℝ
  /-- Parametric equation for y-coordinate -/
  y : ℝ → ℝ

/-- The specific ellipse from the problem -/
noncomputable def problemEllipse : Ellipse where
  x := λ θ ↦ 2 * Real.cos θ
  y := λ θ ↦ Real.sin θ

/-- Convert from parametric form to polar form -/
noncomputable def polarRadius (e : Ellipse) (θ : ℝ) : ℝ :=
  Real.sqrt ((e.x θ)^2 + (e.y θ)^2)

/-- The theorem to be proved -/
theorem ellipse_inverse_square_sum_constant (α : ℝ) : 
  let e := problemEllipse
  let rA := polarRadius e α
  let rB := polarRadius e (α + π/2)
  1 / rA^2 + 1 / rB^2 = 5/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_inverse_square_sum_constant_l635_63556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_M_l635_63580

/-- The set M of x values between 1 and 4 inclusive -/
def M : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 4}

/-- The function f(x) -/
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + b/x + c

/-- The function g(x) -/
noncomputable def g (x : ℝ) : ℝ := (1/4) * x + 1/x

/-- The theorem stating the maximum value of f(x) on M -/
theorem max_value_f_on_M (b c : ℝ) : 
  (∃ x₀ ∈ M, ∀ x ∈ M, f b c x ≥ f b c x₀ ∧ g x ≥ g x₀ ∧ f b c x₀ = g x₀) → 
  (∃ x_max ∈ M, ∀ x ∈ M, f b c x ≤ f b c x_max) ∧ 
  (∃ x_max ∈ M, f b c x_max = 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_M_l635_63580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separated_circles_do_not_intersect_l635_63517

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O2 (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 9

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (0, 0)
def center_O2 : ℝ × ℝ := (3, -4)
def radius_O1 : ℝ := 1
def radius_O2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((3 - 0)^2 + (-4 - 0)^2)

-- Theorem: The circles are separated
theorem circles_are_separated :
  distance_between_centers > radius_O1 + radius_O2 := by
  sorry

-- Theorem: The circles do not intersect
theorem circles_do_not_intersect :
  ∀ x y : ℝ, ¬(circle_O1 x y ∧ circle_O2 x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separated_circles_do_not_intersect_l635_63517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_tangent_directrix_l635_63574

/-- The circle equation -/
def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 + m*x - 1/4 = 0

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := y = 1/4 * x^2

/-- The directrix of the parabola -/
def directrix_eq (y : ℝ) : Prop := y = -1

/-- Theorem: If a circle and parabola have a tangent directrix, then m = ± √3 -/
theorem circle_parabola_tangent_directrix (m : ℝ) :
  (∃ x y : ℝ, circle_eq x y m ∧ parabola_eq x y) →
  (∃ x y : ℝ, circle_eq x y m ∧ directrix_eq y) →
  m = Real.sqrt 3 ∨ m = -Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_parabola_tangent_directrix_l635_63574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l635_63573

/-- An inverse proportion function passing through (2, 3) -/
noncomputable def inverse_proportion (k : ℝ) (x : ℝ) : ℝ := k / x

theorem inverse_proportion_k_value :
  ∃ (k : ℝ), k ≠ 0 ∧ inverse_proportion k 2 = 3 ∧ k = 6 := by
  use 6
  constructor
  · exact ne_of_gt (by norm_num)
  constructor
  · simp [inverse_proportion]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_k_value_l635_63573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l635_63575

-- Define the triangle
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the vectors
noncomputable def p (t : Triangle) : Fin 2 → Real
  | 0 => Real.sqrt 3 * Real.sin t.C
  | 1 => -1/2

noncomputable def q (t : Triangle) : Fin 2 → Real
  | 0 => Real.cos t.C
  | 1 => 1 + Real.cos (2 * t.C)

noncomputable def m (t : Triangle) : Fin 2 → Real
  | 0 => 1
  | 1 => Real.sin t.A

noncomputable def n (t : Triangle) : Fin 2 → Real
  | 0 => 2
  | 1 => Real.sin t.B

-- Define dot product
def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define colinearity
def colinear (v w : Fin 2 → Real) : Prop :=
  ∃ k : Real, v 0 = k * w 0 ∧ v 1 = k * w 1

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 3)
  (h2 : dot_product (p t) (q t) = 1/2)
  (h3 : colinear (m t) (n t)) :
  t.C = Real.pi/3 ∧ t.a = Real.sqrt 3 ∧ t.b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l635_63575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cardinality_of_set_l635_63508

theorem smallest_cardinality_of_set (n k : ℕ) : 
  ∃ (A : Set (Fin (n.choose (n - k + 1)))), 
    ∃ (subsets : Fin n → Set (Fin (n.choose (n - k + 1)))),
      (∀ (indices : Finset (Fin n)), indices.card = k → 
        (⋃ (i ∈ indices), subsets i) = A) ∧
      (∀ (indices : Finset (Fin n)), indices.card = k - 1 → 
        (⋃ (i ∈ indices), subsets i) ≠ A) ∧
      (∀ (B : Type) [Fintype B], ∀ (subsets' : Fin n → Set B),
        (∀ (indices : Finset (Fin n)), indices.card = k → 
          (⋃ (i ∈ indices), subsets' i) = Set.univ) ∧
        (∀ (indices : Finset (Fin n)), indices.card = k - 1 → 
          (⋃ (i ∈ indices), subsets' i) ≠ Set.univ) →
        Fintype.card B ≥ (n.choose (n - k + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cardinality_of_set_l635_63508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_and_phi_l635_63532

noncomputable def f (x φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem min_period_and_phi (φ : ℝ) (h : |φ| < π/2) :
  (∃ T > 0, ∀ x, f x φ = f (x + T) φ ∧ 
    ∀ S, (0 < S ∧ S < T) → ∃ y, f y φ ≠ f (y + S) φ) ∧
  (∀ x ∈ Set.Icc (π/2) (5*π/6), 
    (∀ y ∈ Set.Icc (π/2) (5*π/6), x ≤ y → f x φ ≤ f y φ) ∨
    (∀ y ∈ Set.Icc (π/2) (5*π/6), x ≤ y → f y φ ≤ f x φ)) →
  f (π/2) φ = -f (5*π/6) φ →
  φ = -π/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_period_and_phi_l635_63532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_area_l635_63537

/-- The area of the region covered by the union of four equilateral triangles -/
theorem equilateral_triangles_area (side_length : ℝ) (num_triangles : ℕ) 
  (overlap_fraction : ℝ) : 
  side_length = 3 → 
  num_triangles = 4 → 
  overlap_fraction = 1/3 → 
  (num_triangles * (Real.sqrt 3 / 4 * side_length^2) - 
   (num_triangles - 1) * (Real.sqrt 3 / 4 * (overlap_fraction * side_length)^2)) = 33 * Real.sqrt 3 / 4 := by
  sorry

#check equilateral_triangles_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangles_area_l635_63537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_game_properties_l635_63503

/-- Represents a player in the basketball shooting game -/
inductive Player : Type
| A
| B

/-- The shooting game with two players A and B -/
structure ShootingGame where
  playerA_percentage : ℝ
  playerB_percentage : ℝ
  first_shot_prob : ℝ

/-- The probability that player B takes the second shot -/
noncomputable def prob_B_second_shot (game : ShootingGame) : ℝ :=
  game.first_shot_prob * (1 - game.playerA_percentage) + game.first_shot_prob * game.playerB_percentage

/-- The probability that player A takes the i-th shot -/
noncomputable def prob_A_ith_shot (game : ShootingGame) (i : ℕ) : ℝ :=
  1/3 + (1/6) * (2/5)^(i-1)

/-- The expected number of shots player A takes -/
noncomputable def expected_shots_A (game : ShootingGame) (n : ℕ) : ℝ :=
  (5/18) * (1 - (2/5)^n) + n/3

/-- Main theorem stating the properties of the shooting game -/
theorem shooting_game_properties (game : ShootingGame) 
    (h1 : game.playerA_percentage = 0.6)
    (h2 : game.playerB_percentage = 0.8)
    (h3 : game.first_shot_prob = 0.5) :
  (prob_B_second_shot game = 0.6) ∧
  (∀ i : ℕ, i > 0 → prob_A_ith_shot game i = 1/3 + (1/6) * (2/5)^(i-1)) ∧
  (∀ n : ℕ, n > 0 → expected_shots_A game n = (5/18) * (1 - (2/5)^n) + n/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_game_properties_l635_63503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_average_speed_l635_63551

noncomputable def segment1_speed : ℝ := 15
noncomputable def segment1_distance : ℝ := 10
noncomputable def segment2_speed : ℝ := 20
noncomputable def segment2_distance : ℝ := 20
noncomputable def segment3_speed : ℝ := 25
noncomputable def segment3_time : ℝ := 0.5
noncomputable def segment4_speed : ℝ := 22
noncomputable def segment4_time : ℝ := 0.75

noncomputable def total_distance : ℝ := segment1_distance + segment2_distance + segment3_speed * segment3_time + segment4_speed * segment4_time

noncomputable def total_time : ℝ := segment1_distance / segment1_speed + segment2_distance / segment2_speed + segment3_time + segment4_time

noncomputable def average_speed : ℝ := total_distance / total_time

theorem bicycle_average_speed : 
  |average_speed - 20.21| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_average_speed_l635_63551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonals_l635_63549

/-- A rectangular prism with 12 vertices and 18 edges -/
structure RectangularPrism where
  vertices : Nat
  edges : Nat
  h_vertices : vertices = 12
  h_edges : edges = 18

/-- A diagonal in a rectangular prism -/
def Diagonal (rp : RectangularPrism) : Type :=
  { pair : Fin rp.vertices × Fin rp.vertices // pair.1 ≠ pair.2 }

/-- The number of diagonals in a rectangular prism -/
def numDiagonals (rp : RectangularPrism) : Nat :=
  (rp.vertices.choose 2) - rp.edges

/-- Theorem: A rectangular prism with 12 vertices and 18 edges has 24 diagonals -/
theorem rectangular_prism_diagonals (rp : RectangularPrism) :
  numDiagonals rp = 24 := by
  sorry

#eval numDiagonals { vertices := 12, edges := 18, h_vertices := rfl, h_edges := rfl }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_diagonals_l635_63549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l635_63519

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 6)

theorem function_properties :
  (∀ x, f x ≤ 2) ∧
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (f 0 = 1) →
  ∀ x, f x = 2 * Real.sin (x - Real.pi / 6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l635_63519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_120_eq_245_l635_63568

/-- Sequence a_n where a_n = n for positive integers -/
def a (n : ℕ+) : ℕ := n

/-- Number of twos inserted between a_k and a_{k+1} -/
def twos_inserted (k : ℕ+) : ℕ := 3^(k.val - 1)

/-- Sequence d_n formed by inserting twos between elements of a_n -/
def d : ℕ+ → ℕ := sorry

/-- Sum of the first n terms of the sequence d_n -/
def S (n : ℕ+) : ℕ := sorry

/-- The sum of the first 120 terms of the sequence d_n is 245 -/
theorem S_120_eq_245 : S 120 = 245 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_120_eq_245_l635_63568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_result_l635_63589

/-- Given two functions f and g, where f(x) = 5x + c and g(x) = cx + 3,
    if f(g(x)) = 15x + d, then d = 18 -/
theorem function_composition_result :
  ∃ c d : ℝ, 
  let f : ℝ → ℝ := λ x ↦ 5 * x + c
  let g : ℝ → ℝ := λ x ↦ c * x + 3
  (∀ x, f (g x) = 15 * x + d) → d = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_result_l635_63589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_ratio_l635_63516

/-- Given a triangle XYZ, with P on XY and Q on YZ, prove that PQ/QR = 1/3 under specific conditions -/
theorem triangle_intersection_ratio (X Y Z P Q R : EuclideanSpace ℝ (Fin 2)) : 
  -- Triangle XYZ exists
  (X ≠ Y ∧ Y ≠ Z ∧ Z ≠ X) →
  -- P is on XY with XP:PY = 4:1
  (∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ P = (1 - t) • X + t • Y ∧ t = 1/5) →
  -- Q is on YZ with YQ:QZ = 4:1
  (∃ s : ℝ, s ∈ Set.Ioo 0 1 ∧ Q = (1 - s) • Y + s • Z ∧ s = 4/5) →
  -- PQ and XZ intersect at R
  (∃ u v : ℝ, u ∈ Set.Ioo 0 1 ∧ v ∈ Set.Ioo 0 1 ∧ 
    R = (1 - u) • P + u • Q ∧ 
    R = (1 - v) • X + v • Z) →
  -- Then PQ/QR = 1/3
  ‖P - Q‖ / ‖Q - R‖ = 1/3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_ratio_l635_63516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l635_63593

theorem relationship_abc : 
  let a : ℝ := Real.log 2
  let b : ℝ := (5 : ℝ)^(-(1/2 : ℝ))
  let c : ℝ := Real.sin (30 * π / 180)
  b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l635_63593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_problem_l635_63582

theorem group_size_problem (n : ℕ) 
  (member_contribution : ℕ → ℕ)
  (total_collection : ℕ) : 
  (∀ m, member_contribution m = n) →
  total_collection = 1936 →
  total_collection = n * n →
  n = 44 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_group_size_problem_l635_63582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_specific_factors_l635_63566

theorem smallest_number_with_specific_factors : 
  ∃ (n : ℕ), 
    (∀ (m : ℕ), m ∣ n ↔ m ∈ Finset.range (n + 1) ∧ n % m = 0) ∧ 
    (Finset.filter (λ m ↦ m ∣ n) (Finset.range (n + 1))).card = 17 ∧
    (Finset.filter (λ m ↦ m ∣ n ∧ m ≠ 1 ∧ m ≠ n) (Finset.range (n + 1))).card = 15 ∧
    (∀ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n → m > 1 ∧ m < n) ∧
    (∀ k : ℕ, k < n → 
      (Finset.filter (λ m ↦ m ∣ k) (Finset.range (k + 1))).card < 17 ∨
      (Finset.filter (λ m ↦ m ∣ k ∧ m ≠ 1 ∧ m ≠ k) (Finset.range (k + 1))).card < 15 ∨
      ∃ m, m ∣ k ∧ m ≠ 1 ∧ m ≠ k ∧ (m ≤ 1 ∨ m ≥ k)) ∧
    n = 65536 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_specific_factors_l635_63566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l635_63567

/-- Given a quadratic function f(x) = px^2 + qx + r passing through (0, 9) and (1, 6), 
    prove that p + q + 2r = 15 -/
theorem quadratic_sum (p q r : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = p * x^2 + q * x + r) → 
  f 0 = 9 → 
  f 1 = 6 → 
  p + q + 2 * r = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_sum_l635_63567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparisons_l635_63504

-- Constants and variables
variable (p₀ : ℝ) -- Threshold of auditory perception
variable (p₁ p₂ p₃ : ℝ) -- Actual sound pressures for gasoline, hybrid, and electric cars

-- Definitions based on conditions
noncomputable def sound_pressure_level (p : ℝ) : ℝ := 20 * Real.log (p / p₀) / Real.log 10

-- Theorem statement
theorem sound_pressure_comparisons
  (h_p₀_pos : p₀ > 0)
  (h_gasoline : 60 ≤ sound_pressure_level p₀ p₁ ∧ sound_pressure_level p₀ p₁ ≤ 90)
  (h_hybrid : 50 ≤ sound_pressure_level p₀ p₂ ∧ sound_pressure_level p₀ p₂ ≤ 60)
  (h_electric : sound_pressure_level p₀ p₃ = 40) :
  p₁ ≥ p₂ ∧ p₃ = 100 * p₀ ∧ p₁ ≤ 100 * p₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sound_pressure_comparisons_l635_63504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l635_63540

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the distance between two parallel planes -/
noncomputable def distance_between_planes (p1 p2 : Plane) : ℝ :=
  let n := Real.sqrt (p1.a^2 + p1.b^2 + p1.c^2)
  abs (p1.d - p2.d) / n

/-- The distance between the planes 3x + 4y - z = 12 and 6x + 8y - 2z = 18 is 3√26/26 -/
theorem distance_between_specific_planes : 
  let p1 : Plane := {a := 3, b := 4, c := -1, d := 12}
  let p2 : Plane := {a := 3, b := 4, c := -1, d := 9}
  distance_between_planes p1 p2 = 3 * Real.sqrt 26 / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l635_63540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_3200_l635_63526

/-- Represents the number of days it takes for a person to complete the work alone -/
structure WorkRate where
  days : ℚ
  days_positive : days > 0

/-- Represents the payment for a person's contribution to the work -/
structure Payment where
  amount : ℚ

/-- Calculates the total payment for the work given the conditions -/
noncomputable def calculate_total_payment (a_rate b_rate : WorkRate) (c_payment : Payment) (total_days : ℚ) : ℚ :=
  let a_daily_rate := 1 / a_rate.days
  let b_daily_rate := 1 / b_rate.days
  let ab_daily_rate := a_daily_rate + b_daily_rate
  let abc_daily_rate := 1 / total_days
  let c_daily_rate := abc_daily_rate - ab_daily_rate
  let c_total_work := c_daily_rate * total_days
  c_payment.amount / c_total_work

theorem total_payment_is_3200 (a_rate b_rate : WorkRate) (c_payment : Payment) (total_days : ℚ) :
  a_rate.days = 6 →
  b_rate.days = 8 →
  total_days = 3 →
  c_payment.amount = 400 →
  calculate_total_payment a_rate b_rate c_payment total_days = 3200 := by
  sorry

#eval (3200 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_payment_is_3200_l635_63526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l635_63579

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 3)

theorem smallest_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi / 2 := by
  sorry

#check smallest_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_of_f_l635_63579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_l635_63501

/-- The line equation -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

/-- The circle equation -/
def circleEq (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + k*x - y - 9 = 0

/-- Two points are symmetric about the y-axis -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ = -x₂ ∧ y₁ = y₂

theorem intersection_symmetry (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    line k x₁ y₁ ∧ circleEq k x₁ y₁ ∧
    line k x₂ y₂ ∧ circleEq k x₂ y₂ ∧
    symmetric_about_y_axis x₁ y₁ x₂ y₂) →
  k = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_l635_63501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_square_eccentricity_l635_63520

/-- An ellipse with foci and minor axis vertices forming a square has eccentricity √2/2 -/
theorem ellipse_square_eccentricity (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ellipse : a^2 = b^2 + c^2) (h_square : b = c) : 
  c / a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_square_eccentricity_l635_63520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gadget_price_calculation_l635_63530

noncomputable def calculate_total_price (original_price : ℝ) (initial_discount_rate : ℝ) 
  (additional_discount_rate : ℝ) (additional_discount_threshold : ℝ) (tax_rate : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount_rate)
  let final_discounted_price := 
    if price_after_initial_discount > additional_discount_threshold
    then price_after_initial_discount * (1 - additional_discount_rate)
    else price_after_initial_discount
  final_discounted_price * (1 + tax_rate)

theorem gadget_price_calculation :
  calculate_total_price 120 0.3 0.1 80 0.12 = 84.672 := by
  unfold calculate_total_price
  -- The proof steps would go here, but for now we'll use sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gadget_price_calculation_l635_63530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_to_halfway_point_l635_63598

/-- The time it takes Danny to reach Steve's house in minutes -/
noncomputable def danny_time : ℝ := 33

/-- The time it takes Emma to reach either Danny's or Steve's house in minutes -/
noncomputable def emma_time : ℝ := 40

/-- The time it takes Steve to reach Danny's house in minutes -/
noncomputable def steve_time : ℝ := 2 * danny_time

/-- The time it takes Danny to reach the halfway point -/
noncomputable def danny_halfway_time : ℝ := danny_time / 2

/-- The time it takes Steve to reach the halfway point -/
noncomputable def steve_halfway_time : ℝ := steve_time / 2

/-- The time it takes Emma to reach the halfway point -/
noncomputable def emma_halfway_time : ℝ := emma_time / 2

/-- The additional time it takes Steve to reach the halfway point compared to Danny -/
noncomputable def steve_additional_time : ℝ := steve_halfway_time - danny_halfway_time

/-- The additional time it takes Emma to reach the halfway point compared to Danny -/
noncomputable def emma_additional_time : ℝ := emma_halfway_time - danny_halfway_time

theorem additional_time_to_halfway_point :
  max steve_additional_time emma_additional_time = 16.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_time_to_halfway_point_l635_63598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_triple_volume_l635_63524

/-- The volume of a sphere with radius r -/
noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

/-- The radius of a sphere with volume v -/
noncomputable def sphere_radius (v : ℝ) : ℝ := ((3 * v) / (4 * Real.pi))^(1/3)

theorem sphere_diameter_triple_volume (r : ℝ) (h : r = 6) :
  2 * sphere_radius (3 * sphere_volume r) = 12 * (3 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_diameter_triple_volume_l635_63524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_ratio_proof_l635_63512

theorem number_ratio_proof (a b : ℕ+) (x y : ℕ+) 
  (h1 : Nat.gcd a.val b.val = 4)
  (h2 : Nat.lcm a.val b.val = 48)
  (h3 : a = 4 * x)
  (h4 : b = 4 * y)
  (h5 : x.val * y.val = 12)
  (h6 : Nat.gcd x.val y.val = 1) :
  (a : ℚ) / b = 3 / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_ratio_proof_l635_63512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l635_63511

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point A
def point_A : ℝ × ℝ := (4, 3)

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop := parabola P.1 P.2

-- Distance between two points
noncomputable def distance (P Q : ℝ × ℝ) : ℝ := 
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Statement to prove
theorem min_distance_sum :
  ∀ P : ℝ × ℝ, point_on_parabola P →
    (∀ Q : ℝ × ℝ, point_on_parabola Q →
      distance P point_A + distance P focus ≤ distance Q point_A + distance Q focus) →
    P = (9/4, 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l635_63511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_satisfies_conditions_l635_63538

/-- The ellipse in the problem -/
def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 3 = 1

/-- The line l -/
noncomputable def line_l (x y : ℝ) : Prop := x + Real.sqrt 2 * y - 2 * Real.sqrt 2 = 0

/-- Points A and B are in the first quadrant -/
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

/-- The distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem stating that the line l satisfies all conditions -/
theorem line_l_satisfies_conditions :
  ∃ (x1 y1 x2 y2 : ℝ),
    first_quadrant x1 y1 ∧
    first_quadrant x2 y2 ∧
    ellipse x1 y1 ∧
    ellipse x2 y2 ∧
    line_l x1 y1 ∧
    line_l x2 y2 ∧
    line_l (-2 * Real.sqrt 2) 0 ∧  -- Point M
    line_l 0 2 ∧  -- Point N
    distance (-2 * Real.sqrt 2) 0 x1 y1 = distance 0 2 x2 y2 ∧
    distance (-2 * Real.sqrt 2) 0 0 2 = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_satisfies_conditions_l635_63538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_reaches_watering_hole_l635_63502

/-- Represents the scenario with two lion cubs and a turtle moving towards a watering hole -/
structure WateringHoleScenario where
  turtle_speed : ℝ
  first_cub_speed : ℝ
  second_cub_speed : ℝ
  turtle_initial_distance : ℝ
  first_cub_initial_distance : ℝ

/-- The conditions of the problem -/
def scenario_conditions (s : WateringHoleScenario) : Prop :=
  s.turtle_initial_distance = 32 * s.turtle_speed ∧
  s.first_cub_initial_distance = 6 * s.first_cub_speed ∧
  s.second_cub_speed = 1.5 * s.first_cub_speed ∧
  s.turtle_speed > 0 ∧ s.first_cub_speed > 0

/-- The time it takes for the first lion cub to catch up with the turtle -/
noncomputable def time_to_first_encounter (s : WateringHoleScenario) : ℝ :=
  (s.first_cub_initial_distance - s.turtle_initial_distance) / (s.first_cub_speed - s.turtle_speed)

/-- The time it takes for the second lion cub to catch up with the turtle -/
noncomputable def time_to_second_encounter (s : WateringHoleScenario) : ℝ :=
  s.turtle_initial_distance / (s.turtle_speed + s.second_cub_speed)

/-- The theorem stating that the turtle reaches the watering hole 28.8 minutes after the second encounter -/
theorem turtle_reaches_watering_hole (s : WateringHoleScenario) 
  (h : scenario_conditions s) 
  (h_encounters : time_to_second_encounter s - time_to_first_encounter s = 2.4) :
  32 - time_to_second_encounter s = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_turtle_reaches_watering_hole_l635_63502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_difference_l635_63594

/-- The direction vector of line l -/
def m : ℝ × ℝ × ℝ := (2, -1, 3)

/-- Point A on line l -/
def A (y : ℝ) : ℝ × ℝ × ℝ := (0, y, 3)

/-- Point B on line l -/
def B (z : ℝ) : ℝ × ℝ × ℝ := (-1, 2, z)

/-- The theorem to be proved -/
theorem line_points_difference : ∀ y z : ℝ, y - z = 0 := by
  intro y z
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_points_difference_l635_63594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_91_l635_63545

theorem square_difference_91 : 
  ∃! n : ℕ, n = (Finset.filter (λ p : ℕ × ℕ => p.1 > 0 ∧ p.2 > 0 ∧ p.1^2 - p.2^2 = 91) (Finset.product (Finset.range 1000) (Finset.range 1000))).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_difference_91_l635_63545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_depth_l635_63570

-- Define the cone parameters
noncomputable def total_height : ℝ := 10000
noncomputable def base_radius : ℝ := 2000
noncomputable def above_water_fraction : ℝ := 2/5

-- Define the theorem
theorem cone_water_depth :
  ∀ (h : ℝ) (r : ℝ) (f : ℝ),
    h = total_height →
    r = base_radius →
    f = above_water_fraction →
    (1 - f) * h = 1566 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_water_depth_l635_63570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_passenger_seat_probability_l635_63525

/-- Represents a bus seating scenario -/
structure BusSeating (n : ℕ) where
  seats : Fin n → Fin n
  scientist_seat : Fin n

/-- The probability that the last passenger sits in their assigned seat -/
noncomputable def last_passenger_correct_seat_prob (n : ℕ) : ℝ :=
  1 / 2

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 -/
theorem last_passenger_seat_probability (n : ℕ) (h : n > 0) :
  last_passenger_correct_seat_prob n = 1 / 2 := by
  sorry

#check last_passenger_seat_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_passenger_seat_probability_l635_63525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_problem_l635_63563

theorem matrix_N_problem (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.vecMul ![3, -2] = ![4, 1])
  (h2 : N.vecMul ![-2, 4] = ![0, -2]) :
  N.vecMul ![7, 0] = ![14, 0] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_problem_l635_63563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operation_result_l635_63542

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x > 0}

-- Define the operation ×
def setOperation (A B : Set ℝ) : Set ℝ :=
  {x : ℝ | x ∈ A ∪ B ∧ x ∉ A ∩ B}

-- State the theorem
theorem set_operation_result :
  setOperation A B = Set.Icc 0 1 ∪ Set.Ioi 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operation_result_l635_63542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_mess_expenditure_l635_63554

/-- Represents the hostel mess expenditure problem --/
theorem hostel_mess_expenditure 
  (initial_students : ℕ) 
  (new_students : ℕ) 
  (expense_increase_rate : ℝ) 
  (avg_expenditure_decrease_rate : ℝ) 
  (h1 : initial_students = 75) 
  (h2 : new_students = 12) 
  (h3 : expense_increase_rate = 0.30) 
  (h4 : avg_expenditure_decrease_rate = 0.015) : 
  ∃ (E : ℝ), (E ≥ 1.1374 ∧ E ≤ 1.1376) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hostel_mess_expenditure_l635_63554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_l635_63595

/-- The number of positive integer divisors of n^2 that are less than n but do not divide n, where n = 2^40 * 5^15 -/
theorem divisor_count (n : ℕ) (h : n = 2^40 * 5^15) : 
  (Finset.filter (λ d ↦ d ∣ n^2 ∧ d < n ∧ ¬(d ∣ n)) (Finset.range (n + 1))).card = 599 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_l635_63595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l635_63597

/-- Represents a hyperbola in the real plane -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The asymptote of a hyperbola -/
def AsymptoteOf (h : Hyperbola) (a : ℝ → ℝ) : Prop := sorry

/-- A point lies on a hyperbola -/
def PointOnHyperbola (h : Hyperbola) (p : ℝ × ℝ) : Prop := sorry

/-- The standard equation of a hyperbola -/
def StandardEquationOf (h : Hyperbola) (eq : ℝ → ℝ → Prop) : Prop := sorry

/-- A hyperbola with an asymptote y = x passing through the point (2, 4) has the standard equation y^2 - x^2 = 12 -/
theorem hyperbola_standard_equation (h : Hyperbola) :
  (∃ (a : ℝ → ℝ), AsymptoteOf h a ∧ ∀ x, a x = x) →
  PointOnHyperbola h (2, 4) →
  StandardEquationOf h (fun x y => y^2 - x^2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_standard_equation_l635_63597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_from_triangle120_l635_63528

/-- An isosceles triangle with a vertex angle of 120° -/
structure Triangle120 where
  -- We only need to define one side length, as it's an isosceles triangle
  side : ℝ
  side_pos : side > 0

/-- A rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ
  width_pos : width > 0
  height_pos : height > 0

/-- Predicate stating that the arrangement of triangles covers the entire rectangle -/
def covers_rectangle (n : ℕ) (arrangement : Fin n → ℝ × ℝ) (t : Triangle120) (r : Rectangle) : Prop :=
  sorry

/-- Predicate stating that the triangles in the arrangement do not overlap -/
def no_overlap (n : ℕ) (arrangement : Fin n → ℝ × ℝ) (t : Triangle120) : Prop :=
  sorry

/-- Predicate stating that a rectangle can be constructed from n copies of a Triangle120 -/
def can_construct_rectangle (n : ℕ) (t : Triangle120) (r : Rectangle) : Prop :=
  ∃ (arrangement : Fin n → ℝ × ℝ),
    covers_rectangle n arrangement t r ∧
    no_overlap n arrangement t

/-- Theorem stating that it's possible to construct a rectangle from Triangle120s -/
theorem rectangle_from_triangle120 :
  ∃ (n : ℕ) (t : Triangle120) (r : Rectangle), n > 0 ∧ can_construct_rectangle n t r :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_from_triangle120_l635_63528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angles_l635_63561

theorem cos_sum_special_angles (α β : ℝ) :
  0 < α ∧ α < π / 2 →
  -π / 2 < β ∧ β < 0 →
  Real.cos (π / 4 + α) = 1 / 3 →
  Real.cos (π / 4 - β / 2) = Real.sqrt 3 / 3 →
  Real.cos (α + β / 2) = 5 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_special_angles_l635_63561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_one_two_three_l635_63588

-- Define the nabla operation
noncomputable def nabla (a b : ℝ) : ℝ := (a + b) / (1 + a * b)

-- State the theorem
theorem nabla_one_two_three :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
  nabla (nabla 1 2) 3 = 1 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nabla_one_two_three_l635_63588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_theorem_l635_63560

noncomputable def tan_ratio (x y : ℝ) : ℝ :=
  (x * Real.sin (Real.pi/5) + y * Real.cos (Real.pi/5)) / 
  (x * Real.cos (Real.pi/5) - y * Real.sin (Real.pi/5))

theorem tan_ratio_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : tan_ratio x y = Real.tan (9*Real.pi/20)) :
  y / x = 1 ∧ 
  ∃ (A B C : ℝ), 
    A + B + C = Real.pi ∧ 
    Real.tan C = y / x ∧
    (∀ A' B' C', A' + B' + C' = Real.pi → Real.tan C' = y / x → 
      Real.sin (2*A') + 2 * Real.cos B' ≤ 3/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_ratio_theorem_l635_63560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l635_63548

noncomputable def f (h : ℝ) (x : ℝ) : ℝ := -Real.log x + x + h

theorem triangle_inequality (h : ℝ) :
  (∀ a b c : ℝ, 1/Real.exp 1 ≤ a ∧ a ≤ Real.exp 1 ∧
                1/Real.exp 1 ≤ b ∧ b ≤ Real.exp 1 ∧
                1/Real.exp 1 ≤ c ∧ c ≤ Real.exp 1 →
    f h a + f h b > f h c ∧
    f h b + f h c > f h a ∧
    f h c + f h a > f h b) →
  h > Real.exp 1 - 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l635_63548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_daves_blue_tshirt_packs_l635_63590

theorem daves_blue_tshirt_packs 
  (white_packs : ℕ)
  (white_per_pack : ℕ)
  (blue_per_pack : ℕ)
  (total_tshirts : ℕ) :
  (white_packs * white_per_pack + blue_per_pack * ((total_tshirts - white_packs * white_per_pack) / blue_per_pack) = total_tshirts) →
  ((total_tshirts - white_packs * white_per_pack) / blue_per_pack = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_daves_blue_tshirt_packs_l635_63590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l635_63553

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → (3 - 2*a)^x < (3 - 2*a)^y

def exclusive_or (p q : Prop) : Prop :=
  (p ∨ q) ∧ ¬(p ∧ q)

theorem range_of_a :
  ∀ a : ℝ, exclusive_or (proposition_p a) (proposition_q a) →
    (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l635_63553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_prism_surface_area_theorem_l635_63535

/-- The total surface area of a regular hexagonal prism circumscribed around a sphere -/
noncomputable def hexagonal_prism_surface_area (R : ℝ) : ℝ := 12 * R^2 * Real.sqrt 3

/-- Theorem: The total surface area of a regular hexagonal prism circumscribed around a sphere
    of radius R is equal to 12R^2√3 -/
theorem hexagonal_prism_surface_area_theorem (R : ℝ) (h : R > 0) :
  hexagonal_prism_surface_area R = 12 * R^2 * Real.sqrt 3 := by
  -- Unfold the definition of hexagonal_prism_surface_area
  unfold hexagonal_prism_surface_area
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_prism_surface_area_theorem_l635_63535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_hexagonal_pyramid_cones_l635_63514

/-- The volume of a cone with radius r and height h -/
noncomputable def cone_volume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The radius of a circle inscribed in a regular hexagon, given the radius of its circumscribed circle -/
noncomputable def inscribed_radius (R : ℝ) : ℝ := (Real.sqrt 3 / 2) * R

theorem volume_difference_hexagonal_pyramid_cones (H R : ℝ) (H_pos : H > 0) (R_pos : R > 0) :
  let V_circum := cone_volume R H
  let V_inscribed := cone_volume (inscribed_radius R) H
  V_circum - V_inscribed = (1/12) * Real.pi * R^2 * H := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_difference_hexagonal_pyramid_cones_l635_63514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l635_63533

noncomputable def f (x : ℝ) : ℝ := 8 / (x - 2) + Real.sqrt (x + 3)

theorem f_properties :
  (∀ x : ℝ, f x = 8 / (x - 2) + Real.sqrt (x + 3)) ∧
  (Set.range f = {y | ∃ x : ℝ, x ∈ Set.Icc (-3) 2 ∪ Set.Ioi 2 ∧ y = f x}) ∧
  (f (-2) = -1) ∧
  (f 6 = 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l635_63533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_y_and_nearest_integer_l635_63583

-- Define the function G
noncomputable def G (a b c d : ℝ) : ℝ := a^b + c * d

-- State the theorem
theorem exists_y_and_nearest_integer :
  ∃ y : ℝ, G 3 y 5 18 = 500 ∧ Int.floor y = 6 := by
  -- The proof is omitted and replaced with 'sorry'
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_y_and_nearest_integer_l635_63583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_consistent_assignment_l635_63576

-- Define the three possible roles
inductive Role
| Honest
| Liar
| Cunning

-- Define the three friends
inductive Friend
| Yaroslav
| Kirill
| Andrey

-- Define a function to represent the assignment of roles to friends
def assignment := Friend → Role

-- Define the statements made by each friend
def yaroslav_statement (a : assignment) : Prop :=
  a Friend.Kirill = Role.Liar

def kirill_statement (a : assignment) : Prop :=
  a Friend.Kirill = Role.Cunning

def andrey_statement (a : assignment) : Prop :=
  a Friend.Kirill = Role.Honest

-- Define a predicate for a valid assignment
def valid_assignment (a : assignment) : Prop :=
  (∃! f, a f = Role.Honest) ∧
  (∃! f, a f = Role.Liar) ∧
  (∃! f, a f = Role.Cunning)

-- Define a predicate for consistent statements given an assignment
def consistent_statements (a : assignment) : Prop :=
  (a Friend.Yaroslav = Role.Honest → yaroslav_statement a) ∧
  (a Friend.Yaroslav = Role.Liar → ¬yaroslav_statement a) ∧
  (a Friend.Kirill = Role.Honest → kirill_statement a) ∧
  (a Friend.Kirill = Role.Liar → ¬kirill_statement a) ∧
  (a Friend.Andrey = Role.Honest → andrey_statement a) ∧
  (a Friend.Andrey = Role.Liar → ¬andrey_statement a)

-- Theorem stating that the only consistent assignment is the one given in the solution
theorem unique_consistent_assignment :
  ∀ a : assignment,
    valid_assignment a ∧ consistent_statements a →
    a Friend.Yaroslav = Role.Honest ∧
    a Friend.Andrey = Role.Liar ∧
    a Friend.Kirill = Role.Cunning :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_consistent_assignment_l635_63576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_equivalence_l635_63546

/-- The function f(x) defined as the square root of a quadratic expression -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x^2 + 4 * x + 1)

/-- The theorem stating the equivalence between the range of f and the range of a -/
theorem range_equivalence (a : ℝ) :
  (∀ y : ℝ, y ≥ 0 → ∃ x : ℝ, f a x = y) ↔ (0 ≤ a ∧ a ≤ 4) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_equivalence_l635_63546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l635_63515

/-- Given a parabola and a line passing through its focus, 
    prove the length of the chord cut by the line on the parabola. -/
theorem parabola_chord_length :
  let parabola := {p : ℝ × ℝ | p.1^2 = 4*p.2}
  let focus : ℝ × ℝ := (0, 1)
  let line_angle : ℝ := 3*π/4
  let line := {p : ℝ × ℝ | p.2 = -p.1 + 1}
  let chord_length : ℝ := 8
  (∀ p : ℝ × ℝ, p ∈ parabola → p.1^2 = 4*p.2) →
  (focus ∈ line) →
  (∀ p : ℝ × ℝ, p ∈ line → p.2 = -p.1 + 1) →
  (∃ p₁ p₂ : ℝ × ℝ, 
    p₁ ∈ parabola ∧ p₁ ∈ line ∧
    p₂ ∈ parabola ∧ p₂ ∈ line ∧
    p₁.1 ≠ p₂.1 ∧
    chord_length = p₁.2 + p₂.2 + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l635_63515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_downpayment_percentage_l635_63505

/-- Calculates the percentage of house cost needed for downpayment --/
noncomputable def downpayment_percentage (salary : ℝ) (savings_rate : ℝ) (house_cost : ℝ) (saving_time : ℕ) : ℝ :=
  (salary * savings_rate * (saving_time : ℝ) / house_cost) * 100

/-- Theorem: Given Mike's financial situation, the downpayment percentage is 20% --/
theorem mikes_downpayment_percentage :
  downpayment_percentage 150000 0.1 450000 6 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mikes_downpayment_percentage_l635_63505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_of_zero_specific_dilation_result_l635_63529

noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

theorem dilation_of_zero (center : ℂ) (scale : ℝ) :
  dilation center scale 0 = center + scale • (-center) :=
by sorry

theorem specific_dilation_result :
  dilation (1 + 2*I) 2 0 = -1 - 2*I :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_of_zero_specific_dilation_result_l635_63529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_sets_l635_63569

/-- A function that returns the sum of n consecutive integers starting from a -/
def consecutiveSum (a n : ℕ) : ℕ := n * (2 * a + n - 1) / 2

/-- A function that checks if a given pair (a, n) forms a valid set of consecutive integers summing to 180 -/
def isValidSet (a n : ℕ) : Prop :=
  n ≥ 2 ∧ consecutiveSum a n = 180

/-- The main theorem stating that there are exactly 4 valid sets -/
theorem exactly_four_sets : 
  ∃! (s : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ s ↔ isValidSet p.1 p.2) ∧ s.card = 4 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_sets_l635_63569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_identical_planes_l635_63572

/-- The distance between two planes given by the equations 
    2x - 4y + 4z = 10 and 4x - 8y + 8z = 20 is 0. -/
theorem distance_between_identical_planes :
  let plane1 := (λ (x y z : ℝ) ↦ 2*x - 4*y + 4*z = 10)
  let plane2 := (λ (x y z : ℝ) ↦ 4*x - 8*y + 8*z = 20)
  ∀ (distance : (ℝ → ℝ → ℝ → Prop) → (ℝ → ℝ → ℝ → Prop) → ℝ),
  distance plane1 plane2 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_identical_planes_l635_63572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l635_63543

/-- The area of a triangle given its three vertices -/
noncomputable def triangleArea (A B C : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (1/2) * abs (x₁ * (y₂ - y₃) + x₂ * (y₃ - y₁) + x₃ * (y₁ - y₂))

/-- Theorem: The area of the triangle with vertices at (-2, 3), (5, -1), and (2, 6) is 18.5 square units -/
theorem triangle_area_example : triangleArea (-2, 3) (5, -1) (2, 6) = 37/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l635_63543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l635_63565

-- Define the function f as noncomputable due to the use of Real.sqrt
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x^2 + 2*x + 3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l635_63565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_is_unique_l635_63544

-- Define the set of colleagues
inductive Colleague
| David
| Emma
| Fiona
| George

-- Define a type for rankings
def Ranking := Colleague → Fin 4

-- Define the statements
def statement1 (r : Ranking) : Prop := r Colleague.Emma ≠ 3
def statement2 (r : Ranking) : Prop := r Colleague.David = 0
def statement3 (r : Ranking) : Prop := r Colleague.George > r Colleague.Fiona
def statement4 (r : Ranking) : Prop := r Colleague.Fiona ≠ 0

-- Define the condition that exactly one statement is true
def exactlyOneTrue (r : Ranking) : Prop :=
  (statement1 r ∧ ¬statement2 r ∧ ¬statement3 r ∧ ¬statement4 r) ∨
  (¬statement1 r ∧ statement2 r ∧ ¬statement3 r ∧ ¬statement4 r) ∨
  (¬statement1 r ∧ ¬statement2 r ∧ statement3 r ∧ ¬statement4 r) ∨
  (¬statement1 r ∧ ¬statement2 r ∧ ¬statement3 r ∧ statement4 r)

-- Define the correct ranking
def correctRanking : Ranking :=
  fun c => match c with
  | Colleague.George => 3
  | Colleague.Emma => 2
  | Colleague.David => 1
  | Colleague.Fiona => 0

-- Theorem statement
theorem correct_ranking_is_unique :
  ∀ r : Ranking,
    (∀ c1 c2 : Colleague, c1 ≠ c2 → r c1 ≠ r c2) →
    exactlyOneTrue r →
    r = correctRanking :=
sorry

#check correct_ranking_is_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_ranking_is_unique_l635_63544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_circle_minus_one_l635_63585

theorem definite_integral_sqrt_circle_minus_one :
  (∫ (x : ℝ) in Set.Icc 0 1, (Real.sqrt (1 - (x - 1)^2) - 1)) = π / 4 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_definite_integral_sqrt_circle_minus_one_l635_63585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_error_percent_l635_63581

theorem rectangle_area_error_percent (L W : ℝ) (L_measured W_measured : ℝ) 
  (h1 : L_measured = 1.15 * L) 
  (h2 : W_measured = 0.89 * W) : 
  (L_measured * W_measured - L * W) / (L * W) * 100 = 2.35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_error_percent_l635_63581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l635_63564

/-- The line equation: x - √3y + 2 = 0 -/
def line_eq (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

/-- The circle equation: x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Points A and B are on both the line and the circle -/
def intersection (A B : ℝ × ℝ) : Prop :=
  line_eq A.1 A.2 ∧ circle_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ circle_eq B.1 B.2

/-- The absolute value of the projection of AB on the positive x-axis -/
def projection (A B : ℝ × ℝ) : ℝ := |B.1 - A.1|

theorem projection_value (A B : ℝ × ℝ) (h : intersection A B) : projection A B = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_value_l635_63564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonEquilateralTriangleCombinations_six_dots_l635_63518

/-- Given 6 dots evenly spaced on the circumference of a circle, 
    this function returns the number of combinations of 3 dots 
    that do not form an equilateral triangle. -/
def nonEquilateralTriangleCombinations (n : Nat) : Nat :=
  if n = 6 then
    Nat.choose n 3 - 2
  else
    0

/-- Theorem stating that for 6 evenly spaced dots on a circle,
    there are 18 combinations of 3 dots that do not form an equilateral triangle. -/
theorem nonEquilateralTriangleCombinations_six_dots :
  nonEquilateralTriangleCombinations 6 = 18 := by
  sorry

#eval nonEquilateralTriangleCombinations 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonEquilateralTriangleCombinations_six_dots_l635_63518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l635_63555

open Real

theorem angle_relations (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 π)
  (h2 : β ∈ Set.Ioo 0 π)
  (h3 : sin α - cos α = 17/13)
  (h4 : sin β + cos β = 17/13) :
  α > β ∧ α > 3*β ∧ π/2 < α + β ∧ α + β < 3*π/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_relations_l635_63555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_2017_l635_63596

/-- Represents the triangular arrangement of positive integers -/
def TriangularArrangement : ℕ → ℕ → Prop :=
  fun row col => row > 0 ∧ col > 0 ∧ col ≤ 2 * row - 1

/-- The last number in each row -/
def LastInRow (n : ℕ) : ℕ := n ^ 2

/-- The total count of numbers up to and including row n -/
def TotalUpToRow (n : ℕ) : ℕ := n ^ 2

/-- The position of a number in the triangular arrangement -/
structure Position where
  row : ℕ
  col : ℕ

/-- The theorem stating the position of 2017 in the triangular arrangement -/
theorem position_of_2017 : 
  ∃ (p : Position), TriangularArrangement p.row p.col ∧ p.row = 45 ∧ p.col = 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_position_of_2017_l635_63596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_parallelogram_l635_63522

/-- ConvexQuadrilateral represents a convex quadrilateral in 2D space -/
def ConvexQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  sorry

/-- SegmentLength calculates the length of a line segment between two points -/
def SegmentLength (P Q : ℝ × ℝ) : ℝ :=
  sorry

/-- AngleMeasure calculates the measure of an angle at a vertex of a quadrilateral -/
def AngleMeasure (V : ℝ × ℝ) : ℝ :=
  sorry

/-- IsParallelogram checks if a quadrilateral is a parallelogram -/
def IsParallelogram (A B C D : ℝ × ℝ) : Prop :=
  sorry

/-- A convex quadrilateral with two equal sides and two equal angles is not necessarily a parallelogram -/
theorem not_necessarily_parallelogram :
  ∃ (A B C D : ℝ × ℝ), 
    ConvexQuadrilateral A B C D ∧
    SegmentLength A B = SegmentLength C D ∧
    AngleMeasure A = AngleMeasure C ∧
    ¬IsParallelogram A B C D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_parallelogram_l635_63522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_sum_and_difference_l635_63599

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem angle_between_sum_and_difference (a b : E) (ha : a ≠ 0) (hb : b ≠ 0) (h_norm : ‖a‖ = ‖b‖) :
  InnerProductGeometry.angle (a + b) (a - b) = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_sum_and_difference_l635_63599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_fourier_cosine_transform_l635_63523

/-- The Fourier cosine transform F(p) -/
noncomputable def F (p : ℝ) : ℝ :=
  if 0 < p ∧ p < 1 then 1 else
  if 1 < p then 0 else 0

/-- The inverse Fourier cosine transform f(x) -/
noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (2 / Real.pi) * Real.sin x / x

/-- Theorem stating that f is the inverse Fourier cosine transform of F -/
theorem inverse_fourier_cosine_transform :
  ∀ x : ℝ, f x = Real.sqrt (2 / Real.pi) * ∫ p in Set.Ioi 0, F p * Real.cos (p * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_fourier_cosine_transform_l635_63523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_eq_two_l635_63562

/-- A line with equation x + (2-a)y + 1 = 0 -/
def line (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + (2 - a) * p.2 + 1 = 0}

/-- A circle with equation x^2 + y^2 - 2y = 0 -/
def my_circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.2 = 0}

/-- Definition of a line being tangent to a circle -/
def is_tangent (l : Set (ℝ × ℝ)) (c : Set (ℝ × ℝ)) : Prop :=
  ∃ p, p ∈ l ∧ p ∈ c ∧ ∀ q ≠ p, q ∈ l → q ∉ c

/-- Theorem stating that if the line is tangent to the circle, then a = 2 -/
theorem tangent_line_implies_a_eq_two (a : ℝ) :
  is_tangent (line a) my_circle → a = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_eq_two_l635_63562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l635_63550

theorem problem_solution (m n : ℝ) 
  (h1 : m + n = 2) 
  (h2 : m * n = -2) : 
  (2:ℝ)^m * (2:ℝ)^n - ((2:ℝ)^m)^n = 15/4 ∧ 
  (m - 4) * (n - 4) = 6 ∧ 
  (m - n)^2 = 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l635_63550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l635_63536

/-- Definition of the ellipse C -/
noncomputable def ellipse_C (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- Point P on the ellipse -/
def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse_C (Real.sqrt 2) 1 a b

/-- Theorem statement -/
theorem ellipse_properties (a b : ℝ) :
  ellipse_C (Real.sqrt 2) 1 a b ∧ eccentricity a b = Real.sqrt 2 / 2 →
  (∀ x y, ellipse_C x y a b ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ y₀ x₁ y₁, 
    x₁^2 / 4 + y₁^2 / 2 = 1 → 
    (x₁ * 2 + y₁ * y₀ = 4)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l635_63536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_approx_l635_63587

/-- The rate per meter for fencing a circular field -/
noncomputable def fencing_rate_per_meter (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (Real.pi * diameter)

/-- Theorem: The fencing rate per meter for a circular field with diameter 36m and total cost 395.84 is approximately 3.5 -/
theorem fencing_rate_approx (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, 0 < δ ∧ |fencing_rate_per_meter 36 395.84 - 3.5| < δ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fencing_rate_approx_l635_63587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_distribution_l635_63578

/-- Represents the direction of movement around the circle -/
inductive Direction
| Clockwise
| Counterclockwise

/-- Represents a piece color -/
inductive PieceColor
| White
| Red

/-- Represents the state of the boxes -/
structure BoxState :=
  (white_pieces : Fin 7 → Nat)
  (red_pieces : Fin 7 → Nat)

/-- The rule for placing pieces -/
def place_pieces (initial_box : Nat) (num_pieces : Nat) (direction : Direction) : BoxState :=
  sorry

/-- The final state after both players have placed their pieces -/
def final_state : BoxState :=
  let white_state := place_pieces 1 200 Direction.Clockwise
  let red_state := place_pieces 1 300 Direction.Counterclockwise
  { white_pieces := white_state.white_pieces,
    red_pieces := red_state.red_pieces }

theorem correct_distribution :
  (∀ i : Fin 7, final_state.white_pieces i = [57, 0, 58, 0, 0, 29, 56].get i) ∧
  (∀ i : Fin 7, final_state.red_pieces i = [86, 85, 43, 0, 0, 86, 0].get i) ∧
  (∀ i : Fin 7, final_state.white_pieces i + final_state.red_pieces i = [143, 85, 101, 0, 0, 115, 56].get i) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_distribution_l635_63578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_11_squared_plus_90_squared_l635_63592

theorem largest_prime_divisor_of_11_squared_plus_90_squared :
  (Nat.factors (11^2 + 90^2)).maximum? = some 89 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_11_squared_plus_90_squared_l635_63592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_solution_l635_63534

-- Define the type for statements
inductive Statement : Type
| A : Statement
| B : Statement
| C : Statement
| D : Statement

-- Define a function that returns the number of true statements claimed by each statement
def claimed_true_count (s : Statement) : Nat :=
  match s with
  | Statement.A => 1
  | Statement.B => 2
  | Statement.C => 3
  | Statement.D => 4

-- Define a predicate for a valid solution
def is_valid_solution (true_statements : Finset Statement) : Prop :=
  ∃ (s : Statement), s ∈ true_statements ∧ 
    (∀ (t : Statement), t ∈ true_statements ↔ t = s) ∧
    claimed_true_count s = true_statements.card

-- Theorem stating that there exists a unique valid solution
theorem unique_valid_solution : 
  ∃! (true_statements : Finset Statement), is_valid_solution true_statements := by
  sorry

#check unique_valid_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_valid_solution_l635_63534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l635_63547

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ) + Real.cos (ω * x + φ)

theorem f_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : |φ| < π/2) 
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) 
  (h_point : f ω φ 0 = Real.sqrt 2) :
  ∃ (g : ℝ → ℝ), 
    (∀ x, f ω φ x = g x) ∧ 
    (∀ x, g x = Real.sqrt 2 * Real.cos (2 * x)) ∧
    (∀ x, g x ≤ Real.sqrt 2) ∧
    (∀ x ∈ Set.Ioo 0 (π/2), ∀ y ∈ Set.Ioo 0 (π/2), x < y → g x > g y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l635_63547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l635_63531

-- Define the given constants
def train_length : ℚ := 240
def time_to_pass : ℚ := 26.64
def train_speed_kmh : ℚ := 50

-- Define the function to convert km/h to m/s
def kmh_to_ms (speed : ℚ) : ℚ := speed * 1000 / 3600

-- Define the function to calculate the bridge length
def bridge_length (train_length time_to_pass train_speed_kmh : ℚ) : ℚ :=
  kmh_to_ms train_speed_kmh * time_to_pass - train_length

-- State the theorem
theorem bridge_length_calculation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |bridge_length train_length time_to_pass train_speed_kmh - 129.91| < ε :=
by
  -- We'll use 'sorry' to skip the proof for now
  sorry

#eval bridge_length train_length time_to_pass train_speed_kmh

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l635_63531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_condition_l635_63513

/-- Represents a point in a geometric space -/
structure Point where
  x : ℝ

/-- Represents a line segment between two points -/
def LineSegment (A B : Point) : ℝ :=
  abs (A.x - B.x)

/-- Theorem: If distinct collinear points E, F, G, H (in that order) with EF = p, EG = q, EH = r
    can form a non-degenerate triangle by rotating EF and GH, then p < r/3 must be true -/
theorem triangle_formation_condition 
  (E F G H : Point) 
  (p q r : ℝ) 
  (distinct : E ≠ F ∧ F ≠ G ∧ G ≠ H)
  (collinear : E.x < F.x ∧ F.x < G.x ∧ G.x < H.x)
  (lengths : LineSegment E F = p ∧ LineSegment E G = q ∧ LineSegment E H = r)
  (rotatable : ∃ (E' H' : Point), E'.x = H'.x ∧ 
    LineSegment E' F = p ∧ 
    LineSegment G H' = LineSegment G H ∧
    LineSegment E' F + LineSegment F G > LineSegment G H' ∧
    LineSegment F G + LineSegment G H' > LineSegment E' F ∧
    LineSegment E' F + LineSegment G H' > LineSegment F G) :
  p < r / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_formation_condition_l635_63513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l635_63500

theorem factorial_simplification : 
  (13 * 12 * 11 * Nat.factorial 10) / (Nat.factorial 10 + 3 * Nat.factorial 9) = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_simplification_l635_63500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l635_63577

/-- The area of the region formed by two tangent circles and a tangent line --/
theorem shaded_area_theorem (r : ℝ) (h : r = 10) : 
  let area := r * r * (Real.sqrt 3 - Real.pi)
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |area - 8| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_theorem_l635_63577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_min_l635_63591

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Pyramid structure -/
structure Pyramid where
  S : Point
  A : Point
  B : Point
  C : Point
  D : Point

/-- Volume of a pyramid -/
noncomputable def volume (p : Pyramid) : ℝ := sorry

/-- Theorem statement -/
theorem pyramid_volume_min (SABCD : Pyramid) 
  (P₁ P₂ P₃ Q₁ Q₂ Q₃ R₁ R₂ R₃ R₄ : Point) 
  (h_P : SABCD.B.x < P₁.x ∧ P₁.x < P₂.x ∧ P₂.x < P₃.x ∧ P₃.x < SABCD.C.x)
  (h_Q : SABCD.A.x < Q₁.x ∧ Q₁.x < Q₂.x ∧ Q₂.x < Q₃.x ∧ Q₃.x < SABCD.D.x)
  (h_R₁ : sorry) -- Placeholder for line intersection condition
  (h_R₂ : sorry) -- Placeholder for line intersection condition
  (h_R₃ : sorry) -- Placeholder for line intersection condition
  (h_R₄ : sorry) -- Placeholder for line intersection condition
  (h_vol : volume {S := SABCD.S, A := R₁, B := P₁, C := R₂, D := Q₁} + 
           volume {S := SABCD.S, A := R₃, B := P₃, C := R₄, D := Q₃} = 78) :
  (volume {S := SABCD.S, A := SABCD.A, B := SABCD.B, C := R₁, D := SABCD.D})^2 + 
  (volume {S := SABCD.S, A := R₂, B := P₂, C := R₃, D := Q₂})^2 + 
  (volume {S := SABCD.S, A := SABCD.C, B := SABCD.D, C := R₄, D := SABCD.D})^2 ≥ 2028 :=
by
  sorry -- Placeholder for the actual proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_min_l635_63591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_no_real_roots_l635_63507

-- Define the three equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 12 = 10
def equation2 (x : ℝ) : Prop := abs (3*x - 2) = abs (x + 2)
def equation3 (x : ℝ) : Prop := Real.sqrt (4*x^2 + 16) = Real.sqrt (x^2 + x + 1)

-- Theorem statement
theorem at_least_one_no_real_roots : 
  ∃ (eq : ℝ → Prop), (eq ∈ ({equation1, equation2, equation3} : Set (ℝ → Prop))) ∧ 
  (∀ x : ℝ, ¬(eq x)) :=
by
  -- We'll use equation3 as our example of an equation with no real roots
  use equation3
  constructor
  · simp [Set.mem_insert]
  · intro x
    -- Proof that equation3 has no real solutions
    sorry  -- This part would require a more detailed proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_no_real_roots_l635_63507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l635_63521

noncomputable def complex_number : ℂ := (3 * Complex.I) / (1 - Complex.I)

theorem complex_number_in_second_quadrant :
  Real.sign complex_number.re = -1 ∧ Real.sign complex_number.im = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_second_quadrant_l635_63521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garlic_cloves_needed_l635_63571

/-- Represents the effectiveness rate of garlic for different creature types -/
structure GarlicEffectiveness where
  vampires : ℚ
  wights : ℚ
  vampire_bats : ℚ
  werewolves : ℚ
  ghosts : ℚ

/-- Represents the count of different creature types -/
structure CreatureCount where
  vampires : ℕ
  wights : ℕ
  vampire_bats : ℕ
  werewolves : ℕ
  ghosts : ℕ

/-- Calculates the number of garlic cloves needed for a specific creature type -/
def cloves_needed (effectiveness : ℚ) (count : ℕ) : ℕ :=
  ⌈(3 / effectiveness) * count⌉.toNat

/-- Calculates the total number of garlic cloves needed for all creatures -/
def total_cloves_needed (effectiveness : GarlicEffectiveness) (count : CreatureCount) : ℕ :=
  cloves_needed effectiveness.vampires count.vampires +
  cloves_needed effectiveness.wights count.wights +
  cloves_needed effectiveness.vampire_bats count.vampire_bats +
  cloves_needed effectiveness.werewolves count.werewolves +
  cloves_needed effectiveness.ghosts count.ghosts

/-- Theorem stating that 476 cloves of garlic are needed to repel all creatures -/
theorem garlic_cloves_needed :
  let effectiveness : GarlicEffectiveness := ⟨1, 1, 3/4, 3/5, 1/2⟩
  let count : CreatureCount := ⟨30, 12, 40, 20, 15⟩
  total_cloves_needed effectiveness count = 476 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garlic_cloves_needed_l635_63571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_functions_l635_63559

open Real

/-- Function A --/
noncomputable def funcA (x : ℝ) : ℝ := 2^x + 4 / (2^x)

/-- Function B --/
noncomputable def funcB (x : ℝ) : ℝ := log x + 4 / (log x)

/-- Function C --/
noncomputable def funcC (x : ℝ) : ℝ := abs (sin x) + 4 / abs (sin x)

/-- Function D --/
noncomputable def funcD (x : ℝ) : ℝ := x^2 + 4 / (x^2)

theorem min_value_functions :
  (∃ x : ℝ, funcA x = 4) ∧
  (∃ x : ℝ, funcD x = 4) ∧
  (∀ x : ℝ, funcA x ≥ 4) ∧
  (∀ x : ℝ, funcD x ≥ 4) ∧
  (∀ x : ℝ, x > 0 → funcB x > 4) ∧
  (∀ x : ℝ, funcC x > 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_functions_l635_63559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_APR_l635_63558

-- Define the circle
variable (circle : Set (EuclideanSpace ℝ (Fin 2)))

-- Define the points
variable (A B C P Q R : EuclideanSpace ℝ (Fin 2))

-- Define the tangent property
def is_tangent (p q : EuclideanSpace ℝ (Fin 2)) (circle : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define the condition that A is outside the circle
def outside_circle (p : EuclideanSpace ℝ (Fin 2)) (c : Set (EuclideanSpace ℝ (Fin 2))) : Prop := sorry

-- Define the intersection property
def intersects_at (l₁ l₂ : Set (EuclideanSpace ℝ (Fin 2))) (p : EuclideanSpace ℝ (Fin 2)) : Prop := sorry

-- Define the line through two points
def line (p q : EuclideanSpace ℝ (Fin 2)) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the perimeter of a triangle
noncomputable def triangle_perimeter (p q r : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

-- Theorem statement
theorem perimeter_of_APR (h₁ : outside_circle A circle)
                         (h₂ : is_tangent A B circle)
                         (h₃ : is_tangent A C circle)
                         (h₄ : is_tangent P Q circle)
                         (h₅ : intersects_at (line A B) (line P Q) P)
                         (h₆ : intersects_at (line A C) (line P Q) R)
                         (h₇ : ‖A - B‖ = 20) :
  triangle_perimeter A P R = 40 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_of_APR_l635_63558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l635_63527

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (2 - x) / (3 * x + 4)

-- Theorem statement
theorem unattainable_value :
  ¬∃ (x : ℝ), x ≠ -4/3 ∧ f x = -1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unattainable_value_l635_63527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_l635_63506

theorem student_arrangement (n : ℕ) (m : ℕ) :
  n = 5 ∧ m = 2 →
  (m * Nat.factorial (n - m)) = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_arrangement_l635_63506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l635_63539

/-- The x-intercept of a line passing through two given points -/
noncomputable def x_intercept (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  let m := (y₂ - y₁) / (x₂ - x₁)
  x₁ - y₁ / m

/-- Theorem: The x-intercept of the line passing through (10, 3) and (-4, -4) is 4 -/
theorem x_intercept_specific_line :
  x_intercept 10 3 (-4) (-4) = 4 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l635_63539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l635_63552

/-- Represents the dimensions of a rectangular floor. -/
structure FloorDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a square tile. -/
structure TileDimensions where
  side : ℝ

/-- Calculates the shaded area of a floor given its dimensions, tile dimensions, and the radius of the quarter circles in each tile corner. -/
noncomputable def shaded_area (floor : FloorDimensions) (tile : TileDimensions) (circle_radius : ℝ) : ℝ :=
  let total_area := floor.length * floor.width
  let tile_area := tile.side * tile.side
  let num_tiles := total_area / tile_area
  let white_area_per_tile := Real.pi * circle_radius^2
  let shaded_area_per_tile := tile_area - white_area_per_tile
  num_tiles * shaded_area_per_tile

/-- Theorem stating that the shaded area of a 12x14 foot floor with 2x2 foot tiles and 1-foot radius quarter circles is 168 - 42π square feet. -/
theorem shaded_area_calculation :
  let floor := FloorDimensions.mk 12 14
  let tile := TileDimensions.mk 2
  let circle_radius := 1
  shaded_area floor tile circle_radius = 168 - 42 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l635_63552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l635_63584

/-- Hyperbola C with eccentricity √2 and point T(3, √5) on it -/
def HyperbolaC (a b : ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ C) ∧
  (∃ c : ℝ, c^2 = a^2 + b^2 ∧ c / a = Real.sqrt 2) ∧
  9 / a^2 - 5 / b^2 = 1

/-- The equation of hyperbola C and the fixed point property -/
theorem hyperbola_properties (a b : ℝ) (C : Set (ℝ × ℝ)) (h : HyperbolaC a b C) :
  (∀ x y : ℝ, x^2 / 4 - y^2 / 4 = 1 ↔ (x, y) ∈ C) ∧
  (∀ m : ℝ, ∃ P Q : ℝ × ℝ,
    P ∈ C ∧ Q ∈ C ∧
    P ≠ (-2, 0) ∧ Q ≠ (2, 0) ∧
    (∃ t : ℝ, P.1 = t * (P.2 + 2) - 2 ∧ P.2 = t * (m - 0) + 0) ∧
    (∃ s : ℝ, Q.1 = s * (Q.2 - 0) + 2 ∧ Q.2 = s * (m - 0) + 0) ∧
    (∃ k : ℝ, Q.2 - P.2 = k * (Q.1 - P.1) ∧ k * 4 + Q.2 = 0)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l635_63584
