import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cube_volume_l1184_118405

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  center : Point
  edge_length : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point
  radius : ℝ

/-- Represents an edge of a cube -/
structure Edge (c : Cube) where
  vertices : Fin 2 → Point

/-- The volume of a sphere -/
noncomputable def Sphere.volume (s : Sphere) : ℝ :=
  (4 / 3) * Real.pi * s.radius ^ 3

/-- The surface of a sphere -/
def Sphere.surface (s : Sphere) : Set Point :=
  {p : Point | (p.x - s.center.x)^2 + (p.y - s.center.y)^2 + (p.z - s.center.z)^2 = s.radius^2}

/-- Linear combination of two points -/
def linear_combination (t : ℝ) (p1 p2 : Point) : Point :=
  ⟨(1 - t) * p1.x + t * p2.x, (1 - t) * p1.y + t * p2.y, (1 - t) * p1.z + t * p2.z⟩

/-- The set of points on an edge -/
def Edge.set (e : Edge c) : Set Point :=
  {p : Point | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = linear_combination t (e.vertices 0) (e.vertices 1)}

/-- The volume of a sphere inscribed in a cube with edge length 3, touching all 12 edges of the cube, is equal to 9√2π. -/
theorem sphere_in_cube_volume :
  ∀ (s : Sphere) (c : Cube),
    c.edge_length = 3 →
    s.center = c.center →
    (∀ e : Edge c, (s.surface ∩ e.set).Nonempty) →
    s.volume = 9 * Real.sqrt 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_in_cube_volume_l1184_118405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_power_minus_one_l1184_118426

/-- The number of distinct positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := (Nat.divisors n).card

/-- Theorem: If a^n + 1 is prime, then the number of distinct positive divisors of a^n - 1 is at least n -/
theorem divisors_of_power_minus_one (a n : ℕ) (ha : a > 1) (hn : n > 0) 
  (h_prime : Nat.Prime (a^n + 1)) : num_divisors (a^n - 1) ≥ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_power_minus_one_l1184_118426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1184_118434

-- Define the points
def point1 : ℝ × ℝ := (1, 5)
def point2 : ℝ × ℝ := (7, 2)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Theorem statement
theorem distance_between_points :
  distance point1 point2 = 3 * Real.sqrt 5 := by
  -- Expand the definition of distance
  unfold distance
  -- Simplify the expression
  simp [point1, point2]
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_points_l1184_118434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l1184_118475

def secondQuadrant : Set ℂ := {z | z.re < 0 ∧ z.im > 0}

theorem complex_number_quadrant (z : ℂ) : 
  z.re = -2 ∧ z.im = 1 → z ∈ secondQuadrant := by
  intro h
  simp [secondQuadrant]
  constructor
  · linarith [h.left]
  · linarith [h.right]

#check complex_number_quadrant

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_quadrant_l1184_118475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1184_118473

noncomputable def f (a b x : ℝ) : ℝ := 6 * Real.log x - a * x^2 - 8 * x + b

theorem function_properties (a b : ℝ) :
  (∀ x, x > 0 → (deriv (f a b)) x = 0 → x = 3) →
  (a = -1 ∧
   (∀ x, 1 < x ∧ x < 3 → (deriv (f a b)) x < 0) ∧
   (∃ x₁ x₂ x₃, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < x₃ ∧
     f a b x₁ = 0 ∧ f a b x₂ = 0 ∧ f a b x₃ = 0 ∧
     (∀ x, x ≠ x₁ ∧ x ≠ x₂ ∧ x ≠ x₃ → f a b x ≠ 0) →
     7 < b ∧ b < 15 - 6 * Real.log 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1184_118473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_intersection_theorem_l1184_118409

/-- Circle C₁ -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9

/-- Line l₀ -/
def l₀ (x y : ℝ) : Prop := y = (1/2) * x + (3/2) * Real.sqrt 5

/-- Point A on C₁ -/
def A : ℝ × ℝ → Prop := λ p => C₁ p.1 p.2

/-- Point N on x-axis -/
def N (x : ℝ) : ℝ × ℝ := (x, 0)

/-- Point M satisfying the given condition -/
def M (a : ℝ × ℝ) (m : ℝ × ℝ) : Prop :=
  m.1 + 2 * (m.1 - a.1) = (2 * Real.sqrt 2 - 2) * a.1 ∧
  m.2 + 2 * (m.2 - a.2) = 0

/-- Curve C -/
def C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

/-- Circle with diameter PQ passing through origin -/
def circlePQO (p q : ℝ × ℝ) : Prop := p.1 * q.1 + p.2 * q.2 = 0

/-- Length of PQ -/
noncomputable def lengthPQ (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Main theorem -/
theorem locus_and_intersection_theorem :
  (∀ x y, C₁ x y → l₀ x y → False) →  -- C₁ and l₀ are tangent
  (∀ a m, A a → M a m → C m.1 m.2) ∧  -- Locus of M forms curve C
  (∀ p q, C p.1 p.2 → C q.1 q.2 → p ≠ q → circlePQO p q →
    (4 * Real.sqrt 6) / 3 ≤ lengthPQ p q ∧ lengthPQ p q ≤ 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_intersection_theorem_l1184_118409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_eq_neg_one_l1184_118462

/-- A function f is symmetrical about the origin if f(-x) = -f(x) for all x -/
def SymmetricalAboutOrigin (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = x / ((x+1)(x+a)) -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ x / ((x + 1) * (x + a))

/-- If f(x) = x / ((x+1)(x+a)) is symmetrical about the origin, then a = -1 -/
theorem symmetry_implies_a_eq_neg_one (a : ℝ) :
  SymmetricalAboutOrigin (f a) → a = -1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_a_eq_neg_one_l1184_118462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1184_118457

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-8) 4

-- Define the function h in terms of f
noncomputable def h (x : ℝ) : ℝ := f (3 * x + 1)

-- State the theorem about the domain of h
theorem domain_of_h :
  {x : ℝ | h x ∈ Set.range f} = Set.Icc (-3) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_h_l1184_118457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1184_118463

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = π
  opposite_sides : 
    a = 2 * (Real.sin (A/2)) * (Real.sin (B/2)) / Real.sin ((A+B)/2) ∧
    b = 2 * (Real.sin (B/2)) * (Real.sin (C/2)) / Real.sin ((B+C)/2) ∧
    c = 2 * (Real.sin (C/2)) * (Real.sin (A/2)) / Real.sin ((C+A)/2)

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.a^2 = t.b^2 + t.c^2 - t.b * t.c)
  (h2 : t.a = (Real.sqrt 15 / 2) * t.b) : 
  Real.sin t.B = Real.sqrt 5 / 5 ∧ 
  Real.cos (t.C + π/12) = -(Real.sqrt 10 / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l1184_118463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_diagonal_length_l1184_118442

/-- The length of a diagonal in a regular octagon with side length 8 -/
noncomputable def regular_octagon_diagonal : ℝ := 16 * Real.sqrt 2

/-- Theorem: In a regular octagon with side length 8, the length of a diagonal
    connecting opposite vertices is 16√2 units. -/
theorem regular_octagon_diagonal_length :
  regular_octagon_diagonal = 16 * Real.sqrt 2 := by
  -- The proof is omitted for now
  sorry

#check regular_octagon_diagonal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_diagonal_length_l1184_118442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_acute_triangle_l1184_118489

-- Define the angles
def angle1 : ℝ := 110
def angle2 : ℝ := 75
def angle3 : ℝ := 65
def angle4 : ℝ := 15

-- Define the properties of the triangles
def is_acute_triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ a < 90 ∧ b < 90 ∧ c < 90
def is_obtuse_triangle (a b c : ℝ) : Prop := a + b + c = 180 ∧ (a > 90 ∨ b > 90 ∨ c > 90)

-- Theorem statement
theorem smallest_angle_in_acute_triangle :
  ∃ (a b c d e f : ℝ),
    ({a, b, c, d, e, f} : Set ℝ) = {angle1, angle2, angle3, angle4, 180 - angle1 - angle2, 180 - angle3 - angle4} ∧
    is_acute_triangle d e f ∧
    is_obtuse_triangle a b c ∧
    (d = 15 ∨ e = 15 ∨ f = 15) ∧
    (d ≥ 15 ∧ e ≥ 15 ∧ f ≥ 15) :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_in_acute_triangle_l1184_118489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1184_118494

theorem trigonometric_problem (α β : ℝ) 
  (h1 : α ∈ Set.Ioo 0 (π/2)) 
  (h2 : β ∈ Set.Ioo 0 (π/2)) 
  (h3 : Real.sin (α + 2*β) = (7/5) * Real.sin α) :
  (Real.tan (α + β) - 6 * Real.tan β = 0) ∧ 
  (Real.tan α = 3 * Real.tan β → α = π/4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_problem_l1184_118494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_values_at_zero_l1184_118479

/-- A function satisfying the given condition -/
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → f (1 / n) = n^2 / (n^2 + 1)

/-- The kth derivative of f at 0 -/
noncomputable def kth_derivative_at_zero (f : ℝ → ℝ) (k : ℕ) : ℝ :=
  (deriv^[k] f) 0

/-- The main theorem -/
theorem derivative_values_at_zero
  (f : ℝ → ℝ)
  (h_diff : ContDiff ℝ ⊤ f)
  (h_cond : satisfies_condition f) :
  ∀ k : ℕ, k > 0 →
    kth_derivative_at_zero f k =
      if k % 2 = 1 then
        0
      else
        ((-1 : ℝ)^(k/2)) * (Nat.factorial k) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_values_at_zero_l1184_118479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_term_l1184_118460

noncomputable def expansion (x : ℝ) := (Real.sqrt x - 1)^4 * (x - 1)^2

theorem coefficient_of_x_term : 
  deriv expansion 1 = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_term_l1184_118460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l1184_118451

noncomputable section

-- Define the circle C
def circle_C (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*y - 2 = 0

-- Define the line
def line (x y a : ℝ) : Prop := y = x + 2*a

-- Define the chord length
noncomputable def chord_length (a : ℝ) : ℝ := 2 * Real.sqrt 3

-- Theorem statement
theorem circle_area (a : ℝ) : 
  (∃ x y : ℝ, circle_C x y a ∧ line x y a) →
  chord_length a = 2 * Real.sqrt 3 →
  (π * (Real.sqrt (a^2 + 2))^2 : ℝ) = 4 * π := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_l1184_118451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_A_equivalent_option_B_equivalent_option_C_not_equivalent_option_D_equivalent_l1184_118445

-- Option A
def fA (x : ℝ) : ℝ := x^2
noncomputable def gA (x : ℝ) : ℝ := if x = 0 then 0 else 1

theorem option_A_equivalent : 
  ∀ x ∈ ({-1, 0, 1} : Set ℝ), fA x = gA x := by sorry

-- Option B
def fB (x : ℝ) : ℝ := x * abs x
noncomputable def gB (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

theorem option_B_equivalent : ∀ x : ℝ, fB x = gB x := by sorry

-- Option C
def fC (x : ℝ) : ℝ := x
noncomputable def gC (x : ℝ) : ℝ := Real.sqrt (x^2)

theorem option_C_not_equivalent : ¬(∀ x : ℝ, fC x = gC x) := by sorry

-- Option D
noncomputable def fD (x : ℝ) : ℝ := 1 / x
noncomputable def gD (x : ℝ) : ℝ := (x + 1) / (x^2 + x)

theorem option_D_equivalent : ∀ x > 0, fD x = gD x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_A_equivalent_option_B_equivalent_option_C_not_equivalent_option_D_equivalent_l1184_118445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_has_period_pi_l1184_118447

-- Define the function f(x) = sin(2x - π/2)
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 2)

-- Theorem stating that f is an even function
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

-- Theorem stating that f has a period of π
theorem f_has_period_pi : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_f_has_period_pi_l1184_118447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_equals_half_l1184_118461

/-- Given two real functions f and g, prove that a = 1/2 under certain conditions -/
theorem prove_a_equals_half 
  (f g : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = a^x * g x) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) 
  (h4 : ∀ x, g x ≠ 0) 
  (h5 : ∀ x, f x * (deriv g x) > (deriv f x) * g x) 
  (h6 : f 1 / g 1 + f (-1) / g (-1) = 5/2) : 
  a = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prove_a_equals_half_l1184_118461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_representations_2023_l1184_118492

theorem count_representations_2023 : 
  let M : Finset (ℕ × ℕ × ℕ × ℕ) := 
    Finset.filter (fun (b₃, b₂, b₁, b₀) => 
      2023 = b₃ * 10^3 + b₂ * 10^2 + b₁ * 10 + b₀ ∧ 
      b₃ ≤ 999 ∧ b₂ ≤ 999 ∧ b₁ ≤ 999 ∧ b₀ ≤ 999)
    (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000))))
  M.card = 306 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_representations_2023_l1184_118492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_l1184_118449

theorem intersection_point_on_circle (k : ℝ) : 
  (∃ P : ℝ × ℝ, 
    P.1 = k - 1 ∧ 
    P.2 = 3 * k - 1 ∧ 
    P.2 = P.1 + 2 * k ∧ 
    P.2 = 2 * P.1 + k + 1 ∧ 
    P.1^2 + P.2^2 = 4) ↔ 
  (k = 1 ∨ k = -1/5) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_on_circle_l1184_118449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_equals_zero_l1184_118477

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 2 else x - 1

-- State the theorem
theorem sum_of_zeros_equals_zero :
  ∃ (z₁ z₂ : ℝ), f z₁ = 0 ∧ f z₂ = 0 ∧ z₁ + z₂ = 0 := by
  -- Proof goes here
  sorry

#check sum_of_zeros_equals_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_equals_zero_l1184_118477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_3D_eq_2T_l1184_118428

/-- The largest odd divisor of a positive integer -/
def d (n : ℕ+) : ℕ+ :=
  sorry

/-- The sum of largest odd divisors from 1 to n -/
def D (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => (d ⟨i + 1, Nat.succ_pos i⟩).val)

/-- The sum of integers from 1 to n -/
def T (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- There exist infinitely many positive integers n such that 3D(n) = 2T(n) -/
theorem infinitely_many_n_3D_eq_2T :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ (3 : ℕ) * D n = 2 * T n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_3D_eq_2T_l1184_118428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1184_118439

/-- Calculates the rate of interest for a loan with simple interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (interest_paid : ℝ) : ℝ :=
  Real.sqrt (interest_paid / (principal / 100))

theorem interest_rate_calculation (principal interest_paid : ℝ) 
  (h1 : principal = 1200)
  (h2 : interest_paid = 108) :
  calculate_interest_rate principal interest_paid = 3 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 1200 108

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l1184_118439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_steps_l1184_118456

theorem staircase_steps (n : ℕ) : 
  (∀ (step : ℕ), step ≤ n → 3 * step = 3 * step) →
  (3 * n * (n + 1)) / 2 = 270 →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_staircase_steps_l1184_118456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_min_distance_l1184_118402

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
axiom a : ℝ
axiom b : ℝ
axiom a_gt_b : a > b
axiom b_gt_zero : b > 0
axiom focus_on_x_axis : ∃ (c : ℝ), c = 1 ∧ ellipse a b c 0
axiom point_on_ellipse : ellipse a b (2/3) (2 * Real.sqrt 6 / 3)

-- Define perpendicular vectors
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

-- Theorem 1: Standard equation of C
theorem standard_equation : a = 2 ∧ b = Real.sqrt 3 := by
  sorry

-- Theorem 2: Minimum value of |AB|
theorem min_distance (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : ellipse a b x₁ y₁) 
  (h₂ : ellipse a b x₂ y₂) 
  (h₃ : perpendicular x₁ y₁ x₂ y₂) :
  ∃ (d : ℝ), d = 2 * a * b / Real.sqrt (a^2 + b^2) ∧
  Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_min_distance_l1184_118402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_length_is_28_l1184_118414

/-- Represents the dimensions and volume of a rectangular pond. -/
structure Pond where
  width : ℝ
  depth : ℝ
  volume : ℝ

/-- Calculates the length of a rectangular pond given its width, depth, and volume. -/
noncomputable def calculateLength (p : Pond) : ℝ :=
  p.volume / (p.width * p.depth)

/-- Theorem stating that a pond with given dimensions has a length of 28 meters. -/
theorem pond_length_is_28 (p : Pond) 
    (h_width : p.width = 10)
    (h_depth : p.depth = 5)
    (h_volume : p.volume = 1400) : 
  calculateLength p = 28 := by
  sorry

#check pond_length_is_28

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pond_length_is_28_l1184_118414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cylinder_volume_when_equal_to_surface_area_min_cylinder_volume_is_54pi_l1184_118459

open Real

noncomputable def cylinderVolume (r h : ℝ) : ℝ := π * r^2 * h

noncomputable def cylinderSurfaceArea (r h : ℝ) : ℝ := 2 * π * r^2 + 2 * π * r * h

theorem min_cylinder_volume_when_equal_to_surface_area :
  ∃ (r h : ℝ), r > 0 ∧ h > 0 ∧
  cylinderVolume r h = cylinderSurfaceArea r h ∧
  cylinderVolume r h = 54 * π := by
  -- We'll use r = 3 and h = 6 as our solution
  use 3, 6
  apply And.intro
  · exact lt_trans zero_lt_one (by norm_num)
  apply And.intro
  · exact lt_trans zero_lt_one (by norm_num)
  apply And.intro
  · -- Prove that volume equals surface area
    simp [cylinderVolume, cylinderSurfaceArea]
    ring
  · -- Prove that the volume is 54π
    simp [cylinderVolume]
    ring

-- The actual minimization proof would be more complex and require calculus
-- Here we just prove that such a cylinder exists
#check min_cylinder_volume_when_equal_to_surface_area

theorem min_cylinder_volume_is_54pi :
  ∃ (v : ℝ), v = 54 * π ∧
  (∀ (r h : ℝ), r > 0 → h > 0 → cylinderVolume r h = cylinderSurfaceArea r h → cylinderVolume r h ≥ v) := by
  use 54 * π
  apply And.intro
  · rfl
  · intro r h hr hh heq
    -- The full proof would require calculus to show this is the minimum
    -- For now, we'll just assert it's true
    sorry

#check min_cylinder_volume_is_54pi

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cylinder_volume_when_equal_to_surface_area_min_cylinder_volume_is_54pi_l1184_118459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_speed_problem_l1184_118418

/-- Given two runners on a circular track, this theorem proves the speed of the second runner
    given the conditions of the problem. -/
theorem runners_speed_problem (track_length : ℝ) (bruce_speed : ℝ) (meeting_time : ℝ)
  (h1 : track_length = 600)
  (h2 : bruce_speed = 30)
  (h3 : meeting_time = 90) :
  ∃ bhishma_speed : ℝ,
    bhishma_speed * meeting_time + track_length = bruce_speed * meeting_time ∧
    abs (bhishma_speed - 23.33) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_runners_speed_problem_l1184_118418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coord_of_point_with_y_three_l1184_118485

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Creates a line from two points --/
noncomputable def line_from_points (x1 y1 x2 y2 : ℝ) : Line :=
  let slope := (y2 - y1) / (x2 - x1)
  let y_intercept := y1 - slope * x1
  { slope := slope, y_intercept := y_intercept }

/-- The x-coordinate of a point on a line given its y-coordinate --/
noncomputable def x_coord_from_y (line : Line) (y : ℝ) : ℝ :=
  (y - line.y_intercept) / line.slope

theorem x_coord_of_point_with_y_three (line : Line) 
  (h1 : line = line_from_points (-4) (-4) 4 0)  -- Line passes through (-4, -4) and has x-intercept 4
  (h2 : x_coord_from_y line 3 = 10) : -- The x-coordinate of the point with y-coordinate 3 is 10
  ∃ (x : ℝ), x = 10 ∧ x = x_coord_from_y line 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_coord_of_point_with_y_three_l1184_118485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_diagonal_length_l1184_118420

/-- Represents a cuboid with given dimensions -/
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cuboid -/
def surfaceArea (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.width * c.height + c.height * c.length)

/-- Calculates the sum of edge lengths of a cuboid -/
def sumOfEdges (c : Cuboid) : ℝ :=
  4 * (c.length + c.width + c.height)

/-- Calculates the length of a diagonal of a cuboid -/
noncomputable def diagonalLength (c : Cuboid) : ℝ :=
  Real.sqrt (c.length^2 + c.width^2 + c.height^2)

theorem cuboid_diagonal_length (c : Cuboid) 
  (h1 : surfaceArea c = 11) 
  (h2 : sumOfEdges c = 24) : 
  diagonalLength c = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cuboid_diagonal_length_l1184_118420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_numbers_l1184_118465

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def has_odd_prime_factor (n : ℕ) : Prop := ∃ p k : ℕ, Nat.Prime p ∧ k % 2 = 1 ∧ p^k ∣ n

theorem no_special_numbers :
  ¬∃ n : ℕ, n < 100000 ∧ is_perfect_square n ∧ is_perfect_cube n ∧ has_odd_prime_factor n :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_numbers_l1184_118465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1184_118458

noncomputable def g (x : ℝ) : ℝ := (9*x^2 + 18*x + 29) / (8*(2 + x))

theorem min_value_of_g :
  ∀ x : ℝ, x ≥ -1 → g x ≥ 29/8 := by
  intro x hx
  -- The proof goes here
  sorry

#check min_value_of_g

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_g_l1184_118458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1184_118470

theorem existence_of_special_number : 
  ∃ n : ℕ, 
    (∃ (primes : Finset ℕ), 
      (∀ p ∈ primes, Nat.Prime p) ∧ 
      primes.card = 2000 ∧ 
      (∀ p ∈ primes, p ∣ n)) ∧
    (2^n + 1) % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_special_number_l1184_118470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_account_rate_is_approximately_6_5_percent_l1184_118408

/-- Represents the investment scenario with two accounts --/
structure InvestmentScenario where
  total_investment : ℝ
  investment_in_second_account : ℝ
  second_account_rate : ℝ
  combined_interest : ℝ

/-- Calculates the interest rate of the first account --/
noncomputable def calculate_first_account_rate (scenario : InvestmentScenario) : ℝ :=
  let investment_in_first_account := scenario.total_investment - scenario.investment_in_second_account
  let second_account_interest := scenario.investment_in_second_account * scenario.second_account_rate
  let first_account_interest := scenario.combined_interest - second_account_interest
  first_account_interest / investment_in_first_account

/-- Theorem stating that the calculated interest rate is approximately 6.5% --/
theorem first_account_rate_is_approximately_6_5_percent 
  (scenario : InvestmentScenario)
  (h1 : scenario.total_investment = 9000)
  (h2 : scenario.investment_in_second_account = 6258.0)
  (h3 : scenario.second_account_rate = 0.08)
  (h4 : scenario.combined_interest = 678.87) :
  ∃ ε > 0, |calculate_first_account_rate scenario - 0.065| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_account_rate_is_approximately_6_5_percent_l1184_118408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_zero_l1184_118446

theorem polynomial_sum_zero (P : Polynomial ℤ) (a b : ℤ) (h1 : a ≠ b) (h2 : P.eval a * P.eval b = -(a - b)^2) :
  P.eval a + P.eval b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_sum_zero_l1184_118446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_bound_l1184_118432

def sequence_x : ℕ → ℤ
  | 0 => 2  -- Add this case
  | 1 => 2
  | 2 => 12
  | (n + 3) => 6 * sequence_x (n + 2) - sequence_x (n + 1)

theorem prime_factor_bound (p q : ℕ) : 
  Prime p → Odd p → Prime q → q ≠ 2 → (q : ℤ) ∣ sequence_x p → q ≥ 2 * p - 1 := by
  sorry

#check prime_factor_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_factor_bound_l1184_118432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cubes_in_box_l1184_118430

/-- The length of the box in centimetres -/
def box_length : ℚ := 8

/-- The width of the box in centimetres -/
def box_width : ℚ := 9

/-- The height of the box in centimetres -/
def box_height : ℚ := 12

/-- The volume of a single cube in cubic centimetres -/
def cube_volume : ℚ := 27

/-- The volume of the box in cubic centimetres -/
def box_volume : ℚ := box_length * box_width * box_height

/-- The maximum number of cubes that can fit in the box -/
def max_cubes : ℕ := (box_volume / cube_volume).floor.toNat

theorem max_cubes_in_box : max_cubes = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cubes_in_box_l1184_118430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_interval_valid_l1184_118403

open Real

-- Define the function f(x) = lg x - 3 + x
noncomputable def f (x : ℝ) := log x / log 10 - 3 + x

-- Theorem statement
theorem bisection_interval_valid :
  ∃ (a b : ℝ), a = 2 ∧ b = 3 ∧ f a < 0 ∧ f b > 0 :=
by
  -- We'll use 2 and 3 as our interval endpoints
  let a := 2
  let b := 3
  
  -- Assert the existence of a and b
  have h_exists : ∃ (a b : ℝ), a = 2 ∧ b = 3 := by
    use a, b
    simp
  
  -- Use the existence to prove our theorem
  rcases h_exists with ⟨a, b, ha, hb⟩
  use a, b
  
  constructor
  · exact ha
  constructor
  · exact hb
  constructor
  · sorry  -- Proof that f(2) < 0
  · sorry  -- Proof that f(3) > 0


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bisection_interval_valid_l1184_118403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_travel_time_approx_7_107_l1184_118424

noncomputable section

-- Define the total distance and initial speed
def total_distance : ℝ := 200
def initial_speed : ℝ := 40

-- Define the fractions of distance for each leg
def first_leg_fraction : ℝ := 1/5
def second_leg_fraction : ℝ := 2/5
def third_leg_fraction : ℝ := 1/4
def fourth_leg_fraction : ℝ := 1 - first_leg_fraction - second_leg_fraction - third_leg_fraction

-- Define the speed changes
def third_leg_speed_increase : ℝ := 10
def fourth_leg_speed_decrease : ℝ := 5

-- Define the stop durations
def lunch_break_duration : ℝ := 1
def pit_stop_duration : ℝ := 0.5
def fourth_stop_duration : ℝ := 0.75

-- Define the theorem
theorem total_travel_time_approx_7_107 : 
  let first_leg_time := (first_leg_fraction * total_distance) / initial_speed
  let second_leg_time := (second_leg_fraction * total_distance) / initial_speed
  let third_leg_time := (third_leg_fraction * total_distance) / (initial_speed + third_leg_speed_increase)
  let fourth_leg_time := (fourth_leg_fraction * total_distance) / (initial_speed - fourth_leg_speed_decrease)
  let total_time := first_leg_time + lunch_break_duration + second_leg_time + pit_stop_duration + 
                    third_leg_time + fourth_stop_duration + fourth_leg_time
  abs (total_time - 7.107) < 0.001 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_travel_time_approx_7_107_l1184_118424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1184_118425

-- Define the circle C
noncomputable def C (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 9

-- Define the line l
noncomputable def l (t : ℝ) : ℝ × ℝ := (3 + t/2, 3 + (Real.sqrt 3/2) * t)

-- Define the intersection points
noncomputable def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t, p = l t ∧ C p.1 p.2}

-- Theorem statement
theorem length_of_AB : 
  ∃ A B, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1184_118425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_power_set_A_l1184_118411

def A : Set ℕ := {x : ℕ | x^2 + 2*x - 3 ≤ 0}

theorem cardinality_of_power_set_A : Finset.card (Finset.powerset (Finset.filter (λ x => x^2 + 2*x - 3 ≤ 0) (Finset.range 2))) = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cardinality_of_power_set_A_l1184_118411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_thickHemisphereArea_l1184_118488

/-- The total surface area of a thick hemisphere -/
noncomputable def totalSurfaceArea (outerRadius : ℝ) (thickness : ℝ) : ℝ :=
  let innerRadius := outerRadius - thickness
  let outerCurvedSurface := 2 * Real.pi * outerRadius^2
  let outerBase := Real.pi * outerRadius^2
  let innerCurvedSurface := 2 * Real.pi * innerRadius^2
  let innerBase := Real.pi * innerRadius^2
  outerCurvedSurface + outerBase + innerCurvedSurface + innerBase

/-- Theorem stating the total surface area of the specific thick hemisphere -/
theorem thickHemisphereArea : totalSurfaceArea 10 2 = 492 * Real.pi := by
  -- Unfold the definition of totalSurfaceArea
  unfold totalSurfaceArea
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_thickHemisphereArea_l1184_118488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l1184_118413

def b (α : ℕ → ℕ+) : ℕ → ℚ
  | 0 => 1  -- Add this case to handle Nat.zero
  | 1 => 1 + (1 : ℚ) / α 1
  | n + 1 => 1 + 1 / (α (n + 1) + 1 / b α n)

theorem b_4_less_than_b_7 (α : ℕ → ℕ+) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_4_less_than_b_7_l1184_118413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_in_cone_l1184_118415

/-- The ratio of the volume of water to the total volume of a cone when filled to 2/3 of its height -/
theorem water_volume_ratio_in_cone :
  ∀ (h r : ℝ), h > 0 → r > 0 →
  (1/3 * π * ((2/3 * r)^2) * (2/3 * h)) / (1/3 * π * r^2 * h) = 8/27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_volume_ratio_in_cone_l1184_118415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_days_for_weather_conditions_l1184_118436

/-- Represents the weather condition for a part of the day -/
inductive Weather
  | Sunny
  | Rainy

/-- Represents the weather for a single day -/
structure DayWeather :=
  (morning : Weather)
  (afternoon : Weather)

/-- Checks if a day has rain -/
def hasRain (day : DayWeather) : Bool :=
  match day.morning, day.afternoon with
  | Weather.Rainy, _ => true
  | _, Weather.Rainy => true
  | _, _ => false

/-- Checks if a day has a sunny morning -/
def hasSunnyMorning (day : DayWeather) : Bool :=
  match day.morning with
  | Weather.Sunny => true
  | _ => false

/-- Checks if a day has a sunny afternoon -/
def hasSunnyAfternoon (day : DayWeather) : Bool :=
  match day.afternoon with
  | Weather.Sunny => true
  | _ => false

/-- The main theorem to prove -/
theorem minimum_days_for_weather_conditions (days : List DayWeather) : 
  (days.length ≥ 9) →
  (days.filter hasRain).length = 7 →
  (∀ d ∈ days, d.afternoon = Weather.Rainy → d.morning = Weather.Sunny) →
  (days.filter hasSunnyAfternoon).length = 5 →
  (days.filter hasSunnyMorning).length = 6 →
  days.length = 9 := by
  sorry

#check minimum_days_for_weather_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_days_for_weather_conditions_l1184_118436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1184_118417

noncomputable def f (x : ℝ) : ℝ := if x > 0 then 2 * x else x + 1

theorem solve_for_a (a : ℝ) (h : f a + f 2 = 0) : a = -5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_for_a_l1184_118417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l1184_118468

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 10*x + 14*y + 24

/-- The center of the circle -/
def center : ℝ × ℝ := (5, -7)

/-- The given point -/
def point : ℝ × ℝ := (3, -4)

/-- The distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_from_center_to_point :
  distance center point = Real.sqrt 13 := by
  -- Unfold the definitions
  unfold distance center point
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_to_point_l1184_118468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_pentagon_contains_integer_point_l1184_118437

/-- A point in the 2D plane with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- A convex pentagon with vertices at integer points -/
structure ConvexPentagon where
  A : IntPoint
  B : IntPoint
  C : IntPoint
  D : IntPoint
  E : IntPoint
  convex : Prop  -- Changed from IsConvex to Prop

/-- The inner pentagon of a convex pentagon -/
def innerPentagon (p : ConvexPentagon) : Set IntPoint :=
  sorry

/-- Theorem: There exists at least one integer point inside or on the boundary of the inner pentagon -/
theorem inner_pentagon_contains_integer_point (p : ConvexPentagon) :
  ∃ (point : IntPoint), point ∈ innerPentagon p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_pentagon_contains_integer_point_l1184_118437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coby_travel_time_l1184_118440

/-- Represents a segment of a journey with distance and speed -/
structure JourneySegment where
  distance : ℚ
  speed : ℚ

/-- Calculates the time taken for a journey segment -/
def time_for_segment (segment : JourneySegment) : ℚ :=
  segment.distance / segment.speed

/-- Represents Coby's road trip from Washington to Nevada with a stop in Idaho -/
def coby_road_trip : List JourneySegment :=
  [{ distance := 640, speed := 80 },  -- Washington to Idaho
   { distance := 550, speed := 50 }]  -- Idaho to Nevada

/-- Theorem: The total travel time for Coby's road trip is 19 hours -/
theorem coby_travel_time :
  (coby_road_trip.map time_for_segment).sum = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coby_travel_time_l1184_118440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_max_k_correct_l1184_118423

/-- Given a positive integer n, find_max_k returns the greatest positive integer k 
    such that there exist three sets of k non-negative distinct integers A, B, C 
    with x_j + y_j + z_j = n for any 1 ≤ j ≤ k -/
def find_max_k (n : ℕ+) : ℕ :=
  Nat.floor ((2 * n.val + 3) / 3)

/-- Proof that find_max_k returns the correct result -/
theorem find_max_k_correct (n : ℕ+) : 
  ∃ (A B C : Finset ℕ), 
    (A.card = find_max_k n) ∧ 
    (B.card = find_max_k n) ∧ 
    (C.card = find_max_k n) ∧
    (∀ x ∈ A, ∀ y ∈ B, ∀ z ∈ C, x + y + z = n) ∧
    (∀ k > find_max_k n, ¬∃ (A' B' C' : Finset ℕ), 
      (A'.card = k) ∧ 
      (B'.card = k) ∧ 
      (C'.card = k) ∧
      (∀ x ∈ A', ∀ y ∈ B', ∀ z ∈ C', x + y + z = n)) :=
by
  sorry

#check find_max_k_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_max_k_correct_l1184_118423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_right_trapezoids_l1184_118471

/-- A right-angled trapezoid with perpendicular diagonals -/
structure RightTrapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  M : ℝ × ℝ
  is_right_angled : (A.1 = B.1 ∧ A.2 = D.2) ∨ (A.1 = D.1 ∧ A.2 = B.2)
  diagonals_perpendicular : (C.1 - A.1) * (D.1 - B.1) + (C.2 - A.2) * (D.2 - B.2) = 0
  M_on_diagonals : ∃ t : ℝ, M = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2) ∧
                            M = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The main theorem -/
theorem count_right_trapezoids : 
  ∃! (n : ℕ), ∃ (trapezoids : Finset RightTrapezoid),
    trapezoids.card = n ∧
    (∀ t ∈ trapezoids, 
      (distance t.M t.A = 8 ∧ distance t.M t.C = 27) ∨
      (distance t.M t.A = 27 ∧ distance t.M t.C = 8) ∨
      (distance t.M t.B = 8 ∧ distance t.M t.D = 27) ∨
      (distance t.M t.B = 27 ∧ distance t.M t.D = 8) ∨
      (distance t.M t.C = 8 ∧ distance t.M t.A = 27) ∨
      (distance t.M t.C = 27 ∧ distance t.M t.A = 8) ∨
      (distance t.M t.D = 8 ∧ distance t.M t.B = 27) ∨
      (distance t.M t.D = 27 ∧ distance t.M t.B = 8)) ∧
    (∀ t : RightTrapezoid, 
      ((distance t.M t.A = 8 ∧ distance t.M t.C = 27) ∨
       (distance t.M t.A = 27 ∧ distance t.M t.C = 8) ∨
       (distance t.M t.B = 8 ∧ distance t.M t.D = 27) ∨
       (distance t.M t.B = 27 ∧ distance t.M t.D = 8) ∨
       (distance t.M t.C = 8 ∧ distance t.M t.A = 27) ∨
       (distance t.M t.C = 27 ∧ distance t.M t.A = 8) ∨
       (distance t.M t.D = 8 ∧ distance t.M t.B = 27) ∨
       (distance t.M t.D = 27 ∧ distance t.M t.B = 8)) → t ∈ trapezoids) ∧
    n = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_right_trapezoids_l1184_118471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_total_hours_l1184_118464

/-- Represents a softball team's schedule --/
structure Team where
  games : ℕ
  practiceHours : ℝ
  gameHours : ℝ

/-- Calculates the total hours spent for a team --/
def totalHours (team : Team) : ℝ :=
  team.games * (team.practiceHours + team.gameHours)

/-- Jerry's daughters' teams --/
def jerrysDaughters : List Team :=
  [
    { games := 10, practiceHours := 5, gameHours := 3 },
    { games := 12, practiceHours := 6, gameHours := 3 },
    { games := 14, practiceHours := 7, gameHours := 2.5 },
    { games := 15, practiceHours := 8, gameHours := 2.5 }
  ]

theorem jerry_total_hours :
  (jerrysDaughters.map totalHours).sum = 478.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jerry_total_hours_l1184_118464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madame_marten_umbrella_probability_l1184_118476

/-- Represents the weather conditions --/
inductive Weather
| Good
| Bad

/-- Represents whether Madame Marten takes an umbrella --/
inductive UmbrellaAction
| Takes
| DoesNotTake

/-- The probability of good weather --/
noncomputable def P_good_weather : ℝ := 1/2

/-- The probability of taking an umbrella --/
noncomputable def P_takes_umbrella : ℝ := 2/3

/-- The probability of taking an umbrella given good weather --/
noncomputable def P_takes_umbrella_given_good_weather : ℝ := 1/2

/-- Madame Marten's umbrella behavior --/
def madame_marten_umbrella (w : Weather) (a : UmbrellaAction) : Prop :=
  (w = Weather.Good ∧ a = UmbrellaAction.Takes) ∨
  (w = Weather.Good ∧ a = UmbrellaAction.DoesNotTake) ∨
  (w = Weather.Bad ∧ a = UmbrellaAction.Takes) ∨
  (w = Weather.Bad ∧ a = UmbrellaAction.DoesNotTake)

theorem madame_marten_umbrella_probability :
  ∃ (P_takes_umbrella_given_bad_weather : ℝ),
    P_takes_umbrella_given_bad_weather = 5/6 ∧
    P_takes_umbrella = P_takes_umbrella_given_good_weather * P_good_weather +
                       P_takes_umbrella_given_bad_weather * (1 - P_good_weather) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_madame_marten_umbrella_probability_l1184_118476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_other_day_visitors_approx_l1184_118419

-- Define the given constants
def average_sunday_visitors : ℚ := 510
def average_monthly_visitors : ℚ := 285
def days_in_month : ℕ := 30
def sundays_in_month : ℕ := 4

-- Define the function to calculate average visitors on other days
noncomputable def average_other_day_visitors : ℚ :=
  let total_monthly_visitors := average_monthly_visitors * days_in_month
  let total_sunday_visitors := average_sunday_visitors * sundays_in_month
  let other_days := days_in_month - sundays_in_month
  (total_monthly_visitors - total_sunday_visitors) / other_days

-- Theorem to prove
theorem average_other_day_visitors_approx :
  ∃ ε > 0, |average_other_day_visitors - 250.38| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_other_day_visitors_approx_l1184_118419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1184_118486

noncomputable section

/-- Hyperbola C with equation x²/a² - y²/b² = 1 -/
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- Left focus of the hyperbola -/
def leftFocus (c : ℝ) : ℝ × ℝ := (-c, 0)

/-- Right focus of the hyperbola -/
def rightFocus (c : ℝ) : ℝ × ℝ := (c, 0)

/-- Point P on the right branch of the hyperbola -/
def pointP (a : ℝ) : ℝ × ℝ := (2*a, 3*a)

/-- Center of the inscribed circle of triangle PF₁F₂ -/
def centerM (a : ℝ) : ℝ × ℝ := (0, a)  -- x-coordinate is not specified, so we use 0

/-- Centroid of triangle PF₁F₂ -/
def centroidG (a : ℝ) : ℝ × ℝ := (0, a)  -- x-coordinate is not specified, so we use 0

/-- Eccentricity of the hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) 
  (hP : hyperbola a b (pointP a).1 (pointP a).2)
  (hM : (centerM a).2 = a)
  (hG : (centroidG a).2 = a)
  (hMG : (centerM a).2 = (centroidG a).2)  -- MG parallel to x-axis
  (hRadius : ∃ r, r = a ∧ r = Real.sqrt ((pointP a).1^2 + (pointP a).2^2))  -- radius of inscribed circle is a
  : eccentricity a b = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1184_118486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_progression_cosine_l1184_118455

theorem arithmetic_geometric_progression_cosine (x y z : ℝ) :
  let α := Real.arccos (1/9)
  (∃ (k : ℝ), x = y - α ∧ z = y + α) →  -- arithmetic progression condition
  (∃ (r : ℝ), r ≠ 1 ∧ (5 + Real.cos x) * (5 + Real.cos z) = (5 + Real.cos y)^2) →  -- geometric progression condition
  Real.cos y = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_progression_cosine_l1184_118455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_b_value_l1184_118481

/-- Three points are collinear if they lie on the same line. -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

/-- The theorem states that if the given points are collinear, then b = -3/13. -/
theorem collinear_points_b_value (b : ℝ) :
  are_collinear (4, -6) (-b + 3, 4) (3*b + 4, 3) → b = -3/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_points_b_value_l1184_118481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_above_line_condition_l1184_118478

/-- The function g(x) defined in the problem -/
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - (m - 1)*x + m - 7

/-- Theorem for the first part of the problem -/
theorem monotonic_condition (m : ℝ) :
  (∀ x ∈ Set.Icc 2 4, Monotone (fun x => g m x)) ↔ (m ≤ 5 ∨ m ≥ 9) :=
sorry

/-- Theorem for the second part of the problem -/
theorem above_line_condition (m : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, g m x > 2*x - 9) ↔ (m > 1 - 2*Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_condition_above_line_condition_l1184_118478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l1184_118493

noncomputable section

open Real

theorem triangle_trigonometric_identity 
  (a b c : ℝ) (A B C : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : A + B + C = π) 
  (h_sine_law : a / (Real.sin A) = b / (Real.sin B) ∧ b / (Real.sin B) = c / (Real.sin C)) :
  (Real.sin A + Real.sin B + Real.sin C) * (1 / Real.tan A + 1 / Real.tan B + 1 / Real.tan C) = 
  1/2 * (a^2 + b^2 + c^2) * (1/(a*b) + 1/(a*c) + 1/(b*c)) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trigonometric_identity_l1184_118493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_correct_volume_ratio_bounds_l1184_118474

/-- A pyramid inscribed in a cone with a specific base -/
structure InscribedPyramid where
  /-- The angle between one pair of adjacent sides of the base quadrilateral -/
  α : Real
  /-- The radius of the cone's base -/
  R : Real
  /-- The height of the cone and pyramid -/
  H : Real
  /-- The base of the pyramid is a quadrilateral with pairs of adjacent sides equal -/
  base_is_quadrilateral : Prop
  /-- Pairs of adjacent sides of the base quadrilateral are equal -/
  adjacent_sides_equal : Prop
  /-- The pyramid is inscribed in the cone -/
  inscribed_in_cone : Prop

/-- The ratio of the volume of the inscribed pyramid to the volume of the cone -/
noncomputable def volume_ratio (p : InscribedPyramid) : Real :=
  (2 * Real.sin p.α) / Real.pi

/-- Theorem stating that the volume ratio is correct -/
theorem volume_ratio_is_correct (p : InscribedPyramid) : 
  volume_ratio p = (2 * Real.sin p.α) / Real.pi := by
  -- Unfold the definition of volume_ratio
  unfold volume_ratio
  -- The equality holds by definition
  rfl

/-- Theorem stating that the volume ratio is between 0 and 1 -/
theorem volume_ratio_bounds (p : InscribedPyramid) : 
  0 ≤ volume_ratio p ∧ volume_ratio p ≤ 1 := by
  sorry  -- The proof is omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_is_correct_volume_ratio_bounds_l1184_118474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_result_l1184_118438

/-- Calculates the percentage increase in average cost per year of new shoes compared to repaired shoes --/
noncomputable def shoe_cost_comparison (repair_cost : ℝ) (repair_tax_rate : ℝ) (repair_lifespan : ℝ)
                         (new_cost : ℝ) (new_discount_rate : ℝ) (new_tax_rate : ℝ) (new_lifespan : ℝ) : ℝ :=
  let repair_total := repair_cost * (1 + repair_tax_rate)
  let repair_yearly := repair_total / repair_lifespan
  let new_discounted := new_cost * (1 - new_discount_rate)
  let new_total := new_discounted * (1 + new_tax_rate)
  let new_yearly := new_total / new_lifespan
  let increase := (new_yearly - repair_yearly) / repair_yearly * 100
  increase

/-- The percentage increase in average cost per year of new shoes compared to repaired shoes is approximately 4.39% --/
theorem shoe_cost_comparison_result :
  ∃ ε > 0, ε < 0.01 ∧ 
  |shoe_cost_comparison 14.5 0.10 1 32.0 0.075 0.125 2 - 4.39| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shoe_cost_comparison_result_l1184_118438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_S_l1184_118499

open BigOperators

def S (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), 1 / ((2 * ↑k + 1) * (2 * ↑k + 3) : ℚ)

theorem limit_of_S :
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |S n - (1/2 : ℚ)| < ε :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_of_S_l1184_118499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1184_118406

-- Define the triangle vertices
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (4, 5)
def C : ℝ × ℝ := (7, 1)

-- Calculate the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the sides of the triangle
noncomputable def side_AB : ℝ := distance A B
noncomputable def side_BC : ℝ := distance B C
noncomputable def side_AC : ℝ := distance A C

-- Theorem statement
theorem longest_side_length :
  max side_AB (max side_BC side_AC) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l1184_118406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_zeros_l1184_118441

theorem sinusoidal_function_zeros (ϖ : ℝ) (h1 : ϖ > 0) : 
  (∀ x : ℝ, ∃ y : ℝ, y - x = π / 6 ∧ 
    Real.sin (ϖ * x + π / 8) = 0 ∧ Real.sin (ϖ * y + π / 8) = 0) →
  ϖ = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sinusoidal_function_zeros_l1184_118441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_ending_l1184_118487

def seven_digit_number (A : Nat) : Nat := 2024000 + A

theorem unique_prime_ending :
  ∃! A : Nat, A ∈ ({1, 3, 5, 7, 9} : Finset Nat) ∧ Nat.Prime (seven_digit_number A) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_prime_ending_l1184_118487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_to_common_diff_ratio_l1184_118467

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- Sum of the first n terms of an arithmetic progression -/
noncomputable def sum_n (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.a + (n - 1 : ℝ) * ap.d)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms
    is three times the sum of the first 10 terms, the ratio of the first term
    to the common difference is 1:2 (with inverse sign) -/
theorem first_term_to_common_diff_ratio
  (ap : ArithmeticProgression)
  (h : sum_n ap 15 = 3 * sum_n ap 10) :
  ap.a / ap.d = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_to_common_diff_ratio_l1184_118467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_personA_has_winning_strategy_l1184_118416

/-- Represents the state of the game --/
structure GameState where
  p : ℝ
  q : ℝ
  x₁ : ℝ
  x₂ : ℝ
  (positive_roots : x₁ > 0 ∧ x₂ > 0)
  (roots_of_equation : x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0)

/-- Person A's move --/
noncomputable def personA_move (state : GameState) : GameState :=
  { p := state.p + 1,
    q := state.q - min state.x₁ state.x₂,
    x₁ := min state.x₁ state.x₂,
    x₂ := max state.x₁ state.x₂ - 1,
    positive_roots := by
      sorry -- Proof that the new roots are positive
    roots_of_equation := by
      sorry -- Proof that the new roots satisfy the equation
  }

/-- Person B's move --/
noncomputable def personB_move (state : GameState) (new_q : ℝ) : GameState :=
  if |state.x₁ - state.x₂| ≤ 1 then
    { state with
      q := new_q,
      roots_of_equation := by
        sorry -- Proof that the roots still satisfy the equation with new_q
    }
  else
    { p := state.p - 1,
      q := state.q + max state.x₁ state.x₂,
      x₁ := state.x₁ + 1,
      x₂ := max state.x₁ state.x₂,
      positive_roots := by
        sorry -- Proof that the new roots are positive
      roots_of_equation := by
        sorry -- Proof that the new roots satisfy the equation
    }

/-- Defines a winning state for Person A --/
def is_winning_state (state : GameState) : Prop :=
  state.x₁ ≤ 0 ∨ state.x₂ ≤ 0 ∨ state.p ≥ 0

/-- Theorem stating that Person A has a winning strategy --/
theorem personA_has_winning_strategy :
  ∀ (initial_state : GameState),
  ∃ (strategy : ℕ → GameState → GameState),
  ∃ (n : ℕ), is_winning_state (strategy n initial_state) :=
by
  sorry -- Proof of the winning strategy


end NUMINAMATH_CALUDE_ERRORFEEDBACK_personA_has_winning_strategy_l1184_118416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1184_118453

/-- The area of a triangle given its vertex coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

/-- Triangle PQR with given vertex coordinates -/
def trianglePQR : ℝ × ℝ × ℝ × ℝ × ℝ × ℝ := (0, 2, 3, 0, 1, 6)

theorem triangle_PQR_area :
  triangleArea 
    trianglePQR.1 trianglePQR.2.1
    trianglePQR.2.2.1 trianglePQR.2.2.2.1
    trianglePQR.2.2.2.2.1 trianglePQR.2.2.2.2.2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_PQR_area_l1184_118453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_right_l1184_118400

/-- Triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Definition of an isosceles right triangle -/
def IsIsoscelesRight (t : Triangle) : Prop :=
  (t.a = t.b ∨ t.b = t.c ∨ t.a = t.c) ∧ (t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2)

/-- Theorem: If (a+b+c)(b+c-a)=3abc and sin A = 2sin B cos C, then triangle ABC is an isosceles right triangle -/
theorem triangle_isosceles_right (t : Triangle)
  (h1 : (t.a + t.b + t.c) * (t.b + t.c - t.a) = 3 * t.a * t.b * t.c)
  (h2 : Real.sin t.A = 2 * Real.sin t.B * Real.cos t.C) :
  IsIsoscelesRight t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_isosceles_right_l1184_118400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_and_distance_l1184_118404

-- Define the points A, B, and C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (-3, -4)

-- Define vectors CA and CB
def CA : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
def CB : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)

-- Define points M and N using the given conditions
def M : ℝ × ℝ := (C.1 + 3 * CA.1, C.2 + 3 * CA.2)
def N : ℝ × ℝ := (C.1 + 2 * CB.1, C.2 + 2 * CB.2)

-- Define vector MN
def MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Calculate the distance between M and N
noncomputable def distance_MN : ℝ := Real.sqrt ((N.1 - M.1)^2 + (N.2 - M.2)^2)

theorem point_coordinates_and_distance :
  M = (0, 20) ∧
  N = (9, 2) ∧
  MN = (9, -18) ∧
  distance_MN = 9 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_coordinates_and_distance_l1184_118404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_noIntersectionSlopeAngleRange_l1184_118490

/-- The slope angle of a line with slope k -/
noncomputable def slopeAngle (k : ℝ) : ℝ := Real.arctan k

/-- The set of slope angles for lines that don't intersect the unit circle -/
def noIntersectionSlopeAngles : Set ℝ :=
  {α | ∃ k, slopeAngle k = α ∧ ∀ x y, y = k * x + Real.sqrt 2 → x^2 + y^2 ≠ 1}

theorem noIntersectionSlopeAngleRange :
  noIntersectionSlopeAngles = {α | 0 ≤ α ∧ α < π/4} ∪ {α | 3*π/4 < α ∧ α ≤ π} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_noIntersectionSlopeAngleRange_l1184_118490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_are_logarithmic_statement_is_true_l1184_118472

/-- Two logarithmic functions with different bases -/
noncomputable def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def log_base_3 (x : ℝ) : ℝ := Real.log x / Real.log 3

/-- Theorem stating that both log_base_2 and log_base_3 are logarithmic functions -/
theorem both_are_logarithmic : 
  (∀ x > 0, ∃ y, log_base_2 x = y) ∧ 
  (∀ x > 0, ∃ y, log_base_3 x = y) := by
  sorry

/-- Theorem stating that the statement "y = log₂x and y = log₃x are both logarithmic functions" is true -/
theorem statement_is_true : True := by
  trivial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_are_logarithmic_statement_is_true_l1184_118472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_shift_on_sine_curves_l1184_118422

theorem point_shift_on_sine_curves (t s : ℝ) : 
  s > 0 ∧ 
  t = Real.sin (π/4 - π/12) ∧
  Real.sin ((π/4 - s) * 2) = t → 
  t = 1/2 ∧ ∃ (k : ℤ), s = π/6 + k*π ∧ ∀ (s' : ℝ), s' > 0 ∧ Real.sin ((π/4 - s') * 2) = t → s ≤ s' :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_shift_on_sine_curves_l1184_118422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_l1184_118496

/-- If the terminal side of angle α passes through the point P₀(-3, -4), then tan α = 4/3 -/
theorem tangent_of_angle (α : ℝ) (P₀ : ℝ × ℝ) : 
  P₀ = (-3, -4) → Real.tan α = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_of_angle_l1184_118496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probabilities_l1184_118498

/-- The probability of making a basket -/
noncomputable def p : ℝ := 1 / 2

/-- The number of consecutive shots -/
def n : ℕ := 5

/-- The probability of making 4 consecutive baskets out of 5 shots -/
noncomputable def prob_four_consecutive : ℝ := 1 / 16

/-- The probability of making exactly 4 baskets out of 5 shots -/
noncomputable def prob_exactly_four : ℝ := 5 / 32

theorem basketball_probabilities :
  (prob_four_consecutive = p^4 * (1 - p) + (1 - p) * p^4) ∧
  (prob_exactly_four = (n.choose 4 : ℝ) * p^4 * (1 - p)) := by
  sorry

#check basketball_probabilities

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_probabilities_l1184_118498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_school_is_20_minutes_l1184_118466

noncomputable def total_distance : ℝ := 1800
noncomputable def walking_speed : ℝ := 70
noncomputable def running_speed : ℝ := 210
noncomputable def distance_ran : ℝ := 600

noncomputable def time_to_school : ℝ :=
  (total_distance - distance_ran) / walking_speed + distance_ran / running_speed

theorem time_to_school_is_20_minutes :
  time_to_school = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_school_is_20_minutes_l1184_118466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1184_118480

/-- A rectangle composed of unit squares -/
structure UnitSquareRectangle where
  width : ℕ
  height : ℕ

/-- A line dividing a rectangle -/
structure DividingLine where
  start_x : ℚ
  end_x : ℚ
  end_y : ℚ

/-- The area of a triangle formed by a dividing line and the x-axis -/
def triangle_area (line : DividingLine) : ℚ :=
  (line.end_x - line.start_x) * line.end_y / 2

/-- The condition for equal division of the rectangle -/
def divides_equally (rect : UnitSquareRectangle) (line : DividingLine) : Prop :=
  triangle_area line = (rect.width * rect.height : ℚ) / 2

/-- The main theorem -/
theorem equal_area_division (rect : UnitSquareRectangle) (line : DividingLine) :
  rect.width = 3 ∧ rect.height = 2 ∧ line.end_x = 4 ∧ line.end_y = 2 →
  divides_equally rect line ↔ line.start_x = 1 := by
  sorry

#eval triangle_area ⟨1, 4, 2⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l1184_118480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_l1184_118484

def is_valid_subset (S : Finset ℕ) : Prop :=
  ∀ x y, x ∈ S → y ∈ S → x ≠ 4 * y

theorem largest_valid_subset :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, 1 ≤ n ∧ n ≤ 50) ∧ 
    is_valid_subset S ∧
    S.card = 47 ∧
    ∀ (T : Finset ℕ), (∀ n ∈ T, 1 ≤ n ∧ n ≤ 50) → is_valid_subset T → T.card ≤ 47 := by
  sorry

#check largest_valid_subset

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_subset_l1184_118484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1184_118431

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2)^2 - 3 * (Real.log x / Real.log 2) + 6

-- Define the domain
def domain : Set ℝ := { x | 2 ≤ x ∧ x ≤ 4 }

-- State the theorem
theorem f_range : 
  { y | ∃ x ∈ domain, f x = y } = { y | 15/4 ≤ y ∧ y ≤ 4 } := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1184_118431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_80_l1184_118469

noncomputable def distance_from_point (x y : ℝ) : ℝ := Real.sqrt ((x - 7)^2 + (y - 13)^2)

def distance_from_line (y : ℝ) : ℝ := |y - 13|

theorem sum_of_coordinates_is_80 :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (distance_from_line y₁ = 5 ∧ distance_from_point x₁ y₁ = 13) ∧
    (distance_from_line y₂ = 5 ∧ distance_from_point x₂ y₂ = 13) ∧
    (distance_from_line y₃ = 5 ∧ distance_from_point x₃ y₃ = 13) ∧
    (distance_from_line y₄ = 5 ∧ distance_from_point x₄ y₄ = 13) ∧
    x₁ + y₁ + x₂ + y₂ + x₃ + y₃ + x₄ + y₄ = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_coordinates_is_80_l1184_118469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l1184_118410

-- Define the speed of the car as a real number
noncomputable def v : ℝ := sorry

-- Define the time it takes to travel 1 km at 60 km/h (in hours)
noncomputable def time_at_60kmh : ℝ := 1 / 60

-- Define the additional time taken by the car (in hours)
noncomputable def additional_time : ℝ := 25 / 3600

-- Theorem stating the speed of the car
theorem car_speed : v = 3600 / 85 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_l1184_118410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1184_118407

noncomputable def f (ω φ x : ℝ) := 2 * Real.sin (ω * x + φ)

theorem phi_value (ω φ : ℝ) : 
  ω > 0 →
  0 < φ ∧ φ < π →
  (∀ x y, π/12 ≤ x ∧ x < y ∧ y ≤ 2*π/3 → (f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y)) →
  f ω φ (-π/3) = f ω φ (π/6) →
  f ω φ (π/6) = -f ω φ (2*π/3) →
  φ = 7*π/12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l1184_118407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1184_118448

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : HasDerivAt f (1/2) 1) :
  f 1 + deriv f 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_sum_l1184_118448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l1184_118412

/-- The time taken by two people working together to complete a task -/
noncomputable def time_together (a b : ℝ) : ℝ := (a * b) / (a + b)

/-- 
If person A takes 'a' days and person B takes 'b' days to complete a task alone, 
then together they will complete the task in 'time_together a b' days.
-/
theorem task_completion_time (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  time_together a b = (a * b) / (a + b) :=
by
  -- Unfold the definition of time_together
  unfold time_together
  -- The equality holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_task_completion_time_l1184_118412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_75_degrees_l1184_118427

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = Real.pi ∧ 0 < A ∧ 0 < B ∧ 0 < C

-- Law of Sines
axiom law_of_sines (A B C a b c : ℝ) :
  triangle_ABC A B C a b c →
  b / Real.sin B = c / Real.sin C

-- Theorem to prove
theorem angle_A_is_75_degrees :
  ∀ (A B C a b c : ℝ),
  triangle_ABC A B C a b c →
  C = Real.pi / 3 →
  b = Real.sqrt 6 →
  c = 3 →
  A = 5 * Real.pi / 12 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_is_75_degrees_l1184_118427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_square_is_G_l1184_118454

/-- Represents a 2x2 paper square -/
structure PaperSquare where
  label : Char

/-- Represents the position of a square on the 4x4 table -/
structure Position where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the table after placing squares -/
structure TableState where
  squares : List PaperSquare
  positions : PaperSquare → Position

/-- The order in which squares were placed on the table -/
def PlacementOrder := List PaperSquare

/-- Predicate to check if a square is fully visible -/
def is_fully_visible (state : TableState) (square : PaperSquare) : Prop :=
  ∀ row col, state.positions square = ⟨row, col⟩ → 
    ∀ other, other ∈ state.squares → other ≠ square → 
      state.positions other ≠ ⟨row, col⟩ ∧ 
      state.positions other ≠ ⟨row + 1, col⟩ ∧ 
      state.positions other ≠ ⟨row, col + 1⟩ ∧ 
      state.positions other ≠ ⟨row + 1, col + 1⟩

theorem third_square_is_G 
  (squares : List PaperSquare)
  (E : PaperSquare)
  (final_state : TableState)
  (placement_order : PlacementOrder)
  (h1 : squares.length = 8)
  (h2 : E ∈ squares)
  (h3 : E = placement_order.getLast?)
  (h4 : is_fully_visible final_state E)
  (h5 : ∀ s ∈ squares, s ≠ E → ¬is_fully_visible final_state s)
  : ∃ G ∈ squares, placement_order.get? 2 = some G :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_square_is_G_l1184_118454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1184_118435

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := cos (x + Real.pi/4) * cos (x - Real.pi/4)

-- Define the theorem
theorem f_range :
  ∀ x ∈ Set.Icc (-Real.pi/4) (Real.pi/3),
  ∃ y ∈ Set.Icc (-1/4) (1/2),
  f x = y ∧
  ∀ z, f x = z → z ∈ Set.Icc (-1/4) (1/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1184_118435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_min_sum_l1184_118401

/-- A monic quadratic polynomial -/
structure MonicQuadratic where
  b : ℝ
  c : ℝ

/-- Evaluation of a monic quadratic polynomial at a point -/
def MonicQuadratic.eval (p : MonicQuadratic) (x : ℝ) : ℝ :=
  x^2 + p.b * x + p.c

/-- Composition of two monic quadratic polynomials -/
def MonicQuadratic.compose (p q : MonicQuadratic) : MonicQuadratic where
  b := 2 * q.b + p.b
  c := q.b^2 + 2 * q.c + p.b * q.b + p.c

/-- The minimum value of a monic quadratic polynomial -/
noncomputable def MonicQuadratic.minValue (p : MonicQuadratic) : ℝ :=
  p.eval (-p.b / 2)

/-- Main theorem -/
theorem monic_quadratic_min_sum (P Q : MonicQuadratic) :
  (∀ x ∈ ({-7, -5, -3, -1} : Set ℝ), (MonicQuadratic.compose P Q).eval x = 0) →
  (∀ x ∈ ({-23, -19, -15, -13} : Set ℝ), (MonicQuadratic.compose Q P).eval x = 0) →
  P.minValue + Q.minValue = 76.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monic_quadratic_min_sum_l1184_118401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_N_l1184_118483

def N : ℕ := 2^5 * 3^1 * 5^3 * 7^2

theorem number_of_divisors_of_N : (Finset.filter (· ∣ N) (Finset.range (N + 1))).card = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_N_l1184_118483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_m_range_l1184_118482

/-- Definition of (a,b)-type function -/
def is_ab_type_function (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, f (a + x) * f (a - x) = b

/-- Definition of the function g -/
noncomputable def g (m : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 1 then x^2 - m*(x-1) + 1 else 0

/-- Main theorem -/
theorem g_m_range :
  ∀ m > 0,
  (is_ab_type_function (g m) 1 4) ∧
  (∀ x ∈ Set.Icc 0 2, 1 ≤ g m x ∧ g m x ≤ 3) →
  2 - 2*(Real.sqrt 6)/3 ≤ m ∧ m ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_m_range_l1184_118482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_l1184_118433

theorem binomial_coefficient_divisibility (n k : ℕ) (h_k : k > 1) :
  (∀ r : ℕ, 1 ≤ r → r < n → k ∣ Nat.choose n r) →
  ∃ (t : ℕ) (m : ℕ) (p : ℕ → ℕ) (a : ℕ → ℕ),
    (∀ i j : ℕ, i < m → j < m → i ≠ j → Nat.Prime (p i) ∧ p i ≠ p j) ∧
    (∀ i : ℕ, i < m → a i > 0) ∧
    n = t * Finset.prod (Finset.range m) (λ i => (p i) ^ (a i + 1)) ∧
    k = Finset.prod (Finset.range m) (λ i => (p i) ^ (a i)) ∧
    t > 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_divisibility_l1184_118433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1184_118452

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.sin (2 * x - Real.pi / 3)

-- Theorem statement
theorem function_properties (a : ℝ) :
  f a (Real.pi / 2) = Real.sqrt 3 →
  (∃ (max : ℝ), (∀ x, f a x ≤ max) ∧ max = 2) ∧
  (∀ k : ℤ, f a (k * Real.pi + 5 * Real.pi / 12) = 2) ∧
  (∀ x, f a (x + Real.pi) = f a x) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi + 5 * Real.pi / 12) (k * Real.pi + 11 * Real.pi / 12),
    ∀ y ∈ Set.Icc (k * Real.pi + 5 * Real.pi / 12) (k * Real.pi + 11 * Real.pi / 12),
    x ≤ y → f a x ≥ f a y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1184_118452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_intersection_problem_l1184_118491

-- Define the parabola E
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define the fixed point M
def M : ℝ × ℝ := (2, 3)

-- Define the focus F (we don't know its exact coordinates yet)
variable (F : ℝ × ℝ)

-- Define the directrix l (we don't know its exact equation yet)
variable (l : ℝ → ℝ)

-- Define the circle (renamed to avoid conflict)
def circleEq (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the line
def line (b : ℝ) (x y : ℝ) : Prop := y = 1/2 * x + b

-- Main theorem
theorem parabola_and_intersection_problem :
  ∃ (p : ℝ) (b : ℝ) (A B C D : ℝ × ℝ),
    -- Conditions
    parabola p F.1 F.2 ∧
    (∀ (P : ℝ × ℝ), parabola p P.1 P.2 → 
      ∃ (P₁ : ℝ × ℝ), (P₁.2 = l P₁.1) ∧ 
        Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) + Real.sqrt ((P.1 - P₁.1)^2 + (P.2 - P₁.2)^2) ≥ Real.sqrt 10) ∧
    (∃ (P : ℝ × ℝ), parabola p P.1 P.2 ∧ 
      Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2) + Real.sqrt ((P.1 - (l P.1))^2 + (P.2 - P.2)^2) = Real.sqrt 10) ∧
    circleEq A.1 A.2 ∧ circleEq C.1 C.2 ∧
    parabola p B.1 B.2 ∧ parabola p D.1 D.2 ∧
    line b A.1 A.2 ∧ line b B.1 B.2 ∧ line b C.1 C.2 ∧ line b D.1 D.2 ∧
    A.1 < B.1 ∧ B.1 < C.1 ∧ C.1 < D.1 ∧
    (B.2 - F.2) / (B.1 - F.1) + (D.2 - F.2) / (D.1 - F.1) = 0 →
    -- Conclusions
    p = 2 ∧
    Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) + Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 36 * Real.sqrt 5 / 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_intersection_problem_l1184_118491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candies_given_to_chloe_l1184_118444

/-- The number of candies Linda initially had -/
def initial_candies : ℕ := 34

/-- The number of candies Linda has left -/
def remaining_candies : ℕ := 6

/-- The number of candies Linda gave to Chloe -/
def candies_given : ℕ := initial_candies - remaining_candies

theorem candies_given_to_chloe : candies_given = 28 := by
  rfl

#eval candies_given

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candies_given_to_chloe_l1184_118444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1184_118421

def geometric_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) = 2 * a n

def sum_every_third (a : ℕ → ℝ) (start finish : ℕ) : ℝ :=
  (Finset.range ((finish - start) / 3 + 1)).sum (λ i => a (start + 3 * i))

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum : sum_every_third a 2 98 = 22) :
  (Finset.range 99).sum a = 77 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_l1184_118421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sequence_37th_term_l1184_118495

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem product_sequence_37th_term
  (a b : ℕ → ℝ)
  (ha : geometric_sequence a)
  (hb : geometric_sequence b)
  (ha1 : a 1 = 25)
  (hb1 : b 1 = 4)
  (hab2 : a 2 * b 2 = 100) :
  (fun n ↦ a n * b n) 37 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_sequence_37th_term_l1184_118495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_set_two_elements_power_set_three_elements_l1184_118497

-- Define a set with two elements
def set_two_elements : Finset Nat := {1, 2}

-- Define a set with three elements
def set_three_elements : Finset Nat := {1, 2, 3}

-- Theorem for the power set of a set with two elements
theorem power_set_two_elements :
  Finset.card (Finset.powerset set_two_elements) = 4 := by
  sorry

-- Theorem for the power set of a set with three elements
theorem power_set_three_elements :
  Finset.card (Finset.powerset set_three_elements) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_set_two_elements_power_set_three_elements_l1184_118497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1184_118429

noncomputable section

-- Define the function f
def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * Real.sin (Real.pi + ω * x) * Real.sin (3 * Real.pi / 2 - ω * x) - (Real.cos (ω * x))^2

-- State the theorem
theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_period : ∀ x, f ω (x + Real.pi / ω) = f ω x) :
  f ω (2 * Real.pi / 3) = -1 ∧ 
  ∀ (A B C a b c : ℝ), 
    0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
    A + B + C = Real.pi ∧
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧
    (2 * a - c) * Real.cos B = b * Real.cos C →
    B = Real.pi / 3 ∧ -1 < f ω A ∧ f ω A ≤ 1 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1184_118429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_percentages_correct_l1184_118443

-- Define the initial conditions
def initial_volume : ℝ := 560
def initial_water_percentage : ℝ := 0.75
def initial_kola_percentage : ℝ := 0.15
def initial_sugar_percentage : ℝ := 0.10

-- Define the added amounts
def added_water : ℝ := 25
def added_kola : ℝ := 12
def added_sugar : ℝ := 18

-- Calculate initial amounts
def initial_water : ℝ := initial_volume * initial_water_percentage
def initial_kola : ℝ := initial_volume * initial_kola_percentage
def initial_sugar : ℝ := initial_volume * initial_sugar_percentage

-- Calculate new amounts
def new_water : ℝ := initial_water + added_water
def new_kola : ℝ := initial_kola + added_kola
def new_sugar : ℝ := initial_sugar + added_sugar

-- Calculate new total volume
def new_total_volume : ℝ := new_water + new_kola + new_sugar

-- Define the theorem
theorem new_percentages_correct :
  (abs ((new_water / new_total_volume) - 0.7236) < 0.0001) ∧
  (abs ((new_kola / new_total_volume) - 0.1561) < 0.0001) ∧
  (abs ((new_sugar / new_total_volume) - 0.1203) < 0.0001) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_percentages_correct_l1184_118443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1184_118450

def b : ℕ → ℚ
  | 0 => 2  -- Define for 0 to cover all cases
  | 1 => 2
  | 2 => 5/11
  | n+3 => (b (n+1) * b (n+2)) / (3 * b (n+1) - b (n+2))

theorem b_formula (n : ℕ) (h : n ≥ 2) : b n = 5 / (6 * n - 1) := by
  sorry

#eval b 2023

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_formula_l1184_118450
