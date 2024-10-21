import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pens_per_pack_l920_92006

/-- The number of gel pens Gabby initially has -/
def initial_gel_pens : ℕ := 15

/-- The number of ink pens Gabby initially has -/
def initial_ink_pens : ℕ := 24

/-- The number of pens in each complete pack -/
def x : ℕ := 15

/-- The ratio of gel pens to ink pens after losing one pack of gel pens -/
def ratio_after_loss : ℚ := (initial_gel_pens - x) / initial_ink_pens

/-- The ratio of gel pens to ink pens after finding two packs of ink pens -/
def ratio_after_find : ℚ := initial_gel_pens / (initial_ink_pens + 2 * x)

theorem pens_per_pack : x = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pens_per_pack_l920_92006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l920_92075

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (4 : ℝ)^x - 2^(x+1)

-- State the theorem
theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ f x = y) →
  -1 ≤ y ∧ y ≤ 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l920_92075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2pi_minus_alpha_l920_92009

theorem sin_2pi_minus_alpha (α : ℝ) 
  (h1 : Real.cos (α + π) = Real.sqrt 3 / 2) 
  (h2 : π < α) 
  (h3 : α < 3 * π / 2) : 
  Real.sin (2 * π - α) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2pi_minus_alpha_l920_92009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l920_92033

theorem trig_identity (α : ℝ) :
  (Real.cos (4 * α) * Real.tan (2 * α) - Real.sin (4 * α)) /
  (Real.cos (4 * α) / Real.tan (2 * α) + Real.sin (4 * α)) = -Real.tan (2 * α) ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l920_92033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l920_92037

noncomputable def f (x : ℝ) := 1 / Real.sqrt (2 - x) + Real.log (1 + x)

theorem domain_of_f :
  ∀ x : ℝ, (∃ y : ℝ, f x = y) ↔ -1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l920_92037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l920_92091

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / magnitude_squared * u.1, dot_product / magnitude_squared * u.2)

theorem projection_problem (P : (ℝ × ℝ) → (ℝ × ℝ)) 
  (h : P (2, 4) = (1, 2)) :
  P (3, -6) = (-9/5, -18/5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l920_92091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_100th_bracket_l920_92027

/-- The sequence {2n + 1} -/
def our_sequence (n : ℕ) : ℕ := 2 * n + 1

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The first element of the kth group -/
def first_element (k : ℕ) : ℕ := 2 * triangular_number k + 1

/-- The last element of the kth group -/
def last_element (k : ℕ) : ℕ := 2 * triangular_number (k + 1) - 1

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (l : ℕ) (n : ℕ) : ℕ := n * (a + l) / 2

/-- The theorem to prove -/
theorem sum_of_100th_bracket :
  let k := 14  -- The group containing the 100th bracket
  let n := k   -- The number of elements in the kth group
  let a := first_element k
  let l := last_element k
  arithmetic_sum a l n = 1992 := by
  sorry

#eval arithmetic_sum (first_element 14) (last_element 14) 14

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_100th_bracket_l920_92027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l920_92065

/-- Given vectors in ℝ², prove that if (2a - b) is parallel to c, then λ = -3 -/
theorem parallel_vectors_lambda (a b c : ℝ × ℝ) (l : ℝ) :
  a = (1, 2) →
  b = (0, -2) →
  c = (-1, l) →
  (∃ (k : ℝ), k ≠ 0 ∧ (2 • a - b) = k • c) →
  l = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l920_92065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_to_equality_l920_92098

-- Define the polyhedron and its properties
structure Polyhedron where
  vertices : Finset Nat
  neighbors : Nat → Finset Nat
  is_convex : Bool

-- Define the sequence of values for each vertex
def vertex_value : Nat → Nat → ℤ := sorry

-- Define the neighbor averaging property
axiom neighbor_average {P : Polyhedron} {k n : Nat} :
  k ∈ P.vertices →
  vertex_value k (n + 1) = (P.neighbors k).sum (λ j => vertex_value j n) / (P.neighbors k).card

-- All values are integers
axiom integer_values {P : Polyhedron} {k n : Nat} :
  k ∈ P.vertices → ∃ m : ℤ, vertex_value k n = m

-- The main theorem
theorem convergence_to_equality (P : Polyhedron) :
  P.is_convex →
  ∃ N : Nat, ∀ n k₁ k₂ : Nat, n ≥ N → k₁ ∈ P.vertices → k₂ ∈ P.vertices →
    vertex_value k₁ n = vertex_value k₂ n := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convergence_to_equality_l920_92098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M₀_to_plane_l920_92005

-- Define the points
def M₀ : ℝ × ℝ × ℝ := (-5, -9, 1)
def M₁ : ℝ × ℝ × ℝ := (1, 0, 2)
def M₂ : ℝ × ℝ × ℝ := (1, 2, -1)
def M₃ : ℝ × ℝ × ℝ := (2, -2, 1)

-- Define the plane passing through M₁, M₂, and M₃
def plane (x y z : ℝ) : Prop :=
  -8 * x - 3 * y - 2 * z + 12 = 0

-- Define the distance function from a point to the plane
noncomputable def distance_to_plane (p : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := p
  |(-8 * x - 3 * y - 2 * z + 12)| / Real.sqrt 77

-- Theorem statement
theorem distance_M₀_to_plane :
  distance_to_plane M₀ = Real.sqrt 77 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_M₀_to_plane_l920_92005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_square_factorization_l920_92028

noncomputable section

-- Define the expressions
def expr1 (m : ℝ) : ℝ := (1/16) * m^2 + (1/2) * m + 1
def expr2 (m n : ℝ) : ℝ := 16 * m^2 - 9 * n^2 + 24 * m * n
def expr3 (m n : ℝ) : ℝ := m^2 * n^2 + 64 - 16 * m * n
def expr4 (m n : ℝ) : ℝ := (m - n)^2 - 20 * (m - n) + 100

-- Define a predicate for being directly factorable using the complete square formula
def is_complete_square (f : ℝ → ℝ) : Prop := 
  ∃ a b : ℝ, ∀ x, f x = (a * x + b)^2

-- State the theorem
theorem complete_square_factorization :
  (∃ f : ℝ → ℝ, ∀ m, expr1 m = f m ∧ is_complete_square f) ∧
  (∃ f : ℝ → ℝ, ∀ m n, expr3 m n = f (m * n) ∧ is_complete_square f) ∧
  (∃ f : ℝ → ℝ, ∀ m n, expr4 m n = f (m - n) ∧ is_complete_square f) ∧
  ¬(∃ f : ℝ → ℝ → ℝ, ∀ m n, expr2 m n = f m n ∧ (∀ x y, is_complete_square (λ z ↦ f x y))) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_square_factorization_l920_92028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_Q_radius_final_result_l920_92024

-- Define the triangle ABC
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

-- Define the circles P and Q
structure Circle where
  radius : ℝ

-- Define the problem setup
def problem_setup (ABC : Triangle) (P Q : Circle) : Prop :=
  ABC.AB = 100 ∧
  ABC.AC = 100 ∧
  ABC.BC = 56 ∧
  P.radius = 16 ∧
  -- Additional conditions about tangency and Q being inside ABC are assumed
  True

-- Theorem statement
theorem circle_Q_radius (ABC : Triangle) (P Q : Circle) 
  (h : problem_setup ABC P Q) : 
  Q.radius = 44 - 6 * Real.sqrt 35 := by
  sorry

-- Final result
theorem final_result (ABC : Triangle) (P Q : Circle) 
  (h : problem_setup ABC P Q) : 
  let m : ℕ := 44
  let n : ℕ := 6
  let k : ℕ := 35
  m + n * k = 254 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_Q_radius_final_result_l920_92024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_side_is_2_sqrt_2_x_l920_92049

/-- A configuration of four right-angled triangles forming a square -/
structure TriangleSquareConfig where
  /-- The height of each right-angled triangle -/
  x : ℝ
  /-- The base of each right-angled triangle is 4 times its height -/
  base : ℝ
  /-- The hypotenuse of each triangle forms part of the outer square -/
  hypotenuse : ℝ
  /-- The outer square side length -/
  outer_square_side : ℝ
  /-- The inner square side length -/
  inner_square_side : ℝ
  /-- The base of each right-angled triangle is 4 times its height -/
  base_is_four_times_height : base = 4 * x
  /-- The hypotenuse of each triangle forms part of the outer square -/
  hypotenuse_forms_square : hypotenuse = outer_square_side
  /-- The outer square is formed by the hypotenuses of the four triangles -/
  outer_square_side_eq : outer_square_side = 4 * x
  /-- The inner square exists -/
  inner_square_exists : inner_square_side > 0

/-- The side length of the inner square in the TriangleSquareConfig -/
noncomputable def inner_square_side_length (config : TriangleSquareConfig) : ℝ :=
  2 * Real.sqrt 2 * config.x

/-- Theorem stating that the inner square side length is 2√2x -/
theorem inner_square_side_is_2_sqrt_2_x (config : TriangleSquareConfig) :
    config.inner_square_side = 2 * Real.sqrt 2 * config.x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_square_side_is_2_sqrt_2_x_l920_92049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_reverse_prime_factorization_l920_92023

-- Define the conditions
def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n < 10000

def are_reverses (m n : ℕ) : Prop :=
  ∃ (a b c d : ℕ),
    m = 1000 * a + 100 * b + 10 * c + d ∧
    n = 1000 * d + 100 * c + 10 * b + a

def have_identical_divisors (m n : ℕ) (p q : ℕ) : Prop :=
  (Nat.divisors m).card = q^p - 1 ∧ (Nat.divisors n).card = q^p - 1

def prime_factorization (m n p q r : ℕ) : Prop :=
  m = p * q^q * r ∧ n = q^(p+q) * r

-- Define the theorem
theorem four_digit_reverse_prime_factorization
  (m n p q r : ℕ)
  (h1 : is_four_digit m)
  (h2 : is_four_digit n)
  (h3 : are_reverses m n)
  (h4 : have_identical_divisors m n p q)
  (h5 : prime_factorization m n p q r)
  (h6 : Nat.Prime p)
  (h7 : Nat.Prime q)
  (h8 : Nat.Prime r)
  : m = 1998 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_reverse_prime_factorization_l920_92023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l920_92041

def ComputerGames : Type := String

def game1 : ComputerGames := "Game 1"
def game2 : ComputerGames := "Game 2"

def likedGames : Set ComputerGames := ∅

theorem correct_answer : likedGames = ∅ := by
  rfl

#print correct_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_answer_l920_92041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_constructible_l920_92003

/-- Represents a segment in a plane -/
structure Segment where
  length : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  AB : Segment
  AC : Segment
  AD : Segment
  BC : Segment
  BD : Segment
  CD : Segment

/-- Represents the ability to construct with straight-edge and compass -/
class ConstructibleWith (α : Type) where
  construct : α → Prop

variable (S₁ S₂ S₃ S₄ S₅ S₆ : Segment)
variable (ABCD : Tetrahedron)

/-- The given segments correspond to the edges of the tetrahedron -/
axiom segments_match_tetrahedron :
  S₁.length = ABCD.AB.length ∧
  S₂.length = ABCD.AC.length ∧
  S₃.length = ABCD.AD.length ∧
  S₄.length = ABCD.BC.length ∧
  S₅.length = ABCD.BD.length ∧
  S₆.length = ABCD.CD.length

/-- Definition of an altitude of a tetrahedron -/
noncomputable def altitude (t : Tetrahedron) : Segment :=
  sorry

/-- Theorem stating the constructibility of the altitude -/
theorem altitude_constructible [ConstructibleWith Segment] :
  ConstructibleWith.construct (altitude ABCD) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_constructible_l920_92003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missable_problems_l920_92043

/-- Given a test with 40 problems and a passing score of at least 75%,
    the maximum number of problems a student can miss and still pass is 10. -/
theorem max_missable_problems (total_problems : ℕ) (passing_percentage : ℚ)
    (h1 : total_problems = 40)
    (h2 : passing_percentage ≥ 75 / 100) :
    (total_problems : ℚ) - (passing_percentage * total_problems).floor = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_missable_problems_l920_92043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_distance_l920_92057

/-- An equilateral triangle with side length 1 -/
structure EquilateralTriangle where
  side_length : ℝ
  is_equilateral : side_length = 1

/-- A point in or on the triangle -/
structure Point where
  x : ℝ
  y : ℝ
  in_triangle : Bool

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For any six points in or on an equilateral triangle with side length 1,
    there always exists a pair of points with distance ≤ 1/2 -/
theorem six_points_distance (t : EquilateralTriangle) (points : Fin 6 → Point) :
  ∃ (i j : Fin 6), i ≠ j ∧ distance (points i) (points j) ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_points_distance_l920_92057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l920_92013

/-- The function f(x) = x ln(x) -/
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

/-- The theorem statement -/
theorem tangent_line_and_inequality (a : ℝ) (h : a > 1) :
  (∃ y : ℝ, (1 : ℝ) - y - 1 = 0 ∧ 
    ∀ x : ℝ, x ≠ 1 → (f x - f 1) / (x - 1) = y) ∧
  (∃ c : ℝ, c > 0 ∧ c < 1 / a ∧
    ∀ x : ℝ, c < x → x < 1 → f x > a * x * (x - 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_inequality_l920_92013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_line_equation_for_area_l920_92090

noncomputable section

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := y^2 = -x
def line (k x y : ℝ) : Prop := y = k * (x + 1)

-- Define points A and B as intersections
def intersection_points (k : ℝ) : 
  ∃ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ line k x1 y1 ∧ parabola x2 y2 ∧ line k x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2) := 
  sorry

-- Theorem for dot product
theorem dot_product_zero (k : ℝ) : 
  ∀ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ line k x1 y1 ∧ parabola x2 y2 ∧ line k x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2) → 
  x1 * x2 + y1 * y2 = 0 :=
  sorry

-- Define the area of triangle OAB
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ := (1/2) * abs (x1 * y2 - x2 * y1)

-- Theorem for line equation when area is 5/4
theorem line_equation_for_area (k : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), parabola x1 y1 ∧ line k x1 y1 ∧ parabola x2 y2 ∧ line k x2 y2 ∧ 
   triangle_area x1 y1 x2 y2 = 5/4) →
  (k = 2/3 ∨ k = -2/3) :=
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_product_zero_line_equation_for_area_l920_92090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l920_92032

open Real

noncomputable def f (x : ℝ) : ℝ := (sin (x / 4))^4 + (cos (x / 4))^4

theorem derivative_of_f (x : ℝ) : 
  deriv f x = -sin x / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_f_l920_92032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_on_interval_l920_92073

noncomputable def f (x : ℝ) := 2 * Real.cos x * (Real.sin x - Real.sqrt 3 * Real.cos x) + Real.sqrt 3

theorem f_minimum_on_interval :
  ∃ (x_min : ℝ), x_min ∈ Set.Icc (Real.pi / 2) Real.pi ∧
  f x_min = -2 ∧
  x_min = 11 * Real.pi / 12 ∧
  ∀ x ∈ Set.Icc (Real.pi / 2) Real.pi, f x ≥ f x_min := by
  sorry

#check f_minimum_on_interval

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_minimum_on_interval_l920_92073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_at_distance_sqrt_two_l920_92078

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 4*y - 3 = 1

/-- The line equation -/
def line_eq (x y : ℝ) : Prop := x + y + 1 = 0

/-- The distance from a point (x, y) to the line -/
noncomputable def distance_to_line (x y : ℝ) : ℝ := |x + y + 1| / Real.sqrt 2

/-- The theorem stating that there are exactly 3 points on the circle at distance √2 from the line -/
theorem three_points_at_distance_sqrt_two :
  ∃! (s : Set (ℝ × ℝ)), 
    (∀ p ∈ s, circle_eq p.1 p.2 ∧ distance_to_line p.1 p.2 = Real.sqrt 2) ∧ 
    (∃ (l : List (ℝ × ℝ)), s = l.toFinset ∧ l.length = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_at_distance_sqrt_two_l920_92078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_value_l920_92054

theorem smallest_m_value (m n : ℤ) (h : n > 0) (h2 : m > 0) 
  (h3 : (n : ℚ) - (m : ℚ) / (n : ℚ) = 2011 / 3) : 
  ∃ (k : ℤ), m = k ∧ k > 0 ∧ ∀ (j : ℤ), j > 0 → 
    (∃ (l : ℤ), l > 0 ∧ (n : ℚ) - (j : ℚ) / (n : ℚ) = 2011 / 3) → k ≤ j :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_m_value_l920_92054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l920_92035

/-- The eccentricity of a hyperbola with specific geometric properties --/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := λ x y : ℝ ↦ x^2 / a^2 - y^2 / b^2 = 1
  let c := Real.sqrt (a^2 + b^2)
  let F : ℝ × ℝ := (c, 0)  -- right focus
  let asymptote := λ x : ℝ ↦ b / a * x
  let perpendicular := λ x : ℝ ↦ -a / b * (x - c)
  let P : ℝ × ℝ := (a^2 / c, a * b / c)  -- intersection of asymptote and perpendicular
  let Q : ℝ × ℝ := (0, a * c / (2 * b))  -- intersection of perpendicular bisector and y-axis
  let area_OFQ := c * (a * b / c) / 2
  let area_OPQ := (a * c / (2 * b)) * (a^2 / c) / 2
  area_OFQ = 4 * area_OPQ →
  c / a = Real.sqrt 3
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l920_92035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integer_points_l920_92016

theorem circle_integer_points : 
  let circle := {(x, y) : ℕ × ℕ | x^2 + y^2 = 20}
  ∃! (points : Finset (ℕ × ℕ)), (∀ p ∈ points, p ∈ circle) ∧ points.card = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_integer_points_l920_92016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l920_92061

open Real

-- Define an acute triangle
def is_acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi

-- Define the theorem
theorem triangle_side_range
  (a b c A B C : ℝ)
  (h_acute : is_acute_triangle A B C)
  (h_sides : b^2 - a^2 = a*c)
  (h_c : c = 2) :
  2/3 < a ∧ a < 2 :=
by
  sorry

#check triangle_side_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l920_92061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_symmetry_l920_92088

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := f ω (x - Real.pi / 3)

theorem shifted_symmetry (ω : ℝ) (h_pos : ω > 0) (h_period : ∀ x, f ω (x + 4 * Real.pi) = f ω x) :
  ∀ x, g ω x = -g ω (-x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shifted_symmetry_l920_92088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_equals_one_l920_92020

/-- A function f is an inverse proportion function if there exists a non-zero constant k 
    such that f(x) = k/x for all non-zero x -/
def is_inverse_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, x ≠ 0 → f x = k / x

/-- The function defined by m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1) * (x ^ (|m| - 2))

theorem inverse_proportion_m_equals_one :
  ∀ m : ℝ, is_inverse_proportion (f m) → m = 1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_m_equals_one_l920_92020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_log2_1_to_2009_l920_92071

noncomputable def floor_log2 (n : ℕ) : ℕ := Nat.floor (Real.log n / Real.log 2)

theorem sum_floor_log2_1_to_2009 : 
  (Finset.range 2009).sum (λ i => floor_log2 (i + 1)) = 17944 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_floor_log2_1_to_2009_l920_92071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_and_profit_l920_92000

/-- Represents the sales model for a fruit store -/
structure FruitSalesModel where
  initial_price : ℚ
  initial_sales : ℚ
  price_decrease_effect : ℚ
  price_increase_effect : ℚ
  cost_price : ℚ
  loss_cost : ℚ

/-- Calculates the weekly sales profit given a price change -/
def weekly_profit (model : FruitSalesModel) (price_change : ℚ) : ℚ :=
  let new_price := model.initial_price + price_change
  let new_sales := if price_change ≥ 0 then
    model.initial_sales - model.price_increase_effect * price_change
  else
    model.initial_sales + model.price_decrease_effect * (-price_change)
  (new_price - model.cost_price - model.loss_cost) * new_sales

/-- The theorem stating the optimal price and maximum profit -/
theorem optimal_price_and_profit (model : FruitSalesModel)
  (h1 : model.initial_price = 58)
  (h2 : model.initial_sales = 300)
  (h3 : model.price_decrease_effect = 25)
  (h4 : model.price_increase_effect = 10)
  (h5 : model.cost_price = 35)
  (h6 : model.loss_cost = 3) :
  ∃ (optimal_price_change : ℚ),
    optimal_price_change = -4 ∧
    weekly_profit model optimal_price_change = 6400 ∧
    ∀ (price_change : ℚ),
      weekly_profit model price_change ≤ weekly_profit model optimal_price_change :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_price_and_profit_l920_92000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l920_92022

/-- Given distinct real numbers a, b, c, d, and e where a < b < c < d < e,
    prove that M(m(a,M(b,c)), M(m(a,d), M(b,e))) = e -/
theorem problem_statement 
  (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e)
  (h_order : a < b ∧ b < c ∧ c < d ∧ d < e) : 
  max (min a (max b c)) (max (min a d) (max b e)) = e := by
  sorry

/-- Definition of max function -/
noncomputable def M (x y : ℝ) : ℝ := max x y

/-- Definition of min function -/
noncomputable def m (x y : ℝ) : ℝ := min x y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l920_92022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l920_92053

noncomputable def train_speed_kmh : ℝ := 36
noncomputable def time_passing_tree : ℝ := 14.998800095992321

noncomputable def train_speed_ms : ℝ := train_speed_kmh * 1000 / 3600

noncomputable def train_length : ℝ := train_speed_ms * time_passing_tree

theorem train_length_approx :
  ∃ ε > 0, |train_length - 149.99| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l920_92053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_tangent_l920_92095

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := -Real.exp x + 2 * x

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := -Real.exp x + 2

-- Define the tangent line at (0, f(0))
def tangent_line (x : ℝ) : ℝ := x - 1

-- Theorem statement
theorem shortest_distance_to_tangent :
  ∃ (x : ℝ), 
    let y := Real.exp x
    let dist := |y - tangent_line x| / Real.sqrt 2
    ∀ (x' : ℝ), 
      let y' := Real.exp x'
      let dist' := |y' - tangent_line x'| / Real.sqrt 2
      dist ≤ dist' ∧ dist = Real.sqrt 2 := by
  sorry

#check shortest_distance_to_tangent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_tangent_l920_92095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_fullness_l920_92012

/-- Represents the capacity of a reservoir in billion gallons -/
noncomputable def ReservoirCapacity : ℝ := 340

/-- Represents the amount of water added by the storm in billion gallons -/
noncomputable def StormWater : ℝ := 120

/-- Represents the original content of the reservoir in billion gallons -/
noncomputable def OriginalContent : ℝ := 220

/-- Represents the percentage full after the storm -/
noncomputable def PostStormPercentage : ℝ := 85

/-- Calculates the percentage of the reservoir that was full before the storm -/
noncomputable def preStormPercentage : ℝ := 
  ((PostStormPercentage / 100 * ReservoirCapacity - StormWater) / ReservoirCapacity) * 100

theorem reservoir_fullness :
  |preStormPercentage - 49.7| < 0.1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reservoir_fullness_l920_92012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_route_distance_l920_92042

/-- Represents a route with distance and average speed -/
structure Route where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a route -/
noncomputable def time (r : Route) : ℝ := r.distance / r.speed

theorem first_route_distance :
  ∀ (route1 route2 : Route),
  route1.speed = 75 →
  route2.distance = 750 →
  route2.speed = 25 →
  (min (time route1) (time route2) = 20) →
  route1.distance = 1500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_route_distance_l920_92042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissues_per_pack_l920_92046

/-- Proves the number of tissues in each pack given the problem conditions --/
theorem tissues_per_pack (
  num_boxes : ℕ)
  (packs_per_box : ℕ)
  (cost_per_tissue : ℚ)
  (total_cost : ℚ)
  (h1 : num_boxes = 10)
  (h2 : packs_per_box = 20)
  (h3 : cost_per_tissue = 5 / 100)
  (h4 : total_cost = 1000) :
  (total_cost / (num_boxes : ℚ) / (packs_per_box : ℚ)) / cost_per_tissue = 100 := by
  sorry

-- Remove the #eval line as it's not necessary for building and may cause issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissues_per_pack_l920_92046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_words_for_passing_l920_92080

/-- Represents the exam scoring system for a French vocabulary test. -/
structure FrenchExam where
  total_words : Nat
  semantic_relevance : Float
  minimum_score : Float

/-- Calculates the exam score based on the number of words learned. -/
noncomputable def exam_score (exam : FrenchExam) (words_learned : Nat) : Float :=
  let correct_words := words_learned.toFloat
  let incorrect_words := (exam.total_words - words_learned).toFloat
  let relevant_incorrect := exam.semantic_relevance * incorrect_words
  (correct_words + 0.5 * relevant_incorrect) / exam.total_words.toFloat

/-- Theorem stating that learning at least 574 words guarantees a score of at least 80% on the exam. -/
theorem minimum_words_for_passing (exam : FrenchExam)
  (h1 : exam.total_words = 750)
  (h2 : exam.semantic_relevance = 0.3)
  (h3 : exam.minimum_score = 0.8) :
  ∀ n : Nat, n ≥ 574 → exam_score exam n ≥ exam.minimum_score :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_words_for_passing_l920_92080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l920_92040

/-- The circle C with equation x^2 + y^2 - 4x - 2y + 4 = 0 -/
def C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 1)^2 = 1}

/-- The line L with equation x - 2y - 5 = 0 -/
def L : Set (ℝ × ℝ) :=
  {p | p.1 - 2*p.2 - 5 = 0}

/-- The distance function from a point to the line L -/
noncomputable def dist_to_L (p : ℝ × ℝ) : ℝ :=
  |p.1 - 2*p.2 - 5| / Real.sqrt 5

theorem min_distance_circle_to_line :
  ∀ p ∈ C, dist_to_L p ≥ Real.sqrt 5 - 1 ∧
  ∃ q ∈ C, dist_to_L q = Real.sqrt 5 - 1 := by
  sorry

#check min_distance_circle_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l920_92040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l920_92015

noncomputable def projection (v : ℝ × ℝ) (u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let magnitude_squared := u.1 * u.1 + u.2 * u.2
  let scalar := dot_product / magnitude_squared
  (scalar * u.1, scalar * u.2)

theorem projection_problem (proj : (ℝ × ℝ) → (ℝ × ℝ)) :
  proj (2, -1) = (5, -5/2) →
  proj (-3, 2) = (-16/5, 8/5) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l920_92015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_at_8_l920_92069

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem smallest_sum_at_8 (seq : ArithmeticSequence) 
  (h16 : S seq 16 < 0) (h17 : S seq 17 > 0) :
  ∀ n : ℕ, n ≠ 0 → S seq 8 ≤ S seq n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_at_8_l920_92069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_on_fourth_draw_l920_92070

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the state of a draw -/
inductive DrawState
| Replace
| NoReplace

/-- Represents the bag of balls -/
structure BallBag where
  total : ℕ
  red : ℕ
  white : ℕ
  red_le_total : red ≤ total
  white_le_total : white ≤ total
  sum_eq_total : red + white = total

/-- Represents the drawing process -/
def drawProcess (bag : BallBag) (draws : List DrawState) : Prop :=
  ∀ i : Fin draws.length, 
    (draws.get i = DrawState.Replace ∨ draws.get i = DrawState.NoReplace)

/-- The probability of drawing a red ball on the nth draw -/
noncomputable def probRedOnNthDraw (bag : BallBag) (draws : List DrawState) (n : ℕ) : ℚ :=
  sorry

/-- The theorem to be proved -/
theorem prob_red_on_fourth_draw 
  (bag : BallBag) 
  (draws : List DrawState) 
  (h_bag : bag.total = 8 ∧ bag.red = 5 ∧ bag.white = 3)
  (h_draws : draws = [DrawState.Replace, DrawState.NoReplace, DrawState.Replace, DrawState.NoReplace])
  (h_process : drawProcess bag draws) :
  probRedOnNthDraw bag draws 4 = 5/14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_red_on_fourth_draw_l920_92070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l920_92087

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

theorem triangle_ABC_properties
  (A B C : ℝ)
  (a b c : ℝ)
  (h_triangle : triangle_ABC A B C a b c)
  (h_equation : 3 * (Real.cos B * Real.cos C) + 1 = 3 * (Real.sin B * Real.sin C) + Real.cos (2 * A))
  (h_area : 1/2 * b * c * Real.sin A = 5 * Real.sqrt 3)
  (h_b : b = 5) :
  A = Real.pi/3 ∧ Real.sin B * Real.sin C = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l920_92087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_value_l920_92066

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a ^ 2 = (b + c) / (x - 3))
  (eq2 : b ^ 2 = (a + c) / (y - 3))
  (eq3 : c ^ 2 = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 8)
  (eq5 : x + y + z = 6) :
  x * y * z = 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_value_l920_92066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_equals_9_l920_92068

def t : Finset Int := {-3, 0, 2, 3, 4, 5, 7, 9}
def b : Finset Int := {-2, 4, 5, 6, 7, 8, 10, 12}

def sumEquals9 (x : Int) (y : Int) : Bool := x + y = 9

def favorablePairs : Finset (Int × Int) :=
  (t.product b).filter (fun p => sumEquals9 p.1 p.2)

theorem probability_sum_equals_9 :
  (favorablePairs.card : ℚ) / ((t.card * b.card) : ℚ) = 3 / 32 := by
  sorry

#eval favorablePairs
#eval favorablePairs.card
#eval t.card * b.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_equals_9_l920_92068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_l920_92036

def sean_sum : ℕ := (499 - 1) / 2 + 1

def julie_sum : ℕ := 300

theorem sum_ratio :
  (sean_sum * (1 + 499) / 2 : ℚ) / (julie_sum * (julie_sum + 1) / 2 : ℚ) = 625 / 451.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ratio_l920_92036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_segment_endpoint_l920_92004

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A translation vector in 2D space -/
structure TranslationVector where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def translate (p : Point) (v : TranslationVector) : Point :=
  { x := p.x + v.dx, y := p.y + v.dy }

theorem translated_segment_endpoint (A B A' : Point) : 
  A.x = -4 →
  A.y = -1 →
  B.x = 1 →
  B.y = 1 →
  A'.x = -2 →
  A'.y = 2 →
  let v : TranslationVector := { dx := A'.x - A.x, dy := A'.y - A.y }
  let B' := translate B v
  B'.x = 3 ∧ B'.y = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translated_segment_endpoint_l920_92004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_periodic_function_l920_92050

/-- Represents a periodic function with angular frequency ω --/
noncomputable def periodic_function (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

/-- The minimum positive period of a periodic function --/
noncomputable def min_positive_period (f : ℝ → ℝ) : ℝ := sorry

/-- The minimum distance between adjacent intersection points of y = f(x) and y = 1 --/
noncomputable def min_intersection_distance (f : ℝ → ℝ) : ℝ := sorry

theorem period_of_periodic_function (ω : ℝ) (h_ω : ω > 0) :
  min_intersection_distance (periodic_function ω) = π / 3 →
  min_positive_period (periodic_function ω) = π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_periodic_function_l920_92050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_intervals_of_even_function_l920_92039

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∀ x, x ∈ domain → -x ∈ domain → f (-x) = f x

/-- The decreasing intervals of the absolute value of a function -/
def DecreasingIntervals (f : ℝ → ℝ) (intervals : Set (Set ℝ)) : Prop :=
  ∀ I ∈ intervals, ∀ x y, x ∈ I → y ∈ I → x < y → |f x| > |f y|

/-- Given function f(x) with parameters a and k -/
def f (a k : ℝ) (x : ℝ) : ℝ := -(a+2)*x^2 + (k-1)*x - a

theorem decreasing_intervals_of_even_function (a k : ℝ) :
  let domain := Set.Icc (a-2) (a+4)
  IsEven (f a k) domain →
  DecreasingIntervals (f a k) {Set.Ioo (-3) (-1), Set.Ioo 0 1} :=
by
  sorry

#check decreasing_intervals_of_even_function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_intervals_of_even_function_l920_92039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l920_92089

structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 = 180

def Triangle.isRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

theorem angle_value (ABC CDE : Triangle) 
  (h1 : ABC.angle1 = 70)
  (h2 : ABC.angle2 = 50)
  (h3 : CDE.angle1 = ABC.angle3)
  (h4 : CDE.isRightAngle) : 
  CDE.angle3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_value_l920_92089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l920_92010

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  0 < t.a ∧ 0 < t.b ∧ 0 < t.c ∧
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2 ∧
  Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A ∧
  t.c = 7 ∧
  1/2 * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = Real.pi/3 ∧ t.a + t.b + t.c = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l920_92010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l920_92026

theorem max_log_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 4) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → a + b = 4 → (Real.log a) / (Real.log 2) + (Real.log b) / (Real.log 2) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_log_sum_l920_92026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l920_92064

/-- Given a magnification factor and a magnified diameter, 
    calculate the actual diameter of a tissue sample. -/
noncomputable def actual_diameter (magnification : ℝ) (magnified_diameter : ℝ) : ℝ :=
  magnified_diameter / magnification

/-- Theorem stating that for a magnification of 1000 and a magnified diameter of 0.3 cm,
    the actual diameter is 0.0003 cm. -/
theorem tissue_diameter_calculation :
  let magnification : ℝ := 1000
  let magnified_diameter : ℝ := 0.3
  actual_diameter magnification magnified_diameter = 0.0003 := by
  -- Unfold the definition of actual_diameter
  unfold actual_diameter
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tissue_diameter_calculation_l920_92064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l920_92058

noncomputable def sequence_a (n : ℕ) : ℝ := 1 / (2 * n - 1)

noncomputable def sequence_c (n : ℕ) : ℝ :=
  if n % 2 = 1 then (2 * n - 1) / 19
  else 1 / ((2 * n - 1) * (2 * (n + 2) - 1))

theorem sequence_properties (n : ℕ) (h : n ≥ 1) :
  (∀ k : ℕ, k ≥ 1 → (Finset.range k).sum (λ i => (2 * i + 1) * sequence_a (i + 1)) = k) →
  (∀ m : ℕ, m ≥ 1 → 1 / sequence_a m = 2 * m - 1) ∧
  (Finset.range (2 * n)).sum sequence_c = 
    (1 / 19) * n * (2 * n - 1) + 1 / 12 - 1 / (16 * n + 12) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l920_92058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_zero_l920_92093

theorem polynomial_value_at_zero (p : Polynomial ℝ) :
  Polynomial.degree p = 6 →
  (∀ n : ℕ, n ≤ 6 → p.eval ((3 : ℝ)^n) = ((3 : ℝ)^n)⁻¹) →
  p.eval 0 = 6560 / 2187 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_value_at_zero_l920_92093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l920_92045

/-- Represents the quadrant of an angle -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determines if an angle is in the third quadrant -/
def is_third_quadrant (α : ℝ) : Prop :=
  ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270

/-- Determines the quadrant of an angle -/
noncomputable def angle_quadrant (θ : ℝ) : Quadrant :=
  if 0 ≤ θ % 360 ∧ θ % 360 < 90 then Quadrant.I
  else if 90 ≤ θ % 360 ∧ θ % 360 < 180 then Quadrant.II
  else if 180 ≤ θ % 360 ∧ θ % 360 < 270 then Quadrant.III
  else Quadrant.IV

/-- 
Theorem: If angle α is in the third quadrant, 
then angle α/2 is in either the second or fourth quadrant
-/
theorem half_angle_quadrant (α : ℝ) :
  is_third_quadrant α → 
  (angle_quadrant (α/2) = Quadrant.II ∨ angle_quadrant (α/2) = Quadrant.IV) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_half_angle_quadrant_l920_92045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_12_l920_92060

/-- Represents the dimensions and cost information for a room -/
structure RoomInfo where
  length : ℝ
  width : ℝ
  doorHeight : ℝ
  doorWidth : ℝ
  windowHeight : ℝ
  windowWidth : ℝ
  numWindows : ℕ
  costPerSqFt : ℝ
  totalCost : ℝ

/-- Calculates the height of the room based on the given information -/
noncomputable def calculateRoomHeight (info : RoomInfo) : ℝ :=
  let wallPerimeter := 2 * (info.length + info.width)
  let doorArea := info.doorHeight * info.doorWidth
  let windowArea := info.windowHeight * info.windowWidth * (info.numWindows : ℝ)
  let totalSubtractArea := doorArea + windowArea
  ((info.totalCost / info.costPerSqFt) + totalSubtractArea) / wallPerimeter

/-- Theorem stating that the calculated room height is 12 feet -/
theorem room_height_is_12 (info : RoomInfo) 
    (h1 : info.length = 25)
    (h2 : info.width = 15)
    (h3 : info.doorHeight = 6)
    (h4 : info.doorWidth = 3)
    (h5 : info.windowHeight = 4)
    (h6 : info.windowWidth = 3)
    (h7 : info.numWindows = 3)
    (h8 : info.costPerSqFt = 3)
    (h9 : info.totalCost = 2718) :
  calculateRoomHeight info = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_12_l920_92060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l920_92029

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨h.a * eccentricity h, 0⟩

/-- Checks if three points form an equilateral triangle -/
noncomputable def is_equilateral_triangle (p1 p2 p3 : Point) : Prop :=
  let d12 := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)
  let d23 := Real.sqrt ((p2.x - p3.x)^2 + (p2.y - p3.y)^2)
  let d31 := Real.sqrt ((p3.x - p1.x)^2 + (p3.y - p1.y)^2)
  d12 = d23 ∧ d23 = d31

/-- The main theorem -/
theorem hyperbola_properties (h : Hyperbola) 
  (p q : Point)
  (h_line : p.x = h.a^2 / (h.a * eccentricity h) ∧ q.x = h.a^2 / (h.a * eccentricity h))
  (h_asymptotes : ∃ (m : ℝ), p.y = m * p.x ∧ q.y = -m * q.x)
  (h_equilateral : is_equilateral_triangle (right_focus h) p q)
  (h_chord : ∀ (a b : ℝ), ∃ (l : ℝ), l = h.b^2 * (eccentricity h)^2 / h.a ∧
    l^2 = (1 + a^2) * ((p.x + q.x)^2 - 4 * p.x * q.x)) :
  eccentricity h = 2 ∧
  ((h.a^2 = 2 ∧ h.b^2 = 6) ∨ (h.a^2 = 51/13 ∧ h.b^2 = 153/13)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l920_92029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l920_92051

def U : Set ℕ := {1, 3, 5, 7, 9}

def A (a : ℕ) : Set ℕ := {1, Int.natAbs (a - 5), 9}

def complement_A : Set ℕ := {5, 7}

theorem value_of_a : ∀ a : ℕ, A a ⊆ U ∧ complement_A = U \ A a → a = 2 ∨ a = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_a_l920_92051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_l920_92092

open Real

theorem sin_inequality (x : ℝ) :
  0 < x ∧ x < π / 2 →
  (Real.sin x * Real.sin (2 * x) < Real.sin (3 * x) * Real.sin (4 * x)) ↔
  ((0 < x ∧ x < π / 5) ∨ (2 * π / 5 < x ∧ x < π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_inequality_l920_92092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_probability_l920_92072

/- Define the walk parameters -/
def total_minutes : ℕ := 40
def interval_start : ℝ := -2
def interval_end : ℝ := 2

/- Define the coin flip and walk function -/
noncomputable def coin_flip : Bool → ℝ → ℝ
  | true, n => 1 / n
  | false, n => -1 / n

/- Define the walk process -/
noncomputable def walk (steps : List Bool) : ℝ :=
  steps.enum.foldl (λ acc (i, step) => acc + coin_flip step (i + 1 : ℝ)) 0

/- Define the probability of staying within the interval -/
def prob_in_interval (p : ℝ) : Prop :=
  ∀ steps : List Bool, steps.length = total_minutes →
    interval_start ≤ walk steps ∧ walk steps ≤ interval_end

/- State the theorem -/
theorem walk_probability : 
  ∃ p : ℝ, prob_in_interval p ∧ abs (p - 0.8101502670) < 0.0000000001 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_walk_probability_l920_92072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l920_92076

def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a : ∃ a : ℝ, (A a ∩ B).Nonempty ∧ (A a ∩ C) = ∅ ∧ a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_a_l920_92076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_theorem_l920_92052

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- State that f is bijective (which implies it's invertible)
variable (h_bij : Function.Bijective f)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem inverse_function_point_theorem (h_point : 2 * 2 - f 2 = 1) :
  f_inv f 3 - 2 * 3 = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_theorem_l920_92052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_range_in_obtuse_triangle_l920_92055

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a side
noncomputable def sideLength (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define an obtuse angle
def isObtuse (A B C : ℝ × ℝ) : Prop :=
  (sideLength A C)^2 > (sideLength A B)^2 + (sideLength B C)^2

-- Theorem statement
theorem ac_range_in_obtuse_triangle (t : Triangle) 
  (h_obtuse : isObtuse t.A t.B t.C)
  (h_ab : sideLength t.A t.B = 6)
  (h_cb : sideLength t.C t.B = 8) :
  10 < sideLength t.A t.C ∧ sideLength t.A t.C < 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_range_in_obtuse_triangle_l920_92055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l920_92096

-- Define the circles x and y
def circle_x (r : Real) : Prop := r > 0
def circle_y (r : Real) : Prop := r > 0

-- Define the area of a circle
noncomputable def area (r : Real) : Real := Real.pi * r^2

-- Define the circumference of a circle
noncomputable def circumference (r : Real) : Real := 2 * Real.pi * r

theorem circle_circumference_proof :
  ∀ (rx ry : Real),
  circle_x rx → circle_y ry →
  area rx = area ry →
  ry / 2 = 5 →
  circumference rx = 20 * Real.pi := by
  intros rx ry hx hy ha hr
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_circumference_proof_l920_92096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l920_92038

theorem triangle_cosine_inequality (A B C : ℝ) 
  (h : A + B + C = π) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) : 
  Real.cos A * Real.cos B * Real.cos C ≤ 
    (1 / 24) * (Real.cos (A - B) ^ 2 + Real.cos (B - C) ^ 2 + Real.cos (C - A) ^ 2) ∧
  (1 / 24) * (Real.cos (A - B) ^ 2 + Real.cos (B - C) ^ 2 + Real.cos (C - A) ^ 2) ≤ 1 / 8 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l920_92038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l920_92085

/-- Given a sequence G defined by G(n+1) = (3G(n) + 2)/3 for n ≥ 1 and G(1) = 3,
    prove that G(51) = 109/3 -/
theorem sequence_value (G : ℕ → ℚ) 
    (h₁ : ∀ n, n ≥ 1 → G (n + 1) = (3 * G n + 2) / 3)
    (h₂ : G 1 = 3) : 
  G 51 = 109/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_value_l920_92085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l920_92030

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := (1 + i)^3 / (1 - i)^2

theorem imaginary_part_of_z : Complex.im z = -1 := by
  -- The proof will be skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_z_l920_92030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_DBCF_is_25_l920_92083

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (10, 0)

-- Define midpoints
noncomputable def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
noncomputable def E : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define F (which is the same as B in this case)
def F : ℝ × ℝ := B

-- Define the area of a triangle
noncomputable def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2)) / 2

-- Theorem statement
theorem area_DBCF_is_25 :
  triangle_area D B C + triangle_area D B F = 25 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_DBCF_is_25_l920_92083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solutions_l920_92063

-- Define the properties of exponential inequalities
axiom exp_ineq_prop (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∀ x y : ℝ, a^x > a^y → x < y

-- Define the main theorem
theorem no_positive_integer_solutions (k : ℝ) :
  (∀ x : ℕ+, ((1/2 : ℝ)^(k * (x : ℝ) - 1) ≥ (1/2 : ℝ)^(5 * (x : ℝ) - 2))) ↔ k ≤ 4 := by
  sorry

#check no_positive_integer_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_positive_integer_solutions_l920_92063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_positivity_l920_92007

theorem inequality_and_positivity :
  (∀ a b : ℝ, a^2 + b^2 + 3 ≥ a*b + Real.sqrt 3 * (a + b)) ∧
  (∀ x y z : ℝ, 
    (let a := x^2 + 2*y + π/2;
     let b := y^2 + 2*z + π/3;
     let c := z^2 + 2*x + π/6;
     max a (max b c) > 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_positivity_l920_92007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_bisector_l920_92081

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/5 + y^2 = 1

-- Define a line tangent to the circle
def tangent_line (k m : ℝ) (x y : ℝ) : Prop := y = k*x + m ∧ m^2 = 1 + k^2

-- Define the perpendicular bisector of a chord
def perp_bisector (k m : ℝ) (x y : ℝ) : Prop := 
  x + k*y + (4*k*m)/(1 + 5*k^2) = 0

-- Define the distance from O to the perpendicular bisector
noncomputable def distance_to_bisector (k m : ℝ) : ℝ := 
  |4*k| / (1 + 5*k^2)

theorem max_distance_to_bisector :
  ∀ k m : ℝ, 
  tangent_line k m (1/2) (Real.sqrt 3/2) →
  (∃ x y : ℝ, circle_O x y ∧ ellipse_C x y ∧ tangent_line k m x y) →
  distance_to_bisector k m ≤ 2 * Real.sqrt 5 / 5 :=
by
  sorry

#check max_distance_to_bisector

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_bisector_l920_92081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l920_92018

-- Define the function f as noncomputable
noncomputable def f (t : ℝ) : ℝ := 1 / ((t - 1)^3 + (t + 1)^3)

-- State the theorem about the domain of f
theorem domain_of_f :
  {t : ℝ | ∃ y, f t = y} = {t : ℝ | t ≠ 0} :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l920_92018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_l920_92011

theorem cos_sin_equation_solution (n : ℕ+) :
  ∃ (k : ℤ), (∀ x : ℝ, (Real.cos x) ^ (n : ℕ) - (Real.sin x) ^ (n : ℕ) = 1 → 
    (x = k * π ∨ x = 2 * k * π - π / 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_solution_l920_92011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l920_92034

-- Define the function f(x) = (1/2)^x
noncomputable def f (x : ℝ) : ℝ := (1/2)^x

-- Theorem statement
theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_positive_reals_l920_92034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_paint_cans_l920_92059

-- Define the parameters
def initial_rooms : ℕ := 48
def lost_cans : ℕ := 4
def remaining_rooms : ℕ := 36
def large_rooms : ℕ := 8
def normal_rooms : ℕ := 20

-- Define the theorem
theorem paula_paint_cans :
  let rooms_per_can : ℚ := (initial_rooms - remaining_rooms : ℚ) / lost_cans
  let large_room_equivalent : ℕ := 2
  let total_room_equivalents : ℕ := large_rooms * large_room_equivalent + normal_rooms
  ⌈(total_room_equivalents : ℚ) / rooms_per_can⌉ = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paula_paint_cans_l920_92059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l920_92001

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- Determine if two circles are intersecting -/
def are_circles_intersecting (c1_x c1_y c1_r c2_x c2_y c2_r : ℝ) : Prop :=
  let d := distance c1_x c1_y c2_x c2_y
  (c1_r - c2_r) < d ∧ d < (c1_r + c2_r)

/-- The theorem stating that the given circles are intersecting -/
theorem circles_intersect : 
  are_circles_intersecting (-1) (-4) 5 2 2 3 := by
  -- Proof goes here
  sorry

#check circles_intersect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l920_92001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_theorem_optimal_price_and_sales_theorem_l920_92079

/-- Original price in yuan -/
noncomputable def original_price : ℝ := 25

/-- Original annual sales in units -/
noncomputable def original_sales : ℝ := 80000

/-- Rate of sales decrease per yuan of price increase -/
noncomputable def sales_decrease_rate : ℝ := 2000

/-- Technological innovation cost in millions of yuan -/
noncomputable def tech_cost (x : ℝ) : ℝ := (1/6) * (x^2 - 600)

/-- Fixed advertising cost in millions of yuan -/
noncomputable def fixed_ad_cost : ℝ := 50

/-- Variable advertising cost in millions of yuan -/
noncomputable def var_ad_cost (x : ℝ) : ℝ := x/5

/-- New sales volume in units -/
noncomputable def new_sales (x : ℝ) : ℝ := original_sales - sales_decrease_rate * (x - original_price)

/-- Total revenue function -/
noncomputable def total_revenue (x : ℝ) : ℝ := x * new_sales x

/-- Total cost function -/
noncomputable def total_cost (x : ℝ) : ℝ := (tech_cost x + fixed_ad_cost + var_ad_cost x) * 1000000

/-- Theorem stating the maximum price to maintain or increase revenue -/
theorem max_price_theorem :
  ∃ (max_price : ℝ), max_price = 40 ∧
  ∀ (p : ℝ), p ≤ max_price → total_revenue p ≥ original_price * original_sales :=
sorry

/-- Theorem stating the optimal price and minimum sales volume after reform -/
theorem optimal_price_and_sales_theorem :
  ∃ (optimal_price min_sales : ℝ), 
    optimal_price = 30 ∧
    min_sales = 10.2 * 1000000 ∧
    ∀ (a : ℝ), a ≥ min_sales → 
      a * optimal_price ≥ original_price * original_sales + total_cost optimal_price :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_price_theorem_optimal_price_and_sales_theorem_l920_92079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_bounds_l920_92086

noncomputable section

-- Define the circle
def circle_equation (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 16

-- Define points
def E : ℝ × ℝ := (-Real.sqrt 3, 0)
def F : ℝ × ℝ := (Real.sqrt 3, 0)
def O : ℝ × ℝ := (0, 0)

-- Define the trajectory H
def trajectory_H (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the area of a quadrilateral
noncomputable def area_quadrilateral (A B C D : ℝ × ℝ) : ℝ := sorry

theorem trajectory_and_area_bounds :
  ∃ (M P A B C D : ℝ × ℝ),
    -- M is on the circle
    circle_equation M.1 M.2 ∧
    -- P is on the perpendicular bisector of MF and on EM
    (∃ (t : ℝ), P = E + t • (M - E)) ∧
    -- A and B are on trajectory H
    trajectory_H A.1 A.2 ∧ trajectory_H B.1 B.2 ∧
    -- C is on trajectory H
    trajectory_H C.1 C.2 ∧
    -- |AC| = |CB|
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = (C.1 - B.1)^2 + (C.2 - B.2)^2 ∧
    -- D satisfies CD = CA + CB
    D.1 - C.1 = (A.1 - C.1) + (B.1 - C.1) ∧
    D.2 - C.2 = (A.2 - C.2) + (B.2 - C.2) ∧
    -- P is on trajectory H
    trajectory_H P.1 P.2 ∧
    -- Area of ACBD is bounded
    16/5 ≤ area_quadrilateral A C B D ∧ area_quadrilateral A C B D ≤ 4 :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_area_bounds_l920_92086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_hyperbola_l920_92017

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  ρ : ℝ
  θ : ℝ

/-- Represents a point in Cartesian coordinates -/
structure CartesianPoint where
  x : ℝ
  y : ℝ

/-- Converts a polar point to a Cartesian point -/
noncomputable def polarToCartesian (p : PolarPoint) : CartesianPoint where
  x := p.ρ * Real.cos p.θ
  y := p.ρ * Real.sin p.θ

/-- Defines the curve in polar coordinates -/
def polarCurve (p : PolarPoint) : Prop :=
  p.ρ^2 * Real.cos (2 * p.θ) = 1

/-- Defines a hyperbola in Cartesian coordinates -/
def isHyperbola (f : CartesianPoint → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ p : CartesianPoint, f p ↔ p.x^2 / a^2 - p.y^2 / b^2 = 1

/-- Theorem stating that the polar curve is a hyperbola -/
theorem polar_curve_is_hyperbola :
  isHyperbola (fun c => ∃ p : PolarPoint, polarToCartesian p = c ∧ polarCurve p) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_curve_is_hyperbola_l920_92017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_square_theorem_l920_92056

def circle_tangent_square (arc_length : ℝ) (tangent_angle : ℝ) : Prop :=
  let square_side : ℝ := Real.sqrt (2 + Real.sqrt 2)
  arc_length = Real.sqrt 2 + Real.sqrt (2 - Real.sqrt 2) ∧
  tangent_angle = 45 ∧
  Real.sin (22.5 * Real.pi / 180) = Real.sqrt (2 - Real.sqrt 2) / 2 →
  square_side = Real.sqrt (2 + Real.sqrt 2)

theorem circle_tangent_square_theorem :
  ∀ (arc_length tangent_angle : ℝ),
  circle_tangent_square arc_length tangent_angle :=
by
  sorry

#check circle_tangent_square_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangent_square_theorem_l920_92056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_by_seven_l920_92021

theorem six_digit_divisibility_by_seven (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) :
  (7 ∣ (100000 * a₁ + 10000 * a₂ + 1000 * a₃ + 100 * a₄ + 10 * a₅ + a₆)) →
  (7 ∣ (100000 * a₆ + 10000 * a₁ + 1000 * a₂ + 100 * a₃ + 10 * a₄ + a₅)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_digit_divisibility_by_seven_l920_92021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_ellipse_l920_92074

/-- Definition of the equation -/
def equation (x y a : ℝ) : Prop := x^2 + y^2/a = 1

/-- Definition of an ellipse -/
def is_ellipse (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (x y : ℝ), f x y ↔ x^2/a^2 + y^2/b^2 = 1

/-- Theorem: There exists a real number a such that the equation represents an ellipse -/
theorem exists_a_ellipse : ∃ (a : ℝ), is_ellipse (λ x y => equation x y a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_a_ellipse_l920_92074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_27_5_l920_92048

/-- Trapezoid with vertices P, Q, R, S in 2D space -/
structure Trapezoid where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- Calculate the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  let height := (t.R.1 - t.P.1)
  let base1 := abs (t.Q.2 - t.P.2)
  let base2 := abs (t.R.2 - t.S.2)
  (base1 + base2) * height / 2

/-- The specific trapezoid PQRS from the problem -/
def PQRS : Trapezoid :=
  { P := (2, -3)
    Q := (2, 2)
    R := (7, 9)
    S := (7, 3) }

theorem trapezoid_area_is_27_5 : trapezoidArea PQRS = 27.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_27_5_l920_92048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_of_sequence_l920_92062

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (n : ℕ) (a₁ : ℚ) (aₙ : ℚ) : ℕ → ℚ :=
  λ k ↦ a₁ + (k - 1) * ((aₙ - a₁) / (n - 1))

/-- The 7th term of the specific arithmetic sequence -/
theorem seventh_term_of_sequence :
  arithmetic_sequence 15 3 42 7 = 19.7143 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_of_sequence_l920_92062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l920_92094

theorem m_range_theorem (m : ℝ) : 
  (∀ x : ℝ, (m * (x - 2*m) * (x + m + 3) < 0) ∨ ((2:ℝ)^x - 2 < 0)) ∧
  (∀ x : ℝ, x < -4 → (m * (x - 2*m) * (x + m + 3)) * ((2:ℝ)^x - 2) < 0) ↔
  -4 < m ∧ m < -2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_theorem_l920_92094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_same_eccentricity_and_asymptotes_l920_92019

noncomputable section

-- Define the curves
def curve1 (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1
def curve2 (x y : ℝ) : Prop := x^2 / 9 - y^2 / 3 = 1

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define asymptote
def asymptote (m : ℝ) (x y : ℝ) : Prop := y = m * x

-- Theorem statement
theorem curves_same_eccentricity_and_asymptotes :
  (∃ e : ℝ, eccentricity 3 1 = e ∧ eccentricity (1/3) (1/3) = e) ∧
  (∃ m : ℝ, 
    (∀ x y : ℝ, curve1 x y → (y = m * x ∨ y = -m * x)) ∧
    (∀ x y : ℝ, curve2 x y → (y = m * x ∨ y = -m * x))) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curves_same_eccentricity_and_asymptotes_l920_92019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_through_vertex_l920_92047

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  equation : ℝ × ℝ → Prop := fun (x, y) ↦ y^2 = 2 * p * x

/-- Circle structure -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point on parabola -/
def PointOnParabola (par : Parabola) (point : ℝ × ℝ) : Prop :=
  par.equation point

/-- Chord passing through focus -/
def ChordThroughFocus (par : Parabola) (a b : ℝ × ℝ) : Prop :=
  PointOnParabola par a ∧ PointOnParabola par b ∧ 
  (a.1 + b.1 = par.p) ∧ (a.2 + b.2 = 0)

/-- Circle formed by chord as diameter -/
noncomputable def CircleFromChord (a b : ℝ × ℝ) : Circle :=
  { center := ((a.1 + b.1) / 2, (a.2 + b.2) / 2),
    radius := Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) / 2 }

/-- Common chord of two circles -/
def CommonChord (c1 c2 : Circle) : ℝ × ℝ → Prop :=
  fun (x, y) ↦ (x - c1.center.1)^2 + (y - c1.center.2)^2 = c1.radius^2 ∧
                (x - c2.center.1)^2 + (y - c2.center.2)^2 = c2.radius^2

/-- Main theorem -/
theorem common_chord_through_vertex (par : Parabola) 
  (a b c d : ℝ × ℝ) 
  (hab : ChordThroughFocus par a b) 
  (hcd : ChordThroughFocus par c d) :
  CommonChord (CircleFromChord a b) (CircleFromChord c d) (0, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_through_vertex_l920_92047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_partition_existence_l920_92067

/-- A polygon on a grid that can be partitioned into rectangles -/
structure GridPolygon where
  -- We don't need to define the exact structure, just that it exists
  dummy : Unit

/-- Represents the number of ways a polygon can be partitioned -/
def partition_ways (p : GridPolygon) (n : ℕ) : ℕ :=
  -- This is left abstract as the exact definition is not provided in the problem
  sorry

theorem polygon_partition_existence :
  ∀ n : ℕ, ∃ p : GridPolygon, partition_ways p n = n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_partition_existence_l920_92067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_at_least_eight_sixes_l920_92082

/-- The probability of rolling a six on a fair die -/
def p_six : ℚ := 1/6

/-- The number of times the die is rolled -/
def n_rolls : ℕ := 10

/-- The minimum number of sixes required -/
def min_sixes : ℕ := 8

/-- Calculates the binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of rolling exactly k sixes in n rolls -/
def p_exact (n k : ℕ) : ℚ := 
  (binomial n k : ℚ) * p_six^k * (1 - p_six)^(n - k)

/-- The probability of rolling at least k sixes in n rolls -/
noncomputable def p_at_least (n k : ℕ) : ℚ :=
  Finset.sum (Finset.range (n - k + 1)) (λ i => p_exact n (n - i))

theorem probability_of_at_least_eight_sixes : 
  p_at_least n_rolls min_sixes = 1136/60466176 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_at_least_eight_sixes_l920_92082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_implies_b_equals_one_l920_92099

def is_valid_triple (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ ({1, 2, 4} : Set ℕ) ∧ b ∈ ({1, 2, 4} : Set ℕ) ∧ c ∈ ({1, 2, 4} : Set ℕ)

def expression (a b c : ℕ) : ℚ :=
  (a / 2 : ℚ) / (b / c : ℚ)

theorem max_expression_implies_b_equals_one :
  ∀ a b c : ℕ,
    is_valid_triple a b c →
    (∀ x y z : ℕ, is_valid_triple x y z → expression a b c ≥ expression x y z) →
    expression a b c = 4 →
    b = 1 := by
  sorry

#check max_expression_implies_b_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_implies_b_equals_one_l920_92099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_square_perimeter_distance_graph_shape_l920_92077

-- Define a square with side length 1
def square : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem max_distance_on_square_perimeter :
  ∃ (corner : ℝ × ℝ), corner ∈ square →
    ∀ (p : ℝ × ℝ), p ∈ square →
      distance corner p ≤ Real.sqrt 2 := by
  -- Proof goes here
  sorry

-- Additional lemma to represent the graph shape
theorem distance_graph_shape (t : ℝ) :
  0 ≤ t → t ≤ 4 →
  ∃ (d : ℝ), d = min (Real.sqrt 2) (min t (4 - t)) := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_on_square_perimeter_distance_graph_shape_l920_92077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_m_value_l920_92031

/-- Represents a single digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a single digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a three-digit number in base 5 to base 10 -/
def base5ToBase10 (x y z : Base5Digit) : ℕ :=
  25 * x.val + 5 * y.val + z.val

/-- Converts a three-digit number in base 9 to base 10 -/
def base9ToBase10 (z y x : Base9Digit) : ℕ :=
  81 * z.val + 9 * y.val + x.val

/-- The theorem stating the largest possible value of m -/
theorem largest_m_value (x y : Base5Digit) (z : Base5Digit) :
  (∃ (x' : Base9Digit) (y' z' : Base9Digit), 
    base5ToBase10 x y z = base9ToBase10 z' y' x') →
  base5ToBase10 x y z ≤ 121 := by
  sorry

#check largest_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_m_value_l920_92031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_is_28_4_l920_92025

def tree_heights (h : ℕ → ℕ) : Prop :=
  (∀ i, i ∈ [1, 2, 3, 4, 5] → (h i = 2 * h (i+1) ∨ h i = h (i+1) / 2)) ∧
  (h 2 = 14) ∧ (h 5 = 20)

theorem average_height_is_28_4 (h : ℕ → ℕ) :
  tree_heights h →
  (h 1 + h 2 + h 3 + h 4 + h 5 + h 6) / 6 = 284 / 10 :=
by
  intro hyp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_is_28_4_l920_92025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l920_92014

/-- The maximum area of an equilateral triangle inscribed in a 13 by 14 rectangle --/
theorem max_equilateral_triangle_area_in_rectangle :
  let rectangle_width : ℝ := 14
  let rectangle_height : ℝ := 13
  let max_triangle_area : ℝ := 218 * Real.sqrt 3 - 364
  ∀ (triangle_area : ℝ),
    (∃ (a b c : ℂ),
      -- The triangle is equilateral
      Complex.abs (b - a) = Complex.abs (c - b) ∧
      Complex.abs (c - a) = Complex.abs (b - a) ∧
      -- The triangle is inscribed in the rectangle
      (0 ≤ a.re ∧ a.re ≤ rectangle_width) ∧
      (0 ≤ a.im ∧ a.im ≤ rectangle_height) ∧
      (0 ≤ b.re ∧ b.re ≤ rectangle_width) ∧
      (0 ≤ b.im ∧ b.im ≤ rectangle_height) ∧
      (0 ≤ c.re ∧ c.re ≤ rectangle_width) ∧
      (0 ≤ c.im ∧ c.im ≤ rectangle_height) ∧
      -- The area of the triangle
      triangle_area = (Real.sqrt 3 / 4) * (Complex.abs (b - a))^2) →
    triangle_area ≤ max_triangle_area :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_equilateral_triangle_area_in_rectangle_l920_92014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_problem_l920_92002

/-- Given two lines in a 2D plane, prove their intersection point and the area of a quadrilateral formed by this point and three other points. -/
theorem intersection_and_area_problem (x y : ℝ) : 
  -- Line 1 equation
  y = -3 * x + 18 →
  -- Line 2 equation
  y = -0.8 * x + 8 →
  -- Intersection point E
  (x = 50 / 11 ∧ y = 400 / 11) ∧
  -- Area of quadrilateral OBEC
  let O : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 18)
  let E : ℝ × ℝ := (x, y)
  let C : ℝ × ℝ := (10, 0)
  (1/2 * abs ((O.1 * (E.2 - C.2) + E.1 * (C.2 - O.2) + C.1 * (O.2 - E.2)) +
              (O.1 * (B.2 - E.2) + B.1 * (E.2 - O.2) + E.1 * (O.2 - B.2)))) = 740 / 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_and_area_problem_l920_92002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_two_is_exp_two_l920_92084

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := 2^x

-- Theorem statement
theorem inverse_log_two_is_exp_two :
  Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_two_is_exp_two_l920_92084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_for_specific_angles_l920_92097

/-- The distance between two ships on opposite sides of a lighthouse -/
noncomputable def distance_between_ships (h : ℝ) (α β : ℝ) : ℝ :=
  h * (1 / Real.tan α + 1 / Real.tan β)

/-- Theorem stating the distance between ships given specific conditions -/
theorem distance_for_specific_angles (h : ℝ) :
  h > 0 →
  distance_between_ships h (π/6) (π/4) = h * (Real.sqrt 3 + 1) :=
by
  sorry

#check distance_for_specific_angles 100

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_for_specific_angles_l920_92097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l920_92008

/-- The function f(x) = ln(x+2) + ln(x-2) -/
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (x - 2)

/-- The domain of f(x) is {x | x > 2} -/
def f_domain (x : ℝ) : Prop := x > 2

theorem f_neither_odd_nor_even :
  ¬(∀ x, f_domain x → f (-x) = -f x) ∧
  ¬(∀ x, f_domain x → f (-x) = f x) := by
  sorry

#check f_neither_odd_nor_even

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_neither_odd_nor_even_l920_92008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l920_92044

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ
deriving Inhabited

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents Crystal's running path -/
def crystalPath : List Point :=
  [⟨0, 0⟩, ⟨0, 1⟩, ⟨2, 1⟩, ⟨2, 0⟩]

theorem crystal_run_distance :
  distance (crystalPath.get! 3) (crystalPath.get! 0) = 2 := by
  sorry

#check crystal_run_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_crystal_run_distance_l920_92044
