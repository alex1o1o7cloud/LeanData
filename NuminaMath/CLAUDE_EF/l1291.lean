import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1291_129189

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n_terms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  Finset.sum (Finset.range n) (fun i => seq.a (i + 1))

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ, sum_n_terms a n / sum_n_terms b n = (3 * n + 2 : ℚ) / (2 * n + 1)) →
  a.a 7 / b.a 5 = 41 / 19 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l1291_129189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polyhedron_with_no_visible_vertices_l1291_129139

/-- A polyhedron in 3-dimensional space -/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))
  -- Add necessary conditions for a valid polyhedron

/-- Predicate to check if a point is outside a polyhedron -/
def IsOutside (p : Polyhedron) (q : Fin 3 → ℝ) : Prop :=
  q ∉ p.vertices ∧ ∀ f ∈ p.faces, q ∉ f

/-- Predicate to check if a vertex is visible from a point -/
def IsVisible (p : Polyhedron) (v : Fin 3 → ℝ) (q : Fin 3 → ℝ) : Prop :=
  v ∈ p.vertices ∧
  ∀ t : ℝ, 0 < t ∧ t < 1 →
    (fun i => (1 - t) * q i + t * v i) ∉ p.vertices

/-- Theorem stating the existence of a polyhedron with no visible vertices from an outside point -/
theorem exists_polyhedron_with_no_visible_vertices :
  ∃ (p : Polyhedron) (q : Fin 3 → ℝ),
    IsOutside p q ∧
    ∀ v ∈ p.vertices, ¬IsVisible p v q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_polyhedron_with_no_visible_vertices_l1291_129139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sgn_g_eq_neg_sgn_x_l1291_129159

-- Define the signum function
noncomputable def sgn (x : ℝ) : ℝ :=
  if x > 0 then 1
  else if x < 0 then -1
  else 0

-- Define the properties of f and g
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Theorem statement
theorem sgn_g_eq_neg_sgn_x
  (f : ℝ → ℝ) (hf : IsIncreasing f) (a : ℝ) (ha : a > 1) :
  ∀ x, sgn (f x - f (a * x)) = -sgn x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sgn_g_eq_neg_sgn_x_l1291_129159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_equation_correct_l1291_129134

/-- Represents the number of people transferred from Team B to Team A -/
def x : ℤ := sorry

/-- The initial number of people in Team A -/
def initial_team_a : ℤ := 28

/-- The initial number of people in Team B -/
def initial_team_b : ℤ := 20

/-- The number of people in Team A after the transfer -/
def final_team_a : ℤ := initial_team_a + x

/-- The number of people in Team B after the transfer -/
def final_team_b : ℤ := initial_team_b - x

/-- Theorem stating that the equation correctly represents the situation -/
theorem transfer_equation_correct :
  final_team_a = 2 * final_team_b ↔ initial_team_a + x = 2 * (initial_team_b - x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transfer_equation_correct_l1291_129134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intercept_l1291_129182

/-- The function f(x) = e^x - ax^2 - 2x - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x^2 - 2 * x - 1

/-- The derivative of f -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * a * x - 2

theorem tangent_line_intercept (a : ℝ) : 
  (f' a 1) * (-1) + f a 1 = -2 → a = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_intercept_l1291_129182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_area_bounds_l1291_129171

noncomputable section

-- Define the circle and points
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def circle_radius : ℝ := 2 * Real.sqrt 2

-- Define the locus curve E
def E (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define points on the curve
variable (S T M N : ℝ × ℝ)

-- Define the conditions for the points
def points_on_curve (S T M N : ℝ × ℝ) : Prop :=
  E S.1 S.2 ∧ E T.1 T.2 ∧ E M.1 M.2 ∧ E N.1 N.2

def vectors_collinear (S T M N : ℝ × ℝ) : Prop :=
  ∃ (k₁ k₂ : ℝ), (k₁ • (S.1 - B.1, S.2 - B.2) = (T.1 - B.1, T.2 - B.2)) ∧
                 (k₂ • (M.1 - B.1, M.2 - B.2) = (N.1 - B.1, N.2 - B.2))

def vectors_orthogonal (S M : ℝ × ℝ) : Prop :=
  (S.1 - B.1) * (M.1 - B.1) + (S.2 - B.2) * (M.2 - B.2) = 0

-- Define the area of quadrilateral SMTN
def area_SMTN (S T M N : ℝ × ℝ) : ℝ :=
  abs ((S.1 - M.1) * (T.2 - N.2) - (S.2 - M.2) * (T.1 - N.1)) / 2

-- Theorem statement
theorem locus_and_area_bounds
  (h_points : points_on_curve S T M N)
  (h_collinear : vectors_collinear S T M N)
  (h_orthogonal : vectors_orthogonal S M) :
  (E = λ x y => x^2 / 2 + y^2 = 1) ∧
  (16 / 9 ≤ area_SMTN S T M N) ∧
  (area_SMTN S T M N ≤ 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_and_area_bounds_l1291_129171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1291_129165

/-- The distance from a point in polar coordinates to a line in polar form --/
noncomputable def distance_point_to_line (r : ℝ) (θ : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  sorry

/-- The line equation ρsin(θ+π/3)=1 in polar coordinates --/
def line_equation (ρ θ : ℝ) : Prop :=
  ρ * Real.sin (θ + Real.pi/3) = 1

theorem distance_to_line :
  distance_point_to_line 3 (Real.pi/6) line_equation = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_l1291_129165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_XY_l1291_129136

-- Define the points on the line
variable (A B C D : ℝ)

-- Define the circles
variable (Ω₁ Ω₂ Ω₃ : Set (ℝ × ℝ))

-- Define the intersection points
variable (X Y : ℝ × ℝ)

-- State the conditions
axiom points_order : A < B ∧ B < C ∧ C < D
axiom distance_AB_CD : |B - A| = 4 ∧ |D - C| = 4
axiom distance_BC : |C - B| = 8

-- Define the circles
axiom circle_Ω₁ : Ω₁ = {p : ℝ × ℝ | (p.1 - (A + B)/2)^2 + p.2^2 = 4}
axiom circle_Ω₂ : Ω₂ = {p : ℝ × ℝ | (p.1 - (B + C)/2)^2 + p.2^2 = 16}
axiom circle_Ω₃ : Ω₃ = {p : ℝ × ℝ | (p.1 - (C + D)/2)^2 + p.2^2 = 4}

-- Define the tangent line
axiom tangent_line : ∃ m k : ℝ, ∀ p : ℝ × ℝ, 
  p.2 = m * p.1 + k ∧ 
  (p ∈ Ω₃ → (p.1 - A)^2 + p.2^2 = ((C + D)/2 - A)^2) ∧
  (p ∈ Ω₂ → (X = p ∨ Y = p))

-- State the theorem
theorem length_XY : |X.1 - Y.1| + |X.2 - Y.2| = 24 * Real.sqrt 5 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_XY_l1291_129136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_strictly_increasing_l1291_129146

noncomputable section

-- Define the four functions
def f1 (x : ℝ) : ℝ := -2 * x
def f2 (x : ℝ) : ℝ := 3 * x - 1
def f3 (x : ℝ) : ℝ := 1 / x
def f4 (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem only_f2_strictly_increasing :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f1 x₁ > f1 x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f2 x₁ < f2 x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → x₁ > 0 → x₂ > 0 → f3 x₁ > f3 x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ f4 x₁ > f4 x₂) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_f2_strictly_increasing_l1291_129146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1291_129103

-- Define the inequality and its solution set
def has_solution_set (a b : ℝ) : Prop :=
  ∀ x, x^2 - a*x + b < 0 ↔ 1 < x ∧ x < 2

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := (a-1) * Real.sqrt (x-3) + (b-1) * Real.sqrt (4-x)

-- State the theorem
theorem max_value_of_f (a b : ℝ) (h : has_solution_set a b) :
  ∃ x, f a b x ≤ Real.sqrt 5 ∧ ∀ y, f a b y ≤ f a b x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1291_129103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_l1291_129184

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (y^2 / a^2) + (x^2 / b^2) = 1

-- Define the structure for a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the ellipse parameters and points
variable {a b : ℝ}
variable {F₁ F₂ A M N D : Point}

-- Helper function for distance between two points
noncomputable def dist (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- State the theorem
theorem minor_axis_length
  (h_ellipse : ellipse a b A.x A.y)
  (h_a_gt_b : a > b)
  (h_b_pos : b > 0)
  (h_foci : F₁.x = 0 ∧ F₂.x = 0)  -- Foci on y-axis
  (h_right_vertex : A.x = b ∧ A.y = 0)  -- A is right vertex
  (h_midpoint : D.x = A.x / 2 ∧ D.y = F₂.y / 2)  -- D is midpoint of AF₂
  (h_perpendicular : (F₁.y - D.y) * (F₂.x - A.x) = -(F₁.x - D.x) * (F₂.y - A.y))  -- F₁D ⊥ AF₂
  (h_intersection : ellipse a b M.x M.y ∧ ellipse a b N.x N.y)  -- M and N on ellipse
  (h_perimeter : dist A M + dist M N + dist N A = 28)  -- Perimeter of △AMN is 28
  : 2 * b = 7 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minor_axis_length_l1291_129184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1291_129158

noncomputable section

-- Define the hyperbola C
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the point G
def G : ℝ × ℝ := (Real.sqrt 15 / 3, 2 * Real.sqrt 3 / 3)

-- Define the foci condition
def foci_condition (F₁ F₂ : ℝ × ℝ) : Prop :=
  let (x, y) := G
  (F₁.1 - x) * (F₂.1 - x) + (F₁.2 - y) * (F₂.2 - y) = 0

-- Define the line l
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x - 1

-- Define the theorem
theorem hyperbola_theorem (a b : ℝ) (F₁ F₂ : ℝ × ℝ) (k : ℝ) :
  hyperbola a b G.1 G.2 →
  foci_condition F₁ F₂ →
  (∃ (M N P Q : ℝ × ℝ),
    hyperbola a b M.1 M.2 ∧
    hyperbola a b N.1 N.2 ∧
    M.2 = line_l k M.1 ∧
    N.2 = line_l k N.1 ∧
    P.2 = line_l k P.1 ∧
    Q.2 = line_l k Q.1 ∧
    P.1 < 0 ∧ Q.1 > 0) →
  (a = 1 ∧ b = Real.sqrt 2) ∧
  (∀ (M N P Q : ℝ × ℝ),
    hyperbola a b M.1 M.2 →
    hyperbola a b N.1 N.2 →
    M.2 = line_l k M.1 →
    N.2 = line_l k N.1 →
    P.2 = line_l k P.1 →
    Q.2 = line_l k Q.1 →
    P.1 < 0 →
    Q.1 > 0 →
    0 < (dist M P / dist P Q + dist Q N / dist P Q) ∧
    (dist M P / dist P Q + dist Q N / dist P Q) ≤ Real.sqrt 3 - 1) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_theorem_l1291_129158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1291_129186

-- Define the ordering of p, q, r, s, t
def ordered_reals (p q r s t : ℝ) : Prop :=
  p < q ∧ q < r ∧ r < s ∧ s < t

-- Define the maximum function
noncomputable def M : ℝ → ℝ → ℝ := max

-- Define the minimum function
noncomputable def m : ℝ → ℝ → ℝ := min

-- Theorem statement
theorem problem_statement 
  (p q r s t : ℝ) 
  (h : ordered_reals p q r s t) :
  M (M p (m q r)) (m s (m p t)) = q := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l1291_129186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jump_l1291_129145

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop :=
  (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2 = 
  (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 ∧
  (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2 = 
  (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2

/-- Checks if a triangle is isosceles with a right angle at the third point -/
def isIsoscelesRight (A B C : Point) : Prop :=
  (A.x - C.x)^2 + (A.y - C.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2 ∧
  (A.x - C.x) * (B.x - C.x) + (A.y - C.y) * (B.y - C.y) = 0

/-- Calculates the distance between two points -/
noncomputable def distance (P Q : Point) : ℝ :=
  ((P.x - Q.x)^2 + (P.y - Q.y)^2).sqrt

/-- Reflects a point P with respect to a center C -/
def reflect (P C : Point) : Point :=
  { x := 2 * C.x - P.x, y := 2 * C.y - P.y }

/-- Generates the nth point in the sequence -/
def generatePoint (ABC : Triangle) (P₀ : Point) : ℕ → Point
  | 0 => P₀
  | n + 1 => 
    let Pₙ := generatePoint ABC P₀ n
    reflect Pₙ (match n % 3 with
      | 0 => ABC.A
      | 1 => ABC.B
      | _ => ABC.C)

/-- The main theorem -/
theorem grasshopper_jump (ABC : Triangle) (P₀ : Point) (n : ℕ) :
  isEquilateral ABC →
  isIsoscelesRight ABC.A P₀ ABC.C →
  distance P₀ (generatePoint ABC P₀ n) = distance P₀ (generatePoint ABC P₀ (n % 6)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grasshopper_jump_l1291_129145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_equation_l1291_129175

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_equation (z : ℂ) (a : ℝ) 
  (h1 : is_pure_imaginary z) 
  (h2 : (1 - Complex.I) * z = 1 + a * Complex.I) : 
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_equation_l1291_129175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_front_crawl_time_is_eight_minutes_l1291_129114

/-- Represents the swimming problem with given conditions -/
structure SwimmingProblem where
  total_distance : ℚ
  front_crawl_speed : ℚ
  breaststroke_speed : ℚ
  total_time : ℚ

/-- Calculates the time spent swimming front crawl -/
def front_crawl_time (p : SwimmingProblem) : ℚ :=
  (p.total_distance - p.breaststroke_speed * p.total_time) / (p.front_crawl_speed - p.breaststroke_speed)

/-- Theorem stating that the time spent swimming front crawl is 8 minutes -/
theorem front_crawl_time_is_eight_minutes (p : SwimmingProblem) 
    (h1 : p.total_distance = 500)
    (h2 : p.front_crawl_speed = 45)
    (h3 : p.breaststroke_speed = 35)
    (h4 : p.total_time = 12) :
  front_crawl_time p = 8 := by
  sorry

/-- Evaluation of the front crawl time for the given problem -/
def problem_evaluation : ℚ :=
  front_crawl_time { total_distance := 500, front_crawl_speed := 45, breaststroke_speed := 35, total_time := 12 }

#eval problem_evaluation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_front_crawl_time_is_eight_minutes_l1291_129114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_triangles_l1291_129194

-- Define the number of points on the circle
def n : ℕ := 9

-- Define the number of chords
def num_chords : ℕ := n.choose 2

-- Define the number of intersections inside the circle
def num_intersections : ℕ := n.choose 4

-- Define the number of triangles formed by intersections
def num_triangles : ℕ := num_intersections.choose 3

-- Helper definition for a chord (simplified for demonstration)
def chord (p q : ℕ) : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem chord_intersection_triangles :
  (n = 9) →
  (∀ (p q r : ℕ), p < n ∧ q < n ∧ r < n → 
    ¬(∃ (x : ℝ × ℝ), x ∈ Set.univ \ {(0, 0)} ∧ 
      x ∈ chord p q ∩ chord q r ∩ chord r p)) →
  num_triangles = 315750 :=
by
  intro h_n h_no_triple_intersection
  -- The proof would go here, but we'll use sorry for now
  sorry

#check chord_intersection_triangles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_intersection_triangles_l1291_129194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zaporozhets_equidistant_l1291_129154

/-- Represents a car with its position and velocity -/
structure Car where
  position : ℝ
  velocity : ℝ

/-- Represents the state of the cars at a given time -/
structure CarState where
  moskvich : Car
  zaporozhets : Car
  niva : Car
  observer : ℝ

/-- The condition when Moskvich passes the observer -/
def moskvichPassingCondition (state : CarState) : Prop :=
  state.moskvich.position = state.observer ∧ 
  |state.moskvich.position - state.zaporozhets.position| = 
  |state.moskvich.position - state.niva.position|

/-- The condition when Niva passes the observer -/
def nivaPassingCondition (state : CarState) : Prop :=
  state.niva.position = state.observer ∧
  |state.niva.position - state.moskvich.position| = 
  |state.niva.position - state.zaporozhets.position|

/-- The theorem to be proved -/
theorem zaporozhets_equidistant 
  (initial : CarState) 
  (t : ℝ) 
  (h1 : moskvichPassingCondition initial) 
  (h2 : ∃ t1, nivaPassingCondition 
    { moskvich := { position := initial.moskvich.position + initial.moskvich.velocity * t1, 
                    velocity := initial.moskvich.velocity },
      zaporozhets := { position := initial.zaporozhets.position + initial.zaporozhets.velocity * t1,
                       velocity := initial.zaporozhets.velocity },
      niva := { position := initial.niva.position + initial.niva.velocity * t1,
                velocity := initial.niva.velocity },
      observer := initial.observer }) 
  (h3 : initial.zaporozhets.position + initial.zaporozhets.velocity * t = initial.observer) :
  |initial.moskvich.position + initial.moskvich.velocity * t - (initial.zaporozhets.position + initial.zaporozhets.velocity * t)| = 
  |initial.niva.position + initial.niva.velocity * t - (initial.zaporozhets.position + initial.zaporozhets.velocity * t)| :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zaporozhets_equidistant_l1291_129154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_puzzle_l1291_129180

inductive Person : Type
| A
| B
| C

inductive Role : Type
| Knight
| Liar
| Spy

def assignment : Person → Role := sorry

def statement (p : Person) (s : Prop) : Prop := sorry

axiom knight_truth : ∀ (p : Person) (s : Prop), assignment p = Role.Knight → (statement p s ↔ s)
axiom liar_lie : ∀ (p : Person) (s : Prop), assignment p = Role.Liar → (statement p s ↔ ¬s)

axiom A_statement : statement Person.A (assignment Person.C = Role.Liar)
axiom C_statement1 : statement Person.C (assignment Person.A = Role.Knight)
axiom C_statement2 : statement Person.C (assignment Person.C = Role.Spy)

axiom one_of_each : ∃! (k l s : Person), 
  assignment k = Role.Knight ∧ 
  assignment l = Role.Liar ∧ 
  assignment s = Role.Spy

theorem solve_puzzle : 
  assignment Person.A = Role.Knight ∧ 
  assignment Person.B = Role.Spy ∧ 
  assignment Person.C = Role.Liar := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_puzzle_l1291_129180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_l1291_129196

noncomputable def fractional_part (x : ℝ) : ℝ := x - ⌊x⌋

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = 7 ∧ 
  (∀ y : ℝ, (⌊y⌋ = 7 + 75 * (fractional_part y) ∧ 
             0 ≤ fractional_part y ∧ 
             fractional_part y < 1) → 
            x ≤ y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_solution_floor_equation_l1291_129196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l1291_129104

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x * 2^(x + a) - 1 else -(-x * 2^(-x + a) - 1)

-- State the theorem
theorem odd_function_a_value :
  ∀ a : ℝ, 
  (∀ x : ℝ, f a x = -(f a (-x))) →  -- f is odd
  f a (-1) = 3/4 →                  -- f(-1) = 3/4
  a = -3 :=                         -- a = -3
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l1291_129104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_intersection_l1291_129161

open Real

noncomputable def f (x : ℝ) := (1/8) * x^2 - log x

theorem unique_tangent_intersection :
  ∃! t : ℝ, t ∈ Set.Ioo 0 2 ∧
    ∀ x : ℝ, x ∈ Set.Ioo 0 2 →
      (x ≠ t → f x ≠ f t + (deriv f t) * (x - t)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_intersection_l1291_129161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silvia_order_cost_l1291_129111

/-- Calculates the final cost of an order with a potential discount --/
def calculate_order_cost (quiche_price quiche_quantity croissant_price croissant_quantity biscuit_price biscuit_quantity discount_rate discount_threshold : ℚ) : ℚ :=
  let total_before_discount := quiche_price * quiche_quantity + croissant_price * croissant_quantity + biscuit_price * biscuit_quantity
  let discount_amount := if total_before_discount > discount_threshold then total_before_discount * discount_rate else 0
  total_before_discount - discount_amount

/-- Proves that the final cost of Silvia's order is $54.00 --/
theorem silvia_order_cost : 
  calculate_order_cost 15 2 3 6 2 6 (1/10) 50 = 54 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silvia_order_cost_l1291_129111


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_lower_variance_choose_athlete_B_l1291_129188

/-- Represents an athlete's performance statistics -/
structure AthletePerformance where
  mean : ℝ
  variance : ℝ

/-- Determines which athlete should be chosen based on their performance statistics -/
noncomputable def chooseAthlete (a b : AthletePerformance) : Bool :=
  a.variance < b.variance

/-- Theorem: Given two athletes with equal mean performance, the one with lower variance should be chosen -/
theorem choose_lower_variance 
  (a b : AthletePerformance) 
  (h1 : a.mean = b.mean) 
  (h2 : a.variance > b.variance) : 
  chooseAthlete b a = true := by
  sorry

/-- Application to the specific problem -/
def athleteA : AthletePerformance := { mean := 0, variance := 3.5 }
def athleteB : AthletePerformance := { mean := 0, variance := 2.8 }

/-- Theorem: Athlete B should be chosen for the competition -/
theorem choose_athlete_B : 
  chooseAthlete athleteB athleteA = true := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_choose_lower_variance_choose_athlete_B_l1291_129188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_deduction_l1291_129142

-- Define the universe
variable (U : Type)

-- Define the sets
variable (Dar Par Jar Tar : Set U)

-- Define the conditions
variable (h1 : Dar ⊆ (Set.univ \ Par))
variable (h2 : Par ∩ Jar = ∅)
variable (h3 : ∃ x, x ∈ Dar ∩ Tar)

-- Define the statements that cannot be deduced
def statement1 (U : Type) (Tar Jar : Set U) : Prop := ∃ x, x ∈ Tar ∧ x ∉ Jar
def statement2 (U : Type) (Jar Tar : Set U) : Prop := Jar ∩ Tar = ∅
def statement3 (U : Type) (Dar Jar : Set U) : Prop := ∃ x, x ∈ Dar ∩ Jar
def statement4 (U : Type) (Tar Par : Set U) : Prop := ∃ x, x ∈ Tar ∩ Par

-- Theorem stating that none of the statements can be deduced
theorem no_deduction (U : Type) (Dar Par Jar Tar : Set U) 
  (h1 : Dar ⊆ (Set.univ \ Par)) (h2 : Par ∩ Jar = ∅) (h3 : ∃ x, x ∈ Dar ∩ Tar) : 
  ¬(statement1 U Tar Jar ∨ statement2 U Jar Tar ∨ statement3 U Dar Jar ∨ statement4 U Tar Par) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_deduction_l1291_129142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l1291_129155

/-- The probability of drawing two balls of the same color from a bag containing
    5 green balls and 8 white balls. -/
theorem same_color_probability (green : ℕ) (white : ℕ) 
  (h_green : green = 5) (h_white : white = 8) : 
  let total := green + white
  let prob_green := (green : ℚ) / total * ((green - 1) / (total - 1))
  let prob_white := (white : ℚ) / total * ((white - 1) / (total - 1))
  prob_green + prob_white = 19 / 39 := by
  sorry

#check same_color_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_probability_l1291_129155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l1291_129195

/-- Represents a triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Properties of the specific triangle in the problem -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.A + t.C = 2 * Real.pi / 3 ∧
  t.b = 1 ∧
  0 < t.A ∧ t.A < Real.pi / 2 ∧
  0 < t.B ∧ t.B < Real.pi / 2 ∧
  0 < t.C ∧ t.C < Real.pi / 2

noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.a * t.c * Real.sin t.B

theorem special_triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  Real.sqrt 3 < t.a + t.c ∧ t.a + t.c ≤ 2 ∧
  (∀ (s : Triangle), SpecialTriangle s → area s ≤ Real.sqrt 3 / 4) ∧
  (∃ (s : Triangle), SpecialTriangle s ∧ area s = Real.sqrt 3 / 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_properties_l1291_129195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l1291_129107

noncomputable def original_function (x : ℝ) : ℝ := Real.sin x

noncomputable def transformation1 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (x + Real.pi/3)

noncomputable def transformation2 (f : ℝ → ℝ) : ℝ → ℝ := λ x ↦ f (2*x)

noncomputable def result_function (x : ℝ) : ℝ := Real.sin (2*x + Real.pi/3)

theorem transformations_result :
  ∀ x : ℝ, (transformation2 (transformation1 original_function)) x = result_function x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformations_result_l1291_129107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l1291_129168

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2

-- Theorem stating the condition and conclusion
theorem a_value (a : ℝ) : 
  (∀ x, deriv (f a) x = 2 * a * x) →
  deriv (f a) (-1) = 4 →
  a = -2 := by
  intro h1 h2
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_value_l1291_129168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positive_iff_a_in_range_l1291_129153

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a * x^2 - x + 1/2) / Real.log a

theorem function_positive_iff_a_in_range :
  ∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, f a x > 0) ↔ 
    a ∈ Set.union (Set.Ioo (1/2) (5/8)) (Set.Ioi (3/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_positive_iff_a_in_range_l1291_129153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_max_area_exists_l1291_129162

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid ABCD -/
structure Trapezoid where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Calculates the centroid of a triangle -/
noncomputable def centroid (p q r : Point) : Point :=
  { x := (p.x + q.x + r.x) / 3,
    y := (p.y + q.y + r.y) / 3 }

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (t : Trapezoid) : ℝ :=
  ((distance t.A t.B) + (distance t.C t.D)) * (t.C.y - t.A.y) / 2

/-- Theorem: There exists a maximum area for trapezoid ABCD under given conditions -/
theorem trapezoid_max_area_exists :
  ∃ (t : Trapezoid),
    distance t.A t.B = 4 ∧
    distance t.C t.D = 8 ∧
    distance t.B t.C = 3 ∧
    distance t.A t.D = 2 * Real.sqrt 5 ∧
    (let g1 := centroid t.A t.B t.C
     let g2 := centroid t.B t.C t.D
     let g3 := centroid t.A t.C t.D
     distance g1 g2 = distance g2 g3 ∧ distance g2 g3 = distance g3 g1) ∧
    ∀ (t' : Trapezoid),
      distance t'.A t'.B = 4 →
      distance t'.C t'.D = 8 →
      distance t'.B t'.C = 3 →
      distance t'.A t'.D = 2 * Real.sqrt 5 →
      (let g1' := centroid t'.A t'.B t'.C
       let g2' := centroid t'.B t'.C t'.D
       let g3' := centroid t'.A t'.C t'.D
       distance g1' g2' = distance g2' g3' ∧
       distance g2' g3' = distance g3' g1') →
      trapezoidArea t ≥ trapezoidArea t' :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_max_area_exists_l1291_129162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1291_129131

noncomputable def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := 2*x

def is_covering_function (f g : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ x, m ≤ x ∧ x ≤ n → f x ≤ g x

theorem max_value_theorem (m n : ℝ) 
  (h : is_covering_function f g m n) : 
  (2 : ℝ)^(|m - n|) ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_theorem_l1291_129131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_percentage_l1291_129124

/-- The volume of a right circular cylinder given its height and circumference -/
noncomputable def cylinderVolume (height : ℝ) (circumference : ℝ) : ℝ :=
  (circumference ^ 2 * height) / (4 * Real.pi)

/-- The percentage of one volume relative to another -/
noncomputable def volumePercentage (v1 v2 : ℝ) : ℝ :=
  (v1 / v2) * 100

theorem tank_volume_percentage :
  let tankA_volume := cylinderVolume 7 8
  let tankB_volume := cylinderVolume 8 10
  volumePercentage tankA_volume tankB_volume = 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_volume_percentage_l1291_129124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gigi_mushroom_count_l1291_129183

/-- Given the conditions of GiGi's mushroom cutting and distribution, 
    prove that she initially cut 22 whole mushrooms. -/
theorem gigi_mushroom_count 
  (pieces_per_mushroom : ℕ)
  (kenny_pieces : ℕ)
  (karla_pieces : ℕ)
  (remaining_pieces : ℕ)
  (h1 : pieces_per_mushroom = 4)
  (h2 : kenny_pieces = 38)
  (h3 : karla_pieces = 42)
  (h4 : remaining_pieces = 8) :
  (kenny_pieces + karla_pieces + remaining_pieces) / pieces_per_mushroom = 22 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gigi_mushroom_count_l1291_129183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_angle_theorem_l1291_129149

/-- Regular quadrilateral pyramid with circumscribed sphere -/
structure RegularQuadPyramid where
  a : ℝ  -- Side length of the base
  R : ℝ  -- Radius of the circumscribed sphere
  h : R = (3/4) * a  -- Condition: ratio of sphere radius to base side is 3:4

/-- The angle between a lateral face and the base plane of a regular quadrilateral pyramid -/
noncomputable def lateral_angle (p : RegularQuadPyramid) : ℝ :=
  if h : p.R = (3/4) * p.a
  then Real.arctan (Real.sqrt 5 + 1)
  else Real.pi/4

theorem lateral_angle_theorem (p : RegularQuadPyramid) :
  lateral_angle p = Real.arctan (Real.sqrt 5 + 1) ∨ lateral_angle p = Real.pi/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lateral_angle_theorem_l1291_129149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1291_129130

def geometric_sequence (a₁ : ℕ) (r : ℕ) : ℕ → ℕ
  | 0 => a₁
  | n + 1 => r * geometric_sequence a₁ r n

def sum_geometric_sequence (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  (List.range n).map (geometric_sequence a₁ r) |>.sum

theorem sequence_properties :
  geometric_sequence 1 2 4 = 16 ∧ sum_geometric_sequence 1 2 8 = 255 := by
  sorry

#eval geometric_sequence 1 2 4
#eval sum_geometric_sequence 1 2 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l1291_129130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_is_correct_l1291_129122

/-- Proposition p: For any real number x, ax^2 + ax + 1 > 0 -/
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

/-- Proposition q: The equation x^2 - x + a = 0 has real roots -/
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - x + a = 0

/-- The range of a where at least one of p or q is true -/
def a_range : Set ℝ := {a : ℝ | prop_p a ∨ prop_q a}

theorem a_range_is_correct : a_range = Set.Iio 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_range_is_correct_l1291_129122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_time_calculation_l1291_129128

/-- Calculates the time for a boat to travel upstream given its speed in still water,
    downstream distance and time, and the increase in current speed for the return trip. -/
noncomputable def upstreamTime (boatSpeed : ℝ) (downstreamDistance : ℝ) (downstreamTime : ℝ) (currentSpeedIncrease : ℝ) : ℝ :=
  let downstreamSpeed := downstreamDistance / downstreamTime
  let currentSpeed := downstreamSpeed - boatSpeed
  let newCurrentSpeed := currentSpeed * (1 + currentSpeedIncrease)
  let upstreamSpeed := boatSpeed - newCurrentSpeed
  downstreamDistance / upstreamSpeed

theorem upstream_time_calculation :
  upstreamTime 12 54 3 0.5 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_upstream_time_calculation_l1291_129128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1291_129177

/-- Given a = 2, b = 5^(1/3), and c = (2+e)^(1/e), prove that b < c < a -/
theorem relationship_abc :
  let a : ℝ := 2
  let b : ℝ := Real.rpow 5 (1/3)
  let c : ℝ := Real.rpow (2 + Real.exp 1) (1 / Real.exp 1)
  b < c ∧ c < a := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l1291_129177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_907_l1291_129113

/-- Represents a pentagon with given side lengths -/
structure Pentagon where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ
  side5 : ℝ

/-- Calculates the area of a right triangle -/
noncomputable def rightTriangleArea (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- Calculates the area of a trapezoid -/
noncomputable def trapezoidArea (base1 base2 height : ℝ) : ℝ :=
  ((base1 + base2) / 2) * height

/-- Calculates the area of a pentagon decomposed into a right triangle and a trapezoid -/
noncomputable def pentagonArea (p : Pentagon) : ℝ :=
  rightTriangleArea 17 22 + trapezoidArea 26 22 30

theorem pentagon_area_is_907 (p : Pentagon) 
    (h1 : p.side1 = 17)
    (h2 : p.side2 = 22)
    (h3 : p.side3 = 30)
    (h4 : p.side4 = 26)
    (h5 : p.side5 = 22) : 
  pentagonArea p = 907 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_is_907_l1291_129113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_distance_to_friend_kim_distance_is_30_miles_l1291_129163

/-- The distance in miles that Kim drives to her friend's house -/
def distance_to_friend : ℝ := sorry

/-- Kim's driving speed in miles per hour -/
def driving_speed : ℝ := 44

/-- Total time away from home in hours -/
def total_time : ℝ := 2

/-- Time spent at friend's house in hours -/
def time_at_friends : ℝ := 0.5

/-- Detour factor for the return trip -/
def detour_factor : ℝ := 1.2

theorem kim_distance_to_friend :
  distance_to_friend * (1 + detour_factor) = driving_speed * (total_time - time_at_friends) :=
by sorry

theorem kim_distance_is_30_miles : distance_to_friend = 30 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kim_distance_to_friend_kim_distance_is_30_miles_l1291_129163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_18pi_l1291_129174

/-- Configuration of circles where two smaller circles touch a larger circle and each other at the center of the larger circle -/
structure CircleConfiguration where
  large_radius : ℝ
  small_radius : ℝ
  touch_condition : small_radius = large_radius / 2

/-- The area of the shaded region in the circle configuration -/
noncomputable def shaded_area (config : CircleConfiguration) : ℝ :=
  Real.pi * config.large_radius^2 - 2 * Real.pi * config.small_radius^2

/-- Theorem stating that for a circle configuration with large radius 6, the shaded area is 18π -/
theorem shaded_area_is_18pi (config : CircleConfiguration) 
    (h : config.large_radius = 6) : shaded_area config = 18 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_is_18pi_l1291_129174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1291_129118

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := exp x * (log x - 1)

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∃ a : ℝ, a ∈ Set.Icc (-2) 1 ∧ f (2 - 1/m) ≤ a^2 + 2*a - 3 - exp 1) →
  m ∈ Set.Icc (2/3) 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1291_129118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_factor_numbers_coprime_l1291_129116

/-- A number with equal prime and composite factors -/
def EqualFactorNumber (n : ℕ) : Prop :=
  (Finset.filter (fun d => Nat.Prime d) (Nat.divisors n)).card =
  (Finset.filter (fun d => ¬Nat.Prime d ∧ d ≠ 1) (Nat.divisors n)).card

/-- Theorem: Any two distinct numbers with equal prime and composite factors are coprime -/
theorem equal_factor_numbers_coprime (a b : ℕ) (ha : EqualFactorNumber a) (hb : EqualFactorNumber b) (hab : a ≠ b) :
  Nat.gcd a b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_factor_numbers_coprime_l1291_129116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1291_129172

theorem sqrt_inequality : Real.sqrt 2 - Real.sqrt 6 < Real.sqrt 3 - Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_inequality_l1291_129172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_is_five_l1291_129123

/-- A rectangular prism with edges of length x, y, and z -/
structure RectangularPrism where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The sum of all edge lengths of a rectangular prism -/
def sum_of_edges (p : RectangularPrism) : ℝ :=
  4 * (p.x + p.y + p.z)

/-- The surface area of a rectangular prism -/
def surface_area (p : RectangularPrism) : ℝ :=
  2 * (p.x * p.y + p.x * p.z + p.y * p.z)

/-- The length of the diagonal of a rectangular prism -/
noncomputable def diagonal_length (p : RectangularPrism) : ℝ :=
  Real.sqrt (p.x^2 + p.y^2 + p.z^2)

/-- Theorem: If the sum of all edge lengths is 24 and the surface area is 11,
    then the diagonal length is 5 -/
theorem diagonal_is_five (p : RectangularPrism)
  (h1 : sum_of_edges p = 24)
  (h2 : surface_area p = 11) :
  diagonal_length p = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_is_five_l1291_129123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l1291_129193

theorem integer_pairs_satisfying_equation :
  ∀ a b : ℕ, 
    a ≥ 1 → b ≥ 1 →
    (a : ℝ)^((b : ℝ)^2) = (b : ℝ)^(a : ℝ) → 
    ((a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l1291_129193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_problem_l1291_129132

theorem segment_length_problem (A B P Q : ℝ × ℝ) : 
  (∃ t : ℝ, P = (1 - t) • A + t • B ∧ t = 3/5) →
  (∃ s : ℝ, Q = (1 - s) • A + s • B ∧ s = 5/8) →
  ‖P - Q‖ = 3 →
  ‖B - A‖ = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_problem_l1291_129132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_square_area_l1291_129178

noncomputable def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 + 2 * y^2 - 8 * x + 10 * y = -20

noncomputable def circle_center : ℝ × ℝ := (2, -5/2)

noncomputable def circle_radius : ℝ := 3/2

noncomputable def square_side : ℝ := 6

theorem inscribed_circle_square_area :
  ∀ (x y : ℝ),
  circle_equation x y →
  (∃ (c : ℝ × ℝ) (r : ℝ), c = circle_center ∧ r = circle_radius ∧
    (x - c.1)^2 + (y - c.2)^2 = r^2) →
  square_side = 2 * (2 * circle_radius) →
  square_side^2 = 36 := by
  sorry

#check inscribed_circle_square_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_square_area_l1291_129178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_seven_thirty_thirds_l1291_129121

/-- The probability of selecting 7 distinct integers from {1,2,3,...,12} such that the second smallest number is 4 -/
def probability_second_smallest_is_four : ℚ :=
  let S : Finset ℕ := Finset.range 12
  let n : ℕ := 7
  let favorable_outcomes : ℕ := (Finset.range 3).card * (Finset.range 8 \ Finset.range 5).card
  let total_outcomes : ℕ := Nat.choose S.card n
  ↑favorable_outcomes / ↑total_outcomes

/-- The probability is equal to 7/33 -/
theorem probability_equals_seven_thirty_thirds : 
  probability_second_smallest_is_four = 7 / 33 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_seven_thirty_thirds_l1291_129121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l1291_129191

/-- Represents a tile (bracket) composed of 7 unit squares --/
structure BracketTile where
  squares : Fin 7 → Unit
  indentation : Fin 7

/-- Represents a position on the plane --/
structure Position where
  x : ℤ
  y : ℤ

/-- Represents a tiling of the plane --/
def Tiling := Position → Option BracketTile

/-- Checks if a tiling is valid --/
def is_valid_tiling (t : Tiling) : Prop :=
  ∀ (p : Position), 
    match t p with
    | some b => 
        -- The indentation is filled correctly
        ∃ (p1 p2 : Position), 
          (t p1).isSome ∧ (t p2).isSome ∧ 
          -- Additional conditions for correct filling would be specified here
          True
    | none => 
        -- Every position is covered
        False

/-- Theorem stating that no valid tiling exists --/
theorem no_valid_tiling : ¬∃ (t : Tiling), is_valid_tiling t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_tiling_l1291_129191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_simplify_and_evaluate_l1291_129169

-- Part 1
theorem simplify_expression (a : ℝ) : 3 * (a^2 - 2*a) + (-2*a^2 + 5*a) = a^2 - a := by
  sorry

-- Part 2
theorem simplify_and_evaluate : 
  let x : ℝ := -4;
  let y : ℝ := 1/2;
  -3*x*y^2 - 2*(x*y - 3/2*x^2*y) + (2*x*y^2 - 3*x^2*y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_expression_simplify_and_evaluate_l1291_129169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_and_intersection_l1291_129110

/-- The curve C in the Cartesian plane -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 / 4 = 1}

/-- The line that intersects C -/
def line (k : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = k * p.1 + 1}

/-- The distance function between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem curve_equation_and_intersection (k : ℝ) :
  (∀ p : ℝ × ℝ, p ∈ C ↔ distance p (0, -Real.sqrt 3) + distance p (0, Real.sqrt 3) = 4) ∧
  (∃ A B : ℝ × ℝ, A ∈ C ∧ B ∈ C ∧ A ∈ line k ∧ B ∈ line k ∧ 
    (A.1 * B.1 + A.2 * B.2 = 0 → k = 1/2 ∨ k = -1/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equation_and_intersection_l1291_129110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1291_129156

noncomputable def eddy_distance : ℝ := 600 + 540
noncomputable def eddy_time : ℝ := 3 + 2
noncomputable def freddy_distance : ℝ := 460 + 380
noncomputable def freddy_time : ℝ := 4 + 3

noncomputable def eddy_speed : ℝ := eddy_distance / eddy_time
noncomputable def freddy_speed : ℝ := freddy_distance / freddy_time

theorem speed_ratio : eddy_speed / freddy_speed = 19 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_ratio_l1291_129156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_lambda_sum_l1291_129112

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem stating the relationship between lambda1 and lambda2 for an ellipse -/
theorem ellipse_lambda_sum (e : Ellipse) (F M N P : Point) (lambda1 lambda2 : ℝ) :
  (M.x^2 / e.a^2 + M.y^2 / e.b^2 = 1) →  -- M is on the ellipse
  (N.x^2 / e.a^2 + N.y^2 / e.b^2 = 1) →  -- N is on the ellipse
  (P.x = 0) →  -- P is on the y-axis
  (∃ t : ℝ, M = Point.mk (t * F.x + (1 - t) * P.x) (t * F.y + (1 - t) * P.y)) →  -- M is on line PF
  (∃ s : ℝ, N = Point.mk (s * F.x + (1 - s) * P.x) (s * F.y + (1 - s) * P.y)) →  -- N is on line PF
  (P.x - M.x = lambda1 * (F.x - M.x) ∧ P.y - M.y = lambda1 * (F.y - M.y)) →  -- PM = lambda1 * MF
  (P.x - N.x = lambda2 * (F.x - N.x) ∧ P.y - N.y = lambda2 * (F.y - N.y)) →  -- PN = lambda2 * NF
  lambda1 + lambda2 = -2 * e.a^2 / e.b^2 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_lambda_sum_l1291_129112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l1291_129152

def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry

axiom a_initial : a 1 = 2
axiom b_initial : b 1 = 1

axiom a_recurrence (n : ℕ) : a (n + 1) = 5 * a n + 3 * b n + 7
axiom b_recurrence (n : ℕ) : b (n + 1) = 3 * a n + 5 * b n

theorem a_general_term (n : ℕ) : a n = 2^(3*n - 2) + 2^(n + 1) - 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_general_term_l1291_129152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_expression_comparison_l1291_129157

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 3|

-- Define the solution set
def solution_set : Set ℝ := Set.Iic (-2/3) ∪ Set.Ici 8

-- Theorem for the inequality solution
theorem inequality_solution : 
  {x : ℝ | f x - 5 ≥ x} = solution_set := by sorry

-- Theorem for the comparison of expressions
theorem expression_comparison :
  ∀ m n : ℝ, (∃ x : ℝ, f x = m) → (∃ x : ℝ, f x = n) →
  2 * (m + n) < m * n + 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_expression_comparison_l1291_129157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_eq_4_l1291_129173

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 18

-- Theorem statement
theorem solutions_of_g_eq_4 :
  {x : ℝ | g x = 4} = {-1, 22/3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_of_g_eq_4_l1291_129173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l1291_129147

-- Define the points
def A : ℝ × ℝ := (0, 5)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define A' and B' on the line y = x
def A' : ℝ × ℝ := (15, 15)
def B' : ℝ × ℝ := (5, 5)

-- Define the property that AA' and BB' intersect at C
def intersect_at_C : Prop :=
  ∃ t₁ t₂ : ℝ, 0 < t₁ ∧ 0 < t₂ ∧
    C = (t₁ • A + (1 - t₁) • A') ∧
    C = (t₂ • B + (1 - t₂) • B')

-- Calculate the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem length_of_A'B' :
  line_y_eq_x A' ∧ line_y_eq_x B' ∧ intersect_at_C →
  distance A' B' = 10 * Real.sqrt 2 := by
  sorry

-- Additional lemmas to support the main theorem
lemma A'_on_line : line_y_eq_x A' := by
  simp [line_y_eq_x, A']

lemma B'_on_line : line_y_eq_x B' := by
  simp [line_y_eq_x, B']

lemma C_on_AA' : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = t • A + (1 - t) • A' := by
  sorry

lemma C_on_BB' : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ C = t • B + (1 - t) • B' := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_A_l1291_129147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shares_sale_value_l1291_129190

/-- Proves that selling 3/4 of 2/3 ownership of a 90,000 Rs business yields 45,000 Rs -/
theorem shares_sale_value (business_value : ℚ) (ownership_fraction : ℚ) (sale_fraction : ℚ) : 
  business_value = 90000 →
  ownership_fraction = 2/3 →
  sale_fraction = 3/4 →
  sale_fraction * ownership_fraction * business_value = 45000 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  
-- Note: We remove the #eval line as it's not necessary for the theorem proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shares_sale_value_l1291_129190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_not_perpendicular_l1291_129179

-- Define the types for lines and planes
structure Line : Type
structure Plane : Type

-- Define the relations
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_lines (l1 l2 : Line) : Prop := sorry
def perpendicular_line_plane (l : Line) (p : Plane) : Prop := sorry

-- State the theorem
theorem parallel_not_perpendicular 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : parallel_lines a b) 
  (h3 : ¬ parallel_line_plane b α) : 
  ¬ perpendicular_line_plane b α :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_not_perpendicular_l1291_129179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_cannot_win_l1291_129137

/-- Represents a 3x3 grid of numbers -/
def Grid := Fin 3 → Fin 3 → ℝ

/-- Sum of top and bottom rows -/
def sumTopBottom (g : Grid) : ℝ :=
  (Finset.sum (Finset.range 3) (λ i => g 0 i)) + (Finset.sum (Finset.range 3) (λ i => g 2 i))

/-- Sum of left and right columns -/
def sumLeftRight (g : Grid) : ℝ :=
  (Finset.sum (Finset.range 3) (λ i => g i 0)) + (Finset.sum (Finset.range 3) (λ i => g i 2))

/-- A list of 9 real numbers representing the cards -/
def Cards := List ℝ

theorem player2_cannot_win (cards : Cards) (h : cards.length = 9) :
  ∀ (g : Grid), (∀ i j, g i j ∈ cards.toFinset) →
    sumLeftRight g ≤ sumTopBottom g := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_cannot_win_l1291_129137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_inequality_l1291_129115

noncomputable section

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Axioms
axiom f_domain : ∀ x, x > 0 → f x ≠ 0
axiom f_two : f 2 = 1
axiom f_product : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_pos : ∀ x, x > 1 → f x > 0

-- Theorem
theorem f_monotone_and_inequality :
  (∀ x₁ x₂, 0 < x₁ → 0 < x₂ → x₁ < x₂ → f x₁ < f x₂) ∧
  {x : ℝ | x > 0 ∧ f x + f (x - 2) ≤ 3} = {x : ℝ | 2 < x ∧ x ≤ 4} :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_inequality_l1291_129115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1291_129170

open Real

/-- The function f(x) = x ln x + a/x + 3 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x + a / x + 3

/-- The function g(x) = x³ - x² -/
def g (x : ℝ) : ℝ := x^3 - x^2

/-- The closed interval [1/3, 2] -/
def I : Set ℝ := Set.Icc (1/3 : ℝ) 2

theorem min_value_of_a (a : ℝ) : 
  (∀ x₁ x₂, x₁ ∈ I → x₂ ∈ I → f a x₁ - g x₂ ≥ 0) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_a_l1291_129170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_l1291_129151

-- Define a function f
def f : ℝ → ℝ := sorry

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem point_on_inverse_graph :
  (∃ (f : ℝ → ℝ), f 2 = 6 ∧ (2, 3) ∈ Set.range (λ x => (x, f x / 2))) →
  ∃ (a b : ℝ), (a, b) ∈ Set.range (λ x => (x, f_inv x / 2)) ∧ a + b = 13/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_graph_l1291_129151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_who_announced_6_thought_1_l1291_129141

/-- Represents a circular arrangement of 10 people with their thought numbers -/
structure CircularArrangement where
  numbers : Fin 10 → ℚ

/-- The average of two adjacent numbers in the circular arrangement -/
def average (arr : CircularArrangement) (i : Fin 10) : ℚ :=
  (arr.numbers i + arr.numbers ((i.val + 1) % 10 : Fin 10)) / 2

/-- The theorem statement -/
theorem person_who_announced_6_thought_1 (arr : CircularArrangement) :
  (∀ i : Fin 10, average arr i = (i.val : ℚ) + 1) →
  arr.numbers 5 = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_who_announced_6_thought_1_l1291_129141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_ratio_l1291_129109

-- Define the common perimeter
variable (P : ℝ)
variable (hP : P > 0)

-- Define the side lengths
noncomputable def square_side (P : ℝ) : ℝ := P / 4
noncomputable def hexagon_side (P : ℝ) : ℝ := P / 6

-- Define the radii of the circumscribed circles
noncomputable def square_circle_radius (P : ℝ) : ℝ := (P * Real.sqrt 2) / 8
noncomputable def hexagon_circle_radius (P : ℝ) : ℝ := P / 6

-- Define the areas of the circumscribed circles
noncomputable def A (P : ℝ) : ℝ := Real.pi * (hexagon_circle_radius P) ^ 2
noncomputable def B (P : ℝ) : ℝ := Real.pi * (square_circle_radius P) ^ 2

-- State the theorem
theorem circumscribed_circle_area_ratio (P : ℝ) (hP : P > 0) :
  A P / B P = 8 / 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_ratio_l1291_129109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_sum_correct_max_angle_sum_is_maximum_l1291_129160

/-- The maximum sum of pairwise angles between rays drawn from a point on a plane -/
noncomputable def maxAngleSum (p : ℕ) : ℝ :=
  if p % 2 = 0 then
    (p / 2 : ℝ)^2 * 180
  else
    (p / 2 : ℝ) * ((p / 2 : ℝ) + 1) * 180

/-- Theorem stating the maximum sum of pairwise angles for p rays -/
theorem max_angle_sum_correct (p : ℕ) :
  maxAngleSum p = if p % 2 = 0 then
                    (p / 2 : ℝ)^2 * 180
                  else
                    (p / 2 : ℝ) * ((p / 2 : ℝ) + 1) * 180 :=
by
  sorry

/-- Theorem stating that maxAngleSum gives the maximum possible sum -/
theorem max_angle_sum_is_maximum (p : ℕ) (angleSum : ℝ) 
  (h : angleSum ≤ maxAngleSum p) :
  angleSum ≤ maxAngleSum p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_sum_correct_max_angle_sum_is_maximum_l1291_129160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_l1291_129143

/-- A piecewise function f(x) defined by a parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then x^2 - (2*a - 1)*x + 1
  else (a - 3)*x + a

/-- The theorem stating the range of a for which f is decreasing on ℝ -/
theorem f_decreasing_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Icc (1/2 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_range_l1291_129143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_approx_l1291_129106

/-- A right pyramid with a square base -/
structure SquarePyramid where
  base_area : ℝ
  face_area : ℝ
  total_surface_area : ℝ
  height : ℝ

/-- The volume of a square pyramid -/
noncomputable def volume (p : SquarePyramid) : ℝ := (1/3) * p.base_area * p.height

/-- Theorem: The volume of the specific pyramid is approximately 491.84 cubic units -/
theorem pyramid_volume_approx (p : SquarePyramid) 
  (h1 : p.total_surface_area = 540)
  (h2 : p.face_area = (1/3) * p.base_area)
  (h3 : p.total_surface_area = p.base_area + 4 * p.face_area) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |volume p - 491.84| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_approx_l1291_129106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_genetic_marker_probability_l1291_129144

theorem genetic_marker_probability :
  let total_population : ℕ := 100
  let prob_one_marker : ℚ := 15 / 100
  let prob_two_markers : ℚ := 18 / 100
  let prob_all_given_xy : ℚ := 1 / 4
  let num_all_markers : ℕ := 6
  let num_with_markers : ℕ := 3 * 15 + 3 * 18 - 3 * 6
  let num_no_markers : ℕ := total_population - num_with_markers
  let num_without_x : ℕ := total_population - (15 + 18 + 18 + 6)
  (prob_one_marker * 3 + prob_two_markers * 3 - (num_all_markers : ℚ) * 3 / total_population = (num_with_markers : ℚ) / total_population) →
  (num_no_markers : ℚ) / (num_without_x : ℚ) = 19 / 43 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_genetic_marker_probability_l1291_129144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_range_l1291_129100

noncomputable section

-- Define the function f
def f : ℝ → ℝ := λ x => x^2 + 4*x - 2

-- Define the function g
def g : ℝ → ℝ := λ x => f x / x

theorem quadratic_function_and_range :
  (∃! x, f x = -6) ∧  -- f intersects y = -6 at only one point
  (f 0 = -2) ∧ (f (-4) = -2) ∧  -- f(0) = f(-4) = -2
  (∀ x ∈ Set.Icc 1 2, ∀ t ∈ Set.Icc (-4) 4, g x ≥ -m^2 + t*m) →
  (f = λ x => x^2 + 4*x - 2) ∧
  (m ∈ Set.Ioi 3 ∪ Set.Icc (-1) 1 ∪ Set.Iic (-3)) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_and_range_l1291_129100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ai_equals_nine_l1291_129176

theorem sum_of_ai_equals_nine (a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) : 
  (5 : ℚ) / 7 = a₂ / 2 + a₃ / 6 + a₄ / 24 + a₅ / 120 + a₆ / 720 + a₇ / 5040 →
  (0 ≤ a₂ ∧ a₂ < 2) ∧ 
  (0 ≤ a₃ ∧ a₃ < 3) ∧ 
  (0 ≤ a₄ ∧ a₄ < 4) ∧ 
  (0 ≤ a₅ ∧ a₅ < 5) ∧ 
  (0 ≤ a₆ ∧ a₆ < 6) ∧ 
  (0 ≤ a₇ ∧ a₇ < 7) →
  a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 9 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_ai_equals_nine_l1291_129176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_french_exam_min_words_l1291_129133

/-- Given a total number of vocabulary words and a required minimum score percentage,
    calculate the minimum number of words that must be learned. -/
def min_words_to_learn (total_words : ℕ) (min_score_percent : ℚ) : ℕ :=
  (((min_score_percent / 100) * total_words).ceil).toNat

/-- Theorem: For a French exam with 800 vocabulary words,
    learning at least 720 words ensures a score of at least 90%. -/
theorem french_exam_min_words :
  min_words_to_learn 800 90 = 720 := by
  -- Unfold the definition of min_words_to_learn
  unfold min_words_to_learn
  -- Simplify the expression
  simp [Nat.cast_ofNat, Int.ofNat_eq_cast]
  -- Evaluate the numerical expression
  norm_num
  -- QED
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_french_exam_min_words_l1291_129133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_Y_is_8_l1291_129197

/-- A random variable following a binomial distribution -/
def binomial_rv (n : ℕ) (p : ℝ) : Type :=
  { X : ℝ // ∃ (k : ℕ), X = k ∧ k ≤ n }

/-- The expectation of a binomial random variable -/
noncomputable def expectation (X : binomial_rv 9 (2/3)) : ℝ :=
  9 * (2/3)

/-- The variance of a binomial random variable -/
noncomputable def variance (X : binomial_rv 9 (2/3)) : ℝ :=
  9 * (2/3) * (1 - 2/3)

/-- A linear transformation of the random variable X -/
noncomputable def Y (X : binomial_rv 9 (2/3)) : ℝ :=
  2 * (X.val : ℝ) - 1

/-- The variance of Y -/
noncomputable def variance_Y (X : binomial_rv 9 (2/3)) : ℝ :=
  4 * variance X

theorem variance_Y_is_8 (X : binomial_rv 9 (2/3)) :
  variance_Y X = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_Y_is_8_l1291_129197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1291_129166

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.sqrt (Real.log x + x + m)

noncomputable def curve (x : ℝ) : ℝ := (1 - Real.exp 1) / 2 * Real.cos x + (1 + Real.exp 1) / 2

theorem range_of_m :
  ∀ m : ℝ,
  (∃ x₀ y₀ : ℝ, curve x₀ = y₀ ∧ f m (f m y₀) = y₀) →
  0 ≤ m ∧ m ≤ Real.exp 2 - Real.exp 1 - 1 :=
by
  sorry

#check range_of_m

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1291_129166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_l1291_129192

/-- A rectangular park with a trapezoidal lawn and two flower beds -/
structure Park where
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ
  flower_bed_count : ℕ

/-- Properties of the park -/
def park_properties (p : Park) : Prop :=
  p.trapezoid_short_side = 15 ∧
  p.trapezoid_long_side = 30 ∧
  p.flower_bed_count = 2

/-- Calculate the area of the flower beds -/
noncomputable def flower_bed_area (p : Park) : ℝ :=
  let leg_length := (p.trapezoid_long_side - p.trapezoid_short_side) / 2
  ↑p.flower_bed_count * (leg_length ^ 2 / 2)

/-- Calculate the total area of the park -/
noncomputable def park_area (p : Park) : ℝ :=
  p.trapezoid_long_side * ((p.trapezoid_long_side - p.trapezoid_short_side) / 2)

/-- The main theorem: The fraction of the park occupied by flower beds is 1/4 -/
theorem flower_bed_fraction (p : Park) (h : park_properties p) :
  flower_bed_area p / park_area p = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fraction_l1291_129192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_minus_two_l1291_129185

noncomputable def f (x : ℝ) : ℝ := x + 1/x

theorem product_of_roots_minus_two (a : ℝ) (x₁ x₂ x₃ x₄ : ℝ) : 
  (∃ (g : ℝ → ℝ), g = λ x ↦ (f x)^2 - a * f x + 2 * a) →
  x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ →
  (∃ (g : ℝ → ℝ), g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0 ∧ g x₄ = 0) →
  (2 - f x₁) * (2 - f x₂) * (2 - f x₃) * (2 - f x₄) = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_roots_minus_two_l1291_129185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l1291_129167

/-- Calculates the profit percentage after selling two items with given price relationships and tax. -/
theorem profit_percentage_calculation (S1 : ℝ) (S1_pos : S1 > 0) : 
  let CP1 := 0.81 * S1
  let S2 := 0.9 * S1
  let CP2 := 0.81 * S2
  let TSP := S1 + S2
  let TAR := TSP * 0.95
  let TCP := CP1 + CP2
  let P := TAR - TCP
  let profit_percentage := (P / TCP) * 100
  ∃ (ε : ℝ), abs (profit_percentage - 17.28) < ε ∧ ε > 0 := by
  sorry

#check profit_percentage_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_calculation_l1291_129167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l1291_129120

theorem sum_of_repeating_decimals : 
  ∃ (a b : ℚ), (a = 4/9) ∧ (b = 7/9) ∧ (a + b = 11/9) := by
  use (4/9 : ℚ), (7/9 : ℚ)
  constructor
  · rfl
  constructor
  · rfl
  · norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_repeating_decimals_l1291_129120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_segments_intersect_l1291_129105

/-- A segment on a line represented by its endpoints -/
structure Segment where
  start : ℝ
  finish : ℝ
  h : start ≤ finish

/-- The property that any two segments intersect -/
def anyTwoIntersect (segments : List Segment) : Prop :=
  ∀ i j, i ∈ segments → j ∈ segments → i ≠ j →
    ∃ x, i.start ≤ x ∧ x ≤ i.finish ∧ j.start ≤ x ∧ x ≤ j.finish

/-- The theorem stating that if any two segments intersect, then all segments have a common point -/
theorem all_segments_intersect (segments : List Segment) 
    (h : segments.length = 2019) 
    (h_intersect : anyTwoIntersect segments) : 
    ∃ x, ∀ s ∈ segments, s.start ≤ x ∧ x ≤ s.finish := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_segments_intersect_l1291_129105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1291_129102

/-- Represents the speed of a boat in various conditions -/
structure BoatSpeed where
  alongStream : ℚ
  againstStream : ℚ

/-- Calculates the speed of a boat in still water given its speeds along and against a stream -/
def stillWaterSpeed (speeds : BoatSpeed) : ℚ :=
  (speeds.alongStream + speeds.againstStream) / 2

/-- Theorem stating that a boat with given speeds along and against a stream has a specific speed in still water -/
theorem boat_speed_in_still_water (speeds : BoatSpeed) 
  (h1 : speeds.alongStream = 11) 
  (h2 : speeds.againstStream = 5) : 
  stillWaterSpeed speeds = 8 := by
  -- Unfold the definition of stillWaterSpeed
  unfold stillWaterSpeed
  -- Rewrite using the hypotheses
  rw [h1, h2]
  -- Simplify the arithmetic
  norm_num

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_speed_in_still_water_l1291_129102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polyhedron_ratio_l1291_129117

/-- A polyhedron with specific properties -/
structure SpecialPolyhedron where
  faces : Nat
  edges : Nat
  vertices : Nat
  x : ℝ
  y : ℝ

/-- Properties of the special polyhedron -/
def is_special_polyhedron (p : SpecialPolyhedron) : Prop :=
  p.faces = 12 ∧
  p.edges = 18 ∧
  p.vertices = 8 ∧
  (∀ f, f ≤ p.faces → True) ∧ -- placeholder for isosceles triangle property
  (∀ e, e ≤ p.edges → True) ∧ -- placeholder for edge length property
  (∀ v, v ≤ p.vertices → True) ∧ -- placeholder for vertex degree property
  (∀ a b, a ≤ p.faces → b ≤ p.faces → True) -- placeholder for dihedral angle property

/-- The main theorem -/
theorem special_polyhedron_ratio (p : SpecialPolyhedron) 
  (h : is_special_polyhedron p) : p.x / p.y = 3 / 5 := by
  sorry

#check special_polyhedron_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polyhedron_ratio_l1291_129117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_halving_matrix_exists_zero_matrix_is_answer_l1291_129148

theorem no_halving_matrix_exists : ¬∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    N * A = Matrix.of (fun i j => 
      if j = 0 then A i j else (1 / 2 : ℝ) * A i j) :=
sorry

theorem zero_matrix_is_answer (N : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
    N * A = Matrix.of (fun i j => 
      if j = 0 then A i j else (1 / 2 : ℝ) * A i j)) →
  N = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_halving_matrix_exists_zero_matrix_is_answer_l1291_129148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1291_129135

/-- The distance between two parallel lines in 2D space -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the lines 2x+y+1=0 and 4x+2y-1=0 is 3√5/10 -/
theorem distance_between_given_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y ↦ 2*x + y + 1
  let l₂ : ℝ → ℝ → ℝ := λ x y ↦ 4*x + 2*y - 1
  distance_parallel_lines 4 2 2 (-1) = (3 * Real.sqrt 5) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1291_129135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l1291_129199

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (x - Real.pi/3) + 2 * Real.cos x

theorem phase_shift_of_f :
  ∃ (shift : ℝ), shift = Real.pi/3 ∧
  ∀ (x : ℝ), f x = 5 * Real.sin (x - shift) + 2 * Real.cos x := by
  use Real.pi/3
  constructor
  · rfl
  · intro x
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_phase_shift_of_f_l1291_129199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_l1291_129198

-- Define the curve C
noncomputable def C (θ : ℝ) : ℝ × ℝ := (-2 + Real.cos θ, Real.sin θ)

-- Define the ratio y/x for a point on the curve
noncomputable def ratio (θ : ℝ) : ℝ := (C θ).2 / (C θ).1

-- Theorem statement
theorem ratio_range :
  ∀ θ : ℝ, -Real.sqrt 3 / 3 ≤ ratio θ ∧ ratio θ ≤ Real.sqrt 3 / 3 := by
  sorry

#check ratio_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_range_l1291_129198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_in_hours_l1291_129101

/-- The number of chapters in the textbook -/
def total_chapters : Nat := 31

/-- The time in minutes it takes to read one chapter -/
def minutes_per_chapter : Nat := 20

/-- Predicate to determine if a chapter should be read -/
def should_read (chapter : Nat) : Bool :=
  ¬(chapter % 3 = 0)

/-- The total number of chapters that should be read -/
def chapters_to_read : Nat :=
  (List.range total_chapters).filter (fun n => should_read (n + 1)) |>.length

/-- The total reading time in minutes -/
def total_reading_time : Nat :=
  chapters_to_read * minutes_per_chapter

/-- Theorem stating the total reading time in hours -/
theorem reading_time_in_hours :
  total_reading_time / 60 = 7 := by
  -- Unfold definitions
  unfold total_reading_time
  unfold chapters_to_read
  -- Evaluate the expression
  simp [should_read, minutes_per_chapter, total_chapters]
  -- The proof is complete
  rfl

#eval total_reading_time / 60

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_time_in_hours_l1291_129101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_in_partitioned_triangle_l1291_129125

/-- Represents a triangle with two lines drawn from one vertex to the midpoints of opposite sides -/
structure PartitionedTriangle where
  total_area : ℝ
  small_triangle1_area : ℝ
  small_triangle2_area : ℝ

/-- The area of the quadrilateral formed in a partitioned triangle -/
noncomputable def quadrilateral_area (t : PartitionedTriangle) : ℝ :=
  (t.total_area - t.small_triangle1_area - t.small_triangle2_area) / 2

theorem quadrilateral_area_in_partitioned_triangle (t : PartitionedTriangle) 
  (h1 : t.small_triangle1_area = 6)
  (h2 : t.small_triangle2_area = 10) :
  quadrilateral_area t = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_in_partitioned_triangle_l1291_129125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_platform_l1291_129119

-- Define the train's length
noncomputable def train_length : ℝ := 240

-- Define the time to pass a pole
noncomputable def time_to_pass_pole : ℝ := 24

-- Define the platform's length
noncomputable def platform_length : ℝ := 650

-- Define the train's speed
noncomputable def train_speed : ℝ := train_length / time_to_pass_pole

-- Define the total distance (train length + platform length)
noncomputable def total_distance : ℝ := train_length + platform_length

-- Theorem: The time to pass the platform is 89 seconds
theorem time_to_pass_platform : 
  (total_distance / train_speed) = 89 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_time_to_pass_platform_l1291_129119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_point_l1291_129181

-- Define the piecewise function
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 1 + Real.log x / Real.log 5
  else 2 * x - 1

-- Define what it means for a function to have a zero point
def hasZeroPoint (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = 0

-- Theorem statement
theorem f_has_one_zero_point :
  ∃! x, hasZeroPoint f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_one_zero_point_l1291_129181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l1291_129126

def vector_a : Fin 2 → ℝ := ![2, 1]
def vector_b : Fin 2 → ℝ := ![3, 4]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0) ^ 2 + (v 1) ^ 2)

noncomputable def projection (v w : Fin 2 → ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

theorem projection_of_a_on_b :
  projection vector_a vector_b = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_of_a_on_b_l1291_129126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_representation_l1291_129187

/-- Triangle DEF with side lengths and incenter -/
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  J : ℝ × ℝ

/-- The triangle satisfies the given conditions -/
def validTriangle (t : Triangle) : Prop :=
  t.d = 8 ∧ t.e = 10 ∧ t.f = 6

/-- Vector representation of the incenter -/
def incenterVector (t : Triangle) (p q r : ℝ) : Prop :=
  t.J = (p * t.D.1 + q * t.E.1 + r * t.F.1, p * t.D.2 + q * t.E.2 + r * t.F.2)

/-- The main theorem -/
theorem incenter_representation (t : Triangle) 
  (h : validTriangle t) : 
  incenterVector t (1/3) (5/12) (1/4) ∧ 
  (1/3 : ℝ) + (5/12 : ℝ) + (1/4 : ℝ) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incenter_representation_l1291_129187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l1291_129138

/-- Represents the average annual growth rate of per capita disposable income -/
def x : ℝ := sorry

/-- Initial per capita disposable income in 2020 (in thousands of yuan) -/
def initial_income : ℝ := 3.2

/-- Final per capita disposable income in 2022 (in thousands of yuan) -/
def final_income : ℝ := 3.7

/-- Number of years between 2020 and 2022 -/
def num_years : ℕ := 2

/-- Theorem stating that the equation correctly represents the relationship
    between initial income, final income, and average annual growth rate -/
theorem income_growth_equation :
  initial_income * (1 + x)^num_years = final_income :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_income_growth_equation_l1291_129138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_l1291_129140

/-- Represents the speed of the boat in still water -/
def boat_speed : ℝ := 6

/-- Represents the total distance traveled upstream and downstream -/
def total_distance : ℝ := 64

/-- Represents the current speed in the first section of the river -/
def current_speed1 : ℝ := 2

/-- Represents the current speed in the second section of the river -/
def current_speed2 : ℝ := 3

/-- Represents the current speed in the third section of the river -/
def current_speed3 : ℝ := 1.5

/-- Represents the length of the first section of the river -/
def section1_length : ℝ := 20

/-- Represents the length of the second section of the river -/
def section2_length : ℝ := 24

/-- Represents the length of the third section of the river -/
def section3_length : ℝ := 20

/-- Represents the number of stops during the upstream journey -/
def upstream_stops : ℕ := 3

/-- Represents the number of stops during the downstream journey -/
def downstream_stops : ℕ := 4

/-- Represents the duration of each stop in hours -/
def stop_duration : ℝ := 0.5

/-- Calculates the total journey time -/
noncomputable def total_journey_time : ℝ :=
  let upstream_time1 := section1_length / (boat_speed - current_speed1)
  let upstream_time2 := section2_length / (boat_speed - current_speed2)
  let upstream_time3 := section3_length / (boat_speed - current_speed3)
  let downstream_time1 := section1_length / (boat_speed + current_speed1)
  let downstream_time2 := section2_length / (boat_speed + current_speed2)
  let downstream_time3 := section3_length / (boat_speed + current_speed3)
  let total_stop_time := (upstream_stops + downstream_stops : ℝ) * stop_duration
  upstream_time1 + upstream_time2 + upstream_time3 +
  downstream_time1 + downstream_time2 + downstream_time3 +
  total_stop_time

/-- Theorem stating that the total journey time is approximately 28.78 hours -/
theorem journey_time_approx :
  |total_journey_time - 28.78| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_time_approx_l1291_129140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_RVWX_l1291_129164

-- Define the prism
structure Prism :=
  (height : ℝ)
  (base_side : ℝ)

-- Define the solid RVWX
structure SolidRVWX :=
  (prism : Prism)

-- Define the surface area function
noncomputable def surface_area (s : SolidRVWX) : ℝ :=
  50 + 12.5 * Real.sqrt 3

-- Theorem statement
theorem surface_area_of_RVWX (p : Prism) (s : SolidRVWX) 
  (h1 : p.height = 20)
  (h2 : p.base_side = 10)
  (h3 : s.prism = p) :
  surface_area s = 50 + 12.5 * Real.sqrt 3 := by
  sorry

#check surface_area_of_RVWX

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_RVWX_l1291_129164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_R_12345_units_digit_l1291_129129

noncomputable def a : ℝ := 3 + 2 * Real.sqrt 2
noncomputable def b : ℝ := 3 - 2 * Real.sqrt 2

noncomputable def R (n : ℕ) : ℝ := (1 / 2) * (a ^ n + b ^ n)

theorem R_12345_units_digit (h : ∃ k : ℤ, R 12345 = k) :
  ∃ m : ℕ, R 12345 = 10 * m + 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_R_12345_units_digit_l1291_129129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l1291_129127

-- Define the quadrilateral ABCD
variable (A B C D : ℝ × ℝ)

-- Define E as the midpoint of AB
noncomputable def E (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define F as the midpoint of CD
noncomputable def F (C D : ℝ × ℝ) : ℝ × ℝ := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define scalar multiplication
def vec_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Theorem statement
theorem midpoint_theorem (A B C D : ℝ × ℝ) : 
  vec_mul 2 (vec_sub (F C D) (E A B)) = 
  vec_add (vec_sub D A) (vec_sub C B) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_theorem_l1291_129127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_distances_after_one_hour_l1291_129150

/-- Represents the possible relative positions of two cars -/
inductive RelativePosition
  | TowardsEachOther
  | AwayFromEachOther
  | SameDirectionSlowerAhead
  | SameDirectionFasterAhead

/-- Calculates the distance between two cars after a given time -/
def distanceAfterTime (initialDistance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) (position : RelativePosition) : ℝ :=
  match position with
  | RelativePosition.TowardsEachOther => initialDistance - (speed1 + speed2) * time
  | RelativePosition.AwayFromEachOther => initialDistance + (speed1 + speed2) * time
  | RelativePosition.SameDirectionSlowerAhead => initialDistance - (speed2 - speed1) * time
  | RelativePosition.SameDirectionFasterAhead => initialDistance + (speed2 - speed1) * time

/-- Theorem stating the possible distances between two cars after 1 hour -/
theorem possible_distances_after_one_hour (initialDistance : ℝ) (speed1 : ℝ) (speed2 : ℝ) 
    (h1 : initialDistance = 200)
    (h2 : speed1 = 60)
    (h3 : speed2 = 80) :
    ∃ (d : ℝ), d ∈ ({60, 180, 220, 340} : Set ℝ) ∧
    ∃ (position : RelativePosition), distanceAfterTime initialDistance speed1 speed2 1 position = d := by
  sorry

#check possible_distances_after_one_hour

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_distances_after_one_hour_l1291_129150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_equals_one_l1291_129108

theorem tan_a_equals_one (a : Real) 
  (h1 : Real.sin a + Real.cos a = Real.sqrt 2) 
  (h2 : a ∈ Set.Ioo (-Real.pi/2) (Real.pi/2)) : 
  Real.tan a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_a_equals_one_l1291_129108
