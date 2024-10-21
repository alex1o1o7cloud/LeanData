import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_slope_l339_33962

noncomputable section

/-- Definition of the ellipse E -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

/-- Definition of the line l passing through the left focus -/
def line_through_focus (c k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + c)

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- Theorem for part (1) -/
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : eccentricity a b = Real.sqrt 2 / 2) 
    (h4 : ∃ x1 y1 x2 y2, ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧ 
         line_through_focus c 1 x1 y1 ∧ line_through_focus c 1 x2 y2 ∧
         distance x1 y1 x2 y2 = 8/3) :
  a = 2 ∧ b = Real.sqrt 2 := by sorry

/-- Theorem for part (2) -/
theorem line_slope (a b c k : ℝ) (h1 : a > b) (h2 : b > 0)
    (h3 : eccentricity a b = Real.sqrt 2 / 2)
    (h4 : ∃ x1 y1 x2 y2, ellipse a b x1 y1 ∧ ellipse a b x2 y2 ∧ 
         line_through_focus c k x1 y1 ∧ line_through_focus c k x2 y2)
    (h5 : ∃ AF1 AF2 BF1 BF2, AF2 / AF1 = 5 ∧ BF2 / BF1 = 1/2)
    (h6 : k < 0) :
  k = - Real.sqrt 14 / 6 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_slope_l339_33962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_comparison_l339_33968

theorem square_comparison (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a * b / (a + b)) > (a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_comparison_l339_33968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_expression_l339_33936

noncomputable section

theorem parallel_vectors_expression (θ : ℝ) : 
  let a : ℝ × ℝ := (Real.cos θ, Real.sin θ)
  let b : ℝ × ℝ := (1, -2)
  (∃ (k : ℝ), a = k • b) → (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_expression_l339_33936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_divisible_by_all_l339_33951

theorem least_number_divisible_by_all (n : ℕ) : n = 861 ↔ 
  (∀ m : ℕ, m < n → ¬(∀ d : ℕ, d ∈ ({24, 32, 36, 54} : Set ℕ) → (m + 3) % d = 0)) ∧
  (∀ d : ℕ, d ∈ ({24, 32, 36, 54} : Set ℕ) → (n + 3) % d = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_number_divisible_by_all_l339_33951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_left_branch_intersection_slope_range_l339_33956

/-- Represents a hyperbola with equation x^2/16 - y^2/9 = 1 -/
structure Hyperbola where
  a : ℝ := 4
  b : ℝ := 3

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope k passing through point p -/
structure Line where
  k : ℝ
  p : Point

/-- Checks if a point is on the left branch of the hyperbola -/
def isOnLeftBranch (h : Hyperbola) (p : Point) : Prop :=
  p.x < 0 ∧ p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The left focus of the hyperbola -/
noncomputable def leftFocus (h : Hyperbola) : Point :=
  { x := -Real.sqrt (h.a^2 + h.b^2), y := 0 }

/-- Theorem stating the range of slopes for lines intersecting the left branch -/
theorem slope_range_for_left_branch_intersection (h : Hyperbola) (l : Line) :
  (l.p = leftFocus h) →
  (∃ A B : Point, isOnLeftBranch h A ∧ isOnLeftBranch h B ∧ 
    A ≠ B ∧ A.y - l.p.y = l.k * (A.x - l.p.x) ∧ B.y - l.p.y = l.k * (B.x - l.p.x)) →
  l.k < -3/4 ∨ l.k > 3/4 := by
  sorry

/-- The main theorem stating the range of slopes -/
theorem slope_range (h : Hyperbola) :
  {k : ℝ | ∃ l : Line, l.k = k ∧ l.p = leftFocus h ∧
    ∃ A B : Point, isOnLeftBranch h A ∧ isOnLeftBranch h B ∧ 
    A ≠ B ∧ A.y - l.p.y = l.k * (A.x - l.p.x) ∧ B.y - l.p.y = l.k * (B.x - l.p.x)} =
  {k : ℝ | k < -3/4 ∨ k > 3/4} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_left_branch_intersection_slope_range_l339_33956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_member_in_four_committees_l339_33949

/-- A club is represented by a set of committees -/
structure Club where
  committees : Finset (Finset Nat)

/-- Properties of the club -/
def ValidClub (c : Club) : Prop :=
  (c.committees.card = 11) ∧
  (∀ comm ∈ c.committees, comm.card = 5) ∧
  (∀ comm1 comm2, comm1 ∈ c.committees → comm2 ∈ c.committees → comm1 ≠ comm2 → (comm1 ∩ comm2).Nonempty)

/-- A member belongs to at least 4 committees -/
def MemberInFourCommittees (c : Club) : Prop :=
  ∃ m : Nat, (c.committees.filter (λ comm => m ∈ comm)).card ≥ 4

/-- The main theorem -/
theorem member_in_four_committees (c : Club) (h : ValidClub c) : MemberInFourCommittees c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_member_in_four_committees_l339_33949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l339_33925

/-- Calculates the profit percentage for a dealer's transaction -/
theorem dealer_profit_percentage 
  (purchase_quantity : ℕ) 
  (purchase_price : ℚ) 
  (sale_quantity : ℕ) 
  (sale_price : ℚ) 
  (h1 : purchase_quantity = 15)
  (h2 : purchase_price = 25)
  (h3 : sale_quantity = 12)
  (h4 : sale_price = 33)
  : (((sale_price / sale_quantity) - (purchase_price / purchase_quantity)) / (purchase_price / purchase_quantity)) * 100 = 65 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dealer_profit_percentage_l339_33925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l339_33985

/-- A hyperbola with asymptotes y = ±2x passing through (1, 3) has eccentricity √5/2 -/
theorem hyperbola_eccentricity (C : Set (ℝ × ℝ)) : 
  (∀ (x y : ℝ), (x, y) ∈ C → (y = 2*x ∨ y = -2*x) → False) → -- C is not its asymptotes
  ((1, 3) ∈ C) → -- C passes through (1, 3)
  (∀ ε > 0, ∃ (x y : ℝ), (x, y) ∈ C ∧ (|y - 2*x| < ε ∨ |y + 2*x| < ε)) → -- asymptotes are y = ±2x
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C ↔ (x^2 / a^2) - (y^2 / b^2) = 1) → -- standard form of hyperbola
  (∃ (e : ℝ), e = Real.sqrt 5 / 2 ∧ 
    ∀ (x y : ℝ), (x, y) ∈ C → x^2 / (1 - e^2) - y^2 / ((1 - e^2) * e^2) = 1) -- eccentricity definition
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l339_33985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l339_33972

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (3 - 5 * x)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  deriv f x = -5 * Real.cos (3 - 5 * x) := by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l339_33972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_one_l339_33986

/-- The function f(x) defined as x - a * ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 1 - a / x

theorem tangent_line_implies_a_equals_one (a : ℝ) :
  (∀ x : ℝ, f_derivative a x = deriv (f a) x) →
  f_derivative a 1 = 0 →
  a = 1 := by
  sorry

#check tangent_line_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_one_l339_33986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l339_33923

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the area function for a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the properties of the triangle
def satisfies_conditions (t : Triangle) : Prop :=
  -- Condition 1: The internal angle bisector CD has equation x + y = 0
  ∃ (k : ℝ), t.C = (k, -k) ∧
  -- Condition 2: Vertex A has coordinates (2, 1)
  t.A = (2, 1) ∧
  -- Condition 3: The median BE on side AC has equation 5x - 2y + 10 = 0
  ∃ (x y : ℝ), 5*x - 2*y + 10 = 0 ∧
                x = (t.A.1 + t.C.1) / 2 ∧
                y = (t.A.2 + t.C.2) / 2

-- The main theorem
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.C = (-4, 4) ∧ area t = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l339_33923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_l339_33963

-- Define the parallelogram EFGH
structure Parallelogram :=
  (E F G H : ℝ × ℝ)

-- Define points Q and R
noncomputable def Q (EFGH : Parallelogram) : ℝ × ℝ := sorry
noncomputable def R (EFGH : Parallelogram) : ℝ × ℝ := sorry

-- Define point S as the intersection of EG and QR
noncomputable def S (EFGH : Parallelogram) : ℝ × ℝ := sorry

-- Define the ratio of EQ to EF
def ratio_EQ_EF (EFGH : Parallelogram) : ℚ := 19 / 1000

-- Define the ratio of ER to EH
def ratio_ER_EH (EFGH : Parallelogram) : ℚ := 19 / 2051

-- Define a distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem parallelogram_ratio (EFGH : Parallelogram) :
  ratio_EQ_EF EFGH = 19 / 1000 →
  ratio_ER_EH EFGH = 19 / 2051 →
  (distance EFGH.E EFGH.G) / (distance EFGH.E (S EFGH)) = 3051 / 19 := by
  sorry

#check parallelogram_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_ratio_l339_33963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l339_33966

/-- Represents an ellipse with equation x²/(10-m) + y²/(m-2) = 1 -/
structure Ellipse (m : ℝ) where
  equation : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1

/-- Focal length of an ellipse -/
noncomputable def focal_length (m : ℝ) (e : Ellipse m) : ℝ := 2 * Real.sqrt ((10 - m) - (m - 2))

theorem ellipse_m_value (m : ℝ) (e : Ellipse m) 
  (h1 : 10 - m > m - 2) 
  (h2 : m - 2 > 0) 
  (h3 : focal_length m e = 4) : 
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_m_value_l339_33966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_nonexistence_l339_33905

noncomputable def x_seq (α β r : ℝ) : ℕ → ℝ
  | 0 => r
  | n + 1 => (x_seq α β r n + α) / (β * x_seq α β r n + 1)

noncomputable def M (α β : ℝ) : Set ℝ :=
  if α * β = 1 then
    {-1 / β}
  else
    let lambda := (1 - Real.sqrt (α * β)) / (1 + Real.sqrt (α * β))
    let r (n : ℕ) : ℝ :=
      if α > 0 ∧ β > 0 then
        -Real.sqrt (α / β) * (1 + lambda^(n + 1)) / (1 - lambda^(n + 1))
      else
        Real.sqrt (α / β) * (1 + lambda^(n + 1)) / (1 - lambda^(n + 1))
    {r n | n : ℕ}

theorem sequence_nonexistence (α β : ℝ) (h : α * β > 0) :
  {r : ℝ | ¬∃ (seq : ℕ → ℝ), seq = x_seq α β r} = M α β := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_nonexistence_l339_33905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l339_33954

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * y = -3 * x + 6

-- Define the slope-intercept form of a line
def slope_intercept_form (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Theorem stating that the slope of the given line is -3/2
theorem slope_of_line :
  ∃ (m b : ℝ), m = -3/2 ∧ b = 3 ∧
  ∀ (x y : ℝ), line_equation x y ↔ slope_intercept_form m b x y :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_of_line_l339_33954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_expr_at_two_l339_33913

-- Define the original expression
noncomputable def original_expr (x : ℝ) : ℝ := (x + 2) / (x - 2)

-- Define the transformed expression
noncomputable def transformed_expr (x : ℝ) : ℝ := 
  (original_expr x + 2) / (original_expr x - 2)

-- Theorem statement
theorem transformed_expr_at_two : 
  transformed_expr 2 = -2 := by
  -- Expand the definition of transformed_expr
  unfold transformed_expr
  -- Expand the definition of original_expr
  unfold original_expr
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_expr_at_two_l339_33913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_top_three_teams_l339_33902

/-- Represents a team in the tournament -/
structure Team where
  id : Nat
  deriving Repr, DecidableEq

/-- Represents the result of a match -/
inductive MatchResult
  | Win
  | Draw
  | Loss

/-- Returns the points earned for a given match result -/
def pointsForResult (result : MatchResult) : Nat :=
  match result with
  | MatchResult.Win => 3
  | MatchResult.Draw => 1
  | MatchResult.Loss => 0

/-- Represents the tournament -/
structure Tournament where
  teams : Finset Team
  numTeams : Nat
  results : Team → Team → MatchResult
  numMatchesPerPair : Nat

/-- Calculate the score of a team in the tournament -/
def teamScore (t : Tournament) (team : Team) : Nat :=
  (t.teams.sum fun opponent => 
    if opponent ≠ team then
      t.numMatchesPerPair * pointsForResult (t.results team opponent)
    else
      0)

/-- The theorem to be proved -/
theorem max_score_top_three_teams (t : Tournament) :
  t.numTeams = 6 →
  t.numMatchesPerPair = 2 →
  ∃ (team1 team2 team3 : Team),
    team1 ∈ t.teams ∧ team2 ∈ t.teams ∧ team3 ∈ t.teams ∧
    team1 ≠ team2 ∧ team1 ≠ team3 ∧ team2 ≠ team3 ∧
    teamScore t team1 = teamScore t team2 ∧
    teamScore t team2 = teamScore t team3 ∧
    teamScore t team1 ≤ 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_score_top_three_teams_l339_33902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l339_33950

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- Check if two vectors are parallel -/
def IsParallel (p q : Real × Real) : Prop :=
  ∃ k : Real, p.1 * q.2 = k * p.2 * q.1

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (p q : Real × Real) : 
  p = (2 * t.b - t.c, Real.cos t.C) → 
  q = (2 * t.a, 1) → 
  IsParallel p q → 
  Real.sin t.A = Real.sqrt 3 / 2 ∧ 
  Set.Icc (-1 : Real) (Real.sqrt 2) = { x | ∃ C, (-2 * Real.cos (2 * C)) / (1 + Real.tan C) + 1 = x } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l339_33950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l339_33939

-- Define a proposition type
variable (P : Prop)

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define parallel lines
def parallel (a b : ℝ) : Prop :=
  ∃ k : ℝ, ∀ x y : ℝ, a * x + y - 1 = k * (x + b * y - 1)

theorem correct_statements :
  -- Statement 1
  (∀ P, (¬P → P) → (P → ¬P)) ∧
  -- Statement 2
  (∀ t : Triangle, t.B = 60 ↔ ∃ d : ℝ, t.A = t.B - d ∧ t.C = t.B + d) ∧
  -- Statement 3
  (¬∀ a b : ℝ, (a * b = 1 → parallel a b) ∧ ¬(parallel a b → a * b = 1)) ∧
  -- Statement 4
  (¬∀ a b : ℝ, (∀ m : ℝ, a * m^2 < b * m^2) ↔ a < b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l339_33939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l339_33957

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (Real.log (abs x) / Real.log 10 - 3)

-- State the theorem
theorem f_properties :
  (∀ y ∈ Set.range f, y ≥ 0) ∧ 
  (∀ x, f x = f (-x)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l339_33957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_tetrahedron_l339_33930

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the distance between two points in 3D space -/
noncomputable def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Theorem: Minimum sum of distances in a tetrahedron -/
theorem min_sum_distances_tetrahedron (ABCD : Tetrahedron) 
  (hAC : distance ABCD.A ABCD.C = 8)
  (hAB : distance ABCD.A ABCD.B = 7)
  (hCD : distance ABCD.C ABCD.D = 7)
  (hBC : distance ABCD.B ABCD.C = 5)
  (hAD : distance ABCD.A ABCD.D = 5)
  (hBD : distance ABCD.B ABCD.D = 6) :
  ∃ (min_val : ℝ), min_val = 2 * Real.sqrt 21 ∧
  ∀ (P : Point3D), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 8 ∧ 
    P.x = ABCD.A.x + t * (ABCD.C.x - ABCD.A.x) ∧
    P.y = ABCD.A.y + t * (ABCD.C.y - ABCD.A.y) ∧
    P.z = ABCD.A.z + t * (ABCD.C.z - ABCD.A.z)) →
  distance ABCD.B P + distance P ABCD.D ≥ min_val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_distances_tetrahedron_l339_33930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_roof_leak_ratio_l339_33910

/-- Proves that the ratio of the medium-sized hole's leak rate to the largest hole's leak rate is 3:2 --/
theorem garage_roof_leak_ratio : 
  ∀ (medium_rate : ℚ),
  medium_rate > 0 →
  3 + medium_rate + (medium_rate / 3) = 5 →
  medium_rate / 3 = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_garage_roof_leak_ratio_l339_33910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_to_canonical_form_l339_33916

/-- The original quadratic equation -/
def original_equation (x y : ℝ) : Prop :=
  8 * x^2 + 4 * x * y + 5 * y^2 - 56 * x - 32 * y + 80 = 0

/-- The canonical form of an ellipse -/
def canonical_form (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 9 = 1

/-- Transformation from original coordinates to rotated coordinates -/
def rotate (x y x1 y1 α : ℝ) : Prop :=
  x = x1 * Real.cos α - y1 * Real.sin α ∧
  y = x1 * Real.sin α + y1 * Real.cos α

/-- Translation of coordinates -/
def translate (x1 y1 x2 y2 a b : ℝ) : Prop :=
  x2 = x1 - a ∧
  y2 = y1 - b

/-- Theorem stating that the original equation can be transformed into canonical form -/
theorem equation_to_canonical_form :
  ∃ (α a b : ℝ), ∀ (x y x1 y1 x2 y2 : ℝ),
    rotate x y x1 y1 α →
    translate x1 y1 x2 y2 a b →
    (original_equation x y ↔ canonical_form x2 y2) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_to_canonical_form_l339_33916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l339_33976

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}

theorem complement_of_A_in_U : (U \ A) = {1, 5} := by
  apply Set.ext
  intro x
  simp [U, A]
  sorry

#check complement_of_A_in_U

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_in_U_l339_33976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_and_intersection_product_l339_33993

open Real

-- Define the curves C1 and C2
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (cos α, sin α)

noncomputable def C2_polar (θ : ℝ) : ℝ := -2 * sin θ

-- State the theorem
theorem curve_equations_and_intersection_product :
  -- Part 1: Equations of C1 and C2
  (∀ θ, 0 ≤ θ ∧ θ ≤ π → (∃ α, C1 α = (cos θ, sin θ))) ∧
  (∀ x y, x^2 + (y + 1)^2 = 1 ↔ ∃ θ, x = C2_polar θ * cos θ ∧ y = C2_polar θ * sin θ) ∧
  -- Part 2: Range of |PM| · |PN|
  (∀ P : ℝ × ℝ, (∃ α, C1 α = P) →
    ∀ M N : ℝ × ℝ, (∃ θ₁ θ₂, (C2_polar θ₁ * cos θ₁, C2_polar θ₁ * sin θ₁) = M ∧ 
                              (C2_polar θ₂ * cos θ₂, C2_polar θ₂ * sin θ₂) = N) →
    (∃ t : ℝ, M = P + t • (N - P)) →
    1 ≤ ‖M - P‖ * ‖N - P‖ ∧ ‖M - P‖ * ‖N - P‖ ≤ 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_equations_and_intersection_product_l339_33993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_10_l339_33914

theorem count_divisible_by_10 : 
  (Finset.filter (fun n => n % 10 = 0) (Finset.range 401)).card = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisible_by_10_l339_33914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_formula_l339_33922

/-- The side length of a rhombus with given area and diagonal ratio -/
noncomputable def rhombus_side_length (S m n : ℝ) : ℝ :=
  Real.sqrt (S * (m^2 + n^2) / (2 * m * n))

/-- Theorem: The side length of a rhombus with area S and diagonal ratio m:n
    is √(S(m² + n²) / (2mn)) -/
theorem rhombus_side_length_formula (S m n : ℝ) (h1 : S > 0) (h2 : m > 0) (h3 : n > 0) :
  ∃ (side : ℝ), side > 0 ∧
  side = rhombus_side_length S m n ∧
  side^2 * 2 * m * n = S * (m^2 + n^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_formula_l339_33922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l339_33998

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + 2*a^x - 1

-- State the theorem
theorem max_value_implies_a (a : ℝ) :
  a > 0 →
  a ≠ 1 →
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 14) →
  (∃ x ∈ Set.Icc (-1) 1, f a x = 14) →
  (a = 1/3 ∨ a = 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_implies_a_l339_33998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l339_33931

open Real

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (((1 + x^(2/3))^4)^(1/5)) / (x^2 * x^(1/5))

-- Define the antiderivative
noncomputable def F (x : ℝ) : ℝ := -(5/6) * (((1 + x^(2/3)) / x^(2/3))^(1/5))^9

-- State the theorem
theorem indefinite_integral_correct (x : ℝ) (hx : x > 0) : 
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_correct_l339_33931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_is_18_sqrt_5_l339_33990

/-- The length of a string wrapping around a circular cylindrical post -/
noncomputable def string_length (post_circumference : ℝ) (post_height : ℝ) (num_wraps : ℕ) : ℝ :=
  num_wraps * Real.sqrt ((post_height / num_wraps) ^ 2 + post_circumference ^ 2)

/-- Theorem: The length of the string is 18√5 feet -/
theorem string_length_is_18_sqrt_5 :
  string_length 6 18 6 = 18 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_is_18_sqrt_5_l339_33990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l339_33911

/-- The distance from a point to a line --/
noncomputable def distancePointToLine (x₀ y₀ : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance from the point (1, 2) to the line y = 2x + 1 is √5/5 --/
theorem distance_point_to_line :
  distancePointToLine 1 2 2 (-1) 1 = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_l339_33911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_three_l339_33921

def sequence_a : ℕ → ℚ
  | 0 => 2  -- Add this case for 0
  | 1 => 2
  | (n + 2) => (1 + sequence_a (n + 1)) / (1 - sequence_a (n + 1))

theorem a_2018_equals_negative_three :
  sequence_a 2018 = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2018_equals_negative_three_l339_33921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_and_h_zeros_l339_33929

open Real

noncomputable def f (x : ℝ) : ℝ := log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (2 * a / 3) * x^3 + 2 * (1 - a) * x^2 - 8 * x + 8 * a + 7

noncomputable def h (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then
    min (f x) (g a x)
  else 0

theorem g_range_and_h_zeros (a : ℝ) :
  (∀ x ∈ Set.Icc 0 3, g a x ∈ Set.Icc (-1) 7) ∧
  (∃! x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ h a x₁ = 0 ∧ h a x₂ = 0 ∧ h a x₃ = 0) ↔
  a ∈ ({-3/20} : Set ℝ) ∪ Set.Icc 0 (3/16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_and_h_zeros_l339_33929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_unit_vectors_l339_33903

noncomputable def angle (v w : ℝ × ℝ) : ℝ := Real.arccos ((v.1 * w.1 + v.2 * w.2) / (Real.sqrt (v.1^2 + v.2^2) * Real.sqrt (w.1^2 + w.2^2)))

theorem angle_between_unit_vectors 
  (s t : ℝ) 
  (i j : ℝ × ℝ) 
  (hs : s ≠ 0) 
  (ht : t ≠ 0) 
  (hi : ‖i‖ = 1) 
  (hj : ‖j‖ = 1) 
  (h_equal_mag : ‖s • i + t • j‖ = ‖t • i - s • j‖) : 
  angle i j = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_unit_vectors_l339_33903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_inequality_l339_33977

/-- Helper function to determine if three sides form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Helper function to calculate the area of a triangle given its side lengths -/
noncomputable def area_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Helper function to determine if two triangles are similar -/
def similar_triangles (a b c a' b' c' : ℝ) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ a' = k * a ∧ b' = k * b ∧ c' = k * c

/-- Triangle similarity theorem -/
theorem triangle_similarity_inequality 
  (a b c a' b' c' : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0)
  (hABC : is_triangle a b c) 
  (hA'B'C' : is_triangle a' b' c')
  (S : ℝ) (hS : S = area_triangle a b c)
  (S' : ℝ) (hS' : S' = area_triangle a' b' c') :
  a^2 * (-a'^2 + b'^2 + c'^2) + b^2 * (a'^2 - b'^2 + c'^2) + c^2 * (a'^2 + b'^2 - c'^2) ≥ 16 * S * S' ∧
  (a^2 * (-a'^2 + b'^2 + c'^2) + b^2 * (a'^2 - b'^2 + c'^2) + c^2 * (a'^2 + b'^2 - c'^2) = 16 * S * S' ↔ 
   similar_triangles a b c a' b' c') :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_inequality_l339_33977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisectors_cannot_divide_equally_l339_33909

/-- A triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- An angle bisector of a triangle -/
structure AngleBisector (t : Triangle) where
  vertex : Point
  endpoint : Point

/-- Represents the division of a triangle by two angle bisectors -/
structure BisectorDivision (t : Triangle) where
  bisector1 : AngleBisector t
  bisector2 : AngleBisector t

/-- The areas of the four parts created by the bisector division -/
noncomputable def areas (t : Triangle) (d : BisectorDivision t) : Finset ℝ := sorry

theorem angle_bisectors_cannot_divide_equally (t : Triangle) :
  ¬∃ (d : BisectorDivision t), (areas t d).card = 4 ∧ (∀ a₁ a₂, a₁ ∈ areas t d → a₂ ∈ areas t d → a₁ = a₂) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_bisectors_cannot_divide_equally_l339_33909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l339_33915

theorem problem_statement (a d c : ℝ) (h1 : a < 0) (h2 : c > 0) (h3 : a < d) (h4 : d < c) :
  (a * c < d * c) ∧ (a * d < d * c) ∧ (a + d < d + c) ∧ (d / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l339_33915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l339_33926

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_sixths_l339_33926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_profit_l339_33927

/-- Represents the monthly production volume in units -/
noncomputable def x : ℝ := sorry

/-- Represents the unit selling price in ten thousand yuan -/
noncomputable def y₁ (x : ℝ) : ℝ := 150 - (3/2) * x

/-- Represents the total production cost in ten thousand yuan -/
noncomputable def y₂ (x : ℝ) : ℝ := 600 + 72 * x

/-- The average profit per unit in ten thousand yuan -/
noncomputable def average_profit (x : ℝ) : ℝ := (x * y₁ x - y₂ x) / x

/-- The constraint on the minimum selling price -/
axiom min_price (x : ℝ) : y₁ x ≥ 90

/-- The constraint on the production volume -/
axiom production_constraint (x : ℝ) : 0 < x ∧ x ≤ 40

theorem max_average_profit :
  ∃ (x : ℝ), x = 20 ∧ average_profit x = 18 ∧
  ∀ (x' : ℝ), 0 < x' → x' ≤ 40 → average_profit x' ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_profit_l339_33927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomials_l339_33960

def product_polynomial (z : ℝ) : ℝ := z^8 + 4*z^7 + 5*z^6 + 7*z^5 + 9*z^4 + 8*z^3 + 6*z^2 + 8*z + 9

def is_monic_degree_4 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d, p = λ z ↦ z^4 + a*z^3 + b*z^2 + c*z + d

theorem constant_term_of_polynomials 
  (p q : ℝ → ℝ) 
  (h_monic_p : is_monic_degree_4 p)
  (h_monic_q : is_monic_degree_4 q)
  (h_same_z3 : ∃ k > 0, ∀ z, ∃ r s, p z - z^4 = k*z^3 + r ∧ q z - z^4 = k*z^3 + s)
  (h_same_const : ∃ c > 0, p 0 = c ∧ q 0 = c)
  (h_product : ∀ z, p z * q z = product_polynomial z) :
  p 0 = 3 ∧ q 0 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomials_l339_33960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l339_33975

-- Define the logarithms
noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 10 / Real.log 5
noncomputable def c : ℝ := Real.log 14 / Real.log 7

-- Theorem statement
theorem log_inequality : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_inequality_l339_33975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_l339_33906

theorem simplify_trigonometric_expression (θ : Real) (h : θ ∈ Set.Ioo 0 (π/4)) :
  Real.sqrt (1 - 2 * Real.sin (3 * π - θ) * Real.sin (π/2 + θ)) = Real.cos θ - Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplify_trigonometric_expression_l339_33906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l339_33907

theorem tan_alpha_fourth_quadrant (α : ℝ) :
  (α > -π ∧ α < -π/2) →  -- α is in the fourth quadrant
  Real.cos (π/2 + α) = 4/5 →
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_fourth_quadrant_l339_33907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_factorial_sum_l339_33979

theorem largest_prime_divisor_of_factorial_sum :
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 11 + Nat.factorial 12) ∧
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 11 + Nat.factorial 12) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_factorial_sum_l339_33979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_sine_l339_33928

theorem angle_terminal_side_sine (α : Real) :
  let P : Real × Real := (Real.sin (600 * π / 180), Real.cos (-120 * π / 180))
  (∃ r : Real, r > 0 ∧ (r * Real.cos α, r * Real.sin α) = P) →
  Real.sin α = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_terminal_side_sine_l339_33928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l339_33943

/-- Calculates the percentage increase between two amounts -/
noncomputable def percentageIncrease (originalAmount newAmount : ℝ) : ℝ :=
  ((newAmount - originalAmount) / originalAmount) * 100

theorem johns_earnings_increase (initialEarnings newEarnings : ℝ) 
  (h1 : initialEarnings = 60)
  (h2 : newEarnings = 75) :
  percentageIncrease initialEarnings newEarnings = 25 := by
  sorry

-- Remove the #eval line as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_earnings_increase_l339_33943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_midpoint_to_line_l339_33992

-- Define the curves and line
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, 3 + Real.sin t)
noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, 3 * Real.sin θ)
def C₃ (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 + t)

-- Define point P on C₁
noncomputable def P : ℝ × ℝ := C₁ (Real.pi / 2)

-- Define the midpoint M of PQ
noncomputable def M (θ : ℝ) : ℝ × ℝ :=
  let Q := C₂ θ
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the distance function between a point and a line
noncomputable def distancePointLine (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  abs (a * p.1 + b * p.2 + c) / Real.sqrt (a^2 + b^2)

-- Theorem statement
theorem min_distance_midpoint_to_line :
  ∃ (d : ℝ), d = 8 * Real.sqrt 5 / 5 ∧
  ∀ (θ : ℝ), distancePointLine (M θ) 1 (-2) (-7) ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_midpoint_to_line_l339_33992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_word_game_possible_outcome_l339_33995

/-- Represents a player in the word game -/
inductive Player : Type
  | Anya : Player
  | Borya : Player
  | Vasya : Player

/-- Represents the words created by each player -/
def WordDistribution := Player → Finset String

/-- Calculates the score for a player given the word distribution -/
def score (wd : WordDistribution) (p : Player) : ℕ :=
  sorry

/-- Checks if the word distribution satisfies the game conditions -/
def validDistribution (wd : WordDistribution) : Prop :=
  sorry

/-- Helper function to count words for a player -/
def wordCount (wd : WordDistribution) (p : Player) : ℕ :=
  (wd p).card

theorem word_game_possible_outcome :
  ∃ (wd : WordDistribution),
    validDistribution wd ∧
    (∀ p, wordCount wd Player.Anya ≥ wordCount wd p) ∧
    (∀ p, wordCount wd Player.Vasya ≤ wordCount wd p) ∧
    (∀ p, score wd Player.Vasya ≥ score wd p) ∧
    (∀ p, score wd Player.Anya ≤ score wd p) :=
  by
    sorry

#check word_game_possible_outcome

end NUMINAMATH_CALUDE_ERRORFEEDBACK_word_game_possible_outcome_l339_33995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_distance_equals_radius_l339_33969

/-- The distance from a point (x₀, y₀) to a line Ax + By + C = 0 -/
noncomputable def distance_point_to_line (x₀ y₀ A B C : ℝ) : ℝ :=
  (abs (A * x₀ + B * y₀ + C)) / Real.sqrt (A^2 + B^2)

/-- Theorem: Given a circle and a line, if they have only one common point,
    then the distance from the center of the circle to the line equals the radius -/
theorem tangent_line_distance_equals_radius
  (a : ℝ) -- Center x-coordinate of the circle
  (h_a : a > 0) -- Assumption: a is positive
  : (∃! p : ℝ × ℝ, (p.1 - a)^2 + (p.2 - 1)^2 = 16 ∧ 3*p.1 + 4*p.2 = 5) →
    distance_point_to_line a 1 3 4 (-5) = 4 →
    a = 7 := by
  sorry

#check tangent_line_distance_equals_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_distance_equals_radius_l339_33969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l339_33961

noncomputable def p : ℝ := 5 + 4 * Real.sqrt 2
noncomputable def q : ℝ := 5 - 4 * Real.sqrt 2

def S : ℕ → ℝ
  | 0 => 1
  | 1 => 5
  | (n + 2) => 10 * S (n + 1) - 9 * S n

theorem units_digit_S_6789 : S 6789 % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l339_33961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_numbers_property_l339_33945

/-- A function that returns the number of digits in a natural number -/
def numDigits (n : ℕ) : ℕ :=
  if n < 10 then 1 else 1 + numDigits (n / 10)

/-- A function that returns the four-digit number obtained by removing the leftmost digit of a five-digit number -/
def removeLeftmostDigit (n : ℕ) : ℕ :=
  n % 10000

/-- The main theorem stating that there are exactly 9 five-digit numbers satisfying the given condition -/
theorem five_digit_numbers_property : 
  (Finset.filter (fun n => numDigits n = 5 ∧ removeLeftmostDigit n = n / 11) (Finset.range 100000)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_digit_numbers_property_l339_33945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l339_33934

/-- Represents an ellipse with equation x^2/4 + y^2 = 1 -/
structure Ellipse where
  equation : ∀ x y : ℝ, x^2/4 + y^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt 3 / 2

/-- The length of the focal axis of the ellipse -/
noncomputable def focal_axis_length (e : Ellipse) : ℝ := 2 * Real.sqrt 3

theorem ellipse_properties (e : Ellipse) :
  eccentricity e = Real.sqrt 3 / 2 ∧ focal_axis_length e = 2 * Real.sqrt 3 := by
  sorry

#check ellipse_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l339_33934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_count_possibilities_l339_33983

/-- Represents the number of ants moving in one direction -/
def x : ℕ := sorry

/-- Represents the number of ants moving in the opposite direction -/
def y : ℕ := sorry

/-- The length of the circular track in centimeters -/
def track_length : ℕ := 60

/-- The speed of the ants in cm/s -/
def ant_speed : ℕ := 1

/-- The number of pairwise collisions in one minute -/
def collisions : ℕ := 48

/-- The set of possible total numbers of ants on the track -/
def possible_ant_counts : Set ℕ := {10, 11, 14, 25}

/-- Theorem stating that the possible numbers of ants on the track are 10, 11, 14, and 25 -/
theorem ant_count_possibilities : 
  (x * y = collisions / 2) → 
  (x + y) ∈ possible_ant_counts := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_count_possibilities_l339_33983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l339_33918

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*log x

theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo 0 2, ∀ y ∈ Set.Ioi 2, f x > f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_interval_l339_33918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_calculation_l339_33958

/-- The time taken for a train to pass a platform -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (platform_length : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  total_distance / train_speed_ms

/-- Theorem: The time taken for a train of length 360 m, traveling at 45 km/hr, 
    to pass a platform of length 130 m is approximately 39.2 seconds -/
theorem train_passing_time_calculation : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |train_passing_time 360 45 130 - 39.2| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_calculation_l339_33958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l339_33999

theorem alpha_value (α : Real) 
  (h1 : Real.sin α = -1/2) 
  (h2 : α ∈ Set.Ioo (π/2) (3*π/2)) : 
  α = 7*π/6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_value_l339_33999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_parallels_l339_33955

/-- Represents a line in 2D space with equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Calculates the distance between two parallel lines -/
noncomputable def distance (l1 l2 : Line) : ℝ :=
  abs (l1.c - l2.c) / Real.sqrt (l1.a^2 + l1.b^2)

/-- The main theorem -/
theorem line_equidistant_from_parallels (l l1 l2 : Line) : 
  parallel l l1 ∧ 
  parallel l l2 ∧ 
  l1 = Line.mk 2 (-1) 3 ∧ 
  l2 = Line.mk 2 (-1) (-1) ∧
  distance l l1 = distance l l2 →
  l = Line.mk 2 (-1) 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equidistant_from_parallels_l339_33955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l339_33964

noncomputable def f (x : ℝ) := 2 * Real.cos x - 1

theorem f_max_min :
  (∀ x, f x ≤ 1) ∧ (∃ x, f x = 1) ∧ (∀ x, f x ≥ -3) ∧ (∃ x, f x = -3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_l339_33964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l339_33948

open Real

noncomputable def curve_C (α : ℝ) : ℝ × ℝ :=
  (3 + 5 * cos α, 4 + 5 * sin α)

noncomputable def point_A : ℝ × ℝ :=
  (Real.sqrt 25, π / 6)

def theta_range : Set ℝ :=
  { θ | 0 ≤ θ ∧ θ ≤ π }

noncomputable def polar_equation_C (θ : ℝ) : ℝ :=
  6 * cos θ + 8 * sin θ

noncomputable def triangle_area (θ : ℝ) : ℝ :=
  25 / 2 * sin (2 * θ - π / 3)

noncomputable def max_triangle_area : ℝ :=
  25 / 2

noncomputable def max_area_point_B : ℝ × ℝ :=
  (7 * Real.sqrt 6 / 2, 5 * π / 12)

theorem curve_C_properties :
  (∀ θ ∈ theta_range, polar_equation_C θ = Real.sqrt ((3 + 5 * cos θ)^2 + (4 + 5 * sin θ)^2)) ∧
  (∀ θ ∈ theta_range, triangle_area θ ≤ max_triangle_area) ∧
  (triangle_area (5 * π / 12) = max_triangle_area) ∧
  (polar_equation_C (5 * π / 12) = 7 * Real.sqrt 6 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l339_33948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_odds_l339_33980

/-- Represents the odds against an event occurring -/
structure Odds where
  against : ℚ
  in_favor : ℚ
  valid : against > 0 ∧ in_favor > 0

/-- Calculates the probability of an event given its odds -/
def oddsToProb (o : Odds) : ℚ :=
  o.in_favor / (o.against + o.in_favor)

theorem horse_race_odds (oddsA oddsB : Odds)
    (hA : oddsA.against = 2 ∧ oddsA.in_favor = 1)
    (hB : oddsB.against = 4 ∧ oddsB.in_favor = 1)
    (hNoTies : oddsToProb oddsA + oddsToProb oddsB < 1) :
    ∃ oddsC : Odds, oddsC.against = 8 ∧ oddsC.in_favor = 7 ∧
    oddsToProb oddsA + oddsToProb oddsB + oddsToProb oddsC = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_horse_race_odds_l339_33980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_projection_OB_l339_33904

/-- Point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Vector in 3D space -/
def Vector3D := Point3D

/-- Projection of a point onto the XOZ plane -/
def projectOntoXOZ (p : Point3D) : Point3D :=
  { x := p.x, y := 0, z := p.z }

/-- Magnitude of a vector in the XOZ plane -/
noncomputable def magnitudeXOZ (v : Vector3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.z ^ 2)

/-- The main theorem -/
theorem magnitude_of_projection_OB :
  let A : Point3D := { x := 1, y := -3, z := 2 }
  let B : Point3D := projectOntoXOZ A
  let OB : Vector3D := B
  magnitudeXOZ OB = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_projection_OB_l339_33904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l339_33991

-- Define the lines
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 4
noncomputable def line2 (x : ℝ) : ℝ := -3 * x + 9
def line3 : ℝ := 2

-- Define the vertices of the triangle
noncomputable def vertex1 : ℝ × ℝ := (-1, 2)
noncomputable def vertex2 : ℝ × ℝ := (7/3, 2)
noncomputable def vertex3 : ℝ × ℝ := (1, 6)

-- Define the function to calculate the area of a triangle given base and height
noncomputable def triangle_area (base height : ℝ) : ℝ := (1/2) * base * height

-- Theorem statement
theorem triangle_properties :
  let base := |vertex2.1 - vertex1.1|
  let height := vertex3.2 - line3
  let area := triangle_area base height
  let x_p := (vertex1.1 + vertex2.1) / 2
  area = 20/3 ∧ x_p = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l339_33991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_conditions_l339_33920

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Function to calculate the area of a triangle formed by a line and coordinate axes
noncomputable def triangleArea (l : Line) : ℝ :=
  abs (l.c / l.a * l.c / l.b) / 2

-- Theorem statement
theorem line_equation_with_conditions (l : Line) : 
  pointOnLine { x := -2, y := 2 } l ∧ 
  triangleArea l = 1 →
  (l.a = 2 ∧ l.b = 1 ∧ l.c = -2) ∨ (l.a = 1 ∧ l.b = 2 ∧ l.c = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_with_conditions_l339_33920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_bounds_l339_33938

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 6 = 0

-- Define the sum function
def my_sum (x y : ℝ) : ℝ := x + y

-- Theorem statement
theorem circle_sum_bounds :
  ∀ x y : ℝ, my_circle x y → 2 ≤ my_sum x y ∧ my_sum x y ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_sum_bounds_l339_33938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_limit_l339_33982

noncomputable def partialProduct (n : ℕ) : ℝ :=
  9 * (Finset.range n).prod (fun k => 9 ^ (1 / 3^k))

theorem infinite_product_limit : 
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |partialProduct n - 27| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_product_limit_l339_33982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_x_l339_33970

def g (x : ℤ) : ℤ := (5*x+3)*(11*x+2)*(17*x+7)*(3*x+8)

theorem gcd_g_x (x : ℤ) (h : x % 27720 = 0) : Int.gcd (g x) x = 168 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_g_x_l339_33970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_range_l339_33974

noncomputable section

def f : ℝ → ℝ := sorry

axiom f_second_derivative (x : ℝ) : 
  (deriv^[2] f) x - f x = 2 * x * Real.exp x

axiom f_initial_value : f 0 = 1

theorem f_ratio_range :
  ∀ x > 0, 1 < ((deriv^[2] f) x) / (f x) ∧ ((deriv^[2] f) x) / (f x) ≤ 2 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_ratio_range_l339_33974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_2_l339_33944

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1 / x) * cos x

-- State the theorem
theorem derivative_f_at_pi_over_2 :
  deriv f (π / 2) = -2 / π :=
by
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_f_at_pi_over_2_l339_33944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_1194_l339_33959

/-- A trapezoid with specific measurements -/
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_trapezoid : True  -- Placeholder for the trapezoid property
  AD_length : dist A D = 20
  AB_length : dist A B = 60
  BC_length : dist B C = 30
  altitude : ℝ
  altitude_value : altitude = 15

/-- The area of the trapezoid -/
noncomputable def trapezoid_area (t : Trapezoid) : ℝ := 
  (dist t.A t.B + dist t.C t.D) * t.altitude / 2

/-- Theorem stating that the area of the given trapezoid is 1194 -/
theorem trapezoid_area_is_1194 (t : Trapezoid) : 
  trapezoid_area t = 1194 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_area_is_1194_l339_33959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stephanie_soy_sauce_bottles_l339_33965

/-- Represents the amount of soy sauce in ounces -/
abbrev SoySauce := ℕ

/-- Converts cups to ounces -/
def cupsToOunces (cups : ℕ) : SoySauce :=
  cups * 8

/-- Calculates the number of bottles needed given the total ounces required -/
def bottlesNeeded (totalOunces : SoySauce) : ℕ :=
  (totalOunces + 15) / 16

theorem stephanie_soy_sauce_bottles :
  let ouncesPerBottle : SoySauce := 16
  let recipe1 : SoySauce := cupsToOunces 2
  let recipe2 : SoySauce := cupsToOunces 1
  let recipe3 : SoySauce := cupsToOunces 3
  let totalOunces : SoySauce := recipe1 + recipe2 + recipe3
  bottlesNeeded totalOunces = 3 := by
  -- Unfold definitions
  unfold bottlesNeeded cupsToOunces
  -- Evaluate expressions
  simp
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stephanie_soy_sauce_bottles_l339_33965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_game_outcome_second_player_wins_square_board_first_player_wins_rectangular_board_l339_33941

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
  | FirstPlayerWins
  | SecondPlayerWins

/-- Represents a position on the chessboard -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the game state -/
structure GameState where
  board_size : Nat × Nat
  rook_position : Position

/-- Defines a valid move for the rook -/
def is_valid_move (start finish : Position) : Prop :=
  (finish.row < start.row ∧ finish.col = start.col) ∨ 
  (finish.col < start.col ∧ finish.row = start.row)

/-- Determines the winner of the game based on the board size -/
def determine_winner (k n : Nat) : GameOutcome :=
  if k = n then GameOutcome.SecondPlayerWins else GameOutcome.FirstPlayerWins

/-- The main theorem about the game outcome -/
theorem rook_game_outcome (k n : Nat) :
  determine_winner k n = 
    if k = n 
    then GameOutcome.SecondPlayerWins 
    else GameOutcome.FirstPlayerWins := by
  sorry

/-- Theorem stating that the second player wins when K = N -/
theorem second_player_wins_square_board (n : Nat) :
  determine_winner n n = GameOutcome.SecondPlayerWins := by
  sorry

/-- Theorem stating that the first player wins when K ≠ N -/
theorem first_player_wins_rectangular_board {k n : Nat} (h : k ≠ n) :
  determine_winner k n = GameOutcome.FirstPlayerWins := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rook_game_outcome_second_player_wins_square_board_first_player_wins_rectangular_board_l339_33941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_day_sprint_score_estimate_l339_33994

/-- The score increase function model -/
noncomputable def f (k P t : ℝ) : ℝ := (k * P) / (1 + Real.log (t + 1))

/-- Theorem stating the approximate total score after 100 days -/
theorem hundred_day_sprint_score_estimate 
  (P : ℝ) 
  (h1 : P = 400) 
  (k : ℝ) 
  (h2 : f k P 60 = P / 6) 
  (h3 : 30 ≤ 100 ∧ 100 ≤ 100) : 
  ∃ (ε : ℝ), abs (f k P 100 + P - 460) < ε ∧ ε > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_day_sprint_score_estimate_l339_33994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l339_33981

-- Define an even function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 - 4*x else ((-x)^2 - 4*(-x))

-- State the theorem
theorem solution_set (x : ℝ) : 
  (∀ y, f y = f (-y)) →  -- f is even
  (∀ z ≥ 0, f z = z^2 - 4*z) →  -- definition of f for non-negative reals
  (f (x - 2) < 5 ↔ -3 < x ∧ x < 7) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l339_33981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_function_l339_33987

theorem supremum_of_function (a b : ℝ) : 
  a > 0 → b > 0 → a + b = 1 → 
  IsLUB {y | ∃ x, y = -(1/(2*x)) - 2/((1-x))} (-9/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_supremum_of_function_l339_33987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_lines_l339_33996

/-- A line in the xy-plane with x-intercept a and y-intercept b -/
structure Line where
  a : ℝ
  b : ℝ

/-- Check if a real number is a positive integer -/
def isPositiveInteger (x : ℝ) : Prop := x > 0 ∧ ∃ n : ℕ, x = n

/-- Check if a natural number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

/-- Check if a line passes through the point (6,5) -/
def passesThroughPoint (l : Line) : Prop := 6 / l.a + 5 / l.b = 1

/-- The set of lines that satisfy all conditions -/
def validLines : Set Line := {l : Line | 
  isPositiveInteger l.b ∧ 
  ∃ n : ℕ, l.a = n ∧ isPrime n ∧ 
  passesThroughPoint l}

theorem exactly_two_valid_lines : ∃ s : Finset Line, s.card = 2 ∧ ∀ l : Line, l ∈ s ↔ l ∈ validLines := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_valid_lines_l339_33996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_problem_l339_33935

theorem digit_sum_problem :
  ∀ (a b c d e f g : ℕ),
    a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    b ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    c ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    d ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    e ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    f ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    g ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g →
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g →
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g →
    d ≠ e ∧ d ≠ f ∧ d ≠ g →
    e ≠ f ∧ e ≠ g →
    f ≠ g →
    a + b + c = 24 →
    d + e + f + g = 15 →
    b = e →
    a + b + c + d + f + g = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_problem_l339_33935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l339_33997

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define a line
def Line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define tangency condition for ellipse
def TangentToC₁ (k m : ℝ) : Prop :=
  2 * k^2 - m^2 + 1 = 0

-- Define tangency condition for parabola
def TangentToC₂ (k m : ℝ) : Prop :=
  k * m = 1

theorem ellipse_and_tangent_line 
  (a b : ℝ) (h_ellipse : C₁ a b 0 1) (h_focus : a^2 - b^2 = 1) :
  ∃ (k m : ℝ),
    C₁ (Real.sqrt 2) 1 0 1 ∧
    (Line k m 0 1 ∧ TangentToC₁ k m ∧ TangentToC₂ k m) ∧
    ((k = Real.sqrt 2 / 2 ∧ m = Real.sqrt 2) ∨ (k = -Real.sqrt 2 / 2 ∧ m = -Real.sqrt 2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_tangent_line_l339_33997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_changes_variance_unchanged_l339_33940

open Real BigOperators

variable {α : Type*} [LinearOrderedField α]

def subtract_constant (s : Finset α) (c : α) : Finset α :=
  s.image (fun x => x - c)

noncomputable def mean (s : Finset α) : α :=
  s.sum id / s.card

noncomputable def variance (s : Finset α) : α :=
  s.sum (fun x => (x - mean s) ^ 2) / s.card

theorem mean_changes_variance_unchanged
  (s : Finset α) (c : α) (h_nonempty : s.Nonempty) (h_nonzero : c ≠ 0) :
  let s' := subtract_constant s c
  (mean s' ≠ mean s) ∧ (variance s' = variance s) := by
  sorry

#check mean_changes_variance_unchanged

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_changes_variance_unchanged_l339_33940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_fraction_l339_33984

/-- Operation ⊙ defined on positive real numbers -/
noncomputable def odot (x y : ℝ) : ℝ := (x * y + 4) / (x + y)

/-- Associativity of ⊙ operation -/
axiom odot_assoc (x y z : ℝ) : x > 0 → y > 0 → z > 0 → odot x (odot y z) = odot (odot x y) z

/-- T defined as repeated application of ⊙ operation -/
noncomputable def T (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | 1 => 3
  | n+1 => odot (T n) (n+3)

/-- Main theorem to prove or disprove -/
theorem exists_perfect_square_fraction (n : ℕ) : n ≥ 4 → ∃ k : ℕ, (96 : ℝ) / (T n - 2) = k^2 := by
  sorry

/-- Helper lemma: T(n) > 2 for n ≥ 4 -/
lemma T_gt_two (n : ℕ) (h : n ≥ 4) : T n > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_fraction_l339_33984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solution_angles_l339_33952

noncomputable def complex_polar (r : ℝ) (θ : ℝ) : ℂ := r * Complex.exp (θ * Complex.I)

def is_solution (z : ℂ) : Prop := z^4 = 4 * Complex.I

def valid_angle (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

theorem sum_of_solution_angles :
  ∃ (r : Fin 4 → ℝ) (θ : Fin 4 → ℝ),
    (∀ k, r k > 0) ∧
    (∀ k, valid_angle (θ k)) ∧
    (∀ k, is_solution (complex_polar (r k) (θ k))) ∧
    (θ 0 + θ 1 + θ 2 + θ 3 = 11 * Real.pi / 6) :=
by
  sorry

#check sum_of_solution_angles

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solution_angles_l339_33952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_pairs_in_same_row_or_column_l339_33933

/-- A type representing a 7x7 grid arrangement of numbers 1 through 49 -/
def Grid := Fin 7 → Fin 7 → Fin 49

/-- The number of ways to choose 2 positions in the same row or column -/
def sameRowOrColumnPositions : ℕ := 2 * 7 * (Nat.choose 7 2)

/-- The total number of ways to choose any 2 out of 49 positions -/
def totalPositions : ℕ := Nat.choose 49 2

/-- The probability that a pair is in the same row or column in one arrangement -/
def probSameRowOrColumn : ℚ := sameRowOrColumnPositions / totalPositions

/-- Expected number of pairs in the same row or column in both arrangements -/
noncomputable def expectedPairs : ℚ := totalPositions * (probSameRowOrColumn ^ 2)

theorem expected_pairs_in_same_row_or_column (g1 g2 : Grid) :
  expectedPairs = 147/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_pairs_in_same_row_or_column_l339_33933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_triangle_or_square_l339_33917

/-- The probability of choosing either a triangle or a square from a set of figures -/
theorem probability_triangle_or_square (total : ℕ) (triangles squares circles : ℕ)
  (h_total : total = triangles + squares + circles)
  (h_triangles : triangles = 4)
  (h_squares : squares = 3)
  (h_circles : circles = 3) :
  (triangles + squares : ℚ) / total = 7 / 10 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_triangle_or_square_l339_33917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part1_solution_part2_l339_33947

-- Define the lines
def l₁ (x y : ℝ) : Prop := 2*x + 3*y - 5 = 0
def l₂ (x y : ℝ) : Prop := 7*x + 15*y + 1 = 0
def l₃ (x y : ℝ) : Prop := x + 2*y - 3 = 0
def l₄ (x y : ℝ) : Prop := 3*x + 4*y - 7 = 0

-- Define the solution lines
def solution₁ (x y : ℝ) : Prop := 9*x + 18*y - 4 = 0
def solution₂₁ (x y : ℝ) : Prop := 4*x - 3*y + 30 = 0
def solution₂₂ (x y : ℝ) : Prop := 4*x - 3*y - 30 = 0

-- Define the distance function
noncomputable def distance_from_origin (a b c : ℝ) : ℝ := |c| / Real.sqrt (a^2 + b^2)

-- Theorem for part 1
theorem solution_part1 :
  ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ solution₁ x y ∧
  ∃ (k : ℝ), ∀ (x y : ℝ), solution₁ x y ↔ l₃ x (y + k) := by sorry

-- Theorem for part 2
theorem solution_part2 :
  (∀ (x y : ℝ), solution₂₁ x y ↔ l₄ y (-x)) ∧
  (∀ (x y : ℝ), solution₂₂ x y ↔ l₄ y (-x)) ∧
  distance_from_origin 4 (-3) 30 = 6 ∧
  distance_from_origin 4 (-3) (-30) = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_part1_solution_part2_l339_33947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_ball_probability_theorem_l339_33946

def box_contents (i : ℕ) : ℕ × ℕ :=
  if i = 1 then (2, 1) else (1, 1)

def transfer_probability (n : ℕ) : ℚ → ℚ :=
  λ p ↦ 1/3 * p + 1/3

def white_ball_probability (n : ℕ) : ℚ :=
  1/2 * (1/3)^n + 1/2

theorem white_ball_probability_theorem (n : ℕ) :
  white_ball_probability 2 = 5/9 ∧
  ∀ k : ℕ, k > 0 → white_ball_probability k = transfer_probability k (white_ball_probability (k-1)) :=
by
  sorry

#eval white_ball_probability 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_ball_probability_theorem_l339_33946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ap_l339_33942

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of the legs of the right triangle -/
  leg_length : ℝ
  /-- The radius of the inscribed circle -/
  circle_radius : ℝ
  /-- The point P on the circle -/
  point_p : ℝ × ℝ
  /-- The leg_length is positive -/
  leg_length_pos : 0 < leg_length
  /-- The circle_radius is positive -/
  circle_radius_pos : 0 < circle_radius
  /-- The circle touches both legs -/
  circle_touches_legs : circle_radius = leg_length / (1 + Real.sqrt 2)
  /-- The point P is on the circle -/
  p_on_circle : (point_p.1 - circle_radius)^2 + (point_p.2 - circle_radius)^2 = circle_radius^2
  /-- The point P is on the line y = x -/
  p_on_diagonal : point_p.1 = point_p.2
  /-- The point P is different from M (circle_radius, 0) -/
  p_not_m : point_p.1 ≠ circle_radius

/-- The main theorem -/
theorem length_ap (t : RightTriangleWithInscribedCircle) :
  Real.sqrt (t.point_p.1^2 + t.point_p.2^2) = (Real.sqrt 2 - 1) / (2 * (1 + Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_ap_l339_33942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l339_33989

/-- Compound interest function -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * (1 + r) ^ t

theorem interest_rate_calculation (P : ℝ) (r : ℝ) 
  (h1 : compound_interest P r 2 = 17640)
  (h2 : compound_interest P r 3 = 21168) :
  ∃ ε > 0, |r - 0.1998| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_calculation_l339_33989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_f_strictly_increasing_all_intervals_l339_33971

open Real Set

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 3 * cos (-2 * x + π / 6) + 2

-- Define the interval
def increasing_interval (k : ℤ) : Set ℝ := 
  Icc (7 * π / 12 + ↑k * π) (13 * π / 12 + ↑k * π)

-- Theorem statement
theorem f_strictly_increasing (k : ℤ) :
  StrictMonoOn f (increasing_interval k) := by
  sorry

-- Main theorem combining all intervals
theorem f_strictly_increasing_all_intervals :
  ∀ k : ℤ, StrictMonoOn f (increasing_interval k) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_f_strictly_increasing_all_intervals_l339_33971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_multiple_theorem_special_integers_theorem_l339_33973

/-- Function to reverse an integer -/
def reverse (n : ℤ) : ℤ := sorry

/-- The set of integers that are 4 or 9 times less than their reverse -/
def special_integers : Set ℤ := {n : ℤ | n = reverse n / 4 ∨ n = reverse n / 9}

theorem reverse_multiple_theorem :
  ∀ (n : ℤ) (k : ℤ), n ≠ 0 → k ∈ ({2, 3, 5, 6, 7, 8} : Set ℤ) → n ≠ k * reverse n :=
by
  sorry

theorem special_integers_theorem :
  special_integers = {0, 1089, 10989, 109989, 21978, 219978} ∪ 
    {n : ℤ | ∃ (m : ℕ), n = 10^m * 1089 ∨ n = 10^m * 21978} :=
by
  sorry

#check reverse_multiple_theorem
#check special_integers_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reverse_multiple_theorem_special_integers_theorem_l339_33973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sixty_degrees_l339_33912

theorem triangle_angle_sixty_degrees 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : (a + b + c) * (b + c - a) = 3 * b * c) : 
  Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)) = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_sixty_degrees_l339_33912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_param_sum_squares_l339_33978

/-- A line segment connecting two points in 2D space -/
structure LineSegment where
  start : ℝ × ℝ
  end_ : ℝ × ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  eq_x : ∀ t, 0 ≤ t ∧ t ≤ 1 → start.1 + t * (end_.1 - start.1) = a * t + b
  eq_y : ∀ t, 0 ≤ t ∧ t ≤ 1 → start.2 + t * (end_.2 - start.2) = c * t + d

/-- The sum of squares of parameters for the given line segment is 179 -/
theorem line_segment_param_sum_squares (l : LineSegment) 
  (h1 : l.start = (1, -3))
  (h2 : l.end_ = (-4, 9)) :
  l.a^2 + l.b^2 + l.c^2 + l.d^2 = 179 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_param_sum_squares_l339_33978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l339_33908

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16 * x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (4, 0)

-- Define the line l
def line_l (x : ℝ) : Prop := x = -1

-- Define point A on line l
noncomputable def point_A : ℝ × ℝ := (-1, Real.sqrt 1200)

-- Define point B on parabola C
noncomputable def point_B : ℝ × ℝ := (3, 4 * Real.sqrt 3)

-- Define the vector from F to A
noncomputable def vector_FA : ℝ × ℝ := (point_A.1 - focus.1, point_A.2 - focus.2)

-- Define the vector from F to B
noncomputable def vector_FB : ℝ × ℝ := (point_B.1 - focus.1, point_B.2 - focus.2)

-- Theorem statement
theorem length_AB : 
  parabola point_B.1 point_B.2 ∧ 
  line_l point_A.1 ∧ 
  vector_FA = (5 * vector_FB.1, 5 * vector_FB.2) →
  Real.sqrt ((point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2) = 28 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_l339_33908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_sum_l339_33924

theorem square_roots_sum (a b : ℝ) 
  (h1 : ∃ z₁ z₂ z₃ z₄ : ℂ, z₁ ≠ z₂ ∧ z₁ ≠ z₃ ∧ z₁ ≠ z₄ ∧ z₂ ≠ z₃ ∧ z₂ ≠ z₄ ∧ z₃ ≠ z₄ ∧
    (∀ z : ℂ, (z^2 + a*z + b) * (z^2 + a*z + 2*b) = 0 ↔ z = z₁ ∨ z = z₂ ∨ z = z₃ ∨ z = z₄) ∧
    Complex.abs (z₁ - z₂) = 1 ∧
    Complex.abs (z₂ - z₃) = 1 ∧
    Complex.abs (z₃ - z₄) = 1 ∧
    Complex.abs (z₄ - z₁) = 1) :
  ∃ z₁ z₂ z₃ z₄ : ℂ, Complex.abs z₁ + Complex.abs z₂ + Complex.abs z₃ + Complex.abs z₄ = Real.sqrt 6 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_roots_sum_l339_33924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sin_cos_larger_angle_larger_sin_non_right_triangle_tan_identity_not_necessarily_isosceles_l339_33988

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  sum_angles : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  law_of_sines : a / (Real.sin A) = b / (Real.sin B)

-- Define an acute triangle
def is_acute (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- Theorem A
theorem acute_triangle_sin_cos (t : Triangle) :
  is_acute t → Real.sin t.A > Real.cos t.B := by sorry

-- Theorem B
theorem larger_angle_larger_sin (t : Triangle) :
  t.A > t.B → Real.sin t.A > Real.sin t.B := by sorry

-- Define a non-right triangle
def is_non_right (t : Triangle) : Prop :=
  t.A ≠ Real.pi/2 ∧ t.B ≠ Real.pi/2 ∧ t.C ≠ Real.pi/2

-- Theorem C
theorem non_right_triangle_tan_identity (t : Triangle) :
  is_non_right t → Real.tan t.A + Real.tan t.B + Real.tan t.C = Real.tan t.A * Real.tan t.B * Real.tan t.C := by sorry

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

-- Theorem D (counter-example)
theorem not_necessarily_isosceles (t : Triangle) :
  t.a * Real.cos t.A = t.b * Real.cos t.B → ¬(is_isosceles t → True) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_sin_cos_larger_angle_larger_sin_non_right_triangle_tan_identity_not_necessarily_isosceles_l339_33988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curves_M_N_l339_33901

/-- The circle representing curve M -/
def curve_M (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 4

/-- The line representing curve N -/
def curve_N (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 16 = 0

/-- The minimum distance between a point on the circle and a point on the line -/
theorem min_distance_curves_M_N :
  ∃ (d : ℝ), d = 5 ∧
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    curve_M x₁ y₁ → curve_N x₂ y₂ →
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) ≥ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_curves_M_N_l339_33901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_variance_more_stable_l339_33967

-- Define a data set as a list of real numbers
def DataSet := List ℝ

-- Define the mean of a data set
noncomputable def mean (data : DataSet) : ℝ := (data.sum) / data.length

-- Define the variance of a data set
noncomputable def variance (data : DataSet) : ℝ :=
  (data.map (λ x => (x - mean data) ^ 2)).sum / data.length

-- Define stability as the inverse of variance
noncomputable def stability (data : DataSet) : ℝ := 1 / variance data

-- Theorem statement
theorem smaller_variance_more_stable
  (data1 data2 : DataSet)
  (h1 : mean data1 = mean data2)
  (h2 : variance data1 < variance data2) :
  stability data1 > stability data2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_variance_more_stable_l339_33967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l339_33953

-- Define a Point type for 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a distance function between two points
noncomputable def distance (p q : Point) : ℝ := Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- State the theorem
theorem distance_bounds (A B C : Point) 
  (h1 : distance A C = 3) 
  (h2 : distance B C = 4) : 
  1 ≤ distance A B ∧ distance A B ≤ 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bounds_l339_33953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_logs_l339_33937

theorem sum_of_logs (a b : ℝ) (h1 : (10 : ℝ)^a = 5) (h2 : b = Real.log 2 / Real.log 10) : 
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_logs_l339_33937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_price_is_15_l339_33900

/-- Represents the price of a bracelet in dollars -/
def bracelet_price : ℚ := 0  -- Initialize with a default value

/-- Represents the price of a necklace in dollars -/
def necklace_price : ℚ := 25

/-- Represents the price of a pair of earrings in dollars -/
def earrings_price : ℚ := 10

/-- Represents the price of a complete jewelry ensemble in dollars -/
def ensemble_price : ℚ := 45

/-- Represents the number of necklaces sold -/
def necklaces_sold : ℕ := 5

/-- Represents the number of bracelets sold -/
def bracelets_sold : ℕ := 10

/-- Represents the number of earrings sold -/
def earrings_sold : ℕ := 20

/-- Represents the number of ensembles sold -/
def ensembles_sold : ℕ := 2

/-- Represents the total sales in dollars -/
def total_sales : ℚ := 565

theorem bracelet_price_is_15 :
  necklace_price * necklaces_sold +
  bracelet_price * bracelets_sold +
  earrings_price * earrings_sold +
  ensemble_price * ensembles_sold = total_sales →
  bracelet_price = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bracelet_price_is_15_l339_33900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_special_sequence_length_l339_33919

/-- A strictly increasing sequence of positive integers starting with 2015 -/
def SpecialSequence (a : ℕ → ℕ) (k : ℕ) : Prop :=
  a 1 = 2015 ∧
  (∀ i j, 1 ≤ i → i < j → j ≤ k → a i < a j) ∧
  (∀ m n, 1 ≤ m → m ≤ k → 1 ≤ n → n ≤ k → 
    (a m : ℤ) + (a n : ℤ) ≥ (a (m + n) : ℤ) + (|m - n| : ℤ))

/-- The maximum length of the special sequence is 2016 -/
theorem max_special_sequence_length :
  ∀ a k, SpecialSequence a k → k ≤ 2016 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_special_sequence_length_l339_33919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l339_33932

def num_boys : ℕ := 4
def num_girls : ℕ := 3
def total_people : ℕ := num_boys + num_girls

def arrangements_girls_together : ℕ := Nat.factorial total_people / Nat.factorial num_girls

def arrangements_no_adjacent_girls : ℕ := Nat.factorial num_boys * Nat.choose (num_boys + 1) num_girls

def arrangements_three_between : ℕ := 2 * Nat.choose (total_people - 2) 3 * Nat.factorial (total_people - 2)

theorem arrangement_counts :
  arrangements_girls_together = 720 ∧
  arrangements_no_adjacent_girls = 1440 ∧
  arrangements_three_between = 720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangement_counts_l339_33932
