import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l187_18705

-- Define the equation
noncomputable def f (x a : ℝ) : ℝ := 2 * (1/4)^(-x) - (1/2)^(-x) + a

-- Theorem statement
theorem range_of_a :
  ∀ a : ℝ, (∃ x ∈ Set.Icc (-1 : ℝ) 0, f x a = 0) → a ∈ Set.Icc (-1 : ℝ) 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l187_18705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_three_way_handshake_l187_18758

/-- Represents a meeting with representatives and their handshakes. -/
structure Meeting (m : ℕ) where
  handshakes : Fin (3 * m) → Fin (3 * m) → Bool

/-- A meeting is n-interesting if there exist n representatives with handshake counts 1 to n. -/
def is_n_interesting (n : ℕ) (meeting : Meeting m) : Prop :=
  ∃ (reps : Fin n → Fin (3 * m)), ∀ i : Fin n, 
    (Finset.univ.filter (λ j ↦ meeting.handshakes (reps i) j)).card = i.val + 1

/-- Three representatives have all shaken hands with each other. -/
def has_three_way_handshake (meeting : Meeting m) : Prop :=
  ∃ (a b c : Fin (3 * m)), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    meeting.handshakes a b ∧ meeting.handshakes b c ∧ meeting.handshakes a c

/-- The main theorem to be proved. -/
theorem smallest_n_with_three_way_handshake (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, n ≤ 3 * m - 1 → n < 2 * m + 1 → 
    ∃ meeting : Meeting m, is_n_interesting n meeting ∧ ¬has_three_way_handshake meeting) ∧
  (∀ meeting : Meeting m, is_n_interesting (2 * m + 1) meeting → has_three_way_handshake meeting) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_three_way_handshake_l187_18758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l187_18727

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (Finset.range n).sum (λ i => a (i + 1)) + a 1 = 2 * a n

def arithmetic_property (a : ℕ → ℝ) : Prop :=
  a 1 + a 3 = 2 * (a 2 + 1)

theorem sequence_sum_theorem (a : ℕ → ℝ) 
  (h1 : sequence_property a) 
  (h2 : arithmetic_property a) : 
  a 1 + a 5 = 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_theorem_l187_18727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_proposition_l187_18734

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ P x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ P x) := by sorry

-- Define the specific proposition
def specific_P (x : ℝ) : Prop := (2 : ℝ)^x = 3

-- Theorem stating the negation of the specific proposition
theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, x ≥ 0 ∧ specific_P x) ↔ (∀ x : ℝ, x ≥ 0 → ¬ specific_P x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_proposition_l187_18734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l187_18791

theorem max_omega_value (f : ℝ → ℝ) (ω : ℕ) (T : ℝ) : 
  (∀ x, f x = 2 * Real.cos (ω * x + π / 6)) →
  (T > 0) →
  (T ∈ Set.Ioo 1 3) →
  (T = 2 * π / (ω : ℝ)) →
  (∀ n : ℕ, n > ω → 2 * π / (n : ℝ) ∉ Set.Ioo 1 3) →
  ω = 6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l187_18791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_points_sum_l187_18729

noncomputable section

/-- Ellipse C with parametric equations x = 2cos(α) and y = √3sin(α) -/
def Ellipse (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, Real.sqrt 3 * Real.sin α)

/-- Check if a point (x, y) is on the ellipse -/
def OnEllipse (p : ℝ × ℝ) : Prop :=
  (p.1 ^ 2) / 4 + (p.2 ^ 2) / 3 = 1

/-- Distance from origin to a point -/
noncomputable def DistanceFromOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

/-- Two points are perpendicular with respect to the origin -/
def Perpendicular (p q : ℝ × ℝ) : Prop :=
  p.1 * q.1 + p.2 * q.2 = 0

theorem ellipse_perpendicular_points_sum (A B : ℝ × ℝ) :
  OnEllipse A → OnEllipse B → Perpendicular A B →
  1 / (DistanceFromOrigin A) ^ 2 + 1 / (DistanceFromOrigin B) ^ 2 = 7 / 12 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_perpendicular_points_sum_l187_18729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_percent_of_sum_l187_18790

-- Define variables
variable (x y z w v : ℝ)

-- Define the conditions
def condition1 (x y z w v : ℝ) : Prop := 0.45 * z = 0.72 * y
def condition2 (x y z w v : ℝ) : Prop := y = 0.75 * x
def condition3 (x y z w v : ℝ) : Prop := w = 0.60 * z^2
def condition4 (x y z w v : ℝ) : Prop := z = 0.30 * w^(1/3)
def condition5 (x y z w v : ℝ) : Prop := v = 0.80 * Real.sqrt x

-- Define the theorem
theorem z_percent_of_sum 
  (h1 : condition1 x y z w v)
  (h2 : condition2 x y z w v)
  (h3 : condition3 x y z w v)
  (h4 : condition4 x y z w v)
  (h5 : condition5 x y z w v) :
  ∃ ε > 0, |((z / (x + v)) * 100) - 14.86| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_percent_of_sum_l187_18790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l187_18763

/-- Simple interest calculation function -/
noncomputable def simple_interest (principal rate time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Theorem stating that the rate is 4% given the conditions -/
theorem interest_rate_is_four_percent (principal : ℝ) (principal_positive : 0 < principal) :
  ∃ (rate : ℝ), simple_interest principal rate 10 = (2 / 5) * principal ∧ rate = 4 := by
  use 4
  constructor
  · -- Prove that simple_interest principal 4 10 = (2 / 5) * principal
    simp [simple_interest]
    field_simp
    ring
  · -- Prove that rate = 4
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_four_percent_l187_18763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_parts_expression_l187_18775

-- Define m as the integer part of √5
noncomputable def m : ℤ := ⌊Real.sqrt 5⌋

-- Define n as the fractional part of √5
noncomputable def n : ℝ := Real.sqrt 5 - m

-- Theorem statement
theorem sqrt_five_parts_expression :
  (m : ℝ) * (m - 1 / n)^3 = -10 * Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_five_parts_expression_l187_18775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_field_time_l187_18766

-- Define the side length of the square field in meters
noncomputable def side_length : ℝ := 60

-- Define the boy's speed in km/hr
noncomputable def speed_km_hr : ℝ := 12

-- Define the perimeter of the square field
noncomputable def perimeter : ℝ := 4 * side_length

-- Convert speed from km/hr to m/s
noncomputable def speed_m_s : ℝ := speed_km_hr * (1000 / 3600)

-- Calculate the time taken to run around the field
noncomputable def time_taken : ℝ := perimeter / speed_m_s

-- Theorem statement
theorem run_around_field_time :
  ∃ ε > 0, |time_taken - 72| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_run_around_field_time_l187_18766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_trees_problem_l187_18739

/-- The number of trees in Jim's garden over time -/
def TreeCount (x : ℕ) : ℕ → ℕ
| 0 => 2 * x  -- Initial number of trees
| n + 1 => if n < 5 then TreeCount x n + x else 2 * TreeCount x n  -- Yearly growth and doubling

/-- The problem statement -/
theorem jim_trees_problem : 
  ∃ (x : ℕ),
  (∃ (initial_rows : ℕ), initial_rows = 2) ∧  -- Jim starts with 2 rows
  (∀ (n : ℕ), n ≥ 10 ∧ n < 15 → TreeCount x n = TreeCount x (n-1) + x) ∧  -- New row every year from 10 to 14
  (TreeCount x 15 = 2 * TreeCount x 14) ∧  -- Doubling on 15th birthday
  (TreeCount x 15 = 56) ∧  -- Final count is 56
  x = 4  -- Prove that x (initial trees per row) is 4
:= by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jim_trees_problem_l187_18739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l187_18710

/-- Definition of an isosceles triangle -/
def IsIsosceles (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

/-- An isosceles triangle with side lengths 3, 6, and 6 has a perimeter of 15 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 3 → b = 6 → c = 6 →
  IsIsosceles a b c →
  a + b + c = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_perimeter_l187_18710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_hexagons_balanced_l187_18774

/-- A hexagonal grid filled with numbers 1, 2, and 3 -/
def HexGrid := ℕ → ℕ → Fin 3

/-- Three numbers are balanced if they are all the same or all different -/
def balanced (a b c : Fin 3) : Prop :=
  (a = b ∧ b = c) ∨ (a ≠ b ∧ b ≠ c ∧ a ≠ c)

/-- The grid satisfies the balance property for adjacent hexagons -/
def satisfies_balance_property (grid : HexGrid) : Prop :=
  ∀ i j, balanced (grid i j) (grid (i+1) j) (grid (i+1) (j+1))

/-- The three shaded hexagons in the grid -/
def shaded_hexagons (grid : HexGrid) : Fin 3 × Fin 3 × Fin 3 :=
  (grid 9 0, grid 0 0, grid 0 9)

theorem shaded_hexagons_balanced (grid : HexGrid) 
  (h : satisfies_balance_property grid) : 
  let (a, b, c) := shaded_hexagons grid
  balanced a b c := by
  sorry

#check shaded_hexagons_balanced

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_hexagons_balanced_l187_18774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_about_proofs_l187_18768

-- Define type for proof statements
structure ProofStatement where
  (statement : Prop)

-- Define the statements about proofs in geometry
def statement_a : Prop := ∃ s : ProofStatement, (¬ ∃ p : ProofStatement, p.statement → s.statement) ∧ s.statement

def statement_b : Prop := ∃ prop : ProofStatement, ∃ order1 order2 : ProofStatement → ProofStatement, 
  order1 ≠ order2 ∧ (order1 prop).statement ∧ (order2 prop).statement

def statement_c : Prop := ∀ (t : String) (p : ProofStatement), 
  (∃ uses : ProofStatement → String → Prop, uses p t) → 
  (∃ defined : String → ProofStatement → Prop, defined t p)

def statement_d : Prop := ∀ (p c : ProofStatement), 
  (∃ contains : ProofStatement → Prop, contains p) → 
  ¬((p.statement → c.statement) → c.statement)

-- Define the statement to be proved false
def statement_e : Prop := ∀ (p1 p2 : ProofStatement), 
  (∃ contrary : ProofStatement → ProofStatement → Prop, contrary p1 p2) → 
  ∃ ip : ProofStatement, ip.statement → (p1.statement ∧ p2.statement)

-- Theorem statement
theorem incorrect_statement_about_proofs :
  statement_a ∧ statement_b ∧ statement_c ∧ statement_d →
  ¬ statement_e :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_statement_about_proofs_l187_18768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_implies_k_range_l187_18787

/-- A function f that maps real numbers to real numbers. -/
noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

/-- The property of f being non-monotonic on an open interval. -/
def is_non_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

/-- The theorem stating the range of k given the non-monotonicity of f on (k-2, k+1). -/
theorem non_monotonic_interval_implies_k_range :
  ∀ k : ℝ, is_non_monotonic f (k - 2) (k + 1) → 2 ≤ k ∧ k < 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_interval_implies_k_range_l187_18787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_proposition_l187_18780

theorem negation_of_existence (p : ℝ → Prop) : (¬ ∃ x, p x) ↔ ∀ x, ¬p x := by sorry

theorem negation_of_specific_proposition :
  (¬ ∃ x : ℝ, (2 : ℝ)^x = 1) ↔ (∀ x : ℝ, (2 : ℝ)^x ≠ 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_existence_negation_of_specific_proposition_l187_18780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l187_18741

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define the midpoint of AB
def midpoint_AB : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem line_equation_proof :
  ∀ (A B : ℝ × ℝ),
  my_circle A.1 A.2 ∧ my_circle B.1 B.2 ∧  -- A and B are on the circle
  (A.1 + B.1) / 2 = midpoint_AB.1 ∧ (A.2 + B.2) / 2 = midpoint_AB.2 →  -- Midpoint condition
  ∀ (x y : ℝ), line_l x y ↔ (y - A.2) / (x - A.1) = (B.2 - A.2) / (B.1 - A.1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l187_18741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_equals_three_halves_l187_18726

-- Define f(2) as a constant
noncomputable def f_2 : ℝ := 1/2

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 - f_2 * Real.log x / Real.log 2

-- State the theorem
theorem f_one_half_equals_three_halves :
  f (1/2) = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_half_equals_three_halves_l187_18726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_cube_surface_area_l187_18728

/-- Represents a three-dimensional cube -/
structure Cube where
  side_length : ℝ
  surface_area : ℝ := 6 * side_length ^ 2

/-- Represents a three-dimensional sphere -/
structure Sphere where
  radius : ℝ

/-- Defines what it means for a sphere to be inscribed in a cube -/
def Sphere.inscribed_in (s : Sphere) (c : Cube) : Prop :=
  s.radius * 2 = c.side_length

/-- Defines what it means for a cube to be inscribed in a sphere -/
def Cube.inscribed_in (c : Cube) (s : Sphere) : Prop :=
  c.side_length * Real.sqrt 2 = s.radius * 2

/-- Given a cube with a sphere inscribed within it, and a second cube inscribed within that sphere,
    this theorem proves that if the outer cube has a surface area of 54 square meters,
    then the inner cube has a surface area of 27 square meters. -/
theorem inner_cube_surface_area 
  (outer_cube : Cube) 
  (sphere : Sphere) 
  (inner_cube : Cube) 
  (h1 : sphere.inscribed_in outer_cube)
  (h2 : inner_cube.inscribed_in sphere)
  (h3 : outer_cube.surface_area = 54) :
  inner_cube.surface_area = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inner_cube_surface_area_l187_18728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_theorem_l187_18712

noncomputable def candy_mixture (first_candy_weight : ℝ) (first_candy_price : ℝ) 
                  (total_mixture_weight : ℝ) (mixture_price : ℝ) : ℝ :=
  let second_candy_weight := total_mixture_weight - first_candy_weight
  let total_mixture_cost := mixture_price * total_mixture_weight
  let first_candy_cost := first_candy_weight * first_candy_price
  (total_mixture_cost - first_candy_cost) / second_candy_weight

theorem candy_mixture_theorem :
  candy_mixture 30 8 90 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_mixture_theorem_l187_18712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l187_18707

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (4 - a/2) * x + 2 else a^x

theorem increasing_function_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ 4 ≤ a ∧ a < 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_function_condition_l187_18707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_injective_function_characterization_l187_18735

def IsInjective (f : ℕ+ → ℕ+) : Prop :=
  ∀ x y : ℕ+, f x = f y → x = y

def IsMultiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

def DivisibilityCondition (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, (f (m^2 + n^2) : ℤ) ∣ (f (m^2) + f (n^2) : ℤ)

theorem injective_function_characterization (f : ℕ+ → ℕ+) 
  (h_inj : IsInjective f) 
  (h_mult : IsMultiplicative f) 
  (h_div : DivisibilityCondition f) :
  ∀ n : ℕ+, f n = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_injective_function_characterization_l187_18735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l187_18794

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x

-- Define the point of tangency
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem tangent_triangle_area :
  let slope : ℝ := (deriv f) point.1
  let tangent_line (x : ℝ) : ℝ := slope * (x - point.1) + point.2
  let x_intercept : ℝ := -point.2 / slope + point.1
  let y_intercept : ℝ := tangent_line 0
  (1/2) * x_intercept * (-y_intercept) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l187_18794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l187_18719

-- Define the function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + 1) / Real.log 0.5

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → 
  (a ≤ -2 ∨ a ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l187_18719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_30_60_90_triangle_l187_18725

/-- Represents a triangle -/
structure Triangle where
  -- Add necessary fields
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is a 30-60-90 triangle -/
def Triangle.is_30_60_90_triangle (t : Triangle) : Prop := sorry

/-- Represents a line segment -/
structure Segment where
  length : ℝ

/-- Represents the median to the hypotenuse of a triangle -/
def Triangle.median_to_hypotenuse (t : Triangle) : Segment := sorry

/-- Represents the shortest side of a triangle -/
def Triangle.shortest_side (t : Triangle) : Segment := sorry

/-- A 30-60-90 triangle with a median to the hypotenuse of length 15 units has a shortest side of length 7.5 units. -/
theorem shortest_side_length_30_60_90_triangle (t : Triangle) (h1 : t.is_30_60_90_triangle) 
  (h2 : t.median_to_hypotenuse.length = 15) : t.shortest_side.length = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_30_60_90_triangle_l187_18725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l187_18755

def A : Set ℝ := {x | (1/2 : ℝ) < (2 : ℝ)^x ∧ (2 : ℝ)^x < 8}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ B m → x ∈ A) ∧ (∃ x, x ∈ A ∧ x ∉ B m) ↔ m > 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l187_18755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_l187_18744

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - Real.pi / 2) * Real.cos (3 * Real.pi / 2 + α) * Real.tan (Real.pi - α)) /
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi))

theorem f_simplification (α : Real) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  f α = -Real.cos α := by sorry

theorem f_value (α : Real) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) : 
  f α = -(2 * Real.sqrt 6) / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_value_l187_18744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_privateer_overtakes_merchantman_l187_18747

/-- Represents the chase between a privateer and a merchantman --/
structure ChaseScenario where
  initial_distance : ℝ
  initial_privateer_speed : ℝ
  initial_merchantman_speed : ℝ
  damage_time : ℝ
  post_damage_privateer_distance : ℝ
  post_damage_merchantman_distance : ℝ

/-- Calculates the time when the privateer overtakes the merchantman --/
noncomputable def overtake_time (scenario : ChaseScenario) : ℝ :=
  let initial_relative_speed := scenario.initial_privateer_speed - scenario.initial_merchantman_speed
  let initial_chase_distance := initial_relative_speed * scenario.damage_time
  let remaining_distance := scenario.initial_distance - initial_chase_distance
  let post_damage_relative_speed := scenario.post_damage_privateer_distance / scenario.damage_time - scenario.initial_merchantman_speed
  scenario.damage_time + remaining_distance / post_damage_relative_speed

/-- Theorem stating that the privateer overtakes the merchantman at approximately 4:57 a.m. the next day --/
theorem privateer_overtakes_merchantman (scenario : ChaseScenario) 
  (h1 : scenario.initial_distance = 15)
  (h2 : scenario.initial_privateer_speed = 13)
  (h3 : scenario.initial_merchantman_speed = 9)
  (h4 : scenario.damage_time = 1.5)
  (h5 : scenario.post_damage_privateer_distance = 14)
  (h6 : scenario.post_damage_merchantman_distance = 12) :
  ∃ ε > 0, |overtake_time scenario - 16.95| < ε := by
  sorry

#eval (16.95 : Float)  -- This will output the time in hours since noon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_privateer_overtakes_merchantman_l187_18747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_B_eq_B_l187_18731

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x + 3)}
def B : Set ℝ := {x | x ≥ 2}

-- State the theorem
theorem A_inter_B_eq_B : A ∩ B = B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_inter_B_eq_B_l187_18731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_angle_satisfies_equation_l187_18781

theorem no_angle_satisfies_equation : ∀ α : Real, Real.sin α * Real.cos α ≠ Real.sin (40 * Real.pi / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_angle_satisfies_equation_l187_18781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_class_is_music_l187_18749

-- Define the list of subjects
inductive Subject
| Maths
| History
| Geography
| Science
| Music
deriving BEq, Repr

-- Define the schedule
def schedule : List Subject :=
  [Subject.Maths, Subject.History, Subject.Geography, Subject.Science, Subject.Music]

-- Define the current state
def current_state : Nat × Subject := (4, Subject.Science)

-- Theorem to prove
theorem next_class_is_music :
  let (current_time, current_subject) := current_state
  let next_subject := schedule[schedule.indexOf current_subject + 1]?
  next_subject = some Subject.Music := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_class_is_music_l187_18749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_second_container_l187_18732

/-- Represents a cylindrical container -/
structure Container where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylindrical container -/
noncomputable def volume (c : Container) : ℝ := Real.pi * c.radius^2 * c.height

/-- Given two containers and the price of the first, calculates the price of the second
    assuming price is proportional to volume -/
noncomputable def price_by_volume (c1 c2 : Container) (p1 : ℝ) : ℝ :=
  p1 * (volume c2 / volume c1)

theorem price_of_second_container :
  let c1 : Container := { radius := 2, height := 5 }
  let c2 : Container := { radius := 2, height := 10 }
  let p1 : ℝ := 1.5
  price_by_volume c1 c2 p1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_second_container_l187_18732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2008_l187_18750

/-- Defines the sequence v_n as described in the problem -/
def v : ℕ → ℕ := sorry

/-- The last term in a group with n terms -/
def f (n : ℕ) : ℕ := (5 * n^2 + 9 * n) / 2

/-- The total number of terms up to group n -/
def total_terms (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The group number containing the 2008th term -/
def group_2008 : ℕ := 62

/-- The first term after the group containing the 1953rd term -/
def first_term_after_1953 : ℕ := f group_2008

/-- The 2008th term of the sequence -/
theorem v_2008 : v 2008 = 10109 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_2008_l187_18750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l187_18733

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := (1/2) * x^2 + a * x + a - 1/2
def g (x : ℝ) : ℝ := a * Real.log (x + 1)

def common_point_and_tangent : Prop :=
  ∃ x > -1, f a x = g a x ∧ (deriv (f a)) x = (deriv (g a)) x

def two_common_points : Prop :=
  ∃ x y, x > -1 ∧ y > -1 ∧ x ≠ y ∧ f a x = g a x ∧ f a y = g a y

theorem functions_properties (h : a < 1) :
  (common_point_and_tangent a → a = 1/2) ∧
  (two_common_points a → 0 < a ∧ a < 1/2) :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_functions_properties_l187_18733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_plus_d_equals_four_point_five_l187_18760

-- Define the piecewise function g
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 3 then c * x + d else 10 - 2 * x

-- State the theorem
theorem c_plus_d_equals_four_point_five (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) → c + d = 4.5 := by
  intro h
  -- The proof steps would go here
  sorry

#check c_plus_d_equals_four_point_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_plus_d_equals_four_point_five_l187_18760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_card_selections_eq_6423_l187_18704

/-- The number of cards of each color -/
def num_cards : ℕ := 10

/-- The maximum power of 3 on the cards -/
def max_power : ℕ := 9

/-- The function f(n) represents the number of ways to select cards summing to n -/
def f (n : ℕ) : ℕ := sorry

/-- The sum P of f(n) from 1 to 1000 -/
def P : ℕ := Finset.sum (Finset.range 1001) f

/-- Theorem stating that P equals 6423 -/
theorem sum_of_card_selections_eq_6423 : P = 6423 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_card_selections_eq_6423_l187_18704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triangle_configurations_l187_18713

/-- Represents a configuration of wireframe triangles in space. -/
structure TriangleConfiguration where
  k : ℕ  -- number of triangles
  p : ℕ  -- number of triangles meeting at each vertex

/-- Represents a vertex in the configuration. -/
inductive Vertex

/-- Represents a triangle in the configuration. -/
inductive Triangle

/-- Checks if two triangles share a vertex. -/
def shareVertex (t1 t2 : Triangle) (v : Vertex) : Prop := sorry

/-- Counts the number of triangles meeting at a vertex. -/
def countTrianglesAtVertex (v : Vertex) (config : TriangleConfiguration) : ℕ := sorry

/-- Checks if a given configuration is valid according to the problem conditions. -/
def isValidConfiguration (config : TriangleConfiguration) : Prop :=
  -- Each pair of triangles shares exactly one vertex
  ∀ t1 t2 : Triangle, t1 ≠ t2 → ∃! v : Vertex, shareVertex t1 t2 v
  ∧
  -- p triangles meet at each vertex
  ∀ v : Vertex, countTrianglesAtVertex v config = config.p
  ∧
  -- k and p satisfy the relationship k = 3p - 2 for p ≥ 2
  (config.p ≥ 2 → config.k = 3 * config.p - 2)

/-- The theorem stating the only valid configurations. -/
theorem valid_triangle_configurations :
  ∀ config : TriangleConfiguration,
    isValidConfiguration config ↔
      (config.k = 1 ∧ config.p = 1) ∨
      (config.k = 4 ∧ config.p = 2) ∨
      (config.k = 7 ∧ config.p = 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triangle_configurations_l187_18713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_power_comparison_l187_18743

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem odd_numbers_power_comparison :
  let odd_numbers := {n : ℕ | 1 ≤ n ∧ n < 10000 ∧ n % 2 = 1}
  let greater_count := Finset.filter (λ n => last_four_digits (n^9) > n) (Finset.filter (λ n => n ∈ odd_numbers) (Finset.range 10000))
  let lesser_count := Finset.filter (λ n => last_four_digits (n^9) < n) (Finset.filter (λ n => n ∈ odd_numbers) (Finset.range 10000))
  greater_count.card = lesser_count.card := by
  sorry

#check odd_numbers_power_comparison

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_power_comparison_l187_18743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l187_18748

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 = 9

-- Define point P
def P : ℝ × ℝ := (5, -1)

-- Define the trajectory of midpoint Q
def trajectory (x y : ℝ) : Prop := (2*x - 8)^2 + (2*y - 2)^2 = 9

-- Define the line l
def line_l (x y : ℝ) : Prop := 3*x + 4*y - 11 = 0 ∨ x = 5

-- Theorem statement
theorem circle_and_line_problem :
  ∀ x y : ℝ,
  (∃ a b : ℝ, circle_C a b ∧ x = (a + 5) / 2 ∧ y = (b - 1) / 2) →
  trajectory x y ∧
  (∃ x₀ y₀ x₁ y₁ x₂ y₂ : ℝ,
    circle_C x₀ y₀ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l x₀ y₀ ∧ line_l x₁ y₁ ∧ line_l x₂ y₂ ∧
    (x₀, y₀) ≠ (x₁, y₁) ∧ (x₁, y₁) ≠ (x₂, y₂) ∧ (x₀, y₀) ≠ (x₂, y₂) ∧
    (∀ x₃ y₃ : ℝ, circle_C x₃ y₃ ∧ line_l x₃ y₃ →
      (x₃, y₃) = (x₀, y₀) ∨ (x₃, y₃) = (x₁, y₁) ∨ (x₃, y₃) = (x₂, y₂))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_problem_l187_18748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l187_18708

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b*cos(C) + c*cos(B) = a*sin(A), then the triangle is right-angled at A. -/
theorem triangle_right_angled (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos C + c * Real.cos B = a * Real.sin A →
  A = π / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angled_l187_18708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l187_18711

noncomputable def f (t : ℝ) : ℝ := (4 * t^2) / (1 + 4 * t^2)

theorem system_solution :
  ∀ x y z : ℝ,
  (f x = y ∧ f y = z ∧ f z = x) →
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l187_18711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_fixed_point_l187_18738

-- Define the function f(x) = a^(x-2) + 3
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) + 3

-- State the theorem
theorem exponential_function_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 4 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_fixed_point_l187_18738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l187_18782

theorem infinite_sum_equality (x : ℝ) (h : x > 1) :
  (∑' n : ℕ, 1 / (x^(2^n) - (1/x)^(2^n))) = 1 / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_sum_equality_l187_18782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_assignment_count_l187_18714

def number_of_ways_to_assign_grades (n : ℕ) (k : ℕ) : ℕ := k^n

theorem grade_assignment_count (n : ℕ) (k : ℕ) : 
  number_of_ways_to_assign_grades n k = k^n :=
by
  rfl

#eval number_of_ways_to_assign_grades 12 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_assignment_count_l187_18714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_invariant_under_constant_subtraction_original_variance_equals_new_variance_l187_18730

/-- Given a list of real numbers, calculate its variance -/
noncomputable def variance (data : List ℝ) : ℝ :=
  let mean := data.sum / data.length
  (data.map (λ x => (x - mean)^2)).sum / data.length

/-- Create a new list by subtracting a constant from each element of the original list -/
def subtractConstant (data : List ℝ) (c : ℝ) : List ℝ :=
  data.map (λ x => x - c)

/-- Theorem stating that subtracting a constant from each data point does not change the variance -/
theorem variance_invariant_under_constant_subtraction (data : List ℝ) (c : ℝ) :
  variance (subtractConstant data c) = variance data :=
by sorry

/-- Corollary: If the variance of the new set (after subtracting 100) is 4, 
    then the variance of the original set is also 4 -/
theorem original_variance_equals_new_variance 
  (data : List ℝ) (h : variance (subtractConstant data 100) = 4) :
  variance data = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_invariant_under_constant_subtraction_original_variance_equals_new_variance_l187_18730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_terms_equals_negative_360_l187_18754

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a 1 + (n - 1 : ℝ) * (a 2 - a 1))

theorem sum_of_eight_terms_equals_negative_360
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a2 : a 2 = 18 - a 1) :
  sum_of_arithmetic_sequence a 8 = -360 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_terms_equals_negative_360_l187_18754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_perpendicular_implies_side_length_l187_18778

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the medians
noncomputable def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ := 
  let midpoint := (
    (t.A.1 + t.B.1 + t.C.1 - v.1) / 2,
    (t.A.2 + t.B.2 + t.C.2 - v.2) / 2
  )
  ((v.1 + midpoint.1) / 2, (v.2 + midpoint.2) / 2)

-- Define perpendicularity of vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define the length of a side
noncomputable def side_length (a b : ℝ × ℝ) : ℝ :=
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- State the theorem
theorem median_perpendicular_implies_side_length 
  (t : Triangle) 
  (h1 : perpendicular (median t t.A) (median t t.B))
  (h2 : side_length t.A t.C = 6)
  (h3 : side_length t.B t.C = 7) :
  side_length t.A t.B = Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_perpendicular_implies_side_length_l187_18778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l187_18788

theorem problem_solution (x : ℝ) : (5 : ℝ) * 1.25 * (12 : ℝ) * 0.25 * x^(3/4) = 300 ↔ x = 32 * (2^(1/3 : ℝ)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l187_18788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_permutation_l187_18745

/-- A permutation of digits 1 to 9 -/
def Permutation := Fin 9 → Fin 9

/-- The property that there is an odd number of digits between consecutive pairs -/
def HasOddGaps (p : Permutation) : Prop :=
  ∀ i : Fin 8, ∃ k : ℕ, 
    p (Fin.succ i) - p i = 2 * k + 1 ∨
    p i - p (Fin.succ i) = 2 * k + 1

/-- Theorem stating the impossibility of the arrangement -/
theorem no_valid_permutation : ¬ ∃ p : Permutation, Function.Bijective p ∧ HasOddGaps p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_valid_permutation_l187_18745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_A_l187_18746

def A : Finset ℕ := {2, 4, 5}

theorem number_of_subsets_of_A : (Finset.powerset A).card = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_of_A_l187_18746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cabbage_price_is_two_l187_18721

/-- Represents the earnings and sales data for Johannes' vegetable shop --/
structure VegetableShopData where
  wednesday_earnings : ℚ
  friday_earnings : ℚ
  today_earnings : ℚ
  total_kg_sold : ℚ

/-- Calculates the price per kilogram of cabbage --/
def price_per_kg (data : VegetableShopData) : ℚ :=
  (data.wednesday_earnings + data.friday_earnings + data.today_earnings) / data.total_kg_sold

/-- Theorem stating that the price per kilogram of cabbage is $2 --/
theorem cabbage_price_is_two (data : VegetableShopData)
  (h1 : data.wednesday_earnings = 30)
  (h2 : data.friday_earnings = 24)
  (h3 : data.today_earnings = 42)
  (h4 : data.total_kg_sold = 48) :
  price_per_kg data = 2 := by
  sorry

#eval price_per_kg {
  wednesday_earnings := 30,
  friday_earnings := 24,
  today_earnings := 42,
  total_kg_sold := 48
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cabbage_price_is_two_l187_18721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l187_18770

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (3*x^2 + 7*x - 9) / (x - 2)

/-- The slope of the slant asymptote -/
def m : ℝ := 3

/-- The y-intercept of the slant asymptote -/
def b : ℝ := 13

/-- The slant asymptote function -/
def asymptote (x : ℝ) : ℝ := m * x + b

theorem slant_asymptote_sum :
  m + b = 16 :=
by
  -- Unfold the definitions of m and b
  unfold m b
  -- Perform the addition
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_sum_l187_18770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_equals_rational_l187_18724

/-- The recurring decimal 0.125634125634... as a sequence of digits -/
def recurringDecimal (n : ℕ) : ℕ :=
  match n % 6 with
  | 0 => 1
  | 1 => 2
  | 2 => 5
  | 3 => 6
  | 4 => 3
  | 5 => 4
  | _ => 0  -- This case should never occur, but we need it for completeness

/-- The rational number 125634/999999 -/
def rationalNumber : ℚ := 125634 / 999999

/-- Theorem stating that the recurring decimal is equal to the rational number -/
theorem recurring_decimal_equals_rational :
  (∑' n, (recurringDecimal n : ℚ) / 10^(n+1)) = rationalNumber := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_equals_rational_l187_18724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_m_ab_rotation_is_circle_locus_m_cd_slide_is_circle_l187_18718

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the plane
def Point := ℝ × ℝ

-- Define the diameter AB
def DiameterAB (circle : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (circle.center.1 + t * circle.radius, circle.center.2)}

-- Define the chord CD
def ChordCD (circle : Circle) : Set ((ℝ × ℝ) × (ℝ × ℝ)) :=
  {(c, d) : (ℝ × ℝ) × (ℝ × ℝ) | (c.1 - circle.center.1)^2 + (c.2 - circle.center.2)^2 = circle.radius^2 ∧
                                 (d.1 - circle.center.1)^2 + (d.2 - circle.center.2)^2 = circle.radius^2}

-- Define the intersection point M
noncomputable def IntersectionM (a b c d : ℝ × ℝ) : ℝ × ℝ :=
  sorry  -- The actual calculation of the intersection point

-- Define the locus of M when AB rotates
def LocusMABRotation (circle : Circle) : Set (ℝ × ℝ) :=
  {m : ℝ × ℝ | ∃ (a b : ℝ × ℝ) (c d : ℝ × ℝ),
    a ∈ DiameterAB circle ∧
    b ∈ DiameterAB circle ∧
    (c, d) ∈ ChordCD circle ∧
    m = IntersectionM a b c d}

-- Define the locus of M when CD slides
def LocusMCDSlide (circle : Circle) : Set (ℝ × ℝ) :=
  {m : ℝ × ℝ | ∃ (a b : ℝ × ℝ) (c d : ℝ × ℝ),
    a ∈ DiameterAB circle ∧
    b ∈ DiameterAB circle ∧
    (c, d) ∈ ChordCD circle ∧
    m = IntersectionM a b c d}

-- State the theorems
theorem locus_m_ab_rotation_is_circle (circle : Circle) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), LocusMABRotation circle = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} :=
by sorry

theorem locus_m_cd_slide_is_circle (circle : Circle) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), LocusMCDSlide circle = {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_m_ab_rotation_is_circle_locus_m_cd_slide_is_circle_l187_18718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l187_18740

theorem sin_pi_minus_alpha (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = 4 / 5) :
  Real.sin (π - α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_minus_alpha_l187_18740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_length_l187_18759

/-- A rectangular solid with given total surface area and total edge length -/
structure RectangularSolid where
  a : ℝ
  b : ℝ
  c : ℝ
  total_surface_area : 2 * (a * b + a * c + b * c) = 48
  total_edge_length : 4 * (a + b + c) = 40

/-- The length of the interior diagonal of a rectangular solid -/
noncomputable def interior_diagonal (solid : RectangularSolid) : ℝ :=
  Real.sqrt (solid.a^2 + solid.b^2 + solid.c^2)

/-- Theorem: The interior diagonal of a rectangular solid with total surface area 48 and
    total edge length 40 is 2√13 -/
theorem interior_diagonal_length (solid : RectangularSolid) :
  interior_diagonal solid = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interior_diagonal_length_l187_18759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spongebob_burger_sales_l187_18757

/-- Represents the sales and earnings of Spongebob's burger shop for a day -/
structure BurgerShopSales where
  burger_price : ℚ
  fries_price : ℚ
  fries_quantity : ℕ
  total_earnings : ℚ

/-- Calculates the number of burgers sold given the day's sales information -/
def burgers_sold (sales : BurgerShopSales) : ℚ :=
  (sales.total_earnings - sales.fries_price * sales.fries_quantity) / sales.burger_price

/-- Theorem stating that Spongebob sold 30 burgers -/
theorem spongebob_burger_sales :
  let sales := BurgerShopSales.mk 2 (3/2) 12 78
  burgers_sold sales = 30 := by
  sorry

#eval burgers_sold (BurgerShopSales.mk 2 (3/2) 12 78)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spongebob_burger_sales_l187_18757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l187_18773

/-- The volume of a cylindrical pool -/
noncomputable def pool_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  (Real.pi / 4) * diameter^2 * depth

/-- Theorem: The volume of a cylindrical pool with diameter 20 feet and depth 5 feet is 500π cubic feet -/
theorem pool_volume_calculation :
  pool_volume 20 5 = 500 * Real.pi := by
  -- Unfold the definition of pool_volume
  unfold pool_volume
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pool_volume_calculation_l187_18773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coleen_sprinkle_usage_l187_18795

/-- Represents the number of cans of sprinkles Coleen used and remaining --/
structure SprinkleCans where
  initial : ℕ
  hair : ℕ
  clothes : ℕ
  pets : ℕ
  remaining : ℕ

/-- The conditions of Coleen's sprinkle usage --/
def coleen_sprinkles (h c p R : ℕ) : SprinkleCans where
  initial := 12
  hair := h
  clothes := c
  pets := p
  remaining := R

/-- The theorem stating the remaining cans and total used cans --/
theorem coleen_sprinkle_usage (h c p R : ℕ) :
  (coleen_sprinkles h c p R).initial - (h + c + p) = R ∧
  R = (coleen_sprinkles h c p R).initial / 2 - 3 →
  R = 3 ∧ h + c + p = 9 := by
  sorry

#check coleen_sprinkle_usage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coleen_sprinkle_usage_l187_18795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l187_18702

def a : ℝ × ℝ × ℝ := (2, 3, 1)
def b : ℝ × ℝ × ℝ := (-1, 0, -1)
def c : ℝ × ℝ × ℝ := (2, 2, 2)

theorem vectors_not_coplanar : ¬(∃ (x y z : ℝ), x • a + y • b + z • c = (0, 0, 0) ∧ (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l187_18702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bleach_volume_reduction_l187_18700

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  breadth : ℝ
  thickness : ℝ

/-- Calculates the volume of a rectangular object -/
def volume (d : Dimensions) : ℝ := d.length * d.breadth * d.thickness

/-- Applies the bleaching process to the dimensions -/
def bleach (d : Dimensions) : Dimensions :=
  { length := d.length * 0.75,
    breadth := d.breadth * 0.7,
    thickness := d.thickness * 0.9 }

/-- Theorem: Bleaching reduces volume by 52.75% -/
theorem bleach_volume_reduction (d : Dimensions) :
  (volume d - volume (bleach d)) / volume d = 0.5275 := by
  sorry

-- Remove the #eval statement as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bleach_volume_reduction_l187_18700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l187_18756

/-- Sequence a_n with given properties -/
noncomputable def sequence_a (n : ℕ) : ℝ := 3^(n-1)

/-- Sum of first n terms of sequence a_n -/
noncomputable def S (n : ℕ) : ℝ := (3^n - 1) / 2

/-- Constant c derived from the ratio of consecutive terms -/
def c : ℝ := 3

/-- Sequence b_n defined in terms of a_n -/
noncomputable def sequence_b (n : ℕ) : ℝ := sequence_a n * (Real.log (sequence_a n) / Real.log 3)

/-- Sum of first n terms of sequence b_n -/
noncomputable def T (n : ℕ) : ℝ := ((2*n - 3) * 3^n + 3) / 4

/-- Main theorem stating the properties of sequences a_n and b_n -/
theorem sequence_properties :
  (∀ n : ℕ, n > 0 → sequence_a n = 3^(n-1)) ∧
  (sequence_a 1 + sequence_a 2 = 4) ∧
  (∀ n : ℕ, n > 0 → (2 * S (n+1) + 1) / (2 * S n + 1) = c) ∧
  (c > 0) ∧
  (c = sequence_a 2 / sequence_a 1) ∧
  (∀ n : ℕ, n > 0 → sequence_b n = sequence_a n * (Real.log (sequence_a n) / Real.log 3)) ∧
  (∀ n : ℕ, n > 0 → T n = ((2*n - 3) * 3^n + 3) / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l187_18756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_female_percentage_after_hiring_l187_18769

/-- Represents the workforce of a company --/
structure Workforce where
  total : ℕ
  female : ℕ

/-- Calculates the percentage of female workers in the workforce --/
noncomputable def femalePercentage (w : Workforce) : ℝ :=
  (w.female : ℝ) / (w.total : ℝ) * 100

/-- Proves that the new percentage of female workers is approximately 54.86% --/
theorem new_female_percentage_after_hiring (initial : Workforce) 
  (h1 : femalePercentage initial = 60)
  (h2 : initial.total + 24 = 288) : 
  ∃ (final : Workforce), 
    final.total = 288 ∧ 
    final.female = initial.female ∧ 
    54.85 < femalePercentage final ∧ femalePercentage final < 54.87 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_female_percentage_after_hiring_l187_18769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_seating_l187_18796

/-- Represents a participant in the olympiad -/
structure Participant where
  id : Nat

/-- Represents the friendship relation between participants -/
def Friendship := Participant → Participant → Prop

/-- A clique is a set of participants who are all friends with each other -/
def IsClique (friends : Friendship) (clique : Set Participant) : Prop :=
  ∀ p q, p ∈ clique → q ∈ clique → p ≠ q → friends p q

/-- The size of a clique is the number of participants in it -/
noncomputable def CliqueSize (clique : Set Participant) : ℕ :=
  Nat.card clique

/-- The maximum clique size in a set of participants -/
noncomputable def MaxCliqueSize (friends : Friendship) (participants : Set Participant) : ℕ :=
  ⨆ (clique : Set Participant) (h : clique ⊆ participants ∧ IsClique friends clique), CliqueSize clique

theorem olympiad_seating 
  (participants : Set Participant) 
  (friends : Friendship) 
  (h_mutual : ∀ p q, friends p q ↔ friends q p)
  (h_even_max : Even (MaxCliqueSize friends participants)) :
  ∃ (room1 room2 : Set Participant),
    participants = room1 ∪ room2 ∧
    room1 ∩ room2 = ∅ ∧
    MaxCliqueSize friends room1 = MaxCliqueSize friends room2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_olympiad_seating_l187_18796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_for_identity_l187_18779

noncomputable def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

def is_identity (M : Matrix (Fin 2) (Fin 2) ℝ) : Prop :=
  M = !![1, 0; 0, 1]

theorem smallest_power_for_identity :
  ∀ n : ℕ, n > 0 → (is_identity (rotation_matrix ^ n) ↔ n ≥ 3) ∧
  (∀ m : ℕ, 0 < m → m < 3 → ¬is_identity (rotation_matrix ^ m)) := by
  sorry

#check smallest_power_for_identity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_power_for_identity_l187_18779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_people_round_table_with_pair_l187_18777

/-- The number of ways to seat n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to seat n people around a round table,
    with two people insisting on sitting next to each other. -/
def roundTableWithPair (n : ℕ) : ℕ :=
  2 * roundTableArrangements (n - 1)

theorem seven_people_round_table_with_pair :
  roundTableWithPair 7 = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seven_people_round_table_with_pair_l187_18777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_properties_l187_18797

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a parabola -/
structure Parabola where
  c : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

noncomputable def distance (p1 p2 : Point) : ℝ := Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

theorem ellipse_parabola_properties
  (C₁ : Ellipse) (C₂ : Parabola) 
  (F A B C D : Point) (l : Line)
  (h1 : F.x = C₂.c ∧ F.y = 0)  -- F is focus of C₂
  (h2 : l.m = 0 ∧ l.b = F.y)   -- l is perpendicular to x-axis through F
  (h3 : A.x = F.x ∧ B.x = F.x) -- A and B are on l
  (h4 : C.x = F.x ∧ D.x = F.x) -- C and D are on l
  (h5 : distance C D = 4/3 * distance A B)
  (h6 : (C₁.a + C₂.c) + (C₁.a - C₂.c) + (C₁.a + C₂.c) + (C₁.a - C₂.c) = 12) :
  eccentricity C₁ = 1/2 ∧ 
  C₁.a = 4 ∧ C₁.b = 2 * Real.sqrt 3 ∧
  C₂.c = 2 := by
  sorry

#check ellipse_parabola_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_parabola_properties_l187_18797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l187_18762

noncomputable def f (x : ℝ) : ℝ := 3 / (x - 2) - Real.sqrt (x + 1)

theorem f_domain : 
  Set.range f = {y | ∃ x : ℝ, x ≥ -1 ∧ x ≠ 2 ∧ y = f x} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l187_18762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sqrt_two_l187_18736

-- Define the function f
noncomputable def f (t : ℝ) : ℝ := 
  let x := (t - 1) / 2
  x^2 - 2*x

-- State the theorem
theorem f_sqrt_two : f (Real.sqrt 2) = (5 - 4 * Real.sqrt 2) / 4 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- The rest of the proof is omitted
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sqrt_two_l187_18736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_l187_18767

theorem hyperbola_focus_asymptote (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x : ℝ, x^2 = -12 * (-3)) →
  (∀ x y : ℝ, y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) →
  a = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focus_asymptote_l187_18767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l187_18722

open Real

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A > 0 ∧ A < Real.pi ∧ B > 0 ∧ B < Real.pi ∧ C > 0 ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the given condition
def condition (A B C : ℝ) : Prop :=
  2 * Real.sin A - Real.sqrt 3 * Real.cos C = (Real.sqrt 3 * Real.sin C) / Real.tan B

-- Theorem statement
theorem triangle_theorem (A B C : ℝ) (a b c : ℝ) 
  (h1 : triangle_ABC A B C a b c) (h2 : condition A B C) :
  (B = Real.pi/3 ∨ B = 2*Real.pi/3) ∧ 
  (B = Real.pi/3 → ∃ (x : ℝ), x > Real.sqrt 3 ∧ x ≤ 3*(Real.sqrt 3)/2 ∧ 
    Real.sin A + Real.sin B + Real.sin C = x) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l187_18722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_genuine_given_weight_condition_l187_18753

/-- Represents a coin which can be either genuine or counterfeit -/
inductive Coin
| Genuine
| Counterfeit

/-- The total number of coins -/
def total_coins : ℕ := 13

/-- The number of genuine coins -/
def genuine_coins : ℕ := 10

/-- The number of counterfeit coins -/
def counterfeit_coins : ℕ := 3

/-- A pair of coins -/
structure CoinPair :=
  (first : Coin)
  (second : Coin)

/-- Predicate to check if a coin pair is genuine -/
def is_genuine_pair (pair : CoinPair) : Prop :=
  pair.first = Coin.Genuine ∧ pair.second = Coin.Genuine

/-- Predicate to check if the first pair weighs less than the second pair -/
def first_pair_weighs_less (pair1 pair2 : CoinPair) : Prop :=
  (pair1.first = Coin.Genuine ∧ pair1.second = Coin.Genuine) ∨
  (pair2.first = Coin.Counterfeit ∨ pair2.second = Coin.Counterfeit)

/-- The main theorem to prove -/
theorem probability_all_genuine_given_weight_condition :
  let pair_selection := fun _ => CoinPair
  let first_selection := pair_selection total_coins
  let second_selection := pair_selection (total_coins - 2)
  ∀ (first_pair : first_selection) (second_pair : second_selection),
    first_pair_weighs_less first_pair second_pair →
    (5 : ℚ) / 6 = 
      (genuine_coins * (genuine_coins - 1) * (genuine_coins - 2) * (genuine_coins - 3) : ℚ) /
      (total_coins * (total_coins - 1) * (total_coins - 2) * (total_coins - 3) : ℚ) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_genuine_given_weight_condition_l187_18753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_width_is_15_l187_18703

/-- Represents the dimensions and properties of a cube immersed in a rectangular water vessel -/
structure CubeInVessel where
  cube_edge : ℝ
  vessel_length : ℝ
  water_rise : ℝ
  vessel_width : ℝ

/-- Calculates the width of the vessel's base given the cube and vessel properties -/
noncomputable def calculate_vessel_width (c : CubeInVessel) : ℝ :=
  (c.cube_edge ^ 3) / (c.vessel_length * c.water_rise)

/-- Theorem stating that for the given dimensions, the vessel width is 15 cm -/
theorem vessel_width_is_15 (c : CubeInVessel) 
  (h1 : c.cube_edge = 10)
  (h2 : c.vessel_length = 20)
  (h3 : c.water_rise = 3.3333333333333335)
  : calculate_vessel_width c = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vessel_width_is_15_l187_18703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l187_18715

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Define the base case for 0
  | n + 1 => sequence_a n - 2

theorem a_100_value : sequence_a 100 = -197 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l187_18715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_possible_l187_18720

theorem candy_distribution_possible : ∃ (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ),
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 40 ∧
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧
  a₅ ≠ a₆ ∧
  (∀ i j, i ≠ j → i < 7 → j < 7 → 
    (match (i, j) with
    | (1, 2) | (2, 1) => a₁ + a₂
    | (1, 3) | (3, 1) => a₁ + a₃
    | (1, 4) | (4, 1) => a₁ + a₄
    | (1, 5) | (5, 1) => a₁ + a₅
    | (1, 6) | (6, 1) => a₁ + a₆
    | (2, 3) | (3, 2) => a₂ + a₃
    | (2, 4) | (4, 2) => a₂ + a₄
    | (2, 5) | (5, 2) => a₂ + a₅
    | (2, 6) | (6, 2) => a₂ + a₆
    | (3, 4) | (4, 3) => a₃ + a₄
    | (3, 5) | (5, 3) => a₃ + a₅
    | (3, 6) | (6, 3) => a₃ + a₆
    | (4, 5) | (5, 4) => a₄ + a₅
    | (4, 6) | (6, 4) => a₄ + a₆
    | (5, 6) | (6, 5) => a₅ + a₆
    | _ => 0
    ) < 20) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_possible_l187_18720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_theorem_l187_18776

/-- Represents the contents of a can with milk and water -/
structure CanContents where
  milk : ℚ
  water : ℚ

/-- Calculates the total volume of the contents -/
def CanContents.total (c : CanContents) : ℚ := c.milk + c.water

/-- Calculates the ratio of milk to water -/
noncomputable def CanContents.ratio (c : CanContents) : ℚ := c.milk / c.water

theorem can_capacity_theorem (initial : CanContents) (final : CanContents) :
  CanContents.ratio initial = 5 / 3 →
  final.milk = initial.milk + 8 →
  final.water = initial.water →
  CanContents.ratio final = 2 →
  CanContents.total final = 72 := by
  sorry

#check can_capacity_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_can_capacity_theorem_l187_18776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_l187_18792

/-- Represents a rhombus with given area and one diagonal length -/
structure Rhombus where
  area : ℝ
  diagonal1 : ℝ

/-- Calculates the length of the second diagonal of a rhombus -/
noncomputable def second_diagonal (r : Rhombus) : ℝ := 2 * r.area / r.diagonal1

/-- Theorem: In a rhombus with area 126 cm² and one diagonal 18 cm, the other diagonal is 14 cm -/
theorem rhombus_second_diagonal :
  let r : Rhombus := { area := 126, diagonal1 := 18 }
  second_diagonal r = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_second_diagonal_l187_18792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_sum_l187_18701

def sequence_a : ℕ → ℝ := sorry
def sum_sequence_a : ℕ → ℝ := sorry

theorem geometric_sequence_and_sum 
  (h : ∀ n, 5 * sequence_a n = 2 * sum_sequence_a n + 1) :
  (∀ n, sequence_a n = (1/3) * (5/3)^(n-1)) ∧
  (∀ n, Finset.sum (Finset.range n) (λ m ↦ 3^m * sequence_a m - 1) = 5^n / 4 - n - 1/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_and_sum_l187_18701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_max_area_l187_18785

-- Define the circle
def circle_set (C : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1 - C.1)^2 + (P.2 - C.2)^2 = r^2}

-- Define the points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (3, 4)

-- Define the line on which C lies
def line (x y : ℝ) : Prop := x + 3*y - 15 = 0

-- Helper function to calculate triangle area
noncomputable def area_triangle (A B P : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem circle_and_max_area :
  ∃ (C : ℝ × ℝ) (r : ℝ),
    line C.1 C.2 ∧
    A ∈ circle_set C r ∧
    B ∈ circle_set C r ∧
    (∀ (x y : ℝ), (x+3)^2 + (y-6)^2 = 40 ↔ (x, y) ∈ circle_set C r) ∧
    (∃ (max_area : ℝ), max_area = 16 + 8 * Real.sqrt 5 ∧
      ∀ (P : ℝ × ℝ), P ∈ circle_set C r →
        area_triangle A B P ≤ max_area) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_max_area_l187_18785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_B_shorter_base_length_l187_18717

/-- Represents a trapezoid with its longer and shorter base lengths -/
structure Trapezoid where
  longerBase : ℝ
  shorterBase : ℝ

/-- The length of the line segment joining the midpoints of the diagonals of a trapezoid -/
noncomputable def midpointSegmentLength (t : Trapezoid) : ℝ :=
  (t.longerBase - t.shorterBase) / 2

theorem trapezoid_B_shorter_base_length :
  ∀ (A B : Trapezoid),
    A.longerBase = 105 →
    midpointSegmentLength A = 5 →
    B.longerBase = A.longerBase - 10 →
    midpointSegmentLength B = 5 →
    B.shorterBase = 85 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_B_shorter_base_length_l187_18717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bounds_existence_l187_18784

noncomputable def f (x : ℝ) := (3 * x + 4) / (x + 3)

def T : Set ℝ := {y | ∃ x : ℝ, x ≥ -2 ∧ f x = y}

theorem function_bounds_existence :
  (∃ P : ℝ, ∀ y ∈ T, y < P) ∧
  (∃ q : ℝ, ∀ y ∈ T, y ≥ q) ∧
  (∃ P : ℝ, ∀ y ∈ T, y < P ∧ P ∉ T) ∧
  (∃ q : ℝ, ∀ y ∈ T, y ≥ q ∧ q ∈ T) :=
by
  sorry

#check function_bounds_existence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_bounds_existence_l187_18784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l187_18771

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else a^x

-- State the theorem
theorem decreasing_f_implies_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1/6 : ℝ) (1/3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_decreasing_f_implies_a_range_l187_18771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_discount_theorem_l187_18798

theorem shirt_discount_theorem (original_price : ℝ) (h : original_price > 0) :
  let first_discount := 0.7
  let second_discount := 0.1
  let sale_price := original_price * first_discount
  let final_price := sale_price * (1 - second_discount)
  final_price / original_price = 0.63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_discount_theorem_l187_18798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l187_18706

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < (1/2 : ℝ) then x^2 - 4*x else Real.log (2*x + 1) / Real.log (1/2)

-- Theorem statement
theorem f_properties :
  (f (3/2) = -2) ∧
  (f (f (1/2)) = 5) ∧
  (∀ x : ℝ, f x > -3 ↔ x < 7/2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l187_18706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l187_18793

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem f_properties :
  let a : ℝ := -Real.pi / 6
  let b : ℝ := Real.pi / 4
  (f (Real.pi / 6) = 2) ∧
  (∀ x ∈ Set.Icc a b, f x ≤ 2) ∧
  (∀ x ∈ Set.Icc a b, f x ≥ -1) ∧
  (∃ x ∈ Set.Icc a b, f x = 2) ∧
  (∃ x ∈ Set.Icc a b, f x = -1) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l187_18793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_implies_t_bound_l187_18789

noncomputable def f (x : ℝ) : ℝ := |x * Real.exp x|

noncomputable def g (t : ℝ) (x : ℝ) : ℝ := (f x)^2 - t * (f x)

theorem four_solutions_implies_t_bound 
  (t : ℝ) 
  (h : ∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧ 
       g t x₁ = -1 ∧ g t x₂ = -1 ∧ g t x₃ = -1 ∧ g t x₄ = -1) : 
  t > Real.exp 1 + Real.exp (-1) := by
  sorry

#check four_solutions_implies_t_bound

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_implies_t_bound_l187_18789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sphere_radius_calculation_l187_18723

-- Define the side length of the large cube
noncomputable def largeCubeSideLength : ℝ := 3

-- Define the radius of the corner spheres (inscribed in unit cubes)
noncomputable def cornerSphereRadius : ℝ := 1 / 2

-- Define the function to calculate the radius of the tangent sphere
noncomputable def tangentSphereRadius : ℝ := (3 * Real.sqrt 3) / 2 - 1

-- Theorem statement
theorem tangent_sphere_radius_calculation :
  tangentSphereRadius = (largeCubeSideLength * Real.sqrt 3) / 2 - cornerSphereRadius - cornerSphereRadius :=
by
  -- Unfold the definitions
  unfold tangentSphereRadius largeCubeSideLength cornerSphereRadius
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_sphere_radius_calculation_l187_18723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_range_l187_18709

-- Define the curves C1 and C2
noncomputable def C1 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 + 2 * Real.sin α)

def C2 (θ : ℝ) (ρ : ℝ) : Prop := ρ * Real.sin θ ^ 2 = Real.cos θ

-- Define the ray l
def ray_l (α : ℝ) (ρ : ℝ) : Prop := ρ ≥ 0 ∧ α ∈ Set.Icc (Real.pi / 4) (Real.pi / 3)

-- Define the points A and B
noncomputable def point_A (α : ℝ) : ℝ := 4 * Real.sin α

noncomputable def point_B (α : ℝ) : ℝ := Real.cos α / Real.sin α ^ 2

-- State the theorem
theorem product_range :
  ∀ α, ray_l α (point_A α) →
       ray_l α (point_B α) →
       (point_A α) * (point_B α) ∈ Set.Icc (4 * Real.sqrt 3 / 3) 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_range_l187_18709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_side_relations_l187_18764

noncomputable section

variable (A B C a b c : ℝ)

/-- Vector m as defined in the problem -/
def m : ℝ × ℝ := (Real.sin B + Real.sin C, Real.sin A + Real.sin B)

/-- Vector n as defined in the problem -/
def n : ℝ × ℝ := (Real.sin B - Real.sin C, Real.sin A)

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem triangle_angle_and_side_relations 
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi) 
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_law_of_sines : a / Real.sin A = b / Real.sin B)
  (h_perpendicular : dot_product (m A B C) (n A B C) = 0) :
  (C = 2 * Real.pi / 3) ∧ 
  (c = Real.sqrt 3 → Real.sqrt 3 < 2 * a + b ∧ 2 * a + b < 2 * Real.sqrt 3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_and_side_relations_l187_18764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_sunday_hours_l187_18761

noncomputable def hours_worked_on_sunday (hourly_rate : ℝ) (friday_hours : ℝ) (saturday_hours : ℝ) (total_earnings : ℝ) : ℝ :=
  (total_earnings - hourly_rate * (friday_hours + saturday_hours)) / hourly_rate

theorem dana_sunday_hours :
  let hourly_rate : ℝ := 13
  let friday_hours : ℝ := 9
  let saturday_hours : ℝ := 10
  let total_earnings : ℝ := 286
  hours_worked_on_sunday hourly_rate friday_hours saturday_hours total_earnings = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dana_sunday_hours_l187_18761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_score_l187_18751

/-- Represents the scoring system and Samantha's performance in the math contest -/
structure ContestPerformance where
  correct_points : ℚ                 -- Points awarded for each correct answer
  geometry_bonus : ℚ                 -- Additional points for correct geometry answers
  total_correct : ℕ                  -- Total number of correct answers
  geometry_correct : ℕ               -- Number of correct geometry answers

/-- Calculates the total score based on the contest performance -/
def calculate_score (perf : ContestPerformance) : ℚ :=
  perf.correct_points * perf.total_correct + perf.geometry_bonus * perf.geometry_correct

/-- Theorem stating that Samantha's score is 17 points -/
theorem samantha_score :
  let perf : ContestPerformance := {
    correct_points := 1,
    geometry_bonus := 1/2,
    total_correct := 15,
    geometry_correct := 4
  }
  calculate_score perf = 17 := by
  -- Unfold the definition and simplify
  unfold calculate_score
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_samantha_score_l187_18751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l187_18783

/-- Given a train of length 300 meters that takes 39 seconds to cross a platform
    and 10 seconds to cross a signal pole, prove that the platform length is 870 meters. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300)
  (h2 : time_platform = 39)
  (h3 : time_pole = 10) :
  let train_speed := train_length / time_pole
  let platform_length := train_speed * time_platform - train_length
  platform_length = 870 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_l187_18783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_restoration_l187_18772

/-- The percentage reduction applied to the price in each step -/
def reduction_percentage : ℝ := 0.25

/-- The number of times the price is reduced -/
def num_reductions : ℕ := 2

/-- The required increase percentage to restore the original price -/
def required_increase_percentage : ℝ := 0.7778

/-- Approximation relation for real numbers -/
def approx (x y : ℝ) : Prop := abs (x - y) < 0.0001

theorem jacket_price_restoration :
  let final_price := (1 - reduction_percentage) ^ num_reductions
  approx required_increase_percentage (1 / final_price - 1)
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jacket_price_restoration_l187_18772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cut_cells_9x9_l187_18786

/-- Represents a square board -/
structure Board where
  size : Nat
  deriving Repr

/-- Represents the number of cells cut on the board -/
abbrev CutCells := Nat

/-- Checks if the number of cut cells is valid for a given board size -/
def isValidCutCount (board : Board) (cutCells : CutCells) : Prop :=
  let internalCells := (board.size - 2) ^ 2
  let vertices := internalCells + cutCells
  let edges := 4 * cutCells
  vertices - edges + 1 ≥ 2

/-- Theorem: The maximum number of cells that can be cut on a 9x9 board is 21 -/
theorem max_cut_cells_9x9 (board : Board) (h : board.size = 9) :
  ∃ (maxCut : CutCells), maxCut = 21 ∧
    isValidCutCount board maxCut ∧
    ∀ (k : CutCells), k > maxCut → ¬isValidCutCount board k := by
  sorry

#check max_cut_cells_9x9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cut_cells_9x9_l187_18786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_arrangement_exists_l187_18737

structure Rectangle where
  vertices : Finset (ℝ × ℝ)
  is_rectangle : vertices.card = 4

def share_vertex (r1 r2 : Rectangle) : Prop :=
  ∃ v, v ∈ r1.vertices ∧ v ∈ r2.vertices

theorem rectangle_arrangement_exists : ∃ (r1 r2 r3 r4 : Rectangle),
  -- No vertex is common to all four rectangles
  (∀ v, ¬(v ∈ r1.vertices ∧ v ∈ r2.vertices ∧ v ∈ r3.vertices ∧ v ∈ r4.vertices)) ∧
  -- Any two rectangles have exactly one common vertex
  (∃! v, v ∈ r1.vertices ∧ v ∈ r2.vertices) ∧
  (∃! v, v ∈ r2.vertices ∧ v ∈ r3.vertices) ∧
  (∃! v, v ∈ r3.vertices ∧ v ∈ r4.vertices) ∧
  (∃! v, v ∈ r4.vertices ∧ v ∈ r1.vertices) ∧
  (∃! v, v ∈ r1.vertices ∧ v ∈ r3.vertices) ∧
  (∃! v, v ∈ r2.vertices ∧ v ∈ r4.vertices) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_arrangement_exists_l187_18737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_roots_l187_18752

-- Define the function f(x) = log_a(x) + x - b
noncomputable def f (a b x : ℝ) : ℝ := Real.log x / Real.log a + x - b

-- State the theorem
theorem log_equation_roots (b : ℝ) :
  (∀ a ∈ Set.Icc 2 3, ∃ x ∈ Set.Icc 2 3, f a b x = 0) →
  b ∈ Set.Icc 3 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_roots_l187_18752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l187_18799

noncomputable section

/-- A parabola passing through (4, 4) and intersecting the x-axis at (2, 0) -/
def Parabola (a b : ℝ) : Prop :=
  a ≠ 0 ∧ 16 * a + 4 * b = 4 ∧ 4 * a + 2 * b = 0

/-- The distance between the intersection points of the parabola with the x-axis -/
noncomputable def IntersectionDistance (a b : ℝ) : ℝ :=
  |4 - 1/a|

theorem parabola_theorem :
  ∀ a b : ℝ, Parabola a b →
    (a = 1/2 ∧ b = -1) ∧
    (IntersectionDistance a b > 2 → a < 0 ∨ (0 < a ∧ a < 1/6) ∨ a > 1/2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_theorem_l187_18799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_no_extreme_values_non_positive_local_minimum_positive_l187_18742

noncomputable section

variable (a : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := x - a * Real.log x

theorem tangent_line_at_one (h : a = 2) :
  ∃ (m b : ℝ), ∀ x y, y = m * x + b ↔ y = f a 1 + (deriv (f a)) 1 * (x - 1) :=
sorry

theorem no_extreme_values_non_positive (h : a ≤ 0) :
  ∀ x > 0, deriv (f a) x > 0 :=
sorry

theorem local_minimum_positive (h : a > 0) :
  ∃ x, IsLocalMin (f a) x ∧ f a x = a - a * Real.log a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_no_extreme_values_non_positive_local_minimum_positive_l187_18742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_function_properties_l187_18765

/-- A Friendship Function on [0,1] -/
def FriendshipFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Icc 0 1 → f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x + y ∈ Set.Icc 0 1 → f (x + y) ≥ f x + f y)

theorem friendship_function_properties
    (f : ℝ → ℝ)
    (hf : FriendshipFunction f) :
    f 0 = 0 ∧
    ∀ x₀, x₀ ∈ Set.Icc 0 1 →
      f x₀ ∈ Set.Icc 0 1 →
      f (f x₀) = x₀ →
      f x₀ = x₀ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friendship_function_properties_l187_18765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l187_18716

-- Define a 3D space
variable (S : Type*) [NormedAddCommGroup S] [InnerProductSpace ℝ S] [Fact (finrank ℝ S = 3)]

-- Define a line in 3D space
def Line (p q : S) : Set S := {r : S | ∃ t : ℝ, r = p + t • (q - p)}

-- Define parallelism between lines
def Parallel (l₁ l₂ : Set S) : Prop :=
  ∃ (p₁ q₁ p₂ q₂ : S) (k : ℝ), k ≠ 0 ∧ 
  l₁ = Line S p₁ q₁ ∧ l₂ = Line S p₂ q₂ ∧ 
  q₁ - p₁ = k • (q₂ - p₂)

-- State the theorem
theorem parallel_transitivity (l₁ l₂ l₃ : Set S) :
  Parallel S l₁ l₃ → Parallel S l₂ l₃ → Parallel S l₁ l₂ := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_transitivity_l187_18716
