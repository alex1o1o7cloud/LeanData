import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_representation_1992_l752_75276

/-- Represents a decreasing sequence of positive integers where each term is divisible by the next. -/
def ValidSequence (seq : List Nat) : Prop :=
  seq.length > 0 ∧
  (∀ i j, i < j → j < seq.length → seq.get! i > seq.get! j) ∧
  (∀ i, i < seq.length - 1 → seq.get! i % seq.get! (i + 1) = 0) ∧
  (∀ i, i < seq.length → seq.get! i > 0)

/-- The sum of a list of natural numbers. -/
def ListSum (seq : List Nat) : Nat :=
  seq.foldl (· + ·) 0

/-- Theorem stating that the given sequence is the longest valid representation of 1992. -/
theorem longest_representation_1992 :
  let seq := [1992, 996, 498, 249, 83, 1]
  ValidSequence seq ∧
  ListSum seq = 1992 ∧
  (∀ other : List Nat, ValidSequence other → ListSum other = 1992 → other.length ≤ seq.length) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_representation_1992_l752_75276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_l752_75203

open Real

theorem ln_inequality (x : ℝ) (h : x > 0) : 
  log x ≤ x - 1 := by sorry

-- Definitions and conditions
noncomputable def ln_derivative (x : ℝ) : ℝ := 1 / x

axiom ln_point : log 1 = 0

axiom ln_tangent_slope : deriv log 1 = 1

-- Note: We use 'log' instead of 'ln' as it's the standard notation in Lean for natural logarithm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_inequality_l752_75203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l752_75272

theorem tan_value_in_second_quadrant (α m : ℝ) 
  (h1 : Real.sin α = m) 
  (h2 : |m| < 1) 
  (h3 : π / 2 < α ∧ α < π) : 
  Real.tan α = -m / Real.sqrt (1 - m^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_value_in_second_quadrant_l752_75272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_exponentials_l752_75216

theorem sum_of_complex_exponentials :
  let z₁ := 20 * Complex.exp (Complex.I * (3 * π / 13 : ℝ))
  let z₂ := 20 * Complex.exp (Complex.I * (21 * π / 26 : ℝ))
  let r := 40 * Real.cos (3 * π / 13)
  let θ := 27 * π / 52
  z₁ + z₂ = r * Complex.exp (Complex.I * θ) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_complex_exponentials_l752_75216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l752_75237

noncomputable def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 6*y + 9 = 0
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 2*y + 1 = 0

noncomputable def center1 : ℝ × ℝ := (-1, -3)
noncomputable def center2 : ℝ × ℝ := (3, -1)
noncomputable def radius1 : ℝ := 1
noncomputable def radius2 : ℝ := 2 * Real.sqrt 2

theorem circles_are_separate : 
  Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2) > radius1 + radius2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_are_separate_l752_75237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roja_work_days_l752_75299

/-- Proves that given the conditions on Malar and Roja's work rates, Roja alone will complete the task in 84 days -/
theorem roja_work_days (total_work : ℝ) (h1 : total_work > 0) : 
  (total_work / (total_work / 35 - total_work / 60)) = 84 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roja_work_days_l752_75299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_one_l752_75280

/-- A function that checks if a five-digit number has strictly increasing digits -/
def has_increasing_digits (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧
  let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
  ∀ i j, i < j → i < 5 → j < 5 → digits.get! i < digits.get! j

/-- A function that checks if two five-digit numbers coincide in at least one digit -/
def coincide_in_digit (n m : ℕ) : Prop :=
  ∃ i, i < 5 ∧ (n / 10^i) % 10 = (m / 10^i) % 10

/-- The main theorem stating that the smallest possible value of k is 1 -/
theorem smallest_k_is_one :
  ∃ N : ℕ, N ≥ 10000 ∧ N < 100000 ∧
  (∀ n : ℕ, has_increasing_digits n → coincide_in_digit n N) := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_one_l752_75280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l752_75295

theorem smallest_number_divisible_by_multiple : ∃! n : ℕ, 
  (∀ m : ℕ, m < n → ¬(∀ d ∈ ({12, 16, 18, 21, 28} : Finset ℕ), (m - 3) % d = 0)) ∧ 
  (∀ d ∈ ({12, 16, 18, 21, 28} : Finset ℕ), (n - 3) % d = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_divisible_by_multiple_l752_75295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_sin_2α_minus_tan_α_y_range_y_range_tight_l752_75246

noncomputable def α : Real := Real.arctan (- Real.sqrt 3 / 3)

noncomputable def f (x : Real) : Real := Real.cos (x - α) * Real.cos α - Real.sin (x - α) * Real.sin α

noncomputable def y (x : Real) : Real := Real.sqrt 3 * f (Real.pi / 2 - 2 * x) - 2 * f x ^ 2

theorem angle_properties :
  Real.cos α = -1/2 ∧ Real.sin α = Real.sqrt 3 / 2 := by sorry

theorem sin_2α_minus_tan_α :
  Real.sin (2 * α) - Real.tan α = - Real.sqrt 3 / 6 := by sorry

theorem y_range :
  ∀ x, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → -2 ≤ y x ∧ y x ≤ 1 := by sorry

theorem y_range_tight :
  (∃ x, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 ∧ y x = -2) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 ∧ y x = 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_properties_sin_2α_minus_tan_α_y_range_y_range_tight_l752_75246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_expression_simplification_l752_75257

-- Define the function that represents the sequence of operations
noncomputable def final_expression (c : ℝ) : ℝ :=
  ((3 * c + 6) - 6 * c) / 3

-- Theorem stating that the final expression is equal to -c + 2
theorem final_expression_simplification (c : ℝ) : 
  final_expression c = -c + 2 := by
  -- Expand the definition of final_expression
  unfold final_expression
  -- Simplify the algebraic expression
  ring
  -- The proof is complete
  done

#check final_expression_simplification

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_expression_simplification_l752_75257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l752_75229

-- Define the inner function g(x) = x^2 - 5x - 6
def g (x : ℝ) : ℝ := x^2 - 5*x - 6

-- Define the outer function h(t) = 2^t
noncomputable def h (t : ℝ) : ℝ := 2^t

-- Define the composite function f(x) = h(g(x)) = 2^(x^2 - 5x - 6)
noncomputable def f (x : ℝ) : ℝ := h (g x)

-- State the theorem
theorem f_decreasing_on_interval :
  (∀ x y, x < y → x < (5/2 : ℝ) → y < (5/2 : ℝ) → g x > g y) →
  (∀ t₁ t₂, t₁ < t₂ → h t₁ < h t₂) →
  ∀ x y, x < y → x < (5/2 : ℝ) → y < (5/2 : ℝ) → f x > f y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l752_75229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_wrt_x_axis_l752_75245

/-- Definition of symmetry with respect to the x-axis -/
def is_symmetric_wrt_x_axis (y y' : ℝ → ℝ) : Prop :=
  ∀ x, y' x = -y x

/-- Given a line y = 2x + 1, its symmetric line with respect to the x-axis is y = -2x - 1 -/
theorem symmetric_line_wrt_x_axis :
  ∀ (x y : ℝ), y = 2 * x + 1 → ∃ (y' : ℝ), y' = -2 * x - 1 ∧ is_symmetric_wrt_x_axis (λ x => 2 * x + 1) (λ x => -2 * x - 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_wrt_x_axis_l752_75245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l752_75293

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/2) * (Real.sin x + Real.cos x) - (1/2) * abs (Real.sin x - Real.cos x)

-- State the theorem about the range of f
theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ -1 ≤ y ∧ y ≤ Real.sqrt 2 / 2 := by
  sorry

#check range_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l752_75293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_a_range_l752_75204

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * |x - a|

theorem inequality_solution (x : ℝ) :
  |x - 1/2| + f 3 x ≥ 2 ↔ x ≤ 0 ∨ x ≥ 2 := by sorry

theorem a_range (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2) 1, |x - 1/2| + f a x ≤ x) →
  0 ≤ a ∧ a ≤ 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_a_range_l752_75204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_shift_production_l752_75251

theorem second_shift_production (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (2/3 * x * (3/4 * y)) / (x * y + 2/3 * x * (3/4 * y)) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_shift_production_l752_75251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subgrid_sum_theorem_l752_75253

/-- Represents a 2000 x 2000 grid where each cell contains either 1 or -1 -/
def Grid := Fin 2000 → Fin 2000 → Int

/-- Predicate to check if a grid contains only 1 or -1 in each cell -/
def valid_grid (g : Grid) : Prop :=
  ∀ i j, g i j = 1 ∨ g i j = -1

/-- The sum of all numbers in the grid -/
def grid_sum (g : Grid) : Int :=
  Finset.sum (Finset.univ : Finset (Fin 2000)) (λ i =>
    Finset.sum (Finset.univ : Finset (Fin 2000)) (λ j => g i j))

/-- Predicate to check if there exist 1000 rows and 1000 columns with sum at least 1000 -/
def exists_subgrid_with_sum (g : Grid) : Prop :=
  ∃ (rows cols : Finset (Fin 2000)),
    rows.card = 1000 ∧ cols.card = 1000 ∧
    (Finset.sum rows (λ i => Finset.sum cols (λ j => g i j))) ≥ 1000

/-- The main theorem -/
theorem subgrid_sum_theorem (g : Grid) 
  (h1 : valid_grid g) 
  (h2 : grid_sum g ≥ 0) : 
  exists_subgrid_with_sum g := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subgrid_sum_theorem_l752_75253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_family_reunion_cost_theorem_l752_75209

/-- Represents the cost and capacity of a soda box -/
structure SodaBox where
  cans_per_box : ℕ
  cost_per_box : ℚ

/-- Represents the soda preferences and logistics for a family reunion -/
structure FamilyReunion where
  total_attendees : ℕ
  cans_per_person : ℕ
  cola_preference : ℕ
  lemon_lime_preference : ℕ
  orange_preference : ℕ
  cola_box : SodaBox
  lemon_lime_box : SodaBox
  orange_box : SodaBox
  sales_tax_rate : ℚ
  family_members_paying : ℕ

/-- Calculates the cost per family member for soda at a family reunion -/
noncomputable def cost_per_family_member (reunion : FamilyReunion) : ℚ :=
  sorry

/-- Theorem stating that the cost per family member is approximately $5.21 -/
theorem family_reunion_cost_theorem (reunion : FamilyReunion) 
  (h1 : reunion.total_attendees = 60)
  (h2 : reunion.cans_per_person = 2)
  (h3 : reunion.cola_preference = 20)
  (h4 : reunion.lemon_lime_preference = 15)
  (h5 : reunion.orange_preference = 18)
  (h6 : reunion.cola_box = { cans_per_box := 10, cost_per_box := 2 })
  (h7 : reunion.lemon_lime_box = { cans_per_box := 12, cost_per_box := 3 })
  (h8 : reunion.orange_box = { cans_per_box := 8, cost_per_box := 5/2 })
  (h9 : reunion.sales_tax_rate = 6/100)
  (h10 : reunion.family_members_paying = 6) :
  ∃ (ε : ℚ), ε > 0 ∧ |cost_per_family_member reunion - 521/100| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_family_reunion_cost_theorem_l752_75209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l752_75238

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.cos x, -Real.sin x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x - 2 * Real.sqrt 3 * Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem sin_2x_value (x : ℝ) (h1 : f x = 24/13) (h2 : x ∈ Set.Icc (π/4) (π/2)) :
  Real.sin (2*x) = (12 * Real.sqrt 3 + 5) / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2x_value_l752_75238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_shapes_area_ratio_l752_75214

theorem inscribed_shapes_area_ratio :
  let r : ℝ := 1
  let circle_area := π * r^2
  let square_side := Real.sqrt 2 * r
  let square_area := square_side^2
  let shaded_area := circle_area - square_area
  let triangle_area := (1 / 2) * square_side^2
  triangle_area / shaded_area = 1 / (π - 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_shapes_area_ratio_l752_75214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_P_maximizes_distance_difference_l752_75227

/-- The line on which point P lies -/
def line (x y : ℝ) : Prop := 2 * x - y - 4 = 0

/-- Point A -/
def A : ℝ × ℝ := (4, -1)

/-- Point B -/
def B : ℝ × ℝ := (3, 4)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The point P that maximizes the distance difference -/
def P : ℝ × ℝ := (5, 6)

/-- Theorem stating that P maximizes the distance difference -/
theorem P_maximizes_distance_difference :
  line P.1 P.2 ∧
  ∀ Q : ℝ × ℝ, line Q.1 Q.2 →
    (distance P A - distance P B) ≥ (distance Q A - distance Q B) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_P_maximizes_distance_difference_l752_75227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_arc_ratio_l752_75240

-- Define the circle C
noncomputable def circle_C (θ : Real) : Real × Real :=
  (5 + 2 * Real.cos θ, Real.sqrt 3 + 2 * Real.sin θ)

-- Define the line l₂
noncomputable def line_l₂ (x : Real) : Real := (Real.sqrt 3 / 3) * x

-- Define the intersection points A and B
def intersection_points (C : Real → Real × Real) (l : Real → Real) : Set (Real × Real) :=
  {p | ∃ t, C t = p ∧ (l (p.1) = p.2)}

-- Define the arc ratio
noncomputable def arc_ratio (C : Real → Real × Real) (l : Real → Real) : Real :=
  2 -- This is the expected ratio

-- Theorem statement
theorem intersection_arc_ratio :
  arc_ratio circle_C line_l₂ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_arc_ratio_l752_75240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l752_75288

/-- Definition of the ellipse (C) -/
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

/-- The focus of the ellipse -/
noncomputable def focus : ℝ × ℝ := (0, Real.sqrt 3)

/-- A point on the ellipse -/
noncomputable def point_on_ellipse : ℝ × ℝ := (1/2, Real.sqrt 3)

/-- The fixed point M -/
def point_M : ℝ × ℝ := (0, 1)

/-- Theorem stating the properties of the ellipse and the line A'B -/
theorem ellipse_properties :
  (∀ x y, ellipse x y ↔ x^2 + y^2/4 = 1) ∧
  (∀ k : ℝ, k ≠ 0 →
    ∀ A B : ℝ × ℝ, ellipse A.1 A.2 → ellipse B.1 B.2 →
    (A.2 - point_M.2 = k * (A.1 - point_M.1)) →
    (B.2 - point_M.2 = k * (B.1 - point_M.1)) →
    ∃ y : ℝ, ((-A.1) * B.2 + B.1 * A.2) / (B.1 - A.1) = y ∧ y = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l752_75288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_face_opposite_U_is_M_l752_75219

/-- Represents a face of the cube -/
inductive Face : Type
  | U : Face
  | I : Face
  | P : Face
  | K : Face
  | M : Face
  | O : Face

/-- Represents the adjacency relationship between faces -/
def adjacent : Face → Face → Prop := sorry

/-- Represents the opposite relationship between faces -/
def opposite : Face → Face → Prop := sorry

axiom adjacency_symmetric : ∀ (a b : Face), adjacent a b → adjacent b a

axiom opposite_symmetric : ∀ (a b : Face), opposite a b → opposite b a

axiom opposite_unique : ∀ (a b c : Face), opposite a b → opposite a c → b = c

axiom not_adjacent_to_opposite : ∀ (a b : Face), opposite a b → ¬ adjacent a b

axiom cube_structure :
  (adjacent Face.K Face.I) ∧
  (adjacent Face.K Face.M) ∧
  (adjacent Face.K Face.O) ∧
  (adjacent Face.K Face.U) ∧
  (adjacent Face.O Face.U) ∧
  (adjacent Face.I Face.O)

theorem face_opposite_U_is_M : opposite Face.U Face.M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_face_opposite_U_is_M_l752_75219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distance_l752_75273

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbolaEquation (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The right focus of the hyperbola -/
noncomputable def rightFocus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- A line perpendicular to the x-axis passing through a point -/
def verticalLine (p : Point) (x : ℝ) : Prop :=
  x = p.x

/-- The asymptotes of the hyperbola -/
def asymptotes (h : Hyperbola) (p : Point) : Prop :=
  p.y = h.b / h.a * p.x ∨ p.y = -h.b / h.a * p.x

/-- The theorem to be proved -/
theorem hyperbola_intersection_distance (h : Hyperbola) (A B : Point) :
  h.a = 1 → h.b = Real.sqrt 3 →
  hyperbolaEquation h A →
  hyperbolaEquation h B →
  verticalLine (rightFocus h) A.x →
  verticalLine (rightFocus h) B.x →
  asymptotes h A →
  asymptotes h B →
  (A.x - B.x)^2 + (A.y - B.y)^2 = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_intersection_distance_l752_75273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_for_partition_max_n_is_14_l752_75278

def En (n : ℕ+) : Set ℕ+ := {x | x ≤ n}

def Pn (n : ℕ+) : Set ℚ := {x | ∃ (a b : ℕ+), a ∈ En n ∧ b ∈ En n ∧ x = a / Real.sqrt b}

def hasPropertyOmega (A : Set ℚ) (n : ℕ+) : Prop :=
  A ⊆ Pn n ∧
  ∀ x y, x ∈ A → y ∈ A → x ≠ y → ∀ k : ℕ+, x + y ≠ k^2

theorem max_n_for_partition (n : ℕ+) : Prop :=
  (∃ A B : Set ℚ,
    hasPropertyOmega A n ∧
    hasPropertyOmega B n ∧
    A ∩ B = ∅ ∧
    Pn n = A ∪ B) ∧
  (∀ m : ℕ+, m > n →
    ¬∃ A B : Set ℚ,
      hasPropertyOmega A m ∧
      hasPropertyOmega B m ∧
      A ∩ B = ∅ ∧
      Pn m = A ∪ B)

theorem max_n_is_14 : max_n_for_partition 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_n_for_partition_max_n_is_14_l752_75278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DCE_measure_l752_75258

-- Define the points
variable (A B C D E : EuclideanSpace ℝ 2)

-- Define the angles
def angle_ACD : ℝ := sorry
def angle_DCE : ℝ := sorry
def angle_ECB : ℝ := sorry

-- State the theorem
theorem angle_DCE_measure
  (h1 : angle_ACD = 90)
  (h2 : angle_ECB = 52)
  : angle_DCE = 38 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_DCE_measure_l752_75258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_theorem_l752_75206

/-- The point on the parabola y = x^2 closest to the line 2x - y - 4 = 0 --/
def closest_point : ℝ × ℝ := (1, 1)

/-- The parabola y = x^2 --/
def parabola (p : ℝ × ℝ) : Prop := p.2 = p.1^2

/-- The line 2x - y - 4 = 0 --/
def line (p : ℝ × ℝ) : Prop := 2 * p.1 - p.2 - 4 = 0

/-- Distance function from a point to the line --/
noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |2 * p.1 - p.2 - 4| / Real.sqrt 5

theorem closest_point_theorem :
  parabola closest_point ∧
  ∀ p : ℝ × ℝ, parabola p → distance_to_line closest_point ≤ distance_to_line p :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_theorem_l752_75206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_and_range_l752_75200

-- Define the function f(x) = x^3 - ax^2 + 3x
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*x

-- Define the derivative of f(x)
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 3

theorem critical_point_and_range (a : ℝ) :
  (f' a 3 = 0) →  -- x = 3 is a critical point
  (a = 5 ∧ 
   ∀ x, x ∈ Set.Icc 2 4 → -9 ≤ f 5 x ∧ f 5 x ≤ -4 ∧
   ∃ y z, y ∈ Set.Icc 2 4 ∧ z ∈ Set.Icc 2 4 ∧ f 5 y = -9 ∧ f 5 z = -4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_critical_point_and_range_l752_75200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l752_75260

/-- Given an acute triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2b and sin B = √3/4, then A = π/3 -/
theorem triangle_angle_proof (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 →  -- A is acute
  0 < B ∧ B < π/2 →  -- B is acute
  0 < C ∧ C < π/2 →  -- C is acute
  a = 2*b →          -- Given condition
  Real.sin B = Real.sqrt 3 / 4 →     -- Given condition
  A = π/3 :=         -- Conclusion to prove
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_proof_l752_75260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_taken_l752_75202

/-- Represents a sequence of card numbers based on the given rule -/
def cardSequence (start : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) (fuel : ℕ) : List ℕ :=
    if fuel = 0 then acc.reverse
    else if n > 100 then acc.reverse
    else aux (2 * n + 2) (n :: acc) (fuel - 1)
  aux start [] 100

/-- The set of all valid card sequences -/
def allSequences : List (List ℕ) :=
  (List.range 100).map (fun n => cardSequence (n + 1))

/-- The total number of cards in all valid sequences -/
def totalCards : ℕ :=
  (allSequences.map List.length).sum

theorem max_cards_taken (totalCards : ℕ) :
  totalCards = 50 ∧ 
  ∀ (s : List ℕ), s ∈ allSequences → 
    (∀ (n : ℕ), n ∈ s → n ≤ 100) ∧
    (∀ (n : ℕ), n ∈ s → (2 * n + 2) ∈ s ∨ (2 * n + 2) > 100) :=
by sorry

#eval totalCards

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cards_taken_l752_75202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_perp_plane_l752_75292

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [CompleteSpace V]

-- Define the lines and planes as affine subspaces
variable (m n : AffineSubspace ℝ V) (α : AffineSubspace ℝ V)

-- Define the conditions
variable (h_distinct : m ≠ n)
variable (h_parallel : m.direction = n.direction)
variable (h_perp : m.direction.orthogonal ≤ α.direction)

-- Theorem statement
theorem line_parallel_perp_plane :
  n.direction.orthogonal ≤ α.direction :=
by
  -- The proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_perp_plane_l752_75292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l752_75259

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (4 : ℝ)^x + m * (2 : ℝ)^x

-- Part 1
theorem part_one :
  ∀ x : ℝ, ((4 : ℝ)^x - 3 * (2 : ℝ)^x > 4) ↔ (x > 2) := by sorry

-- Part 2
theorem part_two :
  ∀ m : ℝ, (∃ c : ℝ, (∀ x : ℝ, f m x + f m (-x) ≥ c) ∧ (∃ x₀ : ℝ, f m x₀ + f m (-x₀) = c)) →
  (c = -4 → m = -3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l752_75259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l752_75249

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- A configuration of 4 points on a plane -/
structure Configuration where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The sum of squares of distances between all pairs of points in a configuration -/
noncomputable def sumOfSquaredDistances (c : Configuration) : ℝ :=
  (distance c.p1 c.p2)^2 + (distance c.p1 c.p3)^2 + (distance c.p1 c.p4)^2 +
  (distance c.p2 c.p3)^2 + (distance c.p2 c.p4)^2 + (distance c.p3 c.p4)^2

/-- Helper function to get a point from a configuration by index -/
def getPoint (c : Configuration) (i : Fin 4) : Point :=
  match i with
  | 0 => c.p1
  | 1 => c.p2
  | 2 => c.p3
  | 3 => c.p4

/-- The theorem stating the maximum sum of squared distances -/
theorem max_sum_squared_distances :
  ∀ c : Configuration,
  (∀ (i j : Fin 4), i ≠ j → distance (getPoint c i) (getPoint c j) ≤ 1) →
  sumOfSquaredDistances c ≤ 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_squared_distances_l752_75249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_temperature_difference_proof_l752_75274

def refrigerator_temperature_difference : Int → Int → Int
| 5, -2 => 7
| _, _ => 0  -- Default case for all other inputs

#check refrigerator_temperature_difference

theorem refrigerator_temperature_difference_proof (refrigeration_temp freezer_temp : Int) :
  refrigeration_temp = 5 ∧ freezer_temp = -2 →
  refrigerator_temperature_difference refrigeration_temp freezer_temp = 7 := by
  intro h
  cases h with
  | intro h1 h2 =>
    rw [h1, h2]
    rfl

#check refrigerator_temperature_difference_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_refrigerator_temperature_difference_proof_l752_75274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l752_75220

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + Real.cos x

-- State the theorem
theorem f_inequality (t : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, f x = x + Real.cos x) →
  (f (t^2) > f (2*t - 1) ↔ t ∈ Set.Ioo (1/2) 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l752_75220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_1000_l752_75226

def has_more_ones_than_zeros (n : ℕ) : Bool :=
  let digits := n.digits 2
  digits.count 1 > digits.count 0

def M : ℕ := (Finset.range 1501).filter (fun n => has_more_ones_than_zeros n) |>.card

theorem M_mod_1000 : M ≡ 884 [MOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_mod_1000_l752_75226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l752_75297

open Real

-- Define the curves in polar coordinates
noncomputable def C₁ (θ : ℝ) : ℝ := 2 * sin θ
noncomputable def C₂ (θ : ℝ) : ℝ := Real.sqrt (2 / (1 + cos θ ^ 2))

-- Define the intersection points
noncomputable def A : ℝ := C₁ (π / 3)
noncomputable def B : ℝ := C₂ (π / 3)

-- Theorem to prove
theorem intersection_distance :
  abs (A - B) = Real.sqrt 3 - 2 * Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l752_75297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l752_75211

/-- Converts polar coordinates to rectangular coordinates -/
noncomputable def polar_to_rectangular (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

/-- The given polar coordinates -/
noncomputable def polar_point : ℝ × ℝ := (4, 5 * Real.pi / 3)

/-- The expected rectangular coordinates -/
noncomputable def rectangular_point : ℝ × ℝ := (2, -2 * Real.sqrt 3)

/-- Theorem stating that the conversion from polar to rectangular coordinates is correct -/
theorem polar_to_rectangular_conversion :
  polar_to_rectangular polar_point.1 polar_point.2 = rectangular_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_to_rectangular_conversion_l752_75211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l752_75236

/-- The function f(x) = sin 2 * cos (2x) -/
noncomputable def f (x : ℝ) := Real.sin 2 * Real.cos (2 * x)

/-- The minimum value of f(x) is -1/2 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ -1/2 ∧ ∃ x₀ : ℝ, f x₀ = -1/2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l752_75236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l752_75275

-- Define the functions h and j
noncomputable def h : ℝ → ℝ := sorry
noncomputable def j : ℝ → ℝ := sorry

-- Define the conditions
axiom intersection1 : h 3 = 3 ∧ j 3 = 3
axiom intersection2 : h 6 = 9 ∧ j 6 = 9
axiom intersection3 : h 9 = 18 ∧ j 9 = 18
axiom intersection4 : h 12 = 18 ∧ j 12 = 18

-- Theorem statement
theorem intersection_sum : 
  ∃ (x y : ℝ), h (3 * x) = 3 * j x ∧ h (3 * x) = y ∧ x + y = 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_sum_l752_75275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bran_financial_surplus_l752_75287

/-- Represents Bran's financial situation for a semester --/
structure BranFinances where
  tuition_fee : ℚ
  additional_expenses : ℚ
  savings_goal : ℚ
  part_time_hourly_rate : ℚ
  part_time_hours_per_week : ℚ
  tutoring_biweekly_payment : ℚ
  scholarship_percentage : ℚ
  tax_rate : ℚ
  semester_duration_months : ℚ

/-- Calculates Bran's financial surplus for the semester --/
def calculate_surplus (finances : BranFinances) : ℚ :=
  let total_expenses := finances.tuition_fee * (1 - finances.scholarship_percentage) +
                        finances.additional_expenses +
                        finances.savings_goal
  let weeks_in_semester := finances.semester_duration_months * 4
  let part_time_income := finances.part_time_hourly_rate *
                          finances.part_time_hours_per_week *
                          weeks_in_semester
  let part_time_income_after_tax := part_time_income * (1 - finances.tax_rate)
  let tutoring_income := finances.tutoring_biweekly_payment *
                         (finances.semester_duration_months / (1/2))
  let total_income := part_time_income_after_tax + tutoring_income
  total_income - total_expenses

/-- Theorem stating Bran's financial surplus --/
theorem bran_financial_surplus :
  let finances : BranFinances := {
    tuition_fee := 3000,
    additional_expenses := 800,
    savings_goal := 500,
    part_time_hourly_rate := 20,
    part_time_hours_per_week := 15,
    tutoring_biweekly_payment := 100,
    scholarship_percentage := 2/5,
    tax_rate := 1/10,
    semester_duration_months := 4
  }
  calculate_surplus finances = 2020 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bran_financial_surplus_l752_75287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increment_ratio_equals_average_rate_of_change_l752_75282

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the interval endpoints
variable (x₀ x₁ : ℝ)

-- Define the average rate of change
noncomputable def averageRateOfChange (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

-- Define the ratio of function value increment to independent variable increment
noncomputable def incrementRatio (f : ℝ → ℝ) (x₀ x₁ : ℝ) : ℝ :=
  (f x₁ - f x₀) / (x₁ - x₀)

-- Theorem statement
theorem increment_ratio_equals_average_rate_of_change
  (f : ℝ → ℝ) (x₀ x₁ : ℝ) (h : x₀ ≠ x₁) :
  incrementRatio f x₀ x₁ = averageRateOfChange f x₀ x₁ := by
  -- Unfold the definitions of incrementRatio and averageRateOfChange
  unfold incrementRatio averageRateOfChange
  -- The expressions are identical, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_increment_ratio_equals_average_rate_of_change_l752_75282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_below_x_axis_half_total_l752_75270

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parallelogram defined by four vertices -/
structure Parallelogram where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Calculates the area of a parallelogram given three of its vertices -/
noncomputable def parallelogramArea (p q r : Point) : ℝ :=
  (1/2) * |p.x * (q.y - r.y) + q.x * (r.y - p.y) + r.x * (p.y - q.y)|

/-- The specific parallelogram PQRS from the problem -/
def pqrs : Parallelogram :=
  { p := {x := 4, y := 4}
    q := {x := -2, y := -2}
    r := {x := -8, y := -2}
    s := {x := -2, y := 4} }

/-- Theorem: The area below the x-axis is half the total area of the parallelogram PQRS -/
theorem area_below_x_axis_half_total (para : Parallelogram := pqrs) :
  2 * (parallelogramArea para.p para.q {x := 0, y := 0}) =
    parallelogramArea para.p para.q para.r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_below_x_axis_half_total_l752_75270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_x_value_l752_75232

noncomputable def vector_a : ℝ × ℝ := (1, Real.sqrt (1 + Real.sin (20 * Real.pi / 180)))
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (1 / Real.sin (55 * Real.pi / 180), x)

def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem collinear_implies_x_value (x : ℝ) :
  collinear vector_a (vector_b x) → x = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_implies_x_value_l752_75232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratios_sum_to_one_l752_75250

/-- The ratio of radii of circles tangent to the main circle and two sides of the triangle -/
def RadiusRatio (P Q R : EuclideanSpace ℝ (Fin 2)) (O : EuclideanSpace ℝ (Fin 2)) (radius : ℝ) : ℝ :=
  sorry

/-- Given a triangle ABC inscribed in circle C(O,R), and ratios α, β, γ < 1 defined as the ratios
    of radii of circles tangent to C and pairs of sides of the triangle, prove that α + β + γ = 1. -/
theorem inscribed_triangle_ratios_sum_to_one 
  (O : EuclideanSpace ℝ (Fin 2)) (R : ℝ) (A B C : EuclideanSpace ℝ (Fin 2)) (α β γ : ℝ)
  (h_inscribed : Set.Subset {A, B, C} (Metric.sphere O R))
  (h_α : α < 1)
  (h_β : β < 1)
  (h_γ : γ < 1)
  (h_α_def : α = RadiusRatio A B C O R)
  (h_β_def : β = RadiusRatio B C A O R)
  (h_γ_def : γ = RadiusRatio C A B O R) : 
  α + β + γ = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_triangle_ratios_sum_to_one_l752_75250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_vector_dot_product_bound_l752_75223

/-- A circle with center O and radius 1 -/
structure Circle :=
  (O : ℝ × ℝ)
  (radius : ℝ)
  (is_unit : radius = 1)

/-- A point on or inside the circle -/
structure Point (c : Circle) :=
  (coords : ℝ × ℝ)
  (in_circle : (coords.1 - c.O.1)^2 + (coords.2 - c.O.2)^2 ≤ c.radius^2)

/-- Vector between two points -/
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

/-- Dot product of two vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

theorem circle_vector_dot_product_bound (c : Circle) 
  (A B C : Point c) (P : Point c)
  (h_diameter : vector c.O A.coords = vector B.coords c.O) :
  -4/3 ≤ dot_product (vector P.coords A.coords) (vector P.coords B.coords) +
         dot_product (vector P.coords B.coords) (vector P.coords C.coords) +
         dot_product (vector P.coords C.coords) (vector P.coords A.coords) ∧
   dot_product (vector P.coords A.coords) (vector P.coords B.coords) +
   dot_product (vector P.coords B.coords) (vector P.coords C.coords) +
   dot_product (vector P.coords C.coords) (vector P.coords A.coords) ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_vector_dot_product_bound_l752_75223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_sets_l752_75233

theorem count_possible_sets : 
  ∃! n : ℕ, n = 2 ∧ ∃ S : Finset (Finset ℕ), 
    (∀ M ∈ S, M ∪ {1} = {1, 2, 3}) ∧ Finset.card S = n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_sets_l752_75233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_not_right_triangle_l752_75224

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a function to check if a triangle is right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.a^2 + t.c^2 = t.b^2 ∨ t.b^2 + t.c^2 = t.a^2

-- Define the four triangles
def triangleA : Triangle := ⟨7, 24, 25⟩
def triangleB : Triangle := ⟨1.5, 2, 3⟩
noncomputable def triangleC : Triangle := ⟨1, Real.sqrt 2, 1⟩
def triangleD : Triangle := ⟨9, 12, 15⟩

-- State the theorem
theorem only_B_not_right_triangle :
  isRightTriangle triangleA ∧
  ¬isRightTriangle triangleB ∧
  isRightTriangle triangleC ∧
  isRightTriangle triangleD := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_not_right_triangle_l752_75224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l752_75239

/-- The function g(x) = (px+q)/(rx+s) -/
noncomputable def g (p q r s x : ℝ) : ℝ := (p*x + q) / (r*x + s)

/-- Theorem stating that given the conditions, 42 is not in the range of g -/
theorem unique_number_not_in_range
  (p q r s : ℝ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (hs : s ≠ 0)
  (h_g3 : g p q r s 3 = 3)
  (h_g81 : g p q r s 81 = 81)
  (h_involution : ∀ x, x ≠ -s/r → g p q r s (g p q r s x) = x) :
  ∀ y, y ≠ 42 → ∃ x, g p q r s x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_not_in_range_l752_75239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l752_75235

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (7 * sequence_a n + Real.sqrt (45 * (sequence_a n)^2 - 36)) / 2

theorem sequence_a_properties :
  (∀ n : ℕ, sequence_a n > 0 ∧ (∃ m : ℤ, sequence_a n = m)) ∧
  (∀ n : ℕ, ∃ k : ℕ, (sequence_a n) * (sequence_a (n + 1)) - 1 = k^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l752_75235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_linearity_l752_75218

-- Define the set H
variable (H : Set ℝ)

-- Define the properties of H
variable (h_non_zero : ∃ x : ℝ, x ∈ H ∧ x ≠ 0)
variable (h_closed_add : ∀ (x y : ℝ), x ∈ H → y ∈ H → (x + y) ∈ H)

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_order : ∀ (x y : ℝ), x ∈ H → y ∈ H → x ≤ y → f x ≤ f y)
variable (h_additive : ∀ (x y : ℝ), x ∈ H → y ∈ H → f (x + y) = f x + f y)

-- State the theorem
theorem function_linearity :
  ∃ c : ℝ, c ≥ 0 ∧ ∀ (x : ℝ), x ∈ H → f x = c * x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_linearity_l752_75218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_existence_l752_75254

-- Define the functions based on their graphical representations
noncomputable def F : ℝ → ℝ := λ x ↦ x  -- Linear increasing function
noncomputable def G : ℝ → ℝ := λ x ↦ -x^2 + 4  -- Downward opening parabola
noncomputable def H : ℝ → ℝ := λ x ↦ if x ≤ -1 then 2 else if x < 1 then 0 else -2  -- Horizontal line segments, discontinuous
noncomputable def I : ℝ → ℝ := λ x ↦ if x < 0 then 2*x + 4 else 2*x - 4  -- V-shaped piecewise function

-- Define the property of having an inverse
def has_inverse (f : ℝ → ℝ) : Prop := ∃ g : ℝ → ℝ, ∀ x, g (f x) = x ∧ f (g x) = x

-- State the theorem
theorem inverse_existence :
  has_inverse F ∧ has_inverse I ∧ ¬has_inverse G ∧ ¬has_inverse H := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_existence_l752_75254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_10_degrees_l752_75265

/-- Represents the volume of a gas at a given temperature -/
structure GasVolume where
  /-- The temperature in degrees Celsius -/
  temperature : ℝ
  /-- The volume in cubic centimeters -/
  volume : ℝ

/-- The rate of volume change per degree Celsius -/
noncomputable def volumeChangeRate : ℝ := 6 / 5

/-- Calculates the volume of gas at a given temperature -/
noncomputable def calculateVolume (initialVolume : GasVolume) (finalTemp : ℝ) : ℝ :=
  initialVolume.volume + volumeChangeRate * (finalTemp - initialVolume.temperature)

theorem gas_volume_at_10_degrees 
  (initialVolume : GasVolume)
  (h1 : initialVolume.temperature = 30)
  (h2 : initialVolume.volume = 100) :
  calculateVolume initialVolume 10 = 76 := by
  sorry

#check gas_volume_at_10_degrees

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gas_volume_at_10_degrees_l752_75265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l752_75262

/-- Circle C₁ with equation (x+1)²+y²=36 -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 36

/-- Circle C₂ with equation (x-1)²+y²=4 -/
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4

/-- A moving circle M is internally tangent to C₁ and externally tangent to C₂ -/
def is_tangent_to_C₁_and_C₂ (M : ℝ → ℝ × ℝ) : Prop :=
  ∃ (t x y : ℝ), C₁ x y ∧ C₂ x y ∧
  (∃ (r : ℝ), r > 0 ∧
    ((x - (M t).1)^2 + (y - (M t).2)^2 = r^2) ∧
    ((x - (-1))^2 + y^2 = (6 - r)^2) ∧
    ((x - 1)^2 + y^2 = (r + 2)^2))

/-- The trajectory of the center of M is an ellipse with equation x²/16 + y²/15 = 1 -/
def is_ellipse_trajectory (M : ℝ → ℝ × ℝ) : Prop :=
  ∀ (t : ℝ), let (x, y) := M t; x^2/16 + y^2/15 = 1

/-- Main theorem: The trajectory of the center of M is the given ellipse -/
theorem moving_circle_trajectory (M : ℝ → ℝ × ℝ) :
  is_tangent_to_C₁_and_C₂ M → is_ellipse_trajectory M :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_moving_circle_trajectory_l752_75262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l752_75207

/-- The area of a triangle given its vertices -/
noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  let v := (C.1 - A.1, C.2 - A.2)
  let w := (C.1 - B.1, C.2 - B.2)
  (1/2) * abs (v.1 * w.2 - v.2 * w.1)

/-- Theorem: The area of the triangle with vertices (-3,2), (5,-1), and (9,6) is 34 -/
theorem triangle_area_example : triangle_area (-3, 2) (5, -1) (9, 6) = 34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_example_l752_75207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l752_75244

-- Define the coordinate system
structure CartesianPoint where
  x : ℝ
  y : ℝ

structure PolarPoint where
  r : ℝ
  θ : ℝ

-- Define the line l
def line_l (t : ℝ) : CartesianPoint :=
  { x := t, y := t + 2 }

-- Define the curve C
def curve_C (p : CartesianPoint) : Prop :=
  p.x^2 - 4*p.x + p.y^2 - 2*p.y = 0

-- Define point P
noncomputable def point_P : PolarPoint :=
  { r := 2 * Real.sqrt 2, θ := 7 * Real.pi / 4 }

-- Define the translated line l'
def line_l' (x : ℝ) : CartesianPoint :=
  { x := x, y := x }

-- Theorem statement
theorem area_of_triangle_PAB :
  ∃ (A B : CartesianPoint),
    curve_C A ∧ curve_C B ∧
    (∃ (x : ℝ), A = line_l' x) ∧
    (∃ (x : ℝ), B = line_l' x) ∧
    (1/2) * Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) * (2 * Real.sqrt 2) = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PAB_l752_75244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_divisible_by_seven_l752_75284

theorem least_k_divisible_by_seven : ∃ k : ℕ, k > 0 ∧
  (∀ m : ℕ, 0 < m ∧ m < k → ¬(∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a * 2023 = b * m ∧ (a + b) % 7 = 0)) ∧
  (∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a * 2023 = b * k ∧ (a + b) % 7 = 0) ∧
  k = 7 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_k_divisible_by_seven_l752_75284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_proof_l752_75296

noncomputable def rotation_60 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1/2, -Real.sqrt 3/2; Real.sqrt 3/2, 1/2]

noncomputable def scaling_2 : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 0; 0, 2]

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, -Real.sqrt 3; Real.sqrt 3, 1]

theorem transformation_proof :
  scaling_2 * rotation_60 = transformation_matrix := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_proof_l752_75296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_le_one_l752_75271

/-- The function f(x) defined as x^2 - a*ln(x) - x - 2023 --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x - x - 2023

/-- Theorem stating that if f(x) is monotonically increasing on [1, +∞), then a ≤ 1 --/
theorem f_monotone_implies_a_le_one (a : ℝ) :
  (∀ x y, 1 ≤ x ∧ x ≤ y → f a x ≤ f a y) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_implies_a_le_one_l752_75271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_image_stones_l752_75261

def stone_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | n + 4 => stone_sequence (n + 3) + 3 * (n + 4) - 2

theorem tenth_image_stones : stone_sequence 9 = 145 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_image_stones_l752_75261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l752_75290

noncomputable def sign (x : ℝ) : ℝ :=
  if x < 0 then -1 else if x > 0 then 1 else 0

def satisfies_equations (x y z : ℝ) : Prop :=
  x = 1000 - 1001 * sign (y + z - 1) ∧
  y = 1000 - 1001 * sign (x + z + 2) ∧
  z = 1000 - 1001 * sign (x + y - 3)

theorem number_of_solutions : 
  ∃! (s : Finset (ℝ × ℝ × ℝ)), 
    (∀ (x y z : ℝ), (x, y, z) ∈ s ↔ satisfies_equations x y z) ∧ 
    Finset.card s = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_solutions_l752_75290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l752_75281

def is_prime (p : ℕ) : Prop := Nat.Prime p

def satisfies_conditions (f : ℤ → ℤ) : Prop :=
  (∀ p : ℕ, is_prime p → f p > 0) ∧
  (∀ p : ℕ, ∀ x : ℤ, is_prime p → (p : ℤ) ∣ ((f x + f p : ℤ) ^ (f p).toNat - x))

theorem unique_function_satisfying_conditions :
  ∀ f : ℤ → ℤ, satisfies_conditions f → ∀ x : ℤ, f x = x :=
by
  sorry

#check unique_function_satisfying_conditions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_function_satisfying_conditions_l752_75281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_to_boys_ratio_l752_75283

theorem girls_to_boys_ratio (total : ℕ) (difference : ℕ) 
  (h1 : total = 26) 
  (h2 : difference = 6) : 
  (((total + difference) / 2 : ℚ) / ((total - difference) / 2)) = 8 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_girls_to_boys_ratio_l752_75283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_quadrant_III_l752_75217

noncomputable def f (x : ℝ) : ℝ := -2/3 * x + 3

theorem function_not_in_quadrant_III :
  ∀ x y : ℝ, f x = y → ¬(x < 0 ∧ y < 0) :=
by
  intro x y h
  contrapose! h
  simp
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_not_in_quadrant_III_l752_75217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l752_75222

/-- The coordinates of point E' -/
noncomputable def E' : ℝ × ℝ := (0, Real.sqrt 3)

/-- The coordinates of point F' -/
noncomputable def F' : ℝ × ℝ := (0, -Real.sqrt 3)

/-- The condition for point G -/
def G_condition (G : ℝ × ℝ) : Prop :=
  let (x, y) := G
  x ≠ 0 ∧ ((y - Real.sqrt 3) / x) * ((y + Real.sqrt 3) / x) = -3/4

/-- The trajectory equation -/
def trajectory_equation (G : ℝ × ℝ) : Prop :=
  let (x, y) := G
  x^2 / 4 + y^2 / 3 = 1

/-- The theorem to be proved -/
theorem trajectory_and_min_area :
  ∀ G : ℝ × ℝ, G_condition G → 
  (trajectory_equation G ∧
   ∃ (A B : ℝ × ℝ), 
     A ≠ B ∧
     trajectory_equation A ∧
     trajectory_equation B ∧
     (A.1 * B.1 + A.2 * B.2 = 0) ∧
     ∀ (A' B' : ℝ × ℝ), 
       A' ≠ B' → 
       trajectory_equation A' → 
       trajectory_equation B' → 
       A'.1 * B'.1 + A'.2 * B'.2 = 0 →
       (1/2 * abs (A.1 * B.2 - A.2 * B.1) ≤ 1/2 * abs (A'.1 * B'.2 - A'.2 * B'.1)) ∧
     (1/2 * abs (A.1 * B.2 - A.2 * B.1) = 12/7)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_area_l752_75222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l752_75247

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the foci distance -/
def foci_distance (c : ℝ) : Prop := c = 1

/-- Definition of the equilateral triangle condition -/
def equilateral_triangle (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 = a^2 ∧ (x - a)^2 + y^2 = a^2 ∧ x^2 + (y - a*Real.sqrt 3/2)^2 = a^2

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) (ha : a > b) (hb : b > 0) :
  foci_distance 1 →
  equilateral_triangle a →
  (∀ x y, ellipse a b x y ↔ ellipse (Real.sqrt 3) (Real.sqrt 2) x y) ∧
  (∃ (P : ℝ × ℝ), P.1 = 3/2 ∧ (P.2 = Real.sqrt 2/2 ∨ P.2 = -Real.sqrt 2/2) ∧
    ∀ (M N : ℝ × ℝ), ellipse (Real.sqrt 3) (Real.sqrt 2) M.1 M.2 → 
      ellipse (Real.sqrt 3) (Real.sqrt 2) N.1 N.2 →
      ∃ (k : ℝ), N.2 - M.2 = k * (N.1 - M.1) →
        P.1 = M.1 + N.1 ∧ P.2 = M.2 + N.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l752_75247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l752_75289

-- Define the points and circles
variable (A B C D E F G H : ℝ × ℝ)
variable (O N P : ℝ × ℝ)

-- Define the radii of the circles
def radius_O : ℝ := 12
def radius_N : ℝ := 15
def radius_P : ℝ := 18

-- Define the conditions
axiom on_line_segment : 
  (∃ t₁ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ B = (1 - t₁) • A + t₁ • D) ∧
  (∃ t₂ : ℝ, 0 ≤ t₂ ∧ t₂ ≤ 1 ∧ C = (1 - t₂) • A + t₂ • D)

axiom diameters : 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * radius_O)^2 ∧
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = (2 * radius_N)^2 ∧
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = (2 * radius_P)^2

def is_tangent_line (L : ℝ × ℝ → ℝ × ℝ → Prop) (X Y : ℝ × ℝ) (C : ℝ × ℝ) (R : ℝ) : Prop :=
  L X Y ∧ ((X.1 - C.1)^2 + (X.2 - C.2)^2 = R^2 ∨ (Y.1 - C.1)^2 + (Y.2 - C.2)^2 = R^2)

axiom tangent_line : 
  ∃ (L : ℝ × ℝ → ℝ × ℝ → Prop),
    is_tangent_line L A G P radius_P ∧
    is_tangent_line L A G N radius_N

def on_circle (X : ℝ × ℝ) (C : ℝ × ℝ) (R : ℝ) : Prop :=
  (X.1 - C.1)^2 + (X.2 - C.2)^2 = R^2

axiom intersects_circle :
  on_circle E N radius_N ∧
  on_circle F N radius_N ∧
  E ≠ H ∧ F ≠ H

-- Theorem to prove
theorem chord_length : 
  (E.1 - F.1)^2 + (E.2 - F.2)^2 = 30^2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_l752_75289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l752_75294

/-- Given a periodic function f and its related function g, prove that g(π/3) = 1 -/
theorem periodic_function_value (w φ : ℝ) 
  (f g : ℝ → ℝ)
  (hf : ∀ x : ℝ, f x = 5 * Real.cos (w * x + φ))
  (hf_sym : ∀ x : ℝ, f (π/3 + x) = f (π/3 - x))
  (hg : ∀ x : ℝ, g x = 4 * Real.sin (w * x + φ) + 1) : 
  g (π/3) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_value_l752_75294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_gain_percentage_l752_75221

/-- Proves that given a cost price of ₹1600, if reducing the cost price by 5% and the selling price
by ₹8 results in a 10% profit, then the original gain percentage is 5%. -/
theorem original_gain_percentage (cost_price : ℝ) (reduced_cost_price : ℝ) (reduced_selling_price : ℝ) 
  (original_gain_percentage : ℝ) :
  cost_price = 1600 →
  reduced_cost_price = cost_price * 0.95 →
  reduced_selling_price = reduced_cost_price * 1.1 →
  reduced_selling_price = cost_price * (1 + original_gain_percentage / 100) - 8 →
  original_gain_percentage = 5 := by
  sorry

#check original_gain_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_gain_percentage_l752_75221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_factorial_as_consecutive_product_l752_75212

theorem no_factorial_as_consecutive_product :
  ¬ ∃ (n : ℕ), n > 2 ∧ ∃ (a : ℕ), n.factorial = (Finset.range (n - 2)).prod (λ i => a + i + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_factorial_as_consecutive_product_l752_75212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_black_ball_l752_75286

/-- Given a bag of balls with the following properties:
  * There are 100 balls in total
  * There are 45 red balls
  * The probability of drawing a white ball is 0.23
  Prove that the probability of drawing a black ball is 0.32 -/
theorem probability_of_black_ball 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (prob_white : ℝ) 
  (h1 : total_balls = 100) 
  (h2 : red_balls = 45) 
  (h3 : prob_white = 0.23) :
  (total_balls - red_balls - Int.floor (prob_white * total_balls) : ℝ) / total_balls = 0.32 := by
  sorry

#check probability_of_black_ball

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_black_ball_l752_75286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_theorem_l752_75267

/-- The time taken for two trains to pass each other -/
noncomputable def train_passing_time (train_length : ℝ) (speed1 speed2 : ℝ) : ℝ :=
  (2 * train_length) / (speed1 + speed2)

/-- Conversion from km/hr to m/s -/
noncomputable def km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * (1000 / 3600)

theorem train_passing_theorem (train_length : ℝ) (speed1 speed2 : ℝ) 
  (h1 : train_length = 900)
  (h2 : speed1 = 45)
  (h3 : speed2 = 30) :
  ∃ ε > 0, |train_passing_time train_length 
    (km_per_hr_to_m_per_s speed1) (km_per_hr_to_m_per_s speed2) - 86.4| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_theorem_l752_75267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_warriors_won_30_games_l752_75205

structure Team where
  name : String
  games_won : ℕ

def Hawks : Team := ⟨"Hawks", 0⟩
def Falcons : Team := ⟨"Falcons", 0⟩
def Warriors : Team := ⟨"Warriors", 0⟩
def Lions : Team := ⟨"Lions", 0⟩
def Knights : Team := ⟨"Knights", 0⟩

theorem warriors_won_30_games
  (team_list : List Team)
  (h1 : team_list = [Hawks, Falcons, Warriors, Lions, Knights])
  (h2 : Hawks.games_won > Falcons.games_won)
  (h3 : Warriors.games_won > Lions.games_won ∧ Warriors.games_won < Knights.games_won)
  (h4 : Lions.games_won > 22)
  (h5 : ∀ t1 t2, t1 ∈ team_list → t2 ∈ team_list → t1 ≠ t2 → t1.games_won ≠ t2.games_won) :
  Warriors.games_won = 30 := by
  sorry

#check warriors_won_30_games

end NUMINAMATH_CALUDE_ERRORFEEDBACK_warriors_won_30_games_l752_75205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l752_75234

/-- Time taken for a train to cross a platform -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (platform_length : ℝ) :
  train_length = 180 →
  train_speed_kmph = 72 →
  platform_length = 220.03199999999998 →
  ∃ (time : ℝ), 
    (abs (time - 20) < 0.01) ∧ 
    (time = (train_length + platform_length) / (train_speed_kmph * (1000 / 3600))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l752_75234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_x_product_of_roots_l752_75277

theorem positive_x_product_of_roots (x : ℝ) (hx : x > 0) 
  (h : Real.sqrt (18 * x) * Real.sqrt (50 * x) * Real.sqrt (2 * x) * Real.sqrt (25 * x) = 50) : 
  x = 1 / (Real.sqrt 3 / Real.sqrt (Real.sqrt 2)) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_positive_x_product_of_roots_l752_75277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l752_75210

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - x - a * Real.log (x - a)

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a (x + a) - a * (x + (1/2) * a - 1)

-- State the theorem
theorem extreme_points_inequality (a : ℝ) (x₁ x₂ : ℝ) :
  x₁ < x₂ →
  (∀ x, x₁ < x ∧ x < x₂ → g a x₁ ≥ g a x ∧ g a x₂ ≥ g a x) →
  0 < f a x₁ - f a x₂ ∧ f a x₁ - f a x₂ < 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_inequality_l752_75210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l752_75215

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + 2*a*x^2 + 2

theorem min_a_value (a : ℝ) :
  (∀ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, x ≤ y → f a x ≤ f a y) →
  a ≥ -1/4 := by
  sorry

#check min_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l752_75215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_and_not_square_l752_75268

theorem solution_count_and_not_square (n : ℕ) :
  let count := (2 * n^3 + 7 * n^2 + 7 * n + 3 : ℕ)
  (∃ (s : Finset (ℕ × ℕ)), s.card = count ∧
    ∀ (x y : ℕ), (x, y) ∈ s ↔ x^2 - y^2 = 10^2 * 30^(2*n)) ∧
  ¬∃ (k : ℕ), k^2 = count :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_count_and_not_square_l752_75268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zaraza_almaz_sum_l752_75252

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a 6-digit number -/
def SixDigitNumber := Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10

/-- Represents a 5-digit number -/
def FiveDigitNumber := Fin 10 × Fin 10 × Fin 10 × Fin 10 × Fin 10

/-- Converts a SixDigitNumber to a natural number -/
def toNat6 (n : SixDigitNumber) : ℕ :=
  n.1.val * 100000 + n.2.1.val * 10000 + n.2.2.1.val * 1000 + 
  n.2.2.2.1.val * 100 + n.2.2.2.2.1.val * 10 + n.2.2.2.2.2.val

/-- Converts a FiveDigitNumber to a natural number -/
def toNat5 (n : FiveDigitNumber) : ℕ :=
  n.1.val * 10000 + n.2.1.val * 1000 + n.2.2.1.val * 100 + n.2.2.2.1.val * 10 + n.2.2.2.2.val

/-- Checks if all digits in a SixDigitNumber are unique -/
def allUnique6 (n : SixDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2.1 ∧ n.1 ≠ n.2.2.2.1 ∧ n.1 ≠ n.2.2.2.2.1 ∧ n.1 ≠ n.2.2.2.2.2 ∧
  n.2.1 ≠ n.2.2.1 ∧ n.2.1 ≠ n.2.2.2.1 ∧ n.2.1 ≠ n.2.2.2.2.1 ∧ n.2.1 ≠ n.2.2.2.2.2 ∧
  n.2.2.1 ≠ n.2.2.2.1 ∧ n.2.2.1 ≠ n.2.2.2.2.1 ∧ n.2.2.1 ≠ n.2.2.2.2.2 ∧
  n.2.2.2.1 ≠ n.2.2.2.2.1 ∧ n.2.2.2.1 ≠ n.2.2.2.2.2 ∧
  n.2.2.2.2.1 ≠ n.2.2.2.2.2

/-- Checks if all digits in a FiveDigitNumber are unique -/
def allUnique5 (n : FiveDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2.1 ∧ n.1 ≠ n.2.2.2.1 ∧ n.1 ≠ n.2.2.2.2 ∧
  n.2.1 ≠ n.2.2.1 ∧ n.2.1 ≠ n.2.2.2.1 ∧ n.2.1 ≠ n.2.2.2.2 ∧
  n.2.2.1 ≠ n.2.2.2.1 ∧ n.2.2.1 ≠ n.2.2.2.2 ∧
  n.2.2.2.1 ≠ n.2.2.2.2

theorem zaraza_almaz_sum (zaraza : SixDigitNumber) (almaz : FiveDigitNumber) 
  (h1 : allUnique6 zaraza)
  (h2 : allUnique5 almaz)
  (h3 : zaraza.2.2.2.2.2 = almaz.2.2.2.2)
  (h4 : zaraza.2.2.2.2.1 = almaz.2.2.2.1)
  (h5 : zaraza.1 = almaz.1)
  (h6 : toNat6 zaraza % 4 = 0)
  (h7 : toNat5 almaz % 28 = 0) :
  (toNat6 zaraza + toNat5 almaz) % 100 = 32 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zaraza_almaz_sum_l752_75252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l752_75231

-- Define the function f as noncomputable
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem min_omega (ω φ T : ℝ) : 
  ω > 0 → 
  0 < φ → φ < Real.pi / 2 →
  (∀ x, f ω φ (x + T) = f ω φ x) →  -- T is the period
  f ω φ T = 1/2 →
  (∀ x, f ω φ (14*Real.pi/3 - x) = f ω φ x) →  -- Symmetry about x = 7π/3
  (∀ ω' > 0, (∀ x, f ω' φ (x + T) = f ω' φ x) → ω ≤ ω') →  -- ω is minimal
  ω = 2/7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_l752_75231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_parallelepiped_l752_75208

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A rectangular parallelepiped in 3D space -/
structure Parallelepiped where
  max_x : ℝ
  max_y : ℝ
  max_z : ℝ

/-- The length of the shortest path between two points on the surface of a parallelepiped -/
noncomputable def shortestPathLength (start : Point3D) (end_ : Point3D) (p : Parallelepiped) : ℝ :=
  sorry

/-- The theorem stating the shortest path length between two specific points on a specific parallelepiped -/
theorem shortest_path_on_parallelepiped :
  let start := Point3D.mk 0 1 2
  let end_ := Point3D.mk 22 4 2
  let p := Parallelepiped.mk 22 5 4
  shortestPathLength start end_ p = Real.sqrt 657 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_on_parallelepiped_l752_75208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l752_75269

-- Define the constants
noncomputable def a : ℝ := (0.31 : ℝ) ^ 2
noncomputable def b : ℝ := Real.log 0.31 / Real.log 2
noncomputable def c : ℝ := 2 ^ (0.31 : ℝ)

-- State the theorem
theorem order_of_numbers : b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_numbers_l752_75269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l752_75255

-- Define the binomial expansion
noncomputable def binomial_expansion (x : ℝ) (n : ℕ) : ℝ := (x - 2 / Real.sqrt x) ^ n

-- Define the sum of binomial coefficients
def sum_binomial_coefficients (n : ℕ) : ℕ := 2^n

-- Define the sum of all coefficients
noncomputable def sum_all_coefficients (x : ℝ) (n : ℕ) : ℝ := binomial_expansion 1 n

-- Define the coefficient of x^4 term
def coefficient_x4 (n : ℕ) : ℕ := n.choose 2 * 4

-- State the theorem
theorem binomial_expansion_properties :
  ∃ n : ℕ,
    sum_binomial_coefficients n = 128 ∧
    n = 7 ∧
    sum_all_coefficients 1 n = -1 ∧
    coefficient_x4 n = 84 := by
  -- Proof goes here
  sorry

#eval sum_binomial_coefficients 7
#eval coefficient_x4 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_expansion_properties_l752_75255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sin_alpha_l752_75291

theorem parallel_vectors_sin_alpha :
  ∀ (α : ℝ),
  let a : Fin 2 → ℝ := ![5, 1]
  let b : Fin 2 → ℝ := ![4, Real.cos α]
  α ∈ Set.Ioo (-π/2 : ℝ) 0 →
  (∃ (k : ℝ), ∀ (i : Fin 2), a i = k * b i) →
  Real.sin α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sin_alpha_l752_75291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l752_75213

def is_valid_sequence (n : ℕ) (a : List ℕ) : Prop :=
  ∀ i j, i < j → i < a.length → j < a.length →
    a[i]! < a[j]! ∧ a[j]! < n ∧ Nat.Coprime a[i]! n ∧ Nat.Coprime a[j]! n

def satisfies_condition (n : ℕ) : Prop :=
  ∀ a : List ℕ, is_valid_sequence n a →
    ∀ i, i < a.length - 1 → ¬(3 ∣ (a[i]! + a[i+1]!))

theorem valid_n_characterization :
  ∀ n : ℕ, n > 1 → (satisfies_condition n ↔ n = 2 ∨ n = 4 ∨ n = 10) := by
  sorry

#check valid_n_characterization

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_n_characterization_l752_75213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_focal_length_l752_75266

noncomputable section

-- Define the curve
def curve (θ : Real) : Real × Real :=
  (3 * Real.cos θ, Real.sqrt 6 * Real.sin θ)

-- Define the focal length
def focal_length (curve : Real → Real × Real) : Real :=
  let a := 3  -- Semi-major axis
  let b := Real.sqrt 6  -- Semi-minor axis
  2 * Real.sqrt (a^2 - b^2)

-- Theorem statement
theorem curve_focal_length :
  focal_length curve = 2 * Real.sqrt 3 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_focal_length_l752_75266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_separation_l752_75230

/-- The time it takes for Adam and Simon to be 50 miles apart -/
noncomputable def separation_time : ℝ := 25 / Real.sqrt 41

/-- Adam's speed in miles per hour -/
def adam_speed : ℝ := 10

/-- Simon's speed in miles per hour -/
def simon_speed : ℝ := 8

/-- The distance between Adam and Simon after separation_time hours -/
def separation_distance : ℝ := 50

theorem bicycle_separation :
  separation_distance = Real.sqrt ((adam_speed * separation_time)^2 + (simon_speed * separation_time)^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_separation_l752_75230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_bundle_cost_graph_shape_l752_75201

-- Define the cost function
def cost (n : ℕ) : ℕ :=
  if n ≤ 3 then 12 * n
  else if n ≤ 6 then 36 + 10 * (n - 3)
  else 66 + 8 * (n - 6)

-- Define a type for the graph shape
inductive GraphShape
  | PiecewiseLinear
  | DisconnectedPoints
  | ContinuousStraightLine
  | ExponentialCurve
  | HorizontalParallelLines

-- Theorem statement
theorem pen_bundle_cost_graph_shape :
  ∃ (points : List (ℕ × ℕ)), 
    (∀ n ∈ List.range 10, (n + 1, cost (n + 1)) ∈ points) ∧
    GraphShape.PiecewiseLinear = 
      if points = [] then GraphShape.DisconnectedPoints
      else GraphShape.PiecewiseLinear :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_bundle_cost_graph_shape_l752_75201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_distinct_primes_is_even_l752_75279

theorem sum_of_eight_distinct_primes_is_even
  (q : Finset ℕ)
  (h1 : q.card = 8)
  (h2 : ∀ p ∈ q, Nat.Prime p)
  (h3 : ∀ p ∈ q, p ≥ 3)
  (h4 : ∀ p1 p2, p1 ∈ q → p2 ∈ q → p1 ≠ p2 → p1 ≠ p2) :
  Even (q.sum id) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_eight_distinct_primes_is_even_l752_75279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l752_75225

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line passing through a point -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) (l : Line) : ℝ :=
  abs (l.slope * x - y + l.intercept) / Real.sqrt (l.slope^2 + 1)

/-- The right focus of an ellipse -/
noncomputable def right_focus (e : Ellipse) : ℝ × ℝ := (e.a * eccentricity e, 0)

theorem ellipse_properties (e : Ellipse) (l : Line) :
  distance_point_to_line 0 0 l = Real.sqrt 2 / 2 ∧ 
  l.slope = 1 ∧
  (∃ (x y : ℝ), (x, y) = right_focus e ∧ y = l.slope * x + l.intercept) →
  e.a = 5/3 ∧ 
  e.b = Real.sqrt 5/3 ∧
  (∃ (p1 p2 : ℝ × ℝ), 
    (p1 = (2/3, Real.sqrt 2/3) ∨ p1 = (2/3, -Real.sqrt 2/3)) ∧
    (p2 = (2/3, Real.sqrt 2/3) ∨ p2 = (2/3, -Real.sqrt 2/3)) ∧
    p1 ≠ p2 ∧
    (p1.1)^2 / e.a^2 + (p1.2)^2 / e.b^2 = 1 ∧
    (p2.1)^2 / e.a^2 + (p2.2)^2 / e.b^2 = 1 ∧
    (∃ (l1 l2 : Line),
      (l1.slope = 1 ∧ l1.intercept = -1) ∨ (l1.slope = -1 ∧ l1.intercept = 1) ∧
      (l2.slope = 1 ∧ l2.intercept = -1) ∨ (l2.slope = -1 ∧ l2.intercept = 1) ∧
      l1 ≠ l2 ∧
      (∃ (a1 a2 b1 b2 : ℝ × ℝ),
        (a1.1)^2 / e.a^2 + (a1.2)^2 / e.b^2 = 1 ∧
        (b1.1)^2 / e.a^2 + (b1.2)^2 / e.b^2 = 1 ∧
        (a2.1)^2 / e.a^2 + (a2.2)^2 / e.b^2 = 1 ∧
        (b2.1)^2 / e.a^2 + (b2.2)^2 / e.b^2 = 1 ∧
        a1.2 = l1.slope * a1.1 + l1.intercept ∧
        b1.2 = l1.slope * b1.1 + l1.intercept ∧
        a2.2 = l2.slope * a2.1 + l2.intercept ∧
        b2.2 = l2.slope * b2.1 + l2.intercept ∧
        Real.sqrt (p1.1^2 + p1.2^2) = 
          Real.sqrt (a1.1^2 + a1.2^2) + Real.sqrt (b1.1^2 + b1.2^2) ∧
        Real.sqrt (p2.1^2 + p2.2^2) = 
          Real.sqrt (a2.1^2 + a2.2^2) + Real.sqrt (b2.1^2 + b2.2^2)))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l752_75225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l752_75243

/-- Calculates the speed of a train in km/h given its length and time to cross a signal post. -/
noncomputable def train_speed (length : ℝ) (time : ℝ) : ℝ :=
  (length / time) * 3.6

/-- Theorem stating that a train with given length and crossing time has a specific speed. -/
theorem train_speed_calculation (length time : ℝ) 
  (h1 : length = 150)
  (h2 : time = 14.998800095992321) :
  ∃ (ε : ℝ), ε > 0 ∧ |train_speed length time - 36.002| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l752_75243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l752_75263

-- Define the parameters
def train_a_length : ℚ := 800
def train_b_length : ℚ := 1000
def train_a_speed : ℚ := 72
def train_b_speed : ℚ := 60

-- Define the function to calculate the time
def calculate_passing_time (length_a length_b speed_a speed_b : ℚ) : ℚ :=
  (length_a + length_b) / ((speed_a + speed_b) * (1000 / 3600))

-- State the theorem
theorem train_passing_time_approx :
  ∃ ε : ℚ, ε > 0 ∧ |calculate_passing_time train_a_length train_b_length train_a_speed train_b_speed - 49.05| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_approx_l752_75263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_rhombus_area_equality_l752_75241

/-- The area of a regular octagon with side length c -/
noncomputable def octagon_area (c : ℝ) : ℝ := 2 * c^2 * (1 + Real.sqrt 2)

/-- The area of a rhombus with given angle and side length -/
noncomputable def rhombus_area (angle : ℝ) (c : ℝ) : ℝ := c^2 * Real.sin (angle * Real.pi / 180)

/-- Theorem stating that the area of a regular octagon equals the sum of areas of 4 rhombuses with 45° angles and 2 rhombuses with 90° angles -/
theorem octagon_rhombus_area_equality (c : ℝ) (h : c > 0) :
  octagon_area c = 4 * rhombus_area 45 c + 2 * rhombus_area 90 c := by
  sorry

#check octagon_rhombus_area_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_rhombus_area_equality_l752_75241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_necessary_not_sufficient_l752_75256

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 - 2*Complex.I) * (a + Complex.I)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im < 0

-- Statement of the theorem
theorem a_positive_necessary_not_sufficient :
  ∃ (a : ℝ), a > 0 ∧ ¬(in_fourth_quadrant (z a)) ∧
  ∀ (a : ℝ), in_fourth_quadrant (z a) → a > 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_positive_necessary_not_sufficient_l752_75256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l752_75298

-- Define the function f(x) as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 + (Real.sin x) / (2 + Real.cos x)

-- State the theorem
theorem sum_of_max_min_f : 
  (⨆ (x : ℝ), f x) + (⨅ (x : ℝ), f x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_f_l752_75298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_profit_l752_75264

/-- Calculates John's profit from chopping trees and selling tables -/
def calculate_profit (trees_group1 trees_group2 trees_group3
                      planks_per_tree1 planks_per_tree2 planks_per_tree3
                      labor_cost1 labor_cost2 labor_cost3
                      planks_per_table table_price1 table_price2 table_price3 : ℕ) : ℕ :=
  let total_planks := trees_group1 * planks_per_tree1 +
                      trees_group2 * planks_per_tree2 +
                      trees_group3 * planks_per_tree3
  let total_tables := total_planks / planks_per_table
  let labor_cost := trees_group1 * labor_cost1 +
                    trees_group2 * labor_cost2 +
                    trees_group3 * labor_cost3
  let revenue := min total_tables 10 * table_price1 +
                 min (max (total_tables - 10) 0) 20 * table_price2 +
                 max (total_tables - 30) 0 * table_price3
  revenue - labor_cost

theorem john_profit :
  calculate_profit 10 10 10 20 25 30 120 80 60 15 350 325 300 = 13400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_profit_l752_75264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l752_75228

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

-- State the theorem
theorem f_max_value :
  (∀ x > 0, f x ≤ 1) ∧ (∃ x > 0, f x = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l752_75228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_46_to_nearest_tenth_l752_75242

/-- Rounds a real number to the nearest tenth -/
noncomputable def round_to_nearest_tenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The statement that 3.46 rounded to the nearest tenth equals 3.5 -/
theorem round_3_46_to_nearest_tenth :
  round_to_nearest_tenth 3.46 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_3_46_to_nearest_tenth_l752_75242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_l752_75248

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x - Real.pi/5)

noncomputable def g (x : ℝ) : ℝ := f (x/2)

theorem transformed_function (x : ℝ) : g x = 3 * Real.sin (x/2 - Real.pi/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformed_function_l752_75248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l752_75285

/-- The focus of a parabola defined by y = ax² + bx + c --/
noncomputable def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2 * a)
  let k := c - b^2 / (4 * a)
  (h, k + 1 / (4 * a))

/-- The parabola equation y = -1/8x² + 2x - 1 --/
noncomputable def parabola_equation (x : ℝ) : ℝ :=
  -1/8 * x^2 + 2*x - 1

theorem focus_coordinates :
  parabola_focus (-1/8) 2 (-1) = (8, 5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_focus_coordinates_l752_75285
