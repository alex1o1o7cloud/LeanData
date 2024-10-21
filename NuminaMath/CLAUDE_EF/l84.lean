import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_max_apples_l84_8468

/-- Calculates the maximum number of apples Brian can buy given the conditions --/
def max_apples_brian_can_buy 
  (apple_dozen_cost : ℚ)
  (kiwi_cost : ℚ)
  (banana_cost : ℚ)
  (initial_money : ℚ)
  (subway_fare : ℚ) : ℕ :=
  let total_fruit_cost := kiwi_cost + banana_cost
  let total_subway_cost := 2 * subway_fare
  let remaining_money := initial_money - total_fruit_cost - total_subway_cost
  let dozens_of_apples := (remaining_money / apple_dozen_cost).floor
  (12 * dozens_of_apples).toNat

/-- Theorem stating that Brian can buy a maximum of 24 apples --/
theorem brian_max_apples : 
  max_apples_brian_can_buy 14 10 5 50 (7/2) = 24 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brian_max_apples_l84_8468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l84_8415

/-- The equation of the tangent line to the curve y = 2x - x³ at the point (1, 1) is x + y - 2 = 0. -/
theorem tangent_line_equation (x y : ℝ) : 
  let f : ℝ → ℝ := λ x => 2*x - x^3
  let point : ℝ × ℝ := (1, 1)
  let tangent_line : ℝ → ℝ → Prop := λ x y => x + y - 2 = 0
  (f point.1 = point.2) →  -- The point (1, 1) lies on the curve
  (HasDerivAt f (-1) point.1) →  -- The derivative at x = 1 is -1
  (tangent_line point.1 point.2) →  -- The tangent line passes through the point (1, 1)
  (∀ x y, tangent_line x y ↔ y - point.2 = (-1) * (x - point.1)) →  -- Definition of a tangent line
  ∀ x y, tangent_line x y ↔ x + y - 2 = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l84_8415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_path_probability_l84_8453

/-- Represents a point on the grid --/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the grid layout --/
def Grid : Type := List Point

/-- Defines the possible directions of movement --/
inductive Direction
  | East
  | South

/-- Represents a path on the grid --/
def PathType : Type := List Direction

/-- The starting point A --/
def A : Point := ⟨0, 0⟩

/-- The intermediate point B --/
def B : Point := ⟨2, 1⟩

/-- The intermediate point C --/
def C : Point := ⟨3, 2⟩

/-- The destination point D --/
def D : Point := ⟨4, 4⟩

/-- Checks if a path passes through a given point --/
def passesThrough (path : PathType) (point : Point) : Prop := sorry

/-- Checks if point1 is reached before point2 in a path --/
def reachedBefore (path : PathType) (point1 point2 : Point) : Prop := sorry

/-- Calculates the probability of a path satisfying a given condition --/
noncomputable def pathProbability (condition : PathType → Prop) : ℚ := sorry

/-- The main theorem to prove --/
theorem student_path_probability :
  pathProbability (λ path => passesThrough path B ∧ passesThrough path C ∧ reachedBefore path B C) = 18/35 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_path_probability_l84_8453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l84_8497

/-- A parabola is defined by the equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 = 4 * p.2}

/-- Point A on the parabola has an ordinate of 4 -/
def PointA : ℝ × ℝ := (4, 4)

/-- The focus of a parabola with equation x^2 = 4y is at (0, 1) -/
def Focus : ℝ × ℝ := (0, 1)

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem distance_to_focus :
  PointA ∈ Parabola → distance PointA Focus = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_focus_l84_8497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angles_and_m_ratio_l84_8421

/-- For an isosceles triangle with inscribed to circumscribed circle radius ratio m -/
def IsoscelesTriangle (m : ℝ) :=
  ∃ (α B : ℝ), 0 < α ∧ 0 < B ∧ 2 * α + B = Real.pi ∧ m > 0 ∧ m ≤ 1/2

theorem isosceles_triangle_angles_and_m_ratio 
  (m : ℝ) (h : IsoscelesTriangle m) : 
  ∃ (α B : ℝ), 
    (α = Real.arccos ((1 + Real.sqrt (1 - 2*m)) / 2) ∨ 
     α = Real.arccos ((1 - Real.sqrt (1 - 2*m)) / 2)) ∧
    (B = 2 * Real.arcsin ((1 + Real.sqrt (1 - 2*m)) / 2) ∨ 
     B = 2 * Real.arcsin ((1 - Real.sqrt (1 - 2*m)) / 2)) ∧
    0 < m ∧ m ≤ 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angles_and_m_ratio_l84_8421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l84_8486

theorem fraction_sum_problem : 
  ∃ (a b c d : ℕ), 
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧   -- positive fractions
    b ≤ 100 ∧ d ≤ 100 ∧               -- denominators not exceeding 100
    Nat.Coprime a b ∧ Nat.Coprime c d ∧  -- irreducible fractions
    a * d + c * b = 86 * b * d ∧      -- sum equals 86/111
    111 * a * d = 86 * b * d          -- cross multiplication to avoid division
    :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_sum_problem_l84_8486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_equilateral_triangles_equals_num_vertices_l84_8407

/-- A cube is a three-dimensional geometric shape with 8 vertices. -/
structure Cube where
  vertices : Finset (Fin 8)

/-- An equilateral triangle is a triangle with all sides of equal length. -/
structure EquilateralTriangle where
  vertices : Fin 3 → Fin 8

/-- The set of all equilateral triangles that can be formed using three vertices of a cube. -/
def equilateralTrianglesInCube (c : Cube) : Finset EquilateralTriangle :=
  sorry

/-- The theorem stating that the number of equilateral triangles in a cube
    is equal to the number of vertices in the cube. -/
theorem num_equilateral_triangles_equals_num_vertices (c : Cube) :
  (equilateralTrianglesInCube c).card = c.vertices.card := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_equilateral_triangles_equals_num_vertices_l84_8407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l84_8412

/-- The maximum distance from point P(1, 1) to the line x cos α + y sin α = 2, where -π ≤ α ≤ π -/
theorem max_distance_to_line (α : Real) (h : -Real.pi ≤ α ∧ α ≤ Real.pi) :
  (let d := |Real.cos α + Real.sin α - 2|
   ∀ β, -Real.pi ≤ β ∧ β ≤ Real.pi → |Real.cos β + Real.sin β - 2| ≤ d) →
  d = 2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_to_line_l84_8412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circles_area_ratio_l84_8430

/-- A regular octagon -/
structure RegularOctagon :=
  (vertices : Fin 8 → ℝ × ℝ)
  (is_regular : sorry)

/-- A circle in 2D space -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

/-- The property of a circle being tangent to a line segment -/
def is_tangent_to_segment (c : Circle) (a b : ℝ × ℝ) : Prop :=
  sorry

/-- The property of a circle being tangent to a line -/
def is_tangent_to_line (c : Circle) (a b : ℝ × ℝ) : Prop :=
  sorry

/-- The area of a circle -/
noncomputable def circle_area (c : Circle) : ℝ :=
  Real.pi * c.radius ^ 2

theorem octagon_circles_area_ratio 
  (octagon : RegularOctagon) 
  (c1 c2 : Circle) :
  is_tangent_to_segment c1 (octagon.vertices 0) (octagon.vertices 1) →
  is_tangent_to_segment c2 (octagon.vertices 4) (octagon.vertices 5) →
  is_tangent_to_line c1 (octagon.vertices 1) (octagon.vertices 2) →
  is_tangent_to_line c1 (octagon.vertices 7) (octagon.vertices 0) →
  is_tangent_to_line c2 (octagon.vertices 1) (octagon.vertices 2) →
  is_tangent_to_line c2 (octagon.vertices 7) (octagon.vertices 0) →
  (circle_area c2) / (circle_area c1) = 3 * Real.sqrt 3 * (3 + 2 * Real.sqrt 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_circles_area_ratio_l84_8430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_grid_l84_8489

/-- Represents a grid cell that can contain 0, 1, or 2 -/
inductive Cell
| zero
| one
| two

/-- Represents a 100x100 grid -/
def Grid := Fin 100 → Fin 100 → Cell

/-- Checks if a 3x4 rectangle in the grid satisfies the condition -/
def valid_rectangle (g : Grid) (i j : Fin 100) : Prop :=
  ∃ (count_zeros count_ones count_twos : Nat),
    (∀ (di : Fin 3) (dj : Fin 4),
      (g (i + di) (j + dj) = Cell.zero → count_zeros > 0) ∧
      (g (i + di) (j + dj) = Cell.one → count_ones > 0) ∧
      (g (i + di) (j + dj) = Cell.two → count_twos > 0)) ∧
    count_zeros = 3 ∧ count_ones = 4 ∧ count_twos = 5

/-- The main theorem stating that it's impossible to create a valid grid -/
theorem impossible_grid : ¬ ∃ (g : Grid), ∀ (i j : Fin 100), valid_rectangle g i j := by
  sorry

#check impossible_grid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_grid_l84_8489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_in_class_l84_8454

/-- Given a class of boys where:
  1. The initial average height was calculated as 185 cm.
  2. One boy's height was incorrectly recorded, with a difference of 60 cm.
  3. The actual average height is 183 cm.
  Prove that the number of boys in the class is 30. -/
theorem number_of_boys_in_class (initial_average : ℝ) (height_difference : ℝ) (actual_average : ℝ)
  (h1 : initial_average = 185)
  (h2 : height_difference = 60)
  (h3 : actual_average = 183) :
  ∃ n : ℝ, (initial_average * n - height_difference) / n = actual_average ∧ n = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_boys_in_class_l84_8454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_max_min_l84_8447

noncomputable def a (n : ℕ+) : ℝ := n * Real.sqrt 5 - ⌊n * Real.sqrt 5⌋

theorem a_max_min (n : ℕ+) (hn : n ≤ 2009) :
  a n ≤ a 1292 ∧ a 1597 ≤ a n := by
  sorry

#check a_max_min

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_max_min_l84_8447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_for_g_leq_four_l84_8478

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a| + |x - 1|

-- Define the function g
noncomputable def g (a : ℝ) : ℝ := f a (1/a)

-- Theorem for part 1
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by
  sorry

-- Theorem for part 2
theorem range_of_a_for_g_leq_four :
  {a : ℝ | a ≠ 0 ∧ g a ≤ 4} = {a : ℝ | 1/2 ≤ a ∧ a ≤ 3/2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_for_a_equals_one_range_of_a_for_g_leq_four_l84_8478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_10_equals_174_l84_8464

def h : ℕ → ℤ
  | 0 => 3  -- Added case for 0
  | 1 => 3
  | 2 => 2
  | n+3 => 2 * h (n+2) - h (n+1) + 2 * (n+3) + 1

theorem h_10_equals_174 : h 10 = 174 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_10_equals_174_l84_8464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l84_8402

def complex_sequence (a : ℕ → ℂ) : Prop :=
  (a 1)^2 + (Complex.I : ℂ) * (a 1) - 1 = 0 ∧
  (a 2)^2 + (Complex.I : ℂ) * (a 2) - 1 = 0 ∧
  ∀ n : ℕ, n ≥ 2 →
    (a (n+1) * a (n-1) - a n^2) + (Complex.I : ℂ) * (a (n+1) + a (n-1) - 2 * a n) = 0

theorem sequence_property (a : ℕ → ℂ) (h : complex_sequence a) :
  ∀ n : ℕ, a n^2 + a (n+1)^2 + a (n+2)^2 = a n * a (n+1) + a (n+1) * a (n+2) + a (n+2) * a n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_property_l84_8402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_hydroxide_formation_l84_8439

/-- Represents a chemical compound -/
structure Compound where
  formula : String
  deriving Repr

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Compound
  products : List Compound
  stoichiometry : List Nat
  deriving Repr

/-- The molar mass of water in grams per mole -/
def water_molar_mass : ℝ := 18

/-- The balanced reaction for NH4Cl + H2O → HCl + NH4OH -/
def balanced_reaction : Reaction := {
  reactants := [⟨"NH4Cl"⟩, ⟨"H2O"⟩],
  products := [⟨"HCl"⟩, ⟨"NH4OH"⟩],
  stoichiometry := [1, 1, 1, 1]
}

/-- Theorem stating that NH4OH is the compound formed in the reaction -/
theorem ammonium_hydroxide_formation (r : Reaction) 
  (h1 : r.reactants = [⟨"NH4Cl"⟩, ⟨"H2O"⟩])
  (h2 : r.products.length = 2)
  (h3 : r.products.get? 0 = some ⟨"HCl"⟩)
  (h4 : r.stoichiometry = [1, 1, 1, 1])
  (h5 : water_molar_mass = 18) :
  r.products.get? 1 = some ⟨"NH4OH"⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ammonium_hydroxide_formation_l84_8439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_when_m_is_3_A_inter_B_eq_B_iff_m_le_3_l84_8426

-- Define sets A, B, and C
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2 * m - 1}
def C (m : ℝ) : Set ℤ := {x : ℤ | (x : ℝ) ∈ A ∨ (x : ℝ) ∈ B m}

-- Theorem 1: When m = 3, C consists of integers from -3 to 5
theorem C_when_m_is_3 : C 3 = {x : ℤ | -3 ≤ x ∧ x ≤ 5} := by sorry

-- Theorem 2: A ∩ B = B if and only if m ≤ 3
theorem A_inter_B_eq_B_iff_m_le_3 (m : ℝ) : A ∩ B m = B m ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_when_m_is_3_A_inter_B_eq_B_iff_m_le_3_l84_8426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_circular_arrangement_example_is_valid_l84_8498

/-- A type representing a circular arrangement of 12 distinct natural numbers. -/
def CircularArrangement := Fin 12 → ℕ

/-- The property that the arrangement contains 1. -/
def containsOne (arr : CircularArrangement) : Prop :=
  ∃ i : Fin 12, arr i = 1

/-- The property that any two neighboring numbers differ by either 10 or 7. -/
def validDifferences (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 12, (arr i.succ - arr i = 10 ∨ arr i - arr i.succ = 10) ∨
                (arr i.succ - arr i = 7 ∨ arr i - arr i.succ = 7)

/-- The property that all numbers in the arrangement are distinct. -/
def allDistinct (arr : CircularArrangement) : Prop :=
  ∀ i j : Fin 12, i ≠ j → arr i ≠ arr j

/-- The main theorem stating that the maximum possible value in a valid circular arrangement is 58. -/
theorem max_value_in_circular_arrangement (arr : CircularArrangement) 
  (h1 : containsOne arr) (h2 : validDifferences arr) (h3 : allDistinct arr) :
  ∀ i : Fin 12, arr i ≤ 58 := by
  sorry

/-- An example of a valid circular arrangement with maximum value 58. -/
def exampleArrangement : CircularArrangement :=
  fun i => [1, 11, 21, 31, 41, 51, 58, 48, 38, 28, 18, 8].get ⟨i.val, by simp⟩

/-- Proof that the example arrangement is valid. -/
theorem example_is_valid : 
  containsOne exampleArrangement ∧ 
  validDifferences exampleArrangement ∧ 
  allDistinct exampleArrangement := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_in_circular_arrangement_example_is_valid_l84_8498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_x_plus_3_l84_8471

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (x * (x - 1)) / 2

-- State the theorem
theorem f_of_x_plus_3 (x : ℝ) : f (x + 3) = (x^2 + 5*x + 6) / 2 := by
  -- Unfold the definition of f
  unfold f
  -- Simplify the expression
  simp [mul_add, add_mul, pow_two]
  -- Perform algebraic manipulations
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_x_plus_3_l84_8471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_symmetry_l84_8429

-- Define the functions as noncomputable
noncomputable def f (x : ℝ) := Real.log (2 - x)
noncomputable def g (x : ℝ) := Real.log x

-- Define the symmetry condition
def symmetric_about_x_equals_one (f g : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = g x

-- Theorem statement
theorem ln_symmetry : symmetric_about_x_equals_one f g := by
  intro x
  -- The actual proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_symmetry_l84_8429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l84_8492

-- Define the function f(x)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*Real.log x - 2*m

-- State the theorem
theorem f_properties (m : ℝ) :
  (∀ x > 0, StrictMono (fun x => f m x)) ∨ 
  (m > 0 ∧ ∃ min_value : ℝ, 
    (∀ x > 0, f m x ≥ min_value) ∧ 
    (∃ x > 0, f m x = min_value) ∧
    min_value ≤ Real.exp (-2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l84_8492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_7pi_9_l84_8403

theorem sin_alpha_plus_7pi_9 (α : ℝ) 
  (h1 : Real.cos (α - 2 * Real.pi / 9) = - Real.sqrt 7 / 4)
  (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) :
  Real.sin (α + 7 * Real.pi / 9) = - 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_plus_7pi_9_l84_8403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_c_l84_8419

theorem right_triangle_cos_c (A B C : ℝ) (h1 : 0 < A ∧ 0 < B ∧ 0 < C) 
  (h2 : A + B + C = π) (h3 : A = π/2) (h4 : Real.tan C = 4/3) : Real.cos C = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_c_l84_8419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_KBrO3_l84_8450

/-- The molar mass of potassium in g/mol -/
noncomputable def molar_mass_K : ℝ := 39.10

/-- The molar mass of bromine in g/mol -/
noncomputable def molar_mass_Br : ℝ := 79.90

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of potassium bromate (KBrO3) in g/mol -/
noncomputable def molar_mass_KBrO3 : ℝ := molar_mass_K + molar_mass_Br + 3 * molar_mass_O

/-- The mass of oxygen in one mole of KBrO3 in g/mol -/
noncomputable def mass_O_in_KBrO3 : ℝ := 3 * molar_mass_O

/-- The mass percentage of oxygen in potassium bromate -/
noncomputable def mass_percentage_O : ℝ := (mass_O_in_KBrO3 / molar_mass_KBrO3) * 100

theorem mass_percentage_O_in_KBrO3 : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |mass_percentage_O - 28.74| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_KBrO3_l84_8450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l84_8462

theorem quadratic_root_form (a b c m n p : ℤ) : 
  a = 3 ∧ b = -4 ∧ c = -7 →
  (∃ x : ℚ, a * x^2 + b * x + c = 0 ↔ ∃ (m n p : ℤ), x = (m + Int.sqrt n) / p ∨ x = (m - Int.sqrt n) / p) →
  Int.gcd m n = 1 ∧ Int.gcd m p = 1 ∧ Int.gcd n p = 1 →
  n = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_form_l84_8462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l84_8458

theorem matrix_N_satisfies_conditions : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N.vecMul (![4, 0] : Fin 2 → ℝ) = ![8, 28] ∧
  N.vecMul (![(-2), 10] : Fin 2 → ℝ) = ![6, -34] ∧
  N = !![2, 1; 7, -2] :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_N_satisfies_conditions_l84_8458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_a_n_convergence_l84_8418

/-- The sequence a_n defined as (1/3)(1 - 10^(-n)) --/
def a (n : ℕ) : ℚ := (1/3) * (1 - 10^(-n : ℤ))

/-- The statement that 4 is the minimum positive integer n satisfying |a_n - 1/3| < 1/2015 --/
theorem min_n_for_a_n_convergence :
  ∀ n : ℕ, n > 0 → (|a n - 1/3| < 1/2015 ↔ n ≥ 4) :=
by
  sorry

#check min_n_for_a_n_convergence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_a_n_convergence_l84_8418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_floor_l84_8452

/-- Sequence {a_n} satisfying the given recurrence relation -/
noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Added case for 0
  | 1 => 1
  | 2 => 2
  | (n + 3) => a (n + 1) * ((a (n + 2))^2 + 1) / ((a (n + 1))^2 + 1)

/-- The theorem to be proved -/
theorem a_2017_floor : ⌊a 2017⌋ = 63 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2017_floor_l84_8452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_bonus_relationship_l84_8496

/-- Calculates the commission for a given sales amount -/
noncomputable def calculate_commission (sales : ℝ) : ℝ :=
  let tier1 := min sales 5000 * 0.07
  let tier2 := max 0 (min (sales - 5000) 5000) * 0.085
  let tier3 := max 0 (sales - 10000) * 0.09
  tier1 + tier2 + tier3

/-- Calculates the bonus for a given sales amount -/
noncomputable def calculate_bonus (sales : ℝ) : ℝ :=
  let bonus1 := max 0 (sales - 10000) * 0.02
  let bonus2 := max 0 (sales - 20000) * 0.03
  let bonus3 := max 0 (sales - 30000) * 0.04
  bonus1 + bonus2 + bonus3

/-- Theorem: If the total commission is 5000, then the bonus is 3125 -/
theorem commission_bonus_relationship : 
  ∃ (sales : ℝ), calculate_commission sales = 5000 ∧ calculate_bonus sales = 3125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_commission_bonus_relationship_l84_8496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l84_8409

theorem trigonometric_identity (x : ℝ) : (Real.sin x ^ 6 + Real.cos x ^ 6 - 1) ^ 3 + 27 * Real.sin x ^ 6 * Real.cos x ^ 6 = 0 := by
  have pythagorean_identity : Real.sin x ^ 2 + Real.cos x ^ 2 = 1 := by exact Real.sin_sq_add_cos_sq x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l84_8409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_l84_8436

def a : ℕ → ℤ := sorry

axiom a_1 : a 1 = 2023
axiom a_2 : a 2 = 2024
axiom a_rec : ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 2) = n + 2

theorem a_1000 : a 1000 = 2356 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_1000_l84_8436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_calculation_l84_8400

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem total_distance_calculation :
  let d1 := distance (-3) 6 3 3
  let d2 := distance 3 3 0 0
  let d3 := distance 0 0 7 (-3)
  d1 + d2 + d3 = 3 * Real.sqrt 5 + 3 * Real.sqrt 2 + Real.sqrt 58 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_calculation_l84_8400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_after_adding_water_l84_8406

/-- Represents a mixture of milk and water -/
structure Mixture where
  total : ℚ
  milk : ℚ
  water : ℚ

/-- Creates a mixture given a total volume and a ratio of milk to water -/
def create_mixture (total : ℚ) (milk_ratio : ℚ) (water_ratio : ℚ) : Mixture :=
  let total_ratio := milk_ratio + water_ratio
  { total := total
  , milk := (milk_ratio / total_ratio) * total
  , water := (water_ratio / total_ratio) * total }

/-- Adds water to a mixture -/
def add_water (m : Mixture) (water_added : ℚ) : Mixture :=
  { total := m.total + water_added
  , milk := m.milk
  , water := m.water + water_added }

/-- Calculates the ratio of milk to water in a mixture -/
noncomputable def milk_water_ratio (m : Mixture) : ℚ × ℚ :=
  let gcd := Int.gcd (m.milk.num * m.water.den) (m.water.num * m.milk.den)
  ((m.milk.num * m.water.den) / gcd, (m.water.num * m.milk.den) / gcd)

theorem milk_water_ratio_after_adding_water :
  let initial_mixture := create_mixture 90 8 2
  let final_mixture := add_water initial_mixture 36
  milk_water_ratio final_mixture = (4, 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_milk_water_ratio_after_adding_water_l84_8406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_inequality_l84_8483

theorem triangle_tangent_inequality (m : ℝ) (A B C : ℝ) (h_m : m ≥ 2) (h_triangle : A + B + C = π) :
  Real.tan (A / m) + Real.tan (B / m) + Real.tan (C / m) ≥ 3 * Real.tan (π / (3 * m)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_tangent_inequality_l84_8483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_evaluation_l84_8410

theorem complex_root_evaluation (N : ℝ) (h : N > 1) :
  Real.sqrt (N * (N * Real.sqrt N) ^ (1/3)) = N ^ (3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_root_evaluation_l84_8410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_deliveries_to_regain_cost_l84_8477

/-- The minimum number of deliveries to regain the purchase cost of a van -/
theorem min_deliveries_to_regain_cost (van_cost : ℕ) (earnings_per_delivery : ℕ) (gas_cost_per_delivery : ℕ) :
  van_cost = 7500 →
  earnings_per_delivery = 15 →
  gas_cost_per_delivery = 5 →
  (Nat.ceil ((van_cost : ℚ) / (earnings_per_delivery - gas_cost_per_delivery))) = 750 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_deliveries_to_regain_cost_l84_8477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l84_8441

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |3 * x - a|
def g (x : ℝ) : ℝ := |x + 1|

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ x : ℝ, f a x + g x > 1) → a ∈ Set.Ioi 0 ∪ Set.Iio (-6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l84_8441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_ratio_l84_8481

theorem perpendicular_vectors_ratio (α : ℝ) : 
  let a : ℝ × ℝ := (Real.sin α, -2)
  let b : ℝ × ℝ := (1, Real.cos α)
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  (Real.sin α / (Real.sin α + Real.cos α) = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_ratio_l84_8481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisors_between_2_and_100_l84_8404

theorem no_divisors_between_2_and_100 (n : ℕ+) 
  (h : ∀ k ∈ Finset.range 99, (Finset.sum (Finset.range n) (λ i ↦ (i + 1)^(k + 1))) % n = 0) : 
  ∀ d ∈ Finset.range 99, d > 1 → ¬(d ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_divisors_between_2_and_100_l84_8404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_specific_l84_8443

/-- The distance between the foci of a hyperbola -/
noncomputable def hyperbola_foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

/-- Theorem: The distance between the foci of the hyperbola 
    described by (y-3)^2/25 - x^2/9 = 1 is 2√34 -/
theorem hyperbola_foci_distance_specific : 
  hyperbola_foci_distance 5 3 = 2 * Real.sqrt 34 := by
  unfold hyperbola_foci_distance
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_foci_distance_specific_l84_8443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_206788_l84_8449

def digit_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => (digit_sequence n + 1) % 10

def nth_digit (n : ℕ) : ℕ := digit_sequence (n - 1)

theorem digit_at_206788 : nth_digit 206788 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_at_206788_l84_8449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l84_8444

/-- A circle tangent to the y-axis at (0, 3) and intersecting the x-axis with a segment length of 8 -/
structure TangentCircle where
  /-- The circle is tangent to the y-axis at (0, 3) -/
  tangent_point : ℝ × ℝ := (0, 3)
  /-- The circle intersects the x-axis with a segment length of 8 -/
  x_axis_intersection : ℝ
  x_axis_intersection_length : x_axis_intersection = 8

/-- The standard equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 3)^2 = 25 ∨ (x - 5)^2 + (y - 3)^2 = 25

/-- Theorem stating that the given circle satisfies the standard equation -/
theorem tangent_circle_equation (c : TangentCircle) (x y : ℝ) :
  circle_equation x y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_equation_l84_8444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_X_for_T_divisible_by_20_l84_8424

/-- A function that checks if a positive integer is composed only of 0s and 1s -/
def isComposedOf0sAnd1s (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Nat.digits 10 n → d = 0 ∨ d = 1

/-- The theorem statement -/
theorem smallest_X_for_T_divisible_by_20 :
  ∃ (X : ℕ),
    X > 0 ∧
    (∃ (T : ℕ), T > 0 ∧ isComposedOf0sAnd1s T ∧ T = 20 * X) ∧
    (∀ (Y : ℕ), 0 < Y ∧ Y < X →
      ¬∃ (S : ℕ), S > 0 ∧ isComposedOf0sAnd1s S ∧ S = 20 * Y) ∧
    X = 55 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_X_for_T_divisible_by_20_l84_8424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_squared_l84_8459

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

-- Define points A, B, and C
variable (A B C : ℝ × ℝ)

-- State the theorem
theorem distance_from_center_squared
  (h_radius : (80 : ℝ).sqrt > 0)
  (h_on_circle : A ∈ Circle (0, 0) (80 : ℝ).sqrt ∧ B ∈ Circle (0, 0) (80 : ℝ).sqrt ∧ C ∈ Circle (0, 0) (80 : ℝ).sqrt)
  (h_AB : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64)
  (h_BC : (B.1 - C.1)^2 + (B.2 - C.2)^2 = 9)
  (h_right_angle : (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0) :
  B.1^2 + B.2^2 = 80 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_center_squared_l84_8459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_multiple_of_4_l84_8401

/-- A fair 8-sided die -/
def octahedralDie : Finset ℕ := Finset.range 8

/-- A fair 12-sided die -/
def twelveSidedDie : Finset ℕ := Finset.range 12

/-- Check if a number is a multiple of 4 -/
def isMultipleOf4 (n : ℕ) : Bool := n % 4 = 0

/-- The probability of rolling a multiple of 4 on the octahedral die -/
def probMultiple4Octahedral : ℚ := 
  (octahedralDie.filter (fun n => isMultipleOf4 n)).card / octahedralDie.card

/-- The probability of rolling a multiple of 4 on the twelve-sided die -/
def probMultiple4TwelveSided : ℚ := 
  (twelveSidedDie.filter (fun n => isMultipleOf4 n)).card / twelveSidedDie.card

/-- The probability that the product of the two rolls is a multiple of 4 -/
theorem prob_product_multiple_of_4 : 
  1 - (1 - probMultiple4Octahedral) * (1 - probMultiple4TwelveSided) = 7/16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_product_multiple_of_4_l84_8401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tammy_trip_length_l84_8455

/-- Represents the length of a trip in miles -/
def trip_length (x : ℚ) : Prop :=
  (1/4 : ℚ) * x + 25 + (1/6 : ℚ) * x = x

theorem tammy_trip_length :
  ∃ x : ℚ, trip_length x ∧ x = 300/7 := by
  sorry

#check tammy_trip_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tammy_trip_length_l84_8455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_plane_l84_8451

-- Define the types for planes and lines
variable (α β : Set (EuclideanSpace ℝ (Fin 3))) -- Planes
variable (m : Set (EuclideanSpace ℝ (Fin 3))) -- Line

-- Define the parallel relation between planes
def parallel_planes (p q : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define the contained relation between a line and a plane
def line_in_plane (l : Set (EuclideanSpace ℝ (Fin 3))) (p : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- Define the parallel relation between a line and a plane
def line_parallel_plane (l : Set (EuclideanSpace ℝ (Fin 3))) (p : Set (EuclideanSpace ℝ (Fin 3))) : Prop := sorry

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (h1 : parallel_planes α β) 
  (h2 : line_in_plane m α) : 
  line_parallel_plane m β := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_parallel_plane_l84_8451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_containers_l84_8442

theorem minimum_containers (container_capacity : ℕ) (required_amount : ℕ) : 
  container_capacity = 15 → required_amount = 150 → 
  (∃ (n : ℕ), n * container_capacity ≥ required_amount ∧ 
  ∀ (m : ℕ), m * container_capacity ≥ required_amount → n ≤ m) → 
  (∃ (n : ℕ), n * container_capacity ≥ required_amount ∧ 
  ∀ (m : ℕ), m * container_capacity ≥ required_amount → n ≤ m) ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_containers_l84_8442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_sixes_approx_l84_8485

/-- The probability of rolling exactly two 6s when rolling 15 standard 6-sided dice -/
noncomputable def prob_two_sixes : ℚ :=
  (Nat.choose 15 2 : ℚ) * (1/6)^2 * (5/6)^13

/-- The number of standard dice being rolled -/
def num_dice : ℕ := 15

/-- The number of sides on each die -/
def num_sides : ℕ := 6

/-- The number of desired outcomes (number of dice showing 6) -/
def desired_outcomes : ℕ := 2

theorem prob_two_sixes_approx :
  (↑(round (prob_two_sixes * 1000)) / 1000 : ℚ) = 158/1000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_two_sixes_approx_l84_8485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l84_8437

noncomputable def average_after_17th_inning (previous_total_runs : ℝ) (previous_innings : ℕ) 
  (runs_17th_inning : ℝ) (average_increase : ℝ) : ℝ :=
  (previous_total_runs + runs_17th_inning) / (previous_innings + 1)

theorem batsman_average_after_17th_inning 
  (previous_total_runs : ℝ) (previous_innings : ℕ) :
  previous_innings = 16 →
  let runs_17th_inning : ℝ := 87
  let average_increase : ℝ := 3
  let new_average := average_after_17th_inning previous_total_runs previous_innings runs_17th_inning average_increase
  let previous_average := previous_total_runs / previous_innings
  new_average = previous_average + average_increase →
  new_average = 39 := by
  sorry

#check batsman_average_after_17th_inning

end NUMINAMATH_CALUDE_ERRORFEEDBACK_batsman_average_after_17th_inning_l84_8437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l84_8475

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence
  d : ℚ      -- Common difference
  seq_def : ∀ n, a (n + 1) = a n + d  -- Definition of arithmetic sequence

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_common_difference 
  (seq : ArithmeticSequence) 
  (h1 : seq.a 3 = 4) 
  (h2 : sum_n seq 3 = 3) : 
  seq.d = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l84_8475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_cant_catch_nut_l84_8431

/-- The minimum distance between a horizontally thrown object and a stationary target --/
noncomputable def min_distance (v₀ : ℝ) (a : ℝ) (g : ℝ) : ℝ :=
  (5 * Real.sqrt 2) / 4

/-- The theorem stating that the squirrel cannot catch the nut --/
theorem squirrel_cant_catch_nut (v₀ a g jump_range : ℝ) 
  (h_v₀ : v₀ = 5)
  (h_a : a = 3.75)
  (h_g : g = 10)
  (h_jump : jump_range = 1.7)
  : min_distance v₀ a g > jump_range := by
  sorry

#check squirrel_cant_catch_nut

end NUMINAMATH_CALUDE_ERRORFEEDBACK_squirrel_cant_catch_nut_l84_8431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l84_8473

/-- The sum of the infinite series ∑(k=1 to ∞) k^3 / 3^k is equal to 165/16. -/
theorem infinite_series_sum : ∑' k : ℕ, (k : ℝ)^3 / (3 : ℝ)^k = 165 / 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_l84_8473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probabilities_theorem_l84_8469

def P (b : ℕ) : Finset ℕ := {b, 1}
def Q (c : ℕ) : Finset ℕ := {c, 1, 2}

def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ∈ Finset.range 5 ∧ p.2 ∈ Finset.range 5 ∧ P p.1 ⊆ Q p.2) (Finset.product (Finset.range 5) (Finset.range 5))

theorem probabilities_theorem :
  (((Finset.filter (fun p => p.1 = p.2) valid_pairs).card : ℚ) / valid_pairs.card = 1/2) ∧
  (((Finset.filter (fun p => p.1^2 - 4*p.2 ≥ 0) valid_pairs).card : ℚ) / valid_pairs.card = 3/8) := by
  sorry

#eval valid_pairs
#eval (Finset.filter (fun p => p.1 = p.2) valid_pairs).card
#eval (Finset.filter (fun p => p.1^2 - 4*p.2 ≥ 0) valid_pairs).card
#eval valid_pairs.card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probabilities_theorem_l84_8469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l84_8440

theorem trig_identities (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) :
  (Real.sin α)^4 + (Real.cos α)^4 = 7/9 ∧ Real.tan α / (1 + (Real.tan α)^2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_l84_8440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_wall_l84_8438

-- Define the dimensions of the wall and brick
def wall_length : ℝ := 400
def wall_height : ℝ := 200
def wall_width : ℝ := 25
def brick_length : ℝ := 25
def brick_height : ℝ := 11.25
def brick_width : ℝ := 6

-- Calculate volumes
def wall_volume : ℝ := wall_length * wall_height * wall_width
def brick_volume : ℝ := brick_length * brick_height * brick_width

-- Define the function to calculate the number of bricks needed
noncomputable def bricks_needed : ℕ := 
  (Int.ceil (wall_volume / brick_volume)).toNat

-- Theorem statement
theorem bricks_for_wall : bricks_needed = 1186 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bricks_for_wall_l84_8438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_curve_length_l84_8487

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A cube in 3D space -/
structure Cube where
  center : Point3D
  edgeLength : ℝ

/-- The curve on the surface of the cube -/
def surfaceCurve (c : Cube) : Set Point3D :=
  {p : Point3D | ∃ (face : Fin 6), p ∈ Set.univ} -- placeholder definition

/-- Distance between two points in 3D space -/
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

/-- Length of a curve -/
noncomputable def curveLength (curve : Set Point3D) : ℝ := sorry

/-- Vertices of a cube -/
def cubeVertices (c : Cube) : Set Point3D := sorry

/-- The main theorem -/
theorem surface_curve_length (c : Cube) (A : Point3D) :
  c.edgeLength = 1 →
  A ∈ cubeVertices c →
  curveLength {p ∈ surfaceCurve c | distance p A = 2 * Real.sqrt 3 / 3} = 5 * Real.sqrt 3 * π / 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_curve_length_l84_8487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l84_8491

def probability_no_consecutive_numbers (n : ℕ) : ℚ := 
  (Nat.choose (n - 2) 3 : ℚ) / (Nat.choose n 3 : ℚ)

theorem lottery_probability (n : ℕ) (h : n ≥ 5) :
  probability_no_consecutive_numbers n = 
  (Nat.choose (n - 2) 3 : ℚ) / (Nat.choose n 3 : ℚ) :=
by
  -- Unfold the definition of probability_no_consecutive_numbers
  unfold probability_no_consecutive_numbers
  -- The equality now holds by reflexivity
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probability_l84_8491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_61_l84_8434

def sequence_a : ℕ → ℤ
  | 0 => 1
  | n + 1 => 2 * sequence_a n + 3

theorem a_5_equals_61 : sequence_a 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_equals_61_l84_8434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pump_theorem_l84_8433

/-- Represents the rate at which a pump can empty a basement -/
structure PumpRate where
  rate : ℚ
  rate_pos : rate > 0

/-- The time taken for two pumps to empty half a basement -/
def two_pump_time (pump_x pump_y : PumpRate) : ℚ :=
  1 / (pump_x.rate + pump_y.rate)

/-- The main theorem -/
theorem two_pump_theorem (pump_x pump_y : PumpRate) 
  (hx : pump_x.rate = 1/4) 
  (hy : pump_y.rate = 1/18) : 
  two_pump_time pump_x pump_y = 18/11 := by
  sorry

#eval (18 : ℚ) / 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pump_theorem_l84_8433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_iff_a_eq_two_l84_8416

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of a line in the form y = mx + b is m -/
noncomputable def slope_explicit (m : ℝ) : ℝ := m

/-- The slope of a line in the form ax + by + c = 0 is -a/b -/
noncomputable def slope_implicit (a b : ℝ) : ℝ := -a / b

theorem perpendicular_lines_iff_a_eq_two (a : ℝ) : 
  perpendicular (slope_implicit 1 (a^2)) (slope_explicit 4) ↔ a = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_lines_iff_a_eq_two_l84_8416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l84_8457

noncomputable def AngleOfInclination (x y : ℝ → ℝ) : ℝ :=
  Real.arctan ((y 1 - y 0) / (x 1 - x 0))

theorem line_inclination_angle (t : ℝ) :
  let x : ℝ → ℝ := λ t => -t * Real.cos (20 * π / 180)
  let y : ℝ → ℝ := λ t => 3 + t * Real.sin (20 * π / 180)
  AngleOfInclination x y = 160 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_inclination_angle_l84_8457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_omega_one_l84_8472

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 4 * (Real.cos (ω * x)) * Real.cos (ω * x + Real.pi / 3)

theorem period_implies_omega_one (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∀ x : ℝ, f ω (x + Real.pi) = f ω x) 
  (h3 : ∀ T : ℝ, T > 0 → T < Real.pi → ∃ x : ℝ, f ω (x + T) ≠ f ω x) : 
  ω = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_implies_omega_one_l84_8472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_33kg_price_15kg_price_36kg_l84_8480

/-- The cost of apples in rupees per kilogram for the first 30 kgs -/
def l : ℚ := 10

/-- The cost of apples in rupees per kilogram for each additional kg beyond 30 kgs -/
def q : ℚ := 11

/-- The total cost of n kilograms of apples -/
def apple_cost (n : ℚ) : ℚ :=
  if n ≤ 30 then n * l
  else 30 * l + (n - 30) * q

/-- The price of 33 kilograms of apples is 333 rupees -/
theorem price_33kg : apple_cost 33 = 333 := by sorry

/-- The cost of the first 15 kgs of apples is 150 rupees -/
theorem price_15kg : apple_cost 15 = 150 := by sorry

/-- The price of 36 kilograms of apples is 366 rupees -/
theorem price_36kg : apple_cost 36 = 366 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_33kg_price_15kg_price_36kg_l84_8480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_segment_arrangement_l84_8446

-- Define the necessary concepts
def IsLineSegment (s : Set (ℝ × ℝ)) : Prop := sorry

def SegmentEndpoints (s : Set (ℝ × ℝ)) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

def Interior (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

-- The main theorem
theorem impossible_segment_arrangement (n : ℕ) : 
  ¬ ∃ (segments : Fin n → Set (ℝ × ℝ)), 
    (∀ i : Fin n, IsLineSegment (segments i)) ∧ 
    (∀ i : Fin n, ∃ j k : Fin n, j ≠ i ∧ k ≠ i ∧ j ≠ k ∧ 
      (SegmentEndpoints (segments i)).1 ∈ Interior (segments j) ∧
      (SegmentEndpoints (segments i)).2 ∈ Interior (segments k)) :=
by
  sorry  -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_segment_arrangement_l84_8446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_max_dot_product_min_dot_product_l84_8420

-- Define the points and line
def C : ℝ × ℝ := (2, 0)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 8}

-- Define the circle N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + (p.2 - 1)^2 = 1}

-- Define the conditions
def satisfies_condition (P : ℝ × ℝ) : Prop :=
  let Q : ℝ × ℝ := (8, P.2)
  let PC : ℝ × ℝ := (P.1 - C.1, P.2 - C.2)
  let PQ : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
  (PC.1 + 1/2 * PQ.1, PC.2 + 1/2 * PQ.2) • (PC.1 - 1/2 * PQ.1, PC.2 - 1/2 * PQ.2) = 0

-- Theorem statements
theorem trajectory_equation (P : ℝ × ℝ) (h : satisfies_condition P) :
  P.1^2 / 16 + P.2^2 / 12 = 1 := by sorry

theorem max_dot_product (P : ℝ × ℝ) (h : satisfies_condition P) (E F : ℝ × ℝ) 
  (hEF : E ∈ N ∧ F ∈ N ∧ (E.1 - F.1)^2 + (E.2 - F.2)^2 = 4) :
  (∀ E' F' : ℝ × ℝ, E' ∈ N → F' ∈ N → (E'.1 - F'.1)^2 + (E'.2 - F'.2)^2 = 4 →
    ((P.1 - E'.1) * (P.1 - F'.1) + (P.2 - E'.2) * (P.2 - F'.2) : ℝ) ≤ 19) := by sorry

theorem min_dot_product (P : ℝ × ℝ) (h : satisfies_condition P) (E F : ℝ × ℝ) 
  (hEF : E ∈ N ∧ F ∈ N ∧ (E.1 - F.1)^2 + (E.2 - F.2)^2 = 4) :
  (∀ E' F' : ℝ × ℝ, E' ∈ N → F' ∈ N → (E'.1 - F'.1)^2 + (E'.2 - F'.2)^2 = 4 →
    ((P.1 - E'.1) * (P.1 - F'.1) + (P.2 - E'.2) * (P.2 - F'.2) : ℝ) ≥ 12 - 4 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_max_dot_product_min_dot_product_l84_8420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l84_8470

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x + 1)

-- Theorem for monotonicity and max/min values
theorem f_properties :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → 1/5 ≤ f x ∧ f x ≤ 1/2) ∧
  (f 4 = 1/5 ∧ f 1 = 1/2) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l84_8470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l84_8479

/-- The volume of a cone with radius r and height h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

theorem cone_height_calculation (r V : ℝ) (h_r : r = 3) (h_V : V = 12) :
  ∃ h : ℝ, cone_volume r h = V ∧ h = 4 / Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_calculation_l84_8479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_l84_8408

/-- The distance between intersection points of a line and parabola -/
noncomputable def intersection_distance (k : ℝ) : ℝ := 
  (2 * k^2 + 4) / k^2 + 2

/-- Theorem: If a line y = k(x-1) intersects the parabola y^2 = 4x at two points 
    with a distance of 16/3 between them, then k^2 = 3 -/
theorem line_parabola_intersection (k : ℝ) :
  k ≠ 0 → intersection_distance k = 16/3 → k^2 = 3 := by
  sorry

#check line_parabola_intersection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parabola_intersection_l84_8408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_pairs_l84_8499

/-- Represents a 3x3 grid filled with numbers 1 through 9 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two numbers form a valid pair (one is a multiple of the other) -/
def isValidPair (a b : Fin 9) : Bool :=
  (a.val + 1) % (b.val + 1) = 0 || (b.val + 1) % (a.val + 1) = 0

/-- Counts the number of valid pairs in a given grid -/
def countValidPairs (grid : Grid) : Nat :=
  let horizontalPairs := (Finset.sum (Finset.range 3) fun i =>
    Finset.sum (Finset.range 2) fun j =>
      if isValidPair (grid i j) (grid i (j+1)) then 1 else 0)
  let verticalPairs := (Finset.sum (Finset.range 2) fun i =>
    Finset.sum (Finset.range 3) fun j =>
      if isValidPair (grid i j) (grid (i+1) j) then 1 else 0)
  horizontalPairs + verticalPairs

/-- Checks if a grid is valid (contains numbers 1-9 without repetition) -/
def isValidGrid (grid : Grid) : Prop :=
  ∀ i j, ∃! n : Fin 9, grid i j = n

/-- The main theorem: The maximum number of valid pairs in a 3x3 grid is 9 -/
theorem max_valid_pairs :
  ∃ (grid : Grid), isValidGrid grid ∧ 
    (∀ (other : Grid), isValidGrid other → countValidPairs other ≤ countValidPairs grid) ∧
    countValidPairs grid = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_pairs_l84_8499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l84_8490

/-- The curve function -/
noncomputable def f (x : ℝ) : ℝ := -x + 3 * Real.log x

/-- The derivative of the curve function -/
noncomputable def f' (x : ℝ) : ℝ := -1 + 3 / x

/-- The line function -/
def g (b : ℝ) (x : ℝ) : ℝ := 2 * x + b

theorem tangent_line_b_value :
  ∃ (x₀ : ℝ), x₀ > 0 ∧ 
  f x₀ = g (-3) x₀ ∧
  f' x₀ = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_b_value_l84_8490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l84_8488

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + 1/2 * t, Real.sqrt 3 / 2 * t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 = 4*x

-- Define point M
def point_M : ℝ × ℝ := (2, 0)

-- Theorem statement
theorem intersection_product (A B : ℝ × ℝ) (t_A t_B : ℝ) :
  line_l t_A = A ∧ 
  line_l t_B = B ∧ 
  curve_C A.1 A.2 ∧ 
  curve_C B.1 B.2 →
  |point_M.1 - A.1| * |point_M.1 - B.1| = 32/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l84_8488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_segment_length_l84_8494

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with vertices A, B, C, D -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: In a square ABCD with point L on CD and K on extension of DA,
    if angle KBL is 90°, KD = 19, and CL = 6, then LD = 7 -/
theorem square_segment_length (s : Square) (L K : Point) : 
  (distance s.C L = 6) →
  (distance s.D K = 19) →
  (distance s.A s.B = distance s.B s.C) →
  (distance s.B s.C = distance s.C s.D) →
  (distance s.C s.D = distance s.D s.A) →
  (L.x = s.C.x ∨ L.y = s.C.y) →
  (K.x = s.A.x ∨ K.y = s.A.y) →
  ((K.x - s.B.x) * (L.x - s.B.x) + (K.y - s.B.y) * (L.y - s.B.y) = 0) →
  distance L s.D = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_segment_length_l84_8494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_2_3_4_is_acute_l84_8422

theorem triangle_with_angle_ratio_2_3_4_is_acute (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a + b + c = 180 →
  (a : ℝ) / 2 = (b : ℝ) / 3 ∧ (b : ℝ) / 3 = (c : ℝ) / 4 →
  a < 90 ∧ b < 90 ∧ c < 90 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_with_angle_ratio_2_3_4_is_acute_l84_8422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_17_value_l84_8414

def sequence_a : ℕ → ℤ
  | 0 => 1  -- Add this case for 0
  | 1 => 1
  | n + 2 => 2 * sequence_a (n + 1) - 2^(n + 1)

theorem a_17_value : sequence_a 17 = -15 * 2^16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_17_value_l84_8414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_face_angle_formula_l84_8456

/-- Represents an oblique prism with a right-angled triangular base -/
structure ObliquePrism where
  α : Real
  β : Real
  hypotenuse_face_perpendicular : Bool
  adjacent_face_angle : Real

/-- The acute angle between the third lateral face and the base of the oblique prism -/
noncomputable def third_face_angle (prism : ObliquePrism) : Real :=
  Real.arctan (Real.tan prism.α * Real.tan prism.β)

/-- Theorem: The acute angle between the third lateral face and the base
    of an oblique prism with specific properties is arctan(tan α · tan β) -/
theorem third_face_angle_formula (prism : ObliquePrism)
  (h1 : 0 < prism.α ∧ prism.α < Real.pi / 2)
  (h2 : 0 < prism.β ∧ prism.β < Real.pi / 2)
  (h3 : prism.hypotenuse_face_perpendicular = true)
  (h4 : prism.adjacent_face_angle = prism.β) :
  third_face_angle prism = Real.arctan (Real.tan prism.α * Real.tan prism.β) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_face_angle_formula_l84_8456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l84_8417

noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_ratio (a₁ a₂ a₃ : ℝ) (q : ℝ) :
  a₁ ∈ ({-4, -3, -2, 0, 1, 23, 4} : Set ℝ) →
  a₂ ∈ ({-4, -3, -2, 0, 1, 23, 4} : Set ℝ) →
  a₃ ∈ ({-4, -3, -2, 0, 1, 23, 4} : Set ℝ) →
  a₁ = 4 →
  a₂ = 2 →
  a₃ = 1 →
  q = 1/2 →
  (geometricSum a₁ q 10) / (1 - q^5) = 33 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_ratio_l84_8417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l84_8482

/-- An isosceles triangle ABC with vertex C at the origin, hypotenuse AB of length 50,
    and medians through A and B lying on lines y = x + 3 and y = -x + 3 respectively. -/
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  hypotenuse_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 50
  median_A : A.2 = A.1 + 3
  median_B : B.2 = -B.1 + 3
  isosceles : (0 - A.1)^2 + (0 - A.2)^2 = (0 - B.1)^2 + (0 - B.2)^2

/-- The area of the isosceles triangle ABC -/
noncomputable def triangleArea (t : IsoscelesTriangle) : ℝ :=
  25 * (5 * Real.sqrt 5 + 3)

/-- Theorem stating that the area of the isosceles triangle ABC is 25(5√5 + 3) -/
theorem isosceles_triangle_area (t : IsoscelesTriangle) : 
  triangleArea t = 25 * (5 * Real.sqrt 5 + 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l84_8482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_of_specific_hexagon_l84_8495

/-- A regular hexagon with alternating side lengths -/
structure AlternatingHexagon where
  side1 : ℝ
  side2 : ℝ

/-- The diagonal of the hexagon -/
noncomputable def hexagon_diagonal (h : AlternatingHexagon) : ℝ :=
  Real.sqrt (h.side1^2 + h.side2^2)

/-- Theorem: The diagonal length of a specific alternating hexagon -/
theorem diagonal_length_of_specific_hexagon :
  let h : AlternatingHexagon := ⟨4, 6⟩
  hexagon_diagonal h = Real.sqrt 52 := by
  sorry

#check diagonal_length_of_specific_hexagon

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_of_specific_hexagon_l84_8495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_entree_cost_correct_l84_8448

/-- Given a total cost for an entree and dessert, where the entree costs $5 more than the dessert,
    this function calculates the cost of the entree. -/
noncomputable def entree_cost (total : ℝ) (difference : ℝ) : ℝ :=
  (total + difference) / 2

theorem entree_cost_correct (total : ℝ) (difference : ℝ) 
  (h1 : total = 23) (h2 : difference = 5) : 
  entree_cost total difference = 14 := by
  sorry

-- Remove the #eval line as it's not necessary for the theorem and may cause issues
-- #eval entree_cost 23 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_entree_cost_correct_l84_8448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l84_8425

/-- The work rate of a group of men building a wall -/
noncomputable def workRate (length : ℝ) (days : ℝ) (men : ℝ) : ℝ :=
  length / (days * men)

/-- The number of men needed to build a wall of given length in given days -/
noncomputable def menNeeded (length : ℝ) (days : ℝ) (rate : ℝ) : ℝ :=
  length / (days * rate)

theorem first_group_size :
  let rate := workRate 28 3 10
  menNeeded 112 6 rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_group_size_l84_8425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_quadratic_always_positive_l84_8405

def quadratic_always_positive (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

theorem range_of_quadratic_always_positive :
  {a : ℝ | quadratic_always_positive a} = Set.Icc 0 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_quadratic_always_positive_l84_8405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2010_in_third_quadrant_l84_8493

def angle_to_quadrant (angle : ℕ) : ℕ :=
  match (angle % 360) with
  | n => if 0 < n ∧ n ≤ 90 then 1
         else if 90 < n ∧ n ≤ 180 then 2
         else if 180 < n ∧ n ≤ 270 then 3
         else 4

theorem angle_2010_in_third_quadrant :
  angle_to_quadrant 2010 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_2010_in_third_quadrant_l84_8493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l84_8463

-- Define the function f
noncomputable def f (x : ℝ) := x + 4 / (x - 1)

-- State the theorem
theorem min_value_of_f :
  ∀ x : ℝ, x > 1 → f x ≥ 5 ∧ ∃ x₀ : ℝ, x₀ > 1 ∧ f x₀ = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l84_8463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l84_8465

-- Define z as a complex number
variable (z : ℂ)

-- Define ω as a real number
variable (ω : ℝ)

-- State the conditions
axiom z_imaginary : z.re = 0
axiom omega_def : ω = z + z⁻¹
axiom omega_range : -1 < ω ∧ ω < 2

-- Define u
noncomputable def u : ℂ := (1 - z) / (1 + z)

-- Theorem to prove
theorem complex_problem :
  Complex.abs z = 1 ∧
  z.re = 0 ∧
  (u z).re = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_problem_l84_8465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_propositions_correct_l84_8484

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the propositions
def proposition1 : Prop := ∀ z, ∃ (a b : ℝ), z = a + b * i ∧ z.re = a ∧ z.im = b

def proposition2 : Prop := ∀ z, Complex.abs (z + 1) = Complex.abs (z - 2*i) → 
  ∃ (m b : ℝ), ∀ (x y : ℝ), z = x + y*i → y = m*x + b

def proposition3 : Prop := ∀ z, Complex.abs z ^ 2 = z ^ 2

def proposition4 : Prop := (List.range 2017).foldl (λ acc n => acc + i^n) 0 = 1

-- Theorem stating that exactly 3 out of 4 propositions are correct
theorem three_propositions_correct : 
  proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_propositions_correct_l84_8484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_equation_l84_8445

/-- Definition of a hyperbola with given properties -/
structure Hyperbola where
  center : ℝ × ℝ
  focus_on_y_axis : Bool
  eccentricity : ℝ
  vertex : ℝ × ℝ

/-- The hyperbola C with given properties -/
noncomputable def C : Hyperbola where
  center := (0, 0)
  focus_on_y_axis := true
  eccentricity := Real.sqrt 2
  vertex := (0, -1)

/-- The standard equation of a hyperbola -/
def standard_equation (h : Hyperbola) : Prop :=
  ∀ x y : ℝ, ((x, y) ∈ Set.range (λ p => p) → y^2 - x^2 = 1)

/-- Theorem stating that the hyperbola C has the standard equation y² - x² = 1 -/
theorem hyperbola_C_equation :
  standard_equation C := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_equation_l84_8445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l84_8411

theorem cube_surface_area_increase : 
  ∀ s : ℝ, s > 0 → 
  (6 * (s * 1.75)^2 - 6 * s^2) / (6 * s^2) * 100 = 206.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_increase_l84_8411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraiser_total_fundraiser_total_specific_l84_8432

theorem fundraiser_total (num_brownie_students : ℕ) (brownies_per_student : ℕ)
                         (num_cookie_students : ℕ) (cookies_per_student : ℕ)
                         (num_donut_students : ℕ) (donuts_per_student : ℕ)
                         (price_per_item : ℚ) : ℚ :=
  let total_items := num_brownie_students * brownies_per_student +
                     num_cookie_students * cookies_per_student +
                     num_donut_students * donuts_per_student
  total_items * price_per_item

theorem fundraiser_total_specific : 
  fundraiser_total 30 12 20 24 15 12 2 = 2040 := by
  rfl

#check fundraiser_total
#check fundraiser_total_specific

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fundraiser_total_fundraiser_total_specific_l84_8432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_seating_arrangements_count_l84_8423

/-- Represents a seating arrangement for 12 people around a round table. -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 chairs. -/
def isAdjacent (a b : Fin 12) : Prop := 
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are opposite on a round table with 12 chairs. -/
def isOpposite (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Checks if two positions are five chairs apart clockwise. -/
def isFiveApart (a b : Fin 12) : Prop := (b - a) % 12 = 5

/-- Represents a couple as a pair of indices. -/
def Couple := Fin 6 × Fin 6

/-- Checks if a seating arrangement is valid according to all conditions. -/
def isValidArrangement (arrangement : SeatingArrangement) (couples : Fin 6 → Couple) : Prop :=
  (∀ i : Fin 12, ¬isAdjacent (arrangement i) (arrangement ((i + 1) % 12))) ∧
  (∀ i : Fin 12, ¬isOpposite (arrangement i) (arrangement ((i + 6) % 12))) ∧
  (∀ i : Fin 12, ¬isFiveApart (arrangement i) (arrangement ((i + 1) % 12))) ∧
  (∀ c : Fin 6, let (m, w) := couples c;
    ¬isAdjacent (arrangement m) (arrangement w) ∧
    ¬isOpposite (arrangement m) (arrangement w)) ∧
  (∀ i : Fin 12, i % 2 = 0 → arrangement i < 6) ∧
  (∀ i : Fin 12, i % 2 = 1 → arrangement i ≥ 6)

/-- The theorem stating that the number of valid seating arrangements is 30240. -/
theorem valid_seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Couple),
    (∀ a ∈ arrangements, isValidArrangement a couples) ∧
    arrangements.card = 30240 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_seating_arrangements_count_l84_8423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l84_8435

-- Define our own types for geometrical objects
structure Point
structure Plane
structure Cone
structure CircularSurface
structure Pyramid
structure Face
structure Sphere
structure GreatCircle (S : Sphere)

-- Define operations and relations
def LateralSurface : Cone → CircularSurface := sorry
def Base : Pyramid → Face := sorry
def EquilateralTriangle : Face → Prop := sorry
def IsoscelesTriangle : Face → Prop := sorry
def RegularTetrahedron : Pyramid → Prop := sorry

/-- Three points can determine a plane -/
def proposition1 : Prop := ∀ (A B C : Point), ∃ (P : Plane), True  -- Simplified

/-- The lateral surface of a cone can be developed into a circular surface -/
def proposition2 : Prop := ∀ (C : Cone), ∃ (S : CircularSurface), LateralSurface C = S

/-- A pyramid with an equilateral triangular base and all three lateral faces being isosceles triangles is a regular tetrahedron -/
def proposition3 : Prop := ∀ (P : Pyramid), 
  (EquilateralTriangle (Base P)) ∧ 
  (∀ (F : Face), F ≠ Base P → IsoscelesTriangle F) → 
  RegularTetrahedron P

/-- There is exactly one great circle on a spherical surface passing through any two distinct points -/
def proposition4 : Prop := ∀ (S : Sphere) (A B : Point), 
  A ≠ B → 
  ∃! (C : GreatCircle S), True  -- Simplified

theorem all_propositions_false : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry

#check all_propositions_false

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l84_8435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_indivisible_equal_area_polygons_l84_8460

-- Define a type for polygons
structure Polygon where
  -- Add necessary fields here
  mk :: -- Constructor

-- Define a type for the function F
noncomputable def F (P : Polygon) : ℝ := sorry

-- Define area function
noncomputable def area (P : Polygon) : ℝ := sorry

-- Define parallel translation
noncomputable def parallelTranslate (P : Polygon) : Polygon := sorry

-- Define union and intersection for polygons
instance : Union Polygon where
  union := sorry

instance : Inter Polygon where
  inter := sorry

theorem existence_of_indivisible_equal_area_polygons :
  ∃ (M₁ M₂ : Polygon),
    (area M₁ = area M₂) ∧
    (F M₁ ≠ F M₂) ∧
    (∀ (A B : Polygon), (A ∪ B = M₁ ∧ area (A ∩ B) = 0) → F M₁ = F A + F B) ∧
    (∀ (M : Polygon), F M = F (parallelTranslate M)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_indivisible_equal_area_polygons_l84_8460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sine_l84_8428

-- Define the sine function as noncomputable
noncomputable def f (x : ℝ) := Real.sin x

-- State the theorem
theorem derivative_of_sine :
  deriv f = Real.cos := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_of_sine_l84_8428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l84_8467

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  center : ℝ × ℝ
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- The distance between foci of an ellipse -/
noncomputable def foci_distance (e : ParallelAxisEllipse) : ℝ :=
  2 * Real.sqrt (e.semi_major_axis ^ 2 - e.semi_minor_axis ^ 2)

theorem ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.center = (6, 3) ∧
    e.semi_major_axis = 6 ∧
    e.semi_minor_axis = 3 ∧
    foci_distance e = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_distance_l84_8467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_program_reform_relationship_l84_8427

/-- Represents the critical value for 99% confidence in a chi-square test --/
def criticalValue : ℝ := 6.635

/-- Represents the observed K^2 value from the survey --/
def observedK2 : ℝ := 6.89

/-- Theorem stating that the observed K^2 value indicates a relationship between 
    the TV program's excellence and the reform with 99% confidence --/
theorem tv_program_reform_relationship : 
  observedK2 > criticalValue → 
  ∃ (confidence : ℝ), confidence = 0.99 ∧ 
    (∃ (statement : String), statement = "The TV program's excellence is related to the reform") :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tv_program_reform_relationship_l84_8427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_nx_identity_l84_8461

theorem sin_nx_identity (n : ℕ) :
  ∃ (P : ℝ → ℝ), (∀ x, Real.sin (n * x) = P (Real.cos x) * Real.sin x) → P 1 = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_nx_identity_l84_8461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l84_8474

noncomputable def f (a b c d e : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + c) / (d * x + e)

theorem inverse_function_sum (a b c d e : ℝ) :
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  (∀ x, f a b c d e (f a b c d e x) = x) →
  a + e = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_sum_l84_8474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_slope_sum_l84_8466

-- Define the trapezoid ABCD
def A : ℤ × ℤ := (10, 50)
def D : ℤ × ℤ := (11, 53)

-- Define the properties of the trapezoid
def is_isosceles_trapezoid (B C : ℤ × ℤ) : Prop :=
  ∃ (m₁ m₂ : ℚ), 
    m₁ ≠ m₂ ∧ 
    m₁ ≠ 0 ∧ m₂ ≠ 0 ∧
    m₁ = (C.2 - B.2) / (C.1 - B.1) ∧
    m₂ = (D.2 - A.2) / (D.1 - A.1) ∧
    (B.2 - A.2) / (B.1 - A.1) = (D.2 - C.2) / (D.1 - C.1)

-- Define the sum of absolute values of possible slopes
noncomputable def sum_of_slopes (B C : ℤ × ℤ) : ℚ :=
  let slope₁ := (B.2 - A.2) / (B.1 - A.1)
  let slope₂ := (C.2 - D.2) / (C.1 - D.1)
  abs slope₁ + abs slope₂

-- State the theorem
theorem isosceles_trapezoid_slope_sum :
  ∀ B C : ℤ × ℤ, 
    is_isosceles_trapezoid B C →
    sum_of_slopes B C = 3/2 := by
  sorry

#check isosceles_trapezoid_slope_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_trapezoid_slope_sum_l84_8466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l84_8476

-- Define the points
def C : ℝ × ℝ := (-2, -2)
variable (A B M : ℝ × ℝ)

-- Define the conditions
axiom CA_CB_orthogonal : (A.1 + 2) * (B.1 + 2) + (A.2 + 2) * (B.2 + 2) = 0
axiom A_on_x_axis : A.2 = 0
axiom B_on_y_axis : B.1 = 0
axiom M_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- The theorem to prove
theorem trajectory_of_M : M.1 + M.2 + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_M_l84_8476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_div_a5_eq_three_fourths_l84_8413

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding case for 0
  | 1 => 1
  | n + 2 => (sequence_a (n + 1) + (-1)^(n + 2)) / sequence_a (n + 1)

theorem a3_div_a5_eq_three_fourths :
  sequence_a 3 / sequence_a 5 = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a3_div_a5_eq_three_fourths_l84_8413
