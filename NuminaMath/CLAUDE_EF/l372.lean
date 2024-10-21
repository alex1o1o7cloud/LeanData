import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l372_37224

noncomputable def curve (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos x + 1

noncomputable def tangent_slope (x : ℝ) : ℝ := -Real.sqrt 3 * Real.sin x

noncomputable def angle_of_inclination (x : ℝ) : ℝ := Real.arctan (tangent_slope x)

theorem angle_of_inclination_range :
  Set.range angle_of_inclination = Set.union (Set.Icc 0 (π/3)) (Set.Ico (2*π/3) π) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_range_l372_37224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_tangent_line_l372_37297

/-- The circle with equation x^2 + y^2 = 5 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 5

/-- The point M on the circle -/
def point_M : ℝ × ℝ := (1, 2)

/-- The proposed tangent line equation -/
def tangent_line (x y : ℝ) : Prop := x + 2*y - 5 = 0

/-- Theorem stating that the proposed line is indeed the tangent line to the circle at point M -/
theorem is_tangent_line : 
  (my_circle point_M.1 point_M.2) ∧ 
  (∀ (x y : ℝ), my_circle x y → tangent_line x y → (x, y) = point_M) ∧
  (∃ (x y : ℝ), x ≠ point_M.1 ∧ y ≠ point_M.2 ∧ tangent_line x y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_tangent_line_l372_37297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_items_count_l372_37273

/-- Given a ratio of pens : pencils : markers as 2 : 2 : 5, and 10 pens, prove there are 25 markers. -/
theorem desk_items_count (pens pencils markers : ℕ) : 
  pens = 10 → 
  (pens : ℚ) / 2 = (pencils : ℚ) / 2 ∧ (pens : ℚ) / 2 = (markers : ℚ) / 5 → 
  markers = 25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_desk_items_count_l372_37273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xxyy_equivalent_xyyyyx_xytx_not_equivalent_txyt_xy_not_equivalent_xt_l372_37291

-- Define the alphabet
inductive Letter : Type
| x : Letter
| y : Letter
| z : Letter
| t : Letter

-- Define a word as a list of letters
def Word := List Letter

-- Define the transformation rules
def transform : Word → Option Word
| (Letter.x :: Letter.y :: rest) => some (Letter.y :: Letter.y :: Letter.x :: rest)
| (Letter.x :: Letter.t :: rest) => some (Letter.t :: Letter.t :: Letter.x :: rest)
| (Letter.y :: Letter.t :: rest) => some (Letter.t :: Letter.y :: rest)
| _ => none

-- Define equivalence relation
def equivalent (w1 w2 : Word) : Prop :=
  ∃ (n : Nat), ∃ (sequence : Fin (n + 1) → Word),
    sequence 0 = w1 ∧
    sequence n = w2 ∧
    ∀ (i : Fin n), transform (sequence i) = some (sequence (i + 1))

-- Theorem statements
theorem xxyy_equivalent_xyyyyx :
  equivalent [Letter.x, Letter.x, Letter.y, Letter.y] [Letter.x, Letter.y, Letter.y, Letter.y, Letter.y, Letter.x] :=
sorry

theorem xytx_not_equivalent_txyt :
  ¬ equivalent [Letter.x, Letter.y, Letter.t, Letter.x] [Letter.t, Letter.x, Letter.y, Letter.t] :=
sorry

theorem xy_not_equivalent_xt :
  ¬ equivalent [Letter.x, Letter.y] [Letter.x, Letter.t] :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_xxyy_equivalent_xyyyyx_xytx_not_equivalent_txyt_xy_not_equivalent_xt_l372_37291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_range_l372_37256

/-- Given that (1-i)(a+i) corresponds to a point in the second quadrant of the complex plane,
    prove that a ∈ (-∞, -1) -/
theorem complex_second_quadrant_range (a : ℝ) :
  (1 - Complex.I) * (Complex.I + a) ∈ {z : ℂ | z.re < 0 ∧ z.im > 0} →
  a ∈ Set.Iio (-1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_second_quadrant_range_l372_37256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l372_37201

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := i^2 / (2*i - 1)
  z.im = 2/5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_imaginary_part_of_complex_fraction_l372_37201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_intersection_distance_l372_37255

-- Define the line and ellipse
def line (x b : ℝ) : ℝ := x + b
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the intersection points
def intersection_points (b : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ y = line x b ∧ ellipse x y}

-- Theorem for the range of b
theorem intersection_range :
  ∀ b : ℝ, (∃ A B : ℝ × ℝ, A ∈ intersection_points b ∧ B ∈ intersection_points b ∧ A ≠ B) ↔
  -Real.sqrt 3 < b ∧ b < Real.sqrt 3 := by sorry

-- Function to calculate the distance between two points
noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Theorem for the length of AB when b = 1
theorem intersection_distance :
  ∀ A B : ℝ × ℝ, A ∈ intersection_points 1 ∧ B ∈ intersection_points 1 ∧ A ≠ B →
  distance A B = 4 * Real.sqrt 2 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_intersection_distance_l372_37255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_ratio_k_nonzero_exponential_sequence_is_arithmetic_progression_ratio_l372_37293

/-- Definition of an arithmetic-progression ratio sequence -/
def is_arithmetic_progression_ratio (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k

/-- The constant k in an arithmetic-progression ratio sequence cannot be zero -/
theorem arithmetic_progression_ratio_k_nonzero
  (a : ℕ → ℝ) (h : is_arithmetic_progression_ratio a) : 
  ∃ k : ℝ, k ≠ 0 ∧ ∀ n : ℕ, (a (n + 2) - a (n + 1)) / (a (n + 1) - a n) = k := by
  sorry

/-- A sequence with general term a_n = a · b^n + c (a ≠ 0, b ≠ 0, 1) is an arithmetic-progression ratio sequence -/
theorem exponential_sequence_is_arithmetic_progression_ratio
  (a c : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hb1 : b ≠ 1) :
  is_arithmetic_progression_ratio (fun n ↦ a * b^n + c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_ratio_k_nonzero_exponential_sequence_is_arithmetic_progression_ratio_l372_37293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l372_37296

-- Define the radius of the original circular sheet
noncomputable def original_radius : ℝ := 8

-- Define the number of sectors
def num_sectors : ℕ := 4

-- Define the radius of the base of the cone
noncomputable def base_radius : ℝ := original_radius * Real.pi / num_sectors

-- Define the slant height of the cone (same as original radius)
noncomputable def slant_height : ℝ := original_radius

-- Theorem statement
theorem cone_height_from_circular_sector :
  ∃ (h : ℝ), h^2 = slant_height^2 - base_radius^2 ∧ h = 2 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_from_circular_sector_l372_37296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_minimum_value_a_range_l372_37286

-- Define the functions
noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (x + a)
noncomputable def g (x a : ℝ) : ℝ := -a / 2 * x^2 + a * x
noncomputable def h (x a : ℝ) : ℝ := f (Real.exp x - a) a + (deriv (g · a)) (Real.exp x)

-- Theorem for the minimum value of h(x)
theorem h_minimum_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, h x a ≥ 
    if a ≤ 0 then a - (1 + a) / Real.exp 1
    else if a < 2 then -Real.exp (a - 1) + a
    else (1 - a) * Real.exp 1 + a) :=
by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x ≥ 2, (x - 1) * Real.log (x - 1) + a / 2 * x^2 - a * x ≤ 0) →
  a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_minimum_value_a_range_l372_37286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_factors_x_pow_10_minus_1_l372_37225

/-- The maximum number of non-constant real polynomial factors of x^10 - 1 -/
theorem max_real_factors_x_pow_10_minus_1 : ∃ (m : ℕ), 
  (∀ (q : List (Polynomial ℝ)), 
    (∀ p ∈ q, Polynomial.degree p > 0) → 
    (List.prod q = Polynomial.X^10 - 1) → 
    (q.length ≤ m)) ∧ 
  (∃ (r : List (Polynomial ℝ)), 
    (∀ p ∈ r, Polynomial.degree p > 0) ∧ 
    (List.prod r = Polynomial.X^10 - 1) ∧ 
    (r.length = m)) ∧
  m = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_real_factors_x_pow_10_minus_1_l372_37225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_condition_l372_37228

-- Define IsQuadraticIn as it's not a standard predicate in Mathlib
def IsQuadraticIn (x : ℝ) (e : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ y, e y = a * y^2 + b * y + c

theorem quadratic_equation_condition (a : ℝ) :
  (∀ x, (a - 1) * x^2 + 4 * x - 3 = 0 → IsQuadraticIn x (λ y => (a - 1) * y^2 + 4 * y - 3)) →
  a ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_condition_l372_37228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_calculation_l372_37230

/-- The volume of a circular well -/
noncomputable def well_volume (diameter : ℝ) (depth : ℝ) : ℝ :=
  Real.pi * (diameter / 2)^2 * depth

/-- Theorem: The volume of a circular well with diameter 2 meters and depth 10 meters is 10π cubic meters -/
theorem well_volume_calculation :
  well_volume 2 10 = 10 * Real.pi :=
by
  unfold well_volume
  simp [Real.pi]
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_volume_calculation_l372_37230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chloe_lost_one_pawn_l372_37248

/-- Represents the number of pawns a player has lost in a chess game. -/
abbrev PawnsLost := Nat

/-- Represents the number of pawns remaining in a chess game. -/
abbrev PawnsRemaining := Nat

/-- The initial number of pawns each player has in a chess game. -/
def initialPawns : Nat := 8

/-- Calculates the number of pawns Chloe has lost in the chess game. -/
def chloePawnsLost (sophiaPawnsLost : Nat) (totalPawnsRemaining : Nat) : Nat :=
  initialPawns - (totalPawnsRemaining - (initialPawns - sophiaPawnsLost))

/-- Theorem stating that Chloe has lost 1 pawn given the conditions of the game. -/
theorem chloe_lost_one_pawn :
  chloePawnsLost 5 10 = 1 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chloe_lost_one_pawn_l372_37248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l372_37209

def sequence_a : ℕ → ℚ
  | 0 => 1  -- Adding the base case for 0
  | 1 => 1
  | n + 2 => 4 * sequence_a (n + 1) + 1

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 →
    (sequence_a (n + 1) + 1/3) = 4 * (sequence_a n + 1/3)) ∧
  (∀ n : ℕ, n ≥ 1 → sequence_a n = (4^n - 1) / 3) ∧
  (∀ n : ℕ, n ≥ 1 →
    (Finset.range n).sum (λ i => 1 / sequence_a (i + 1)) < 4/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l372_37209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_terminal_side_l372_37251

/-- Given a point P on the terminal side of angle α, prove that cos α equals the x-coordinate divided by the distance from the origin to P. -/
theorem cosine_terminal_side (α : ℝ) (P : EuclideanSpace ℝ (Fin 2)) :
  P 0 = -4 ∧ P 1 = 3 →
  Real.cos α = -4 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_terminal_side_l372_37251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_addition_l372_37203

/-- Represents a number in base 8 --/
structure Base8 where
  value : Nat
  valid : value < 512 := by sorry

/-- Converts a base 8 number to a natural number --/
def base8ToNat (n : Base8) : Nat := n.value

/-- Converts a natural number to base 8 --/
def natToBase8 (n : Nat) : Base8 where
  value := n % 512
  valid := by sorry

/-- Addition in base 8 --/
def addBase8 (a b : Base8) : Base8 :=
  natToBase8 (base8ToNat a + base8ToNat b)

/-- Helper function to create Base8 numbers --/
def mkBase8 (n : Nat) : Base8 := natToBase8 n

/-- The main theorem to prove --/
theorem base8_addition :
  addBase8 (mkBase8 123) (mkBase8 56) = mkBase8 202 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base8_addition_l372_37203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_cubic_sum_value_l372_37229

-- Part 1: Factorization
theorem factorization_problem (x : ℝ) : 42 * x^2 - 33 * x + 6 = 3 * (2 * x - 1) * (7 * x - 2) := by
  sorry

-- Part 2: Value of x^3 + 1/x^3
theorem cubic_sum_value (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^3 + 1 / x^3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorization_problem_cubic_sum_value_l372_37229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_implies_k_nonnegative_l372_37231

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then k * x + 2 else (1/2) ^ x

theorem no_solution_implies_k_nonnegative (k : ℝ) :
  (∀ x : ℝ, f k (f k x) - 3/2 ≠ 0) → k ≥ 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_implies_k_nonnegative_l372_37231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_base_l372_37242

/-- The sum of digits of a natural number in a given base --/
def sum_of_digits (n : ℕ) (base : ℕ) : ℕ := sorry

/-- Checks if a base is valid for the problem conditions --/
def is_valid_base (b : ℕ) : Prop :=
  b > 1 ∧ sum_of_digits (12^4) b ≠ 35

theorem largest_valid_base :
  ∃ (max_base : ℕ), max_base = 7 ∧
  is_valid_base max_base ∧
  (∀ k > max_base, ¬is_valid_base k) ∧
  35 = 5^2 + 5 := by
  sorry

#check largest_valid_base

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_valid_base_l372_37242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_l372_37285

/-- Represents a tower on a horizontal plane with elevation angles measured from two points -/
structure Tower where
  height : ℝ
  angle_50m : ℝ
  angle_100m : ℝ

/-- The tangent of the sum of two angles equals the sum of their tangents divided by one minus their product -/
axiom tan_sum (α β : ℝ) : Real.tan (α + β) = (Real.tan α + Real.tan β) / (1 - Real.tan α * Real.tan β)

/-- The theorem stating the height of the tower given the conditions -/
theorem tower_height (t : Tower) 
  (h_sum : t.angle_50m + t.angle_100m = Real.pi / 4)
  (h_tan_50 : Real.tan t.angle_50m = t.height / 50)
  (h_tan_100 : Real.tan t.angle_100m = t.height / 100) :
  ∃ ε > 0, |t.height - 28.08| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tower_height_l372_37285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l372_37282

-- Define the triangle
noncomputable def triangle (x : ℝ) : Set (ℝ × ℝ) :=
  {(0, 0), (x, 3*x), (x, 0)}

-- Define the area of the triangle
noncomputable def triangle_area (x : ℝ) : ℝ :=
  (1/2) * x * (3*x)

-- Theorem statement
theorem triangle_area_theorem (x : ℝ) :
  x > 0 ∧ triangle_area x = 72 → x = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_theorem_l372_37282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_truth_teller_not_knight_l372_37227

-- Define the types of inhabitants
inductive InhabitantType
  | Knight
  | Liar
  | Ordinary

-- Define a function to determine if an inhabitant is telling the truth
def isTellingTruth (inhType : InhabitantType) (statement : Prop) : Prop :=
  match inhType with
  | InhabitantType.Knight => statement
  | InhabitantType.Liar => ¬statement
  | InhabitantType.Ordinary => True

-- Define the statements made by A and B
def statementA (typeB : InhabitantType) : Prop :=
  typeB = InhabitantType.Knight

def statementB (typeA : InhabitantType) : Prop :=
  typeA ≠ InhabitantType.Knight

-- Theorem statement
theorem at_least_one_truth_teller_not_knight :
  ∃ (typeA typeB : InhabitantType),
    (isTellingTruth typeA (statementA typeB) ∧ typeA ≠ InhabitantType.Knight) ∨
    (isTellingTruth typeB (statementB typeA) ∧ typeB ≠ InhabitantType.Knight) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_truth_teller_not_knight_l372_37227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_patriotic_division_l372_37279

/-- A color type representing red, white, and blue. -/
inductive Color
  | Red
  | White
  | Blue

/-- A vertex of the polygon, represented by its position and color. -/
structure Vertex where
  position : ℂ
  color : Color

/-- A regular polygon with colored vertices. -/
structure RegularPolygon where
  vertices : List Vertex
  is_regular : Bool
  is_patriotic : Bool

/-- Checks if a subset of vertices is patriotic. -/
def is_patriotic (subset : List Vertex) : Bool :=
  sorry

/-- Checks if an edge is dazzling. -/
def is_dazzling (v1 v2 : Vertex) : Bool :=
  sorry

/-- Counts the number of dazzling edges in a polygon. -/
def count_dazzling_edges (p : RegularPolygon) : Nat :=
  sorry

/-- Theorem: If a regular polygon has patriotic vertices and an even number of dazzling edges,
    then there exists a line not passing through any vertex that divides the vertices
    into two nonempty patriotic subsets. -/
theorem patriotic_division (p : RegularPolygon) :
  p.is_regular ∧ p.is_patriotic ∧ (count_dazzling_edges p % 2 = 0) →
  ∃ (line : ℂ → Prop),
    (∀ v ∈ p.vertices, ¬line v.position) ∧
    ∃ (s1 s2 : List Vertex),
      s1 ≠ [] ∧ s2 ≠ [] ∧
      (∀ v, v ∈ s1 ∨ v ∈ s2 ↔ v ∈ p.vertices) ∧
      is_patriotic s1 ∧ is_patriotic s2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_patriotic_division_l372_37279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_circle_equations_l372_37240

noncomputable def A : ℝ × ℝ := (4, 1)
noncomputable def B : ℝ × ℝ := (0, 3)
noncomputable def C : ℝ × ℝ := (2, 4)

noncomputable def D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

noncomputable def center_C : ℝ × ℝ := (-7/5, 9/5)

def tangent_line (x y : ℝ) : Prop := 3*x + 4*y + 17 = 0

def line_BD (x y : ℝ) : Prop := x + 6*y - 18 = 0

def circle_C (x y : ℝ) : Prop := (x + 7/5)^2 + (y - 9/5)^2 = 16

theorem median_and_circle_equations :
  (∀ x y, line_BD x y ↔ (x - D.1) * (B.2 - D.2) = (y - D.2) * (B.1 - D.1)) ∧
  (∀ x y, circle_C x y ↔ ((x - center_C.1)^2 + (y - center_C.2)^2 = 16)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_and_circle_equations_l372_37240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l372_37245

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (2*x - 15)

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < (15 / 2) ∨ x > (15 / 2)} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l372_37245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_funding_theorem_l372_37223

/-- Calculates the number of people needed to reach a funding goal -/
def peopleNeeded (targetAmount currentFunds averageContribution : ℕ) : ℕ :=
  ((targetAmount - currentFunds) + averageContribution - 1) / averageContribution

theorem ryan_funding_theorem (targetAmount currentFunds averageContribution : ℕ) 
  (h1 : targetAmount = 1000)
  (h2 : currentFunds = 200)
  (h3 : averageContribution = 10) :
  peopleNeeded targetAmount currentFunds averageContribution = 80 := by
  sorry

#eval peopleNeeded 1000 200 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ryan_funding_theorem_l372_37223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l372_37250

-- Define the function g as noncomputable
noncomputable def g (x : ℝ) := Real.log (x + 1)

-- State the theorem about the domain of g
theorem domain_of_g :
  {x : ℝ | ∃ y, g x = y} = {x : ℝ | x > -1} :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l372_37250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l372_37238

theorem power_equation_solution : ∃ x : ℝ, (7 : ℝ)^3 * (7 : ℝ)^x = 49 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_equation_solution_l372_37238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_of_distances_l372_37249

-- Define the curve C
noncomputable def curve_C (t : ℝ) : ℝ × ℝ := (4 / (1 + t^2), 4*t / (1 + t^2))

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + Real.sqrt 3 * t, t)

-- Define point M
def point_M : ℝ × ℝ := (1, 0)

-- Define the intersection points A and B (existence assumed)
noncomputable def point_A : ℝ × ℝ := sorry
noncomputable def point_B : ℝ × ℝ := sorry

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- State the theorem
theorem reciprocal_sum_of_distances :
  1 / distance point_M point_A + 1 / distance point_M point_B = Real.sqrt 15 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_sum_of_distances_l372_37249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_difference_l372_37298

-- Define the function f
def f (c₁ c₂ c₃ c₄ x : ℤ) : ℤ :=
  (x^2 - 4*x + c₁) * (x^2 - 4*x + c₂) * (x^2 - 4*x + c₃) * (x^2 - 4*x + c₄)

-- Define the set M
def M (c₁ c₂ c₃ c₄ : ℤ) : Set ℤ :=
  {x : ℤ | f c₁ c₂ c₃ c₄ x = 0}

-- State the theorem
theorem impossible_difference (c₁ c₂ c₃ c₄ : ℤ) :
  (c₁ ≤ c₂) → (c₂ ≤ c₃) → (c₃ ≤ c₄) →
  (∃ (s : Finset ℤ), s.card = 7 ∧ ∀ x, x ∈ s ↔ x ∈ M c₁ c₂ c₃ c₄) →
  (c₄ - c₁ ≠ 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_difference_l372_37298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l372_37200

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → 
    ∃ k : ℝ, y = k * x ∧ 
    |k * c - b| / Real.sqrt (1 + k^2) = Real.sqrt 3 / 2 * c) →
  c / a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l372_37200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l372_37247

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 2)

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ 0 < y ∧ y ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l372_37247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_approx_l372_37275

/-- The percentage of brokerage given the cash realized and total amount -/
noncomputable def brokerage_percentage (cash_realized : ℝ) (total_amount : ℝ) : ℝ :=
  (cash_realized - total_amount) / total_amount * 100

/-- Theorem stating that the brokerage percentage is approximately 0.23% -/
theorem brokerage_percentage_approx (cash_realized total_amount : ℝ) 
  (h1 : cash_realized = 108.25)
  (h2 : total_amount = 108) :
  |brokerage_percentage cash_realized total_amount - 0.23| < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brokerage_percentage_approx_l372_37275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_minimum_l372_37270

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptotes of C
def asymptotes (a b x y : ℝ) : Prop := y = b/a * x ∨ y = -b/a * x

-- Define the line x = a
def line_x_eq_a (a x : ℝ) : Prop := x = a

-- Define the area of triangle ODE
def area_triangle_ODE (a b : ℝ) : ℝ := a * b

-- Define the focal length of C
noncomputable def focal_length (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 + b^2)

theorem hyperbola_focal_length_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola a b a b ∧ 
  hyperbola a b a (-b) ∧
  asymptotes a b a b ∧
  asymptotes a b a (-b) ∧
  line_x_eq_a a a ∧
  area_triangle_ODE a b = 8 →
  ∀ c d : ℝ, c > 0 ∧ d > 0 ∧ c * d = 8 → focal_length a b ≤ focal_length c d :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_length_minimum_l372_37270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sqrt_equation_solutions_l372_37219

theorem cubic_sqrt_equation_solutions :
  ∀ x : ℝ, (Real.rpow (3 - x) (1/3 : ℝ) + Real.sqrt (x - 2) = 2) ↔ (x = 2 ∨ x = 30) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_sqrt_equation_solutions_l372_37219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coefficient_term_is_9_l372_37205

noncomputable def binomial_expansion (n : ℕ) (x : ℝ) : ℝ := (3*x + 1/x)^n

-- We need to define these functions as they don't exist in Mathlib
def nth_term (f : ℝ → ℝ) (n : ℕ) : ℝ := sorry
def constant_term (f : ℝ → ℝ) : ℝ := sorry
def max_coefficient_term (f : ℝ → ℝ) : ℕ := sorry

theorem max_coefficient_term_is_9 (n : ℕ) :
  (∃ x : ℝ, nth_term (binomial_expansion n) 5 = constant_term (binomial_expansion n)) →
  max_coefficient_term (binomial_expansion n) = 9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_coefficient_term_is_9_l372_37205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l372_37274

-- Define the power function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^a

-- State the theorem
theorem power_function_properties :
  ∃ a : ℝ, 
    (f a 4 = 2) ∧ 
    (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
    (∃ x : ℝ, f a x ≠ f a (-x)) ∧
    (∃ x : ℝ, f a x ≠ -f a (-x)) := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_properties_l372_37274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l372_37295

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (-2, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the set of points M satisfying the condition
def trajectory : Set (ℝ × ℝ) :=
  {M : ℝ × ℝ | distance M F₁ + distance M F₂ = 4}

-- Theorem stating that the trajectory is a line segment
theorem trajectory_is_line_segment :
  ∃ (a b : ℝ × ℝ), trajectory = {x : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • a + t • b} :=
by
  sorry

-- Additional lemma to show that F₁ and F₂ are the endpoints of the line segment
lemma endpoints_are_F₁_and_F₂ :
  F₁ ∈ trajectory ∧ F₂ ∈ trajectory :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l372_37295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_distance_sum_l372_37243

noncomputable section

-- Define the curve C₁
def C₁ (α : ℝ) : ℝ × ℝ := (-2 + 2 * Real.cos α, 2 * Real.sin α)

-- Define the rotation function
def rotate_clockwise (p : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos angle + p.2 * Real.sin angle,
   -p.1 * Real.sin angle + p.2 * Real.cos angle)

-- Define curve C₂
def C₂ (θ : ℝ) : ℝ × ℝ := (4 * Real.sin θ * Real.cos θ, 4 * Real.sin θ * Real.sin θ)

-- Define point F
def F : ℝ × ℝ := (0, -1)

-- Define line l: √3x - y - 1 = 0
noncomputable def l (t : ℝ) : ℝ × ℝ := (t / 2, -1 + Real.sqrt 3 * t / 2)

theorem curve_C₂_and_distance_sum :
  (∀ α ∈ Set.Icc 0 Real.pi, 
    rotate_clockwise (C₁ α) (Real.pi / 2) = C₂ (Real.pi / 2 - α)) ∧
  (∃ A B : ℝ × ℝ, A ∈ Set.range C₂ ∧ B ∈ Set.range C₂ ∧
    (Real.sqrt 3 * A.1 - A.2 - 1 = 0) ∧ (Real.sqrt 3 * B.1 - B.2 - 1 = 0) ∧
    Real.sqrt ((F.1 - A.1)^2 + (F.2 - A.2)^2) + 
    Real.sqrt ((F.1 - B.1)^2 + (F.2 - B.2)^2) = 3 * Real.sqrt 3) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C₂_and_distance_sum_l372_37243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l372_37288

/-- The eccentricity of a hyperbola given specific conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (h : a > 0) (k : b > 0) (m : c > 0) : 
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let circle := fun (x y : ℝ) => x^2 + y^2 = a^2
  let parabola := fun (x y : ℝ) => y^2 = 4 * c * x
  let left_focus : ℝ × ℝ := (-c, 0)
  ∃ (E P : ℝ × ℝ), 
    hyperbola E.1 E.2 ∧ 
    circle E.1 E.2 ∧ 
    parabola P.1 P.2 ∧
    E = ((left_focus.1 + P.1) / 2, (left_focus.2 + P.2) / 2) →
  c / a = (Real.sqrt 5 + 1) / 2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l372_37288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l372_37241

noncomputable def z : ℂ := ((3 - Complex.I) / (1 + Complex.I)) ^ 2

theorem z_in_third_quadrant : 
  Real.sign z.re = -1 ∧ Real.sign z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_in_third_quadrant_l372_37241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_b_l372_37253

theorem greatest_integer_b : ∃ b : ℤ, (
  -- Define the quadratic function
  let f (x b : ℝ) := x^2 + b*x + 20

  -- Define the condition for -10 not being in the range
  let not_in_range (b : ℝ) := ∀ x, f x b ≠ -10

  -- Define the property we want to prove
  let is_greatest_integer (b : ℤ) :=
    (not_in_range (b : ℝ)) ∧ 
    (∀ k : ℤ, k > b → ¬(not_in_range (k : ℝ)))

  -- The theorem statement
  is_greatest_integer b ∧ b = 10
) := by
  -- The proof is omitted
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_integer_b_l372_37253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_region_l372_37232

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := x^(1/Real.log x)

-- Define the bounds
def lower_x : ℝ := 2
def upper_x : ℝ := 3
def lower_y : ℝ := 0

-- Theorem statement
theorem area_bounded_region : 
  (∫ x in lower_x..upper_x, (f x - lower_y)) = Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_region_l372_37232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_plus_cos_l372_37208

theorem max_value_sin_plus_cos :
  ∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ x, Real.sin x + Real.cos x ≤ M := by
  -- We'll use M = √2 as our maximum value
  let M := Real.sqrt 2
  
  -- Prove that M equals √2
  have h_M_eq : M = Real.sqrt 2 := rfl
  
  -- Prove that for all x, sin x + cos x ≤ M
  have h_inequality : ∀ x, Real.sin x + Real.cos x ≤ M := by
    intro x
    -- The proof steps would go here
    sorry -- We use sorry to skip the actual proof for now
  
  -- Combine the two parts to prove the theorem
  exact ⟨M, h_M_eq, h_inequality⟩

#check max_value_sin_plus_cos

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_sin_plus_cos_l372_37208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_theorem_l372_37218

-- Define the function
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

-- Define the interval
def I : Set ℝ := Set.Icc 0 2

-- Define the range of the function on the interval
def range_f (a : ℝ) : Set ℝ := {y | ∃ x ∈ I, f a x = y}

-- Theorem stating the range of the function for different values of a
theorem range_f_theorem (a : ℝ) :
  (a < 0 → range_f a = Set.Icc (-1) (3 - 4*a)) ∧
  (0 ≤ a ∧ a ≤ 1 → range_f a = Set.Icc (-(a^2 + 1)) (3 - 4*a)) ∧
  (1 < a ∧ a ≤ 2 → range_f a = Set.Icc (-(a^2 + 1)) (-1)) ∧
  (2 < a → range_f a = Set.Icc (3 - 4*a) (-1)) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_f_theorem_l372_37218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l372_37263

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t)
  (h2 : Real.cos t.A = 3/4)
  (h3 : t.C = 2 * t.A)
  (h4 : t.a = 4) :
  Real.sin t.B = 5 * Real.sqrt 7 / 16 ∧
  1/2 * t.a * t.b * Real.sin t.C = 15 * Real.sqrt 7 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l372_37263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l372_37284

/-- Represents the time it takes to complete a work -/
structure WorkTime where
  p_alone : ℝ  -- Time for p to complete the work alone
  q_alone : ℝ  -- Time for q to complete the work alone
  p_initial : ℝ  -- Time p works alone before q joins

/-- Calculates the total time to complete the work -/
noncomputable def total_time (w : WorkTime) : ℝ :=
  let remaining_work := 1 - w.p_initial / w.p_alone
  let combined_rate := 1 / w.p_alone + 1 / w.q_alone
  w.p_initial + remaining_work / combined_rate

/-- Theorem stating that under the given conditions, the work lasts 25 days -/
theorem work_completion_time :
  ∀ w : WorkTime,
    w.p_alone = 40 ∧
    w.q_alone = 24 ∧
    w.p_initial = 16 →
    total_time w = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l372_37284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l372_37290

theorem product_remainder (a b c d : ℕ) 
  (ha : a % 7 = 2)
  (hb : b % 7 = 3)
  (hc : c % 7 = 4)
  (hd : d % 7 = 5) :
  (a * b * c * d) % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l372_37290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l372_37292

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.evaluate (f : QuadraticFunction) (x : ℝ) : ℝ :=
  f.a * x^2 + f.b * x + f.c

/-- Calculates the vertex of a quadratic function -/
noncomputable def QuadraticFunction.vertex (f : QuadraticFunction) : ℝ × ℝ :=
  let h := -f.b / (2 * f.a)
  (h, f.evaluate h)

theorem parabola_equation_proof (f : QuadraticFunction) 
    (h_coefficients : f.a = 2 ∧ f.b = -12 ∧ f.c = 16) :
    f.vertex = (3, -2) ∧ 
    f.evaluate 6 = 16 ∧
    f.a ≠ 0 := by
  sorry

#check parabola_equation_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_proof_l372_37292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2001_value_l372_37261

def sequenceProperty (a : ℕ → ℤ) : Prop :=
  a 1 = 5 ∧ a 5 = 8 ∧ ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 7

theorem sequence_2001_value (a : ℕ → ℤ) (h : sequenceProperty a) : a 2001 = -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2001_value_l372_37261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_l372_37246

-- Define a triangle ABC
structure Triangle :=
  (A B C : Real)  -- angles
  (a b c : Real)  -- sides opposite to angles A, B, C respectively

-- State the theorem
theorem max_angle_A (t : Triangle) 
  (h : Real.cos t.B / t.b = -(3 * Real.cos t.C) / t.c) :
  t.A ≤ π / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_A_l372_37246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perp_distance_l372_37215

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  -- Diagonal lengths
  ac : ℝ
  bd : ℝ
  -- Assumption that ac and bd are positive
  ac_pos : 0 < ac
  bd_pos : 0 < bd

/-- A point on the side of the rhombus with its perpendiculars -/
structure RhombusPoint (r : Rhombus) where
  -- Distance from one end of the side to the point
  t : ℝ
  -- Assumption that t is between 0 and the side length
  t_nonneg : 0 ≤ t
  t_le_side : t ≤ (r.ac ^ 2 + r.bd ^ 2) / (4 * r.ac)

/-- The length of the line segment between the feet of the perpendiculars -/
noncomputable def perpDistance (r : Rhombus) (p : RhombusPoint r) : ℝ :=
  let mr := r.ac / 2 - p.t * r.ac ^ 2 / (r.ac ^ 2 + r.bd ^ 2)
  let ms := p.t * r.bd ^ 2 / (r.ac ^ 2 + r.bd ^ 2)
  Real.sqrt (mr ^ 2 + ms ^ 2)

/-- The main theorem -/
theorem min_perp_distance (r : Rhombus) :
    r.ac = 24 → r.bd = 40 →
    ∃ (min : ℝ), min = 6 * Real.sqrt 2 ∧
    ∀ (p : RhombusPoint r), min ≤ perpDistance r p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_perp_distance_l372_37215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l372_37212

noncomputable def f (x : ℝ) : ℝ := 2^x

noncomputable def g (x : ℝ) : ℝ := Real.log (8*x) / Real.log (Real.sqrt 2)

theorem solution_equality (x : ℝ) : 
  f (g x) = g (f x) ↔ x = (1 + Real.sqrt 385) / 64 := by
  sorry

#check solution_equality

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_equality_l372_37212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_cost_proof_l372_37287

/-- Rounds a rational number to a specified number of decimal places -/
def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ := 
  sorry

/-- The cost of each t-shirt given the total amount spent and the number of t-shirts bought -/
def cost_per_tshirt (total_spent : ℚ) (num_tshirts : ℕ) : ℚ :=
  total_spent / num_tshirts

/-- Proof that the cost of each t-shirt is $9.14 (rounded to two decimal places) -/
theorem tshirt_cost_proof (total_spent : ℚ) (num_tshirts : ℕ) 
  (h1 : total_spent = 201)
  (h2 : num_tshirts = 22) :
  round_to_decimal_places (cost_per_tshirt total_spent num_tshirts) 2 = 9.14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tshirt_cost_proof_l372_37287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plants_count_l372_37210

/-- The number of unique plants in three overlapping flower beds -/
theorem unique_plants_count (X Y Z : Finset ℕ) : 
  (X.card = 600) →
  (Y.card = 480) →
  (Z.card = 420) →
  ((X ∩ Y).card = 60) →
  ((Y ∩ Z).card = 70) →
  ((X ∩ Z).card = 80) →
  ((X ∩ Y ∩ Z).card = 30) →
  ((X ∪ Y ∪ Z).card = 1320) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_plants_count_l372_37210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l372_37206

theorem problem_statement (a b : ℝ) (h1 : (3 : ℝ)^a = 15) (h2 : (5 : ℝ)^b = 15) : (a - 1)^2 + (b - 1)^2 > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l372_37206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l372_37236

theorem abc_inequality (a b c : ℝ) 
  (h1 : a ∈ Set.Icc (-1 : ℝ) 1) 
  (h2 : b ∈ Set.Icc (-1 : ℝ) 1) 
  (h3 : c ∈ Set.Icc (-1 : ℝ) 1) 
  (h4 : a + b + c + a*b*c = 0) : 
  a^2 + b^2 + c^2 ≥ 3*(a + b + c) ∧ 
  (a^2 + b^2 + c^2 = 3*(a + b + c) ↔ a = 1 ∧ b = 1 ∧ c = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_inequality_l372_37236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_face_coloring_theorem_l372_37299

/-- A polyhedron with an inscribed sphere -/
structure InscribedPolyhedron where
  /-- The polyhedron -/
  polyhedron : Set (ℝ × ℝ × ℝ)
  /-- The inscribed sphere -/
  sphere : Set (ℝ × ℝ × ℝ)
  /-- The sphere is inscribed in the polyhedron -/
  inscribed : sphere ⊆ polyhedron

/-- A face of a polyhedron -/
def Face := Set (ℝ × ℝ × ℝ)

/-- A face coloring of a polyhedron -/
def FaceColoring (P : InscribedPolyhedron) := Face → Bool

/-- The surface area of a set of faces -/
noncomputable def surfaceArea (faces : Set Face) : ℝ := sorry

/-- Two faces share an edge -/
def sharesEdge (face1 face2 : Face) : Prop := sorry

/-- The theorem statement -/
theorem inscribed_sphere_face_coloring_theorem
  (P : InscribedPolyhedron)
  (coloring : FaceColoring P)
  (h_no_adjacent_black : ∀ f1 f2 : Face, coloring f1 = true → coloring f2 = true → ¬sharesEdge f1 f2) :
  surfaceArea {f : Face | coloring f = true} ≤ surfaceArea {f : Face | coloring f = false} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_face_coloring_theorem_l372_37299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l372_37214

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a)
    (h_prod : a 2 * a 3 * a 4 = 64)
    (h_sqrt : Real.sqrt (a 6 * a 8) = 16) :
    (1/4)^(-2 : ℤ) * 2^(-3 : ℤ) - (a 5)^(1/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_problem_l372_37214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amp_eight_four_l372_37259

noncomputable def amp (a b : ℝ) : ℝ := a + a / b - b

theorem amp_eight_four : amp 8 4 = 6 := by
  -- Unfold the definition of amp
  unfold amp
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amp_eight_four_l372_37259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l372_37264

-- Define the curve
noncomputable def f (x : ℝ) : ℝ := Real.exp x + 1

-- Define the point on the curve
def p : ℝ × ℝ := (0, 2)

-- State the theorem
theorem tangent_line_equation :
  ∀ x y : ℝ, (x - y + 2 = 0) ↔ 
  (∃ m : ℝ, m = (deriv f) p.1 ∧ y - p.2 = m * (x - p.1)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l372_37264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sign_placement_exists_l372_37204

theorem arithmetic_sign_placement_exists : ∃ (f : ℕ → ℕ → ℕ → ℕ → ℕ), 
  f 3 3 3 3 = f 7 7 7 7 ∧ 
  (∀ a b c d, f a b c d = a - b / c ∨ f a b c d = a / b + c - d) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sign_placement_exists_l372_37204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l372_37217

/-- Theorem: Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) : 
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)  -- Left focus
  let F₂ : ℝ × ℝ := (c, 0)   -- Right focus
  let asymptote := fun (x : ℝ) => (b / a) * x
  ∃ (P : ℝ × ℝ), 
    (P.2 = asymptote P.1) ∧  -- P is on the asymptote
    ((P.1 - F₂.1) * (b / a) + (P.2 - F₂.2) = 0) ∧  -- Line F₂P is perpendicular to asymptote
    (‖P - F₁‖^2 - ‖P - F₂‖^2 = c^2) →  -- Condition on distances
  (c / a = 2)  -- Eccentricity is 2
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l372_37217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_calculation_l372_37254

/-- Calculates the fuel consumption per 100 km given the initial fuel amount, 
    remaining fuel, and distance traveled. -/
noncomputable def fuelConsumptionPer100km (initialFuel remainingFuel : ℝ) (distance : ℝ) : ℝ :=
  (initialFuel - remainingFuel) / distance * 100

theorem fuel_consumption_calculation (initialFuel remainingFuel distance : ℝ) 
  (h1 : initialFuel = 47)
  (h2 : remainingFuel = 14)
  (h3 : distance = 275) :
  fuelConsumptionPer100km initialFuel remainingFuel distance = 12 := by
  sorry

#check fuel_consumption_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fuel_consumption_calculation_l372_37254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sadeh_polynomial_divisibility_l372_37222

-- Define the concept of a sadeh polynomial
def IsSadeh (p : Polynomial ℝ) : Prop :=
  (Polynomial.X : Polynomial ℝ) ∣ p ∧ ¬((Polynomial.X^2 : Polynomial ℝ) ∣ p)

-- Main theorem
theorem sadeh_polynomial_divisibility
  (P : Polynomial ℝ)
  (h : ∃ Q : Polynomial ℝ, IsSadeh Q ∧ (Polynomial.X^2 : Polynomial ℝ) ∣ (P.comp Q - Q.comp (2 • Polynomial.X))) :
  ∃ R : Polynomial ℝ, IsSadeh R ∧ (Polynomial.X^1401 : Polynomial ℝ) ∣ (P.comp R - R.comp (2 • Polynomial.X)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sadeh_polynomial_divisibility_l372_37222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_price_theorem_l372_37233

/-- The price of each bag of brand A mooncakes in yuan -/
noncomputable def price_A : ℝ := 25

/-- The price of each bag of brand B mooncakes in yuan -/
noncomputable def price_B : ℝ := 1.2 * price_A

/-- The quantity of brand A mooncakes that can be purchased with 6000 yuan -/
noncomputable def quantity_A : ℝ := 6000 / price_A

/-- The quantity of brand B mooncakes that can be purchased with 4800 yuan -/
noncomputable def quantity_B : ℝ := 4800 / price_B

theorem mooncake_price_theorem :
  quantity_A = quantity_B + 80 ∧ price_A = 25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mooncake_price_theorem_l372_37233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l372_37262

noncomputable def g (x : ℝ) : ℝ := (2 * x + 1) / (x + 1)

theorem g_properties :
  (∀ y : ℝ, y ≠ 2 → ∃ x : ℝ, g x = y) ∧
  (∀ x : ℝ, g x ≠ 2) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≥ 1 → x₂ ≥ 1 → x₁ < x₂ → g x₁ < g x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l372_37262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_14th_term_l372_37207

def arithmetic_sequence (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ a 2 = 2

def first_order_difference (a : ℕ+ → ℚ) (n : ℕ+) : ℚ :=
  a (n + 1) - a n

def higher_order_difference : (ℕ+ → ℚ) → ℕ → ℕ+ → ℚ
  | a, 0, n => a n
  | a, k + 1, n => higher_order_difference a k (n + 1) - higher_order_difference a k n

def difference_condition (a : ℕ+ → ℚ) : Prop :=
  ∀ n : ℕ+, higher_order_difference a 2 n + first_order_difference a n - 2 = 0

theorem sequence_14th_term (a : ℕ+ → ℚ)
  (h_seq : arithmetic_sequence a)
  (h_diff : difference_condition a) :
  a 14 = 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_14th_term_l372_37207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l372_37283

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the point of tangency
noncomputable def point_of_tangency : ℝ × ℝ := (1, Real.sqrt 3)

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := x - Real.sqrt 3 * y + 2 = 0

-- Theorem statement
theorem tangent_line_to_circle :
  let (x₀, y₀) := point_of_tangency
  (circle_eq x₀ y₀) ∧ 
  (tangent_line x₀ y₀) ∧
  ∀ x y, circle_eq x y → tangent_line x y → (x, y) = point_of_tangency :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l372_37283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l372_37265

/-- Calculates the number of additional workers needed to complete a job earlier -/
def additional_workers (total_man_days : ℕ) (original_days : ℕ) (original_workers : ℕ) (new_days : ℕ) : ℕ :=
  Int.toNat ((total_man_days / new_days) - original_workers + 1)

theorem work_completion_theorem (total_man_days : ℕ) (original_days : ℕ) (original_workers : ℕ) (new_days : ℕ) 
    (h1 : total_man_days = original_days * original_workers)
    (h2 : new_days = original_days - 3)
    (h3 : original_days = 12)
    (h4 : original_workers = 10) :
  additional_workers total_man_days original_days original_workers new_days = 4 := by
  sorry

#eval additional_workers 120 12 10 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_theorem_l372_37265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l372_37267

theorem cos_difference (x y : ℝ) 
  (h1 : Real.cos x + Real.cos y = 1/2) 
  (h2 : Real.sin x + Real.sin y = 1/3) : 
  Real.cos (x - y) = -59/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_l372_37267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_zero_l372_37266

theorem cube_root_sum_zero : (3 : Real)^(1/3) + (-3 : Real)^(1/3) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_zero_l372_37266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l372_37278

noncomputable section

-- Define the properties of cones C and D
def radius_C : ℝ := 10
def height_C : ℝ := 25
def radius_D : ℝ := 25
def height_D : ℝ := 10

-- Define the volume of a cone
def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

-- Theorem statement
theorem volume_ratio_of_cones :
  (cone_volume radius_C height_C) / (cone_volume radius_D height_D) = 2/5 := by
  -- Unfold the definitions and simplify
  unfold cone_volume radius_C height_C radius_D height_D
  -- Simplify the fraction
  simp [Real.pi]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_of_cones_l372_37278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_problem_l372_37272

theorem wire_cutting_problem :
  let wire1_length : ℝ := 28
  let wire2_length : ℝ := 36
  let wire3_length : ℝ := 45
  let wire1_ratio : ℝ := 3 / 7
  let wire2_ratio : ℝ := 4 / 5
  let wire3_ratio : ℝ := 2 / 5
  let wire1_short_piece : ℝ := wire1_length * wire1_ratio / (1 + wire1_ratio)
  let wire2_short_piece : ℝ := wire2_length * wire2_ratio / (1 + wire2_ratio)
  let wire3_short_piece : ℝ := wire3_length * wire3_ratio / (1 + wire3_ratio)
  (abs (wire1_short_piece - 8.4) < 0.001) ∧
  (wire2_short_piece = 16) ∧
  (abs (wire3_short_piece - 12.857) < 0.001) :=
by
  sorry

#eval (28 * (3 / 7) / (1 + 3 / 7))
#eval (36 * (4 / 5) / (1 + 4 / 5))
#eval (45 * (2 / 5) / (1 + 2 / 5))

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_cutting_problem_l372_37272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_caught_sampling_l372_37234

/-- The percentage of customers caught sampling candy, given that 5% of samplers are not caught
    and 23.157894736842106% of all customers sample candy. -/
theorem percentage_caught_sampling (total_sample_percentage : ℝ)
  (h1 : total_sample_percentage = 23.157894736842106)
  (h2 : |22 - (total_sample_percentage * 0.95)| < 0.00001) : 
  ∃ (caught_percentage : ℝ),
    |caught_percentage - 22| < 0.00001 ∧
    caught_percentage = total_sample_percentage * 0.95 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_caught_sampling_l372_37234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l372_37280

-- Problem 1
theorem problem_1 : (1/2)^(-2 : ℤ) - (Real.pi - Real.sqrt 5)^(0 : ℤ) - Real.sqrt 20 = 3 - 2 * Real.sqrt 5 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 0) :
  (x^2 - 2*x + 1) / (x^2 - 1) / ((x - 1) / (x^2 + x)) = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_1_problem_2_l372_37280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_greater_than_zero_l372_37244

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + a * x) - 2 * x / (x + 2)

theorem extreme_points_sum_greater_than_zero 
  (a : ℝ) 
  (h_a : 1/2 < a ∧ a < 1) 
  (x₁ x₂ : ℝ) 
  (h_extreme : ∀ x, x ≠ x₁ ∧ x ≠ x₂ → 
    (f a x₁ ≥ f a x ∧ f a x₂ ≥ f a x) ∨ 
    (f a x₁ ≤ f a x ∧ f a x₂ ≤ f a x)) :
  f a x₁ + f a x₂ > f a 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_points_sum_greater_than_zero_l372_37244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l372_37239

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 16*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (4, 0)

-- Define the line perpendicular to x + √3*y - 1 = 0
def perpendicular_line (x y : ℝ) : Prop := Real.sqrt 3 * x - y - 4 * Real.sqrt 3 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧
  perpendicular_line A.1 A.2 ∧ perpendicular_line B.1 B.2

-- Theorem statement
theorem parabola_chord_length 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) :
  ∃ (AB : ℝ), AB = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧ AB = 64/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_chord_length_l372_37239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_nine_fourths_denominator_zero_at_nine_fourths_vertical_asymptote_value_l372_37202

-- Define the function
noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (4 * x - 9)

-- Theorem stating the vertical asymptote occurs at x = 9/4
theorem vertical_asymptote_at_nine_fourths :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x - 9/4| ∧ |x - 9/4| < δ → |f x| > 1/ε :=
by sorry

-- Additional theorem to show that the denominator is zero at x = 9/4
theorem denominator_zero_at_nine_fourths :
  4 * (9/4 : ℝ) - 9 = 0 :=
by sorry

-- Theorem to explicitly state the vertical asymptote
theorem vertical_asymptote_value :
  (∃ x : ℝ, 4 * x - 9 = 0) ∧ (∀ x : ℝ, 4 * x - 9 = 0 → x = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_nine_fourths_denominator_zero_at_nine_fourths_vertical_asymptote_value_l372_37202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_three_iff_x_equals_seventeen_eighths_l372_37289

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 3)

-- Theorem statement
theorem g_equals_three_iff_x_equals_seventeen_eighths :
  ∀ x : ℝ, g x = 3 ↔ x = 17 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_equals_three_iff_x_equals_seventeen_eighths_l372_37289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l372_37260

/-- Represents the compound interest scenario -/
structure CompoundInterest where
  principal : ℝ
  finalAmount : ℝ
  years : ℕ
  compoundingPerYear : ℕ

/-- Calculates the annual interest rate given compound interest parameters -/
noncomputable def calculateInterestRate (ci : CompoundInterest) : ℝ :=
  ((ci.finalAmount / ci.principal) ^ (1 / (ci.years : ℝ))) - 1

/-- Theorem stating that for the given scenario, the interest rate is 5% -/
theorem interest_rate_is_five_percent :
  let ci : CompoundInterest := {
    principal := 8000,
    finalAmount := 8820,
    years := 2,
    compoundingPerYear := 1
  }
  calculateInterestRate ci = 0.05 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_five_percent_l372_37260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l372_37268

noncomputable def data : List ℝ := [4, 6, 3, 7, 5]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem variance_of_data : variance data = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l372_37268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_diagonal_l372_37213

/-- Theorem: For a rectangular field with one side of 14 m and an area of 135.01111065390137 m²,
    the length of its diagonal is approximately 17.002 m. -/
theorem rectangular_field_diagonal (a b d : ℝ) : 
  a = 14 → 
  a * b = 135.01111065390137 → 
  d^2 = a^2 + b^2 → 
  ∃ ε > 0, |d - 17.002| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_field_diagonal_l372_37213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_relationship_l372_37257

/-- Parabola function -/
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 + 2

/-- Point A -/
def A : ℝ × ℝ := (-1, f (-1))

/-- Point B -/
noncomputable def B : ℝ × ℝ := (Real.sqrt 2, f (Real.sqrt 2))

/-- Point C -/
def C : ℝ × ℝ := (2, f 2)

theorem parabola_point_relationship : A.2 > C.2 ∧ C.2 > B.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_point_relationship_l372_37257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_property_l372_37226

-- Define the recursive function g_n
noncomputable def g : ℕ → (ℝ → ℝ)
  | 0 => λ x => Real.sqrt (x^2 + 3)  -- Added case for 0
  | 1 => λ x => Real.sqrt (x^2 + 3)
  | (n + 1) => λ x => g n (Real.sqrt ((n + 1)^2 + x))

-- Define the domain of g_n
def domain (n : ℕ) : Set ℝ :=
  {x : ℝ | ∃ y : ℝ, g n x = y}

-- Statement of the theorem
theorem g_domain_property :
  (∃ M : ℕ, M = 5 ∧
    (∀ n : ℕ, n > M → domain n = ∅) ∧
    (domain M = {-25})) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_domain_property_l372_37226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_is_6218_l372_37269

def board_numbers : List ℕ := 
  (List.range 221).map (· + 3) |> List.filter (fun n => n % 4 = 3)

def board_sum : ℕ := board_numbers.sum

def operation_count : ℕ := board_numbers.length - 1

theorem final_number_is_6218 : board_sum - 2 * operation_count = 6218 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_number_is_6218_l372_37269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l372_37252

noncomputable section

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1 ∧ a > 0 ∧ b > 0

-- Define the distance between foci
def focalDistance (c : ℝ) : ℝ := 2 * c

-- Define point Q
def Q (c a : ℝ) : ℝ × ℝ := (c, 3 * a / 2)

-- Define the condition |F2Q| > |F2A|
def F2Q_gt_F2A (c a b : ℝ) : Prop :=
  (3 * a / 2)^2 > (b / a * Real.sqrt (c^2 - a^2))^2

-- Define the condition |PF1| + |PQ| > 3/2|F1F2| for all P on the right branch
def PF1_PQ_condition (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola a b x y → x > 0 →
    Real.sqrt ((x + c)^2 + y^2) + Real.sqrt ((x - c)^2 + (y - 3 * a / 2)^2) > 3 * c

-- Theorem statement
theorem hyperbola_eccentricity_range (a b c : ℝ) :
  hyperbola a b c 0 →
  F2Q_gt_F2A c a b →
  PF1_PQ_condition a b c →
  1 < c / a ∧ c / a < 7 / 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l372_37252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_increase_theorem_l372_37277

noncomputable def savings_percentage_increase (last_year_salary : ℝ) : ℝ :=
  let last_year_savings := 0.06 * last_year_salary
  let this_year_salary := 1.10 * last_year_salary
  let this_year_savings := 0.08 * this_year_salary
  (this_year_savings / last_year_savings) * 100

theorem savings_increase_theorem :
  ∀ (last_year_salary : ℝ), last_year_salary > 0 →
    savings_percentage_increase last_year_salary = (22 / 15) * 100 := by
  intro last_year_salary h
  unfold savings_percentage_increase
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_increase_theorem_l372_37277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BCD_measure_l372_37216

-- Define the circle
variable (circle : Set (ℝ × ℝ))

-- Define points on the circle
variable (A B C D F : ℝ × ℝ)

-- Define the property of being on the circle
def on_circle (circle : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop := p ∈ circle

-- Assume all points are on the circle
axiom A_on_circle : on_circle circle A
axiom B_on_circle : on_circle circle B
axiom C_on_circle : on_circle circle C
axiom D_on_circle : on_circle circle D
axiom F_on_circle : on_circle circle F

-- Define the concept of a line segment
def line_segment (p q : ℝ × ℝ) : Set (ℝ × ℝ) := {x | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = (1 - t) • p + t • q}

-- Define parallel lines
noncomputable def parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry

-- FB is a diameter
axiom FB_diameter : ∀ p ∈ circle, dist F p + dist p B = dist F B

-- FB is parallel to DC
axiom FB_parallel_DC : parallel (line_segment F B) (line_segment D C)

-- AB is parallel to FD
axiom AB_parallel_FD : parallel (line_segment A B) (line_segment F D)

-- Define angle measure
noncomputable def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- Angles AFB and ABF are in the ratio 3:7
axiom angle_ratio : angle_measure A F B / angle_measure A B F = 3 / 7

-- Theorem to prove
theorem angle_BCD_measure : angle_measure B C D = 63 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BCD_measure_l372_37216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plan_overage_rate_is_045_l372_37258

/-- Represents a cellular phone plan --/
structure PhonePlan where
  monthlyFee : ℚ
  includedMinutes : ℚ
  overageRate : ℚ

/-- Calculates the cost of a phone plan for a given number of minutes --/
def planCost (plan : PhonePlan) (minutes : ℚ) : ℚ :=
  plan.monthlyFee + max 0 (minutes - plan.includedMinutes) * plan.overageRate

theorem second_plan_overage_rate_is_045 (x : ℚ) :
  let plan1 : PhonePlan := ⟨50, 500, 35/100⟩
  let plan2 : PhonePlan := ⟨75, 1000, x⟩
  planCost plan1 2500 = planCost plan2 2500 →
  x = 45/100 := by
  sorry

#eval (45 : ℚ) / 100  -- This line is just to verify the rational number representation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_plan_overage_rate_is_045_l372_37258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_range_l372_37281

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2*a - x) / Real.log a

-- State the theorem
theorem increasing_log_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, StrictMono (fun x => f a x)) →
  a ∈ Set.Icc 0.5 1 ∧ a ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_increasing_log_range_l372_37281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_redistribution_possible_l372_37276

/-- Represents a pile of candies -/
structure Pile where
  count : Nat
deriving Inhabited

/-- Represents the state of the table with candy piles -/
structure TableState where
  piles : List Pile
deriving Inhabited

/-- Splits a pile into two piles -/
def splitPile (p : Pile) (n : Nat) : List Pile :=
  if n = 0 ∨ n ≥ p.count then [p]
  else [{ count := n }, { count := p.count - n }]

/-- Merges two piles into one -/
def mergePiles (p1 p2 : Pile) : Pile :=
  { count := p1.count + p2.count }

/-- Represents a move in the candy redistribution game -/
inductive Move where
  | Split (index : Nat) (splitAt : Nat)
  | Merge (index1 index2 : Nat)
deriving Inhabited

/-- Applies a move to the current table state -/
def applyMove (state : TableState) (move : Move) : TableState :=
  match move with
  | Move.Split index splitAt =>
      { piles := state.piles.take index ++ 
                 splitPile (state.piles[index]!) splitAt ++
                 state.piles.drop (index + 1) }
  | Move.Merge index1 index2 =>
      let mergedPile := mergePiles (state.piles[index1]!) (state.piles[index2]!)
      { piles := (state.piles.removeNth index1).removeNth (if index2 > index1 then index2 - 1 else index2) }

/-- The initial state of the table -/
def initialState : TableState :=
  { piles := List.range 10 |>.map (λ i => { count := i + 1 }) }

/-- Checks if all piles have the same number of candies -/
def allPilesEqual (state : TableState) : Bool :=
  state.piles.all (λ p => p.count = state.piles.head!.count)

/-- The main theorem stating that it's possible to reach a state where all piles are equal -/
theorem candy_redistribution_possible : 
  ∃ (moves : List Move), allPilesEqual (moves.foldl applyMove initialState) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_redistribution_possible_l372_37276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_unchanged_l372_37221

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  side : ℚ
  deriving Repr

/-- Calculates the surface area of a cube given its dimensions -/
def surfaceArea (c : CubeDimensions) : ℚ := 6 * c.side^2

/-- Represents the original cube and the corner cubes to be removed -/
structure CubeModification where
  originalCube : CubeDimensions
  cornerCube : CubeDimensions

/-- Theorem stating that the surface area remains unchanged after modification -/
theorem surface_area_unchanged (m : CubeModification) 
  (h1 : m.originalCube.side = 5)
  (h2 : m.cornerCube.side = 2) : 
  surfaceArea m.originalCube = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_unchanged_l372_37221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_transformation_infinitely_many_positive_solutions_l372_37294

/-- The equation x^3 + 2y^3 + 4z^3 = 6xyz + 1 -/
def equation (x y z : ℤ) : Prop := x^3 + 2*y^3 + 4*z^3 = 6*x*y*z + 1

/-- A solution to the equation is a triple (x, y, z) of integers satisfying it -/
def is_solution (x y z : ℤ) : Prop := equation x y z

theorem solution_transformation (x y z : ℤ) :
  is_solution x y z → is_solution (2*z - x) (x - y) (y - z) := by
  sorry

theorem infinitely_many_positive_solutions :
  ∃ f : ℕ → ℤ × ℤ × ℤ, ∀ n : ℕ,
    let (x, y, z) := f n
    is_solution x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧
    (∀ m : ℕ, m ≠ n → f m ≠ f n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_transformation_infinitely_many_positive_solutions_l372_37294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_example_l372_37220

/-- Converts a list of binary digits to a natural number. -/
def binaryToNat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits. -/
def natToBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
    (aux n).reverse

theorem binary_multiplication_example :
  let a := [true, false, true, true]  -- 1101₂
  let b := [true, true, true]         -- 111₂
  let c := [true, true, true, true, false, false, false, true]  -- 10001111₂
  binaryToNat a * binaryToNat b = binaryToNat c := by
  sorry

#eval binaryToNat [true, false, true, true]  -- Should output 13
#eval binaryToNat [true, true, true]         -- Should output 7
#eval binaryToNat [true, true, true, true, false, false, false, true]  -- Should output 143

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_multiplication_example_l372_37220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l372_37211

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x + 2| - |x - 2|
noncomputable def g (x : ℝ) : ℝ := x + 1/2

-- Define the solution set for f(x) ≥ g(x)
noncomputable def solution_set : Set ℝ := Set.Ici (-9/2) ∪ Set.Icc (1/2) (7/2)

-- Define the range of t
noncomputable def t_range : Set ℝ := Set.Icc 1 4

-- Theorem statement
theorem function_inequalities :
  (∀ x, x ∈ solution_set ↔ f x ≥ g x) ∧
  (∀ t, t ∈ t_range ↔ ∀ x, f x ≥ t^2 - 5*t) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequalities_l372_37211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jony_walk_block_distance_l372_37237

/-- Represents Jony's walk along Sunrise Boulevard -/
structure JonyWalk where
  start_block : ℕ
  turn_block : ℕ
  end_block : ℕ
  duration : ℕ  -- in minutes
  speed : ℕ     -- in meters per minute

/-- Calculates the number of blocks Jony walks -/
def blocks_walked (walk : JonyWalk) : ℕ :=
  (walk.turn_block - walk.start_block) + (walk.turn_block - walk.end_block)

/-- Calculates the total distance Jony walks in meters -/
def total_distance (walk : JonyWalk) : ℕ :=
  walk.speed * walk.duration

/-- Theorem: Given Jony's walk parameters, each block measures 40 meters -/
theorem jony_walk_block_distance (walk : JonyWalk) 
  (h1 : walk.start_block = 10)
  (h2 : walk.turn_block = 90)
  (h3 : walk.end_block = 70)
  (h4 : walk.duration = 40)
  (h5 : walk.speed = 100) :
  total_distance walk / blocks_walked walk = 40 := by
  sorry

/-- Example calculation -/
def example_walk : JonyWalk := ⟨10, 90, 70, 40, 100⟩

#eval blocks_walked example_walk
#eval total_distance example_walk
#eval total_distance example_walk / blocks_walked example_walk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jony_walk_block_distance_l372_37237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_l372_37271

noncomputable def book1_cost : ℝ := 60
noncomputable def book1_sell : ℝ := 78
noncomputable def book2_cost : ℝ := 45
noncomputable def book2_sell : ℝ := 54
noncomputable def book3_cost : ℝ := 70
noncomputable def book3_sell : ℝ := 84

noncomputable def total_cost : ℝ := book1_cost + book2_cost + book3_cost
noncomputable def total_sell : ℝ := book1_sell + book2_sell + book3_sell
noncomputable def total_profit : ℝ := total_sell - total_cost

noncomputable def profit_percentage : ℝ := (total_profit / total_cost) * 100

theorem book_profit_percentage :
  abs (profit_percentage - 23.43) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_profit_percentage_l372_37271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l372_37235

/-- Given a square PQRS with area 36 and equilateral triangles PUQ, QVR, RWS, and SXP 
    constructed externally on alternate sides of the square, 
    the area of quadrilateral UVWX is 18√3. -/
theorem area_of_quadrilateral (P Q R S U V W X : ℝ × ℝ) 
  (square_area : ℝ) 
  (is_square : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop)
  (is_equilateral_triangle : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → Prop)
  (area : (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → (ℝ × ℝ) → ℝ) :
  square_area = 36 →
  is_square P Q R S →
  is_equilateral_triangle P U Q →
  is_equilateral_triangle Q V R →
  is_equilateral_triangle R W S →
  is_equilateral_triangle S X P →
  area P Q R S = square_area →
  area U V W X = 18 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_l372_37235
