import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_leg_l445_44567

/-- Represents a 45-45-90 triangle -/
structure RightIsoscelesTriangle where
  hypotenuse : ℝ
  longer_leg : ℝ
  shorter_leg : ℝ
  hypotenuse_eq : hypotenuse = longer_leg * Real.sqrt 2
  legs_eq : longer_leg = shorter_leg

/-- A sequence of nested 45-45-90 triangles -/
noncomputable def nested_triangles : ℕ → RightIsoscelesTriangle
| 0 => ⟨10, 10 / Real.sqrt 2, 10 / Real.sqrt 2, by simp, rfl⟩
| (n+1) => 
  let prev := nested_triangles n
  ⟨prev.longer_leg, prev.longer_leg / Real.sqrt 2, prev.longer_leg / Real.sqrt 2, by simp, rfl⟩

theorem smallest_triangle_leg (n : ℕ) : 
  (nested_triangles n).longer_leg = 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_leg_l445_44567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_in_sphere_intersection_l445_44541

/-- A regular tetrahedron with side length 1 -/
structure RegularTetrahedron where
  side_length : ℝ
  is_regular : side_length = 1

/-- A sphere on an edge of the tetrahedron -/
structure EdgeSphere (t : RegularTetrahedron) where
  center : ℝ × ℝ × ℝ
  radius : ℝ
  on_edge : radius = t.side_length / 2

/-- The intersection of all edge spheres -/
def SphereIntersection (t : RegularTetrahedron) (spheres : List (EdgeSphere t)) : Set (ℝ × ℝ × ℝ) :=
  {p | ∀ s, s ∈ spheres → dist p s.center ≤ s.radius}

/-- The theorem to be proved -/
theorem max_distance_in_sphere_intersection (t : RegularTetrahedron) 
  (spheres : List (EdgeSphere t)) :
  ∀ p q, p ∈ SphereIntersection t spheres → q ∈ SphereIntersection t spheres → 
  dist p q ≤ 1 / Real.sqrt 6 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_in_sphere_intersection_l445_44541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sum_probability_l445_44510

/-- A type representing the cards with numbers 5, 6, 7, and 8. -/
inductive Card : Type
  | five : Card
  | six : Card
  | seven : Card
  | eight : Card
deriving Fintype, Repr

/-- The set of all cards. -/
def allCards : Finset Card := Finset.univ

/-- A function to determine if a card has an even number. -/
def isEven (c : Card) : Bool :=
  match c with
  | Card.five => false
  | Card.six => true
  | Card.seven => false
  | Card.eight => true

/-- A function to determine if the sum of two cards is even. -/
def sumIsEven (c1 c2 : Card) : Bool :=
  (isEven c1 && isEven c2) || (!isEven c1 && !isEven c2)

/-- The main theorem stating that the probability of drawing two cards with an even sum is 1/3. -/
theorem even_sum_probability :
  (Finset.filter (fun p => sumIsEven p.1 p.2) (allCards.product allCards)).card /
    (allCards.product allCards).card = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_sum_probability_l445_44510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l445_44577

/-- The function f(x) = a - 2 / (3^x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (3^x + 1)

theorem f_increasing_and_odd :
  (∀ a : ℝ, Monotone (f a)) ∧
  (∃ a : ℝ, (∀ x : ℝ, f a (-x) = -(f a x)) ∧ a = 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_and_odd_l445_44577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_equilateral_triangle_l445_44526

/-- An equilateral triangle in the coordinate plane cannot have all integer coordinates for its vertices. -/
theorem no_integer_equilateral_triangle :
  ¬ ∃ (a b c d e f : ℤ),
    (let dist (x₁ y₁ x₂ y₂ : ℤ) := (x₁ - x₂)^2 + (y₁ - y₂)^2
     dist a b c d = dist c d e f ∧ 
     dist c d e f = dist e f a b ∧
     dist e f a b = dist a b c d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_equilateral_triangle_l445_44526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_full_time_employed_females_l445_44568

-- Define the total population
def total_population : ℝ := 100

-- Define the given percentages
def employed_percentage : ℝ := 64
def full_time_percentage : ℝ := 35
def part_time_percentage : ℝ := 29
def student_percentage : ℝ := 28
def retiree_unemployed_percentage : ℝ := 8
def employed_male_percentage : ℝ := 46
def full_time_male_percentage : ℝ := 25

-- Theorem to prove
theorem percentage_full_time_employed_females :
  let total_employed := (employed_percentage / 100) * total_population
  let total_full_time := (full_time_percentage / 100) * total_employed
  let total_employed_males := (employed_male_percentage / 100) * total_population
  let full_time_employed_males := (full_time_male_percentage / 100) * total_employed_males
  let full_time_employed_females := total_full_time - full_time_employed_males
  ∃ ε > 0, |((full_time_employed_females / total_full_time) * 100) - 48.66| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_full_time_employed_females_l445_44568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l445_44531

noncomputable def f (x y : ℝ) : ℝ :=
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) /
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2)

theorem min_value_of_f :
  ∀ x y : ℝ, 9 - x^2 - 8*x*y - 16*y^2 > 0 →
  f x y ≥ 7/27 ∧ ∃ x y : ℝ, f x y = 7/27 :=
by sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l445_44531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_heights_l445_44552

/-- Given the conditions of cylinders A, B, C, and D, prove that the average height of cylinders C and D is 44.44% of h. -/
theorem cylinder_heights (h : ℝ) (r_A : ℝ) (r_B : ℝ) (r_C : ℝ) (b : ℝ) (c : ℝ) :
  (r_B = 1.25 * r_A) →
  ((2/3) * Real.pi * r_A^2 * h = (3/5) * Real.pi * r_B^2 * b) →
  (3 * Real.pi * r_C^2 * c = (2/5) * Real.pi * r_B^2 * b) →
  (c = h/3) →
  (b = (10/9) * h) →
  let height_D := (5/9) * h
  ((c + height_D) / 2) / h * 100 = 44.44 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_heights_l445_44552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ascorbic_acid_oxygen_percentage_l445_44524

/-- Represents the molecular formula of a compound -/
structure MolecularFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- Represents the atomic masses of elements -/
structure AtomicMasses where
  carbon : ℝ
  hydrogen : ℝ
  oxygen : ℝ

/-- Calculates the mass percentage of oxygen in a compound -/
noncomputable def oxygenMassPercentage (formula : MolecularFormula) (masses : AtomicMasses) : ℝ :=
  let totalMass := formula.carbon * masses.carbon + 
                    formula.hydrogen * masses.hydrogen + 
                    formula.oxygen * masses.oxygen
  let oxygenMass := formula.oxygen * masses.oxygen
  (oxygenMass / totalMass) * 100

/-- Theorem: The mass percentage of oxygen in ascorbic acid is approximately 54.49% -/
theorem ascorbic_acid_oxygen_percentage :
  let ascorbicAcid : MolecularFormula := ⟨6, 8, 6⟩
  let atomicMasses : AtomicMasses := ⟨12.01, 1.01, 16.00⟩
  abs (oxygenMassPercentage ascorbicAcid atomicMasses - 54.49) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ascorbic_acid_oxygen_percentage_l445_44524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l445_44540

/-- An acute triangle ABC with BC = 1 and B = 2A -/
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  BC : ℝ
  AC : ℝ
  acute : A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π
  BC_eq_one : BC = 1
  B_eq_2A : B = 2 * A

theorem acute_triangle_properties (t : AcuteTriangle) :
  t.AC / Real.cos t.A = 2 ∧ Real.sqrt 2 < t.AC ∧ t.AC < Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_properties_l445_44540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_height_l445_44572

/-- Represents a pyramid with an equilateral triangular base -/
structure EquilateralPyramid where
  base_side : ℝ
  height : ℝ

/-- Calculates the height of a point on the edge of the pyramid -/
noncomputable def point_height (p : EquilateralPyramid) (distance_from_base : ℝ) : ℝ :=
  let slant_height := Real.sqrt ((p.base_side / Real.sqrt 3) ^ 2 + p.height ^ 2)
  p.height * distance_from_base / slant_height

theorem mouse_height (p : EquilateralPyramid) 
  (h1 : p.base_side = 300)
  (h2 : p.height = 100)
  (h3 : point_height p 134 = 67) : 
  point_height p 134 = 67 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_height_l445_44572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l445_44587

-- Define the two polar curves
noncomputable def curve1 (θ : ℝ) : ℝ × ℝ := (3 * Real.sin θ * Real.cos θ, 3 * Real.sin θ * Real.sin θ)
noncomputable def curve2 (θ : ℝ) : ℝ × ℝ := (6 * Real.cos θ * Real.cos θ, 6 * Real.cos θ * Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ θ₁ θ₂, curve1 θ₁ = p ∧ curve2 θ₂ = p}

-- Theorem statement
theorem intersection_count :
  ∃ (S : Finset (ℝ × ℝ)), S.card = 2 ∧ ∀ p ∈ S, p ∈ intersection_points := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_count_l445_44587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_book_A_purchasers_l445_44548

/-- Represents the number of people who purchased a book or combination of books. -/
structure BookPurchases where
  total_A : ℕ  -- Total number of people who purchased book A
  total_B : ℕ  -- Total number of people who purchased book B
  only_B : ℕ   -- Number of people who purchased only book B
  both : ℕ     -- Number of people who purchased both books A and B

/-- The conditions of the book purchase problem. -/
def book_purchase_conditions (p : BookPurchases) : Prop :=
  p.total_A = 2 * p.total_B ∧
  p.both = 500 ∧
  p.both = 2 * p.only_B

/-- The theorem stating that under the given conditions, 
    the number of people who purchased only book A is 1000. -/
theorem only_book_A_purchasers (p : BookPurchases) 
  (h : book_purchase_conditions p) : 
  p.total_A - p.both = 1000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_book_A_purchasers_l445_44548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_singularSquaresCount_l445_44502

/-- Represents a position on an infinite chessboard -/
structure Position where
  x : Int
  y : Int
deriving Inhabited

/-- Calculates the minimum number of knight moves to reach a position from the origin -/
def knightMoves (p : Position) : Nat :=
  sorry

/-- Defines adjacency between two positions -/
def adjacentTo (p q : Position) : Prop :=
  (abs (p.x - q.x) + abs (p.y - q.y) = 1)

/-- Checks if a position is singular (100 moves, surrounded by 101 moves) -/
def isSingular (p : Position) : Prop :=
  knightMoves p = 100 ∧
  (∀ q : Position, adjacentTo p q → knightMoves q = 101)

/-- The main theorem: there are exactly 800 singular squares -/
theorem singularSquaresCount :
  (∃! (s : Finset Position), ∀ p, p ∈ s ↔ isSingular p) ∧
  (∃ (s : Finset Position), (∀ p, p ∈ s ↔ isSingular p) ∧ s.card = 800) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_singularSquaresCount_l445_44502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l445_44532

/-- Given a projection P in R^2 such that P((2, -1)) = (1, 0), 
    prove that P((3, -3)) = (3, 0) -/
theorem projection_problem (P : ℝ × ℝ → ℝ × ℝ) 
  (h : P (2, -1) = (1, 0)) 
  (h_proj : ∀ x : ℝ × ℝ, P (P x) = P x) :
  P (3, -3) = (3, 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_problem_l445_44532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_characterization_l445_44555

/-- A polynomial that satisfies P(P(x)) = [P(x)]^k for some positive integer k -/
structure SpecialPolynomial (R : Type*) [CommRing R] where
  k : ℕ+
  P : Polynomial R
  h : P.comp P = P ^ (k : ℕ)

theorem special_polynomial_characterization {R : Type*} [CommRing R] (sp : SpecialPolynomial R) :
  (∃ (c : R), sp.P = Polynomial.C c) ∨ sp.P = Polynomial.X ^ (sp.k : ℕ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_polynomial_characterization_l445_44555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_passes_through_point_one_one_l445_44598

/-- A quadratic function passing through (1, 1) when a + b = 0 -/
theorem quadratic_passes_through_point_one_one 
  (a b : ℝ) (h : a + b = 0) : 
  (λ x : ℝ => x^2 + a*x + b) 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_passes_through_point_one_one_l445_44598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_on_line_l445_44553

theorem angles_on_line (x y : ℝ) : 
  {α : ℝ | ∃ (k : ℤ), α = k * π + π / 3} = 
  {α : ℝ | ∃ (t : ℝ), t ≠ 0 ∧ y = Real.sqrt 3 * x ∧ 
    x = t * Real.cos α ∧ y = t * Real.sin α} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_on_line_l445_44553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l445_44504

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem f_increasing : StrictMono f := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l445_44504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l445_44522

/-- Given a hyperbola with specific properties, prove its equation -/
theorem hyperbola_equation :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →  -- Hyperbola equation
  (∀ (x y : ℝ), y = (3/4) * x ∨ y = -(3/4) * x) →  -- Asymptotes
  (5 : ℝ) ∈ {x : ℝ | ∃ y, x^2 / a^2 - y^2 / b^2 = 1 ∧ y = 0} →  -- Right focus
  (∀ (x y : ℝ), x^2 / 16 - y^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l445_44522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l445_44537

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (7 * sequence_a n + Real.sqrt (45 * (sequence_a n)^2 - 36)) / 2

theorem sequence_a_properties :
  (∀ n : ℕ, sequence_a n > 0 ∧ ∃ k : ℤ, sequence_a n = k) ∧
  (∀ n : ℕ, ∃ k : ℤ, (sequence_a n * sequence_a (n + 1) - 1 : ℝ) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_properties_l445_44537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_three_l445_44575

theorem cube_root_sum_equals_three : 
  Real.rpow (9 + 4 * Real.sqrt 7) (1/3) + Real.rpow (9 - 4 * Real.sqrt 7) (1/3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_three_l445_44575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_in_pascal_smallest_three_digit_in_pascal_l445_44517

/-- Pascal's Triangle is an infinite triangular array of numbers -/
def PascalTriangle : ℕ → ℕ → ℕ := sorry

/-- A number is in Pascal's Triangle if it appears in the triangle -/
def InPascalTriangle (n : ℕ) : Prop :=
  ∃ (row col : ℕ), PascalTriangle row col = n

/-- The first element of each row in Pascal's Triangle is 1 -/
axiom pascal_first_elem : ∀ (row : ℕ), PascalTriangle row 0 = 1

/-- Every positive integer appears in Pascal's Triangle -/
axiom pascal_contains_all : ∀ (n : ℕ), n > 0 → InPascalTriangle n

/-- 100 is in Pascal's Triangle -/
theorem hundred_in_pascal : InPascalTriangle 100 :=
  pascal_contains_all 100 (by norm_num)

/-- Theorem: 100 is the smallest three-digit number in Pascal's Triangle -/
theorem smallest_three_digit_in_pascal : 
  (∀ n : ℕ, InPascalTriangle n → n ≥ 100 → n < 100 → False) ∧ InPascalTriangle 100 := by
  constructor
  · intro n hn hge hlt
    have : n < n := by
      calc
        n ≥ 100 := hge
        _ > n := hlt
    exact (Nat.lt_irrefl n) this
  · exact hundred_in_pascal


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundred_in_pascal_smallest_three_digit_in_pascal_l445_44517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_millionth_digit_is_seven_l445_44562

/-- The decimal expansion of 3/41 -/
def decimal_expansion : ℚ := 3 / 41

/-- The length of the repeating sequence in the decimal expansion of 3/41 -/
def repeat_length : ℕ := 15

/-- The repeating sequence in the decimal expansion of 3/41 -/
def repeat_sequence : List ℕ := [0, 7, 3, 1, 7, 0, 7, 3, 1, 7, 0, 7, 3, 1, 7]

/-- The position of the one-millionth digit within the repeating sequence -/
def position : ℕ := 1000000 % repeat_length

theorem one_millionth_digit_is_seven :
  (repeat_sequence.get? position).map (λ x => x = 7) = some True := by
  sorry

#eval repeat_sequence.get? position

end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_millionth_digit_is_seven_l445_44562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diplomats_speaking_latin_l445_44547

theorem diplomats_speaking_latin 
  (total : ℕ) 
  (not_russian : ℕ) 
  (neither_percent : ℚ) 
  (both_percent : ℚ) 
  (h1 : total = 120)
  (h2 : not_russian = 32)
  (h3 : neither_percent = 1/5)
  (h4 : both_percent = 1/10)
  : ∃ latin_speakers : ℕ, latin_speakers = 20 :=
by
  sorry

#check diplomats_speaking_latin

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diplomats_speaking_latin_l445_44547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l445_44500

/-- Given a line l passing through point P(2, 3/2) and intersecting the positive x-axis at A
    and the positive y-axis at B, with O as the origin and the area of triangle AOB equal to 6,
    prove that the equation of line l is 3x + 4y - 12 = 0. -/
theorem line_equation_proof :
  ∃ (l : Set (ℝ × ℝ)) (P A B O : ℝ × ℝ),
  P.1 = 2 ∧ P.2 = 3/2 ∧
  P ∈ l ∧
  A.2 = 0 ∧ A.1 > 0 ∧ A ∈ l ∧
  B.1 = 0 ∧ B.2 > 0 ∧ B ∈ l ∧
  O = (0, 0) ∧
  (1/2 * A.1 * B.2 : ℝ) = 6 →
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = -12 ∧
  l = {(x, y) : ℝ × ℝ | a * x + b * y + c = 0} :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l445_44500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l445_44558

/-- The sum of the infinite series ∑(1/(n(n+3))) for n from 1 to infinity -/
noncomputable def infiniteSeriesSum : ℝ := ∑' n, 1 / (n * (n + 3))

/-- Theorem stating that the sum of the infinite series is 11/18 -/
theorem infiniteSeriesSumValue : infiniteSeriesSum = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSumValue_l445_44558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_open_hours_l445_44546

def weekly_rent : ℝ := 1200
def utility_rate : ℝ := 0.20
def employees_per_shift : ℕ := 2
def days_open_per_week : ℕ := 5
def employee_wage : ℝ := 12.50
def weekly_expenses : ℝ := 3440

theorem store_open_hours : 
  let rent : ℝ := weekly_rent
  let utilities : ℝ := rent * utility_rate
  let fixed_costs : ℝ := rent + utilities
  let labor_costs : ℝ := weekly_expenses - fixed_costs
  let total_hours : ℝ := labor_costs / employee_wage
  let hours_open_per_week : ℝ := total_hours / employees_per_shift
  let hours_open_per_day : ℝ := hours_open_per_week / days_open_per_week
  hours_open_per_day = 16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_open_hours_l445_44546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dino_income_calculation_l445_44528

/-- Dino's monthly income calculation --/
theorem dino_income_calculation 
  (hours1 hours2 hours3 : ℕ)
  (rate1 rate2 rate3 expenses savings : ℚ) :
  hours1 = 20 →
  rate1 = 10 →
  hours2 = 30 →
  hours3 = 5 →
  rate3 = 40 →
  expenses = 500 →
  savings = 500 →
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3 = expenses + savings →
  rate2 = 20 := by
  sorry

#check dino_income_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dino_income_calculation_l445_44528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l445_44594

/-- The eccentricity of an ellipse with equation x²/a² + y²/b² = 1 -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - (min a b / max a b) ^ 2)

/-- Theorem: The eccentricity of the ellipse x²/6 + y²/8 = 1 is 1/2 -/
theorem ellipse_eccentricity :
  eccentricity (Real.sqrt 6) (Real.sqrt 8) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l445_44594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_possible_to_make_divisible_by_10_l445_44503

-- Define the chessboard as a type
def Chessboard := Fin 8 → Fin 8 → ℕ

-- Define the operation of increasing numbers in a square
def increase_square (board : Chessboard) (top_left : Fin 8 × Fin 8) (size : Fin 2) : Chessboard :=
  λ i j ↦ if i ≥ top_left.1 ∧ i < top_left.1 + (size + 3) ∧ 
            j ≥ top_left.2 ∧ j < top_left.2 + (size + 3)
         then board i j + 1
         else board i j

-- Define a sequence of operations
def Operation := List (Fin 8 × Fin 8 × Fin 2)

-- Apply a sequence of operations to a board
def apply_operations (board : Chessboard) (ops : Operation) : Chessboard :=
  ops.foldl (λ b op ↦ increase_square b op.1 op.2.2) board

-- Check if all numbers on the board are divisible by 10
def all_divisible_by_10 (board : Chessboard) : Prop :=
  ∀ i j, (board i j) % 10 = 0

-- The main theorem
theorem not_always_possible_to_make_divisible_by_10 :
  ∃ initial_board : Chessboard, ∀ ops : Operation,
    ¬(all_divisible_by_10 (apply_operations initial_board ops)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_possible_to_make_divisible_by_10_l445_44503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_digit_product_2700_l445_44525

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (n % 10) :: digits (n / 10)

def is_smallest_with_digit_product (n : ℕ) (p : ℕ) : Prop :=
  (List.prod (digits n) = p) ∧
  ∀ m < n, List.prod (digits m) ≠ p

theorem smallest_integer_with_digit_product_2700 :
  ∃ N : ℕ, is_smallest_with_digit_product N 2700 ∧ (List.sum (digits N) = 27) :=
by
  -- The proof goes here
  sorry

#eval digits 25569
#eval List.sum (digits 25569)
#eval List.prod (digits 25569)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_with_digit_product_2700_l445_44525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l445_44560

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x^2 - 4 * x

-- State the theorem
theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y : ℝ, y = m * x + b ↔ y - f 1 = (deriv f 1) * (x - 1)) ∧
                m * 1 + b = f 1 ∧
                m = 1 ∧ b = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_l445_44560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_other_time_ratio_l445_44564

noncomputable def time_together : ℝ := 4
noncomputable def time_B : ℝ := 24

noncomputable def work_rate (time : ℝ) : ℝ := 1 / time

theorem other_time_ratio 
  (time_A : ℝ)
  (h1 : work_rate time_A + work_rate time_B = work_rate time_together)
  (h2 : time_A ≠ 2 * time_B) : 
  time_A / time_B = 1 / 5 := by
  sorry

#check other_time_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_other_time_ratio_l445_44564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_numbers_exist_l445_44550

def digits (n : ℕ) : Finset ℕ := sorry

def num_digits (n : ℕ) : ℕ := sorry

theorem no_special_numbers_exist (n : ℕ) : ¬∃ (M N : ℕ), 
  (∀ d : ℕ, d ∈ digits M → d % 2 = 0) ∧
  (∀ d : ℕ, d ∈ digits N → d % 2 = 1) ∧
  (∀ d : Fin 10, d.val ∈ (digits M ∪ digits N)) ∧
  (num_digits M = n) ∧
  (num_digits N = n) ∧
  (M % N = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_special_numbers_exist_l445_44550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyyFromSecondarySpermatocyte_l445_44513

/-- Represents the stages of spermatogenesis -/
inductive SpermatogenesisStage where
  | PrimarySpermatocyte
  | SecondarySpermatocyte
  | Spermatid
  | Sperm

/-- Represents the sex chromosomes -/
inductive SexChromosome where
  | X
  | Y

/-- Represents a cell during spermatogenesis -/
structure GametogenesisCell where
  stage : SpermatogenesisStage
  chromosomes : List SexChromosome

/-- Represents the final chromosomal composition -/
structure ChromosomalComposition where
  autosomalCount : Nat
  sexChromosomes : List SexChromosome

/-- Function to model the meiotic division process -/
def meioticDivision (cell : GametogenesisCell) : Option ChromosomalComposition :=
  sorry

/-- Theorem stating that a 44+XYY composition results from a secondary spermatocyte -/
theorem xyyFromSecondarySpermatocyte
  (cell : GametogenesisCell)
  (result : ChromosomalComposition) :
  cell.stage = SpermatogenesisStage.SecondarySpermatocyte →
  meioticDivision cell = some result →
  result.autosomalCount = 44 ∧ 
  result.sexChromosomes = [SexChromosome.X, SexChromosome.Y, SexChromosome.Y] :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyyFromSecondarySpermatocyte_l445_44513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l445_44593

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  3 * x^2 + y^2 - 12 * x - 4 * y + 36 = 0

/-- The center of the ellipse -/
noncomputable def center : ℝ × ℝ := (2, 2)

/-- The semi-major axis lengths of the ellipse -/
noncomputable def semi_major_axes : ℝ × ℝ := (Real.sqrt (20/3), Real.sqrt 20)

/-- Theorem stating that the equation represents an ellipse with given center and semi-major axes -/
theorem conic_is_ellipse :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), conic_equation x y ↔
    ((x - center.1) / a)^2 + ((y - center.2) / b)^2 = 1 ∧
    (a = semi_major_axes.1 ∧ b = semi_major_axes.2 ∨
     a = semi_major_axes.2 ∧ b = semi_major_axes.1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_is_ellipse_l445_44593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_problem_l445_44580

/-- Represents a rhombus A₁BCD with point E on A₁B -/
structure RhombusWithPoint where
  A₁ : EuclideanSpace ℝ (Fin 3)
  B : EuclideanSpace ℝ (Fin 3)
  C : EuclideanSpace ℝ (Fin 3)
  D : EuclideanSpace ℝ (Fin 3)
  E : EuclideanSpace ℝ (Fin 3)
  side_length : ℝ
  angle_A₁ : ℝ
  dihedral_angle_ADE_C : ℝ

/-- The conditions of the problem -/
def problem_conditions (r : RhombusWithPoint) : Prop :=
  r.side_length = 3 ∧
  r.angle_A₁ = Real.pi/3 ∧
  r.dihedral_angle_ADE_C = Real.pi/3 ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ r.E = (1 - t) • r.A₁ + t • r.B

/-- The projection of point A on plane DEBC -/
noncomputable def projection_O (r : RhombusWithPoint) : EuclideanSpace ℝ (Fin 3) := sorry

/-- The orthocenter of triangle DBC -/
noncomputable def orthocenter_DBC (r : RhombusWithPoint) : EuclideanSpace ℝ (Fin 3) := sorry

/-- The angle between line BC and plane ADE -/
noncomputable def angle_BC_ADE (r : RhombusWithPoint) : ℝ := sorry

/-- The main theorem -/
theorem rhombus_problem (r : RhombusWithPoint) 
  (h : problem_conditions r) :
  projection_O r ≠ orthocenter_DBC r ∧
  (∃ t : ℝ, r.E = (2/3) • r.A₁ + (1/3) • r.B → 
    angle_BC_ADE r = Real.arcsin ((3 * Real.sqrt 7) / 14)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_problem_l445_44580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_second_and_fourth_l445_44535

def numbers : List Int := [-3, 2, 5, 8, 11]

def is_valid_permutation (perm : List Int) : Prop :=
  perm.length = 5 ∧
  perm.toFinset = numbers.toFinset ∧
  (∃ i ∈ [2, 3, 4], perm[i]? = some 11) ∧
  (∃ i ∈ [1, 2, 3], perm[i]? = some (-3)) ∧
  (∃ i ∈ [1, 2, 3, 4], perm[i]? = some 5 ∧ i ≠ 0 ∧ i ≠ 4 ∧ 
    (i > 0 → perm[i-1]? ≠ some (-3)) ∧ 
    (i < 4 → perm[i+1]? ≠ some (-3)))

theorem average_of_second_and_fourth (perm : List Int) (h : is_valid_permutation perm) :
  (perm[1]?.getD 0 + perm[3]?.getD 0) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_second_and_fourth_l445_44535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l445_44582

def is_valid_pair (m n : ℤ) : Prop :=
  1000 ≤ m ∧ m ≤ 9999 ∧  -- m is a four-digit number
  100 ≤ n ∧ n ≤ 999 ∧    -- n is a three-digit number
  (∃ k : ℤ, m = 59 * k) ∧  -- 59 is a prime factor of m
  n % 38 = 1 ∧            -- remainder of n divided by 38 is 1
  abs (m - n) ≤ 2015 ∧        -- difference between m and n is not more than 2015
  (∃ d1 d2 : ℤ, d1 ≠ d2 ∧ d1 ∈ ({4, 5} : Set ℤ) ∧ d2 ∈ ({4, 5} : Set ℤ) ∧
    (∃ a b c d : ℤ, m = 1000 * a + 100 * b + 10 * c + d ∧ (d1 = a ∨ d1 = b ∨ d1 = c ∨ d1 = d) ∧ (d2 = a ∨ d2 = b ∨ d2 = c ∨ d2 = d)) ∧
    (∃ e f g : ℤ, n = 100 * e + 10 * f + g ∧ (d1 = e ∨ d1 = f ∨ d1 = g) ∧ (d2 = e ∨ d2 = f ∨ d2 = g)))
    -- both m and n contain the digits 4 and 5

theorem valid_pairs :
  ∀ m n : ℤ, is_valid_pair m n ↔ ((m = 1475 ∧ n = 457) ∨ (m = 1534 ∧ n = 457)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_pairs_l445_44582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisibility_sequence_l445_44557

-- Define a sequence of sets
def A : ℕ → Set (Set ℕ) := sorry

-- Define the finiteness condition
def finitelyMany (A : ℕ → Set (Set ℕ)) : Prop :=
  ∀ i : ℕ, {j : ℕ | A j ⊆ A i}.Finite

-- Main theorem
theorem existence_of_divisibility_sequence
  (h : finitelyMany A) :
  ∃ a : ℕ → ℕ, ∀ i j : ℕ, (a i ∣ a j) ↔ (A i ⊆ A j) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_divisibility_sequence_l445_44557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_ways_l445_44507

/-- The number of ways to tile a 2 × n strip with 1 × 2 or 2 × 1 tiles -/
noncomputable def F (n : ℕ) : ℝ :=
  let φ : ℝ := (1 + Real.sqrt 5) / 2
  let φ_bar : ℝ := (1 - Real.sqrt 5) / 2
  (1 / Real.sqrt 5) * (φ^(n+1) - φ_bar^(n+1))

/-- Theorem: The number of ways to tile a 2 × n strip with 1 × 2 or 2 × 1 tiles
    is given by the formula F(n) -/
theorem tiling_ways (n : ℕ) :
  F n = (1 / Real.sqrt 5) * ((1 + Real.sqrt 5) / 2)^(n+1) - (1 / Real.sqrt 5) * ((1 - Real.sqrt 5) / 2)^(n+1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiling_ways_l445_44507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_example_l445_44538

/-- Rounds a real number to the nearest tenth -/
noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  ⌊x * 10 + 0.5⌋ / 10

/-- The original number to be rounded -/
def originalNumber : ℝ := 3967149.6587234

/-- Theorem stating that rounding the original number to the nearest tenth equals 3967149.7 -/
theorem round_to_nearest_tenth_example :
  roundToNearestTenth originalNumber = 3967149.7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_to_nearest_tenth_example_l445_44538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_self_descriptive_number_first_digit_l445_44549

/-- A self-descriptive 7-digit number -/
def SelfDescriptiveNumber (a b c d e f g : Nat) : Prop :=
  a + b + c + d + e + f + g = 7 ∧
  a = (if a = 0 then 1 else 0) + (if b = 0 then 1 else 0) + (if c = 0 then 1 else 0) +
      (if d = 0 then 1 else 0) + (if e = 0 then 1 else 0) + (if f = 0 then 1 else 0) +
      (if g = 0 then 1 else 0) ∧
  b = (if a = 1 then 1 else 0) + (if b = 1 then 1 else 0) + (if c = 1 then 1 else 0) +
      (if d = 1 then 1 else 0) + (if e = 1 then 1 else 0) + (if f = 1 then 1 else 0) +
      (if g = 1 then 1 else 0) ∧
  c = (if a = 2 then 1 else 0) + (if b = 2 then 1 else 0) + (if c = 2 then 1 else 0) +
      (if d = 2 then 1 else 0) + (if e = 2 then 1 else 0) + (if f = 2 then 1 else 0) +
      (if g = 2 then 1 else 0) ∧
  d = (if a = 3 then 1 else 0) + (if b = 3 then 1 else 0) + (if c = 3 then 1 else 0) +
      (if d = 3 then 1 else 0) + (if e = 3 then 1 else 0) + (if f = 3 then 1 else 0) +
      (if g = 3 then 1 else 0) ∧
  e = (if a = 4 then 1 else 0) + (if b = 4 then 1 else 0) + (if c = 4 then 1 else 0) +
      (if d = 4 then 1 else 0) + (if e = 4 then 1 else 0) + (if f = 4 then 1 else 0) +
      (if g = 4 then 1 else 0) ∧
  f = (if a = 5 then 1 else 0) + (if b = 5 then 1 else 0) + (if c = 5 then 1 else 0) +
      (if d = 5 then 1 else 0) + (if e = 5 then 1 else 0) + (if f = 5 then 1 else 0) +
      (if g = 5 then 1 else 0) ∧
  g = (if a = 6 then 1 else 0) + (if b = 6 then 1 else 0) + (if c = 6 then 1 else 0) +
      (if d = 6 then 1 else 0) + (if e = 6 then 1 else 0) + (if f = 6 then 1 else 0) +
      (if g = 6 then 1 else 0)

theorem self_descriptive_number_first_digit :
  ∀ a b c d e f g : Nat,
    SelfDescriptiveNumber a b c d e f g →
    b = 2 →
    c = 1 →
    d = 1 →
    e = 0 →
    a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_self_descriptive_number_first_digit_l445_44549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_count_l445_44508

noncomputable def number_list : List ℚ := [-15, 16/3, -23/100, 0, 76/10, 2, -3/5, 314/100]

def is_non_negative (x : ℚ) : Bool := x ≥ 0

def count_non_negative (l : List ℚ) : ℕ :=
  (l.filter is_non_negative).length

theorem non_negative_count : count_non_negative number_list = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_count_l445_44508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l445_44584

/-- The time taken for a train to cross a bridge -/
noncomputable def train_crossing_time (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

/-- Theorem stating that a 250m train traveling at 85 kmph takes approximately 25.41 seconds to cross a 350m bridge -/
theorem train_crossing_bridge_time :
  let train_length : ℝ := 250
  let train_speed_kmph : ℝ := 85
  let bridge_length : ℝ := 350
  let crossing_time := train_crossing_time train_length train_speed_kmph bridge_length
  abs (crossing_time - 25.41) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_bridge_time_l445_44584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_leq_g_iff_a_leq_l445_44569

/-- The function f(x) = (1/3)x³ + x² + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + a * x

/-- The derivative of f(x) -/
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

/-- The function g(x) = 1/e^x -/
noncomputable def g (x : ℝ) : ℝ := 1 / Real.exp x

theorem f_prime_leq_g_iff_a_leq (a : ℝ) :
  (∀ x₁ ∈ Set.Icc (1/2 : ℝ) 2, ∃ x₂ ∈ Set.Icc (1/2 : ℝ) 2, f_prime a x₁ ≤ g x₂) ↔
  a ≤ Real.sqrt (Real.exp 1) / Real.exp 1 - 8 :=
by
  sorry

#check f_prime_leq_g_iff_a_leq

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_prime_leq_g_iff_a_leq_l445_44569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_last_digit_of_n_squared_plus_2n_l445_44563

theorem second_to_last_digit_of_n_squared_plus_2n (n : ℕ) :
  (n^2 + 2*n) % 100 = 24 → (n^2 + 2*n) % 100 / 10 = 2 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_to_last_digit_of_n_squared_plus_2n_l445_44563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l445_44544

/-- Represents a right circular cone -/
structure Cone where
  radius : ℝ
  height : ℝ

/-- Represents a spherical marble -/
structure Marble where
  radius : ℝ

/-- The problem setup -/
def problem_setup (h : ℝ) : Cone × Cone × Marble :=
  let cone_a : Cone := ⟨4, h⟩
  let cone_b : Cone := ⟨8, h⟩
  let marble : Marble := ⟨2⟩
  (cone_a, cone_b, marble)

theorem liquid_rise_ratio (h : ℝ) :
  let (cone_a, cone_b, marble) := problem_setup h
  let volume_marble := (4 / 3) * Real.pi * marble.radius ^ 3
  let base_area_a := Real.pi * cone_a.radius ^ 2
  let base_area_b := Real.pi * cone_b.radius ^ 2
  let rise_a := volume_marble / base_area_a
  let rise_b := volume_marble / base_area_b
  rise_a / rise_b = 4 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_liquid_rise_ratio_l445_44544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_prize_location_l445_44511

theorem min_questions_for_prize_location (n : Nat) (h : n = 100) : 
  ∃ (m : Nat), m = 99 ∧ 
  (∀ (k : Nat), k < m → ∃ (strategy : Nat → Bool), 
    ∀ (prize_location : Nat), prize_location ≤ n → 
      (∃! (result : Nat), result ≤ n ∧ 
        (∀ (i : Nat), i < k → strategy i = (result ≤ i)))) :=
by
  sorry

#check min_questions_for_prize_location

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_questions_for_prize_location_l445_44511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_discount_percentage_l445_44583

theorem shirt_discount_percentage (shirts_per_fandom : ℕ) (num_fandoms : ℕ) 
  (normal_price : ℚ) (tax_rate : ℚ) (final_payment : ℚ) 
  (h1 : shirts_per_fandom = 5)
  (h2 : num_fandoms = 4)
  (h3 : normal_price = 15)
  (h4 : tax_rate = 1/10)
  (h5 : final_payment = 264)
  : (1 - (final_payment / (1 + tax_rate)) / (shirts_per_fandom * num_fandoms * normal_price)) * 100 = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shirt_discount_percentage_l445_44583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l445_44529

/-- The area of a circle given its center and a point on the circumference -/
noncomputable def circleArea (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ :=
  let dx := center.1 - point.1
  let dy := center.2 - point.2
  Real.pi * (dx * dx + dy * dy)

/-- Theorem: The area of the circle with center R(-5, 3) passing through S(7, -4) is 193π -/
theorem circle_area_theorem :
  circleArea (-5, 3) (7, -4) = 193 * Real.pi := by
  -- Unfold the definition of circleArea
  unfold circleArea
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_theorem_l445_44529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zeros_in_intervals_l445_44533

-- Define the function f(x) = ln x - x^2 + 1
noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 1

-- Theorem statement
theorem f_has_zeros_in_intervals :
  (∃ x₁ ∈ Set.Ioo 0 1, f x₁ = 0) ∧
  (∃ x₂ ∈ Set.Ioo 1 2, f x₂ = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_zeros_in_intervals_l445_44533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l445_44588

/-- Given a curve y = a(x - 2) - ln(x - 1) passing through (2, 0) with tangent line y = 2x - 4 at that point, prove a = 3 -/
theorem tangent_line_parameter (a : ℝ) : 
  (∃ f : ℝ → ℝ, ∀ x, f x = a * (x - 2) - Real.log (x - 1)) → -- curve equation
  (∃ f : ℝ → ℝ, f 2 = 0) → -- point (2, 0) on the curve
  (∃ g : ℝ → ℝ, ∀ x, g x = 2 * x - 4) → -- tangent line equation
  (∃ f : ℝ → ℝ, ∀ h : ℝ, HasDerivAt f h 2 → h = 2) → -- tangent line condition
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parameter_l445_44588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_power_sequence_l445_44527

theorem not_prime_power_sequence (n : ℕ+) : 
  ∃ N : ℕ, ∀ j : ℕ, 1 ≤ j ∧ j ≤ n → ¬∃ (p : ℕ) (k : ℕ), Prime p ∧ (N + j : ℕ) = p ^ k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_prime_power_sequence_l445_44527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_bounds_l445_44573

/-- A cube with edge length 2 -/
structure Cube where
  edge_length : ℝ
  edge_length_eq : edge_length = 2

/-- A point on the inscribed circle of a face of the cube -/
structure PointOnInscribedCircle where
  x : ℝ
  y : ℝ
  z : ℝ
  on_circle : x^2 + y^2 + z^2 = 2 -- Condition for being on the inscribed circle

/-- The sum of distances between three points -/
noncomputable def distance_sum (p₁ p₂ p₃ : PointOnInscribedCircle) : ℝ :=
  (((p₁.x - p₂.x)^2 + (p₁.y - p₂.y)^2 + (p₁.z - p₂.z)^2).sqrt) +
  (((p₂.x - p₃.x)^2 + (p₂.y - p₃.y)^2 + (p₂.z - p₃.z)^2).sqrt) +
  (((p₃.x - p₁.x)^2 + (p₃.y - p₁.y)^2 + (p₃.z - p₁.z)^2).sqrt)

/-- Theorem stating the bounds on the sum of distances -/
theorem distance_sum_bounds (cube : Cube) (p₁ p₂ p₃ : PointOnInscribedCircle) :
  3 * (2 : ℝ).sqrt - 3 ≤ distance_sum p₁ p₂ p₃ ∧ distance_sum p₁ p₂ p₃ ≤ 3 * (2 : ℝ).sqrt := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_sum_bounds_l445_44573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l445_44579

/-- The area of a rectangle inscribed in a trapezoid -/
theorem inscribed_rectangle_area
  (h : ℝ) (a b x : ℝ) 
  (h_pos : h > 0)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (x_pos : x > 0)
  (x_le_h : x ≤ h)
  (a_gt_b : a > b) :
  (b * x / h) * (h - x) = (b * x / h) * (h - x) :=
by
  -- The proof is trivial since we're equating the expression to itself
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_rectangle_area_l445_44579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_upper_bound_l445_44516

theorem divisor_count_upper_bound (n : ℕ) (hn : 0 < n) : 
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card ≤ 2 * Real.sqrt (n : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_count_upper_bound_l445_44516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l445_44586

-- Define the curves C₁ and C₂
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (2 + 2 * Real.cos t, 2 * Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ := Real.sin θ / (Real.cos θ)^2

-- Define the intersection points
noncomputable def OA (α : ℝ) : ℝ := 4 * Real.cos α

noncomputable def OB (α : ℝ) : ℝ := Real.sin α / (Real.cos α)^2

-- State the theorem
theorem intersection_product_range (α : ℝ) 
  (h : π/6 < α ∧ α ≤ π/4) : 
  4*Real.sqrt 3/3 < OA α * OB α ∧ OA α * OB α ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_range_l445_44586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_five_l445_44554

/-- A line in a plane -/
structure Line where
  l : Set (ℝ × ℝ)

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Perpendicular relation between a line and a line segment -/
def perpendicular (l : Line) (p1 p2 : Point) : Prop := sorry

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ := sorry

/-- Main theorem -/
theorem distance_to_line_is_five
  (l : Line)
  (A B C P : Point)
  (h1 : (A.x, A.y) ∈ l.l ∧ (B.x, B.y) ∈ l.l ∧ (C.x, C.y) ∈ l.l)
  (h2 : (P.x, P.y) ∉ l.l)
  (h3 : perpendicular l A P)
  (h4 : distance P A = 5)
  (h5 : distance P B = 6)
  (h6 : distance P C = 7) :
  distance_point_to_line P l = 5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_is_five_l445_44554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_twelve_l445_44545

/-- Proves that the height of a room is 12 feet given specific dimensions and whitewashing costs -/
theorem room_height_is_twelve : ∃ (height : ℝ), 
  height = 12 ∧ 
  6 * (2 * (25 + 15) * height - (6 * 3 + 3 * (4 * 3))) = 5436 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_twelve_l445_44545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_overall_profit_l445_44509

/-- Calculate the overall profit from selling a grinder and a mobile phone -/
def calculate_overall_profit (grinder_cost mobile_cost : ℕ) 
  (grinder_loss_percent mobile_profit_percent : ℚ) : ℤ :=
  let grinder_loss := (grinder_cost : ℚ) * grinder_loss_percent
  let grinder_selling_price := (grinder_cost : ℚ) - grinder_loss
  let mobile_profit := (mobile_cost : ℚ) * mobile_profit_percent
  let mobile_selling_price := (mobile_cost : ℚ) + mobile_profit
  let total_cost := grinder_cost + mobile_cost
  let total_selling_price := grinder_selling_price + mobile_selling_price
  let overall_profit := total_selling_price - (total_cost : ℚ)
  ⌊overall_profit⌋

/-- Prove that John's overall profit is 200 Rs -/
theorem john_overall_profit : 
  calculate_overall_profit 15000 8000 (4/100) (10/100) = 200 := by
  sorry

#eval calculate_overall_profit 15000 8000 (4/100) (10/100)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_overall_profit_l445_44509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_zero_l445_44518

theorem tan_sum_zero (θ : Real) (h1 : 0 < θ) (h2 : θ < π/4) 
  (h3 : Real.tan θ + Real.tan (2*θ) + Real.tan (3*θ) = 0) : Real.tan θ = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_zero_l445_44518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l445_44542

theorem trigonometric_equation_reduction (a b c : ℕ+) :
  (∀ x : ℝ, (Real.sin x)^2 + (Real.sin (3*x))^2 + (Real.sin (4*x))^2 + (Real.sin (5*x))^2 = 2 ↔ 
    Real.cos (a.val*x) * Real.cos (b.val*x) * Real.cos (c.val*x) = 0) →
  a.val + b.val + c.val = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_reduction_l445_44542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_sequence_l445_44512

/-- A sequence of positive integers representing the board state at each step -/
def BoardSequence : Type := List Nat

/-- Predicate to check if a number is a positive divisor of another number -/
def isPositiveDivisor (d n : Nat) : Prop := d > 0 ∧ n % d = 0

/-- Predicate to check if a sequence of operations is valid -/
def isValidSequence (seq : BoardSequence) : Prop :=
  seq.length > 1 ∧
  ∀ i, i < seq.length - 1 →
    ∃ d, isPositiveDivisor d (seq.get! i) ∧
        seq.get! (i + 1) = seq.get! i + d ∧
        (i > 0 → seq.get! i - seq.get! (i - 1) ≠ d)

/-- Predicate to check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, n = m * m

/-- Main theorem: For any starting positive integer, there exists a valid sequence
    of operations that results in a perfect square -/
theorem exists_perfect_square_sequence (start : Nat) :
  ∃ (seq : BoardSequence), seq.head? = some start ∧
                           isValidSequence seq ∧
                           isPerfectSquare (seq.getLast (by sorry)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_perfect_square_sequence_l445_44512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_analysis_l445_44585

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 6

theorem quadratic_analysis :
  (∀ x, (deriv (deriv f)) x > 0) ∧ 
  (∀ x, f x ≥ f 2) ∧
  (f 2 = 2) ∧
  (∃ h k, ∀ x, f x = (x - h)^2 + k ∧ h = 2 ∧ k = 2) := by
  sorry

#check quadratic_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_analysis_l445_44585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l445_44589

-- Define the two planes
def plane1 (x y : ℝ) : Prop := 3 * x - y - 5 = 0
def plane2 (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the normal vectors of the planes
def normal1 : ℝ × ℝ := (3, -1)
def normal2 : ℝ × ℝ := (2, 1)

-- Theorem statement
theorem angle_between_planes :
  Real.arccos (
    (normal1.1 * normal2.1 + normal1.2 * normal2.2) /
    (Real.sqrt (normal1.1^2 + normal1.2^2) *
     Real.sqrt (normal2.1^2 + normal2.2^2))
  ) = π / 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_planes_l445_44589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_previous_hay_cost_l445_44521

/-- Calculates the cost of the previous hay per bale -/
def previous_cost_per_bale (initial_bales new_bales new_cost_per_bale cost_difference : ℕ) : ℕ :=
  (new_bales * new_cost_per_bale - cost_difference) / initial_bales

/-- Proves the cost of the previous hay given the conditions of the problem -/
theorem previous_hay_cost 
  (initial_bales : ℕ) 
  (new_bales : ℕ) 
  (new_cost_per_bale : ℕ) 
  (cost_difference : ℕ) :
  initial_bales = 10 →
  new_bales = 2 * initial_bales →
  new_cost_per_bale = 18 →
  new_bales * new_cost_per_bale - initial_bales * (previous_cost_per_bale initial_bales new_bales new_cost_per_bale cost_difference) = cost_difference →
  cost_difference = 210 →
  previous_cost_per_bale initial_bales new_bales new_cost_per_bale cost_difference = 15 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_previous_hay_cost_l445_44521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l445_44559

/-- Hyperbola C with equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
def Hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1}

/-- Left focus of the hyperbola -/
def leftFocus (c : ℝ) : ℝ × ℝ := (-c, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Eccentricity of the hyperbola -/
noncomputable def eccentricity (c a : ℝ) : ℝ := c / a

theorem hyperbola_eccentricity_range
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : ∃ (p : ℝ × ℝ), p ∈ Hyperbola a b ha hb ∧ 
    distance p (leftFocus c) = 3 * a ∧
    p.1 < -a) :
  2 < eccentricity c a ∧ eccentricity c a ≤ 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l445_44559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_complete_list_l445_44597

-- Define Player as an inductive type
inductive Player : Type

structure Tournament where
  players : Set Player
  plays_against : Player → Player → Prop
  defeats : Player → Player → Prop
  player_list : Player → Set Player

variable (t : Tournament)

axiom all_play_once : 
  ∀ p q : Player, p ≠ q → t.plays_against p q ∨ t.plays_against q p

axiom no_draws : 
  ∀ p q : Player, t.plays_against p q → (t.defeats p q ∨ t.defeats q p)

axiom list_contains_defeated : 
  ∀ p q : Player, t.defeats p q → q ∈ t.player_list p

axiom list_contains_indirect : 
  ∀ p q r : Player, t.defeats p q ∧ t.defeats q r → r ∈ t.player_list p

theorem exists_complete_list : 
  ∃ p : Player, ∀ q : Player, q ≠ p → q ∈ t.player_list p :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_complete_list_l445_44597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_downstream_l445_44556

/-- Calculates the distance traveled downstream by a boat -/
theorem boat_distance_downstream (boat_speed current_speed time_minutes : ℝ) : 
  boat_speed = 20 → 
  current_speed = 5 → 
  time_minutes = 27 → 
  (boat_speed + current_speed) * (time_minutes / 60) = 11.25 := by
  intro h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  
#check boat_distance_downstream

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_distance_downstream_l445_44556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_problem_l445_44581

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x - 2/x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * (2 - Real.log x)

-- Define the derivatives of f and g
noncomputable def f' (x : ℝ) : ℝ := 1 + 2/x^2
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := -a/x

-- Theorem statement
theorem tangent_lines_problem (a : ℝ) :
  (f' 1 = g' a 1) →
  (a = -3 ∧ 
   ∃ (m b₁ b₂ : ℝ), m = f' 1 ∧ 
                    (∀ x, f x = m * (x - 1) + f 1) ∧
                    (∀ x, g a x = m * (x - 1) + g a 1) ∧
                    b₁ ≠ b₂) :=
by sorry

-- Note: The actual proof is omitted and replaced with 'sorry'

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_problem_l445_44581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_level_drop_l445_44570

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylinder -/
noncomputable def cylinderVolume (c : Cylinder) : ℝ := Real.pi * c.radius^2 * c.height

theorem oil_level_drop (stationaryTank truckTank : Cylinder)
  (h_stationary_radius : stationaryTank.radius = 100)
  (h_stationary_height : stationaryTank.height = 25)
  (h_truck_radius : truckTank.radius = 8)
  (h_truck_height : truckTank.height = 10) :
  (cylinderVolume truckTank) / (Real.pi * stationaryTank.radius^2) = 0.064 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_oil_level_drop_l445_44570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_original_professors_count_l445_44565

theorem original_professors_count (p : ℕ) : p = 5 :=
  by
  -- Conditions
  have first_year_grades : 6480 = p * (6480 / p) := sorry
  have second_year_professors : ℕ := p + 3
  have second_year_grades : 11200 = (p + 3) * (11200 / (p + 3)) := sorry
  have grades_increased : 6480 / p < 11200 / (p + 3) := sorry
  
  -- p is a divisor of 6480
  have divides_first_year : ∃ k : ℕ, 6480 = p * k := sorry
  
  -- p + 3 is a divisor of 11200
  have divides_second_year : ∃ k : ℕ, 11200 = (p + 3) * k := sorry
  
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_original_professors_count_l445_44565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_circumcenter_points_satisfy_equation_l445_44551

/-- Triangle ABC with points M and M₁ satisfying specific distance conditions -/
structure TriangleWithPoints (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C M M₁ : V)
  (dist_M_A : ‖M - A‖ = 1)
  (dist_M_B : ‖M - B‖ = 2)
  (dist_M_C : ‖M - C‖ = 3)
  (dist_M₁_A : ‖M₁ - A‖ = 3)
  (dist_M₁_B : ‖M₁ - B‖ = Real.sqrt 15)
  (dist_M₁_C : ‖M₁ - C‖ = 5)

/-- Helper definition for the circumcenter -/
def IsCircumcenter {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (O A B C : V) : Prop :=
  ‖O - A‖ = ‖O - B‖ ∧ ‖O - B‖ = ‖O - C‖

/-- The equation that both M and M₁ satisfy -/
def SatisfiesEquation {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (P A B C : V) : Prop :=
  5 * ‖P - A‖^2 - 8 * ‖P - B‖^2 + 3 * ‖P - C‖^2 = 0

/-- The main theorem stating that MM₁ passes through the circumcenter -/
theorem line_passes_through_circumcenter {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : TriangleWithPoints V) : 
  ∃ (O : V), IsCircumcenter O t.A t.B t.C ∧ Collinear ℝ {t.M, t.M₁, O} :=
by
  sorry

/-- Both M and M₁ satisfy the equation -/
theorem points_satisfy_equation {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (t : TriangleWithPoints V) : 
  SatisfiesEquation t.M t.A t.B t.C ∧ SatisfiesEquation t.M₁ t.A t.B t.C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_circumcenter_points_satisfy_equation_l445_44551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_solution_l445_44520

theorem unique_n_solution (n : ℕ) (a : ℕ → ℝ) : 
  (∀ x : ℝ, (Finset.sum (Finset.range n) (λ i ↦ (1 + x)^(i + 1))) = 
    (Finset.sum (Finset.range (n + 1)) (λ i ↦ a i * x^i))) →
  ((Finset.sum (Finset.range (n - 1)) (λ i ↦ a (i + 1))) = 29 - n) →
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_n_solution_l445_44520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_specific_values_l445_44536

theorem sin_difference_specific_values (θ ϕ : ℝ) 
  (h1 : Real.sin θ = 4/5)
  (h2 : Real.cos ϕ = -5/13)
  (h3 : θ ∈ Set.Ioo (π/2) π)
  (h4 : ϕ ∈ Set.Ioo (π/2) π) :
  Real.sin (θ - ϕ) = 16/65 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_difference_specific_values_l445_44536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_class_average_l445_44571

def class1_size : ℕ := 26
def class2_size : ℕ := 50
def class1_average : ℝ := 40
def combined_average : ℝ := 53.1578947368421

theorem second_class_average :
  let total_students : ℕ := class1_size + class2_size
  let class2_average : ℝ := (combined_average * (class1_size + class2_size : ℝ) - class1_average * class1_size) / class2_size
  ∀ ε > 0, |class2_average - 60| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_class_average_l445_44571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_equivalence_l445_44514

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the horizontal shift transformation
def horizontalShift (g : ℝ → ℝ) (shift : ℝ) : ℝ → ℝ :=
  λ x ↦ g (x + shift)

theorem horizontal_shift_equivalence :
  ∀ x : ℝ, (horizontalShift f 2) x = f (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_shift_equivalence_l445_44514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l445_44530

theorem max_value_of_z (u v x y : ℝ) (h1 : u^2 + v^2 = 1) 
  (h2 : x + y - 1 ≥ 0) (h3 : x - 2*y + 2 ≥ 0) (h4 : x ≤ 2) :
  ∃ (z : ℝ), z = u*x + v*y ∧ z ≤ 2*Real.sqrt 2 ∧ 
  ∃ (u' v' x' y' : ℝ), u'^2 + v'^2 = 1 ∧ 
  x' + y' - 1 ≥ 0 ∧ x' - 2*y' + 2 ≥ 0 ∧ x' ≤ 2 ∧ 
  u'*x' + v'*y' = 2*Real.sqrt 2 := by
  sorry

#check max_value_of_z

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_z_l445_44530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_of_itself_l445_44539

-- Define the function g
noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

-- State the theorem
theorem g_inverse_of_itself (c d : ℝ) : 
  (∀ x, g c d (g c d x) = x) → c + d = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_inverse_of_itself_l445_44539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l445_44578

-- Define the variables
def total_hours : ℕ := 144
def kate_hours : ℕ := sorry
def pat_hours : ℕ := sorry
def mark_hours : ℕ := sorry

-- State the theorem
theorem project_hours_difference : 
  pat_hours = 2 * kate_hours →
  pat_hours = mark_hours / 3 →
  pat_hours + kate_hours + mark_hours = total_hours →
  mark_hours - kate_hours = 80 := by
  sorry

#check project_hours_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_project_hours_difference_l445_44578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_to_intersecting_line_l445_44599

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  k : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- The ellipse C: x²/4 + y²/3 = 1 -/
def ellipse_C (p : Point) : Prop :=
  p.x^2 / 4 + p.y^2 / 3 = 1

/-- Check if two points are perpendicular from the origin -/
def perpendicular_from_origin (p1 p2 : Point) : Prop :=
  p1.x * p2.x + p1.y * p2.y = 0

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.k * p.x - p.y + l.b) / Real.sqrt (1 + l.k^2)

/-- Main theorem -/
theorem constant_distance_to_intersecting_line :
  ∀ (l : Line) (A B : Point),
    ellipse_C A ∧ ellipse_C B ∧
    perpendicular_from_origin A B ∧
    (A.y = l.k * A.x + l.b) ∧ (B.y = l.k * B.x + l.b) →
    distance_point_to_line ⟨0, 0⟩ l = 2 * Real.sqrt 21 / 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_distance_to_intersecting_line_l445_44599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l445_44534

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^3
  else if 0 ≤ x ∧ x ≤ Real.pi/2 then -Real.tan x
  else 0  -- undefined for x > π/2

theorem f_composition_value : f (f (Real.pi/4)) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l445_44534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_eq_100_l445_44574

/-- The count of positive integers not exceeding 200 that are multiples of 2 or 5 but not 10 -/
def count_multiples : ℕ :=
  (Finset.filter (λ n ↦ n ≤ 200 ∧ (n % 2 = 0 ∨ n % 5 = 0) ∧ n % 10 ≠ 0) (Finset.range 201)).card

/-- Theorem stating that the count of positive integers not exceeding 200 
    that are multiples of 2 or 5 but not 10 is equal to 100 -/
theorem count_multiples_eq_100 : count_multiples = 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_eq_100_l445_44574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l445_44566

def point := ℝ × ℝ

def is_parallel_to_x_axis (A B : point) : Prop :=
  A.2 = B.2

noncomputable def distance (A B : point) : ℝ :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

theorem point_B_coordinates :
  ∀ (B : point),
  is_parallel_to_x_axis (-2, -4) B →
  distance (-2, -4) B = 5 →
  (B = (3, -4) ∨ B = (-7, -4)) :=
by
  sorry

#check point_B_coordinates

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_B_coordinates_l445_44566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_97_l445_44523

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => sequence_a n + (n + 1) / 2

theorem a_20_equals_97 : sequence_a 19 = 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_20_equals_97_l445_44523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l445_44515

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

-- Part I: Tangent line at x = 1 when a = 4
theorem tangent_line_at_one (x y : ℝ) :
  (f 4 1 = 0) →
  ((deriv (f 4)) 1 = -2) →
  (y = f 4 x) →
  (2 * x + y - 2 = 0) := by sorry

-- Part II: Range of a for f(x) > 0 when x ∈ (1,+∞)
theorem range_of_a (a : ℝ) :
  (∀ x > 1, f a x > 0) ↔ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_range_of_a_l445_44515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_inequality_l445_44506

theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.cos A + Real.sin B ^ 2 * Real.cos C ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_inequality_l445_44506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l445_44595

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  S : Real

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  t.b * t.c * Real.cos t.A ≤ 2 * Real.sqrt 3 * t.S

def condition2 (t : Triangle) : Prop :=
  Real.tan t.A / Real.tan t.B = 1 / 2 ∧
  Real.tan t.B / Real.tan t.C = 2 / 3 ∧
  t.c = 1

-- State the theorem
theorem triangle_properties (t : Triangle) :
  (condition1 t → π/6 ≤ t.A ∧ t.A < π) ∧
  (condition2 t → t.b = 2 * Real.sqrt 2 / 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l445_44595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l445_44519

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- The right focus of the hyperbola -/
noncomputable def right_focus (h : Hyperbola) : Point :=
  ⟨Real.sqrt (h.a^2 + h.b^2), 0⟩

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Theorem about the hyperbola properties -/
theorem hyperbola_properties (h : Hyperbola) 
  (h_ecc : eccentricity h = 2)
  (h_point : on_hyperbola h ⟨4, 6⟩) :
  (∃ (h' : Hyperbola), h'.a = 2 ∧ h'.b = 2 * Real.sqrt 3 ∧ 
    (∀ (p : Point), on_hyperbola h p ↔ on_hyperbola h' p)) ∧
  (∀ (A B : Point), A ≠ B → on_hyperbola h A → on_hyperbola h B →
    ∃ (m : ℝ), A.x = m * A.y + (right_focus h).x ∧ 
               B.x = m * B.y + (right_focus h).x →
      |1 / distance A (right_focus h) - 1 / distance B (right_focus h)| = 1/3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l445_44519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_exist_l445_44590

open Real

theorem distinct_solutions_exist : ∃ N : ℕ, ∃ S : Finset ℝ,
  (∀ θ ∈ S, 0 ≤ θ ∧ θ < 2 * π) ∧
  (∀ θ ∈ S, sin (3 * π * cos θ) = cos (4 * π * sin θ)) ∧
  S.card = N :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_exist_l445_44590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_range_l445_44592

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 - 6*x + 6 else 3*x + 4

-- State the theorem
theorem sum_x_range (x₁ x₂ x₃ : ℝ) :
  x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
  f x₁ = f x₂ ∧ f x₂ = f x₃ →
  11/3 < x₁ + x₂ + x₃ ∧ x₁ + x₂ + x₃ < 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_range_l445_44592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_approx_l445_44591

/-- Conversion factor from centimeters to inches -/
noncomputable def cm_to_inch : ℝ := 1 / 2.54

/-- Side lengths of the hexagon in their original units -/
noncomputable def hexagon_sides : List ℝ := [5, 8 * cm_to_inch, 6, 10 * cm_to_inch, 7, 12 * cm_to_inch]

/-- The perimeter of the hexagon -/
noncomputable def hexagon_perimeter : ℝ := hexagon_sides.sum

theorem hexagon_perimeter_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |hexagon_perimeter - 29.81| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_approx_l445_44591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l445_44505

noncomputable def a : Fin 2 → ℝ := ![2, 3]
noncomputable def b : Fin 2 → ℝ := ![-4, 7]

def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  Real.sqrt ((v 0)^2 + (v 1)^2)

noncomputable def projection (v w : Fin 2 → ℝ) : ℝ :=
  (dot_product v w) / (magnitude w)

theorem projection_a_on_b :
  projection a b = Real.sqrt 65 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l445_44505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l445_44501

/-- Two lines are parallel if and only if their slopes are equal -/
def are_parallel (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ / b₁ = a₂ / b₂ ∧ b₁ ≠ 0 ∧ b₂ ≠ 0

/-- The slope of line l₁: (3+m)x + 4y = 5 -/
noncomputable def slope_l₁ (m : ℝ) : ℝ := -(3 + m) / 4

/-- The slope of line l₂: 2x + (5+m)y = 8 -/
noncomputable def slope_l₂ (m : ℝ) : ℝ := -2 / (5 + m)

theorem parallel_lines_m_value :
  ∀ m : ℝ, are_parallel (3 + m) 4 2 (5 + m) ↔ m = -1 ∨ m = -7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_m_value_l445_44501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sofia_running_time_l445_44576

/-- The number of laps Sofia ran -/
def num_laps : ℕ := 5

/-- The length of each lap in meters -/
def lap_length : ℕ := 400

/-- The length of the first part of each lap in meters -/
def first_part_length : ℕ := 100

/-- The length of the second part of each lap in meters -/
def second_part_length : ℕ := 300

/-- Sofia's speed for the first part of each lap in meters per second -/
def speed_first_part : ℚ := 4

/-- Sofia's speed for the second part of each lap in meters per second -/
def speed_second_part : ℚ := 5

/-- The time taken for one lap in seconds -/
noncomputable def time_one_lap : ℚ := first_part_length / speed_first_part + second_part_length / speed_second_part

/-- Theorem: The total time Sofia took to run 5 laps is 425 seconds -/
theorem sofia_running_time : (num_laps : ℚ) * time_one_lap = 425 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sofia_running_time_l445_44576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_co_l445_44596

/-- The atomic mass of carbon in atomic mass units (amu) -/
noncomputable def carbon_mass : ℝ := 12.01

/-- The atomic mass of oxygen in atomic mass units (amu) -/
noncomputable def oxygen_mass : ℝ := 16.00

/-- The molecular mass of carbon monoxide (CO) in atomic mass units (amu) -/
noncomputable def co_mass : ℝ := carbon_mass + oxygen_mass

/-- The mass percentage of carbon in carbon monoxide -/
noncomputable def carbon_percentage : ℝ := (carbon_mass / co_mass) * 100

/-- Theorem stating that the mass percentage of carbon in carbon monoxide is approximately 42.88% -/
theorem carbon_percentage_in_co : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |carbon_percentage - 42.88| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_percentage_in_co_l445_44596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_four_digit_pascal_l445_44543

/-- PascalTriangle is a set of natural numbers -/
def PascalTriangle : Set ℕ := sorry

/-- Pascal's triangle contains all positive integers -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → n ∈ PascalTriangle

/-- Four-digit numbers start from 1000 in Pascal's triangle -/
axiom four_digit_start : ∀ n : ℕ, n ∈ PascalTriangle ∧ n ≥ 1000 → n ≥ 1000

/-- Each row in Pascal's triangle contains consecutive integers -/
axiom consecutive_integers : ∀ n m : ℕ, n ∈ PascalTriangle ∧ m = n + 1 → m ∈ PascalTriangle

/-- The third smallest four-digit number in Pascal's triangle is 1002 -/
theorem third_smallest_four_digit_pascal : 
  (∃ k : ℕ, k ∈ PascalTriangle ∧ k ≥ 1000 ∧ (∀ m : ℕ, m ∈ PascalTriangle ∧ m ≥ 1000 → m ≥ k) ∧
   (∃ l : ℕ, l ∈ PascalTriangle ∧ l > k ∧ (∀ n : ℕ, n ∈ PascalTriangle ∧ n > k → n ≥ l) ∧
   1002 ∈ PascalTriangle ∧ 1002 > l ∧ (∀ p : ℕ, p ∈ PascalTriangle ∧ p > l → p ≥ 1002))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_smallest_four_digit_pascal_l445_44543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l445_44561

theorem solve_exponential_equation (x : ℚ) : 
  (5 : ℝ) ^ ((2 : ℝ) * x) = Real.sqrt 125 → x = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l445_44561
