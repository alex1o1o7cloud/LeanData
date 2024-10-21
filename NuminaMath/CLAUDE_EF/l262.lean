import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l262_26255

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a point lies on the ellipse x^2 + 4y^2 = 4 -/
def onEllipse (p : Point) : Prop :=
  p.x^2 + 4 * p.y^2 = 4

/-- Checks if a triangle is right-angled at vertex C -/
def isRightAngled (t : Triangle) : Prop :=
  (t.A.x - t.C.x) * (t.B.x - t.C.x) + (t.A.y - t.C.y) * (t.B.y - t.C.y) = 0

/-- Checks if point C is the centroid of the triangle -/
def isCentroid (t : Triangle) : Prop :=
  t.C.x = (t.A.x + t.B.x) / 3 ∧ t.C.y = (t.A.y + t.B.y) / 3

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  abs ((t.B.x - t.A.x) * (t.C.y - t.A.y) - (t.C.x - t.A.x) * (t.B.y - t.A.y)) / 2

/-- Main theorem statement -/
theorem triangle_area_is_one (t : Triangle) :
  onEllipse t.A ∧ onEllipse t.B ∧ onEllipse t.C ∧
  isRightAngled t ∧
  isCentroid t →
  triangleArea t = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l262_26255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_720_pointed_stars_l262_26214

/-- The number of non-similar regular n-pointed stars -/
def nonSimilarStars (n : ℕ) : ℕ :=
  (Nat.totient n) / 2

/-- A regular n-pointed star is formed when the step size m and n are coprime -/
def isRegularStar (n m : ℕ) : Prop :=
  Nat.Coprime m n

theorem count_720_pointed_stars :
  nonSimilarStars 720 = 96 := by
  -- Proof goes here
  sorry

#eval nonSimilarStars 720  -- This will evaluate the function for n = 720

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_720_pointed_stars_l262_26214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_bill_calculation_l262_26228

-- Define the grocery items and their prices
def hamburger_price : Float := 5.00
def crackers_price : Float := 3.50
def vegetable_price : Float := 2.00
def vegetable_quantity : Nat := 4
def cheese_price : Float := 3.50
def chicken_price : Float := 6.50
def cereal_price : Float := 4.00
def wine_price : Float := 10.00
def cookies_price : Float := 3.00

-- Define discount rates
def discount_10_percent : Float := 0.10
def discount_5_percent : Float := 0.05
def discount_15_percent : Float := 0.15

-- Define tax rates
def food_tax_rate : Float := 0.06
def alcohol_tax_rate : Float := 0.09

-- Function to apply discount
def apply_discount (price : Float) (discount : Float) : Float :=
  price - (price * discount)

-- Theorem to prove
theorem grocery_bill_calculation :
  let discounted_hamburger := apply_discount hamburger_price discount_10_percent
  let discounted_crackers := apply_discount crackers_price discount_10_percent
  let discounted_vegetable := apply_discount vegetable_price discount_10_percent
  let discounted_cheese := apply_discount cheese_price discount_5_percent
  let discounted_chicken := apply_discount chicken_price discount_5_percent
  let discounted_wine := apply_discount wine_price discount_15_percent

  let total_discounted_price := discounted_hamburger + discounted_crackers + 
    (discounted_vegetable * vegetable_quantity.toFloat) + discounted_cheese + 
    discounted_chicken + discounted_wine + cereal_price + cookies_price

  let food_items_price := total_discounted_price - discounted_wine
  let food_tax := food_items_price * food_tax_rate
  let wine_tax := discounted_wine * alcohol_tax_rate

  let total_cost := total_discounted_price + food_tax + wine_tax

  (total_cost - 42.51).abs < 0.01 := by sorry

-- Note: We use an approximation check instead of exact equality due to floating-point arithmetic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grocery_bill_calculation_l262_26228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l262_26271

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ q : ℝ), a₁ > 0 ∧ q = 2 ∧ ∀ n, a n = a₁ * q ^ (n - 1)

theorem problem_statement (a : ℕ → ℝ) :
  geometric_sequence a →
  f (a 2 * a 4 * a 6 * a 8 * a 10) = 25 →
  f (a 2012) = 2011 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l262_26271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_lattice_shortest_path_l262_26230

/-- A point in the hexagonal lattice --/
structure HexPoint where
  x : ℤ
  y : ℤ

/-- The distance between two points in the hexagonal lattice --/
noncomputable def hex_distance (a b : HexPoint) : ℝ :=
  Real.sqrt (3 * (b.x - a.x)^2 + 3 * (b.x - a.x) * (b.y - a.y) + 3 * (b.y - a.y)^2)

/-- The length of the shortest path between two points in the hexagonal lattice --/
def shortest_path_length (a b : HexPoint) : ℤ :=
  (b.x - a.x).natAbs + (b.y - a.y).natAbs + ((b.x - a.x) - (b.y - a.y)).natAbs

/-- The theorem to be proved --/
theorem hexagonal_lattice_shortest_path (a b : HexPoint) :
  ∃ (direction : HexPoint), 
    (hex_distance a direction ≥ hex_distance a b / 2) ∧
    (hex_distance a direction = hex_distance a b / 2 → shortest_path_length a b = 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_lattice_shortest_path_l262_26230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_arc_angle_l262_26289

/-- Represents a circle in a plane. -/
structure Circle where
  -- Add necessary fields here, e.g., center : Point, radius : ℝ

/-- Represents a point in a plane. -/
structure Point where
  -- Add necessary fields here, e.g., x : ℝ, y : ℝ

/-- Predicate that checks if a chord divides the circumference in a given ratio. -/
def ChordDivideCircumference (circle : Circle) (A B : Point) (m n : ℕ) : Prop :=
  sorry -- Define the predicate here

/-- Computes the arc angle for a given chord in a circle. -/
noncomputable def ArcAngle (circle : Circle) (A B : Point) : ℝ :=
  sorry -- Define the function here

/-- Theorem stating the relationship between chord division and arc angle. -/
theorem chord_arc_angle (circle : Circle) (A B : Point) (h : ChordDivideCircumference circle A B 1 5) :
  ArcAngle circle A B = 30 ∨ ArcAngle circle A B = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_arc_angle_l262_26289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_month_seashells_l262_26229

def seashells_in_jar (n : ℕ) : ℕ := 50 + 20 * n

def total_seashells (weeks : ℕ) : ℕ :=
  (List.range weeks).map seashells_in_jar |>.sum

theorem month_seashells :
  total_seashells 4 = 320 := by
  -- Proof steps would go here
  sorry

#eval total_seashells 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_month_seashells_l262_26229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l262_26283

/-- A cubic function with specific properties -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 - 6*x + b

/-- The conditions given in the problem -/
class ProblemConditions where
  a : ℝ
  b : ℝ
  f_zero_eq_one : f a b 0 = 1
  slope_at_one : HasDerivAt (f a b) (-6) 1

/-- The main theorem to prove -/
theorem cubic_function_properties [ProblemConditions] :
  (∀ x, f (-3/2) 1 x = x^3 - (3/2)*x^2 - 6*x + 1) ∧
  (∀ m, (∀ x ∈ Set.Ioo (-2) 2, f (-3/2) 1 x ≤ |2*m - 1|) → (m ≥ 11/4 ∨ m ≤ -7/4)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_properties_l262_26283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l262_26236

noncomputable def log2_3 : ℝ := Real.log 3 / Real.log 2

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ ∈ Set.Ici 1 ∧ (log2_3 ^ x₀ > 1)) ↔
  (∀ x : ℝ, x ∈ Set.Ici 1 → log2_3 ^ x ≤ 1) := by
  sorry

#check negation_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_equivalence_l262_26236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l262_26211

/-- The length of a bridge in kilometers -/
noncomputable def bridge_length (walking_speed : ℝ) (crossing_time_minutes : ℝ) : ℝ :=
  walking_speed * (crossing_time_minutes / 60)

/-- Theorem: A man walking at 10 km/hr crossing a bridge in 15 minutes implies the bridge is 2.5 km long -/
theorem bridge_length_calculation : 
  bridge_length 10 15 = 2.5 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- Check that 10 * (15 / 60) = 2.5
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l262_26211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_percentage_liquid_x_l262_26263

/-- Represents the composition of a solution -/
structure SolutionComposition where
  liquidX : Float
  water : Float
  liquidZ : Float

/-- Represents the amount of each component in a solution -/
structure SolutionAmount where
  liquidX : Float
  water : Float
  liquidZ : Float

/-- The initial composition of solution Y -/
def initialComposition : SolutionComposition :=
  { liquidX := 0.40, water := 0.45, liquidZ := 0.15 }

/-- The initial amount of solution Y in kg -/
def initialAmount : Float := 18

/-- The amount of water that evaporates in kg -/
def evaporatedWater : Float := 6

/-- The amount of solution Y added after evaporation in kg -/
def addedSolutionY : Float := 5

/-- Calculates the amount of each component in a solution given its composition and total amount -/
def calculateAmounts (comp : SolutionComposition) (total : Float) : SolutionAmount :=
  { liquidX := comp.liquidX * total,
    water := comp.water * total,
    liquidZ := comp.liquidZ * total }

/-- Calculates the percentage of liquid X in a solution -/
def percentageLiquidX (amounts : SolutionAmount) : Float :=
  amounts.liquidX / (amounts.liquidX + amounts.water + amounts.liquidZ) * 100

theorem final_percentage_liquid_x :
  let initialAmounts := calculateAmounts initialComposition initialAmount
  let afterEvaporation : SolutionAmount := 
    { liquidX := initialAmounts.liquidX,
      water := initialAmounts.water - evaporatedWater,
      liquidZ := initialAmounts.liquidZ }
  let addedAmounts := calculateAmounts initialComposition addedSolutionY
  let finalAmounts : SolutionAmount := 
    { liquidX := afterEvaporation.liquidX + addedAmounts.liquidX,
      water := afterEvaporation.water + addedAmounts.water,
      liquidZ := afterEvaporation.liquidZ + addedAmounts.liquidZ }
  (percentageLiquidX finalAmounts - 54.12).abs < 0.01 := by
  sorry

#eval let initialAmounts := calculateAmounts initialComposition initialAmount
      let afterEvaporation : SolutionAmount := 
        { liquidX := initialAmounts.liquidX,
          water := initialAmounts.water - evaporatedWater,
          liquidZ := initialAmounts.liquidZ }
      let addedAmounts := calculateAmounts initialComposition addedSolutionY
      let finalAmounts : SolutionAmount := 
        { liquidX := afterEvaporation.liquidX + addedAmounts.liquidX,
          water := afterEvaporation.water + addedAmounts.water,
          liquidZ := afterEvaporation.liquidZ + addedAmounts.liquidZ }
      percentageLiquidX finalAmounts

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_percentage_liquid_x_l262_26263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_seven_l262_26206

theorem not_divisible_by_seven : ¬ (7 ∣ (2^2023 + 3^2023)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_divisible_by_seven_l262_26206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_comparison_l262_26223

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem factorial_comparison :
  (factorial (factorial 100)) < (factorial 99)^(factorial 100) * (factorial 100)^(factorial 99) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_comparison_l262_26223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_new_alloy_l262_26201

/-- Represents an alloy with its chromium percentage and weight -/
structure Alloy where
  chromiumPercentage : ℝ
  weight : ℝ

/-- Calculates the percentage of chromium in a new alloy formed by combining two alloys -/
noncomputable def newAlloyChromiumPercentage (alloy1 alloy2 : Alloy) : ℝ :=
  ((alloy1.chromiumPercentage * alloy1.weight + alloy2.chromiumPercentage * alloy2.weight) /
   (alloy1.weight + alloy2.weight)) * 100

/-- Theorem stating that combining two specific alloys results in a new alloy with 7.2% chromium -/
theorem chromium_percentage_in_new_alloy :
  let alloy1 : Alloy := { chromiumPercentage := 10, weight := 15 }
  let alloy2 : Alloy := { chromiumPercentage := 6, weight := 35 }
  newAlloyChromiumPercentage alloy1 alloy2 = 7.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chromium_percentage_in_new_alloy_l262_26201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_calculation_l262_26298

/-- Given a set of observations with an incorrect mean due to a misrecorded value,
    calculate the corrected mean after fixing the error. -/
theorem corrected_mean_calculation
  (n : ℕ)
  (original_mean : ℚ)
  (incorrect_value correct_value : ℚ)
  (h_n : n = 50)
  (h_mean : original_mean = 36)
  (h_incorrect : incorrect_value = 21)
  (h_correct : correct_value = 48) :
  (n * original_mean + (correct_value - incorrect_value)) / n = 36.54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_corrected_mean_calculation_l262_26298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_4500_nearest_integer_l262_26222

-- Define the logarithm base 5
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- Theorem statement
theorem log5_4500_nearest_integer :
  Int.floor (log5 4500 + 0.5) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log5_4500_nearest_integer_l262_26222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l262_26242

/-- The y-coordinate of the third vertex of an equilateral triangle -/
noncomputable def third_vertex_y_coordinate (x1 y1 x2 y2 : ℝ) : ℝ :=
  y1 + (Real.sqrt 3) * (x2 - x1) / 2

/-- Theorem: The y-coordinate of the third vertex of an equilateral triangle
    with two vertices at (2,3) and (10,3) in the first quadrant -/
theorem equilateral_triangle_third_vertex :
  third_vertex_y_coordinate 2 3 10 3 = 3 + 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_third_vertex_l262_26242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_n_digit_numbers_appear_l262_26275

/-- A sequence of real numbers -/
def IncreasingUnboundedSequence (a : ℕ → ℝ) : Prop :=
  (∀ i, a i < a (i + 1)) ∧ (∀ M, ∃ i, a i > M)

/-- The condition that consecutive terms differ by less than 2020 -/
def BoundedDifference (a : ℕ → ℝ) : Prop :=
  ∀ i, a (i + 1) < a i + 2020

/-- The sequence of digits formed by the floor of absolute values of f(a_i) -/
noncomputable def DigitSequence (f : ℝ → ℝ) (a : ℕ → ℝ) : ℕ → Fin 10 :=
  sorry

/-- A number formed by n consecutive digits starting from index i -/
noncomputable def NDigitNumber (s : ℕ → Fin 10) (n i : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem all_n_digit_numbers_appear
  (f : ℝ → ℝ)
  (hf : ∃ p : Polynomial ℝ, (∀ x, f x = p.eval x) ∧ p ≠ Polynomial.C 0)
  (a : ℕ → ℝ)
  (ha : IncreasingUnboundedSequence a ∧ BoundedDifference a)
  (n : ℕ) :
  ∀ m : ℕ, ∃ k : ℕ, NDigitNumber (DigitSequence f a) n (n * (k - 1) + 1) = m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_n_digit_numbers_appear_l262_26275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l262_26215

noncomputable def f (x : ℝ) : ℝ := Real.tan (x + Real.pi/3)

-- State the theorem
theorem f_symmetry : 
  ∀ (x : ℝ), f (-Real.pi/3 + (x - (-Real.pi/3))) = -f (-Real.pi/3 - (x - (-Real.pi/3))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_symmetry_l262_26215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersection_l262_26276

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 - 6*x + 8) / (x^2 - 6*x + 9)

-- Theorem statement
theorem asymptote_intersection :
  ∃ (x y : ℝ), x = 3 ∧ y = 1 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ t : ℝ, 0 < |t - x| ∧ |t - x| < δ → |f t - y| < ε) ∧
  (∀ ε > 0, ∃ M : ℝ, ∀ t : ℝ, |t| > M → |f t - y| < ε) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersection_l262_26276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_first_20kg_l262_26208

-- Define the cost function for apples
noncomputable def apple_cost (l q : ℝ) (kg : ℝ) : ℝ :=
  if kg ≤ 30 then l * kg
  else l * 30 + q * (kg - 30)

-- Define the theorem
theorem apple_cost_first_20kg
  (l q : ℝ)
  (h1 : apple_cost l q 33 = 168)
  (h2 : apple_cost l q 36 = 186) :
  apple_cost l q 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_cost_first_20kg_l262_26208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unreflected_area_formula_l262_26238

/-- A trapezoid with segments reflected inward symmetrically -/
structure ReflectedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The total area of the reflected segments -/
  reflected_area : ℝ

/-- The area of the figure consisting of points that do not belong to any reflected segment -/
def unreflected_area (t : ReflectedTrapezoid) : ℝ :=
  t.area - t.reflected_area

/-- Theorem stating the area of the unreflected region in the trapezoid -/
theorem unreflected_area_formula (t : ReflectedTrapezoid) :
  unreflected_area t = 20 + 4 * Real.sqrt 55 - 64 * Real.arcsin (Real.sqrt 5 / 4) := by
  sorry

#check unreflected_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unreflected_area_formula_l262_26238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l262_26240

/-- Golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The golden section point divides a segment into two parts with the ratio of φ : 1 -/
def isGoldenSectionPoint (a b p : ℝ) : Prop :=
  (p - a) / (b - p) = φ - 1 ∨ (b - p) / (p - a) = φ - 1

theorem golden_section_length (a b p : ℝ) (h1 : b - a = 2) 
  (h2 : isGoldenSectionPoint a b p) (h3 : p - a > b - p) : 
  p - a = Real.sqrt 5 - 1 := by
  sorry

#check golden_section_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l262_26240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unbounded_f_max_value_l262_26252

open Real

-- Define the expression as a function of θ
noncomputable def f (a b c θ : ℝ) : ℝ := a * cos θ + b * sin θ + c * tan θ

-- Statement 1: If c ≠ 0, then f is unbounded
theorem f_unbounded (a b c : ℝ) (hc : c ≠ 0) :
  ∀ M : ℝ, ∃ θ : ℝ, f a b c θ > M := by
  sorry

-- Statement 2: If c = 0, then the maximum value of f is √(a² + b²)
theorem f_max_value (a b : ℝ) :
  (⨆ θ : ℝ, f a b 0 θ) = sqrt (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_unbounded_f_max_value_l262_26252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_at_minus_one_l262_26282

/-- A line in 3D space parameterized by t -/
structure ParametricLine where
  point_at : ℝ → Fin 3 → ℝ

/-- Given conditions for the parameterized line -/
noncomputable def given_line : ParametricLine where
  point_at := λ t i =>
    if t = 0 then
      if i = 0 then 2 else if i = 1 then 6 else 16
    else if t = 1 then
      if i = 0 then 1 else if i = 1 then 1 else 8
    else
      0 -- placeholder for other t values

/-- Theorem stating that the vector at t = -1 is (3, 11, 24) -/
theorem vector_at_minus_one (line : ParametricLine) 
  (h0 : ∀ i, line.point_at 0 i = given_line.point_at 0 i)
  (h1 : ∀ i, line.point_at 1 i = given_line.point_at 1 i) :
  ∀ i, line.point_at (-1) i = 
    if i = 0 then 3 else if i = 1 then 11 else 24 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_at_minus_one_l262_26282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_1234_l262_26265

def sequenceA : ℕ → ℕ
  | 0 => 1
  | 1 => 9
  | 2 => 8
  | 3 => 1
  | n + 4 => (sequenceA n + sequenceA (n + 1) + sequenceA (n + 2) + sequenceA (n + 3)) % 10

theorem no_consecutive_1234 : ¬ ∃ k : ℕ, 
  sequenceA k = 1 ∧ 
  sequenceA (k + 1) = 2 ∧ 
  sequenceA (k + 2) = 3 ∧ 
  sequenceA (k + 3) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_consecutive_1234_l262_26265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_visitors_count_l262_26291

/-- Represents a friend of Harry --/
inductive Friend
| Liam
| Noah
| Olivia
| Emma

/-- The number of days in the period we're considering --/
def period : ℕ := 360

/-- The visiting frequency of each friend --/
def visit_frequency (f : Friend) : ℕ :=
  match f with
  | Friend.Liam => 6
  | Friend.Noah => 8
  | Friend.Olivia => 10
  | Friend.Emma => 12

/-- Checks if a friend visits on a given day --/
def visits_on_day (f : Friend) (day : ℕ) : Bool :=
  (day - 1) % visit_frequency f = 0

/-- Counts the number of friends visiting on a given day --/
def visitors_count (day : ℕ) : ℕ :=
  (List.filter (fun f => visits_on_day f day) [Friend.Liam, Friend.Noah, Friend.Olivia, Friend.Emma]).length

/-- The main theorem to prove --/
theorem exactly_three_visitors_count : 
  (List.filter (fun day => visitors_count day = 3) (List.range period)).length = 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_three_visitors_count_l262_26291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l262_26281

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The parabola y^2 = 4x -/
def onParabola (p : Point) : Prop :=
  p.y^2 = 4 * p.x

theorem min_distance_sum :
  ∃ (m : ℝ), m = 5 ∧ ∀ (p : Point), onParabola p →
    distance p ⟨1, 0⟩ + distance p ⟨4, 2⟩ ≥ m := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l262_26281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l262_26232

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - (3/2) * x^2 + a * x + 4

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x^2 - 3*x + a

theorem monotonic_decreasing_interval (a : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 4, (f_derivative a) x ≤ 0) ∧ 
  (f_derivative a) (-1) = 0 ∧ 
  (f_derivative a) 4 = 0 → 
  a = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_l262_26232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l262_26273

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α - Real.cos α = Real.sqrt 2) : Real.sin (2 * α) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l262_26273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l262_26278

theorem right_triangle_side_length 
  (Q : ℝ) 
  (QP QR : ℝ) 
  (h1 : 0 < QP) 
  (h2 : 0 < QR) 
  (h3 : Real.cos Q = 0.6) 
  (h4 : QP = 18) 
  (h5 : Real.cos Q = QP / QR) : QR = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_side_length_l262_26278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_greater_than_e_squared_l262_26207

-- Define the function f(x) = ln x - kx
noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.log x - k * x

-- State the theorem
theorem zeros_product_greater_than_e_squared (k : ℝ) (x₁ x₂ : ℝ) 
  (hk : k > 0) 
  (hx₁ : x₁ > 0) 
  (hx₂ : x₂ > 0) 
  (hx_distinct : x₁ ≠ x₂) 
  (hf₁ : f k x₁ = 0) 
  (hf₂ : f k x₂ = 0) : 
  x₁ * x₂ > Real.exp 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_product_greater_than_e_squared_l262_26207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l262_26269

theorem polynomial_divisibility (n : ℕ) :
  let Q : Polynomial ℤ := (X + 1)^n - X^n - 1
  let P : Polynomial ℤ := X^2 + X + 1
  (∃ k : Polynomial ℤ, Q = k * P) ↔ (n % 6 = 1 ∨ n % 6 = 5) ∧
  (∃ k : Polynomial ℤ, Q = k * P^2) ↔ n % 6 = 1 ∧
  (∃ k : Polynomial ℤ, Q = k * P^3) ↔ n = 1 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_divisibility_l262_26269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l262_26216

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x)) + x + 1

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f a + f (a + 1) > 2) ↔ (-1/2 < a ∧ a < 0) :=
by
  sorry -- Skip the proof for now

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l262_26216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_l262_26246

theorem board_numbers (a b c d : ℕ) 
  (h : ∃ (s₁ s₂ s₃ s₄ s₅ s₆ : ℕ → ℕ → ℕ), 
    (Multiset.ofList [s₁, s₂, s₃, s₄, s₅, s₆] = Multiset.ofList [(·+·), (·+·), (·+·), (·+·), (·+·), (·+·)]) ∧ 
    (Multiset.ofList [s₁ a b, s₂ b c, s₃ c d, s₄ d a, s₅ a c, s₆ b d] = Multiset.ofList [23, 23, 23, 34, 34, 34])) :
  (a + b + c + d = 57) ∧ (min a (min b (min c d)) = 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_board_numbers_l262_26246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_drain_rate_proof_l262_26279

/-- The rate at which water is leaving a tank -/
noncomputable def water_drain_rate (total_volume : ℝ) (drain_time : ℝ) : ℝ :=
  total_volume / drain_time

theorem water_drain_rate_proof (total_volume drain_time : ℝ) 
  (h1 : total_volume = 300)
  (h2 : drain_time = 25) :
  water_drain_rate total_volume drain_time = 12 := by
  unfold water_drain_rate
  rw [h1, h2]
  norm_num

#check water_drain_rate_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_drain_rate_proof_l262_26279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l262_26290

noncomputable section

/-- The efficiency of worker q -/
def q_efficiency : ℝ := 1

/-- The efficiency of worker p -/
def p_efficiency : ℝ := 1.4 * q_efficiency

/-- The efficiency of worker r -/
def r_efficiency : ℝ := 0.7 * q_efficiency

/-- The time taken by p to complete the work alone -/
def p_time : ℝ := 24

/-- The total amount of work -/
def total_work : ℝ := p_efficiency * p_time

/-- The combined efficiency of p, q, and r -/
def combined_efficiency : ℝ := p_efficiency + q_efficiency + r_efficiency

/-- The time taken by p, q, and r working together -/
def combined_time : ℝ := total_work / combined_efficiency

theorem work_completion_time : 
  ∃ ε > 0, |combined_time - 10.84| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l262_26290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_theorem_l262_26249

-- Define the circles and line
def C1 (x y : ℝ) := (x + 2)^2 + y^2 = 1
def C2 (x y : ℝ) := (x + 1)^2 + y^2 = 1
def C3 (x y : ℝ) := (x - 1)^2 + y^2 = 16
def C4 (x y : ℝ) := (x - 2)^2 + y^2 = 4
def l (x : ℝ) := x = 2

-- Define the locus of centers for each condition
def locusA (x y : ℝ) := ∃ r : ℝ, (x - 2)^2 + y^2 = (r + 2)^2 ∧ (x + 2)^2 + y^2 = (r + 1)^2
def locusB (x y : ℝ) := ∃ r : ℝ, (x + 1)^2 + y^2 = (r + 1)^2 ∧ (x - 1)^2 + y^2 = (4 - r)^2
def locusC (x y : ℝ) := ∃ r : ℝ, C1 (-2) 0 ∧ (x - 2)^2 = r^2
def locusD (x y : ℝ) := ∃ r : ℝ, (x + 2)^2 + y^2 = (r + 1)^2 ∧ (x + 1)^2 + y^2 = (r + 1)^2

-- Theorem statements
theorem locus_theorem :
  (∃ a b : ℝ, ∀ x y : ℝ, locusA x y ↔ (x/a)^2 - (y/b)^2 = 1) ∧
  (∃ a b : ℝ, ∀ x y : ℝ, locusB x y ↔ (x/a)^2 + (y/b)^2 = 1) ∧
  (∃ a p : ℝ, ∀ x y : ℝ, locusC x y ↔ y^2 = 4*p*(x - a)) ∧
  ¬(∃ m b : ℝ, ∀ x y : ℝ, locusD x y ↔ y = m*x + b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_theorem_l262_26249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_correct_answers_l262_26254

/-- Represents the score for a multiple choice examination. -/
structure ExamScore where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  unanswered_points : ℤ
  total_score : ℤ

/-- Represents the number of questions answered in each category. -/
structure AnswerCounts where
  correct : ℕ
  incorrect : ℕ
  unanswered : ℕ

/-- Calculates the score based on the exam rules and answer counts. -/
def calculateScore (exam : ExamScore) (answers : AnswerCounts) : ℤ :=
  exam.correct_points * (answers.correct : ℤ) +
  exam.incorrect_points * (answers.incorrect : ℤ) +
  exam.unanswered_points * (answers.unanswered : ℤ)

/-- Theorem stating the maximum number of correct answers for John's exam. -/
theorem max_correct_answers (exam : ExamScore) (answers : AnswerCounts) :
  exam.total_questions = 20 ∧
  exam.correct_points = 6 ∧
  exam.incorrect_points = -2 ∧
  exam.unanswered_points = -1 ∧
  exam.total_score = 52 ∧
  answers.correct + answers.incorrect + answers.unanswered = exam.total_questions ∧
  calculateScore exam answers = exam.total_score →
  answers.correct ≤ 11 ∧ 
  ∃ (max_answers : AnswerCounts), 
    max_answers.correct = 11 ∧
    max_answers.correct + max_answers.incorrect + max_answers.unanswered = exam.total_questions ∧
    calculateScore exam max_answers = exam.total_score :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_correct_answers_l262_26254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_arctan_l262_26286

-- Define the bounds of the region
noncomputable def lower_bound : ℝ := 0
noncomputable def upper_bound : ℝ := Real.sqrt 3

-- Define the functions that bound the region
noncomputable def f (x : ℝ) : ℝ := Real.arctan x
def g (_ : ℝ) : ℝ := 0

-- State the theorem
theorem area_bounded_by_arctan :
  ∫ x in lower_bound..upper_bound, f x - g x = Real.pi / Real.sqrt 3 - Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_bounded_by_arctan_l262_26286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l262_26274

open Real Set

-- Define the function f(x) = x ln x
noncomputable def f (x : ℝ) : ℝ := x * log x

-- State the theorem
theorem monotonic_decreasing_interval_f :
  ∀ x ∈ Ioo (0 : ℝ) (5 : ℝ),
    StrictMonoOn f (Ioo 0 (exp (-1))) ∧
    ¬StrictMonoOn f (Ioo (exp (-1)) 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_decreasing_interval_f_l262_26274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_distances_l262_26203

noncomputable def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 - 1)^2 = 5}

noncomputable def line_L : Set (ℝ × ℝ) :=
  {p | p.1 + 2*p.2 + 4 = 0}

noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 + 2*p.2 + 4| / Real.sqrt 5

theorem circle_and_line_distances :
  ∀ p, p ∈ circle_C →
    (distance_to_line p ≤ (12/5) * Real.sqrt 5) ∧
    (distance_to_line p ≥ (2/5) * Real.sqrt 5) ∧
    (∃ p1 p2, p1 ∈ circle_C ∧ p2 ∈ circle_C ∧
      distance_to_line p1 = (12/5) * Real.sqrt 5 ∧
      distance_to_line p2 = (2/5) * Real.sqrt 5) := by
  sorry

#check circle_and_line_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_and_line_distances_l262_26203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_b_value_l262_26227

noncomputable def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 1/2

theorem two_zeros_implies_b_value (b : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f b x = 0 ∧ f b y = 0 ∧
  (∀ z : ℝ, f b z = 0 → z = x ∨ z = y)) →
  b = 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_b_value_l262_26227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_divisor_conditions_l262_26243

def n : ℕ := 2^6 * 3^5 * 5^4

def num_divisors (m : ℕ) : ℕ :=
  (Nat.factors m).length + 1

theorem unique_number_with_divisor_conditions :
  (num_divisors (n / 2) = num_divisors n - 30) ∧
  (num_divisors (n / 3) = num_divisors n - 35) ∧
  (num_divisors (n / 5) = num_divisors n - 42) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_with_divisor_conditions_l262_26243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_φ_satisfies_equation_l262_26233

/-- Given a function f and a constant lambda, φ is defined as the solution to the integral equation. -/
noncomputable def φ (f : ℝ → ℝ) (lambda : ℝ) (x : ℝ) : ℝ :=
  f x / (1 - lambda^2) + (lambda / (1 - lambda^2)) * Real.sqrt (2 / Real.pi) *
    ∫ t in Set.Ioi 0, f t * Real.cos (x * t)

/-- The integral equation that φ should satisfy. -/
def integral_equation (f : ℝ → ℝ) (lambda : ℝ) (φ : ℝ → ℝ) : Prop :=
  ∀ x, φ x = f x + lambda * Real.sqrt (2 / Real.pi) *
    ∫ t in Set.Ioi 0, φ t * Real.cos (x * t)

/-- Theorem stating that the defined φ satisfies the integral equation. -/
theorem φ_satisfies_equation (f : ℝ → ℝ) (lambda : ℝ) :
  integral_equation f lambda (φ f lambda) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_φ_satisfies_equation_l262_26233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_transitive_perpendicular_plane_parallel_not_always_perpendicular_implies_parallel_not_always_parallel_plane_perpendicular_implies_perpendicular_l262_26237

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perp : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)
variable (parallel_plane : Line → Plane → Prop)

-- State the theorems to be proved
theorem perpendicular_parallel_transitive (a b : Line) (α : Plane) :
  parallel a b → perp_plane a α → perp_plane b α := by sorry

theorem perpendicular_plane_parallel (a b : Line) (α : Plane) :
  perp_plane a α → perp_plane b α → parallel a b := by sorry

theorem not_always_perpendicular_implies_parallel (a b : Line) (α : Plane) :
  ¬ (perp_plane a α → perp a b → parallel_plane b α) := by sorry

theorem not_always_parallel_plane_perpendicular_implies_perpendicular (a b : Line) (α : Plane) :
  ¬ (parallel_plane a α → perp a b → perp_plane b α) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_parallel_transitive_perpendicular_plane_parallel_not_always_perpendicular_implies_parallel_not_always_parallel_plane_perpendicular_implies_perpendicular_l262_26237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_diameter_endpoints_l262_26277

/-- The area of a circle with diameter endpoints at (-1, 4) and (3, 11) is 65π/4 -/
theorem circle_area_from_diameter_endpoints : 
  let C : ℝ × ℝ := (-1, 4)
  let D : ℝ × ℝ := (3, 11)
  let diameter_squared := (D.1 - C.1)^2 + (D.2 - C.2)^2
  let radius := Real.sqrt diameter_squared / 2
  let area := π * radius^2
  area = 65 * π / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_from_diameter_endpoints_l262_26277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_shortest_chord_l262_26287

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y - 3 = 0

-- Define the line
def line_eq (k x y : ℝ) : Prop := k*x - y - 4*k + 2 = 0

-- Theorem statement
theorem circle_line_intersection_and_shortest_chord :
  (∀ k : ℝ, ∃! (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧
    line_eq k x₁ y₁ ∧ line_eq k x₂ y₂) ∧
  (∃! k : ℝ, k = 2 ∧
    ∀ k' : ℝ, k' ≠ k →
      ∀ x₁ y₁ x₂ y₂ : ℝ,
        circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq k x₁ y₁ ∧ line_eq k x₂ y₂ →
        circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq k' x₁ y₁ ∧ line_eq k' x₂ y₂ →
        (x₁ - x₂)^2 + (y₁ - y₂)^2 < (x₁ - x₂)^2 + (y₁ - y₂)^2) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ line_eq 2 x₁ y₁ ∧ line_eq 2 x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 44) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_and_shortest_chord_l262_26287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_pentagon_square_measure_l262_26218

/-- The measure of the exterior angle between a regular pentagon and a square sharing a common side -/
def exterior_angle_pentagon_square : ℝ := 162

/-- A regular pentagon -/
structure RegularPentagon where
  -- We'll add a field to represent the side length
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A square -/
structure Square where
  -- We'll add a field to represent the side length
  side_length : ℝ
  side_length_pos : side_length > 0

/-- Predicate to check if a pentagon and a square share a common side -/
def has_common_side (pentagon : RegularPentagon) (square : Square) : Prop :=
  pentagon.side_length = square.side_length

/-- Theorem: The exterior angle between a regular pentagon and a square sharing a common side is 162° -/
theorem exterior_angle_pentagon_square_measure
  (pentagon : RegularPentagon)
  (square : Square)
  (share_side : has_common_side pentagon square) :
  exterior_angle_pentagon_square = 162 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exterior_angle_pentagon_square_measure_l262_26218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l262_26235

/-- Calculates the average speed of a journey with two segments -/
noncomputable def average_speed (distance_qb : ℝ) (distance_bc : ℝ) (speed_qb : ℝ) (speed_bc : ℝ) : ℝ :=
  (distance_qb + distance_bc) / (distance_qb / speed_qb + distance_bc / speed_bc)

theorem journey_average_speed :
  ∀ (d : ℝ),
    d > 0 →
    average_speed (2 * d) d 60 20 = 36 := by
  intro d h_d_pos
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_speed_l262_26235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l262_26210

theorem cos_double_angle_on_unit_circle (α : ℝ) (x : ℝ) :
  x^2 + (Real.sqrt 3 / 2)^2 = 1 →
  Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_on_unit_circle_l262_26210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_theorem_l262_26256

noncomputable def ζ : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem smallest_value_theorem (a b c d : ℕ) (ha : a > b) (hb : b > c) (hc : c > d) 
  (hd : d > 0) (hsum : Complex.abs (ζ^a + ζ^b + ζ^c + ζ^d) = Real.sqrt 3) :
  1000 * a + 100 * b + 10 * c + d ≥ 7521 := by
  sorry

#check smallest_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_value_theorem_l262_26256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l262_26225

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x - Real.sin (x - Real.pi/4)^2

theorem f_properties :
  (∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ ∀ q : ℝ, q > 0 ∧ (∀ x : ℝ, f (x + q) = f x) → p ≤ q) ∧
  (∃ M : ℝ, M = (Real.sqrt 3 - 1)/2 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) → f (x - Real.pi/6) ≤ M) ∧
  (∃ m : ℝ, m = -(Real.sqrt 3 + 1)/2 ∧ ∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi/2) → m ≤ f (x - Real.pi/6)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l262_26225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_necessary_not_sufficient_for_B_l262_26205

-- Define proposition A
def proposition_A (a : ℝ) : Prop :=
  ∃ x, x^2 + 2*a*x + 4 ≤ 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop :=
  ∀ x > 1, Real.log (x + a - 2) > 0

-- Theorem statement
theorem A_necessary_not_sufficient_for_B :
  (∀ a, proposition_B a → proposition_A a) ∧
  (∃ a, proposition_A a ∧ ¬proposition_B a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_necessary_not_sufficient_for_B_l262_26205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_free_subset_l262_26294

theorem arithmetic_mean_free_subset (k : ℕ) :
  ∃ (S : Finset ℕ), 
    S.card = 2^k ∧
    S ⊆ Finset.range (3^k) ∧
    ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → 2 * z ≠ x + y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_free_subset_l262_26294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_angle_sum_l262_26241

theorem tangent_and_angle_sum (α β γ : ℝ) 
  (h1 : Real.tan (α + β) = 9/13)
  (h2 : Real.tan (β - π/4) = -1/3)
  (h3 : Real.cos γ = 3 * Real.sqrt 10 / 10)
  (h4 : 0 < α ∧ α < π/2)
  (h5 : 0 < γ ∧ γ < π/2) : 
  Real.tan α = 1/7 ∧ α + 2*γ = π/4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_angle_sum_l262_26241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l262_26270

noncomputable def f (x : ℝ) : ℝ := (x^2 - 16) / (x - 4)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ 4} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l262_26270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_white_stones_theorem_l262_26204

/-- Represents the state of Snow White's stone piles -/
structure SnowWhiteStones where
  total_stones : ℕ
  num_piles : ℕ

/-- The process of splitting piles and adding stones -/
def split_and_add (state : SnowWhiteStones) : SnowWhiteStones :=
  { total_stones := state.total_stones + 1,
    num_piles := state.num_piles + 1 }

/-- Theorem stating the final state of Snow White's stone piles -/
theorem snow_white_stones_theorem :
  ∀ (n : ℕ),
  (Nat.iterate split_and_add n { total_stones := 36, num_piles := 1 }).num_piles = 7 →
  (Nat.iterate split_and_add n { total_stones := 36, num_piles := 1 }).total_stones / 7 = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_white_stones_theorem_l262_26204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_concentration_problem_l262_26245

/-- Represents the sugar concentration of a solution -/
structure SugarConcentration where
  value : ℝ
  nonneg : 0 ≤ value
  le_one : value ≤ 1

/-- Calculates the resulting sugar concentration when mixing two solutions -/
def mixSolutions (c1 c2 : SugarConcentration) (r : ℝ) : ℝ :=
  r * c1.value + (1 - r) * c2.value

theorem sugar_concentration_problem (c1 c2 : SugarConcentration) :
  c1.value = 0.12 →
  mixSolutions c1 c2 0.75 = 0.16 →
  c2.value = 0.28 := by
  sorry

#check sugar_concentration_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sugar_concentration_problem_l262_26245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_for_given_triangle_l262_26267

/-- Represents a right triangle with a square in its right angle corner. -/
structure RightTriangleWithSquare where
  /-- Length of the first leg of the right triangle -/
  leg1 : ℝ
  /-- Length of the second leg of the right triangle -/
  leg2 : ℝ
  /-- Side length of the square in the right angle corner -/
  square_side : ℝ
  /-- Distance from the closest point of the square to the hypotenuse -/
  square_to_hypotenuse : ℝ
  /-- The square's side length is determined by the triangle's dimensions and its distance to the hypotenuse -/
  square_side_constraint : square_side * Real.sqrt 2 + square_to_hypotenuse = (leg1 * leg2) / Real.sqrt (leg1^2 + leg2^2)

/-- The fraction of the triangle area not covered by the square -/
noncomputable def planted_fraction (t : RightTriangleWithSquare) : ℝ :=
  (t.leg1 * t.leg2 / 2 - t.square_side^2) / (t.leg1 * t.leg2 / 2)

/-- The main theorem stating the planted fraction for the given triangle -/
theorem planted_fraction_for_given_triangle :
  ∃ t : RightTriangleWithSquare,
    t.leg1 = 5 ∧
    t.leg2 = 12 ∧
    t.square_to_hypotenuse = 3 ∧
    planted_fraction t = 9693 / 10140 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_planted_fraction_for_given_triangle_l262_26267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_condition_l262_26221

-- Define sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {y | ∃ x, y = 2^x ∧ 2 ≤ x ∧ x < 3}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem set_operations_and_subset_condition :
  ∀ a : ℝ,
  (C a) ⊆ B →
  (A ∩ B = Set.Icc 4 6) ∧
  ((Set.univ \ B) ∪ A = Set.Iic 6 ∪ Set.Ici 8) ∧
  (4 ≤ a ∧ a ≤ 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_operations_and_subset_condition_l262_26221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_l262_26200

noncomputable def smallest_positive_angle : ℝ :=
  (1/2) * Real.arcsin (3^(-(1/6 : ℝ)))

theorem smallest_angle_satisfies_equation :
  12 * (Real.sin smallest_positive_angle)^3 * (Real.cos smallest_positive_angle)^3 = Real.sqrt 3/2 ∧
  ∀ y, 0 < y ∧ y < smallest_positive_angle →
    12 * (Real.sin y)^3 * (Real.cos y)^3 ≠ Real.sqrt 3/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_angle_satisfies_equation_l262_26200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a1_range_l262_26296

def sequence_increasing (a : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, a (n + 1) > a n

def sum_property (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n : ℕ, n ≥ 2 → S (n + 1) + S n + S (n - 1) = 3 * n^2 + 2

theorem a1_range (a : ℕ → ℚ) (S : ℕ → ℚ) :
  sequence_increasing a →
  a 2 = 3 * a 1 →
  sum_property a S →
  (∀ n : ℕ, S (n + 1) = S n + a (n + 1)) →
  13/15 < a 1 ∧ a 1 < 7/6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a1_range_l262_26296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_n_l262_26244

open Nat

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 18^5) (h2 : m % 10 = 9) : n % 10 = 2 := by
  have h3 : 18^5 % 10 = 8 := by
    -- Proof that 18^5 ≡ 8 (mod 10)
    sorry
  
  have h4 : (m * n) % 10 = 8 := by
    rw [h1]
    exact h3

  have h5 : (9 * (n % 10)) % 10 = 8 := by
    -- Proof that 9 * (n % 10) ≡ 8 (mod 10)
    sorry

  -- Show that n % 10 = 2 is the only solution
  have h6 : n % 10 = 2 := by
    -- Proof by exhaustion or direct calculation
    sorry

  exact h6

#eval (18^5) % 10  -- Should output 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_of_n_l262_26244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_parttime_hours_l262_26259

/-- Madeline's weekly schedule -/
structure MadelineSchedule where
  class_hours : ℕ
  homework_hours_per_day : ℕ
  sleep_hours_per_day : ℕ
  leftover_hours : ℕ
  parttime_hours : ℕ

/-- The given conditions for Madeline's schedule -/
def given_schedule : MadelineSchedule :=
  { class_hours := 18
  , homework_hours_per_day := 4
  , sleep_hours_per_day := 8
  , leftover_hours := 46
  , parttime_hours := 0 }  -- We'll prove this should be 20

/-- Total hours in a week -/
def hours_per_week : ℕ := 24 * 7

/-- Theorem: Madeline works part-time for 20 hours per week -/
theorem madeline_parttime_hours (s : MadelineSchedule) 
  (h1 : s.class_hours = given_schedule.class_hours)
  (h2 : s.homework_hours_per_day = given_schedule.homework_hours_per_day)
  (h3 : s.sleep_hours_per_day = given_schedule.sleep_hours_per_day)
  (h4 : s.leftover_hours = given_schedule.leftover_hours) :
  s.parttime_hours = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_madeline_parttime_hours_l262_26259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_given_arcs_l262_26266

/-- The area of a triangle inscribed in a circle, where the vertices divide the circle
    into three arcs of lengths 6, 8, and 10. -/
noncomputable def triangle_area (arc1 arc2 arc3 : ℝ) : ℝ :=
  let r := (arc1 + arc2 + arc3) / (2 * Real.pi)
  let θ := Real.pi / 12  -- 15 degrees in radians
  (1 / 2) * r^2 * (Real.sin (3 * θ) + Real.sin (4 * θ) + Real.sin (5 * θ))

/-- Theorem stating that the area of the triangle with given arc lengths is 72(2+√3)/(2π²) -/
theorem triangle_area_with_given_arcs :
  triangle_area 6 8 10 = 72 * (2 + Real.sqrt 3) / (2 * Real.pi^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_with_given_arcs_l262_26266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l262_26295

-- Define the ellipse C
def C (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
variable (a b : ℝ)
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0
axiom P_on_C : C a b 1 (Real.sqrt 2 / 2)
axiom c_eq_1 : a^2 - b^2 = 1

-- Define the equation of line AB
def line_AB (k m x y : ℝ) : Prop := y = k * x + m

-- Define the slopes of MA and MB
variable (k₁ k₂ : ℝ)
axiom k1_k2_sum : k₁ + k₂ = 2

-- Theorem to prove
theorem fixed_point_theorem (k m : ℝ) (h : m ≠ 1) :
  ∃ (x₀ y₀ : ℝ), ∀ (k m : ℝ), m ≠ 1 → line_AB k m x₀ y₀ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_theorem_l262_26295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l262_26288

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * Real.sin (x + Real.pi/8) * (Real.sin (x + Real.pi/8) - Real.cos (x + Real.pi/8))

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∃ M, M = f ((-Real.pi/2 : ℝ) + Real.pi/8) ∧ ∀ x ∈ Set.Icc (-Real.pi/2 : ℝ) 0, f (x + Real.pi/8) ≤ M) ∧
  (∃ m, m = f (0 + Real.pi/8) ∧ ∀ x ∈ Set.Icc (-Real.pi/2 : ℝ) 0, m ≤ f (x + Real.pi/8)) ∧
  (∃ T > 0, T = Real.pi) ∧
  (∃ M, M = Real.sqrt 2) ∧
  (∃ m, m = -1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l262_26288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l262_26234

noncomputable section

open Real

theorem triangle_inequalities (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  (0 < A ∧ 0 < B ∧ 0 < C) →
  (A + B + C = π) →
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a > 0 ∧ b > 0 ∧ c > 0) →
  -- Law of sines
  (a / Real.sin A = b / Real.sin B) →
  (
    -- 1. If a > b, then sin A > sin B
    (a > b → Real.sin A > Real.sin B) ∧
    -- 2. If sin A > sin B, then A > B
    (Real.sin A > Real.sin B → A > B) ∧
    -- 3. If ABC is acute, then sin A > cos B
    (A < π/2 ∧ B < π/2 ∧ C < π/2 → Real.sin A > Real.cos B)
  ) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequalities_l262_26234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l262_26239

/-- The lateral surface area of a cylinder with given radius and height -/
noncomputable def lateralSurfaceArea (r h : Real) : Real := 2 * Real.pi * r * h

/-- Theorem: The lateral surface area of a cylinder with base radius 2 decimeters
    and height 5 decimeters is approximately equal to 62.8 square decimeters -/
theorem cylinder_lateral_surface_area :
  ∃ ε > 0, |lateralSurfaceArea 2 5 - 62.8| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_lateral_surface_area_l262_26239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_division_exists_l262_26212

/-- Represents a robber's valuation of the loot -/
def RobberValuation := ℝ → ℝ

/-- Represents a division of the loot -/
def LootDivision := List ℝ

/-- A function that checks if a division is fair according to a set of valuations -/
def is_fair_division (n : ℕ) (valuations : List RobberValuation) (div : LootDivision) : Prop :=
  div.length = n ∧ 
  ∀ (i : Fin n), (valuations.get ⟨i, by sorry⟩) (div.get ⟨i, by sorry⟩) ≥ (1 : ℝ) / n

/-- The main theorem: there exists a fair division for any number of robbers -/
theorem fair_division_exists (n : ℕ) (h : n ≥ 2) (valuations : List RobberValuation) 
  (h_val : valuations.length = n) :
  ∃ (div : LootDivision), is_fair_division n valuations div := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fair_division_exists_l262_26212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_product_range_l262_26202

noncomputable def f (x : ℝ) : ℝ := abs (abs (x - 1) - 1)

theorem four_roots_product_range :
  ∀ (m : ℝ) (x₁ x₂ x₃ x₄ : ℝ),
    (∀ x : ℝ, f x = m ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) →
    x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ →
    x₁ * x₂ * x₃ * x₄ ∈ Set.Ioo (-3 : ℝ) 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_roots_product_range_l262_26202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_plus_minus_one_l262_26224

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (2^x - a) / (2^x + a) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (2^x - a) / (2^x + a)

/-- Theorem: If f(x) = (2^x - a) / (2^x + a) is an odd function and a ∈ ℝ, then a = 1 or a = -1 -/
theorem odd_function_implies_a_plus_minus_one (a : ℝ) :
  IsOdd (f a) → (a = 1 ∨ a = -1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_plus_minus_one_l262_26224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ivanov_receives_12_million_l262_26285

/-- Represents the contribution of each businessman in million rubles -/
structure Contribution where
  ivanov : ℝ
  petrov : ℝ
  sidorov : ℝ

/-- The number of cars bought by each businessman -/
def cars : Contribution where
  ivanov := 70
  petrov := 40
  sidorov := 0

/-- Sidorov's monetary contribution in million rubles -/
def sidorov_money : ℝ := 44

/-- The price of each car in million rubles -/
def car_price : ℝ → ℝ := id

/-- Calculate the total contribution of each businessman -/
def total_contribution (cp : ℝ) : Contribution :=
  { ivanov := cars.ivanov * car_price cp,
    petrov := cars.petrov * car_price cp,
    sidorov := sidorov_money }

/-- The amount Ivanov should receive to equalize contributions -/
def ivanov_receives (cp : ℝ) : ℝ :=
  (total_contribution cp).ivanov - sidorov_money

/-- Theorem stating that Ivanov should receive 12 million rubles -/
theorem ivanov_receives_12_million :
  ∃ cp : ℝ, cp > 0 ∧ (total_contribution cp).ivanov = (total_contribution cp).petrov
           ∧ (total_contribution cp).ivanov = (total_contribution cp).sidorov
           ∧ ivanov_receives cp = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ivanov_receives_12_million_l262_26285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequence_recurrence_valid_sequences_of_length_15_l262_26257

/-- Represents a sequence of A's and B's -/
inductive ABSequence
| A : ABSequence
| B : ABSequence

/-- Predicate for valid sequences where runs of A's have odd length and runs of B's have even length -/
def IsValidSequence : List ABSequence → Prop := sorry

/-- Number of valid sequences of length n ending in A -/
def a : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 1
| (n + 3) => a (n + 2) + a (n + 1)

/-- Number of valid sequences of length n ending in B -/
def b : ℕ → ℕ
| 0 => 0
| 1 => 0
| 2 => 1
| (n + 3) => b (n + 1)

/-- Theorem stating the recurrence relations for valid sequences -/
theorem valid_sequence_recurrence (n : ℕ) (h : n ≥ 3) :
  a n = a (n - 1) + b (n - 1) ∧ b n = b (n - 2) := by sorry

/-- Total number of valid sequences of length n -/
def total_sequences (n : ℕ) : ℕ := a n + b n

/-- The main theorem to prove, which gives the number of valid sequences of length 15 -/
theorem valid_sequences_of_length_15 :
  total_sequences 15 = a 15 + b 15 := by sorry

#eval total_sequences 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_sequence_recurrence_valid_sequences_of_length_15_l262_26257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l262_26261

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (2 + Real.exp (x - 1)) / Real.log a

-- State the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f a x ≤ -1) → a ∈ Set.Icc (1/2) 1 ∧ a ≠ 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l262_26261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_inscribed_triangle_l262_26213

/-- An equilateral triangle with side length √3 -/
structure EquilateralTriangle :=
  (side : ℝ)
  (is_equilateral : side = Real.sqrt 3)

/-- A point that divides a side of the triangle in the ratio 2:1 -/
noncomputable def dividing_point (t : EquilateralTriangle) : ℝ := 2 * t.side / 3

/-- The minimal perimeter of an inscribed triangle -/
noncomputable def minimal_perimeter (t : EquilateralTriangle) : ℝ := Real.sqrt 7

theorem minimal_perimeter_inscribed_triangle 
  (t : EquilateralTriangle) 
  (m : ℝ) 
  (h : m = dividing_point t) :
  minimal_perimeter t = Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_perimeter_inscribed_triangle_l262_26213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l262_26220

/-- The surface area of a cube with side length a is 6a^2 -/
theorem cube_surface_area (a : ℝ) (h : a > 0) : 
  6 * a^2 = (6 : ℝ) * a^2 := by
  -- The proof is trivial as this is just asserting equality
  rfl

#check cube_surface_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l262_26220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_cycle_exists_unique_l262_26284

-- Define a structure for a line in a plane
structure PlaneLine where
  -- Add necessary fields to represent a line
  dummy : Unit

-- Define a structure for a point in a plane
structure PlanePoint where
  -- Add necessary fields to represent a point
  dummy : Unit

-- Function to check if two lines are parallel
def are_parallel (l₁ l₂ : PlaneLine) : Prop :=
  sorry

-- Function to check if a set of lines are not all parallel
def not_all_parallel (lines : List PlaneLine) : Prop :=
  ∃ l₁ l₂, l₁ ∈ lines ∧ l₂ ∈ lines ∧ l₁ ≠ l₂ ∧ ¬ are_parallel l₁ l₂

-- Function to check if a point is on a line
def point_on_line (p : PlanePoint) (l : PlaneLine) : Prop :=
  sorry

-- Function to check if a line is perpendicular to another line at a given point
def perpendicular_at_point (l₁ l₂ : PlaneLine) (p : PlanePoint) : Prop :=
  sorry

-- Main theorem
theorem perpendicular_cycle_exists_unique 
  (lines : List PlaneLine) 
  (h_not_parallel : not_all_parallel lines) :
  ∃! (points : List PlanePoint), 
    points.length = lines.length ∧ 
    (∀ i : Fin lines.length, 
      point_on_line (points.get ⟨i.val, by sorry⟩) (lines.get i) ∧
      perpendicular_at_point 
        (lines.get i) 
        (lines.get ⟨(i.val + 1) % lines.length, by sorry⟩)
        (points.get ⟨(i.val + 1) % lines.length, by sorry⟩)) :=
by
  sorry

#check perpendicular_cycle_exists_unique

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_cycle_exists_unique_l262_26284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_minimized_l262_26264

/-- Definition of a hyperbola -/
noncomputable def is_hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

/-- Definition of eccentricity for a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + b^2 / a^2)

/-- Definition of the function to be minimized -/
noncomputable def f (k : ℝ) : ℝ := 2 / k + Real.log k

/-- Theorem statement -/
theorem hyperbola_eccentricity_when_minimized (a b : ℝ) :
  (∃ x y : ℝ, is_hyperbola a b x y) →
  (∃ k : ℝ, k > 0 ∧ ∀ t > 0, f k ≤ f t) →
  eccentricity a b = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_minimized_l262_26264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mosquito_fly_distance_ratio_l262_26272

/-- Represents the position of an insect on a clock hand -/
structure ClockPosition where
  angle : ℝ  -- Angle in radians
  hand : Bool  -- true for hour hand, false for minute hand

/-- Calculates the new position of an insect after one hour -/
noncomputable def new_position (pos : ClockPosition) : ClockPosition :=
  if pos.hand then
    { angle := pos.angle + Real.pi / 6, hand := pos.hand }
  else
    { angle := pos.angle + 2 * Real.pi, hand := pos.hand }

/-- Switches the hand of the insect when the hands overlap -/
def switch_hands (pos : ClockPosition) : ClockPosition :=
  { angle := pos.angle, hand := ¬pos.hand }

/-- Calculates the total distance traveled by an insect over 12 hours -/
noncomputable def total_distance (initial_pos : ClockPosition) : ℝ :=
  sorry

/-- The main theorem stating the ratio of distances traveled -/
theorem mosquito_fly_distance_ratio : 
  ∃ (mosquito_start fly_start : ClockPosition),
    mosquito_start.angle = Real.pi / 6 ∧ 
    mosquito_start.hand = true ∧
    fly_start.angle = 0 ∧ 
    fly_start.hand = false ∧
    (total_distance mosquito_start) / (total_distance fly_start) = 83 / 73 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mosquito_fly_distance_ratio_l262_26272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pdo_is_45_degrees_l262_26253

/-- A square in a 2D plane -/
structure Square where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

/-- A point on the diagonal of a square -/
def DiagonalPoint (sq : Square) : Type :=
  {L : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ L = (1 - t) • sq.A + t • sq.C}

/-- A point on the side of a square -/
def SidePoint (sq : Square) : Type :=
  {P : ℝ × ℝ // ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • sq.A + t • sq.B}

/-- The center of a square -/
noncomputable def SquareCenter (sq : Square) : ℝ × ℝ :=
  ((sq.A.1 + sq.C.1) / 2, (sq.A.2 + sq.C.2) / 2)

/-- The angle between three points -/
noncomputable def Angle (A B C : ℝ × ℝ) : ℝ := sorry

theorem angle_pdo_is_45_degrees
  (ABCD : Square)
  (L : DiagonalPoint ABCD)
  (P : SidePoint ABCD)
  (APLQ CMLN : Square)
  (h1 : L.val = APLQ.C ∧ L.val = CMLN.A)
  (h2 : P.val = APLQ.B)
  (O : ℝ × ℝ)
  (h3 : O = SquareCenter CMLN) :
  Angle P.val ABCD.D O = 45 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_pdo_is_45_degrees_l262_26253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_green_iff_is_green_dec_probability_green_tile_l262_26258

def is_green (n : ℕ) : Prop := n % 7 = 3

-- We need to make this function decidable
@[simp] def is_green_dec (n : ℕ) : Bool := n % 7 = 3

theorem is_green_iff_is_green_dec (n : ℕ) : is_green n ↔ is_green_dec n = true := by
  simp [is_green, is_green_dec]

def green_tiles : Finset ℕ := Finset.filter (fun n => is_green_dec n) (Finset.range 100)

theorem probability_green_tile :
  (Finset.card green_tiles : ℚ) / 100 = 7 / 50 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_green_iff_is_green_dec_probability_green_tile_l262_26258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_k_l262_26219

-- Define the variables
variable (x y z k : ℝ)

-- Define the conditions
def condition1 (x y : ℝ) : Prop := x - 4*y + 3 ≤ 0
def condition2 (x y : ℝ) : Prop := 3*x + 5*y - 25 ≤ 0
def condition3 (k : ℝ) : Prop := ∃ (z_min z_max : ℝ), z_min = 3 ∧ z_max = 12 ∧ 
                         (∀ x y, z_min ≤ k*x + y ∧ k*x + y ≤ z_max)
def condition4 (x : ℝ) : Prop := x ≥ 1

-- State the theorem
theorem value_of_k (h1 : condition1 x y) (h2 : condition2 x y) (h3 : condition3 k) (h4 : condition4 x) : 
  k = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_k_l262_26219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_efficiency_ranking_l262_26217

-- Define the types of cleaning wipe packs
inductive PackSize
| S -- Small
| M -- Medium
| L -- Large

-- Define the structure for a pack of wipes
structure WipePack where
  size : PackSize
  cost : ℚ
  quantity : ℚ

-- Define the packs based on the given conditions
def small_pack : WipePack := { size := PackSize.S, cost := 1, quantity := 10 }

def medium_pack : WipePack := { 
  size := PackSize.M, 
  cost := small_pack.cost * (5/4), 
  quantity := small_pack.quantity * (3/2) 
}

def large_pack : WipePack := { 
  size := PackSize.L, 
  cost := medium_pack.cost * (7/5), 
  quantity := small_pack.quantity * (5/2) 
}

-- Define cost efficiency
def cost_efficiency (pack : WipePack) : ℚ := pack.cost / pack.quantity

-- Theorem to prove the cost efficiency ranking
theorem cost_efficiency_ranking : 
  cost_efficiency large_pack < cost_efficiency medium_pack ∧ 
  cost_efficiency medium_pack < cost_efficiency small_pack := by
  -- Expand the definitions and simplify
  unfold cost_efficiency
  unfold large_pack medium_pack small_pack
  -- Perform the arithmetic
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_efficiency_ranking_l262_26217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_52_l262_26248

-- Define the cost function R(x)
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 50 then 10 * x^2 + 100 * x + 800
  else if x ≥ 50 then 504 * x + 10000 / (x - 2) - 6450
  else 0

-- Define the profit function y(x)
noncomputable def y (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 50 then 500 * x - R x - 250
  else if x ≥ 50 then 500 * x - R x - 250
  else 0

-- Define the maximum profit and output
def max_profit : ℝ := 5792
def max_output : ℝ := 52

-- Theorem statement
theorem max_profit_at_52 :
  ∀ x : ℝ, x > 0 → y x ≤ max_profit ∧
  y max_output = max_profit := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_52_l262_26248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l262_26293

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x) ^ 2

theorem f_properties :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ ∀ S, 0 < S ∧ S < T → ∃ y, f (y + S) ≠ f y) ∧
  (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≥ 0) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l262_26293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l262_26231

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the curve C
noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
  let ρ := 4 * Real.cos θ / (Real.sin θ)^2
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define point M
def point_M : ℝ × ℝ := (2, 0)

-- Define the intersection points A and B
axiom exists_intersection : ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
  ∃ (θ₁ θ₂ : ℝ), line_l t₁ = curve_C θ₁ ∧ line_l t₂ = curve_C θ₂

-- State the theorem
theorem intersection_distance_difference :
  ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
  (∃ (t₁ t₂ θ₁ θ₂ : ℝ), 
    line_l t₁ = curve_C θ₁ ∧ line_l t₂ = curve_C θ₂ ∧
    A = line_l t₁ ∧ B = line_l t₂) ∧
  |1 / dist point_M A - 1 / dist point_M B| = 1/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l262_26231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_stone_placements_count_l262_26209

/-- A hexagram is a regular six-pointed star with 12 intersection points -/
structure Hexagram :=
  (points : Fin 12 → Point)

/-- A stone placement is an assignment of unique stones to the points of a hexagram -/
def StonePlacement := Fin 12 → Fin 12

/-- The group of symmetries of a hexagram -/
def HexagramSymmetryGroup := Subgroup (Equiv.Perm (Fin 12))

/-- The number of elements in the hexagram symmetry group -/
def hexagramSymmetryGroupOrder : ℕ := 24

/-- The number of distinct stone placements on a hexagram -/
def distinctStonePlacements : ℕ := (Nat.factorial 12) / hexagramSymmetryGroupOrder

theorem distinct_stone_placements_count :
  distinctStonePlacements = 19958400 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_stone_placements_count_l262_26209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_calculation_correct_l262_26262

/-- Calculate the import tax for an item --/
noncomputable def calculate_import_tax (total_value : ℝ) (tax_rate : ℝ) (threshold : ℝ) : ℝ :=
  max 0 ((total_value - threshold) * tax_rate)

/-- The import tax calculation is correct --/
theorem import_tax_calculation_correct :
  let total_value : ℝ := 2570
  let tax_rate : ℝ := 0.07
  let threshold : ℝ := 1000
  calculate_import_tax total_value tax_rate threshold = 109.90 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_import_tax_calculation_correct_l262_26262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_oranges_problem_l262_26292

theorem tom_oranges_problem (initial_apples : ℕ) (oranges_sold_fraction : ℚ) 
  (apples_sold_fraction : ℚ) (total_fruits_left : ℕ) :
  initial_apples = 70 →
  oranges_sold_fraction = 1/4 →
  apples_sold_fraction = 1/2 →
  total_fruits_left = 65 →
  ∃ initial_oranges : ℕ, 
    initial_oranges = 40 ∧ 
    (1 - oranges_sold_fraction) * (initial_oranges : ℚ) + 
    (1 - apples_sold_fraction) * (initial_apples : ℚ) = (total_fruits_left : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_oranges_problem_l262_26292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_T_is_parallel_lines_l262_26280

/-- Two fixed points in a plane -/
def D : ℝ × ℝ := sorry
def E : ℝ × ℝ := sorry

/-- The distance between points D and E -/
noncomputable def DE : ℝ := Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2)

/-- The area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := sorry

/-- The set T of all points F such that the area of triangle DEF is 4 -/
def T : Set (ℝ × ℝ) :=
  {F | area_triangle D E F = 4}

/-- The perpendicular distance from a point to the line DE -/
noncomputable def perp_distance (F : ℝ × ℝ) : ℝ :=
  8 / DE

/-- The two parallel lines that describe set T -/
def parallel_lines : Set (ℝ × ℝ) :=
  {F | perp_distance F = perp_distance F ∨ 
       perp_distance F = -perp_distance F}

/-- Theorem: The set T is described by two parallel lines -/
theorem set_T_is_parallel_lines :
  T = parallel_lines := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_T_is_parallel_lines_l262_26280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_ticket_price_l262_26299

theorem circus_ticket_price (num_kids num_adults : ℕ) (total_cost : ℚ) :
  num_kids = 6 →
  num_adults = 2 →
  total_cost = 50 →
  ∃ (kid_price : ℚ),
    kid_price * num_kids + (2 * kid_price) * num_adults = total_cost ∧
    kid_price = 5 :=
by
  intro h_kids h_adults h_total
  use 5
  constructor
  · rw [h_kids, h_adults, h_total]
    norm_num
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circus_ticket_price_l262_26299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_money_l262_26247

/-- Represents the amount of money each person has -/
structure MoneyDistribution where
  adrian : ℤ
  brenda : ℤ
  charlie : ℤ
  dana : ℤ
  evan : ℤ

/-- Checks if the given money distribution satisfies all conditions -/
def satisfiesConditions (m : MoneyDistribution) : Prop :=
  m.adrian + m.brenda + m.charlie + m.dana + m.evan = 72 ∧
  |m.adrian - m.brenda| = 21 ∧
  |m.brenda - m.charlie| = 8 ∧
  |m.charlie - m.dana| = 6 ∧
  |m.dana - m.evan| = 5 ∧
  |m.evan - m.adrian| = 14

theorem evans_money (m : MoneyDistribution) :
  satisfiesConditions m → m.evan = 21 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_evans_money_l262_26247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_exponentials_l262_26297

theorem order_of_exponentials : (3 : ℝ)^Real.sqrt 2 > (3 : ℝ)^(0.3 : ℝ) ∧ (3 : ℝ)^(0.3 : ℝ) > (0.9 : ℝ)^(3.1 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_exponentials_l262_26297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_failing_l262_26260

def prob_above_80 : ℝ := 0.69
def prob_70_to_79 : ℝ := 0.15
def prob_60_to_69 : ℝ := 0.09

theorem probability_of_failing :
  1 - (prob_above_80 + prob_70_to_79 + prob_60_to_69) = 0.07 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_failing_l262_26260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l262_26226

/-- The function representing the abscissa of the moving point -/
noncomputable def x (t a : ℝ) : ℝ := 5 * (t + 1)^2 + a / (t + 1)^5

/-- The theorem stating the minimum value of a -/
theorem min_a_value : 
  ∃ (a_min : ℝ), a_min > 0 ∧ 
  (∀ (t : ℝ), t ≥ 0 → x t a_min ≥ 24) ∧
  (∀ (a : ℝ), a > 0 → (∀ (t : ℝ), t ≥ 0 → x t a ≥ 24) → a ≥ a_min) ∧
  a_min = 2 * Real.sqrt ((24/7)^7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_value_l262_26226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l262_26250

theorem cos_alpha_value (θ α : Real) 
  (h1 : Real.sin θ + Real.cos θ = Real.sin α)
  (h2 : Real.sin θ * Real.cos θ = -Real.sin (2 * α))
  (h3 : 0 < α ∧ α < Real.pi / 2) :
  Real.cos α = 4 * Real.sqrt 17 / 17 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l262_26250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BCA_is_30_l262_26268

-- Define the angles as real numbers
variable (angle_BCA angle_BCD angle_CBA : ℝ)

-- State the theorem
theorem angle_BCA_is_30 :
  angle_BCD = 90 →
  angle_BCA + angle_BCD + angle_CBA = 190 →
  angle_CBA = 70 →
  angle_BCA = 30 :=
by
  -- Introduce the hypotheses
  intro h1 h2 h3
  -- Substitute known values into the equation
  have h4 : angle_BCA + 90 + 70 = 190 := by
    rw [h1, h3] at h2
    exact h2
  -- Simplify the equation
  have h5 : angle_BCA + 160 = 190 := by
    linarith
  -- Solve for angle_BCA
  have h6 : angle_BCA = 30 := by
    linarith
  -- Conclude the proof
  exact h6


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_BCA_is_30_l262_26268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_median_soda_distribution_l262_26251

/-- Represents the distribution of soda cans to customers -/
structure SodaDistribution where
  total_cans : ℕ
  total_customers : ℕ
  cans_per_customer : Fin total_customers → ℕ
  h_total : (Finset.sum Finset.univ cans_per_customer) = total_cans
  h_min_one : ∀ i, cans_per_customer i ≥ 1

/-- The median of a sorted list is the average of the middle two elements if the list has even length -/
def median (l : List ℕ) : ℚ :=
  let n := l.length
  if n % 2 = 0 then
    (l.get! ((n / 2) - 1) + l.get! (n / 2)) / 2
  else
    l.get! (n / 2)

/-- The main theorem stating the maximum possible median -/
theorem max_median_soda_distribution :
  ∃ (d : SodaDistribution),
    d.total_cans = 310 ∧
    d.total_customers = 120 ∧
    (∀ d' : SodaDistribution,
      d'.total_cans = 310 →
      d'.total_customers = 120 →
      median (Finset.sort (λ a b : ℕ ↦ a ≤ b) (Finset.univ.image d'.cans_per_customer))
      ≤ median (Finset.sort (λ a b : ℕ ↦ a ≤ b) (Finset.univ.image d.cans_per_customer))) ∧
    median (Finset.sort (λ a b : ℕ ↦ a ≤ b) (Finset.univ.image d.cans_per_customer)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_median_soda_distribution_l262_26251
