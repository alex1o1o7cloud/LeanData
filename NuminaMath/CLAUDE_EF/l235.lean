import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_cos2theta_l235_23531

noncomputable def vector_a (θ : Real) : Real × Real := (2 * Real.cos θ, 1)
noncomputable def vector_b (θ : Real) : Real × Real := (1, Real.cos θ)

theorem collinear_vectors_cos2theta (θ : Real) :
  (∃ k : Real, vector_a θ = k • vector_b θ) → Real.cos (2 * θ) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_collinear_vectors_cos2theta_l235_23531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_e_cannot_be_formed_l235_23586

/-- Represents a tile with a specific coloring scheme -/
structure Tile where
  shape : String
  coloring : String

/-- Represents a pattern formed by arranging tiles -/
inductive Pattern where
  | A
  | B
  | C
  | D
  | E

/-- Predicate to check if all tiles in an arrangement are equal to a given tile -/
def all_tiles_equal (arrangement : List Tile) (t : Tile) : Prop :=
  ∀ tile ∈ arrangement, tile = t

/-- Predicate to check if an arrangement forms a specific pattern -/
def forms_pattern (arrangement : List Tile) (p : Pattern) : Prop :=
  sorry -- Definition would depend on specific rules for each pattern

/-- Predicate to check if a pattern can be formed using copies of a given tile -/
def can_form_pattern (t : Tile) (p : Pattern) : Prop :=
  ∃ (arrangement : List Tile), all_tiles_equal arrangement t ∧ forms_pattern arrangement p

/-- The main theorem stating that pattern E cannot be formed while others can -/
theorem pattern_e_cannot_be_formed (t : Tile) 
  (h_rhombus : t.shape = "rhombus") 
  (h_diagonal : t.coloring = "diagonal_black_white") :
  ¬(can_form_pattern t Pattern.E) ∧ 
  (can_form_pattern t Pattern.A) ∧
  (can_form_pattern t Pattern.B) ∧
  (can_form_pattern t Pattern.C) ∧
  (can_form_pattern t Pattern.D) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pattern_e_cannot_be_formed_l235_23586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_completing_square_reflects_transformation_l235_23592

/-- Represents a quadratic equation in the form ax² + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Represents the transformed form of a quadratic equation (x+m)² = n -/
structure TransformedQuadratic where
  m : ℝ
  n : ℝ
  n_nonneg : n ≥ 0

/-- Represents the process of completing the square -/
def complete_square (eq : QuadraticEquation) : TransformedQuadratic :=
  sorry

/-- Enum representing different mathematical ideas -/
inductive MathIdea
  | NumberShapeCombination
  | Function
  | Transformation
  | Axiomatization

/-- Helper function to determine the dominant math idea -/
def dominantIdea (ideas : List MathIdea) : MathIdea :=
  match ideas with
  | [] => MathIdea.Axiomatization  -- Default to Axiomatization if list is empty
  | (x :: xs) => x  -- For simplicity, we're just taking the first idea in the list

theorem completing_square_reflects_transformation 
  (eq : QuadraticEquation) 
  (transformed : TransformedQuadratic) 
  (h : transformed = complete_square eq) : 
  dominantIdea [MathIdea.Transformation, MathIdea.NumberShapeCombination, 
                MathIdea.Function, MathIdea.Axiomatization] = MathIdea.Transformation := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_completing_square_reflects_transformation_l235_23592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_grid_by_folding_l235_23527

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents a folding operation -/
inductive FoldingOperation where
  | MidpointFold : Point → Point → FoldingOperation
  | PerpendicularFold : Point → Point → Point → FoldingOperation

/-- Represents a grid of equally spaced lines -/
structure Grid where
  horizontalLines : List (Point → Point → Prop)
  verticalLines : List (Point → Point → Prop)

/-- Main theorem: It is possible to construct a grid on an isosceles triangle using only folding operations -/
theorem construct_grid_by_folding (t : Triangle) 
  (h_isosceles : t.A.x^2 + t.A.y^2 = t.B.x^2 + t.B.y^2) : 
  ∃ (folds : List FoldingOperation) (g : Grid), 
    (∀ p : Point, (p = t.A ∨ p = t.B ∨ p = t.C) → 
      ∃ l, (l ∈ g.horizontalLines ∨ l ∈ g.verticalLines) ∧ l p p) ∧
    (∀ l1 l2, l1 ∈ g.horizontalLines → l2 ∈ g.horizontalLines → l1 ≠ l2 → 
      ∃ d : ℝ, ∀ p1 p2 : Point, l1 p1 p1 → l2 p2 p2 → |p1.y - p2.y| = d) ∧
    (∀ l1 l2, l1 ∈ g.verticalLines → l2 ∈ g.verticalLines → l1 ≠ l2 → 
      ∃ d : ℝ, ∀ p1 p2 : Point, l1 p1 p1 → l2 p2 p2 → |p1.x - p2.x| = d) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_construct_grid_by_folding_l235_23527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_flight_time_l235_23524

/-- The time Lisa flew given her distance and speed -/
noncomputable def flight_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem stating that Lisa's flight time is 8 hours -/
theorem lisa_flight_time :
  flight_time 256 32 = 8 := by
  -- Unfold the definition of flight_time
  unfold flight_time
  -- Perform the division
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lisa_flight_time_l235_23524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_mn_l235_23570

theorem unique_solution_mn (m n : ℕ) 
  (hm : m > 0) (hn : n > 0)
  (h : -(m^4 : ℤ) + 4*m^2 + 2^n*m^2 + 2^n + 5 = 0) : 
  m = 3 ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_mn_l235_23570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l235_23588

theorem polynomial_division_theorem :
  let p : Polynomial ℚ := X^5 - 25*X^3 + 16*X^2 - 9*X + 15
  let d : Polynomial ℚ := X - 3
  let q : Polynomial ℚ := X^4 + 3*X^3 - 16*X^2 - 32*X - 105
  let r : Polynomial ℚ := -270
  p = d * q + r := by
    -- The proof goes here
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_division_theorem_l235_23588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_not_less_than_three_l235_23503

def numbers : Finset ℕ := {0, 1, 2, 3}

def valid_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun (a, b) => a ≠ b ∧ a + b ≥ 3) (numbers.product numbers)

def total_pairs : Finset (ℕ × ℕ) :=
  Finset.filter (fun (a, b) => a ≠ b) (numbers.product numbers)

theorem probability_sum_not_less_than_three :
  (Finset.card valid_pairs : ℚ) / (Finset.card total_pairs : ℚ) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_sum_not_less_than_three_l235_23503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_decrease_percentage_l235_23525

theorem stock_decrease_percentage (x : ℝ) (h : x > 0) :
  let first_year_value := 1.3 * x
  let second_year_value := 0.9 * x
  let decrease_percentage := (first_year_value - second_year_value) / first_year_value * 100
  ∃ ε > 0, |decrease_percentage - 30.77| < ε := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_decrease_percentage_l235_23525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l235_23590

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  0 < t.B ∧ t.B < Real.pi / 2 ∧  -- B is acute
  Real.sin t.A / Real.sin t.B = 5 * t.c / (2 * t.b) ∧
  Real.sin t.B = Real.sqrt 7 / 4 ∧
  1/2 * t.a * t.c * Real.sin t.B = 5 * Real.sqrt 7 / 4

-- State the theorem
theorem triangle_side_b (t : Triangle) :
  triangle_conditions t → t.b = Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_b_l235_23590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_average_rainfall_l235_23579

/-- Represents the rainfall data for a storm -/
structure StormRainfall where
  first_30min : ℝ
  next_30min : ℝ
  last_60min : ℝ

/-- Calculates the total rainfall during the storm -/
noncomputable def total_rainfall (storm : StormRainfall) : ℝ :=
  storm.first_30min + storm.next_30min + storm.last_60min

/-- Calculates the average rainfall for the duration of the storm -/
noncomputable def average_rainfall (storm : StormRainfall) : ℝ :=
  total_rainfall storm / 2

/-- Theorem stating that for the given storm conditions, the average rainfall is 8 inches -/
theorem storm_average_rainfall :
  let storm : StormRainfall := {
    first_30min := 5,
    next_30min := 5 / 2,
    last_60min := 1 / 2
  }
  average_rainfall storm = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_storm_average_rainfall_l235_23579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cocoon_time_is_90_l235_23578

/-- Represents the time (in days) a butterfly species spends in various stages -/
structure ButterflyTime where
  total : ℚ
  cocoon : ℚ

/-- The butterfly species -/
inductive Species
  | A
  | B
  | C

/-- Given the total time and the relationship between larva and cocoon time,
    calculates the time spent in cocoon -/
def calculateCocoonTime (total : ℚ) : ℚ :=
  total / 4

/-- Returns the ButterflyTime for a given species -/
def getSpeciesTime (s : Species) : ButterflyTime :=
  match s with
  | Species.A => ⟨90, calculateCocoonTime 90⟩
  | Species.B => ⟨120, calculateCocoonTime 120⟩
  | Species.C => ⟨150, calculateCocoonTime 150⟩

/-- Theorem: The sum of cocoon times for all three species is 90 days -/
theorem total_cocoon_time_is_90 :
  (getSpeciesTime Species.A).cocoon +
  (getSpeciesTime Species.B).cocoon +
  (getSpeciesTime Species.C).cocoon = 90 := by
  -- Unfold definitions
  unfold getSpeciesTime calculateCocoonTime
  -- Simplify arithmetic
  simp [add_assoc]
  -- Check equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cocoon_time_is_90_l235_23578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l235_23511

/-- Calculate the simple interest rate given principal, time, and total interest -/
noncomputable def calculate_interest_rate (principal : ℝ) (time : ℝ) (total_interest : ℝ) : ℝ :=
  (total_interest * 100) / (principal * time)

/-- Theorem stating that given the problem conditions, the interest rate is 9% -/
theorem interest_rate_is_nine_percent 
  (principal : ℝ) 
  (time : ℝ) 
  (total_interest : ℝ) 
  (h1 : principal = 8935)
  (h2 : time = 5)
  (h3 : total_interest = 4020.75) :
  calculate_interest_rate principal time total_interest = 9 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval calculate_interest_rate 8935 5 4020.75

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_nine_percent_l235_23511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_7_5_l235_23577

def S : Set ℚ := {-30, -4, 0, 3, 5, 10}

theorem largest_quotient_is_7_5 :
  (∀ a b : ℚ, a ∈ S → b ∈ S → b ≠ 0 → (a / b : ℚ) ≤ 7.5) ∧
  (∃ x y : ℚ, x ∈ S ∧ y ∈ S ∧ y ≠ 0 ∧ (x / y : ℚ) = 7.5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_quotient_is_7_5_l235_23577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l235_23562

/-- Given points A(a,1), B(2,b), and C(3,4) in a Cartesian coordinate system,
    if the projections of OA and OB onto OC are equal,
    then the minimum value of a^2 + b^2 is 4/25. -/
theorem min_sum_squares (a b : ℝ) : 
  (3*a + 4 = 6 + 4*b) →
  (∀ x y : ℝ, x^2 + y^2 ≥ (4/25)) ∧
  (∃ x y : ℝ, x^2 + y^2 = (4/25) ∧ 3*x - 4*y = 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squares_l235_23562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_seven_increasing_triangles_l235_23564

/-- Represents a vertex in the hexagon -/
structure Vertex where
  label : Option ℝ

/-- Represents a triangle in the hexagon -/
structure Triangle where
  vertices : Fin 3 → Vertex

/-- Checks if a triangle's vertices are in counterclockwise increasing order -/
def isIncreasingTriangle (t : Triangle) : Bool :=
  sorry

/-- The hexagon configuration -/
structure HexagonConfig where
  triangles : Vector Triangle 24
  vertices : Vector ℝ 19

/-- The main theorem -/
theorem at_least_seven_increasing_triangles (config : HexagonConfig) :
  (config.triangles.toList.filter isIncreasingTriangle).length ≥ 7 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_seven_increasing_triangles_l235_23564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_not_even_divisible_l235_23522

theorem range_of_not_even_divisible :
  ∃ (start : ℕ), 
    start ≤ 100 ∧
    (∀ n ∈ Finset.range 6, ¬ ∃ (k : ℕ), k > 1 ∧ Even k ∧ (start + n) % k = 0) ∧
    (∀ m > start, ¬(∀ n ∈ Finset.range 6, ¬ ∃ (k : ℕ), k > 1 ∧ Even k ∧ (m + n) % k = 0)) ∧
    start = 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_not_even_divisible_l235_23522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_beneficial_l235_23500

/-- Represents the farm's feed purchasing strategy --/
structure FeedPurchaseStrategy where
  dailyNeed : ℝ  -- Daily feed need in kilograms
  basePrice : ℝ  -- Base price per kilogram in yuan
  storageCost : ℝ  -- Storage cost per kilogram per day in yuan
  transportFee : ℝ  -- Transportation fee per purchase in yuan
  discountThreshold : ℝ  -- Minimum purchase for discount in kilograms
  discountRate : ℝ  -- Discount rate (percentage of original price)

/-- Calculates the minimum average daily total cost without discount --/
noncomputable def minCostWithoutDiscount (s : FeedPurchaseStrategy) : ℝ :=
  s.dailyNeed * s.basePrice + 2 * Real.sqrt (s.dailyNeed * s.storageCost * s.transportFee)

/-- Calculates the minimum average daily total cost with discount --/
noncomputable def minCostWithDiscount (s : FeedPurchaseStrategy) : ℝ :=
  let discountDays := s.discountThreshold / s.dailyNeed
  (s.dailyNeed * s.basePrice * s.discountRate) +
  (s.transportFee / discountDays) +
  (s.dailyNeed * s.storageCost * (discountDays - 1) / 2)

/-- The main theorem to prove --/
theorem discount_is_beneficial (s : FeedPurchaseStrategy) :
  s.dailyNeed = 200 ∧
  s.basePrice = 1.8 ∧
  s.storageCost = 0.03 ∧
  s.transportFee = 300 ∧
  s.discountThreshold = 5000 ∧
  s.discountRate = 0.85 →
  minCostWithDiscount s < minCostWithoutDiscount s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_is_beneficial_l235_23500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_processing_cost_optimization_l235_23505

-- Define the processing cost function
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 300*x + 64800

-- Define the average processing cost function
noncomputable def g (x : ℝ) : ℝ := f x / x

-- Theorem statement
theorem processing_cost_optimization :
  (∃ (x : ℝ), 30 ≤ x ∧ x ≤ 400 ∧
    (∀ (y : ℝ), 30 ≤ y ∧ y ≤ 400 → f x ≤ f y) ∧
    f x = 19800 ∧
    x = 300) ∧
  (∃ (x : ℝ), 30 ≤ x ∧ x ≤ 400 ∧
    (∀ (y : ℝ), 30 ≤ y ∧ y ≤ 400 → g x ≤ g y) ∧
    g x = 60 ∧
    x = 360) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_processing_cost_optimization_l235_23505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l235_23558

-- Define the constants
noncomputable def a : ℝ := (4 : ℝ) ^ (0.3 : ℝ)
noncomputable def b : ℝ := (1/2 : ℝ) ^ (-(0.9 : ℝ))
noncomputable def c : ℝ := 2 * (Real.log 2 / Real.log 6)

-- State the theorem
theorem order_relation : c < a ∧ a < b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_relation_l235_23558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_for_48_cubic_inches_l235_23504

/-- The weight (in ounces) of a substance with a given volume (in cubic inches),
    given the direct proportionality between volume and weight -/
noncomputable def weight_from_volume (k : ℝ) (v : ℝ) : ℝ := v / k

/-- The volume (in cubic inches) of a substance with a given weight (in ounces),
    given the direct proportionality between volume and weight -/
noncomputable def volume_from_weight (k : ℝ) (w : ℝ) : ℝ := k * w

theorem weight_for_48_cubic_inches (k : ℝ) (h : volume_from_weight k 84 = 36) :
  weight_from_volume k 48 = 112 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_for_48_cubic_inches_l235_23504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_inequality_trapezoid_equality_l235_23548

structure Trapezoid where
  a : ℝ
  b : ℝ
  n : ℕ
  h_positive : 0 < a ∧ 0 < b

noncomputable def x_n (t : Trapezoid) : ℝ := t.a * t.b / (t.a + t.n * t.b)

theorem trapezoid_inequality (t : Trapezoid) : 
  x_n t ≤ (t.a + t.n * t.b) / (4 * t.n) := by
  sorry

theorem trapezoid_equality (t : Trapezoid) :
  x_n t = t.a * t.b / (t.a + t.n * t.b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_inequality_trapezoid_equality_l235_23548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_addition_for_divisibility_l235_23526

theorem least_addition_for_divisibility : ∃! x : ℕ, 
  (∀ d : ℕ, d ∈ ({5, 7, 11, 13} : Set ℕ) → (1789 + x) % d = 0) ∧
  (∀ y : ℕ, y < x → ∃ d : ℕ, d ∈ ({5, 7, 11, 13} : Set ℕ) ∧ (1789 + y) % d ≠ 0) ∧
  x = 3216 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_addition_for_divisibility_l235_23526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_2_5_eq_l235_23550

def non_negative_integers_less_than_2_5 : Set ℕ :=
  {n : ℕ | (n : ℝ) < 2.5}

theorem non_negative_integers_less_than_2_5_eq :
  non_negative_integers_less_than_2_5 = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_negative_integers_less_than_2_5_eq_l235_23550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_projectile_speed_approx_l235_23596

/-- The speed of the second projectile given the initial conditions --/
noncomputable def second_projectile_speed (initial_distance : ℝ) (first_speed : ℝ) (time_minutes : ℝ) : ℝ :=
  let time_hours := time_minutes / 60
  let first_distance := first_speed * time_hours
  let second_distance := initial_distance - first_distance
  second_distance / time_hours

/-- Theorem stating the speed of the second projectile under given conditions --/
theorem second_projectile_speed_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |second_projectile_speed 1386 445 84 - 545| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_projectile_speed_approx_l235_23596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_angle_measure_l235_23560

/-- An obtuse isosceles triangle with specific side length properties -/
structure ObtuseSideProductTriangle where
  -- The length of each congruent side
  a : ℝ
  -- The length of the base
  b : ℝ
  -- The height to the base
  h : ℝ
  -- The measure of each base angle in radians
  θ : ℝ
  -- The measure of the vertex angle in radians
  φ : ℝ
  -- Conditions
  obtuse : φ > Real.pi / 2
  isosceles : a > 0
  side_product : a^2 = 3 * b * h
  base_relation : b = 2 * a * Real.cos θ
  height_relation : h = a * Real.sin θ
  angle_sum : φ + 2 * θ = Real.pi

/-- The vertex angle of an ObtuseSideProductTriangle is approximately 160.53 degrees -/
theorem vertex_angle_measure (t : ObtuseSideProductTriangle) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |t.φ - 160.53 * (Real.pi / 180)| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_angle_measure_l235_23560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_proof_l235_23532

theorem trig_ratio_proof (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) : 
  (Real.sin α)^2 + Real.sin (2*α) / ((Real.cos α)^2 + Real.cos (2*α)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_ratio_proof_l235_23532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_inequality_l235_23535

open Real

/-- The function f(x) = (x^2 + 1) / x -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / x

/-- The function g(x) = x / e^x -/
noncomputable def g (x : ℝ) : ℝ := x / (exp x)

/-- The theorem stating the minimum value of k that satisfies the inequality -/
theorem min_k_for_inequality :
  ∃ (k : ℝ), k > 0 ∧ 
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → g x₁ / k ≤ f x₂ / (k + 1)) ∧
  (∀ (k' : ℝ), k' > 0 → 
    (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → g x₁ / k' ≤ f x₂ / (k' + 1)) → 
    k' ≥ k) ∧
  k = 1 / (2 * exp 1 - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_inequality_l235_23535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_divisibility_l235_23521

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate the polynomial at a given value -/
def CubicPolynomial.eval (f : CubicPolynomial) (x : ℂ) : ℂ :=
  x^3 + f.a * x^2 + f.b * x + f.c

/-- The property that one root is the product of the other two -/
def HasProductRoot (f : CubicPolynomial) : Prop :=
  ∃ (x y z : ℂ), f.eval x = 0 ∧ f.eval y = 0 ∧ f.eval z = 0 ∧ x = y * z

/-- The main theorem -/
theorem cubic_divisibility (f : CubicPolynomial) (h : HasProductRoot f) :
  ∃ k : ℤ, 2 * (f.eval (-1)).re = k * ((f.eval 1).re + (f.eval (-1)).re - 2 * (1 + (f.eval 0).re)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_divisibility_l235_23521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_alpha_two_implies_expression_value_l235_23502

theorem tan_half_alpha_two_implies_expression_value (α : ℝ) :
  Real.tan (α / 2) = 2 →
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_alpha_two_implies_expression_value_l235_23502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_side_length_l235_23565

-- Define the triangles
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define similarity relation
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the length of a side
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem triangle_similarity_side_length 
  (ABC DEF : Triangle) 
  (h1 : similar ABC DEF) 
  (h2 : side_length ABC.A ABC.B / side_length DEF.A DEF.B = 1/2) 
  (h3 : side_length ABC.B ABC.C = 2) : 
  side_length DEF.B DEF.C = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_similarity_side_length_l235_23565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_property_l235_23529

def data_set : List ℝ := [9, 10, 12, 15, 0, 17, 0, 22, 26]

def is_valid_data_set (xs : List ℝ) : Prop :=
  xs.length = 9 ∧ 
  xs[0]! = 9 ∧ xs[1]! = 10 ∧ xs[2]! = 12 ∧ xs[3]! = 15 ∧ 
  xs[5]! = 17 ∧ xs[7]! = 22 ∧ xs[8]! = 26

def is_sorted (xs : List ℝ) : Prop :=
  ∀ i j, i < j → i < xs.length → j < xs.length → xs[i]! ≤ xs[j]!

def median (xs : List ℝ) : ℝ := xs[xs.length / 2]!

def percentile_75 (xs : List ℝ) : ℝ := xs[6]!

theorem data_set_property (xs : List ℝ) (hvalid : is_valid_data_set xs) 
    (hsorted : is_sorted xs) (hmedian : median xs = 16) (hpercentile : percentile_75 xs = 20) :
  xs[4]! + xs[6]! = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_data_set_property_l235_23529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l235_23584

theorem range_of_sum (x y : ℝ) (h : (2 : ℝ)^x + (2 : ℝ)^y = 1) : 
  x + y ≤ -2 ∧ (x + y = -2 ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_sum_l235_23584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_APB_l235_23556

-- Define the circles
noncomputable def circle_C (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 1

noncomputable def circle_M (x y θ : ℝ) : Prop := ((x - 3 - 3 * Real.cos θ)^2 + (y - 3 * Real.sin θ)^2) = 1

-- Define the point P on circle M
noncomputable def point_P (x y θ : ℝ) : Prop := circle_M x y θ

-- Define the tangent points A and B on circle C
noncomputable def point_A (x y : ℝ) : Prop := circle_C x y
noncomputable def point_B (x y : ℝ) : Prop := circle_C x y

-- Define the angle APB
noncomputable def angle_APB (xp yp θ xa ya xb yb : ℝ) : ℝ :=
  Real.arccos ((xa - xp) * (xb - xp) + (ya - yp) * (yb - yp)) /
    (Real.sqrt ((xa - xp)^2 + (ya - yp)^2) * Real.sqrt ((xb - xp)^2 + (yb - yp)^2))

-- Theorem statement
theorem max_angle_APB :
  ∀ xp yp θ xa ya xb yb,
    point_P xp yp θ →
    point_A xa ya →
    point_B xb yb →
    angle_APB xp yp θ xa ya xb yb ≤ π / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_angle_APB_l235_23556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_games_l235_23571

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) (h1 : n = 10) (h2 : total_games = 45) :
  (n.choose 2 = total_games) → ∀ i j : Fin n, i ≠ j → 1 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_tournament_games_l235_23571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_l235_23536

theorem three_digit_numbers_count : 
  Finset.card (Finset.filter (λ n => 300 ≤ n ∧ n < 306) (Finset.range 1000)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_numbers_count_l235_23536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l235_23591

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) + 2^(x+1) + 3

theorem f_range :
  (∀ x : ℝ, f x > 3) ∧ 
  (∀ y : ℝ, y > 3 → ∃ x : ℝ, f x = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l235_23591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_plane_perpendicular_lines_in_perpendicular_planes_l235_23501

-- Define basic geometric objects
variable (Point Line Plane : Type)

-- Define geometric relationships
variable (perpendicular : Line → Line → Prop)
variable (perpendicularPL : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (mutuallyPerpendicular : Plane → Plane → Prop)
variable (onPlane : Point → Plane → Prop)

-- Proposition 1
theorem unique_perpendicular_plane (p : Point) (l : Line) :
  ∃! α : Plane, perpendicularPL l α ∧ onPlane p α :=
sorry

-- Proposition 4
theorem perpendicular_lines_in_perpendicular_planes (α β : Plane) (l : Line) :
  mutuallyPerpendicular α β →
  contains α l →
  ∃ (S : Set Line), Set.Infinite S ∧ ∀ m ∈ S, contains β m ∧ perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_perpendicular_plane_perpendicular_lines_in_perpendicular_planes_l235_23501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_diagonal_maximizes_area_optimal_diagonal_approx_l235_23515

/-- The perimeter of the pentagon --/
noncomputable def perimeter : ℝ := 255.3

/-- The diagonal length of the pentagon --/
noncomputable def diagonal (x : ℝ) : ℝ := x

/-- The side length of the rectangle perpendicular to the diagonal --/
noncomputable def rect_side (x : ℝ) : ℝ := (perimeter - (1 + Real.sqrt 2) * x) / 2

/-- The area of the pentagon as a function of the diagonal length --/
noncomputable def area (x : ℝ) : ℝ := (perimeter * x) / 2 - ((1 + 2 * Real.sqrt 2) * x^2) / 4

/-- The optimal diagonal length that maximizes the area --/
noncomputable def optimal_diagonal : ℝ := perimeter * 2 / (1 + 2 * Real.sqrt 2)

theorem optimal_diagonal_maximizes_area :
  ∀ x : ℝ, x > 0 → area x ≤ area optimal_diagonal :=
by sorry

theorem optimal_diagonal_approx :
  abs (optimal_diagonal - 66.69) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_diagonal_maximizes_area_optimal_diagonal_approx_l235_23515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l235_23546

theorem triangle_side_length (A B C : ℝ) (b c : ℝ) :
  Real.cos (3 * A) + Real.cos (3 * B) + Real.cos (3 * C) = 1 →
  b = 10 →
  c = 13 →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  ∃ a : ℝ, a ≤ Real.sqrt 399 ∧ 
    a^2 = b^2 + c^2 - 2*b*c*(Real.cos A) ∧
    ∀ a' : ℝ, a'^2 = b^2 + c^2 - 2*b*c*(Real.cos A) → a' ≤ a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l235_23546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_distance_correct_l235_23508

structure Plane :=
  (Point : Type)
  (Line : Type)
  (on_line : Point → Line → Prop)
  (distance : Point → Point → ℝ)
  (perpendicular_foot : Point → Line → Point)
  (perpendicular_bisector : Point → Point → Line)
  (intersect : Line → Line → Point)

noncomputable def minimize_max_distance (π : Plane) (A B : π.Point) (L : π.Line) : π.Point :=
  let B' := π.perpendicular_foot B L
  let A' := π.perpendicular_foot A L
  let X := π.intersect (π.perpendicular_bisector A B) L
  if π.distance B B' > π.distance A B' then B' else X

theorem minimize_max_distance_correct (π : Plane) (A B : π.Point) (L : π.Line) :
  let P := minimize_max_distance π A B L
  (∀ Q : π.Point, π.on_line Q L →
    max (π.distance A P) (π.distance B P) ≤ max (π.distance A Q) (π.distance B Q)) := by
  sorry

#check minimize_max_distance
#check minimize_max_distance_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_max_distance_correct_l235_23508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l235_23563

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on the hyperbola -/
def isOnHyperbola (h : Hyperbola) (p : Point) : Prop :=
  p.x^2 / h.a^2 - p.y^2 / h.b^2 = 1

/-- Theorem: Eccentricity of a specific hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) (F₁ F₂ A B : Point) :
  -- The hyperbola equation
  (∀ x y, x^2 / h.a^2 - y^2 / h.b^2 = 1 → isOnHyperbola h (Point.mk x y)) →
  -- F₁ and F₂ are the foci
  (F₁.x < 0 ∧ F₂.x > 0 ∧ F₁.y = 0 ∧ F₂.y = 0) →
  -- Line through F₁ at 60° angle
  (B.y - F₁.y) / (B.x - F₁.x) = Real.tan (π / 3) →
  -- A is on the y-axis
  A.x = 0 →
  -- B is on the right branch of the hyperbola
  isOnHyperbola h B ∧ B.x > 0 →
  -- A bisects F₁B
  A.x - F₁.x = (B.x - F₁.x) / 2 ∧ A.y - F₁.y = (B.y - F₁.y) / 2 →
  -- The eccentricity is 2 + √3
  h.a / Real.sqrt (h.a^2 - h.b^2) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l235_23563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_fourth_l235_23576

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = (bx+c)/(ax^2+1) -/
noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ :=
  (b * x + c) / (a * x^2 + 1)

theorem odd_function_implies_a_eq_one_fourth (a b c : ℝ) :
  IsOdd (f a b c) →
  (∀ x, f a b c (-2) ≤ f a b c x ∧ f a b c x ≤ f a b c 2) →
  a = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_one_fourth_l235_23576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l235_23528

structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

structure Point where
  coords : ℝ × ℝ

def is_acute_angled (t : Triangle) : Prop :=
  sorry

def is_interior_point (p : Point) (t : Triangle) : Prop :=
  sorry

def reflect_point (p : Point) (line : (ℝ × ℝ) × (ℝ × ℝ)) : Point :=
  sorry

def is_orthocentre (p : Point) (t : Triangle) : Prop :=
  sorry

noncomputable def angle_between (a b c : Point) : ℝ :=
  sorry

noncomputable def max_angle (a b c : Point) : ℝ :=
  sorry

theorem largest_angle_is_120_degrees 
  (ABC : Triangle) 
  (P : Point) 
  (P_A P_B P_C : Point) :
  is_acute_angled ABC →
  is_interior_point P ABC →
  P_A = reflect_point P (ABC.B, ABC.C) →
  P_B = reflect_point P (ABC.C, ABC.A) →
  P_C = reflect_point P (ABC.A, ABC.B) →
  is_orthocentre P (Triangle.mk P_A.coords P_B.coords P_C.coords) →
  max_angle P_A P_B P_C = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_angle_is_120_degrees_l235_23528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_perfect_square_divisor_15_factorial_l235_23534

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def isPerfectSquare (n : ℕ) : Bool := 
  match (n.sqrt * n.sqrt) with
  | m => m == n

def countDivisors (n : ℕ) : ℕ := (List.range n).filter (· ∣ n) |>.length

def countPerfectSquareDivisors (n : ℕ) : ℕ := 
  (List.range n).filter (λ x => (x ∣ n) && isPerfectSquare x) |>.length

theorem probability_perfect_square_divisor_15_factorial :
  (countPerfectSquareDivisors (factorial 15) : ℚ) / 
  (countDivisors (factorial 15) : ℚ) = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_perfect_square_divisor_15_factorial_l235_23534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bottle_volume_l235_23551

/-- Represents the volume of milk in a bottle -/
structure MilkBottle where
  volume : ℚ
  unit : String

/-- Converts milliliters to liters -/
def mlToL (ml : ℚ) : ℚ := ml / 1000

/-- Converts liters to milliliters -/
def lToMl (l : ℚ) : ℚ := l * 1000

theorem second_bottle_volume 
  (bottle1 : MilkBottle)
  (bottle2 : MilkBottle)
  (bottle3 : MilkBottle)
  (h1 : bottle1.volume = 2 ∧ bottle1.unit = "L")
  (h2 : bottle3.volume = 250 ∧ bottle3.unit = "ml")
  (h_total : bottle1.volume + bottle2.volume + mlToL bottle3.volume = 3) :
  lToMl bottle2.volume = 750 := by
  sorry

#eval lToMl (3 - 2 - mlToL 250)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_bottle_volume_l235_23551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l235_23553

-- Define the board
def Board := Fin 8 × Fin 8

-- Define a piece
structure Piece where
  position : Board

-- Define the game state
structure GameState where
  pieces : List Piece
  currentPlayer : Nat

-- Define valid moves
def validMove : Board → Board → Prop
  | (x₁, y₁), (x₂, y₂) => (x₁ = x₂ ∧ y₁ < y₂) ∨ (y₁ = y₂ ∧ x₁ < x₂)

-- Define the winning condition
def isWinningPosition (p : Piece) : Prop :=
  p.position = (7, 7)

-- Define the initial game state
def initialState : GameState :=
  { pieces := [{ position := (0, 0) }, { position := (2, 2) }],
    currentPlayer := 0 }

-- Theorem stating that the second player has a winning strategy
theorem second_player_wins :
  ∃ (strategy : GameState → Board × Board),
    ∀ (game : GameState),
      game.currentPlayer = 1 →
      ∃ (newGame : GameState),
        (validMove (strategy game).1 (strategy game).2 ∧
         isWinningPosition { position := (strategy game).2 }) ∨
        (∀ (opponentMove : Board × Board),
           validMove opponentMove.1 opponentMove.2 →
           ∃ (nextMove : Board × Board),
             validMove nextMove.1 nextMove.2 ∧
             isWinningPosition { position := nextMove.2 }) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l235_23553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_sum_l235_23507

noncomputable def f (x : ℝ) : ℝ := (3 * x^2 + 5 * x + 1) / (x + 2)

def slant_asymptote (x : ℝ) : ℝ := 3 * x - 1

theorem slant_asymptote_and_sum :
  (∀ ε > 0, ∃ M, ∀ x > M, |f x - slant_asymptote x| < ε) ∧
  (3 + (-1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slant_asymptote_and_sum_l235_23507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_of_nine_l235_23598

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- State the theorem
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

-- Additional lemmas to support the main theorem
lemma arithmetic_sqrt_nonnegative (x : ℝ) (h : x ≥ 0) : arithmetic_sqrt x ≥ 0 := by
  sorry

lemma arithmetic_sqrt_squared (x : ℝ) (h : x ≥ 0) : (arithmetic_sqrt x) ^ 2 = x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sqrt_of_nine_l235_23598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spinsters_and_cats_l235_23589

/-- Given a ratio of spinsters to cats and the number of spinsters, 
    calculate the difference between the number of cats and spinsters. -/
theorem spinsters_and_cats (ratio_spinsters : ℚ) (ratio_cats : ℚ) (num_spinsters : ℕ) : 
  ratio_spinsters * num_spinsters * ratio_cats = 
  ratio_cats * num_spinsters * ratio_spinsters → 
  ratio_spinsters = 2 → 
  ratio_cats = 7 → 
  num_spinsters = 22 → 
  (ratio_cats * num_spinsters) / (ratio_spinsters) - num_spinsters = 55 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spinsters_and_cats_l235_23589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l235_23514

/-- The sum of an arithmetic sequence with first term 18, last term 58, and common difference 4 is 418 -/
theorem arithmetic_sequence_sum : 
  let a₁ : ℤ := 18  -- first term
  let aₙ : ℤ := 58  -- last term
  let d : ℤ := 4    -- common difference
  let n : ℤ := (aₙ - a₁) / d + 1  -- number of terms
  n * (a₁ + aₙ) / 2 = 418 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l235_23514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_equation_interest_difference_l235_23566

/-- The principal amount that satisfies the given conditions -/
noncomputable def principal : ℝ := 20 / (1.05^2 * 1.06^2 - 1 - 0.22)

/-- Theorem stating that the principal satisfies the equation -/
theorem principal_satisfies_equation :
  principal * (1.05^2 * 1.06^2 - 1 - 0.22) = 20 := by
  sorry

/-- The compound interest rate for the first year (semi-annual) -/
def first_year_rate : ℝ := 0.05

/-- The compound interest rate for the second year (semi-annual) -/
def second_year_rate : ℝ := 0.06

/-- The simple interest rate for the first year -/
def first_year_simple_rate : ℝ := 0.10

/-- The simple interest rate for the second year -/
def second_year_simple_rate : ℝ := 0.12

/-- The time period in years -/
def time_period : ℕ := 2

/-- The number of compounding periods per year -/
def compounding_periods : ℕ := 2

/-- Theorem stating that the difference between compound and simple interest is 20 -/
theorem interest_difference :
  principal * (1.05^2 * 1.06^2 - 1) - principal * (first_year_simple_rate + second_year_simple_rate) = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_principal_satisfies_equation_interest_difference_l235_23566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l235_23509

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a hyperbola -/
structure Hyperbola where
  leftFocus : Point
  rightFocus : Point
  a : ℝ
  b : ℝ

/-- Defines the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ :=
  Real.sqrt (1 + h.b^2 / h.a^2)

/-- States that a line is perpendicular to the real axis -/
def perpendicularToRealAxis (p q : Point) : Prop :=
  p.x = q.x

/-- States that a point is on the y-axis -/
def onYAxis (p : Point) : Prop :=
  p.x = 0

/-- States that two line segments are perpendicular -/
def perpendicular (p1 p2 p3 p4 : Point) : Prop :=
  (p2.x - p1.x) * (p4.x - p3.x) + (p2.y - p1.y) * (p4.y - p3.y) = 0

/-- Main theorem -/
theorem hyperbola_eccentricity
  (h : Hyperbola)
  (A B C : Point)
  (hAB : perpendicularToRealAxis A B)
  (hF2AB : perpendicularToRealAxis h.rightFocus A)
  (hC : onYAxis C)
  (hBF1C : perpendicular B h.leftFocus A C) :
  eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l235_23509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l235_23537

noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => fun x => Real.sqrt (x^2 + 48)  -- Adding case for 0
  | n + 1 => fun x => Real.sqrt (x^2 + 6 * f n x)

theorem unique_solution (n : ℕ) (h : n ≥ 1) :
  ∃! x : ℝ, f n x = 2 * x ∧ x > 0 :=
by sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l235_23537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_real_l235_23561

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem range_of_f_is_real : Set.range f = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_is_real_l235_23561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepts_of_line_l235_23541

/-- Represents a line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-intercept of a line -/
noncomputable def y_intercept (l : Line) : ℝ × ℝ :=
  (0, l.c / l.b)

/-- The x-intercept of a line -/
noncomputable def x_intercept (l : Line) : ℝ × ℝ :=
  (l.c / l.a, 0)

/-- The line 4x + 7y = 28 -/
def line : Line :=
  { a := 4, b := 7, c := 28 }

theorem intercepts_of_line :
  y_intercept line = (0, 4) ∧ x_intercept line = (7, 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercepts_of_line_l235_23541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_semicircle_is_right_angle_l235_23559

/-- A circle with a diameter and a point on its circumference -/
structure CircleWithDiameterAndPoint where
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The radius of the circle -/
  radius : ℝ
  /-- One endpoint of the diameter -/
  A : ℝ × ℝ
  /-- The other endpoint of the diameter -/
  B : ℝ × ℝ
  /-- A point on the circle -/
  C : ℝ × ℝ
  /-- A is on the circle -/
  h_A_on_circle : (A.1 - center.1)^2 + (A.2 - center.2)^2 = radius^2
  /-- B is on the circle -/
  h_B_on_circle : (B.1 - center.1)^2 + (B.2 - center.2)^2 = radius^2
  /-- C is on the circle -/
  h_C_on_circle : (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2
  /-- AB is a diameter -/
  h_AB_diameter : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * radius^2

/-- The theorem stating that the angle ACB is a right angle -/
theorem angle_in_semicircle_is_right_angle (circle : CircleWithDiameterAndPoint) :
  let angle_ACB := Real.arccos ((circle.A.1 - circle.C.1) * (circle.B.1 - circle.C.1) + 
                                (circle.A.2 - circle.C.2) * (circle.B.2 - circle.C.2)) / 
                               (((circle.A.1 - circle.C.1)^2 + (circle.A.2 - circle.C.2)^2)^(1/2) * 
                                ((circle.B.1 - circle.C.1)^2 + (circle.B.2 - circle.C.2)^2)^(1/2))
  angle_ACB = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_semicircle_is_right_angle_l235_23559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l235_23597

-- Define a, b, and c
noncomputable def a : ℝ := (0.5 : ℝ) ^ (3.4 : ℝ)
noncomputable def b : ℝ := Real.log 4.3 / Real.log 0.5
noncomputable def c : ℝ := Real.log 6.7 / Real.log 0.5

-- Theorem statement
theorem order_of_abc : c < b ∧ b < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_abc_l235_23597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l235_23547

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x < -1 then 4 * x + 8
  else if x < 5 then 5 * x - 10
  else x^2 - 5

-- Define the set of solutions
def solutions : Set ℝ := {-5/4, 13/5, 2 * Real.sqrt 2}

-- Theorem statement
theorem g_solutions :
  ∀ x : ℝ, g x = 3 ↔ x ∈ solutions :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l235_23547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_outcome_satisfies_conditions_l235_23530

-- Define the type for students
inductive Student : Type
  | A | B | C | D | E

-- Define the type for positions
def Position := Fin 5

-- Define the type for predictions
def Prediction := List Student

-- Define the first prediction
def firstPrediction : Prediction := [Student.A, Student.B, Student.C, Student.D, Student.E]

-- Define the second prediction
def secondPrediction : Prediction := [Student.D, Student.A, Student.E, Student.C, Student.B]

-- Define the outcome
def outcome : Prediction := [Student.E, Student.D, Student.A, Student.C, Student.B]

-- Function to check if two students are consecutive in a prediction
def areConsecutive (s1 s2 : Student) (pred : Prediction) : Prop :=
  ∃ i : Nat, pred.get? i = some s1 ∧ pred.get? (i + 1) = some s2

-- Function to check if a student is in the correct position
def isInCorrectPosition (s : Student) (pos : Nat) (pred outcome : Prediction) : Prop :=
  pred.get? pos = some s ∧ outcome.get? pos = some s

-- Theorem statement
theorem outcome_satisfies_conditions : 
  (∀ s : Student, ∀ pos : Nat, pos < 5 → ¬isInCorrectPosition s pos firstPrediction outcome) ∧
  (∀ s1 s2 : Student, areConsecutive s1 s2 firstPrediction → ¬areConsecutive s1 s2 outcome) ∧
  (∃ s1 s2 : Student, ∃ pos1 pos2 : Nat, 
    pos1 < 5 ∧ pos2 < 5 ∧
    isInCorrectPosition s1 pos1 secondPrediction outcome ∧
    isInCorrectPosition s2 pos2 secondPrediction outcome ∧
    s1 ≠ s2) ∧
  (∃ s1 s2 s3 s4 : Student, 
    areConsecutive s1 s2 secondPrediction ∧
    areConsecutive s3 s4 secondPrediction ∧
    areConsecutive s1 s2 outcome ∧
    areConsecutive s3 s4 outcome ∧
    s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_outcome_satisfies_conditions_l235_23530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cosine_phase_l235_23542

-- Define the function f as noncomputable
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (3 * x + φ)

-- State the theorem
theorem odd_cosine_phase (φ : ℝ) (h1 : 0 ≤ φ) (h2 : φ ≤ π) :
  (∀ x, f φ (-x) = -f φ x) → φ = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_cosine_phase_l235_23542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_result_l235_23510

theorem sequence_result : 
  (((10^8 : ℚ) / 3 * 7 / 3 * 7)^8 : ℚ) = (490/9)^8 := by
  sorry

#eval (((10^8 : ℚ) / 3 * 7 / 3 * 7)^8 : ℚ) == (490/9)^8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_result_l235_23510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_measurement_l235_23581

def WeightSet : Finset Nat := {1, 4, 9}

def maxWeight (s : Finset Nat) : Nat :=
  s.sum id

def possibleWeights (s : Finset Nat) : Finset Nat :=
  s.powerset.image (λ subset => subset.sum id)

theorem weight_measurement (s : Finset Nat := WeightSet) :
  (maxWeight s = 14) ∧ (possibleWeights s).card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_measurement_l235_23581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_l235_23554

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_8_value (a : ℕ → ℚ) :
  arithmetic_sequence (λ n ↦ 1 / (a n + 2)) →
  a 3 = -11/6 →
  a 5 = -13/7 →
  a 8 = -32/17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_8_value_l235_23554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l235_23572

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x^2 + a*x + 1)

-- Define the domain condition
def domain_condition (a : ℝ) : Prop := ∀ x, x^2 + a*x + 1 > 0

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := x^2 - 2*x + a*(2-a) < 0

-- State the theorem
theorem solution_set (a : ℝ) (h : domain_condition a) :
  ((-2 < a ∧ a < 1) ∧ (∀ x, inequality a x ↔ (a < x ∧ x < 2 - a))) ∨
  (a = 1 ∧ ∀ x, ¬ inequality a x) ∨
  ((1 < a ∧ a < 2) ∧ (∀ x, inequality a x ↔ (2 - a < x ∧ x < a))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l235_23572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_greater_than_two_l235_23593

theorem log_sum_greater_than_two (x y a : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < a) (h4 : a < 1) :
  let m := Real.log x / Real.log a + Real.log y / Real.log a
  m > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_greater_than_two_l235_23593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l235_23540

noncomputable section

open Real

theorem trigonometric_identities (α : ℝ) : 
  (sin α)^4 + (tan α)^2 * (cos α)^4 + (cos α)^2 = 1 ∧ 
  (cos (π + α) * sin (α + 2*π)) / (sin (-α - π) * cos (-π - α)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l235_23540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_differential_existence_u_satisfies_exactness_l235_23599

/-- Given a differential form (2x - 3y^2 + 1)dx + (2 - 6xy)dy, 
    prove the existence of a function u(x,y) satisfying the exactness condition. -/
theorem exact_differential_existence :
  ∃ u : ℝ → ℝ → ℝ,
    (∀ x y : ℝ, deriv (fun x => u x y) x = 2*x - 3*y^2 + 1) ∧
    (∀ x y : ℝ, deriv (fun y => u x y) y = 2 - 6*x*y) := by
  sorry

/-- The function u(x,y) satisfying the exactness condition for the given differential form. -/
def u (x y : ℝ) : ℝ := x^2 - 3*y^2*x + x + 2*y

/-- Prove that the function u(x,y) satisfies the exactness condition. -/
theorem u_satisfies_exactness :
  (∀ x y : ℝ, deriv (fun x => u x y) x = 2*x - 3*y^2 + 1) ∧
  (∀ x y : ℝ, deriv (fun y => u x y) y = 2 - 6*x*y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_differential_existence_u_satisfies_exactness_l235_23599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_set_characterization_l235_23557

/-- A set of natural numbers is interesting if it satisfies the given conditions -/
def is_interesting (p : ℕ) (s : Finset ℕ) : Prop :=
  Nat.Prime p ∧ 
  s.card = p + 2 ∧ 
  ∀ (t : Finset ℕ), t ⊆ s → t.card = p → 
    ∀ x ∈ s \ t, (t.sum id) % x = 0

/-- The characterization of interesting sets -/
theorem interesting_set_characterization (p : ℕ) (s : Finset ℕ) :
  is_interesting p s ↔ 
  (∃ a : ℕ, s = (Finset.range (p + 1)).image (λ _ => a) ∪ {p * a}) ∨
  (∃ a : ℕ, s = (Finset.range (p + 2)).image (λ _ => a)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interesting_set_characterization_l235_23557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l235_23520

noncomputable def M : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

noncomputable def g (x : ℝ) : ℝ := (1/4) * x + 1/x

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + b/x + c

theorem max_value_of_f (b c : ℝ) :
  (∃ x₀ ∈ M, ∀ x ∈ M, f b c x ≥ f b c x₀ ∧ g x ≥ g x₀ ∧ f b c x₀ = g x₀) →
  (∃ x ∈ M, ∀ y ∈ M, f b c x ≥ f b c y) →
  (∃ x ∈ M, f b c x = 5) ∧ (∀ x ∈ M, f b c x ≤ 5) := by
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l235_23520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_g_derivative_l235_23506

-- Function 1
noncomputable def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem f_derivative :
  deriv f = fun x => -x * Real.sin x := by sorry

-- Function 2
noncomputable def g (x : ℝ) : ℝ := x / (Real.exp x - 1)

theorem g_derivative (x : ℝ) (h : x ≠ 0) :
  deriv g x = (Real.exp x * (1 - x) - 1) / (Real.exp x - 1)^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_g_derivative_l235_23506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l235_23549

theorem expression_evaluation : 
  ((2⁻¹ - 5⁻¹ : ℚ) * 6)⁻¹ = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l235_23549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l235_23539

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + 2 * (x - 6))

theorem domain_of_f :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x < 5 ∨ x > 5} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l235_23539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_green_light_duration_l235_23512

/-- Represents the duration of traffic light colors in a cycle -/
structure TrafficLightCycle where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Calculates the duration of a specific color in a 24-hour day given a traffic light cycle -/
noncomputable def colorDuration (cycle : TrafficLightCycle) (color : ℕ) : ℝ :=
  24 * (color : ℝ) / ((cycle.green + cycle.yellow + cycle.red) : ℝ)

/-- The theorem stating that the green light duration is 14.4 hours in a day -/
theorem green_light_duration (cycle : TrafficLightCycle) 
  (h : cycle = { green := 6, yellow := 1, red := 3 }) : 
  colorDuration cycle cycle.green = 14.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_green_light_duration_l235_23512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_chord_product_l235_23538

/-- The radius of the semicircle -/
def radius : ℝ := 3

/-- The number of equally spaced points on the semicircle -/
def num_points : ℕ := 8

/-- Complex number representing rotation by 2π/18 -/
noncomputable def ω : ℂ := Complex.exp (2 * Real.pi * Complex.I / 18)

/-- The product of lengths of chords from endpoints to equally spaced points -/
def chord_product : ℝ := (radius ^ (2 * num_points)) * 9

/-- Theorem stating that the chord product equals 472392 -/
theorem semicircle_chord_product :
  chord_product = 472392 := by
  sorry

#eval chord_product -- This will evaluate to 472392

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_chord_product_l235_23538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_possible_placement_l235_23574

/-- Represents the score in a single fishing category -/
def CategoryScore := Nat

/-- Represents the total score across all fishing categories -/
def TotalScore := Nat

/-- Represents a participant's placement in the competition -/
def Placement := Nat

/-- The number of categories in the fishing competition -/
def num_categories : Nat := 3

/-- The number of awarded places in each category -/
def places_per_category : Nat := 3

/-- Calculate the total score for a participant given their placements in each category -/
def calculate_total_score (placements : List Nat) : Nat :=
  placements.sum

/-- The specific placements of our participant -/
def participant_placements : List Nat := [1, 2, 3]

/-- Theorem stating the lowest possible placement for the participant -/
theorem lowest_possible_placement :
  ∃ (lowest_place : Nat),
    lowest_place = 7 ∧
    (∀ (other_total_score : Nat),
      other_total_score < calculate_total_score participant_placements →
      (∃ (n : Nat), n < lowest_place ∧
        n * places_per_category ≥ other_total_score)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lowest_possible_placement_l235_23574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_inequality_l235_23568

open Real

theorem min_k_for_inequality (α β : ℝ) (h_α : α ∈ Set.Ioo 0 (2*π/3)) (h_β : β ∈ Set.Ioo 0 (2*π/3)) :
  ∃ (k : ℝ), ∀ (k' : ℝ), 
    (∀ (α' β' : ℝ), α' ∈ Set.Ioo 0 (2*π/3) → β' ∈ Set.Ioo 0 (2*π/3) → 
      4 * (cos α')^2 + 2 * cos α' * cos β' + 4 * (cos β')^2 - 3 * cos α' - 3 * cos β' - k' < 0) →
    k ≤ k' ∧ k = 12 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_k_for_inequality_l235_23568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_balls_sold_l235_23567

/-- Proves that the number of balls sold is 20 given the conditions of the problem -/
theorem number_of_balls_sold (cost_price selling_price loss number_of_balls : ℕ) 
  (h1 : cost_price = 48)
  (h2 : selling_price = 720)
  (h3 : loss = 5 * cost_price)
  (h4 : number_of_balls * cost_price - selling_price = loss) :
  number_of_balls = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_balls_sold_l235_23567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l235_23517

/-- Given three vertices of a cube, prove that its surface area is 294 -/
theorem cube_surface_area (A B C : ℝ × ℝ × ℝ) : 
  A = (3, 7, 4) → B = (4, 3, -5) → C = (7, -2, 3) → 
  ∃ (cube : Set (ℝ × ℝ × ℝ)), A ∈ cube ∧ B ∈ cube ∧ C ∈ cube ∧ 
  (∃ (SA : ℝ), SA = 294) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_surface_area_l235_23517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l235_23543

/-- The function f(x) = a(ln|x| - 1/2) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log (abs x) - 1/2)

/-- The function g(x) = x^2 -/
def g (x : ℝ) : ℝ := x^2

/-- The number of distinct intersection points between f and g -/
def num_intersections (a : ℝ) : ℕ := sorry

theorem intersection_points_imply_a_range :
  ∀ a : ℝ, num_intersections a = 4 → a ∈ Set.Ioi (2 * Real.exp 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_imply_a_range_l235_23543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_in_three_l235_23544

-- Define the people and seats
inductive Person : Type
  | Abby | Bret | Carl | Dana

inductive Seat : Type
  | One | Two | Three | Four

-- Define the seating arrangement
def seating : Person → Seat := sorry

-- Conditions
axiom bret_in_two : seating Person.Bret = Seat.Two
axiom all_seated : ∀ s : Seat, ∃ p : Person, seating p = s
axiom one_per_seat : ∀ p1 p2 : Person, seating p1 = seating p2 → p1 = p2

-- Joe's false statements as axioms
axiom not_dana_next_to_bret : 
  (seating Person.Dana ≠ Seat.One) ∧ (seating Person.Dana ≠ Seat.Three)

axiom not_carl_between_dana_and_bret :
  (seating Person.Carl ≠ Seat.One) ∧ (seating Person.Carl ≠ Seat.Three)

-- Theorem to prove
theorem abby_in_three : seating Person.Abby = Seat.Three := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abby_in_three_l235_23544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_property_l235_23569

-- Define a quadratic polynomial with integer coefficients
def QuadraticPolynomial (a b c : ℤ) : ℤ → ℤ :=
  fun n => a * n^2 + b * n + c

-- Define the property of being relatively prime
def IsRelativelyPrime (a b : ℤ) : Prop := Nat.gcd a.natAbs b.natAbs = 1

-- State the theorem
theorem quadratic_polynomial_property (P : ℤ → ℤ) :
  (∃ a b c : ℤ, ∀ n : ℤ, P n = QuadraticPolynomial a b c n) →
  (∀ n : ℤ, n > 0 → IsRelativelyPrime (P n) n ∧ IsRelativelyPrime (P (P n)) n) →
  P 3 = 89 →
  P 10 = 859 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_polynomial_property_l235_23569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_in_still_water_l235_23575

/-- The upstream speed as a function of time -/
noncomputable def V_u (t : ℝ) : ℝ := 32 * (1 - 0.05 * t)

/-- The downstream speed as a function of time -/
noncomputable def V_d (t : ℝ) : ℝ := 48 * (1 + 0.04 * t)

/-- The speed in still water as a function of time -/
noncomputable def S (t : ℝ) : ℝ := (V_u t + V_d t) / 2

/-- Theorem stating that the speed in still water is equal to 40 + 0.16t -/
theorem speed_in_still_water (t : ℝ) : S t = 40 + 0.16 * t := by
  -- Unfold the definitions
  unfold S V_u V_d
  -- Simplify the expression
  simp [mul_add, add_mul, mul_comm, mul_assoc]
  -- Perform arithmetic
  ring

end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_in_still_water_l235_23575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l235_23552

/-- Represents the position and velocity of a train -/
structure Train where
  initial_distance : ℝ  -- Initial distance from intersection in km
  velocity : ℝ          -- Velocity in km/min

/-- Calculates the distance between two trains at a given time -/
noncomputable def distance_between_trains (t : ℝ) (train1 train2 : Train) : ℝ :=
  let d1 := train1.initial_distance - train1.velocity * t
  let d2 := train2.initial_distance - train2.velocity * t
  Real.sqrt (d1^2 + d2^2)

/-- The main theorem stating the minimum distance and time -/
theorem min_distance_theorem (train1 train2 : Train)
    (h1 : train1 = { initial_distance := 50, velocity := 0.6 })
    (h2 : train2 = { initial_distance := 40, velocity := 0.8 }) :
    ∃ (t : ℝ), t = 62 ∧
    distance_between_trains t train1 train2 = 16 ∧
    ∀ (t' : ℝ), distance_between_trains t' train1 train2 ≥ 16 := by
  sorry

#check min_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_theorem_l235_23552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l235_23573

/-- The slope angle of the line √3x + y + 3 = 0 is 120° -/
theorem slope_angle_of_line :
  Real.arctan (-Real.sqrt 3) = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_line_l235_23573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l235_23555

-- Define the vectors
def a : Fin 3 → ℝ := ![5, 3, 4]
def b : Fin 3 → ℝ := ![4, 3, 3]
def c : Fin 3 → ℝ := ![9, 5, 8]

-- Theorem statement
theorem vectors_not_coplanar :
  ¬(∃ (α β γ : ℝ), (∀ i, α * a i + β * b i + γ * c i = 0) ∧ (α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vectors_not_coplanar_l235_23555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_neg_one_sufficient_not_necessary_l235_23545

-- Define the complex number z
def z (a : ℝ) : ℂ := (a - 2*Complex.I)*Complex.I

-- Define the point M in the complex plane
def M (a : ℝ) : ℂ := z a

-- Define what it means for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℂ) : Prop := 0 < p.re ∧ p.im < 0

-- Theorem statement
theorem a_equals_neg_one_sufficient_not_necessary :
  (∀ a : ℝ, a = -1 → in_fourth_quadrant (M a)) ∧
  (∃ a : ℝ, a ≠ -1 ∧ in_fourth_quadrant (M a)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_neg_one_sufficient_not_necessary_l235_23545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l235_23583

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line segment from point p to point q -/
structure LineSegment where
  p : Point
  q : Point

/-- Represents a square with side length s and lower left corner at (x, y) -/
structure Square where
  s : ℝ
  x : ℝ
  y : ℝ

/-- Calculates the area of a triangle given its base and height -/
noncomputable def triangleArea (base height : ℝ) : ℝ :=
  (base * height) / 2

/-- Theorem: A line from (c, 0) to (2, 2) divides a 2x2 square with lower left corner at origin into two equal areas iff c = 0 -/
theorem equal_area_division (c : ℝ) :
  let square := Square.mk 2 0 0
  let line := LineSegment.mk (Point.mk c 0) (Point.mk 2 2)
  (triangleArea (2 - c) 2 = square.s^2 / 2) ↔ (c = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_division_l235_23583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l235_23533

noncomputable def f (x : ℝ) : ℝ := 5 * Real.sin (2 * x + Real.pi / 4)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := f (x + φ)

theorem symmetry_implies_phi_value (φ : ℝ) 
  (h1 : 0 < φ) (h2 : φ < Real.pi / 2) 
  (h3 : ∀ x, g φ x = g φ (-x)) : 
  φ = Real.pi / 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_implies_phi_value_l235_23533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_with_more_than_three_factors_eq_nine_l235_23516

/-- The number of positive integer factors of 4410 that have more than 3 factors -/
def factors_with_more_than_three_factors : ℕ :=
  (Finset.filter (fun d => (Nat.divisors d).card > 3) (Nat.divisors 4410)).card

/-- Theorem stating that the number of positive integer factors of 4410 
    that have more than 3 factors is equal to 9 -/
theorem factors_with_more_than_three_factors_eq_nine :
  factors_with_more_than_three_factors = 9 := by
  sorry

#eval factors_with_more_than_three_factors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_with_more_than_three_factors_eq_nine_l235_23516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_deletion_theorem_l235_23513

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- Check if a polynomial has at least one real root -/
def has_real_root (p : RealPolynomial) : Prop :=
  ∃ x : ℝ, p x = 0

/-- The constant term of a polynomial -/
def constant_term (p : RealPolynomial) : ℝ :=
  p 0

/-- A sequence of polynomials obtained by successive monomial deletions -/
def monomial_deletion_sequence (p : RealPolynomial) (a₀ : ℝ) : List RealPolynomial → Prop
  | [] => False
  | [q] => q = p ∧ constant_term q = a₀ ∧ has_real_root q
  | (q::r::rs) => q = p ∧ has_real_root q ∧ monomial_deletion_sequence r a₀ (r::rs)

/-- Main theorem -/
theorem monomial_deletion_theorem (p : RealPolynomial) (hp : has_real_root p) 
  (ha₀ : constant_term p ≠ 0) :
  ∃ seq : List RealPolynomial, monomial_deletion_sequence p (constant_term p) seq :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monomial_deletion_theorem_l235_23513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l235_23519

-- Define the functions f and g
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom f_g_sum : ∀ x, f x + g x = 2^x

-- Define the inequality condition
def inequality_condition (a : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 1 2 → a * f x + g (2 * x) ≥ 0

-- State the theorem
theorem minimum_a_value :
  ∃ a_min : ℝ, a_min = -17/6 ∧
  (∀ a, inequality_condition a ↔ a ≥ a_min) := by
  sorry

#check minimum_a_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_a_value_l235_23519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l235_23523

-- Define sets A and B
def A : Set ℝ := {x | x - 2 < 0}
def B : Set ℝ := {x | Real.rpow 2 x > 1}

-- Define the open interval (0, 2)
def openInterval : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_equals_open_interval : A ∩ B = openInterval := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_open_interval_l235_23523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l235_23518

theorem remainder_theorem (n : ℕ) (a b c : ℤ) 
  (h_pos : n > 0)
  (h_inv_a : ∃ k : ℤ, a * k ≡ 1 [ZMOD n])
  (h_inv_b : ∃ k : ℤ, b * k ≡ 1 [ZMOD n])
  (h_cong : ∃ k : ℤ, k * b ≡ 1 [ZMOD n] ∧ a ≡ k + c [ZMOD n]) :
  (a - c) * b ≡ 1 [ZMOD n] := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l235_23518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_even_product_is_even_even_number_of_odd_product_is_odd_odd_number_of_even_product_is_even_odd_number_of_odd_product_is_odd_l235_23582

-- Define a function to check if a number is even
def isEven (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

-- Define a function to check if a number is odd
def isOdd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

-- Define a function to represent the product of a list of integers
def product (list : List ℤ) : ℤ := list.foldl (· * ·) 1

-- Theorem 1: Product of an even number of even integers is even
theorem even_number_of_even_product_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, isEven x) : 
  isEven (product list) := by sorry

-- Theorem 2: Product of an even number of odd integers is odd
theorem even_number_of_odd_product_is_odd (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n) 
  (h2 : ∀ x ∈ list, isOdd x) : 
  isOdd (product list) := by sorry

-- Theorem 3: Product of an odd number of even integers is even
theorem odd_number_of_even_product_is_even (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, isEven x) : 
  isEven (product list) := by sorry

-- Theorem 4: Product of an odd number of odd integers is odd
theorem odd_number_of_odd_product_is_odd (n : ℕ) (list : List ℤ) 
  (h1 : list.length = 2 * n + 1) 
  (h2 : ∀ x ∈ list, isOdd x) : 
  isOdd (product list) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_number_of_even_product_is_even_even_number_of_odd_product_is_odd_odd_number_of_even_product_is_even_odd_number_of_odd_product_is_odd_l235_23582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_11_negation_greater_than_neg_150_l235_23595

theorem largest_multiple_11_negation_greater_than_neg_150 : 
  ∀ n : Int, (11 ∣ n) → -n > -150 → n ≤ 143 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_multiple_11_negation_greater_than_neg_150_l235_23595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odds_against_C_in_three_horse_race_l235_23580

/-- Represents the odds against an event occurring -/
structure Odds where
  against : ℕ
  in_favor : ℕ

/-- Calculates the probability of an event given its odds -/
def oddsToProb (o : Odds) : ℚ :=
  o.in_favor / (o.against + o.in_favor)

/-- Represents a three-horse race with no ties -/
structure ThreeHorseRace where
  oddsAgainstA : Odds
  oddsAgainstB : Odds
  oddsAgainstC : Odds
  sumOfProbs : oddsToProb oddsAgainstA + oddsToProb oddsAgainstB + oddsToProb oddsAgainstC = 1

theorem odds_against_C_in_three_horse_race :
  ∀ (race : ThreeHorseRace),
  race.oddsAgainstA = Odds.mk 5 2 →
  race.oddsAgainstB = Odds.mk 7 4 →
  race.oddsAgainstC = Odds.mk 50 27 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odds_against_C_in_three_horse_race_l235_23580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_speed_correct_l235_23594

/-- The speed required to arrive exactly on time -/
noncomputable def exact_speed : ℝ := 42.78

/-- The speed that causes a 5-minute late arrival -/
noncomputable def late_speed : ℝ := 35

/-- The speed that causes a 5-minute early arrival -/
noncomputable def early_speed : ℝ := 55

/-- The time difference (in hours) for the late arrival -/
noncomputable def late_time_diff : ℝ := 5 / 60

/-- The time difference (in hours) for the early arrival -/
noncomputable def early_time_diff : ℝ := -5 / 60

/-- The theorem stating that the calculated speed is correct -/
theorem exact_speed_correct :
  ∃ (distance : ℝ) (ideal_time : ℝ),
    distance > 0 ∧
    ideal_time > 0 ∧
    distance = late_speed * (ideal_time + late_time_diff) ∧
    distance = early_speed * (ideal_time + early_time_diff) ∧
    distance / ideal_time = exact_speed :=
by sorry

#check exact_speed_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exact_speed_correct_l235_23594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unique_sums_l235_23585

def coin_values : List ℕ := [1, 1, 1, 5, 5, 5, 10, 10, 10, 25, 50]

def unique_sums (coins : List ℕ) : Finset ℕ :=
  (List.join (List.map (λ x => List.map (λ y => x + y) coins) coins)).toFinset

theorem max_unique_sums :
  Finset.card (unique_sums coin_values) = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_unique_sums_l235_23585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_sum_3_powers_l235_23587

-- Define the function that gives the unit digit of 3^n
def unitDigitOf3Power (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 3
  | 2 => 9
  | 3 => 7
  | _ => 1

-- Define the sum of unit digits of 3^n from 1 to m
def sumOfUnitDigits (m : ℕ) : ℕ :=
  (List.range m).map (fun i => unitDigitOf3Power (i + 1)) |>.sum

-- Theorem statement
theorem unit_digit_of_sum_3_powers :
  sumOfUnitDigits 2023 % 10 = 9 := by
  sorry

#eval sumOfUnitDigits 2023 % 10

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unit_digit_of_sum_3_powers_l235_23587
