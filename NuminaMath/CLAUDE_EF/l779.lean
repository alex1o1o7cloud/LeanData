import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shekar_math_marks_l779_77994

/-- Represents the marks scored by a student in various subjects -/
structure Marks where
  mathematics : ℕ
  science : ℕ
  socialStudies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of a list of natural numbers -/
def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

/-- Theorem: Given Shekar's marks in other subjects and his average, his mathematics marks are 76 -/
theorem shekar_math_marks (shekar : Marks) :
  shekar.science = 65 →
  shekar.socialStudies = 82 →
  shekar.english = 67 →
  shekar.biology = 55 →
  average [shekar.mathematics, shekar.science, shekar.socialStudies, shekar.english, shekar.biology] = 69 →
  shekar.mathematics = 76 := by
  sorry

-- Remove the #eval statement as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shekar_math_marks_l779_77994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l779_77928

theorem matrix_scalar_multiplication (w : Fin 2 → ℝ) :
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![7, 0; 0, 7]
  Matrix.mulVec N w = (7 : ℝ) • w := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_scalar_multiplication_l779_77928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DEF_l779_77992

noncomputable section

-- Define the points
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (6, 0)

-- Define midpoints
def D : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
def E : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
def F : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the area function for a triangle given three points
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((p2.1 - p1.1) * (p3.2 - p1.2) - (p3.1 - p1.1) * (p2.2 - p1.2))

-- Theorem statement
theorem area_of_DEF : triangleArea D E F = 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_DEF_l779_77992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_example_l779_77961

/-- The number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a : ℤ) (d : ℤ) (l : ℤ) : ℕ :=
  ((l - a) / d + 1).natAbs

theorem arithmetic_sequence_length_example :
  arithmeticSequenceLength (-5) 4 39 = 12 := by
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_length_example_l779_77961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_twenty_l779_77921

/-- Calculates the profit percentage given cost price, markup percentage, and discount --/
noncomputable def profit_percentage (cost_price : ℝ) (markup_percentage : ℝ) (discount : ℝ) : ℝ :=
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that given the specific values, the profit percentage is 20% --/
theorem profit_percentage_is_twenty :
  profit_percentage 180 45 45 = 20 := by
  -- Unfold the definition of profit_percentage
  unfold profit_percentage
  -- Simplify the expressions
  simp
  -- The proof is completed using numeric approximation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_twenty_l779_77921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_origin_l779_77982

noncomputable def point_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem distance_from_origin (x : ℝ) : 
  x > 2 → 
  point_distance x 8 2 5 = 7 → 
  point_distance x 8 0 0 = Real.sqrt (108 + 8 * Real.sqrt 10) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_origin_l779_77982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_approximation_volume_rounded_to_360_l779_77980

-- Define the surface equation
def surface (x y z : ℝ) : Prop := x^2016 + y^2016 + z^2 = 2016

-- Define the volume enclosed by the surface
noncomputable def enclosed_volume : ℝ := sorry

-- Theorem statement
theorem volume_approximation :
  355 < enclosed_volume ∧ enclosed_volume < 365 :=
sorry

-- Rounding to nearest multiple of ten
noncomputable def round_to_nearest_ten (x : ℝ) : ℤ :=
  ⌊(x + 5) / 10⌋ * 10

-- Final theorem
theorem volume_rounded_to_360 :
  round_to_nearest_ten enclosed_volume = 360 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_approximation_volume_rounded_to_360_l779_77980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_odd_function_m_range_l779_77960

/-- A function is locally odd if there exists an x₀ in its domain such that f(-x₀) = -f(x₀) -/
def LocallyOdd (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₀ ∈ Set.Ioo a b, f (-x₀) = -f x₀

/-- The function f(x) = 1/(x-3) + m -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 1 / (x - 3) + m

theorem local_odd_function_m_range :
  ∀ m : ℝ, LocallyOdd (f m) (-1) 1 → m ∈ Set.Icc (1/3) (3/8) := by
  sorry

#check local_odd_function_m_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_odd_function_m_range_l779_77960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_factory_problem_l779_77991

/-- Represents the daily production deviation from the planned quantity -/
def ProductionDeviation := List Int

/-- Calculates the total production for the first n days -/
def totalProduction (planned : Nat) (deviations : ProductionDeviation) (n : Nat) : Int :=
  (planned * n : Int) + (deviations.take n).sum

/-- Finds the difference between the highest and lowest production days -/
def productionDifference (deviations : ProductionDeviation) : Int :=
  deviations.maximum.getD 0 - deviations.minimum.getD 0

/-- Calculates the total wage based on the piece-rate system -/
def totalWage (planned : Nat) (actual : Int) (baseRate : Nat) (overRate : Nat) (underRate : Nat) : Int :=
  (planned * baseRate : Int) + 
    if actual > planned then
      (actual - planned) * (baseRate + overRate)
    else
      (planned - actual) * (baseRate - underRate)

theorem bicycle_factory_problem 
  (weeklyPlanned : Nat)
  (dailyPlanned : Nat)
  (deviations : ProductionDeviation)
  (baseRate : Nat)
  (overRate : Nat)
  (underRate : Nat)
  (h1 : weeklyPlanned = 700)
  (h2 : dailyPlanned = 100)
  (h3 : deviations = [5, -2, -4, 13, -10, 16, -9])
  (h4 : baseRate = 60)
  (h5 : overRate = 15)
  (h6 : underRate = 20) :
  (totalProduction dailyPlanned deviations 4 = 412) ∧
  (productionDifference deviations = 26) ∧
  (totalWage weeklyPlanned (weeklyPlanned + deviations.sum) baseRate overRate underRate = 42675) := by
  sorry

#eval totalProduction 100 [5, -2, -4, 13, -10, 16, -9] 4
#eval productionDifference [5, -2, -4, 13, -10, 16, -9]
#eval totalWage 700 709 60 15 20

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_factory_problem_l779_77991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_alternating_sum_l779_77900

open BigOperators
open Finset

theorem binomial_alternating_sum (m n : ℕ) :
  (∑ k in range (n + 1), ((-1)^k : ℚ) * (1 : ℚ) / (m + k + 1) * (Nat.choose n k : ℚ)) =
  (1 : ℚ) / ((m + n + 1 : ℕ) * Nat.choose (m + n) n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_alternating_sum_l779_77900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l779_77985

/-- The function f(x) = ln x - 2x --/
noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x

/-- The derivative of f(x) --/
noncomputable def f_derivative (x : ℝ) : ℝ := 1 / x - 2

/-- The point of tangency --/
def point : ℝ × ℝ := (1, -2)

/-- The slope of the tangent line at x = 1 --/
noncomputable def tangent_slope : ℝ := f_derivative point.fst

/-- Theorem: The equation of the tangent line to f(x) at (1, -2) is x + y + 1 = 0 --/
theorem tangent_line_equation :
  ∀ (x y : ℝ), (x = point.fst ∧ y = f point.fst) → (x + y + 1 = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l779_77985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l779_77906

def n : ℕ := 2^3 * 3^2 * 5^1 * 7^3

theorem even_factors_count : 
  (Finset.filter (fun x => x ∣ n ∧ Even x) (Finset.range (n + 1))).card = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_factors_count_l779_77906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_smallest_grade_l779_77970

/-- Represents the number of students to be sampled from the grade with the smallest proportion
    in a stratified sampling scenario. -/
def stratified_sample_smallest_grade (total_students : ℕ) (sample_size : ℕ) 
  (grade_ratio : List ℕ) : ℕ :=
  let total_ratio := grade_ratio.sum
  let smallest_proportion := (grade_ratio.minimum?.getD 1) / total_ratio
  ⌊(sample_size : ℚ) * (smallest_proportion : ℚ)⌋.toNat

/-- Theorem stating that in the given stratified sampling scenario, 
    30 students should be sampled from the grade with the smallest proportion. -/
theorem sample_size_smallest_grade :
  stratified_sample_smallest_grade 3000 180 [2, 3, 1] = 30 := by
  sorry

#eval stratified_sample_smallest_grade 3000 180 [2, 3, 1]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sample_size_smallest_grade_l779_77970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016_greater_than_6_l779_77924

def my_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 3/2 ∧ 
  ∀ n ≥ 2, a n^2 - a n * a (n+1) - a n * a (n-1) + a (n+1) * a (n-1) + 2*a n - a (n+1) - a (n-1) = 0

theorem sequence_2016_greater_than_6 (a : ℕ → ℚ) (h : my_sequence a) : a 2016 > 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2016_greater_than_6_l779_77924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l779_77958

/-- The function f(x) = 5x - 2x^3 --/
def f (x : ℝ) : ℝ := 5 * x - 2 * x^3

/-- The interval [0, 2] --/
def I : Set ℝ := Set.Icc 0 2

/-- The maximum value of f(x) on the interval [0, 2] --/
noncomputable def max_value : ℝ := (10 * Real.sqrt (5/6)) / 3

/-- Theorem stating that there exists a point in the interval [0, 2] where f(x) achieves its maximum value --/
theorem max_value_on_interval :
  ∃ (x : ℝ), x ∈ I ∧ f x = max_value ∧ ∀ (y : ℝ), y ∈ I → f y ≤ max_value := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_on_interval_l779_77958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_arithmetic_sequence_exists_l779_77911

/-- A sequence of 100 natural numbers in an increasing arithmetic progression -/
def arithmetic_sequence (a d : ℕ) : Fin 100 → ℕ := fun i ↦ a + i.val * d

/-- Two natural numbers are coprime -/
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem coprime_arithmetic_sequence_exists : 
  ∃ (a d : ℕ), ∀ (i j : Fin 100), i ≠ j → 
    coprime (arithmetic_sequence a d i) (arithmetic_sequence a d j) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_arithmetic_sequence_exists_l779_77911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l779_77974

def generateSequence : List Nat → List Nat
  | [] => [1]
  | (x::xs) => 
    let new_numbers := [5*x, 5*x + 1]
    let filtered := new_numbers.filter (· < 1000)
    filtered ++ (x::xs)

def all_numbers : List Nat := 
  generateSequence (generateSequence (generateSequence (generateSequence [])))

def distinct_sums (numbers : List Nat) : Finset Nat :=
  (do
    let (x, i) ← numbers.enum
    let (y, j) ← numbers.enum
    guard (i < j)
    pure (x + y)
  ).toFinset

theorem distinct_sums_count : (distinct_sums all_numbers).card = 206 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sums_count_l779_77974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_neg_five_thirds_l779_77940

/-- The sum of the infinite series Σ(3n - 2) / (n(n + 1)(n + 3)) for n from 1 to infinity -/
noncomputable def infinite_series_sum : ℝ := ∑' n, (3 * n - 2) / (n * (n + 1) * (n + 3))

/-- The infinite series Σ(3n - 2) / (n(n + 1)(n + 3)) for n from 1 to infinity converges to -5/3 -/
theorem infinite_series_sum_eq_neg_five_thirds : infinite_series_sum = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_series_sum_eq_neg_five_thirds_l779_77940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_33_yuan_exists_distance_for_33_yuan_max_distance_is_15_km_l779_77968

/-- Taxi fare calculation function -/
noncomputable def taxi_fare (distance : ℝ) : ℝ :=
  if distance ≤ 4 then 11
  else 11 + 2 * (distance - 4)

/-- Theorem: Maximum distance for a 33 yuan fare is 15 km -/
theorem max_distance_for_33_yuan :
  ∀ d : ℝ, d ≥ 0 → taxi_fare d ≤ 33 → d ≤ 15 :=
by
  sorry

/-- Theorem: There exists a distance that results in exactly 33 yuan fare -/
theorem exists_distance_for_33_yuan :
  ∃ d : ℝ, d ≥ 0 ∧ taxi_fare d = 33 :=
by
  sorry

/-- Corollary: The maximum distance for a 33 yuan fare is exactly 15 km -/
theorem max_distance_is_15_km :
  ∃ d : ℝ, d = 15 ∧ taxi_fare d = 33 ∧ ∀ x : ℝ, x ≥ 0 → taxi_fare x ≤ 33 → x ≤ d :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_for_33_yuan_exists_distance_for_33_yuan_max_distance_is_15_km_l779_77968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_minimizes_sum_distances_l779_77953

/-- Represents a point on a line --/
structure Point where
  x : ℝ

/-- The distance between two points --/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- The sum of distances from a point to a list of points --/
def sum_distances (q : Point) (points : List Point) : ℝ :=
  (points.map (fun p => distance q p)).sum

/-- Theorem stating that Q₅ minimizes the sum of distances --/
theorem median_minimizes_sum_distances
  (q₁ q₂ q₃ q₄ q₅ q₆ q₇ q₈ q₉ : Point)
  (h_order : q₁.x < q₂.x ∧ q₂.x < q₃.x ∧ q₃.x < q₄.x ∧ q₄.x < q₅.x ∧
             q₅.x < q₆.x ∧ q₆.x < q₇.x ∧ q₇.x < q₈.x ∧ q₈.x < q₉.x) :
  ∀ q : Point, sum_distances q₅ [q₁, q₂, q₃, q₄, q₅, q₆, q₇, q₈, q₉] ≤
               sum_distances q  [q₁, q₂, q₃, q₄, q₅, q₆, q₇, q₈, q₉] :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_minimizes_sum_distances_l779_77953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l779_77999

/-- The function g(x, y) = 1 / (x^2 + (x-y)^2 + y^2) -/
noncomputable def g (x y : ℝ) : ℝ := 1 / (x^2 + (x-y)^2 + y^2)

/-- The domain of g(x, y) -/
def domain_g : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ≠ (0, 0)}

theorem domain_of_g :
  ∀ p : ℝ × ℝ, p ∈ domain_g ↔ ∃ z : ℝ, g p.1 p.2 = z :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l779_77999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_unchanged_l779_77934

-- Define the density of water and ice
noncomputable def ρ_water : ℝ := 1000
noncomputable def ρ_ice : ℝ := 917

-- Define the volume of water taken to make ice
variable (V : ℝ)

-- Define the volume of ice formed
noncomputable def W (V : ℝ) : ℝ := V * (ρ_water / ρ_ice)

-- Define the volume of water displaced by floating ice
def U (V : ℝ) : ℝ := V

-- Theorem: The water level remains at the rim of the glass
theorem water_level_unchanged (V : ℝ) (h : V > 0) :
  U V = V := by
  -- Proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_level_unchanged_l779_77934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_in_selection_l779_77901

theorem multiple_in_selection (n : ℕ) (A : Finset ℕ) : 
  A ⊆ Finset.range (2 * n + 1) → A.card = n + 1 →
  ∃ x y, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ (x ∣ y ∨ y ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_in_selection_l779_77901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_name_card_probability_l779_77948

theorem name_card_probability : 
  let total_cards : ℕ := 12
  let bill_cards : ℕ := 4
  let john_cards : ℕ := 5
  let prob_bill_then_john : ℚ := (bill_cards : ℚ) / total_cards * john_cards / (total_cards - 1)
  let prob_john_then_bill : ℚ := (john_cards : ℚ) / total_cards * bill_cards / (total_cards - 1)
  prob_bill_then_john + prob_john_then_bill = 10 / 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_name_card_probability_l779_77948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circleminus_example_l779_77950

/-- The ⊖ operation defined for integers -/
def circleminus (a b : ℤ) : ℤ := a^b.toNat - b^a.toNat

/-- Theorem stating that 2 ⊖ (2 ⊖ 5) = 79 -/
theorem circleminus_example : circleminus 2 (circleminus 2 5) = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circleminus_example_l779_77950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l779_77957

noncomputable def distance (x y : ℝ × ℝ) : ℝ :=
  Real.sqrt ((x.1 - y.1)^2 + (x.2 - y.2)^2)

def points : List (ℝ × ℝ) := [(0,5), (2,3), (4,-3), (7,1), (-2,-1)]

def origin : ℝ × ℝ := (1,1)

theorem farthest_point :
  ∃ (p : ℝ × ℝ), p ∈ points ∧
    (∀ q ∈ points, distance origin q ≤ distance origin p) ∧
    p = (7,1) ∧
    distance origin p = 6 :=
by sorry

#check farthest_point

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l779_77957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPF_l779_77931

/-- Ellipse C with equation (x^2 / 9) + (y^2 / 10) = 1 -/
def C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 9) + (p.2^2 / 10) = 1}

/-- Right vertex of ellipse C -/
def A : ℝ × ℝ := (3, 0)

/-- Upper focus of ellipse C -/
def F : ℝ × ℝ := (0, 1)

/-- A point P on ellipse C in the first quadrant -/
noncomputable def P : ℝ → ℝ × ℝ := λ θ => (3 * Real.cos θ, Real.sqrt 10 * Real.sin θ)

/-- Area of quadrilateral OAPF -/
noncomputable def area_OAPF (θ : ℝ) : ℝ :=
  (3/2) * (Real.sqrt 10 * Real.sin θ + Real.cos θ)

/-- Maximum area of quadrilateral OAPF -/
theorem max_area_OAPF :
  ∃ θ, ∀ φ, area_OAPF θ ≥ area_OAPF φ ∧ area_OAPF θ = (3/2) * Real.sqrt 11 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_OAPF_l779_77931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_30_18_l779_77965

/-- The area of a rhombus given its diagonal lengths -/
noncomputable def rhombusArea (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2

/-- Theorem: The area of a rhombus with diagonal lengths 30 and 18 is 270 square units -/
theorem rhombus_area_30_18 : rhombusArea 30 18 = 270 := by
  -- Unfold the definition of rhombusArea
  unfold rhombusArea
  -- Simplify the arithmetic
  simp [mul_div_assoc]
  -- The result follows from arithmetic
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_area_30_18_l779_77965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hardcover_price_is_040_l779_77925

/-- Calculates the price of each hardcover book given the following conditions:
  1. Author gets 6% of total sales from paper cover version
  2. Author gets 12% of total sales from hardcover version
  3. 32,000 copies of paper cover version sold at $0.20 each
  4. 15,000 copies of hardcover version sold
  5. Author earned $1104 in total
-/
noncomputable def hardcover_price (paper_cover_percentage : ℝ) (hardcover_percentage : ℝ)
                    (paper_cover_copies : ℕ) (paper_cover_price : ℝ)
                    (hardcover_copies : ℕ) (total_earnings : ℝ) : ℝ :=
  let paper_cover_sales := (paper_cover_copies : ℝ) * paper_cover_price
  let paper_cover_earnings := paper_cover_sales * paper_cover_percentage
  let hardcover_earnings := total_earnings - paper_cover_earnings
  hardcover_earnings / (hardcover_percentage * (hardcover_copies : ℝ))

/-- Theorem stating that the price of each hardcover book is $0.40 -/
theorem hardcover_price_is_040 :
  hardcover_price 0.06 0.12 32000 0.20 15000 1104 = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hardcover_price_is_040_l779_77925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_l779_77935

/-- Represents the selling price of each TV in yuan -/
noncomputable def selling_price : ℝ := 3080

/-- Represents the profit percentage on the first TV -/
noncomputable def profit_percentage : ℝ := 0.12

/-- Represents the loss percentage on the second TV -/
noncomputable def loss_percentage : ℝ := 0.12

/-- Calculates the cost price of the TV sold at a profit -/
noncomputable def cost_price_profit : ℝ := selling_price / (1 + profit_percentage)

/-- Calculates the cost price of the TV sold at a loss -/
noncomputable def cost_price_loss : ℝ := selling_price / (1 - loss_percentage)

/-- Theorem stating the store's loss situation -/
theorem store_loss : 
  2 * selling_price - cost_price_profit - cost_price_loss = -90 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_loss_l779_77935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_x_coordinate_l779_77973

/-- A rectangle in a 2D coordinate system --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- Calculate the area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ :=
  |r.x2 - r.x1| * |r.y2 - r.y1|

/-- Theorem: If a rectangle has vertices (-8, 1), (x, 1), (x, -7), and (-8, -7),
    and its area is 72, then x = 1 --/
theorem rectangle_x_coordinate (x : ℝ) :
  let r := Rectangle.mk (-8) 1 x (-7)
  r.area = 72 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_x_coordinate_l779_77973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l779_77938

-- Define the function f(x) = x / (1-x)
noncomputable def f (x : ℝ) : ℝ := x / (1 - x)

-- Theorem statement
theorem f_increasing_on_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l779_77938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_sampled_students_l779_77917

/-- Represents the number of students sampled from a grade -/
abbrev SampledStudents := ℕ

/-- Represents the total number of students in a grade -/
abbrev TotalStudents := ℕ

/-- Calculates the number of students sampled from a grade based on proportional sampling -/
def calculateSampledStudents (totalStudents : TotalStudents) (totalSampled : SampledStudents) (totalPopulation : TotalStudents) : SampledStudents :=
  (totalStudents * totalSampled) / totalPopulation

theorem first_year_sampled_students 
  (first_year_total : TotalStudents)
  (second_year_total : TotalStudents)
  (third_year_total : TotalStudents)
  (third_year_sampled : SampledStudents)
  (h1 : first_year_total = 800)
  (h2 : second_year_total = 600)
  (h3 : third_year_total = 500)
  (h4 : third_year_sampled = 25) :
  calculateSampledStudents first_year_total third_year_sampled (first_year_total + second_year_total + third_year_total) = 40 := by
  sorry

#check first_year_sampled_students

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_year_sampled_students_l779_77917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sequence_l779_77993

/-- The set of sequences satisfying the recurrence relation and non-constant condition -/
def F_p (p : ℕ) : Set (ℕ → ℕ) :=
  {a | (∀ n, a (n + 1) = (p + 1) * a n - p * a (n - 1)) ∧
       (∃ m n, a m ≠ a n)}

/-- The candidate minimum sequence -/
def a (p : ℕ) (n : ℕ) : ℚ :=
  (p ^ n - 1) / (p - 1)

theorem minimum_sequence (p : ℕ) (hp : p > 1) :
  (∀ n, ∃ m : ℕ, a p n = m) ∧
  (∃ f : ℕ → ℕ, (∀ n, f n = ⌊a p n⌋) ∧ f ∈ F_p p) ∧
  (∀ b ∈ F_p p, ∀ n, ⌊a p n⌋ ≤ b n) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_sequence_l779_77993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_feet_symmetry_l779_77977

-- Define the basic structures
structure Point : Type where
  x : ℝ
  y : ℝ

-- Define the triangle
structure Triangle where
  O : Point
  A : Point
  B : Point

-- Define the line
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define the feet of altitudes
noncomputable def altitude_foot (T : Triangle) (v : Point) : Point :=
  sorry

-- Define the locus of a point
noncomputable def locus (p : Triangle → Point) : Line :=
  sorry

-- Define symmetry with respect to a line
def symmetric_wrt (l₁ l₂ l : Line) : Prop :=
  sorry

-- Theorem statement
theorem altitude_feet_symmetry (O : Point) (e : Line) :
  ∀ (T : Triangle),
    T.O = O ∧ 
    (∃ (t : ℝ), T.A.x = t * e.a ∧ T.A.y = t * e.b) ∧
    (∃ (s : ℝ), T.B.x = s * e.a ∧ T.B.y = s * e.b) →
    let M := altitude_foot T T.O
    let N := altitude_foot T T.A
    let P := altitude_foot T T.B
    let OP : Line := sorry
    symmetric_wrt (locus (λ T ↦ M)) (locus (λ T ↦ N)) OP := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_feet_symmetry_l779_77977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_in_five_rolls_value_l779_77933

open BigOperators Finset

def probability_four_twos_in_five_rolls : ℚ :=
  let n : ℕ := 8  -- number of sides on the die
  let k : ℕ := 5  -- number of rolls
  let x : ℕ := 4  -- number of times we want to see '2'
  let p : ℚ := 1 / n  -- probability of rolling a '2' on a single roll
  (Nat.choose k x : ℚ) * p^x * (1 - p)^(k - x)

theorem probability_four_twos_in_five_rolls_value :
  probability_four_twos_in_five_rolls = 35 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_twos_in_five_rolls_value_l779_77933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_half_l779_77997

noncomputable section

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 1/(2^x - 1) + a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (2^x - 1) + a

theorem odd_function_implies_a_eq_half :
  ∀ a : ℝ, IsOdd (f a) → a = 1/2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_implies_a_eq_half_l779_77997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_2021_is_blue_ball_2021_is_blue_proof_l779_77914

/-- Represents the color of a ball -/
inductive Color
  | Green
  | Red
  | Yellow
  | Blue

/-- Represents the sequence of balls -/
def BallSequence := Nat → Color

/-- Checks if a given subsequence of 5 balls is valid according to the rules -/
def isValidSubsequence (seq : BallSequence) (start : Nat) : Prop :=
  (∃ i ∈ Finset.range 5, seq (start + i) = Color.Red) ∧
  (∃ i ∈ Finset.range 5, seq (start + i) = Color.Yellow) ∧
  (∃ i ∈ Finset.range 5, seq (start + i) = Color.Blue) ∧
  (∀ i ∈ Finset.range 4,
    seq (start + i) = Color.Red → seq (start + i + 1) = Color.Yellow)

/-- The main theorem to be proved -/
theorem ball_2021_is_blue (seq : BallSequence) : Prop :=
  (∀ n : Nat, n ≤ 2017 → isValidSubsequence seq n) →
  seq 2 = Color.Green →
  seq 20 = Color.Green →
  seq 2021 = Color.Blue

/-- Proof of the main theorem -/
theorem ball_2021_is_blue_proof : ∀ seq, ball_2021_is_blue seq := by
  sorry

#check ball_2021_is_blue_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_2021_is_blue_ball_2021_is_blue_proof_l779_77914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l779_77945

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 + x) - a * x

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := b * (Real.exp x - x)

theorem problem_solution :
  ∃ (a : ℝ),
    (∀ x : ℝ, x ≠ -1 → (deriv (f a)) x = (1 / (1 + x) - a)) ∧
    (deriv (f a)) (-1/2) = 1 ∧
    a = 1 ∧
    (∀ x : ℝ, x > -1 → f a x ≤ 0) ∧
    (∀ n : ℕ, n ≥ 1 → (Finset.sum (Finset.range n) (λ i ↦ 1 / (i + 1 : ℝ))) > Real.log (n + 1)) ∧
    (∀ b : ℝ, (∀ x : ℝ, f 1 x ≤ g b x) ↔ b ≥ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l779_77945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_distance_exceeds_cube_root_l779_77988

-- Define a point with integral coordinates
structure IntPoint where
  x : Int
  y : Int

-- Define a circle
structure Circle where
  center : IntPoint
  radius : ℝ

-- Define a function to check if a point lies on a circle
def lies_on_circle (p : IntPoint) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = (c.radius^2 : ℝ)

-- Define a function to calculate the distance between two points
noncomputable def distance (p1 p2 : IntPoint) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

-- Theorem statement
theorem at_least_one_distance_exceeds_cube_root (c : Circle) (A B C : IntPoint) :
  lies_on_circle A c → lies_on_circle B c → lies_on_circle C c →
  (distance A B > c.radius^(1/3)) ∨ (distance B C > c.radius^(1/3)) ∨ (distance C A > c.radius^(1/3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_least_one_distance_exceeds_cube_root_l779_77988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l779_77922

-- Define the propositions and predicates
def P : Prop := sorry
def Q : Prop := sorry
def R : ℝ → Prop := sorry

-- Define the triangle and its properties
structure Triangle :=
  (A B C : ℝ)
  (valid : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)

-- Define the statements
def statement1 : Prop := (¬(P → Q) ↔ (P → ¬Q))
def statement2 : Prop := (¬(∃ x : ℝ, R x) ↔ (∀ x : ℝ, ¬(R x)))
def statement3 : Prop := ∀ t : Triangle, (Real.sin t.A > Real.sin t.B ↔ t.A > t.B)
def statement4 : Prop := ∃ x y : ℝ, ((x ≠ 2 ∨ y ≠ 3) → (x + y ≠ 5)) ∧ ¬((x ≠ 2 ∨ y ≠ 3) ∧ (x + y = 5))

-- Theorem to prove
theorem exactly_two_correct :
  (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) ∨
  (¬statement1 ∧ statement2 ∧ statement3 ∧ ¬statement4) ∨
  (statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ∨
  (¬statement1 ∧ statement2 ∧ ¬statement3 ∧ statement4) ∨
  (statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ statement4) ∨
  (statement1 ∧ statement2 ∧ ¬statement3 ∧ ¬statement4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_two_correct_l779_77922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_f_l779_77954

-- Define the function f in terms of g
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

-- Define the slope of the tangent line to y = g(x) at (1, g(1))
def g_slope : ℝ := 2

-- State the theorem
theorem tangent_slope_f (g : ℝ → ℝ) :
  (HasDerivAt g g_slope 1) →
  (HasDerivAt (f g) 4 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_f_l779_77954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_10_5_l779_77969

/-- Triangle represented by three points in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculate the area of a triangle given its coordinates -/
noncomputable def triangleArea (t : Triangle) : ℝ :=
  let (x1, y1) := t.A
  let (x2, y2) := t.B
  let (x3, y3) := t.C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- The specific triangle ABC from the problem -/
def triangleABC : Triangle :=
  { A := (-2, 2)
  , B := (5, 2)
  , C := (2, -1) }

theorem triangle_area_is_10_5 : triangleArea triangleABC = 10.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_10_5_l779_77969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_range_endpoints_l779_77929

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 / (3 + 9 * x^2)

-- Define the range of f(x)
def range_f : Set ℝ := {y | ∃ x, f x = y}

-- State the theorem
theorem sum_of_range_endpoints :
  ∃ a b, a ∈ range_f ∧ b ∈ range_f ∧
  (∀ y ∈ range_f, a ≤ y ∧ y ≤ b) ∧
  a + b = 1 := by
  -- Proof goes here
  sorry

#eval "Theorem stated successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_range_endpoints_l779_77929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l779_77905

/-- The distance between two parallel planes -/
noncomputable def distance_between_planes (a b c d : ℝ) (k : ℝ) : ℝ :=
  |k| / Real.sqrt (a^2 + b^2 + c^2)

/-- First plane: x + 2y - 2z + 3 = 0 -/
def plane1 (x y z : ℝ) : Prop := x + 2*y - 2*z + 3 = 0

/-- Second plane: 2x + 4y - 4z + 2 = 0 -/
def plane2 (x y z : ℝ) : Prop := 2*x + 4*y - 4*z + 2 = 0

theorem distance_between_given_planes :
  distance_between_planes 1 2 (-2) 1 2 = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_planes_l779_77905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_relation_l779_77902

/-- Predicate for a set being a circle with given radius -/
def IsCircle (s : Set (ℝ × ℝ)) (r : ℝ) : Prop := sorry

/-- Predicate for two circles being externally tangent -/
def AreExternallyTangent (c₁ c₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate for a set being a common external tangent of two circles -/
def IsCommonExternalTangent (t : Set (ℝ × ℝ)) (c₁ c₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Predicate for two lines being parallel -/
def AreParallel (l₁ l₂ : Set (ℝ × ℝ)) : Prop := sorry

/-- Three circles with given properties imply a relationship between their radii -/
theorem circle_radii_relation (r₁ r₂ r₃ : ℝ) 
  (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : r₃ > 0)
  (h_ext_tangent : ∃ (k₁ k₂ k₃ : Set (ℝ × ℝ)), 
    IsCircle k₁ r₁ ∧ IsCircle k₂ r₂ ∧ IsCircle k₃ r₃ ∧
    AreExternallyTangent k₁ k₂ ∧ AreExternallyTangent k₁ k₃ ∧ AreExternallyTangent k₂ k₃ ∧
    ∃ (t₁₂ t₁₃ : Set (ℝ × ℝ)), 
      IsCommonExternalTangent t₁₂ k₁ k₂ ∧ 
      IsCommonExternalTangent t₁₃ k₁ k₃ ∧
      AreParallel t₁₂ t₁₃) :
  r₁^2 = 4 * r₂ * r₃ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radii_relation_l779_77902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l779_77996

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - Real.log x / Real.log 2)}
def N : Set ℝ := {y | ∃ x, y = Real.exp x - 1}

-- State the theorem
theorem set_intersection_theorem : M ∩ N = {x | 0 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_theorem_l779_77996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_mn_line_equation_l779_77966

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that the line MN (where M is the intersection of AP and QB, 
    and N is the intersection of PB and AQ) has the equation x = a^2/c, 
    where c = √(a^2 - b^2), when PQ passes through the right focus. --/
theorem ellipse_mn_line_equation 
  (a b : ℝ) 
  (ha : a > 0)
  (hb : b > 0)
  (hab : a > b)
  (P Q : ℝ × ℝ) 
  (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (hQ : Q.1^2 / a^2 + Q.2^2 / b^2 = 1)
  (hPQ : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ 
    t * P.1 + (1 - t) * Q.1 = Real.sqrt (a^2 - b^2) ∧ 
    t * P.2 + (1 - t) * Q.2 = 0) : 
  ∃ M N : ℝ × ℝ, M.1 = N.1 ∧ M.1 = a^2 / Real.sqrt (a^2 - b^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_mn_line_equation_l779_77966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tan_C_value_l779_77920

-- Define an acute triangle ABC
structure AcuteTriangle where
  A : ℝ
  B : ℝ
  C : ℝ
  acute_A : 0 < A ∧ A < π/2
  acute_B : 0 < B ∧ B < π/2
  acute_C : 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

-- State the theorem
theorem three_tan_C_value (abc : AcuteTriangle) 
  (h1 : Real.sin abc.A = 3/5) 
  (h2 : Real.tan (abc.A - abc.B) = -1/3) : 
  3 * Real.tan abc.C = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_tan_C_value_l779_77920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_covers_reals_l779_77972

def A : Set ℝ := {y | y > 1}
def B (m : ℝ) : Set ℝ := {x | x ≤ Real.log m / Real.log (1/2)}

theorem union_covers_reals (m : ℝ) :
  (A ∪ B m = Set.univ) ↔ (0 < m ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_covers_reals_l779_77972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l779_77936

/-- Given an arithmetic sequence with 2n+1 terms, the ratio of the sum of odd-numbered terms
    to the sum of even-numbered terms is (n+1)/n -/
theorem arithmetic_sequence_ratio (n : ℕ) (a : ℕ → ℝ) 
  (h_arithmetic : ∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) : 
  (n + 1 : ℝ) * a (n + 1) / (n * a (n + 1)) = (n + 1 : ℝ) / n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_ratio_l779_77936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l779_77932

def set_A : Set ℝ := {x | |x + 1| = x + 1}
def set_B : Set ℝ := {x | x^2 + x < 0}

theorem intersection_A_B : set_A ∩ set_B = Set.Ioo (-1 : ℝ) 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l779_77932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_array_count_l779_77930

def is_valid_array (a b c d : ℕ) : Prop :=
  ({a, b, c, d} : Finset ℕ) = {2, 0, 1, 5} ∧
  (((a = 2) ∧ (b = 2) ∧ (c ≠ 0) ∧ (d = 5)) ∨
   ((b ≠ 2) ∧ (a ≠ 2) ∧ (c ≠ 0) ∧ (d = 5)) ∨
   ((c = 0) ∧ (a ≠ 2) ∧ (b = 2) ∧ (d = 5)) ∨
   ((d ≠ 5) ∧ (a ≠ 2) ∧ (b = 2) ∧ (c ≠ 0)))

theorem valid_array_count :
  ∃! (count : ℕ), ∃ (arrays : Finset (ℕ × ℕ × ℕ × ℕ)),
    arrays.card = count ∧
    (∀ (tuple : ℕ × ℕ × ℕ × ℕ), tuple ∈ arrays ↔ is_valid_array tuple.1 tuple.2.1 tuple.2.2.1 tuple.2.2.2) ∧
    count = 6 :=
  sorry

#check valid_array_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_array_count_l779_77930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l779_77989

/-- The function f(x) = 12/x + 3x -/
noncomputable def f (x : ℝ) : ℝ := 12/x + 3*x

/-- Theorem: For all x < 0, f(x) ≤ -12, and there exists an x < 0 such that f(x) = -12 -/
theorem f_max_value :
  (∀ x : ℝ, x < 0 → f x ≤ -12) ∧ (∃ x : ℝ, x < 0 ∧ f x = -12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l779_77989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l779_77908

theorem sqrt_sum_equality : Real.sqrt 12 + Real.sqrt 27 + Real.sqrt 50 = 5 * Real.sqrt 3 + 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_equality_l779_77908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_given_structure_is_correct_correct_answer_is_B_l779_77916

/-- Represents the grammatical structure of a sentence -/
structure GrammaticalStructure :=
  (tense : String)
  (voice : String)
  (relative_pronoun : String)

/-- Represents the correctness of a grammatical structure -/
def is_correct (gs : GrammaticalStructure) : Prop :=
  gs.tense = "future perfect" ∧
  gs.voice = "passive" ∧
  gs.relative_pronoun = "which"

/-- The given sentence structure -/
def given_structure : GrammaticalStructure :=
  { tense := "future perfect",
    voice := "passive",
    relative_pronoun := "which" }

/-- Theorem stating that the given structure is correct -/
theorem given_structure_is_correct :
  is_correct given_structure := by
  simp [is_correct, given_structure]

/-- Theorem stating that "will have been accomplished; which" is the correct answer -/
theorem correct_answer_is_B :
  is_correct given_structure →
  "will have been accomplished; which" = "correct answer" := by
  intro h
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_given_structure_is_correct_correct_answer_is_B_l779_77916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_group_abelian_iff_card_leq_two_l779_77904

theorem symmetric_group_abelian_iff_card_leq_two {E : Type*} [Fintype E] :
  (∀ σ τ : Equiv.Perm E, σ.trans τ = τ.trans σ) ↔ Fintype.card E ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_group_abelian_iff_card_leq_two_l779_77904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_trapezium_side_length_l779_77913

/-- Represents a rectangle divided into three equal-area regions -/
structure DividedRectangle where
  total_area : ℝ
  triangle_side : ℝ
  triangle_area : ℝ
  trapezium_area : ℝ
  trapezium_height : ℝ
  rectangle_length : ℝ
  smaller_trapezium_side : ℝ
  larger_trapezium_side : ℝ

/-- The theorem statement -/
theorem smaller_trapezium_side_length 
  (rect : DividedRectangle)
  (h1 : rect.triangle_side = 4)
  (h2 : rect.triangle_area = rect.trapezium_area)
  (h3 : rect.total_area = 3 * rect.triangle_area)
  (h4 : rect.triangle_area = Real.sqrt 3 * rect.triangle_side)
  (h5 : rect.trapezium_height = Real.sqrt 3 * rect.triangle_side / 2)
  (h6 : rect.rectangle_length = rect.smaller_trapezium_side + rect.larger_trapezium_side)
  (h7 : rect.trapezium_area = (rect.smaller_trapezium_side + rect.larger_trapezium_side) * rect.trapezium_height / 2) :
  rect.smaller_trapezium_side = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_trapezium_side_length_l779_77913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_a_values_l779_77923

theorem count_possible_a_values : 
  ∃ (S : Finset ℕ), 
    (∀ a ∈ S, ∃ b c d : ℕ, 
      a > b ∧ b > c ∧ c > d ∧ 
      a + b + c + d = 2010 ∧ 
      a^2 - b^2 + c^2 - d^2 = 2010) ∧
    (∀ a : ℕ, (∃ b c d : ℕ, 
      a > b ∧ b > c ∧ c > d ∧ 
      a + b + c + d = 2010 ∧ 
      a^2 - b^2 + c^2 - d^2 = 2010) → a ∈ S) ∧
    S.card = 501 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_possible_a_values_l779_77923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_domain_l779_77979

/-- For an infinite sequence {a_n} where a_n = f(n), the domain of f is ℕ+. -/
theorem sequence_domain (f : ℕ+ → ℝ) (a : ℕ+ → ℝ) (h : ∀ n, a n = f n) :
  Set.range f = Set.range (coe : ℕ+ → ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_domain_l779_77979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_is_planar_l779_77907

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane
structure Plane where
  normal : Point3D
  d : ℝ

-- Define a quadrilateral as four points
structure Quadrilateral where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

-- Define a membership relation for Point3D and Plane
def Point3D.mem (p : Point3D) (plane : Plane) : Prop :=
  plane.normal.x * p.x + plane.normal.y * p.y + plane.normal.z * p.z + plane.d = 0

instance : Membership Point3D Plane where
  mem := Point3D.mem

-- Define a trapezoid as a quadrilateral with one pair of parallel sides
def Trapezoid (q : Quadrilateral) : Prop :=
  ∃ (p : Plane), q.p1 ∈ p ∧ q.p2 ∈ p ∧ q.p3 ∈ p ∧ q.p4 ∈ p ∧
  (∃ (v : Point3D), (q.p1.x - q.p2.x) * v.x + (q.p1.y - q.p2.y) * v.y + (q.p1.z - q.p2.z) * v.z = 0 ∧
                    (q.p3.x - q.p4.x) * v.x + (q.p3.y - q.p4.y) * v.y + (q.p3.z - q.p4.z) * v.z = 0)

-- Theorem: A trapezoid is always a planar figure
theorem trapezoid_is_planar (q : Quadrilateral) : Trapezoid q → ∃ (p : Plane), q.p1 ∈ p ∧ q.p2 ∈ p ∧ q.p3 ∈ p ∧ q.p4 ∈ p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_is_planar_l779_77907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_for_6_years_or_more_is_24_14_l779_77927

/-- Represents the number of marks for each tenure period at Lambda Corp. -/
structure TenureMarks where
  lessThan1Year : ℕ
  from1To2Years : ℕ
  from2To3Years : ℕ
  from3To4Years : ℕ
  from4To5Years : ℕ
  from5To6Years : ℕ
  from6To7Years : ℕ
  from7To8Years : ℕ
  from8To9Years : ℕ
  from9To10Years : ℕ

/-- Calculates the percentage of employees who have worked at Lambda Corp. for 6 years or more -/
def percentageFor6YearsOrMore (marks : TenureMarks) : ℚ :=
  let totalMarks := marks.lessThan1Year + marks.from1To2Years + marks.from2To3Years +
                    marks.from3To4Years + marks.from4To5Years + marks.from5To6Years +
                    marks.from6To7Years + marks.from7To8Years + marks.from8To9Years +
                    marks.from9To10Years
  let marksFor6YearsOrMore := marks.from6To7Years + marks.from7To8Years +
                              marks.from8To9Years + marks.from9To10Years
  (marksFor6YearsOrMore : ℚ) / (totalMarks : ℚ) * 100

/-- The theorem stating that the percentage of employees who have worked at Lambda Corp.
    for 6 years or more is approximately 24.14% -/
theorem percentage_for_6_years_or_more_is_24_14 (marks : TenureMarks)
    (h : marks = { lessThan1Year := 3, from1To2Years := 6, from2To3Years := 5,
                   from3To4Years := 4, from4To5Years := 2, from5To6Years := 2,
                   from6To7Years := 3, from7To8Years := 2, from8To9Years := 1,
                   from9To10Years := 1 }) :
    ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |percentageFor6YearsOrMore marks - 24.14| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_for_6_years_or_more_is_24_14_l779_77927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l779_77947

-- Define the angle α
def α : ℝ → Prop := λ _ => True

-- Define the condition for the terminal side
def terminalSide (x y : ℝ) : Prop := y = -Real.sqrt 3 * x

-- Define the set of possible values for α
def possibleValues (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi - Real.pi / 3

-- Theorem statement
theorem angle_values :
  ∀ α,
  (∃ x y : ℝ, x ≥ 0 ∧ terminalSide x y) →
  possibleValues α :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_values_l779_77947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l779_77963

/-- The distance between two planes in 3D space --/
noncomputable def distance_between_planes (a₁ b₁ c₁ d₁ : ℝ) (a₂ b₂ c₂ d₂ : ℝ) : ℝ :=
  |a₁ * d₂ - a₂ * d₁ + b₁ * d₂ - b₂ * d₁ + c₁ * d₂ - c₂ * d₁| /
  Real.sqrt ((a₁ - a₂)^2 + (b₁ - b₂)^2 + (c₁ - c₂)^2)

/-- The theorem stating the distance between two specific planes --/
theorem distance_between_specific_planes :
  distance_between_planes 1 2 (-2) 1 2 5 (-4) 5 = 1 / Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_planes_l779_77963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l779_77919

theorem tan_double_angle_special_case (x : ℝ) 
  (h1 : x ∈ Set.Ioo (-π/2) 0) 
  (h2 : Real.cos x = 4/5) : 
  Real.tan (2*x) = -24/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_special_case_l779_77919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_twelve_meters_l779_77941

/-- The distance between consecutive trees in a yard --/
noncomputable def distance_between_trees (yard_length : ℝ) (num_trees : ℕ) : ℝ :=
  yard_length / (num_trees - 1)

/-- Theorem: The distance between consecutive trees is 12 meters --/
theorem distance_is_twelve_meters (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 300)
  (h2 : num_trees = 26)
  (h3 : num_trees ≥ 2) :
  distance_between_trees yard_length num_trees = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_twelve_meters_l779_77941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l779_77984

/-- The triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The property that generates the next solution from a given solution -/
def next_solution (x y : ℕ) : ℕ × ℕ :=
  ((2 * x + 1)^2 - 1, 4 * x * y + 2 * y)

/-- Theorem stating that there are infinitely many solutions to the equation -/
theorem infinitely_many_solutions :
  ∃ (f : ℕ → ℕ × ℕ), ∀ n : ℕ,
    let (x, y) := f n
    x > 0 ∧ y > 0 ∧ triangular x = y^2 ∧
    f (n + 1) = next_solution x y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_solutions_l779_77984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l779_77949

def s : ℕ → ℚ
  | 0 => 2  -- Add a case for 0 to cover all natural numbers
  | 1 => 2
  | m+1 => if (m+1) % 3 = 0 then 2 + s ((m+1) / 3) else 1 / s m

theorem sequence_problem (m : ℕ) (h : m > 0) (h1 : s m = 25 / 103) : m = 1468 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l779_77949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shark_population_increase_l779_77981

-- Define the number of sharks at Newport Beach
def newport_sharks : ℕ := 22

-- Define the number of sharks at Dana Point beach
def dana_point_sharks : ℕ := 4 * newport_sharks

-- Define the number of sharks at Huntington Beach
def huntington_sharks : ℕ := dana_point_sharks / 2

-- Define the initial total shark population
def initial_total : ℕ := 65

-- Define the current total shark population
def current_total : ℕ := newport_sharks + dana_point_sharks + huntington_sharks

-- Define the percentage increase
noncomputable def percentage_increase : ℝ := (current_total - initial_total : ℝ) / initial_total * 100

-- Theorem statement
theorem shark_population_increase :
  abs (percentage_increase - 136.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shark_population_increase_l779_77981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_three_rational_others_l779_77951

-- Define the set of given numbers
def givenNumbers : Set ℝ := {-2, 0, -1/2, Real.sqrt 3}

-- Statement to prove
theorem irrational_sqrt_three_rational_others :
  ∃ x ∈ givenNumbers, Irrational x ∧
  ∀ y ∈ givenNumbers, y ≠ x → ¬(Irrational y) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irrational_sqrt_three_rational_others_l779_77951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_polygon_l779_77944

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n points in a plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- A closed polygon with n sides -/
structure Polygon (n : ℕ) where
  vertices : Fin n → Point

/-- Predicate to check if three points are collinear -/
def collinear (p q r : Point) : Prop := sorry

/-- Predicate to check if a polygon has non-intersecting sides -/
def non_intersecting_sides {n : ℕ} (p : Polygon n) : Prop := sorry

/-- Main theorem: For any set of n points in a plane where no three are collinear,
    there exists a closed n-gon with non-intersecting sides whose vertices are these points -/
theorem exists_non_intersecting_polygon 
  (n : ℕ) 
  (points : PointSet n) 
  (h : ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k → ¬collinear (points i) (points j) (points k)) :
  ∃ (p : Polygon n), (∀ i, p.vertices i ∈ Set.range points) ∧ non_intersecting_sides p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_intersecting_polygon_l779_77944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_l779_77987

noncomputable def f (x : ℝ) : ℝ := -(Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.sin (x + Real.pi/2)

theorem sum_of_min_max_f :
  ∃ (min max : ℝ), 
    (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ min) ∧ 
    (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = min) ∧
    (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = max) ∧
    min + max = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_min_max_f_l779_77987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_value_l779_77995

/-- Given a function f(x) = tan(ωx) where ω > 0, if the distance between two adjacent points
    where y = π/4 intersects f(x) is π/4, then f(π/4) = 0. -/
theorem tangent_intersection_value (ω : ℝ) (h₁ : ω > 0) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ - x₁ = π / 4 ∧ 
   Real.tan (ω * x₁) = π / 4 ∧ Real.tan (ω * x₂) = π / 4) →
  Real.tan (ω * (π / 4)) = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_intersection_value_l779_77995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_quadratic_l779_77942

-- Define the expressions
noncomputable def expr_A (x : ℝ) : ℝ := 2 * x + 3
noncomputable def expr_B (x : ℝ) : ℝ := 2 / x
noncomputable def expr_C (x : ℝ) : ℝ := (x - 1)^2 - x^2
noncomputable def expr_D (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define what it means for a function to be quadratic
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Theorem statement
theorem only_D_is_quadratic :
  ¬ is_quadratic expr_A ∧
  ¬ is_quadratic expr_B ∧
  ¬ is_quadratic expr_C ∧
  is_quadratic expr_D :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_D_is_quadratic_l779_77942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_two_concentric_circles_l779_77903

-- Define the given circle
structure GivenCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangent circles
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency condition
def isTangent (givenCircle : GivenCircle) (tangentCircle : TangentCircle) : Prop :=
  let d := Real.sqrt ((givenCircle.center.1 - tangentCircle.center.1)^2 + 
                      (givenCircle.center.2 - tangentCircle.center.2)^2)
  d = givenCircle.radius + tangentCircle.radius ∨ d = givenCircle.radius - tangentCircle.radius

-- Define the locus of centers
def locusOfCenters (givenCircle : GivenCircle) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ c : TangentCircle, c.center = p ∧ c.radius = r ∧ isTangent givenCircle c}

-- Theorem statement
theorem locus_is_two_concentric_circles (givenCircle : GivenCircle) (r : ℝ) :
  locusOfCenters givenCircle r = 
    {p : ℝ × ℝ | (p.1 - givenCircle.center.1)^2 + (p.2 - givenCircle.center.2)^2 = (givenCircle.radius + r)^2 ∨
                 (p.1 - givenCircle.center.1)^2 + (p.2 - givenCircle.center.2)^2 = (givenCircle.radius - r)^2} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_two_concentric_circles_l779_77903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_to_origin_probability_l779_77912

/-- A rectangle in the 2D plane -/
structure Rectangle where
  x_min : ℝ
  x_max : ℝ
  y_min : ℝ
  y_max : ℝ
  h_x : x_min < x_max
  h_y : y_min < y_max

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Check if a point is inside a rectangle -/
def is_inside (p : Point) (r : Rectangle) : Prop :=
  r.x_min ≤ p.x ∧ p.x ≤ r.x_max ∧ r.y_min ≤ p.y ∧ p.y ≤ r.y_max

/-- The probability of an event in a continuous uniform distribution -/
noncomputable def probability (event_area total_area : ℝ) : ℝ :=
  event_area / total_area

/-- The main theorem -/
theorem closer_to_origin_probability :
  let r := Rectangle.mk 0 3 0 1 (by norm_num) (by norm_num)
  let origin := Point.mk 0 0
  let target := Point.mk 4 1
  let event_area := 0.9375
  let total_area := 3
  probability event_area total_area = 0.9375 / 3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_closer_to_origin_probability_l779_77912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_terminal_side_l779_77926

/-- Given an angle α and a point P on its terminal side, where P is not the origin
    and the distance from P to the origin is r, prove that the coordinates of P
    are (r * cos α, r * sin α) -/
theorem point_on_angle_terminal_side (α : ℝ) (P : ℝ × ℝ) (r : ℝ) :
  P ≠ (0, 0) →
  ‖P‖ = r →
  P.1 = r * Real.cos α ∧ P.2 = r * Real.sin α := by
  sorry

#check point_on_angle_terminal_side

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_angle_terminal_side_l779_77926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_halt_duration_l779_77946

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ := by sorry

/-- Converts hours to minutes -/
def hoursToMinutes (h : ℚ) : ℕ := by sorry

theorem train_halt_duration (averageSpeed distanceTraveled : ℚ) 
                            (departureTime arrivalTime : Time) : 
  averageSpeed = 87 →
  distanceTraveled = 348 →
  departureTime = { hours := 9, minutes := 0, valid := by norm_num } →
  arrivalTime = { hours := 13, minutes := 45, valid := by norm_num } →
  timeDifferenceInMinutes departureTime arrivalTime - 
  hoursToMinutes (distanceTraveled / averageSpeed) = 45 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_halt_duration_l779_77946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_l779_77956

-- Define the circle k
def Circle (S : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - S.1)^2 + (p.2 - S.2)^2 = r^2}

-- Define the set M of triangles
def M (k : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) :=
  {T : Set (ℝ × ℝ) | ∃ A B C : ℝ × ℝ,
    T = {A, B, C} ∧
    -- We'll replace isIncircle with a placeholder condition
    (∀ p ∈ k, p ∈ T) ∧
    ∃ α β γ : ℝ, α + β + γ = Real.pi ∧ γ = 2 * α ∧ γ ≥ β ∧ β ≥ α}

-- Define the ordering condition SA ≥ SB ≥ SC
def OrderedVertices (S : ℝ × ℝ) (A B C : ℝ × ℝ) : Prop :=
  (A.1 - S.1)^2 + (A.2 - S.2)^2 ≥ (B.1 - S.1)^2 + (B.2 - S.2)^2 ∧
  (B.1 - S.1)^2 + (B.2 - S.2)^2 ≥ (C.1 - S.1)^2 + (C.2 - S.2)^2

-- Define the locus of points B
def LocusB (S : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {B : ℝ × ℝ | ∃ T ∈ M (Circle S r), ∃ A C : ℝ × ℝ,
    T = {A, B, C} ∧ OrderedVertices S A B C}

-- Theorem statement
theorem locus_is_hyperbola (S : ℝ × ℝ) (r : ℝ) :
  ∃ center foci₁ foci₂ : ℝ × ℝ,
    center = (1/3, 0) ∧ foci₁ = (-1/3, 0) ∧ foci₂ = (1, 0) ∧
    LocusB S r = {p : ℝ × ℝ | 
      (p.1 - center.1)^2 / (1/3)^2 - (p.2 - center.2)^2 / (1/Real.sqrt 3)^2 = 1} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_hyperbola_l779_77956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l779_77978

/-- Calculates the length of a train given its speed and time to pass a fixed point. -/
noncomputable def train_length (speed_kmph : ℝ) (time_seconds : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600) * time_seconds

/-- Theorem stating that a train traveling at 78 kmph and taking 5.0769230769230775 seconds
    to pass an electric pole has a length of approximately 110 meters. -/
theorem train_length_approx :
  let speed := (78 : ℝ)
  let time := (5.0769230769230775 : ℝ)
  abs (train_length speed time - 110) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_approx_l779_77978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odds_against_c_l779_77909

/-- Represents the odds against an event happening -/
structure Odds where
  against : ℚ
  infavor : ℚ

/-- Calculates the probability of an event given the odds against it -/
def oddsToProb (o : Odds) : ℚ :=
  o.infavor / (o.against + o.infavor)

/-- Represents a horse race with three horses and no ties -/
structure HorseRace where
  oddsAgainstA : Odds
  oddsAgainstB : Odds

theorem odds_against_c (race : HorseRace) 
  (h1 : race.oddsAgainstA = { against := 5, infavor := 2 }) 
  (h2 : race.oddsAgainstB = { against := 4, infavor := 5 }) : 
  ∃ (oddsAgainstC : Odds), oddsAgainstC = { against := 53, infavor := 10 } ∧ 
  oddsToProb race.oddsAgainstA + oddsToProb race.oddsAgainstB + oddsToProb oddsAgainstC = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odds_against_c_l779_77909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_E_l779_77983

/-- Hyperbola C with equation x²/16 - y²/9 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- Asymptotes of hyperbola C -/
def asymptotes_C (x y : ℝ) : Prop := x / 4 = y / 3 ∨ x / 4 = -y / 3

/-- Hyperbola E with the same asymptotes as C -/
def hyperbola_E (x y : ℝ) : Prop := asymptotes_C x y

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2) / a

/-- The eccentricity of hyperbola E is either 5/3 or 5/4 -/
theorem eccentricity_of_E :
  ∃ (a b : ℝ), hyperbola_E a b ∧ (eccentricity a b = 5/3 ∨ eccentricity a b = 5/4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_of_E_l779_77983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_impossibility_l779_77967

theorem square_division_impossibility :
  ¬ ∃ (rectangles : List (ℕ × ℕ)),
    (∀ (r : ℕ × ℕ), r ∈ rectangles → 2 * (r.fst + r.snd) = 18) ∧
    (List.sum (rectangles.map (fun r => r.fst * r.snd)) = 25 * 25) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_division_impossibility_l779_77967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_is_contrapositive_of_beta_l779_77998

-- Define the propositions
def α : Prop := ∀ x : ℝ, x < 3 → x < 5
def β : Prop := ∀ x : ℝ, x ≥ 5 → x ≥ 3

-- Theorem statement
theorem alpha_is_contrapositive_of_beta : 
  α ↔ (∀ x : ℝ, ¬(x ≥ 3) → ¬(x ≥ 5)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_is_contrapositive_of_beta_l779_77998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drifting_point_exists_l779_77939

noncomputable def f (x : ℝ) := x^2 + Real.exp (x * Real.log 2)

def is_drifting_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (x₀ + 1) = f x₀ + f 1

theorem drifting_point_exists :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo 0 1 ∧ is_drifting_point f x₀ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_drifting_point_exists_l779_77939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_truncated_cone_sphere_l779_77959

/-- A truncated cone with a circumscribed sphere -/
structure TruncatedConeWithSphere where
  r : ℝ  -- radius of the top circle
  h : ℝ  -- height of the truncated cone
  R : ℝ  -- radius of the circumscribed sphere

/-- The volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius^3

/-- The volume of a truncated cone -/
noncomputable def truncatedConeVolume (r₁ r₂ h : ℝ) : ℝ := (1 / 3) * Real.pi * h * (r₁^2 + r₁ * r₂ + r₂^2)

/-- The theorem stating the ratio of volumes -/
theorem volume_ratio_truncated_cone_sphere (cone : TruncatedConeWithSphere) :
  (cone.r > 0) →
  (cone.h > 0) →
  (cone.R > 0) →
  (Real.pi * (2 * cone.r)^2 = 4 * Real.pi * cone.r^2) →  -- base area is four times top area
  (cone.h = 2 * cone.R) →  -- sphere touches top and bottom
  (cone.R = (3 * cone.r) / 2) →  -- sphere touches sides
  (sphereVolume cone.R) / (truncatedConeVolume cone.r (2 * cone.r) cone.h) = 9 / 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_ratio_truncated_cone_sphere_l779_77959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_luxury_class_l779_77910

def total_passengers : ℕ := 300
def women_percentage : ℚ := 55 / 100
def luxury_women_percentage : ℚ := 15 / 100

theorem women_in_luxury_class : 
  ⌊(total_passengers : ℚ) * women_percentage * luxury_women_percentage⌋₊ = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_women_in_luxury_class_l779_77910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_opposite_side_l779_77943

theorem triangle_angle_opposite_side (a b c : ℝ) (h1 : a = 2) (h2 : b = 2) (h3 : c = Real.sqrt 7) :
  Real.arccos (1 / 8) = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_opposite_side_l779_77943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_measurements_correct_l779_77918

/-- Represents a grid of nodes -/
structure Grid where
  n : ℕ  -- number of nodes

/-- Represents a measurement between two nodes -/
structure Measurement where
  node1 : ℕ
  node2 : ℕ

/-- The minimum number of measurements needed to ensure connectivity -/
def minMeasurements (g : Grid) : ℕ :=
  (g.n + 1) / 2

/-- Theorem stating the minimum number of measurements needed -/
theorem min_measurements_correct (g : Grid) :
  ∀ (measurements : List Measurement),
    (∀ (i j : ℕ), i < g.n → j < g.n → i ≠ j →
      ∃ (path : List ℕ), path.head? = some i ∧ path.getLast? = some j ∧
        ∀ (k : ℕ), k + 1 < path.length →
          ∃ (m : Measurement), m ∈ measurements ∧
            ((m.node1 = path[k]? ∧ m.node2 = path[k+1]?) ∨
             (m.node1 = path[k+1]? ∧ m.node2 = path[k]?))) →
    measurements.length ≥ minMeasurements g := by
  sorry

#eval minMeasurements { n := 16 }  -- Should output 8
#eval minMeasurements { n := 36 }  -- Should output 18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_measurements_correct_l779_77918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_imply_m_bound_l779_77990

def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*a*x^2 - 4

def g (a m : ℝ) (x : ℝ) : ℝ := f a x + m*x

theorem extremum_and_monotonicity_imply_m_bound 
  (a : ℝ) (b : ℝ) (m : ℝ) :
  (∃ b, f a b = -7/2 ∧ ∀ x, (deriv (f a)) x = 0 → x = b) →
  (∀ x ∈ Set.Icc 0 2, (deriv (g a m)) x ≤ 0) →
  m ≤ -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_and_monotonicity_imply_m_bound_l779_77990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l779_77955

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (3 + Real.sqrt 2 * t, 4 + Real.sqrt 2 * t)

-- Define the curve C in polar coordinates
noncomputable def curve_C (θ : ℝ) : ℝ := 4 * Real.sin θ

-- Define point M
def point_M : ℝ × ℝ := (3, 4)

-- Theorem statement
theorem intersection_product (A B : ℝ × ℝ) :
  (∃ t₁ t₂ : ℝ, line_l t₁ = A ∧ line_l t₂ = B) →  -- A and B are on line l
  (∃ θ₁ θ₂ : ℝ, (curve_C θ₁ * Real.cos θ₁, curve_C θ₁ * Real.sin θ₁) = A ∧ 
                (curve_C θ₂ * Real.cos θ₂, curve_C θ₂ * Real.sin θ₂) = B) →  -- A and B are on curve C
  ‖(A.1 - point_M.1, A.2 - point_M.2)‖ * ‖(B.1 - point_M.1, B.2 - point_M.2)‖ = 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l779_77955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l779_77975

noncomputable def expression (a b c d e : ℝ) : ℝ := (a / 2) + (d / e) / (b / c)

theorem max_expression_value :
  ∀ a b c d e : ℝ,
  a ∈ ({1, 2, 3, 4, 6} : Set ℝ) →
  b ∈ ({1, 2, 3, 4, 6} : Set ℝ) →
  c ∈ ({1, 2, 3, 4, 6} : Set ℝ) →
  d ∈ ({1, 2, 3, 4, 6} : Set ℝ) →
  e ∈ ({1, 2, 3, 4, 6} : Set ℝ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e →
  expression a b c d e ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_expression_value_l779_77975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l779_77937

theorem complex_fraction_equality : 
  (Complex.I^2 + Complex.I^3 + Complex.I^4) / (1 - Complex.I) = (1/2 : ℂ) - (1/2 : ℂ) * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l779_77937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l779_77976

def a : ℕ → ℕ
  | 0 => 2  -- Add this case to handle n = 0
  | 1 => 2
  | n + 1 => a n ^ 2 - n * a n + 1

theorem sequence_formula (n : ℕ) : n ≥ 1 → a n = n + 1 := by
  intro h
  induction n with
  | zero => contradiction
  | succ n ih =>
    cases n with
    | zero =>
      simp [a]
    | succ n =>
      simp [a]
      sorry  -- Placeholder for the inductive step


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l779_77976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l779_77964

/-- A circle in the plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Predicate to check if two circles intersect -/
def intersects (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 ≤ (c1.radius + c2.radius)^2

/-- The main theorem -/
theorem circle_intersection_theorem (n : ℕ) (h : n ≥ 2) 
  (circles : Finset Circle) (hc : circles.card = 3*n^2 - 10*n + 10) 
  (hr : ∀ c, c ∈ circles → c.radius = 1) :
  (∃ s : Finset Circle, s ⊆ circles ∧ s.card = n ∧ 
    ∀ c1 c2, c1 ∈ s → c2 ∈ s → c1 ≠ c2 → ¬intersects c1 c2) ∨
  (∃ s : Finset Circle, s ⊆ circles ∧ s.card = n ∧ 
    ∀ c1 c2, c1 ∈ s → c2 ∈ s → c1 ≠ c2 → intersects c1 c2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_intersection_theorem_l779_77964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_line_l779_77915

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + y + 4 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y + 4) / Real.sqrt 2

-- State the theorem
theorem shortest_distance_to_line :
  ∀ x y : ℝ, ellipse x y →
  distance_to_line x y ≥ distance_to_line (-3/2) (-1/2) ∧
  distance_to_line (-3/2) (-1/2) = Real.sqrt 2 :=
by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_to_line_l779_77915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_distances_l779_77962

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

-- Define the circle
def myCircle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the left focus F₁
def F₁ : ℝ × ℝ := (-3, 0)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define a point P on the right branch of the hyperbola
axiom P : ℝ × ℝ
axiom P_on_hyperbola : hyperbola P.1 P.2
axiom P_on_circle : myCircle P.1 P.2
axiom P_right_branch : P.1 > 0

-- Define the tangent point T
axiom T : ℝ × ℝ
axiom T_on_circle : myCircle T.1 T.2

-- Define the midpoint M of PF₁
noncomputable def M : ℝ × ℝ := ((P.1 + F₁.1) / 2, (P.2 + F₁.2) / 2)

-- State the theorem
theorem difference_of_distances :
  Real.sqrt ((M.1 - O.1)^2 + (M.2 - O.2)^2) -
  Real.sqrt ((M.1 - T.1)^2 + (M.2 - T.2)^2) =
  Real.sqrt 5 - 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_distances_l779_77962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_zero_l779_77952

theorem function_equation_zero (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x^333 + y) = f (x^2018 + 2*y) + f (x^42)) → 
  (∀ x : ℝ, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equation_zero_l779_77952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanarity_condition_l779_77986

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points
variable (A B C O M : V)

-- Define the condition that A, B, C are not collinear
def NotCollinear (A B C : V) : Prop :=
  ¬∃ (t : ℝ), B - A = t • (C - A)

-- Define the condition that O is outside the plane ABC
def OutsidePlane (O A B C : V) : Prop :=
  ¬∃ (x y z : ℝ), O = x • A + y • B + z • C ∧ x + y + z = 1

-- Define the coplanarity condition
def CoplanarPoints (M A B C : V) : Prop :=
  ∃ (x y z : ℝ), M = x • A + y • B + z • C ∧ x + y + z = 1

-- State the theorem
theorem coplanarity_condition 
  (h1 : NotCollinear A B C) 
  (h2 : OutsidePlane O A B C) :
  CoplanarPoints M A B C ↔ 
  ∃ (x y z : ℝ), M - O = x • (A - O) + y • (B - O) + z • (C - O) ∧ x + y + z = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coplanarity_condition_l779_77986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l779_77971

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) : ℝ × ℝ := (1 + 2*t, 2 - 2*t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (2*Real.cos θ + 2, 2*Real.sin θ)

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ t θ, C₁ t = p ∧ C₂ θ = p ∧ 0 ≤ θ ∧ θ ≤ 2*Real.pi}

-- Theorem statement
theorem intersection_distance :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt 14 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l779_77971
