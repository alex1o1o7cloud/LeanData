import Mathlib

namespace NUMINAMATH_CALUDE_linear_equation_passes_through_points_l339_33979

/-- The linear equation passing through points A(1, 2) and B(3, 4) -/
def linear_equation (x y : ℝ) : Prop := y = x + 1

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- Point B -/
def point_B : ℝ × ℝ := (3, 4)

/-- Theorem: The linear equation passes through points A and B -/
theorem linear_equation_passes_through_points :
  linear_equation point_A.1 point_A.2 ∧ linear_equation point_B.1 point_B.2 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_passes_through_points_l339_33979


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l339_33975

-- Define propositions p and q
variable (p q : Prop)

-- Define the original implication and its contrapositive
def original_implication := p → q
def contrapositive := ¬q → ¬p

-- Define necessary and sufficient conditions
def necessary (p q : Prop) := q → p
def sufficient (p q : Prop) := p → q

theorem p_necessary_not_sufficient (h1 : ¬(original_implication p q)) (h2 : contrapositive p q) :
  necessary p q ∧ ¬(sufficient p q) := by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l339_33975


namespace NUMINAMATH_CALUDE_line_intercept_product_l339_33999

/-- Given a line 8x + 5y + c = 0, if the product of its x-intercept and y-intercept is 24,
    then c = ±8√15 -/
theorem line_intercept_product (c : ℝ) : 
  (∃ x y : ℝ, 8*x + 5*y + c = 0 ∧ x * y = 24) → 
  (c = 8 * Real.sqrt 15 ∨ c = -8 * Real.sqrt 15) := by
sorry

end NUMINAMATH_CALUDE_line_intercept_product_l339_33999


namespace NUMINAMATH_CALUDE_mark_recapture_not_suitable_for_centipedes_l339_33902

/-- Represents a method for population quantity experiments -/
inductive PopulationExperimentMethod
| MarkRecapture
| Sampling

/-- Represents an animal type -/
inductive AnimalType
| Centipede
| Rodent

/-- Represents the size of an animal -/
inductive AnimalSize
| Small
| Large

/-- Function to determine if a method is suitable for an animal type -/
def methodSuitability (method : PopulationExperimentMethod) (animal : AnimalType) : Prop :=
  match method, animal with
  | PopulationExperimentMethod.MarkRecapture, AnimalType.Centipede => False
  | PopulationExperimentMethod.Sampling, AnimalType.Centipede => True
  | _, _ => True

/-- Function to determine the size of an animal -/
def animalSize (animal : AnimalType) : AnimalSize :=
  match animal with
  | AnimalType.Centipede => AnimalSize.Small
  | AnimalType.Rodent => AnimalSize.Large

/-- Theorem stating that the mark-recapture method is not suitable for investigating centipedes -/
theorem mark_recapture_not_suitable_for_centipedes :
  ¬(methodSuitability PopulationExperimentMethod.MarkRecapture AnimalType.Centipede) :=
by sorry

end NUMINAMATH_CALUDE_mark_recapture_not_suitable_for_centipedes_l339_33902


namespace NUMINAMATH_CALUDE_dorchester_washed_16_puppies_l339_33961

/-- Represents the number of puppies washed by Dorchester on Wednesday -/
def puppies_washed : ℕ := sorry

/-- Dorchester's daily base pay in cents -/
def daily_base_pay : ℕ := 4000

/-- Pay per puppy washed in cents -/
def pay_per_puppy : ℕ := 225

/-- Total earnings on Wednesday in cents -/
def total_earnings : ℕ := 7600

/-- Theorem stating that Dorchester washed 16 puppies on Wednesday -/
theorem dorchester_washed_16_puppies : 
  puppies_washed = 16 ∧
  total_earnings = daily_base_pay + puppies_washed * pay_per_puppy :=
sorry

end NUMINAMATH_CALUDE_dorchester_washed_16_puppies_l339_33961


namespace NUMINAMATH_CALUDE_factorial_ratio_50_48_l339_33963

theorem factorial_ratio_50_48 : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_50_48_l339_33963


namespace NUMINAMATH_CALUDE_inequality_solution_range_l339_33932

theorem inequality_solution_range (m : ℝ) : 
  (∃ (a b : ℤ), ∀ (x : ℤ), (x : ℝ)^2 + (m + 1) * (x : ℝ) + m < 0 ↔ x = a ∨ x = b) →
  (-2 ≤ m ∧ m < -1) ∨ (3 < m ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l339_33932


namespace NUMINAMATH_CALUDE_carson_gold_stars_l339_33998

/-- Proves that Carson earned 6 gold stars yesterday -/
theorem carson_gold_stars :
  ∀ (yesterday today total : ℕ),
    today = 9 →
    total = 15 →
    total = yesterday + today →
    yesterday = 6 := by
  sorry

end NUMINAMATH_CALUDE_carson_gold_stars_l339_33998


namespace NUMINAMATH_CALUDE_basketball_win_rate_l339_33974

theorem basketball_win_rate (total_games : ℕ) (first_part_games : ℕ) (games_won : ℕ) 
  (remaining_games : ℕ) (target_percentage : ℚ) :
  total_games = first_part_games + remaining_games →
  games_won ≤ first_part_games →
  target_percentage = 3 / 4 →
  ∃ (x : ℕ), x ≤ remaining_games ∧ 
    (games_won + x : ℚ) / total_games = target_percentage ∧
    x = 38 :=
by
  sorry

#check basketball_win_rate 130 80 60 50 (3/4)

end NUMINAMATH_CALUDE_basketball_win_rate_l339_33974


namespace NUMINAMATH_CALUDE_intersecting_lines_l339_33937

theorem intersecting_lines (k : ℚ) : 
  (∃! p : ℚ × ℚ, 
    p.1 + k * p.2 = 0 ∧ 
    2 * p.1 + 3 * p.2 + 8 = 0 ∧ 
    p.1 - p.2 - 1 = 0) → 
  k = -1/2 := by
sorry

end NUMINAMATH_CALUDE_intersecting_lines_l339_33937


namespace NUMINAMATH_CALUDE_mikes_work_hours_l339_33920

/-- Given that Mike worked for a total of 15 hours over 5 days, 
    prove that he worked 3 hours each day. -/
theorem mikes_work_hours (total_hours : ℕ) (total_days : ℕ) 
  (h1 : total_hours = 15) (h2 : total_days = 5) :
  total_hours / total_days = 3 := by
  sorry

end NUMINAMATH_CALUDE_mikes_work_hours_l339_33920


namespace NUMINAMATH_CALUDE_fraction_problem_l339_33955

theorem fraction_problem (n : ℝ) (F : ℝ) (h1 : n = 70.58823529411765) (h2 : 0.85 * F * n = 36) : F = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l339_33955


namespace NUMINAMATH_CALUDE_fifteenth_triangular_number_is_120_and_even_l339_33983

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 15th triangular number is 120 and it is even -/
theorem fifteenth_triangular_number_is_120_and_even :
  triangular_number 15 = 120 ∧ Even (triangular_number 15) := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_triangular_number_is_120_and_even_l339_33983


namespace NUMINAMATH_CALUDE_min_blocks_for_wall_l339_33908

/-- Represents the dimensions of a wall --/
structure WallDimensions where
  length : ℕ
  height : ℕ

/-- Represents the dimensions of a block --/
structure BlockDimensions where
  length : ℚ
  height : ℕ

/-- Calculates the number of blocks needed for a wall --/
def calculateBlocksNeeded (wall : WallDimensions) (block1 : BlockDimensions) (block2 : BlockDimensions) : ℕ :=
  sorry

/-- Theorem stating the minimum number of blocks needed for the specified wall --/
theorem min_blocks_for_wall :
  let wall := WallDimensions.mk 120 8
  let block1 := BlockDimensions.mk 1 1
  let block2 := BlockDimensions.mk (3/2) 1
  calculateBlocksNeeded wall block1 block2 = 648 :=
sorry

end NUMINAMATH_CALUDE_min_blocks_for_wall_l339_33908


namespace NUMINAMATH_CALUDE_constant_term_expansion_l339_33926

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The constant term in the expansion of (1/x + 2x)^6 -/
def constantTerm : ℕ :=
  binomial 6 3 * (2^3)

theorem constant_term_expansion :
  constantTerm = 160 := by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l339_33926


namespace NUMINAMATH_CALUDE_harmonic_quadratic_radical_simplification_l339_33962

theorem harmonic_quadratic_radical_simplification :
  ∃ (x y : ℕ+), (x + y : ℝ) = 11 ∧ (x * y : ℝ) = 28 →
  Real.sqrt (11 + 2 * Real.sqrt 28) = 2 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_quadratic_radical_simplification_l339_33962


namespace NUMINAMATH_CALUDE_eeyore_triangle_problem_l339_33987

/-- A type representing a stick with a length -/
structure Stick :=
  (length : ℝ)

/-- A function to check if three sticks can form a triangle -/
def canFormTriangle (s1 s2 s3 : Stick) : Prop :=
  s1.length + s2.length > s3.length ∧
  s1.length + s3.length > s2.length ∧
  s2.length + s3.length > s1.length

/-- A function to split six sticks into two sets of three, with the three shortest in one set -/
def splitSticks (sticks : Fin 6 → Stick) : (Fin 3 → Stick) × (Fin 3 → Stick) :=
  sorry

theorem eeyore_triangle_problem :
  ∃ (sticks : Fin 6 → Stick),
    (∃ (t1 t2 t3 t4 t5 t6 : Fin 6), canFormTriangle (sticks t1) (sticks t2) (sticks t3) ∧
                                    canFormTriangle (sticks t4) (sticks t5) (sticks t6)) ∧
    let (yellow, green) := splitSticks sticks
    ¬(canFormTriangle (yellow 0) (yellow 1) (yellow 2) ∧
      canFormTriangle (green 0) (green 1) (green 2)) :=
  sorry

end NUMINAMATH_CALUDE_eeyore_triangle_problem_l339_33987


namespace NUMINAMATH_CALUDE_product_of_exponents_l339_33901

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^5 = 270 →
  2^r + 46 = 78 →
  6^s + 5^4 = 1921 →
  p * r * s = 60 := by
  sorry

end NUMINAMATH_CALUDE_product_of_exponents_l339_33901


namespace NUMINAMATH_CALUDE_min_rooms_correct_min_rooms_optimal_l339_33970

/-- The minimum number of rooms required for 100 tourists given k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  if k % 2 = 0
  then 100 * (k / 2 + 1)
  else 100 * ((k - 1) / 2 + 1) + 1

/-- Theorem stating the minimum number of rooms required for 100 tourists -/
theorem min_rooms_correct (k : ℕ) :
  ∀ n : ℕ, n ≥ min_rooms k →
    ∃ (strategy : Fin 100 → Fin n → Option (Fin n)),
      ∀ (permutation : Fin 100 → Fin 100) (renovated : Finset (Fin n)),
        renovated.card = k →
        ∀ i : Fin 100, ∃ j : Fin n,
          strategy (permutation i) j = some j ∧
          j ∉ renovated ∧
          ∀ i' : Fin 100, i' < i →
            ∀ j' : Fin n, strategy (permutation i') j' = some j' → j ≠ j' :=
by sorry

/-- Theorem stating the optimality of the minimum number of rooms -/
theorem min_rooms_optimal (k : ℕ) :
  ∀ n : ℕ, n < min_rooms k →
    ¬∃ (strategy : Fin 100 → Fin n → Option (Fin n)),
      ∀ (permutation : Fin 100 → Fin 100) (renovated : Finset (Fin n)),
        renovated.card = k →
        ∀ i : Fin 100, ∃ j : Fin n,
          strategy (permutation i) j = some j ∧
          j ∉ renovated ∧
          ∀ i' : Fin 100, i' < i →
            ∀ j' : Fin n, strategy (permutation i') j' = some j' → j ≠ j' :=
by sorry

end NUMINAMATH_CALUDE_min_rooms_correct_min_rooms_optimal_l339_33970


namespace NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l339_33922

/-- Represents the number of individuals in each stratum -/
structure StrataSize where
  general : ℕ
  deputy : ℕ
  logistics : ℕ

/-- Represents the sample size for each stratum -/
structure StrataSample where
  general : ℕ
  deputy : ℕ
  logistics : ℕ

/-- The total population size -/
def totalPopulation (s : StrataSize) : ℕ :=
  s.general + s.deputy + s.logistics

/-- The total sample size -/
def totalSample (s : StrataSample) : ℕ :=
  s.general + s.deputy + s.logistics

/-- The probability of selection for an individual in a given stratum -/
def selectionProbability (popSize : ℕ) (sampleSize : ℕ) : ℚ :=
  sampleSize / popSize

theorem stratified_sampling_equal_probability 
  (strata : StrataSize) 
  (sample : StrataSample) 
  (h1 : totalPopulation strata = 160)
  (h2 : strata.general = 112)
  (h3 : strata.deputy = 16)
  (h4 : strata.logistics = 32)
  (h5 : totalSample sample = 20) :
  ∃ (p : ℚ), 
    selectionProbability strata.general sample.general = p ∧
    selectionProbability strata.deputy sample.deputy = p ∧
    selectionProbability strata.logistics sample.logistics = p :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_equal_probability_l339_33922


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l339_33969

theorem quadratic_root_condition (a : ℝ) : 
  (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ x^2 - a*x + 1 = 0) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l339_33969


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l339_33928

theorem arithmetic_expression_equality : 7 ^ 8 - 6 / 2 + 9 ^ 3 + 3 + 12 = 5765542 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l339_33928


namespace NUMINAMATH_CALUDE_cubic_root_interval_l339_33949

theorem cubic_root_interval (a b : ℤ) : 
  (∃ x : ℝ, x^3 - x + 1 = 0 ∧ a < x ∧ x < b) →
  b - a = 1 →
  a + b = -3 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_interval_l339_33949


namespace NUMINAMATH_CALUDE_f_triangle_condition_l339_33905

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^4 - 4*x + m

-- Define the interval [0, 2]
def I : Set ℝ := Set.Icc 0 2

-- Define the triangle existence condition
def triangle_exists (m : ℝ) : Prop :=
  ∃ (a b c : ℝ), a ∈ I ∧ b ∈ I ∧ c ∈ I ∧
    f m a + f m b > f m c ∧
    f m b + f m c > f m a ∧
    f m c + f m a > f m b

-- State the theorem
theorem f_triangle_condition (m : ℝ) :
  triangle_exists m → m > 14 := by sorry

end NUMINAMATH_CALUDE_f_triangle_condition_l339_33905


namespace NUMINAMATH_CALUDE_rs_length_l339_33990

/-- Triangle ABC with altitude CH, inscribed circles tangent points R and S --/
structure SpecialTriangle where
  -- Points
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  H : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  M : ℝ × ℝ
  -- CH is altitude
  altitude : (C.1 - H.1) * (A.1 - B.1) + (C.2 - H.2) * (A.2 - B.2) = 0
  -- R and S are on CH
  r_on_ch : ∃ t : ℝ, R = (1 - t) • C + t • H
  s_on_ch : ∃ t : ℝ, S = (1 - t) • C + t • H
  -- M is midpoint of AB
  midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Given side lengths
  ab_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20
  ac_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 21
  bc_length : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 29

/-- The length of RS in the special triangle is 4 --/
theorem rs_length (t : SpecialTriangle) : Real.sqrt ((t.R.1 - t.S.1)^2 + (t.R.2 - t.S.2)^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_rs_length_l339_33990


namespace NUMINAMATH_CALUDE_octagonal_pyramid_cross_section_distance_l339_33954

-- Define the pyramid and cross sections
structure OctagonalPyramid where
  crossSection1Area : ℝ
  crossSection2Area : ℝ
  planeDistance : ℝ

-- Define the theorem
theorem octagonal_pyramid_cross_section_distance
  (pyramid : OctagonalPyramid)
  (h1 : pyramid.crossSection1Area = 324 * Real.sqrt 2)
  (h2 : pyramid.crossSection2Area = 648 * Real.sqrt 2)
  (h3 : pyramid.planeDistance = 12)
  : ∃ (distance : ℝ), distance = 24 + 12 * Real.sqrt 2 ∧
    distance = (pyramid.planeDistance) / (1 - Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_octagonal_pyramid_cross_section_distance_l339_33954


namespace NUMINAMATH_CALUDE_inequality_solution_set_l339_33934

/-- The solution set of the inequality (a^2-1)x^2-(a-1)x-1 < 0 is ℝ if and only if -3/5 < a ≤ 1 -/
theorem inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ -3/5 < a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l339_33934


namespace NUMINAMATH_CALUDE_cosine_equality_degrees_l339_33991

theorem cosine_equality_degrees (n : ℤ) : ∃ n : ℤ, 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) ∧ n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_degrees_l339_33991


namespace NUMINAMATH_CALUDE_sequence_integer_count_l339_33929

def sequence_term (n : ℕ) : ℚ :=
  15625 / (5 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∀ (m : ℕ), m > 0 →
    ((∀ (n : ℕ), n < m → is_integer (sequence_term n)) ∧
     (∀ (n : ℕ), n ≥ m → ¬ is_integer (sequence_term n)))
    → m = 7) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l339_33929


namespace NUMINAMATH_CALUDE_company_workers_l339_33996

theorem company_workers (total : ℕ) (men : ℕ) (women : ℕ) : 
  (3 * total / 10 = men) →  -- One-third without plan, 40% of those with plan are men
  (3 * total / 5 = men + women) →  -- Total workers
  (men = 120) →  -- Given number of men
  (women = 180) :=
sorry

end NUMINAMATH_CALUDE_company_workers_l339_33996


namespace NUMINAMATH_CALUDE_optimal_selling_price_l339_33943

-- Define the parameters
def purchasePrice : ℝ := 40
def initialSellingPrice : ℝ := 50
def initialQuantitySold : ℝ := 50

-- Define the relationship between price increase and quantity decrease
def priceIncrease : ℝ → ℝ := λ x => x
def quantityDecrease : ℝ → ℝ := λ x => x

-- Define the selling price and quantity sold as functions of price increase
def sellingPrice : ℝ → ℝ := λ x => initialSellingPrice + priceIncrease x
def quantitySold : ℝ → ℝ := λ x => initialQuantitySold - quantityDecrease x

-- Define the revenue function
def revenue : ℝ → ℝ := λ x => sellingPrice x * quantitySold x

-- Define the cost function
def cost : ℝ → ℝ := λ x => purchasePrice * quantitySold x

-- Define the profit function
def profit : ℝ → ℝ := λ x => revenue x - cost x

-- State the theorem
theorem optimal_selling_price :
  ∃ x : ℝ, x = 20 ∧ sellingPrice x = 70 ∧ 
  ∀ y : ℝ, profit y ≤ profit x :=
sorry

end NUMINAMATH_CALUDE_optimal_selling_price_l339_33943


namespace NUMINAMATH_CALUDE_complex_division_result_l339_33977

theorem complex_division_result : 
  let z : ℂ := (3 + 7*I) / I
  (z.re = 7) ∧ (z.im = -3) := by sorry

end NUMINAMATH_CALUDE_complex_division_result_l339_33977


namespace NUMINAMATH_CALUDE_quadratic_equation_in_y_l339_33948

theorem quadratic_equation_in_y (x y : ℝ) 
  (eq1 : 3 * x^2 + 5 * x + 4 * y + 2 = 0)
  (eq2 : 3 * x + y + 4 = 0) : 
  y^2 + 15 * y + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_in_y_l339_33948


namespace NUMINAMATH_CALUDE_alan_market_cost_l339_33921

/-- Calculates the total cost of Alan's market purchase including discount and tax --/
def market_cost (egg_price : ℝ) (egg_quantity : ℕ) 
                (chicken_price : ℝ) (chicken_quantity : ℕ) 
                (milk_price : ℝ) (milk_quantity : ℕ) 
                (bread_price : ℝ) (bread_quantity : ℕ) 
                (chicken_discount : ℕ → ℕ) (tax_rate : ℝ) : ℝ :=
  let egg_cost := egg_price * egg_quantity
  let chicken_cost := chicken_price * (chicken_quantity - chicken_discount chicken_quantity)
  let milk_cost := milk_price * milk_quantity
  let bread_cost := bread_price * bread_quantity
  let subtotal := egg_cost + chicken_cost + milk_cost + bread_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that Alan's market cost is $103.95 --/
theorem alan_market_cost : 
  market_cost 2 20 8 6 4 3 3.5 2 (fun n => n / 4) 0.05 = 103.95 := by
  sorry

end NUMINAMATH_CALUDE_alan_market_cost_l339_33921


namespace NUMINAMATH_CALUDE_solution_problem_l339_33956

theorem solution_problem : ∃ (x y : ℤ), x > y ∧ y > 0 ∧ x + y + x * y = 80 ∧ x = 26 := by
  sorry

end NUMINAMATH_CALUDE_solution_problem_l339_33956


namespace NUMINAMATH_CALUDE_smallest_n_value_l339_33933

/-- The number of ordered quadruplets (a, b, c, d) satisfying the conditions -/
def num_quadruplets : ℕ := 60000

/-- The greatest common divisor of the quadruplets -/
def gcd_value : ℕ := 60

/-- The function that counts the number of ordered quadruplets (a, b, c, d) 
    such that gcd(a, b, c, d) = gcd_value and lcm(a, b, c, d) = n -/
def count_quadruplets (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that 6480 is the smallest value of n 
    satisfying the given conditions -/
theorem smallest_n_value : 
  (∃ n : ℕ, count_quadruplets n = num_quadruplets) →
  (∀ m : ℕ, count_quadruplets m = num_quadruplets → m ≥ 6480) ∧
  (count_quadruplets 6480 = num_quadruplets) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_value_l339_33933


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l339_33981

theorem root_sum_reciprocal (p q r A B C : ℝ) : 
  (p ≠ q ∧ q ≠ r ∧ p ≠ r) →
  (∀ x, x^3 - 24*x^2 + 88*x - 75 = 0 ↔ x = p ∨ x = q ∨ x = r) →
  (∀ s, s ≠ p ∧ s ≠ q ∧ s ≠ r → 
    1 / (s^3 - 24*s^2 + 88*s - 75) = A / (s - p) + B / (s - q) + C / (s - r)) →
  1 / A + 1 / B + 1 / C = 256 :=
by
  sorry

#check root_sum_reciprocal

end NUMINAMATH_CALUDE_root_sum_reciprocal_l339_33981


namespace NUMINAMATH_CALUDE_tangent_line_at_zero_l339_33935

/-- The curve function -/
def f (a b x : ℝ) : ℝ := x^2 + a*x + b

/-- The derivative of the curve function -/
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x + a

/-- The tangent line function -/
def tangent_line (x y : ℝ) : ℝ := x - y + 1

theorem tangent_line_at_zero (a b : ℝ) :
  (∀ x y, y = f a b x → tangent_line x y = 0 → x = 0 ∧ y = b) →
  (f' a 0 = -1) →
  a = -1 ∧ b = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_zero_l339_33935


namespace NUMINAMATH_CALUDE_kittens_at_shelter_l339_33959

def number_of_puppies : ℕ := 32

def number_of_kittens : ℕ := 2 * number_of_puppies + 14

theorem kittens_at_shelter : number_of_kittens = 78 := by
  sorry

end NUMINAMATH_CALUDE_kittens_at_shelter_l339_33959


namespace NUMINAMATH_CALUDE_min_value_theorem_l339_33919

theorem min_value_theorem (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) :
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 400 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l339_33919


namespace NUMINAMATH_CALUDE_solve_for_a_l339_33986

theorem solve_for_a : ∃ a : ℝ, (3 + 2 * a = -1) ∧ (a = -2) := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l339_33986


namespace NUMINAMATH_CALUDE_subset_implies_lower_bound_l339_33960

theorem subset_implies_lower_bound (a : ℝ) : 
  let M := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
  let N := {x : ℝ | x ≤ a}
  M ⊆ N → a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_lower_bound_l339_33960


namespace NUMINAMATH_CALUDE_candy_parade_total_l339_33950

/-- The total number of candy pieces caught by Tabitha and her friends at the Christmas parade -/
theorem candy_parade_total (tabitha stan : ℕ) (julie carlos : ℕ) 
    (h1 : tabitha = 22)
    (h2 : stan = 13)
    (h3 : julie = tabitha / 2)
    (h4 : carlos = 2 * stan) :
  tabitha + stan + julie + carlos = 72 := by
  sorry

end NUMINAMATH_CALUDE_candy_parade_total_l339_33950


namespace NUMINAMATH_CALUDE_modulus_of_z_l339_33909

open Complex

theorem modulus_of_z (z : ℂ) (h : (1 + I) * (1 - z) = 1) : abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l339_33909


namespace NUMINAMATH_CALUDE_tangent_product_l339_33900

theorem tangent_product (α β : Real) (h : α + β = 3 * Real.pi / 4) :
  (1 - Real.tan α) * (1 - Real.tan β) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_l339_33900


namespace NUMINAMATH_CALUDE_kat_weekly_training_hours_l339_33915

/-- Represents Kat's weekly training schedule --/
structure TrainingSchedule where
  strength_sessions : ℕ
  strength_hours_per_session : ℝ
  boxing_sessions : ℕ
  boxing_hours_per_session : ℝ
  cardio_sessions : ℕ
  cardio_hours_per_session : ℝ
  flexibility_sessions : ℕ
  flexibility_hours_per_session : ℝ
  interval_sessions : ℕ
  interval_hours_per_session : ℝ

/-- Calculates the total weekly training hours --/
def total_weekly_hours (schedule : TrainingSchedule) : ℝ :=
  schedule.strength_sessions * schedule.strength_hours_per_session +
  schedule.boxing_sessions * schedule.boxing_hours_per_session +
  schedule.cardio_sessions * schedule.cardio_hours_per_session +
  schedule.flexibility_sessions * schedule.flexibility_hours_per_session +
  schedule.interval_sessions * schedule.interval_hours_per_session

/-- Kat's actual training schedule --/
def kat_schedule : TrainingSchedule := {
  strength_sessions := 3
  strength_hours_per_session := 1
  boxing_sessions := 4
  boxing_hours_per_session := 1.5
  cardio_sessions := 2
  cardio_hours_per_session := 0.5
  flexibility_sessions := 1
  flexibility_hours_per_session := 0.75
  interval_sessions := 1
  interval_hours_per_session := 1.25
}

/-- Theorem stating that Kat's total weekly training time is 12 hours --/
theorem kat_weekly_training_hours :
  total_weekly_hours kat_schedule = 12 := by sorry

end NUMINAMATH_CALUDE_kat_weekly_training_hours_l339_33915


namespace NUMINAMATH_CALUDE_van_rental_cost_l339_33913

/-- Calculates the total cost of van rental given the specified conditions -/
theorem van_rental_cost 
  (daily_rate : ℝ) 
  (mileage_rate : ℝ) 
  (num_days : ℕ) 
  (num_miles : ℕ) 
  (booking_fee : ℝ) 
  (h1 : daily_rate = 30)
  (h2 : mileage_rate = 0.25)
  (h3 : num_days = 3)
  (h4 : num_miles = 450)
  (h5 : booking_fee = 15) :
  daily_rate * num_days + mileage_rate * num_miles + booking_fee = 217.5 := by
  sorry


end NUMINAMATH_CALUDE_van_rental_cost_l339_33913


namespace NUMINAMATH_CALUDE_tangent_sum_simplification_l339_33916

theorem tangent_sum_simplification :
  (Real.tan (10 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (50 * π / 180) + Real.tan (70 * π / 180)) / 
  Real.cos (30 * π / 180) = 
  4 * Real.sin (40 * π / 180) + 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_simplification_l339_33916


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l339_33944

theorem root_condition_implies_m_range :
  ∀ m : ℝ,
  (∀ x : ℝ, x^2 - (3*m + 2)*x + 2*(m + 6) = 0 → x > 3) →
  m ≥ 2 ∧ m < 15/7 := by
sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l339_33944


namespace NUMINAMATH_CALUDE_cubic_sum_nonnegative_l339_33971

theorem cubic_sum_nonnegative (c : ℝ) (X Y : ℝ) 
  (hX : X^2 - c*X - c = 0) 
  (hY : Y^2 - c*Y - c = 0) : 
  X^3 + Y^3 + (X*Y)^3 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_nonnegative_l339_33971


namespace NUMINAMATH_CALUDE_circle_intersection_condition_tangent_length_l339_33941

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_O1 (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2
def line (m x y : ℝ) : Prop := m*x + y - m - 1 = 0

-- Statement 1
theorem circle_intersection_condition (r : ℝ) (h : r > 0) :
  (∀ m : ℝ, ∃ x1 y1 x2 y2 : ℝ, 
    x1 ≠ x2 ∧ 
    circle_O1 x1 y1 r ∧ line m x1 y1 ∧
    circle_O1 x2 y2 r ∧ line m x2 y2) ↔ 
  r > Real.sqrt 2 :=
sorry

-- Statement 2
theorem tangent_length (A B : ℝ × ℝ) :
  (∃ t : ℝ, circle_O (A.1) (A.2) ∧ circle_O (B.1) (B.2) ∧
    (∀ x y : ℝ, circle_O x y → (x - 0)*(A.2 - 2) = (y - 2)*(A.1 - 0) ∧
                               (x - 0)*(B.2 - 2) = (y - 2)*(B.1 - 0))) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_tangent_length_l339_33941


namespace NUMINAMATH_CALUDE_triangle_problem_l339_33988

-- Define the triangles and their properties
def Triangle (A B C : ℝ × ℝ) := True

def is_45_45_90_triangle (A B D : ℝ × ℝ) : Prop :=
  Triangle A B D ∧ 
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (D.1 - A.1)^2 + (D.2 - A.2)^2 ∧
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = (B.1 - D.1)^2 + (B.2 - D.2)^2

def is_30_60_90_triangle (A C D : ℝ × ℝ) : Prop :=
  Triangle A C D ∧ 
  4 * ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 3 * ((D.1 - A.1)^2 + (D.2 - A.2)^2)

-- Define the theorem
theorem triangle_problem (A B C D : ℝ × ℝ) :
  is_45_45_90_triangle A B D →
  is_30_60_90_triangle A C D →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 24 :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l339_33988


namespace NUMINAMATH_CALUDE_subtraction_result_l339_33931

theorem subtraction_result : 2014 - 4102 = -2088 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l339_33931


namespace NUMINAMATH_CALUDE_parabola_properties_l339_33973

/-- Parabola equation -/
def parabola (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

/-- Vertex of the parabola -/
def vertex : ℝ × ℝ := (2, 2)

/-- Axis of symmetry -/
def axis_of_symmetry : ℝ := 2

theorem parabola_properties :
  (∀ x : ℝ, parabola x = -2 * (x - 2)^2 + 2) ∧
  (vertex = (2, 2)) ∧
  (axis_of_symmetry = 2) ∧
  (∀ x : ℝ, x ≥ 2 → ∀ y : ℝ, y > x → parabola y < parabola x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l339_33973


namespace NUMINAMATH_CALUDE_total_classes_is_nine_l339_33995

/-- The number of classes taught by Eduardo and Frankie -/
def total_classes (eduardo_classes : ℕ) (frankie_multiplier : ℕ) : ℕ :=
  eduardo_classes + eduardo_classes * frankie_multiplier

/-- Theorem stating that the total number of classes taught by Eduardo and Frankie is 9 -/
theorem total_classes_is_nine :
  total_classes 3 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_classes_is_nine_l339_33995


namespace NUMINAMATH_CALUDE_double_price_increase_l339_33985

theorem double_price_increase (original_price : ℝ) (h : original_price > 0) :
  (original_price * (1 + 0.06) * (1 + 0.06)) = (original_price * (1 + 0.1236)) := by
  sorry

end NUMINAMATH_CALUDE_double_price_increase_l339_33985


namespace NUMINAMATH_CALUDE_meals_without_restrictions_l339_33978

theorem meals_without_restrictions (total clients vegan kosher gluten_free vegan_kosher vegan_gluten_free kosher_gluten_free vegan_kosher_gluten_free : ℕ) 
  (h1 : total = 50)
  (h2 : vegan = 10)
  (h3 : kosher = 12)
  (h4 : gluten_free = 6)
  (h5 : vegan_kosher = 3)
  (h6 : vegan_gluten_free = 4)
  (h7 : kosher_gluten_free = 2)
  (h8 : vegan_kosher_gluten_free = 1) :
  total - (vegan + kosher + gluten_free - vegan_kosher - vegan_gluten_free - kosher_gluten_free + vegan_kosher_gluten_free) = 30 := by
  sorry

end NUMINAMATH_CALUDE_meals_without_restrictions_l339_33978


namespace NUMINAMATH_CALUDE_square_sum_identity_l339_33945

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l339_33945


namespace NUMINAMATH_CALUDE_min_value_expression_l339_33980

theorem min_value_expression (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 2) (hab : a + b = 2) :
  ∃ (min : ℝ), min = Real.sqrt 10 + Real.sqrt 5 ∧
  ∀ (x : ℝ), (a * c / b) + (c / (a * b)) - (c / 2) + (Real.sqrt 5 / (c - 2)) ≥ x → x ≤ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l339_33980


namespace NUMINAMATH_CALUDE_partner_calculation_l339_33972

theorem partner_calculation (x : ℝ) : 4 * (3 * (x + 2) - 2) = 4 * (3 * x + 4) := by
  sorry

#check partner_calculation

end NUMINAMATH_CALUDE_partner_calculation_l339_33972


namespace NUMINAMATH_CALUDE_cos_two_sum_l339_33925

theorem cos_two_sum (α β : Real) 
  (h1 : Real.sin α + Real.sin β = 1) 
  (h2 : Real.cos α + Real.cos β = 0) : 
  Real.cos (2 * α) + Real.cos (2 * β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_sum_l339_33925


namespace NUMINAMATH_CALUDE_test_total_points_l339_33907

theorem test_total_points : 
  ∀ (total_problems : ℕ) 
    (three_point_problems : ℕ) 
    (four_point_problems : ℕ),
  total_problems = 30 →
  four_point_problems = 10 →
  three_point_problems + four_point_problems = total_problems →
  3 * three_point_problems + 4 * four_point_problems = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_test_total_points_l339_33907


namespace NUMINAMATH_CALUDE_probability_three_common_books_is_32_495_l339_33953

def total_books : ℕ := 12
def books_selected : ℕ := 4

def probability_three_common_books : ℚ :=
  (Nat.choose total_books 3 * Nat.choose (total_books - 3) 1 * Nat.choose (total_books - 4) 1) /
  (Nat.choose total_books books_selected * Nat.choose total_books books_selected)

theorem probability_three_common_books_is_32_495 :
  probability_three_common_books = 32 / 495 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_common_books_is_32_495_l339_33953


namespace NUMINAMATH_CALUDE_no_valid_sequence_exists_l339_33923

theorem no_valid_sequence_exists : ¬ ∃ (seq : Fin 100 → ℤ),
  (∀ i, Odd (seq i)) ∧ 
  (∀ i, i + 4 < 100 → ∃ k, (seq i + seq (i+1) + seq (i+2) + seq (i+3) + seq (i+4)) = k^2) ∧
  (∀ i, i + 8 < 100 → ∃ k, (seq i + seq (i+1) + seq (i+2) + seq (i+3) + seq (i+4) + 
                             seq (i+5) + seq (i+6) + seq (i+7) + seq (i+8)) = k^2) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_sequence_exists_l339_33923


namespace NUMINAMATH_CALUDE_x_value_l339_33942

theorem x_value (x : ℝ) (h1 : x ≠ 0) (h2 : Real.sqrt ((5 * x) / 7) = x) : x = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l339_33942


namespace NUMINAMATH_CALUDE_fraction_zero_l339_33940

theorem fraction_zero (x : ℝ) : x = 1/2 → (2*x - 1) / (x + 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l339_33940


namespace NUMINAMATH_CALUDE_monkey_swinging_speed_l339_33958

/-- Represents the speed and time of a monkey's movement --/
structure MonkeyMovement where
  speed : ℝ
  time : ℝ

/-- Calculates the total distance traveled by the monkey --/
def totalDistance (running : MonkeyMovement) (swinging : MonkeyMovement) : ℝ :=
  running.speed * running.time + swinging.speed * swinging.time

/-- Theorem: The monkey's swinging speed is 10 feet per second --/
theorem monkey_swinging_speed 
  (running_speed : ℝ) 
  (running_time : ℝ) 
  (swinging_time : ℝ) 
  (total_distance : ℝ)
  (h1 : running_speed = 15)
  (h2 : running_time = 5)
  (h3 : swinging_time = 10)
  (h4 : total_distance = 175)
  (h5 : totalDistance 
    { speed := running_speed, time := running_time } 
    { speed := (total_distance - running_speed * running_time) / swinging_time, time := swinging_time } = total_distance) :
  (total_distance - running_speed * running_time) / swinging_time = 10 := by
  sorry

#check monkey_swinging_speed

end NUMINAMATH_CALUDE_monkey_swinging_speed_l339_33958


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l339_33924

theorem rectangular_prism_volume 
  (side_area front_area bottom_area : ℝ)
  (h_side : side_area = 15)
  (h_front : front_area = 10)
  (h_bottom : bottom_area = 6) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    c * a = bottom_area ∧ 
    a * b * c = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l339_33924


namespace NUMINAMATH_CALUDE_birthday_cards_count_l339_33927

def total_amount_spent : ℕ := 70
def cost_per_card : ℕ := 2
def christmas_cards : ℕ := 20

def total_cards : ℕ := total_amount_spent / cost_per_card

def birthday_cards : ℕ := total_cards - christmas_cards

theorem birthday_cards_count : birthday_cards = 15 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cards_count_l339_33927


namespace NUMINAMATH_CALUDE_z_value_theorem_l339_33914

theorem z_value_theorem (x y z k : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ k ≠ 0) 
  (eq : 1/x - 1/y = k * 1/z) : z = x*y / (k*(y-x)) := by
  sorry

end NUMINAMATH_CALUDE_z_value_theorem_l339_33914


namespace NUMINAMATH_CALUDE_cube_edge_length_l339_33989

theorem cube_edge_length (box_edge : ℝ) (num_cubes : ℕ) (cube_edge : ℝ) : 
  box_edge = 1 →
  num_cubes = 8 →
  num_cubes = (box_edge / cube_edge) ^ 3 →
  cube_edge * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_l339_33989


namespace NUMINAMATH_CALUDE_candy_exchange_theorem_l339_33966

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem candy_exchange_theorem :
  (choose 7 5) * (choose 9 5) = 2646 := by sorry

end NUMINAMATH_CALUDE_candy_exchange_theorem_l339_33966


namespace NUMINAMATH_CALUDE_stamp_selection_l339_33965

theorem stamp_selection (n k : ℕ) (stamps : Finset ℕ) : 
  0 < n → 
  stamps.card = k → 
  n ≤ stamps.sum id → 
  stamps.sum id < 2 * k → 
  ∃ s : Finset ℕ, s ⊆ stamps ∧ s.sum id = n := by
  sorry

#check stamp_selection

end NUMINAMATH_CALUDE_stamp_selection_l339_33965


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l339_33984

/-- Eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a = b) →  -- Perpendicular asymptotes condition
  (2 * (a^2 + b^2).sqrt = 8) →  -- Focal length condition
  ((a^2 + b^2).sqrt / a = Real.sqrt 2) := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l339_33984


namespace NUMINAMATH_CALUDE_product_of_12_and_3460_l339_33906

theorem product_of_12_and_3460 : ∃ x : ℕ, x * 12 = x * 240 → 12 * 3460 = 41520 := by
  sorry

end NUMINAMATH_CALUDE_product_of_12_and_3460_l339_33906


namespace NUMINAMATH_CALUDE_final_a_is_three_l339_33936

/-- Given initial values of a and b, compute the final value of a after the operation a = a + b -/
def compute_final_a (initial_a : ℕ) (initial_b : ℕ) : ℕ :=
  initial_a + initial_b

/-- Theorem stating that given the initial conditions a = 1 and b = 2, 
    after the operation a = a + b, the final value of a is 3 -/
theorem final_a_is_three : compute_final_a 1 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_final_a_is_three_l339_33936


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l339_33982

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverselyProportional x₁ y₁)
  (h2 : InverselyProportional x₂ y₂)
  (h3 : x₁ = 30)
  (h4 : y₁ = 8)
  (h5 : y₂ = 24) :
  x₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l339_33982


namespace NUMINAMATH_CALUDE_remainder_theorem_l339_33951

theorem remainder_theorem : 7 * 10^20 + 1^20 ≡ 8 [ZMOD 9] := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l339_33951


namespace NUMINAMATH_CALUDE_complement_of_union_M_N_l339_33997

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3, 4}
def N : Set Nat := {4, 5}

theorem complement_of_union_M_N :
  (U \ (M ∪ N)) = {1, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_M_N_l339_33997


namespace NUMINAMATH_CALUDE_right_angled_triangle_l339_33911

theorem right_angled_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a * Real.cos B + b * Real.cos A = c * Real.sin A →
  a * Real.sin C = b * Real.sin B →
  b * Real.sin C = c * Real.sin A →
  c * Real.sin B = a * Real.sin C →
  A = π / 2 := by sorry

end NUMINAMATH_CALUDE_right_angled_triangle_l339_33911


namespace NUMINAMATH_CALUDE_sin_cos_range_l339_33947

theorem sin_cos_range (x : ℝ) : 
  -1 ≤ Real.sin x + Real.cos x + Real.sin x * Real.cos x ∧ 
  Real.sin x + Real.cos x + Real.sin x * Real.cos x ≤ 1/2 + Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_range_l339_33947


namespace NUMINAMATH_CALUDE_A_greater_than_B_l339_33952

theorem A_greater_than_B (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  a^(2*a) * b^(2*b) * c^(2*c) > a^(b+c) * b^(c+a) * c^(a+b) := by
  sorry

end NUMINAMATH_CALUDE_A_greater_than_B_l339_33952


namespace NUMINAMATH_CALUDE_fraction_simplification_l339_33938

theorem fraction_simplification (y : ℝ) (h : y = 3) :
  (y^8 + 10*y^4 + 25) / (y^4 + 5) = 86 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l339_33938


namespace NUMINAMATH_CALUDE_range_of_a_l339_33993

def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x < 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x < a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ∈ {x : ℝ | x ≥ 5} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l339_33993


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_M_l339_33910

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x, y = 2^x}
def N : Set ℝ := {y | ∃ x, y = x^2}

-- State the theorem
theorem M_intersect_N_eq_M : M ∩ N = M := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_M_l339_33910


namespace NUMINAMATH_CALUDE_alyssa_gave_away_seven_puppies_l339_33903

/-- The number of puppies Alyssa gave to her friends -/
def puppies_given_away (initial : ℕ) (current : ℕ) : ℕ :=
  initial - current

/-- Theorem stating that Alyssa gave away 7 puppies -/
theorem alyssa_gave_away_seven_puppies :
  puppies_given_away 12 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_gave_away_seven_puppies_l339_33903


namespace NUMINAMATH_CALUDE_gdp_scientific_notation_l339_33930

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The GDP value in billions of yuan -/
def gdp_billions : ℝ := 53100

/-- Theorem stating that the GDP in billions is equal to its scientific notation -/
theorem gdp_scientific_notation : 
  to_scientific_notation gdp_billions = ScientificNotation.mk 5.31 12 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_gdp_scientific_notation_l339_33930


namespace NUMINAMATH_CALUDE_exactly_two_points_l339_33992

/-- Given two points A and B in a plane that are 12 units apart, this function
    returns the number of points C such that triangle ABC has a perimeter of 36 units,
    an area of 72 square units, and is isosceles. -/
def count_valid_points (A B : ℝ × ℝ) : ℕ :=
  sorry

/-- The main theorem stating that there are exactly two points C satisfying the conditions. -/
theorem exactly_two_points (A B : ℝ × ℝ) 
    (h_distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 12) : 
    count_valid_points A B = 2 :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_points_l339_33992


namespace NUMINAMATH_CALUDE_circle_radii_order_l339_33994

theorem circle_radii_order (rA rB rC : ℝ) : 
  rA = Real.sqrt 10 →
  2 * Real.pi * rB = 10 * Real.pi →
  Real.pi * rC^2 = 25 * Real.pi →
  rA ≤ rB ∧ rB ≤ rC := by
  sorry

end NUMINAMATH_CALUDE_circle_radii_order_l339_33994


namespace NUMINAMATH_CALUDE_inequality_implication_l339_33964

theorem inequality_implication (x y : ℝ) (h : x > y) : (1/2 : ℝ)^x < (1/2 : ℝ)^y := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l339_33964


namespace NUMINAMATH_CALUDE_correct_bullseyes_needed_l339_33912

/-- Represents the archery contest scenario -/
structure ArcheryContest where
  totalShots : Nat
  shotsCompleted : Nat
  pointsAhead : Nat
  minPointsPerShot : Nat

/-- Calculates the minimum number of bullseyes needed to secure victory -/
def minBullseyesNeeded (contest : ArcheryContest) : Nat :=
  sorry

/-- Theorem stating the correct number of bullseyes needed -/
theorem correct_bullseyes_needed (contest : ArcheryContest) 
  (h1 : contest.totalShots = 150)
  (h2 : contest.shotsCompleted = 75)
  (h3 : contest.pointsAhead = 70)
  (h4 : contest.minPointsPerShot = 2) :
  minBullseyesNeeded contest = 67 := by
  sorry

end NUMINAMATH_CALUDE_correct_bullseyes_needed_l339_33912


namespace NUMINAMATH_CALUDE_square_sum_equals_ten_l339_33976

theorem square_sum_equals_ten (a b : ℝ) 
  (h1 : a + 3 = (b - 1)^2) 
  (h2 : b + 3 = (a - 1)^2) 
  (h3 : a ≠ b) : 
  a^2 + b^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_ten_l339_33976


namespace NUMINAMATH_CALUDE_point_in_region_l339_33917

-- Define the plane region
def in_region (x y : ℝ) : Prop := 2 * x + y - 6 < 0

-- Theorem to prove
theorem point_in_region : in_region 0 1 := by
  sorry

end NUMINAMATH_CALUDE_point_in_region_l339_33917


namespace NUMINAMATH_CALUDE_olaf_toy_cars_l339_33967

/-- Represents the toy car collection problem -/
def toy_car_problem (initial_collection : ℕ) (grandpa_factor : ℕ) (dad_gift : ℕ) 
  (mum_dad_diff : ℕ) (auntie_gift : ℕ) (final_total : ℕ) : Prop :=
  ∃ (uncle_gift : ℕ),
    initial_collection + (grandpa_factor * uncle_gift) + uncle_gift + 
    dad_gift + (dad_gift + mum_dad_diff) + auntie_gift = final_total ∧
    auntie_gift - uncle_gift = 1

/-- The specific instance of the toy car problem -/
theorem olaf_toy_cars : 
  toy_car_problem 150 2 10 5 6 196 := by
  sorry

end NUMINAMATH_CALUDE_olaf_toy_cars_l339_33967


namespace NUMINAMATH_CALUDE_eggs_left_over_l339_33904

def total_eggs : ℕ := 114
def carton_size : ℕ := 15

theorem eggs_left_over : total_eggs % carton_size = 9 := by
  sorry

end NUMINAMATH_CALUDE_eggs_left_over_l339_33904


namespace NUMINAMATH_CALUDE_angle_measure_l339_33957

theorem angle_measure : 
  ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ (180 - x) = 3 * x + 10 → x = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l339_33957


namespace NUMINAMATH_CALUDE_hundred_chicken_equations_l339_33918

def hundred_chicken_problem (x y : ℝ) : Prop :=
  (x + y + 81 = 100) ∧ (5*x + 3*y + (1/3) * 81 = 100)

theorem hundred_chicken_equations :
  ∀ x y : ℝ,
  (x ≥ 0) → (y ≥ 0) →
  (x + y + 81 = 100) →
  (5*x + 3*y + 27 = 100) →
  hundred_chicken_problem x y :=
by
  sorry

end NUMINAMATH_CALUDE_hundred_chicken_equations_l339_33918


namespace NUMINAMATH_CALUDE_train_length_l339_33939

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 60 → time_s = 6 → 
  ∃ (length_m : ℝ), abs (length_m - 100.02) < 0.01 := by sorry

end NUMINAMATH_CALUDE_train_length_l339_33939


namespace NUMINAMATH_CALUDE_geometric_progression_with_conditions_l339_33968

/-- A geometric progression of four terms satisfying specific conditions -/
theorem geometric_progression_with_conditions :
  ∃ (b₁ b₂ b₃ b₄ : ℝ),
    -- The sequence forms a geometric progression
    (∃ (q : ℝ), b₂ = b₁ * q ∧ b₃ = b₁ * q^2 ∧ b₄ = b₁ * q^3) ∧
    -- The third term is 9 greater than the first term
    b₃ - b₁ = 9 ∧
    -- The second term is 18 greater than the fourth term
    b₂ - b₄ = 18 ∧
    -- The sequence is (3, -6, 12, -24)
    b₁ = 3 ∧ b₂ = -6 ∧ b₃ = 12 ∧ b₄ = -24 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_with_conditions_l339_33968


namespace NUMINAMATH_CALUDE_mean_temperature_is_88_point_2_l339_33946

def temperatures : List ℝ := [78, 80, 82, 85, 88, 90, 92, 95, 97, 95]

theorem mean_temperature_is_88_point_2 :
  (temperatures.sum / temperatures.length : ℝ) = 88.2 := by
  sorry

end NUMINAMATH_CALUDE_mean_temperature_is_88_point_2_l339_33946
