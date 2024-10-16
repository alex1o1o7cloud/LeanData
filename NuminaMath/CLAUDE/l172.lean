import Mathlib

namespace NUMINAMATH_CALUDE_parallel_lines_l172_17233

/-- Two lines are parallel if and only if their slopes are equal -/
def parallel (m₁ n₁ c₁ m₂ n₂ c₂ : ℝ) : Prop :=
  m₁ * n₂ = m₂ * n₁ ∧ m₁ * c₂ ≠ n₁ * c₂

theorem parallel_lines (a : ℝ) :
  parallel 1 (2*a) (-1) (a-1) (-a) 1 ↔ a = 1/2 :=
sorry

#check parallel_lines

end NUMINAMATH_CALUDE_parallel_lines_l172_17233


namespace NUMINAMATH_CALUDE_min_cost_container_l172_17240

/-- Represents the dimensions and costs of a rectangular container. -/
structure Container where
  volume : ℝ
  height : ℝ
  baseCost : ℝ
  sideCost : ℝ

/-- Calculates the total cost of constructing the container. -/
def totalCost (c : Container) (length width : ℝ) : ℝ :=
  c.baseCost * length * width + c.sideCost * 2 * (length + width) * c.height

/-- Theorem stating that the minimum cost to construct the given container is 1600 yuan. -/
theorem min_cost_container (c : Container) 
  (h_volume : c.volume = 4)
  (h_height : c.height = 1)
  (h_baseCost : c.baseCost = 200)
  (h_sideCost : c.sideCost = 100) :
  ∃ (cost : ℝ), cost = 1600 ∧ ∀ (length width : ℝ), length * width * c.height = c.volume → 
    totalCost c length width ≥ cost := by
  sorry

#check min_cost_container

end NUMINAMATH_CALUDE_min_cost_container_l172_17240


namespace NUMINAMATH_CALUDE_julia_tag_game_l172_17278

/-- 
Given that Julia played tag with a total of 18 kids over two days,
and she played with 14 kids on Tuesday, prove that she played with 4 kids on Monday.
-/
theorem julia_tag_game (total : ℕ) (tuesday : ℕ) (monday : ℕ) 
    (h1 : total = 18) 
    (h2 : tuesday = 14) 
    (h3 : total = monday + tuesday) : 
  monday = 4 := by
  sorry

end NUMINAMATH_CALUDE_julia_tag_game_l172_17278


namespace NUMINAMATH_CALUDE_eventually_one_first_l172_17247

/-- Represents a permutation of integers from 1 to 1993 -/
def Permutation := Fin 1993 → Fin 1993

/-- The reversal operation on a permutation -/
def reverseOperation (p : Permutation) : Permutation :=
  sorry

/-- Predicate to check if 1 is the first element in the permutation -/
def isOneFirst (p : Permutation) : Prop :=
  p 0 = 0

/-- Main theorem: The reversal operation will eventually make 1 the first element -/
theorem eventually_one_first (p : Permutation) : 
  ∃ n : ℕ, isOneFirst (n.iterate reverseOperation p) :=
sorry

end NUMINAMATH_CALUDE_eventually_one_first_l172_17247


namespace NUMINAMATH_CALUDE_all_N_composite_l172_17207

def N (n : ℕ) : ℕ := 200 * 10^n + 88 * ((10^n - 1) / 9) + 21

theorem all_N_composite (n : ℕ) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b := by
  sorry

end NUMINAMATH_CALUDE_all_N_composite_l172_17207


namespace NUMINAMATH_CALUDE_train_length_calculation_l172_17214

/-- Prove that a train traveling at 60 km/hour that takes 30 seconds to pass a bridge of 140 meters in length has a length of approximately 360.1 meters. -/
theorem train_length_calculation (train_speed : ℝ) (bridge_pass_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 →
  bridge_pass_time = 30 →
  bridge_length = 140 →
  ∃ (train_length : ℝ), abs (train_length - 360.1) < 0.1 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l172_17214


namespace NUMINAMATH_CALUDE_sum_squares_regression_example_l172_17265

/-- Given a total sum of squared deviations and a correlation coefficient,
    calculate the sum of squares due to regression -/
def sum_squares_regression (total_sum_squared_dev : ℝ) (correlation_coeff : ℝ) : ℝ :=
  total_sum_squared_dev * correlation_coeff^2

/-- Theorem stating that given the specified conditions, 
    the sum of squares due to regression is 72 -/
theorem sum_squares_regression_example :
  sum_squares_regression 120 0.6 = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_regression_example_l172_17265


namespace NUMINAMATH_CALUDE_male_students_count_l172_17252

theorem male_students_count (total : ℕ) (male : ℕ) (female : ℕ) :
  total = 48 →
  female = (4 * male) / 5 + 3 →
  total = male + female →
  male = 25 := by
sorry

end NUMINAMATH_CALUDE_male_students_count_l172_17252


namespace NUMINAMATH_CALUDE_cubes_in_box_percentage_l172_17287

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (b : BoxDimensions) : ℕ :=
  b.length * b.width * b.height

/-- Calculates the number of cubes that can fit along a dimension -/
def cubesFit (dimension : ℕ) (cubeSize : ℕ) : ℕ :=
  dimension / cubeSize

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (b : BoxDimensions) (cubeSize : ℕ) : ℕ :=
  (cubesFit b.length cubeSize) * (cubesFit b.width cubeSize) * (cubesFit b.height cubeSize)

/-- Calculates the volume occupied by cubes -/
def cubesVolume (numCubes : ℕ) (cubeSize : ℕ) : ℕ :=
  numCubes * (cubeSize * cubeSize * cubeSize)

/-- Calculates the percentage of box volume occupied by cubes -/
def percentageOccupied (boxVol : ℕ) (cubesVol : ℕ) : ℚ :=
  (cubesVol : ℚ) / (boxVol : ℚ) * 100

/-- Theorem: The percentage of volume occupied by 3-inch cubes in an 8x6x12 inch box is 75% -/
theorem cubes_in_box_percentage :
  let box := BoxDimensions.mk 8 6 12
  let cubeSize := 3
  let boxVol := boxVolume box
  let numCubes := totalCubes box cubeSize
  let cubesVol := cubesVolume numCubes cubeSize
  percentageOccupied boxVol cubesVol = 75 := by
  sorry

end NUMINAMATH_CALUDE_cubes_in_box_percentage_l172_17287


namespace NUMINAMATH_CALUDE_symmetric_point_theorem_l172_17234

/-- Given a point (a, b) and a line x + y = 0, find the symmetric point -/
def symmetricPoint (a b : ℝ) : ℝ × ℝ :=
  (-b, -a)

/-- The theorem states that the point symmetric to (2, 5) with respect to x + y = 0 is (-5, -2) -/
theorem symmetric_point_theorem :
  symmetricPoint 2 5 = (-5, -2) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_theorem_l172_17234


namespace NUMINAMATH_CALUDE_worker_efficiency_l172_17261

/-- Given two workers A and B, where A is half as efficient as B,
    this theorem proves that if they together complete a job in 13 days,
    then B alone can complete the job in 19.5 days. -/
theorem worker_efficiency (A B : ℝ) (h1 : A = (1/2) * B) (h2 : (A + B) * 13 = 1) :
  (1 / B) = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_worker_efficiency_l172_17261


namespace NUMINAMATH_CALUDE_village_foods_sales_l172_17295

/-- Represents the pricing structure for lettuce -/
structure LettucePricing where
  first : Float
  second : Float
  additional : Float

/-- Represents the pricing structure for tomatoes -/
structure TomatoPricing where
  firstTwo : Float
  nextTwo : Float
  additional : Float

/-- Calculates the total sales per month for Village Foods -/
def totalSalesPerMonth (
  customersPerMonth : Nat
) (
  lettucePerCustomer : Nat
) (
  tomatoesPerCustomer : Nat
) (
  lettucePricing : LettucePricing
) (
  tomatoPricing : TomatoPricing
) (
  discountThreshold : Float
) (
  discountRate : Float
) : Float :=
  sorry

/-- Theorem stating that the total sales per month is $2350 -/
theorem village_foods_sales :
  totalSalesPerMonth
    500  -- customers per month
    2    -- lettuce per customer
    4    -- tomatoes per customer
    { first := 1.50, second := 1.00, additional := 0.75 }  -- lettuce pricing
    { firstTwo := 0.60, nextTwo := 0.50, additional := 0.40 }  -- tomato pricing
    10.00  -- discount threshold
    0.10   -- discount rate
  = 2350.00 :=
by sorry

end NUMINAMATH_CALUDE_village_foods_sales_l172_17295


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l172_17299

theorem sqrt_sum_equals_twelve : 
  Real.sqrt ((5 - 3 * Real.sqrt 2)^2) + Real.sqrt ((5 + 3 * Real.sqrt 2)^2) + 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_twelve_l172_17299


namespace NUMINAMATH_CALUDE_count_numbers_with_property_l172_17209

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def satisfies_property (n : ℕ) : Prop :=
  is_two_digit n ∧ (n + reverse_digits n) % 13 = 0

theorem count_numbers_with_property :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_property n) ∧ S.card = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_property_l172_17209


namespace NUMINAMATH_CALUDE_double_wardrobe_socks_l172_17241

/-- Represents the number of items in Jonas' wardrobe -/
structure Wardrobe :=
  (socks : ℕ)
  (shoes : ℕ)
  (pants : ℕ)
  (tshirts : ℕ)

/-- Calculates the total number of individual items in the wardrobe -/
def totalItems (w : Wardrobe) : ℕ :=
  w.socks * 2 + w.shoes * 2 + w.pants + w.tshirts

/-- Calculates the number of sock pairs needed to double the wardrobe -/
def sockPairsNeeded (w : Wardrobe) : ℕ :=
  totalItems w

theorem double_wardrobe_socks (w : Wardrobe) 
  (h1 : w.socks = 20)
  (h2 : w.shoes = 5)
  (h3 : w.pants = 10)
  (h4 : w.tshirts = 10) :
  sockPairsNeeded w = 35 := by
  sorry

#eval sockPairsNeeded { socks := 20, shoes := 5, pants := 10, tshirts := 10 }

end NUMINAMATH_CALUDE_double_wardrobe_socks_l172_17241


namespace NUMINAMATH_CALUDE_union_intersection_relation_l172_17216

theorem union_intersection_relation (M N : Set α) : 
  (∃ (x : α), x ∈ M ∩ N → x ∈ M ∪ N) ∧ 
  (∃ (M N : Set α), (∃ (x : α), x ∈ M ∪ N) ∧ M ∩ N = ∅) :=
by sorry

end NUMINAMATH_CALUDE_union_intersection_relation_l172_17216


namespace NUMINAMATH_CALUDE_parallel_condition_l172_17211

/-- Two lines are parallel if their slopes are equal -/
def are_parallel (m₁ n₁ : ℝ) (m₂ n₂ : ℝ) : Prop := m₁ * n₂ = m₂ * n₁

/-- Definition of line l₁ -/
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y - 1 = 0

/-- Definition of line l₂ -/
def l₂ (a : ℝ) (x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

theorem parallel_condition (a : ℝ) :
  (a = 1 → are_parallel a 2 1 (a + 1)) ∧
  (∃ b : ℝ, b ≠ 1 ∧ are_parallel b 2 1 (b + 1)) :=
by sorry

end NUMINAMATH_CALUDE_parallel_condition_l172_17211


namespace NUMINAMATH_CALUDE_constant_dot_product_l172_17215

-- Define the curve E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define point D
def D : ℝ × ℝ := (-2, 0)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from D to a point
def vector_DA (A : ℝ × ℝ) : ℝ × ℝ := (A.1 - D.1, A.2 - D.2)

theorem constant_dot_product :
  ∀ A B : ℝ × ℝ, A ∈ E → B ∈ E →
  dot_product (vector_DA A) (vector_DA B) = 3 := by sorry

end NUMINAMATH_CALUDE_constant_dot_product_l172_17215


namespace NUMINAMATH_CALUDE_line_l_properties_l172_17200

/-- A line passing through (-2, 1) with y-intercept twice the x-intercept -/
def line_l (x y : ℝ) : Prop := 2*x + y + 3 = 0

theorem line_l_properties :
  (∃ x y : ℝ, line_l x y ∧ x = -2 ∧ y = 1) ∧
  (∃ a : ℝ, a ≠ 0 → line_l a 0 ∧ line_l 0 (2*a)) :=
sorry

end NUMINAMATH_CALUDE_line_l_properties_l172_17200


namespace NUMINAMATH_CALUDE_correct_prediction_probability_l172_17250

def num_monday_classes : ℕ := 5
def num_tuesday_classes : ℕ := 6
def total_classes : ℕ := num_monday_classes + num_tuesday_classes
def correct_predictions : ℕ := 7
def monday_correct_predictions : ℕ := 3

theorem correct_prediction_probability :
  (Nat.choose num_monday_classes monday_correct_predictions * (1/2)^num_monday_classes) *
  (Nat.choose num_tuesday_classes (correct_predictions - monday_correct_predictions) * (1/2)^num_tuesday_classes) /
  (Nat.choose total_classes correct_predictions * (1/2)^total_classes) = 5/11 := by
sorry

end NUMINAMATH_CALUDE_correct_prediction_probability_l172_17250


namespace NUMINAMATH_CALUDE_group_dance_arrangements_l172_17270

/-- The number of boys in the group dance -/
def num_boys : ℕ := 10

/-- The number of girls in the group dance -/
def num_girls : ℕ := 10

/-- The total number of people in the group dance -/
def total_people : ℕ := num_boys + num_girls

/-- The number of columns in the group dance -/
def num_columns : ℕ := 2

/-- The number of arrangements when boys and girls are in separate columns -/
def separate_columns_arrangements : ℕ := 2 * (Nat.factorial num_boys)^2

/-- The number of arrangements when boys and girls can stand in any column -/
def mixed_columns_arrangements : ℕ := Nat.factorial total_people

/-- The number of pairings when boys and girls are in separate columns and internal order doesn't matter -/
def pairings_separate_columns : ℕ := 2 * Nat.factorial num_boys

theorem group_dance_arrangements :
  (separate_columns_arrangements = 2 * (Nat.factorial num_boys)^2) ∧
  (mixed_columns_arrangements = Nat.factorial total_people) ∧
  (pairings_separate_columns = 2 * Nat.factorial num_boys) :=
sorry

end NUMINAMATH_CALUDE_group_dance_arrangements_l172_17270


namespace NUMINAMATH_CALUDE_rowing_time_ratio_l172_17262

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 
    given the man's rowing speed in still water and the current speed. -/
theorem rowing_time_ratio 
  (man_speed : ℝ) 
  (current_speed : ℝ) 
  (h1 : man_speed = 3.9)
  (h2 : current_speed = 1.3) :
  (man_speed - current_speed) / (man_speed + current_speed) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rowing_time_ratio_l172_17262


namespace NUMINAMATH_CALUDE_only_23_is_prime_l172_17279

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 0 → d ∣ n → d = 1 ∨ d = n

theorem only_23_is_prime :
  isPrime 23 ∧
  ¬isPrime 20 ∧
  ¬isPrime 21 ∧
  ¬isPrime 25 ∧
  ¬isPrime 27 :=
by
  sorry

end NUMINAMATH_CALUDE_only_23_is_prime_l172_17279


namespace NUMINAMATH_CALUDE_divisible_by_ten_l172_17223

theorem divisible_by_ten : ∃ k : ℕ, 11^11 + 12^12 + 13^13 = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_ten_l172_17223


namespace NUMINAMATH_CALUDE_no_solution_iff_m_leq_two_l172_17238

/-- The system of inequalities has no solution if and only if m ≤ 2 -/
theorem no_solution_iff_m_leq_two (m : ℝ) :
  (∀ x : ℝ, ¬(x - 2 < 3*x - 6 ∧ x < m)) ↔ m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_leq_two_l172_17238


namespace NUMINAMATH_CALUDE_quadratic_negative_root_l172_17255

theorem quadratic_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ a * x^2 + 4 * x + 1 = 0) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_negative_root_l172_17255


namespace NUMINAMATH_CALUDE_package_weight_problem_l172_17285

theorem package_weight_problem (x y z w : ℝ) 
  (h1 : x + y + z = 150)
  (h2 : y + z + w = 160)
  (h3 : z + w + x = 170) :
  x + y + z + w = 160 := by
sorry

end NUMINAMATH_CALUDE_package_weight_problem_l172_17285


namespace NUMINAMATH_CALUDE_water_added_proof_l172_17266

/-- The amount of water added to a pool given initial and final amounts -/
def water_added (initial : Real) (final : Real) : Real :=
  final - initial

/-- Theorem: Given an initial amount of 1 bucket and a final amount of 9.8 buckets,
    the amount of water added later is 8.8 buckets -/
theorem water_added_proof :
  water_added 1 9.8 = 8.8 := by
  sorry

end NUMINAMATH_CALUDE_water_added_proof_l172_17266


namespace NUMINAMATH_CALUDE_blanket_folding_ratio_l172_17297

theorem blanket_folding_ratio (initial_thickness final_thickness : ℝ) 
  (num_folds : ℕ) (ratio : ℝ) 
  (h1 : initial_thickness = 3)
  (h2 : final_thickness = 48)
  (h3 : num_folds = 4)
  (h4 : final_thickness = initial_thickness * ratio ^ num_folds) :
  ratio = 2 := by
sorry

end NUMINAMATH_CALUDE_blanket_folding_ratio_l172_17297


namespace NUMINAMATH_CALUDE_common_root_equations_unique_integer_solution_l172_17288

theorem common_root_equations (x p : ℤ) : 
  (3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0) ↔ (p = 3 ∧ x = 1) :=
by sorry

theorem unique_integer_solution : 
  ∃! p : ℤ, ∃ x : ℤ, 3 * x^2 - 4 * x + p - 2 = 0 ∧ x^2 - 2 * p * x + 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_equations_unique_integer_solution_l172_17288


namespace NUMINAMATH_CALUDE_range_of_x_minus_cos_y_l172_17236

theorem range_of_x_minus_cos_y :
  ∀ x y : ℝ, x^2 + 2 * Real.cos y = 1 →
  ∃ z : ℝ, z = x - Real.cos y ∧ -1 ≤ z ∧ z ≤ Real.sqrt 3 + 1 ∧
  (∃ x₁ y₁ : ℝ, x₁^2 + 2 * Real.cos y₁ = 1 ∧ x₁ - Real.cos y₁ = -1) ∧
  (∃ x₂ y₂ : ℝ, x₂^2 + 2 * Real.cos y₂ = 1 ∧ x₂ - Real.cos y₂ = Real.sqrt 3 + 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_minus_cos_y_l172_17236


namespace NUMINAMATH_CALUDE_find_b_plus_c_l172_17232

theorem find_b_plus_c (a b c d : ℚ) 
  (h1 : a * b + a * c + b * d + c * d = 40) 
  (h2 : a + d = 6) : 
  b + c = 20 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_b_plus_c_l172_17232


namespace NUMINAMATH_CALUDE_average_of_series_l172_17273

/-- The average of a series z, 3z, 5z, 9z, and 17z is 7z -/
theorem average_of_series (z : ℝ) : (z + 3*z + 5*z + 9*z + 17*z) / 5 = 7*z := by
  sorry

end NUMINAMATH_CALUDE_average_of_series_l172_17273


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l172_17293

theorem partial_fraction_decomposition (A B C : ℝ) :
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 3 →
    1 / (x^3 - x^2 - 21*x + 45) = A / (x + 5) + B / (x - 3) + C / ((x - 3)^2)) →
  A = 1/64 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l172_17293


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l172_17280

theorem factorization_of_quadratic (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l172_17280


namespace NUMINAMATH_CALUDE_square_overlap_percentage_l172_17260

/-- The percentage of overlap between two squares forming a rectangle -/
theorem square_overlap_percentage (s1 s2 l w : ℝ) (h1 : s1 = 10) (h2 : s2 = 15) 
  (h3 : l = 25) (h4 : w = 20) : 
  (min s1 s2)^2 / (l * w) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_square_overlap_percentage_l172_17260


namespace NUMINAMATH_CALUDE_circle_circumference_inscribed_rectangle_l172_17289

theorem circle_circumference_inscribed_rectangle (a b r : ℝ) (h1 : a = 9) (h2 : b = 12) 
  (h3 : r * r = (a * a + b * b) / 4) : 2 * π * r = 15 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_circumference_inscribed_rectangle_l172_17289


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l172_17227

theorem digit_sum_puzzle : ∀ (a b c d e f g : ℕ),
  a ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  b ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  c ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  d ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  e ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  f ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  g ∈ ({0, 1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g →
  a + b + c = 24 →
  d + e + f + g = 14 →
  (b = e ∨ a = e ∨ c = e) →
  a + b + c + d + f + g = 30 :=
by sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l172_17227


namespace NUMINAMATH_CALUDE_a_most_stable_l172_17292

/-- Represents a participant in the shooting test -/
inductive Participant
| A
| B
| C
| D

/-- Returns the variance of a participant's scores -/
def variance (p : Participant) : ℝ :=
  match p with
  | Participant.A => 0.54
  | Participant.B => 0.61
  | Participant.C => 0.7
  | Participant.D => 0.63

/-- Determines if a participant has the most stable performance -/
def has_most_stable_performance (p : Participant) : Prop :=
  ∀ q : Participant, variance p ≤ variance q

/-- Theorem: A has the most stable shooting performance -/
theorem a_most_stable : has_most_stable_performance Participant.A := by
  sorry

end NUMINAMATH_CALUDE_a_most_stable_l172_17292


namespace NUMINAMATH_CALUDE_olaf_sailing_speed_l172_17235

/-- Given the conditions of Olaf's sailing trip, prove the boat's daily travel distance. -/
theorem olaf_sailing_speed :
  -- Total distance to travel
  ∀ (total_distance : ℝ)
  -- Total number of men
  (total_men : ℕ)
  -- Water consumption per man per day (in gallons)
  (water_per_man_per_day : ℝ)
  -- Total water available (in gallons)
  (total_water : ℝ),
  total_distance = 4000 →
  total_men = 25 →
  water_per_man_per_day = 1/2 →
  total_water = 250 →
  -- The boat can travel this many miles per day
  (total_distance / (total_water / (total_men * water_per_man_per_day))) = 200 :=
by
  sorry


end NUMINAMATH_CALUDE_olaf_sailing_speed_l172_17235


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l172_17218

def f (x : ℝ) : ℝ := |x| - 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l172_17218


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l172_17208

/-- A circle tangent to coordinate axes and the hypotenuse of a 45-45-90 triangle --/
structure TangentCircle where
  O : ℝ × ℝ  -- Center of the circle
  r : ℝ      -- Radius of the circle
  h : r > 0  -- Radius is positive

/-- A 45-45-90 triangle with side length 2 --/
structure RightIsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  h1 : B.1 - A.1 = 2  -- AB has length 2
  h2 : C.2 - A.2 = 2  -- AC has length 2 in y-direction
  h3 : B.2 = A.2      -- AB is horizontal

/-- The main theorem --/
theorem tangent_circle_radius
  (t : TangentCircle)
  (tri : RightIsoscelesTriangle)
  (h_tangent_x : t.O.2 = t.r)
  (h_tangent_y : t.O.1 = t.r)
  (h_tangent_hyp : t.O.2 + t.r = tri.C.2) :
  t.r = 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l172_17208


namespace NUMINAMATH_CALUDE_side_length_of_octagon_l172_17217

theorem side_length_of_octagon (perimeter : ℝ) (num_sides : ℕ) (h1 : perimeter = 23.6) (h2 : num_sides = 8) :
  perimeter / num_sides = 2.95 := by
  sorry

end NUMINAMATH_CALUDE_side_length_of_octagon_l172_17217


namespace NUMINAMATH_CALUDE_spider_trade_l172_17296

/-- The number of spiders Pugsley and Wednesday trade --/
theorem spider_trade (P W x : ℕ) : 
  P = 4 →  -- Pugsley's initial number of spiders
  W + x = 9 * (P - x) →  -- First scenario equation
  P + 6 = W - 6 →  -- Second scenario equation
  x = 2  -- Number of spiders Pugsley gives to Wednesday
:= by sorry

end NUMINAMATH_CALUDE_spider_trade_l172_17296


namespace NUMINAMATH_CALUDE_football_games_per_month_l172_17254

theorem football_games_per_month 
  (total_games : ℕ) 
  (num_months : ℕ) 
  (h1 : total_games = 323) 
  (h2 : num_months = 17) 
  (h3 : total_games % num_months = 0) : 
  total_games / num_months = 19 := by
sorry

end NUMINAMATH_CALUDE_football_games_per_month_l172_17254


namespace NUMINAMATH_CALUDE_jet_flight_time_l172_17258

theorem jet_flight_time (distance : ℝ) (time_with_wind : ℝ) (wind_speed : ℝ) 
  (h1 : distance = 2000)
  (h2 : time_with_wind = 4)
  (h3 : wind_speed = 50)
  : ∃ (jet_speed : ℝ), 
    (jet_speed + wind_speed) * time_with_wind = distance ∧
    distance / (jet_speed - wind_speed) = 5 := by
  sorry

end NUMINAMATH_CALUDE_jet_flight_time_l172_17258


namespace NUMINAMATH_CALUDE_kitten_price_l172_17230

theorem kitten_price (kitten_count puppy_count : ℕ) 
                     (puppy_price total_earnings : ℚ) :
  kitten_count = 2 →
  puppy_count = 1 →
  puppy_price = 5 →
  total_earnings = 17 →
  ∃ kitten_price : ℚ, 
    kitten_price * kitten_count + puppy_price * puppy_count = total_earnings ∧
    kitten_price = 6 :=
by sorry

end NUMINAMATH_CALUDE_kitten_price_l172_17230


namespace NUMINAMATH_CALUDE_abs_m_minus_n_eq_two_sqrt_three_l172_17201

theorem abs_m_minus_n_eq_two_sqrt_three (m n : ℝ) 
  (h1 : m * n = 6) 
  (h2 : m + n = 6) : 
  |m - n| = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_abs_m_minus_n_eq_two_sqrt_three_l172_17201


namespace NUMINAMATH_CALUDE_integer_divisibility_l172_17283

theorem integer_divisibility (n : ℕ) (h : ∃ m : ℤ, (2^n - 2 : ℤ) = n * m) :
  ∃ k : ℤ, (2^(2^n - 1) - 2 : ℤ) = (2^n - 1) * k :=
sorry

end NUMINAMATH_CALUDE_integer_divisibility_l172_17283


namespace NUMINAMATH_CALUDE_triangle_probability_l172_17256

def stick_lengths : List ℕ := [3, 4, 6, 8, 10, 12, 15, 18]

def can_form_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def valid_triangle_combinations : List (ℕ × ℕ × ℕ) :=
  [(4, 6, 8), (6, 8, 10), (8, 10, 12), (10, 12, 15)]

def total_combinations : ℕ := Nat.choose 8 3

theorem triangle_probability :
  (List.length valid_triangle_combinations : ℚ) / total_combinations = 1 / 14 := by
  sorry

end NUMINAMATH_CALUDE_triangle_probability_l172_17256


namespace NUMINAMATH_CALUDE_min_value_sum_l172_17290

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 3 * x + y) :
  x + y ≥ 4 + 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_l172_17290


namespace NUMINAMATH_CALUDE_zeroes_of_f_range_of_a_l172_17268

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + b - 1

-- Theorem for the zeroes of f(x) when a = 1 and b = -2
theorem zeroes_of_f : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f 1 (-2) x₁ = 0 ∧ f 1 (-2) x₂ = 0 ∧ x₁ = 3 ∧ x₂ = -1 :=
sorry

-- Theorem for the range of a when f(x) always has two distinct zeroes
theorem range_of_a (a : ℝ) : 
  (a ≠ 0 ∧ ∀ b : ℝ, ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f a b x₁ = 0 ∧ f a b x₂ = 0) ↔ 0 < a ∧ a < 1 :=
sorry

end NUMINAMATH_CALUDE_zeroes_of_f_range_of_a_l172_17268


namespace NUMINAMATH_CALUDE_subtraction_of_negative_l172_17271

theorem subtraction_of_negative : 3 - (-3) = 6 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_l172_17271


namespace NUMINAMATH_CALUDE_intersection_is_open_interval_l172_17275

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 1}
def N : Set ℝ := {x | ∃ y : ℝ, y = Real.sqrt (3 - x^2)}

-- Define the complement of M relative to ℝ
def M_complement : Set ℝ := {y | y ∉ M}

-- Define the intersection of M_complement and N
def intersection : Set ℝ := M_complement ∩ N

-- Theorem statement
theorem intersection_is_open_interval :
  intersection = {x | -Real.sqrt 3 < x ∧ x < -1} :=
by sorry

end NUMINAMATH_CALUDE_intersection_is_open_interval_l172_17275


namespace NUMINAMATH_CALUDE_convex_pentagon_probability_l172_17205

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 5

/-- The total number of chords possible with n points -/
def total_chords : ℕ := n.choose 2

/-- The total number of ways to select k chords from total_chords -/
def total_selections : ℕ := total_chords.choose k

/-- The number of ways to select k points from n points -/
def favorable_outcomes : ℕ := n.choose k

/-- The probability of k randomly selected chords from n points on a circle forming a convex polygon -/
def probability : ℚ := favorable_outcomes / total_selections

theorem convex_pentagon_probability :
  n = 7 ∧ k = 5 → probability = 1 / 969 := by
  sorry

end NUMINAMATH_CALUDE_convex_pentagon_probability_l172_17205


namespace NUMINAMATH_CALUDE_systematic_sampling_l172_17263

/-- Represents a sampling method -/
inductive SamplingMethod
  | Lottery
  | RandomNumberTable
  | Systematic
  | Stratified

/-- Represents an auditorium -/
structure Auditorium where
  totalSeats : Nat
  seatsPerRow : Nat

/-- Represents a selection of students -/
structure Selection where
  seatNumber : Nat
  count : Nat

/-- Determines the sampling method based on the auditorium and selection -/
def determineSamplingMethod (a : Auditorium) (s : Selection) : SamplingMethod :=
  sorry

/-- Theorem: Selecting students with seat number 15 from the given auditorium is a systematic sampling method -/
theorem systematic_sampling (a : Auditorium) (s : Selection) :
  a.totalSeats = 25 →
  a.seatsPerRow = 20 →
  s.seatNumber = 15 →
  s.count = 25 →
  determineSamplingMethod a s = SamplingMethod.Systematic :=
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l172_17263


namespace NUMINAMATH_CALUDE_adjacent_triangles_toothpicks_l172_17294

/-- Calculates the number of toothpicks needed for an equilateral triangle -/
def toothpicks_for_triangle (base : ℕ) : ℕ :=
  3 * (base * (base + 1) / 2) / 2

/-- The number of toothpicks needed for two adjacent equilateral triangles -/
def total_toothpicks (large_base small_base : ℕ) : ℕ :=
  toothpicks_for_triangle large_base + toothpicks_for_triangle small_base - small_base

theorem adjacent_triangles_toothpicks :
  total_toothpicks 100 50 = 9462 :=
sorry

end NUMINAMATH_CALUDE_adjacent_triangles_toothpicks_l172_17294


namespace NUMINAMATH_CALUDE_quadratic_function_solution_set_l172_17257

/-- Given a quadratic function f(x) = ax^2 - (a+2)x - b, where a and b are real numbers,
    if the solution set of f(x) > 0 is (-3,2), then a + b = -7. -/
theorem quadratic_function_solution_set (a b : ℝ) :
  (∀ x, (a * x^2 - (a + 2) * x - b > 0) ↔ (-3 < x ∧ x < 2)) →
  a + b = -7 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_solution_set_l172_17257


namespace NUMINAMATH_CALUDE_cos_negative_52_thirds_pi_l172_17291

theorem cos_negative_52_thirds_pi : 
  Real.cos (-52 / 3 * Real.pi) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_52_thirds_pi_l172_17291


namespace NUMINAMATH_CALUDE_f_odd_and_increasing_l172_17298

-- Define the function f(x) = x|x|
def f (x : ℝ) : ℝ := x * abs x

-- Theorem stating that f is an odd function and an increasing function
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_increasing_l172_17298


namespace NUMINAMATH_CALUDE_beanie_tickets_l172_17276

def arcade_tickets (initial_tickets : ℕ) (additional_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  initial_tickets + additional_tickets - remaining_tickets

theorem beanie_tickets : arcade_tickets 11 10 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_beanie_tickets_l172_17276


namespace NUMINAMATH_CALUDE_initial_salty_cookies_count_l172_17272

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := sorry

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 3

/-- The number of salty cookies Paco had left after eating -/
def remaining_salty_cookies : ℕ := 3

/-- Theorem stating that the initial number of salty cookies is 6 -/
theorem initial_salty_cookies_count : initial_salty_cookies = 6 := by sorry

end NUMINAMATH_CALUDE_initial_salty_cookies_count_l172_17272


namespace NUMINAMATH_CALUDE_expression_simplification_l172_17248

theorem expression_simplification : 
  ((0.2 * 0.4 - (0.3 / 0.5)) + ((0.6 * 0.8 + (0.1 / 0.2)) - (0.9 * (0.3 - 0.2 * 0.4)))^2) * (1 - (0.4^2 / (0.2 * 0.8))) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l172_17248


namespace NUMINAMATH_CALUDE_button_fraction_proof_l172_17242

theorem button_fraction_proof (mari kendra sue : ℕ) : 
  mari = 5 * kendra + 4 →
  mari = 64 →
  sue = 6 →
  sue / kendra = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_button_fraction_proof_l172_17242


namespace NUMINAMATH_CALUDE_tangent_line_equation_l172_17253

theorem tangent_line_equation (x y : ℝ) : 
  y = 2 * x * Real.tan x →
  (2 + Real.pi / 2) * (Real.pi / 4) - (Real.pi / 2) - Real.pi^2 / 4 = 0 →
  (2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l172_17253


namespace NUMINAMATH_CALUDE_f_is_even_l172_17220

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Theorem statement
theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end NUMINAMATH_CALUDE_f_is_even_l172_17220


namespace NUMINAMATH_CALUDE_intersection_M_N_l172_17286

def M : Set ℕ := {1, 2, 4, 8}

def N : Set ℕ := {x : ℕ | x > 0 ∧ 4 % x = 0}

theorem intersection_M_N : M ∩ N = {1, 2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l172_17286


namespace NUMINAMATH_CALUDE_shells_added_calculation_l172_17246

/-- Calculates the amount of shells added given initial weight, percentage increase, and final weight -/
def shells_added (initial_weight : ℝ) (percent_increase : ℝ) (final_weight : ℝ) : ℝ :=
  final_weight - initial_weight

/-- Theorem stating that given the problem conditions, the amount of shells added is 23 pounds -/
theorem shells_added_calculation (initial_weight : ℝ) (percent_increase : ℝ) (final_weight : ℝ)
  (h1 : initial_weight = 5)
  (h2 : percent_increase = 150)
  (h3 : final_weight = 28) :
  shells_added initial_weight percent_increase final_weight = 23 := by
  sorry

#eval shells_added 5 150 28

end NUMINAMATH_CALUDE_shells_added_calculation_l172_17246


namespace NUMINAMATH_CALUDE_paco_cookies_l172_17229

/-- The number of sweet cookies Paco had initially -/
def initial_sweet_cookies : ℕ := 34

/-- The number of salty cookies Paco had initially -/
def initial_salty_cookies : ℕ := 97

/-- The number of sweet cookies Paco ate -/
def eaten_sweet_cookies : ℕ := 15

/-- The number of salty cookies Paco ate -/
def eaten_salty_cookies : ℕ := 56

/-- The number of sweet cookies Paco had left after eating -/
def remaining_sweet_cookies : ℕ := 19

theorem paco_cookies : initial_sweet_cookies = eaten_sweet_cookies + remaining_sweet_cookies :=
by sorry

end NUMINAMATH_CALUDE_paco_cookies_l172_17229


namespace NUMINAMATH_CALUDE_inequality_system_solution_l172_17277

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x > a ∧ x ≥ 3) ↔ x ≥ 3) → a < 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l172_17277


namespace NUMINAMATH_CALUDE_girls_fraction_at_joint_event_l172_17213

/-- Represents a middle school with a given number of students and boy-to-girl ratio --/
structure MiddleSchool where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a middle school --/
def girls_count (school : MiddleSchool) : ℚ :=
  (school.total_students : ℚ) * school.girl_ratio / (school.boy_ratio + school.girl_ratio)

/-- The fraction of girls at a joint event of two middle schools --/
def girls_fraction (school1 school2 : MiddleSchool) : ℚ :=
  (girls_count school1 + girls_count school2) / (school1.total_students + school2.total_students)

theorem girls_fraction_at_joint_event :
  let jasper_creek : MiddleSchool := { total_students := 360, boy_ratio := 7, girl_ratio := 5 }
  let brookstone : MiddleSchool := { total_students := 240, boy_ratio := 3, girl_ratio := 5 }
  girls_fraction jasper_creek brookstone = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_girls_fraction_at_joint_event_l172_17213


namespace NUMINAMATH_CALUDE_sequence_properties_l172_17251

def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => (a n)^2 - a n + 1

theorem sequence_properties :
  (∀ m n : ℕ, m ≠ n → Nat.gcd (a m) (a n) = 1) ∧
  (∑' k : ℕ, (1 : ℝ) / (a k)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l172_17251


namespace NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l172_17284

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The nth term of an arithmetic sequence. -/
def nthTerm (a : ℕ → ℝ) (n : ℕ) : ℝ := a n

theorem tenth_term_of_arithmetic_sequence
    (a : ℕ → ℝ)
    (h_arith : ArithmeticSequence a)
    (h_4th : nthTerm a 4 = 23)
    (h_6th : nthTerm a 6 = 43) :
  nthTerm a 10 = 83 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_arithmetic_sequence_l172_17284


namespace NUMINAMATH_CALUDE_onion_sale_earnings_is_66_l172_17226

/-- Calculates the money earned from selling onions given the initial quantities and conditions --/
def onion_sale_earnings (sally_onions fred_onions : ℕ) 
  (sara_plant_multiplier sara_harvest_multiplier : ℕ) 
  (onions_given_to_sara total_after_giving remaining_onions price_per_onion : ℕ) : ℕ :=
  let sara_planted := sara_plant_multiplier * sally_onions
  let sara_harvested := sara_harvest_multiplier * fred_onions
  let total_before_giving := sally_onions + fred_onions + sara_harvested
  let total_after_giving := total_before_giving - onions_given_to_sara
  let onions_sold := total_after_giving - remaining_onions
  onions_sold * price_per_onion

/-- Theorem stating that given the problem conditions, the earnings from selling onions is $66 --/
theorem onion_sale_earnings_is_66 : 
  onion_sale_earnings 5 9 3 2 4 24 6 3 = 66 := by
  sorry

end NUMINAMATH_CALUDE_onion_sale_earnings_is_66_l172_17226


namespace NUMINAMATH_CALUDE_cycle_price_proof_l172_17245

/-- Proves that given a cycle sold at a 20% loss for Rs. 1280, the original price was Rs. 1600 -/
theorem cycle_price_proof (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1280)
  (h2 : loss_percentage = 20) : 
  ∃ (original_price : ℝ), 
    original_price * (1 - loss_percentage / 100) = selling_price ∧ 
    original_price = 1600 :=
by sorry

end NUMINAMATH_CALUDE_cycle_price_proof_l172_17245


namespace NUMINAMATH_CALUDE_election_win_condition_l172_17249

theorem election_win_condition 
  (total_students : ℕ) 
  (boy_percentage : ℚ) 
  (girl_percentage : ℚ) 
  (male_vote_percentage : ℚ) 
  (h1 : total_students = 200)
  (h2 : boy_percentage = 3/5)
  (h3 : girl_percentage = 2/5)
  (h4 : boy_percentage + girl_percentage = 1)
  (h5 : male_vote_percentage = 27/40)
  : ∃ (female_vote_percentage : ℚ),
    female_vote_percentage ≥ 1/4 ∧
    (boy_percentage * male_vote_percentage + girl_percentage * female_vote_percentage) * total_students > total_students / 2 ∧
    ∀ (x : ℚ), x < female_vote_percentage →
      (boy_percentage * male_vote_percentage + girl_percentage * x) * total_students ≤ total_students / 2 :=
by sorry

end NUMINAMATH_CALUDE_election_win_condition_l172_17249


namespace NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l172_17203

theorem abs_two_minus_sqrt_five : 
  |2 - Real.sqrt 5| = Real.sqrt 5 - 2 := by sorry

end NUMINAMATH_CALUDE_abs_two_minus_sqrt_five_l172_17203


namespace NUMINAMATH_CALUDE_petya_wins_against_sasha_l172_17282

/-- Represents a player in the elimination tennis game -/
inductive Player : Type
| Petya : Player
| Sasha : Player
| Misha : Player

/-- The number of matches played by each player -/
def matches_played (p : Player) : ℕ :=
  match p with
  | Player.Petya => 12
  | Player.Sasha => 7
  | Player.Misha => 11

/-- The total number of matches played -/
def total_matches : ℕ := 15

/-- The number of wins by one player against another -/
def wins_against (winner loser : Player) : ℕ := sorry

theorem petya_wins_against_sasha :
  wins_against Player.Petya Player.Sasha = 4 :=
by sorry

end NUMINAMATH_CALUDE_petya_wins_against_sasha_l172_17282


namespace NUMINAMATH_CALUDE_hoseok_fruit_difference_l172_17274

/-- The number of lemons eaten minus the number of pears eaten by Hoseok -/
def lemon_pear_difference (apples pears tangerines lemons watermelons : ℕ) : ℤ :=
  lemons - pears

theorem hoseok_fruit_difference :
  lemon_pear_difference 8 5 12 17 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_fruit_difference_l172_17274


namespace NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l172_17259

theorem complex_exp_thirteen_pi_over_two (z : ℂ) : z = Complex.exp (13 * Real.pi * Complex.I / 2) → z = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_exp_thirteen_pi_over_two_l172_17259


namespace NUMINAMATH_CALUDE_car_cost_car_cost_proof_l172_17206

/-- The cost of Alex's car, given his savings and earnings from grocery deliveries -/
theorem car_cost (initial_savings : ℝ) (trip_charge : ℝ) (grocery_percentage : ℝ) 
  (num_trips : ℕ) (grocery_value : ℝ) : ℝ :=
  let earnings_from_trips := num_trips * trip_charge
  let earnings_from_groceries := grocery_percentage * grocery_value
  let total_earnings := earnings_from_trips + earnings_from_groceries
  let total_savings := initial_savings + total_earnings
  total_savings

/-- Proof that the car costs $14,600 -/
theorem car_cost_proof : 
  car_cost 14500 1.5 0.05 40 800 = 14600 := by
  sorry

end NUMINAMATH_CALUDE_car_cost_car_cost_proof_l172_17206


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_27_l172_17281

def polynomial (x : ℝ) : ℝ :=
  -3 * (x^7 - 2*x^6 + x^4 - 3*x^2 + 6) + 6 * (x^3 - 4*x + 1) - 2 * (x^5 - 5*x + 7)

theorem sum_of_coefficients_is_negative_27 : 
  (polynomial 1) = -27 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_27_l172_17281


namespace NUMINAMATH_CALUDE_radical_conjugate_sum_product_l172_17237

theorem radical_conjugate_sum_product (a b : ℝ) : 
  (a + Real.sqrt b) + (a - Real.sqrt b) = -6 ∧ 
  (a + Real.sqrt b) * (a - Real.sqrt b) = 1 → 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_radical_conjugate_sum_product_l172_17237


namespace NUMINAMATH_CALUDE_giraffe_height_difference_l172_17222

/-- The height of the tallest giraffe in inches -/
def tallest_giraffe : ℕ := 96

/-- The height of the shortest giraffe in inches -/
def shortest_giraffe : ℕ := 68

/-- The number of adult giraffes at the zoo -/
def num_giraffes : ℕ := 14

/-- The difference in height between the tallest and shortest giraffe -/
def height_difference : ℕ := tallest_giraffe - shortest_giraffe

theorem giraffe_height_difference :
  height_difference = 28 :=
sorry

end NUMINAMATH_CALUDE_giraffe_height_difference_l172_17222


namespace NUMINAMATH_CALUDE_z_in_third_quadrant_l172_17239

theorem z_in_third_quadrant (z : ℂ) (h : Complex.I * z = (4 + 3 * Complex.I) / (1 + 2 * Complex.I)) : 
  z.re < 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_z_in_third_quadrant_l172_17239


namespace NUMINAMATH_CALUDE_handshakes_count_l172_17267

/-- Represents the number of people in the gathering -/
def total_people : ℕ := 40

/-- Represents the number of people in Group A who all know each other -/
def group_a_size : ℕ := 25

/-- Represents the number of people in Group B -/
def group_b_size : ℕ := 15

/-- Represents the number of people in Group B who know exactly 3 people from Group A -/
def group_b_connected : ℕ := 5

/-- Represents the number of people in Group B who know no one -/
def group_b_isolated : ℕ := 10

/-- Represents the number of people each connected person in Group B knows in Group A -/
def connections_per_person : ℕ := 3

/-- Calculates the total number of handshakes in the gathering -/
def total_handshakes : ℕ := 
  (group_b_isolated * group_a_size) + 
  (group_b_connected * (group_a_size - connections_per_person)) + 
  (group_b_isolated.choose 2)

/-- Theorem stating that the total number of handshakes is 405 -/
theorem handshakes_count : total_handshakes = 405 := by
  sorry

end NUMINAMATH_CALUDE_handshakes_count_l172_17267


namespace NUMINAMATH_CALUDE_largest_power_l172_17244

theorem largest_power : 
  3^4000 > 2^5000 ∧ 
  3^4000 > 4^3000 ∧ 
  3^4000 > 5^2000 ∧ 
  3^4000 > 6^1000 := by sorry

end NUMINAMATH_CALUDE_largest_power_l172_17244


namespace NUMINAMATH_CALUDE_interest_rate_equation_l172_17243

/-- Given a principal that doubles in 10 years with quarterly compound interest,
    prove that the annual interest rate satisfies the equation 2 = (1 + r/4)^40 -/
theorem interest_rate_equation (r : ℝ) : 2 = (1 + r/4)^40 ↔ 
  ∀ (P : ℝ), P > 0 → 2*P = P * (1 + r/4)^40 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l172_17243


namespace NUMINAMATH_CALUDE_min_sum_of_powers_with_same_last_four_digits_l172_17269

theorem min_sum_of_powers_with_same_last_four_digits :
  ∀ m n : ℕ+,
    m ≠ n →
    (10000 : ℤ) ∣ (2019^(m.val) - 2019^(n.val)) →
    ∀ k l : ℕ+,
      k ≠ l →
      (10000 : ℤ) ∣ (2019^(k.val) - 2019^(l.val)) →
      m.val + n.val ≤ k.val + l.val →
      m.val + n.val = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_powers_with_same_last_four_digits_l172_17269


namespace NUMINAMATH_CALUDE_min_value_expression_l172_17210

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + 1/y) * (x + 1/y - 1024) + (y + 1/x) * (y + 1/x - 1024) ≥ -524288 ∧
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (a + 1/b) * (a + 1/b - 1024) + (b + 1/a) * (b + 1/a - 1024) = -524288 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l172_17210


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l172_17264

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 7 + a^(x - 1)
  f 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l172_17264


namespace NUMINAMATH_CALUDE_trigonometric_fraction_equality_l172_17228

open Real

theorem trigonometric_fraction_equality (x : ℝ) : 
  let f : ℝ → ℝ := λ x => sin x + cos x
  let f' : ℝ → ℝ := λ x => cos x - sin x
  (f x = 2 * f' x) → 
  (1 + sin x ^ 2) / (cos x ^ 2 - sin x * cos x) = 11 / 6 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_fraction_equality_l172_17228


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l172_17219

/-- The line equation is satisfied by the point (2, 3) for all values of k -/
theorem fixed_point_on_line (k : ℝ) : (2*k - 1) * 2 - (k - 2) * 3 - (k + 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l172_17219


namespace NUMINAMATH_CALUDE_jack_son_birth_time_l172_17202

def jack_lifetime : ℝ := 84

theorem jack_son_birth_time (adolescence : ℝ) (facial_hair : ℝ) (marriage : ℝ) (son_lifetime : ℝ) :
  adolescence = jack_lifetime / 6 →
  facial_hair = jack_lifetime / 6 + jack_lifetime / 12 →
  marriage = jack_lifetime / 6 + jack_lifetime / 12 + jack_lifetime / 7 →
  son_lifetime = jack_lifetime / 2 →
  jack_lifetime - (marriage + (jack_lifetime - son_lifetime - 4)) = 5 := by
sorry

end NUMINAMATH_CALUDE_jack_son_birth_time_l172_17202


namespace NUMINAMATH_CALUDE_line_reflection_x_axis_l172_17225

/-- Given a line with equation x - y + 1 = 0, its reflection with respect to the x-axis has the equation x + y + 1 = 0 -/
theorem line_reflection_x_axis :
  let original_line := {(x, y) : ℝ × ℝ | x - y + 1 = 0}
  let reflected_line := {(x, y) : ℝ × ℝ | x + y + 1 = 0}
  (∀ (x y : ℝ), (x, y) ∈ original_line ↔ (x, -y) ∈ reflected_line) :=
by sorry

end NUMINAMATH_CALUDE_line_reflection_x_axis_l172_17225


namespace NUMINAMATH_CALUDE_divisor_problem_l172_17204

theorem divisor_problem (n : ℕ) (h : n = 13294) : 
  ∃ (d : ℕ), d > 1 ∧ (n - 5) % d = 0 ∧ d = 13289 := by
  sorry

end NUMINAMATH_CALUDE_divisor_problem_l172_17204


namespace NUMINAMATH_CALUDE_potato_price_proof_l172_17221

/-- The original price of one bag of potatoes in rubles -/
def original_price : ℝ := 250

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 100

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 60

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 40

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey in rubles -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof :
  let andrey_earnings := bags_bought * original_price * (1 + andrey_increase / 100)
  let boris_first_earnings := boris_first_sale * original_price * (1 + boris_first_increase / 100)
  let boris_second_earnings := boris_second_sale * original_price * (1 + boris_first_increase / 100) * (1 + boris_second_increase / 100)
  boris_first_earnings + boris_second_earnings - andrey_earnings = earnings_difference :=
by sorry

end NUMINAMATH_CALUDE_potato_price_proof_l172_17221


namespace NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l172_17212

theorem smallest_cube_root_with_small_fraction (m n : ℕ) (r : ℝ) : 
  0 < n → 0 < r → r < 1 / 500 → 
  (↑m : ℝ) ^ (1/3 : ℝ) = n + r → 
  (∀ k < m, ¬∃ (s : ℝ), 0 < s ∧ s < 1/500 ∧ (↑k : ℝ) ^ (1/3 : ℝ) = ↑(n-1) + s) →
  n = 13 := by
sorry

end NUMINAMATH_CALUDE_smallest_cube_root_with_small_fraction_l172_17212


namespace NUMINAMATH_CALUDE_rate_squares_sum_l172_17224

theorem rate_squares_sum : ∀ (b j s : ℕ),
  (3 * b + 4 * j + 2 * s = 86) →
  (5 * b + 2 * j + 4 * s = 110) →
  (b * b + j * j + s * s = 3349) :=
by sorry

end NUMINAMATH_CALUDE_rate_squares_sum_l172_17224


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l172_17231

/-- Given a function f(x) = x + 1/x - 2 and a real number a such that f(a) = 3,
    prove that f(-a) = -7. -/
theorem function_value_at_negative_a 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x + 1/x - 2) 
  (h2 : f a = 3) : 
  f (-a) = -7 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l172_17231
