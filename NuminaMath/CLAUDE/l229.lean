import Mathlib

namespace NUMINAMATH_CALUDE_dennis_floors_above_charlie_l229_22981

/-- The floor number on which Frank lives -/
def frank_floor : ℕ := 16

/-- The floor number on which Charlie lives -/
def charlie_floor : ℕ := frank_floor / 4

/-- The floor number on which Dennis lives -/
def dennis_floor : ℕ := 6

/-- The number of floors Dennis lives above Charlie -/
def floors_above : ℕ := dennis_floor - charlie_floor

theorem dennis_floors_above_charlie : floors_above = 2 := by
  sorry

end NUMINAMATH_CALUDE_dennis_floors_above_charlie_l229_22981


namespace NUMINAMATH_CALUDE_cube_root_64_minus_sqrt_8_squared_l229_22929

theorem cube_root_64_minus_sqrt_8_squared : 
  (64 ^ (1/3) - Real.sqrt 8) ^ 2 = 24 - 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_64_minus_sqrt_8_squared_l229_22929


namespace NUMINAMATH_CALUDE_min_value_a_l229_22901

theorem min_value_a : 
  (∃ (a : ℝ), ∀ (x₁ x₂ x₃ x₄ : ℝ), 
    ∃ (k₁ k₂ k₃ k₄ : ℤ), 
      (x₂ - k₁ - (x₁ - k₂))^2 + 
      (x₃ - k₁ - (x₁ - k₃))^2 + 
      (x₄ - k₁ - (x₁ - k₄))^2 + 
      (x₃ - k₂ - (x₂ - k₃))^2 + 
      (x₄ - k₂ - (x₂ - k₄))^2 + 
      (x₄ - k₃ - (x₃ - k₄))^2 ≤ a) ∧ 
  (∀ (b : ℝ), (∀ (x₁ x₂ x₃ x₄ : ℝ), 
    ∃ (k₁ k₂ k₃ k₄ : ℤ), 
      (x₂ - k₁ - (x₁ - k₂))^2 + 
      (x₃ - k₁ - (x₁ - k₃))^2 + 
      (x₄ - k₁ - (x₁ - k₄))^2 + 
      (x₃ - k₂ - (x₂ - k₃))^2 + 
      (x₄ - k₂ - (x₂ - k₄))^2 + 
      (x₄ - k₃ - (x₃ - k₄))^2 ≤ b) → b ≥ 5/4) := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l229_22901


namespace NUMINAMATH_CALUDE_school_dinner_drink_choice_l229_22926

theorem school_dinner_drink_choice (total_students : ℕ) 
  (juice_percentage : ℚ) (water_percentage : ℚ) (juice_students : ℕ) :
  juice_percentage = 3/4 →
  water_percentage = 1/4 →
  juice_students = 90 →
  ∃ water_students : ℕ, water_students = 30 ∧ 
    (juice_students : ℚ) / total_students = juice_percentage ∧
    (water_students : ℚ) / total_students = water_percentage :=
by sorry

end NUMINAMATH_CALUDE_school_dinner_drink_choice_l229_22926


namespace NUMINAMATH_CALUDE_pigeonhole_apples_l229_22970

theorem pigeonhole_apples (n : ℕ) (m : ℕ) (h1 : n = 25) (h2 : m = 3) :
  ∃ (c : Fin m), (n / m : ℚ) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_apples_l229_22970


namespace NUMINAMATH_CALUDE_system_solution_proof_l229_22960

theorem system_solution_proof (x y : ℝ) : 
  (2 * x + y = 2 ∧ x - y = 1) → (x = 1 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_proof_l229_22960


namespace NUMINAMATH_CALUDE_invested_sum_is_700_l229_22987

/-- Represents the simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the invested sum is $700 given the problem conditions --/
theorem invested_sum_is_700 
  (peter_amount : ℝ) 
  (david_amount : ℝ) 
  (peter_time : ℝ) 
  (david_time : ℝ) 
  (h1 : peter_amount = 815)
  (h2 : david_amount = 850)
  (h3 : peter_time = 3)
  (h4 : david_time = 4)
  : ∃ (principal rate : ℝ),
    simple_interest principal rate peter_time = peter_amount ∧
    simple_interest principal rate david_time = david_amount ∧
    principal = 700 := by
  sorry

end NUMINAMATH_CALUDE_invested_sum_is_700_l229_22987


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l229_22968

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  S : ℕ+ → ℚ
  is_arithmetic : ∀ n : ℕ+, a (n + 1) - a n = a 2 - a 1
  sum_def : ∀ n : ℕ+, S n = (n : ℚ) * (a 1 + a n) / 2

/-- The main theorem -/
theorem arithmetic_sequence_sum_10
  (seq : ArithmeticSequence)
  (h1 : seq.a 3 = 16)
  (h2 : seq.S 20 = 20) :
  seq.S 10 = 110 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_10_l229_22968


namespace NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_digit_product_ratio_l229_22944

/-- Given a natural number, return the product of its non-zero digits -/
def productOfNonZeroDigits (n : ℕ) : ℕ := sorry

/-- Theorem stating that there exist two consecutive natural numbers
    such that the product of all non-zero digits of the larger number
    multiplied by 54 equals the product of all non-zero digits of the smaller number -/
theorem exists_consecutive_numbers_with_54_digit_product_ratio :
  ∃ n : ℕ, productOfNonZeroDigits n = 54 * productOfNonZeroDigits (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_consecutive_numbers_with_54_digit_product_ratio_l229_22944


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_in_arithmetic_sequence_l229_22907

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The given sequence 3, 9, x, y, 27 -/
def givenSequence (x y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9
  | 2 => x
  | 3 => y
  | 4 => 27
  | _ => 0  -- For indices beyond 4, we return 0 (this part is not relevant to our problem)

theorem sum_of_x_and_y_in_arithmetic_sequence (x y : ℝ) 
    (h : isArithmeticSequence (givenSequence x y)) : x + y = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_in_arithmetic_sequence_l229_22907


namespace NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l229_22950

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (a = 3) ∧ 
    (∀ (x : ℂ), a * x^2 + b * x + c = 0 ↔ x = 4 + I ∨ x = 4 - I) ∧
    (a * (4 + I)^2 + b * (4 + I) + c = 0) ∧
    (a = 3 ∧ b = -24 ∧ c = 51) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_with_complex_root_l229_22950


namespace NUMINAMATH_CALUDE_largest_number_is_541_l229_22962

def digits : List Nat := [1, 4, 5]

def is_valid_number (n : Nat) : Prop :=
  let digits_of_n := n.digits 10
  digits_of_n.length = 3 ∧ digits_of_n.toFinset = digits.toFinset

theorem largest_number_is_541 :
  ∀ n : Nat, is_valid_number n → n ≤ 541 :=
sorry

end NUMINAMATH_CALUDE_largest_number_is_541_l229_22962


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l229_22975

/-- Represents the dimensions of Carrie's garden -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Represents the planting and yield information -/
structure PlantingInfo where
  plantsPerSquareFoot : ℕ
  strawberriesPerPlant : ℕ

/-- Calculates the expected strawberry harvest given garden dimensions and planting information -/
def expectedHarvest (garden : GardenDimensions) (info : PlantingInfo) : ℕ :=
  garden.length * garden.width * info.plantsPerSquareFoot * info.strawberriesPerPlant

/-- Theorem stating that Carrie's expected strawberry harvest is 3150 -/
theorem carries_strawberry_harvest :
  let garden := GardenDimensions.mk 7 9
  let info := PlantingInfo.mk 5 10
  expectedHarvest garden info = 3150 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l229_22975


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l229_22993

theorem min_value_quadratic (x : ℝ) : 
  7 * x^2 - 28 * x + 1702 ≥ 1674 := by
sorry

theorem min_value_quadratic_achieved : 
  ∃ x : ℝ, 7 * x^2 - 28 * x + 1702 = 1674 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l229_22993


namespace NUMINAMATH_CALUDE_coffee_shop_usage_l229_22906

/-- The number of bags of coffee beans used every morning -/
def morning_bags : ℕ := 3

/-- The number of bags of coffee beans used every afternoon -/
def afternoon_bags : ℕ := 3 * morning_bags

/-- The number of bags of coffee beans used every evening -/
def evening_bags : ℕ := 2 * morning_bags

/-- The total number of bags used in a week -/
def weekly_bags : ℕ := 126

theorem coffee_shop_usage :
  7 * (morning_bags + afternoon_bags + evening_bags) = weekly_bags :=
sorry

end NUMINAMATH_CALUDE_coffee_shop_usage_l229_22906


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l229_22991

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 2; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 16384; 0, -8192] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l229_22991


namespace NUMINAMATH_CALUDE_alex_bike_trip_downhill_speed_alex_bike_trip_downhill_speed_is_24_l229_22916

/-- Calculates the average speed on the downhill section of Alex's bike trip --/
theorem alex_bike_trip_downhill_speed : ℝ :=
  let total_distance : ℝ := 164
  let flat_time : ℝ := 4.5
  let flat_speed : ℝ := 20
  let uphill_time : ℝ := 2.5
  let uphill_speed : ℝ := 12
  let downhill_time : ℝ := 1.5
  let walking_distance : ℝ := 8
  let flat_distance : ℝ := flat_time * flat_speed
  let uphill_distance : ℝ := uphill_time * uphill_speed
  let distance_before_puncture : ℝ := total_distance - walking_distance
  let downhill_distance : ℝ := distance_before_puncture - flat_distance - uphill_distance
  let downhill_speed : ℝ := downhill_distance / downhill_time
  downhill_speed

theorem alex_bike_trip_downhill_speed_is_24 : alex_bike_trip_downhill_speed = 24 := by
  sorry

end NUMINAMATH_CALUDE_alex_bike_trip_downhill_speed_alex_bike_trip_downhill_speed_is_24_l229_22916


namespace NUMINAMATH_CALUDE_tables_needed_l229_22967

theorem tables_needed (total_children : ℕ) (children_per_table : ℕ) (tables : ℕ) : 
  total_children = 152 → 
  children_per_table = 7 → 
  tables = 22 → 
  tables = (total_children + children_per_table - 1) / children_per_table :=
by sorry

end NUMINAMATH_CALUDE_tables_needed_l229_22967


namespace NUMINAMATH_CALUDE_smaller_number_problem_l229_22985

theorem smaller_number_problem (a b : ℝ) : 
  a + b = 18 → a * b = 45 → min a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l229_22985


namespace NUMINAMATH_CALUDE_special_sequence_coprime_l229_22917

/-- A polynomial with integer coefficients that maps 0 and 1 to 1 -/
def SpecialPolynomial (p : ℤ → ℤ) : Prop :=
  (∀ x y : ℤ, p (x + y) = p x + p y - 1) ∧ p 0 = 1 ∧ p 1 = 1

/-- The sequence defined by the special polynomial -/
def SpecialSequence (p : ℤ → ℤ) (a : ℕ → ℤ) : Prop :=
  a 0 ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = p (a n)

/-- The theorem stating that any two terms in the sequence are coprime -/
theorem special_sequence_coprime (p : ℤ → ℤ) (a : ℕ → ℤ) 
  (hp : SpecialPolynomial p) (ha : SpecialSequence p a) :
  ∀ i j : ℕ, Nat.gcd (a i).natAbs (a j).natAbs = 1 :=
sorry

end NUMINAMATH_CALUDE_special_sequence_coprime_l229_22917


namespace NUMINAMATH_CALUDE_inverse_square_relation_l229_22945

/-- Given that x varies inversely as the square of y, and y = 3 when x = 1,
    prove that x = 0.5625 when y = 4. -/
theorem inverse_square_relation (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y ^ 2)) →  -- x varies inversely as the square of y
  (1 = k / (3 ^ 2)) →               -- y = 3 when x = 1
  (k = 9) →                         -- derived from the previous condition
  (x = 9 / (4 ^ 2)) →               -- x when y = 4
  x = 0.5625 := by
sorry

end NUMINAMATH_CALUDE_inverse_square_relation_l229_22945


namespace NUMINAMATH_CALUDE_dice_product_composite_probability_l229_22964

def num_dice : ℕ := 6
def num_sides : ℕ := 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(is_prime n)

def total_outcomes : ℕ := num_sides ^ num_dice

def non_composite_outcomes : ℕ := 25

theorem dice_product_composite_probability :
  (total_outcomes - non_composite_outcomes) / total_outcomes = 262119 / 262144 :=
sorry

end NUMINAMATH_CALUDE_dice_product_composite_probability_l229_22964


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l229_22992

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.a = t.c

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

theorem similar_triangle_perimeter 
  (t1 : Triangle) 
  (h1 : t1.isIsosceles)
  (h2 : t1.a = 16 ∧ t1.b = 16 ∧ t1.c = 8)
  (t2 : Triangle)
  (h3 : Triangle.isSimilar t1 t2)
  (h4 : min t2.a (min t2.b t2.c) = 40) :
  t2.perimeter = 200 := by
  sorry


end NUMINAMATH_CALUDE_similar_triangle_perimeter_l229_22992


namespace NUMINAMATH_CALUDE_dime_difference_l229_22940

/-- Represents the types of coins in the piggy bank -/
inductive Coin
  | Nickel
  | Dime
  | Quarter

/-- Represents the piggy bank with its coin composition -/
structure PiggyBank where
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ
  total_coins : nickels + dimes + quarters = 100
  total_value : 5 * nickels + 10 * dimes + 25 * quarters = 1005

/-- The value of a given coin in cents -/
def coinValue : Coin → ℕ
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Theorem stating the difference between max and min number of dimes -/
theorem dime_difference (pb : PiggyBank) : 
  ∃ (min_dimes max_dimes : ℕ), 
    (∀ pb' : PiggyBank, pb'.dimes ≥ min_dimes) ∧ 
    (∃ pb' : PiggyBank, pb'.dimes = min_dimes) ∧
    (∀ pb' : PiggyBank, pb'.dimes ≤ max_dimes) ∧ 
    (∃ pb' : PiggyBank, pb'.dimes = max_dimes) ∧
    max_dimes - min_dimes = 100 :=
  sorry


end NUMINAMATH_CALUDE_dime_difference_l229_22940


namespace NUMINAMATH_CALUDE_intersection_line_equation_l229_22989

/-- Given two lines l₁ and l₂ in the plane, and a line l passing through their
    intersection point and the origin, prove that l has the equation x - 10y = 0. -/
theorem intersection_line_equation :
  let l₁ : ℝ × ℝ → Prop := λ p => 2 * p.1 + p.2 = 3
  let l₂ : ℝ × ℝ → Prop := λ p => p.1 + 4 * p.2 = 2
  let P : ℝ × ℝ := (10/7, 1/7)  -- Intersection point of l₁ and l₂
  let l : ℝ × ℝ → Prop := λ p => p.1 - 10 * p.2 = 0
  (l₁ P ∧ l₂ P) →  -- P is the intersection of l₁ and l₂
  (l (0, 0)) →     -- l passes through the origin
  (l P) →          -- l passes through P
  ∀ p : ℝ × ℝ, (l₁ p ∧ l₂ p) → l p  -- For any point on both l₁ and l₂, it's also on l
  :=
by sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l229_22989


namespace NUMINAMATH_CALUDE_triangle_perimeter_l229_22994

theorem triangle_perimeter (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  b / a = 4 / 3 →          -- ratio of second to first side is 4:3
  c / a = 5 / 3 →          -- ratio of third to first side is 5:3
  c - a = 6 →              -- difference between longest and shortest side is 6
  a + b + c = 36 :=        -- perimeter is 36
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l229_22994


namespace NUMINAMATH_CALUDE_triangle_problem_geometric_sequence_problem_l229_22938

-- Triangle problem
theorem triangle_problem (a b : ℝ) (B : ℝ) 
  (ha : a = Real.sqrt 3) 
  (hb : b = Real.sqrt 2) 
  (hB : B = 45 * π / 180) :
  (∃ (A C c : ℝ),
    (A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
    (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) :=
by sorry

-- Geometric sequence problem
theorem geometric_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (hS2 : S 2 = 7)
  (hS6 : S 6 = 91)
  (h_geom : ∀ n, S (n+1) - S n = (S 2 - S 1) * (S 2 / S 1) ^ (n-1)) :
  S 4 = 28 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_geometric_sequence_problem_l229_22938


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l229_22919

theorem sarahs_bowling_score (sarah_score greg_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 110 →
  sarah_score + greg_score < 450 →
  sarah_score = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l229_22919


namespace NUMINAMATH_CALUDE_first_number_value_l229_22979

theorem first_number_value (x : ℝ) : x + 2 * (8 - 3) = 24.16 → x = 14.16 := by
  sorry

end NUMINAMATH_CALUDE_first_number_value_l229_22979


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l229_22952

theorem floor_negative_seven_fourths :
  ⌊(-7 : ℝ) / 4⌋ = -2 := by sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l229_22952


namespace NUMINAMATH_CALUDE_fraction_equality_l229_22980

theorem fraction_equality (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 4) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = 1 := by sorry

end NUMINAMATH_CALUDE_fraction_equality_l229_22980


namespace NUMINAMATH_CALUDE_range_of_a_l229_22923

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log (x / 2) - (3 * x - 6) / (x + 1)

noncomputable def g (x t a : ℝ) : ℝ := (x - t)^2 + (Real.log x - a * t)^2

theorem range_of_a :
  ∀ a : ℝ,
  (∀ x₁ : ℝ, x₁ > 1 → ∃ t x₂ : ℝ, x₂ > 0 ∧ f x₁ ≥ g x₂ t a) ↔
  a ≤ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l229_22923


namespace NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l229_22937

theorem not_p_and_q_implies_at_most_one_true (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) :=
by sorry

end NUMINAMATH_CALUDE_not_p_and_q_implies_at_most_one_true_l229_22937


namespace NUMINAMATH_CALUDE_stock_price_decrease_l229_22905

theorem stock_price_decrease (P : ℝ) (X : ℝ) : 
  P > 0 →
  1.20 * P * (1 - X) * 1.35 = 1.215 * P →
  X = 0.25 := by
sorry

end NUMINAMATH_CALUDE_stock_price_decrease_l229_22905


namespace NUMINAMATH_CALUDE_popcorn_servings_for_jared_and_friends_l229_22925

/-- Calculate the number of popcorn servings needed for a group -/
def popcorn_servings (pieces_per_serving : ℕ) (jared_pieces : ℕ) (num_friends : ℕ) (friend_pieces : ℕ) : ℕ :=
  ((jared_pieces + num_friends * friend_pieces) + pieces_per_serving - 1) / pieces_per_serving

theorem popcorn_servings_for_jared_and_friends :
  popcorn_servings 30 90 3 60 = 9 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_servings_for_jared_and_friends_l229_22925


namespace NUMINAMATH_CALUDE_inequality_proof_l229_22977

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0) 
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) : 
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l229_22977


namespace NUMINAMATH_CALUDE_first_customer_boxes_l229_22920

def cookie_problem (x : ℚ) : Prop :=
  let second_customer := 4 * x
  let third_customer := second_customer / 2
  let fourth_customer := 3 * third_customer
  let final_customer := 10
  let total_sold := x + second_customer + third_customer + fourth_customer + final_customer
  let goal := 150
  let left_to_sell := 75
  total_sold + left_to_sell = goal

theorem first_customer_boxes : ∃ x : ℚ, cookie_problem x ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_first_customer_boxes_l229_22920


namespace NUMINAMATH_CALUDE_smallest_integer_value_is_two_l229_22997

/-- Represents a digit assignment for the kangaroo/game expression -/
structure DigitAssignment where
  k : Nat
  a : Nat
  n : Nat
  g : Nat
  r : Nat
  o : Nat
  m : Nat
  e : Nat
  k_nonzero : k ≠ 0
  a_nonzero : a ≠ 0
  n_nonzero : n ≠ 0
  g_nonzero : g ≠ 0
  r_nonzero : r ≠ 0
  o_nonzero : o ≠ 0
  m_nonzero : m ≠ 0
  e_nonzero : e ≠ 0
  all_different : k ≠ a ∧ k ≠ n ∧ k ≠ g ∧ k ≠ r ∧ k ≠ o ∧ k ≠ m ∧ k ≠ e ∧
                  a ≠ n ∧ a ≠ g ∧ a ≠ r ∧ a ≠ o ∧ a ≠ m ∧ a ≠ e ∧
                  n ≠ g ∧ n ≠ r ∧ n ≠ o ∧ n ≠ m ∧ n ≠ e ∧
                  g ≠ r ∧ g ≠ o ∧ g ≠ m ∧ g ≠ e ∧
                  r ≠ o ∧ r ≠ m ∧ r ≠ e ∧
                  o ≠ m ∧ o ≠ e ∧
                  m ≠ e
  all_digits : k < 10 ∧ a < 10 ∧ n < 10 ∧ g < 10 ∧ r < 10 ∧ o < 10 ∧ m < 10 ∧ e < 10

/-- Calculates the value of the kangaroo/game expression for a given digit assignment -/
def expressionValue (d : DigitAssignment) : Rat :=
  (d.k * d.a * d.n * d.g * d.a * d.r * d.o * d.o) / (d.g * d.a * d.m * d.e)

/-- States that the smallest integer value of the kangaroo/game expression is 2 -/
theorem smallest_integer_value_is_two :
  ∃ (d : DigitAssignment), expressionValue d = 2 ∧
  ∀ (d' : DigitAssignment), (expressionValue d').isInt → expressionValue d' ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_value_is_two_l229_22997


namespace NUMINAMATH_CALUDE_x_value_proof_l229_22976

theorem x_value_proof (x : ℝ) 
  (h : (x^2 - x - 6) / (x + 1) = (x^2 - 2*x - 3)*Complex.I) : 
  x = 3 := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l229_22976


namespace NUMINAMATH_CALUDE_logarithmic_equation_proof_l229_22947

theorem logarithmic_equation_proof : 2 * (Real.log 10 / Real.log 5) + Real.log (1/4) / Real.log 5 + 2^(Real.log 3 / Real.log 4) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_logarithmic_equation_proof_l229_22947


namespace NUMINAMATH_CALUDE_celine_library_charge_l229_22966

/-- Represents the library's charge policy and Celine's borrowing details -/
structure LibraryCharge where
  daily_rate : ℚ
  books_borrowed : ℕ
  days_for_first_book : ℕ
  days_in_may : ℕ

/-- Calculates the total charge for Celine's borrowed books -/
def calculate_total_charge (lc : LibraryCharge) : ℚ :=
  lc.daily_rate * lc.days_for_first_book +
  lc.daily_rate * lc.days_in_may * 2

/-- Theorem stating that Celine's total charge is $41.00 -/
theorem celine_library_charge :
  let lc : LibraryCharge := {
    daily_rate := 1/2,
    books_borrowed := 3,
    days_for_first_book := 20,
    days_in_may := 31
  }
  calculate_total_charge lc = 41
  := by sorry

end NUMINAMATH_CALUDE_celine_library_charge_l229_22966


namespace NUMINAMATH_CALUDE_train_average_speed_l229_22959

/-- Calculates the average speed of a train journey with a stop -/
theorem train_average_speed 
  (distance1 : ℝ) 
  (time1 : ℝ) 
  (stop_time : ℝ) 
  (distance2 : ℝ) 
  (time2 : ℝ) 
  (h1 : distance1 = 240) 
  (h2 : time1 = 3) 
  (h3 : stop_time = 0.5) 
  (h4 : distance2 = 450) 
  (h5 : time2 = 5) :
  (distance1 + distance2) / (time1 + stop_time + time2) = (240 + 450) / (3 + 0.5 + 5) :=
by sorry

end NUMINAMATH_CALUDE_train_average_speed_l229_22959


namespace NUMINAMATH_CALUDE_most_cars_are_blue_l229_22910

theorem most_cars_are_blue (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
  sorry

end NUMINAMATH_CALUDE_most_cars_are_blue_l229_22910


namespace NUMINAMATH_CALUDE_tom_age_l229_22946

theorem tom_age (carla dave emily tom : ℕ) : 
  tom = 2 * carla - 1 →
  dave = carla + 3 →
  emily = carla / 2 →
  carla + dave + emily + tom = 48 →
  tom = 19 := by
  sorry

end NUMINAMATH_CALUDE_tom_age_l229_22946


namespace NUMINAMATH_CALUDE_commute_speed_theorem_l229_22941

theorem commute_speed_theorem (d : ℝ) (t : ℝ) (h1 : d = 50 * (t + 1/15)) (h2 : d = 70 * (t - 1/15)) :
  d / t = 58 := by sorry

end NUMINAMATH_CALUDE_commute_speed_theorem_l229_22941


namespace NUMINAMATH_CALUDE_series_sum_equals_half_l229_22935

noncomputable def series_sum (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

theorem series_sum_equals_half :
  ∑' n, series_sum n = 1/2 := by sorry

end NUMINAMATH_CALUDE_series_sum_equals_half_l229_22935


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l229_22978

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₅ = 3 and a₉ = 6,
    prove that a₁₃ = 9 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ)
    (h_arith : is_arithmetic_sequence a)
    (h_a5 : a 5 = 3)
    (h_a9 : a 9 = 6) :
  a 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l229_22978


namespace NUMINAMATH_CALUDE_distance_to_outside_point_gt_three_l229_22953

/-- A circle with center O and radius 3 -/
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)
  (h_radius : radius = 3)

/-- A point outside the circle -/
structure OutsidePoint (c : Circle) :=
  (point : ℝ × ℝ)
  (h_outside : dist point c.center > c.radius)

/-- The theorem stating that the distance from the center to an outside point is greater than 3 -/
theorem distance_to_outside_point_gt_three (c : Circle) (p : OutsidePoint c) :
  dist p.point c.center > 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_outside_point_gt_three_l229_22953


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l229_22957

theorem square_area_with_four_circles (r : ℝ) (h : r = 3) :
  let circle_diameter := 2 * r
  let square_side := 2 * circle_diameter
  square_side ^ 2 = 144 := by sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l229_22957


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l229_22999

/-- Given a quadratic expression of the form ak² + bk + d, 
    rewrite it as c(k + p)² + q and return (c, p, q) -/
def rewrite_quadratic (a b d : ℚ) : ℚ × ℚ × ℚ := sorry

theorem quadratic_rewrite_ratio : 
  let (c, p, q) := rewrite_quadratic 8 (-12) 20
  q / p = -62 / 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l229_22999


namespace NUMINAMATH_CALUDE_lcm_gcd_product_40_100_l229_22928

theorem lcm_gcd_product_40_100 : Nat.lcm 40 100 * Nat.gcd 40 100 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_40_100_l229_22928


namespace NUMINAMATH_CALUDE_gcd_of_225_and_135_l229_22969

theorem gcd_of_225_and_135 : Nat.gcd 225 135 = 45 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_225_and_135_l229_22969


namespace NUMINAMATH_CALUDE_intersection_with_complement_l229_22912

def U : Set Nat := {2, 3, 4, 5, 6}
def A : Set Nat := {2, 3, 4}
def B : Set Nat := {2, 3, 5}

theorem intersection_with_complement : A ∩ (U \ B) = {4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l229_22912


namespace NUMINAMATH_CALUDE_triangle_angle_B_l229_22936

theorem triangle_angle_B (A B C : Real) (a b c : Real) : 
  A = 2 * Real.pi / 3 →  -- 120° in radians
  a = 2 →
  b = 2 * Real.sqrt 3 / 3 →
  B = Real.pi / 6  -- 30° in radians
:= by sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l229_22936


namespace NUMINAMATH_CALUDE_first_dog_bones_l229_22955

theorem first_dog_bones (total_bones : ℕ) (total_dogs : ℕ) 
  (h_total_bones : total_bones = 12)
  (h_total_dogs : total_dogs = 5)
  (first_dog : ℕ)
  (second_dog : ℕ)
  (third_dog : ℕ)
  (fourth_dog : ℕ)
  (fifth_dog : ℕ)
  (h_second_dog : second_dog = first_dog - 1)
  (h_third_dog : third_dog = 2 * second_dog)
  (h_fourth_dog : fourth_dog = 1)
  (h_fifth_dog : fifth_dog = 2 * fourth_dog)
  (h_all_bones : first_dog + second_dog + third_dog + fourth_dog + fifth_dog = total_bones) :
  first_dog = 3 := by
sorry

end NUMINAMATH_CALUDE_first_dog_bones_l229_22955


namespace NUMINAMATH_CALUDE_shekar_weighted_average_l229_22973

def weightedAverage (scores : List ℝ) (weights : List ℝ) : ℝ :=
  (List.zip scores weights).map (fun (s, w) => s * w) |> List.sum

theorem shekar_weighted_average :
  let scores : List ℝ := [76, 65, 82, 62, 85]
  let weights : List ℝ := [0.20, 0.15, 0.25, 0.25, 0.15]
  weightedAverage scores weights = 73.7 := by
sorry

end NUMINAMATH_CALUDE_shekar_weighted_average_l229_22973


namespace NUMINAMATH_CALUDE_investment_difference_theorem_l229_22972

/-- Calculates the difference in total amounts between two investment schemes after one year -/
def investment_difference (initial_a : ℝ) (initial_b : ℝ) (yield_a : ℝ) (yield_b : ℝ) : ℝ :=
  (initial_a * (1 + yield_a)) - (initial_b * (1 + yield_b))

/-- Theorem stating the difference in total amounts between schemes A and B after one year -/
theorem investment_difference_theorem :
  investment_difference 300 200 0.3 0.5 = 90 := by
  sorry

end NUMINAMATH_CALUDE_investment_difference_theorem_l229_22972


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l229_22909

theorem sqrt_fourth_power_equals_256 (y : ℝ) : (Real.sqrt y)^4 = 256 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_equals_256_l229_22909


namespace NUMINAMATH_CALUDE_geometry_biology_overlap_l229_22930

theorem geometry_biology_overlap (total : ℕ) (geometry : ℕ) (biology : ℕ) 
  (h1 : total = 350) (h2 : geometry = 210) (h3 : biology = 175) :
  let max_overlap := min geometry biology
  let min_overlap := max 0 (geometry + biology - total)
  max_overlap - min_overlap = 140 := by
sorry

end NUMINAMATH_CALUDE_geometry_biology_overlap_l229_22930


namespace NUMINAMATH_CALUDE_value_of_x_l229_22902

theorem value_of_x (x y z : ℚ) : x = y / 3 → y = z / 4 → z = 48 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l229_22902


namespace NUMINAMATH_CALUDE_pentagon_extension_theorem_l229_22984

/-- Given a pentagon ABCDE with extended sides, prove the relation between A and A', B', C', D', E' -/
theorem pentagon_extension_theorem (A B C D E A' B' C' D' E' : ℝ × ℝ) : 
  (A'B = AB) → (B'C = BC) → (C'D = CD) → (D'E = DE) → (E'A = EA) →
  ∃ (p q r s t : ℝ), 
    (A : ℝ × ℝ) = p • A' + q • B' + r • C' + s • D' + t • E' ∧ 
    p = (1 : ℝ) / 31 ∧ q = 2 / 31 ∧ r = 4 / 31 ∧ s = 8 / 31 ∧ t = 16 / 31 :=
by sorry


end NUMINAMATH_CALUDE_pentagon_extension_theorem_l229_22984


namespace NUMINAMATH_CALUDE_polynomial_roots_l229_22922

theorem polynomial_roots : 
  let p (x : ℝ) := x^4 - 3*x^3 + 3*x^2 - x - 6
  ∀ x : ℝ, p x = 0 ↔ x = 3 ∨ x = 2 ∨ x = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_l229_22922


namespace NUMINAMATH_CALUDE_restaurant_bill_split_l229_22986

theorem restaurant_bill_split (num_people : ℕ) (individual_payment : ℚ) (original_bill : ℚ) : 
  num_people = 8 →
  individual_payment = 314.15 →
  original_bill = num_people * individual_payment →
  original_bill = 2513.20 := by
sorry

end NUMINAMATH_CALUDE_restaurant_bill_split_l229_22986


namespace NUMINAMATH_CALUDE_log_inequality_l229_22949

theorem log_inequality (a b c : ℝ) (h1 : a < b) (h2 : 0 < c) (h3 : c < 1) :
  a * Real.log c > b * Real.log c := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l229_22949


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l229_22900

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l229_22900


namespace NUMINAMATH_CALUDE_special_calculator_input_l229_22903

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Applies the special calculator operation to a number -/
def calculatorOperation (n : ℕ) : ℕ := reverseDigits (3 * n) + 2

theorem special_calculator_input (x : ℕ) :
  (1000 ≤ x ∧ x < 10000) →  -- x is a four-digit number
  calculatorOperation x = 2015 →
  x = 1034 := by sorry

end NUMINAMATH_CALUDE_special_calculator_input_l229_22903


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l229_22939

-- Define the rhombus
structure Rhombus where
  d1 : ℝ  -- First diagonal
  d2 : ℝ  -- Second diagonal
  area : ℝ  -- Area of the rhombus

-- Define the theorem
theorem inscribed_circle_radius (r : Rhombus) (h1 : r.d1 = 8) (h2 : r.d2 = 30) (h3 : r.area = 120) :
  let side := Real.sqrt ((r.d1/2)^2 + (r.d2/2)^2)
  let radius := r.area / (2 * side)
  radius = 60 / Real.sqrt 241 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l229_22939


namespace NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l229_22996

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral grid -/
def SpiralGrid :=
  Position → ℕ

/-- The size of the grid -/
def gridSize : ℕ := 17

/-- The center position of the grid -/
def centerPos : Position :=
  { row := 9, col := 9 }

/-- Creates a spiral grid with the given properties -/
def createSpiralGrid : SpiralGrid :=
  sorry

/-- Checks if a position is in the third row from the top -/
def isInThirdRow (p : Position) : Prop :=
  p.row = 3

/-- Finds the greatest number in the third row -/
def greatestInThirdRow (grid : SpiralGrid) : ℕ :=
  sorry

/-- Finds the least number in the third row -/
def leastInThirdRow (grid : SpiralGrid) : ℕ :=
  sorry

theorem spiral_grid_third_row_sum :
  let grid := createSpiralGrid
  greatestInThirdRow grid + leastInThirdRow grid = 528 := by
  sorry

end NUMINAMATH_CALUDE_spiral_grid_third_row_sum_l229_22996


namespace NUMINAMATH_CALUDE_base7_to_base10_conversion_l229_22998

/-- Converts a base-7 number represented as a list of digits to base 10 -/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base-7 representation of the number -/
def base7Number : List Nat := [5, 4, 6]

theorem base7_to_base10_conversion :
  base7ToBase10 base7Number = 327 := by sorry

end NUMINAMATH_CALUDE_base7_to_base10_conversion_l229_22998


namespace NUMINAMATH_CALUDE_triangle_shape_l229_22942

theorem triangle_shape (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + 2*b^2 = 2*b*(a+c) - c^2) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_shape_l229_22942


namespace NUMINAMATH_CALUDE_workshop_efficiency_l229_22904

theorem workshop_efficiency (x : ℝ) : 
  (1500 / x - 1500 / (2.5 * x) = 18) → x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_workshop_efficiency_l229_22904


namespace NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l229_22924

theorem sqrt_pi_squared_minus_6pi_plus_9 : 
  Real.sqrt (π^2 - 6*π + 9) = π - 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_pi_squared_minus_6pi_plus_9_l229_22924


namespace NUMINAMATH_CALUDE_speed_calculation_l229_22982

-- Define the given conditions
def field_area : ℝ := 50
def travel_time_minutes : ℝ := 2

-- Define the theorem
theorem speed_calculation :
  let diagonal := Real.sqrt (2 * field_area)
  let speed_m_per_hour := diagonal / (travel_time_minutes / 60)
  speed_m_per_hour / 1000 = 0.3 := by
sorry

end NUMINAMATH_CALUDE_speed_calculation_l229_22982


namespace NUMINAMATH_CALUDE_laura_five_dollar_bills_l229_22963

/-- Represents the number of bills of each denomination in Laura's piggy bank -/
structure PiggyBank where
  ones : ℕ
  twos : ℕ
  fives : ℕ

/-- The conditions of Laura's piggy bank -/
def laura_piggy_bank (pb : PiggyBank) : Prop :=
  pb.ones + pb.twos + pb.fives = 40 ∧
  pb.ones + 2 * pb.twos + 5 * pb.fives = 120 ∧
  pb.twos = 2 * pb.ones

theorem laura_five_dollar_bills :
  ∃ (pb : PiggyBank), laura_piggy_bank pb ∧ pb.fives = 16 :=
sorry

end NUMINAMATH_CALUDE_laura_five_dollar_bills_l229_22963


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l229_22958

theorem simplify_and_evaluate (x y : ℝ) 
  (hx : x = 2 + 3 * Real.sqrt 3) 
  (hy : y = 2 - 3 * Real.sqrt 3) : 
  (x^2 / (x - y)) - (y^2 / (x - y)) = 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l229_22958


namespace NUMINAMATH_CALUDE_problem_solution_l229_22995

theorem problem_solution (x y z : ℝ) 
  (h : y^2 + |x - 2023| + Real.sqrt (z - 4) = 6*y - 9) : 
  (y - z)^x = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l229_22995


namespace NUMINAMATH_CALUDE_number_of_factors_of_60_l229_22932

theorem number_of_factors_of_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_of_60_l229_22932


namespace NUMINAMATH_CALUDE_biathlon_bicycle_distance_l229_22990

/-- Given a biathlon with specified conditions, prove the distance of the bicycle race. -/
theorem biathlon_bicycle_distance 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (run_distance : ℝ) 
  (run_velocity : ℝ) :
  total_distance = 155 →
  total_time = 6 →
  run_distance = 10 →
  run_velocity = 10 →
  total_distance = run_distance + (total_time - run_distance / run_velocity) * 
    ((total_distance - run_distance) / (total_time - run_distance / run_velocity)) →
  total_distance - run_distance = 145 := by
  sorry

#check biathlon_bicycle_distance

end NUMINAMATH_CALUDE_biathlon_bicycle_distance_l229_22990


namespace NUMINAMATH_CALUDE_equation_solution_l229_22918

theorem equation_solution (x y : ℝ) : 
  (x^4 + 1) * (y^4 + 1) = 4 * x^2 * y^2 ↔ 
  ((x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) ∨ (x = 1 ∧ y = -1) ∨ (x = -1 ∧ y = 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l229_22918


namespace NUMINAMATH_CALUDE_jelly_bean_multiple_l229_22933

/-- The number of vanilla jelly beans -/
def vanilla_beans : ℕ := 120

/-- The total number of jelly beans -/
def total_beans : ℕ := 770

/-- The number of grape jelly beans as a function of the multiple -/
def grape_beans (x : ℕ) : ℕ := 50 + x * vanilla_beans

/-- The theorem stating that the multiple of vanilla jelly beans taken as grape jelly beans is 5 -/
theorem jelly_bean_multiple :
  ∃ x : ℕ, x = 5 ∧ vanilla_beans + grape_beans x = total_beans :=
sorry

end NUMINAMATH_CALUDE_jelly_bean_multiple_l229_22933


namespace NUMINAMATH_CALUDE_geese_ratio_l229_22913

/-- Represents the number of ducks and geese bought by a person -/
structure DucksAndGeese where
  ducks : ℕ
  geese : ℕ

/-- The problem setup -/
def market_problem (lily rayden : DucksAndGeese) : Prop :=
  rayden.ducks = 3 * lily.ducks ∧
  lily.ducks = 20 ∧
  lily.geese = 10 ∧
  rayden.ducks + rayden.geese = lily.ducks + lily.geese + 70

/-- The theorem to prove -/
theorem geese_ratio (lily rayden : DucksAndGeese) 
  (h : market_problem lily rayden) : 
  rayden.geese = 4 * lily.geese := by
  sorry


end NUMINAMATH_CALUDE_geese_ratio_l229_22913


namespace NUMINAMATH_CALUDE_y_congruence_l229_22971

theorem y_congruence (y : ℤ) 
  (h1 : (2 + y) % (2^3) = (2 * 2) % (2^3))
  (h2 : (4 + y) % (4^3) = (4 * 2) % (4^3))
  (h3 : (6 + y) % (6^3) = (6 * 2) % (6^3)) :
  y % 24 = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_congruence_l229_22971


namespace NUMINAMATH_CALUDE_sum_even_factors_630_eq_1248_l229_22911

/-- The sum of all positive even factors of 630 -/
def sum_even_factors_630 : ℕ := sorry

/-- 630 is the number we're examining -/
def n : ℕ := 630

/-- Theorem stating that the sum of all positive even factors of 630 is 1248 -/
theorem sum_even_factors_630_eq_1248 : sum_even_factors_630 = 1248 := by sorry

end NUMINAMATH_CALUDE_sum_even_factors_630_eq_1248_l229_22911


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l229_22948

theorem fraction_to_decimal : (47 : ℚ) / (2^2 * 5^4) = 0.0188 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l229_22948


namespace NUMINAMATH_CALUDE_alex_income_l229_22934

/-- Represents the tax structure and Alex's tax payment --/
structure TaxSystem where
  q : ℝ  -- Base tax rate as a percentage
  income : ℝ  -- Alex's annual income
  total_tax : ℝ  -- Total tax paid by Alex

/-- The tax system satisfies the given conditions --/
def valid_tax_system (ts : TaxSystem) : Prop :=
  ts.total_tax = 
    (if ts.income ≤ 50000 then
      (ts.q / 100) * ts.income
    else
      (ts.q / 100) * 50000 + ((ts.q + 3) / 100) * (ts.income - 50000))
  ∧ ts.total_tax = ((ts.q + 0.5) / 100) * ts.income

/-- Theorem stating that Alex's income is $60000 --/
theorem alex_income (ts : TaxSystem) (h : valid_tax_system ts) : ts.income = 60000 := by
  sorry

end NUMINAMATH_CALUDE_alex_income_l229_22934


namespace NUMINAMATH_CALUDE_x_wins_probability_l229_22965

/-- Represents a soccer tournament with the given conditions -/
structure SoccerTournament where
  num_teams : Nat
  games_per_team : Nat
  win_probability : ℚ
  
/-- Represents the outcome of the tournament for two specific teams -/
structure TournamentOutcome where
  team_x_points : Nat
  team_y_points : Nat

/-- Calculates the probability of team X finishing with more points than team Y -/
def probability_x_wins (t : SoccerTournament) : ℚ :=
  sorry

/-- The main theorem stating the probability for the given conditions -/
theorem x_wins_probability (t : SoccerTournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_team = 7)
  (h3 : t.win_probability = 1/2) :
  probability_x_wins t = 561/1024 := by
  sorry

end NUMINAMATH_CALUDE_x_wins_probability_l229_22965


namespace NUMINAMATH_CALUDE_solve_for_t_l229_22974

theorem solve_for_t (Q m h t : ℝ) (hQ : Q > 0) (hm : m ≠ 0) (hh : h > -2) :
  Q = m^2 / (2 + h)^t ↔ t = Real.log (m^2 / Q) / Real.log (2 + h) :=
sorry

end NUMINAMATH_CALUDE_solve_for_t_l229_22974


namespace NUMINAMATH_CALUDE_inequality_preservation_l229_22927

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l229_22927


namespace NUMINAMATH_CALUDE_tv_conditional_probability_l229_22983

theorem tv_conditional_probability 
  (p_10000 : ℝ) 
  (p_15000 : ℝ) 
  (h1 : p_10000 = 0.80) 
  (h2 : p_15000 = 0.60) : 
  p_15000 / p_10000 = 0.75 := by
sorry

end NUMINAMATH_CALUDE_tv_conditional_probability_l229_22983


namespace NUMINAMATH_CALUDE_pauls_crayons_l229_22914

/-- Paul's crayon problem -/
theorem pauls_crayons (initial given lost broken traded : ℕ) : 
  initial = 250 → 
  given = 150 → 
  lost = 512 → 
  broken = 75 → 
  traded = 35 → 
  lost - (given + broken + traded) = 252 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_l229_22914


namespace NUMINAMATH_CALUDE_city_budget_properties_l229_22988

/-- CityBudget represents the budget allocation for a city's rail transit line project over three years. -/
structure CityBudget where
  total : ℝ  -- Total budget in billion yuan
  track_laying : ℝ → ℝ  -- Investment in track laying for each year
  relocation : ℝ → ℝ  -- Investment in relocation for each year
  auxiliary : ℝ → ℝ  -- Investment in auxiliary facilities for each year
  b : ℝ  -- Annual increase in track laying investment

/-- Investment ratios and conditions for the city budget -/
def budget_conditions (budget : CityBudget) : Prop :=
  ∃ x : ℝ,
    budget.track_laying 0 = 2 * x ∧
    budget.relocation 0 = 4 * x ∧
    budget.auxiliary 0 = x ∧
    (∀ t : ℝ, t ≥ 0 ∧ t < 3 → budget.track_laying (t + 1) = budget.track_laying t + budget.b) ∧
    (budget.track_laying 0 + budget.track_laying 1 + budget.track_laying 2 = 54) ∧
    (∃ y : ℝ, y > 0 ∧ y < 1 ∧ 
      budget.relocation 1 = budget.relocation 0 * (1 - y) ∧
      budget.relocation 2 = budget.relocation 1 * (1 - y) ∧
      budget.relocation 2 = 5) ∧
    (budget.auxiliary 1 = budget.auxiliary 0 * (1 + 1.5 * budget.b / (2 * x))) ∧
    (budget.auxiliary 2 = budget.auxiliary 0 + budget.auxiliary 1 + 4) ∧
    (budget.track_laying 0 + budget.track_laying 1 + budget.track_laying 2) / 
    (budget.auxiliary 0 + budget.auxiliary 1 + budget.auxiliary 2) = 3 / 2

/-- Main theorem stating the properties of the city budget -/
theorem city_budget_properties (budget : CityBudget) (h : budget_conditions budget) :
  (budget.auxiliary 0 + budget.auxiliary 1 + budget.auxiliary 2 = 36) ∧
  (budget.track_laying 0 + budget.relocation 0 + budget.auxiliary 0 = 35) ∧
  (∃ y : ℝ, y = 0.5 ∧ 
    budget.relocation 1 = budget.relocation 0 * (1 - y) ∧
    budget.relocation 2 = budget.relocation 1 * (1 - y)) := by
  sorry

end NUMINAMATH_CALUDE_city_budget_properties_l229_22988


namespace NUMINAMATH_CALUDE_trajectory_equation_l229_22951

-- Define the fixed circle C
def C (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

-- Define the line that M is tangent to
def L (y : ℝ) : Prop := y = 2

-- Define the moving circle M
def M (x y : ℝ) : Prop := ∃ (r : ℝ), r > 0 ∧ 
  (∀ (x' y' : ℝ), C x' y' → (x - x')^2 + (y - y')^2 = (1 + r)^2) ∧
  (∀ (y' : ℝ), L y' → |y - y'| = r)

-- State the theorem
theorem trajectory_equation :
  ∀ (x y : ℝ), M x y → x^2 = -12*y := by sorry

end NUMINAMATH_CALUDE_trajectory_equation_l229_22951


namespace NUMINAMATH_CALUDE_expand_expression_l229_22921

theorem expand_expression (y : ℝ) : 5 * (4 * y^3 - 3 * y^2 + y - 7) = 20 * y^3 - 15 * y^2 + 5 * y - 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l229_22921


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_iff_l229_22915

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.mk (|a| - 1) (a + 1)

/-- A point is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

/-- Theorem stating the necessary and sufficient condition for z to be in the fourth quadrant -/
theorem z_in_fourth_quadrant_iff (a : ℝ) : in_fourth_quadrant (z a) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_iff_l229_22915


namespace NUMINAMATH_CALUDE_max_sum_perfect_square_fraction_l229_22954

def is_perfect_square (n : ℚ) : Prop := ∃ m : ℕ, n = (m : ℚ) ^ 2

def is_digit (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 9

theorem max_sum_perfect_square_fraction :
  ∀ A B C D : ℕ,
    is_digit A → is_digit B → is_digit C → is_digit D →
    A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
    is_perfect_square ((A + B : ℚ) / (C + D)) →
    ∀ A' B' C' D' : ℕ,
      is_digit A' → is_digit B' → is_digit C' → is_digit D' →
      A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
      is_perfect_square ((A' + B' : ℚ) / (C' + D')) →
      (A + B : ℚ) / (C + D) ≥ (A' + B' : ℚ) / (C' + D') →
      A + B = 16 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_perfect_square_fraction_l229_22954


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l229_22943

theorem absolute_value_equation_solution :
  ∃ x : ℝ, (|x - 25| + |x - 21| = |3*x - 75|) ∧ (x = 71/3) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l229_22943


namespace NUMINAMATH_CALUDE_algorithm_output_l229_22908

def sum_odd_numbers (n : Nat) : Nat :=
  List.sum (List.range n |>.filter (λ x => x % 2 = 1))

theorem algorithm_output : 1 + sum_odd_numbers 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_algorithm_output_l229_22908


namespace NUMINAMATH_CALUDE_hidden_primes_average_l229_22961

-- Define the visible numbers on the cards
def visible_numbers : List Nat := [44, 59, 38]

-- Define a function to check if a number is prime
def is_prime (n : Nat) : Prop := Nat.Prime n

-- Define the property that the sum of numbers on each card is equal
def equal_sums (x y z : Nat) : Prop :=
  44 + x = 59 + y ∧ 59 + y = 38 + z

-- The main theorem
theorem hidden_primes_average (x y z : Nat) : 
  is_prime x ∧ is_prime y ∧ is_prime z ∧ 
  equal_sums x y z → 
  (x + y + z) / 3 = 14 :=
sorry

end NUMINAMATH_CALUDE_hidden_primes_average_l229_22961


namespace NUMINAMATH_CALUDE_pizza_coverage_l229_22956

theorem pizza_coverage (pizza_diameter : ℝ) (pepperoni_diameter : ℝ) (num_pepperoni : ℕ) : 
  pizza_diameter = 2 * pepperoni_diameter →
  num_pepperoni = 32 →
  (num_pepperoni * (pepperoni_diameter / 2)^2 * π) / ((pizza_diameter / 2)^2 * π) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_pizza_coverage_l229_22956


namespace NUMINAMATH_CALUDE_factor_t_squared_minus_64_l229_22931

theorem factor_t_squared_minus_64 (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end NUMINAMATH_CALUDE_factor_t_squared_minus_64_l229_22931
