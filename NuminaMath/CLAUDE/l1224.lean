import Mathlib

namespace sum_square_free_coefficients_eight_l1224_122497

/-- A function that calculates the sum of coefficients of square-free terms -/
def sumSquareFreeCoefficients (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | k + 2 => sumSquareFreeCoefficients (k + 1) + (k + 1) * sumSquareFreeCoefficients k

/-- The main theorem stating that the sum of coefficients of square-free terms
    in the product of (1 + xᵢxⱼ) for 1 ≤ i < j ≤ 8 is 764 -/
theorem sum_square_free_coefficients_eight :
  sumSquareFreeCoefficients 8 = 764 := by
  sorry

/-- Helper lemma: The recurrence relation for sumSquareFreeCoefficients -/
lemma sum_square_free_coefficients_recurrence (n : ℕ) :
  n ≥ 2 →
  sumSquareFreeCoefficients n = 
    sumSquareFreeCoefficients (n-1) + (n-1) * sumSquareFreeCoefficients (n-2) := by
  sorry

end sum_square_free_coefficients_eight_l1224_122497


namespace unique_integer_divisibility_l1224_122479

theorem unique_integer_divisibility : ∃! n : ℕ, n > 1 ∧
  ∀ p : ℕ, Prime p → (p ∣ (n^6 - 1) → p ∣ ((n^3 - 1) * (n^2 - 1))) ∧
  n = 2 := by
  sorry

end unique_integer_divisibility_l1224_122479


namespace point_A_moves_to_vertex_3_l1224_122436

/-- Represents a vertex of a cube --/
structure Vertex where
  label : Nat
  onGreenFace : Bool
  onDistantWhiteFace : Bool
  onBottomRightWhiteFace : Bool

/-- Represents the rotation of a cube --/
def rotatedCube : List Vertex → List Vertex := sorry

/-- The initial position of point A --/
def pointA : Vertex := {
  label := 0,
  onGreenFace := true,
  onDistantWhiteFace := true,
  onBottomRightWhiteFace := true
}

/-- Theorem stating that point A moves to vertex 3 after rotation --/
theorem point_A_moves_to_vertex_3 (cube : List Vertex) :
  ∃ v ∈ rotatedCube cube,
    v.label = 3 ∧
    v.onGreenFace = true ∧
    v.onDistantWhiteFace = true ∧
    v.onBottomRightWhiteFace = true :=
  sorry

end point_A_moves_to_vertex_3_l1224_122436


namespace floor_ceiling_sum_l1224_122484

theorem floor_ceiling_sum : ⌊(0.999 : ℝ)⌋ + ⌈(2.001 : ℝ)⌉ = 3 := by sorry

end floor_ceiling_sum_l1224_122484


namespace min_box_height_l1224_122451

def box_height (x : ℝ) : ℝ := x + 5

def surface_area (x : ℝ) : ℝ := 6 * x^2 + 20 * x

def base_perimeter_plus_height (x : ℝ) : ℝ := 5 * x + 5

theorem min_box_height :
  ∀ x : ℝ,
  x > 0 →
  surface_area x ≥ 150 →
  base_perimeter_plus_height x ≥ 25 →
  box_height x ≥ 9 ∧
  (∃ y : ℝ, y > 0 ∧ surface_area y ≥ 150 ∧ base_perimeter_plus_height y ≥ 25 ∧ box_height y = 9) :=
by sorry

end min_box_height_l1224_122451


namespace expression_factorization_l1224_122470

theorem expression_factorization (x : ℝ) :
  (4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8) = 2 * x^2 * (5 * x + 31) :=
by sorry

end expression_factorization_l1224_122470


namespace adults_on_bus_l1224_122483

theorem adults_on_bus (total_passengers : ℕ) (children_fraction : ℚ) : 
  total_passengers = 360 → children_fraction = 3/7 → 
  (total_passengers : ℚ) * (1 - children_fraction) = 205 := by
sorry

end adults_on_bus_l1224_122483


namespace marble_problem_l1224_122405

theorem marble_problem (A V : ℤ) (x : ℤ) : 
  (A + x = V - x) ∧ (V + 2*x = A - 2*x + 30) → x = 5 :=
by
  sorry

end marble_problem_l1224_122405


namespace propositions_truth_l1224_122488

-- Proposition ①
def proposition_1 : Prop := 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 ≥ 0)

-- Proposition ②
def proposition_2 : Prop := 
  (¬ ∃ x₀ : ℝ, x₀^2 - x₀ > 0) ↔ (∀ x : ℝ, x^2 - x < 0)

-- Proposition ③
def proposition_3 : Prop := 
  ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 2*x ≤ 3

-- Proposition ④
def proposition_4 : Prop := 
  ∃ x₀ : ℝ, x₀^2 + 1/(x₀^2 + 1) ≤ 1

theorem propositions_truth : 
  ¬ proposition_1 ∧ 
  ¬ proposition_2 ∧ 
  proposition_3 ∧ 
  proposition_4 := by sorry

end propositions_truth_l1224_122488


namespace increasing_function_property_l1224_122426

def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem increasing_function_property (f : ℝ → ℝ) (h : increasing_function f) (a b : ℝ) :
  (a + b ≥ 0) ↔ (f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end increasing_function_property_l1224_122426


namespace boat_speed_theorem_l1224_122466

/-- Represents the speed of a boat in different conditions --/
structure BoatSpeed where
  stillWater : ℝ  -- Speed in still water
  upstream : ℝ    -- Speed against the stream
  downstream : ℝ  -- Speed with the stream

/-- 
Given a man's rowing speed in still water is 4 km/h and 
his speed against the stream is 4 km/h, 
his speed with the stream is also 4 km/h.
-/
theorem boat_speed_theorem (speed : BoatSpeed) 
  (h1 : speed.stillWater = 4)
  (h2 : speed.upstream = 4) :
  speed.downstream = 4 := by
  sorry

#check boat_speed_theorem

end boat_speed_theorem_l1224_122466


namespace triangle_altitude_segment_length_l1224_122444

theorem triangle_altitude_segment_length 
  (a b c h d : ℝ) 
  (triangle_sides : a = 30 ∧ b = 70 ∧ c = 80) 
  (altitude_condition : h^2 = b^2 - d^2) 
  (segment_condition : a^2 = h^2 + (c - d)^2) : 
  d = 65 := by sorry

end triangle_altitude_segment_length_l1224_122444


namespace A_union_complement_B_equals_one_three_l1224_122442

-- Define the universal set U
def U : Set Nat := {1, 2, 3}

-- Define set A
def A : Set Nat := {1}

-- Define set B
def B : Set Nat := {1, 2}

-- Theorem statement
theorem A_union_complement_B_equals_one_three :
  A ∪ (U \ B) = {1, 3} := by sorry

end A_union_complement_B_equals_one_three_l1224_122442


namespace initial_peanuts_count_l1224_122480

theorem initial_peanuts_count (initial final added : ℕ) 
  (h1 : added = 4)
  (h2 : final = 8)
  (h3 : final = initial + added) : 
  initial = 4 := by
sorry

end initial_peanuts_count_l1224_122480


namespace tan_value_for_given_point_l1224_122402

/-- If the terminal side of angle θ passes through the point (-√3/2, 1/2), then tan θ = -√3/3 -/
theorem tan_value_for_given_point (θ : Real) (h : ∃ (r : Real), r * (Real.cos θ) = -Real.sqrt 3 / 2 ∧ r * (Real.sin θ) = 1 / 2) : 
  Real.tan θ = -Real.sqrt 3 / 3 := by
  sorry

end tan_value_for_given_point_l1224_122402


namespace unique_three_digit_number_l1224_122493

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Sum of all permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  100 * (n.hundreds + n.tens + n.ones) +
  10 * (n.hundreds + n.tens + n.ones) +
  (n.hundreds + n.tens + n.ones)

/-- The sum of digits of a three-digit number -/
def sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

theorem unique_three_digit_number :
  ∀ n : ThreeDigitNumber,
    sumOfPermutations n = 4410 ∧
    Even (sumOfDigits n) →
    n.hundreds = 4 ∧ n.tens = 4 ∧ n.ones = 4 :=
by sorry

end unique_three_digit_number_l1224_122493


namespace isabellas_hair_growth_l1224_122496

/-- Calculates the final length of Isabella's hair after growth -/
def final_hair_length (initial_length growth : ℕ) : ℕ := initial_length + growth

/-- Theorem: Isabella's hair length after growth -/
theorem isabellas_hair_growth (initial_length growth : ℕ) 
  (h1 : initial_length = 18) 
  (h2 : growth = 4) : 
  final_hair_length initial_length growth = 22 := by
  sorry

end isabellas_hair_growth_l1224_122496


namespace binary_arithmetic_equality_l1224_122422

/-- Convert a binary number (represented as a list of 0s and 1s) to its decimal equivalent -/
def binary_to_decimal (binary : List Nat) : Nat :=
  binary.foldr (fun bit acc => 2 * acc + bit) 0

/-- Convert a decimal number to its binary representation -/
def decimal_to_binary (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 2) ((m % 2) :: acc)
  aux n []

theorem binary_arithmetic_equality :
  let a := [1, 0, 0, 1, 1, 0]  -- 100110₂
  let b := [1, 0, 0, 1]        -- 1001₂
  let c := [1, 1, 0]           -- 110₂
  let d := [1, 1]              -- 11₂
  let result := [1, 0, 1, 1, 1, 1, 0]  -- 1011110₂
  (binary_to_decimal a + binary_to_decimal b) * binary_to_decimal c / binary_to_decimal d =
  binary_to_decimal result := by sorry

end binary_arithmetic_equality_l1224_122422


namespace common_tangent_sum_l1224_122415

-- Define the parabolas
def Q₁ (x y : ℝ) : Prop := y = x^2 + 53/50
def Q₂ (x y : ℝ) : Prop := x = y^2 + 91/8

-- Define the common tangent line
def M (p q r : ℕ) (x y : ℝ) : Prop := p * x + q * y = r

-- Main theorem
theorem common_tangent_sum (p q r : ℕ) :
  (p > 0) →
  (q > 0) →
  (r > 0) →
  (Nat.gcd p q = 1) →
  (Nat.gcd p r = 1) →
  (Nat.gcd q r = 1) →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    Q₁ x₁ y₁ ∧ 
    Q₂ x₂ y₂ ∧ 
    M p q r x₁ y₁ ∧ 
    M p q r x₂ y₂ ∧
    (∃ (m : ℚ), q = m * p)) →
  p + q + r = 9 := by
  sorry

end common_tangent_sum_l1224_122415


namespace max_residents_is_475_l1224_122450

/-- Represents the configuration of a section in the block of flats -/
structure SectionConfig where
  floors : Nat
  apartments_per_floor : Nat
  apartment_capacities : List Nat

/-- Calculates the maximum capacity of a single section -/
def section_capacity (config : SectionConfig) : Nat :=
  config.floors * (config.apartment_capacities.sum)

/-- Represents the configuration of the entire block of flats -/
structure BlockConfig where
  lower : SectionConfig
  middle : SectionConfig
  upper : SectionConfig

/-- Calculates the maximum capacity of the entire block -/
def block_capacity (config : BlockConfig) : Nat :=
  section_capacity config.lower + section_capacity config.middle + section_capacity config.upper

/-- The actual configuration of the block as described in the problem -/
def actual_block_config : BlockConfig :=
  { lower := { floors := 4, apartments_per_floor := 5, apartment_capacities := [4, 4, 5, 5, 6] },
    middle := { floors := 5, apartments_per_floor := 6, apartment_capacities := [3, 3, 3, 4, 4, 6] },
    upper := { floors := 6, apartments_per_floor := 5, apartment_capacities := [8, 8, 8, 10, 10] } }

/-- Theorem stating that the maximum capacity of the block is 475 -/
theorem max_residents_is_475 : block_capacity actual_block_config = 475 := by
  sorry

end max_residents_is_475_l1224_122450


namespace selling_price_is_50_l1224_122475

/-- Represents the manufacturing and sales data for horseshoe sets -/
structure HorseshoeData where
  initial_outlay : ℕ
  cost_per_set : ℕ
  sets_sold : ℕ
  profit : ℕ
  selling_price : ℕ

/-- Calculates the total manufacturing cost -/
def total_manufacturing_cost (data : HorseshoeData) : ℕ :=
  data.initial_outlay + data.cost_per_set * data.sets_sold

/-- Calculates the total revenue -/
def total_revenue (data : HorseshoeData) : ℕ :=
  data.selling_price * data.sets_sold

/-- Theorem stating that the selling price is $50 given the conditions -/
theorem selling_price_is_50 (data : HorseshoeData) 
    (h1 : data.initial_outlay = 10000)
    (h2 : data.cost_per_set = 20)
    (h3 : data.sets_sold = 500)
    (h4 : data.profit = 5000)
    (h5 : data.profit = total_revenue data - total_manufacturing_cost data) :
  data.selling_price = 50 := by
  sorry


end selling_price_is_50_l1224_122475


namespace jewelry_thief_l1224_122490

-- Define the set of people
inductive Person : Type
  | A | B | C | D

-- Define the properties
def is_telling_truth (p : Person) : Prop := sorry
def is_thief (p : Person) : Prop := sorry

-- State the theorem
theorem jewelry_thief :
  -- Only one person is telling the truth
  (∃! p : Person, is_telling_truth p) →
  -- Only one person stole the jewelry
  (∃! p : Person, is_thief p) →
  -- A's statement
  (is_telling_truth Person.A ↔ ¬is_thief Person.A) →
  -- B's statement
  (is_telling_truth Person.B ↔ is_thief Person.C) →
  -- C's statement
  (is_telling_truth Person.C ↔ is_thief Person.D) →
  -- D's statement
  (is_telling_truth Person.D ↔ ¬is_thief Person.D) →
  -- Conclusion: A stole the jewelry
  is_thief Person.A :=
by
  sorry


end jewelry_thief_l1224_122490


namespace range_of_a_l1224_122495

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x ≤ a - 4 ∨ x ≥ a + 4) → (x ≤ 1 ∨ x ≥ 2)) → 
  a ∈ Set.Icc (-2) 5 := by
  sorry

end range_of_a_l1224_122495


namespace division_result_l1224_122489

theorem division_result : (180 : ℚ) / (12 + 13 * 3 - 5) = 90 / 23 := by
  sorry

end division_result_l1224_122489


namespace last_two_digits_of_7_power_last_two_digits_of_7_2017_l1224_122430

theorem last_two_digits_of_7_power (n : ℕ) :
  (7^n) % 100 = (7^(n % 4 + 4)) % 100 :=
sorry

theorem last_two_digits_of_7_2017 :
  (7^2017) % 100 = 49 :=
sorry

end last_two_digits_of_7_power_last_two_digits_of_7_2017_l1224_122430


namespace students_per_row_l1224_122401

theorem students_per_row (S R x : ℕ) : 
  S = x * R + 6 →  -- First scenario
  S = 12 * (R - 3) →  -- Second scenario
  S = 6 * R →  -- Third condition
  x = 5 := by
sorry

end students_per_row_l1224_122401


namespace min_lcm_a_c_l1224_122453

theorem min_lcm_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 24) (h2 : Nat.lcm b c = 18) :
  ∃ (a' c' : ℕ), Nat.lcm a' c' = 12 ∧ 
    (∀ (x y : ℕ), Nat.lcm x b = 24 → Nat.lcm b y = 18 → Nat.lcm x y ≥ 12) := by
  sorry

end min_lcm_a_c_l1224_122453


namespace lunchroom_students_l1224_122456

theorem lunchroom_students (tables : ℕ) (avg_students : ℚ) : 
  tables = 34 →
  avg_students = 5666666667 / 1000000000 →
  ∃ (total_students : ℕ), total_students = 204 ∧ total_students % tables = 0 := by
  sorry

end lunchroom_students_l1224_122456


namespace automotive_test_time_l1224_122412

/-- Proves that given a car driven the same distance three times at speeds of 4, 5, and 6 miles per hour respectively, and a total distance of 180 miles, the total time taken is 37 hours. -/
theorem automotive_test_time (total_distance : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_distance = 180 ∧ 
  speed1 = 4 ∧ 
  speed2 = 5 ∧ 
  speed3 = 6 → 
  (total_distance / (3 * speed1) + total_distance / (3 * speed2) + total_distance / (3 * speed3)) = 37 := by
  sorry

#check automotive_test_time

end automotive_test_time_l1224_122412


namespace third_term_is_nine_l1224_122432

/-- A sequence of 5 numbers with specific properties -/
def MagazineSequence (a : Fin 5 → ℕ) : Prop :=
  a 0 = 3 ∧ a 1 = 4 ∧ a 3 = 9 ∧ a 4 = 13 ∧
  ∀ i : Fin 3, (a (i + 1) - a i) - (a (i + 2) - a (i + 1)) = 
               (a (i + 2) - a (i + 1)) - (a (i + 3) - a (i + 2))

/-- The third term in the sequence is 9 -/
theorem third_term_is_nine (a : Fin 5 → ℕ) (h : MagazineSequence a) : a 2 = 9 := by
  sorry

end third_term_is_nine_l1224_122432


namespace product_of_sqrt_diff_eq_one_l1224_122417

theorem product_of_sqrt_diff_eq_one :
  let A := Real.sqrt 3008 + Real.sqrt 3009
  let B := -Real.sqrt 3008 - Real.sqrt 3009
  let C := Real.sqrt 3008 - Real.sqrt 3009
  let D := Real.sqrt 3009 - Real.sqrt 3008
  A * B * C * D = 1 := by
sorry

end product_of_sqrt_diff_eq_one_l1224_122417


namespace leak_empty_time_proof_l1224_122462

/-- Represents the time it takes for a pipe to fill a tank -/
def fill_time_no_leak : ℝ := 8

/-- Represents the time it takes for a pipe to fill a tank with a leak -/
def fill_time_with_leak : ℝ := 12

/-- Represents the time it takes for the leak to empty a full tank -/
def leak_empty_time : ℝ := 24

/-- Theorem stating that given the fill times with and without a leak, 
    the time for the leak to empty the tank is 24 hours -/
theorem leak_empty_time_proof : 
  ∀ (fill_rate : ℝ) (leak_rate : ℝ),
  fill_rate = 1 / fill_time_no_leak →
  fill_rate - leak_rate = 1 / fill_time_with_leak →
  1 / leak_rate = leak_empty_time :=
by sorry

end leak_empty_time_proof_l1224_122462


namespace binomial_20_19_l1224_122449

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by sorry

end binomial_20_19_l1224_122449


namespace ten_year_old_dog_human_years_l1224_122486

/-- Calculates the equivalent human years for a dog's age. -/
def dogYearsToHumanYears (dogAge : ℕ) : ℕ :=
  if dogAge = 0 then 0
  else if dogAge = 1 then 15
  else if dogAge = 2 then 24
  else 24 + 5 * (dogAge - 2)

/-- Theorem: A 10-year-old dog has lived 64 human years. -/
theorem ten_year_old_dog_human_years :
  dogYearsToHumanYears 10 = 64 := by
  sorry

end ten_year_old_dog_human_years_l1224_122486


namespace fifth_match_goals_l1224_122481

/-- A football player's goal-scoring record over 5 matches -/
structure FootballRecord where
  total_matches : Nat
  total_goals : Nat
  fifth_match_goals : Nat
  average_increase : Rat

/-- The conditions of the problem -/
def problem_conditions (record : FootballRecord) : Prop :=
  record.total_matches = 5 ∧
  record.total_goals = 8 ∧
  record.average_increase = 1/10 ∧
  (4 * (record.total_goals - record.fifth_match_goals)) / 4 + record.average_increase = 
    record.total_goals / record.total_matches

/-- The theorem stating that under the given conditions, the player scored 2 goals in the fifth match -/
theorem fifth_match_goals (record : FootballRecord) 
  (h : problem_conditions record) : record.fifth_match_goals = 2 := by
  sorry


end fifth_match_goals_l1224_122481


namespace emily_calculation_l1224_122474

def round_to_nearest_ten (x : ℤ) : ℤ :=
  (x + 5) / 10 * 10

theorem emily_calculation : round_to_nearest_ten ((68 + 74 + 59) - 20) = 180 := by
  sorry

end emily_calculation_l1224_122474


namespace sams_work_days_l1224_122418

theorem sams_work_days (total_days : ℕ) (daily_wage : ℤ) (daily_loss : ℤ) (total_earnings : ℤ) :
  total_days = 20 ∧ daily_wage = 60 ∧ daily_loss = 30 ∧ total_earnings = 660 →
  ∃ (days_not_worked : ℕ),
    days_not_worked = 6 ∧
    days_not_worked ≤ total_days ∧
    (total_days - days_not_worked) * daily_wage - days_not_worked * daily_loss = total_earnings :=
by sorry

end sams_work_days_l1224_122418


namespace five_people_six_chairs_l1224_122471

/-- The number of ways to arrange n people in m chairs in a row -/
def arrangements (n m : ℕ) : ℕ :=
  (m.factorial) / ((m - n).factorial)

/-- Theorem: There are 720 ways to arrange 5 people in a row of 6 chairs -/
theorem five_people_six_chairs : arrangements 5 6 = 720 := by
  sorry

end five_people_six_chairs_l1224_122471


namespace geometric_sequence_formula_l1224_122445

/-- A geometric sequence {a_n} with a_1 = 3 and a_4 = 81 has the general term formula a_n = 3^n -/
theorem geometric_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∀ n : ℕ, a n = 3^n := by
  sorry

end geometric_sequence_formula_l1224_122445


namespace evaluate_expression_l1224_122419

theorem evaluate_expression : 2003^3 - 2002 * 2003^2 - 2002^2 * 2003 + 2002^3 = 4005 := by
  sorry

end evaluate_expression_l1224_122419


namespace product_of_successive_numbers_l1224_122457

theorem product_of_successive_numbers : 
  let n : ℝ := 64.4980619863884
  let product := n * (n + 1)
  ∀ ε > 0, |product - 4225| < ε
:= by sorry

end product_of_successive_numbers_l1224_122457


namespace pretzel_problem_l1224_122406

theorem pretzel_problem (barry_pretzels shelly_pretzels angie_pretzels : ℕ) : 
  barry_pretzels = 12 →
  shelly_pretzels = barry_pretzels / 2 →
  angie_pretzels = 3 * shelly_pretzels →
  angie_pretzels = 18 := by
  sorry

end pretzel_problem_l1224_122406


namespace commercial_length_l1224_122463

theorem commercial_length (original_length : ℝ) : 
  (original_length * 0.7 = 21) → original_length = 30 := by
  sorry

end commercial_length_l1224_122463


namespace sum_of_squares_l1224_122465

theorem sum_of_squares (x y : ℝ) (h1 : x * y = 120) (h2 : x + y = 23) : x^2 + y^2 = 289 := by
  sorry

end sum_of_squares_l1224_122465


namespace fuel_tank_capacity_l1224_122435

/-- Represents the problem of determining a fuel tank's capacity --/
theorem fuel_tank_capacity :
  ∀ (capacity : ℝ) 
    (ethanol_percent_A : ℝ) 
    (ethanol_percent_B : ℝ) 
    (total_ethanol : ℝ) 
    (fuel_A_volume : ℝ),
  ethanol_percent_A = 0.12 →
  ethanol_percent_B = 0.16 →
  total_ethanol = 30 →
  fuel_A_volume = 106 →
  ethanol_percent_A * fuel_A_volume + 
  ethanol_percent_B * (capacity - fuel_A_volume) = total_ethanol →
  capacity = 214 := by
sorry

end fuel_tank_capacity_l1224_122435


namespace cube_surface_area_equal_volume_l1224_122434

/-- The surface area of a cube with the same volume as a rectangular prism -/
theorem cube_surface_area_equal_volume (a b c : ℝ) (ha : a = 5) (hb : b = 7) (hc : c = 10) :
  6 * ((a * b * c) ^ (1/3 : ℝ))^2 = 6 * (350 ^ (1/3 : ℝ))^2 := by
  sorry

end cube_surface_area_equal_volume_l1224_122434


namespace polynomial_factorization_l1224_122455

theorem polynomial_factorization (a : ℝ) : 
  (a^2 - 4*a + 2) * (a^2 - 4*a + 6) + 4 = (a - 2)^4 := by
  sorry

end polynomial_factorization_l1224_122455


namespace min_value_reciprocal_sum_l1224_122460

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2 : ℝ)^(2*x + 2*y) = 2) : 
  ∃ (m : ℝ), m = 3 + 2 * Real.sqrt 2 ∧ ∀ (a b : ℝ), a > 0 → b > 0 → (2 : ℝ)^(2*a + 2*b) = 2 → 1/a + 1/b ≥ m :=
sorry

end min_value_reciprocal_sum_l1224_122460


namespace distinct_positive_numbers_properties_l1224_122441

theorem distinct_positive_numbers_properties (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) : 
  ((a - b)^2 + (b - c)^2 + (c - a)^2 ≠ 0) ∧ 
  (a > b ∨ a < b ∨ a = b) ∧ 
  (a ≠ c ∧ b ≠ c ∧ a ≠ b) := by
  sorry

end distinct_positive_numbers_properties_l1224_122441


namespace tShirts_per_package_example_l1224_122440

/-- Given a total number of white t-shirts and a number of packages,
    calculate the number of t-shirts per package. -/
def tShirtsPerPackage (total : ℕ) (packages : ℕ) : ℕ :=
  total / packages

/-- Theorem: Given 70 white t-shirts in 14 packages,
    prove that each package contains 5 t-shirts. -/
theorem tShirts_per_package_example :
  tShirtsPerPackage 70 14 = 5 := by
  sorry

end tShirts_per_package_example_l1224_122440


namespace leadership_combinations_l1224_122472

def tribe_size : ℕ := 15
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def num_inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem leadership_combinations : 
  (tribe_size) * 
  (tribe_size - num_chiefs) * 
  (tribe_size - num_chiefs - 1) * 
  (choose (tribe_size - num_chiefs - num_supporting_chiefs) num_inferior_officers_per_chief) *
  (choose (tribe_size - num_chiefs - num_supporting_chiefs - num_inferior_officers_per_chief) num_inferior_officers_per_chief) = 3243240 := by
  sorry

end leadership_combinations_l1224_122472


namespace total_marbles_l1224_122433

theorem total_marbles (x : ℝ) : (4*x + 2) + 2*x + (3*x + 1) = 9*x + 3 := by
  sorry

end total_marbles_l1224_122433


namespace min_product_of_three_numbers_l1224_122494

theorem min_product_of_three_numbers (x y z : ℝ) : 
  x > 0 → y > 0 → z > 0 → 
  x + y + z = 1 → 
  x ≤ 3*y ∧ x ≤ 3*z ∧ y ≤ 3*x ∧ y ≤ 3*z ∧ z ≤ 3*x ∧ z ≤ 3*y →
  x * y * z ≥ 1/36 :=
by sorry

end min_product_of_three_numbers_l1224_122494


namespace valid_pairs_count_l1224_122461

def is_valid_pair (x y : ℕ) : Prop :=
  x ≠ y ∧
  x < 10 ∧ y < 10 ∧
  let product := (x * 1111) * (y * 1111)
  product ≥ 1000000 ∧ product < 10000000 ∧
  product % 10 = x ∧ (product / 1000000) % 10 = x

theorem valid_pairs_count :
  ∃ (S : Finset (ℕ × ℕ)), (∀ p ∈ S, is_valid_pair p.1 p.2) ∧ S.card = 3 :=
sorry

end valid_pairs_count_l1224_122461


namespace arithmetic_geometric_sequence_relation_l1224_122437

theorem arithmetic_geometric_sequence_relation (a : ℕ → ℤ) (b : ℕ → ℝ) (d k m : ℕ) (q : ℝ) :
  (∀ n, a (n + 1) - a n = d) →
  (a k = k^2 + 2) →
  (a (2*k) = (k + 2)^2) →
  (k > 0) →
  (a 1 > 1) →
  (∀ n, b n = q^(n-1)) →
  (q > 0) →
  (∃ m : ℕ, m > 0 ∧ (3 * 2^2) / (3 * m^2) = 1 + q + q^2) →
  q = (Real.sqrt 13 - 1) / 2 :=
by sorry

end arithmetic_geometric_sequence_relation_l1224_122437


namespace quadratic_function_m_value_l1224_122414

/-- A quadratic function of the form y = mx^2 - 8x + m(m-1) that passes through the origin -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 8 * x + m * (m - 1)

/-- The quadratic function passes through the origin -/
def passes_through_origin (m : ℝ) : Prop := quadratic_function m 0 = 0

/-- The theorem stating that m = 1 for the given quadratic function passing through the origin -/
theorem quadratic_function_m_value :
  ∃ m : ℝ, passes_through_origin m ∧ m = 1 :=
sorry

end quadratic_function_m_value_l1224_122414


namespace trigonometric_expression_evaluation_l1224_122448

theorem trigonometric_expression_evaluation : 3 * Real.cos 0 + 4 * Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end trigonometric_expression_evaluation_l1224_122448


namespace base_seven_addition_l1224_122454

/-- Given a base 7 addition problem 5XY₇ + 52₇ = 62X₇, prove that X + Y = 6 in base 10 -/
theorem base_seven_addition (X Y : ℕ) : 
  (5 * 7^2 + X * 7 + Y) + (5 * 7 + 2) = 6 * 7^2 + 2 * 7 + X → X + Y = 6 := by
  sorry

end base_seven_addition_l1224_122454


namespace sufficient_not_necessary_condition_l1224_122409

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a > 1 → (1 / a < 1)) ∧ ¬((1 / a < 1) → (a > 1)) := by
  sorry

end sufficient_not_necessary_condition_l1224_122409


namespace constant_shift_invariance_l1224_122446

variable {n : ℕ}
variable (X Y : Fin n → ℝ)
variable (c : ℝ)

def addConstant (X : Fin n → ℝ) (c : ℝ) : Fin n → ℝ :=
  fun i => X i + c

def sampleStandardDeviation (X : Fin n → ℝ) : ℝ :=
  sorry

def sampleRange (X : Fin n → ℝ) : ℝ :=
  sorry

theorem constant_shift_invariance (hc : c ≠ 0) (hY : Y = addConstant X c) :
  sampleStandardDeviation X = sampleStandardDeviation Y ∧
  sampleRange X = sampleRange Y :=
sorry

end constant_shift_invariance_l1224_122446


namespace smallest_add_subtract_for_perfect_square_l1224_122429

theorem smallest_add_subtract_for_perfect_square (n m : ℕ) : 
  (∀ k : ℕ, k < 470 → ¬∃ i : ℕ, 92555 + k = i^2) ∧
  (∃ i : ℕ, 92555 + 470 = i^2) ∧
  (∀ j : ℕ, j < 139 → ¬∃ i : ℕ, 92555 - j = i^2) ∧
  (∃ i : ℕ, 92555 - 139 = i^2) :=
by sorry

end smallest_add_subtract_for_perfect_square_l1224_122429


namespace apollo_chariot_wheels_l1224_122492

theorem apollo_chariot_wheels (months_in_year : ℕ) (total_cost : ℕ) 
  (initial_rate : ℕ) (h1 : months_in_year = 12) (h2 : total_cost = 54) 
  (h3 : initial_rate = 3) : 
  ∃ (x : ℕ), x * initial_rate + (months_in_year - x) * (2 * initial_rate) = total_cost ∧ x = 6 := by
  sorry

end apollo_chariot_wheels_l1224_122492


namespace unit_circle_point_coordinate_l1224_122404

/-- Theorem: For a point P(x₀, y₀) on the unit circle in the xy-plane, where ∠xOP = α, 
α ∈ (π/4, 3π/4), and cos(α + π/4) = -12/13, the value of x₀ is equal to -7√2/26. -/
theorem unit_circle_point_coordinate (x₀ y₀ α : Real) : 
  x₀^2 + y₀^2 = 1 → -- Point P lies on the unit circle
  x₀ = Real.cos α → -- Definition of cosine
  y₀ = Real.sin α → -- Definition of sine
  π/4 < α → α < 3*π/4 → -- α ∈ (π/4, 3π/4)
  Real.cos (α + π/4) = -12/13 → -- Given condition
  x₀ = -7 * Real.sqrt 2 / 26 := by
sorry

end unit_circle_point_coordinate_l1224_122404


namespace bake_four_pans_l1224_122407

/-- The number of pans of cookies that can be baked in a given time -/
def pans_of_cookies (total_time minutes_per_pan : ℕ) : ℕ :=
  total_time / minutes_per_pan

/-- Proof that 4 pans of cookies can be baked in 28 minutes when each pan takes 7 minutes -/
theorem bake_four_pans : pans_of_cookies 28 7 = 4 := by
  sorry

end bake_four_pans_l1224_122407


namespace remaining_quantities_l1224_122427

theorem remaining_quantities (total : ℕ) (avg_all : ℚ) (subset : ℕ) (avg_subset : ℚ) (avg_remaining : ℚ) :
  total = 5 →
  avg_all = 12 →
  subset = 3 →
  avg_subset = 4 →
  avg_remaining = 24 →
  total - subset = 2 :=
by sorry

end remaining_quantities_l1224_122427


namespace sum_of_triangle_perimeters_sum_of_specific_triangle_perimeters_l1224_122491

/-- The sum of perimeters of an infinite series of equilateral triangles -/
theorem sum_of_triangle_perimeters (initial_perimeter : ℝ) : 
  initial_perimeter > 0 →
  (∑' n, initial_perimeter * (1/2)^n) = 2 * initial_perimeter :=
by
  sorry

/-- The specific case where the initial triangle has a perimeter of 180 cm -/
theorem sum_of_specific_triangle_perimeters : 
  (∑' n, 180 * (1/2)^n) = 360 :=
by
  sorry

end sum_of_triangle_perimeters_sum_of_specific_triangle_perimeters_l1224_122491


namespace tangent_line_y_intercept_l1224_122420

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of the tangent line to two circles -/
def yIntercept (line : TangentLine) : ℝ := sorry

theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (3, 0), radius := 3 }
  let c2 : Circle := { center := (7, 0), radius := 2 }
  let line : TangentLine := {
    circle1 := c1,
    circle2 := c2,
    tangentPoint1 := sorry,  -- Exact point not given, but in first quadrant
    tangentPoint2 := sorry   -- Exact point not given, but in first quadrant
  }
  yIntercept line = 4.5 := by sorry

end tangent_line_y_intercept_l1224_122420


namespace perpendicular_vectors_l1224_122431

/-- Given two vectors a and b in ℝ³, prove that they are perpendicular if and only if x = 10/3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ × ℝ) :
  a = (2, -1, 3) → b = (-4, 2, x) → (a • b = 0 ↔ x = 10/3) :=
by sorry

end perpendicular_vectors_l1224_122431


namespace train_length_l1224_122425

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 90 → time_s = 9 → speed_kmh * (5/18) * time_s = 225 :=
by sorry

end train_length_l1224_122425


namespace parabola_satisfies_conditions_l1224_122464

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  y^2 - 8*x - 8*y + 16 = 0

-- Define the conditions
def passes_through_point (eq : ℝ → ℝ → Prop) : Prop :=
  eq 2 8

def focus_y_coordinate (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ x, eq x 4 ∧ (∀ y, eq x y → (y - 4)^2 ≤ (8 - 4)^2)

def axis_parallel_to_x (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq x y → eq x (-y + 8)

def vertex_on_y_axis (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ y, eq 0 y

def coefficients_are_integers (a b c d e f : ℤ) : Prop :=
  ∀ x y : ℝ, (a:ℝ)*x^2 + (b:ℝ)*x*y + (c:ℝ)*y^2 + (d:ℝ)*x + (e:ℝ)*y + (f:ℝ) = 0 ↔ parabola_equation x y

def c_is_positive (c : ℤ) : Prop :=
  c > 0

def gcd_is_one (a b c d e f : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd (a.natAbs) (b.natAbs)) (c.natAbs)) (d.natAbs)) (e.natAbs)) (f.natAbs) = 1

-- State the theorem
theorem parabola_satisfies_conditions :
  ∃ a b c d e f : ℤ,
    passes_through_point parabola_equation ∧
    focus_y_coordinate parabola_equation ∧
    axis_parallel_to_x parabola_equation ∧
    vertex_on_y_axis parabola_equation ∧
    coefficients_are_integers a b c d e f ∧
    c_is_positive c ∧
    gcd_is_one a b c d e f :=
  sorry

end parabola_satisfies_conditions_l1224_122464


namespace garden_roller_diameter_l1224_122452

/-- The diameter of a garden roller given its length, area covered, and number of revolutions -/
theorem garden_roller_diameter 
  (length : ℝ) 
  (area_covered : ℝ) 
  (revolutions : ℝ) 
  (h1 : length = 3) 
  (h2 : area_covered = 66) 
  (h3 : revolutions = 5) : 
  ∃ (diameter : ℝ), diameter = 1.4 ∧ 
    area_covered = revolutions * 2 * (22/7) * (diameter/2) * length := by
sorry

end garden_roller_diameter_l1224_122452


namespace sum_of_exterior_angles_of_triangle_l1224_122487

theorem sum_of_exterior_angles_of_triangle (α β γ : ℝ) (α' β' γ' : ℝ) : 
  α + β + γ = 180 →
  α + α' = 180 →
  β + β' = 180 →
  γ + γ' = 180 →
  α' + β' + γ' = 360 := by
sorry

end sum_of_exterior_angles_of_triangle_l1224_122487


namespace cereal_serving_size_l1224_122424

/-- Represents the number of cups of cereal in a box -/
def total_cups : ℕ := 18

/-- Represents the number of servings in a box -/
def total_servings : ℕ := 9

/-- Represents the number of cups per serving -/
def cups_per_serving : ℚ := total_cups / total_servings

theorem cereal_serving_size : cups_per_serving = 2 := by
  sorry

end cereal_serving_size_l1224_122424


namespace yoongi_multiplication_l1224_122408

theorem yoongi_multiplication (x : ℝ) : 8 * x = 64 → x = 8 := by
  sorry

end yoongi_multiplication_l1224_122408


namespace min_angles_in_circle_l1224_122469

theorem min_angles_in_circle (n : ℕ) (h : n ≥ 3) : ℕ :=
  let S : ℕ → ℕ := fun n =>
    if n % 2 = 1 then
      (n - 1)^2 / 4
    else
      n^2 / 4 - n / 2
  S n

#check min_angles_in_circle

end min_angles_in_circle_l1224_122469


namespace left_movement_denoted_negative_l1224_122477

/-- Represents the direction of movement -/
inductive Direction
| Left
| Right

/-- Represents a movement with a distance and direction -/
structure Movement where
  distance : ℝ
  direction : Direction

/-- Denotes a movement as a signed real number -/
def denoteMovement (m : Movement) : ℝ :=
  match m.direction with
  | Direction.Right => m.distance
  | Direction.Left => -m.distance

theorem left_movement_denoted_negative (d : ℝ) (h : d > 0) :
  denoteMovement { distance := d, direction := Direction.Right } = d →
  denoteMovement { distance := d, direction := Direction.Left } = -d :=
by
  sorry

#check left_movement_denoted_negative

end left_movement_denoted_negative_l1224_122477


namespace total_weight_is_410_l1224_122467

/-- The number of A4 sheets Jane has -/
def num_a4_sheets : ℕ := 28

/-- The number of A3 sheets Jane has -/
def num_a3_sheets : ℕ := 27

/-- The weight of a single A4 sheet in grams -/
def weight_a4_sheet : ℕ := 5

/-- The weight of a single A3 sheet in grams -/
def weight_a3_sheet : ℕ := 10

/-- The total weight of all drawing papers in grams -/
def total_weight : ℕ := num_a4_sheets * weight_a4_sheet + num_a3_sheets * weight_a3_sheet

theorem total_weight_is_410 : total_weight = 410 := by sorry

end total_weight_is_410_l1224_122467


namespace fallen_piece_theorem_l1224_122468

/-- A function that checks if two numbers have the same digits --/
def same_digits (a b : ℕ) : Prop := sorry

/-- The number of pages in a fallen piece of a book --/
def fallen_piece_pages (first_page last_page : ℕ) : ℕ :=
  last_page - first_page + 1

theorem fallen_piece_theorem :
  ∃ (last_page : ℕ),
    last_page > 328 ∧
    same_digits last_page 328 ∧
    fallen_piece_pages 328 last_page = 496 := by
  sorry

end fallen_piece_theorem_l1224_122468


namespace tan_neg_five_pi_sixths_l1224_122400

theorem tan_neg_five_pi_sixths : Real.tan (-5 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end tan_neg_five_pi_sixths_l1224_122400


namespace star_three_four_l1224_122410

/-- Custom binary operation ※ -/
def star (a b : ℝ) : ℝ := 2 * a + b

/-- Theorem stating that 3※4 = 10 -/
theorem star_three_four : star 3 4 = 10 := by
  sorry

end star_three_four_l1224_122410


namespace last_remaining_number_l1224_122478

def josephus_sequence (n : ℕ) : ℕ → ℕ
| 0 => 1
| m + 1 => 2 * josephus_sequence n m

theorem last_remaining_number :
  ∃ k : ℕ, josephus_sequence 200 k = 128 ∧ josephus_sequence 200 (k + 1) > 200 :=
by sorry

end last_remaining_number_l1224_122478


namespace cube_volume_from_paper_area_l1224_122482

/-- Given a rectangular piece of paper with length 48 inches and width 72 inches
    that covers exactly the surface area of a cube, prove that the volume of the cube
    is 8 cubic feet, where 1 foot is 12 inches. -/
theorem cube_volume_from_paper_area (paper_length : ℝ) (paper_width : ℝ) 
    (inches_per_foot : ℝ) (h1 : paper_length = 48) (h2 : paper_width = 72) 
    (h3 : inches_per_foot = 12) : 
    let paper_area := paper_length * paper_width
    let cube_side_length := Real.sqrt (paper_area / 6)
    let cube_side_length_feet := cube_side_length / inches_per_foot
    cube_side_length_feet ^ 3 = 8 := by
  sorry

end cube_volume_from_paper_area_l1224_122482


namespace smallest_sum_B_plus_b_l1224_122421

/-- Given that B is a digit in base 5 and b is a base greater than 6,
    such that BBB₅ = 44ᵦ, prove that the smallest possible sum of B + b is 8. -/
theorem smallest_sum_B_plus_b : ∃ (B b : ℕ),
  (0 < B) ∧ (B < 5) ∧  -- B is a digit in base 5
  (b > 6) ∧            -- b is a base greater than 6
  (31 * B = 4 * b + 4) ∧  -- BBB₅ = 44ᵦ
  (∀ (B' b' : ℕ), 
    (0 < B') ∧ (B' < 5) ∧ (b' > 6) ∧ (31 * B' = 4 * b' + 4) →
    B + b ≤ B' + b') :=
by sorry

end smallest_sum_B_plus_b_l1224_122421


namespace log_equation_solution_l1224_122413

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_equation_solution (x k : ℝ) :
  log k x * log 3 k = 4 → k = 9 → x = 81 := by
  sorry

end log_equation_solution_l1224_122413


namespace common_difference_proof_l1224_122411

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem common_difference_proof (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h2 : a 2 = 14) (h5 : a 5 = 5) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -3 := by
sorry

end common_difference_proof_l1224_122411


namespace pentagon_obtuse_angles_dodecagon_diagonals_four_sided_angle_sum_equality_l1224_122403

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- Add necessary fields here

/-- The number of obtuse angles in a polygon -/
def numObtuseAngles (p : Polygon n) : ℕ := sorry

/-- The number of diagonals in a polygon -/
def numDiagonals (p : Polygon n) : ℕ := sorry

/-- The sum of interior angles of a polygon -/
def sumInteriorAngles (p : Polygon n) : ℝ := sorry

/-- The sum of exterior angles of a polygon -/
def sumExteriorAngles (p : Polygon n) : ℝ := sorry

theorem pentagon_obtuse_angles :
  ∀ p : Polygon 5, numObtuseAngles p ≥ 2 := by sorry

theorem dodecagon_diagonals :
  ∀ p : Polygon 12, numDiagonals p = 54 := by sorry

theorem four_sided_angle_sum_equality :
  ∀ n : ℕ, ∀ p : Polygon n,
    (sumInteriorAngles p = sumExteriorAngles p) ↔ n = 4 := by sorry

end pentagon_obtuse_angles_dodecagon_diagonals_four_sided_angle_sum_equality_l1224_122403


namespace evaluate_expression_l1224_122473

theorem evaluate_expression (a : ℝ) : 
  let x := a + 9
  (x - a + 6) = 15 := by
sorry

end evaluate_expression_l1224_122473


namespace quadratic_equation_roots_l1224_122439

theorem quadratic_equation_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x - 1 = 0 ↔ x = x₁ ∨ x = x₂) →
  (1/x₁ + 1/x₂ = -3) →
  m = -3 := by
sorry

end quadratic_equation_roots_l1224_122439


namespace line_direction_vector_l1224_122459

def point1 : ℝ × ℝ := (-4, 3)
def point2 : ℝ × ℝ := (2, -2)
def direction_vector (a : ℝ) : ℝ × ℝ := (a, -1)

theorem line_direction_vector (a : ℝ) :
  (∃ k : ℝ, k ≠ 0 ∧ (point2.1 - point1.1, point2.2 - point1.2) = k • direction_vector a) →
  a = 6/5 := by
sorry

end line_direction_vector_l1224_122459


namespace tangent_line_and_inequality_l1224_122499

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_and_inequality (a : ℝ) :
  (∃ x₀ : ℝ, (∀ x : ℝ, f a x ≥ x - 1) ∧ (f a x₀ = x₀ - 1)) →
  (a = 1) ∧
  (∀ x : ℝ, 1 < x → x < 2 → (1 / Real.log x) - (1 / Real.log (x - 1)) < 1 / ((x - 1) * (2 - x))) :=
by sorry

end tangent_line_and_inequality_l1224_122499


namespace orange_trees_l1224_122485

theorem orange_trees (total_fruits : ℕ) (fruits_per_tree : ℕ) (remaining_ratio : ℚ) : 
  total_fruits = 960 →
  fruits_per_tree = 200 →
  remaining_ratio = 3/5 →
  (total_fruits : ℚ) / (remaining_ratio * fruits_per_tree) = 8 :=
by sorry

end orange_trees_l1224_122485


namespace range_of_a_l1224_122458

theorem range_of_a (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 4 = 4*x*y) :
  (∀ a : ℝ, x*y + 1/2*a^2*x + a^2*y + a - 17 ≥ 0) ↔ 
  (∀ a : ℝ, a ≤ -3 ∨ a ≥ 5/2) :=
sorry

end range_of_a_l1224_122458


namespace min_triangles_to_cover_l1224_122443

theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) : 
  small_side = 1 →
  large_side = 15 →
  (large_side / small_side) ^ 2 = 225 := by
  sorry

end min_triangles_to_cover_l1224_122443


namespace parabola_directrix_l1224_122423

/-- Represents a parabola in the form y = ax^2 -/
structure Parabola where
  a : ℝ
  equation : ℝ → ℝ := fun x => a * x^2

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => ∃ k, y = -k ∧ p.a = 1 / (4 * k)

theorem parabola_directrix (p : Parabola) (h : p.a = 1/4) :
  directrix p = fun y => y = -1 := by sorry

end parabola_directrix_l1224_122423


namespace collinear_points_x_value_l1224_122416

/-- Given three points in a 2D plane, checks if they are collinear --/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₁) = (y₃ - y₁) * (x₂ - x₁)

/-- Theorem: If points A(1, 1), B(-4, 5), and C(x, 13) are collinear, then x = -14 --/
theorem collinear_points_x_value :
  ∀ x : ℝ, collinear 1 1 (-4) 5 x 13 → x = -14 := by
  sorry

end collinear_points_x_value_l1224_122416


namespace all_roots_integer_iff_a_eq_50_l1224_122438

/-- The polynomial P(x) = x^3 - 2x^2 - 25x + a -/
def P (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*x^2 - 25*x + a

/-- A function that checks if a real number is an integer -/
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n

/-- The theorem stating that all roots of P(x) are integers iff a = 50 -/
theorem all_roots_integer_iff_a_eq_50 (a : ℝ) :
  (∀ x : ℝ, P a x = 0 → isInteger x) ↔ a = 50 := by sorry

end all_roots_integer_iff_a_eq_50_l1224_122438


namespace necessary_to_sufficient_contrapositive_l1224_122476

theorem necessary_to_sufficient_contrapositive (p q : Prop) :
  (q → p) → (¬p → ¬q) :=
by
  sorry

end necessary_to_sufficient_contrapositive_l1224_122476


namespace original_speed_theorem_l1224_122428

def distance : ℝ := 160
def speed_increase : ℝ := 0.25
def time_saved : ℝ := 0.4

theorem original_speed_theorem (original_speed : ℝ) 
  (h1 : original_speed > 0) 
  (h2 : distance / original_speed - distance / (original_speed * (1 + speed_increase)) = time_saved) : 
  original_speed = 80 := by
sorry

end original_speed_theorem_l1224_122428


namespace algebraic_identity_l1224_122498

theorem algebraic_identity (a b : ℝ) : 3 * a^2 * b - 2 * b * a^2 = a^2 * b := by
  sorry

end algebraic_identity_l1224_122498


namespace expression_evaluation_l1224_122447

theorem expression_evaluation :
  let x : ℚ := 2
  let y : ℚ := -1/4
  2 * (x - 2*y) - 1/3 * (3*x - 6*y) + 2*x = 13/2 := by
  sorry

end expression_evaluation_l1224_122447
