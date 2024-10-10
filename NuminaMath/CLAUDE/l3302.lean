import Mathlib

namespace ice_cream_flavors_l3302_330252

theorem ice_cream_flavors (n : ℕ) (k : ℕ) : 
  n = 4 → k = 5 → (n + k - 1).choose k = 56 := by
  sorry

end ice_cream_flavors_l3302_330252


namespace first_week_cases_l3302_330281

/-- Given the number of coronavirus cases in New York over three weeks,
    prove that the number of cases in the first week was 3750. -/
theorem first_week_cases (first_week : ℕ) : 
  (first_week + first_week / 2 + (first_week / 2 + 2000) = 9500) → 
  first_week = 3750 := by
  sorry

end first_week_cases_l3302_330281


namespace binomial_expansion_coefficients_l3302_330214

theorem binomial_expansion_coefficients :
  let n : ℕ := 50
  let a : ℕ := 2
  -- Coefficient of x^3
  (n.choose 3) * a^(n - 3) = 19600 * 2^47 ∧
  -- Constant term
  (n.choose 0) * a^n = 2^50 :=
by sorry

end binomial_expansion_coefficients_l3302_330214


namespace no_solution_for_equation_l3302_330292

theorem no_solution_for_equation :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / (a + b)) := by
  sorry

end no_solution_for_equation_l3302_330292


namespace triples_satisfying_equation_l3302_330286

theorem triples_satisfying_equation : 
  ∀ (a b p : ℕ), 
    0 < a ∧ 0 < b ∧ 0 < p ∧ 
    Nat.Prime p ∧
    a^p - b^p = 2013 →
    ((a = 337 ∧ b = 334 ∧ p = 2) ∨ 
     (a = 97 ∧ b = 86 ∧ p = 2) ∨ 
     (a = 47 ∧ b = 14 ∧ p = 2)) :=
by sorry

end triples_satisfying_equation_l3302_330286


namespace max_volume_container_l3302_330268

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  shortSide : ℝ
  longSide : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : ℝ :=
  d.shortSide * d.longSide * d.height

/-- Represents the constraints of the problem --/
def isValidContainer (d : ContainerDimensions) : Prop :=
  d.longSide = d.shortSide + 0.5 ∧
  2 * (d.shortSide + d.longSide) + 4 * d.height = 14.8 ∧
  d.shortSide > 0 ∧ d.longSide > 0 ∧ d.height > 0

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    isValidContainer d ∧
    volume d = 1.8 ∧
    d.height = 1.2 ∧
    ∀ (d' : ContainerDimensions), isValidContainer d' → volume d' ≤ volume d :=
by sorry

end max_volume_container_l3302_330268


namespace females_with_advanced_degrees_only_l3302_330249

theorem females_with_advanced_degrees_only (total_employees : ℕ) 
  (female_employees : ℕ) (employees_with_advanced_degrees : ℕ)
  (employees_with_college_only : ℕ) (employees_with_multiple_degrees : ℕ)
  (males_with_college_only : ℕ) (males_with_multiple_degrees : ℕ)
  (females_with_multiple_degrees : ℕ)
  (h1 : total_employees = 148)
  (h2 : female_employees = 92)
  (h3 : employees_with_advanced_degrees = 78)
  (h4 : employees_with_college_only = 55)
  (h5 : employees_with_multiple_degrees = 15)
  (h6 : males_with_college_only = 31)
  (h7 : males_with_multiple_degrees = 8)
  (h8 : females_with_multiple_degrees = 10) :
  total_employees - female_employees - males_with_college_only - males_with_multiple_degrees +
  employees_with_advanced_degrees - females_with_multiple_degrees - males_with_multiple_degrees = 35 :=
by sorry

end females_with_advanced_degrees_only_l3302_330249


namespace system_solution_l3302_330258

theorem system_solution (a b : ℚ) : 
  (a/3 - 1) + 2*(b/5 + 2) = 4 ∧ 
  2*(a/3 - 1) + (b/5 + 2) = 5 → 
  a = 9 ∧ b = -5 := by
sorry

end system_solution_l3302_330258


namespace binomial_expectation_and_variance_l3302_330293

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_expectation_and_variance :
  ∀ ξ : BinomialRV, ξ.n = 10 ∧ ξ.p = 0.6 → 
  expected_value ξ = 6 ∧ variance ξ = 2.4 := by
  sorry

end binomial_expectation_and_variance_l3302_330293


namespace quadratic_inequality_solution_set_l3302_330261

/-- Given that the solution set of ax^2 + 5x + b > 0 is {x | 2 < x < 3},
    prove that the solution set of bx^2 - 5x + a < 0 is {x | x < -1/2 or x > -1/3} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | a * x^2 + 5 * x + b > 0}) :
  {x : ℝ | b * x^2 - 5 * x + a < 0} = {x : ℝ | x < -1/2 ∨ x > -1/3} :=
by sorry

end quadratic_inequality_solution_set_l3302_330261


namespace projection_of_b_onto_a_l3302_330241

def a : Fin 3 → ℝ := ![2, -1, 2]
def b : Fin 3 → ℝ := ![1, -2, 1]

theorem projection_of_b_onto_a :
  let proj := (a • b) / (a • a) • a
  proj 0 = 4/3 ∧ proj 1 = -2/3 ∧ proj 2 = 4/3 := by
  sorry

end projection_of_b_onto_a_l3302_330241


namespace unicycle_count_l3302_330296

theorem unicycle_count :
  ∀ (num_bicycles num_tricycles num_unicycles : ℕ),
    num_bicycles = 3 →
    num_tricycles = 4 →
    num_bicycles * 2 + num_tricycles * 3 + num_unicycles * 1 = 25 →
    num_unicycles = 7 := by
  sorry

end unicycle_count_l3302_330296


namespace certain_number_is_eleven_l3302_330244

theorem certain_number_is_eleven : ∃ x : ℕ, 
  x + (3 * 13 + 3 * 14 + 3 * 17) = 143 ∧ x = 11 := by
  sorry

end certain_number_is_eleven_l3302_330244


namespace triangle_side_calculation_l3302_330264

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a = 2, c = 2√3, and C = π/3, then b = 4 -/
theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  (a = 2) → (c = 2 * Real.sqrt 3) → (C = π / 3) → (b = 4) := by
  sorry

end triangle_side_calculation_l3302_330264


namespace max_leftover_oranges_l3302_330295

theorem max_leftover_oranges (n : ℕ) : ∃ (q : ℕ), n = 8 * q + (n % 8) ∧ n % 8 ≤ 7 := by
  sorry

end max_leftover_oranges_l3302_330295


namespace b_cubed_is_zero_l3302_330208

theorem b_cubed_is_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 3 = 0 := by
  sorry

end b_cubed_is_zero_l3302_330208


namespace class_test_problem_l3302_330206

theorem class_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.7)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.65 := by
  sorry

end class_test_problem_l3302_330206


namespace sampling_probabilities_equal_l3302_330262

/-- Represents the composition of a batch of components -/
structure BatchComposition where
  total : ℕ
  first_class : ℕ
  second_class : ℕ
  third_class : ℕ
  unqualified : ℕ

/-- Represents the probabilities of selecting an individual component using different sampling methods -/
structure SamplingProbabilities where
  simple_random : ℚ
  stratified : ℚ
  systematic : ℚ

/-- Theorem stating that all sampling probabilities are equal to 1/8 for the given batch composition and sample size -/
theorem sampling_probabilities_equal (batch : BatchComposition) (sample_size : ℕ) 
  (h1 : batch.total = 160)
  (h2 : batch.first_class = 48)
  (h3 : batch.second_class = 64)
  (h4 : batch.third_class = 32)
  (h5 : batch.unqualified = 16)
  (h6 : sample_size = 20)
  (h7 : batch.total = batch.first_class + batch.second_class + batch.third_class + batch.unqualified) :
  ∃ (probs : SamplingProbabilities), 
    probs.simple_random = 1/8 ∧ 
    probs.stratified = 1/8 ∧ 
    probs.systematic = 1/8 := by
  sorry


end sampling_probabilities_equal_l3302_330262


namespace divisibility_condition_l3302_330230

theorem divisibility_condition (n : ℕ) : 
  n > 0 ∧ (n - 1) ∣ (n^3 + 4) ↔ n = 2 ∨ n = 6 := by
  sorry

end divisibility_condition_l3302_330230


namespace first_year_after_2020_with_sum_of_digits_10_l3302_330226

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2020WithSumOfDigits10 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 10 ∧ 
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2020_with_sum_of_digits_10 :
  isFirstYearAfter2020WithSumOfDigits10 2026 := by
  sorry

#eval sumOfDigits 2026  -- Should output 10

end first_year_after_2020_with_sum_of_digits_10_l3302_330226


namespace range_when_a_is_one_a_values_for_all_x_geq_one_l3302_330209

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

-- Theorem for part I
theorem range_when_a_is_one :
  Set.range (fun x => f x 1) = Set.Ici 5 := by sorry

-- Theorem for part II
theorem a_values_for_all_x_geq_one :
  {a : ℝ | ∀ x, f x a ≥ 1} = Set.Iic (-5) ∪ Set.Ici (-3) := by sorry

end range_when_a_is_one_a_values_for_all_x_geq_one_l3302_330209


namespace balloon_rearrangements_eq_36_l3302_330221

/-- The number of distinguishable rearrangements of "BALLOON" with specific conditions -/
def balloon_rearrangements : ℕ :=
  let vowels := ['A', 'O', 'O']
  let consonants := ['B', 'L', 'L', 'N']
  let consonant_arrangements := Nat.factorial 4 / Nat.factorial 2
  let vowel_arrangements := Nat.factorial 3 / Nat.factorial 2
  consonant_arrangements * vowel_arrangements

/-- Theorem stating that the number of rearrangements is 36 -/
theorem balloon_rearrangements_eq_36 : balloon_rearrangements = 36 := by
  sorry

end balloon_rearrangements_eq_36_l3302_330221


namespace total_pears_l3302_330294

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17

theorem total_pears : alyssa_pears + nancy_pears = 59 := by
  sorry

end total_pears_l3302_330294


namespace tank_capacity_l3302_330265

theorem tank_capacity (initial_fraction : Rat) (added_amount : Rat) (final_fraction : Rat) :
  initial_fraction = 3/4 →
  added_amount = 9 →
  final_fraction = 9/10 →
  ∃ (capacity : Rat), capacity = 60 ∧
    final_fraction * capacity - initial_fraction * capacity = added_amount :=
by sorry

end tank_capacity_l3302_330265


namespace candy_calories_per_serving_l3302_330273

/-- Calculates the number of calories per serving in a package of candy. -/
def calories_per_serving (total_servings : ℕ) (half_package_calories : ℕ) : ℕ :=
  (2 * half_package_calories) / total_servings

/-- Proves that the number of calories per serving is 120, given the problem conditions. -/
theorem candy_calories_per_serving :
  calories_per_serving 3 180 = 120 := by
  sorry

end candy_calories_per_serving_l3302_330273


namespace water_depth_conversion_l3302_330242

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : Real
  baseDiameter : Real

/-- Calculates the volume of water in the tank when horizontal -/
def horizontalWaterVolume (tank : WaterTank) (depth : Real) : Real :=
  sorry

/-- Calculates the depth of water when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) (horizontalDepth : Real) : Real :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_conversion (tank : WaterTank) (horizontalDepth : Real) :
  tank.height = 10 ∧ tank.baseDiameter = 6 ∧ horizontalDepth = 4 →
  verticalWaterDepth tank horizontalDepth = 4.5 := by
  sorry

end water_depth_conversion_l3302_330242


namespace probability_heart_spade_queen_value_l3302_330282

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (valid : cards.card = 52)

/-- Represents a suit in a deck of cards -/
inductive Suit
| Hearts | Spades | Diamonds | Clubs

/-- Represents a rank in a deck of cards -/
inductive Rank
| Two | Three | Four | Five | Six | Seven | Eight | Nine | Ten | Jack | Queen | King | Ace

/-- A function to check if a card is a heart -/
def is_heart (card : Nat × Nat) : Prop := sorry

/-- A function to check if a card is a spade -/
def is_spade (card : Nat × Nat) : Prop := sorry

/-- A function to check if a card is a queen -/
def is_queen (card : Nat × Nat) : Prop := sorry

/-- The probability of drawing a heart first, a spade second, and a queen third -/
def probability_heart_spade_queen (d : Deck) : ℚ := sorry

/-- Theorem stating the probability of drawing a heart first, a spade second, and a queen third -/
theorem probability_heart_spade_queen_value (d : Deck) : 
  probability_heart_spade_queen d = 221 / 44200 := by sorry

end probability_heart_spade_queen_value_l3302_330282


namespace rainfall_rate_calculation_l3302_330210

/-- Proves that the rainfall rate is 5 cm/hour given the specified conditions -/
theorem rainfall_rate_calculation (depth : ℝ) (area : ℝ) (time : ℝ) 
  (h_depth : depth = 15)
  (h_area : area = 300)
  (h_time : time = 3) :
  (depth * area) / (time * area) = 5 := by
sorry

end rainfall_rate_calculation_l3302_330210


namespace polynomial_coefficient_equality_l3302_330245

theorem polynomial_coefficient_equality (a b c : ℚ) : 
  (∀ x, (7*x^2 - 5*x + 9/4)*(a*x^2 + b*x + c) = 21*x^4 - 24*x^3 + 28*x^2 - 37/4*x + 21/4) →
  (a = 3 ∧ b = -9/7) := by
sorry

end polynomial_coefficient_equality_l3302_330245


namespace estimate_N_l3302_330283

-- Define f(n) as the largest prime factor of n
def f (n : ℕ) : ℕ := sorry

-- Define the sum of f(n^2-1) for n from 2 to 10^6
def sum_f_nsquared_minus_one : ℕ := sorry

-- Define the sum of f(n) for n from 2 to 10^6
def sum_f_n : ℕ := sorry

-- Theorem statement
theorem estimate_N : 
  ⌊(10^4 : ℝ) * (sum_f_nsquared_minus_one : ℝ) / (sum_f_n : ℝ)⌋ = 18215 := by sorry

end estimate_N_l3302_330283


namespace geometric_sequence_sixth_term_l3302_330239

theorem geometric_sequence_sixth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^7 = 2) : 
  a * r^5 = 16 := by
  sorry

end geometric_sequence_sixth_term_l3302_330239


namespace system_solution_l3302_330267

theorem system_solution :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (2*x - Real.sqrt (x*y) - 4*Real.sqrt (x/y) + 2 = 0 ∧
   2*x^2 + x^2*y^4 = 18*y^2) →
  ((x = 2 ∧ y = 2) ∨
   (x = (Real.sqrt (Real.sqrt 286))/4 ∧ y = Real.sqrt (Real.sqrt 286))) :=
by sorry

end system_solution_l3302_330267


namespace sum_equals_square_l3302_330237

theorem sum_equals_square (k : ℕ) (N : ℕ) : N < 100 →
  (k * (k + 1)) / 2 = N^2 ↔ k = 1 ∨ k = 8 ∨ k = 49 := by sorry

end sum_equals_square_l3302_330237


namespace trigonometric_sum_equality_l3302_330289

theorem trigonometric_sum_equality : 
  Real.cos (π / 3) + Real.sin (π / 3) - Real.sqrt (3 / 4) + (Real.tan (π / 4))⁻¹ = 3 / 2 := by
  sorry

end trigonometric_sum_equality_l3302_330289


namespace parallel_lines_min_value_l3302_330298

theorem parallel_lines_min_value (m n : ℕ+) : 
  (∀ x y : ℝ, x + (n.val - 1) * y - 2 = 0 ↔ m.val * x + y + 3 = 0) →
  (∀ k : ℕ+, 2 * m.val + n.val ≤ k.val → k.val = 11) :=
sorry

end parallel_lines_min_value_l3302_330298


namespace opposite_of_negative_three_l3302_330299

theorem opposite_of_negative_three :
  -((-3 : ℤ)) = 3 :=
by sorry

end opposite_of_negative_three_l3302_330299


namespace ted_blue_mushrooms_l3302_330279

theorem ted_blue_mushrooms :
  let bill_red : ℕ := 12
  let bill_brown : ℕ := 6
  let ted_green : ℕ := 14
  let ted_blue : ℕ := x
  let white_spotted_total : ℕ := 17
  let white_spotted_bill_red : ℕ := bill_red * 2 / 3
  let white_spotted_bill_brown : ℕ := bill_brown
  let white_spotted_ted_blue : ℕ := ted_blue / 2
  white_spotted_total = white_spotted_bill_red + white_spotted_bill_brown + white_spotted_ted_blue →
  ted_blue = 10 := by
sorry

end ted_blue_mushrooms_l3302_330279


namespace cost_price_calculation_l3302_330223

/-- Proves that the cost price of an article is 350, given the selling price and profit percentage. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 455 → profit_percentage = 30 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 350 := by
  sorry

end cost_price_calculation_l3302_330223


namespace expression_evaluation_l3302_330246

/-- Evaluates the expression (3x^3 - 7x^2 + 4x - 9) / (2x - 0.5) for x = 100 -/
theorem expression_evaluation :
  let x : ℝ := 100
  let numerator := 3 * x^3 - 7 * x^2 + 4 * x - 9
  let denominator := 2 * x - 0.5
  abs ((numerator / denominator) - 14684.73534) < 0.00001 := by
  sorry

end expression_evaluation_l3302_330246


namespace negative_square_times_a_l3302_330284

theorem negative_square_times_a (a : ℝ) : -a^2 * a = -a^3 := by
  sorry

end negative_square_times_a_l3302_330284


namespace circle_radius_calculation_l3302_330256

-- Define the circles and triangle
def circleA : ℝ := 13  -- radius of circle A
def circleB : ℝ := 4   -- radius of circle B
def circleC : ℝ := 3   -- radius of circle C

-- Define the theorem
theorem circle_radius_calculation (r : ℝ) : 
  -- Right triangle T inscribed in circle A
  -- Circle B internally tangent to A at one vertex of T
  -- Circle C internally tangent to A at another vertex of T
  -- Circles B and C externally tangent to circle E with radius r
  -- Angle between radii of A touching vertices related to B and C is 90°
  r = (Real.sqrt 181 - 7) / 2 := by
  sorry

end circle_radius_calculation_l3302_330256


namespace sector_central_angle_l3302_330288

/-- Given a sector with radius 10 and area 50π/3, its central angle is π/3. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (h1 : r = 10) (h2 : S = 50 * Real.pi / 3) :
  S = 1/2 * r^2 * (Real.pi/3) := by
  sorry

end sector_central_angle_l3302_330288


namespace complex_norm_squared_l3302_330274

theorem complex_norm_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a (-b)
  Complex.normSq z = a^2 + b^2 := by sorry

end complex_norm_squared_l3302_330274


namespace largest_lcm_with_15_l3302_330229

theorem largest_lcm_with_15 : 
  let lcm_list := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 6, Nat.lcm 15 9, Nat.lcm 15 10, Nat.lcm 15 15]
  List.maximum lcm_list = some 45 := by
sorry

end largest_lcm_with_15_l3302_330229


namespace stratified_sample_for_model_a_l3302_330266

/-- Calculates the number of items to be selected in a stratified sample -/
def stratified_sample_size (model_volume : ℕ) (total_volume : ℕ) (total_sample : ℕ) : ℕ :=
  (model_volume * total_sample) / total_volume

theorem stratified_sample_for_model_a 
  (volume_a volume_b volume_c total_sample : ℕ) 
  (h_positive : volume_a > 0 ∧ volume_b > 0 ∧ volume_c > 0 ∧ total_sample > 0) :
  stratified_sample_size volume_a (volume_a + volume_b + volume_c) total_sample = 
    (volume_a * total_sample) / (volume_a + volume_b + volume_c) :=
by
  sorry

#eval stratified_sample_size 1200 9200 46

end stratified_sample_for_model_a_l3302_330266


namespace classroom_students_l3302_330236

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of dozens of pencils each student gets -/
def dozens_per_student : ℕ := 4

/-- The total number of pencils to be given out -/
def total_pencils : ℕ := 2208

/-- The number of students in the classroom -/
def num_students : ℕ := total_pencils / (dozens_per_student * pencils_per_dozen)

theorem classroom_students :
  num_students = 46 := by sorry

end classroom_students_l3302_330236


namespace strawberry_weight_difference_l3302_330220

/-- The weight difference between Marco's and his dad's strawberries -/
def weight_difference (marco_weight : ℕ) (total_weight : ℕ) : ℕ :=
  marco_weight - (total_weight - marco_weight)

/-- Theorem stating the weight difference given the problem conditions -/
theorem strawberry_weight_difference :
  weight_difference 30 47 = 13 := by
  sorry

end strawberry_weight_difference_l3302_330220


namespace greatest_common_divisor_450_90_under_60_l3302_330257

theorem greatest_common_divisor_450_90_under_60 : 
  ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 450 ∧ 
  n < 60 ∧ 
  n ∣ 90 ∧ 
  ∀ (m : ℕ), m ∣ 450 ∧ m < 60 ∧ m ∣ 90 → m ≤ n := by
  sorry

end greatest_common_divisor_450_90_under_60_l3302_330257


namespace inequality_relationship_l3302_330233

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def monotone_decreasing_on_unit_interval (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x < y → y ≤ 1 → f y < f x

theorem inequality_relationship (f : ℝ → ℝ) 
  (h1 : is_even f) 
  (h2 : has_period_two f) 
  (h3 : monotone_decreasing_on_unit_interval f) : 
  f (-1) < f 2.5 ∧ f 2.5 < f 0 := by
  sorry

end inequality_relationship_l3302_330233


namespace arithmetic_sequence_property_l3302_330240

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) 
    (eq : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 := by
  sorry

end arithmetic_sequence_property_l3302_330240


namespace product_profit_properties_l3302_330207

/-- A product with given cost and sales characteristics -/
structure Product where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  sales_increase : ℝ

/-- Daily profit as a function of price decrease -/
def daily_profit (p : Product) (x : ℝ) : ℝ :=
  (p.initial_price - x - p.cost_price) * (p.initial_sales + p.sales_increase * x)

/-- Theorem stating the properties of the product and its profit function -/
theorem product_profit_properties (p : Product) 
  (h_cost : p.cost_price = 3.5)
  (h_initial_price : p.initial_price = 14.5)
  (h_initial_sales : p.initial_sales = 500)
  (h_sales_increase : p.sales_increase = 100) :
  (∀ x, 0 ≤ x ∧ x ≤ 11 → daily_profit p x = -100 * (x - 3)^2 + 6400) ∧
  (∃ max_profit, max_profit = 6400 ∧ 
    ∀ x, 0 ≤ x ∧ x ≤ 11 → daily_profit p x ≤ max_profit) ∧
  (∃ optimal_price, optimal_price = 11.5 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 11 → 
      daily_profit p ((p.initial_price - optimal_price) : ℝ) ≥ daily_profit p x) :=
sorry

end product_profit_properties_l3302_330207


namespace no_solution_system_l3302_330260

theorem no_solution_system :
  ¬ ∃ x : ℝ, x > 2 ∧ x < 2 := by
  sorry

end no_solution_system_l3302_330260


namespace sufficient_not_necessary_condition_l3302_330227

theorem sufficient_not_necessary_condition (x : ℝ) :
  (|x - 1/2| < 1/2 → x < 1) ∧ ¬(x < 1 → |x - 1/2| < 1/2) := by
  sorry

end sufficient_not_necessary_condition_l3302_330227


namespace arithmetic_sequence_middle_term_l3302_330290

/-- 
Given an arithmetic sequence with three terms where the first term is 3² and the third term is 3⁴,
prove that the second term (z) is equal to 45.
-/
theorem arithmetic_sequence_middle_term : 
  ∀ (a : ℕ → ℤ), 
    (∀ k, a (k + 1) - a k = a (k + 2) - a (k + 1)) →  -- arithmetic sequence condition
    a 0 = 3^2 →                                       -- first term is 3²
    a 2 = 3^4 →                                       -- third term is 3⁴
    a 1 = 45 :=                                       -- second term (z) is 45
by sorry

end arithmetic_sequence_middle_term_l3302_330290


namespace original_denominator_problem_l3302_330224

theorem original_denominator_problem (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 1 / 3 →
  d = 23 :=
by
  sorry

end original_denominator_problem_l3302_330224


namespace square_roots_of_nine_l3302_330278

theorem square_roots_of_nine :
  {x : ℝ | x ^ 2 = 9} = {3, -3} := by sorry

end square_roots_of_nine_l3302_330278


namespace diophantine_equation_solutions_l3302_330297

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, 7^x - 3 * 2^y = 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) :=
by sorry

end diophantine_equation_solutions_l3302_330297


namespace no_zero_points_implies_a_leq_two_l3302_330277

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (x - 1) - 2 * Real.log x

theorem no_zero_points_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f a x ≠ 0) →
  a ≤ 2 := by sorry

end no_zero_points_implies_a_leq_two_l3302_330277


namespace arithmetic_sequence_common_difference_range_l3302_330253

/-- An arithmetic sequence with first term -5 and positive terms starting from the 10th term
    has a common difference d in the range (5/9, 5/8] -/
theorem arithmetic_sequence_common_difference_range (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a n = -5 + (n - 1) * d) →  -- Definition of arithmetic sequence
  (a 1 = -5) →                         -- First term is -5
  (∀ n ≥ 10, a n > 0) →                -- Terms from 10th onwards are positive
  5/9 < d ∧ d ≤ 5/8 :=                 -- Range of common difference
by sorry

end arithmetic_sequence_common_difference_range_l3302_330253


namespace consecutive_odd_numbers_l3302_330276

theorem consecutive_odd_numbers (a b c d e : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →              -- b is the next odd number after a
  c = a + 4 →              -- c is the third odd number
  d = a + 6 →              -- d is the fourth odd number
  e = a + 8 →              -- e is the fifth odd number
  a + c = 146 →            -- sum of a and c is 146
  e = 79 := by             -- prove that e equals 79
sorry

end consecutive_odd_numbers_l3302_330276


namespace max_three_digit_divisible_by_15_existence_of_solution_l3302_330211

def is_valid_assignment (n a b c d e : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 1 ∧ b ≤ 9 ∧ c ≥ 1 ∧ c ≤ 9 ∧ d ≥ 1 ∧ d ≤ 9 ∧ e ≥ 1 ∧ e ≤ 9 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  n / (a * b + c + d * e) = 15

theorem max_three_digit_divisible_by_15 :
  ∀ n a b c d e : ℕ,
    is_valid_assignment n a b c d e →
    n ≤ 975 :=
by sorry

theorem existence_of_solution :
  ∃ n a b c d e : ℕ,
    is_valid_assignment n a b c d e ∧
    n = 975 :=
by sorry

end max_three_digit_divisible_by_15_existence_of_solution_l3302_330211


namespace angelinas_speed_l3302_330234

theorem angelinas_speed (v : ℝ) 
  (home_to_grocery : 840 / v = 510 / (1.5 * v) + 40)
  (grocery_to_library : 510 / (1.5 * v) = 480 / (2 * v) + 20) :
  2 * v = 25 := by
  sorry

end angelinas_speed_l3302_330234


namespace boat_speed_difference_l3302_330285

/-- Proves that the difference between boat speed and current speed in a channel is 1 km/h -/
theorem boat_speed_difference (V : ℝ) : ∃ (U : ℝ),
  (1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) ∧ (U - V = 1) :=
by
  sorry

#check boat_speed_difference

end boat_speed_difference_l3302_330285


namespace find_divisor_l3302_330217

theorem find_divisor (dividend quotient remainder divisor : ℕ) : 
  dividend = 14698 →
  quotient = 89 →
  remainder = 14 →
  dividend = divisor * quotient + remainder →
  divisor = 165 := by
sorry

end find_divisor_l3302_330217


namespace canvas_cost_decrease_canvas_cost_decrease_is_40_l3302_330225

theorem canvas_cost_decrease (paint_decrease : Real) (total_decrease : Real) 
  (paint_canvas_ratio : Real) (canvas_decrease : Real) : Real :=
  if paint_decrease = 60 ∧ 
     total_decrease = 55.99999999999999 ∧ 
     paint_canvas_ratio = 4 ∧ 
     ((1 - paint_decrease / 100) * paint_canvas_ratio + (1 - canvas_decrease / 100)) / 
     (paint_canvas_ratio + 1) = 1 - total_decrease / 100
  then canvas_decrease
  else 0

#check canvas_cost_decrease

theorem canvas_cost_decrease_is_40 :
  canvas_cost_decrease 60 55.99999999999999 4 40 = 40 := by
  sorry

end canvas_cost_decrease_canvas_cost_decrease_is_40_l3302_330225


namespace four_digit_divisible_by_36_l3302_330213

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ ∃ x y : ℕ, x < 10 ∧ y < 10 ∧ n = x * 1000 + 410 + y

theorem four_digit_divisible_by_36 :
  ∀ n : ℕ, is_valid_number n ∧ n % 36 = 0 ↔ n = 2412 ∨ n = 7416 := by sorry

end four_digit_divisible_by_36_l3302_330213


namespace quadratic_equation_solution_l3302_330200

theorem quadratic_equation_solution (m : ℝ) : 
  (∀ x, (m - 1) * x^2 + 5 * x + m^2 - 3 * m + 2 = 0) → 
  m^2 - 3 * m + 2 = 0 → 
  m - 1 ≠ 0 → 
  m = 2 := by
  sorry

end quadratic_equation_solution_l3302_330200


namespace boys_who_quit_l3302_330287

theorem boys_who_quit (initial_girls : ℕ) (initial_boys : ℕ) (girls_joined : ℕ) (final_total : ℕ) : 
  initial_girls = 18 → 
  initial_boys = 15 → 
  girls_joined = 7 → 
  final_total = 36 → 
  initial_boys - (final_total - (initial_girls + girls_joined)) = 4 := by
sorry

end boys_who_quit_l3302_330287


namespace correct_operation_l3302_330248

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - 3 * a^2 * b = -a^2 * b := by
  sorry

end correct_operation_l3302_330248


namespace curve_symmetry_about_origin_l3302_330269

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop := 3 * x^2 - 8 * x * y + 2 * y^2 = 0

-- Theorem stating the symmetry about the origin
theorem curve_symmetry_about_origin :
  ∀ (x y : ℝ), curve_equation x y ↔ curve_equation (-x) (-y) :=
by sorry

end curve_symmetry_about_origin_l3302_330269


namespace expression_value_l3302_330250

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : m = 2 ∨ m = -2) : 
  (2 * a + 2 * b) / 3 - 5 * c * d + 8 * m = 11 ∨ 
  (2 * a + 2 * b) / 3 - 5 * c * d + 8 * m = -21 := by
  sorry

end expression_value_l3302_330250


namespace division_problem_l3302_330222

theorem division_problem (d : ℕ) : d > 0 ∧ 23 = d * 7 + 2 → d = 3 := by
  sorry

end division_problem_l3302_330222


namespace certain_number_problem_l3302_330215

theorem certain_number_problem : ∃ x : ℕ, 
  220025 = (x + 445) * (2 * (x - 445)) + 25 ∧ 
  x = 555 := by
sorry

end certain_number_problem_l3302_330215


namespace ellipse_semi_minor_axis_l3302_330201

/-- Given an ellipse with specified center, focus, and endpoint of semi-major axis,
    prove that its semi-minor axis has length 2√3. -/
theorem ellipse_semi_minor_axis 
  (center : ℝ × ℝ)
  (focus : ℝ × ℝ)
  (semi_major_endpoint : ℝ × ℝ)
  (h_center : center = (2, -1))
  (h_focus : focus = (2, -3))
  (h_semi_major_endpoint : semi_major_endpoint = (2, 3)) :
  let c := Real.sqrt ((center.1 - focus.1)^2 + (center.2 - focus.2)^2)
  let a := Real.sqrt ((center.1 - semi_major_endpoint.1)^2 + (center.2 - semi_major_endpoint.2)^2)
  let b := Real.sqrt (a^2 - c^2)
  b = 2 * Real.sqrt 3 := by
  sorry

end ellipse_semi_minor_axis_l3302_330201


namespace tileIV_in_rectangle_C_l3302_330275

-- Define the structure for a tile
structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

-- Define the tiles
def tileI : Tile := ⟨1, 2, 5, 6⟩
def tileII : Tile := ⟨6, 3, 1, 5⟩
def tileIII : Tile := ⟨5, 7, 2, 3⟩
def tileIV : Tile := ⟨3, 5, 7, 2⟩

-- Define a function to check if two tiles can be adjacent
def canBeAdjacent (t1 t2 : Tile) (side : String) : Prop :=
  match side with
  | "right" => t1.right = t2.left
  | "left" => t1.left = t2.right
  | "top" => t1.top = t2.bottom
  | "bottom" => t1.bottom = t2.top
  | _ => False

-- Theorem stating that Tile IV is the only tile that can be placed in Rectangle C
theorem tileIV_in_rectangle_C :
  (canBeAdjacent tileIV tileIII "left") ∧
  (¬ canBeAdjacent tileI tileIII "left") ∧
  (¬ canBeAdjacent tileII tileIII "left") ∧
  (∃ (t : Tile), t = tileIV ∧ canBeAdjacent t tileIII "left") :=
sorry

end tileIV_in_rectangle_C_l3302_330275


namespace plane_line_relations_l3302_330270

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Plane → Prop)

-- State the theorem
theorem plane_line_relations
  (α β : Plane) (l m : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : l ≠ m)
  (h_l_perp_α : perpendicular l α)
  (h_m_subset_β : subset m β) :
  (parallel α β → line_perpendicular l m) ∧
  (perpendicular l β → line_parallel m α) :=
sorry

end plane_line_relations_l3302_330270


namespace sqrt_equation_solution_l3302_330204

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (9 - 2 * x) = 8 → x = -55/2 := by
  sorry

end sqrt_equation_solution_l3302_330204


namespace ab_greater_than_sum_l3302_330212

theorem ab_greater_than_sum (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b := by
  sorry

end ab_greater_than_sum_l3302_330212


namespace inequality_proof_l3302_330216

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^2 + a*b + b^2 = 3*c^2) (h2 : a^3 + a^2*b + a*b^2 + b^3 = 4*d^3) :
  a + b + d ≤ 3*c := by
  sorry

end inequality_proof_l3302_330216


namespace payment_sequence_aperiodic_l3302_330254

/-- A sequence of daily payments (1 or 2 rubles) -/
def PaymentSequence := ℕ → Fin 2

/-- The sum of the first n payments -/
def TotalPayment (seq : PaymentSequence) (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun i => seq i + 1)

/-- A payment sequence is valid if the total payment is always the nearest integer to n√2 -/
def IsValidPaymentSequence (seq : PaymentSequence) : Prop :=
  ∀ n : ℕ, |TotalPayment seq n - n * Real.sqrt 2| ≤ 1/2

/-- A sequence is periodic if it repeats after some point -/
def IsPeriodic (seq : PaymentSequence) : Prop :=
  ∃ (N T : ℕ), T > 0 ∧ ∀ n ≥ N, seq (n + T) = seq n

theorem payment_sequence_aperiodic (seq : PaymentSequence) 
  (hvalid : IsValidPaymentSequence seq) : ¬IsPeriodic seq := by
  sorry

end payment_sequence_aperiodic_l3302_330254


namespace triangle_angle_from_side_ratio_l3302_330255

theorem triangle_angle_from_side_ratio :
  ∀ (a b c : ℝ) (A B C : ℝ),
  (a > 0) → (b > 0) → (c > 0) →
  (a / b = 1 / Real.sqrt 3) →
  (a / c = 1 / 2) →
  (A + B + C = π) →
  (a^2 = b^2 + c^2 - 2*b*c*(Real.cos A)) →
  (b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) →
  (c^2 = a^2 + b^2 - 2*a*b*(Real.cos C)) →
  B = π / 3 := by
sorry

end triangle_angle_from_side_ratio_l3302_330255


namespace money_problem_l3302_330271

theorem money_problem (a b : ℝ) 
  (h1 : 5 * a + b > 51)
  (h2 : 3 * a - b = 21) :
  a > 9 ∧ b > 6 := by
sorry

end money_problem_l3302_330271


namespace bob_has_77_pennies_l3302_330205

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob three pennies, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bob_pennies + 3 = 4 * (alex_pennies - 3)

/-- If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 3 * (alex_pennies + 2)

/-- Bob currently has 77 pennies -/
theorem bob_has_77_pennies : bob_pennies = 77 := by sorry

end bob_has_77_pennies_l3302_330205


namespace average_score_calculation_l3302_330202

theorem average_score_calculation (total : ℝ) (male_ratio : ℝ) (male_avg : ℝ) (female_avg : ℝ)
  (h1 : male_ratio = 0.4)
  (h2 : male_avg = 75)
  (h3 : female_avg = 80) :
  (male_ratio * male_avg + (1 - male_ratio) * female_avg) = 78 := by
  sorry

end average_score_calculation_l3302_330202


namespace problem_solution_l3302_330218

theorem problem_solution (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 119) : 
  y = 1 := by
  sorry

end problem_solution_l3302_330218


namespace oranges_added_correct_l3302_330280

/-- The number of oranges added to make apples 50% of the total fruit -/
def oranges_added (initial_apples initial_oranges : ℕ) : ℕ :=
  let total := initial_apples + initial_oranges
  (2 * total) - initial_oranges

theorem oranges_added_correct (initial_apples initial_oranges : ℕ) :
  initial_apples = 10 →
  initial_oranges = 5 →
  oranges_added initial_apples initial_oranges = 5 :=
by
  sorry

#eval oranges_added 10 5

end oranges_added_correct_l3302_330280


namespace circle_and_line_intersection_l3302_330203

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 25}

-- Define the line that contains the center of C
def center_line : Set (ℝ × ℝ) :=
  {p | 2 * p.1 - p.2 - 2 = 0}

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 - 5 = k * (p.1 + 2)}

-- State the theorem
theorem circle_and_line_intersection
  (h1 : ((-3, 3) : ℝ × ℝ) ∈ circle_C)
  (h2 : ((1, -5) : ℝ × ℝ) ∈ circle_C)
  (h3 : ∃ c, c ∈ circle_C ∧ c ∈ center_line)
  (h4 : ∀ k > 0, ∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k)
  (h5 : (-2, 5) ∈ line_l k) :
  (∀ p ∈ circle_C, (p.1 - 1)^2 + p.2^2 = 25) ∧
  (∀ k > 15/8, ∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k) ∧
  (∀ k ≤ 15/8, ¬∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k) :=
by sorry

end circle_and_line_intersection_l3302_330203


namespace combined_transformation_correct_l3302_330259

def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![k, 0; 0, k]

def rotation_matrix_90_ccw : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -1; 1, 0]

def combined_transformation : Matrix (Fin 2) (Fin 2) ℝ :=
  !![0, -2; 2, 0]

theorem combined_transformation_correct :
  combined_transformation = rotation_matrix_90_ccw * dilation_matrix 2 := by
  sorry

end combined_transformation_correct_l3302_330259


namespace f_inequality_l3302_330238

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Condition 1: Periodicity
axiom periodic (x : ℝ) : f (x + 4) = f x

-- Condition 2: Decreasing on [0, 2]
axiom decreasing (x₁ x₂ : ℝ) (h : 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2) : f x₁ > f x₂

-- Condition 3: Symmetry about y-axis for f(x-2)
axiom symmetry (x : ℝ) : f ((-x) - 2) = f (x - 2)

-- Theorem to prove
theorem f_inequality : f (-1.5) < f 7 ∧ f 7 < f (-4.5) := by sorry

end f_inequality_l3302_330238


namespace circle_area_above_line_l3302_330235

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 16*y + 56 = 0

/-- The line equation -/
def line_eq (y : ℝ) : Prop := y = 4

/-- The area of the circle portion above the line -/
noncomputable def area_above_line : ℝ := 99 * Real.pi / 4

/-- Theorem stating that the area of the circle portion above the line is approximately equal to 99π/4 -/
theorem circle_area_above_line :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ x y : ℝ, circle_eq x y → line_eq y → 
    abs (area_above_line - (Real.pi * 33 * 3 / 4)) < ε) :=
sorry

end circle_area_above_line_l3302_330235


namespace one_and_two_thirds_of_x_is_45_l3302_330251

theorem one_and_two_thirds_of_x_is_45 (x : ℚ) : (5 / 3 : ℚ) * x = 45 → x = 27 := by
  sorry

end one_and_two_thirds_of_x_is_45_l3302_330251


namespace cone_volume_l3302_330247

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) (h : π > 0) : 
  let slant_height : ℝ := 15
  let height : ℝ := 9
  let radius : ℝ := Real.sqrt (slant_height^2 - height^2)
  let volume : ℝ := (1/3) * π * radius^2 * height
  volume = 432 * π :=
by sorry

end cone_volume_l3302_330247


namespace company_theorem_l3302_330291

-- Define the type for people
variable {Person : Type}

-- Define the "knows" relation
variable (knows : Person → Person → Prop)

-- Define the company as a finite set of people
variable [Finite Person]

-- State the theorem
theorem company_theorem 
  (h : ∀ (S : Finset Person), S.card = 9 → ∃ (x y : Person), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ knows x y) :
  ∃ (G : Finset Person), G.card = 8 ∧ 
    ∀ p, p ∉ G → ∃ q ∈ G, knows p q :=
sorry

end company_theorem_l3302_330291


namespace range_of_a_and_m_l3302_330263

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

-- Define the theorem
theorem range_of_a_and_m (a m : ℝ) 
  (h1 : A ∪ B a = A) 
  (h2 : A ∩ C m = C m) : 
  (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := by
  sorry


end range_of_a_and_m_l3302_330263


namespace star_difference_equals_28_l3302_330243

def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

theorem star_difference_equals_28 : (star 3 5) - (star 2 4) = 28 := by
  sorry

end star_difference_equals_28_l3302_330243


namespace vector_operation_proof_l3302_330228

def vector_operation : ℝ × ℝ := sorry

theorem vector_operation_proof :
  vector_operation = (5, 2) := by
  sorry

end vector_operation_proof_l3302_330228


namespace most_economical_cost_l3302_330219

/-- Represents the problem of finding the most economical cost for purchasing warm reminder signs and garbage bins. -/
theorem most_economical_cost
  (price_difference : ℕ)
  (price_ratio : ℕ)
  (total_items : ℕ)
  (max_cost : ℕ)
  (bin_sign_ratio : ℚ)
  (h1 : price_difference = 350)
  (h2 : price_ratio = 3)
  (h3 : total_items = 3000)
  (h4 : max_cost = 350000)
  (h5 : bin_sign_ratio = 3/2)
  : ∃ (sign_price bin_price : ℕ) (num_signs num_bins : ℕ) (total_cost : ℕ),
    -- Price relationship between bins and signs
    4 * bin_price - 5 * sign_price = price_difference ∧
    bin_price = price_ratio * sign_price ∧
    -- Total number of items constraint
    num_signs + num_bins = total_items ∧
    -- Cost constraint
    num_signs * sign_price + num_bins * bin_price ≤ max_cost ∧
    -- Ratio constraint
    (num_bins : ℚ) ≥ bin_sign_ratio * (num_signs : ℚ) ∧
    -- Most economical solution
    num_signs = 1200 ∧
    total_cost = 330000 ∧
    -- No cheaper solution exists
    ∀ (other_signs : ℕ), 
      other_signs ≠ num_signs →
      other_signs + (total_items - other_signs) = total_items →
      (total_items - other_signs : ℚ) ≥ bin_sign_ratio * (other_signs : ℚ) →
      other_signs * sign_price + (total_items - other_signs) * bin_price ≥ total_cost :=
by sorry


end most_economical_cost_l3302_330219


namespace max_customers_interviewed_l3302_330232

theorem max_customers_interviewed (total : ℕ) (impulsive : ℕ) (ad_influence_percent : ℚ) (consultant_ratio : ℚ) : 
  total ≤ 50 ∧ 
  impulsive = 7 ∧ 
  ad_influence_percent = 3/4 ∧ 
  consultant_ratio = 1/3 →
  ∃ (max_customers : ℕ), 
    max_customers ≤ 50 ∧
    (∃ (ad_influenced : ℕ) (consultant_advised : ℕ),
      max_customers = impulsive + ad_influenced + consultant_advised ∧
      ad_influenced = ⌊(max_customers - impulsive) * ad_influence_percent⌋ ∧
      consultant_advised = ⌊ad_influenced * consultant_ratio⌋) ∧
    ∀ (n : ℕ), n > max_customers →
      ¬(∃ (ad_influenced : ℕ) (consultant_advised : ℕ),
        n = impulsive + ad_influenced + consultant_advised ∧
        ad_influenced = ⌊(n - impulsive) * ad_influence_percent⌋ ∧
        consultant_advised = ⌊ad_influenced * consultant_ratio⌋) ∧
    max_customers = 47 :=
by sorry

end max_customers_interviewed_l3302_330232


namespace repeating_decimal_subtraction_l3302_330272

/-- Represents a repeating decimal with a given numerator and denominator. -/
structure RepeatingDecimal where
  numerator : ℕ
  denominator : ℕ
  denom_nonzero : denominator ≠ 0

/-- Converts a repeating decimal to a rational number. -/
def RepeatingDecimal.toRational (r : RepeatingDecimal) : ℚ :=
  ↑r.numerator / ↑r.denominator

theorem repeating_decimal_subtraction :
  let a : RepeatingDecimal := ⟨845, 999, by norm_num⟩
  let b : RepeatingDecimal := ⟨267, 999, by norm_num⟩
  let c : RepeatingDecimal := ⟨159, 999, by norm_num⟩
  a.toRational - b.toRational - c.toRational = 419 / 999 := by
  sorry

end repeating_decimal_subtraction_l3302_330272


namespace polynomial_remainder_theorem_l3302_330231

/-- A polynomial P(x) with a real parameter r satisfies:
    1) P(x) has remainder 2 when divided by (x-r)
    2) P(x) has remainder (-2x^2 - 3x + 4) when divided by (2x^2 + 7x - 4)(x-r)
    This theorem states that r can only be 1/2 or -2. -/
theorem polynomial_remainder_theorem (P : ℝ → ℝ) (r : ℝ) :
  (∃ Q₁ : ℝ → ℝ, ∀ x, P x = (x - r) * Q₁ x + 2) ∧
  (∃ Q₂ : ℝ → ℝ, ∀ x, P x = (2*x^2 + 7*x - 4)*(x - r) * Q₂ x + (-2*x^2 - 3*x + 4)) →
  r = 1/2 ∨ r = -2 := by
sorry

end polynomial_remainder_theorem_l3302_330231
