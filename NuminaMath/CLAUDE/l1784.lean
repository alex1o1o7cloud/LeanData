import Mathlib

namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1784_178406

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l1784_178406


namespace NUMINAMATH_CALUDE_expression_value_l1784_178477

theorem expression_value (x y z : ℚ) 
  (h1 : 3 * x - 2 * y - 2 * z = 0)
  (h2 : x - 4 * y + 8 * z = 0)
  (h3 : z ≠ 0) :
  (3 * x^2 - 2 * x * y) / (y^2 + 4 * z^2) = 120 / 269 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1784_178477


namespace NUMINAMATH_CALUDE_scooter_price_l1784_178433

theorem scooter_price (upfront_payment : ℝ) (upfront_percentage : ℝ) (total_price : ℝ) : 
  upfront_payment = 240 → 
  upfront_percentage = 0.20 → 
  upfront_payment = upfront_percentage * total_price → 
  total_price = 1200 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_l1784_178433


namespace NUMINAMATH_CALUDE_kishore_savings_percentage_l1784_178411

def rent : ℕ := 5000
def milk : ℕ := 1500
def groceries : ℕ := 4500
def education : ℕ := 2500
def petrol : ℕ := 2000
def miscellaneous : ℕ := 700
def savings : ℕ := 1800

def total_expenses : ℕ := rent + milk + groceries + education + petrol + miscellaneous
def total_salary : ℕ := total_expenses + savings

theorem kishore_savings_percentage :
  (savings : ℚ) / (total_salary : ℚ) * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_kishore_savings_percentage_l1784_178411


namespace NUMINAMATH_CALUDE_system_solution_l1784_178479

theorem system_solution :
  ∃! (s : Set (ℝ × ℝ)), s = {(2, 4), (4, 2)} ∧
  ∀ (x y : ℝ), (x / y + y / x) * (x + y) = 15 ∧
                (x^2 / y^2 + y^2 / x^2) * (x^2 + y^2) = 85 →
                (x, y) ∈ s :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1784_178479


namespace NUMINAMATH_CALUDE_volcano_lake_depth_l1784_178428

/-- Represents a cone-shaped volcano partially submerged in a lake -/
structure Volcano :=
  (height : ℝ)
  (above_water_ratio : ℝ)

/-- Calculates the depth of the lake at the base of the volcano -/
def lake_depth (v : Volcano) : ℝ :=
  v.height * (1 - (v.above_water_ratio ^ (1/3)))

/-- Theorem stating the depth of the lake for a specific volcano -/
theorem volcano_lake_depth :
  let v := Volcano.mk 6000 (1/6)
  lake_depth v = 390 := by
  sorry

end NUMINAMATH_CALUDE_volcano_lake_depth_l1784_178428


namespace NUMINAMATH_CALUDE_tan_negative_five_pi_fourths_l1784_178475

theorem tan_negative_five_pi_fourths : Real.tan (-5 * Real.pi / 4) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_five_pi_fourths_l1784_178475


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l1784_178470

theorem complex_number_imaginary_part (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) / (1 - Complex.I)
  Complex.im z = 2 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l1784_178470


namespace NUMINAMATH_CALUDE_raisins_amount_l1784_178410

/-- The amount of peanuts used in the trail mix -/
def peanuts : ℝ := 0.16666666666666666

/-- The amount of chocolate chips used in the trail mix -/
def chocolate_chips : ℝ := 0.16666666666666666

/-- The total amount of trail mix -/
def total_mix : ℝ := 0.4166666666666667

/-- The amount of raisins used in the trail mix -/
def raisins : ℝ := total_mix - (peanuts + chocolate_chips)

theorem raisins_amount : raisins = 0.08333333333333337 := by sorry

end NUMINAMATH_CALUDE_raisins_amount_l1784_178410


namespace NUMINAMATH_CALUDE_winnieThePoohServings_l1784_178485

/-- Represents the number of servings eaten by each character -/
structure Servings where
  cheburashka : ℕ
  winnieThePooh : ℕ
  carlson : ℕ

/-- The rate at which characters eat relative to each other -/
def eatingRate (s : Servings) : Prop :=
  5 * s.cheburashka = 2 * s.winnieThePooh ∧
  7 * s.winnieThePooh = 3 * s.carlson

/-- The total number of servings eaten by Cheburashka and Carlson -/
def totalServings (s : Servings) : Prop :=
  s.cheburashka + s.carlson = 82

/-- Theorem stating that Winnie-the-Pooh ate 30 servings -/
theorem winnieThePoohServings (s : Servings) 
  (h1 : eatingRate s) (h2 : totalServings s) : s.winnieThePooh = 30 := by
  sorry

end NUMINAMATH_CALUDE_winnieThePoohServings_l1784_178485


namespace NUMINAMATH_CALUDE_training_effect_l1784_178478

/-- Represents the scores and their frequencies in a test --/
structure TestScores :=
  (scores : List Nat)
  (frequencies : List Nat)

/-- Calculates the median of a list of scores --/
def median (scores : List Nat) : Nat :=
  sorry

/-- Calculates the mode of a list of scores --/
def mode (scores : List Nat) (frequencies : List Nat) : Nat :=
  sorry

/-- Calculates the average score --/
def average (scores : List Nat) (frequencies : List Nat) : Real :=
  sorry

/-- Calculates the number of students with scores greater than or equal to a threshold --/
def countExcellent (scores : List Nat) (frequencies : List Nat) (threshold : Nat) : Nat :=
  sorry

theorem training_effect (baselineScores simExamScores : TestScores)
  (totalStudents sampleSize : Nat)
  (hTotalStudents : totalStudents = 800)
  (hSampleSize : sampleSize = 50)
  (hBaselineScores : baselineScores = ⟨[6, 7, 8, 9, 10], [16, 8, 9, 9, 8]⟩)
  (hSimExamScores : simExamScores = ⟨[6, 7, 8, 9, 10], [5, 8, 6, 12, 19]⟩) :
  (median baselineScores.scores = 8 ∧ 
   mode simExamScores.scores simExamScores.frequencies = 10) ∧
  (average simExamScores.scores simExamScores.frequencies - 
   average baselineScores.scores baselineScores.frequencies = 0.94) ∧
  (totalStudents * (countExcellent simExamScores.scores simExamScores.frequencies 9) / sampleSize = 496) :=
by sorry

end NUMINAMATH_CALUDE_training_effect_l1784_178478


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l1784_178472

theorem negation_of_universal_statement (S : Set ℝ) :
  (¬ ∀ x ∈ S, |x| > 1) ↔ (∃ x ∈ S, |x| ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l1784_178472


namespace NUMINAMATH_CALUDE_cubic_sum_l1784_178436

theorem cubic_sum (p q r : ℝ) 
  (h1 : p + q + r = 5) 
  (h2 : p * q + p * r + q * r = 3) 
  (h3 : p * q * r = -2) : 
  p^3 + q^3 + r^3 = 89 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_l1784_178436


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l1784_178467

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := k * x + y - 2 = 3 * k

-- State the theorem
theorem fixed_point_theorem :
  ∀ k : ℝ, line_equation k 3 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l1784_178467


namespace NUMINAMATH_CALUDE_stanley_distance_difference_l1784_178448

/-- Given Stanley's running and walking distances, prove the difference between them. -/
theorem stanley_distance_difference (run walk : ℝ) 
  (h1 : run = 0.4) 
  (h2 : walk = 0.2) : 
  run - walk = 0.2 := by
sorry

end NUMINAMATH_CALUDE_stanley_distance_difference_l1784_178448


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_192_18_is_six_less_than_200_exists_no_greater_main_result_l1784_178453

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192 :=
by sorry

theorem gcd_192_18_is_six : Nat.gcd 192 18 = 6 :=
by sorry

theorem less_than_200 : 192 < 200 :=
by sorry

theorem exists_no_greater : ¬∃ m : ℕ, 192 < m ∧ m < 200 ∧ Nat.gcd m 18 = 6 :=
by sorry

theorem main_result : 
  (∃ n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6) ∧ 
  (∀ n : ℕ, n < 200 ∧ Nat.gcd n 18 = 6 → n ≤ 192) ∧
  (Nat.gcd 192 18 = 6) ∧
  (192 < 200) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_gcd_192_18_is_six_less_than_200_exists_no_greater_main_result_l1784_178453


namespace NUMINAMATH_CALUDE_experts_win_probability_value_l1784_178494

/-- The probability of Experts winning a single round -/
def p : ℝ := 0.6

/-- The probability of Audience winning a single round -/
def q : ℝ := 1 - p

/-- The current score of Experts -/
def experts_score : ℕ := 3

/-- The current score of Audience -/
def audience_score : ℕ := 4

/-- The number of wins needed to win the game -/
def wins_needed : ℕ := 6

/-- The probability that the Experts will eventually win the game -/
def experts_win_probability : ℝ := p^4 + 4 * p^3 * q

/-- Theorem stating that the probability of Experts winning is 0.4752 -/
theorem experts_win_probability_value : 
  experts_win_probability = 0.4752 := by sorry

end NUMINAMATH_CALUDE_experts_win_probability_value_l1784_178494


namespace NUMINAMATH_CALUDE_heloise_pets_l1784_178497

theorem heloise_pets (total_pets : ℕ) (dogs_given : ℕ) : 
  total_pets = 189 →
  dogs_given = 10 →
  (∃ (dogs cats : ℕ), 
    dogs + cats = total_pets ∧ 
    dogs * 17 = cats * 10) →
  ∃ (remaining_dogs : ℕ), remaining_dogs = 60 :=
by sorry

end NUMINAMATH_CALUDE_heloise_pets_l1784_178497


namespace NUMINAMATH_CALUDE_car_speed_problem_l1784_178439

/-- Proves that for a journey of 225 km, if a car arrives 45 minutes late when traveling at 50 kmph, then its on-time average speed is 60 kmph. -/
theorem car_speed_problem (journey_length : ℝ) (late_speed : ℝ) (delay : ℝ) :
  journey_length = 225 →
  late_speed = 50 →
  delay = 3/4 →
  ∃ (on_time_speed : ℝ),
    (journey_length / on_time_speed) + delay = (journey_length / late_speed) ∧
    on_time_speed = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1784_178439


namespace NUMINAMATH_CALUDE_cos_90_degrees_eq_zero_l1784_178452

theorem cos_90_degrees_eq_zero : Real.cos (π / 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_90_degrees_eq_zero_l1784_178452


namespace NUMINAMATH_CALUDE_product_of_square_roots_l1784_178434

theorem product_of_square_roots (y : ℝ) (h : y ≥ 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 60 * y * Real.sqrt (3 * y) :=
by sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l1784_178434


namespace NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l1784_178493

-- Define the size of the grid
def gridSize : ℕ := 5

-- Define the number of dots to choose
def chosenDots : ℕ := 5

-- Define the number of collinear sets of 5 dots in a 5x5 grid
def collinearSets : ℕ := 12

-- Define the total number of ways to choose 5 dots out of 25
def totalCombinations : ℕ := Nat.choose (gridSize * gridSize) chosenDots

-- Theorem statement
theorem collinear_probability_in_5x5_grid :
  (collinearSets : ℚ) / totalCombinations = 2 / 8855 :=
sorry

end NUMINAMATH_CALUDE_collinear_probability_in_5x5_grid_l1784_178493


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l1784_178442

/-- The number of ways to arrange n distinct objects. -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways six people can sit in a row of seven chairs with the third chair vacant. -/
def seatingArrangements : ℕ := factorial 6

theorem seating_arrangements_count : seatingArrangements = 720 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_count_l1784_178442


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l1784_178492

theorem square_root_of_sixteen (x : ℝ) : x^2 = 16 ↔ x = 4 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l1784_178492


namespace NUMINAMATH_CALUDE_crayon_ratio_l1784_178423

theorem crayon_ratio (total : ℕ) (blue : ℕ) (red : ℕ) : 
  total = 15 → blue = 3 → red = total - blue → (red : ℚ) / blue = 4 / 1 := by
  sorry

end NUMINAMATH_CALUDE_crayon_ratio_l1784_178423


namespace NUMINAMATH_CALUDE_card_distribution_exists_iff_odd_l1784_178463

/-- A magic pair is a pair of consecutive numbers or the pair (1, n(n-1)/2) -/
def is_magic_pair (a b : Nat) (n : Nat) : Prop :=
  (a + 1 = b ∨ b + 1 = a) ∨ (a = 1 ∧ b = n * (n - 1) / 2) ∨ (b = 1 ∧ a = n * (n - 1) / 2)

/-- A valid distribution of cards into stacks -/
def valid_distribution (n : Nat) (stacks : Fin n → Finset Nat) : Prop :=
  (∀ i : Fin n, ∀ x ∈ stacks i, x ≤ n * (n - 1) / 2) ∧
  (∀ i j : Fin n, i ≠ j → ∃! (a b : Nat), a ∈ stacks i ∧ b ∈ stacks j ∧ is_magic_pair a b n)

theorem card_distribution_exists_iff_odd (n : Nat) (h : n > 2) :
  (∃ stacks : Fin n → Finset Nat, valid_distribution n stacks) ↔ Odd n :=
sorry

end NUMINAMATH_CALUDE_card_distribution_exists_iff_odd_l1784_178463


namespace NUMINAMATH_CALUDE_total_revenue_calculation_l1784_178451

/-- Calculate the total revenue from vegetable sales --/
theorem total_revenue_calculation :
  let morning_potatoes : ℕ := 29
  let morning_onions : ℕ := 15
  let morning_carrots : ℕ := 12
  let afternoon_potatoes : ℕ := 17
  let afternoon_onions : ℕ := 22
  let afternoon_carrots : ℕ := 9
  let potato_weight : ℕ := 7
  let onion_weight : ℕ := 5
  let carrot_weight : ℕ := 4
  let potato_price : ℚ := 1.75
  let onion_price : ℚ := 2.50
  let carrot_price : ℚ := 3.25

  let total_potatoes : ℕ := morning_potatoes + afternoon_potatoes
  let total_onions : ℕ := morning_onions + afternoon_onions
  let total_carrots : ℕ := morning_carrots + afternoon_carrots

  let potato_revenue : ℚ := (total_potatoes * potato_weight : ℚ) * potato_price
  let onion_revenue : ℚ := (total_onions * onion_weight : ℚ) * onion_price
  let carrot_revenue : ℚ := (total_carrots * carrot_weight : ℚ) * carrot_price

  let total_revenue : ℚ := potato_revenue + onion_revenue + carrot_revenue

  total_revenue = 1299.00 := by sorry

end NUMINAMATH_CALUDE_total_revenue_calculation_l1784_178451


namespace NUMINAMATH_CALUDE_inequality_and_equality_l1784_178486

theorem inequality_and_equality (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  (b < -1 ∨ b > 0 → (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b) ∧
  ((1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_l1784_178486


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1784_178449

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 25 = 1

/-- Definition of an asymptote -/
def is_asymptote (m b : ℝ) : Prop :=
  ∀ ε > 0, ∃ M > 0, ∀ x y : ℝ, 
    hyperbola x y → (|x| > M → |y - (m * x + b)| < ε)

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  (is_asymptote (4/5) (13/5)) ∧ (is_asymptote (-4/5) (13/5)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1784_178449


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1784_178462

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (1/3)*(m*x - 1) > 2 - m ↔ x < -4) → m = -7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1784_178462


namespace NUMINAMATH_CALUDE_tan_sqrt3_inequality_l1784_178491

open Set Real

-- Define the set of x that satisfies the inequality
def S : Set ℝ := {x | tan x - Real.sqrt 3 ≤ 0}

-- Define the solution set
def T : Set ℝ := {x | ∃ k : ℤ, -π/2 + k*π < x ∧ x ≤ π/3 + k*π}

-- Theorem statement
theorem tan_sqrt3_inequality : S = T := by sorry

end NUMINAMATH_CALUDE_tan_sqrt3_inequality_l1784_178491


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1784_178414

theorem product_of_repeating_decimal_and_eight :
  let t : ℚ := 456 / 999
  t * 8 = 48 / 13 := by sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1784_178414


namespace NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_7_mod_9_l1784_178447

theorem unique_prime_between_30_and_40_with_remainder_7_mod_9 :
  ∃! p : ℕ, Prime p ∧ 30 < p ∧ p < 40 ∧ p % 9 = 7 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_between_30_and_40_with_remainder_7_mod_9_l1784_178447


namespace NUMINAMATH_CALUDE_number_equation_l1784_178457

theorem number_equation (x : ℝ) (h : 5 * x = 2 * x + 10) : 5 * x - 2 * x = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l1784_178457


namespace NUMINAMATH_CALUDE_age_sum_problem_l1784_178441

theorem age_sum_problem (a b : ℕ) (h1 : a > b) (h2 : a * b * b * b = 256) : 
  a + b + b + b = 38 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1784_178441


namespace NUMINAMATH_CALUDE_cookies_baked_l1784_178437

/-- Given 5 pans of cookies with 8 cookies per pan, prove that the total number of cookies is 40. -/
theorem cookies_baked (pans : ℕ) (cookies_per_pan : ℕ) (h1 : pans = 5) (h2 : cookies_per_pan = 8) :
  pans * cookies_per_pan = 40 := by
  sorry

end NUMINAMATH_CALUDE_cookies_baked_l1784_178437


namespace NUMINAMATH_CALUDE_widget_earnings_calculation_l1784_178455

/-- Calculates the earnings per widget given the hourly wage, work hours, 
    required widget production, and total weekly earnings. -/
def earnings_per_widget (hourly_wage : ℚ) (work_hours : ℕ) 
  (required_widgets : ℕ) (total_earnings : ℚ) : ℚ :=
  (total_earnings - hourly_wage * work_hours) / required_widgets

theorem widget_earnings_calculation : 
  earnings_per_widget (12.5) 40 500 580 = (16 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_widget_earnings_calculation_l1784_178455


namespace NUMINAMATH_CALUDE_garden_length_l1784_178424

theorem garden_length (columns : ℕ) (tree_distance : ℝ) (boundary : ℝ) : 
  columns > 0 → 
  tree_distance > 0 → 
  boundary > 0 → 
  (columns - 1) * tree_distance + 2 * boundary = 32 → 
  columns = 12 ∧ tree_distance = 2 ∧ boundary = 5 := by
  sorry

end NUMINAMATH_CALUDE_garden_length_l1784_178424


namespace NUMINAMATH_CALUDE_integer_1025_column_l1784_178418

def column_sequence := ["B", "C", "D", "E", "A"]

theorem integer_1025_column :
  let n := 1025 - 1
  let column_index := n % (List.length column_sequence)
  List.get! column_sequence column_index = "E" := by
  sorry

end NUMINAMATH_CALUDE_integer_1025_column_l1784_178418


namespace NUMINAMATH_CALUDE_cockatiel_eats_fifty_grams_weekly_l1784_178488

/-- The amount of birdseed a cockatiel eats per week -/
def cockatiel_weekly_consumption (
  boxes_bought : ℕ
  ) (boxes_in_pantry : ℕ
  ) (parrot_weekly_consumption : ℕ
  ) (grams_per_box : ℕ
  ) (weeks_of_feeding : ℕ
  ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let parrot_total_consumption := parrot_weekly_consumption * weeks_of_feeding
  let cockatiel_total_consumption := total_grams - parrot_total_consumption
  cockatiel_total_consumption / weeks_of_feeding

/-- Theorem stating that given the conditions in the problem, 
    the cockatiel eats 50 grams of seeds each week -/
theorem cockatiel_eats_fifty_grams_weekly :
  cockatiel_weekly_consumption 3 5 100 225 12 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cockatiel_eats_fifty_grams_weekly_l1784_178488


namespace NUMINAMATH_CALUDE_square_fits_in_unit_cube_l1784_178498

theorem square_fits_in_unit_cube : ∃ (s : ℝ), s ≥ 1.05 ∧ 
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 
    s = min (Real.sqrt (2 * (1 - x)^2)) (Real.sqrt (1 + 2 * x^2)) :=
sorry

end NUMINAMATH_CALUDE_square_fits_in_unit_cube_l1784_178498


namespace NUMINAMATH_CALUDE_symmetry_about_x_2_symmetry_about_2_0_l1784_178468

-- Define a function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Theorem for symmetry about x = 2
theorem symmetry_about_x_2 (h : ∀ x, f (1 - x) = f (3 + x)) :
  ∀ x, f (2 - x) = f (2 + x) := by sorry

-- Theorem for symmetry about (2,0)
theorem symmetry_about_2_0 (h : ∀ x, f (1 - x) = -f (3 + x)) :
  ∀ x, f (2 - x) = -f (2 + x) := by sorry

end NUMINAMATH_CALUDE_symmetry_about_x_2_symmetry_about_2_0_l1784_178468


namespace NUMINAMATH_CALUDE_probability_both_presidents_selected_l1784_178429

/-- Represents a math club with its total number of members -/
structure MathClub where
  members : Nat
  presidents : Nat
  mascots : Nat

/-- The list of math clubs in the district -/
def mathClubs : List MathClub := [
  { members := 6, presidents := 2, mascots := 1 },
  { members := 9, presidents := 2, mascots := 1 },
  { members := 10, presidents := 2, mascots := 1 },
  { members := 11, presidents := 2, mascots := 1 }
]

/-- The number of members to be selected from a club -/
def selectCount : Nat := 4

/-- Calculates the probability of selecting both presidents when choosing
    a specific number of members from a given club -/
def probBothPresidentsSelected (club : MathClub) (selectCount : Nat) : Rat :=
  sorry

/-- Calculates the overall probability of selecting both presidents when
    choosing from a randomly selected club -/
def overallProbability (clubs : List MathClub) (selectCount : Nat) : Rat :=
  sorry

/-- The main theorem stating the probability of selecting both presidents -/
theorem probability_both_presidents_selected :
  overallProbability mathClubs selectCount = 7/25 := by sorry

end NUMINAMATH_CALUDE_probability_both_presidents_selected_l1784_178429


namespace NUMINAMATH_CALUDE_factorial_sum_equality_l1784_178421

theorem factorial_sum_equality : 7 * Nat.factorial 7 + 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_factorial_sum_equality_l1784_178421


namespace NUMINAMATH_CALUDE_coconuts_per_crab_calculation_l1784_178402

/-- The number of coconuts Max has -/
def total_coconuts : ℕ := 342

/-- The number of goats Max will have after conversion -/
def total_goats : ℕ := 19

/-- The number of crabs that can be traded for a goat -/
def crabs_per_goat : ℕ := 6

/-- The number of coconuts needed to trade for a crab -/
def coconuts_per_crab : ℕ := 3

theorem coconuts_per_crab_calculation :
  coconuts_per_crab * crabs_per_goat * total_goats = total_coconuts :=
sorry

end NUMINAMATH_CALUDE_coconuts_per_crab_calculation_l1784_178402


namespace NUMINAMATH_CALUDE_range_of_a_l1784_178426

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 12*x + 20 < 0
def q (x a : ℝ) : Prop := x^2 - 2*x + 1 - a^2 > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (a > 0) →
  (∀ x : ℝ, ¬(q x a) → ¬(p x)) →
  (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1784_178426


namespace NUMINAMATH_CALUDE_adjacent_probability_l1784_178405

/-- The number of people -/
def total_people : ℕ := 9

/-- The number of rows -/
def num_rows : ℕ := 3

/-- The number of chairs in each row -/
def chairs_per_row : ℕ := 3

/-- The probability of two specific people sitting next to each other in the same row -/
def probability_adjacent : ℚ := 2 / 9

theorem adjacent_probability :
  probability_adjacent = (2 : ℚ) / (total_people : ℚ) := by sorry

end NUMINAMATH_CALUDE_adjacent_probability_l1784_178405


namespace NUMINAMATH_CALUDE_quadrilateral_perimeter_area_inequality_l1784_178465

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry

-- Define the perimeter of a quadrilateral
def perimeter (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the area of a quadrilateral
def area (q : ConvexQuadrilateral) : ℝ := sorry

-- Define the perimeter of the quadrilateral formed by the centers of inscribed circles
def inscribed_centers_perimeter (q : ConvexQuadrilateral) : ℝ := sorry

-- Statement of the theorem
theorem quadrilateral_perimeter_area_inequality (q : ConvexQuadrilateral) :
  perimeter q * inscribed_centers_perimeter q > 4 * area q := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_perimeter_area_inequality_l1784_178465


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1784_178489

theorem remainder_divisibility (x : ℕ) (h : x > 0) :
  (200 % x = 2) → (398 % x = 2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1784_178489


namespace NUMINAMATH_CALUDE_exponent_rules_l1784_178440

theorem exponent_rules (a b : ℝ) : 
  (a^3 * a^3 = a^6) ∧ 
  ¬((a*b)^3 = a*b^3) ∧ 
  ¬((a^3)^3 = a^6) ∧ 
  ¬(a^8 / a^4 = a^2) := by
  sorry

end NUMINAMATH_CALUDE_exponent_rules_l1784_178440


namespace NUMINAMATH_CALUDE_minimum_students_l1784_178422

theorem minimum_students (n : ℕ) (h1 : n > 1000) 
  (h2 : n % 10 = 0) (h3 : n % 14 = 0) (h4 : n % 18 = 0) :
  n ≥ 1260 := by
  sorry

end NUMINAMATH_CALUDE_minimum_students_l1784_178422


namespace NUMINAMATH_CALUDE_solve_q_l1784_178458

theorem solve_q (p q : ℝ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 1/p + 1/q = 3/2) 
  (h4 : p*q = 6) : 
  q = (9 + Real.sqrt 57) / 2 := by
sorry

end NUMINAMATH_CALUDE_solve_q_l1784_178458


namespace NUMINAMATH_CALUDE_john_hat_days_l1784_178459

def total_cost : ℕ := 700
def hat_cost : ℕ := 50

theorem john_hat_days : (total_cost / hat_cost : ℕ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_john_hat_days_l1784_178459


namespace NUMINAMATH_CALUDE_pot_holds_three_liters_l1784_178464

/-- Represents the volume of a pot in liters -/
def pot_volume (drops_per_minute : ℕ) (ml_per_drop : ℕ) (minutes_to_fill : ℕ) : ℚ :=
  (drops_per_minute * ml_per_drop * minutes_to_fill : ℚ) / 1000

/-- Theorem stating that a pot filled by a leak with given parameters holds 3 liters -/
theorem pot_holds_three_liters :
  pot_volume 3 20 50 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pot_holds_three_liters_l1784_178464


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1784_178430

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetric_point_coordinates :
  let B : Point := { x := 4, y := -1 }
  let A : Point := symmetricXAxis B
  A.x = 4 ∧ A.y = 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1784_178430


namespace NUMINAMATH_CALUDE_mall_audit_sampling_is_systematic_l1784_178445

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | Other

/-- Represents an invoice stub --/
structure InvoiceStub :=
  (number : ℕ)

/-- Represents a book of invoice stubs --/
def InvoiceBook := List InvoiceStub

/-- Represents a sampling process --/
structure SamplingProcess :=
  (book : InvoiceBook)
  (initialSelection : ℕ)
  (interval : ℕ)

/-- Determines if a sampling process is systematic --/
def isSystematicSampling (process : SamplingProcess) : Prop :=
  process.initialSelection ≤ 50 ∧ 
  process.interval = 50 ∧
  (∀ n : ℕ, (process.initialSelection + n * process.interval) ∈ (process.book.map InvoiceStub.number))

/-- The main theorem to prove --/
theorem mall_audit_sampling_is_systematic 
  (book : InvoiceBook)
  (initialStub : ℕ)
  (h1 : initialStub ≤ 50)
  (h2 : initialStub ∈ (book.map InvoiceStub.number))
  : isSystematicSampling ⟨book, initialStub, 50⟩ := by
  sorry

#check mall_audit_sampling_is_systematic

end NUMINAMATH_CALUDE_mall_audit_sampling_is_systematic_l1784_178445


namespace NUMINAMATH_CALUDE_average_of_specific_odds_l1784_178484

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_in_range (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 10

def is_less_than_6 (n : ℕ) : Prop := n < 6

def meets_conditions (n : ℕ) : Prop :=
  is_odd n ∧ is_in_range n ∧ is_less_than_6 n

def numbers_meeting_conditions : List ℕ :=
  [1, 3, 5]

theorem average_of_specific_odds :
  (numbers_meeting_conditions.sum : ℚ) / numbers_meeting_conditions.length = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_of_specific_odds_l1784_178484


namespace NUMINAMATH_CALUDE_equation_solutions_l1784_178460

theorem equation_solutions : 
  {x : ℝ | x^6 + (2-x)^6 = 272} = {1 + Real.sqrt 3, 1 - Real.sqrt 3} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1784_178460


namespace NUMINAMATH_CALUDE_smallest_number_l1784_178417

theorem smallest_number (a b c d : ℝ) (ha : a = -1) (hb : b = 0) (hc : c = Real.sqrt 2) (hd : d = -1/2) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l1784_178417


namespace NUMINAMATH_CALUDE_product_divisible_by_49_l1784_178495

theorem product_divisible_by_49 (a b : ℕ) (h : 7 ∣ (a^2 + b^2)) : 49 ∣ (a * b) := by
  sorry

end NUMINAMATH_CALUDE_product_divisible_by_49_l1784_178495


namespace NUMINAMATH_CALUDE_band_member_earnings_l1784_178408

theorem band_member_earnings 
  (attendees : ℕ) 
  (revenue_share : ℚ) 
  (ticket_price : ℕ) 
  (band_members : ℕ) 
  (h1 : attendees = 500) 
  (h2 : revenue_share = 7/10) 
  (h3 : ticket_price = 30) 
  (h4 : band_members = 4) :
  (attendees * ticket_price * revenue_share) / band_members = 2625 := by
sorry

end NUMINAMATH_CALUDE_band_member_earnings_l1784_178408


namespace NUMINAMATH_CALUDE_furniture_by_design_salary_l1784_178487

/-- The monthly salary from Furniture by Design -/
def S : ℝ := 1800

/-- The base salary for the commission-based option -/
def base_salary : ℝ := 1600

/-- The commission rate for the commission-based option -/
def commission_rate : ℝ := 0.04

/-- The sales amount at which both payment options are equal -/
def equal_sales : ℝ := 5000

theorem furniture_by_design_salary :
  S = base_salary + commission_rate * equal_sales :=
by sorry

end NUMINAMATH_CALUDE_furniture_by_design_salary_l1784_178487


namespace NUMINAMATH_CALUDE_inequality_solution_range_l1784_178407

theorem inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, 2 * k * x^2 + k * x - 3/8 < 0) ↔ k ∈ Set.Ioc (-3) 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l1784_178407


namespace NUMINAMATH_CALUDE_system_solution_l1784_178466

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y = 3 ∧ x - y = 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(2, 1)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), system x y ↔ (x, y) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1784_178466


namespace NUMINAMATH_CALUDE_dogGroupings_eq_2520_l1784_178481

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Rover in the 4-dog group and Spot in the 6-dog group. -/
def dogGroupings : ℕ :=
  (Nat.choose 10 3) * (Nat.choose 7 5)

/-- Theorem stating that the number of ways to divide the dogs is 2520. -/
theorem dogGroupings_eq_2520 : dogGroupings = 2520 := by
  sorry

end NUMINAMATH_CALUDE_dogGroupings_eq_2520_l1784_178481


namespace NUMINAMATH_CALUDE_calculation_proof_l1784_178401

theorem calculation_proof :
  (1 / (Real.sqrt 5 + 2) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5) = 2) ∧
  (2 * Real.sqrt 3 * 612 * (3 + 3/2) = 5508 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1784_178401


namespace NUMINAMATH_CALUDE_solve_for_y_l1784_178482

theorem solve_for_y (x y : ℝ) (h1 : x = 100) (h2 : x^3*y - 3*x^2*y + 3*x*y = 3000000) : 
  y = 3000000 / 970299 := by sorry

end NUMINAMATH_CALUDE_solve_for_y_l1784_178482


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1784_178435

/-- A right triangle with specific medians -/
structure RightTriangle where
  /-- Length of one leg -/
  x : ℝ
  /-- Length of the other leg -/
  y : ℝ
  /-- The triangle is right-angled -/
  right_angle : x ^ 2 + y ^ 2 > 0
  /-- One median has length 3 -/
  median1 : x ^ 2 + (y / 2) ^ 2 = 3 ^ 2
  /-- The other median has length 2√13 -/
  median2 : y ^ 2 + (x / 2) ^ 2 = (2 * Real.sqrt 13) ^ 2

/-- The hypotenuse of the right triangle is 8√1.1 -/
theorem hypotenuse_length (t : RightTriangle) : 
  Real.sqrt (4 * (t.x ^ 2 + t.y ^ 2)) = 8 * Real.sqrt 1.1 := by
  sorry

#check hypotenuse_length

end NUMINAMATH_CALUDE_hypotenuse_length_l1784_178435


namespace NUMINAMATH_CALUDE_equation_solution_l1784_178496

theorem equation_solution : ∃! x : ℝ, 90 + 5 * 12 / (x / 3) = 91 ∧ x = 180 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1784_178496


namespace NUMINAMATH_CALUDE_infinitely_many_primes_with_special_property_l1784_178490

theorem infinitely_many_primes_with_special_property :
  ∀ k : ℕ, ∃ (p n : ℕ), 
    p > k ∧ 
    Prime p ∧ 
    n > 0 ∧ 
    ¬(n ∣ (p - 1)) ∧ 
    (p ∣ (Nat.factorial n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_with_special_property_l1784_178490


namespace NUMINAMATH_CALUDE_regression_line_estimate_l1784_178427

-- Define the regression line
def regression_line (b a x : ℝ) : ℝ := b * x + a

-- State the theorem
theorem regression_line_estimate :
  ∀ (b a : ℝ),
  b = 1.23 →
  regression_line b a 4 = 5 →
  regression_line b a 2 = 2.54 := by
sorry

end NUMINAMATH_CALUDE_regression_line_estimate_l1784_178427


namespace NUMINAMATH_CALUDE_undefined_expression_l1784_178471

theorem undefined_expression (b : ℝ) : 
  ¬ (∃ x : ℝ, x = (b - 1) / (b^2 - 9)) ↔ b = -3 ∨ b = 3 :=
sorry

end NUMINAMATH_CALUDE_undefined_expression_l1784_178471


namespace NUMINAMATH_CALUDE_ellipse_rolling_conditions_l1784_178444

/-- 
An ellipse with semi-axes a and b rolls without slipping on the curve y = c sin(x/a) 
and completes one revolution in one period of the sine curve. 
This theorem states the conditions that a, b, and c must satisfy.
-/
theorem ellipse_rolling_conditions 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c ≠ 0) 
  (h_ellipse : ∀ (t : ℝ), ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1) 
  (h_curve : ∀ (x : ℝ), ∃ (y : ℝ), y = c * Real.sin (x / a)) 
  (h_roll : ∀ (t : ℝ), ∃ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 ∧ y = c * Real.sin (x / a)) 
  (h_period : ∃ (T : ℝ), T > 0 ∧ ∀ (x : ℝ), c * Real.sin (x / a) = c * Real.sin ((x + T) / a)) :
  b ≥ a ∧ c^2 = b^2 - a^2 ∧ c * b^2 < a^3 :=
sorry

end NUMINAMATH_CALUDE_ellipse_rolling_conditions_l1784_178444


namespace NUMINAMATH_CALUDE_max_intersections_eight_l1784_178415

/-- Represents a tiled floor with equilateral triangles -/
structure TriangularFloor where
  side_length : ℝ
  side_length_positive : side_length > 0

/-- Represents a needle -/
structure Needle where
  length : ℝ
  length_positive : length > 0

/-- Counts the maximum number of triangles intersected by a needle -/
def max_intersected_triangles (floor : TriangularFloor) (needle : Needle) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersected triangles -/
theorem max_intersections_eight
  (floor : TriangularFloor)
  (needle : Needle)
  (h_floor : floor.side_length = 1)
  (h_needle : needle.length = 2) :
  max_intersected_triangles floor needle = 8 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_eight_l1784_178415


namespace NUMINAMATH_CALUDE_no_solution_to_equation_l1784_178432

theorem no_solution_to_equation :
  ¬∃ (x : ℝ), (x - 1) / (x + 1) - 4 / (x^2 - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l1784_178432


namespace NUMINAMATH_CALUDE_train_cars_estimate_l1784_178483

/-- The number of cars Trey counted -/
def cars_counted : ℕ := 8

/-- The time (in seconds) Trey spent counting -/
def counting_time : ℕ := 15

/-- The total time (in seconds) the train took to pass -/
def total_time : ℕ := 210

/-- The estimated number of cars in the train -/
def estimated_cars : ℕ := 112

/-- Theorem stating that the estimated number of cars is approximately correct -/
theorem train_cars_estimate :
  abs ((cars_counted : ℚ) / counting_time * total_time - estimated_cars) < 1 := by
  sorry


end NUMINAMATH_CALUDE_train_cars_estimate_l1784_178483


namespace NUMINAMATH_CALUDE_rational_absolute_value_inequality_l1784_178473

theorem rational_absolute_value_inequality (a : ℚ) (h : a - |a| = 2*a) : a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_inequality_l1784_178473


namespace NUMINAMATH_CALUDE_meat_market_sales_ratio_l1784_178413

/-- Given the sales data for a Meat Market over four days, prove the ratio of Sunday to Saturday sales --/
theorem meat_market_sales_ratio :
  let thursday_sales : ℕ := 210
  let friday_sales : ℕ := 2 * thursday_sales
  let saturday_sales : ℕ := 130
  let planned_total : ℕ := 500
  let actual_total : ℕ := planned_total + 325
  let sunday_sales : ℕ := actual_total - (thursday_sales + friday_sales + saturday_sales)
  (sunday_sales : ℚ) / saturday_sales = 1 / 2 := by
  sorry

#check meat_market_sales_ratio

end NUMINAMATH_CALUDE_meat_market_sales_ratio_l1784_178413


namespace NUMINAMATH_CALUDE_cake_muffin_mix_probability_l1784_178404

theorem cake_muffin_mix_probability :
  ∀ (total buyers cake_buyers muffin_buyers both_buyers : ℕ),
    total = 100 →
    cake_buyers = 50 →
    muffin_buyers = 40 →
    both_buyers = 18 →
    (total - (cake_buyers + muffin_buyers - both_buyers)) / total = 28 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_muffin_mix_probability_l1784_178404


namespace NUMINAMATH_CALUDE_squares_pattern_squares_figure_100_l1784_178419

/-- The number of squares in figure n -/
def num_squares (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

/-- The sequence of squares follows the given pattern for the first four figures -/
theorem squares_pattern :
  num_squares 0 = 1 ∧
  num_squares 1 = 7 ∧
  num_squares 2 = 19 ∧
  num_squares 3 = 37 := by sorry

/-- The number of squares in figure 100 is 30301 -/
theorem squares_figure_100 :
  num_squares 100 = 30301 := by sorry

end NUMINAMATH_CALUDE_squares_pattern_squares_figure_100_l1784_178419


namespace NUMINAMATH_CALUDE_parallelogram_diagonals_sides_sum_l1784_178438

/-- A parallelogram with vertices A, B, C, and D. -/
structure Parallelogram :=
  (A B C D : ℝ × ℝ)
  (is_parallelogram : (A.1 - B.1, A.2 - B.2) = (D.1 - C.1, D.2 - C.2) ∧ 
                      (A.1 - D.1, A.2 - D.2) = (B.1 - C.1, B.2 - C.2))

/-- The squared distance between two points in ℝ² -/
def dist_squared (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Theorem: The sum of the squares of the diagonals of a parallelogram 
    is equal to the sum of the squares of its four sides -/
theorem parallelogram_diagonals_sides_sum (P : Parallelogram) : 
  dist_squared P.A P.C + dist_squared P.B P.D = 
  dist_squared P.A P.B + dist_squared P.B P.C + 
  dist_squared P.C P.D + dist_squared P.D P.A :=
sorry

end NUMINAMATH_CALUDE_parallelogram_diagonals_sides_sum_l1784_178438


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l1784_178443

theorem sum_of_reciprocals_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  ∃ (z : ℝ), z ≥ 2 ∧ (∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = 2 ∧ 1 / x' + 1 / y' = z) ∧
  (∀ (a b : ℝ), a > 0 → b > 0 → a + b = 2 → 1 / a + 1 / b ≥ 2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_bound_l1784_178443


namespace NUMINAMATH_CALUDE_phi_value_is_65_degrees_l1784_178476

-- Define the condition that φ is an acute angle
def is_acute_angle (φ : Real) : Prop := 0 < φ ∧ φ < Real.pi / 2

-- State the theorem
theorem phi_value_is_65_degrees :
  ∀ φ : Real,
  is_acute_angle φ →
  Real.sqrt 2 * Real.cos (20 * Real.pi / 180) = Real.sin φ - Real.cos φ →
  φ = 65 * Real.pi / 180 := by
sorry

end NUMINAMATH_CALUDE_phi_value_is_65_degrees_l1784_178476


namespace NUMINAMATH_CALUDE_prob_four_draws_ge_ten_expected_value_two_draws_l1784_178409

-- Define the bags and their contents
def bagA : Finset (Fin 10) := {0,1,2,3,4,5,6,7,8,9}
def bagB : Finset (Fin 10) := {0,1,2,3,4,5,6,7,8,9}

-- Define the probabilities of drawing each color
def probRedA : ℝ := 0.8
def probWhiteA : ℝ := 0.2
def probYellowB : ℝ := 0.9
def probBlackB : ℝ := 0.1

-- Define the scoring system
def scoreRed : ℤ := 4
def scoreWhite : ℤ := -1
def scoreYellow : ℤ := 6
def scoreBlack : ℤ := -2

-- Define the game rules
def fourDraws : ℕ := 4
def minScore : ℤ := 10

-- Theorem for Question 1
theorem prob_four_draws_ge_ten (p : ℝ) : 
  p = probRedA^4 + 4 * probRedA^3 * probWhiteA → p = 0.8192 := by sorry

-- Theorem for Question 2
theorem expected_value_two_draws (ev : ℝ) :
  ev = scoreRed * probRedA * probYellowB + 
        scoreRed * probRedA * probBlackB + 
        scoreWhite * probWhiteA * probYellowB + 
        scoreWhite * probWhiteA * probBlackB → ev = 8.2 := by sorry

end NUMINAMATH_CALUDE_prob_four_draws_ge_ten_expected_value_two_draws_l1784_178409


namespace NUMINAMATH_CALUDE_movie_watching_time_l1784_178431

/-- Represents the duration of a part of the movie watching session -/
structure MoviePart where
  watch_time : Nat
  rewind_time : Nat

/-- Calculates the total time for a movie watching session -/
def total_movie_time (parts : List MoviePart) : Nat :=
  parts.foldl (fun acc part => acc + part.watch_time + part.rewind_time) 0

/-- Theorem stating that the total time to watch the movie is 120 minutes -/
theorem movie_watching_time :
  let part1 : MoviePart := { watch_time := 35, rewind_time := 5 }
  let part2 : MoviePart := { watch_time := 45, rewind_time := 15 }
  let part3 : MoviePart := { watch_time := 20, rewind_time := 0 }
  total_movie_time [part1, part2, part3] = 120 := by
  sorry


end NUMINAMATH_CALUDE_movie_watching_time_l1784_178431


namespace NUMINAMATH_CALUDE_solution_set_cubic_inequality_l1784_178420

theorem solution_set_cubic_inequality :
  {x : ℝ | x + x^3 ≥ 0} = {x : ℝ | x ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_cubic_inequality_l1784_178420


namespace NUMINAMATH_CALUDE_mary_stickers_left_l1784_178425

/-- The number of stickers Mary has left over after distributing them in class -/
def stickers_left_over (total_stickers : ℕ) (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (total_students : ℕ) (stickers_per_other : ℕ) : ℕ :=
  total_stickers - 
  (num_friends * stickers_per_friend + 
   (total_students - 1 - num_friends) * stickers_per_other)

/-- Theorem stating that Mary has 8 stickers left over -/
theorem mary_stickers_left : stickers_left_over 50 5 4 17 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_mary_stickers_left_l1784_178425


namespace NUMINAMATH_CALUDE_num_syt_54321_l1784_178416

/-- A partition is a non-increasing sequence of natural numbers. -/
def Partition : Type := List Nat

/-- A Standard Young Tableau is a filling of a partition shape with integers
    such that rows and columns are strictly increasing. -/
def StandardYoungTableau (p : Partition) : Type := sorry

/-- Hook length of a cell in a partition -/
def hookLength (p : Partition) (i j : Nat) : Nat := sorry

/-- Number of Standard Young Tableaux for a given partition -/
def numSYT (p : Partition) : Nat := sorry

/-- The main theorem: number of Standard Young Tableaux for shape (5,4,3,2,1) -/
theorem num_syt_54321 :
  numSYT [5, 4, 3, 2, 1] = 292864 := by sorry

end NUMINAMATH_CALUDE_num_syt_54321_l1784_178416


namespace NUMINAMATH_CALUDE_intercepts_sum_l1784_178456

/-- A line is described by the equation y - 3 = 6(x - 5). -/
def line_equation (x y : ℝ) : Prop := y - 3 = 6 * (x - 5)

/-- The x-intercept of the line. -/
def x_intercept : ℝ := 4.5

/-- The y-intercept of the line. -/
def y_intercept : ℝ := -27

theorem intercepts_sum :
  line_equation x_intercept 0 ∧
  line_equation 0 y_intercept ∧
  x_intercept + y_intercept = -22.5 := by sorry

end NUMINAMATH_CALUDE_intercepts_sum_l1784_178456


namespace NUMINAMATH_CALUDE_joshs_initial_money_l1784_178400

theorem joshs_initial_money (hat_cost pencil_cost cookie_cost : ℚ)
  (num_cookies : ℕ) (money_left : ℚ) :
  hat_cost = 10 →
  pencil_cost = 2 →
  cookie_cost = 5/4 →
  num_cookies = 4 →
  money_left = 3 →
  hat_cost + pencil_cost + num_cookies * cookie_cost + money_left = 20 :=
by sorry

end NUMINAMATH_CALUDE_joshs_initial_money_l1784_178400


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1784_178480

/-- Given an ellipse C and a line l, prove the eccentricity e --/
theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃ (e : ℝ), 
    (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → 
      ∃ (A B M : ℝ × ℝ), 
        (A.2 = 0 ∧ B.1 = 0) ∧ 
        (M.2 = e * M.1 + a) ∧
        (A.1 = -a / e ∧ B.2 = a) ∧
        (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
        ((M.1 - A.1)^2 + (M.2 - A.2)^2 = e^2 * ((B.1 - A.1)^2 + (B.2 - A.2)^2))) →
    e = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1784_178480


namespace NUMINAMATH_CALUDE_factoring_expression_l1784_178469

theorem factoring_expression (x : ℝ) : 3*x*(x+2) + 2*(x+2) + 5*(x+2) = (x+2)*(3*x+7) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l1784_178469


namespace NUMINAMATH_CALUDE_oplus_example_1_oplus_example_2_l1784_178474

-- Define the ⊕ operation for rational numbers
def oplus (a b : ℚ) : ℚ := (a + 3 * b) / 2

-- Theorem for part (1)
theorem oplus_example_1 : 4 * (oplus 2 5) = 34 := by sorry

-- Define polynomials A and B
def A (x y : ℚ) : ℚ := x^2 + 2*x*y + y^2
def B (x y : ℚ) : ℚ := -2*x*y + y^2

-- Theorem for part (2)
theorem oplus_example_2 (x y : ℚ) : 
  (oplus (A x y) (B x y)) + (oplus (B x y) (A x y)) = 2*x^2 + 4*y^2 := by sorry

end NUMINAMATH_CALUDE_oplus_example_1_oplus_example_2_l1784_178474


namespace NUMINAMATH_CALUDE_total_profit_is_390_4_l1784_178461

/-- Represents the partnership of A, B, and C -/
structure Partnership where
  a_share : Rat
  b_share : Rat
  c_share : Rat
  a_withdrawal_time : Nat
  a_withdrawal_fraction : Rat
  profit_distribution_time : Nat
  b_profit_share : Rat

/-- Calculates the total profit given the partnership conditions -/
def calculate_total_profit (p : Partnership) : Rat :=
  sorry

/-- Theorem stating that the total profit is 390.4 given the specified conditions -/
theorem total_profit_is_390_4 (p : Partnership) 
  (h1 : p.a_share = 1/2)
  (h2 : p.b_share = 1/3)
  (h3 : p.c_share = 1/4)
  (h4 : p.a_withdrawal_time = 2)
  (h5 : p.a_withdrawal_fraction = 1/2)
  (h6 : p.profit_distribution_time = 10)
  (h7 : p.b_profit_share = 144) :
  calculate_total_profit p = 390.4 := by
  sorry

end NUMINAMATH_CALUDE_total_profit_is_390_4_l1784_178461


namespace NUMINAMATH_CALUDE_meal_cost_is_25_l1784_178412

/-- The cost of Hilary's meal at Delicious Delhi restaurant -/
def meal_cost : ℝ :=
  let samosa_price : ℝ := 2
  let pakora_price : ℝ := 3
  let lassi_price : ℝ := 2
  let samosa_quantity : ℕ := 3
  let pakora_quantity : ℕ := 4
  let tip_percentage : ℝ := 0.25
  let subtotal : ℝ := samosa_price * samosa_quantity + pakora_price * pakora_quantity + lassi_price
  let tip : ℝ := subtotal * tip_percentage
  subtotal + tip

theorem meal_cost_is_25 : meal_cost = 25 := by
  sorry

end NUMINAMATH_CALUDE_meal_cost_is_25_l1784_178412


namespace NUMINAMATH_CALUDE_decreasing_linear_function_not_in_first_quadrant_l1784_178499

/-- A linear function y = kx + b that decreases as x increases and has a negative y-intercept -/
structure DecreasingLinearFunction where
  k : ℝ
  b : ℝ
  k_neg : k < 0
  b_neg : b < 0

/-- The first quadrant of the Cartesian plane -/
def FirstQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

/-- The theorem stating that a decreasing linear function with negative y-intercept does not pass through the first quadrant -/
theorem decreasing_linear_function_not_in_first_quadrant (f : DecreasingLinearFunction) :
  ∀ x y, y = f.k * x + f.b → (x, y) ∉ FirstQuadrant := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_not_in_first_quadrant_l1784_178499


namespace NUMINAMATH_CALUDE_t_shape_perimeter_l1784_178403

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the perimeter of a rectangle --/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- Represents a T-shaped figure formed by two rectangles --/
structure TShape where
  vertical : Rectangle
  horizontal : Rectangle

/-- Calculates the perimeter of a T-shaped figure --/
def TShape.perimeter (t : TShape) : ℝ :=
  t.vertical.perimeter + t.horizontal.perimeter - 4 * t.horizontal.width

theorem t_shape_perimeter :
  let t : TShape := {
    vertical := { width := 2, height := 6 },
    horizontal := { width := 2, height := 4 }
  }
  t.perimeter = 24 := by sorry

end NUMINAMATH_CALUDE_t_shape_perimeter_l1784_178403


namespace NUMINAMATH_CALUDE_number_equation_solution_l1784_178450

theorem number_equation_solution : ∃ x : ℝ, 3 * x + 4 = 19 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l1784_178450


namespace NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1784_178446

theorem square_sum_given_difference_and_product (x y : ℝ) 
  (h1 : x - y = 10) (h2 : x * y = 9) : x^2 + y^2 = 118 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_difference_and_product_l1784_178446


namespace NUMINAMATH_CALUDE_cubic_function_properties_l1784_178454

-- Define the function f
def f (a b c x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

-- Define the derivative of f
def f' (a b x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∃ x : ℝ, f' a b x = 0 ∧ x = 2) →
  (f' a b 1 = -3) →
  (a = -1 ∧ b = 0) ∧
  (∃ x_max x_min : ℝ, f (-1) 0 c x_max - f (-1) 0 c x_min = 4) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_properties_l1784_178454
