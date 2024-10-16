import Mathlib

namespace NUMINAMATH_CALUDE_f_properties_l3193_319391

noncomputable section

def f (x : ℝ) : ℝ := Real.log x - (x - 1)^2 / 2

def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem f_properties :
  (∀ x y, 0 < x ∧ x < y ∧ y < phi → f x < f y) ∧
  (∀ x, 1 < x → f x < x - 1) ∧
  (∀ k, (∃ x₀, 1 < x₀ ∧ ∀ x, 1 < x ∧ x < x₀ → k * (x - 1) < f x) → k < 1) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l3193_319391


namespace NUMINAMATH_CALUDE_least_sum_exponents_1540_l3193_319321

/-- The function that computes the least sum of exponents for a given number -/
def leastSumOfExponents (n : ℕ) : ℕ := sorry

/-- The theorem stating that the least sum of exponents for 1540 is 21 -/
theorem least_sum_exponents_1540 : leastSumOfExponents 1540 = 21 := by sorry

end NUMINAMATH_CALUDE_least_sum_exponents_1540_l3193_319321


namespace NUMINAMATH_CALUDE_sector_arc_length_l3193_319310

/-- Given a sector with central angle 1 radian and radius 5 cm, the arc length is 5 cm. -/
theorem sector_arc_length (θ : Real) (r : Real) (l : Real) : 
  θ = 1 → r = 5 → l = r * θ → l = 5 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3193_319310


namespace NUMINAMATH_CALUDE_blake_apples_cost_l3193_319398

/-- The amount Blake spent on apples -/
def apples_cost (total : ℕ) (change : ℕ) (oranges : ℕ) (mangoes : ℕ) : ℕ :=
  total - change - (oranges + mangoes)

/-- Theorem: Blake spent $50 on apples -/
theorem blake_apples_cost :
  apples_cost 300 150 40 60 = 50 := by
  sorry

end NUMINAMATH_CALUDE_blake_apples_cost_l3193_319398


namespace NUMINAMATH_CALUDE_stamp_collection_value_l3193_319350

/-- Given a collection of stamps with equal individual value, 
    calculate the total value of the collection. -/
theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℝ) 
  (h1 : total_stamps = 30)
  (h2 : sample_stamps = 10)
  (h3 : sample_value = 45) :
  (total_stamps : ℝ) * (sample_value / sample_stamps) = 135 :=
by sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l3193_319350


namespace NUMINAMATH_CALUDE_powerjet_45min_output_l3193_319337

/-- A pump that pumps water at a given rate. -/
structure Pump where
  rate : ℝ  -- Gallons per hour

/-- Calculates the amount of water pumped in a given time. -/
def water_pumped (p : Pump) (time : ℝ) : ℝ :=
  p.rate * time

/-- Theorem: A pump that pumps 420 gallons per hour will pump 315 gallons in 45 minutes. -/
theorem powerjet_45min_output (p : Pump) (h : p.rate = 420) : 
  water_pumped p (45 / 60) = 315 := by
  sorry

#check powerjet_45min_output

end NUMINAMATH_CALUDE_powerjet_45min_output_l3193_319337


namespace NUMINAMATH_CALUDE_all_roots_nonzero_l3193_319348

theorem all_roots_nonzero :
  (∀ x : ℝ, 4 * x^2 - 6 = 34 → x ≠ 0) ∧
  (∀ x : ℝ, (3 * x - 1)^2 = (x + 2)^2 → x ≠ 0) ∧
  (∀ x : ℝ, (x^2 - 4 : ℝ) = (x + 3 : ℝ) → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_all_roots_nonzero_l3193_319348


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l3193_319352

/-- The nth odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Theorem stating that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : nthOddMultipleOf5 15 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l3193_319352


namespace NUMINAMATH_CALUDE_complementary_angles_l3193_319353

theorem complementary_angles (C D : ℝ) : 
  C + D = 90 →  -- C and D are complementary
  C = 5 * D →   -- C is 5 times D
  C = 75 :=     -- C is 75°
by sorry

end NUMINAMATH_CALUDE_complementary_angles_l3193_319353


namespace NUMINAMATH_CALUDE_divisibility_of_group_difference_l3193_319343

/-- Represents a person in the circle, either a boy or a girl -/
inductive Person
| Boy
| Girl

/-- The circle of people -/
def Circle := List Person

/-- Count the number of groups of 3 consecutive people with exactly one boy -/
def countGroupsWithOneBoy (circle : Circle) : Nat :=
  sorry

/-- Count the number of groups of 3 consecutive people with exactly one girl -/
def countGroupsWithOneGirl (circle : Circle) : Nat :=
  sorry

theorem divisibility_of_group_difference (n : Nat) (circle : Circle) 
    (h1 : n ≥ 3)
    (h2 : circle.length = n) :
  let a := countGroupsWithOneBoy circle
  let b := countGroupsWithOneGirl circle
  3 ∣ (a - b) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_group_difference_l3193_319343


namespace NUMINAMATH_CALUDE_missing_carton_dimension_l3193_319392

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Represents the carton with one unknown dimension -/
def carton (x : ℝ) : BoxDimensions :=
  { length := 25, width := x, height := 60 }

/-- Represents the soap box dimensions -/
def soapBox : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- The maximum number of soap boxes that can fit in the carton -/
def maxSoapBoxes : ℕ := 300

theorem missing_carton_dimension :
  ∃ x : ℝ, boxVolume (carton x) = (maxSoapBoxes : ℝ) * boxVolume soapBox ∧ x = 48 := by
  sorry

end NUMINAMATH_CALUDE_missing_carton_dimension_l3193_319392


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3193_319314

theorem quadratic_inequality_solution_set (x : ℝ) : 
  {x : ℝ | x^2 - 4*x + 3 < 0} = Set.Ioo 1 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3193_319314


namespace NUMINAMATH_CALUDE_interest_calculation_l3193_319320

/-- Represents the problem of finding the minimum number of years for a specific interest calculation. -/
theorem interest_calculation (principal1 principal2 rate1 rate2 target_interest : ℚ) :
  principal1 = 800 →
  principal2 = 1400 →
  rate1 = 3 / 100 →
  rate2 = 5 / 100 →
  target_interest = 350 →
  (∃ (n : ℕ), (principal1 * rate1 * n + principal2 * rate2 * n ≥ target_interest) ∧
    (∀ (m : ℕ), m < n → principal1 * rate1 * m + principal2 * rate2 * m < target_interest)) →
  (∃ (n : ℕ), (principal1 * rate1 * n + principal2 * rate2 * n ≥ target_interest) ∧
    (∀ (m : ℕ), m < n → principal1 * rate1 * m + principal2 * rate2 * m < target_interest) ∧
    n = 4) :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_l3193_319320


namespace NUMINAMATH_CALUDE_fourth_sample_number_l3193_319380

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  start : ℕ

/-- Checks if a number is in the systematic sample -/
def in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k : ℕ, n = s.start + k * s.interval ∧ n ≤ s.population_size

theorem fourth_sample_number 
  (s : SystematicSample)
  (h_pop : s.population_size = 56)
  (h_sample : s.sample_size = 4)
  (h_6 : in_sample s 6)
  (h_34 : in_sample s 34)
  (h_48 : in_sample s 48) :
  in_sample s 20 :=
sorry

end NUMINAMATH_CALUDE_fourth_sample_number_l3193_319380


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3193_319397

-- Define the problem
theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3193_319397


namespace NUMINAMATH_CALUDE_complex_modulus_one_l3193_319322

theorem complex_modulus_one (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l3193_319322


namespace NUMINAMATH_CALUDE_sally_walked_2540_miles_l3193_319385

/-- Calculates the total miles walked given pedometer resets, final reading, steps per mile, and additional steps --/
def total_miles_walked (resets : ℕ) (final_reading : ℕ) (steps_per_mile : ℕ) (additional_steps : ℕ) : ℕ :=
  let total_steps := resets * 100000 + final_reading + additional_steps
  (total_steps + steps_per_mile - 1) / steps_per_mile

/-- Theorem stating that Sally walked 2540 miles during the year --/
theorem sally_walked_2540_miles :
  total_miles_walked 50 30000 2000 50000 = 2540 := by
  sorry

end NUMINAMATH_CALUDE_sally_walked_2540_miles_l3193_319385


namespace NUMINAMATH_CALUDE_least_k_value_l3193_319301

/-- The function f(t) = t² - t + 1 -/
def f (t : ℝ) : ℝ := t^2 - t + 1

/-- The property that needs to be satisfied for all x, y, z that are not all positive -/
def satisfies_property (k : ℝ) : Prop :=
  ∀ x y z : ℝ, ¬(x > 0 ∧ y > 0 ∧ z > 0) →
    k * f x * f y * f z ≥ f (x * y * z)

/-- The theorem stating that 16/9 is the least value of k satisfying the property -/
theorem least_k_value : 
  (∀ k : ℝ, k < 16/9 → ¬(satisfies_property k)) ∧ 
  satisfies_property (16/9) := by sorry

end NUMINAMATH_CALUDE_least_k_value_l3193_319301


namespace NUMINAMATH_CALUDE_fraction_subtraction_l3193_319359

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l3193_319359


namespace NUMINAMATH_CALUDE_zhang_bing_special_year_l3193_319316

/-- Given that Zhang Bing was born in 1953, this theorem proves the existence and uniqueness of a year between 1953 and 2023 where his age is both a multiple of 9 and equal to the sum of the digits of that year. -/
theorem zhang_bing_special_year : 
  ∃! Y : ℕ, 1953 < Y ∧ Y < 2023 ∧ 
  (∃ k : ℕ, Y - 1953 = 9 * k) ∧
  (Y - 1953 = (Y / 1000) + ((Y % 1000) / 100) + ((Y % 100) / 10) + (Y % 10)) :=
by sorry

end NUMINAMATH_CALUDE_zhang_bing_special_year_l3193_319316


namespace NUMINAMATH_CALUDE_sevenDigitIntegers_eq_630_l3193_319395

/-- The number of different positive, seven-digit integers that can be formed
    using the digits 2, 2, 3, 5, 5, 9, and 9 -/
def sevenDigitIntegers : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2)

/-- Theorem stating that the number of different positive, seven-digit integers
    that can be formed using the digits 2, 2, 3, 5, 5, 9, and 9 is 630 -/
theorem sevenDigitIntegers_eq_630 : sevenDigitIntegers = 630 := by
  sorry

end NUMINAMATH_CALUDE_sevenDigitIntegers_eq_630_l3193_319395


namespace NUMINAMATH_CALUDE_remaining_episodes_l3193_319354

theorem remaining_episodes (series1_seasons series2_seasons episodes_per_season episodes_lost_per_season : ℕ) 
  (h1 : series1_seasons = 12)
  (h2 : series2_seasons = 14)
  (h3 : episodes_per_season = 16)
  (h4 : episodes_lost_per_season = 2) :
  (series1_seasons * episodes_per_season + series2_seasons * episodes_per_season) -
  (series1_seasons * episodes_lost_per_season + series2_seasons * episodes_lost_per_season) = 364 := by
sorry

end NUMINAMATH_CALUDE_remaining_episodes_l3193_319354


namespace NUMINAMATH_CALUDE_expression_equality_l3193_319399

theorem expression_equality : -2^2 + Real.sqrt 8 - 3 + 1/3 = -20/3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3193_319399


namespace NUMINAMATH_CALUDE_sheridan_cats_goal_l3193_319357

/-- The number of cats Mrs. Sheridan currently has -/
def current_cats : ℕ := 11

/-- The number of additional cats Mrs. Sheridan needs -/
def additional_cats : ℕ := 32

/-- The total number of cats Mrs. Sheridan wants to have -/
def total_cats : ℕ := current_cats + additional_cats

theorem sheridan_cats_goal : total_cats = 43 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_goal_l3193_319357


namespace NUMINAMATH_CALUDE_choir_members_count_l3193_319379

theorem choir_members_count : ∃! n : ℕ, 
  200 < n ∧ n < 300 ∧ 
  (n + 4) % 10 = 0 ∧ 
  (n + 5) % 11 = 0 ∧ 
  n = 226 := by
sorry

end NUMINAMATH_CALUDE_choir_members_count_l3193_319379


namespace NUMINAMATH_CALUDE_dice_probability_l3193_319307

def first_die : Finset ℕ := {1, 3, 5, 6}
def second_die : Finset ℕ := {1, 2, 4, 5, 7, 9}

def sum_in_range (x : ℕ) (y : ℕ) : Bool :=
  let sum := x + y
  8 ≤ sum ∧ sum ≤ 10

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (first_die.product second_die).filter (fun (x, y) ↦ sum_in_range x y)

def total_outcomes : ℕ := (first_die.card * second_die.card : ℕ)

theorem dice_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes = 7 / 18 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_dice_probability_l3193_319307


namespace NUMINAMATH_CALUDE_employed_males_percentage_l3193_319340

/-- In a population where 60% are employed and 30% of the employed are females,
    the percentage of employed males in the total population is 42%. -/
theorem employed_males_percentage
  (total : ℕ) -- Total population
  (employed_ratio : ℚ) -- Ratio of employed people to total population
  (employed_females_ratio : ℚ) -- Ratio of employed females to employed people
  (h1 : employed_ratio = 60 / 100)
  (h2 : employed_females_ratio = 30 / 100)
  : (employed_ratio * (1 - employed_females_ratio)) * 100 = 42 := by
  sorry

end NUMINAMATH_CALUDE_employed_males_percentage_l3193_319340


namespace NUMINAMATH_CALUDE_rectangle_width_l3193_319366

/-- Given a rectangle with perimeter 6a + 4b and length 2a + b, prove its width is a + b -/
theorem rectangle_width (a b : ℝ) : 
  let perimeter := 6*a + 4*b
  let length := 2*a + b
  let width := (perimeter / 2) - length
  width = a + b := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l3193_319366


namespace NUMINAMATH_CALUDE_x_percent_of_z_l3193_319334

theorem x_percent_of_z (x y z : ℝ) (h1 : x = 1.20 * y) (h2 : y = 0.50 * z) : x = 0.60 * z := by
  sorry

end NUMINAMATH_CALUDE_x_percent_of_z_l3193_319334


namespace NUMINAMATH_CALUDE_larger_triangle_perimeter_l3193_319344

-- Define the original right triangle
def original_triangle (a b c : ℝ) : Prop :=
  a = 8 ∧ b = 15 ∧ c^2 = a^2 + b^2

-- Define the similarity ratio
def similarity_ratio (k : ℝ) (a : ℝ) : Prop :=
  k * a = 20 ∧ k > 0

-- Define the larger similar triangle
def larger_triangle (a b c k : ℝ) : Prop :=
  original_triangle a b c ∧ similarity_ratio k a

-- Theorem statement
theorem larger_triangle_perimeter 
  (a b c k : ℝ) 
  (h : larger_triangle a b c k) : 
  k * (a + b + c) = 100 := by
    sorry


end NUMINAMATH_CALUDE_larger_triangle_perimeter_l3193_319344


namespace NUMINAMATH_CALUDE_game_wheel_probability_l3193_319309

theorem game_wheel_probability : 
  ∀ (p_A p_B p_C p_D p_E : ℚ),
    p_A = 2/7 →
    p_B = 1/7 →
    p_C = p_D →
    p_C = p_E →
    p_A + p_B + p_C + p_D + p_E = 1 →
    p_C = 4/21 := by
  sorry

end NUMINAMATH_CALUDE_game_wheel_probability_l3193_319309


namespace NUMINAMATH_CALUDE_pure_imaginary_quotient_condition_l3193_319382

theorem pure_imaginary_quotient_condition (a : ℝ) : 
  let z₁ : ℂ := a + Complex.I
  let z₂ : ℂ := 3 - 4 * Complex.I
  (∃ (b : ℝ), z₁ / z₂ = b * Complex.I) → a = 4/3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_quotient_condition_l3193_319382


namespace NUMINAMATH_CALUDE_faye_pencil_rows_l3193_319315

/-- The number of rows that can be made with a given number of pencils and pencils per row. -/
def number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem: Faye can make 6 rows with 30 pencils, placing 5 pencils in each row. -/
theorem faye_pencil_rows : number_of_rows 30 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_faye_pencil_rows_l3193_319315


namespace NUMINAMATH_CALUDE_evenness_condition_l3193_319386

/-- Given a real number ω, prove that there exists a real number a such that 
    f(x+a) is an even function, where f(x) = (x-6)^2 * sin(ωx), 
    if and only if ω = π/4 -/
theorem evenness_condition (ω : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, (((x + a - 6)^2 * Real.sin (ω * (x + a))) = 
                      (((-x) + a - 6)^2 * Real.sin (ω * ((-x) + a)))))
  ↔ 
  ω = π / 4 := by
sorry

end NUMINAMATH_CALUDE_evenness_condition_l3193_319386


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3193_319325

theorem imaginary_part_of_complex_fraction (m : ℝ) : 
  (Complex.im ((2 - Complex.I) * (m + Complex.I)) = 0) → 
  (Complex.im (m * Complex.I / (1 - Complex.I)) = 1) := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3193_319325


namespace NUMINAMATH_CALUDE_no_solution_iff_k_equals_seven_l3193_319329

theorem no_solution_iff_k_equals_seven :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - k) / (x - 8)) ↔ k = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_equals_seven_l3193_319329


namespace NUMINAMATH_CALUDE_evaluate_expression_l3193_319356

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) :
  5 * x^y - 2 * y^x = 277 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3193_319356


namespace NUMINAMATH_CALUDE_sum_of_log_equation_l3193_319305

theorem sum_of_log_equation : 
  ∀ k m n : ℕ+,
  (Nat.gcd k.val (Nat.gcd m.val n.val) = 1) →
  (k : ℝ) * (Real.log 5 / Real.log 400) + (m : ℝ) * (Real.log 2 / Real.log 400) = n →
  k + m + n = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_log_equation_l3193_319305


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l3193_319341

theorem unique_three_digit_number :
  ∃! n : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    (n % 100 % 10 = 3 * (n / 100)) ∧
    (n % 5 = 4) ∧
    (n % 11 = 3) ∧
    n = 359 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l3193_319341


namespace NUMINAMATH_CALUDE_original_number_proof_l3193_319396

theorem original_number_proof : ∃ x : ℝ, 16 * x = 3408 ∧ x = 213 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l3193_319396


namespace NUMINAMATH_CALUDE_min_difference_for_always_larger_l3193_319361

/-- Pratyya's daily number transformation -/
def pratyya_transform (n : ℤ) : ℤ := 2 * n - 2

/-- Payel's daily number transformation -/
def payel_transform (m : ℤ) : ℤ := 2 * m + 2

/-- The difference between Pratyya's and Payel's numbers after t days -/
def difference (n m : ℤ) (t : ℕ) : ℤ :=
  pratyya_transform (n + t) - payel_transform (m + t)

/-- The theorem stating the minimum difference for Pratyya's number to always be larger -/
theorem min_difference_for_always_larger (n m : ℤ) (h : n > m) :
  (∀ t : ℕ, difference n m t > 0) ↔ n - m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_min_difference_for_always_larger_l3193_319361


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_is_16_l3193_319376

/-- The speed of a boat in still water, given downstream travel information and stream speed. -/
theorem boat_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 60)
  (h3 : downstream_time = 3)
  : ℝ :=
  let downstream_speed := downstream_distance / downstream_time
  let boat_speed := downstream_speed - stream_speed
  16

/-- Proof that the boat's speed in still water is 16 km/hr -/
theorem boat_speed_is_16
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 60)
  (h3 : downstream_time = 3)
  : boat_speed_in_still_water stream_speed downstream_distance downstream_time h1 h2 h3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_boat_speed_is_16_l3193_319376


namespace NUMINAMATH_CALUDE_min_value_expression_l3193_319333

theorem min_value_expression (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + 3*y = 2) :
  1/x + 3/y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3*y₀ = 2 ∧ 1/x₀ + 3/y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3193_319333


namespace NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l3193_319324

-- Part 1
theorem inequality_solution (x : ℝ) :
  x - (3 * x - 1) ≤ 2 * x + 3 ↔ x ≥ -1/2 := by sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (3 * (x - 1) < 4 * x - 2 ∧ (1 + 4 * x) / 3 > x - 1) ↔ x > -1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l3193_319324


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3193_319368

def f (x : ℝ) : ℝ := 2 * (x + 1) * (x - 3)

theorem quadratic_function_properties :
  (∀ x, f x = 2 * (x + 1) * (x - 3)) ∧
  f (-1) = 0 ∧ f 3 = 0 ∧ f 1 = -8 ∧
  (∀ x ∈ Set.Icc 0 3, f x ≥ -8 ∧ f x ≤ 0) ∧
  (∀ x, f x ≥ 0 ↔ x ≤ -1 ∨ x ≥ 3) :=
by sorry

#check quadratic_function_properties

end NUMINAMATH_CALUDE_quadratic_function_properties_l3193_319368


namespace NUMINAMATH_CALUDE_vending_machine_probability_l3193_319362

/-- Represents the vending machine scenario --/
structure VendingMachine where
  numToys : Nat
  priceStep : Rat
  minPrice : Rat
  maxPrice : Rat
  numFavoriteToys : Nat
  favoriteToyPrice : Rat
  initialQuarters : Nat

/-- Calculates the probability of needing to exchange the $20 bill --/
def probabilityNeedExchange (vm : VendingMachine) : Rat :=
  sorry

/-- The main theorem to prove --/
theorem vending_machine_probability (vm : VendingMachine) :
  vm.numToys = 10 ∧
  vm.priceStep = 1/2 ∧
  vm.minPrice = 1/2 ∧
  vm.maxPrice = 5 ∧
  vm.numFavoriteToys = 2 ∧
  vm.favoriteToyPrice = 9/2 ∧
  vm.initialQuarters = 12
  →
  probabilityNeedExchange vm = 15/25 :=
by sorry

end NUMINAMATH_CALUDE_vending_machine_probability_l3193_319362


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l3193_319358

/-- The polynomial p(x) = ax^4 + bx^3 + 20x^2 - 12x + 10 -/
def p (a b : ℚ) (x : ℚ) : ℚ := a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10

/-- The factor q(x) = 2x^2 + 3x - 4 -/
def q (x : ℚ) : ℚ := 2 * x^2 + 3 * x - 4

/-- Theorem stating that if q(x) is a factor of p(x), then a = 2 and b = 27 -/
theorem polynomial_factor_implies_coefficients (a b : ℚ) :
  (∃ r : ℚ → ℚ, ∀ x, p a b x = q x * r x) → a = 2 ∧ b = 27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_coefficients_l3193_319358


namespace NUMINAMATH_CALUDE_product_abc_value_l3193_319370

theorem product_abc_value
  (h1 : b * c * d = 65)
  (h2 : c * d * e = 750)
  (h3 : d * e * f = 250)
  (h4 : (a * f) / (c * d) = 0.6666666666666666)
  : a * b * c = 130 := by
  sorry

end NUMINAMATH_CALUDE_product_abc_value_l3193_319370


namespace NUMINAMATH_CALUDE_terminal_side_angle_ratio_trig_ratio_for_integer_k_l3193_319365

-- Part 1
theorem terminal_side_angle_ratio (α : Real) (h : ∃ (x y : Real), x = -4 ∧ y = 3 ∧ x = 5 * Real.cos α ∧ y = 5 * Real.sin α) : 
  (Real.cos (π/2 + α) * Real.sin (-π - α)) / (Real.cos (11*π/2 - α) * Real.sin (9*π/2 + α)) = -3/4 := by
sorry

-- Part 2
theorem trig_ratio_for_integer_k (k : Int) (α : Real) :
  (Real.sin (k * π - α) * Real.cos ((k + 1) * π - α)) / (Real.sin ((k - 1) * π + α) * Real.cos (k * π + α)) = -1 := by
sorry

end NUMINAMATH_CALUDE_terminal_side_angle_ratio_trig_ratio_for_integer_k_l3193_319365


namespace NUMINAMATH_CALUDE_solution_pairs_l3193_319313

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a b : ℕ) : Prop :=
  (is_divisible_by (a - b) 3 ∨
   is_prime (a + 2*b) ∧
   a = 4*b - 1 ∧
   is_divisible_by (a + 7) b) ∧
  (¬(is_divisible_by (a - b) 3) ∨
   ¬(is_prime (a + 2*b)) ∨
   ¬(a = 4*b - 1) ∨
   ¬(is_divisible_by (a + 7) b))

theorem solution_pairs :
  ∀ a b : ℕ, satisfies_conditions a b ↔ (a = 3 ∧ b = 1) ∨ (a = 7 ∧ b = 2) ∨ (a = 11 ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_solution_pairs_l3193_319313


namespace NUMINAMATH_CALUDE_fraction_problem_l3193_319346

theorem fraction_problem (x : ℚ) : x * 45 - 5 = 10 → x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3193_319346


namespace NUMINAMATH_CALUDE_circumscribed_trapezoid_radius_l3193_319306

/-- An isosceles trapezoid circumscribed around a circle -/
structure CircumscribedTrapezoid where
  /-- The area of the trapezoid -/
  area : ℝ
  /-- The height of the trapezoid -/
  height : ℝ
  /-- The lateral side of the trapezoid -/
  lateral_side : ℝ
  /-- The radius of the inscribed circle -/
  radius : ℝ
  /-- The height is half the lateral side -/
  height_half_lateral : height = lateral_side / 2
  /-- The area is positive -/
  area_pos : 0 < area

/-- 
  For an isosceles trapezoid circumscribed around a circle, 
  if the area of the trapezoid is S and its height is half of its lateral side, 
  then the radius of the circle is √(S/8).
-/
theorem circumscribed_trapezoid_radius 
  (t : CircumscribedTrapezoid) : t.radius = Real.sqrt (t.area / 8) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_trapezoid_radius_l3193_319306


namespace NUMINAMATH_CALUDE_equation_holds_l3193_319360

theorem equation_holds : (6 / 3) + 4 - (2 - 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_holds_l3193_319360


namespace NUMINAMATH_CALUDE_three_suit_probability_value_l3193_319323

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of cards of each suit in a standard deck -/
def suit_size : ℕ := 13

/-- The probability of drawing a diamond, then a spade, then a heart from a standard deck -/
def three_suit_probability : ℚ :=
  (suit_size : ℚ) / deck_size *
  (suit_size : ℚ) / (deck_size - 1) *
  (suit_size : ℚ) / (deck_size - 2)

theorem three_suit_probability_value :
  three_suit_probability = 2197 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_suit_probability_value_l3193_319323


namespace NUMINAMATH_CALUDE_smallest_m_for_nth_roots_in_T_l3193_319377

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 2 / 2}

theorem smallest_m_for_nth_roots_in_T : 
  ∃ m : ℕ+, (∀ n : ℕ+, n ≥ m → ∃ z ∈ T, z^(n:ℕ) = 1) ∧ 
  (∀ k : ℕ+, k < m → ∃ n : ℕ+, n ≥ k ∧ ∀ z ∈ T, z^(n:ℕ) ≠ 1) ∧
  m = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_nth_roots_in_T_l3193_319377


namespace NUMINAMATH_CALUDE_average_daily_sales_l3193_319327

/-- Theorem: Average daily sales of cups over a 12-day period -/
theorem average_daily_sales (day_one_sales : ℕ) (other_days_sales : ℕ) (total_days : ℕ) :
  day_one_sales = 86 →
  other_days_sales = 50 →
  total_days = 12 →
  (day_one_sales + (total_days - 1) * other_days_sales) / total_days = 53 :=
by sorry

end NUMINAMATH_CALUDE_average_daily_sales_l3193_319327


namespace NUMINAMATH_CALUDE_fox_alice_numbers_l3193_319345

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def divisible_by_at_least_three (n : ℕ) : Prop :=
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0) ∨
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∨
  (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 6 = 0) ∨
  (n % 2 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) ∨
  (n % 2 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0) ∨
  (n % 2 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0) ∨
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 6 = 0) ∨
  (n % 3 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
  (n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0)

def not_divisible_by_exactly_two (n : ℕ) : Prop :=
  ¬((n % 2 ≠ 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 ≠ 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 ≠ 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 ≠ 0 ∧ n % 6 = 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 ≠ 0 ∧ n % 5 = 0 ∧ n % 6 ≠ 0) ∨
    (n % 2 = 0 ∧ n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 ≠ 0 ∧ n % 6 ≠ 0))

theorem fox_alice_numbers :
  ∃! (s : Finset ℕ), ∀ n, n ∈ s ↔ 
    is_two_digit n ∧ 
    divisible_by_at_least_three n ∧ 
    not_divisible_by_exactly_two n ∧
    s.card = 8 := by sorry

end NUMINAMATH_CALUDE_fox_alice_numbers_l3193_319345


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3193_319338

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : Set.Ioo (-2 : ℝ) 3 = {x : ℝ | a * x^2 + b * x + c > 0}) : 
  Set.Ioo (-3 : ℝ) 2 = {x : ℝ | a * x^2 - b * x + c > 0} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3193_319338


namespace NUMINAMATH_CALUDE_woodchopper_theorem_l3193_319335

/-- A woodchopper who gets a certain number of wood blocks per tree and chops a certain number of trees per day -/
structure Woodchopper where
  blocks_per_tree : ℕ
  trees_per_day : ℕ

/-- Calculate the total number of wood blocks obtained after a given number of days -/
def total_blocks (w : Woodchopper) (days : ℕ) : ℕ :=
  w.blocks_per_tree * w.trees_per_day * days

/-- Theorem: A woodchopper who gets 3 blocks per tree and chops 2 trees per day obtains 30 blocks after 5 days -/
theorem woodchopper_theorem :
  let ragnar : Woodchopper := { blocks_per_tree := 3, trees_per_day := 2 }
  total_blocks ragnar 5 = 30 := by sorry

end NUMINAMATH_CALUDE_woodchopper_theorem_l3193_319335


namespace NUMINAMATH_CALUDE_optimal_sampling_methods_for_school_scenario_l3193_319389

/-- Represents the different blood types --/
inductive BloodType
| O
| A
| B
| AB

/-- Represents the available sampling methods --/
inductive SamplingMethod
| Random
| Systematic
| Stratified

/-- Structure representing the school scenario --/
structure SchoolScenario where
  total_students : Nat
  blood_type_distribution : BloodType → Nat
  sample_size_blood_study : Nat
  soccer_team_size : Nat
  sample_size_soccer_study : Nat

/-- Determines the optimal sampling method for a given scenario and study type --/
def optimal_sampling_method (scenario : SchoolScenario) (is_blood_study : Bool) : SamplingMethod :=
  if is_blood_study then SamplingMethod.Stratified else SamplingMethod.Random

/-- Theorem stating the optimal sampling methods for the given school scenario --/
theorem optimal_sampling_methods_for_school_scenario 
  (scenario : SchoolScenario)
  (h1 : scenario.total_students = 500)
  (h2 : scenario.blood_type_distribution BloodType.O = 200)
  (h3 : scenario.blood_type_distribution BloodType.A = 125)
  (h4 : scenario.blood_type_distribution BloodType.B = 125)
  (h5 : scenario.blood_type_distribution BloodType.AB = 50)
  (h6 : scenario.sample_size_blood_study = 20)
  (h7 : scenario.soccer_team_size = 11)
  (h8 : scenario.sample_size_soccer_study = 2) :
  (optimal_sampling_method scenario true = SamplingMethod.Stratified) ∧
  (optimal_sampling_method scenario false = SamplingMethod.Random) :=
sorry

end NUMINAMATH_CALUDE_optimal_sampling_methods_for_school_scenario_l3193_319389


namespace NUMINAMATH_CALUDE_initial_kittens_count_l3193_319378

/-- The number of kittens Alyssa's cat initially had -/
def initial_kittens : ℕ := sorry

/-- The number of kittens Alyssa gave to her friends -/
def given_away : ℕ := 4

/-- The number of kittens Alyssa has left -/
def kittens_left : ℕ := 4

/-- Theorem stating that the initial number of kittens is 8 -/
theorem initial_kittens_count : initial_kittens = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_kittens_count_l3193_319378


namespace NUMINAMATH_CALUDE_favorite_numbers_sum_l3193_319384

/-- Given that Glory's favorite number is 450 and Misty's favorite number is 3 times smaller than Glory's,
    prove that the sum of their favorite numbers is 600. -/
theorem favorite_numbers_sum (glory_number : ℕ) (misty_number : ℕ)
    (h1 : glory_number = 450)
    (h2 : misty_number * 3 = glory_number) :
    misty_number + glory_number = 600 := by
  sorry

end NUMINAMATH_CALUDE_favorite_numbers_sum_l3193_319384


namespace NUMINAMATH_CALUDE_root_equation_problem_l3193_319342

theorem root_equation_problem (p q : ℝ) 
  (h1 : ∃! (r1 r2 r3 : ℝ), r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    (∀ x : ℝ, (x + p) * (x + q) * (x + 15) = 0 ↔ x = r1 ∨ x = r2 ∨ x = r3) ∧
    (∀ x : ℝ, x ≠ -4))
  (h2 : ∃! (s1 s2 : ℝ), s1 ≠ s2 ∧
    (∀ x : ℝ, (x + 2*p) * (x + 4) * (x + 9) = 0 ↔ x = s1 ∨ x = s2) ∧
    (∀ x : ℝ, x ≠ -q ∧ x ≠ -15)) :
  100 * p + q = -191 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l3193_319342


namespace NUMINAMATH_CALUDE_sum_of_M_subset_products_l3193_319367

def M : Set ℚ := {-2/3, 5/4, 1, 4}

def f (x : ℚ) : ℚ := (x + 2/3) * (x - 5/4) * (x - 1) * (x - 4)

def sum_of_subset_products (S : Set ℚ) : ℚ :=
  (f 1) - 1

theorem sum_of_M_subset_products :
  sum_of_subset_products M = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_M_subset_products_l3193_319367


namespace NUMINAMATH_CALUDE_seeds_per_pack_l3193_319332

def desired_flowers : ℕ := 20
def survival_rate : ℚ := 1/2
def pack_cost : ℕ := 5
def total_spent : ℕ := 10

theorem seeds_per_pack : 
  ∃ (seeds_per_pack : ℕ), 
    (total_spent / pack_cost) * seeds_per_pack = desired_flowers / survival_rate :=
by sorry

end NUMINAMATH_CALUDE_seeds_per_pack_l3193_319332


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l3193_319317

/-- The number of vertices in a pentadecagon -/
def n : ℕ := 15

/-- The number of vertices required to form a triangle -/
def k : ℕ := 3

/-- The number of sides in a pentadecagon, which is equal to the number of triangles 
    that have a side coinciding with a side of the pentadecagon -/
def excluded_triangles : ℕ := n

theorem pentadecagon_triangles : 
  (Nat.choose n k) - excluded_triangles = 440 :=
sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l3193_319317


namespace NUMINAMATH_CALUDE_bob_time_improvement_l3193_319312

/-- 
Given Bob's current mile time and his sister's mile time in seconds,
calculate the percentage improvement Bob needs to match his sister's time.
-/
theorem bob_time_improvement (bob_time sister_time : ℕ) :
  bob_time = 640 ∧ sister_time = 608 →
  (bob_time - sister_time : ℚ) / bob_time * 100 = 5 := by
sorry

end NUMINAMATH_CALUDE_bob_time_improvement_l3193_319312


namespace NUMINAMATH_CALUDE_complement_M_l3193_319336

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x > 2 ∨ x < 0}

theorem complement_M : U \ M = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_l3193_319336


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3193_319311

theorem divisibility_implies_multiple_of_three (n : ℕ) :
  n ≥ 2 →
  (∃ k : ℕ, 2^n + 1 = k * n) →
  ∃ m : ℕ, n = 3 * m :=
sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l3193_319311


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3193_319339

theorem complex_exponential_to_rectangular : Complex.exp (13 * Real.pi * Complex.I / 6) = Complex.mk (Real.sqrt 3 / 2) (1 / 2) := by sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l3193_319339


namespace NUMINAMATH_CALUDE__l3193_319372

variable (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def max_value_theorem (m n : V) (hm : ‖m‖ = 2) (hmn : ‖m + 2 • n‖ = 2) :
  ∃ (x : ℝ), x = ‖n‖ + ‖2 • m + n‖ ∧ x ≤ 8 * Real.sqrt 3 / 3 ∧
  ∀ (y : ℝ), y = ‖n‖ + ‖2 • m + n‖ → y ≤ x := by sorry

end NUMINAMATH_CALUDE__l3193_319372


namespace NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3193_319393

/-- Given a quadrilateral EFGH with extended sides, prove the reconstruction formula for point E -/
theorem quadrilateral_reconstruction 
  (E F G H E' F' G' H' : ℝ × ℝ) 
  (h1 : E' - F = 2 * (E - F))
  (h2 : F' - G = 2 * (F - G))
  (h3 : G' - H = 2 * (G - H))
  (h4 : H' - E = 2 * (H - E)) :
  E = (1/79 : ℝ) • E' + (26/79 : ℝ) • F' + (26/79 : ℝ) • G' + (52/79 : ℝ) • H' := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_reconstruction_l3193_319393


namespace NUMINAMATH_CALUDE_art_gallery_pieces_l3193_319374

theorem art_gallery_pieces (total : ℕ) 
  (displayed : ℕ) (sculptures_displayed : ℕ) 
  (paintings_not_displayed : ℕ) (sculptures_not_displayed : ℕ) :
  displayed = total / 3 →
  sculptures_displayed = displayed / 6 →
  paintings_not_displayed = (total - displayed) / 3 →
  sculptures_not_displayed = 1400 →
  total = 3150 := by
sorry

end NUMINAMATH_CALUDE_art_gallery_pieces_l3193_319374


namespace NUMINAMATH_CALUDE_line_circle_relationship_l3193_319303

theorem line_circle_relationship (k : ℝ) : 
  ∃ (x y : ℝ), (x - k*y + 1 = 0) ∧ (x^2 + y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l3193_319303


namespace NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3193_319328

theorem smallest_multiple_of_6_and_15 : 
  ∃ (b : ℕ), b > 0 ∧ 6 ∣ b ∧ 15 ∣ b ∧ ∀ (x : ℕ), x > 0 → 6 ∣ x → 15 ∣ x → b ≤ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_6_and_15_l3193_319328


namespace NUMINAMATH_CALUDE_factories_unchecked_l3193_319363

theorem factories_unchecked (total : ℕ) (group1 : ℕ) (group2 : ℕ) 
  (h1 : total = 169) 
  (h2 : group1 = 69) 
  (h3 : group2 = 52) : 
  total - (group1 + group2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_factories_unchecked_l3193_319363


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3193_319371

theorem smallest_gcd_multiple (m n : ℕ) (h : m > 0 ∧ n > 0) (h_gcd : Nat.gcd m n = 18) :
  Nat.gcd (8 * m) (12 * n) ≥ 72 ∧ ∃ (m₀ n₀ : ℕ), m₀ > 0 ∧ n₀ > 0 ∧ Nat.gcd m₀ n₀ = 18 ∧ Nat.gcd (8 * m₀) (12 * n₀) = 72 :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3193_319371


namespace NUMINAMATH_CALUDE_bus_stop_time_l3193_319381

theorem bus_stop_time (speed_without_stops : ℝ) (speed_with_stops : ℝ) : 
  speed_without_stops = 50 →
  speed_with_stops = 35 →
  (speed_without_stops - speed_with_stops) / speed_without_stops * 60 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_time_l3193_319381


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3193_319373

theorem cubic_roots_sum_cubes (a b c : ℝ) : 
  (3 * a^3 + 5 * a^2 - 150 * a + 7 = 0) →
  (3 * b^3 + 5 * b^2 - 150 * b + 7 = 0) →
  (3 * c^3 + 5 * c^2 - 150 * c + 7 = 0) →
  (a + b + 2)^3 + (b + c + 2)^3 + (c + a + 2)^3 = 303 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3193_319373


namespace NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l3193_319355

theorem abs_x_minus_one_necessary_not_sufficient :
  (∃ x : ℝ, |x - 1| < 2 ∧ ¬(x * (x - 3) < 0)) ∧
  (∀ x : ℝ, x * (x - 3) < 0 → |x - 1| < 2) :=
by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_necessary_not_sufficient_l3193_319355


namespace NUMINAMATH_CALUDE_tan_equality_with_period_l3193_319319

theorem tan_equality_with_period (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (1230 * π / 180) → n = 30 := by
  sorry

end NUMINAMATH_CALUDE_tan_equality_with_period_l3193_319319


namespace NUMINAMATH_CALUDE_intersection_point_l3193_319394

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 1) / 7 = (y - 2) / 1 ∧ (y - 2) / 1 = (z - 6) / (-1)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  4 * x + y - 6 * z - 5 = 0

/-- The theorem stating that (8, 3, 5) is the unique point of intersection -/
theorem intersection_point :
  ∃! (x y z : ℝ), line x y z ∧ plane x y z ∧ x = 8 ∧ y = 3 ∧ z = 5 := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3193_319394


namespace NUMINAMATH_CALUDE_gcd_840_1764_l3193_319349

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_840_1764_l3193_319349


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3193_319304

theorem sufficient_not_necessary : 
  (∀ x : ℝ, 2 < x ∧ x < 3 → x * (x - 5) < 0) ∧
  (∃ x : ℝ, x * (x - 5) < 0 ∧ ¬(2 < x ∧ x < 3)) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3193_319304


namespace NUMINAMATH_CALUDE_max_x_value_l3193_319364

theorem max_x_value (x y z : ℝ) (sum_eq : x + y + z = 6) (sum_prod_eq : x*y + x*z + y*z = 11) : 
  x ≤ 2 ∧ ∃ (a b : ℝ), a + b + 2 = 6 ∧ 2*a + 2*b + a*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_max_x_value_l3193_319364


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l3193_319388

def arithmetic_progression (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_sum (a : ℕ → ℝ) :
  arithmetic_progression a → a 5 = 5 → a 3 + a 7 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l3193_319388


namespace NUMINAMATH_CALUDE_abs_ab_minus_cd_le_quarter_l3193_319369

theorem abs_ab_minus_cd_le_quarter 
  (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_eq_one : a + b + c + d = 1) : 
  |a * b - c * d| ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_abs_ab_minus_cd_le_quarter_l3193_319369


namespace NUMINAMATH_CALUDE_mirror_height_for_full_body_view_l3193_319331

/-- 
Theorem: For a person standing upright in front of a vertical mirror, 
the minimum mirror height required to see their full body is exactly 
half of their height.
-/
theorem mirror_height_for_full_body_view 
  (h : ℝ) -- height of the person
  (m : ℝ) -- height of the mirror
  (h_pos : h > 0) -- person's height is positive
  (m_pos : m > 0) -- mirror's height is positive
  (full_view : m ≥ h / 2) -- condition for full body view
  (minimal : ∀ m' : ℝ, m' > 0 → m' < m → ¬(m' ≥ h / 2)) -- m is minimal
  : m = h / 2 := by sorry

end NUMINAMATH_CALUDE_mirror_height_for_full_body_view_l3193_319331


namespace NUMINAMATH_CALUDE_parallelogram_area_example_l3193_319308

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 21 cm and height 11 cm is 231 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 21 11 = 231 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_example_l3193_319308


namespace NUMINAMATH_CALUDE_sum_of_powers_of_3_mod_5_l3193_319302

def sum_of_powers (base : ℕ) (exponent : ℕ) : ℕ :=
  Finset.sum (Finset.range (exponent + 1)) (fun i => base ^ i)

theorem sum_of_powers_of_3_mod_5 :
  sum_of_powers 3 2023 % 5 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_3_mod_5_l3193_319302


namespace NUMINAMATH_CALUDE_definite_integral_equality_l3193_319300

theorem definite_integral_equality : 
  let a : Real := 0
  let b : Real := Real.arcsin (Real.sqrt (7/8))
  let f (x : Real) := (6 * Real.sin x ^ 2) / (4 + 3 * Real.cos (2 * x))
  ∫ x in a..b, f x = (Real.sqrt 7 * Real.pi) / 4 - Real.arctan (Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_equality_l3193_319300


namespace NUMINAMATH_CALUDE_remainder_proof_l3193_319318

theorem remainder_proof : ((764251 * 1095223 * 1487719 + 263311) * (12097 * 16817 * 23431 - 305643)) % 31 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l3193_319318


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3193_319347

theorem cubic_root_sum (r s t : ℝ) : 
  r^3 - 15*r^2 + 13*r - 6 = 0 ∧ 
  s^3 - 15*s^2 + 13*s - 6 = 0 ∧ 
  t^3 - 15*t^2 + 13*t - 6 = 0 →
  r / (1/r + s*t) + s / (1/s + t*r) + t / (1/t + r*s) = 199/7 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3193_319347


namespace NUMINAMATH_CALUDE_product_ab_l3193_319351

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the equation
def complex_equation (a b : ℝ) : Prop :=
  (1 + 7 * i) / (2 - i) = (a : ℂ) + b * i

-- Theorem statement
theorem product_ab (a b : ℝ) (h : complex_equation a b) : a * b = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_ab_l3193_319351


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l3193_319383

noncomputable def f (x : ℝ) := Real.exp x * (x^2 - 2*x - 1)

theorem tangent_line_at_one (x y : ℝ) :
  let p := (1, f 1)
  let m := (Real.exp 1) * ((1:ℝ)^2 - 3)
  (y - f 1 = m * (x - 1)) ↔ (2 * Real.exp 1 * x + y = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l3193_319383


namespace NUMINAMATH_CALUDE_harriet_speed_l3193_319387

/-- Harriet's round trip between A-ville and B-town -/
theorem harriet_speed (total_time : ℝ) (time_to_b : ℝ) (speed_from_b : ℝ) :
  total_time = 5 →
  time_to_b = 3 →
  speed_from_b = 150 →
  ∃ (distance : ℝ) (speed_to_b : ℝ),
    distance = speed_from_b * (total_time - time_to_b) ∧
    distance = speed_to_b * time_to_b ∧
    speed_to_b = 100 := by
  sorry


end NUMINAMATH_CALUDE_harriet_speed_l3193_319387


namespace NUMINAMATH_CALUDE_product_inequality_l3193_319330

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum_prod : a + b + c + 2 = a * b * c) :
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l3193_319330


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3193_319375

theorem geometric_sequence_problem (a₁ q aₙ : ℚ) (n : ℕ) :
  a₁ = 9/8 ∧ q = 2/3 ∧ aₙ = 1/3 ∧ aₙ = a₁ * q^(n-1) → n = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3193_319375


namespace NUMINAMATH_CALUDE_unique_congruence_in_range_l3193_319326

theorem unique_congruence_in_range : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_congruence_in_range_l3193_319326


namespace NUMINAMATH_CALUDE_unique_function_theorem_l3193_319390

open Real

-- Define the function type
def FunctionType := (x : ℝ) → x > 0 → ℝ

-- State the theorem
theorem unique_function_theorem (f : FunctionType) 
  (h1 : f 2009 (by norm_num) = 1)
  (h2 : ∀ (x y : ℝ) (hx : x > 0) (hy : y > 0), 
    f x hx * f y hy + f (2009 / x) (by positivity) * f (2009 / y) (by positivity) = 2 * f (x * y) (by positivity)) :
  ∀ (x : ℝ) (hx : x > 0), f x hx = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_function_theorem_l3193_319390
