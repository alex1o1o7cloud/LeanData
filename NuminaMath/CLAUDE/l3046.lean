import Mathlib

namespace NUMINAMATH_CALUDE_sixty_eighth_digit_of_largest_n_l3046_304650

def largest_n : ℕ := (10^100 - 1) / 14

def digit_at_position (n : ℕ) (pos : ℕ) : ℕ :=
  (n / 10^(pos - 1)) % 10

theorem sixty_eighth_digit_of_largest_n :
  digit_at_position largest_n 68 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sixty_eighth_digit_of_largest_n_l3046_304650


namespace NUMINAMATH_CALUDE_annual_pension_calculation_l3046_304607

/-- Represents the annual pension calculation for a retiring employee. -/
theorem annual_pension_calculation
  (c d r s : ℝ)
  (h_cd : d ≠ c)
  (h_positive : c > 0 ∧ d > 0 ∧ r > 0 ∧ s > 0)
  (h_prop : ∃ (k x : ℝ), k > 0 ∧ x > 0 ∧
    k * (x + c)^(3/2) = k * x^(3/2) + r ∧
    k * (x + d)^(3/2) = k * x^(3/2) + s) :
  ∃ (pension : ℝ), pension = (4 * r^2) / (9 * c^2) :=
sorry

end NUMINAMATH_CALUDE_annual_pension_calculation_l3046_304607


namespace NUMINAMATH_CALUDE_joan_balloons_l3046_304681

theorem joan_balloons (initial_balloons : ℕ) (lost_balloons : ℕ) (remaining_balloons : ℕ) : 
  initial_balloons = 9 → lost_balloons = 2 → remaining_balloons = initial_balloons - lost_balloons →
  remaining_balloons = 7 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloons_l3046_304681


namespace NUMINAMATH_CALUDE_census_suitability_l3046_304696

-- Define the type for survey options
inductive SurveyOption
| A : SurveyOption  -- Favorite TV programs of middle school students
| B : SurveyOption  -- Printing errors on a certain exam paper
| C : SurveyOption  -- Survey on the service life of batteries
| D : SurveyOption  -- Internet usage of middle school students

-- Define what it means for a survey to be suitable for a census
def suitableForCensus (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.B => True
  | _ => False

-- Define the property of examining every item in a population
def examinesEveryItem (s : SurveyOption) : Prop :=
  match s with
  | SurveyOption.B => True
  | _ => False

-- Theorem statement
theorem census_suitability :
  ∀ s : SurveyOption, suitableForCensus s ↔ examinesEveryItem s :=
by sorry

end NUMINAMATH_CALUDE_census_suitability_l3046_304696


namespace NUMINAMATH_CALUDE_johns_bonus_last_year_l3046_304604

/-- Represents John's yearly financial information -/
structure YearlyFinance where
  salary : ℝ
  bonus_percentage : ℝ
  total_income : ℝ

/-- Calculates the bonus amount given a salary and bonus percentage -/
def calculate_bonus (salary : ℝ) (bonus_percentage : ℝ) : ℝ :=
  salary * bonus_percentage

theorem johns_bonus_last_year 
  (last_year : YearlyFinance)
  (this_year : YearlyFinance)
  (h1 : last_year.salary = 100000)
  (h2 : this_year.salary = 200000)
  (h3 : this_year.total_income = 220000)
  (h4 : last_year.bonus_percentage = this_year.bonus_percentage) :
  calculate_bonus last_year.salary last_year.bonus_percentage = 10000 := by
sorry

end NUMINAMATH_CALUDE_johns_bonus_last_year_l3046_304604


namespace NUMINAMATH_CALUDE_normal_distribution_mean_l3046_304684

/-- 
Given a normal distribution with standard deviation σ,
if the value that is exactly k standard deviations less than the mean is x,
then the arithmetic mean μ of the distribution is x + k * σ.
-/
theorem normal_distribution_mean 
  (σ : ℝ) (k : ℝ) (x : ℝ) (μ : ℝ) 
  (hσ : σ = 1.5) 
  (hk : k = 2) 
  (hx : x = 11.5) 
  (h : x = μ - k * σ) : 
  μ = 14.5 := by
  sorry

#check normal_distribution_mean

end NUMINAMATH_CALUDE_normal_distribution_mean_l3046_304684


namespace NUMINAMATH_CALUDE_isosceles_triangle_larger_angle_l3046_304627

/-- The measure of a right angle in degrees -/
def right_angle : ℝ := 90

/-- An isosceles triangle with one angle 20% smaller than a right angle -/
structure IsoscelesTriangle where
  /-- The measure of the smallest angle in degrees -/
  small_angle : ℝ
  /-- The measure of one of the two equal larger angles in degrees -/
  large_angle : ℝ
  /-- The triangle is isosceles with two equal larger angles -/
  isosceles : large_angle = large_angle
  /-- The small angle is 20% smaller than a right angle -/
  small_angle_def : small_angle = right_angle * (1 - 0.2)
  /-- The sum of all angles in the triangle is 180° -/
  angle_sum : small_angle + 2 * large_angle = 180

/-- Theorem: In an isosceles triangle where one angle is 20% smaller than a right angle,
    each of the two equal larger angles measures 54° -/
theorem isosceles_triangle_larger_angle (t : IsoscelesTriangle) : t.large_angle = 54 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_larger_angle_l3046_304627


namespace NUMINAMATH_CALUDE_seconds_in_12_5_minutes_l3046_304646

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- The number of minutes to convert -/
def minutes_to_convert : ℚ := 12.5

/-- Theorem: The number of seconds in 12.5 minutes is 750 -/
theorem seconds_in_12_5_minutes :
  (minutes_to_convert * seconds_per_minute : ℚ) = 750 := by sorry

end NUMINAMATH_CALUDE_seconds_in_12_5_minutes_l3046_304646


namespace NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l3046_304686

theorem floor_plus_x_eq_seventeen_fourths :
  ∃ (x : ℚ), (⌊x⌋ : ℚ) + x = 17 / 4 ∧ x = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_plus_x_eq_seventeen_fourths_l3046_304686


namespace NUMINAMATH_CALUDE_large_square_area_l3046_304689

theorem large_square_area (s : ℝ) (S : ℝ) 
  (h1 : S = s + 20)
  (h2 : S^2 - s^2 = 880) :
  S^2 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_large_square_area_l3046_304689


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3046_304693

/-- The surface area of a sphere with radius 14 meters is 4 * π * 14^2 square meters. -/
theorem sphere_surface_area :
  let r : ℝ := 14
  4 * Real.pi * r^2 = 4 * Real.pi * 14^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l3046_304693


namespace NUMINAMATH_CALUDE_race_total_time_l3046_304648

theorem race_total_time (total_runners : Nat) (fast_runners : Nat) (fast_time : Nat) (extra_time : Nat) :
  total_runners = 8 →
  fast_runners = 5 →
  fast_time = 8 →
  extra_time = 2 →
  (fast_runners * fast_time) + ((total_runners - fast_runners) * (fast_time + extra_time)) = 70 := by
  sorry

end NUMINAMATH_CALUDE_race_total_time_l3046_304648


namespace NUMINAMATH_CALUDE_equilateral_triangle_division_l3046_304667

/-- Given an equilateral triangle with sides divided into three equal parts and an inner equilateral
    triangle formed by connecting corresponding division points, if the inscribed circle in the inner
    triangle has radius 6 cm, then the side length of the inner triangle is 12√3 cm and the side
    length of the outer triangle is 36 cm. -/
theorem equilateral_triangle_division (r : ℝ) (inner_side outer_side : ℝ) :
  r = 6 →
  inner_side = 12 * Real.sqrt 3 →
  outer_side = 36 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_division_l3046_304667


namespace NUMINAMATH_CALUDE_Z_in_first_quadrant_l3046_304642

def Z : ℂ := (5 + 4*Complex.I) + (-1 + 2*Complex.I)

theorem Z_in_first_quadrant : 
  Z.re > 0 ∧ Z.im > 0 := by sorry

end NUMINAMATH_CALUDE_Z_in_first_quadrant_l3046_304642


namespace NUMINAMATH_CALUDE_quadratic_expression_minimum_l3046_304660

theorem quadratic_expression_minimum :
  ∀ x y : ℝ, x^2 + 4*x*y + 5*y^2 - 8*x - 6*y ≥ -22 ∧
  ∃ x y : ℝ, x^2 + 4*x*y + 5*y^2 - 8*x - 6*y = -22 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_expression_minimum_l3046_304660


namespace NUMINAMATH_CALUDE_problem_solution_l3046_304624

theorem problem_solution (a b c d : ℝ) (h1 : a - b = -3) (h2 : c + d = 2) :
  (b + c) - (a - d) = 5 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l3046_304624


namespace NUMINAMATH_CALUDE_savings_increase_percentage_l3046_304690

theorem savings_increase_percentage (I : ℝ) (I_pos : I > 0) : 
  let regular_expense_ratio : ℝ := 0.75
  let additional_expense_ratio : ℝ := 0.10
  let income_increase_ratio : ℝ := 0.20
  let regular_expense_increase_ratio : ℝ := 0.10
  let additional_expense_increase_ratio : ℝ := 0.25
  
  let initial_savings := I * (1 - regular_expense_ratio - additional_expense_ratio)
  let new_income := I * (1 + income_increase_ratio)
  let new_regular_expense := I * regular_expense_ratio * (1 + regular_expense_increase_ratio)
  let new_additional_expense := I * additional_expense_ratio * (1 + additional_expense_increase_ratio)
  let new_savings := new_income - new_regular_expense - new_additional_expense
  
  (new_savings - initial_savings) / initial_savings = 2/3 :=
by
  sorry

end NUMINAMATH_CALUDE_savings_increase_percentage_l3046_304690


namespace NUMINAMATH_CALUDE_count_valid_distributions_l3046_304680

/-- Represents an envelope containing two cards -/
def Envelope := Fin 6 × Fin 6

/-- Represents a valid distribution of cards into envelopes -/
def ValidDistribution := { d : Fin 3 → Envelope // 
  (∀ i j : Fin 3, i ≠ j → d i ≠ d j) ∧ 
  (∃ i : Fin 3, d i = ⟨1, 2⟩ ∨ d i = ⟨2, 1⟩) }

/-- The number of valid distributions -/
def numValidDistributions : ℕ := sorry

theorem count_valid_distributions : numValidDistributions = 18 := by sorry

end NUMINAMATH_CALUDE_count_valid_distributions_l3046_304680


namespace NUMINAMATH_CALUDE_complement_of_M_l3046_304606

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | |x| > 2}

-- State the theorem
theorem complement_of_M :
  Mᶜ = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3046_304606


namespace NUMINAMATH_CALUDE_jenny_pokemon_cards_l3046_304647

theorem jenny_pokemon_cards (J : ℕ) : 
  J + (J + 2) + 3 * (J + 2) = 38 → J = 6 := by
  sorry

end NUMINAMATH_CALUDE_jenny_pokemon_cards_l3046_304647


namespace NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l3046_304638

theorem cos_pi_third_minus_alpha (α : Real) 
  (h : Real.sin (π / 6 + α) = 2 / 3) : 
  Real.cos (π / 3 - α) = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_third_minus_alpha_l3046_304638


namespace NUMINAMATH_CALUDE_reciprocal_of_2023_l3046_304682

theorem reciprocal_of_2023 : (2023⁻¹ : ℚ) = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_2023_l3046_304682


namespace NUMINAMATH_CALUDE_intersection_A_B_l3046_304620

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3046_304620


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3046_304632

/-- Represents a repeating decimal with a single digit repeating -/
def repeating_decimal (n : ℕ) : ℚ := n / 9

theorem repeating_decimal_sum : 
  2 * (repeating_decimal 8 - repeating_decimal 2 + repeating_decimal 4) = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3046_304632


namespace NUMINAMATH_CALUDE_number_percentage_equality_l3046_304626

theorem number_percentage_equality (x : ℚ) : 
  (35 / 100) * x = (15 / 100) * 40 → x = 17 + 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_percentage_equality_l3046_304626


namespace NUMINAMATH_CALUDE_susan_first_turn_l3046_304633

/-- The number of spaces on the board game --/
def total_spaces : ℕ := 48

/-- The number of spaces Susan moves on the first turn --/
def first_turn : ℕ := sorry

/-- The net movement on the second turn --/
def second_turn : ℤ := 2 - 5

/-- The movement on the third turn --/
def third_turn : ℕ := 6

/-- The remaining spaces to win after three turns --/
def remaining_spaces : ℕ := 37

/-- Theorem stating that Susan moved 8 spaces on the first turn --/
theorem susan_first_turn : first_turn = 8 := by sorry

end NUMINAMATH_CALUDE_susan_first_turn_l3046_304633


namespace NUMINAMATH_CALUDE_earth_pile_fraction_l3046_304668

theorem earth_pile_fraction (P : ℚ) (P_pos : P > 0) : 
  P * (1 - 1/2) * (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * (1 - 1/7) = P * (1/7) := by
  sorry

end NUMINAMATH_CALUDE_earth_pile_fraction_l3046_304668


namespace NUMINAMATH_CALUDE_sin_cos_sum_equality_l3046_304602

theorem sin_cos_sum_equality : 
  Real.sin (20 * π / 180) * Real.cos (40 * π / 180) + 
  Real.cos (20 * π / 180) * Real.sin (140 * π / 180) = 
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equality_l3046_304602


namespace NUMINAMATH_CALUDE_set_of_a_values_l3046_304669

theorem set_of_a_values (a : ℝ) : 
  (2 ∉ {x : ℝ | x - a < 0}) ↔ (a ∈ {a : ℝ | a ≤ 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_of_a_values_l3046_304669


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3046_304623

theorem imaginary_power_sum (i : ℂ) (hi : i^2 = -1) :
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3046_304623


namespace NUMINAMATH_CALUDE_prob_at_least_two_correct_value_l3046_304643

/-- The number of questions Jessica randomly guesses -/
def n : ℕ := 6

/-- The number of possible answers for each question -/
def m : ℕ := 3

/-- The probability of guessing a single question correctly -/
def p : ℚ := 1 / m

/-- The probability of guessing a single question incorrectly -/
def q : ℚ := 1 - p

/-- The probability of getting at least two correct answers out of n randomly guessed questions -/
def prob_at_least_two_correct : ℚ :=
  1 - (q ^ n + n * p * q ^ (n - 1))

theorem prob_at_least_two_correct_value : 
  prob_at_least_two_correct = 473 / 729 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_two_correct_value_l3046_304643


namespace NUMINAMATH_CALUDE_max_value_of_x_l3046_304622

/-- Given a > 0 and b > 0, x is defined as the minimum of {1, a, b / (a² + b²)}.
    This theorem states that the maximum value of x is √2 / 2. -/
theorem max_value_of_x (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let x := min 1 (min a (b / (a^2 + b^2)))
  ∃ (max_x : ℝ), max_x = Real.sqrt 2 / 2 ∧ ∀ (a' b' : ℝ), a' > 0 → b' > 0 →
    min 1 (min a' (b' / (a'^2 + b'^2))) ≤ max_x :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_x_l3046_304622


namespace NUMINAMATH_CALUDE_workers_days_per_week_l3046_304651

/-- The number of toys produced per week -/
def weekly_production : ℕ := 5500

/-- The number of toys produced per day -/
def daily_production : ℕ := 1100

/-- The number of days worked per week -/
def days_worked : ℕ := weekly_production / daily_production

theorem workers_days_per_week :
  days_worked = 5 :=
sorry

end NUMINAMATH_CALUDE_workers_days_per_week_l3046_304651


namespace NUMINAMATH_CALUDE_well_depth_calculation_l3046_304610

/-- The depth of the well in feet -/
def well_depth : ℝ := 1255.64

/-- The time it takes for the stone to hit the bottom and the sound to reach the top -/
def total_time : ℝ := 10

/-- The gravitational constant for the stone's fall -/
def gravity_constant : ℝ := 16

/-- The velocity of sound in feet per second -/
def sound_velocity : ℝ := 1100

/-- The function describing the stone's fall distance after t seconds -/
def stone_fall (t : ℝ) : ℝ := gravity_constant * t^2

/-- Theorem stating that the calculated well depth is correct given the conditions -/
theorem well_depth_calculation :
  ∃ (t_fall : ℝ), 
    t_fall > 0 ∧ 
    t_fall + (well_depth / sound_velocity) = total_time ∧ 
    stone_fall t_fall = well_depth := by
  sorry

end NUMINAMATH_CALUDE_well_depth_calculation_l3046_304610


namespace NUMINAMATH_CALUDE_simplify_nested_roots_l3046_304694

theorem simplify_nested_roots (b : ℝ) (hb : b > 0) :
  (((b ^ 16) ^ (1 / 8)) ^ (1 / 4)) ^ 3 * (((b ^ 16) ^ (1 / 4)) ^ (1 / 8)) ^ 3 = b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_roots_l3046_304694


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l3046_304629

theorem inverse_variation_problem (z w : ℝ) (k : ℝ) (h1 : z * Real.sqrt w = k) 
  (h2 : 6 * Real.sqrt 3 = k) (h3 : z = 3/2) : w = 48 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l3046_304629


namespace NUMINAMATH_CALUDE_error_permutations_l3046_304644

/-- The number of incorrect permutations of the letters in "error" -/
def incorrect_permutations : ℕ :=
  Nat.factorial 5 / Nat.factorial 3 - 1

/-- The word "error" has 5 letters -/
def word_length : ℕ := 5

/-- The letter 'r' is repeated three times -/
def r_count : ℕ := 3

/-- The letters 'e' and 'o' appear once each -/
def unique_letters : ℕ := 2

theorem error_permutations :
  incorrect_permutations = (Nat.factorial word_length / Nat.factorial r_count) - 1 :=
by sorry

end NUMINAMATH_CALUDE_error_permutations_l3046_304644


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l3046_304662

theorem complex_equation_modulus : ∀ (x y : ℝ),
  (Complex.I * (x + 2 * Complex.I) = y - Complex.I) →
  Complex.abs (x - y * Complex.I) = Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l3046_304662


namespace NUMINAMATH_CALUDE_tara_bank_balance_l3046_304685

/-- Calculates the balance after one year given an initial amount and an annual interest rate. -/
def balance_after_one_year (initial_amount : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_amount * (1 + interest_rate)

/-- Theorem stating that with an initial amount of $90 and a 10% annual interest rate, 
    the balance after one year will be $99. -/
theorem tara_bank_balance : 
  balance_after_one_year 90 0.1 = 99 := by
  sorry

end NUMINAMATH_CALUDE_tara_bank_balance_l3046_304685


namespace NUMINAMATH_CALUDE_largest_valid_coloring_l3046_304665

/-- A coloring of an n × n grid with two colors. -/
def Coloring (n : ℕ) := Fin n → Fin n → Bool

/-- Checks if a rectangle in the grid has all four corners the same color. -/
def hasMonochromaticRectangle (c : Coloring n) : Prop :=
  ∃ (i j k l : Fin n), i < k ∧ j < l ∧
    c i j = c i l ∧ c i l = c k j ∧ c k j = c k l

/-- The largest n for which a valid coloring exists. -/
def largestValidN : ℕ := 4

theorem largest_valid_coloring :
  (∃ (c : Coloring largestValidN), ¬hasMonochromaticRectangle c) ∧
  (∀ (m : ℕ), m > largestValidN →
    ∀ (c : Coloring m), hasMonochromaticRectangle c) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_coloring_l3046_304665


namespace NUMINAMATH_CALUDE_exactly_fourteen_plus_signs_l3046_304674

/-- Represents a board with plus and minus signs -/
structure SignBoard where
  total_symbols : ℕ
  plus_signs : ℕ
  minus_signs : ℕ
  total_is_sum : total_symbols = plus_signs + minus_signs

/-- Predicate to check if any subset of size n contains at least one plus sign -/
def has_plus_in_subset (board : SignBoard) (n : ℕ) : Prop :=
  board.minus_signs < n

/-- Predicate to check if any subset of size n contains at least one minus sign -/
def has_minus_in_subset (board : SignBoard) (n : ℕ) : Prop :=
  board.plus_signs < n

/-- The main theorem to prove -/
theorem exactly_fourteen_plus_signs (board : SignBoard) 
  (h_total : board.total_symbols = 23)
  (h_plus_10 : has_plus_in_subset board 10)
  (h_minus_15 : has_minus_in_subset board 15) :
  board.plus_signs = 14 :=
sorry

end NUMINAMATH_CALUDE_exactly_fourteen_plus_signs_l3046_304674


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l3046_304640

/-- Given a boat that travels 13 km along a stream and 5 km against the same stream
    in one hour each, its speed in still water is 9 km/hr. -/
theorem boat_speed_in_still_water
  (along_stream : ℝ) (against_stream : ℝ)
  (h_along : along_stream = 13)
  (h_against : against_stream = 5)
  (h_time : along_stream = (boat_speed + stream_speed) * 1 ∧
            against_stream = (boat_speed - stream_speed) * 1)
  : boat_speed = 9 :=
by
  sorry

#check boat_speed_in_still_water

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l3046_304640


namespace NUMINAMATH_CALUDE_apple_calculation_correct_l3046_304645

/-- Represents the weight difference from the standard weight and its frequency --/
structure WeightDifference :=
  (difference : ℝ)
  (frequency : ℕ)

/-- Calculates the total weight and profit from a batch of apples --/
def apple_calculation (total_boxes : ℕ) (price_per_box : ℝ) (selling_price_per_kg : ℝ) 
  (weight_differences : List WeightDifference) : ℝ × ℝ :=
  sorry

/-- The main theorem stating the correctness of the calculation --/
theorem apple_calculation_correct : 
  let weight_differences := [
    ⟨-0.2, 5⟩, ⟨-0.1, 8⟩, ⟨0, 2⟩, ⟨0.1, 6⟩, ⟨0.2, 8⟩, ⟨0.5, 1⟩
  ]
  let (total_weight, profit) := apple_calculation 400 60 10 weight_differences
  total_weight = 300.9 ∧ profit = 16120 :=
by sorry

end NUMINAMATH_CALUDE_apple_calculation_correct_l3046_304645


namespace NUMINAMATH_CALUDE_johns_weekly_consumption_l3046_304616

/-- Represents John's daily beverage consumption --/
structure DailyConsumption where
  water : ℝ  -- in gallons
  milk : ℝ   -- in pints
  juice : ℝ  -- in fluid ounces

/-- Conversion factors --/
def gallon_to_quart : ℝ := 4
def pint_to_quart : ℝ := 0.5
def floz_to_quart : ℝ := 0.03125

/-- John's daily consumption --/
def johns_consumption : DailyConsumption := {
  water := 1.5,
  milk := 3,
  juice := 20
}

/-- Number of days in a week --/
def days_in_week : ℕ := 7

/-- Theorem stating John's weekly beverage consumption in quarts --/
theorem johns_weekly_consumption :
  (johns_consumption.water * gallon_to_quart +
   johns_consumption.milk * pint_to_quart +
   johns_consumption.juice * floz_to_quart) * days_in_week = 56.875 := by
  sorry

end NUMINAMATH_CALUDE_johns_weekly_consumption_l3046_304616


namespace NUMINAMATH_CALUDE_rectangular_prism_sum_l3046_304655

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ

/-- The number of edges in a rectangular prism -/
def num_edges (p : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (p : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (p : RectangularPrism) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces of any rectangular prism is 26 -/
theorem rectangular_prism_sum (p : RectangularPrism) :
  num_edges p + num_corners p + num_faces p = 26 := by
  sorry

#check rectangular_prism_sum

end NUMINAMATH_CALUDE_rectangular_prism_sum_l3046_304655


namespace NUMINAMATH_CALUDE_expected_score_is_seven_sixths_l3046_304618

/-- Represents the score obtained from a single die roll -/
inductive Score
| one
| two
| three

/-- The probability of getting each score -/
def prob (s : Score) : ℚ :=
  match s with
  | Score.one => 1/2
  | Score.two => 1/3
  | Score.three => 1/6

/-- The point value associated with each score -/
def value (s : Score) : ℕ :=
  match s with
  | Score.one => 1
  | Score.two => 2
  | Score.three => 3

/-- The expected score for a single roll of the die -/
def expected_score : ℚ :=
  (prob Score.one * value Score.one) +
  (prob Score.two * value Score.two) +
  (prob Score.three * value Score.three)

theorem expected_score_is_seven_sixths :
  expected_score = 7/6 := by
  sorry

end NUMINAMATH_CALUDE_expected_score_is_seven_sixths_l3046_304618


namespace NUMINAMATH_CALUDE_factorization_equality_l3046_304613

theorem factorization_equality (a b : ℝ) : 5 * a^2 * b - 20 * b^3 = 5 * b * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3046_304613


namespace NUMINAMATH_CALUDE_j20_most_suitable_for_census_l3046_304601

/-- Represents a survey option -/
inductive SurveyOption
  | HuaweiPhoneBattery
  | J20Components
  | SpringFestivalMovie
  | HomeworkTime

/-- Determines if a survey option is suitable for a comprehensive survey (census) -/
def isSuitableForCensus (option : SurveyOption) : Prop :=
  match option with
  | SurveyOption.HuaweiPhoneBattery => False
  | SurveyOption.J20Components => True
  | SurveyOption.SpringFestivalMovie => False
  | SurveyOption.HomeworkTime => False

/-- Theorem stating that the J20Components survey is the most suitable for a comprehensive survey -/
theorem j20_most_suitable_for_census :
  ∀ (option : SurveyOption),
    option ≠ SurveyOption.J20Components →
    ¬(isSuitableForCensus option) ∧ isSuitableForCensus SurveyOption.J20Components :=
by sorry

end NUMINAMATH_CALUDE_j20_most_suitable_for_census_l3046_304601


namespace NUMINAMATH_CALUDE_gcd_of_sequence_is_three_l3046_304659

def a (n : ℕ) : ℕ := (2*n - 1) * (2*n + 1) * (2*n + 3)

theorem gcd_of_sequence_is_three :
  ∃ d : ℕ, d > 0 ∧ 
  (∀ k : ℕ, k ≥ 1 → k ≤ 2008 → d ∣ a k) ∧
  (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≥ 1 → k ≤ 2008 → m ∣ a k) → m ≤ d) ∧
  d = 3 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_sequence_is_three_l3046_304659


namespace NUMINAMATH_CALUDE_marble_calculation_l3046_304676

/-- Calculate the final number of marbles and prove its square root is 7 --/
theorem marble_calculation (initial : ℕ) (triple : ℕ → ℕ) (add : ℕ → ℕ → ℕ) 
  (lose_percent : ℕ → ℚ → ℕ) (find : ℕ → ℕ → ℕ) : 
  initial = 16 →
  (∀ x, triple x = 3 * x) →
  (∀ x y, add x y = x + y) →
  (∀ x p, lose_percent x p = x - ⌊(p * x : ℚ)⌋) →
  (∀ x y, find x y = x + y) →
  ∃ (final : ℕ), final = find (lose_percent (add (triple initial) 10) (1/4)) 5 ∧ 
  (final : ℝ).sqrt = 7 := by
sorry

end NUMINAMATH_CALUDE_marble_calculation_l3046_304676


namespace NUMINAMATH_CALUDE_problem_statement_l3046_304657

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 3)^2 = 0) :
  (a + b)^2015 = 1 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l3046_304657


namespace NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3046_304672

theorem tangent_line_to_logarithmic_curve (a : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    y₀ = x₀ + 1 ∧ 
    y₀ = Real.log (x₀ + a) ∧ 
    (Real.exp y₀)⁻¹ = 1) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_logarithmic_curve_l3046_304672


namespace NUMINAMATH_CALUDE_backyard_area_l3046_304630

/-- Proves that the area of a rectangular backyard with given conditions is 400 square meters -/
theorem backyard_area (length walk_length perimeter : ℝ) 
  (h1 : length * 30 = 1200)
  (h2 : perimeter * 12 = 1200)
  (h3 : perimeter = 2 * length + 2 * (perimeter / 2 - length)) : 
  length * (perimeter / 2 - length) = 400 := by
  sorry

end NUMINAMATH_CALUDE_backyard_area_l3046_304630


namespace NUMINAMATH_CALUDE_fraction_equality_l3046_304636

theorem fraction_equality (a b : ℝ) (h : a + b ≠ 0) : (-a - b) / (a + b) = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3046_304636


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l3046_304692

/-- The volume of a regular tetrahedron with given base side length and lateral face angle -/
theorem tetrahedron_volume 
  (base_side : ℝ) 
  (lateral_angle : ℝ) 
  (h_base : base_side = Real.sqrt 3) 
  (h_angle : lateral_angle = π / 3) : 
  (1 / 3 : ℝ) * base_side ^ 2 * (base_side / 2) / Real.tan lateral_angle = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l3046_304692


namespace NUMINAMATH_CALUDE_triangle_area_l3046_304675

theorem triangle_area (base height : ℝ) (h1 : base = 25) (h2 : height = 60) :
  (base * height) / 2 = 750 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3046_304675


namespace NUMINAMATH_CALUDE_smallest_gcd_multiple_l3046_304617

theorem smallest_gcd_multiple (p q : ℕ+) (h : Nat.gcd p q = 9) :
  (∀ p q : ℕ+, Nat.gcd p q = 9 → Nat.gcd (8 * p) (18 * q) ≥ 18) ∧
  (∃ p q : ℕ+, Nat.gcd p q = 9 ∧ Nat.gcd (8 * p) (18 * q) = 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_gcd_multiple_l3046_304617


namespace NUMINAMATH_CALUDE_intersection_point_solution_l3046_304608

-- Define the lines
def line1 (x y : ℝ) : Prop := y = -x + 4
def line2 (x y m : ℝ) : Prop := y = 2*x + m

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y - 4 = 0
def equation2 (x y m : ℝ) : Prop := 2*x - y + m = 0

-- Theorem statement
theorem intersection_point_solution (m n : ℝ) :
  (line1 3 n ∧ line2 3 n m) →
  (equation1 3 1 ∧ equation2 3 1 m) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_solution_l3046_304608


namespace NUMINAMATH_CALUDE_not_all_tv_owners_have_gellert_pass_l3046_304631

-- Define the universe of discourse
variable (Person : Type)

-- Define predicates
variable (isTelevisionOwner : Person → Prop)
variable (isPainter : Person → Prop)
variable (hasGellertPass : Person → Prop)

-- State the theorem
theorem not_all_tv_owners_have_gellert_pass
  (h1 : ∃ x, isTelevisionOwner x ∧ ¬isPainter x)
  (h2 : ∀ x, hasGellertPass x ∧ ¬isPainter x → ¬isTelevisionOwner x) :
  ∃ x, isTelevisionOwner x ∧ ¬hasGellertPass x :=
by sorry

end NUMINAMATH_CALUDE_not_all_tv_owners_have_gellert_pass_l3046_304631


namespace NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_7_9_and_8_l3046_304663

theorem unique_number_divisible_by_24_with_cube_root_between_7_9_and_8 :
  ∃! (n : ℕ), n > 0 ∧ 24 ∣ n ∧ (7.9 : ℝ) < n^(1/3) ∧ n^(1/3) < 8 :=
by sorry

end NUMINAMATH_CALUDE_unique_number_divisible_by_24_with_cube_root_between_7_9_and_8_l3046_304663


namespace NUMINAMATH_CALUDE_triangle_abc_equilateral_l3046_304654

theorem triangle_abc_equilateral 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * a = b + c) 
  (h2 : Real.sin A ^ 2 = Real.sin B * Real.sin C) : 
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_equilateral_l3046_304654


namespace NUMINAMATH_CALUDE_smallest_number_formed_by_2_and_4_l3046_304699

def is_formed_by_2_and_4 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 10 * a + b ∧ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 2)

theorem smallest_number_formed_by_2_and_4 :
  ∀ n : ℕ, is_formed_by_2_and_4 n → n ≥ 24 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_formed_by_2_and_4_l3046_304699


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3046_304658

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 15120 →
  e = 9 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3046_304658


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3046_304628

theorem binomial_coefficient_divisibility 
  (p : Nat) (α : Nat) (m : Nat) 
  (hp : Nat.Prime p) 
  (hp_odd : Odd p) 
  (hα : α ≥ 2) 
  (hm : m ≥ 2) : 
  ∃ k : Nat, Nat.choose (p^(α-2)) m = k * p^(α-m) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_l3046_304628


namespace NUMINAMATH_CALUDE_nestedRadical_eq_six_l3046_304697

/-- The value of the infinite nested radical sqrt(18 + sqrt(18 + sqrt(18 + ...))) -/
noncomputable def nestedRadical : ℝ :=
  Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt (18 + Real.sqrt 18))))

/-- Theorem stating that the value of the nested radical is 6 -/
theorem nestedRadical_eq_six : nestedRadical = 6 := by
  sorry

end NUMINAMATH_CALUDE_nestedRadical_eq_six_l3046_304697


namespace NUMINAMATH_CALUDE_total_soldiers_l3046_304641

theorem total_soldiers (n : ℕ) 
  (h1 : ∃ x y : ℕ, x + y = n ∧ y = x / 6)
  (h2 : ∃ x' y' : ℕ, x' + y' = n ∧ y' = x' / 7)
  (h3 : ∃ y y' : ℕ, y - y' = 2)
  (h4 : ∀ z : ℕ, z + n = n → z = 0) :
  n = 98 := by
  sorry

end NUMINAMATH_CALUDE_total_soldiers_l3046_304641


namespace NUMINAMATH_CALUDE_log_equation_equals_zero_l3046_304670

-- Define the base-10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_equals_zero :
  (log10 2)^2 + log10 2 * log10 50 - log10 4 = 0 := by sorry

end NUMINAMATH_CALUDE_log_equation_equals_zero_l3046_304670


namespace NUMINAMATH_CALUDE_joe_initial_money_l3046_304611

/-- The amount of money Joe spends on video games each month -/
def monthly_spend : ℕ := 50

/-- The amount of money Joe earns from selling games each month -/
def monthly_earn : ℕ := 30

/-- The number of months Joe can continue buying and selling games -/
def months : ℕ := 12

/-- The initial amount of money Joe has -/
def initial_money : ℕ := (monthly_spend - monthly_earn) * months

theorem joe_initial_money :
  initial_money = 240 :=
sorry

end NUMINAMATH_CALUDE_joe_initial_money_l3046_304611


namespace NUMINAMATH_CALUDE_volunteer_schedule_l3046_304698

theorem volunteer_schedule (sasha leo uma kim : ℕ) 
  (h1 : sasha = 5) 
  (h2 : leo = 8) 
  (h3 : uma = 9) 
  (h4 : kim = 10) : 
  Nat.lcm (Nat.lcm (Nat.lcm sasha leo) uma) kim = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_l3046_304698


namespace NUMINAMATH_CALUDE_star_three_neg_two_thirds_l3046_304639

-- Define the ☆ operation
def star (a b : ℚ) : ℚ := a^2 + a*b - 5

-- Theorem statement
theorem star_three_neg_two_thirds : star 3 (-2/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_star_three_neg_two_thirds_l3046_304639


namespace NUMINAMATH_CALUDE_m_range_l3046_304695

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + (m - 1) * x

-- State the theorem
theorem m_range :
  (∀ (x : ℝ), x^2 + 4*x - m ≥ 0) ∧
  (∀ (x y : ℝ), x < y → x ≤ -3 → y ≤ -3 → f m x ≤ f m y) →
  m ∈ Set.Icc (-5 : ℝ) (-4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3046_304695


namespace NUMINAMATH_CALUDE_only_f₂_passes_through_origin_l3046_304649

-- Define the functions
def f₁ (x : ℝ) := x + 1
def f₂ (x : ℝ) := x^2
def f₃ (x : ℝ) := (x - 4)^2
noncomputable def f₄ (x : ℝ) := 1/x

-- Theorem statement
theorem only_f₂_passes_through_origin :
  (f₁ 0 ≠ 0) ∧ 
  (f₂ 0 = 0) ∧ 
  (f₃ 0 ≠ 0) ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x| ∧ |x| < δ → |f₄ x| > 1/ε) :=
by sorry

end NUMINAMATH_CALUDE_only_f₂_passes_through_origin_l3046_304649


namespace NUMINAMATH_CALUDE_cone_height_l3046_304635

/-- For a cone with slant height 2 and lateral area 4 times the area of its base, 
    the height of the cone is π/2. -/
theorem cone_height (r : ℝ) (h : ℝ) : 
  r > 0 → h > 0 → 
  r^2 + h^2 = 4 → -- slant height is 2
  2 * π * r = 4 * π * r^2 → -- lateral area is 4 times base area
  h = π / 2 := by
sorry

end NUMINAMATH_CALUDE_cone_height_l3046_304635


namespace NUMINAMATH_CALUDE_bill_toilet_paper_usage_l3046_304612

/-- Calculates the number of toilet paper squares used per bathroom visit -/
def toilet_paper_usage (bathroom_visits_per_day : ℕ) (total_rolls : ℕ) (squares_per_roll : ℕ) (total_days : ℕ) : ℕ :=
  (total_rolls * squares_per_roll) / (total_days * bathroom_visits_per_day)

/-- Proves that Bill uses 5 squares of toilet paper per bathroom visit -/
theorem bill_toilet_paper_usage :
  toilet_paper_usage 3 1000 300 20000 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bill_toilet_paper_usage_l3046_304612


namespace NUMINAMATH_CALUDE_distance_theorem_l3046_304603

/-- The configuration of three squares where the middle one is rotated and lowered -/
structure SquareConfiguration where
  /-- Side length of each square -/
  side_length : ℝ
  /-- Rotation angle of the middle square in radians -/
  rotation_angle : ℝ

/-- Calculate the distance from point B to the original line -/
def distance_to_line (config : SquareConfiguration) : ℝ :=
  sorry

/-- The theorem stating the distance from point B to the original line -/
theorem distance_theorem (config : SquareConfiguration) :
  config.side_length = 1 ∧ config.rotation_angle = π / 4 →
  distance_to_line config = Real.sqrt 2 + 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_distance_theorem_l3046_304603


namespace NUMINAMATH_CALUDE_intersection_M_N_l3046_304679

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (1 - x) < 0}
def N : Set ℝ := {x | x^2 ≤ 1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3046_304679


namespace NUMINAMATH_CALUDE_inequality_proof_l3046_304625

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  1/a + 1/b + 4/c + 16/d ≥ 64 / (a + b + c + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3046_304625


namespace NUMINAMATH_CALUDE_line_through_circle_center_l3046_304664

/-- The center of a circle given by the equation x^2 + y^2 + 2x - 4y = 0 -/
def circle_center : ℝ × ℝ := (-1, 2)

/-- The line equation 3x + y + a = 0 -/
def line_equation (a : ℝ) (x y : ℝ) : Prop := 3 * x + y + a = 0

/-- The theorem stating that if the line 3x + y + a = 0 passes through the center of the circle x^2 + y^2 + 2x - 4y = 0, then a = 1 -/
theorem line_through_circle_center (a : ℝ) : 
  line_equation a (circle_center.1) (circle_center.2) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_through_circle_center_l3046_304664


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3046_304688

/-- The equation of the tangent line to y = x^3 - 2x at (1, -1) is x - y - 2 = 0 -/
theorem tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 - 2*x →
  x₀ = 1 →
  y₀ = -1 →
  f x₀ = y₀ →
  (deriv f) x₀ = 1 →
  ∀ x y, (x - x₀) = (deriv f x₀) * (y - y₀) ↔ x - y - 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3046_304688


namespace NUMINAMATH_CALUDE_survey_B_most_suitable_for_census_l3046_304619

-- Define the characteristics of a survey
structure Survey where
  population : Set String
  method : String
  is_destructive : Bool
  is_manageable : Bool

-- Define the conditions for a census
def is_census_suitable (s : Survey) : Prop :=
  s.is_manageable ∧ ¬s.is_destructive ∧ s.method = "complete enumeration"

-- Define the surveys
def survey_A : Survey := {
  population := {"televisions"},
  method := "sampling",
  is_destructive := true,
  is_manageable := false
}

def survey_B : Survey := {
  population := {"ninth grade students in a certain middle school class"},
  method := "complete enumeration",
  is_destructive := false,
  is_manageable := true
}

def survey_C : Survey := {
  population := {"middle school students in Chongqing"},
  method := "sampling",
  is_destructive := false,
  is_manageable := false
}

def survey_D : Survey := {
  population := {"middle school students in Chongqing"},
  method := "sampling",
  is_destructive := false,
  is_manageable := false
}

-- Theorem stating that survey B is the most suitable for a census
theorem survey_B_most_suitable_for_census :
  is_census_suitable survey_B ∧
  ¬is_census_suitable survey_A ∧
  ¬is_census_suitable survey_C ∧
  ¬is_census_suitable survey_D :=
sorry

end NUMINAMATH_CALUDE_survey_B_most_suitable_for_census_l3046_304619


namespace NUMINAMATH_CALUDE_sum_vertices_is_nine_l3046_304614

/-- The number of vertices in a rectangle --/
def rectangle_vertices : ℕ := 4

/-- The number of vertices in a pentagon --/
def pentagon_vertices : ℕ := 5

/-- The sum of vertices of a rectangle and a pentagon --/
def sum_vertices : ℕ := rectangle_vertices + pentagon_vertices

theorem sum_vertices_is_nine : sum_vertices = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_vertices_is_nine_l3046_304614


namespace NUMINAMATH_CALUDE_problem_solution_l3046_304600

/-- The function f(x) = x^2 - 2ax + 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem problem_solution (a : ℝ) (h : a > 1) :
  /- Part 1 -/
  (∀ x, x ∈ Set.Icc 1 a ↔ f a x ∈ Set.Icc 1 a) →
  a = 2
  ∧
  /- Part 2 -/
  ((∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) ∧
   (∀ x ∈ Set.Icc 1 2, f a x ≤ 0)) →
  a ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3046_304600


namespace NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3046_304653

theorem smallest_positive_integer_congruence :
  ∃ (n : ℕ), n > 0 ∧ (77 * n) % 385 = 308 % 385 ∧
  ∀ (m : ℕ), m > 0 → (77 * m) % 385 = 308 % 385 → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_congruence_l3046_304653


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_projection_trajectory_l3046_304677

/-- The line equation as a function of x, y, and m -/
def line_equation (x y m : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0

/-- The fixed point through which all lines pass -/
def fixed_point : ℝ × ℝ := (1, -2)

/-- Point P coordinates -/
def point_p : ℝ × ℝ := (-1, 0)

/-- Trajectory equation of point M -/
def trajectory_equation (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 2

theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation (fixed_point.1) (fixed_point.2) m := by sorry

theorem projection_trajectory :
  ∀ x y : ℝ, (∃ m : ℝ, line_equation x y m ∧ 
    (x - point_p.1)^2 + (y - point_p.2)^2 = 
    ((x - point_p.1) * (fixed_point.1 - point_p.1) + (y - point_p.2) * (fixed_point.2 - point_p.2))^2 / 
    ((fixed_point.1 - point_p.1)^2 + (fixed_point.2 - point_p.2)^2)) →
  trajectory_equation x y := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_projection_trajectory_l3046_304677


namespace NUMINAMATH_CALUDE_pump_water_in_35_minutes_l3046_304661

theorem pump_water_in_35_minutes : 
  let pump_rate : ℚ := 300  -- gallons per hour
  let time : ℚ := 35 / 60   -- 35 minutes converted to hours
  pump_rate * time = 175
  := by sorry

end NUMINAMATH_CALUDE_pump_water_in_35_minutes_l3046_304661


namespace NUMINAMATH_CALUDE_correlated_relationships_l3046_304683

-- Define the set of all relationships
inductive Relationship
| A  -- A person's height and weight
| B  -- The distance traveled by a vehicle moving at a constant speed and the time of travel
| C  -- A person's height and eyesight
| D  -- The volume of a cube and its edge length

-- Define a function to check if a relationship is correlated
def is_correlated (r : Relationship) : Prop :=
  match r with
  | Relationship.A => true  -- Height and weight are correlated
  | Relationship.B => true  -- Distance and time at constant speed are correlated (functional)
  | Relationship.C => false -- Height and eyesight are not correlated
  | Relationship.D => true  -- Volume and edge length of a cube are correlated (functional)

-- Theorem stating that the set of correlated relationships is {A, B, D}
theorem correlated_relationships :
  {r : Relationship | is_correlated r} = {Relationship.A, Relationship.B, Relationship.D} :=
by sorry

end NUMINAMATH_CALUDE_correlated_relationships_l3046_304683


namespace NUMINAMATH_CALUDE_leg_equals_sum_of_radii_l3046_304621

/-- An isosceles right triangle with its inscribed and circumscribed circles -/
structure IsoscelesRightTriangle where
  /-- The length of each leg of the triangle -/
  a : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The leg length is positive -/
  a_pos : 0 < a
  /-- The inscribed circle radius is half the leg length -/
  r_def : r = a / 2
  /-- The circumscribed circle radius is (a√2)/2 -/
  R_def : R = (a * Real.sqrt 2) / 2

/-- 
The length of the legs of an isosceles right triangle is equal to 
the sum of the radii of its inscribed and circumscribed circles 
-/
theorem leg_equals_sum_of_radii (t : IsoscelesRightTriangle) : t.a = t.r + t.R := by
  sorry

end NUMINAMATH_CALUDE_leg_equals_sum_of_radii_l3046_304621


namespace NUMINAMATH_CALUDE_expression_evaluation_l3046_304666

theorem expression_evaluation : 3^(0^(2^5)) + ((3^0)^2)^5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3046_304666


namespace NUMINAMATH_CALUDE_area_under_curve_l3046_304609

-- Define the curve
def f (x : ℝ) := x^2

-- Define the boundaries
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_under_curve :
  (∫ x in lower_bound..upper_bound, f x) = (1 : ℝ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_area_under_curve_l3046_304609


namespace NUMINAMATH_CALUDE_partition_scores_with_equal_average_l3046_304673

theorem partition_scores_with_equal_average 
  (N : ℕ) 
  (scores : List ℤ) 
  (h_length : scores.length = 3 * N)
  (h_range : ∀ s ∈ scores, 60 ≤ s ∧ s ≤ 100)
  (h_freq : ∀ s ∈ scores, (scores.filter (· = s)).length ≥ 2)
  (h_avg : scores.sum / (3 * N) = 824 / 10) :
  ∃ (class1 class2 class3 : List ℤ),
    class1.length = N ∧ 
    class2.length = N ∧ 
    class3.length = N ∧
    scores = class1 ++ class2 ++ class3 ∧
    class1.sum / N = 824 / 10 ∧
    class2.sum / N = 824 / 10 ∧
    class3.sum / N = 824 / 10 :=
by sorry

end NUMINAMATH_CALUDE_partition_scores_with_equal_average_l3046_304673


namespace NUMINAMATH_CALUDE_parallel_lines_k_values_l3046_304687

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The first line -/
def l1 (k : ℝ) : Line :=
  { a := k - 3
    b := 4 - k
    c := 1 }

/-- The second line -/
def l2 (k : ℝ) : Line :=
  { a := 2 * (k - 3)
    b := -2
    c := 3 }

/-- Theorem: If l1 and l2 are parallel, then k is either 3 or 5 -/
theorem parallel_lines_k_values :
  ∀ k : ℝ, parallel (l1 k) (l2 k) → k = 3 ∨ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_values_l3046_304687


namespace NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3046_304656

-- Part 1
theorem calculate_expression : (-2)^2 + (Real.sqrt 3 - Real.pi)^0 + abs (1 - Real.sqrt 3) = 4 + Real.sqrt 3 := by
  sorry

-- Part 2
theorem solve_system_of_equations :
  ∃ x y : ℝ, 2*x + y = 1 ∧ x - 2*y = 3 ∧ x = 1 ∧ y = -1 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_solve_system_of_equations_l3046_304656


namespace NUMINAMATH_CALUDE_sum_of_squares_l3046_304634

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) : x^2 + y^2 = 840 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3046_304634


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3046_304615

theorem triangle_angle_measure (A B C : ℝ) (h1 : A = 3 * Real.pi / 4) (h2 : C > 0) (h3 : C < Real.pi / 4) (h4 : Real.sin C = 1 / 2) : C = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3046_304615


namespace NUMINAMATH_CALUDE_dessert_shop_theorem_l3046_304671

/-- Represents the dessert shop problem -/
structure DessertShop where
  x : ℕ  -- portions of dessert A
  y : ℕ  -- portions of dessert B
  a : ℕ  -- profit per portion of dessert A in yuan

/-- Conditions of the dessert shop problem -/
def DessertShopConditions (shop : DessertShop) : Prop :=
  shop.a > 0 ∧
  30 * shop.x + 10 * shop.y = 2000 ∧
  15 * shop.x + 20 * shop.y ≤ 3100

/-- Theorem stating the main results of the dessert shop problem -/
theorem dessert_shop_theorem (shop : DessertShop) 
  (h : DessertShopConditions shop) : 
  (shop.y = 200 - 3 * shop.x) ∧ 
  (shop.a = 3 → 3 * shop.x + 2 * shop.y ≥ 220 → 15 * shop.x + 20 * shop.y ≥ 1300) ∧
  (3 * shop.x + 2 * shop.y = 450 → shop.a = 8) := by
  sorry

end NUMINAMATH_CALUDE_dessert_shop_theorem_l3046_304671


namespace NUMINAMATH_CALUDE_list_property_l3046_304652

theorem list_property (S : ℝ) (n : ℝ) :
  let list_size : ℕ := 21
  let other_numbers_sum : ℝ := S - n
  let other_numbers_count : ℕ := list_size - 1
  let other_numbers_avg : ℝ := other_numbers_sum / other_numbers_count
  n = 4 * other_numbers_avg →
  n = S / 6 →
  other_numbers_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_list_property_l3046_304652


namespace NUMINAMATH_CALUDE_smallest_divisible_by_8_13_14_l3046_304678

theorem smallest_divisible_by_8_13_14 : ∃ n : ℕ, n > 0 ∧ 
  8 ∣ n ∧ 13 ∣ n ∧ 14 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 8 ∣ m → 13 ∣ m → 14 ∣ m → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_8_13_14_l3046_304678


namespace NUMINAMATH_CALUDE_power_function_through_point_l3046_304691

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 4 = 2 → f 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3046_304691


namespace NUMINAMATH_CALUDE_mrs_hilt_friday_miles_l3046_304637

/-- Mrs. Hilt's running schedule for a week -/
structure RunningSchedule where
  monday : ℕ
  wednesday : ℕ
  friday : ℕ
  total : ℕ

/-- Theorem: Given Mrs. Hilt's running schedule, prove she ran 7 miles on Friday -/
theorem mrs_hilt_friday_miles (schedule : RunningSchedule) 
  (h1 : schedule.monday = 3)
  (h2 : schedule.wednesday = 2)
  (h3 : schedule.total = 12)
  (h4 : schedule.total = schedule.monday + schedule.wednesday + schedule.friday) :
  schedule.friday = 7 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_friday_miles_l3046_304637


namespace NUMINAMATH_CALUDE_triangle_radius_inequality_l3046_304605

structure Triangle where
  R : ℝ  -- circumradius
  r : ℝ  -- inradius
  P : ℝ  -- perimeter
  is_acute_or_obtuse : Bool
  is_right_angled : Bool
  positive_R : R > 0
  positive_r : r > 0
  positive_P : P > 0

theorem triangle_radius_inequality (t : Triangle) : 
  (t.is_acute_or_obtuse ∧ t.R > (Real.sqrt 3 / 3) * Real.sqrt (t.P * t.r)) ∨
  (t.is_right_angled ∧ t.R ≥ (Real.sqrt 2 / 2) * Real.sqrt (t.P * t.r)) :=
sorry

end NUMINAMATH_CALUDE_triangle_radius_inequality_l3046_304605
