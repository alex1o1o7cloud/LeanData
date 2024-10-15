import Mathlib

namespace NUMINAMATH_CALUDE_weighted_average_score_l126_12631

-- Define the scores and weights
def interview_score : ℕ := 90
def computer_score : ℕ := 85
def design_score : ℕ := 80

def interview_weight : ℕ := 5
def computer_weight : ℕ := 2
def design_weight : ℕ := 3

-- Define the total weighted score
def total_weighted_score : ℕ := 
  interview_score * interview_weight + 
  computer_score * computer_weight + 
  design_score * design_weight

-- Define the sum of weights
def sum_of_weights : ℕ := 
  interview_weight + computer_weight + design_weight

-- Theorem to prove
theorem weighted_average_score : 
  total_weighted_score / sum_of_weights = 86 := by
  sorry

end NUMINAMATH_CALUDE_weighted_average_score_l126_12631


namespace NUMINAMATH_CALUDE_simplify_expression_l126_12658

theorem simplify_expression (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l126_12658


namespace NUMINAMATH_CALUDE_captain_age_l126_12693

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  size : ℕ
  captainAge : ℕ
  wicketKeeperAge : ℕ
  teamAverageAge : ℝ
  remainingPlayersAverageAge : ℝ

/-- Theorem stating the captain's age in the given cricket team scenario -/
theorem captain_age (team : CricketTeam) 
  (h1 : team.size = 11)
  (h2 : team.wicketKeeperAge = team.captainAge + 5)
  (h3 : team.teamAverageAge = 23)
  (h4 : team.remainingPlayersAverageAge = team.teamAverageAge - 1)
  : team.captainAge = 25 := by
  sorry

end NUMINAMATH_CALUDE_captain_age_l126_12693


namespace NUMINAMATH_CALUDE_angle_B_is_pi_over_four_l126_12635

/-- Given a triangle ABC with circumradius R, if 2R(sin²A - sin²B) = (√2a - c)sinC, 
    then the measure of angle B is π/4. -/
theorem angle_B_is_pi_over_four 
  (A B C : ℝ) 
  (a b c R : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π) 
  (h3 : 0 < C ∧ C < π) 
  (h4 : A + B + C = π) 
  (h5 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h6 : 0 < R) 
  (h7 : a = 2 * R * Real.sin A) 
  (h8 : b = 2 * R * Real.sin B) 
  (h9 : c = 2 * R * Real.sin C) 
  (h10 : 2 * R * (Real.sin A ^ 2 - Real.sin B ^ 2) = (Real.sqrt 2 * a - c) * Real.sin C) :
  B = π / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_over_four_l126_12635


namespace NUMINAMATH_CALUDE_systematic_sample_theorem_l126_12600

/-- Represents a systematic sample of bottles -/
structure SystematicSample where
  total : Nat
  sample_size : Nat
  start : Nat
  step : Nat

/-- Generates the sample numbers for a systematic sample -/
def generate_sample (s : SystematicSample) : List Nat :=
  List.range s.sample_size |>.map (fun i => s.start + i * s.step)

/-- Theorem: The systematic sample for 60 bottles with 6 selections starts at 3 with step 10 -/
theorem systematic_sample_theorem :
  let s : SystematicSample := ⟨60, 6, 3, 10⟩
  generate_sample s = [3, 13, 23, 33, 43, 53] := by sorry

end NUMINAMATH_CALUDE_systematic_sample_theorem_l126_12600


namespace NUMINAMATH_CALUDE_arrangement_count_l126_12665

/-- The number of volunteers -/
def num_volunteers : ℕ := 5

/-- The number of elderly people -/
def num_elderly : ℕ := 2

/-- The number of positions where the elderly pair can be placed -/
def elderly_pair_positions : ℕ := num_volunteers - 1

/-- The number of arrangements of volunteers -/
def volunteer_arrangements : ℕ := Nat.factorial num_volunteers

/-- The number of arrangements of elderly people -/
def elderly_arrangements : ℕ := Nat.factorial num_elderly

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := elderly_pair_positions * volunteer_arrangements * elderly_arrangements

theorem arrangement_count : total_arrangements = 960 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l126_12665


namespace NUMINAMATH_CALUDE_sin_330_degrees_l126_12623

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l126_12623


namespace NUMINAMATH_CALUDE_cori_age_relation_cori_current_age_l126_12645

/-- Cori's current age -/
def cori_age : ℕ := sorry

/-- Cori's aunt's current age -/
def aunt_age : ℕ := 19

/-- In 5 years, Cori will be one-third the age of her aunt -/
theorem cori_age_relation : cori_age + 5 = (aunt_age + 5) / 3 := sorry

theorem cori_current_age : cori_age = 3 := by sorry

end NUMINAMATH_CALUDE_cori_age_relation_cori_current_age_l126_12645


namespace NUMINAMATH_CALUDE_bug_return_probability_l126_12676

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - P n)

/-- The probability of the bug returning to its starting vertex after 12 moves -/
theorem bug_return_probability : P 12 = 14762 / 59049 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l126_12676


namespace NUMINAMATH_CALUDE_johnny_guitar_practice_l126_12651

/-- Represents the number of days Johnny has been practicing guitar -/
def current_practice : ℕ := 40

/-- Represents the daily practice amount -/
def daily_practice : ℕ := 2

theorem johnny_guitar_practice :
  let days_to_triple := (3 * current_practice - current_practice) / daily_practice
  (2 * (current_practice - 20 * daily_practice) = current_practice) →
  days_to_triple = 80 := by
  sorry

end NUMINAMATH_CALUDE_johnny_guitar_practice_l126_12651


namespace NUMINAMATH_CALUDE_vegetables_for_movie_day_l126_12662

theorem vegetables_for_movie_day 
  (points_needed : ℕ) 
  (points_per_vegetable : ℕ) 
  (num_students : ℕ) 
  (num_days : ℕ) 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : num_students = 25) 
  (h4 : num_days = 10) : 
  (points_needed / (points_per_vegetable * num_students * (num_days / 2))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_for_movie_day_l126_12662


namespace NUMINAMATH_CALUDE_bus_problem_l126_12642

theorem bus_problem (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 28 → got_on = 82 → final = 30 → 
  ∃ (got_off : ℕ), got_on - got_off = 2 ∧ initial + got_on - got_off = final :=
by sorry

end NUMINAMATH_CALUDE_bus_problem_l126_12642


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l126_12679

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 = 1
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  let asymptote_slope := 1 / a
  let perpendicular_slope := -a
  let y_coordinate_P := a * c / (1 + a^2)
  y_coordinate_P = 2 * Real.sqrt 5 / 5 →
  e = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l126_12679


namespace NUMINAMATH_CALUDE_darry_total_steps_l126_12613

/-- Represents the number of steps climbed on a ladder -/
structure LadderClimb where
  steps : Nat
  times : Nat

/-- Calculates the total number of steps climbed on a ladder -/
def totalStepsOnLadder (climb : LadderClimb) : Nat :=
  climb.steps * climb.times

/-- Represents Darry's ladder climbs for the day -/
structure DarryClimbs where
  largest : LadderClimb
  medium : LadderClimb
  smaller : LadderClimb
  smallest : LadderClimb

/-- Darry's actual climbs for the day -/
def darryActualClimbs : DarryClimbs :=
  { largest := { steps := 20, times := 12 }
  , medium := { steps := 15, times := 8 }
  , smaller := { steps := 10, times := 10 }
  , smallest := { steps := 5, times := 15 }
  }

/-- Calculates the total number of steps Darry climbed -/
def totalStepsClimbed (climbs : DarryClimbs) : Nat :=
  totalStepsOnLadder climbs.largest +
  totalStepsOnLadder climbs.medium +
  totalStepsOnLadder climbs.smaller +
  totalStepsOnLadder climbs.smallest

/-- Theorem stating that Darry climbed 535 steps in total -/
theorem darry_total_steps :
  totalStepsClimbed darryActualClimbs = 535 := by
  sorry

end NUMINAMATH_CALUDE_darry_total_steps_l126_12613


namespace NUMINAMATH_CALUDE_average_age_of_six_students_l126_12649

theorem average_age_of_six_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : average_age_group1 = 14)
  (h5 : age_last_student = 17)
  : ∃ (num_group2 : Nat) (average_age_group2 : ℝ),
    num_group2 = total_students - num_group1 - 1 ∧
    average_age_group2 = 16 :=
by
  sorry

#check average_age_of_six_students

end NUMINAMATH_CALUDE_average_age_of_six_students_l126_12649


namespace NUMINAMATH_CALUDE_bill_calculation_l126_12632

def original_bill : ℝ := 500

def first_late_charge_rate : ℝ := 0.02

def second_late_charge_rate : ℝ := 0.03

def final_bill_amount : ℝ := original_bill * (1 + first_late_charge_rate) * (1 + second_late_charge_rate)

theorem bill_calculation :
  final_bill_amount = 525.30 := by
  sorry

end NUMINAMATH_CALUDE_bill_calculation_l126_12632


namespace NUMINAMATH_CALUDE_bridge_length_proof_l126_12605

/-- The length of a bridge given train parameters -/
theorem bridge_length_proof (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600) * crossing_time - train_length = 275 :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_proof_l126_12605


namespace NUMINAMATH_CALUDE_problem_solution_l126_12619

noncomputable section

variables (a : ℝ) (x : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - 4 * x + a

def g (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

def t1 (a : ℝ) (x : ℝ) : ℝ := (1/2) * f a x

def t2 (a : ℝ) (x : ℝ) : ℝ := g a x

def t3 (x : ℝ) : ℝ := 2^x

theorem problem_solution (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ m : ℝ, (¬∃ y : ℝ, (∀ x ∈ Set.Icc (-1) (2*m), (f a x ≤ f a y) ∨ (∀ x ∈ Set.Icc (-1) (2*m), f a x ≥ f a y))) ↔ m > 1/2) ∧
  (f a 1 = g a 1 ↔ a = 2) ∧
  (∀ x ∈ Set.Ioo 0 1, t2 a x < t1 a x ∧ t1 a x < t3 x) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l126_12619


namespace NUMINAMATH_CALUDE_triangle_angle_calculation_l126_12657

theorem triangle_angle_calculation (a b : ℝ) (B : ℝ) (hA : a = Real.sqrt 2) (hB : B = 45 * π / 180) (hb : b = 2) :
  ∃ (A : ℝ), A = 30 * π / 180 ∧ a / Real.sin A = b / Real.sin B := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_calculation_l126_12657


namespace NUMINAMATH_CALUDE_probability_male_student_id_l126_12614

theorem probability_male_student_id (male_count female_count : ℕ) 
  (h1 : male_count = 6) (h2 : female_count = 4) : 
  (male_count : ℚ) / ((male_count : ℚ) + (female_count : ℚ)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_male_student_id_l126_12614


namespace NUMINAMATH_CALUDE_sum_pqr_values_l126_12608

theorem sum_pqr_values (p q r : ℝ) (distinct : p ≠ q ∧ q ≠ r ∧ p ≠ r)
  (eq1 : q = p * (4 - p)) (eq2 : r = q * (4 - q)) (eq3 : p = r * (4 - r)) :
  p + q + r = 6 ∨ p + q + r = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_pqr_values_l126_12608


namespace NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_27_l126_12696

theorem difference_of_cubes_divisible_by_27 (a b : ℤ) :
  ∃ k : ℤ, (3 * a + 2)^3 - (3 * b + 2)^3 = 27 * k := by sorry

end NUMINAMATH_CALUDE_difference_of_cubes_divisible_by_27_l126_12696


namespace NUMINAMATH_CALUDE_hundredth_odd_and_plus_ten_l126_12687

/-- The nth odd positive integer -/
def nthOddPositive (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_and_plus_ten :
  (nthOddPositive 100 = 199) ∧ (nthOddPositive 100 + 10 = 209) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_odd_and_plus_ten_l126_12687


namespace NUMINAMATH_CALUDE_exists_increasing_interval_l126_12634

open Real

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := log x + 1 / log x

-- State the theorem
theorem exists_increasing_interval :
  ∃ x₀ : ℝ, x₀ > 0 ∧ ∀ x > x₀, deriv f x > 0 :=
sorry

end NUMINAMATH_CALUDE_exists_increasing_interval_l126_12634


namespace NUMINAMATH_CALUDE_range_of_exponential_function_l126_12652

theorem range_of_exponential_function :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, 3^x = y := by
  sorry

end NUMINAMATH_CALUDE_range_of_exponential_function_l126_12652


namespace NUMINAMATH_CALUDE_periodic_odd_function_value_l126_12673

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_value
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 6)
  (h_odd : is_odd f)
  (h_value : f (-1) = -1) :
  f 5 = -1 := by
  sorry

end NUMINAMATH_CALUDE_periodic_odd_function_value_l126_12673


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l126_12683

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4*I)*z = Complex.abs (4 - 3*I)) : 
  z.im = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l126_12683


namespace NUMINAMATH_CALUDE_optimal_strategy_l126_12618

/-- Represents the profit function for zongzi sales -/
def profit_function (x : ℝ) (a : ℝ) : ℝ := (a - 5) * x + 6000

/-- Represents the constraints on the number of boxes of type A zongzi -/
def valid_quantity (x : ℝ) : Prop := 100 ≤ x ∧ x ≤ 150

/-- Theorem stating the optimal purchasing strategy to maximize profit -/
theorem optimal_strategy (a : ℝ) (h1 : 0 < a) (h2 : a < 10) :
  (0 < a ∧ a < 5 → 
    ∀ x, valid_quantity x → profit_function 100 a ≥ profit_function x a) ∧
  (5 ≤ a ∧ a < 10 → 
    ∀ x, valid_quantity x → profit_function 150 a ≥ profit_function x a) :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l126_12618


namespace NUMINAMATH_CALUDE_inequality_proof_l126_12667

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l126_12667


namespace NUMINAMATH_CALUDE_exponential_dominance_l126_12625

theorem exponential_dominance (k : ℝ) (hk : k > 0) :
  ∃ x₀ : ℝ, ∀ x ≥ x₀, (2 : ℝ) ^ ((2 : ℝ) ^ x) > ((2 : ℝ) ^ x) ^ k :=
sorry

end NUMINAMATH_CALUDE_exponential_dominance_l126_12625


namespace NUMINAMATH_CALUDE_special_divisors_count_l126_12660

/-- The number of positive integer divisors of 2022^2022 that are divisible by exactly 2022 positive integers -/
def num_special_divisors : ℕ := 6

/-- 2022 factorized as 2 * 3 * 337 -/
def factorization_2022 : ℕ × ℕ × ℕ := (2, 3, 337)

theorem special_divisors_count :
  (factorization_2022.1 * factorization_2022.2.1 * factorization_2022.2.2 = 2022) →
  (∃ (a b c : ℕ), (a + 1) * (b + 1) * (c + 1) = 2022 ∧
    num_special_divisors = (List.length [
      (factorization_2022.1, factorization_2022.2.1, factorization_2022.2.2),
      (factorization_2022.1, factorization_2022.2.2, factorization_2022.2.1),
      (factorization_2022.2.1, factorization_2022.1, factorization_2022.2.2),
      (factorization_2022.2.1, factorization_2022.2.2, factorization_2022.1),
      (factorization_2022.2.2, factorization_2022.1, factorization_2022.2.1),
      (factorization_2022.2.2, factorization_2022.2.1, factorization_2022.1)
    ])) :=
by sorry

end NUMINAMATH_CALUDE_special_divisors_count_l126_12660


namespace NUMINAMATH_CALUDE_not_cube_sum_l126_12669

theorem not_cube_sum (a b : ℕ) : ¬ ∃ c : ℤ, (a : ℤ)^3 + (b : ℤ)^3 + 4 = c^3 := by
  sorry

end NUMINAMATH_CALUDE_not_cube_sum_l126_12669


namespace NUMINAMATH_CALUDE_willies_cream_calculation_l126_12640

/-- The amount of whipped cream Willie needs in total (in lbs) -/
def total_cream : ℕ := 300

/-- The amount of cream Willie needs to buy (in lbs) -/
def cream_to_buy : ℕ := 151

/-- The amount of cream Willie got from his farm (in lbs) -/
def cream_from_farm : ℕ := total_cream - cream_to_buy

theorem willies_cream_calculation :
  cream_from_farm = 149 := by
  sorry

end NUMINAMATH_CALUDE_willies_cream_calculation_l126_12640


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l126_12644

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f y + x^2 + 1) + 2*x = y + (f (x + 1))^2) : 
  ∀ x : ℝ, f x = x := by
sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l126_12644


namespace NUMINAMATH_CALUDE_missing_digit_in_103rd_rising_number_l126_12610

/-- A rising number is a positive integer each digit of which is larger than each of the digits to its left. -/
def IsRisingNumber (n : ℕ) : Prop := sorry

/-- The set of all five-digit rising numbers using digits from 1 to 9. -/
def FiveDigitRisingNumbers : Set ℕ := {n : ℕ | IsRisingNumber n ∧ n ≥ 10000 ∧ n < 100000}

/-- The 103rd element in the ordered set of five-digit rising numbers. -/
def OneHundredThirdRisingNumber : ℕ := sorry

theorem missing_digit_in_103rd_rising_number :
  ¬ (∃ (d : ℕ), d = 5 ∧ 10 * (OneHundredThirdRisingNumber / 10) + d = OneHundredThirdRisingNumber) :=
sorry

end NUMINAMATH_CALUDE_missing_digit_in_103rd_rising_number_l126_12610


namespace NUMINAMATH_CALUDE_R_zero_value_l126_12664

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^2 - 3*x - 7

-- Define the properties for Q and R
def is_valid_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b, ∀ x, f x = x^2 + a*x + b

-- Define the condition that P + Q, P + R, and Q + R each have a common root
def have_common_roots (P Q R : ℝ → ℝ) : Prop :=
  ∃ p q r, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (P p + Q p = 0 ∧ P p + R p = 0) ∧
    (P q + Q q = 0 ∧ Q q + R q = 0) ∧
    (P r + R r = 0 ∧ Q r + R r = 0)

-- Main theorem
theorem R_zero_value (Q R : ℝ → ℝ) 
  (hQ : is_valid_polynomial Q)
  (hR : is_valid_polynomial R)
  (hQR : have_common_roots P Q R)
  (hQ0 : Q 0 = 2) :
  R 0 = 52 / 19 :=
sorry

end NUMINAMATH_CALUDE_R_zero_value_l126_12664


namespace NUMINAMATH_CALUDE_molecular_weight_NaOCl_approx_l126_12689

/-- The atomic weight of Sodium in g/mol -/
def atomic_weight_Na : ℝ := 22.99

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The molecular weight of NaOCl in g/mol -/
def molecular_weight_NaOCl : ℝ := atomic_weight_Na + atomic_weight_O + atomic_weight_Cl

/-- Theorem stating that the molecular weight of NaOCl is approximately 74.44 g/mol -/
theorem molecular_weight_NaOCl_approx :
  ∀ ε > 0, |molecular_weight_NaOCl - 74.44| < ε :=
sorry

end NUMINAMATH_CALUDE_molecular_weight_NaOCl_approx_l126_12689


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l126_12655

/-- Given three square regions A, B, and C, where the perimeter of A is 16 units,
    the perimeter of B is 32 units, and the side length of each subsequent region doubles,
    prove that the ratio of the area of region B to the area of region C is 1/4. -/
theorem area_ratio_of_squares (side_A side_B side_C : ℝ) : 
  side_A * 4 = 16 →
  side_B * 4 = 32 →
  side_C = 2 * side_B →
  (side_B ^ 2) / (side_C ^ 2) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l126_12655


namespace NUMINAMATH_CALUDE_game_result_l126_12699

def point_function (n : Nat) : Nat :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def alex_rolls : List Nat := [6, 4, 3, 2, 1]
def bob_rolls : List Nat := [5, 6, 2, 3, 3]

def calculate_points (rolls : List Nat) : Nat :=
  (rolls.map point_function).sum

theorem game_result : 
  (calculate_points alex_rolls) * (calculate_points bob_rolls) = 672 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l126_12699


namespace NUMINAMATH_CALUDE_people_who_left_line_l126_12617

theorem people_who_left_line (initial_people final_people joined_people : ℕ) 
  (h1 : initial_people = 30)
  (h2 : final_people = 25)
  (h3 : joined_people = 5)
  (h4 : final_people = initial_people - (people_who_left : ℕ) + joined_people) :
  people_who_left = 10 := by
sorry

end NUMINAMATH_CALUDE_people_who_left_line_l126_12617


namespace NUMINAMATH_CALUDE_tan_70_cos_10_sqrt_3_tan_20_minus_1_l126_12636

theorem tan_70_cos_10_sqrt_3_tan_20_minus_1 : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_tan_70_cos_10_sqrt_3_tan_20_minus_1_l126_12636


namespace NUMINAMATH_CALUDE_shortest_tangent_theorem_l126_12688

noncomputable def circle_C3 (x y : ℝ) : Prop := (x - 8) ^ 2 + (y - 3) ^ 2 = 49

noncomputable def circle_C4 (x y : ℝ) : Prop := (x + 12) ^ 2 + (y + 4) ^ 2 = 16

noncomputable def shortest_tangent_length : ℝ := (Real.sqrt 7840 + Real.sqrt 24181) / 11 - 11

theorem shortest_tangent_theorem :
  ∃ (R S : ℝ × ℝ),
    circle_C3 R.1 R.2 ∧
    circle_C4 S.1 S.2 ∧
    (∀ (P Q : ℝ × ℝ),
      circle_C3 P.1 P.2 →
      circle_C4 Q.1 Q.2 →
      Real.sqrt ((R.1 - S.1) ^ 2 + (R.2 - S.2) ^ 2) ≤ Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)) ∧
    Real.sqrt ((R.1 - S.1) ^ 2 + (R.2 - S.2) ^ 2) = shortest_tangent_length :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_tangent_theorem_l126_12688


namespace NUMINAMATH_CALUDE_difference_of_differences_l126_12671

theorem difference_of_differences (a b c : ℤ) 
  (hab : a - b = 2) (hbc : b - c = -3) : a - c = -1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_differences_l126_12671


namespace NUMINAMATH_CALUDE_book_arrangement_count_l126_12692

theorem book_arrangement_count : 
  let total_books : ℕ := 12
  let arabic_books : ℕ := 3
  let german_books : ℕ := 4
  let spanish_books : ℕ := 3
  let french_books : ℕ := 2
  let grouped_units : ℕ := 3  -- Arabic, Spanish, French groups
  let total_arrangements : ℕ := 
    (Nat.factorial (grouped_units + german_books)) * 
    (Nat.factorial arabic_books) * 
    (Nat.factorial spanish_books) * 
    (Nat.factorial french_books)
  total_books = arabic_books + german_books + spanish_books + french_books →
  total_arrangements = 362880 := by
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l126_12692


namespace NUMINAMATH_CALUDE_smallest_n_below_threshold_l126_12626

/-- The probability of drawing a red marble on the nth draw -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 1000

/-- The threshold probability -/
def threshold : ℚ := 1 / 1000

theorem smallest_n_below_threshold :
  (∀ k < 32, P k ≥ threshold) ∧ P 32 < threshold := by sorry

end NUMINAMATH_CALUDE_smallest_n_below_threshold_l126_12626


namespace NUMINAMATH_CALUDE_algebraic_simplification_l126_12611

theorem algebraic_simplification (y : ℝ) (h : y ≠ 0) :
  (-24 * y^3) * (5 * y^2) * (1 / (2*y)^3) = -15 * y^2 :=
sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l126_12611


namespace NUMINAMATH_CALUDE_inequality_solution_set_l126_12694

theorem inequality_solution_set (x : ℝ) : 
  1 / (x + 2) + 8 / (x + 6) ≥ 1 ↔ -6 < x ∧ x ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l126_12694


namespace NUMINAMATH_CALUDE_subtraction_from_percentage_l126_12603

theorem subtraction_from_percentage (n : ℝ) : n = 100 → (0.8 * n - 20 = 60) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_from_percentage_l126_12603


namespace NUMINAMATH_CALUDE_sparklers_to_crackers_theorem_value_comparison_theorem_l126_12675

/-- Represents the exchange rates between different holiday items -/
structure ExchangeRates where
  ornament_to_cracker : ℚ
  sparkler_to_garland : ℚ
  ornament_to_garland : ℚ

/-- Converts sparklers to crackers based on the given exchange rates -/
def sparklers_to_crackers (rates : ExchangeRates) (sparklers : ℚ) : ℚ :=
  let garlands := (sparklers / 5) * 2
  let ornaments := garlands * 4
  ornaments * rates.ornament_to_cracker

/-- Compares the value of ornaments and crackers to sparklers -/
def compare_values (rates : ExchangeRates) (ornaments crackers sparklers : ℚ) : Prop :=
  ornaments * rates.ornament_to_cracker + crackers > 
  (sparklers / 5) * 2 * 4 * rates.ornament_to_cracker

/-- Theorem stating the equivalence of 10 sparklers to 32 crackers -/
theorem sparklers_to_crackers_theorem (rates : ExchangeRates) : 
  sparklers_to_crackers rates 10 = 32 :=
by sorry

/-- Theorem comparing the value of 5 ornaments and 1 cracker to 2 sparklers -/
theorem value_comparison_theorem (rates : ExchangeRates) : 
  compare_values rates 5 1 2 :=
by sorry

end NUMINAMATH_CALUDE_sparklers_to_crackers_theorem_value_comparison_theorem_l126_12675


namespace NUMINAMATH_CALUDE_mom_t_shirt_purchase_l126_12604

/-- The number of packages of white t-shirts Mom bought -/
def num_packages : ℕ := 14

/-- The number of white t-shirts in each package -/
def t_shirts_per_package : ℕ := 5

/-- The total number of white t-shirts Mom bought -/
def total_t_shirts : ℕ := num_packages * t_shirts_per_package

theorem mom_t_shirt_purchase : total_t_shirts = 70 := by
  sorry

end NUMINAMATH_CALUDE_mom_t_shirt_purchase_l126_12604


namespace NUMINAMATH_CALUDE_swap_counts_correct_l126_12622

/-- Represents a circular sequence of letters -/
def CircularSequence := List Char

/-- Counts the minimum number of adjacent swaps needed to transform one sequence into another -/
def minAdjacentSwaps (seq1 seq2 : CircularSequence) : Nat :=
  sorry

/-- Counts the minimum number of arbitrary swaps needed to transform one sequence into another -/
def minArbitrarySwaps (seq1 seq2 : CircularSequence) : Nat :=
  sorry

/-- The two given sequences -/
def sequence1 : CircularSequence := ['A', 'Z', 'O', 'R', 'S', 'Z', 'Á', 'G', 'H', 'Á', 'Z', 'A']
def sequence2 : CircularSequence := ['S', 'Á', 'R', 'G', 'A', 'A', 'Z', 'H', 'O', 'Z', 'Z', 'Ā']

theorem swap_counts_correct :
  minAdjacentSwaps sequence1 sequence2 = 14 ∧
  minArbitrarySwaps sequence1 sequence2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_swap_counts_correct_l126_12622


namespace NUMINAMATH_CALUDE_equal_chance_in_all_methods_l126_12695

/-- Represents a sampling method -/
structure SamplingMethod where
  name : String
  equal_chance : Bool

/-- Simple random sampling -/
def simple_random_sampling : SamplingMethod :=
  { name := "Simple Random Sampling", equal_chance := true }

/-- Systematic sampling -/
def systematic_sampling : SamplingMethod :=
  { name := "Systematic Sampling", equal_chance := true }

/-- Stratified sampling -/
def stratified_sampling : SamplingMethod :=
  { name := "Stratified Sampling", equal_chance := true }

/-- Theorem: All three sampling methods have equal chance of selection for each individual -/
theorem equal_chance_in_all_methods :
  simple_random_sampling.equal_chance ∧
  systematic_sampling.equal_chance ∧
  stratified_sampling.equal_chance :=
by sorry

end NUMINAMATH_CALUDE_equal_chance_in_all_methods_l126_12695


namespace NUMINAMATH_CALUDE_simplify_fraction_l126_12643

theorem simplify_fraction (x : ℝ) (h : x > 0) :
  (Real.sqrt x * 3 * x^2) / (x * 6 * x) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l126_12643


namespace NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l126_12615

open Real

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (x - 1) / (Real.exp x))
  (h2 : ∀ t ∈ Set.Icc (1/2) 2, f t > t) :
  ∃ a, a > Real.exp 2 + 1/2 ∧ ∀ x ∈ Set.Icc (1/2) 2, (a - 1) / (Real.exp x) > x :=
sorry

end NUMINAMATH_CALUDE_function_inequality_implies_a_bound_l126_12615


namespace NUMINAMATH_CALUDE_right_triangle_angle_calculation_l126_12674

/-- In a triangle ABC with a right angle at A, prove that x = 10/3 degrees -/
theorem right_triangle_angle_calculation (x y : ℝ) : 
  x + y = 40 →
  3 * x + 2 * y = 90 →
  x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_calculation_l126_12674


namespace NUMINAMATH_CALUDE_odd_sum_probability_l126_12656

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 1

def odd_sum_pairs : Finset (ℕ × ℕ) :=
  (cards.product cards).filter (λ pair => pair.1 < pair.2 ∧ is_odd_sum pair)

theorem odd_sum_probability :
  (odd_sum_pairs.card : ℚ) / (cards.card.choose 2) = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l126_12656


namespace NUMINAMATH_CALUDE_initial_trees_count_l126_12638

theorem initial_trees_count (died cut left : ℕ) 
  (h1 : died = 15)
  (h2 : cut = 23)
  (h3 : left = 48) :
  died + cut + left = 86 := by
  sorry

end NUMINAMATH_CALUDE_initial_trees_count_l126_12638


namespace NUMINAMATH_CALUDE_power_six_equivalence_l126_12677

theorem power_six_equivalence (m : ℝ) : m^2 * m^4 = m^6 := by
  sorry

end NUMINAMATH_CALUDE_power_six_equivalence_l126_12677


namespace NUMINAMATH_CALUDE_vector_parallel_and_dot_product_l126_12607

/-- Given two vectors a and b, and an angle α, prove the following statements -/
theorem vector_parallel_and_dot_product (α : Real) 
    (h1 : α ∈ Set.Ioo 0 (π/4)) 
    (a : Fin 2 → Real) (b : Fin 2 → Real)
    (h2 : a = λ i => if i = 0 then 2 * Real.sin α else 1)
    (h3 : b = λ i => if i = 0 then Real.cos α else 1) :
  (∃ (k : Real), a = k • b → Real.tan α = 1/2) ∧
  (a • b = 9/5 → Real.sin (2*α + π/4) = 7*Real.sqrt 2/10) := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_and_dot_product_l126_12607


namespace NUMINAMATH_CALUDE_shaded_area_of_grid_square_l126_12606

theorem shaded_area_of_grid_square (d : ℝ) (h1 : d = 10) : 
  let s := d / Real.sqrt 2
  let small_square_side := s / 5
  let small_square_area := small_square_side ^ 2
  let total_area := 25 * small_square_area
  total_area = 50 := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_grid_square_l126_12606


namespace NUMINAMATH_CALUDE_arrangement_probability_l126_12684

/-- The probability of arranging n(n + 1)/2 distinct numbers into n rows,
    where the i-th row has i numbers, such that the largest number in each row
    is smaller than the largest number in all rows with more numbers. -/
def probability (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (Nat.factorial (n + 1) : ℚ)

/-- Theorem stating that the probability of the described arrangement
    is equal to 2^n / (n+1)! -/
theorem arrangement_probability (n : ℕ) :
  probability n = (2 ^ n : ℚ) / (Nat.factorial (n + 1) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_arrangement_probability_l126_12684


namespace NUMINAMATH_CALUDE_absolute_value_equation_l126_12633

theorem absolute_value_equation : ∃! x : ℝ, |x - 30| + |x - 24| = |3*x - 72| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l126_12633


namespace NUMINAMATH_CALUDE_negative_product_implies_positive_fraction_l126_12698

theorem negative_product_implies_positive_fraction
  (x y z : ℝ) (h : x * y^3 * z^2 < 0) (hy : y ≠ 0) :
  -(x^3 * z^4) / y^5 > 0 :=
by sorry

end NUMINAMATH_CALUDE_negative_product_implies_positive_fraction_l126_12698


namespace NUMINAMATH_CALUDE_isosceles_triangle_l126_12668

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  c / b = Real.cos C / Real.cos B →
  C = B :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l126_12668


namespace NUMINAMATH_CALUDE_sally_napkins_l126_12653

def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54
def napkin_length : ℕ := 6
def napkin_width : ℕ := 7
def total_material : ℕ := 5844

theorem sally_napkins :
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let remaining_material := total_material - tablecloth_area
  remaining_material / napkin_area = 8 := by sorry

end NUMINAMATH_CALUDE_sally_napkins_l126_12653


namespace NUMINAMATH_CALUDE_race_difference_l126_12678

/-- Represents a racer in the competition -/
structure Racer where
  time : ℝ  -- Time taken to complete the race in seconds
  speed : ℝ  -- Speed of the racer in meters per second

/-- Calculates the distance covered by a racer in a given time -/
def distance_covered (r : Racer) (t : ℝ) : ℝ := r.speed * t

theorem race_difference (race_distance : ℝ) (a b : Racer) 
  (h1 : race_distance = 80)
  (h2 : a.time = 20)
  (h3 : b.time = 25)
  (h4 : a.speed = race_distance / a.time)
  (h5 : b.speed = race_distance / b.time) :
  race_distance - distance_covered b a.time = 16 := by
  sorry

end NUMINAMATH_CALUDE_race_difference_l126_12678


namespace NUMINAMATH_CALUDE_molecule_count_l126_12646

-- Define Avogadro's constant
def avogadro_constant : ℝ := 6.022e23

-- Define the number of molecules
def number_of_molecules : ℝ := 3e26

-- Theorem to prove
theorem molecule_count : number_of_molecules = 3e26 := by
  sorry

end NUMINAMATH_CALUDE_molecule_count_l126_12646


namespace NUMINAMATH_CALUDE_solve_equation_l126_12681

theorem solve_equation : 
  ∃ x : ℝ, (2 * x + 10 = (1/2) * (5 * x + 30)) ∧ (x = -10) := by sorry

end NUMINAMATH_CALUDE_solve_equation_l126_12681


namespace NUMINAMATH_CALUDE_equation_roots_arithmetic_progression_l126_12620

theorem equation_roots_arithmetic_progression (a : ℝ) : 
  (∃ r d : ℝ, (∀ x : ℝ, x^8 + a*x^4 + 1 = 0 ↔ 
    x = (r - 3*d)^(1/4) ∨ x = (r - d)^(1/4) ∨ x = (r + d)^(1/4) ∨ x = (r + 3*d)^(1/4))) 
  → a = -82/9 :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_arithmetic_progression_l126_12620


namespace NUMINAMATH_CALUDE_two_digit_number_proof_l126_12666

theorem two_digit_number_proof : 
  ∀ n : ℕ, 
  (10 ≤ n ∧ n < 100) → -- two-digit number
  (∃ x y : ℕ, n = 10 * x + y ∧ y = x + 3 ∧ n = y * y) → -- conditions
  (n = 25 ∨ n = 36) := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_proof_l126_12666


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_solve_cubic_equation_l126_12616

-- Problem 1
theorem solve_quadratic_equation (x : ℝ) :
  4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2 := by
sorry

-- Problem 2
theorem solve_cubic_equation (x : ℝ) :
  8 * (x + 1)^3 = 27 ↔ x = 1/2 := by
sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_solve_cubic_equation_l126_12616


namespace NUMINAMATH_CALUDE_part_not_scrap_l126_12663

/-- The probability of producing scrap in the first process -/
def p1 : ℝ := 0.01

/-- The probability of producing scrap in the second process -/
def p2 : ℝ := 0.02

/-- The probability that a part is not scrap after two independent processes -/
def prob_not_scrap : ℝ := (1 - p1) * (1 - p2)

theorem part_not_scrap : prob_not_scrap = 0.9702 := by sorry

end NUMINAMATH_CALUDE_part_not_scrap_l126_12663


namespace NUMINAMATH_CALUDE_proportional_increase_l126_12612

/-- Given the equation 3x - 2y = 7, this theorem proves that y increases proportionally to x
    and determines the proportionality coefficient. -/
theorem proportional_increase (x y : ℝ) (h : 3 * x - 2 * y = 7) :
  ∃ (k b : ℝ), y = k * x + b ∧ k = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_proportional_increase_l126_12612


namespace NUMINAMATH_CALUDE_fraction_product_exponents_l126_12629

theorem fraction_product_exponents : (3 / 4 : ℚ)^5 * (4 / 3 : ℚ)^2 = 8 / 19 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_exponents_l126_12629


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l126_12685

/-- The total number of dimes Melanie has after receiving gifts from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Given Melanie's initial dimes and the dimes given by her parents, 
    the total number of dimes Melanie has now is 83. -/
theorem melanie_dimes_count : total_dimes 19 39 25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l126_12685


namespace NUMINAMATH_CALUDE_pictures_per_album_l126_12628

theorem pictures_per_album (total_pictures : ℕ) (num_albums : ℕ) 
  (h1 : total_pictures = 480) (h2 : num_albums = 24) :
  total_pictures / num_albums = 20 := by
  sorry

end NUMINAMATH_CALUDE_pictures_per_album_l126_12628


namespace NUMINAMATH_CALUDE_round_trip_speed_l126_12624

/-- Proves that given specific conditions for a round trip, the return speed must be 48 mph -/
theorem round_trip_speed (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) (speed_ba : ℝ) : 
  distance = 120 →
  speed_ab = 80 →
  avg_speed = 60 →
  (2 * distance) / (distance / speed_ab + distance / speed_ba) = avg_speed →
  speed_ba = 48 := by
sorry

end NUMINAMATH_CALUDE_round_trip_speed_l126_12624


namespace NUMINAMATH_CALUDE_travel_distance_l126_12601

theorem travel_distance (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 75 → time = 4 → distance = speed * time → distance = 300 := by
  sorry

end NUMINAMATH_CALUDE_travel_distance_l126_12601


namespace NUMINAMATH_CALUDE_complex_number_in_quadrant_iv_l126_12697

/-- The complex number (2-i)/(1+i) corresponds to a point in Quadrant IV of the complex plane -/
theorem complex_number_in_quadrant_iv : 
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_quadrant_iv_l126_12697


namespace NUMINAMATH_CALUDE_initial_ribbon_amount_l126_12670

/-- The number of gifts Josh is preparing -/
def num_gifts : ℕ := 6

/-- The amount of ribbon used for each gift in yards -/
def ribbon_per_gift : ℕ := 2

/-- The amount of ribbon left after preparing the gifts in yards -/
def leftover_ribbon : ℕ := 6

/-- Theorem: Josh initially has 18 yards of ribbon -/
theorem initial_ribbon_amount :
  num_gifts * ribbon_per_gift + leftover_ribbon = 18 := by
  sorry

end NUMINAMATH_CALUDE_initial_ribbon_amount_l126_12670


namespace NUMINAMATH_CALUDE_complex_exponential_sum_l126_12661

theorem complex_exponential_sum (γ δ : ℝ) :
  Complex.exp (Complex.I * γ) + Complex.exp (Complex.I * δ) = (2 / 5 : ℂ) + Complex.I * (1 / 2 : ℂ) →
  Complex.exp (-Complex.I * γ) + Complex.exp (-Complex.I * δ) = (2 / 5 : ℂ) - Complex.I * (1 / 2 : ℂ) :=
by sorry

end NUMINAMATH_CALUDE_complex_exponential_sum_l126_12661


namespace NUMINAMATH_CALUDE_free_throw_difference_l126_12682

/-- The number of free-throws made by each player in one minute -/
structure FreeThrows where
  deshawn : ℕ
  kayla : ℕ
  annieka : ℕ

/-- The conditions of the basketball free-throw practice -/
def free_throw_practice (ft : FreeThrows) : Prop :=
  ft.deshawn = 12 ∧
  ft.kayla = ft.deshawn + ft.deshawn / 2 ∧
  ft.annieka = 14 ∧
  ft.annieka < ft.kayla

/-- The theorem stating the difference between Kayla's and Annieka's free-throws -/
theorem free_throw_difference (ft : FreeThrows) 
  (h : free_throw_practice ft) : ft.kayla - ft.annieka = 4 := by
  sorry

#check free_throw_difference

end NUMINAMATH_CALUDE_free_throw_difference_l126_12682


namespace NUMINAMATH_CALUDE_rotation_180_complex_l126_12690

def rotate_180_degrees (z : ℂ) : ℂ := -z

theorem rotation_180_complex :
  rotate_180_degrees (3 - 4*I) = -3 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_rotation_180_complex_l126_12690


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l126_12602

theorem system_of_equations_solution :
  ∃! (x y : ℝ), (2 * x - y = 3) ∧ (3 * x + 2 * y = 8) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l126_12602


namespace NUMINAMATH_CALUDE_divisible_by_nine_sequence_l126_12637

theorem divisible_by_nine_sequence (n : ℕ) : 
  (n % 9 = 0) ∧ 
  (n + 54 ≤ 97) ∧ 
  (∀ k : ℕ, k < 7 → (n + 9 * k) % 9 = 0) →
  n = 36 :=
sorry

end NUMINAMATH_CALUDE_divisible_by_nine_sequence_l126_12637


namespace NUMINAMATH_CALUDE_exists_valid_set_l126_12627

def is_valid_set (S : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (n ∈ S ↔ 
    (∃ a b : ℕ, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b = n) ∨
    (∃ a b : ℕ, a ∉ S ∧ b ∉ S ∧ a ≠ b ∧ a > 0 ∧ b > 0 ∧ a + b = n))

theorem exists_valid_set : ∃ S : Set ℕ, is_valid_set S := by
  sorry

end NUMINAMATH_CALUDE_exists_valid_set_l126_12627


namespace NUMINAMATH_CALUDE_curve_symmetry_l126_12659

-- Define the original curve E
def E (x y : ℝ) : Prop := 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of symmetry l
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric curve E'
def E' (x y : ℝ) : Prop := 5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0

-- Theorem statement
theorem curve_symmetry :
  ∀ (x y : ℝ), E x y ↔ ∃ (x' y' : ℝ), l ((x + x') / 2) ((y + y') / 2) ∧ E' x' y' :=
sorry

end NUMINAMATH_CALUDE_curve_symmetry_l126_12659


namespace NUMINAMATH_CALUDE_complex_quadrant_l126_12647

theorem complex_quadrant (z : ℂ) (h : -2 * I * z = 1 - I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_complex_quadrant_l126_12647


namespace NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l126_12672

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  let bridge_length := total_distance - train_length
  bridge_length

/-- Proves that the bridge length is approximately 140 meters --/
theorem bridge_length_proof :
  let train_length : ℝ := 360
  let train_speed_kmh : ℝ := 56
  let time_to_pass : ℝ := 32.142857142857146
  let calculated_bridge_length := bridge_length_calculation train_length train_speed_kmh time_to_pass
  ∃ ε > 0, |calculated_bridge_length - 140| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_bridge_length_proof_l126_12672


namespace NUMINAMATH_CALUDE_fraction_product_sum_l126_12648

theorem fraction_product_sum : (1/3 : ℚ) * (17/6 : ℚ) * (3/7 : ℚ) + (1/4 : ℚ) * (1/8 : ℚ) = 101/672 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_sum_l126_12648


namespace NUMINAMATH_CALUDE_tan_product_zero_l126_12639

theorem tan_product_zero (a b : ℝ) 
  (h : 6 * (Real.cos a + Real.cos b) + 3 * (Real.cos a * Real.cos b + 1) = 0) : 
  Real.tan (a / 2) * Real.tan (b / 2) = 0 := by
sorry

end NUMINAMATH_CALUDE_tan_product_zero_l126_12639


namespace NUMINAMATH_CALUDE_inverse_proportion_order_l126_12654

theorem inverse_proportion_order (k : ℝ) :
  let f (x : ℝ) := (k^2 + 1) / x
  let y₁ := f (-1)
  let y₂ := f 1
  let y₃ := f 2
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_order_l126_12654


namespace NUMINAMATH_CALUDE_imaginary_axis_length_of_given_hyperbola_l126_12680

/-- The length of the imaginary axis of a hyperbola -/
def imaginary_axis_length (a b : ℝ) : ℝ := 2 * b

/-- The equation of the hyperbola in standard form -/
def hyperbola_equation (x y a b : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

theorem imaginary_axis_length_of_given_hyperbola :
  ∃ (a b : ℝ), a^2 = 3 ∧ b^2 = 1 ∧
  (∀ x y : ℝ, hyperbola_equation x y a b ↔ x^2 / 3 - y^2 = 1) ∧
  imaginary_axis_length a b = 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_axis_length_of_given_hyperbola_l126_12680


namespace NUMINAMATH_CALUDE_yellow_parrots_count_l126_12641

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (green_fraction : ℚ) :
  total = 180 →
  red_fraction = 2/3 →
  green_fraction = 1/6 →
  (total : ℚ) * (1 - (red_fraction + green_fraction)) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_parrots_count_l126_12641


namespace NUMINAMATH_CALUDE_cross_in_square_l126_12650

theorem cross_in_square (a : ℝ) (h : a > 0) : 
  (2 * (a/2)^2 + 2 * (a/4)^2 = 810) → a = 36 := by
  sorry

end NUMINAMATH_CALUDE_cross_in_square_l126_12650


namespace NUMINAMATH_CALUDE_second_derivative_at_x₀_l126_12691

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the point x₀
def x₀ : ℝ := sorry

-- Define constants a and b
def a : ℝ := sorry
def b : ℝ := sorry

-- State the theorem
theorem second_derivative_at_x₀ (h : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * Δx^2) :
  deriv (deriv f) x₀ = 2 * b := by sorry

end NUMINAMATH_CALUDE_second_derivative_at_x₀_l126_12691


namespace NUMINAMATH_CALUDE_set_union_problem_l126_12686

theorem set_union_problem (a b : ℝ) : 
  let A : Set ℝ := {3, 2^a}
  let B : Set ℝ := {a, b}
  (A ∩ B = {2}) → (A ∪ B = {1, 2, 3}) := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l126_12686


namespace NUMINAMATH_CALUDE_divisible_by_eleven_l126_12630

theorem divisible_by_eleven (n : ℕ) : ∃ k : ℤ, 3^(2*n + 2) + 2^(6*n + 1) = 11 * k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eleven_l126_12630


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l126_12609

/-- A function that returns the product of the digits of a three-digit number -/
def digit_product (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * tens * ones

/-- A predicate that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 126 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l126_12609


namespace NUMINAMATH_CALUDE_masha_result_non_negative_l126_12621

theorem masha_result_non_negative (a b c d : ℝ) 
  (sum_eq_prod : a + b = c * d) 
  (prod_eq_sum : a * b = c + d) : 
  (a + 1) * (b + 1) * (c + 1) * (d + 1) ≥ 0 := by
sorry

end NUMINAMATH_CALUDE_masha_result_non_negative_l126_12621
