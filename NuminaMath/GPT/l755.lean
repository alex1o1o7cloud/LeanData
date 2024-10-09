import Mathlib

namespace H_H_H_one_eq_three_l755_75541

noncomputable def H : ℝ → ℝ := sorry

theorem H_H_H_one_eq_three :
  H 1 = -3 ∧ H (-3) = 3 ∧ H 3 = 3 → H (H (H 1)) = 3 :=
by
  sorry

end H_H_H_one_eq_three_l755_75541


namespace cube_largest_ne_sum_others_l755_75516

theorem cube_largest_ne_sum_others (n : ℕ) : (n + 1)^3 ≠ n^3 + (n - 1)^3 :=
by
  sorry

end cube_largest_ne_sum_others_l755_75516


namespace number_of_integer_solutions_l755_75529

theorem number_of_integer_solutions : 
  ∃ S : Finset ℤ, (∀ x ∈ S, (x + 3)^2 ≤ 4) ∧ S.card = 5 := by
  sorry

end number_of_integer_solutions_l755_75529


namespace find_length_of_AB_l755_75520

-- Definitions of the conditions
def areas_ratio (A B C D : Point) (areaABC areaADC : ℝ) :=
  (areaABC / areaADC) = (7 / 3)

def total_length (A B C D : Point) (AB CD : ℝ) :=
  AB + CD = 280

-- Statement of the proof problem
theorem find_length_of_AB
  (A B C D : Point)
  (AB CD : ℝ)
  (areaABC areaADC : ℝ)
  (h_height_not_zero : h ≠ 0) -- Assumption to ensure height is non-zero
  (h_areas_ratio : areas_ratio A B C D areaABC areaADC)
  (h_total_length : total_length A B C D AB CD) :
  AB = 196 := sorry

end find_length_of_AB_l755_75520


namespace extreme_values_sin_2x0_l755_75590

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos (Real.pi / 2 + x)^2 - 
  2 * Real.sin (Real.pi + x) * Real.cos x - Real.sqrt 3

-- Part (1)
theorem extreme_values : 
  (∀ x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), 1 ≤ f x ∧ f x ≤ 2) :=
sorry

-- Part (2)
theorem sin_2x0 (x0 : ℝ) (h : x0 ∈ Set.Icc (3 * Real.pi / 4) Real.pi) (hx : f (x0 - Real.pi / 6) = 10 / 13) : 
  Real.sin (2 * x0) = - (5 + 12 * Real.sqrt 3) / 26 :=
sorry

end extreme_values_sin_2x0_l755_75590


namespace john_total_spent_l755_75560

-- Define the initial conditions
def other_toys_cost : ℝ := 1000
def lightsaber_cost : ℝ := 2 * other_toys_cost

-- Define the total cost spent by John
def total_cost : ℝ := other_toys_cost + lightsaber_cost

-- Prove that the total cost is $3000
theorem john_total_spent :
  total_cost = 3000 :=
by
  -- Sorry will be used to skip the proof
  sorry

end john_total_spent_l755_75560


namespace mary_number_l755_75504

-- Definitions of the properties and conditions
def is_two_digit_number (x : ℕ) : Prop :=
  10 ≤ x ∧ x < 100

def switch_digits (n : ℕ) : ℕ :=
  let a := n / 10
  let b := n % 10
  10 * b + a

def conditions_met (x : ℕ) : Prop :=
  is_two_digit_number x ∧ 91 ≤ switch_digits (4 * x - 7) ∧ switch_digits (4 * x - 7) ≤ 95

-- The statement to prove
theorem mary_number : ∃ x : ℕ, conditions_met x ∧ x = 14 :=
by {
  sorry
}

end mary_number_l755_75504


namespace minimize_total_cost_l755_75537

open Real

noncomputable def total_cost (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100) : ℝ :=
  (130 / x) * 2 * (2 + (x^2 / 360)) + (14 * 130 / x)

theorem minimize_total_cost :
  ∀ (x : ℝ) (h : 50 ≤ x ∧ x ≤ 100),
  total_cost x h = (2340 / x) + (13 * x / 18)
  ∧ (x = 18 * sqrt 10 → total_cost x h = 26 * sqrt 10) :=
by
  sorry

end minimize_total_cost_l755_75537


namespace y_intercept_l755_75586

theorem y_intercept : ∀ (x y : ℝ), 4 * x + 7 * y = 28 → (0, 4) = (0, y) :=
by
  intros x y h
  sorry

end y_intercept_l755_75586


namespace carla_gas_cost_l755_75577

theorem carla_gas_cost:
  let distance_grocery := 8
  let distance_school := 6
  let distance_bank := 12
  let distance_practice := 9
  let distance_dinner := 15
  let distance_home := 2 * distance_practice
  let total_distance := distance_grocery + distance_school + distance_bank + distance_practice + distance_dinner + distance_home
  let miles_per_gallon := 25
  let price_per_gallon_first := 2.35
  let price_per_gallon_second := 2.65
  let total_gallons := total_distance / miles_per_gallon
  let gallons_per_fill_up := total_gallons / 2
  let cost_first := gallons_per_fill_up * price_per_gallon_first
  let cost_second := gallons_per_fill_up * price_per_gallon_second
  let total_cost := cost_first + cost_second
  total_cost = 6.80 :=
by sorry

end carla_gas_cost_l755_75577


namespace hours_per_day_l755_75500

variable (M : ℕ)

noncomputable def H : ℕ := 9
noncomputable def D1 : ℕ := 24
noncomputable def Men2 : ℕ := 12
noncomputable def D2 : ℕ := 16

theorem hours_per_day (H_new : ℝ) : 
  (M * H * D1 : ℝ) = (Men2 * H_new * D2) → 
  H_new = (M * 9 : ℝ) / 8 := 
  sorry

end hours_per_day_l755_75500


namespace linear_function_passing_quadrants_l755_75547

theorem linear_function_passing_quadrants (b : ℝ) :
  (∀ x : ℝ, (y = x + b) ∧ (y > 0 ↔ (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0))) →
  b > 0 :=
sorry

end linear_function_passing_quadrants_l755_75547


namespace christmas_day_december_25_l755_75594

-- Define the conditions
def is_thursday (d: ℕ) : Prop := d % 7 = 4
def thanksgiving := 26
def december_christmas := 25

-- Define the problem as a proof problem
theorem christmas_day_december_25 :
  is_thursday (thanksgiving) → thanksgiving = 26 →
  december_christmas = 25 → 
  30 - 26 + 25 = 28 → 
  is_thursday (30 - 26 + 25) :=
by
  intro h_thursday h_thanksgiving h_christmas h_days
  -- skipped proof
  sorry

end christmas_day_december_25_l755_75594


namespace x_intercept_of_quadratic_l755_75544

theorem x_intercept_of_quadratic (a b c : ℝ) (h_vertex : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 4 ∧ y = -2) 
(h_intercept : ∃ x y : ℝ, y = a * x^2 + b * x + c ∧ x = 1 ∧ y = 0) : 
∃ x : ℝ, x = 7 ∧ ∃ y : ℝ, y = a * x^2 + b * x + c ∧ y = 0 :=
sorry

end x_intercept_of_quadratic_l755_75544


namespace percentage_of_sikh_boys_l755_75517

-- Define the conditions
def total_boys : ℕ := 650
def muslim_boys : ℕ := (44 * total_boys) / 100
def hindu_boys : ℕ := (28 * total_boys) / 100
def other_boys : ℕ := 117
def sikh_boys : ℕ := total_boys - (muslim_boys + hindu_boys + other_boys)

-- Define and prove the theorem
theorem percentage_of_sikh_boys : (sikh_boys * 100) / total_boys = 10 :=
by
  have h_muslims: muslim_boys = 286 := by sorry
  have h_hindus: hindu_boys = 182 := by sorry
  have h_total: muslim_boys + hindu_boys + other_boys = 585 := by sorry
  have h_sikhs: sikh_boys = 65 := by sorry
  have h_percentage: (65 * 100) / 650 = 10 := by sorry
  exact h_percentage

end percentage_of_sikh_boys_l755_75517


namespace coolant_left_l755_75566

theorem coolant_left (initial_volume : ℝ) (initial_concentration : ℝ) (x : ℝ) (replacement_concentration : ℝ) (final_concentration : ℝ) :
  initial_volume = 19 ∧ 
  initial_concentration = 0.30 ∧ 
  replacement_concentration = 0.80 ∧ 
  final_concentration = 0.50 ∧ 
  (0.30 * initial_volume - 0.30 * x + 0.80 * x = 0.50 * initial_volume) →
  initial_volume - x = 11.4 :=
by sorry

end coolant_left_l755_75566


namespace jumps_per_second_l755_75578

-- Define the conditions and known values
def record_jumps : ℕ := 54000
def hours : ℕ := 5
def seconds_per_hour : ℕ := 3600

-- Define the target question as a theorem to prove
theorem jumps_per_second :
  (record_jumps / (hours * seconds_per_hour)) = 3 := by
  sorry

end jumps_per_second_l755_75578


namespace sum_of_coefficients_l755_75556

-- Define a namespace to encapsulate the problem
namespace PolynomialCoefficients

-- Problem statement as a Lean theorem
theorem sum_of_coefficients (α β : ℂ) (h1 : α + β = 1) (h2 : α * β = 1) :
  α^2005 + β^2005 = 1 :=
sorry -- Placeholder for the proof

end PolynomialCoefficients

end sum_of_coefficients_l755_75556


namespace fraction_solved_l755_75536

theorem fraction_solved (N f : ℝ) (h1 : N * f^2 = 6^3) (h2 : N * f^2 = 7776) : f = 1 / 6 :=
by sorry

end fraction_solved_l755_75536


namespace melanie_dimes_l755_75528

theorem melanie_dimes (original_dimes dad_dimes mom_dimes total_dimes : ℕ) :
  original_dimes = 7 →
  mom_dimes = 4 →
  total_dimes = 19 →
  (total_dimes = original_dimes + dad_dimes + mom_dimes) →
  dad_dimes = 8 :=
by
  intros h1 h2 h3 h4
  sorry -- The proof is omitted as instructed.

end melanie_dimes_l755_75528


namespace solutions_to_deqs_l755_75572

noncomputable def x1 (t : ℝ) : ℝ := -1 / t^2
noncomputable def x2 (t : ℝ) : ℝ := -t * Real.log t

theorem solutions_to_deqs (t : ℝ) (ht : 0 < t) :
  (deriv x1 t = 2 * t * (x1 t)^2) ∧ (deriv x2 t = x2 t / t - 1) :=
by
  sorry

end solutions_to_deqs_l755_75572


namespace triangle_rectangle_ratio_l755_75511

/--
An equilateral triangle and a rectangle both have perimeters of 60 inches.
The rectangle has a length to width ratio of 2:1.
We need to prove that the ratio of the length of the side of the triangle to
the length of the rectangle is 1.
-/
theorem triangle_rectangle_ratio
  (triangle_perimeter rectangle_perimeter : ℕ)
  (triangle_side rectangle_length rectangle_width : ℕ)
  (h1 : triangle_perimeter = 60)
  (h2 : rectangle_perimeter = 60)
  (h3 : rectangle_length = 2 * rectangle_width)
  (h4 : triangle_side = triangle_perimeter / 3)
  (h5 : rectangle_perimeter = 2 * rectangle_length + 2 * rectangle_width)
  (h6 : rectangle_width = 10)
  (h7 : rectangle_length = 20)
  : triangle_side / rectangle_length = 1 := 
sorry

end triangle_rectangle_ratio_l755_75511


namespace divisor_of_100_by_quotient_9_and_remainder_1_l755_75588

theorem divisor_of_100_by_quotient_9_and_remainder_1 :
  ∃ d : ℕ, 100 = d * 9 + 1 ∧ d = 11 :=
by
  sorry

end divisor_of_100_by_quotient_9_and_remainder_1_l755_75588


namespace cole_average_speed_l755_75558

noncomputable def cole_average_speed_to_work : ℝ :=
  let time_to_work := 1.2
  let return_trip_speed := 105
  let total_round_trip_time := 2
  let time_to_return := total_round_trip_time - time_to_work
  let distance_to_work := return_trip_speed * time_to_return
  distance_to_work / time_to_work

theorem cole_average_speed : cole_average_speed_to_work = 70 := by
  sorry

end cole_average_speed_l755_75558


namespace dorothy_score_l755_75503

theorem dorothy_score (T I D : ℝ) 
  (hT : T = 2 * I)
  (hI : I = (3 / 5) * D)
  (hSum : T + I + D = 252) : 
  D = 90 := 
by {
  sorry
}

end dorothy_score_l755_75503


namespace fifth_number_in_eighth_row_l755_75501

theorem fifth_number_in_eighth_row : 
  (∀ n : ℕ, ∃ k : ℕ, k = n * n ∧ 
    ∀ m : ℕ, 1 ≤ m ∧ m ≤ n → 
      k - (n - m) = 54 → m = 5 ∧ n = 8) := by sorry

end fifth_number_in_eighth_row_l755_75501


namespace money_distribution_l755_75571

theorem money_distribution (a b : ℝ) 
  (h1 : 4 * a - b = 40)
  (h2 : 6 * a + b = 110) :
  a = 15 ∧ b = 20 :=
by
  sorry

end money_distribution_l755_75571


namespace find_a_b_l755_75561

theorem find_a_b (a b : ℝ) : 
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0) → a = 5 ∧ b = -6 :=
sorry

end find_a_b_l755_75561


namespace no_three_parabolas_l755_75523

theorem no_three_parabolas (a b c : ℝ) : ¬ (b^2 > 4*a*c ∧ a^2 > 4*b*c ∧ c^2 > 4*a*b) := by
  sorry

end no_three_parabolas_l755_75523


namespace find_x_l755_75593

theorem find_x : 
  (5 * 12 / (180 / 3) = 1) → (∃ x : ℕ, 1 + x = 81 ∧ x = 80) :=
by
  sorry

end find_x_l755_75593


namespace bedroom_light_energy_usage_l755_75564

-- Define the conditions and constants
def noahs_bedroom_light_usage (W : ℕ) : ℕ := W
def noahs_office_light_usage (W : ℕ) : ℕ := 3 * W
def noahs_living_room_light_usage (W : ℕ) : ℕ := 4 * W
def total_energy_used (W : ℕ) : ℕ := 2 * (noahs_bedroom_light_usage W + noahs_office_light_usage W + noahs_living_room_light_usage W)
def energy_consumption := 96

-- The main theorem to be proven
theorem bedroom_light_energy_usage : ∃ W : ℕ, total_energy_used W = energy_consumption ∧ W = 6 :=
by
  sorry

end bedroom_light_energy_usage_l755_75564


namespace flower_count_l755_75591

variables (o y p : ℕ)

theorem flower_count (h1 : y + p = 7) (h2 : o + p = 10) (h3 : o + y = 5) : o + y + p = 11 := sorry

end flower_count_l755_75591


namespace number_of_true_statements_is_two_l755_75514

def line_plane_geometry : Type :=
  -- Types representing lines and planes
  sorry

def l : line_plane_geometry := sorry
def alpha : line_plane_geometry := sorry
def m : line_plane_geometry := sorry
def beta : line_plane_geometry := sorry

def is_perpendicular (x y : line_plane_geometry) : Prop := sorry
def is_parallel (x y : line_plane_geometry) : Prop := sorry
def is_contained_in (x y : line_plane_geometry) : Prop := sorry

axiom l_perpendicular_alpha : is_perpendicular l alpha
axiom m_contained_in_beta : is_contained_in m beta

def statement_1 : Prop := is_parallel alpha beta → is_perpendicular l m
def statement_2 : Prop := is_perpendicular alpha beta → is_parallel l m
def statement_3 : Prop := is_parallel l m → is_perpendicular alpha beta

theorem number_of_true_statements_is_two : 
  (statement_1 ↔ true) ∧ (statement_2 ↔ false) ∧ (statement_3 ↔ true) := 
sorry

end number_of_true_statements_is_two_l755_75514


namespace largest_prime_divisor_of_360_is_5_l755_75534

theorem largest_prime_divisor_of_360_is_5 (p : ℕ) (hp₁ : Nat.Prime p) (hp₂ : p ∣ 360) : p ≤ 5 :=
by 
sorry

end largest_prime_divisor_of_360_is_5_l755_75534


namespace smallest_n_for_4n_square_and_5n_cube_l755_75502

theorem smallest_n_for_4n_square_and_5n_cube :
  ∃ (n : ℕ), (n > 0 ∧ (∃ k : ℕ, 4 * n = k^2) ∧ (∃ m : ℕ, 5 * n = m^3)) ∧ n = 400 :=
by
  sorry

end smallest_n_for_4n_square_and_5n_cube_l755_75502


namespace negation_of_proposition_l755_75563

theorem negation_of_proposition :
  (¬ ∀ (a b : ℝ), (a = b → a^2 = a * b)) = ∀ (a b : ℝ), (a ≠ b → a^2 ≠ a * b) :=
sorry

end negation_of_proposition_l755_75563


namespace green_more_than_red_l755_75555

def red_peaches : ℕ := 7
def green_peaches : ℕ := 8

theorem green_more_than_red : green_peaches - red_peaches = 1 := by
  sorry

end green_more_than_red_l755_75555


namespace area_enclosed_by_cosine_l755_75584

theorem area_enclosed_by_cosine :
  ∫ x in -Real.pi..Real.pi, (1 + Real.cos x) = 2 * Real.pi := by
  sorry

end area_enclosed_by_cosine_l755_75584


namespace central_angle_proof_l755_75568

noncomputable def central_angle (l r : ℝ) : ℝ :=
  l / r

theorem central_angle_proof :
  central_angle 300 100 = 3 :=
by
  -- The statement of the theorem aligns with the given problem conditions and the expected answer.
  sorry

end central_angle_proof_l755_75568


namespace check_correct_digit_increase_l755_75542

-- Definition of the numbers involved
def number1 : ℕ := 732
def number2 : ℕ := 648
def number3 : ℕ := 985
def given_sum : ℕ := 2455
def calc_sum : ℕ := number1 + number2 + number3
def difference : ℕ := given_sum - calc_sum

-- Specify the smallest digit that needs to be increased by 1
def smallest_digit_to_increase : ℕ := 8

-- Theorem to check the validity of the problem's claim
theorem check_correct_digit_increase :
  (smallest_digit_to_increase = 8) →
  (calc_sum + 10 = given_sum - 80) :=
by
  intro h
  sorry

end check_correct_digit_increase_l755_75542


namespace sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l755_75581

def seven_pow_seven_minus_seven_pow_four : ℤ := 7^7 - 7^4
def prime_factors_of_three_hundred_forty_two : List ℤ := [2, 3, 19]

theorem sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four : 
  let distinct_prime_factors := prime_factors_of_three_hundred_forty_two.head!
  + prime_factors_of_three_hundred_forty_two.tail!.head!
  + prime_factors_of_three_hundred_forty_two.tail!.tail!.head!
  seven_pow_seven_minus_seven_pow_four = 7^4 * (7^3 - 1) ∧
  7^3 - 1 = 342 ∧
  prime_factors_of_three_hundred_forty_two = [2, 3, 19] ∧
  distinct_prime_factors = 24 := 
sorry

end sum_of_distinct_prime_factors_of_seven_pow_seven_minus_seven_pow_four_l755_75581


namespace nat_nums_division_by_7_l755_75506

theorem nat_nums_division_by_7 (n : ℕ) : 
  (∃ q r, n = 7 * q + r ∧ q = r ∧ 1 ≤ r ∧ r < 7) ↔ 
  n = 8 ∨ n = 16 ∨ n = 24 ∨ n = 32 ∨ n = 40 ∨ n = 48 := by
  sorry

end nat_nums_division_by_7_l755_75506


namespace distinct_square_roots_l755_75570

theorem distinct_square_roots (m : ℝ) (h : 2 * m - 4 ≠ 3 * m - 1) : ∃ n : ℝ, (2 * m - 4) * (2 * m - 4) = n ∧ (3 * m - 1) * (3 * m - 1) = n ∧ n = 4 :=
by
  sorry

end distinct_square_roots_l755_75570


namespace determine_C_l755_75522
noncomputable def A : ℕ := sorry
noncomputable def B : ℕ := sorry
noncomputable def C : ℕ := sorry

-- Conditions
axiom cond1 : A + B + 1 = C + 10
axiom cond2 : B = A + 2

-- Proof statement
theorem determine_C : C = 1 :=
by {
  -- using the given conditions, deduce that C must equal 1
  sorry
}

end determine_C_l755_75522


namespace daily_reading_goal_l755_75519

-- Define the problem conditions
def total_days : ℕ := 30
def goal_pages : ℕ := 600
def busy_days_13_16 : ℕ := 4
def busy_days_20_25 : ℕ := 6
def flight_day : ℕ := 1
def flight_pages : ℕ := 100

-- Define the mathematical equivalent proof problem in Lean 4
theorem daily_reading_goal :
  (total_days - busy_days_13_16 - busy_days_20_25 - flight_day) * 27 + flight_pages >= goal_pages :=
by
  sorry

end daily_reading_goal_l755_75519


namespace sqrt_factorial_mul_factorial_l755_75589

theorem sqrt_factorial_mul_factorial :
  (Real.sqrt (Nat.factorial 4 * Nat.factorial 4) = 24) := by
  sorry

end sqrt_factorial_mul_factorial_l755_75589


namespace mushroom_picking_l755_75533

theorem mushroom_picking (n T : ℕ) (hn_min : n ≥ 5) (hn_max : n ≤ 7)
  (hmax : ∀ (M_max M_min : ℕ), M_max = T / 5 → M_min = T / 7 → 
    T ≠ 0 → M_max ≤ T / n ∧ M_min ≥ T / n) : n = 6 :=
by
  sorry

end mushroom_picking_l755_75533


namespace minutes_practiced_other_days_l755_75569

theorem minutes_practiced_other_days (total_hours : ℕ) (minutes_per_day : ℕ) (num_days : ℕ) :
  total_hours = 450 ∧ minutes_per_day = 86 ∧ num_days = 2 → (total_hours - num_days * minutes_per_day) = 278 := by
  sorry

end minutes_practiced_other_days_l755_75569


namespace candies_of_different_flavors_l755_75515

theorem candies_of_different_flavors (total_treats chewing_gums chocolate_bars : ℕ) (h1 : total_treats = 155) (h2 : chewing_gums = 60) (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := 
by 
  sorry

end candies_of_different_flavors_l755_75515


namespace rectangle_other_side_length_l755_75548

/-- Theorem: Consider a rectangle with one side of length 10 cm. Another rectangle of dimensions 
10 cm x 1 cm fits diagonally inside this rectangle. We need to prove that the length 
of the other side of the larger rectangle is 2.96 cm. -/
theorem rectangle_other_side_length :
  ∃ (x : ℝ), (x ≠ 0) ∧ (0 < x) ∧ (10 * 10 - x * x = 1 * 1) ∧ x = 2.96 :=
sorry

end rectangle_other_side_length_l755_75548


namespace gcd_problem_l755_75554

open Int -- Open the integer namespace to use gcd.

theorem gcd_problem : Int.gcd (Int.gcd 188094 244122) 395646 = 6 :=
by
  -- provide the proof here
  sorry

end gcd_problem_l755_75554


namespace solve_eqn_l755_75598

theorem solve_eqn (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 56) : x + y = 2 := by
  sorry

end solve_eqn_l755_75598


namespace second_to_last_digit_of_special_number_l755_75579

theorem second_to_last_digit_of_special_number :
  ∀ (N : ℕ), (N % 10 = 0) ∧ (∃ k : ℕ, k > 0 ∧ N = 2 * 5^k) →
  (N / 10) % 10 = 5 :=
by
  sorry

end second_to_last_digit_of_special_number_l755_75579


namespace smallest_four_digit_number_l755_75510

theorem smallest_four_digit_number (N : ℕ) (a b : ℕ) (h1 : N = 100 * a + b) (h2 : N = (a + b)^2) (h3 : 1000 ≤ N) (h4 : N < 10000) : N = 2025 :=
sorry

end smallest_four_digit_number_l755_75510


namespace tetrahedron_ratio_l755_75551

open Real

theorem tetrahedron_ratio (a b : ℝ) (h1 : a = PA ∧ PB = a) (h2 : PC = b ∧ AB = b ∧ BC = b ∧ CA = b) (h3 : a < b) :
  (sqrt 6 - sqrt 2) / 2 < a / b ∧ a / b < 1 :=
by
  sorry

end tetrahedron_ratio_l755_75551


namespace michael_left_money_l755_75531

def michael_initial_money : Nat := 100
def michael_spent_on_snacks : Nat := 25
def michael_spent_on_rides : Nat := 3 * michael_spent_on_snacks
def michael_spent_on_games : Nat := 15
def total_expenditure : Nat := michael_spent_on_snacks + michael_spent_on_rides + michael_spent_on_games
def michael_money_left : Nat := michael_initial_money - total_expenditure

theorem michael_left_money : michael_money_left = 15 := by
  sorry

end michael_left_money_l755_75531


namespace portions_of_milk_l755_75526

theorem portions_of_milk (liters_to_ml : ℕ) (total_liters : ℕ) (portion : ℕ) (total_volume_ml : ℕ) (num_portions : ℕ) :
  liters_to_ml = 1000 →
  total_liters = 2 →
  portion = 200 →
  total_volume_ml = total_liters * liters_to_ml →
  num_portions = total_volume_ml / portion →
  num_portions = 10 := by
  sorry

end portions_of_milk_l755_75526


namespace sequence_values_l755_75562

variable {a1 a2 b2 : ℝ}

theorem sequence_values
  (arithmetic : 2 * a1 = 1 + a2 ∧ 2 * a2 = a1 + 4)
  (geometric : b2 ^ 2 = 1 * 4) :
  (a1 + a2) / b2 = 5 / 2 :=
by
  sorry

end sequence_values_l755_75562


namespace gcd_78_36_l755_75545

theorem gcd_78_36 : Nat.gcd 78 36 = 6 :=
by
  sorry

end gcd_78_36_l755_75545


namespace tray_height_l755_75576

noncomputable def height_of_tray : ℝ :=
  let side_length := 120
  let cut_distance := 4 * Real.sqrt 2
  let angle := 45 * (Real.pi / 180)
  -- Define the function that calculates height based on given conditions
  
  sorry

theorem tray_height (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  side_length = 120 ∧ cut_distance = 4 * Real.sqrt 2 ∧ angle = 45 * (Real.pi / 180) →
  height_of_tray = 4 * Real.sqrt 2 :=
by
  intros
  unfold height_of_tray
  sorry

end tray_height_l755_75576


namespace proof_problem_l755_75530

axiom sqrt (x : ℝ) : ℝ
axiom cbrt (x : ℝ) : ℝ
noncomputable def sqrtValue : ℝ :=
  sqrt 81

theorem proof_problem (m n : ℝ) (hm : sqrt m = 3) (hn : cbrt n = -4) : sqrt (2 * m - n - 1) = 9 ∨ sqrt (2 * m - n - 1) = -9 :=
by
  sorry

end proof_problem_l755_75530


namespace integer_solutions_to_equation_l755_75582

-- Define the problem statement in Lean 4
theorem integer_solutions_to_equation :
  ∀ (x y : ℤ), (x ≠ 0) → (y ≠ 0) → (1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 19) →
      (x, y) = (38, 38) ∨ (x, y) = (380, 20) ∨ (x, y) = (-342, 18) ∨ 
      (x, y) = (20, 380) ∨ (x, y) = (18, -342) :=
by
  sorry

end integer_solutions_to_equation_l755_75582


namespace min_value_of_exponential_l755_75553

theorem min_value_of_exponential (x y : ℝ) (h : x + 2 * y = 3) : 2^x + 4^y = 4 * Real.sqrt 2 := by
  sorry

end min_value_of_exponential_l755_75553


namespace little_john_money_left_l755_75540

-- Define the variables with the given conditions
def initAmount : ℚ := 5.10
def spentOnSweets : ℚ := 1.05
def givenToEachFriend : ℚ := 1.00

-- The problem statement
theorem little_john_money_left :
  (initAmount - spentOnSweets - 2 * givenToEachFriend) = 2.05 :=
by
  sorry

end little_john_money_left_l755_75540


namespace weight_of_one_bowling_ball_l755_75573

def weight_of_one_canoe : ℕ := 35

def ten_bowling_balls_equal_four_canoes (W: ℕ) : Prop :=
  ∀ w, (10 * w = 4 * W)

theorem weight_of_one_bowling_ball (W: ℕ) (h : W = weight_of_one_canoe) : 
  (10 * 14 = 4 * W) → 14 = 140 / 10 :=
by
  intros H
  sorry

end weight_of_one_bowling_ball_l755_75573


namespace daves_earnings_l755_75525

theorem daves_earnings
  (hourly_wage : ℕ)
  (monday_hours : ℕ)
  (tuesday_hours : ℕ)
  (monday_earning : monday_hours * hourly_wage = 36)
  (tuesday_earning : tuesday_hours * hourly_wage = 12) :
  monday_hours * hourly_wage + tuesday_hours * hourly_wage = 48 :=
by
  sorry

end daves_earnings_l755_75525


namespace ratio_of_shorts_to_pants_is_half_l755_75595

-- Define the parameters
def shirts := 4
def pants := 2 * shirts
def total_clothes := 16

-- Define the number of shorts
def shorts := total_clothes - (shirts + pants)

-- Define the ratio
def ratio := shorts / pants

-- Prove the ratio is 1/2
theorem ratio_of_shorts_to_pants_is_half : ratio = 1 / 2 :=
by
  -- Start the proof, but leave it as sorry
  sorry

end ratio_of_shorts_to_pants_is_half_l755_75595


namespace jasmine_percentage_l755_75538

namespace ProofExample

variables (original_volume : ℝ) (initial_percent_jasmine : ℝ) (added_jasmine : ℝ) (added_water : ℝ)
variables (initial_jasmine : ℝ := initial_percent_jasmine * original_volume / 100)
variables (total_jasmine : ℝ := initial_jasmine + added_jasmine)
variables (total_volume : ℝ := original_volume + added_jasmine + added_water)
variables (final_percent_jasmine : ℝ := (total_jasmine / total_volume) * 100)

theorem jasmine_percentage 
  (h1 : original_volume = 80)
  (h2 : initial_percent_jasmine = 10)
  (h3 : added_jasmine = 8)
  (h4 : added_water = 12)
  : final_percent_jasmine = 16 := 
sorry

end ProofExample

end jasmine_percentage_l755_75538


namespace factorize_expression_l755_75552

variable (b : ℝ)

theorem factorize_expression : 2 * b^3 - 4 * b^2 + 2 * b = 2 * b * (b - 1)^2 := by
  sorry

end factorize_expression_l755_75552


namespace add_55_result_l755_75543

theorem add_55_result (x : ℤ) (h : x - 69 = 37) : x + 55 = 161 :=
sorry

end add_55_result_l755_75543


namespace max_x_minus_y_l755_75587

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^3 + y^3) = x + y) : x - y ≤ (Real.sqrt 2 / 2) :=
by {
  sorry
}

end max_x_minus_y_l755_75587


namespace solve_problem_l755_75507

theorem solve_problem (Δ q : ℝ) (h1 : 2 * Δ + q = 134) (h2 : 2 * (Δ + q) + q = 230) : Δ = 43 := by
  sorry

end solve_problem_l755_75507


namespace num_points_within_and_on_boundary_is_six_l755_75527

noncomputable def num_points_within_boundary : ℕ :=
  let points := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (4, 1)]
  points.length

theorem num_points_within_and_on_boundary_is_six :
  num_points_within_boundary = 6 :=
  by
    -- proof steps would go here
    sorry

end num_points_within_and_on_boundary_is_six_l755_75527


namespace arithmetic_sequence_identification_l755_75580

variable (a : ℕ → ℤ)
variable (d : ℤ)

def is_arithmetic (a : ℕ → ℤ) (d : ℤ) : Prop := ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_identification (h : is_arithmetic a d) :
  (is_arithmetic (fun n => a n + 3) d) ∧
  ¬ (is_arithmetic (fun n => a n ^ 2) d) ∧
  (is_arithmetic (fun n => a (n + 1) - a n) d) ∧
  (is_arithmetic (fun n => 2 * a n) (2 * d)) ∧
  (is_arithmetic (fun n => 2 * a n + n) (2 * d + 1)) :=
by
  sorry

end arithmetic_sequence_identification_l755_75580


namespace real_roots_system_l755_75509

theorem real_roots_system :
  ∃ (x y : ℝ), 
    (x * y * (x^2 + y^2) = 78 ∧ x^4 + y^4 = 97) ↔ 
    (x, y) = (3, 2) ∨ (x, y) = (2, 3) ∨ (x, y) = (-3, -2) ∨ (x, y) = (-2, -3) := 
by 
  sorry

end real_roots_system_l755_75509


namespace morales_sisters_revenue_l755_75539

variable (Gabriela Alba Maricela : Nat)
variable (trees_per_grove : Nat := 110)
variable (oranges_per_tree : (Nat × Nat × Nat) := (600, 400, 500))
variable (oranges_per_cup : Nat := 3)
variable (price_per_cup : Nat := 4)

theorem morales_sisters_revenue :
  let G := trees_per_grove * oranges_per_tree.fst
  let A := trees_per_grove * oranges_per_tree.snd
  let M := trees_per_grove * oranges_per_tree.snd.snd
  let total_oranges := G + A + M
  let total_cups := total_oranges / oranges_per_cup
  let total_revenue := total_cups * price_per_cup
  total_revenue = 220000 :=
by 
  sorry

end morales_sisters_revenue_l755_75539


namespace mark_increase_reading_time_l755_75532

theorem mark_increase_reading_time : 
  (let hours_per_day := 2
   let days_per_week := 7
   let desired_weekly_hours := 18
   let current_weekly_hours := hours_per_day * days_per_week
   let increase_per_week := desired_weekly_hours - current_weekly_hours
   increase_per_week = 4) :=
by
  let hours_per_day := 2
  let days_per_week := 7
  let desired_weekly_hours := 18
  let current_weekly_hours := hours_per_day * days_per_week
  let increase_per_week := desired_weekly_hours - current_weekly_hours
  have h1 : current_weekly_hours = 14 := by norm_num
  have h2 : increase_per_week = desired_weekly_hours - current_weekly_hours := rfl
  have h3 : increase_per_week = 18 - 14 := by rw [h2, h1]
  have h4 : increase_per_week = 4 := by norm_num
  exact h4

end mark_increase_reading_time_l755_75532


namespace fraction_left_after_3_days_l755_75565

-- Defining work rates of A and B
def A_rate := 1 / 15
def B_rate := 1 / 20

-- Total work rate of A and B when working together
def combined_rate := A_rate + B_rate

-- Work completed by A and B in 3 days
def work_done := 3 * combined_rate

-- Fraction of work left
def fraction_work_left := 1 - work_done

-- Statement to prove:
theorem fraction_left_after_3_days : fraction_work_left = 13 / 20 :=
by
  have A_rate_def: A_rate = 1 / 15 := rfl
  have B_rate_def: B_rate = 1 / 20 := rfl
  have combined_rate_def: combined_rate = A_rate + B_rate := rfl
  have work_done_def: work_done = 3 * combined_rate := rfl
  have fraction_work_left_def: fraction_work_left = 1 - work_done := rfl
  sorry

end fraction_left_after_3_days_l755_75565


namespace problem_a9_b9_l755_75574

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions
axiom h1 : a + b = 1
axiom h2 : a^2 + b^2 = 3
axiom h3 : a^3 + b^3 = 4
axiom h4 : a^4 + b^4 = 7
axiom h5 : a^5 + b^5 = 11

-- Prove the goal
theorem problem_a9_b9 : a^9 + b^9 = 76 :=
by
  -- the proof will come here
  sorry

end problem_a9_b9_l755_75574


namespace water_amount_in_sport_formulation_l755_75575

/-
The standard formulation has the ratios:
F : CS : W = 1 : 12 : 30
Where F is flavoring, CS is corn syrup, and W is water.
-/

def standard_flavoring_ratio : ℚ := 1
def standard_corn_syrup_ratio : ℚ := 12
def standard_water_ratio : ℚ := 30

/-
In the sport formulation:
1) The ratio of flavoring to corn syrup is three times as great as in the standard formulation.
2) The ratio of flavoring to water is half that of the standard formulation.
-/
def sport_flavor_to_corn_ratio : ℚ := 3 * (standard_flavoring_ratio / standard_corn_syrup_ratio)
def sport_flavor_to_water_ratio : ℚ := 1 / 2 * (standard_flavoring_ratio / standard_water_ratio)

/-
The sport formulation contains 6 ounces of corn syrup.
The target is to find the amount of water in the sport formulation.
-/
def corn_syrup_in_sport_formulation : ℚ := 6
def flavoring_in_sport_formulation : ℚ := sport_flavor_to_corn_ratio * corn_syrup_in_sport_formulation

def water_in_sport_formulation : ℚ := 
  (flavoring_in_sport_formulation / sport_flavor_to_water_ratio)

theorem water_amount_in_sport_formulation : water_in_sport_formulation = 90 := by
  sorry

end water_amount_in_sport_formulation_l755_75575


namespace pencil_distribution_l755_75546

theorem pencil_distribution (C C' : ℕ) (pencils : ℕ) (remaining : ℕ) (less_per_class : ℕ) 
  (original_classes : C = 4) 
  (total_pencils : pencils = 172) 
  (remaining_pencils : remaining = 7) 
  (less_pencils : less_per_class = 28)
  (actual_classes : C' > C) 
  (distribution_mistake : (pencils - remaining) / C' + less_per_class = pencils / C) :
  C' = 11 := 
sorry

end pencil_distribution_l755_75546


namespace am_gm_inequality_example_am_gm_inequality_equality_condition_l755_75583

theorem am_gm_inequality_example (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y^3) * (x^3 + y) ≥ 4 * x^2 * y^2 :=
sorry

theorem am_gm_inequality_equality_condition (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  ((x + y^3) * (x^3 + y) = 4 * x^2 * y^2) ↔ (x = 0 ∧ y = 0 ∨ x = 1 ∧ y = 1) :=
sorry

end am_gm_inequality_example_am_gm_inequality_equality_condition_l755_75583


namespace magic_square_base_l755_75550

theorem magic_square_base :
  ∃ b : ℕ, (b + 1 + (b + 5) + 2 = 9 + (b + 3)) ∧ b = 3 :=
by
  use 3
  -- Proof in Lean goes here
  sorry

end magic_square_base_l755_75550


namespace polyhedron_with_n_edges_l755_75513

noncomputable def construct_polyhedron_with_n_edges (n : ℤ) : Prop :=
  ∃ (k : ℤ) (m : ℤ), (k = 8 ∨ k = 9 ∨ k = 10) ∧ (n = k + 3 * m)

theorem polyhedron_with_n_edges (n : ℤ) (h : n ≥ 8) : 
  construct_polyhedron_with_n_edges n :=
sorry

end polyhedron_with_n_edges_l755_75513


namespace time_in_1876_minutes_from_6AM_is_116PM_l755_75535

def minutesToTime (startTime : Nat) (minutesToAdd : Nat) : Nat × Nat :=
  let totalMinutes := startTime + minutesToAdd
  let totalHours := totalMinutes / 60
  let remainderMinutes := totalMinutes % 60
  let resultHours := (totalHours % 24)
  (resultHours, remainderMinutes)

theorem time_in_1876_minutes_from_6AM_is_116PM :
  minutesToTime (6 * 60) 1876 = (13, 16) :=
  sorry

end time_in_1876_minutes_from_6AM_is_116PM_l755_75535


namespace optimal_saving_is_45_cents_l755_75592

def initial_price : ℝ := 18
def fixed_discount : ℝ := 3
def percentage_discount : ℝ := 0.15

def price_after_fixed_discount (price fixed_discount : ℝ) : ℝ :=
  price - fixed_discount

def price_after_percentage_discount (price percentage_discount : ℝ) : ℝ :=
  price * (1 - percentage_discount)

def optimal_saving (initial_price fixed_discount percentage_discount : ℝ) : ℝ :=
  let price1 := price_after_fixed_discount initial_price fixed_discount
  let final_price1 := price_after_percentage_discount price1 percentage_discount
  let price2 := price_after_percentage_discount initial_price percentage_discount
  let final_price2 := price_after_fixed_discount price2 fixed_discount
  final_price1 - final_price2

theorem optimal_saving_is_45_cents : optimal_saving initial_price fixed_discount percentage_discount = 0.45 :=
by 
  sorry

end optimal_saving_is_45_cents_l755_75592


namespace trajectory_equation_l755_75585

theorem trajectory_equation (x y : ℝ) (M O A : ℝ × ℝ)
    (hO : O = (0, 0)) (hA : A = (3, 0))
    (h_ratio : dist M O / dist M A = 1 / 2) : 
    x^2 + y^2 + 2 * x - 3 = 0 :=
by
  -- Definition of points
  let M := (x, y)
  exact sorry

end trajectory_equation_l755_75585


namespace sum_difference_of_odd_and_even_integers_l755_75521

noncomputable def sum_of_first_n_odds (n : ℕ) : ℕ :=
  n * n

noncomputable def sum_of_first_n_evens (n : ℕ) : ℕ :=
  n * (n + 1)

theorem sum_difference_of_odd_and_even_integers :
  sum_of_first_n_evens 50 - sum_of_first_n_odds 50 = 50 := 
by
  sorry

end sum_difference_of_odd_and_even_integers_l755_75521


namespace factorial_quotient_52_50_l755_75512

theorem factorial_quotient_52_50 : (Nat.factorial 52) / (Nat.factorial 50) = 2652 := 
by 
  sorry

end factorial_quotient_52_50_l755_75512


namespace complete_square_solution_l755_75597

theorem complete_square_solution :
  ∀ x : ℝ, x^2 - 4 * x - 22 = 0 → (x - 2)^2 = 26 :=
by
  intro x h
  sorry

end complete_square_solution_l755_75597


namespace number_of_aquariums_l755_75549

theorem number_of_aquariums (total_animals animals_per_aquarium : ℕ) (h1 : total_animals = 40) (h2 : animals_per_aquarium = 2) :
  total_animals / animals_per_aquarium = 20 := by
  sorry

end number_of_aquariums_l755_75549


namespace eccentricity_of_hyperbola_l755_75567

variable (a b c e : ℝ)

-- The hyperbola definition and conditions.
def hyperbola (a b : ℝ) := (a > 0) ∧ (b > 0) ∧ (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)

-- Eccentricity is greater than 1 and less than the specified upper bound
def eccentricity_range (e : ℝ) := 1 < e ∧ e < (2 * Real.sqrt 3) / 3

-- Main theorem statement: Given the hyperbola with conditions, prove eccentricity lies in the specified range.
theorem eccentricity_of_hyperbola (h : hyperbola a b) (h_line : ∀ (x y : ℝ), y = x * (Real.sqrt 3) / 3 - 0 -> y^2 ≤ (c^2 - x^2 * a^2)) :
  eccentricity_range e :=
sorry

end eccentricity_of_hyperbola_l755_75567


namespace area_of_inscribed_square_l755_75557

noncomputable def circle_eq (x y : ℝ) : Prop := 
  3*x^2 + 3*y^2 - 15*x + 9*y + 27 = 0

theorem area_of_inscribed_square :
  (∃ x y : ℝ, circle_eq x y) →
  ∃ s : ℝ, s^2 = 25 :=
by
  sorry

end area_of_inscribed_square_l755_75557


namespace yards_in_a_mile_l755_75518

def mile_eq_furlongs : Prop := 1 = 5 * 1
def furlong_eq_rods : Prop := 1 = 50 * 1
def rod_eq_yards : Prop := 1 = 5 * 1

theorem yards_in_a_mile (h1 : mile_eq_furlongs) (h2 : furlong_eq_rods) (h3 : rod_eq_yards) :
  1 * (5 * (50 * 5)) = 1250 :=
by
-- Given conditions, translate them:
-- h1 : 1 mile = 5 furlongs -> 1 * 1 = 5 * 1
-- h2 : 1 furlong = 50 rods -> 1 * 1 = 50 * 1
-- h3 : 1 rod = 5 yards -> 1 * 1 = 5 * 1
-- Prove that the number of yards in one mile is 1250
sorry

end yards_in_a_mile_l755_75518


namespace employee_n_weekly_wage_l755_75508

theorem employee_n_weekly_wage (Rm Rn : ℝ) (Hm Hn : ℝ) 
    (h1 : (Rm * Hm) + (Rn * Hn) = 770) 
    (h2 : (Rm * Hm) = 1.3 * (Rn * Hn)) :
    Rn * Hn = 335 :=
by
  sorry

end employee_n_weekly_wage_l755_75508


namespace distance_focus_directrix_l755_75559

theorem distance_focus_directrix (θ : ℝ) : 
  (∃ d : ℝ, (∀ (ρ : ℝ), ρ = 5 / (3 - 2 * Real.cos θ)) ∧ d = 5 / 2) :=
sorry

end distance_focus_directrix_l755_75559


namespace smallest_positive_period_intervals_monotonic_increase_max_min_values_l755_75599

noncomputable def f (x : ℝ) := 2 * Real.sqrt 3 * Real.sin x * Real.cos x - 2 * (Real.sin x)^2

theorem smallest_positive_period (x : ℝ) : (f (x + π)) = f x :=
sorry

theorem intervals_monotonic_increase (k : ℤ) (x : ℝ) : (k * π - π/3) ≤ x ∧ x ≤ (k * π + π/6) → ∃ a b : ℝ, a < b ∧ ∀ x : ℝ, (a ≤ x ∧ x ≤ b) →
  (f x < f (x + 1)) :=
sorry

theorem max_min_values (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ π/4) : (∃ y : ℝ, y = max (f 0) (f (π/6)) ∧ y = 1) ∧ (∃ z : ℝ, z = min (f 0) (f (π/6)) ∧ z = 0) :=
sorry

end smallest_positive_period_intervals_monotonic_increase_max_min_values_l755_75599


namespace no_largest_integer_exists_l755_75524

/--
  Define a predicate to check whether an integer is a non-square.
-/
def is_non_square (n : ℕ) : Prop :=
  ¬ ∃ m : ℕ, m * m = n

/--
  Define the main theorem which states that there is no largest positive integer
  that cannot be expressed as the sum of a positive integral multiple of 36
  and a positive non-square integer less than 36.
-/
theorem no_largest_integer_exists : ¬ ∃ (n : ℕ), 
  ∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b < 36 ∧ is_non_square b →
  n ≠ 36 * a + b :=
sorry

end no_largest_integer_exists_l755_75524


namespace num_teacher_volunteers_l755_75596

theorem num_teacher_volunteers (total_needed volunteers_from_classes extra_needed teacher_volunteers : ℕ)
  (h1 : teacher_volunteers + extra_needed + volunteers_from_classes = total_needed) 
  (h2 : total_needed = 50)
  (h3 : volunteers_from_classes = 6 * 5)
  (h4 : extra_needed = 7) :
  teacher_volunteers = 13 :=
by
  sorry

end num_teacher_volunteers_l755_75596


namespace cubic_root_of_determinant_l755_75505

open Complex 
open Matrix

noncomputable def matrix_d (a b c n : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![
    ![b + n^3 * c, n * (c - b), n^2 * (b - c)],
    ![n^2 * (c - a), c + n^3 * a, n * (a - c)],
    ![n * (b - a), n^2 * (a - b), a + n^3 * b]
  ]

theorem cubic_root_of_determinant (a b c n : ℂ) (h : a * b * c = 1) :
  (det (matrix_d a b c n))^(1/3 : ℂ) = n^3 + 1 :=
  sorry

end cubic_root_of_determinant_l755_75505
