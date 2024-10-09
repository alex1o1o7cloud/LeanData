import Mathlib

namespace exterior_angle_DEF_l1303_130383

theorem exterior_angle_DEF :
  let heptagon_angle := (180 * (7 - 2)) / 7
  let octagon_angle := (180 * (8 - 2)) / 8
  let total_degrees := 360
  total_degrees - (heptagon_angle + octagon_angle) = 96.43 :=
by
  sorry

end exterior_angle_DEF_l1303_130383


namespace vacation_months_away_l1303_130330

theorem vacation_months_away (total_savings : ℕ) (pay_per_check : ℕ) (checks_per_month : ℕ) :
  total_savings = 3000 → pay_per_check = 100 → checks_per_month = 2 → 
  total_savings / pay_per_check / checks_per_month = 15 :=
by 
  intros h1 h2 h3
  sorry

end vacation_months_away_l1303_130330


namespace river_depth_in_mid_may_l1303_130305

variable (D : ℕ)
variable (h1 : D + 10 - 5 + 8 = 45)

theorem river_depth_in_mid_may (h1 : D + 13 = 45) : D = 32 := by
  sorry

end river_depth_in_mid_may_l1303_130305


namespace Kayla_score_fifth_level_l1303_130326

theorem Kayla_score_fifth_level :
  ∃ (a b c d e f : ℕ),
  a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 8 ∧ f = 17 ∧
  (b - a = 1) ∧ (c - b = 2) ∧ (d - c = 3) ∧ (e - d = 4) ∧ (f - e = 5) ∧ e = 12 :=
sorry

end Kayla_score_fifth_level_l1303_130326


namespace div_1988_form_1989_div_1989_form_1988_l1303_130348

/-- There exists a number of the form 1989...19890... (1989 repeated several times followed by several zeros), which is divisible by 1988. -/
theorem div_1988_form_1989 (k : ℕ) : ∃ n : ℕ, (n = 1989 * 10^(4*k) ∧ n % 1988 = 0) := sorry

/-- There exists a number of the form 1988...1988 (1988 repeated several times), which is divisible by 1989. -/
theorem div_1989_form_1988 (k : ℕ) : ∃ n : ℕ, (n = 1988 * ((10^(4*k)) - 1) ∧ n % 1989 = 0) := sorry

end div_1988_form_1989_div_1989_form_1988_l1303_130348


namespace sqrt_7_minus_a_l1303_130320

theorem sqrt_7_minus_a (a : ℝ) (h : a = -1) : Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_7_minus_a_l1303_130320


namespace total_distance_l1303_130319

noncomputable def total_distance_covered 
  (radius1 radius2 radius3 : ℝ) 
  (rev1 rev2 rev3 : ℕ) : ℝ :=
  let π := Real.pi
  let circumference r := 2 * π * r
  let distance r rev := circumference r * rev
  distance radius1 rev1 + distance radius2 rev2 + distance radius3 rev3

theorem total_distance
  (h1 : radius1 = 20.4) 
  (h2 : radius2 = 15.3) 
  (h3 : radius3 = 25.6) 
  (h4 : rev1 = 400) 
  (h5 : rev2 = 320) 
  (h6 : rev3 = 500) :
  total_distance_covered 20.4 15.3 25.6 400 320 500 = 162436.6848 := 
sorry

end total_distance_l1303_130319


namespace interest_calculation_years_l1303_130375

noncomputable def principal : ℝ := 625
noncomputable def rate : ℝ := 0.04
noncomputable def difference : ℝ := 1

theorem interest_calculation_years (n : ℕ) : 
    (principal * (1 + rate)^n - principal - (principal * rate * n) = difference) → 
    n = 2 :=
by sorry

end interest_calculation_years_l1303_130375


namespace min_rubles_reaching_50_points_l1303_130332

-- Define conditions and prove the required rubles amount
def min_rubles_needed : ℕ := 11

theorem min_rubles_reaching_50_points (points : ℕ) (rubles : ℕ) : points = 50 ∧ rubles = min_rubles_needed → rubles = 11 :=
by
  intro h
  sorry

end min_rubles_reaching_50_points_l1303_130332


namespace maximize_magnitude_l1303_130360

theorem maximize_magnitude (a x y : ℝ) 
(h1 : 4 * x^2 + 4 * y^2 = -a^2 + 16 * a - 32)
(h2 : 2 * x * y = a) : a = 8 := 
sorry

end maximize_magnitude_l1303_130360


namespace number_of_children_l1303_130371

-- Definitions based on the conditions provided in the problem
def total_population : ℕ := 8243
def grown_ups : ℕ := 5256

-- Statement of the proof problem
theorem number_of_children : total_population - grown_ups = 2987 := by
-- This placeholder 'sorry' indicates that the proof is omitted.
sorry

end number_of_children_l1303_130371


namespace polynomial_evaluation_l1303_130363

theorem polynomial_evaluation (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2 * a^2 + 2 = 3 :=
by
  sorry

end polynomial_evaluation_l1303_130363


namespace rectangle_area_l1303_130354

theorem rectangle_area (s : ℕ) (P : ℕ) (A : ℕ)
  (h_perimeter : P = 160)
  (h_squares : P = 10 * s)
  (h_area : A = 4 * s^2) :
  A = 1024 :=
by
  sorry

end rectangle_area_l1303_130354


namespace car_speed_ratio_l1303_130367

-- Assuming the bridge length as L, pedestrian's speed as v_p, and car's speed as v_c.
variables (L v_p v_c : ℝ)

-- Mathematically equivalent proof problem statement in Lean 4.
theorem car_speed_ratio (h1 : 2/5 * L = 2/5 * L)
                       (h2 : (L - 2/5 * L) / v_p = L / v_c) :
    v_c = 5 * v_p := 
  sorry

end car_speed_ratio_l1303_130367


namespace monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l1303_130369

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - 1) / x

theorem monotonic_intervals_of_f :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ ≤ f x₂) ∧ (∀ x₁ x₂ : ℝ, x₁ > x₂ → f x₁ ≥ f x₂) :=
sorry

theorem f_gt_x_ln_x_plus_1 (x : ℝ) (hx : x > 0) : f x > x * Real.log (x + 1) :=
sorry

end monotonic_intervals_of_f_f_gt_x_ln_x_plus_1_l1303_130369


namespace manager_salary_proof_l1303_130309

noncomputable def manager_salary 
    (avg_salary_without_manager : ℝ) 
    (num_employees_without_manager : ℕ) 
    (increase_in_avg_salary : ℝ) 
    (new_total_salary : ℝ) : ℝ :=
    new_total_salary - (num_employees_without_manager * avg_salary_without_manager)

theorem manager_salary_proof :
    manager_salary 3500 100 800 (101 * (3500 + 800)) = 84300 :=
by
    sorry

end manager_salary_proof_l1303_130309


namespace cone_prism_volume_ratio_l1303_130347

-- Define the volumes and the ratio proof problem
theorem cone_prism_volume_ratio (r h : ℝ) (h_pos : 0 < r) (h_height : 0 < h) :
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    (V_cone / V_prism) = (π / 36) :=
by
    -- Here we define the volumes of the cone and prism as given in the problem
    let V_cone := (1 / 12) * π * r^2 * h
    let V_prism := 3 * r^2 * h
    -- We then assert the ratio condition based on the solution
    sorry

end cone_prism_volume_ratio_l1303_130347


namespace range_of_a_l1303_130382

def P (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def Q (x : ℝ) (a : ℝ) : Prop := x < a

theorem range_of_a (a : ℝ) : (∀ x, P x → Q x a) → (∀ x, Q x a → P x) → a ≥ 5 :=
by
  sorry

end range_of_a_l1303_130382


namespace round_trip_completion_percentage_l1303_130396

-- Define the distances for each section
def sectionA_distance : Float := 10
def sectionB_distance : Float := 20
def sectionC_distance : Float := 15

-- Define the speeds for each section
def sectionA_speed : Float := 50
def sectionB_speed : Float := 40
def sectionC_speed : Float := 60

-- Define the delays for each section
def sectionA_delay : Float := 1.15
def sectionB_delay : Float := 1.10

-- Calculate the time for each section without delays
def sectionA_time : Float := sectionA_distance / sectionA_speed
def sectionB_time : Float := sectionB_distance / sectionB_speed
def sectionC_time : Float := sectionC_distance / sectionC_speed

-- Calculate the time with delays for the trip to the center
def sectionA_time_with_delay : Float := sectionA_time * sectionA_delay
def sectionB_time_with_delay : Float := sectionB_time * sectionB_delay
def sectionC_time_with_delay : Float := sectionC_time

-- Total time with delays to the center
def total_time_to_center : Float := sectionA_time_with_delay + sectionB_time_with_delay + sectionC_time_with_delay

-- Total distance to the center
def total_distance_to_center : Float := sectionA_distance + sectionB_distance + sectionC_distance

-- Total round trip distance
def total_round_trip_distance : Float := total_distance_to_center * 2

-- Distance covered on the way back
def distance_back : Float := total_distance_to_center * 0.2

-- Total distance covered considering the delays and the return trip
def total_distance_covered : Float := total_distance_to_center + distance_back

-- Effective completion percentage of the round trip
def completion_percentage : Float := (total_distance_covered / total_round_trip_distance) * 100

-- The main theorem statement
theorem round_trip_completion_percentage :
  completion_percentage = 60 := by
  sorry

end round_trip_completion_percentage_l1303_130396


namespace ratio_child_to_jane_babysit_l1303_130334

-- Definitions of the conditions
def jane_current_age : ℕ := 32
def years_since_jane_stopped_babysitting : ℕ := 10
def oldest_person_current_age : ℕ := 24

-- Derived definitions
def jane_age_when_stopped : ℕ := jane_current_age - years_since_jane_stopped_babysitting
def oldest_person_age_when_jane_stopped : ℕ := oldest_person_current_age - years_since_jane_stopped_babysitting

-- Statement of the problem to be proven in Lean 4
theorem ratio_child_to_jane_babysit :
  (oldest_person_age_when_jane_stopped : ℚ) / (jane_age_when_stopped : ℚ) = 7 / 11 :=
by
  sorry

end ratio_child_to_jane_babysit_l1303_130334


namespace largest_perimeter_l1303_130317

noncomputable def interior_angle (n : ℕ) : ℝ :=
  180 * (n - 2) / n

noncomputable def condition (n1 n2 n3 n4 : ℕ) : Prop :=
  2 * interior_angle n1 + interior_angle n2 + interior_angle n3 = 360

theorem largest_perimeter
  {n1 n2 n3 n4 : ℕ}
  (h : n1 = n4)
  (h_condition : condition n1 n2 n3 n4) :
  4 * n1 + 2 * n2 + 2 * n3 - 8 ≤ 22 :=
sorry

end largest_perimeter_l1303_130317


namespace exists_right_triangle_area_twice_hypotenuse_l1303_130301

theorem exists_right_triangle_area_twice_hypotenuse : 
  ∃ (a : ℝ), a ≠ 0 ∧ (a^2 / 2 = 2 * a * Real.sqrt 2) ∧ (a = 4 * Real.sqrt 2) :=
by
  sorry

end exists_right_triangle_area_twice_hypotenuse_l1303_130301


namespace limo_gas_price_l1303_130346

theorem limo_gas_price
  (hourly_wage : ℕ := 15)
  (ride_payment : ℕ := 5)
  (review_bonus : ℕ := 20)
  (hours_worked : ℕ := 8)
  (rides_given : ℕ := 3)
  (gallons_gas : ℕ := 17)
  (good_reviews : ℕ := 2)
  (total_owed : ℕ := 226) :
  total_owed = (hours_worked * hourly_wage) + (rides_given * ride_payment) + (good_reviews * review_bonus) + (gallons_gas * 3) :=
by
  sorry

end limo_gas_price_l1303_130346


namespace textbook_weight_ratio_l1303_130361

def jon_textbooks_weights : List ℕ := [2, 8, 5, 9]
def brandon_textbooks_weight : ℕ := 8

theorem textbook_weight_ratio : 
  (jon_textbooks_weights.sum : ℚ) / (brandon_textbooks_weight : ℚ) = 3 :=
by 
  sorry

end textbook_weight_ratio_l1303_130361


namespace unique_function_satisfying_conditions_l1303_130307

theorem unique_function_satisfying_conditions :
  ∀ (f : ℝ → ℝ), 
    (∀ x : ℝ, f x ≥ 0) → 
    (∀ x : ℝ, f (x^2) = f x ^ 2 - 2 * x * f x) →
    (∀ x : ℝ, f (-x) = f (x - 1)) → 
    (∀ x y : ℝ, 1 < x → x < y → f x < f y) →
    (∀ x : ℝ, f x = x^2 + x + 1) :=
by
  -- formal proof would go here
  sorry

end unique_function_satisfying_conditions_l1303_130307


namespace min_value_of_squares_l1303_130313

theorem min_value_of_squares (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 := 
by
  -- Proof omitted
  sorry

end min_value_of_squares_l1303_130313


namespace sum_of_squares_of_ages_eq_35_l1303_130373

theorem sum_of_squares_of_ages_eq_35
  (d t h : ℕ)
  (h1 : 3 * d + 4 * t = 2 * h + 2)
  (h2 : 2 * d^2 + t^2 = 6 * h)
  (relatively_prime : Nat.gcd (Nat.gcd d t) h = 1) :
  d^2 + t^2 + h^2 = 35 := 
sorry

end sum_of_squares_of_ages_eq_35_l1303_130373


namespace paolo_sevilla_birthday_l1303_130358

theorem paolo_sevilla_birthday (n : ℕ) :
  (12 * (n + 2) = 16 * n) -> n = 6 :=
by
  intro h
    
  -- expansion and solving should go here
  -- sorry, since only statement required
  sorry

end paolo_sevilla_birthday_l1303_130358


namespace geometric_sequence_n_value_l1303_130335

theorem geometric_sequence_n_value
  (a : ℕ → ℝ) (n : ℕ)
  (h1 : a 1 * a 2 * a 3 = 4)
  (h2 : a 4 * a 5 * a 6 = 12)
  (h3 : a (n-1) * a n * a (n+1) = 324)
  (h_geometric : ∃ r > 0, ∀ i, a (i+1) = a i * r) :
  n = 14 :=
sorry

end geometric_sequence_n_value_l1303_130335


namespace passengers_final_count_l1303_130392

structure BusStop :=
  (initial_passengers : ℕ)
  (first_stop_increase : ℕ)
  (other_stops_decrease : ℕ)
  (other_stops_increase : ℕ)

def passengers_at_last_stop (b : BusStop) : ℕ :=
  b.initial_passengers + b.first_stop_increase - b.other_stops_decrease + b.other_stops_increase

theorem passengers_final_count :
  passengers_at_last_stop ⟨50, 16, 22, 5⟩ = 49 := by
  rfl

end passengers_final_count_l1303_130392


namespace time_until_heavy_lifting_l1303_130385

-- Define the conditions given
def pain_subside_days : ℕ := 3
def healing_multiplier : ℕ := 5
def additional_wait_days : ℕ := 3
def weeks_before_lifting : ℕ := 3
def days_in_week : ℕ := 7

-- Define the proof statement
theorem time_until_heavy_lifting : 
    let full_healing_days := pain_subside_days * healing_multiplier
    let total_days_before_exercising := full_healing_days + additional_wait_days
    let lifting_wait_days := weeks_before_lifting * days_in_week
    total_days_before_exercising + lifting_wait_days = 39 := 
by
  sorry

end time_until_heavy_lifting_l1303_130385


namespace range_of_a_l1303_130381

def p (a : ℝ) : Prop := a > -1
def q (a : ℝ) : Prop := ∀ m : ℝ, -2 ≤ m ∧ m ≤ 4 → a^2 - a ≥ 4 - m

theorem range_of_a (a : ℝ) : (p a ∧ ¬q a) ∨ (¬p a ∧ q a) ↔ (-1 < a ∧ a < 3) ∨ a ≤ -2 := by
  sorry

end range_of_a_l1303_130381


namespace max_product_l1303_130391

theorem max_product (x y : ℕ) (h1 : 7 * x + 4 * y = 140) : x * y ≤ 168 :=
sorry

end max_product_l1303_130391


namespace profit_percentage_correct_l1303_130321

-- Statement of the problem in Lean
theorem profit_percentage_correct (SP CP : ℝ) (hSP : SP = 400) (hCP : CP = 320) : 
  ((SP - CP) / CP) * 100 = 25 := by
  -- Proof goes here
  sorry

end profit_percentage_correct_l1303_130321


namespace min_value_a_l1303_130331

noncomputable def equation_has_real_solutions (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 9 * x1 - (4 + a) * 3 * x1 + 4 = 0 ∧ 9 * x2 - (4 + a) * 3 * x2 + 4 = 0

theorem min_value_a : ∀ a : ℝ, 
  equation_has_real_solutions a → 
  a ≥ 2 :=
sorry

end min_value_a_l1303_130331


namespace factorize_16x2_minus_1_l1303_130366

theorem factorize_16x2_minus_1 (x : ℝ) : 16 * x^2 - 1 = (4 * x + 1) * (4 * x - 1) := by
  sorry

end factorize_16x2_minus_1_l1303_130366


namespace sin_double_angle_subst_l1303_130327

open Real

theorem sin_double_angle_subst 
  (α : ℝ)
  (h : sin (α + π / 6) = -1 / 3) :
  sin (2 * α - π / 6) = -7 / 9 := 
by
  sorry

end sin_double_angle_subst_l1303_130327


namespace probability_three_black_balls_probability_white_ball_l1303_130394

-- Definitions representing conditions
def total_ratio (A B C : ℕ) := A / B = 5 / 4 ∧ B / C = 4 / 6

-- Proportions of black balls in each box
def proportion_black_A (black_A total_A : ℕ) := black_A = 40 * total_A / 100
def proportion_black_B (black_B total_B : ℕ) := black_B = 25 * total_B / 100
def proportion_black_C (black_C total_C : ℕ) := black_C = 50 * total_C / 100

-- Problem 1: Probability of selecting a black ball from each box
theorem probability_three_black_balls
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C) :
  (black_A / total_A) * (black_B / total_B) * (black_C / total_C) = 1 / 20 :=
  sorry

-- Problem 2: Probability of selecting a white ball from the mixed total
theorem probability_white_ball
  (A B C : ℕ)
  (total_A total_B total_C : ℕ)
  (black_A black_B black_C : ℕ)
  (white_A white_B white_C : ℕ)
  (h1 : total_ratio A B C)
  (h2 : proportion_black_A black_A total_A)
  (h3 : proportion_black_B black_B total_B)
  (h4 : proportion_black_C black_C total_C)
  (h5 : white_A = total_A - black_A)
  (h6 : white_B = total_B - black_B)
  (h7 : white_C = total_C - black_C) :
  (white_A + white_B + white_C) / (total_A + total_B + total_C) = 3 / 5 :=
  sorry

end probability_three_black_balls_probability_white_ball_l1303_130394


namespace rice_on_8th_day_l1303_130337

variable (a1 : ℕ) (d : ℕ) (n : ℕ)
variable (rice_per_laborer : ℕ)

def is_arithmetic_sequence (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

theorem rice_on_8th_day (ha1 : a1 = 64) (hd : d = 7) (hr : rice_per_laborer = 3) :
  let a8 := is_arithmetic_sequence a1 d 8
  (a8 * rice_per_laborer = 339) :=
by
  sorry

end rice_on_8th_day_l1303_130337


namespace probability_no_correct_letter_for_7_envelopes_l1303_130356

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1
  else n * factorial (n - 1)

def derangement (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangement (n - 1) + derangement (n - 2))

noncomputable def probability_no_correct_letter (n : ℕ) : ℚ :=
  derangement n / factorial n

theorem probability_no_correct_letter_for_7_envelopes :
  probability_no_correct_letter 7 = 427 / 1160 :=
by sorry

end probability_no_correct_letter_for_7_envelopes_l1303_130356


namespace sqrt_of_square_neg7_l1303_130310

theorem sqrt_of_square_neg7 : Real.sqrt ((-7:ℝ)^2) = 7 := by
  sorry

end sqrt_of_square_neg7_l1303_130310


namespace sunil_total_amount_back_l1303_130342

theorem sunil_total_amount_back 
  (CI : ℝ) (P : ℝ) (r : ℝ) (t : ℕ) (total_amount : ℝ) 
  (h1 : CI = 2828.80) 
  (h2 : r = 8) 
  (h3 : t = 2) 
  (h4 : CI = P * ((1 + r / 100) ^ t - 1)) : 
  total_amount = P + CI → 
  total_amount = 19828.80 :=
by
  sorry

end sunil_total_amount_back_l1303_130342


namespace multiples_of_3_or_5_but_not_6_l1303_130315

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 150) :
  (∃ m : ℕ, m ≤ 150 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ m % 6 ≠ 0)) ↔ n = 45 :=
by {
  sorry
}

end multiples_of_3_or_5_but_not_6_l1303_130315


namespace prob_ending_game_after_five_distribution_and_expectation_l1303_130355

-- Define the conditions
def shooting_accuracy_rate : ℚ := 2 / 3
def game_clear_coupon : ℕ := 9
def game_fail_coupon : ℕ := 3
def game_no_clear_no_fail_coupon : ℕ := 6

-- Define the probabilities for ending the game after 5 shots
def ending_game_after_five : ℚ := (shooting_accuracy_rate^2 * (1 - shooting_accuracy_rate)^3 * 2) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate))

-- Define the distribution table
def P_clear : ℚ := (shooting_accuracy_rate^3) + (shooting_accuracy_rate^3 * (1 - shooting_accuracy_rate)) + (shooting_accuracy_rate^4 * (1 - shooting_accuracy_rate) * 2)
def P_fail : ℚ := ((1 - shooting_accuracy_rate)^2) + ((1 - shooting_accuracy_rate)^2 * shooting_accuracy_rate * 2) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^2 * 3) + ((1 - shooting_accuracy_rate)^3 * shooting_accuracy_rate^3)
def P_neither : ℚ := 1 - P_clear - P_fail

-- Expected value calculation
def expectation : ℚ := (P_fail * game_fail_coupon) + (P_neither * game_no_clear_no_fail_coupon) + (P_clear * game_clear_coupon)

-- The Part I proof statement
theorem prob_ending_game_after_five : ending_game_after_five = 8 / 81 :=
by
  sorry

-- The Part II proof statement
theorem distribution_and_expectation (X : ℕ → ℚ) :
  (X game_fail_coupon = 233 / 729) ∧
  (X game_no_clear_no_fail_coupon = 112 / 729) ∧
  (X game_clear_coupon = 128 / 243) ∧
  (expectation = 1609 / 243) :=
by
  sorry

end prob_ending_game_after_five_distribution_and_expectation_l1303_130355


namespace greatest_possible_integer_l1303_130364

theorem greatest_possible_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℕ, n = 9 * k - 1) (h3 : ∃ l : ℕ, n = 10 * l - 4) : n = 86 := 
sorry

end greatest_possible_integer_l1303_130364


namespace manager_final_price_l1303_130374

noncomputable def wholesale_cost : ℝ := 200
noncomputable def retail_price : ℝ := wholesale_cost + 0.2 * wholesale_cost
noncomputable def manager_discount : ℝ := 0.1 * retail_price
noncomputable def price_after_manager_discount : ℝ := retail_price - manager_discount
noncomputable def weekend_sale_discount : ℝ := 0.1 * price_after_manager_discount
noncomputable def price_after_weekend_sale : ℝ := price_after_manager_discount - weekend_sale_discount
noncomputable def sales_tax : ℝ := 0.08 * price_after_weekend_sale
noncomputable def total_price : ℝ := price_after_weekend_sale + sales_tax

theorem manager_final_price : total_price = 209.95 := by
  sorry

end manager_final_price_l1303_130374


namespace number_of_valid_triples_l1303_130384

theorem number_of_valid_triples :
  ∃ (count : ℕ), count = 3 ∧
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z →
  Nat.lcm x y = 120 → Nat.lcm y z = 1000 → Nat.lcm x z = 480 →
  (∃ (u v w : ℕ), u = x ∧ v = y ∧ w = z ∧ count = 3) :=
by
  sorry

end number_of_valid_triples_l1303_130384


namespace water_consumption_l1303_130351

theorem water_consumption (num_cows num_goats num_pigs num_sheep : ℕ)
  (water_per_cow water_per_goat water_per_pig water_per_sheep daily_total weekly_total : ℕ)
  (h1 : num_cows = 40)
  (h2 : num_goats = 25)
  (h3 : num_pigs = 30)
  (h4 : water_per_cow = 80)
  (h5 : water_per_goat = water_per_cow / 2)
  (h6 : water_per_pig = water_per_cow / 3)
  (h7 : num_sheep = 10 * num_cows)
  (h8 : water_per_sheep = water_per_cow / 4)
  (h9 : daily_total = num_cows * water_per_cow + num_goats * water_per_goat + num_pigs * water_per_pig + num_sheep * water_per_sheep)
  (h10 : weekly_total = daily_total * 7) :
  weekly_total = 91000 := by
  sorry

end water_consumption_l1303_130351


namespace no_such_b_exists_l1303_130386

theorem no_such_b_exists (b : ℝ) (hb : 0 < b) :
  ¬(∃ k : ℝ, 0 < k ∧ ∀ n : ℕ, 0 < n → (n - k ≤ (⌊b * n⌋ : ℤ) ∧ (⌊b * n⌋ : ℤ) < n)) :=
by
  sorry

end no_such_b_exists_l1303_130386


namespace triangle_right_angle_l1303_130398

theorem triangle_right_angle {A B C : ℝ} 
  (h1 : A + B + C = 180)
  (h2 : A = B)
  (h3 : A = (1/2) * C) :
  C = 90 :=
by 
  sorry

end triangle_right_angle_l1303_130398


namespace speed_ratio_l1303_130370

-- Definition of speeds
def B_speed : ℚ := 1 / 12
def combined_speed : ℚ := 1 / 4

-- The theorem statement to be proven
theorem speed_ratio (A_speed B_speed combined_speed : ℚ) (h1 : B_speed = 1 / 12) (h2 : combined_speed = 1 / 4) (h3 : A_speed + B_speed = combined_speed) :
  A_speed / B_speed = 2 :=
by
  sorry

end speed_ratio_l1303_130370


namespace Ferris_break_length_l1303_130308

noncomputable def Audrey_rate_per_hour := (1:ℝ) / 4
noncomputable def Ferris_rate_per_hour := (1:ℝ) / 3
noncomputable def total_completion_time := (2:ℝ)
noncomputable def number_of_breaks := (6:ℝ)
noncomputable def job_completion_audrey := total_completion_time * Audrey_rate_per_hour
noncomputable def job_completion_ferris := 1 - job_completion_audrey
noncomputable def working_time_ferris := job_completion_ferris / Ferris_rate_per_hour
noncomputable def total_break_time := total_completion_time - working_time_ferris
noncomputable def break_length := total_break_time / number_of_breaks

theorem Ferris_break_length :
  break_length = (5:ℝ) / 60 := 
sorry

end Ferris_break_length_l1303_130308


namespace average_weight_of_class_l1303_130329

variable (SectionA_students : ℕ := 26)
variable (SectionB_students : ℕ := 34)
variable (SectionA_avg_weight : ℝ := 50)
variable (SectionB_avg_weight : ℝ := 30)

theorem average_weight_of_class :
  (SectionA_students * SectionA_avg_weight + SectionB_students * SectionB_avg_weight) / (SectionA_students + SectionB_students) = 38.67 := by
  sorry

end average_weight_of_class_l1303_130329


namespace greatest_possible_integer_l1303_130399

theorem greatest_possible_integer (n k l : ℕ) (h1 : n < 150) (h2 : n = 11 * k - 1) (h3 : n = 9 * l + 2) : n = 65 :=
by sorry

end greatest_possible_integer_l1303_130399


namespace christine_makes_two_cakes_l1303_130387

theorem christine_makes_two_cakes (tbsp_per_egg_white : ℕ) 
  (egg_whites_per_cake : ℕ) 
  (total_tbsp_aquafaba : ℕ)
  (h1 : tbsp_per_egg_white = 2) 
  (h2 : egg_whites_per_cake = 8) 
  (h3 : total_tbsp_aquafaba = 32) : 
  total_tbsp_aquafaba / tbsp_per_egg_white / egg_whites_per_cake = 2 := by 
  sorry

end christine_makes_two_cakes_l1303_130387


namespace smaller_mold_radius_l1303_130325

theorem smaller_mold_radius (R : ℝ) (third_volume_sharing : ℝ) (molds_count : ℝ) (r : ℝ) 
  (hR : R = 3) 
  (h_third_volume_sharing : third_volume_sharing = 1/3) 
  (h_molds_count : molds_count = 9) 
  (h_r : (2/3) * Real.pi * r^3 = (2/3) * Real.pi / molds_count) : 
  r = 1 := 
by
  sorry

end smaller_mold_radius_l1303_130325


namespace percentage_increase_in_surface_area_l1303_130343

variable (a : ℝ)

theorem percentage_increase_in_surface_area (ha : a > 0) :
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  percentage_increase = 125 := 
by 
  let original_surface_area := 6 * a^2
  let new_edge_length := 1.5 * a
  let new_surface_area := 6 * (new_edge_length)^2
  let area_increase := new_surface_area - original_surface_area
  let percentage_increase := (area_increase / original_surface_area) * 100
  sorry

end percentage_increase_in_surface_area_l1303_130343


namespace smallest_positive_period_l1303_130345

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 6)

theorem smallest_positive_period (ω : ℝ) (hω : ω > 0)
  (H : ∀ x1 x2 : ℝ, abs (f ω x1 - f ω x2) = 2 → abs (x1 - x2) = Real.pi / 2) :
  ∃ T > 0, T = Real.pi ∧ (∀ x : ℝ, f ω (x + T) = f ω x) := 
sorry

end smallest_positive_period_l1303_130345


namespace length_BE_l1303_130379

-- Define points and distances
variables (A B C D E : Type)
variable {AB : ℝ}
variable {BC : ℝ}
variable {CD : ℝ}
variable {DA : ℝ}

-- Given conditions
axiom AB_length : AB = 5
axiom BC_length : BC = 7
axiom CD_length : CD = 8
axiom DA_length : DA = 6

-- Bugs travelling in opposite directions from point A meet at E
axiom bugs_meet_at_E : True

-- Proving the length BE
theorem length_BE : BE = 6 :=
by
  -- Currently, this is a statement. The proof is not included.
  sorry

end length_BE_l1303_130379


namespace quadratic_has_two_real_roots_find_m_for_roots_difference_l1303_130312

theorem quadratic_has_two_real_roots (m : ℝ) : 
  ∃ (x1 x2 : ℝ), x1^2 + (2 - m) * x1 + (1 - m) = 0 ∧
                 x2^2 + (2 - m) * x2 + (1 - m) = 0 :=
by sorry

theorem find_m_for_roots_difference (m x1 x2 : ℝ) (h1 : x1^2 + (2 - m) * x1 + (1 - m) = 0) 
  (h2 : x2^2 + (2 - m) * x2 + (1 - m) = 0) (hm : m < 0) (hd : x1 - x2 = 3) : 
  m = -3 :=
by sorry

end quadratic_has_two_real_roots_find_m_for_roots_difference_l1303_130312


namespace sum_of_zeros_l1303_130323

-- Defining the conditions and the result
theorem sum_of_zeros (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) (a b c : ℝ)
  (h1 : f a = 0) (h2 : f b = 0) (h3 : f c = 0) : 
  a + b + c = 3 := 
by 
  sorry

end sum_of_zeros_l1303_130323


namespace remainder_when_divided_l1303_130300

/-- Given integers T, E, N, S, E', N', S'. When T is divided by E, 
the quotient is N and the remainder is S. When N is divided by E', 
the quotient is N' and the remainder is S'. Prove that the remainder 
when T is divided by E + E' is ES' + S. -/
theorem remainder_when_divided (T E N S E' N' S' : ℤ) (h1 : T = N * E + S) (h2 : N = N' * E' + S') :
  (T % (E + E')) = (E * S' + S) :=
by
  sorry

end remainder_when_divided_l1303_130300


namespace find_range_of_a_l1303_130365

def setA (x : ℝ) : Prop := 1 < x ∧ x < 2
def setB (x : ℝ) : Prop := 3 / 2 < x ∧ x < 4
def setUnion (x : ℝ) : Prop := 1 < x ∧ x < 4
def setP (a x : ℝ) : Prop := a < x ∧ x < a + 2

theorem find_range_of_a (a : ℝ) :
  (∀ x, setP a x → setUnion x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end find_range_of_a_l1303_130365


namespace number_of_students_l1303_130372

-- Definitions based on problem conditions
def age_condition (a n : ℕ) : Prop :=
  7 * (a - 1) + 2 * (a + 2) + (n - 9) * a = 330

-- Main theorem to prove the correct number of students
theorem number_of_students (a n : ℕ) (h : age_condition a n) : n = 37 :=
  sorry

end number_of_students_l1303_130372


namespace base7_perfect_square_xy5z_l1303_130304

theorem base7_perfect_square_xy5z (n : ℕ) (x y z : ℕ) (hx : x ≠ 0) (hn : n = 343 * x + 49 * y + 35 + z) (hsq : ∃ m : ℕ, n = m * m) : z = 1 ∨ z = 6 :=
sorry

end base7_perfect_square_xy5z_l1303_130304


namespace number_of_zeros_g_l1303_130336

variable (f : ℝ → ℝ)
variable (hf_cont : continuous f)
variable (hf_diff : differentiable ℝ f)
variable (h_condition : ∀ x : ℝ, x * (deriv f x) + f x > 0)

theorem number_of_zeros_g (hg : ∀ x : ℝ, x > 0 → x * f x + 1 = 0 → false) : 
    ∀ x : ℝ , x > 0 → ¬ (x * f x + 1 = 0) :=
by
  sorry

end number_of_zeros_g_l1303_130336


namespace girls_more_than_boys_l1303_130377

theorem girls_more_than_boys (boys girls : ℕ) (ratio_boys ratio_girls : ℕ) 
  (h1 : ratio_boys = 5)
  (h2 : ratio_girls = 13)
  (h3 : boys = 50)
  (h4 : girls = (boys / ratio_boys) * ratio_girls) : 
  girls - boys = 80 :=
by
  sorry

end girls_more_than_boys_l1303_130377


namespace range_of_k_for_real_roots_l1303_130314

theorem range_of_k_for_real_roots (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 = x2 ∧ x^2 - 2*x + k = 0) ↔ k ≤ 1 := 
by
  sorry

end range_of_k_for_real_roots_l1303_130314


namespace salary_percentage_change_l1303_130389

theorem salary_percentage_change (S : ℝ) (x : ℝ) :
  (S * (1 - (x / 100)) * (1 + (x / 100)) = S * 0.84) ↔ (x = 40) :=
by
  sorry

end salary_percentage_change_l1303_130389


namespace range_of_expression_l1303_130316

theorem range_of_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
  0 < (x * y + y * z + z * x - 2 * x * y * z) ∧ (x * y + y * z + z * x - 2 * x * y * z) ≤ 7 / 27 := by
  sorry

end range_of_expression_l1303_130316


namespace function_even_and_monotonically_increasing_l1303_130322

-- Definition: Even Function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Definition: Monotonically Increasing on (0, ∞)
def is_monotonically_increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y

-- Given Function
def f (x : ℝ) : ℝ := |x| + 1

-- Theorem to prove
theorem function_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on_pos f := by
  sorry

end function_even_and_monotonically_increasing_l1303_130322


namespace find_n_l1303_130397

theorem find_n (n : ℕ) 
    (h : 6 * 4 * 3 * n = Nat.factorial 8) : n = 560 := 
sorry

end find_n_l1303_130397


namespace point_B_third_quadrant_l1303_130318

theorem point_B_third_quadrant (m n : ℝ) (hm : m < 0) (hn : n < 0) :
  (-m * n < 0) ∧ (m < 0) :=
by
  sorry

end point_B_third_quadrant_l1303_130318


namespace group_capacity_l1303_130368

theorem group_capacity (total_students : ℕ) (selected_students : ℕ) (removed_students : ℕ) :
  total_students = 5008 → selected_students = 200 → removed_students = 8 →
  (total_students - removed_students) / selected_students = 25 :=
by
  intros h1 h2 h3
  sorry

end group_capacity_l1303_130368


namespace fraction_computation_l1303_130357

theorem fraction_computation : (1 / 2) * (1 / 3) * (1 / 6) * 72 = 2 :=
by
  sorry

end fraction_computation_l1303_130357


namespace staircase_tile_cover_possible_l1303_130378
-- Import the necessary Lean Lean libraries

-- We use natural numbers here
open Nat

-- Declare the problem as a theorem in Lean
theorem staircase_tile_cover_possible (m n : ℕ) (h_m : 6 ≤ m) (h_n : 6 ≤ n) :
  (∃ a b, m = 12 * a ∧ n = b ∧ a ≥ 1 ∧ b ≥ 6) ∨ 
  (∃ c d, m = 3 * c ∧ n = 4 * d ∧ c ≥ 2 ∧ d ≥ 3) :=
sorry

end staircase_tile_cover_possible_l1303_130378


namespace find_k_l1303_130306

theorem find_k (α β k : ℝ) (h₁ : α^2 - α + k - 1 = 0) (h₂ : β^2 - β + k - 1 = 0) (h₃ : α^2 - 2*α - β = 4) :
  k = -4 :=
sorry

end find_k_l1303_130306


namespace equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l1303_130344

-- Definition of a peculiar triangle.
def is_peculiar_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2

-- Problem 1: Proving an equilateral triangle is a peculiar triangle
theorem equilateral_is_peculiar (a : ℝ) : is_peculiar_triangle a a a :=
sorry

-- Problem 2: Proving the case when b is the hypotenuse in Rt△ABC makes it peculiar
theorem rt_triangle_is_peculiar (a b c : ℝ) (ha : a = 5 * Real.sqrt 2) (hc : c = 10) : 
  is_peculiar_triangle a b c ↔ b = Real.sqrt (c^2 + a^2) :=
sorry

-- Problem 3: Proving the ratio of the sides in a peculiar right triangle is 1 : √2 : √3
theorem peculiar_rt_triangle_ratio (a b c : ℝ) (hc : c^2 = a^2 + b^2) (hpeculiar : is_peculiar_triangle a c b) :
  (b = Real.sqrt 2 * a) ∧ (c = Real.sqrt 3 * a) :=
sorry

end equilateral_is_peculiar_rt_triangle_is_peculiar_peculiar_rt_triangle_ratio_l1303_130344


namespace cubic_sum_identity_l1303_130393

   theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 3) (h3 : abc = -1) :
     a^3 + b^3 + c^3 = 12 :=
   by
     sorry
   
end cubic_sum_identity_l1303_130393


namespace intersection_eq_l1303_130349

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {x | |x| < 1}

theorem intersection_eq : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_eq_l1303_130349


namespace complement_union_of_sets_l1303_130339

variable {U M N : Set ℕ}

theorem complement_union_of_sets (h₁ : M ⊆ N) (h₂ : N ⊆ U) :
  (U \ M) ∪ (U \ N) = U \ M :=
by
  sorry

end complement_union_of_sets_l1303_130339


namespace probability_sum_odd_l1303_130380

theorem probability_sum_odd (x y : ℕ) 
  (hx : x > 0) (hy : y > 0) 
  (h_even : ∃ z : ℕ, z % 2 = 0 ∧ z > 0) 
  (h_odd : ∃ z : ℕ, z % 2 = 1 ∧ z > 0) : 
  (∃ p : ℝ, 0 < p ∧ p < 1 ∧ p = 0.5) :=
sorry

end probability_sum_odd_l1303_130380


namespace positive_integer_solutions_l1303_130338

theorem positive_integer_solutions :
  ∀ m n : ℕ, 0 < m ∧ 0 < n ∧ 3^m - 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 3) :=
by
  sorry

end positive_integer_solutions_l1303_130338


namespace operation_1_and_2004_l1303_130350

def operation (m n : ℕ) : ℕ :=
  if m = 1 ∧ n = 1 then 2
  else if m = 1 ∧ n > 1 then 2 + 3 * (n - 1)
  else 0 -- handle other cases generically, although specifics are not given

theorem operation_1_and_2004 : operation 1 2004 = 6011 :=
by
  unfold operation
  sorry

end operation_1_and_2004_l1303_130350


namespace jellybean_count_l1303_130359

def black_beans : Nat := 8
def green_beans : Nat := black_beans + 2
def orange_beans : Nat := green_beans - 1
def total_jelly_beans : Nat := black_beans + green_beans + orange_beans

theorem jellybean_count : total_jelly_beans = 27 :=
by
  -- proof steps would go here.
  sorry

end jellybean_count_l1303_130359


namespace white_squares_95th_figure_l1303_130390

theorem white_squares_95th_figure : ∀ (T : ℕ → ℕ),
  T 1 = 8 →
  (∀ n ≥ 1, T (n + 1) = T n + 5) →
  T 95 = 478 :=
by
  intros T hT1 hTrec
  -- Skipping the proof
  sorry

end white_squares_95th_figure_l1303_130390


namespace magnitude_of_2a_minus_b_l1303_130311

/-- Definition of the vectors a and b --/
def a : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)

/-- Proposition stating the magnitude of 2a - b --/
theorem magnitude_of_2a_minus_b : 
  (Real.sqrt ((2 * a.1 - b.1) ^ 2 + (2 * a.2 - b.2) ^ 2)) = Real.sqrt 10 :=
by
  sorry

end magnitude_of_2a_minus_b_l1303_130311


namespace number_of_valid_six_tuples_l1303_130324

def is_valid_six_tuple (p : ℕ) (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 3 * p ∧
  (a + b) % (c + d) = 0 ∧
  (b + c) % (d + e) = 0 ∧
  (c + d) % (e + f) = 0 ∧
  (d + e) % (f + a) = 0 ∧
  (e + f) % (a + b) = 0

theorem number_of_valid_six_tuples (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : 
  ∃! n, n = p + 2 ∧ ∀ (a b c d e f : ℕ), is_valid_six_tuple p a b c d e f → n = p + 2 :=
sorry

end number_of_valid_six_tuples_l1303_130324


namespace reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l1303_130340

theorem reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs
  (a b c h : Real)
  (area_legs : ℝ := (1 / 2) * a * b)
  (area_hypotenuse : ℝ := (1 / 2) * c * h)
  (eq_areas : a * b = c * h)
  (height_eq : h = a * b / c)
  (pythagorean_theorem : c ^ 2 = a ^ 2 + b ^ 2) :
  1 / h ^ 2 = 1 / a ^ 2 + 1 / b ^ 2 := 
by
  sorry

end reciprocal_of_square_of_altitude_eq_sum_of_reciprocals_of_squares_of_legs_l1303_130340


namespace line_through_PQ_l1303_130302

theorem line_through_PQ (x y : ℝ) (P Q : ℝ × ℝ)
  (hP : P = (3, 2)) (hQ : Q = (1, 4))
  (h_line : ∀ t, (x, y) = (1 - t) • P + t • Q):
  y = x - 2 :=
by
  have h1 : P = ((3 : ℝ), (2 : ℝ)) := hP
  have h2 : Q = ((1 : ℝ), (4 : ℝ)) := hQ
  sorry

end line_through_PQ_l1303_130302


namespace remainder_product_mod_eq_l1303_130303

theorem remainder_product_mod_eq (n : ℤ) :
  ((12 - 2 * n) * (n + 5)) % 11 = (-2 * n^2 + 2 * n + 5) % 11 := by
  sorry

end remainder_product_mod_eq_l1303_130303


namespace dollar_op_5_neg2_l1303_130388

def dollar_op (x y : Int) : Int := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_5_neg2 :
  dollar_op 5 (-2) = -45 := by
  sorry

end dollar_op_5_neg2_l1303_130388


namespace intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l1303_130362

-- Definitions for the sets A and B
def setA (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (3 * a + 1)) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a ^ 2 + 1)) < 0 }

-- Theorem for question (1): Intersection of A and B when a = 2
theorem intersection_of_A_and_B_when_a_is_2 :
  setA 2 ∩ setB 2 = { x | 4 < x ∧ x < 5 } :=
sorry

-- Theorem for question (2): Range of a such that B ⊆ A
theorem range_of_a_such_that_B_subset_A :
  { a : ℝ | setB a ⊆ setA a } = { x | 1 < x ∧ x ≤ 3 } ∪ { -1 } :=
sorry

end intersection_of_A_and_B_when_a_is_2_range_of_a_such_that_B_subset_A_l1303_130362


namespace tan_six_theta_eq_l1303_130353

theorem tan_six_theta_eq (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (6 * θ) = 21 / 8 :=
by
  sorry

end tan_six_theta_eq_l1303_130353


namespace ax5_plus_by5_l1303_130333

-- Declare real numbers a, b, x, y
variables (a b x y : ℝ)

theorem ax5_plus_by5 (h1 : a * x + b * y = 3)
                     (h2 : a * x^2 + b * y^2 = 7)
                     (h3 : a * x^3 + b * y^3 = 6)
                     (h4 : a * x^4 + b * y^4 = 42) :
                     a * x^5 + b * y^5 = 20 := 
sorry

end ax5_plus_by5_l1303_130333


namespace curve_in_second_quadrant_range_l1303_130395

theorem curve_in_second_quadrant_range (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0)) → a > 2 :=
by
  sorry

end curve_in_second_quadrant_range_l1303_130395


namespace perimeter_of_new_rectangle_l1303_130341

-- Definitions based on conditions
def side_of_square : ℕ := 8
def length_of_rectangle : ℕ := 8
def breadth_of_rectangle : ℕ := 4

-- Perimeter calculation
def perimeter (length breadth : ℕ) : ℕ := 2 * (length + breadth)

-- Formal statement of the problem
theorem perimeter_of_new_rectangle :
  perimeter (side_of_square + length_of_rectangle) side_of_square = 48 :=
  by sorry

end perimeter_of_new_rectangle_l1303_130341


namespace mean_of_three_l1303_130352

theorem mean_of_three (a b c : ℝ) (h : (a + b + c + 105) / 4 = 92) : (a + b + c) / 3 = 87.7 :=
by
  sorry

end mean_of_three_l1303_130352


namespace smallest_of_five_consecutive_l1303_130328

theorem smallest_of_five_consecutive (n : ℤ) (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 2015) : n - 2 = 401 :=
by sorry

end smallest_of_five_consecutive_l1303_130328


namespace sqrt_200_eq_10_sqrt_2_l1303_130376

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l1303_130376
