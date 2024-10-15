import Mathlib

namespace NUMINAMATH_GPT_total_money_l2088_208888

theorem total_money (p q r : ℕ)
  (h1 : r = 2000)
  (h2 : r = (2 / 3) * (p + q)) : 
  p + q + r = 5000 :=
by
  sorry

end NUMINAMATH_GPT_total_money_l2088_208888


namespace NUMINAMATH_GPT_point_p_locus_equation_l2088_208813

noncomputable def locus_point_p (x y : ℝ) : Prop :=
  ∀ (k b x1 y1 x2 y2 : ℝ), 
  (x1^2 + y1^2 = 1) ∧ 
  (x2^2 + y2^2 = 1) ∧ 
  (3 * x1 * x + 4 * y1 * y = 12) ∧ 
  (3 * x2 * x + 4 * y2 * y = 12) ∧ 
  (1 + k^2 = b^2) ∧ 
  (y = 3 / b) ∧ 
  (x = -4 * k / (3 * b)) → 
  x^2 / 16 + y^2 / 9 = 1

theorem point_p_locus_equation :
  ∀ (x y : ℝ), locus_point_p x y → (x^2 / 16 + y^2 / 9 = 1) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_point_p_locus_equation_l2088_208813


namespace NUMINAMATH_GPT_only_zero_function_satisfies_conditions_l2088_208855

def is_increasing (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n > m → f n ≥ f m

def satisfies_functional_equation (f : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, f (n * m) = f n + f m

theorem only_zero_function_satisfies_conditions :
  ∀ f : ℕ → ℕ, 
  (is_increasing f) ∧ (satisfies_functional_equation f) → (∀ n : ℕ, f n = 0) :=
by
  sorry

end NUMINAMATH_GPT_only_zero_function_satisfies_conditions_l2088_208855


namespace NUMINAMATH_GPT_possible_values_of_K_l2088_208843

theorem possible_values_of_K (K N : ℕ) (h1 : K * (K + 1) = 2 * N^2) (h2 : N < 100) :
  K = 1 ∨ K = 8 ∨ K = 49 :=
sorry

end NUMINAMATH_GPT_possible_values_of_K_l2088_208843


namespace NUMINAMATH_GPT_diamonds_in_G_15_l2088_208828

/-- Define the number of diamonds in G_n -/
def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1
  else 3 * n ^ 2 - 3 * n + 1

/-- Theorem to prove the number of diamonds in G_15 is 631 -/
theorem diamonds_in_G_15 : diamonds_in_G 15 = 631 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_diamonds_in_G_15_l2088_208828


namespace NUMINAMATH_GPT_second_pipe_fill_time_l2088_208857

theorem second_pipe_fill_time (x : ℝ) :
  (1 / 18) + (1 / x) - (1 / 45) = (1 / 15) → x = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_pipe_fill_time_l2088_208857


namespace NUMINAMATH_GPT_synthetic_analytic_incorrect_statement_l2088_208880

theorem synthetic_analytic_incorrect_statement
  (basic_methods : ∀ (P Q : Prop), (P → Q) ∨ (Q → P))
  (synthetic_forward : ∀ (P Q : Prop), (P → Q))
  (analytic_backward : ∀ (P Q : Prop), (Q → P)) :
  ¬ (∀ (P Q : Prop), (P → Q) ∧ (Q → P)) :=
by
  sorry

end NUMINAMATH_GPT_synthetic_analytic_incorrect_statement_l2088_208880


namespace NUMINAMATH_GPT_steve_speed_back_l2088_208864

open Real

noncomputable def steves_speed_on_way_back : ℝ := 15

theorem steve_speed_back
  (distance_to_work : ℝ)
  (traffic_time_to_work : ℝ)
  (traffic_time_back : ℝ)
  (total_time : ℝ)
  (speed_ratio : ℝ) :
  distance_to_work = 30 →
  traffic_time_to_work = 30 →
  traffic_time_back = 15 →
  total_time = 405 →
  speed_ratio = 2 →
  steves_speed_on_way_back = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_steve_speed_back_l2088_208864


namespace NUMINAMATH_GPT_maximum_ratio_squared_l2088_208890

theorem maximum_ratio_squared (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ge : a ≥ b)
  (x y : ℝ) (h_x : 0 ≤ x) (h_xa : x < a) (h_y : 0 ≤ y) (h_yb : y < b)
  (h_eq1 : a^2 + y^2 = b^2 + x^2)
  (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2) :
  (a / b)^2 ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_maximum_ratio_squared_l2088_208890


namespace NUMINAMATH_GPT_unique_solution_l2088_208811

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l2088_208811


namespace NUMINAMATH_GPT_resistance_of_second_resistor_l2088_208878

theorem resistance_of_second_resistor 
  (R1 R_total R2 : ℝ) 
  (hR1: R1 = 9) 
  (hR_total: R_total = 4.235294117647059) 
  (hFormula: 1/R_total = 1/R1 + 1/R2) : 
  R2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_resistance_of_second_resistor_l2088_208878


namespace NUMINAMATH_GPT_simplify_expression_l2088_208816

theorem simplify_expression (x : ℝ) : 3 * x^2 - 4 * x^2 = -x^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2088_208816


namespace NUMINAMATH_GPT_leftover_value_is_correct_l2088_208826

def value_of_leftover_coins (total_quarters total_dimes quarters_per_roll dimes_per_roll : ℕ) : ℝ :=
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters * 0.25) + (leftover_dimes * 0.10)

def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40

def total_quarters : ℕ := michael_quarters + anna_quarters
def total_dimes : ℕ := michael_dimes + anna_dimes

theorem leftover_value_is_correct : 
  value_of_leftover_coins total_quarters total_dimes quarters_per_roll dimes_per_roll = 10.65 :=
by
  sorry

end NUMINAMATH_GPT_leftover_value_is_correct_l2088_208826


namespace NUMINAMATH_GPT_gp_condition_necessity_l2088_208852

theorem gp_condition_necessity {a b c : ℝ} 
    (h_gp: ∃ r: ℝ, b = a * r ∧ c = a * r^2 ) : b^2 = a * c :=
by
  sorry

end NUMINAMATH_GPT_gp_condition_necessity_l2088_208852


namespace NUMINAMATH_GPT_most_probable_light_l2088_208845

theorem most_probable_light (red_duration : ℕ) (yellow_duration : ℕ) (green_duration : ℕ) :
  red_duration = 30 ∧ yellow_duration = 5 ∧ green_duration = 40 →
  (green_duration / (red_duration + yellow_duration + green_duration) > red_duration / (red_duration + yellow_duration + green_duration)) ∧
  (green_duration / (red_duration + yellow_duration + green_duration) > yellow_duration / (red_duration + yellow_duration + green_duration)) :=
by
  sorry

end NUMINAMATH_GPT_most_probable_light_l2088_208845


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l2088_208863

theorem simplify_and_evaluate_expression (a b : ℤ) (h₁ : a = -2) (h₂ : b = 1) :
  ((a - 2 * b) ^ 2 - (a + 3 * b) * (a - 2 * b)) / b = 20 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l2088_208863


namespace NUMINAMATH_GPT_kira_travel_time_l2088_208825

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end NUMINAMATH_GPT_kira_travel_time_l2088_208825


namespace NUMINAMATH_GPT_angle_SQR_measure_l2088_208834

theorem angle_SQR_measure
    (angle_PQR : ℝ)
    (angle_PQS : ℝ)
    (h1 : angle_PQR = 40)
    (h2 : angle_PQS = 15) : 
    angle_PQR - angle_PQS = 25 := 
by
    sorry

end NUMINAMATH_GPT_angle_SQR_measure_l2088_208834


namespace NUMINAMATH_GPT_maria_final_bottle_count_l2088_208820

-- Define the initial conditions
def initial_bottles : ℕ := 14
def bottles_drunk : ℕ := 8
def bottles_bought : ℕ := 45

-- State the theorem to prove
theorem maria_final_bottle_count : initial_bottles - bottles_drunk + bottles_bought = 51 :=
by
  sorry

end NUMINAMATH_GPT_maria_final_bottle_count_l2088_208820


namespace NUMINAMATH_GPT_cubic_expression_value_l2088_208817

theorem cubic_expression_value (m : ℝ) (h : m^2 + 3 * m - 2023 = 0) :
  m^3 + 2 * m^2 - 2026 * m - 2023 = -4046 :=
by
  sorry

end NUMINAMATH_GPT_cubic_expression_value_l2088_208817


namespace NUMINAMATH_GPT_hollis_student_loan_l2088_208822

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end NUMINAMATH_GPT_hollis_student_loan_l2088_208822


namespace NUMINAMATH_GPT_mean_of_xyz_l2088_208810

theorem mean_of_xyz (x y z : ℝ) (seven_mean : ℝ)
  (h1 : seven_mean = 45)
  (h2 : (7 * seven_mean + x + y + z) / 10 = 58) :
  (x + y + z) / 3 = 265 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_xyz_l2088_208810


namespace NUMINAMATH_GPT_aeroplane_speed_l2088_208807

theorem aeroplane_speed (D : ℝ) (S : ℝ) (h1 : D = S * 6) (h2 : D = 540 * (14 / 3)) :
  S = 420 := by
  sorry

end NUMINAMATH_GPT_aeroplane_speed_l2088_208807


namespace NUMINAMATH_GPT_children_difference_l2088_208868

theorem children_difference (initial_count : ℕ) (remaining_count : ℕ) (difference : ℕ) 
  (h1 : initial_count = 41) (h2 : remaining_count = 18) :
  difference = initial_count - remaining_count := 
by
  sorry

end NUMINAMATH_GPT_children_difference_l2088_208868


namespace NUMINAMATH_GPT_sum_of_integers_is_34_l2088_208801

theorem sum_of_integers_is_34 (a b : ℕ) (h1 : a - b = 6) (h2 : a * b = 272) (h3a : a > 0) (h3b : b > 0) : a + b = 34 :=
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_34_l2088_208801


namespace NUMINAMATH_GPT_radius_of_circle_is_zero_l2088_208837

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 2 * x^2 - 8 * x + 2 * y^2 - 4 * y + 10 = 0

-- Define the goal: To prove that given this equation, the radius of the circle is 0
theorem radius_of_circle_is_zero :
  ∀ x y : ℝ, circle_eq x y → (x - 2)^2 + (y - 1)^2 = 0 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_is_zero_l2088_208837


namespace NUMINAMATH_GPT_number_of_friends_l2088_208819

theorem number_of_friends (total_bill : ℝ) (discount_rate : ℝ) (paid_amount : ℝ) (n : ℝ) 
  (h_total_bill : total_bill = 400) 
  (h_discount_rate : discount_rate = 0.05)
  (h_paid_amount : paid_amount = 63.59) 
  (h_total_paid : n * paid_amount = total_bill * (1 - discount_rate)) : n = 6 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_number_of_friends_l2088_208819


namespace NUMINAMATH_GPT_restore_example_l2088_208875

theorem restore_example (x : ℕ) (y : ℕ) :
  (10 ≤ x * 8 ∧ x * 8 < 100) ∧ (100 ≤ x * 9 ∧ x * 9 < 1000) ∧ y = 98 → x = 12 ∧ x * y = 1176 :=
by
  sorry

end NUMINAMATH_GPT_restore_example_l2088_208875


namespace NUMINAMATH_GPT_slowest_pipe_time_l2088_208893

noncomputable def fill_tank_rate (R : ℝ) : Prop :=
  let rate1 := 6 * R
  let rate3 := 2 * R
  let combined_rate := 9 * R
  combined_rate = 1 / 30

theorem slowest_pipe_time (R : ℝ) (h : fill_tank_rate R) : 1 / R = 270 :=
by
  have h1 := h
  sorry

end NUMINAMATH_GPT_slowest_pipe_time_l2088_208893


namespace NUMINAMATH_GPT_net_effect_on_sale_l2088_208832

variable (P S : ℝ) (orig_revenue : ℝ := P * S) (new_revenue : ℝ := 0.7 * P * 1.8 * S)

theorem net_effect_on_sale : new_revenue = orig_revenue * 1.26 := by
  sorry

end NUMINAMATH_GPT_net_effect_on_sale_l2088_208832


namespace NUMINAMATH_GPT_trailing_zeros_sum_15_factorial_l2088_208867

theorem trailing_zeros_sum_15_factorial : 
  let k := 5
  let h := 3
  k + h = 8 := by
  sorry

end NUMINAMATH_GPT_trailing_zeros_sum_15_factorial_l2088_208867


namespace NUMINAMATH_GPT_samia_walking_distance_l2088_208840

theorem samia_walking_distance
  (speed_bike : ℝ)
  (speed_walk : ℝ)
  (total_time : ℝ) 
  (fraction_bike : ℝ) 
  (d : ℝ)
  (walking_distance : ℝ) :
  speed_bike = 15 ∧ 
  speed_walk = 4 ∧ 
  total_time = 1 ∧ 
  fraction_bike = 2/3 ∧ 
  walking_distance = (1/3) * d ∧ 
  (53 * d / 180 = total_time) → 
  walking_distance = 1.1 := 
by 
  sorry

end NUMINAMATH_GPT_samia_walking_distance_l2088_208840


namespace NUMINAMATH_GPT_smallest_a_value_l2088_208896

theorem smallest_a_value (α β γ : ℕ) (hαβγ : α * β * γ = 2010) (hα : α > 0) (hβ : β > 0) (hγ : γ > 0) :
  α + β + γ = 78 :=
by
-- Proof would go here
sorry

end NUMINAMATH_GPT_smallest_a_value_l2088_208896


namespace NUMINAMATH_GPT_total_number_of_girls_l2088_208851

-- Define the given initial number of girls and the number of girls joining the school
def initial_girls : Nat := 732
def girls_joined : Nat := 682
def total_girls : Nat := 1414

-- Formalize the problem
theorem total_number_of_girls :
  initial_girls + girls_joined = total_girls :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_total_number_of_girls_l2088_208851


namespace NUMINAMATH_GPT_least_four_digit_divisible_1_2_4_8_l2088_208808

theorem least_four_digit_divisible_1_2_4_8 : ∃ n : ℕ, ∀ d1 d2 d3 d4 : ℕ, 
  n = d1*1000 + d2*100 + d3*10 + d4 ∧
  1000 ≤ n ∧ n < 10000 ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4 ∧
  n % 1 = 0 ∧
  n % 2 = 0 ∧
  n % 4 = 0 ∧
  n % 8 = 0 ∧
  n = 1248 :=
by
  sorry

end NUMINAMATH_GPT_least_four_digit_divisible_1_2_4_8_l2088_208808


namespace NUMINAMATH_GPT_seven_nat_sum_divisible_by_5_l2088_208821

theorem seven_nat_sum_divisible_by_5 
  (a b c d e f g : ℕ)
  (h1 : (b + c + d + e + f + g) % 5 = 0)
  (h2 : (a + c + d + e + f + g) % 5 = 0)
  (h3 : (a + b + d + e + f + g) % 5 = 0)
  (h4 : (a + b + c + e + f + g) % 5 = 0)
  (h5 : (a + b + c + d + f + g) % 5 = 0)
  (h6 : (a + b + c + d + e + g) % 5 = 0)
  (h7 : (a + b + c + d + e + f) % 5 = 0)
  : 
  a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0 ∧ f % 5 = 0 ∧ g % 5 = 0 :=
sorry

end NUMINAMATH_GPT_seven_nat_sum_divisible_by_5_l2088_208821


namespace NUMINAMATH_GPT_circle_intersection_unique_point_l2088_208848

open Complex

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem circle_intersection_unique_point :
  ∃ k : ℝ, (distance (0, 0) (-5 / 2, 0) - 3 / 2 = k ∨ distance (0, 0) (-5 / 2, 0) + 3 / 2 = k)
  ↔ (k = 2 ∨ k = 5) := sorry

end NUMINAMATH_GPT_circle_intersection_unique_point_l2088_208848


namespace NUMINAMATH_GPT_mark_total_votes_l2088_208829

-- Definitions for the problem conditions
def first_area_registered_voters : ℕ := 100000
def first_area_undecided_percentage : ℕ := 5
def first_area_mark_votes_percentage : ℕ := 70

def remaining_area_increase_percentage : ℕ := 20
def remaining_area_undecided_percentage : ℕ := 7
def multiplier_for_remaining_area_votes : ℕ := 2

-- The Lean statement
theorem mark_total_votes : 
  let first_area_undecided_voters := first_area_registered_voters * first_area_undecided_percentage / 100
  let first_area_votes_cast := first_area_registered_voters - first_area_undecided_voters
  let first_area_mark_votes := first_area_votes_cast * first_area_mark_votes_percentage / 100

  let remaining_area_registered_voters := first_area_registered_voters * (1 + remaining_area_increase_percentage / 100)
  let remaining_area_undecided_voters := remaining_area_registered_voters * remaining_area_undecided_percentage / 100
  let remaining_area_votes_cast := remaining_area_registered_voters - remaining_area_undecided_voters
  let remaining_area_mark_votes := first_area_mark_votes * multiplier_for_remaining_area_votes

  let total_mark_votes := first_area_mark_votes + remaining_area_mark_votes
  total_mark_votes = 199500 := 
by
  -- We skipped the proof (it's not required as per instructions)
  sorry

end NUMINAMATH_GPT_mark_total_votes_l2088_208829


namespace NUMINAMATH_GPT_mountain_hill_school_absent_percentage_l2088_208809

theorem mountain_hill_school_absent_percentage :
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := (1 / 7) * boys
  let absent_girls := (1 / 5) * girls
  let absent_students := absent_boys + absent_girls
  let absent_percentage := (absent_students / total_students) * 100
  absent_percentage = 16.67 := sorry

end NUMINAMATH_GPT_mountain_hill_school_absent_percentage_l2088_208809


namespace NUMINAMATH_GPT_count_three_digit_multiples_13_and_5_l2088_208815

theorem count_three_digit_multiples_13_and_5 : 
  ∃ count : ℕ, count = 14 ∧ 
  ∀ n : ℕ, (100 ≤ n ∧ n ≤ 999) ∧ (n % 65 = 0) → 
  (∃ k : ℕ, n = k * 65 ∧ 2 ≤ k ∧ k ≤ 15) → count = 14 :=
by
  sorry

end NUMINAMATH_GPT_count_three_digit_multiples_13_and_5_l2088_208815


namespace NUMINAMATH_GPT_series_sum_eq_l2088_208803

noncomputable def sum_series (k : ℝ) : ℝ :=
  (∑' n : ℕ, (4 * (n + 1) + k) / 3^(n + 1))

theorem series_sum_eq (k : ℝ) : sum_series k = 3 + k / 2 := 
  sorry

end NUMINAMATH_GPT_series_sum_eq_l2088_208803


namespace NUMINAMATH_GPT_prove_a1_geq_2k_l2088_208824

variable (n k : ℕ) (a : ℕ → ℕ)
variable (h1: ∀ i, 1 ≤ i → i ≤ n → 1 < a i)
variable (h2: ∀ i j, 1 ≤ i → i < j → j ≤ n → ¬ (a i ∣ a j))
variable (h3: 3^k < 2*n ∧ 2*n < 3^(k + 1))

theorem prove_a1_geq_2k : a 1 ≥ 2^k :=
by
  sorry

end NUMINAMATH_GPT_prove_a1_geq_2k_l2088_208824


namespace NUMINAMATH_GPT_equation_solution_l2088_208859

theorem equation_solution (x y z : ℕ) :
  x^2 + y^2 = 2^z ↔ ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 1 := 
sorry

end NUMINAMATH_GPT_equation_solution_l2088_208859


namespace NUMINAMATH_GPT_box_surface_area_l2088_208814

theorem box_surface_area
  (a b c : ℕ)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : a * b * c = 280) : 2 * (a * b + b * c + c * a) = 262 := 
sorry

end NUMINAMATH_GPT_box_surface_area_l2088_208814


namespace NUMINAMATH_GPT_remainder_when_product_divided_by_5_l2088_208872

def n1 := 1483
def n2 := 1773
def n3 := 1827
def n4 := 2001
def mod5 (n : Nat) : Nat := n % 5

theorem remainder_when_product_divided_by_5 :
  mod5 (n1 * n2 * n3 * n4) = 3 :=
sorry

end NUMINAMATH_GPT_remainder_when_product_divided_by_5_l2088_208872


namespace NUMINAMATH_GPT_total_steps_to_times_square_l2088_208830

-- Define the conditions
def steps_to_rockefeller : ℕ := 354
def steps_to_times_square_from_rockefeller : ℕ := 228

-- State the theorem using the conditions
theorem total_steps_to_times_square : 
  steps_to_rockefeller + steps_to_times_square_from_rockefeller = 582 := 
  by 
    -- We skip the proof for now
    sorry

end NUMINAMATH_GPT_total_steps_to_times_square_l2088_208830


namespace NUMINAMATH_GPT_b_gets_more_than_c_l2088_208879

-- Define A, B, and C as real numbers
variables (A B C : ℝ)

theorem b_gets_more_than_c 
  (h1 : A = 3 * B)
  (h2 : B = C + 25)
  (h3 : A + B + C = 645)
  (h4 : B = 134) : 
  B - C = 25 :=
by
  -- Using the conditions from the problem
  sorry

end NUMINAMATH_GPT_b_gets_more_than_c_l2088_208879


namespace NUMINAMATH_GPT_fg_of_3_eq_97_l2088_208866

def f (x : ℕ) : ℕ := 4 * x - 3
def g (x : ℕ) : ℕ := (x + 2) ^ 2

theorem fg_of_3_eq_97 : f (g 3) = 97 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_eq_97_l2088_208866


namespace NUMINAMATH_GPT_tan_double_angle_l2088_208856

theorem tan_double_angle (α : ℝ) (h1 : Real.sin (5 * Real.pi / 6) = 1 / 2)
  (h2 : Real.cos (5 * Real.pi / 6) = -Real.sqrt 3 / 2) : 
  Real.tan (2 * α) = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_tan_double_angle_l2088_208856


namespace NUMINAMATH_GPT_solve_equation_l2088_208835

theorem solve_equation :
  ∀ x y : ℝ, (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 :=
by
  intro x y
  sorry

end NUMINAMATH_GPT_solve_equation_l2088_208835


namespace NUMINAMATH_GPT_rope_length_eqn_l2088_208841

theorem rope_length_eqn (x : ℝ) : 8^2 + (x - 3)^2 = x^2 := 
by 
  sorry

end NUMINAMATH_GPT_rope_length_eqn_l2088_208841


namespace NUMINAMATH_GPT_find_unknown_number_l2088_208884

-- Define the problem conditions and required proof
theorem find_unknown_number (a b : ℕ) (h1 : 2 * a = 3 + b) (h2 : (a - 6)^2 = 3 * b) : b = 3 ∨ b = 27 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l2088_208884


namespace NUMINAMATH_GPT_ratio_of_width_to_length_l2088_208839

theorem ratio_of_width_to_length (w l : ℕ) (h1 : w * l = 800) (h2 : l - w = 20) : w / l = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_width_to_length_l2088_208839


namespace NUMINAMATH_GPT_batsman_average_l2088_208831

theorem batsman_average (A : ℝ) (h1 : 24 * A < 95) 
                        (h2 : 24 * A + 95 = 25 * (A + 3.5)) : A + 3.5 = 11 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_l2088_208831


namespace NUMINAMATH_GPT_find_larger_integer_l2088_208885

theorem find_larger_integer (x : ℕ) (hx₁ : 4 * x > 0) (hx₂ : (x + 6) * 3 = 4 * x) : 4 * x = 72 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_integer_l2088_208885


namespace NUMINAMATH_GPT_child_stops_incur_yearly_cost_at_age_18_l2088_208871

def john_contribution (years: ℕ) (cost_per_year: ℕ) : ℕ :=
  years * cost_per_year / 2

def university_contribution (university_cost: ℕ) : ℕ :=
  university_cost / 2

def total_contribution (years_after_8: ℕ) : ℕ :=
  john_contribution 8 10000 +
  john_contribution years_after_8 20000 +
  university_contribution 250000

theorem child_stops_incur_yearly_cost_at_age_18 :
  (total_contribution n = 265000) → (n + 8 = 18) :=
by
  sorry

end NUMINAMATH_GPT_child_stops_incur_yearly_cost_at_age_18_l2088_208871


namespace NUMINAMATH_GPT_molecular_weight_of_3_moles_l2088_208838

def molecular_weight_one_mole : ℝ := 176.14
def number_of_moles : ℝ := 3
def total_weight := number_of_moles * molecular_weight_one_mole

theorem molecular_weight_of_3_moles :
  total_weight = 528.42 := sorry

end NUMINAMATH_GPT_molecular_weight_of_3_moles_l2088_208838


namespace NUMINAMATH_GPT_find_number_l2088_208806

theorem find_number (x : ℤ) (h : 42 + 3 * x - 10 = 65) : x = 11 := 
by 
  sorry 

end NUMINAMATH_GPT_find_number_l2088_208806


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2088_208881

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x - 1
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - a

-- Prove that if f is increasing on ℝ, then a ∈ (-∞, 0]
theorem problem1 (a : ℝ) : (∀ x y : ℝ, x ≤ y → f x a ≤ f y a) → a ≤ 0 :=
sorry

-- Prove that if f is decreasing on (-1, 1), then a ∈ [3, ∞)
theorem problem2 (a : ℝ) : (∀ x y : ℝ, -1 < x → x < 1 → -1 < y → y < 1 → x ≤ y → f x a ≥ f y a) → 3 ≤ a :=
sorry

-- Prove that if the decreasing interval of f is (-1, 1), then a = 3
theorem problem3 (a : ℝ) : (∀ x : ℝ, (abs x < 1) ↔ f' x a < 0) → a = 3 :=
sorry

-- Prove that if f is not monotonic on (-1, 1), then a ∈ (0, 3)
theorem problem4 (a : ℝ) : (¬(∀ x : ℝ, -1 < x → x < 1 → (f' x a = 0) ∨ (f' x a ≠ 0))) → (0 < a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l2088_208881


namespace NUMINAMATH_GPT_fair_coin_flip_difference_l2088_208860

theorem fair_coin_flip_difference :
  let P1 := 5 * (1 / 2)^5;
  let P2 := (1 / 2)^5;
  P1 - P2 = 9 / 32 :=
by
  sorry

end NUMINAMATH_GPT_fair_coin_flip_difference_l2088_208860


namespace NUMINAMATH_GPT_balloon_count_correct_l2088_208887

def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def silver_balloons : ℕ := 2 * gold_balloons
def total_balloons : ℕ := gold_balloons + silver_balloons + black_balloons

theorem balloon_count_correct : total_balloons = 573 := by
  sorry

end NUMINAMATH_GPT_balloon_count_correct_l2088_208887


namespace NUMINAMATH_GPT_dave_has_20_more_than_derek_l2088_208870

-- Define the amounts of money Derek and Dave start with
def initial_amount_derek : ℕ := 40
def initial_amount_dave : ℕ := 50

-- Define the amounts Derek spends
def spend_derek_lunch_self1 : ℕ := 14
def spend_derek_lunch_dad : ℕ := 11
def spend_derek_lunch_self2 : ℕ := 5
def spend_derek_dessert_sister : ℕ := 8

-- Define the amounts Dave spends
def spend_dave_lunch_mom : ℕ := 7
def spend_dave_lunch_cousin : ℕ := 12
def spend_dave_snacks_friends : ℕ := 9

-- Define calculations for total spending
def total_spend_derek : ℕ :=
  spend_derek_lunch_self1 + spend_derek_lunch_dad + spend_derek_lunch_self2 + spend_derek_dessert_sister

def total_spend_dave : ℕ :=
  spend_dave_lunch_mom + spend_dave_lunch_cousin + spend_dave_snacks_friends

-- Define remaining amount of money
def remaining_derek : ℕ :=
  initial_amount_derek - total_spend_derek

def remaining_dave : ℕ :=
  initial_amount_dave - total_spend_dave

-- Define the property to be proved
theorem dave_has_20_more_than_derek : remaining_dave - remaining_derek = 20 := by
  sorry

end NUMINAMATH_GPT_dave_has_20_more_than_derek_l2088_208870


namespace NUMINAMATH_GPT_pyramid_base_is_octagon_l2088_208862
-- Import necessary library

-- Declare the problem
theorem pyramid_base_is_octagon (A : Nat) (h : A = 8) : A = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pyramid_base_is_octagon_l2088_208862


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2088_208882

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d) 
  (h1 : a 1 + a 5 = 2) 
  (h2 : a 2 + a 14 = 12) : 
  (10 / 2) * (2 * a 1 + 9 * d) = 35 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2088_208882


namespace NUMINAMATH_GPT_smallest_largest_sum_l2088_208858

theorem smallest_largest_sum (a b c : ℝ) (m M : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : m = (1/3))
  (h4 : M = 1) :
  (m + M) = 4 / 3 := by
sorry

end NUMINAMATH_GPT_smallest_largest_sum_l2088_208858


namespace NUMINAMATH_GPT_minimum_questions_to_find_number_l2088_208804

theorem minimum_questions_to_find_number (n : ℕ) (h : n ≤ 2020) :
  ∃ m, m = 64 ∧ (∀ (strategy : ℕ → ℕ), ∃ questions : ℕ, questions ≤ m ∧ (strategy questions = n)) :=
sorry

end NUMINAMATH_GPT_minimum_questions_to_find_number_l2088_208804


namespace NUMINAMATH_GPT_car_distance_problem_l2088_208899

theorem car_distance_problem
  (d y z r : ℝ)
  (initial_distance : d = 113)
  (right_turn_distance : y = 15)
  (second_car_distance : z = 35)
  (remaining_distance : r = 28)
  (x : ℝ) :
  2 * x + z + y + r = d → 
  x = 17.5 :=
by
  intros h
  sorry  

end NUMINAMATH_GPT_car_distance_problem_l2088_208899


namespace NUMINAMATH_GPT_largest_divisor_of_n_l2088_208869

theorem largest_divisor_of_n (n : ℕ) (h_pos : 0 < n) (h_div : 7200 ∣ n^2) : 60 ∣ n := 
sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l2088_208869


namespace NUMINAMATH_GPT_total_area_calculations_l2088_208865

noncomputable def total_area_in_hectares : ℝ :=
  let sections := 5
  let area_per_section := 60
  let conversion_factor_acre_to_hectare := 0.404686
  sections * area_per_section * conversion_factor_acre_to_hectare

noncomputable def total_area_in_square_meters : ℝ :=
  let conversion_factor_hectare_to_square_meter := 10000
  total_area_in_hectares * conversion_factor_hectare_to_square_meter

theorem total_area_calculations :
  total_area_in_hectares = 121.4058 ∧ total_area_in_square_meters = 1214058 := by
  sorry

end NUMINAMATH_GPT_total_area_calculations_l2088_208865


namespace NUMINAMATH_GPT_benito_juarez_birth_year_l2088_208876

theorem benito_juarez_birth_year (x : ℕ) (h1 : 1801 ≤ x ∧ x ≤ 1850) (h2 : x*x = 1849) : x = 1806 :=
by sorry

end NUMINAMATH_GPT_benito_juarez_birth_year_l2088_208876


namespace NUMINAMATH_GPT_gcd_not_perfect_square_l2088_208894

theorem gcd_not_perfect_square
  (m n : ℕ)
  (h1 : (m % 3 = 0 ∨ n % 3 = 0) ∧ ¬(m % 3 = 0 ∧ n % 3 = 0))
  : ¬ ∃ k : ℕ, k * k = Nat.gcd (m^2 + n^2 + 2) (m^2 * n^2 + 3) :=
by
  sorry

end NUMINAMATH_GPT_gcd_not_perfect_square_l2088_208894


namespace NUMINAMATH_GPT_circle_integer_points_l2088_208800

theorem circle_integer_points (m n : ℤ) (h : ∃ m n : ℤ, m^2 + n^2 = r ∧ 
  ∃ p q : ℤ, m^2 + n^2 = p ∧ ∃ s t : ℤ, m^2 + n^2 = q ∧ ∃ u v : ℤ, m^2 + n^2 = s ∧ 
  ∃ j k : ℤ, m^2 + n^2 = t ∧ ∃ l w : ℤ, m^2 + n^2 = u ∧ ∃ x y : ℤ, m^2 + n^2 = v ∧ 
  ∃ i b : ℤ, m^2 + n^2 = w ∧ ∃ c d : ℤ, m^2 + n^2 = b ) :
  ∃ r, r = 25 := by
    sorry

end NUMINAMATH_GPT_circle_integer_points_l2088_208800


namespace NUMINAMATH_GPT_total_profit_percentage_l2088_208812

theorem total_profit_percentage (total_apples : ℕ) (percent_sold_10 : ℝ) (percent_sold_30 : ℝ) (profit_10 : ℝ) (profit_30 : ℝ) : 
  total_apples = 280 → 
  percent_sold_10 = 0.40 → 
  percent_sold_30 = 0.60 → 
  profit_10 = 0.10 → 
  profit_30 = 0.30 → 
  ((percent_sold_10 * total_apples * (1 + profit_10) + percent_sold_30 * total_apples * (1 + profit_30) - total_apples) / total_apples * 100) = 22 := 
by 
  intros; sorry

end NUMINAMATH_GPT_total_profit_percentage_l2088_208812


namespace NUMINAMATH_GPT_track_width_track_area_l2088_208892

theorem track_width (r1 r2 : ℝ) (h1 : 2 * π * r1 - 2 * π * r2 = 24 * π) : r1 - r2 = 12 :=
by sorry

theorem track_area (r1 r2 : ℝ) (h1 : r1 = r2 + 12) : π * (r1^2 - r2^2) = π * (24 * r2 + 144) :=
by sorry

end NUMINAMATH_GPT_track_width_track_area_l2088_208892


namespace NUMINAMATH_GPT_number_of_n_factorizable_l2088_208898

theorem number_of_n_factorizable :
  ∃! n_values : Finset ℕ, (∀ n ∈ n_values, n ≤ 100 ∧ ∃ a b : ℤ, a + b = -2 ∧ a * b = -n) ∧ n_values.card = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_n_factorizable_l2088_208898


namespace NUMINAMATH_GPT_parabola_c_value_l2088_208847

theorem parabola_c_value {b c : ℝ} :
  (1:ℝ)^2 + b * (1:ℝ) + c = 2 → 
  (4:ℝ)^2 + b * (4:ℝ) + c = 5 → 
  (7:ℝ)^2 + b * (7:ℝ) + c = 2 →
  c = 9 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_parabola_c_value_l2088_208847


namespace NUMINAMATH_GPT_meaningful_fraction_l2088_208895

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_meaningful_fraction_l2088_208895


namespace NUMINAMATH_GPT_range_of_m_max_value_of_t_l2088_208854

-- Define the conditions for the quadratic equation problem
def quadratic_eq_has_real_roots (m n : ℝ) := 
  m^2 - 4 * n ≥ 0

def roots_are_negative (m : ℝ) := 
  2 ≤ m ∧ m < 3

-- Question 1: Prove range of m
theorem range_of_m (m : ℝ) (h1 : quadratic_eq_has_real_roots m (3 - m)) : 
  roots_are_negative m :=
sorry

-- Define the conditions for the inequality problem
def quadratic_inequality (m n : ℝ) (t : ℝ) := 
  t ≤ (m-1)^2 + (n-1)^2 + (m-n)^2

-- Question 2: Prove maximum value of t
theorem max_value_of_t (m n t : ℝ) (h1 : quadratic_eq_has_real_roots m n) : 
  quadratic_inequality m n t -> t ≤ 9/8 :=
sorry

end NUMINAMATH_GPT_range_of_m_max_value_of_t_l2088_208854


namespace NUMINAMATH_GPT_platform_length_correct_l2088_208818

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_cross_platform : ℝ := 30
noncomputable def time_cross_man : ℝ := 19
noncomputable def length_train : ℝ := train_speed_mps * time_cross_man
noncomputable def total_distance_cross_platform : ℝ := train_speed_mps * time_cross_platform
noncomputable def length_platform : ℝ := total_distance_cross_platform - length_train

theorem platform_length_correct : length_platform = 220 := by
  sorry

end NUMINAMATH_GPT_platform_length_correct_l2088_208818


namespace NUMINAMATH_GPT_average_speed_trip_l2088_208891

theorem average_speed_trip :
  let distance_1 := 65
  let distance_2 := 45
  let distance_3 := 55
  let distance_4 := 70
  let distance_5 := 60
  let total_time := 5
  let total_distance := distance_1 + distance_2 + distance_3 + distance_4 + distance_5
  let average_speed := total_distance / total_time
  average_speed = 59 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_trip_l2088_208891


namespace NUMINAMATH_GPT_find_a_l2088_208836

theorem find_a (a : ℂ) (h : a / (1 - I) = (1 + I) / I) : a = -2 * I := 
by
  sorry

end NUMINAMATH_GPT_find_a_l2088_208836


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l2088_208861

theorem arithmetic_sequence_value (a_1 d : ℤ) (h : (a_1 + 2 * d) + (a_1 + 7 * d) = 10) : 
  3 * (a_1 + 4 * d) + (a_1 + 6 * d) = 20 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l2088_208861


namespace NUMINAMATH_GPT_possible_ages_l2088_208897

-- Define the set of digits
def digits : Multiset ℕ := {1, 1, 2, 2, 2, 3}

-- Condition: The age must start with "211"
def starting_sequence : List ℕ := [2, 1, 1]

-- Calculate the count of possible ages
def count_ages : ℕ :=
  let remaining_digits := [2, 2, 1, 3]
  let total_permutations := Nat.factorial 4
  let repetitions := Nat.factorial 2
  total_permutations / repetitions

theorem possible_ages : count_ages = 12 := by
  -- Proof should go here but it's omitted according to instructions.
  sorry

end NUMINAMATH_GPT_possible_ages_l2088_208897


namespace NUMINAMATH_GPT_joyce_apples_l2088_208823

/-- Joyce starts with some apples. She gives 52 apples to Larry and ends up with 23 apples. 
    Prove that Joyce initially had 75 apples. -/
theorem joyce_apples (initial_apples given_apples final_apples : ℕ) 
  (h1 : given_apples = 52) 
  (h2 : final_apples = 23) 
  (h3 : initial_apples = given_apples + final_apples) : 
  initial_apples = 75 := 
by 
  sorry

end NUMINAMATH_GPT_joyce_apples_l2088_208823


namespace NUMINAMATH_GPT_find_n_from_sequence_l2088_208877

theorem find_n_from_sequence (a : ℕ → ℝ) (h₁ : ∀ n : ℕ, a n = (1 / (Real.sqrt n + Real.sqrt (n + 1))))
  (h₂ : ∃ n : ℕ, a n + a (n + 1) = Real.sqrt 11 - 3) : 9 ∈ {n | a n + a (n + 1) = Real.sqrt 11 - 3} :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_sequence_l2088_208877


namespace NUMINAMATH_GPT_total_loss_is_correct_l2088_208827

variable (A P : ℝ)
variable (Ashok_loss Pyarelal_loss : ℝ)

-- Condition 1: Ashok's capital is 1/9 of Pyarelal's capital
def ashokCapital (A P : ℝ) : Prop :=
  A = (1 / 9) * P

-- Condition 2: Pyarelal's loss was Rs 1800
def pyarelalLoss (Pyarelal_loss : ℝ) : Prop :=
  Pyarelal_loss = 1800

-- Question: What was the total loss in the business?
def totalLoss (Ashok_loss Pyarelal_loss : ℝ) : ℝ :=
  Ashok_loss + Pyarelal_loss

-- The mathematically equivalent proof problem statement
theorem total_loss_is_correct (P A : ℝ) (Ashok_loss Pyarelal_loss : ℝ)
  (h1 : ashokCapital A P)
  (h2 : pyarelalLoss Pyarelal_loss)
  (h3 : Ashok_loss = (1 / 9) * Pyarelal_loss) :
  totalLoss Ashok_loss Pyarelal_loss = 2000 := by
  sorry

end NUMINAMATH_GPT_total_loss_is_correct_l2088_208827


namespace NUMINAMATH_GPT_brianna_books_l2088_208833

theorem brianna_books :
  ∀ (books_per_month : ℕ) (given_books : ℕ) (bought_books : ℕ) (borrowed_books : ℕ) (total_books_needed : ℕ),
    (books_per_month = 2) →
    (given_books = 6) →
    (bought_books = 8) →
    (borrowed_books = bought_books - 2) →
    (total_books_needed = 12 * books_per_month) →
    (total_books_needed - (given_books + bought_books + borrowed_books)) = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_brianna_books_l2088_208833


namespace NUMINAMATH_GPT_lcm_18_28_45_65_eq_16380_l2088_208889

theorem lcm_18_28_45_65_eq_16380 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 :=
sorry

end NUMINAMATH_GPT_lcm_18_28_45_65_eq_16380_l2088_208889


namespace NUMINAMATH_GPT_find_ks_l2088_208805

theorem find_ks (n : ℕ) (h_pos : 0 < n) :
  ∀ k, k ∈ (Finset.range (2 * n * n + 1)).erase 0 ↔ (n^2 - n + 1 ≤ k ∧ k ≤ n^2) ∨ (2*n ∣ k ∧ k ≥ n^2 - n + 1) :=
sorry

end NUMINAMATH_GPT_find_ks_l2088_208805


namespace NUMINAMATH_GPT_tan_alpha_expression_l2088_208842

theorem tan_alpha_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 2 * Real.cos α) = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_expression_l2088_208842


namespace NUMINAMATH_GPT_joshua_bottle_caps_l2088_208873

theorem joshua_bottle_caps (initial_caps : ℕ) (additional_caps : ℕ) (total_caps : ℕ) 
  (h1 : initial_caps = 40) 
  (h2 : additional_caps = 7) 
  (h3 : total_caps = initial_caps + additional_caps) : 
  total_caps = 47 := 
by 
  sorry

end NUMINAMATH_GPT_joshua_bottle_caps_l2088_208873


namespace NUMINAMATH_GPT_nancy_packs_l2088_208802

theorem nancy_packs (total_bars packs_bars : ℕ) (h_total : total_bars = 30) (h_packs : packs_bars = 5) :
  total_bars / packs_bars = 6 :=
by
  sorry

end NUMINAMATH_GPT_nancy_packs_l2088_208802


namespace NUMINAMATH_GPT_monthly_installment_amount_l2088_208846

variable (cashPrice : ℕ) (deposit : ℕ) (monthlyInstallments : ℕ) (savingsIfCash : ℕ)

-- Defining the conditions
def conditions := 
  cashPrice = 8000 ∧ 
  deposit = 3000 ∧ 
  monthlyInstallments = 30 ∧ 
  savingsIfCash = 4000

-- Proving the amount of each monthly installment
theorem monthly_installment_amount (h : conditions cashPrice deposit monthlyInstallments savingsIfCash) : 
  (12000 - deposit) / monthlyInstallments = 300 :=
sorry

end NUMINAMATH_GPT_monthly_installment_amount_l2088_208846


namespace NUMINAMATH_GPT_geom_prog_common_ratio_l2088_208886

-- Definition of a geometric progression
def geom_prog (u : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)

-- Statement of the problem
theorem geom_prog_common_ratio (u : ℕ → ℝ) (q : ℝ) (hq : ∀ n ≥ 1, u (n + 1) = u n + u (n - 1)) :
  (q = (1 + Real.sqrt 5) / 2) ∨ (q = (1 - Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_geom_prog_common_ratio_l2088_208886


namespace NUMINAMATH_GPT_correct_operation_l2088_208874

variables {a b : ℝ}

theorem correct_operation : (5 * a * b - 6 * a * b = -1 * a * b) := by
  sorry

end NUMINAMATH_GPT_correct_operation_l2088_208874


namespace NUMINAMATH_GPT_sum_of_coordinates_l2088_208850

-- Define the points C and D and the conditions
def point_C : ℝ × ℝ := (0, 0)

def point_D (x : ℝ) : ℝ × ℝ := (x, 5)

def slope_CD (x : ℝ) : Prop :=
  (5 - 0) / (x - 0) = 3 / 4

-- The required theorem to be proved
theorem sum_of_coordinates (D : ℝ × ℝ)
  (hD : D.snd = 5)
  (h_slope : slope_CD D.fst) :
  D.fst + D.snd = 35 / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_l2088_208850


namespace NUMINAMATH_GPT_rational_roots_iff_a_eq_b_l2088_208849

theorem rational_roots_iff_a_eq_b (a b : ℤ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℚ, x^2 + (a + b)^2 * x + 4 * a * b = 1) ↔ a = b :=
by
  sorry

end NUMINAMATH_GPT_rational_roots_iff_a_eq_b_l2088_208849


namespace NUMINAMATH_GPT_simplify_expression_l2088_208853

variable (x : ℝ)

theorem simplify_expression :
  3 * x^3 + 4 * x + 5 * x^2 + 2 - (7 - 3 * x^3 - 4 * x - 5 * x^2) =
  6 * x^3 + 10 * x^2 + 8 * x - 5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2088_208853


namespace NUMINAMATH_GPT_proof_problem_l2088_208883

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Definition for statement 1
def statement1 := f 0 = 0

-- Definition for statement 2
def statement2 := (∃ x > 0, ∀ y > 0, f x ≥ f y) → (∃ x < 0, ∀ y < 0, f x ≤ f y)

-- Definition for statement 3
def statement3 := (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) → (∀ x ≤ -1, ∀ y ≤ -1, x < y → f y < f x)

-- Definition for statement 4
def statement4 := (∀ x > 0, f x = x^2 - 2 * x) → (∀ x < 0, f x = -x^2 - 2 * x)

-- Combined proof problem
theorem proof_problem :
  (statement1 f) ∧ (statement2 f) ∧ (statement4 f) ∧ ¬ (statement3 f) :=
by sorry

end NUMINAMATH_GPT_proof_problem_l2088_208883


namespace NUMINAMATH_GPT_sequence_a4_eq_15_l2088_208844

theorem sequence_a4_eq_15 (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * a n + 1) → a 4 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a4_eq_15_l2088_208844
