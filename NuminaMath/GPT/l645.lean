import Mathlib

namespace find_r_given_conditions_l645_64545

theorem find_r_given_conditions (p c r : ℝ) (h1 : p * r = 360) (h2 : 6 * c * r = 15) (h3 : r = 4) : r = 4 :=
by
  sorry

end find_r_given_conditions_l645_64545


namespace find_A_l645_64576

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) (h : 10 * A + 3 + 610 + B = 695) : A = 8 :=
by {
  sorry
}

end find_A_l645_64576


namespace seokgi_money_l645_64520

open Classical

variable (S Y : ℕ)

theorem seokgi_money (h1 : ∃ S, S + 2000 < S + Y + 2000)
                     (h2 : ∃ Y, Y + 1500 < S + Y + 1500)
                     (h3 : 3500 + (S + Y + 2000) = (S + Y) + 3500)
                     (boat_price1: ∀ S, S + 2000 = S + 2000)
                     (boat_price2: ∀ Y, Y + 1500 = Y + 1500) :
  S = 5000 :=
by sorry

end seokgi_money_l645_64520


namespace pq_combined_work_rate_10_days_l645_64543

/-- Conditions: 
1. wr_p = wr_qr, where wr_qr is the combined work rate of q and r
2. wr_r allows completing the work in 30 days
3. wr_q allows completing the work in 30 days

We need to prove that the combined work rate of p and q allows them to complete the work in 10 days.
-/
theorem pq_combined_work_rate_10_days
  (wr_p wr_q wr_r wr_qr : ℝ)
  (h1 : wr_p = wr_qr)
  (h2 : wr_r = 1/30)
  (h3 : wr_q = 1/30) :
  wr_p + wr_q = 1/10 := by
  sorry

end pq_combined_work_rate_10_days_l645_64543


namespace fraction_e_over_d_l645_64594

theorem fraction_e_over_d :
  ∃ (d e : ℝ), (∀ (x : ℝ), x^2 + 2600 * x + 2600 = (x + d)^2 + e) ∧ e / d = -1298 :=
by 
  sorry

end fraction_e_over_d_l645_64594


namespace president_savings_l645_64550

theorem president_savings (total_funds : ℕ) (friends_percentage : ℕ) (family_percentage : ℕ) 
  (friends_contradiction funds_left family_contribution fundraising_amount : ℕ) :
  total_funds = 10000 →
  friends_percentage = 40 →
  family_percentage = 30 →
  friends_contradiction = (total_funds * friends_percentage) / 100 →
  funds_left = total_funds - friends_contradiction →
  family_contribution = (funds_left * family_percentage) / 100 →
  fundraising_amount = funds_left - family_contribution →
  fundraising_amount = 4200 :=
by
  intros
  sorry

end president_savings_l645_64550


namespace earnings_difference_l645_64557

-- We define the price per bottle for each company and the number of bottles sold by each company.
def priceA : ℝ := 4
def priceB : ℝ := 3.5
def quantityA : ℕ := 300
def quantityB : ℕ := 350

-- We define the earnings for each company based on the provided conditions.
def earningsA : ℝ := priceA * quantityA
def earningsB : ℝ := priceB * quantityB

-- We state the theorem that the difference in earnings is $25.
theorem earnings_difference : (earningsB - earningsA) = 25 := by
  -- Proof omitted.
  sorry

end earnings_difference_l645_64557


namespace peculiar_looking_less_than_500_l645_64599

def is_composite (n : ℕ) : Prop :=
  1 < n ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

def peculiar_looking (n : ℕ) : Prop :=
  is_composite n ∧ ¬ (n % 2 = 0 ∨ n % 3 = 0 ∨ n % 7 = 0 ∨ n % 11 = 0)

theorem peculiar_looking_less_than_500 :
  ∃ n, n = 33 ∧ ∀ k, k < 500 → peculiar_looking k → k = n :=
sorry

end peculiar_looking_less_than_500_l645_64599


namespace arithmetic_sequence_term_12_l645_64597

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_term_12 (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_sum : a 6 + a 10 = 16)
  (h_a4 : a 4 = 1) :
  a 12 = 15 :=
by
  -- The following line ensures the theorem compiles correctly.
  sorry

end arithmetic_sequence_term_12_l645_64597


namespace x1_value_l645_64515

theorem x1_value (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h2 : (1 - x1)^2 + 2 * (x1 - x2)^2 + 2 * (x2 - x3)^2 + x3^2 = 1 / 2) : 
  x1 = 2 / 3 :=
sorry

end x1_value_l645_64515


namespace cost_of_apples_is_2_l645_64570

variable (A : ℝ)

def cost_of_apples (A : ℝ) : ℝ := 5 * A
def cost_of_sugar (A : ℝ) : ℝ := 3 * (A - 1)
def cost_of_walnuts : ℝ := 0.5 * 6
def total_cost (A : ℝ) : ℝ := cost_of_apples A + cost_of_sugar A + cost_of_walnuts

theorem cost_of_apples_is_2 (A : ℝ) (h : total_cost A = 16) : A = 2 := 
by 
  sorry

end cost_of_apples_is_2_l645_64570


namespace cooking_ways_l645_64517

noncomputable def comb (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem cooking_ways : comb 5 2 = 10 :=
  by
  sorry

end cooking_ways_l645_64517


namespace biff_break_even_hours_l645_64534

theorem biff_break_even_hours :
  let ticket := 11
  let drinks_snacks := 3
  let headphones := 16
  let expenses := ticket + drinks_snacks + headphones
  let hourly_income := 12
  let hourly_wifi_cost := 2
  let net_income_per_hour := hourly_income - hourly_wifi_cost
  expenses / net_income_per_hour = 3 :=
by
  sorry

end biff_break_even_hours_l645_64534


namespace day_of_18th_day_of_month_is_tuesday_l645_64567

theorem day_of_18th_day_of_month_is_tuesday
  (day_of_24th_is_monday : ℕ → ℕ)
  (mod_seven : ∀ n, n % 7 = n)
  (h24 : day_of_24th_is_monday 24 = 1) : day_of_24th_is_monday 18 = 2 :=
by
  sorry

end day_of_18th_day_of_month_is_tuesday_l645_64567


namespace sisters_work_together_days_l645_64559

-- Definitions based on conditions
def task_completion_rate_older_sister : ℚ := 1/10
def task_completion_rate_younger_sister : ℚ := 1/20
def work_done_by_older_sister_alone : ℚ := 4 * task_completion_rate_older_sister
def remaining_task_after_older_sister : ℚ := 1 - work_done_by_older_sister_alone
def combined_work_rate : ℚ := task_completion_rate_older_sister + task_completion_rate_younger_sister

-- Statement of the proof problem
theorem sisters_work_together_days : 
  (combined_work_rate * x = remaining_task_after_older_sister) → 
  (x = 4) :=
by
  sorry

end sisters_work_together_days_l645_64559


namespace S_5_value_l645_64581

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop := 
  ∃ q > 0, ∀ n, a (n + 1) = q * a n

variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

axiom a2a4 (h : geometric_sequence a) : a 1 * a 3 = 16
axiom S3 : S 3 = 7

theorem S_5_value 
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = a 0 * (1 - (a 1)^(n)) / (1 - a 1)) :
  S 5 = 31 :=
sorry

end S_5_value_l645_64581


namespace smallest_b_for_factorization_l645_64560

theorem smallest_b_for_factorization : ∃ (p q : ℕ), p * q = 2007 ∧ p + q = 232 :=
by
  sorry

end smallest_b_for_factorization_l645_64560


namespace prime_p_geq_5_div_24_l645_64522

theorem prime_p_geq_5_div_24 (p : ℕ) (hp : Nat.Prime p) (hp_geq_5 : p ≥ 5) : 24 ∣ (p^2 - 1) :=
sorry

end prime_p_geq_5_div_24_l645_64522


namespace fill_pool_with_B_only_l645_64501

theorem fill_pool_with_B_only
    (time_AB : ℝ)
    (R_AB : time_AB = 30)
    (time_A_B_then_B : ℝ)
    (R_A_B_then_B : (10 / 30 + (time_A_B_then_B - 10) / time_A_B_then_B) = 1)
    (only_B_time : ℝ)
    (R_B : only_B_time = 60) :
    only_B_time = 60 :=
by
    sorry

end fill_pool_with_B_only_l645_64501


namespace sandy_age_l645_64588

theorem sandy_age (S M : ℕ) (h1 : M = S + 14) (h2 : S / M = 7 / 9) : S = 49 :=
sorry

end sandy_age_l645_64588


namespace total_reams_of_paper_l645_64593

def reams_for_haley : ℕ := 2
def reams_for_sister : ℕ := 3

theorem total_reams_of_paper : reams_for_haley + reams_for_sister = 5 := by
  sorry

end total_reams_of_paper_l645_64593


namespace inequality_solution_set_min_value_of_x_plus_y_l645_64507

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (2 * a + 1) * x + 2

theorem inequality_solution_set (a : ℝ) :
  (if a < 0 then (∀ x : ℝ, f a x > 0 ↔ (1/a < x ∧ x < 2))
   else if a = 0 then (∀ x : ℝ, f a x > 0 ↔ x < 2)
   else if 0 < a ∧ a < 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 2 ∨ 1/a < x))
   else if a = 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x ≠ 2))
   else if a > 1/2 then (∀ x : ℝ, f a x > 0 ↔ (x < 1/a ∨ x > 2))
   else false) := 
sorry

theorem min_value_of_x_plus_y (a : ℝ) (h : 0 < a) (x y : ℝ) (hx : y ≥ f a (|x|)) :
  x + y ≥ -a - (1/a) := 
sorry

end inequality_solution_set_min_value_of_x_plus_y_l645_64507


namespace computer_hardware_contract_prob_l645_64551

theorem computer_hardware_contract_prob :
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  ∃ P_H : ℝ, P_at_least_one = P_H + P_S - P_H_and_S ∧ P_H = 0.8 :=
by
  -- Let definitions and initial conditions
  let P_not_S := 3 / 5
  let P_at_least_one := 5 / 6
  let P_H_and_S := 0.3666666666666667
  let P_S := 1 - P_not_S
  -- Solve for P(H)
  let P_H := 0.8
  -- Show the proof of the calculation
  sorry

end computer_hardware_contract_prob_l645_64551


namespace largest_divisor_of_expression_l645_64592

theorem largest_divisor_of_expression 
  (x : ℤ) (h_odd : x % 2 = 1) :
  384 ∣ (8*x + 4) * (8*x + 8) * (4*x + 2) :=
sorry

end largest_divisor_of_expression_l645_64592


namespace find_real_solutions_l645_64569

theorem find_real_solutions (x : ℝ) (h1 : x ≠ 4) (h2 : x ≠ 5) :
  ( (x - 3) * (x - 4) * (x - 5) * (x - 4) * (x - 3) ) / ( (x - 4) * (x - 5) ) = -1 ↔ x = 10 / 3 ∨ x = 2 / 3 :=
by sorry

end find_real_solutions_l645_64569


namespace polynomial_degree_le_one_l645_64528

theorem polynomial_degree_le_one {P : ℝ → ℝ} (h : ∀ x : ℝ, 2 * P x = P (x + 3) + P (x - 3)) :
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x + b :=
sorry

end polynomial_degree_le_one_l645_64528


namespace integer_values_of_a_l645_64552

theorem integer_values_of_a (a : ℤ) : 
  (∃ x : ℤ, x^4 + 4 * x^3 + a * x^2 + 8 = 0) ↔ (a = -14 ∨ a = -13 ∨ a = -5 ∨ a = 2) :=
sorry

end integer_values_of_a_l645_64552


namespace magnitude_of_power_l645_64565

noncomputable def z : ℂ := 4 + 2 * Real.sqrt 2 * Complex.I

theorem magnitude_of_power :
  Complex.abs (z ^ 4) = 576 := by
  sorry

end magnitude_of_power_l645_64565


namespace problem_1_problem_2_l645_64531

theorem problem_1 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c)) ≥ 3 / 4 := 
sorry

theorem problem_2 (a b c d : ℝ) (h : d > 0) (h_sum : a + b + c + d = 3) :
  (a / (b + 2 * c + 3 * d) + b / (c + 2 * d + 3 * a) + c / (d + 2 * a + 3 * b) + d / (a + 2 * b + 3 * c)) ≥ 2 / 3 :=
sorry

end problem_1_problem_2_l645_64531


namespace problem_1_problem_2_problem_3_l645_64549

-- Problem 1: Prove that if the inequality |x-1| - |x-2| < a holds for all x in ℝ, then a > 1.
theorem problem_1 (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a) → a > 1 :=
sorry

-- Problem 2: Prove that if the inequality |x-1| - |x-2| < a has at least one real solution, then a > -1.
theorem problem_2 (a : ℝ) :
  (∃ x : ℝ, |x - 1| - |x - 2| < a) → a > -1 :=
sorry

-- Problem 3: Prove that if the solution set of the inequality |x-1| - |x-2| < a is empty, then a ≤ -1.
theorem problem_3 (a : ℝ) :
  (¬∃ x : ℝ, |x - 1| - |x - 2| < a) → a ≤ -1 :=
sorry

end problem_1_problem_2_problem_3_l645_64549


namespace passenger_waiting_time_probability_l645_64539

def bus_arrival_interval : ℕ := 5

def waiting_time_limit : ℕ := 3

/-- 
  Prove that for a bus arriving every 5 minutes,
  the probability that a passenger's waiting time 
  is no more than 3 minutes, given the passenger 
  arrives at a random time, is 3/5. 
--/
theorem passenger_waiting_time_probability 
  (bus_interval : ℕ) (time_limit : ℕ) 
  (random_arrival : ℝ) :
  bus_interval = 5 →
  time_limit = 3 →
  0 ≤ random_arrival ∧ random_arrival < bus_interval →
  (random_arrival ≤ time_limit) →
  (random_arrival / ↑bus_interval) = 3 / 5 :=
by
  sorry

end passenger_waiting_time_probability_l645_64539


namespace lateral_surface_area_of_prism_l645_64575

theorem lateral_surface_area_of_prism (h : ℝ) (angle : ℝ) (h_pos : 0 < h) (angle_eq : angle = 60) :
  ∃ S : ℝ, S = 6 * h^2 :=
by
  sorry

end lateral_surface_area_of_prism_l645_64575


namespace overall_gain_percent_l645_64563

theorem overall_gain_percent {initial_cost first_repair second_repair third_repair sell_price : ℝ} 
  (h1 : initial_cost = 800) 
  (h2 : first_repair = 150) 
  (h3 : second_repair = 75) 
  (h4 : third_repair = 225) 
  (h5 : sell_price = 1600) :
  (sell_price - (initial_cost + first_repair + second_repair + third_repair)) / 
  (initial_cost + first_repair + second_repair + third_repair) * 100 = 28 := 
by 
  sorry

end overall_gain_percent_l645_64563


namespace geometric_series_sum_l645_64577

theorem geometric_series_sum (n : ℕ) : 
  let a₁ := 2
  let q := 2
  let S_n := a₁ * (1 - q^n) / (1 - q)
  S_n = 2 - 2^(n + 1) := 
by
  sorry

end geometric_series_sum_l645_64577


namespace combination_equality_l645_64530

theorem combination_equality : 
  Nat.choose 5 2 + Nat.choose 5 3 = 20 := 
by 
  sorry

end combination_equality_l645_64530


namespace train_crossing_time_l645_64568

namespace TrainCrossingProblem

def length_of_train : ℕ := 250
def length_of_bridge : ℕ := 300
def speed_of_train_kmph : ℕ := 36
def speed_of_train_mps : ℕ := 10 -- conversion from 36 kmph to m/s
def total_distance : ℕ := length_of_train + length_of_bridge -- 250 + 300
def expected_time : ℕ := 55

theorem train_crossing_time : 
  (total_distance / speed_of_train_mps) = expected_time :=
by
  sorry
end TrainCrossingProblem

end train_crossing_time_l645_64568


namespace smallest_x_y_sum_l645_64536

theorem smallest_x_y_sum (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x ≠ y) (h_fraction : 1/x + 1/y = 1/20) : x + y = 90 :=
sorry

end smallest_x_y_sum_l645_64536


namespace absolute_value_neg_2022_l645_64546

theorem absolute_value_neg_2022 : abs (-2022) = 2022 :=
by sorry

end absolute_value_neg_2022_l645_64546


namespace ticket_price_increase_l645_64538

noncomputable def y (x : ℕ) : ℝ :=
  if x ≤ 100 then
    30 * x - 50 * Real.sqrt x - 500
  else
    30 * x - 50 * Real.sqrt x - 700

theorem ticket_price_increase (m : ℝ) : 
  m * 20 - 50 * Real.sqrt 20 - 500 ≥ 0 → m ≥ 37 := sorry

end ticket_price_increase_l645_64538


namespace adam_spent_on_ferris_wheel_l645_64578

-- Define the conditions
def ticketsBought : Nat := 13
def ticketsLeft : Nat := 4
def costPerTicket : Nat := 9

-- Define the question and correct answer as a proof goal
theorem adam_spent_on_ferris_wheel : (ticketsBought - ticketsLeft) * costPerTicket = 81 := by
  sorry

end adam_spent_on_ferris_wheel_l645_64578


namespace total_fruits_correct_l645_64509

def total_fruits 
  (Jason_watermelons : Nat) (Jason_pineapples : Nat)
  (Mark_watermelons : Nat) (Mark_pineapples : Nat)
  (Sandy_watermelons : Nat) (Sandy_pineapples : Nat) : Nat :=
  Jason_watermelons + Jason_pineapples +
  Mark_watermelons + Mark_pineapples +
  Sandy_watermelons + Sandy_pineapples

theorem total_fruits_correct :
  total_fruits 37 56 68 27 11 14 = 213 :=
by
  sorry

end total_fruits_correct_l645_64509


namespace Julie_monthly_salary_l645_64558

theorem Julie_monthly_salary 
(hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) (weeks_per_month : ℕ) (missed_days : ℕ) 
(h1 : hourly_rate = 5) (h2 : hours_per_day = 8) 
(h3 : days_per_week = 6) (h4 : weeks_per_month = 4) 
(h5 : missed_days = 1) : 
hourly_rate * hours_per_day * days_per_week * weeks_per_month - hourly_rate * hours_per_day * missed_days = 920 :=
by sorry

end Julie_monthly_salary_l645_64558


namespace monroe_collection_legs_l645_64514

theorem monroe_collection_legs : 
  let ants := 12 
  let spiders := 8 
  let beetles := 15 
  let centipedes := 5 
  let legs_ants := 6 
  let legs_spiders := 8 
  let legs_beetles := 6 
  let legs_centipedes := 100
  (ants * legs_ants + spiders * legs_spiders + beetles * legs_beetles + centipedes * legs_centipedes = 726) := 
by 
  sorry

end monroe_collection_legs_l645_64514


namespace Jakes_weight_is_198_l645_64544

variable (Jake Kendra : ℕ)

-- Conditions
variable (h1 : Jake - 8 = 2 * Kendra)
variable (h2 : Jake + Kendra = 293)

theorem Jakes_weight_is_198 : Jake = 198 :=
by
  sorry

end Jakes_weight_is_198_l645_64544


namespace Jasmine_shopping_time_l645_64523

-- Define the variables for the times in minutes
def T_start := 960  -- 4:00 pm in minutes (4*60)
def T_commute := 30
def T_dryClean := 10
def T_dog := 20
def T_cooking := 90
def T_dinner := 1140  -- 7:00 pm in minutes (19*60)

-- The calculated start time for cooking in minutes
def T_startCooking := T_dinner - T_cooking

-- The time Jasmine has between arriving home and starting cooking
def T_groceryShopping := T_startCooking - (T_start + T_commute + T_dryClean + T_dog)

theorem Jasmine_shopping_time :
  T_groceryShopping = 30 := by
  sorry

end Jasmine_shopping_time_l645_64523


namespace third_number_is_42_l645_64586

variable (x : ℕ)

def number1 : ℕ := 5 * x
def number2 : ℕ := 6 * x
def number3 : ℕ := 8 * x

theorem third_number_is_42 (h : number1 x + number3 x = number2 x + 49) : number2 x = 42 :=
by
  sorry

end third_number_is_42_l645_64586


namespace num_distinct_triangles_in_octahedron_l645_64506

theorem num_distinct_triangles_in_octahedron : ∃ n : ℕ, n = 48 ∧ ∀ (V : Finset (Fin 8)), 
  V.card = 3 → (∀ {a b c : Fin 8}, a ∈ V ∧ b ∈ V ∧ c ∈ V → 
  ¬((a = 0 ∧ b = 1 ∧ c = 2) ∨ (a = 3 ∧ b = 4 ∧ c = 5) ∨ (a = 6 ∧ b = 7 ∧ c = 8)
  ∨ (a = 7 ∧ b = 0 ∧ c = 1) ∨ (a = 2 ∧ b = 3 ∧ c = 4) ∨ (a = 5 ∧ b = 6 ∧ c = 7))) :=
by sorry

end num_distinct_triangles_in_octahedron_l645_64506


namespace range_of_a_l645_64596

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x > 0 ∧ 2^x * (x - a) < 1) → a > -1 :=
by
  sorry

end range_of_a_l645_64596


namespace solution_set_l645_64580

variable (f : ℝ → ℝ)

def cond1 := ∀ x, f x = f (-x)
def cond2 := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
def cond3 := f (1/3) = 0

theorem solution_set (hf1 : cond1 f) (hf2 : cond2 f) (hf3 : cond3 f) :
  { x : ℝ | f (Real.log x / Real.log (1/8)) > 0 } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | 2 < x } :=
sorry

end solution_set_l645_64580


namespace inequality_2_inequality_4_l645_64508

variables (a b : ℝ)
variables (h₁ : 0 < a) (h₂ : 0 < b)

theorem inequality_2 (h₁ : 0 < a) (h₂ : 0 < b) : a > |a - b| - b :=
by
  sorry

theorem inequality_4 (h₁ : 0 < a) (h₂ : 0 < b) : ab + 2 / ab > 2 :=
by
  sorry

end inequality_2_inequality_4_l645_64508


namespace cristina_pace_correct_l645_64556

-- Definitions of the conditions
def head_start : ℕ := 30
def nicky_pace : ℕ := 3  -- meters per second
def time_for_catch_up : ℕ := 15  -- seconds

-- Distance covers by Nicky
def nicky_distance : ℕ := nicky_pace * time_for_catch_up

-- Total distance covered by Cristina to catch up Nicky
def cristina_distance : ℕ := nicky_distance + head_start

-- Cristina's pace
def cristina_pace : ℕ := cristina_distance / time_for_catch_up

-- Theorem statement
theorem cristina_pace_correct : cristina_pace = 5 := by 
  sorry

end cristina_pace_correct_l645_64556


namespace maries_trip_distance_l645_64591

theorem maries_trip_distance (x : ℚ)
  (h1 : x = x / 4 + 15 + x / 6) :
  x = 180 / 7 :=
by
  sorry

end maries_trip_distance_l645_64591


namespace real_no_impure_l645_64587

theorem real_no_impure {x : ℝ} (h1 : x^2 - 1 = 0) (h2 : x^2 + 3 * x + 2 ≠ 0) : x = 1 :=
by
  sorry

end real_no_impure_l645_64587


namespace scientific_notation_3080000_l645_64555

theorem scientific_notation_3080000 : (3080000 : ℝ) = 3.08 * 10^6 := 
by
  sorry

end scientific_notation_3080000_l645_64555


namespace mary_walking_speed_l645_64518

-- Definitions based on the conditions:
def distance_sharon (t : ℝ) : ℝ := 6 * t
def distance_mary (x t : ℝ) : ℝ := x * t
def total_distance (x t : ℝ) : ℝ := distance_sharon t + distance_mary x t

-- Lean statement to prove that the speed x is 4 given the conditions
theorem mary_walking_speed (x : ℝ) (t : ℝ) (h1 : t = 0.3) (h2 : total_distance x t = 3) : x = 4 :=
by
  sorry

end mary_walking_speed_l645_64518


namespace find_value_of_a_l645_64535

noncomputable def value_of_a (a : ℝ) (hyp_asymptotes_tangent_circle : Prop) : Prop :=
  a = (Real.sqrt 3) / 3 → hyp_asymptotes_tangent_circle

theorem find_value_of_a (a : ℝ) (condition1 : 0 < a)
  (condition_hyperbola : ∀ x y, x^2 / a^2 - y^2 = 1)
  (condition_circle : ∀ x y, x^2 + y^2 - 4*y + 3 = 0)
  (hyp_asymptotes_tangent_circle : Prop) :
  value_of_a a hyp_asymptotes_tangent_circle := 
sorry

end find_value_of_a_l645_64535


namespace quadratic_average_of_roots_l645_64553

theorem quadratic_average_of_roots (a b c : ℝ) (h_eq : a ≠ 0) (h_b : b = -6) (h_c : c = 3) 
  (discriminant : (b^2 - 4 * a * c) = 12) : 
  (b^2 - 4 * a * c = 12) → ((-b / (2 * a)) / 2 = 1.5) :=
by
  have a_val : a = 2 := sorry
  sorry

end quadratic_average_of_roots_l645_64553


namespace diameter_increase_l645_64582

theorem diameter_increase (D D' : ℝ) (h : π * (D' / 2) ^ 2 = 2.4336 * π * (D / 2) ^ 2) : D' / D = 1.56 :=
by
  -- Statement only, proof is omitted
  sorry

end diameter_increase_l645_64582


namespace f_36_l645_64554

variable {R : Type*} [CommRing R]
variable (f : R → R) (p q : R)

-- Conditions
axiom f_mult_add : ∀ x y, f (x * y) = f x + f y
axiom f_2 : f 2 = p
axiom f_3 : f 3 = q

-- Statement to prove
theorem f_36 : f 36 = 2 * (p + q) :=
by
  sorry

end f_36_l645_64554


namespace popsicles_eaten_l645_64595

theorem popsicles_eaten (total_minutes : ℕ) (minutes_per_popsicle : ℕ) (h : total_minutes = 405) (k : minutes_per_popsicle = 12) :
  (total_minutes / minutes_per_popsicle) = 33 :=
by
  sorry

end popsicles_eaten_l645_64595


namespace average_selling_price_is_86_l645_64524

def selling_prices := [82, 86, 90, 85, 87, 85, 86, 82, 90, 87, 85, 86, 82, 86, 87, 90]

def average (prices : List Nat) : Nat :=
  (prices.sum) / prices.length

theorem average_selling_price_is_86 :
  average selling_prices = 86 :=
by
  sorry

end average_selling_price_is_86_l645_64524


namespace geometric_sequence_term_l645_64527

noncomputable def b_n (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 1 => Real.sin x ^ 2
  | 2 => Real.sin x * Real.cos x
  | 3 => Real.cos x ^ 2 / Real.sin x
  | n + 4 => (Real.cos x / Real.sin x) ^ n * Real.cos x ^ 3 / Real.sin x ^ 2
  | _ => 0 -- Placeholder to cover all case

theorem geometric_sequence_term (x : ℝ) :
  ∃ n, b_n n x = Real.cos x + Real.sin x ∧ n = 7 := by
  sorry

end geometric_sequence_term_l645_64527


namespace max_triangle_area_l645_64510

noncomputable def parabola (x y : ℝ) : Prop := x^2 = 4 * y

theorem max_triangle_area
  (x1 y1 x2 y2 : ℝ)
  (hA : parabola x1 y1)
  (hB : parabola x2 y2)
  (h_sum_y : y1 + y2 = 2)
  (h_neq : y1 ≠ y2) :
  ∃ area : ℝ, area = 121 / 12 :=
sorry

end max_triangle_area_l645_64510


namespace smallest_divisible_by_15_18_20_is_180_l645_64500

theorem smallest_divisible_by_15_18_20_is_180 :
  ∃ n : ℕ, n > 0 ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ (20 ∣ n) ∧ ∀ m : ℕ, (m > 0 ∧ (15 ∣ m) ∧ (18 ∣ m) ∧ (20 ∣ m)) → n ≤ m ∧ n = 180 := by
  sorry

end smallest_divisible_by_15_18_20_is_180_l645_64500


namespace smallest_lcm_of_4_digit_integers_with_gcd_5_l645_64532

-- Definition of the given integers k and l
def positive_4_digit_integers (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

-- The main theorem we want to prove
theorem smallest_lcm_of_4_digit_integers_with_gcd_5 :
  ∃ (k l : ℕ), positive_4_digit_integers k ∧ positive_4_digit_integers l ∧ gcd k l = 5 ∧ lcm k l = 201000 :=
by {
  sorry
}

end smallest_lcm_of_4_digit_integers_with_gcd_5_l645_64532


namespace positive_integers_satisfy_inequality_l645_64513

theorem positive_integers_satisfy_inequality :
  ∀ (n : ℕ), 2 * n - 5 < 5 - 2 * n ↔ n = 1 ∨ n = 2 :=
by
  intro n
  sorry

end positive_integers_satisfy_inequality_l645_64513


namespace area_inside_C_but_outside_A_and_B_l645_64533

def radius_A := 1
def radius_B := 1
def radius_C := 2
def tangency_AB := true
def tangency_AC_non_midpoint := true

theorem area_inside_C_but_outside_A_and_B :
  let areaC := π * (radius_C ^ 2)
  let areaA := π * (radius_A ^ 2)
  let areaB := π * (radius_B ^ 2)
  let overlapping_area := 2 * (π * (radius_A ^ 2) / 2) -- approximation
  areaC - overlapping_area = 3 * π - 2 :=
by
  sorry

end area_inside_C_but_outside_A_and_B_l645_64533


namespace derivative_of_f_l645_64511

noncomputable def f (x : ℝ) : ℝ :=
  (Nat.choose 4 0 : ℝ) - (Nat.choose 4 1 : ℝ) * x + (Nat.choose 4 2 : ℝ) * x^2 - (Nat.choose 4 3 : ℝ) * x^3 + (Nat.choose 4 4 : ℝ) * x^4

theorem derivative_of_f : 
  ∀ (x : ℝ), (deriv f x) = 4 * (-1 + x)^3 :=
by
  sorry

end derivative_of_f_l645_64511


namespace jill_spent_50_percent_on_clothing_l645_64573

theorem jill_spent_50_percent_on_clothing (
  T : ℝ) (hT : T ≠ 0)
  (h : 0.05 * T * C + 0.10 * 0.30 * T = 0.055 * T):
  C = 0.5 :=
by
  sorry

end jill_spent_50_percent_on_clothing_l645_64573


namespace geom_seq_prod_of_terms_l645_64571

theorem geom_seq_prod_of_terms (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end geom_seq_prod_of_terms_l645_64571


namespace solve_quadratic_eq_l645_64537

theorem solve_quadratic_eq (x : ℝ) : x^2 = 2024 * x ↔ x = 0 ∨ x = 2024 :=
by sorry

end solve_quadratic_eq_l645_64537


namespace meter_to_skips_l645_64525

/-!
# Math Proof Problem
Suppose hops, skips and jumps are specific units of length. Given the following conditions:
1. \( b \) hops equals \( c \) skips.
2. \( d \) jumps equals \( e \) hops.
3. \( f \) jumps equals \( g \) meters.

Prove that one meter equals \( \frac{cef}{bdg} \) skips.
-/

theorem meter_to_skips (b c d e f g : ℝ) (h1 : b ≠ 0) (h2 : c ≠ 0) (h3 : d ≠ 0) (h4 : e ≠ 0) (h5 : f ≠ 0) (h6 : g ≠ 0) :
  (1 : ℝ) = (cef) / (bdg) :=
by
  -- skipping the proof
  sorry

end meter_to_skips_l645_64525


namespace expected_number_of_digits_is_1_55_l645_64504

def probability_one_digit : ℚ := 9 / 20
def probability_two_digits : ℚ := 1 / 2
def probability_twenty : ℚ := 1 / 20
def expected_digits : ℚ := (1 * probability_one_digit) + (2 * probability_two_digits) + (2 * probability_twenty)

theorem expected_number_of_digits_is_1_55 :
  expected_digits = 1.55 :=
sorry

end expected_number_of_digits_is_1_55_l645_64504


namespace quadratic_vertex_coordinates_l645_64590

theorem quadratic_vertex_coordinates : ∀ x : ℝ,
  (∃ y : ℝ, y = 2 * x^2 - 4 * x + 5) →
  (1, 3) = (1, 3) :=
by
  intro x
  intro h
  sorry

end quadratic_vertex_coordinates_l645_64590


namespace find_n_find_m_constant_term_find_m_max_coefficients_l645_64585

-- 1. Prove that if the sum of the binomial coefficients is 256, then n = 8.
theorem find_n (n : ℕ) (h : 2^n = 256) : n = 8 :=
by sorry

-- 2. Prove that if the constant term is 35/8, then m = ±1/2.
theorem find_m_constant_term (m : ℚ) (h : m^4 * (Nat.choose 8 4) = 35/8) : m = 1/2 ∨ m = -1/2 :=
by sorry

-- 3. Prove that if only the 6th and 7th terms have the maximum coefficients, then m = 2.
theorem find_m_max_coefficients (m : ℚ) (h1 : m ≠ 0) (h2 : m^5 * (Nat.choose 8 5) = m^6 * (Nat.choose 8 6)) : m = 2 :=
by sorry

end find_n_find_m_constant_term_find_m_max_coefficients_l645_64585


namespace minimum_black_edges_5x5_l645_64579

noncomputable def minimum_black_edges_on_border (n : ℕ) : ℕ :=
if n = 5 then 5 else 0

theorem minimum_black_edges_5x5 : 
  minimum_black_edges_on_border 5 = 5 :=
by sorry

end minimum_black_edges_5x5_l645_64579


namespace triangle_area_difference_l645_64541

theorem triangle_area_difference 
  (b h : ℝ)
  (hb : 0 < b)
  (hh : 0 < h)
  (A_base : ℝ) (A_height : ℝ)
  (hA_base: A_base = 1.20 * b)
  (hA_height: A_height = 0.80 * h)
  (A_area: ℝ) (B_area: ℝ)
  (hA_area: A_area = 0.5 * A_base * A_height)
  (hB_area: B_area = 0.5 * b * h) :
  (B_area - A_area) / B_area = 0.04 := 
by sorry

end triangle_area_difference_l645_64541


namespace contractor_total_amount_l645_64564

-- Definitions for conditions
def total_days : ℕ := 30
def absent_days : ℕ := 10
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.5

-- Definitions for calculations
def worked_days : ℕ := total_days - absent_days
def total_earned : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day

-- Goal is to prove total amount is 425
noncomputable def total_amount_received : ℝ := total_earned - total_fine

theorem contractor_total_amount : total_amount_received = 425 := by
  sorry

end contractor_total_amount_l645_64564


namespace tangent_circles_locus_l645_64561

theorem tangent_circles_locus :
  ∃ (a b : ℝ), ∀ (C1_center : ℝ × ℝ) (C2_center : ℝ × ℝ) (C1_radius : ℝ) (C2_radius : ℝ),
    C1_center = (0, 0) ∧ C2_center = (2, 0) ∧ C1_radius = 1 ∧ C2_radius = 3 ∧
    (∀ (r : ℝ), (a - 0)^2 + (b - 0)^2 = (r + C1_radius)^2 ∧ (a - 2)^2 + (b - 0)^2 = (C2_radius - r)^2) →
    84 * a^2 + 100 * b^2 - 64 * a - 64 = 0 := sorry

end tangent_circles_locus_l645_64561


namespace max_score_per_student_l645_64526

theorem max_score_per_student (score_tests : ℕ → ℕ) (avg_score_tests_lt_8 : ℕ) (combined_score_two_tests : ℕ) : (∀ i, 1 ≤ i ∧ i ≤ 8 → score_tests i ≤ 100) ∧ avg_score_tests_lt_8 = 70 ∧ combined_score_two_tests = 290 →
  ∃ max_score : ℕ, max_score = 145 := 
by
  sorry

end max_score_per_student_l645_64526


namespace find_a100_find_a1983_l645_64566

open Nat

def is_strictly_increasing (a : ℕ → ℕ) : Prop :=
  ∀ n m, n < m → a n < a m

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ k, a (a k) = 3 * k

theorem find_a100 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 100 = 181 := 
sorry

theorem find_a1983 (a : ℕ → ℕ) 
  (h_inc: is_strictly_increasing a) 
  (h_prop: sequence_property a) :
  a 1983 = 3762 := 
sorry

end find_a100_find_a1983_l645_64566


namespace count_positive_n_l645_64572

def is_factorable (n : ℕ) : Prop :=
  ∃ a b : ℤ, (a + b = -2) ∧ (a * b = - (n:ℤ))

theorem count_positive_n : 
  (∃ (S : Finset ℕ), S.card = 45 ∧ ∀ n ∈ S, (1 ≤ n ∧ n ≤ 2000) ∧ is_factorable n) :=
by
  -- Placeholder for the proof
  sorry

end count_positive_n_l645_64572


namespace profit_increase_l645_64529

theorem profit_increase (x y : ℝ) (a : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (profit_eq : y - x = x * (a / 100))
  (new_profit_eq : y - 0.95 * x = 0.95 * x * (a / 100) + 0.95 * x * (15 / 100)) :
  a = 185 :=
by
  sorry

end profit_increase_l645_64529


namespace negation_correct_l645_64574

-- Define the initial proposition
def initial_proposition : Prop :=
  ∃ x : ℝ, x < 0 ∧ x^2 - 2 * x > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∀ x : ℝ, x < 0 → x^2 - 2 * x ≤ 0

-- Statement of the theorem
theorem negation_correct :
  (¬ initial_proposition) = negated_proposition :=
by
  sorry

end negation_correct_l645_64574


namespace number_plus_273_l645_64583

theorem number_plus_273 (x : ℤ) (h : x - 477 = 273) : x + 273 = 1023 := by
  sorry

end number_plus_273_l645_64583


namespace total_concrete_weight_l645_64598

theorem total_concrete_weight (w1 w2 : ℝ) (c1 c2 : ℝ) (total_weight : ℝ)
  (h1 : w1 = 1125)
  (h2 : w2 = 1125)
  (h3 : c1 = 0.093)
  (h4 : c2 = 0.113)
  (h5 : (w1 * c1 + w2 * c2) / (w1 + w2) = 0.108) :
  total_weight = w1 + w2 :=
by
  sorry

end total_concrete_weight_l645_64598


namespace three_digit_odds_factors_count_l645_64505

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l645_64505


namespace range_of_t_l645_64521

def ellipse (x y t : ℝ) : Prop := (x^2) / 4 + (y^2) / t = 1

def distance_greater_than_one (x y t : ℝ) : Prop := 
  let a := if t > 4 then Real.sqrt t else 2
  let b := if t > 4 then 2 else Real.sqrt t
  let c := if t > 4 then Real.sqrt (t - 4) else Real.sqrt (4 - t)
  a - c > 1

theorem range_of_t (t : ℝ) : 
  (∀ x y, ellipse x y t → distance_greater_than_one x y t) ↔ 
  (3 < t ∧ t < 4) ∨ (4 < t ∧ t < 25 / 4) := 
sorry

end range_of_t_l645_64521


namespace no_minimum_value_l645_64512

noncomputable def f (x : ℝ) : ℝ :=
  (1 + 1 / Real.log (Real.sqrt (x^2 + 10) - x)) *
  (1 + 2 / Real.log (Real.sqrt (x^2 + 10) - x))

theorem no_minimum_value : ¬ ∃ x, (0 < x ∧ x < 4.5) ∧ (∀ y, (0 < y ∧ y < 4.5) → f x ≤ f y) :=
sorry

end no_minimum_value_l645_64512


namespace linear_expressions_constant_multiple_l645_64584

theorem linear_expressions_constant_multiple 
    (a b c p q r : ℝ)
    (h : (a*x + p)^2 + (b*x + q)^2 = (c*x + r)^2) : 
    a*b ≠ 0 → p*q ≠ 0 → (a / b = p / q) :=
by
  -- Given: (ax + p)^2 + (bx + q)^2 = (cx + r)^2
  -- Prove: a / b = p / q, implying that A(x) and B(x) can be expressed as the constant times C(x)
  sorry

end linear_expressions_constant_multiple_l645_64584


namespace pipes_fill_cistern_in_12_minutes_l645_64540

noncomputable def time_to_fill_cistern_with_pipes (A_fill : ℝ) (B_fill : ℝ) (C_empty : ℝ) : ℝ :=
  let A_rate := 1 / (12 * 3)          -- Pipe A's rate
  let B_rate := 1 / (8 * 3)           -- Pipe B's rate
  let C_rate := -1 / 24               -- Pipe C's rate
  let combined_rate := A_rate + B_rate - C_rate
  (1 / 3) / combined_rate             -- Time to fill remaining one-third

theorem pipes_fill_cistern_in_12_minutes :
  time_to_fill_cistern_with_pipes 12 8 24 = 12 :=
by
  sorry

end pipes_fill_cistern_in_12_minutes_l645_64540


namespace ratio_of_kids_to_adult_meals_l645_64589

theorem ratio_of_kids_to_adult_meals (k a : ℕ) (h1 : k = 8) (h2 : k + a = 12) : k / a = 2 := 
by 
  sorry

end ratio_of_kids_to_adult_meals_l645_64589


namespace probability_of_red_jelly_bean_l645_64542

-- Definitions based on conditions
def total_jelly_beans := 7 + 9 + 4 + 10
def red_jelly_beans := 7

-- Statement we want to prove
theorem probability_of_red_jelly_bean : (red_jelly_beans : ℚ) / total_jelly_beans = 7 / 30 :=
by
  -- Proof here
  sorry

end probability_of_red_jelly_bean_l645_64542


namespace add_decimal_l645_64502

theorem add_decimal (a b : ℝ) (h1 : a = 0.35) (h2 : b = 124.75) : a + b = 125.10 :=
by sorry

end add_decimal_l645_64502


namespace logical_impossibility_of_thoughts_l645_64503

variable (K Q : Prop)

/-- Assume that King and Queen are sane (sane is represented by them not believing they're insane) -/
def sane (p : Prop) : Prop :=
  ¬(p = true)

/-- Define the nested thoughts -/
def KingThinksQueenThinksKingThinksQueenOutOfMind (K Q : Prop) :=
  K ∧ Q ∧ K ∧ Q = ¬sane Q

/-- The main proposition -/
theorem logical_impossibility_of_thoughts (hK : sane K) (hQ : sane Q) : 
  ¬KingThinksQueenThinksKingThinksQueenOutOfMind K Q :=
by sorry

end logical_impossibility_of_thoughts_l645_64503


namespace ted_cookies_eaten_l645_64548

def cookies_per_tray : ℕ := 12
def trays_per_day : ℕ := 2
def days_baking : ℕ := 6
def cookies_per_day : ℕ := trays_per_day * cookies_per_tray
def total_cookies_baked : ℕ := days_baking * cookies_per_day
def cookies_eaten_by_frank : ℕ := days_baking
def cookies_before_ted : ℕ := total_cookies_baked - cookies_eaten_by_frank
def cookies_left_after_ted : ℕ := 134

theorem ted_cookies_eaten : cookies_before_ted - cookies_left_after_ted = 4 := by
  sorry

end ted_cookies_eaten_l645_64548


namespace triangle_area_inscribed_in_circle_l645_64516

theorem triangle_area_inscribed_in_circle (R : ℝ) 
    (h_pos : R > 0) 
    (h_ratio : ∃ (x : ℝ)(hx : x > 0), 2*x + 5*x + 17*x = 2*π) :
  (∃ (area : ℝ), area = (R^2 / 4)) :=
by
  sorry

end triangle_area_inscribed_in_circle_l645_64516


namespace problem_nine_chapters_l645_64562

theorem problem_nine_chapters (x y : ℝ) :
  (x + (1 / 2) * y = 50) →
  (y + (2 / 3) * x = 50) →
  (x + (1 / 2) * y = 50) ∧ (y + (2 / 3) * x = 50) :=
by
  intros h1 h2
  exact ⟨h1, h2⟩

end problem_nine_chapters_l645_64562


namespace total_hours_worked_l645_64547

def hours_per_day : ℕ := 8 -- Frank worked 8 hours on each day
def number_of_days : ℕ := 4 -- First 4 days of the week

theorem total_hours_worked : hours_per_day * number_of_days = 32 := by
  sorry

end total_hours_worked_l645_64547


namespace find_b_l645_64519

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (317212435 * 101 - b) % 25 = 0 ∧ b = 13 := by
  sorry

end find_b_l645_64519
