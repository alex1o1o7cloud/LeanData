import Mathlib

namespace willam_tax_payment_correct_l436_43684

noncomputable def willamFarmTax : ℝ :=
  let totalTax := 3840
  let willamPercentage := 0.2777777777777778
  totalTax * willamPercentage

-- Lean theorem statement for the problem
theorem willam_tax_payment_correct : 
  willamFarmTax = 1066.67 :=
by
  sorry

end willam_tax_payment_correct_l436_43684


namespace john_moves_3594_pounds_l436_43699

def bench_press_weight := 15
def bench_press_reps := 10
def bench_press_sets := 3

def bicep_curls_weight := 12
def bicep_curls_reps := 8
def bicep_curls_sets := 4

def squats_weight := 50
def squats_reps := 12
def squats_sets := 3

def deadlift_weight := 80
def deadlift_reps := 6
def deadlift_sets := 2

def total_weight_moved : Nat :=
  (bench_press_weight * bench_press_reps * bench_press_sets) +
  (bicep_curls_weight * bicep_curls_reps * bicep_curls_sets) +
  (squats_weight * squats_reps * squats_sets) +
  (deadlift_weight * deadlift_reps * deadlift_sets)

theorem john_moves_3594_pounds :
  total_weight_moved = 3594 := by {
    sorry
}

end john_moves_3594_pounds_l436_43699


namespace calculation_l436_43602

def operation_e (x y z : ℕ) : ℕ := 3 * x * y * z

theorem calculation :
  operation_e 3 (operation_e 4 5 6) 1 = 3240 :=
by
  sorry

end calculation_l436_43602


namespace inscribed_sphere_radius_l436_43620

theorem inscribed_sphere_radius (a α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) :
  ∃ (ρ : ℝ), ρ = a * (1 - Real.cos α) / (2 * Real.sqrt (1 + Real.cos α) * (1 + Real.sqrt (- Real.cos α))) :=
  sorry

end inscribed_sphere_radius_l436_43620


namespace total_score_is_248_l436_43659

def geography_score : ℕ := 50
def math_score : ℕ := 70
def english_score : ℕ := 66

def history_score : ℕ := (geography_score + math_score + english_score) / 3

theorem total_score_is_248 : geography_score + math_score + english_score + history_score = 248 := by
  -- proofs go here
  sorry

end total_score_is_248_l436_43659


namespace candy_per_packet_l436_43691

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l436_43691


namespace max_value_l436_43644

noncomputable def satisfies_equation (x y : ℝ) : Prop :=
  x + 4 * y - x * y = 0

theorem max_value (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : satisfies_equation x y) :
  ∃ m, m = (4 / (x + y)) ∧ m ≤ (4 / 9) :=
by
  sorry

end max_value_l436_43644


namespace find_q_l436_43601

theorem find_q (a b m p q : ℚ) (h1 : a * b = 3) (h2 : a + b = m) 
  (h3 : (a + 1/b) * (b + 1/a) = q) : 
  q = 13 / 3 := by
  sorry

end find_q_l436_43601


namespace num_valid_pairs_equals_four_l436_43666

theorem num_valid_pairs_equals_four 
  (a b : ℕ) (ha : a > 0) (hb : b > 0) (hba : b > a)
  (hcond : a * b = 3 * (a - 4) * (b - 4)) :
  ∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ s → p.1 > 0 ∧ p.2 > 0 ∧ p.2 > p.1 ∧
      p.1 * p.2 = 3 * (p.1 - 4) * (p.2 - 4) := sorry

end num_valid_pairs_equals_four_l436_43666


namespace necessary_but_not_sufficient_l436_43625

theorem necessary_but_not_sufficient (x : ℝ) : (x > 1 → x > 2) = (false) ∧ (x > 2 → x > 1) = (true) := by
  sorry

end necessary_but_not_sufficient_l436_43625


namespace favorite_number_l436_43624

theorem favorite_number (S₁ S₂ S₃ : ℕ) (total_sum : ℕ) (adjacent_sum : ℕ) 
  (h₁ : S₁ = 8) (h₂ : S₂ = 14) (h₃ : S₃ = 12) 
  (h_total_sum : total_sum = 17) 
  (h_adjacent_sum : adjacent_sum = 12) : 
  ∃ x : ℕ, x = 5 := 
by 
  sorry

end favorite_number_l436_43624


namespace range_of_x_l436_43629

theorem range_of_x (x : ℝ) (h : 4 * x - 12 ≥ 0) : x ≥ 3 := 
sorry

end range_of_x_l436_43629


namespace N_square_solutions_l436_43639

theorem N_square_solutions :
  ∀ N : ℕ, (N > 0 → ∃ k : ℕ, 2^N - 2 * N = k^2) → (N = 1 ∨ N = 2) :=
by
  sorry

end N_square_solutions_l436_43639


namespace simplified_value_l436_43661

noncomputable def simplify_expression : ℝ :=
  1 / (Real.log (3) / Real.log (20) + 1) + 
  1 / (Real.log (4) / Real.log (15) + 1) + 
  1 / (Real.log (7) / Real.log (12) + 1)

theorem simplified_value : simplify_expression = 2 :=
by {
  sorry
}

end simplified_value_l436_43661


namespace total_turnover_in_first_quarter_l436_43651

theorem total_turnover_in_first_quarter (x : ℝ) : 
  200 + 200 * (1 + x) + 200 * (1 + x) ^ 2 = 1000 :=
sorry

end total_turnover_in_first_quarter_l436_43651


namespace middle_number_is_11_l436_43647

theorem middle_number_is_11 (a b c : ℕ) (h1 : a + b = 18) (h2 : a + c = 22) (h3 : b + c = 26) (h4 : c - a = 10) :
  b = 11 :=
by
  sorry

end middle_number_is_11_l436_43647


namespace simplify_and_evaluate_expression_l436_43627

-- Define the condition
def condition (x y : ℝ) := (x - 2) ^ 2 + |y + 1| = 0

-- Define the expression
def expression (x y : ℝ) := 3 * x ^ 2 * y - (2 * x ^ 2 * y - 3 * (2 * x * y - x ^ 2 * y) + 5 * x * y)

-- State the theorem
theorem simplify_and_evaluate_expression (x y : ℝ) (h : condition x y) : expression x y = 6 :=
by
  sorry

end simplify_and_evaluate_expression_l436_43627


namespace jen_shooting_game_times_l436_43690

theorem jen_shooting_game_times (x : ℕ) (h1 : 5 * x + 9 = 19) : x = 2 := by
  sorry

end jen_shooting_game_times_l436_43690


namespace sum_abcd_eq_16_l436_43622

variable (a b c d : ℝ)

def cond1 : Prop := a^2 + b^2 + c^2 + d^2 = 250
def cond2 : Prop := a * b + b * c + c * a + a * d + b * d + c * d = 3

theorem sum_abcd_eq_16 (h1 : cond1 a b c d) (h2 : cond2 a b c d) : a + b + c + d = 16 := 
by 
  sorry

end sum_abcd_eq_16_l436_43622


namespace hotel_fee_original_flat_fee_l436_43657

theorem hotel_fee_original_flat_fee
  (f n : ℝ)
  (H1 : 0.85 * (f + 3 * n) = 210)
  (H2 : f + 6 * n = 400) :
  f = 94.12 :=
by
  -- Sorry is used to indicate that the proof is not provided
  sorry

end hotel_fee_original_flat_fee_l436_43657


namespace quadratic_coefficients_l436_43610

theorem quadratic_coefficients (x : ℝ) : 
  let a := 3
  let b := -5
  let c := 1
  3 * x^2 + 1 = 5 * x → a * x^2 + b * x + c = 0 := by
sorry

end quadratic_coefficients_l436_43610


namespace n_n_plus_1_divisible_by_2_l436_43634

theorem n_n_plus_1_divisible_by_2 (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 99) : (n * (n + 1)) % 2 = 0 := 
sorry

end n_n_plus_1_divisible_by_2_l436_43634


namespace ninth_term_is_83_l436_43682

-- Definitions based on conditions
def a : ℕ := 3
def d : ℕ := 10
def arith_sequence (n : ℕ) : ℕ := a + n * d

-- Theorem to prove the 9th term is 83
theorem ninth_term_is_83 : arith_sequence 8 = 83 :=
by
  sorry

end ninth_term_is_83_l436_43682


namespace interest_rate_l436_43611

theorem interest_rate (SI P T R : ℝ) (h1 : SI = 100) (h2 : P = 500) (h3 : T = 4) (h4 : SI = (P * R * T) / 100) :
  R = 5 :=
by
  sorry

end interest_rate_l436_43611


namespace base6_addition_correct_l436_43697

-- We define the numbers in base 6
def a_base6 : ℕ := 2 * 6^3 + 4 * 6^2 + 5 * 6^1 + 3 * 6^0
def b_base6 : ℕ := 1 * 6^4 + 6 * 6^3 + 4 * 6^2 + 3 * 6^1 + 2 * 6^0

-- Define the expected result in base 6 and its base 10 equivalent
def result_base6 : ℕ := 2 * 6^4 + 5 * 6^3 + 5 * 6^2 + 4 * 6^1 + 5 * 6^0
def result_base10 : ℕ := 3881

-- The proof statement
theorem base6_addition_correct : (a_base6 + b_base6 = result_base6) ∧ (result_base6 = result_base10) := by
  sorry

end base6_addition_correct_l436_43697


namespace washes_per_bottle_l436_43662

def bottle_cost : ℝ := 4.0
def total_weeks : ℕ := 20
def total_cost : ℝ := 20.0

theorem washes_per_bottle : (total_weeks / (total_cost / bottle_cost)) = 4 := by
  sorry

end washes_per_bottle_l436_43662


namespace total_pears_picked_l436_43679

theorem total_pears_picked :
  let mike_pears := 8
  let jason_pears := 7
  let fred_apples := 6
  -- The total number of pears picked is the sum of Mike's and Jason's pears.
  mike_pears + jason_pears = 15 :=
by {
  sorry
}

end total_pears_picked_l436_43679


namespace least_number_to_subtract_l436_43668

theorem least_number_to_subtract (n : ℕ) (d : ℕ) (h1: n = 509) (h2 : d = 9): ∃ k : ℕ, k = 5 ∧ ∃ m : ℕ, n - k = d * m :=
by
  sorry

end least_number_to_subtract_l436_43668


namespace emails_received_afternoon_is_one_l436_43600

-- Define the number of emails received by Jack in the morning
def emails_received_morning : ℕ := 4

-- Define the total number of emails received by Jack in a day
def total_emails_received : ℕ := 5

-- Define the number of emails received by Jack in the afternoon
def emails_received_afternoon : ℕ := total_emails_received - emails_received_morning

-- Prove the number of emails received by Jack in the afternoon
theorem emails_received_afternoon_is_one : emails_received_afternoon = 1 :=
by 
  -- Proof is neglected as per instructions.
  sorry

end emails_received_afternoon_is_one_l436_43600


namespace shirt_cost_l436_43655

def george_initial_money : ℕ := 100
def total_spent_on_clothes (initial_money remaining_money : ℕ) : ℕ := initial_money - remaining_money
def socks_cost : ℕ := 11
def remaining_money_after_purchase : ℕ := 65

theorem shirt_cost
  (initial_money : ℕ)
  (remaining_money : ℕ)
  (total_spent : ℕ)
  (socks_cost : ℕ)
  (remaining_money_after_purchase : ℕ) :
  initial_money = 100 →
  remaining_money = 65 →
  total_spent = initial_money - remaining_money →
  total_spent = 35 →
  socks_cost = 11 →
  remaining_money_after_purchase = remaining_money →
  (total_spent - socks_cost = 24) :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h4] at *
  exact sorry

end shirt_cost_l436_43655


namespace solve_system_eq_l436_43694

theorem solve_system_eq (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x + y ≠ 0) 
  (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) :
  (xy / (x + y) = 1 / 3) ∧ (yz / (y + z) = 1 / 4) ∧ (zx / (z + x) = 1 / 5) →
  (x = 1 / 2) ∧ (y = 1) ∧ (z = 1 / 3) :=
  sorry

end solve_system_eq_l436_43694


namespace part_a_part_b_part_c_l436_43681

-- Define the conditions for Payneful pairs
def isPaynefulPair (f g : ℝ → ℝ) : Prop :=
  (∀ x, f x ∈ Set.univ) ∧
  (∀ x, g x ∈ Set.univ) ∧
  (∀ x y, f (x + y) = f x * g y + g x * f y) ∧
  (∀ x y, g (x + y) = g x * g y - f x * f y) ∧
  (∃ a, f a ≠ 0)

-- Questions and corresponding proofs as Lean theorems
theorem part_a (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : f 0 = 0 ∧ g 0 = 1 := sorry

def h (f g : ℝ → ℝ) (x : ℝ) : ℝ := (f x) ^ 2 + (g x) ^ 2

theorem part_b (f g : ℝ → ℝ) (hf : isPaynefulPair f g) : h f g 5 * h f g (-5) = 1 := sorry

theorem part_c (f g : ℝ → ℝ) (hf : isPaynefulPair f g)
  (h_bound_f : ∀ x, -10 ≤ f x ∧ f x ≤ 10) (h_bound_g : ∀ x, -10 ≤ g x ∧ g x ≤ 10):
  h f g 2021 = 1 := sorry

end part_a_part_b_part_c_l436_43681


namespace triangle_side_length_b_l436_43649

theorem triangle_side_length_b (a b c : ℝ) (A B C : ℝ)
  (hB : B = 30) 
  (h_area : 1/2 * a * c * Real.sin (B * Real.pi/180) = 3/2) 
  (h_sine : Real.sin (A * Real.pi/180) + Real.sin (C * Real.pi/180) = 2 * Real.sin (B * Real.pi/180)) :
  b = Real.sqrt 3 + 1 :=
by
  sorry

end triangle_side_length_b_l436_43649


namespace angle_sum_proof_l436_43685

theorem angle_sum_proof (x y : ℝ) (h : 3 * x + 6 * x + (x + y) + 4 * y = 360) : x = 0 ∧ y = 72 :=
by {
  sorry
}

end angle_sum_proof_l436_43685


namespace smallest_common_multiple_l436_43640

theorem smallest_common_multiple (n : ℕ) (h8 : n % 8 = 0) (h15 : n % 15 = 0) : n = 120 :=
sorry

end smallest_common_multiple_l436_43640


namespace thirteenth_result_is_878_l436_43695

-- Definitions based on the conditions
def avg_25_results : ℕ := 50
def num_25_results : ℕ := 25

def avg_first_12_results : ℕ := 14
def num_first_12_results : ℕ := 12

def avg_last_12_results : ℕ := 17
def num_last_12_results : ℕ := 12

-- Prove the 13th result is 878 given the above conditions.
theorem thirteenth_result_is_878 : 
  ((avg_25_results * num_25_results) - ((avg_first_12_results * num_first_12_results) + (avg_last_12_results * num_last_12_results))) = 878 :=
by
  sorry

end thirteenth_result_is_878_l436_43695


namespace num_subsets_with_even_is_24_l436_43630

def A : Set ℕ := {1, 2, 3, 4, 5}
def odd_subsets_count : ℕ := 2^3

theorem num_subsets_with_even_is_24 : 
  let total_subsets := 2^5
  total_subsets - odd_subsets_count = 24 := by
  sorry

end num_subsets_with_even_is_24_l436_43630


namespace series_product_solution_l436_43613

theorem series_product_solution (y : ℚ) :
  ( (∑' n, (1 / 2) * (1 / 3) ^ n) * (∑' n, (1 / 3) * (-1 / 3) ^ n) ) = ∑' n, (1 / y) ^ (n + 1) → y = 19 / 3 :=
by
  sorry

end series_product_solution_l436_43613


namespace min_value_sq_distance_l436_43676

theorem min_value_sq_distance {x y : ℝ} (h : x^2 + y^2 - 4 * x + 2 = 0) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x^2 + y^2 - 4 * x + 2 = 0 → x^2 + (y - 2)^2 ≥ m) :=
sorry

end min_value_sq_distance_l436_43676


namespace find_quantities_of_raib_ornaments_and_pendants_l436_43667

theorem find_quantities_of_raib_ornaments_and_pendants (x y : ℕ)
  (h1 : x + y = 90)
  (h2 : 40 * x + 25 * y = 2850) :
  x = 40 ∧ y = 50 :=
sorry

end find_quantities_of_raib_ornaments_and_pendants_l436_43667


namespace roots_of_quadratic_eq_l436_43670

theorem roots_of_quadratic_eq : ∃ (x : ℝ), (x^2 - 4 = 0) ↔ (x = 2 ∨ x = -2) :=
sorry

end roots_of_quadratic_eq_l436_43670


namespace rocky_running_ratio_l436_43683

theorem rocky_running_ratio (x y : ℕ) (h1 : x = 4) (h2 : 2 * x + y = 36) : y / (2 * x) = 3 :=
by
  sorry

end rocky_running_ratio_l436_43683


namespace prism_faces_l436_43688

theorem prism_faces (E L F : ℕ) (h1 : E = 18) (h2 : 3 * L = E) (h3 : F = L + 2) : F = 8 :=
sorry

end prism_faces_l436_43688


namespace total_sand_arrived_l436_43619

theorem total_sand_arrived :
  let truck1_carry := 4.1
  let truck1_loss := 2.4
  let truck2_carry := 5.7
  let truck2_loss := 3.6
  let truck3_carry := 8.2
  let truck3_loss := 1.9
  (truck1_carry - truck1_loss) + 
  (truck2_carry - truck2_loss) + 
  (truck3_carry - truck3_loss) = 10.1 :=
by
  sorry

end total_sand_arrived_l436_43619


namespace exists_integers_for_linear_combination_l436_43609

theorem exists_integers_for_linear_combination 
  (a b c d b1 b2 : ℤ)
  (h1 : ad - bc ≠ 0)
  (h2 : ∃ k : ℤ, b1 = (ad - bc) * k)
  (h3 : ∃ q : ℤ, b2 = (ad - bc) * q) :
  ∃ x y : ℤ, a * x + b * y = b1 ∧ c * x + d * y = b2 :=
sorry

end exists_integers_for_linear_combination_l436_43609


namespace find_other_number_l436_43686

theorem find_other_number (B : ℕ)
  (HCF : Nat.gcd 24 B = 12)
  (LCM : Nat.lcm 24 B = 312) :
  B = 156 :=
by
  sorry

end find_other_number_l436_43686


namespace range_of_m_l436_43656

theorem range_of_m (m x y : ℝ) 
  (h1 : x + y = -1) 
  (h2 : 5 * x + 2 * y = 6 * m + 7) 
  (h3 : 2 * x - y < 19) : 
  m < 3 / 2 := 
sorry

end range_of_m_l436_43656


namespace point_in_third_quadrant_cos_sin_l436_43642

theorem point_in_third_quadrant_cos_sin (P : ℝ × ℝ) (hP : P = (Real.cos (2009 * Real.pi / 180), Real.sin (2009 * Real.pi / 180))) :
  P.1 < 0 ∧ P.2 < 0 :=
by
  sorry

end point_in_third_quadrant_cos_sin_l436_43642


namespace least_positive_multiple_of_13_gt_418_l436_43665

theorem least_positive_multiple_of_13_gt_418 : ∃ (n : ℕ), n > 418 ∧ (13 ∣ n) ∧ n = 429 :=
by
  sorry

end least_positive_multiple_of_13_gt_418_l436_43665


namespace determine_f_value_l436_43658

noncomputable def f (t : ℝ) : ℝ := t^2 + 2

theorem determine_f_value : f 3 = 11 := by
  sorry

end determine_f_value_l436_43658


namespace handshakes_max_number_of_men_l436_43654

theorem handshakes_max_number_of_men (n : ℕ) (h: n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end handshakes_max_number_of_men_l436_43654


namespace ticket_cost_per_ride_l436_43646

theorem ticket_cost_per_ride (total_tickets : ℕ) (spent_tickets : ℕ) (rides : ℕ) (remaining_tickets : ℕ) (cost_per_ride : ℕ) 
  (h1 : total_tickets = 79) 
  (h2 : spent_tickets = 23) 
  (h3 : rides = 8) 
  (h4 : remaining_tickets = total_tickets - spent_tickets) 
  (h5 : remaining_tickets / rides = cost_per_ride) 
  : cost_per_ride = 7 := 
sorry

end ticket_cost_per_ride_l436_43646


namespace hexagon_angle_sum_l436_43637

theorem hexagon_angle_sum 
  (mA mB mC x y : ℝ)
  (hA : mA = 34)
  (hB : mB = 80)
  (hC : mC = 30)
  (hx' : x = 36 - y) : x + y = 36 :=
by
  sorry

end hexagon_angle_sum_l436_43637


namespace factor_expression_l436_43635

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) := 
by
  sorry

end factor_expression_l436_43635


namespace apples_left_l436_43672

-- Define the initial number of apples and the conditions
def initial_apples := 150
def percent_sold_to_jill := 20 / 100
def percent_sold_to_june := 30 / 100
def apples_given_to_teacher := 2

-- Formulate the problem statement in Lean
theorem apples_left (initial_apples percent_sold_to_jill percent_sold_to_june apples_given_to_teacher : ℕ) :
  let sold_to_jill := percent_sold_to_jill * initial_apples
  let remaining_after_jill := initial_apples - sold_to_jill
  let sold_to_june := percent_sold_to_june * remaining_after_jill
  let remaining_after_june := remaining_after_jill - sold_to_june
  let final_apples := remaining_after_june - apples_given_to_teacher
  final_apples = 82 := 
by 
  sorry

end apples_left_l436_43672


namespace days_to_complete_work_l436_43643

-- Let's define the conditions as Lean definitions based on the problem.

variables (P D : ℕ)
noncomputable def original_work := P * D
noncomputable def half_work_by_double_people := 2 * P * 3

-- Here is our theorem statement
theorem days_to_complete_work : original_work P D = 2 * half_work_by_double_people P :=
by sorry

end days_to_complete_work_l436_43643


namespace find_angle_l436_43614

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end find_angle_l436_43614


namespace jane_played_rounds_l436_43689

-- Define the conditions
def points_per_round := 10
def points_ended_with := 60
def points_lost := 20

-- Define the proof problem
theorem jane_played_rounds : (points_ended_with + points_lost) / points_per_round = 8 :=
by
  sorry

end jane_played_rounds_l436_43689


namespace necessary_but_not_sufficient_l436_43618

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (b < -1 → |a| + |b| > 1) ∧ (∃ a b : ℝ, |a| + |b| > 1 ∧ b >= -1) :=
by
  sorry

end necessary_but_not_sufficient_l436_43618


namespace problem_statement_l436_43608

theorem problem_statement (a b c m : ℝ) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0) (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 :=
sorry

end problem_statement_l436_43608


namespace max_value_F_l436_43616

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x^2
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def F (x : ℝ) : ℝ :=
if f x ≥ g x then f x else g x

theorem max_value_F : ∃ x : ℝ, ∀ y : ℝ, F y ≤ F x ∧ F x = 7 / 9 := 
sorry

end max_value_F_l436_43616


namespace cells_after_one_week_l436_43632

theorem cells_after_one_week : (3 ^ 7) = 2187 :=
by sorry

end cells_after_one_week_l436_43632


namespace alex_min_additional_coins_l436_43636

theorem alex_min_additional_coins (n m k : ℕ) (h_n : n = 15) (h_m : m = 120) :
  k = 0 ↔ m = (n * (n + 1)) / 2 :=
by
  sorry

end alex_min_additional_coins_l436_43636


namespace complement_of_intersection_l436_43687

-- Declare the universal set U
def U : Set ℤ := {-1, 1, 2, 3}

-- Declare the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the given quadratic equation
def is_solution (x : ℤ) : Prop := x^2 - 2 * x - 3 = 0
def B : Set ℤ := {x : ℤ | is_solution x}

-- The main theorem to prove
theorem complement_of_intersection (A_inter_B_complement : Set ℤ) :
  A_inter_B_complement = {1, 2, 3} :=
by
  sorry

end complement_of_intersection_l436_43687


namespace C_increases_with_n_l436_43623

variables (n e R r : ℝ)
variables (h_pos_e : e > 0) (h_pos_R : R > 0)
variables (h_pos_r : r > 0) (h_R_nr : R > n * r)
noncomputable def C : ℝ := (e * n) / (R - n * r)

theorem C_increases_with_n (h_pos_e : e > 0) (h_pos_R : R > 0)
(h_pos_r : r > 0) (h_R_nr : R > n * r) (hn1 hn2 : ℝ)
(h_inequality : hn1 < hn2) : 
((e*hn1) / (R - hn1*r)) < ((e*hn2) / (R - hn2*r)) :=
by sorry

end C_increases_with_n_l436_43623


namespace fixed_point_l436_43652

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  2 + a^(1-1) = 3 :=
by
  sorry

end fixed_point_l436_43652


namespace empty_subset_singleton_l436_43628

theorem empty_subset_singleton : (∅ ⊆ ({0} : Set ℕ)) = true :=
by sorry

end empty_subset_singleton_l436_43628


namespace probability_of_odd_divisor_l436_43606

noncomputable def prime_factorization_15! : ℕ :=
  (2 ^ 11) * (3 ^ 6) * (5 ^ 3) * (7 ^ 2) * 11 * 13

def total_factors_15! : ℕ :=
  (11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def odd_factors_15! : ℕ :=
  (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)

def probability_odd_divisor_15! : ℚ :=
  odd_factors_15! / total_factors_15!

theorem probability_of_odd_divisor : probability_odd_divisor_15! = 1 / 12 :=
by
  sorry

end probability_of_odd_divisor_l436_43606


namespace correct_multiplicand_l436_43674

theorem correct_multiplicand (x : ℕ) (h1 : x * 467 = 1925817) : 
  ∃ n : ℕ, n * 467 = 1325813 :=
by
  sorry

end correct_multiplicand_l436_43674


namespace proposition_D_is_true_l436_43604

-- Define the propositions
def proposition_A : Prop := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
def proposition_B : Prop := ∀ x : ℝ, 2^x > x^2
def proposition_C : Prop := ∀ a b : ℝ, (a + b = 0 ↔ a / b = -1)
def proposition_D : Prop := ∀ a b : ℝ, (a > 1 ∧ b > 1) → a * b > 1

-- Problem statement: Proposition D is true
theorem proposition_D_is_true : proposition_D := 
by sorry

end proposition_D_is_true_l436_43604


namespace dad_steps_l436_43650

theorem dad_steps (total_steps_Masha_Yasha : ℕ) (h1 : ∀ d_steps m_steps, d_steps = 3 * m_steps) 
  (h2 : ∀ m_steps y_steps, m_steps = 3 * (y_steps / 5)) 
  (h3 : total_steps_Masha_Yasha = 400) : 
  ∃ d_steps : ℕ, d_steps = 90 :=
by
  sorry

end dad_steps_l436_43650


namespace archer_score_below_8_probability_l436_43693

theorem archer_score_below_8_probability :
  ∀ (p10 p9 p8 : ℝ), p10 = 0.2 → p9 = 0.3 → p8 = 0.3 → 
  (1 - (p10 + p9 + p8) = 0.2) :=
by
  intros p10 p9 p8 hp10 hp9 hp8
  rw [hp10, hp9, hp8]
  sorry

end archer_score_below_8_probability_l436_43693


namespace max_number_of_small_boxes_l436_43675

def volume_of_large_box (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_small_box (length width height : ℕ) : ℕ :=
  length * width * height

def number_of_small_boxes (large_volume small_volume : ℕ) : ℕ :=
  large_volume / small_volume

theorem max_number_of_small_boxes :
  let large_box_length := 4 * 100  -- in cm
  let large_box_width := 2 * 100  -- in cm
  let large_box_height := 4 * 100  -- in cm
  let small_box_length := 4  -- in cm
  let small_box_width := 2  -- in cm
  let small_box_height := 2  -- in cm
  let large_volume := volume_of_large_box large_box_length large_box_width large_box_height
  let small_volume := volume_of_small_box small_box_length small_box_width small_box_height
  number_of_small_boxes large_volume small_volume = 2000000 := by
  -- Prove the statement
  sorry

end max_number_of_small_boxes_l436_43675


namespace rewrite_sum_l436_43617

theorem rewrite_sum (S_b S : ℕ → ℕ) (n S_1 : ℕ) (a b c : ℕ) :
  b = 4 → (a + b + c) / 3 = 6 →
  S_b n = b * n + (a + b + c) / 3 * (S n - n * S_1) →
  S_b n = 4 * n + 6 * (S n - n * S_1) := by
sorry

end rewrite_sum_l436_43617


namespace exponent_logarithm_simplifies_l436_43626

theorem exponent_logarithm_simplifies :
  (1/2 : ℝ) ^ (Real.log 3 / Real.log 2 - 1) = 2 / 3 :=
by sorry

end exponent_logarithm_simplifies_l436_43626


namespace minimum_value_of_function_l436_43603

theorem minimum_value_of_function :
  ∃ x y : ℝ, 2 * x ^ 2 + 3 * x * y + 4 * y ^ 2 - 8 * x + y = 3.7391 := by
  sorry

end minimum_value_of_function_l436_43603


namespace deer_families_initial_count_l436_43615

theorem deer_families_initial_count (stayed moved_out : ℕ) (h_stayed : stayed = 45) (h_moved_out : moved_out = 34) :
  stayed + moved_out = 79 :=
by
  sorry

end deer_families_initial_count_l436_43615


namespace regular_polygons_enclosing_hexagon_l436_43621

theorem regular_polygons_enclosing_hexagon (m n : ℕ) 
  (hm : m = 6)
  (h_exterior_angle_central : 180 - ((m - 2) * 180 / m) = 60)
  (h_exterior_angle_enclosing : 2 * 60 = 120): 
  n = 3 := sorry

end regular_polygons_enclosing_hexagon_l436_43621


namespace two_mathematicians_contemporaries_l436_43664

def contemporaries_probability :=
  let total_area := 600 * 600
  let triangle_area := 1/2 * 480 * 480
  let non_contemporaneous_area := 2 * triangle_area
  let contemporaneous_area := total_area - non_contemporaneous_area
  let probability := contemporaneous_area / total_area
  probability

theorem two_mathematicians_contemporaries :
  contemporaries_probability = 9 / 25 :=
by
  -- Skipping the intermediate proof steps
  sorry

end two_mathematicians_contemporaries_l436_43664


namespace ordered_triples_eq_l436_43605

theorem ordered_triples_eq :
  ∃! (x y z : ℤ), x + y = 4 ∧ xy - z^2 = 3 ∧ (x = 2 ∧ y = 2 ∧ z = 0) :=
by
  -- Proof goes here
  sorry

end ordered_triples_eq_l436_43605


namespace dragon_cake_votes_l436_43671

theorem dragon_cake_votes (W U D : ℕ) (x : ℕ) 
  (hW : W = 7) 
  (hU : U = 3 * W) 
  (hD : D = W + x) 
  (hTotal : W + U + D = 60) 
  (hx : x = D - W) : 
  x = 25 := 
by
  sorry

end dragon_cake_votes_l436_43671


namespace total_profit_l436_43612

-- Definitions based on the conditions
def tom_investment : ℝ := 30000
def tom_duration : ℝ := 12
def jose_investment : ℝ := 45000
def jose_duration : ℝ := 10
def jose_share_profit : ℝ := 25000

-- Theorem statement
theorem total_profit (tom_investment tom_duration jose_investment jose_duration jose_share_profit : ℝ) :
  (jose_share_profit / (jose_investment * jose_duration / (tom_investment * tom_duration + jose_investment * jose_duration)) = 5 / 9) →
  ∃ P : ℝ, P = 45000 :=
by
  sorry

end total_profit_l436_43612


namespace GCF_30_90_75_l436_43607

theorem GCF_30_90_75 : Nat.gcd (Nat.gcd 30 90) 75 = 15 := by
  sorry

end GCF_30_90_75_l436_43607


namespace age_difference_l436_43673

-- Definitions based on the problem statement
def son_present_age : ℕ := 33

-- Represent the problem in terms of Lean
theorem age_difference (M : ℕ) (h : M + 2 = 2 * (son_present_age + 2)) : M - son_present_age = 35 :=
by
  sorry

end age_difference_l436_43673


namespace num_red_balls_l436_43677

theorem num_red_balls (x : ℕ) (h1 : 60 = 60) (h2 : (x : ℝ) / (x + 60) = 0.25) : x = 20 :=
sorry

end num_red_balls_l436_43677


namespace max_cookies_Andy_could_have_eaten_l436_43648

theorem max_cookies_Andy_could_have_eaten (cookies : ℕ) (Andy Alexa : ℕ) 
  (h1 : cookies = 24) 
  (h2 : Alexa = k * Andy) 
  (h3 : k > 0) 
  (h4 : Andy + Alexa = cookies) 
  : Andy ≤ 12 := 
sorry

end max_cookies_Andy_could_have_eaten_l436_43648


namespace middle_number_consecutive_sum_l436_43692

theorem middle_number_consecutive_sum (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : a + b + c = 30) : b = 10 :=
by
  sorry

end middle_number_consecutive_sum_l436_43692


namespace g_eq_g_inv_l436_43669

-- Define the function g
def g (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 3

-- Define the inverse function of g
noncomputable def g_inv (y : ℝ) : ℝ := (5 + Real.sqrt (1 + 8 * y)) / 4 -- simplified to handle the principal value

theorem g_eq_g_inv (x : ℝ) : g x = g_inv x → x = 1 := by
  -- Placeholder for proof
  sorry

end g_eq_g_inv_l436_43669


namespace discriminant_of_quadratic_l436_43660

theorem discriminant_of_quadratic :
  let a := (5 : ℚ)
  let b := (5 + 1/5 : ℚ)
  let c := (1/5 : ℚ)
  let Δ := b^2 - 4 * a * c
  Δ = 576 / 25 :=
by
  sorry

end discriminant_of_quadratic_l436_43660


namespace no_monochromatic_ap_11_l436_43641

open Function

theorem no_monochromatic_ap_11 :
  ∃ (coloring : ℕ → Fin 4), (∀ a r : ℕ, r > 0 → a + 10 * r ≤ 2014 → ∃ i j : ℕ, (i ≠ j) ∧ (a + i * r < 1 ∨ a + j * r > 2014 ∨ coloring (a + i * r) ≠ coloring (a + j * r))) :=
sorry

end no_monochromatic_ap_11_l436_43641


namespace sum_of_x_coords_f_eq_3_l436_43631

section
-- Define the piecewise linear function, splits into five segments
def f1 (x : ℝ) : ℝ := 2 * x + 6
def f2 (x : ℝ) : ℝ := -2 * x + 6
def f3 (x : ℝ) : ℝ := 2 * x + 2
def f4 (x : ℝ) : ℝ := -x + 2
def f5 (x : ℝ) : ℝ := 2 * x - 4

-- The sum of x-coordinates where f(x) = 3
noncomputable def x_coords_3_sum : ℝ := -1.5 + 0.5 + 3.5

-- Goal statement
theorem sum_of_x_coords_f_eq_3 : -1.5 + 0.5 + 3.5 = 2.5 := by
  sorry
end

end sum_of_x_coords_f_eq_3_l436_43631


namespace o_l436_43638

theorem o'hara_triple_example (a b x : ℕ) (h₁ : a = 49) (h₂ : b = 16) (h₃ : x = (Int.sqrt a).toNat + (Int.sqrt b).toNat) : x = 11 := 
by
  sorry

end o_l436_43638


namespace mul_powers_same_base_l436_43680

theorem mul_powers_same_base : 2^2 * 2^3 = 2^5 :=
by sorry

end mul_powers_same_base_l436_43680


namespace sequence_first_equals_last_four_l436_43633

theorem sequence_first_equals_last_four (n : ℕ) (S : ℕ → ℕ) (h_length : ∀ i < n, S i = 0 ∨ S i = 1)
  (h_condition : ∀ (i j : ℕ), 1 ≤ i ∧ i < j ∧ j ≤ n - 4 → 
    (S i = S j ∧ S (i + 1) = S (j + 1) ∧ S (i + 2) = S (j + 2) ∧ S (i + 3) = S (j + 3) ∧ S (i + 4) = S (j + 4)) → false) :
  S 1 = S (n - 3) ∧ S 2 = S (n - 2) ∧ S 3 = S (n - 1) ∧ S 4 = S n :=
sorry

end sequence_first_equals_last_four_l436_43633


namespace total_points_scored_l436_43663

-- Define the points scored by Sam and his friend
def points_scored_by_sam : ℕ := 75
def points_scored_by_friend : ℕ := 12

-- The main theorem stating the total points
theorem total_points_scored : points_scored_by_sam + points_scored_by_friend = 87 := by
  -- Proof goes here
  sorry

end total_points_scored_l436_43663


namespace polynomial_identity_sum_l436_43645

theorem polynomial_identity_sum (A B C D : ℤ) (h : (x + 3) * (4 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) : 
  A + B + C + D = 36 := 
by 
  sorry

end polynomial_identity_sum_l436_43645


namespace cos_alpha_minus_beta_l436_43678

theorem cos_alpha_minus_beta : 
  ∀ (α β : ℝ), 
  2 * Real.cos α - Real.cos β = 3 / 2 →
  2 * Real.sin α - Real.sin β = 2 →
  Real.cos (α - β) = -5 / 16 :=
by
  intros α β h1 h2
  sorry

end cos_alpha_minus_beta_l436_43678


namespace find_y_l436_43698

theorem find_y 
  (x y z : ℕ) 
  (h₁ : x + y + z = 25)
  (h₂ : x + y = 19) 
  (h₃ : y + z = 18) :
  y = 12 :=
by
  sorry

end find_y_l436_43698


namespace f_at_6_5_l436_43696

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f x = f (-x)

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x

def specific_values (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem f_at_6_5:
  (∀ x : ℝ, f (x + 2) = -1 / f x) →
  even_function f →
  specific_values f →
  f 6.5 = -0.5 :=
by
  sorry

end f_at_6_5_l436_43696


namespace cost_of_4_bags_of_ice_l436_43653

theorem cost_of_4_bags_of_ice (
  cost_per_2_bags : ℝ := 1.46
) 
  (h : cost_per_2_bags / 2 = 0.73)
  :
  4 * (cost_per_2_bags / 2) = 2.92 :=
by 
  sorry

end cost_of_4_bags_of_ice_l436_43653
