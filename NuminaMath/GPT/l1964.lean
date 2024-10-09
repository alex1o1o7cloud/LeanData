import Mathlib

namespace blue_eyes_count_l1964_196400

theorem blue_eyes_count (total_students students_both students_neither : ℕ)
  (ratio_blond_to_blue : ℕ → ℕ)
  (h_total : total_students = 40)
  (h_ratio : ratio_blond_to_blue 3 = 2)
  (h_both : students_both = 8)
  (h_neither : students_neither = 5) :
  ∃ y : ℕ, y = 18 :=
by
  sorry

end blue_eyes_count_l1964_196400


namespace middle_number_between_52_and_certain_number_l1964_196409

theorem middle_number_between_52_and_certain_number :
  ∃ n, n > 52 ∧ (∀ k, 52 ≤ k ∧ k ≤ n → ∃ l, k = 52 + l) ∧ (n = 52 + 16) :=
sorry

end middle_number_between_52_and_certain_number_l1964_196409


namespace minimum_f_value_l1964_196413

noncomputable def f (x : ℝ) : ℝ :=
   Real.sqrt (2 * x ^ 2 - 4 * x + 4) + 
   Real.sqrt (2 * x ^ 2 - 16 * x + (Real.log x / Real.log 2) ^ 2 - 2 * x * (Real.log x / Real.log 2) + 
              2 * (Real.log x / Real.log 2) + 50)

theorem minimum_f_value : ∀ x : ℝ, x > 0 → f x ≥ 7 ∧ f 2 = 7 :=
by
  sorry

end minimum_f_value_l1964_196413


namespace solve_for_y_l1964_196428

theorem solve_for_y (x y : ℝ) (h1 : x * y = 9) (h2 : x / y = 36) (h3 : x > 0) (h4 : y > 0) : y = 1/2 := 
by
  sorry

end solve_for_y_l1964_196428


namespace determine_w_arithmetic_seq_l1964_196465

theorem determine_w_arithmetic_seq (w : ℝ) (h : (w ≠ 0) ∧ 
  (1 / w - 1 / 2 = 1 / 2 - 1 / 3) ∧ (1 / 2 - 1 / 3 = 1 / 3 - 1 / 6)) :
  w = 3 / 2 := 
sorry

end determine_w_arithmetic_seq_l1964_196465


namespace vertex_of_parabola_l1964_196479

theorem vertex_of_parabola :
  ∃ (x y : ℝ), y^2 - 8*x + 6*y + 17 = 0 ∧ (x, y) = (1, -3) :=
by
  use 1, -3
  sorry

end vertex_of_parabola_l1964_196479


namespace total_participating_students_l1964_196496

-- Define the given conditions
def field_events_participants : ℕ := 15
def track_events_participants : ℕ := 13
def both_events_participants : ℕ := 5

-- Define the total number of students calculation
def total_students_participating : ℕ :=
  (field_events_participants - both_events_participants) + 
  (track_events_participants - both_events_participants) + 
  both_events_participants

-- State the theorem that needs to be proved
theorem total_participating_students : total_students_participating = 23 := by
  sorry

end total_participating_students_l1964_196496


namespace angles_does_not_exist_l1964_196431

theorem angles_does_not_exist (a1 a2 a3 : ℝ) 
  (h1 : a1 + a2 = 90) 
  (h2 : a2 + a3 = 180) 
  (h3 : a3 = 18) : False :=
by
  sorry

end angles_does_not_exist_l1964_196431


namespace silver_coin_value_l1964_196402

--- Definitions from the conditions
def total_value_hoard (value_silver : ℕ) := 100 * 3 * value_silver + 60 * value_silver + 33

--- Statement of the theorem to prove
theorem silver_coin_value (x : ℕ) (h : total_value_hoard x = 2913) : x = 8 :=
by {
  sorry
}

end silver_coin_value_l1964_196402


namespace factor_expression_l1964_196468

variable (b : ℝ)

theorem factor_expression : 221 * b * b + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end factor_expression_l1964_196468


namespace vanessa_video_files_initial_l1964_196410

theorem vanessa_video_files_initial (m v r d t : ℕ) (h1 : m = 13) (h2 : r = 33) (h3 : d = 10) (h4 : t = r + d) (h5 : t = m + v) : v = 30 :=
by
  sorry

end vanessa_video_files_initial_l1964_196410


namespace option_A_option_B_option_C_option_D_l1964_196483

-- Option A
theorem option_A (x : ℝ) (h : x^2 - 2*x + 1 = 0) : 
  (x-1)^2 + x*(x-4) + (x-2)*(x+2) ≠ 0 := 
sorry

-- Option B
theorem option_B (x : ℝ) (h : x^2 - 3*x + 1 = 0) : 
  x^3 + (1/x)^3 - 3 = 15 := 
sorry

-- Option C
theorem option_C (x : ℝ) (a b c : ℝ) (h_a : a = 1 / 20 * x + 20) (h_b : b = 1 / 20 * x + 19) (h_c : c = 1 / 20 * x + 21) : 
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 3 := 
sorry

-- Option D
theorem option_D (x m n : ℝ) (h : 2*x^2 - 8*x + 7 = 0) (h_roots : m + n = 4 ∧ m * n = 7/2) : 
  Real.sqrt (m^2 + n^2) = 3 := 
sorry

end option_A_option_B_option_C_option_D_l1964_196483


namespace smallest_number_l1964_196414

theorem smallest_number (a b c d e: ℕ) (h1: a = 5) (h2: b = 8) (h3: c = 1) (h4: d = 2) (h5: e = 6) :
  min (min (min (min a b) c) d) e = 1 :=
by
  -- Proof skipped using sorry
  sorry

end smallest_number_l1964_196414


namespace number_of_buyers_l1964_196426

theorem number_of_buyers 
  (today yesterday day_before : ℕ) 
  (h1 : today = yesterday + 40) 
  (h2 : yesterday = day_before / 2) 
  (h3 : day_before + yesterday + today = 140) : 
  day_before = 67 :=
by
  -- skip the proof
  sorry

end number_of_buyers_l1964_196426


namespace line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l1964_196436

-- Define the point (M) and the properties of the line
def point_M : ℝ × ℝ := (-1, 3)

def parallel_to_y_axis (line : ℝ × ℝ → Prop) : Prop :=
  ∃ b : ℝ, ∀ y : ℝ, line (b, y)

-- Statement we need to prove
theorem line_through_point_parallel_to_y_axis_eq_x_eq_neg1 :
  (∃ line : ℝ × ℝ → Prop, line point_M ∧ parallel_to_y_axis line) → ∀ p : ℝ × ℝ, (p.1 = -1 ↔ (∃ line : ℝ × ℝ → Prop, line p ∧ line point_M ∧ parallel_to_y_axis line)) :=
by
  sorry

end line_through_point_parallel_to_y_axis_eq_x_eq_neg1_l1964_196436


namespace find_N_value_l1964_196493

-- Definitions based on given conditions
def M (n : ℕ) : ℕ := 4^n
def N (n : ℕ) : ℕ := 2^n
def condition (n : ℕ) : Prop := M n - N n = 240

-- Theorem statement to prove N == 16 given the conditions
theorem find_N_value (n : ℕ) (h : condition n) : N n = 16 := 
  sorry

end find_N_value_l1964_196493


namespace factor_difference_of_squares_l1964_196482

theorem factor_difference_of_squares (t : ℤ) : t^2 - 64 = (t - 8) * (t + 8) :=
by {
  sorry
}

end factor_difference_of_squares_l1964_196482


namespace minimum_value_of_f_l1964_196420

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / Real.sqrt (x^2 + 5)

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 6 :=
by 
  sorry

end minimum_value_of_f_l1964_196420


namespace cost_of_45_lilies_l1964_196437

-- Definitions of the given conditions
def cost_per_lily := 30 / 18
def lilies_18_bouquet_cost := 30
def number_of_lilies_in_bouquet := 45

-- Theorem stating the mathematical proof problem
theorem cost_of_45_lilies : cost_per_lily * number_of_lilies_in_bouquet = 75 := by
  -- The proof is omitted
  sorry

end cost_of_45_lilies_l1964_196437


namespace range_of_2m_plus_n_l1964_196498

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x / Real.log 3)

theorem range_of_2m_plus_n {m n : ℝ} (hmn : 0 < m ∧ m < n) (heq : f m = f n) :
  ∃ y, y ∈ Set.Ici (2 * Real.sqrt 2) ∧ (2 * m + n = y) :=
sorry

end range_of_2m_plus_n_l1964_196498


namespace no_integers_p_and_q_l1964_196448

theorem no_integers_p_and_q (p q : ℤ) : ¬(∀ x : ℤ, 3 ∣ (x^2 + p * x + q)) :=
by
  sorry

end no_integers_p_and_q_l1964_196448


namespace unique_representation_l1964_196459

theorem unique_representation (n : ℕ) (h_pos : 0 < n) : 
  ∃! (a b : ℚ), a = 1 / n ∧ b = 1 / (n + 1) ∧ (a + b = (2 * n + 1) / (n * (n + 1))) :=
by
  sorry

end unique_representation_l1964_196459


namespace calc_625_to_4_div_5_l1964_196481

theorem calc_625_to_4_div_5 :
  (625 : ℝ)^(4/5) = 238 :=
sorry

end calc_625_to_4_div_5_l1964_196481


namespace smallest_whole_number_larger_than_perimeter_l1964_196457

theorem smallest_whole_number_larger_than_perimeter (c : ℝ) (h1 : 13 < c) (h2 : c < 25) : 50 = Nat.ceil (6 + 19 + c) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l1964_196457


namespace cube_volume_surface_area_l1964_196443

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ s : ℝ, s^3 = 8 * x ∧ 6 * s^2 = 2 * x) → x = 0 :=
by
  sorry

end cube_volume_surface_area_l1964_196443


namespace negation_of_conditional_l1964_196433

-- Define the propositions
def P (x : ℝ) : Prop := x > 2015
def Q (x : ℝ) : Prop := x > 0

-- Negate the propositions
def notP (x : ℝ) : Prop := x <= 2015
def notQ (x : ℝ) : Prop := x <= 0

-- Theorem: Negation of the conditional statement
theorem negation_of_conditional (x : ℝ) : ¬ (P x → Q x) ↔ (notP x → notQ x) :=
by
  sorry

end negation_of_conditional_l1964_196433


namespace relationship_y1_y2_l1964_196434

variables {x1 x2 : ℝ}

noncomputable def f (x : ℝ) : ℝ := -3 * x ^ 2 + 6 * x - 5

theorem relationship_y1_y2 (hx1 : 0 ≤ x1) (hx1_lt : x1 < 1) (hx2 : 2 ≤ x2) (hx2_lt : x2 < 3) :
  f x1 ≥ f x2 :=
sorry

end relationship_y1_y2_l1964_196434


namespace acid_volume_16_liters_l1964_196454

theorem acid_volume_16_liters (V A_0 B_0 A_1 B_1 : ℝ) 
  (h_initial_ratio : 4 * B_0 = A_0)
  (h_initial_volume : A_0 + B_0 = V)
  (h_remove_mixture : 10 * A_0 / V = A_1)
  (h_remove_mixture_base : 10 * B_0 / V = B_1)
  (h_new_A : A_1 = A_0 - 8)
  (h_new_B : B_1 = B_0 - 2 + 10)
  (h_new_ratio : 2 * B_1 = 3 * A_1) :
  A_0 = 16 :=
by {
  -- Here we will have the proof steps, which are omitted.
  sorry
}

end acid_volume_16_liters_l1964_196454


namespace evaluate_expression_l1964_196473

theorem evaluate_expression :
  54 + 98 / 14 + 23 * 17 - 200 - 312 / 6 = 200 :=
by
  sorry

end evaluate_expression_l1964_196473


namespace rectangle_area_ratio_l1964_196461

theorem rectangle_area_ratio (s x y : ℝ) (h_square : s > 0)
    (h_side_ae : x > 0) (h_side_ag : y > 0)
    (h_ratio_area : x * y = (1 / 4) * s^2) :
    ∃ (r : ℝ), r > 0 ∧ r = x / y := 
sorry

end rectangle_area_ratio_l1964_196461


namespace train_stops_15_min_per_hour_l1964_196407

/-
Without stoppages, a train travels a certain distance with an average speed of 80 km/h,
and with stoppages, it covers the same distance with an average speed of 60 km/h.
Prove that the train stops for 15 minutes per hour.
-/
theorem train_stops_15_min_per_hour (D : ℝ) (h1 : 0 < D) :
  let T_no_stop := D / 80
  let T_stop := D / 60
  let T_lost := T_stop - T_no_stop
  let mins_per_hour := T_lost * 60
  mins_per_hour = 15 := by
  sorry

end train_stops_15_min_per_hour_l1964_196407


namespace system1_solution_system2_solution_l1964_196490

theorem system1_solution :
  ∃ (x y : ℤ), (4 * x - y = 1) ∧ (y = 2 * x + 3) ∧ (x = 2) ∧ (y = 7) :=
by
  sorry

theorem system2_solution :
  ∃ (x y : ℤ), (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ∧ (x = 5) ∧ (y = 5) :=
by
  sorry

end system1_solution_system2_solution_l1964_196490


namespace evaluate_expression_l1964_196411

theorem evaluate_expression : - (16 / 4 * 7 + 25 - 2 * 7) = -39 :=
by sorry

end evaluate_expression_l1964_196411


namespace total_cost_l1964_196445

-- Define the cost of a neutral pen and a pencil
variables (x y : ℝ)

-- The total cost of buying 5 neutral pens and 3 pencils
theorem total_cost (x y : ℝ) : 5 * x + 3 * y = 5 * x + 3 * y :=
by
  -- The statement is self-evident, hence can be written directly
  sorry

end total_cost_l1964_196445


namespace sum_of_primes_less_than_20_eq_77_l1964_196467

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ k : ℕ, k ∣ n → k = 1 ∨ k = n

def primes_less_than_20 : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19]

def sum_primes_less_than_20 := List.sum primes_less_than_20

theorem sum_of_primes_less_than_20_eq_77 :
  sum_primes_less_than_20 = 77 :=
by
  sorry

end sum_of_primes_less_than_20_eq_77_l1964_196467


namespace intersect_at_2d_l1964_196406

def g (x : ℝ) (c : ℝ) : ℝ := 4 * x + c

theorem intersect_at_2d (c d : ℤ) (h₁ : d = 8 + c) (h₂ : 2 = g d c) : d = 2 :=
by
  sorry

end intersect_at_2d_l1964_196406


namespace prime_difference_condition_l1964_196401

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_difference_condition :
  ∃ (x y : ℕ), is_prime x ∧ is_prime y ∧ 4 < x ∧ x < 18 ∧ 4 < y ∧ y < 18 ∧ x ≠ y ∧ (x * y - (x + y)) = 119 :=
by
  sorry

end prime_difference_condition_l1964_196401


namespace least_integer_greater_than_sqrt_500_l1964_196486

theorem least_integer_greater_than_sqrt_500 (x: ℕ) (h1: 22^2 = 484) (h2: 23^2 = 529) (h3: 484 < 500 ∧ 500 < 529) : x = 23 :=
  sorry

end least_integer_greater_than_sqrt_500_l1964_196486


namespace solve_for_k_l1964_196439

theorem solve_for_k (a k : ℝ) (h : a ^ 10 / (a ^ k) ^ 4 = a ^ 2) : k = 2 :=
by
  sorry

end solve_for_k_l1964_196439


namespace jill_water_filled_jars_l1964_196469

variable (gallons : ℕ) (quart_halfGallon_gallon : ℕ)
variable (h_eq : gallons = 14)
variable (h_eq_n : quart_halfGallon_gallon = 3 * 8)
variable (h_total : quart_halfGallon_gallon = 24)

theorem jill_water_filled_jars :
  3 * (gallons * 4 / 7) = 24 :=
sorry

end jill_water_filled_jars_l1964_196469


namespace game_cost_l1964_196488

theorem game_cost
    (initial_amount : ℕ)
    (cost_per_toy : ℕ)
    (num_toys : ℕ)
    (remaining_amount := initial_amount - cost_per_toy * num_toys)
    (cost_of_game := initial_amount - remaining_amount)
    (h1 : initial_amount = 57)
    (h2 : cost_per_toy = 6)
    (h3 : num_toys = 5) :
  cost_of_game = 27 :=
by
  sorry

end game_cost_l1964_196488


namespace undefined_expr_iff_l1964_196487

theorem undefined_expr_iff (a : ℝ) : (∃ x, x = (a^2 - 9) ∧ x = 0) ↔ (a = -3 ∨ a = 3) :=
by
  sorry

end undefined_expr_iff_l1964_196487


namespace square_of_square_root_l1964_196444

theorem square_of_square_root (x : ℝ) (hx : (Real.sqrt x)^2 = 49) : x = 49 :=
by 
  sorry

end square_of_square_root_l1964_196444


namespace g_positive_l1964_196458

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 1 / 2 + 1 / (2^x - 1) else 0

noncomputable def g (x : ℝ) : ℝ :=
  x^3 * f x

theorem g_positive (x : ℝ) (hx : x ≠ 0) : g x > 0 :=
  sorry -- Proof to be filled in

end g_positive_l1964_196458


namespace compute_expression_l1964_196440

theorem compute_expression : (3 + 6 + 9)^3 + (3^3 + 6^3 + 9^3) = 6804 := by
  sorry

end compute_expression_l1964_196440


namespace university_diploma_percentage_l1964_196484

theorem university_diploma_percentage
  (A : ℝ) (B : ℝ) (C : ℝ)
  (hA : A = 0.40)
  (hB : B = 0.10)
  (hC : C = 0.15) :
  A - B + C * (1 - A) = 0.39 := 
sorry

end university_diploma_percentage_l1964_196484


namespace marbles_in_bag_l1964_196463

theorem marbles_in_bag (r b : ℕ) : 
  (r - 2) * 10 = (r + b - 2) →
  (r * 6 = (r + b - 3)) →
  ((r - 2) * 8 = (r + b - 4)) →
  r + b = 42 :=
by
  intros h1 h2 h3
  sorry

end marbles_in_bag_l1964_196463


namespace increase_in_average_weight_l1964_196419

theorem increase_in_average_weight 
    (A : ℝ) 
    (weight_left : ℝ)
    (weight_new : ℝ)
    (h_weight_left : weight_left = 67)
    (h_weight_new : weight_new = 87) : 
    ((8 * A - weight_left + weight_new) / 8 - A) = 2.5 := 
by
  sorry

end increase_in_average_weight_l1964_196419


namespace number_of_paths_l1964_196430

theorem number_of_paths (n : ℕ) (h1 : n > 3) : 
  (2 * (8 * n^3 - 48 * n^2 + 88 * n - 48) + (4 * n^2 - 12 * n + 8) + (2 * n - 2)) = 16 * n^3 - 92 * n^2 + 166 * n - 90 :=
by
  sorry

end number_of_paths_l1964_196430


namespace average_attendance_percentage_l1964_196477

theorem average_attendance_percentage :
  let total_laborers := 300
  let day1_present := 150
  let day2_present := 225
  let day3_present := 180
  let day1_percentage := (day1_present / total_laborers) * 100
  let day2_percentage := (day2_present / total_laborers) * 100
  let day3_percentage := (day3_present / total_laborers) * 100
  let average_percentage := (day1_percentage + day2_percentage + day3_percentage) / 3
  average_percentage = 61.7 := by
  sorry

end average_attendance_percentage_l1964_196477


namespace find_second_number_l1964_196492

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number_l1964_196492


namespace auntie_em_parking_probability_l1964_196412

theorem auntie_em_parking_probability :
  let total_spaces := 20
  let cars := 15
  let empty_spaces := total_spaces - cars
  let possible_configurations := Nat.choose total_spaces cars
  let unfavourable_configurations := Nat.choose (empty_spaces - 8 + 5) (empty_spaces - 8)
  let favourable_probability := 1 - ((unfavourable_configurations : ℚ) / (possible_configurations : ℚ))
  (favourable_probability = 1839 / 1938) :=
by
  -- sorry to skip the actual proof
  sorry

end auntie_em_parking_probability_l1964_196412


namespace quadratic_eq_two_distinct_real_roots_l1964_196470

theorem quadratic_eq_two_distinct_real_roots :
    ∃ x y : ℝ, x ≠ y ∧ (x^2 + x - 1 = 0) ∧ (y^2 + y - 1 = 0) :=
by
    sorry

end quadratic_eq_two_distinct_real_roots_l1964_196470


namespace infinite_geometric_series_sum_l1964_196455

-- Definition of the infinite geometric series with given first term and common ratio
def infinite_geometric_series (a : ℚ) (r : ℚ) : ℚ := a / (1 - r)

-- Problem statement
theorem infinite_geometric_series_sum :
  infinite_geometric_series (5 / 3) (-2 / 9) = 15 / 11 :=
sorry

end infinite_geometric_series_sum_l1964_196455


namespace domain_range_sum_l1964_196460

theorem domain_range_sum (m n : ℝ) 
  (h1 : ∀ x, m ≤ x ∧ x ≤ n → 3 * m ≤ -x ^ 2 + 2 * x ∧ -x ^ 2 + 2 * x ≤ 3 * n)
  (h2 : -m ^ 2 + 2 * m = 3 * m)
  (h3 : -n ^ 2 + 2 * n = 3 * n) :
  m = -1 ∧ n = 0 ∧ m + n = -1 := 
by 
  sorry

end domain_range_sum_l1964_196460


namespace gifted_subscribers_l1964_196499

theorem gifted_subscribers (initial_subs : ℕ) (revenue_per_sub : ℕ) (total_revenue : ℕ) (h1 : initial_subs = 150) (h2 : revenue_per_sub = 9) (h3 : total_revenue = 1800) :
  total_revenue / revenue_per_sub - initial_subs = 50 :=
by
  sorry

end gifted_subscribers_l1964_196499


namespace first_place_points_l1964_196450

-- Definitions for the conditions
def num_teams : Nat := 4
def points_win : Nat := 2
def points_draw : Nat := 1
def points_loss : Nat := 0

def games_played (n : Nat) : Nat :=
  let pairs := n * (n - 1) / 2  -- Binomial coefficient C(n, 2)
  2 * pairs  -- Each pair plays twice

def total_points_distributed (n : Nat) (points_per_game : Nat) : Nat :=
  (games_played n) * points_per_game

def last_place_points : Nat := 5

-- The theorem to prove
theorem first_place_points : ∃ a b c : Nat, a + b + c = total_points_distributed num_teams points_win - last_place_points ∧ (a = 7 ∨ b = 7 ∨ c = 7) :=
by
  sorry

end first_place_points_l1964_196450


namespace adam_teaches_650_students_in_10_years_l1964_196495

noncomputable def students_in_n_years (n : ℕ) : ℕ :=
  if n = 1 then 40
  else if n = 2 then 60
  else if n = 3 then 70
  else if n <= 10 then 70
  else 0 -- beyond the scope of this problem

theorem adam_teaches_650_students_in_10_years :
  (students_in_n_years 1 + students_in_n_years 2 + students_in_n_years 3 +
   students_in_n_years 4 + students_in_n_years 5 + students_in_n_years 6 +
   students_in_n_years 7 + students_in_n_years 8 + students_in_n_years 9 +
   students_in_n_years 10) = 650 :=
by
  sorry

end adam_teaches_650_students_in_10_years_l1964_196495


namespace perfect_square_solution_l1964_196466

theorem perfect_square_solution (n : ℕ) : ∃ a : ℕ, n * 2^(n+1) + 1 = a^2 ↔ n = 0 ∨ n = 3 := by
  sorry

end perfect_square_solution_l1964_196466


namespace problem_statement_l1964_196475

noncomputable def original_expression (x : ℕ) : ℚ :=
(1 - 1 / (x - 1)) / ((x^2 - 4*x + 4) / (x - 1))

theorem problem_statement (x : ℕ) (hx1 : 3 - x ≥ 0) (hx2 : x ≠ 2) (hx3 : x ≠ 1) :
  original_expression 3 = 1 :=
by
  sorry

end problem_statement_l1964_196475


namespace question1_question2_l1964_196446

def f (x : ℝ) : ℝ := |2 * x - 1|

theorem question1 (m : ℝ) (h1 : m > 0) 
(h2 : ∀ (x : ℝ), f (x + 1/2) ≤ 2 * m + 1 ↔ x ∈ [-2, 2]) : m = 3 / 2 := 
sorry

theorem question2 (x y : ℝ) : f x ≤ 2^y + 4 / 2^y + |2 * x + 3| := 
sorry

end question1_question2_l1964_196446


namespace final_amounts_calculation_l1964_196408

noncomputable def article_A_original_cost : ℚ := 200
noncomputable def article_B_original_cost : ℚ := 300
noncomputable def article_C_original_cost : ℚ := 400
noncomputable def exchange_rate_euro_to_usd : ℚ := 1.10
noncomputable def exchange_rate_gbp_to_usd : ℚ := 1.30
noncomputable def discount_A : ℚ := 0.50
noncomputable def discount_B : ℚ := 0.30
noncomputable def discount_C : ℚ := 0.40
noncomputable def sales_tax_rate : ℚ := 0.05
noncomputable def reward_points : ℚ := 100
noncomputable def reward_point_value : ℚ := 0.05

theorem final_amounts_calculation :
  let discounted_A := article_A_original_cost * discount_A
  let final_A := (article_A_original_cost - discounted_A) * exchange_rate_euro_to_usd
  let discounted_B := article_B_original_cost * discount_B
  let final_B := (article_B_original_cost - discounted_B) * exchange_rate_gbp_to_usd
  let discounted_C := article_C_original_cost * discount_C
  let final_C := article_C_original_cost - discounted_C
  let total_discounted_cost_usd := final_A + final_B + final_C
  let sales_tax := total_discounted_cost_usd * sales_tax_rate
  let reward := reward_points * reward_point_value
  let final_amount_usd := total_discounted_cost_usd + sales_tax - reward
  let final_amount_euro := final_amount_usd / exchange_rate_euro_to_usd
  final_amount_usd = 649.15 ∧ final_amount_euro = 590.14 :=
by
  sorry

end final_amounts_calculation_l1964_196408


namespace matrix_expression_l1964_196447
open Matrix

variables {n : Type*} [Fintype n] [DecidableEq n]
variables (B : Matrix n n ℝ) (I : Matrix n n ℝ)

noncomputable def B_inverse := B⁻¹

-- Condition 1: B is a matrix with an inverse
variable [Invertible B]

-- Condition 2: (B - 3*I) * (B - 5*I) = 0
variable (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0)

-- Theorem to prove
theorem matrix_expression (B: Matrix n n ℝ) [Invertible B] 
  (H : (B - (3 : ℝ) • I) * (B - (5 : ℝ) • I) = 0) : 
  B + 10 * (B_inverse B) = (160 / 15 : ℝ) • I := 
sorry

end matrix_expression_l1964_196447


namespace domain_of_c_is_all_reals_l1964_196451

theorem domain_of_c_is_all_reals (k : ℝ) :
  (∀ x : ℝ, -3 * x^2 - 4 * x + k ≠ 0) ↔ k < -4 / 3 := 
by
  sorry

end domain_of_c_is_all_reals_l1964_196451


namespace solve_for_x_l1964_196476

theorem solve_for_x (x : ℝ) : (x - 20) / 3 = (4 - 3 * x) / 4 → x = 7.08 := by
  sorry

end solve_for_x_l1964_196476


namespace relationship_between_a_and_b_l1964_196416

open Real

theorem relationship_between_a_and_b
  (a b x : ℝ)
  (h1 : a ≠ 1)
  (h2 : b ≠ 1)
  (h3 : 4 * (log x / log a)^3 + 5 * (log x / log b)^3 = 7 * (log x)^3) :
  b = a ^ (3 / 5)^(1 / 3) := 
sorry

end relationship_between_a_and_b_l1964_196416


namespace highest_probability_white_ball_l1964_196429

theorem highest_probability_white_ball :
  let red_balls := 2
  let black_balls := 3
  let white_balls := 4
  let total_balls := red_balls + black_balls + white_balls
  let prob_red := red_balls / total_balls
  let prob_black := black_balls / total_balls
  let prob_white := white_balls / total_balls
  prob_white > prob_black ∧ prob_black > prob_red :=
by
  sorry

end highest_probability_white_ball_l1964_196429


namespace find_a_l1964_196404

theorem find_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x^2 - 2*a*x - 8*(a^2) < 0) (h3 : x2 - x1 = 15) : a = 5 / 2 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end find_a_l1964_196404


namespace simplify_expression_l1964_196435

variable (y : ℝ)

theorem simplify_expression : 3 * y + 4 * y^2 - 2 - (7 - 3 * y - 4 * y^2) = 8 * y^2 + 6 * y - 9 := 
  by
  sorry

end simplify_expression_l1964_196435


namespace largest_four_digit_number_with_digits_sum_25_l1964_196464

def four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def digits_sum_to (n : ℕ) (s : ℕ) : Prop :=
  (n / 1000 + (n % 1000) / 100 + (n % 100) / 10 + (n % 10) = s)

theorem largest_four_digit_number_with_digits_sum_25 :
  ∃ n, four_digit n ∧ digits_sum_to n 25 ∧ ∀ m, four_digit m → digits_sum_to m 25 → m ≤ n :=
sorry

end largest_four_digit_number_with_digits_sum_25_l1964_196464


namespace minimum_value_of_a_l1964_196427

theorem minimum_value_of_a (a b : ℕ) (h₁ : b - a = 2013) 
(h₂ : ∃ x : ℕ, x^2 - a * x + b = 0) : a = 93 :=
sorry

end minimum_value_of_a_l1964_196427


namespace find_divisor_l1964_196472

theorem find_divisor (N D k : ℤ) (h1 : N = 5 * D) (h2 : N % 11 = 2) : D = 7 :=
by
  sorry

end find_divisor_l1964_196472


namespace num_ways_distinct_letters_l1964_196452

def letters : List String := ["A₁", "A₂", "A₃", "N₁", "N₂", "N₃", "B₁", "B₂"]

theorem num_ways_distinct_letters : (letters.permutations.length = 40320) := by
  sorry

end num_ways_distinct_letters_l1964_196452


namespace integer_points_between_A_B_l1964_196438

/-- 
Prove that the number of integer coordinate points strictly between 
A(2, 3) and B(50, 80) on the line passing through A and B is c.
-/
theorem integer_points_between_A_B 
  (A B : ℤ × ℤ) (hA : A = (2, 3)) (hB : B = (50, 80)) 
  (c : ℕ) :
  ∃ (n : ℕ), n = c ∧ ∀ (x y : ℤ), (A.1 < x ∧ x < B.1) → (A.2 < y ∧ y < B.2) → 
              (y = ((A.2 - B.2) / (A.1 - B.1) * x + 3 - (A.2 - B.2) / (A.1 - B.1) * 2)) :=
by {
  sorry
}

end integer_points_between_A_B_l1964_196438


namespace complex_division_product_l1964_196424

theorem complex_division_product
  (i : ℂ)
  (h_exp: i * i = -1)
  (a b : ℝ)
  (h_div: (1 + 7 * i) / (2 - i) = a + b * i)
  : a * b = -3 := 
sorry

end complex_division_product_l1964_196424


namespace strategy2_is_better_final_cost_strategy2_correct_l1964_196489

def initial_cost : ℝ := 12000

def strategy1_discount : ℝ := 
  let after_first_discount := initial_cost * 0.70
  let after_second_discount := after_first_discount * 0.85
  let after_third_discount := after_second_discount * 0.95
  after_third_discount

def strategy2_discount : ℝ := 
  let after_first_discount := initial_cost * 0.55
  let after_second_discount := after_first_discount * 0.90
  let after_third_discount := after_second_discount * 0.90
  let final_cost := after_third_discount + 150
  final_cost

theorem strategy2_is_better : strategy2_discount < strategy1_discount :=
by {
  sorry -- proof goes here
}

theorem final_cost_strategy2_correct : strategy2_discount = 5496 :=
by {
  sorry -- proof goes here
}

end strategy2_is_better_final_cost_strategy2_correct_l1964_196489


namespace part1_proof_part2_proof_l1964_196485

-- Given conditions
variables (a b x : ℝ)
def y (a b x : ℝ) := a*x^2 + (b-2)*x + 3

-- The initial conditions
noncomputable def conditions := 
  (∀ x, -1 < x ∧ x < 3 → y a b x > 0) ∧
  (∃ a b, a > 0 ∧ b > 0 ∧ y a b 1 = 2)

-- Part (1): Prove that the solution set of y >= 4 is {1}
theorem part1_proof :
  conditions a b →
  {x | y a b x ≥ 4} = {1} :=
  by
    sorry

-- Part (2): Prove that the minimum value of (1/a + 4/b) is 9
theorem part2_proof :
  conditions a b →
  ∃ x, x = 1/a + 4/b ∧ x = 9 :=
  by
    sorry

end part1_proof_part2_proof_l1964_196485


namespace terminating_decimal_representation_l1964_196462

theorem terminating_decimal_representation : 
  (67 / (2^3 * 5^4) : ℝ) = 0.0134 :=
    sorry

end terminating_decimal_representation_l1964_196462


namespace number_of_k_for_lcm_l1964_196478

theorem number_of_k_for_lcm (a b : ℕ) :
  (∀ a b, k = 2^a * 3^b) → 
  (∀ (a : ℕ), 0 ≤ a ∧ a ≤ 24) →
  (∃ b, b = 12) →
  (∀ k, k = 2^a * 3^b) →
  (Nat.lcm (Nat.lcm (6^6) (8^8)) k = 12^12) :=
sorry

end number_of_k_for_lcm_l1964_196478


namespace decimal_to_fraction_l1964_196449

theorem decimal_to_fraction {a b c : ℚ} (H1 : a = 2.75) (H2 : b = 11) (H3 : c = 4) : (a = b / c) :=
by {
  sorry
}

end decimal_to_fraction_l1964_196449


namespace N_is_perfect_square_l1964_196471

def N (n : ℕ) : ℕ :=
  (10^(2*n+1) - 1) / 9 * 10 + 
  2 * (10^(n+1) - 1) / 9 + 25

theorem N_is_perfect_square (n : ℕ) : ∃ k, k^2 = N n :=
  sorry

end N_is_perfect_square_l1964_196471


namespace ordered_pairs_sum_reciprocal_l1964_196456

theorem ordered_pairs_sum_reciprocal (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (1 / a + 1 / b : ℚ) = 1 / 6) → ∃ n : ℕ, n = 9 :=
by
  sorry

end ordered_pairs_sum_reciprocal_l1964_196456


namespace like_terms_m_n_sum_l1964_196480

theorem like_terms_m_n_sum :
  ∃ (m n : ℕ), (2 : ℤ) * x ^ (3 * n) * y ^ (m + 4) = (-3 : ℤ) * x ^ 9 * y ^ (2 * n) ∧ m + n = 5 :=
by 
  sorry

end like_terms_m_n_sum_l1964_196480


namespace students_later_than_Yoongi_l1964_196442

theorem students_later_than_Yoongi (total_students finished_before_Yoongi : ℕ) (h1 : total_students = 20) (h2 : finished_before_Yoongi = 11) :
  total_students - (finished_before_Yoongi + 1) = 8 :=
by {
  -- Proof is omitted as it's not required.
  sorry
}

end students_later_than_Yoongi_l1964_196442


namespace chromium_percentage_in_second_alloy_l1964_196474

theorem chromium_percentage_in_second_alloy (x : ℝ) :
  (15 * 0.12) + (35 * (x / 100)) = 50 * 0.106 → x = 10 :=
by
  sorry

end chromium_percentage_in_second_alloy_l1964_196474


namespace alpha_beta_working_together_time_l1964_196417

theorem alpha_beta_working_together_time
  (A B C : ℝ)
  (h : ℝ)
  (hA : A = B + 5)
  (work_together_A : A > 0)
  (work_together_B : B > 0)
  (work_together_C : C > 0)
  (combined_work : 1/A + 1/B + 1/C = 1/(A - 6))
  (combined_work2 : 1/A + 1/B + 1/C = 1/(B - 1))
  (time_gamma : 1/A + 1/B + 1/C = 2/C) :
  h = 4/3 :=
sorry

end alpha_beta_working_together_time_l1964_196417


namespace mean_equality_l1964_196405

theorem mean_equality (z : ℝ) :
  (8 + 15 + 24) / 3 = (16 + z) / 2 → z = 15.34 :=
by
  intro h
  sorry

end mean_equality_l1964_196405


namespace price_reduction_l1964_196494

theorem price_reduction (P : ℝ) : 
  let first_day_reduction := 0.91 * P
  let second_day_reduction := 0.90 * first_day_reduction
  second_day_reduction = 0.819 * P :=
by 
  sorry

end price_reduction_l1964_196494


namespace tan_alpha_minus_pi_over_4_l1964_196422

theorem tan_alpha_minus_pi_over_4 (α : ℝ) (h1 : 0 < α ∧ α < π)
  (h2 : Real.sin α = 3 / 5) : Real.tan (α - π / 4) = -1 / 7 ∨ Real.tan (α - π / 4) = -7 := 
sorry

end tan_alpha_minus_pi_over_4_l1964_196422


namespace find_n_series_sum_l1964_196453

theorem find_n_series_sum 
  (first_term_I : ℝ) (second_term_I : ℝ) (first_term_II : ℝ) (second_term_II : ℝ) (sum_multiplier : ℝ) (n : ℝ)
  (h_I_first_term : first_term_I = 12)
  (h_I_second_term : second_term_I = 4)
  (h_II_first_term : first_term_II = 12)
  (h_II_second_term : second_term_II = 4 + n)
  (h_sum_multiplier : sum_multiplier = 5) :
  n = 152 :=
by
  sorry

end find_n_series_sum_l1964_196453


namespace no_common_points_range_a_l1964_196441

theorem no_common_points_range_a (a k : ℝ) (hl : ∃ k, ∀ x y : ℝ, k * x - y - k + 2 = 0) :
  (∀ x y : ℝ, x^2 + 2 * a * x + y^2 - a + 2 ≠ 0) → (-7 < a ∧ a < -2) ∨ (1 < a) := by
  sorry

end no_common_points_range_a_l1964_196441


namespace maxOccursAt2_l1964_196432

def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

theorem maxOccursAt2 {m : ℝ} :
  (∀ x, 0 ≤ x ∧ x ≤ m → f x ≤ f m) ∧ 0 ≤ m ∧ m ≤ 2 → (0 < m ∧ m ≤ 2) :=
sorry

end maxOccursAt2_l1964_196432


namespace limit_hours_overtime_l1964_196425

theorem limit_hours_overtime (R O : ℝ) (earnings total_hours : ℕ) (L : ℕ) 
    (hR : R = 16)
    (hO : O = R + 0.75 * R)
    (h_earnings : earnings = 864)
    (h_total_hours : total_hours = 48)
    (calc_earnings : earnings = L * R + (total_hours - L) * O) :
    L = 40 := by
  sorry

end limit_hours_overtime_l1964_196425


namespace g_func_eq_l1964_196497

theorem g_func_eq (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → g (x / y) = y * g x)
  (h2 : g 50 = 10) :
  g 25 = 20 :=
sorry

end g_func_eq_l1964_196497


namespace cricket_average_increase_l1964_196415

theorem cricket_average_increase
    (A : ℝ) -- average score after 18 innings
    (score19 : ℝ) -- runs scored in 19th inning
    (new_average : ℝ) -- new average after 19 innings
    (score19_def : score19 = 97)
    (new_average_def :  new_average = 25)
    (total_runs_def : 19 * new_average = 18 * A + 97) : 
    new_average - (18 * A + score19) / 19 = 4 := 
by
  sorry

end cricket_average_increase_l1964_196415


namespace teal_bakery_pumpkin_pie_l1964_196403

theorem teal_bakery_pumpkin_pie (P : ℕ) 
    (pumpkin_price_per_slice : ℕ := 5)
    (custard_price_per_slice : ℕ := 6)
    (pumpkin_pies_sold : ℕ := 4)
    (custard_pies_sold : ℕ := 5)
    (custard_pieces_per_pie : ℕ := 6)
    (total_revenue : ℕ := 340) :
    4 * P * pumpkin_price_per_slice + custard_pies_sold * custard_pieces_per_pie * custard_price_per_slice = total_revenue → P = 8 := 
by
  sorry

end teal_bakery_pumpkin_pie_l1964_196403


namespace break_even_production_volume_l1964_196423

theorem break_even_production_volume
  (Q : ℕ) 
  (ATC : ℕ → ℚ)
  (P : ℚ)
  (h1 : ∀ Q, ATC Q = 100 + 100000 / Q)
  (h2 : P = 300) :
  ATC 500 = P :=
by
  sorry

end break_even_production_volume_l1964_196423


namespace problem_statement_l1964_196418

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, 8 < x → f (x) > f (x + 1))
variable (h2 : ∀ x, f (x + 8) = f (-x + 8))

theorem problem_statement : f 7 > f 10 := by
  sorry

end problem_statement_l1964_196418


namespace girls_attending_event_l1964_196421

theorem girls_attending_event (total_students girls_attending boys_attending : ℕ) 
    (h1 : total_students = 1500) 
    (h2 : girls_attending = 3 / 5 * girls) 
    (h3 : boys_attending = 2 / 3 * (total_students - girls)) 
    (h4 : girls_attending + boys_attending = 900) : 
    girls_attending = 900 := 
by 
    sorry

end girls_attending_event_l1964_196421


namespace crayons_per_box_l1964_196491

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end crayons_per_box_l1964_196491
