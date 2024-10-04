import Mathlib

namespace reflection_line_slope_intercept_l245_245531

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l245_245531


namespace vlad_score_l245_245379

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end vlad_score_l245_245379


namespace probability_sum_even_l245_245350

def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def sum_is_even (a b : ℕ) : Prop :=
  (a + b) % 2 = 0

theorem probability_sum_even :
  (nat.choose 10 2) = 45 →
  (∀ a b : ℕ, a ∈ primes → b ∈ primes → a ≠ b →
    (sum_is_even a b ↔ a ≠ 2 ∧ b ≠ 2)) →
  ((45 - 9) / 45 : ℚ) = 4 / 5 :=
by sorry

end probability_sum_even_l245_245350


namespace general_term_sequence_l245_245536

theorem general_term_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2)
  (h2 : ∀ (m : ℕ), m ≥ 2 → a m - a (m - 1) + 1 = 0) : 
  a n = 3 - n :=
sorry

end general_term_sequence_l245_245536


namespace greatest_number_of_roses_l245_245690

noncomputable def individual_rose_price: ℝ := 2.30
noncomputable def dozen_rose_price: ℝ := 36
noncomputable def two_dozen_rose_price: ℝ := 50
noncomputable def budget: ℝ := 680

theorem greatest_number_of_roses (P: ℝ → ℝ → ℝ → ℝ → ℕ) :
  P individual_rose_price dozen_rose_price two_dozen_rose_price budget = 325 :=
sorry

end greatest_number_of_roses_l245_245690


namespace find_a2016_l245_245354

-- Define the sequence according to the conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 2) = a (n + 1) - a n

-- State the main theorem we want to prove
theorem find_a2016 :
  ∃ a : ℕ → ℤ, seq a ∧ a 2016 = -4 :=
by
  sorry

end find_a2016_l245_245354


namespace max_n_sum_pos_largest_term_seq_l245_245507

-- Define the arithmetic sequence {a_n} and sum of first n terms S_n along with given conditions
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d
def sum_arith_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

variable (a_1 d : ℤ)
-- Conditions from problem
axiom a8_pos : arithmetic_seq a_1 d 8 > 0
axiom a8_a9_neg : arithmetic_seq a_1 d 8 + arithmetic_seq a_1 d 9 < 0

-- Prove the maximum n for which Sum S_n > 0 is 15
theorem max_n_sum_pos : ∃ n_max : ℤ, sum_arith_seq a_1 d n_max > 0 ∧ 
  ∀ n : ℤ, n > n_max → sum_arith_seq a_1 d n ≤ 0 := by
    exact ⟨15, sorry⟩  -- Substitute 'sorry' for the proof part

-- Determine the largest term in the sequence {S_n / a_n} for 1 ≤ n ≤ 15
theorem largest_term_seq : ∃ n_largest : ℤ, ∀ n : ℤ, 1 ≤ n → n ≤ 15 → 
  (sum_arith_seq a_1 d n / arithmetic_seq a_1 d n) ≤ (sum_arith_seq a_1 d n_largest / arithmetic_seq a_1 d n_largest) := by
    exact ⟨8, sorry⟩  -- Substitute 'sorry' for the proof part

end max_n_sum_pos_largest_term_seq_l245_245507


namespace perimeter_of_large_square_l245_245905

theorem perimeter_of_large_square (squares : List ℕ) (h : squares = [1, 1, 2, 3, 5, 8, 13]) : 2 * (21 + 13) = 68 := by
  sorry

end perimeter_of_large_square_l245_245905


namespace find_x_l245_245741

def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (4, x)

theorem find_x (x : ℝ) (h : ∃k : ℝ, b x = (k * a.1, k * a.2)) : x = 6 := 
by 
  sorry

end find_x_l245_245741


namespace probability_same_color_l245_245694

-- Definitions based on conditions
def num_green_balls : ℕ := 7
def num_white_balls : ℕ := 7
def total_balls : ℕ := num_green_balls + num_white_balls
noncomputable def total_combinations : ℚ := (nat.choose total_balls 2 : ℕ)
noncomputable def combinations_green : ℚ := (nat.choose num_green_balls 2 : ℕ)
noncomputable def combinations_white : ℚ := (nat.choose num_white_balls 2 : ℕ)

-- The statement to prove
theorem probability_same_color : (combinations_green + combinations_white) / total_combinations = 42 / 91 :=
by
  sorry

end probability_same_color_l245_245694


namespace inequality_transformation_l245_245750

theorem inequality_transformation (x y : ℝ) (h : 2 * x - 5 < 2 * y - 5) : x < y := 
by 
  sorry

end inequality_transformation_l245_245750


namespace division_theorem_l245_245272

noncomputable def p (z : ℝ) : ℝ := 4 * z ^ 3 - 8 * z ^ 2 + 9 * z - 7
noncomputable def d (z : ℝ) : ℝ := 4 * z + 2
noncomputable def q (z : ℝ) : ℝ := z ^ 2 - 2.5 * z + 3.5
def r : ℝ := -14

theorem division_theorem (z : ℝ) : p z = d z * q z + r := 
by
  sorry

end division_theorem_l245_245272


namespace number_of_white_balls_l245_245352

theorem number_of_white_balls (x : ℕ) (h : (5 : ℚ) / (5 + x) = 1 / 4) : x = 15 :=
by 
  sorry

end number_of_white_balls_l245_245352


namespace find_original_number_l245_245706

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end find_original_number_l245_245706


namespace necessary_and_sufficient_condition_l245_245220

variable (p q : Prop)

theorem necessary_and_sufficient_condition (h1 : p → q) (h2 : q → p) : (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l245_245220


namespace find_a_l245_245806

theorem find_a (a b c : ℝ) (h1 : ∀ x, x = 2 → y = 5) (h2 : ∀ x, x = 3 → y = 7) :
  a = 2 :=
sorry

end find_a_l245_245806


namespace rancher_monetary_loss_l245_245566

def rancher_head_of_cattle := 500
def market_rate_per_head := 700
def sick_cattle := 350
def additional_cost_per_sick_animal := 80
def reduced_price_per_head := 450

def expected_revenue := rancher_head_of_cattle * market_rate_per_head
def loss_from_death := sick_cattle * market_rate_per_head
def additional_sick_cost := sick_cattle * additional_cost_per_sick_animal
def remaining_cattle := rancher_head_of_cattle - sick_cattle
def revenue_from_remaining_cattle := remaining_cattle * reduced_price_per_head

def total_loss := (expected_revenue - revenue_from_remaining_cattle) + additional_sick_cost

theorem rancher_monetary_loss : total_loss = 310500 := by
  sorry

end rancher_monetary_loss_l245_245566


namespace flower_shop_percentage_l245_245230

theorem flower_shop_percentage (C : ℕ) : 
  let V := (1/3 : ℝ) * C
  let T := (1/12 : ℝ) * C
  let R := T
  let total := C + V + T + R
  (C / total) * 100 = 66.67 := 
by
  sorry

end flower_shop_percentage_l245_245230


namespace area_ratio_proof_l245_245353

open Real

noncomputable def area_ratio (FE AF DE CD ABCE : ℝ) :=
  (AF = 3 * FE) ∧ (CD = 3 * DE) ∧ (ABCE = 16 * FE^2) →
  (10 * FE^2 / ABCE = (5 / 8))

theorem area_ratio_proof (FE AF DE CD ABCE : ℝ) :
  AF = 3 * FE → CD = 3 * DE → ABCE = 16 * FE^2 →
  10 * FE^2 / ABCE = 5 / 8 :=
by
  intro hAF hCD hABCE
  sorry

end area_ratio_proof_l245_245353


namespace g_2002_equals_1_l245_245206

theorem g_2002_equals_1 (f : ℝ → ℝ)
  (hf1 : f 1 = 1)
  (hf2 : ∀ x : ℝ, f (x + 5) ≥ f x + 5)
  (hf3 : ∀ x : ℝ, f (x + 1) ≤ f x + 1)
  (g : ℝ → ℝ := fun x => f x + 1 - x)
  : g 2002 = 1 :=
by
  sorry

end g_2002_equals_1_l245_245206


namespace net_distance_from_start_total_distance_driven_fuel_consumption_l245_245113

def driving_distances : List Int := [14, -3, 7, -3, 11, -4, -3, 11, 6, -7, 9]

theorem net_distance_from_start : List.sum driving_distances = 38 := by
  sorry

theorem total_distance_driven : List.sum (List.map Int.natAbs driving_distances) = 78 := by
  sorry

theorem fuel_consumption (fuel_rate : Float) (total_distance : Nat) : total_distance = 78 → total_distance.toFloat * fuel_rate = 7.8 := by
  intros h_total_distance
  rw [h_total_distance]
  norm_num
  sorry

end net_distance_from_start_total_distance_driven_fuel_consumption_l245_245113


namespace rosa_called_pages_sum_l245_245068

theorem rosa_called_pages_sum :
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  week1_pages + week2_pages + week3_pages = 31.2 :=
by
  let week1_pages := 10.2
  let week2_pages := 8.6
  let week3_pages := 12.4
  sorry  -- proof will be done here

end rosa_called_pages_sum_l245_245068


namespace remaining_amount_needed_l245_245823

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end remaining_amount_needed_l245_245823


namespace highest_geometric_frequency_count_l245_245038

-- Define the problem conditions and the statement to be proved
theorem highest_geometric_frequency_count :
  ∀ (vol : ℕ) (num_groups : ℕ) (cum_freq_first_seven : ℝ)
  (remaining_freqs : List ℕ) (total_freq_remaining : ℕ)
  (r : ℕ) (a : ℕ),
  vol = 100 → 
  num_groups = 10 → 
  cum_freq_first_seven = 0.79 → 
  total_freq_remaining = 21 → 
  r > 1 →
  remaining_freqs = [a, a * r, a * r ^ 2] → 
  a * (1 + r + r ^ 2) = total_freq_remaining → 
  ∃ max_freq, max_freq ∈ remaining_freqs ∧ max_freq = 12 :=
by
  intro vol num_groups cum_freq_first_seven remaining_freqs total_freq_remaining r a
  intros h_vol h_num_groups h_cum_freq_first h_total_freq_remaining h_r_pos h_geom_seq h_freq_sum
  use 12
  sorry

end highest_geometric_frequency_count_l245_245038


namespace carl_highway_miles_l245_245912

theorem carl_highway_miles
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (city_miles : ℕ)
  (gas_cost_per_gallon : ℕ)
  (total_cost : ℕ)
  (h1 : city_mpg = 30)
  (h2 : highway_mpg = 40)
  (h3 : city_miles = 60)
  (h4 : gas_cost_per_gallon = 3)
  (h5 : total_cost = 42)
  : (total_cost - (city_miles / city_mpg) * gas_cost_per_gallon) / gas_cost_per_gallon * highway_mpg = 480 := 
by
  sorry

end carl_highway_miles_l245_245912


namespace solve_equation_l245_245108

theorem solve_equation (x : ℝ) (h : (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2) : 
  x = -2/3 :=
sorry

end solve_equation_l245_245108


namespace first_chapter_length_l245_245864

theorem first_chapter_length (total_pages : ℕ) (second_chapter_pages : ℕ) (third_chapter_pages : ℕ)
  (h : total_pages = 125) (h2 : second_chapter_pages = 35) (h3 : third_chapter_pages  = 24) :
  total_pages - second_chapter_pages - third_chapter_pages = 66 :=
by
  -- Construct the proof using the provided conditions
  sorry

end first_chapter_length_l245_245864


namespace problem_l245_245777

def T := {n : ℤ | ∃ (k : ℤ), n = 4 * (2*k + 1)^2 + 13}

theorem problem :
  (∀ n ∈ T, ¬ 2 ∣ n) ∧ (∀ n ∈ T, ¬ 5 ∣ n) :=
by
  sorry

end problem_l245_245777


namespace max_suitable_pairs_l245_245467

theorem max_suitable_pairs (m n : ℕ) (hm : 1 < m) (hn : 1 < n)
  (As : Fin m → Finset (ℕ)) (h_size : ∀ i, (As i).card = n)
  (h_disjoint : ∀ i j, i ≠ j → As i ∩ As j = ∅)
  (h_div_cond : ∀ i (a ∈ As i) (b ∈ As ((i+1) % m)), ¬ (a ∣ b)) :
  (∑ i in Finset.range m, (As i).sum) ∑ a, ∑ b, (a ∣ b) = n^2 * (nat.choose (m-1) 2) :=
sorry

end max_suitable_pairs_l245_245467


namespace round_trip_ticket_percentage_l245_245276

theorem round_trip_ticket_percentage (P R : ℝ) 
  (h1 : 0.20 * P = 0.50 * R) : R = 0.40 * P :=
by
  sorry

end round_trip_ticket_percentage_l245_245276


namespace smallest_five_digit_divisible_by_3_and_4_l245_245680

theorem smallest_five_digit_divisible_by_3_and_4 : 
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ n = 10008 :=
sorry

end smallest_five_digit_divisible_by_3_and_4_l245_245680


namespace gcf_60_75_l245_245132

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l245_245132


namespace find_original_number_l245_245704

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end find_original_number_l245_245704


namespace jane_oldest_child_age_l245_245006

-- Define the conditions
def jane_start_age : ℕ := 20
def jane_current_age : ℕ := 32
def stopped_babysitting_years_ago : ℕ := 10
def baby_sat_condition (jane_age child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- Define the proof problem
theorem jane_oldest_child_age :
  (∃ age_stopped child_age,
    stopped_babysitting_years_ago = jane_current_age - age_stopped ∧
    baby_sat_condition age_stopped child_age ∧
    (32 - stopped_babysitting_years_ago = 22) ∧ -- Jane's age when she stopped baby-sitting
    child_age = 22 / 2 ∧ -- Oldest child she could have baby-sat at age 22
    child_age + stopped_babysitting_years_ago = 21) --  current age of the oldest person for whom Jane could have baby-sat
:= sorry

end jane_oldest_child_age_l245_245006


namespace smallest_x_value_l245_245376

theorem smallest_x_value : ∃ x : ℤ, ∃ y : ℤ, (xy + 7 * x + 6 * y = -8) ∧ x = -40 :=
by
  sorry

end smallest_x_value_l245_245376


namespace parabola_vertex_calc_l245_245425

noncomputable def vertex_parabola (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem parabola_vertex_calc 
  (a b c : ℝ) 
  (h_vertex : vertex_parabola a b c 2 = 5)
  (h_point : vertex_parabola a b c 1 = 8) : 
  a - b + c = 32 :=
sorry

end parabola_vertex_calc_l245_245425


namespace find_m_l245_245564

def g (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5
  else if n % 3 = 0 then n / 3
  else n / 2

theorem find_m (m : ℤ) (h_odd : m % 2 = 1) (h_g : g (g (g m)) = 16) : m = 59 ∨ m = 91 :=
by sorry

end find_m_l245_245564


namespace min_value_of_sum_l245_245611

theorem min_value_of_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 4 / x + 1 / y = 1) : x + y = 9 :=
by
  -- sorry used to skip the proof
  sorry

end min_value_of_sum_l245_245611


namespace david_age_l245_245435

theorem david_age (A B C D : ℕ)
  (h1 : A = B - 5)
  (h2 : B = C + 2)
  (h3 : D = C + 4)
  (h4 : A = 12) : D = 19 :=
sorry

end david_age_l245_245435


namespace range_of_x_in_second_quadrant_l245_245080

theorem range_of_x_in_second_quadrant (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end range_of_x_in_second_quadrant_l245_245080


namespace seating_arrangements_exactly_two_adjacent_empty_seats_l245_245830

theorem seating_arrangements_exactly_two_adjacent_empty_seats : 
  (∃ (arrangements : ℕ), arrangements = 72) :=
by
  sorry

end seating_arrangements_exactly_two_adjacent_empty_seats_l245_245830


namespace quadratic_square_binomial_l245_245305

theorem quadratic_square_binomial (a r s : ℚ) (h1 : a = r^2) (h2 : 2 * r * s = 26) (h3 : s^2 = 9) :
  a = 169/9 := sorry

end quadratic_square_binomial_l245_245305


namespace quadratic_two_distinct_real_roots_l245_245342

theorem quadratic_two_distinct_real_roots (m : ℝ) (h : -4 * m > 0) : m = -1 :=
sorry

end quadratic_two_distinct_real_roots_l245_245342


namespace positive_difference_volumes_l245_245577

open Real

noncomputable def charlies_height := 12
noncomputable def charlies_circumference := 10
noncomputable def danas_height := 8
noncomputable def danas_circumference := 10

theorem positive_difference_volumes (hC : ℝ := charlies_height) (CC : ℝ := charlies_circumference)
                                   (hD : ℝ := danas_height) (CD : ℝ := danas_circumference) :
    (π * (π * ((CD / (2 * π)) ^ 2) * hD - π * ((CC / (2 * π)) ^ 2) * hC)) = 100 :=
by
  have rC := CC / (2 * π)
  have VC := π * (rC ^ 2) * hC
  have rD := CD / (2 * π)
  have VD := π * (rD ^ 2) * hD
  sorry

end positive_difference_volumes_l245_245577


namespace probability_A_wins_l245_245558

theorem probability_A_wins 
  (prob_draw : ℚ)
  (prob_B_wins : ℚ)
  (h_draw : prob_draw = 1/2)
  (h_B_wins : prob_B_wins = 1/3) : 
  1 - prob_draw - prob_B_wins = 1 / 6 :=
by
  rw [h_draw, h_B_wins]
  norm_num

end probability_A_wins_l245_245558


namespace determine_value_of_y_l245_245121

variable (s y : ℕ)
variable (h_pos : s > 30)
variable (h_eq : s * s = (s - 15) * (s + y))

theorem determine_value_of_y (h_pos : s > 30) (h_eq : s * s = (s - 15) * (s + y)) : 
  y = 15 * s / (s + 15) :=
by
  sorry

end determine_value_of_y_l245_245121


namespace square_pattern_1111111_l245_245119

theorem square_pattern_1111111 :
  11^2 = 121 ∧ 111^2 = 12321 ∧ 1111^2 = 1234321 → 1111111^2 = 1234567654321 :=
by
  sorry

end square_pattern_1111111_l245_245119


namespace simplify_expression_correct_l245_245980

variable (a b x y : ℝ) (i : ℂ)

noncomputable def simplify_expression (a b x y : ℝ) (i : ℂ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (i^2 = -1) → (a * x + b * i * y) * (a * x - b * i * y) = a^2 * x^2 + b^2 * y^2

theorem simplify_expression_correct (a b x y : ℝ) (i : ℂ) :
  simplify_expression a b x y i := by
  sorry

end simplify_expression_correct_l245_245980


namespace find_base_l245_245945

theorem find_base (b : ℕ) (h : (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 2 * b + 5) : b = 7 :=
sorry

end find_base_l245_245945


namespace complex_number_quadrant_l245_245474

open Complex

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2 * Complex.I) / z = Complex.I) : 
  (0 < z.re) ∧ (0 < z.im) :=
by
  -- sorry to skip the actual proof
  sorry

end complex_number_quadrant_l245_245474


namespace least_subtracted_number_l245_245148

theorem least_subtracted_number (r : ℕ) : r = 10^1000 % 97 := 
sorry

end least_subtracted_number_l245_245148


namespace tic_tac_toe_ways_l245_245952

theorem tic_tac_toe_ways :
  let n := 4 in
  let k := 4 in
  let remaining := n * n - k in
  (2 * nat.choose remaining k + 8 * nat.choose remaining k = 4950) := 
  by sorry

end tic_tac_toe_ways_l245_245952


namespace minimum_value_l245_245311

noncomputable def f (x : ℝ) : ℝ := x^2 / (x - 10)

theorem minimum_value (x : ℝ) (h : x > 10) : (∃ y : ℝ, (∀ x' : ℝ, x' > 10 → f x' ≥ y) ∧ y = 40) := 
sorry

end minimum_value_l245_245311


namespace penniless_pete_dime_difference_l245_245642

theorem penniless_pete_dime_difference :
  ∃ a b c : ℕ, 
  (a + b + c = 100) ∧ 
  (5 * a + 10 * b + 50 * c = 1350) ∧ 
  (b = 170 ∨ b = 8) ∧ 
  (b - 8 = 162 ∨ 170 - b = 162) :=
sorry

end penniless_pete_dime_difference_l245_245642


namespace sqrt_of_9_l245_245828

theorem sqrt_of_9 (x : ℝ) (h : x^2 = 9) : x = 3 ∨ x = -3 :=
by {
  sorry
}

end sqrt_of_9_l245_245828


namespace line_passes_through_fixed_point_minimum_area_triangle_l245_245320

theorem line_passes_through_fixed_point (k : ℝ) :
  ∃ P : ℝ × ℝ, P = (2, 1) ∧ ∀ k : ℝ, (k * 2 - 1 + 1 - 2 * k = 0) :=
sorry

theorem minimum_area_triangle (k : ℝ) :
  ∀ k: ℝ, k < 0 → 1/2 * (2 - 1/k) * (1 - 2*k) ≥ 4 ∧ 
           (1/2 * (2 - 1/k) * (1 - 2*k) = 4 ↔ k = -1/2) :=
sorry

end line_passes_through_fixed_point_minimum_area_triangle_l245_245320


namespace gcd_of_lcm_and_ratio_l245_245615

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end gcd_of_lcm_and_ratio_l245_245615


namespace find_number_l245_245829

theorem find_number (x : ℤ) (h : x + x^2 + 15 = 96) : x = -9 :=
sorry

end find_number_l245_245829


namespace total_length_of_sticks_l245_245770

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end total_length_of_sticks_l245_245770


namespace find_original_number_l245_245707

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end find_original_number_l245_245707


namespace find_multiplier_l245_245682

theorem find_multiplier (n m : ℕ) (h1 : 2 * n = (26 - n) + 19) (h2 : n = 15) : m = 2 :=
by
  sorry

end find_multiplier_l245_245682


namespace circles_intersect_l245_245605

theorem circles_intersect (t : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * t * x + t^2 - 4 = 0 ∧ x^2 + y^2 + 2 * x - 4 * t * y + 4 * t^2 - 8 = 0) ↔ 
  (-12 / 5 < t ∧ t < -2 / 5) ∨ (0 < t ∧ t < 2) :=
sorry

end circles_intersect_l245_245605


namespace even_of_even_square_sqrt_two_irrational_l245_245413

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end even_of_even_square_sqrt_two_irrational_l245_245413


namespace solve_diophantine_l245_245374

theorem solve_diophantine :
  {xy : ℤ × ℤ | 5 * (xy.1 ^ 2) + 5 * xy.1 * xy.2 + 5 * (xy.2 ^ 2) = 7 * xy.1 + 14 * xy.2} = {(-1, 3), (0, 0), (1, 2)} :=
by sorry

end solve_diophantine_l245_245374


namespace probability_both_boys_or_both_girls_l245_245196

theorem probability_both_boys_or_both_girls 
  (total_students : ℕ) (boys : ℕ) (girls : ℕ) :
  total_students = 5 → boys = 2 → girls = 3 →
    (∃ (p : ℚ), p = 2/5) :=
by
  intros ht hb hg
  sorry

end probability_both_boys_or_both_girls_l245_245196


namespace license_plates_count_correct_l245_245297

/-- Calculate the number of five-character license plates. -/
def count_license_plates : Nat :=
  let num_consonants := 20
  let num_vowels := 6
  let num_digits := 10
  num_consonants^2 * num_vowels^2 * num_digits

theorem license_plates_count_correct :
  count_license_plates = 144000 :=
by
  sorry

end license_plates_count_correct_l245_245297


namespace correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l245_245550

variable {a b : ℝ}

theorem correct_calculation : a ^ 3 * a = a ^ 4 := 
by
  sorry

theorem incorrect_calculation_A : a ^ 3 + a ^ 3 ≠ 2 * a ^ 6 := 
by
  sorry

theorem incorrect_calculation_B : (a ^ 3) ^ 3 ≠ a ^ 6 :=
by
  sorry

theorem incorrect_calculation_D : (a - b) ^ 2 ≠ a ^ 2 - b ^ 2 :=
by
  sorry

end correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l245_245550


namespace nate_cooking_for_people_l245_245693

/-- Given that 8 jumbo scallops weigh one pound, scallops cost $24.00 per pound, Nate is pairing 2 scallops with a corn bisque per person, and he spends $48 on scallops. We want to prove that Nate is cooking for 8 people. -/
theorem nate_cooking_for_people :
  (8 : ℕ) = 8 →
  (24 : ℕ) = 24 →
  (2 : ℕ) = 2 →
  (48 : ℕ) = 48 →
  let scallops_per_pound := 8
  let cost_per_pound := 24
  let scallops_per_person := 2
  let money_spent := 48
  let pounds_of_scallops := money_spent / cost_per_pound
  let total_scallops := scallops_per_pound * pounds_of_scallops
  let people := total_scallops / scallops_per_person
  people = 8 :=
by
  sorry

end nate_cooking_for_people_l245_245693


namespace cyclic_sum_inequality_l245_245053

variable (a b c : ℝ)
variable (pos_a : a > 0)
variable (pos_b : b > 0)
variable (pos_c : c > 0)

theorem cyclic_sum_inequality :
  ( (a^3 + b^3) / (a^2 + a * b + b^2) + 
    (b^3 + c^3) / (b^2 + b * c + c^2) + 
    (c^3 + a^3) / (c^2 + c * a + a^2) ) ≥ 
  (2 / 3) * (a + b + c) := 
  sorry

end cyclic_sum_inequality_l245_245053


namespace expected_coin_worth_is_two_l245_245011

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end expected_coin_worth_is_two_l245_245011


namespace middle_of_three_consecutive_integers_is_60_l245_245264

theorem middle_of_three_consecutive_integers_is_60 (n : ℤ)
    (h : (n - 1) + n + (n + 1) = 180) : n = 60 := by
  sorry

end middle_of_three_consecutive_integers_is_60_l245_245264


namespace system_of_equations_solution_system_of_inequalities_solution_l245_245983

theorem system_of_equations_solution (x y : ℝ) :
  (3 * x - 4 * y = 1) → (5 * x + 2 * y = 6) → 
  x = 1 ∧ y = 0.5 := by
  sorry

theorem system_of_inequalities_solution (x : ℝ) :
  (3 * x + 6 > 0) → (x - 2 < -x) → 
  -2 < x ∧ x < 1 := by
  sorry

end system_of_equations_solution_system_of_inequalities_solution_l245_245983


namespace max_electronic_thermometers_l245_245490

-- Definitions
def budget : ℕ := 300
def price_mercury : ℕ := 3
def price_electronic : ℕ := 10
def total_students : ℕ := 53

-- The theorem statement
theorem max_electronic_thermometers : 
  (∃ x : ℕ, x ≤ total_students ∧ 10 * x + 3 * (total_students - x) ≤ budget ∧ 
            ∀ y : ℕ, y ≤ total_students ∧ 10 * y + 3 * (total_students - y) ≤ budget → y ≤ x) :=
sorry

end max_electronic_thermometers_l245_245490


namespace inequality_proof_l245_245511

theorem inequality_proof {k l m n : ℕ} (h_pos_k : 0 < k) (h_pos_l : 0 < l) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_klmn : k < l ∧ l < m ∧ m < n)
  (h_equation : k * n = l * m) : 
  (n - k) / 2 ^ 2 ≥ k + 2 := 
by sorry

end inequality_proof_l245_245511


namespace infinite_nested_radicals_solution_l245_245587

theorem infinite_nested_radicals_solution :
  ∃ x : ℝ, 
    (∃ y z : ℝ, (y = (x * y)^(1/3) ∧ z = (x + z)^(1/3)) ∧ y = z) ∧ 
    0 < x ∧ x = (3 + Real.sqrt 5) / 2 := 
sorry

end infinite_nested_radicals_solution_l245_245587


namespace reflection_line_slope_l245_245529

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l245_245529


namespace sqrt_sqrt_81_is_9_l245_245657

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l245_245657


namespace contrapositive_statement_l245_245062

theorem contrapositive_statement 
  (a : ℝ) (b : ℝ) 
  (h1 : a > 0) 
  (h3 : a + b < 0) : 
  b < 0 :=
sorry

end contrapositive_statement_l245_245062


namespace average_rate_l245_245540

variable (d_run : ℝ) (d_swim : ℝ) (r_run : ℝ) (r_swim : ℝ)
variable (t_run : ℝ := d_run / r_run) (t_swim : ℝ := d_swim / r_swim)

theorem average_rate (h_dist_run : d_run = 4) (h_dist_swim : d_swim = 4)
                      (h_run_rate : r_run = 10) (h_swim_rate : r_swim = 6) : 
                      ((d_run + d_swim) / (t_run + t_swim)) / 60 = 0.125 :=
by
  -- Properly using all the conditions given
  have := (4 + 4) / (4 / 10 + 4 / 6) / 60 = 0.125
  sorry

end average_rate_l245_245540


namespace find_sum_of_variables_l245_245112

variables (a b c d : ℤ)

theorem find_sum_of_variables
    (h1 : a - b + c = 7)
    (h2 : b - c + d = 8)
    (h3 : c - d + a = 4)
    (h4 : d - a + b = 3)
    (h5 : a + b + c - d = 10) :
    a + b + c + d = 16 := 
sorry

end find_sum_of_variables_l245_245112


namespace part_a_l245_245502

open Complex

theorem part_a (z : ℂ) (hz : abs z = 1) :
  (abs (z + 1) - Real.sqrt 2) * (abs (z - 1) - Real.sqrt 2) ≤ 0 :=
by
  -- Proof will go here
  sorry

end part_a_l245_245502


namespace problem_statement_l245_245089

theorem problem_statement (x y : ℝ) (M N P : ℝ) 
  (hM_def : M = 2 * x + y)
  (hN_def : N = 2 * x - y)
  (hP_def : P = x * y)
  (hM : M = 4)
  (hN : N = 2) : P = 1.5 :=
by
  sorry

end problem_statement_l245_245089


namespace inclination_angle_of_line_l245_245385

theorem inclination_angle_of_line : 
  ∃ θ ∈ set.Ico 0 180, tan (θ * (π / 180)) = -1 ∧ θ = 135 := 
by {
  use 135,
  split,
  { show 135 ∈ set.Ico 0 180, from ⟨le_refl _, by norm_num⟩ },
  split,
  { show tan (135 * (π / 180)) = -1, from by simp [real.tan_pi_div_four, mul_div_cancel_left' (by norm_num : 0 < 2)] },
  { show 135 = 135, from eq.refl 135 }
}

end inclination_angle_of_line_l245_245385


namespace distance_to_y_axis_l245_245492

theorem distance_to_y_axis {x y : ℝ} (h : x = -3 ∧ y = 4) : abs x = 3 :=
by
  sorry

end distance_to_y_axis_l245_245492


namespace largest_number_of_gold_coins_l245_245551

theorem largest_number_of_gold_coins (n : ℕ) (h1 : n % 15 = 4) (h2 : n < 150) : n ≤ 139 :=
by {
  -- This is where the proof would go.
  sorry
}

end largest_number_of_gold_coins_l245_245551


namespace max_value_of_E_l245_245643

theorem max_value_of_E (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ^ 5 + b ^ 5 = a ^ 3 + b ^ 3) : 
  a^2 - a*b + b^2 ≤ 1 :=
sorry

end max_value_of_E_l245_245643


namespace rainfall_ratio_l245_245919

theorem rainfall_ratio (R1 R2 : ℕ) (hR2 : R2 = 24) (hTotal : R1 + R2 = 40) : 
  R2 / R1 = 3 / 2 := by
  sorry

end rainfall_ratio_l245_245919


namespace find_m_l245_245330

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 + m
noncomputable def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem find_m (m : ℝ) : 
  ∃ a b : ℝ, (0 < a) ∧ (f a m = b) ∧ (g a = b) ∧ (2 * a = (6 / a) - 4) → m = -5 := 
by
  sorry

end find_m_l245_245330


namespace reception_time_l245_245715

-- Definitions of conditions
def noon : ℕ := 12 * 60 -- define noon in minutes
def rabbit_walk_speed (v : ℕ) : Prop := v > 0
def rabbit_run_speed (v : ℕ) : Prop := 2 * v > 0
def distance (D : ℕ) : Prop := D > 0
def delay (minutes : ℕ) : Prop := minutes = 10

-- Definition of the problem
theorem reception_time (v D : ℕ) (h_v : rabbit_walk_speed v) (h_D : distance D) (h_delay : delay 10) :
  noon + (D / v) * 2 / 3 = 12 * 60 + 40 :=
by sorry

end reception_time_l245_245715


namespace correct_division_l245_245836

theorem correct_division (x : ℝ) (h : 8 * x + 8 = 56) : x / 8 = 0.75 :=
by
  sorry

end correct_division_l245_245836


namespace difference_increased_decreased_l245_245383

theorem difference_increased_decreased (x : ℝ) (hx : x = 80) : 
  ((x * 1.125) - (x * 0.75)) = 30 := by
  have h1 : x * 1.125 = 90 := by rw [hx]; norm_num
  have h2 : x * 0.75 = 60 := by rw [hx]; norm_num
  rw [h1, h2]
  norm_num
  done

end difference_increased_decreased_l245_245383


namespace gcf_60_75_l245_245131

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l245_245131


namespace lucy_reads_sixty_pages_l245_245576

-- Define the number of pages Carter, Lucy, and Oliver can read in an hour.
def pages_carter : ℕ := 30
def pages_oliver : ℕ := 40

-- Carter reads half as many pages as Lucy.
def reads_half_as_much_as (a b : ℕ) : Prop := a = b / 2

-- Lucy reads more pages than Oliver.
def reads_more_than (a b : ℕ) : Prop := a > b

-- The goal is to show that Lucy can read 60 pages in an hour.
theorem lucy_reads_sixty_pages (pages_lucy : ℕ) (h1 : reads_half_as_much_as pages_carter pages_lucy)
  (h2 : reads_more_than pages_lucy pages_oliver) : pages_lucy = 60 :=
sorry

end lucy_reads_sixty_pages_l245_245576


namespace x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l245_245412

theorem x_gt_1_implies_inv_x_lt_1 (x : ℝ) (h : x > 1) : 1 / x < 1 :=
by
  sorry

theorem inv_x_lt_1_not_necessitates_x_gt_1 (x : ℝ) (h : 1 / x < 1) : ¬(x > 1) ∨ (x ≤ 1) :=
by
  sorry

end x_gt_1_implies_inv_x_lt_1_inv_x_lt_1_not_necessitates_x_gt_1_l245_245412


namespace largest_rectangle_area_l245_245819

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l245_245819


namespace problem_1_problem_2_problem_3_l245_245329

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem problem_1 : f (Real.pi / 2) = 1 := 
sorry

theorem problem_2 : (∃ p > 0, ∀ x, f (x + p) = f x) ∧ (∀ p, p > 0 ∧ (∀ x, f (x + p) = f x) → p ≥ 2 * Real.pi) := 
sorry

theorem problem_3 : ∃ x : ℝ, g x = -2 := 
sorry

end problem_1_problem_2_problem_3_l245_245329


namespace compare_abc_l245_245319

noncomputable def a : ℝ := (1 / 2) * Real.cos (4 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (4 * Real.pi / 180)
noncomputable def b : ℝ := Real.cos (2 * 13 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (2 * 23 * Real.pi / 180)

theorem compare_abc : a < b ∧ b < c := by
  sorry

end compare_abc_l245_245319


namespace max_value_of_expression_l245_245637

theorem max_value_of_expression (x y z : ℝ) (h : 0 < x) (h' : 0 < y) (h'' : 0 < z) (hxyz : x * y * z = 1) :
  (∃ s, s = x ∧ ∃ t, t = y ∧ ∃ u, u = z ∧ 
  (x^2 * y / (x + y) + y^2 * z / (y + z) + z^2 * x / (z + x) ≤ 3 / 2)) :=
sorry

end max_value_of_expression_l245_245637


namespace harvest_season_weeks_l245_245242

-- Definitions based on given conditions
def weekly_earnings : ℕ := 491
def weekly_rent : ℕ := 216
def total_savings : ℕ := 324775

-- Definition to calculate net earnings per week
def net_earnings_per_week (earnings rent : ℕ) : ℕ :=
  earnings - rent

-- Definition to calculate number of weeks
def number_of_weeks (savings net_earnings : ℕ) : ℕ :=
  savings / net_earnings

theorem harvest_season_weeks :
  number_of_weeks total_savings (net_earnings_per_week weekly_earnings weekly_rent) = 1181 :=
by
  sorry

end harvest_season_weeks_l245_245242


namespace total_cows_l245_245288

theorem total_cows (Matthews Aaron Tyron Marovich : ℕ) 
  (h1 : Matthews = 60)
  (h2 : Aaron = 4 * Matthews)
  (h3 : Tyron = Matthews - 20)
  (h4 : Aaron + Matthews + Tyron = Marovich + 30) :
  Aaron + Matthews + Tyron + Marovich = 650 :=
by
  sorry

end total_cows_l245_245288


namespace correct_operation_l245_245402

theorem correct_operation (a : ℝ) : (a^3)^3 = a^9 := 
sorry

end correct_operation_l245_245402


namespace parameterized_line_solution_l245_245661

theorem parameterized_line_solution :
  ∃ (s l : ℚ), 
  (∀ t : ℚ, 
    ∃ x y : ℚ, 
      x = -3 + t * l ∧ 
      y = s + t * (-7) ∧ 
      y = 3 * x + 2
  ) ∧
  s = -7 ∧ l = -7 / 3 := 
sorry

end parameterized_line_solution_l245_245661


namespace solution_l245_245179

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (4 + 2 * x) / (6 + 3 * x) = (3 + 2 * x) / (5 + 3 * x) ∧ x = -2

theorem solution : problem_statement :=
by
  sorry

end solution_l245_245179


namespace sum_of_digits_of_smallest_number_l245_245303

noncomputable def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.foldl (· + ·) 0

theorem sum_of_digits_of_smallest_number :
  (n : Nat) → (h1 : (Nat.ceil (n / 2) - Nat.ceil (n / 3) = 15)) → 
  sum_of_digits n = 9 :=
by
  sorry

end sum_of_digits_of_smallest_number_l245_245303


namespace largest_rectangle_area_l245_245810

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l245_245810


namespace cost_of_black_and_white_drawing_l245_245359

-- Given the cost of the color drawing is 1.5 times the cost of the black and white drawing
-- and John paid $240 for the color drawing, we need to prove the cost of the black and white drawing is $160.

theorem cost_of_black_and_white_drawing (C : ℝ) (h : 1.5 * C = 240) : C = 160 :=
by
  sorry

end cost_of_black_and_white_drawing_l245_245359


namespace box_cookies_count_l245_245518

theorem box_cookies_count (cookies_per_bag : ℕ) (cookies_per_box : ℕ) :
  cookies_per_bag = 7 →
  8 * cookies_per_box = 9 * cookies_per_bag + 33 →
  cookies_per_box = 12 :=
by
  intros h1 h2
  sorry

end box_cookies_count_l245_245518


namespace meal_problem_solution_l245_245541

open Nat

-- Definitions based on the conditions in part a)

-- Twelve people situation
def num_people : ℕ := 12

-- Numbers of each meal type
def num_meals : ℕ := 4
def num_each_meal : ℕ := 3

-- Given the two people received their correct meal, it is related to derangements of remaining
def num_correct : ℕ := 2

-- The derangements of 10 people (!10)
def derangements_10 : ℕ := 1334961

-- The total number of ways the waiter can serve the meal types
def total_ways : ℕ :=
  nat.choose num_people num_correct * derangements_10

-- Proving the value total_ways
theorem meal_problem_solution : total_ways = 88047666 :=
by
  -- Use nat.choose as combinatorial selection and derangements calculation.
  have h : nat.choose num_people num_correct = 66 := by norm_num
  have derangements_eq : derangements_10 = 1334961 := by norm_num
  unfold total_ways
  rw [h, derangements_eq]
  norm_num
  exact rfl

end meal_problem_solution_l245_245541


namespace cost_of_each_magazine_l245_245246

theorem cost_of_each_magazine
  (books_about_cats : ℕ)
  (books_about_solar_system : ℕ)
  (magazines : ℕ)
  (cost_per_book : ℝ)
  (total_spent : ℝ)
  (books_total : ℕ := books_about_cats + books_about_solar_system)
  (total_books_cost : ℝ := books_total * cost_per_book)
  (total_magazine_cost : ℝ := total_spent - total_books_cost)
  (magazine_cost : ℝ := total_magazine_cost / magazines)
  (h1 : books_about_cats = 7)
  (h2 : books_about_solar_system = 2)
  (h3 : magazines = 3)
  (h4 : cost_per_book = 7)
  (h5 : total_spent = 75) :
  magazine_cost = 4 :=
by
  sorry

end cost_of_each_magazine_l245_245246


namespace poisson_theorem_estimate_l245_245372

noncomputable theory

variable (n : ℕ) (λ : ℝ)

/-- Given independent Poisson-distributed random variables ηᵢ with parameter λ / n,
and independent Bernoulli-distributed random variables ξᵢ defined as in the problem,
prove that the following estimate holds:
supₖ |Pₙ(k) - λᵏ e^(-λ) / k!| ≤ λ² / n.
-/
theorem poisson_theorem_estimate :
  ∀ (η : ℕ → measure_theory.measurable_space ℕ)
    (P_n : ℕ → ℝ)
  (hyp1 : ∀ i, measure_theory.probability_measure (η i))
  (hyp2 : ∀ i, measure_theory.prob (λ ω, η i ω = 0) = λ / n)
  (hyp3 : ∀ i j, i ≠ j → measure_theory.indep (η i) (η j))
  (hyp4 : ∀ k, P_n k = (λ ^ k) * real.exp (-λ) / k!) ,
  (∀ k, |P_n k - (λ ^ k * real.exp (-λ) / k!)| ≤ λ ^ 2 / n) := 
sorry

end poisson_theorem_estimate_l245_245372


namespace initial_oranges_l245_245016

theorem initial_oranges (X : ℕ) (h1 : X - 37 + 7 = 10) : X = 40 :=
by
  sorry

end initial_oranges_l245_245016


namespace ashok_borrowed_l245_245408

theorem ashok_borrowed (P : ℝ) (h : 11400 = P * (6 / 100 * 2 + 9 / 100 * 3 + 14 / 100 * 4)) : P = 12000 :=
by
  sorry

end ashok_borrowed_l245_245408


namespace part1_part2_l245_245739

-- Definitions as per the conditions
def A (a b : ℚ) := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℚ) := - a^2 + (1/2) * a * b + 2 / 3

-- Part (1)
theorem part1 (a b : ℚ) (h1 : a = -1) (h2 : b = -2) : 
  4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3 := 
by 
  sorry

-- Part (2)
theorem part2 (a : ℚ) : 
  (∀ a : ℚ, 4 * A a b - (3 * A a b - 2 * B a b) = 10 + 1/3) → 
  b = 1/2 :=
by 
  sorry

end part1_part2_l245_245739


namespace application_methods_l245_245422

variables (students : Fin 6) (colleges : Fin 3)

def total_applications_without_restriction : ℕ := 3^6
def applications_missing_one_college : ℕ := 2^6
def overcounted_applications_missing_two_college : ℕ := 1

theorem application_methods (h1 : total_applications_without_restriction = 729)
    (h2 : applications_missing_one_college = 64)
    (h3 : overcounted_applications_missing_two_college = 1) :
    ∀ (students : Fin 6), ∀ (colleges : Fin 3),
      (total_applications_without_restriction - 3 * applications_missing_one_college + 3 * overcounted_applications_missing_two_college = 540) :=
by {
  sorry
}

end application_methods_l245_245422


namespace circle_intersection_range_l245_245755

noncomputable def circleIntersectionRange (r : ℝ) : Prop :=
  1 < r ∧ r < 11

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) :
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) ↔ circleIntersectionRange r :=
by
  sorry

end circle_intersection_range_l245_245755


namespace find_polynomial_l245_245459

def polynomial (a b c : ℚ) : ℚ → ℚ := λ x => a * x^2 + b * x + c

theorem find_polynomial
  (a b c : ℚ)
  (h1 : polynomial a b c (-3) = 0)
  (h2 : polynomial a b c 6 = 0)
  (h3 : polynomial a b c 2 = -24) :
  a = 6/5 ∧ b = -18/5 ∧ c = -108/5 :=
by 
  sorry

end find_polynomial_l245_245459


namespace land_plot_side_length_l245_245154

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end land_plot_side_length_l245_245154


namespace probability_even_sum_l245_245345

def firstTenPrimes : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

def isOdd (n : ℕ) : Prop := n % 2 = 1

def distinctPairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y }

def eventEvenSum (s : Set ℕ) : Set (ℕ × ℕ) :=
  { (x, y) | x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ (x + y) % 2 = 0 }

theorem probability_even_sum :
  (Set.card (eventEvenSum firstTenPrimes)).toRat / (Set.card (distinctPairs firstTenPrimes)).toRat = 4 / 5 :=
  sorry

end probability_even_sum_l245_245345


namespace extremum_f_at_neg_four_thirds_monotonicity_g_l245_245936

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + x ^ 2
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (f a x) * Real.exp x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x ^ 2 + 2 * x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 
  let f_a_x := f a x
  ( f' a x * Real.exp x ) + ( f_a_x * Real.exp x)

theorem extremum_f_at_neg_four_thirds (a : ℝ) :
  f' a (-4/3) = 0 ↔ a = 1/2 := sorry

-- Assuming a = 1/2 from the previous theorem
theorem monotonicity_g :
  let a := 1/2
  ∀ x : ℝ, 
    ((x < -4 → g' a x < 0) ∧ 
     (-4 < x ∧ x < -1 → g' a x > 0) ∧
     (-1 < x ∧ x < 0 → g' a x < 0) ∧
     (x > 0 → g' a x > 0)) := sorry

end extremum_f_at_neg_four_thirds_monotonicity_g_l245_245936


namespace evaluate_x_squared_plus_y_squared_l245_245934

theorem evaluate_x_squared_plus_y_squared (x y : ℚ) (h1 : x + 2 * y = 20) (h2 : 3 * x + y = 19) : x^2 + y^2 = 401 / 5 :=
sorry

end evaluate_x_squared_plus_y_squared_l245_245934


namespace gcd_of_60_and_75_l245_245138

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l245_245138


namespace solutions_to_cube_eq_27_l245_245926

theorem solutions_to_cube_eq_27 (z : ℂ) : 
  (z^3 = 27) ↔ (z = 3 ∨ z = (Complex.mk (-3 / 2) (3 * Real.sqrt 3 / 2)) ∨ z = (Complex.mk (-3 / 2) (-3 * Real.sqrt 3 / 2))) :=
by sorry

end solutions_to_cube_eq_27_l245_245926


namespace trigonometric_identity_application_l245_245859

theorem trigonometric_identity_application :
  2 * (Real.sin (35 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) +
       Real.cos (35 * Real.pi / 180) * Real.cos (65 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end trigonometric_identity_application_l245_245859


namespace gcd_60_75_l245_245147

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l245_245147


namespace group_size_of_bananas_l245_245265

theorem group_size_of_bananas (totalBananas numberOfGroups : ℕ) (h1 : totalBananas = 203) (h2 : numberOfGroups = 7) :
  totalBananas / numberOfGroups = 29 :=
sorry

end group_size_of_bananas_l245_245265


namespace find_d_l245_245336

theorem find_d (a d : ℝ) (h : ∀ x : ℝ, (x + 3) * (x + a) = x^2 + d * x + 12) :
  d = 7 :=
sorry

end find_d_l245_245336


namespace diff_x_y_l245_245949

theorem diff_x_y (x y : ℤ) (h1 : x + y = 14) (h2 : x = 37) : x - y = 60 :=
sorry

end diff_x_y_l245_245949


namespace Diane_net_loss_l245_245094

variable (x y a b: ℝ)

axiom h1 : x * a = 65
axiom h2 : y * b = 150

theorem Diane_net_loss : (y * b) - (x * a) = 50 := by
  sorry

end Diane_net_loss_l245_245094


namespace gasoline_price_increase_l245_245262

theorem gasoline_price_increase 
  (P Q : ℝ)
  (h_intends_to_spend : ∃ M, M = P * Q * 1.15)
  (h_reduction : ∃ N, N = Q * (1 - 0.08))
  (h_equation : P * Q * 1.15 = P * (1 + x) * Q * (1 - 0.08)) :
  x = 0.25 :=
by
  sorry

end gasoline_price_increase_l245_245262


namespace arithmetic_sequence_sum_ratio_l245_245090

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end arithmetic_sequence_sum_ratio_l245_245090


namespace train_travel_time_l245_245386

theorem train_travel_time 
  (speed : ℝ := 120) -- speed in kmph
  (distance : ℝ := 80) -- distance in km
  (minutes_in_hour : ℝ := 60) -- conversion factor
  : (distance / speed) * minutes_in_hour = 40 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end train_travel_time_l245_245386


namespace mariela_cards_total_l245_245681

theorem mariela_cards_total : 
  let a := 287.0
  let b := 116
  a + b = 403 := 
by
  sorry

end mariela_cards_total_l245_245681


namespace quadrilateral_iff_segments_lt_half_l245_245426

theorem quadrilateral_iff_segments_lt_half (a b c d : ℝ) (h₁ : a + b + c + d = 1) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : c ≤ d) : 
    (a + b > d) ∧ (a + c > d) ∧ (a + b + c > d) ∧ (b + c > d) ↔ a < 1/2 ∧ b < 1/2 ∧ c < 1/2 ∧ d < 1/2 :=
by
  sorry

end quadrilateral_iff_segments_lt_half_l245_245426


namespace common_ratio_of_geometric_sequence_l245_245263

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a_n (n + 1) = a_n n + d

def is_geometric_sequence (x y z : ℝ) (q : ℝ) : Prop :=
  y^2 = x * z

theorem common_ratio_of_geometric_sequence 
    (a_n : ℕ → ℝ) 
    (h_arith : is_arithmetic_sequence a_n) 
    (a1 a3 a5 : ℝ)
    (h1 : a1 = a_n 1 + 1) 
    (h3 : a3 = a_n 3 + 3) 
    (h5 : a5 = a_n 5 + 5) 
    (h_geom : is_geometric_sequence a1 a3 a5 1) : 
  1 = 1 :=
by
  sorry

end common_ratio_of_geometric_sequence_l245_245263


namespace price_decrease_for_original_price_l245_245686

theorem price_decrease_for_original_price (P : ℝ) (h : P > 0) :
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  decrease = 20 :=
by
  let new_price := 1.25 * P
  let decrease := (new_price - P) / new_price * 100
  sorry

end price_decrease_for_original_price_l245_245686


namespace problem1_xy_xplusy_l245_245858

theorem problem1_xy_xplusy (x y: ℝ) (h1: x * y = 5) (h2: x + y = 6) : x - y = 4 ∨ x - y = -4 := 
sorry

end problem1_xy_xplusy_l245_245858


namespace largest_multiple_of_6_neg_greater_than_neg_150_l245_245677

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end largest_multiple_of_6_neg_greater_than_neg_150_l245_245677


namespace evaluate_expression_l245_245452

theorem evaluate_expression : 5000 * (5000 ^ 3000) = 5000 ^ 3001 := 
by
  sorry

end evaluate_expression_l245_245452


namespace new_class_mean_l245_245619

theorem new_class_mean 
  (n1 n2 : ℕ) 
  (mean1 mean2 : ℝ)
  (students_total : ℕ)
  (total_score1 total_score2 : ℝ)
  (h1 : n1 = 45)
  (h2 : n2 = 5)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : students_total = 50)
  (h6 : total_score1 = n1 * mean1)
  (h7 : total_score2 = n2 * mean2) :
  (total_score1 + total_score2) / students_total = 81 :=
by
  sorry

end new_class_mean_l245_245619


namespace cos_pi_minus_alpha_trigonometric_identity_l245_245201

noncomputable def alpha : ℝ := sorry

-- Condition 1: α is in the third quadrant
def isInThirdQuadrant (α : ℝ) : Prop := π < α ∧ α < (3 * π) / 2

-- Condition 2: 2sin(α) = cos(α)
def condition (α : ℝ) := 2 * Real.sin α = Real.cos α

-- Statement 1: Prove cos(π - α) = 2 * sqrt(5) / 5
theorem cos_pi_minus_alpha (α : ℝ) (h1 : isInThirdQuadrant α) (h2 : condition α) : 
  Real.cos (π - α) = 2 * Real.sqrt 5 / 5 :=
sorry

-- Statement 2: Prove (1 + 2 * sin(α) * sin(π / 2 - α)) / (sin^2(α) - cos^2(α)) = -3
theorem trigonometric_identity (α : ℝ) (h2 : condition α) : 
  (1 + 2 * Real.sin α * Real.sin (π / 2 - α)) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -3 :=
sorry

end cos_pi_minus_alpha_trigonometric_identity_l245_245201


namespace Fermat_numbers_are_not_cubes_l245_245645

def F (n : ℕ) : ℕ := 2^(2^n) + 1

theorem Fermat_numbers_are_not_cubes : ∀ n : ℕ, ¬ ∃ k : ℕ, F n = k^3 :=
by
  sorry

end Fermat_numbers_are_not_cubes_l245_245645


namespace age_of_b_l245_245683

-- Definition of conditions
variable (a b c : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : a + b + c = 12)

-- The statement of the proof problem
theorem age_of_b : b = 4 :=
by {
   sorry
}

end age_of_b_l245_245683


namespace not_divisible_by_5_l245_245792

theorem not_divisible_by_5 (n : ℤ) : ¬ (n^2 - 8) % 5 = 0 :=
by sorry

end not_divisible_by_5_l245_245792


namespace tetrahedron_volume_correct_l245_245957

noncomputable def tetrahedron_volume (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABD_ABC : ℝ) : ℝ :=
  let h_ABD := (2 * area_ABD) / AB
  let h := h_ABD * Real.sin angle_ABD_ABC
  (1 / 3) * area_ABC * h

theorem tetrahedron_volume_correct:
  tetrahedron_volume 3 15 12 (Real.pi / 6) = 20 :=
by
  sorry

end tetrahedron_volume_correct_l245_245957


namespace boys_of_other_communities_l245_245954

axiom total_boys : ℕ
axiom muslim_percentage : ℝ
axiom hindu_percentage : ℝ
axiom sikh_percentage : ℝ

noncomputable def other_boy_count (total_boys : ℕ) 
                                   (muslim_percentage : ℝ) 
                                   (hindu_percentage : ℝ) 
                                   (sikh_percentage : ℝ) : ℝ :=
  let total_percentage := muslim_percentage + hindu_percentage + sikh_percentage
  let other_percentage := 1 - total_percentage
  other_percentage * total_boys

theorem boys_of_other_communities : 
    other_boy_count 850 0.44 0.32 0.10 = 119 :=
  by 
    sorry

end boys_of_other_communities_l245_245954


namespace Masha_initial_ball_count_l245_245976

theorem Masha_initial_ball_count (r w n p : ℕ) (h1 : r + n * w = 101) (h2 : p * r + w = 103) (hn : n ≠ 0) :
  r + w = 51 ∨ r + w = 68 :=
  sorry

end Masha_initial_ball_count_l245_245976


namespace max_rectangle_area_l245_245817

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l245_245817


namespace cuboid_height_l245_245190

theorem cuboid_height (l b A : ℝ) (hl : l = 10) (hb : b = 8) (hA : A = 480) :
  ∃ h : ℝ, A = 2 * (l * b + b * h + l * h) ∧ h = 320 / 36 := by
  sorry

end cuboid_height_l245_245190


namespace prod_of_three_consec_ints_l245_245397

theorem prod_of_three_consec_ints (a : ℤ) (h : a + (a + 1) + (a + 2) = 27) :
  a * (a + 1) * (a + 2) = 720 :=
by
  sorry

end prod_of_three_consec_ints_l245_245397


namespace sophie_perceived_height_in_mirror_l245_245800

noncomputable def inch_to_cm : ℝ := 2.5

noncomputable def sophie_height_in_inches : ℝ := 50

noncomputable def sophie_height_in_cm := sophie_height_in_inches * inch_to_cm

noncomputable def perceived_height := sophie_height_in_cm * 2

theorem sophie_perceived_height_in_mirror : perceived_height = 250 :=
by
  unfold perceived_height
  unfold sophie_height_in_cm
  unfold sophie_height_in_inches
  unfold inch_to_cm
  sorry

end sophie_perceived_height_in_mirror_l245_245800


namespace reflection_sum_coordinates_l245_245101

theorem reflection_sum_coordinates :
  ∀ (C D : ℝ × ℝ), 
  C = (5, -3) →
  D = (5, -C.2) →
  (C.1 + C.2 + D.1 + D.2 = 10) :=
by
  intros C D hC hD
  rw [hC, hD]
  simp
  sorry

end reflection_sum_coordinates_l245_245101


namespace value_of_a_if_1_in_S_l245_245938

variable (a : ℤ)
def S := { x : ℤ | 3 * x + a = 0 }

theorem value_of_a_if_1_in_S (h : 1 ∈ S a) : a = -3 :=
sorry

end value_of_a_if_1_in_S_l245_245938


namespace probability_sum_even_l245_245344

-- Let's define the set of the first ten prime numbers.
def first_ten_primes : Finset ℕ := Finset.of_list [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the proposition to prove:
theorem probability_sum_even :
  let n := first_ten_primes.card in 
  let total_pairs := Finset.card (Finset.powersetLen 2 first_ten_primes) in 
  let even_pairs := Finset.card (Finset.filter (λ s, s.sum % 2 = 0) (Finset.powersetLen 2 first_ten_primes)) in 
  (total_pairs = 45 ∧ even_pairs = 36) →
  (even_pairs / total_pairs = 4 / 5) :=
by
  sorry

end probability_sum_even_l245_245344


namespace no_positive_solution_l245_245625

theorem no_positive_solution (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) :
  ¬ (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) :=
sorry

end no_positive_solution_l245_245625


namespace find_x_l245_245484

-- Definitions of binomial coefficients as conditions
def binomial (n k : ℕ) : ℕ := n.choose k

-- The specific conditions given
def C65_eq_6 : Prop := binomial 6 5 = 6
def C64_eq_15 : Prop := binomial 6 4 = 15

-- The theorem we need to prove: ∃ x, binomial 7 x = 21
theorem find_x (h1 : C65_eq_6) (h2 : C64_eq_15) : ∃ x, binomial 7 x = 21 :=
by
  -- Proof will go here
  sorry

end find_x_l245_245484


namespace total_tiles_cost_is_2100_l245_245965

noncomputable def total_tile_cost : ℕ :=
  let length := 10
  let width := 25
  let tiles_per_sq_ft := 4
  let green_tile_percentage := 0.40
  let cost_per_green_tile := 3
  let cost_per_red_tile := 1.5
  let area := length * width
  let total_tiles := area * tiles_per_sq_ft
  let green_tiles := green_tile_percentage * total_tiles
  let red_tiles := total_tiles - green_tiles
  let cost_green := green_tiles * cost_per_green_tile
  let cost_red := red_tiles * cost_per_red_tile
  cost_green + cost_red

theorem total_tiles_cost_is_2100 : total_tile_cost = 2100 := by 
  sorry

end total_tiles_cost_is_2100_l245_245965


namespace exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l245_245098

theorem exceeding_speed_limit_percentages
  (percentage_A : ℕ) (percentage_B : ℕ) (percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  percentage_A = 30 ∧ percentage_B = 20 ∧ percentage_C = 25 := by
  sorry

theorem overall_exceeding_speed_limit_percentage
  (percentage_A percentage_B percentage_C : ℕ)
  (H_A : percentage_A = 30)
  (H_B : percentage_B = 20)
  (H_C : percentage_C = 25) :
  (percentage_A + percentage_B + percentage_C) / 3 = 25 := by
  sorry

end exceeding_speed_limit_percentages_overall_exceeding_speed_limit_percentage_l245_245098


namespace perimeter_difference_l245_245534

-- Define the height of the screen
def height_of_screen : ℕ := 100

-- Define the side length of the square paper
def side_of_square_paper : ℕ := 20

-- Define the perimeter of the square paper
def perimeter_of_paper : ℕ := 4 * side_of_square_paper

-- Prove the difference between the height of the screen and the perimeter of the paper
theorem perimeter_difference : height_of_screen - perimeter_of_paper = 20 := by
  -- Sorry is used here to skip the actual proof
  sorry

end perimeter_difference_l245_245534


namespace max_rectangle_area_l245_245815

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l245_245815


namespace pyramid_surface_area_l245_245015

-- Definitions based on conditions
def upper_base_edge_length : ℝ := 2
def lower_base_edge_length : ℝ := 4
def side_edge_length : ℝ := 2

-- Problem statement in Lean
theorem pyramid_surface_area :
  let slant_height := Real.sqrt ((side_edge_length ^ 2) - (1 ^ 2))
  let perimeter_base := (4 * upper_base_edge_length) + (4 * lower_base_edge_length)
  let lsa := (perimeter_base * slant_height) / 2
  let total_surface_area := lsa + (upper_base_edge_length ^ 2) + (lower_base_edge_length ^ 2)
  total_surface_area = 10 * Real.sqrt 3 + 20 := sorry

end pyramid_surface_area_l245_245015


namespace probability_greater_than_a_l245_245240

   variable {σ : ℝ} (X : ℝ → ℝ)

   noncomputable def normal_distribution : Prop :=
     ∀ X, X ~ Normal 5 σ

   theorem probability_greater_than_a (h1 : normal_distribution X)
     (h2 : ∀ (a : ℝ), P(X > 10 - a) = 0.4) :
     ∀ (a : ℝ), P(X > a) = 0.6 := by
     sorry
   
end probability_greater_than_a_l245_245240


namespace liangliang_distance_to_school_l245_245781

theorem liangliang_distance_to_school :
  (∀ (t : ℕ), (40 * t = 50 * (t - 5)) → (40 * 25 = 1000)) :=
sorry

end liangliang_distance_to_school_l245_245781


namespace average_weight_calculation_l245_245999

noncomputable def new_average_weight (initial_people : ℕ) (initial_avg_weight : ℝ) 
                                     (new_person_weight : ℝ) (total_people : ℕ) : ℝ :=
  (initial_people * initial_avg_weight + new_person_weight) / total_people

theorem average_weight_calculation :
  new_average_weight 6 160 97 7 = 151 := by
  sorry

end average_weight_calculation_l245_245999


namespace geometric_sequence_value_l245_245959

theorem geometric_sequence_value 
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_condition : a 4 * a 6 * a 8 * a 10 * a 12 = 32) :
  (a 10 ^ 2) / (a 12) = 2 :=
sorry

end geometric_sequence_value_l245_245959


namespace glass_volume_230_l245_245870

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l245_245870


namespace son_present_age_l245_245684

variable (S M : ℕ)

-- Condition 1: M = S + 20
def man_age_relation (S M : ℕ) : Prop := M = S + 20

-- Condition 2: In two years, the man's age will be twice the age of his son
def age_relation_in_two_years (S M : ℕ) : Prop := M + 2 = 2*(S + 2)

theorem son_present_age : 
  ∀ (S M : ℕ), man_age_relation S M → age_relation_in_two_years S M → S = 18 :=
by
  intros S M h1 h2
  sorry

end son_present_age_l245_245684


namespace glass_volume_230_l245_245873

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l245_245873


namespace steve_speed_back_home_l245_245519

-- Define a structure to hold the given conditions:
structure Conditions where
  home_to_work_distance : Float := 35 -- km
  v  : Float -- speed on the way to work in km/h
  additional_stop_time : Float := 0.25 -- hours
  total_weekly_time : Float := 30 -- hours

-- Define the main proposition:
theorem steve_speed_back_home (c: Conditions)
  (h1 : 5 * ((c.home_to_work_distance / c.v) + (c.home_to_work_distance / (2 * c.v))) + 3 * c.additional_stop_time = c.total_weekly_time) :
  2 * c.v = 18 := by
  sorry

end steve_speed_back_home_l245_245519


namespace length_of_first_platform_l245_245019

-- Definitions corresponding to conditions
def length_train := 310
def time_first_platform := 15
def length_second_platform := 250
def time_second_platform := 20

-- Time-speed relationship
def speed_first_platform (L : ℕ) : ℚ := (length_train + L) / time_first_platform
def speed_second_platform : ℚ := (length_train + length_second_platform) / time_second_platform

-- Theorem to prove length of first platform
theorem length_of_first_platform (L : ℕ) (h : speed_first_platform L = speed_second_platform) : L = 110 :=
by
  sorry

end length_of_first_platform_l245_245019


namespace mass_15_implies_age_7_l245_245660

-- Define the mass function m which depends on age a
variable (m : ℕ → ℕ)

-- Define the condition for the mass to be 15 kg
def is_age_when_mass_is_15 (a : ℕ) : Prop :=
  m a = 15

-- The problem statement to be proven
theorem mass_15_implies_age_7 : ∀ a, is_age_when_mass_is_15 m a → a = 7 :=
by
  -- Proof details would follow here
  sorry

end mass_15_implies_age_7_l245_245660


namespace maximize_total_profit_maximize_average_annual_profit_l245_245571

-- Define the profit function
def total_profit (x : ℤ) : ℤ := -x^2 + 18*x - 36

-- Define the average annual profit function
def average_annual_profit (x : ℤ) : ℤ :=
  let y := total_profit x
  y / x

-- Prove the maximum total profit
theorem maximize_total_profit : 
  ∃ x : ℤ, (total_profit x = 45) ∧ (x = 9) := 
  sorry

-- Prove the maximum average annual profit
theorem maximize_average_annual_profit : 
  ∃ x : ℤ, (average_annual_profit x = 6) ∧ (x = 6) :=
  sorry

end maximize_total_profit_maximize_average_annual_profit_l245_245571


namespace english_textbook_cost_l245_245020

variable (cost_english_book : ℝ)

theorem english_textbook_cost :
  let geography_book_cost := 10.50
  let num_books := 35
  let total_order_cost := 630
  (num_books * cost_english_book + num_books * geography_book_cost = total_order_cost) →
  cost_english_book = 7.50 :=
by {
sorry
}

end english_textbook_cost_l245_245020


namespace probability_point_on_line_l245_245617

namespace ProbabilityOfPointOnLine

open ProbabilityTheory

theorem probability_point_on_line (m n : ℕ) (hp : 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) :
    (∃ (m n : ℕ), m + n = 4 ∧ 1 ≤ m ∧ m ≤ 6 ∧ 1 ≤ n ∧ n ≤ 6) →
    (∑' (p : ℕ × ℕ), ite (p.1 + p.2 = 4) 1 0) / 36 = 1 / 12 :=
-- Proof goes here
by
  sorry

end ProbabilityOfPointOnLine

end probability_point_on_line_l245_245617


namespace transformer_minimum_load_l245_245117

-- Define the conditions as hypotheses
def running_current_1 := 40
def running_current_2 := 60
def running_current_3 := 25

def start_multiplier_1 := 2
def start_multiplier_2 := 3
def start_multiplier_3 := 4

def units_1 := 3
def units_2 := 2
def units_3 := 1

def starting_current_1 := running_current_1 * start_multiplier_1
def starting_current_2 := running_current_2 * start_multiplier_2
def starting_current_3 := running_current_3 * start_multiplier_3

def total_starting_current_1 := starting_current_1 * units_1
def total_starting_current_2 := starting_current_2 * units_2
def total_starting_current_3 := starting_current_3 * units_3

def total_combined_minimum_current_load := 
  total_starting_current_1 + total_starting_current_2 + total_starting_current_3

-- The theorem to prove that the total combined minimum current load is 700A
theorem transformer_minimum_load : total_combined_minimum_current_load = 700 := by
  sorry

end transformer_minimum_load_l245_245117


namespace ratio_problem_l245_245944

theorem ratio_problem 
  (a b c d : ℚ)
  (h₁ : a / b = 8)
  (h₂ : c / b = 5)
  (h₃ : c / d = 1 / 3) : 
  d / a = 15 / 8 := 
by 
  sorry

end ratio_problem_l245_245944


namespace remainder_of_polynomial_division_l245_245192

theorem remainder_of_polynomial_division
  (x : ℝ)
  (h : 2 * x - 4 = 0) :
  (8 * x^4 - 18 * x^3 + 6 * x^2 - 4 * x + 30) % (2 * x - 4) = 30 := by
  sorry

end remainder_of_polynomial_division_l245_245192


namespace smallest_m_l245_245450

theorem smallest_m (m : ℕ) (h1 : 7 ≡ 2 [MOD 5]) : 
  (7^m ≡ m^7 [MOD 5]) ↔ (m = 7) :=
by sorry

end smallest_m_l245_245450


namespace compute_expression_l245_245028

theorem compute_expression :
  20 * ((144 / 3) + (36 / 6) + (16 / 32) + 2) = 1130 := sorry

end compute_expression_l245_245028


namespace gcd_of_60_and_75_l245_245137

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l245_245137


namespace polynomial_value_l245_245070

theorem polynomial_value (x y : ℝ) (h : x + 2 * y = 6) : 2 * x + 4 * y - 5 = 7 :=
by
  sorry

end polynomial_value_l245_245070


namespace max_value_eq_two_l245_245236

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end max_value_eq_two_l245_245236


namespace work_together_l245_245695

theorem work_together (A_days B_days : ℕ) (hA : A_days = 8) (hB : B_days = 4)
  (A_work : ℚ := 1 / A_days)
  (B_work : ℚ := 1 / B_days) :
  (A_work + B_work = 3 / 8) :=
by
  rw [hA, hB]
  sorry

end work_together_l245_245695


namespace four_digit_number_divisible_by_18_l245_245315

theorem four_digit_number_divisible_by_18 : ∃ n : ℕ, (n % 2 = 0) ∧ (10 + n) % 9 = 0 ∧ n = 8 :=
by
  sorry

end four_digit_number_divisible_by_18_l245_245315


namespace distance_to_place_l245_245565

variables {r c1 c2 t D : ℝ}

theorem distance_to_place (h : t = (D / (r - c1)) + (D / (r + c2))) :
  D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) :=
by
  have h1 : D * (r + c2) / (r - c1) * (r - c1) = D * (r + c2) := by sorry
  have h2 : D * (r - c1) / (r + c2) * (r + c2) = D * (r - c1) := by sorry
  have h3 : D * (r + c2) = D * (r + c2) := by sorry
  have h4 : D * (r - c1) = D * (r - c1) := by sorry
  have h5 : t * (r - c1) * (r + c2) = D * (r + c2) + D * (r - c1) := by sorry
  have h6 : t * (r^2 - c1 * c2) = D * (2 * r + c2 - c1) := by sorry
  have h7 : D = t * (r^2 - c1 * c2) / (2 * r + c2 - c1) := by sorry
  exact h7

end distance_to_place_l245_245565


namespace total_length_of_sticks_l245_245772

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end total_length_of_sticks_l245_245772


namespace sequence_inequality_l245_245322

-- Define the problem
theorem sequence_inequality (a : ℕ → ℕ) (h0 : ∀ n, 0 < a n) (h1 : a 1 > a 0) (h2 : ∀ n ≥ 2, a n = 3 * a (n-1) - 2 * a (n-2)) : a 100 > 2^99 :=
by
  sorry

end sequence_inequality_l245_245322


namespace odd_n_cube_minus_n_div_by_24_l245_245369

theorem odd_n_cube_minus_n_div_by_24 (n : ℤ) (h_odd : n % 2 = 1) : 24 ∣ (n^3 - n) :=
sorry

end odd_n_cube_minus_n_div_by_24_l245_245369


namespace intersection_of_P_and_Q_l245_245213

noncomputable def P : Set ℝ := {x | 0 < Real.log x / Real.log 8 ∧ Real.log x / Real.log 8 < 2 * (Real.log 3 / Real.log 8)}
noncomputable def Q : Set ℝ := {x | 2 / (2 - x) > 1}

theorem intersection_of_P_and_Q :
  P ∩ Q = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l245_245213


namespace watermelon_and_banana_weight_l245_245127

variables (w b : ℕ)
variables (h1 : 2 * w + b = 8100)
variables (h2 : 2 * w + 3 * b = 8300)

theorem watermelon_and_banana_weight (Hw : w = 4000) (Hb : b = 100) :
  2 * w + b = 8100 ∧ 2 * w + 3 * b = 8300 :=
by
  sorry

end watermelon_and_banana_weight_l245_245127


namespace no_real_roots_contradiction_l245_245122

open Real

variables (a b : ℝ)

theorem no_real_roots_contradiction (h : ∀ x : ℝ, a * x^3 + a * x + b ≠ 0) : false :=
by
  sorry

end no_real_roots_contradiction_l245_245122


namespace powers_greater_than_thresholds_l245_245583

theorem powers_greater_than_thresholds :
  (1.01^2778 > 1000000000000) ∧
  (1.001^27632 > 1000000000000) ∧
  (1.000001^27631000 > 1000000000000) ∧
  (1.01^4165 > 1000000000000000000) ∧
  (1.001^41447 > 1000000000000000000) ∧
  (1.000001^41446000 > 1000000000000000000) :=
by sorry

end powers_greater_than_thresholds_l245_245583


namespace isosceles_right_triangle_quotient_l245_245434

theorem isosceles_right_triangle_quotient (a : ℝ) (h : a > 0) :
  (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
sorry

end isosceles_right_triangle_quotient_l245_245434


namespace max_possible_value_xv_l245_245505

noncomputable def max_xv_distance (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) : ℝ :=
|x - v|

theorem max_possible_value_xv 
  (x y z w v : ℝ)
  (h1 : |x - y| = 1)
  (h2 : |y - z| = 2)
  (h3 : |z - w| = 3)
  (h4 : |w - v| = 5) :
  max_xv_distance x y z w v h1 h2 h3 h4 = 11 :=
sorry

end max_possible_value_xv_l245_245505


namespace smaller_investment_value_l245_245289

theorem smaller_investment_value :
  ∃ (x : ℝ), 0.07 * x + 0.27 * 1500 = 0.22 * (x + 1500) ∧ x = 500 :=
by
  sorry

end smaller_investment_value_l245_245289


namespace Alyssa_cookie_count_l245_245021

/--
  Alyssa had some cookies.
  Aiyanna has 140 cookies.
  Aiyanna has 11 more cookies than Alyssa.
  How many cookies does Alyssa have? 
-/
theorem Alyssa_cookie_count 
  (aiyanna_cookies : ℕ) 
  (more_cookies : ℕ)
  (h1 : aiyanna_cookies = 140)
  (h2 : more_cookies = 11)
  (h3 : aiyanna_cookies = alyssa_cookies + more_cookies) :
  alyssa_cookies = 129 := 
sorry

end Alyssa_cookie_count_l245_245021


namespace largest_multiple_negation_greater_than_neg150_l245_245678

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end largest_multiple_negation_greater_than_neg150_l245_245678


namespace certain_number_is_120_l245_245421

theorem certain_number_is_120 : ∃ certain_number : ℤ, 346 * certain_number = 173 * 240 ∧ certain_number = 120 :=
by
  sorry

end certain_number_is_120_l245_245421


namespace initial_students_l245_245955

variable (x : ℕ) -- let x be the initial number of students

-- each condition defined as a function
def first_round_rem (x : ℕ) : ℕ := (40 * x) / 100
def second_round_rem (x : ℕ) : ℕ := first_round_rem x / 2
def third_round_rem (x : ℕ) : ℕ := second_round_rem x / 4

theorem initial_students (x : ℕ) (h : third_round_rem x = 15) : x = 300 := 
by sorry  -- proof will be inserted here

end initial_students_l245_245955


namespace equal_area_centroid_S_l245_245267

noncomputable def P : ℝ × ℝ := (-4, 3)
noncomputable def Q : ℝ × ℝ := (7, -5)
noncomputable def R : ℝ × ℝ := (0, 6)
noncomputable def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

theorem equal_area_centroid_S (x y : ℝ) (h : (x, y) = centroid P Q R) :
  10 * x + y = 34 / 3 := by
  sorry

end equal_area_centroid_S_l245_245267


namespace new_selling_price_l245_245561

theorem new_selling_price (C : ℝ) (h1 : 1.10 * C = 88) :
  1.15 * C = 92 :=
sorry

end new_selling_price_l245_245561


namespace additional_cost_l245_245301

theorem additional_cost (original_bales : ℕ) (previous_cost_per_bale : ℕ) (better_quality_cost_per_bale : ℕ) :
  let new_bales := original_bales * 2
  let original_cost := original_bales * previous_cost_per_bale
  let new_cost := new_bales * better_quality_cost_per_bale
  new_cost - original_cost = 210 :=
by sorry

end additional_cost_l245_245301


namespace optimal_green_tiles_l245_245916

variable (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ)

def conditions (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ) :=
  n_indigo ≥ n_red + n_orange + n_yellow + n_green + n_blue ∧
  n_blue ≥ n_red + n_orange + n_yellow + n_green ∧
  n_green ≥ n_red + n_orange + n_yellow ∧
  n_yellow ≥ n_red + n_orange ∧
  n_orange ≥ n_red ∧
  n_red + n_orange + n_yellow + n_green + n_blue + n_indigo = 100

theorem optimal_green_tiles : 
  conditions n_red n_orange n_yellow n_green n_blue n_indigo → 
  n_green = 13 := by
    sorry

end optimal_green_tiles_l245_245916


namespace integer_not_always_greater_decimal_l245_245168

-- Definitions based on conditions
def is_decimal (d : ℚ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), 0 ≤ f ∧ f < 1 ∧ d = i + f

def is_greater (a : ℤ) (b : ℚ) : Prop :=
  (a : ℚ) > b

theorem integer_not_always_greater_decimal : ¬ ∀ n : ℤ, ∀ d : ℚ, is_decimal d → (is_greater n d) :=
by
  sorry

end integer_not_always_greater_decimal_l245_245168


namespace percentage_length_more_than_breadth_l245_245995

-- Define the basic conditions
variables {C r l b : ℝ}
variable {p : ℝ}

-- Assume the conditions
def conditions (C r l b : ℝ) : Prop :=
  C = 400 ∧ r = 3 ∧ l = 20 ∧ 20 * b = 400 / 3

-- Define the statement that we want to prove
theorem percentage_length_more_than_breadth (C r l b : ℝ) (h : conditions C r l b) :
  ∃ (p : ℝ), l = b * (1 + p / 100) ∧ p = 200 :=
sorry

end percentage_length_more_than_breadth_l245_245995


namespace first_girl_productivity_higher_l245_245842

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l245_245842


namespace side_length_of_square_l245_245988

-- Define the conditions
def area_rectangle (length width : ℝ) : ℝ := length * width
def area_square (side : ℝ) : ℝ := side * side

-- Given conditions
def rect_length : ℝ := 2
def rect_width : ℝ := 8
def area_of_rectangle : ℝ := area_rectangle rect_length rect_width
def area_of_square : ℝ := area_of_rectangle

-- Main statement to prove
theorem side_length_of_square : ∃ (s : ℝ), s^2 = 16 ∧ s = 4 :=
by {
  -- use the conditions here
  sorry
}

end side_length_of_square_l245_245988


namespace trig_identity_l245_245478

theorem trig_identity :
  let s60 := Real.sin (60 * Real.pi / 180)
  let c1 := Real.cos (1 * Real.pi / 180)
  let c20 := Real.cos (20 * Real.pi / 180)
  let s10 := Real.sin (10 * Real.pi / 180)
  s60 * c1 * c20 - s10 = Real.sqrt 3 / 2 - s10 :=
by
  sorry

end trig_identity_l245_245478


namespace isosceles_triangle_base_angles_l245_245956

theorem isosceles_triangle_base_angles (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : A = B ∨ B = C ∨ C = A) (h₃ : A = 80 ∨ B = 80 ∨ C = 80) :
  A = 50 ∨ B = 50 ∨ C = 50 ∨ A = 80 ∨ B = 80 ∨ C = 80 := 
by
  sorry

end isosceles_triangle_base_angles_l245_245956


namespace minimum_value_expr_l245_245092

noncomputable def expr (x y z : ℝ) : ℝ :=
  (1 / ((1 - x^2) * (1 - y^2) * (1 - z^2))) +
  (1 / ((1 + x^2) * (1 + y^2) * (1 + z^2)))

theorem minimum_value_expr : ∀ (x y z : ℝ), (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → (0 ≤ z ∧ z ≤ 1) →
  expr x y z ≥ 2 :=
by sorry

end minimum_value_expr_l245_245092


namespace probability_of_two_jacob_one_isaac_l245_245626

-- Definition of the problem conditions
def jacob_letters := 5
def isaac_letters := 5
def total_cards := 12
def cards_drawn := 3

-- Combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probability calculation
def probability_two_jacob_one_isaac : ℚ :=
  (C jacob_letters 2 * C isaac_letters 1 : ℚ) / (C total_cards cards_drawn : ℚ)

-- The statement of the problem
theorem probability_of_two_jacob_one_isaac :
  probability_two_jacob_one_isaac = 5 / 22 :=
  by sorry

end probability_of_two_jacob_one_isaac_l245_245626


namespace units_digit_of_result_is_eight_l245_245994

def three_digit_number_reverse_subtract (a b c : ℕ) : ℕ :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  original - reversed

theorem units_digit_of_result_is_eight (a b c : ℕ) (h : a = c + 2) :
  (three_digit_number_reverse_subtract a b c) % 10 = 8 :=
by
  sorry

end units_digit_of_result_is_eight_l245_245994


namespace breadth_of_added_rectangle_l245_245897

theorem breadth_of_added_rectangle 
  (s : ℝ) (b : ℝ) 
  (h_square_side : s = 8) 
  (h_perimeter_new_rectangle : 2 * s + 2 * (s + b) = 40) : 
  b = 4 :=
by
  sorry

end breadth_of_added_rectangle_l245_245897


namespace find_k_l245_245217

-- Definitions for the vectors and collinearity condition.

def vector := ℝ × ℝ

def collinear (v1 v2 : vector) : Prop :=
  v1.1 * v2.2 - v1.2 * v2.1 = 0

-- Given vectors a and b.
def a (k : ℝ) : vector := (1, k)
def b : vector := (2, 2)

-- Vector addition.
def add (v1 v2 : vector) : vector := (v1.1 + v2.1, v1.2 + v2.2)

-- Problem statement
theorem find_k (k : ℝ) (h : collinear (add (a k) b) (a k)) : k = 1 :=
by
  sorry

end find_k_l245_245217


namespace binary_division_remainder_l245_245273

theorem binary_division_remainder (n : ℕ) (h_n : n = 0b110110011011) : n % 8 = 3 :=
by {
  -- This sorry statement skips the actual proof
  sorry
}

end binary_division_remainder_l245_245273


namespace original_total_price_l245_245709

theorem original_total_price (total_selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (selling_price_with_profit : total_selling_price/2 = original_price * (1 + profit_percent))
  (selling_price_with_loss : total_selling_price/2 = original_price * (1 - loss_percent)) :
  (original_price / (1 + profit_percent) + original_price / (1 - loss_percent) = 1333 + 1 / 3) := 
by
  sorry

end original_total_price_l245_245709


namespace hours_to_destination_l245_245731

def num_people := 4
def water_per_person_per_hour := 1 / 2
def total_water_bottles_needed := 32

theorem hours_to_destination : 
  ∃ h : ℕ, (num_people * water_per_person_per_hour * 2 * h = total_water_bottles_needed) → h = 8 :=
by
  sorry

end hours_to_destination_l245_245731


namespace sub_neg_four_l245_245439

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end sub_neg_four_l245_245439


namespace real_part_of_solution_l245_245107

theorem real_part_of_solution (a b : ℝ) (z : ℂ) (h : z = a + b * Complex.I): 
  z * (z + Complex.I) * (z + 2 * Complex.I) = 1800 * Complex.I → a = 20.75 := by
  sorry

end real_part_of_solution_l245_245107


namespace centroid_has_integer_coordinates_l245_245370

open Finset

noncomputable def centroid (p1 p2 p3 : ℤ × ℤ) : ℚ × ℚ :=
  ((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

theorem centroid_has_integer_coordinates (points : Finset (ℤ × ℤ))
  (h_points : points.card = 13)
  (h_no_three_collinear: ∀ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
    (p1.1 - p2.1) * (p1.2 - p3.2) ≠ (p1.1 - p3.1) * (p1.2 - p2.2)) :
  ∃ (p1 p2 p3 : ℤ × ℤ), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ 
    ∃ (c : ℤ × ℤ), centroid p1 p2 p3 = c := 
sorry

end centroid_has_integer_coordinates_l245_245370


namespace largest_rectangle_area_l245_245820

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l245_245820


namespace marching_band_formations_l245_245553

theorem marching_band_formations :
  (∃ (s t : ℕ), s * t = 240 ∧ 8 ≤ t ∧ t ≤ 30) →
  ∃ (z : ℕ), z = 4 := sorry

end marching_band_formations_l245_245553


namespace camp_cedar_counselors_l245_245911

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h1 : boys = 40)
  (h2 : girls = 3 * boys)
  (h3 : total_children = boys + girls)
  (h4 : counselors = total_children / 8) : 
  counselors = 20 :=
by sorry

end camp_cedar_counselors_l245_245911


namespace productivity_comparison_l245_245845

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l245_245845


namespace no_positive_integer_n_eqn_l245_245729

theorem no_positive_integer_n_eqn (n : ℕ) : (120^5 + 97^5 + 79^5 + 44^5 ≠ n^5) ∨ n = 144 :=
by
  -- Proof omitted for brevity
  sorry

end no_positive_integer_n_eqn_l245_245729


namespace evaluate_expression_l245_245183

theorem evaluate_expression : (16 ^ 24) / (64 ^ 8) = 16 ^ 12 :=
by sorry

end evaluate_expression_l245_245183


namespace bob_gave_terry_24_bushels_l245_245296

def bushels_given_to_terry (total_bushels : ℕ) (ears_per_bushel : ℕ) (ears_left : ℕ) : ℕ :=
    (total_bushels * ears_per_bushel - ears_left) / ears_per_bushel

theorem bob_gave_terry_24_bushels : bushels_given_to_terry 50 14 357 = 24 := by
    sorry

end bob_gave_terry_24_bushels_l245_245296


namespace hyperbola_eccentricity_l245_245780

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variables (c e : ℝ)

-- Define the eccentricy condition for hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity :
  -- Conditions regarding the hyperbola and the distances
  (∀ x y : ℝ, hyperbola a b x y → 
    (∃ x y : ℝ, y = (2 / 3) * c ∧ x = 2 * a + (2 / 3) * c ∧
    ((2 / 3) * c)^2 + (2 * a + (2 / 3) * c)^2 = 4 * c^2 ∧
    (7 * e^2 - 6 * e - 9 = 0))) →
  -- Proving that the eccentricity e is as given
  e = (3 + Real.sqrt 6) / 7 :=
sorry

end hyperbola_eccentricity_l245_245780


namespace probability_picasso_consecutive_l245_245785

-- Given Conditions
def total_pieces : Nat := 12
def picasso_paintings : Nat := 4

-- Desired probability calculation
theorem probability_picasso_consecutive :
  (Nat.factorial (total_pieces - picasso_paintings + 1) * Nat.factorial picasso_paintings) / 
  Nat.factorial total_pieces = 1 / 55 :=
by
  sorry

end probability_picasso_consecutive_l245_245785


namespace bernoulli_inequality_l245_245365

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) : 1 + n * x ≤ (1 + x) ^ n :=
sorry

end bernoulli_inequality_l245_245365


namespace number_of_players_taking_mathematics_l245_245292

def total_players : ℕ := 25
def players_taking_physics : ℕ := 12
def players_taking_both : ℕ := 5

theorem number_of_players_taking_mathematics :
  total_players - players_taking_physics + players_taking_both = 18 :=
by
  sorry

end number_of_players_taking_mathematics_l245_245292


namespace train_crosses_signal_pole_in_20_seconds_l245_245557

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 285
noncomputable def total_time_to_cross_platform : ℝ := 39

-- Define the speed of the train
noncomputable def train_speed : ℝ := (train_length + platform_length) / total_time_to_cross_platform

-- Define the expected time to cross the signal pole
noncomputable def time_to_cross_signal_pole : ℝ := train_length / train_speed

theorem train_crosses_signal_pole_in_20_seconds :
  time_to_cross_signal_pole = 20 := by
  sorry

end train_crosses_signal_pole_in_20_seconds_l245_245557


namespace additional_money_needed_l245_245298

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_money_needed_l245_245298


namespace ratio_of_girls_to_boys_l245_245567

theorem ratio_of_girls_to_boys (total_people : ℕ) (girls : ℕ) (boys : ℕ) 
  (h1 : total_people = 96) (h2 : girls = 80) (h3 : boys = total_people - girls) :
  (5 : ℚ) = (girls : ℚ) / (boys : ℚ) :=
by
  sorry

end ratio_of_girls_to_boys_l245_245567


namespace stella_profit_loss_l245_245985

theorem stella_profit_loss :
  let dolls := 6
  let clocks := 4
  let glasses := 8
  let vases := 3
  let postcards := 10
  let dolls_price := 8
  let clocks_price := 25
  let glasses_price := 6
  let vases_price := 12
  let postcards_price := 3
  let cost := 250
  let clocks_discount_threshold := 2
  let clocks_discount := 10 / 100
  let glasses_bundle := 3
  let glasses_bundle_price := 2 * glasses_price
  let sales_tax_rate := 5 / 100
  let dolls_revenue := dolls * dolls_price
  let clocks_revenue_full := clocks * clocks_price
  let clocks_discounts_count := clocks / clocks_discount_threshold
  let clocks_discount_amount := clocks_discounts_count * clocks_discount * clocks_discount_threshold * clocks_price
  let clocks_revenue := clocks_revenue_full - clocks_discount_amount
  let glasses_discount_quantity := glasses / glasses_bundle
  let glasses_revenue := (glasses - glasses_discount_quantity) * glasses_price
  let vases_revenue := vases * vases_price
  let postcards_revenue := postcards * postcards_price
  let total_revenue_without_discounts := dolls_revenue + clocks_revenue_full + glasses_revenue + vases_revenue + postcards_revenue
  let total_revenue_with_discounts := dolls_revenue + clocks_revenue + glasses_revenue + vases_revenue + postcards_revenue
  let sales_tax := sales_tax_rate * total_revenue_with_discounts
  let profit := total_revenue_with_discounts - cost - sales_tax
  profit = -17.25 := by sorry

end stella_profit_loss_l245_245985


namespace min_PA_squared_plus_PB_squared_l245_245599

-- Let points A, B, and the circle be defined as given in the problem.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

def PA_squared (P : Point) : ℝ :=
  (P.x - A.x)^2 + (P.y - A.y)^2

def PB_squared (P : Point) : ℝ :=
  (P.x - B.x)^2 + (P.y - B.y)^2

def F (P : Point) : ℝ := PA_squared P + PB_squared P

theorem min_PA_squared_plus_PB_squared : ∃ P : Point, on_circle P ∧ F P = 26 := sorry

end min_PA_squared_plus_PB_squared_l245_245599


namespace max_area_of_region_S_l245_245953

-- Define the radii of the circles
def radii : List ℕ := [2, 4, 6, 8]

-- Define the function for the maximum area of region S given the conditions
def max_area_region_S : ℕ := 75

-- Prove the maximum area of region S is 75π
theorem max_area_of_region_S {radii : List ℕ} (h : radii = [2, 4, 6, 8]) 
: max_area_region_S = 75 := by 
  sorry

end max_area_of_region_S_l245_245953


namespace Trishul_investment_percentage_l245_245268

-- Definitions from the conditions
def Vishal_invested (T : ℝ) : ℝ := 1.10 * T
def total_investment (T : ℝ) (V : ℝ) : ℝ := T + V + 2000

-- Problem statement
theorem Trishul_investment_percentage (T : ℝ) (V : ℝ) (H1 : V = Vishal_invested T) (H2 : total_investment T V = 5780) :
  ((2000 - T) / 2000) * 100 = 10 :=
sorry

end Trishul_investment_percentage_l245_245268


namespace range_of_a_l245_245600

-- Definitions for propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, ¬(x^2 + (a-1)*x + 1 ≤ 0)

def q (a : ℝ) : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (a - 1)^x₁ < (a - 1)^x₂

-- The final theorem to prove
theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → (-1 < a ∧ a ≤ 2) ∨ (a ≥ 3) :=
by
  sorry

end range_of_a_l245_245600


namespace equation_of_line_through_point_with_equal_intercepts_l245_245309

-- Define a structure for a 2D point
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define the problem-specific points and conditions
def A : Point := {x := 4, y := -1}

-- Define the conditions and the theorem to be proven
theorem equation_of_line_through_point_with_equal_intercepts
  (p : Point)
  (h : p = A) : 
  ∃ (a : ℝ), a ≠ 0 → (∀ (a : ℝ), ((∀ (b : ℝ), b = a → b ≠ 0 → x + y - a = 0)) ∨ (x + 4 * y = 0)) :=
sorry

end equation_of_line_through_point_with_equal_intercepts_l245_245309


namespace sin_identity_l245_245593

theorem sin_identity {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 4) : 
  Real.sin (π / 6 - 2 * α) = -7 / 8 := 
by 
  sorry

end sin_identity_l245_245593


namespace simplify_expression_l245_245258

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : (2 / (x^2 - 1)) / (1 / (x - 1)) = 2 / (x + 1) :=
by sorry

end simplify_expression_l245_245258


namespace necessary_and_sufficient_condition_for_absolute_inequality_l245_245464

theorem necessary_and_sufficient_condition_for_absolute_inequality (a : ℝ) :
  (a < 3) ↔ (∀ x : ℝ, |x + 2| + |x - 1| > a) :=
sorry

end necessary_and_sufficient_condition_for_absolute_inequality_l245_245464


namespace probability_of_even_sum_is_four_fifths_l245_245347

-- Define the first ten prime numbers
def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- The function to calculate the probability that the sum of two distinct primes is even
def probability_even_sum (primes : List ℕ) : ℚ :=
  let pairs := primes.product primes
  let distinct_pairs := pairs.filter (λ p, p.1 ≠ p.2)
  let even_sum_pairs := distinct_pairs.filter (λ p, (p.1 + p.2) % 2 = 0)
  (even_sum_pairs.length : ℚ) / (distinct_pairs.length : ℚ)

-- Prove that the probability is 4/5
theorem probability_of_even_sum_is_four_fifths :
  probability_even_sum first_ten_primes = 4 / 5 := sorry

end probability_of_even_sum_is_four_fifths_l245_245347


namespace glass_volume_l245_245878

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l245_245878


namespace minimum_value_of_C_over_D_is_three_l245_245337

variable (x : ℝ) (C D : ℝ)
variables (hxC : x^3 + 1/(x^3) = C) (hxD : x - 1/(x) = D)

theorem minimum_value_of_C_over_D_is_three (hC : C = D^3 + 3 * D) :
  ∃ x : ℝ, x^3 + 1/(x^3) = C ∧ x - 1/(x) = D → C / D ≥ 3 :=
by
  sorry

end minimum_value_of_C_over_D_is_three_l245_245337


namespace carol_first_toss_six_probability_l245_245431

theorem carol_first_toss_six_probability :
  let p := 1 / 6
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  (prob_carol_first_six / (1 - prob_cycle)) = 125 / 671 :=
by
  let p := (1 / 6:ℚ)
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  have sum_geo_series : prob_carol_first_six / (1 - prob_cycle) = 125 / 671 := sorry
  exact sum_geo_series

end carol_first_toss_six_probability_l245_245431


namespace complex_mul_im_unit_l245_245202

theorem complex_mul_im_unit (i : ℂ) (h : i^2 = -1) : i * (1 - i) = 1 + i := by
  sorry

end complex_mul_im_unit_l245_245202


namespace bird_cost_l245_245489

variable (scost bcost : ℕ)

theorem bird_cost (h1 : bcost = 2 * scost)
                  (h2 : (5 * bcost + 3 * scost) = (3 * bcost + 5 * scost) + 20) :
                  scost = 10 ∧ bcost = 20 :=
by {
  sorry
}

end bird_cost_l245_245489


namespace isosceles_triangle_height_ratio_l245_245259

theorem isosceles_triangle_height_ratio (a b : ℝ) (h₁ : b = (4 / 3) * a) :
  ∃ m n : ℝ, b / 2 = m + n ∧ m = (2 / 3) * a ∧ n = (1 / 3) * a ∧ (m / n) = 2 :=
by
  sorry

end isosceles_triangle_height_ratio_l245_245259


namespace scientific_notation_of_4212000_l245_245950

theorem scientific_notation_of_4212000 :
  4212000 = 4.212 * 10^6 :=
by
  sorry

end scientific_notation_of_4212000_l245_245950


namespace ellipse_foci_on_y_axis_l245_245947

theorem ellipse_foci_on_y_axis (k : ℝ) :
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (∀ x y, x^2 + k * y^2 = 2 ↔ x^2/a^2 + y^2/b^2 = 1) ∧ b^2 > a^2)
  → (0 < k ∧ k < 1) :=
sorry

end ellipse_foci_on_y_axis_l245_245947


namespace average_minutes_run_per_day_l245_245906

theorem average_minutes_run_per_day (e : ℕ)
  (sixth_grade_avg : ℕ := 16)
  (seventh_grade_avg : ℕ := 18)
  (eighth_grade_avg : ℕ := 12)
  (sixth_graders : ℕ := 3 * e)
  (seventh_graders : ℕ := 2 * e)
  (eighth_graders : ℕ := e) :
  ((sixth_grade_avg * sixth_graders + seventh_grade_avg * seventh_graders + eighth_grade_avg * eighth_graders)
   / (sixth_graders + seventh_graders + eighth_graders) : ℕ) = 16 := 
by
  sorry

end average_minutes_run_per_day_l245_245906


namespace vlad_score_l245_245380

theorem vlad_score :
  ∀ (rounds wins : ℕ) (totalPoints taroPoints vladPoints : ℕ),
    rounds = 30 →
    (wins = 5) →
    (totalPoints = rounds * wins) →
    (taroPoints = (3 * totalPoints) / 5 - 4) →
    (vladPoints = totalPoints - taroPoints) →
    vladPoints = 64 :=
by
  intros rounds wins totalPoints taroPoints vladPoints h1 h2 h3 h4 h5
  sorry

end vlad_score_l245_245380


namespace identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l245_245278

-- Proving the identification of the counterfeit coin among 13 coins in 3 weighings
theorem identify_counterfeit_13_coins (coins : Fin 13 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0) :=
sorry

-- Proving counterfeit coin weight determination with an additional genuine coin using 3 weighings
theorem identify_and_determine_weight_14_coins (coins : Fin 14 → Real) (genuine : Real) (is_counterfeit : ∃! i, coins i ≠ genuine) :
  ∃ method_exists : Prop, 
    (method_exists ∧ ∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ (i : Fin 14), coins i ≠ genuine) :=
sorry

-- Proving the impossibility of identifying counterfeit coin among 14 coins using 3 weighings
theorem impossible_with_14_coins (coins : Fin 14 → Real) (is_counterfeit : ∃! i, coins i ≠ coins 0) :
  ¬ (∃ measure_count : ℕ, measure_count <= 3 ∧ 
    ∃ i, (coins i ≠ coins 0)) :=
sorry

end identify_counterfeit_13_coins_identify_and_determine_weight_14_coins_impossible_with_14_coins_l245_245278


namespace octal_to_base5_conversion_l245_245031

-- Define the octal to decimal conversion
def octalToDecimal (n : ℕ) : ℕ :=
  2 * 8^3 + 0 * 8^2 + 1 * 8^1 + 1 * 8^0

-- Define the base-5 number
def base5Representation : ℕ := 13113

-- Theorem statement
theorem octal_to_base5_conversion :
  octalToDecimal 2011 = base5Representation := 
sorry

end octal_to_base5_conversion_l245_245031


namespace largest_multiple_of_6_neg_greater_than_neg_150_l245_245676

theorem largest_multiple_of_6_neg_greater_than_neg_150 : 
  ∃ m : ℤ, m % 6 = 0 ∧ -m > -150 ∧ m = 144 :=
by
  sorry

end largest_multiple_of_6_neg_greater_than_neg_150_l245_245676


namespace train_crossing_time_l245_245687

theorem train_crossing_time 
  (train_length : ℕ) 
  (train_speed_kmph : ℕ) 
  (conversion_factor : ℚ := 1000/3600) 
  (train_speed_mps : ℚ := train_speed_kmph * conversion_factor) :
  train_length = 100 →
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  train_length / train_speed_mps = 5 :=
by
  intros
  sorry

end train_crossing_time_l245_245687


namespace fraction_product_is_simplified_form_l245_245024

noncomputable def fraction_product : ℚ := (2 / 3) * (5 / 11) * (3 / 8)

theorem fraction_product_is_simplified_form :
  fraction_product = 5 / 44 :=
by
  sorry

end fraction_product_is_simplified_form_l245_245024


namespace gina_keeps_170_l245_245591

theorem gina_keeps_170 (initial_amount : ℕ)
    (money_to_mom : ℕ)
    (money_to_clothes : ℕ)
    (money_to_charity : ℕ)
    (remaining_money : ℕ) :
  initial_amount = 400 →
  money_to_mom = (1 / 4) * initial_amount →
  money_to_clothes = (1 / 8) * initial_amount →
  money_to_charity = (1 / 5) * initial_amount →
  remaining_money = initial_amount - (money_to_mom + money_to_clothes + money_to_charity) →
  remaining_money = 170 := sorry

end gina_keeps_170_l245_245591


namespace max_brownie_cakes_l245_245869

theorem max_brownie_cakes (m n : ℕ) (h : (m-2)*(n-2) = (1/2)*m*n) :  m * n ≤ 60 :=
sorry

end max_brownie_cakes_l245_245869


namespace fraction_of_green_balls_l245_245621

theorem fraction_of_green_balls (T G : ℝ)
    (h1 : (1 / 8) * T = 6)
    (h2 : (1 / 12) * T + (1 / 8) * T + 26 = T - G)
    (h3 : (1 / 8) * T = 6)
    (h4 : 26 ≥ 0):
  G / T = 1 / 4 :=
by
  sorry

end fraction_of_green_balls_l245_245621


namespace simplify_expression_correct_l245_245981

noncomputable def simplify_expr (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :=
  ((a / b) * ((b - (4 * (a^6) / b^3)) ^ (1 / 3))
    - a^2 * ((b / a^6 - (4 / b^3)) ^ (1 / 3))
    + (2 / (a * b)) * ((a^3 * b^4 - 4 * a^9) ^ (1 / 3))) /
    ((b^2 - 2 * a^3) ^ (1 / 3) / b^2)

theorem simplify_expression_correct (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  simplify_expr a b ha hb = (a + b) * ((b^2 + 2 * a^3) ^ (1 / 3)) :=
sorry

end simplify_expression_correct_l245_245981


namespace triangle_inequality_sum_zero_l245_245252

theorem triangle_inequality_sum_zero (a b c p q r : ℝ) (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) (hpqr : p + q + r = 0) : a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
by 
  sorry

end triangle_inequality_sum_zero_l245_245252


namespace length_of_AE_l245_245580

variable (A B C D E : Type) [AddGroup A]
variable (AB CD AC AE EC : ℝ)
variable 
  (hAB : AB = 8)
  (hCD : CD = 18)
  (hAC : AC = 20)
  (hEqualAreas : ∀ (AED BEC : Type), (area AED = area BEC) → (AED = BEC))

theorem length_of_AE (hRatio : AE / EC = 4 / 9) (hSum : AC = AE + EC) : AE = 80 / 13 :=
by
  sorry

end length_of_AE_l245_245580


namespace john_website_days_l245_245969

theorem john_website_days
  (monthly_visits : ℕ)
  (cents_per_visit : ℝ)
  (dollars_per_day : ℝ)
  (monthly_visits_eq : monthly_visits = 30000)
  (cents_per_visit_eq : cents_per_visit = 0.01)
  (dollars_per_day_eq : dollars_per_day = 10) :
  (monthly_visits / (dollars_per_day / cents_per_visit)) = 30 :=
by
  sorry

end john_website_days_l245_245969


namespace f_at_five_l245_245700

-- Define the function f with the property given in the condition
axiom f : ℝ → ℝ
axiom f_prop : ∀ x : ℝ, f (3 * x - 1) = x^2 + x + 1

-- Prove that f(5) = 7 given the properties above
theorem f_at_five : f 5 = 7 :=
by
  sorry

end f_at_five_l245_245700


namespace productivity_difference_l245_245847

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l245_245847


namespace gcd_of_60_and_75_l245_245139

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l245_245139


namespace original_price_of_book_l245_245535

theorem original_price_of_book (final_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) 
  (h1 : final_price = 360) (h2 : increase_percentage = 0.20) 
  (h3 : final_price = (1 + increase_percentage) * original_price) : original_price = 300 := 
by
  sorry

end original_price_of_book_l245_245535


namespace truncated_cone_sphere_radius_l245_245899

theorem truncated_cone_sphere_radius :
  ∀ (r1 r2 h : ℝ), 
  r1 = 24 → 
  r2 = 6 → 
  h = 20 → 
  ∃ r, 
  r = 17 * Real.sqrt 2 / 2 := by
  intros r1 r2 h hr1 hr2 hh
  sorry

end truncated_cone_sphere_radius_l245_245899


namespace blue_cards_in_box_l245_245762

theorem blue_cards_in_box (x : ℕ) (h : 0.6 = (x : ℝ) / (x + 8)) : x = 12 :=
sorry

end blue_cards_in_box_l245_245762


namespace scientific_notation_of_169200000000_l245_245658

theorem scientific_notation_of_169200000000 : 169200000000 = 1.692 * 10^11 :=
by sorry

end scientific_notation_of_169200000000_l245_245658


namespace sqrt_of_sqrt_81_l245_245651

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l245_245651


namespace marks_of_A_l245_245152

variable (a b c d e : ℕ)

theorem marks_of_A:
  (a + b + c = 144) →
  (a + b + c + d = 188) →
  (e = d + 3) →
  (b + c + d + e = 192) →
  a = 43 := 
by 
  intros h1 h2 h3 h4
  sorry

end marks_of_A_l245_245152


namespace area_of_annulus_l245_245167

-- Define the conditions
def concentric_circles (r s : ℝ) (h : r > s) (x : ℝ) := 
  r^2 = s^2 + x^2

-- State the theorem
theorem area_of_annulus (r s x : ℝ) (h : r > s) (h₁ : concentric_circles r s h x) :
  π * x^2 = π * r^2 - π * s^2 :=
by 
  rw [concentric_circles] at h₁
  sorry

end area_of_annulus_l245_245167


namespace students_absent_afternoon_l245_245510

theorem students_absent_afternoon
  (morning_registered afternoon_registered total_students morning_absent : ℕ)
  (h_morning_registered : morning_registered = 25)
  (h_morning_absent : morning_absent = 3)
  (h_afternoon_registered : afternoon_registered = 24)
  (h_total_students : total_students = 42) :
  (afternoon_registered - (total_students - (morning_registered - morning_absent))) = 4 :=
by
  sorry

end students_absent_afternoon_l245_245510


namespace timber_volume_after_two_years_correct_l245_245866

-- Definitions based on the conditions in the problem
variables (a p b : ℝ) -- Assume a, p, and b are real numbers

-- Timber volume after one year
def timber_volume_one_year (a p b : ℝ) : ℝ := a * (1 + p) - b

-- Timber volume after two years
def timber_volume_two_years (a p b : ℝ) : ℝ := (timber_volume_one_year a p b) * (1 + p) - b

-- Prove that the timber volume after two years is equal to the given expression
theorem timber_volume_after_two_years_correct (a p b : ℝ) :
  timber_volume_two_years a p b = a * (1 + p)^2 - (2 + p) * b := sorry

end timber_volume_after_two_years_correct_l245_245866


namespace max_min_PA_l245_245052

open Classical

variables (A B P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace P]
          (dist_AB : ℝ) (dist_PA_PB : ℝ)

noncomputable def max_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry
noncomputable def min_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry

theorem max_min_PA (A B : Type) [MetricSpace A] [MetricSpace B] [Inhabited P]
                   (dist_AB : ℝ) (dist_PA_PB : ℝ) :
  dist_AB = 4 → dist_PA_PB = 6 → max_PA A B 4 = 5 ∧ min_PA A B 4 = 1 :=
by
  intros h_AB h_PA_PB
  sorry

end max_min_PA_l245_245052


namespace num_even_digits_in_base7_of_528_is_zero_l245_245046

def is_digit_even_base7 (d : ℕ) : Prop :=
  (d = 0) ∨ (d = 2) ∨ (d = 4) ∨ (d = 6)

def base7_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0] else (List.unfoldr (λ x, if x = 0 then none else some (x % 7, x / 7)) n).reverse

def number_of_even_digits_base7 (n : ℕ) : ℕ :=
  List.countp is_digit_even_base7 (base7_representation n)

theorem num_even_digits_in_base7_of_528_is_zero : number_of_even_digits_base7 528 = 0 :=
by
  sorry

end num_even_digits_in_base7_of_528_is_zero_l245_245046


namespace glass_volume_230_l245_245876

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l245_245876


namespace jackson_pays_2100_l245_245966

def tile_cost (length : ℝ) (width : ℝ) (tiles_per_sqft : ℝ) (percent_green : ℝ) (cost_green : ℝ) (cost_red : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * percent_green
  let red_tiles := total_tiles - green_tiles
  let cost_green_total := green_tiles * cost_green
  let cost_red_total := red_tiles * cost_red
  cost_green_total + cost_red_total

theorem jackson_pays_2100 :
  tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by
  sorry

end jackson_pays_2100_l245_245966


namespace glass_volume_is_230_l245_245889

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l245_245889


namespace tom_batteries_used_total_l245_245125

def batteries_used_in_flashlights : Nat := 2 * 3
def batteries_used_in_toys : Nat := 4 * 5
def batteries_used_in_controllers : Nat := 2 * 6
def total_batteries_used : Nat := batteries_used_in_flashlights + batteries_used_in_toys + batteries_used_in_controllers

theorem tom_batteries_used_total : total_batteries_used = 38 :=
by
  sorry

end tom_batteries_used_total_l245_245125


namespace calculate_expression_l245_245910

theorem calculate_expression : 
  |(-3)| - 2 * Real.tan (Real.pi / 4) + (-1:ℤ)^(2023) - (Real.sqrt 3 - Real.pi)^(0:ℤ) = -1 :=
  by
  sorry

end calculate_expression_l245_245910


namespace find_k_parallel_find_k_perpendicular_l245_245740

noncomputable def veca : (ℝ × ℝ) := (1, 2)
noncomputable def vecb : (ℝ × ℝ) := (-3, 2)

def is_parallel (u v : (ℝ × ℝ)) : Prop := 
  ∃ k : ℝ, k ≠ 0 ∧ u = (k * v.1, k * v.2)

def is_perpendicular (u v : (ℝ × ℝ)) : Prop := 
  u.1 * v.1 + u.2 * v.2 = 0

def calc_vector (k : ℝ) (a b : (ℝ × ℝ)) : (ℝ × ℝ) :=
  (k * a.1 + b.1, k * a.2 + b.2)

theorem find_k_parallel : 
  ∃ k : ℝ, is_parallel (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

theorem find_k_perpendicular :
  ∃ k : ℝ, k = 25 / 3 ∧ is_perpendicular (calc_vector k veca vecb) (calc_vector 1 veca (-2 * vecb)) := sorry

end find_k_parallel_find_k_perpendicular_l245_245740


namespace geometric_sequence_problem_l245_245764

noncomputable def a : ℕ → ℝ := sorry

theorem geometric_sequence_problem :
  a 4 = 4 →
  a 8 = 8 →
  a 12 = 16 :=
by
  intros h4 h8
  sorry

end geometric_sequence_problem_l245_245764


namespace angle_A_l245_245077

variable (a b c : ℝ) (A B C : ℝ)

-- Hypothesis: In triangle ABC, (a + c)(a - c) = b(b + c)
def condition (a b c : ℝ) : Prop := (a + c) * (a - c) = b * (b + c)

-- The goal is to show that under given conditions, ∠A = 2π/3
theorem angle_A (h : condition a b c) : A = 2 * π / 3 :=
sorry

end angle_A_l245_245077


namespace max_rectangle_area_l245_245814

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l245_245814


namespace glass_volume_l245_245883

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l245_245883


namespace Lindas_savings_l245_245244

theorem Lindas_savings (S : ℝ) (h1 : (1/3) * S = 250) : S = 750 := 
by
  sorry

end Lindas_savings_l245_245244


namespace additional_money_needed_l245_245299

def original_num_bales : ℕ := 10
def original_cost_per_bale : ℕ := 15
def new_cost_per_bale : ℕ := 18

theorem additional_money_needed :
  (2 * original_num_bales * new_cost_per_bale) - (original_num_bales * original_cost_per_bale) = 210 :=
by
  sorry

end additional_money_needed_l245_245299


namespace solve_quadratic_eq_l245_245517

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end solve_quadratic_eq_l245_245517


namespace secretaries_ratio_l245_245671

theorem secretaries_ratio (A B C : ℝ) (hA: A = 75) (h_total: A + B + C = 120) : B + C = 45 :=
by {
  -- sorry: We define this part to be explored by the theorem prover
  sorry
}

end secretaries_ratio_l245_245671


namespace line_bisects_circle_perpendicular_l245_245935

theorem line_bisects_circle_perpendicular :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, x^2 + y^2 + x - 2*y + 1 = 0 → l x = y)
               ∧ (∀ x y : ℝ, x + 2*y + 3 = 0 → x ∈ { x | ∃ k:ℝ, y = -1/2 * k + l x})
               ∧ (∀ x y : ℝ, l x = 2 * x - 2)) :=
sorry

end line_bisects_circle_perpendicular_l245_245935


namespace least_even_perimeter_l245_245996

def triangle_perimeter (a b c : ℕ) : ℕ := a + b + c

theorem least_even_perimeter
  (a b : ℕ) (h1 : a = 24) (h2 : b = 37) (c : ℕ)
  (h3 : c > b) (h4 : a + b > c)
  (h5 : ∃ k : ℕ, k * 2 = triangle_perimeter a b c) :
  triangle_perimeter a b c = 100 :=
sorry

end least_even_perimeter_l245_245996


namespace sum_of_digits_of_N_l245_245898

open Nat

theorem sum_of_digits_of_N (T : ℕ) (hT : T = 3003) :
  ∃ N : ℕ, (N * (N + 1)) / 2 = T ∧ (digits 10 N).sum = 14 :=
by 
  sorry

end sum_of_digits_of_N_l245_245898


namespace nonagon_arithmetic_mean_property_l245_245022

open Real

theorem nonagon_arithmetic_mean_property :
  ∃ (f : Fin 9 → ℝ), 
  (∀ i, f i = 2016 + i) ∧ (
    ∀ i j k : Fin 9, 
    (j = (i + 3) % 9.toFin) ∧ (k = (i + 6) % 9.toFin) →
    f j = (f i + f k) / 2
  ) := 
sorry

end nonagon_arithmetic_mean_property_l245_245022


namespace no_preimage_iff_lt_one_l245_245748

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem no_preimage_iff_lt_one (k : ℝ) :
  (∀ x : ℝ, f x ≠ k) ↔ k < 1 := 
by
  sorry

end no_preimage_iff_lt_one_l245_245748


namespace janet_extra_cost_l245_245233

theorem janet_extra_cost :
  let clarinet_hourly_rate := 40
  let clarinet_hours_per_week := 3
  let clarinet_weeks_per_year := 50
  let clarinet_yearly_cost := clarinet_hourly_rate * clarinet_hours_per_week * clarinet_weeks_per_year

  let piano_hourly_rate := 28
  let piano_hours_per_week := 5
  let piano_weeks_per_year := 50
  let piano_yearly_cost := piano_hourly_rate * piano_hours_per_week * piano_weeks_per_year
  let piano_discount_rate := 0.10
  let piano_discounted_yearly_cost := piano_yearly_cost * (1 - piano_discount_rate)

  let violin_hourly_rate := 35
  let violin_hours_per_week := 2
  let violin_weeks_per_year := 50
  let violin_yearly_cost := violin_hourly_rate * violin_hours_per_week * violin_weeks_per_year
  let violin_discount_rate := 0.15
  let violin_discounted_yearly_cost := violin_yearly_cost * (1 - violin_discount_rate)

  let singing_hourly_rate := 45
  let singing_hours_per_week := 1
  let singing_weeks_per_year := 50
  let singing_yearly_cost := singing_hourly_rate * singing_hours_per_week * singing_weeks_per_year

  let combined_cost := piano_discounted_yearly_cost + violin_discounted_yearly_cost + singing_yearly_cost
  combined_cost - clarinet_yearly_cost = 5525 := 
  sorry

end janet_extra_cost_l245_245233


namespace round_trip_ticket_percentage_l245_245275

variable (P : ℝ) -- Denotes total number of passengers
variable (R : ℝ) -- Denotes number of round-trip ticket holders

-- Condition 1: 15% of passengers held round-trip tickets and took their cars aboard
def condition1 : Prop := 0.15 * P = 0.40 * R

-- Prove that 37.5% of the ship's passengers held round-trip tickets.
theorem round_trip_ticket_percentage (h1 : condition1 P R) : R / P = 0.375 :=
by
  sorry

end round_trip_ticket_percentage_l245_245275


namespace glass_volume_correct_l245_245891

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l245_245891


namespace reciprocal_of_neg_three_l245_245860

theorem reciprocal_of_neg_three : (1:ℝ) / (-3:ℝ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg_three_l245_245860


namespace probability_even_sum_l245_245348

-- Definition of the prime numbers set and the selection scenario
def firstTenPrimes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Condition stating that we select two distinct numbers at random from the first ten primes
def randomSelection (s : List ℕ) := {x : ℕ × ℕ // x.1 ≠ x.2 ∧ x.1 ∈ s ∧ x.2 ∈ s}

-- Define the event that their sum is even
def evenSum (p : ℕ × ℕ) := (p.1 + p.2) % 2 = 0

-- Define the probability calculation
def probabilityEvenSum := 
  let totalPairs := (firstTenPrimes.length.choose 2) -- Calculate the number of ways to choose 2 numbers from the 10 primes
  let evenPairs := (randomSelection firstTenPrimes).count (λ p => evenSum p.val)
  evenPairs.toFloat / totalPairs.toFloat

-- Theorem statement that the probability of their sum being even is 1/5
theorem probability_even_sum : probabilityEvenSum = 1/5 := by
  sorry -- The actual proof is to be provided

end probability_even_sum_l245_245348


namespace parallelogram_diagonal_length_l245_245525

-- Define a structure to represent a parallelogram
structure Parallelogram :=
  (side_length : ℝ) 
  (diagonal_length : ℝ)
  (perpendicular : Bool)

-- State the theorem about the relationship between the diagonals in a parallelogram
theorem parallelogram_diagonal_length (a b : ℝ) (P : Parallelogram) (h₀ : P.side_length = a) (h₁ : P.diagonal_length = b) (h₂ : P.perpendicular = true) : 
  ∃ (AC : ℝ), AC = Real.sqrt (4 * a^2 + b^2) :=
by
  sorry

end parallelogram_diagonal_length_l245_245525


namespace fifth_largest_divisor_l245_245578

theorem fifth_largest_divisor (n : ℕ) (h : n = 1020000000) : 
    nat.nth_largest_divisor n 5 = 63750000 :=
by
  rw h
  sorry

end fifth_largest_divisor_l245_245578


namespace correct_calculation_l245_245399

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l245_245399


namespace actual_distance_traveled_l245_245338

theorem actual_distance_traveled (D : ℝ) (T : ℝ) (h1 : D = 15 * T) (h2 : D + 35 = 25 * T) : D = 52.5 := 
by
  sorry

end actual_distance_traveled_l245_245338


namespace simplify_expression_l245_245025

theorem simplify_expression (x : ℝ) (h : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x :=
by 
  -- We provide 'sorry' hack to skip the proof
  -- Replace this with the actual proof to ensure correctness.
  sorry

end simplify_expression_l245_245025


namespace A_greater_than_B_l245_245608

theorem A_greater_than_B (A B : ℝ) (h₁ : A * 4 = B * 5) (h₂ : A ≠ 0) (h₃ : B ≠ 0) : A > B :=
by
  sorry

end A_greater_than_B_l245_245608


namespace emilia_cartons_total_l245_245040

theorem emilia_cartons_total (strawberries blueberries supermarket : ℕ) (total_needed : ℕ)
  (h1 : strawberries = 2)
  (h2 : blueberries = 7)
  (h3 : supermarket = 33)
  (h4 : total_needed = strawberries + blueberries + supermarket) :
  total_needed = 42 :=
sorry

end emilia_cartons_total_l245_245040


namespace even_of_even_square_sqrt_two_irrational_l245_245416

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end even_of_even_square_sqrt_two_irrational_l245_245416


namespace min_value_of_function_l245_245457

noncomputable def func (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + Real.sin (2 * x)

theorem min_value_of_function : ∃ x : ℝ, func x = 1 - Real.sqrt 2 :=
by sorry

end min_value_of_function_l245_245457


namespace problem_correct_calculation_l245_245400

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end problem_correct_calculation_l245_245400


namespace maximum_area_of_garden_l245_245293

noncomputable def max_area (perimeter : ℕ) : ℕ :=
  let half_perimeter := perimeter / 2
  let x := half_perimeter / 2
  x * x

theorem maximum_area_of_garden :
  max_area 148 = 1369 :=
by
  sorry

end maximum_area_of_garden_l245_245293


namespace Anchuria_min_crooks_l245_245493

noncomputable def min_number_of_crooks : ℕ :=
  91

theorem Anchuria_min_crooks (H : ℕ) (C : ℕ) (total_ministers : H + C = 100)
  (ten_minister_condition : ∀ (n : ℕ) (A : Finset ℕ), A.card = 10 → ∃ x ∈ A, ¬ x ∈ H) :
  C ≥ min_number_of_crooks :=
sorry

end Anchuria_min_crooks_l245_245493


namespace compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l245_245914

theorem compare_neg5_neg2 : -5 < -2 :=
by sorry

theorem compare_neg_third_neg_half : -(1/3) > -(1/2) :=
by sorry

theorem compare_absneg5_0 : abs (-5) > 0 :=
by sorry

end compare_neg5_neg2_compare_neg_third_neg_half_compare_absneg5_0_l245_245914


namespace probability_of_specific_sequence_l245_245664

variable (p : ℝ) (h : 0 < p ∧ p < 1)

theorem probability_of_specific_sequence :
  (1 - p)^7 * p^3 = sorry :=
by sorry

end probability_of_specific_sequence_l245_245664


namespace total_distance_run_l245_245766

-- Define the distances run each day based on the distance run on Monday (x)
def distance_on_monday (x : ℝ) := x
def distance_on_tuesday (x : ℝ) := 2 * x
def distance_on_wednesday (x : ℝ) := x
def distance_on_thursday (x : ℝ) := (1/2) * x
def distance_on_friday (x : ℝ) := x

-- Define the condition for the shortest distance
def shortest_distance_condition (x : ℝ) :=
  min (distance_on_monday x)
    (min (distance_on_tuesday x)
      (min (distance_on_wednesday x)
        (min (distance_on_thursday x) 
          (distance_on_friday x)))) = 5

-- State and prove the total distance run over the week
theorem total_distance_run (x : ℝ) (hx : shortest_distance_condition x) : 
  distance_on_monday x + distance_on_tuesday x + distance_on_wednesday x + distance_on_thursday x + distance_on_friday x = 55 :=
by
  sorry

end total_distance_run_l245_245766


namespace min_value_ge_54_l245_245504

open Real

noncomputable def min_value (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) : ℝ :=
2 * x + 3 * y + 6 * z

theorem min_value_ge_54 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 27) :
  min_value x y z h1 h2 h3 h4 ≥ 54 :=
sorry

end min_value_ge_54_l245_245504


namespace train_length_l245_245570

theorem train_length
  (t1 : ℕ) (t2 : ℕ)
  (d_platform : ℕ)
  (h1 : t1 = 8)
  (h2 : t2 = 20)
  (h3 : d_platform = 279)
  : ∃ (L : ℕ), (L : ℕ) = 186 :=
by
  sorry

end train_length_l245_245570


namespace find_m_l245_245779

open Set

def A (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 2}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem find_m (m : ℝ) :
  (A m ∩ B = ∅ ∧ A m ∪ B = B) ↔ (m ≤ -2 ∨ m ≥ 3) :=
by
  sorry

end find_m_l245_245779


namespace polynomial_sum_l245_245805

theorem polynomial_sum :
  let f := (x^3 + 9*x^2 + 26*x + 24) 
  let g := (x + 3)
  let A := 1
  let B := 6
  let C := 8
  let D := -3
  (y = f/g) → (A + B + C + D = 12) :=
by 
  sorry

end polynomial_sum_l245_245805


namespace x_squared_minus_y_squared_l245_245221

theorem x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 9 / 16)
  (h2 : x - y = 5 / 16) :
  x^2 - y^2 = 45 / 256 :=
by
  sorry

end x_squared_minus_y_squared_l245_245221


namespace smallest_number_divisible_l245_245393

/-- The smallest number which, when diminished by 20, is divisible by 15, 30, 45, and 60 --/
theorem smallest_number_divisible (n : ℕ) (h : ∀ k : ℕ, n - 20 = k * Int.lcm 15 (Int.lcm 30 (Int.lcm 45 60))) : n = 200 :=
sorry

end smallest_number_divisible_l245_245393


namespace magic_ink_combinations_l245_245900

def herbs : ℕ := 4
def essences : ℕ := 6
def incompatible_herbs : ℕ := 3

theorem magic_ink_combinations :
  herbs * essences - incompatible_herbs = 21 := 
  by
  sorry

end magic_ink_combinations_l245_245900


namespace gcd_3_1200_1_3_1210_1_l245_245675

theorem gcd_3_1200_1_3_1210_1 : 
  Int.gcd (3^1200 - 1) (3^1210 - 1) = 59048 := 
by 
  sorry

end gcd_3_1200_1_3_1210_1_l245_245675


namespace rectangle_area_l245_245163

theorem rectangle_area (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x * y = 5 :=
by
  -- Conditions given to us:
  -- 1. (h1) The sum of the sides is 5.
  -- 2. (h2) The sum of the squares of the sides is 15.
  -- We need to prove that the product of the sides is 5.
  sorry

end rectangle_area_l245_245163


namespace first_girl_productivity_higher_l245_245841

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l245_245841


namespace range_of_objective_function_l245_245604

def objective_function (x y : ℝ) : ℝ := 3 * x - y

theorem range_of_objective_function (x y : ℝ) 
  (h1 : x + 2 * y ≥ 2)
  (h2 : 2 * x + y ≤ 4)
  (h3 : 4 * x - y ≥ -1)
  : - 3 / 2 ≤ objective_function x y ∧ objective_function x y ≤ 6 := 
sorry

end range_of_objective_function_l245_245604


namespace maximum_value_a_plus_b_cubed_plus_c_fourth_l245_245239

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end maximum_value_a_plus_b_cubed_plus_c_fourth_l245_245239


namespace savings_duration_before_investment_l245_245159

---- Definitions based on conditions ----
def weekly_savings_wife : ℕ := 100
def monthly_savings_husband : ℕ := 225
def stock_price_per_share : ℕ := 50
def shares_bought : ℕ := 25

---- Derived conditions based on definitions ----
def total_spent_on_stocks := shares_bought * stock_price_per_share
def total_savings_before_investment := 2 * total_spent_on_stocks
def monthly_savings_wife := weekly_savings_wife * 4
def total_monthly_savings := monthly_savings_wife + monthly_savings_husband

---- The theorem statement ----
theorem savings_duration_before_investment :
  total_savings_before_investment / total_monthly_savings = 4 :=
sorry

end savings_duration_before_investment_l245_245159


namespace problem_statement_l245_245323

variable {x y : ℝ}

theorem problem_statement 
  (h1 : y > x)
  (h2 : x > 0)
  (h3 : x + y = 1) :
  x < 2 * x * y ∧ 2 * x * y < (x + y) / 2 ∧ (x + y) / 2 < y := by
  sorry

end problem_statement_l245_245323


namespace complement_intersection_l245_245487

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {2, 3}
def N : Set Nat := {1, 4}

theorem complement_intersection :
  (U \ M) ∩ (U \ N) = {5, 6} := by
  -- Proof is omitted.
  sorry

end complement_intersection_l245_245487


namespace geometric_sequence_solution_l245_245960

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, a n = a1 * r ^ (n - 1)

theorem geometric_sequence_solution :
  ∀ (a : ℕ → ℝ),
    (geometric_sequence a) →
    (∃ a2 a18, a2 + a18 = -6 ∧ a2 * a18 = 4 ∧ a 2 = a2 ∧ a 18 = a18) →
    a 4 * a 16 + a 10 = 6 :=
by
  sorry

end geometric_sequence_solution_l245_245960


namespace gcf_60_75_l245_245135

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l245_245135


namespace range_of_a_l245_245075

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 :=
by
  intro h
  sorry

end range_of_a_l245_245075


namespace divisibility_of_a81_l245_245853

theorem divisibility_of_a81 
  (p : ℕ) (hp : Nat.Prime p) (hp_gt2 : 2 < p)
  (a : ℕ → ℕ) (h_rec : ∀ n, n * a (n + 1) = (n + 1) * a n - (p / 2)^4) 
  (h_a1 : a 1 = 5) :
  16 ∣ a 81 := 
sorry

end divisibility_of_a81_l245_245853


namespace even_function_and_inverse_property_l245_245479

noncomputable def f (x : ℝ) : ℝ := (1 + x^2) / (1 - x^2)

theorem even_function_and_inverse_property (x : ℝ) (hx : x ≠ 1 ∧ x ≠ -1) :
  f (-x) = f x ∧ f (1 / x) = -f x := by
  sorry

end even_function_and_inverse_property_l245_245479


namespace solve_inequality_l245_245589

theorem solve_inequality (x : ℝ) : (x^2 - 50 * x + 625 ≤ 25) = (20 ≤ x ∧ x ≤ 30) :=
sorry

end solve_inequality_l245_245589


namespace simplify_expression_l245_245798

theorem simplify_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 9 / Real.log 18 + 1)) = 7 / 4 := 
sorry

end simplify_expression_l245_245798


namespace factorize_expression_l245_245042

theorem factorize_expression (x : ℝ) : -2 * x^2 + 2 * x - (1 / 2) = -2 * (x - (1 / 2))^2 :=
by
  sorry

end factorize_expression_l245_245042


namespace roots_range_l245_245997

theorem roots_range (b : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + b = 0 → 0 < x) ↔ 0 < b ∧ b ≤ 1 :=
sorry

end roots_range_l245_245997


namespace difference_in_max_min_distance_from_circle_to_line_l245_245023

noncomputable def circle_center (x y : ℝ) : ℝ × ℝ := (2, 2)
noncomputable def circle_radius : ℝ := 3 * Real.sqrt 2

def point_to_line_distance (x₀ y₀ A B C : ℝ) : ℝ :=
  abs (A * x₀ + B * y₀ + C) / Real.sqrt (A^2 + B^2)

theorem difference_in_max_min_distance_from_circle_to_line :
  let A := 1
  let B := 1
  let C := -14
  let x₀ := 2
  let y₀ := 2
  point_to_line_distance x₀ y₀ A B C = 5 * Real.sqrt 2 → 
  2 * circle_radius = 6 * Real.sqrt 2
:=
by
  sorry

end difference_in_max_min_distance_from_circle_to_line_l245_245023


namespace purchased_both_books_l245_245533

theorem purchased_both_books: 
  ∀ (A B AB C : ℕ), A = 2 * B → AB = 2 * (B - AB) → C = 1000 → C = A - AB → AB = 500 := 
by
  intros A B AB C h1 h2 h3 h4
  sorry

end purchased_both_books_l245_245533


namespace orthogonal_trajectory_eqn_l245_245044

theorem orthogonal_trajectory_eqn (a C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 2 * a * x) → 
  (∃ C : ℝ, ∀ x y : ℝ, x^2 + y^2 = C * y) :=
sorry

end orthogonal_trajectory_eqn_l245_245044


namespace primes_diff_power_of_two_divisible_by_three_l245_245009

theorem primes_diff_power_of_two_divisible_by_three
  (p q : ℕ) (m n : ℕ)
  (hp : Prime p) (hq : Prime q) (hp_gt : p > 3) (hq_gt : q > 3)
  (diff : q - p = 2^n ∨ p - q = 2^n) :
  3 ∣ (p^(2*m+1) + q^(2*m+1)) := by
  sorry

end primes_diff_power_of_two_divisible_by_three_l245_245009


namespace tile_count_l245_245894

theorem tile_count (room_length room_width tile_length tile_width : ℝ)
  (h1 : room_length = 10)
  (h2 : room_width = 15)
  (h3 : tile_length = 1 / 4)
  (h4 : tile_width = 3 / 4) :
  (room_length * room_width) / (tile_length * tile_width) = 800 :=
by
  sorry

end tile_count_l245_245894


namespace find_smaller_number_l245_245803

theorem find_smaller_number (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) : y = 18.5 :=
by
  sorry

end find_smaller_number_l245_245803


namespace janice_purchases_l245_245769

theorem janice_purchases (a b c : ℕ) : 
  a + b + c = 50 ∧ 30 * a + 200 * b + 300 * c = 5000 → a = 10 :=
sorry

end janice_purchases_l245_245769


namespace min_sine_range_l245_245198

noncomputable def min_sine_ratio (α β γ : ℝ) := min (Real.sin β / Real.sin α) (Real.sin γ / Real.sin β)

theorem min_sine_range (α β γ : ℝ) (h1 : 0 < α) (h2 : α ≤ β) (h3 : β ≤ γ) (h4 : α + β + γ = Real.pi) :
  1 ≤ min_sine_ratio α β γ ∧ min_sine_ratio α β γ < (1 + Real.sqrt 5) / 2 :=
by
  sorry

end min_sine_range_l245_245198


namespace complex_poly_root_exists_l245_245778

noncomputable def polynomial_has_complex_root (P : Polynomial ℂ) : Prop :=
  ∃ z : ℂ, P.eval z = 0

theorem complex_poly_root_exists (P : Polynomial ℂ) : polynomial_has_complex_root P :=
sorry

end complex_poly_root_exists_l245_245778


namespace equal_distribution_l245_245837

theorem equal_distribution 
  (total_profit : ℕ) 
  (num_employees : ℕ) 
  (profit_kept_percent : ℕ) 
  (remaining_to_distribute : ℕ)
  (each_employee_gets : ℕ) :
  total_profit = 50 →
  num_employees = 9 →
  profit_kept_percent = 10 →
  remaining_to_distribute = total_profit - (total_profit * profit_kept_percent / 100) →
  each_employee_gets = remaining_to_distribute / num_employees →
  each_employee_gets = 5 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_distribution_l245_245837


namespace faces_of_prism_with_24_edges_l245_245286

theorem faces_of_prism_with_24_edges (L : ℕ) (h1 : 3 * L = 24) : L + 2 = 10 := by
  sorry

end faces_of_prism_with_24_edges_l245_245286


namespace min_value_expr_min_max_value_expr_max_l245_245335

noncomputable def min_value_expr (a b : ℝ) : ℝ := 
  1 / (a - b) + 4 / (b - 1)

noncomputable def max_value_expr (a b : ℝ) : ℝ :=
  a * b - b^2 - a + b

theorem min_value_expr_min (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) : 
  min_value_expr a b = 25 :=
sorry

theorem max_value_expr_max (a b : ℝ) (h1 : a > b) (h2 : b > 1) (h3 : a + 3 * b = 5) :
  max_value_expr a b = 1 / 16 :=
sorry

end min_value_expr_min_max_value_expr_max_l245_245335


namespace expand_expression_l245_245184

theorem expand_expression (x : ℝ) : 3 * (8 * x^2 - 2 * x + 1) = 24 * x^2 - 6 * x + 3 :=
by
  sorry

end expand_expression_l245_245184


namespace height_of_larger_box_l245_245228

/-- Define the dimensions of the larger box and smaller boxes, 
    and show that given the constraints, the height of the larger box must be 4 meters.-/
theorem height_of_larger_box 
  (L H : ℝ) (V_small : ℝ) (N_small : ℕ) (h : ℝ) 
  (dim_large : L = 6) (width_large : H = 5)
  (vol_small : V_small = 0.6 * 0.5 * 0.4) 
  (num_boxes : N_small = 1000) 
  (vol_large : 6 * 5 * h = N_small * V_small) : 
  h = 4 :=
by 
  sorry

end height_of_larger_box_l245_245228


namespace math_problem_l245_245722

theorem math_problem : (2 + (2 / 3 : ℚ) + 6.3 - ((5 / 3 : ℚ) - (1 + (3 / 5 : ℚ)))) = 8.9 := 
by
  norm_num
-- If there's any simplification required, such as converting 6.3 to (63 / 10 : ℚ), it can be added.

end math_problem_l245_245722


namespace decimal_equivalent_of_fraction_squared_l245_245692

theorem decimal_equivalent_of_fraction_squared : (1 / 4 : ℝ) ^ 2 = 0.0625 :=
by sorry

end decimal_equivalent_of_fraction_squared_l245_245692


namespace cistern_depth_l245_245562

noncomputable def length : ℝ := 9
noncomputable def width : ℝ := 4
noncomputable def total_wet_surface_area : ℝ := 68.5

theorem cistern_depth (h : ℝ) (h_def : 68.5 = 36 + 18 * h + 8 * h) : h = 1.25 :=
by sorry

end cistern_depth_l245_245562


namespace pencils_given_out_l245_245118

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end pencils_given_out_l245_245118


namespace sum_original_numbers_is_five_l245_245710

noncomputable def sum_original_numbers (a b c d : ℤ) : ℤ :=
  a + b + c + d

theorem sum_original_numbers_is_five (a b c d : ℤ) (hab : 10 * a + b = overline_ab) 
  (h : 100 * (10 * a + b) + 10 * c + 7 * d = 2024) : sum_original_numbers a b c d = 5 :=
sorry

end sum_original_numbers_is_five_l245_245710


namespace largest_rectangle_area_l245_245812

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l245_245812


namespace find_x_l245_245180

theorem find_x (x : ℝ) (y : ℝ) : 
  (10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) ↔ x = (3 / 2) :=
by
  sorry

end find_x_l245_245180


namespace log_expression_equals_eight_l245_245394

theorem log_expression_equals_eight :
  (Real.log 4 / Real.log 10) + 
  2 * (Real.log 5 / Real.log 10) + 
  3 * (Real.log 2 / Real.log 10) + 
  6 * (Real.log 5 / Real.log 10) + 
  (Real.log 8 / Real.log 10) = 8 := 
by 
  sorry

end log_expression_equals_eight_l245_245394


namespace solve_system_of_equations_l245_245110

theorem solve_system_of_equations :
  (∃ x y : ℚ, 2 * x + 4 * y = 9 ∧ 3 * x - 5 * y = 8) ↔ 
  (∃ x y : ℚ, x = 7 / 2 ∧ y = 1 / 2) := by
  sorry

end solve_system_of_equations_l245_245110


namespace total_children_estimate_l245_245928

theorem total_children_estimate (k m n : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : k > 0) 
(h4 : n ≤ m) (h5 : n ≤ k) (h6 : m ≤ k) :
  (∃ (total : ℕ), total = k * m / n) :=
sorry

end total_children_estimate_l245_245928


namespace productivity_comparison_l245_245852

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l245_245852


namespace red_balls_count_l245_245116

-- Define the conditions
def white_red_ratio : ℕ × ℕ := (5, 3)
def num_white_balls : ℕ := 15

-- Define the theorem to prove
theorem red_balls_count (r : ℕ) : r = num_white_balls / (white_red_ratio.1) * (white_red_ratio.2) :=
by sorry

end red_balls_count_l245_245116


namespace fred_earned_correctly_l245_245775

-- Assuming Fred's earnings from different sources
def fred_earned_newspapers := 16 -- dollars
def fred_earned_cars := 74 -- dollars

-- Total earnings over the weekend
def fred_earnings := fred_earned_newspapers + fred_earned_cars

-- Given condition that Fred earned 90 dollars over the weekend
def fred_earnings_given := 90 -- dollars

-- The theorem stating that Fred's total earnings match the given earnings
theorem fred_earned_correctly : fred_earnings = fred_earnings_given := by
  sorry

end fred_earned_correctly_l245_245775


namespace wire_length_between_poles_l245_245992

theorem wire_length_between_poles :
  let x_dist := 20
  let y_dist := (18 / 2) - 8
  (x_dist ^ 2 + y_dist ^ 2 = 401) :=
by
  sorry

end wire_length_between_poles_l245_245992


namespace p_sq_plus_q_sq_l245_245503

theorem p_sq_plus_q_sq (p q : ℝ) (h1 : p * q = 12) (h2 : p + q = 8) : p^2 + q^2 = 40 := 
by 
  sorry

end p_sq_plus_q_sq_l245_245503


namespace condition_on_a_and_b_l245_245091

variable (x a b : ℝ)

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem condition_on_a_and_b
  (h1 : a > 0)
  (h2 : b > 0) :
  (∀ x : ℝ, |f x + 3| < a ↔ |x - 1| < b) ↔ (b^2 + 2*b + 3 ≤ a) :=
sorry

end condition_on_a_and_b_l245_245091


namespace volume_box_constraint_l245_245150

theorem volume_box_constraint : ∀ x : ℕ, ((2 * x + 6) * (x^3 - 8) * (x^2 + 4) < 1200) → x = 2 :=
by
  intros x h
  -- Proof is skipped
  sorry

end volume_box_constraint_l245_245150


namespace minimum_x_y_l245_245476

theorem minimum_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
by sorry

end minimum_x_y_l245_245476


namespace distinct_integer_roots_l245_245187

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end distinct_integer_roots_l245_245187


namespace solution_to_fraction_l245_245226

theorem solution_to_fraction (x : ℝ) (h_fraction : (x^2 - 4) / (x + 4) = 0) (h_denom : x ≠ -4) : x = 2 ∨ x = -2 :=
sorry

end solution_to_fraction_l245_245226


namespace total_chickens_l245_245373

theorem total_chickens (hens : ℕ) (roosters : ℕ) (h1 : hens = 52) (h2 : roosters = hens + 16) : hens + roosters = 120 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_chickens_l245_245373


namespace at_least_two_same_books_l245_245316

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def satisfied (n : Nat) : Prop :=
  n / sum_of_digits n = 13

theorem at_least_two_same_books (n1 n2 n3 n4 : Nat) (h1 : satisfied n1) (h2 : satisfied n2) (h3 : satisfied n3) (h4 : satisfied n4) :
  n1 = n2 ∨ n1 = n3 ∨ n1 = n4 ∨ n2 = n3 ∨ n2 = n4 ∨ n3 = n4 :=
sorry

end at_least_two_same_books_l245_245316


namespace check_perfect_squares_l245_245000

-- Define the prime factorizations of each option
def optionA := 3^3 * 4^5 * 7^7
def optionB := 3^4 * 4^4 * 7^6
def optionC := 3^6 * 4^3 * 7^8
def optionD := 3^5 * 4^6 * 7^5
def optionE := 3^4 * 4^6 * 7^7

-- Definition of a perfect square (all exponents in prime factorization are even)
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p : ℕ, (p ^ 2 ∣ n) -> (p ∣ n)

-- The Lean statement asserting which options are perfect squares
theorem check_perfect_squares :
  (is_perfect_square optionB) ∧ (is_perfect_square optionC) ∧
  ¬(is_perfect_square optionA) ∧ ¬(is_perfect_square optionD) ∧ ¬(is_perfect_square optionE) :=
by sorry

end check_perfect_squares_l245_245000


namespace Dima_floor_l245_245728

theorem Dima_floor (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 9)
  (h2 : 60 = (n - 1))
  (h3 : 70 = (n - 1) / (n - 1) * 60 + (n - n / 2) * 2 * 60)
  (h4 : ∀ m : ℕ, 1 ≤ m ∧ m ≤ 9 → (5 * n = 6 * m + 1) → (n = 7 ∧ m = 6)) :
  n = 7 :=
by
  sorry

end Dima_floor_l245_245728


namespace weight_of_replaced_sailor_l245_245991

theorem weight_of_replaced_sailor (avg_increase : ℝ) (total_sailors : ℝ) (new_sailor_weight : ℝ) : 
  avg_increase = 1 ∧ total_sailors = 8 ∧ new_sailor_weight = 64 → 
  ∃ W, W = 56 :=
by
  intro h
  sorry

end weight_of_replaced_sailor_l245_245991


namespace derivative_of_f_l245_245733

noncomputable def f (x : ℝ) : ℝ :=
  (1 / Real.sqrt 2) * Real.arctan ((2 * x + 1) / Real.sqrt 2) + (2 * x + 1) / (4 * x^2 + 4 * x + 3)

theorem derivative_of_f (x : ℝ) : deriv f x = 8 / (4 * x^2 + 4 * x + 3)^2 :=
by
  -- Proof will be provided here
  sorry

end derivative_of_f_l245_245733


namespace area_of_figure_l245_245251

-- Define the conditions using a predicate
def satisfies_condition (x y : ℝ) : Prop :=
  |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

-- Define the set of points satisfying the condition
def S : set (ℝ × ℝ) := { p | satisfies_condition p.1 p.2 }

-- Define a function to calculate the area of the resulting figure
noncomputable def area_of_S : ℝ :=
  -- define interior triangular region using vertices (0,0), (8,0), and (0,15)
  1 / 2 * 8 * 15

-- The theorem to be proved
theorem area_of_figure : area_of_S = 60 :=
by
  -- This is where the actual proof would go.
  sorry

end area_of_figure_l245_245251


namespace probability_between_C_and_D_l245_245102

theorem probability_between_C_and_D :
  ∀ (A B C D : ℝ) (AB AD BC : ℝ),
    AB = 3 * AD ∧ AB = 6 * BC ∧ D - A = AD ∧ C - A = AD + BC ∧ B - A = AB →
    (C < D) →
    ∃ p : ℝ, p = 1 / 2 := by
  sorry

end probability_between_C_and_D_l245_245102


namespace problem1_solution_problem2_solution_problem3_solution_l245_245026

noncomputable def problem1 : Real :=
  3 * Real.sqrt 3 + Real.sqrt 8 - Real.sqrt 2 + Real.sqrt 27

theorem problem1_solution : problem1 = 6 * Real.sqrt 3 + Real.sqrt 2 := by
  sorry

noncomputable def problem2 : Real :=
  (1/2) * (Real.sqrt 3 + Real.sqrt 5) - (3/4) * (Real.sqrt 5 - Real.sqrt 12)

theorem problem2_solution : problem2 = 2 * Real.sqrt 3 - (1/4) * Real.sqrt 5 := by
  sorry

noncomputable def problem3 : Real :=
  (2 * Real.sqrt 5 + Real.sqrt 6) * (2 * Real.sqrt 5 - Real.sqrt 6) - (Real.sqrt 5 - Real.sqrt 6) ^ 2

theorem problem3_solution : problem3 = 3 + 2 * Real.sqrt 30 := by
  sorry

end problem1_solution_problem2_solution_problem3_solution_l245_245026


namespace unique_fraction_increased_by_20_percent_l245_245579

def relatively_prime (x y : ℕ) : Prop := Nat.gcd x y = 1

theorem unique_fraction_increased_by_20_percent (x y : ℕ) (h1 : relatively_prime x y) (h2 : x > 0) (h3 : y > 0) :
  (∃! (x y : ℕ), relatively_prime x y ∧ (x > 0) ∧ (y > 0) ∧ (x + 2) * y = 6 * (y + 2) * x) :=
sorry

end unique_fraction_increased_by_20_percent_l245_245579


namespace hypotenuse_length_l245_245224

theorem hypotenuse_length (x a b: ℝ) (h1: a = 7) (h2: b = x - 1) (h3: a^2 + b^2 = x^2) : x = 25 :=
by {
  -- Condition h1 states that one leg 'a' is 7 cm.
  -- Condition h2 states that the other leg 'b' is 1 cm shorter than the hypotenuse 'x', i.e., b = x - 1.
  -- Condition h3 is derived from the Pythagorean theorem, i.e., a^2 + b^2 = x^2.
  -- We need to prove that x = 25 cm.
  sorry
}

end hypotenuse_length_l245_245224


namespace geometric_sequence_general_term_arithmetic_sequence_sum_l245_245937

noncomputable def a (n : ℕ) : ℕ := 2^n

def b (n : ℕ) : ℕ := 4 * (n - 1)

def S (n : ℕ) : ℕ := 2 * n^2 - 2 * n

theorem geometric_sequence_general_term
    (a1 : ℕ := 2)
    (a4 : ℕ := 16)
    (h1 : a 1 = a1)
    (h2 : a 4 = a4)
    : ∀ n : ℕ, a n = a 1 * 2^(n-1) :=
by
  sorry

theorem arithmetic_sequence_sum
    (a2 : ℕ := 4)
    (a5 : ℕ := 32)
    (b2 : ℕ := a 2)
    (b9 : ℕ := a 5)
    (h1 : b 2 = b2)
    (h2 : b 9 = b9)
    : ∀ n : ℕ, S n = n * (n - 1) * 2 :=
by
  sorry

end geometric_sequence_general_term_arithmetic_sequence_sum_l245_245937


namespace quadratic_poly_no_fixed_point_has_large_bound_l245_245049

/-- Given a quadratic polynomial f(x) with real coefficients such that the coefficient
of the x^2 term is positive and there is no real alpha such that f(alpha) = alpha, 
then there exists a positive integer n such that for any sequence {a_i} where 
a_i = f(a_{i-1}) for 1 ≤ i ≤ n, we have a_n > 2021. -/
theorem quadratic_poly_no_fixed_point_has_large_bound
  (f : ℝ → ℝ)
  (h_quad : ∃ a b c : ℝ, a > 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h_no_fixed : ∀ α : ℝ, f α ≠ α) :
  ∃ n : ℕ, ∀ (a0 : ℝ) (a : ℕ → ℝ), 
    (∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a (i + 1) = f (a i)) →
    a n > 2021 :=
sorry

end quadratic_poly_no_fixed_point_has_large_bound_l245_245049


namespace additional_cost_l245_245300

theorem additional_cost (original_bales : ℕ) (previous_cost_per_bale : ℕ) (better_quality_cost_per_bale : ℕ) :
  let new_bales := original_bales * 2
  let original_cost := original_bales * previous_cost_per_bale
  let new_cost := new_bales * better_quality_cost_per_bale
  new_cost - original_cost = 210 :=
by sorry

end additional_cost_l245_245300


namespace Jake_needs_to_lose_12_pounds_l245_245358

theorem Jake_needs_to_lose_12_pounds (J S : ℕ) (h1 : J + S = 156) (h2 : J = 108) : J - 2 * S = 12 := by
  sorry

end Jake_needs_to_lose_12_pounds_l245_245358


namespace squares_are_equal_l245_245738

theorem squares_are_equal (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
    (h₄ : a * (b + c + d) = b * (a + c + d)) 
    (h₅ : a * (b + c + d) = c * (a + b + d)) 
    (h₆ : a * (b + c + d) = d * (a + b + c)) : 
    a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := 
by
  sorry

end squares_are_equal_l245_245738


namespace tangent_lines_parallel_to_line_l245_245734

theorem tangent_lines_parallel_to_line (a : ℝ) (b : ℝ)
  (h1 : b = a^3 + a - 2)
  (h2 : 3 * a^2 + 1 = 4) :
  (b = 4 * a - 4 ∨ b = 4 * a) :=
sorry

end tangent_lines_parallel_to_line_l245_245734


namespace no_such_quadratic_exists_l245_245742

theorem no_such_quadratic_exists : ¬ ∃ (b c : ℝ), 
  (∀ x : ℝ, 6 * x ≤ 3 * x^2 + 3 ∧ 3 * x^2 + 3 ≤ x^2 + b * x + c) ∧
  (x^2 + b * x + c = 1) :=
by
  sorry

end no_such_quadratic_exists_l245_245742


namespace calculate_expression_l245_245719

theorem calculate_expression : 2.4 * 8.2 * (5.3 - 4.7) = 11.52 := by
  sorry

end calculate_expression_l245_245719


namespace total_number_of_animals_l245_245669

-- Define the data and conditions
def total_legs : ℕ := 38
def chickens : ℕ := 5
def chicken_legs : ℕ := 2
def sheep_legs : ℕ := 4

-- Define the proof problem
theorem total_number_of_animals (h1 : total_legs = 38) 
                                (h2 : chickens = 5) 
                                (h3 : chicken_legs = 2) 
                                (h4 : sheep_legs = 4) : 
  (∃ sheep : ℕ, chickens + sheep = 12) :=
by 
  sorry

end total_number_of_animals_l245_245669


namespace odd_sum_probability_in_3x3_grid_l245_245951

theorem odd_sum_probability_in_3x3_grid :
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9}
  let odd_numbers := {1, 3, 5, 7, 9}
  let even_numbers := {2, 4, 6, 8}
  (probability (grid : Fin 3 × Fin 3 → ℤ) 
    (∀ i, (∑ j, grid (i, j)) ∈ odd_numbers ∧ ∀ j, (∑ i, grid (i, j)) ∈ odd_numbers ∧
          ∀ i j, grid (i, j) ∈ numbers ∧ ∀ i j k l, grid (i, j) ≠ grid (k, l))) = 1/21 :=
sorry

end odd_sum_probability_in_3x3_grid_l245_245951


namespace scientific_notation_example_l245_245714

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l245_245714


namespace joe_probability_diff_fruit_l245_245501

-- Define the problem conditions in Lean
def fruits : Type := {f : fin 4 // f.val < 4}
instance : DecidableEq fruits := by apply_instance

def meal := {m : fin 3 // m.val < 3}
instance : DecidableEq meal := by apply_instance

noncomputable def random_fruit (m : meal) : Prob fruits := uniform

-- Define the problem and state the theorem
theorem joe_probability_diff_fruit :
  let probability_same_fruit := ((1 : ℚ) / 4) ^ 3 * 4 in
  1 - probability_same_fruit = 15/16 :=
  by
    let probability_same_fruit := ((1 : ℚ) / 4) ^ 3 * 4
    exact calc
      1 - probability_same_fruit = 1 - (1/4)^3 * 4 : by sorry
      ... = 1 - 4 * (1/4)^3 : by sorry
      ... = 1 - 4 * 1/64 : by sorry
      ... = 1 - 1/16               : by sorry
      ... = 15/16                  : by sorry

end joe_probability_diff_fruit_l245_245501


namespace Jans_original_speed_l245_245499

theorem Jans_original_speed
  (doubled_speed : ℕ → ℕ) (skips_after_training : ℕ) (time_in_minutes : ℕ) (original_speed : ℕ) :
  (∀ (s : ℕ), doubled_speed s = 2 * s) → 
  skips_after_training = 700 → 
  time_in_minutes = 5 → 
  (original_speed = (700 / 5) / 2) → 
  original_speed = 70 := 
by
  intros h1 h2 h3 h4
  exact h4

end Jans_original_speed_l245_245499


namespace shaded_fraction_correct_l245_245896

noncomputable def shaded_fraction : ℚ :=
  let a := (1/4 : ℚ) in
  let r := (1/16 : ℚ) in
  a / (1 - r)

theorem shaded_fraction_correct :
  shaded_fraction = 4 / 15 :=
by
  sorry

end shaded_fraction_correct_l245_245896


namespace find_profit_range_l245_245097

noncomputable def profit_range (x : ℝ) : Prop :=
  0 < x → 0.15 * (1 + 0.25 * x) * (100000 - x) ≥ 0.15 * 100000

theorem find_profit_range (x : ℝ) : profit_range x → 0 < x ∧ x ≤ 6 :=
by
  sorry

end find_profit_range_l245_245097


namespace convex_polygon_sum_of_squares_leq_four_l245_245697

theorem convex_polygon_sum_of_squares_leq_four (P : Polyhedron ℝ) (h_convex : P.IsConvex) (h_contained : P.IsContainedIn (Cube 1)) :
  (∑ i in P.sides, i.length^2) ≤ 4 := 
sorry

end convex_polygon_sum_of_squares_leq_four_l245_245697


namespace unique_numbers_l245_245685

theorem unique_numbers (x y : ℕ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (S : x + y = 17) 
  (Q : x^2 + y^2 = 145) 
  : x = 8 ∧ y = 9 ∨ x = 9 ∧ y = 8 :=
by
  sorry

end unique_numbers_l245_245685


namespace z_is_200_percent_of_x_l245_245753

theorem z_is_200_percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : y = 0.75 * x) :
  z = 2 * x :=
sorry

end z_is_200_percent_of_x_l245_245753


namespace sum_of_coefficients_l245_245724

theorem sum_of_coefficients :
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ,
  (∀ x : ℤ, (2 - x)^7 = a₀ + a₁ * (1 + x)^2 + a₂ * (1 + x)^3 + a₃ * (1 + x)^4 + a₄ * (1 + x)^5 + a₅ * (1 + x)^6 + a₆ * (1 + x)^7 + a₇ * (1 + x)^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 129 := by sorry

end sum_of_coefficients_l245_245724


namespace minimal_storing_capacity_required_l245_245699

theorem minimal_storing_capacity_required (k : ℕ) (h1 : k > 0)
    (bins : ℕ → ℕ → ℕ → Prop)
    (h_initial : bins 0 0 0)
    (h_laundry_generated : ∀ n, bins (10 * n) (10 * n) (10 * n))
    (h_heaviest_bin_emptied : ∀ n r b g, (r + b + g = 10 * n) → max r (max b g) + 10 * n - max r (max b g) = 10 * n)
    : ∀ (capacity : ℕ), capacity = 25 :=
sorry

end minimal_storing_capacity_required_l245_245699


namespace jacob_initial_fish_count_l245_245718

theorem jacob_initial_fish_count : 
  ∃ J : ℕ, 
    (∀ A : ℕ, A = 7 * J) → 
    (A' = A - 23) → 
    (J + 26 = A' + 1) → 
    J = 8 := 
by 
  sorry

end jacob_initial_fish_count_l245_245718


namespace net_increase_proof_l245_245087

def initial_cars := 50
def initial_motorcycles := 75
def initial_vans := 25

def car_arrival_rate : ℝ := 70
def car_departure_rate : ℝ := 40
def motorcycle_arrival_rate : ℝ := 120
def motorcycle_departure_rate : ℝ := 60
def van_arrival_rate : ℝ := 30
def van_departure_rate : ℝ := 20

def play_duration : ℝ := 2.5

def net_increase_car : ℝ := play_duration * (car_arrival_rate - car_departure_rate)
def net_increase_motorcycle : ℝ := play_duration * (motorcycle_arrival_rate - motorcycle_departure_rate)
def net_increase_van : ℝ := play_duration * (van_arrival_rate - van_departure_rate)

theorem net_increase_proof :
  net_increase_car = 75 ∧
  net_increase_motorcycle = 150 ∧
  net_increase_van = 25 :=
by
  -- Proof would go here.
  sorry

end net_increase_proof_l245_245087


namespace nate_current_age_l245_245039

open Real

variables (E N : ℝ)

/-- Ember is half as old as Nate, so E = 1/2 * N. -/
def ember_half_nate (h1 : E = 1/2 * N) : Prop := True

/-- The age difference of 7 years remains constant, so 21 - 14 = N - E. -/
def age_diff_constant (h2 : 7 = N - E) : Prop := True

/-- Prove that Nate is currently 14 years old given the conditions. -/
theorem nate_current_age (h1 : E = 1/2 * N) (h2 : 7 = N - E) : N = 14 :=
by sorry

end nate_current_age_l245_245039


namespace multiplication_factor_average_l245_245990

theorem multiplication_factor_average (a : ℕ) (b : ℕ) (c : ℕ) (F : ℝ) 
  (h1 : a = 7) 
  (h2 : b = 26) 
  (h3 : (c : ℝ) = 130) 
  (h4 : (a * b * F : ℝ) = a * c) :
  F = 5 := 
by 
  sorry

end multiplication_factor_average_l245_245990


namespace borrowed_sheets_l245_245306

theorem borrowed_sheets (sheets borrowed: ℕ) (average_page : ℝ) 
  (total_pages : ℕ := 80) (pages_per_sheet : ℕ := 2) (total_sheets : ℕ := 40) 
  (h1 : borrowed ≤ total_sheets)
  (h2 : sheets = total_sheets - borrowed)
  (h3 : average_page = 26) : borrowed = 17 :=
sorry 

end borrowed_sheets_l245_245306


namespace tan_A_tan_B_value_max_value_ab_sin_C_l245_245472

noncomputable def given_triangle_conditions (a b c : ℝ) (A B C : ℝ)
  (m n : ℝ × ℝ) : Prop :=
  ∃ (A B C : ℝ), 
  m = (1 - Real.cos (A + B), Real.cos ((A - B) / 2)) ∧
  n = (5 / 8, Real.cos ((A - B) / 2)) ∧
  (m.1 * n.1 + m.2 * n.2 = 9 / 8)

theorem tan_A_tan_B_value (a b c A B C : ℝ) (m n : ℝ × ℝ)
  (h : given_triangle_conditions a b c A B C m n) :
  Real.tan A * Real.tan B = 1 / 9 :=
sorry

theorem max_value_ab_sin_C (a b c A B C : ℝ) (m n : ℝ × ℝ)
  (h : given_triangle_conditions a b c A B C m n) :
  ∃ (M : ℝ), M = -3 / 8 ∧
  ∀ (A B : ℝ), max (a * b * Real.sin C / (a^2 + b^2 - c^2)) = M :=
sorry

end tan_A_tan_B_value_max_value_ab_sin_C_l245_245472


namespace no_solutions_l245_245513

theorem no_solutions (x : ℝ) (hx : x ≠ 0): ¬ (12 * Real.sin x + 5 * Real.cos x = 13 + 1 / |x|) := 
by 
  sorry

end no_solutions_l245_245513


namespace parallel_lines_m_l245_245215

noncomputable def lines_parallel (m : ℝ) : Prop :=
  let l1 := λ x y : ℝ, mx + 2 * y - 2 = 0
  let l2 := λ x y : ℝ, 5 * x + (m + 3) * y - 5 = 0
  (-m / 2) = (-5 / (m + 3))

theorem parallel_lines_m :
  (∃ m : ℝ, lines_parallel m) →
  lines_parallel (-5) :=
by
  sorry

end parallel_lines_m_l245_245215


namespace goldfish_equal_months_l245_245437

theorem goldfish_equal_months :
  ∃ (n : ℕ), 
    let B_n := 3 * 3^n 
    let G_n := 125 * 5^n 
    B_n = G_n ∧ n = 5 :=
by
  sorry

end goldfish_equal_months_l245_245437


namespace school_band_fundraising_l245_245825

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end school_band_fundraising_l245_245825


namespace possibleValuesOfSum_l245_245195

noncomputable def symmetricMatrixNonInvertible (x y z : ℝ) : Prop := 
  -(x + y + z) * ( x^2 + y^2 + z^2 - x * y - x * z - y * z ) = 0

theorem possibleValuesOfSum (x y z : ℝ) (h : symmetricMatrixNonInvertible x y z) :
  ∃ v : ℝ, v = -3 ∨ v = 3 / 2 := 
sorry

end possibleValuesOfSum_l245_245195


namespace marbles_left_mrs_hilt_marbles_left_l245_245784

-- Define the initial number of marbles
def initial_marbles : ℕ := 38

-- Define the number of marbles lost
def marbles_lost : ℕ := 15

-- Define the number of marbles given away
def marbles_given_away : ℕ := 6

-- Define the number of marbles found
def marbles_found : ℕ := 8

-- Use these definitions to calculate the total number of marbles left
theorem marbles_left : ℕ :=
  initial_marbles - marbles_lost - marbles_given_away + marbles_found

-- Prove that total number of marbles left is 25
theorem mrs_hilt_marbles_left : marbles_left = 25 := by 
  sorry

end marbles_left_mrs_hilt_marbles_left_l245_245784


namespace ben_time_to_school_l245_245549

/-- Amy's steps per minute -/
def amy_steps_per_minute : ℕ := 80

/-- Length of each of Amy's steps in cm -/
def amy_step_length : ℕ := 70

/-- Time taken by Amy to reach school in minutes -/
def amy_time_to_school : ℕ := 20

/-- Ben's steps per minute -/
def ben_steps_per_minute : ℕ := 120

/-- Length of each of Ben's steps in cm -/
def ben_step_length : ℕ := 50

/-- Given the above conditions, we aim to prove that Ben takes 18 2/3 minutes to reach school. -/
theorem ben_time_to_school : (112000 / 6000 : ℚ) = 18 + 2 / 3 := 
by sorry

end ben_time_to_school_l245_245549


namespace general_term_formula_exponential_seq_l245_245931

variable (n : ℕ)

def exponential_sequence (a1 r : ℕ) (n : ℕ) : ℕ := a1 * r^(n-1)

theorem general_term_formula_exponential_seq :
  exponential_sequence 2 3 n = 2 * 3^(n-1) :=
by
  sorry

end general_term_formula_exponential_seq_l245_245931


namespace point_distance_to_y_axis_l245_245491

def point := (x : ℝ , y : ℝ)

def distance_to_y_axis (p : point) : ℝ :=
  |p.1|

theorem point_distance_to_y_axis (P : point):
  P = (-3, 4) →
  distance_to_y_axis P = 3 :=
by
  intro h
  rw [h, distance_to_y_axis]
  sorry

end point_distance_to_y_axis_l245_245491


namespace messenger_speed_l245_245429

noncomputable def team_length : ℝ := 6

noncomputable def team_speed : ℝ := 5

noncomputable def total_time : ℝ := 0.5

theorem messenger_speed (x : ℝ) :
  (6 / (x + team_speed) + 6 / (x - team_speed) = total_time) →
  x = 25 := by
  sorry

end messenger_speed_l245_245429


namespace intersection_of_M_and_N_l245_245200

open Set

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
by {
  sorry
}

end intersection_of_M_and_N_l245_245200


namespace sum_of_digits_of_10_pow_30_minus_36_l245_245411

def sum_of_digits (n : ℕ) : ℕ := 
  (n.digits 10).sum

theorem sum_of_digits_of_10_pow_30_minus_36 : 
  sum_of_digits (10^30 - 36) = 11 := 
by 
  -- proof goes here
  sorry

end sum_of_digits_of_10_pow_30_minus_36_l245_245411


namespace probability_jqka_is_correct_l245_245462

noncomputable def probability_sequence_is_jqka : ℚ :=
  (4 / 52) * (4 / 51) * (4 / 50) * (4 / 49)

theorem probability_jqka_is_correct :
  probability_sequence_is_jqka = (16 / 4048375) :=
by
  sorry

end probability_jqka_is_correct_l245_245462


namespace triangle_cosine_sum_l245_245371

theorem triangle_cosine_sum (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hsum : A + B + C = π) : 
  (Real.cos A + Real.cos B + Real.cos C > 1) :=
sorry

end triangle_cosine_sum_l245_245371


namespace digit_after_decimal_l245_245391

theorem digit_after_decimal (n : ℕ) : (n = 123) → (123 % 12 ≠ 0) → (123 % 12 = 3) → (∃ d : ℕ, d = 1 ∧ (43 / 740 : ℚ)^123 = 0 + d / 10^(123)) := 
by
    intros h₁ h₂ h₃
    sorry

end digit_after_decimal_l245_245391


namespace glass_volume_l245_245884

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l245_245884


namespace divisible_by_7_of_sum_of_squares_l245_245948

theorem divisible_by_7_of_sum_of_squares (a b : ℤ) (h : 7 ∣ (a^2 + b^2)) : 
    (7 ∣ a) ∧ (7 ∣ b) :=
sorry

end divisible_by_7_of_sum_of_squares_l245_245948


namespace december_28_is_saturday_l245_245918

def days_per_week := 7

def thanksgiving_day : Nat := 28

def november_length : Nat := 30

def december_28_day_of_week : Nat :=
  (thanksgiving_day % days_per_week + november_length + 28 - thanksgiving_day) % days_per_week

theorem december_28_is_saturday :
  (december_28_day_of_week = 6) :=
by
  sorry

end december_28_is_saturday_l245_245918


namespace cost_ratio_l245_245050

theorem cost_ratio (S J M : ℝ) (h1 : S = 4) (h2 : M = 0.75 * (S + J)) (h3 : S + J + M = 21) : J / S = 2 :=
by
  sorry

end cost_ratio_l245_245050


namespace salary_increase_l245_245254

theorem salary_increase (S P : ℝ) (h1 : 0.70 * S + P * (0.70 * S) = 0.91 * S) : P = 0.30 :=
by
  have eq1 : 0.70 * S * (1 + P) = 0.91 * S := by sorry
  have eq2 : S * (0.70 + 0.70 * P) = 0.91 * S := by sorry
  have eq3 : 0.70 + 0.70 * P = 0.91 := by sorry
  have eq4 : 0.70 * P = 0.21 := by sorry
  have eq5 : P = 0.21 / 0.70 := by sorry
  have eq6 : P = 0.30 := by sorry
  exact eq6

end salary_increase_l245_245254


namespace sarah_toy_cars_l245_245646

theorem sarah_toy_cars (initial_money toy_car_cost scarf_cost beanie_cost remaining_money: ℕ) 
  (h_initial: initial_money = 53) 
  (h_toy_car_cost: toy_car_cost = 11) 
  (h_scarf_cost: scarf_cost = 10) 
  (h_beanie_cost: beanie_cost = 14) 
  (h_remaining: remaining_money = 7) : 
  (initial_money - remaining_money - scarf_cost - beanie_cost) / toy_car_cost = 2 := 
by 
  sorry

end sarah_toy_cars_l245_245646


namespace max_a_no_lattice_points_l245_245736

theorem max_a_no_lattice_points :
  ∀ (m : ℝ), (1 / 3) < m → m < (17 / 51) →
  ¬ ∃ (x : ℕ) (y : ℕ), 0 < x ∧ x ≤ 50 ∧ y = m * x + 3 := 
by
  sorry

end max_a_no_lattice_points_l245_245736


namespace magpies_gather_7_trees_magpies_not_gather_6_trees_l245_245902

-- Define the problem conditions.
def trees (n : ℕ) := (∀ (i : ℕ), i < n → ∃ (m : ℕ), m = i * 10)

-- Define the movement condition for magpies.
def magpie_move (n : ℕ) (d : ℕ) :=
  (∀ (i j : ℕ), i < n ∧ j < n ∧ i ≠ j → ∃ (k : ℕ), k = d ∧ ((i + d < n ∧ j - d < n) ∨ (i - d < n ∧ j + d < n)))

-- Prove that all magpies can gather on one tree for 7 trees.
theorem magpies_gather_7_trees : 
  ∃ (i : ℕ), i < 7 ∧ trees 7 ∧ magpie_move 7 (i * 10) → True :=
by
  -- proof steps here, which are not necessary for the task
  sorry

-- Prove that all magpies cannot gather on one tree for 6 trees.
theorem magpies_not_gather_6_trees : 
  ∀ (i : ℕ), i < 6 ∧ trees 6 ∧ magpie_move 6 (i * 10) → False :=
by
  -- proof steps here, which are not necessary for the task
  sorry

end magpies_gather_7_trees_magpies_not_gather_6_trees_l245_245902


namespace square_division_l245_245106

theorem square_division (n : Nat) : (n > 5 → ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) ∧ (n = 2 ∨ n = 3 → ¬ ∃ smaller_squares : List (Nat × Nat), smaller_squares.length = n ∧ (∀ s ∈ smaller_squares, IsSquare s)) := 
by
  sorry

end square_division_l245_245106


namespace steven_peaches_l245_245498

theorem steven_peaches (jake_peaches : ℕ) (steven_peaches : ℕ) (h1 : jake_peaches = 3) (h2 : jake_peaches + 10 = steven_peaches) : steven_peaches = 13 :=
by
  sorry

end steven_peaches_l245_245498


namespace general_term_formula_sum_of_first_n_terms_l245_245065

noncomputable def a (n : ℕ) : ℕ :=
(n + 2^n)^2

theorem general_term_formula :
  ∀ n : ℕ, a n = n^2 + n * 2^(n+1) + 4^n :=
sorry

noncomputable def S (n : ℕ) : ℕ :=
(n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3

theorem sum_of_first_n_terms :
  ∀ n : ℕ, S n = (n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3 :=
sorry

end general_term_formula_sum_of_first_n_terms_l245_245065


namespace monthly_salary_l245_245703

variables (S : ℝ) (savings : ℝ) (new_expenses : ℝ)

theorem monthly_salary (h1 : savings = 0.20 * S)
                      (h2 : new_expenses = 0.96 * S)
                      (h3 : S = 200 + new_expenses) :
                      S = 5000 :=
by
  sorry

end monthly_salary_l245_245703


namespace sufficient_not_necessary_condition_l245_245008

theorem sufficient_not_necessary_condition (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (x^2 - 1 > 0 → x < -1 ∨ x > 1) :=
by
  sorry

end sufficient_not_necessary_condition_l245_245008


namespace log_simplify_l245_245648

open Real

theorem log_simplify : 
  (1 / (log 12 / log 3 + 1)) + 
  (1 / (log 8 / log 2 + 1)) + 
  (1 / (log 30 / log 5 + 1)) = 2 :=
by
  sorry

end log_simplify_l245_245648


namespace reflection_line_slope_l245_245528

theorem reflection_line_slope (m b : ℝ)
  (h_reflection : ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = 2 ∧ y1 = 3 ∧ x2 = 10 ∧ y2 = 7 → 
    (x1 + x2) / 2 = (10 - 2) / 2 ∧ (y1 + y2) / 2 = (7 - 3) / 2 ∧ 
    y1 = m * x1 + b ∧ y2 = m * x2 + b) :
  m + b = 15 :=
sorry

end reflection_line_slope_l245_245528


namespace probability_even_sum_l245_245346

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def is_odd (n : ℕ) : Prop := n % 2 = 1

def count_odd_primes (primes : List ℕ) : ℕ :=
  primes.countp is_odd

def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

open Nat

theorem probability_even_sum :
  let primes := first_ten_primes in
  let odd_primes := count_odd_primes primes in
  let num_ways_even_sum := binom odd_primes 2 in
  let total_ways := binom primes.length 2 in
  (num_ways_even_sum : ℚ) / (total_ways : ℚ) = 4 / 5 :=
by
  sorry

end probability_even_sum_l245_245346


namespace sufficient_but_not_necessary_condition_l245_245854

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end sufficient_but_not_necessary_condition_l245_245854


namespace max_rectangle_area_l245_245818

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l245_245818


namespace remainder_of_3_pow_101_plus_5_mod_11_l245_245270

theorem remainder_of_3_pow_101_plus_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := by
  -- The theorem statement includes the condition that (3^101 + 5) mod 11 equals 8.
  -- The proof will make use of repetitive behavior and modular arithmetic.
  sorry

end remainder_of_3_pow_101_plus_5_mod_11_l245_245270


namespace x_gt_1_sufficient_not_necessary_x_squared_gt_1_l245_245856

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end x_gt_1_sufficient_not_necessary_x_squared_gt_1_l245_245856


namespace pastries_sold_l245_245908

def initial_pastries : ℕ := 148
def pastries_left : ℕ := 45

theorem pastries_sold : initial_pastries - pastries_left = 103 := by
  sorry

end pastries_sold_l245_245908


namespace grade_point_average_l245_245261

theorem grade_point_average (X : ℝ) (GPA_rest : ℝ) (GPA_whole : ℝ) 
  (h1 : GPA_rest = 66) (h2 : GPA_whole = 64) 
  (h3 : (1 / 3) * X + (2 / 3) * GPA_rest = GPA_whole) : X = 60 :=
sorry

end grade_point_average_l245_245261


namespace glass_volume_230_l245_245877

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l245_245877


namespace equivalent_weeks_l245_245942

def hoursPerDay := 24
def daysPerWeek := 7
def hoursPerWeek := daysPerWeek * hoursPerDay
def totalHours := 2016

theorem equivalent_weeks : totalHours / hoursPerWeek = 12 := 
by
  sorry

end equivalent_weeks_l245_245942


namespace lcm_144_132_eq_1584_l245_245310

theorem lcm_144_132_eq_1584 :
  Nat.lcm 144 132 = 1584 :=
by
  sorry

end lcm_144_132_eq_1584_l245_245310


namespace neg_p_is_true_neg_q_is_true_l245_245403

theorem neg_p_is_true : ∃ m : ℝ, ∀ x : ℝ, (x^2 + x - m = 0 → False) :=
sorry

theorem neg_q_is_true : ∀ x : ℝ, (x^2 + x + 1 > 0) :=
sorry

end neg_p_is_true_neg_q_is_true_l245_245403


namespace vector_subtraction_result_l245_245066

-- Defining the vectors a and b as given in the conditions
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- The main theorem stating that a - 2b results in the expected coordinates
theorem vector_subtraction_result :
  a - 2 • b = (7, -2) := by
  sorry

end vector_subtraction_result_l245_245066


namespace aluminum_atomic_weight_l245_245458

theorem aluminum_atomic_weight (Al_w : ℤ) 
  (compound_molecular_weight : ℤ) 
  (num_fluorine_atoms : ℕ) 
  (fluorine_atomic_weight : ℤ) 
  (h1 : compound_molecular_weight = 84) 
  (h2 : num_fluorine_atoms = 3) 
  (h3 : fluorine_atomic_weight = 19) :
  Al_w = 27 := 
by
  -- Proof goes here, but it is skipped.
  sorry

end aluminum_atomic_weight_l245_245458


namespace not_divisible_by_4_8_16_32_l245_245970

def x := 80 + 112 + 144 + 176 + 304 + 368 + 3248 + 17

theorem not_divisible_by_4_8_16_32 : 
  ¬ (x % 4 = 0) ∧ ¬ (x % 8 = 0) ∧ ¬ (x % 16 = 0) ∧ ¬ (x % 32 = 0) := 
by 
  sorry

end not_divisible_by_4_8_16_32_l245_245970


namespace max_height_of_projectile_l245_245708

def projectile_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

theorem max_height_of_projectile : 
  ∃ t : ℝ, projectile_height t = 161 :=
sorry

end max_height_of_projectile_l245_245708


namespace tank_fraction_l245_245568

theorem tank_fraction (x : ℚ) (h₁ : 48 * x + 8 = 48 * (9 / 10)) : x = 2 / 5 :=
by
  sorry

end tank_fraction_l245_245568


namespace no_solution_m1_no_solution_m2_solution_m3_l245_245644

-- Problem 1: No positive integer solutions for m = 1
theorem no_solution_m1 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ x * y * z := sorry

-- Problem 2: No positive integer solutions for m = 2
theorem no_solution_m2 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 ≠ 2 * x * y * z := sorry

-- Problem 3: Only solutions for m = 3 are x = y = z = k for some k
theorem solution_m3 (x y z : ℕ) (h: x > 0 ∧ y > 0 ∧ z > 0) : x^3 + y^3 + z^3 = 3 * x * y * z ↔ x = y ∧ y = z := sorry

end no_solution_m1_no_solution_m2_solution_m3_l245_245644


namespace simple_interest_years_l245_245287

variables (T R : ℝ)

def principal : ℝ := 1000
def additional_interest : ℝ := 90

theorem simple_interest_years
  (H: principal * (R + 3) * T / 100 - principal * R * T / 100 = additional_interest) :
  T = 3 :=
by sorry

end simple_interest_years_l245_245287


namespace difference_of_squares_l245_245758

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
sorry

end difference_of_squares_l245_245758


namespace jim_out_of_pocket_cost_l245_245500

theorem jim_out_of_pocket_cost {price1 price2 sale : ℕ} 
    (h1 : price1 = 10000)
    (h2 : price2 = 2 * price1)
    (h3 : sale = price1 / 2) :
    (price1 + price2 - sale = 25000) :=
by
  sorry

end jim_out_of_pocket_cost_l245_245500


namespace geometric_sum_five_terms_l245_245241

theorem geometric_sum_five_terms (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_a2a4 : a 1 * a 3 = 16)
  (h_ratio : (a 3 + a 4 + a 7) / (a 0 + a 1 + a 4) = 8) :
  S 5 = 31 :=
sorry

end geometric_sum_five_terms_l245_245241


namespace max_value_eq_two_l245_245237

noncomputable def max_value (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) : ℝ :=
  a + b^3 + c^4

theorem max_value_eq_two (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a + b + c = 2) :
  max_value a b c h1 h2 h3 h4 ≤ 2 :=
sorry

end max_value_eq_two_l245_245237


namespace james_total_earnings_l245_245627

def january_earnings : ℕ := 4000
def february_earnings : ℕ := january_earnings + (50 * january_earnings / 100)
def march_earnings : ℕ := february_earnings - (20 * february_earnings / 100)
def total_earnings : ℕ := january_earnings + february_earnings + march_earnings

theorem james_total_earnings :
  total_earnings = 14800 :=
by
  -- skip the proof
  sorry

end james_total_earnings_l245_245627


namespace lattice_points_in_bounded_region_l245_245867

def isLatticePoint (p : ℤ × ℤ) : Prop :=
  true  -- All (n, m) ∈ ℤ × ℤ are lattice points

def boundedRegion (x y : ℤ) : Prop :=
  y = x ^ 2 ∨ y = 8 - x ^ 2
  
theorem lattice_points_in_bounded_region :
  ∃ S : Finset (ℤ × ℤ), 
    (∀ p ∈ S, isLatticePoint p ∧ boundedRegion p.1 p.2) ∧ S.card = 17 :=
by
  sorry

end lattice_points_in_bounded_region_l245_245867


namespace exponent_division_l245_245547

theorem exponent_division :
  (1000 ^ 7) / (10 ^ 17) = 10 ^ 4 := 
  sorry

end exponent_division_l245_245547


namespace average_of_P_Q_R_is_correct_l245_245176

theorem average_of_P_Q_R_is_correct (P Q R : ℝ) 
  (h1 : 1001 * R - 3003 * P = 6006) 
  (h2 : 2002 * Q + 4004 * P = 8008) : 
  (P + Q + R)/3 = (2 * (P + 5))/3 :=
sorry

end average_of_P_Q_R_is_correct_l245_245176


namespace real_values_satisfying_inequality_l245_245586

theorem real_values_satisfying_inequality :
  { x : ℝ | (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 } =
  Set.Icc (-1 : ℝ) ((-3 - Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪ 
  Set.Ioi 0 :=
by
  sorry

end real_values_satisfying_inequality_l245_245586


namespace greatest_int_radius_of_circle_l245_245613

theorem greatest_int_radius_of_circle (r : ℝ) (A : ℝ) :
  (A < 200 * Real.pi) ∧ (A = Real.pi * r^2) →
  ∃k : ℕ, (k : ℝ) = 14 ∧ ∀n : ℕ, (n : ℝ) = r → n ≤ k := by
  sorry

end greatest_int_radius_of_circle_l245_245613


namespace area_enclosed_by_curves_l245_245029

theorem area_enclosed_by_curves (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, (x + a * y)^2 = 16 * a^2) ∧ (∀ x y : ℝ, (a * x - y)^2 = 4 * a^2) →
  ∃ A : ℝ, A = 32 * a^2 / (1 + a^2) :=
by
  sorry

end area_enclosed_by_curves_l245_245029


namespace gcd_of_60_and_75_l245_245141

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l245_245141


namespace correct_equation_l245_245082

theorem correct_equation (x : ℕ) : 8 * x - 3 = 7 * x + 4 :=
by sorry

end correct_equation_l245_245082


namespace interest_rate_l245_245164

variable (P : ℝ) (T : ℝ) (SI : ℝ)

theorem interest_rate (h_P : P = 535.7142857142857) (h_T : T = 4) (h_SI : SI = 75) :
    (SI / (P * T)) * 100 = 3.5 := by
  sorry

end interest_rate_l245_245164


namespace inequality_solution_l245_245538

theorem inequality_solution (x : ℝ) : |2 * x - 7| < 3 → 2 < x ∧ x < 5 :=
by
  sorry

end inequality_solution_l245_245538


namespace find_a_max_and_min_values_l245_245327

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + x + a)
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 + 3*x + a + 1)

theorem find_a (a : ℝ) : (f' a 0) = 2 → a = 1 :=
by {
  -- Proof omitted
  sorry
}

theorem max_and_min_values (a : ℝ) :
  (a = 1) →
  (Real.exp (-2) * (4 - 2 + 1) = (3 / Real.exp 2)) ∧
  (Real.exp (-1) * (1 - 1 + 1) = (1 / Real.exp 1)) ∧
  (Real.exp 2 * (4 + 2 + 1) = (7 * Real.exp 2)) :=
by {
  -- Proof omitted
  sorry
}

end find_a_max_and_min_values_l245_245327


namespace Ravi_probability_l245_245672

-- Conditions from the problem
def P_Ram : ℚ := 4 / 7
def P_BothSelected : ℚ := 0.11428571428571428

-- Statement to prove
theorem Ravi_probability :
  ∃ P_Ravi : ℚ, P_Rami = 0.2 ∧ P_Ram * P_Ravi = P_BothSelected := by
  sorry

end Ravi_probability_l245_245672


namespace square_perimeter_l245_245191

variable (side : ℕ) (P : ℕ)

theorem square_perimeter (h : side = 19) : P = 4 * side → P = 76 := by
  intro hp
  rw [h] at hp
  norm_num at hp
  exact hp

end square_perimeter_l245_245191


namespace connie_initial_marbles_l245_245915

theorem connie_initial_marbles (marbles_given : ℕ) (marbles_left : ℕ) (initial_marbles : ℕ) 
    (h1 : marbles_given = 183) (h2 : marbles_left = 593) : initial_marbles = 776 :=
by
  sorry

end connie_initial_marbles_l245_245915


namespace glass_volume_230_l245_245874

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l245_245874


namespace gcd_60_75_l245_245145

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l245_245145


namespace gcd_of_60_and_75_l245_245136

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l245_245136


namespace find_p_l245_245555

theorem find_p (m n p : ℝ)
  (h1 : m = 4 * n + 5)
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 :=
sorry

end find_p_l245_245555


namespace f_1996x_l245_245972

noncomputable def f : ℝ → ℝ := sorry

axiom f_equation (x y : ℝ) : f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

theorem f_1996x (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end f_1996x_l245_245972


namespace glass_volume_l245_245881

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l245_245881


namespace number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l245_245155

-- Define the statement about the total number of 5-letter words.
theorem number_of_5_letter_words : 26^5 = 26^5 := by
  sorry

-- Define the statement about the total number of 5-letter words with all different letters.
theorem number_of_5_letter_words_with_all_different_letters : 
  26 * 25 * 24 * 23 * 22 = 26 * 25 * 24 * 23 * 22 := by
  sorry

-- Define the statement about the total number of 5-letter words with no consecutive letters being the same.
theorem number_of_5_letter_words_with_no_consecutive_repeating_letters : 
  26 * 25 * 25 * 25 * 25 = 26 * 25 * 25 * 25 * 25 := by
  sorry

end number_of_5_letter_words_number_of_5_letter_words_with_all_different_letters_number_of_5_letter_words_with_no_consecutive_repeating_letters_l245_245155


namespace problem_statement_l245_245485

variable (a : ℝ)

theorem problem_statement (h : a^2 + 3*a - 4 = 0) : 2*a^2 + 6*a - 3 = 5 := 
by sorry

end problem_statement_l245_245485


namespace inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l245_245930

theorem inequality_d_over_c_lt_d_plus_4_over_c_plus_4
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : c > d)
  (h3 : d > 0) :
  (d / c) < ((d + 4) / (c + 4)) :=
by
  sorry

end inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l245_245930


namespace sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l245_245175

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l245_245175


namespace friends_meeting_distance_l245_245410

theorem friends_meeting_distance (R_q : ℝ) (t : ℝ) (D_p D_q trail_length : ℝ) :
  trail_length = 36 ∧ D_p = 1.25 * R_q * t ∧ D_q = R_q * t ∧ D_p + D_q = trail_length → D_p = 20 := by
  sorry

end friends_meeting_distance_l245_245410


namespace B_work_days_l245_245282

-- Define work rates and conditions
def A_work_rate : ℚ := 1 / 18
def B_work_rate : ℚ := 1 / 15
def A_days_after_B_left : ℚ := 6
def total_work : ℚ := 1

-- Theorem statement
theorem B_work_days : ∃ x : ℚ, (x * B_work_rate + A_days_after_B_left * A_work_rate = total_work) → x = 10 := by
  sorry

end B_work_days_l245_245282


namespace sqrt_of_sqrt_81_l245_245650

theorem sqrt_of_sqrt_81 : Real.sqrt 81 = 9 := 
  by
  sorry

end sqrt_of_sqrt_81_l245_245650


namespace part1_l245_245211

noncomputable def f (a x : ℝ) : ℝ := a * x - 2 * Real.log x + 2 * (1 + a) + (a - 2) / x

theorem part1 (a : ℝ) (h : 0 < a) : 
  (∀ x : ℝ, 1 ≤ x → f a x ≥ 0) ↔ 1 ≤ a :=
sorry

end part1_l245_245211


namespace algebraic_expression_value_l245_245222

theorem algebraic_expression_value (a b : ℝ) (h : 4 * a - 2 * b + 2 = 0) :
  2024 + 2 * a - b = 2023 :=
by
  sorry

end algebraic_expression_value_l245_245222


namespace bus_stops_for_18_minutes_l245_245453

-- Definitions based on conditions
def speed_without_stoppages : ℝ := 50 -- kmph
def speed_with_stoppages : ℝ := 35 -- kmph
def distance_reduced_due_to_stoppage_per_hour : ℝ := speed_without_stoppages - speed_with_stoppages

noncomputable def time_bus_stops_per_hour (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem bus_stops_for_18_minutes :
  time_bus_stops_per_hour distance_reduced_due_to_stoppage_per_hour (speed_without_stoppages / 60) = 18 := by
  sorry

end bus_stops_for_18_minutes_l245_245453


namespace angles_sum_n_l245_245405

/-- Given that the sum of the measures in degrees of angles A, B, C, D, E, and F is 90 * n,
    we need to prove that n = 4. -/
theorem angles_sum_n (A B C D E F : ℝ) (n : ℕ) 
  (h : A + B + C + D + E + F = 90 * n) :
  n = 4 :=
sorry

end angles_sum_n_l245_245405


namespace joanne_first_hour_coins_l245_245921

theorem joanne_first_hour_coins 
  (X : ℕ)
  (H1 : 70 = 35 + 35)
  (H2 : 120 = X + 70 + 35)
  (H3 : 35 = 50 - 15) : 
  X = 15 :=
sorry

end joanne_first_hour_coins_l245_245921


namespace determine_m_l245_245603

theorem determine_m {m : ℕ} : 
  (∃ (p : ℕ), p = 5 ∧ p = max (max (max 1 (1 + (m+1))) (3+1)) 4) → m = 3 := by
  sorry

end determine_m_l245_245603


namespace glass_volume_230_l245_245871

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l245_245871


namespace solve_floor_equation_l245_245808

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem solve_floor_equation :
  (∃ x : ℝ, (floor ((x - 1) / 2))^2 + 2 * x + 2 = 0) → x = -3 :=
by
  sorry

end solve_floor_equation_l245_245808


namespace third_term_is_18_l245_245470

-- Define the first term and the common ratio
def a_1 : ℕ := 2
def q : ℕ := 3

-- Define the function for the nth term of an arithmetic-geometric sequence
def a_n (n : ℕ) : ℕ :=
  a_1 * q^(n-1)

-- Prove that the third term is 18
theorem third_term_is_18 : a_n 3 = 18 := by
  sorry

end third_term_is_18_l245_245470


namespace proof_problem_l245_245317

variable (α β : ℝ) (a b : ℝ × ℝ) (m : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 4)
variable (hβ : β = Real.pi)
variable (ha_def : a = (Real.tan (α + β / 4) - 1, 0))
variable (hb : b = (Real.cos α, 2))
variable (ha_dot : a.1 * b.1 + a.2 * b.2 = m)

-- Proof statement
theorem proof_problem :
  (0 < α ∧ α < Real.pi / 4) ∧
  β = Real.pi ∧
  a = (Real.tan (α + β / 4) - 1, 0) ∧
  b = (Real.cos α, 2) ∧
  (a.1 * b.1 + a.2 * b.2 = m) →
  (2 * Real.cos α * Real.cos α + Real.sin (2 * (α + β))) / (Real.cos α - Real.sin β) = 2 * (m + 2) := by
  sorry

end proof_problem_l245_245317


namespace intersection_points_form_rectangle_l245_245209

theorem intersection_points_form_rectangle
  (x y : ℝ)
  (h1 : x * y = 8)
  (h2 : x^2 + y^2 = 34) :
  ∃ (a b u v : ℝ), (a * b = 8) ∧ (a^2 + b^2 = 34) ∧ 
  (u * v = 8) ∧ (u^2 + v^2 = 34) ∧
  ((a = x ∧ b = y) ∨ (a = y ∧ b = x)) ∧ 
  ((u = -x ∧ v = -y) ∨ (u = -y ∧ v = -x)) ∧
  ((a = u ∧ b = v) ∨ (a = v ∧ b = u)) ∧ 
  ((x = -u ∧ y = -v) ∨ (x = -v ∧ y = -u)) ∧
  (
    (a, b) ≠ (u, v) ∧ (a, b) ≠ (-u, -v) ∧ 
    (a, b) ≠ (v, u) ∧ (a, b) ≠ (-v, -u) ∧
    (u, v) ≠ (-a, -b) ∧ (u, v) ≠ (b, a) ∧ 
    (u, v) ≠ (-b, -a)
  ) :=
by sorry

end intersection_points_form_rectangle_l245_245209


namespace problem_statement_l245_245446

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem problem_statement : ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 := by
  sorry

end problem_statement_l245_245446


namespace ethan_hours_per_day_l245_245920

-- Define the known constants
def hourly_wage : ℝ := 18
def work_days_per_week : ℕ := 5
def total_earnings : ℝ := 3600
def weeks_worked : ℕ := 5

-- Define the main theorem
theorem ethan_hours_per_day :
  (∃ hours_per_day : ℝ, 
    hours_per_day = total_earnings / (weeks_worked * work_days_per_week * hourly_wage)) →
  hours_per_day = 8 :=
by
  sorry

end ethan_hours_per_day_l245_245920


namespace glass_volume_is_230_l245_245888

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l245_245888


namespace toby_friends_girls_l245_245156

theorem toby_friends_girls (total_friends : ℕ) (num_boys : ℕ) (perc_boys : ℕ) 
  (h1 : perc_boys = 55) (h2 : num_boys = 33) (h3 : total_friends = 60) : 
  (total_friends - num_boys = 27) :=
by
  sorry

end toby_friends_girls_l245_245156


namespace t_mobile_additional_line_cost_l245_245248

variable (T : ℕ)

def t_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 50 else 50 + (n - 2) * T

def m_mobile_cost (n : ℕ) : ℕ :=
  if n ≤ 2 then 45 else 45 + (n - 2) * 14

theorem t_mobile_additional_line_cost
  (h : t_mobile_cost 5 = m_mobile_cost 5 + 11) :
  T = 16 :=
by
  sorry

end t_mobile_additional_line_cost_l245_245248


namespace calculate_distribution_l245_245442

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end calculate_distribution_l245_245442


namespace elements_map_to_4_l245_245321

def f (x : ℝ) : ℝ := x^2

theorem elements_map_to_4 :
  { x : ℝ | f x = 4 } = {2, -2} :=
by
  sorry

end elements_map_to_4_l245_245321


namespace glass_volume_l245_245880

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l245_245880


namespace find_number_l245_245754

theorem find_number (X : ℝ) (h : 0.8 * X = 0.7 * 60.00000000000001 + 30) : X = 90.00000000000001 :=
sorry

end find_number_l245_245754


namespace first_girl_productivity_higher_l245_245843

theorem first_girl_productivity_higher :
  let T1 := 6 in -- (5 minutes work + 1 minute break) cycle for the first girl
  let T2 := 8 in -- (7 minutes work + 1 minute break) cycle for the second girl
  let LCM := Nat.lcm T1 T2 in
  let total_work_time_first_girl := (5 * (LCM / T1)) in
  let total_work_time_second_girl := (7 * (LCM / T2)) in
  total_work_time_first_girl = total_work_time_second_girl →
  (total_work_time_first_girl / 5) / (total_work_time_second_girl / 7) = 21/20 :=
by
  intros
  sorry

end first_girl_productivity_higher_l245_245843


namespace angle_sum_155_l245_245865

theorem angle_sum_155
  (AB AC DE DF : ℝ)
  (h1 : AB = AC)
  (h2 : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h3 : angle_BAC = 20)
  (h4 : angle_EDF = 30) :
  ∃ (angle_DAC angle_ADE : ℝ), angle_DAC + angle_ADE = 155 :=
by
  sorry

end angle_sum_155_l245_245865


namespace find_n_l245_245035

theorem find_n (n a b : ℕ) (h1 : n ≥ 2)
  (h2 : n = a^2 + b^2)
  (h3 : a = Nat.minFac n)
  (h4 : b ∣ n) : n = 8 ∨ n = 20 := 
sorry

end find_n_l245_245035


namespace z_pow12_plus_inv_z_pow12_l245_245205

open Complex

theorem z_pow12_plus_inv_z_pow12 (z: ℂ) (h: z + z⁻¹ = 2 * cos (10 * Real.pi / 180)) :
  z^12 + z⁻¹^12 = -1 := by
  sorry

end z_pow12_plus_inv_z_pow12_l245_245205


namespace correct_calculation_l245_245398

variable {a : ℝ}

theorem correct_calculation : a^2 * a^3 = a^5 :=
by sorry

end correct_calculation_l245_245398


namespace polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l245_245208

theorem polynomial_three_positive_roots_inequality
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  2 * a^3 + 9 * c ≤ 7 * a * b :=
sorry

theorem polynomial_three_positive_roots_equality_condition
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  (2 * a^3 + 9 * c = 7 * a * b) ↔ (x1 = x2 ∧ x2 = x3) :=
sorry

end polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l245_245208


namespace petyas_square_is_larger_l245_245788

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l245_245788


namespace find_function_expression_find_range_of_m_l245_245958

-- Statement for Part 1
theorem find_function_expression (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) : 
  y = -1/2 * x - 2 := 
sorry

-- Statement for Part 2
theorem find_range_of_m (m x : ℝ) (hx : x > -2) (k b : ℝ) (hk : k ≠ 0) (h1 : 2 * k + b = -3) (h2 : -4 * k + b = 0) :
  (-x + m < -1/2 * x - 2) ↔ (m ≤ -3) := 
sorry

end find_function_expression_find_range_of_m_l245_245958


namespace jen_age_when_son_born_l245_245968

theorem jen_age_when_son_born (S : ℕ) (Jen_present_age : ℕ) 
  (h1 : S = 16) (h2 : Jen_present_age = 3 * S - 7) : 
  Jen_present_age - S = 25 :=
by {
  sorry -- Proof would be here, but it is not required as per the instructions.
}

end jen_age_when_son_born_l245_245968


namespace find_a_plus_c_l245_245232

noncomputable def triangle_ABC (A B C a b c : ℝ) : Prop :=
  (b * Real.cos C + c * Real.cos B = 3 * a * Real.cos B) ∧
  (b = 2) ∧
  ((1 / 2) * b * c * Real.sin A = (3 * Real.sqrt 2) / 2)

theorem find_a_plus_c {A B C a b c : ℝ} (h : triangle_ABC A B C a b c) :
  a + c = 4 :=
by
  rcases h with ⟨hc1, hc2, hc3⟩
  sorry

end find_a_plus_c_l245_245232


namespace least_number_subtracted_l245_245271

theorem least_number_subtracted {
  x : ℕ
} : 
  (∀ (m : ℕ), m ∈ [5, 9, 11] → (997 - x) % m = 3) → x = 4 :=
by
  sorry

end least_number_subtracted_l245_245271


namespace problem_statement_l245_245465

theorem problem_statement (x y : ℝ) (p : x > 0 ∧ y > 0) : (∃ p, p → xy > 0) ∧ ¬(xy > 0 → x > 0 ∧ y > 0) :=
by
  sorry

end problem_statement_l245_245465


namespace molecular_weight_correct_l245_245392

-- Define the atomic weights
def atomic_weight_K : ℝ := 39.10
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01

-- Define the number of atoms of each element
def num_atoms_K : ℕ := 2
def num_atoms_Br : ℕ := 2
def num_atoms_O : ℕ := 4
def num_atoms_H : ℕ := 3
def num_atoms_N : ℕ := 1

-- Calculate the molecular weight
def molecular_weight : ℝ :=
  num_atoms_K * atomic_weight_K +
  num_atoms_Br * atomic_weight_Br +
  num_atoms_O * atomic_weight_O +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N

-- Define the expected molecular weight
def expected_molecular_weight : ℝ := 319.04

-- The theorem stating that the calculated molecular weight matches the expected molecular weight
theorem molecular_weight_correct : molecular_weight = expected_molecular_weight :=
  by
  sorry -- Proof is skipped

end molecular_weight_correct_l245_245392


namespace find_y_l245_245624

theorem find_y {x y : ℝ} (hx : (8 : ℝ) = (1/4 : ℝ) * x) (hy : (y : ℝ) = (1/4 : ℝ) * (20 : ℝ)) (hprod : x * y = 160) : y = 5 :=
by {
  sorry
}

end find_y_l245_245624


namespace work_together_l245_245005

theorem work_together (A B : ℝ) (hA : A = 1/3) (hB : B = 1/6) : (1 / (A + B)) = 2 := by
  sorry

end work_together_l245_245005


namespace scientific_notation_example_l245_245713

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l245_245713


namespace IH_length_eq_l245_245291

noncomputable def length_IH : ℝ :=
  let AB := 4
  let CE := 12
  let DF := real.sqrt (16^2 + 12^2)
  let r := 12 / 20
  let GJ := r * 12
  GJ

theorem IH_length_eq : length_IH = 36 / 5 := by
  -- Definitions
  let AB := 4
  let CE := 12
  let DE := AB + CE
  let EF := CE
  let DF := real.sqrt (DE^2 + EF^2)
  let r := EF / DF
  let GJ := r * EF

  -- Proof sketch (not required)
  -- 1. Calculate DF using Pythagorean theorem
  -- 2. Use similarity ratio to find GJ
  -- 3. Show that IH equals GJ and simplify to obtain the result

  rw [DE, EF, DF, GJ] -- Replace with actual values
  repeat { rw [pow_two] }
  repeat { rw [real.sqrt_eq_rpow] }
  norm_num -- Simplify the result
  exact_mod_cast sorry-- Exact proof of the final simplified result

end IH_length_eq_l245_245291


namespace polynomial_product_l245_245735

noncomputable def sum_of_coefficients (g h : ℤ) : ℤ := g + h

theorem polynomial_product (g h : ℤ) :
  (9 * d^3 - 5 * d^2 + g) * (4 * d^2 + h * d - 9) = 36 * d^5 - 11 * d^4 - 49 * d^3 + 45 * d^2 - 9 * d →
  sum_of_coefficients g h = 18 :=
by
  intro
  sorry

end polynomial_product_l245_245735


namespace train_crosses_tunnel_in_45_sec_l245_245569

/-- Given the length of the train, the length of the platform, the length of the tunnel, 
and the time taken to cross the platform, prove the time taken for the train to cross the tunnel is 45 seconds. -/
theorem train_crosses_tunnel_in_45_sec (l_train : ℕ) (l_platform : ℕ) (t_platform : ℕ) (l_tunnel : ℕ)
  (h_train_length : l_train = 330)
  (h_platform_length : l_platform = 180)
  (h_time_platform : t_platform = 15)
  (h_tunnel_length : l_tunnel = 1200) :
  (l_train + l_tunnel) / ((l_train + l_platform) / t_platform) = 45 :=
by
  -- placeholder for the actual proof
  sorry

end train_crosses_tunnel_in_45_sec_l245_245569


namespace reflection_line_slope_intercept_l245_245530

theorem reflection_line_slope_intercept (m b : ℝ) :
  let P1 := (2, 3)
  let P2 := (10, 7)
  let midpoint := ((P1.1 + P2.1) / 2, (P1.2 + P2.2) / 2)
  midpoint = (6, 5) ∧
  ∃(m b : ℝ), 
    m = -2 ∧
    b = 17 ∧
    P2 = (2 * midpoint.1 - P1.1, 2 * midpoint.2 - P1.2)
→ m + b = 15 := by
  intros
  sorry

end reflection_line_slope_intercept_l245_245530


namespace range_of_a_l245_245061

variable {a : ℝ}

theorem range_of_a (h : ∃ x : ℝ, x < 0 ∧ 5^x = (a + 3) / (5 - a)) : -3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l245_245061


namespace sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l245_245174

theorem sqrt_8_plus_sqrt_2_minus_sqrt_18 :
  (Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 18 = 0) :=
sorry

theorem sqrt_3_minus_2_squared :
  ((Real.sqrt 3 - 2) ^ 2 = 7 - 4 * Real.sqrt 3) :=
sorry

end sqrt_8_plus_sqrt_2_minus_sqrt_18_sqrt_3_minus_2_squared_l245_245174


namespace range_of_a_l245_245481

variable {α : Type*} [LinearOrderedField α]

def setA (a : α) : Set α := {x | abs (x - a) < 1}
def setB : Set α := {x | 1 < x ∧ x < 5}

theorem range_of_a (a : α) (h : setA a ∩ setB = ∅) : a ≤ 0 ∨ a ≥ 6 :=
sorry

end range_of_a_l245_245481


namespace geometric_sequence_common_ratio_l245_245760

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 = 1)
  (h2 : a 5 = 16)
  (h_pos : ∀ n : ℕ, 0 < a n) :
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l245_245760


namespace mary_paid_amount_l245_245640

-- Definitions for the conditions:
def is_adult (person : String) : Prop := person = "Mary"
def children_count (n : ℕ) : Prop := n = 3
def ticket_cost_adult : ℕ := 2  -- $2 for adults
def ticket_cost_child : ℕ := 1  -- $1 for children
def change_received : ℕ := 15   -- $15 change

-- Mathematical proof to find the amount Mary paid given the conditions
theorem mary_paid_amount (person : String) (n : ℕ) 
  (h1 : is_adult person) (h2 : children_count n) :
  ticket_cost_adult + ticket_cost_child * n + change_received = 20 := 
by 
  -- Sorry as the proof is not required
  sorry

end mary_paid_amount_l245_245640


namespace books_added_l245_245559

theorem books_added (initial_books sold_books current_books added_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : sold_books = 3)
  (h3 : current_books = 11)
  (h4 : added_books = current_books - (initial_books - sold_books)) :
  added_books = 10 :=
by
  rw [h1, h2, h3] at h4
  simp at h4
  exact h4

end books_added_l245_245559


namespace highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l245_245036

theorem highest_power_of_2_dividing_15_pow_4_minus_9_pow_4 :
  (∃ k, 15^4 - 9^4 = 2^k * m ∧ ¬ ∃ m', m = 2 * m') ∧ (k = 5) :=
by
  sorry

end highest_power_of_2_dividing_15_pow_4_minus_9_pow_4_l245_245036


namespace Cody_spent_25_tickets_on_beanie_l245_245572

-- Introducing the necessary definitions and assumptions
variable (x : ℕ)

-- Define the conditions translated from the problem statement
def initial_tickets := 49
def tickets_left (x : ℕ) := initial_tickets - x + 6

-- State the main problem as Theorem
theorem Cody_spent_25_tickets_on_beanie (H : tickets_left x = 30) : x = 25 := by
  sorry

end Cody_spent_25_tickets_on_beanie_l245_245572


namespace rectangular_prism_volume_l245_245614

theorem rectangular_prism_volume
  (L W h : ℝ)
  (h1 : L - W = 23)
  (h2 : 2 * L + 2 * W = 166) :
  L * W * h = 1590 * h :=
by
  sorry

end rectangular_prism_volume_l245_245614


namespace polynomial_roots_a_ge_five_l245_245636

theorem polynomial_roots_a_ge_five (a b c : ℤ) (h_a_pos : a > 0)
    (h_distinct_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 0 < x₁ ∧ x₁ < 1 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
        a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) : a ≥ 5 := sorry

end polynomial_roots_a_ge_five_l245_245636


namespace miki_pear_juice_l245_245783

def total_pears : ℕ := 18
def total_oranges : ℕ := 10
def pear_juice_per_pear : ℚ := 10 / 2
def orange_juice_per_orange : ℚ := 12 / 3
def max_blend_volume : ℚ := 44

theorem miki_pear_juice : (total_oranges * orange_juice_per_orange = 40) ∧ (max_blend_volume - 40 = 4) → 
  ∃ p : ℚ, p * pear_juice_per_pear = 4 ∧ p = 0 :=
by
  sorry

end miki_pear_juice_l245_245783


namespace number_of_aluminum_atoms_l245_245563

def molecular_weight (n : ℕ) : ℝ :=
  n * 26.98 + 30.97 + 4 * 16.0

theorem number_of_aluminum_atoms (n : ℕ) (h : molecular_weight n = 122) : n = 1 :=
by
  sorry

end number_of_aluminum_atoms_l245_245563


namespace range_of_a_l245_245331

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - |x + 1| + 2 * a ≥ 0) ↔ a ∈ (Set.Ici ((Real.sqrt 3 + 1) / 4)) := by
  sorry

end range_of_a_l245_245331


namespace parallel_vectors_sum_l245_245333

variable (x y : ℝ)
variable (k : ℝ)

theorem parallel_vectors_sum :
  (k * 3 = 2) ∧ (k * x = 4) ∧ (k * y = 5) → x + y = 27 / 2 :=
by
  sorry

end parallel_vectors_sum_l245_245333


namespace calculate_arithmetic_expression_l245_245909

noncomputable def arithmetic_sum (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem calculate_arithmetic_expression :
  3 * (arithmetic_sum 71 2 99) = 3825 :=
by
  sorry

end calculate_arithmetic_expression_l245_245909


namespace pool_filling_time_l245_245002

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end pool_filling_time_l245_245002


namespace triangle_is_obtuse_l245_245088

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h1 : 3 * A > 5 * B)
  (h2 : 3 * C < 2 * B)
  (h3 : A + B + C = 180) :
  A > 90 :=
sorry

end triangle_is_obtuse_l245_245088


namespace invertible_elements_mod_8_l245_245129

theorem invertible_elements_mod_8 :
  {x : ℤ | (x * x) % 8 = 1} = {1, 3, 5, 7} :=
by
  sorry

end invertible_elements_mod_8_l245_245129


namespace pencils_in_each_box_l245_245111

open Nat

theorem pencils_in_each_box (boxes pencils_given_to_Lauren pencils_left pencils_each_box more_pencils : ℕ)
  (h1 : boxes = 2)
  (h2 : pencils_given_to_Lauren = 6)
  (h3 : pencils_left = 9)
  (h4 : more_pencils = 3)
  (h5 : pencils_given_to_Matt = pencils_given_to_Lauren + more_pencils)
  (h6 : pencils_each_box = (pencils_given_to_Lauren + pencils_given_to_Matt + pencils_left) / boxes) :
  pencils_each_box = 12 := by
  sorry

end pencils_in_each_box_l245_245111


namespace even_of_even_square_sqrt_two_irrational_l245_245415

-- Problem 1: Prove that if \( p^2 \) is even, then \( p \) is even given \( p \in \mathbb{Z} \).
theorem even_of_even_square (p : ℤ) (h : Even (p * p)) : Even p := 
sorry 

-- Problem 2: Prove that \( \sqrt{2} \) is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ Int.gcd a b = 1 ∧ (a : ℝ) / b = Real.sqrt 2 :=
sorry

end even_of_even_square_sqrt_two_irrational_l245_245415


namespace jims_final_paycheck_l245_245628

noncomputable def final_paycheck (g r t h m b btr : ℝ) := 
  let retirement := g * r
  let gym := m / 2
  let net_before_bonus := g - retirement - t - h - gym
  let after_tax_bonus := b * (1 - btr)
  net_before_bonus + after_tax_bonus

theorem jims_final_paycheck :
  final_paycheck 1120 0.25 100 200 50 500 0.30 = 865 :=
by
  sorry

end jims_final_paycheck_l245_245628


namespace table_relationship_l245_245304

theorem table_relationship (x y : ℕ) (h : (x, y) ∈ [(1, 1), (2, 8), (3, 27), (4, 64), (5, 125)]) : y = x^3 :=
sorry

end table_relationship_l245_245304


namespace minimum_crooks_l245_245496

theorem minimum_crooks (total_ministers : ℕ) (H C : ℕ) (h1 : total_ministers = 100) 
  (h2 : ∀ (s : Finset ℕ), s.card = 10 → ∃ x ∈ s, x = C) :
  C ≥ 91 :=
by
  have h3 : H = total_ministers - C, from sorry,
  have h4 : H ≤ 9, from sorry,
  have h5 : C = total_ministers - H, from sorry,
  have h6 : C ≥ 100 - 9, from sorry,
  exact h6

end minimum_crooks_l245_245496


namespace ratio_of_riding_to_total_l245_245012

-- Define the primary conditions from the problem
variables (H R W : ℕ)
variables (legs_on_ground : ℕ := 50)
variables (total_owners : ℕ := 10)
variables (legs_per_horse : ℕ := 4)
variables (legs_per_owner : ℕ := 2)

-- Express the conditions
def conditions : Prop :=
  (legs_on_ground = 6 * W) ∧
  (total_owners = H) ∧
  (H = R + W) ∧
  (H = 10)

-- Define the theorem with the given conditions and prove the required ratio
theorem ratio_of_riding_to_total (H R W : ℕ) (h : conditions H R W) : R / 10 = 1 / 5 := by
  sorry

end ratio_of_riding_to_total_l245_245012


namespace negatively_added_marks_l245_245078

theorem negatively_added_marks 
  (correct_marks_per_question : ℝ) 
  (total_marks : ℝ) 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (x : ℝ) 
  (h1 : correct_marks_per_question = 4)
  (h2 : total_marks = 420)
  (h3 : total_questions = 150)
  (h4 : correct_answers = 120) 
  (h5 : total_marks = (correct_answers * correct_marks_per_question) - ((total_questions - correct_answers) * x)) :
  x = 2 :=
by 
  sorry

end negatively_added_marks_l245_245078


namespace area_equivalence_l245_245235

noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def angle_bisector (A B C : Point) : Point := sorry
noncomputable def arc_midpoint (A B C : Point) : Point := sorry
noncomputable def is_concyclic (P Q R S : Point) : Prop := sorry
noncomputable def area_of_quad (A B C D : Point) : ℝ := sorry
noncomputable def area_of_pent (A B C D E : Point) : ℝ := sorry

theorem area_equivalence (A B C I X Y M : Point)
  (h1 : I = incenter A B C)
  (h2 : X = angle_bisector B A C)
  (h3 : Y = angle_bisector C A B)
  (h4 : M = arc_midpoint A B C)
  (h5 : is_concyclic M X I Y) :
  area_of_quad M B I C = area_of_pent B X I Y C := 
sorry

end area_equivalence_l245_245235


namespace translate_right_l245_245620

-- Definition of the initial point and translation distance
def point_A : ℝ × ℝ := (2, -1)
def translation_distance : ℝ := 3

-- The proof statement
theorem translate_right (x_A y_A : ℝ) (d : ℝ) 
  (h1 : point_A = (x_A, y_A))
  (h2 : translation_distance = d) : 
  (x_A + d, y_A) = (5, -1) := 
sorry

end translate_right_l245_245620


namespace find_n_l245_245486

open Classical

theorem find_n (n : ℕ) (h : (8 * Nat.choose n 3) = 8 * (2 * Nat.choose n 1)) : n = 5 := by
  sorry

end find_n_l245_245486


namespace degrees_to_radians_150_l245_245725

theorem degrees_to_radians_150 :
  (150 : ℝ) * (Real.pi / 180) = (5 * Real.pi) / 6 :=
by
  sorry

end degrees_to_radians_150_l245_245725


namespace smallest_angle_of_trapezoid_l245_245382

theorem smallest_angle_of_trapezoid (a d : ℝ) (h1 : a + 3 * d = 120) (h2 : 4 * a + 6 * d = 360) :
  a = 60 := by
  sorry

end smallest_angle_of_trapezoid_l245_245382


namespace jack_black_balloons_l245_245096

def nancy_balloons := 7
def mary_balloons := 4 * nancy_balloons
def total_mary_nancy_balloons := nancy_balloons + mary_balloons
def jack_balloons := total_mary_nancy_balloons + 3

theorem jack_black_balloons : jack_balloons = 38 := by
  -- proof goes here
  sorry

end jack_black_balloons_l245_245096


namespace gcd_60_75_l245_245146

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l245_245146


namespace adjacent_angles_l245_245100

theorem adjacent_angles (α β : ℝ) (h1 : α = β + 30) (h2 : α + β = 180) : α = 105 ∧ β = 75 := by
  sorry

end adjacent_angles_l245_245100


namespace jia_profits_1_yuan_l245_245085

-- Definition of the problem conditions
def initial_cost : ℝ := 1000
def profit_rate : ℝ := 0.1
def loss_rate : ℝ := 0.1
def resale_rate : ℝ := 0.9

-- Defined transactions with conditions
def jia_selling_price1 : ℝ := initial_cost * (1 + profit_rate)
def yi_selling_price_to_jia : ℝ := jia_selling_price1 * (1 - loss_rate)
def jia_selling_price2 : ℝ := yi_selling_price_to_jia * resale_rate

-- Final net income calculation
def jia_net_income : ℝ := -initial_cost + jia_selling_price1 - yi_selling_price_to_jia + jia_selling_price2

-- Lean statement to be proved
theorem jia_profits_1_yuan : jia_net_income = 1 := sorry

end jia_profits_1_yuan_l245_245085


namespace proof_problem_l245_245279

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}
def complement (s : Set ℕ) : Set ℕ := {x | x ∉ s}

theorem proof_problem : ((complement A ∪ A) ∪ B) = U :=
by sorry

end proof_problem_l245_245279


namespace operation_result_l245_245546

def a : ℝ := 0.8
def b : ℝ := 0.5
def c : ℝ := 0.40

theorem operation_result :
  (a ^ 3 - b ^ 3 / a ^ 2 + c + b ^ 2) = 0.9666875 := by
  sorry

end operation_result_l245_245546


namespace owen_work_hours_l245_245368

def total_hours := 24
def chores_hours := 7
def sleep_hours := 11

theorem owen_work_hours : total_hours - chores_hours - sleep_hours = 6 := by
  sorry

end owen_work_hours_l245_245368


namespace geometric_sequence_a5_value_l245_245231

-- Definition of geometric sequence and the specific condition a_3 * a_7 = 8
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (geom_seq : is_geometric_sequence a)
  (cond : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
sorry

end geometric_sequence_a5_value_l245_245231


namespace largest_rectangle_area_l245_245821

noncomputable def max_rectangle_area_with_perimeter (p : ℕ) : ℕ := sorry

theorem largest_rectangle_area (p : ℕ) (h : p = 60) : max_rectangle_area_with_perimeter p = 225 :=
sorry

end largest_rectangle_area_l245_245821


namespace translate_triangle_vertex_l245_245324

theorem translate_triangle_vertex 
    (a b : ℤ) 
    (hA : (-3, a) = (-1, 2) + (-2, a - 2)) 
    (hB : (b, 3) = (1, -1) + (b - 1, 4)) :
    (2 + (-3 - (-1)), 1 + (3 - (-1))) = (0, 5) :=
by 
  -- proof is omitted as instructed
  sorry

end translate_triangle_vertex_l245_245324


namespace eval_expression_l245_245633

theorem eval_expression :
  let a := 3
  let b := 2
  (2 ^ a ∣ 200) ∧ ¬(2 ^ (a + 1) ∣ 200) ∧ (5 ^ b ∣ 200) ∧ ¬(5 ^ (b + 1) ∣ 200)
→ (1 / 3)^(b - a) = 3 :=
by sorry

end eval_expression_l245_245633


namespace books_total_l245_245266

theorem books_total (Tim_books Sam_books : ℕ) (h1 : Tim_books = 44) (h2 : Sam_books = 52) : Tim_books + Sam_books = 96 := 
by
  sorry

end books_total_l245_245266


namespace minimize_x_l245_245552

theorem minimize_x (x y : ℝ) (h₀ : 0 < x) (h₁ : 0 < y) (h₂ : x + y^2 = x * y) : x ≥ 3 :=
sorry

end minimize_x_l245_245552


namespace area_of_triangle_l245_245673

theorem area_of_triangle (m1 m2 : ℝ) (P : ℝ × ℝ) (l3 : ℝ → ℝ) :
  m1 = 2 → m2 = 1 / 2 → P = (2, 2) → l3 = (λ x, -x + 10) →
  let l1 := (λ x, m1 * x + (P.2 - m1 * P.1)) in
  let l2 := (λ x, m2 * x + (P.2 - m2 * P.1)) in
  let intersect (f g : ℝ → ℝ) (x : ℝ) : Prop := f x = g x in
  let A := P in
  let Bx := (4 : ℝ) in
  let By := l1 4 in
  let B := (Bx, By) in
  let Cx := (6 : ℝ) in
  let Cy := l2 6 in
  let C := (Cx, Cy) in
  let area (A B C : ℝ × ℝ) : ℝ :=
    0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area A B C = 6 :=
begin
  intros,
  sorry
end

end area_of_triangle_l245_245673


namespace find_f_l245_245662

theorem find_f (d e f : ℝ) (h_g : 16 = g) 
  (h_mean_of_zeros : -d / 12 = 3 + d + e + f + 16) 
  (h_product_of_zeros_two_at_a_time : -d / 12 = e / 3) : 
  f = -39 :=
by
  sorry

end find_f_l245_245662


namespace polynomial_determination_l245_245177

theorem polynomial_determination (P : Polynomial ℝ) :
  (∀ X : ℝ, P.eval (X^2) = (X^2 + 1) * P.eval X) →
  (∃ a : ℝ, ∀ X : ℝ, P.eval X = a * (X^2 - 1)) :=
by
  sorry

end polynomial_determination_l245_245177


namespace rate_of_interest_l245_245689

/-
Let P be the principal amount, SI be the simple interest paid, R be the rate of interest, and N be the number of years. 
The problem states:
- P = 1200
- SI = 432
- R = N

We need to prove that R = 6.
-/

theorem rate_of_interest (P SI R N : ℝ) (h1 : P = 1200) (h2 : SI = 432) (h3 : R = N) :
  R = 6 :=
  sorry

end rate_of_interest_l245_245689


namespace point_on_coordinate_axes_l245_245073

theorem point_on_coordinate_axes {x y : ℝ} 
  (h : x * y = 0) : (x = 0 ∨ y = 0) :=
by {
  sorry
}

end point_on_coordinate_axes_l245_245073


namespace divisibility_of_powers_l245_245105

theorem divisibility_of_powers (n : ℤ) : 65 ∣ (7^4 * n - 4^4 * n) :=
by
  sorry

end divisibility_of_powers_l245_245105


namespace geometric_sequence_a_eq_one_l245_245649

theorem geometric_sequence_a_eq_one (a : ℝ) 
  (h₁ : ∃ (r : ℝ), a = 1 / (1 - r) ∧ r = a - 1/2 ∧ r ≠ 0) : 
  a = 1 := 
sorry

end geometric_sequence_a_eq_one_l245_245649


namespace parabola_line_intersection_sum_l245_245093

theorem parabola_line_intersection_sum (r s : ℝ) (h_r : r = 20 - 10 * Real.sqrt 38) (h_s : s = 20 + 10 * Real.sqrt 38) :
  r + s = 40 := by
  sorry

end parabola_line_intersection_sum_l245_245093


namespace lattice_points_on_hyperbola_l245_245939

-- The hyperbola equation
def hyperbola_eq (x y : ℤ) : Prop :=
  x^2 - y^2 = 1800^2

-- The final number of lattice points lying on the hyperbola
theorem lattice_points_on_hyperbola : 
  ∃ (n : ℕ), n = 250 ∧ (∃ (x y : ℤ), hyperbola_eq x y) :=
sorry

end lattice_points_on_hyperbola_l245_245939


namespace buttons_pattern_total_buttons_sum_l245_245001

-- Define the sequence of the number of buttons in each box
def buttons_in_box (n : ℕ) : ℕ := 3^(n-1)

-- Define the sum of buttons up to the n-th box
def total_buttons (n : ℕ) : ℕ := (3^n - 1) / 2

-- Theorem statements to prove
theorem buttons_pattern (n : ℕ) : buttons_in_box n = 3^(n-1) := by
  sorry

theorem total_buttons_sum (n : ℕ) : total_buttons n = (3^n - 1) / 2 := by
  sorry

end buttons_pattern_total_buttons_sum_l245_245001


namespace cannot_divide_m_l245_245737

/-
  A proof that for the real number m = 2009^3 - 2009, 
  the number 2007 does not divide m.
-/

theorem cannot_divide_m (m : ℤ) (h : m = 2009^3 - 2009) : ¬ (2007 ∣ m) := 
by sorry

end cannot_divide_m_l245_245737


namespace product_of_millions_l245_245194

-- Define the conditions
def a := 5 * (10 : ℝ) ^ 6
def b := 8 * (10 : ℝ) ^ 6

-- State the proof problem
theorem product_of_millions : (a * b) = 40 * (10 : ℝ) ^ 12 := 
by
  sorry

end product_of_millions_l245_245194


namespace christina_has_three_snakes_l245_245443

def snake_lengths : List ℕ := [24, 16, 10]

def total_length : ℕ := 50

theorem christina_has_three_snakes
  (lengths : List ℕ)
  (total : ℕ)
  (h_lengths : lengths = snake_lengths)
  (h_total : total = total_length)
  : lengths.length = 3 :=
by
  sorry

end christina_has_three_snakes_l245_245443


namespace correct_statements_l245_245473

namespace ProofProblem

variable (f : ℕ+ × ℕ+ → ℕ+)
variable (h1 : f (1, 1) = 1)
variable (h2 : ∀ m n : ℕ+, f (m, n + 1) = f (m, n) + 2)
variable (h3 : ∀ m : ℕ+, f (m + 1, 1) = 2 * f (m, 1))

theorem correct_statements :
  f (1, 5) = 9 ∧ f (5, 1) = 16 ∧ f (5, 6) = 26 :=
by
  sorry

end ProofProblem

end correct_statements_l245_245473


namespace find_a10_l245_245497

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def a2_eq_4 (a : ℕ → ℝ) := a 2 = 4

def a6_eq_6 (a : ℕ → ℝ) := a 6 = 6

theorem find_a10 (a : ℕ → ℝ) (h_geom : geometric_sequence a) (h2 : a2_eq_4 a) (h6 : a6_eq_6 a) : 
  a 10 = 9 :=
sorry

end find_a10_l245_245497


namespace number_of_possible_flags_l245_245448

theorem number_of_possible_flags : 
  let colors := { "purple", "gold", "silver" } in
  ∃ (f : Fin 3 → colors), (∀ i : Fin 2, f i ≠ f (i + 1)) → (Fin 3 → colors) :=
by
  let colors := { "purple", "gold", "silver" }
  have h1 : 3 = card colors := by sorry
  have h2 : 3 = card { c // ¬(f 1 = c) } := by sorry
  have h3 : 3 = card { c // ¬(f 2 = c) } := by sorry
  have total_possibilities : 3 * 2 * 2 = 12 := by sorry
  existsi total_possibilities
  exact sorry

end number_of_possible_flags_l245_245448


namespace largest_multiple_negation_greater_than_neg150_l245_245679

theorem largest_multiple_negation_greater_than_neg150 (n : ℤ) (h₁ : n % 6 = 0) (h₂ : -n > -150) : n = 144 := 
sorry

end largest_multiple_negation_greater_than_neg150_l245_245679


namespace old_supervisor_salary_correct_l245_245524

def old_supervisor_salary (W S_old : ℝ) : Prop :=
  let avg_old := (W + S_old) / 9
  let avg_new := (W + 510) / 9
  avg_old = 430 ∧ avg_new = 390 → S_old = 870

theorem old_supervisor_salary_correct (W : ℝ) :
  old_supervisor_salary W 870 :=
by
  unfold old_supervisor_salary
  intro h
  sorry

end old_supervisor_salary_correct_l245_245524


namespace total_length_of_sticks_l245_245773

theorem total_length_of_sticks :
  ∃ (s1 s2 s3 : ℝ), s1 = 3 ∧ s2 = 2 * s1 ∧ s3 = s2 - 1 ∧ (s1 + s2 + s3 = 14) := by
  sorry

end total_length_of_sticks_l245_245773


namespace boris_climbs_needed_l245_245483

-- Definitions
def elevation_hugo : ℕ := 10000
def shorter_difference : ℕ := 2500
def climbs_hugo : ℕ := 3

-- Derived Definitions
def elevation_boris : ℕ := elevation_hugo - shorter_difference
def total_climbed_hugo : ℕ := climbs_hugo * elevation_hugo

-- Theorem
theorem boris_climbs_needed : (total_climbed_hugo / elevation_boris) = 4 :=
by
  -- conditions and definitions are used here
  sorry

end boris_climbs_needed_l245_245483


namespace jorge_goals_this_season_l245_245629

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end jorge_goals_this_season_l245_245629


namespace restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l245_245014

-- Let P be the original price of the jacket
variable (P : ℝ)

-- The price of the jacket after successive reductions
def price_after_discount (P : ℝ) : ℝ := 0.60 * P

-- The price of the jacket after all discounts including the limited-time offer
def price_after_full_discount (P : ℝ) : ℝ := 0.54 * P

-- Prove that to restore 0.60P back to P a 66.67% increase is needed
theorem restore_to_original_without_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.60 * P) * (1 + 66.67 / 100) = P :=
by sorry

-- Prove that to restore 0.54P back to P an 85.19% increase is needed
theorem restore_to_original_with_limited_discount :
  ∀ (P : ℝ), (P > 0) → (0.54 * P) * (1 + 85.19 / 100) = P :=
by sorry

end restore_to_original_without_limited_discount_restore_to_original_with_limited_discount_l245_245014


namespace transform_to_A_plus_one_l245_245389

theorem transform_to_A_plus_one (A : ℕ) (hA : A > 0) : 
  ∃ n : ℕ, (∀ i : ℕ, (i ≤ n) → ((A + 9 * i) = A + 1 ∨ ∃ j : ℕ, (A + 9 * i) = (A + 1 + 10 * j))) :=
sorry

end transform_to_A_plus_one_l245_245389


namespace int_coeffs_square_sum_l245_245219

theorem int_coeffs_square_sum (a b c d e f : ℤ)
  (h : ∀ x, 8 * x^3 + 125 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 767 := 
sorry

end int_coeffs_square_sum_l245_245219


namespace proof_problem_l245_245072

-- Definitions of the conditions
def cond1 (r : ℕ) : Prop := 2^r = 16
def cond2 (s : ℕ) : Prop := 5^s = 25

-- Statement of the problem
theorem proof_problem (r s : ℕ) (h₁ : cond1 r) (h₂ : cond2 s) : r + s = 6 :=
by
  sorry

end proof_problem_l245_245072


namespace part1_part2_l245_245932

open Set Real

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | m - 3 ≤ x ∧ x ≤ m + 3}
noncomputable def C : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem part1 (m : ℝ) (h : A ∩ B m = C) : m = 5 :=
  sorry

theorem part2 (m : ℝ) (h : A ⊆ (B m)ᶜ) : m < -4 ∨ 6 < m :=
  sorry

end part1_part2_l245_245932


namespace solve_equation_l245_245799

theorem solve_equation (x : ℝ) (h1 : x + 1 ≠ 0) (h2 : 2 * x - 1 ≠ 0) :
  (2 / (x + 1) = 3 / (2 * x - 1)) ↔ (x = 5) := 
sorry

end solve_equation_l245_245799


namespace Victor_worked_hours_l245_245543

theorem Victor_worked_hours (h : ℕ) (pay_rate : ℕ) (total_earnings : ℕ) 
  (H1 : pay_rate = 6) 
  (H2 : total_earnings = 60) 
  (H3 : 2 * (pay_rate * h) = total_earnings): 
  h = 5 := 
by 
  sorry

end Victor_worked_hours_l245_245543


namespace functional_relationship_and_point_l245_245204

noncomputable def directly_proportional (y x : ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ y = k * x

theorem functional_relationship_and_point :
  (∀ x y, directly_proportional y x → y = 2 * x) ∧ 
  (∀ a : ℝ, (∃ (y : ℝ), y = 3 ∧ directly_proportional y a) → a = 3 / 2) :=
by
  sorry

end functional_relationship_and_point_l245_245204


namespace belle_biscuits_l245_245294

-- Define the conditions
def cost_per_rawhide_bone : ℕ := 1
def num_rawhide_bones_per_evening : ℕ := 2
def cost_per_biscuit : ℚ := 0.25
def total_weekly_cost : ℚ := 21
def days_in_week : ℕ := 7

-- Define the number of biscuits Belle eats every evening
def num_biscuits_per_evening : ℚ := 4

-- Define the statement that encapsulates the problem
theorem belle_biscuits :
  (total_weekly_cost = days_in_week * (num_rawhide_bones_per_evening * cost_per_rawhide_bone + num_biscuits_per_evening * cost_per_biscuit)) :=
sorry

end belle_biscuits_l245_245294


namespace glass_volume_correct_l245_245893

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l245_245893


namespace midpoint_between_points_l245_245455

theorem midpoint_between_points : 
  let (x1, y1, z1) := (2, -3, 5)
  let (x2, y2, z2) := (8, 1, 3)
  (1 / 2 * (x1 + x2), 1 / 2 * (y1 + y2), 1 / 2 * (z1 + z2)) = (5, -1, 4) :=
by
  sorry

end midpoint_between_points_l245_245455


namespace quadratic_inequality_solution_l245_245757

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x^2 + m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
by
  sorry

end quadratic_inequality_solution_l245_245757


namespace scientific_notation_of_284000000_l245_245712

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l245_245712


namespace division_result_l245_245460

theorem division_result : (0.284973 / 29 = 0.009827) := 
by sorry

end division_result_l245_245460


namespace algebraic_expression_value_l245_245797

theorem algebraic_expression_value (x : ℝ) (hx : x = Real.sqrt 7 + 1) :
  (x^2 / (x - 3) - 2 * x / (x - 3)) / (x / (x - 3)) = Real.sqrt 7 - 1 :=
by
  sorry

end algebraic_expression_value_l245_245797


namespace find_principal_l245_245018

-- Define the conditions
variables (P R : ℝ) -- Define P and R as real numbers
variable (h : (P * 50) / 100 = 300) -- Introduce the equation obtained from the conditions

-- State the theorem
theorem find_principal (P R : ℝ) (h : (P * 50) / 100 = 300) : P = 600 :=
sorry

end find_principal_l245_245018


namespace number_of_girls_l245_245761

theorem number_of_girls (boys girls : ℕ) (h1 : boys = 337) (h2 : girls = boys + 402) : girls = 739 := by
  sorry

end number_of_girls_l245_245761


namespace glass_volume_l245_245885

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l245_245885


namespace petya_square_larger_l245_245790

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l245_245790


namespace asian_population_percentage_in_west_l245_245171

theorem asian_population_percentage_in_west
    (NE MW South West : ℕ)
    (H_NE : NE = 2)
    (H_MW : MW = 3)
    (H_South : South = 2)
    (H_West : West = 6)
    : (West * 100) / (NE + MW + South + West) = 46 :=
sorry

end asian_population_percentage_in_west_l245_245171


namespace seventh_root_of_unity_sum_l245_245054

theorem seventh_root_of_unity_sum (z : ℂ) (h1 : z^7 = 1) (h2 : z ≠ 1) :
  z + z^2 + z^4 = (-1 + Complex.I * Real.sqrt 11) / 2 ∨ z + z^2 + z^4 = (-1 - Complex.I * Real.sqrt 11) / 2 := 
by sorry

end seventh_root_of_unity_sum_l245_245054


namespace movie_watching_l245_245447

theorem movie_watching :
  let total_duration := 120 
  let watched1 := 35
  let watched2 := 20
  let watched3 := 15
  let total_watched := watched1 + watched2 + watched3
  total_duration - total_watched = 50 :=
by
  sorry

end movie_watching_l245_245447


namespace solve_y_l245_245982

theorem solve_y (y : ℝ) (h : 5 * y^(1/4) - 3 * (y / y^(3/4)) = 9 + y^(1/4)) : y = 6561 := 
by 
  sorry

end solve_y_l245_245982


namespace sequence_product_l245_245081

-- Definitions and conditions
def geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

def a4_value (a : ℕ → ℕ) : Prop :=
a 4 = 2

-- The statement to be proven
theorem sequence_product (a : ℕ → ℕ) (q : ℕ) (h_geo_seq : geometric_sequence a q) (h_a4 : a4_value a) :
  a 2 * a 3 * a 5 * a 6 = 16 :=
sorry

end sequence_product_l245_245081


namespace plot_length_l245_245688

def breadth : ℝ := 40 -- Derived from conditions and cost equation solution
def length : ℝ := breadth + 20
def cost_per_meter : ℝ := 26.50
def total_cost : ℝ := 5300

theorem plot_length :
  (2 * (breadth + (breadth + 20))) * cost_per_meter = total_cost → length = 60 :=
by {
  sorry
}

end plot_length_l245_245688


namespace trays_from_second_table_l245_245638

def trays_per_trip : ℕ := 4
def trips : ℕ := 9
def trays_from_first_table : ℕ := 20

theorem trays_from_second_table :
  trays_per_trip * trips - trays_from_first_table = 16 :=
by
  sorry

end trays_from_second_table_l245_245638


namespace perpendicular_line_through_circle_center_l245_245924

theorem perpendicular_line_through_circle_center :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (y = m * x + b) → (x = -1 ∧ y = 0) ) ∧ m = 1 ∧ b = 1 ∧ (∀ (x y : ℝ), (y = x + 1) → (x - y + 1 = 0)) :=
sorry

end perpendicular_line_through_circle_center_l245_245924


namespace apron_more_than_recipe_book_l245_245245

-- Define the prices and the total spent
def recipe_book_cost : ℕ := 6
def baking_dish_cost : ℕ := 2 * recipe_book_cost
def ingredient_cost : ℕ := 3
def total_ingredient_cost : ℕ := 5 * ingredient_cost
def total_spent : ℕ := 40

-- Define the condition that the total cost including the apron is $40
def total_without_apron : ℕ := recipe_book_cost + baking_dish_cost + total_ingredient_cost
def apron_cost : ℕ := total_spent - total_without_apron

-- Prove that the apron cost $1 more than the recipe book
theorem apron_more_than_recipe_book : apron_cost - recipe_book_cost = 1 := by
  -- The proof goes here
  sorry

end apron_more_than_recipe_book_l245_245245


namespace prove_fn_value_l245_245602

noncomputable def f (x : ℝ) : ℝ := 2^x / (2^x + 3 * x)

theorem prove_fn_value
  (m n : ℝ)
  (h1 : 2^(m + n) = 3 * m * n)
  (h2 : f m = -1 / 3) :
  f n = 4 :=
by
  sorry

end prove_fn_value_l245_245602


namespace union_intersection_l245_245975

variable (a : ℝ)

def setA (a : ℝ) : Set ℝ := { x | (x - 3) * (x - a) = 0 }
def setB : Set ℝ := {1, 4}

theorem union_intersection (a : ℝ) :
  (if a = 3 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = ∅ else 
   if a = 1 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {1} else
   if a = 4 then setA a ∪ setB = {1, 3, 4} ∧ setA a ∩ setB = {4} else
   setA a ∪ setB = {1, 3, 4, a} ∧ setA a ∩ setB = ∅) := sorry

end union_intersection_l245_245975


namespace original_price_l245_245284

-- Definitions based on the conditions
def selling_price : ℝ := 1080
def gain_percent : ℝ := 80

-- The proof problem: Prove that the cost price is Rs. 600
theorem original_price (CP : ℝ) (h_sp : CP + CP * (gain_percent / 100) = selling_price) : CP = 600 :=
by
  -- We skip the proof itself
  sorry

end original_price_l245_245284


namespace jane_project_time_l245_245083

theorem jane_project_time
  (J : ℝ)
  (work_rate_jane_ashley : ℝ := 1 / J + 1 / 40)
  (time_together : ℝ := 15.2 - 8)
  (work_done_together : ℝ := time_together * work_rate_jane_ashley)
  (ashley_alone_time : ℝ := 8)
  (work_done_ashley : ℝ := ashley_alone_time / 40)
  (jane_alone_time : ℝ := 4)
  (work_done_jane_alone : ℝ := jane_alone_time / J) :
  7.2 * (1 / J + 1 / 40) + 8 / 40 + 4 / J = 1 ↔ J = 18.06 :=
by 
  sorry

end jane_project_time_l245_245083


namespace ab_gt_c_l245_245362

theorem ab_gt_c {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 1 / a + 4 / b = 1) (hc : c < 9) : a + b > c :=
sorry

end ab_gt_c_l245_245362


namespace inv_prop_func_point_l245_245225

theorem inv_prop_func_point {k : ℝ} :
  (∃ y x : ℝ, y = k / x ∧ (x = 2 ∧ y = -1)) → k = -2 :=
by
  intro h
  -- Proof would go here
  sorry

end inv_prop_func_point_l245_245225


namespace weeks_to_fill_moneybox_l245_245247

-- Monica saves $15 every week
def savings_per_week : ℕ := 15

-- Number of cycles Monica repeats
def cycles : ℕ := 5

-- Total amount taken to the bank
def total_savings : ℕ := 4500

-- Prove that the number of weeks it takes for the moneybox to get full is 60
theorem weeks_to_fill_moneybox : ∃ W : ℕ, (cycles * savings_per_week * W = total_savings) ∧ W = 60 := 
by 
  sorry

end weeks_to_fill_moneybox_l245_245247


namespace total_students_correct_l245_245520

def num_first_graders : ℕ := 358
def num_second_graders : ℕ := num_first_graders - 64
def total_students : ℕ := num_first_graders + num_second_graders

theorem total_students_correct : total_students = 652 :=
by
  sorry

end total_students_correct_l245_245520


namespace quadratic_completing_square_t_value_l245_245984

theorem quadratic_completing_square_t_value :
  ∃ q t : ℝ, 4 * x^2 - 24 * x - 96 = 0 → (x + q) ^ 2 = t ∧ t = 33 :=
by
  sorry

end quadratic_completing_square_t_value_l245_245984


namespace correct_expression_l245_245903

theorem correct_expression :
  (2 + Real.sqrt 3 ≠ 2 * Real.sqrt 3) ∧ 
  (Real.sqrt 8 - Real.sqrt 3 ≠ Real.sqrt 5) ∧ 
  (Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6) ∧ 
  (Real.sqrt 27 / Real.sqrt 3 ≠ 9) := 
by
  sorry

end correct_expression_l245_245903


namespace valid_combinations_l245_245166

def herbs : Nat := 4
def crystals : Nat := 6
def incompatible_pairs : Nat := 3

theorem valid_combinations : 
  (herbs * crystals) - incompatible_pairs = 21 := by
  sorry

end valid_combinations_l245_245166


namespace heartsuit_ratio_l245_245034

-- Define the operation ⧡
def heartsuit (n m : ℕ) := n^(3+m) * m^(2+n)

-- The problem statement to prove
theorem heartsuit_ratio : heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l245_245034


namespace geometric_first_term_l245_245993

theorem geometric_first_term (a r : ℝ) (h1 : a * r^3 = 720) (h2 : a * r^6 = 5040) : 
a = 720 / 7 :=
by
  sorry

end geometric_first_term_l245_245993


namespace welders_that_left_first_day_l245_245861

-- Definitions of conditions
def welders := 12
def days_to_complete_order := 3
def days_remaining_work_after_first_day := 8
def work_done_first_day (r : ℝ) := welders * r * 1
def total_work (r : ℝ) := welders * r * days_to_complete_order

-- Theorem statement
theorem welders_that_left_first_day (r : ℝ) : 
  ∃ x : ℝ, 
    (welders - x) * r * days_remaining_work_after_first_day = total_work r - work_done_first_day r 
    ∧ x = 9 :=
by
  sorry

end welders_that_left_first_day_l245_245861


namespace students_prefer_windows_l245_245419

theorem students_prefer_windows (total_students students_prefer_mac equally_prefer_both no_preference : ℕ) 
  (h₁ : total_students = 210)
  (h₂ : students_prefer_mac = 60)
  (h₃ : equally_prefer_both = 20)
  (h₄ : no_preference = 90) :
  total_students - students_prefer_mac - equally_prefer_both - no_preference = 40 := 
  by
    -- Proof goes here
    sorry

end students_prefer_windows_l245_245419


namespace sqrt_sqrt_81_is_9_l245_245656

theorem sqrt_sqrt_81_is_9 : Real.sqrt (Real.sqrt 81) = 3 := sorry

end sqrt_sqrt_81_is_9_l245_245656


namespace quadratic_two_distinct_real_roots_l245_245744

theorem quadratic_two_distinct_real_roots (k : ℝ) :
    (∃ x : ℝ, 2 * k * x^2 + (8 * k + 1) * x + 8 * k = 0 ∧ 2 * k ≠ 0) →
    k > -1/16 ∧ k ≠ 0 :=
by
  intro h
  sorry

end quadratic_two_distinct_real_roots_l245_245744


namespace probability_even_sum_l245_245349

def first_ten_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

def distinct_pairs (l : List ℕ) : List (ℕ × ℕ) :=
  (l.product l).filter (λ p, p.1 < p.2)

theorem probability_even_sum : 
  (first_ten_primes.length = 10) →
  (∀ a b : ℕ, a ∈ first_ten_primes → b ∈ first_ten_primes → a ≠ b → 
    ((a + b) % 2 = 0 ↔ 2 ∉ [a, b])) →
  (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat) 
    = 36 → 
  ((∑ pair in distinct_pairs first_ten_primes, (1:ℚ)) ⁻¹ * 
   (∑ pair in distinct_pairs first_ten_primes, ((pair.1 + pair.2) % 2 = 0).toNat))
    = 4 / 5 := by
  sorry

end probability_even_sum_l245_245349


namespace length_BC_l245_245963

theorem length_BC (AB AC AM : ℝ)
  (hAB : AB = 5)
  (hAC : AC = 7)
  (hAM : AM = 4)
  (M_midpoint_of_BC : ∃ (BM MC : ℝ), BM = MC ∧ ∀ (BC: ℝ), BC = BM + MC) :
  ∃ (BC : ℝ), BC = 2 * Real.sqrt 21 := by
  sorry

end length_BC_l245_245963


namespace probability_boxes_l245_245539

def box_A_tiles := (Finset.range 30).map (λ n, n+1)
def box_B_tiles := (Finset.range 20).map (λ n, n+21)

def prob_box_A_less_20 : nnreal :=
let favorable_A := (Finset.range 20).map (λ n, n+1) in
(favorable_A.card : nnreal) / (box_A_tiles.card : nnreal)

def prob_box_B_odd_or_greater_35 : nnreal :=
let odd_B := (Finset.range 10).filter (λ n, (n+21) % 2 = 1) in
let greater_35 := (Finset.range (40-35+1)).map (λ n, n+35) in
((odd_B.card + greater_35.card - odd_B.bUnion (λ n, if n + 21 >= 35 then {n+21} else ∅ ).card) : nnreal) / (box_B_tiles.card : nnreal)

theorem probability_boxes :
  prob_box_A_less_20 * prob_box_B_odd_or_greater_35 = 19 / 50 := by
sorry

end probability_boxes_l245_245539


namespace grace_age_l245_245606

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end grace_age_l245_245606


namespace rotation_volumes_l245_245055

theorem rotation_volumes (a b c V1 V2 V3 : ℝ) (h : a^2 + b^2 = c^2)
    (hV1 : V1 = (1 / 3) * Real.pi * a^2 * b^2 / c)
    (hV2 : V2 = (1 / 3) * Real.pi * b^2 * a)
    (hV3 : V3 = (1 / 3) * Real.pi * a^2 * b) : 
    (1 / V1^2) = (1 / V2^2) + (1 / V3^2) :=
sorry

end rotation_volumes_l245_245055


namespace largest_4_digit_divisible_by_50_l245_245269

-- Define the condition for a 4-digit number
def is_4_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

-- Define the largest 4-digit number
def largest_4_digit : ℕ := 9999

-- Define the property that a number is exactly divisible by 50
def divisible_by_50 (n : ℕ) : Prop := n % 50 = 0

-- Main statement to be proved
theorem largest_4_digit_divisible_by_50 :
  ∃ n, is_4_digit n ∧ divisible_by_50 n ∧ ∀ m, is_4_digit m → divisible_by_50 m → m ≤ n ∧ n = 9950 :=
by
  sorry

end largest_4_digit_divisible_by_50_l245_245269


namespace problem_statement_l245_245343

def oper (x : ℕ) (w : ℕ) := (2^x) / (2^w)

theorem problem_statement : ∃ n : ℕ, oper (oper 4 2) n = 2 ↔ n = 3 :=
by sorry

end problem_statement_l245_245343


namespace gcd_of_60_and_75_l245_245140

theorem gcd_of_60_and_75 : Nat.gcd 60 75 = 15 := by
  -- Definitions based on the conditions
  have factorization_60 : Nat.factors 60 = [2, 2, 3, 5] := rfl
  have factorization_75 : Nat.factors 75 = [3, 5, 5] := rfl
  
  -- Sorry as the placeholder for the proof
  sorry

end gcd_of_60_and_75_l245_245140


namespace problem_statement_l245_245047

def base7_representation (n : ℕ) : ℕ :=
  let rec digits (n : ℕ) (acc : ℕ) (power : ℕ) : ℕ :=
    if n = 0 then acc
    else digits (n / 7) (acc + (n % 7) * power) (power * 10)
  digits n 0 1

def even_digits_count (n : ℕ) : ℕ :=
  let rec count (n : ℕ) (acc : ℕ) : ℕ :=
    if n = 0 then acc
    else let d := n % 10 in
         count (n / 10) (if d % 2 = 0 then acc + 1 else acc)
  count n 0

theorem problem_statement : even_digits_count (base7_representation 528) = 0 := sorry

end problem_statement_l245_245047


namespace least_possible_b_l245_245522

noncomputable def a : ℕ := 8

theorem least_possible_b (b : ℕ) (h1 : ∀ n : ℕ, n > 0 → a.factors.count n = 1 → a = n^3)
  (h2 : b.factors.count a = 1)
  (h3 : b % a = 0) :
  b = 24 :=
sorry

end least_possible_b_l245_245522


namespace upper_side_length_l245_245341

variable (L U h : ℝ)

-- Given conditions
def condition1 : Prop := U = L - 6
def condition2 : Prop := 72 = (1 / 2) * (L + U) * 8
def condition3 : Prop := h = 8

-- The length of the upper side of the trapezoid
theorem upper_side_length (h : h = 8) (c1 : U = L - 6) (c2 : 72 = (1 / 2) * (L + U) * 8) : U = 6 := 
by
  sorry

end upper_side_length_l245_245341


namespace gcd_60_75_l245_245143

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l245_245143


namespace april_earnings_l245_245904

def price_per_rose := 7
def price_per_lily := 5
def initial_roses := 9
def initial_lilies := 6
def remaining_roses := 4
def remaining_lilies := 2

def total_roses_sold := initial_roses - remaining_roses
def total_lilies_sold := initial_lilies - remaining_lilies

def total_earnings := (total_roses_sold * price_per_rose) + (total_lilies_sold * price_per_lily)

theorem april_earnings : total_earnings = 55 := by
  sorry

end april_earnings_l245_245904


namespace prime_cubed_plus_prime_plus_one_not_square_l245_245257

theorem prime_cubed_plus_prime_plus_one_not_square (p : ℕ) (hp : Nat.Prime p) :
  ¬ ∃ k : ℕ, k * k = p^3 + p + 1 :=
by
  sorry

end prime_cubed_plus_prime_plus_one_not_square_l245_245257


namespace adult_meal_cost_l245_245172

theorem adult_meal_cost (x : ℝ) 
  (total_people : ℕ) (kids : ℕ) (total_cost : ℝ)  
  (h_total_people : total_people = 11) 
  (h_kids : kids = 2) 
  (h_total_cost : total_cost = 72)
  (h_adult_meals : (total_people - kids : ℕ) • x = total_cost) : 
  x = 8 := 
by
  -- Proof will go here
  sorry

end adult_meal_cost_l245_245172


namespace petya_square_larger_l245_245791

noncomputable def dimension_petya_square (a b : ℝ) : ℝ :=
  (a * b) / (a + b)

noncomputable def dimension_vasya_square (a b : ℝ) : ℝ :=
  (a * b * Real.sqrt (a^2 + b^2)) / (a^2 + a * b + b^2)

theorem petya_square_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  dimension_vasya_square a b < dimension_petya_square a b :=
by
  sorry

end petya_square_larger_l245_245791


namespace salt_quantity_l245_245663

-- Conditions translated to Lean definitions
def cost_of_sugar_per_kg : ℝ := 1.50
def total_cost_sugar_2kg_and_salt (x : ℝ) : ℝ := 5.50
def total_cost_sugar_3kg_and_1kg_salt : ℝ := 5.00

-- Theorem statement
theorem salt_quantity (x : ℝ) : 
  2 * cost_of_sugar_per_kg + x * cost_of_sugar_per_kg / 3 = total_cost_sugar_2kg_and_salt x 
  → 3 * cost_of_sugar_per_kg + x = total_cost_sugar_3kg_and_1kg_salt 
  → x = 5 := 
sorry

end salt_quantity_l245_245663


namespace sufficient_but_not_necessary_condition_l245_245855

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end sufficient_but_not_necessary_condition_l245_245855


namespace first_tree_height_l245_245032

theorem first_tree_height
  (branches_first : ℕ)
  (branches_second : ℕ)
  (height_second : ℕ)
  (branches_third : ℕ)
  (height_third : ℕ)
  (branches_fourth : ℕ)
  (height_fourth : ℕ)
  (average_branches_per_foot : ℕ) :
  branches_first = 200 →
  height_second = 40 →
  branches_second = 180 →
  height_third = 60 →
  branches_third = 180 →
  height_fourth = 34 →
  branches_fourth = 153 →
  average_branches_per_foot = 4 →
  branches_first / average_branches_per_foot = 50 :=
by
  sorry

end first_tree_height_l245_245032


namespace inequality_ineq_l245_245978

theorem inequality_ineq (x y : ℝ) (hx: x > Real.sqrt 2) (hy: y > Real.sqrt 2) : 
  x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 := 
  sorry

end inequality_ineq_l245_245978


namespace dot_product_is_constant_l245_245468

-- Define the trajectory C as the parabola given by the equation y^2 = 4x
def trajectory (x y : ℝ) : Prop := y^2 = 4 * x

-- Prove the range of k for the line passing through point (-1, 0) and intersecting trajectory C
def valid_slope (k : ℝ) : Prop := (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)

-- Prove that ∀ D ≠ A, B on the parabola y^2 = 4x, and lines DA and DB intersect vertical line through (1, 0) on points P, Q, OP ⋅ OQ = 5
theorem dot_product_is_constant (D A B P Q : ℝ × ℝ) 
  (hD : trajectory D.1 D.2)
  (hA : trajectory A.1 A.2)
  (hB : trajectory B.1 B.2)
  (hDiff : D ≠ A ∧ D ≠ B)
  (hP : P = (1, (D.2 * A.2 + 4) / (D.2 + A.2))) 
  (hQ : Q = (1, (D.2 * B.2 + 4) / (D.2 + B.2))) :
  (1 + (D.2 * A.2 + 4) / (D.2 + A.2)) * (1 + (D.2 * B.2 + 4) / (D.2 + B.2)) = 5 :=
sorry

end dot_product_is_constant_l245_245468


namespace curve_is_hyperbola_l245_245314

theorem curve_is_hyperbola (u : ℝ) (x y : ℝ) 
  (h1 : x = Real.cos u ^ 2)
  (h2 : y = Real.sin u ^ 4) : 
  ∃ (a b : ℝ), a ≠ 0 ∧  b ≠ 0 ∧ x / a ^ 2 - y / b ^ 2 = 1 := 
sorry

end curve_is_hyperbola_l245_245314


namespace slope_angle_of_line_l245_245666

theorem slope_angle_of_line (θ : ℝ) : 
  (∃ m : ℝ, ∀ x y : ℝ, 4 * x + y - 1 = 0 ↔ y = m * x + 1) ∧ (m = -4) → 
  θ = Real.pi - Real.arctan 4 :=
by
  sorry

end slope_angle_of_line_l245_245666


namespace min_value_g_squared_plus_f_l245_245332

def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_g_squared_plus_f (a b c : ℝ) (h : a ≠ 0) 
  (min_f_squared_plus_g : ∀ x : ℝ, (f a b x)^2 + g a c x ≥ 4)
  (exists_x_min : ∃ x : ℝ, (f a b x)^2 + g a c x = 4) :
  ∃ x : ℝ, (g a c x)^2 + f a b x = -9 / 2 :=
sorry

end min_value_g_squared_plus_f_l245_245332


namespace largest_number_of_cakes_without_ingredients_l245_245639

theorem largest_number_of_cakes_without_ingredients :
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  ∃ (max_no_ingredients : ℕ), max_no_ingredients = 24 :=
by
  let total_cakes := 60
  let cakes_with_strawberries := (1 / 3) * total_cakes
  let cakes_with_blueberries := (1 / 2) * total_cakes
  let cakes_with_raspberries := (3 / 5) * total_cakes
  let cakes_with_coconut := (1 / 10) * total_cakes
  existsi (60 - max 20 (max 30 (max 36 6))) -- max value should be used to reflect maximum coverage content
  sorry -- Proof to be completed

end largest_number_of_cakes_without_ingredients_l245_245639


namespace problem_statement_l245_245199

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum of the first n terms of the sequence
variable (d : ℝ) -- the common difference
variable (a1 : ℝ) -- the first term

-- Conditions
axiom arithmetic_sequence (n : ℕ) : a n = a1 + (n - 1) * d
axiom sum_of_first_n_terms (n : ℕ) : S n = n * (a1 + a n) / 2
axiom S_15_eq_45 : S 15 = 45

-- The statement to prove
theorem problem_statement : 2 * a 12 - a 16 = 3 :=
by
  sorry

end problem_statement_l245_245199


namespace table_price_l245_245427

theorem table_price
  (C T : ℝ)
  (h1 : 2 * C + T = 0.6 * (C + 2 * T))
  (h2 : C + T = 60) :
  T = 52.5 :=
by
  sorry

end table_price_l245_245427


namespace scientific_notation_of_284000000_l245_245711

/--
Given the number 284000000, prove that it can be expressed in scientific notation as 2.84 * 10^8.
-/
theorem scientific_notation_of_284000000 :
  284000000 = 2.84 * 10^8 :=
sorry

end scientific_notation_of_284000000_l245_245711


namespace wait_time_difference_l245_245668

noncomputable def kids_waiting_for_swings : ℕ := 3
noncomputable def kids_waiting_for_slide : ℕ := 2 * kids_waiting_for_swings
noncomputable def wait_per_kid_swings : ℕ := 2 * 60 -- 2 minutes in seconds
noncomputable def wait_per_kid_slide : ℕ := 15 -- in seconds

noncomputable def total_wait_swings : ℕ := kids_waiting_for_swings * wait_per_kid_swings
noncomputable def total_wait_slide : ℕ := kids_waiting_for_slide * wait_per_kid_slide

theorem wait_time_difference : total_wait_swings - total_wait_slide = 270 := by
  sorry

end wait_time_difference_l245_245668


namespace number_of_valid_paths_l245_245445

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_valid_paths (n : ℕ) :
  let valid_paths := binomial (2 * n) n / (n + 1)
  valid_paths = binomial (2 * n) n - binomial (2 * n) (n + 1) := 
sorry

end number_of_valid_paths_l245_245445


namespace g_of_f_neg_5_l245_245364

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 8

-- Assume g(42) = 17
axiom g_f_5_eq_17 : ∀ (g : ℝ → ℝ), g (f 5) = 17

-- State the theorem to be proven
theorem g_of_f_neg_5 (g : ℝ → ℝ) : g (f (-5)) = 17 :=
by
  sorry

end g_of_f_neg_5_l245_245364


namespace solve_quadratic_eq_l245_245516

theorem solve_quadratic_eq (x : ℝ) :
  x^2 + 4 * x + 2 = 0 ↔ (x = -2 + Real.sqrt 2 ∨ x = -2 - Real.sqrt 2) :=
by
  -- This is a statement only. No proof is required.
  sorry

end solve_quadratic_eq_l245_245516


namespace Emily_beads_l245_245041

-- Define the conditions and question
theorem Emily_beads (n k : ℕ) (h1 : k = 4) (h2 : n = 5) : n * k = 20 := by
  -- Sorry: this is a placeholder for the actual proof
  sorry

end Emily_beads_l245_245041


namespace number_of_integers_l245_245178

open Int

theorem number_of_integers (n : ℤ) :
  (1 + (floor (120 * n / 121) : ℤ) = (ceil (119 * n / 120) : ℤ)) ↔ (n % 14520 = 0) :=
sorry

end number_of_integers_l245_245178


namespace right_triangle_sides_l245_245801

/-- Given a right triangle with area 2 * r^2 / 3 where r is the radius of a circle touching one leg,
the extension of the other leg, and the hypotenuse, the sides of the triangle are given by r, 4/3 * r, and 5/3 * r. -/
theorem right_triangle_sides (r : ℝ) (x y : ℝ)
  (h_area : (x * y) / 2 = 2 * r^2 / 3)
  (h_hypotenuse : (x^2 + y^2) = (2 * r + x - y)^2) :
  x = r ∧ y = 4 * r / 3 :=
sorry

end right_triangle_sides_l245_245801


namespace Vlad_score_l245_245378

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end Vlad_score_l245_245378


namespace find_x_if_perpendicular_l245_245067

-- Define the vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (x, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x - 5)

-- Define the condition that vectors a and b are perpendicular
def perpendicular (x : ℝ) : Prop :=
  let a := vector_a x
  let b := vector_b x
  a.1 * b.1 + a.2 * b.2 = 0

-- Prove that x = 3 if a and b are perpendicular
theorem find_x_if_perpendicular :
  ∃ x : ℝ, perpendicular x ∧ x = 3 :=
by
  sorry

end find_x_if_perpendicular_l245_245067


namespace condition1_not_sufficient_nor_necessary_condition2_necessary_l245_245454

variable (x y : ℝ)

-- ① Neither sufficient nor necessary
theorem condition1_not_sufficient_nor_necessary (h1 : x ≠ 1 ∧ y ≠ 2) : ¬ ((x ≠ 1 ∧ y ≠ 2) → x + y ≠ 3) ∧ ¬ (x + y ≠ 3 → x ≠ 1 ∧ y ≠ 2) := sorry

-- ② Necessary condition
theorem condition2_necessary (h2 : x ≠ 1 ∨ y ≠ 2) : x + y ≠ 3 → (x ≠ 1 ∨ y ≠ 2) := sorry

end condition1_not_sufficient_nor_necessary_condition2_necessary_l245_245454


namespace sum_of_five_consecutive_even_integers_l245_245834

theorem sum_of_five_consecutive_even_integers (a : ℤ) 
  (h : a + (a + 4) = 144) : a + (a + 2) + (a + 4) + (a + 6) + (a + 8) = 370 := by
  sorry

end sum_of_five_consecutive_even_integers_l245_245834


namespace problem_statement_l245_245971

-- Define A as the number of four-digit odd numbers
def A : ℕ := 4500

-- Define B as the number of four-digit multiples of 3
def B : ℕ := 3000

-- The main theorem stating the sum A + B equals 7500
theorem problem_statement : A + B = 7500 := by
  -- The exact proof is omitted using sorry
  sorry

end problem_statement_l245_245971


namespace eval_expression_l245_245307

theorem eval_expression (a : ℕ) (h : a = 2) : 
  8^3 + 4 * a * 8^2 + 6 * a^2 * 8 + a^3 = 1224 := 
by
  rw [h]
  sorry

end eval_expression_l245_245307


namespace colby_mango_sales_l245_245302

theorem colby_mango_sales
  (total_kg : ℕ)
  (mangoes_per_kg : ℕ)
  (remaining_mangoes : ℕ)
  (half_sold_to_market : ℕ) :
  total_kg = 60 →
  mangoes_per_kg = 8 →
  remaining_mangoes = 160 →
  half_sold_to_market = 20 := by
    sorry

end colby_mango_sales_l245_245302


namespace larger_number_is_72_l245_245612

theorem larger_number_is_72 (a b : ℕ) (h1 : 5 * b = 6 * a) (h2 : b - a = 12) : b = 72 :=
by
  sorry

end larger_number_is_72_l245_245612


namespace total_length_of_sticks_l245_245771

-- Definitions of stick lengths based on the conditions
def length_first_stick : ℕ := 3
def length_second_stick : ℕ := 2 * length_first_stick
def length_third_stick : ℕ := length_second_stick - 1

-- Proof statement
theorem total_length_of_sticks : length_first_stick + length_second_stick + length_third_stick = 14 :=
by
  sorry

end total_length_of_sticks_l245_245771


namespace alice_number_l245_245716

theorem alice_number (n : ℕ) 
  (h1 : 180 ∣ n) 
  (h2 : 75 ∣ n) 
  (h3 : 900 ≤ n) 
  (h4 : n ≤ 3000) : 
  n = 900 ∨ n = 1800 ∨ n = 2700 := 
by
  sorry

end alice_number_l245_245716


namespace left_handed_ratio_l245_245099

-- Given the conditions:
-- total number of players
def total_players : ℕ := 70
-- number of throwers who are all right-handed 
def throwers : ℕ := 37 
-- total number of right-handed players
def right_handed : ℕ := 59

-- Define the necessary variables based on the given conditions.
def non_throwers : ℕ := total_players - throwers
def non_throwing_right_handed : ℕ := right_handed - throwers
def left_handed_non_throwers : ℕ := non_throwers - non_throwing_right_handed

-- State the theorem to prove that the ratio of 
-- left-handed non-throwers to the rest of the team (excluding throwers) is 1:3
theorem left_handed_ratio : 
  (left_handed_non_throwers : ℚ) / (non_throwers : ℚ) = 1 / 3 := by
    sorry

end left_handed_ratio_l245_245099


namespace center_of_circle_l245_245260

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 10) (h4 : y2 = 7) :
  (x1 + x2) / 2 = 6 ∧ (y1 + y2) / 2 = 2 :=
by
  rw [h1, h2, h3, h4]
  constructor
  · norm_num
  · norm_num

end center_of_circle_l245_245260


namespace solve_inequality_l245_245601

variable {c : ℝ}
variable (h_c_ne_2 : c ≠ 2)

theorem solve_inequality :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - (1 + 2) * x + 2 ≤ 0) ∧
  (c > 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x > c ∨ x < 2)) ∧
  (c < 2 → (∀ x : ℝ, (x - c) * (x - 2) > 0 ↔ x < c ∨ x > 2)) :=
by
  sorry

end solve_inequality_l245_245601


namespace fibonacci_determinant_identity_l245_245943

def fibonacci_matrix (n : ℕ) : Matrix (Fin 2) (Fin 2) ℕ :=
  ![![Nat.fib (n + 1), Nat.fib n], ![Nat.fib n, Nat.fib (n - 1)]]

theorem fibonacci_determinant_identity :
  ∀ n : ℕ, Matrix.det (fibonacci_matrix n) = (-1 : ℤ)^n :=
by sorry

example : Nat.fib 100 * Nat.fib 102 - Nat.fib 101 ^ 2 = -1 := 
by {
  have f_identity := fibonacci_determinant_identity 101,
  simp [fibonacci_matrix, Matrix.det] at f_identity,
  exact f_identity,
}

end fibonacci_determinant_identity_l245_245943


namespace friends_picked_strawberries_with_Lilibeth_l245_245095

-- Define the conditions
def Lilibeth_baskets : ℕ := 6
def strawberries_per_basket : ℕ := 50
def total_strawberries : ℕ := 1200

-- Define the calculation of strawberries picked by Lilibeth
def Lilibeth_strawberries : ℕ := Lilibeth_baskets * strawberries_per_basket

-- Define the calculation of strawberries picked by friends
def friends_strawberries : ℕ := total_strawberries - Lilibeth_strawberries

-- Define the number of friends who picked strawberries
def friends_picked_with_Lilibeth : ℕ := friends_strawberries / Lilibeth_strawberries

-- The theorem we need to prove
theorem friends_picked_strawberries_with_Lilibeth : friends_picked_with_Lilibeth = 3 :=
by
  -- Proof goes here
  sorry

end friends_picked_strawberries_with_Lilibeth_l245_245095


namespace greatest_solution_of_equation_l245_245045

theorem greatest_solution_of_equation : ∀ x : ℝ, x ≠ 9 ∧ (x^2 - x - 90) / (x - 9) = 4 / (x + 6) → x ≤ -7 :=
by
  intros x hx
  sorry

end greatest_solution_of_equation_l245_245045


namespace relationship_among_a_b_c_l245_245318

noncomputable def a : ℝ := (0.8 : ℝ)^(5.2 : ℝ)
noncomputable def b : ℝ := (0.8 : ℝ)^(5.5 : ℝ)
noncomputable def c : ℝ := (5.2 : ℝ)^(0.1 : ℝ)

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l245_245318


namespace lauren_time_8_miles_l245_245360

-- Conditions
def time_alex_run_6_miles : ℕ := 36
def time_lauren_run_5_miles : ℕ := time_alex_run_6_miles / 3
def time_per_mile_lauren : ℚ := time_lauren_run_5_miles / 5

-- Proof statement
theorem lauren_time_8_miles : 8 * time_per_mile_lauren = 19.2 := by
  sorry

end lauren_time_8_miles_l245_245360


namespace tom_pays_l245_245123

-- Definitions based on the conditions
def number_of_lessons : Nat := 10
def cost_per_lesson : Nat := 10
def free_lessons : Nat := 2

-- Desired proof statement
theorem tom_pays {number_of_lessons cost_per_lesson free_lessons : Nat} :
  (number_of_lessons - free_lessons) * cost_per_lesson = 80 :=
by
  sorry

end tom_pays_l245_245123


namespace evaluate_expression_l245_245634

def greatest_power_of_factor_2 (n : ℕ) : ℕ :=
  (nat.factors n).count 2

def greatest_power_of_factor_5 (n : ℕ) : ℕ :=
  (nat.factors n).count 5

theorem evaluate_expression (a b : ℕ) (h₁ : 2^a = 8) (h₂ : 5^b = 25) :
  (1 / 3) ^ (b - a) = 3 := by
  have ha : a = greatest_power_of_factor_2 200 := by sorry
  have hb : b = greatest_power_of_factor_5 200 := by sorry
  rw [greatest_power_of_factor_2, greatest_power_of_factor_5] at ha hb
  simp at ha hb
  exact sorry

end evaluate_expression_l245_245634


namespace largest_number_l245_245523

theorem largest_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 42) (h_dvd_a : 42 ∣ a) (h_dvd_b : 42 ∣ b)
  (a_eq : a = 42 * 11) (b_eq : b = 42 * 12) : max a b = 504 := by
  sorry

end largest_number_l245_245523


namespace power_is_seventeen_l245_245665

theorem power_is_seventeen (x : ℕ) : (1000^7 : ℝ) / (10^x) = (10000 : ℝ) ↔ x = 17 := by
  sorry

end power_is_seventeen_l245_245665


namespace green_beans_to_onions_ratio_l245_245509

def cut_conditions
  (potatoes : ℕ)
  (carrots : ℕ)
  (onions : ℕ)
  (green_beans : ℕ) : Prop :=
  carrots = 6 * potatoes ∧ onions = 2 * carrots ∧ potatoes = 2 ∧ green_beans = 8

theorem green_beans_to_onions_ratio (potatoes carrots onions green_beans : ℕ) :
  cut_conditions potatoes carrots onions green_beans →
  green_beans / gcd green_beans onions = 1 ∧ onions / gcd green_beans onions = 3 :=
by
  sorry

end green_beans_to_onions_ratio_l245_245509


namespace incorrect_square_root_0_2_l245_245274

theorem incorrect_square_root_0_2 :
  (0.45)^2 = 0.2 ∧ (0.02)^2 ≠ 0.2 :=
by
  sorry

end incorrect_square_root_0_2_l245_245274


namespace sheila_hourly_wage_l245_245007

-- Sheila works 8 hours per day on Monday, Wednesday, and Friday
-- Sheila works 6 hours per day on Tuesday and Thursday
-- Sheila does not work on Saturday and Sunday
-- Sheila earns $288 per week

def hours_worked (monday_wednesday_friday_hours : Nat) (tuesday_thursday_hours : Nat) : Nat :=
  (monday_wednesday_friday_hours * 3) + (tuesday_thursday_hours * 2)

def weekly_earnings : Nat := 288
def total_hours_worked : Nat := hours_worked 8 6
def hourly_wage : Nat := weekly_earnings / total_hours_worked

theorem sheila_hourly_wage : hourly_wage = 8 := by
  -- Proof (omitted)
  sorry

end sheila_hourly_wage_l245_245007


namespace triangle_inequality_equivalence_l245_245104

theorem triangle_inequality_equivalence
    (a b c : ℝ) :
  (a < b + c ∧ b < a + c ∧ c < a + b) ↔
  (|b - c| < a ∧ a < b + c ∧ |a - c| < b ∧ b < a + c ∧ |a - b| < c ∧ c < a + b) ∧
  (max a (max b c) < b + c ∧ max a (max b c) < a + c ∧ max a (max b c) < a + b) :=
by sorry

end triangle_inequality_equivalence_l245_245104


namespace solve_quadratic_l245_245515

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end solve_quadratic_l245_245515


namespace parabola_2_second_intersection_x_l245_245161

-- Definitions of the conditions in the problem
def parabola_1_intersects : Prop := 
  (∀ x : ℝ, (x = 10 ∨ x = 13) → (∃ y : ℝ, (x, y) ∈ ({p | p = (10, 0)} ∪ {p | p = (13, 0)})))

def parabola_2_intersects : Prop := 
  (∃ x : ℝ, x = 13)

def vertex_bisects_segment : Prop := 
  (∃ a : ℝ, 2 * 11.5 = a)

-- The theorem we want to prove
theorem parabola_2_second_intersection_x : 
  parabola_1_intersects ∧ parabola_2_intersects ∧ vertex_bisects_segment → 
  (∃ t : ℝ, t = 33) := 
  by
  sorry

end parabola_2_second_intersection_x_l245_245161


namespace henry_games_given_l245_245218

theorem henry_games_given (G : ℕ) (henry_initial : ℕ) (neil_initial : ℕ) (henry_now : ℕ) (neil_now : ℕ) :
  henry_initial = 58 →
  neil_initial = 7 →
  henry_now = henry_initial - G →
  neil_now = neil_initial + G →
  henry_now = 4 * neil_now →
  G = 6 :=
by
  intros h_initial n_initial h_now n_now eq_henry
  sorry

end henry_games_given_l245_245218


namespace glass_volume_correct_l245_245890

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l245_245890


namespace weight_replacement_proof_l245_245802

noncomputable def weight_of_replaced_person (increase_in_average_weight new_person_weight : ℝ) : ℝ :=
  new_person_weight - (5 * increase_in_average_weight)

theorem weight_replacement_proof (h1 : ∀ w : ℝ, increase_in_average_weight = 5.5) (h2 : new_person_weight = 95.5) :
  weight_of_replaced_person 5.5 95.5 = 68 := by
  sorry

end weight_replacement_proof_l245_245802


namespace card_probability_sequence_l245_245461

/-- 
Four cards are dealt at random from a standard deck of 52 cards without replacement.
The probability that the first card is a Jack, the second card is a Queen, the third card is a King, and the fourth card is an Ace is given by:
-/
theorem card_probability_sequence :
  let p1 := 4 / 52,
      p2 := 4 / 51,
      p3 := 4 / 50,
      p4 := 4 / 49
  in p1 * p2 * p3 * p4 = 64 / 1624350 :=
by
  let p1 := 4 / 52
  let p2 := 4 / 51
  let p3 := 4 / 50
  let p4 := 4 / 49
  show p1 * p2 * p3 * p4 = 64 / 1624350
  sorry

end card_probability_sequence_l245_245461


namespace range_of_m_l245_245477

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  ∀ m, m = 9 / 4 → (1 / x + 4 / y) ≥ m := 
by
  sorry

end range_of_m_l245_245477


namespace lemonade_water_l245_245508

theorem lemonade_water (L S W : ℝ) (h1 : S = 1.5 * L) (h2 : W = 3 * S) (h3 : L = 4) : W = 18 :=
by
  sorry

end lemonade_water_l245_245508


namespace sequence_count_less_than_1969_l245_245674

-- Define the function f that describes the number of terms needed for sequences ending in n
def f (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 2
  else if n = 4 then 2
  else 1 + (Nat.sqrt n).sum (λ i, f i)

-- Define the main theorem
theorem sequence_count_less_than_1969 : f 1969 < 1969 :=
  sorry

end sequence_count_less_than_1969_l245_245674


namespace fraction_simplest_sum_l245_245809

theorem fraction_simplest_sum (a b : ℕ) (ha : a > 0) (hb : b > 0) 
  (h : (3975 : ℚ) / 10000 = (a : ℚ) / b) 
  (simp : ∀ (c : ℕ), c ∣ a ∧ c ∣ b → c = 1) : a + b = 559 :=
sorry

end fraction_simplest_sum_l245_245809


namespace smallest_m_l245_245765

theorem smallest_m (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : n - m / n = 2011 / 3) : m = 1120 :=
sorry

end smallest_m_l245_245765


namespace distribute_items_in_identical_bags_l245_245573

noncomputable def count_ways_to_distribute_items (num_items : ℕ) (num_bags : ℕ) : ℕ :=
  if h : num_items = 5 ∧ num_bags = 3 then 36 else 0

theorem distribute_items_in_identical_bags :
  count_ways_to_distribute_items 5 3 = 36 :=
by
  -- Proof is skipped as per instructions
  sorry

end distribute_items_in_identical_bags_l245_245573


namespace expectation_S_tau_eq_varliminf_ratio_S_tau_l245_245361

noncomputable def xi : ℕ → ℝ := sorry
noncomputable def tau : ℝ := sorry

-- Statement (a)
theorem expectation_S_tau_eq (ES_tau : ℝ := sorry) (E_tau : ℝ := sorry) (E_xi1 : ℝ := sorry) :
  ES_tau = E_tau * E_xi1 := sorry

-- Statement (b)
theorem varliminf_ratio_S_tau (liminf_val : ℝ := sorry) (E_tau : ℝ := sorry) :
  (liminf_val = E_tau) := sorry

end expectation_S_tau_eq_varliminf_ratio_S_tau_l245_245361


namespace problem_evaluation_l245_245313

theorem problem_evaluation (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹ + d⁻¹) * (ab + bc + cd + da + ac + bd)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (cd)⁻¹ + (da)⁻¹ + (ac)⁻¹ + (bd)⁻¹) = 
  (1 / (a * b * c * d)) * (1 / (a * b * c * d)) :=
by
  sorry

end problem_evaluation_l245_245313


namespace compare_travel_times_l245_245079

variable (v : ℝ) (t1 t2 : ℝ)

def travel_time_first := t1 = 100 / v
def travel_time_second := t2 = 200 / v

theorem compare_travel_times (h1 : travel_time_first v t1) (h2 : travel_time_second v t2) : 
  t2 = 2 * t1 :=
by
  sorry

end compare_travel_times_l245_245079


namespace balls_of_yarn_per_sweater_l245_245913

-- Define the conditions as constants
def cost_per_ball := 6
def sell_price_per_sweater := 35
def total_gain := 308
def number_of_sweaters := 28

-- Define a function that models the total gain given the number of balls of yarn per sweater.
def total_gain_formula (x : ℕ) : ℕ :=
  number_of_sweaters * (sell_price_per_sweater - cost_per_ball * x)

-- State the theorem which proves the number of balls of yarn per sweater
theorem balls_of_yarn_per_sweater (x : ℕ) (h : total_gain_formula x = total_gain): x = 4 :=
sorry

end balls_of_yarn_per_sweater_l245_245913


namespace xia_sheets_left_l245_245901

def stickers_left (initial : ℕ) (shared : ℕ) (per_sheet : ℕ) : ℕ :=
  (initial - shared) / per_sheet

theorem xia_sheets_left :
  stickers_left 150 100 10 = 5 :=
by
  sorry

end xia_sheets_left_l245_245901


namespace An_integer_and_parity_l245_245840

theorem An_integer_and_parity (k : Nat) (h : k > 0) : 
  ∀ n ≥ 1, ∃ A : Nat, 
   (A = 1 ∨ (∀ A' : Nat, A' = ( (A * n + 2 * (n+1) ^ (2 * k)) / (n+2)))) 
  ∧ (A % 2 = 1 ↔ n % 4 = 1 ∨ n % 4 = 2) := 
by 
  sorry

end An_integer_and_parity_l245_245840


namespace no_snuggly_two_digit_l245_245165

theorem no_snuggly_two_digit (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) : ¬ (10 * a + b = a + b^3) :=
by {
  sorry
}

end no_snuggly_two_digit_l245_245165


namespace sufficient_but_not_necessary_for_circle_l245_245659

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (m = 0 → ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0) ∧ ¬(∀m, ∃ x y : ℝ, x^2 + y^2 - 4 * x + 2 * y + m = 0 → m = 0) :=
 by
  sorry

end sufficient_but_not_necessary_for_circle_l245_245659


namespace parallelogram_base_length_l245_245153

variable (base height : ℝ)
variable (Area : ℝ)

theorem parallelogram_base_length (h₁ : Area = 162) (h₂ : height = 2 * base) (h₃ : Area = base * height) : base = 9 := 
by
  sorry

end parallelogram_base_length_l245_245153


namespace minimum_crooks_l245_245495

theorem minimum_crooks (total_ministers : ℕ)
  (h_total : total_ministers = 100)
  (cond : ∀ (s : finset ℕ), s.card = 10 → ∃ m ∈ s, m > 90) :
  ∃ crooks ≥ 91, crooks + (total_ministers - crooks) = total_ministers :=
by
  -- We need to prove that there are at least 91 crooks.
  sorry

end minimum_crooks_l245_245495


namespace BrotherUpperLimit_l245_245759

variable (w : ℝ) -- Arun's weight
variable (b : ℝ) -- Upper limit of Arun's weight according to his brother's opinion

-- Conditions as per the problem
def ArunOpinion (w : ℝ) := 64 < w ∧ w < 72
def BrotherOpinion (w b : ℝ) := 60 < w ∧ w < b
def MotherOpinion (w : ℝ) := w ≤ 67

-- The average of probable weights
def AverageWeight (weights : Set ℝ) (avg : ℝ) := (∀ w ∈ weights, 64 < w ∧ w ≤ 67) ∧ avg = 66

-- The main theorem to be proven
theorem BrotherUpperLimit (hA : ArunOpinion w) (hB : BrotherOpinion w b) (hM : MotherOpinion w) (hAvg : AverageWeight {w | 64 < w ∧ w ≤ 67} 66) : b = 67 := by
  sorry

end BrotherUpperLimit_l245_245759


namespace find_other_solution_l245_245475

theorem find_other_solution (x : ℚ) :
  (72 * x ^ 2 + 43 = 113 * x - 12) → (x = 3 / 8) → (x = 43 / 36 ∨ x = 3 / 8) :=
by
  sorry

end find_other_solution_l245_245475


namespace joan_total_seashells_l245_245086

def seashells_given_to_Sam : ℕ := 43
def seashells_left_with_Joan : ℕ := 27
def total_seashells_found := seashells_given_to_Sam + seashells_left_with_Joan

theorem joan_total_seashells : total_seashells_found = 70 := by
  -- proof goes here, but for now we will use sorry
  sorry

end joan_total_seashells_l245_245086


namespace arithmetic_square_root_of_sqrt_81_l245_245653

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l245_245653


namespace coefficient_of_term_x7_in_expansion_l245_245763

theorem coefficient_of_term_x7_in_expansion:
  let general_term (r : ℕ) := (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r * (x : ℤ)^(12 - (5 * r) / 2)
  ∃ r : ℕ, 12 - (5 * r) / 2 = 7 ∧ (Nat.choose 6 r) * (2 : ℤ)^(6 - r) * (-1 : ℤ)^r = 240 := 
sorry

end coefficient_of_term_x7_in_expansion_l245_245763


namespace compound_interest_calculation_l245_245537

theorem compound_interest_calculation :
  let SI := (1833.33 * 16 * 6) / 100
  let CI := 2 * SI
  let principal_ci := 8000
  let rate_ci := 20
  let n := Real.log (1.4399995) / Real.log (1 + rate_ci / 100)
  n = 2 := by
  sorry

end compound_interest_calculation_l245_245537


namespace max_ab_bc_cd_da_l245_245363

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
sorry

end max_ab_bc_cd_da_l245_245363


namespace range_of_a_l245_245746

noncomputable def g (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x + 3

theorem range_of_a 
  (h_mono_inc : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → g a x1 ≤ g a x2)
  (h_nonneg : ∀ x : ℝ, -1 < x ∧ x < 1 → 0 ≤ g a x) :
  (-2 : ℝ) ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l245_245746


namespace distinct_nat_numbers_l245_245594

theorem distinct_nat_numbers 
  (a b c : ℕ) (p q r : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_sum : a + b + c = 55) 
  (h_ab : a + b = p * p) 
  (h_bc : b + c = q * q) 
  (h_ca : c + a = r * r) : 
  a = 19 ∧ b = 6 ∧ c = 30 :=
sorry

end distinct_nat_numbers_l245_245594


namespace relationship_between_number_and_square_l245_245013

theorem relationship_between_number_and_square (n : ℕ) (h : n = 9) :
  (n + n^2) / 2 = 5 * n := by
    sorry

end relationship_between_number_and_square_l245_245013


namespace question_l245_245471

variable (a : ℝ)

def condition_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2 * x ≤ a^2 - a - 3

def condition_q (a : ℝ) : Prop := ∀ (x y : ℝ) , x > y → (5 - 2 * a)^x < (5 - 2 * a)^y

theorem question (h1 : condition_p a ∨ condition_q a)
                (h2 : ¬ (condition_p a ∧ condition_q a)) : a = 2 ∨ a ≥ 5 / 2 :=
sorry

end question_l245_245471


namespace productivity_difference_l245_245848

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l245_245848


namespace probability_red_ball_l245_245312

noncomputable def event_A : Type := "a red ball is drawn"

def boxType := {H1 | H2 | H3: Prop}

def prob_H1 : ℝ := 2/5
def prob_H2 : ℝ := 2/5
def prob_H3 : ℝ := 1/5

def prob_A_given_H1 : ℝ := 4/10
def prob_A_given_H2 : ℝ := 2/10
def prob_A_given_H3 : ℝ := 8/10

theorem probability_red_ball :
  let P (A : event_A) :=
    prob_H1 * prob_A_given_H1 +
    prob_H2 * prob_A_given_H2 +
    prob_H3 * prob_A_given_H3 in
  P(A) = 0.4 :=
  by
  -- Proof omitted
  sorry

end probability_red_ball_l245_245312


namespace glass_volume_230_l245_245875

noncomputable def total_volume_of_glass : ℝ :=
  let V_P (V : ℝ) := 0.40 * V
  let V_O (V : ℝ) := 0.60 * V
  let condition : ∀ V, V_O V - V_P V = 46 := sorry
  classical.some (exists V, condition V)

theorem glass_volume_230 :
  total_volume_of_glass = 230 := sorry

end glass_volume_230_l245_245875


namespace negation_of_universal_l245_245532

theorem negation_of_universal (P : ∀ x : ℝ, x^2 > 0) : ¬ ( ∀ x : ℝ, x^2 > 0) ↔ ∃ x : ℝ, x^2 ≤ 0 :=
by 
  sorry

end negation_of_universal_l245_245532


namespace count_subsets_sum_eight_l245_245480

open Finset

theorem count_subsets_sum_eight :
  let M := (range 10).map (λ x => x + 1)
  in (M.powerset.filter (λ A => A.sum id = 8)).card = 6 := 
by
  sorry

end count_subsets_sum_eight_l245_245480


namespace exists_z0_l245_245207

open Complex

-- Define the given polynomial f(z)
noncomputable def polynomial (C : Fin n → ℂ) (n : ℕ) (z : ℂ) : ℂ :=
  (Finset.range n).sum (λ i, C i * z^(n - i))

-- Lean statement of the theorem
theorem exists_z0 (C : Fin (n + 1) → ℂ) (n : ℕ) :
  ∃ (z0 : ℂ), abs z0 ≤ 1 ∧
  abs (polynomial C n z0) ≥ abs (C 0) + abs (C n) :=
  sorry

end exists_z0_l245_245207


namespace sum_of_integers_l245_245256

theorem sum_of_integers (numbers : List ℕ) (h1 : numbers.Nodup) 
(h2 : ∃ a b, (a ≠ b ∧ a * b = 16 ∧ a ∈ numbers ∧ b ∈ numbers)) 
(h3 : ∃ c d, (c ≠ d ∧ c * d = 225 ∧ c ∈ numbers ∧ d ∈ numbers)) :
  numbers.sum = 44 :=
sorry

end sum_of_integers_l245_245256


namespace gcd_60_75_l245_245142

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l245_245142


namespace problem_correct_calculation_l245_245401

theorem problem_correct_calculation (a b : ℕ) : 
  (4 * a - 2 * a ≠ 2) ∧ 
  (a^8 / a^4 ≠ a^2) ∧ 
  (a^2 * a^3 = a^5) ∧ 
  ((b^2)^3 ≠ b^5) :=
by {
  sorry
}

end problem_correct_calculation_l245_245401


namespace smallest_positive_integer_n_mean_squares_l245_245925

theorem smallest_positive_integer_n_mean_squares :
  ∃ n : ℕ, n > 1 ∧ (∃ m : ℕ, (n * m ^ 2 = (n + 1) * (2 * n + 1) / 6) ∧ Nat.gcd (n + 1) (2 * n + 1) = 1 ∧ n = 337) :=
sorry

end smallest_positive_integer_n_mean_squares_l245_245925


namespace prove_f_2_eq_3_l245_245326

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then 3 * a ^ x else Real.log (2 * x + 4) / Real.log a

theorem prove_f_2_eq_3 (a : ℝ) (h1 : f 1 a = 6) : f 2 a = 3 :=
by
  -- Define the conditions
  have h1 : 3 * a = 6 := by simp [f] at h1; assumption
  -- Two subcases: x <= 1 and x > 1
  have : a = 2 := by linarith
  simp [f, this]
  sorry

end prove_f_2_eq_3_l245_245326


namespace total_cost_pencils_l245_245017

theorem total_cost_pencils
  (boxes : ℕ)
  (cost_per_box : ℕ → ℕ → ℕ)
  (price_regular : ℕ)
  (price_bulk : ℕ)
  (box_size : ℕ)
  (bulk_threshold : ℕ)
  (total_pencils : ℕ) :
  total_pencils = 3150 →
  box_size = 150 →
  price_regular = 40 →
  price_bulk = 35 →
  bulk_threshold = 2000 →
  boxes = (total_pencils + box_size - 1) / box_size →
  (total_pencils > bulk_threshold → cost_per_box boxes price_bulk = boxes * price_bulk) →
  (total_pencils ≤ bulk_threshold → cost_per_box boxes price_regular = boxes * price_regular) →
  total_pencils > bulk_threshold →
  cost_per_box boxes price_bulk = 735 :=
by
  intro h_total_pencils
  intro h_box_size
  intro h_price_regular
  intro h_price_bulk
  intro h_bulk_threshold
  intro h_boxes
  intro h_cost_bulk
  intro h_cost_regular
  intro h_bulk_discount_passt
  -- sorry statement as we don't provide the actual proof here
  sorry

end total_cost_pencils_l245_245017


namespace multiply_decimals_l245_245438

theorem multiply_decimals :
  0.25 * 0.08 = 0.02 :=
sorry

end multiply_decimals_l245_245438


namespace solve_quadratic_l245_245514

theorem solve_quadratic : 
  ∃ x1 x2 : ℝ, (x1 = -2 + Real.sqrt 2) ∧ (x2 = -2 - Real.sqrt 2) ∧ (∀ x : ℝ, x^2 + 4 * x + 2 = 0 → (x = x1 ∨ x = x2)) :=
by {
  sorry
}

end solve_quadratic_l245_245514


namespace triangle_area_squared_l245_245430

theorem triangle_area_squared
  (R : ℝ)
  (A : ℝ)
  (AC_minus_AB : ℝ)
  (area : ℝ)
  (hx : R = 4)
  (hy : A = 60)
  (hz : AC_minus_AB = 4)
  (area_eq : area = 8 * Real.sqrt 3) :
  area^2 = 192 :=
by
  -- We include the conditions 
  have hR := hx
  have hA := hy
  have hAC_AB := hz
  have harea := area_eq
  -- We will use these to construct the required proof 
  sorry

end triangle_area_squared_l245_245430


namespace closest_integers_to_2013_satisfy_trig_eq_l245_245588

noncomputable def closestIntegersSatisfyingTrigEq (x : ℝ) : Prop := 
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2)

theorem closest_integers_to_2013_satisfy_trig_eq : closestIntegersSatisfyingTrigEq (1935 * (Real.pi / 180)) ∧ closestIntegersSatisfyingTrigEq (2025 * (Real.pi / 180)) :=
sorry

end closest_integers_to_2013_satisfy_trig_eq_l245_245588


namespace glass_volume_230_l245_245872

variable (V : ℝ) -- Define the total volume of the glass

-- Define the conditions
def pessimist_glass_volume (V : ℝ) := 0.40 * V
def optimist_glass_volume (V : ℝ) := 0.60 * V
def volume_difference (V : ℝ) := optimist_glass_volume V - pessimist_glass_volume V

theorem glass_volume_230 
  (h1 : volume_difference V = 46) : V = 230 := 
sorry

end glass_volume_230_l245_245872


namespace greatest_value_of_squares_l245_245776

theorem greatest_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 18)
  (h2 : ab + c + d = 85)
  (h3 : ad + bc = 170)
  (h4 : cd = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 308 :=
sorry

end greatest_value_of_squares_l245_245776


namespace initial_ratio_l245_245862

variables {p q : ℝ}

theorem initial_ratio (h₁ : p + q = 20) (h₂ : p / (q + 1) = 4 / 3) : p / q = 3 / 2 :=
sorry

end initial_ratio_l245_245862


namespace find_value_l245_245308

theorem find_value : (1 / 4 * (5 * 9 * 4) - 7) = 38 := 
by
  sorry

end find_value_l245_245308


namespace eval_expression_l245_245575

theorem eval_expression : 
  (20-19 + 18-17 + 16-15 + 14-13 + 12-11 + 10-9 + 8-7 + 6-5 + 4-3 + 2-1) / 
  (1-2 + 3-4 + 5-6 + 7-8 + 9-10 + 11-12 + 13-14 + 15-16 + 17-18 + 19-20) = -1 := by
  sorry

end eval_expression_l245_245575


namespace teachers_per_grade_correct_l245_245375

def fifth_graders : ℕ := 109
def sixth_graders : ℕ := 115
def seventh_graders : ℕ := 118
def parents_per_grade : ℕ := 2
def number_of_grades : ℕ := 3
def buses : ℕ := 5
def seats_per_bus : ℕ := 72

-- Total number of students
def total_students : ℕ := fifth_graders + sixth_graders + seventh_graders

-- Total number of parents
def total_parents : ℕ := parents_per_grade * number_of_grades

-- Total number of seats available on the buses
def total_seats : ℕ := buses * seats_per_bus

-- Seats left for teachers
def seats_for_teachers : ℕ := total_seats - total_students - total_parents

-- Teachers per grade
def teachers_per_grade : ℕ := seats_for_teachers / number_of_grades

theorem teachers_per_grade_correct : teachers_per_grade = 4 := sorry

end teachers_per_grade_correct_l245_245375


namespace seven_lines_regions_l245_245795

theorem seven_lines_regions (n : ℕ) (hn : n = 7) (h1 : ¬ ∃ l1 l2 : ℝ, l1 = l2) (h2 : ∀ l1 l2 l3 : ℝ, ¬ (l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧ (l1 = l2 ∧ l2 = l3))) :
  ∃ R : ℕ, R = 29 :=
by
  sorry

end seven_lines_regions_l245_245795


namespace middle_segment_proportion_l245_245058

theorem middle_segment_proportion (a b c : ℝ) (h_a : a = 1) (h_b : b = 3) :
  (a / c = c / b) → c = Real.sqrt 3 :=
by
  sorry

end middle_segment_proportion_l245_245058


namespace find_f_l245_245747

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x / (a * x + b)

theorem find_f (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f 2 a b = 1) (h₂ : ∃! x, f x a b = x) :
  f x (1/2) 1 = 2 * x / (x + 2) :=
by
  sorry

end find_f_l245_245747


namespace students_per_group_l245_245420

-- Defining the conditions
def total_students : ℕ := 256
def number_of_teachers : ℕ := 8

-- The statement to prove
theorem students_per_group :
  total_students / number_of_teachers = 32 :=
by
  sorry

end students_per_group_l245_245420


namespace multiplication_mistake_l245_245424

theorem multiplication_mistake (x : ℕ) (H : 43 * x - 34 * x = 1215) : x = 135 :=
sorry

end multiplication_mistake_l245_245424


namespace work_days_together_l245_245157

theorem work_days_together (A_days B_days : ℕ) (work_left_fraction : ℚ) 
  (hA : A_days = 15) (hB : B_days = 20) (h_fraction : work_left_fraction = 8 / 15) : 
  ∃ d : ℕ, d * (1 / 15 + 1 / 20) = 1 - 8 / 15 ∧ d = 4 :=
by
  sorry

end work_days_together_l245_245157


namespace two_digit_multiples_of_6_and_9_l245_245940

theorem two_digit_multiples_of_6_and_9 : ∃ n : ℕ, n = 5 ∧ (∀ k : ℤ, 10 ≤ k ∧ k < 100 ∧ (k % 6 = 0) ∧ (k % 9 = 0) → 
    k = 18 ∨ k = 36 ∨ k = 54 ∨ k = 72 ∨ k = 90) := 
sorry

end two_digit_multiples_of_6_and_9_l245_245940


namespace find_constants_l245_245063

def f (x : ℝ) (a : ℝ) : ℝ := 2 * x ^ 3 + a * x
def g (x : ℝ) (b c : ℝ) : ℝ := b * x ^ 2 + c
def f' (x : ℝ) (a : ℝ) : ℝ := 6 * x ^ 2 + a
def g' (x : ℝ) (b : ℝ) : ℝ := 2 * b * x

theorem find_constants (a b c : ℝ) :
  f 2 a = 0 ∧ g 2 b c = 0 ∧ f' 2 a = g' 2 b →
  a = -8 ∧ b = 4 ∧ c = -16 :=
by
  intro h
  sorry

end find_constants_l245_245063


namespace fraction_sum_eq_five_fourths_l245_245051

theorem fraction_sum_eq_five_fourths (a b c : ℚ) (h : a / 2 = b / 3 ∧ b / 3 = c / 4) :
  (a + b) / c = 5 / 4 :=
by
  sorry

end fraction_sum_eq_five_fourths_l245_245051


namespace min_policemen_needed_l245_245696

-- Definitions of the problem parameters
def city_layout (n m : ℕ) := n > 0 ∧ m > 0

-- Function to calculate the minimum number of policemen
def min_policemen (n m : ℕ) : ℕ := (m - 1) * (n - 1)

-- The theorem to prove
theorem min_policemen_needed (n m : ℕ) (h : city_layout n m) : min_policemen n m = (m - 1) * (n - 1) :=
by
  unfold city_layout at h
  unfold min_policemen
  sorry

end min_policemen_needed_l245_245696


namespace prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l245_245227

section card_draws

/-- A card draw experiment with 10 cards: 5 red, 3 white, 2 blue. --/
inductive CardColor
| red
| white
| blue

def bag : List CardColor := List.replicate 5 CardColor.red ++ List.replicate 3 CardColor.white ++ List.replicate 2 CardColor.blue

/-- Probability of drawing exactly 2 red cards in up to 3 draws with the given conditions. --/
def prob_two_reds : ℚ :=
  (5 / 10) * (5 / 10) + 
  (5 / 10) * (2 / 10) * (5 / 10) + 
  (2 / 10) * (5 / 10) * (5 / 10)

theorem prob_of_two_reds_is_7_over_20 : prob_two_reds = 7 / 20 :=
  sorry

/-- Probability distribution of the number of draws necessary. --/
def prob_ξ_1 : ℚ := 3 / 10
def prob_ξ_2 : ℚ := 21 / 100
def prob_ξ_3 : ℚ := 49 / 100
def expected_value_ξ : ℚ :=
  1 * prob_ξ_1 + 2 * prob_ξ_2 + 3 * prob_ξ_3

theorem expected_value_is_2_19 : expected_value_ξ = 219 / 100 :=
  sorry

end card_draws

end prob_of_two_reds_is_7_over_20_expected_value_is_2_19_l245_245227


namespace roots_quadratic_l245_245186

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end roots_quadratic_l245_245186


namespace expressions_equal_iff_conditions_l245_245182

theorem expressions_equal_iff_conditions (a b c : ℝ) :
  (2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c)) ↔ (a = 0 ∨ a + 2 * b + 1.5 * c = 0) :=
by
  sorry

end expressions_equal_iff_conditions_l245_245182


namespace solve_eq54_l245_245193

noncomputable def eq54 (x : ℝ) := (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 54

theorem solve_eq54 : (eq54 0) ∨ (eq54 (-1)) ∨ (eq54 (-3)) ∨ (eq54 (-3.5)) := by
  sorry

end solve_eq54_l245_245193


namespace min_acute_triangles_for_isosceles_l245_245169

noncomputable def isosceles_triangle_acute_division : ℕ :=
  sorry

theorem min_acute_triangles_for_isosceles {α : ℝ} (hα : α = 108) (isosceles : ∀ β γ : ℝ, β = γ) :
  isosceles_triangle_acute_division = 7 :=
sorry

end min_acute_triangles_for_isosceles_l245_245169


namespace sum_of_ages_l245_245967

variable (J L : ℝ)
variable (h1 : J = L + 8)
variable (h2 : J + 10 = 5 * (L - 5))

theorem sum_of_ages (J L : ℝ) (h1 : J = L + 8) (h2 : J + 10 = 5 * (L - 5)) : J + L = 29.5 := by
  sorry

end sum_of_ages_l245_245967


namespace minnie_penny_time_difference_l245_245367

noncomputable def minnie_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def minnie_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_flat (distance speed: ℝ) := distance / speed
noncomputable def penny_time_downhill (distance speed: ℝ) := distance / speed
noncomputable def penny_time_uphill (distance speed: ℝ) := distance / speed
noncomputable def break_time (minutes: ℝ) := minutes / 60

noncomputable def minnie_total_time :=
  minnie_time_uphill 12 6 + minnie_time_downhill 18 25 + minnie_time_flat 25 18

noncomputable def penny_total_time :=
  penny_time_flat 25 25 + penny_time_downhill 12 35 + 
  penny_time_uphill 18 12 + break_time 10

noncomputable def time_difference := (minnie_total_time - penny_total_time) * 60

theorem minnie_penny_time_difference :
  time_difference = 66 := by
  sorry

end minnie_penny_time_difference_l245_245367


namespace wraps_add_more_l245_245234

/-- Let John's raw squat be 600 pounds. Let sleeves add 30 pounds to his lift. Let wraps add 25% 
to his squat. We aim to prove that wraps add 120 pounds more to John's squat than sleeves. -/
theorem wraps_add_more (raw_squat : ℝ) (sleeves_bonus : ℝ) (wraps_percentage : ℝ) : 
  raw_squat = 600 → sleeves_bonus = 30 → wraps_percentage = 0.25 → 
  (raw_squat * wraps_percentage) - sleeves_bonus = 120 :=
by
  intros h1 h2 h3
  sorry

end wraps_add_more_l245_245234


namespace find_a_l245_245609

-- Defining the conditions as hypotheses
variables (a b d : ℕ)
hypothesis h1 : a + b = d
hypothesis h2 : b + d = 7
hypothesis h3 : d = 4

theorem find_a : a = 1 :=
by
  sorry

end find_a_l245_245609


namespace reciprocal_of_sum_of_repeating_decimals_l245_245545

theorem reciprocal_of_sum_of_repeating_decimals :
  let x := 5 / 33
  let y := 1 / 3
  1 / (x + y) = 33 / 16 :=
by
  -- The following is the proof, but it will be skipped for this exercise.
  sorry

end reciprocal_of_sum_of_repeating_decimals_l245_245545


namespace percentage_increase_14point4_from_12_l245_245076

theorem percentage_increase_14point4_from_12 (x : ℝ) (h : x = 14.4) : 
  ((x - 12) / 12) * 100 = 20 := 
by
  sorry

end percentage_increase_14point4_from_12_l245_245076


namespace arithmetic_square_root_sqrt_81_l245_245654

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l245_245654


namespace optimal_p_closest_to_1000p_l245_245170

theorem optimal_p_closest_to_1000p (p : ℝ) (h : ℕ) (nonneg_p : 0 ≤ p) (le_p : p ≤ 1)
  (ht : 1 < h)
  (prob_recurrence : ∀ h, P h = (1 - p) * P (h - 2) + p * P (h - 3))
  (win_cond : P 0 = 1)
  (lose_cond : P 1 = 0) :
  closestTo (1000 * p) = 618 :=
by
  sorry

end optimal_p_closest_to_1000p_l245_245170


namespace find_a_l245_245610

theorem find_a (a b d : ℤ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l245_245610


namespace prob_green_ball_l245_245030

-- Definitions for the conditions
def red_balls_X := 3
def green_balls_X := 7
def total_balls_X := red_balls_X + green_balls_X

def red_balls_YZ := 7
def green_balls_YZ := 3
def total_balls_YZ := red_balls_YZ + green_balls_YZ

-- The probability of selecting any container
def prob_select_container := 1 / 3

-- The probabilities of drawing a green ball from each container
def prob_green_given_X := green_balls_X / total_balls_X
def prob_green_given_YZ := green_balls_YZ / total_balls_YZ

-- The combined probability of selecting a green ball
theorem prob_green_ball : 
  prob_select_container * prob_green_given_X + 
  prob_select_container * prob_green_given_YZ + 
  prob_select_container * prob_green_given_YZ = 13 / 30 := 
  by sorry

end prob_green_ball_l245_245030


namespace committee_probability_l245_245987

/--
Suppose there are 24 members in a club: 12 boys and 12 girls.
A 5-person committee is chosen at random.
Prove that the probability of having at least 2 boys and at least 2 girls in the committee is 121/177.
-/
theorem committee_probability :
  let boys := 12
  let girls := 12
  let total_members := 24
  let committee_size := 5
  let all_ways := Nat.choose total_members committee_size
  let invalid_ways := 2 * Nat.choose boys committee_size + 2 * (Nat.choose boys 1 * Nat.choose girls 4)
  let valid_ways := all_ways - invalid_ways
  let probability := valid_ways / all_ways
  probability = 121 / 177 :=
by
  sorry

end committee_probability_l245_245987


namespace initially_calculated_average_l245_245989

open List

theorem initially_calculated_average (numbers : List ℝ) (h_len : numbers.length = 10) 
  (h_wrong_reading : ∃ (n : ℝ), n ∈ numbers ∧ n ≠ 26 ∧ (numbers.erase n).sum + 26 = numbers.sum - 36 + 26) 
  (h_correct_avg : numbers.sum / 10 = 16) : 
  ((numbers.sum - 10) / 10 = 15) := 
sorry

end initially_calculated_average_l245_245989


namespace max_triangles_9261_l245_245596

-- Define the problem formally
noncomputable def max_triangles (points : ℕ) (circ_radius : ℝ) (min_side_length : ℝ) : ℕ :=
  -- Function definition for calculating the maximum number of triangles
  sorry

-- State the conditions and the expected maximum number of triangles
theorem max_triangles_9261 :
  max_triangles 63 10 9 = 9261 :=
sorry

end max_triangles_9261_l245_245596


namespace probability_30_to_50_l245_245822

noncomputable def xi_distribution : ProbabilityDistribution ℝ := 
  NormalDist.mk 40 σ

axiom P_xi_less_than_30 (σ : ℝ) : ∫ x in Iic 30, xi_distribution.to_density (Pdf) x = 0.2

theorem probability_30_to_50 (σ : ℝ) : ∫ x in Ioc 30 50, xi_distribution.to_density (Pdf) x = 0.6 :=
by sorry

end probability_30_to_50_l245_245822


namespace parallel_lines_l245_245214

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end parallel_lines_l245_245214


namespace express_f12_in_terms_of_a_l245_245340

variable {f : ℝ → ℝ}
variable {a : ℝ}
variable (f_add : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (f_neg_three : f (-3) = a)

theorem express_f12_in_terms_of_a : f 12 = -4 * a := sorry

end express_f12_in_terms_of_a_l245_245340


namespace height_ratio_l245_245436

noncomputable def Anne_height := 80
noncomputable def Bella_height := 3 * Anne_height
noncomputable def Sister_height := Bella_height - 200

theorem height_ratio : Anne_height / Sister_height = 2 :=
by
  /-
  The proof here is omitted as requested.
  -/
  sorry

end height_ratio_l245_245436


namespace largest_prime_factor_of_4620_l245_245833

open Nat

theorem largest_prime_factor_of_4620 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4620 ∧ (∀ q : ℕ, (Nat.Prime q ∧ q ∣ 4620) → q ≤ p) :=
begin
  use 11,
  split,
  { apply Nat.prime_of_nat_prime, exact prime_11, },
  split,
  { apply divides_prime_factors, norm_num, },
  { intros q hq,
    apply le_trans (prime_le_magnitude hq.1),
    suffices : q ∣ 4620 ∧ q ∈ { 2, 5, 3, 7, 11 }, from this.elim (λ H h, H.symm ▸ Nat.le_of_eq (set.eq_of_mem_singleton h)),
    exact ⟨hq.2,Hq⟩,
    { apply and.intro, exact Nat.prime_divisors.mem_list.mp _, exact list.mem_cons_of_mem _, exact hq } 
  },
end

end largest_prime_factor_of_4620_l245_245833


namespace teacher_age_is_56_l245_245838

theorem teacher_age_is_56 (s t : ℝ) (h1 : s = 40 * 15) (h2 : s + t = 41 * 16) : t = 56 := by
  sorry

end teacher_age_is_56_l245_245838


namespace parallel_lines_suff_cond_not_necess_l245_245059

theorem parallel_lines_suff_cond_not_necess (a : ℝ) :
  a = -2 → 
  (∀ x y : ℝ, (2 * x + y - 3 = 0) ∧ (2 * x + y + 4 = 0) → 
    (∃ a : ℝ, a = -2 ∨ a = 1)) ∧
    (a = -2 → ∃ a : ℝ, a = -2 ∨ a = 1) :=
by {
  sorry
}

end parallel_lines_suff_cond_not_necess_l245_245059


namespace age_difference_between_two_children_l245_245417

theorem age_difference_between_two_children 
  (avg_age_10_years_ago : ℕ)
  (present_avg_age : ℕ)
  (youngest_child_present_age : ℕ)
  (initial_family_members : ℕ)
  (current_family_members : ℕ)
  (H1 : avg_age_10_years_ago = 24)
  (H2 : present_avg_age = 24)
  (H3 : youngest_child_present_age = 3)
  (H4 : initial_family_members = 4)
  (H5 : current_family_members = 6) :
  ∃ (D: ℕ), D = 2 :=
by
  sorry

end age_difference_between_two_children_l245_245417


namespace simplify_expression_l245_245796

theorem simplify_expression :
  (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4 * 6 * a^5) = 720 * a^15 :=
by
  sorry

end simplify_expression_l245_245796


namespace constant_term_value_l245_245210

theorem constant_term_value :
  ∀ (x y z k : ℤ), (4 * x + y + z = 80) → (2 * x - y - z = 40) → (x = 20) → (3 * x + y - z = k) → (k = 60) :=
by 
  intros x y z k h₁ h₂ hx h₃
  sorry

end constant_term_value_l245_245210


namespace probability_of_six_event_l245_245839

-- The faces of the die are represented as natural numbers from 1 to 6.
-- The outcomes of the three rolls are represented by a tuple (a1, a2, a3).

noncomputable def probability_event : ℝ :=
  let outcomes := {(a1, a2, a3) | a1, a2, a3 ∈ {1, 2, 3, 4, 5, 6}} in
  let favorable_event := {(a1, a2, a3) ∈ outcomes | 
                          abs (a1 - a2) + abs (a2 - a3) + abs (a3 - a1) = 6} in
  (favorable_event.to_finset.card : ℝ) / (outcomes.to_finset.card : ℝ)

-- The desired probability is proved as follows:
theorem probability_of_six_event : probability_event = 1/4 := 
by sorry

end probability_of_six_event_l245_245839


namespace triangle_inequality_l245_245356

theorem triangle_inequality 
  (A B C : ℝ) -- angle measures
  (a b c : ℝ) -- side lengths
  (h1 : a = b * (Real.cos C) + c * (Real.cos B)) 
  (cos_half_C_pos : 0 < Real.cos (C/2)) 
  (cos_half_C_lt_one : Real.cos (C/2) < 1)
  (cos_half_B_pos : 0 < Real.cos (B/2)) 
  (cos_half_B_lt_one : Real.cos (B/2) < 1) :
  2 * b * Real.cos (C / 2) + 2 * c * Real.cos (B / 2) > a + b + c :=
by
  sorry

end triangle_inequality_l245_245356


namespace most_likely_units_digit_sum_is_zero_l245_245804

theorem most_likely_units_digit_sum_is_zero :
  ∃ (units_digit : ℕ), 
  (∀ m n : ℕ, (1 ≤ m ∧ m ≤ 9) ∧ (1 ≤ n ∧ n ≤ 9) → 
    units_digit = (m + n) % 10) ∧ 
  units_digit = 0 :=
sorry

end most_likely_units_digit_sum_is_zero_l245_245804


namespace tom_dance_lessons_cost_l245_245124

theorem tom_dance_lessons_cost (total_lessons free_lessons : ℕ) (cost_per_lesson : ℕ) (h1 : total_lessons = 10) (h2 : free_lessons = 2) (h3 : cost_per_lesson = 10) : total_lessons * cost_per_lesson - free_lessons * cost_per_lesson = 80 :=
by
  rw [h1, h2, h3]
  sorry

end tom_dance_lessons_cost_l245_245124


namespace compound_interest_interest_l245_245243

theorem compound_interest_interest :
  let P := 2000
  let r := 0.05
  let n := 5
  let A := P * (1 + r)^n
  let interest := A - P
  interest = 552.56 := by
  sorry

end compound_interest_interest_l245_245243


namespace class_grades_l245_245907

theorem class_grades (boys girls n : ℕ) (h1 : girls = boys + 3) (h2 : ∀ (fours fives : ℕ), fours = fives + 6) (h3 : ∀ (threes : ℕ), threes = 2 * (fives + 6)) : ∃ k, k = 2 ∨ k = 1 :=
by
  sorry

end class_grades_l245_245907


namespace school_band_fundraising_l245_245826

-- Definitions
def goal : Nat := 150
def earned_from_three_families : Nat := 10 * 3
def earned_from_fifteen_families : Nat := 5 * 15
def total_earned : Nat := earned_from_three_families + earned_from_fifteen_families
def needed_more : Nat := goal - total_earned

-- Theorem stating the problem in Lean 4
theorem school_band_fundraising : needed_more = 45 := by
  sorry

end school_band_fundraising_l245_245826


namespace find_original_number_l245_245705

-- Given definitions and conditions
def doubled_add_nine (x : ℝ) : ℝ := 2 * x + 9
def trebled (y : ℝ) : ℝ := 3 * y

-- The proof problem we need to solve
theorem find_original_number (x : ℝ) (h : trebled (doubled_add_nine x) = 69) : x = 7 := 
by sorry

end find_original_number_l245_245705


namespace circle_equation_with_focus_center_and_tangent_directrix_l245_245384

theorem circle_equation_with_focus_center_and_tangent_directrix :
  ∃ (x y : ℝ), (∃ k : ℝ, y^2 = -8 * x ∧ k = 2 ∧ (x = -2 ∧ y = 0) ∧ (x + 2)^2 + y^2 = 16) :=
by
  sorry

end circle_equation_with_focus_center_and_tangent_directrix_l245_245384


namespace area_of_figure_l245_245250

def equation (x y : ℝ) : Prop := |15 * x| + |8 * y| + |120 - 15 * x - 8 * y| = 120

theorem area_of_figure : ∃ (A : ℝ), A = 60 ∧ 
  (∃ (x y : ℝ), equation x y) :=
sorry

end area_of_figure_l245_245250


namespace nature_of_singularity_at_1_l245_245449

open Complex

def f (z : ℂ) : ℂ := (z - 1) * exp (1 / (z - 1))

theorem nature_of_singularity_at_1 : 
  ∃ g : ℂ → ℂ, is_essential_singularity (g) 1 :=
begin
  use f,
  sorry
end

end nature_of_singularity_at_1_l245_245449


namespace problem1_problem2_problem3_problem4_l245_245574

-- Proof statement for problem 1
theorem problem1 : (1 : ℤ) * (-8) + 10 + 2 + (-1) = 3 := sorry

-- Proof statement for problem 2
theorem problem2 : (-21.6 : ℝ) - (-3) - |(-7.4)| + (-2 / 5) = -26.4 := sorry

-- Proof statement for problem 3
theorem problem3 : (-12 / 5) / (-1 / 10) * (-5 / 6) * (-0.4 : ℝ) = 8 := sorry

-- Proof statement for problem 4
theorem problem4 : ((5 / 8) - (1 / 6) + (7 / 12)) * (-24 : ℝ) = -25 := sorry

end problem1_problem2_problem3_problem4_l245_245574


namespace minimum_value_of_f_div_f_l245_245064

noncomputable def quadratic_function_min_value (a b c : ℝ) (h : 0 < b) (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) : ℝ :=
  (a + b + c) / b

theorem minimum_value_of_f_div_f' (a b c : ℝ) (h : 0 < b)
  (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) :
  quadratic_function_min_value a b c h h₀ h₁ h₂ = 2 :=
sorry

end minimum_value_of_f_div_f_l245_245064


namespace jackson_volume_discount_l245_245768

-- Given conditions as parameters
def hotTubVolume := 40 -- gallons
def quartsPerGallon := 4 -- quarts per gallon
def bottleVolume := 1 -- quart per bottle
def bottleCost := 50 -- dollars per bottle
def totalSpent := 6400 -- dollars spent by Jackson

-- Calculation related definitions
def totalQuarts := hotTubVolume * quartsPerGallon
def totalBottles := totalQuarts / bottleVolume
def costWithoutDiscount := totalBottles * bottleCost
def discountAmount := costWithoutDiscount - totalSpent
def discountPercentage := (discountAmount / costWithoutDiscount) * 100

-- The proof problem
theorem jackson_volume_discount : discountPercentage = 20 :=
by
  sorry

end jackson_volume_discount_l245_245768


namespace determine_ts_l245_245726

theorem determine_ts :
  ∃ t s : ℝ, 
  (⟨3, 1⟩ : ℝ × ℝ) + t • (⟨4, -6⟩) = (⟨0, 2⟩ : ℝ × ℝ) + s • (⟨-3, 5⟩) :=
by
  use 6, -9
  sorry

end determine_ts_l245_245726


namespace Emma_age_ratio_l245_245585

theorem Emma_age_ratio (E M : ℕ) (h1 : E = E) (h2 : E = E) 
(h3 : E - M = 3 * (E - 4 * M)) : E / M = 11 / 2 :=
sorry

end Emma_age_ratio_l245_245585


namespace west_movement_80_eq_neg_80_l245_245027

-- Define conditions
def east_movement (distance : ℤ) : ℤ := distance

-- Prove that moving westward is represented correctly
theorem west_movement_80_eq_neg_80 : east_movement (-80) = -80 :=
by
  -- Theorem proof goes here
  sorry

end west_movement_80_eq_neg_80_l245_245027


namespace solve_for_x_l245_245149

theorem solve_for_x (x : ℝ) (h : (8 - x)^2 = x^2) : x = 4 := 
by 
  sorry

end solve_for_x_l245_245149


namespace installation_rates_l245_245114

variables (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ)
variables (rate_teamA : ℕ) (rate_teamB : ℕ)

-- Conditions
def conditions : Prop :=
  units_total = 140 ∧
  teamA_units = 80 ∧
  teamB_units = units_total - teamA_units ∧
  team_units_gap = 5 ∧
  rate_teamA = rate_teamB + team_units_gap

-- Question to prove
def solution : Prop :=
  rate_teamB = 15 ∧ rate_teamA = 20

-- Statement of the proof
theorem installation_rates (units_total : ℕ) (teamA_units : ℕ) (teamB_units : ℕ) (team_units_gap : ℕ) (rate_teamA : ℕ) (rate_teamB : ℕ) :
  conditions units_total teamA_units teamB_units team_units_gap rate_teamA rate_teamB →
  solution rate_teamA rate_teamB :=
sorry

end installation_rates_l245_245114


namespace remaining_amount_needed_l245_245824

def goal := 150
def earnings_from_3_families := 3 * 10
def earnings_from_15_families := 15 * 5
def total_earnings := earnings_from_3_families + earnings_from_15_families
def remaining_amount := goal - total_earnings

theorem remaining_amount_needed : remaining_amount = 45 := by
  sorry

end remaining_amount_needed_l245_245824


namespace UF_opponent_score_l245_245128

theorem UF_opponent_score 
  (total_points : ℕ)
  (games_played : ℕ)
  (previous_points_avg : ℕ)
  (championship_score : ℕ)
  (opponent_score : ℕ)
  (total_points_condition : total_points = 720)
  (games_played_condition : games_played = 24)
  (previous_points_avg_condition : previous_points_avg = total_points / games_played)
  (championship_score_condition : championship_score = previous_points_avg / 2 - 2)
  (loss_by_condition : opponent_score = championship_score - 2) :
  opponent_score = 11 :=
by
  sorry

end UF_opponent_score_l245_245128


namespace exists_divisible_diff_l245_245598

theorem exists_divisible_diff (l : List ℤ) (h_len : l.length = 2022) :
  ∃ i j, i ≠ j ∧ (l.nthLe i sorry - l.nthLe j sorry) % 2021 = 0 :=
by
  apply sorry -- Placeholder for proof

end exists_divisible_diff_l245_245598


namespace certain_number_is_14_l245_245406

theorem certain_number_is_14 
  (a b n : ℕ) 
  (h1 : ∃ k1, a = k1 * n) 
  (h2 : ∃ k2, b = k2 * n) 
  (h3 : b = a + 11 * n) 
  (h4 : b = a + 22 * 7) : n = 14 := 
by 
  sorry

end certain_number_is_14_l245_245406


namespace gcd_of_lcm_and_ratio_l245_245616

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end gcd_of_lcm_and_ratio_l245_245616


namespace arithmetic_square_root_sqrt_81_l245_245655

theorem arithmetic_square_root_sqrt_81 : sqrt (sqrt 81) = 3 := 
by 
  have h1 : sqrt 81 = 9 := sorry
  have h2 : sqrt 9 = 3 := sorry
  show sqrt (sqrt 81) = 3 from sorry

end arithmetic_square_root_sqrt_81_l245_245655


namespace petya_square_larger_than_vasya_square_l245_245787

variable (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)

def petya_square_side (a b : ℝ) : ℝ := a * b / (a + b)

def vasya_square_side (a b : ℝ) : ℝ := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petya_square_larger_than_vasya_square
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  petya_square_side a b > vasya_square_side a b :=
by sorry

end petya_square_larger_than_vasya_square_l245_245787


namespace glass_volume_l245_245879

theorem glass_volume (V : ℝ) :
  let V_P := 0.40 * V
  let V_O := 0.60 * V
  V_O - V_P = 46 → V = 230 := 
by {
  intro h,
  have h1 : V_O = 0.60 * V := rfl,
  have h2 : V_P = 0.40 * V := rfl,
  rw [h2, h1] at h,
  linarith,
  sorry
}

end glass_volume_l245_245879


namespace intersection_of_lines_l245_245544

theorem intersection_of_lines : ∃ x y : ℚ, y = 3 * x ∧ y - 5 = -7 * x ∧ x = 1 / 2 ∧ y = 3 / 2 :=
by
  sorry

end intersection_of_lines_l245_245544


namespace pyramid_cross_section_distance_l245_245390

theorem pyramid_cross_section_distance 
  (A1 A2 : ℝ) (d : ℝ) (h : ℝ) 
  (hA1 : A1 = 125 * Real.sqrt 3)
  (hA2 : A2 = 500 * Real.sqrt 3)
  (hd : d = 12) :
  h = 24 :=
by
  sorry

end pyramid_cross_section_distance_l245_245390


namespace largest_rectangle_area_l245_245811

theorem largest_rectangle_area (x y : ℝ) (h : 2*x + 2*y = 60) : x * y ≤ 225 :=
sorry

end largest_rectangle_area_l245_245811


namespace length_BA_correct_area_ABCDE_correct_l245_245355

variables {BE CD CE CA : ℝ}
axiom BE_eq : BE = 13
axiom CD_eq : CD = 3
axiom CE_eq : CE = 10
axiom CA_eq : CA = 10

noncomputable def length_BA : ℝ := 3
noncomputable def area_ABCDE : ℝ := 4098 / 61

theorem length_BA_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  length_BA = 3 := 
by { sorry }

theorem area_ABCDE_correct (h1 : BE = 13) (h2 : CD = 3) (h3 : CE = 10) (h4 : CA = 10) :
  area_ABCDE = 4098 / 61 := 
by { sorry }

end length_BA_correct_area_ABCDE_correct_l245_245355


namespace dima_floor_l245_245727

-- Definitions of the constants from the problem statement
def nine_story_building := 9
def elevator_descend_time := 60 -- seconds
def journey_upstairs_time := 70 -- seconds
def elevator_speed := (λ n : ℕ, (n - 1) / 60)
def dima_walk_speed := (λ n : ℕ, (n - 1) / 120)

-- Define the main problem statement
theorem dima_floor :
  ∃ n : ℕ, 
    n ≤ nine_story_building ∧
    (∃ m : ℕ, m < n ∧
    (journey_upstairs_time =
      ((m - 1) / elevator_speed n +
       (n - m) / (dima_walk_speed n))) ∧
    n = 7) :=
sorry

end dima_floor_l245_245727


namespace solution_pairs_l245_245923

theorem solution_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b)
    (h_coprime: Nat.gcd (2 * a - 1) (2 * b + 1) = 1) 
    (h_divides : (a + b) ∣ (4 * a * b + 1)) :
    ∃ n : ℕ, a = n ∧ b = n + 1 :=
by
  -- statement
  sorry

end solution_pairs_l245_245923


namespace domain_sqrt_frac_l245_245581

theorem domain_sqrt_frac (x : ℝ) :
  (x^2 + 4*x + 3 ≠ 0) ∧ (x + 3 ≥ 0) ↔ ((x ∈ Set.Ioc (-3) (-1)) ∨ (x ∈ Set.Ioi (-1))) :=
by
  sorry

end domain_sqrt_frac_l245_245581


namespace inequality_proof_l245_245103

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  2 * (a + b + c) + 9 / (a * b + b * c + c * a)^2 ≥ 7 :=
by
  sorry

end inequality_proof_l245_245103


namespace solution_set_l245_245074

noncomputable def f : ℝ → ℝ := sorry

-- The function f is defined to be odd.
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- The function f is increasing on (-∞, 0).
axiom increasing_f : ∀ x y : ℝ, x < y ∧ y < 0 → f x < f y

-- Given f(2) = 0
axiom f_at_2 : f 2 = 0

-- Prove the solution set for x f(x + 1) < 0
theorem solution_set : { x : ℝ | x * f (x + 1) < 0 } = {x : ℝ | (-3 < x ∧ x < -1) ∨ (0 < x ∧ x < 1)} :=
by
  sorry

end solution_set_l245_245074


namespace gcf_60_75_l245_245133

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l245_245133


namespace total_expenditure_of_Louis_l245_245782

def fabric_cost (yards price_per_yard : ℕ) : ℕ :=
  yards * price_per_yard

def thread_cost (spools price_per_spool : ℕ) : ℕ :=
  spools * price_per_spool

def total_cost (yards price_per_yard pattern_cost spools price_per_spool : ℕ) : ℕ :=
  fabric_cost yards price_per_yard + pattern_cost + thread_cost spools price_per_spool

theorem total_expenditure_of_Louis :
  total_cost 5 24 15 2 3 = 141 :=
by
  sorry

end total_expenditure_of_Louis_l245_245782


namespace positive_divisors_multiple_of_15_l245_245069

theorem positive_divisors_multiple_of_15 (a b c : ℕ) (n : ℕ) (divisor : ℕ) (h_factorization : n = 6480)
  (h_prime_factorization : n = 2^4 * 3^4 * 5^1)
  (h_divisor : divisor = 2^a * 3^b * 5^c)
  (h_a_range : 0 ≤ a ∧ a ≤ 4)
  (h_b_range : 1 ≤ b ∧ b ≤ 4)
  (h_c_range : 1 ≤ c ∧ c ≤ 1) : sorry :=
sorry

end positive_divisors_multiple_of_15_l245_245069


namespace compute_value_l245_245444

theorem compute_value : 302^2 - 298^2 = 2400 :=
by
  sorry

end compute_value_l245_245444


namespace find_y_eq_54_div_23_l245_245927

open BigOperators

theorem find_y_eq_54_div_23 (y : ℚ) (h : (Real.sqrt (8 * y) / Real.sqrt (6 * (y - 2))) = 3) : y = 54 / 23 := 
by
  sorry

end find_y_eq_54_div_23_l245_245927


namespace exists_acute_triangle_l245_245595

theorem exists_acute_triangle (a b c d e : ℝ) 
  (h1 : a ≤ b) (h2 : b ≤ c) (h3 : c ≤ d) (h4 : d ≤ e)
  (h_triangle_abc : a + b > c) (h_triangle_abd : a + b > d) (h_triangle_abe : a + b > e)
  (h_triangle_bcd : b + c > d) (h_triangle_bce : b + c > e) (h_triangle_cde : c + d > e)
  (h_triangle_abc2 : a + c > b) (h_triangle_abd2 : a + d > b) (h_triangle_abe2 : a + e > b)
  (h_triangle_bcd2 : b + d > c) (h_triangle_bce2 : b + e > c) (h_triangle_cde2 : c + e > d)
  (h_triangle_abc3 : b + c > a) (h_triangle_abd3 : b + d > a) (h_triangle_abe3 : b + e > a)
  (h_triangle_bcd3 : b + d > a) (h_triangle_bce3 : c + e > a) (h_triangle_cde3 : d + e > c) :
  ∃ x y z : ℝ, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧ 
              (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧ 
              (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
              (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z) ∧
              x + y > z ∧ 
              ¬ (x^2 + y^2 ≤ z^2) :=
by
  sorry

end exists_acute_triangle_l245_245595


namespace remaining_three_digit_numbers_l245_245941

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_invalid_number (n : ℕ) : Prop :=
  ∃ (A B : ℕ), A ≠ B ∧ B ≠ 0 ∧ n = 100 * A + 10 * B + A

def count_valid_three_digit_numbers : ℕ :=
  let total_numbers := 900
  let invalid_numbers := 10 * 9
  total_numbers - invalid_numbers

theorem remaining_three_digit_numbers : count_valid_three_digit_numbers = 810 := by
  sorry

end remaining_three_digit_numbers_l245_245941


namespace original_savings_l245_245409

-- Given conditions:
def total_savings (s : ℝ) : Prop :=
  1 / 4 * s = 230

-- Theorem statement: 
theorem original_savings (s : ℝ) (h : total_savings s) : s = 920 :=
sorry

end original_savings_l245_245409


namespace train_speed_l245_245407

theorem train_speed (length_train length_bridge time_crossing speed : ℝ)
  (h1 : length_train = 100)
  (h2 : length_bridge = 300)
  (h3 : time_crossing = 24)
  (h4 : speed = (length_train + length_bridge) / time_crossing) :
  speed = 16.67 := 
sorry

end train_speed_l245_245407


namespace Pythagorean_triple_example_1_Pythagorean_triple_example_2_l245_245404

theorem Pythagorean_triple_example_1 : 3^2 + 4^2 = 5^2 := by
  sorry

theorem Pythagorean_triple_example_2 : 5^2 + 12^2 = 13^2 := by
  sorry

end Pythagorean_triple_example_1_Pythagorean_triple_example_2_l245_245404


namespace find_b_l245_245506

theorem find_b (b : ℝ) : (∃ x : ℝ, (x^3 - 3*x^2 = -3*x + b ∧ (3*x^2 - 6*x = -3))) → b = 1 :=
by
  intros h
  sorry

end find_b_l245_245506


namespace max_rectangle_area_l245_245813

theorem max_rectangle_area (x y : ℝ) (h : 2 * x + 2 * y = 60) : x * y ≤ 225 :=
sorry

end max_rectangle_area_l245_245813


namespace petyas_square_is_larger_l245_245789

noncomputable def side_petya_square (a b : ℝ) : ℝ :=
  a * b / (a + b)

noncomputable def side_vasya_square (a b : ℝ) : ℝ :=
  a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petyas_square_is_larger (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  side_petya_square a b > side_vasya_square a b := by
  sorry

end petyas_square_is_larger_l245_245789


namespace productivity_difference_l245_245849

/-- Two girls knit at constant, different speeds, where the first takes a 1-minute tea break every 
5 minutes and the second takes a 1-minute tea break every 7 minutes. Each tea break lasts exactly 
1 minute. When they went for a tea break together, they had knitted the same amount. Prove that 
the first girl's productivity is 5% higher if they started knitting at the same time. -/
theorem productivity_difference :
  let first_cycle := 5 + 1 in
  let second_cycle := 7 + 1 in
  let lcm_value := Nat.lcm first_cycle second_cycle in
  let first_working_time := 5 * (lcm_value / first_cycle) in
  let second_working_time := 7 * (lcm_value / second_cycle) in
  let rate1 := 5.0 / first_working_time * 100.0 in
  let rate2 := 7.0 / second_working_time * 100.0 in
  (rate1 - rate2) = 5 ->
  true := 
by 
  sorry

end productivity_difference_l245_245849


namespace prime_triples_l245_245732

open Nat

theorem prime_triples (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) :
    (p ∣ q^r + 1) → (q ∣ r^p + 1) → (r ∣ p^q + 1) → (p, q, r) = (2, 5, 3) ∨ (p, q, r) = (3, 2, 5) ∨ (p, q, r) = (5, 3, 2) :=
  by
  sorry

end prime_triples_l245_245732


namespace mass_percentage_O_is_correct_l245_245456

noncomputable def molar_mass_Al : ℝ := 26.98
noncomputable def molar_mass_O : ℝ := 16.00
noncomputable def num_Al_atoms : ℕ := 2
noncomputable def num_O_atoms : ℕ := 3

noncomputable def molar_mass_Al2O3 : ℝ :=
  (num_Al_atoms * molar_mass_Al) + (num_O_atoms * molar_mass_O)

noncomputable def mass_percentage_O_in_Al2O3 : ℝ :=
  ((num_O_atoms * molar_mass_O) / molar_mass_Al2O3) * 100

theorem mass_percentage_O_is_correct :
  mass_percentage_O_in_Al2O3 = 47.07 :=
by
  sorry

end mass_percentage_O_is_correct_l245_245456


namespace fraction_of_7000_l245_245548

theorem fraction_of_7000 (x : ℝ) 
  (h1 : (1 / 10 / 100) * 7000 = 7) 
  (h2 : x * 7000 - 7 = 700) : 
  x = 0.101 :=
by
  sorry

end fraction_of_7000_l245_245548


namespace minimum_crooks_l245_245494

theorem minimum_crooks (total_ministers : ℕ) (condition : ∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest) : ∃ (minimum_crooks : ℕ), minimum_crooks = 91 :=
by
  let total_ministers := 100
  let set_of_dishonest : Finset ℕ := sorry -- arbitrary set for dishonest ministers satisfying the conditions
  have condition := (∀ (S : Finset ℕ), S.card = 10 → ∃ x ∈ S, x ∈ set_of_dishonest)
  exact ⟨91, sorry⟩

end minimum_crooks_l245_245494


namespace productivity_comparison_l245_245851

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l245_245851


namespace glass_volume_is_230_l245_245887

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l245_245887


namespace Gemma_ordered_pizzas_l245_245590

-- Definitions of conditions
def pizza_cost : ℕ := 10
def tip : ℕ := 5
def paid_amount : ℕ := 50
def change : ℕ := 5
def total_spent : ℕ := paid_amount - change

-- Statement of the proof problem
theorem Gemma_ordered_pizzas : 
  ∃ (P : ℕ), pizza_cost * P + tip = total_spent ∧ P = 4 :=
sorry

end Gemma_ordered_pizzas_l245_245590


namespace calculate_distribution_l245_245441

theorem calculate_distribution (a b : ℝ) : 3 * a * (5 * a - 2 * b) = 15 * a^2 - 6 * a * b :=
by
  sorry

end calculate_distribution_l245_245441


namespace average_speed_of_the_car_l245_245560

noncomputable def averageSpeed (d1 d2 d3 d4 t1 t2 t3 t4 : ℝ) : ℝ :=
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := t1 + t2 + t3 + t4
  totalDistance / totalTime

theorem average_speed_of_the_car :
  averageSpeed 30 35 65 (40 * 0.5) (30 / 45) (35 / 55) 1 0.5 = 54 := 
  by 
    sorry

end average_speed_of_the_car_l245_245560


namespace sub_neg_four_l245_245440

theorem sub_neg_four : -3 - 1 = -4 :=
by
  sorry

end sub_neg_four_l245_245440


namespace domain_g_l245_245756

noncomputable def f : ℝ → ℝ := sorry  -- f is a real-valued function

theorem domain_g:
  (∀ x, x ∈ [-2, 4] ↔ f x ∈ [-2, 4]) →  -- The domain of f(x) is [-2, 4]
  (∀ x, x ∈ [-2, 2] ↔ (f x + f (-x)) ∈ [-2, 2]) :=  -- The domain of g(x) = f(x) + f(-x) is [-2, 2]
by
  intros h
  sorry

end domain_g_l245_245756


namespace smallest_marbles_l245_245868

theorem smallest_marbles
  : ∃ n : ℕ, ((n % 8 = 5) ∧ (n % 7 = 2) ∧ (n = 37) ∧ (37 % 9 = 1)) :=
by
  sorry

end smallest_marbles_l245_245868


namespace roots_eqn_l245_245743

/-- Given that \alpha and \beta are the roots of x^2 - 3x - 4 = 0, prove that 4\alpha^3 + 9\beta^2 = -72. -/
theorem roots_eqn (α β : ℝ) (h_root_α : is_root (Polynomial.mk [ -4, -3, 1 ]) α)
  (h_root_β : is_root (Polynomial.mk [ -4, -3, 1 ]) β) :
  4 * α ^ 3 + 9 * β ^ 2 = -72 :=
by
  sorry

end roots_eqn_l245_245743


namespace solve_for_a_l245_245071

theorem solve_for_a (a : ℚ) (h : a + a / 4 = 10 / 4) : a = 2 :=
sorry

end solve_for_a_l245_245071


namespace age_ratio_l245_245488
open Nat

theorem age_ratio (B_c : ℕ) (h1 : B_c = 42) (h2 : ∀ A_c, A_c = B_c + 12) : (A_c + 10) / (B_c - 10) = 2 :=
by
  sorry

end age_ratio_l245_245488


namespace domain_of_logarithmic_function_l245_245917

theorem domain_of_logarithmic_function :
  ∀ x : ℝ, 2 - x > 0 ↔ x < 2 := 
by
  intro x
  sorry

end domain_of_logarithmic_function_l245_245917


namespace even_of_even_square_sqrt_two_irrational_l245_245414

-- Problem 1: Let p ∈ ℤ. Show that if p² is even, then p is even.
theorem even_of_even_square (p : ℤ) (h : p^2 % 2 = 0) : p % 2 = 0 :=
by
  sorry

-- Problem 2: Show that √2 is irrational.
theorem sqrt_two_irrational : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ a * a = 2 * b * b :=
by
  sorry

end even_of_even_square_sqrt_two_irrational_l245_245414


namespace trapezoid_angles_sum_l245_245126

theorem trapezoid_angles_sum {α β γ δ : ℝ} (h : α + β + γ + δ = 360) (h1 : α = 60) (h2 : β = 120) :
  γ + δ = 180 :=
by
  sorry

end trapezoid_angles_sum_l245_245126


namespace cupcakes_frosted_l245_245366

def Cagney_rate := 1 / 25
def Lacey_rate := 1 / 35
def time_duration := 600
def combined_rate := Cagney_rate + Lacey_rate
def total_cupcakes := combined_rate * time_duration

theorem cupcakes_frosted (Cagney_rate Lacey_rate time_duration combined_rate total_cupcakes : ℝ) 
  (hC: Cagney_rate = 1 / 25)
  (hL: Lacey_rate = 1 / 35)
  (hT: time_duration = 600)
  (hCR: combined_rate = Cagney_rate + Lacey_rate)
  (hTC: total_cupcakes = combined_rate * time_duration) :
  total_cupcakes = 41 :=
sorry

end cupcakes_frosted_l245_245366


namespace pool_filling_time_l245_245003

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end pool_filling_time_l245_245003


namespace ivan_running_distance_l245_245767

theorem ivan_running_distance (x MondayDistance TuesdayDistance WednesdayDistance ThursdayDistance FridayDistance : ℝ) 
  (h1 : MondayDistance = x)
  (h2 : TuesdayDistance = 2 * x)
  (h3 : WednesdayDistance = x)
  (h4 : ThursdayDistance = (1 / 2) * x)
  (h5 : FridayDistance = x)
  (hShortest : ThursdayDistance = 5) :
  MondayDistance + TuesdayDistance + WednesdayDistance + ThursdayDistance + FridayDistance = 55 :=
by
  sorry

end ivan_running_distance_l245_245767


namespace factor_expression_l245_245197

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) :=
by
  sorry

end factor_expression_l245_245197


namespace glass_volume_correct_l245_245892

-- Define the constants based on the problem conditions
def pessimist_empty_percent : ℝ := 0.60
def optimist_full_percent : ℝ := 0.60
def water_difference : ℝ := 46

-- Let V be the volume of the glass
def glass_volume (V : ℝ) : Prop :=
  let pessimist_full := (1 - pessimist_empty_percent) * V
  let optimist_full := optimist_full_percent * V
  optimist_full - pessimist_full = water_difference

-- The theorem to prove
theorem glass_volume_correct :
  ∃ V : ℝ, glass_volume V ∧ V = 230 :=
by
  sorry

end glass_volume_correct_l245_245892


namespace boris_needs_to_climb_four_times_l245_245482

/-- Hugo's mountain elevation is 10,000 feet above sea level. --/
def hugo_mountain_elevation : ℕ := 10_000

/-- Boris' mountain is 2,500 feet shorter than Hugo's mountain. --/
def boris_mountain_elevation : ℕ := hugo_mountain_elevation - 2_500

/-- Hugo climbed his mountain 3 times. --/
def hugo_climbs : ℕ := 3

/-- The total number of feet Hugo climbed. --/
def total_hugo_climb : ℕ := hugo_mountain_elevation * hugo_climbs

/-- The number of times Boris needs to climb his mountain to equal Hugo's climb. --/
def boris_climbs_needed : ℕ := total_hugo_climb / boris_mountain_elevation

theorem boris_needs_to_climb_four_times :
  boris_climbs_needed = 4 :=
by
  sorry

end boris_needs_to_climb_four_times_l245_245482


namespace union_of_A_and_B_l245_245212

open Set

def A : Set ℕ := {1, 3, 7, 8}
def B : Set ℕ := {1, 5, 8}

theorem union_of_A_and_B : A ∪ B = {1, 3, 5, 7, 8} := by
  sorry

end union_of_A_and_B_l245_245212


namespace productivity_comparison_l245_245850

theorem productivity_comparison 
  (cycle1_work_time cycle1_tea_time : ℕ)
  (cycle2_work_time cycle2_tea_time : ℕ)
  (cycle1_tea_interval cycle2_tea_interval : ℕ) :
  cycle1_work_time = 5 →
  cycle1_tea_time = 1 →
  cycle1_tea_interval = 5 →
  cycle2_work_time = 7 →
  cycle2_tea_time = 1 →
  cycle2_tea_interval = 7 →
  (by exact (((cycle2_tea_interval + cycle2_tea_time) * cycle1_work_time -
    (cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time) * 100 /
    ((cycle1_tea_interval + cycle1_tea_time) * cycle2_work_time)) = 5 : Prop) :=
begin
    intros h1 h2 h3 h4 h5 h6,
    calc (((7 + 1) * 5 - (5 + 1) * 7) * 100 / ((5 + 1) * 7)) : Int = 5 : by sorry
end

end productivity_comparison_l245_245850


namespace grace_age_l245_245607

/-- Grace's age calculation based on given family ages. -/
theorem grace_age (mother_age : ℕ) (grandmother_age : ℕ) (grace_age : ℕ)
  (h1 : mother_age = 80)
  (h2 : grandmother_age = 2 * mother_age)
  (h3 : grace_age = (3 * grandmother_age) / 8) : grace_age = 60 :=
by
  sorry

end grace_age_l245_245607


namespace problem1_problem2_l245_245933

variable (α : ℝ)

axiom tan_alpha_condition : Real.tan (Real.pi + α) = -1/2

-- Problem 1 Statement
theorem problem1 
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) : 
  (2 * Real.cos (Real.pi - α) - 3 * Real.sin (Real.pi + α)) / 
  (4 * Real.cos (α - 2 * Real.pi) + Real.cos (3 * Real.pi / 2 - α)) = -7/9 := 
sorry

-- Problem 2 Statement
theorem problem2
  (tan_alpha_condition : Real.tan (Real.pi + α) = -1/2) :
  Real.sin α ^ 2 - 2 * Real.sin α * Real.cos α + 4 * Real.cos α ^ 2 = 21/5 := 
sorry

end problem1_problem2_l245_245933


namespace average_gas_mileage_round_trip_l245_245428

/-
A student drives 150 miles to university in a sedan that averages 25 miles per gallon.
The same student drives 150 miles back home in a minivan that averages 15 miles per gallon.
Calculate the average gas mileage for the entire round trip.
-/
theorem average_gas_mileage_round_trip (d1 d2 m1 m2 : ℝ) (h1 : d1 = 150) (h2 : m1 = 25) 
  (h3 : d2 = 150) (h4 : m2 = 15) : 
  (2 * d1) / ((d1/m1) + (d2/m2)) = 18.75 := by
  sorry

end average_gas_mileage_round_trip_l245_245428


namespace slide_wait_is_shorter_l245_245667

theorem slide_wait_is_shorter 
  (kids_waiting_for_swings : ℕ)
  (kids_waiting_for_slide_multiplier : ℕ)
  (wait_per_kid_swings_minutes : ℕ)
  (wait_per_kid_slide_seconds : ℕ)
  (kids_waiting_for_swings = 3)
  (kids_waiting_for_slide_multiplier = 2)
  (wait_per_kid_swings_minutes = 2) 
  (wait_per_kid_slide_seconds = 15) :
  let total_wait_swings_seconds := wait_per_kid_swings_minutes * kids_waiting_for_swings * 60,
      kids_waiting_for_slide := kids_waiting_for_swings * kids_waiting_for_slide_multiplier,
      total_wait_slide_seconds := wait_per_kid_slide_seconds * kids_waiting_for_slide in
  270 = total_wait_swings_seconds - total_wait_slide_seconds :=
by
  sorry

end slide_wait_is_shorter_l245_245667


namespace find_m_l245_245717

-- Define the conditions
variables (a b r s m : ℝ)
variables (S1 S2 : ℝ)

-- Conditions
def first_term_first_series := a = 12
def second_term_first_series := a * r = 6
def first_term_second_series := b = 12
def second_term_second_series := 12 * s = 6 + m
def sum_relation := S2 = 3 * S1
def sum_first_series := S1 = a / (1 - r)
def sum_second_series := S2 = b / (1 - s)

-- Proof statement
theorem find_m (h1 : first_term_first_series a)
              (h2 : second_term_first_series a r)
              (h3 : first_term_second_series b)
              (h4 : second_term_second_series s m)
              (h5 : sum_relation S2 S1)
              (h6 : sum_first_series S1 a r)
              (h7 : sum_second_series S2 b s) : 
              m = 4 := 
sorry

end find_m_l245_245717


namespace product_of_distances_to_intersections_l245_245597

noncomputable def line_parametric {α : ℝ} (t : ℝ) :=
  (1 + t * Real.cos α, 1 + t * Real.sin α)

theorem product_of_distances_to_intersections (α : ℝ) (hα : α = Real.pi / 6) :
  ∃ A B : ℝ × ℝ, (line_parametric α A.fst = 2 * Real.cos hα) ∧ (line_parametric α B.fst = 2 * Real.sin hα) ∧
  (distance (1,1) A) * (distance (1,1) B) = 2 :=
by
  sorry

end product_of_distances_to_intersections_l245_245597


namespace percentage_students_school_A_l245_245622

theorem percentage_students_school_A
  (A B : ℝ)
  (h1 : A + B = 100)
  (h2 : 0.30 * A + 0.40 * B = 34) :
  A = 60 :=
sorry

end percentage_students_school_A_l245_245622


namespace Connie_correct_number_l245_245723

theorem Connie_correct_number (x : ℤ) (h : x + 2 = 80) : x - 2 = 76 := by
  sorry

end Connie_correct_number_l245_245723


namespace product_of_sums_of_two_squares_l245_245351

theorem product_of_sums_of_two_squares
  (a b a1 b1 : ℤ) :
  ((a^2 + b^2) * (a1^2 + b1^2)) = ((a * a1 - b * b1)^2 + (a * b1 + b * a1)^2) := 
sorry

end product_of_sums_of_two_squares_l245_245351


namespace car_highway_miles_per_tankful_l245_245283

-- Defining conditions as per given problem
def city_miles_per_tank : ℕ := 336
def city_miles_per_gallon : ℕ := 8
def difference_miles_per_gallon : ℕ := 3
def highway_miles_per_gallon := city_miles_per_gallon + difference_miles_per_gallon
def tank_size := city_miles_per_tank / city_miles_per_gallon
def highway_miles_per_tank := highway_miles_per_gallon * tank_size

-- Theorem statement to prove
theorem car_highway_miles_per_tankful :
  highway_miles_per_tank = 462 :=
sorry

end car_highway_miles_per_tankful_l245_245283


namespace find_number_l245_245277

theorem find_number : ∃ x : ℝ, (x / 6 * 12 = 10) ∧ x = 5 :=
by
 sorry

end find_number_l245_245277


namespace x_gt_1_sufficient_not_necessary_x_squared_gt_1_l245_245857

variable {x : ℝ}

-- Condition: $x > 1$
def condition_x_gt_1 (x : ℝ) : Prop := x > 1

-- Condition: $x^2 > 1$
def condition_x_squared_gt_1 (x : ℝ) : Prop := x^2 > 1

-- Theorem: Prove that $x > 1$ is a sufficient but not necessary condition for $x^2 > 1$
theorem x_gt_1_sufficient_not_necessary_x_squared_gt_1 :
  (condition_x_gt_1 x → condition_x_squared_gt_1 x) ∧ (¬ ∀ x, condition_x_squared_gt_1 x → condition_x_gt_1 x) :=
sorry

end x_gt_1_sufficient_not_necessary_x_squared_gt_1_l245_245857


namespace not_divisible_by_1998_l245_245423

theorem not_divisible_by_1998 (n : ℕ) :
  ∀ k : ℕ, ¬ (2^(k+1) * n + 2^k - 1) % 2 = 0 → ¬ (2^(k+1) * n + 2^k - 1) % 1998 = 0 :=
by
  intros _ _
  sorry

end not_divisible_by_1998_l245_245423


namespace calc_S_5_minus_S_4_l245_245060

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 2

theorem calc_S_5_minus_S_4 {a : ℕ → ℕ} {S : ℕ → ℕ}
  (h : sum_sequence a S) : S 5 - S 4 = 32 :=
by
  sorry

end calc_S_5_minus_S_4_l245_245060


namespace probability_same_color_l245_245863

theorem probability_same_color :
  let red_marble_prob := (5 / 21) * (4 / 20) * (3 / 19)
  let white_marble_prob := (6 / 21) * (5 / 20) * (4 / 19)
  let blue_marble_prob := (7 / 21) * (6 / 20) * (5 / 19)
  let green_marble_prob := (3 / 21) * (2 / 20) * (1 / 19)
  red_marble_prob + white_marble_prob + blue_marble_prob + green_marble_prob = 66 / 1330 := by
  sorry

end probability_same_color_l245_245863


namespace natural_number_triplets_l245_245043

theorem natural_number_triplets (x y z : ℕ) : 
  3^x + 4^y = 5^z → 
  (x = 2 ∧ y = 2 ∧ z = 2) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by 
  sorry

end natural_number_triplets_l245_245043


namespace max_x_plus_y_l245_245973

theorem max_x_plus_y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 9) (h2 : 2 * x + 4 * y ≤ 8) : 
  x + y ≤ 7 / 3 :=
sorry

end max_x_plus_y_l245_245973


namespace exists_infinitely_many_pairs_exists_single_statement_only_l245_245057

-- Define the sequence
def sequence (α β a b c : ℤ) : ℤ → ℤ
| 1 => α
| 2 => β
| (n + 2) => a * (sequence (α) (β) (a) (b) (c) (n + 1)) + b * (sequence (α) (β) (a) (b) (c) n) + c

-- First part: there are infinitely many pairs (α, β) such that u_2023 = 2^2022
theorem exists_infinitely_many_pairs (α β : ℤ) :
  ∀ k : ℤ, ∃ (α β : ℤ), sequence α β 3 -2 -1 2023 = 2^(2022) := 
  sorry

-- Second part: proves that there exists an n_0 such that only one of the given conditions is true
theorem exists_single_statement_only (α β : ℤ) :
  ∃ (n_0 : ℕ), (∀ m : ℕ, (sequence α β 3 -2 -1 (n_0 + m + 1) = 7^(2023) ∨ sequence α β 3 -2 -1 (n_0 + m + 1) = 17^(2023)) 
  ∨ (∀ k : ℕ, sequence α β 3 -2 -1 (n_0 + k + 1) - 1 = 2023)) := 
  sorry

end exists_infinitely_many_pairs_exists_single_statement_only_l245_245057


namespace tangent_sum_half_angles_l245_245512

-- Lean statement for the proof problem
theorem tangent_sum_half_angles (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.tan (A / 2) * Real.tan (B / 2) + 
  Real.tan (B / 2) * Real.tan (C / 2) + 
  Real.tan (C / 2) * Real.tan (A / 2) = 1 := 
by
  sorry

end tangent_sum_half_angles_l245_245512


namespace find_m_and_b_sum_l245_245527

noncomputable theory
open Classical

-- Definitions of points and line
def point (x y : ℝ) := (x, y)

def reflected (p₁ p₂ : ℝ × ℝ) (m b : ℝ) : Prop := 
  let (x₁, y₁) := p₁ in 
  let (x₂, y₂) := p₂ in
  y₂ = 2 * (-m * x₁ + y₁ + b) - y₁ ∧ x₂ = 2 * (m * y₂ + b * m - b * m) / m - x₁

-- Given conditions
def original := point 2 3
def image := point 10 7

-- Assertion to prove
theorem find_m_and_b_sum
  (m b : ℝ)
  (h : reflected original image m b) : m + b = 15 :=
sorry

end find_m_and_b_sum_l245_245527


namespace problem_l245_245745

open Real

theorem problem (x y : ℝ) (h1 : 3 * x + 2 * y = 8) (h2 : 2 * x + 3 * y = 11) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 2041 / 25 :=
sorry

end problem_l245_245745


namespace total_wings_count_l245_245084

theorem total_wings_count (num_planes : ℕ) (wings_per_plane : ℕ) (h_planes : num_planes = 54) (h_wings : wings_per_plane = 2) : num_planes * wings_per_plane = 108 :=
by 
  sorry

end total_wings_count_l245_245084


namespace potatoes_left_l245_245641

theorem potatoes_left (initial_potatoes : ℕ) (potatoes_for_salads : ℕ) (potatoes_for_mashed : ℕ)
  (h1 : initial_potatoes = 52)
  (h2 : potatoes_for_salads = 15)
  (h3 : potatoes_for_mashed = 24) :
  initial_potatoes - (potatoes_for_salads + potatoes_for_mashed) = 13 := by
  sorry

end potatoes_left_l245_245641


namespace probability_failed_both_tests_eq_l245_245418

variable (total_students pass_test1 pass_test2 pass_both : ℕ)

def students_failed_both_tests (total pass1 pass2 both : ℕ) : ℕ :=
  total - (pass1 + pass2 - both)

theorem probability_failed_both_tests_eq 
  (h_total : total_students = 100)
  (h_pass1 : pass_test1 = 60)
  (h_pass2 : pass_test2 = 40)
  (h_pass_both : pass_both = 20) :
  students_failed_both_tests total_students pass_test1 pass_test2 pass_both / (total_students : ℚ) = 0.2 :=
by
  sorry

end probability_failed_both_tests_eq_l245_245418


namespace part1_part2_l245_245584

def f (x : ℝ) := |x + 4| - |x - 1|
def g (x : ℝ) := |2 * x - 1| + 3

theorem part1 (x : ℝ) : (f x > 3) → x > 0 :=
by sorry

theorem part2 (a : ℝ) : (∃ x, f x + 1 < 4^a - 5 * 2^a) ↔ (a < 0 ∨ a > 2) :=
by sorry

end part1_part2_l245_245584


namespace evaluate_expression_l245_245721

-- Define the mathematical expressions using Lean's constructs
def expr1 : ℕ := 201 * 5 + 1220 - 2 * 3 * 5 * 7

-- State the theorem we aim to prove
theorem evaluate_expression : expr1 = 2015 := by
  sorry

end evaluate_expression_l245_245721


namespace largest_prime_factor_of_4620_l245_245832

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / m → ¬ (m ∣ n)

def prime_factors (n : ℕ) : List ℕ :=
  -- assumes a well-defined function that generates the prime factor list
  -- this is a placeholder function for demonstrating purposes
  sorry

def largest_prime_factor (l : List ℕ) : ℕ :=
  l.foldr max 0

theorem largest_prime_factor_of_4620 : largest_prime_factor (prime_factors 4620) = 11 :=
by
  sorry

end largest_prime_factor_of_4620_l245_245832


namespace chef_earns_less_than_manager_l245_245554

noncomputable def manager_wage : ℝ := 6.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage + 0.2 * dishwasher_wage

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 2.60 :=
by
  sorry

end chef_earns_less_than_manager_l245_245554


namespace rojo_speed_l245_245793

theorem rojo_speed (R : ℝ) 
  (H : 32 = (R + 3) * 4) : R = 5 :=
sorry

end rojo_speed_l245_245793


namespace problem_statement_l245_245835

-- Defining the terms x, y, and d as per the problem conditions
def x : ℕ := 2351
def y : ℕ := 2250
def d : ℕ := 121

-- Stating the proof problem in Lean
theorem problem_statement : (x - y)^2 / d = 84 := by
  sorry

end problem_statement_l245_245835


namespace tim_paid_correct_amount_l245_245120

-- Define the conditions given in the problem
def mri_cost : ℝ := 1200
def doctor_hourly_rate : ℝ := 300
def doctor_time_hours : ℝ := 0.5 -- 30 minutes is half an hour
def fee_for_being_seen : ℝ := 150
def insurance_coverage_rate : ℝ := 0.80

-- Total amount Tim paid calculation
def total_cost_before_insurance : ℝ :=
  mri_cost + (doctor_hourly_rate * doctor_time_hours) + fee_for_being_seen

def insurance_coverage : ℝ :=
  total_cost_before_insurance * insurance_coverage_rate

def amount_tim_paid : ℝ :=
  total_cost_before_insurance - insurance_coverage

-- Prove that Tim paid $300
theorem tim_paid_correct_amount : amount_tim_paid = 300 :=
by
  sorry

end tim_paid_correct_amount_l245_245120


namespace correct_sum_l245_245979

theorem correct_sum (a b c n : ℕ) (h_m_pos : 100 * a + 10 * b + c > 0) (h_n_pos : n > 0)
    (h_err_sum : 100 * a + 10 * c + b + n = 128) : 100 * a + 10 * b + c + n = 128 := 
by
  sorry

end correct_sum_l245_245979


namespace must_be_true_l245_245466

noncomputable def f (x : ℝ) := |Real.log x|

theorem must_be_true (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) 
                     (h3 : f b < f a) (h4 : f a < f c) :
                     (c > 1) ∧ (1 / c < a) ∧ (a < 1) ∧ (a < b) ∧ (b < 1 / a) :=
by
  sorry

end must_be_true_l245_245466


namespace angle_equality_l245_245998

open EuclideanGeometry

variables {A B C D P : Point}
variables {α β γ δ : ℝ}

-- Definition of a parallelogram by its vertices
def parallelogram (A B C D : Point) : Prop :=
  collinear A B C ∧ collinear B C D ∧ collinear C D A ∧ collinear D A B

theorem angle_equality (parallelogram A B C D : parallelogram A B C D)
  (P_outside : ¬_inside_of_parallelogram P A B C D)
  (angle_PAB_eq_angle_PCB : ∠ P A B = ∠ P C B)
  (opposite_directions : ∠ P A B + ∠ P C B = π):
  ∠ A P B = ∠ D P C :=
by
  sorry

end angle_equality_l245_245998


namespace jorge_goals_this_season_l245_245630

def jorge_goals_last_season : Nat := 156
def jorge_goals_total : Nat := 343

theorem jorge_goals_this_season :
  ∃ g_s : Nat, g_s = jorge_goals_total - jorge_goals_last_season ∧ g_s = 187 :=
by
  -- proof goes here, we use 'sorry' for now
  sorry

end jorge_goals_this_season_l245_245630


namespace amount_received_by_Sam_l245_245794

noncomputable def final_amount (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem amount_received_by_Sam 
  (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ)
  (hP : P = 12000) (hr : r = 0.10) (hn : n = 2) (ht : t = 1) :
  final_amount P r n t = 12607.50 :=
by
  sorry

end amount_received_by_Sam_l245_245794


namespace feet_heads_difference_l245_245229

theorem feet_heads_difference :
  let hens := 60
  let goats := 35
  let camels := 6
  let keepers := 10
  let heads := hens + goats + camels + keepers
  let feet := (2 * hens) + (4 * goats) + (4 * camels) + (2 * keepers)
  feet - heads = 193 :=
by
  sorry

end feet_heads_difference_l245_245229


namespace evaluate_expression_l245_245451

theorem evaluate_expression :
  (4^1001 * 9^1002) / (6^1002 * 4^1000) = (3^1002) / (2^1000) :=
by sorry

end evaluate_expression_l245_245451


namespace hot_peppers_percentage_correct_l245_245774

def sunday_peppers : ℕ := 7
def monday_peppers : ℕ := 12
def tuesday_peppers : ℕ := 14
def wednesday_peppers : ℕ := 12
def thursday_peppers : ℕ := 5
def friday_peppers : ℕ := 18
def saturday_peppers : ℕ := 12
def non_hot_peppers : ℕ := 64

def total_peppers : ℕ := sunday_peppers + monday_peppers + tuesday_peppers + wednesday_peppers + thursday_peppers + friday_peppers + saturday_peppers
def hot_peppers : ℕ := total_peppers - non_hot_peppers
def hot_peppers_percentage : ℕ := (hot_peppers * 100) / total_peppers

theorem hot_peppers_percentage_correct : hot_peppers_percentage = 20 := 
by 
  sorry

end hot_peppers_percentage_correct_l245_245774


namespace angle_between_apothems_correct_l245_245189

noncomputable def angle_between_apothems (n : ℕ) (α : ℝ) : ℝ :=
  2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2))

theorem angle_between_apothems_correct (n : ℕ) (α : ℝ) (h1 : 0 < n) (h2 : 0 < α) (h3 : α < 2 * Real.pi) :
  angle_between_apothems n α = 2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2)) :=
by
  sorry

end angle_between_apothems_correct_l245_245189


namespace increase_in_expenses_is_20_percent_l245_245702

noncomputable def man's_salary : ℝ := 6500
noncomputable def initial_savings : ℝ := 0.20 * man's_salary
noncomputable def new_savings : ℝ := 260
noncomputable def reduction_in_savings : ℝ := initial_savings - new_savings
noncomputable def initial_expenses : ℝ := 0.80 * man's_salary
noncomputable def increase_in_expenses_percentage : ℝ := (reduction_in_savings / initial_expenses) * 100

theorem increase_in_expenses_is_20_percent :
  increase_in_expenses_percentage = 20 := by
  sorry

end increase_in_expenses_is_20_percent_l245_245702


namespace no_linear_term_in_product_l245_245618

theorem no_linear_term_in_product (a : ℝ) (h : ∀ x : ℝ, (x + 4) * (x + a) - x^2 - 4 * a = 0) : a = -4 :=
sorry

end no_linear_term_in_product_l245_245618


namespace totalFourOfAKindCombinations_l245_245698

noncomputable def numberOfFourOfAKindCombinations : Nat :=
  13 * 48

theorem totalFourOfAKindCombinations : numberOfFourOfAKindCombinations = 624 := by
  sorry

end totalFourOfAKindCombinations_l245_245698


namespace beth_crayon_packs_l245_245295

theorem beth_crayon_packs (P : ℕ) (h1 : 10 * P + 6 = 46) : P = 4 :=
by
  sorry

end beth_crayon_packs_l245_245295


namespace gcf_60_75_l245_245134

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l245_245134


namespace mechanism_parts_l245_245158

-- Definitions
def total_parts (S L : Nat) : Prop := S + L = 25
def condition1 (S L : Nat) : Prop := ∀ (A : Finset (Fin 25)), (A.card = 12) → ∃ i, i ∈ A ∧ i < S
def condition2 (S L : Nat) : Prop := ∀ (B : Finset (Fin 25)), (B.card = 15) → ∃ i, i ∈ B ∧ i >= S

-- Main statement
theorem mechanism_parts :
  ∃ (S L : Nat), 
  total_parts S L ∧ 
  condition1 S L ∧ 
  condition2 S L ∧ 
  S = 14 ∧ 
  L = 11 :=
sorry

end mechanism_parts_l245_245158


namespace trigonometric_identity_proof_l245_245751

variable (α : ℝ)

theorem trigonometric_identity_proof
  (h : Real.tan (α + Real.pi / 4) = -3) :
  Real.cos (2 * α) + 2 * Real.sin (2 * α) = 1 :=
by
  sorry

end trigonometric_identity_proof_l245_245751


namespace art_group_students_count_l245_245357

theorem art_group_students_count (x : ℕ) (h1 : x * (1 / 60) + 2 * (x + 15) * (1 / 60) = 1) : x = 10 :=
by {
  sorry
}

end art_group_students_count_l245_245357


namespace students_prefer_dogs_l245_245181

theorem students_prefer_dogs (total_students : ℕ) (perc_dogs_vg perc_dogs_mv : ℕ) (h_total: total_students = 30)
  (h_perc_dogs_vg: perc_dogs_vg = 50) (h_perc_dogs_mv: perc_dogs_mv = 10) :
  total_students * perc_dogs_vg / 100 + total_students * perc_dogs_mv / 100 = 18 := by
  sorry

end students_prefer_dogs_l245_245181


namespace lines_parallel_m_eq_neg5_l245_245216

theorem lines_parallel_m_eq_neg5 (m : ℝ) :
  (∀ x y : ℝ, mx + 2y - 2 = 0 → 5x + (m + 3)y - 5 = 0) → m = -5 :=
by
  sorry

end lines_parallel_m_eq_neg5_l245_245216


namespace find_a_equals_two_l245_245946

noncomputable def a := ((7 + 4 * Real.sqrt 3) ^ (1 / 2) - (7 - 4 * Real.sqrt 3) ^ (1 / 2)) / Real.sqrt 3

theorem find_a_equals_two : a = 2 := 
sorry

end find_a_equals_two_l245_245946


namespace petya_square_larger_than_vasya_square_l245_245786

variable (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)

def petya_square_side (a b : ℝ) : ℝ := a * b / (a + b)

def vasya_square_side (a b : ℝ) : ℝ := a * b * Real.sqrt (a^2 + b^2) / (a^2 + a * b + b^2)

theorem petya_square_larger_than_vasya_square
  (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  petya_square_side a b > vasya_square_side a b :=
by sorry

end petya_square_larger_than_vasya_square_l245_245786


namespace min_value_proof_l245_245961

open Classical

variable (a : Nat → ℕ)
variable (m n : ℕ)
variable {q : ℕ}

axiom geom_seq (q_pos : 0 < q) : ∀ (n : ℕ), a (n + 1) = a 1 * q^n

axiom condition1 (h1 : a 2016 = a 2015 + 2 * a 2014) : True

axiom condition2 (h2 : ∀ m n, a m * a n = 16 * a 1 ^ 2) : m + n = 6

noncomputable def min_value : ℚ := 
  let frac_sum := (4 / m : ℚ) + (1 / n : ℚ)
  frac_sum

theorem min_value_proof 
  (q_eq : q = 2)
  (mn_eq : m + n = 6) :
  ∀ m n, min_value m n = 3 / 2 := 
  by
    sorry

end min_value_proof_l245_245961


namespace arithmetic_square_root_of_sqrt_81_l245_245652

def square_root (x : ℝ) : ℝ :=
  real.sqrt x

theorem arithmetic_square_root_of_sqrt_81 : square_root 81 = 9 :=
by
  sorry

end arithmetic_square_root_of_sqrt_81_l245_245652


namespace max_daily_sales_l245_245285

def f (t : ℕ) : ℝ := -2 * t + 200
def g (t : ℕ) : ℝ :=
  if t ≤ 30 then 12 * t + 30
  else 45

def S (t : ℕ) : ℝ := f t * g t

theorem max_daily_sales : ∃ t, 1 ≤ t ∧ t ≤ 50 ∧ S t = 54600 := 
  sorry

end max_daily_sales_l245_245285


namespace minimum_x_plus_y_l245_245203

variable (x y : ℝ)
variable (h1 : 0 < x)
variable (h2 : 0 < y)
variable (h3 : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1)

theorem minimum_x_plus_y (hx : 0 < x) (hy : 0 < y) (h : (1 / (2 * x + y)) + (4 / (2 * x + 3 * y)) = 1) : x + y ≥ 9 / 4 :=
sorry

end minimum_x_plus_y_l245_245203


namespace sqrt_D_rational_sometimes_not_l245_245974

-- Definitions and conditions
def D (a : ℤ) : ℤ := a^2 + (a + 2)^2 + (a * (a + 2))^2

-- The statement to prove
theorem sqrt_D_rational_sometimes_not (a : ℤ) : ∃ x : ℚ, x = Real.sqrt (D a) ∧ ¬(∃ y : ℤ, x = y) ∨ ∃ y : ℤ, Real.sqrt (D a) = y :=
by 
  sorry

end sqrt_D_rational_sometimes_not_l245_245974


namespace valid_numbers_count_l245_245749

def is_valid_digit (d : ℕ) : Prop :=
  d ≠ 5 ∧ d < 10

def count_valid_numbers : ℕ :=
  let first_digit_choices := 8 -- from 1 to 9 excluding 5
  let second_digit_choices := 8 -- from the digits (0-9 excluding 5 and first digit)
  let third_digit_choices := 7 -- from the digits (0-9 excluding 5 and first two digits)
  let fourth_digit_choices := 6 -- from the digits (0-9 excluding 5 and first three digits)
  first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices

theorem valid_numbers_count : count_valid_numbers = 2688 :=
  by
  sorry

end valid_numbers_count_l245_245749


namespace shaded_fraction_is_4_over_15_l245_245895

-- Define the geometric series sum function
def geom_series_sum (a r : ℝ) (hr : |r| < 1) : ℝ := a / (1 - r)

-- The target statement for the given problem
theorem shaded_fraction_is_4_over_15 :
  let a := (1 / 4 : ℝ)
  let r := (1 / 16 : ℝ)
  geom_series_sum a r (by norm_num : |r| < 1) = (4 / 15 : ℝ) :=
by
  -- Proof is omitted with sorry
  sorry

end shaded_fraction_is_4_over_15_l245_245895


namespace factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l245_245922

theorem factorize_x3_minus_9x (x : ℝ) : x^3 - 9 * x = x * (x + 3) * (x - 3) :=
sorry

theorem factorize_a3b_minus_2a2b_plus_ab (a b : ℝ) : a^3 * b - 2 * a^2 * b + a * b = a * b * (a - 1)^2 :=
sorry

end factorize_x3_minus_9x_factorize_a3b_minus_2a2b_plus_ab_l245_245922


namespace solve_equation_l245_245109

theorem solve_equation : ∀ x : ℝ, (x + 2) / 4 - 1 = (2 * x + 1) / 3 → x = -2 :=
by
  intro x
  intro h
  sorry  

end solve_equation_l245_245109


namespace jorge_goals_l245_245632

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end jorge_goals_l245_245632


namespace min_tries_to_get_blue_and_yellow_l245_245151

theorem min_tries_to_get_blue_and_yellow 
  (purple blue yellow : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5)
  (h_yellow : yellow = 11) :
  ∃ n, n = 9 ∧ (∀ tries, tries ≥ n → (∃ i j, (i ≤ purple ∧ j ≤ tries - i ∧ j ≤ blue) → (∃ k, k = tries - i - j ∧ k ≤ yellow))) :=
by sorry

end min_tries_to_get_blue_and_yellow_l245_245151


namespace find_m_and_b_sum_l245_245526

noncomputable theory
open Classical

-- Definitions of points and line
def point (x y : ℝ) := (x, y)

def reflected (p₁ p₂ : ℝ × ℝ) (m b : ℝ) : Prop := 
  let (x₁, y₁) := p₁ in 
  let (x₂, y₂) := p₂ in
  y₂ = 2 * (-m * x₁ + y₁ + b) - y₁ ∧ x₂ = 2 * (m * y₂ + b * m - b * m) / m - x₁

-- Given conditions
def original := point 2 3
def image := point 10 7

-- Assertion to prove
theorem find_m_and_b_sum
  (m b : ℝ)
  (h : reflected original image m b) : m + b = 15 :=
sorry

end find_m_and_b_sum_l245_245526


namespace gcf_60_75_l245_245130

theorem gcf_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcf_60_75_l245_245130


namespace loaned_books_during_month_l245_245162

-- Definitions corresponding to the conditions
def initial_books : ℕ := 75
def returned_percent : ℚ := 0.65
def end_books : ℕ := 68

-- Proof statement
theorem loaned_books_during_month (x : ℕ) 
  (h1 : returned_percent = 0.65)
  (h2 : initial_books = 75)
  (h3 : end_books = 68) :
  (0.35 * x : ℚ) = (initial_books - end_books) :=
sorry

end loaned_books_during_month_l245_245162


namespace linear_transformation_proof_l245_245929

theorem linear_transformation_proof (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) :
  ∃ (k b : ℝ), k = 4 ∧ b = -1 ∧ (y = k * x + b ∧ -1 ≤ y ∧ y ≤ 3) :=
by
  sorry

end linear_transformation_proof_l245_245929


namespace tangent_circles_t_value_l245_245339

theorem tangent_circles_t_value (t : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 = t^2 → x^2 + y^2 + 6 * x - 8 * y + 24 = 0 → dist (0, 0) (-3, 4) = t + 1) → t = 4 :=
by
  sorry

end tangent_circles_t_value_l245_245339


namespace at_least_one_number_greater_than_16000_l245_245831

theorem at_least_one_number_greater_than_16000 
    (numbers : Fin 20 → ℕ) 
    (h_distinct : Function.Injective numbers)
    (h_square_product : ∀ i : Fin 19, ∃ k : ℕ, numbers i * numbers (i + 1) = k^2)
    (h_first : numbers 0 = 42) :
    ∃ i : Fin 20, numbers i > 16000 :=
by
  sorry

end at_least_one_number_greater_than_16000_l245_245831


namespace cost_per_pizza_is_12_l245_245249

def numberOfPeople := 15
def peoplePerPizza := 3
def earningsPerNight := 4
def nightsBabysitting := 15

-- We aim to prove that the cost per pizza is $12
theorem cost_per_pizza_is_12 : 
  (earningsPerNight * nightsBabysitting) / (numberOfPeople / peoplePerPizza) = 12 := 
by 
  sorry

end cost_per_pizza_is_12_l245_245249


namespace maximum_value_a_plus_b_cubed_plus_c_fourth_l245_245238

theorem maximum_value_a_plus_b_cubed_plus_c_fourth (a b c : ℝ)
    (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
    (h_sum : a + b + c = 2) : a + b^3 + c^4 ≤ 2 :=
sorry

end maximum_value_a_plus_b_cubed_plus_c_fourth_l245_245238


namespace task2_X_alone_l245_245004

namespace TaskWork

variables (r_X r_Y r_Z : ℝ)

-- Task 1 conditions
axiom task1_XY : r_X + r_Y = 1 / 4
axiom task1_YZ : r_Y + r_Z = 1 / 6
axiom task1_XZ : r_X + r_Z = 1 / 3

-- Task 2 condition
axiom task2_XYZ : r_X + r_Y + r_Z = 1 / 2

-- Theorem to be proven
theorem task2_X_alone : 1 / r_X = 4.8 :=
sorry

end TaskWork

end task2_X_alone_l245_245004


namespace glass_volume_is_230_l245_245886

theorem glass_volume_is_230
  (is_60_percent_empty : ∀ V : ℝ, V_pess : ℝ, V_pess = 0.40 * V)
  (is_60_percent_full : ∀ V : ℝ, V_opt : ℝ, V_opt = 0.60 * V)
  (water_difference : ∀ V_pess V_opt : ℝ, V_opt - V_pess = 46) :
  ∃ V : ℝ, V = 230 :=
by {
  sorry
}

end glass_volume_is_230_l245_245886


namespace units_digit_fib_cycle_length_60_l245_245986

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib n + fib (n+1)

-- Define the function to get the units digit (mod 10)
def units_digit_fib (n : ℕ) : ℕ :=
  (fib n) % 10

-- State the theorem about the cycle length of the units digits in Fibonacci sequence
theorem units_digit_fib_cycle_length_60 :
  ∃ k, k = 60 ∧ ∀ n, units_digit_fib (n + k) = units_digit_fib n := sorry

end units_digit_fib_cycle_length_60_l245_245986


namespace Vlad_score_l245_245377

-- Defining the initial conditions of the problem
def total_rounds : ℕ := 30
def points_per_win : ℕ := 5
def total_points : ℕ := total_rounds * points_per_win

-- Taro's score as described in the problem
def Taros_score := (3 * total_points / 5) - 4

-- Prove that Vlad's score is 64 points
theorem Vlad_score : total_points - Taros_score = 64 := by
  sorry

end Vlad_score_l245_245377


namespace factored_quadratic_even_b_l245_245521

theorem factored_quadratic_even_b
  (c d e f y : ℤ)
  (h1 : c * e = 45)
  (h2 : d * f = 45) 
  (h3 : ∃ b, 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) :
  ∃ b, (45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) ∧ (b % 2 = 0) :=
by
  sorry

end factored_quadratic_even_b_l245_245521


namespace max_length_shortest_arc_l245_245056

theorem max_length_shortest_arc (C : ℝ) (hC : C = 84) : 
  ∃ shortest_arc_length : ℝ, shortest_arc_length = 2 :=
by
  -- now prove it
  sorry

end max_length_shortest_arc_l245_245056


namespace divide_circle_three_equal_areas_l245_245730

theorem divide_circle_three_equal_areas (OA : ℝ) (r1 r2 : ℝ) 
  (hr1 : r1 = (OA * Real.sqrt 3) / 3) 
  (hr2 : r2 = (OA * Real.sqrt 6) / 3) : 
  ∀ (r : ℝ), r = OA → 
  (∀ (A1 A2 A3 : ℝ), A1 = π * r1 ^ 2 ∧ A2 = π * (r2 ^ 2 - r1 ^ 2) ∧ A3 = π * (r ^ 2 - r2 ^ 2) →
  A1 = A2 ∧ A2 = A3) :=
by
  sorry

end divide_circle_three_equal_areas_l245_245730


namespace triangle_must_be_equilateral_l245_245807

-- Given an incircle touching the sides at points A', B', and C', respectively
def incircle_touches (A B C A' B' C': Point) (triangleABC : Triangle A B C) :=
  touches (incircle triangleABC) (segment A A') ∧
  touches (incircle triangleABC) (segment B B') ∧
  touches (incircle triangleABC) (segment C C')

-- Given the condition that AA' = BB' = CC'
def equal_distances_from_vertices_to_tangency_points (A B C A' B' C': Point) := 
  dist A A' = dist B B' ∧ 
  dist B B' = dist C C'

-- Prove that triangle ABC must be equilateral
theorem triangle_must_be_equilateral
  (A B C A' B' C' : Point)
  (triangleABC : Triangle A B C)
  (h1: incircle_touches A B C A' B' C' triangleABC)
  (h2: equal_distances_from_vertices_to_tangency_points A B C A' B' C') :
  is_equilateral triangleABC :=
sorry

end triangle_must_be_equilateral_l245_245807


namespace solution_set_of_inequality_l245_245827

theorem solution_set_of_inequality (a : ℝ) (h1 : 2 * a - 3 < 0) (h2 : 1 - a < 0) : 1 < a ∧ a < 3 / 2 :=
by
  sorry

end solution_set_of_inequality_l245_245827


namespace percentage_calculation_l245_245280

-- Definitions based on conditions
def x : ℕ := 5200
def p1 : ℚ := 0.50
def p2 : ℚ := 0.30
def p3 : ℚ := 0.15

-- The theorem stating the desired proof
theorem percentage_calculation : p3 * (p2 * (p1 * x)) = 117 := by
  sorry

end percentage_calculation_l245_245280


namespace productivity_comparison_l245_245844

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l245_245844


namespace alice_age_l245_245463

theorem alice_age (x : ℕ) (h1 : ∃ n : ℕ, x - 4 = n^2) (h2 : ∃ m : ℕ, x + 2 = m^3) : x = 58 :=
sorry

end alice_age_l245_245463


namespace math_problem_proof_l245_245173

variable {a : ℝ} (ha : a > 0)

theorem math_problem_proof : ((36 * a^9)^4 * (63 * a^9)^4 = a^(72)) :=
by sorry

end math_problem_proof_l245_245173


namespace range_of_f_l245_245325

noncomputable def f (x : ℝ) : ℝ := 2^(2*x) + 2^(x+1) + 3

theorem range_of_f : ∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y > 3 :=
by
  sorry

end range_of_f_l245_245325


namespace student_distribution_l245_245670

theorem student_distribution (a b : ℕ) (h1 : a + b = 81) (h2 : a = b - 9) : a = 36 ∧ b = 45 := 
by
  sorry

end student_distribution_l245_245670


namespace glass_volume_l245_245882

theorem glass_volume (V_P V_O : ℝ) (V : ℝ) (h1 : V_P = 0.40 * V) (h2 : V_O = 0.60 * V) (h3 : V_O - V_P = 46) : V = 230 := by
  sorry

end glass_volume_l245_245882


namespace number_of_even_digits_in_base7_of_528_l245_245048

/-
  Define the base-7 representation of a number and a predicate to count even digits.
-/

-- Definition of base-7 digit representation
def base7_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else (List.unfoldr (λ n, if n = 0 then Option.none else some (n % 7, n / 7)) n).reverse

-- Predicate to check if a digit is even
def is_even (d : ℕ) : Bool := d % 2 = 0

-- Counting the even digits in base-7 representation
def count_even_digits_in_base7 (n : ℕ) : ℕ :=
  (base7_repr n).countp is_even

-- The target theorem to prove
theorem number_of_even_digits_in_base7_of_528 : count_even_digits_in_base7 528 = 0 :=
by sorry

end number_of_even_digits_in_base7_of_528_l245_245048


namespace divisibility_equiv_l245_245253

theorem divisibility_equiv (n : ℕ) : (7 ∣ 3^n + n^3) ↔ (7 ∣ 3^n * n^3 + 1) :=
by sorry

end divisibility_equiv_l245_245253


namespace sqrt_a_minus_b_squared_eq_one_l245_245977

noncomputable def PointInThirdQuadrant (a b : ℝ) : Prop :=
  a < 0 ∧ b < 0

noncomputable def DistanceToYAxis (a : ℝ) : Prop :=
  abs a = 5

noncomputable def BCondition (b : ℝ) : Prop :=
  abs (b + 1) = 3

theorem sqrt_a_minus_b_squared_eq_one
    (a b : ℝ)
    (h1 : PointInThirdQuadrant a b)
    (h2 : DistanceToYAxis a)
    (h3 : BCondition b) :
    Real.sqrt ((a - b) ^ 2) = 1 := 
  sorry

end sqrt_a_minus_b_squared_eq_one_l245_245977


namespace arithmetic_sequence_terms_l245_245623

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 3 + a 4 = 10) 
  (h2 : a (n - 3) + a (n - 2) = 30) 
  (h3 : (n * (a 1 + a n)) / 2 = 100) : 
  n = 10 :=
sorry

end arithmetic_sequence_terms_l245_245623


namespace express_as_scientific_notation_l245_245290

-- Define the question and condition
def trillion : ℝ := 1000000000000
def num := 6.13 * trillion

-- The main statement to be proven
theorem express_as_scientific_notation : num = 6.13 * 10^12 :=
by
  sorry

end express_as_scientific_notation_l245_245290


namespace y_in_interval_l245_245160

theorem y_in_interval :
  ∃ (y : ℝ), y = 5 + (1/y) * -y ∧ 2 < y ∧ y ≤ 4 :=
by
  sorry

end y_in_interval_l245_245160


namespace arithmetic_progr_property_l245_245469

theorem arithmetic_progr_property (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 + a 3 = 5 / 2)
  (h2 : a 2 + a 4 = 5 / 4)
  (h3 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2)
  (h4 : a 3 = a 1 + 2 * (a 2 - a 1))
  (h5 : a 2 = a 1 + (a 2 - a 1)) :
  S 3 / a 3 = 6 := sorry

end arithmetic_progr_property_l245_245469


namespace Gina_kept_170_l245_245592

def initial_amount : ℕ := 400
def mom_share : ℚ := 1 / 4
def clothes_share : ℚ := 1 / 8
def charity_share : ℚ := 1 / 5

def amount_to_mom : ℕ := initial_amount * (mom_share.to_nat)
def amount_on_clothes : ℕ := initial_amount * (clothes_share.to_nat)
def amount_to_charity : ℕ := initial_amount * (charity_share.to_nat)

def total_given_away : ℕ := amount_to_mom + amount_on_clothes + amount_to_charity
def remaining_amount : ℕ := initial_amount - total_given_away

theorem Gina_kept_170 :
  remaining_amount = 170 :=
by
  sorry

end Gina_kept_170_l245_245592


namespace units_digit_of_exponentiated_product_l245_245037

theorem units_digit_of_exponentiated_product :
  (2 ^ 2101 * 5 ^ 2102 * 11 ^ 2103) % 10 = 0 := 
sorry

end units_digit_of_exponentiated_product_l245_245037


namespace gcd_45_81_63_l245_245334

theorem gcd_45_81_63 : Nat.gcd 45 (Nat.gcd 81 63) = 9 := 
sorry

end gcd_45_81_63_l245_245334


namespace distinct_integer_roots_l245_245188

theorem distinct_integer_roots (a : ℤ) : 
  (∃ u v : ℤ, u ≠ v ∧ (u + v = -a) ∧ (u * v = 2 * a)) ↔ a = -1 ∨ a = 9 :=
by
  sorry

end distinct_integer_roots_l245_245188


namespace cost_of_items_l245_245115

variable (p q r : ℝ)

theorem cost_of_items :
  8 * p + 2 * q + r = 4.60 → 
  2 * p + 5 * q + r = 3.90 → 
  p + q + 3 * r = 2.75 → 
  4 * p + 3 * q + 2 * r = 7.4135 :=
by
  intros h1 h2 h3
  sorry

end cost_of_items_l245_245115


namespace value_of_expression_l245_245395

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end value_of_expression_l245_245395


namespace evaluate_T_l245_245033

def T (a b : ℤ) : ℤ := 4 * a - 7 * b

theorem evaluate_T : T 6 3 = 3 := by
  sorry

end evaluate_T_l245_245033


namespace sequence_value_l245_245962

theorem sequence_value (x : ℕ) : 
  (5 - 2 = 3) ∧ (11 - 5 = 6) ∧ (20 - 11 = 9) ∧ (x - 20 = 12) → x = 32 := 
by intros; sorry

end sequence_value_l245_245962


namespace value_of_expression_l245_245396

theorem value_of_expression : 7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end value_of_expression_l245_245396


namespace expected_yolks_correct_l245_245701

-- Define the conditions
def total_eggs : ℕ := 15
def double_yolk_eggs : ℕ := 5
def triple_yolk_eggs : ℕ := 3
def single_yolk_eggs : ℕ := total_eggs - double_yolk_eggs - triple_yolk_eggs
def extra_yolk_prob : ℝ := 0.10

-- Define the expected number of yolks calculation
noncomputable def expected_yolks : ℝ :=
  (single_yolk_eggs * 1) + 
  (double_yolk_eggs * 2) + 
  (triple_yolk_eggs * 3) + 
  (double_yolk_eggs * extra_yolk_prob) + 
  (triple_yolk_eggs * extra_yolk_prob)

-- State that the expected number of total yolks is 26.8
theorem expected_yolks_correct : expected_yolks = 26.8 := by
  -- solution would go here
  sorry

end expected_yolks_correct_l245_245701


namespace calculate_expression_l245_245720

theorem calculate_expression :
  (50 - (2050 - 250)) + (2050 - (250 - 50)) = 100 := by
  sorry

end calculate_expression_l245_245720


namespace school_year_hours_per_week_l245_245432

-- Definitions based on the conditions of the problem
def summer_weeks : ℕ := 8
def summer_hours_per_week : ℕ := 40
def summer_earnings : ℕ := 3200

def school_year_weeks : ℕ := 24
def needed_school_year_earnings : ℕ := 6400

-- Question translated to a Lean statement
theorem school_year_hours_per_week :
  let hourly_rate := summer_earnings / (summer_hours_per_week * summer_weeks)
  let total_school_year_hours := needed_school_year_earnings / hourly_rate
  total_school_year_hours / school_year_weeks = (80 / 3) :=
by {
  -- The implementation of the proof goes here
  sorry
}

end school_year_hours_per_week_l245_245432


namespace first_part_is_13_l245_245010

-- Definitions for the conditions
variables (x y : ℕ)

-- Conditions given in the problem
def condition1 : Prop := x + y = 24
def condition2 : Prop := 7 * x + 5 * y = 146

-- The theorem we need to prove
theorem first_part_is_13 (h1 : condition1 x y) (h2 : condition2 x y) : x = 13 :=
sorry

end first_part_is_13_l245_245010


namespace arnaldo_bernaldo_distribute_toys_l245_245255

noncomputable def num_ways_toys_distributed (total_toys remaining_toys : ℕ) : ℕ :=
  if total_toys = 10 ∧ remaining_toys = 8 then 6561 - 256 else 0

theorem arnaldo_bernaldo_distribute_toys : num_ways_toys_distributed 10 8 = 6305 :=
by 
  -- Lean calculation for 3^8 = 6561 and 2^8 = 256 can be done as follows
  -- let three_power_eight := 3^8
  -- let two_power_eight := 2^8
  -- three_power_eight - two_power_eight = 6305
  sorry

end arnaldo_bernaldo_distribute_toys_l245_245255


namespace blueberry_jelly_amount_l245_245647

-- Definition of the conditions
def total_jelly : ℕ := 6310
def strawberry_jelly : ℕ := 1792

-- Formal statement of the problem
theorem blueberry_jelly_amount : 
  total_jelly - strawberry_jelly = 4518 :=
by
  sorry

end blueberry_jelly_amount_l245_245647


namespace mass_of_man_l245_245281

theorem mass_of_man (L B h ρ V m: ℝ) (boat_length: L = 3) (boat_breadth: B = 2) 
  (boat_sink_depth: h = 0.01) (water_density: ρ = 1000) 
  (displaced_volume: V = L * B * h) (displaced_mass: m = ρ * V): m = 60 := 
by 
  sorry

end mass_of_man_l245_245281


namespace gcd_60_75_l245_245144

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end gcd_60_75_l245_245144


namespace find_m_l245_245223

theorem find_m (x y m : ℤ) 
  (h1 : 4 * x + y = 34)
  (h2 : m * x - y = 20)
  (h3 : y ^ 2 = 4) 
  : m = 2 :=
sorry

end find_m_l245_245223


namespace productivity_comparison_l245_245846

def productivity (work_time cycle_time : ℕ) : ℚ := work_time / cycle_time

theorem productivity_comparison:
  ∀ (D : ℕ),
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  (productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm) = 
  productivity ((girl1_work_time * 21) / 20) lcm → 
  1.05 = 1 + (5 / 100) :=
by
  let girl1_work_time := 5
  let girl1_cycle_time := 6
  let girl2_work_time := 7
  let girl2_cycle_time := 8
  let lcm := Nat.lcm 6 8
  
  -- Define the productivity rates for each girl
  let productivity1 := productivity (girl1_work_time * (lcm / girl1_cycle_time)) lcm
  let productivity2 := productivity (girl2_work_time * (lcm / girl2_cycle_time)) lcm
  
  have h1 : productivity1 = (D / 20) := sorry
  have h2 : productivity2 = (D / 21) := sorry
  have productivity_ratio : productivity1 / productivity2 = 21 / 20 := sorry
  have productivity_diff : productivity_ratio = 1.05 := sorry
  
  exact this sorry

end productivity_comparison_l245_245846


namespace roots_quadratic_l245_245185

open Real

theorem roots_quadratic (a : ℤ) :
  (∃ (u v : ℤ), u ≠ v ∧ u + v = -a ∧ u * v = 2 * a) ↔ (a = -1 ∨ a = 9) :=
by
  sorry

end roots_quadratic_l245_245185


namespace perfect_square_selection_l245_245388

theorem perfect_square_selection :
  let k_range := {k | 1 ≤ k ∧ k ≤ 153}
  let cards := (k_range.map (λ k => 3^k)).union (k_range.map (λ k => 19^k))
  (cards.card = 306) →
  let even_count := 76
  let odd_count := 77
  (Nat.choose even_count 2 + Nat.choose odd_count 2) * 2 + even_count * even_count =
  17328 :=
by 
  intro k_range cards h
  unfold even_count
  unfold odd_count
  sorry

end perfect_square_selection_l245_245388


namespace part1_part2_l245_245328

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Statement 1: If f(x) is an odd function, then a = 1.
theorem part1 (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) → a = 1 :=
sorry

-- Statement 2: If f(x) is defined on [-4, +∞), and for all x in the domain, 
-- f(cos(x) + b + 1/4) ≥ f(sin^2(x) - b - 3), then b ∈ [-1,1].
theorem part2 (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, f a (Real.cos x + b + 1/4) ≥ f a (Real.sin x ^ 2 - b - 3)) ∧
  (∀ x : ℝ, -4 ≤ x) ∧ -4 ≤ a ∧ a = 1 → -1 ≤ b ∧ b ≤ 1 :=
sorry

end part1_part2_l245_245328


namespace sum_of_six_terms_arithmetic_sequence_l245_245387

theorem sum_of_six_terms_arithmetic_sequence (S : ℕ → ℕ)
    (h1 : S 2 = 2)
    (h2 : S 4 = 10) :
    S 6 = 42 :=
by
  sorry

end sum_of_six_terms_arithmetic_sequence_l245_245387


namespace ratio_of_areas_l245_245542

theorem ratio_of_areas (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : r₁ = (Real.sqrt 2) / 4)
  (h₂ : A₁ = π * r₁^2) (h₃ : r₂ = (Real.sqrt 2) * r₁) (h₄ : A₂ = π * r₂^2) :
  A₂ / A₁ = 2 :=
by
  sorry

end ratio_of_areas_l245_245542


namespace max_rectangle_area_l245_245816

variables {a b : ℝ}

theorem max_rectangle_area (h : 2 * a + 2 * b = 60) : a * b ≤ 225 :=
by 
  -- Proof to be filled in
  sorry

end max_rectangle_area_l245_245816


namespace greater_number_is_twelve_l245_245691

theorem greater_number_is_twelve (x : ℕ) (a b : ℕ) 
  (h1 : a = 3 * x) 
  (h2 : b = 4 * x) 
  (h3 : a + b = 21) : 
  max a b = 12 :=
by 
  sorry

end greater_number_is_twelve_l245_245691


namespace a_eq_3_suff_not_nec_l245_245752

theorem a_eq_3_suff_not_nec (a : ℝ) : (a = 3 → a^2 = 9) ∧ (a^2 = 9 → ∃ b : ℝ, b = a ∧ (b = 3 ∨ b = -3)) :=
by
  sorry

end a_eq_3_suff_not_nec_l245_245752


namespace small_circle_area_l245_245556

theorem small_circle_area (r R : ℝ) (n : ℕ)
  (h_n : n = 6)
  (h_area_large : π * R^2 = 120)
  (h_relation : r = R / 2) :
  π * r^2 = 40 :=
by
  sorry

end small_circle_area_l245_245556


namespace distribution_ways_l245_245582

theorem distribution_ways (books students : ℕ) (h_books : books = 6) (h_students : students = 6) :
  ∃ ways : ℕ, ways = 6 * 5^6 ∧ ways = 93750 :=
by
  sorry

end distribution_ways_l245_245582


namespace sin_is_odd_l245_245964

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

theorem sin_is_odd : is_odd_function sin :=
by
  sorry

end sin_is_odd_l245_245964


namespace multiplicative_inverse_correct_l245_245635

def A : ℕ := 123456
def B : ℕ := 654321
def m : ℕ := 1234567
def AB_mod : ℕ := (A * B) % m

def N : ℕ := 513629

theorem multiplicative_inverse_correct (h : AB_mod = 470160) : (470160 * N) % m = 1 := 
by 
  have hN : N = 513629 := rfl
  have hAB : AB_mod = 470160 := h
  sorry

end multiplicative_inverse_correct_l245_245635


namespace minimize_tank_construction_cost_l245_245433

noncomputable def minimum_cost (l w h : ℝ) (P_base P_wall : ℝ) : ℝ :=
  P_base * (l * w) + P_wall * (2 * h * (l + w))

theorem minimize_tank_construction_cost :
  ∃ l w : ℝ, l * w = 9 ∧ l = w ∧
  minimum_cost l w 2 200 150 = 5400 :=
by
  sorry

end minimize_tank_construction_cost_l245_245433


namespace jorge_goals_l245_245631

theorem jorge_goals (g_last g_total g_this : ℕ) (h_last : g_last = 156) (h_total : g_total = 343) :
  g_this = g_total - g_last → g_this = 187 :=
by
  intro h
  rw [h_last, h_total] at h
  apply h

end jorge_goals_l245_245631


namespace scientist_birth_day_is_wednesday_l245_245381

noncomputable def calculate_birth_day : String :=
  let years := 150
  let leap_years := 36
  let regular_years := years - leap_years
  let total_days_backward := regular_years + 2 * leap_years -- days to move back
  let days_mod := total_days_backward % 7
  let day_of_birth := (5 + 7 - days_mod) % 7 -- 5 is for backward days from Monday
  match day_of_birth with
  | 0 => "Monday"
  | 1 => "Sunday"
  | 2 => "Saturday"
  | 3 => "Friday"
  | 4 => "Thursday"
  | 5 => "Wednesday"
  | 6 => "Tuesday"
  | _ => "Error"

theorem scientist_birth_day_is_wednesday :
  calculate_birth_day = "Wednesday" :=
  by
    sorry

end scientist_birth_day_is_wednesday_l245_245381
