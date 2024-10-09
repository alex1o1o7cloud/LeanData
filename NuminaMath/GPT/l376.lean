import Mathlib

namespace graph_intersection_l376_37677

noncomputable def log : ℝ → ℝ := sorry

lemma log_properties (a b : ℝ) (ha : 0 < a) (hb : 0 < b): log (a * b) = log a + log b := sorry

theorem graph_intersection :
  ∃! x : ℝ, 2 * log x = log (2 * x) :=
by
  sorry

end graph_intersection_l376_37677


namespace cannot_form_shape_B_l376_37683

-- Define the given pieces
def pieces : List (List (Nat × Nat)) :=
  [ [(1, 1)],
    [(1, 2)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 3)],
    [(1, 3)] ]

-- Define shape B requirement
def shapeB : List (Nat × Nat) := [(1, 6)]

theorem cannot_form_shape_B :
  ¬ (∃ (combinations : List (List (Nat × Nat))), combinations ⊆ pieces ∧ 
     (List.foldr (λ x acc => acc + x) 0 (combinations.map (List.foldr (λ y acc => acc + (y.1 * y.2)) 0)) = 6)) :=
sorry

end cannot_form_shape_B_l376_37683


namespace geometric_sequence_a8_l376_37631

theorem geometric_sequence_a8 (a : ℕ → ℝ) (q : ℝ) 
  (h₁ : a 3 = 3)
  (h₂ : a 6 = 24)
  (h₃ : ∀ n, a (n + 1) = a n * q) : 
  a 8 = 96 :=
by
  sorry

end geometric_sequence_a8_l376_37631


namespace Katie_has_more_games_than_friends_l376_37606

def katie_new_games : ℕ := 57
def katie_old_games : ℕ := 39
def friends_new_games : ℕ := 34

theorem Katie_has_more_games_than_friends :
  (katie_new_games + katie_old_games) - friends_new_games = 62 := by
  sorry

end Katie_has_more_games_than_friends_l376_37606


namespace rate_of_work_l376_37644

theorem rate_of_work (A : ℝ) (h1: 0 < A) (h_eq : 1 / A + 1 / 6 = 1 / 2) : A = 3 := sorry

end rate_of_work_l376_37644


namespace fraction_of_remaining_birds_left_l376_37660

theorem fraction_of_remaining_birds_left (B : ℕ) (F : ℚ) (hB : B = 60)
  (H : (1/3) * (2/3 : ℚ) * B * (1 - F) = 8) :
  F = 4/5 := 
sorry

end fraction_of_remaining_birds_left_l376_37660


namespace ribbon_tape_length_l376_37636

theorem ribbon_tape_length
  (one_ribbon: ℝ)
  (remaining_cm: ℝ)
  (num_ribbons: ℕ)
  (total_used: ℝ)
  (remaining_meters: remaining_cm = 0.50)
  (ribbon_meter: one_ribbon = 0.84)
  (ribbons_made: num_ribbons = 10)
  (used_len: total_used = one_ribbon * num_ribbons):
  total_used + 0.50 = 8.9 :=
by
  sorry

end ribbon_tape_length_l376_37636


namespace max_area_of_garden_l376_37624

theorem max_area_of_garden (l w : ℝ) 
  (h : 2 * l + w = 400) : 
  l * w ≤ 20000 :=
sorry

end max_area_of_garden_l376_37624


namespace ineq_power_sum_lt_pow_two_l376_37607

theorem ineq_power_sum_lt_pow_two (x : ℝ) (n : ℕ) (hx : |x| < 1) (hn : 2 ≤ n) : 
  (1 + x)^n + (1 - x)^n < 2^n :=
by
  sorry

end ineq_power_sum_lt_pow_two_l376_37607


namespace intersection_A_B_l376_37690

-- Definition of set A
def A : Set ℝ := { x | x ≤ 3 }

-- Definition of set B
def B : Set ℝ := {2, 3, 4, 5}

-- Proof statement
theorem intersection_A_B :
  A ∩ B = {2, 3} :=
sorry

end intersection_A_B_l376_37690


namespace sasha_tree_planting_cost_l376_37693

theorem sasha_tree_planting_cost :
  ∀ (initial_temperature final_temperature : ℝ)
    (temp_drop_per_tree : ℝ) (cost_per_tree : ℝ)
    (temperature_drop : ℝ) (num_trees : ℕ)
    (total_cost : ℝ),
    initial_temperature = 80 →
    final_temperature = 78.2 →
    temp_drop_per_tree = 0.1 →
    cost_per_tree = 6 →
    temperature_drop = initial_temperature - final_temperature →
    num_trees = temperature_drop / temp_drop_per_tree →
    total_cost = num_trees * cost_per_tree →
    total_cost = 108 :=
by
  intros initial_temperature final_temperature temp_drop_per_tree
    cost_per_tree temperature_drop num_trees total_cost
    h_initial h_final h_drop_tree h_cost_tree
    h_temp_drop h_num_trees h_total_cost
  rw [h_initial, h_final] at h_temp_drop
  rw [h_temp_drop] at h_num_trees
  rw [h_num_trees] at h_total_cost
  rw [h_drop_tree] at h_total_cost
  rw [h_cost_tree] at h_total_cost
  norm_num at h_total_cost
  exact h_total_cost

end sasha_tree_planting_cost_l376_37693


namespace inequality_proof_l376_37675

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := 
by sorry

end inequality_proof_l376_37675


namespace cost_of_pencils_l376_37645

def cost_of_notebooks : ℝ := 3 * 1.2
def cost_of_pens : ℝ := 1.7
def total_spent : ℝ := 6.8

theorem cost_of_pencils :
  total_spent - (cost_of_notebooks + cost_of_pens) = 1.5 :=
by
  sorry

end cost_of_pencils_l376_37645


namespace cost_price_of_cupboard_l376_37670

theorem cost_price_of_cupboard (C S S_profit : ℝ) (h1 : S = 0.88 * C) (h2 : S_profit = 1.12 * C) (h3 : S_profit - S = 1650) :
  C = 6875 := by
  sorry

end cost_price_of_cupboard_l376_37670


namespace find_a_l376_37647

theorem find_a (x y a : ℕ) (h1 : ((10 : ℕ) ^ ((32 : ℕ) / y)) ^ a - (64 : ℕ) = (279 : ℕ))
                 (h2 : a > 0)
                 (h3 : x * y = 32) :
  a = 1 :=
sorry

end find_a_l376_37647


namespace points_on_line_relation_l376_37685

theorem points_on_line_relation (b y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-2) + b) 
  (h2 : y2 = -3 * (-1) + b) 
  (h3 : y3 = -3 * 1 + b) : 
  y1 > y2 ∧ y2 > y3 :=
sorry

end points_on_line_relation_l376_37685


namespace max_product_not_less_than_993_squared_l376_37604

theorem max_product_not_less_than_993_squared :
  ∀ (a : Fin 1985 → ℕ), 
    (∀ i, ∃ j, a j = i + 1) →  -- representation of permutation
    (∃ i : Fin 1985, i * (a i) ≥ 993 * 993) :=
by
  intros a h
  sorry

end max_product_not_less_than_993_squared_l376_37604


namespace pairs_solution_l376_37699

theorem pairs_solution (x y : ℝ) :
  (4 * x^2 - y^2)^2 + (7 * x + 3 * y - 39)^2 = 0 ↔ (x = 3 ∧ y = 6) ∨ (x = 39 ∧ y = -78) := 
by
  sorry

end pairs_solution_l376_37699


namespace direct_proportion_function_decrease_no_first_quadrant_l376_37627

-- Part (1)
theorem direct_proportion_function (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a ≠ 2 ∧ b = 3 :=
sorry

-- Part (2)
theorem decrease_no_first_quadrant (a b : ℝ) (h : y = (2*a-4)*x + (3-b)) : a < 2 ∧ b ≥ 3 :=
sorry

end direct_proportion_function_decrease_no_first_quadrant_l376_37627


namespace largest_four_digit_number_l376_37694

theorem largest_four_digit_number
  (n : ℕ) (hn1 : n % 8 = 2) (hn2 : n % 7 = 4) (hn3 : 1000 ≤ n) (hn4 : n ≤ 9999) :
  n = 9990 :=
sorry

end largest_four_digit_number_l376_37694


namespace impossible_seed_germinate_without_water_l376_37623

-- Definitions for the conditions
def heats_up_when_conducting (conducts : Bool) : Prop := conducts
def determines_plane (non_collinear : Bool) : Prop := non_collinear
def germinates_without_water (germinates : Bool) : Prop := germinates
def wins_lottery_consecutively (wins_twice : Bool) : Prop := wins_twice

-- The fact that a seed germinates without water is impossible
theorem impossible_seed_germinate_without_water 
  (conducts : Bool) 
  (non_collinear : Bool) 
  (germinates : Bool) 
  (wins_twice : Bool) 
  (h1 : heats_up_when_conducting conducts) 
  (h2 : determines_plane non_collinear) 
  (h3 : ¬germinates_without_water germinates) 
  (h4 : wins_lottery_consecutively wins_twice) :
  ¬germinates_without_water true :=
sorry

end impossible_seed_germinate_without_water_l376_37623


namespace sufficient_but_not_necessary_l376_37696

variable {a : ℝ}

theorem sufficient_but_not_necessary (h : a > 1) : a^2 > a :=
by
  sorry

end sufficient_but_not_necessary_l376_37696


namespace solve_for_q_l376_37656

theorem solve_for_q (x y q : ℚ) 
  (h1 : 7 / 8 = x / 96) 
  (h2 : 7 / 8 = (x + y) / 104) 
  (h3 : 7 / 8 = (q - y) / 144) : 
  q = 133 := 
sorry

end solve_for_q_l376_37656


namespace closest_fraction_to_team_alpha_medals_l376_37682

theorem closest_fraction_to_team_alpha_medals :
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 5) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 6) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 7) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 9) := 
by
  sorry

end closest_fraction_to_team_alpha_medals_l376_37682


namespace rectangle_length_increase_l376_37629

variable (L B : ℝ) -- Original length and breadth
variable (A : ℝ) -- Original area
variable (p : ℝ) -- Percentage increase in length
variable (A' : ℝ) -- New area

theorem rectangle_length_increase (hA : A = L * B) 
  (hp : L' = L + (p / 100) * L) 
  (hB' : B' = B * 0.9) 
  (hA' : A' = 1.035 * A)
  (hl' : L' = (1 + (p / 100)) * L)
  (hb_length : L' * B' = A') :
  p = 15 :=
by
  sorry

end rectangle_length_increase_l376_37629


namespace fair_total_revenue_l376_37603

noncomputable def price_per_ticket : ℝ := 8
noncomputable def total_ticket_revenue : ℝ := 8000
noncomputable def total_tickets_sold : ℝ := total_ticket_revenue / price_per_ticket

noncomputable def food_revenue : ℝ := (3/5) * total_tickets_sold * 10
noncomputable def rounded_ride_revenue : ℝ := (333 : ℝ) * 6
noncomputable def ride_revenue : ℝ := rounded_ride_revenue
noncomputable def rounded_souvenir_revenue : ℝ := (166 : ℝ) * 18
noncomputable def souvenir_revenue : ℝ := rounded_souvenir_revenue
noncomputable def game_revenue : ℝ := (1/10) * total_tickets_sold * 5

noncomputable def total_additional_revenue : ℝ := food_revenue + ride_revenue + souvenir_revenue + game_revenue
noncomputable def total_revenue : ℝ := total_ticket_revenue + total_additional_revenue

theorem fair_total_revenue : total_revenue = 19486 := by
  sorry

end fair_total_revenue_l376_37603


namespace solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l376_37638

-- Definitions only directly appearing in the conditions problem
def consecutive_integers (x y z : ℤ) : Prop := x = y - 1 ∧ z = y + 1
def consecutive_even_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 0
def consecutive_odd_integers (x y z : ℤ) : Prop := x = y - 2 ∧ z = y + 2 ∧ y % 2 = 1

-- Problem Statements
theorem solvable_consecutive_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_integers x y z :=
sorry

theorem solvable_consecutive_even_integers : ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_even_integers x y z :=
sorry

theorem not_solvable_consecutive_odd_integers : ¬ ∃ (x y z : ℤ), x + y + z = 72 ∧ consecutive_odd_integers x y z :=
sorry

end solvable_consecutive_integers_solvable_consecutive_even_integers_not_solvable_consecutive_odd_integers_l376_37638


namespace amoeba_growth_one_week_l376_37628

theorem amoeba_growth_one_week :
  (3 ^ 7 = 2187) :=
by
  sorry

end amoeba_growth_one_week_l376_37628


namespace range_of_2alpha_minus_beta_over_3_l376_37621

theorem range_of_2alpha_minus_beta_over_3 (α β : ℝ) (hα : 0 < α) (hα' : α < π / 2) (hβ : 0 < β) (hβ' : β < π / 2) : 
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := 
sorry

end range_of_2alpha_minus_beta_over_3_l376_37621


namespace smallest_k_base_representation_l376_37634

theorem smallest_k_base_representation :
  ∃ k : ℕ, (k > 0) ∧ (∀ n k, 0 = (42 * (1 - k^(n+1))/(1 - k))) ∧ (0 = (4 * (53 * (1 - k^(n+1))/(1 - k)))) →
  (k = 11) := sorry

end smallest_k_base_representation_l376_37634


namespace smallest_repeating_block_7_over_13_l376_37673

theorem smallest_repeating_block_7_over_13 : 
  ∃ n : ℕ, (∀ d : ℕ, d < n → 
  (∃ (q r : ℕ), r < 13 ∧ 10 ^ (d + 1) * 7 % 13 = q * 10 ^ n + r)) ∧ n = 6 := sorry

end smallest_repeating_block_7_over_13_l376_37673


namespace competition_results_correct_l376_37637

theorem competition_results_correct :
  ∃ (first second third fourth : String), 
    (first = "Oleg" ∧ second = "Olya" ∧ third = "Polya" ∧ fourth = "Pasha") ∧
    ∀ (claims : String → String → Prop),
      (claims "Olya" "all_odd_places_boys") ∧ 
      (claims "Oleg" "consecutive_places_with_olya") ∧
      (claims "Pasha" "all_odd_places_names_start_O") ∧
      ∃ (truth_teller : String), 
        truth_teller = "Oleg" ∧ 
        (claims "Oleg" "first_place") ∧ 
        ¬ (claims "Olya" "first_place") ∧ 
        ¬ (claims "Pasha" "first_place") ∧ 
        ¬ (claims "Polya" "first_place") :=
sorry

end competition_results_correct_l376_37637


namespace meaningful_fraction_x_range_l376_37698

theorem meaningful_fraction_x_range (x : ℝ) : (x-2 ≠ 0) ↔ (x ≠ 2) :=
by 
  sorry

end meaningful_fraction_x_range_l376_37698


namespace product_divisibility_l376_37616

theorem product_divisibility (a b c : ℤ)
  (h₁ : (a + b + c) ^ 2 = -(a * b + a * c + b * c))
  (h₂ : a + b ≠ 0)
  (h₃ : b + c ≠ 0)
  (h₄ : a + c ≠ 0) :
  (a + b) * (a + c) % (b + c) = 0 ∧
  (a + b) * (b + c) % (a + c) = 0 ∧
  (a + c) * (b + c) % (a + b) = 0 := by
  sorry

end product_divisibility_l376_37616


namespace rice_in_each_container_l376_37610

theorem rice_in_each_container 
  (total_weight : ℚ) 
  (num_containers : ℕ)
  (conversion_factor : ℚ) 
  (equal_division : total_weight = 29 / 4 ∧ num_containers = 4 ∧ conversion_factor = 16) : 
  (total_weight / num_containers) * conversion_factor = 29 := 
by 
  sorry

end rice_in_each_container_l376_37610


namespace train_crosses_in_26_seconds_l376_37691

def speed_km_per_hr := 72
def length_of_train := 250
def length_of_platform := 270

def total_distance := length_of_train + length_of_platform

noncomputable def speed_m_per_s := (speed_km_per_hr * 1000 / 3600)  -- Convert km/hr to m/s

noncomputable def time_to_cross := total_distance / speed_m_per_s

theorem train_crosses_in_26_seconds :
  time_to_cross = 26 := 
sorry

end train_crosses_in_26_seconds_l376_37691


namespace imaginary_part_of_conjugate_l376_37668

def z : Complex := Complex.mk 1 2

def z_conj : Complex := Complex.mk 1 (-2)

theorem imaginary_part_of_conjugate :
  z_conj.im = -2 := by
  sorry

end imaginary_part_of_conjugate_l376_37668


namespace erased_number_is_30_l376_37661

-- Definitions based on conditions
def consecutiveNumbers (start n : ℕ) : List ℕ :=
  List.range' start n

def erase (l : List ℕ) (x : ℕ) : List ℕ :=
  List.filter (λ y => y ≠ x) l

def average (l : List ℕ) : ℚ :=
  l.sum / l.length

-- Statement to prove
theorem erased_number_is_30 :
  ∃ n x, average (erase (consecutiveNumbers 11 n) x) = 23 ∧ x = 30 := by
  sorry

end erased_number_is_30_l376_37661


namespace mean_of_two_means_eq_l376_37648

theorem mean_of_two_means_eq (z : ℚ) (h : (5 + 10 + 20) / 3 = (15 + z) / 2) : z = 25 / 3 :=
by
  sorry

end mean_of_two_means_eq_l376_37648


namespace find_A_and_B_l376_37622

theorem find_A_and_B (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ -6 → (5 * x - 3) / (x^2 + 3 * x - 18) = A / (x - 3) + B / (x + 6)) →
  A = 4 / 3 ∧ B = 11 / 3 :=
by
  intros h
  sorry

end find_A_and_B_l376_37622


namespace colin_speed_l376_37641

variable (B T Bn C : ℝ)
variable (m : ℝ)

-- Given conditions
def condition1 := B = 1
def condition2 := T = m * B
def condition3 := Bn = T / 3
def condition4 := C = 6 * Bn
def condition5 := C = 4

-- We need to prove C = 4 given the conditions
theorem colin_speed :
  (B = 1) →
  (T = m * B) →
  (Bn = T / 3) →
  (C = 6 * Bn) →
  C = 4 :=
by
  intros _ _ _ _
  sorry

end colin_speed_l376_37641


namespace vertical_asymptotes_sum_l376_37619

theorem vertical_asymptotes_sum : 
  (∀ x : ℝ, 4 * x^2 + 7 * x + 3 = 0 → x = -3 / 4 ∨ x = -1) →
  (-3 / 4) + (-1) = -7 / 4 :=
by
  intro h
  sorry

end vertical_asymptotes_sum_l376_37619


namespace leak_empties_tank_in_24_hours_l376_37617

theorem leak_empties_tank_in_24_hours (A L : ℝ) (hA : A = 1 / 8) (h_comb : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- Proof will be here
  sorry

end leak_empties_tank_in_24_hours_l376_37617


namespace line_CD_area_triangle_equality_line_CD_midpoint_l376_37649

theorem line_CD_area_triangle_equality :
  ∃ k : ℝ, 4 * k - 1 = 1 - k := sorry

theorem line_CD_midpoint :
  ∃ k : ℝ, 9 * k - 2 = 1 := sorry

end line_CD_area_triangle_equality_line_CD_midpoint_l376_37649


namespace percentage_needed_to_pass_l376_37646

-- Definitions for conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def total_marks : ℕ := 500
def passing_marks := obtained_marks + failed_by

-- Assertion to prove
theorem percentage_needed_to_pass : (passing_marks : ℕ) * 100 / total_marks = 33 := by
  sorry

end percentage_needed_to_pass_l376_37646


namespace union_of_P_and_Q_l376_37687

noncomputable def P : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_of_P_and_Q :
  P ∪ Q = {x | -1 < x ∧ x < 2} :=
sorry

end union_of_P_and_Q_l376_37687


namespace race_winner_laps_l376_37613

/-- Given:
  * A lap equals 100 meters.
  * Award per hundred meters is $3.5.
  * The winner earned $7 per minute.
  * The race lasted 12 minutes.
  Prove that the number of laps run by the winner is 24.
-/ 
theorem race_winner_laps :
  let lap_distance := 100 -- meters
  let award_per_100meters := 3.5 -- dollars per 100 meters
  let earnings_per_minute := 7 -- dollars per minute
  let race_duration := 12 -- minutes
  let total_earnings := earnings_per_minute * race_duration
  let total_100meters := total_earnings / award_per_100meters
  let laps := total_100meters
  laps = 24 := by
  sorry

end race_winner_laps_l376_37613


namespace c_share_of_profit_l376_37671

theorem c_share_of_profit (a b c total_profit : ℕ) 
  (h₁ : a = 5000) (h₂ : b = 8000) (h₃ : c = 9000) (h₄ : total_profit = 88000) :
  c * total_profit / (a + b + c) = 36000 :=
by
  sorry

end c_share_of_profit_l376_37671


namespace pipe_empty_cistern_l376_37609

theorem pipe_empty_cistern (h : 1 / 3 * t = 6) : 2 / 3 * t = 12 :=
sorry

end pipe_empty_cistern_l376_37609


namespace train_length_correct_l376_37669

noncomputable def train_length (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * time

theorem train_length_correct :
  train_length 17.998560115190784 36 = 179.98560115190784 :=
by
  sorry

end train_length_correct_l376_37669


namespace find_x_values_l376_37651

noncomputable def condition (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4

theorem find_x_values : 
  {x : ℝ | condition  x} = {x : ℝ | x < -3} ∪ {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end find_x_values_l376_37651


namespace mary_characters_initial_D_l376_37614

theorem mary_characters_initial_D (total_characters initial_A initial_C initial_D initial_E : ℕ)
  (h1 : total_characters = 60)
  (h2 : initial_A = total_characters / 2)
  (h3 : initial_C = initial_A / 2)
  (remaining := total_characters - initial_A - initial_C)
  (h4 : remaining = initial_D + initial_E)
  (h5 : initial_D = 2 * initial_E) : initial_D = 10 := by
  sorry

end mary_characters_initial_D_l376_37614


namespace vinegar_ratio_to_total_capacity_l376_37665

theorem vinegar_ratio_to_total_capacity (bowl_capacity : ℝ) (oil_fraction : ℝ) 
  (oil_density : ℝ) (vinegar_density : ℝ) (total_weight : ℝ) :
  bowl_capacity = 150 ∧ oil_fraction = 2/3 ∧ oil_density = 5 ∧ vinegar_density = 4 ∧ total_weight = 700 →
  (total_weight - (bowl_capacity * oil_fraction * oil_density)) / vinegar_density / bowl_capacity = 1/3 :=
by
  sorry

end vinegar_ratio_to_total_capacity_l376_37665


namespace total_questions_attempted_l376_37663

/-- 
In an examination, a student scores 3 marks for every correct answer and loses 1 mark for
every wrong answer. He attempts some questions and secures 180 marks. The number of questions
he attempts correctly is 75. Prove that the total number of questions he attempts is 120. 
-/
theorem total_questions_attempted
  (marks_per_correct : ℕ := 3)
  (marks_lost_per_wrong : ℕ := 1)
  (total_marks : ℕ := 180)
  (correct_answers : ℕ := 75) :
  ∃ (wrong_answers total_questions : ℕ), 
    total_marks = (marks_per_correct * correct_answers) - (marks_lost_per_wrong * wrong_answers) ∧
    total_questions = correct_answers + wrong_answers ∧
    total_questions = 120 := 
by {
  sorry -- proof omitted
}

end total_questions_attempted_l376_37663


namespace decode_division_problem_l376_37680

theorem decode_division_problem :
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  dividend / divisor = quotient :=
by {
  -- Definitions of given and derived values
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  -- The statement to prove
  sorry
}

end decode_division_problem_l376_37680


namespace isosceles_triangle_base_length_l376_37672

theorem isosceles_triangle_base_length
  (perimeter_eq_triangle : ℕ)
  (perimeter_isosceles_triangle : ℕ)
  (side_eq_triangle_isosceles : ℕ)
  (side_eq : side_eq_triangle_isosceles = perimeter_eq_triangle / 3)
  (perimeter_eq : perimeter_isosceles_triangle = 2 * side_eq_triangle_isosceles + 15) :
  15 = perimeter_isosceles_triangle - 2 * side_eq_triangle_isosceles :=
sorry

end isosceles_triangle_base_length_l376_37672


namespace max_square_test_plots_l376_37633

theorem max_square_test_plots (length width fence : ℕ)
  (h_length : length = 36)
  (h_width : width = 66)
  (h_fence : fence = 2200) :
  ∃ (n : ℕ), n * (11 / 6) * n = 264 ∧
      (36 * n + (11 * n - 6) * 66) ≤ 2200 := sorry

end max_square_test_plots_l376_37633


namespace lines_intersection_l376_37676

theorem lines_intersection :
  ∃ (t u : ℚ), 
    (∃ (x y : ℚ),
    (x = 2 - t ∧ y = 3 + 4 * t) ∧ 
    (x = -1 + 3 * u ∧ y = 6 + 5 * u) ∧ 
    (x = 28 / 17 ∧ y = 75 / 17)) := sorry

end lines_intersection_l376_37676


namespace watch_cost_l376_37674

-- Definitions based on conditions
def initial_money : ℤ := 1
def money_from_david : ℤ := 12
def money_needed : ℤ := 7

-- Indicating the total money Evan has after receiving money from David
def total_money := initial_money + money_from_david

-- The cost of the watch based on total money Evan has and additional money needed
def cost_of_watch := total_money + money_needed

-- Proving the cost of the watch
theorem watch_cost : cost_of_watch = 20 := by
  -- We are skipping the proof steps here
  sorry

end watch_cost_l376_37674


namespace minimum_value_of_reciprocals_l376_37681

theorem minimum_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : a - b = 1) :
  (1 / a) + (1 / b) ≥ 4 :=
sorry

end minimum_value_of_reciprocals_l376_37681


namespace trig_identity_example_l376_37678

theorem trig_identity_example :
  (2 * (Real.sin (Real.pi / 6)) - Real.tan (Real.pi / 4)) = 0 :=
by
  -- Definitions from conditions
  have h1 : Real.sin (Real.pi / 6) = 1/2 := Real.sin_pi_div_six
  have h2 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  rw [h1, h2]
  sorry -- The proof is omitted as per instructions

end trig_identity_example_l376_37678


namespace negation_of_universal_to_existential_l376_37608

theorem negation_of_universal_to_existential :
  (¬(∀ x : ℝ, x^2 > 0)) ↔ (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end negation_of_universal_to_existential_l376_37608


namespace max_value_of_function_l376_37640

theorem max_value_of_function : 
  ∀ (x : ℝ), 0 ≤ x → x ≤ 1 → (3 * x - 4 * x^3) ≤ 1 :=
by
  intro x hx0 hx1
  -- proof goes here
  sorry

end max_value_of_function_l376_37640


namespace value_of_expression_l376_37642

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end value_of_expression_l376_37642


namespace find_first_set_length_l376_37655

def length_of_second_set : ℤ := 20
def ratio := 5

theorem find_first_set_length (x : ℤ) (h1 : length_of_second_set = ratio * x) : x = 4 := 
sorry

end find_first_set_length_l376_37655


namespace sin_double_angle_shift_l376_37601

variable (θ : Real)

theorem sin_double_angle_shift (h : Real.cos (θ + Real.pi) = -1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by 
  sorry

end sin_double_angle_shift_l376_37601


namespace op_plus_18_plus_l376_37692

def op_plus (y: ℝ) : ℝ := 9 - y
def plus_op (y: ℝ) : ℝ := y - 9

theorem op_plus_18_plus :
  plus_op (op_plus 18) = -18 := by
  sorry

end op_plus_18_plus_l376_37692


namespace stratified_sampling_number_l376_37654

noncomputable def students_in_grade_10 : ℕ := 150
noncomputable def students_in_grade_11 : ℕ := 180
noncomputable def students_in_grade_12 : ℕ := 210
noncomputable def total_students : ℕ := students_in_grade_10 + students_in_grade_11 + students_in_grade_12
noncomputable def sample_size : ℕ := 72
noncomputable def selection_probability : ℚ := sample_size / total_students
noncomputable def combined_students_grade_10_11 : ℕ := students_in_grade_10 + students_in_grade_11

theorem stratified_sampling_number :
  combined_students_grade_10_11 * selection_probability = 44 := 
by
  sorry

end stratified_sampling_number_l376_37654


namespace quadratic_no_real_solutions_l376_37697

theorem quadratic_no_real_solutions (k : ℝ) :
  k < -9 / 4 ↔ ∀ x : ℝ, ¬ (x^2 - 3 * x - k = 0) :=
by
  sorry

end quadratic_no_real_solutions_l376_37697


namespace flowers_total_l376_37679

theorem flowers_total (yoojung_flowers : ℕ) (namjoon_flowers : ℕ)
 (h1 : yoojung_flowers = 32)
 (h2 : yoojung_flowers = 4 * namjoon_flowers) :
  yoojung_flowers + namjoon_flowers = 40 := by
  sorry

end flowers_total_l376_37679


namespace xyz_value_l376_37667

-- Define real numbers x, y, z
variables {x y z : ℝ}

-- Define the theorem with the given conditions and conclusion
theorem xyz_value 
  (h1 : (x + y + z) * (xy + xz + yz) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := 
sorry

end xyz_value_l376_37667


namespace total_animals_to_spay_l376_37605

theorem total_animals_to_spay : 
  ∀ (c d : ℕ), c = 7 → d = 2 * c → c + d = 21 :=
by
  intros c d h1 h2
  sorry

end total_animals_to_spay_l376_37605


namespace max_distance_bicycle_l376_37688

theorem max_distance_bicycle (front_tire_last : ℕ) (rear_tire_last : ℕ) :
  front_tire_last = 5000 ∧ rear_tire_last = 3000 →
  ∃ (max_distance : ℕ), max_distance = 3750 :=
by
  sorry

end max_distance_bicycle_l376_37688


namespace compare_neg_rational_decimal_l376_37626

theorem compare_neg_rational_decimal : 
  -3 / 4 > -0.8 := 
by 
  sorry

end compare_neg_rational_decimal_l376_37626


namespace john_gets_30_cans_l376_37612

def normal_price : ℝ := 0.60
def total_paid : ℝ := 9.00

theorem john_gets_30_cans :
  (total_paid / normal_price) * 2 = 30 :=
by
  sorry

end john_gets_30_cans_l376_37612


namespace sum_of_n_values_l376_37662

theorem sum_of_n_values : ∃ n1 n2 : ℚ, (abs (3 * n1 - 4) = 5) ∧ (abs (3 * n2 - 4) = 5) ∧ n1 + n2 = 8 / 3 :=
by
  sorry

end sum_of_n_values_l376_37662


namespace number_of_valid_m_l376_37657

def is_right_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  (Qx - Px) * (Qx - Px) + (Qy - Py) * (Qy - Py) + (Rx - Qx) * (Rx - Qx) + (Ry - Qy) * (Ry - Qy) ==
  (Px - Rx) * (Px - Rx) + (Py - Ry) * (Py - Ry) + 2 * ((Qx - Px) * (Rx - Qx) + (Qy - Py) * (Ry - Qy))

def legs_parallel_to_axes (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  Px = Qx ∨ Px = Rx ∨ Qx = Rx ∧ Py = Qy ∨ Py = Ry ∨ Qy = Ry

def medians_condition (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  let M_PQ := ((Px + Qx) / 2, (Py + Qy) / 2);
  let M_PR := ((Px + Rx) / 2, (Py + Ry) / 2);
  (M_PQ.2 = 3 * M_PQ.1 + 1) ∧ (M_PR.2 = 2)

theorem number_of_valid_m (a b c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (P := (a, b)) (Q := (a, b+2*c)) (R := (a-2*d, b)) :
  is_right_triangle P Q R →
  legs_parallel_to_axes P Q R →
  medians_condition P Q R →
  ∃ m, m = 1 :=
sorry

end number_of_valid_m_l376_37657


namespace largest_visits_is_four_l376_37689

noncomputable def largest_num_visits (stores people visits : ℕ) (eight_people_two_stores : ℕ) 
  (one_person_min : ℕ) : ℕ := 4 -- This represents the largest number of stores anyone could have visited.

theorem largest_visits_is_four 
  (stores : ℕ) (total_visits : ℕ) (people_shopping : ℕ) 
  (eight_people_two_stores : ℕ) (each_one_store : ℕ) 
  (H1 : stores = 8) 
  (H2 : total_visits = 23) 
  (H3 : people_shopping = 12) 
  (H4 : eight_people_two_stores = 8)
  (H5 : each_one_store = 1) :
  largest_num_visits stores people_shopping total_visits eight_people_two_stores each_one_store = 4 :=
by
  sorry

end largest_visits_is_four_l376_37689


namespace solve_f_l376_37620

open Nat

theorem solve_f (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) + f n = 2 * n + 3) : f 1993 = 1994 := by
  -- assumptions and required proof
  sorry

end solve_f_l376_37620


namespace phillip_remaining_amount_l376_37602

-- Define the initial amount of money
def initial_amount : ℕ := 95

-- Define the amounts spent on various items
def amount_spent_on_oranges : ℕ := 14
def amount_spent_on_apples : ℕ := 25
def amount_spent_on_candy : ℕ := 6

-- Calculate the total amount spent
def total_spent : ℕ := amount_spent_on_oranges + amount_spent_on_apples + amount_spent_on_candy

-- Calculate the remaining amount of money
def remaining_amount : ℕ := initial_amount - total_spent

-- Statement to be proved
theorem phillip_remaining_amount : remaining_amount = 50 :=
by
  sorry

end phillip_remaining_amount_l376_37602


namespace amplitude_of_cosine_wave_l376_37639

theorem amplitude_of_cosine_wave 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_max_min : ∀ x : ℝ, d + a = 5 ∧ d - a = 1) 
  : a = 2 :=
by
  sorry

end amplitude_of_cosine_wave_l376_37639


namespace min_value_2x_plus_y_l376_37630

theorem min_value_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 2 / (y + 1) = 2) :
  2 * x + y = 3 :=
sorry

end min_value_2x_plus_y_l376_37630


namespace least_tablets_l376_37618

theorem least_tablets (num_A num_B : ℕ) (hA : num_A = 10) (hB : num_B = 14) :
  ∃ n, n = 12 ∧
  ∀ extracted_tablets, extracted_tablets > 0 →
    (∃ (a b : ℕ), a + b = extracted_tablets ∧ a ≥ 2 ∧ b ≥ 2) :=
by
  sorry

end least_tablets_l376_37618


namespace largest_value_WY_cyclic_quadrilateral_l376_37650

theorem largest_value_WY_cyclic_quadrilateral :
  ∃ WZ ZX ZY YW : ℕ, 
    WZ ≠ ZX ∧ WZ ≠ ZY ∧ WZ ≠ YW ∧ ZX ≠ ZY ∧ ZX ≠ YW ∧ ZY ≠ YW ∧ 
    WZ < 20 ∧ ZX < 20 ∧ ZY < 20 ∧ YW < 20 ∧ 
    WZ * ZY = ZX * YW ∧
    (∀ WY', (∃ WY : ℕ, WY' < WY → WY <= 19 )) :=
sorry

end largest_value_WY_cyclic_quadrilateral_l376_37650


namespace left_handed_rock_lovers_l376_37632

def total_people := 30
def left_handed := 12
def like_rock_music := 20
def right_handed_dislike_rock := 3

theorem left_handed_rock_lovers : ∃ x, x + (left_handed - x) + (like_rock_music - x) + right_handed_dislike_rock = total_people ∧ x = 5 :=
by
  sorry

end left_handed_rock_lovers_l376_37632


namespace part1_part2_l376_37652

variable (x : ℝ)
def A : ℝ := 2 * x^2 - 3 * x + 2
def B : ℝ := x^2 - 3 * x - 2

theorem part1 : A x - B x = x^2 + 4 := sorry

theorem part2 (h : x = -2) : A x - B x = 8 := sorry

end part1_part2_l376_37652


namespace total_tickets_sold_l376_37643

-- Define the conditions
variables (V G : ℕ)

-- Condition 1: Total revenue from VIP and general admission
def total_revenue_eq : Prop := 40 * V + 15 * G = 7500

-- Condition 2: There are 212 fewer VIP tickets than general admission
def vip_tickets_eq : Prop := V = G - 212

-- Main statement to prove: the total number of tickets sold
theorem total_tickets_sold (h1 : total_revenue_eq V G) (h2 : vip_tickets_eq V G) : V + G = 370 :=
sorry

end total_tickets_sold_l376_37643


namespace line_passes_through_fixed_point_l376_37666

-- Define the condition that represents the family of lines
def family_of_lines (k : ℝ) (x y : ℝ) : Prop := k * x + y + 2 * k + 1 = 0

-- Formulate the theorem stating that (-2, -1) always lies on the line
theorem line_passes_through_fixed_point (k : ℝ) : family_of_lines k (-2) (-1) :=
by
  -- Proof skipped with sorry.
  sorry

end line_passes_through_fixed_point_l376_37666


namespace find_cost_of_chocolate_l376_37611

theorem find_cost_of_chocolate
  (C : ℕ)
  (h1 : 5 * C + 10 = 90 - 55)
  (h2 : 5 * 2 = 10)
  (h3 : 55 = 90 - (5 * C + 10)):
  C = 5 :=
by
  sorry

end find_cost_of_chocolate_l376_37611


namespace best_fitting_model_l376_37658

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.25) 
  (h2 : R2_2 = 0.50) 
  (h3 : R2_3 = 0.80) 
  (h4 : R2_4 = 0.98) : 
  (R2_4 = max (max R2_1 (max R2_2 R2_3)) R2_4) :=
by
  sorry

end best_fitting_model_l376_37658


namespace division_result_l376_37615

-- Definitions for the values used in the problem
def numerator := 0.0048 * 3.5
def denominator := 0.05 * 0.1 * 0.004

-- Theorem statement
theorem division_result : numerator / denominator = 840 := by 
  sorry

end division_result_l376_37615


namespace analytic_expression_and_symmetry_l376_37684

noncomputable def f (A : ℝ) (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem analytic_expression_and_symmetry {A ω φ : ℝ}
  (hA : A > 0) 
  (hω : ω > 0)
  (h_period : ∀ x, f A ω φ (x + 2) = f A ω φ x)
  (h_max : f A ω φ (1 / 3) = 2) :
  (f 2 π (π / 6) = fun x => 2 * Real.sin (π * x + π / 6)) ∧
  (∃ k : ℤ, k = 5 ∧ (1 / 3 + k = 16 / 3) ∧ (21 / 4 ≤ 1 / 3 + ↑k) ∧ (1 / 3 + ↑k ≤ 23 / 4)) :=
  sorry

end analytic_expression_and_symmetry_l376_37684


namespace average_annual_growth_rate_equation_l376_37664

variable (x : ℝ)
axiom seventh_to_ninth_reading_increase : (1 : ℝ) * (1 + x) * (1 + x) = 1.21

theorem average_annual_growth_rate_equation :
  100 * (1 + x) ^ 2 = 121 :=
by
  have h : (1 : ℝ) * (1 + x) * (1 + x) = 1.21 := seventh_to_ninth_reading_increase x
  sorry

end average_annual_growth_rate_equation_l376_37664


namespace angle_B_value_value_of_k_l376_37635

variable {A B C a b c : ℝ}
variable {k : ℝ}
variable {m n : ℝ × ℝ}

theorem angle_B_value
  (h1 : (2 * a - c) * Real.cos B = b * Real.cos C) :
  B = Real.pi / 3 :=
by sorry

theorem value_of_k
  (hA : 0 < A ∧ A < 2 * Real.pi / 3)
  (hm : m = (Real.sin A, Real.cos (2 * A)))
  (hn : n = (4 * k, 1))
  (hM : 4 * k * Real.sin A + Real.cos (2 * A) = 7) :
  k = 2 :=
by sorry

end angle_B_value_value_of_k_l376_37635


namespace polygon_sides_sum_l376_37695

theorem polygon_sides_sum :
  let triangle_sides := 3
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  -- The sides of shapes that are adjacent on one side each (triangle and nonagon)
  let adjacent_triangle_nonagon := triangle_sides + nonagon_sides - 2
  -- The sides of the intermediate shapes that are each adjacent on two sides
  let adjacent_other_shapes := square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides - 5 * 2
  -- Summing up all the sides exposed to the outside
  adjacent_triangle_nonagon + adjacent_other_shapes = 30 := by
sorry

end polygon_sides_sum_l376_37695


namespace isosceles_triangle_circles_distance_l376_37686

theorem isosceles_triangle_circles_distance (h α : ℝ) (hα : α ≤ π / 6) :
    let R := h / (2 * (Real.cos α)^2)
    let r := h * (Real.tan α) * (Real.tan (π / 4 - α / 2))
    let OO1 := h * (1 - 1 / (2 * (Real.cos α)^2) - (Real.tan α) * (Real.tan (π / 4 - α / 2)))
    OO1 = (2 * h * Real.sin (π / 12 - α / 2) * Real.cos (π / 12 + α / 2)) / (Real.cos α)^2 :=
    sorry

end isosceles_triangle_circles_distance_l376_37686


namespace find_day_for_balance_l376_37659

-- Define the initial conditions and variables
def initialEarnings : ℤ := 20
def secondDaySpending : ℤ := 15
variables (X Y : ℤ)

-- Define the function for net balance on day D
def netBalance (D : ℤ) : ℤ :=
  initialEarnings + (D - 1) * X - (secondDaySpending + (D - 2) * Y)

-- The main theorem proving the day D for net balance of Rs. 60
theorem find_day_for_balance (X Y : ℤ) : ∃ D : ℤ, netBalance X Y D = 60 → 55 = (D + 1) * (X - Y) :=
by
  sorry

end find_day_for_balance_l376_37659


namespace inequality_not_satisfied_integer_values_count_l376_37625

theorem inequality_not_satisfied_integer_values_count :
  ∃ (n : ℕ), n = 5 ∧ ∀ (x : ℤ), 3 * x^2 + 17 * x + 20 ≤ 25 → x ∈ [-4, -3, -2, -1, 0] :=
  sorry

end inequality_not_satisfied_integer_values_count_l376_37625


namespace negation_of_universal_proposition_l376_37653

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l376_37653


namespace factor_of_polynomial_l376_37600

theorem factor_of_polynomial (t : ℚ) : (8 * t^2 + 17 * t - 10 = 0) ↔ (t = 5/8 ∨ t = -2) :=
by sorry

end factor_of_polynomial_l376_37600
