import Mathlib

namespace find_charge_federal_return_l2081_208199

-- Definitions based on conditions
def charge_federal_return (F : ℝ) : ℝ := F
def charge_state_return : ℝ := 30
def charge_quarterly_return : ℝ := 80
def sold_federal_returns : ℝ := 60
def sold_state_returns : ℝ := 20
def sold_quarterly_returns : ℝ := 10
def total_revenue : ℝ := 4400

-- Lean proof statement to verify the value of F
theorem find_charge_federal_return (F : ℝ) (h : sold_federal_returns * charge_federal_return F + sold_state_returns * charge_state_return + sold_quarterly_returns * charge_quarterly_return = total_revenue) : 
  F = 50 :=
by
  sorry

end find_charge_federal_return_l2081_208199


namespace probability_red_white_red_l2081_208116

-- Definitions and assumptions
def total_marbles := 10
def red_marbles := 4
def white_marbles := 6

def P_first_red : ℚ := red_marbles / total_marbles
def P_second_white_given_first_red : ℚ := white_marbles / (total_marbles - 1)
def P_third_red_given_first_red_and_second_white : ℚ := (red_marbles - 1) / (total_marbles - 2)

-- The target probability hypothesized
theorem probability_red_white_red :
  P_first_red * P_second_white_given_first_red * P_third_red_given_first_red_and_second_white = 1 / 10 :=
by
  sorry

end probability_red_white_red_l2081_208116


namespace seventeen_in_base_three_l2081_208167

theorem seventeen_in_base_three : (17 : ℕ) = 1 * 3^2 + 2 * 3^1 + 2 * 3^0 :=
by
  -- This is the arithmetic representation of the conversion,
  -- proving that 17 in base 10 equals 122 in base 3
  sorry

end seventeen_in_base_three_l2081_208167


namespace order_of_four_l2081_208142

theorem order_of_four {m n p q : ℝ} (hmn : m < n) (hpq : p < q) (h1 : (p - m) * (p - n) < 0) (h2 : (q - m) * (q - n) < 0) : m < p ∧ p < q ∧ q < n :=
by
  sorry

end order_of_four_l2081_208142


namespace solve_inequality_l2081_208151

variable {x : ℝ}

theorem solve_inequality :
  (x - 8) / (x^2 - 4 * x + 13) ≥ 0 ↔ x ≥ 8 :=
by
  sorry

end solve_inequality_l2081_208151


namespace savings_from_discount_l2081_208165

-- Define the initial price
def initial_price : ℝ := 475.00

-- Define the discounted price
def discounted_price : ℝ := 199.00

-- The theorem to prove the savings amount
theorem savings_from_discount : initial_price - discounted_price = 276.00 :=
by 
  -- This is where the actual proof would go
  sorry

end savings_from_discount_l2081_208165


namespace curved_surface_area_of_cone_l2081_208187

noncomputable def slant_height : ℝ := 22
noncomputable def radius : ℝ := 7
noncomputable def pi : ℝ := Real.pi

theorem curved_surface_area_of_cone :
  abs (pi * radius * slant_height - 483.22) < 0.01 := 
by
  sorry

end curved_surface_area_of_cone_l2081_208187


namespace isosceles_triangle_perimeter_l2081_208173

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def roots_of_quadratic_eq := {x : ℕ | x^2 - 5 * x + 6 = 0}

theorem isosceles_triangle_perimeter
  (a b c : ℕ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_roots : (a ∈ roots_of_quadratic_eq) ∧ (b ∈ roots_of_quadratic_eq) ∧ (c ∈ roots_of_quadratic_eq)) :
  (a + b + c = 7 ∨ a + b + c = 8) :=
by
  sorry

end isosceles_triangle_perimeter_l2081_208173


namespace tan_alpha_eq_neg_one_third_l2081_208135

open Real

theorem tan_alpha_eq_neg_one_third
  (h : cos (π / 4 - α) / cos (π / 4 + α) = 1 / 2) :
  tan α = -1 / 3 :=
sorry

end tan_alpha_eq_neg_one_third_l2081_208135


namespace income_calculation_l2081_208131

-- Define the conditions
def ratio (i e : ℕ) : Prop := 9 * e = 8 * i
def savings (i e : ℕ) : Prop := i - e = 4000

-- The theorem statement
theorem income_calculation (i e : ℕ) (h1 : ratio i e) (h2 : savings i e) : i = 36000 := by
  sorry

end income_calculation_l2081_208131


namespace find_p_power_l2081_208169

theorem find_p_power (p : ℕ) (h1 : p % 2 = 0) (h2 : (p + 1) % 10 = 7) : 
  (p % 10)^3 % 10 = (p % 10)^1 % 10 :=
by
  sorry

end find_p_power_l2081_208169


namespace simplify_fraction_l2081_208179

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l2081_208179


namespace sum_sequence_S_n_l2081_208168

variable {S : ℕ+ → ℚ}
noncomputable def S₁ : ℚ := 1 / 2
noncomputable def S₂ : ℚ := 5 / 6
noncomputable def S₃ : ℚ := 49 / 72
noncomputable def S₄ : ℚ := 205 / 288

theorem sum_sequence_S_n (n : ℕ+) :
  (S 1 = S₁) ∧ (S 2 = S₂) ∧ (S 3 = S₃) ∧ (S 4 = S₄) ∧ (∀ n : ℕ+, S n = n / (n + 1)) :=
by
  sorry

end sum_sequence_S_n_l2081_208168


namespace pond_depth_range_l2081_208162

theorem pond_depth_range (d : ℝ) (adam_false : d < 10) (ben_false : d > 8) (carla_false : d ≠ 7) : 
    8 < d ∧ d < 10 :=
by
  sorry

end pond_depth_range_l2081_208162


namespace square_difference_l2081_208139

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end square_difference_l2081_208139


namespace distinct_solutions_equation_l2081_208126

theorem distinct_solutions_equation (a b : ℝ) (h1 : a ≠ b) (h2 : a > b) (h3 : ∀ x, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1) (sol_a : x = a) (sol_b : x = b) :
  a - b = 1 :=
sorry

end distinct_solutions_equation_l2081_208126


namespace D_144_l2081_208196

def D (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry

theorem D_144 : D 144 = 51 := by
  sorry

end D_144_l2081_208196


namespace domain_of_f_2x_minus_3_l2081_208132

noncomputable def f (x : ℝ) := 2 * x + 1

theorem domain_of_f_2x_minus_3 :
  (∀ x, 1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 5 → (2 ≤ x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_2x_minus_3_l2081_208132


namespace find_f1_and_f_prime1_l2081_208150

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_def : ∀ x : ℝ, f x = 2 * x^2 - f' 1 * x - 3

-- Proof using conditions
theorem find_f1_and_f_prime1 : f 1 + (f' 1) = -1 :=
sorry

end find_f1_and_f_prime1_l2081_208150


namespace A_wins_match_prob_correct_l2081_208141

def probA_wins_game : ℝ := 0.6
def probB_wins_game : ℝ := 0.4

def probA_wins_match : ℝ :=
  let probA_wins_first_two := probA_wins_game * probA_wins_game
  let probA_wins_first_and_third := probA_wins_game * probB_wins_game * probA_wins_game
  let probA_wins_last_two := probB_wins_game * probA_wins_game * probA_wins_game
  probA_wins_first_two + probA_wins_first_and_third + probA_wins_last_two

theorem A_wins_match_prob_correct : probA_wins_match = 0.648 := by
  sorry

end A_wins_match_prob_correct_l2081_208141


namespace min_value_x_plus_one_over_x_plus_two_l2081_208123

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1/(x + 2) ∧ y ≥ 0 :=
by
  sorry

end min_value_x_plus_one_over_x_plus_two_l2081_208123


namespace coin_draws_expected_value_l2081_208113

theorem coin_draws_expected_value :
  ∃ f : ℕ → ℝ, (∀ (n : ℕ), n ≥ 4 → f n = (3 : ℝ)) := sorry

end coin_draws_expected_value_l2081_208113


namespace determine_k_l2081_208140

variable (x y z w : ℝ)

theorem determine_k
  (h₁ : 9 / (x + y + w) = k / (x + z + w))
  (h₂ : k / (x + z + w) = 12 / (z - y)) :
  k = 21 :=
sorry

end determine_k_l2081_208140


namespace cost_per_rose_l2081_208112

theorem cost_per_rose (P : ℝ) (h1 : 5 * 12 = 60) (h2 : 0.8 * 60 * P = 288) : P = 6 :=
by
  -- Proof goes here
  sorry

end cost_per_rose_l2081_208112


namespace apples_total_l2081_208156

theorem apples_total (initial_apples : ℕ) (additional_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 56 → 
  additional_apples = 49 → 
  total_apples = initial_apples + additional_apples → 
  total_apples = 105 :=
by 
  intros h_initial h_additional h_total 
  rw [h_initial, h_additional] at h_total 
  exact h_total

end apples_total_l2081_208156


namespace rotated_triangle_surface_area_l2081_208192

theorem rotated_triangle_surface_area :
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (ACLength : ℝ) (BCLength : ℝ) (right_angle : ℝ -> ℝ -> ℝ -> Prop)
    (pi_def : Real) (surface_area : ℝ -> ℝ -> ℝ),
    (right_angle 90 0 90) → (ACLength = 3) → (BCLength = 4) →
    surface_area ACLength BCLength = 24 * pi_def  :=
by
  sorry

end rotated_triangle_surface_area_l2081_208192


namespace binary_remainder_div_4_is_1_l2081_208137

def binary_to_base_10_last_two_digits (b1 b0 : Nat) : Nat :=
  2 * b1 + b0

noncomputable def remainder_of_binary_by_4 (n : Nat) : Nat :=
  match n with
  | 111010110101 => binary_to_base_10_last_two_digits 0 1
  | _ => 0

theorem binary_remainder_div_4_is_1 :
  remainder_of_binary_by_4 111010110101 = 1 := by
  sorry

end binary_remainder_div_4_is_1_l2081_208137


namespace lineup_possibilities_l2081_208109

theorem lineup_possibilities (total_players : ℕ) (all_stars_in_lineup : ℕ) (injured_player : ℕ) :
  total_players = 15 ∧ all_stars_in_lineup = 2 ∧ injured_player = 1 →
  Nat.choose 12 4 = 495 :=
by
  intro h
  sorry

end lineup_possibilities_l2081_208109


namespace gcd_lcm_product_l2081_208177

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 75) :
  (Nat.gcd a b) * (Nat.lcm a b) = 2250 := by
  sorry

end gcd_lcm_product_l2081_208177


namespace man_rate_in_still_water_l2081_208153

theorem man_rate_in_still_water (speed_with_stream speed_against_stream : ℝ)
  (h1 : speed_with_stream = 22) (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end man_rate_in_still_water_l2081_208153


namespace parabola_properties_l2081_208117

-- Given conditions
variables (a b c : ℝ)
variable (h_vertex : ∃ a b c : ℝ, (∀ x, a * (x+1)^2 + 4 = ax^2 + b * x + c))
variable (h_intersection : ∃ A : ℝ, 2 < A ∧ A < 3 ∧ a * A^2 + b * A + c = 0)

-- Define the proof problem
theorem parabola_properties (h_vertex : (b = 2 * a)) (h_a : a < 0) (h_c : c = 4 + a) : 
  ∃ x : ℕ, x = 2 ∧ 
  (∀ a b c : ℝ, a * b * c < 0 → false) ∧ 
  (-4 < a ∧ a < -1 → false) ∧
  (a * c + 2 * b > 1 → false) :=
sorry

end parabola_properties_l2081_208117


namespace earth_surface_inhabitable_fraction_l2081_208118

theorem earth_surface_inhabitable_fraction :
  (1 / 3 : ℝ) * (2 / 3 : ℝ) = 2 / 9 := 
by 
  sorry

end earth_surface_inhabitable_fraction_l2081_208118


namespace tan_x_eq_2_solution_l2081_208175

noncomputable def solution_set_tan_2 : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2}

theorem tan_x_eq_2_solution :
  {x : ℝ | Real.tan x = 2} = solution_set_tan_2 :=
by
  sorry

end tan_x_eq_2_solution_l2081_208175


namespace area_of_enclosed_shape_l2081_208166

noncomputable def enclosed_area : ℝ := 
∫ x in (0 : ℝ)..(2/3 : ℝ), (2 * x - 3 * x^2)

theorem area_of_enclosed_shape : enclosed_area = 4 / 27 := by
  sorry

end area_of_enclosed_shape_l2081_208166


namespace julia_tuesday_kids_l2081_208147

theorem julia_tuesday_kids :
  ∃ x : ℕ, (∃ y : ℕ, y = 6 ∧ y = x + 1) → x = 5 := 
by
  sorry

end julia_tuesday_kids_l2081_208147


namespace three_Z_five_l2081_208157

def Z (a b : ℤ) : ℤ := b + 10 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = 8 := sorry

end three_Z_five_l2081_208157


namespace number_of_ways_to_choose_4_captains_from_15_l2081_208185

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l2081_208185


namespace geom_progression_common_ratio_l2081_208119

theorem geom_progression_common_ratio (x y z r : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : ∃ a, a ≠ 0 ∧ x * (2 * y - z) = a ∧ y * (2 * z - x) = a * r ∧ z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end geom_progression_common_ratio_l2081_208119


namespace fraction_calculation_l2081_208149

theorem fraction_calculation : 
  ( (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) ) = (864 / 1505) := 
by
  sorry

end fraction_calculation_l2081_208149


namespace lucy_bought_cakes_l2081_208160

theorem lucy_bought_cakes (cookies chocolate total c : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) (h4 : c = total - (cookies + chocolate)) : c = 22 := by
  sorry

end lucy_bought_cakes_l2081_208160


namespace monotonic_increasing_interval_l2081_208191

theorem monotonic_increasing_interval : ∀ x : ℝ, (x > 2) → ((x-3) * Real.exp x > 0) :=
sorry

end monotonic_increasing_interval_l2081_208191


namespace equilateral_triangle_octagon_area_ratio_l2081_208184

theorem equilateral_triangle_octagon_area_ratio
  (s_t s_o : ℝ)
  (h_triangle_area : (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2)) :
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by
  sorry

end equilateral_triangle_octagon_area_ratio_l2081_208184


namespace initial_tax_rate_l2081_208163

variable (R : ℝ)

theorem initial_tax_rate
  (income : ℝ := 48000)
  (new_rate : ℝ := 0.30)
  (savings : ℝ := 7200)
  (tax_savings : income * (R / 100) - income * new_rate = savings) :
  R = 45 := by
  sorry

end initial_tax_rate_l2081_208163


namespace all_points_lie_on_parabola_l2081_208125

noncomputable def parabola_curve (u : ℝ) : ℝ × ℝ :=
  let x := 3^u - 4
  let y := 9^u - 7 * 3^u - 2
  (x, y)

theorem all_points_lie_on_parabola (u : ℝ) :
  let (x, y) := parabola_curve u
  y = x^2 + x - 6 := sorry

end all_points_lie_on_parabola_l2081_208125


namespace probability_king_then_queen_l2081_208144

-- Definitions based on the conditions:
def total_cards : ℕ := 52
def ranks_per_suit : ℕ := 13
def suits : ℕ := 4
def kings : ℕ := 4
def queens : ℕ := 4

-- The problem statement rephrased as a theorem:
theorem probability_king_then_queen :
  (kings / total_cards : ℚ) * (queens / (total_cards - 1)) = 4 / 663 := 
by {
  sorry
}

end probability_king_then_queen_l2081_208144


namespace train_crossing_time_l2081_208186

theorem train_crossing_time
  (length_train : ℕ)
  (speed_train_kmph : ℕ)
  (total_length : ℕ)
  (htotal_length : total_length = 225)
  (hlength_train : length_train = 150)
  (hspeed_train_kmph : speed_train_kmph = 45) : 
  (total_length / (speed_train_kmph * 1000 / 3600)) = 18 := by 
  sorry

end train_crossing_time_l2081_208186


namespace total_boys_and_girls_sum_to_41_l2081_208171

theorem total_boys_and_girls_sum_to_41 (Rs : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) (total_amount : ℕ) (num_boys : ℕ) :
  Rs = 1 ∧ amount_per_boy = 12 * Rs ∧ amount_per_girl = 8 * Rs ∧ total_amount = 460 * Rs ∧ num_boys = 33 →
  ∃ num_girls : ℕ, num_boys + num_girls = 41 :=
by
  sorry

end total_boys_and_girls_sum_to_41_l2081_208171


namespace relationship_among_abc_l2081_208100

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l2081_208100


namespace sum_of_common_ratios_eq_three_l2081_208194

theorem sum_of_common_ratios_eq_three
  (k a2 a3 b2 b3 : ℕ)
  (p r : ℕ)
  (h_nonconst1 : k ≠ 0)
  (h_nonconst2 : p ≠ r)
  (h_seq1 : a3 = k * p ^ 2)
  (h_seq2 : b3 = k * r ^ 2)
  (h_seq3 : a2 = k * p)
  (h_seq4 : b2 = k * r)
  (h_eq : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 := 
sorry

end sum_of_common_ratios_eq_three_l2081_208194


namespace bryan_travel_ratio_l2081_208195

theorem bryan_travel_ratio
  (walk_time : ℕ)
  (bus_time : ℕ)
  (evening_walk_time : ℕ)
  (total_travel_hours : ℕ)
  (days_per_year : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_total : ℕ)
  (daily_travel_time : ℕ) :
  walk_time = 5 →
  bus_time = 20 →
  evening_walk_time = 5 →
  total_travel_hours = 365 →
  days_per_year = 365 →
  minutes_per_hour = 60 →
  minutes_total = total_travel_hours * minutes_per_hour →
  daily_travel_time = (walk_time + bus_time + evening_walk_time) * 2 →
  (minutes_total / daily_travel_time = days_per_year) →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 :=
by
  intros
  sorry

end bryan_travel_ratio_l2081_208195


namespace least_positive_integer_exists_l2081_208193

theorem least_positive_integer_exists 
  (exists_k : ∃ k, (1 ≤ k ∧ k ≤ 2 * 5) ∧ (5^2 - 5 + k) % k = 0)
  (not_all_k : ¬(∀ k, (1 ≤ k ∧ k ≤ 2 * 5) → (5^2 - 5 + k) % k = 0)) :
  5 = 5 := 
by
  trivial

end least_positive_integer_exists_l2081_208193


namespace sum_areas_of_eight_disks_l2081_208189

noncomputable def eight_disks_sum_areas (C_radius disk_count : ℝ) 
  (cover_C : ℝ) (no_overlap : ℝ) (tangent_neighbors : ℝ) : ℕ :=
  let r := (2 - Real.sqrt 2)
  let area_one_disk := Real.pi * r^2
  let total_area := disk_count * area_one_disk
  let a := 48
  let b := 32
  let c := 2
  a + b + c

theorem sum_areas_of_eight_disks : eight_disks_sum_areas 1 8 1 1 1 = 82 :=
  by
  -- sorry is used to skip the proof
  sorry

end sum_areas_of_eight_disks_l2081_208189


namespace maximum_ab_value_l2081_208170

noncomputable def ab_max (a b : ℝ) : ℝ :=
  if a > 0 then 2 * a * a - a * a * Real.log a else 0

theorem maximum_ab_value : ∀ (a b : ℝ), (∀ (x : ℝ), (Real.exp x - a * x + a) ≥ b) →
   ab_max a b ≤ if a = Real.exp (3 / 2) then (Real.exp 3) / 2 else sorry :=
by
  intros a b h
  sorry

end maximum_ab_value_l2081_208170


namespace least_incorrect_option_is_A_l2081_208106

def dozen_units : ℕ := 12
def chairs_needed : ℕ := 4

inductive CompletionOption
| dozen
| dozens
| dozen_of
| dozens_of

def correct_option (op : CompletionOption) : Prop :=
  match op with
  | CompletionOption.dozen => dozen_units >= chairs_needed
  | CompletionOption.dozens => False
  | CompletionOption.dozen_of => False
  | CompletionOption.dozens_of => False

theorem least_incorrect_option_is_A : correct_option CompletionOption.dozen :=
by {
  sorry
}

end least_incorrect_option_is_A_l2081_208106


namespace probability_of_red_buttons_l2081_208178

noncomputable def initialJarA : ℕ := 16 -- total buttons in Jar A (6 red, 10 blue)
noncomputable def initialRedA : ℕ := 6 -- initial red buttons in Jar A
noncomputable def initialBlueA : ℕ := 10 -- initial blue buttons in Jar A

noncomputable def initialJarB : ℕ := 5 -- total buttons in Jar B (2 red, 3 blue)
noncomputable def initialRedB : ℕ := 2 -- initial red buttons in Jar B
noncomputable def initialBlueB : ℕ := 3 -- initial blue buttons in Jar B

noncomputable def transferRed : ℕ := 3
noncomputable def transferBlue : ℕ := 3

noncomputable def finalRedA : ℕ := initialRedA - transferRed
noncomputable def finalBlueA : ℕ := initialBlueA - transferBlue

noncomputable def finalRedB : ℕ := initialRedB + transferRed
noncomputable def finalBlueB : ℕ := initialBlueB + transferBlue

noncomputable def remainingJarA : ℕ := finalRedA + finalBlueA
noncomputable def finalJarB : ℕ := finalRedB + finalBlueB

noncomputable def probRedA : ℚ := finalRedA / remainingJarA
noncomputable def probRedB : ℚ := finalRedB / finalJarB

noncomputable def combinedProb : ℚ := probRedA * probRedB

theorem probability_of_red_buttons :
  combinedProb = 3 / 22 := sorry

end probability_of_red_buttons_l2081_208178


namespace inverse_matrix_equation_of_line_l_l2081_208110

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![3, 4]]
noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![-2, 1], ![3/2, -1/2]]

theorem inverse_matrix :
  M⁻¹ = M_inv :=
by
  sorry

def transformed_line (x y : ℚ) : Prop := 2 * (x + 2 * y) - (3 * x + 4 * y) = 4 

theorem equation_of_line_l (x y : ℚ) :
  transformed_line x y → x + 4 = 0 :=
by
  sorry

end inverse_matrix_equation_of_line_l_l2081_208110


namespace monthly_income_of_P_l2081_208146

theorem monthly_income_of_P (P Q R : ℕ) (h1 : P + Q = 10100) (h2 : Q + R = 12500) (h3 : P + R = 10400) : 
  P = 4000 := 
by 
  sorry

end monthly_income_of_P_l2081_208146


namespace min_value_when_a_is_half_range_of_a_for_positivity_l2081_208180

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 2*x + a) / x

theorem min_value_when_a_is_half : 
  ∀ x ∈ Set.Ici (1 : ℝ), f x (1/2) ≥ (7 / 2) := 
by 
  sorry

theorem range_of_a_for_positivity :
  ∀ x ∈ Set.Ici (1 : ℝ), f x a > 0 ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by 
  sorry

end min_value_when_a_is_half_range_of_a_for_positivity_l2081_208180


namespace price_of_book_l2081_208128

-- Definitions based on the problem conditions
def money_xiaowang_has (p : ℕ) : ℕ := 2 * p - 6
def money_xiaoli_has (p : ℕ) : ℕ := 2 * p - 31

def combined_money (p : ℕ) : ℕ := money_xiaowang_has p + money_xiaoli_has p

-- Lean statement to prove the price of each book
theorem price_of_book (p : ℕ) : combined_money p = 3 * p → p = 37 :=
by
  sorry

end price_of_book_l2081_208128


namespace complement_union_M_N_l2081_208198

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l2081_208198


namespace JacobNeed_l2081_208154

-- Definitions of the conditions
def jobEarningsBeforeTax : ℝ := 25 * 15
def taxAmount : ℝ := 0.10 * jobEarningsBeforeTax
def jobEarningsAfterTax : ℝ := jobEarningsBeforeTax - taxAmount

def cookieEarnings : ℝ := 5 * 30

def tutoringEarnings : ℝ := 100 * 4

def lotteryWinnings : ℝ := 700 - 20
def friendShare : ℝ := 0.30 * lotteryWinnings
def netLotteryWinnings : ℝ := lotteryWinnings - friendShare

def giftFromSisters : ℝ := 700 * 2

def totalEarnings : ℝ := jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters

def travelGearExpenses : ℝ := 3 + 47

def netSavings : ℝ := totalEarnings - travelGearExpenses

def tripCost : ℝ := 8000

-- Statement to be proven
theorem JacobNeed (jobEarningsBeforeTax taxAmount jobEarningsAfterTax cookieEarnings tutoringEarnings 
netLotteryWinnings giftFromSisters totalEarnings travelGearExpenses netSavings tripCost : ℝ) : 
  (jobEarningsAfterTax == (25 * 15) - (0.10 * (25 * 15))) → 
  (cookieEarnings == 5 * 30) →
  (tutoringEarnings == 100 * 4) →
  (netLotteryWinnings == (700 - 20) - (0.30 * (700 - 20))) →
  (giftFromSisters == 700 * 2) →
  (totalEarnings == jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters) →
  (travelGearExpenses == 3 + 47) →
  (netSavings == totalEarnings - travelGearExpenses) →
  (tripCost == 8000) →
  (tripCost - netSavings = 5286.50) :=
by
  intros
  sorry

end JacobNeed_l2081_208154


namespace three_letter_words_with_A_at_least_once_l2081_208148

theorem three_letter_words_with_A_at_least_once :
  let total_words := 4^3
  let words_without_A := 3^3
  total_words - words_without_A = 37 :=
by
  let total_words := 4^3
  let words_without_A := 3^3
  sorry

end three_letter_words_with_A_at_least_once_l2081_208148


namespace john_spent_l2081_208174

-- Given definitions from the conditions.
def total_time_in_hours := 4
def additional_minutes := 35
def break_time_per_break := 10
def number_of_breaks := 5
def cost_per_5_minutes := 0.75
def playing_cost (total_time_in_hours additional_minutes break_time_per_break number_of_breaks : ℕ) 
  (cost_per_5_minutes : ℝ) : ℝ :=
  let total_minutes := total_time_in_hours * 60 + additional_minutes
  let break_time := number_of_breaks * break_time_per_break
  let actual_playing_time := total_minutes - break_time
  let number_of_intervals := actual_playing_time / 5
  number_of_intervals * cost_per_5_minutes

-- Statement to be proved.
theorem john_spent (total_time_in_hours := 4) (additional_minutes := 35) (break_time_per_break := 10) 
  (number_of_breaks := 5) (cost_per_5_minutes := 0.75) :
  playing_cost total_time_in_hours additional_minutes break_time_per_break number_of_breaks cost_per_5_minutes = 33.75 := 
by
  sorry

end john_spent_l2081_208174


namespace cost_of_each_cake_l2081_208129

-- Define the conditions
def cakes : ℕ := 3
def payment_by_john : ℕ := 18
def total_payment : ℕ := payment_by_john * 2

-- Statement to prove that each cake costs $12
theorem cost_of_each_cake : (total_payment / cakes) = 12 := by
  sorry

end cost_of_each_cake_l2081_208129


namespace max_load_truck_l2081_208158

theorem max_load_truck (bag_weight : ℕ) (num_bags : ℕ) (remaining_load : ℕ) 
  (h1 : bag_weight = 8) (h2 : num_bags = 100) (h3 : remaining_load = 100) : 
  bag_weight * num_bags + remaining_load = 900 :=
by
  -- We leave the proof step intentionally, as per instructions.
  sorry

end max_load_truck_l2081_208158


namespace part1_part2_l2081_208133

noncomputable def f (x m : ℝ) := |x + 1| + |m - x|

theorem part1 (x : ℝ) : (f x 3) ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2 (m : ℝ) : (∀ x, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
by sorry

end part1_part2_l2081_208133


namespace sum_angles_triangle_complement_l2081_208152

theorem sum_angles_triangle_complement (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) : A + B = 130 :=
by
  have hC : C = 50 := by linarith
  linarith

end sum_angles_triangle_complement_l2081_208152


namespace even_integers_count_l2081_208176

theorem even_integers_count (n : ℤ) (m : ℤ) (total_even : ℤ) 
  (h1 : m = 45) (h2 : total_even = 10) (h3 : m % 2 = 1) :
  (∃ k : ℤ, ∀ x : ℤ, 0 ≤ x ∧ x < total_even → k = n + 2 * x) ∧ (n = 26) :=
by
  sorry

end even_integers_count_l2081_208176


namespace sin_sq_sub_cos_sq_l2081_208138

-- Given condition
variable {α : ℝ}
variable (h : Real.sin α = Real.sqrt 5 / 5)

-- Proof goal
theorem sin_sq_sub_cos_sq (h : Real.sin α = Real.sqrt 5 / 5) : Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 := sorry

end sin_sq_sub_cos_sq_l2081_208138


namespace sandy_receives_correct_change_l2081_208114

-- Define the costs of each item
def cost_cappuccino : ℕ := 2
def cost_iced_tea : ℕ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℕ := 1

-- Define the quantities ordered
def qty_cappuccino : ℕ := 3
def qty_iced_tea : ℕ := 2
def qty_cafe_latte : ℕ := 2
def qty_espresso : ℕ := 2

-- Calculate the total cost
def total_cost : ℝ := (qty_cappuccino * cost_cappuccino) + 
                      (qty_iced_tea * cost_iced_tea) + 
                      (qty_cafe_latte * cost_cafe_latte) + 
                      (qty_espresso * cost_espresso)

-- Define the amount paid
def amount_paid : ℝ := 20

-- Calculate the change
def change : ℝ := amount_paid - total_cost

theorem sandy_receives_correct_change : change = 3 := by
  -- Detailed steps would go here
  sorry

end sandy_receives_correct_change_l2081_208114


namespace largest_possible_perimeter_l2081_208111

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end largest_possible_perimeter_l2081_208111


namespace area_of_path_cost_of_constructing_path_l2081_208143

-- Definitions for the problem
def original_length : ℕ := 75
def original_width : ℕ := 40
def path_width : ℕ := 25 / 10  -- 2.5 converted to a Lean-readable form

-- Conditions
def new_length := original_length + 2 * path_width
def new_width := original_width + 2 * path_width

def area_with_path := new_length * new_width
def area_without_path := original_length * original_width

-- Statements to prove
theorem area_of_path : area_with_path - area_without_path = 600 := sorry

def cost_per_sq_m : ℕ := 2
def total_cost := (area_with_path - area_without_path) * cost_per_sq_m

theorem cost_of_constructing_path : total_cost = 1200 := sorry

end area_of_path_cost_of_constructing_path_l2081_208143


namespace other_student_questions_l2081_208145

theorem other_student_questions (m k o : ℕ) (h1 : m = k - 3) (h2 : k = o + 8) (h3 : m = 40) : o = 35 :=
by
  -- proof goes here
  sorry

end other_student_questions_l2081_208145


namespace football_championship_min_games_l2081_208164

theorem football_championship_min_games :
  (∃ (teams : Finset ℕ) (games : Finset (ℕ × ℕ)),
    teams.card = 20 ∧
    (∀ (a b c : ℕ), a ∈ teams → b ∈ teams → c ∈ teams → a ≠ b → b ≠ c → c ≠ a →
      (a, b) ∈ games ∨ (b, c) ∈ games ∨ (c, a) ∈ games) ∧
    games.card = 90) :=
sorry

end football_championship_min_games_l2081_208164


namespace ratio_of_falls_l2081_208136

variable (SteveFalls : ℕ) (StephFalls : ℕ) (SonyaFalls : ℕ)
variable (H1 : SteveFalls = 3)
variable (H2 : StephFalls = SteveFalls + 13)
variable (H3 : SonyaFalls = 6)

theorem ratio_of_falls : SonyaFalls / (StephFalls / 2) = 3 / 4 := by
  sorry

end ratio_of_falls_l2081_208136


namespace no_solution_exists_l2081_208134

theorem no_solution_exists (x y : ℝ) : 9^(y + 1) / (1 + 4 / x^2) ≠ 1 :=
by
  sorry

end no_solution_exists_l2081_208134


namespace f_g_of_1_l2081_208159

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 5 * x + 6
def g (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

-- The statement we need to prove
theorem f_g_of_1 : f (g 1) = 132 := by
  sorry

end f_g_of_1_l2081_208159


namespace domain_of_f2x_l2081_208104

theorem domain_of_f2x (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = f x) : 
  ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = f (2 * x) :=
by
  sorry

end domain_of_f2x_l2081_208104


namespace Cherry_weekly_earnings_l2081_208190

theorem Cherry_weekly_earnings :
  let cost_3_5 := 2.50
  let cost_6_8 := 4.00
  let cost_9_12 := 6.00
  let cost_13_15 := 8.00
  let num_5kg := 4
  let num_8kg := 2
  let num_10kg := 3
  let num_14kg := 1
  let daily_earnings :=
    (num_5kg * cost_3_5) + (num_8kg * cost_6_8) + (num_10kg * cost_9_12) + (num_14kg * cost_13_15)
  let weekly_earnings := daily_earnings * 7
  weekly_earnings = 308 := by
  sorry

end Cherry_weekly_earnings_l2081_208190


namespace gcd_180_126_l2081_208124

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end gcd_180_126_l2081_208124


namespace problem_l2081_208107

noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 2) (hy : y = Real.sqrt 3 - Real.sqrt 2) :
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 :=
by
  rw [hx, hy]
  sorry

end problem_l2081_208107


namespace michael_total_revenue_l2081_208181

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end michael_total_revenue_l2081_208181


namespace find_x_l2081_208182

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end find_x_l2081_208182


namespace lightest_ball_box_is_blue_l2081_208122

-- Define the weights and counts of balls
def yellow_ball_weight : ℕ := 50
def yellow_ball_count_per_box : ℕ := 50
def white_ball_weight : ℕ := 45
def white_ball_count_per_box : ℕ := 60
def blue_ball_weight : ℕ := 55
def blue_ball_count_per_box : ℕ := 40

-- Calculate the total weight of balls per type
def yellow_box_weight : ℕ := yellow_ball_weight * yellow_ball_count_per_box
def white_box_weight : ℕ := white_ball_weight * white_ball_count_per_box
def blue_box_weight : ℕ := blue_ball_weight * blue_ball_count_per_box

theorem lightest_ball_box_is_blue :
  (blue_box_weight < yellow_box_weight) ∧ (blue_box_weight < white_box_weight) :=
by
  -- Proof can go here
  sorry

end lightest_ball_box_is_blue_l2081_208122


namespace todd_runs_faster_l2081_208105

-- Define the times taken by Brian and Todd
def brian_time : ℕ := 96
def todd_time : ℕ := 88

-- The theorem stating the problem
theorem todd_runs_faster : brian_time - todd_time = 8 :=
by
  -- Solution here
  sorry

end todd_runs_faster_l2081_208105


namespace square_side_length_l2081_208130

theorem square_side_length (x : ℝ) 
  (h : x^2 = 6^2 + 8^2) : x = 10 := 
by sorry

end square_side_length_l2081_208130


namespace symmetric_points_origin_l2081_208172

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end symmetric_points_origin_l2081_208172


namespace min_value_fraction_l2081_208197

theorem min_value_fraction (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m + 2 * n = 1) : 
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_fraction_l2081_208197


namespace factorize_expression_l2081_208102

-- Define variables m and n
variables (m n : ℤ)

-- The theorem stating the equality
theorem factorize_expression : m^3 * n - m * n = m * n * (m - 1) * (m + 1) :=
by sorry

end factorize_expression_l2081_208102


namespace inequality_ratios_l2081_208127

theorem inequality_ratios (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (c / a) > (d / b) :=
sorry

end inequality_ratios_l2081_208127


namespace largest_area_of_rotating_triangle_l2081_208155

def Point := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (13, 0)
def C : Point := (21, 0)

def line (P : Point) (slope : ℝ) (x : ℝ) : ℝ := P.2 + slope * (x - P.1)

def l_A (x : ℝ) : ℝ := line A 1 x
def l_B (x : ℝ) : ℝ := x
def l_C (x : ℝ) : ℝ := line C (-1) x

def rotating_triangle_max_area (l_A l_B l_C : ℝ → ℝ) : ℝ := 116.5

theorem largest_area_of_rotating_triangle :
  rotating_triangle_max_area l_A l_B l_C = 116.5 :=
sorry

end largest_area_of_rotating_triangle_l2081_208155


namespace sum_of_two_numbers_l2081_208161

theorem sum_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x * y = 200) : (x + y = 30) :=
by sorry

end sum_of_two_numbers_l2081_208161


namespace convex_polyhedron_faces_same_edges_l2081_208120

theorem convex_polyhedron_faces_same_edges (n : ℕ) (f : Fin n → ℕ) 
  (n_ge_4 : 4 ≤ n)
  (h : ∀ i : Fin n, 3 ≤ f i ∧ f i ≤ n - 1) : 
  ∃ (i j : Fin n), i ≠ j ∧ f i = f j := 
by
  sorry

end convex_polyhedron_faces_same_edges_l2081_208120


namespace meteorological_forecasts_inaccuracy_l2081_208183

theorem meteorological_forecasts_inaccuracy :
  let pA_accurate := 0.8
  let pB_accurate := 0.7
  let pA_inaccurate := 1 - pA_accurate
  let pB_inaccurate := 1 - pB_accurate
  pA_inaccurate * pB_inaccurate = 0.06 :=
by
  sorry

end meteorological_forecasts_inaccuracy_l2081_208183


namespace find_p_l2081_208188

theorem find_p (p : ℝ) (h : 0 < p ∧ p < 1) : 
  p + (1 - p) * p + (1 - p)^2 * p = 0.784 → p = 0.4 :=
by
  intros h_eq
  sorry

end find_p_l2081_208188


namespace p_implies_q_not_q_implies_p_l2081_208115

def p (a : ℝ) := a = Real.sqrt 2

def q (a : ℝ) := ∀ x y : ℝ, y = -(x : ℝ) → (x^2 + (y - a)^2 = 1)

theorem p_implies_q_not_q_implies_p (a : ℝ) : (p a → q a) ∧ (¬(q a → p a)) := 
    sorry

end p_implies_q_not_q_implies_p_l2081_208115


namespace evaluate_expression_l2081_208101

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 5)) = 15 / 16 :=
by 
  sorry

end evaluate_expression_l2081_208101


namespace eval_f_neg_2_l2081_208121

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem eval_f_neg_2 : f (-2) = 19 :=
by
  sorry

end eval_f_neg_2_l2081_208121


namespace tomatoes_multiplier_l2081_208108

theorem tomatoes_multiplier (before_vacation : ℕ) (grown_during_vacation : ℕ)
  (h1 : before_vacation = 36)
  (h2 : grown_during_vacation = 3564) :
  (before_vacation + grown_during_vacation) / before_vacation = 100 :=
by
  -- Insert proof here later
  sorry

end tomatoes_multiplier_l2081_208108


namespace number_of_paper_cups_is_40_l2081_208103

noncomputable def cost_paper_plate : ℝ := sorry
noncomputable def cost_paper_cup : ℝ := sorry
noncomputable def num_paper_cups_in_second_purchase : ℝ := sorry

-- Conditions
axiom first_condition : 100 * cost_paper_plate + 200 * cost_paper_cup = 7.50
axiom second_condition : 20 * cost_paper_plate + num_paper_cups_in_second_purchase * cost_paper_cup = 1.50

-- Goal
theorem number_of_paper_cups_is_40 : num_paper_cups_in_second_purchase = 40 := 
by 
  sorry

end number_of_paper_cups_is_40_l2081_208103
