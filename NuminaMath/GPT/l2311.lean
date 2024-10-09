import Mathlib

namespace distinct_left_views_l2311_231138

/-- Consider 10 small cubes each having dimension 1 cm × 1 cm × 1 cm.
    Each pair of adjacent cubes shares at least one edge (1 cm) or one face (1 cm × 1 cm).
    The cubes must not be suspended in the air and each cube's edges should be either
    perpendicular or parallel to the horizontal lines. Prove that the number of distinct
    left views of any arrangement of these 10 cubes is 16. -/
theorem distinct_left_views (cube_count : ℕ) (dimensions : ℝ) 
  (shared_edge : (ℝ × ℝ) → Prop) (no_suspension : Prop) (alignment : Prop) :
  cube_count = 10 →
  dimensions = 1 →
  (∀ x y, shared_edge (x, y) ↔ x = y ∨ x - y = 1) →
  no_suspension →
  alignment →
  distinct_left_views_count = 16 :=
by
  sorry

end distinct_left_views_l2311_231138


namespace crickets_total_l2311_231194

noncomputable def initial_amount : ℝ := 7.5
noncomputable def additional_amount : ℝ := 11.25
noncomputable def total_amount : ℝ := 18.75

theorem crickets_total : initial_amount + additional_amount = total_amount :=
by
  sorry

end crickets_total_l2311_231194


namespace kenny_jumps_l2311_231109

theorem kenny_jumps (M : ℕ) (h : 34 + M + 0 + 123 + 64 + 23 + 61 = 325) : M = 20 :=
by
  sorry

end kenny_jumps_l2311_231109


namespace max_value_quadratic_l2311_231135

theorem max_value_quadratic (r : ℝ) : 
  ∃ M, (∀ r, -3 * r^2 + 36 * r - 9 ≤ M) ∧ M = 99 :=
sorry

end max_value_quadratic_l2311_231135


namespace exists_b_mod_5_l2311_231126

theorem exists_b_mod_5 (p q r s : ℤ) (h1 : ¬ (s % 5 = 0)) (a : ℤ) (h2 : (p * a^3 + q * a^2 + r * a + s) % 5 = 0) : 
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end exists_b_mod_5_l2311_231126


namespace prove_trig_values_l2311_231129

/-- Given angles A and B, where both are acute angles,
  and their sine values are known,
  we aim to prove the cosine of (A + B) and the measure
  of angle C in triangle ABC. -/
theorem prove_trig_values (A B : ℝ)
  (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2)
  (sin_A_eq : Real.sin A = (Real.sqrt 5) / 5)
  (sin_B_eq : Real.sin B = (Real.sqrt 10) / 10) :
  Real.cos (A + B) = (Real.sqrt 2) / 2 ∧ (π - (A + B)) = 3 * π / 4 := by
sorry

end prove_trig_values_l2311_231129


namespace hexagon_arithmetic_sum_l2311_231120

theorem hexagon_arithmetic_sum (a n : ℝ) (h : 6 * a + 15 * n = 720) : 2 * a + 5 * n = 240 :=
by
  sorry

end hexagon_arithmetic_sum_l2311_231120


namespace num_frisbees_more_than_deck_cards_l2311_231130

variables (M F D x : ℕ)
variable (bought_fraction : ℝ)

theorem num_frisbees_more_than_deck_cards :
  M = 60 ∧ M = 2 * F ∧ F = D + x ∧
  M + bought_fraction * M + F + bought_fraction * F + D + bought_fraction * D = 140 ∧ bought_fraction = 2/5 →
  x = 20 :=
by
  sorry

end num_frisbees_more_than_deck_cards_l2311_231130


namespace planes_meet_in_50_minutes_l2311_231114

noncomputable def time_to_meet (d : ℕ) (vA vB : ℕ) : ℚ :=
  d / (vA + vB : ℚ)

theorem planes_meet_in_50_minutes
  (d : ℕ) (vA vB : ℕ)
  (h_d : d = 500) (h_vA : vA = 240) (h_vB : vB = 360) :
  (time_to_meet d vA vB * 60 : ℚ) = 50 := by
  sorry

end planes_meet_in_50_minutes_l2311_231114


namespace selected_room_l2311_231131

theorem selected_room (room_count interval selected initial_room : ℕ) 
  (h_init : initial_room = 5)
  (h_interval : interval = 8)
  (h_room_count : room_count = 64) : 
  ∃ (nth_room : ℕ), nth_room = initial_room + interval * 6 ∧ nth_room = 53 :=
by
  sorry

end selected_room_l2311_231131


namespace algorithm_characteristics_l2311_231178

theorem algorithm_characteristics (finiteness : Prop) (definiteness : Prop) (output_capability : Prop) (unique : Prop) 
  (h1 : finiteness = true) 
  (h2 : definiteness = true) 
  (h3 : output_capability = true) 
  (h4 : unique = false) : 
  incorrect_statement = unique := 
by
  sorry

end algorithm_characteristics_l2311_231178


namespace total_money_l2311_231199

namespace MoneyProof

variables (B J T : ℕ)

-- Given conditions
def condition_beth : Prop := B + 35 = 105
def condition_jan : Prop := J - 10 = B
def condition_tom : Prop := T = 3 * (J - 10)

-- Proof that the total money is $360
theorem total_money (h1 : condition_beth B) (h2 : condition_jan B J) (h3 : condition_tom J T) :
  B + J + T = 360 :=
by
  sorry

end MoneyProof

end total_money_l2311_231199


namespace homework_duration_equation_l2311_231115

-- Define the initial and final durations and the rate of decrease
def initial_duration : ℝ := 100
def final_duration : ℝ := 70
def rate_of_decrease (x : ℝ) : ℝ := x

-- Statement of the proof problem
theorem homework_duration_equation (x : ℝ) :
  initial_duration * (1 - rate_of_decrease x) ^ 2 = final_duration :=
sorry

end homework_duration_equation_l2311_231115


namespace josh_initial_money_l2311_231103

/--
Josh spent $1.75 on a drink, and then spent another $1.25, and has $6.00 left. 
Prove that initially Josh had $9.00.
-/
theorem josh_initial_money : 
  ∃ (initial : ℝ), (initial - 1.75 - 1.25 = 6) ∧ initial = 9 := 
sorry

end josh_initial_money_l2311_231103


namespace car_and_bicycle_distances_l2311_231163

noncomputable def train_speed : ℝ := 100 -- speed of the train in mph
noncomputable def car_speed : ℝ := (2 / 3) * train_speed -- speed of the car in mph
noncomputable def bicycle_speed : ℝ := (1 / 5) * train_speed -- speed of the bicycle in mph
noncomputable def travel_time_hours : ℝ := 30 / 60 -- travel time in hours, which is 0.5 hours

noncomputable def car_distance : ℝ := car_speed * travel_time_hours
noncomputable def bicycle_distance : ℝ := bicycle_speed * travel_time_hours

theorem car_and_bicycle_distances :
  car_distance = 100 / 3 ∧ bicycle_distance = 10 :=
by
  sorry

end car_and_bicycle_distances_l2311_231163


namespace good_games_count_l2311_231198

theorem good_games_count :
  ∀ (g1 g2 b : ℕ), g1 = 50 → g2 = 27 → b = 74 → g1 + g2 - b = 3 := by
  intros g1 g2 b hg1 hg2 hb
  sorry

end good_games_count_l2311_231198


namespace downstream_speed_l2311_231132

variable (Vu Vs Vd Vc : ℝ)

theorem downstream_speed
  (h1 : Vu = 25)
  (h2 : Vs = 32)
  (h3 : Vu = Vs - Vc)
  (h4 : Vd = Vs + Vc) :
  Vd = 39 := by
  sorry

end downstream_speed_l2311_231132


namespace books_per_shelf_l2311_231165

theorem books_per_shelf (mystery_shelves picture_shelves total_books : ℕ)
    (h₁ : mystery_shelves = 5)
    (h₂ : picture_shelves = 3)
    (h₃ : total_books = 32) :
    (total_books / (mystery_shelves + picture_shelves) = 4) :=
by
    sorry

end books_per_shelf_l2311_231165


namespace polygon_sides_eq_7_l2311_231188

theorem polygon_sides_eq_7 (n : ℕ) (h : n * (n - 3) / 2 = 2 * n) : n = 7 := 
by 
  sorry

end polygon_sides_eq_7_l2311_231188


namespace max_s_value_l2311_231162

theorem max_s_value (p q r s : ℝ) (h1 : p + q + r + s = 10) (h2 : (p * q) + (p * r) + (p * s) + (q * r) + (q * s) + (r * s) = 20) :
  s ≤ (5 * (1 + Real.sqrt 21)) / 2 :=
sorry

end max_s_value_l2311_231162


namespace calculate_x_l2311_231148

variable (a b x : ℝ)
variable (h1 : r = (3 * a) ^ (3 * b))
variable (h2 : r = a ^ b * x ^ b)
variable (h3 : x > 0)

theorem calculate_x (a b x : ℝ) (h1 : r = (3 * a) ^ (3 * b)) (h2 : r = a ^ b * x ^ b) (h3 : x > 0) : x = 27 * a ^ 2 := by
  sorry

end calculate_x_l2311_231148


namespace range_of_a_l2311_231195

noncomputable def p (x: ℝ) : Prop := |4 * x - 1| ≤ 1
noncomputable def q (x a: ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a: ℝ) :
  (¬ (∀ x, p x) → (¬ (∀ x, q x a))) ∧ (¬ (¬ (∀ x, p x) → (¬ (∀ x, q x a))))
  ↔ (-1 / 2 ≤ a ∧ a ≤ 0) :=
sorry

end range_of_a_l2311_231195


namespace somu_fathers_age_ratio_l2311_231123

noncomputable def somus_age := 16

def proof_problem (S F : ℕ) : Prop :=
  S = 16 ∧ 
  (S - 8 = (1 / 5) * (F - 8)) ∧
  (S / F = 1 / 3)

theorem somu_fathers_age_ratio (S F : ℕ) : proof_problem S F :=
by
  sorry

end somu_fathers_age_ratio_l2311_231123


namespace solution_l2311_231157

theorem solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 := 
by 
  -- Insert proof here
  sorry

end solution_l2311_231157


namespace james_puzzle_completion_time_l2311_231190

theorem james_puzzle_completion_time :
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10
  total_minutes = 400 :=
by
  -- Definitions based on conditions
  let num_puzzles := 2
  let pieces_per_puzzle := 2000
  let pieces_per_10_minutes := 100
  let total_pieces := num_puzzles * pieces_per_puzzle
  let sets_of_100_pieces := total_pieces / pieces_per_10_minutes
  let total_minutes := sets_of_100_pieces * 10

  -- Using sorry to skip proof
  sorry

end james_puzzle_completion_time_l2311_231190


namespace no_solution_xy_in_nat_star_l2311_231156

theorem no_solution_xy_in_nat_star (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
by
  -- The proof would go here, but we'll leave it out for now.
  sorry

end no_solution_xy_in_nat_star_l2311_231156


namespace smallest_angle_in_right_triangle_l2311_231110

-- Given conditions
def angle_α := 90 -- The right-angle in degrees
def angle_β := 55 -- The given angle in degrees

-- Goal: Prove that the smallest angle is 35 degrees.
theorem smallest_angle_in_right_triangle (a b c : ℕ) (h1 : a = angle_α) (h2 : b = angle_β) (h3 : c = 180 - a - b) : c = 35 := 
by {
  -- use sorry to skip the proof steps
  sorry
}

end smallest_angle_in_right_triangle_l2311_231110


namespace base_five_equals_base_b_l2311_231181

theorem base_five_equals_base_b : ∃ (b : ℕ), b > 0 ∧ (2 * 5^1 + 4 * 5^0) = (1 * b^2 + 0 * b^1 + 1 * b^0) := by
  sorry

end base_five_equals_base_b_l2311_231181


namespace fraction_pizza_covered_by_pepperoni_l2311_231141

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end fraction_pizza_covered_by_pepperoni_l2311_231141


namespace potion_kit_cost_is_18_l2311_231116

def price_spellbook : ℕ := 5
def count_spellbooks : ℕ := 5
def price_owl : ℕ := 28
def count_potion_kits : ℕ := 3
def payment_total_silver : ℕ := 537
def silver_per_gold : ℕ := 9

def cost_each_potion_kit_in_silver (payment_total_silver : ℕ)
                                   (price_spellbook : ℕ)
                                   (count_spellbooks : ℕ)
                                   (price_owl : ℕ)
                                   (count_potion_kits : ℕ)
                                   (silver_per_gold : ℕ) : ℕ :=
  let total_gold := payment_total_silver / silver_per_gold
  let cost_spellbooks := count_spellbooks * price_spellbook
  let cost_remaining_gold := total_gold - cost_spellbooks - price_owl
  let cost_each_potion_kit_gold := cost_remaining_gold / count_potion_kits
  cost_each_potion_kit_gold * silver_per_gold

theorem potion_kit_cost_is_18 :
  cost_each_potion_kit_in_silver payment_total_silver
                                 price_spellbook
                                 count_spellbooks
                                 price_owl
                                 count_potion_kits
                                 silver_per_gold = 18 :=
by sorry

end potion_kit_cost_is_18_l2311_231116


namespace total_kids_l2311_231159

theorem total_kids (girls boys: ℕ) (h1: girls = 3) (h2: boys = 6) : girls + boys = 9 :=
by
  sorry

end total_kids_l2311_231159


namespace product_of_roots_proof_l2311_231145

noncomputable def product_of_roots : ℚ :=
  let leading_coeff_poly1 := 3
  let leading_coeff_poly2 := 4
  let constant_term_poly1 := -15
  let constant_term_poly2 := 9
  let a := leading_coeff_poly1 * leading_coeff_poly2
  let b := constant_term_poly1 * constant_term_poly2
  (b : ℚ) / a

theorem product_of_roots_proof :
  product_of_roots = -45/4 :=
by
  sorry

end product_of_roots_proof_l2311_231145


namespace supplementary_angle_l2311_231101

theorem supplementary_angle (θ : ℝ) (k : ℤ) : (θ = 10) → (∃ k, θ + 250 = k * 360 + 360) :=
by
  sorry

end supplementary_angle_l2311_231101


namespace equilateral_triangle_area_ratio_l2311_231167

theorem equilateral_triangle_area_ratio :
  let side_small := 1
  let perim_small := 3 * side_small
  let total_fencing := 6 * perim_small
  let side_large := total_fencing / 3
  let area_small := (Real.sqrt 3) / 4 * side_small ^ 2
  let area_large := (Real.sqrt 3) / 4 * side_large ^ 2
  let total_area_small := 6 * area_small
  total_area_small / area_large = 1 / 6 :=
by
  sorry

end equilateral_triangle_area_ratio_l2311_231167


namespace net_profit_calc_l2311_231175

theorem net_profit_calc:
  ∃ (x y : ℕ), x + y = 25 ∧ 1700 * x + 1800 * y = 44000 ∧ 2400 * x + 2600 * y = 63000 := by
  sorry

end net_profit_calc_l2311_231175


namespace P_plus_Q_l2311_231191

theorem P_plus_Q (P Q : ℝ) (h : ∀ x : ℝ, x ≠ 4 → (P / (x - 4) + Q * (x + 2) = (-4 * x^2 + 16 * x + 30) / (x - 4))) : P + Q = 42 :=
sorry

end P_plus_Q_l2311_231191


namespace time_to_hit_ground_l2311_231108

theorem time_to_hit_ground : ∃ t : ℝ, 
  (y = -4.9 * t^2 + 7.2 * t + 8) → (y - (-0.6 * t) * t = 0) → t = 223/110 :=
by
  sorry

end time_to_hit_ground_l2311_231108


namespace combination_permutation_value_l2311_231173

theorem combination_permutation_value (n : ℕ) (h : (n * (n - 1)) = 42) : (Nat.factorial n) / (Nat.factorial 3 * Nat.factorial (n - 3)) = 35 := 
by
  sorry

end combination_permutation_value_l2311_231173


namespace problem_solution_l2311_231174
open Real

theorem problem_solution (a b c : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) :
  a * (1 - b) ≤ 1 / 4 ∨ b * (1 - c) ≤ 1 / 4 ∨ c * (1 - a) ≤ 1 / 4 :=
by
  sorry

end problem_solution_l2311_231174


namespace greatest_drop_in_price_is_May_l2311_231100

def priceChangeJan := -1.25
def priceChangeFeb := 2.75
def priceChangeMar := -0.75
def priceChangeApr := 1.50
def priceChangeMay := -3.00
def priceChangeJun := -1.00

theorem greatest_drop_in_price_is_May :
  priceChangeMay < priceChangeJan ∧
  priceChangeMay < priceChangeMar ∧
  priceChangeMay < priceChangeApr ∧
  priceChangeMay < priceChangeJun ∧
  priceChangeMay < priceChangeFeb :=
by sorry

end greatest_drop_in_price_is_May_l2311_231100


namespace seats_per_bus_l2311_231170

-- Conditions
def total_students : ℕ := 180
def total_buses : ℕ := 3

-- Theorem Statement
theorem seats_per_bus : (total_students / total_buses) = 60 := 
by 
  sorry

end seats_per_bus_l2311_231170


namespace plus_one_eq_next_plus_l2311_231196

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end plus_one_eq_next_plus_l2311_231196


namespace hair_cut_first_day_l2311_231179

theorem hair_cut_first_day 
  (total_hair_cut : ℝ) 
  (hair_cut_second_day : ℝ) 
  (h_total : total_hair_cut = 0.875) 
  (h_second : hair_cut_second_day = 0.5) : 
  total_hair_cut - hair_cut_second_day = 0.375 := 
  by
  simp [h_total, h_second]
  sorry

end hair_cut_first_day_l2311_231179


namespace solution_l2311_231121

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ),
    x - y = 1 ∧
    x^3 - y^3 = 2 ∧
    x^4 + y^4 = 23 / 9 ∧
    x^5 - y^5 = 29 / 9

theorem solution : problem_statement := sorry

end solution_l2311_231121


namespace M_intersect_N_l2311_231153

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def intersection (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∈ N}

theorem M_intersect_N :
  intersection M N = {x | 1 ≤ x ∧ x < 2} := 
sorry

end M_intersect_N_l2311_231153


namespace general_term_of_sequence_l2311_231161

theorem general_term_of_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, a (n + 1) = (n^2 * (a n)^2 + 5) / ((n^2 - 1) * a (n - 1))) :
  ∀ n : ℕ, a n = 
    if n = 0 then 0 else
    (1 / n) * ( (63 - 13 * Real.sqrt 21) / 42 * ((5 + Real.sqrt 21) / 2) ^ n + 
                (63 + 13 * Real.sqrt 21) / 42 * ((5 - Real.sqrt 21) / 2) ^ n) :=
by
  sorry

end general_term_of_sequence_l2311_231161


namespace determine_k_for_circle_l2311_231152

theorem determine_k_for_circle (x y k : ℝ) (h : x^2 + 14*x + y^2 + 8*y - k = 0) (r : ℝ) :
  r = 5 → k = 40 :=
by
  intros radius_eq_five
  sorry

end determine_k_for_circle_l2311_231152


namespace geometric_seq_sum_l2311_231164

theorem geometric_seq_sum (a : ℝ) (q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) 
    (hS4 : a * (1 - q^4) / (1 - q) = 1) 
    (hS12 : a * (1 - q^12) / (1 - q) = 13) 
    : a * q^12 * (1 + q + q^2 + q^3) = 27 := 
by
  sorry

end geometric_seq_sum_l2311_231164


namespace bridge_length_l2311_231105

theorem bridge_length (length_train : ℝ) (speed_train : ℝ) (time : ℝ) (h1 : length_train = 15) (h2 : speed_train = 275) (h3 : time = 48) : 
    (speed_train / 100) * time - length_train = 117 := 
by
    -- these are the provided conditions, enabling us to skip actual proof steps with 'sorry'
    sorry

end bridge_length_l2311_231105


namespace amy_lily_tie_l2311_231197

noncomputable def tie_probability : ℚ :=
    let amy_win := (2 / 5 : ℚ)
    let lily_win := (1 / 4 : ℚ)
    let total_win := amy_win + lily_win
    1 - total_win

theorem amy_lily_tie (h1 : (2 / 5 : ℚ) = 2 / 5) 
                     (h2 : (1 / 4 : ℚ) = 1 / 4)
                     (h3 : (2 / 5 : ℚ) ≥ 2 * (1 / 4 : ℚ) ∨ (1 / 4 : ℚ) ≥ 2 * (2 / 5 : ℚ)) :
    tie_probability = 7 / 20 :=
by
  sorry

end amy_lily_tie_l2311_231197


namespace number_of_pieces_from_rod_l2311_231149

theorem number_of_pieces_from_rod (rod_length_m : ℕ) (piece_length_cm : ℕ) (meter_to_cm : ℕ) 
  (h1 : rod_length_m = 34) (h2 : piece_length_cm = 85) (h3 : meter_to_cm = 100) : 
  rod_length_m * meter_to_cm / piece_length_cm = 40 := by
  sorry

end number_of_pieces_from_rod_l2311_231149


namespace sequence_periodic_l2311_231155

theorem sequence_periodic (a : ℕ → ℝ) (h1 : a 1 = 0) (h2 : ∀ n, a n + a (n + 1) = 2) : a 2011 = 0 := by
  sorry

end sequence_periodic_l2311_231155


namespace two_beta_plus_alpha_eq_pi_div_two_l2311_231182

theorem two_beta_plus_alpha_eq_pi_div_two
  (α β : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2)
  (hβ1 : 0 < β) (hβ2 : β < π / 2)
  (h : Real.tan α + Real.tan β = 1 / Real.cos α) :
  2 * β + α = π / 2 :=
sorry

end two_beta_plus_alpha_eq_pi_div_two_l2311_231182


namespace max_equal_product_l2311_231183

theorem max_equal_product (a b c d e f : ℕ) (h1 : a = 10) (h2 : b = 15) (h3 : c = 20) (h4 : d = 30) (h5 : e = 40) (h6 : f = 60) :
  ∃ S, (a * b * c * d * e * f) * 450 = S^3 ∧ S = 18000 := 
by
  sorry

end max_equal_product_l2311_231183


namespace division_remainder_l2311_231169

theorem division_remainder (n : ℕ) (h : n = 8 * 8 + 0) : n % 5 = 4 := by
  sorry

end division_remainder_l2311_231169


namespace find_a5_plus_a7_l2311_231139

variable {a : ℕ → ℝ}

theorem find_a5_plus_a7 (h : a 3 + a 9 = 16) : a 5 + a 7 = 16 := 
sorry

end find_a5_plus_a7_l2311_231139


namespace distance_between_points_l2311_231140

theorem distance_between_points :
  let (x1, y1) := (1, 2)
  let (x2, y2) := (6, 5)
  let d := Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)
  d = Real.sqrt 34 :=
by
  sorry

end distance_between_points_l2311_231140


namespace expected_value_range_of_p_l2311_231143

theorem expected_value_range_of_p (p : ℝ) (X : ℕ → ℝ) :
  (∀ n, (n = 1 → X n = p) ∧ 
        (n = 2 → X n = p * (1 - p)) ∧ 
        (n = 3 → X n = (1 - p) ^ 2)) →
  (p^2 - 3 * p + 3 > 1.75) → 
  0 < p ∧ p < 0.5 := by
  intros hprob hexp
  -- Proof would be filled in here
  sorry

end expected_value_range_of_p_l2311_231143


namespace find_a_l2311_231171

noncomputable def coefficient_of_x3_in_expansion (a : ℝ) : ℝ :=
  6 * a^2 - 15 * a + 20 

theorem find_a (a : ℝ) (h : coefficient_of_x3_in_expansion a = 56) : a = 6 ∨ a = -1 :=
  sorry

end find_a_l2311_231171


namespace triple_sum_of_45_point_2_and_one_fourth_l2311_231186

theorem triple_sum_of_45_point_2_and_one_fourth : 
  (3 * (45.2 + 0.25)) = 136.35 :=
by
  sorry

end triple_sum_of_45_point_2_and_one_fourth_l2311_231186


namespace minor_axis_length_l2311_231160

theorem minor_axis_length {x y : ℝ} (h : x^2 / 16 + y^2 / 9 = 1) : 6 = 6 :=
by
  sorry

end minor_axis_length_l2311_231160


namespace tetrahedron_volume_l2311_231128

noncomputable def volume_of_tetrahedron (AB : ℝ) (area_ABC : ℝ) (area_ABD : ℝ) (angle_ABC_ABD : ℝ) : ℝ :=
  (1/3) * area_ABC * area_ABD * (Real.sin angle_ABC_ABD) * (AB / (Real.sqrt 2))

theorem tetrahedron_volume :
  let AB := 5 -- edge AB length in cm
  let area_ABC := 18 -- area of face ABC in cm^2
  let area_ABD := 24 -- area of face ABD in cm^2
  let angle_ABC_ABD := Real.pi / 4 -- 45 degrees in radians
  volume_of_tetrahedron AB area_ABC area_ABD angle_ABC_ABD = 43.2 :=
by
  sorry

end tetrahedron_volume_l2311_231128


namespace Guido_costs_42840_l2311_231154

def LightningMcQueenCost : ℝ := 140000
def MaterCost : ℝ := 0.1 * LightningMcQueenCost
def SallyCostBeforeModifications : ℝ := 3 * MaterCost
def SallyCostAfterModifications : ℝ := SallyCostBeforeModifications + 0.2 * SallyCostBeforeModifications
def GuidoCost : ℝ := SallyCostAfterModifications - 0.15 * SallyCostAfterModifications

theorem Guido_costs_42840 :
  GuidoCost = 42840 :=
sorry

end Guido_costs_42840_l2311_231154


namespace unique_decomposition_of_two_reciprocals_l2311_231177

theorem unique_decomposition_of_two_reciprocals (p : ℕ) (hp : Nat.Prime p) (hp_ne_two : p ≠ 2) :
  ∃ (x y : ℕ), x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 2 / (p : ℝ)) := sorry

end unique_decomposition_of_two_reciprocals_l2311_231177


namespace repeating_decimal_sum_num_denom_l2311_231187

noncomputable def repeating_decimal_to_fraction (n d : ℕ) (rep : ℚ) : ℚ :=
(rep * (10^d) - rep) / ((10^d) - 1)

theorem repeating_decimal_sum_num_denom
  (x : ℚ)
  (h1 : x = repeating_decimal_to_fraction 45 2 0.45)
  (h2 : repeating_decimal_to_fraction 45 2 0.45 = 5/11) : 
  (5 + 11) = 16 :=
by 
  sorry

end repeating_decimal_sum_num_denom_l2311_231187


namespace floor_ceil_difference_l2311_231127

theorem floor_ceil_difference : 
  let a := (18 / 5) * (-33 / 4)
  let b := ⌈(-33 / 4 : ℝ)⌉
  let c := (18 / 5) * (b : ℝ)
  let d := ⌈c⌉
  ⌊a⌋ - d = -2 :=
by
  sorry

end floor_ceil_difference_l2311_231127


namespace find_date_behind_l2311_231136

variables (x y : ℕ)
-- Conditions
def date_behind_C := x
def date_behind_A := x + 1
def date_behind_B := x + 13
def date_behind_P := x + 14

-- Statement to prove
theorem find_date_behind : (x + y = (x + 1) + (x + 13)) → (y = date_behind_P) :=
by
  sorry

end find_date_behind_l2311_231136


namespace opposite_face_of_X_is_Y_l2311_231158

-- Define the labels for the cube faces
inductive Label
| X | V | Z | W | U | Y

-- Define adjacency relations
def adjacent (a b : Label) : Prop :=
  (a = Label.X ∧ (b = Label.V ∨ b = Label.Z ∨ b = Label.W ∨ b = Label.U)) ∨
  (b = Label.X ∧ (a = Label.V ∨ a = Label.Z ∨ a = Label.W ∨ a = Label.U))

-- Define the theorem to prove the face opposite to X
theorem opposite_face_of_X_is_Y : ∀ l1 l2 l3 l4 l5 l6 : Label,
  l1 = Label.X →
  l2 = Label.V →
  l3 = Label.Z →
  l4 = Label.W →
  l5 = Label.U →
  l6 = Label.Y →
  ¬ adjacent l1 l6 →
  ¬ adjacent l2 l6 →
  ¬ adjacent l3 l6 →
  ¬ adjacent l4 l6 →
  ¬ adjacent l5 l6 →
  ∃ (opposite : Label), opposite = Label.Y ∧ opposite = l6 :=
by sorry

end opposite_face_of_X_is_Y_l2311_231158


namespace total_worth_of_produce_is_630_l2311_231185

def bundles_of_asparagus : ℕ := 60
def price_per_bundle_asparagus : ℝ := 3.00

def boxes_of_grapes : ℕ := 40
def price_per_box_grapes : ℝ := 2.50

def num_apples : ℕ := 700
def price_per_apple : ℝ := 0.50

def total_worth : ℝ :=
  bundles_of_asparagus * price_per_bundle_asparagus +
  boxes_of_grapes * price_per_box_grapes +
  num_apples * price_per_apple

theorem total_worth_of_produce_is_630 : 
  total_worth = 630 := by
  sorry

end total_worth_of_produce_is_630_l2311_231185


namespace find_multiple_of_savings_l2311_231133

variable (A K m : ℝ)

-- Conditions
def condition1 : Prop := A - 150 = (1 / 3) * K
def condition2 : Prop := A + K = 750

-- Question
def question : Prop := m * K = 3 * A

-- Proof Problem Statement
theorem find_multiple_of_savings (h1 : condition1 A K) (h2 : condition2 A K) : 
  question A K 2 :=
sorry

end find_multiple_of_savings_l2311_231133


namespace find_intersection_l2311_231113

variable (A : Set ℝ)
variable (B : Set ℝ := {1, 2})
variable (f : ℝ → ℝ := λ x => x^2)

theorem find_intersection (h : ∀ x, x ∈ A → f x ∈ B) : A ∩ B = ∅ ∨ A ∩ B = {1} :=
by
  sorry

end find_intersection_l2311_231113


namespace yard_length_l2311_231112

theorem yard_length (father_step : ℝ) (son_step : ℝ) (total_footprints : ℕ) 
  (h_father_step : father_step = 0.72) 
  (h_son_step : son_step = 0.54) 
  (h_total_footprints : total_footprints = 61) : 
  ∃ length : ℝ, length = 21.6 :=
by
  sorry

end yard_length_l2311_231112


namespace gcd_7920_14553_l2311_231107

theorem gcd_7920_14553 : Int.gcd 7920 14553 = 11 := by
  sorry

end gcd_7920_14553_l2311_231107


namespace intersection_of_set_M_with_complement_of_set_N_l2311_231172

theorem intersection_of_set_M_with_complement_of_set_N (U M N : Set ℕ) (hU : U = {1, 2, 3, 4, 5})
  (hM : M = {1, 4, 5}) (hN : N = {1, 3}) : M ∩ (U \ N) = {4, 5} :=
by
  sorry

end intersection_of_set_M_with_complement_of_set_N_l2311_231172


namespace find_subtracted_number_l2311_231166

-- Given conditions
def t : ℕ := 50
def k : ℕ := 122
def eq_condition (n : ℤ) : Prop := t = (5 / 9 : ℚ) * (k - n)

-- The proof problem proving the number subtracted from k is 32
theorem find_subtracted_number : eq_condition 32 :=
by
  -- implementation here will demonstrate that t = 50 implies the number is 32
  sorry

end find_subtracted_number_l2311_231166


namespace john_max_questions_correct_l2311_231150

variable (c w b : ℕ)

theorem john_max_questions_correct (H1 : c + w + b = 20) (H2 : 5 * c - 2 * w = 48) : c ≤ 12 := sorry

end john_max_questions_correct_l2311_231150


namespace minimum_value_l2311_231147

noncomputable def f : ℝ → ℝ
| x => if h : 0 < x ∧ x ≤ 1 then x^2 - x else
         if h : 1 < x ∧ x ≤ 2 then -2 * (x - 1)^2 + 6 * (x - 1) - 5
         else 0 -- extend as appropriate outside given ranges

noncomputable def g (x : ℝ) : ℝ := x - 1

theorem minimum_value (x_1 x_2 : ℝ) (h1 : 1 < x_1 ∧ x_1 ≤ 2) : 
  (x_1 - x_2)^2 + (f x_1 - g x_2)^2 = 49 / 128 :=
sorry

end minimum_value_l2311_231147


namespace smallest_prime_divisor_of_sum_l2311_231122

theorem smallest_prime_divisor_of_sum (h1 : ∃ k : ℕ, 3^19 = 2*k + 1)
                                      (h2 : ∃ l : ℕ, 11^13 = 2*l + 1) :
  Nat.minFac (3^19 + 11^13) = 2 := 
by
  sorry

end smallest_prime_divisor_of_sum_l2311_231122


namespace arcsin_one_half_eq_pi_six_l2311_231134

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_eq_pi_six_l2311_231134


namespace sqrt_sum_eq_six_l2311_231142

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l2311_231142


namespace simplify_expression_l2311_231119

variable (i : ℂ)

-- Define the conditions

def i_squared_eq_neg_one : Prop := i^2 = -1
def i_cubed_eq_neg_i : Prop := i^3 = i * i^2 ∧ i^3 = -i
def i_fourth_eq_one : Prop := i^4 = (i^2)^2 ∧ i^4 = 1
def i_fifth_eq_i : Prop := i^5 = i * i^4 ∧ i^5 = i

-- Define the proof problem

theorem simplify_expression (h1 : i_squared_eq_neg_one i) (h2 : i_cubed_eq_neg_i i) (h3 : i_fourth_eq_one i) (h4 : i_fifth_eq_i i) : 
  i + i^2 + i^3 + i^4 + i^5 = i := 
  by sorry

end simplify_expression_l2311_231119


namespace siamese_cats_initially_l2311_231124

theorem siamese_cats_initially (house_cats: ℕ) (cats_sold: ℕ) (cats_left: ℕ) (initial_siamese: ℕ) :
  house_cats = 5 → 
  cats_sold = 10 → 
  cats_left = 8 → 
  (initial_siamese + house_cats - cats_sold = cats_left) → 
  initial_siamese = 13 :=
by
  intros h1 h2 h3 h4
  sorry

end siamese_cats_initially_l2311_231124


namespace percent_of_x_is_y_in_terms_of_z_l2311_231118

theorem percent_of_x_is_y_in_terms_of_z (x y z : ℝ) (h1 : 0.7 * (x - y) = 0.3 * (x + y))
    (h2 : 0.6 * (x + z) = 0.4 * (y - z)) : y / x = 0.4 :=
  sorry

end percent_of_x_is_y_in_terms_of_z_l2311_231118


namespace regular_polygon_sides_and_exterior_angle_l2311_231111

theorem regular_polygon_sides_and_exterior_angle (n : ℕ) (exterior_sum : ℝ) :
  (180 * (n - 2) = 360 + exterior_sum) → (exterior_sum = 360) → n = 6 ∧ (360 / n = 60) :=
by
  intro h1 h2
  sorry

end regular_polygon_sides_and_exterior_angle_l2311_231111


namespace fruit_store_initial_quantities_l2311_231146

-- Definitions from conditions:
def total_fruit (a b c : ℕ) := a + b + c = 275
def sold_apples (a : ℕ) := a - 30
def added_peaches (b : ℕ) := b + 45
def sold_pears (c : ℕ) := c - c / 4
def final_ratio (a b c : ℕ) := (sold_apples a) / 4 = (added_peaches b) / 3 ∧ (added_peaches b) / 3 = (sold_pears c) / 2

-- The proof problem:
theorem fruit_store_initial_quantities (a b c : ℕ) (h1 : total_fruit a b c) 
  (h2 : final_ratio a b c) : a = 150 ∧ b = 45 ∧ c = 80 :=
sorry

end fruit_store_initial_quantities_l2311_231146


namespace non_consecutive_heads_probability_l2311_231104

-- Define the total number of basic events (n).
def total_events : ℕ := 2^4

-- Define the number of events where heads do not appear consecutively (m).
def non_consecutive_heads_events : ℕ := 1 + (Nat.choose 4 1) + (Nat.choose 3 2)

-- Define the probability of heads not appearing consecutively.
def probability_non_consecutive_heads : ℚ := non_consecutive_heads_events / total_events

-- The theorem we seek to prove
theorem non_consecutive_heads_probability :
  probability_non_consecutive_heads = 1 / 2 :=
by
  sorry

end non_consecutive_heads_probability_l2311_231104


namespace fewer_onions_correct_l2311_231193

-- Define the quantities
def tomatoes : ℕ := 2073
def corn : ℕ := 4112
def onions : ℕ := 985

-- Calculate the total number of tomatoes and corn
def tomatoes_and_corn : ℕ := tomatoes + corn

-- Calculate the number of fewer onions
def fewer_onions : ℕ := tomatoes_and_corn - onions

-- State the theorem and provide the proof
theorem fewer_onions_correct : fewer_onions = 5200 :=
by
  -- The statement is proved directly by the calculations above
  -- Providing the actual proof is not necessary as per the guidelines
  sorry

end fewer_onions_correct_l2311_231193


namespace sum_of_angles_of_parallelepiped_diagonal_lt_pi_l2311_231151

/-- In a rectangular parallelepiped, if the main diagonal forms angles α, β, and γ with the three edges meeting at a vertex, then the sum of these angles is less than π. -/
theorem sum_of_angles_of_parallelepiped_diagonal_lt_pi {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (h_sum : 2 * α + 2 * β + 2 * γ < 2 * π) :
  α + β + γ < π := by
sorry

end sum_of_angles_of_parallelepiped_diagonal_lt_pi_l2311_231151


namespace find_b_and_sinA_find_sin_2A_plus_pi_over_4_l2311_231106

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (sinB : ℝ)

-- Conditions
def triangle_conditions :=
  (a > b) ∧
  (a = 5) ∧
  (c = 6) ∧
  (sinB = 3 / 5)

-- Question 1: Prove b = sqrt 13 and sin A = (3 * sqrt 13) / 13
theorem find_b_and_sinA (h : triangle_conditions a b c sinB) :
  b = Real.sqrt 13 ∧
  ∃ sinA : ℝ, sinA = (3 * Real.sqrt 13) / 13 :=
  sorry

-- Question 2: Prove sin (2A + π/4) = 7 * sqrt 2 / 26
theorem find_sin_2A_plus_pi_over_4 (h : triangle_conditions a b c sinB)
  (hb : b = Real.sqrt 13)
  (sinA : ℝ)
  (h_sinA : sinA = (3 * Real.sqrt 13) / 13) :
  ∃ sin2Aπ4 : ℝ, sin2Aπ4 = (7 * Real.sqrt 2) / 26 :=
  sorry

end find_b_and_sinA_find_sin_2A_plus_pi_over_4_l2311_231106


namespace dog_group_division_l2311_231176

theorem dog_group_division:
  let total_dogs := 12
  let group1_size := 4
  let group2_size := 5
  let group3_size := 3
  let Rocky_in_group1 := true
  let Bella_in_group2 := true
  (total_dogs == 12 ∧ group1_size == 4 ∧ group2_size == 5 ∧ group3_size == 3 ∧ Rocky_in_group1 ∧ Bella_in_group2) →
  (∃ ways: ℕ, ways = 4200)
  :=
  sorry

end dog_group_division_l2311_231176


namespace average_age_inhabitants_Campo_Verde_l2311_231117

theorem average_age_inhabitants_Campo_Verde
  (H M : ℕ)
  (ratio_h_m : H / M = 2 / 3)
  (avg_age_men : ℕ := 37)
  (avg_age_women : ℕ := 42) :
  ((37 * H + 42 * M) / (H + M) : ℕ) = 40 := 
sorry

end average_age_inhabitants_Campo_Verde_l2311_231117


namespace max_pancake_pieces_3_cuts_l2311_231189

open Nat

def P : ℕ → ℕ
| 0 => 1
| n => n * (n + 1) / 2 + 1

theorem max_pancake_pieces_3_cuts : P 3 = 7 := by
  have h0: P 0 = 1 := by rfl
  have h1: P 1 = 2 := by rfl
  have h2: P 2 = 4 := by rfl
  show P 3 = 7
  calc
    P 3 = 3 * (3 + 1) / 2 + 1 := by rfl
    _ = 3 * 4 / 2 + 1 := by rfl
    _ = 6 + 1 := by norm_num
    _ = 7 := by norm_num

end max_pancake_pieces_3_cuts_l2311_231189


namespace parametric_to_standard_l2311_231125

theorem parametric_to_standard (t a b x y : ℝ)
(h1 : x = (a / 2) * (t + 1 / t))
(h2 : y = (b / 2) * (t - 1 / t)) :
  (x^2 / a^2) - (y^2 / b^2) = 1 :=
by
  sorry

end parametric_to_standard_l2311_231125


namespace a5_value_l2311_231192

def sequence_sum (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem a5_value (a : ℕ → ℤ) (h : ∀ n : ℕ, 0 < n → sequence_sum n a = (1 / 2 : ℚ) * (a n : ℚ) + 1) :
  a 5 = 2 := by
  sorry

end a5_value_l2311_231192


namespace sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l2311_231168

theorem sufficient_but_not_necessary (a b : ℝ) (h : 0 < a ∧ a < b) : (1 / a) > (1 / b) :=
by
  sorry

theorem sufficient_but_not_necessary_rel (a b : ℝ) : 0 < a ∧ a < b ↔ (1 / a) > (1 / b) :=
by
  sorry

end sufficient_but_not_necessary_sufficient_but_not_necessary_rel_l2311_231168


namespace cost_of_12_roll_package_is_correct_l2311_231137

variable (cost_per_roll_package : ℝ)
variable (individual_cost_per_roll : ℝ := 1)
variable (number_of_rolls : ℕ := 12)
variable (percent_savings : ℝ := 0.25)

-- The definition of the total cost of the package
def total_cost_package := number_of_rolls * (individual_cost_per_roll - (percent_savings * individual_cost_per_roll))

-- The goal is to prove that the total cost of the package is $9
theorem cost_of_12_roll_package_is_correct : total_cost_package = 9 := 
by
  sorry

end cost_of_12_roll_package_is_correct_l2311_231137


namespace bob_investment_correct_l2311_231102

noncomputable def initial_investment_fundA : ℝ := 2000
noncomputable def interest_rate_fundA : ℝ := 0.12
noncomputable def initial_investment_fundB : ℝ := 1000
noncomputable def interest_rate_fundB : ℝ := 0.30
noncomputable def fundA_after_two_years := initial_investment_fundA * (1 + interest_rate_fundA)
noncomputable def fundB_after_two_years (B : ℝ) := B * (1 + interest_rate_fundB)^2
noncomputable def extra_value : ℝ := 549.9999999999998

theorem bob_investment_correct :
  fundA_after_two_years = fundB_after_two_years initial_investment_fundB + extra_value :=
by
  sorry

end bob_investment_correct_l2311_231102


namespace ratio_result_l2311_231144

theorem ratio_result (p q r s : ℚ) 
(h1 : p / q = 2) 
(h2 : q / r = 4 / 5) 
(h3 : r / s = 3) : 
  s / p = 5 / 24 :=
sorry

end ratio_result_l2311_231144


namespace shooter_prob_l2311_231180

variable (hit_prob : ℝ)
variable (miss_prob : ℝ := 1 - hit_prob)
variable (p1 : hit_prob = 0.85)
variable (independent_shots : true)

theorem shooter_prob :
  miss_prob * miss_prob * hit_prob = 0.019125 :=
by
  rw [p1]
  sorry

end shooter_prob_l2311_231180


namespace value_range_f_at_4_l2311_231184

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_range_f_at_4 (f : ℝ → ℝ)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 1 ≤ f (1) ∧ f (1) ≤ 3)
  (h3 : 2 ≤ f (2) ∧ f (2) ≤ 4)
  (h4 : -1 ≤ f (3) ∧ f (3) ≤ 1) :
  -21.75 ≤ f 4 ∧ f 4 ≤ 1 :=
sorry

end value_range_f_at_4_l2311_231184
