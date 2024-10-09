import Mathlib

namespace rows_in_initial_patios_l2054_205484

theorem rows_in_initial_patios (r c : ℕ) (h1 : r * c = 60) (h2 : (2 * c : ℚ) / r = 3 / 2) (h3 : (r + 5) * (c - 3) = 60) : r = 10 :=
sorry

end rows_in_initial_patios_l2054_205484


namespace inconsistent_proportion_l2054_205478

theorem inconsistent_proportion (a b : ℝ) (h1 : 3 * a = 5 * b) (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (a / b = 3 / 5) :=
sorry

end inconsistent_proportion_l2054_205478


namespace vitamin_C_relationship_l2054_205442

variables (A O G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + O + G = 275
def condition2 : Prop := 2 * A + 3 * O + 4 * G = 683

-- Rewrite the math proof problem statement
theorem vitamin_C_relationship (h1 : condition1 A O G) (h2 : condition2 A O G) : O + 2 * G = 133 :=
by {
  sorry
}

end vitamin_C_relationship_l2054_205442


namespace value_of_star_l2054_205417

theorem value_of_star :
  ∀ x : ℕ, 45 - (28 - (37 - (15 - x))) = 55 → x = 16 :=
by
  intro x
  intro h
  sorry

end value_of_star_l2054_205417


namespace distance_between_parallel_lines_l2054_205449

theorem distance_between_parallel_lines :
  let A := 3
  let B := 2
  let C1 := -1
  let C2 := 1 / 2
  let d := |C2 - C1| / Real.sqrt (A^2 + B^2)
  d = 3 / Real.sqrt 13 :=
by
  -- Proof goes here
  sorry

end distance_between_parallel_lines_l2054_205449


namespace number_of_integers_inequality_l2054_205461

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end number_of_integers_inequality_l2054_205461


namespace compute_expr_l2054_205485

theorem compute_expr : 6^2 - 4 * 5 + 2^2 = 20 := by
  sorry

end compute_expr_l2054_205485


namespace find_B_l2054_205464

theorem find_B (N : ℕ) (A B : ℕ) (H1 : N = 757000000 + A * 10000 + B * 1000 + 384) (H2 : N % 357 = 0) : B = 5 :=
sorry

end find_B_l2054_205464


namespace bonnie_roark_wire_length_ratio_l2054_205480

noncomputable def ratio_of_wire_lengths : ℚ :=
let bonnie_wire_per_piece := 8
let bonnie_pieces := 12
let bonnie_total_wire := bonnie_pieces * bonnie_wire_per_piece

let bonnie_side := bonnie_wire_per_piece
let bonnie_volume := bonnie_side^3

let roark_side := 2
let roark_volume := roark_side^3
let roark_cubes := bonnie_volume / roark_volume

let roark_wire_per_piece := 2
let roark_pieces_per_cube := 12
let roark_wire_per_cube := roark_pieces_per_cube * roark_wire_per_piece
let roark_total_wire := roark_cubes * roark_wire_per_cube

let ratio := bonnie_total_wire / roark_total_wire
ratio 

theorem bonnie_roark_wire_length_ratio :
  ratio_of_wire_lengths = (1 : ℚ) / 16 := 
sorry

end bonnie_roark_wire_length_ratio_l2054_205480


namespace prime_divisors_of_1890_l2054_205470

theorem prime_divisors_of_1890 : ∃ (S : Finset ℕ), (S.card = 4) ∧ (∀ p ∈ S, Nat.Prime p) ∧ 1890 = S.prod id :=
by
  sorry

end prime_divisors_of_1890_l2054_205470


namespace candidate_percentage_valid_votes_l2054_205490

theorem candidate_percentage_valid_votes (total_votes invalid_percentage valid_votes_received : ℕ) 
    (h_total_votes : total_votes = 560000)
    (h_invalid_percentage : invalid_percentage = 15)
    (h_valid_votes_received : valid_votes_received = 333200) :
    (valid_votes_received : ℚ) / (total_votes * (1 - invalid_percentage / 100) : ℚ) * 100 = 70 :=
by
  sorry

end candidate_percentage_valid_votes_l2054_205490


namespace driver_net_rate_of_pay_l2054_205433

theorem driver_net_rate_of_pay
  (hours : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ)
  (net_rate_of_pay : ℚ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_cost_per_gallon = 2.50)
  (h6 : net_rate_of_pay = 25) :
  net_rate_of_pay = (hours * speed * pay_per_mile - (hours * speed / fuel_efficiency) * gas_cost_per_gallon) / hours := 
by sorry

end driver_net_rate_of_pay_l2054_205433


namespace speed_of_man_upstream_l2054_205469

def speed_of_man_in_still_water : ℝ := 32
def speed_of_man_downstream : ℝ := 39

theorem speed_of_man_upstream (V_m V_s : ℝ) :
  V_m = speed_of_man_in_still_water →
  V_m + V_s = speed_of_man_downstream →
  V_m - V_s = 25 :=
sorry

end speed_of_man_upstream_l2054_205469


namespace ashok_avg_first_five_l2054_205426

-- Define the given conditions 
def avg (n : ℕ) (s : ℕ) : ℕ := s / n

def total_marks (average : ℕ) (num_subjects : ℕ) : ℕ := average * num_subjects

variables (avg_six_subjects : ℕ := 76)
variables (sixth_subject_marks : ℕ := 86)
variables (total_six_subjects : ℕ := total_marks avg_six_subjects 6)
variables (total_first_five_subjects : ℕ := total_six_subjects - sixth_subject_marks)
variables (avg_first_five_subjects : ℕ := avg 5 total_first_five_subjects)

-- State the theorem
theorem ashok_avg_first_five 
  (h1 : avg_six_subjects = 76)
  (h2 : sixth_subject_marks = 86)
  (h3 : avg_first_five_subjects = 74)
  : avg 5 (total_marks 76 6 - 86) = 74 := 
sorry

end ashok_avg_first_five_l2054_205426


namespace complement_union_l2054_205473

variable (x : ℝ)

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def P : Set ℝ := {x | x ≥ 2}

theorem complement_union (x : ℝ) : x ∈ U → (¬ (x ∈ M ∨ x ∈ P)) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complement_union_l2054_205473


namespace tangent_segment_length_l2054_205455

-- Setting up the necessary definitions and theorem.
def radius := 10
def seg1 := 4
def seg2 := 2

theorem tangent_segment_length :
  ∃ X : ℝ, X = 8 ∧
  (radius^2 = X^2 + ((X + seg1 + seg2) / 2)^2) :=
by
  sorry

end tangent_segment_length_l2054_205455


namespace find_number_of_piles_l2054_205494

theorem find_number_of_piles 
  (Q : ℕ) 
  (h1 : Q = Q) 
  (h2 : ∀ (piles : ℕ), piles = 3) 
  (total_coins : ℕ) 
  (h3 : total_coins = 30) 
  (e : 6 * Q = total_coins) :
  Q = 5 := 
by sorry

end find_number_of_piles_l2054_205494


namespace sequence_property_l2054_205402

def Sn (n : ℕ) (a : ℕ → ℕ) : ℕ := (Finset.range (n + 1)).sum a

theorem sequence_property (a : ℕ → ℕ) (h : ∀ n : ℕ, Sn (n + 1) a = 2 * a n + 1) : a 3 = 2 :=
sorry

end sequence_property_l2054_205402


namespace first_month_sale_eq_6435_l2054_205454

theorem first_month_sale_eq_6435 (s2 s3 s4 s5 s6 : ℝ)
  (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) (h5 : s5 = 6562) (h6 : s6 = 7391)
  (avg : ℝ) (h_avg : avg = 6900) :
  let total_sales := 6 * avg
  let other_months_sales := s2 + s3 + s4 + s5 + s6
  let first_month_sale := total_sales - other_months_sales
  first_month_sale = 6435 :=
by
  sorry

end first_month_sale_eq_6435_l2054_205454


namespace sugar_recipes_l2054_205409

theorem sugar_recipes (container_sugar recipe_sugar : ℚ) 
  (h1 : container_sugar = 56 / 3) 
  (h2 : recipe_sugar = 3 / 2) :
  container_sugar / recipe_sugar = 112 / 9 := sorry

end sugar_recipes_l2054_205409


namespace tan_alpha_value_l2054_205425

open Real

theorem tan_alpha_value 
  (α : ℝ) 
  (hα_range : 0 < α ∧ α < π) 
  (h_cos_alpha : cos α = -3/5) :
  tan α = -4/3 := 
by
  sorry

end tan_alpha_value_l2054_205425


namespace difference_of_squares_l2054_205427

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
by
  sorry

end difference_of_squares_l2054_205427


namespace angle_B_is_40_degrees_l2054_205458

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end angle_B_is_40_degrees_l2054_205458


namespace solution_pairs_l2054_205471

def equation (r p : ℤ) : Prop := r^2 - r * (p + 6) + p^2 + 5 * p + 6 = 0

theorem solution_pairs :
  ∀ (r p : ℤ),
    equation r p ↔ (r = 3 ∧ p = 1) ∨ (r = 4 ∧ p = 1) ∨ 
                    (r = 0 ∧ p = -2) ∨ (r = 4 ∧ p = -2) ∨ 
                    (r = 0 ∧ p = -3) ∨ (r = 3 ∧ p = -3) :=
by
  sorry

end solution_pairs_l2054_205471


namespace exists_same_color_rectangle_l2054_205446

variable (coloring : ℕ × ℕ → Fin 3)

theorem exists_same_color_rectangle :
  (∃ (r1 r2 r3 r4 c1 c2 c3 c4 : ℕ), 
    r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
    c1 ≠ c2 ∧ 
    coloring (4, 82) = 4 ∧ 
    coloring (r1, c1) = coloring (r1, c2) ∧ coloring (r1, c2) = coloring (r2, c1) ∧ 
    coloring (r2, c1) = coloring (r2, c2)) :=
sorry

end exists_same_color_rectangle_l2054_205446


namespace lengths_of_trains_l2054_205435

noncomputable def km_per_hour_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def length_of_train (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem lengths_of_trains (Va Vb : ℝ) : Va = 60 ∧ Vb < Va ∧ length_of_train (km_per_hour_to_m_per_s Va) 42 = (700 : ℝ) 
    → length_of_train (km_per_hour_to_m_per_s Vb * (42 / 56)) 56 = (700 : ℝ) :=
by
  intros h
  sorry

end lengths_of_trains_l2054_205435


namespace evaluate_expression_l2054_205422

theorem evaluate_expression : 
  (1 / 10 : ℝ) + (2 / 20 : ℝ) - (3 / 60 : ℝ) = 0.15 :=
by
  sorry

end evaluate_expression_l2054_205422


namespace solution_inequality_part1_solution_inequality_part2_l2054_205489

noncomputable def f (x : ℝ) : ℝ := abs (x - 2)
noncomputable def g (x m : ℝ) : ℝ := -abs (x + 7) + 3 * m

theorem solution_inequality_part1 (x : ℝ) :
  (f x + x^2 - 4 > 0) ↔ (x > 2 ∨ x < -1) :=
sorry

theorem solution_inequality_part2 (m : ℝ) :
  (∃ x : ℝ, f x < g x m) → (m > 3) :=
sorry

end solution_inequality_part1_solution_inequality_part2_l2054_205489


namespace ganesh_average_speed_l2054_205431

variable (D : ℝ) -- the distance between towns X and Y

theorem ganesh_average_speed :
  let time_x_to_y := D / 43
  let time_y_to_x := D / 34
  let total_distance := 2 * D
  let total_time := time_x_to_y + time_y_to_x
  let avg_speed := total_distance / total_time
  avg_speed = 37.97 := by
    sorry

end ganesh_average_speed_l2054_205431


namespace find_k_l2054_205463

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l2054_205463


namespace find_large_number_l2054_205492

theorem find_large_number (L S : ℕ) (h1 : L - S = 1515) (h2 : L = 16 * S + 15) : L = 1615 :=
sorry

end find_large_number_l2054_205492


namespace difference_of_squares_l2054_205476

-- Definition of the constants a and b as given in the problem
def a := 502
def b := 498

theorem difference_of_squares : a^2 - b^2 = 4000 := by
  sorry

end difference_of_squares_l2054_205476


namespace describe_S_l2054_205440

def S : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) }

theorem describe_S :
  S = { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) } := 
by
  -- proof is omitted
  sorry

end describe_S_l2054_205440


namespace integer_solutions_system_inequalities_l2054_205438

theorem integer_solutions_system_inequalities:
  {x : ℤ} → (2 * x - 1 < x + 1) → (1 - 2 * (x - 1) ≤ 3) → x = 0 ∨ x = 1 := 
by
  intros x h1 h2
  sorry

end integer_solutions_system_inequalities_l2054_205438


namespace sequence_properties_l2054_205437

theorem sequence_properties (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, (a n)^2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0) :
  a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_properties_l2054_205437


namespace submarine_rise_l2054_205453

theorem submarine_rise (initial_depth final_depth : ℤ) (h_initial : initial_depth = -27) (h_final : final_depth = -18) :
  final_depth - initial_depth = 9 :=
by
  rw [h_initial, h_final]
  norm_num 

end submarine_rise_l2054_205453


namespace find_a4_in_geometric_seq_l2054_205439

variable {q : ℝ} -- q is the common ratio of the geometric sequence

noncomputable def geometric_seq (q : ℝ) (n : ℕ) : ℝ := 16 * q ^ (n - 1)

theorem find_a4_in_geometric_seq (h1 : geometric_seq q 1 = 16)
  (h2 : geometric_seq q 6 = 2 * geometric_seq q 5 * geometric_seq q 7) :
  geometric_seq q 4 = 2 := 
  sorry

end find_a4_in_geometric_seq_l2054_205439


namespace compute_division_l2054_205406

theorem compute_division : 0.182 / 0.0021 = 86 + 14 / 21 :=
by
  sorry

end compute_division_l2054_205406


namespace factor_t_squared_minus_144_l2054_205412

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end factor_t_squared_minus_144_l2054_205412


namespace starting_number_of_range_l2054_205413

theorem starting_number_of_range (multiples: ℕ) (end_of_range: ℕ) (span: ℕ)
  (h1: multiples = 991) (h2: end_of_range = 10000) (h3: span = multiples * 10) :
  end_of_range - span = 90 := 
by 
  sorry

end starting_number_of_range_l2054_205413


namespace prove_inequality_l2054_205414

open Real

noncomputable def inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ 
  3 * (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1)

theorem prove_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  inequality a b c h1 h2 h3 := 
  sorry

end prove_inequality_l2054_205414


namespace cloth_gain_percentage_l2054_205410

theorem cloth_gain_percentage 
  (x : ℝ) -- x represents the cost price of 1 meter of cloth
  (CP : ℝ := 30 * x) -- CP of 30 meters of cloth
  (profit : ℝ := 10 * x) -- profit from selling 30 meters of cloth
  (SP : ℝ := CP + profit) -- selling price of 30 meters of cloth
  (gain_percentage : ℝ := (profit / CP) * 100) : 
  gain_percentage = 33.33 := 
sorry

end cloth_gain_percentage_l2054_205410


namespace father_current_age_l2054_205474

theorem father_current_age (F S : ℕ) 
  (h₁ : F - 6 = 5 * (S - 6)) 
  (h₂ : (F + 6) + (S + 6) = 78) : 
  F = 51 := 
sorry

end father_current_age_l2054_205474


namespace printers_ratio_l2054_205498

theorem printers_ratio (Rate_X : ℝ := 1 / 16) (Rate_Y : ℝ := 1 / 10) (Rate_Z : ℝ := 1 / 20) :
  let Time_X := 1 / Rate_X
  let Time_YZ := 1 / (Rate_Y + Rate_Z)
  (Time_X / Time_YZ) = 12 / 5 := by
  sorry

end printers_ratio_l2054_205498


namespace prob_same_color_seven_red_and_five_green_l2054_205429

noncomputable def probability_same_color (red_plat : ℕ) (green_plat : ℕ) : ℚ :=
  let total_plates := red_plat + green_plat
  let total_pairs := (total_plates.choose 2) -- total ways to select 2 plates
  let red_pairs := (red_plat.choose 2) -- ways to select 2 red plates
  let green_pairs := (green_plat.choose 2) -- ways to select 2 green plates
  (red_pairs + green_pairs) / total_pairs

theorem prob_same_color_seven_red_and_five_green :
  probability_same_color 7 5 = 31 / 66 :=
by
  sorry

end prob_same_color_seven_red_and_five_green_l2054_205429


namespace calculate_polygon_sides_l2054_205481

-- Let n be the number of sides of the regular polygon with each exterior angle of 18 degrees
def regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 18 ∧ n * exterior_angle = 360

theorem calculate_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  regular_polygon_sides n exterior_angle → n = 20 :=
by
  intro h
  sorry

end calculate_polygon_sides_l2054_205481


namespace eel_species_count_l2054_205418

theorem eel_species_count (sharks eels whales total : ℕ)
    (h_sharks : sharks = 35)
    (h_whales : whales = 5)
    (h_total : total = 55)
    (h_species_sum : sharks + eels + whales = total) : eels = 15 :=
by
  -- Proof goes here
  sorry

end eel_species_count_l2054_205418


namespace fill_bathtub_time_l2054_205408

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end fill_bathtub_time_l2054_205408


namespace sawyer_saw_octopuses_l2054_205468

def number_of_legs := 40
def legs_per_octopus := 8

theorem sawyer_saw_octopuses : number_of_legs / legs_per_octopus = 5 := 
by
  sorry

end sawyer_saw_octopuses_l2054_205468


namespace sn_values_l2054_205424

noncomputable def s (x1 x2 x3 : ℂ) (n : ℕ) : ℂ :=
  x1^n + x2^n + x3^n

theorem sn_values (p q x1 x2 x3 : ℂ) (h_root1 : x1^3 + p * x1 + q = 0)
                    (h_root2 : x2^3 + p * x2 + q = 0)
                    (h_root3 : x3^3 + p * x3 + q = 0) :
  s x1 x2 x3 2 = -3 * q ∧
  s x1 x2 x3 3 = 3 * q^2 ∧
  s x1 x2 x3 4 = 2 * p^2 ∧
  s x1 x2 x3 5 = 5 * p * q ∧
  s x1 x2 x3 6 = -2 * p^3 + 3 * q^2 ∧
  s x1 x2 x3 7 = -7 * p^2 * q ∧
  s x1 x2 x3 8 = 2 * p^4 - 8 * p * q^2 ∧
  s x1 x2 x3 9 = 9 * p^3 * q - 3 * q^3 ∧
  s x1 x2 x3 10 = -2 * p^5 + 15 * p^2 * q^2 :=
by {
  sorry
}

end sn_values_l2054_205424


namespace find_possible_values_of_y_l2054_205403

theorem find_possible_values_of_y (x : ℝ) (h : x^2 + 9 * (3 * x / (x - 3))^2 = 90) :
  y = (x - 3)^3 * (x + 2) / (2 * x - 4) → y = 28 / 3 ∨ y = 169 :=
by
  sorry

end find_possible_values_of_y_l2054_205403


namespace product_eq_one_l2054_205450

noncomputable def f (x : ℝ) : ℝ := |Real.logb 3 x|

theorem product_eq_one (a b : ℝ) (h_diff : a ≠ b) (h_eq : f a = f b) : a * b = 1 := by
  sorry

end product_eq_one_l2054_205450


namespace red_balls_in_box_l2054_205404

theorem red_balls_in_box (initial_red_balls added_red_balls : ℕ) (initial_blue_balls : ℕ) 
  (h_initial : initial_red_balls = 5) (h_added : added_red_balls = 2) : 
  initial_red_balls + added_red_balls = 7 :=
by {
  sorry
}

end red_balls_in_box_l2054_205404


namespace minimum_value_expression_l2054_205401

theorem minimum_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ k, k = 729 ∧ ∀ x y z, 0 < x → 0 < y → 0 < z → k ≤ (x^2 + 4*x + 4) * (y^2 + 4*y + 4) * (z^2 + 4*z + 4) / (x * y * z) :=
by 
  use 729
  sorry

end minimum_value_expression_l2054_205401


namespace arnel_kept_fifty_pencils_l2054_205436

theorem arnel_kept_fifty_pencils
    (num_boxes : ℕ) (pencils_each_box : ℕ) (friends : ℕ) (pencils_each_friend : ℕ) (total_pencils : ℕ)
    (boxes_pencils : ℕ) (friends_pencils : ℕ) :
    num_boxes = 10 →
    pencils_each_box = 5 →
    friends = 5 →
    pencils_each_friend = 8 →
    friends_pencils = friends * pencils_each_friend →
    boxes_pencils = num_boxes * pencils_each_box →
    total_pencils = boxes_pencils + friends_pencils →
    (total_pencils - friends_pencils) = 50 :=
by
    sorry

end arnel_kept_fifty_pencils_l2054_205436


namespace find_dividend_l2054_205493

theorem find_dividend (dividend divisor quotient : ℕ) 
  (h_sum : dividend + divisor + quotient = 103)
  (h_quotient : quotient = 3)
  (h_divisor : divisor = dividend / quotient) : 
  dividend = 75 :=
by
  rw [h_quotient, h_divisor] at h_sum
  sorry

end find_dividend_l2054_205493


namespace fraction_inequality_l2054_205459

theorem fraction_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a < b) (h2 : c < d) : (a + c) / (b + c) < (a + d) / (b + d) :=
by
  sorry

end fraction_inequality_l2054_205459


namespace no_tiling_10x10_1x4_l2054_205472

-- Define the problem using the given conditions
def checkerboard_tiling (n k : ℕ) : Prop :=
  ∃ t : ℕ, t * k = n * n ∧ n % k = 0

-- Prove that it is impossible to tile a 10x10 board with 1x4 tiles
theorem no_tiling_10x10_1x4 : ¬ checkerboard_tiling 10 4 :=
sorry

end no_tiling_10x10_1x4_l2054_205472


namespace no_solution_a4_plus_6_eq_b3_mod_13_l2054_205411

theorem no_solution_a4_plus_6_eq_b3_mod_13 :
  ¬ ∃ (a b : ℤ), (a^4 + 6) % 13 = b^3 % 13 :=
by
  sorry

end no_solution_a4_plus_6_eq_b3_mod_13_l2054_205411


namespace new_student_weight_l2054_205416

theorem new_student_weight :
  ∀ (W : ℝ) (total_weight_19 : ℝ) (total_weight_20 : ℝ),
    total_weight_19 = 19 * 15 →
    total_weight_20 = 20 * 14.8 →
    total_weight_19 + W = total_weight_20 →
    W = 11 :=
by
  intros W total_weight_19 total_weight_20 h1 h2 h3
  -- Skipping the proof as instructed
  sorry

end new_student_weight_l2054_205416


namespace time_to_reach_ticket_window_l2054_205457

-- Define the conditions as per the problem
def rate_kit : ℕ := 2 -- feet per minute (rate)
def remaining_distance : ℕ := 210 -- feet

-- Goal: To prove the time required to reach the ticket window is 105 minutes
theorem time_to_reach_ticket_window : remaining_distance / rate_kit = 105 :=
by sorry

end time_to_reach_ticket_window_l2054_205457


namespace find_a_b_max_profit_allocation_l2054_205421

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (a * Real.log x) / x + 5 / x - b

theorem find_a_b :
  (∃ (a b : ℝ), f 1 a b = 5 ∧ f 10 a b = 16.515) :=
sorry

noncomputable def g (x : ℝ) := 2 * Real.sqrt x / x

noncomputable def profit (x : ℝ) := x * (5 * Real.log x / x + 5 / x) + (50 - x) * (2 * Real.sqrt (50 - x) / (50 - x))

theorem max_profit_allocation :
  (∃ (x : ℝ), 10 ≤ x ∧ x ≤ 40 ∧ ∀ y, (10 ≤ y ∧ y ≤ 40) → profit x ≥ profit y)
  ∧ profit 25 = 31.09 :=
sorry

end find_a_b_max_profit_allocation_l2054_205421


namespace john_moves_correct_total_weight_l2054_205486

noncomputable def johns_total_weight_moved : ℝ := 5626.398

theorem john_moves_correct_total_weight :
  let initial_back_squat : ℝ := 200
  let back_squat_increase : ℝ := 50
  let front_squat_ratio : ℝ := 0.8
  let back_squat_set_increase : ℝ := 0.05
  let front_squat_ratio_increase : ℝ := 0.04
  let front_squat_effort : ℝ := 0.9
  let deadlift_ratio : ℝ := 1.2
  let deadlift_effort : ℝ := 0.85
  let deadlift_set_increase : ℝ := 0.03
  let updated_back_squat := (initial_back_squat + back_squat_increase)
  let back_squat_set_1 := updated_back_squat
  let back_squat_set_2 := back_squat_set_1 * (1 + back_squat_set_increase)
  let back_squat_set_3 := back_squat_set_2 * (1 + back_squat_set_increase)
  let back_squat_total := 3 * (back_squat_set_1 + back_squat_set_2 + back_squat_set_3)
  let updated_front_squat := updated_back_squat * front_squat_ratio
  let front_squat_set_1 := updated_front_squat * front_squat_effort
  let front_squat_set_2 := (updated_front_squat * (1 + front_squat_ratio_increase)) * front_squat_effort
  let front_squat_set_3 := (updated_front_squat * (1 + 2 * front_squat_ratio_increase)) * front_squat_effort
  let front_squat_total := 3 * (front_squat_set_1 + front_squat_set_2 + front_squat_set_3)
  let updated_deadlift := updated_back_squat * deadlift_ratio
  let deadlift_set_1 := updated_deadlift * deadlift_effort
  let deadlift_set_2 := (updated_deadlift * (1 + deadlift_set_increase)) * deadlift_effort
  let deadlift_set_3 := (updated_deadlift * (1 + 2 * deadlift_set_increase)) * deadlift_effort
  let deadlift_total := 2 * (deadlift_set_1 + deadlift_set_2 + deadlift_set_3)
  (back_squat_total + front_squat_total + deadlift_total) = johns_total_weight_moved :=
by sorry

end john_moves_correct_total_weight_l2054_205486


namespace sum_of_squares_of_roots_l2054_205497

theorem sum_of_squares_of_roots :
  (∀ y : ℝ, y ^ 3 - 8 * y ^ 2 + 9 * y + 2 = 0 → y ≥ 0) →
  let s : ℝ := 8
  let p : ℝ := 9
  let q : ℝ := -2
  (s ^ 2 - 2 * p = 46) :=
by
  -- Placeholders for definitions extracted from the conditions
  -- and additional necessary let-bindings from Vieta's formulas
  intro h
  sorry

end sum_of_squares_of_roots_l2054_205497


namespace solve_problem_l2054_205448

noncomputable def problem_statement : Prop :=
  (2015 : ℝ) / (2015^2 - 2016 * 2014) = 2015

theorem solve_problem : problem_statement := by
  -- Proof steps will be filled in here.
  sorry

end solve_problem_l2054_205448


namespace john_average_speed_l2054_205482

theorem john_average_speed :
  let distance_uphill := 2 -- distance in km
  let distance_downhill := 2 -- distance in km
  let time_uphill := 45 / 60 -- time in hours (45 minutes)
  let time_downhill := 15 / 60 -- time in hours (15 minutes)
  let total_distance := distance_uphill + distance_downhill -- total distance in km
  let total_time := time_uphill + time_downhill -- total time in hours
  total_distance / total_time = 4 := by
  sorry

end john_average_speed_l2054_205482


namespace percentage_of_tip_is_25_l2054_205456

-- Definitions of the costs
def cost_samosas : ℕ := 3 * 2
def cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2

-- Definition of total food cost
def total_food_cost : ℕ := cost_samosas + cost_pakoras + cost_mango_lassi

-- Definition of the total meal cost including tax
def total_meal_cost_with_tax : ℕ := 25

-- Definition of the tip
def tip : ℕ := total_meal_cost_with_tax - total_food_cost

-- Definition of the percentage of the tip
def percentage_tip : ℕ := (tip * 100) / total_food_cost

-- The theorem to be proved
theorem percentage_of_tip_is_25 :
  percentage_tip = 25 :=
by
  sorry

end percentage_of_tip_is_25_l2054_205456


namespace eval_expression_correct_l2054_205477

def eval_expression : ℤ :=
  -(-1) + abs (-1)

theorem eval_expression_correct : eval_expression = 2 :=
  by
    sorry

end eval_expression_correct_l2054_205477


namespace part1_part2_l2054_205479
-- Importing the entire Mathlib library for required definitions

-- Define the sequence a_n with the conditions given in the problem
def a : ℕ → ℚ
| 0       => 1
| (n + 1) => a n / (2 * a n + 1)

-- Prove the given claims
theorem part1 (n : ℕ) : a n = (1 : ℚ) / (2 * n + 1) :=
sorry

def b (n : ℕ) : ℚ := a n * a (n + 1)

-- The sum of the first n terms of the sequence b_n is denoted as T_n
def T : ℕ → ℚ
| 0       => 0
| (n + 1) => T n + b n

-- Prove the given sum
theorem part2 (n : ℕ) : T n = (n : ℚ) / (2 * n + 1) :=
sorry

end part1_part2_l2054_205479


namespace cricket_game_initial_overs_l2054_205445

theorem cricket_game_initial_overs
    (run_rate_initial : ℝ)
    (run_rate_remaining : ℝ)
    (remaining_overs : ℕ)
    (target_score : ℝ)
    (initial_overs : ℕ) :
    run_rate_initial = 3.2 →
    run_rate_remaining = 5.25 →
    remaining_overs = 40 →
    target_score = 242 →
    initial_overs = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_game_initial_overs_l2054_205445


namespace immigration_per_year_l2054_205496

-- Definitions based on the initial conditions
def initial_population : ℕ := 100000
def birth_rate : ℕ := 60 -- this represents 60%
def duration_years : ℕ := 10
def emigration_per_year : ℕ := 2000
def final_population : ℕ := 165000

-- Theorem statement: The number of people that immigrated per year
theorem immigration_per_year (immigration_per_year : ℕ) :
  immigration_per_year = 2500 :=
  sorry

end immigration_per_year_l2054_205496


namespace total_dots_not_visible_l2054_205405

theorem total_dots_not_visible
    (num_dice : ℕ)
    (dots_per_die : ℕ)
    (visible_faces : ℕ → ℕ)
    (visible_faces_count : ℕ)
    (total_dots : ℕ)
    (dots_visible : ℕ) :
    num_dice = 4 →
    dots_per_die = 21 →
    visible_faces 0 = 1 →
    visible_faces 1 = 2 →
    visible_faces 2 = 2 →
    visible_faces 3 = 3 →
    visible_faces 4 = 4 →
    visible_faces 5 = 5 →
    visible_faces 6 = 6 →
    visible_faces 7 = 6 →
    visible_faces_count = 8 →
    total_dots = num_dice * dots_per_die →
    dots_visible = visible_faces 0 + visible_faces 1 + visible_faces 2 + visible_faces 3 + visible_faces 4 + visible_faces 5 + visible_faces 6 + visible_faces 7 →
    total_dots - dots_visible = 55 := by
  sorry

end total_dots_not_visible_l2054_205405


namespace max_diff_distance_l2054_205430

def hyperbola_right_branch (x y : ℝ) : Prop := 
  (x^2 / 9) - (y^2 / 16) = 1 ∧ x > 0

def circle_1 (x y : ℝ) : Prop := 
  (x + 5)^2 + y^2 = 4

def circle_2 (x y : ℝ) : Prop := 
  (x - 5)^2 + y^2 = 1

theorem max_diff_distance 
  (P M N : ℝ × ℝ) 
  (hp : hyperbola_right_branch P.fst P.snd) 
  (hm : circle_1 M.fst M.snd) 
  (hn : circle_2 N.fst N.snd) :
  |dist P M - dist P N| ≤ 9 := 
sorry

end max_diff_distance_l2054_205430


namespace remaining_string_length_l2054_205475

theorem remaining_string_length (original_length : ℝ) (given_to_Minyoung : ℝ) (fraction_used : ℝ) :
  original_length = 70 →
  given_to_Minyoung = 27 →
  fraction_used = 7/9 →
  abs (original_length - given_to_Minyoung - fraction_used * (original_length - given_to_Minyoung) - 9.56) < 0.01 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end remaining_string_length_l2054_205475


namespace smallest_multiple_3_4_5_l2054_205483

theorem smallest_multiple_3_4_5 : ∃ (n : ℕ), (∀ (m : ℕ), (m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0) → n ≤ m) ∧ n = 60 := 
sorry

end smallest_multiple_3_4_5_l2054_205483


namespace parabola_translation_vertex_l2054_205432

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation of the parabola
def translated_parabola (x : ℝ) : ℝ := (x + 3)^2 - 4*(x + 3) + 2 - 2 -- Adjust x + 3 for shift left and subtract 2 for shift down

-- The vertex coordinates function
def vertex_coords (f : ℝ → ℝ) (x_vertex : ℝ) : ℝ × ℝ := (x_vertex, f x_vertex)

-- Define the original vertex
def original_vertex : ℝ × ℝ := vertex_coords original_parabola 2

-- Define the translated vertex we expect
def expected_translated_vertex : ℝ × ℝ := vertex_coords translated_parabola (-1)

-- Statement of the problem
theorem parabola_translation_vertex :
  expected_translated_vertex = (-1, -4) :=
  sorry

end parabola_translation_vertex_l2054_205432


namespace integer_values_of_n_satisfy_inequality_l2054_205444

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end integer_values_of_n_satisfy_inequality_l2054_205444


namespace ratio_circle_to_triangle_area_l2054_205467

theorem ratio_circle_to_triangle_area 
  (h d : ℝ) 
  (h_pos : 0 < h) 
  (d_pos : 0 < d) 
  (R : ℝ) 
  (R_def : R = h / 2) :
  (π * R^2) / (1/2 * h * d) = (π * h) / (2 * d) :=
by sorry

end ratio_circle_to_triangle_area_l2054_205467


namespace batsman_average_increase_l2054_205428

theorem batsman_average_increase
  (A : ℤ)
  (h1 : (16 * A + 85) / 17 = 37) :
  37 - A = 3 :=
by
  sorry

end batsman_average_increase_l2054_205428


namespace simplify_expression_correct_l2054_205423

noncomputable def simplify_expression (m n : ℝ) : ℝ :=
  ( (2 - n) / (n - 1) + 4 * ((m - 1) / (m - 2)) ) /
  ( n^2 * ((m - 1) / (n - 1)) + m^2 * ((2 - n) / (m - 2)) )

theorem simplify_expression_correct :
  simplify_expression (Real.rpow 400 (1/4)) (Real.sqrt 5) = (Real.sqrt 5) / 5 := 
sorry

end simplify_expression_correct_l2054_205423


namespace number_of_integer_terms_l2054_205441

noncomputable def count_integer_terms_in_sequence (n : ℕ) (k : ℕ) (a : ℕ) : ℕ :=
  if h : a = k * 3 ^ n then n + 1 else 0

theorem number_of_integer_terms :
  count_integer_terms_in_sequence 5 (2^3 * 5) 9720 = 6 :=
by sorry

end number_of_integer_terms_l2054_205441


namespace min_value_expression_l2054_205420

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + 1 / b) * (b + 4 / a) ≥ 9 :=
by
  sorry

end min_value_expression_l2054_205420


namespace find_b_l2054_205460

noncomputable def a_and_b_integers_and_factor (a b : ℤ) : Prop :=
  ∀ (x : ℝ), (x^2 - x - 1) * (a*x^3 + b*x^2 - x + 1) = 0

theorem find_b (a b : ℤ) (h : a_and_b_integers_and_factor a b) : b = -1 :=
by 
  sorry

end find_b_l2054_205460


namespace find_general_term_l2054_205465

theorem find_general_term (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2 * a n + n^2) :
  ∀ n, a n = 7 * 2^(n - 1) - n^2 - 2 * n - 3 :=
by
  sorry

end find_general_term_l2054_205465


namespace sector_angle_l2054_205434

theorem sector_angle (r α : ℝ) (h₁ : 2 * r + α * r = 4) (h₂ : (1 / 2) * α * r^2 = 1) : α = 2 :=
sorry

end sector_angle_l2054_205434


namespace percent_profit_l2054_205419

theorem percent_profit (C S : ℝ) (h : 55 * C = 50 * S) : 
  100 * ((S - C) / C) = 10 :=
by
  sorry

end percent_profit_l2054_205419


namespace rectangle_width_l2054_205452

theorem rectangle_width (w : ℝ) (h_length : w * 2 = l) (h_area : w * l = 50) : w = 5 :=
by
  sorry

end rectangle_width_l2054_205452


namespace value_of_a8_l2054_205451

theorem value_of_a8 (a : ℕ → ℝ) :
  (1 + x) ^ 10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x) ^ 2 + a 3 * (1 - x) ^ 3 +
  a 4 * (1 - x) ^ 4 + a 5 * (1 - x) ^ 5 + a 6 * (1 - x) ^ 6 + a 7 * (1 - x) ^ 7 + 
  a 8 * (1 - x) ^ 8 + a 9 * (1 - x) ^ 9 + a 10 * (1 - x) ^ 10 → 
  a 8 = 180 :=
by
  sorry

end value_of_a8_l2054_205451


namespace sum_of_three_rel_prime_pos_integers_l2054_205488

theorem sum_of_three_rel_prime_pos_integers (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h_rel_prime_ab : Nat.gcd a b = 1) (h_rel_prime_ac : Nat.gcd a c = 1) (h_rel_prime_bc : Nat.gcd b c = 1)
  (h_product : a * b * c = 2700) :
  a + b + c = 56 := by
  sorry

end sum_of_three_rel_prime_pos_integers_l2054_205488


namespace negation_of_P_l2054_205415

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- Define the negation of P
def not_P : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

-- The theorem statement
theorem negation_of_P : ¬ P = not_P :=
by
  sorry

end negation_of_P_l2054_205415


namespace smallest_nonprime_with_large_primes_l2054_205495

theorem smallest_nonprime_with_large_primes
  (n : ℕ)
  (h1 : n > 1)
  (h2 : ¬ Prime n)
  (h3 : ∀ p : ℕ, Prime p → p ∣ n → p ≥ 20) :
  660 < n ∧ n ≤ 670 :=
sorry

end smallest_nonprime_with_large_primes_l2054_205495


namespace baby_guppies_calculation_l2054_205407

-- Define the problem in Lean
theorem baby_guppies_calculation :
  ∀ (initial_guppies first_sighting two_days_gups total_guppies_after_two_days : ℕ), 
  initial_guppies = 7 →
  first_sighting = 36 →
  total_guppies_after_two_days = 52 →
  total_guppies_after_two_days = initial_guppies + first_sighting + two_days_gups →
  two_days_gups = 9 :=
by
  intros initial_guppies first_sighting two_days_gups total_guppies_after_two_days
  intros h_initial h_first h_total h_eq
  sorry

end baby_guppies_calculation_l2054_205407


namespace sum_a_m_eq_2_pow_n_b_n_l2054_205487

noncomputable def a_n (x : ℝ) (n : ℕ) : ℝ := (Finset.range (n + 1)).sum (λ k => x ^ k)

noncomputable def b_n (x : ℝ) (n : ℕ) : ℝ := 
  (Finset.range (n + 1)).sum (λ k => ((x + 1) / 2) ^ k)

theorem sum_a_m_eq_2_pow_n_b_n 
  (x : ℝ) (n : ℕ) : 
  (Finset.range (n + 1)).sum (λ m => a_n x m * Nat.choose (n + 1) (m + 1)) = 2 ^ n * b_n x n :=
by
  sorry

end sum_a_m_eq_2_pow_n_b_n_l2054_205487


namespace supremum_neg_frac_l2054_205466

noncomputable def supremum_expression (a b : ℝ) : ℝ :=
  - (1 / (2 * a) + 2 / b)

theorem supremum_neg_frac {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  ∃ M : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ M)
  ∧ (∀ N : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ N) → M ≤ N)
  ∧ M = -9 / 2 :=
sorry

end supremum_neg_frac_l2054_205466


namespace remainder_of_7_pow_4_div_100_l2054_205443

theorem remainder_of_7_pow_4_div_100 :
  (7^4) % 100 = 1 := 
sorry

end remainder_of_7_pow_4_div_100_l2054_205443


namespace max_marks_l2054_205400

theorem max_marks (M : ℝ) (score passing shortfall : ℝ)
  (h_score : score = 212)
  (h_shortfall : shortfall = 44)
  (h_passing : passing = score + shortfall)
  (h_pass_cond : passing = 0.4 * M) :
  M = 640 :=
by
  sorry

end max_marks_l2054_205400


namespace intersection_A_B_l2054_205491

def A : Set ℝ := { x | abs x ≤ 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x ≤ 1 } :=
sorry

end intersection_A_B_l2054_205491


namespace heartsuit_ratio_l2054_205499

-- Define the operation \heartsuit
def heartsuit (n m : ℕ) : ℕ := n^3 * m^2

-- The proposition we want to prove
theorem heartsuit_ratio :
  heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l2054_205499


namespace apples_to_pears_l2054_205462

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end apples_to_pears_l2054_205462


namespace at_least_one_not_less_than_two_l2054_205447

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := sorry

end at_least_one_not_less_than_two_l2054_205447
