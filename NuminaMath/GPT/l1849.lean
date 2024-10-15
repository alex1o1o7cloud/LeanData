import Mathlib

namespace NUMINAMATH_GPT_system_of_equations_solution_l1849_184936

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 : ℝ), 
    (x1 + 2 * x2 = 10) ∧
    (3 * x1 + 2 * x2 + x3 = 23) ∧
    (x2 + 2 * x3 = 13) ∧
    (x1 = 4) ∧
    (x2 = 3) ∧
    (x3 = 5) :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l1849_184936


namespace NUMINAMATH_GPT_bucket_fill_proof_l1849_184918

variables (x y : ℕ)
def tank_capacity : ℕ := 4 * x

theorem bucket_fill_proof (hx: y = x + 4) (hy: 4 * x = 3 * y): tank_capacity x = 48 :=
by {
  -- Proof steps will be here, but are elided for now
  sorry 
}

end NUMINAMATH_GPT_bucket_fill_proof_l1849_184918


namespace NUMINAMATH_GPT_average_speed_l1849_184992

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end NUMINAMATH_GPT_average_speed_l1849_184992


namespace NUMINAMATH_GPT_find_focus_of_parabola_l1849_184998

-- Define the given parabola equation
def parabola_eqn (x : ℝ) : ℝ := -4 * x^2

-- Define a predicate to check if the point is the focus
def is_focus (x y : ℝ) := x = 0 ∧ y = -1 / 16

theorem find_focus_of_parabola :
  is_focus 0 (parabola_eqn 0) :=
sorry

end NUMINAMATH_GPT_find_focus_of_parabola_l1849_184998


namespace NUMINAMATH_GPT_find_a_l1849_184991

theorem find_a (a : ℝ) (h : (a + 3) = 0) : a = -3 :=
by sorry

end NUMINAMATH_GPT_find_a_l1849_184991


namespace NUMINAMATH_GPT_abc_divisibility_l1849_184960

theorem abc_divisibility (a b c : ℕ) (h1 : c ∣ a^b) (h2 : a ∣ b^c) (h3 : b ∣ c^a) : abc ∣ (a + b + c)^(a + b + c) := 
sorry

end NUMINAMATH_GPT_abc_divisibility_l1849_184960


namespace NUMINAMATH_GPT_weight_of_fish_in_barrel_l1849_184909

/-- 
Given a barrel with an initial weight of 54 kg when full of fish,
and a weight of 29 kg after removing half of the fish,
prove that the initial weight of the fish in the barrel was 50 kg.
-/
theorem weight_of_fish_in_barrel (B F : ℝ)
  (h1: B + F = 54)
  (h2: B + F / 2 = 29) : F = 50 := 
sorry

end NUMINAMATH_GPT_weight_of_fish_in_barrel_l1849_184909


namespace NUMINAMATH_GPT_vector_at_t_zero_l1849_184950

theorem vector_at_t_zero :
  ∃ a d : ℝ × ℝ, (a + d = (2, 5) ∧ a + 4 * d = (11, -7)) ∧ a = (-1, 9) ∧ a + 0 * d = (-1, 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_vector_at_t_zero_l1849_184950


namespace NUMINAMATH_GPT_car_z_mpg_decrease_l1849_184979

theorem car_z_mpg_decrease :
  let mpg_45 := 51
  let mpg_60 := 408 / 10
  let decrease := mpg_45 - mpg_60
  let percentage_decrease := (decrease / mpg_45) * 100
  percentage_decrease = 20 := by
  sorry

end NUMINAMATH_GPT_car_z_mpg_decrease_l1849_184979


namespace NUMINAMATH_GPT_max_expression_value_l1849_184980

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end NUMINAMATH_GPT_max_expression_value_l1849_184980


namespace NUMINAMATH_GPT_inequality_proof_l1849_184928

variable {a b : ℝ}

theorem inequality_proof (h : a > b) : 2 - a < 2 - b :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1849_184928


namespace NUMINAMATH_GPT_no_positive_integral_solution_l1849_184912

theorem no_positive_integral_solution :
  ¬ ∃ n : ℕ, n > 0 ∧ ∃ p : ℕ, Prime p ∧ n^2 - 45 * n + 520 = p :=
by {
  -- Since we only need the statement, we'll introduce the necessary steps without the full proof
  sorry
}

end NUMINAMATH_GPT_no_positive_integral_solution_l1849_184912


namespace NUMINAMATH_GPT_sequence_sum_periodic_l1849_184994

theorem sequence_sum_periodic (a : ℕ → ℕ) (a1 a8 : ℕ) :
  a 1 = 11 →
  a 8 = 12 →
  (∀ i, 1 ≤ i → i ≤ 6 → a i + a (i + 1) + a (i + 2) = 50) →
  (a 1 = 11 ∧ a 2 = 12 ∧ a 3 = 27 ∧ a 4 = 11 ∧ a 5 = 12 ∧ a 6 = 27 ∧ a 7 = 11 ∧ a 8 = 12) :=
by
  intros h1 h8 hsum
  sorry

end NUMINAMATH_GPT_sequence_sum_periodic_l1849_184994


namespace NUMINAMATH_GPT_coefficient_of_x_in_expression_l1849_184976

theorem coefficient_of_x_in_expression : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2) + 3 * (x + 4)
  ∃ k : ℤ, (expr = k * x + term) ∧ 
  (∃ coefficient_x : ℤ, coefficient_x = 8) := 
sorry

end NUMINAMATH_GPT_coefficient_of_x_in_expression_l1849_184976


namespace NUMINAMATH_GPT_maximize_area_l1849_184917

noncomputable def optimal_fencing (L W : ℝ) : Prop :=
  (2 * L + W = 1200) ∧ (∀ L1 W1, 2 * L1 + W1 = 1200 → L * W ≥ L1 * W1)

theorem maximize_area : ∃ L W, optimal_fencing L W ∧ L + W = 900 := sorry

end NUMINAMATH_GPT_maximize_area_l1849_184917


namespace NUMINAMATH_GPT_sum_arithmetic_series_eq_250500_l1849_184914

theorem sum_arithmetic_series_eq_250500 :
  let a1 := 2
  let d := 2
  let an := 1000
  let n := 500
  (a1 + (n-1) * d = an) →
  ((n * (a1 + an)) / 2 = 250500) :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_series_eq_250500_l1849_184914


namespace NUMINAMATH_GPT_Julio_limes_expense_l1849_184981

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end NUMINAMATH_GPT_Julio_limes_expense_l1849_184981


namespace NUMINAMATH_GPT_median_of_right_triangle_l1849_184964

theorem median_of_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) : 
  c / 2 = 5 :=
by
  rw [h3]
  norm_num

end NUMINAMATH_GPT_median_of_right_triangle_l1849_184964


namespace NUMINAMATH_GPT_octadecagon_diagonals_l1849_184922

def num_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem octadecagon_diagonals : num_of_diagonals 18 = 135 := by
  sorry

end NUMINAMATH_GPT_octadecagon_diagonals_l1849_184922


namespace NUMINAMATH_GPT_factor_as_complete_square_l1849_184954

theorem factor_as_complete_square (k : ℝ) : (∃ a : ℝ, x^2 + k*x + 9 = (x + a)^2) ↔ k = 6 ∨ k = -6 := 
sorry

end NUMINAMATH_GPT_factor_as_complete_square_l1849_184954


namespace NUMINAMATH_GPT_simplify_expr_l1849_184986

variable (x y : ℝ)

def expr (x y : ℝ) := (x + y) * (x - y) - y * (2 * x - y)

theorem simplify_expr :
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  expr x y = 2 - 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1849_184986


namespace NUMINAMATH_GPT_ratio_in_range_l1849_184906

theorem ratio_in_range {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end NUMINAMATH_GPT_ratio_in_range_l1849_184906


namespace NUMINAMATH_GPT_relationship_l1849_184926

noncomputable def a : ℝ := Real.log (Real.log Real.pi)
noncomputable def b : ℝ := Real.log Real.pi
noncomputable def c : ℝ := 2^Real.log Real.pi

theorem relationship (a b c : ℝ) (ha : a = Real.log (Real.log Real.pi)) (hb : b = Real.log Real.pi) (hc : c = 2^Real.log Real.pi)
: a < b ∧ b < c := 
by
  sorry

end NUMINAMATH_GPT_relationship_l1849_184926


namespace NUMINAMATH_GPT_expression_in_terms_of_p_and_q_l1849_184944

theorem expression_in_terms_of_p_and_q (x : ℝ) :
  let p := (1 - Real.cos x) * (1 + Real.sin x)
  let q := (1 + Real.cos x) * (1 - Real.sin x)
  (Real.cos x ^ 2 - Real.cos x ^ 4 - Real.sin (2 * x) + 2) = p * q - (p + q) :=
by
  sorry

end NUMINAMATH_GPT_expression_in_terms_of_p_and_q_l1849_184944


namespace NUMINAMATH_GPT_probability_diff_color_ball_l1849_184978

variable (boxA : List String) (boxB : List String)
def P_A (boxA := ["white", "white", "red", "red", "black"]) (boxB := ["white", "white", "white", "white", "red", "red", "red", "black", "black"]) : ℚ := sorry

theorem probability_diff_color_ball :
  P_A boxA boxB = 29 / 50 :=
sorry

end NUMINAMATH_GPT_probability_diff_color_ball_l1849_184978


namespace NUMINAMATH_GPT_tan_sum_angles_l1849_184999

theorem tan_sum_angles : (Real.tan (17 * Real.pi / 180) + Real.tan (28 * Real.pi / 180)) / (1 - Real.tan (17 * Real.pi / 180) * Real.tan (28 * Real.pi / 180)) = 1 := 
by sorry

end NUMINAMATH_GPT_tan_sum_angles_l1849_184999


namespace NUMINAMATH_GPT_max_product_l1849_184995

theorem max_product (a b : ℝ) (h1 : 9 * a ^ 2 + 16 * b ^ 2 = 25) (h2 : a > 0) (h3 : b > 0) :
  a * b ≤ 25 / 24 :=
sorry

end NUMINAMATH_GPT_max_product_l1849_184995


namespace NUMINAMATH_GPT_tom_speed_first_part_l1849_184987

-- Definitions of conditions in Lean
def total_distance : ℕ := 20
def distance_first_part : ℕ := 10
def speed_second_part : ℕ := 10
def average_speed : ℚ := 10.909090909090908
def distance_second_part := total_distance - distance_first_part

-- Lean statement to prove the speed during the first part of the trip
theorem tom_speed_first_part (v : ℚ) :
  (distance_first_part / v + distance_second_part / speed_second_part) = total_distance / average_speed → v = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tom_speed_first_part_l1849_184987


namespace NUMINAMATH_GPT_systematic_sampling_condition_l1849_184919

theorem systematic_sampling_condition (population sample_size total_removed segments individuals_per_segment : ℕ) 
  (h_population : population = 1650)
  (h_sample_size : sample_size = 35)
  (h_total_removed : total_removed = 5)
  (h_segments : segments = sample_size)
  (h_individuals_per_segment : individuals_per_segment = (population - total_removed) / sample_size)
  (h_modulo : population % sample_size = total_removed)
  :
  total_removed = 5 ∧ segments = 35 ∧ individuals_per_segment = 47 := 
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_condition_l1849_184919


namespace NUMINAMATH_GPT_minimum_operations_to_transfer_beer_l1849_184963

-- Definition of the initial conditions
structure InitialState where
  barrel_quarts : ℕ := 108
  seven_quart_vessel : ℕ := 0
  five_quart_vessel : ℕ := 0

-- Definition of the desired final state after minimum steps
structure FinalState where
  operations : ℕ := 17

-- Main theorem statement
theorem minimum_operations_to_transfer_beer (s : InitialState) : FinalState :=
  sorry

end NUMINAMATH_GPT_minimum_operations_to_transfer_beer_l1849_184963


namespace NUMINAMATH_GPT_solve_inner_parentheses_l1849_184989

theorem solve_inner_parentheses (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 57 ↔ x = 18 := by
  sorry

end NUMINAMATH_GPT_solve_inner_parentheses_l1849_184989


namespace NUMINAMATH_GPT_combined_mean_correct_l1849_184996

section MeanScore

variables (score_first_section mean_first_section : ℝ)
variables (score_second_section mean_second_section : ℝ)
variables (num_first_section num_second_section : ℝ)

axiom mean_first : mean_first_section = 92
axiom mean_second : mean_second_section = 78
axiom ratio_students : num_first_section / num_second_section = 5 / 7

noncomputable def combined_mean_score : ℝ := 
  let total_score := (mean_first_section * num_first_section + mean_second_section * num_second_section)
  let total_students := (num_first_section + num_second_section)
  total_score / total_students

theorem combined_mean_correct : combined_mean_score 92 78 (5 / 7 * num_second_section) num_second_section = 83.8 := by
  sorry

end MeanScore

end NUMINAMATH_GPT_combined_mean_correct_l1849_184996


namespace NUMINAMATH_GPT_mineral_samples_per_shelf_l1849_184956

theorem mineral_samples_per_shelf (total_samples : ℕ) (num_shelves : ℕ) (h1 : total_samples = 455) (h2 : num_shelves = 7) :
  total_samples / num_shelves = 65 :=
by
  sorry

end NUMINAMATH_GPT_mineral_samples_per_shelf_l1849_184956


namespace NUMINAMATH_GPT_total_number_of_trees_l1849_184952

variable {T : ℕ} -- Define T as a natural number (total number of trees)
variable (h1 : 70 / 100 * T + 105 = T) -- Indicates 30% of T is 105

theorem total_number_of_trees (h1 : 70 / 100 * T + 105 = T) : T = 350 :=
by
sorry

end NUMINAMATH_GPT_total_number_of_trees_l1849_184952


namespace NUMINAMATH_GPT_quadrant_of_angle_l1849_184921

theorem quadrant_of_angle (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  ∃! (q : ℕ), q = 2 :=
sorry

end NUMINAMATH_GPT_quadrant_of_angle_l1849_184921


namespace NUMINAMATH_GPT_game_winner_Aerith_first_game_winner_Bob_first_l1849_184943

-- Conditions: row of 20 squares, players take turns crossing out one square,
-- game ends when there are two squares left, Aerith wins if two remaining squares
-- are adjacent, Bob wins if they are not adjacent.

-- Definition of the game and winning conditions
inductive Player
| Aerith
| Bob

-- Function to determine the winner given the initial player
def winning_strategy (initial_player : Player) : Player :=
  match initial_player with
  | Player.Aerith => Player.Bob  -- Bob wins if Aerith goes first
  | Player.Bob    => Player.Aerith  -- Aerith wins if Bob goes first

-- Statement to prove
theorem game_winner_Aerith_first : 
  winning_strategy Player.Aerith = Player.Bob :=
by 
  sorry -- Proof is to be done

theorem game_winner_Bob_first :
  winning_strategy Player.Bob = Player.Aerith :=
by
  sorry -- Proof is to be done

end NUMINAMATH_GPT_game_winner_Aerith_first_game_winner_Bob_first_l1849_184943


namespace NUMINAMATH_GPT_value_of_M_l1849_184932

theorem value_of_M (G A M E: ℕ) (hG : G = 15)
(hGAME : G + A + M + E = 50)
(hMEGA : M + E + G + A = 55)
(hAGE : A + G + E = 40) : 
M = 15 := sorry

end NUMINAMATH_GPT_value_of_M_l1849_184932


namespace NUMINAMATH_GPT_camels_in_caravan_l1849_184965

theorem camels_in_caravan : 
  ∃ (C : ℕ), 
  (60 + 35 + 10 + C) * 1 + 60 * 2 + 35 * 4 + 10 * 2 + 4 * C - (60 + 35 + 10 + C) = 193 ∧ 
  C = 6 :=
by
  sorry

end NUMINAMATH_GPT_camels_in_caravan_l1849_184965


namespace NUMINAMATH_GPT_arc_length_of_f_l1849_184970

noncomputable def f (x : ℝ) : ℝ := 2 - Real.exp x

theorem arc_length_of_f :
  ∫ x in Real.log (Real.sqrt 3)..Real.log (Real.sqrt 8), Real.sqrt (1 + (Real.exp x)^2) = 1 + 1/2 * Real.log (3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_f_l1849_184970


namespace NUMINAMATH_GPT_roots_seventh_sum_l1849_184955

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_roots_seventh_sum_l1849_184955


namespace NUMINAMATH_GPT_animal_count_in_hollow_l1849_184968

theorem animal_count_in_hollow (heads legs : ℕ) (animals_with_odd_legs animals_with_even_legs : ℕ) :
  heads = 18 →
  legs = 24 →
  (∀ n, n % 2 = 1 → animals_with_odd_legs * 2 = heads - 2 * n) →
  (∀ m, m % 2 = 0 → animals_with_even_legs * 1 = heads - m) →
  (animals_with_odd_legs + animals_with_even_legs = 10 ∨
   animals_with_odd_legs + animals_with_even_legs = 12 ∨
   animals_with_odd_legs + animals_with_even_legs = 14) :=
sorry

end NUMINAMATH_GPT_animal_count_in_hollow_l1849_184968


namespace NUMINAMATH_GPT_sufficient_condition_l1849_184972

theorem sufficient_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2 * x + a = 0 → a < 1) ↔ 
  (∀ c : ℝ, x^2 - 2 * x + c = 0 ↔ 4 - 4 * c ≥ 0 ∧ c < 1 → ¬ (∀ d : ℝ, d ≤ 1 → d < 1)) := 
by 
sorry

end NUMINAMATH_GPT_sufficient_condition_l1849_184972


namespace NUMINAMATH_GPT_selling_price_is_320_l1849_184942

noncomputable def sales_volume (x : ℝ) : ℝ := 8000 / x

def cost_price : ℝ := 180

def desired_profit : ℝ := 3500

def selling_price_for_desired_profit (x : ℝ) : Prop :=
  (x - cost_price) * sales_volume x = desired_profit

/-- The selling price of the small electrical appliance to achieve a daily sales profit 
    of $3500 dollars is $320 dollars. -/
theorem selling_price_is_320 : selling_price_for_desired_profit 320 :=
by
  -- We skip the proof as per instructions
  sorry

end NUMINAMATH_GPT_selling_price_is_320_l1849_184942


namespace NUMINAMATH_GPT_perimeter_C_l1849_184902

theorem perimeter_C :
  ∀ (x y : ℕ),
  (6 * x + 2 * y = 56) →
  (4 * x + 6 * y = 56) →
  (2 * x + 6 * y = 40) :=
by
  intros x y hA hB
  sorry

end NUMINAMATH_GPT_perimeter_C_l1849_184902


namespace NUMINAMATH_GPT_roots_relationship_l1849_184920

theorem roots_relationship (a b c : ℝ) (α β : ℝ) 
  (h_eq : a * α^2 + b * α + c = 0)
  (h_triple : β = 3 * α)
  (h_vieta1 : α + β = -b / a)
  (h_vieta2 : α * β = c / a) : 
  3 * b^2 = 16 * a * c :=
sorry

end NUMINAMATH_GPT_roots_relationship_l1849_184920


namespace NUMINAMATH_GPT_find_x_l1849_184904

theorem find_x (x : ℝ) (h : 0.65 * x = 0.2 * 617.50) : x = 190 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1849_184904


namespace NUMINAMATH_GPT_total_amount_is_47_69_l1849_184959

noncomputable def Mell_order_cost : ℝ :=
  2 * 4 + 7

noncomputable def friend_order_cost : ℝ :=
  2 * 4 + 7 + 3

noncomputable def total_cost_before_discount : ℝ :=
  Mell_order_cost + 2 * friend_order_cost

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def sales_tax : ℝ :=
  0.10 * total_after_discount

noncomputable def total_to_pay : ℝ :=
  total_after_discount + sales_tax

theorem total_amount_is_47_69 : total_to_pay = 47.69 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_is_47_69_l1849_184959


namespace NUMINAMATH_GPT_area_of_tangency_triangle_l1849_184916

noncomputable def area_of_triangle : ℝ :=
  let r1 := 2
  let r2 := 3
  let r3 := 4
  let s := (r1 + r2 + r3) / 2
  let A := Real.sqrt (s * (s - (r1 + r2)) * (s - (r2 + r3)) * (s - (r1 + r3)))
  let inradius := A / s
  let area_points_of_tangency := A * (inradius / r1) * (inradius / r2) * (inradius / r3)
  area_points_of_tangency

theorem area_of_tangency_triangle :
  area_of_triangle = (16 * Real.sqrt 6) / 3 :=
sorry

end NUMINAMATH_GPT_area_of_tangency_triangle_l1849_184916


namespace NUMINAMATH_GPT_total_expenditure_correct_l1849_184939

-- Define the weekly costs based on the conditions
def cost_white_bread : Float := 2 * 3.50
def cost_baguette : Float := 1.50
def cost_sourdough_bread : Float := 2 * 4.50
def cost_croissant : Float := 2.00

-- Total weekly cost calculation
def weekly_cost : Float := cost_white_bread + cost_baguette + cost_sourdough_bread + cost_croissant

-- Total cost over 4 weeks
def total_cost_4_weeks (weeks : Float) : Float := weekly_cost * weeks

-- The assertion that needs to be proved
theorem total_expenditure_correct :
  total_cost_4_weeks 4 = 78.00 := by
  sorry

end NUMINAMATH_GPT_total_expenditure_correct_l1849_184939


namespace NUMINAMATH_GPT_union_A_B_correct_l1849_184962

def A : Set ℕ := {0, 1}
def B : Set ℕ := {x | 0 < x ∧ x < 3}

theorem union_A_B_correct : A ∪ B = {0, 1, 2} :=
by sorry

end NUMINAMATH_GPT_union_A_B_correct_l1849_184962


namespace NUMINAMATH_GPT_gas_volume_at_12_l1849_184925

variable (VolumeTemperature : ℕ → ℕ) -- a function representing the volume of gas at a given temperature 

axiom condition1 : ∀ t : ℕ, VolumeTemperature (t + 4) = VolumeTemperature t + 5

axiom condition2 : VolumeTemperature 28 = 35

theorem gas_volume_at_12 :
  VolumeTemperature 12 = 15 := 
sorry

end NUMINAMATH_GPT_gas_volume_at_12_l1849_184925


namespace NUMINAMATH_GPT_fraction_spent_on_dvd_l1849_184957

theorem fraction_spent_on_dvd (r l m d x : ℝ) (h1 : r = 200) (h2 : l = (1/4) * r) (h3 : m = r - l) (h4 : x = 50) (h5 : d = m - x) : d / r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_spent_on_dvd_l1849_184957


namespace NUMINAMATH_GPT_total_ants_correct_l1849_184969

-- Define the conditions
def park_width_ft : ℕ := 450
def park_length_ft : ℕ := 600
def ants_per_sq_inch_first_half : ℕ := 2
def ants_per_sq_inch_second_half : ℕ := 4

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Convert width and length from feet to inches
def park_width_inch : ℕ := park_width_ft * feet_to_inches
def park_length_inch : ℕ := park_length_ft * feet_to_inches

-- Define the area of each half of the park in square inches
def half_length_inch : ℕ := park_length_inch / 2
def area_first_half_sq_inch : ℕ := park_width_inch * half_length_inch
def area_second_half_sq_inch : ℕ := park_width_inch * half_length_inch

-- Define the number of ants in each half
def ants_first_half : ℕ := ants_per_sq_inch_first_half * area_first_half_sq_inch
def ants_second_half : ℕ := ants_per_sq_inch_second_half * area_second_half_sq_inch

-- Define the total number of ants
def total_ants : ℕ := ants_first_half + ants_second_half

-- The proof problem
theorem total_ants_correct : total_ants = 116640000 := by
  sorry

end NUMINAMATH_GPT_total_ants_correct_l1849_184969


namespace NUMINAMATH_GPT_problem_conditions_equation_right_triangle_vertex_coordinates_l1849_184915

theorem problem_conditions_equation : 
  ∃ (a b c : ℝ), a = -1 ∧ b = -2 ∧ c = 3 ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - (-(x + 1))^2 + 4) ∧ 
  (∀ x : ℝ, - (x - 1)^2 + 4 = - x^2 - 2 * x + 3)
:= sorry

theorem right_triangle_vertex_coordinates :
  ∀ x y : ℝ, x = -1 ∧ 
  (y = -2 ∨ y = 4 ∨ y = (3 + (17:ℝ).sqrt) / 2 ∨ y = (3 - (17:ℝ).sqrt) / 2)
  ∧ 
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (-3, 0)
  let C : ℝ × ℝ := (0, 3)
  let P : ℝ × ℝ := (x, y)
  let BC : ℝ := (B.1 - C.1)^2 + (B.2 - C.2)^2
  let PB : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
  let PC : ℝ := (P.1 - C.1)^2 + (P.2 - C.2)^2
  (BC + PB = PC ∨ BC + PC = PB ∨ PB + PC = BC)
:= sorry

end NUMINAMATH_GPT_problem_conditions_equation_right_triangle_vertex_coordinates_l1849_184915


namespace NUMINAMATH_GPT_bananas_used_l1849_184930

-- Define the conditions
def bananas_per_loaf : Nat := 4
def loaves_on_monday : Nat := 3
def loaves_on_tuesday : Nat := 2 * loaves_on_monday
def total_loaves : Nat := loaves_on_monday + loaves_on_tuesday

-- Define the total bananas used
def total_bananas : Nat := bananas_per_loaf * total_loaves

-- Prove that the total bananas used is 36
theorem bananas_used : total_bananas = 36 :=
by
  sorry

end NUMINAMATH_GPT_bananas_used_l1849_184930


namespace NUMINAMATH_GPT_x_intercept_of_perpendicular_line_l1849_184938

theorem x_intercept_of_perpendicular_line (x y : ℝ) (b : ℕ) :
  let line1 := 2 * x + 3 * y
  let slope1 := -2/3
  let slope2 := 3/2
  let y_intercept := -1
  let perp_line := slope2 * x + y_intercept
  let x_intercept := 2/3
  line1 = 12 → perp_line = 0 → x = x_intercept :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_of_perpendicular_line_l1849_184938


namespace NUMINAMATH_GPT_sum_of_numbers_is_919_l1849_184901

-- Problem Conditions
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99
def is_three_digit (y : ℕ) : Prop := 100 ≤ y ∧ y ≤ 999
def satisfies_equation (x y : ℕ) : Prop := 1000 * x + y = 11 * x * y

-- Main Statement
theorem sum_of_numbers_is_919 (x y : ℕ) 
  (h1 : is_two_digit x) 
  (h2 : is_three_digit y) 
  (h3 : satisfies_equation x y) : 
  x + y = 919 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_is_919_l1849_184901


namespace NUMINAMATH_GPT_range_of_m_l1849_184967

theorem range_of_m :
  ∀ m, (∀ x, m ≤ x ∧ x ≤ 4 → (0 ≤ -x^2 + 4*x ∧ -x^2 + 4*x ≤ 4)) ↔ (0 ≤ m ∧ m ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1849_184967


namespace NUMINAMATH_GPT_greater_number_is_25_l1849_184908

theorem greater_number_is_25 (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
sorry

end NUMINAMATH_GPT_greater_number_is_25_l1849_184908


namespace NUMINAMATH_GPT_product_of_means_eq_pm20_l1849_184900

theorem product_of_means_eq_pm20 :
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  a * b = 20 ∨ a * b = -20 :=
by
  -- Placeholders for the actual proof
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  sorry

end NUMINAMATH_GPT_product_of_means_eq_pm20_l1849_184900


namespace NUMINAMATH_GPT_rationalize_denominator_l1849_184937

theorem rationalize_denominator (t : ℝ) (h : t = 1 / (1 - Real.sqrt (Real.sqrt 2))) : 
  t = -(1 + Real.sqrt (Real.sqrt 2)) * (1 + Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1849_184937


namespace NUMINAMATH_GPT_distance_between_house_and_school_l1849_184941

theorem distance_between_house_and_school (T D : ℕ) 
    (h1 : D = 10 * (T + 2)) 
    (h2 : D = 20 * (T - 1)) : 
    D = 60 := by
  sorry

end NUMINAMATH_GPT_distance_between_house_and_school_l1849_184941


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l1849_184951

theorem arithmetic_sequence_general_term (a : ℕ → ℤ) : 
  (∀ n : ℕ, a n = a 1 + (n - 1) * (-2)) → 
  a 2 = 1 → 
  a 5 = -5 → 
  ∀ n : ℕ, a n = -2 * n + 5 :=
by
  intros h₁ h₂ h₅
  sorry

theorem arithmetic_sequence_max_sum (a : ℕ → ℤ) (S : ℕ → ℤ) : 
  ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (-2)) →
  a 2 = 1 → 
  a 5 = -5 → 
  ∃ n : ℕ, n = 2 ∧ S n = 4 :=
by
  intros hSn h₂ h₅
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_arithmetic_sequence_max_sum_l1849_184951


namespace NUMINAMATH_GPT_George_spending_l1849_184911

theorem George_spending (B m s : ℝ) (h1 : m = 0.25 * (B - s)) (h2 : s = 0.05 * (B - m)) : 
  (m + s) / B = 1 := 
by
  sorry

end NUMINAMATH_GPT_George_spending_l1849_184911


namespace NUMINAMATH_GPT_regular_octagon_side_length_sum_l1849_184934

theorem regular_octagon_side_length_sum (s : ℝ) (h₁ : s = 2.3) (h₂ : 1 = 100) : 
  8 * (s * 100) = 1840 :=
by
  sorry

end NUMINAMATH_GPT_regular_octagon_side_length_sum_l1849_184934


namespace NUMINAMATH_GPT_circle_passes_through_points_l1849_184971

-- Define the points
def P1 : ℝ × ℝ := (0, 0)
def P2 : ℝ × ℝ := (4, 0)
def P3 : ℝ × ℝ := (-1, 1)

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 6 * y = 0

-- Prove that the circle passes through the given points
theorem circle_passes_through_points :
  circle_eq P1.1 P1.2 ∧ circle_eq P2.1 P2.2 ∧ circle_eq P3.1 P3.2 :=
by
  -- Placeholders to write the proof later
  sorry

end NUMINAMATH_GPT_circle_passes_through_points_l1849_184971


namespace NUMINAMATH_GPT_profit_june_correct_l1849_184961

-- Define conditions
def profit_in_May : ℝ := 20000
def profit_in_July : ℝ := 28800

-- Define the monthly growth rate variable
variable (x : ℝ)

-- The growth factor per month
def growth_factor : ℝ := 1 + x

-- Given condition translated to an equation
def profit_relation (x : ℝ) : Prop :=
  profit_in_May * (growth_factor x) * (growth_factor x) = profit_in_July

-- The profit in June should be computed
def profit_in_June (x : ℝ) : ℝ :=
  profit_in_May * (growth_factor x)

-- The target profit in June we want to prove
def target_profit_in_June := 24000

-- Statement to prove
theorem profit_june_correct (h : profit_relation x) : profit_in_June x = target_profit_in_June :=
  sorry  -- proof to be completed

end NUMINAMATH_GPT_profit_june_correct_l1849_184961


namespace NUMINAMATH_GPT_album_ways_10_l1849_184949

noncomputable def total_album_ways : ℕ := 
  let photo_albums := 2
  let stamp_albums := 3
  let total_albums := 4
  let friends := 4
  ((total_albums.choose photo_albums) * (total_albums - photo_albums).choose stamp_albums) / friends

theorem album_ways_10 :
  total_album_ways = 10 := 
by sorry

end NUMINAMATH_GPT_album_ways_10_l1849_184949


namespace NUMINAMATH_GPT_hulk_jump_kilometer_l1849_184933

theorem hulk_jump_kilometer (n : ℕ) (h : ∀ n : ℕ, n ≥ 1 → (2^(n-1) : ℕ) ≤ 1000 → n-1 < 10) : n = 11 :=
by
  sorry

end NUMINAMATH_GPT_hulk_jump_kilometer_l1849_184933


namespace NUMINAMATH_GPT_vanessa_phone_pictures_l1849_184977

theorem vanessa_phone_pictures
  (C : ℕ) (P : ℕ) (hC : C = 7)
  (hAlbums : 5 * 6 = 30)
  (hTotal : 30 = P + C) :
  P = 23 := by
  sorry

end NUMINAMATH_GPT_vanessa_phone_pictures_l1849_184977


namespace NUMINAMATH_GPT_guarantee_min_points_l1849_184973

-- Define points for positions
def points_for_position (pos : ℕ) : ℕ :=
  if pos = 1 then 6
  else if pos = 2 then 4
  else if pos = 3 then 2
  else 0

-- Define the maximum points
def max_points_per_race := 6
def races := 4
def max_points := max_points_per_race * races

-- Define the condition of no ties
def no_ties := true

-- Define the problem statement
theorem guarantee_min_points (no_ties: true) (h1: points_for_position 1 = 6)
  (h2: points_for_position 2 = 4) (h3: points_for_position 3 = 2)
  (h4: max_points = 24) : 
  ∃ min_points, (min_points = 22) ∧ (∀ points, (points < min_points) → (∃ another_points, (another_points > points))) :=
  sorry

end NUMINAMATH_GPT_guarantee_min_points_l1849_184973


namespace NUMINAMATH_GPT_smaug_silver_coins_l1849_184985

theorem smaug_silver_coins :
  ∀ (num_gold num_copper num_silver : ℕ)
  (value_per_silver value_per_gold conversion_factor value_total : ℕ),
  num_gold = 100 →
  num_copper = 33 →
  value_per_silver = 8 →
  value_per_gold = 3 →
  conversion_factor = value_per_gold * value_per_silver →
  value_total = 2913 →
  (num_gold * conversion_factor + num_silver * value_per_silver + num_copper = value_total) →
  num_silver = 60 :=
by
  intros num_gold num_copper num_silver value_per_silver value_per_gold conversion_factor value_total
  intros h1 h2 h3 h4 h5 h6 h_eq
  sorry

end NUMINAMATH_GPT_smaug_silver_coins_l1849_184985


namespace NUMINAMATH_GPT_different_color_socks_l1849_184966

def total_socks := 15
def white_socks := 6
def brown_socks := 5
def blue_socks := 4

theorem different_color_socks (total : ℕ) (white : ℕ) (brown : ℕ) (blue : ℕ) :
  total = white + brown + blue →
  white ≠ 0 → brown ≠ 0 → blue ≠ 0 →
  (white * brown + brown * blue + white * blue) = 74 :=
by
  intros
  -- proof goes here
  sorry

end NUMINAMATH_GPT_different_color_socks_l1849_184966


namespace NUMINAMATH_GPT_eleven_pow_2010_mod_19_l1849_184910

theorem eleven_pow_2010_mod_19 : (11 ^ 2010) % 19 = 3 := sorry

end NUMINAMATH_GPT_eleven_pow_2010_mod_19_l1849_184910


namespace NUMINAMATH_GPT_three_over_x_solution_l1849_184948

theorem three_over_x_solution (x : ℝ) (h : 1 - 9 / x + 9 / (x^2) = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_three_over_x_solution_l1849_184948


namespace NUMINAMATH_GPT_largest_sum_ABC_l1849_184974

noncomputable def max_sum_ABC (A B C : ℕ) : ℕ :=
if A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 then
  A + B + C
else
  0

theorem largest_sum_ABC : ∃ A B C : ℕ, A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2310 ∧ max_sum_ABC A B C = 52 :=
sorry

end NUMINAMATH_GPT_largest_sum_ABC_l1849_184974


namespace NUMINAMATH_GPT_unique_zero_of_f_inequality_of_x1_x2_l1849_184947

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

theorem unique_zero_of_f (a : ℝ) (h : a ≥ 0) : ∃! x, f a x = 0 := sorry

theorem inequality_of_x1_x2 (a x1 x2 : ℝ) (h : f a x1 = g a x1 - g a x2) (hₐ: a ≥ 0) :
  x1 - 2 * x2 ≥ 1 - 2 * Real.log 2 := sorry

end NUMINAMATH_GPT_unique_zero_of_f_inequality_of_x1_x2_l1849_184947


namespace NUMINAMATH_GPT_range_m_l1849_184982

-- Definitions for propositions p and q
def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0)

def q (m : ℝ) : Prop :=
  (m - 1 > 0)

-- Given conditions:
-- 1. p ∨ q is true
-- 2. p ∧ q is false

theorem range_m (m : ℝ) (h1: p m ∨ q m) (h2: ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_m_l1849_184982


namespace NUMINAMATH_GPT_q_is_false_l1849_184924

theorem q_is_false (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ p) : ¬ q :=
by
  sorry

end NUMINAMATH_GPT_q_is_false_l1849_184924


namespace NUMINAMATH_GPT_group_D_forms_a_definite_set_l1849_184903

theorem group_D_forms_a_definite_set : 
  ∃ (S : Set ℝ), S = { x : ℝ | x = 1 ∨ x = -1 } :=
by
  sorry

end NUMINAMATH_GPT_group_D_forms_a_definite_set_l1849_184903


namespace NUMINAMATH_GPT_negation_proposition_l1849_184984

theorem negation_proposition :
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - 3 * x + 2 ≤ 0)) =
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ x^2 - 3 * x + 2 > 0) := 
sorry

end NUMINAMATH_GPT_negation_proposition_l1849_184984


namespace NUMINAMATH_GPT_operation_preserves_remainder_l1849_184975

theorem operation_preserves_remainder (N : ℤ) (k : ℤ) (m : ℤ) 
(f : ℤ → ℤ) (hN : N = 6 * k + 3) (hf : f N = 6 * m + 3) : f N % 6 = 3 :=
by
  sorry

end NUMINAMATH_GPT_operation_preserves_remainder_l1849_184975


namespace NUMINAMATH_GPT_largest_of_four_consecutive_odd_numbers_l1849_184990

theorem largest_of_four_consecutive_odd_numbers (x : ℤ) : 
  (x % 2 = 1) → 
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →
  (x + 6 = 27) :=
by
  sorry

end NUMINAMATH_GPT_largest_of_four_consecutive_odd_numbers_l1849_184990


namespace NUMINAMATH_GPT_remainder_mod_105_l1849_184953

theorem remainder_mod_105 (x : ℤ) 
  (h1 : 3 + x ≡ 4 [ZMOD 27])
  (h2 : 5 + x ≡ 9 [ZMOD 125])
  (h3 : 7 + x ≡ 25 [ZMOD 343]) :
  x % 105 = 4 :=
  sorry

end NUMINAMATH_GPT_remainder_mod_105_l1849_184953


namespace NUMINAMATH_GPT_find_angle_2_l1849_184913

theorem find_angle_2 (angle1 : ℝ) (angle2 : ℝ) 
  (h1 : angle1 = 60) 
  (h2 : angle1 + angle2 = 180) : 
  angle2 = 120 := 
by
  sorry

end NUMINAMATH_GPT_find_angle_2_l1849_184913


namespace NUMINAMATH_GPT_mariela_cards_total_l1849_184997

theorem mariela_cards_total : 
  let a := 287.0
  let b := 116
  a + b = 403 := 
by
  sorry

end NUMINAMATH_GPT_mariela_cards_total_l1849_184997


namespace NUMINAMATH_GPT_target_destroyed_probability_l1849_184945

noncomputable def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  let miss1 := 1 - p1
  let miss2 := 1 - p2
  let miss3 := 1 - p3
  let prob_all_miss := miss1 * miss2 * miss3
  let prob_one_hit := (p1 * miss2 * miss3) + (miss1 * p2 * miss3) + (miss1 * miss2 * p3)
  let prob_destroyed := 1 - (prob_all_miss + prob_one_hit)
  prob_destroyed

theorem target_destroyed_probability :
  probability_hit 0.9 0.9 0.8 = 0.954 :=
sorry

end NUMINAMATH_GPT_target_destroyed_probability_l1849_184945


namespace NUMINAMATH_GPT_find_a_l1849_184935

open Real

variable (a : ℝ)

theorem find_a (h : 4 * a + -5 * 3 = 0) : a = 15 / 4 :=
sorry

end NUMINAMATH_GPT_find_a_l1849_184935


namespace NUMINAMATH_GPT_find_four_numbers_l1849_184931

theorem find_four_numbers (a b c d : ℕ) (h1 : b^2 = a * c) (h2 : a * b * c = 216) (h3 : 2 * c = b + d) (h4 : b + c + d = 12) :
  a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2 :=
sorry

end NUMINAMATH_GPT_find_four_numbers_l1849_184931


namespace NUMINAMATH_GPT_exists_prime_mod_greater_remainder_l1849_184929

theorem exists_prime_mod_greater_remainder (a b : ℕ) (h1 : 0 < a) (h2 : a < b) :
  ∃ p : ℕ, Prime p ∧ a % p > b % p :=
sorry

end NUMINAMATH_GPT_exists_prime_mod_greater_remainder_l1849_184929


namespace NUMINAMATH_GPT_range_of_b_l1849_184927

theorem range_of_b (b : ℝ) :
  (∀ x : ℤ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) ↔ 5 < b ∧ b < 7 := 
sorry

end NUMINAMATH_GPT_range_of_b_l1849_184927


namespace NUMINAMATH_GPT_sector_area_l1849_184958

theorem sector_area (α : ℝ) (l : ℝ) (S : ℝ) (hα : α = 60 * Real.pi / 180) (hl : l = 6 * Real.pi) : S = 54 * Real.pi :=
sorry

end NUMINAMATH_GPT_sector_area_l1849_184958


namespace NUMINAMATH_GPT_same_root_implies_a_vals_l1849_184940

-- Define the first function f(x) = x - a
def f (x a : ℝ) : ℝ := x - a

-- Define the second function g(x) = x^2 + ax - 2
def g (x a : ℝ) : ℝ := x^2 + a * x - 2

-- Theorem statement
theorem same_root_implies_a_vals (a : ℝ) (x : ℝ) (hf : f x a = 0) (hg : g x a = 0) : a = 1 ∨ a = -1 := 
sorry

end NUMINAMATH_GPT_same_root_implies_a_vals_l1849_184940


namespace NUMINAMATH_GPT_walt_total_invested_l1849_184905

-- Given Conditions
def invested_at_seven : ℝ := 5500
def total_interest : ℝ := 970
def interest_rate_seven : ℝ := 0.07
def interest_rate_nine : ℝ := 0.09

-- Define the total amount invested
noncomputable def total_invested : ℝ := 12000

-- Prove the total amount invested
theorem walt_total_invested :
  interest_rate_seven * invested_at_seven + interest_rate_nine * (total_invested - invested_at_seven) = total_interest :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_walt_total_invested_l1849_184905


namespace NUMINAMATH_GPT_compute_expression_l1849_184993

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1849_184993


namespace NUMINAMATH_GPT_remainder_when_x150_divided_by_x1_4_l1849_184983

noncomputable def remainder_div_x150_by_x1_4 (x : ℝ) : ℝ :=
  x^150 % (x-1)^4

theorem remainder_when_x150_divided_by_x1_4 (x : ℝ) :
  remainder_div_x150_by_x1_4 x = -551300 * x^3 + 1665075 * x^2 - 1667400 * x + 562626 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_x150_divided_by_x1_4_l1849_184983


namespace NUMINAMATH_GPT_percentage_of_y_in_relation_to_25_percent_of_x_l1849_184923

variable (y x : ℕ) (p : ℕ)

-- Conditions
def condition1 : Prop := (y = (p * 25 * x) / 10000)
def condition2 : Prop := (y * x = 100 * 100)
def condition3 : Prop := (y = 125)

-- The proof goal
theorem percentage_of_y_in_relation_to_25_percent_of_x :
  condition1 y x p ∧ condition2 y x ∧ condition3 y → ((y * 100) / (25 * x / 100) = 625)
:= by
-- Here we would insert the proof steps, but they are omitted as per the requirements.
sorry

end NUMINAMATH_GPT_percentage_of_y_in_relation_to_25_percent_of_x_l1849_184923


namespace NUMINAMATH_GPT_range_of_a_l1849_184946

theorem range_of_a
  (a : ℝ)
  (h : ∀ x : ℝ, |x + 1| + |x - 3| ≥ a) : a ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1849_184946


namespace NUMINAMATH_GPT_disproves_proposition_l1849_184988

theorem disproves_proposition (a b : ℤ) (h₁ : a = -4) (h₂ : b = 3) : (a^2 > b^2) ∧ ¬ (a > b) :=
by
  sorry

end NUMINAMATH_GPT_disproves_proposition_l1849_184988


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1849_184907

theorem solution_set_of_inequality :
  {x : ℝ | |x - 5| + |x + 3| >= 10} = {x : ℝ | x ≤ -4} ∪ {x : ℝ | x ≥ 6} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1849_184907
