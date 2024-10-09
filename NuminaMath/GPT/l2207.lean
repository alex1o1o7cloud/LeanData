import Mathlib

namespace hyperbola_eccentricity_l2207_220709

theorem hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : 4 * c^2 = 25) (h₃ : a = 1/2) : c/a = 5 :=
by
  sorry

end hyperbola_eccentricity_l2207_220709


namespace upper_bound_y_l2207_220756

/-- 
  Theorem:
  For any real numbers x and y such that 3 < x < 6 and 6 < y, 
  if the greatest possible positive integer difference between x and y is 6,
  then the upper bound for y is 11.
 -/
theorem upper_bound_y (x y : ℝ) (h₁ : 3 < x) (h₂ : x < 6) (h₃ : 6 < y) (h₄ : y < some_number) (h₅ : y - x = 6) : y = 11 := 
by
  sorry

end upper_bound_y_l2207_220756


namespace walkway_time_against_direction_l2207_220757

theorem walkway_time_against_direction (v_p v_w t : ℝ) (h1 : 90 = (v_p + v_w) * 30)
  (h2 : v_p * 48 = 90) 
  (h3 : 90 = (v_p - v_w) * t) :
  t = 120 := by 
  sorry

end walkway_time_against_direction_l2207_220757


namespace simplify_and_evaluate_l2207_220747

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sqrt 3 + 1) : 
  ( ( (2 * x + 1) / x - 1 ) / ( (x^2 - 1) / x ) ) = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l2207_220747


namespace distance_home_to_school_l2207_220716

theorem distance_home_to_school :
  ∃ (D : ℝ) (T : ℝ), 
    3 * (T + 7 / 60) = D ∧
    6 * (T - 8 / 60) = D ∧
    D = 1.5 :=
by
  sorry

end distance_home_to_school_l2207_220716


namespace abs_sub_abs_eq_six_l2207_220702

theorem abs_sub_abs_eq_six
  (a b : ℝ)
  (h₁ : |a| = 4)
  (h₂ : |b| = 2)
  (h₃ : a * b < 0) :
  |a - b| = 6 :=
sorry

end abs_sub_abs_eq_six_l2207_220702


namespace rectangle_to_square_l2207_220782

variable (k n : ℕ)

theorem rectangle_to_square (h1 : k > 5) (h2 : k * (k - 5) = n^2) : n = 6 := by 
  sorry

end rectangle_to_square_l2207_220782


namespace price_per_yellow_stamp_l2207_220712

theorem price_per_yellow_stamp 
    (num_red_stamps : ℕ) (price_red_stamp : ℝ) 
    (num_blue_stamps : ℕ) (price_blue_stamp : ℝ)
    (num_yellow_stamps : ℕ) (goal : ℝ)
    (sold_red_stamps : ℕ) (sold_red_price : ℝ)
    (sold_blue_stamps : ℕ) (sold_blue_price : ℝ):

    num_red_stamps = 20 ∧ 
    num_blue_stamps = 80 ∧ 
    num_yellow_stamps = 7 ∧ 
    sold_red_stamps = 20 ∧ 
    sold_red_price = 1.1 ∧ 
    sold_blue_stamps = 80 ∧ 
    sold_blue_price = 0.8 ∧ 
    goal = 100 → 
    (goal - (sold_red_stamps * sold_red_price + sold_blue_stamps * sold_blue_price)) / num_yellow_stamps = 2 := 
  by
  sorry

end price_per_yellow_stamp_l2207_220712


namespace share_of_a_is_240_l2207_220722

theorem share_of_a_is_240 (A B C : ℝ) 
  (h1 : A = (2/3) * (B + C)) 
  (h2 : B = (2/3) * (A + C)) 
  (h3 : A + B + C = 600) : 
  A = 240 := 
by sorry

end share_of_a_is_240_l2207_220722


namespace cone_sphere_volume_ratio_l2207_220733

theorem cone_sphere_volume_ratio (r h : ℝ) 
  (radius_eq : r > 0)
  (volume_rel : (1 / 3 : ℝ) * π * r^2 * h = (1 / 3 : ℝ) * (4 / 3) * π * r^3) : 
  h / r = 4 / 3 :=
by
  sorry

end cone_sphere_volume_ratio_l2207_220733


namespace intersection_of_P_and_Q_l2207_220769

def P (x : ℝ) : Prop := 1 < x ∧ x < 4
def Q (x : ℝ) : Prop := 2 < x ∧ x < 3

theorem intersection_of_P_and_Q (x : ℝ) : P x ∧ Q x ↔ 2 < x ∧ x < 3 := by
  sorry

end intersection_of_P_and_Q_l2207_220769


namespace seeds_per_plant_l2207_220758

theorem seeds_per_plant :
  let trees := 2
  let plants_per_tree := 20
  let total_plants := trees * plants_per_tree
  let planted_trees := 24
  let planting_fraction := 0.60
  exists S : ℝ, planting_fraction * (total_plants * S) = planted_trees ∧ S = 1 :=
by
  sorry

end seeds_per_plant_l2207_220758


namespace simplify_expression_l2207_220741

theorem simplify_expression :
  (2^5 + 4^3) * (2^2 - (-2)^3)^8 = 96 * 12^8 :=
by
  sorry

end simplify_expression_l2207_220741


namespace total_tickets_sold_l2207_220754

theorem total_tickets_sold 
(adult_ticket_price : ℕ) (child_ticket_price : ℕ) 
(total_revenue : ℕ) (adult_tickets_sold : ℕ) 
(child_tickets_sold : ℕ) (total_tickets : ℕ) : 
adult_ticket_price = 5 → 
child_ticket_price = 2 → 
total_revenue = 275 → 
adult_tickets_sold = 35 → 
(child_tickets_sold * child_ticket_price) + (adult_tickets_sold * adult_ticket_price) = total_revenue →
total_tickets = adult_tickets_sold + child_tickets_sold →
total_tickets = 85 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_tickets_sold_l2207_220754


namespace chocolate_bars_left_l2207_220704

noncomputable def chocolateBarsCount : ℕ :=
  let initial_bars := 800
  let thomas_friends_bars := (3 * initial_bars) / 8
  let adjusted_thomas_friends_bars := thomas_friends_bars + 1  -- Adjust for the extra bar rounding issue
  let piper_bars_taken := initial_bars / 4
  let piper_bars_returned := 8
  let adjusted_piper_bars := piper_bars_taken - piper_bars_returned
  let paul_club_bars := 9
  let polly_club_bars := 7
  let catherine_bars_returned := 15
  
  initial_bars
  - adjusted_thomas_friends_bars
  - adjusted_piper_bars
  - paul_club_bars
  - polly_club_bars
  + catherine_bars_returned

theorem chocolate_bars_left : chocolateBarsCount = 308 := by
  sorry

end chocolate_bars_left_l2207_220704


namespace height_of_barbed_wire_l2207_220773

theorem height_of_barbed_wire (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (total_cost : ℝ) (h : ℝ) :
  area = 3136 →
  cost_per_meter = 1.50 →
  gate_width = 2 →
  total_cost = 999 →
  h = 3 := 
by
  sorry

end height_of_barbed_wire_l2207_220773


namespace division_example_l2207_220799

theorem division_example : ∃ A B : ℕ, 23 = 6 * A + B ∧ A = 3 ∧ B < 6 := 
by sorry

end division_example_l2207_220799


namespace circle_is_axisymmetric_and_centrally_symmetric_l2207_220761

structure Shape where
  isAxisymmetric : Prop
  isCentrallySymmetric : Prop

theorem circle_is_axisymmetric_and_centrally_symmetric :
  ∃ (s : Shape), s.isAxisymmetric ∧ s.isCentrallySymmetric :=
by
  sorry

end circle_is_axisymmetric_and_centrally_symmetric_l2207_220761


namespace speed_downstream_is_correct_l2207_220703

-- Definitions corresponding to the conditions
def speed_boat_still_water : ℕ := 60
def speed_current : ℕ := 17

-- Definition of speed downstream from the conditions and proving the result
theorem speed_downstream_is_correct :
  speed_boat_still_water + speed_current = 77 :=
by
  -- Proof is omitted
  sorry

end speed_downstream_is_correct_l2207_220703


namespace contribution_per_student_l2207_220739

theorem contribution_per_student (total_contribution : ℝ) (class_funds : ℝ) (num_students : ℕ) 
(h1 : total_contribution = 90) (h2 : class_funds = 14) (h3 : num_students = 19) : 
  (total_contribution - class_funds) / num_students = 4 :=
by
  sorry

end contribution_per_student_l2207_220739


namespace rational_square_of_1_minus_xy_l2207_220750

theorem rational_square_of_1_minus_xy (x y : ℚ) (h : x^5 + y^5 = 2 * x^2 * y^2) : ∃ (q : ℚ), 1 - x * y = q^2 :=
by
  sorry

end rational_square_of_1_minus_xy_l2207_220750


namespace total_surface_area_of_cylinder_l2207_220785

theorem total_surface_area_of_cylinder 
  (r h : ℝ) 
  (hr : r = 3) 
  (hh : h = 8) : 
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 66 * Real.pi := by
  sorry

end total_surface_area_of_cylinder_l2207_220785


namespace find_line_and_intersection_l2207_220753

def direct_proportion_function (k : ℝ) (x : ℝ) : ℝ :=
  k * x

def shifted_function (k : ℝ) (x b : ℝ) : ℝ :=
  k * x + b

theorem find_line_and_intersection
  (k : ℝ) (b : ℝ) (h₀ : direct_proportion_function k 1 = 2) (h₁ : b = 5) :
  (shifted_function k 1 b = 7) ∧ (shifted_function k (-5/2) b = 0) :=
by
  -- This is just a placeholder to indicate where the proof would go
  sorry

end find_line_and_intersection_l2207_220753


namespace sequence_general_formula_l2207_220721

theorem sequence_general_formula (a : ℕ → ℝ) (h : ∀ n, a (n+2) = 2 * a (n+1) / (2 + a (n+1))) :
  (a 1 = 1) → ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end sequence_general_formula_l2207_220721


namespace geometric_series_sum_l2207_220764

  theorem geometric_series_sum :
    let a := (1 / 4 : ℚ)
    let r := (1 / 4 : ℚ)
    let n := 4
    let S_n := a * (1 - r^n) / (1 - r)
    S_n = 255 / 768 := by
  sorry
  
end geometric_series_sum_l2207_220764


namespace inequality_am_gm_l2207_220743

theorem inequality_am_gm 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_sum : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) >= 6 := 
sorry

end inequality_am_gm_l2207_220743


namespace area_KLMQ_l2207_220723

structure Rectangle :=
(length : ℝ)
(width : ℝ)

def JR := 2
def RQ := 3
def JL := 8

def JLMR : Rectangle := {length := JL, width := JR}
def JKQR : Rectangle := {length := RQ, width := JR}

def RM : ℝ := JL
def QM : ℝ := RM - RQ
def LM : ℝ := JR

def KLMQ : Rectangle := {length := QM, width := LM}

theorem area_KLMQ : KLMQ.length * KLMQ.width = 10 :=
by
  sorry

end area_KLMQ_l2207_220723


namespace jeremy_remaining_money_l2207_220766

-- Conditions as definitions
def computer_cost : ℝ := 3000
def accessories_cost : ℝ := 0.1 * computer_cost
def initial_money : ℝ := 2 * computer_cost

-- Theorem statement for the proof problem
theorem jeremy_remaining_money : initial_money - computer_cost - accessories_cost = 2700 := by
  -- Proof will be added here
  sorry

end jeremy_remaining_money_l2207_220766


namespace area_of_triangle_l2207_220788

theorem area_of_triangle (a b c : ℝ) (h₁ : a + b = 14) (h₂ : c = 10) (h₃ : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 24 :=
  sorry

end area_of_triangle_l2207_220788


namespace max_marks_l2207_220762

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 165): M = 500 :=
by
  sorry

end max_marks_l2207_220762


namespace max_f_eq_find_a_l2207_220777

open Real

noncomputable def f (α : ℝ) : ℝ :=
  let a := (sin α, cos α)
  let b := (6 * sin α + cos α, 7 * sin α - 2 * cos α)
  a.1 * b.1 + a.2 * b.2

theorem max_f_eq : 
  ∃ α : ℝ, f α = 4 * sqrt 2 + 2 :=
sorry

structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_A : ℝ)

noncomputable def f_triangle (A : ℝ) : ℝ :=
  let a := (sin A, cos A)
  let b := (6 * sin A + cos A, 7 * sin A - 2 * cos A)
  a.1 * b.1 + a.2 * b.2

axiom f_A_eq (A : ℝ) : f_triangle A = 6

theorem find_a (A B C a b c : ℝ) (h₁ : f_triangle A = 6) (h₂ : 1 / 2 * b * c * sin A = 3) (h₃ : b + c = 2 + 3 * sqrt 2) :
  a = sqrt 10 :=
sorry

end max_f_eq_find_a_l2207_220777


namespace price_of_each_shirt_l2207_220711

theorem price_of_each_shirt 
  (toys_cost : ℕ := 3 * 10)
  (cards_cost : ℕ := 2 * 5)
  (total_spent : ℕ := 70)
  (remaining_cost: ℕ := total_spent - (toys_cost + cards_cost))
  (num_shirts : ℕ := 3 + 2) :
  (remaining_cost / num_shirts) = 6 :=
by
  sorry

end price_of_each_shirt_l2207_220711


namespace pieces_of_green_candy_l2207_220744

theorem pieces_of_green_candy (total_pieces red_pieces blue_pieces : ℝ)
  (h_total : total_pieces = 3409.7)
  (h_red : red_pieces = 145.5)
  (h_blue : blue_pieces = 785.2) :
  total_pieces - red_pieces - blue_pieces = 2479 := by
  sorry

end pieces_of_green_candy_l2207_220744


namespace problem_1_problem_2_problem_3_problem_4_l2207_220765

theorem problem_1 : 2 * Real.sqrt 7 - 6 * Real.sqrt 7 = -4 * Real.sqrt 7 :=
by sorry

theorem problem_2 : Real.sqrt (2 / 3) / Real.sqrt (8 / 27) = (3 / 2) :=
by sorry

theorem problem_3 : Real.sqrt 18 + Real.sqrt 98 - Real.sqrt 27 = (10 * Real.sqrt 2 - 3 * Real.sqrt 3) :=
by sorry

theorem problem_4 : (Real.sqrt 0.5 + Real.sqrt 6) - (Real.sqrt (1 / 8) - Real.sqrt 24) = (Real.sqrt 2 / 4) + 3 * Real.sqrt 6 :=
by sorry

end problem_1_problem_2_problem_3_problem_4_l2207_220765


namespace statement_A_l2207_220745

theorem statement_A (x : ℝ) (h : x < -1) : x^2 > x :=
sorry

end statement_A_l2207_220745


namespace fraction_shaded_in_cube_l2207_220707

theorem fraction_shaded_in_cube :
  let side_length := 2
  let face_area := side_length * side_length
  let total_surface_area := 6 * face_area
  let shaded_faces := 3
  let shaded_face_area := face_area / 2
  let total_shaded_area := shaded_faces * shaded_face_area
  total_shaded_area / total_surface_area = 1 / 4 :=
by
  sorry

end fraction_shaded_in_cube_l2207_220707


namespace fifteenth_term_is_143_l2207_220778

noncomputable def first_term : ℕ := 3
noncomputable def second_term : ℕ := 13
noncomputable def third_term : ℕ := 23
noncomputable def common_difference : ℕ := second_term - first_term
noncomputable def nth_term (n : ℕ) : ℕ := first_term + (n - 1) * common_difference

theorem fifteenth_term_is_143 :
  nth_term 15 = 143 := by
  sorry

end fifteenth_term_is_143_l2207_220778


namespace find_n_from_degree_l2207_220796

theorem find_n_from_degree (n : ℕ) (h : 2 + n = 5) : n = 3 :=
by {
  sorry
}

end find_n_from_degree_l2207_220796


namespace sam_more_than_avg_l2207_220740

def bridget_count : ℕ := 14
def reginald_count : ℕ := bridget_count - 2
def sam_count : ℕ := reginald_count + 4
def average_count : ℕ := (bridget_count + reginald_count + sam_count) / 3

theorem sam_more_than_avg 
    (h1 : bridget_count = 14) 
    (h2 : reginald_count = bridget_count - 2) 
    (h3 : sam_count = reginald_count + 4) 
    (h4 : average_count = (bridget_count + reginald_count + sam_count) / 3): 
    sam_count - average_count = 2 := 
  sorry

end sam_more_than_avg_l2207_220740


namespace max_x1_squared_plus_x2_squared_l2207_220775

theorem max_x1_squared_plus_x2_squared (k : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ + x₂ = k - 2)
  (h2 : x₁ * x₂ = k^2 + 3 * k + 5)
  (h3 : -4 ≤ k ∧ k ≤ -4 / 3) :
  x₁ ^ 2 + x₂ ^ 2 ≤ 18 :=
sorry

end max_x1_squared_plus_x2_squared_l2207_220775


namespace arithmetic_seq_a11_l2207_220790

variable (a : ℕ → ℤ)
variable (d : ℕ → ℤ)

-- Conditions
def arithmetic_sequence : Prop := ∀ n, a (n + 2) - a n = 6
def a1 : Prop := a 1 = 1

-- Statement of the problem
theorem arithmetic_seq_a11 : arithmetic_sequence a ∧ a1 a → a 11 = 31 :=
by sorry

end arithmetic_seq_a11_l2207_220790


namespace time_to_eliminate_mice_l2207_220786

def total_work : ℝ := 1
def work_done_by_2_cats_in_5_days : ℝ := 0.5
def initial_2_cats : ℕ := 2
def additional_cats : ℕ := 3
def total_initial_days : ℝ := 5
def total_cats : ℕ := initial_2_cats + additional_cats

theorem time_to_eliminate_mice (h : total_initial_days * (work_done_by_2_cats_in_5_days / total_initial_days) = work_done_by_2_cats_in_5_days) : 
  total_initial_days + (total_work - work_done_by_2_cats_in_5_days) / (total_cats * (work_done_by_2_cats_in_5_days / total_initial_days / initial_2_cats)) = 7 := 
by
  sorry

end time_to_eliminate_mice_l2207_220786


namespace min_C_over_D_l2207_220717

-- Define y + 1/y = D and y^2 + 1/y^2 = C.
theorem min_C_over_D (y C D : ℝ) (hy_pos : 0 < y) (hC : y ^ 2 + 1 / (y ^ 2) = C) (hD : y + 1 / y = D) (hC_pos : 0 < C) (hD_pos : 0 < D) :
  C / D = 2 := by
  sorry

end min_C_over_D_l2207_220717


namespace simplify_fraction_l2207_220755

-- We state the problem as a theorem.
theorem simplify_fraction : (3^2011 + 3^2011) / (3^2010 + 3^2012) = 3 / 5 := by sorry

end simplify_fraction_l2207_220755


namespace cost_of_pure_milk_l2207_220760

theorem cost_of_pure_milk (C : ℝ) (total_milk : ℝ) (pure_milk : ℝ) (water : ℝ) (profit : ℝ) :
  total_milk = pure_milk + water → profit = (total_milk * C) - (pure_milk * C) → profit = 35 → C = 7 :=
by
  intros h1 h2 h3
  sorry

end cost_of_pure_milk_l2207_220760


namespace annie_total_blocks_l2207_220768

-- Definitions of the blocks traveled in each leg of Annie's journey
def walk_to_bus_stop := 5
def ride_bus_to_train_station := 7
def train_to_friends_house := 10
def walk_to_coffee_shop := 4
def walk_back_to_friends_house := walk_to_coffee_shop

-- The total blocks considering the round trip and additional walk to/from coffee shop
def total_blocks_traveled :=
  2 * (walk_to_bus_stop + ride_bus_to_train_station + train_to_friends_house) +
  walk_to_coffee_shop + walk_back_to_friends_house

-- Statement to prove
theorem annie_total_blocks : total_blocks_traveled = 52 :=
by
  sorry

end annie_total_blocks_l2207_220768


namespace q_investment_correct_l2207_220738

-- Define the conditions
def profit_ratio := (4, 6)
def p_investment := 60000
def expected_q_investment := 90000

-- Define the theorem statement
theorem q_investment_correct (p_investment: ℕ) (q_investment: ℕ) (profit_ratio : ℕ × ℕ)
  (h_ratio: profit_ratio = (4, 6)) (hp_investment: p_investment = 60000) :
  q_investment = 90000 := by
  sorry

end q_investment_correct_l2207_220738


namespace sum_of_digits_next_perfect_square_222_l2207_220771

-- Define the condition for the perfect square that begins with "222"
def starts_with_222 (n: ℕ) : Prop :=
  n / 10^3 = 222

-- Define the sum of the digits function
def sum_of_digits (n: ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Statement for the Lean 4 statement: 
-- Prove that the sum of the digits of the next perfect square that starts with "222" is 18
theorem sum_of_digits_next_perfect_square_222 : sum_of_digits (492 ^ 2) = 18 :=
by
  sorry -- Proof omitted

end sum_of_digits_next_perfect_square_222_l2207_220771


namespace find_A_from_eq_l2207_220708

theorem find_A_from_eq (A : ℕ) (h : 10 - A = 6) : A = 4 :=
by
  sorry

end find_A_from_eq_l2207_220708


namespace sequence_periodicity_l2207_220772

noncomputable def a : ℕ → ℚ
| 0       => 0
| (n + 1) => (a n - 2) / ((5/4) * a n - 2)

theorem sequence_periodicity : a 2017 = 0 := by
  sorry

end sequence_periodicity_l2207_220772


namespace carlson_fraction_jam_l2207_220700

-- Definitions and conditions.
def total_time (T : ℕ) := T > 0
def time_maloish_cookies (t : ℕ) := t > 0
def equal_cookies (c : ℕ) := c > 0
def carlson_rate := 3

-- Let j_k and j_m be the amounts of jam eaten by Carlson and Maloish respectively.
def fraction_jam_carlson (j_k j_m : ℕ) : ℚ := j_k / (j_k + j_m)

-- The problem statement
theorem carlson_fraction_jam (T t c j_k j_m : ℕ)
  (hT : total_time T)
  (ht : time_maloish_cookies t)
  (hc : equal_cookies c)
  (h_carlson_rate : carlson_rate = 3)
  (h_equal_cookies : c > 0)  -- Both ate equal cookies
  (h_jam : j_k + j_m = j_k * 9 / 10 + j_m / 10) :
  fraction_jam_carlson j_k j_m = 9 / 10 :=
by
  sorry

end carlson_fraction_jam_l2207_220700


namespace translation_result_l2207_220763

-- Define the original point M
def M : ℝ × ℝ := (-10, 1)

-- Define the translation on the y-axis by 4 units
def translate_y (p : ℝ × ℝ) (dy : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + dy)

-- Define the resulting point M1 after translation
def M1 : ℝ × ℝ := translate_y M 4

-- The theorem we want to prove: the coordinates of M1 are (-10, 5)
theorem translation_result : M1 = (-10, 5) :=
by
  -- Proof goes here
  sorry

end translation_result_l2207_220763


namespace decimal_palindrome_multiple_l2207_220734

def is_decimal_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem decimal_palindrome_multiple (n : ℕ) (h : ¬ (10 ∣ n)) : 
  ∃ m : ℕ, is_decimal_palindrome m ∧ m % n = 0 :=
by sorry

end decimal_palindrome_multiple_l2207_220734


namespace probability_diff_suits_l2207_220783

theorem probability_diff_suits (n : ℕ) (h₁ : n = 65) (suits : ℕ) (h₂ : suits = 5) (cards_per_suit : ℕ) (h₃ : cards_per_suit = n / suits) : 
  (52 : ℚ) / (64 : ℚ) = (13 : ℚ) / (16 : ℚ) := 
by 
  sorry

end probability_diff_suits_l2207_220783


namespace second_train_start_time_l2207_220719

theorem second_train_start_time :
  let start_time_first_train := 14 -- 2:00 pm in 24-hour format
  let catch_up_time := 22          -- 10:00 pm in 24-hour format
  let speed_first_train := 70      -- km/h
  let speed_second_train := 80     -- km/h
  let travel_time_first_train := catch_up_time - start_time_first_train
  let distance_first_train := speed_first_train * travel_time_first_train
  let t := distance_first_train / speed_second_train
  let start_time_second_train := catch_up_time - t
  start_time_second_train = 15 := -- 3:00 pm in 24-hour format
by
  sorry

end second_train_start_time_l2207_220719


namespace prove_union_l2207_220787

variable (M N : Set ℕ)
variable (x : ℕ)

def M_definition := (0 ∈ M) ∧ (x ∈ M) ∧ (M = {0, x})
def N_definition := (N = {1, 2})
def intersection_condition := (M ∩ N = {2})
def union_result := (M ∪ N = {0, 1, 2})

theorem prove_union (M : Set ℕ) (N : Set ℕ) (x : ℕ) :
  M_definition M x → N_definition N → intersection_condition M N → union_result M N :=
by
  sorry

end prove_union_l2207_220787


namespace equal_cake_distribution_l2207_220706

theorem equal_cake_distribution (total_cakes : ℕ) (total_friends : ℕ) (h_cakes : total_cakes = 150) (h_friends : total_friends = 50) :
  total_cakes / total_friends = 3 := by
  sorry

end equal_cake_distribution_l2207_220706


namespace intersection_point_value_l2207_220731

theorem intersection_point_value (c d: ℤ) (h1: d = 2 * -4 + c) (h2: -4 = 2 * d + c) : d = -4 :=
by
  sorry

end intersection_point_value_l2207_220731


namespace count_silver_coins_l2207_220770

theorem count_silver_coins 
  (gold_value : ℕ)
  (silver_value : ℕ)
  (num_gold_coins : ℕ)
  (cash : ℕ)
  (total_money : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  num_gold_coins = 3 →
  cash = 30 →
  total_money = 305 →
  ∃ S : ℕ, num_gold_coins * gold_value + S * silver_value + cash = total_money ∧ S = 5 := 
by
  sorry

end count_silver_coins_l2207_220770


namespace negation_of_proposition_l2207_220713

open Classical

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) :=
by
  sorry

end negation_of_proposition_l2207_220713


namespace period_start_time_l2207_220797

/-- A period of time had 4 hours of rain and 5 hours without rain, ending at 5 pm. 
Prove that the period started at 8 am. -/
theorem period_start_time :
  let end_time := 17 -- 5 pm in 24-hour format
  let rainy_hours := 4
  let non_rainy_hours := 5
  let total_hours := rainy_hours + non_rainy_hours
  let start_time := end_time - total_hours
  start_time = 8 :=
by
  sorry

end period_start_time_l2207_220797


namespace max_n_for_factored_poly_l2207_220701

theorem max_n_for_factored_poly : 
  ∃ (n : ℤ), (∀ (A B : ℤ), 2 * B + A = n → A * B = 50) ∧ 
            (∀ (m : ℤ), (∀ (A B : ℤ), 2 * B + A = m → A * B = 50) → m ≤ 101) ∧ 
            n = 101 :=
by
  sorry

end max_n_for_factored_poly_l2207_220701


namespace Owen_spent_720_dollars_on_burgers_l2207_220784

def days_in_June : ℕ := 30
def burgers_per_day : ℕ := 2
def cost_per_burger : ℕ := 12

def total_burgers (days : ℕ) (burgers_per_day : ℕ) : ℕ :=
  days * burgers_per_day

def total_cost (burgers : ℕ) (cost_per_burger : ℕ) : ℕ :=
  burgers * cost_per_burger

theorem Owen_spent_720_dollars_on_burgers :
  total_cost (total_burgers days_in_June burgers_per_day) cost_per_burger = 720 := by
  sorry

end Owen_spent_720_dollars_on_burgers_l2207_220784


namespace total_students_exam_l2207_220779

theorem total_students_exam (N T T' T'' : ℕ) (h1 : T = 88 * N) (h2 : T' = T - 8 * 50) 
  (h3 : T' = 92 * (N - 8)) (h4 : T'' = T' - 100) (h5 : T'' = 92 * (N - 9)) : N = 84 :=
by
  sorry

end total_students_exam_l2207_220779


namespace packs_of_cake_l2207_220798

-- Given conditions
def total_grocery_packs : ℕ := 27
def cookie_packs : ℕ := 23

-- Question: How many packs of cake did Lucy buy?
-- Mathematically equivalent problem: Proving that cake_packs is 4
theorem packs_of_cake : (total_grocery_packs - cookie_packs) = 4 :=
by
  -- Proof goes here. Using sorry to skip the proof.
  sorry

end packs_of_cake_l2207_220798


namespace correct_area_ratio_l2207_220767

noncomputable def area_ratio (P : ℝ) : ℝ :=
  let x := P / 6 
  let length := P / 3
  let diagonal := (P * Real.sqrt 5) / 6
  let r := diagonal / 2
  let A := (5 * (P^2) * Real.pi) / 144
  let s := P / 5
  let R := P / (10 * Real.sin (36 * Real.pi / 180))
  let B := (P^2 * Real.pi) / (100 * (Real.sin (36 * Real.pi / 180))^2)
  A / B

theorem correct_area_ratio (P : ℝ) : area_ratio P = 500 * (Real.sin (36 * Real.pi / 180))^2 / 144 := 
  sorry

end correct_area_ratio_l2207_220767


namespace total_ticket_cost_l2207_220776

theorem total_ticket_cost (x y : ℕ) 
  (h1 : x + y = 380) 
  (h2 : y = x + 240) 
  (cost_orchestra : ℕ := 12) 
  (cost_balcony : ℕ := 8): 
  12 * x + 8 * y = 3320 := 
by 
  sorry

end total_ticket_cost_l2207_220776


namespace tank_capacity_l2207_220792

theorem tank_capacity (C : ℝ) :
  (C / 10 - 960 = C / 18) → C = 21600 := by
  intro h
  sorry

end tank_capacity_l2207_220792


namespace sum_arithmetic_sequence_l2207_220791

def first_term (k : ℕ) : ℕ := k^2 - k + 1

def sum_of_first_k_plus_3_terms (k : ℕ) : ℕ := (k + 3) * (k^2 + (k / 2) + 2)

theorem sum_arithmetic_sequence (k : ℕ) (k_pos : 0 < k) : 
    sum_of_first_k_plus_3_terms k = k^3 + (7 * k^2) / 2 + (15 * k) / 2 + 6 := 
by
  sorry

end sum_arithmetic_sequence_l2207_220791


namespace distance_from_center_of_C_to_line_l2207_220718

def circle_center_distance : ℝ :=
  let line1 (x y : ℝ) := x - y - 4
  let circle1 (x y : ℝ) := x^2 + y^2 - 4 * x - 6
  let circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6
  let line2 (x y : ℝ) := 3 * x + 4 * y + 5
  sorry

theorem distance_from_center_of_C_to_line :
  circle_center_distance = 2 := sorry

end distance_from_center_of_C_to_line_l2207_220718


namespace fraction_books_sold_l2207_220710

theorem fraction_books_sold :
  (∃ B F : ℝ, 3.50 * (B - 40) = 280.00000000000006 ∧ B ≠ 0 ∧ F = ((B - 40) / B) ∧ B = 120) → (F = 2 / 3) :=
by
  intro h
  obtain ⟨B, F, h1, h2, e⟩ := h
  sorry

end fraction_books_sold_l2207_220710


namespace cost_of_notebook_is_12_l2207_220730

/--
In a class of 36 students, a majority purchased notebooks. Each student bought the same number of notebooks (greater than 2). The price of a notebook in cents was double the number of notebooks each student bought, and the total expense was 2772 cents.
Prove that the cost of one notebook in cents is 12.
-/
theorem cost_of_notebook_is_12
  (s n c : ℕ) (total_students : ℕ := 36) 
  (h_majority : s > 18) 
  (h_notebooks : n > 2) 
  (h_cost : c = 2 * n) 
  (h_total_cost : s * c * n = 2772) 
  : c = 12 :=
by sorry

end cost_of_notebook_is_12_l2207_220730


namespace smallest_positive_integer_l2207_220746

theorem smallest_positive_integer (N : ℕ) :
  (N % 2 = 1) ∧
  (N % 3 = 2) ∧
  (N % 4 = 3) ∧
  (N % 5 = 4) ∧
  (N % 6 = 5) ∧
  (N % 7 = 6) ∧
  (N % 8 = 7) ∧
  (N % 9 = 8) ∧
  (N % 10 = 9) ↔ 
  N = 2519 := by {
  sorry
}

end smallest_positive_integer_l2207_220746


namespace reciprocal_sum_of_roots_l2207_220732

theorem reciprocal_sum_of_roots
  (a b c : ℝ)
  (ha : a^3 - 2022 * a + 1011 = 0)
  (hb : b^3 - 2022 * b + 1011 = 0)
  (hc : c^3 - 2022 * c + 1011 = 0)
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (1 / a) + (1 / b) + (1 / c) = 2 :=
sorry

end reciprocal_sum_of_roots_l2207_220732


namespace smallest_possible_value_l2207_220728

/-
Given:
1. m and n are positive integers.
2. gcd of m and n is (x + 5).
3. lcm of m and n is x * (x + 5).
4. m = 60.
5. x is a positive integer.

Prove:
The smallest possible value of n is 100.
-/

theorem smallest_possible_value 
  (m n x : ℕ) 
  (h1 : m = 60) 
  (h2 : x > 0) 
  (h3 : Nat.gcd m n = x + 5) 
  (h4 : Nat.lcm m n = x * (x + 5)) : 
  n = 100 := 
by 
  sorry

end smallest_possible_value_l2207_220728


namespace distinct_arithmetic_progression_roots_l2207_220714

theorem distinct_arithmetic_progression_roots (a b : ℝ) : 
  (∃ (d : ℝ), d ≠ 0 ∧ ∀ x, x^3 + a * x + b = 0 ↔ x = -d ∨ x = 0 ∨ x = d) → a < 0 ∧ b = 0 :=
by
  sorry

end distinct_arithmetic_progression_roots_l2207_220714


namespace constant_term_expansion_l2207_220749

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) :
  ∃ k : ℝ, k = -21/2 ∧
  (∀ r : ℕ, (9 : ℕ).choose r * (x^(1/2))^(9-r) * ((-(1/(2*x)))^r) = k) :=
sorry

end constant_term_expansion_l2207_220749


namespace solve_for_V_l2207_220735

open Real

theorem solve_for_V :
  ∃ k V, 
    (U = k * (V / W) ∧ (U = 16 ∧ W = 1 / 4 ∧ V = 2) ∧ (U = 25 ∧ W = 1 / 5 ∧ V = 2.5)) :=
by {
  sorry
}

end solve_for_V_l2207_220735


namespace find_first_number_l2207_220736

noncomputable def x : ℕ := 7981
noncomputable def y : ℕ := 9409
noncomputable def mean_proportional : ℕ := 8665

theorem find_first_number (mean_is_correct : (mean_proportional^2 = x * y)) : x = 7981 := by
-- Given: mean_proportional^2 = x * y
-- Goal: x = 7981
  sorry

end find_first_number_l2207_220736


namespace max_square_test_plots_l2207_220752

theorem max_square_test_plots
    (length : ℕ)
    (width : ℕ)
    (fence : ℕ)
    (fields_measure : length = 30 ∧ width = 45)
    (fence_measure : fence = 2250) :
  ∃ (number_of_plots : ℕ),
    number_of_plots = 150 :=
by
  sorry

end max_square_test_plots_l2207_220752


namespace equilateral_triangle_l2207_220725

theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + c^2 = ab + bc + ca) : a = b ∧ b = c := 
by sorry

end equilateral_triangle_l2207_220725


namespace sum_of_reciprocals_l2207_220774

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 32) :
  (1 / x) + (1 / y) = 3 / 8 := 
sorry

end sum_of_reciprocals_l2207_220774


namespace find_f_of_minus_five_l2207_220729

theorem find_f_of_minus_five (a b : ℝ) (f : ℝ → ℝ) (h1 : f 5 = 7) (h2 : ∀ x, f x = a * x + b * Real.sin x + 1) : f (-5) = -5 :=
by
  sorry

end find_f_of_minus_five_l2207_220729


namespace smallest_number_of_hikers_l2207_220720

theorem smallest_number_of_hikers (n : ℕ) :
  (n % 6 = 1) ∧ (n % 8 = 2) ∧ (n % 9 = 4) ↔ n = 154 :=
by sorry

end smallest_number_of_hikers_l2207_220720


namespace fewer_cans_today_l2207_220737

variable (nc_sarah_yesterday : ℕ)
variable (nc_lara_yesterday : ℕ)
variable (nc_alex_yesterday : ℕ)
variable (nc_sarah_today : ℕ)
variable (nc_lara_today : ℕ)
variable (nc_alex_today : ℕ)

-- Given conditions
def yesterday_collected_cans : Prop :=
  nc_sarah_yesterday = 50 ∧
  nc_lara_yesterday = nc_sarah_yesterday + 30 ∧
  nc_alex_yesterday = 90

def today_collected_cans : Prop :=
  nc_sarah_today = 40 ∧
  nc_lara_today = 70 ∧
  nc_alex_today = 55

theorem fewer_cans_today :
  yesterday_collected_cans nc_sarah_yesterday nc_lara_yesterday nc_alex_yesterday →
  today_collected_cans nc_sarah_today nc_lara_today nc_alex_today →
  (nc_sarah_yesterday + nc_lara_yesterday + nc_alex_yesterday) -
  (nc_sarah_today + nc_lara_today + nc_alex_today) = 55 :=
by
  intros h1 h2
  sorry

end fewer_cans_today_l2207_220737


namespace top_card_is_queen_probability_l2207_220742

theorem top_card_is_queen_probability :
  let total_cards := 54
  let number_of_queens := 4
  (number_of_queens / total_cards) = (2 / 27) := by
    sorry

end top_card_is_queen_probability_l2207_220742


namespace average_growth_rate_equation_l2207_220751

-- Define the current and target processing capacities
def current_capacity : ℝ := 1000
def target_capacity : ℝ := 1200

-- Define the time period in months
def months : ℕ := 2

-- Define the monthly average growth rate
variable (x : ℝ)

-- The statement to be proven: current capacity increased by the growth rate over 2 months equals the target capacity 
theorem average_growth_rate_equation :
  current_capacity * (1 + x) ^ months = target_capacity :=
sorry

end average_growth_rate_equation_l2207_220751


namespace find_n_for_quadratic_roots_l2207_220724

noncomputable def quadratic_root_properties (d c e n : ℝ) : Prop :=
  let A := (n + 2)
  let B := -((n + 2) * d + (n - 2) * c)
  let C := e * (n - 2)
  ∃ y1 y2 : ℝ, (A * y1 * y1 + B * y1 + C = 0) ∧ (A * y2 * y2 + B * y2 + C = 0) ∧ (y1 = -y2) ∧ (y1 + y2 = 0)

theorem find_n_for_quadratic_roots (d c e : ℝ) (h : d ≠ c) : 
  (quadratic_root_properties d c e (-2)) :=
sorry

end find_n_for_quadratic_roots_l2207_220724


namespace ratio_time_B_to_A_l2207_220793

-- Definitions for the given conditions
def T_A : ℕ := 10
def work_rate_A : ℚ := 1 / T_A
def combined_work_rate : ℚ := 0.3

-- Lean 4 statement for the problem
theorem ratio_time_B_to_A (T_B : ℚ) (h : (work_rate_A + 1 / T_B) = combined_work_rate) :
  (T_B / T_A) = (1 / 2) := by
  sorry

end ratio_time_B_to_A_l2207_220793


namespace combined_area_of_tracts_l2207_220789

theorem combined_area_of_tracts :
  let length1 := 300
  let width1 := 500
  let length2 := 250
  let width2 := 630
  let area1 := length1 * width1
  let area2 := length2 * width2
  let combined_area := area1 + area2
  combined_area = 307500 :=
by
  sorry

end combined_area_of_tracts_l2207_220789


namespace quiz_common_difference_l2207_220715

theorem quiz_common_difference 
  (x d : ℕ) 
  (h1 : x + 2 * d = 39) 
  (h2 : 8 * x + 28 * d = 360) 
  : d = 4 := 
  sorry

end quiz_common_difference_l2207_220715


namespace greatest_sum_first_quadrant_l2207_220794

theorem greatest_sum_first_quadrant (x y : ℤ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_circle : x^2 + y^2 = 49) : x + y ≤ 7 :=
sorry

end greatest_sum_first_quadrant_l2207_220794


namespace village_household_count_l2207_220727

theorem village_household_count
  (H : ℕ)
  (water_per_household_per_month : ℕ := 20)
  (total_water : ℕ := 2000)
  (duration_months : ℕ := 10)
  (total_consumption_condition : water_per_household_per_month * H * duration_months = total_water) :
  H = 10 :=
by
  sorry

end village_household_count_l2207_220727


namespace factorial_sum_perfect_square_iff_l2207_220795

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, m * m = n

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map Nat.factorial |>.sum

theorem factorial_sum_perfect_square_iff (n : Nat) :
  n = 1 ∨ n = 3 ↔ is_perfect_square (sum_of_factorials n) := by {
  sorry
}

end factorial_sum_perfect_square_iff_l2207_220795


namespace range_of_k_if_intersection_empty_l2207_220748

open Set

variable (k : ℝ)

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem range_of_k_if_intersection_empty (h : M ∩ N k = ∅) : k ≤ -1 :=
by {
  sorry
}

end range_of_k_if_intersection_empty_l2207_220748


namespace count_multiples_of_14_between_100_and_400_l2207_220705

theorem count_multiples_of_14_between_100_and_400 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (100 ≤ k ∧ k ≤ 400 ∧ 14 ∣ k) ↔ (∃ i : ℕ, k = 14 * i ∧ 8 ≤ i ∧ i ≤ 28)) :=
sorry

end count_multiples_of_14_between_100_and_400_l2207_220705


namespace maximum_sum_minimum_difference_l2207_220726

-- Definitions based on problem conditions
def is_least_common_multiple (m n lcm: ℕ) : Prop := Nat.lcm m n = lcm
def is_greatest_common_divisor (m n gcd: ℕ) : Prop := Nat.gcd m n = gcd

-- The target theorem to prove
theorem maximum_sum_minimum_difference (x y: ℕ) (h_lcm: is_least_common_multiple x y 2010) (h_gcd: is_greatest_common_divisor x y 2) :
  (x + y = 2012 ∧ x - y = 104 ∨ y - x = 104) :=
by
  sorry

end maximum_sum_minimum_difference_l2207_220726


namespace parabola_equation_l2207_220759

theorem parabola_equation (a b c d e f: ℤ) (ha: a = 2) (hb: b = 0) (hc: c = 0) (hd: d = -16) (he: e = -1) (hf: f = 32) :
  ∃ x y : ℝ, 2 * x ^ 2 - 16 * x + 32 - y = 0 ∧ gcd (abs a) (gcd (abs b) (gcd (abs c) (gcd (abs d) (gcd (abs e) (abs f))))) = 1 :=
by
  sorry

end parabola_equation_l2207_220759


namespace arithmetic_progression_sum_l2207_220781

theorem arithmetic_progression_sum (a : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 0 + n * d)
  (h2 : a 0 = 2)
  (h3 : a 1 + a 2 = 13) :
  a 3 + a 4 + a 5 = 42 :=
sorry

end arithmetic_progression_sum_l2207_220781


namespace problem_inequality_l2207_220780

theorem problem_inequality (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h: (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  a / b + b / c + c / a = 4.5 :=
by
  sorry

end problem_inequality_l2207_220780
