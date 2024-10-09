import Mathlib

namespace R_and_D_calculation_l74_7458

-- Define the given conditions and required calculation
def R_and_D_t : ℝ := 2640.92
def delta_APL_t_plus_1 : ℝ := 0.12

theorem R_and_D_calculation :
  (R_and_D_t / delta_APL_t_plus_1) = 22008 := by sorry

end R_and_D_calculation_l74_7458


namespace janets_garden_area_l74_7476

theorem janets_garden_area :
  ∃ (s l : ℕ), 2 * (s + l) = 24 ∧ (l + 1) = 3 * (s + 1) ∧ 6 * (s + 1 - 1) * 6 * (l + 1 - 1) = 576 := 
by
  sorry

end janets_garden_area_l74_7476


namespace find_d_l74_7459

theorem find_d (a d : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * d) : d = 49 :=
sorry

end find_d_l74_7459


namespace mean_of_quadrilateral_angles_l74_7416

theorem mean_of_quadrilateral_angles :
  ∀ (angles : List ℝ), angles.length = 4 → angles.sum = 360 → angles.sum / angles.length = 90 :=
by
  intros
  sorry

end mean_of_quadrilateral_angles_l74_7416


namespace slope_of_line_l74_7436

theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : 3 / x + 4 / y = 0) : 
  ∃ m : ℝ, m = -4 / 3 := 
sorry

end slope_of_line_l74_7436


namespace fred_earned_63_dollars_l74_7453

-- Definitions for the conditions
def initial_money_fred : ℕ := 23
def initial_money_jason : ℕ := 46
def money_per_car : ℕ := 5
def money_per_lawn : ℕ := 10
def money_per_dog : ℕ := 3
def total_money_after_chores : ℕ := 86
def cars_washed : ℕ := 4
def lawns_mowed : ℕ := 3
def dogs_walked : ℕ := 7

-- The equivalent proof problem in Lean
theorem fred_earned_63_dollars :
  (initial_money_fred + (cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = total_money_after_chores) → 
  ((cars_washed * money_per_car) + 
      (lawns_mowed * money_per_lawn) + 
      (dogs_walked * money_per_dog) = 63) :=
by
  sorry

end fred_earned_63_dollars_l74_7453


namespace petya_friends_count_l74_7478

-- Define the number of classmates
def total_classmates : ℕ := 28

-- Each classmate has a unique number of friends from 0 to 27
def unique_friends (n : ℕ) : Prop :=
  n ≥ 0 ∧ n < total_classmates

-- We state the problem where Petya's number of friends is to be proven as 14
theorem petya_friends_count (friends : ℕ) (h : unique_friends friends) : friends = 14 :=
sorry

end petya_friends_count_l74_7478


namespace cogs_produced_after_speed_increase_l74_7480

-- Define the initial conditions of the problem
def initial_cogs := 60
def initial_rate := 15
def increased_rate := 60
def average_output := 24

-- Variables to represent the number of cogs produced after the speed increase and the total time taken for each phase
variable (x : ℕ)

-- Assuming the equations representing the conditions
def initial_time := initial_cogs / initial_rate
def increased_time := x / increased_rate

def total_cogs := initial_cogs + x
def total_time := initial_time + increased_time

-- Define the overall average output equation
def average_eq := average_output * total_time = total_cogs

-- The proposition we want to prove
theorem cogs_produced_after_speed_increase : x = 60 :=
by
  -- Using the equation from the conditions
  have h1 : average_eq := sorry
  sorry

end cogs_produced_after_speed_increase_l74_7480


namespace solve_system_l74_7432

noncomputable def solutions (a b c : ℝ) : Prop :=
  a^4 - b^4 = c ∧ b^4 - c^4 = a ∧ c^4 - a^4 = b

theorem solve_system :
  { (a, b, c) | solutions a b c } =
  { (0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0) } :=
by
  sorry

end solve_system_l74_7432


namespace x_is_one_if_pure_imaginary_l74_7438

theorem x_is_one_if_pure_imaginary
  (x : ℝ)
  (h1 : x^2 - 1 = 0)
  (h2 : x^2 + 3 * x + 2 ≠ 0) :
  x = 1 :=
sorry

end x_is_one_if_pure_imaginary_l74_7438


namespace chameleon_color_change_l74_7442

variable (x : ℕ)

-- Initial and final conditions definitions.
def total_chameleons : ℕ := 140
def initial_blue_chameleons : ℕ := 5 * x 
def initial_red_chameleons : ℕ := 140 - 5 * x 
def final_blue_chameleons : ℕ := x
def final_red_chameleons : ℕ := 3 * (140 - 5 * x )

-- Proof statement
theorem chameleon_color_change :
  (140 - 5 * x) * 3 + x = 140 → 4 * x = 80 :=
by sorry

end chameleon_color_change_l74_7442


namespace exponent_division_is_equal_l74_7470

variable (a : ℝ) 

theorem exponent_division_is_equal :
  (a^11) / (a^2) = a^9 := 
sorry

end exponent_division_is_equal_l74_7470


namespace algebraic_expression_value_l74_7447

theorem algebraic_expression_value (a : ℝ) (h : a = Real.sqrt 6 + 2) : a^2 - 4 * a + 4 = 6 :=
by
  sorry

end algebraic_expression_value_l74_7447


namespace bobs_walking_rate_l74_7483

theorem bobs_walking_rate (distance_XY : ℕ) 
  (yolanda_rate : ℕ) 
  (bob_distance_when_met : ℕ) 
  (yolanda_extra_hour : ℕ)
  (meet_covered_distance : distance_XY = yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1 + bob_distance_when_met / bob_distance_when_met)) 
  (yolanda_distance_when_met : yolanda_rate * (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) + bob_distance_when_met = distance_XY) 
  : 
  (bob_distance_when_met / (bob_distance_when_met / yolanda_rate + yolanda_extra_hour - 1) = yolanda_rate) :=
  sorry

end bobs_walking_rate_l74_7483


namespace rod_volume_proof_l74_7490

-- Definitions based on given conditions
def original_length : ℝ := 2
def increase_in_surface_area : ℝ := 0.6
def rod_volume : ℝ := 0.3

-- Problem statement
theorem rod_volume_proof
  (len : ℝ)
  (inc_surface_area : ℝ)
  (vol : ℝ)
  (h_len : len = original_length)
  (h_inc_surface_area : inc_surface_area = increase_in_surface_area) :
  vol = rod_volume :=
sorry

end rod_volume_proof_l74_7490


namespace doughnuts_per_box_l74_7460

theorem doughnuts_per_box
  (total_doughnuts : ℕ)
  (boxes_sold : ℕ)
  (doughnuts_given_away : ℕ)
  (doughnuts_per_box : ℕ)
  (h1 : total_doughnuts = 300)
  (h2 : boxes_sold = 27)
  (h3 : doughnuts_given_away = 30) :
  doughnuts_per_box = (total_doughnuts - doughnuts_given_away) / boxes_sold := by
  -- proof goes here
  sorry

end doughnuts_per_box_l74_7460


namespace cubic_geometric_sequence_conditions_l74_7492

-- Conditions from the problem
def cubic_eq (a b c x : ℝ) : Prop := x^3 + a * x^2 + b * x + c = 0

-- The statement to be proven
theorem cubic_geometric_sequence_conditions (a b c : ℝ) :
  (∃ x q : ℝ, q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 ∧ 
    cubic_eq a b c x ∧ cubic_eq a b c (x*q) ∧ cubic_eq a b c (x*q^2)) → 
  (b^3 = a^3 * c ∧ c ≠ 0 ∧ -a^3 < c ∧ c < a^3 / 27 ∧ a < m ∧ m < - a / 3) :=
by 
  sorry

end cubic_geometric_sequence_conditions_l74_7492


namespace point_on_x_axis_coord_l74_7493

theorem point_on_x_axis_coord (m : ℝ) (h : (m - 1, 2 * m).snd = 0) : (m - 1, 2 * m) = (-1, 0) :=
by
  sorry

end point_on_x_axis_coord_l74_7493


namespace find_c_l74_7440

theorem find_c (x : ℝ) (c : ℝ) (h1 : 3 * x + 5 = 4) (h2 : c * x + 6 = 3) : c = 9 :=
by
  sorry

end find_c_l74_7440


namespace find_value_of_expression_l74_7403

theorem find_value_of_expression :
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 :=
by
  sorry

end find_value_of_expression_l74_7403


namespace time_to_overflow_equals_correct_answer_l74_7410

-- Definitions based on conditions
def pipeA_fill_time : ℚ := 32
def pipeB_fill_time : ℚ := pipeA_fill_time / 5

-- Derived rates from the conditions
def pipeA_rate : ℚ := 1 / pipeA_fill_time
def pipeB_rate : ℚ := 1 / pipeB_fill_time
def combined_rate : ℚ := pipeA_rate + pipeB_rate

-- The time to overflow when both pipes are filling the tank simultaneously
def time_to_overflow : ℚ := 1 / combined_rate

-- The statement we are going to prove
theorem time_to_overflow_equals_correct_answer : time_to_overflow = 16 / 3 :=
by sorry

end time_to_overflow_equals_correct_answer_l74_7410


namespace range_g_l74_7423

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + 2 * Real.arcsin x

theorem range_g : 
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -Real.pi / 2 ≤ g x ∧ g x ≤ 3 * Real.pi / 2) := 
by {
  sorry
}

end range_g_l74_7423


namespace man_walking_time_l74_7452

theorem man_walking_time (D V_w V_m T : ℝ) (t : ℝ) :
  D = V_w * T →
  D_w = V_m * t →
  D - V_m * t = V_w * (T - t) →
  T - (T - t) = 16 →
  t = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end man_walking_time_l74_7452


namespace f_4_1981_eq_l74_7465

def f : ℕ → ℕ → ℕ
| 0, y     => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_4_1981_eq : f 4 1981 = 2 ^ 16 - 3 := sorry

end f_4_1981_eq_l74_7465


namespace compute_expression_l74_7421

theorem compute_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b^3) / (a^2 - 2*a*b + b^2 + a*b) = 5 :=
by
  have h : a = 3 := h1
  have k : b = 2 := h2
  rw [h, k]
  sorry

end compute_expression_l74_7421


namespace cristina_speed_cristina_running_speed_l74_7413

theorem cristina_speed 
  (head_start : ℕ)
  (nicky_speed : ℕ)
  (catch_up_time : ℕ)
  (distance : ℕ := head_start + (nicky_speed * catch_up_time))
  : distance / catch_up_time = 6
  := by
  sorry

-- Given conditions used as definitions in Lean 4:
-- head_start = 36 (meters)
-- nicky_speed = 3 (meters/second)
-- catch_up_time = 12 (seconds)

theorem cristina_running_speed
  (head_start : ℕ := 36)
  (nicky_speed : ℕ := 3)
  (catch_up_time : ℕ := 12)
  : (head_start + (nicky_speed * catch_up_time)) / catch_up_time = 6
  := by
  sorry

end cristina_speed_cristina_running_speed_l74_7413


namespace situps_together_l74_7457

theorem situps_together (hani_rate diana_rate : ℕ) (diana_situps diana_time hani_situps total_situps : ℕ)
  (h1 : hani_rate = diana_rate + 3)
  (h2 : diana_rate = 4)
  (h3 : diana_situps = 40)
  (h4 : diana_time = diana_situps / diana_rate)
  (h5 : hani_situps = hani_rate * diana_time)
  (h6 : total_situps = diana_situps + hani_situps) : 
  total_situps = 110 :=
sorry

end situps_together_l74_7457


namespace triangle_altitude_l74_7486

theorem triangle_altitude (A b : ℝ) (h : ℝ) 
  (hA : A = 750) 
  (hb : b = 50) 
  (area_formula : A = (1 / 2) * b * h) : 
  h = 30 :=
  sorry

end triangle_altitude_l74_7486


namespace lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l74_7461

/- Define lines l1 and l2 -/
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/- Prove intersection condition -/
theorem lines_intersect (a : ℝ) : (∃ x y, l1 a x y ∧ l2 a x y) ↔ (a ≠ -1 ∧ a ≠ 2) := 
sorry

/- Prove perpendicular condition -/
theorem lines_perpendicular (a : ℝ) : (∃ x1 y1 x2 y2, l1 a x1 y1 ∧ l2 a x2 y2 ∧ x1 * x2 + y1 * y2 = 0) ↔ (a = 2 / 3) :=
sorry

/- Prove coincident condition -/
theorem lines_coincide (a : ℝ) : (∀ x y, l1 a x y ↔ l2 a x y) ↔ (a = 2) := 
sorry

/- Prove parallel condition -/
theorem lines_parallel (a : ℝ) : (∀ x1 y1 x2 y2, l1 a x1 y1 → l2 a x2 y2 → (x1 * y2 - y1 * x2) = 0) ↔ (a = -1) := 
sorry

end lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l74_7461


namespace ana_bonita_age_difference_l74_7404

theorem ana_bonita_age_difference (A B n : ℕ) 
  (h1 : A = B + n)
  (h2 : A - 1 = 7 * (B - 1))
  (h3 : A = B^3) : 
  n = 6 :=
sorry

end ana_bonita_age_difference_l74_7404


namespace johns_initial_playtime_l74_7494

theorem johns_initial_playtime :
  ∃ (x : ℝ), (14 * x = 0.40 * (14 * x + 84)) → x = 4 :=
by
  sorry

end johns_initial_playtime_l74_7494


namespace event_day_price_l74_7430

theorem event_day_price (original_price : ℝ) (first_discount second_discount : ℝ)
  (h1 : original_price = 250) (h2 : first_discount = 0.4) (h3 : second_discount = 0.25) : 
  ∃ discounted_price : ℝ, 
  discounted_price = (original_price * (1 - first_discount)) * (1 - second_discount) → 
  discounted_price = 112.5 :=
by
  use (250 * (1 - 0.4) * (1 - 0.25))
  sorry

end event_day_price_l74_7430


namespace james_marbles_l74_7443

def marbles_in_bag_D (bag_C : ℕ) := 2 * bag_C - 1
def marbles_in_bag_E (bag_A : ℕ) := bag_A / 2
def marbles_in_bag_G (bag_E : ℕ) := bag_E

theorem james_marbles :
    ∀ (A B C D E F G : ℕ),
      A = 4 →
      B = 3 →
      C = 5 →
      D = marbles_in_bag_D C →
      E = marbles_in_bag_E A →
      F = 3 →
      G = marbles_in_bag_G E →
      28 - (D + F) + 4 = 20 := by
    intros A B C D E F G hA hB hC hD hE hF hG
    sorry

end james_marbles_l74_7443


namespace people_on_bus_now_l74_7437

variable (x : ℕ)

def original_people_on_bus : ℕ := 38
def people_got_on_bus (x : ℕ) : ℕ := x
def people_left_bus (x : ℕ) : ℕ := x + 9

theorem people_on_bus_now (x : ℕ) : original_people_on_bus - people_left_bus x + people_got_on_bus x = 29 := 
by
  sorry

end people_on_bus_now_l74_7437


namespace coin_flip_probability_difference_l74_7411

theorem coin_flip_probability_difference :
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  abs (p3 - p4) = (7/16:ℝ) :=
by
  let p3 := (Nat.choose 4 3) * (1/2:ℝ)^3 * (1/2:ℝ)
  let p4 := (1/2:ℝ)^4
  sorry

end coin_flip_probability_difference_l74_7411


namespace prob1_prob2_prob3_l74_7481

-- Define the sequences for rows ①, ②, and ③
def seq1 (n : ℕ) : ℤ := (-2) ^ n
def seq2 (m : ℕ) : ℤ := (-2) ^ (m - 1)
def seq3 (m : ℕ) : ℤ := (-2) ^ (m - 1) - 1

-- Prove the $n^{th}$ number in row ①
theorem prob1 (n : ℕ) : seq1 n = (-2) ^ n :=
by sorry

-- Prove the relationship between $m^{th}$ numbers in row ② and row ③
theorem prob2 (m : ℕ) : seq3 m = seq2 m - 1 :=
by sorry

-- Prove the value of $x + y + z$ where $x$, $y$, and $z$ are the $2019^{th}$ numbers in rows ①, ②, and ③, respectively
theorem prob3 : seq1 2019 + seq2 2019 + seq3 2019 = -1 :=
by sorry

end prob1_prob2_prob3_l74_7481


namespace coordinates_of_point_l74_7479

theorem coordinates_of_point (x : ℝ) (P : ℝ × ℝ) (h : P = (1 - x, 2 * x + 1)) (y_axis : P.1 = 0) : P = (0, 3) :=
by
  sorry

end coordinates_of_point_l74_7479


namespace william_max_riding_time_l74_7451

theorem william_max_riding_time (x : ℝ) :
  (2 * x + 2 * 1.5 + 2 * (1 / 2 * x) = 21) → (x = 6) :=
by
  sorry

end william_max_riding_time_l74_7451


namespace min_xy_of_conditions_l74_7499

open Real

theorem min_xy_of_conditions
  (x y : ℝ)
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy_eq : 1 / (2 + x) + 1 / (2 + y) = 1 / 3) : 
  xy ≥ 16 :=
by
  sorry

end min_xy_of_conditions_l74_7499


namespace square_side_length_l74_7448

theorem square_side_length (length_rect width_rect : ℕ) (h_length : length_rect = 400) (h_width : width_rect = 300)
  (h_perimeter : 4 * side_length = 2 * (2 * (length_rect + width_rect))) : side_length = 700 := by
  -- Proof goes here
  sorry

end square_side_length_l74_7448


namespace number_of_markings_l74_7414

def markings (L : ℕ → ℕ) := ∀ n, (n > 0) → L n = L (n - 1) + 1

theorem number_of_markings : ∃ L : ℕ → ℕ, (∀ n, n = 1 → L n = 2) ∧ markings L ∧ L 200 = 201 := 
sorry

end number_of_markings_l74_7414


namespace intersection_of_A_and_B_l74_7489

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a - 1}

-- The main statement to prove
theorem intersection_of_A_and_B : A ∩ B = {1, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l74_7489


namespace circle_B_area_l74_7407

theorem circle_B_area
  (r R : ℝ)
  (h1 : ∀ (x : ℝ), x = 5)  -- derived from r = 5
  (h2 : R = 2 * r)
  (h3 : 25 * Real.pi = Real.pi * r^2)
  (h4 : R = 10)  -- derived from diameter relation
  : ∃ A_B : ℝ, A_B = 100 * Real.pi :=
by
  sorry

end circle_B_area_l74_7407


namespace determine_k_l74_7441

theorem determine_k (k : ℕ) : 2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 → k = 3 :=
by
  intro h
  -- now we would proceed to prove it, but we'll skip proof here
  sorry

end determine_k_l74_7441


namespace problem_dividing_remainder_l74_7467

-- The conditions exported to Lean
def tiling_count (n : ℕ) : ℕ :=
  -- This function counts the number of valid tilings for a board size n with all colors used
  sorry

def remainder_when_divide (num divisor : ℕ) : ℕ := num % divisor

-- The statement problem we need to prove
theorem problem_dividing_remainder :
  remainder_when_divide (tiling_count 9) 1000 = 545 := 
sorry

end problem_dividing_remainder_l74_7467


namespace interest_rate_for_first_part_l74_7498

def sum_amount : ℝ := 2704
def part2 : ℝ := 1664
def part1 : ℝ := sum_amount - part2
def rate2 : ℝ := 0.05
def years2 : ℝ := 3
def interest2 : ℝ := part2 * rate2 * years2
def years1 : ℝ := 8

theorem interest_rate_for_first_part (r1 : ℝ) :
  part1 * r1 * years1 = interest2 → r1 = 0.03 :=
by
  sorry

end interest_rate_for_first_part_l74_7498


namespace remainder_three_l74_7426

-- Define the condition that x % 6 = 3
def condition (x : ℕ) : Prop := x % 6 = 3

-- Proof statement that if condition is met, then (3 * x) % 6 = 3
theorem remainder_three {x : ℕ} (h : condition x) : (3 * x) % 6 = 3 :=
sorry

end remainder_three_l74_7426


namespace number_exceeds_its_3_over_8_part_by_20_l74_7473

theorem number_exceeds_its_3_over_8_part_by_20 (x : ℝ) (h : x = (3 / 8) * x + 20) : x = 32 :=
by
  sorry

end number_exceeds_its_3_over_8_part_by_20_l74_7473


namespace sphere_in_cone_volume_l74_7400

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem sphere_in_cone_volume :
  let d := 12
  let θ := 45
  let r := 3 * Real.sqrt 2
  let V := volume_of_sphere r
  d = 12 → θ = 45 → V = 72 * Real.sqrt 2 * Real.pi := by
  intros h1 h2
  sorry

end sphere_in_cone_volume_l74_7400


namespace equal_areas_greater_perimeter_l74_7439

noncomputable def side_length_square := Real.sqrt 3 + 3

noncomputable def length_rectangle := Real.sqrt 72 + 3 * Real.sqrt 6
noncomputable def width_rectangle := Real.sqrt 2

noncomputable def area_square := (side_length_square) ^ 2

noncomputable def area_rectangle := length_rectangle * width_rectangle

noncomputable def perimeter_square := 4 * side_length_square

noncomputable def perimeter_rectangle := 2 * (length_rectangle + width_rectangle)

theorem equal_areas : area_square = area_rectangle := sorry

theorem greater_perimeter : perimeter_square < perimeter_rectangle := sorry

end equal_areas_greater_perimeter_l74_7439


namespace isosceles_triangle_same_area_l74_7482

-- Given conditions of the original isosceles triangle
def original_base : ℝ := 10
def original_side : ℝ := 13

-- The problem states that an isosceles triangle has the base 10 cm and side lengths 13 cm, 
-- we need to show there's another isosceles triangle with a different base but the same area.
theorem isosceles_triangle_same_area : 
  ∃ (new_base : ℝ) (new_side : ℝ), 
    new_base ≠ original_base ∧ 
    (∃ (h1 h2: ℝ), 
      h1 = 12 ∧ 
      h2 = 5 ∧
      1/2 * original_base * h1 = 60 ∧ 
      1/2 * new_base * h2 = 60) := 
sorry

end isosceles_triangle_same_area_l74_7482


namespace total_people_3522_l74_7456

def total_people (M W: ℕ) : ℕ := M + W

theorem total_people_3522 
    (M W: ℕ) 
    (h1: M / 9 * 45 + W / 12 * 60 = 17760)
    (h2: M % 9 = 0)
    (h3: W % 12 = 0) : 
    total_people M W = 3552 :=
by {
  sorry
}

end total_people_3522_l74_7456


namespace range_of_a_for_inequality_l74_7427

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ p q : ℝ, (0 < p ∧ p < 1) → (0 < q ∧ q < 1) → p ≠ q → (f a p - f a q) / (p - q) > 1) ↔ 3 ≤ a :=
sorry

end range_of_a_for_inequality_l74_7427


namespace smallest_angle_CBD_l74_7424

-- Definitions for given conditions
def angle_ABC : ℝ := 40
def angle_ABD : ℝ := 15

-- Theorem statement
theorem smallest_angle_CBD : ∃ (angle_CBD : ℝ), angle_CBD = angle_ABC - angle_ABD := by
  use 25
  sorry

end smallest_angle_CBD_l74_7424


namespace line_through_intersections_of_circles_l74_7428

-- Define the first circle
def circle₁ (x y : ℝ) : Prop :=
  x^2 + y^2 = 10

-- Define the second circle
def circle₂ (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 20

-- The statement of the mathematically equivalent proof problem
theorem line_through_intersections_of_circles : 
    (∃ (x y : ℝ), circle₁ x y ∧ circle₂ x y) → (∃ (x y : ℝ), x + 3 * y - 5 = 0) :=
by
  intro h
  sorry

end line_through_intersections_of_circles_l74_7428


namespace complex_evaluation_l74_7495

theorem complex_evaluation (a b : ℂ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a^2 + a * b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := 
by 
  sorry

end complex_evaluation_l74_7495


namespace hyperbola_asymptotes_m_value_l74_7405

theorem hyperbola_asymptotes_m_value : 
    (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 1) → (y = (3/4) * x ∨ y = -(3/4) * x)) := 
by sorry

end hyperbola_asymptotes_m_value_l74_7405


namespace tax_free_amount_is_600_l74_7402

variable (X : ℝ) -- X is the tax-free amount

-- Given conditions
variable (total_value : ℝ := 1720)
variable (tax_paid : ℝ := 89.6)
variable (tax_rate : ℝ := 0.08)

-- Proof problem
theorem tax_free_amount_is_600
  (h1 : 0.08 * (total_value - X) = tax_paid) :
  X = 600 :=
by
  sorry

end tax_free_amount_is_600_l74_7402


namespace distance_travelled_first_hour_l74_7454

noncomputable def initial_distance (x : ℕ) : Prop :=
  let distance_travelled := (12 / 2) * (2 * x + (12 - 1) * 2)
  distance_travelled = 552

theorem distance_travelled_first_hour : ∃ x : ℕ, initial_distance x ∧ x = 35 :=
by
  use 35
  unfold initial_distance
  sorry

end distance_travelled_first_hour_l74_7454


namespace water_consumption_comparison_l74_7496

-- Define the given conditions
def waterConsumptionWest : ℝ := 21428
def waterConsumptionNonWest : ℝ := 26848.55
def waterConsumptionRussia : ℝ := 302790.13

-- Theorem statement to prove that the water consumption per person matches the given values
theorem water_consumption_comparison :
  waterConsumptionWest = 21428 ∧
  waterConsumptionNonWest = 26848.55 ∧
  waterConsumptionRussia = 302790.13 :=
by
  -- Sorry to skip the proof
  sorry

end water_consumption_comparison_l74_7496


namespace planes_parallel_or_intersect_l74_7462

variables {Plane : Type} {Line : Type}
variables (α β : Plane) (a b : Line)

-- Conditions
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def not_parallel (l1 l2 : Line) : Prop := sorry

-- Given conditions
axiom h₁ : line_in_plane a α
axiom h₂ : line_in_plane b β
axiom h₃ : not_parallel a b

-- The theorem statement
theorem planes_parallel_or_intersect : (exists l : Line, line_in_plane l α ∧ line_in_plane l β) ∨ (α = β) :=
sorry

end planes_parallel_or_intersect_l74_7462


namespace general_term_sum_first_n_terms_l74_7472

variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}
variable (d : ℝ) (h1 : d ≠ 0)
variable (a10 : a 10 = 19)
variable (geo_seq : ∀ {x y z}, x * z = y ^ 2 → x = 1 → y = a 2 → z = a 5)
variable (arith_seq : ∀ n, a n = a 1 + (n - 1) * d)

-- General term of the arithmetic sequence
theorem general_term (a_1 : ℝ) (h1 : a 1 = a_1) : a n = 2 * n - 1 :=
sorry

-- Sum of the first n terms of the sequence b_n
theorem sum_first_n_terms (n : ℕ) : S n = (2 * n - 3) * 2^(n + 1) + 6 :=
sorry

end general_term_sum_first_n_terms_l74_7472


namespace line_slope_intercept_sum_l74_7408

theorem line_slope_intercept_sum (m b : ℝ)
    (h1 : m = 4)
    (h2 : ∃ b, ∀ x y : ℝ, y = mx + b → y = 5 ∧ x = -2)
    : m + b = 17 := by
  sorry

end line_slope_intercept_sum_l74_7408


namespace part_a_part_b_l74_7420

theorem part_a {a b c : ℝ} : ∃ x : ℝ, (x - a) * (x - b) + (x - b) * (x - c) + (x - c) * (x - a) = 0 :=
sorry

theorem part_b {a b c : ℝ} : (a + b + c) ^ 2 ≥ 3 * (a * b + b * c + c * a) :=
sorry

end part_a_part_b_l74_7420


namespace more_boys_than_girls_l74_7474

noncomputable def class1_4th_girls : ℕ := 12
noncomputable def class1_4th_boys : ℕ := 13
noncomputable def class2_4th_girls : ℕ := 15
noncomputable def class2_4th_boys : ℕ := 11

noncomputable def class1_5th_girls : ℕ := 9
noncomputable def class1_5th_boys : ℕ := 13
noncomputable def class2_5th_girls : ℕ := 10
noncomputable def class2_5th_boys : ℕ := 11

noncomputable def total_4th_girls : ℕ := class1_4th_girls + class2_4th_girls
noncomputable def total_4th_boys : ℕ := class1_4th_boys + class2_4th_boys

noncomputable def total_5th_girls : ℕ := class1_5th_girls + class2_5th_girls
noncomputable def total_5th_boys : ℕ := class1_5th_boys + class2_5th_boys

noncomputable def total_girls : ℕ := total_4th_girls + total_5th_girls
noncomputable def total_boys : ℕ := total_4th_boys + total_5th_boys

theorem more_boys_than_girls :
  (total_boys - total_girls) = 2 :=
by
  -- placeholder for the proof
  sorry

end more_boys_than_girls_l74_7474


namespace product_is_zero_l74_7464

variables {a b c d : ℤ}

def system_of_equations (a b c d : ℤ) :=
  2 * a + 3 * b + 5 * c + 7 * d = 34 ∧
  3 * (d + c) = b ∧
  3 * b + c = a ∧
  c - 1 = d

theorem product_is_zero (h : system_of_equations a b c d) : 
  a * b * c * d = 0 :=
sorry

end product_is_zero_l74_7464


namespace domain_of_f_l74_7401

open Real

noncomputable def f (x : ℝ) : ℝ := log ((2 - x) / (2 + x))

theorem domain_of_f : ∀ x : ℝ, (2 - x) / (2 + x) > 0 ∧ 2 + x ≠ 0 ↔ -2 < x ∧ x < 2 :=
by
  intro x
  sorry

end domain_of_f_l74_7401


namespace area_increase_percentage_l74_7419

variable (r : ℝ) (π : ℝ := Real.pi)

theorem area_increase_percentage (h₁ : r > 0) (h₂ : π > 0) : 
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  (new_area - original_area) / original_area * 100 = 525 := 
by
  let new_radius := 2.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  sorry

end area_increase_percentage_l74_7419


namespace relationship_among_a_b_c_l74_7422

noncomputable def a : ℝ := (0.6:ℝ) ^ (0.2:ℝ)
noncomputable def b : ℝ := (0.2:ℝ) ^ (0.2:ℝ)
noncomputable def c : ℝ := (0.2:ℝ) ^ (0.6:ℝ)

theorem relationship_among_a_b_c : a > b ∧ b > c :=
by
  -- The proof can be added here if needed
  sorry

end relationship_among_a_b_c_l74_7422


namespace T_n_formula_l74_7487

-- Define the given sequence sum S_n
def S (n : ℕ) : ℚ := (n^2 : ℚ) / 2 + (3 * n : ℚ) / 2

-- Define the general term a_n for the sequence {a_n}
def a (n : ℕ) : ℚ := if n = 1 then 2 else n + 1

-- Define the sequence b_n
def b (n : ℕ) : ℚ := a (n + 2) - a n + 1 / (a (n + 2) * a n)

-- Define the sum of the first n terms of the sequence {b_n}
def T (n : ℕ) : ℚ := 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3))

-- Prove the equality of T_n with the given expression
theorem T_n_formula (n : ℕ) : T n = 2 * n + 5 / 12 - (2 * n + 5) / (2 * (n + 2) * (n + 3)) := sorry

end T_n_formula_l74_7487


namespace warriors_won_40_games_l74_7449

variable (H F W K R S : ℕ)

-- Conditions as given in the problem
axiom hawks_won_more_games_than_falcons : H > F
axiom knights_won_more_than_30 : K > 30
axiom warriors_won_more_than_knights_but_fewer_than_royals : W > K ∧ W < R
axiom squires_tied_with_falcons : S = F

-- The proof statement
theorem warriors_won_40_games : W = 40 :=
sorry

end warriors_won_40_games_l74_7449


namespace phi_value_l74_7450

noncomputable def f (x φ : ℝ) := Real.sin (2 * x + φ)

theorem phi_value (φ : ℝ) (h1 : ∀ x : ℝ, f x φ ≤ |f (π / 6) φ|) (h2 : f (π / 3) φ > f (π / 2) φ) : φ = π / 6 :=
by
  sorry

end phi_value_l74_7450


namespace range_of_sum_of_reciprocals_l74_7415

theorem range_of_sum_of_reciprocals (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  ∃ (r : ℝ), ∀ t ∈ Set.Ici (3 + 2 * Real.sqrt 2), t = (1 / x + 1 / y) := 
sorry

end range_of_sum_of_reciprocals_l74_7415


namespace head_start_l74_7446

theorem head_start (V_b : ℝ) (S : ℝ) : 
  ((7 / 4) * V_b) = V_b → 
  196 = (196 - S) → 
  S = 84 := 
sorry

end head_start_l74_7446


namespace record_loss_of_300_l74_7488

-- Definitions based on conditions
def profit (x : Int) : String := "+" ++ toString x
def loss (x : Int) : String := "-" ++ toString x

-- The theorem to prove that a loss of 300 is recorded as "-300" based on the recording system
theorem record_loss_of_300 : loss 300 = "-300" :=
by
  sorry

end record_loss_of_300_l74_7488


namespace prime_of_two_pow_sub_one_prime_l74_7469

theorem prime_of_two_pow_sub_one_prime {n : ℕ} (h : Nat.Prime (2^n - 1)) : Nat.Prime n :=
sorry

end prime_of_two_pow_sub_one_prime_l74_7469


namespace bananas_per_friend_l74_7429

-- Define constants and conditions
def totalBananas : Nat := 40
def totalFriends : Nat := 40

-- Define the main theorem to prove
theorem bananas_per_friend : totalBananas / totalFriends = 1 := by
  sorry

end bananas_per_friend_l74_7429


namespace company_pays_300_per_month_l74_7434

theorem company_pays_300_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box_per_month : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1080000)
  (h5 : cost_per_box_per_month = 0.5) :
  (total_volume / (length * width * height)) * cost_per_box_per_month = 300 := by
  sorry

end company_pays_300_per_month_l74_7434


namespace pyramid_boxes_l74_7445

theorem pyramid_boxes (a₁ a₂ aₙ : ℕ) (d : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12) 
  (h₂ : a₂ = 15) 
  (h₃ : aₙ = 39) 
  (h₄ : d = 3) 
  (h₅ : a₂ = a₁ + d)
  (h₆ : aₙ = a₁ + (n - 1) * d) 
  (h₇ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 255 :=
by
  sorry

end pyramid_boxes_l74_7445


namespace speed_of_boat_l74_7468

-- Given conditions
variables (V_b : ℝ) (V_s : ℝ) (T : ℝ) (D : ℝ)

-- Problem statement in Lean
theorem speed_of_boat (h1 : V_s = 5) (h2 : T = 1) (h3 : D = 45) :
  D = T * (V_b + V_s) → V_b = 40 := 
by
  intro h4
  rw [h1, h2, h3] at h4
  linarith

end speed_of_boat_l74_7468


namespace find_k_l74_7484

theorem find_k (m n k : ℝ) 
  (h1 : 3^m = k) 
  (h2 : 5^n = k) 
  (h3 : 1/m + 1/n = 2) : 
  k = Real.sqrt 15 := 
sorry

end find_k_l74_7484


namespace Mika_water_left_l74_7463

theorem Mika_water_left :
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  initial_amount - used_amount = 5 / 4 :=
by
  let initial_amount := 7 / 2
  let used_amount := 9 / 4
  show initial_amount - used_amount = 5 / 4
  sorry

end Mika_water_left_l74_7463


namespace average_of_pqrs_l74_7475

theorem average_of_pqrs (p q r s : ℚ) (h : (5/4) * (p + q + r + s) = 20) : ((p + q + r + s) / 4) = 4 :=
sorry

end average_of_pqrs_l74_7475


namespace bears_on_each_shelf_l74_7417

theorem bears_on_each_shelf (initial_bears : ℕ) (additional_bears : ℕ) (shelves : ℕ) (total_bears : ℕ) (bears_per_shelf : ℕ) :
  initial_bears = 5 → additional_bears = 7 → shelves = 2 → total_bears = initial_bears + additional_bears → bears_per_shelf = total_bears / shelves → bears_per_shelf = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end bears_on_each_shelf_l74_7417


namespace calculate_truncated_cone_volume_l74_7435

noncomputable def volume_of_truncated_cone (R₁ R₂ h : ℝ) :
    ℝ := ((1 / 3) * Real.pi * h * (R₁ ^ 2 + R₁ * R₂ + R₂ ^ 2))

theorem calculate_truncated_cone_volume : 
    volume_of_truncated_cone 10 5 10 = (1750 / 3) * Real.pi := by
sorry

end calculate_truncated_cone_volume_l74_7435


namespace number_of_five_ruble_coins_l74_7425

theorem number_of_five_ruble_coins (total_coins a b c : Nat) (h1 : total_coins = 25) (h2 : 19 = total_coins - a) (h3 : 20 = total_coins - b) (h4 : 16 = total_coins - c) :
  total_coins - (a + b + c) = 5 :=
by
  sorry

end number_of_five_ruble_coins_l74_7425


namespace negation_of_exists_l74_7409

-- Lean definition of the proposition P
def P (a : ℝ) : Prop :=
  ∃ x0 : ℝ, x0 > 0 ∧ 2^x0 * (x0 - a) > 1

-- The negation of the proposition P
def neg_P (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 2^x * (x - a) ≤ 1

-- Theorem stating that the negation of P is neg_P
theorem negation_of_exists (a : ℝ) : ¬ P a ↔ neg_P a :=
by
  -- (Proof to be provided)
  sorry

end negation_of_exists_l74_7409


namespace pascals_triangle_53_rows_l74_7418

theorem pascals_triangle_53_rows : 
  ∃! row, (∃ k, 1 ≤ k ∧ k ≤ row ∧ 53 = Nat.choose row k) ∧ 
          (∀ k, 1 ≤ k ∧ k ≤ row → 53 = Nat.choose row k → row = 53) :=
sorry

end pascals_triangle_53_rows_l74_7418


namespace line_passes_through_fixed_point_l74_7433

-- Given a line equation kx - y + 1 - 3k = 0
def line_equation (k x y : ℝ) : Prop := k * x - y + 1 - 3 * k = 0

-- We need to prove that this line passes through the point (3,1)
theorem line_passes_through_fixed_point (k : ℝ) : line_equation k 3 1 :=
by
  sorry

end line_passes_through_fixed_point_l74_7433


namespace three_hour_classes_per_week_l74_7412

theorem three_hour_classes_per_week (x : ℕ) : 
  (24 * (3 * x + 4 + 4) = 336) → x = 2 := by {
  sorry
}

end three_hour_classes_per_week_l74_7412


namespace Mike_ride_distance_l74_7477

theorem Mike_ride_distance 
  (M : ℕ)
  (total_cost_Mike : ℝ)
  (total_cost_Annie : ℝ)
  (h1 : total_cost_Mike = 4.50 + 0.30 * M)
  (h2: total_cost_Annie = 15.00)
  (h3: total_cost_Mike = total_cost_Annie) : 
  M = 35 := 
by
  sorry

end Mike_ride_distance_l74_7477


namespace chick_hits_at_least_five_l74_7497

theorem chick_hits_at_least_five (x y z : ℕ) (h1 : 9 * x + 5 * y + 2 * z = 61) (h2 : x + y + z = 10) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : x ≥ 5 :=
sorry

end chick_hits_at_least_five_l74_7497


namespace find_multiplier_l74_7491

theorem find_multiplier (x : ℝ) (y : ℝ) (h1 : x = 62.5) (h2 : (y * (x + 5)) / 5 - 5 = 22) : y = 2 :=
sorry

end find_multiplier_l74_7491


namespace quadratic_graph_above_x_axis_l74_7444

theorem quadratic_graph_above_x_axis (a b c : ℝ) :
  ¬ ((b^2 - 4*a*c < 0) ↔ ∀ x : ℝ, a*x^2 + b*x + c > 0) :=
sorry

end quadratic_graph_above_x_axis_l74_7444


namespace greatest_number_of_large_chips_l74_7431

theorem greatest_number_of_large_chips (s l p : ℕ) (h1 : s + l = 60) (h2 : s = l + p) 
  (hp_prime : Nat.Prime p) (hp_div : p ∣ l) : l ≤ 29 :=
by
  sorry

end greatest_number_of_large_chips_l74_7431


namespace find_k_value_l74_7466

theorem find_k_value (k : ℚ) (h1 : (3, -5) ∈ {p : ℚ × ℚ | p.snd = k * p.fst}) (h2 : k ≠ 0) : k = -5 / 3 :=
sorry

end find_k_value_l74_7466


namespace evaluate_f_at_2_l74_7471

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x + 1

theorem evaluate_f_at_2 : f 2 = 259 := 
by
  -- Substitute x = 2 into the polynomial and simplify the expression.
  sorry

end evaluate_f_at_2_l74_7471


namespace consumer_installment_credit_l74_7406

theorem consumer_installment_credit : 
  ∃ C : ℝ, 
    (0.43 * C = 200) ∧ 
    (C = 465.116) :=
by
  sorry

end consumer_installment_credit_l74_7406


namespace current_at_time_l74_7455

noncomputable def I (t : ℝ) : ℝ := 5 * (Real.sin (100 * Real.pi * t + Real.pi / 3))

theorem current_at_time (t : ℝ) (h : t = 1 / 200) : I t = 5 / 2 := by
  sorry

end current_at_time_l74_7455


namespace sum_of_arithmetic_series_l74_7485

theorem sum_of_arithmetic_series (a1 an : ℕ) (d n : ℕ) (s : ℕ) :
  a1 = 2 ∧ an = 100 ∧ d = 2 ∧ n = (an - a1) / d + 1 ∧ s = n * (a1 + an) / 2 → s = 2550 :=
by
  sorry

end sum_of_arithmetic_series_l74_7485
