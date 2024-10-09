import Mathlib

namespace probability_of_hitting_target_at_least_once_l616_61679

noncomputable def prob_hit_target_once : ℚ := 2/3

noncomputable def prob_miss_target_once : ℚ := 1 - prob_hit_target_once

noncomputable def prob_miss_target_three_times : ℚ := prob_miss_target_once ^ 3

noncomputable def prob_hit_target_at_least_once : ℚ := 1 - prob_miss_target_three_times

theorem probability_of_hitting_target_at_least_once :
  prob_hit_target_at_least_once = 26 / 27 := 
sorry

end probability_of_hitting_target_at_least_once_l616_61679


namespace perimeter_of_first_square_l616_61662

theorem perimeter_of_first_square (p1 p2 p3 : ℕ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24) :
  p1 = 40 := 
  sorry

end perimeter_of_first_square_l616_61662


namespace quotient_is_10_l616_61616

theorem quotient_is_10 (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 161)
  (h2 : divisor = 16)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  quotient = 10 := 
by
  sorry

end quotient_is_10_l616_61616


namespace shift_parabola_upwards_l616_61642

theorem shift_parabola_upwards (y x : ℝ) (h : y = x^2) : y + 5 = (x^2 + 5) := by 
  sorry

end shift_parabola_upwards_l616_61642


namespace train_length_at_constant_acceleration_l616_61638

variables (u : ℝ) (t : ℝ) (a : ℝ) (s : ℝ)

theorem train_length_at_constant_acceleration (h₁ : u = 16.67) (h₂ : t = 30) : 
  s = u * t + 0.5 * a * t^2 :=
sorry

end train_length_at_constant_acceleration_l616_61638


namespace fourth_vertex_parallelogram_coordinates_l616_61682

def fourth_vertex_of_parallelogram (A B C : ℝ × ℝ) :=
  ∃ D : ℝ × ℝ, (D = (11, 4) ∨ D = (-1, 12) ∨ D = (3, -12))

theorem fourth_vertex_parallelogram_coordinates :
  fourth_vertex_of_parallelogram (1, 0) (5, 8) (7, -4) :=
by
  sorry

end fourth_vertex_parallelogram_coordinates_l616_61682


namespace intersection_eq_l616_61615

def setA : Set ℕ := {0, 1, 2, 3, 4, 5 }
def setB : Set ℕ := { x | |(x : ℤ) - 2| ≤ 1 }

theorem intersection_eq :
  setA ∩ setB = {1, 2, 3} := by
  sorry

end intersection_eq_l616_61615


namespace find_a_l616_61684

theorem find_a (x y a : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - y = 3) : a = 2 :=
by
  sorry

end find_a_l616_61684


namespace quadratic_no_real_solutions_l616_61683

theorem quadratic_no_real_solutions (a : ℝ) (h₀ : 0 < a) (h₁ : a^3 = 6 * (a + 1)) : 
  ∀ x : ℝ, ¬ (x^2 + a * x + a^2 - 6 = 0) :=
by
  sorry

end quadratic_no_real_solutions_l616_61683


namespace randys_trip_length_l616_61619

theorem randys_trip_length
  (trip_length : ℚ)
  (fraction_gravel : trip_length = (1 / 4) * trip_length)
  (middle_miles : 30 = (7 / 12) * trip_length)
  (fraction_dirt : trip_length = (1 / 6) * trip_length) :
  trip_length = 360 / 7 :=
by
  sorry

end randys_trip_length_l616_61619


namespace triangle_is_obtuse_l616_61643

noncomputable def is_exterior_smaller (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle < interior_angle

noncomputable def sum_of_angles (exterior_angle interior_angle : ℝ) : Prop :=
  exterior_angle + interior_angle = 180

theorem triangle_is_obtuse (exterior_angle interior_angle : ℝ) (h1 : is_exterior_smaller exterior_angle interior_angle) 
  (h2 : sum_of_angles exterior_angle interior_angle) : ∃ b, 90 < b ∧ b = interior_angle :=
sorry

end triangle_is_obtuse_l616_61643


namespace neg_p_equiv_l616_61660

theorem neg_p_equiv :
  (¬ (∀ x : ℝ, x > 0 → x - Real.log x > 0)) ↔ (∃ x_0 : ℝ, x_0 > 0 ∧ x_0 - Real.log x_0 ≤ 0) :=
by
  sorry

end neg_p_equiv_l616_61660


namespace foci_coordinates_l616_61617

-- Define the parameters for the hyperbola
def a_squared : ℝ := 3
def b_squared : ℝ := 1
def c_squared : ℝ := a_squared + b_squared

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

-- State the theorem about the coordinates of the foci
theorem foci_coordinates : {foci : ℝ × ℝ // foci = (-2, 0) ∨ foci = (2, 0)} :=
by 
  have ha : a_squared = 3 := rfl
  have hb : b_squared = 1 := rfl
  have hc : c_squared = a_squared + b_squared := rfl
  have c := Real.sqrt c_squared
  have hc' : c = 2 := 
  -- sqrt part can be filled if detailed, for now, just direct conclusion
  sorry
  exact ⟨(2, 0), Or.inr rfl⟩

end foci_coordinates_l616_61617


namespace father_seven_times_as_old_l616_61670

theorem father_seven_times_as_old (x : ℕ) (father_age : ℕ) (son_age : ℕ) :
  father_age = 38 → son_age = 14 → (father_age - x = 7 * (son_age - x) → x = 10) :=
by
  intros h_father_age h_son_age h_equation
  rw [h_father_age, h_son_age] at h_equation
  sorry

end father_seven_times_as_old_l616_61670


namespace square_areas_l616_61614

variables (a b : ℝ)

def is_perimeter_difference (a b : ℝ) : Prop :=
  4 * a - 4 * b = 12

def is_area_difference (a b : ℝ) : Prop :=
  a^2 - b^2 = 69

theorem square_areas (a b : ℝ) (h1 : is_perimeter_difference a b) (h2 : is_area_difference a b) :
  a^2 = 169 ∧ b^2 = 100 :=
by {
  sorry
}

end square_areas_l616_61614


namespace logarithmic_inequality_l616_61677

noncomputable def log_a_b (a b : ℝ) := Real.log b / Real.log a

theorem logarithmic_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) :
  log_a_b a b + log_a_b b c + log_a_b a c ≥ 3 :=
by
  sorry

end logarithmic_inequality_l616_61677


namespace estimated_number_of_red_balls_l616_61644

theorem estimated_number_of_red_balls (total_balls : ℕ) (red_draws : ℕ) (total_draws : ℕ)
    (h_total_balls : total_balls = 8) (h_red_draws : red_draws = 75) (h_total_draws : total_draws = 100) :
    total_balls * (red_draws / total_draws : ℚ) = 6 := 
by
  sorry

end estimated_number_of_red_balls_l616_61644


namespace parrot_initial_phrases_l616_61647

theorem parrot_initial_phrases (current_phrases : ℕ) (days_with_parrot : ℕ) (phrases_per_week : ℕ) (initial_phrases : ℕ) :
  current_phrases = 17 →
  days_with_parrot = 49 →
  phrases_per_week = 2 →
  initial_phrases = current_phrases - phrases_per_week * (days_with_parrot / 7) :=
by
  sorry

end parrot_initial_phrases_l616_61647


namespace fraction_of_height_of_head_l616_61652

theorem fraction_of_height_of_head (h_leg: ℝ) (h_total: ℝ) (h_rest: ℝ) (h_head: ℝ):
  h_leg = 1 / 3 ∧ h_total = 60 ∧ h_rest = 25 ∧ h_head = h_total - (h_leg * h_total + h_rest) 
  → h_head / h_total = 1 / 4 :=
by sorry

end fraction_of_height_of_head_l616_61652


namespace ethel_subtracts_l616_61666

theorem ethel_subtracts (h : 50^2 = 2500) : 2500 - 99 = 49^2 :=
by
  sorry

end ethel_subtracts_l616_61666


namespace Isabela_spent_l616_61610

theorem Isabela_spent (num_pencils : ℕ) (cost_per_item : ℕ) (num_cucumbers : ℕ)
  (h1 : cost_per_item = 20)
  (h2 : num_cucumbers = 100)
  (h3 : num_cucumbers = 2 * num_pencils)
  (discount : ℚ := 0.20) :
  let pencil_cost := num_pencils * cost_per_item
  let cucumber_cost := num_cucumbers * cost_per_item
  let discounted_pencil_cost := pencil_cost * (1 - discount)
  let total_cost := cucumber_cost + discounted_pencil_cost
  total_cost = 2800 := by
  -- Begin proof. We will add actual proof here later.
  sorry

end Isabela_spent_l616_61610


namespace part1_part2_i_part2_ii_l616_61688

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.log x + 1 / Real.exp x

theorem part1 (k : ℝ) (h : ¬ MonotoneOn (f k) (Set.Icc 2 3)) :
  3 / Real.exp 3 < k ∧ k < 2 / Real.exp 2 :=
sorry

variables {x1 x2 : ℝ}
variable (k : ℝ)
variable (h0 : 0 < x1)
variable (h1 : x1 < x2)
variable (h2 : k = x1 / Real.exp x1 ∧ k = x2 / Real.exp x2)

theorem part2_i :
  e / Real.exp x2 - e / Real.exp x1 > -Real.log (x2 / x1) ∧ -Real.log (x2 / x1) > 1 - x2 / x1 :=
sorry

theorem part2_ii : |f k x1 - f k x2| < 1 :=
sorry

end part1_part2_i_part2_ii_l616_61688


namespace avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l616_61671

-- Average fuel consumption per kilometer
noncomputable def avgFuelConsumption (initial_fuel: ℝ) (final_fuel: ℝ) (distance: ℝ) : ℝ :=
  (initial_fuel - final_fuel) / distance

-- Relationship between remaining fuel Q and distance x
noncomputable def remainingFuel (initial_fuel: ℝ) (consumption_rate: ℝ) (distance: ℝ) : ℝ :=
  initial_fuel - consumption_rate * distance

-- Check if the car can return home without refueling
noncomputable def canReturnHome (initial_fuel: ℝ) (consumption_rate: ℝ) (round_trip_distance: ℝ) (alarm_fuel_level: ℝ) : Bool :=
  initial_fuel - consumption_rate * round_trip_distance ≥ alarm_fuel_level

-- Theorem statements to prove
theorem avg_fuel_consumption_correct :
  avgFuelConsumption 45 27 180 = 0.1 :=
sorry

theorem remaining_fuel_correct :
  ∀ x, remainingFuel 45 0.1 x = 45 - 0.1 * x :=
sorry

theorem cannot_return_home_without_refueling :
  ¬canReturnHome 45 0.1 (220 * 2) 3 :=
sorry

end avg_fuel_consumption_correct_remaining_fuel_correct_cannot_return_home_without_refueling_l616_61671


namespace union_of_M_and_N_is_correct_l616_61640

def M : Set ℤ := { m | -3 < m ∧ m < 2 }
def N : Set ℤ := { n | -1 ≤ n ∧ n ≤ 3 }

theorem union_of_M_and_N_is_correct : M ∪ N = { -2, -1, 0, 1, 2, 3 } := 
by
  sorry

end union_of_M_and_N_is_correct_l616_61640


namespace highway_speed_l616_61656

theorem highway_speed 
  (local_distance : ℝ) (local_speed : ℝ)
  (highway_distance : ℝ) (avg_speed : ℝ)
  (h_local : local_distance = 90) 
  (h_local_speed : local_speed = 30)
  (h_highway : highway_distance = 75)
  (h_avg : avg_speed = 38.82) :
  ∃ v : ℝ, v = 60 := 
sorry

end highway_speed_l616_61656


namespace radian_to_degree_equivalent_l616_61657

theorem radian_to_degree_equivalent : 
  (7 / 12) * (180 : ℝ) = 105 :=
by
  sorry

end radian_to_degree_equivalent_l616_61657


namespace new_volume_of_balloon_l616_61634

def initial_volume : ℝ := 2.00  -- Initial volume in liters
def initial_pressure : ℝ := 745  -- Initial pressure in mmHg
def initial_temperature : ℝ := 293.15  -- Initial temperature in Kelvin
def final_pressure : ℝ := 700  -- Final pressure in mmHg
def final_temperature : ℝ := 283.15  -- Final temperature in Kelvin
def final_volume : ℝ := 2.06  -- Expected final volume in liters

theorem new_volume_of_balloon :
  (initial_pressure * initial_volume / initial_temperature) = (final_pressure * final_volume / final_temperature) :=
  sorry  -- Proof to be filled in later

end new_volume_of_balloon_l616_61634


namespace min_side_length_l616_61698

noncomputable def side_length_min : ℝ := 30

theorem min_side_length (s r : ℝ) (hs₁ : s^2 ≥ 900) (hr₁ : π * r^2 ≥ 100) (hr₂ : 2 * r ≤ s) :
  s ≥ side_length_min :=
by
  sorry

end min_side_length_l616_61698


namespace relationship_y1_y2_l616_61635

theorem relationship_y1_y2 (k b y1 y2 : ℝ) (h₀ : k < 0) (h₁ : y1 = k * (-1) + b) (h₂ : y2 = k * 1 + b) : y1 > y2 := 
by
  sorry

end relationship_y1_y2_l616_61635


namespace individual_weights_l616_61626

theorem individual_weights (A P : ℕ) 
    (h1 : 12 * A + 14 * P = 692)
    (h2 : P = A - 10) : 
    A = 32 ∧ P = 22 :=
by
  sorry

end individual_weights_l616_61626


namespace breadth_halved_of_percentage_change_area_l616_61606

theorem breadth_halved_of_percentage_change_area {L B B' : ℝ} (h : 0 < L ∧ 0 < B) 
  (h1 : L / 2 * B' = 0.5 * (L * B)) : B' = 0.5 * B :=
sorry

end breadth_halved_of_percentage_change_area_l616_61606


namespace simplify_sqrt_24_l616_61691

theorem simplify_sqrt_24 : Real.sqrt 24 = 2 * Real.sqrt 6 :=
sorry

end simplify_sqrt_24_l616_61691


namespace max_correct_answers_l616_61602

theorem max_correct_answers :
  ∀ (a b c : ℕ), a + b + c = 60 ∧ 4 * a - c = 112 → a ≤ 34 :=
by
  sorry

end max_correct_answers_l616_61602


namespace tom_days_to_finish_l616_61680

noncomputable def days_to_finish_show
  (episodes : Nat) 
  (minutes_per_episode : Nat) 
  (hours_per_day : Nat) : Nat :=
  let total_minutes := episodes * minutes_per_episode
  let total_hours := total_minutes / 60
  total_hours / hours_per_day

theorem tom_days_to_finish :
  days_to_finish_show 90 20 2 = 15 :=
by
  -- the proof steps go here
  sorry

end tom_days_to_finish_l616_61680


namespace both_locks_stall_time_l616_61624

-- Definitions of the conditions
def first_lock_time : ℕ := 5
def second_lock_time : ℕ := 3 * first_lock_time - 3
def both_locks_time : ℕ := 5 * second_lock_time

-- The proof statement
theorem both_locks_stall_time : both_locks_time = 60 := by
  sorry

end both_locks_stall_time_l616_61624


namespace inequality_proof_l616_61621

variable (a b c d : ℝ)

theorem inequality_proof (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (1 / (1 / a + 1 / b)) + (1 / (1 / c + 1 / d)) ≤ (1 / (1 / (a + c) + 1 / (b + d))) :=
by
  sorry

end inequality_proof_l616_61621


namespace number_of_valid_b_l616_61641

theorem number_of_valid_b : ∃ (bs : Finset ℂ), bs.card = 2 ∧ ∀ b ∈ bs, ∃ (x : ℂ), (x + b = b^2) :=
by
  sorry

end number_of_valid_b_l616_61641


namespace stable_equilibrium_condition_l616_61603

theorem stable_equilibrium_condition
  (a b : ℝ)
  (h_condition1 : a > b)
  (h_condition2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  : (b / a) < (1 / Real.sqrt 2) :=
sorry

end stable_equilibrium_condition_l616_61603


namespace problem_1_problem_2_problem_3_l616_61609

open Real

theorem problem_1 : (1 * (-12)) - (-20) + (-8) - 15 = -15 := by
  sorry

theorem problem_2 : -3^2 + ((2/3) - (1/2) + (5/8)) * (-24) = -28 := by
  sorry

theorem problem_3 : -1^(2023) + 3 * (-2)^2 - (-6) / ((-1/3)^2) = 65 := by
  sorry

end problem_1_problem_2_problem_3_l616_61609


namespace geometric_sequence_a1_cannot_be_2_l616_61659

theorem geometric_sequence_a1_cannot_be_2
  (a : ℕ → ℕ)
  (q : ℕ)
  (h1 : 2 * a 2 + a 3 = a 4)
  (h2 : (a 2 + 1) * (a 3 + 1) = a 5 - 1)
  (h3 : ∀ n, a (n + 1) = a n * q) :
  a 1 ≠ 2 :=
by sorry

end geometric_sequence_a1_cannot_be_2_l616_61659


namespace problem_l616_61699

theorem problem (a₅ b₅ a₆ b₆ a₇ b₇ : ℤ) (S₇ S₅ T₆ T₄ : ℤ)
  (h1 : a₅ = b₅)
  (h2 : a₆ = b₆)
  (h3 : S₇ - S₅ = 4 * (T₆ - T₄)) :
  (a₇ + a₅) / (b₇ + b₅) = -1 :=
sorry

end problem_l616_61699


namespace no_integer_solution_for_system_l616_61675

theorem no_integer_solution_for_system :
  (¬ ∃ x y : ℤ, 18 * x + 27 * y = 21 ∧ 27 * x + 18 * y = 69) :=
by
  sorry

end no_integer_solution_for_system_l616_61675


namespace weight_of_replaced_student_l616_61607

theorem weight_of_replaced_student (W : ℝ) : 
  (W - 12 = 5 * 12) → W = 72 :=
by
  intro hyp
  linarith

end weight_of_replaced_student_l616_61607


namespace intersection_of_S_and_T_l616_61630

open Set

def setS : Set ℝ := { x | (x-2)*(x+3) > 0 }
def setT : Set ℝ := { x | 3 - x ≥ 0 }

theorem intersection_of_S_and_T : setS ∩ setT = { x | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_of_S_and_T_l616_61630


namespace probability_at_least_one_white_ball_l616_61667

noncomputable def total_combinations : ℕ := (Nat.choose 5 3)
noncomputable def no_white_combinations : ℕ := (Nat.choose 3 3)
noncomputable def prob_no_white_balls : ℚ := no_white_combinations / total_combinations
noncomputable def prob_at_least_one_white_ball : ℚ := 1 - prob_no_white_balls

theorem probability_at_least_one_white_ball :
  prob_at_least_one_white_ball = 9 / 10 :=
by
  have h : total_combinations = 10 := by sorry
  have h1 : no_white_combinations = 1 := by sorry
  have h2 : prob_no_white_balls = 1 / 10 := by sorry
  have h3 : prob_at_least_one_white_ball = 1 - prob_no_white_balls := by sorry
  norm_num [prob_no_white_balls, prob_at_least_one_white_ball, h, h1, h2, h3]

end probability_at_least_one_white_ball_l616_61667


namespace binomial_mod_prime_eq_floor_l616_61673

-- Define the problem's conditions and goal in Lean.
theorem binomial_mod_prime_eq_floor (n p : ℕ) (hp : Nat.Prime p) : (Nat.choose n p) % p = n / p := by
  sorry

end binomial_mod_prime_eq_floor_l616_61673


namespace rate_calculation_l616_61653

def principal : ℝ := 910
def simple_interest : ℝ := 260
def time : ℝ := 4
def rate : ℝ := 7.14

theorem rate_calculation :
  (simple_interest / (principal * time)) * 100 = rate :=
by
  sorry

end rate_calculation_l616_61653


namespace derivative_of_odd_is_even_l616_61693

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Assume f is an odd function
axiom f_odd : ∀ x, f (-x) = -f x

-- Assume g is the derivative of f
axiom g_derivative : ∀ x, g x = deriv f x

-- Goal: Prove that g is an even function, i.e., g(-x) = g(x)
theorem derivative_of_odd_is_even : ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_is_even_l616_61693


namespace coeff_z_in_third_eq_l616_61620

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end coeff_z_in_third_eq_l616_61620


namespace sequence_a113_l616_61654

theorem sequence_a113 {a : ℕ → ℝ} 
  (h1 : ∀ n, a n > 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n, (a (n+1))^2 + (a n)^2 = 2 * n * ((a (n+1))^2 - (a n)^2)) :
  a 113 = 15 :=
sorry

end sequence_a113_l616_61654


namespace repeating_decimal_to_fraction_l616_61661

theorem repeating_decimal_to_fraction : (let a := (0.28282828 : ℚ); a = 28/99) := sorry

end repeating_decimal_to_fraction_l616_61661


namespace b_is_geometric_T_sum_l616_61681

noncomputable def a (n : ℕ) : ℝ := 1/2 + (n-1) * (1/2)
noncomputable def S (n : ℕ) : ℝ := n * (1/2) + (n * (n-1) / 2) * (1/2)
noncomputable def b (n : ℕ) : ℝ := 4 ^ (a n)
noncomputable def c (n : ℕ) : ℝ := a n + b n
noncomputable def T (n : ℕ) : ℝ := (n * (n+1) / 4) + 2^(n+1) - 2

theorem b_is_geometric : ∀ n : ℕ, (n > 0) → b (n+1) / b n = 2 := by
  sorry

theorem T_sum : ∀ n : ℕ, T n = (n * (n + 1) / 4) + 2^(n + 1) - 2 := by
  sorry

end b_is_geometric_T_sum_l616_61681


namespace arithmetic_sequence_n_l616_61694

theorem arithmetic_sequence_n (a1 d an n : ℕ) (h1 : a1 = 1) (h2 : d = 3) (h3 : an = 298) (h4 : an = a1 + (n - 1) * d) : n = 100 :=
by
  sorry

end arithmetic_sequence_n_l616_61694


namespace simplify_expression_l616_61622

theorem simplify_expression(x : ℝ) : 2 * x * (4 * x^2 - 3 * x + 1) - 7 * (2 * x^2 - 3 * x + 4) = 8 * x^3 - 20 * x^2 + 23 * x - 28 :=
by
  sorry

end simplify_expression_l616_61622


namespace crayons_remaining_l616_61637

def initial_crayons : ℕ := 87
def eaten_crayons : ℕ := 7

theorem crayons_remaining : (initial_crayons - eaten_crayons) = 80 := by
  sorry

end crayons_remaining_l616_61637


namespace value_of_a_l616_61692

theorem value_of_a (a : ℝ) : (1 / (Real.log 3 / Real.log a) + 1 / (Real.log 4 / Real.log a) + 1 / (Real.log 5 / Real.log a) = 1) → a = 60 :=
by
  sorry

end value_of_a_l616_61692


namespace tan_alpha_fraction_value_l616_61649

theorem tan_alpha_fraction_value {α : Real} (h : Real.tan α = 2) : 
  (3 * Real.sin α + Real.cos α) / (5 * Real.sin α + 2 * Real.cos α) = 7 / 12 :=
by
  sorry

end tan_alpha_fraction_value_l616_61649


namespace cylinder_surface_area_l616_61689

theorem cylinder_surface_area
  (l : ℝ) (r : ℝ) (unfolded_square_side : ℝ) (base_circumference : ℝ)
  (hl : unfolded_square_side = 2 * π)
  (hl_gen : l = 2 * π)
  (hc : base_circumference = 2 * π)
  (hr : r = 1) :
  2 * π * r * (r + l) = 2 * π + 4 * π^2 :=
by
  sorry

end cylinder_surface_area_l616_61689


namespace heptagonal_prism_faces_and_vertices_l616_61627

structure HeptagonalPrism where
  heptagonal_basis : ℕ
  lateral_faces : ℕ
  basis_vertices : ℕ

noncomputable def faces (h : HeptagonalPrism) : ℕ :=
  2 + h.lateral_faces

noncomputable def vertices (h : HeptagonalPrism) : ℕ :=
  h.basis_vertices * 2

theorem heptagonal_prism_faces_and_vertices : ∀ h : HeptagonalPrism,
  (h.heptagonal_basis = 2) →
  (h.lateral_faces = 7) →
  (h.basis_vertices = 7) →
  faces h = 9 ∧ vertices h = 14 :=
by
  intros
  simp [faces, vertices]
  sorry

end heptagonal_prism_faces_and_vertices_l616_61627


namespace problem1_l616_61669

theorem problem1 : 1361 + 972 + 693 + 28 = 3000 :=
by
  sorry

end problem1_l616_61669


namespace longest_side_obtuse_triangle_l616_61650

theorem longest_side_obtuse_triangle (a b c : ℝ) (h₀ : a = 2) (h₁ : b = 4) 
  (h₂ : a^2 + b^2 < c^2) : 
  2 * Real.sqrt 5 < c ∧ c < 6 :=
by 
  sorry

end longest_side_obtuse_triangle_l616_61650


namespace car_wash_cost_l616_61605

-- Definitions based on the conditions
def washes_per_bottle : ℕ := 4
def bottle_cost : ℕ := 4   -- Assuming cost is recorded in dollars
def total_weeks : ℕ := 20

-- Stating the problem
theorem car_wash_cost : (total_weeks / washes_per_bottle) * bottle_cost = 20 := 
by
  -- Placeholder for the proof
  sorry

end car_wash_cost_l616_61605


namespace compare_powers_l616_61631

theorem compare_powers (a b c : ℕ) (h1 : a = 81^31) (h2 : b = 27^41) (h3 : c = 9^61) : a > b ∧ b > c := by
  sorry

end compare_powers_l616_61631


namespace sister_ages_l616_61685

theorem sister_ages (x y : ℕ) (h1 : x = y + 4) (h2 : x^3 - y^3 = 988) : y = 7 ∧ x = 11 :=
by
  sorry

end sister_ages_l616_61685


namespace sum_is_square_l616_61625

theorem sum_is_square (a b c : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : Nat.gcd a b = 1) (h5 : Nat.gcd b c = 1) (h6 : Nat.gcd c a = 1) 
  (h7 : (1:ℚ)/a + (1:ℚ)/b = (1:ℚ)/c) : ∃ k : ℕ, a + b = k ^ 2 := 
by 
  sorry

end sum_is_square_l616_61625


namespace two_digit_number_l616_61608

theorem two_digit_number (x : ℕ) (h1 : x ≥ 10 ∧ x < 100)
  (h2 : ∃ k : ℤ, 3 * x - 4 = 10 * k)
  (h3 : 60 < 4 * x - 15 ∧ 4 * x - 15 < 100) :
  x = 28 :=
by
  sorry

end two_digit_number_l616_61608


namespace symmetric_line_equation_l616_61628

theorem symmetric_line_equation (l : ℝ × ℝ → Prop)
  (h1 : ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0)
  (h2 : ∀ p : ℝ × ℝ, l p ↔ p = (0, 2) ∨ p = ⟨-3, 2⟩) :
  ∀ x y, l (x, y) ↔ 3 * x + y - 2 = 0 :=
by
  sorry

end symmetric_line_equation_l616_61628


namespace triangular_region_area_l616_61676

theorem triangular_region_area : 
  ∀ (x y : ℝ),  (3 * x + 4 * y = 12) →
  (0 ≤ x ∧ 0 ≤ y) →
  ∃ (A : ℝ), A = 6 := 
by 
  sorry

end triangular_region_area_l616_61676


namespace range_of_a_l616_61663

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a + 3) * Real.exp (a * x)

theorem range_of_a (a : ℝ) : 
  (∀ x y, x ≤ y → f a x ≤ f a y) ∨ (∀ x y, x ≤ y → f a x ≥ f a y) → 
  a ∈ Set.Ico (-2 : ℝ) 0 :=
sorry

end range_of_a_l616_61663


namespace digit_for_divisibility_by_5_l616_61612

theorem digit_for_divisibility_by_5 (B : ℕ) (B_digit_condition : B < 10) :
  (∃ k : ℕ, 6470 + B = 5 * k) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end digit_for_divisibility_by_5_l616_61612


namespace Daniella_savings_l616_61665

def initial_savings_of_Daniella (D : ℤ) := D
def initial_savings_of_Ariella (D : ℤ) := D + 200
def interest_rate : ℚ := 0.10
def time_years : ℚ := 2
def total_amount_after_two_years (initial_amount : ℤ) : ℚ :=
  initial_amount + initial_amount * interest_rate * time_years
def final_amount_of_Ariella : ℚ := 720

theorem Daniella_savings :
  ∃ D : ℤ, total_amount_after_two_years (initial_savings_of_Ariella D) = final_amount_of_Ariella ∧ initial_savings_of_Daniella D = 400 :=
by
  sorry

end Daniella_savings_l616_61665


namespace find_x_l616_61632

theorem find_x
  (x : ℤ)
  (h1 : 71 * x % 9 = 8) :
  x = 1 :=
sorry

end find_x_l616_61632


namespace find_x_plus_y_l616_61611

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (a b c p : V) (x y : ℝ)

-- Conditions: Definitions as the given problem requires
-- Basis definitions
def basis1 := [a, b, c]
def basis2 := [a + b, a - b, c]

-- Conditions on p
def condition1 : p = 3 • a + b + c := sorry
def condition2 : p = x • (a + b) + y • (a - b) + c := sorry

-- The proof statement
theorem find_x_plus_y (h1 : p = 3 • a + b + c) (h2 : p = x • (a + b) + y • (a - b) + c) :
  x + y = 3 :=
sorry

end find_x_plus_y_l616_61611


namespace total_area_of_combined_figure_l616_61629

noncomputable def combined_area (A_triangle : ℕ) (b : ℕ) : ℕ :=
  let h := (2 * A_triangle) / b
  let A_square := b * b
  A_square + A_triangle

theorem total_area_of_combined_figure :
  combined_area 720 40 = 2320 := by
  sorry

end total_area_of_combined_figure_l616_61629


namespace bill_has_correct_final_amount_l616_61600

def initial_amount : ℕ := 42
def pizza_cost : ℕ := 11
def pizzas_bought : ℕ := 3
def bill_initial_amount : ℕ := 30
def amount_spent := pizzas_bought * pizza_cost
def frank_remaining_amount := initial_amount - amount_spent
def bill_final_amount := bill_initial_amount + frank_remaining_amount

theorem bill_has_correct_final_amount : bill_final_amount = 39 := by
  sorry

end bill_has_correct_final_amount_l616_61600


namespace age_sum_l616_61633

theorem age_sum (P Q : ℕ) (h1 : P - 12 = (1 / 2 : ℚ) * (Q - 12)) (h2 : (P : ℚ) / Q = (3 / 4 : ℚ)) : P + Q = 42 :=
sorry

end age_sum_l616_61633


namespace simplify_power_of_product_l616_61651

theorem simplify_power_of_product (x : ℝ) : (5 * x^2)^4 = 625 * x^8 :=
by
  sorry

end simplify_power_of_product_l616_61651


namespace fuse_length_must_be_80_l616_61690

-- Define the basic conditions
def distanceToSafeArea : ℕ := 400
def personSpeed : ℕ := 5
def fuseBurnSpeed : ℕ := 1

-- Calculate the time required to reach the safe area
def timeToSafeArea (distance speed : ℕ) : ℕ := distance / speed

-- Calculate the minimum length of the fuse based on the time to reach the safe area
def minFuseLength (time burnSpeed : ℕ) : ℕ := time * burnSpeed

-- The main problem statement: The fuse must be at least 80 meters long.
theorem fuse_length_must_be_80:
  minFuseLength (timeToSafeArea distanceToSafeArea personSpeed) fuseBurnSpeed = 80 :=
by
  sorry

end fuse_length_must_be_80_l616_61690


namespace min_value_x_plus_y_l616_61678

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : y + 9 * x = x * y) :
  x + y ≥ 16 :=
by
  sorry

end min_value_x_plus_y_l616_61678


namespace vector_c_solution_l616_61648

def vector_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k * w.1, k * w.2)

def vector_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_c_solution
  (a b c : ℝ × ℝ)
  (h1 : a = (1, 2))
  (h2 : b = (2, -3))
  (h3 : vector_parallel (c.1 + 1, c.2 + 2) b)
  (h4 : vector_perpendicular c (3, -1)) :
  c = (-7/9, -7/3) :=
sorry

end vector_c_solution_l616_61648


namespace expand_and_simplify_l616_61687

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5 * x - 21 := 
by 
  sorry

end expand_and_simplify_l616_61687


namespace friend_initial_marbles_l616_61645

theorem friend_initial_marbles (total_games : ℕ) (bids_per_game : ℕ) (games_lost : ℕ) (final_marbles : ℕ) 
  (h_games_eq : total_games = 9) (h_bids_eq : bids_per_game = 10) 
  (h_lost_eq : games_lost = 1) (h_final_eq : final_marbles = 90) : 
  ∃ initial_marbles : ℕ, initial_marbles = 20 := by
  sorry

end friend_initial_marbles_l616_61645


namespace greatest_integer_property_l616_61646

theorem greatest_integer_property :
  ∃ n : ℤ, n < 1000 ∧ (∃ m : ℤ, 4 * n^3 - 3 * n = (2 * m - 1) * (2 * m + 1)) ∧ 
  (∀ k : ℤ, k < 1000 ∧ (∃ m : ℤ, 4 * k^3 - 3 * k = (2 * m - 1) * (2 * m + 1)) → k ≤ n) := by
  -- skipped the proof with sorry
  sorry

end greatest_integer_property_l616_61646


namespace inv_sum_mod_l616_61672

theorem inv_sum_mod (x y : ℤ) (h1 : 5 * x ≡ 1 [ZMOD 23]) (h2 : 25 * y ≡ 1 [ZMOD 23]) : (x + y) ≡ 3 [ZMOD 23] := by
  sorry

end inv_sum_mod_l616_61672


namespace shortest_total_distance_piglet_by_noon_l616_61674

-- Define the distances
def distance_fs : ℕ := 1300  -- Distance through the forest (Piglet to Winnie-the-Pooh)
def distance_pr : ℕ := 600   -- Distance (Piglet to Rabbit)
def distance_rw : ℕ := 500   -- Distance (Rabbit to Winnie-the-Pooh)

-- Define the total distance via Rabbit and via forest
def total_distance_rabbit_path : ℕ := distance_pr + distance_rw + distance_rw
def total_distance_forest_path : ℕ := distance_fs + distance_rw

-- Prove that shortest distance Piglet covers by noon
theorem shortest_total_distance_piglet_by_noon : 
  min (total_distance_forest_path) (total_distance_rabbit_path) = 1600 := by
  sorry

end shortest_total_distance_piglet_by_noon_l616_61674


namespace measure_one_kg_grain_l616_61686

/-- Proving the possibility of measuring exactly 1 kg of grain
    using a balance scale, one 3 kg weight, and three weighings. -/
theorem measure_one_kg_grain :
  ∃ (weighings : ℕ) (balance_scale : ℕ → ℤ) (weight_3kg : ℤ → Prop),
  weighings = 3 ∧
  (∀ w, weight_3kg w ↔ w = 3) ∧
  ∀ n m, balance_scale n = 0 ∧ balance_scale m = 1 → true :=
sorry

end measure_one_kg_grain_l616_61686


namespace sequence_integers_l616_61658

theorem sequence_integers (a : ℕ → ℤ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 1) 
  (h₃ : ∀ n, n ≥ 3 → a n = (a (n-1)) ^ 2 + 2 / a (n-2)) : 
  ∀ n, ∃ k : ℤ, a n = k := 
by 
  sorry

end sequence_integers_l616_61658


namespace parallelogram_area_l616_61639

open Real

def line1 (p : ℝ × ℝ) : Prop := p.2 = 2
def line2 (p : ℝ × ℝ) : Prop := p.2 = -2
def line3 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 - 10 = 0
def line4 (p : ℝ × ℝ) : Prop := 4 * p.1 + 7 * p.2 + 20 = 0

theorem parallelogram_area :
  ∃ D : ℝ, D = 30 ∧
  (∀ p : ℝ × ℝ, line1 p ∨ line2 p ∨ line3 p ∨ line4 p) :=
sorry

end parallelogram_area_l616_61639


namespace raft_capacity_l616_61618

theorem raft_capacity (total_without_life_jackets : ℕ) (reduction_with_life_jackets : ℕ)
  (people_needing_life_jackets : ℕ) (total_capacity_with_life_jackets : ℕ)
  (no_life_jackets_capacity : total_without_life_jackets = 21)
  (life_jackets_reduction : reduction_with_life_jackets = 7)
  (life_jackets_needed : people_needing_life_jackets = 8) :
  total_capacity_with_life_jackets = 14 :=
by
  -- Proof should be here
  sorry

end raft_capacity_l616_61618


namespace x_y_sum_cube_proof_l616_61655

noncomputable def x_y_sum_cube (x y : ℝ) : ℝ := x^3 + y^3

theorem x_y_sum_cube_proof (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h_eq : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x_y_sum_cube x y = 307 :=
sorry

end x_y_sum_cube_proof_l616_61655


namespace exist_2022_good_numbers_with_good_sum_l616_61695

def is_good (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (k + 1)

theorem exist_2022_good_numbers_with_good_sum :
  ∃ (a : Fin 2022 → ℕ), (∀ i j : Fin 2022, i ≠ j → a i ≠ a j) ∧ (∀ i : Fin 2022, is_good (a i)) ∧ is_good (Finset.univ.sum a) :=
sorry

end exist_2022_good_numbers_with_good_sum_l616_61695


namespace katie_baked_5_cookies_l616_61623

theorem katie_baked_5_cookies (cupcakes cookies sold left : ℕ) 
  (h1 : cupcakes = 7) 
  (h2 : sold = 4) 
  (h3 : left = 8) 
  (h4 : cupcakes + cookies = sold + left) : 
  cookies = 5 :=
by sorry

end katie_baked_5_cookies_l616_61623


namespace problem_equivalence_l616_61697

variable {x y z w : ℝ}

theorem problem_equivalence (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end problem_equivalence_l616_61697


namespace retail_price_increase_l616_61613

theorem retail_price_increase (R W : ℝ) (h1 : 0.80 * R = 1.44000000000000014 * W)
  : ((R - W) / W) * 100 = 80 :=
by 
  sorry

end retail_price_increase_l616_61613


namespace partI_partII_l616_61664

theorem partI (m : ℝ) (h1 : ∀ t : ℝ, |t + 3| - |t - 2| ≤ 6 * m - m^2) :
  1 ≤ m ∧ m ≤ 5 :=
sorry

noncomputable def lambda : ℝ := 5

theorem partII (x y z : ℝ) (h2 : 3 * x + 4 * y + 5 * z = lambda) :
  x^2 + y^2 + z^2 ≥ 1/2 :=
sorry

end partI_partII_l616_61664


namespace rearrangement_count_correct_l616_61601

def original_number := "1234567890"

def is_valid_rearrangement (n : String) : Prop :=
  n.length = 10 ∧ n.front ≠ '0'
  
def count_rearrangements (n : String) : ℕ :=
  if n = original_number 
  then 232
  else 0

theorem rearrangement_count_correct :
  count_rearrangements original_number = 232 :=
sorry


end rearrangement_count_correct_l616_61601


namespace number_of_solutions_sine_exponential_l616_61636

theorem number_of_solutions_sine_exponential :
  let f := λ x => Real.sin x
  let g := λ x => (1 / 3) ^ x
  ∃ n, n = 150 ∧ ∀ k ∈ Set.Icc (0 : ℝ) (150 * Real.pi), f k = g k → (k : ℝ) ∈ {n : ℝ | n ∈ Set.Icc (0 : ℝ) (150 * Real.pi)} :=
sorry

end number_of_solutions_sine_exponential_l616_61636


namespace combined_time_to_finish_cereal_l616_61668

theorem combined_time_to_finish_cereal : 
  let rate_fat := 1 / 15
  let rate_thin := 1 / 45
  let combined_rate := rate_fat + rate_thin
  let time_needed := 4 / combined_rate
  time_needed = 45 := 
by 
  sorry

end combined_time_to_finish_cereal_l616_61668


namespace ticket_costs_l616_61604

theorem ticket_costs (ticket_price : ℕ) (number_of_tickets : ℕ) : ticket_price = 44 ∧ number_of_tickets = 7 → ticket_price * number_of_tickets = 308 :=
by
  intros h
  cases h
  sorry

end ticket_costs_l616_61604


namespace area_of_fourth_rectangle_l616_61696

theorem area_of_fourth_rectangle (a b c d : ℕ) (x y z w : ℕ)
  (h1 : a = x * y)
  (h2 : b = x * w)
  (h3 : c = z * w)
  (h4 : d = y * w)
  (h5 : (x + z) * (y + w) = a + b + c + d) : d = 15 :=
sorry

end area_of_fourth_rectangle_l616_61696
