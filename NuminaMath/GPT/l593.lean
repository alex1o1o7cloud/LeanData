import Mathlib

namespace isosceles_triangle_perimeter_l593_59372

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

theorem isosceles_triangle_perimeter 
  (a b c : ℝ) 
  (h_iso : is_isosceles a b c) 
  (h1 : a = 2 ∨ a = 4) 
  (h2 : b = 2 ∨ b = 4) 
  (h3 : c = 2 ∨ c = 4) :
  a + b + c = 10 :=
  sorry

end isosceles_triangle_perimeter_l593_59372


namespace bob_initial_cats_l593_59302

theorem bob_initial_cats (B : ℕ) (h : 21 - 4 = B + 14) : B = 3 := 
by
  -- Placeholder for the proof
  sorry

end bob_initial_cats_l593_59302


namespace find_original_number_l593_59307

def digitsGPA (A B C : ℕ) : Prop := B^2 = A * C
def digitsAPA (X Y Z : ℕ) : Prop := 2 * Y = X + Z

theorem find_original_number (A B C X Y Z : ℕ) :
  100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C ≤ 999 ∧
  digitsGPA A B C ∧
  100 * X + 10 * Y + Z = (100 * A + 10 * B + C) - 200 ∧
  digitsAPA X Y Z →
  (100 * A + 10 * B + C) = 842 :=
sorry

end find_original_number_l593_59307


namespace remainder_101_pow_47_mod_100_l593_59348

theorem remainder_101_pow_47_mod_100 : (101 ^ 47) % 100 = 1 := by 
  sorry

end remainder_101_pow_47_mod_100_l593_59348


namespace max_product_xy_l593_59381

theorem max_product_xy (x y : ℕ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : 7 * x + 4 * y = 150) : xy = 200 :=
by
  sorry

end max_product_xy_l593_59381


namespace range_of_a_l593_59375

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x + x^2 else x - x^2

theorem range_of_a (a : ℝ) : (∀ x, -1/2 ≤ x ∧ x ≤ 1/2 → f (x^2 + 1) > f (a * x)) ↔ -5/2 < a ∧ a < 5/2 := 
sorry

end range_of_a_l593_59375


namespace raft_travel_time_l593_59396

-- Define the problem conditions:
def steamboat_time (distance : ℕ) := 1 -- in hours
def motorboat_time (distance : ℕ) : ℚ := 3 / 4 -- in hours
def speed_ratio := 2 -- motorboat speed is twice the speed of steamboat

-- Define the time for the raft to travel the distance:
def raft_time (distance : ℕ) (current_speed : ℚ) := distance / current_speed

-- Given the conditions, prove that the raft time equals to 90 minutes
theorem raft_travel_time (distance : ℕ) (rafter_speed : ℚ) (current_speed : ℚ) :
  steamboat_time distance = 1 ∧ motorboat_time distance = 3 / 4 ∧ rafter_speed = current_speed →
  rafter_speed = current_speed ∧ raft_time distance current_speed = 3 / 2 → -- hours
  raft_time distance current_speed * 60 = 90 := -- convert hours to minutes
by
  intros h1 h2
  sorry

end raft_travel_time_l593_59396


namespace quadratic_form_m_neg3_l593_59358

theorem quadratic_form_m_neg3
  (m : ℝ)
  (h_exp : m^2 - 7 = 2)
  (h_coef : m ≠ 3) :
  m = -3 := by
  sorry

end quadratic_form_m_neg3_l593_59358


namespace leah_ride_time_l593_59352

theorem leah_ride_time (x y : ℝ) (h1 : 90 * x = y) (h2 : 30 * (x + 2 * x) = y)
: ∃ t : ℝ, t = 67.5 :=
by
  -- Define 50% increase in length
  let y' := 1.5 * y
  -- Define escalator speed without Leah walking
  let k := 2 * x
  -- Calculate the time taken
  let t := y' / k
  -- Prove that this time is 67.5 seconds
  have ht : t = 67.5 := sorry
  exact ⟨t, ht⟩

end leah_ride_time_l593_59352


namespace initial_birds_in_cage_l593_59304

-- Define a theorem to prove the initial number of birds in the cage
theorem initial_birds_in_cage (B : ℕ) 
  (H1 : 2 / 15 * B = 8) : B = 60 := 
by sorry

end initial_birds_in_cage_l593_59304


namespace total_guppies_l593_59362

-- Define conditions
def Haylee_guppies : ℕ := 3 * 12
def Jose_guppies : ℕ := Haylee_guppies / 2
def Charliz_guppies : ℕ := Jose_guppies / 3
def Nicolai_guppies : ℕ := Charliz_guppies * 4

-- Theorem statement: total number of guppies is 84
theorem total_guppies : Haylee_guppies + Jose_guppies + Charliz_guppies + Nicolai_guppies = 84 := 
by 
  sorry

end total_guppies_l593_59362


namespace max_cookies_andy_could_have_eaten_l593_59316

theorem max_cookies_andy_could_have_eaten (x k : ℕ) (hk : k > 0) 
  (h_total : x + k * x + 2 * x = 36) : x ≤ 9 :=
by
  -- Using the conditions to construct the proof (which is not required based on the instructions)
  sorry

end max_cookies_andy_could_have_eaten_l593_59316


namespace train_speed_clicks_l593_59399

theorem train_speed_clicks (x : ℝ) (v : ℝ) (t : ℝ) 
  (h1 : v = x * 5280 / 60) 
  (h2 : t = 25) 
  (h3 : 70 * t = v * 25) : v = 70 := sorry

end train_speed_clicks_l593_59399


namespace minimize_fees_at_5_l593_59365

noncomputable def minimize_costs (x : ℝ) (y1 y2 : ℝ) : Prop :=
  let k1 := 40
  let k2 := 8 / 5
  y1 = k1 / x ∧ y2 = k2 * x ∧ (∀ x, y1 + y2 ≥ 16 ∧ (y1 + y2 = 16 ↔ x = 5))

theorem minimize_fees_at_5 :
  minimize_costs 5 4 16 :=
sorry

end minimize_fees_at_5_l593_59365


namespace sara_initial_peaches_l593_59320

variable (p : ℕ)

def initial_peaches (picked_peaches total_peaches : ℕ) :=
  total_peaches - picked_peaches

theorem sara_initial_peaches :
  initial_peaches 37 61 = 24 :=
by
  -- This follows directly from the definition of initial_peaches
  sorry

end sara_initial_peaches_l593_59320


namespace miles_monday_calculation_l593_59384

-- Define the constants
def flat_fee : ℕ := 150
def cost_per_mile : ℝ := 0.50
def miles_thursday : ℕ := 744
def total_cost : ℕ := 832

-- Define the equation to be proved
theorem miles_monday_calculation :
  ∃ M : ℕ, (flat_fee + (M : ℝ) * cost_per_mile + (miles_thursday : ℝ) * cost_per_mile = total_cost) ∧ M = 620 :=
by
  sorry

end miles_monday_calculation_l593_59384


namespace rectangular_plot_area_l593_59380

theorem rectangular_plot_area (breadth length : ℕ) (h1 : breadth = 14) (h2 : length = 3 * breadth) : (length * breadth) = 588 := 
by 
  -- imports, noncomputable keyword, and placeholder proof for compilation
  sorry

end rectangular_plot_area_l593_59380


namespace integer_pairs_satisfy_equation_l593_59364

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end integer_pairs_satisfy_equation_l593_59364


namespace sum_of_digits_next_exact_multiple_l593_59313

noncomputable def Michael_next_age_sum_of_digits (L M T n : ℕ) : ℕ :=
  let next_age := M + n
  ((next_age / 10) % 10) + (next_age % 10)

theorem sum_of_digits_next_exact_multiple :
  ∀ (L M T n : ℕ),
    T = 2 →
    M = L + 4 →
    (∀ k : ℕ, k < 8 → ∃ m : ℕ, L = m * T + k * T) →
    (∃ n, (M + n) % (T + n) = 0) →
    Michael_next_age_sum_of_digits L M T n = 9 :=
by
  intros
  sorry

end sum_of_digits_next_exact_multiple_l593_59313


namespace math_proof_problem_l593_59382

variable (a d e : ℝ)

theorem math_proof_problem (h1 : a < 0) (h2 : a < d) (h3 : d < e) :
  (a * d < a * e) ∧ (a + d < d + e) ∧ (e / a < 1) :=
by {
  sorry
}

end math_proof_problem_l593_59382


namespace breadth_increase_25_percent_l593_59327

variable (L B : ℝ) 

-- Conditions
def original_area := L * B
def increased_length := 1.10 * L
def increased_area := 1.375 * (original_area L B)

-- The breadth increase percentage (to be proven as 25)
def percentage_increase_breadth (p : ℝ) := 
  increased_area L B = increased_length L * (B * (1 + p/100))

-- The statement to be proven
theorem breadth_increase_25_percent : 
  percentage_increase_breadth L B 25 := 
sorry

end breadth_increase_25_percent_l593_59327


namespace miki_pear_juice_l593_59305

def total_pears : ℕ := 18
def total_oranges : ℕ := 10
def pear_juice_per_pear : ℚ := 10 / 2
def orange_juice_per_orange : ℚ := 12 / 3
def max_blend_volume : ℚ := 44

theorem miki_pear_juice : (total_oranges * orange_juice_per_orange = 40) ∧ (max_blend_volume - 40 = 4) → 
  ∃ p : ℚ, p * pear_juice_per_pear = 4 ∧ p = 0 :=
by
  sorry

end miki_pear_juice_l593_59305


namespace jack_walked_time_l593_59312

def jack_distance : ℝ := 9
def jack_rate : ℝ := 7.2
def jack_time : ℝ := 1.25

theorem jack_walked_time : jack_time = jack_distance / jack_rate := by
  sorry

end jack_walked_time_l593_59312


namespace find_kg_of_mangoes_l593_59363

-- Define the conditions
def cost_of_grapes : ℕ := 8 * 70
def total_amount_paid : ℕ := 965
def cost_of_mangoes (m : ℕ) : ℕ := 45 * m

-- Formalize the proof problem
theorem find_kg_of_mangoes (m : ℕ) :
  cost_of_grapes + cost_of_mangoes m = total_amount_paid → m = 9 :=
by
  intros h
  sorry

end find_kg_of_mangoes_l593_59363


namespace jill_sales_goal_l593_59336

def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def boxes_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer
def boxes_left : ℕ := 75
def sales_goal : ℕ := boxes_sold + boxes_left

theorem jill_sales_goal : sales_goal = 150 := by
  sorry

end jill_sales_goal_l593_59336


namespace am_gm_inequality_l593_59340

open Real

theorem am_gm_inequality (
    a b c d e f : ℝ
) (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_nonneg_c : 0 ≤ c)
  (h_nonneg_d : 0 ≤ d)
  (h_nonneg_e : 0 ≤ e)
  (h_nonneg_f : 0 ≤ f)
  (h_cond_ab : a + b ≤ e)
  (h_cond_cd : c + d ≤ f) :
  sqrt (a * c) + sqrt (b * d) ≤ sqrt (e * f) := 
  by sorry

end am_gm_inequality_l593_59340


namespace total_students_l593_59322

variables (B G : ℕ)
variables (two_thirds_boys : 2 * B = 3 * 400)
variables (three_fourths_girls : 3 * G = 4 * 150)
variables (total_participants : B + G = 800)

theorem total_students (B G : ℕ)
  (two_thirds_boys : 2 * B = 3 * 400)
  (three_fourths_girls : 3 * G = 4 * 150)
  (total_participants : B + G = 800) :
  B + G = 800 :=
by
  sorry

end total_students_l593_59322


namespace Mickey_horses_per_week_l593_59306

-- Definitions based on the conditions
def days_in_week : Nat := 7
def Minnie_mounts_per_day : Nat := days_in_week + 3 
def Mickey_mounts_per_day : Nat := 2 * Minnie_mounts_per_day - 6
def Mickey_mounts_per_week : Nat := Mickey_mounts_per_day * days_in_week

-- Theorem statement
theorem Mickey_horses_per_week : Mickey_mounts_per_week = 98 :=
by
  sorry

end Mickey_horses_per_week_l593_59306


namespace function_is_odd_and_increasing_l593_59373

theorem function_is_odd_and_increasing :
  (∀ x : ℝ, (x^(1/3) : ℝ) = -( (-x)^(1/3) : ℝ)) ∧ (∀ x y : ℝ, x < y → (x^(1/3) : ℝ) < (y^(1/3) : ℝ)) :=
by
  sorry

end function_is_odd_and_increasing_l593_59373


namespace at_least_one_l593_59318

axiom P : Prop  -- person A is an outstanding student
axiom Q : Prop  -- person B is an outstanding student

theorem at_least_one (H : ¬(¬P ∧ ¬Q)) : P ∨ Q :=
sorry

end at_least_one_l593_59318


namespace M_inter_N_l593_59389

namespace ProofProblem

def M : Set ℝ := { x | 3 * x - x^2 > 0 }
def N : Set ℝ := { x | x^2 - 4 * x + 3 > 0 }

theorem M_inter_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
sorry

end ProofProblem

end M_inter_N_l593_59389


namespace range_of_m_l593_59360

def f (x : ℝ) : ℝ := x^3 + x

theorem range_of_m
  (m : ℝ)
  (hθ : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2)
  (h : ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ Real.pi / 2 → f (m * Real.sin θ) + f (1 - m) > 0) :
  m < 1 :=
by
  sorry

end range_of_m_l593_59360


namespace polynomial_satisfies_condition_l593_59376

open Polynomial

noncomputable def polynomial_f : Polynomial ℝ := 6 * X ^ 2 + 5 * X + 1
noncomputable def polynomial_g : Polynomial ℝ := 3 * X ^ 2 + 7 * X + 2

def sum_of_squares (p : Polynomial ℝ) : ℝ :=
  p.coeff 0 ^ 2 + p.coeff 1 ^ 2 + p.coeff 2 ^ 2 + p.coeff 3 ^ 2 + -- ...
  sorry -- Extend as necessary for the degree of the polynomial

theorem polynomial_satisfies_condition :
  (∀ n : ℕ, sum_of_squares (polynomial_f ^ n) = sum_of_squares (polynomial_g ^ n)) :=
by
  sorry

end polynomial_satisfies_condition_l593_59376


namespace fractional_eq_no_real_roots_l593_59397

theorem fractional_eq_no_real_roots (k : ℝ) :
  (∀ x : ℝ, (x - 1) ≠ 0 → (k / (x - 1) + 3 ≠ x / (1 - x))) → k = -1 :=
by
  sorry

end fractional_eq_no_real_roots_l593_59397


namespace sum_of_first_five_terms_sequence_l593_59335

-- Definitions derived from conditions
def seventh_term : ℤ := 4
def eighth_term : ℤ := 10
def ninth_term : ℤ := 16

-- The main theorem statement
theorem sum_of_first_five_terms_sequence : 
  ∃ (a d : ℤ), 
    a + 6 * d = seventh_term ∧
    a + 7 * d = eighth_term ∧
    a + 8 * d = ninth_term ∧
    (a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) = -100) :=
by
  sorry

end sum_of_first_five_terms_sequence_l593_59335


namespace total_distance_covered_l593_59359

variable (h : ℝ) (initial_height : ℝ := h) (bounce_ratio : ℝ := 0.8)

theorem total_distance_covered :
  initial_height + 2 * initial_height * bounce_ratio / (1 - bounce_ratio) = 13 * h :=
by 
  -- Proof omitted for now
  sorry

end total_distance_covered_l593_59359


namespace quadratic_function_passing_through_origin_l593_59391

-- Define the quadratic function y
def quadratic_function (m x : ℝ) : ℝ :=
  (m - 2) * x^2 - 4 * x + m^2 + 2 * m - 8

-- State the problem as a theorem
theorem quadratic_function_passing_through_origin (m : ℝ) (h: quadratic_function m 0 = 0) : m = -4 :=
by
  -- Since we only need the statement, we put sorry here
  sorry

end quadratic_function_passing_through_origin_l593_59391


namespace petya_vasya_meet_at_lamp_64_l593_59317

-- Definitions of positions of Petya and Vasya
def Petya_position (x : ℕ) : ℕ := x - 21 -- Petya starts from the 1st lamp and is at the 22nd lamp
def Vasya_position (x : ℕ) : ℕ := 88 - x -- Vasya starts from the 100th lamp and is at the 88th lamp

-- Condition that both lanes add up to 64
theorem petya_vasya_meet_at_lamp_64 : ∀ x y : ℕ, 
    Petya_position x = Vasya_position y -> x = 64 :=
by
  intro x y
  rw [Petya_position, Vasya_position]
  sorry

end petya_vasya_meet_at_lamp_64_l593_59317


namespace part1_A_intersect_B_l593_59319

def setA : Set ℝ := { x | x ^ 2 - 2 * x - 3 ≤ 0 }
def setB (m : ℝ) : Set ℝ := { x | (x - (m - 1)) * (x - (m + 1)) > 0 }

theorem part1_A_intersect_B (m : ℝ) (h : m = 0) : 
  setA ∩ setB m = { x | 1 < x ∧ x ≤ 3 } :=
sorry

end part1_A_intersect_B_l593_59319


namespace num_cells_after_10_moves_l593_59390

def is_adjacent (p1 p2 : ℕ × ℕ) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p1.2 + 1 = p2.2)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p1.1 + 1 = p2.1))

def num_reachable_cells (n m k : ℕ) (start : ℕ × ℕ) : ℕ :=
  sorry -- Calculation of the number of reachable cells after k moves

theorem num_cells_after_10_moves :
  let board_size := 21
  let start := (11, 11)
  let moves := 10
  num_reachable_cells board_size board_size moves start = 121 :=
sorry

end num_cells_after_10_moves_l593_59390


namespace tangent_lines_count_l593_59328

noncomputable def number_of_tangent_lines (r1 r2 : ℝ) (k : ℕ) : ℕ :=
if r1 = 2 ∧ r2 = 3 then 5 else 0

theorem tangent_lines_count: 
∃ k : ℕ, number_of_tangent_lines 2 3 k = 5 :=
by sorry

end tangent_lines_count_l593_59328


namespace arc_length_parametric_l593_59383

open Real Interval

noncomputable def arc_length (f_x f_y : ℝ → ℝ) (t1 t2 : ℝ) :=
  ∫ t in Set.Icc t1 t2, sqrt ((deriv f_x t)^2 + (deriv f_y t)^2)

theorem arc_length_parametric :
  arc_length
    (λ t => 2.5 * (t - sin t))
    (λ t => 2.5 * (1 - cos t))
    (π / 2) π = 5 * sqrt 2 :=
by
  sorry

end arc_length_parametric_l593_59383


namespace solutions_to_x_squared_eq_x_l593_59395

theorem solutions_to_x_squared_eq_x (x : ℝ) : x^2 = x ↔ x = 0 ∨ x = 1 := 
sorry

end solutions_to_x_squared_eq_x_l593_59395


namespace shortest_distance_to_circle_l593_59338

variable (A O T : Type)
variable (r d : ℝ)
variable [MetricSpace A]
variable [MetricSpace O]
variable [MetricSpace T]

open Real

theorem shortest_distance_to_circle (h : d = (4 / 3) * r) : 
  OA = (5 / 3) * r → shortest_dist = (2 / 3) * r :=
by
  sorry

end shortest_distance_to_circle_l593_59338


namespace min_function_value_in_domain_l593_59369

theorem min_function_value_in_domain :
  ∃ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) ∧ (∀ (x y : ℝ), (1 / 3 ≤ x ∧ x ≤ 3 / 5) ∧ (1 / 4 ≤ y ∧ y ≤ 1 / 2) → (xy / (x^2 + y^2)) ≥ (60 / 169)) :=
sorry

end min_function_value_in_domain_l593_59369


namespace triangle_inequality_right_triangle_l593_59350

theorem triangle_inequality_right_triangle
  (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a + b) / Real.sqrt 2 ≤ c :=
by sorry

end triangle_inequality_right_triangle_l593_59350


namespace other_leg_length_l593_59393

theorem other_leg_length (a b c : ℕ) (ha : a = 24) (hc : c = 25) 
  (h : a * a + b * b = c * c) : b = 7 := 
by 
  sorry

end other_leg_length_l593_59393


namespace space_is_volume_stuff_is_capacity_film_is_surface_area_l593_59326

-- Let's define the properties based on the conditions
def size_of_space (box : Type) : Type := 
  sorry -- This will be volume later

def stuff_can_hold (box : Type) : Type :=
  sorry -- This will be capacity later

def film_needed_to_cover (box : Type) : Type :=
  sorry -- This will be surface area later

-- Now prove the correspondences
theorem space_is_volume (box : Type) :
  size_of_space box = volume := 
by 
  sorry

theorem stuff_is_capacity (box : Type) :
  stuff_can_hold box = capacity := 
by 
  sorry

theorem film_is_surface_area (box : Type) :
  film_needed_to_cover box = surface_area := 
by 
  sorry

end space_is_volume_stuff_is_capacity_film_is_surface_area_l593_59326


namespace determine_b_l593_59324

theorem determine_b (A B C : ℝ) (a b c : ℝ)
  (angle_C_eq_4A : C = 4 * A)
  (a_eq_30 : a = 30)
  (c_eq_48 : c = 48)
  (law_of_sines : ∀ x y, x / Real.sin A = y / Real.sin (4 * A))
  (cos_eq_solution : 4 * Real.cos A ^ 3 - 4 * Real.cos A = 8 / 5) :
  ∃ b : ℝ, b = 30 * (5 - 20 * (1 - Real.cos A ^ 2) + 16 * (1 - Real.cos A ^ 2) ^ 2) :=
by 
  sorry

end determine_b_l593_59324


namespace max_crosses_in_grid_l593_59341

theorem max_crosses_in_grid : ∀ (n : ℕ), n = 16 → (∃ X : ℕ, X = 30 ∧
  ∀ (i j : ℕ), i < n → j < n → 
    (∀ k, k < n → (i ≠ k → X ≠ k)) ∧ 
    (∀ l, l < n → (j ≠ l → X ≠ l))) :=
by
  sorry

end max_crosses_in_grid_l593_59341


namespace division_remainder_example_l593_59311

theorem division_remainder_example :
  ∃ n, n = 20 * 10 + 10 ∧ n = 210 :=
by
  sorry

end division_remainder_example_l593_59311


namespace cistern_fill_time_l593_59353

theorem cistern_fill_time (F : ℝ) (E : ℝ) (net_rate : ℝ) (time : ℝ)
  (h_F : F = 1 / 4)
  (h_E : E = 1 / 8)
  (h_net : net_rate = F - E)
  (h_time : time = 1 / net_rate) :
  time = 8 := 
sorry

end cistern_fill_time_l593_59353


namespace log_expression_evaluation_l593_59301

noncomputable def log2 : ℝ := Real.log 2
noncomputable def log5 : ℝ := Real.log 5

theorem log_expression_evaluation (condition : log2 + log5 = 1) :
  log2^2 + log2 * log5 + log5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  sorry

end log_expression_evaluation_l593_59301


namespace total_number_of_bottles_l593_59346

def water_bottles := 2 * 12
def orange_juice_bottles := (7 / 4) * 12
def apple_juice_bottles := water_bottles + 6
def total_bottles := water_bottles + orange_juice_bottles + apple_juice_bottles

theorem total_number_of_bottles :
  total_bottles = 75 :=
by
  sorry

end total_number_of_bottles_l593_59346


namespace binom_n_n_sub_2_l593_59325

theorem binom_n_n_sub_2 (n : ℕ) (h : n > 0) : (Nat.choose n (n - 2)) = (n * (n - 1)) / 2 := by
  sorry

end binom_n_n_sub_2_l593_59325


namespace digging_depth_l593_59331

theorem digging_depth :
  (∃ (D : ℝ), 750 * D = 75000) → D = 100 :=
by
  sorry

end digging_depth_l593_59331


namespace conference_duration_l593_59378

theorem conference_duration (hours minutes lunch_break total_minutes active_session : ℕ) 
  (h1 : hours = 8) 
  (h2 : minutes = 40) 
  (h3 : lunch_break = 15) 
  (h4 : total_minutes = hours * 60 + minutes)
  (h5 : active_session = total_minutes - lunch_break) :
  active_session = 505 := 
by {
  sorry
}

end conference_duration_l593_59378


namespace Force_Inversely_Proportional_l593_59394

theorem Force_Inversely_Proportional
  (L₁ F₁ L₂ F₂ : ℝ)
  (h₁ : L₁ = 12)
  (h₂ : F₁ = 480)
  (h₃ : L₂ = 18)
  (h_inv : F₁ * L₁ = F₂ * L₂) :
  F₂ = 320 :=
by
  sorry

end Force_Inversely_Proportional_l593_59394


namespace max_distance_right_triangle_l593_59354

theorem max_distance_right_triangle (a b : ℝ) 
  (h1: ∃ (A B : ℝ × ℝ), A ≠ B ∧ 
    (A.1 ^ 2 + A.2 ^ 2 = 1) ∧ (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
    (a * A.1 + 2 * b * A.2 = 1) ∧ (a * B.1 + 2 * b * B.2 = 1) ∧ 
    ∃ (C : ℝ × ℝ), C = (0,0) ∧ (A.1 * B.1 + A.2 * B.2 = 0)): 
  ∃ (d : ℝ), d = (Real.sqrt (a^2 + b^2)) ∧ d ≤ Real.sqrt 2 :=
sorry

end max_distance_right_triangle_l593_59354


namespace percent_gold_coins_l593_59300

variables (total_objects : ℝ) (coins_beads_percent beads_percent gold_coins_percent : ℝ)
           (h1 : coins_beads_percent = 0.75)
           (h2 : beads_percent = 0.15)
           (h3 : gold_coins_percent = 0.60)

theorem percent_gold_coins : (gold_coins_percent * (coins_beads_percent - beads_percent)) = 0.36 :=
by
  have coins_percent := coins_beads_percent - beads_percent
  have gold_coins_total_percent := gold_coins_percent * coins_percent
  exact sorry

end percent_gold_coins_l593_59300


namespace continuity_necessity_not_sufficiency_l593_59321

theorem continuity_necessity_not_sufficiency (f : ℝ → ℝ) (x₀ : ℝ) :
  ((∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) → f x₀ = f x₀) ∧ ¬ ((f x₀ = f x₀) → (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε)) := 
sorry

end continuity_necessity_not_sufficiency_l593_59321


namespace ratio_of_lengths_l593_59308

noncomputable def total_fence_length : ℝ := 640
noncomputable def short_side_length : ℝ := 80

theorem ratio_of_lengths (L S : ℝ) (h1 : 2 * L + 2 * S = total_fence_length) (h2 : S = short_side_length) :
  L / S = 3 :=
by {
  sorry
}

end ratio_of_lengths_l593_59308


namespace problem_l593_59398

theorem problem (p q : ℕ) (hp: p > 1) (hq: q > 1) (h1 : (2 * p - 1) % q = 0) (h2 : (2 * q - 1) % p = 0) : p + q = 8 := 
sorry

end problem_l593_59398


namespace cookies_per_day_l593_59356

theorem cookies_per_day (cost_per_cookie : ℕ) (total_spent : ℕ) (days_in_march : ℕ) (h1 : cost_per_cookie = 16) (h2 : total_spent = 992) (h3 : days_in_march = 31) :
  (total_spent / cost_per_cookie) / days_in_march = 2 :=
by sorry

end cookies_per_day_l593_59356


namespace difference_of_numbers_l593_59310

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 25) (h2 : x * y = 144) : |x - y| = 7 := 
by
  sorry

end difference_of_numbers_l593_59310


namespace length_of_hallway_is_six_l593_59370

noncomputable def length_of_hallway (total_area_square_feet : ℝ) (central_area_side_length : ℝ) (hallway_width : ℝ) : ℝ :=
  (total_area_square_feet - (central_area_side_length * central_area_side_length)) / hallway_width

theorem length_of_hallway_is_six 
  (total_area_square_feet : ℝ)
  (central_area_side_length : ℝ)
  (hallway_width : ℝ)
  (h1 : total_area_square_feet = 124)
  (h2 : central_area_side_length = 10)
  (h3 : hallway_width = 4) :
  length_of_hallway total_area_square_feet central_area_side_length hallway_width = 6 := by
  sorry

end length_of_hallway_is_six_l593_59370


namespace total_cows_l593_59303

def number_of_cows_in_herd : ℕ := 40
def number_of_herds : ℕ := 8
def total_number_of_cows (cows_per_herd herds : ℕ) : ℕ := cows_per_herd * herds

theorem total_cows : total_number_of_cows number_of_cows_in_herd number_of_herds = 320 := by
  sorry

end total_cows_l593_59303


namespace bullet_train_speed_is_70kmph_l593_59329

noncomputable def bullet_train_speed (train_length time_man  : ℚ) (man_speed_kmph : ℕ) : ℚ :=
  let man_speed_ms : ℚ := man_speed_kmph * 1000 / 3600
  let relative_speed : ℚ := train_length / time_man
  let train_speed_ms : ℚ := relative_speed - man_speed_ms
  train_speed_ms * 3600 / 1000

theorem bullet_train_speed_is_70kmph :
  bullet_train_speed 160 7.384615384615384 8 = 70 :=
by {
  -- Proof is omitted
  sorry
}

end bullet_train_speed_is_70kmph_l593_59329


namespace series_largest_prime_factor_of_111_l593_59333

def series := [368, 689, 836]  -- given sequence series

def div_condition (n : Nat) := 
  ∃ k : Nat, n = 111 * k

def largest_prime_factor (n : Nat) (p : Nat) := 
  Prime p ∧ ∀ q : Nat, Prime q → q ∣ n → q ≤ p

theorem series_largest_prime_factor_of_111 :
  largest_prime_factor 111 37 := 
by
  sorry

end series_largest_prime_factor_of_111_l593_59333


namespace domain_of_f_exp_l593_59349

theorem domain_of_f_exp (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x + 1 ∧ x + 1 < 4 → ∃ y, f y = f (x + 1)) →
  (∀ x, 1 ≤ 2^x ∧ 2^x < 4 → ∃ y, f y = f (2^x)) :=
by
  sorry

end domain_of_f_exp_l593_59349


namespace customer_payment_strawberries_watermelons_max_discount_value_l593_59367

-- Definitions for prices
def price_strawberries : ℕ := 60
def price_jingbai_pears : ℕ := 65
def price_watermelons : ℕ := 80
def price_peaches : ℕ := 90

-- Definition for condition on minimum purchase for promotion
def min_purchase_for_promotion : ℕ := 120

-- Definition for percentage Li Ming receives
def li_ming_percentage : ℕ := 80
def customer_percentage : ℕ := 100

-- Proof problem for part 1
theorem customer_payment_strawberries_watermelons (x : ℕ) (total_price : ℕ) :
  x = 10 →
  total_price = price_strawberries + price_watermelons →
  total_price >= min_purchase_for_promotion →
  total_price - x = 130 :=
  by sorry

-- Proof problem for part 2
theorem max_discount_value (m x : ℕ) :
  m >= min_purchase_for_promotion →
  (m - x) * li_ming_percentage / customer_percentage ≥ m * 7 / 10 →
  x ≤ m / 8 :=
  by sorry

end customer_payment_strawberries_watermelons_max_discount_value_l593_59367


namespace inclination_angle_of_line_m_l593_59337

theorem inclination_angle_of_line_m
  (m : ℝ → ℝ → Prop)
  (l₁ l₂ : ℝ → ℝ → Prop)
  (h₁ : ∀ x y, l₁ x y ↔ x - y + 1 = 0)
  (h₂ : ∀ x y, l₂ x y ↔ x - y - 1 = 0)
  (intersect_segment_length : ℝ)
  (h₃ : intersect_segment_length = 2 * Real.sqrt 2) :
  (∃ α : ℝ, (α = 15 ∨ α = 75) ∧ (∃ k : ℝ, ∀ x y, m x y ↔ y = k * x)) :=
by
  sorry

end inclination_angle_of_line_m_l593_59337


namespace magazine_ad_extra_cost_l593_59330

/--
The cost of purchasing a laptop through a magazine advertisement includes four monthly 
payments of $60.99 each and a one-time shipping and handling fee of $19.99. The in-store 
price of the laptop is $259.99. Prove that purchasing the laptop through the magazine 
advertisement results in an extra cost of 396 cents.
-/
theorem magazine_ad_extra_cost : 
  let in_store_price := 259.99
  let monthly_payment := 60.99
  let num_payments := 4
  let shipping_handling := 19.99
  let total_magazine_cost := (num_payments * monthly_payment) + shipping_handling
  (total_magazine_cost - in_store_price) * 100 = 396 := 
by
  sorry

end magazine_ad_extra_cost_l593_59330


namespace average_stoppage_time_per_hour_l593_59323

theorem average_stoppage_time_per_hour :
    ∀ (v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl : ℝ),
    v1_excl = 54 → v1_incl = 36 →
    v2_excl = 72 → v2_incl = 48 →
    v3_excl = 90 → v3_incl = 60 →
    ( ((54 / v1_excl - 54 / v1_incl) + (72 / v2_excl - 72 / v2_incl) + (90 / v3_excl - 90 / v3_incl)) / 3 = 0.5 ) := 
by
    intros v1_excl v1_incl v2_excl v2_incl v3_excl v3_incl
    sorry

end average_stoppage_time_per_hour_l593_59323


namespace triangle_area_50_l593_59366

theorem triangle_area_50 :
  let A := (0, 0)
  let B := (0, 10)
  let C := (-10, 0)
  let base := 10
  let height := 10
  0 + base * height / 2 = 50 := by
sorry

end triangle_area_50_l593_59366


namespace gross_revenue_is_47_l593_59339

def total_net_profit : ℤ := 44
def babysitting_profit : ℤ := 31
def lemonade_stand_expense : ℤ := 34

def gross_revenue_from_lemonade_stand (P_t P_b E : ℤ) : ℤ :=
  P_t - P_b + E

theorem gross_revenue_is_47 :
  gross_revenue_from_lemonade_stand total_net_profit babysitting_profit lemonade_stand_expense = 47 :=
by
  sorry

end gross_revenue_is_47_l593_59339


namespace avg_transformation_l593_59344

theorem avg_transformation
  (x₁ x₂ x₃ x₄ x₅ : ℝ)
  (h : (x₁ + x₂ + x₃ + x₄ + x₅) / 5 = 2) :
  ((3 * x₁ + 1) + (3 * x₂ + 1) + (3 * x₃ + 1) + (3 * x₄ + 1) + (3 * x₅ + 1)) / 5 = 7 :=
by
  sorry

end avg_transformation_l593_59344


namespace total_length_correct_l593_59377

def segment_lengths_Figure1 : List ℕ := [10, 3, 1, 1, 5, 7]

def removed_segments : List ℕ := [3, 1, 1, 5]

def remaining_segments_Figure2 : List ℕ := [10, (3 + 1 + 1), 7, 1]

def total_length_Figure2 : ℕ := remaining_segments_Figure2.sum

theorem total_length_correct :
  total_length_Figure2 = 23 :=
by
  sorry

end total_length_correct_l593_59377


namespace area_of_cross_l593_59334

-- Definitions based on the conditions
def congruent_squares (n : ℕ) := n = 5
def perimeter_of_cross (p : ℕ) := p = 72

-- Targeting the proof that the area of the cross formed by the squares is 180 square units
theorem area_of_cross (n p : ℕ) (h1 : congruent_squares n) (h2 : perimeter_of_cross p) : 
  5 * (p / 12) ^ 2 = 180 := 
by 
  sorry

end area_of_cross_l593_59334


namespace product_of_solutions_l593_59388

theorem product_of_solutions (x : ℝ) :
  ∃ (α β : ℝ), (x^2 - 4*x - 21 = 0) ∧ α * β = -21 := sorry

end product_of_solutions_l593_59388


namespace sum_is_zero_l593_59361

-- Define the conditions: the function f is invertible, and f(a) = 3, f(b) = 7
variables {α β : Type} [Inhabited α] [Inhabited β]

def invertible {α β : Type} (f : α → β) :=
  ∃ g : β → α, (∀ x, g (f x) = x) ∧ (∀ y, f (g y) = y)

variables (f : ℝ → ℝ) (a b : ℝ)

-- Assume f is invertible and the given conditions f(a) = 3 and f(b) = 7
axiom f_invertible : invertible f
axiom f_a : f a = 3
axiom f_b : f b = 7

-- Prove that a + b = 0
theorem sum_is_zero : a + b = 0 :=
sorry

end sum_is_zero_l593_59361


namespace bus_ride_difference_l593_59387

def oscars_bus_ride : ℝ := 0.75
def charlies_bus_ride : ℝ := 0.25

theorem bus_ride_difference :
  oscars_bus_ride - charlies_bus_ride = 0.50 :=
by
  sorry

end bus_ride_difference_l593_59387


namespace find_x_l593_59315

theorem find_x (x y z : ℤ) (h1 : 4 * x + y + z = 80) (h2 : 2 * x - y - z = 40) (h3 : 3 * x + y - z = 20) : x = 20 := by
  sorry

end find_x_l593_59315


namespace simplify_exponents_product_l593_59342

theorem simplify_exponents_product :
  (10^0.5) * (10^0.25) * (10^0.15) * (10^0.05) * (10^1.05) = 100 := by
sorry

end simplify_exponents_product_l593_59342


namespace find_g_3_l593_59345

def g (x : ℝ) : ℝ := sorry

theorem find_g_3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 3) : g 3 = 0 := 
by
  sorry

end find_g_3_l593_59345


namespace total_amount_l593_59357

theorem total_amount (W X Y Z : ℝ) (h1 : X = 0.8 * W) (h2 : Y = 0.65 * W) (h3 : Z = 0.45 * W) (h4 : Y = 78) : 
  W + X + Y + Z = 348 := by
  sorry

end total_amount_l593_59357


namespace abs_diff_roots_eq_sqrt_13_l593_59371

theorem abs_diff_roots_eq_sqrt_13 {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 3 = 0 ∧ x₂^2 + x₂ - 3 = 0) :
  |x₁ - x₂| = Real.sqrt 13 :=
sorry

end abs_diff_roots_eq_sqrt_13_l593_59371


namespace find_constant_l593_59368

theorem find_constant
  (k : ℝ)
  (r : ℝ := 36)
  (C : ℝ := 72 * k)
  (h1 : C = 2 * Real.pi * r)
  : k = Real.pi := by
  sorry

end find_constant_l593_59368


namespace calculate_expression_l593_59347

theorem calculate_expression (x : ℝ) (h : x + 1/x = 3) : x^12 - 7 * x^6 + x^2 = 45363 * x - 17327 :=
by
  sorry

end calculate_expression_l593_59347


namespace polynomial_roots_arithmetic_progression_l593_59332

theorem polynomial_roots_arithmetic_progression (m n : ℝ)
  (h : ∃ a : ℝ, ∃ d : ℝ, ∃ b : ℝ,
   (a = b ∧ (b + d) + (b + 2*d) + (b + 3*d) + b = 0) ∧
   (b * (b + d) * (b + 2*d) * (b + 3*d) = 144) ∧
   b ≠ (b + d) ∧ (b + d) ≠ (b + 2*d) ∧ (b + 2*d) ≠ (b + 3*d)) :
  m = -40 := sorry

end polynomial_roots_arithmetic_progression_l593_59332


namespace cos_C_in_triangle_l593_59343

theorem cos_C_in_triangle (A B C : ℝ)
  (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_cos_A : Real.cos A = 3/5)
  (h_sin_B : Real.sin B = 12/13) :
  Real.cos C = 63/65 ∨ Real.cos C = 33/65 :=
sorry

end cos_C_in_triangle_l593_59343


namespace find_total_bricks_l593_59386

variable (y : ℕ)
variable (B_rate : ℕ)
variable (N_rate : ℕ)
variable (eff_rate : ℕ)
variable (time : ℕ)
variable (reduction : ℕ)

-- The wall is completed in 6 hours
def completed_in_time (y B_rate N_rate eff_rate time reduction : ℕ) : Prop := 
  time = 6 ∧
  reduction = 8 ∧
  B_rate = y / 8 ∧
  N_rate = y / 12 ∧
  eff_rate = (B_rate + N_rate) - reduction ∧
  y = eff_rate * time

-- Prove that the number of bricks in the wall is 192
theorem find_total_bricks : 
  ∀ (y B_rate N_rate eff_rate time reduction : ℕ), 
  completed_in_time y B_rate N_rate eff_rate time reduction → 
  y = 192 := 
by 
  sorry

end find_total_bricks_l593_59386


namespace cost_of_each_pair_of_shorts_l593_59309

variable (C : ℝ)
variable (h_discount : 3 * C - 2.7 * C = 3)

theorem cost_of_each_pair_of_shorts : C = 10 :=
by 
  sorry

end cost_of_each_pair_of_shorts_l593_59309


namespace ball_distribution_ways_l593_59355

theorem ball_distribution_ways :
  ∃ (ways : ℕ), ways = 10 ∧
    ∀ (balls boxes : ℕ), 
    balls = 6 ∧ boxes = 4 ∧ 
    (∀ (b : ℕ), b < boxes → b > 0) →
    ways = 10 :=
sorry

end ball_distribution_ways_l593_59355


namespace trapezium_area_l593_59314

theorem trapezium_area (a b d : ℕ) (h₁ : a = 28) (h₂ : b = 18) (h₃ : d = 15) :
  (a + b) * d / 2 = 345 := by
{
  sorry
}

end trapezium_area_l593_59314


namespace number_of_girls_in_class_l593_59385

theorem number_of_girls_in_class (B G : ℕ) (h1 : G = 4 * B / 10) (h2 : B + G = 35) : G = 10 :=
by
  sorry

end number_of_girls_in_class_l593_59385


namespace width_of_wall_is_6_l593_59392

-- Definitions of the conditions given in the problem
def height_of_wall (w : ℝ) := 4 * w
def length_of_wall (h : ℝ) := 3 * h
def volume_of_wall (w h l : ℝ) := w * h * l

-- Proof statement that the width of the wall is 6 meters given the conditions
theorem width_of_wall_is_6 :
  ∃ w : ℝ, 
  (height_of_wall w = 4 * w) ∧ 
  (length_of_wall (height_of_wall w) = 3 * (height_of_wall w)) ∧ 
  (volume_of_wall w (height_of_wall w) (length_of_wall (height_of_wall w)) = 10368) ∧ 
  (w = 6) :=
sorry

end width_of_wall_is_6_l593_59392


namespace fraction_exponent_multiplication_l593_59379

theorem fraction_exponent_multiplication :
  ( (8/9 : ℚ)^2 * (1/3 : ℚ)^2 = (64/729 : ℚ) ) :=
by
  -- here we would write out the detailed proof
  sorry

end fraction_exponent_multiplication_l593_59379


namespace ajays_monthly_income_l593_59374

theorem ajays_monthly_income :
  ∀ (I : ℝ), 
  (0.50 * I) + (0.25 * I) + (0.15 * I) + 9000 = I → I = 90000 :=
by
  sorry

end ajays_monthly_income_l593_59374


namespace peter_runs_more_than_andrew_each_day_l593_59351

-- Define the constants based on the conditions
def miles_andrew : ℕ := 2
def total_days : ℕ := 5
def total_miles : ℕ := 35

-- Define a theorem to prove the number of miles Peter runs more than Andrew each day
theorem peter_runs_more_than_andrew_each_day : 
  ∃ x : ℕ, total_days * (miles_andrew + x) + total_days * miles_andrew = total_miles ∧ x = 3 :=
by
  sorry

end peter_runs_more_than_andrew_each_day_l593_59351
