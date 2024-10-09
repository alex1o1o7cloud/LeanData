import Mathlib

namespace total_volume_l2083_208390

open Real

noncomputable def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h
noncomputable def volume_cone (r h : ℝ) : ℝ := (1/3) * π * r^2 * h

theorem total_volume {d_cylinder d_cone_top d_cone_bottom h_cylinder h_cone : ℝ}
  (h1 : d_cylinder = 2) (h2 : d_cone_top = 2) (h3 : d_cone_bottom = 1)
  (h4 : h_cylinder = 14) (h5 : h_cone = 4) :
  volume_cylinder (d_cylinder / 2) h_cylinder +
  volume_cone (d_cone_top / 2) h_cone =
  (46 / 3) * π :=
by
  sorry

end total_volume_l2083_208390


namespace prime_root_range_l2083_208387

-- Let's define our conditions first
def is_prime (p : ℕ) : Prop := Nat.Prime p

def has_integer_roots (p : ℕ) : Prop :=
  ∃ (x y : ℤ), x ≠ y ∧ x + y = p ∧ x * y = -156 * p

-- Now state the theorem
theorem prime_root_range (p : ℕ) (hp : is_prime p) (hr : has_integer_roots p) : 11 < p ∧ p ≤ 21 :=
by
  sorry

end prime_root_range_l2083_208387


namespace find_x_l2083_208347

theorem find_x (x y z p q r: ℝ) 
  (h1 : (x * y) / (x + y) = p)
  (h2 : (x * z) / (x + z) = q)
  (h3 : (y * z) / (y + z) = r)
  (hp_nonzero : p ≠ 0)
  (hq_nonzero : q ≠ 0)
  (hr_nonzero : r ≠ 0)
  (hxy : x ≠ -y)
  (hxz : x ≠ -z)
  (hyz : y ≠ -z)
  (hpq : p = 3 * q)
  (hpr : p = 2 * r) : x = 3 * p / 2 := 
sorry

end find_x_l2083_208347


namespace twentieth_fisherman_catch_l2083_208379

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l2083_208379


namespace sum_mod_9237_9241_l2083_208386

theorem sum_mod_9237_9241 :
  (9237 + 9238 + 9239 + 9240 + 9241) % 9 = 2 :=
by
  sorry

end sum_mod_9237_9241_l2083_208386


namespace max_possible_x_l2083_208376

theorem max_possible_x (x y z : ℝ) 
  (h1 : 3 * x + 2 * y + z = 10)
  (h2 : x * y + x * z + y * z = 6) :
  x ≤ 2 * Real.sqrt 5 / 5 :=
sorry

end max_possible_x_l2083_208376


namespace seats_per_row_and_total_students_l2083_208305

theorem seats_per_row_and_total_students (R S : ℕ) 
  (h1 : S = 5 * R + 6) 
  (h2 : S = 12 * (R - 3)) : 
  R = 6 ∧ S = 36 := 
by 
  sorry

end seats_per_row_and_total_students_l2083_208305


namespace problem_eight_sided_polygon_interiors_l2083_208370

-- Define the condition of the problem
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

-- The sum of the interior angles of a regular polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- One interior angle of a regular polygon
def one_interior_angle (n : ℕ) : ℚ := sum_of_interior_angles n / n

-- The main theorem stating the problem
theorem problem_eight_sided_polygon_interiors (n : ℕ) (h1: diagonals_from_vertex n = 5) : 
  one_interior_angle n = 135 :=
by
  -- Proof would go here
  sorry

end problem_eight_sided_polygon_interiors_l2083_208370


namespace fill_time_two_pipes_l2083_208303

variable (R : ℝ)
variable (c : ℝ)
variable (t1 : ℝ) (t2 : ℝ)

noncomputable def fill_time_with_pipes (num_pipes : ℝ) (time_per_tank : ℝ) : ℝ :=
  time_per_tank / num_pipes

theorem fill_time_two_pipes (h1 : fill_time_with_pipes 3 t1 = 12) 
                            (h2 : c = R)
                            : fill_time_with_pipes 2 (3 * R * t1) = 18 := 
by
  sorry

end fill_time_two_pipes_l2083_208303


namespace graph_always_passes_fixed_point_l2083_208322

theorem graph_always_passes_fixed_point (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) : 
  ∃ A : ℝ × ℝ, A = (-2, -1) ∧ (∀ x : ℝ, y = a^(x+2)-2 → y = -1 ∧ x = -2) :=
by
  use (-2, -1)
  sorry

end graph_always_passes_fixed_point_l2083_208322


namespace check_ratio_l2083_208313

theorem check_ratio (initial_balance check_amount new_balance : ℕ) 
  (h1 : initial_balance = 150) (h2 : check_amount = 50) (h3 : new_balance = initial_balance + check_amount) :
  (check_amount : ℚ) / new_balance = 1 / 4 := 
by { 
  sorry 
}

end check_ratio_l2083_208313


namespace beavers_still_working_l2083_208384

theorem beavers_still_working (total_beavers : ℕ) (wood_beavers dam_beavers lodge_beavers : ℕ)
  (wood_swimming dam_swimming lodge_swimming : ℕ) :
  total_beavers = 12 →
  wood_beavers = 5 →
  dam_beavers = 4 →
  lodge_beavers = 3 →
  wood_swimming = 3 →
  dam_swimming = 2 →
  lodge_swimming = 1 →
  (wood_beavers - wood_swimming) + (dam_beavers - dam_swimming) + (lodge_beavers - lodge_swimming) = 6 :=
by
  intros h_total h_wood h_dam h_lodge h_wood_swim h_dam_swim h_lodge_swim
  sorry

end beavers_still_working_l2083_208384


namespace ratio_out_of_school_friends_to_classmates_l2083_208306

variable (F : ℕ) (classmates : ℕ := 20) (parents : ℕ := 2) (sister : ℕ := 1) (total : ℕ := 33)

theorem ratio_out_of_school_friends_to_classmates (h : classmates + F + parents + sister = total) :
  (F : ℚ) / classmates = 1 / 2 := by
    -- sorry allows this to build even if proof is not provided
    sorry

end ratio_out_of_school_friends_to_classmates_l2083_208306


namespace system_solution_in_first_quadrant_l2083_208309

theorem system_solution_in_first_quadrant (c x y : ℝ)
  (h1 : x - y = 5)
  (h2 : c * x + y = 7)
  (hx : x > 3)
  (hy : y > 1) : c < 1 :=
sorry

end system_solution_in_first_quadrant_l2083_208309


namespace new_cylinder_volume_l2083_208343

theorem new_cylinder_volume (r h : ℝ) (π_ne_zero : 0 < π) (original_volume : π * r^2 * h = 10) : 
  π * (3 * r)^2 * (2 * h) = 180 :=
by
  sorry

end new_cylinder_volume_l2083_208343


namespace sugar_total_l2083_208398

variable (sugar_for_frosting sugar_for_cake : ℝ)

theorem sugar_total (h1 : sugar_for_frosting = 0.6) (h2 : sugar_for_cake = 0.2) :
  sugar_for_frosting + sugar_for_cake = 0.8 :=
by
  sorry

end sugar_total_l2083_208398


namespace graph_of_equation_l2083_208369

theorem graph_of_equation (x y : ℝ) :
  x^2 - y^2 = 0 ↔ (y = x ∨ y = -x) := 
by sorry

end graph_of_equation_l2083_208369


namespace find_p_at_8_l2083_208302

noncomputable def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

noncomputable def p (x : ℝ) : ℝ :=
  let a := sorry ; -- root 1 of h
  let b := sorry ; -- root 2 of h
  let c := sorry ; -- root 3 of h
  let B := 2 / ((1 - a^3) * (1 - b^3) * (1 - c^3))
  B * (x - a^3) * (x - b^3) * (x - c^3)

theorem find_p_at_8 : p 8 = 1008 := sorry

end find_p_at_8_l2083_208302


namespace tangent_30_degrees_l2083_208334

theorem tangent_30_degrees (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) (hA : ∃ α : ℝ, α = 30 ∧ (y / x) = Real.tan (π / 6)) :
  y / x = Real.sqrt 3 / 3 :=
by
  sorry

end tangent_30_degrees_l2083_208334


namespace space_station_cost_share_l2083_208382

def total_cost : ℤ := 50 * 10^9
def people_count : ℤ := 500 * 10^6
def per_person_share (C N : ℤ) : ℤ := C / N

theorem space_station_cost_share :
  per_person_share total_cost people_count = 100 :=
by
  sorry

end space_station_cost_share_l2083_208382


namespace triangle_area_from_perimeter_and_inradius_l2083_208337

theorem triangle_area_from_perimeter_and_inradius
  (P : ℝ) (r : ℝ) (A : ℝ)
  (h₁ : P = 24)
  (h₂ : r = 2.5) :
  A = 30 := 
by
  sorry

end triangle_area_from_perimeter_and_inradius_l2083_208337


namespace horner_evaluation_l2083_208372

def f (x : ℝ) := x^5 + 3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 11

theorem horner_evaluation : f 4 = 1559 := by
  sorry

end horner_evaluation_l2083_208372


namespace miles_remaining_l2083_208304

theorem miles_remaining (total_miles driven_miles : ℕ) (h1 : total_miles = 1200) (h2 : driven_miles = 768) :
    total_miles - driven_miles = 432 := by
  sorry

end miles_remaining_l2083_208304


namespace min_students_in_class_l2083_208333

theorem min_students_in_class (b g : ℕ) (hb : 3 * b = 4 * g) : b + g = 7 :=
sorry

end min_students_in_class_l2083_208333


namespace best_possible_overall_standing_l2083_208335

noncomputable def N : ℕ := 100 -- number of participants
noncomputable def M : ℕ := 14  -- number of stages

-- Define a competitor finishing 93rd in each stage
def finishes_93rd_each_stage (finishes : ℕ → ℕ) : Prop :=
  ∀ i, i < M → finishes i = 93

-- Define the best possible overall standing
theorem best_possible_overall_standing
  (finishes : ℕ → ℕ) -- function representing stage finishes for the competitor
  (h : finishes_93rd_each_stage finishes) :
  ∃ k, k = 2 := 
sorry

end best_possible_overall_standing_l2083_208335


namespace eval_expression_l2083_208344

theorem eval_expression (a x : ℕ) (h : x = a + 9) : x - a + 5 = 14 :=
by 
  sorry

end eval_expression_l2083_208344


namespace find_F1C_CG1_l2083_208397

variable {A B C D E F G H E1 F1 G1 H1 : Type*}
variables (AE EB BF FC CG GD DH HA E1A AH1 F1C CG1 : ℝ) (a : ℝ)

axiom convex_quadrilateral (AE EB BF FC CG GD DH HA : ℝ) : 
  AE / EB * BF / FC * CG / GD * DH / HA = 1 

axiom quadrilaterals_similar 
  (E1F1 EF F1G1 FG G1H1 GH H1E1 HE : Prop) :
  E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True)

axiom given_ratio (E1A AH1 : ℝ) (a : ℝ) :
  E1A / AH1 = a

theorem find_F1C_CG1
  (conv : AE / EB * BF / FC * CG / GD * DH / HA = 1)
  (parallel_lines : E1F1 → EF → F1G1 → FG → G1H1 → GH → H1E1 → HE → (True))
  (ratio : E1A / AH1 = a) :
  F1C / CG1 = a := 
sorry

end find_F1C_CG1_l2083_208397


namespace find_c_l2083_208358

-- Definitions of r and s
def r (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

-- Given and proved statement
theorem find_c (c : ℝ) : r (s 2 c) = 11 → c = 5 := 
by 
  sorry

end find_c_l2083_208358


namespace possible_values_of_reciprocal_sum_l2083_208341

theorem possible_values_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 2) (h4 : x * y = 1) : 
  1/x + 1/y = 2 := 
sorry

end possible_values_of_reciprocal_sum_l2083_208341


namespace sum_of_six_least_n_l2083_208355

def tau (n : ℕ) : ℕ := Nat.totient n -- Assuming as an example for tau definition

theorem sum_of_six_least_n (h1 : tau 8 + tau 9 = 7)
                           (h2 : tau 9 + tau 10 = 7)
                           (h3 : tau 16 + tau 17 = 7)
                           (h4 : tau 25 + tau 26 = 7)
                           (h5 : tau 121 + tau 122 = 7)
                           (h6 : tau 361 + tau 362 = 7) :
  8 + 9 + 16 + 25 + 121 + 361 = 540 :=
by sorry

end sum_of_six_least_n_l2083_208355


namespace sum_of_diagonals_l2083_208377

def FG : ℝ := 4
def HI : ℝ := 4
def GH : ℝ := 11
def IJ : ℝ := 11
def FJ : ℝ := 15

theorem sum_of_diagonals (x y z : ℝ) (h1 : z^2 = 4 * x + 121) (h2 : z^2 = 11 * y + 16)
  (h3 : x * y = 44 + 15 * z) (h4 : x * z = 4 * z + 225) (h5 : y * z = 11 * z + 60) :
  3 * z + x + y = 90 :=
sorry

end sum_of_diagonals_l2083_208377


namespace part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l2083_208310

noncomputable def f (x : ℝ) := Real.log x
noncomputable def deriv_f (x : ℝ) := 1 / x

theorem part1_am_eq_ln_am1_minus_1 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m = Real.log (a_n (m - 1)) - 1 :=
sorry

theorem part2_am_le_am1_minus_2 (a_n : ℕ → ℝ) (m : ℕ) (h : m ≥ 2) :
  a_n m ≤ a_n (m - 1) - 2 :=
sorry

theorem part3_k_is_3 (a_n : ℕ → ℝ) :
  ∃ k : ℕ, k = 3 ∧ ∀ n : ℕ, n ≤ k → (a_n n) - (a_n (n - 1)) = (a_n 2) - (a_n 1) :=
sorry

end part1_am_eq_ln_am1_minus_1_part2_am_le_am1_minus_2_part3_k_is_3_l2083_208310


namespace rectangle_ratio_l2083_208352

open Real

-- Definition of the terms
variables {x y : ℝ}

-- Conditions as per the problem statement
def diagonalSavingsRect (x y : ℝ) := x + y - sqrt (x^2 + y^2) = (2 / 3) * y

-- The ratio of the shorter side to the longer side of the rectangle
theorem rectangle_ratio
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (h : diagonalSavingsRect x y) : x / y = 8 / 9 :=
by
sorry

end rectangle_ratio_l2083_208352


namespace time_to_ascend_non_working_escalator_l2083_208359

-- Definitions from the conditions
def length_of_escalator := 1
def time_standing := 1
def time_running := 24 / 60
def escalator_speed := 1 / 60
def gavrila_speed := 1 / 40

-- The proof problem statement 
theorem time_to_ascend_non_working_escalator 
  (length_of_escalator : ℝ)
  (time_standing : ℝ)
  (time_running : ℝ)
  (escalator_speed : ℝ)
  (gavrila_speed : ℝ) :
  time_standing = 1 →
  time_running = 24 / 60 →
  escalator_speed = 1 / 60 →
  gavrila_speed = 1 / 40 →
  length_of_escalator = 1 →
  1 / gavrila_speed = 40 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end time_to_ascend_non_working_escalator_l2083_208359


namespace simplify_expression_l2083_208301

variable (y : ℝ)

theorem simplify_expression :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 :=
by
  sorry

end simplify_expression_l2083_208301


namespace customers_in_other_countries_l2083_208321

def total_customers : ℕ := 7422
def us_customers : ℕ := 723
def other_customers : ℕ := total_customers - us_customers

theorem customers_in_other_countries : other_customers = 6699 := by
  sorry

end customers_in_other_countries_l2083_208321


namespace yellow_tint_percentage_l2083_208332

theorem yellow_tint_percentage (V₀ : ℝ) (P₀Y : ℝ) (V_additional : ℝ) 
  (hV₀ : V₀ = 40) (hP₀Y : P₀Y = 0.35) (hV_additional : V_additional = 8) : 
  (100 * ((V₀ * P₀Y + V_additional) / (V₀ + V_additional)) = 45.83) :=
by
  sorry

end yellow_tint_percentage_l2083_208332


namespace infection_in_fourth_round_l2083_208330

-- Define the initial conditions and the function for the geometric sequence
def initial_infected : ℕ := 1
def infection_ratio : ℕ := 20

noncomputable def infected_computers (rounds : ℕ) : ℕ :=
  initial_infected * infection_ratio^(rounds - 1)

-- The theorem to prove
theorem infection_in_fourth_round : infected_computers 4 = 8000 :=
by
  -- proof will be added later
  sorry

end infection_in_fourth_round_l2083_208330


namespace license_plate_palindrome_probability_l2083_208360

theorem license_plate_palindrome_probability :
  let p := 507
  let q := 2028
  p + q = 2535 :=
by
  sorry

end license_plate_palindrome_probability_l2083_208360


namespace black_haired_girls_count_l2083_208329

def initial_total_girls : ℕ := 80
def added_blonde_girls : ℕ := 10
def initial_blonde_girls : ℕ := 30

def total_girls := initial_total_girls + added_blonde_girls
def total_blonde_girls := initial_blonde_girls + added_blonde_girls
def black_haired_girls := total_girls - total_blonde_girls

theorem black_haired_girls_count : black_haired_girls = 50 := by
  sorry

end black_haired_girls_count_l2083_208329


namespace partition_value_l2083_208307

variable {a m n p x k l : ℝ}

theorem partition_value :
  (m * (a - n * x) = k * (a - n * x)) ∧
  (n * x = l * x) ∧
  (a - x = p * (a - m * (a - n * x)))
  → x = (a * (m * p - p + 1)) / (n * m * p + 1) :=
by
  sorry

end partition_value_l2083_208307


namespace problem_statement_l2083_208380

theorem problem_statement (x y : ℝ) (h1 : x + y = 2) (h2 : xy = -2) : (1 - x) * (1 - y) = -3 := by
  sorry

end problem_statement_l2083_208380


namespace dot_product_neg_vec_n_l2083_208336

-- Vector definitions
def vec_m : ℝ × ℝ := (2, -1)
def vec_n : ℝ × ℝ := (3, 2)
def neg_vec_n : ℝ × ℝ := (-vec_n.1, -vec_n.2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Proof statement
theorem dot_product_neg_vec_n :
  dot_product vec_m neg_vec_n = -4 :=
by
  -- Sorry to skip the proof
  sorry

end dot_product_neg_vec_n_l2083_208336


namespace sandy_initial_fish_l2083_208315

theorem sandy_initial_fish (bought_fish : ℕ) (total_fish : ℕ) (h1 : bought_fish = 6) (h2 : total_fish = 32) :
  total_fish - bought_fish = 26 :=
by
  sorry

end sandy_initial_fish_l2083_208315


namespace total_students_in_line_l2083_208327

-- Define the conditions
def students_in_front : Nat := 15
def students_behind : Nat := 12

-- Define the statement to prove: total number of students in line is 28
theorem total_students_in_line : students_in_front + 1 + students_behind = 28 := 
by 
  -- Placeholder for the proof
  sorry

end total_students_in_line_l2083_208327


namespace alice_paid_percentage_l2083_208381

theorem alice_paid_percentage (SRP P : ℝ) (h1 : P = 0.60 * SRP) (h2 : P_alice = 0.60 * P) :
  (P_alice / SRP) * 100 = 36 := by
sorry

end alice_paid_percentage_l2083_208381


namespace trader_profit_l2083_208346

theorem trader_profit
  (CP : ℝ)
  (MP : ℝ)
  (SP : ℝ)
  (h1 : MP = CP * 1.12)
  (discount_percent : ℝ)
  (h2 : discount_percent = 0.09821428571428571)
  (discount : ℝ)
  (h3 : discount = MP * discount_percent)
  (actual_SP : ℝ)
  (h4 : actual_SP = MP - discount)
  (h5 : CP = 100) :
  (actual_SP / CP = 1.01) :=
by
  sorry

end trader_profit_l2083_208346


namespace find_M_l2083_208392

variable (p q r M : ℝ)
variable (h1 : p + q + r = 100)
variable (h2 : p + 10 = M)
variable (h3 : q - 5 = M)
variable (h4 : r / 5 = M)

theorem find_M : M = 15 := by
  sorry

end find_M_l2083_208392


namespace range_of_a_l2083_208363

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ 1 :=
by
  intro h
  sorry

end range_of_a_l2083_208363


namespace geometric_series_sum_l2083_208342

theorem geometric_series_sum :
  2016 * (1 / (1 + (1 / 2) + (1 / 4) + (1 / 8) + (1 / 16) + (1 / 32))) = 1024 :=
by
  sorry

end geometric_series_sum_l2083_208342


namespace IMO1991Q1_l2083_208364

theorem IMO1991Q1 (x y z : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
    (h4 : 3^x + 4^y = 5^z) : x = 2 ∧ y = 2 ∧ z = 2 := by
  sorry

end IMO1991Q1_l2083_208364


namespace hexagon_tiling_min_colors_l2083_208312

theorem hexagon_tiling_min_colors :
  ∀ (s₁ s₂ : ℝ) (hex_area : ℝ) (tile_area : ℝ) (tiles_needed : ℕ) (n : ℕ),
    s₁ = 6 →
    s₂ = 0.5 →
    hex_area = (3 * Real.sqrt 3 / 2) * s₁^2 →
    tile_area = (Real.sqrt 3 / 4) * s₂^2 →
    tiles_needed = hex_area / tile_area →
    tiles_needed ≤ (Nat.choose n 3) →
    n ≥ 19 :=
by
  intros s₁ s₂ hex_area tile_area tiles_needed n
  intros s₁_eq s₂_eq hex_area_eq tile_area_eq tiles_needed_eq color_constraint
  sorry

end hexagon_tiling_min_colors_l2083_208312


namespace polynomial_divisible_by_7_polynomial_divisible_by_12_l2083_208320

theorem polynomial_divisible_by_7 (x : ℤ) : (x^7 - x) % 7 = 0 := 
sorry

theorem polynomial_divisible_by_12 (x : ℤ) : (x^4 - x^2) % 12 = 0 := 
sorry

end polynomial_divisible_by_7_polynomial_divisible_by_12_l2083_208320


namespace solve_abs_eq_l2083_208366

theorem solve_abs_eq (x : ℝ) : |x - 4| = 3 - x ↔ x = 7 / 2 := by
  sorry

end solve_abs_eq_l2083_208366


namespace find_a_no_solution_l2083_208371

noncomputable def no_solution_eq (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (8 * |x - 4 * a| + |x - a^2| + 7 * x - 2 * a = 0)

theorem find_a_no_solution :
  ∀ a : ℝ, no_solution_eq a ↔ (a < -22 ∨ a > 0) :=
by
  intro a
  sorry

end find_a_no_solution_l2083_208371


namespace cubic_has_three_zeros_l2083_208324

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem cubic_has_three_zeros : (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f a = 0 ∧ f b = 0 ∧ f c = 0) :=
sorry

end cubic_has_three_zeros_l2083_208324


namespace hyperbola_focus_and_distance_l2083_208316

noncomputable def right_focus_of_hyperbola (a b : ℝ) : ℝ × ℝ := 
  (Real.sqrt (a^2 + b^2), 0)

noncomputable def distance_to_asymptote (a b : ℝ) : ℝ := 
  let c := Real.sqrt (a^2 + b^2)
  abs c / Real.sqrt (1 + (b/a)^2)

theorem hyperbola_focus_and_distance (a b : ℝ) (h₁ : a^2 = 6) (h₂ : b^2 = 3) :
  right_focus_of_hyperbola a b = (3, 0) ∧ distance_to_asymptote a b = Real.sqrt 3 :=
by
  sorry

end hyperbola_focus_and_distance_l2083_208316


namespace distinct_cubed_mod_7_units_digits_l2083_208396

theorem distinct_cubed_mod_7_units_digits : 
  (∃ S : Finset ℕ, S.card = 3 ∧ ∀ n ∈ (Finset.range 7), (n^3 % 7) ∈ S) :=
  sorry

end distinct_cubed_mod_7_units_digits_l2083_208396


namespace weighted_arithmetic_geometric_mean_l2083_208319
-- Importing required library

-- Definitions of the problem variables and conditions
variables (a b c : ℝ)

-- Non-negative constraints on the lengths of the line segments
variables (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)

-- Problem statement, we need to prove
theorem weighted_arithmetic_geometric_mean :
  0.2 * a + 0.3 * b + 0.5 * c ≥ (a * b * c)^(1/3) :=
sorry

end weighted_arithmetic_geometric_mean_l2083_208319


namespace march_1_falls_on_friday_l2083_208338

-- Definitions of conditions
def march_days : ℕ := 31
def mondays_in_march : ℕ := 4
def thursdays_in_march : ℕ := 4

-- Lean 4 statement to prove March 1 falls on a Friday
theorem march_1_falls_on_friday 
  (h1 : march_days = 31)
  (h2 : mondays_in_march = 4)
  (h3 : thursdays_in_march = 4)
  : ∃ d : ℕ, d = 5 :=
by sorry

end march_1_falls_on_friday_l2083_208338


namespace arrangement_of_70616_l2083_208317

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangement_count (digits : List ℕ) : ℕ :=
  let count := digits.length
  let duplicates := List.length (List.filter (fun x => x = 6) digits)
  factorial count / factorial duplicates

theorem arrangement_of_70616 : arrangement_count [7, 0, 6, 6, 1] = 4 * 12 := by
  -- We need to prove that the number of ways to arrange the digits 7, 0, 6, 6, 1 without starting with 0 is 48
  sorry

end arrangement_of_70616_l2083_208317


namespace original_number_of_bullets_each_had_l2083_208339

theorem original_number_of_bullets_each_had (x : ℕ) (h₁ : 5 * (x - 4) = x) : x = 5 := 
sorry

end original_number_of_bullets_each_had_l2083_208339


namespace inequality_l2083_208395

def domain (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality (a b : ℝ) (ha : domain a) (hb : domain b) :
  |a + b| < |3 + ab / 3| :=
by
  sorry

end inequality_l2083_208395


namespace average_percentage_revenue_fall_l2083_208300

theorem average_percentage_revenue_fall
  (initial_revenue_A final_revenue_A : ℝ)
  (initial_revenue_B final_revenue_B : ℝ) (exchange_rate_B : ℝ)
  (initial_revenue_C final_revenue_C : ℝ) (exchange_rate_C : ℝ) :
  initial_revenue_A = 72.0 →
  final_revenue_A = 48.0 →
  initial_revenue_B = 20.0 →
  final_revenue_B = 15.0 →
  exchange_rate_B = 1.30 →
  initial_revenue_C = 6000.0 →
  final_revenue_C = 5500.0 →
  exchange_rate_C = 0.0091 →
  (33.33 + 25 + 8.33) / 3 = 22.22 :=
by
  sorry

end average_percentage_revenue_fall_l2083_208300


namespace ducks_in_marsh_l2083_208340

theorem ducks_in_marsh 
  (num_geese : ℕ) 
  (total_birds : ℕ) 
  (num_ducks : ℕ)
  (h1 : num_geese = 58) 
  (h2 : total_birds = 95) 
  (h3 : total_birds = num_geese + num_ducks) : 
  num_ducks = 37 :=
by
  sorry

end ducks_in_marsh_l2083_208340


namespace atomic_weight_of_oxygen_l2083_208331

theorem atomic_weight_of_oxygen (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) (molecular_weight_Al2O3 : ℝ) (n_Al : ℕ) (n_O : ℕ) :
  atomic_weight_Al = 26.98 →
  molecular_weight_Al2O3 = 102 →
  n_Al = 2 →
  n_O = 3 →
  (molecular_weight_Al2O3 - n_Al * atomic_weight_Al) / n_O = 16.01 :=
by
  sorry

end atomic_weight_of_oxygen_l2083_208331


namespace students_band_and_chorus_l2083_208385

theorem students_band_and_chorus (Total Band Chorus Union Intersection : ℕ) 
  (h₁ : Total = 300) 
  (h₂ : Band = 110) 
  (h₃ : Chorus = 140) 
  (h₄ : Union = 220) :
  Intersection = Band + Chorus - Union :=
by
  -- Given the conditions, the proof would follow here.
  sorry

end students_band_and_chorus_l2083_208385


namespace boundary_line_f_g_l2083_208362

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

noncomputable def g (x : ℝ) : ℝ := 0.5 * (x - 1 / x)

theorem boundary_line_f_g :
  ∀ (x : ℝ), 1 ≤ x → (x - 1) ≤ f x ∧ (g x) ≤ (x - 1) :=
by
  intro x hx
  sorry

end boundary_line_f_g_l2083_208362


namespace aluminum_weight_l2083_208367

variable {weight_iron : ℝ}
variable {weight_aluminum : ℝ}
variable {difference : ℝ}

def weight_aluminum_is_correct (weight_iron weight_aluminum difference : ℝ) : Prop := 
  weight_iron = weight_aluminum + difference

theorem aluminum_weight 
  (H1 : weight_iron = 11.17)
  (H2 : difference = 10.33)
  (H3 : weight_aluminum_is_correct weight_iron weight_aluminum difference) : 
  weight_aluminum = 0.84 :=
sorry

end aluminum_weight_l2083_208367


namespace unit_digit_of_15_pow_100_l2083_208394

-- Define a function to extract the unit digit of a number
def unit_digit (n : ℕ) : ℕ := n % 10

-- Given conditions:
def base : ℕ := 15
def exponent : ℕ := 100

-- Define what 'unit_digit' of a number raised to an exponent means
def unit_digit_pow (base exponent : ℕ) : ℕ :=
  unit_digit (base ^ exponent)

-- Goal: Prove that the unit digit of 15^100 is 5.
theorem unit_digit_of_15_pow_100 : unit_digit_pow base exponent = 5 :=
by
  sorry

end unit_digit_of_15_pow_100_l2083_208394


namespace gcd_12m_18n_with_gcd_mn_18_l2083_208328

theorem gcd_12m_18n_with_gcd_mn_18 (m n : ℕ) (hm : Nat.gcd m n = 18) (hm_pos : 0 < m) (hn_pos : 0 < n) :
  Nat.gcd (12 * m) (18 * n) = 108 :=
by sorry

end gcd_12m_18n_with_gcd_mn_18_l2083_208328


namespace b1f_hex_to_dec_l2083_208375

/-- 
  Convert the given hexadecimal digit to its corresponding decimal value.
  -/
def hex_to_dec (c : Char) : Nat :=
  match c with
  | 'A' => 10
  | 'B' => 11
  | 'C' => 12
  | 'D' => 13
  | 'E' => 14
  | 'F' => 15
  | '0' => 0
  | '1' => 1
  | '2' => 2
  | '3' => 3
  | '4' => 4
  | '5' => 5
  | '6' => 6
  | '7' => 7
  | '8' => 8
  | '9' => 9
  | _ => 0

/-- 
  Convert a hexadecimal string to a decimal number.
  -/
def hex_string_to_dec (s : String) : Nat :=
  s.foldl (λ acc c => acc * 16 + hex_to_dec c) 0

theorem b1f_hex_to_dec : hex_string_to_dec "B1F" = 2847 :=
by
  sorry

end b1f_hex_to_dec_l2083_208375


namespace fraction_of_number_l2083_208368

theorem fraction_of_number (a b c d : ℝ) (h1 : a = 7) (h2 : b = 8) (h3 : c = 48) (h4 : d = 42) :
  (a / b) * c = d :=
by 
  rw [h1, h2, h3, h4]
  -- The proof steps would go here
  sorry

end fraction_of_number_l2083_208368


namespace sum_of_even_factors_900_l2083_208393

theorem sum_of_even_factors_900 : 
  ∃ (S : ℕ), 
  (∀ a b c : ℕ, 900 = 2^a * 3^b * 5^c → 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → 0 ≤ c ∧ c ≤ 2) → 
  (∀ a : ℕ, 1 ≤ a ∧ a ≤ 2 → ∃ b c : ℕ, 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ (2^a * 3^b * 5^c = 900 ∧ a ≠ 0)) → 
  S = 2418 := 
sorry

end sum_of_even_factors_900_l2083_208393


namespace geometric_sum_four_terms_l2083_208323

/-- 
Given that the sequence {a_n} is a geometric sequence with the sum of its 
first n terms denoted as S_n, if S_4=1 and S_8=4, prove that a_{13}+a_{14}+a_{15}+a_{16}=27 
-/ 
theorem geometric_sum_four_terms (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : ∀ (n : ℕ), S (n + 1) = a (n + 1) + S n) 
  (h2 : S 4 = 1) 
  (h3 : S 8 = 4) 
  : (a 13) + (a 14) + (a 15) + (a 16) = 27 := 
sorry

end geometric_sum_four_terms_l2083_208323


namespace girls_not_playing_soccer_l2083_208314

-- Define the given conditions
def students_total : Nat := 420
def boys_total : Nat := 312
def soccer_players_total : Nat := 250
def percent_boys_playing_soccer : Float := 0.78

-- Define the main goal based on the question and correct answer
theorem girls_not_playing_soccer : 
  students_total = 420 → 
  boys_total = 312 → 
  soccer_players_total = 250 → 
  percent_boys_playing_soccer = 0.78 → 
  ∃ (girls_not_playing_soccer : Nat), girls_not_playing_soccer = 53 :=
by 
  sorry

end girls_not_playing_soccer_l2083_208314


namespace equal_even_odd_probability_l2083_208348

theorem equal_even_odd_probability : 
  let total_dice := 8
  let even_odd_combinations := Nat.choose total_dice 4
  let single_arrangement_probability := (1 / 2) ^ total_dice
  even_odd_combinations * single_arrangement_probability = 35 / 128 := by
  sorry

end equal_even_odd_probability_l2083_208348


namespace value_of_m_l2083_208325

theorem value_of_m (m : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 - m * x + m - 1) 
  (h_eq : f 0 = f 2) : m = 2 :=
sorry

end value_of_m_l2083_208325


namespace empty_can_weight_l2083_208353

theorem empty_can_weight (W w : ℝ) :
  (W + 2 * w = 0.6) →
  (W + 5 * w = 0.975) →
  W = 0.35 :=
by sorry

end empty_can_weight_l2083_208353


namespace election_votes_l2083_208391

theorem election_votes (V : ℝ) 
  (h1 : 0.15 * V = 0.15 * V)
  (h2 : 0.85 * V = 309400 / 0.65)
  (h3 : 0.65 * (0.85 * V) = 309400) : 
  V = 560000 :=
by {
  sorry
}

end election_votes_l2083_208391


namespace inequality_proof_l2083_208349

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 
  b + (1 / a) > a + (1 / b) := 
by sorry

end inequality_proof_l2083_208349


namespace sum_of_roots_eq_seventeen_l2083_208354

theorem sum_of_roots_eq_seventeen : 
  ∀ (x : ℝ), (x - 8)^2 = 49 → x^2 - 16 * x + 15 = 0 → (∃ a b : ℝ, x = a ∨ x = b ∧ a + b = 16) := 
by sorry

end sum_of_roots_eq_seventeen_l2083_208354


namespace total_weight_proof_l2083_208365

-- Definitions of the conditions in the problem.
def bags_on_first_trip : ℕ := 10
def common_ratio : ℕ := 2
def number_of_trips : ℕ := 20
def weight_per_bag_kg : ℕ := 50

-- Function to compute the total number of bags transported.
noncomputable def total_number_of_bags : ℕ :=
  bags_on_first_trip * (1 - common_ratio^number_of_trips) / (1 - common_ratio)

-- Function to compute the total weight of onions harvested.
noncomputable def total_weight_of_onions : ℕ :=
  total_number_of_bags * weight_per_bag_kg

-- Theorem stating that the total weight of onions harvested is 524,287,500 kgs.
theorem total_weight_proof : total_weight_of_onions = 524287500 := by
  sorry

end total_weight_proof_l2083_208365


namespace average_percent_decrease_is_35_percent_l2083_208374

-- Given conditions
def last_week_small_price_per_pack := 7 / 3
def this_week_small_price_per_pack := 5 / 4
def last_week_large_price_per_pack := 8 / 2
def this_week_large_price_per_pack := 9 / 3

-- Calculate percent decrease for small packs
def small_pack_percent_decrease := ((last_week_small_price_per_pack - this_week_small_price_per_pack) / last_week_small_price_per_pack) * 100

-- Calculate percent decrease for large packs
def large_pack_percent_decrease := ((last_week_large_price_per_pack - this_week_large_price_per_pack) / last_week_large_price_per_pack) * 100

-- Calculate average percent decrease
def average_percent_decrease := (small_pack_percent_decrease + large_pack_percent_decrease) / 2

theorem average_percent_decrease_is_35_percent : average_percent_decrease = 35 := by
  sorry

end average_percent_decrease_is_35_percent_l2083_208374


namespace sum_common_seq_first_n_l2083_208373

def seq1 (n : ℕ) := 2 * n - 1
def seq2 (n : ℕ) := 3 * n - 2

def common_seq (n : ℕ) := 6 * n - 5

def sum_first_n_terms (a : ℕ) (d : ℕ) (n : ℕ) := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_common_seq_first_n (n : ℕ) : 
  sum_first_n_terms 1 6 n = 3 * n^2 - 2 * n := 
by sorry

end sum_common_seq_first_n_l2083_208373


namespace no_such_function_exists_l2083_208311

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), ∀ x : ℝ, f (Real.sin x) + f (Real.cos x) = Real.sin x :=
by
  sorry

end no_such_function_exists_l2083_208311


namespace total_percentage_change_l2083_208361

theorem total_percentage_change (X : ℝ) (fall_increase : X' = 1.08 * X) (spring_decrease : X'' = 0.8748 * X) :
  ((X'' - X) / X) * 100 = -12.52 := 
by
  sorry

end total_percentage_change_l2083_208361


namespace number_of_valid_subsets_l2083_208356

theorem number_of_valid_subsets (n : ℕ) :
  let total      := 16^n
  let invalid1   := 3 * 12^n
  let invalid2   := 2 * 10^n
  let invalidAll := 8^n
  let valid      := total - invalid1 + invalid2 + 9^n - invalidAll
  valid = 16^n - 3 * 12^n + 2 * 10^n + 9^n - 8^n :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_valid_subsets_l2083_208356


namespace managers_in_sample_l2083_208326

-- Definitions based on the conditions
def total_employees : ℕ := 160
def number_salespeople : ℕ := 104
def number_managers : ℕ := 32
def number_logistics : ℕ := 24
def sample_size : ℕ := 20

-- Theorem statement
theorem managers_in_sample : (number_managers * sample_size) / total_employees = 4 := by
  -- Proof omitted, as per the instructions
  sorry

end managers_in_sample_l2083_208326


namespace circle_area_l2083_208345

/-!

# Problem: Prove that the area of the circle defined by the equation \( x^2 + y^2 - 2x + 4y + 1 = 0 \) is \( 4\pi \).
-/

theorem circle_area : 
  (∃ x y : ℝ, x^2 + y^2 - 2*x + 4*y + 1 = 0) →
  ∃ (A : ℝ), A = 4 * Real.pi := 
by
  sorry

end circle_area_l2083_208345


namespace cats_remained_on_island_l2083_208351

theorem cats_remained_on_island : 
  ∀ (n m1 : ℕ), 
  n = 1800 → 
  m1 = 600 → 
  (n - m1) / 2 = 600 → 
  (n - m1) - ((n - m1) / 2) = 600 :=
by sorry

end cats_remained_on_island_l2083_208351


namespace exists_alpha_l2083_208389

variable {a : ℕ → ℝ}

axiom nonzero_sequence (n : ℕ) : a n ≠ 0
axiom recurrence_relation (n : ℕ) : a n ^ 2 - a (n - 1) * a (n + 1) = 1

theorem exists_alpha (n : ℕ) : ∃ α : ℝ, ∀ n ≥ 1, a (n + 1) = α * a n - a (n - 1) :=
by
  sorry

end exists_alpha_l2083_208389


namespace total_sales_correct_l2083_208357

-- Define the conditions
def total_tickets : ℕ := 65
def senior_ticket_price : ℕ := 10
def regular_ticket_price : ℕ := 15
def regular_tickets_sold : ℕ := 41

-- Calculate the senior citizen tickets sold
def senior_tickets_sold : ℕ := total_tickets - regular_tickets_sold

-- Calculate the revenue from senior citizen tickets
def revenue_senior : ℕ := senior_ticket_price * senior_tickets_sold

-- Calculate the revenue from regular tickets
def revenue_regular : ℕ := regular_ticket_price * regular_tickets_sold

-- Define the total sales amount
def total_sales_amount : ℕ := revenue_senior + revenue_regular

-- The statement we need to prove
theorem total_sales_correct : total_sales_amount = 855 := by
  sorry

end total_sales_correct_l2083_208357


namespace calc_fraction_cube_l2083_208383

theorem calc_fraction_cube : (88888 ^ 3 / 22222 ^ 3) = 64 := by 
    sorry

end calc_fraction_cube_l2083_208383


namespace mascot_toy_profit_l2083_208399

theorem mascot_toy_profit (x : ℝ) :
  (∀ (c : ℝ) (sales : ℝ), c = 40 → sales = 1000 - 10 * x → (x - c) * sales = 8000) →
  (x = 60 ∨ x = 80) :=
by
  intro h
  sorry

end mascot_toy_profit_l2083_208399


namespace contrapositive_equivalent_l2083_208318

variable {α : Type*} (A B : Set α) (x : α)

theorem contrapositive_equivalent : (x ∈ A → x ∈ B) ↔ (x ∉ B → x ∉ A) :=
by
  sorry

end contrapositive_equivalent_l2083_208318


namespace least_lcm_of_x_and_z_l2083_208350

theorem least_lcm_of_x_and_z (x y z : ℕ) (h₁ : Nat.lcm x y = 20) (h₂ : Nat.lcm y z = 28) : 
  ∃ l, l = Nat.lcm x z ∧ l = 35 := 
sorry

end least_lcm_of_x_and_z_l2083_208350


namespace cube_surface_area_l2083_208308

/-- A cube with an edge length of 10 cm has smaller cubes with edge length 2 cm 
    dug out from the middle of each face. The surface area of the new shape is 696 cm². -/
theorem cube_surface_area (original_edge : ℝ) (small_cube_edge : ℝ)
  (original_edge_eq : original_edge = 10) (small_cube_edge_eq : small_cube_edge = 2) :
  let original_surface := 6 * original_edge ^ 2
  let removed_area := 6 * small_cube_edge ^ 2
  let added_area := 6 * 5 * small_cube_edge ^ 2
  let new_surface := original_surface - removed_area + added_area
  new_surface = 696 := by
  sorry

end cube_surface_area_l2083_208308


namespace max_area_rectangle_l2083_208378

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l2083_208378


namespace share_of_a_l2083_208388

theorem share_of_a 
  (A B C : ℝ)
  (h1 : A = (2/3) * (B + C))
  (h2 : B = (2/3) * (A + C))
  (h3 : A + B + C = 200) :
  A = 60 :=
by {
  sorry
}

end share_of_a_l2083_208388
