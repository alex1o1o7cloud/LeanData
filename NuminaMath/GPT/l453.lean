import Mathlib

namespace NUMINAMATH_GPT_intersection_A_CRB_l453_45399

-- Definition of sets A and C_{R}B
def is_in_A (x: ℝ) := 0 < x ∧ x < 2

def is_in_CRB (x: ℝ) := x ≤ 1 ∨ x ≥ Real.exp 2

-- Proof that the intersection of A and C_{R}B is (0, 1]
theorem intersection_A_CRB : {x : ℝ | is_in_A x} ∩ {x : ℝ | is_in_CRB x} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_CRB_l453_45399


namespace NUMINAMATH_GPT_total_sheets_of_paper_l453_45384

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_sheets_of_paper_l453_45384


namespace NUMINAMATH_GPT_cos_alpha_value_l453_45343

open Real

theorem cos_alpha_value (α : ℝ) : 
  (sin (α - (π / 3)) = 1 / 5) ∧ (0 < α) ∧ (α < π / 2) → 
  (cos α = (2 * sqrt 6 - sqrt 3) / 10) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l453_45343


namespace NUMINAMATH_GPT_largest_operation_result_is_div_l453_45357

noncomputable def max_operation_result : ℚ :=
  max (max (-1 + (-1 / 2)) (-1 - (-1 / 2)))
      (max (-1 * (-1 / 2)) (-1 / (-1 / 2)))

theorem largest_operation_result_is_div :
  max_operation_result = 2 := by
  sorry

end NUMINAMATH_GPT_largest_operation_result_is_div_l453_45357


namespace NUMINAMATH_GPT_jennifer_money_left_l453_45339

theorem jennifer_money_left (initial_amount : ℕ) (sandwich_fraction museum_ticket_fraction book_fraction : ℚ) 
  (h_initial : initial_amount = 90) 
  (h_sandwich : sandwich_fraction = 1/5)
  (h_museum_ticket : museum_ticket_fraction = 1/6)
  (h_book : book_fraction = 1/2) : 
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_ticket_fraction + initial_amount * book_fraction) = 12 :=
by
  sorry

end NUMINAMATH_GPT_jennifer_money_left_l453_45339


namespace NUMINAMATH_GPT_double_exceeds_one_fifth_by_nine_l453_45367

theorem double_exceeds_one_fifth_by_nine (x : ℝ) (h : 2 * x = (1 / 5) * x + 9) : x^2 = 25 :=
sorry

end NUMINAMATH_GPT_double_exceeds_one_fifth_by_nine_l453_45367


namespace NUMINAMATH_GPT_meal_combinations_correct_l453_45306

-- Let E denote the total number of dishes on the menu
def E : ℕ := 12

-- Let V denote the number of vegetarian dishes on the menu
def V : ℕ := 5

-- Define the function that computes the number of different combinations of meals Elena and Nasir can order
def meal_combinations (e : ℕ) (v : ℕ) : ℕ :=
  e * v

-- The theorem to prove that the number of different combinations of meals Elena and Nasir can order is 60
theorem meal_combinations_correct : meal_combinations E V = 60 := by
  sorry

end NUMINAMATH_GPT_meal_combinations_correct_l453_45306


namespace NUMINAMATH_GPT_find_cos_C_l453_45315

noncomputable def cos_C_eq (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) : Prop :=
  Real.cos C = 7 / 25

theorem find_cos_C (A B C a b c : ℝ) (h1 : 8 * b = 5 * c) (h2 : C = 2 * B) :
  cos_C_eq A B C a b c h1 h2 :=
sorry

end NUMINAMATH_GPT_find_cos_C_l453_45315


namespace NUMINAMATH_GPT_cabinets_and_perimeter_l453_45330

theorem cabinets_and_perimeter :
  ∀ (original_cabinets : ℕ) (install_factor : ℕ) (num_counters : ℕ) 
    (cabinets_L_1 cabinets_L_2 cabinets_L_3 removed_cabinets cabinet_height total_cabinets perimeter : ℕ),
    original_cabinets = 3 →
    install_factor = 2 →
    num_counters = 4 →
    cabinets_L_1 = 3 →
    cabinets_L_2 = 5 →
    cabinets_L_3 = 7 →
    removed_cabinets = 2 →
    cabinet_height = 2 →
    total_cabinets = (original_cabinets * install_factor * num_counters) + 
                     (cabinets_L_1 + cabinets_L_2 + cabinets_L_3) - removed_cabinets →
    perimeter = (cabinets_L_1 * cabinet_height) +
                (cabinets_L_3 * cabinet_height) +
                2 * (cabinets_L_2 * cabinet_height) →
    total_cabinets = 37 ∧
    perimeter = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cabinets_and_perimeter_l453_45330


namespace NUMINAMATH_GPT_standard_equation_line_BC_fixed_point_l453_45358

section EllipseProof

open Real

-- Define the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions from the problem
axiom a_gt_b_gt_0 : ∀ (a b : ℝ), a > b → b > 0
axiom passes_through_point : ∀ (a b x y : ℝ), ellipse a b x y → (x = 1 ∧ y = sqrt 2 / 2)
axiom has_eccentricity : ∀ (a b c : ℝ), c / a = sqrt 2 / 2 → c^2 = a^2 - b^2 → b = 1

-- The standard equation of the ellipse
theorem standard_equation (a b : ℝ) (x y : ℝ) :
  a = sqrt 2 → b = 1 → ellipse a b x y → ellipse (sqrt 2) 1 x y :=
sorry

-- Prove that BC always passes through a fixed point
theorem line_BC_fixed_point (a b x1 x2 y1 y2 : ℝ) :
  a = sqrt 2 → b = 1 → 
  ellipse a b x1 y1 → ellipse a b x2 y2 →
  y1 = -y2 → x1 ≠ x2 → (-1, 0) = (-1, 0) →
  ∃ (k : ℝ) (x : ℝ), x = -2 ∧ y = 0 :=
sorry

end EllipseProof

end NUMINAMATH_GPT_standard_equation_line_BC_fixed_point_l453_45358


namespace NUMINAMATH_GPT_investment_of_q_is_correct_l453_45396

-- Define investments and the profit ratio
def p_investment : ℝ := 30000
def profit_ratio_p : ℝ := 2
def profit_ratio_q : ℝ := 3

-- Define q's investment as x
def q_investment : ℝ := 45000

-- The goal is to prove that q_investment is indeed 45000 given the above conditions
theorem investment_of_q_is_correct :
  (p_investment / q_investment) = (profit_ratio_p / profit_ratio_q) :=
sorry

end NUMINAMATH_GPT_investment_of_q_is_correct_l453_45396


namespace NUMINAMATH_GPT_train_passes_jogger_time_l453_45394

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 75
noncomputable def jogger_head_start_m : ℝ := 500
noncomputable def train_length_m : ℝ := 300

noncomputable def km_per_hr_to_m_per_s (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def jogger_speed_m_per_s := km_per_hr_to_m_per_s jogger_speed_km_per_hr
noncomputable def train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr

noncomputable def relative_speed_m_per_s := train_speed_m_per_s - jogger_speed_m_per_s

noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m

theorem train_passes_jogger_time :
  let time_to_pass := total_distance_to_cover_m / relative_speed_m_per_s
  abs (time_to_pass - 43.64) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_train_passes_jogger_time_l453_45394


namespace NUMINAMATH_GPT_part_a_l453_45305

theorem part_a (x y : ℝ) (hx : x ≠ 1) (hy : y ≠ 1) (hxy : x * y ≠ 1) :
  (x * y) / (1 - x * y) = x / (1 - x) + y / (1 - y) :=
sorry

end NUMINAMATH_GPT_part_a_l453_45305


namespace NUMINAMATH_GPT_total_weight_of_rings_l453_45329

-- Conditions
def weight_orange : ℝ := 0.08333333333333333
def weight_purple : ℝ := 0.3333333333333333
def weight_white : ℝ := 0.4166666666666667

-- Goal
theorem total_weight_of_rings : weight_orange + weight_purple + weight_white = 0.8333333333333333 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_rings_l453_45329


namespace NUMINAMATH_GPT_intersection_is_line_l453_45383

-- Define the two planes as given in the conditions
def plane1 (x y z : ℝ) : Prop := x + 5 * y + 2 * z - 5 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x - 5 * y - z + 5 = 0

-- The intersection of the planes should satisfy both plane equations
def is_on_line (x y z : ℝ) : Prop := plane1 x y z ∧ plane2 x y z

-- Define the canonical equation of the line
def line_eq (x y z : ℝ) : Prop := (∃ k : ℝ, x = 5 * k ∧ y = 5 * k + 1 ∧ z = -15 * k)

-- The proof statement
theorem intersection_is_line :
  (∀ x y z : ℝ, is_on_line x y z → line_eq x y z) ∧ 
  (∀ x y z : ℝ, line_eq x y z → is_on_line x y z) :=
by
  sorry

end NUMINAMATH_GPT_intersection_is_line_l453_45383


namespace NUMINAMATH_GPT_proof_f_g_f3_l453_45317

def f (x: ℤ) : ℤ := 2*x + 5
def g (x: ℤ) : ℤ := 5*x + 2

theorem proof_f_g_f3 :
  f (g (f 3)) = 119 := by
  sorry

end NUMINAMATH_GPT_proof_f_g_f3_l453_45317


namespace NUMINAMATH_GPT_simplify_expression_l453_45336

variable (a b : ℝ)

theorem simplify_expression :
  (-2 * a^2 * b^3) * (-a * b^2)^2 + (- (1 / 2) * a^2 * b^3)^2 * 4 * b = -a^4 * b^7 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l453_45336


namespace NUMINAMATH_GPT_line_intersects_extension_of_segment_l453_45368

theorem line_intersects_extension_of_segment
  (A B C x1 y1 x2 y2 : ℝ)
  (hnz : A ≠ 0 ∨ B ≠ 0)
  (h1 : (A * x1 + B * y1 + C) * (A * x2 + B * y2 + C) > 0)
  (h2 : |A * x1 + B * y1 + C| > |A * x2 + B * y2 + C|) :
  ∃ t : ℝ, t ≥ 0 ∧ l * (t * (x2 - x1) + x1) + m * (t * (y2 - y1) + y1) = 0 :=
sorry

end NUMINAMATH_GPT_line_intersects_extension_of_segment_l453_45368


namespace NUMINAMATH_GPT_number_eq_1925_l453_45325

theorem number_eq_1925 (x : ℝ) (h : x / 7 - x / 11 = 100) : x = 1925 :=
sorry

end NUMINAMATH_GPT_number_eq_1925_l453_45325


namespace NUMINAMATH_GPT_autumn_sales_l453_45334

theorem autumn_sales (T : ℝ) (spring summer winter autumn : ℝ) 
    (h1 : spring = 3)
    (h2 : summer = 6)
    (h3 : winter = 5)
    (h4 : T = (3 / 0.2)) :
    autumn = 1 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_autumn_sales_l453_45334


namespace NUMINAMATH_GPT_roots_cubic_identity_l453_45302

theorem roots_cubic_identity (p q r s : ℝ) (h1 : r + s = p) (h2 : r * s = -q) (h3 : ∀ x : ℝ, x^2 - p*x - q = 0 → (x = r ∨ x = s)) :
  r^3 + s^3 = p^3 + 3*p*q := by
  sorry

end NUMINAMATH_GPT_roots_cubic_identity_l453_45302


namespace NUMINAMATH_GPT_pet_food_cost_is_correct_l453_45301

-- Define the given conditions
def rabbit_toy_cost := 6.51
def cage_cost := 12.51
def total_cost := 24.81
def found_dollar := 1.00

-- Define the cost of pet food
def pet_food_cost := total_cost - (rabbit_toy_cost + cage_cost) + found_dollar

-- The statement to prove
theorem pet_food_cost_is_correct : pet_food_cost = 6.79 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_pet_food_cost_is_correct_l453_45301


namespace NUMINAMATH_GPT_range_of_x_l453_45376

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 * x else 2 * -x

theorem range_of_x {x : ℝ} :
  f (1 - 2 * x) < f 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l453_45376


namespace NUMINAMATH_GPT_compartments_count_l453_45398

-- Definition of initial pennies per compartment
def initial_pennies_per_compartment : ℕ := 2

-- Definition of additional pennies added to each compartment
def additional_pennies_per_compartment : ℕ := 6

-- Definition of total pennies is 96
def total_pennies : ℕ := 96

-- Prove the number of compartments is 12
theorem compartments_count (c : ℕ) 
  (h1 : initial_pennies_per_compartment + additional_pennies_per_compartment = 8)
  (h2 : 8 * c = total_pennies) : 
  c = 12 :=
by
  sorry

end NUMINAMATH_GPT_compartments_count_l453_45398


namespace NUMINAMATH_GPT_total_widgets_sold_after_20_days_l453_45355

-- Definition of the arithmetic sequence
def widgets_sold_on_day (n : ℕ) : ℕ :=
  2 * n - 1

-- Sum of the first n terms of the sequence
def sum_of_widgets_sold (n : ℕ) : ℕ :=
  n * (widgets_sold_on_day 1 + widgets_sold_on_day n) / 2

-- Prove that the total widgets sold after 20 days is 400
theorem total_widgets_sold_after_20_days : sum_of_widgets_sold 20 = 400 :=
by
  sorry

end NUMINAMATH_GPT_total_widgets_sold_after_20_days_l453_45355


namespace NUMINAMATH_GPT_find_b_l453_45340

theorem find_b (b : ℚ) : (∃ x y : ℚ, x = 3 ∧ y = -5 ∧ (b * x - (b + 2) * y = b - 3)) → b = -13 / 7 :=
sorry

end NUMINAMATH_GPT_find_b_l453_45340


namespace NUMINAMATH_GPT_each_player_plays_36_minutes_l453_45313

-- Definitions based on the conditions
def players := 10
def players_on_field := 8
def match_duration := 45
def equal_play_time (t : Nat) := 360 / players = t

-- Theorem: Each player plays for 36 minutes
theorem each_player_plays_36_minutes : ∃ t, equal_play_time t ∧ t = 36 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_each_player_plays_36_minutes_l453_45313


namespace NUMINAMATH_GPT_rice_in_each_container_l453_45349

-- Given conditions from the problem
def total_weight_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- A theorem that each container has 25 ounces of rice given the conditions
theorem rice_in_each_container (h : total_weight_pounds * pounds_to_ounces / num_containers = 25) : True :=
  sorry

end NUMINAMATH_GPT_rice_in_each_container_l453_45349


namespace NUMINAMATH_GPT_eq_3_solutions_l453_45307

theorem eq_3_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∃! (x y : ℕ), (0 < x) ∧ (0 < y) ∧ ((1 / x) + (1 / y) = (1 / p)) ∧
  ((x = p + 1 ∧ y = p^2 + p) ∨ (x = p + p ∧ y = p + p) ∨ (x = p^2 + p ∧ y = p + 1)) :=
sorry

end NUMINAMATH_GPT_eq_3_solutions_l453_45307


namespace NUMINAMATH_GPT_calculate_a3_l453_45392

theorem calculate_a3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S n = 2^n - 1) (h2 : ∀ n, a n = S n - S (n-1)) : 
  a 3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_a3_l453_45392


namespace NUMINAMATH_GPT_find_integer_solutions_l453_45354

theorem find_integer_solutions :
  {n : ℤ | n + 2 ∣ n^2 + 3} = {-9, -3, -1, 5} :=
  sorry

end NUMINAMATH_GPT_find_integer_solutions_l453_45354


namespace NUMINAMATH_GPT_sum_all_products_eq_l453_45388

def group1 : List ℚ := [3/4, 3/20] -- Using 0.15 as 3/20 to work with rationals
def group2 : List ℚ := [4, 2/3]
def group3 : List ℚ := [3/5, 6/5] -- Using 1.2 as 6/5 to work with rationals

def allProducts (a b c : List ℚ) : List ℚ :=
  List.bind a (fun x =>
  List.bind b (fun y =>
  List.map (fun z => x * y * z) c))

theorem sum_all_products_eq :
  (allProducts group1 group2 group3).sum = 7.56 := by
  sorry

end NUMINAMATH_GPT_sum_all_products_eq_l453_45388


namespace NUMINAMATH_GPT_value_of_expression_l453_45389

theorem value_of_expression (x : ℝ) (h : (3 / (x - 3)) + (5 / (2 * x - 6)) = 11 / 2) : 2 * x - 6 = 2 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l453_45389


namespace NUMINAMATH_GPT_find_number_l453_45393

theorem find_number (N : ℤ) (h1 : ∃ k : ℤ, N - 3 = 5 * k) (h2 : ∃ l : ℤ, N - 2 = 7 * l) (h3 : 50 < N ∧ N < 70) : N = 58 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l453_45393


namespace NUMINAMATH_GPT_tank_inflow_rate_l453_45342

/-- 
  Tanks A and B have the same capacity of 20 liters. Tank A has
  an inflow rate of 2 liters per hour and takes 5 hours longer to
  fill than tank B. Show that the inflow rate in tank B is 4 liters 
  per hour.
-/
theorem tank_inflow_rate (capacity : ℕ) (rate_A : ℕ) (extra_time : ℕ) (rate_B : ℕ) 
  (h1 : capacity = 20) (h2 : rate_A = 2) (h3 : extra_time = 5) (h4 : capacity / rate_A = (capacity / rate_B) + extra_time) :
  rate_B = 4 :=
sorry

end NUMINAMATH_GPT_tank_inflow_rate_l453_45342


namespace NUMINAMATH_GPT_total_money_spent_l453_45371

-- Definitions based on conditions
def num_bars_of_soap : Nat := 20
def weight_per_bar_of_soap : Float := 1.5
def cost_per_pound_of_soap : Float := 0.5

def num_bottles_of_shampoo : Nat := 15
def weight_per_bottle_of_shampoo : Float := 2.2
def cost_per_pound_of_shampoo : Float := 0.8

-- The theorem to prove
theorem total_money_spent :
  let cost_per_bar_of_soap := weight_per_bar_of_soap * cost_per_pound_of_soap
  let total_cost_of_soap := Float.ofNat num_bars_of_soap * cost_per_bar_of_soap
  let cost_per_bottle_of_shampoo := weight_per_bottle_of_shampoo * cost_per_pound_of_shampoo
  let total_cost_of_shampoo := Float.ofNat num_bottles_of_shampoo * cost_per_bottle_of_shampoo
  total_cost_of_soap + total_cost_of_shampoo = 41.40 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_money_spent_l453_45371


namespace NUMINAMATH_GPT_numBills_is_9_l453_45300

-- Define the conditions: Mike has 45 dollars in 5-dollar bills
def totalDollars : ℕ := 45
def billValue : ℕ := 5
def numBills : ℕ := 9

-- Prove that the number of 5-dollar bills Mike has is 9
theorem numBills_is_9 : (totalDollars = billValue * numBills) → (numBills = 9) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_numBills_is_9_l453_45300


namespace NUMINAMATH_GPT_inequality_l453_45377

theorem inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a / (b.sqrt) + b / (a.sqrt)) ≥ (a.sqrt + b.sqrt) :=
by
  sorry

end NUMINAMATH_GPT_inequality_l453_45377


namespace NUMINAMATH_GPT_three_children_meet_l453_45326

theorem three_children_meet 
  (children : Finset ℕ)
  (visited_times : ℕ → ℕ)
  (meet_at_stand : ℕ → ℕ → Prop)
  (h_children_count : children.card = 7)
  (h_visited_times : ∀ c ∈ children, visited_times c = 3)
  (h_meet_pairwise : ∀ (c1 c2 : ℕ), c1 ∈ children → c2 ∈ children → c1 ≠ c2 → meet_at_stand c1 c2) :
  ∃ (t : ℕ), ∃ (c1 c2 c3 : ℕ), c1 ≠ c2 ∧ c2 ≠ c3 ∧ c1 ≠ c3 ∧ 
  c1 ∈ children ∧ c2 ∈ children ∧ c3 ∈ children ∧ 
  meet_at_stand c1 t ∧ meet_at_stand c2 t ∧ meet_at_stand c3 t := 
sorry

end NUMINAMATH_GPT_three_children_meet_l453_45326


namespace NUMINAMATH_GPT_sam_cleaner_meetings_two_times_l453_45321

open Nat

noncomputable def sam_and_cleaner_meetings (sam_rate cleaner_rate cleaner_stop_time bench_distance : ℕ) : ℕ :=
  let cycle_time := (bench_distance / cleaner_rate) + cleaner_stop_time
  let distance_covered_in_cycle_sam := sam_rate * cycle_time
  let distance_covered_in_cycle_cleaner := bench_distance
  let effective_distance_reduction := distance_covered_in_cycle_cleaner - distance_covered_in_cycle_sam
  let number_of_cycles_until_meeting := bench_distance / effective_distance_reduction
  number_of_cycles_until_meeting + 1

theorem sam_cleaner_meetings_two_times :
  sam_and_cleaner_meetings 3 9 40 300 = 2 :=
by sorry

end NUMINAMATH_GPT_sam_cleaner_meetings_two_times_l453_45321


namespace NUMINAMATH_GPT_at_least_one_root_l453_45344

theorem at_least_one_root 
  (a b c d : ℝ)
  (h : a * c = 2 * b + 2 * d) :
  (∃ x : ℝ, x^2 + a * x + b = 0) ∨ (∃ x : ℝ, x^2 + c * x + d = 0) :=
sorry

end NUMINAMATH_GPT_at_least_one_root_l453_45344


namespace NUMINAMATH_GPT_percentage_error_in_side_measurement_l453_45374

theorem percentage_error_in_side_measurement :
  (forall (S S' : ℝ) (A A' : ℝ), 
    A = S^2 ∧ A' = S'^2 ∧ (A' - A) / A * 100 = 25.44 -> 
    (S' - S) / S * 100 = 12.72) :=
by
  intros S S' A A' h
  sorry

end NUMINAMATH_GPT_percentage_error_in_side_measurement_l453_45374


namespace NUMINAMATH_GPT_expected_value_correct_l453_45360

-- Define the probabilities
def prob_8 : ℚ := 3 / 8
def prob_other : ℚ := 5 / 56 -- Derived from the solution steps but using only given conditions explicitly.

-- Define the expected value calculation
def expected_value_die : ℚ :=
  (1 * prob_other) + (2 * prob_other) + (3 * prob_other) + (4 * prob_other) +
  (5 * prob_other) + (6 * prob_other) + (7 * prob_other) + (8 * prob_8)

-- The theorem to prove
theorem expected_value_correct : expected_value_die = 77 / 14 := by
  sorry

end NUMINAMATH_GPT_expected_value_correct_l453_45360


namespace NUMINAMATH_GPT_initial_oranges_per_rupee_l453_45390

theorem initial_oranges_per_rupee (loss_rate_gain_rate cost_rate : ℝ) (initial_oranges : ℤ) : 
  loss_rate_gain_rate = 0.92 ∧ cost_rate = 18.4 ∧ 1.25 * cost_rate = 1.25 * 0.92 * (initial_oranges : ℝ) →
  initial_oranges = 14 := by
  sorry

end NUMINAMATH_GPT_initial_oranges_per_rupee_l453_45390


namespace NUMINAMATH_GPT_subset_M_N_l453_45320

def is_element_of_M (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 4) + (Real.pi / 4)

def is_element_of_N (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (k * Real.pi / 8) - (Real.pi / 4)

theorem subset_M_N : ∀ x, is_element_of_M x → is_element_of_N x :=
by
  sorry

end NUMINAMATH_GPT_subset_M_N_l453_45320


namespace NUMINAMATH_GPT_find_value_l453_45352

variables {p q s u : ℚ}

theorem find_value
  (h1 : p / q = 5 / 6)
  (h2 : s / u = 7 / 15) :
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 :=
sorry

end NUMINAMATH_GPT_find_value_l453_45352


namespace NUMINAMATH_GPT_initial_bees_l453_45351

variable (B : ℕ)

theorem initial_bees (h : B + 10 = 26) : B = 16 :=
by sorry

end NUMINAMATH_GPT_initial_bees_l453_45351


namespace NUMINAMATH_GPT_correct_option_l453_45378

theorem correct_option :
  (3 * a^2 + 5 * a^2 ≠ 8 * a^4) ∧
  (5 * a^2 * b - 6 * a * b^2 ≠ -a * b^2) ∧
  (2 * x + 3 * y ≠ 5 * x * y) ∧
  (9 * x * y - 6 * x * y = 3 * x * y) :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l453_45378


namespace NUMINAMATH_GPT_autograph_value_after_changes_l453_45365

def initial_value : ℝ := 100
def drop_percent : ℝ := 0.30
def increase_percent : ℝ := 0.40

theorem autograph_value_after_changes :
  let value_after_drop := initial_value * (1 - drop_percent)
  let value_after_increase := value_after_drop * (1 + increase_percent)
  value_after_increase = 98 :=
by
  sorry

end NUMINAMATH_GPT_autograph_value_after_changes_l453_45365


namespace NUMINAMATH_GPT_tire_price_l453_45304

theorem tire_price (x : ℝ) (h : 3 * x + 10 = 310) : x = 100 :=
sorry

end NUMINAMATH_GPT_tire_price_l453_45304


namespace NUMINAMATH_GPT_effective_speed_against_current_l453_45366

theorem effective_speed_against_current
  (speed_with_current : ℝ)
  (speed_of_current : ℝ)
  (headwind_speed : ℝ)
  (obstacle_reduction_pct : ℝ)
  (h_speed_with_current : speed_with_current = 25)
  (h_speed_of_current : speed_of_current = 4)
  (h_headwind_speed : headwind_speed = 2)
  (h_obstacle_reduction_pct : obstacle_reduction_pct = 0.15) :
  let speed_in_still_water := speed_with_current - speed_of_current
  let speed_against_current_headwind := speed_in_still_water - speed_of_current - headwind_speed
  let reduction_due_to_obstacles := obstacle_reduction_pct * speed_against_current_headwind
  let effective_speed := speed_against_current_headwind - reduction_due_to_obstacles
  effective_speed = 12.75 := by
{
  sorry
}

end NUMINAMATH_GPT_effective_speed_against_current_l453_45366


namespace NUMINAMATH_GPT_remainder_when_subtract_div_by_6_l453_45362

theorem remainder_when_subtract_div_by_6 (m n : ℕ) (h1 : m % 6 = 2) (h2 : n % 6 = 3) (h3 : m > n) : (m - n) % 6 = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_when_subtract_div_by_6_l453_45362


namespace NUMINAMATH_GPT_find_triangle_value_l453_45387

theorem find_triangle_value 
  (triangle : ℕ)
  (h_units : (triangle + 3) % 7 = 2)
  (h_tens : (1 + 4 + triangle) % 7 = 4)
  (h_hundreds : (2 + triangle + 1) % 7 = 2)
  (h_thousands : 3 + 0 + 1 = 4) :
  triangle = 6 :=
sorry

end NUMINAMATH_GPT_find_triangle_value_l453_45387


namespace NUMINAMATH_GPT_smallest_possible_a_l453_45382

theorem smallest_possible_a (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c) (h4 : a^2 = c * b) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_a_l453_45382


namespace NUMINAMATH_GPT_xiao_hong_mistake_l453_45323

theorem xiao_hong_mistake (a : ℕ) (h : 31 - a = 12) : 31 + a = 50 :=
by
  sorry

end NUMINAMATH_GPT_xiao_hong_mistake_l453_45323


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l453_45391

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x = Real.sqrt (a + 2)) ↔ a ≥ -2 := 
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l453_45391


namespace NUMINAMATH_GPT_not_divisible_by_n_l453_45311

theorem not_divisible_by_n (n : ℕ) (h : n > 1) : ¬n ∣ 2^n - 1 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_not_divisible_by_n_l453_45311


namespace NUMINAMATH_GPT_new_container_volume_l453_45375

theorem new_container_volume (original_volume : ℕ) (factor : ℕ) (new_volume : ℕ) 
    (h1 : original_volume = 5) (h2 : factor = 4 * 4 * 4) : new_volume = 320 :=
by
  sorry

end NUMINAMATH_GPT_new_container_volume_l453_45375


namespace NUMINAMATH_GPT_JimAgeInXYears_l453_45346

-- Definitions based on conditions
def TomCurrentAge := 37
def JimsAge7YearsAgo := 5 + (TomCurrentAge - 7) / 2

-- We introduce a variable X to represent the number of years into the future.
variable (X : ℕ)

-- Lean 4 statement to prove that Jim will be 27 + X years old in X years from now.
theorem JimAgeInXYears : JimsAge7YearsAgo + 7 + X = 27 + X := 
by
  sorry

end NUMINAMATH_GPT_JimAgeInXYears_l453_45346


namespace NUMINAMATH_GPT_valentines_count_l453_45361

theorem valentines_count (x y : ℕ) (h1 : (x = 2 ∧ y = 48) ∨ (x = 48 ∧ y = 2)) : 
  x * y - (x + y) = 46 := by
  sorry

end NUMINAMATH_GPT_valentines_count_l453_45361


namespace NUMINAMATH_GPT_find_second_half_profit_l453_45381

variable (P : ℝ)
variable (profit_difference total_annual_profit : ℝ)
variable (h_difference : profit_difference = 2750000)
variable (h_total : total_annual_profit = 3635000)

theorem find_second_half_profit (h_eq : P + (P + profit_difference) = total_annual_profit) : 
  P = 442500 :=
by
  rw [h_difference, h_total] at h_eq
  sorry

end NUMINAMATH_GPT_find_second_half_profit_l453_45381


namespace NUMINAMATH_GPT_f_value_l453_45319

def B := {x : ℚ | x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2}

def f (x : ℚ) : ℝ := sorry

axiom f_property : ∀ x ∈ B, f x + f (2 - (1 / x)) = Real.log (abs (x ^ 2))

theorem f_value : f 2023 = Real.log 2023 :=
by
  sorry

end NUMINAMATH_GPT_f_value_l453_45319


namespace NUMINAMATH_GPT_problem_statement_l453_45372

def S : ℤ := (-2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19)

theorem problem_statement (hS : S = -2^20 + 4) : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19 + 2^20 = 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l453_45372


namespace NUMINAMATH_GPT_latest_leave_time_correct_l453_45369

-- Define the conditions
def flight_time := 20 -- 8:00 pm in 24-hour format
def check_in_early := 2 -- 2 hours early
def drive_time := 45 -- 45 minutes
def park_time := 15 -- 15 minutes

-- Define the target time to be at the airport
def at_airport_time := flight_time - check_in_early -- 18:00 or 6:00 pm

-- Total travel time required (minutes)
def total_travel_time := drive_time + park_time -- 60 minutes

-- Convert total travel time to hours
def travel_time_in_hours : ℕ := total_travel_time / 60

-- Define the latest time to leave the house
def latest_leave_time := at_airport_time - travel_time_in_hours

-- Theorem to state the equivalence of the latest time they can leave their house
theorem latest_leave_time_correct : latest_leave_time = 17 :=
    by
    sorry

end NUMINAMATH_GPT_latest_leave_time_correct_l453_45369


namespace NUMINAMATH_GPT_peach_ratios_and_percentages_l453_45356

def red_peaches : ℕ := 8
def yellow_peaches : ℕ := 14
def green_peaches : ℕ := 6
def orange_peaches : ℕ := 4
def total_peaches : ℕ := red_peaches + yellow_peaches + green_peaches + orange_peaches

theorem peach_ratios_and_percentages :
  ((green_peaches : ℚ) / total_peaches = 3 / 16) ∧
  ((green_peaches : ℚ) / total_peaches * 100 = 18.75) ∧
  ((yellow_peaches : ℚ) / total_peaches = 7 / 16) ∧
  ((yellow_peaches : ℚ) / total_peaches * 100 = 43.75) :=
by {
  sorry
}

end NUMINAMATH_GPT_peach_ratios_and_percentages_l453_45356


namespace NUMINAMATH_GPT_james_ride_time_l453_45395

theorem james_ride_time (distance speed : ℝ) (h_distance : distance = 200) (h_speed : speed = 25) : distance / speed = 8 :=
by
  rw [h_distance, h_speed]
  norm_num

end NUMINAMATH_GPT_james_ride_time_l453_45395


namespace NUMINAMATH_GPT_area_of_parallelogram_l453_45364

theorem area_of_parallelogram (base : ℝ) (height : ℝ)
  (h1 : base = 3.6)
  (h2 : height = 2.5 * base) :
  base * height = 32.4 :=
by
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l453_45364


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l453_45353

def M := {x : ℝ | -1 < x ∧ x < 3}
def N := {x : ℝ | x < 1}

theorem intersection_of_M_and_N : (M ∩ N = {x : ℝ | -1 < x ∧ x < 1}) :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l453_45353


namespace NUMINAMATH_GPT_farmer_ploughing_problem_l453_45350

theorem farmer_ploughing_problem (A D : ℕ) (h1 : A = 120 * D) (h2 : A - 40 = 85 * (D + 2)) : 
  A = 720 ∧ D = 6 :=
by
  sorry

end NUMINAMATH_GPT_farmer_ploughing_problem_l453_45350


namespace NUMINAMATH_GPT_find_a_l453_45335

theorem find_a 
  (x y z a : ℤ)
  (h1 : z + a = -2)
  (h2 : y + z = 1)
  (h3 : x + y = 0) : 
  a = -2 := 
  by 
    sorry

end NUMINAMATH_GPT_find_a_l453_45335


namespace NUMINAMATH_GPT_min_degree_g_l453_45380

theorem min_degree_g (f g h : Polynomial ℝ) (hf : f.degree = 8) (hh : h.degree = 9) (h_eq : 3 * f + 4 * g = h) : g.degree ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_degree_g_l453_45380


namespace NUMINAMATH_GPT_values_of_n_l453_45347

theorem values_of_n (n : ℕ) : ∃ (m : ℕ), n^2 = 9 + 7 * m ∧ n % 7 = 3 := 
sorry

end NUMINAMATH_GPT_values_of_n_l453_45347


namespace NUMINAMATH_GPT_months_for_three_times_collection_l453_45348

def Kymbrea_collection (n : ℕ) : ℕ := 40 + 3 * n
def LaShawn_collection (n : ℕ) : ℕ := 20 + 5 * n

theorem months_for_three_times_collection : ∃ n : ℕ, LaShawn_collection n = 3 * Kymbrea_collection n ∧ n = 25 := 
by
  sorry

end NUMINAMATH_GPT_months_for_three_times_collection_l453_45348


namespace NUMINAMATH_GPT_stability_of_scores_requires_variance_l453_45332

-- Define the conditions
variable (scores : List ℝ)

-- Define the main theorem
theorem stability_of_scores_requires_variance : True :=
  sorry

end NUMINAMATH_GPT_stability_of_scores_requires_variance_l453_45332


namespace NUMINAMATH_GPT_chord_length_of_tangent_l453_45318

theorem chord_length_of_tangent (R r : ℝ) (h : R^2 - r^2 = 25) : ∃ c : ℝ, c = 10 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_of_tangent_l453_45318


namespace NUMINAMATH_GPT_speed_of_each_train_l453_45386

theorem speed_of_each_train (v : ℝ) (train_length time_cross : ℝ) (km_pr_s : ℝ) 
  (h_train_length : train_length = 120)
  (h_time_cross : time_cross = 8)
  (h_km_pr_s : km_pr_s = 3.6)
  (h_relative_speed : 2 * v = (2 * train_length) / time_cross) :
  v * km_pr_s = 54 := 
by sorry

end NUMINAMATH_GPT_speed_of_each_train_l453_45386


namespace NUMINAMATH_GPT_property_value_at_beginning_l453_45333

theorem property_value_at_beginning 
  (r : ℝ) (v3 : ℝ) (V : ℝ) (rate : ℝ) (years : ℕ) 
  (h_rate : rate = 6.25 / 100) 
  (h_years : years = 3) 
  (h_v3 : v3 = 21093) 
  (h_r : r = 1 - rate) 
  (h_V : V * r ^ years = v3) 
  : V = 25656.25 :=
by
  sorry

end NUMINAMATH_GPT_property_value_at_beginning_l453_45333


namespace NUMINAMATH_GPT_original_purchase_price_l453_45324

-- Define the conditions and question
theorem original_purchase_price (P S : ℝ) (h1 : S = P + 0.25 * S) (h2 : 16 = 0.80 * S - P) : P = 240 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_original_purchase_price_l453_45324


namespace NUMINAMATH_GPT_trapezium_other_side_l453_45370

theorem trapezium_other_side (x : ℝ) :
  1/2 * (20 + x) * 10 = 150 → x = 10 :=
by
  sorry

end NUMINAMATH_GPT_trapezium_other_side_l453_45370


namespace NUMINAMATH_GPT_gcd_lcm_sum_ge_sum_l453_45341

theorem gcd_lcm_sum_ge_sum (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  Nat.gcd a b + Nat.lcm a b ≥ a + b := 
sorry

end NUMINAMATH_GPT_gcd_lcm_sum_ge_sum_l453_45341


namespace NUMINAMATH_GPT_minimum_value_expression_l453_45312

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) :
  (∃ x : ℝ, x = 9 ∧ ∀ y : ℝ, y = (1 / a^2 - 1) * (1 / b^2 - 1) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l453_45312


namespace NUMINAMATH_GPT_area_square_II_is_6a_squared_l453_45359

-- Problem statement:
-- Given the diagonal of square I is 2a and the area of square II is three times the area of square I,
-- prove that the area of square II is 6a^2

noncomputable def area_square_II (a : ℝ) : ℝ :=
  let side_I := (2 * a) / Real.sqrt 2
  let area_I := side_I ^ 2
  3 * area_I

theorem area_square_II_is_6a_squared (a : ℝ) : area_square_II a = 6 * a ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_area_square_II_is_6a_squared_l453_45359


namespace NUMINAMATH_GPT_rectangle_area_l453_45309

theorem rectangle_area (x : ℕ) (hx : x > 0)
  (h₁ : (x + 5) * 2 * (x + 10) = 3 * x * (x + 10))
  (h₂ : (x - 10) = x + 10 - 10) :
  x * (x + 10) = 200 :=
by {
  sorry
}

end NUMINAMATH_GPT_rectangle_area_l453_45309


namespace NUMINAMATH_GPT_avg_salary_supervisors_l453_45363

-- Definitions based on the conditions of the problem
def total_workers : Nat := 48
def supervisors : Nat := 6
def laborers : Nat := 42
def avg_salary_total : Real := 1250
def avg_salary_laborers : Real := 950

-- Given the above conditions, we need to prove the average salary of the supervisors.
theorem avg_salary_supervisors :
  (supervisors * (supervisors * total_workers * avg_salary_total - laborers * avg_salary_laborers) / supervisors) = 3350 :=
by
  sorry

end NUMINAMATH_GPT_avg_salary_supervisors_l453_45363


namespace NUMINAMATH_GPT_cone_base_circumference_l453_45338

theorem cone_base_circumference (r : ℝ) (sector_angle : ℝ) (total_angle : ℝ) (C : ℝ) (h1 : r = 6) (h2 : sector_angle = 180) (h3 : total_angle = 360) (h4 : C = 2 * r * Real.pi) :
  (sector_angle / total_angle) * C = 6 * Real.pi :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l453_45338


namespace NUMINAMATH_GPT_minimum_value_of_ex_4e_negx_l453_45385

theorem minimum_value_of_ex_4e_negx : 
  ∃ (x : ℝ), (∀ (y : ℝ), y = Real.exp x + 4 * Real.exp (-x) → y ≥ 4) ∧ (Real.exp x + 4 * Real.exp (-x) = 4) :=
sorry

end NUMINAMATH_GPT_minimum_value_of_ex_4e_negx_l453_45385


namespace NUMINAMATH_GPT_algorithm_output_is_127_l453_45314
-- Import the entire Mathlib library

-- Define the possible values the algorithm can output
def possible_values : List ℕ := [15, 31, 63, 127]

-- Define the property where the value is of the form 2^n - 1
def is_exp2_minus_1 (x : ℕ) := ∃ n : ℕ, x = 2^n - 1

-- Define the main theorem to prove the algorithm's output is 127
theorem algorithm_output_is_127 : (∀ x ∈ possible_values, is_exp2_minus_1 x) →
                                      ∃ n : ℕ, 127 = 2^n - 1 :=
by
  -- Define the conditions and the proof steps are left out
  sorry

end NUMINAMATH_GPT_algorithm_output_is_127_l453_45314


namespace NUMINAMATH_GPT_range_of_a_for_three_zeros_l453_45303

-- Define the function f
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 2

theorem range_of_a_for_three_zeros (a : ℝ) (h : ∃ x1 x2 x3 : ℝ, f x1 a = 0 ∧ f x2 a = 0 ∧ f x3 a = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3) : a < -3 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_three_zeros_l453_45303


namespace NUMINAMATH_GPT_total_chairs_all_together_l453_45337

-- Definitions of given conditions
def rows := 7
def chairs_per_row := 12
def extra_chairs := 11

-- Main statement we want to prove
theorem total_chairs_all_together : 
  (rows * chairs_per_row + extra_chairs = 95) := 
by
  sorry

end NUMINAMATH_GPT_total_chairs_all_together_l453_45337


namespace NUMINAMATH_GPT_gcd_of_expression_l453_45322

noncomputable def gcd_expression (a b c d : ℤ) : ℤ :=
  (a - b) * (b - c) * (c - d) * (d - a) * (b - d) * (a - c)

theorem gcd_of_expression : 
  ∀ (a b c d : ℤ), ∃ (k : ℤ), gcd_expression a b c d = 12 * k :=
sorry

end NUMINAMATH_GPT_gcd_of_expression_l453_45322


namespace NUMINAMATH_GPT_PQ_sum_l453_45310

-- Define the problem conditions
variable (P Q x : ℝ)
variable (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3)))

-- Define the proof goal
theorem PQ_sum (h1 : (∀ x, x ≠ 3 → P / (x - 3) + Q * (x - 2) = (-4 * x^2 + 20 * x + 32) / (x - 3))) : P + Q = 52 :=
sorry

end NUMINAMATH_GPT_PQ_sum_l453_45310


namespace NUMINAMATH_GPT_trigonometric_identity_l453_45373

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l453_45373


namespace NUMINAMATH_GPT_sequence_a_n_a5_eq_21_l453_45316

theorem sequence_a_n_a5_eq_21 
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n + 2 * n) :
  a 5 = 21 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_a5_eq_21_l453_45316


namespace NUMINAMATH_GPT_sum_series_eq_two_l453_45327

theorem sum_series_eq_two : (∑' n : ℕ, (4 * (n + 1) - 2) / (3 ^ (n + 1))) = 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_series_eq_two_l453_45327


namespace NUMINAMATH_GPT_mean_marks_second_section_l453_45308

-- Definitions for the problem conditions
def num_students (section1 section2 section3 section4 : ℕ) : ℕ :=
  section1 + section2 + section3 + section4

def total_marks (section1 section2 section3 section4 : ℕ) (mean1 mean2 mean3 mean4 : ℝ) : ℝ :=
  section1 * mean1 + section2 * mean2 + section3 * mean3 + section4 * mean4

-- The final problem translated into a lean statement
theorem mean_marks_second_section :
  let section1 := 65
  let section2 := 35
  let section3 := 45
  let section4 := 42
  let mean1 := 50
  let mean3 := 55
  let mean4 := 45
  let overall_average := 51.95
  num_students section1 section2 section3 section4 = 187 →
  ((section1 : ℝ) * mean1 + (section2 : ℝ) * M + (section3 : ℝ) * mean3 + (section4 : ℝ) * mean4)
    = 187 * overall_average →
  M = 59.99 :=
by
  intros section1 section2 section3 section4 mean1 mean3 mean4 overall_average Hnum Htotal
  sorry

end NUMINAMATH_GPT_mean_marks_second_section_l453_45308


namespace NUMINAMATH_GPT_sara_added_onions_l453_45328

theorem sara_added_onions
  (initial_onions X : ℤ) 
  (h : initial_onions + X - 5 + 9 = initial_onions + 8) :
  X = 4 :=
by
  sorry

end NUMINAMATH_GPT_sara_added_onions_l453_45328


namespace NUMINAMATH_GPT_last_three_digits_of_16_pow_128_l453_45345

theorem last_three_digits_of_16_pow_128 : (16 ^ 128) % 1000 = 721 := 
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_16_pow_128_l453_45345


namespace NUMINAMATH_GPT_point_in_which_quadrant_l453_45379

noncomputable def quadrant_of_point (x y : ℝ) : String :=
if (x > 0) ∧ (y > 0) then
    "First"
else if (x < 0) ∧ (y > 0) then
    "Second"
else if (x < 0) ∧ (y < 0) then
    "Third"
else if (x > 0) ∧ (y < 0) then
    "Fourth"
else
    "On Axis"

theorem point_in_which_quadrant (α : ℝ) (h1 : π / 2 < α) (h2 : α < π) : quadrant_of_point (Real.sin α) (Real.cos α) = "Fourth" :=
by {
    sorry
}

end NUMINAMATH_GPT_point_in_which_quadrant_l453_45379


namespace NUMINAMATH_GPT_find_m_plus_n_l453_45397

def operation (m n : ℕ) : ℕ := m^n + m * n

theorem find_m_plus_n :
  ∃ (m n : ℕ), (2 ≤ m) ∧ (2 ≤ n) ∧ (operation m n = 64) ∧ (m + n = 6) :=
by {
  -- Begin the proof context
  sorry
}

end NUMINAMATH_GPT_find_m_plus_n_l453_45397


namespace NUMINAMATH_GPT_not_possible_to_construct_l453_45331

/-- The frame consists of 54 unit segments. -/
def frame_consists_of_54_units : Prop := sorry

/-- Each part of the construction set consists of three unit segments. -/
def part_is_three_units : Prop := sorry

/-- Each vertex of a cube is shared by three edges. -/
def vertex_shares_three_edges : Prop := sorry

/-- Six segments emerge from the center of the cube. -/
def center_has_six_segments : Prop := sorry

/-- It is not possible to construct the frame with exactly 18 parts. -/
theorem not_possible_to_construct
  (h1 : frame_consists_of_54_units)
  (h2 : part_is_three_units)
  (h3 : vertex_shares_three_edges)
  (h4 : center_has_six_segments) : 
  ¬ ∃ (parts : ℕ), parts = 18 :=
sorry

end NUMINAMATH_GPT_not_possible_to_construct_l453_45331
