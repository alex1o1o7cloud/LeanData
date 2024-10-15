import Mathlib

namespace NUMINAMATH_GPT_prime_square_sum_of_cubes_equals_three_l1015_101542

open Nat

theorem prime_square_sum_of_cubes_equals_three (p : ℕ) (h_prime : p.Prime) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ p^2 = a^3 + b^3) → (p = 3) :=
by
  sorry

end NUMINAMATH_GPT_prime_square_sum_of_cubes_equals_three_l1015_101542


namespace NUMINAMATH_GPT_f_at_1_l1015_101586

noncomputable def f (x : ℝ) : ℝ := 
  if x < 0 then x^2 + (5 : ℝ) * x
  else if x = 2 then 6
  else  - (x^2 + (5 : ℝ) * x)

theorem f_at_1 : f 1 = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_f_at_1_l1015_101586


namespace NUMINAMATH_GPT_previous_year_profit_percentage_l1015_101541

variables {R P: ℝ}

theorem previous_year_profit_percentage (h1: R > 0)
    (h2: P = 0.1 * R)
    (h3: 0.7 * P = 0.07 * R) :
    (P / R) * 100 = 10 :=
by
  -- Since we have P = 0.1 * R from the conditions and definitions,
  -- it follows straightforwardly that (P / R) * 100 = 10.
  -- We'll continue the proof from here.
  sorry

end NUMINAMATH_GPT_previous_year_profit_percentage_l1015_101541


namespace NUMINAMATH_GPT_probability_all_letters_SUPERBLOOM_l1015_101592

noncomputable def choose (n k : ℕ) : ℕ := sorry

theorem probability_all_letters_SUPERBLOOM :
  let P1 := 1 / (choose 6 3)
  let P2 := 9 / (choose 8 5)
  let P3 := 1 / (choose 5 4)
  P1 * P2 * P3 = 9 / 1120 :=
by
  sorry

end NUMINAMATH_GPT_probability_all_letters_SUPERBLOOM_l1015_101592


namespace NUMINAMATH_GPT_maximize_revenue_l1015_101583

-- Defining the revenue function
def revenue (p : ℝ) : ℝ := 200 * p - 4 * p^2

-- Defining the maximum price constraint
def price_constraint (p : ℝ) : Prop := p ≤ 40

-- Statement to be proven
theorem maximize_revenue : ∃ (p : ℝ), price_constraint p ∧ revenue p = 2500 ∧ (∀ q : ℝ, price_constraint q → revenue q ≤ revenue p) :=
sorry

end NUMINAMATH_GPT_maximize_revenue_l1015_101583


namespace NUMINAMATH_GPT_mod_remainder_of_expression_l1015_101582

theorem mod_remainder_of_expression : (7 * 10^20 + 2^20) % 9 = 2 := by
  sorry

end NUMINAMATH_GPT_mod_remainder_of_expression_l1015_101582


namespace NUMINAMATH_GPT_David_Marks_in_Mathematics_are_85_l1015_101572

theorem David_Marks_in_Mathematics_are_85
  (english_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (biology_marks : ℕ)
  (average_marks : ℕ)
  (num_subjects : ℕ)
  (h1 : english_marks = 86)
  (h2 : physics_marks = 92)
  (h3 : chemistry_marks = 87)
  (h4 : biology_marks = 95)
  (h5 : average_marks = 89)
  (h6 : num_subjects = 5) : 
  (86 + 92 + 87 + 95 + 85) / 5 = 89 :=
by sorry

end NUMINAMATH_GPT_David_Marks_in_Mathematics_are_85_l1015_101572


namespace NUMINAMATH_GPT_koala_fiber_intake_l1015_101527

theorem koala_fiber_intake (absorption_percentage : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) :
  absorption_percentage = 0.30 → absorbed_fiber = 12 → absorbed_fiber = absorption_percentage * total_fiber → total_fiber = 40 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_koala_fiber_intake_l1015_101527


namespace NUMINAMATH_GPT_find_AC_length_l1015_101504

theorem find_AC_length (AB BC CD DA : ℕ) 
  (hAB : AB = 10) (hBC : BC = 9) (hCD : CD = 19) (hDA : DA = 5) : 
  14 < AC ∧ AC < 19 → AC = 15 := 
by
  sorry

end NUMINAMATH_GPT_find_AC_length_l1015_101504


namespace NUMINAMATH_GPT_possible_values_of_c_l1015_101508

-- Definition of c(S) based on the problem conditions
def c (S : String) (m : ℕ) : ℕ := sorry

-- Condition: m > 1
variable {m : ℕ} (hm : m > 1)

-- Goal: To prove the possible values that c(S) can take
theorem possible_values_of_c (S : String) : ∃ n : ℕ, c S m = 0 ∨ c S m = 2^n :=
sorry

end NUMINAMATH_GPT_possible_values_of_c_l1015_101508


namespace NUMINAMATH_GPT_gina_keeps_170_l1015_101540

theorem gina_keeps_170 (initial_amount : ℕ)
    (money_to_mom : ℕ)
    (money_to_clothes : ℕ)
    (money_to_charity : ℕ)
    (remaining_money : ℕ) :
  initial_amount = 400 →
  money_to_mom = (1 / 4) * initial_amount →
  money_to_clothes = (1 / 8) * initial_amount →
  money_to_charity = (1 / 5) * initial_amount →
  remaining_money = initial_amount - (money_to_mom + money_to_clothes + money_to_charity) →
  remaining_money = 170 := sorry

end NUMINAMATH_GPT_gina_keeps_170_l1015_101540


namespace NUMINAMATH_GPT_range_of_x_of_sqrt_x_plus_3_l1015_101506

theorem range_of_x_of_sqrt_x_plus_3 (x : ℝ) (h : x + 3 ≥ 0) : x ≥ -3 := sorry

end NUMINAMATH_GPT_range_of_x_of_sqrt_x_plus_3_l1015_101506


namespace NUMINAMATH_GPT_video_call_cost_l1015_101531

-- Definitions based on the conditions
def charge_rate : ℕ := 30    -- Charge rate in won per ten seconds
def call_duration : ℕ := 2 * 60 + 40  -- Call duration in seconds

-- The proof statement, anticipating the solution to be a total cost calculation
theorem video_call_cost : (call_duration / 10) * charge_rate = 480 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_video_call_cost_l1015_101531


namespace NUMINAMATH_GPT_lcm_18_24_eq_72_l1015_101529

-- Definitions of the numbers whose LCM we need to find.
def a : ℕ := 18
def b : ℕ := 24

-- Statement that the least common multiple of 18 and 24 is 72.
theorem lcm_18_24_eq_72 : Nat.lcm a b = 72 := by
  sorry

end NUMINAMATH_GPT_lcm_18_24_eq_72_l1015_101529


namespace NUMINAMATH_GPT_staffing_battle_station_l1015_101584

-- Define the qualifications
def num_assistant_engineer := 3
def num_maintenance_1 := 4
def num_maintenance_2 := 4
def num_field_technician := 5
def num_radio_specialist := 5

-- Prove the total number of ways to fill the positions
theorem staffing_battle_station : 
  num_assistant_engineer * num_maintenance_1 * num_maintenance_2 * num_field_technician * num_radio_specialist = 960 := by
  sorry

end NUMINAMATH_GPT_staffing_battle_station_l1015_101584


namespace NUMINAMATH_GPT_solve_y_from_expression_l1015_101564

-- Define the conditions
def given_conditions := (784 = 28^2) ∧ (49 = 7^2)

-- Define the equivalency to prove based on the given conditions
theorem solve_y_from_expression (h : given_conditions) : 784 + 2 * 28 * 7 + 49 = 1225 := by
  sorry

end NUMINAMATH_GPT_solve_y_from_expression_l1015_101564


namespace NUMINAMATH_GPT_tank_fill_time_l1015_101557

/-- Given the rates at which pipes fill a tank, prove the total time to fill the tank using all three pipes. --/
theorem tank_fill_time (R_a R_b R_c : ℝ) (T : ℝ)
  (h1 : R_a = 1 / 35)
  (h2 : R_b = 2 * R_a)
  (h3 : R_c = 2 * R_b)
  (h4 : T = 5) :
  1 / (R_a + R_b + R_c) = T := by
  sorry

end NUMINAMATH_GPT_tank_fill_time_l1015_101557


namespace NUMINAMATH_GPT_parabola_directrix_l1015_101593

theorem parabola_directrix (p : ℝ) (hp : p > 0) (H : - (p / 2) = -3) : p = 6 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1015_101593


namespace NUMINAMATH_GPT_leo_current_weight_l1015_101598

variable (L K : ℝ)

noncomputable def leo_current_weight_predicate :=
  (L + 10 = 1.5 * K) ∧ (L + K = 180)

theorem leo_current_weight : leo_current_weight_predicate L K → L = 104 := by
  sorry

end NUMINAMATH_GPT_leo_current_weight_l1015_101598


namespace NUMINAMATH_GPT_geometric_shape_is_sphere_l1015_101516

-- Define the spherical coordinate system conditions
def spherical_coordinates (ρ θ φ r : ℝ) : Prop :=
  ρ = r

-- The theorem we want to prove
theorem geometric_shape_is_sphere (ρ θ φ r : ℝ) (h : spherical_coordinates ρ θ φ r) : ∀ (x y z : ℝ), (x^2 + y^2 + z^2 = r^2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_shape_is_sphere_l1015_101516


namespace NUMINAMATH_GPT_hannah_total_payment_l1015_101535

def washing_machine_cost : ℝ := 100
def dryer_cost : ℝ := washing_machine_cost - 30
def total_cost_before_discount : ℝ := washing_machine_cost + dryer_cost
def discount : ℝ := 0.10
def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount)

theorem hannah_total_payment : total_cost_after_discount = 153 := by
  sorry

end NUMINAMATH_GPT_hannah_total_payment_l1015_101535


namespace NUMINAMATH_GPT_find_ordered_pair_l1015_101595

theorem find_ordered_pair (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) 
  (hroots : ∀ x, x^2 + c * x + d = (x - c) * (x - d)) : 
  (c, d) = (1, -2) :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l1015_101595


namespace NUMINAMATH_GPT_remainder_of_n_squared_plus_4n_plus_5_l1015_101530

theorem remainder_of_n_squared_plus_4n_plus_5 {n : ℤ} (h : n % 50 = 1) : (n^2 + 4*n + 5) % 50 = 10 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_n_squared_plus_4n_plus_5_l1015_101530


namespace NUMINAMATH_GPT_trihedral_angle_properties_l1015_101523

-- Definitions for the problem's conditions
variables {α β γ : ℝ}
variables {A B C S : Type}
variables (angle_ASB angle_BSC angle_CSA : ℝ)

-- Given the conditions of the trihedral angle and the dihedral angles
theorem trihedral_angle_properties 
  (h1 : angle_ASB + angle_BSC + angle_CSA < 2 * Real.pi)
  (h2 : α + β + γ > Real.pi) : 
  true := 
by
  sorry

end NUMINAMATH_GPT_trihedral_angle_properties_l1015_101523


namespace NUMINAMATH_GPT_num_ways_to_turn_off_lights_l1015_101549

-- Let's define our problem in terms of the conditions given
-- Define the total number of lights
def total_lights : ℕ := 12

-- Define that we need to turn off 3 lights
def lights_to_turn_off : ℕ := 3

-- Define that we have 10 possible candidates for being turned off 
def candidates := total_lights - 2

-- Define the gap consumption statement that effectively reduce choices to 7 lights
def effective_choices := candidates - lights_to_turn_off

-- Define the combination formula for the number of ways to turn off the lights
def num_ways := Nat.choose effective_choices lights_to_turn_off

-- Final statement to prove
theorem num_ways_to_turn_off_lights : num_ways = Nat.choose 7 3 :=
by
  sorry

end NUMINAMATH_GPT_num_ways_to_turn_off_lights_l1015_101549


namespace NUMINAMATH_GPT_series_sum_eq_one_fourth_l1015_101503

noncomputable def sum_series : ℝ :=
  ∑' n, (3 ^ n / (1 + 3 ^ n + 3 ^ (n + 2) + 3 ^ (2 * n + 2)))

theorem series_sum_eq_one_fourth :
  sum_series = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_eq_one_fourth_l1015_101503


namespace NUMINAMATH_GPT_number_of_points_max_45_lines_l1015_101538

theorem number_of_points_max_45_lines (n : ℕ) (h : n * (n - 1) / 2 ≤ 45) : n = 10 := 
  sorry

end NUMINAMATH_GPT_number_of_points_max_45_lines_l1015_101538


namespace NUMINAMATH_GPT_find_a_parallel_lines_l1015_101548

theorem find_a_parallel_lines (a : ℝ) :
  (∃ k : ℝ, ∀ x y : ℝ, x * a + 2 * y + 2 = 0 ↔ 3 * x - y - 2 = k * (x * a + 2 * y + 2)) ↔ a = -6 := by
  sorry

end NUMINAMATH_GPT_find_a_parallel_lines_l1015_101548


namespace NUMINAMATH_GPT_parallelogram_area_correct_l1015_101555

def parallelogram_area (b h : ℝ) : ℝ := b * h

theorem parallelogram_area_correct :
  parallelogram_area 15 5 = 75 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_correct_l1015_101555


namespace NUMINAMATH_GPT_rectangle_to_square_l1015_101585

-- Definitions based on conditions
def rectangle_width : ℕ := 12
def rectangle_height : ℕ := 3
def area : ℕ := rectangle_width * rectangle_height
def parts : ℕ := 3
def part_area : ℕ := area / parts
def square_side : ℕ := Nat.sqrt area

-- Theorem to restate the problem
theorem rectangle_to_square : (area = 36) ∧ (part_area = 12) ∧ (square_side = 6) ∧
  (rectangle_width / parts = 4) ∧ (rectangle_height = 3) ∧ 
  ((rectangle_width / parts * parts) = rectangle_width) ∧ (parts * rectangle_height = square_side ^ 2) := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_rectangle_to_square_l1015_101585


namespace NUMINAMATH_GPT_unit_circle_chords_l1015_101580

theorem unit_circle_chords (
    s t u v : ℝ
) (hs : s = 1) (ht : t = 1) (hu : u = 2) (hv : v = 3) :
    (v - u = 1) ∧ (v * u = 6) ∧ (v^2 - u^2 = 5) :=
by
  have h1 : v - u = 1 := by rw [hv, hu]; norm_num
  have h2 : v * u = 6 := by rw [hv, hu]; norm_num
  have h3 : v^2 - u^2 = 5 := by rw [hv, hu]; norm_num
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_unit_circle_chords_l1015_101580


namespace NUMINAMATH_GPT_find_a_if_even_function_l1015_101518

-- Definitions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * 2^x + 2^(-x)

-- Theorem statement
theorem find_a_if_even_function (a : ℝ) (h : is_even_function (f a)) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_if_even_function_l1015_101518


namespace NUMINAMATH_GPT_find_multiplier_l1015_101559

theorem find_multiplier (x : ℝ) (h : (9 / 6) * x = 18) : x = 12 := sorry

end NUMINAMATH_GPT_find_multiplier_l1015_101559


namespace NUMINAMATH_GPT_angle_ratio_l1015_101513

-- Definitions as per the conditions
def bisects (x y z : ℝ) : Prop := x = y / 2
def trisects (x y z : ℝ) : Prop := y = x / 3

theorem angle_ratio (ABC PBQ BM x : ℝ) (h1 : bisects PBQ ABC PQ)
                                    (h2 : trisects PBQ BM M) :
  PBQ = 2 * x →
  PBQ = ABC / 2 →
  MBQ = x →
  ABQ = 4 * x →
  MBQ / ABQ = 1 / 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_angle_ratio_l1015_101513


namespace NUMINAMATH_GPT_sixty_first_batch_is_1211_l1015_101591

-- Definitions based on conditions
def total_bags : ℕ := 3000
def total_batches : ℕ := 150
def first_batch_number : ℕ := 11

-- Define the calculation of the 61st batch number
def batch_interval : ℕ := total_bags / total_batches
def sixty_first_batch_number : ℕ := first_batch_number + 60 * batch_interval

-- The statement of the proof
theorem sixty_first_batch_is_1211 : sixty_first_batch_number = 1211 := by
  sorry

end NUMINAMATH_GPT_sixty_first_batch_is_1211_l1015_101591


namespace NUMINAMATH_GPT_value_of_D_l1015_101563

theorem value_of_D (D : ℤ) (h : 80 - (5 - (6 + 2 * (7 - 8 - D))) = 89) : D = -5 :=
by sorry

end NUMINAMATH_GPT_value_of_D_l1015_101563


namespace NUMINAMATH_GPT_truck_capacities_transportation_plan_l1015_101536

-- Definitions of given conditions
def A_truck_capacity (x y : ℕ) : Prop := x + 2 * y = 50
def B_truck_capacity (x y : ℕ) : Prop := 5 * x + 4 * y = 160
def total_transport_cost (m n : ℕ) : ℕ := 500 * m + 400 * n
def most_cost_effective_plan (m n cost : ℕ) : Prop := 
  m + 2 * n = 10 ∧ (20 * m + 15 * n = 190) ∧ cost = total_transport_cost m n ∧ cost = 4800

-- Proving the capacities of trucks A and B
theorem truck_capacities : 
  ∃ x y : ℕ, A_truck_capacity x y ∧ B_truck_capacity x y ∧ x = 20 ∧ y = 15 := 
sorry

-- Proving the most cost-effective transportation plan
theorem transportation_plan : 
  ∃ m n cost, (total_transport_cost m n = cost) ∧ most_cost_effective_plan m n cost := 
sorry

end NUMINAMATH_GPT_truck_capacities_transportation_plan_l1015_101536


namespace NUMINAMATH_GPT_BANANA_arrangement_l1015_101576

theorem BANANA_arrangement : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2) * (Nat.factorial 1)) = 60 := 
by
  sorry

end NUMINAMATH_GPT_BANANA_arrangement_l1015_101576


namespace NUMINAMATH_GPT_find_initial_red_marbles_l1015_101550

theorem find_initial_red_marbles (x y : ℚ) 
  (h1 : 2 * x = 3 * y) 
  (h2 : 5 * (x - 15) = 2 * (y + 25)) 
  : x = 375 / 11 := 
by
  sorry

end NUMINAMATH_GPT_find_initial_red_marbles_l1015_101550


namespace NUMINAMATH_GPT_bernardo_receives_l1015_101568

theorem bernardo_receives :
  let amount_distributed (n : ℕ) : ℕ := (n * (n + 1)) / 2
  let is_valid (n : ℕ) : Prop := amount_distributed n ≤ 1000
  let bernardo_amount (k : ℕ) : ℕ := (k * (2 + (k - 1) * 3)) / 2
  ∃ k : ℕ, is_valid (15 * 3) ∧ bernardo_amount 15 = 345 :=
sorry

end NUMINAMATH_GPT_bernardo_receives_l1015_101568


namespace NUMINAMATH_GPT_avoid_loss_maximize_profit_max_profit_per_unit_l1015_101566

-- Definitions of the functions as per problem conditions
noncomputable def C (x : ℝ) : ℝ := 2 + x
noncomputable def R (x : ℝ) : ℝ := if x ≤ 4 then 4 * x - (1 / 2) * x^2 - (1 / 2) else 7.5
noncomputable def L (x : ℝ) : ℝ := R x - C x

-- Proof statements

-- 1. Range to avoid loss
theorem avoid_loss (x : ℝ) : 1 ≤ x ∧ x ≤ 5.5 ↔ L x ≥ 0 :=
by
  sorry

-- 2. Production to maximize profit
theorem maximize_profit (x : ℝ) : x = 3 ↔ ∀ y, L y ≤ L 3 :=
by
  sorry

-- 3. Maximum profit per unit selling price
theorem max_profit_per_unit (x : ℝ) : x = 3 ↔ (R 3 / 3 = 2.33) :=
by
  sorry

end NUMINAMATH_GPT_avoid_loss_maximize_profit_max_profit_per_unit_l1015_101566


namespace NUMINAMATH_GPT_find_prime_p_l1015_101524

theorem find_prime_p :
  ∃ p : ℕ, Prime p ∧ (∃ a b : ℤ, p = 5 ∧ 1 < p ∧ p ≤ 11 ∧ (a^2 + p * a - 720 * p = 0) ∧ (b^2 - p * b + 720 * p = 0)) :=
sorry

end NUMINAMATH_GPT_find_prime_p_l1015_101524


namespace NUMINAMATH_GPT_fraction_equality_l1015_101512

def op_at (a b : ℕ) : ℕ := a * b + b^2
def op_hash (a b : ℕ) : ℕ := a + b + a * (b^2)

theorem fraction_equality : (op_at 5 3 : ℚ) / (op_hash 5 3 : ℚ) = 24 / 53 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_equality_l1015_101512


namespace NUMINAMATH_GPT_total_cups_needed_l1015_101599

-- Define the known conditions
def ratio_butter : ℕ := 2
def ratio_flour : ℕ := 3
def ratio_sugar : ℕ := 5
def total_sugar_in_cups : ℕ := 10

-- Define the parts-to-cups conversion
def cup_per_part := total_sugar_in_cups / ratio_sugar

-- Define the amounts of each ingredient in cups
def butter_in_cups := ratio_butter * cup_per_part
def flour_in_cups := ratio_flour * cup_per_part
def sugar_in_cups := ratio_sugar * cup_per_part

-- Define the total number of cups
def total_cups := butter_in_cups + flour_in_cups + sugar_in_cups

-- Theorem to prove
theorem total_cups_needed : total_cups = 20 := by
  sorry

end NUMINAMATH_GPT_total_cups_needed_l1015_101599


namespace NUMINAMATH_GPT_distance_city_A_C_l1015_101526

-- Define the conditions
def starts_simultaneously (A : Prop) (Eddy Freddy : Prop) := Eddy ∧ Freddy
def travels (A B C : Prop) (Eddy Freddy : Prop) := Eddy → 3 = 3 ∧ Freddy → 4 = 4
def distance_AB (A B : Prop) := 600
def speed_ratio (Eddy_speed Freddy_speed : ℝ) := Eddy_speed / Freddy_speed = 1.7391304347826086

noncomputable def distance_AC (Eddy_time Freddy_time : ℝ) (Eddy_speed Freddy_speed : ℝ) 
  := (Eddy_speed / 1.7391304347826086) * Freddy_time

theorem distance_city_A_C 
  (A B C Eddy Freddy : Prop)
  (Eddy_time Freddy_time : ℝ) 
  (Eddy_speed effective_Freddy_speed : ℝ)
  (h1 : starts_simultaneously A Eddy Freddy)
  (h2 : travels A B C Eddy Freddy)
  (h3 : distance_AB A B = 600)
  (h4 : speed_ratio Eddy_speed effective_Freddy_speed)
  (h5 : Eddy_speed = 200)
  (h6 : effective_Freddy_speed = 115)
  : distance_AC Eddy_time Freddy_time Eddy_speed effective_Freddy_speed = 460 := 
  by sorry

end NUMINAMATH_GPT_distance_city_A_C_l1015_101526


namespace NUMINAMATH_GPT_value_calculation_l1015_101552

-- Define the given number
def given_number : ℝ := 93.75

-- Define the percentages as ratios
def forty_percent : ℝ := 0.4
def sixteen_percent : ℝ := 0.16

-- Calculate the intermediate value for 40% of the given number
def intermediate_value := forty_percent * given_number

-- Final value calculation for 16% of the intermediate value
def final_value := sixteen_percent * intermediate_value

-- The theorem to prove
theorem value_calculation : final_value = 6 := by
  -- Expanding definitions to substitute and simplify
  unfold final_value intermediate_value forty_percent sixteen_percent given_number
  -- Proving the correctness by calculating
  sorry

end NUMINAMATH_GPT_value_calculation_l1015_101552


namespace NUMINAMATH_GPT_smallest_square_value_l1015_101558

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (r s : ℕ) (hr : 15 * a + 16 * b = r^2) (hs : 16 * a - 15 * b = s^2) :
  min (r^2) (s^2) = 481^2 :=
  sorry

end NUMINAMATH_GPT_smallest_square_value_l1015_101558


namespace NUMINAMATH_GPT_value_of_expr_l1015_101596

-- Definitions
def operation (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

-- The proof statement
theorem value_of_expr (a b : ℕ) (h₀ : operation a b = 100) : (a + b) + 6 = 11 := by
  sorry

end NUMINAMATH_GPT_value_of_expr_l1015_101596


namespace NUMINAMATH_GPT_factorize_expression_find_xy_l1015_101515

-- Problem 1: Factorizing the quadratic expression
theorem factorize_expression (x : ℝ) : 
  x^2 - 120 * x + 3456 = (x - 48) * (x - 72) :=
sorry

-- Problem 2: Finding the product xy from the given equation
theorem find_xy (x y : ℝ) (h : x^2 + y^2 + 8 * x - 12 * y + 52 = 0) : 
  x * y = -24 :=
sorry

end NUMINAMATH_GPT_factorize_expression_find_xy_l1015_101515


namespace NUMINAMATH_GPT_weight_in_one_hand_l1015_101561

theorem weight_in_one_hand (total_weight : ℕ) (h : total_weight = 16) : total_weight / 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_weight_in_one_hand_l1015_101561


namespace NUMINAMATH_GPT_angle_terminal_side_l1015_101511

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * (180 / Real.pi)

theorem angle_terminal_side :
  ∃ k : ℤ, rad_to_deg (π / 12) + 360 * k = 375 :=
sorry

end NUMINAMATH_GPT_angle_terminal_side_l1015_101511


namespace NUMINAMATH_GPT_solve_for_cubic_l1015_101547

theorem solve_for_cubic (x y : ℝ) (h₁ : x * (x + y) = 49) (h₂: y * (x + y) = 63) : (x + y)^3 = 448 * Real.sqrt 7 := 
sorry

end NUMINAMATH_GPT_solve_for_cubic_l1015_101547


namespace NUMINAMATH_GPT_complete_the_square_problem_l1015_101597

theorem complete_the_square_problem :
  ∃ r s : ℝ, (r = -2) ∧ (s = 9) ∧ (r + s = 7) ∧ ∀ x : ℝ, 15 * x ^ 2 - 60 * x - 135 = 0 ↔ (x + r) ^ 2 = s := 
by
  sorry

end NUMINAMATH_GPT_complete_the_square_problem_l1015_101597


namespace NUMINAMATH_GPT_perfect_square_term_l1015_101590

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 2
  else 4 * seq (n - 1) - seq (n - 2)

theorem perfect_square_term : ∀ n, (∃ k, seq n = k * k) ↔ n = 0 := by
  sorry

end NUMINAMATH_GPT_perfect_square_term_l1015_101590


namespace NUMINAMATH_GPT_geometric_solid_is_tetrahedron_l1015_101528

-- Definitions based on the conditions provided
def top_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def front_view_is_triangle : Prop := sorry -- Placeholder for the actual definition
def side_view_is_triangle : Prop := sorry -- Placeholder for the actual definition

-- Theorem statement to prove the geometric solid is a triangular pyramid
theorem geometric_solid_is_tetrahedron 
  (h_top : top_view_is_triangle)
  (h_front : front_view_is_triangle)
  (h_side : side_view_is_triangle) :
  -- Conclusion that the solid is a triangular pyramid (tetrahedron)
  is_tetrahedron :=
sorry

end NUMINAMATH_GPT_geometric_solid_is_tetrahedron_l1015_101528


namespace NUMINAMATH_GPT_common_ratio_geometric_series_l1015_101544

-- Define the terms of the geometric series
def term (n : ℕ) : ℚ :=
  match n with
  | 0     => 7 / 8
  | 1     => -21 / 32
  | 2     => 63 / 128
  | _     => sorry  -- Placeholder for further terms if necessary

-- Define the common ratio
def common_ratio : ℚ := -3 / 4

-- Prove that the common ratio is consistent for the given series
theorem common_ratio_geometric_series :
  ∀ (n : ℕ), term (n + 1) / term n = common_ratio :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_geometric_series_l1015_101544


namespace NUMINAMATH_GPT_reciprocal_inequality_l1015_101505

open Real

theorem reciprocal_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / a) + (1 / b) > 1 / (a + b) :=
sorry

end NUMINAMATH_GPT_reciprocal_inequality_l1015_101505


namespace NUMINAMATH_GPT_connected_paper_area_l1015_101577

def side_length := 30 -- side of each square paper in cm
def overlap_length := 7 -- overlap length in cm
def num_pieces := 6 -- number of paper pieces

def effective_length (side_length overlap_length : ℕ) := side_length - overlap_length
def total_connected_length (num_pieces : ℕ) (side_length overlap_length : ℕ) :=
  side_length + (num_pieces - 1) * (effective_length side_length overlap_length)

def width := side_length -- width of the connected paper is the side of each square piece of paper

def area (length width : ℕ) := length * width

theorem connected_paper_area : area (total_connected_length num_pieces side_length overlap_length) width = 4350 :=
by
  sorry

end NUMINAMATH_GPT_connected_paper_area_l1015_101577


namespace NUMINAMATH_GPT_range_of_m_l1015_101534

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

def increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioi 2, 0 ≤ deriv (f m) x) ↔ m ≤ 5 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1015_101534


namespace NUMINAMATH_GPT_combined_age_l1015_101560

theorem combined_age (H : ℕ) (Ryanne : ℕ) (Jamison : ℕ) 
  (h1 : Ryanne = H + 7) 
  (h2 : H + Ryanne = 15) 
  (h3 : Jamison = 2 * H) : 
  H + Ryanne + Jamison = 23 := 
by 
  sorry

end NUMINAMATH_GPT_combined_age_l1015_101560


namespace NUMINAMATH_GPT_no_roots_one_and_neg_one_l1015_101533

theorem no_roots_one_and_neg_one (a b : ℝ) : ¬ ((1 + a + b = 0) ∧ (-1 + a + b = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_roots_one_and_neg_one_l1015_101533


namespace NUMINAMATH_GPT_cost_of_apples_l1015_101594

theorem cost_of_apples 
  (total_cost : ℕ)
  (cost_bananas : ℕ)
  (cost_bread : ℕ)
  (cost_milk : ℕ)
  (cost_apples : ℕ)
  (h_total : total_cost = 42)
  (h_bananas : cost_bananas = 12)
  (h_bread : cost_bread = 9)
  (h_milk : cost_milk = 7)
  (h_combined : cost_apples = total_cost - (cost_bananas + cost_bread + cost_milk)) : 
  cost_apples = 14 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_apples_l1015_101594


namespace NUMINAMATH_GPT_price_of_third_variety_l1015_101537

theorem price_of_third_variety 
    (price1 price2 price3 : ℝ)
    (mix_ratio1 mix_ratio2 mix_ratio3 : ℝ)
    (mixture_price : ℝ)
    (h1 : price1 = 126)
    (h2 : price2 = 135)
    (h3 : mix_ratio1 = 1)
    (h4 : mix_ratio2 = 1)
    (h5 : mix_ratio3 = 2)
    (h6 : mixture_price = 153) :
    price3 = 175.5 :=
by
  sorry

end NUMINAMATH_GPT_price_of_third_variety_l1015_101537


namespace NUMINAMATH_GPT_rate_of_current_l1015_101571

theorem rate_of_current (c : ℝ) (h1 : ∀ d : ℝ, d / (3.9 - c) = 2 * (d / (3.9 + c))) : c = 1.3 :=
sorry

end NUMINAMATH_GPT_rate_of_current_l1015_101571


namespace NUMINAMATH_GPT_retirement_amount_l1015_101588

-- Define the principal amount P
def P : ℝ := 750000

-- Define the annual interest rate r
def r : ℝ := 0.08

-- Define the time period in years t
def t : ℝ := 12

-- Define the accumulated amount A
def A : ℝ := P * (1 + r * t)

-- Prove that the accumulated amount A equals 1470000
theorem retirement_amount : A = 1470000 := by
  -- The proof will involve calculating the compound interest
  sorry

end NUMINAMATH_GPT_retirement_amount_l1015_101588


namespace NUMINAMATH_GPT_marked_price_of_article_l1015_101589

noncomputable def marked_price (discounted_total : ℝ) (num_articles : ℕ) (discount_rate : ℝ) : ℝ :=
  let selling_price_each := discounted_total / num_articles
  let discount_factor := 1 - discount_rate
  selling_price_each / discount_factor

theorem marked_price_of_article :
  marked_price 50 2 0.10 = 250 / 9 :=
by
  unfold marked_price
  -- Instantiate values:
  -- discounted_total = 50
  -- num_articles = 2
  -- discount_rate = 0.10
  sorry

end NUMINAMATH_GPT_marked_price_of_article_l1015_101589


namespace NUMINAMATH_GPT_cricket_average_increase_l1015_101522

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end NUMINAMATH_GPT_cricket_average_increase_l1015_101522


namespace NUMINAMATH_GPT_Elle_practice_time_l1015_101539

variable (x : ℕ)

theorem Elle_practice_time : 
  (5 * x) + (3 * x) = 240 → x = 30 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Elle_practice_time_l1015_101539


namespace NUMINAMATH_GPT_train_length_300_l1015_101514

/-- 
Proving the length of the train given the conditions on crossing times and length of the platform.
-/
theorem train_length_300 (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 200 = V * 30) : 
  L = 300 := 
by
  sorry

end NUMINAMATH_GPT_train_length_300_l1015_101514


namespace NUMINAMATH_GPT_unique_solution_l1015_101579

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  x > 0 ∧ (x * Real.sqrt (18 - x) + Real.sqrt (24 * x - x^3) ≥ 18)

theorem unique_solution :
  ∀ x : ℝ, satisfies_condition x ↔ x = 6 :=
by
  intro x
  unfold satisfies_condition
  sorry

end NUMINAMATH_GPT_unique_solution_l1015_101579


namespace NUMINAMATH_GPT_triangle_perimeter_problem_l1015_101545

theorem triangle_perimeter_problem : 
  ∀ (c : ℝ), 20 + 15 > c ∧ 20 + c > 15 ∧ 15 + c > 20 → ¬ (35 + c = 72) :=
by
  intros c h
  sorry

end NUMINAMATH_GPT_triangle_perimeter_problem_l1015_101545


namespace NUMINAMATH_GPT_input_language_is_input_l1015_101575

def is_print_statement (statement : String) : Prop := 
  statement = "PRINT"

def is_input_statement (statement : String) : Prop := 
  statement = "INPUT"

def is_conditional_statement (statement : String) : Prop := 
  statement = "IF"

theorem input_language_is_input :
  is_input_statement "INPUT" := 
by
  -- Here we need to show "INPUT" is an input statement
  sorry

end NUMINAMATH_GPT_input_language_is_input_l1015_101575


namespace NUMINAMATH_GPT_ratio_of_abc_l1015_101501

theorem ratio_of_abc {a b c : ℕ} (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) 
                     (h_ratio : ∃ x : ℕ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x)
                     (h_mean : (a + b + c) / 3 = 42) : 
  a = 28 := 
sorry

end NUMINAMATH_GPT_ratio_of_abc_l1015_101501


namespace NUMINAMATH_GPT_greatest_t_value_l1015_101510

theorem greatest_t_value :
  ∃ t_max : ℝ, (∀ t : ℝ, ((t ≠  8) ∧ (t ≠ -7) → (t^2 - t - 90) / (t - 8) = 6 / (t + 7) → t ≤ t_max)) ∧ t_max = -1 :=
sorry

end NUMINAMATH_GPT_greatest_t_value_l1015_101510


namespace NUMINAMATH_GPT_quadratic_equation_root_and_coef_l1015_101546

theorem quadratic_equation_root_and_coef (k x : ℤ) (h1 : x^2 - 3 * x + k = 0)
  (root4 : x = 4) : (x = 4 ∧ k = -4 ∧ ∀ y, y ≠ 4 → y^2 - 3 * y + k = 0 → y = -1) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_equation_root_and_coef_l1015_101546


namespace NUMINAMATH_GPT_find_m_and_n_l1015_101553

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_and_n_l1015_101553


namespace NUMINAMATH_GPT_evaluate_power_l1015_101507

theorem evaluate_power : (3^3)^2 = 729 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_power_l1015_101507


namespace NUMINAMATH_GPT_solve_for_m_l1015_101587

theorem solve_for_m (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_m_l1015_101587


namespace NUMINAMATH_GPT_part_a_part_b_l1015_101574

def P (m n : ℕ) : ℕ := m^2003 * n^2017 - m^2017 * n^2003

theorem part_a (m n : ℕ) : P m n % 24 = 0 := 
by sorry

theorem part_b : ∃ (m n : ℕ), P m n % 7 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l1015_101574


namespace NUMINAMATH_GPT_num_integer_solutions_quadratic_square_l1015_101500

theorem num_integer_solutions_quadratic_square : 
  (∃ xs : Finset ℤ, 
    (∀ x ∈ xs, ∃ k : ℤ, (x^4 + 8*x^3 + 18*x^2 + 8*x + 64) = k^2) ∧ 
    xs.card = 2) := sorry

end NUMINAMATH_GPT_num_integer_solutions_quadratic_square_l1015_101500


namespace NUMINAMATH_GPT_math_problem_l1015_101581

theorem math_problem 
  (a b c : ℕ) 
  (h_primea : Nat.Prime a)
  (h_posa : 0 < a)
  (h_posb : 0 < b)
  (h_posc : 0 < c)
  (h_eq : a^2 + b^2 = c^2) :
  (b % 2 ≠ c % 2) ∧ (∃ k, 2 * (a + b + 1) = k^2) := 
sorry

end NUMINAMATH_GPT_math_problem_l1015_101581


namespace NUMINAMATH_GPT_number_square_of_digits_l1015_101570

theorem number_square_of_digits (x y : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) (h3 : 0 ≤ y) (h4 : y ≤ 9) :
  ∃ n : ℕ, (∃ (k : ℕ), (1001 * x + 110 * y) = k^2) ↔ (x = 7 ∧ y = 4) :=
by
  sorry

end NUMINAMATH_GPT_number_square_of_digits_l1015_101570


namespace NUMINAMATH_GPT_parrot_consumption_l1015_101569

theorem parrot_consumption :
  ∀ (parakeet_daily : ℕ) (finch_daily : ℕ) (num_parakeets : ℕ) (num_parrots : ℕ) (num_finches : ℕ) (weekly_birdseed : ℕ),
    parakeet_daily = 2 →
    finch_daily = parakeet_daily / 2 →
    num_parakeets = 3 →
    num_parrots = 2 →
    num_finches = 4 →
    weekly_birdseed = 266 →
    14 = (weekly_birdseed - ((num_parakeets * parakeet_daily + num_finches * finch_daily) * 7)) / num_parrots / 7 :=
by
  intros parakeet_daily finch_daily num_parakeets num_parrots num_finches weekly_birdseed
  intros hp1 hp2 hp3 hp4 hp5 hp6
  sorry

end NUMINAMATH_GPT_parrot_consumption_l1015_101569


namespace NUMINAMATH_GPT_complex_modulus_eq_one_l1015_101519

open Complex

theorem complex_modulus_eq_one (a b : ℝ) (h : (1 + 2 * Complex.I) / (a + b * Complex.I) = 2 - Complex.I) :
  abs (a - b * Complex.I) = 1 := by
  sorry

end NUMINAMATH_GPT_complex_modulus_eq_one_l1015_101519


namespace NUMINAMATH_GPT_vectors_are_perpendicular_l1015_101578

def vector_a : ℝ × ℝ := (-5, 6)
def vector_b : ℝ × ℝ := (6, 5)

theorem vectors_are_perpendicular :
  (vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_vectors_are_perpendicular_l1015_101578


namespace NUMINAMATH_GPT_number_of_small_jars_l1015_101551

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 :=
by
  sorry

end NUMINAMATH_GPT_number_of_small_jars_l1015_101551


namespace NUMINAMATH_GPT_square_side_length_leq_half_l1015_101565

theorem square_side_length_leq_half
    (l : ℝ)
    (h_square_inside_unit : l ≤ 1)
    (h_no_center_contain : ∀ (x y : ℝ), x^2 + y^2 > (l/2)^2 → (0.5 ≤ x ∨ 0.5 ≤ y)) :
    l ≤ 0.5 := 
sorry

end NUMINAMATH_GPT_square_side_length_leq_half_l1015_101565


namespace NUMINAMATH_GPT_total_length_of_rope_l1015_101567

theorem total_length_of_rope (x : ℝ) : (∃ r1 r2 : ℝ, r1 / r2 = 2 / 3 ∧ r1 = 16 ∧ x = r1 + r2) → x = 40 :=
by
  intro h
  cases' h with r1 hr
  cases' hr with r2 hs
  sorry

end NUMINAMATH_GPT_total_length_of_rope_l1015_101567


namespace NUMINAMATH_GPT_remainder_of_3456_div_97_l1015_101543

theorem remainder_of_3456_div_97 :
  3456 % 97 = 61 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3456_div_97_l1015_101543


namespace NUMINAMATH_GPT_quadratic_function_properties_l1015_101502

def quadratic_function (x : ℝ) : ℝ :=
  -6 * x^2 + 36 * x - 48

theorem quadratic_function_properties :
  quadratic_function 2 = 0 ∧ quadratic_function 4 = 0 ∧ quadratic_function 3 = 6 :=
by
  -- The proof is omitted
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_quadratic_function_properties_l1015_101502


namespace NUMINAMATH_GPT_cost_price_computer_table_l1015_101521

-- Define the variables
def cost_price : ℝ := 3840
def selling_price (CP : ℝ) := CP * 1.25

-- State the conditions and the proof problem
theorem cost_price_computer_table 
  (SP : ℝ) 
  (h1 : SP = 4800)
  (h2 : ∀ CP : ℝ, SP = selling_price CP) :
  cost_price = 3840 :=
by 
  sorry

end NUMINAMATH_GPT_cost_price_computer_table_l1015_101521


namespace NUMINAMATH_GPT_numbers_not_all_less_than_six_l1015_101520

theorem numbers_not_all_less_than_six (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ¬ (a + 4 / b < 6 ∧ b + 9 / c < 6 ∧ c + 16 / a < 6) :=
sorry

end NUMINAMATH_GPT_numbers_not_all_less_than_six_l1015_101520


namespace NUMINAMATH_GPT_partition_sum_condition_l1015_101532

theorem partition_sum_condition (X : Finset ℕ) (hX : X = {1, 2, 3, 4, 5, 6, 7, 8, 9}) :
  ∀ (A B : Finset ℕ), A ∪ B = X → A ∩ B = ∅ →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a + b = c := 
by
  -- sorry is here to acknowledge that no proof is required per instructions.
  sorry

end NUMINAMATH_GPT_partition_sum_condition_l1015_101532


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l1015_101573

theorem solve_eq1 : (∃ x : ℚ, (5 * x - 1) / 4 = (3 * x + 1) / 2 - (2 - x) / 3) ↔ x = -1 / 7 :=
sorry

theorem solve_eq2 : (∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 - (2 * x + 1) / 5) ↔ x = -9 / 28 :=
sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l1015_101573


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l1015_101554

theorem cylinder_volume_ratio
  (h : ℝ)
  (r1 : ℝ)
  (r3 : ℝ := 3 * r1)
  (V1 : ℝ := 40) :
  let V2 := π * r3^2 * h
  (π * r1^2 * h = V1) → 
  V2 = 360 := by
{
  sorry
}

end NUMINAMATH_GPT_cylinder_volume_ratio_l1015_101554


namespace NUMINAMATH_GPT_workshop_average_salary_l1015_101525

theorem workshop_average_salary :
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  A = 8000 :=
by
  -- Definitions according to given conditions
  let technicians := 8
  let rest := 24 - technicians
  let avg_technician_salary := 12000
  let avg_rest_salary := 6000
  let total_workers := 24
  let total_staff_salary := (technicians * avg_technician_salary) + (rest * avg_rest_salary)
  let A := total_staff_salary / total_workers
  -- We need to show that A = 8000
  show A = 8000
  sorry

end NUMINAMATH_GPT_workshop_average_salary_l1015_101525


namespace NUMINAMATH_GPT_factor_expression_l1015_101556

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1015_101556


namespace NUMINAMATH_GPT_total_cantaloupes_l1015_101509

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by sorry

end NUMINAMATH_GPT_total_cantaloupes_l1015_101509


namespace NUMINAMATH_GPT_malcolm_initial_white_lights_l1015_101562

-- Definitions based on the conditions
def red_lights : Nat := 12
def blue_lights : Nat := 3 * red_lights
def green_lights : Nat := 6
def total_colored_lights := red_lights + blue_lights + green_lights
def lights_left_to_buy : Nat := 5
def initially_white_lights := total_colored_lights + lights_left_to_buy

-- Proof statement
theorem malcolm_initial_white_lights : initially_white_lights = 59 := by
  sorry

end NUMINAMATH_GPT_malcolm_initial_white_lights_l1015_101562


namespace NUMINAMATH_GPT_consecutive_even_sum_l1015_101517

theorem consecutive_even_sum (n : ℤ) (h : (n - 2) + (n + 2) = 156) : n = 78 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_even_sum_l1015_101517
