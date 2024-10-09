import Mathlib

namespace three_digit_numbers_l2353_235358

theorem three_digit_numbers (a b c n : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
    (h5 : 0 ≤ c) (h6 : c ≤ 9) (h7 : n = 100 * a + 10 * b + c) (h8 : 10 * b + c = (100 * a + 10 * b + c) / 5) :
    n = 125 ∨ n = 250 ∨ n = 375 := 
by 
  sorry

end three_digit_numbers_l2353_235358


namespace cos_squared_sin_pi_over_2_plus_alpha_l2353_235378

variable (α : ℝ)

-- Given conditions
def cond1 : Prop := (Real.pi / 2) < α * Real.pi
def cond2 : Prop := Real.cos α = -3 / 5

-- Proof goal
theorem cos_squared_sin_pi_over_2_plus_alpha :
  cond1 α → cond2 α →
  (Real.cos (Real.sin (Real.pi / 2 + α)))^2 = 8 / 25 :=
by
  intro h1 h2
  sorry

end cos_squared_sin_pi_over_2_plus_alpha_l2353_235378


namespace total_amount_of_money_l2353_235391

theorem total_amount_of_money (N50 N500 : ℕ) (h1 : N50 = 97) (h2 : N50 + N500 = 108) : 
  50 * N50 + 500 * N500 = 10350 := by
  sorry

end total_amount_of_money_l2353_235391


namespace fraction_sum_eq_l2353_235385

variable {x : ℝ}

theorem fraction_sum_eq (h : x ≠ -1) : 
  (x / (x + 1) ^ 2) + (1 / (x + 1) ^ 2) = 1 / (x + 1) := 
by
  sorry

end fraction_sum_eq_l2353_235385


namespace fraction_of_work_completed_l2353_235398

-- Definitions
def work_rate_x : ℚ := 1 / 14
def work_rate_y : ℚ := 1 / 20
def work_rate_z : ℚ := 1 / 25

-- Given the combined work rate and time
def combined_work_rate : ℚ := work_rate_x + work_rate_y + work_rate_z
def time_worked : ℚ := 5

-- The fraction of work completed
def fraction_work_completed : ℚ := combined_work_rate * time_worked

-- Statement to prove
theorem fraction_of_work_completed : fraction_work_completed = 113 / 140 := by
  sorry

end fraction_of_work_completed_l2353_235398


namespace tan_a_div_tan_b_l2353_235382

variable {a b : ℝ}

-- Conditions
axiom sin_a_plus_b : Real.sin (a + b) = 1/2
axiom sin_a_minus_b : Real.sin (a - b) = 1/4

-- Proof statement (without the explicit proof)
theorem tan_a_div_tan_b : (Real.tan a) / (Real.tan b) = 3 := by
  sorry

end tan_a_div_tan_b_l2353_235382


namespace arithmetic_sequence_ratio_l2353_235316

theorem arithmetic_sequence_ratio (a d : ℕ) (h : b = a + 3 * d) : a = 1 -> d = 1 -> (a / b = 1 / 4) :=
by
  sorry

end arithmetic_sequence_ratio_l2353_235316


namespace contrapositive_of_x_squared_gt_1_l2353_235352

theorem contrapositive_of_x_squared_gt_1 (x : ℝ) (h : x ≤ 1) : x^2 ≤ 1 :=
sorry

end contrapositive_of_x_squared_gt_1_l2353_235352


namespace usual_time_to_school_l2353_235368

theorem usual_time_to_school (R : ℝ) (T : ℝ) (h : (17 / 13) * (T - 7) = T) : T = 29.75 :=
sorry

end usual_time_to_school_l2353_235368


namespace basketball_game_score_difference_l2353_235310

theorem basketball_game_score_difference :
  let blueFreeThrows := 18
  let blueTwoPointers := 25
  let blueThreePointers := 6
  let redFreeThrows := 15
  let redTwoPointers := 22
  let redThreePointers := 5
  let blueScore := blueFreeThrows * 1 + blueTwoPointers * 2 + blueThreePointers * 3
  let redScore := redFreeThrows * 1 + redTwoPointers * 2 + redThreePointers * 3
  blueScore - redScore = 12 := by
  sorry

end basketball_game_score_difference_l2353_235310


namespace teal_bakery_revenue_l2353_235320

theorem teal_bakery_revenue :
    let pumpkin_pies := 4
    let pumpkin_pie_slices := 8
    let pumpkin_slice_price := 5
    let custard_pies := 5
    let custard_pie_slices := 6
    let custard_slice_price := 6
    let total_pumpkin_slices := pumpkin_pies * pumpkin_pie_slices
    let total_custard_slices := custard_pies * custard_pie_slices
    let pumpkin_revenue := total_pumpkin_slices * pumpkin_slice_price
    let custard_revenue := total_custard_slices * custard_slice_price
    let total_revenue := pumpkin_revenue + custard_revenue
    total_revenue = 340 :=
by
  sorry

end teal_bakery_revenue_l2353_235320


namespace inequality_always_true_l2353_235397

variable (a b c : ℝ)

theorem inequality_always_true (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 0) : c * a < c * b := by
  sorry

end inequality_always_true_l2353_235397


namespace moles_of_C6H5CH3_formed_l2353_235370

-- Stoichiometry of the reaction
def balanced_reaction (C6H6 CH4 C6H5CH3 H2 : ℝ) : Prop :=
  C6H6 + CH4 = C6H5CH3 + H2

-- Given conditions
def reaction_conditions (initial_CH4 : ℝ) (initial_C6H6 final_C6H5CH3 final_H2 : ℝ) : Prop :=
  balanced_reaction initial_C6H6 initial_CH4 final_C6H5CH3 final_H2 ∧ initial_CH4 = 3 ∧ final_H2 = 3

-- Theorem to prove
theorem moles_of_C6H5CH3_formed (initial_CH4 final_C6H5CH3 : ℝ) : reaction_conditions initial_CH4 3 final_C6H5CH3 3 → final_C6H5CH3 = 3 :=
by
  intros h
  sorry

end moles_of_C6H5CH3_formed_l2353_235370


namespace digit_150_of_17_div_70_is_2_l2353_235349

def repeating_cycle_17_div_70 : List ℕ := [2, 4, 2, 8, 5, 7, 1]

theorem digit_150_of_17_div_70_is_2 : 
  (repeating_cycle_17_div_70[(150 % 7) - 1] = 2) :=
by
  -- the proof will go here
  sorry

end digit_150_of_17_div_70_is_2_l2353_235349


namespace find_c_eq_3_l2353_235315

theorem find_c_eq_3 (m b c : ℝ) :
  (∀ x y, y = m * x + c → ((x = b + 4 ∧ y = 5) ∨ (x = -2 ∧ y = 2))) →
  c = 3 :=
by
  sorry

end find_c_eq_3_l2353_235315


namespace shells_in_afternoon_l2353_235334

-- Conditions: Lino picked up 292 shells in the morning and 616 shells in total.
def shells_in_morning : ℕ := 292
def total_shells : ℕ := 616

-- Theorem: The number of shells Lino picked up in the afternoon is 324.
theorem shells_in_afternoon : (total_shells - shells_in_morning) = 324 := 
by sorry

end shells_in_afternoon_l2353_235334


namespace solve_for_a_l2353_235372

theorem solve_for_a (a : ℝ) (h : a⁻¹ = (-1 : ℝ)^0) : a = 1 :=
sorry

end solve_for_a_l2353_235372


namespace car_collision_frequency_l2353_235371

theorem car_collision_frequency
  (x : ℝ)
  (h_collision : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * x)
  (h_big_crash : ∀ t : ℝ, t > 0 → ∃ n : ℕ, t = n * 20)
  (h_total_accidents : 240 / x + 240 / 20 = 36) :
  x = 10 :=
by
  sorry

end car_collision_frequency_l2353_235371


namespace trapezium_area_l2353_235319

theorem trapezium_area (a b h : ℝ) (h_a : a = 4) (h_b : b = 5) (h_h : h = 6) :
  (1 / 2 * (a + b) * h) = 27 :=
by
  rw [h_a, h_b, h_h]
  norm_num

end trapezium_area_l2353_235319


namespace copy_is_better_l2353_235304

variable (α : ℝ)

noncomputable def p_random : ℝ := 1 / 2
noncomputable def I_mistake : ℝ := α
noncomputable def p_caught : ℝ := 1 / 10
noncomputable def I_caught : ℝ := 3 * α
noncomputable def p_neighbor_wrong : ℝ := 1 / 5
noncomputable def p_not_caught : ℝ := 9 / 10

theorem copy_is_better (α : ℝ) : 
  (12 * α / 25) < (α / 2) := by
  -- Proof goes here
  sorry

end copy_is_better_l2353_235304


namespace reciprocal_of_fraction_l2353_235339

noncomputable def fraction := (Real.sqrt 5 + 1) / 2

theorem reciprocal_of_fraction :
  (fraction⁻¹) = (Real.sqrt 5 - 1) / 2 :=
by
  -- proof steps
  sorry

end reciprocal_of_fraction_l2353_235339


namespace points_earned_l2353_235311

-- Define the number of pounds required to earn one point
def pounds_per_point : ℕ := 4

-- Define the number of pounds Paige recycled
def paige_recycled : ℕ := 14

-- Define the number of pounds Paige's friends recycled
def friends_recycled : ℕ := 2

-- Define the total number of pounds recycled
def total_recycled : ℕ := paige_recycled + friends_recycled

-- Define the total number of points earned
def total_points : ℕ := total_recycled / pounds_per_point

-- Theorem to prove the total points earned
theorem points_earned : total_points = 4 := by
  sorry

end points_earned_l2353_235311


namespace triangle_area_l2353_235383

theorem triangle_area (a b c : ℝ)
    (h1 : Polynomial.eval a (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h2 : Polynomial.eval b (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (h3 : Polynomial.eval c (Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 8 * Polynomial.X^2 + Polynomial.C 10 * Polynomial.X - Polynomial.C 2) = 0)
    (sum_roots : a + b + c = 4)
    (sum_prod_roots : a * b + a * c + b * c = 5)
    (prod_roots : a * b * c = 1):
    Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c)) = 1 :=
  sorry

end triangle_area_l2353_235383


namespace scoops_arrangement_count_l2353_235362

theorem scoops_arrangement_count :
  (5 * 4 * 3 * 2 * 1 = 120) :=
by
  sorry

end scoops_arrangement_count_l2353_235362


namespace distance_to_town_l2353_235390

theorem distance_to_town (fuel_efficiency : ℝ) (fuel_used : ℝ) (distance : ℝ) : 
  fuel_efficiency = 70 / 10 → 
  fuel_used = 20 → 
  distance = fuel_efficiency * fuel_used → 
  distance = 140 :=
by
  intros
  sorry

end distance_to_town_l2353_235390


namespace solve_for_a_l2353_235343

theorem solve_for_a (x a : ℝ) (h : x = 5) (h_eq : a * x - 8 = 10 + 4 * a) : a = 18 :=
by
  sorry

end solve_for_a_l2353_235343


namespace sum_1_to_50_l2353_235335

-- Given conditions: initial values, and the loop increments
def initial_index : ℕ := 1
def initial_sum : ℕ := 0
def loop_condition (i : ℕ) : Prop := i ≤ 50

-- Increment step for index and running total in loop
def increment_index (i : ℕ) : ℕ := i + 1
def increment_sum (S : ℕ) (i : ℕ) : ℕ := S + i

-- Expected sum output for the given range
def sum_up_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Prove the sum of integers from 1 to 50
theorem sum_1_to_50 : sum_up_to_n 50 = 1275 := by
  sorry

end sum_1_to_50_l2353_235335


namespace two_numbers_with_difference_less_than_half_l2353_235359

theorem two_numbers_with_difference_less_than_half
  (x1 x2 x3 : ℝ)
  (h1 : 0 ≤ x1) (h2 : x1 < 1)
  (h3 : 0 ≤ x2) (h4 : x2 < 1)
  (h5 : 0 ≤ x3) (h6 : x3 < 1) :
  ∃ a b, 
    (a = x1 ∨ a = x2 ∨ a = x3) ∧
    (b = x1 ∨ b = x2 ∨ b = x3) ∧
    a ≠ b ∧ 
    |b - a| < 1 / 2 :=
sorry

end two_numbers_with_difference_less_than_half_l2353_235359


namespace simplify_fractions_l2353_235344

theorem simplify_fractions :
  (240 / 20) * (6 / 180) * (10 / 4) = 1 :=
by sorry

end simplify_fractions_l2353_235344


namespace expression_a_equals_half_expression_c_equals_half_l2353_235381

theorem expression_a_equals_half :
  (A : ℝ) = (1 / 2) :=
by
  let A := (Real.sqrt 2 / 2) * (Real.cos (15 * Real.pi / 180) - Real.sin (15 * Real.pi / 180))
  sorry

theorem expression_c_equals_half :
  (C : ℝ) = (1 / 2) :=
by
  let C := (Real.tan (22.5 * Real.pi / 180)) / (1 - (Real.tan (22.5 * Real.pi / 180))^2)
  sorry

end expression_a_equals_half_expression_c_equals_half_l2353_235381


namespace angle_Z_of_triangle_l2353_235341

theorem angle_Z_of_triangle (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : X + Y + Z = 180) : 
  Z = 90 := 
sorry

end angle_Z_of_triangle_l2353_235341


namespace upstream_swim_distance_l2353_235323

-- Definition of the speeds and distances
def downstream_speed (v : ℝ) := 5 + v
def upstream_speed (v : ℝ) := 5 - v
def distance := 54
def time := 6
def woman_speed_in_still_water := 5

-- Given condition: downstream_speed * time = distance
def downstream_condition (v : ℝ) := downstream_speed v * time = distance

-- Given condition: upstream distance is 'd' km
def upstream_distance (v : ℝ) := upstream_speed v * time

-- Prove that given the above conditions and solving the necessary equations, 
-- the distance swam upstream is 6 km.
theorem upstream_swim_distance {d : ℝ} (v : ℝ) (h1 : downstream_condition v) : upstream_distance v = 6 :=
by
  sorry

end upstream_swim_distance_l2353_235323


namespace strawberries_harvest_l2353_235326

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end strawberries_harvest_l2353_235326


namespace A_minus_3B_A_minus_3B_independent_of_y_l2353_235389

variables (x y : ℝ)
def A : ℝ := 3*x^2 - x + 2*y - 4*x*y
def B : ℝ := x^2 - 2*x - y + x*y - 5

theorem A_minus_3B (x y : ℝ) : A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15 :=
by
  sorry

theorem A_minus_3B_independent_of_y (x : ℝ) (hyp : ∀ y : ℝ, A x y - 3 * B x y = 5*x + 5*y - 7*x*y + 15) :
  5 - 7*x = 0 → x = 5 / 7 :=
by
  sorry

end A_minus_3B_A_minus_3B_independent_of_y_l2353_235389


namespace greatest_positive_integer_difference_l2353_235363

-- Define the conditions
def condition_x (x : ℝ) : Prop := 4 < x ∧ x < 6
def condition_y (y : ℝ) : Prop := 6 < y ∧ y < 10

-- Define the problem statement
theorem greatest_positive_integer_difference (x y : ℕ) (hx : condition_x x) (hy : condition_y y) : y - x = 4 :=
sorry

end greatest_positive_integer_difference_l2353_235363


namespace functional_equation_l2353_235375

theorem functional_equation 
  (f : ℝ → ℝ)
  (h1 : ∀ x y : ℝ, f (x * y) = f x * f y)
  (h2 : f 0 ≠ 0) :
  f 2009 = 1 :=
sorry

end functional_equation_l2353_235375


namespace tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l2353_235384

noncomputable def tangent_line (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_at_point_is_x_minus_y_plus_1_eq_0:
  tangent_line 0 = 1 →
  ∀ x y, y = tangent_line x → x - y + 1 = 0 → y = x * Real.exp x + 1 →
  x = 0 ∧ y = 1 → x - y + 1 = 0 :=
by
  intro h_point x y h_tangent h_eq h_coord
  sorry

end tangent_line_at_point_is_x_minus_y_plus_1_eq_0_l2353_235384


namespace algebra_expr_eval_l2353_235374

theorem algebra_expr_eval {x y : ℝ} (h : x - 2 * y = 3) : 5 - 2 * x + 4 * y = -1 :=
by sorry

end algebra_expr_eval_l2353_235374


namespace bank_record_withdrawal_l2353_235395

def deposit (x : ℤ) := x
def withdraw (x : ℤ) := -x

theorem bank_record_withdrawal : withdraw 500 = -500 :=
by
  sorry

end bank_record_withdrawal_l2353_235395


namespace eighth_term_of_arithmetic_sequence_l2353_235399

theorem eighth_term_of_arithmetic_sequence
  (a l : ℕ) (n : ℕ) (h₁ : a = 4) (h₂ : l = 88) (h₃ : n = 30) :
  (a + 7 * (l - a) / (n - 1) = (676 : ℚ) / 29) :=
by
  sorry

end eighth_term_of_arithmetic_sequence_l2353_235399


namespace percentage_difference_l2353_235366

variables (P P' : ℝ)

theorem percentage_difference (h : P' = 1.25 * P) :
  ((P' - P) / P') * 100 = 20 :=
by sorry

end percentage_difference_l2353_235366


namespace slant_heights_of_cones_l2353_235313

-- Define the initial conditions
variables (r r1 x y : Real)

-- Define the surface area condition
def surface_area_condition : Prop :=
  r * Real.sqrt (r ^ 2 + x ^ 2) + r ^ 2 = r1 * Real.sqrt (r1 ^ 2 + y ^ 2) + r1 ^ 2

-- Define the volume condition
def volume_condition : Prop :=
  r ^ 2 * Real.sqrt (x ^ 2 - r ^ 2) = r1 ^ 2 * Real.sqrt (y ^ 2 - r1 ^ 2)

-- Statement of the proof problem: Prove that the slant heights x and y are given by
theorem slant_heights_of_cones
  (h1 : surface_area_condition r r1 x y)
  (h2 : volume_condition r r1 x y) :
  x = (r ^ 2 + 2 * r1 ^ 2) / r ∧ y = (r1 ^ 2 + 2 * r ^ 2) / r1 := 
  sorry

end slant_heights_of_cones_l2353_235313


namespace consecutive_negative_integers_product_sum_l2353_235331

theorem consecutive_negative_integers_product_sum (n : ℤ) 
  (h_neg1 : n < 0) 
  (h_neg2 : n + 1 < 0) 
  (h_product : n * (n + 1) = 2720) :
  n + (n + 1) = -105 :=
sorry

end consecutive_negative_integers_product_sum_l2353_235331


namespace prime_condition_l2353_235379

theorem prime_condition (p : ℕ) (hp : Nat.Prime p) (h2p : Nat.Prime (p + 2)) : p = 3 ∨ 6 ∣ (p + 1) := 
sorry

end prime_condition_l2353_235379


namespace solve_for_y_l2353_235377

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 + 4 * x + 7 * y + 2 = 0) (h2 : 3 * x + 2 * y + 5 = 0) : 4 * y^2 + 33 * y + 11 = 0 :=
sorry

end solve_for_y_l2353_235377


namespace reverse_difference_198_l2353_235364

theorem reverse_difference_198 (a : ℤ) : 
  let N := 100 * (a - 1) + 10 * a + (a + 1)
  let M := 100 * (a + 1) + 10 * a + (a - 1)
  M - N = 198 := 
by
  sorry

end reverse_difference_198_l2353_235364


namespace arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l2353_235325

noncomputable def arsh (x : ℝ) := Real.log (x + Real.sqrt (x^2 + 1))
noncomputable def arch_pos (x : ℝ) := Real.log (x + Real.sqrt (x^2 - 1))
noncomputable def arch_neg (x : ℝ) := Real.log (x - Real.sqrt (x^2 - 1))
noncomputable def arth (x : ℝ) := (1 / 2) * Real.log ((1 + x) / (1 - x))

theorem arsh_eq (x : ℝ) : arsh x = Real.log (x + Real.sqrt (x^2 + 1)) := by
  sorry

theorem arch_pos_eq (x : ℝ) : arch_pos x = Real.log (x + Real.sqrt (x^2 - 1)) := by
  sorry

theorem arch_neg_eq (x : ℝ) : arch_neg x = Real.log (x - Real.sqrt (x^2 - 1)) := by
  sorry

theorem arth_eq (x : ℝ) : arth x = (1 / 2) * Real.log ((1 + x) / (1 - x)) := by
  sorry

end arsh_eq_arch_pos_eq_arch_neg_eq_arth_eq_l2353_235325


namespace solve_equation_l2353_235356

theorem solve_equation (x : ℝ) :
  (x^2 + 2*x + 1 = abs (3*x - 2)) ↔ 
  (x = (-7 + Real.sqrt 37) / 2) ∨ 
  (x = (-7 - Real.sqrt 37) / 2) :=
by
  sorry

end solve_equation_l2353_235356


namespace my_op_eq_l2353_235306

-- Define the custom operation
def my_op (m n : ℝ) : ℝ := m * n * (m - n)

-- State the theorem
theorem my_op_eq :
  ∀ (a b : ℝ), my_op (a + b) a = a^2 * b + a * b^2 :=
by intros a b; sorry

end my_op_eq_l2353_235306


namespace largest_divisor_of_462_and_231_l2353_235342

def is_factor (a b : ℕ) : Prop := a ∣ b

def largest_common_divisor (a b c : ℕ) : Prop :=
  is_factor c a ∧ is_factor c b ∧ (∀ d, (is_factor d a ∧ is_factor d b) → d ≤ c)

theorem largest_divisor_of_462_and_231 :
  largest_common_divisor 462 231 231 :=
by
  sorry

end largest_divisor_of_462_and_231_l2353_235342


namespace totalCerealInThreeBoxes_l2353_235392

def firstBox := 14
def secondBox := firstBox / 2
def thirdBox := secondBox + 5
def totalCereal := firstBox + secondBox + thirdBox

theorem totalCerealInThreeBoxes : totalCereal = 33 := 
by {
  sorry
}

end totalCerealInThreeBoxes_l2353_235392


namespace floor_div_eq_floor_floor_div_l2353_235303

theorem floor_div_eq_floor_floor_div (α : ℝ) (d : ℕ) (hα : 0 < α) :
  ⌊α / d⌋ = ⌊⌊α⌋ / d⌋ :=
by sorry

end floor_div_eq_floor_floor_div_l2353_235303


namespace number_of_factors_l2353_235353

theorem number_of_factors (K : ℕ) (hK : K = 2^4 * 3^3 * 5^2 * 7^1) : 
  ∃ n : ℕ, (∀ d e f g : ℕ, (0 ≤ d ∧ d ≤ 4) → (0 ≤ e ∧ e ≤ 3) → (0 ≤ f ∧ f ≤ 2) → (0 ≤ g ∧ g ≤ 1) → n = 120) :=
sorry

end number_of_factors_l2353_235353


namespace ratio_in_sequence_l2353_235369

theorem ratio_in_sequence (a1 a2 b1 b2 b3 : ℝ)
  (h1 : ∃ d, a1 = 1 + d ∧ a2 = 1 + 2 * d ∧ 9 = 1 + 3 * d)
  (h2 : ∃ r, b1 = 1 * r ∧ b2 = 1 * r^2 ∧ b3 = 1 * r^3 ∧ 9 = 1 * r^4) :
  b2 / (a1 + a2) = 3 / 10 := by
  sorry

end ratio_in_sequence_l2353_235369


namespace point_in_second_quadrant_l2353_235354

theorem point_in_second_quadrant (a : ℝ) :
  ∃ q : ℕ, q = 2 ∧ (-3 : ℝ) < 0 ∧ (a^2 + 1) > 0 := 
by sorry

end point_in_second_quadrant_l2353_235354


namespace cannot_determine_right_triangle_from_conditions_l2353_235330

-- Let triangle ABC have side lengths a, b, c opposite angles A, B, C respectively.
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Condition A: c^2 = a^2 - b^2 is rearranged to c^2 + b^2 = a^2 implying right triangle
def condition_A (a b c : ℝ) : Prop := c^2 = a^2 - b^2

-- Condition B: Triangle angles in the ratio A:B:C = 3:4:5 means not a right triangle
def condition_B : Prop := 
  let A := 45.0
  let B := 60.0
  let C := 75.0
  A ≠ 90.0 ∧ B ≠ 90.0 ∧ C ≠ 90.0

-- Condition C: Specific lengths 7, 24, 25 form a right triangle
def condition_C : Prop := 
  let a := 7.0
  let b := 24.0
  let c := 25.0
  is_right_triangle a b c

-- Condition D: A = B - C can be shown to always form at least one 90 degree angle, a right triangle
def condition_D (A B C : ℝ) : Prop := A = B - C ∧ (A + B + C = 180)

-- The actual mathematical proof that option B does not determine a right triangle
theorem cannot_determine_right_triangle_from_conditions :
  ∀ a b c (A B C : ℝ),
    (condition_A a b c → is_right_triangle a b c) ∧
    (condition_C → is_right_triangle 7 24 25) ∧
    (condition_D A B C → is_right_triangle a b c) ∧
    ¬condition_B :=
by
  sorry

end cannot_determine_right_triangle_from_conditions_l2353_235330


namespace Brad_has_9_green_balloons_l2353_235393

theorem Brad_has_9_green_balloons
  (total_balloons : ℕ)
  (red_balloons : ℕ)
  (green_balloons : ℕ)
  (h1 : total_balloons = 17)
  (h2 : red_balloons = 8)
  (h3 : total_balloons = red_balloons + green_balloons) :
  green_balloons = 9 := 
sorry

end Brad_has_9_green_balloons_l2353_235393


namespace area_of_rectangle_PQRS_l2353_235327

-- Definitions for the lengths of the sides of triangle ABC.
def AB : ℝ := 15
def AC : ℝ := 20
def BC : ℝ := 25

-- Definition for the length of PQ in rectangle PQRS.
def PQ : ℝ := 12

-- Definition for the condition that PQ is parallel to BC and RS is parallel to AB.
def PQ_parallel_BC : Prop := True
def RS_parallel_AB : Prop := True

-- The theorem to be proved: the area of rectangle PQRS is 115.2.
theorem area_of_rectangle_PQRS : 
  (∃ h: ℝ, h = (AC * PQ / BC) ∧ PQ * h = 115.2) :=
by {
  sorry
}

end area_of_rectangle_PQRS_l2353_235327


namespace ratio_of_45_and_9_l2353_235355

theorem ratio_of_45_and_9 : (45 / 9) = 5 := 
by
  sorry

end ratio_of_45_and_9_l2353_235355


namespace height_to_width_ratio_l2353_235336

theorem height_to_width_ratio (w h l : ℝ) (V : ℝ) (x : ℝ) :
  (h = x * w) →
  (l = 7 * h) →
  (V = l * w * h) →
  (V = 129024) →
  (w = 8) →
  (x = 6) :=
by
  intros h_eq_xw l_eq_7h V_eq_lwh V_val w_val
  -- Proof omitted
  sorry

end height_to_width_ratio_l2353_235336


namespace sequence_inequality_l2353_235348

theorem sequence_inequality (a : ℕ → ℝ) 
  (h₀ : a 0 = 5) 
  (h₁ : ∀ n, a (n + 1) * a n - a n ^ 2 = 1) : 
  35 < a 600 ∧ a 600 < 35.1 :=
sorry

end sequence_inequality_l2353_235348


namespace cost_of_one_pencil_l2353_235332

theorem cost_of_one_pencil (students : ℕ) (more_than_half : ℕ) (pencil_cost : ℕ) (pencils_each : ℕ)
  (total_cost : ℕ) (students_condition : students = 36) 
  (more_than_half_condition : more_than_half > 18) 
  (pencil_count_condition : pencils_each > 1) 
  (cost_condition : pencil_cost > pencils_each) 
  (total_cost_condition : students * pencil_cost * pencils_each = 1881) : 
  pencil_cost = 17 :=
sorry

end cost_of_one_pencil_l2353_235332


namespace min_value_of_f_l2353_235318

noncomputable def f (x : ℝ) := max (3 - x) (x^2 - 4*x + 3)

theorem min_value_of_f : ∃ x : ℝ, f x = -1 :=
by {
  use 2,
  sorry
}

end min_value_of_f_l2353_235318


namespace symmetric_point_y_axis_l2353_235380

theorem symmetric_point_y_axis (x y : ℝ) (hx : x = -3) (hy : y = 2) :
  (-x, y) = (3, 2) :=
by
  sorry

end symmetric_point_y_axis_l2353_235380


namespace prob_of_three_successes_correct_l2353_235300

noncomputable def prob_of_three_successes (p : ℝ) : ℝ :=
  (Nat.choose 10 3) * (p^3) * (1-p)^7

theorem prob_of_three_successes_correct (p : ℝ) :
  prob_of_three_successes p = (Nat.choose 10 3 : ℝ) * (p^3) * (1-p)^7 :=
by
  sorry

end prob_of_three_successes_correct_l2353_235300


namespace book_pages_total_l2353_235308

-- Definitions based on conditions
def pages_first_three_days: ℕ := 3 * 28
def pages_next_three_days: ℕ := 3 * 35
def pages_following_three_days: ℕ := 3 * 42
def pages_last_day: ℕ := 15

-- Total pages read calculated from above conditions
def total_pages_read: ℕ :=
  pages_first_three_days + pages_next_three_days + pages_following_three_days + pages_last_day

-- Proof problem statement: prove that the total pages read equal 330
theorem book_pages_total:
  total_pages_read = 330 :=
by
  sorry

end book_pages_total_l2353_235308


namespace usable_area_is_correct_l2353_235312

variable (x : ℝ)

def total_field_area : ℝ := (x + 9) * (x + 7)
def flooded_area : ℝ := (2 * x - 2) * (x - 1)
def usable_area : ℝ := total_field_area x - flooded_area x

theorem usable_area_is_correct : usable_area x = -x^2 + 20 * x + 61 :=
by
  sorry

end usable_area_is_correct_l2353_235312


namespace sum_of_tetrahedron_properties_eq_14_l2353_235396

-- Define the regular tetrahedron properties
def regular_tetrahedron_edges : ℕ := 6
def regular_tetrahedron_vertices : ℕ := 4
def regular_tetrahedron_faces : ℕ := 4

-- State the theorem that needs to be proven
theorem sum_of_tetrahedron_properties_eq_14 :
  regular_tetrahedron_edges + regular_tetrahedron_vertices + regular_tetrahedron_faces = 14 :=
by
  sorry

end sum_of_tetrahedron_properties_eq_14_l2353_235396


namespace natasha_average_speed_l2353_235347

theorem natasha_average_speed :
  (4 * 2.625 * 2) / (4 + 2) = 3.5 := 
by
  sorry

end natasha_average_speed_l2353_235347


namespace fraction_equality_l2353_235314

theorem fraction_equality (x y : ℚ) (hx : x = 4 / 7) (hy : y = 5 / 11) : 
  (7 * x + 11 * y) / (77 * x * y) = 9 / 20 :=
by
  -- proof can be provided here.
  sorry

end fraction_equality_l2353_235314


namespace race_time_l2353_235337

theorem race_time 
  (v t : ℝ)
  (h1 : 1000 = v * t)
  (h2 : 960 = v * (t + 10)) :
  t = 250 :=
by
  sorry

end race_time_l2353_235337


namespace factor_expression_l2353_235345

theorem factor_expression (x : ℤ) : 63 * x + 28 = 7 * (9 * x + 4) :=
by sorry

end factor_expression_l2353_235345


namespace fraction_sum_is_integer_l2353_235394

theorem fraction_sum_is_integer (n : ℤ) : 
  ∃ k : ℤ, (n / 3 + (n^2) / 2 + (n^3) / 6) = k := 
sorry

end fraction_sum_is_integer_l2353_235394


namespace positive_expression_l2353_235328

variable (a b c d : ℝ)

theorem positive_expression (ha : a < b) (hb : b < 0) (hc : 0 < c) (hd : c < d) : d - c - b - a > 0 := 
sorry

end positive_expression_l2353_235328


namespace sum_and_product_roots_l2353_235365

structure quadratic_data where
  m : ℝ
  n : ℝ

def roots_sum_eq (qd : quadratic_data) : Prop :=
  qd.m / 3 = 9

def roots_product_eq (qd : quadratic_data) : Prop :=
  qd.n / 3 = 20

theorem sum_and_product_roots (qd : quadratic_data) :
  roots_sum_eq qd → roots_product_eq qd → qd.m + qd.n = 87 := by
  sorry

end sum_and_product_roots_l2353_235365


namespace mistaken_divisor_l2353_235301

theorem mistaken_divisor (x : ℕ) (h1 : ∀ (d : ℕ), d ∣ 840 → d = 21 ∨ d = x) 
(h2 : 840 = 70 * x) : x = 12 := 
by sorry

end mistaken_divisor_l2353_235301


namespace chocolate_mixture_l2353_235360

theorem chocolate_mixture (x : ℝ) (h_initial : 110 / 220 = 0.5)
  (h_equation : (110 + x) / (220 + x) = 0.75) : x = 220 := by
  sorry

end chocolate_mixture_l2353_235360


namespace log_equation_solution_l2353_235307

theorem log_equation_solution {x : ℝ} (hx : x > 0) (hx1 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 4 :=
by 
  sorry

end log_equation_solution_l2353_235307


namespace tank_capacity_is_32_l2353_235338

noncomputable def capacity_of_tank (C : ℝ) : Prop :=
  (3/4) * C + 4 = (7/8) * C

theorem tank_capacity_is_32 : ∃ C : ℝ, capacity_of_tank C ∧ C = 32 :=
sorry

end tank_capacity_is_32_l2353_235338


namespace average_of_consecutive_integers_l2353_235317

theorem average_of_consecutive_integers (n m : ℕ) 
  (h1 : m = (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7) : 
  (n + 6) = (m + (m+1) + (m+2) + (m+3) + (m+4) + (m+5) + (m+6)) / 7 :=
by
  sorry

end average_of_consecutive_integers_l2353_235317


namespace h_value_l2353_235333

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ :=
  3 * x^2 + 9 * x + 20

-- State the desired form
def desired_form (x h k : ℝ) : ℝ :=
  3 * (x - h)^2 + k

-- Prove that h = -1.5
theorem h_value (h : ℝ) :
  ∃ k, (∀ x, quadratic_expr x = desired_form x h k) → h = -1.5 :=
by
  sorry

end h_value_l2353_235333


namespace necessary_but_not_sufficient_l2353_235388

theorem necessary_but_not_sufficient (x : ℝ) :
  (x^2 < x) → ((x^2 < x) ↔ (0 < x ∧ x < 1)) ∧ ((1/x > 2) ↔ (0 < x ∧ x < 1/2)) := 
by 
  sorry

end necessary_but_not_sufficient_l2353_235388


namespace square_area_l2353_235386

theorem square_area (y1 y2 y3 y4 : ℤ) 
  (h1 : y1 = 0) (h2 : y2 = 3) (h3 : y3 = 0) (h4 : y4 = -3) : 
  ∃ (area : ℤ), area = 36 :=
by
  sorry

end square_area_l2353_235386


namespace only_valid_pairs_l2353_235346

theorem only_valid_pairs (a b : ℕ) (h₁ : a ≥ 1) (h₂ : b ≥ 1) :
  a^b^2 = b^a ↔ (a = 1 ∧ b = 1) ∨ (a = 16 ∧ b = 2) ∨ (a = 27 ∧ b = 3) :=
by
  sorry

end only_valid_pairs_l2353_235346


namespace original_cost_l2353_235309

theorem original_cost (P : ℝ) (h : 0.76 * P = 608) : P = 800 :=
by
  sorry

end original_cost_l2353_235309


namespace symmetric_point_correct_l2353_235357

-- Define the point and line
def point : ℝ × ℝ := (-1, 2)
def line (x : ℝ) : ℝ := x - 1

-- Define a function that provides the symmetric point with respect to the line
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ :=
  -- Since this function is a critical part of the problem, we won't define it explicitly. Using a placeholder.
  sorry

-- The proof problem
theorem symmetric_point_correct : symmetric_point point line = (3, -2) :=
  sorry

end symmetric_point_correct_l2353_235357


namespace problem_lean_l2353_235302

theorem problem_lean (a : ℝ) (h : a - 1/a = 5) : a^2 + 1/a^2 = 27 := by
  sorry

end problem_lean_l2353_235302


namespace bisection_interval_l2353_235376

def f(x : ℝ) := x^3 - 2 * x - 5

theorem bisection_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 > 0 →
  ∃ a b : ℝ, a = 2 ∧ b = 2.5 ∧ f a * f b ≤ 0 :=
by
  sorry

end bisection_interval_l2353_235376


namespace cody_paid_amount_l2353_235350

/-- Cody buys $40 worth of stuff,
    the tax rate is 5%,
    he receives an $8 discount after taxes,
    and he and his friend split the final price equally.
    Prove that Cody paid $17. -/
theorem cody_paid_amount
  (initial_cost : ℝ)
  (tax_rate : ℝ)
  (discount : ℝ)
  (final_split : ℝ)
  (H1 : initial_cost = 40)
  (H2 : tax_rate = 0.05)
  (H3 : discount = 8)
  (H4 : final_split = 2) :
  (initial_cost * (1 + tax_rate) - discount) / final_split = 17 :=
by
  sorry

end cody_paid_amount_l2353_235350


namespace wall_height_l2353_235324

noncomputable def brickVolume (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def wallVolume (L W H : ℝ) : ℝ :=
  L * W * H

theorem wall_height (bricks_needed : ℝ) (brick_length_cm brick_width_cm brick_height_cm wall_length wall_width wall_height : ℝ)
  (H1 : bricks_needed = 4094.3396226415093)
  (H2 : brick_length_cm = 20)
  (H3 : brick_width_cm = 13.25)
  (H4 : brick_height_cm = 8)
  (H5 : wall_length = 7)
  (H6 : wall_width = 8)
  (H7 : brickVolume (brick_length_cm / 100) (brick_width_cm / 100) (brick_height_cm / 100) * bricks_needed = wallVolume wall_length wall_width wall_height) :
  wall_height = 0.155 :=
by
  sorry

end wall_height_l2353_235324


namespace prime_power_sum_l2353_235351

theorem prime_power_sum (a b p : ℕ) (hp : p = a ^ b + b ^ a) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (hp_prime : Nat.Prime p) : 
  p = 17 := 
sorry

end prime_power_sum_l2353_235351


namespace distinct_numbers_on_board_l2353_235305

def count_distinct_numbers (Mila_divisors : ℕ) (Zhenya_divisors : ℕ) (common : ℕ) : ℕ :=
  Mila_divisors + Zhenya_divisors - (common - 1)

theorem distinct_numbers_on_board :
  count_distinct_numbers 10 9 2 = 13 := by
  sorry

end distinct_numbers_on_board_l2353_235305


namespace fraction_to_terminating_decimal_l2353_235373

theorem fraction_to_terminating_decimal :
  (47 : ℚ) / (2^2 * 5^4) = 0.0188 :=
sorry

end fraction_to_terminating_decimal_l2353_235373


namespace triangle_perimeter_l2353_235340

theorem triangle_perimeter (r A : ℝ) (p : ℝ)
  (h1 : r = 2.5)
  (h2 : A = 25)
  (h3 : A = r * (p / 2)) : 
  p = 20 :=
by 
  sorry

end triangle_perimeter_l2353_235340


namespace rectangles_same_area_l2353_235329

theorem rectangles_same_area (x y : ℕ) 
  (h1 : x * y = (x + 4) * (y - 3)) 
  (h2 : x * y = (x + 8) * (y - 4)) : x + y = 10 := 
by
  sorry

end rectangles_same_area_l2353_235329


namespace proof_complement_U_A_l2353_235367

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A }

-- The theorem statement
theorem proof_complement_U_A :
  complement_U_A = {1, 5} :=
by
  -- Proof goes here
  sorry

end proof_complement_U_A_l2353_235367


namespace jillian_distance_l2353_235361

theorem jillian_distance : 
  ∀ (x y z : ℝ),
  (1 / 63) * x + (1 / 77) * y + (1 / 99) * z = 11 / 3 →
  (1 / 63) * z + (1 / 77) * y + (1 / 99) * x = 13 / 3 →
  x + y + z = 308 :=
by
  sorry

end jillian_distance_l2353_235361


namespace number_of_integer_solutions_is_zero_l2353_235321

-- Define the problem conditions
def eq1 (x y z : ℤ) : Prop := x^2 - 3 * x * y + 2 * y^2 - z^2 = 27
def eq2 (x y z : ℤ) : Prop := -x^2 + 6 * y * z + 2 * z^2 = 52
def eq3 (x y z : ℤ) : Prop := x^2 + x * y + 8 * z^2 = 110

-- State the theorem to be proved
theorem number_of_integer_solutions_is_zero :
  ∀ (x y z : ℤ), eq1 x y z → eq2 x y z → eq3 x y z → false :=
by
  sorry

end number_of_integer_solutions_is_zero_l2353_235321


namespace theater_ticket_difference_l2353_235322

theorem theater_ticket_difference
  (O B V : ℕ) 
  (h₁ : O + B + V = 550) 
  (h₂ : 15 * O + 10 * B + 20 * V = 8000) : 
  B - (O + V) = 370 := 
sorry

end theater_ticket_difference_l2353_235322


namespace painting_time_equation_l2353_235387

theorem painting_time_equation
  (Hannah_rate : ℝ)
  (Sarah_rate : ℝ)
  (combined_rate : ℝ)
  (temperature_factor : ℝ)
  (break_time : ℝ)
  (t : ℝ)
  (condition1 : Hannah_rate = 1 / 6)
  (condition2 : Sarah_rate = 1 / 8)
  (condition3 : combined_rate = (Hannah_rate + Sarah_rate) * temperature_factor)
  (condition4 : temperature_factor = 0.9)
  (condition5 : break_time = 1.5) :
  (combined_rate * (t - break_time) = 1) ↔ (t = 1 + break_time + 1 / combined_rate) :=
by
  sorry

end painting_time_equation_l2353_235387
