import Mathlib

namespace toms_total_score_l703_70360

def regular_enemy_points : ℕ := 10
def elite_enemy_points : ℕ := 25
def boss_enemy_points : ℕ := 50

def regular_enemy_bonus (kills : ℕ) : ℚ :=
  if 100 ≤ kills ∧ kills < 150 then 0.50
  else if 150 ≤ kills ∧ kills < 200 then 0.75
  else if kills ≥ 200 then 1.00
  else 0.00

def elite_enemy_bonus (kills : ℕ) : ℚ :=
  if 15 ≤ kills ∧ kills < 25 then 0.30
  else if 25 ≤ kills ∧ kills < 35 then 0.50
  else if kills >= 35 then 0.70
  else 0.00

def boss_enemy_bonus (kills : ℕ) : ℚ :=
  if 5 ≤ kills ∧ kills < 10 then 0.20
  else if kills ≥ 10 then 0.40
  else 0.00

noncomputable def total_score (regular_kills elite_kills boss_kills : ℕ) : ℚ :=
  let regular_points := regular_kills * regular_enemy_points
  let elite_points := elite_kills * elite_enemy_points
  let boss_points := boss_kills * boss_enemy_points
  let regular_total := regular_points + regular_points * regular_enemy_bonus regular_kills
  let elite_total := elite_points + elite_points * elite_enemy_bonus elite_kills
  let boss_total := boss_points + boss_points * boss_enemy_bonus boss_kills
  regular_total + elite_total + boss_total

theorem toms_total_score :
  total_score 160 20 8 = 3930 := by
  sorry

end toms_total_score_l703_70360


namespace minimum_value_inequality_l703_70389

open Real

theorem minimum_value_inequality
  (a b c : ℝ)
  (ha : 2 ≤ a) 
  (hb : a ≤ b)
  (hc : b ≤ c)
  (hd : c ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (5 / c - 1)^2 = 4 * (sqrt 5 ^ (1 / 4) - 1)^2 :=
sorry

end minimum_value_inequality_l703_70389


namespace steven_needs_more_seeds_l703_70326

def apple_seeds : Nat := 6
def pear_seeds : Nat := 2
def grape_seeds : Nat := 3
def apples_set_aside : Nat := 4
def pears_set_aside : Nat := 3
def grapes_set_aside : Nat := 9
def seeds_required : Nat := 60

theorem steven_needs_more_seeds : 
  seeds_required - (apples_set_aside * apple_seeds + pears_set_aside * pear_seeds + grapes_set_aside * grape_seeds) = 3 := by
  sorry

end steven_needs_more_seeds_l703_70326


namespace fraction_equiv_l703_70317

def repeating_decimal := 0.4 + (37 / 1000) / (1 - 1 / 1000)

theorem fraction_equiv : repeating_decimal = 43693 / 99900 :=
by
  sorry

end fraction_equiv_l703_70317


namespace number_of_students_passed_both_tests_l703_70379

theorem number_of_students_passed_both_tests 
  (total_students : ℕ) 
  (passed_long_jump : ℕ) 
  (passed_shot_put : ℕ) 
  (failed_both_tests : ℕ) 
  (students_with_union : ℕ := total_students) :
  (students_with_union = passed_long_jump + passed_shot_put - passed_both_tests + failed_both_tests) 
  → passed_both_tests = 25 :=
by sorry

end number_of_students_passed_both_tests_l703_70379


namespace correct_transformation_l703_70301

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0) : 
  (a / b) = ((a + 2 * a) / (b + 2 * b)) :=
by 
  sorry

end correct_transformation_l703_70301


namespace probability_of_x_plus_y_less_than_4_l703_70361

-- Define the square and the probability that x + y < 4 within this square.
theorem probability_of_x_plus_y_less_than_4 : 
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  (desired_area / square_area = 7 / 9) :=
by
  let square_area := (3 : ℝ) * (3 : ℝ)
  let excluded_triangle_area := 1/2 * (2 : ℝ) * (2 : ℝ)
  let desired_area := square_area - excluded_triangle_area
  show (desired_area / square_area = 7 / 9)
  sorry

end probability_of_x_plus_y_less_than_4_l703_70361


namespace range_of_set_l703_70339

-- Given conditions
variables {a b c : ℕ}
axiom mean_condition : (a + b + c) / 3 = 5
axiom median_condition : b = 5
axiom smallest_condition : a = 2

-- Definition of range
def range (x y z : ℕ) : ℕ := max (max x y) z - min (min x y) z

-- Theorem statement
theorem range_of_set : range a b c = 6 :=
by
  sorry

end range_of_set_l703_70339


namespace curve_passes_through_fixed_point_l703_70337

theorem curve_passes_through_fixed_point (k : ℝ) (x y : ℝ) (h : k ≠ -1) :
  (x ^ 2 + y ^ 2 + 2 * k * x + (4 * k + 10) * y + 10 * k + 20 = 0) → (x = 1 ∧ y = -3) :=
by
  sorry

end curve_passes_through_fixed_point_l703_70337


namespace find_plot_width_l703_70393

theorem find_plot_width:
  let length : ℝ := 360
  let area_acres : ℝ := 10
  let square_feet_per_acre : ℝ := 43560
  let area_square_feet := area_acres * square_feet_per_acre
  let width := area_square_feet / length
  area_square_feet = 435600 ∧ length = 360 ∧ square_feet_per_acre = 43560
  → width = 1210 :=
by
  intro h
  sorry

end find_plot_width_l703_70393


namespace sine_curve_transformation_l703_70382

theorem sine_curve_transformation (x y x' y' : ℝ) 
  (h1 : x' = (1 / 2) * x) 
  (h2 : y' = 3 * y) :
  (y = Real.sin x) ↔ (y' = 3 * Real.sin (2 * x')) := by 
  sorry

end sine_curve_transformation_l703_70382


namespace maximize_fraction_l703_70384

theorem maximize_fraction (A B C D : ℕ) (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A ≤ 9 ∧ B ≤ 9 ∧ C ≤ 9 ∧ D ≤ 9)
  (h_nonneg : 0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ 0 ≤ D)
  (h_integer : (A + B) % (C + D) = 0) : A + B = 17 :=
sorry

end maximize_fraction_l703_70384


namespace find_a_plus_b_plus_c_l703_70300

noncomputable def parabola_satisfies_conditions (a b c : ℝ) : Prop :=
  (∀ x, a * x ^ 2 + b * x + c ≥ 61) ∧
  (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = 0) ∧
  (a * (3:ℝ) ^ 2 + b * (3:ℝ) + c = 0)

theorem find_a_plus_b_plus_c (a b c : ℝ) 
  (h_minimum : parabola_satisfies_conditions a b c) :
  a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l703_70300


namespace problem_solution_l703_70302

theorem problem_solution :
  { x : ℝ // (x / 4 ≤ 3 + x) ∧ (3 + x < -3 * (1 + x)) } = { x : ℝ // x ∈ Set.Ico (-4 : ℝ) (-(3 / 2) : ℝ) } :=
by
  sorry

end problem_solution_l703_70302


namespace largest_integer_x_cubed_lt_three_x_squared_l703_70371

theorem largest_integer_x_cubed_lt_three_x_squared : 
  ∃ x : ℤ, x^3 < 3 * x^2 ∧ (∀ y : ℤ, y^3 < 3 * y^2 → y ≤ x) :=
  sorry

end largest_integer_x_cubed_lt_three_x_squared_l703_70371


namespace perpendicular_vectors_l703_70380

theorem perpendicular_vectors (a : ℝ) 
  (v1 : ℝ × ℝ := (4, -5))
  (v2 : ℝ × ℝ := (a, 2))
  (perpendicular : v1.fst * v2.fst + v1.snd * v2.snd = 0) :
  a = 5 / 2 :=
sorry

end perpendicular_vectors_l703_70380


namespace driving_hours_fresh_l703_70377

theorem driving_hours_fresh (x : ℚ) : (25 * x + 15 * (9 - x) = 152) → x = 17 / 10 :=
by
  intros h
  sorry

end driving_hours_fresh_l703_70377


namespace factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l703_70391

-- Math Proof Problem 1
theorem factorize_a_squared_minus_25 (a : ℝ) : a^2 - 25 = (a + 5) * (a - 5) :=
by
  sorry

-- Math Proof Problem 2
theorem factorize_2x_squared_y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 :=
by
  sorry

end factorize_a_squared_minus_25_factorize_2x_squared_y_minus_8xy_plus_8y_l703_70391


namespace problem_statement_l703_70386

theorem problem_statement (c d : ℤ) (h1 : 5 + c = 7 - d) (h2 : 6 + d = 10 + c) : 5 - c = 6 := 
by {
  sorry
}

end problem_statement_l703_70386


namespace proposition_q_must_be_true_l703_70316

theorem proposition_q_must_be_true (p q : Prop) (h1 : p ∨ q) (h2 : ¬ p) : q :=
by
  sorry

end proposition_q_must_be_true_l703_70316


namespace range_of_a_l703_70305

noncomputable def acute_angle_condition (a : ℝ) : Prop :=
  let M := (-2, 0)
  let N := (0, 2)
  let A := (-1, 1)
  (a > 0) ∧ (∀ P : ℝ × ℝ, (P.1 - a) ^ 2 + P.2 ^ 2 = 2 →
    (dist P A) > 2 * Real.sqrt 2)

theorem range_of_a (a : ℝ) : acute_angle_condition a ↔ a > Real.sqrt 7 - 1 :=
by sorry

end range_of_a_l703_70305


namespace maximum_combined_power_l703_70318

theorem maximum_combined_power (x1 x2 x3 : ℝ) (hx : x1 < 1 ∧ x2 < 1 ∧ x3 < 1) 
    (hcond : 2 * (x1 + x2 + x3) + 4 * (x1 * x2 * x3) = 3 * (x1 * x2 + x1 * x3 + x2 * x3) + 1) : 
    x1 + x2 + x3 ≤ 3 / 4 := 
sorry

end maximum_combined_power_l703_70318


namespace series_sum_solution_l703_70330

noncomputable def series_sum (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) : ℝ :=
  ∑' n : ℕ, (1 / ((n * c - (n - 1) * b) * ((n + 1) * c - n * b)))

theorem series_sum_solution (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a > b) (h₄ : b > c) :
  series_sum a b c h₀ h₁ h₂ h₃ h₄ = 1 / ((c - b) * c) := 
  sorry

end series_sum_solution_l703_70330


namespace correctness_check_l703_70303

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l703_70303


namespace solve_fraction_zero_l703_70363

theorem solve_fraction_zero (x : ℝ) (h1 : (x^2 - 16) / (4 - x) = 0) (h2 : 4 - x ≠ 0) : x = -4 :=
sorry

end solve_fraction_zero_l703_70363


namespace problem_statement_l703_70394

theorem problem_statement
  (a b c : ℝ) 
  (X : ℝ) 
  (hX : X = a + b + c + 2 * Real.sqrt (a^2 + b^2 + c^2 - a * b - b * c - c * a)) :
  X ≥ max (max (3 * a) (3 * b)) (3 * c) ∧ 
  ∃ (u v w : ℝ), 
    (u = Real.sqrt (X - 3 * a) ∧ v = Real.sqrt (X - 3 * b) ∧ w = Real.sqrt (X - 3 * c) ∧ 
     ((u + v = w) ∨ (v + w = u) ∨ (w + u = v))) :=
by
  sorry

end problem_statement_l703_70394


namespace parker_total_weight_l703_70354

-- Define the number of initial dumbbells and their weight
def initial_dumbbells := 4
def weight_per_dumbbell := 20

-- Define the number of additional dumbbells
def additional_dumbbells := 2

-- Define the total weight calculation
def total_weight := initial_dumbbells * weight_per_dumbbell + additional_dumbbells * weight_per_dumbbell

-- Prove that the total weight is 120 pounds
theorem parker_total_weight : total_weight = 120 :=
by
  -- proof skipped
  sorry

end parker_total_weight_l703_70354


namespace students_remaining_l703_70325

theorem students_remaining (n : ℕ) (h1 : n = 1000)
  (h_beach : n / 2 = 500)
  (h_home : (n - n / 2) / 2 = 250) :
  n - (n / 2 + (n - n / 2) / 2) = 250 :=
by
  sorry

end students_remaining_l703_70325


namespace math_problem_l703_70349

variable {x y z : ℝ}

def condition1 (x : ℝ) := x = 1.2 * 40
def condition2 (x y : ℝ) := y = x - 0.35 * x
def condition3 (x y z : ℝ) := z = (x + y) / 2

theorem math_problem (x y z : ℝ) (h1 : condition1 x) (h2 : condition2 x y) (h3 : condition3 x y z) :
  z = 39.6 :=
by
  sorry

end math_problem_l703_70349


namespace geometric_inequality_l703_70365

variable {q : ℝ} {b : ℕ → ℝ}

def geometric_sequence (b : ℕ → ℝ) (q : ℝ) : Prop := ∀ n : ℕ, b (n + 1) = b n * q

theorem geometric_inequality
  (h_geometric : geometric_sequence b q)
  (h_q_gt_one : q > 1)
  (h_pos : ∀ n : ℕ, b n > 0) :
  b 4 + b 8 > b 5 + b 7 :=
by
  sorry

end geometric_inequality_l703_70365


namespace cylinder_volume_ratio_l703_70395

variable (h r : ℝ)

theorem cylinder_volume_ratio (h r : ℝ) :
  let V_original := π * r^2 * h
  let h_new := 2 * h
  let r_new := 4 * r
  let V_new := π * (r_new)^2 * h_new
  V_new = 32 * V_original :=
by
  sorry

end cylinder_volume_ratio_l703_70395


namespace x_y_divisible_by_7_l703_70357

theorem x_y_divisible_by_7
  (x y a b : ℤ)
  (hx : 3 * x + 4 * y = a ^ 2)
  (hy : 4 * x + 3 * y = b ^ 2)
  (hx_pos : x > 0) (hy_pos : y > 0) :
  7 ∣ x ∧ 7 ∣ y :=
by
  sorry

end x_y_divisible_by_7_l703_70357


namespace pencil_groups_l703_70327

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end pencil_groups_l703_70327


namespace fourth_year_students_without_glasses_l703_70350

theorem fourth_year_students_without_glasses (total_students: ℕ) (x: ℕ) (y: ℕ) 
  (h1: total_students = 1152) 
  (h2: total_students = 8 * x - 32) 
  (h3: x = 148) 
  (h4: 2 * y + 10 = x) 
  : y = 69 :=
by {
sorry
}

end fourth_year_students_without_glasses_l703_70350


namespace complement_union_example_l703_70323

open Set

theorem complement_union_example :
  ∀ (U A B : Set ℕ), 
  U = {1, 2, 3, 4, 5, 6, 7, 8} → 
  A = {1, 3, 5, 7} → 
  B = {2, 4, 5} → 
  (U \ (A ∪ B)) = {6, 8} := by 
  intros U A B hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_union_example_l703_70323


namespace option_B_correct_l703_70358

-- Define the commutativity of multiplication
def commutativity_of_mul (a b : Nat) : Prop :=
  a * b = b * a

-- State the problem, which is to prove that 2ab + 3ba = 5ab given commutativity
theorem option_B_correct (a b : Nat) : commutativity_of_mul a b → 2 * (a * b) + 3 * (b * a) = 5 * (a * b) :=
by
  intro h_comm
  rw [←h_comm]
  sorry

end option_B_correct_l703_70358


namespace pyramid_volume_correct_l703_70383

noncomputable def pyramid_volume (A_PQRS A_PQT A_RST: ℝ) (side: ℝ) (height: ℝ) : ℝ :=
  (1 / 3) * A_PQRS * height

theorem pyramid_volume_correct 
  (A_PQRS : ℝ) (A_PQT : ℝ) (A_RST : ℝ) (side : ℝ) (height_PQT : ℝ) (height_RST : ℝ)
  (h_PQT : 2 * A_PQT / side = height_PQT)
  (h_RST : 2 * A_RST / side = height_RST)
  (eq1 : height_PQT^2 + side^2 = height_RST^2 + (side - height_PQT)^2) 
  (eq2 : height_RST^2 = height_PQT^2 + (height_PQT - side)^2)
  : pyramid_volume A_PQRS A_PQT A_RST = 5120 / 3 :=
by
  -- Skipping the proof steps
  sorry

end pyramid_volume_correct_l703_70383


namespace original_number_is_perfect_square_l703_70310

variable (n : ℕ)

theorem original_number_is_perfect_square
  (h1 : n = 1296)
  (h2 : ∃ m : ℕ, (n + 148) = m^2) : ∃ k : ℕ, n = k^2 :=
by
  sorry

end original_number_is_perfect_square_l703_70310


namespace noah_yearly_bill_l703_70367

-- Define the length of each call in minutes
def call_duration : ℕ := 30

-- Define the cost per minute in dollars
def cost_per_minute : ℝ := 0.05

-- Define the number of weeks in a year
def weeks_in_year : ℕ := 52

-- Define the cost per call in dollars
def cost_per_call : ℝ := call_duration * cost_per_minute

-- Define the total cost for a year in dollars
def yearly_cost : ℝ := cost_per_call * weeks_in_year

-- State the theorem
theorem noah_yearly_bill : yearly_cost = 78 := by
  -- Proof follows here
  sorry

end noah_yearly_bill_l703_70367


namespace trajectory_of_M_l703_70333

theorem trajectory_of_M
  (A : ℝ × ℝ := (3, 0))
  (P_circle : ∀ (P : ℝ × ℝ), P.1^2 + P.2^2 = 1)
  (M_midpoint : ∀ (P M : ℝ × ℝ), M = ((P.1 + 3) / 2, P.2 / 2) → M.1 = x ∧ M.2 = y) :
  (∀ (x y : ℝ), (x - 3/2)^2 + y^2 = 1/4) := 
sorry

end trajectory_of_M_l703_70333


namespace enclosed_area_eq_two_l703_70319

noncomputable def enclosed_area : ℝ :=
  -∫ x in (2 * Real.pi / 3)..Real.pi, (Real.sin x - Real.sqrt 3 * Real.cos x)

theorem enclosed_area_eq_two : enclosed_area = 2 := 
  sorry

end enclosed_area_eq_two_l703_70319


namespace beadshop_wednesday_profit_l703_70304

theorem beadshop_wednesday_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) :
  monday_fraction = 1/3 → tuesday_fraction = 1/4 → total_profit = 1200 →
  let monday_profit := monday_fraction * total_profit;
  let tuesday_profit := tuesday_fraction * total_profit;
  let wednesday_profit := total_profit - monday_profit - tuesday_profit;
  wednesday_profit = 500 :=
sorry

end beadshop_wednesday_profit_l703_70304


namespace admission_price_for_children_l703_70392

theorem admission_price_for_children (people_at_play : ℕ) (admission_price_adult : ℕ) (total_receipts : ℕ) (adults_attended : ℕ) 
  (h1 : people_at_play = 610) (h2 : admission_price_adult = 2) (h3 : total_receipts = 960) (h4 : adults_attended = 350) : 
  ∃ (admission_price_child : ℕ), admission_price_child = 1 :=
by
  sorry

end admission_price_for_children_l703_70392


namespace A_inter_B_l703_70387

-- Define the sets A and B
def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := { abs x | x ∈ A }

-- Statement of the theorem to be proven
theorem A_inter_B :
  A ∩ B = {0, 2} := 
by 
  sorry

end A_inter_B_l703_70387


namespace total_fruits_in_bowl_l703_70397

theorem total_fruits_in_bowl (bananas apples oranges : ℕ) 
  (h1 : bananas = 2) 
  (h2 : apples = 2 * bananas) 
  (h3 : oranges = 6) : 
  bananas + apples + oranges = 12 := 
by 
  sorry

end total_fruits_in_bowl_l703_70397


namespace Megan_full_folders_l703_70355

def initial_files : ℕ := 256
def deleted_files : ℕ := 67
def files_per_folder : ℕ := 12

def remaining_files : ℕ := initial_files - deleted_files
def number_of_folders : ℕ := remaining_files / files_per_folder

theorem Megan_full_folders : number_of_folders = 15 := by
  sorry

end Megan_full_folders_l703_70355


namespace prime_factors_difference_l703_70352

theorem prime_factors_difference (n : ℤ) (h₁ : n = 180181) : ∃ p q : ℤ, Prime p ∧ Prime q ∧ p > q ∧ n % p = 0 ∧ n % q = 0 ∧ (p - q) = 2 :=
by
  sorry

end prime_factors_difference_l703_70352


namespace harrys_total_cost_l703_70328

def cost_large_pizza : ℕ := 14
def cost_per_topping : ℕ := 2
def number_of_pizzas : ℕ := 2
def number_of_toppings_per_pizza : ℕ := 3
def tip_percentage : ℚ := 0.25

def total_cost (c_pizza c_topping tip_percent : ℚ) (n_pizza n_topping : ℕ) : ℚ :=
  let inital_cost := (c_pizza + c_topping * n_topping) * n_pizza
  let tip := inital_cost * tip_percent
  inital_cost + tip

theorem harrys_total_cost : total_cost 14 2 0.25 2 3 = 50 := 
  sorry

end harrys_total_cost_l703_70328


namespace parametric_curve_to_general_form_l703_70388

theorem parametric_curve_to_general_form :
  ∃ (a b c : ℚ), ∀ (t : ℝ), 
  (a = 8 / 225) ∧ (b = 4 / 75) ∧ (c = 1 / 25) ∧ 
  (a * (3 * Real.sin t)^2 + b * (3 * Real.sin t) * (5 * Real.cos t - 2 * Real.sin t) + c * (5 * Real.cos t - 2 * Real.sin t)^2 = 1) :=
by
  use 8 / 225, 4 / 75, 1 / 25
  sorry

end parametric_curve_to_general_form_l703_70388


namespace simplify_fraction_l703_70322

variable (a b y : ℝ)
variable (h1 : y = (a + 2 * b) / a)
variable (h2 : a ≠ -2 * b)
variable (h3 : a ≠ 0)

theorem simplify_fraction : (2 * a + 2 * b) / (a - 2 * b) = (y + 1) / (3 - y) :=
by
  sorry

end simplify_fraction_l703_70322


namespace good_numbers_count_l703_70345

theorem good_numbers_count : 
  ∃ (count : ℕ), 
    count = 10 ∧ 
    (∀ n : ℕ, 2020 % n = 22 ↔ (n ∣ 1998 ∧ n > 22)) :=
by {
  sorry
}

end good_numbers_count_l703_70345


namespace sum_of_repeating_decimal_digits_of_five_thirteenths_l703_70344

theorem sum_of_repeating_decimal_digits_of_five_thirteenths 
  (a b : ℕ)
  (h1 : 5 / 13 = (a * 10 + b) / 99)
  (h2 : (a * 10 + b) = 38) :
  a + b = 11 :=
sorry

end sum_of_repeating_decimal_digits_of_five_thirteenths_l703_70344


namespace find_original_selling_price_l703_70369

noncomputable def original_selling_price (purchase_price : ℝ) := 
  1.10 * purchase_price

noncomputable def new_selling_price (purchase_price : ℝ) := 
  1.17 * purchase_price

theorem find_original_selling_price (P : ℝ)
  (h1 : new_selling_price P - original_selling_price P = 56) :
  original_selling_price P = 880 := by 
  sorry

end find_original_selling_price_l703_70369


namespace polynomial_without_xy_l703_70308

theorem polynomial_without_xy (k : ℝ) (x y : ℝ) :
  ¬(∃ c : ℝ, (x^2 + k * x * y + 4 * x - 2 * x * y + y^2 - 1 = c * x * y)) → k = 2 := by
  sorry

end polynomial_without_xy_l703_70308


namespace smallest_positive_angle_l703_70320

theorem smallest_positive_angle (α : ℝ) (h : α = 2012) : ∃ β : ℝ, 0 < β ∧ β < 360 ∧ β = α % 360 := by
  sorry

end smallest_positive_angle_l703_70320


namespace average_of_possible_values_of_x_l703_70313

theorem average_of_possible_values_of_x (x : ℝ) (h : (2 * x^2 + 3) = 21) : (x = 3 ∨ x = -3) → (3 + -3) / 2 = 0 := by
  sorry

end average_of_possible_values_of_x_l703_70313


namespace liam_balloons_remainder_l703_70373

def balloons : Nat := 24 + 45 + 78 + 96
def friends : Nat := 10
def remainder := balloons % friends

theorem liam_balloons_remainder : remainder = 3 := by
  sorry

end liam_balloons_remainder_l703_70373


namespace basketball_weight_l703_70342

theorem basketball_weight (b s : ℝ) (h1 : s = 20) (h2 : 5 * b = 4 * s) : b = 16 :=
by
  sorry

end basketball_weight_l703_70342


namespace circle_symmetry_y_axis_eq_l703_70356

theorem circle_symmetry_y_axis_eq (x y : ℝ) :
  (x^2 + y^2 + 2 * x = 0) ↔ (x^2 + y^2 - 2 * x = 0) :=
sorry

end circle_symmetry_y_axis_eq_l703_70356


namespace Problem_statements_l703_70378

theorem Problem_statements (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = a * b) :
  (a + b ≥ 4) ∧
  ¬(a * b ≤ 4) ∧
  (a + 4 * b ≥ 9) ∧
  (1 / a ^ 2 + 2 / b ^ 2 ≥ 2 / 3) :=
by sorry

end Problem_statements_l703_70378


namespace bottle_caps_per_child_l703_70396

-- Define the conditions
def num_children : ℕ := 9
def total_bottle_caps : ℕ := 45

-- State the theorem that needs to be proved: each child has 5 bottle caps
theorem bottle_caps_per_child : (total_bottle_caps / num_children) = 5 := by
  sorry

end bottle_caps_per_child_l703_70396


namespace train_speed_l703_70351

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 700) (h_time : time = 40) : length / time = 17.5 :=
by
  -- length / time represents the speed of the train
  -- given length = 700 meters and time = 40 seconds
  -- we have to prove that 700 / 40 = 17.5
  sorry

end train_speed_l703_70351


namespace nested_fraction_l703_70338

theorem nested_fraction
  : 1 / (3 + 1 / (3 + 1 / (3 - 1 / (3 + 1 / (2 * (3 + 2 / 5))))))
  = 968 / 3191 := 
by
  sorry

end nested_fraction_l703_70338


namespace isabel_total_problems_l703_70364

theorem isabel_total_problems
  (math_pages : ℕ)
  (reading_pages : ℕ)
  (problems_per_page : ℕ)
  (h1 : math_pages = 2)
  (h2 : reading_pages = 4)
  (h3 : problems_per_page = 5) :
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end isabel_total_problems_l703_70364


namespace bankers_gain_is_126_l703_70336

-- Define the given conditions
def present_worth : ℝ := 600
def interest_rate : ℝ := 0.10
def time_period : ℕ := 2

-- Define the formula for compound interest to find the amount due A
def amount_due (PW : ℝ) (R : ℝ) (T : ℕ) : ℝ := PW * (1 + R) ^ T

-- Define the banker's gain as the difference between the amount due and the present worth
def bankers_gain (A : ℝ) (PW : ℝ) : ℝ := A - PW

-- The theorem to prove that the banker's gain is Rs. 126 given the conditions
theorem bankers_gain_is_126 : bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 126 := by
  sorry

end bankers_gain_is_126_l703_70336


namespace javier_total_time_spent_l703_70340

def outlining_time : ℕ := 30
def writing_time : ℕ := outlining_time + 28
def practicing_time : ℕ := writing_time / 2

theorem javier_total_time_spent : outlining_time + writing_time + practicing_time = 117 := by
  sorry

end javier_total_time_spent_l703_70340


namespace max_at_zero_l703_70374

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem max_at_zero : ∃ x, (∀ y, f y ≤ f x) ∧ x = 0 :=
by 
  sorry

end max_at_zero_l703_70374


namespace march_first_is_tuesday_l703_70335

theorem march_first_is_tuesday (march_15_tuesday : true) :
  true :=
sorry

end march_first_is_tuesday_l703_70335


namespace xy_sum_of_squares_l703_70324

theorem xy_sum_of_squares (x y : ℝ) (h1 : x - y = 5) (h2 : -x * y = 4) : x^2 + y^2 = 17 := 
sorry

end xy_sum_of_squares_l703_70324


namespace not_a_perfect_square_l703_70390

theorem not_a_perfect_square :
  ¬ (∃ x, (x: ℝ)^2 = 5^2025) :=
by
  sorry

end not_a_perfect_square_l703_70390


namespace apples_per_box_l703_70341

theorem apples_per_box (x : ℕ) (h1 : 10 * x > 0) (h2 : 3 * (10 * x) / 4 > 0) (h3 : (10 * x) / 4 = 750) : x = 300 :=
by
  sorry

end apples_per_box_l703_70341


namespace four_digit_numbers_count_l703_70368

open Nat

theorem four_digit_numbers_count :
  let valid_a := [5, 6]
  let valid_d := 0
  let valid_bc_pairs := [(3, 4), (3, 6)]
  valid_a.length * 1 * valid_bc_pairs.length = 4 :=
by
  sorry

end four_digit_numbers_count_l703_70368


namespace cubes_identity_l703_70332

theorem cubes_identity (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 40) : 
    a^3 + b^3 + c^3 - 3 * a * b * c = 1575 :=
by 
  sorry

end cubes_identity_l703_70332


namespace powers_greater_than_thresholds_l703_70376

theorem powers_greater_than_thresholds :
  (1.01^2778 > 1000000000000) ∧
  (1.001^27632 > 1000000000000) ∧
  (1.000001^27631000 > 1000000000000) ∧
  (1.01^4165 > 1000000000000000000) ∧
  (1.001^41447 > 1000000000000000000) ∧
  (1.000001^41446000 > 1000000000000000000) :=
by sorry

end powers_greater_than_thresholds_l703_70376


namespace split_numbers_cubic_l703_70346

theorem split_numbers_cubic (m : ℕ) (hm : 1 < m) (assumption : m^2 - m + 1 = 73) : m = 9 :=
sorry

end split_numbers_cubic_l703_70346


namespace min_value_of_xsquare_ysquare_l703_70372

variable {x y : ℝ}

theorem min_value_of_xsquare_ysquare (h : 5 * x^2 * y^2 + y^4 = 1) : x^2 + y^2 ≥ 4 / 5 :=
sorry

end min_value_of_xsquare_ysquare_l703_70372


namespace function_is_decreasing_on_R_l703_70334

def is_decreasing (a : ℝ) : Prop := a - 1 < 0

theorem function_is_decreasing_on_R (a : ℝ) : (1 < a ∧ a < 2) ↔ is_decreasing a :=
by
  sorry

end function_is_decreasing_on_R_l703_70334


namespace parrots_per_cage_l703_70375

theorem parrots_per_cage (P : ℕ) (parakeets_per_cage : ℕ) (cages : ℕ) (total_birds : ℕ) 
    (h1 : parakeets_per_cage = 7) (h2 : cages = 8) (h3 : total_birds = 72) 
    (h4 : total_birds = cages * P + cages * parakeets_per_cage) : 
    P = 2 :=
by
  sorry

end parrots_per_cage_l703_70375


namespace white_paint_amount_l703_70370

theorem white_paint_amount (total_paint green_paint brown_paint : ℕ) 
  (h_total : total_paint = 69)
  (h_green : green_paint = 15)
  (h_brown : brown_paint = 34) :
  total_paint - (green_paint + brown_paint) = 20 := by
  sorry

end white_paint_amount_l703_70370


namespace abs_eq_solution_diff_l703_70348

theorem abs_eq_solution_diff : 
  ∀ x₁ x₂ : ℝ, 
  (2 * x₁ - 3 = 18 ∨ 2 * x₁ - 3 = -18) → 
  (2 * x₂ - 3 = 18 ∨ 2 * x₂ - 3 = -18) → 
  |x₁ - x₂| = 18 :=
by
  sorry

end abs_eq_solution_diff_l703_70348


namespace sample_size_l703_70385

theorem sample_size (total_employees : ℕ) (male_employees : ℕ) (sampled_males : ℕ) (sample_size : ℕ) 
  (h1 : total_employees = 120) (h2 : male_employees = 80) (h3 : sampled_males = 24) : 
  sample_size = 36 :=
by
  sorry

end sample_size_l703_70385


namespace line_intersects_y_axis_at_point_intersection_at_y_axis_l703_70329

theorem line_intersects_y_axis_at_point :
  ∃ y, 5 * 0 - 7 * y = 35 := sorry

theorem intersection_at_y_axis :
  (∃ y, 5 * 0 - 7 * y = 35) → 0 - 7 * (-5) = 35 := sorry

end line_intersects_y_axis_at_point_intersection_at_y_axis_l703_70329


namespace proof_theorem_l703_70306

noncomputable def proof_problem 
  (m n : ℕ) 
  (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : Prop :=
0 ≤ x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ∧ 
x^p + y^p + z^p - x^m * y^n - y^m * z^n - z^m * x^n ≤ 1

theorem proof_theorem (m n : ℕ) (x y z : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 1) 
  (h3 : 0 ≤ y) (h4 : y ≤ 1) 
  (h5 : 0 ≤ z) (h6 : z ≤ 1) 
  (h7 : m > 0) (h8 : n > 0) 
  (h9 : m + n = p) : 
  proof_problem m n x y z h1 h2 h3 h4 h5 h6 h7 h8 h9 :=
by {
  sorry
}

end proof_theorem_l703_70306


namespace find_f_8_6_l703_70343

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem find_f_8_6 (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_def : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = - (1 / 2) * x) :
  f 8.6 = 0.3 :=
sorry

end find_f_8_6_l703_70343


namespace mass_of_barium_sulfate_l703_70314

-- Definitions of the chemical equation and molar masses
def barium_molar_mass : ℝ := 137.327
def sulfur_molar_mass : ℝ := 32.065
def oxygen_molar_mass : ℝ := 15.999
def molar_mass_BaSO4 : ℝ := barium_molar_mass + sulfur_molar_mass + 4 * oxygen_molar_mass

-- Given conditions
def moles_BaBr2 : ℝ := 4
def moles_BaSO4_produced : ℝ := moles_BaBr2 -- from balanced equation

-- Calculate mass of BaSO4 produced
def mass_BaSO4 : ℝ := moles_BaSO4_produced * molar_mass_BaSO4

-- Mass of Barium sulfate produced
theorem mass_of_barium_sulfate : mass_BaSO4 = 933.552 :=
by 
  -- Skip the proof
  sorry

end mass_of_barium_sulfate_l703_70314


namespace length_real_axis_hyperbola_l703_70331

theorem length_real_axis_hyperbola (a : ℝ) (h : a^2 = 4) : 2 * a = 4 := by
  sorry

end length_real_axis_hyperbola_l703_70331


namespace digit_A_unique_solution_l703_70381

theorem digit_A_unique_solution :
  ∃ (A : ℕ), 0 ≤ A ∧ A < 10 ∧ (100 * A + 72 - 23 = 549) ∧ A = 5 :=
by
  sorry

end digit_A_unique_solution_l703_70381


namespace find_missing_number_l703_70399

theorem find_missing_number :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 :=
by
  intros h1 h2
  sorry

end find_missing_number_l703_70399


namespace tan_alpha_value_complicated_expression_value_l703_70309

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) : 
  Real.tan α = -2 := by 
  sorry

theorem complicated_expression_value (α : ℝ) (h1 : Real.sin α = -2 * Real.sqrt 5 / 5) (h2 : Real.tan α < 0) (h3 : Real.tan α = -2) :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) / 
  (Real.cos (α - Real.pi / 2) - Real.sin (2 * Real.pi / 2 + α)) = -5 := by 
  sorry

end tan_alpha_value_complicated_expression_value_l703_70309


namespace projectile_height_30_in_2_seconds_l703_70321

theorem projectile_height_30_in_2_seconds (t y : ℝ) : 
  (y = -5 * t^2 + 25 * t ∧ y = 30) → t = 2 :=
by
  sorry

end projectile_height_30_in_2_seconds_l703_70321


namespace heartsuit_zero_heartsuit_self_heartsuit_pos_l703_70359

def heartsuit (x y : Real) : Real := x^2 - y^2

theorem heartsuit_zero (x : Real) : heartsuit x 0 = x^2 :=
by
  sorry

theorem heartsuit_self (x : Real) : heartsuit x x = 0 :=
by
  sorry

theorem heartsuit_pos (x y : Real) (h : x > y) : heartsuit x y > 0 :=
by
  sorry

end heartsuit_zero_heartsuit_self_heartsuit_pos_l703_70359


namespace profits_ratio_l703_70307

-- Definitions
def investment_ratio (p q : ℕ) := 7 * p = 5 * q
def investment_period_p := 10
def investment_period_q := 20

-- Prove the ratio of profits
theorem profits_ratio (p q : ℕ) (h1 : investment_ratio p q) :
  (7 * p * investment_period_p / (5 * q * investment_period_q)) = 7 / 10 :=
sorry

end profits_ratio_l703_70307


namespace find_other_number_l703_70362

theorem find_other_number (LCM HCF number1 number2 : ℕ) 
  (hLCM : LCM = 7700) 
  (hHCF : HCF = 11) 
  (hNumber1 : number1 = 308)
  (hProductEquality : number1 * number2 = LCM * HCF) :
  number2 = 275 :=
by
  -- proof omitted
  sorry

end find_other_number_l703_70362


namespace pqr_value_l703_70353

theorem pqr_value
  (p q r : ℤ) -- p, q, and r are integers
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) -- non-zero condition
  (h1 : p + q + r = 27) -- sum condition
  (h2 : (1 : ℚ) / p + (1 : ℚ) / q + (1 : ℚ) / r + 300 / (p * q * r) = 1) -- equation condition
  : p * q * r = 984 := 
sorry 

end pqr_value_l703_70353


namespace sequence_sum_zero_l703_70398

theorem sequence_sum_zero (n : ℕ) (h : n > 1) :
  (∃ (a : ℕ → ℤ), (∀ k : ℕ, k > 0 → a k ≠ 0) ∧ (∀ k : ℕ, k > 0 → a k + 2 * a (2 * k) + n * a (n * k) = 0)) ↔ n ≥ 3 := 
by sorry

end sequence_sum_zero_l703_70398


namespace least_alpha_condition_l703_70311

variables {a b α : ℝ}

theorem least_alpha_condition (a_gt_1 : a > 1) (b_gt_0 : b > 0) : 
  ∀ x, (x ≥ α) → (a + b) ^ x ≥ a ^ x + b ↔ α = 1 :=
by
  sorry

end least_alpha_condition_l703_70311


namespace agatha_amount_left_l703_70312

noncomputable def initial_amount : ℝ := 60
noncomputable def frame_cost : ℝ := 15 * (1 - 0.10)
noncomputable def wheel_cost : ℝ := 25 * (1 - 0.05)
noncomputable def seat_cost : ℝ := 8 * (1 - 0.15)
noncomputable def handlebar_tape_cost : ℝ := 5
noncomputable def bell_cost : ℝ := 3
noncomputable def hat_cost : ℝ := 10 * (1 - 0.25)

noncomputable def total_cost : ℝ :=
  frame_cost + wheel_cost + seat_cost + handlebar_tape_cost + bell_cost + hat_cost

noncomputable def amount_left : ℝ := initial_amount - total_cost

theorem agatha_amount_left : amount_left = 0.45 :=
by
  -- interim calculations would go here
  sorry

end agatha_amount_left_l703_70312


namespace positive_numbers_l703_70347

theorem positive_numbers {a b c : ℝ} (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end positive_numbers_l703_70347


namespace jane_can_buy_9_tickets_l703_70366

-- Definitions
def ticket_price : ℕ := 15
def jane_amount_initial : ℕ := 160
def scarf_cost : ℕ := 25
def jane_amount_after_scarf : ℕ := jane_amount_initial - scarf_cost
def max_tickets (amount : ℕ) (price : ℕ) := amount / price

-- The main statement
theorem jane_can_buy_9_tickets :
  max_tickets jane_amount_after_scarf ticket_price = 9 :=
by
  -- Proof goes here (proof steps would be outlined)
  sorry

end jane_can_buy_9_tickets_l703_70366


namespace right_triangle_short_leg_l703_70315

theorem right_triangle_short_leg (a b c : ℕ) (h : a^2 + b^2 = c^2) (h_c : c = 65) (h_int : ∃ x y z : ℕ, a = x ∧ b = y ∧ c = z) :
  a = 39 ∨ b = 39 :=
sorry

end right_triangle_short_leg_l703_70315
