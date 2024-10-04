import Mathlib

namespace correct_proposition_l62_62611

def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem correct_proposition : 
  (∀ x ∈ ℝ, f x = 4 * Real.sin (2 * x + Real.pi / 3)) ∧
  ∃ x : ℝ, f x = 0 ∧ (x = - Real.pi / 6) :=
by
  sorry

end correct_proposition_l62_62611


namespace complex_problem_l62_62739

theorem complex_problem (a b : ℝ) (h : (⟨a, 3⟩ : ℂ) + ⟨2, -1⟩ = ⟨5, b⟩) : a * b = 6 := by
  sorry

end complex_problem_l62_62739


namespace prove_inequality_l62_62241

variable (f : ℝ → ℝ)

-- Conditions
axiom condition : ∀ x y : ℝ, (f x + y) * (f y + x) > 0 → f x + y = f y + x

-- Proof of the desired statement
theorem prove_inequality : ∀ x y : ℝ, x > y → f x + y ≤ f y + x :=
by
  sorry

end prove_inequality_l62_62241


namespace arithmetic_expression_evaluation_l62_62606

theorem arithmetic_expression_evaluation :
  3^2 + 4 * 2 - 6 / 3 + 7 = 22 :=
by 
  -- Use tactics to break down the arithmetic expression evaluation (steps are abstracted)
  sorry

end arithmetic_expression_evaluation_l62_62606


namespace avg_age_second_group_l62_62821

theorem avg_age_second_group (avg_age_class : ℕ) (avg_age_first_group : ℕ) (age_15th_student : ℕ) (students_class : ℕ) (students_first_group : ℕ) (students_second_group : ℕ) :
  avg_age_class = 15 →
  avg_age_first_group = 14 →
  age_15th_student = 15 →
  students_class = 15 →
  students_first_group = 7 →
  students_second_group = 7 →
  let total_age_class := students_class * avg_age_class,
      total_age_first_group := students_first_group * avg_age_first_group,
      total_age_combined_groups := total_age_class - age_15th_student,
      total_age_second_group := total_age_combined_groups - total_age_first_group,
      avg_age_second_group := total_age_second_group / students_second_group
  in avg_age_second_group = 16 :=
by
  intros h1 h2 h3 h4 h5 h6,
  let total_age_class := students_class * avg_age_class,
  let total_age_first_group := students_first_group * avg_age_first_group,
  let total_age_combined_groups := total_age_class - age_15th_student,
  let total_age_second_group := total_age_combined_groups - total_age_first_group,
  let avg_age_second_group := total_age_second_group / students_second_group,
  rw [h1, h2, h3, h4, h5, h6] at *,
  exact calc 
    avg_age_second_group
    = (total_age_combined_groups - total_age_first_group) / students_second_group : rfl
    ... = 16 : by sorry

end avg_age_second_group_l62_62821


namespace valve_difference_l62_62558

theorem valve_difference (time_both : ℕ) (time_first : ℕ) (pool_capacity : ℕ) (V1 V2 diff : ℕ) :
  time_both = 48 → 
  time_first = 120 → 
  pool_capacity = 12000 → 
  V1 = pool_capacity / time_first → 
  V1 + V2 = pool_capacity / time_both → 
  diff = V2 - V1 → 
  diff = 50 :=
by sorry

end valve_difference_l62_62558


namespace factorize_expression_l62_62417

theorem factorize_expression (x : ℝ) : 
  x^3 - 5 * x^2 + 4 * x = x * (x - 1) * (x - 4) :=
by
  sorry

end factorize_expression_l62_62417


namespace hexagon_transformation_l62_62055

-- Define a shape composed of 36 identical small equilateral triangles
def Shape := { s : ℕ // s = 36 }

-- Define the number of triangles needed to form a hexagon
def TrianglesNeededForHexagon : ℕ := 18

-- Proof statement: Given a shape of 36 small triangles, we need 18 more triangles to form a hexagon
theorem hexagon_transformation (shape : Shape) : TrianglesNeededForHexagon = 18 :=
by
  -- This is our formalization of the problem statement which asserts
  -- that the transformation to a hexagon needs exactly 18 additional triangles.
  sorry

end hexagon_transformation_l62_62055


namespace set_union_proof_l62_62298

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end set_union_proof_l62_62298


namespace constant_term_binomial_expansion_n_6_middle_term_coefficient_l62_62151

open Nat

-- Define the binomial expansion term
def binomial_term (n : ℕ) (r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (2 ^ r) * x^(2 * (n-r) - r)

-- (I) Prove the constant term of the binomial expansion when n = 6
theorem constant_term_binomial_expansion_n_6 :
  binomial_term 6 4 (1 : ℝ) = 240 := 
sorry

-- (II) Prove the coefficient of the middle term under given conditions
theorem middle_term_coefficient (n : ℕ) :
  (Nat.choose 8 2 = Nat.choose 8 6) →
  binomial_term 8 4 (1 : ℝ) = 1120 := 
sorry

end constant_term_binomial_expansion_n_6_middle_term_coefficient_l62_62151


namespace area_of_mirror_l62_62944

theorem area_of_mirror (outer_width : ℝ) (outer_height : ℝ) (frame_width : ℝ) (mirror_area : ℝ) :
  outer_width = 70 → outer_height = 100 → frame_width = 15 → mirror_area = (outer_width - 2 * frame_width) * (outer_height - 2 * frame_width) → mirror_area = 2800 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  rw [h4]
  sorry

end area_of_mirror_l62_62944


namespace meat_sales_beyond_plan_l62_62809

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end meat_sales_beyond_plan_l62_62809


namespace sufficient_but_not_necessary_for_ax_square_pos_l62_62153

variables (a x : ℝ)

theorem sufficient_but_not_necessary_for_ax_square_pos (h : a > 0) : 
  (a > 0 → ax^2 > 0) ∧ ((ax^2 > 0) → a > 0) :=
sorry

end sufficient_but_not_necessary_for_ax_square_pos_l62_62153


namespace tan_subtraction_l62_62738

variable {α β : ℝ}

theorem tan_subtraction (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) : Real.tan (α - β) = -1 := by
  sorry

end tan_subtraction_l62_62738


namespace initial_calculated_average_l62_62965

theorem initial_calculated_average (S : ℕ) (initial_average correct_average : ℕ) (num_wrongly_read correctly_read wrong_value correct_value : ℕ)
    (h1 : num_wrongly_read = 36) 
    (h2 : correctly_read = 26) 
    (h3 : correct_value = 6)
    (h4 : S = 10 * correct_value) :
    initial_average = (S - (num_wrongly_read - correctly_read)) / 10 → initial_average = 5 :=
sorry

end initial_calculated_average_l62_62965


namespace solve_equation_l62_62838

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end solve_equation_l62_62838


namespace input_for_output_16_l62_62351

theorem input_for_output_16 (x : ℝ) (y : ℝ) : 
  (y = (if x < 0 then (x + 1)^2 else (x - 1)^2)) → 
  y = 16 → 
  (x = 5 ∨ x = -5) :=
by sorry

end input_for_output_16_l62_62351


namespace clients_select_two_cars_l62_62591

theorem clients_select_two_cars (cars clients selections : ℕ) (total_selections : ℕ)
  (h1 : cars = 10) (h2 : clients = 15) (h3 : total_selections = cars * 3) (h4 : total_selections = clients * selections) :
  selections = 2 :=
by 
  sorry

end clients_select_two_cars_l62_62591


namespace liquid_level_ratio_l62_62356

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end liquid_level_ratio_l62_62356


namespace system1_solution_system2_solution_l62_62336

theorem system1_solution (x y : ℤ) 
  (h1 : x = 2 * y - 1) 
  (h2 : 3 * x + 4 * y = 17) : 
  x = 3 ∧ y = 2 :=
by 
  sorry

theorem system2_solution (x y : ℤ) 
  (h1 : 2 * x - y = 0) 
  (h2 : 3 * x - 2 * y = 5) : 
  x = -5 ∧ y = -10 := 
by 
  sorry

end system1_solution_system2_solution_l62_62336


namespace fraction_value_eq_l62_62540

theorem fraction_value_eq : (5 * 8) / 10 = 4 := 
by 
  sorry

end fraction_value_eq_l62_62540


namespace probability_different_topics_l62_62582

theorem probability_different_topics (topics : ℕ) (h : topics = 6) : 
  let total_combinations := topics * topics,
      different_combinations := topics * (topics - 1) 
  in (different_combinations / total_combinations : ℚ) = 5 / 6 :=
by
  -- This is just a place holder proof.
  sorry

end probability_different_topics_l62_62582


namespace total_profit_proof_l62_62530
-- Import the necessary libraries

-- Define the investments and profits
def investment_tom : ℕ := 3000 * 12
def investment_jose : ℕ := 4500 * 10
def profit_jose : ℕ := 3500

-- Define the ratio and profit parts
def ratio_tom : ℕ := investment_tom / Nat.gcd investment_tom investment_jose
def ratio_jose : ℕ := investment_jose / Nat.gcd investment_tom investment_jose
def ratio_total : ℕ := ratio_tom + ratio_jose
def one_part_value : ℕ := profit_jose / ratio_jose
def profit_tom : ℕ := ratio_tom * one_part_value

-- The total profit
def total_profit : ℕ := profit_tom + profit_jose

-- The theorem to prove
theorem total_profit_proof : total_profit = 6300 := by
  sorry

end total_profit_proof_l62_62530


namespace part1_part2_l62_62164

def f (x a : ℝ) := x^2 + 4 * a * x + 2 * a + 6

theorem part1 (a : ℝ) : (∃ x : ℝ, f x a = 0) ↔ (a = -1 ∨ a = 3 / 2) := 
by 
  sorry

def g (a : ℝ) := 2 - a * |a + 3|

theorem part2 (a : ℝ) :
  (-1 ≤ a ∧ a ≤ 3 / 2) →
  -19 / 4 ≤ g a ∧ g a ≤ 4 :=
by 
  sorry

end part1_part2_l62_62164


namespace inverse_square_relationship_l62_62232

theorem inverse_square_relationship (k : ℝ) (y : ℝ) (h1 : ∀ x y, x = k / y^2)
  (h2 : ∃ y, 1 = k / y^2) (h3 : 0.5625 = k / 4^2) :
  ∃ y, 1 = 9 / y^2 ∧ y = 3 :=
by
  sorry

end inverse_square_relationship_l62_62232


namespace tablet_screen_area_difference_l62_62370

theorem tablet_screen_area_difference (d1 d2 : ℝ) (A1 A2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 7) :
  A1 - A2 = 7.5 :=
by
  -- Note: The proof is omitted as the prompt requires only the statement.
  sorry

end tablet_screen_area_difference_l62_62370


namespace deduce_pi_from_cylinder_volume_l62_62830

theorem deduce_pi_from_cylinder_volume 
  (C h V : ℝ) 
  (Circumference : C = 20) 
  (Height : h = 11)
  (VolumeFormula : V = (1 / 12) * C^2 * h) : 
  pi = 3 :=
by 
  -- Carry out the proof
  sorry

end deduce_pi_from_cylinder_volume_l62_62830


namespace train_speed_in_km_per_hr_l62_62250

def train_length : ℝ := 116.67 -- length of the train in meters
def crossing_time : ℝ := 7 -- time to cross the pole in seconds

theorem train_speed_in_km_per_hr : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by
  sorry

end train_speed_in_km_per_hr_l62_62250


namespace arithmetic_sequence_binomial_l62_62564

theorem arithmetic_sequence_binomial {n k u : ℕ} (h₁ : u ≥ 3)
    (h₂ : n = u^2 - 2)
    (h₃ : k = Nat.choose u 2 - 1 ∨ k = Nat.choose (u + 1) 2 - 1)
    : (Nat.choose n (k - 1)) - 2 * (Nat.choose n k) + (Nat.choose n (k + 1)) = 0 :=
by
  sorry

end arithmetic_sequence_binomial_l62_62564


namespace cubic_function_properties_l62_62432

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 9 * x

theorem cubic_function_properties :
  (∀ (x : ℝ), deriv f x = 3 * x^2 - 12 * x + 9) ∧
  (f 1 = 4) ∧ 
  (deriv f 1 = 0) ∧
  (f 3 = 0) ∧ 
  (deriv f 3 = 0) ∧
  (f 0 = 0) :=
by
  sorry

end cubic_function_properties_l62_62432


namespace debate_students_handshake_l62_62219

theorem debate_students_handshake 
    (S1 S2 S3 : ℕ)
    (h1 : S1 = 2 * S2)
    (h2 : S2 = S3 + 40)
    (h3 : S3 = 200) :
    S1 + S2 + S3 = 920 :=
by
  sorry

end debate_students_handshake_l62_62219


namespace tens_digit_of_square_ending_in_six_odd_l62_62175

theorem tens_digit_of_square_ending_in_six_odd 
   (N : ℤ) 
   (a : ℤ) 
   (b : ℕ) 
   (hle : 0 ≤ b) 
   (hge : b < 10) 
   (hexp : N = 10 * a + b) 
   (hsqr : (N^2) % 10 = 6) : 
   ∃ k : ℕ, (N^2 / 10) % 10 = 2 * k + 1 :=
sorry -- Proof goes here

end tens_digit_of_square_ending_in_six_odd_l62_62175


namespace pies_sold_in_a_week_l62_62245

theorem pies_sold_in_a_week : 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 114 :=
by 
  let Monday := 8
  let Tuesday := 12
  let Wednesday := 14
  let Thursday := 20
  let Friday := 20
  let Saturday := 20
  let Sunday := 20
  have h1 : Monday + Tuesday + Wednesday + Thursday + Friday + Saturday + Sunday = 8 + 12 + 14 + 20 + 20 + 20 + 20 := by rfl
  have h2 : 8 + 12 + 14 + 20 + 20 + 20 + 20 = 114 := by norm_num
  exact h1.trans h2

end pies_sold_in_a_week_l62_62245


namespace point_in_third_quadrant_l62_62785

theorem point_in_third_quadrant (x y : ℝ) (h1 : x = -3) (h2 : y = -2) : 
  x < 0 ∧ y < 0 :=
by
  sorry

end point_in_third_quadrant_l62_62785


namespace coins_distribution_l62_62217

theorem coins_distribution :
  ∃ (x y z : ℕ), x + y + z = 1000 ∧ x + 2 * y + 5 * z = 2000 ∧ Nat.Prime x ∧ x = 3 ∧ y = 996 ∧ z = 1 :=
by
  sorry

end coins_distribution_l62_62217


namespace bella_truck_stamps_more_l62_62728

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end bella_truck_stamps_more_l62_62728


namespace value_of_expression_is_one_l62_62646

theorem value_of_expression_is_one : 
  ∃ (a b c d : ℚ), (a = 1) ∧ (b = -1) ∧ (c = 0) ∧ (d = 1 ∨ d = -1) ∧ (a - b + c^2 - |d| = 1) :=
by
  sorry

end value_of_expression_is_one_l62_62646


namespace percentage_favoring_all_three_l62_62780

variable (A B C A_union_B_union_C Y X : ℝ)

-- Conditions
axiom hA : A = 0.50
axiom hB : B = 0.30
axiom hC : C = 0.20
axiom hA_union_B_union_C : A_union_B_union_C = 0.78
axiom hY : Y = 0.17

-- Question: Prove that the percentage of those asked favoring all three proposals is 5%
theorem percentage_favoring_all_three :
  A = 0.50 → B = 0.30 → C = 0.20 →
  A_union_B_union_C = 0.78 →
  Y = 0.17 →
  X = 0.05 :=
by
  intros
  sorry

end percentage_favoring_all_three_l62_62780


namespace polynomial_factorization_l62_62364

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := 
by sorry

end polynomial_factorization_l62_62364


namespace bird_families_left_near_mountain_l62_62367

def total_bird_families : ℕ := 85
def bird_families_flew_to_africa : ℕ := 23
def bird_families_flew_to_asia : ℕ := 37

theorem bird_families_left_near_mountain : total_bird_families - (bird_families_flew_to_africa + bird_families_flew_to_asia) = 25 := by
  sorry

end bird_families_left_near_mountain_l62_62367


namespace matrix_corner_sum_eq_l62_62308

theorem matrix_corner_sum_eq (M : Matrix (Fin 2000) (Fin 2000) ℤ)
  (h : ∀ i j : Fin 1999, M i j + M (i+1) (j+1) = M i (j+1) + M (i+1) j) :
  M 0 0 + M 1999 1999 = M 0 1999 + M 1999 0 :=
sorry

end matrix_corner_sum_eq_l62_62308


namespace Tobias_monthly_allowance_l62_62529

noncomputable def monthly_allowance (shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways : ℕ) : ℕ :=
  (shoes_cost + change - (num_lawns * lawn_charge + num_driveways * driveway_charge)) / monthly_saving_period

theorem Tobias_monthly_allowance :
  let shoes_cost := 95
  let monthly_saving_period := 3
  let lawn_charge := 15
  let driveway_charge := 7
  let change := 15
  let num_lawns := 4
  let num_driveways := 5
  monthly_allowance shoes_cost monthly_saving_period lawn_charge driveway_charge change num_lawns num_driveways = 5 :=
by
  sorry

end Tobias_monthly_allowance_l62_62529


namespace n_is_prime_or_power_of_2_l62_62659

noncomputable def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

noncomputable def is_power_of_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 ^ k

noncomputable def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem n_is_prime_or_power_of_2 {n : ℕ} (h1 : n > 6)
  (h2 : ∃ (a : ℕ → ℕ) (k : ℕ), 
    (∀ i : ℕ, i < k → a i < n ∧ coprime (a i) n) ∧ 
    (∀ i : ℕ, 1 ≤ i → i < k → a (i + 1) - a i = a 2 - a 1)) 
  : is_prime n ∨ is_power_of_2 n := 
sorry

end n_is_prime_or_power_of_2_l62_62659


namespace cheese_wedge_volume_l62_62374

theorem cheese_wedge_volume (r h : ℝ) (n : ℕ) (V : ℝ) (π : ℝ) 
: r = 8 → h = 10 → n = 3 → V = π * r^2 * h → V / n = (640 * π) / 3  :=
by
  intros r_eq h_eq n_eq V_eq
  rw [r_eq, h_eq] at V_eq
  rw [V_eq]
  sorry

end cheese_wedge_volume_l62_62374


namespace fractions_order_l62_62365

theorem fractions_order :
  let f1 := (18 : ℚ) / 14
  let f2 := (16 : ℚ) / 12
  let f3 := (20 : ℚ) / 16
  f3 < f1 ∧ f1 < f2 :=
by {
  sorry
}

end fractions_order_l62_62365


namespace find_s_l62_62339

variable {t s : Real}

theorem find_s (h1 : t = 8 * s^2) (h2 : t = 4) : s = Real.sqrt 2 / 2 :=
by
  sorry

end find_s_l62_62339


namespace length_of_cable_l62_62015

-- Conditions
def condition1 (x y z : ℝ) : Prop := x + y + z = 8
def condition2 (x y z : ℝ) : Prop := x * y + y * z + x * z = -18

-- Conclusion we want to prove
theorem length_of_cable (x y z : ℝ) (h1 : condition1 x y z) (h2 : condition2 x y z) :
  4 * π * Real.sqrt (59 / 3) = 4 * π * (Real.sqrt ((x^2 + y^2 + z^2 - ((x + y + z)^2 - 4*(x*y + y*z + x*z))) / 3)) :=
sorry

end length_of_cable_l62_62015


namespace larger_square_side_length_l62_62836

theorem larger_square_side_length :
  ∃ (a : ℕ), ∃ (b : ℕ), a^2 = b^2 + 2001 ∧ (a = 1001 ∨ a = 335 ∨ a = 55 ∨ a = 49) :=
by
  sorry

end larger_square_side_length_l62_62836


namespace g_g_g_g_3_l62_62943

def g (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 else 5 * x + 3

theorem g_g_g_g_3 : g (g (g (g 3))) = 24 := by
  sorry

end g_g_g_g_3_l62_62943


namespace tomato_plant_relationship_l62_62654

theorem tomato_plant_relationship :
  ∃ (T1 T2 T3 : ℕ), T1 = 24 ∧ T3 = T2 + 2 ∧ T1 + T2 + T3 = 60 ∧ T1 - T2 = 7 :=
by
  sorry

end tomato_plant_relationship_l62_62654


namespace sum_of_numbers_l62_62369

theorem sum_of_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 149)
  (h2 : ab + bc + ca = 70) : 
  a + b + c = 17 :=
sorry

end sum_of_numbers_l62_62369


namespace ratio_of_numbers_l62_62354

theorem ratio_of_numbers (A B : ℕ) (h_lcm : Nat.lcm A B = 48) (h_hcf : Nat.gcd A B = 4) : A / 4 = 3 ∧ B / 4 = 4 :=
sorry

end ratio_of_numbers_l62_62354


namespace thabo_HNF_calculation_l62_62052

variable (THABO_BOOKS : ℕ)

-- Conditions as definitions
def total_books : ℕ := 500
def fiction_books : ℕ := total_books * 40 / 100
def non_fiction_books : ℕ := total_books * 60 / 100
def paperback_non_fiction_books (HNF : ℕ) : ℕ := HNF + 50
def total_non_fiction_books (HNF : ℕ) : ℕ := HNF + paperback_non_fiction_books HNF

-- Lean statement to prove
theorem thabo_HNF_calculation (HNF : ℕ) :
  total_books = 500 →
  fiction_books = 200 →
  non_fiction_books = 300 →
  total_non_fiction_books HNF = 300 →
  2 * HNF + 50 = 300 →
  HNF = 125 :=
by
  intros _
         _
         _
         _
         _
  sorry

end thabo_HNF_calculation_l62_62052


namespace worker_hourly_rate_l62_62252

theorem worker_hourly_rate (x : ℝ) (h1 : 8 * 0.90 = 7.20) (h2 : 42 * x + 7.20 = 32.40) : x = 0.60 :=
by
  sorry

end worker_hourly_rate_l62_62252


namespace perimeter_of_region_l62_62678

-- Define the conditions as Lean definitions
def area_of_region (a : ℝ) := a = 400
def number_of_squares (n : ℕ) := n = 8
def arrangement := "2x4 rectangle"

-- Define the statement we need to prove
theorem perimeter_of_region (a : ℝ) (n : ℕ) (s : ℝ) 
  (h_area_region : area_of_region a) 
  (h_number_of_squares : number_of_squares n) 
  (h_arrangement : arrangement = "2x4 rectangle")
  (h_area_one_square : a / n = s^2) :
  4 * 10 * (s) = 80 * 2^(1/2)  :=
by sorry

end perimeter_of_region_l62_62678


namespace cows_total_l62_62218

theorem cows_total (M F : ℕ) 
  (h1 : F = 2 * M) 
  (h2 : F / 2 = M / 2 + 50) : 
  M + F = 300 :=
by
  sorry

end cows_total_l62_62218


namespace velocity_of_current_correct_l62_62379

-- Definitions based on the conditions in the problem
def rowing_speed_in_still_water : ℝ := 10
def distance_to_place : ℝ := 24
def total_time_round_trip : ℝ := 5

-- Define the velocity of the current
def velocity_of_current : ℝ := 2

-- Main theorem statement
theorem velocity_of_current_correct :
  ∃ (v : ℝ), (v = 2) ∧ 
  (total_time_round_trip = (distance_to_place / (rowing_speed_in_still_water + v) + 
                            distance_to_place / (rowing_speed_in_still_water - v))) :=
by {
  sorry
}

end velocity_of_current_correct_l62_62379


namespace last_two_digits_sum_of_factorials_1_to_15_l62_62988

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l62_62988


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l62_62992

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l62_62992


namespace triangle_square_side_ratio_l62_62603

theorem triangle_square_side_ratio :
  (∀ (a : ℝ), (a * 3 = 60) → (∀ (b : ℝ), (b * 4 = 60) → (a / b = 4 / 3))) :=
by
  intros a h1 b h2
  sorry

end triangle_square_side_ratio_l62_62603


namespace find_distance_l62_62938

def field_width (b : ℝ) : ℝ := 2 * b
def goalpost_width (a : ℝ) : ℝ := 2 * a
def distance_to_sideline (c : ℝ) : ℝ := c
def radius_of_circle (b c : ℝ) : ℝ := b - c

theorem find_distance
    (b a c : ℝ)
    (h_bw : field_width b = 2 * b)
    (h_gw : goalpost_width a = 2 * a)
    (h_ds : distance_to_sideline c = c) :
    let r := radius_of_circle b c in
    (b - c) ^ 2 = a ^ 2 + (√((b - c) ^ 2 - a ^ 2)) ^ 2 := by
  sorry

end find_distance_l62_62938


namespace carpenter_needs_more_logs_l62_62570

-- Define the given conditions in Lean 4
def total_woodblocks_needed : ℕ := 80
def logs_on_hand : ℕ := 8
def woodblocks_per_log : ℕ := 5

-- Statement: Proving the number of additional logs the carpenter needs
theorem carpenter_needs_more_logs :
  let woodblocks_available := logs_on_hand * woodblocks_per_log
  let additional_woodblocks := total_woodblocks_needed - woodblocks_available
  additional_woodblocks / woodblocks_per_log = 8 :=
by
  sorry

end carpenter_needs_more_logs_l62_62570


namespace find_a2_b2_c2_l62_62196

theorem find_a2_b2_c2 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) : 
  a^2 + b^2 + c^2 = 7 / 5 := 
sorry

end find_a2_b2_c2_l62_62196


namespace wrapping_paper_area_correct_l62_62869

structure Box :=
  (l : ℝ)  -- length of the box
  (w : ℝ)  -- width of the box
  (h : ℝ)  -- height of the box
  (h_lw : l > w)  -- condition that length is greater than width

def wrapping_paper_area (b : Box) : ℝ :=
  3 * (b.l + b.w) * b.h

theorem wrapping_paper_area_correct (b : Box) : 
  wrapping_paper_area b = 3 * (b.l + b.w) * b.h :=
sorry

end wrapping_paper_area_correct_l62_62869


namespace train_speed_is_60_kmh_l62_62249

def length_of_train := 116.67 -- in meters
def time_to_cross_pole := 7 -- in seconds

def length_in_km := length_of_train / 1000 -- converting meters to kilometers
def time_in_hours := time_to_cross_pole / 3600 -- converting seconds to hours

theorem train_speed_is_60_kmh :
  (length_in_km / time_in_hours) = 60 :=
sorry

end train_speed_is_60_kmh_l62_62249


namespace original_solution_percentage_l62_62240

theorem original_solution_percentage (P : ℝ) (h1 : 0.5 * P + 0.5 * 30 = 40) : P = 50 :=
by
  sorry

end original_solution_percentage_l62_62240


namespace team_leads_per_supervisor_l62_62107

def num_workers : ℕ := 390
def num_supervisors : ℕ := 13
def leads_per_worker_ratio : ℕ := 10

theorem team_leads_per_supervisor : (num_workers / leads_per_worker_ratio) / num_supervisors = 3 :=
by
  sorry

end team_leads_per_supervisor_l62_62107


namespace range_a_inequality_l62_62733

theorem range_a_inequality (a : ℝ) : (∀ x : ℝ, (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ 1 < a ∧ a ≤ 2 :=
by {
    sorry
}

end range_a_inequality_l62_62733


namespace intersection_complement_l62_62176

universe u

def U := Real

def M : Set Real := { x | -2 ≤ x ∧ x ≤ 2 }

def N : Set Real := { x | x * (x - 3) ≤ 0 }

def complement_U (S : Set Real) : Set Real := { x | x ∉ S }

theorem intersection_complement :
  M ∩ (complement_U N) = { x | -2 ≤ x ∧ x < 0 } := by
  sorry

end intersection_complement_l62_62176


namespace perimeter_increase_ratio_of_sides_l62_62328

def width_increase (a : ℝ) : ℝ := 1.1 * a
def length_increase (b : ℝ) : ℝ := 1.2 * b
def original_perimeter (a b : ℝ) : ℝ := 2 * (a + b)
def new_perimeter (a b : ℝ) : ℝ := 2 * (1.1 * a + 1.2 * b)

theorem perimeter_increase : ∀ a b : ℝ, 
  (a > 0) → (b > 0) → 
  (new_perimeter a b - original_perimeter a b) / (original_perimeter a b) * 100 < 20 := 
by
  sorry

theorem ratio_of_sides (a b : ℝ) (h : new_perimeter a b = 1.18 * original_perimeter a b) : a / b = 1 / 4 := 
by
  sorry

end perimeter_increase_ratio_of_sides_l62_62328


namespace dishwasher_manager_wage_ratio_l62_62126

theorem dishwasher_manager_wage_ratio
  (chef_wage dishwasher_wage manager_wage : ℝ)
  (h1 : chef_wage = 1.22 * dishwasher_wage)
  (h2 : dishwasher_wage = r * manager_wage)
  (h3 : manager_wage = 8.50)
  (h4 : chef_wage = manager_wage - 3.315) :
  r = 0.5 :=
sorry

end dishwasher_manager_wage_ratio_l62_62126


namespace find_ks_l62_62420

theorem find_ks (k : ℕ) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end find_ks_l62_62420


namespace sample_capacity_l62_62380

theorem sample_capacity (freq : ℕ) (freq_rate : ℚ) (H_freq : freq = 36) (H_freq_rate : freq_rate = 0.25) : 
  ∃ n : ℕ, n = 144 :=
by
  sorry

end sample_capacity_l62_62380


namespace parallel_lines_slope_eq_l62_62833

theorem parallel_lines_slope_eq (m : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y - 3 = 0 → 6 * x + m * y + 1 = 0) → m = 4 :=
by
  sorry

end parallel_lines_slope_eq_l62_62833


namespace arithmetic_sequence_a4_l62_62901

theorem arithmetic_sequence_a4 (a : ℕ → ℤ) (a2 a4 a3 : ℤ) (S5 : ℤ)
  (h₁ : S5 = 25)
  (h₂ : a 2 = 3)
  (h₃ : S5 = a 1 + a 2 + a 3 + a 4 + a 5)
  (h₄ : a 3 = (a 1 + a 5) / 2)
  (h₅ : ∀ n : ℕ, (a (n+1) - a n) = (a 2 - a 1)) :
  a 4 = 7 := by
  sorry

end arithmetic_sequence_a4_l62_62901


namespace problem_condition_l62_62439

variable (x y z : ℝ)

theorem problem_condition (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) :
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end problem_condition_l62_62439


namespace probability_at_least_one_woman_correct_l62_62920

noncomputable def probability_at_least_one_woman (total_men: ℕ) (total_women: ℕ) (k: ℕ) : ℚ :=
  let total_people := total_men + total_women
  let total_combinations := Nat.choose total_people k
  let men_combinations := Nat.choose total_men k
  let prob_only_men := (men_combinations : ℚ) / total_combinations
  1 - prob_only_men

theorem probability_at_least_one_woman_correct:
  probability_at_least_one_woman 9 6 4 = 137 / 151 :=
by
  sorry

end probability_at_least_one_woman_correct_l62_62920


namespace sand_needed_for_sandbox_l62_62108

def length1 : ℕ := 50
def width1 : ℕ := 30
def length2 : ℕ := 20
def width2 : ℕ := 15
def area_per_bag : ℕ := 80
def weight_per_bag : ℕ := 30

theorem sand_needed_for_sandbox :
  (length1 * width1 + length2 * width2 + area_per_bag - 1) / area_per_bag * weight_per_bag = 690 :=
by sorry

end sand_needed_for_sandbox_l62_62108


namespace remainder_4873_div_29_l62_62537

theorem remainder_4873_div_29 : 4873 % 29 = 1 := 
by sorry

end remainder_4873_div_29_l62_62537


namespace employed_males_percentage_l62_62786

theorem employed_males_percentage (total_population employed employed_as_percent employed_females female_as_percent employed_males employed_males_percentage : ℕ) 
(total_population_eq : total_population = 100)
(employed_eq : employed = employed_as_percent * total_population / 100)
(employed_as_percent_eq : employed_as_percent = 60)
(employed_females_eq : employed_females = female_as_percent * employed / 100)
(female_as_percent_eq : female_as_percent = 25)
(employed_males_eq : employed_males = employed - employed_females)
(employed_males_percentage_eq : employed_males_percentage = employed_males * 100 / total_population) :
employed_males_percentage = 45 :=
sorry

end employed_males_percentage_l62_62786


namespace equation_graph_is_ellipse_l62_62262

theorem equation_graph_is_ellipse :
  ∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * (x^2 - 72 * y^2) + a * x + d = a * c * (y - 6)^2 :=
sorry

end equation_graph_is_ellipse_l62_62262


namespace concert_cost_l62_62042

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end concert_cost_l62_62042


namespace meat_sales_beyond_plan_l62_62810

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end meat_sales_beyond_plan_l62_62810


namespace Carson_returned_l62_62314

theorem Carson_returned :
  ∀ (initial_oranges ate_oranges stolen_oranges final_oranges : ℕ), 
  initial_oranges = 60 →
  ate_oranges = 10 →
  stolen_oranges = (initial_oranges - ate_oranges) / 2 →
  final_oranges = 30 →
  final_oranges = (initial_oranges - ate_oranges - stolen_oranges) + 5 :=
by 
  sorry

end Carson_returned_l62_62314


namespace euclid_middle_school_math_students_l62_62604

theorem euclid_middle_school_math_students
  (students_Germain : ℕ)
  (students_Newton : ℕ)
  (students_Young : ℕ)
  (students_Euler : ℕ)
  (h_Germain : students_Germain = 12)
  (h_Newton : students_Newton = 10)
  (h_Young : students_Young = 7)
  (h_Euler : students_Euler = 6) :
  students_Germain + students_Newton + students_Young + students_Euler = 35 :=
by {
  sorry
}

end euclid_middle_school_math_students_l62_62604


namespace sarah_min_correct_l62_62071

theorem sarah_min_correct (c : ℕ) (hc : c * 8 + 10 ≥ 110) : c ≥ 13 :=
sorry

end sarah_min_correct_l62_62071


namespace Andrey_knows_the_secret_l62_62625

/-- Question: Does Andrey know the secret?
    Conditions:
    - Andrey says: "I know the secret!"
    - Boris says to Andrey: "No, you don't!"
    - Victor says to Boris: "Boris, you are wrong!"
    - Gosha says to Victor: "No, you are wrong!"
    - Dima says to Gosha: "Gosha, you are lying!"
    - More than half of the kids told the truth (i.e., at least 3 out of 5). --/
theorem Andrey_knows_the_secret (Andrey Boris Victor Gosha Dima : Prop) (truth_count : ℕ)
    (h1 : Andrey)   -- Andrey says he knows the secret
    (h2 : ¬Andrey → Boris)   -- Boris says Andrey does not know the secret
    (h3 : ¬Boris → Victor)   -- Victor says Boris is wrong
    (h4 : ¬Victor → Gosha)   -- Gosha says Victor is wrong
    (h5 : ¬Gosha → Dima)   -- Dima says Gosha is lying
    (h6 : truth_count > 2)   -- More than half of the friends tell the truth (at least 3 out of 5)
    : Andrey := 
sorry

end Andrey_knows_the_secret_l62_62625


namespace triangle_is_isosceles_l62_62183

open Real

variables (α β γ : ℝ) (a b : ℝ)

theorem triangle_is_isosceles
(h1 : a + b = tan (γ / 2) * (a * tan α + b * tan β)) :
α = β :=
by
  sorry

end triangle_is_isosceles_l62_62183


namespace total_blue_balloons_l62_62018

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l62_62018


namespace valve_difference_l62_62556

theorem valve_difference (time_both : ℕ) (time_first : ℕ) (pool_capacity : ℕ) (V1 V2 diff : ℕ) :
  time_both = 48 → 
  time_first = 120 → 
  pool_capacity = 12000 → 
  V1 = pool_capacity / time_first → 
  V1 + V2 = pool_capacity / time_both → 
  diff = V2 - V1 → 
  diff = 50 :=
by sorry

end valve_difference_l62_62556


namespace alice_paper_cranes_l62_62599

theorem alice_paper_cranes (total_cranes : ℕ) (alice_fraction : ℚ) (friend_fraction : ℚ) :
  total_cranes = 1000 →
  alice_fraction = 1/2 →
  friend_fraction = 1/5 →
  let alice_folded := total_cranes * (alice_fraction) in
  let remaining_after_alice := total_cranes - alice_folded in
  let friend_folded := remaining_after_alice / (5) in 
  let remaining := total_cranes - alice_folded - friend_folded in
  remaining = 400 :=
begin
  intros h_total h_alice_fraction h_friend_fraction,
  let alice_folded := total_cranes * alice_fraction,
  let remaining_after_alice := total_cranes - alice_folded,
  let friend_folded := remaining_after_alice * friend_fraction,
  let remaining := total_cranes - alice_folded - friend_folded,
  sorry,
end

end alice_paper_cranes_l62_62599


namespace ratio_thursday_to_wednesday_l62_62947

variables (T : ℕ)

def time_studied_wednesday : ℕ := 2
def time_studied_thursday : ℕ := T
def time_studied_friday : ℕ := T / 2
def time_studied_weekend : ℕ := 2 + T + T / 2
def total_time_studied : ℕ := 22

theorem ratio_thursday_to_wednesday (h : 
  time_studied_wednesday + time_studied_thursday + time_studied_friday + time_studied_weekend = total_time_studied
) : (T : ℚ) / time_studied_wednesday = 3 := by
  sorry

end ratio_thursday_to_wednesday_l62_62947


namespace eval_operations_l62_62890

def star (a b : ℤ) : ℤ := a + b - 1
def hash (a b : ℤ) : ℤ := a * b - 1

theorem eval_operations : star (star 6 8) (hash 3 5) = 26 := by
  sorry

end eval_operations_l62_62890


namespace kittens_given_to_Jessica_is_3_l62_62528

def kittens_initial := 18
def kittens_given_to_Sara := 6
def kittens_now := 9

def kittens_after_Sara := kittens_initial - kittens_given_to_Sara
def kittens_given_to_Jessica := kittens_after_Sara - kittens_now

theorem kittens_given_to_Jessica_is_3 : kittens_given_to_Jessica = 3 := by
  sorry

end kittens_given_to_Jessica_is_3_l62_62528


namespace multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l62_62492

def x : ℤ := 50 + 100 + 140 + 180 + 320 + 400 + 5000

theorem multiple_of_5 : x % 5 = 0 := by 
  sorry

theorem multiple_of_10 : x % 10 = 0 := by 
  sorry

theorem not_multiple_of_20 : x % 20 ≠ 0 := by 
  sorry

theorem not_multiple_of_40 : x % 40 ≠ 0 := by 
  sorry

end multiple_of_5_multiple_of_10_not_multiple_of_20_not_multiple_of_40_l62_62492


namespace students_in_both_clubs_l62_62390

theorem students_in_both_clubs:
  ∀ (U D S : Finset ℕ ), (U.card = 300) → (D.card = 100) → (S.card = 140) → (D ∪ S).card = 210 → (D ∩ S).card = 30 := 
sorry

end students_in_both_clubs_l62_62390


namespace find_m_of_hyperbola_l62_62450

theorem find_m_of_hyperbola (m : ℝ) (h : mx^2 + y^2 = 1) (s : ∃ x : ℝ, x = 2) : m = -4 := 
by
  sorry

end find_m_of_hyperbola_l62_62450


namespace interest_calculation_years_l62_62214

theorem interest_calculation_years
  (principal : ℤ) (rate : ℝ) (difference : ℤ) (n : ℤ)
  (h_principal : principal = 2400)
  (h_rate : rate = 0.04)
  (h_difference : difference = 1920)
  (h_equation : (principal : ℝ) * rate * n = principal - difference) :
  n = 5 := 
sorry

end interest_calculation_years_l62_62214


namespace M_subset_N_l62_62165

variable (f g : ℝ → ℝ) (a : ℝ)

def M : Set ℝ := {x | abs (f x) + abs (g x) < a}
def N : Set ℝ := {x | abs (f x + g x) < a}

theorem M_subset_N (h : a > 0) : M f g a ⊆ N f g a := by
  sorry

end M_subset_N_l62_62165


namespace sequence_sqrt_l62_62631

theorem sequence_sqrt (a : ℕ → ℝ) (h₁ : a 1 = 1) 
  (h₂ : ∀ n, a n > 0)
  (h₃ : ∀ n, a (n+1 - 1) ^ 2 = a (n+1) ^ 2 + 4) :
  ∀ n, a n = Real.sqrt (4 * n - 3) :=
by
  sorry

end sequence_sqrt_l62_62631


namespace problem_diamond_value_l62_62764

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem problem_diamond_value :
  diamond 3 4 = 36 := 
by
  sorry

end problem_diamond_value_l62_62764


namespace best_marksman_score_l62_62593

theorem best_marksman_score (n : ℕ) (hypothetical_score : ℕ) (average_if_hypothetical : ℕ) (actual_total_score : ℕ) (H1 : n = 8) (H2 : hypothetical_score = 92) (H3 : average_if_hypothetical = 84) (H4 : actual_total_score = 665) :
    ∃ (actual_best_score : ℕ), actual_best_score = 77 :=
by
    have hypothetical_total_score : ℕ := 7 * average_if_hypothetical + hypothetical_score
    have difference : ℕ := hypothetical_total_score - actual_total_score
    use hypothetical_score - difference
    sorry

end best_marksman_score_l62_62593


namespace compute_expression_eq_162_l62_62608

theorem compute_expression_eq_162 : 
  3 * 3^4 - 9^35 / 9^33 = 162 := 
by 
  sorry

end compute_expression_eq_162_l62_62608


namespace units_digit_of_6_to_the_6_l62_62999

theorem units_digit_of_6_to_the_6 : (6^6) % 10 = 6 := by
  sorry

end units_digit_of_6_to_the_6_l62_62999


namespace square_area_l62_62382

theorem square_area (side_length : ℕ) (h : side_length = 12) : side_length * side_length = 144 :=
by
  sorry

end square_area_l62_62382


namespace RahulPlayedMatchesSolver_l62_62230

noncomputable def RahulPlayedMatches (current_average new_average runs_in_today current_matches : ℕ) : ℕ :=
  let total_runs_before := current_average * current_matches
  let total_runs_after := total_runs_before + runs_in_today
  let total_matches_after := current_matches + 1
  total_runs_after / new_average

theorem RahulPlayedMatchesSolver:
  RahulPlayedMatches 52 54 78 12 = 12 :=
by
  sorry

end RahulPlayedMatchesSolver_l62_62230


namespace value_of_r_when_n_is_3_l62_62658

def r (s : ℕ) : ℕ := 4^s - 2 * s
def s (n : ℕ) : ℕ := 3^n + 2
def n : ℕ := 3

theorem value_of_r_when_n_is_3 : r (s n) = 4^29 - 58 :=
by
  sorry

end value_of_r_when_n_is_3_l62_62658


namespace borgnine_tarantulas_needed_l62_62128

def total_legs_goal : ℕ := 1100
def chimp_legs : ℕ := 12 * 4
def lion_legs : ℕ := 8 * 4
def lizard_legs : ℕ := 5 * 4
def tarantula_legs : ℕ := 8

theorem borgnine_tarantulas_needed : 
  let total_legs_seen := chimp_legs + lion_legs + lizard_legs
  let legs_needed := total_legs_goal - total_legs_seen
  let num_tarantulas := legs_needed / tarantula_legs
  num_tarantulas = 125 := 
by
  sorry

end borgnine_tarantulas_needed_l62_62128


namespace triangle_area_l62_62695

theorem triangle_area (a b c : ℝ) (h1: a = 15) (h2: c = 17) (h3: a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 60 :=
by
  sorry

end triangle_area_l62_62695


namespace plane_split_into_regions_l62_62134

theorem plane_split_into_regions : 
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  ∃ regions : ℕ, regions = 7 :=
by
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  existsi 7
  sorry

end plane_split_into_regions_l62_62134


namespace fraction_of_janes_age_is_five_eighths_l62_62190

/-- Jane's current age -/
def jane_current_age : ℕ := 34

/-- Number of years ago Jane stopped babysitting -/
def years_since_stopped_babysitting : ℕ := 10

/-- Current age of the oldest child Jane could have babysat -/
def oldest_child_current_age : ℕ := 25

/-- Calculate Jane's age when she stopped babysitting -/
def jane_age_when_stopped_babysitting : ℕ := jane_current_age - years_since_stopped_babysitting

/-- Calculate the child's age when Jane stopped babysitting -/
def oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped_babysitting 

/-- Calculate the fraction of Jane's age that the child could be at most -/
def babysitting_age_fraction : ℚ := (oldest_child_age_when_jane_stopped : ℚ) / (jane_age_when_stopped_babysitting : ℚ)

theorem fraction_of_janes_age_is_five_eighths :
  babysitting_age_fraction = 5 / 8 :=
by 
  -- Declare the proof steps (this part is the placeholder as proof is not required)
  sorry

end fraction_of_janes_age_is_five_eighths_l62_62190


namespace work_completion_by_C_l62_62700

theorem work_completion_by_C
  (A_work_rate : ℝ)
  (B_work_rate : ℝ)
  (C_work_rate : ℝ)
  (A_days_worked : ℝ)
  (B_days_worked : ℝ)
  (C_days_worked : ℝ)
  (A_total_days : ℝ)
  (B_total_days : ℝ)
  (C_completion_partial_work : ℝ)
  (H1 : A_work_rate = 1 / 40)
  (H2 : B_work_rate = 1 / 40)
  (H3 : A_days_worked = 10)
  (H4 : B_days_worked = 10)
  (H5 : C_days_worked = 10)
  (H6 : C_completion_partial_work = 1/2) :
  C_work_rate = 1 / 20 :=
by
  sorry

end work_completion_by_C_l62_62700


namespace number_of_devices_bought_l62_62600

-- Define the essential parameters
def original_price : Int := 800000
def discounted_price : Int := 450000
def total_discount : Int := 16450000

-- Define the main statement to prove
theorem number_of_devices_bought : (total_discount / (original_price - discounted_price) = 47) :=
by
  -- The essential proof is skipped here with sorry
  sorry

end number_of_devices_bought_l62_62600


namespace area_of_triangle_l62_62755

-- Define the function to calculate the area of a right isosceles triangle given the side lengths of squares
theorem area_of_triangle (a b c : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : c = 10) (right_isosceles : true) :
  (1 / 2) * a * c = 50 :=
by
  -- We state the theorem but leave the proof as sorry.
  sorry

end area_of_triangle_l62_62755


namespace min_value_of_f_l62_62615

noncomputable def f (x : ℝ) : ℝ :=
  (3 * Real.sin x - 4 * Real.cos x - 10) * (3 * Real.sin x + 4 * Real.cos x - 10)

theorem min_value_of_f : ∃ x : ℝ, f x = real.minValue := sorry

-- Given the problem we have
def real.minValue : ℝ := (25 / 9) - 10 - (80 * Real.sqrt 2 / 3) - 116

end min_value_of_f_l62_62615


namespace unique_integer_solution_l62_62895

theorem unique_integer_solution (x : ℤ) : x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by
  sorry

end unique_integer_solution_l62_62895


namespace wendy_adds_18_gallons_l62_62359

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end wendy_adds_18_gallons_l62_62359


namespace friends_same_group_probability_l62_62521

theorem friends_same_group_probability :
  let n := 800 in
  let k := 4 in
  let d := n / k in
  (d = 200) →
  let P := (1 / k : ℚ) in
  let Al := 1 in
  ∀ (friend1 friend2 friend3 : ℚ),
  (friend1 = P) →
  (friend2 = P) →
  (friend3 = P) →
  (friend1 * friend2 * friend3 = (1 / 64 : ℚ)) :=
by
  intros n k d d_eq P Al friend1 friend2 friend3 f1_eq f2_eq f3_eq
  exact sorry

end friends_same_group_probability_l62_62521


namespace circle_tangent_locus_l62_62823

theorem circle_tangent_locus (a b : ℝ) :
  (∃ r : ℝ, (a ^ 2 + b ^ 2 = (r + 1) ^ 2) ∧ ((a - 3) ^ 2 + b ^ 2 = (5 - r) ^ 2)) →
  3 * a ^ 2 + 4 * b ^ 2 - 14 * a - 49 = 0 := by
  sorry

end circle_tangent_locus_l62_62823


namespace black_and_white_films_l62_62106

theorem black_and_white_films (y x B : ℕ) 
  (h1 : ∀ B, B = 40 * x)
  (h2 : (4 * y : ℚ) / (((y / x : ℚ) * B / 100) + 4 * y) = 10 / 11) :
  B = 40 * x :=
by sorry

end black_and_white_films_l62_62106


namespace probability_correct_l62_62590

-- Definition for the total number of ways to select topics
def total_ways : ℕ := 6 * 6

-- Definition for the number of ways two students select different topics
def different_topics_ways : ℕ := 6 * 5

-- Definition for the probability of selecting different topics
def probability_different_topics : ℚ := different_topics_ways / total_ways

-- The statement to be proved in Lean
theorem probability_correct :
  probability_different_topics = 5 / 6 := 
sorry

end probability_correct_l62_62590


namespace part_I_part_II_l62_62662

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2 * |x - 1|

-- Part I
theorem part_I (x : ℝ) : (f x 3) ≥ 1 ↔ (0 ≤ x ∧ x ≤ 4 / 3) :=
by sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 2 → f x a - |2 * x - 5| ≤ 0) ↔ (-1 ≤ a ∧ a ≤ 4) :=
by sorry

end part_I_part_II_l62_62662


namespace people_in_room_l62_62080

theorem people_in_room (total_chairs seated_chairs total_people : ℕ) 
  (h1 : 3 * total_people = 5 * seated_chairs)
  (h2 : 4 * total_chairs = 5 * seated_chairs) 
  (h3 : total_chairs - seated_chairs = 8) : 
  total_people = 54 :=
by
  sorry

end people_in_room_l62_62080


namespace seq_product_l62_62443

theorem seq_product (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 2^n - 1)
  (ha : ∀ n, a n = if n = 1 then 1 else 2^(n-1)) :
  a 2 * a 6 = 64 :=
by 
  sorry

end seq_product_l62_62443


namespace imaginary_part_of_f_i_div_i_is_one_l62_62430

def f (x : ℂ) : ℂ := x^3 - 1

theorem imaginary_part_of_f_i_div_i_is_one 
    (i : ℂ) (h : i^2 = -1) :
    ( (f i) / i ).im = 1 := 
sorry

end imaginary_part_of_f_i_div_i_is_one_l62_62430


namespace sequence_value_2009_l62_62660

theorem sequence_value_2009 
  (a : ℕ → ℝ)
  (h_recur : ∀ n ≥ 2, a n = a (n - 1) * a (n + 1))
  (h_a1 : a 1 = 1 + Real.sqrt 3)
  (h_a1776 : a 1776 = 4 + Real.sqrt 3) :
  a 2009 = (3 / 2) + (3 * Real.sqrt 3 / 2) := 
sorry

end sequence_value_2009_l62_62660


namespace balance_objects_l62_62074

open Finset

theorem balance_objects (weights : Fin 10 → ℕ)
  (h_pos : ∀ i, 0 < weights i)
  (h_bound : ∀ i, weights i ≤ 10)
  (h_sum : (∑ i, weights i) = 20) :
  ∃ (s : Finset (Fin 10)), (∑ i in s, weights i) = 10 :=
sorry

end balance_objects_l62_62074


namespace median_of_first_15_integers_l62_62535

theorem median_of_first_15_integers :
  150 * (8 / 100 : ℝ) = 12.0 :=
by
  sorry

end median_of_first_15_integers_l62_62535


namespace slices_per_sandwich_l62_62038

theorem slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) (h1 : total_sandwiches = 5) (h2 : total_slices = 15) :
  total_slices / total_sandwiches = 3 :=
by sorry

end slices_per_sandwich_l62_62038


namespace tank_fill_time_l62_62813

theorem tank_fill_time (R1 R2 t_required : ℝ) (hR1: R1 = 1 / 8) (hR2: R2 = 1 / 12) (hT : t_required = 4.8) :
  t_required = 1 / (R1 + R2) :=
by 
  -- Proof goes here
  sorry

end tank_fill_time_l62_62813


namespace cubic_inequality_l62_62954

theorem cubic_inequality (a : ℝ) (h : a ≠ -1) : 
  (1 + a^3) / (1 + a)^3 ≥ 1 / 4 :=
by sorry

end cubic_inequality_l62_62954


namespace number_of_dimes_l62_62385

theorem number_of_dimes (p n d : ℕ) (h1 : p + n + d = 50) (h2 : p + 5 * n + 10 * d = 200) : d = 14 := 
sorry

end number_of_dimes_l62_62385


namespace determinant_value_l62_62896

-- Define the determinant calculation for a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the initial conditions
variables {x : ℝ}
axiom h : x^2 - 3*x + 1 = 0

-- State the theorem to be proved
theorem determinant_value : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = 1 :=
by
  sorry

end determinant_value_l62_62896


namespace average_age_of_second_group_is_16_l62_62822

theorem average_age_of_second_group_is_16
  (total_age_15_students : ℕ := 225)
  (total_age_first_group_7_students : ℕ := 98)
  (age_15th_student : ℕ := 15) :
  (total_age_15_students - total_age_first_group_7_students - age_15th_student) / 7 = 16 := 
by
  sorry

end average_age_of_second_group_is_16_l62_62822


namespace taehyung_mom_age_l62_62053

variables (taehyung_age_diff_mom : ℕ) (taehyung_age_diff_brother : ℕ) (brother_age : ℕ)

theorem taehyung_mom_age 
  (h1 : taehyung_age_diff_mom = 31) 
  (h2 : taehyung_age_diff_brother = 5) 
  (h3 : brother_age = 7) 
  : 43 = brother_age + taehyung_age_diff_brother + taehyung_age_diff_mom := 
by 
  -- Proof goes here
  sorry

end taehyung_mom_age_l62_62053


namespace Katya_saves_enough_l62_62931

theorem Katya_saves_enough {h c_pool_sauna x y : ℕ} (hc : h = 275) (hcs : c_pool_sauna = 250)
  (hx : x = y + 200) (heq : x + y = c_pool_sauna) : (h / (c_pool_sauna - x)) = 11 :=
by
  sorry

end Katya_saves_enough_l62_62931


namespace smallest_whole_number_larger_than_perimeter_l62_62699

theorem smallest_whole_number_larger_than_perimeter {s : ℝ} (h1 : 16 < s) (h2 : s < 30) :
  61 > 7 + 23 + s :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l62_62699


namespace students_with_exactly_two_skills_l62_62182

-- Definitions based on the conditions:
def total_students : ℕ := 150
def students_can_write : ℕ := total_students - 60 -- 150 - 60 = 90
def students_can_direct : ℕ := total_students - 90 -- 150 - 90 = 60
def students_can_produce : ℕ := total_students - 40 -- 150 - 40 = 110

-- The theorem statement
theorem students_with_exactly_two_skills :
  students_can_write + students_can_direct + students_can_produce - total_students = 110 := 
sorry

end students_with_exactly_two_skills_l62_62182


namespace largest_integral_ratio_l62_62803

theorem largest_integral_ratio (P A : ℕ) (rel_prime_sides : ∃ (a b c : ℕ), gcd a b = 1 ∧ gcd b c = 1 ∧ gcd c a = 1 ∧ a^2 + b^2 = c^2 ∧ P = a + b + c ∧ A = a * b / 2) :
  (∃ (k : ℕ), k = 45 ∧ ∀ l, l < 45 → l ≠ (P^2 / A)) :=
sorry

end largest_integral_ratio_l62_62803


namespace total_students_surveyed_l62_62596

-- Define the constants for liked and disliked students.
def liked_students : ℕ := 235
def disliked_students : ℕ := 165

-- The theorem to prove the total number of students surveyed.
theorem total_students_surveyed : liked_students + disliked_students = 400 :=
by
  -- The proof will go here.
  sorry

end total_students_surveyed_l62_62596


namespace apples_fell_out_l62_62621

theorem apples_fell_out (initial_apples stolen_apples remaining_apples : ℕ) 
  (h₁ : initial_apples = 79) 
  (h₂ : stolen_apples = 45) 
  (h₃ : remaining_apples = 8) 
  : initial_apples - stolen_apples - remaining_apples = 26 := by
  sorry

end apples_fell_out_l62_62621


namespace common_factor_polynomials_l62_62331

-- Define the two polynomials
def poly1 (x y z : ℝ) := 3 * x^2 * y^3 * z + 9 * x^3 * y^3 * z
def poly2 (x y z : ℝ) := 6 * x^4 * y * z^2

-- Define the common factor
def common_factor (x y z : ℝ) := 3 * x^2 * y * z

-- The statement to prove that the common factor of poly1 and poly2 is 3 * x^2 * y * z
theorem common_factor_polynomials (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (poly1 x y z) = (f x y z) * (common_factor x y z) ∧
                          (poly2 x y z) = (f x y z) * (common_factor x y z) :=
sorry

end common_factor_polynomials_l62_62331


namespace number_of_classmates_late_l62_62885

-- Definitions based on conditions from problem statement
def charlizeLate : ℕ := 20
def classmateLate : ℕ := charlizeLate + 10
def totalLateTime : ℕ := 140

-- The proof statement
theorem number_of_classmates_late (x : ℕ) (h1 : totalLateTime = charlizeLate + x * classmateLate) : x = 4 :=
by
  sorry

end number_of_classmates_late_l62_62885


namespace ratio_triangle_square_sides_l62_62602

-- Defining the conditions
def triangle_perimeter : ℝ := 60
def square_perimeter : ℝ := 60
def triangle_side_length : ℝ := triangle_perimeter / 3
def square_side_length : ℝ := square_perimeter / 4

-- Statement of the desired theorem
theorem ratio_triangle_square_sides : triangle_side_length / square_side_length = (4 : ℚ) / 3 :=
by
  sorry

end ratio_triangle_square_sides_l62_62602


namespace find_a_l62_62767

theorem find_a (f : ℝ → ℝ)
  (h : ∀ x : ℝ, x < 2 → a - 3 * x > 0) :
  a = 6 :=
by sorry

end find_a_l62_62767


namespace smallest_n_l62_62361

theorem smallest_n (n : ℕ) (h : 5 * n % 26 = 220 % 26) : n = 18 :=
by
  -- Initial congruence simplification
  have h1 : 220 % 26 = 12 := by norm_num
  rw [h1] at h
  -- Reformulation of the problem
  have h2 : 5 * n % 26 = 12 := h
  -- Conclude the smallest n
  sorry

end smallest_n_l62_62361


namespace sum_of_acute_angles_l62_62754

theorem sum_of_acute_angles (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
    (h2 : Real.sin β = 3 * Real.sqrt 10 / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_acute_angles_l62_62754


namespace cary_needs_6_weekends_l62_62401

variable (shoe_cost : ℕ)
variable (current_savings : ℕ)
variable (earn_per_lawn : ℕ)
variable (lawns_per_weekend : ℕ)
variable (w : ℕ)

theorem cary_needs_6_weekends
    (h1 : shoe_cost = 120)
    (h2 : current_savings = 30)
    (h3 : earn_per_lawn = 5)
    (h4 : lawns_per_weekend = 3)
    (h5 : w * (earn_per_lawn * lawns_per_weekend) = shoe_cost - current_savings) :
    w = 6 :=
by sorry

end cary_needs_6_weekends_l62_62401


namespace impossible_to_maintain_Gini_l62_62233

variables (X Y G0 Y' Z : ℝ)
variables (G1 : ℝ)

-- Conditions
axiom initial_Gini : G0 = 0.1
axiom proportion_poor : X = 0.5
axiom income_poor_initial : Y = 0.4
axiom income_poor_half : Y' = 0.2
axiom population_split : ∀ a b c : ℝ, (a + b + c = 1) ∧ (a = b ∧ b = c)
axiom Gini_constant : G1 = G0

-- Equation system representation final value post situation
axiom Gini_post_reform : 
  G1 = (1 / 2 - ((1 / 6) * 0.2 + (1 / 6) * (0.2 + Z) + (1 / 6) * (1 - 0.2 - Z))) / (1 / 2)

-- Proof problem: to prove inconsistency or inability to maintain Gini coefficient given the conditions
theorem impossible_to_maintain_Gini : false :=
sorry

end impossible_to_maintain_Gini_l62_62233


namespace bernardo_larger_prob_l62_62257

open ProbabilityTheory

-- Definitions
def bernardo_set : Finset ℕ := (Finset.range 10)  -- {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
def silvia_set : Finset ℕ := (Finset.range 9).erase 0  -- {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Main Statement
theorem bernardo_larger_prob :
  let P := (Finset.choose 2 (bernardo_set.card - 1)) / (Finset.choose 3 bernardo_set.card)
          + (1 - (Finset.choose 2 (bernardo_set.card - 1)) / (Finset.choose 3 bernardo_set.card))
            * (1 - (Finset.choose 2 (silvia_set.card - 1)) / (Finset.choose 2 bernardo_set.card)) in
  P = 41 / 90 :=
by
  sorry

end bernardo_larger_prob_l62_62257


namespace sales_revenue_nonnegative_l62_62856

def revenue (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

theorem sales_revenue_nonnegative (x : ℝ) (hx : x = 9 ∨ x = 11) : revenue x ≥ 15950 :=
by
  cases hx
  case inl h₁ =>
    sorry -- calculation for x = 9
  case inr h₂ =>
    sorry -- calculation for x = 11

end sales_revenue_nonnegative_l62_62856


namespace line_equation_l62_62460

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4)) 
  (h_intercept_sum : ∃ b c, b + c = 0 ∧ (∀ x y, A.1 * x + A.2 * y = 1 ∨ A.1 * x + A.2 * y = -1)) :
  (∃ m n, m = 4 ∧ n = -1 ∧ (∀ x y, m * x + n * y = 0)) ∨ 
  (∃ p q r, p = 1 ∧ q = -1 ∧ r = 3 ∧ (∀ x y, p * x + q * y + r = 0)) :=
by
  sorry

end line_equation_l62_62460


namespace booknote_unique_elements_l62_62060

def booknote_string : String := "booknote"
def booknote_set : Finset Char := { 'b', 'o', 'k', 'n', 't', 'e' }

theorem booknote_unique_elements : booknote_set.card = 6 :=
by
  sorry

end booknote_unique_elements_l62_62060


namespace find_m_of_hyperbola_l62_62449

theorem find_m_of_hyperbola (m : ℝ) (h : mx^2 + y^2 = 1) (s : ∃ x : ℝ, x = 2) : m = -4 := 
by
  sorry

end find_m_of_hyperbola_l62_62449


namespace polynomial_simplification_l62_62817

/-- The polynomial simplification problem -/
theorem polynomial_simplification :
  (Polynomial.Coeff.1 * Polynomial.X 3 + 4 * Polynomial.X 2 - 7 * Polynomial.X + 11) +
  (-4 * Polynomial.X 4 - Polynomial.X 3 + Polynomial.X 2 + 7 * Polynomial.X + 3) +
  (3 * Polynomial.X 4 - 2 * Polynomial.X 3 + 5 * Polynomial.X - 1) =
  -Polynomial.Coeff.1 * Polynomial.X 4 - 2 * Polynomial.X 3 + 5 * Polynomial.X 2 + 5 * Polynomial.X + 13 := by
  sorry

end polynomial_simplification_l62_62817


namespace total_genuine_purses_and_handbags_l62_62848

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem total_genuine_purses_and_handbags : GenuinePurses + GenuineHandbags = 31 := by
  sorry

end total_genuine_purses_and_handbags_l62_62848


namespace helly_half_planes_helly_convex_polygons_l62_62086

-- Helly's theorem for half-planes
theorem helly_half_planes (n : ℕ) (H : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (H i ∩ H j ∩ H k).Nonempty) : 
  (⋂ i, H i).Nonempty :=
sorry

-- Helly's theorem for convex polygons
theorem helly_convex_polygons (n : ℕ) (P : Fin n → Set ℝ) 
  (h : ∀ (i j k : Fin n), (P i ∩ P j ∩ P k).Nonempty) : 
  (⋂ i, P i).Nonempty :=
sorry

end helly_half_planes_helly_convex_polygons_l62_62086


namespace solve_absolute_value_equation_l62_62334

theorem solve_absolute_value_equation (y : ℝ) :
  (|y - 8| + 3 * y = 11) → (y = 1.5) :=
by
  sorry

end solve_absolute_value_equation_l62_62334


namespace kim_morning_routine_time_l62_62797

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end kim_morning_routine_time_l62_62797


namespace oranges_thrown_away_l62_62115

theorem oranges_thrown_away (initial_oranges old_oranges_thrown new_oranges final_oranges : ℕ) 
    (h1 : initial_oranges = 34)
    (h2 : new_oranges = 13)
    (h3 : final_oranges = 27)
    (h4 : initial_oranges - old_oranges_thrown + new_oranges = final_oranges) :
    old_oranges_thrown = 20 :=
by
  sorry

end oranges_thrown_away_l62_62115


namespace k_value_l62_62863

theorem k_value (k m : ℤ) (h : (m - 8) ∣ (m^2 - k * m - 24)) : k = 5 := by
  have : (m - 8) ∣ (m^2 - 8 * m - 24) := sorry
  sorry

end k_value_l62_62863


namespace rug_shorter_side_l62_62719

theorem rug_shorter_side (x : ℝ) :
  (64 - x * 7) / 64 = 0.78125 → x = 2 :=
by
  sorry

end rug_shorter_side_l62_62719


namespace elephant_distribution_l62_62814

theorem elephant_distribution (unions nonunions : ℕ) (elephants : ℕ) :
  unions = 28 ∧ nonunions = 37 ∧ (∀ k : ℕ, elephants = 28 * k ∨ elephants = 37 * k) ∧ (∀ k : ℕ, ((28 * k ≤ elephants) ∧ (37 * k ≤ elephants))) → 
  elephants = 2072 :=
by
  sorry

end elephant_distribution_l62_62814


namespace surface_area_of_cube_l62_62268

noncomputable def cube_edge_length : ℝ := 20

theorem surface_area_of_cube (edge_length : ℝ) (h : edge_length = cube_edge_length) : 
    6 * edge_length ^ 2 = 2400 :=
by
  rw [h]
  sorry  -- proof placeholder

end surface_area_of_cube_l62_62268


namespace melies_meat_purchase_l62_62032

-- Define the relevant variables and conditions
variable (initial_amount : ℕ) (amount_left : ℕ) (cost_per_kg : ℕ)

-- State the main theorem we want to prove
theorem melies_meat_purchase (h1 : initial_amount = 180) (h2 : amount_left = 16) (h3 : cost_per_kg = 82) :
  (initial_amount - amount_left) / cost_per_kg = 2 := by
  sorry

end melies_meat_purchase_l62_62032


namespace line_equation_l62_62462

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4))
  (sum_intercepts_zero : ∃ a b : ℝ, (a + b = 0) ∧ (A.1 * b + A.2 * a = a * b)) :
  (∀ x y : ℝ, x - A.1 = (y - A.2) * 4 → 4 * x - y = 0) ∨
  (∀ x y : ℝ, (x / (-3)) + (y / 3) = 1 → x - y + 3 = 0) :=
sorry

end line_equation_l62_62462


namespace apples_to_grapes_proof_l62_62962

theorem apples_to_grapes_proof :
  (3 / 4 * 12 = 9) → (1 / 3 * 9 = 3) :=
by
  sorry

end apples_to_grapes_proof_l62_62962


namespace girls_in_school_l62_62829

noncomputable def num_of_girls (total_students : ℕ) (sampled_students : ℕ) (sampled_diff : ℤ) : ℕ :=
  sorry

theorem girls_in_school :
  let total_students := 1600
  let sampled_students := 200
  let sampled_diff := 10
  num_of_girls total_students sampled_students sampled_diff = 760 :=
  sorry

end girls_in_school_l62_62829


namespace possible_values_of_S_l62_62942

open Function Fintype

theorem possible_values_of_S : 
  ∀ (σ : Fin 10 → Fin 10) (hσ : Perm σ), 
  let S := |σ 0 - σ 1| + |σ 2 - σ 3| + 
           |σ 4 - σ 5| + |σ 6 - σ 7| + 
           |σ 8 - σ 9| in 
  S ∈ {5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25}.
Proof
  sorry

end possible_values_of_S_l62_62942


namespace smallest_number_of_marbles_l62_62388

theorem smallest_number_of_marbles :
  ∃ (r w b g y : ℕ), 
  (r + w + b + g + y = 13) ∧ 
  (r ≥ 5) ∧
  (r - 4 = 5 * w) ∧
  ((r - 3) * (r - 4) = 20 * w * b) ∧
  sorry := sorry

end smallest_number_of_marbles_l62_62388


namespace eval_operation_l62_62612

-- Definition of the * operation based on the given table
def op (a b : ℕ) : ℕ :=
  match a, b with
  | 1, 1 => 4
  | 1, 2 => 1
  | 1, 3 => 2
  | 1, 4 => 3
  | 2, 1 => 1
  | 2, 2 => 3
  | 2, 3 => 4
  | 2, 4 => 2
  | 3, 1 => 2
  | 3, 2 => 4
  | 3, 3 => 1
  | 3, 4 => 3
  | 4, 1 => 3
  | 4, 2 => 2
  | 4, 3 => 3
  | 4, 4 => 4
  | _, _ => 0 -- Default case (not needed as per the given problem definition)

-- Statement of the problem in Lean 4
theorem eval_operation : op (op 3 1) (op 4 2) = 3 :=
by {
  sorry -- Proof to be provided
}

end eval_operation_l62_62612


namespace equivalent_statements_l62_62858

variable (P Q : Prop)

theorem equivalent_statements : 
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by 
  sorry

end equivalent_statements_l62_62858


namespace max_area_rectangle_l62_62065

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end max_area_rectangle_l62_62065


namespace analytical_expression_maximum_value_l62_62682

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 2 * Real.sin (ω * x - Real.pi / 6) + 1

theorem analytical_expression (ω : ℝ) (h1 : ω > 0) (h2 : ∀ x, abs (x - (x + (Real.pi / (2 * ω)))) = Real.pi / 2) : 
  f x 2 = 2 * Real.sin (2 * x - Real.pi / 6) + 1 :=
sorry

theorem maximum_value (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  2 * Real.sin (2 * x - Real.pi / 6) + 1 ≤ 3 :=
sorry

end analytical_expression_maximum_value_l62_62682


namespace probability_of_alternating_colors_l62_62102

-- The setup for our problem
variables (B W : Type) 

/-- A box contains 5 white balls and 5 black balls.
    Prove that the probability that all of my draws alternate colors
    is 1/126. -/
theorem probability_of_alternating_colors (W B : ℕ) (hw : W = 5) (hb : B = 5) :
  let total_ways := Nat.choose 10 5 in
  let successful_ways := 2 in
  successful_ways / total_ways = (1 : ℚ) / 126 :=
sorry

end probability_of_alternating_colors_l62_62102


namespace egg_laying_hens_l62_62508

theorem egg_laying_hens (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) :
  total_chickens = 325 →
  roosters = 28 →
  non_laying_hens = 20 →
  (total_chickens - roosters - non_laying_hens = 277) :=
by
  intros
  sorry

end egg_laying_hens_l62_62508


namespace room_width_l62_62345

theorem room_width (length : ℕ) (total_cost : ℕ) (cost_per_sqm : ℕ) : ℚ :=
  let area := total_cost / cost_per_sqm
  let width := area / length
  width

example : room_width 9 38475 900 = 4.75 := by
  sorry

end room_width_l62_62345


namespace work_completion_in_16_days_l62_62228

theorem work_completion_in_16_days (A B : ℕ) :
  (1 / A + 1 / B = 1 / 40) → (10 * (1 / A + 1 / B) = 1 / 4) →
  (12 * 1 / A = 3 / 4) → A = 16 :=
by
  intros h1 h2 h3
  -- Proof is omitted by "sorry".
  sorry

end work_completion_in_16_days_l62_62228


namespace nina_running_distance_l62_62033

theorem nina_running_distance (total_distance : ℝ) (initial_run : ℝ) (num_initial_runs : ℕ) :
  total_distance = 0.8333333333333334 →
  initial_run = 0.08333333333333333 →
  num_initial_runs = 2 →
  (total_distance - initial_run * num_initial_runs = 0.6666666666666667) :=
by
  intros h_total h_initial h_num
  sorry

end nina_running_distance_l62_62033


namespace application_outcomes_l62_62693

theorem application_outcomes :
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  (choices_A * choices_B * choices_C) = 18 :=
by
  let choices_A := 3
  let choices_B := 2
  let choices_C := 3
  show (choices_A * choices_B * choices_C = 18)
  sorry

end application_outcomes_l62_62693


namespace workman_problem_l62_62715

theorem workman_problem 
  {A B : Type}
  (W : ℕ)
  (RA RB : ℝ)
  (h1 : RA = (1/2) * RB)
  (h2 : RA + RB = W / 14)
  : W / RB = 21 :=
by
  sorry

end workman_problem_l62_62715


namespace minimal_total_cost_l62_62077

def waterway_length : ℝ := 100
def max_speed : ℝ := 50
def other_costs_per_hour : ℝ := 3240
def speed_at_ten_cost : ℝ := 10
def fuel_cost_at_ten : ℝ := 60
def proportionality_constant : ℝ := 0.06

noncomputable def total_cost (v : ℝ) : ℝ :=
  6 * v^2 + 324000 / v

theorem minimal_total_cost :
  (∃ v : ℝ, 0 < v ∧ v ≤ max_speed ∧ total_cost v = 16200) ∧ 
  (∀ v : ℝ, 0 < v ∧ v ≤ max_speed → total_cost v ≥ 16200) :=
sorry

end minimal_total_cost_l62_62077


namespace solve_equation_l62_62839

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end solve_equation_l62_62839


namespace simplify_correct_l62_62044

def simplify_expression (a b : ℤ) : ℤ :=
  (30 * a + 70 * b) + (15 * a + 45 * b) - (12 * a + 60 * b)

theorem simplify_correct (a b : ℤ) : simplify_expression a b = 33 * a + 55 * b :=
by 
  sorry -- Proof to be filled in later

end simplify_correct_l62_62044


namespace product_zero_probability_l62_62338

noncomputable def probability_product_is_zero : ℚ :=
  let S := [-3, -1, 0, 0, 2, 5]
  let total_ways := 15 -- Calculated as 6 choose 2 taking into account repetition
  let favorable_ways := 8 -- Calculated as (2 choose 1) * (4 choose 1)
  favorable_ways / total_ways

theorem product_zero_probability : probability_product_is_zero = 8 / 15 := by
  sorry

end product_zero_probability_l62_62338


namespace Johnny_is_8_l62_62362

-- Define Johnny's current age
def johnnys_age (x : ℕ) : Prop :=
  x + 2 = 2 * (x - 3)

theorem Johnny_is_8 (x : ℕ) (h : johnnys_age x) : x = 8 :=
sorry

end Johnny_is_8_l62_62362


namespace james_net_profit_l62_62652

def totalCandyBarsSold (boxes : Nat) (candyBarsPerBox : Nat) : Nat :=
  boxes * candyBarsPerBox

def revenue30CandyBars (pricePerCandyBar : Real) : Real :=
  30 * pricePerCandyBar

def revenue20CandyBars (pricePerCandyBar : Real) : Real :=
  20 * pricePerCandyBar

def totalRevenue (revenue1 : Real) (revenue2 : Real) : Real :=
  revenue1 + revenue2

def costNonDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def costDiscountedBoxes (candyBars : Nat) (pricePerCandyBar : Real) : Real :=
  candyBars * pricePerCandyBar

def totalCost (cost1 : Real) (cost2 : Real) : Real :=
  cost1 + cost2

def salesTax (totalRevenue : Real) (taxRate : Real) : Real :=
  totalRevenue * taxRate

def totalExpenses (cost : Real) (salesTax : Real) (fixedExpense : Real) : Real :=
  cost + salesTax + fixedExpense

def netProfit (totalRevenue : Real) (totalExpenses : Real) : Real :=
  totalRevenue - totalExpenses

theorem james_net_profit :
  let boxes := 5
  let candyBarsPerBox := 10
  let totalCandyBars := totalCandyBarsSold boxes candyBarsPerBox

  let priceFirst30 := 1.50
  let priceNext20 := 1.30
  let priceSubsequent := 1.10

  let revenueFirst30 := revenue30CandyBars priceFirst30
  let revenueNext20 := revenue20CandyBars priceNext20
  let totalRevenue := totalRevenue revenueFirst30 revenueNext20

  let priceNonDiscounted := 1.00
  let candyBarsNonDiscounted := 20
  let costNonDiscounted := costNonDiscountedBoxes candyBarsNonDiscounted priceNonDiscounted

  let priceDiscounted := 0.80
  let candyBarsDiscounted := 30
  let costDiscounted := costDiscountedBoxes candyBarsDiscounted priceDiscounted

  let totalCost := totalCost costNonDiscounted costDiscounted

  let taxRate := 0.07
  let salesTax := salesTax totalRevenue taxRate

  let fixedExpense := 15.0
  let totalExpenses := totalExpenses totalCost salesTax fixedExpense

  netProfit totalRevenue totalExpenses = 7.03 :=
by
  sorry

end james_net_profit_l62_62652


namespace measure_of_angle_A_l62_62650

variables (A B C a b c : ℝ)
variables (triangle_acute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
variables (sides_relation : (a^2 + b^2 - c^2) * tan A = a * b)

theorem measure_of_angle_A :
  A = π / 6 :=
by 
  sorry

end measure_of_angle_A_l62_62650


namespace points_for_victory_l62_62180

theorem points_for_victory (V : ℕ) :
  (∃ (played total_games : ℕ) (points_after_games : ℕ) (remaining_games : ℕ) (needed_points : ℕ) 
     (draw_points defeat_points : ℕ) (minimum_wins : ℕ), 
     played = 5 ∧
     total_games = 20 ∧ 
     points_after_games = 12 ∧
     remaining_games = total_games - played ∧
     needed_points = 40 - points_after_games ∧
     draw_points = 1 ∧
     defeat_points = 0 ∧
     minimum_wins = 7 ∧
     7 * V ≥ needed_points ∧
     remaining_games = total_games - played ∧
     needed_points = 28) → V = 4 :=
sorry

end points_for_victory_l62_62180


namespace option_A_correct_l62_62305

theorem option_A_correct (a b : ℝ) (h : a > b) : a + 2 > b + 2 :=
by sorry

end option_A_correct_l62_62305


namespace find_values_of_symbols_l62_62371

theorem find_values_of_symbols (a b : ℕ) (h1 : a + b + b = 55) (h2 : a + b = 40) : b = 15 ∧ a = 25 :=
  by
    sorry

end find_values_of_symbols_l62_62371


namespace find_x_l62_62011

-- Define the conditions as given in the problem
def angle1 (x : ℝ) : ℝ := 6 * x
def angle2 (x : ℝ) : ℝ := 3 * x
def angle3 (x : ℝ) : ℝ := x
def angle4 (x : ℝ) : ℝ := 5 * x
def sum_of_angles (x : ℝ) : ℝ := angle1 x + angle2 x + angle3 x + angle4 x

-- State the problem: prove that x equals 24 given the sum of angles is 360 degrees
theorem find_x (x : ℝ) (h : sum_of_angles x = 360) : x = 24 :=
by
  sorry

end find_x_l62_62011


namespace total_hours_until_joy_sees_grandma_l62_62653

theorem total_hours_until_joy_sees_grandma
  (days_until_grandma: ℕ)
  (hours_in_a_day: ℕ)
  (timezone_difference: ℕ)
  (H_days : days_until_grandma = 2)
  (H_hours : hours_in_a_day = 24)
  (H_timezone : timezone_difference = 3) :
  (days_until_grandma * hours_in_a_day = 48) :=
by
  sorry

end total_hours_until_joy_sees_grandma_l62_62653


namespace part1_l62_62198

def p (m x : ℝ) := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) := (x + 2)^2 < 1

theorem part1 (x : ℝ) (m : ℝ) (hm : m = -2) : p m x ∧ q x ↔ -3 < x ∧ x ≤ -2 :=
by
  unfold p q
  sorry

end part1_l62_62198


namespace vanessa_earnings_l62_62619

def cost : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost

theorem vanessa_earnings : money_made = 16 := by
  sorry

end vanessa_earnings_l62_62619


namespace cone_to_prism_volume_ratio_l62_62874

noncomputable def ratio_of_volumes (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) : ℝ :=
  let r := a / 2
  let V_cone := (1/3) * Real.pi * r^2 * h
  let V_prism := a * (2 * a) * h
  V_cone / V_prism

theorem cone_to_prism_volume_ratio (a h : ℝ) (pos_a : 0 < a) (pos_h : 0 < h) :
  ratio_of_volumes a h pos_a pos_h = Real.pi / 24 := by
  sorry

end cone_to_prism_volume_ratio_l62_62874


namespace min_value_l62_62953

open Real

theorem min_value (x y : ℝ) (h : x + y = 4) : x^2 + y^2 ≥ 8 := by
  sorry

end min_value_l62_62953


namespace sum_coefficients_eq_neg_one_l62_62468

theorem sum_coefficients_eq_neg_one (a a1 a2 a3 a4 a5 : ℝ) :
  (∀ x y : ℝ, (x - 2 * y)^5 = a * x^5 + a1 * x^4 * y + a2 * x^3 * y^2 + a3 * x^2 * y^3 + a4 * x * y^4 + a5 * y^5) →
  a + a1 + a2 + a3 + a4 + a5 = -1 :=
by
  sorry

end sum_coefficients_eq_neg_one_l62_62468


namespace solve_equation_l62_62335

theorem solve_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 1) :
  (x / (x - 1) - 2 / x = 1) ↔ x = 2 :=
sorry

end solve_equation_l62_62335


namespace least_number_of_marbles_l62_62113

theorem least_number_of_marbles :
  ∃ n, (∀ d ∈ ({3, 4, 5, 7, 8} : Set ℕ), d ∣ n) ∧ n = 840 :=
by
  sorry

end least_number_of_marbles_l62_62113


namespace factorize_expression_find_xy_l62_62350

-- Problem 1: Factorizing the quadratic expression
theorem factorize_expression (x : ℝ) : 
  x^2 - 120 * x + 3456 = (x - 48) * (x - 72) :=
sorry

-- Problem 2: Finding the product xy from the given equation
theorem find_xy (x y : ℝ) (h : x^2 + y^2 + 8 * x - 12 * y + 52 = 0) : 
  x * y = -24 :=
sorry

end factorize_expression_find_xy_l62_62350


namespace sandy_marbles_correct_l62_62487

namespace MarbleProblem

-- Define the number of dozens Jessica has
def jessica_dozens : ℕ := 3

-- Define the conversion from dozens to individual marbles
def dozens_to_marbles (d : ℕ) : ℕ := 12 * d

-- Calculate the number of marbles Jessica has
def jessica_marbles : ℕ := dozens_to_marbles jessica_dozens

-- Define the multiplier for Sandy's marbles
def sandy_multiplier : ℕ := 4

-- Define the number of marbles Sandy has
def sandy_marbles : ℕ := sandy_multiplier * jessica_marbles

theorem sandy_marbles_correct : sandy_marbles = 144 :=
by
  sorry

end MarbleProblem

end sandy_marbles_correct_l62_62487


namespace max_f_value_l62_62900

open Real

noncomputable def problem (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) : Prop :=
  x1 * x2 * x3 = ((12 - x1) * (12 - x2) * (12 - x3))^2

theorem max_f_value (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 ≤ 12) (h2 : 0 ≤ x2 ∧ x2 ≤ 12) (h3 : 0 ≤ x3 ∧ x3 ≤ 12) (h : problem x1 x2 x3 h1 h2 h3) : 
  x1 * x2 * x3 ≤ 729 :=
sorry

end max_f_value_l62_62900


namespace george_team_final_round_average_required_less_than_record_l62_62150

theorem george_team_final_round_average_required_less_than_record :
  ∀ (old_record average_score : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ),
    old_record = 287 →
    players = 4 →
    rounds = 10 →
    current_score = 10440 →
    (old_record - ((rounds * (old_record * players) - current_score) / players)) = 27 :=
by
  -- Given the values and conditions, prove the equality here
  sorry

end george_team_final_round_average_required_less_than_record_l62_62150


namespace average_marks_in_6_subjects_l62_62884

/-- The average marks Ashok secured in 6 subjects is 72
Given:
1. The average of marks in 5 subjects is 74.
2. Ashok secured 62 marks in the 6th subject.
-/
theorem average_marks_in_6_subjects (avg_5 : ℕ) (marks_6th : ℕ) (h_avg_5 : avg_5 = 74) (h_marks_6th : marks_6th = 62) : 
  ((avg_5 * 5 + marks_6th) / 6) = 72 :=
  by
  sorry

end average_marks_in_6_subjects_l62_62884


namespace system_solution_l62_62689

theorem system_solution (x y : ℝ) :
  (x + y = 4) ∧ (2 * x - y = 2) → x = 2 ∧ y = 2 := by 
sorry

end system_solution_l62_62689


namespace find_YZ_l62_62482

noncomputable def triangle_YZ (angle_Y : ℝ) (XY : ℝ) (XZ : ℝ) : ℝ :=
  if angle_Y = 45 ∧ XY = 100 ∧ XZ = 50 * Real.sqrt 2 then
    50 * Real.sqrt 6
  else
    0

theorem find_YZ :
  triangle_YZ 45 100 (50 * Real.sqrt 2) = 50 * Real.sqrt 6 :=
by
  sorry

end find_YZ_l62_62482


namespace value_of_a6_l62_62842

theorem value_of_a6 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 * n^2 - 5 * n)
  (ha : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h1 : a 1 = S 1):
  a 6 = 28 :=
sorry

end value_of_a6_l62_62842


namespace find_value_of_expression_l62_62469

theorem find_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 + 4 ≤ ab + 3 * b + 2 * c) :
  200 * a + 9 * b + c = 219 :=
sorry

end find_value_of_expression_l62_62469


namespace find_percentage_l62_62172

theorem find_percentage (P : ℝ) (h : (P / 100) * 600 = (50 / 100) * 720) : P = 60 :=
by
  sorry

end find_percentage_l62_62172


namespace verify_incorrect_operation_l62_62549

theorem verify_incorrect_operation (a : ℝ) :
  ¬ ((-a^2)^3 = -a^5) :=
by
  sorry

end verify_incorrect_operation_l62_62549


namespace cary_needs_six_weekends_l62_62400

theorem cary_needs_six_weekends
  (shoe_cost : ℕ)
  (saved : ℕ)
  (earn_per_lawn : ℕ)
  (lawns_per_weekend : ℕ)
  (additional_needed : ℕ := shoe_cost - saved)
  (earn_per_weekend : ℕ := earn_per_lawn * lawns_per_weekend)
  (weekends_needed : ℕ := additional_needed / earn_per_weekend) :
  shoe_cost = 120 ∧ saved = 30 ∧ earn_per_lawn = 5 ∧ lawns_per_weekend = 3 → weekends_needed = 6 := by 
  sorry

end cary_needs_six_weekends_l62_62400


namespace cary_needs_six_weekends_l62_62398

theorem cary_needs_six_weekends
  (shoe_cost : ℕ)
  (saved : ℕ)
  (earn_per_lawn : ℕ)
  (lawns_per_weekend : ℕ)
  (additional_needed : ℕ := shoe_cost - saved)
  (earn_per_weekend : ℕ := earn_per_lawn * lawns_per_weekend)
  (weekends_needed : ℕ := additional_needed / earn_per_weekend) :
  shoe_cost = 120 ∧ saved = 30 ∧ earn_per_lawn = 5 ∧ lawns_per_weekend = 3 → weekends_needed = 6 := by 
  sorry

end cary_needs_six_weekends_l62_62398


namespace quadratic_roots_l62_62525

theorem quadratic_roots:
  ∀ x : ℝ, x^2 - 1 = 0 ↔ (x = -1 ∨ x = 1) :=
by
  sorry

end quadratic_roots_l62_62525


namespace determine_parabola_coefficients_l62_62732

noncomputable def parabola_coefficients (a b c : ℚ) : Prop :=
  ∀ (x y : ℚ), 
      (y = a * x^2 + b * x + c) ∧
      (
        ((4, 5) = (x, y)) ∧
        ((2, 3) = (x, y))
      )

theorem determine_parabola_coefficients :
  parabola_coefficients (-1/2) 4 (-3) :=
by
  sorry

end determine_parabola_coefficients_l62_62732


namespace Tn_gt_Sn_l62_62287

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l62_62287


namespace proof_problem_l62_62500

noncomputable def f (x : ℝ) : ℝ := -x / (1 + |x|)

def M (a b : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ b}
def N (a b : ℝ) : Set ℝ := {y | ∃ x ∈ M a b, y = f x}

theorem proof_problem (a b : ℝ) (h : a < b) : M a b = N a b → False := by
  sorry

end proof_problem_l62_62500


namespace andrew_calculation_l62_62726

theorem andrew_calculation (x y : ℝ) (hx : x ≠ 0) :
  0.4 * 0.5 * x = 0.2 * 0.3 * y → y = (10 / 3) * x :=
by
  sorry

end andrew_calculation_l62_62726


namespace calculation_of_expression_l62_62394

theorem calculation_of_expression :
  (1.99 ^ 2 - 1.98 * 1.99 + 0.99 ^ 2) = 1 := 
by sorry

end calculation_of_expression_l62_62394


namespace circle_regions_l62_62533

def regions_divided_by_chords (n : ℕ) : ℕ :=
  (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24

theorem circle_regions (n : ℕ) : 
  regions_divided_by_chords n = (n^4 - 6 * n^3 + 23 * n^2 - 18 * n + 24) / 24 := 
  by 
  sorry

end circle_regions_l62_62533


namespace alice_still_needs_to_fold_l62_62598

theorem alice_still_needs_to_fold (total_cranes alice_folds friend_folds remains: ℕ) 
  (h1 : total_cranes = 1000)
  (h2 : alice_folds = total_cranes / 2)
  (h3 : friend_folds = (total_cranes - alice_folds) / 5)
  (h4 : remains = total_cranes - alice_folds - friend_folds) :
  remains = 400 := 
  by
    sorry

end alice_still_needs_to_fold_l62_62598


namespace calculate_expression_l62_62259

theorem calculate_expression : (4 + Real.sqrt 6) * (4 - Real.sqrt 6) = 10 := by
  sorry

end calculate_expression_l62_62259


namespace pq_proof_l62_62285

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l62_62285


namespace polygon_side_count_l62_62002

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end polygon_side_count_l62_62002


namespace problem_statement_l62_62442

noncomputable def sum_binom_coeff (x : ℝ) (n : ℕ) : ℝ :=
  (x - (2 / x^2)) ^ n

noncomputable def is_even_sum_correct (n : ℕ) (x : ℝ) : Prop :=
  (n = 6 → sum_binom_coeff x n = 64) ∧
  (n = 6 → ∑ i in (Finset.range (n / 2 + 1)).map (λ i, 2 * i), Nat.choose n (2 * i) = 32)

noncomputable def is_constant_term_correct (n : ℕ): Prop :=
  (n = 6 → Nat.choose n 2 * (-2) ^ 2 = 60)

noncomputable def is_largest_coeff_correct (n : ℕ) : Prop :=
  (n = 6 → Nat.choose n 4 * (-2) ^ 4 = 240)

theorem problem_statement (n : ℕ) (x : ℝ) :
  is_even_sum_correct n x ∧ is_constant_term_correct n ∧ is_largest_coeff_correct n :=
by
  sorry

end problem_statement_l62_62442


namespace intersection_M_N_l62_62297

open Set

variable (x : ℝ)
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := sorry

end intersection_M_N_l62_62297


namespace sum_of_two_smallest_l62_62627

variable (a b c d : ℕ)
variable (x : ℕ)

-- Four numbers a, b, c, d are in the ratio 3:5:7:9
def ratios := (a = 3 * x) ∧ (b = 5 * x) ∧ (c = 7 * x) ∧ (d = 9 * x)

-- The average of these numbers is 30
def average := (a + b + c + d) / 4 = 30

-- The theorem to prove the sum of the two smallest numbers (a and b) is 40
theorem sum_of_two_smallest (h1 : ratios a b c d x) (h2 : average a b c d) : a + b = 40 := by
  sorry

end sum_of_two_smallest_l62_62627


namespace sum_of_a_b_c_d_e_l62_62657

theorem sum_of_a_b_c_d_e (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) 
  (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) : a + b + c + d + e = 33 := by
  sorry

end sum_of_a_b_c_d_e_l62_62657


namespace find_ab_sum_l62_62636

theorem find_ab_sum 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) 
  : a + b = -14 := by
  sorry

end find_ab_sum_l62_62636


namespace at_least_one_not_less_than_two_l62_62197

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l62_62197


namespace Q_investment_time_l62_62562

theorem Q_investment_time  
  (P Q x t : ℝ)
  (h_ratio_investments : P = 7 * x ∧ Q = 5 * x)
  (h_ratio_profits : (7 * x * 10) / (5 * x * t) = 7 / 10) :
  t = 20 :=
by {
  sorry
}

end Q_investment_time_l62_62562


namespace largest_expr_is_expr1_l62_62656

def U : ℝ := 3 * 2005 ^ 2006
def V : ℝ := 2005 ^ 2006
def W : ℝ := 2004 * 2005 ^ 2005
def X : ℝ := 3 * 2005 ^ 2005
def Y : ℝ := 2005 ^ 2005
def Z : ℝ := 2005 ^ 2004

def expr1 : ℝ := U - V
def expr2 : ℝ := V - W
def expr3 : ℝ := W - X
def expr4 : ℝ := X - Y
def expr5 : ℝ := Y - Z

theorem largest_expr_is_expr1 : 
  max (max (max expr1 expr2) (max expr3 expr4)) expr5 = expr1 := 
sorry

end largest_expr_is_expr1_l62_62656


namespace investment_share_l62_62203

variable (P_investment Q_investment : ℝ)

theorem investment_share (h1 : Q_investment = 60000) (h2 : P_investment / Q_investment = 2 / 3) : P_investment = 40000 := by
  sorry

end investment_share_l62_62203


namespace find_radius_k_l62_62601

/-- Mathematical conditions for the given geometry problem -/
structure problem_conditions where
  radius_F : ℝ := 15
  radius_G : ℝ := 4
  radius_H : ℝ := 3
  radius_I : ℝ := 3
  radius_J : ℝ := 1

/-- Proof problem statement defining the required theorem -/
theorem find_radius_k (conditions : problem_conditions) :
  let r := (137:ℝ) / 8
  20 * r = (342.5 : ℝ) :=
by
  sorry

end find_radius_k_l62_62601


namespace ribbon_tape_remaining_l62_62192

theorem ribbon_tape_remaining 
  (initial_length used_for_ribbon used_for_gift : ℝ)
  (h_initial: initial_length = 1.6)
  (h_ribbon: used_for_ribbon = 0.8)
  (h_gift: used_for_gift = 0.3) : 
  initial_length - used_for_ribbon - used_for_gift = 0.5 :=
by 
  sorry

end ribbon_tape_remaining_l62_62192


namespace inverse_g_of_87_l62_62677

noncomputable def g (x : ℝ) : ℝ := 3 * x^3 + 6

theorem inverse_g_of_87 : (g x = 87) → (x = 3) :=
by
  intro h
  sorry

end inverse_g_of_87_l62_62677


namespace inequality_pow_gt_linear_l62_62035

theorem inequality_pow_gt_linear {a : ℝ} (n : ℕ) (h₁ : a > -1) (h₂ : a ≠ 0) (h₃ : n ≥ 2) :
  (1 + a:ℝ)^n > 1 + n * a :=
sorry

end inequality_pow_gt_linear_l62_62035


namespace Seokjin_total_problems_l62_62597

theorem Seokjin_total_problems (initial_problems : ℕ) (additional_problems : ℕ)
  (h1 : initial_problems = 12) (h2 : additional_problems = 7) :
  initial_problems + additional_problems = 19 :=
by
  sorry

end Seokjin_total_problems_l62_62597


namespace percentage_increase_first_to_second_l62_62017

theorem percentage_increase_first_to_second (D1 D2 D3 : ℕ) (h1 : D2 = 12)
  (h2 : D3 = D2 + Nat.div (D2 * 25) 100) (h3 : D1 + D2 + D3 = 37) :
  Nat.div ((D2 - D1) * 100) D1 = 20 := by
  sorry

end percentage_increase_first_to_second_l62_62017


namespace easter_eggs_problem_l62_62807

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end easter_eggs_problem_l62_62807


namespace equation_of_line_l62_62908

theorem equation_of_line (P : ℝ × ℝ) (m : ℝ) : 
  P = (3, 3) → m = 2 * 1 → ∃ b : ℝ, ∀ x : ℝ, P.2 = m * (x - P.1) + b ↔ y = 2 * x - 3 := 
by {
  sorry
}

end equation_of_line_l62_62908


namespace logs_needed_l62_62569

theorem logs_needed (needed_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) 
  (H1 : needed_woodblocks = 80) 
  (H2 : current_logs = 8) 
  (H3 : woodblocks_per_log = 5) : 
  current_logs * woodblocks_per_log < needed_woodblocks → 
  (needed_woodblocks - current_logs * woodblocks_per_log) / woodblocks_per_log = 8 := by
  sorry

end logs_needed_l62_62569


namespace find_mn_l62_62519

theorem find_mn (sec_x_plus_tan_x : ℝ) (sec_tan_eq : sec_x_plus_tan_x = 24 / 7) :
  ∃ (m n : ℕ) (h : Int.gcd m n = 1), (∃ y, y = (m:ℝ) / (n:ℝ) ∧ (y^2)*527^2 - 2*y*527*336 + 336^2 = 1) ∧
  m + n = boxed_mn :=
by
  sorry

end find_mn_l62_62519


namespace prove_equation_1_prove_equation_2_l62_62960

theorem prove_equation_1 : 
  ∀ x, (x - 3) / (x - 2) - 1 = 3 / x ↔ x = 3 / 2 :=
by
  sorry

theorem prove_equation_2 :
  ¬∃ x, (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
by
  sorry

end prove_equation_1_prove_equation_2_l62_62960


namespace calculate_opening_price_l62_62886

theorem calculate_opening_price (C : ℝ) (r : ℝ) (P : ℝ) 
  (h1 : C = 15)
  (h2 : r = 0.5)
  (h3 : C = P + r * P) :
  P = 10 :=
by sorry

end calculate_opening_price_l62_62886


namespace batsman_average_46_innings_l62_62100

theorem batsman_average_46_innings {hs ls t_44 : ℕ} (h_diff: hs - ls = 180) (h_avg_44: t_44 = 58 * 44) (h_hiscore: hs = 194) : 
  (t_44 + hs + ls) / 46 = 60 := 
sorry

end batsman_average_46_innings_l62_62100


namespace problem1_problem2_l62_62730

-- Problem 1
theorem problem1 :
  (1 : ℝ) * (2 * Real.sqrt 12 - (1 / 2) * Real.sqrt 18) - (Real.sqrt 75 - (1 / 4) * Real.sqrt 32)
  = -Real.sqrt 3 - (Real.sqrt 2) / 2 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (2 : ℝ) * (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1 / 2)) - Real.sqrt 30 / Real.sqrt 5
  = 1 + Real.sqrt 6 :=
by
  sorry

end problem1_problem2_l62_62730


namespace cary_needs_6_weekends_l62_62403

variable (shoe_cost : ℕ)
variable (current_savings : ℕ)
variable (earn_per_lawn : ℕ)
variable (lawns_per_weekend : ℕ)
variable (w : ℕ)

theorem cary_needs_6_weekends
    (h1 : shoe_cost = 120)
    (h2 : current_savings = 30)
    (h3 : earn_per_lawn = 5)
    (h4 : lawns_per_weekend = 3)
    (h5 : w * (earn_per_lawn * lawns_per_weekend) = shoe_cost - current_savings) :
    w = 6 :=
by sorry

end cary_needs_6_weekends_l62_62403


namespace no_arithmetic_progression_in_squares_l62_62037

theorem no_arithmetic_progression_in_squares :
  ∀ (a d : ℕ), d > 0 → ¬ (∃ (f : ℕ → ℕ), 
    (∀ n, f n = a + n * d) ∧ 
    (∀ n, ∃ m, n ^ 2 = f m)) :=
by
  sorry

end no_arithmetic_progression_in_squares_l62_62037


namespace towel_bleach_percentage_decrease_l62_62118

-- Define the problem
theorem towel_bleach_percentage_decrease (L B : ℝ) (x : ℝ) (h_length : 0 < L) (h_breadth : 0 < B) 
  (h1 : 0.64 * L * B = 0.8 * L * (1 - x / 100) * B) :
  x = 20 :=
by
  -- The actual proof is not needed, providing "sorry" as a placeholder for the proof.
  sorry

end towel_bleach_percentage_decrease_l62_62118


namespace algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l62_62515

theorem algebraic_simplification (x : ℤ) (h1 : -3 < x) (h2 : x < 3) (h3 : x ≠ 0) (h4 : x ≠ 1) (h5 : x ≠ -1) :
  (x - (x / (x + 1))) / (1 + (1 / (x^2 - 1))) = x - 1 :=
sorry

theorem evaluate_expression_for_x2 (h1 : -3 < 2) (h2 : 2 < 3) (h3 : 2 ≠ 0) (h4 : 2 ≠ 1) (h5 : 2 ≠ -1) :
  (2 - (2 / (2 + 1))) / (1 + (1 / (2^2 - 1))) = 1 :=
sorry

theorem evaluate_expression_for_x_neg2 (h1 : -3 < -2) (h2 : -2 < 3) (h3 : -2 ≠ 0) (h4 : -2 ≠ 1) (h5 : -2 ≠ -1) :
  (-2 - (-2 / (-2 + 1))) / (1 + (1 / ((-2)^2 - 1))) = -3 :=
sorry

end algebraic_simplification_evaluate_expression_for_x2_evaluate_expression_for_x_neg2_l62_62515


namespace carpenter_needs_more_logs_l62_62571

-- Define the given conditions in Lean 4
def total_woodblocks_needed : ℕ := 80
def logs_on_hand : ℕ := 8
def woodblocks_per_log : ℕ := 5

-- Statement: Proving the number of additional logs the carpenter needs
theorem carpenter_needs_more_logs :
  let woodblocks_available := logs_on_hand * woodblocks_per_log
  let additional_woodblocks := total_woodblocks_needed - woodblocks_available
  additional_woodblocks / woodblocks_per_log = 8 :=
by
  sorry

end carpenter_needs_more_logs_l62_62571


namespace question1_question2_question3_question4_l62_62123

theorem question1 : (2 * 3) ^ 2 = 2 ^ 2 * 3 ^ 2 := by admit

theorem question2 : (-1 / 2 * 2) ^ 3 = (-1 / 2) ^ 3 * 2 ^ 3 := by admit

theorem question3 : (3 / 2) ^ 2019 * (-2 / 3) ^ 2019 = -1 := by admit

theorem question4 (a b : ℝ) (n : ℕ) (h : 0 < n): (a * b) ^ n = a ^ n * b ^ n := by admit

end question1_question2_question3_question4_l62_62123


namespace exists_positive_integer_m_l62_62799

theorem exists_positive_integer_m (n : ℕ) (hn : 0 < n) : ∃ m : ℕ, 0 < m ∧ 7^n ∣ (3^m + 5^m - 1) :=
sorry

end exists_positive_integer_m_l62_62799


namespace plates_usage_when_parents_join_l62_62806

theorem plates_usage_when_parents_join
  (total_plates : ℕ)
  (plates_per_day_matt_and_son : ℕ)
  (days_matt_and_son : ℕ)
  (days_with_parents : ℕ)
  (total_days_in_week : ℕ)
  (total_plates_needed : total_plates = 38)
  (plates_used_matt_and_son : plates_per_day_matt_and_son = 2)
  (days_matt_and_son_eq : days_matt_and_son = 3)
  (days_with_parents_eq : days_with_parents = 4)
  (total_days_in_week_eq : total_days_in_week = 7)
  (plates_used_when_parents_join : total_plates - plates_per_day_matt_and_son * days_matt_and_son = days_with_parents * 8) :
  true :=
sorry

end plates_usage_when_parents_join_l62_62806


namespace percentage_of_alcohol_in_mixture_A_l62_62202

theorem percentage_of_alcohol_in_mixture_A (x : ℝ) :
  (10 * x / 100 + 5 * 50 / 100 = 15 * 30 / 100) → x = 20 :=
by
  intro h
  sorry

end percentage_of_alcohol_in_mixture_A_l62_62202


namespace number_of_solutions_l62_62138

theorem number_of_solutions :
  (∀ (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 → x ≠ 0 ∧ x ≠ 5) →
  ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 :=
by
  sorry

end number_of_solutions_l62_62138


namespace interest_earned_l62_62026

noncomputable def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) := P * (1 + r) ^ t

theorem interest_earned :
  let P := 2000
  let r := 0.05
  let t := 5
  let A := compound_interest P r t
  A - P = 552.56 :=
by
  sorry

end interest_earned_l62_62026


namespace inequality_1_inequality_2_inequality_4_l62_62888

theorem inequality_1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

theorem inequality_2 (a : ℝ) : a * (1 - a) ≤ 1 / 4 := sorry

theorem inequality_4 (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := sorry

end inequality_1_inequality_2_inequality_4_l62_62888


namespace MeatMarket_sales_l62_62812

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end MeatMarket_sales_l62_62812


namespace valve_difference_l62_62557

theorem valve_difference (time_both : ℕ) (time_first : ℕ) (pool_capacity : ℕ) (V1 V2 diff : ℕ) :
  time_both = 48 → 
  time_first = 120 → 
  pool_capacity = 12000 → 
  V1 = pool_capacity / time_first → 
  V1 + V2 = pool_capacity / time_both → 
  diff = V2 - V1 → 
  diff = 50 :=
by sorry

end valve_difference_l62_62557


namespace total_investment_amount_l62_62594

theorem total_investment_amount 
    (x : ℝ) 
    (h1 : 6258.0 * 0.08 + x * 0.065 = 678.87) : 
    x + 6258.0 = 9000.0 :=
sorry

end total_investment_amount_l62_62594


namespace shopper_savings_percentage_l62_62247

theorem shopper_savings_percentage
  (amount_saved : ℝ) (final_price : ℝ)
  (h_saved : amount_saved = 3)
  (h_final : final_price = 27) :
  (amount_saved / (final_price + amount_saved)) * 100 = 10 := 
by
  sorry

end shopper_savings_percentage_l62_62247


namespace right_triangle_roots_l62_62435

theorem right_triangle_roots (α β : ℝ) (k : ℕ) (h_triangle : (α^2 + β^2 = 100) ∧ (α + β = 14) ∧ (α * β = 4 * k - 4)) : k = 13 :=
sorry

end right_triangle_roots_l62_62435


namespace base7_addition_XY_l62_62418

theorem base7_addition_XY (X Y : ℕ) (h1 : (Y + 2) % 7 = X % 7) (h2 : (X + 5) % 7 = 9 % 7) : X + Y = 6 :=
by sorry

end base7_addition_XY_l62_62418


namespace votes_difference_l62_62781

theorem votes_difference (V : ℝ) (h1 : 0.62 * V = 899) :
  |(0.62 * V) - (0.38 * V)| = 348 :=
by
  -- The solution goes here
  sorry

end votes_difference_l62_62781


namespace grandson_age_l62_62031

variable (G F : ℕ)

-- Define the conditions given in the problem
def condition1 := F = 6 * G
def condition2 := (F + 4) + (G + 4) = 78

-- The theorem to prove
theorem grandson_age : condition1 G F → condition2 G F → G = 10 :=
by
  intros h1 h2
  sorry

end grandson_age_l62_62031


namespace find_k_values_l62_62760

noncomputable def parallel_vectors (k : ℝ) : Prop :=
  (k^2 / k = (k + 1) / 4)

theorem find_k_values (k : ℝ) : parallel_vectors k ↔ (k = 0 ∨ k = 1 / 3) :=
by sorry

end find_k_values_l62_62760


namespace division_multiplication_identity_l62_62534

theorem division_multiplication_identity (a b c d : ℕ) (h1 : b = 6) (h2 : c = 2) (h3 : d = 3) :
  a = 120 → 120 * (b / c) * d = 120 := by
  intro h
  rw [h2, h3, h1]
  sorry

end division_multiplication_identity_l62_62534


namespace tan_alpha_value_l62_62431

theorem tan_alpha_value : 
  ∀ (α : ℝ), (sin α = (2 * real.sqrt 5) / 5) ∧ (real.pi / 2 ≤ α ∧ α ≤ real.pi) → tan α = -2 :=
by
  intros α h
  sorry

end tan_alpha_value_l62_62431


namespace chess_tournament_games_l62_62865

def players : ℕ := 12

def games_per_pair : ℕ := 2

theorem chess_tournament_games (n : ℕ) (h : n = players) : 
  (n * (n - 1) * games_per_pair) = 264 := by
  sorry

end chess_tournament_games_l62_62865


namespace find_number_l62_62509

theorem find_number (n : ℕ) (h1 : n % 20 = 1) (h2 : n / 20 = 9) : n = 181 := 
by {
  -- proof not required
  sorry
}

end find_number_l62_62509


namespace smallest_integer_solution_l62_62736

open Int

theorem smallest_integer_solution :
  ∃ x : ℤ, (⌊ (x : ℚ) / 8 ⌋ - ⌊ (x : ℚ) / 40 ⌋ + ⌊ (x : ℚ) / 240 ⌋ = 210) ∧ x = 2016 :=
by
  sorry

end smallest_integer_solution_l62_62736


namespace vacation_expense_sharing_l62_62880

def alice_paid : ℕ := 90
def bob_paid : ℕ := 150
def charlie_paid : ℕ := 120
def donna_paid : ℕ := 240
def total_paid : ℕ := alice_paid + bob_paid + charlie_paid + donna_paid
def individual_share : ℕ := total_paid / 4

def alice_owes : ℕ := individual_share - alice_paid
def charlie_owes : ℕ := individual_share - charlie_paid
def donna_owes : ℕ := donna_paid - individual_share

def a : ℕ := charlie_owes
def b : ℕ := donna_owes - (donna_owes - charlie_owes)

theorem vacation_expense_sharing : a - b = 0 :=
by
  sorry

end vacation_expense_sharing_l62_62880


namespace cost_per_adult_meal_l62_62727

-- Definitions and given conditions
def total_people : ℕ := 13
def num_kids : ℕ := 9
def total_cost : ℕ := 28

-- Question translated into a proof statement
theorem cost_per_adult_meal : (total_cost / (total_people - num_kids)) = 7 := 
by
  sorry

end cost_per_adult_meal_l62_62727


namespace gini_inconsistency_l62_62234

-- Definitions of given conditions
def X : ℝ := 0.5
def G0 : ℝ := 0.1
def Y : ℝ := 0.4
def Y' : ℝ := 0.2

-- After the reform, defining the shares
def Z : ℝ -- income share for the middle class
def share_rich : ℝ := 1 - (Y' + Z)

-- Correct value to be checked
def G : ℝ := 0.1

-- Proof object to express that maintaining the same Gini coefficient is impossible
theorem gini_inconsistency : ∀ (Z : ℝ), G ≠ (1 / 2 - (1 / 6 * 0.2 + 1 / 6 * (0.2 + Z) + 1 / 6)) / (1 / 2) := 
by
  sorry

end gini_inconsistency_l62_62234


namespace same_number_of_friends_l62_62036

-- Definitions and conditions
def num_people (n : ℕ) := true   -- Placeholder definition to indicate the number of people
def num_friends (person : ℕ) (n : ℕ) : ℕ := sorry -- The number of friends a given person has (needs to be defined)
def friends_range (n : ℕ) := ∀ person, 0 ≤ num_friends person n ∧ num_friends person n < n

-- Theorem statement
theorem same_number_of_friends (n : ℕ) (h1 : num_people n) (h2 : friends_range n) : 
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ num_friends p1 n = num_friends p2 n :=
by
  sorry

end same_number_of_friends_l62_62036


namespace sum_of_consecutive_powers_divisible_l62_62671

theorem sum_of_consecutive_powers_divisible (a : ℕ) (n : ℕ) (h : 0 ≤ n) : 
  a^n + a^(n + 1) ∣ a * (a + 1) :=
sorry

end sum_of_consecutive_powers_divisible_l62_62671


namespace length_of_EC_l62_62236

variable (AC : ℝ) (AB : ℝ) (CD : ℝ) (EC : ℝ)

def is_trapezoid (AB CD : ℝ) : Prop := AB = 3 * CD
def perimeter (AB CD AC : ℝ) : Prop := AB + CD + AC + (AC / 3) = 36

theorem length_of_EC
  (h1 : is_trapezoid AB CD)
  (h2 : AC = 18)
  (h3 : perimeter AB CD AC) :
  EC = 9 / 2 :=
  sorry

end length_of_EC_l62_62236


namespace cans_increment_l62_62775

/--
If there are 9 rows of cans in a triangular display, where each successive row increases 
by a certain number of cans \( x \) compared to the row above it, with the seventh row having 
19 cans, and the total number of cans being fewer than 120, then 
each row has 4 more cans than the row above it.
-/
theorem cans_increment (x : ℕ) : 
  9 * 19 - 16 * x < 120 → x > 51 / 16 → x = 4 :=
by
  intros h1 h2
  sorry

end cans_increment_l62_62775


namespace find_k_eq_3_l62_62144

theorem find_k_eq_3 (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3) → k = 3 :=
by sorry

end find_k_eq_3_l62_62144


namespace rhombus_diagonal_length_l62_62054

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) 
(h_d2 : d2 = 18) (h_area : area = 126) (h_formula : area = (d1 * d2) / 2) : 
d1 = 14 :=
by
  -- We're skipping the proof steps.
  sorry

end rhombus_diagonal_length_l62_62054


namespace initial_percentage_salt_l62_62867

theorem initial_percentage_salt :
  ∀ (P : ℝ),
  let Vi := 64 
  let Vf := 80
  let target_percent := 0.08
  (Vi * P = Vf * target_percent) → P = 0.1 :=
by
  intros P Vi Vf target_percent h
  have h1 : Vi = 64 := rfl
  have h2 : Vf = 80 := rfl
  have h3 : target_percent = 0.08 := rfl
  rw [h1, h2, h3] at h
  sorry

end initial_percentage_salt_l62_62867


namespace probability_at_least_one_woman_selected_l62_62919

-- Definitions corresponding to the conditions:
def total_people : ℕ := 15
def men : ℕ := 9
def women : ℕ := 6
def select_people : ℕ := 4

-- Define binomial coefficient function using Lean's binomial notation
def binom : ℕ → ℕ → ℕ
| n, k := Nat.choose n k

-- The main theorem to state the probability that at least one woman is selected
theorem probability_at_least_one_woman_selected :
  let total_ways := binom total_people select_people
  let men_ways := binom men select_people
  1 - (men_ways / total_ways : ℚ) = 13 / 15 :=
by
  -- Let Lean ignore the details of this proof for now
  sorry

end probability_at_least_one_woman_selected_l62_62919


namespace min_value_of_quadratic_l62_62802

theorem min_value_of_quadratic (x y s : ℝ) (h : x + y = s) : 
  ∃ x y, 3 * x^2 + 2 * y^2 = 6 * s^2 / 5 := sorry

end min_value_of_quadratic_l62_62802


namespace cost_per_pack_l62_62485

theorem cost_per_pack (total_bill : ℕ) (change_given : ℕ) (packs : ℕ) (total_cost := total_bill - change_given) (cost_per_pack := total_cost / packs) 
  (h1 : total_bill = 20) 
  (h2 : change_given = 11) 
  (h3 : packs = 3) : 
  cost_per_pack = 3 := by
  sorry

end cost_per_pack_l62_62485


namespace noah_large_paintings_last_month_l62_62325

-- problem definitions
def large_painting_price : ℕ := 60
def small_painting_price : ℕ := 30
def small_paintings_sold_last_month : ℕ := 4
def sales_this_month : ℕ := 1200

-- to be proven
theorem noah_large_paintings_last_month (L : ℕ) (last_month_sales_eq : large_painting_price * L + small_painting_price * small_paintings_sold_last_month = S) 
   (this_month_sales_eq : 2 * S = sales_this_month) :
  L = 8 :=
sorry

end noah_large_paintings_last_month_l62_62325


namespace mike_passing_percentage_l62_62028

theorem mike_passing_percentage (mike_score shortfall max_marks : ℝ)
  (h_mike_score : mike_score = 212)
  (h_shortfall : shortfall = 16)
  (h_max_marks : max_marks = 760) :
  (mike_score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end mike_passing_percentage_l62_62028


namespace no_integer_solution_l62_62168

theorem no_integer_solution (x y : ℤ) : ¬(x^4 + y^2 = 4 * y + 4) :=
by
  sorry

end no_integer_solution_l62_62168


namespace desired_line_equation_l62_62970

-- Define the center of the circle and the equation of the given line
def center : (ℝ × ℝ) := (-1, 0)
def line1 (x y : ℝ) : Prop := x + y = 0

-- Define the desired line passing through the center of the circle and perpendicular to line1
def line2 (x y : ℝ) : Prop := x + y + 1 = 0

-- The theorem stating that the desired line equation is x + y + 1 = 0
theorem desired_line_equation : ∀ (x y : ℝ),
  (center = (-1, 0)) → (∀ x y, line1 x y → line2 x y) :=
by
  sorry

end desired_line_equation_l62_62970


namespace Mateen_garden_area_l62_62849

theorem Mateen_garden_area :
  ∀ (L W : ℝ), (50 * L = 2000) ∧ (20 * (2 * L + 2 * W) = 2000) → (L * W = 400) :=
by
  intros L W h
  -- We have two conditions based on the problem:
  -- 1. Mateen must walk the length 50 times to cover 2000 meters.
  -- 2. Mateen must walk the perimeter 20 times to cover 2000 meters.
  have h1 : 50 * L = 2000 := h.1
  have h2 : 20 * (2 * L + 2 * W) = 2000 := h.2
  -- We can use these conditions to derive the area of the garden
  sorry

end Mateen_garden_area_l62_62849


namespace find_positive_number_l62_62927

theorem find_positive_number (x n : ℝ) (h₁ : (x + 1) ^ 2 = n) (h₂ : (x - 5) ^ 2 = n) : n = 9 := 
sorry

end find_positive_number_l62_62927


namespace avg_age_across_rooms_l62_62473

namespace AverageAgeProof

def Room := Type

-- Conditions
def people_in_room_a : ℕ := 8
def avg_age_room_a : ℕ := 35

def people_in_room_b : ℕ := 5
def avg_age_room_b : ℕ := 30

def people_in_room_c : ℕ := 7
def avg_age_room_c : ℕ := 25

-- Combined Calculations
def total_people := people_in_room_a + people_in_room_b + people_in_room_c
def total_age := (people_in_room_a * avg_age_room_a) + (people_in_room_b * avg_age_room_b) + (people_in_room_c * avg_age_room_c)

noncomputable def average_age : ℚ := total_age / total_people

-- Proof that the average age of all the people across the three rooms is 30.25
theorem avg_age_across_rooms : average_age = 30.25 := 
sorry

end AverageAgeProof

end avg_age_across_rooms_l62_62473


namespace nested_expression_sum_l62_62132

theorem nested_expression_sum : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))))))))) = 5592404 :=
by 
  sorry

end nested_expression_sum_l62_62132


namespace find_y_is_90_l62_62143

-- Definitions for given conditions
def angle_ABC : ℝ := 120
def angle_ABD : ℝ := 180 - angle_ABC
def angle_BDA : ℝ := 30

-- The theorem to prove y = 90 degrees
theorem find_y_is_90 :
  ∃ y : ℝ, angle_ABD = 60 ∧ angle_BDA = 30 ∧ (30 + 60 + y = 180) → y = 90 :=
by
  sorry

end find_y_is_90_l62_62143


namespace original_price_of_sweater_l62_62116

theorem original_price_of_sweater (sold_price : ℝ) (discount : ℝ) (original_price : ℝ) 
    (h1 : sold_price = 120) (h2 : discount = 0.40) (h3: (1 - discount) * original_price = sold_price) : 
    original_price = 200 := by 
  sorry

end original_price_of_sweater_l62_62116


namespace initial_deadlift_weight_l62_62791

theorem initial_deadlift_weight
    (initial_squat : ℕ := 700)
    (initial_bench : ℕ := 400)
    (D : ℕ)
    (squat_loss : ℕ := 30)
    (deadlift_loss : ℕ := 200)
    (new_total : ℕ := 1490) :
    (initial_squat * (100 - squat_loss) / 100) + initial_bench + (D - deadlift_loss) = new_total → D = 800 :=
by
  sorry

end initial_deadlift_weight_l62_62791


namespace length_of_train_l62_62872

theorem length_of_train (speed_km_hr : ℝ) (platform_length_m : ℝ) (time_sec : ℝ) 
  (h1 : speed_km_hr = 72) (h2 : platform_length_m = 250) (h3 : time_sec = 30) : 
  ∃ (train_length : ℝ), train_length = 350 := 
by 
  -- Definitions of the given conditions
  let speed_m_per_s := speed_km_hr * (5 / 18)
  let total_distance := speed_m_per_s * time_sec
  let train_length := total_distance - platform_length_m
  -- Verifying the length of the train
  use train_length
  sorry

end length_of_train_l62_62872


namespace distance_from_neg6_to_origin_l62_62057

theorem distance_from_neg6_to_origin :
  abs (-6) = 6 :=
by
  sorry

end distance_from_neg6_to_origin_l62_62057


namespace stockholm_to_uppsala_distance_l62_62827

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end stockholm_to_uppsala_distance_l62_62827


namespace max_value_of_gems_l62_62010

/-- Conditions -/
structure Gem :=
  (weight : ℕ)
  (value : ℕ)

def Gem1 : Gem := ⟨3, 9⟩
def Gem2 : Gem := ⟨6, 20⟩
def Gem3 : Gem := ⟨2, 5⟩

-- Laura can carry maximum of 21 pounds.
def max_weight : ℕ := 21

-- She is able to carry at least 15 of each type
def min_count := 15

/-- Prove that the maximum value Laura can carry is $69 -/
theorem max_value_of_gems : ∃ (n1 n2 n3 : ℕ), (n1 >= min_count) ∧ (n2 >= min_count) ∧ (n3 >= min_count) ∧ 
  (Gem1.weight * n1 + Gem2.weight * n2 + Gem3.weight * n3 ≤ max_weight) ∧ 
  (Gem1.value * n1 + Gem2.value * n2 + Gem3.value * n3 = 69) :=
sorry

end max_value_of_gems_l62_62010


namespace max_area_rect_l62_62066

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end max_area_rect_l62_62066


namespace problem_statement_l62_62787

open Real
open EuclideanGeometry

noncomputable def isosceles_triangle_AB_AC : Prop :=
  ∃ (A B C D : Point) (AB AC BC BD DA : ℝ),
    Triangle A B C ∧
    Segment AB = Segment AC ∧
    ∠ BAC = 100 ∧
    Bisection D A C B ∧
    BC = BD + DA

theorem problem_statement (A B C D : Point) (AB AC BC BD DA : ℝ) :
  isosceles_triangle_AB_AC A B C D AB AC BC BD DA →
  BC = BD + DA :=
by
  intro h
  sorry

end problem_statement_l62_62787


namespace b_should_pay_l62_62087

def TotalRent : ℕ := 725
def Cost_a : ℕ := 12 * 8 * 5
def Cost_b : ℕ := 16 * 9 * 6
def Cost_c : ℕ := 18 * 6 * 7
def Cost_d : ℕ := 20 * 4 * 4
def TotalCost : ℕ := Cost_a + Cost_b + Cost_c + Cost_d
def Payment_b (Cost_b TotalCost TotalRent : ℕ) : ℕ := (Cost_b * TotalRent) / TotalCost

theorem b_should_pay :
  Payment_b Cost_b TotalCost TotalRent = 259 := 
  by
  unfold Payment_b
  -- Leaving the proof body empty as per instructions
  sorry

end b_should_pay_l62_62087


namespace inverse_proposition_l62_62059

theorem inverse_proposition (a b : ℝ) (h1 : a < 1) (h2 : b < 1) : a + b ≠ 2 :=
by sorry

end inverse_proposition_l62_62059


namespace discriminant_of_quadratic_eq_l62_62056

theorem discriminant_of_quadratic_eq : 
  ∀ (x : ℝ), x^2 - 4 * x + 3 = 0 →
  let a := 1
  let b := -4
  let c := 3
  discriminant a b c = 4 := 
by
  intro x h_eq
  let a := 1
  let b := -4
  let c := 3
  have H : discriminant a b c = b^2 - 4 * a * c := rfl
  rw [H, <-h_eq]
  -- We would have the proof steps here
  sorry

end discriminant_of_quadratic_eq_l62_62056


namespace zoe_pop_albums_l62_62859

theorem zoe_pop_albums (total_songs country_albums songs_per_album : ℕ) (h1 : total_songs = 24) (h2 : country_albums = 3) (h3 : songs_per_album = 3) :
  total_songs - (country_albums * songs_per_album) = 15 ↔ (total_songs - (country_albums * songs_per_album)) / songs_per_album = 5 :=
by
  sorry

end zoe_pop_albums_l62_62859


namespace parallelogram_area_l62_62703

theorem parallelogram_area (base height : ℝ) (h_base : base = 22) (h_height : height = 14) :
  base * height = 308 := by
  sorry

end parallelogram_area_l62_62703


namespace last_two_digits_factorials_sum_l62_62989

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l62_62989


namespace decimal_to_fraction_l62_62226

theorem decimal_to_fraction :
  (3.56 : ℚ) = 89 / 25 := 
sorry

end decimal_to_fraction_l62_62226


namespace find_r_l62_62173

variable (p r s : ℝ)

theorem find_r (h : ∀ x : ℝ, (y : ℝ) = x^2 + p * x + r + s → (y = 10 ↔ x = -p / 2)) : r = 10 - s + p^2 / 4 := by
  sorry

end find_r_l62_62173


namespace total_amount_paid_l62_62122

def price_grapes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_mangoes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_pineapple (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_kiwi (kg: ℕ) (rate: ℕ) : ℕ := kg * rate

theorem total_amount_paid :
  price_grapes 14 54 + price_mangoes 10 62 + price_pineapple 8 40 + price_kiwi 5 30 = 1846 :=
by
  sorry

end total_amount_paid_l62_62122


namespace decimal_to_fraction_l62_62227

theorem decimal_to_fraction (a b : ℚ) (h : a = 3.56) (h1 : b = 56/100) (h2 : 56.gcd 100 = 4) :
  a = 89/25 := by
  sorry

end decimal_to_fraction_l62_62227


namespace immune_response_problem_l62_62239

open ProbabilityTheory

variables (M: ℝ → ℝ) (σ: ℝ) (μ: ℝ)
variables (number_of_people: ℝ)
variables (condition1 : 1 ≤ 1000)
variables (condition2 : ∀ x, M x ≈ Normal 15 σ^2)
variables (condition3 : (P (λ x, 10 < x ∧ x < 20) (Normal 15 σ^2)) = 19 / 25)

theorem immune_response_problem
  (h1 : ∀ x, True → P (λ x, x ≤ 10) (Normal 15 σ^2) = 3 / 25)
  (h2 : ∀ x, True → P (λ x, x ≥ 20) (Normal 15 σ^2) = 3 / 25)
  (h3 : ∀ x, True → P (λ x, x ≤ 20) (Normal 15 σ^2) = 22 / 25)
  : (number_of_people = 1000) →
    number_of_people * 22 / 25 = 880 := 
begin
  intros,
  sorry
end

end immune_response_problem_l62_62239


namespace logs_needed_l62_62568

theorem logs_needed (needed_woodblocks : ℕ) (current_logs : ℕ) (woodblocks_per_log : ℕ) 
  (H1 : needed_woodblocks = 80) 
  (H2 : current_logs = 8) 
  (H3 : woodblocks_per_log = 5) : 
  current_logs * woodblocks_per_log < needed_woodblocks → 
  (needed_woodblocks - current_logs * woodblocks_per_log) / woodblocks_per_log = 8 := by
  sorry

end logs_needed_l62_62568


namespace part1_part2_l62_62273

-- Define Set A
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3/2}

-- Define Set B, parameterized by m
def B (m : ℝ) : Set ℝ := {x | 1-m < x ∧ x ≤ 3*m + 1}

-- Proof Problem (1): When m = 1, A ∩ B = {x | 0 < x ∧ x ≤ 3/2}
theorem part1 (x : ℝ) : (x ∈ A ∩ B 1) ↔ (0 < x ∧ x ≤ 3/2) := by
  sorry

-- Proof Problem (2): If ∀ x, x ∈ A → x ∈ B m, then m ∈ (-∞, 1/6]
theorem part2 (m : ℝ) : (∀ x, x ∈ A → x ∈ B m) → m ≤ 1/6 := by
  sorry

end part1_part2_l62_62273


namespace probability_correct_l62_62588

-- Definition for the total number of ways to select topics
def total_ways : ℕ := 6 * 6

-- Definition for the number of ways two students select different topics
def different_topics_ways : ℕ := 6 * 5

-- Definition for the probability of selecting different topics
def probability_different_topics : ℚ := different_topics_ways / total_ways

-- The statement to be proved in Lean
theorem probability_correct :
  probability_different_topics = 5 / 6 := 
sorry

end probability_correct_l62_62588


namespace line_length_l62_62344

theorem line_length (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := 
by
  sorry

end line_length_l62_62344


namespace fitness_center_cost_effectiveness_l62_62076

noncomputable def f (x : ℝ) : ℝ := 5 * x

noncomputable def g (x : ℝ) : ℝ :=
  if 15 ≤ x ∧ x ≤ 30 then 90 
  else 2 * x + 30

def cost_comparison (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : Prop :=
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x)

theorem fitness_center_cost_effectiveness (x : ℝ) (h1 : 15 ≤ x) (h2 : x ≤ 40) : cost_comparison x h1 h2 :=
by
  sorry

end fitness_center_cost_effectiveness_l62_62076


namespace positive_value_of_X_l62_62493

-- Definition for the problem's conditions
def hash (X Y : ℝ) : ℝ := X^2 + Y^2

-- Statement of the proof problem
theorem positive_value_of_X (X : ℝ) (h : hash X 7 = 170) : X = 11 :=
by
  sorry

end positive_value_of_X_l62_62493


namespace calculate_expression_l62_62607

theorem calculate_expression (y : ℤ) (hy : y = 2) : (3 * y + 4)^2 = 100 :=
by
  sorry

end calculate_expression_l62_62607


namespace age_ratio_3_2_l62_62737

/-
Define variables: 
  L : ℕ -- Liam's current age
  M : ℕ -- Mia's current age
  y : ℕ -- number of years until the age ratio is 3:2
-/

theorem age_ratio_3_2 (L M : ℕ) 
  (h1 : L - 4 = 2 * (M - 4)) 
  (h2 : L - 10 = 3 * (M - 10)) 
  (h3 : ∃ y, (L + y) * 2 = (M + y) * 3) : 
  ∃ y, y = 8 :=
by
  sorry

end age_ratio_3_2_l62_62737


namespace abs_sub_self_nonneg_l62_62906

theorem abs_sub_self_nonneg (a : ℚ) : (|a| - a) ≥ 0 :=
by
  sorry

end abs_sub_self_nonneg_l62_62906


namespace sum_of_natural_numbers_l62_62425

theorem sum_of_natural_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end sum_of_natural_numbers_l62_62425


namespace polynomial_divisible_x_minus_2_l62_62139

theorem polynomial_divisible_x_minus_2 (m : ℝ) : 
  (3 * 2^2 - 9 * 2 + m = 0) → m = 6 :=
by
  sorry

end polynomial_divisible_x_minus_2_l62_62139


namespace price_36kg_apples_l62_62124

-- Definitions based on given conditions
def cost_per_kg_first_30 (l : ℕ) (n₁ : ℕ) (total₁ : ℕ) : Prop :=
  n₁ = 10 ∧ l = total₁ / n₁

def total_cost_33kg (l q : ℕ) (total₂ : ℕ) : Prop :=
  30 * l + 3 * q = total₂

-- Question to prove
def total_cost_36kg (l q : ℕ) (cost_36 : ℕ) : Prop :=
  30 * l + 6 * q = cost_36

theorem price_36kg_apples (l q cost_36 : ℕ) :
  (cost_per_kg_first_30 l 10 200) →
  (total_cost_33kg l q 663) →
  cost_36 = 726 :=
by
  intros h₁ h₂
  sorry

end price_36kg_apples_l62_62124


namespace ratio_Bill_to_Bob_l62_62976

-- Define the shares
def Bill_share : ℕ := 300
def Bob_share : ℕ := 900

-- The theorem statement
theorem ratio_Bill_to_Bob : Bill_share / Bob_share = 1 / 3 := by
  sorry

end ratio_Bill_to_Bob_l62_62976


namespace value_of_x_squared_plus_reciprocal_squared_l62_62320

theorem value_of_x_squared_plus_reciprocal_squared (x : ℝ) (hx : 0 < x) (h : x + 1/x = Real.sqrt 2020) : x^2 + 1/x^2 = 2018 :=
sorry

end value_of_x_squared_plus_reciprocal_squared_l62_62320


namespace range_of_m_l62_62757

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x^4) / Real.log 3
noncomputable def g (x : ℝ) : ℝ := x * f x

theorem range_of_m (m : ℝ) : g (1 - m) < g (2 * m) → m > 1 / 3 :=
  by
  sorry

end range_of_m_l62_62757


namespace total_papers_delivered_l62_62329

-- Definitions based on given conditions
def papers_saturday : ℕ := 45
def papers_sunday : ℕ := 65
def total_papers := papers_saturday + papers_sunday

-- The statement we need to prove
theorem total_papers_delivered : total_papers = 110 := by
  -- Proof steps would go here
  sorry

end total_papers_delivered_l62_62329


namespace circle_diameter_eq_l62_62004

-- Definitions
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0
def point_A (x y : ℝ) : Prop := x = 0 ∧ y = 3
def point_B (x y : ℝ) : Prop := x = -4 ∧ y = 0
def midpoint_AB (x y : ℝ) : Prop := x = -2 ∧ y = 3 / 2 -- Midpoint of A(0,3) and B(-4,0)
def diameter_AB : ℝ := 5

-- The equation of the circle with diameter AB
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 3 * y = 0

-- The proof statement
theorem circle_diameter_eq :
  (∃ A B : ℝ × ℝ, point_A A.1 A.2 ∧ point_B B.1 B.2 ∧ 
                   midpoint_AB ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧ diameter_AB = 5) →
  (∀ x y : ℝ, circle_eq x y) :=
sorry

end circle_diameter_eq_l62_62004


namespace alternating_colors_probability_l62_62104

def box_contains_five_white_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 1) = 5
def box_contains_five_black_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 0) = 5
def balls_drawn_one_at_a_time : Prop := true -- This condition is trivially satisfied without more specific constraints

theorem alternating_colors_probability (h1 : box_contains_five_white_balls) (h2 : box_contains_five_black_balls) (h3 : balls_drawn_one_at_a_time) :
  ∃ p : ℚ, p = 1 / 126 :=
sorry

end alternating_colors_probability_l62_62104


namespace sin_cos_product_l62_62804

theorem sin_cos_product (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2) (h₃ : Real.sin x = 3 * Real.cos x) : 
  Real.sin x * Real.cos x = 3 / 10 :=
by
  sorry

end sin_cos_product_l62_62804


namespace find_x_l62_62939

variables (a b c : ℝ)

theorem find_x (h : a ≥ 0) (h' : b ≥ 0) (h'' : c ≥ 0) : 
  ∃ x ≥ 0, x = Real.sqrt ((b - c)^2 - a^2) :=
by
  use Real.sqrt ((b - c)^2 - a^2)
  sorry

end find_x_l62_62939


namespace expand_product_l62_62415

-- Define x as a variable within the real numbers
variable (x : ℝ)

-- Statement of the theorem
theorem expand_product : (x + 3) * (x - 4) = x^2 - x - 12 := 
by 
  sorry

end expand_product_l62_62415


namespace ratio_greater_than_two_ninths_l62_62773

-- Define the conditions
def M : ℕ := 8
def N : ℕ := 36

-- State the theorem
theorem ratio_greater_than_two_ninths : (M : ℚ) / (N : ℚ) > 2 / 9 := 
by {
    -- skipping the proof with sorry
    sorry
}

end ratio_greater_than_two_ninths_l62_62773


namespace find_polygon_sides_l62_62000

theorem find_polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by
  sorry

end find_polygon_sides_l62_62000


namespace negation_of_proposition_l62_62061

variable (l : ℝ)

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x + l ≥ 0) ↔ (∀ x : ℝ, x + l < 0) := by
  sorry

end negation_of_proposition_l62_62061


namespace sum_is_composite_l62_62661

theorem sum_is_composite (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
    (h : a^2 - a * b + b^2 = c^2 - c * d + d^2) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ k * l = a + b + c + d :=
by sorry

end sum_is_composite_l62_62661


namespace probability_at_least_two_boys_one_girl_l62_62389

-- Define what constitutes a family of four children
def family := {s : Fin 4 → Bool // ∃ (b g : Fin 4), b ≠ g}

-- Define the probability equation
noncomputable def probability_of_boy_or_girl : ℚ := 1 / 2

-- Define what it means to have at least two boys and one girl
def at_least_two_boys_one_girl (f : family) : Prop :=
  ∃ (count_boys count_girls : ℕ), count_boys + count_girls = 4 
  ∧ count_boys ≥ 2 
  ∧ count_girls ≥ 1

-- Calculate the probability
theorem probability_at_least_two_boys_one_girl : 
  (∃ (f : family), at_least_two_boys_one_girl f) →
  probability_of_boy_or_girl ^ 4 * ( (6 / 16 : ℚ) + (4 / 16 : ℚ) + (1 / 16 : ℚ) ) = 11 / 16 :=
by
  sorry

end probability_at_least_two_boys_one_girl_l62_62389


namespace average_increase_fraction_l62_62792

-- First, we define the given conditions:
def incorrect_mark : ℕ := 82
def correct_mark : ℕ := 62
def number_of_students : ℕ := 80

-- We state the theorem to prove that the fraction by which the average marks increased is 1/4. 
theorem average_increase_fraction (incorrect_mark correct_mark : ℕ) (number_of_students : ℕ) :
  (incorrect_mark - correct_mark) / number_of_students = 1 / 4 :=
by
  sorry

end average_increase_fraction_l62_62792


namespace fourth_group_trees_l62_62876

theorem fourth_group_trees (x : ℕ) :
  5 * 13 = 12 + 15 + 12 + x + 11 → x = 15 :=
by
  sorry

end fourth_group_trees_l62_62876


namespace number_of_arrangements_of_six_students_l62_62348

/-- A and B cannot stand together -/
noncomputable def arrangements_A_B_not_together (n: ℕ) (A B: ℕ) : ℕ :=
  if n = 6 then 480 else 0

theorem number_of_arrangements_of_six_students :
  arrangements_A_B_not_together 6 1 2 = 480 :=
sorry

end number_of_arrangements_of_six_students_l62_62348


namespace train_pass_time_correct_l62_62704

noncomputable def train_time_to_pass_post (length_of_train : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * (5 / 18)
  length_of_train / speed_mps

theorem train_pass_time_correct :
  train_time_to_pass_post 60 36 = 6 := by
  sorry

end train_pass_time_correct_l62_62704


namespace rhombus_side_length_l62_62072

variable {L S : ℝ}

theorem rhombus_side_length (hL : 0 ≤ L) (hS : 0 ≤ S) :
  (∃ m : ℝ, m = 1 / 2 * Real.sqrt (L^2 - 4 * S)) :=
sorry

end rhombus_side_length_l62_62072


namespace ducks_joined_l62_62237

theorem ducks_joined (initial_ducks total_ducks ducks_joined : ℕ) 
  (h_initial: initial_ducks = 13)
  (h_total: total_ducks = 33) :
  initial_ducks + ducks_joined = total_ducks → ducks_joined = 20 :=
by
  intros h_equation
  rw [h_initial, h_total] at h_equation
  sorry

end ducks_joined_l62_62237


namespace alpha_arctan_l62_62318

open Real

theorem alpha_arctan {α : ℝ} (h1 : α ∈ Set.Ioo 0 (π/4)) (h2 : tan (α + (π/4)) = 2 * cos (2 * α)) : 
  α = arctan (2 - sqrt 3) := by
  sorry

end alpha_arctan_l62_62318


namespace problem_equivalent_l62_62014

-- Define the problem conditions
def an (n : ℕ) : ℤ := -4 * n + 2

-- Arithmetic sequence: given conditions
axiom arith_seq_cond1 : an 2 + an 7 = -32
axiom arith_seq_cond2 : an 3 + an 8 = -40

-- Suppose the sequence {an + bn} is geometric with first term 1 and common ratio 2
def geom_seq (n : ℕ) : ℤ := 2 ^ (n - 1)
def bn (n : ℕ) : ℤ := geom_seq n - an n

-- To prove: sum of the first n terms of {bn}, denoted as Sn
def Sn (n : ℕ) : ℤ := (n * (2 + 4 * n - 2)) / 2 + (1 - 2 ^ n) / (1 - 2)

theorem problem_equivalent (n : ℕ) :
  an 2 + an 7 = -32 ∧
  an 3 + an 8 = -40 ∧
  (∀ n : ℕ, an n + bn n = geom_seq n) →
  Sn n = 2 * n ^ 2 + 2 ^ n - 1 :=
by
  intros h
  sorry

end problem_equivalent_l62_62014


namespace john_collects_crabs_l62_62315

-- Definitions for the conditions
def baskets_per_week : ℕ := 3
def crabs_per_basket : ℕ := 4
def price_per_crab : ℕ := 3
def total_income : ℕ := 72

-- Definition for the question
def times_per_week_to_collect (baskets_per_week crabs_per_basket price_per_crab total_income : ℕ) : ℕ :=
  (total_income / price_per_crab) / (baskets_per_week * crabs_per_basket)

-- The theorem statement
theorem john_collects_crabs (h1 : baskets_per_week = 3) (h2 : crabs_per_basket = 4) (h3 : price_per_crab = 3) (h4 : total_income = 72) :
  times_per_week_to_collect baskets_per_week crabs_per_basket price_per_crab total_income = 2 :=
by
  sorry

end john_collects_crabs_l62_62315


namespace positive_solution_unique_l62_62670

theorem positive_solution_unique (x y z : ℝ) (h1 : x + y^2 + z^3 = 3) (h2 : y + z^2 + x^3 = 3) (h3 : z + x^2 + y^3 = 3) (h_pos : x > 0 ∧ y > 0 ∧ z > 0) : x = 1 ∧ y = 1 ∧ z = 1 := 
by {
  -- The proof would go here, but for now, we just state sorry.
  sorry,
}

end positive_solution_unique_l62_62670


namespace man_l62_62718

-- Given conditions
def V_m := 15 - 3.2
def V_c := 3.2
def man's_speed_with_current : Real := 15

-- Required to prove
def man's_speed_against_current := V_m - V_c

theorem man's_speed_against_current_is_correct : man's_speed_against_current = 8.6 := by
  sorry

end man_l62_62718


namespace impossible_score_53_l62_62777

def quizScoring (total_questions correct_answers incorrect_answers unanswered_questions score: ℤ) : Prop :=
  total_questions = 15 ∧
  correct_answers + incorrect_answers + unanswered_questions = 15 ∧
  score = 4 * correct_answers - incorrect_answers ∧
  unanswered_questions ≥ 0 ∧ correct_answers ≥ 0 ∧ incorrect_answers ≥ 0

theorem impossible_score_53 :
  ¬ ∃ (correct_answers incorrect_answers unanswered_questions : ℤ), quizScoring 15 correct_answers incorrect_answers unanswered_questions 53 := 
sorry

end impossible_score_53_l62_62777


namespace circles_intersect_l62_62005

-- Definition of the first circle
def circle1 (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Definition of the second circle
def circle2 (x y : ℝ) (r : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Statement proving the range of r for which the circles intersect
theorem circles_intersect (r : ℝ) (h : r > 0) : (∃ x y : ℝ, circle1 x y r ∧ circle2 x y r) → (2 ≤ r ∧ r ≤ 12) :=
by
  -- Definition of the distance between centers and conditions for intersection
  sorry

end circles_intersect_l62_62005


namespace fixed_cost_is_50000_l62_62112

-- Definition of conditions
def fixed_cost : ℕ := 50000
def books_sold : ℕ := 10000
def revenue_per_book : ℕ := 9 - 4

-- Theorem statement: Proving that the fixed cost of making books is $50,000
theorem fixed_cost_is_50000 (F : ℕ) (h : revenue_per_book * books_sold = F) : 
  F = fixed_cost :=
by sorry

end fixed_cost_is_50000_l62_62112


namespace find_b_l62_62444

theorem find_b (a b : ℝ) (h1 : 3 * a - 2 = 1) (h2 : 2 * b - 3 * a = 2) : b = 5 / 2 := 
by 
  sorry

end find_b_l62_62444


namespace probability_different_topics_l62_62584

theorem probability_different_topics (topics : ℕ) (h : topics = 6) : 
  let total_combinations := topics * topics,
      different_combinations := topics * (topics - 1) 
  in (different_combinations / total_combinations : ℚ) = 5 / 6 :=
by
  -- This is just a place holder proof.
  sorry

end probability_different_topics_l62_62584


namespace sin_alpha_value_l62_62440

-- Define the given conditions
def α : ℝ := sorry -- α is an acute angle
def β : ℝ := sorry -- β has an unspecified value

-- Given conditions translated to Lean
def condition1 : Prop := 2 * Real.tan (Real.pi - α) - 3 * Real.cos (Real.pi / 2 + β) + 5 = 0
def condition2 : Prop := Real.tan (Real.pi + α) + 6 * Real.sin (Real.pi + β) = 1

-- Acute angle condition
def α_acute : Prop := 0 < α ∧ α < Real.pi / 2

-- The proof statement
theorem sin_alpha_value (h1 : condition1) (h2 : condition2) (h3 : α_acute) : Real.sin α = 3 * Real.sqrt 10 / 10 :=
by sorry

end sin_alpha_value_l62_62440


namespace wendy_total_gas_to_add_l62_62357

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end wendy_total_gas_to_add_l62_62357


namespace triangle_type_l62_62179

theorem triangle_type (a b c : ℝ) (A B C : ℝ) (h1 : A = 30) (h2 : a = 2 * b ∨ b = 2 * c ∨ c = 2 * a) :
  (C > 90 ∨ B > 90) ∨ C = 90 :=
sorry

end triangle_type_l62_62179


namespace collinear_vectors_parallel_right_angle_triangle_abc_l62_62167

def vec_ab (k : ℝ) : ℝ × ℝ := (2 - k, -1)
def vec_ac (k : ℝ) : ℝ × ℝ := (1, k)

-- Prove that if vectors AB and AC are collinear, then k = 1 ± √2
theorem collinear_vectors_parallel (k : ℝ) :
  (2 - k) * k - 1 = 0 ↔ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2 :=
by
  sorry

def vec_bc (k : ℝ) : ℝ × ℝ := (k - 1, k + 1)

-- Prove that if triangle ABC is right-angled, then k = 1 or k = -1 ± √2
theorem right_angle_triangle_abc (k : ℝ) :
  ( (2 - k) * 1 + (-1) * k = 0 ∨ (k - 1) * 1 + (k + 1) * k = 0 ) ↔ 
  k = 1 ∨ k = -1 + Real.sqrt 2 ∨ k = -1 - Real.sqrt 2 :=
by
  sorry

end collinear_vectors_parallel_right_angle_triangle_abc_l62_62167


namespace rectangle_area_l62_62924

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 :=
by sorry

end rectangle_area_l62_62924


namespace cost_of_ABC_book_l62_62391

theorem cost_of_ABC_book (x : ℕ) 
  (h₁ : 8 = 8)  -- Cost of "TOP" book is 8 dollars
  (h₂ : 13 * 8 = 104)  -- Thirteen "TOP" books sold last week
  (h₃ : 104 - 4 * x = 12)  -- Difference in earnings is $12
  : x = 23 :=
sorry

end cost_of_ABC_book_l62_62391


namespace sam_and_david_licks_l62_62410

theorem sam_and_david_licks :
  let Dan_licks := 58
  let Michael_licks := 63
  let Lance_licks := 39
  let avg_licks := 60
  let total_people := 5
  let total_licks := avg_licks * total_people
  let total_licks_Dan_Michael_Lance := Dan_licks + Michael_licks + Lance_licks
  total_licks - total_licks_Dan_Michael_Lance = 140 := by
  sorry

end sam_and_david_licks_l62_62410


namespace general_formula_T_greater_S_l62_62283

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l62_62283


namespace company_fund_initial_amount_l62_62834

theorem company_fund_initial_amount (n : ℕ) :
  (70 * n + 75 = 80 * n - 20) →
  (n = 9) →
  (80 * n - 20 = 700) :=
by
  intros h1 h2
  rw [h2] at h1
  linarith

end company_fund_initial_amount_l62_62834


namespace sum_derivs_at_zero_is_l62_62494

noncomputable def f (x : ℝ) : ℝ := x * Real.cos x

def f_deriv (n : ℕ) : (ℝ → ℝ) := 
  match n with
  | 0       => f
  | n + 1   => fun x => deriv (f_deriv n) x

def sum_derivs_at_zero : ℝ := (Finset.range 2014).sum (λ k, f_deriv k 0)

theorem sum_derivs_at_zero_is : sum_derivs_at_zero = -1012029 :=
by
  sorry

end sum_derivs_at_zero_is_l62_62494


namespace probability_both_hit_l62_62985

-- Conditions
def prob_A_hits : ℝ := 0.9
def prob_B_hits : ℝ := 0.8

-- Question and proof problem
theorem probability_both_hit : prob_A_hits * prob_B_hits = 0.72 :=
by
  sorry

end probability_both_hit_l62_62985


namespace equivalent_knicks_l62_62765

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end equivalent_knicks_l62_62765


namespace common_number_exists_l62_62981

def sum_of_list (l : List ℚ) : ℚ := l.sum

theorem common_number_exists (l1 l2 : List ℚ) (commonNumber : ℚ) 
    (h1 : l1.length = 5) 
    (h2 : l2.length = 5) 
    (h3 : sum_of_list l1 / 5 = 7) 
    (h4 : sum_of_list l2 / 5 = 10) 
    (h5 : (sum_of_list l1 + sum_of_list l2 - commonNumber) / 9 = 74 / 9) 
    : commonNumber = 11 :=
sorry

end common_number_exists_l62_62981


namespace solve_for_y_l62_62145

theorem solve_for_y : (12^3 * 6^2) / 432 = 144 := 
by 
  sorry

end solve_for_y_l62_62145


namespace ordered_pairs_l62_62412

theorem ordered_pairs (a b : ℝ) (hapos : 0 < a) (hbpos : 0 < b) (x : ℕ → ℝ)
  (h : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a * x (n + 1) - b * x n| < ε) :
  (a = 0 ∧ 0 < b) ∨ (0 < a ∧ |b / a| < 1) :=
sorry

end ordered_pairs_l62_62412


namespace chromium_percentage_l62_62184

theorem chromium_percentage (x : ℝ) : 
  (15 * x / 100 + 35 * 8 / 100 = 50 * 8.6 / 100) → 
  x = 10 := 
sorry

end chromium_percentage_l62_62184


namespace import_rate_for_rest_of_1997_l62_62889

theorem import_rate_for_rest_of_1997
    (import_1996: ℝ)
    (import_first_two_months_1997: ℝ)
    (excess_imports_1997: ℝ)
    (import_rate_first_two_months: ℝ)
    (expected_total_imports_1997: ℝ)
    (remaining_imports_1997: ℝ)
    (R: ℝ):
    excess_imports_1997 = 720e6 →
    expected_total_imports_1997 = import_1996 + excess_imports_1997 →
    remaining_imports_1997 = expected_total_imports_1997 - import_first_two_months_1997 →
    10 * R = remaining_imports_1997 →
    R = 180e6 :=
by
    intros h_import1996 h_import_first_two_months h_excess_imports h_import_rate_first_two_months 
           h_expected_total_imports h_remaining_imports h_equation
    sorry

end import_rate_for_rest_of_1997_l62_62889


namespace mahmoud_gets_at_least_two_heads_l62_62201

def probability_of_at_least_two_heads := 1 - ((1/2)^5 + 5 * (1/2)^5)

theorem mahmoud_gets_at_least_two_heads (n : ℕ) (hn : n = 5) :
  probability_of_at_least_two_heads = 13 / 16 :=
by
  simp only [probability_of_at_least_two_heads, hn]
  sorry

end mahmoud_gets_at_least_two_heads_l62_62201


namespace server_multiplications_in_half_hour_l62_62246

theorem server_multiplications_in_half_hour : 
  let rate := 5000
  let seconds_in_half_hour := 1800
  rate * seconds_in_half_hour = 9000000 := by
  sorry

end server_multiplications_in_half_hour_l62_62246


namespace total_boxes_sold_l62_62575

-- Define the number of boxes of plain cookies
def P : ℝ := 793.375

-- Define the combined value of cookies sold
def total_value : ℝ := 1586.75

-- Define the cost per box of each type of cookie
def cost_chocolate_chip : ℝ := 1.25
def cost_plain : ℝ := 0.75

-- State the theorem to prove
theorem total_boxes_sold :
  ∃ C : ℝ, cost_chocolate_chip * C + cost_plain * P = total_value ∧ C + P = 1586.75 :=
by
  sorry

end total_boxes_sold_l62_62575


namespace monotonically_increasing_interval_l62_62912

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem monotonically_increasing_interval : 
  ∃ (a b : ℝ), a = -Real.pi / 3 ∧ b = Real.pi / 6 ∧ ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x < f y :=
by
  sorry

end monotonically_increasing_interval_l62_62912


namespace a_minus_b_is_30_l62_62561

-- Definition of the sum of the arithmetic series
def sum_arithmetic_series (first last : ℕ) (n : ℕ) : ℕ :=
  (n * (first + last)) / 2

-- Definitions based on problem conditions
def a : ℕ := sum_arithmetic_series 2 60 30
def b : ℕ := sum_arithmetic_series 1 59 30

theorem a_minus_b_is_30 : a - b = 30 :=
  by sorry

end a_minus_b_is_30_l62_62561


namespace bulbs_sampling_l62_62099

theorem bulbs_sampling (total_bulbs : ℕ) (twenty_w_bulbs forty_w_bulbs sixty_w_bulbs sample_bulbs : ℕ)
  (ratio_twenty forty sixty : ℚ)
  (h_total : total_bulbs = 400)
  (h_ratio : ratio_twenty = 4 ∧ ratio_forty = 3 ∧ ratio_sixty = 1)
  (h_sum_ratio : ratio_twenty + ratio_forty + ratio_sixty = 8) 
  (h_type_count : twenty_w_bulbs = (ratio_twenty / 8 : ℚ) * total_bulbs ∧ 
                   forty_w_bulbs = (ratio_forty / 8 : ℚ) * total_bulbs ∧ 
                   sixty_w_bulbs = (ratio_sixty / 8 : ℚ) * total_bulbs)
  (h_sample : sample_bulbs = 40) :
  let sample_twenty := (twenty_w_bulbs / total_bulbs : ℚ) * sample_bulbs,
      sample_forty := (forty_w_bulbs / total_bulbs : ℚ) * sample_bulbs,
      sample_sixty := (sixty_w_bulbs / total_bulbs : ℚ) * sample_bulbs in
  sample_twenty = 20 ∧ sample_forty = 15 ∧ sample_sixty = 5 :=
by sorry

end bulbs_sampling_l62_62099


namespace alcohol_percentage_new_mixture_l62_62097

namespace AlcoholMixtureProblem

def original_volume : ℝ := 3
def alcohol_percentage : ℝ := 0.33
def additional_water_volume : ℝ := 1
def new_volume : ℝ := original_volume + additional_water_volume
def alcohol_amount : ℝ := original_volume * alcohol_percentage

theorem alcohol_percentage_new_mixture : (alcohol_amount / new_volume) * 100 = 24.75 := by
  sorry

end AlcoholMixtureProblem

end alcohol_percentage_new_mixture_l62_62097


namespace right_triangle_side_length_l62_62752

theorem right_triangle_side_length (area : ℝ) (side1 : ℝ) (side2 : ℝ) (h_area : area = 8) (h_side1 : side1 = Real.sqrt 10) (h_area_eq : area = 0.5 * side1 * side2) :
  side2 = 1.6 * Real.sqrt 10 :=
by 
  sorry

end right_triangle_side_length_l62_62752


namespace boat_speed_in_still_water_l62_62782

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 :=
by 
  sorry

end boat_speed_in_still_water_l62_62782


namespace fraction_of_age_l62_62789

theorem fraction_of_age (jane_age_current : ℕ) (years_since_babysit : ℕ) (age_oldest_babysat_current : ℕ) :
  jane_age_current = 32 →
  years_since_babysit = 12 →
  age_oldest_babysat_current = 23 →
  ∃ (f : ℚ), f = 11 / 20 :=
by
  intros
  sorry

end fraction_of_age_l62_62789


namespace math_problem_statements_l62_62971

theorem math_problem_statements :
  (∀ a : ℝ, (a = -a) → (a = 0)) ∧
  (∀ b : ℝ, (1 / b = b) ↔ (b = 1 ∨ b = -1)) ∧
  (∀ c : ℝ, (c < -1) → (1 / c > c)) ∧
  (∀ d : ℝ, (d > 1) → (1 / d < d)) ∧
  (∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → n ≤ m) :=
by {
  sorry
}

end math_problem_statements_l62_62971


namespace max_quadratic_function_l62_62211

def quadratic_function (x : ℝ) : ℝ := -3 * x^2 + 12 * x - 5

theorem max_quadratic_function : ∃ x, quadratic_function x = 7 ∧ ∀ x', quadratic_function x' ≤ 7 :=
by
  sorry

end max_quadratic_function_l62_62211


namespace pacworm_probability_within_08_cm_of_edge_l62_62951

/-- 
  Pacworm is situated in the center of a cube-shaped piece of cheese with an edge length of 5 cm.
  The worm moves exactly 1 cm at a time in a direction parallel to one of the edges, and then changes direction.
  Each time it changes direction, it ensures there is more than 1 cm of untouched cheese in front of it.
  At both the initial movement and each change of direction, the worm chooses its direction with equal probability.
  We want to prove that the probability that after traveling 5 cm, the worm will be within 0.8 cm of one of the edges is 5/24.
-/
theorem pacworm_probability_within_08_cm_of_edge :
  let edge_length := 5
  let travel_distance := 5
  let change_count := 5
  let proximity := 0.8
  let total_prob := (5 / 24 : ℝ) in
  -- Considering the initial conditions and problem constraints
  -- Prove that the required probability is 5/24
  sorry

end pacworm_probability_within_08_cm_of_edge_l62_62951


namespace real_polynomial_has_exactly_one_real_solution_l62_62421

theorem real_polynomial_has_exactly_one_real_solution:
  ∀ a : ℝ, ∃! x : ℝ, x^3 - a * x^2 - 3 * a * x + a^2 - 1 = 0 := 
by
  sorry

end real_polynomial_has_exactly_one_real_solution_l62_62421


namespace ratio_M_N_l62_62457

theorem ratio_M_N (M Q P N : ℝ) (hM : M = 0.40 * Q) (hQ : Q = 0.25 * P) (hN : N = 0.60 * P) (hP : P ≠ 0) : 
  (M / N) = (1 / 6) := 
by 
  sorry

end ratio_M_N_l62_62457


namespace five_letter_sequences_l62_62595

-- Define the quantities of each vowel.
def quantity_vowel_A : Nat := 3
def quantity_vowel_E : Nat := 4
def quantity_vowel_I : Nat := 5
def quantity_vowel_O : Nat := 6
def quantity_vowel_U : Nat := 7

-- Define the number of choices for each letter in a five-letter sequence.
def choices_per_letter : Nat := 5

-- Define the total number of five-letter sequences.
noncomputable def total_sequences : Nat := choices_per_letter ^ 5

-- Prove that the number of five-letter sequences is 3125.
theorem five_letter_sequences : total_sequences = 3125 :=
by sorry

end five_letter_sequences_l62_62595


namespace polygon_sidedness_l62_62843

-- Define the condition: the sum of the interior angles of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Given condition
def given_condition : ℝ := 1260

-- Target proposition to prove
theorem polygon_sidedness (n : ℕ) (h : sum_of_interior_angles n = given_condition) : n = 9 :=
by
  sorry

end polygon_sidedness_l62_62843


namespace boys_in_class_l62_62177

theorem boys_in_class
  (g b : ℕ)
  (h_ratio : g = (3 * b) / 5)
  (h_total : g + b = 32) :
  b = 20 :=
sorry

end boys_in_class_l62_62177


namespace line_equation_l62_62461

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4)) 
  (h_intercept_sum : ∃ b c, b + c = 0 ∧ (∀ x y, A.1 * x + A.2 * y = 1 ∨ A.1 * x + A.2 * y = -1)) :
  (∃ m n, m = 4 ∧ n = -1 ∧ (∀ x y, m * x + n * y = 0)) ∨ 
  (∃ p q r, p = 1 ∧ q = -1 ∧ r = 3 ∧ (∀ x y, p * x + q * y + r = 0)) :=
by
  sorry

end line_equation_l62_62461


namespace car_travel_distance_l62_62572

-- Define the original gas mileage as x
variable (x : ℝ) (D : ℝ)

-- Define the conditions
def initial_condition : Prop := D = 12 * x
def revised_condition : Prop := D = 10 * (x + 2)

-- The proof goal
theorem car_travel_distance
  (h1 : initial_condition x D)
  (h2 : revised_condition x D) :
  D = 120 := by
  sorry

end car_travel_distance_l62_62572


namespace probability_A_will_receive_2_awards_l62_62346

def classes := Fin 4
def awards := 8

-- The number of ways to distribute 4 remaining awards to 4 classes
noncomputable def total_distributions : ℕ :=
  Nat.choose (awards - 4 + 4 - 1) (4 - 1)

-- The number of ways when class A receives exactly 2 awards
noncomputable def favorable_distributions : ℕ :=
  Nat.choose (2 + 3 - 1) (4 - 1)

-- The probability that class A receives exactly 2 out of 8 awards
noncomputable def probability_A_receives_2_awards : ℚ :=
  favorable_distributions / total_distributions

theorem probability_A_will_receive_2_awards :
  probability_A_receives_2_awards = 2 / 7 := by
  sorry

end probability_A_will_receive_2_awards_l62_62346


namespace polygon_divided_l62_62111

theorem polygon_divided (p q r : ℕ) : p - q + r = 1 :=
sorry

end polygon_divided_l62_62111


namespace egg_laying_hens_l62_62507

theorem egg_laying_hens (total_chickens : ℕ) (roosters : ℕ) (non_laying_hens : ℕ) :
  total_chickens = 325 →
  roosters = 28 →
  non_laying_hens = 20 →
  (total_chickens - roosters - non_laying_hens = 277) :=
by
  intros
  sorry

end egg_laying_hens_l62_62507


namespace speed_second_hour_l62_62690

noncomputable def speed_in_first_hour : ℝ := 95
noncomputable def average_speed : ℝ := 77.5
noncomputable def total_time : ℝ := 2
def speed_in_second_hour : ℝ := sorry -- to be deduced

theorem speed_second_hour :
  speed_in_second_hour = 60 :=
by
  sorry

end speed_second_hour_l62_62690


namespace product_gcd_lcm_150_90_l62_62392

theorem product_gcd_lcm_150_90 (a b : ℕ) (h1 : a = 150) (h2 : b = 90): Nat.gcd a b * Nat.lcm a b = a * b := by
  rw [h1, h2]
  sorry

end product_gcd_lcm_150_90_l62_62392


namespace area_of_given_right_triangle_l62_62696

variable (a c : ℝ)
def is_right_triangle (a c : ℝ) : Prop := ∃ b : ℝ, a^2 + b^2 = c^2
def area_of_right_triangle (a : ℝ) (b : ℝ) : ℝ := (1 / 2) * a * b

theorem area_of_given_right_triangle (h : is_right_triangle 15 17) : 
  ∃ (b : ℝ), (15^2 + b^2 = 17^2) ∧ (area_of_right_triangle 15 b = 60) :=
by sorry

end area_of_given_right_triangle_l62_62696


namespace seq_sum_l62_62620

theorem seq_sum (r : ℚ) (x y : ℚ) (h1 : r = 1 / 4)
    (h2 : 1024 * r = x) (h3 : x * r = y) : 
    x + y = 320 := by
  sorry

end seq_sum_l62_62620


namespace range_of_m_l62_62451

theorem range_of_m (m : ℝ) : (∀ x > 1, 2*x + m + 8/(x-1) > 0) → m > -10 := 
by
  -- The formal proof will be completed here.
  sorry

end range_of_m_l62_62451


namespace exist_indices_l62_62381

-- Define the sequence and the conditions.
variable (x : ℕ → ℤ)
variable (H1 : x 1 = 1)
variable (H2 : ∀ n : ℕ, x n < x (n + 1) ∧ x (n + 1) ≤ 2 * n)

theorem exist_indices (k : ℕ) (hk : 0 < k) :
  ∃ r s : ℕ, x r - x s = k := 
sorry

end exist_indices_l62_62381


namespace candy_factory_days_l62_62340

noncomputable def candies_per_hour := 50
noncomputable def total_candies := 4000
noncomputable def working_hours_per_day := 10
noncomputable def total_hours_needed := total_candies / candies_per_hour
noncomputable def total_days_needed := total_hours_needed / working_hours_per_day

theorem candy_factory_days :
  total_days_needed = 8 := 
by
  -- (Proof steps will be filled here)
  sorry

end candy_factory_days_l62_62340


namespace f_always_positive_l62_62857

def f (x : ℝ) : ℝ := x^2 + 3 * x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := 
by 
  sorry

end f_always_positive_l62_62857


namespace cost_of_berries_and_cheese_l62_62946

variables (b m l c : ℕ)

theorem cost_of_berries_and_cheese (h1 : b + m + l + c = 25)
                                  (h2 : m = 2 * l)
                                  (h3 : c = b + 2) : 
                                  b + c = 10 :=
by {
  -- proof omitted, this is just the statement
  sorry
}

end cost_of_berries_and_cheese_l62_62946


namespace wendy_total_gas_to_add_l62_62358

-- Conditions as definitions
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_current_gas : ℕ := truck_tank_capacity / 2
def car_current_gas : ℕ := car_tank_capacity / 3

-- The proof problem statement
theorem wendy_total_gas_to_add :
  (truck_tank_capacity - truck_current_gas) + (car_tank_capacity - car_current_gas) = 18 := 
by
  sorry

end wendy_total_gas_to_add_l62_62358


namespace vladimir_can_invest_more_profitably_l62_62092

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end vladimir_can_invest_more_profitably_l62_62092


namespace arithmetic_seq_a7_l62_62310

theorem arithmetic_seq_a7 (a : ℕ → ℕ) (d : ℕ)
  (h1 : a 1 = 2)
  (h2 : a 3 + a 5 = 8)
  (h3 : ∀ n, a n = a 1 + (n - 1) * d) :
  a 7 = 6 := by
  sorry

end arithmetic_seq_a7_l62_62310


namespace giant_slide_wait_is_15_l62_62731

noncomputable def wait_time_for_giant_slide
  (hours_at_carnival : ℕ) 
  (roller_coaster_wait : ℕ)
  (tilt_a_whirl_wait : ℕ)
  (rides_roller_coaster : ℕ)
  (rides_tilt_a_whirl : ℕ)
  (rides_giant_slide : ℕ) : ℕ :=
  
  (hours_at_carnival * 60 - (roller_coaster_wait * rides_roller_coaster + tilt_a_whirl_wait * rides_tilt_a_whirl)) / rides_giant_slide

theorem giant_slide_wait_is_15 :
  wait_time_for_giant_slide 4 30 60 4 1 4 = 15 := 
sorry

end giant_slide_wait_is_15_l62_62731


namespace organization_population_after_six_years_l62_62477

theorem organization_population_after_six_years :
  ∀ (b : ℕ → ℕ),
  (b 0 = 20) →
  (∀ k, b (k + 1) = 3 * (b k - 5) + 5) →
  b 6 = 10895 :=
by
  intros b h0 hr
  sorry

end organization_population_after_six_years_l62_62477


namespace triangle_inequality_difference_l62_62016

theorem triangle_inequality_difference :
  (∀ (x : ℤ), (x + 7 > 9) ∧ (x + 9 > 7) ∧ (7 + 9 > x) → (3 ≤ x ∧ x ≤ 15) ∧ (15 - 3 = 12)) :=
by
  sorry

end triangle_inequality_difference_l62_62016


namespace cos_180_eq_neg_one_l62_62426

/-- The cosine of 180 degrees is -1. -/
theorem cos_180_eq_neg_one : Real.cos (180 * Real.pi / 180) = -1 :=
by sorry

end cos_180_eq_neg_one_l62_62426


namespace kims_morning_routine_total_time_l62_62795

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end kims_morning_routine_total_time_l62_62795


namespace students_in_school_l62_62186

variable (S : ℝ)
variable (W : ℝ)
variable (L : ℝ)

theorem students_in_school {S W L : ℝ} 
  (h1 : W = 0.55 * 0.25 * S)
  (h2 : L = 0.45 * 0.25 * S)
  (h3 : W = L + 50) : 
  S = 2000 := 
sorry

end students_in_school_l62_62186


namespace correct_option_b_l62_62545

theorem correct_option_b (a : ℝ) : 
  (-2 * a) ^ 3 = -8 * a ^ 3 :=
by sorry

end correct_option_b_l62_62545


namespace solution_set_l62_62905

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- conditions
axiom differentiable_on_f : ∀ x < 0, DifferentiableAt ℝ f x
axiom derivative_f_x : ∀ x < 0, deriv f x = f' x

axiom condition_3fx_xf'x : ∀ x < 0, 3 * f x + x * f' x > 0

-- goal
theorem solution_set :
  ∀ x, (-2020 < x ∧ x < -2017) ↔ ((x + 2017)^3 * f (x + 2017) + 27 * f (-3) > 0) :=
by
  sorry

end solution_set_l62_62905


namespace problem1_problem2_l62_62707

-- Problem 1
theorem problem1 : 3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem2 : abs (Real.sqrt 3 - Real.sqrt 2) + abs (Real.sqrt 3 - 2) + Real.sqrt 4 = 4 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l62_62707


namespace concert_cost_l62_62040

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end concert_cost_l62_62040


namespace compare_log_inequalities_l62_62637

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem compare_log_inequalities (a x1 x2 : ℝ) 
  (ha_pos : a > 0) (ha_neq_one : a ≠ 1) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (a > 1 → 1 / 2 * (f a x1 + f a x2) ≤ f a ((x1 + x2) / 2)) ∧
  (0 < a ∧ a < 1 → 1 / 2 * (f a x1 + f a x2) ≥ f a ((x1 + x2) / 2)) :=
by { sorry }

end compare_log_inequalities_l62_62637


namespace transmitted_word_is_PAROHOD_l62_62897

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end transmitted_word_is_PAROHOD_l62_62897


namespace sum_of_coefficients_l62_62051

theorem sum_of_coefficients (a b c d : ℝ) (f : ℝ → ℝ)
    (h1 : ∀ x, f (x + 2) = 2*x^3 + 5*x^2 + 3*x + 6)
    (h2 : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) :
  a + b + c + d = 6 :=
by sorry

end sum_of_coefficients_l62_62051


namespace general_formula_sums_inequality_for_large_n_l62_62277

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l62_62277


namespace translation_up_by_one_l62_62984

def initial_function (x : ℝ) : ℝ := x^2

def translated_function (x : ℝ) : ℝ := x^2 + 1

theorem translation_up_by_one (x : ℝ) : translated_function x = initial_function x + 1 :=
by sorry

end translation_up_by_one_l62_62984


namespace last_two_digits_of_factorial_sum_l62_62994

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l62_62994


namespace min_value_of_x_plus_2y_l62_62899

noncomputable def min_value_condition (x y : ℝ) : Prop :=
x > -1 ∧ y > 0 ∧ (1 / (x + 1) + 2 / y = 1)

theorem min_value_of_x_plus_2y (x y : ℝ) (h : min_value_condition x y) : x + 2 * y ≥ 8 :=
sorry

end min_value_of_x_plus_2y_l62_62899


namespace probability_calculation_l62_62801

noncomputable def probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 : ℝ :=
  let total_interval_length := 100
  let intersection_interval_length := 324 - 312.5
  intersection_interval_length / total_interval_length

theorem probability_calculation : probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 = 23 / 200 := by
  sorry

end probability_calculation_l62_62801


namespace max_sin_B_l62_62934

theorem max_sin_B (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (AB BC : ℝ)
    (hAB : AB = 25) (hBC : BC = 20) :
    ∃ sinB : ℝ, sinB = 3 / 5 := sorry

end max_sin_B_l62_62934


namespace tiles_difference_ninth_eighth_rectangle_l62_62720

theorem tiles_difference_ninth_eighth_rectangle : 
  let width (n : Nat) := 2 * n
  let height (n : Nat) := n
  let tiles (n : Nat) := width n * height n
  tiles 9 - tiles 8 = 34 :=
by
  intro width height tiles
  sorry

end tiles_difference_ninth_eighth_rectangle_l62_62720


namespace det_is_even_l62_62023

open Matrix

-- Definitions and assumptions based on given problem conditions
def is_finite_group (G : Type) [Group G] : Prop := Fintype G

def a (G : Type) [Group G] [Fintype G] (x : G) (y : G) : Finset(Fintype.elems(G)) → Matrix Fintype.card _ Fintype.card _
  | xi, xj => if (xi * xj⁻¹) = (xj * xi⁻¹) then 0 else 1

-- The proof problem statement
theorem det_is_even (G : Type) [Group G] [Fintype G] :
  ∀ (x : G) (y : G), (x_1 ... x_n : List G) (a_ij : Fin n → Fin n → \fin 2) (ha_ij : ∀ i j, a_ij i j = if x_i * x_j⁻¹ = x_j * x_i⁻¹ then 0 else 1),
  (det (a_ij)).val % 2 = 0 :=
by 
  sorry

end det_is_even_l62_62023


namespace part_a_exists_part_b_impossible_l62_62578

def gridSize : Nat := 7 * 14
def cellCount (x y : Nat) : Nat := 4 * x + 3 * y
def x_equals_y_condition (x y : Nat) : Prop := x = y
def x_greater_y_condition (x y : Nat) : Prop := x > y

theorem part_a_exists (x y : Nat) (h : cellCount x y = gridSize) : ∃ (x y : Nat), x_equals_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry

theorem part_b_impossible (x y : Nat) (h : cellCount x y = gridSize) : ¬ ∃ (x y : Nat), x_greater_y_condition x y ∧ cellCount x y = gridSize :=
by
  sorry


end part_a_exists_part_b_impossible_l62_62578


namespace subtract_some_number_l62_62961

theorem subtract_some_number
  (x : ℤ)
  (h : 913 - x = 514) :
  514 - x = 115 :=
by {
  sorry
}

end subtract_some_number_l62_62961


namespace find_a3_l62_62159

-- Given conditions
def sequence_sum (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n^2 + n

-- Define the sequence term calculation from the sum function.
def seq_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem find_a3 (S : ℕ → ℕ) (h : sequence_sum S) :
  seq_term S 3 = 6 :=
by
  sorry

end find_a3_l62_62159


namespace num_valid_triples_l62_62616

theorem num_valid_triples : ∃! (count : ℕ), count = 22 ∧
  ∀ k m n : ℕ, (0 ≤ k) ∧ (k ≤ 100) ∧ (0 ≤ m) ∧ (m ≤ 100) ∧ (0 ≤ n) ∧ (n ≤ 100) → 
  (2^m * n - 2^n * m = 2^k) → count = 22 :=
sorry

end num_valid_triples_l62_62616


namespace determine_constants_l62_62140

theorem determine_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → 
    (x^2 - 5*x + 6) / ((x - 1) * (x - 4) * (x - 6)) =
    P / (x - 1) + Q / (x - 4) + R / (x - 6)) →
  P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 :=
by {
  sorry
}

end determine_constants_l62_62140


namespace hyperbola_asymptote_slope_l62_62447

theorem hyperbola_asymptote_slope (m : ℝ) :
  (∀ x y : ℝ, mx^2 + y^2 = 1) →
  (∀ x y : ℝ, y = 2 * x) →
  m = -4 :=
by
  sorry

end hyperbola_asymptote_slope_l62_62447


namespace avg_age_grandparents_is_64_l62_62712

-- Definitions of conditions
def num_grandparents : ℕ := 2
def num_parents : ℕ := 2
def num_grandchildren : ℕ := 3
def num_family_members : ℕ := num_grandparents + num_parents + num_grandchildren

def avg_age_parents : ℕ := 39
def avg_age_grandchildren : ℕ := 6
def avg_age_family : ℕ := 32

-- Total number of family members
theorem avg_age_grandparents_is_64 (G : ℕ) :
  (num_grandparents * G) + (num_parents * avg_age_parents) + (num_grandchildren * avg_age_grandchildren) = (num_family_members * avg_age_family) →
  G = 64 :=
by
  intro h
  sorry

end avg_age_grandparents_is_64_l62_62712


namespace obrien_hats_theorem_l62_62605

-- Define the number of hats Fire Chief Simpson has.
def simpson_hats : ℕ := 15

-- Define the number of hats Policeman O'Brien had before any hats were stolen.
def obrien_hats_before (simpson_hats : ℕ) : ℕ := 2 * simpson_hats + 5

-- Define the number of hats Policeman O'Brien has now, after x hats were stolen.
def obrien_hats_now (x : ℕ) : ℕ := obrien_hats_before simpson_hats - x

-- Define the theorem stating the problem
theorem obrien_hats_theorem (x : ℕ) : obrien_hats_now x = 35 - x :=
by
  sorry

end obrien_hats_theorem_l62_62605


namespace fraction_value_eq_l62_62541

theorem fraction_value_eq : (5 * 8) / 10 = 4 := 
by 
  sorry

end fraction_value_eq_l62_62541


namespace constant_term_zero_quadratic_l62_62922

theorem constant_term_zero_quadratic (m : ℝ) :
  (-m^2 + 1 = 0) → m = -1 :=
by
  intro h
  sorry

end constant_term_zero_quadratic_l62_62922


namespace sin_double_angle_log_simplification_l62_62096

-- Problem 1: Prove sin(2 * α) = 7 / 25 given sin(α - π / 4) = 3 / 5
theorem sin_double_angle (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 3 / 5) : Real.sin (2 * α) = 7 / 25 :=
by
  sorry

-- Problem 2: Prove 2 * log₅ 10 + log₅ 0.25 = 2
theorem log_simplification : 2 * Real.log 10 / Real.log 5 + Real.log (0.25) / Real.log 5 = 2 :=
by
  sorry

end sin_double_angle_log_simplification_l62_62096


namespace new_average_weight_calculation_l62_62978

noncomputable def new_average_weight (total_weight : ℝ) (number_of_people : ℝ) : ℝ :=
  total_weight / number_of_people

theorem new_average_weight_calculation :
  let initial_people := 6
  let initial_avg_weight := 156
  let new_person_weight := 121
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

end new_average_weight_calculation_l62_62978


namespace mean_proportional_l62_62441

theorem mean_proportional (a b c : ℝ) (ha : a = 1) (hb : b = 2) (h : c ^ 2 = a * b) : c = Real.sqrt 2 :=
by
  sorry

end mean_proportional_l62_62441


namespace kim_morning_routine_time_l62_62796

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end kim_morning_routine_time_l62_62796


namespace no_valid_n_for_conditions_l62_62146

theorem no_valid_n_for_conditions :
  ∀ (n : ℕ), (100 ≤ n / 5 ∧ n / 5 ≤ 999) ∧ (100 ≤ 5 * n ∧ 5 * n ≤ 999) → false :=
by
  sorry

end no_valid_n_for_conditions_l62_62146


namespace equivalent_knicks_l62_62766

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end equivalent_knicks_l62_62766


namespace correct_operation_l62_62366

theorem correct_operation (a : ℝ) (h : a ≠ 0) : a * a⁻¹ = 1 :=
by
  sorry

end correct_operation_l62_62366


namespace selling_price_A_count_purchasing_plans_refund_amount_l62_62868

-- Problem 1
theorem selling_price_A (last_revenue this_revenue last_price this_price cars_sold : ℝ) 
    (last_revenue_eq : last_revenue = 1) (this_revenue_eq : this_revenue = 0.9)
    (diff_eq : last_price = this_price + 1)
    (same_cars : cars_sold ≠ 0) :
    this_price = 9 := by
  sorry

-- Problem 2
theorem count_purchasing_plans (cost_A cost_B total_cars min_cost max_cost : ℝ)
    (cost_A_eq : cost_A = 0.75) (cost_B_eq : cost_B = 0.6)
    (total_cars_eq : total_cars = 15) (min_cost_eq : min_cost = 0.99)
    (max_cost_eq : max_cost = 1.05) :
    ∃ n : ℕ, n = 5 := by
  sorry

-- Problem 3
theorem refund_amount (refund_A refund_B revenue_A revenue_B cost_A cost_B total_profits a : ℝ)
    (revenue_B_eq : revenue_B = 0.8) (cost_A_eq : cost_A = 0.75)
    (cost_B_eq : cost_B = 0.6) (total_profits_eq : total_profits = 30 - 15 * a) :
    a = 0.5 := by
  sorry

end selling_price_A_count_purchasing_plans_refund_amount_l62_62868


namespace kims_morning_routine_total_time_l62_62794

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end kims_morning_routine_total_time_l62_62794


namespace average_minutes_per_day_is_correct_l62_62256
-- Import required library for mathematics

-- Define the conditions
def sixth_grade_minutes := 10
def seventh_grade_minutes := 12
def eighth_grade_minutes := 8
def sixth_grade_ratio := 3
def eighth_grade_ratio := 1/2

-- We use noncomputable since we'll rely on some real number operations that are not trivially computable.
noncomputable def total_minutes_per_week (s : ℝ) : ℝ :=
  sixth_grade_minutes * (sixth_grade_ratio * s) * 2 + 
  seventh_grade_minutes * s * 2 + 
  eighth_grade_minutes * (eighth_grade_ratio * s) * 1

noncomputable def total_students (s : ℝ) : ℝ :=
  sixth_grade_ratio * s + s + eighth_grade_ratio * s

noncomputable def average_minutes_per_day : ℝ :=
  (total_minutes_per_week 1) / (total_students 1 / 5)

theorem average_minutes_per_day_is_correct : average_minutes_per_day = 176 / 9 :=
by
  sorry

end average_minutes_per_day_is_correct_l62_62256


namespace two_rel_prime_exists_l62_62852

theorem two_rel_prime_exists (A : Finset ℕ) (h1 : A.card = 2011) (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 4020) : 
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ Nat.gcd a b = 1 :=
by
  sorry

end two_rel_prime_exists_l62_62852


namespace simplify_expression_l62_62047

variable (y : ℝ)

theorem simplify_expression : 
  3 * y - 5 * y^2 + 2 + (8 - 5 * y + 2 * y^2) = -3 * y^2 - 2 * y + 10 := 
by
  sorry

end simplify_expression_l62_62047


namespace pair_exists_l62_62937

def exists_pair (a b : ℕ → ℕ) : Prop :=
  ∃ p q : ℕ, p < q ∧ a p ≤ a q ∧ b p ≤ b q

theorem pair_exists (a b : ℕ → ℕ) : exists_pair a b :=
sorry

end pair_exists_l62_62937


namespace jack_lap_time_improvement_l62_62484

/-!
Jack practices running in a stadium. Initially, he completed 15 laps in 45 minutes.
After a month of training, he completed 18 laps in 42 minutes. By how many minutes 
has he improved his lap time?
-/

theorem jack_lap_time_improvement:
  ∀ (initial_laps current_laps : ℕ) 
    (initial_time current_time : ℝ), 
    initial_laps = 15 → 
    current_laps = 18 → 
    initial_time = 45 → 
    current_time = 42 → 
    (initial_time / initial_laps - current_time / current_laps = 2/3) :=
by 
  intros _ _ _ _ h_initial_laps h_current_laps h_initial_time h_current_time
  rw [h_initial_laps, h_current_laps, h_initial_time, h_current_time]
  sorry

end jack_lap_time_improvement_l62_62484


namespace sum_of_integers_eq_28_24_23_l62_62069

theorem sum_of_integers_eq_28_24_23 
  (a b : ℕ) 
  (h1 : a * b + a + b = 143)
  (h2 : Nat.gcd a b = 1)
  (h3 : a < 30)
  (h4 : b < 30) 
  : a + b = 28 ∨ a + b = 24 ∨ a + b = 23 :=
by
  sorry

end sum_of_integers_eq_28_24_23_l62_62069


namespace sum_a_b_l62_62411

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem sum_a_b (a b : ℝ) 
  (H : ∀ x, 2 < x ∧ x < 3 → otimes (x - a) (x - b) > 0) : a + b = 4 :=
by
  sorry

end sum_a_b_l62_62411


namespace uniformity_of_scores_l62_62332

/- Problem statement:
  Randomly select 10 students from class A and class B to participate in an English oral test. 
  The variances of their test scores are S1^2 = 13.2 and S2^2 = 26.26, respectively. 
  Then, we show that the scores of the 10 students from class A are more uniform than 
  those of the 10 students from class B.
-/

theorem uniformity_of_scores (S1 S2 : ℝ) (h1 : S1^2 = 13.2) (h2 : S2^2 = 26.26) : 
    13.2 < 26.26 := 
by 
  sorry

end uniformity_of_scores_l62_62332


namespace canoes_more_than_kayaks_l62_62706

theorem canoes_more_than_kayaks (C K : ℕ)
  (h1 : 14 * C + 15 * K = 288)
  (h2 : C = 3 * K / 2) :
  C - K = 4 :=
sorry

end canoes_more_than_kayaks_l62_62706


namespace necessary_but_not_sufficient_condition_l62_62940

theorem necessary_but_not_sufficient_condition (a : ℕ → ℝ) (a1_pos : 0 < a 1) (q : ℝ) (geo_seq : ∀ n, a (n+1) = q * a n) : 
  (∀ n : ℕ, a (2*n + 1) + a (2*n + 2) < 0) → q < 0 :=
sorry

end necessary_but_not_sufficient_condition_l62_62940


namespace seventeen_divides_l62_62903

theorem seventeen_divides (a b : ℤ) (h : 17 ∣ (2 * a + 3 * b)) : 17 ∣ (9 * a + 5 * b) :=
sorry

end seventeen_divides_l62_62903


namespace simplify_fraction_l62_62405

theorem simplify_fraction : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end simplify_fraction_l62_62405


namespace sum_of_integers_l62_62824

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 14) (h3 : x * y = 180) :
  x + y = 2 * Int.sqrt 229 :=
sorry

end sum_of_integers_l62_62824


namespace min_bound_of_gcd_condition_l62_62672

theorem min_bound_of_gcd_condition :
  ∃ c > 0, ∀ a b n : ℕ, 0 < a ∧ 0 < b ∧ 0 < n ∧
  (∀ i j : ℕ, i ≤ n ∧ j ≤ n → Nat.gcd (a + i) (b + j) > 1) →
  min a b > (c * n) ^ (n / 2) :=
sorry

end min_bound_of_gcd_condition_l62_62672


namespace josh_points_l62_62475

variable (x y : ℕ)
variable (three_point_success_rate two_point_success_rate : ℚ)
variable (total_shots : ℕ)
variable (points : ℚ)

theorem josh_points (h1 : three_point_success_rate = 0.25)
                    (h2 : two_point_success_rate = 0.40)
                    (h3 : total_shots = 40)
                    (h4 : x + y = total_shots) :
                    points = 32 :=
by sorry

end josh_points_l62_62475


namespace integer_solution_unique_l62_62272

theorem integer_solution_unique (x y z : ℤ) : x^3 - 2*y^3 - 4*z^3 = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by 
  sorry

end integer_solution_unique_l62_62272


namespace trapezoid_larger_base_length_l62_62470

theorem trapezoid_larger_base_length
  (x : ℝ)
  (h_ratio : 3 = 3 * 1)
  (h_midline : (x + 3 * x) / 2 = 24) :
  3 * x = 36 :=
by
  sorry

end trapezoid_larger_base_length_l62_62470


namespace solve_inequality_l62_62675

theorem solve_inequality :
  {x : ℝ | (x - 3)*(x - 4)*(x - 5) / ((x - 2)*(x - 6)*(x - 7)) > 0} =
  {x : ℝ | x < 2} ∪ {x : ℝ | 4 < x ∧ x < 5} ∪ {x : ℝ | 6 < x ∧ x < 7} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end solve_inequality_l62_62675


namespace probability_different_topics_l62_62587

theorem probability_different_topics (topics : ℕ) (choices : Finset ℕ) (A B : choices) 
(h_topic_count : topics = 6)
(h_totals : choices.card = topics) :
  (probability A B choosing_different := (choices.card - 1) * choices.card = 30) → 
  (total_possible_outcomes := choices.card * choices.card = 36) →
  (probability_different := 30 / 36 = 5 / 6) :=
sorry

end probability_different_topics_l62_62587


namespace geometric_locus_points_l62_62623

theorem geometric_locus_points :
  (∀ x y : ℝ, (y^2 = x^2) ↔ (y = x ∨ y = -x)) ∧
  (∀ x : ℝ, (x^2 - 2 * x + 1 = 0) ↔ (x = 1)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 4 * (y - 1)) ↔ (x = 0 ∧ y = 2)) ∧
  (∀ x y : ℝ, (x^2 - 2 * x * y + y^2 = -1) ↔ false) :=
by
  sorry

end geometric_locus_points_l62_62623


namespace same_heads_probability_l62_62189

theorem same_heads_probability
  (fair_coin : Real := 1/2)
  (biased_coin : Real := 5/8)
  (prob_Jackie_eq_Phil : Real := 77/225) :
  let m := 77
  let n := 225
  (m : ℕ) + (n : ℕ) = 302 := 
by {
  -- The proof would involve constructing the generating functions,
  -- calculating the sum of corresponding coefficients and showing that the
  -- resulting probability reduces to 77/225
  sorry
}

end same_heads_probability_l62_62189


namespace bicyclist_speed_first_100_km_l62_62669

theorem bicyclist_speed_first_100_km (v : ℝ) :
  (16 = 400 / ((100 / v) + 20)) →
  v = 20 :=
by
  sorry

end bicyclist_speed_first_100_km_l62_62669


namespace tangent_line_at_2_m_range_for_three_roots_l62_62756

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 3

theorem tangent_line_at_2 :
  ∃ k b, k = 12 ∧ b = -17 ∧ (∀ x, 12 * x - (k * (x - 2) + f 2) = b) :=
by
  sorry

theorem m_range_for_three_roots :
  {m : ℝ | ∃ x₀ x₁ x₂, x₀ < x₁ ∧ x₁ < x₂ ∧ f x₀ + m = 0 ∧ f x₁ + m = 0 ∧ f x₂ + m = 0} = 
  {m : ℝ | -3 < m ∧ m < -2} :=
by
  sorry

end tangent_line_at_2_m_range_for_three_roots_l62_62756


namespace Micheal_completion_time_l62_62503

variable (W M A : ℝ)

-- Conditions
def condition1 (W M A : ℝ) : Prop := M + A = W / 20
def condition2 (W M A : ℝ) : Prop := A = (W - 14 * (M + A)) / 10

-- Goal
theorem Micheal_completion_time :
  (condition1 W M A) →
  (condition2 W M A) →
  M = W / 50 :=
by
  intros h1 h2
  sorry

end Micheal_completion_time_l62_62503


namespace neither_rain_nor_snow_l62_62068

theorem neither_rain_nor_snow 
  (p_rain : ℚ)
  (p_snow : ℚ)
  (independent : Prop) 
  (h_rain : p_rain = 4/10)
  (h_snow : p_snow = 1/5)
  (h_independent : independent)
  : (1 - p_rain) * (1 - p_snow) = 12 / 25 := 
by
  sorry

end neither_rain_nor_snow_l62_62068


namespace no_common_root_of_polynomials_l62_62968

theorem no_common_root_of_polynomials (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) : 
  ∀ x : ℝ, ¬ (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by
  intro x
  sorry

end no_common_root_of_polynomials_l62_62968


namespace no_integers_satisfy_l62_62956

def P (x a b c d : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integers_satisfy :
  ∀ a b c d : ℤ, ¬ (P 19 a b c d = 1 ∧ P 62 a b c d = 2) :=
by
  intro a b c d
  sorry

end no_integers_satisfy_l62_62956


namespace Romeo_bars_of_chocolate_l62_62206

theorem Romeo_bars_of_chocolate 
  (cost_per_bar : ℕ) (packaging_cost : ℕ) (total_sale : ℕ) (profit : ℕ) (x : ℕ) :
  cost_per_bar = 5 →
  packaging_cost = 2 →
  total_sale = 90 →
  profit = 55 →
  (total_sale - (cost_per_bar + packaging_cost) * x = profit) →
  x = 5 :=
by
  sorry

end Romeo_bars_of_chocolate_l62_62206


namespace solve_case1_solve_case2_l62_62676

variables (a b c A B C x y z : ℝ)

-- Define the conditions for the first special case
def conditions_case1 := (A = b + c) ∧ (B = c + a) ∧ (C = a + b)

-- State the proposition to prove for the first special case
theorem solve_case1 (h : conditions_case1 a b c A B C) :
  z = 0 ∧ y = -1 ∧ x = A + b := by
  sorry

-- Define the conditions for the second special case
def conditions_case2 := (A = b * c) ∧ (B = c * a) ∧ (C = a * b)

-- State the proposition to prove for the second special case
theorem solve_case2 (h : conditions_case2 a b c A B C) :
  z = 1 ∧ y = -(a + b + c) ∧ x = a * b * c := by
  sorry

end solve_case1_solve_case2_l62_62676


namespace egg_laying_hens_l62_62506

theorem egg_laying_hens (total_chickens : ℕ) (num_roosters : ℕ) (non_egg_laying_hens : ℕ)
  (h1 : total_chickens = 325)
  (h2 : num_roosters = 28)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - num_roosters - non_egg_laying_hens = 277 :=
by sorry

end egg_laying_hens_l62_62506


namespace number_of_posts_needed_l62_62577

-- Define the conditions
def length_of_field : ℕ := 80
def width_of_field : ℕ := 60
def distance_between_posts : ℕ := 10

-- Statement to prove the number of posts needed to completely fence the field
theorem number_of_posts_needed : 
  (2 * (length_of_field / distance_between_posts + 1) + 
   2 * (width_of_field / distance_between_posts + 1) - 
   4) = 28 := 
by
  -- Skipping the proof for this theorem
  sorry

end number_of_posts_needed_l62_62577


namespace general_formula_a_n_T_n_greater_than_S_n_l62_62280

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l62_62280


namespace a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l62_62747

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l62_62747


namespace inequality_solution_set_range_of_m_l62_62446

-- Proof Problem 1
theorem inequality_solution_set :
  {x : ℝ | -2 < x ∧ x < 4} = { x : ℝ | 2 * x^2 - 4 * x - 16 < 0 } :=
sorry

-- Proof Problem 2
theorem range_of_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x > 2, f x ≥ (m + 2) * x - m - 15) ↔ m ≤ 2 :=
  sorry

end inequality_solution_set_range_of_m_l62_62446


namespace daryl_age_l62_62147

theorem daryl_age (d j : ℕ) 
  (h1 : d - 4 = 3 * (j - 4)) 
  (h2 : d + 5 = 2 * (j + 5)) :
  d = 31 :=
by sorry

end daryl_age_l62_62147


namespace power_division_l62_62997

theorem power_division : 3^18 / (27^3) = 19683 := by
  have h1 : 27 = 3^3 := by sorry
  have h2 : (3^3)^3 = 3^(3*3) := by sorry
  have h3 : 27^3 = 3^9 := by
    rw [h1]
    exact h2
  rw [h3]
  have h4 : 3^18 / 3^9 = 3^(18 - 9) := by sorry
  rw [h4]
  norm_num

end power_division_l62_62997


namespace coefficient_x2_expansion_l62_62341

theorem coefficient_x2_expansion :
  let f := λ x : ℂ, (x - (1 / complex.sqrt x))^8 in
  (∃ c : ℂ, c * x^2 = @ereal.get ℂ ℝ_domain 
  (series.coeff (λ x, complex.coeff (x - 1/complex.sqrt x)^8) 2)) ↔ c = 70 :=
by
  sorry

end coefficient_x2_expansion_l62_62341


namespace find_angle_C_find_perimeter_l62_62187

-- Definitions related to the triangle problem
variables {A B C : ℝ}
variables {a b c : ℝ} -- sides opposite to A, B, C

-- Condition: (2a - b) * cos C = c * cos B
def condition_1 (a b c C B : ℝ) : Prop := (2 * a - b) * Real.cos C = c * Real.cos B

-- Given C in radians (part 1: find angle C)
theorem find_angle_C 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : condition_1 a b c C B) 
  (H1 : 0 < C) (H2 : C < Real.pi) :
  C = Real.pi / 3 := 
sorry

-- More conditions for part 2
variables (area : ℝ) -- given area of triangle
def condition_2 (a b C area : ℝ) : Prop := 0.5 * a * b * Real.sin C = area

-- Given c = 2 and area = sqrt(3) (part 2: find perimeter)
theorem find_perimeter 
  (A B C : ℝ) (a b : ℝ) (c : ℝ) (area : ℝ) 
  (h2 : condition_2 a b C area) 
  (Hc : c = 2) (Harea : area = Real.sqrt 3) :
  a + b + c = 6 := 
sorry

end find_angle_C_find_perimeter_l62_62187


namespace mandy_yoga_time_l62_62188

-- Define the conditions
def ratio_swimming := 1
def ratio_running := 2
def ratio_gym := 3
def ratio_biking := 5
def ratio_yoga := 4

def time_biking := 30

-- Define the Lean 4 statement to prove
theorem mandy_yoga_time : (time_biking / ratio_biking) * ratio_yoga = 24 :=
by
  sorry

end mandy_yoga_time_l62_62188


namespace solve_for_y_l62_62674

theorem solve_for_y (y : ℝ) : (5:ℝ)^(2*y + 3) = (625:ℝ)^y → y = 3/2 :=
by
  intro h
  sorry

end solve_for_y_l62_62674


namespace liquid_level_ratio_l62_62355

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end liquid_level_ratio_l62_62355


namespace checkerboard_pattern_exists_l62_62563

-- Definitions for the given conditions
def is_black_white_board (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n ∧ j < n → (board (i, j) = true ∨ board (i, j) = false)

def boundary_cells_black (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i, (i < n → (board (i, 0) = true ∧ board (i, n-1) = true ∧ 
                  board (0, i) = true ∧ board (n-1, i) = true))

def no_monochromatic_square (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∀ i j, i < n-1 ∧ j < n-1 → ¬(board (i, j) = board (i+1, j) ∧ 
                               board (i, j) = board (i, j+1) ∧ 
                               board (i, j) = board (i+1, j+1))

def exists_checkerboard_2x2 (board : ℕ × ℕ → Prop) (n : ℕ) : Prop :=
  ∃ i j, i < n-1 ∧ j < n-1 ∧ 
         (board (i, j) ≠ board (i+1, j) ∧ board (i, j) ≠ board (i, j+1) ∧ 
          board (i+1, j) ≠ board (i+1, j+1) ∧ board (i, j+1) ≠ board (i+1, j+1))

-- The theorem statement
theorem checkerboard_pattern_exists (board : ℕ × ℕ → Prop) (n : ℕ) 
  (coloring : is_black_white_board board n)
  (boundary_black : boundary_cells_black board n)
  (no_mono_2x2 : no_monochromatic_square board n) : 
  exists_checkerboard_2x2 board n :=
by
  sorry

end checkerboard_pattern_exists_l62_62563


namespace probability_different_topics_l62_62583

theorem probability_different_topics (topics : ℕ) (h : topics = 6) : 
  let total_combinations := topics * topics,
      different_combinations := topics * (topics - 1) 
  in (different_combinations / total_combinations : ℚ) = 5 / 6 :=
by
  -- This is just a place holder proof.
  sorry

end probability_different_topics_l62_62583


namespace sandy_books_from_first_shop_l62_62511

theorem sandy_books_from_first_shop 
  (cost_first_shop : ℕ)
  (books_second_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (total_cost : ℕ)
  (total_books : ℕ)
  (num_books_first_shop : ℕ) :
  cost_first_shop = 1480 →
  books_second_shop = 55 →
  cost_second_shop = 920 →
  average_price = 20 →
  total_cost = cost_first_shop + cost_second_shop →
  total_books = total_cost / average_price →
  num_books_first_shop + books_second_shop = total_books →
  num_books_first_shop = 65 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end sandy_books_from_first_shop_l62_62511


namespace runners_speeds_and_track_length_l62_62532

/-- Given two runners α and β on a circular track starting at point P and running with uniform speeds,
when α reaches the halfway point Q, β is 16 meters behind α. At a later time, their positions are 
symmetric with respect to the diameter PQ. In 1 2/15 seconds, β reaches point Q, and 13 13/15 seconds later, 
α finishes the race. This theorem calculates the speeds of the runners and the distance of the lap. -/
theorem runners_speeds_and_track_length (x y : ℕ)
    (distance : ℝ)
    (runner_speed_alpha runner_speed_beta : ℝ) 
    (half_track_time_alpha half_track_time_beta : ℝ)
    (mirror_time_alpha mirror_time_beta : ℝ)
    (additional_time_beta : ℝ) :
    half_track_time_alpha = 16 ∧ 
    half_track_time_beta = (272/15) ∧ 
    mirror_time_alpha = (17/15) * (272/15 - 16/32) ∧ 
    mirror_time_beta = (17/15) ∧ 
    additional_time_beta = (13 + (13/15))  ∧ 
    runner_speed_beta = (15/2) ∧ 
    runner_speed_alpha = (85/10) ∧ 
    distance = 272 :=
  sorry

end runners_speeds_and_track_length_l62_62532


namespace cube_root_rational_l62_62518

theorem cube_root_rational (a b : ℚ) (r : ℚ) (h1 : ∃ x : ℚ, x^3 = a) (h2 : ∃ y : ℚ, y^3 = b) (h3 : ∃ x y : ℚ, x + y = r ∧ x^3 = a ∧ y^3 = b) :
  (∃ x : ℚ, x^3 = a) ∧ (∃ y : ℚ, y^3 = b) :=
sorry

end cube_root_rational_l62_62518


namespace valve_rate_difference_l62_62552

section ValveRates

-- Conditions
variables (V1 V2 : ℝ) (t1 t2 : ℝ) (C : ℝ)
-- Given Conditions
-- The first valve alone would fill the pool in 2 hours (120 minutes)
def valve1_rate := V1 = 12000 / 120
-- With both valves open, the pool will be filled with water in 48 minutes
def combined_rate := V1 + V2 = 12000 / 48
-- Capacity of the pool is 12000 cubic meters
def pool_capacity := C = 12000

-- The Proof of the question
theorem valve_rate_difference : V1 = 100 → V2 = 150 → (V2 - V1) = 50 :=
by
  intros hV1 hV2
  rw [hV1, hV2]
  norm_num

end ValveRates

end valve_rate_difference_l62_62552


namespace ratio_of_pens_to_pencils_l62_62248

/-
The store ordered pens and pencils:
1. The number of pens was some multiple of the number of pencils plus 300.
2. The cost of a pen was $5.
3. The cost of a pencil was $4.
4. The store ordered 15 boxes, each having 80 pencils.
5. The store paid a total of $18,300 for the stationery.
Prove that the ratio of the number of pens to the number of pencils is 2.25.
-/

variables (e p k : ℕ)
variables (cost_pen : ℕ := 5) (cost_pencil : ℕ := 4) (total_cost : ℕ := 18300)

def number_of_pencils := 15 * 80

def number_of_pens := p -- to be defined in terms of e and k

def total_cost_pens := p * cost_pen
def total_cost_pencils := e * cost_pencil

theorem ratio_of_pens_to_pencils :
  p = k * e + 300 →
  e = 1200 →
  5 * p + 4 * e = 18300 →
  (p : ℚ) / e = 2.25 :=
by
  intros hp he htotal
  sorry

end ratio_of_pens_to_pencils_l62_62248


namespace max_value_of_ex1_ex2_l62_62295

open Real

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then exp x else -(x^3)

-- Define the function g
noncomputable def g (x a : ℝ) : ℝ := 
  f (f x) - a

-- Define the condition that g(x) = 0 has two distinct zeros
def has_two_distinct_zeros (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ g x1 a = 0 ∧ g x2 a = 0

-- Define the target function h
noncomputable def h (m : ℝ) : ℝ := 
  m^3 * exp (-m)

-- Statement of the final proof
theorem max_value_of_ex1_ex2 (a : ℝ) (hpos : 0 < a) (zeros : has_two_distinct_zeros a) :
  (∃ x1 x2 : ℝ, e^x1 * e^x2 = (27 : ℝ) / (exp 3) ∧ g x1 a = 0 ∧ g x2 a = 0) :=
sorry

end max_value_of_ex1_ex2_l62_62295


namespace log_relation_l62_62304

theorem log_relation (a b : ℝ) (log_7 : ℝ → ℝ) (log_6 : ℝ → ℝ) (log_6_343 : log_6 343 = a) (log_7_18 : log_7 18 = b) :
  a = 6 / (b + 2 * log_7 2) :=
by
  sorry

end log_relation_l62_62304


namespace total_revenue_correct_l62_62684

-- Definitions and conditions
def number_of_fair_tickets : ℕ := 60
def price_per_fair_ticket : ℕ := 15
def price_per_baseball_ticket : ℕ := 10
def number_of_baseball_tickets : ℕ := number_of_fair_tickets / 3

-- Calculate revenues
def revenue_from_fair_tickets : ℕ := number_of_fair_tickets * price_per_fair_ticket
def revenue_from_baseball_tickets : ℕ := number_of_baseball_tickets * price_per_baseball_ticket
def total_revenue : ℕ := revenue_from_fair_tickets + revenue_from_baseball_tickets

-- Proof statement
theorem total_revenue_correct : total_revenue = 1100 := by
  sorry

end total_revenue_correct_l62_62684


namespace T_gt_S_for_n_gt_5_l62_62289

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l62_62289


namespace new_supervisor_salary_l62_62231

namespace FactorySalaries

variables (W S2 : ℝ)

def old_supervisor_salary : ℝ := 870
def old_average_salary : ℝ := 430
def new_average_salary : ℝ := 440

theorem new_supervisor_salary :
  (W + old_supervisor_salary) / 9 = old_average_salary →
  (W + S2) / 9 = new_average_salary →
  S2 = 960 :=
by
  intros h1 h2
  -- Proof steps would go here
  sorry

end FactorySalaries

end new_supervisor_salary_l62_62231


namespace integer_solution_xy_eq_yx_l62_62861

theorem integer_solution_xy_eq_yx (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (e : x < y) :
  x ^ y = y ^ x ↔ (x = 2 ∧ y = 4) :=
sorry

end integer_solution_xy_eq_yx_l62_62861


namespace journey_time_approx_24_hours_l62_62502

noncomputable def journey_time_in_hours : ℝ :=
  let t1 := 70 / 60  -- time for destination 1
  let t2 := 50 / 35  -- time for destination 2
  let t3 := 20 / 60 + 20 / 30  -- time for destination 3
  let t4 := 30 / 40 + 60 / 70  -- time for destination 4
  let t5 := 60 / 35  -- time for destination 5
  let return_distance := 70 + 50 + 40 + 90 + 60 + 100  -- total return distance
  let return_time := return_distance / 55  -- time for return journey
  let stay_time := 1 + 3 + 2 + 2.5 + 0.75  -- total stay time
  t1 + t2 + t3 + t4 + t5 + return_time + stay_time  -- total journey time

theorem journey_time_approx_24_hours : abs (journey_time_in_hours - 24) < 1 :=
by
  sorry

end journey_time_approx_24_hours_l62_62502


namespace smallest_value_of_a_l62_62321

theorem smallest_value_of_a (a b c d : ℤ) (h1 : (a - 2 * b) > 0) (h2 : (b - 3 * c) > 0) (h3 : (c - 4 * d) > 0) (h4 : d > 100) : a ≥ 2433 := sorry

end smallest_value_of_a_l62_62321


namespace domain_of_function_l62_62679

theorem domain_of_function:
  {x : ℝ | x + 1 ≥ 0 ∧ 3 - x ≠ 0} = {x : ℝ | x ≥ -1 ∧ x ≠ 3} :=
by
  sorry

end domain_of_function_l62_62679


namespace ratio_of_average_speeds_l62_62862

-- Define the conditions as constants
def distance_ab : ℕ := 510
def distance_ac : ℕ := 300
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4

-- Define the speeds
def speed_eddy := distance_ab / time_eddy
def speed_freddy := distance_ac / time_freddy

-- The ratio calculation and verification function
def speed_ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- Define the main theorem to be proved
theorem ratio_of_average_speeds : speed_ratio speed_eddy speed_freddy = (34, 15) := by
  sorry

end ratio_of_average_speeds_l62_62862


namespace simplify_expression_l62_62514

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : (2 * x ^ 3) ^ 3 = 8 * x ^ 9 := by
  sorry

end simplify_expression_l62_62514


namespace gcd_polynomials_l62_62292

theorem gcd_polynomials (b : ℕ) (hb : 2160 ∣ b) : 
  Nat.gcd (b ^ 2 + 9 * b + 30) (b + 6) = 12 := 
  sorry

end gcd_polynomials_l62_62292


namespace acute_triangle_angles_l62_62974

theorem acute_triangle_angles (α β γ : ℕ) (h1 : α ≥ β) (h2 : β ≥ γ) (h3 : α = 5 * γ) (h4 : α + β + γ = 180) :
  (α = 85 ∧ β = 78 ∧ γ = 17) :=
sorry

end acute_triangle_angles_l62_62974


namespace Kesten_Spitzer_Whitman_theorem_l62_62094

noncomputable def i.i.d_sequence (X : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), n ≠ m → Statistics.IID X n X m

noncomputable def partial_sum (X : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, X i

noncomputable def count_distinct (X : ℕ → ℝ) (n : ℕ) : ℕ :=
(finset.range n).filter (λ i, X i != 0).card

noncomputable def first_return_to_zero (X : ℕ → ℝ) : ℕ :=
inf {n : ℕ | n > 0 ∧ X n = 0}

theorem Kesten_Spitzer_Whitman_theorem 
  (X : ℕ → ℝ) (h_iid : i.i.d_sequence X) 
  (h_distinct : ∀ n, (count_distinct (partial_sum X) n) = (count_distinct X n) ) :
  filter_at_top (λ n, (count_distinct (partial_sum X) n) / n) 
    (tendsto_const_nhds (π (λ n, first_return_to_zero (partial_sum X n)) = ∞))) sorry

end Kesten_Spitzer_Whitman_theorem_l62_62094


namespace correct_calculation_l62_62547

variable (a : ℝ)

theorem correct_calculation : (-2 * a) ^ 3 = -8 * a ^ 3 := by
  sorry

end correct_calculation_l62_62547


namespace sin_15_plus_sin_75_cos_sum_cos_diff_max_cos_product_eq_l62_62510

section
  -- Importing the necessary library 

  -- Definitions for conditions
  variables {α β A B : ℝ}

  -- The sine formulas for the sum and difference of two angles
  axiom sin_add : ∀ {α β : ℝ}, Real.sin (α + β) = Real.sin α * Real.cos β + Real.cos α * Real.sin β
  axiom sin_sub : ∀ {α β : ℝ}, Real.sin (α - β) = Real.sin α * Real.cos β - Real.cos α * Real.sin β
  
  -- The cosine formulas for the sum and difference of two angles
  axiom cos_add : ∀ {α β : ℝ}, Real.cos (α + β) = Real.cos α * Real.cos β - Real.sin α * Real.sin β
  axiom cos_sub : ∀ {α β : ℝ}, Real.cos (α - β) = Real.cos α * Real.cos β + Real.sin α * Real.sin β

  -- Problem 1
  theorem sin_15_plus_sin_75 : Real.sin 15 + Real.sin 75 = Real.sqrt 6 / 2 :=
  by sorry

  -- Problem 2
  theorem cos_sum_cos_diff (A B : ℝ) : Real.cos A + Real.cos B = 2 * Real.cos ((A + B) / 2) * Real.cos ((A - B) / 2) :=
  by sorry

  -- Problem 3
  theorem max_cos_product_eq : 
     (∃ x ∈ Icc 0 (Real.pi / 4), 
        ∀ y ∈ Icc 0 (Real.pi / 4), 
        cos 2 * x * cos (2 * x + Real.pi / 6) <= cos 2 * y * cos (2 * y + Real.pi / 6)) ∧
        cos 2 * x * cos (2 * x + Real.pi / 6) = Real.sqrt 3 / 2 :=
  by sorry

end

end sin_15_plus_sin_75_cos_sum_cos_diff_max_cos_product_eq_l62_62510


namespace problem_statement_l62_62408

theorem problem_statement :
  (81000 ^ 3) / (27000 ^ 3) = 27 :=
by sorry

end problem_statement_l62_62408


namespace tenth_battery_replacement_in_january_l62_62452

theorem tenth_battery_replacement_in_january : ∀ (months_to_replace: ℕ) (start_month: ℕ), 
  months_to_replace = 4 → start_month = 1 → (4 * (10 - 1)) % 12 = 0 → start_month = 1 :=
by
  intros months_to_replace start_month h_replace h_start h_calc
  sorry

end tenth_battery_replacement_in_january_l62_62452


namespace food_drive_total_cans_l62_62576

def total_cans_brought (M J R : ℕ) : ℕ := M + J + R

theorem food_drive_total_cans (M J R : ℕ) 
  (h1 : M = 4 * J) 
  (h2 : J = 2 * R + 5) 
  (h3 : M = 100) : 
  total_cans_brought M J R = 135 :=
by sorry

end food_drive_total_cans_l62_62576


namespace inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l62_62024

theorem inf_div_p_n2n_plus_one (p : ℕ) (hp : Nat.Prime p) (h_odd : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ (n * 2^n + 1) :=
sorry

theorem n_div_3_n2n_plus_one :
  (∃ k : ℕ, ∀ n, n = 6 * k + 1 ∨ n = 6 * k + 2 → 3 ∣ (n * 2^n + 1)) :=
sorry

end inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l62_62024


namespace inequality_solution_l62_62048

theorem inequality_solution (a : ℝ) (h : a > 0) :
  {x : ℝ | ax ^ 2 - (a + 1) * x + 1 < 0} =
    if a = 1 then ∅
    else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1 / a}
    else if a > 1 then {x : ℝ | 1 / a < x ∧ x < 1} 
    else ∅ := sorry

end inequality_solution_l62_62048


namespace a_is_multiple_of_2_l62_62701

theorem a_is_multiple_of_2 (a : ℕ) (h1 : 0 < a) (h2 : (4 ^ a) % 10 = 6) : a % 2 = 0 :=
sorry

end a_is_multiple_of_2_l62_62701


namespace average_increase_l62_62710

variable (A : ℕ) -- The batsman's average before the 17th inning
variable (runs_in_17th_inning : ℕ := 86) -- Runs made in the 17th inning
variable (new_average : ℕ := 38) -- The average after the 17th inning
variable (total_runs_16_innings : ℕ := 16 * A) -- Total runs after 16 innings
variable (total_runs_after_17_innings : ℕ := total_runs_16_innings + runs_in_17th_inning) -- Total runs after 17 innings
variable (total_runs_should_be : ℕ := 17 * new_average) -- Theoretical total runs after 17 innings

theorem average_increase :
  total_runs_after_17_innings = total_runs_should_be → (new_average - A) = 3 :=
by
  sorry

end average_increase_l62_62710


namespace relationship_between_x_x2_and_x3_l62_62453

theorem relationship_between_x_x2_and_x3 (x : ℝ) (h : -1 < x ∧ x < 0) :
  x ^ 3 < x ∧ x < x ^ 2 :=
by
  sorry

end relationship_between_x_x2_and_x3_l62_62453


namespace incorrect_statement_D_l62_62296

noncomputable def hyperbola_C1 := x^2 / 4 - y^2 / 3 = 1
noncomputable def hyperbola_C2 := x^2 / 4 - y^2 / 3 = -1

theorem incorrect_statement_D :
  ∀ (C1 C2 : ℝ → ℝ → Prop),
    (C1 = (λ x y, x^2 / 4 - y^2 / 3 = 1)) →
    (C2 = (λ x y, x^2 / 4 - y^2 / 3 = -1)) →
    ¬ (eccentricity C1 = eccentricity C2) :=
by sorry

end incorrect_statement_D_l62_62296


namespace last_two_digits_of_factorial_sum_l62_62993

theorem last_two_digits_of_factorial_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i) % 100) % 100 = 13 :=
sorry

end last_two_digits_of_factorial_sum_l62_62993


namespace correct_option_l62_62740

-- Define the given conditions
def a : ℕ := 7^5
def b : ℕ := 5^7

-- State the theorem to be proven
theorem correct_option : a^7 * b^5 = 35^35 := by
  -- insert the proof here
  sorry

end correct_option_l62_62740


namespace location_D_meets_condition_l62_62734

-- Definitions
def locA := (average 3 ∧ median 4)
def locB := (average 1 ∧ variance > 0)
def locC := (median 2 ∧ mode 3)
def locD := (average 2 ∧ variance = 3)

-- Condition that needs to be met
def condition_met := ∀ (daily_cases : Finset ℝ), (∀ x ∈ daily_cases, x ≤ 7) ∧ daily_cases.card = 10

-- Locations information
variables {A B C D : Finset ℝ}

-- Stating that location D meets the condition
theorem location_D_meets_condition :
  locD → condition_met D := by
  intros locD
  sorry

end location_D_meets_condition_l62_62734


namespace smallest_N_of_seven_integers_is_twelve_l62_62207

open BigOperators

theorem smallest_N_of_seven_integers_is_twelve {a : Fin 7 → ℕ} 
  (h_distinct : Function.Injective a)
  (h_product : (∏ i, a i) = n^3) : 
  ∃ i, a i = 12 ∧ ∀ j, a j ≤ a i := sorry

end smallest_N_of_seven_integers_is_twelve_l62_62207


namespace percent_defective_units_l62_62933

variable (D : ℝ) -- Let D represent the percent of units produced that are defective

theorem percent_defective_units
  (h1 : 0.05 * D = 0.4) : 
  D = 8 :=
by sorry

end percent_defective_units_l62_62933


namespace cary_mow_weekends_l62_62396

theorem cary_mow_weekends :
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  remaining_amount / earn_per_weekend = 6 :=
by
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  have needed_weekends : remaining_amount / earn_per_weekend = 6 :=
    sorry
  exact needed_weekends

end cary_mow_weekends_l62_62396


namespace janet_initial_crayons_proof_l62_62504

-- Define the initial number of crayons Michelle has
def michelle_initial_crayons : ℕ := 2

-- Define the final number of crayons Michelle will have after receiving Janet's crayons
def michelle_final_crayons : ℕ := 4

-- Define the function that calculates Janet's initial crayons
def janet_initial_crayons (m_i m_f : ℕ) : ℕ := m_f - m_i

-- The Lean statement to prove Janet's initial number of crayons
theorem janet_initial_crayons_proof : janet_initial_crayons michelle_initial_crayons michelle_final_crayons = 2 :=
by
  -- Proof steps go here (we use sorry to skip the proof)
  sorry

end janet_initial_crayons_proof_l62_62504


namespace perp_to_par_perp_l62_62433

variable (m : Line)
variable (α β : Plane)

-- Conditions
axiom parallel_planes (α β : Plane) : Prop
axiom perp (m : Line) (α : Plane) : Prop

-- Statements
axiom parallel_planes_ax : parallel_planes α β
axiom perp_ax : perp m α

-- Goal
theorem perp_to_par_perp {m : Line} {α β : Plane} (h1 : perp m α) (h2 : parallel_planes α β) : perp m β := sorry

end perp_to_par_perp_l62_62433


namespace point_inside_circle_implies_range_l62_62952

theorem point_inside_circle_implies_range (a : ℝ) : 
  (1 - a)^2 + (1 + a)^2 < 4 → -1 < a ∧ a < 1 :=
by
  intro h
  sorry

end point_inside_circle_implies_range_l62_62952


namespace general_formula_T_greater_S_l62_62284

section arithmetic_sequence

variable {a : ℕ → ℤ} -- Define the arithmetic sequence {a_n}
variable {b : ℕ → ℤ} -- Define the sequence {b_n}
variable {S : ℕ → ℤ} -- Define the sum sequence {S_n}
variable {T : ℕ → ℤ} -- Define the sum sequence {T_n}
variable (a1 : ℤ) (d : ℤ) -- Variables for the first term and common difference 

-- Conditions
def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a n = a1 + n * d

def sequence_b (b a : ℕ → ℤ) : Prop :=
  ∀ n, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_sequence_S (S a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (2 * a1 + (n - 1) * d)) / 2

def sum_sequence_T (T b : ℕ → ℤ) : Prop :=
  ∀ n, T n = ∑ i in Finset.range n, b i

def conditions (a b S T : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  arithmetic_sequence a a1 d ∧ sequence_b b a ∧ sum_sequence_S S a ∧ sum_sequence_T T b ∧ S 4 = 32 ∧ T 3 = 16

-- Problem 1: Prove the general formula for {a_n}
theorem general_formula (a : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, a n = 2 * n + 3 :=
sorry

-- Problem 2: Prove that for n > 5, T_n > S_n
theorem T_greater_S (a b S T : ℕ → ℤ) (a1 d : ℤ) (h : conditions a b S T a1 d) :
  ∀ n, n > 5 → T n > S n :=
sorry

end arithmetic_sequence

end general_formula_T_greater_S_l62_62284


namespace artist_painted_total_pictures_l62_62255

theorem artist_painted_total_pictures :
  ∃ x : ℕ, (x / 8 = 9) ∧ (x + 72 = 153) :=
begin
  use 81,
  split,
  {
    show 81 / 8 = 9,
    sorry
  },
  {
    show 81 + 72 = 153,
    sorry
  }
end

end artist_painted_total_pictures_l62_62255


namespace washingMachineCapacity_l62_62301

-- Definitions based on the problem's conditions
def numberOfShirts : ℕ := 2
def numberOfSweaters : ℕ := 33
def numberOfLoads : ℕ := 5

-- Statement we need to prove
theorem washingMachineCapacity : 
  (numberOfShirts + numberOfSweaters) / numberOfLoads = 7 := sorry

end washingMachineCapacity_l62_62301


namespace quadratic_min_n_l62_62926

theorem quadratic_min_n (m n : ℝ) : 
  (∃ x : ℝ, (x^2 + (m - 2023) * x + (n - 1)) = 0) ∧ 
  (m - 2023)^2 - 4 * (n - 1) = 0 → 
  n = 1 := 
sorry

end quadratic_min_n_l62_62926


namespace niu_fraction_property_l62_62768

open Nat

-- Given mn <= 2009, where m, n are positive integers and (n/m) is in lowest terms
-- Prove that for adjacent terms in the sequence, m_k n_{k+1} - m_{k+1} n_k = 1.

noncomputable def is_numerator_denom_pair_in_seq (m n : ℕ) : Bool :=
  m > 0 ∧ n > 0 ∧ m * n ≤ 2009

noncomputable def are_sorted_adjacent_in_seq (m_k n_k m_k1 n_k1 : ℕ) : Bool :=
  m_k * n_k1 - m_k1 * n_k = 1

theorem niu_fraction_property :
  ∀ (m_k n_k m_k1 n_k1 : ℕ),
  is_numerator_denom_pair_in_seq m_k n_k →
  is_numerator_denom_pair_in_seq m_k1 n_k1 →
  m_k < m_k1 →
  are_sorted_adjacent_in_seq m_k n_k m_k1 n_k1
:=
sorry

end niu_fraction_property_l62_62768


namespace find_abc_of_N_l62_62894

theorem find_abc_of_N :
  ∃ N : ℕ, (N % 10000) = (N + 2) % 10000 ∧ 
            (N % 16 = 15 ∧ (N + 2) % 16 = 1) ∧ 
            ∃ abc : ℕ, (100 ≤ abc ∧ abc < 1000) ∧ 
            (N % 1000) = 100 * abc + 99 := sorry

end find_abc_of_N_l62_62894


namespace initial_persons_count_is_eight_l62_62966

noncomputable def number_of_persons_initially 
  (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : ℝ := 
  (new_weight - old_weight) / avg_increase

theorem initial_persons_count_is_eight 
  (avg_increase : ℝ := 2.5) (old_weight : ℝ := 60) (new_weight : ℝ := 80) : 
  number_of_persons_initially avg_increase old_weight new_weight = 8 :=
by
  sorry

end initial_persons_count_is_eight_l62_62966


namespace initial_percentage_of_alcohol_l62_62110

variable (P : ℝ)
variables (x y : ℝ) (initial_percent replacement_percent replaced_quantity final_percent : ℝ)

def whisky_problem :=
  initial_percent = P ∧
  replacement_percent = 0.19 ∧
  replaced_quantity = 2/3 ∧
  final_percent = 0.26 ∧
  (P * (1 - replaced_quantity) + replacement_percent * replaced_quantity = final_percent)

theorem initial_percentage_of_alcohol :
  whisky_problem P 0.40 0.19 (2/3) 0.26 := sorry

end initial_percentage_of_alcohol_l62_62110


namespace train_length_l62_62878

theorem train_length (L : ℝ) (V : ℝ)
  (h1 : V = L / 8)
  (h2 : V = (L + 273) / 20) :
  L = 182 :=
  by
  sorry

end train_length_l62_62878


namespace circle_equation_solution_l62_62709

theorem circle_equation_solution (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 2 * m * y + 2 * m^2 + m - 1 = 0) ↔ m < 1 :=
sorry

end circle_equation_solution_l62_62709


namespace concert_cost_l62_62043

noncomputable def ticket_price : ℝ := 50.0
noncomputable def processing_fee_rate : ℝ := 0.15
noncomputable def parking_fee : ℝ := 10.0
noncomputable def entrance_fee : ℝ := 5.0
def number_of_people : ℕ := 2

noncomputable def processing_fee_per_ticket : ℝ := processing_fee_rate * ticket_price
noncomputable def total_cost_per_ticket : ℝ := ticket_price + processing_fee_per_ticket
noncomputable def total_ticket_cost : ℝ := number_of_people * total_cost_per_ticket
noncomputable def total_cost_with_parking : ℝ := total_ticket_cost + parking_fee
noncomputable def total_entrance_fee : ℝ := number_of_people * entrance_fee
noncomputable def total_cost : ℝ := total_cost_with_parking + total_entrance_fee

theorem concert_cost : total_cost = 135.0 := by
  sorry

end concert_cost_l62_62043


namespace haleys_current_height_l62_62639

-- Define the conditions
def growth_rate : ℕ := 3
def years : ℕ := 10
def future_height : ℕ := 50

-- Define the proof problem
theorem haleys_current_height : (future_height - growth_rate * years) = 20 :=
by {
  -- This is where the actual proof would go
  sorry
}

end haleys_current_height_l62_62639


namespace age_of_youngest_child_l62_62215

theorem age_of_youngest_child
  (x : ℕ)
  (sum_of_ages : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) :
  x = 4 :=
sorry

end age_of_youngest_child_l62_62215


namespace problem_statement_l62_62407

theorem problem_statement :
  (81000 ^ 3) / (27000 ^ 3) = 27 :=
by sorry

end problem_statement_l62_62407


namespace sum_of_consecutive_integers_with_product_1680_l62_62686

theorem sum_of_consecutive_integers_with_product_1680 : 
  ∃ (a b c d : ℤ), (a * b * c * d = 1680 ∧ b = a + 1 ∧ c = a + 2 ∧ d = a + 3) → (a + b + c + d = 26) := sorry

end sum_of_consecutive_integers_with_product_1680_l62_62686


namespace longest_train_length_l62_62079

theorem longest_train_length :
  ∀ (speedA : ℝ) (timeA : ℝ) (speedB : ℝ) (timeB : ℝ) (speedC : ℝ) (timeC : ℝ),
  speedA = 60 * (5 / 18) → timeA = 5 →
  speedB = 80 * (5 / 18) → timeB = 7 →
  speedC = 50 * (5 / 18) → timeC = 9 →
  speedB * timeB > speedA * timeA ∧ speedB * timeB > speedC * timeC ∧ speedB * timeB = 155.54 := by
  sorry

end longest_train_length_l62_62079


namespace line_equation_l62_62459

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4)) 
  (h_intercept_sum : ∃ b c, b + c = 0 ∧ (∀ x y, A.1 * x + A.2 * y = 1 ∨ A.1 * x + A.2 * y = -1)) :
  (∃ m n, m = 4 ∧ n = -1 ∧ (∀ x y, m * x + n * y = 0)) ∨ 
  (∃ p q r, p = 1 ∧ q = -1 ∧ r = 3 ∧ (∀ x y, p * x + q * y + r = 0)) :=
by
  sorry

end line_equation_l62_62459


namespace contrapositive_iff_l62_62342

theorem contrapositive_iff (a b : ℝ) :
  (a^2 - b^2 = 0 → a = b) ↔ (a ≠ b → a^2 - b^2 ≠ 0) :=
by
  sorry

end contrapositive_iff_l62_62342


namespace counts_duel_with_marquises_l62_62776

theorem counts_duel_with_marquises (x y z k : ℕ) (h1 : 3 * x = 2 * y) (h2 : 6 * y = 3 * z)
    (h3 : ∀ c : ℕ, c = x → ∃ m : ℕ, m = k) : k = 6 :=
by
  sorry

end counts_duel_with_marquises_l62_62776


namespace mom_buys_tshirts_l62_62667

theorem mom_buys_tshirts 
  (tshirts_per_package : ℕ := 3) 
  (num_packages : ℕ := 17) :
  tshirts_per_package * num_packages = 51 :=
by
  sorry

end mom_buys_tshirts_l62_62667


namespace wheel_radius_correct_l62_62383
noncomputable def wheel_radius (total_distance : ℝ) (n_revolutions : ℕ) : ℝ :=
  total_distance / (n_revolutions * 2 * Real.pi)

theorem wheel_radius_correct :
  wheel_radius 450.56 320 = 0.224 :=
by
  sorry

end wheel_radius_correct_l62_62383


namespace add_base3_numbers_l62_62253

theorem add_base3_numbers : 
  (2 + 1 * 3) + (0 + 2 * 3 + 1 * 3^2) + 
  (1 + 2 * 3 + 0 * 3^2 + 2 * 3^3) + (2 + 0 * 3 + 1 * 3^2 + 2 * 3^3)
  = 2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 := 
by sorry

end add_base3_numbers_l62_62253


namespace calculate_expression_l62_62129

theorem calculate_expression :
  ((-1 -2 -3 -4 -5 -6 -7 -8 -9 -10) * (1 -2 +3 -4 +5 -6 +7 -8 +9 -10) = 275) :=
by
  sorry

end calculate_expression_l62_62129


namespace evaluate_f_at_13_l62_62501

noncomputable def f : ℝ → ℝ
| x := if 0 < x ∧ x ≤ 9 then real.log x / real.log 3 else f (x - 4)

theorem evaluate_f_at_13 : f 13 = 2 :=
by
  -- Since x > 9, we use the recursive definition: f(x) = f(x - 4)
  unfold f
  -- f(13) = f(9)
  have h1 : f 13 = f 9 := by rw if_neg (λ h, h.1.not_le)
  rw h1
  -- Simplify f(9) using the piecewise definition
  unfold f
  rw if_pos
  { simp [real.log, real.log_div, real.log_one, real.exp_log],
    field_simp [ne_zero_of_mem_Ioo],
    norm_num,
  sorry

end evaluate_f_at_13_l62_62501


namespace Q_lies_in_third_quadrant_l62_62171

theorem Q_lies_in_third_quadrant (b : ℝ) (P_in_fourth_quadrant : 2 > 0 ∧ b < 0) :
    b < 0 ∧ -2 < 0 ↔
    (b < 0 ∧ -2 < 0) :=
by
  sorry

end Q_lies_in_third_quadrant_l62_62171


namespace barrels_left_for_fourth_neighborhood_l62_62879

-- Let's define the conditions:
def tower_capacity : ℕ := 1200
def neighborhood1_usage : ℕ := 150
def neighborhood2_usage : ℕ := 2 * neighborhood1_usage
def neighborhood3_usage : ℕ := neighborhood2_usage + 100

-- Now, let's state the theorem:
theorem barrels_left_for_fourth_neighborhood (total_usage : ℕ) :
  total_usage = neighborhood1_usage + neighborhood2_usage + neighborhood3_usage →
  tower_capacity - total_usage = 350 := by
  intro h
  rw [h, neighborhood1_usage, neighborhood2_usage, neighborhood3_usage]
  simp
  sorry

end barrels_left_for_fourth_neighborhood_l62_62879


namespace youngest_brother_age_l62_62820

theorem youngest_brother_age 
  (x : ℤ) 
  (h1 : ∃ (a b c : ℤ), a = x ∧ b = x + 1 ∧ c = x + 2 ∧ a + b + c = 96) : 
  x = 31 :=
by sorry

end youngest_brother_age_l62_62820


namespace second_valve_rate_difference_l62_62554

theorem second_valve_rate_difference (V1 V2 : ℝ) 
  (h1 : V1 = 12000 / 120)
  (h2 : V1 + V2 = 12000 / 48) :
  V2 - V1 = 50 :=
by
  -- Since h1: V1 = 100
  -- And V1 + V2 = 250 from h2
  -- Therefore V2 = 250 - 100 = 150
  -- And V2 - V1 = 150 - 100 = 50
  sorry

end second_valve_rate_difference_l62_62554


namespace part1_price_light_bulb_motor_part2_minimal_cost_l62_62783

-- Define the conditions
noncomputable def sum_price : ℕ := 12
noncomputable def total_cost_light_bulbs : ℕ := 30
noncomputable def total_cost_motors : ℕ := 45
noncomputable def ratio_light_bulbs_motors : ℕ := 2
noncomputable def total_items : ℕ := 90
noncomputable def max_ratio_light_bulbs_motors : ℕ := 2

-- Statement of the problems
theorem part1_price_light_bulb_motor (x : ℕ) (y : ℕ):
  x + y = sum_price → 
  total_cost_light_bulbs = 30 →
  total_cost_motors = 45 →
  total_cost_light_bulbs / x = ratio_light_bulbs_motors * (total_cost_motors / y) → 
  x = 3 ∧ y = 9 := 
sorry

theorem part2_minimal_cost (m : ℕ) (n : ℕ):
  m + n = total_items →
  m ≤ total_items / max_ratio_light_bulbs_motors →
  let cost := 3 * m + 9 * n in
  (∀ x y, x + y = total_items → x ≤ total_items / max_ratio_light_bulbs_motors → cost ≤ 3 * x + 9 * y) → 
  m = 30 ∧ n = 60 ∧ cost = 630 :=
sorry

end part1_price_light_bulb_motor_part2_minimal_cost_l62_62783


namespace average_leaves_per_hour_l62_62039

theorem average_leaves_per_hour :
  let leaves_first_hour := 7
  let leaves_second_hour := 4
  let leaves_third_hour := 4
  let total_hours := 3
  let total_leaves := leaves_first_hour + leaves_second_hour + leaves_third_hour
  let average_leaves_per_hour := total_leaves / total_hours
  average_leaves_per_hour = 5 := by
  sorry

end average_leaves_per_hour_l62_62039


namespace shirley_sold_10_boxes_l62_62513

variable (cases boxes_per_case : ℕ)

-- Define the conditions
def number_of_cases := 5
def boxes_in_each_case := 2

-- Prove the total number of boxes is 10
theorem shirley_sold_10_boxes (H1 : cases = number_of_cases) (H2 : boxes_per_case = boxes_in_each_case) :
  cases * boxes_per_case = 10 := by
  sorry

end shirley_sold_10_boxes_l62_62513


namespace arithmetic_seq_a3_a9_zero_l62_62436

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_11_zero (a : ℕ → ℝ) : Prop :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 = 0

theorem arithmetic_seq_a3_a9_zero (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_11_zero a) :
  a 3 + a 9 = 0 :=
sorry

end arithmetic_seq_a3_a9_zero_l62_62436


namespace two_distinct_nonzero_complex_numbers_l62_62913

noncomputable def count_distinct_nonzero_complex_numbers_satisfying_conditions : ℕ :=
sorry

theorem two_distinct_nonzero_complex_numbers :
  count_distinct_nonzero_complex_numbers_satisfying_conditions = 2 :=
sorry

end two_distinct_nonzero_complex_numbers_l62_62913


namespace negation_of_p_l62_62687

open Classical

variable {x : ℝ}

def p : Prop := ∃ x : ℝ, x > 1

theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x ≤ 1 :=
by
  sorry

end negation_of_p_l62_62687


namespace price_of_light_bulb_and_motor_l62_62784

theorem price_of_light_bulb_and_motor
  (x : ℝ) (motor_price : ℝ)
  (h1 : x + motor_price = 12)
  (h2 : 10 / x = 2 * 45 / (12 - x)) :
  x = 3 ∧ motor_price = 9 :=
sorry

end price_of_light_bulb_and_motor_l62_62784


namespace Tn_gt_Sn_l62_62288

-- Definitions of the sequences and initial conditions
def a : ℕ → ℕ := λ n, 2 * n + 3
def b (n : ℕ) : ℕ := if n % 2 = 1 then a n - 6 else 2 * a n
def S (n : ℕ) : ℕ := (n * (2 * n + 8))/2
def T : ℕ → ℕ
| 0     := 0
| (n+1) := T n + b (n + 1)

-- Given initial conditions
axiom S_4_eq_32 : S 4 = 32
axiom T_3_eq_16 : T 3 = 16
axiom a_general : ∀ n : ℕ, a n = 2 * n + 3

-- Proof of the main theorem
theorem Tn_gt_Sn (n : ℕ) (h : n > 5) : T n > S n := by
  sorry

end Tn_gt_Sn_l62_62288


namespace min_value_5_l62_62998

theorem min_value_5 (x y : ℝ) : ∃ x y : ℝ, (xy - 2)^2 + (x + y + 1)^2 = 5 :=
sorry

end min_value_5_l62_62998


namespace pencil_length_difference_l62_62244

theorem pencil_length_difference (a b : ℝ) (h1 : a = 1) (h2 : b = 4/9) :
  a - b - b = 1/9 :=
by
  rw [h1, h2]
  sorry

end pencil_length_difference_l62_62244


namespace count_values_of_b_l62_62427

theorem count_values_of_b : 
  ∃! n : ℕ, (n = 4) ∧ (∀ b : ℕ, (b > 0) → (b ≤ 100) → (∃ k : ℤ, 5 * b^2 + 12 * b + 4 = k^2) → 
    (b = 4 ∨ b = 20 ∨ b = 44 ∨ b = 76)) :=
by
  sorry

end count_values_of_b_l62_62427


namespace moles_of_NaHSO4_l62_62423

def react_eq (naoh h2so4 nahso4 h2o : ℕ) : Prop :=
  naoh + h2so4 = nahso4 + h2o

theorem moles_of_NaHSO4
  (naoh h2so4 : ℕ)
  (h : 2 = naoh ∧ 2 = h2so4)
  (react : react_eq naoh h2so4 2 2):
  2 = 2 :=
by
  sorry

end moles_of_NaHSO4_l62_62423


namespace probability_abs_diff_two_l62_62148

open Finset

def S : Finset ℤ := {1, 2, 3, 4, 5}

def num_pairs_with_abs_diff_two : ℤ :=
  (S.product S).filter (λ (x : ℤ × ℤ), x.1 ≠ x.2 ∧ |x.1 - x.2| = 2).card

def total_pairs : ℤ :=
  (S.product S).filter (λ (x : ℤ × ℤ), x.1 ≠ x.2).card

theorem probability_abs_diff_two : 
  (num_pairs_with_abs_diff_two / total_pairs : ℝ) = 3 / 10 :=
by 
  -- Insert steps for the proof here or transformation from ℤ to ℝ w.r.t cardinality.
  sorry

end probability_abs_diff_two_l62_62148


namespace equal_probability_of_selection_l62_62573

-- Let n = 2012 be the initial number of students.
-- Let k = 12 be the number of students eliminated by simple random sampling.
-- Let m = 2000 be the number of remaining students after elimination.
-- Let s = 50 be the number of students selected.
-- We need to prove that the probability of each student being selected is equal.

theorem equal_probability_of_selection (n k s m : ℕ)
  (h_n: n = 2012)
  (h_k: k = 12)
  (h_m: m = n - k)
  (h_s: s = 50)
  (simple_random_sampling : ∀ i, 1 ≤ i ∧ i ≤ n → Nat.inhabited (Fin k)) -- assume simple random sampling
  (systematic_sampling : ∀ r : ℕ, Nat.inhabited (Fin s) → ((r + 1) * s <= m)) -- assume systematic sampling
  : ∀ (i j : ℕ), 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 
  (Mathlib.Prob.event_eq (simple_random_sampling j) (simple_random_sampling i)) ∧
  (Mathlib.Prob.event_eq (systematic_sampling j) (systematic_sampling i)) :=
sorry

end equal_probability_of_selection_l62_62573


namespace michael_savings_l62_62238

theorem michael_savings :
  let price := 45
  let tax_rate := 0.08
  let promo_A_dis := 0.40
  let promo_B_dis := 15
  let before_tax_A := price + price * (1 - promo_A_dis)
  let before_tax_B := price + (price - promo_B_dis)
  let after_tax_A := before_tax_A * (1 + tax_rate)
  let after_tax_B := before_tax_B * (1 + tax_rate)
  after_tax_B - after_tax_A = 3.24 :=
by
  sorry

end michael_savings_l62_62238


namespace simplify_fraction_l62_62404

theorem simplify_fraction : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end simplify_fraction_l62_62404


namespace vicky_download_time_l62_62986

noncomputable def download_time_in_hours (speed_mb_per_sec : ℕ) (program_size_gb : ℕ) (mb_per_gb : ℕ) (seconds_per_hour : ℕ) : ℕ :=
  let program_size_mb := program_size_gb * mb_per_gb
  let time_seconds := program_size_mb / speed_mb_per_sec
  time_seconds / seconds_per_hour

theorem vicky_download_time :
  download_time_in_hours 50 360 1000 3600 = 2 :=
by
  unfold download_time_in_hours
  have h1 : 360 * 1000 = 360000 := by norm_num
  rw [h1]
  have h2 : 360000 / 50 = 7200 := by norm_num
  rw [h2]
  have h3 : 7200 / 3600 = 2 := by norm_num
  rw [h3]
  exact rfl

end vicky_download_time_l62_62986


namespace simplify_tan_pi_over_24_add_tan_7pi_over_24_l62_62046

theorem simplify_tan_pi_over_24_add_tan_7pi_over_24 :
  let a := Real.tan (Real.pi / 24)
  let b := Real.tan (7 * Real.pi / 24)
  a + b = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  -- conditions and definitions:
  let tan_eq_sin_div_cos := ∀ x, Real.tan x = Real.sin x / Real.cos x
  let sin_add := ∀ a b, Real.sin (a + b) = Real.sin a * Real.cos b + Real.cos a * Real.sin b
  let cos_mul := ∀ a b, Real.cos a * Real.cos b = 1 / 2 * (Real.cos (a + b) + Real.cos (a - b))
  let sin_pi_over_3 := Real.sin (Real.pi / 3) = Real.sqrt 3 / 2
  let cos_pi_over_3 := Real.cos (Real.pi / 3) = 1 / 2
  let cos_pi_over_4 := Real.cos (Real.pi / 4) = Real.sqrt 2 / 2
  have cond1 := tan_eq_sin_div_cos
  have cond2 := sin_add
  have cond3 := cos_mul
  have cond4 := sin_pi_over_3
  have cond5 := cos_pi_over_3
  have cond6 := cos_pi_over_4
  sorry

end simplify_tan_pi_over_24_add_tan_7pi_over_24_l62_62046


namespace geom_seq_a3_value_l62_62635

theorem geom_seq_a3_value (a_n : ℕ → ℝ) (h1 : ∃ r : ℝ, ∀ n : ℕ, a_n (n+1) = a_n (1) * r^n) 
                          (h2 : a_n (2) * a_n (4) = 2 * a_n (3) - 1) :
  a_n (3) = 1 :=
sorry

end geom_seq_a3_value_l62_62635


namespace combined_sum_correct_l62_62130

-- Define the sum of integers in a range
def sum_of_integers (a b : Int) : Int := (b - a + 1) * (a + b) / 2

-- Define the sum of squares of integers in a range
def sum_of_squares (a b : Int) : Int :=
  let sum_sq (n : Int) : Int := n * (n + 1) * (2 * n + 1) / 6
  sum_sq b - sum_sq (a - 1)

-- Define the combined sum function
def combined_sum (a b c d : Int) : Int :=
  sum_of_integers a b + sum_of_squares c d

-- Theorem statement: Prove the combined sum of integers from -50 to 40 and squares of integers from 10 to 40 is 21220
theorem combined_sum_correct :
  combined_sum (-50) 40 10 40 = 21220 :=
by
  -- leaving the proof as a sorry
  sorry

end combined_sum_correct_l62_62130


namespace bag_with_cracks_number_l62_62208

def marbles : List ℕ := [18, 19, 21, 23, 25, 34]

def total_marbles : ℕ := marbles.sum

def modulo_3 (n : ℕ) : ℕ := n % 3

theorem bag_with_cracks_number :
  ∃ (c : ℕ), c ∈ marbles ∧ 
    (total_marbles - c) % 3 = 0 ∧
    c = 23 :=
by 
  sorry

end bag_with_cracks_number_l62_62208


namespace roots_quadratic_l62_62941

theorem roots_quadratic (d e : ℝ) (h1 : 3 * d ^ 2 + 5 * d - 2 = 0) (h2 : 3 * e ^ 2 + 5 * e - 2 = 0) :
  (d - 1) * (e - 1) = 2 :=
sorry

end roots_quadratic_l62_62941


namespace min_value_objective_function_l62_62633

theorem min_value_objective_function :
  (∃ x y : ℝ, x ≥ 1 ∧ x + y ≤ 3 ∧ x - 2 * y - 3 ≤ 0 ∧ (∀ x' y', (x' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - 2 * y' - 3 ≤ 0) → 2 * x' + y' ≥ 2 * x + y)) →
  2 * x + y = 1 :=
by
  sorry

end min_value_objective_function_l62_62633


namespace solve_for_x_l62_62303

theorem solve_for_x (x : ℝ) (h : sqrt (3 / x + 3) = 5 / 3) : x = -27 / 2 :=
sorry

end solve_for_x_l62_62303


namespace amount_spent_on_machinery_l62_62936

-- Define the given conditions
def raw_materials_spent : ℤ := 80000
def total_amount : ℤ := 137500
def cash_spent : ℤ := (20 * total_amount) / 100

-- The goal is to prove the amount spent on machinery
theorem amount_spent_on_machinery : 
  ∃ M : ℤ, raw_materials_spent + M + cash_spent = total_amount ∧ M = 30000 := by
  sorry

end amount_spent_on_machinery_l62_62936


namespace neg_a_pow4_div_neg_a_eq_neg_a_pow3_l62_62393

variable (a : ℝ)

theorem neg_a_pow4_div_neg_a_eq_neg_a_pow3 : (-a)^4 / (-a) = -a^3 := sorry

end neg_a_pow4_div_neg_a_eq_neg_a_pow3_l62_62393


namespace sweets_ratio_l62_62078

theorem sweets_ratio (x : ℕ) (h1 : x + 4 + 7 = 22) : x / 22 = 1 / 2 :=
by
  sorry

end sweets_ratio_l62_62078


namespace KayleeAgeCorrect_l62_62769

-- Define Kaylee's current age
def KayleeCurrentAge (k : ℕ) : Prop :=
  (3 * 5 + (7 - k) = 7)

-- State the theorem
theorem KayleeAgeCorrect : ∃ k : ℕ, KayleeCurrentAge k ∧ k = 8 := 
sorry

end KayleeAgeCorrect_l62_62769


namespace cost_of_paving_floor_l62_62213

-- Define the constants given in the problem
def length1 : ℝ := 5.5
def width1 : ℝ := 3.75
def length2 : ℝ := 4
def width2 : ℝ := 3
def cost_per_sq_meter : ℝ := 800

-- Define the areas of the two rectangles
def area1 : ℝ := length1 * width1
def area2 : ℝ := length2 * width2

-- Define the total area of the floor
def total_area : ℝ := area1 + area2

-- Define the total cost of paving the floor
def total_cost : ℝ := total_area * cost_per_sq_meter

-- The statement to prove: the total cost equals 26100 Rs
theorem cost_of_paving_floor : total_cost = 26100 := by
  -- Proof skipped
  sorry

end cost_of_paving_floor_l62_62213


namespace correct_number_of_selection_formulas_l62_62428

theorem correct_number_of_selection_formulas :
  let males := 20
  let females := 30
  let total_students := 50
  let select_4 := nat.choose total_students 4
  let all_males := nat.choose males 4
  let all_females := nat.choose females 4
  let one_male_three_females := (nat.choose males 1) * (nat.choose females 3)
  let two_males_two_females := (nat.choose males 2) * (nat.choose females 2)
  let three_males_one_female := (nat.choose males 3) * (nat.choose females 1)
  let formula1 := select_4 - all_males - all_females
  let formula2 := one_male_three_females + two_males_two_females + three_males_one_female
  let formula3 := (nat.choose males 1) * (nat.choose females 1) * (nat.choose 48 2)
  (if formula1 == select_4 - all_males - all_females then 1 else 0) +
  (if formula2 == one_male_three_females + two_males_two_females + three_males_one_female then 1 else 0) +
  (if formula3 == one_male_three_females + two_males_two_females + three_males_one_female then 1 else 0) = 2 := 
sorry

end correct_number_of_selection_formulas_l62_62428


namespace find_y_given_conditions_l62_62640

theorem find_y_given_conditions (x y : ℝ) (h₁ : 3 * x^2 = y - 6) (h₂ : x = 4) : y = 54 :=
  sorry

end find_y_given_conditions_l62_62640


namespace correct_statements_l62_62224

theorem correct_statements : 
  (∃ n : ℕ, 24 = 4 * n) ∧ (∃ n : ℕ, 180 = 9 * n) :=
by
  sorry

end correct_statements_l62_62224


namespace remainder_55_pow_55_plus_10_mod_8_l62_62212

theorem remainder_55_pow_55_plus_10_mod_8 : (55 ^ 55 + 10) % 8 = 1 :=
by
  sorry

end remainder_55_pow_55_plus_10_mod_8_l62_62212


namespace geometric_sum_l62_62932

theorem geometric_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
    (h1 : S 3 = 8)
    (h2 : S 6 = 7)
    (h3 : ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)) :
  a 7 + a 8 + a 9 = 1 / 8 :=
by
  sorry

end geometric_sum_l62_62932


namespace framed_painting_ratio_l62_62243

theorem framed_painting_ratio
  (width_painting : ℕ)
  (height_painting : ℕ)
  (frame_side : ℕ)
  (frame_top_bottom : ℕ)
  (h1 : width_painting = 20)
  (h2 : height_painting = 30)
  (h3 : frame_top_bottom = 3 * frame_side)
  (h4 : (width_painting + 2 * frame_side) * (height_painting + 2 * frame_top_bottom) = 2 * width_painting * height_painting):
  (width_painting + 2 * frame_side) = 1/2 * (height_painting + 2 * frame_top_bottom) := 
by
  sorry

end framed_painting_ratio_l62_62243


namespace general_formula_a_n_T_n_greater_S_n_l62_62281

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l62_62281


namespace problem_Z_value_l62_62641

def Z (a b : ℕ) : ℕ := 3 * (a - b) ^ 2

theorem problem_Z_value : Z 5 3 = 12 := by
  sorry

end problem_Z_value_l62_62641


namespace min_moves_seven_chests_l62_62673

/-
Problem:
Seven chests are placed in a circle, each containing a certain number of coins: [20, 15, 5, 6, 10, 17, 18].
Prove that the minimum number of moves required to equalize the number of coins in all chests is 22.
-/

def min_moves_to_equalize_coins (coins: List ℕ) : ℕ :=
  -- Function that would calculate the minimum number of moves
  sorry

theorem min_moves_seven_chests :
  min_moves_to_equalize_coins [20, 15, 5, 6, 10, 17, 18] = 22 :=
sorry

end min_moves_seven_chests_l62_62673


namespace villagers_count_l62_62006

theorem villagers_count (V : ℕ) (milk_per_villager apples_per_villager bread_per_villager : ℕ) :
  156 = V * milk_per_villager ∧
  195 = V * apples_per_villager ∧
  234 = V * bread_per_villager →
  V = Nat.gcd (Nat.gcd 156 195) 234 :=
by
  sorry

end villagers_count_l62_62006


namespace min_avg_less_than_old_record_l62_62149

variable old_record_avg : ℕ := 287
variable num_players : ℕ := 4
variable num_rounds : ℕ := 10
variable points_scored_9_rounds : ℕ := 10440

theorem min_avg_less_than_old_record:
  let total_points_needed := old_record_avg * num_players * num_rounds in
  let points_needed_final_round := total_points_needed - points_scored_9_rounds in
  let min_avg_final_round := points_needed_final_round / num_players in
  min_avg_final_round = old_record_avg - 27 :=
by
  sorry

end min_avg_less_than_old_record_l62_62149


namespace infinite_n_dividing_sum_p1_l62_62655

theorem infinite_n_dividing_sum_p1 (p : ℕ) [Fact p.Prime] :
  ∃∞ (n : ℕ), p ∣ (∑ i in Finset.range (p + 1) | i) := 
sorry

end infinite_n_dividing_sum_p1_l62_62655


namespace solve_for_m_l62_62542

theorem solve_for_m (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 :=
by {
  sorry
}

end solve_for_m_l62_62542


namespace max_area_rectangle_l62_62064

theorem max_area_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : x * y ≤ 100 :=
by
  sorry

end max_area_rectangle_l62_62064


namespace sin_13pi_over_4_l62_62419

theorem sin_13pi_over_4 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_13pi_over_4_l62_62419


namespace suff_not_necessary_condition_l62_62629

noncomputable def p : ℝ := 1
noncomputable def q (x : ℝ) : Prop := x^3 - 2 * x + 1 = 0

theorem suff_not_necessary_condition :
  (∀ x, x = p → q x) ∧ (∃ x, q x ∧ x ≠ p) :=
by
  sorry

end suff_not_necessary_condition_l62_62629


namespace sum_of_numbers_l62_62220

theorem sum_of_numbers : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 :=
by
  sorry

end sum_of_numbers_l62_62220


namespace sequence_eighth_term_is_sixteen_l62_62424

-- Define the sequence based on given patterns
def oddPositionTerm (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

def evenPositionTerm (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

-- Formalize the proof problem
theorem sequence_eighth_term_is_sixteen : evenPositionTerm 4 = 16 :=
by 
  unfold evenPositionTerm
  sorry

end sequence_eighth_term_is_sixteen_l62_62424


namespace polynomial_factorization_l62_62950

theorem polynomial_factorization : ∃ q : Polynomial ℝ, (Polynomial.X ^ 4 - 6 * Polynomial.X ^ 2 + 25) = (Polynomial.X ^ 2 + 5) * q :=
by
  sorry

end polynomial_factorization_l62_62950


namespace olympic_medals_l62_62526

theorem olympic_medals (total_sprinters british_sprinters non_british_sprinters ways_case1 ways_case2 ways_case3 : ℕ)
  (h_total : total_sprinters = 10)
  (h_british : british_sprinters = 4)
  (h_non_british : non_british_sprinters = 6)
  (h_case1 : ways_case1 = 6 * 5 * 4)
  (h_case2 : ways_case2 = 4 * 3 * (6 * 5))
  (h_case3 : ways_case3 = (4 * 3) * (3 * 2) * 6) :
  ways_case1 + ways_case2 + ways_case3 = 912 := by
  sorry

end olympic_medals_l62_62526


namespace ratio_of_daily_wages_l62_62343

-- Definitions for daily wages and conditions
def daily_wage_man : ℝ := sorry
def daily_wage_woman : ℝ := sorry

axiom condition_for_men (M : ℝ) : 16 * M * 25 = 14400
axiom condition_for_women (W : ℝ) : 40 * W * 30 = 21600

-- Theorem statement for the ratio of daily wages
theorem ratio_of_daily_wages 
  (M : ℝ) (W : ℝ) 
  (hM : 16 * M * 25 = 14400) 
  (hW : 40 * W * 30 = 21600) :
  M / W = 2 := 
  sorry

end ratio_of_daily_wages_l62_62343


namespace greatest_integer_2e_minus_5_l62_62194

noncomputable def e : ℝ := 2.718

theorem greatest_integer_2e_minus_5 : ⌊2 * e - 5⌋ = 0 :=
by
  -- This is a placeholder for the actual proof. 
  sorry

end greatest_integer_2e_minus_5_l62_62194


namespace system_eq_solution_l62_62471

theorem system_eq_solution (x y c d : ℝ) (hd : d ≠ 0) 
  (h1 : 4 * x - 2 * y = c) 
  (h2 : 6 * y - 12 * x = d) :
  c / d = -1 / 3 := 
by 
  sorry

end system_eq_solution_l62_62471


namespace no_preimage_iff_k_less_than_neg2_l62_62434

theorem no_preimage_iff_k_less_than_neg2 (k : ℝ) :
  ¬∃ x : ℝ, x^2 - 2 * x - 1 = k ↔ k < -2 :=
sorry

end no_preimage_iff_k_less_than_neg2_l62_62434


namespace find_F_l62_62763

theorem find_F (C : ℝ) (F : ℝ) (h₁ : C = 35) (h₂ : C = 4 / 7 * (F - 40)) : F = 101.25 := by
  sorry

end find_F_l62_62763


namespace Tim_age_l62_62352

theorem Tim_age (T t : ℕ) (h1 : T = 22) (h2 : T = 2 * t + 6) : t = 8 := by
  sorry

end Tim_age_l62_62352


namespace point_not_in_second_quadrant_l62_62923

theorem point_not_in_second_quadrant (a : ℝ) :
  (∃ b : ℝ, b = 2 * a - 1) ∧ ¬(a < 0 ∧ (2 * a - 1 > 0)) := 
by sorry

end point_not_in_second_quadrant_l62_62923


namespace total_balloons_correct_l62_62021

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end total_balloons_correct_l62_62021


namespace big_container_capacity_l62_62560

theorem big_container_capacity (C : ℝ)
    (h1 : 0.75 * C - 0.40 * C = 14) : C = 40 :=
  sorry

end big_container_capacity_l62_62560


namespace right_triangle_area_l62_62697

def hypotenuse := 17
def leg1 := 15
def leg2 := 8
def area := (1 / 2:ℝ) * leg1 * leg2 

theorem right_triangle_area (a b c : ℝ) (h : a^2 + b^2 = c^2) (hl1 : a = 15) (hl2 : c = 17) :
  area = 60 := by 
  sorry

end right_triangle_area_l62_62697


namespace students_in_second_class_l62_62846

-- Definitions based on the conditions
def students_first_class : ℕ := 30
def avg_mark_first_class : ℕ := 40
def avg_mark_second_class : ℕ := 80
def combined_avg_mark : ℕ := 65

-- Proposition to prove
theorem students_in_second_class (x : ℕ) 
  (h1 : students_first_class * avg_mark_first_class + x * avg_mark_second_class = (students_first_class + x) * combined_avg_mark) : 
  x = 50 :=
sorry

end students_in_second_class_l62_62846


namespace find_F_l62_62915

theorem find_F (F C : ℝ) (hC_eq : C = (4/7) * (F - 40)) (hC_val : C = 35) : F = 101.25 :=
by
  sorry

end find_F_l62_62915


namespace minimize_cost_l62_62251

noncomputable def avg_comprehensive_cost (x : ℝ) : ℝ :=
  (560 + 48 * x) + 10800 / x

theorem minimize_cost (x : ℝ) (hx : x ≥ 10) (hx_int : ∃ n : ℕ, x = n) :
  argmin avg_comprehensive_cost (≥ 10) = 15 :=
by
  sorry

end minimize_cost_l62_62251


namespace binary_operation_result_l62_62260

theorem binary_operation_result :
  let a := 0b1101
  let b := 0b111
  let c := 0b1010
  let d := 0b1001
  a + b - c + d = 0b10011 :=
by {
  sorry
}

end binary_operation_result_l62_62260


namespace endangered_animal_population_after_3_years_l62_62975

-- Given conditions and definitions
def population (m : ℕ) (n : ℕ) : ℝ := m * (0.90 ^ n)

theorem endangered_animal_population_after_3_years :
  population 8000 3 = 5832 :=
by
  sorry

end endangered_animal_population_after_3_years_l62_62975


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l62_62991

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l62_62991


namespace sarah_interview_combinations_l62_62883

theorem sarah_interview_combinations : 
  (1 * 2 * (2 + 3) * 5 * 1) = 50 := 
by
  sorry

end sarah_interview_combinations_l62_62883


namespace value_of_a_l62_62692

theorem value_of_a (k : ℝ) (a : ℝ) (b : ℝ) (h1 : a = k / b^2) (h2 : a = 10) (h3 : b = 24) :
  a = 40 :=
sorry

end value_of_a_l62_62692


namespace pos_real_x_plus_inv_ge_two_l62_62498

theorem pos_real_x_plus_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end pos_real_x_plus_inv_ge_two_l62_62498


namespace track_width_l62_62875

variable (r1 r2 r3 : ℝ)

def cond1 : Prop := 2 * Real.pi * r2 - 2 * Real.pi * r1 = 20 * Real.pi
def cond2 : Prop := 2 * Real.pi * r3 - 2 * Real.pi * r2 = 30 * Real.pi

theorem track_width (h1 : cond1 r1 r2) (h2 : cond2 r2 r3) : r3 - r1 = 25 := by
  sorry

end track_width_l62_62875


namespace length_BE_l62_62481

-- Definitions and Conditions
def is_square (ABCD : Type) (side_length : ℝ) : Prop :=
  side_length = 2

def triangle_area (base : ℝ) (height : ℝ) : ℝ :=
  0.5 * base * height

def rectangle_area (length : ℝ) (width : ℝ) : ℝ :=
  length * width

-- Problem statement in Lean
theorem length_BE 
(ABCD : Type) (side_length : ℝ) 
(JKHG : Type) (BC : ℝ) (x : ℝ) 
(E : Type) (E_on_BC : E) 
(area_fact : rectangle_area BC x = 2 * triangle_area x BC) 
(h1 : is_square ABCD side_length) 
(h2 : BC = 2) : 
x = 1 :=
by {
  sorry
}

end length_BE_l62_62481


namespace last_two_digits_factorials_sum_l62_62990

theorem last_two_digits_factorials_sum : 
  (∑ i in Finset.range 16, (Nat.factorial i % 100)) % 100 = 13 := 
by
  sorry

end last_two_digits_factorials_sum_l62_62990


namespace price_per_butterfly_l62_62490

theorem price_per_butterfly (jars : ℕ) (caterpillars_per_jar : ℕ) (fail_percentage : ℝ) (total_money : ℝ) (price : ℝ) :
  jars = 4 →
  caterpillars_per_jar = 10 →
  fail_percentage = 0.40 →
  total_money = 72 →
  price = 3 :=
by
  intros h_jars h_caterpillars h_fail_percentage h_total_money
  -- Full proof here
  sorry

end price_per_butterfly_l62_62490


namespace part_a_part_b_first_question_part_b_second_question_part_c_l62_62702

open EuclideanGeometry

-- Definition and theorems for problem part (a)
theorem part_a (A B C D X Y Z Y' : Point) (h_square : square A B C D 1)
  (hX : X ∈ segment A B) (hY : Y ∈ segment B C) (hZ : Z ∈ segment C D)
  (hY' : Y' ∈ segment C D):
  area_triangle X Y Z = area_triangle X Y' Z :=
sorry

-- Definition and theorems for problem part (b)
theorem part_b_first_question (A B C D X Y' Z : Point) (h_square : square A B C D 1)
  (hX : X ∈ segment A B) (hY' : Y' ∈ segment C D) (hZ : Z ∈ segment C D):
  max_area_triangle_one_side AB CD = 1 / 2 :=
sorry

theorem part_b_second_question (A B C D : Point) (h_square : square A B C D 1):
  max_area_triangle_inside_square (A, B, C, D) = 1 / 2 :=
sorry

-- Definition and theorems for problem part (c)
theorem part_c (pts : Fin 9 → Point) (h_square : square (pts 0) (pts 1) (pts 2) (pts 3) 2)
  (h_non_collinear : ∀ i j k, ¬ collinear i j k) :
  ∃ pt1 pt2 pt3, pt1 ≠ pt2 ∧ pt2 ≠ pt3 ∧ pt1 ≠ pt3 ∧ area_triangle pt1 pt2 pt3 ≤ 1 / 2 :=
sorry

end part_a_part_b_first_question_part_b_second_question_part_c_l62_62702


namespace time_to_eat_cereal_l62_62948

noncomputable def MrFatRate : ℝ := 1 / 40
noncomputable def MrThinRate : ℝ := 1 / 15
noncomputable def CombinedRate : ℝ := MrFatRate + MrThinRate
noncomputable def CerealAmount : ℝ := 4
noncomputable def TimeToFinish : ℝ := CerealAmount / CombinedRate
noncomputable def expected_time : ℝ := 96

theorem time_to_eat_cereal :
  TimeToFinish = expected_time :=
by
  sorry

end time_to_eat_cereal_l62_62948


namespace div_five_times_eight_by_ten_l62_62539

theorem div_five_times_eight_by_ten : (5 * 8) / 10 = 4 := by
  sorry

end div_five_times_eight_by_ten_l62_62539


namespace blue_segments_count_l62_62648

def grid_size : ℕ := 16
def total_dots : ℕ := grid_size * grid_size
def red_dots : ℕ := 133
def boundary_red_dots : ℕ := 32
def corner_red_dots : ℕ := 2
def yellow_segments : ℕ := 196

-- Dummy hypotheses representing the given conditions
axiom red_dots_on_grid : red_dots <= total_dots
axiom boundary_red_dots_count : boundary_red_dots = 32
axiom corner_red_dots_count : corner_red_dots = 2
axiom total_yellow_segments : yellow_segments = 196

-- Proving the number of blue line segments
theorem blue_segments_count :  ∃ (blue_segments : ℕ), blue_segments = 134 := 
sorry

end blue_segments_count_l62_62648


namespace ratio_of_x_to_y_l62_62854

theorem ratio_of_x_to_y (x y : ℚ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 2 / 7) : 
  x / y = 29 / 34 :=
sorry

end ratio_of_x_to_y_l62_62854


namespace most_appropriate_survey_is_D_l62_62085

-- Define the various scenarios as Lean definitions
def survey_A := "Testing whether a certain brand of fresh milk meets food hygiene standards, using a census method."
def survey_B := "Security check before taking the subway, using a sampling survey method."
def survey_C := "Understanding the sleep time of middle school students in Jiangsu Province, using a census method."
def survey_D := "Understanding the way Nanjing residents commemorate the Qingming Festival, using a sampling survey method."

-- Define the type for specifying which survey method is the most appropriate
def appropriate_survey (survey : String) : Prop := 
  survey = survey_D

-- The theorem statement proving that the most appropriate survey is D
theorem most_appropriate_survey_is_D : appropriate_survey survey_D :=
by sorry

end most_appropriate_survey_is_D_l62_62085


namespace solve_system_l62_62049

def eq1 (x y : ℝ) : Prop := x^2 * y - x * y^2 - 5 * x + 5 * y + 3 = 0
def eq2 (x y : ℝ) : Prop := x^3 * y - x * y^3 - 5 * x^2 + 5 * y^2 + 15 = 0

theorem solve_system :
  ∃ (x y : ℝ), eq1 x y ∧ eq2 x y ∧ x = 4 ∧ y = 1 := by
  sorry

end solve_system_l62_62049


namespace zero_in_interval_l62_62910

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x + 2 * x^2 - 4 * x

theorem zero_in_interval : ∃ (c : ℝ), 1 < c ∧ c < Real.exp 1 ∧ f c = 0 := sorry

end zero_in_interval_l62_62910


namespace complement_A_A_inter_complement_B_l62_62372

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem complement_A : compl A = {x | x ≤ 1 ∨ 4 ≤ x} :=
by sorry

theorem A_inter_complement_B : A ∩ compl B = {x | 3 < x ∧ x < 4} :=
by sorry

end complement_A_A_inter_complement_B_l62_62372


namespace scientific_notation_of_0_0000023_l62_62265

theorem scientific_notation_of_0_0000023 : 
  0.0000023 = 2.3 * 10 ^ (-6) :=
by
  sorry

end scientific_notation_of_0_0000023_l62_62265


namespace arithmetic_sequence_9th_term_l62_62929

theorem arithmetic_sequence_9th_term (S : ℕ → ℕ) (d : ℕ) (Sn : ℕ) (a9 : ℕ) :
  (∀ n, S n = (n * (2 * S 1 + (n - 1) * d)) / 2) →
  d = 2 →
  Sn = 81 →
  S 9 = Sn →
  a9 = S 1 + 8 * d →
  a9 = 17 :=
by
  sorry

end arithmetic_sequence_9th_term_l62_62929


namespace MeatMarket_sales_l62_62811

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end MeatMarket_sales_l62_62811


namespace integral_f_l62_62154

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then x^2 else 2 - x

theorem integral_f :
  ∫ x in 0 .. 2, f x = 5 / 6 :=
by
  sorry

end integral_f_l62_62154


namespace alternating_sequence_probability_l62_62103

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l62_62103


namespace ratio_fifteenth_term_l62_62025

-- Definitions of S_n and T_n based on the given conditions
def S_n (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
def T_n (b e n : ℕ) : ℕ := n * (2 * b + (n - 1) * e) / 2

-- Statement of the problem
theorem ratio_fifteenth_term 
  (a b d e : ℕ) 
  (h : ∀ n, (S_n a d n : ℚ) / (T_n b e n : ℚ) = (9 * n + 5) / (6 * n + 31)) : 
  (a + 14 * d : ℚ) / (b + 14 * e : ℚ) = (92 : ℚ) / 71 :=
by sorry

end ratio_fifteenth_term_l62_62025


namespace inequality_iff_l62_62746

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end inequality_iff_l62_62746


namespace smallest_unpayable_amount_l62_62073

theorem smallest_unpayable_amount :
  ∀ (coins_1p coins_2p coins_3p coins_4p coins_5p : ℕ), 
    coins_1p = 1 → 
    coins_2p = 2 → 
    coins_3p = 3 → 
    coins_4p = 4 → 
    coins_5p = 5 → 
    ∃ (x : ℕ), x = 56 ∧ 
    ¬ (∃ (a b c d e : ℕ), a * 1 + b * 2 + c * 3 + d * 4 + e * 5 = x ∧ 
    a ≤ coins_1p ∧
    b ≤ coins_2p ∧
    c ≤ coins_3p ∧
    d ≤ coins_4p ∧
    e ≤ coins_5p) :=
by {
  -- Here we skip the actual proof
  sorry
}

end smallest_unpayable_amount_l62_62073


namespace line_through_A_with_zero_sum_of_intercepts_l62_62467

-- Definitions
def passesThroughPoint (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l A.1 A.2

def sumInterceptsZero (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, l a 0 ∧ l 0 b ∧ a + b = 0

-- Theorem statement
theorem line_through_A_with_zero_sum_of_intercepts (l : ℝ → ℝ → Prop) :
  passesThroughPoint (1, 4) l ∧ sumInterceptsZero l →
  (∀ x y, l x y ↔ 4 * x - y = 0) ∨ (∀ x y, l x y ↔ x - y + 3 = 0) :=
sorry

end line_through_A_with_zero_sum_of_intercepts_l62_62467


namespace hyperbola_equation_l62_62735

noncomputable def hyperbola : Prop :=
  ∃ (a b : ℝ), 
    (2 : ℝ) * a = (3 : ℝ) * b ∧
    ∀ (x y : ℝ), (4 * x^2 - 9 * y^2 = -32) → (x = 1) ∧ (y = 2)

theorem hyperbola_equation (a b : ℝ) :
  (2 * a = 3 * b) ∧ (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = -32 → x = 1 ∧ y = 2) → 
  (9 / 32 * y^2 - x^2 / 8 = 1) :=
by
  sorry

end hyperbola_equation_l62_62735


namespace sales_fifth_month_l62_62713

theorem sales_fifth_month (s1 s2 s3 s4 s6 s5 : ℝ) (target_avg total_sales : ℝ)
  (h1 : s1 = 4000)
  (h2 : s2 = 6524)
  (h3 : s3 = 5689)
  (h4 : s4 = 7230)
  (h6 : s6 = 12557)
  (h_avg : target_avg = 7000)
  (h_total_sales : total_sales = 42000) :
  s5 = 6000 :=
by
  sorry

end sales_fifth_month_l62_62713


namespace mark_first_part_playing_time_l62_62474

open Nat

theorem mark_first_part_playing_time (x : ℕ) (total_game_time second_part_playing_time sideline_time : ℕ)
  (h1 : total_game_time = 90) (h2 : second_part_playing_time = 35) (h3 : sideline_time = 35) 
  (h4 : x + second_part_playing_time + sideline_time = total_game_time) : x = 20 := 
by
  sorry

end mark_first_part_playing_time_l62_62474


namespace relationship_between_x_x_squared_and_x_cubed_l62_62455

theorem relationship_between_x_x_squared_and_x_cubed (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : x < x^3 ∧ x^3 < x^2 :=
by
  sorry

end relationship_between_x_x_squared_and_x_cubed_l62_62455


namespace johns_brother_age_l62_62935

variable (B : ℕ)
variable (J : ℕ)

-- Conditions given in the problem
def condition1 : Prop := J = 6 * B - 4
def condition2 : Prop := J + B = 10

-- The statement we want to prove, which is the answer to the problem:
theorem johns_brother_age (h1 : condition1 B J) (h2 : condition2 B J) : B = 2 := 
by 
  sorry

end johns_brother_age_l62_62935


namespace highest_power_of_2_divides_n_highest_power_of_3_divides_n_l62_62136

noncomputable def n : ℕ := 15^4 - 11^4

theorem highest_power_of_2_divides_n : ∃ k : ℕ, 2^4 = 16 ∧ 2^(k) ∣ n :=
by
  sorry

theorem highest_power_of_3_divides_n : ∃ m : ℕ, 3^0 = 1 ∧ 3^(m) ∣ n :=
by
  sorry

end highest_power_of_2_divides_n_highest_power_of_3_divides_n_l62_62136


namespace abs_diff_inequality_l62_62916

theorem abs_diff_inequality (a b c h : ℝ) (hab : |a - c| < h) (hbc : |b - c| < h) : |a - b| < 2 * h := 
by
  sorry

end abs_diff_inequality_l62_62916


namespace max_abs_value_l62_62266

theorem max_abs_value (x y : ℝ) (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : |x - 2 * y + 1| ≤ 6 :=
sorry

end max_abs_value_l62_62266


namespace minimum_triangle_area_l62_62524

theorem minimum_triangle_area :
  ∀ (m n : ℝ), (m > 0) ∧ (n > 0) ∧ (1 / m + 2 / n = 1) → (1 / 2 * m * n) = 4 :=
by
  sorry

end minimum_triangle_area_l62_62524


namespace line_equation_l62_62464

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4))
  (sum_intercepts_zero : ∃ a b : ℝ, (a + b = 0) ∧ (A.1 * b + A.2 * a = a * b)) :
  (∀ x y : ℝ, x - A.1 = (y - A.2) * 4 → 4 * x - y = 0) ∨
  (∀ x y : ℝ, (x / (-3)) + (y / 3) = 1 → x - y + 3 = 0) :=
sorry

end line_equation_l62_62464


namespace XF_XG_value_l62_62815

-- Define the given conditions
noncomputable def AB := 4
noncomputable def BC := 3
noncomputable def CD := 7
noncomputable def DA := 9

noncomputable def DX (BD : ℚ) := (1 / 3) * BD
noncomputable def BY (BD : ℚ) := (1 / 4) * BD

-- Variables and points in the problem
variables (BD p q : ℚ)
variables (A B C D X Y E F G : Point)

-- Proof statement
theorem XF_XG_value 
(AB_eq : AB = 4) (BC_eq : BC = 3) (CD_eq : CD = 7) (DA_eq : DA = 9)
(DX_eq : DX BD = (1 / 3) * BD) (BY_eq : BY BD = (1 / 4) * BD)
(AC_BD_prod : p * q = 55) :
  XF * XG = (110 / 9) := 
by
  sorry

end XF_XG_value_l62_62815


namespace find_n_divides_2_pow_2000_l62_62622

theorem find_n_divides_2_pow_2000 (n : ℕ) (h₁ : n > 2) :
  (1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ (2 ^ 2000) →
  n = 3 ∨ n = 7 ∨ n = 23 :=
sorry

end find_n_divides_2_pow_2000_l62_62622


namespace probability_same_heads_l62_62531

noncomputable def probability_heads_after_flips (p : ℚ) (n : ℕ) : ℚ :=
  (1 - p)^(n-1) * p

theorem probability_same_heads (p : ℚ) (n : ℕ) : p = 1/3 → 
  ∑' n : ℕ, (probability_heads_after_flips p n)^4 = 1/65 := 
sorry

end probability_same_heads_l62_62531


namespace line_through_A_with_zero_sum_of_intercepts_l62_62465

-- Definitions
def passesThroughPoint (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l A.1 A.2

def sumInterceptsZero (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, l a 0 ∧ l 0 b ∧ a + b = 0

-- Theorem statement
theorem line_through_A_with_zero_sum_of_intercepts (l : ℝ → ℝ → Prop) :
  passesThroughPoint (1, 4) l ∧ sumInterceptsZero l →
  (∀ x y, l x y ↔ 4 * x - y = 0) ∨ (∀ x y, l x y ↔ x - y + 3 = 0) :=
sorry

end line_through_A_with_zero_sum_of_intercepts_l62_62465


namespace total_balloons_correct_l62_62020

-- Define the number of blue balloons Joan and Melanie have
def Joan_balloons : ℕ := 40
def Melanie_balloons : ℕ := 41

-- Define the total number of blue balloons
def total_balloons : ℕ := Joan_balloons + Melanie_balloons

-- Prove that the total number of blue balloons is 81
theorem total_balloons_correct : total_balloons = 81 := by
  sorry

end total_balloons_correct_l62_62020


namespace gcd_20020_11011_l62_62142

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := 
by
  sorry

end gcd_20020_11011_l62_62142


namespace pq_proof_l62_62286

section ProofProblem

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, ∃ d, a (n + 1) = a n + d

def b_n (a : ℕ → ℤ) (n : ℕ) : ℤ :=
if n % 2 = 1 then a n - 6 else 2 * a n

def sum_first_n (seq : ℕ → ℤ) (n : ℕ) : ℤ :=
(n.sum (λ i, seq (i + 1)))

variables (a : ℕ → ℤ)
variables (S T : ℕ → ℤ)

axiom a_is_arithmetic : arithmetic_sequence a
axiom S_4_eq_32 : sum_first_n a 4 = 32
axiom T_3_eq_16 : sum_first_n (b_n a) 3 = 16

theorem pq_proof : (∀ n, a n = 2 * n + 3) ∧ (∀ n > 5, sum_first_n (b_n a) n > sum_first_n a n) :=
sorry

end ProofProblem

end pq_proof_l62_62286


namespace find_constants_and_calculate_result_l62_62222

theorem find_constants_and_calculate_result :
  ∃ (a b : ℤ), 
    (∀ (x : ℤ), (x + a) * (x + 6) = x^2 + 8 * x + 12) ∧ 
    (∀ (x : ℤ), (x - a) * (x + b) = x^2 + x - 6) ∧ 
    (∀ (x : ℤ), (x + a) * (x + b) = x^2 + 5 * x + 6) :=
by
  sorry

end find_constants_and_calculate_result_l62_62222


namespace grandfather_age_correct_l62_62559

-- Let's define the conditions
def xiaowen_age : ℕ := 13
def grandfather_age : ℕ := 5 * xiaowen_age + 8

-- The statement to prove
theorem grandfather_age_correct : grandfather_age = 73 := by
  sorry

end grandfather_age_correct_l62_62559


namespace profitable_allocation_2015_l62_62090

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end profitable_allocation_2015_l62_62090


namespace not_divisible_by_11_check_divisibility_by_11_l62_62483

theorem not_divisible_by_11 : Nat := 8

theorem check_divisibility_by_11 (n : Nat) (h: n = 98473092) : ¬ (11 ∣ not_divisible_by_11) := by
  sorry

end not_divisible_by_11_check_divisibility_by_11_l62_62483


namespace sum_of_numbers_l62_62349

-- Define the conditions
variables (a b : ℝ) (r d : ℝ)
def geometric_progression := a = 3 * r ∧ b = 3 * r^2
def arithmetic_progression := b = a + d ∧ 9 = b + d

-- Define the problem as proving the sum of a and b
theorem sum_of_numbers (h1 : geometric_progression a b r)
                       (h2 : arithmetic_progression a b d) : 
  a + b = 45 / 4 :=
sorry

end sum_of_numbers_l62_62349


namespace probability_one_and_three_painted_faces_l62_62871

-- Define the conditions of the problem
def side_length := 5
def total_unit_cubes := side_length^3
def painted_faces := 2
def unit_cubes_one_painted_face := 26
def unit_cubes_three_painted_faces := 4

-- Define the probability statement in Lean
theorem probability_one_and_three_painted_faces :
  (unit_cubes_one_painted_face * unit_cubes_three_painted_faces : ℝ) / (total_unit_cubes * (total_unit_cubes - 1) / 2) = 52 / 3875 :=
by
  sorry

end probability_one_and_three_painted_faces_l62_62871


namespace smallest_integer_solution_m_l62_62174

theorem smallest_integer_solution_m :
  (∃ x y m : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) →
  ∃ m : ℤ, (∀ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) ↔ m = -1 :=
by
  sorry

end smallest_integer_solution_m_l62_62174


namespace problem_l62_62003

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem problem (f : ℝ → ℝ) (h : isOddFunction f) : 
  f (-2012) + f (-2011) + f 0 + f 2011 + f 2012 = 0 :=
by
  sorry

end problem_l62_62003


namespace stockholm_to_uppsala_distance_l62_62828

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end stockholm_to_uppsala_distance_l62_62828


namespace additional_money_needed_l62_62030

theorem additional_money_needed :
  let total_budget := 500
  let budget_dresses := 300
  let budget_shoes := 150
  let budget_accessories := 50
  let extra_fraction := 2 / 5
  let discount_rate := 0.15
  let total_without_discount := 
    budget_dresses * (1 + extra_fraction) +
    budget_shoes * (1 + extra_fraction) +
    budget_accessories * (1 + extra_fraction)
  let discounted_total := total_without_discount * (1 - discount_rate)
  discounted_total > total_budget :=
sorry

end additional_money_needed_l62_62030


namespace inverse_prop_l62_62330

theorem inverse_prop (a b : ℝ) : (a > b) → (|a| > |b|) :=
sorry

end inverse_prop_l62_62330


namespace min_value_xy_l62_62162

theorem min_value_xy (x y : ℝ) (h1 : x + y = -1) (h2 : x < 0) (h3 : y < 0) :
  ∃ (xy_min : ℝ), (∀ (xy : ℝ), xy = x * y → xy + 1 / xy ≥ xy_min) ∧ xy_min = 17 / 4 :=
by
  sorry

end min_value_xy_l62_62162


namespace find_S2019_l62_62293

-- Conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Definitions and conditions extracted: conditions for sum of arithmetic sequence
axiom arithmetic_sum (n : ℕ) : S n = n * a (n / 2)
axiom OB_condition : a 3 + a 2017 = 1

-- Lean statement to prove S2019
theorem find_S2019 : S 2019 = 2019 / 2 := by
  sorry

end find_S2019_l62_62293


namespace real_part_of_z_l62_62921

theorem real_part_of_z (z : ℂ) (h : ∃ (r : ℝ), z^2 + z = r) : z.re = -1 / 2 :=
by
  sorry

end real_part_of_z_l62_62921


namespace speed_of_man_in_still_water_l62_62860

variable (v_m v_s : ℝ)

-- Conditions
def downstream_distance : ℝ := 51
def upstream_distance : ℝ := 18
def time : ℝ := 3

-- Equations based on the conditions
def downstream_speed_eq : Prop := downstream_distance = (v_m + v_s) * time
def upstream_speed_eq : Prop := upstream_distance = (v_m - v_s) * time

-- The theorem to prove
theorem speed_of_man_in_still_water : downstream_speed_eq v_m v_s ∧ upstream_speed_eq v_m v_s → v_m = 11.5 :=
by
  intro h
  sorry

end speed_of_man_in_still_water_l62_62860


namespace kaylee_current_age_l62_62770

-- Define the initial conditions
def matt_current_age : ℕ := 5
def kaylee_future_age_in_7_years : ℕ := 3 * matt_current_age

-- Define the main theorem to be proven
theorem kaylee_current_age : ∃ x : ℕ, x + 7 = kaylee_future_age_in_7_years ∧ x = 8 :=
by
  -- Use given conditions to instantiate the future age
  have h1 : kaylee_future_age_in_7_years = 3 * 5 := rfl
  have h2 : 3 * 5 = 15 := rfl
  have h3 : kaylee_future_age_in_7_years = 15 := by rw [h1, h2]
  -- Prove that there exists an x such that x + 7 = 15 and x = 8
  use 8
  split
  . rw [h3]
    norm_num
  . rfl

end kaylee_current_age_l62_62770


namespace probability_sum_30_l62_62724

-- Define the sets representing the faces of the two dice
def die1_faces : Set ℕ := { n | (n ∈ Finset.range 18) ∨ n = 19 }
def die2_faces : Set ℕ := { n | (n ∈ Finset.range 16) ∨ (n ∈ Finset.Icc 17 20) }

-- Define the rolling function for both dice
def valid_pairs : Set (ℕ × ℕ) := { (a, b) | a ∈ die1_faces ∧ b ∈ die2_faces ∧ a + b = 30 }

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 400

-- Prove the probability statement
theorem probability_sum_30 : 
  (valid_pairs.card : ℚ) / total_outcomes = 1 / 100 :=
sorry

end probability_sum_30_l62_62724


namespace factorization_correct_l62_62363

theorem factorization_correct (x y : ℝ) : x^2 * y - x * y^2 = x * y * (x - y) :=
by
  sorry

end factorization_correct_l62_62363


namespace line_through_A_with_zero_sum_of_intercepts_l62_62466

-- Definitions
def passesThroughPoint (A : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l A.1 A.2

def sumInterceptsZero (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a b : ℝ, l a 0 ∧ l 0 b ∧ a + b = 0

-- Theorem statement
theorem line_through_A_with_zero_sum_of_intercepts (l : ℝ → ℝ → Prop) :
  passesThroughPoint (1, 4) l ∧ sumInterceptsZero l →
  (∀ x y, l x y ↔ 4 * x - y = 0) ∨ (∀ x y, l x y ↔ x - y + 3 = 0) :=
sorry

end line_through_A_with_zero_sum_of_intercepts_l62_62466


namespace remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l62_62729

theorem remainder_of_x_pow_105_div_x_sq_sub_4x_add_3 :
  ∀ (x : ℤ), (x^105) % (x^2 - 4*x + 3) = (3^105 * (x-1) - (x-2)) / 2 :=
by sorry

end remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l62_62729


namespace percent_democrats_is_60_l62_62649
-- Import the necessary library

-- Define the problem conditions
variables (D R : ℝ)
variables (h1 : D + R = 100)
variables (h2 : 0.70 * D + 0.20 * R = 50)

-- State the theorem to be proved
theorem percent_democrats_is_60 (D R : ℝ) (h1 : D + R = 100) (h2 : 0.70 * D + 0.20 * R = 50) : D = 60 :=
by
  sorry

end percent_democrats_is_60_l62_62649


namespace luncheon_tables_needed_l62_62235

theorem luncheon_tables_needed (invited : ℕ) (no_show : ℕ) (people_per_table : ℕ) (people_attended : ℕ) (tables_needed : ℕ) :
  invited = 47 →
  no_show = 7 →
  people_per_table = 5 →
  people_attended = invited - no_show →
  tables_needed = people_attended / people_per_table →
  tables_needed = 8 := by {
  -- Proof here
  sorry
}

end luncheon_tables_needed_l62_62235


namespace simplify_expression_l62_62818

noncomputable def givenExpression : ℝ := 
  abs (-0.01) ^ 2 - (-5 / 8) ^ 0 - 3 ^ (Real.log 2 / Real.log 3) + 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 5) + Real.log 5

theorem simplify_expression : givenExpression = -1.9999 := by
  sorry

end simplify_expression_l62_62818


namespace probability_of_matching_colors_l62_62384

theorem probability_of_matching_colors :
  let abe_jelly_beans := ["green", "red", "blue"]
  let bob_jelly_beans := ["green", "green", "yellow", "yellow", "red", "red", "red"]
  let abe_probs := (1 / 3, 1 / 3, 1 / 3)
  let bob_probs := (2 / 7, 3 / 7, 0)
  let matching_prob := (1 / 3 * 2 / 7) + (1 / 3 * 3 / 7)
  matching_prob = 5 / 21 := by sorry

end probability_of_matching_colors_l62_62384


namespace jennifer_money_left_l62_62790

def money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ) : ℚ :=
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_money_left :
  money_left 150 (1/5) (1/6) (1/2) = 20 := by
  -- Proof goes here
  sorry

end jennifer_money_left_l62_62790


namespace little_john_initial_money_l62_62945

def sweets_cost : ℝ := 2.25
def friends_donation : ℝ := 2 * 2.20
def money_left : ℝ := 3.85

theorem little_john_initial_money :
  sweets_cost + friends_donation + money_left = 10.50 :=
by
  sorry

end little_john_initial_money_l62_62945


namespace Probability_X_in_Interval_0_4_l62_62774

noncomputable def X : Type := sorry

def normal_distribution (μ σ: ℝ) (σ_pos: σ > 0) :  X → ℝ := sorry
def prob_event : set X → ℝ := sorry

variable (σ : ℝ)
variable (hσ : σ > 0)
variable (P : ℝ)

-- Given X follows a normal distribution N(4, σ^2)
axiom h1 : normal_distribution 4 σ hσ

-- Given the probability of X in interval (0, 8) is 0.6
axiom h2 : prob_event {x : X | 0 < x ∧ x < 8} = 0.6

-- Prove the probability of X in interval (0, 4) is 0.3
theorem Probability_X_in_Interval_0_4 : prob_event {x : X | 0 < x ∧ x < 4} = 0.3 := 
by sorry

end Probability_X_in_Interval_0_4_l62_62774


namespace perfect_square_trinomial_l62_62169

theorem perfect_square_trinomial (m : ℝ) :
  ∃ (a : ℝ), (∀ (x : ℝ), x^2 - 2*(m-3)*x + 16 = (x - a)^2) ↔ (m = 7 ∨ m = -1) := by
  sorry

end perfect_square_trinomial_l62_62169


namespace sugar_needed_287_163_l62_62982

theorem sugar_needed_287_163 :
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sugar_stored + additional_sugar_needed = 450 :=
by
  let sugar_stored := 287
  let additional_sugar_needed := 163
  sorry

end sugar_needed_287_163_l62_62982


namespace work_efficiency_ratio_l62_62714

theorem work_efficiency_ratio (A B : ℝ) (k : ℝ)
  (h1 : A = k * B)
  (h2 : B = 1 / 27)
  (h3 : A + B = 1 / 9) :
  k = 2 :=
by
  sorry

end work_efficiency_ratio_l62_62714


namespace probability_identical_cubes_l62_62527

-- Definitions translating given conditions
def total_ways_to_paint_single_cube : Nat := 3^6
def total_ways_to_paint_three_cubes : Nat := total_ways_to_paint_single_cube^3

-- Cases counting identical painting schemes
def identical_painting_schemes : Nat :=
  let case_A := 3
  let case_B := 90
  let case_C := 540
  case_A + case_B + case_C

-- The main theorem stating the desired probability
theorem probability_identical_cubes :
  let total_ways := (387420489 : ℚ) -- 729^3
  let favorable_ways := (633 : ℚ)  -- sum of all cases (3 + 90 + 540)
  favorable_ways / total_ways = (211 / 129140163 : ℚ) :=
by
  sorry

end probability_identical_cubes_l62_62527


namespace tour_group_size_l62_62844

def adult_price : ℕ := 8
def child_price : ℕ := 3
def total_spent : ℕ := 44

theorem tour_group_size :
  ∃ (x y : ℕ), adult_price * x + child_price * y = total_spent ∧ (x + y = 8 ∨ x + y = 13) :=
by
  sorry

end tour_group_size_l62_62844


namespace fixed_point_l62_62832

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : func a 1 = 1 :=
by {
  -- We need to prove that func a 1 = 1 for any a > 0 and a ≠ 1
  sorry
}

end fixed_point_l62_62832


namespace cary_needs_six_weekends_l62_62399

theorem cary_needs_six_weekends
  (shoe_cost : ℕ)
  (saved : ℕ)
  (earn_per_lawn : ℕ)
  (lawns_per_weekend : ℕ)
  (additional_needed : ℕ := shoe_cost - saved)
  (earn_per_weekend : ℕ := earn_per_lawn * lawns_per_weekend)
  (weekends_needed : ℕ := additional_needed / earn_per_weekend) :
  shoe_cost = 120 ∧ saved = 30 ∧ earn_per_lawn = 5 ∧ lawns_per_weekend = 3 → weekends_needed = 6 := by 
  sorry

end cary_needs_six_weekends_l62_62399


namespace weight_shaina_receives_l62_62316

namespace ChocolateProblem

-- Definitions based on conditions
def total_chocolate : ℚ := 60 / 7
def piles : ℚ := 5
def weight_per_pile : ℚ := total_chocolate / piles
def shaina_piles : ℚ := 2

-- Proposition to represent the question and correct answer
theorem weight_shaina_receives : 
  (weight_per_pile * shaina_piles) = 24 / 7 := 
by
  sorry

end ChocolateProblem

end weight_shaina_receives_l62_62316


namespace other_team_scored_l62_62664

open Nat

def points_liz_scored (free_throws three_pointers jump_shots : Nat) : Nat :=
  free_throws * 1 + three_pointers * 3 + jump_shots * 2

def points_deficit := 20
def points_liz_deficit := points_liz_scored 5 3 4 - points_deficit
def final_loss_margin := 8
def other_team_score := points_liz_scored 5 3 4 + final_loss_margin

theorem other_team_scored
  (points_liz : Nat := points_liz_scored 5 3 4)
  (final_deficit : Nat := points_deficit)
  (final_margin : Nat := final_loss_margin)
  (other_team_points : Nat := other_team_score) :
  other_team_points = 30 := 
sorry

end other_team_scored_l62_62664


namespace find_z_l62_62543

theorem find_z : 
    ∃ z : ℝ, ( ( 2 ^ 5 ) * ( 9 ^ 2 ) ) / ( z * ( 3 ^ 5 ) ) = 0.16666666666666666 ↔ z = 64 :=
by
    sorry

end find_z_l62_62543


namespace calculate_expression_l62_62258

theorem calculate_expression : 3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := 
by sorry

end calculate_expression_l62_62258


namespace sequence_formula_l62_62317

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n →  1 / a (n + 1) = 1 / a n + 1) :
  ∀ n : ℕ, 0 < n → a n = 1 / n :=
by {
  sorry
}

end sequence_formula_l62_62317


namespace painter_rooms_painted_l62_62378

theorem painter_rooms_painted (total_rooms : ℕ) (hours_per_room : ℕ) (remaining_hours : ℕ) 
    (h1 : total_rooms = 9) (h2 : hours_per_room = 8) (h3 : remaining_hours = 32) : 
    total_rooms - (remaining_hours / hours_per_room) = 5 :=
by
  sorry

end painter_rooms_painted_l62_62378


namespace ellipse_circle_parallelogram_condition_l62_62156

theorem ellipse_circle_parallelogram_condition
  (a b : ℝ)
  (C₀ : ∀ x y : ℝ, x^2 + y^2 = 1)
  (C₁ : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h : a > 0 ∧ b > 0 ∧ a > b) :
  1 / a^2 + 1 / b^2 = 1 := by
  sorry

end ellipse_circle_parallelogram_condition_l62_62156


namespace last_two_digits_sum_of_factorials_1_to_15_l62_62987

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_sum_of_factorials_1_to_15 :
  last_two_digits ( (∑ i in Finset.range 16, factorial i) ) = 13 := 
sorry

end last_two_digits_sum_of_factorials_1_to_15_l62_62987


namespace ab_zero_if_conditions_l62_62095

theorem ab_zero_if_conditions 
  (a b : ℤ)
  (h : |a - b| + |a * b| = 2) : a * b = 0 :=
  sorry

end ab_zero_if_conditions_l62_62095


namespace difference_of_two_smallest_integers_divisors_l62_62847

theorem difference_of_two_smallest_integers_divisors (n m : ℕ) (h₁ : n > 1) (h₂ : m > 1) 
(h₃ : n % 2 = 1) (h₄ : n % 3 = 1) (h₅ : n % 4 = 1) (h₆ : n % 5 = 1) 
(h₇ : n % 6 = 1) (h₈ : n % 7 = 1) (h₉ : n % 8 = 1) (h₁₀ : n % 9 = 1) 
(h₁₁ : n % 10 = 1) (h₃' : m % 2 = 1) (h₄' : m % 3 = 1) (h₅' : m % 4 = 1) 
(h₆' : m % 5 = 1) (h₇' : m % 6 = 1) (h₈' : m % 7 = 1) (h₉' : m % 8 = 1) 
(h₁₀' : m % 9 = 1) (h₁₁' : m % 10 = 1): m - n = 2520 :=
sorry

end difference_of_two_smallest_integers_divisors_l62_62847


namespace find_a5_div_b5_l62_62681

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ := n * (a 0 + a (n - 1)) / 2

-- Main statement
theorem find_a5_div_b5 (a b : ℕ → ℤ) (S T : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : ∀ n : ℕ, S n = sum_first_n_terms a n)
  (h4 : ∀ n : ℕ, T n = sum_first_n_terms b n)
  (h5 : ∀ n : ℕ, S n * (3 * n + 1) = 2 * n * T n) :
  (a 5 : ℚ) / b 5 = 9 / 14 :=
by
  sorry

end find_a5_div_b5_l62_62681


namespace wendy_adds_18_gallons_l62_62360

-- Definitions based on the problem
def truck_tank_capacity : ℕ := 20
def car_tank_capacity : ℕ := 12
def truck_tank_fraction_full : ℚ := 1 / 2
def car_tank_fraction_full : ℚ := 1 / 3

-- Conditions on the amount of gallons currently in the tanks
def truck_current_gallons : ℚ := truck_tank_capacity * truck_tank_fraction_full
def car_current_gallons : ℚ := car_tank_capacity * car_tank_fraction_full

-- Amount of gallons needed to fill up each tank
def truck_gallons_to_add : ℚ := truck_tank_capacity - truck_current_gallons
def car_gallons_to_add : ℚ := car_tank_capacity - car_current_gallons

-- Total gallons needed to fill both tanks
def total_gallons_to_add : ℚ := truck_gallons_to_add + car_gallons_to_add

-- Theorem statement
theorem wendy_adds_18_gallons :
  total_gallons_to_add = 18 := sorry

end wendy_adds_18_gallons_l62_62360


namespace total_blue_balloons_l62_62019

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l62_62019


namespace flu_infection_equation_l62_62472

theorem flu_infection_equation
  (x : ℝ) :
  (1 + x)^2 = 25 :=
sorry

end flu_infection_equation_l62_62472


namespace original_number_is_perfect_square_l62_62221

variable (n : ℕ)

theorem original_number_is_perfect_square
  (h1 : n = 1296)
  (h2 : ∃ m : ℕ, (n + 148) = m^2) : ∃ k : ℕ, n = k^2 :=
by
  sorry

end original_number_is_perfect_square_l62_62221


namespace problem_l62_62270

variable (a b : ℝ)

theorem problem (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 4) : a^2 + b^2 ≥ 8 := 
sorry

end problem_l62_62270


namespace find_angle_A_l62_62291

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h1 : 2 * Real.sin B = Real.sqrt 3 * b) 
  (h2 : a = 2) (h3 : ∃ area : ℝ, area = Real.sqrt 3 ∧ area = (1 / 2) * b * c * Real.sin A) :
  A = Real.pi / 3 ∧ b = 2 ∧ c = 2 :=
by
  sorry

end find_angle_A_l62_62291


namespace relationship_between_x_x_squared_and_x_cubed_l62_62456

theorem relationship_between_x_x_squared_and_x_cubed (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : x < x^3 ∧ x^3 < x^2 :=
by
  sorry

end relationship_between_x_x_squared_and_x_cubed_l62_62456


namespace solve_equation_l62_62840

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end solve_equation_l62_62840


namespace real_life_distance_between_cities_l62_62826

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end real_life_distance_between_cities_l62_62826


namespace probability_of_drawing_white_ball_l62_62742

theorem probability_of_drawing_white_ball (P_A P_B P_C : ℝ) 
    (hA : P_A = 0.4) 
    (hB : P_B = 0.25)
    (hSum : P_A + P_B + P_C = 1) : 
    P_C = 0.35 :=
by
    -- Placeholder for the proof
    sorry

end probability_of_drawing_white_ball_l62_62742


namespace number_of_visits_to_save_enough_l62_62930

-- Define constants
def total_cost_with_sauna : ℕ := 250
def pool_cost_no_sauna (y : ℕ) : ℕ := y + 200
def headphone_cost : ℕ := 275

-- Define assumptions
axiom sauna_cost (y : ℕ) : total_cost_with_sauna = pool_cost_no_sauna y + y
axiom savings_per_visit (y x : ℕ) : x = pool_cost_no_sauna y -> total_cost_with_sauna - x = 25
axiom visits_needed (savings_per_visit headphone_cost : ℕ) : headphone_cost = savings_per_visit * 11

-- Formulate the theorem
theorem number_of_visits_to_save_enough (y : ℕ) (x : ℕ) :
  sauna_cost y -> savings_per_visit y x -> visits_needed 25 headphone_cost -> x / 25 = 11 :=
by {
  sorry
}

end number_of_visits_to_save_enough_l62_62930


namespace relationship_between_x_x2_and_x3_l62_62454

theorem relationship_between_x_x2_and_x3 (x : ℝ) (h : -1 < x ∧ x < 0) :
  x ^ 3 < x ∧ x < x ^ 2 :=
by
  sorry

end relationship_between_x_x2_and_x3_l62_62454


namespace percentage_of_engineers_from_university_A_l62_62387

theorem percentage_of_engineers_from_university_A :
  let original_engineers := 20
  let new_hired_engineers := 8
  let percentage_original_from_A := 0.65
  let original_from_A := percentage_original_from_A * original_engineers
  let total_engineers := original_engineers + new_hired_engineers
  let total_from_A := original_from_A + new_hired_engineers
  (total_from_A / total_engineers) * 100 = 75 :=
by
  sorry

end percentage_of_engineers_from_university_A_l62_62387


namespace line_equation_l62_62463

theorem line_equation (A : ℝ × ℝ) (hA : A = (1, 4))
  (sum_intercepts_zero : ∃ a b : ℝ, (a + b = 0) ∧ (A.1 * b + A.2 * a = a * b)) :
  (∀ x y : ℝ, x - A.1 = (y - A.2) * 4 → 4 * x - y = 0) ∨
  (∀ x y : ℝ, (x / (-3)) + (y / 3) = 1 → x - y + 3 = 0) :=
sorry

end line_equation_l62_62463


namespace shirley_cases_needed_l62_62816

-- Define the given conditions
def trefoils_boxes := 54
def samoas_boxes := 36
def boxes_per_case := 6

-- The statement to prove
theorem shirley_cases_needed : trefoils_boxes / boxes_per_case >= samoas_boxes / boxes_per_case ∧ 
                               samoas_boxes / boxes_per_case = 6 :=
by
  let n_cases := samoas_boxes / boxes_per_case
  have h1 : trefoils_boxes / boxes_per_case = 9 := sorry
  have h2 : samoas_boxes / boxes_per_case = 6 := sorry
  have h3 : 9 >= 6 := by linarith
  exact ⟨h3, h2⟩


end shirley_cases_needed_l62_62816


namespace find_n_l62_62166

noncomputable def tangent_line_problem (x0 : ℝ) (n : ℕ) : Prop :=
(x0 ∈ Set.Ioo (Real.sqrt n) (Real.sqrt (n + 1))) ∧
(∃ m : ℝ, 0 < m ∧ m < 1 ∧ (2 * x0 = 1 / m) ∧ (x0^2 = (Real.log m - 1)))

theorem find_n (x0 : ℝ) (n : ℕ) :
  tangent_line_problem x0 n → n = 2 :=
sorry

end find_n_l62_62166


namespace find_value_of_a_l62_62158

theorem find_value_of_a 
  (P : ℝ × ℝ)
  (a : ℝ)
  (α : ℝ)
  (point_on_terminal_side : P = (-4, a))
  (sin_cos_condition : Real.sin α * Real.cos α = Real.sqrt 3 / 4) : 
  a = -4 * Real.sqrt 3 ∨ a = - (4 * Real.sqrt 3 / 3) :=
sorry

end find_value_of_a_l62_62158


namespace max_rank_awarded_l62_62098

theorem max_rank_awarded (num_participants rank_threshold total_possible_points : ℕ)
  (H1 : num_participants = 30)
  (H2 : rank_threshold = (30 * 29 / 2 : ℚ) * 0.60)
  (H3 : total_possible_points = (30 * 29 / 2)) :
  ∃ max_awarded : ℕ, max_awarded ≤ 23 :=
by {
  -- Proof omitted
  sorry
}

end max_rank_awarded_l62_62098


namespace area_of_square_containing_circle_l62_62522

theorem area_of_square_containing_circle (r : ℝ) (hr : r = 4) :
  ∃ (a : ℝ), a = 64 ∧ (∀ (s : ℝ), s = 2 * r → a = s * s) :=
by
  use 64
  sorry

end area_of_square_containing_circle_l62_62522


namespace jack_walking_rate_l62_62229

variables (distance : ℝ) (time_hours : ℝ)
#check distance  -- ℝ (real number)
#check time_hours  -- ℝ (real number)

-- Define the conditions
def jack_distance : Prop := distance = 9
def jack_time : Prop := time_hours = 1 + 15 / 60

-- Define the statement to prove
theorem jack_walking_rate (h1 : jack_distance distance) (h2 : jack_time time_hours) :
  (distance / time_hours) = 7.2 :=
sorry

end jack_walking_rate_l62_62229


namespace shirt_cost_l62_62013

theorem shirt_cost (S : ℝ) (hats_cost jeans_cost total_cost : ℝ)
  (h_hats : hats_cost = 4)
  (h_jeans : jeans_cost = 10)
  (h_total : total_cost = 51)
  (h_eq : 3 * S + 2 * jeans_cost + 4 * hats_cost = total_cost) :
  S = 5 :=
by
  -- The main proof will be provided here
  sorry

end shirt_cost_l62_62013


namespace chord_ratio_l62_62851

theorem chord_ratio (EQ GQ HQ FQ : ℝ) (h1 : EQ = 5) (h2 : GQ = 12) (h3 : HQ = 3) (h4 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by
  sorry

end chord_ratio_l62_62851


namespace clock1_runs_10_months_longer_l62_62819

noncomputable def battery_a_charge (C_B : ℝ) := 6 * C_B
noncomputable def clock1_total_charge (C_B : ℝ) := 2 * battery_a_charge C_B
noncomputable def clock2_total_charge (C_B : ℝ) := 2 * C_B
noncomputable def clock2_operating_time := 2
noncomputable def clock1_operating_time (C_B : ℝ) := clock1_total_charge C_B / C_B
noncomputable def operating_time_difference (C_B : ℝ) := clock1_operating_time C_B - clock2_operating_time

theorem clock1_runs_10_months_longer (C_B : ℝ) :
  operating_time_difference C_B = 10 :=
by
  unfold operating_time_difference clock1_operating_time clock2_operating_time clock1_total_charge battery_a_charge
  sorry

end clock1_runs_10_months_longer_l62_62819


namespace incorrect_statement_l62_62907

-- Define the operation (x * y)
def op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem to show the incorrectness of the given statement
theorem incorrect_statement (x y z : ℝ) : op x (y + z) ≠ op x y + op x z :=
  sorry

end incorrect_statement_l62_62907


namespace gcf_48_160_120_l62_62698

theorem gcf_48_160_120 : Nat.gcd (Nat.gcd 48 160) 120 = 8 := by
  sorry

end gcf_48_160_120_l62_62698


namespace car_truck_ratio_l62_62327

theorem car_truck_ratio (total_vehicles trucks cars : ℕ)
  (h1 : total_vehicles = 300)
  (h2 : trucks = 100)
  (h3 : cars + trucks = total_vehicles)
  (h4 : ∃ (k : ℕ), cars = k * trucks) : 
  cars / trucks = 2 :=
by
  sorry

end car_truck_ratio_l62_62327


namespace monotonicity_f_range_of_b_l62_62911

noncomputable def f (a x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

def p (a b : ℝ) (x : ℝ) : Prop := f a x ≤ 2 * b
def q (b : ℝ) : Prop := ∀ x, (x = -3 → (x^2 + (2*b + 1)*x - b - 1) > 0) ∧ 
                           (x = -2 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 0 → (x^2 + (2*b + 1)*x - b - 1) < 0) ∧ 
                           (x = 1 → (x^2 + (2*b + 1)*x - b - 1) > 0)

theorem monotonicity_f (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : ∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 := by
  sorry

theorem range_of_b (b : ℝ) (hp_or : ∃ x, p a b x ∨ q b) (hp_and : ∀ x, ¬(p a b x ∧ q b)) :
    (1/5 < b ∧ b < 1/2) ∨ (b ≥ 5/7) := by
    sorry

end monotonicity_f_range_of_b_l62_62911


namespace neg_p_equiv_l62_62161

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x / (x - 1) > 0

theorem neg_p_equiv :
  ¬p I ↔ ∃ x ∈ I, x / (x - 1) ≤ 0 ∨ x - 1 = 0 :=
by
  sorry

end neg_p_equiv_l62_62161


namespace identity_implies_a_minus_b_l62_62302

theorem identity_implies_a_minus_b (a b : ℚ) (y : ℚ) (h : y > 0) :
  (∀ y, y > 0 → (a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5)))) → (a - b = 1) :=
by
  sorry

end identity_implies_a_minus_b_l62_62302


namespace sufficient_but_not_necessary_l62_62300

def l1 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => m * x + (m + 1) * y + 2

def l2 (m : ℝ) : ℝ × ℝ → ℝ
| (x, y) => (m + 1) * x + (m + 4) * y - 3

def perpendicular_slopes (m : ℝ) : Prop :=
  let slope_l1 := -m / (m + 1)
  let slope_l2 := -(m + 1) / (m + 4)
  slope_l1 * slope_l2 = -1

theorem sufficient_but_not_necessary (m : ℝ) : m = -2 → (∃ k, m = -k ∧ perpendicular_slopes k) :=
by
  sorry

end sufficient_but_not_necessary_l62_62300


namespace sum_of_possible_g9_values_l62_62497

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (y : ℝ) : ℝ := 3 * y + 2

theorem sum_of_possible_g9_values : ∀ {x1 x2 : ℝ}, f x1 = 9 → f x2 = 9 → g x1 + g x2 = 22 := by
  intros
  sorry

end sum_of_possible_g9_values_l62_62497


namespace clips_ratio_l62_62324

def clips (April May: Nat) : Prop :=
  April = 48 ∧ April + May = 72 → (48 / (72 - 48)) = 2

theorem clips_ratio : clips 48 (72 - 48) :=
by
  sorry

end clips_ratio_l62_62324


namespace equations_of_ellipse_and_parabola_range_of_x0_l62_62902

theorem equations_of_ellipse_and_parabola (a b p x y : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (h4 : 0 < p)
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (h_parabola : y^2 = 2 * p * x)
  (h_focus : ∃ F2 : ℝ × ℝ, F2 = (1, 0))
  (h_distance_M_yaxis : ∀ M : ℝ × ℝ, M.1 = x → abs (M.1 - F2.1) - 1 = abs M.2 - abs (M.1 - F2.1))
  (h_Q : ∃ Q : ℝ × ℝ, Q.1 = x ∧ abs (Q.1 - F2.1) = 5/2)
  : ∃ p_eq : ℝ, p_eq = 2 ∧ (y^2 = 4 * x ∧ (x^2 / 9 + y^2 / 8 = 1)) := sorry

theorem range_of_x0 (k m : ℝ) 
  (h1 : k ≠ 0) (h2 : m ≠ 0)
  (h_tangent : ∀ y : ℝ, y = k * y + m → y^2 = 4 * (y - m / k))
  (h_midpoint : ∃ x0 y0 : ℝ, x0 = (-9) / (9 * (k^2) + 8) ∧ -1 < x0 ∧ x0 < 0)
  : -1 < x0 ∧ x0 < 0 := sorry

end equations_of_ellipse_and_parabola_range_of_x0_l62_62902


namespace ten_percent_of_n_l62_62105

variable (n f : ℝ)

theorem ten_percent_of_n (h : n - (1 / 4 * 2) - (1 / 3 * 3) - f * n = 27) : 
  0.10 * n = 0.10 * (28.5 / (1 - f)) :=
by
  simp only [*, mul_one_div_cancel, mul_sub, sub_eq_add_neg, add_div, div_self, one_div, mul_add]
  sorry

end ten_percent_of_n_l62_62105


namespace sixtieth_term_of_arithmetic_sequence_l62_62831

theorem sixtieth_term_of_arithmetic_sequence (a1 a15 : ℚ) (d : ℚ) (h1 : a1 = 7) (h2 : a15 = 37)
  (h3 : a15 = a1 + 14 * d) : a1 + 59 * d = 134.5 := by
  sorry

end sixtieth_term_of_arithmetic_sequence_l62_62831


namespace Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l62_62949

-- Define P and Q as propositions where P indicates submission of all required essays and Q indicates failing the course.
variable (P Q : Prop)

-- Ms. Thompson's statement translated to logical form.
theorem Ms_Thompsons_statement : ¬P → Q := sorry

-- The goal is to prove that if a student did not fail the course, then they submitted all the required essays.
theorem contrapositive_of_Ms_Thompsons_statement (h : ¬Q) : P := 
by {
  -- Proof will go here
  sorry 
}

end Ms_Thompsons_statement_contrapositive_of_Ms_Thompsons_statement_l62_62949


namespace prob_different_topics_l62_62581

theorem prob_different_topics (T : ℕ) (hT : T = 6) :
  let total_outcomes := T * T,
      favorable_outcomes := T * (T - 1),
      probability_different := favorable_outcomes / total_outcomes
  in probability_different = 5 / 6 :=
by
  have : total_outcomes = 36 := by rw [hT]; norm_num
  have : favorable_outcomes = 30 := by rw [hT]; norm_num
  have : probability_different = 5 / 6 := by norm_num
  sorry

end prob_different_topics_l62_62581


namespace distinct_license_plates_l62_62717

theorem distinct_license_plates :
  let num_digits := 10
  let num_letters := 26
  let num_digit_positions := 5
  let num_letter_pairs := num_letters * num_letters
  let num_letter_positions := num_digit_positions + 1
  num_digits^num_digit_positions * num_letter_pairs * num_letter_positions = 40560000 := by
  sorry

end distinct_license_plates_l62_62717


namespace complement_of_A_in_U_l62_62759

noncomputable def U : Set ℤ := {-3, -1, 0, 1, 3}

noncomputable def A : Set ℤ := {x | x^2 - 2 * x - 3 = 0}

theorem complement_of_A_in_U : (U \ A) = {-3, 0, 1} :=
by sorry

end complement_of_A_in_U_l62_62759


namespace apples_per_pie_l62_62523

theorem apples_per_pie (total_apples handed_out_apples pies made_pies remaining_apples : ℕ) 
  (h_initial : total_apples = 86)
  (h_handout : handed_out_apples = 30)
  (h_made_pies : made_pies = 7)
  (h_remaining : remaining_apples = total_apples - handed_out_apples) :
  remaining_apples / made_pies = 8 :=
by
  sorry

end apples_per_pie_l62_62523


namespace compare_store_costs_l62_62711

-- Define the conditions mathematically
def StoreA_cost (x : ℕ) : ℝ := 5 * x + 125
def StoreB_cost (x : ℕ) : ℝ := 4.5 * x + 135

theorem compare_store_costs (x : ℕ) (h : x ≥ 5) : 
  5 * 15 + 125 = 200 ∧ 4.5 * 15 + 135 = 202.5 ∧ 200 < 202.5 := 
by
  -- Here the theorem states the claims to be proved.
  sorry

end compare_store_costs_l62_62711


namespace maximum_volume_l62_62722

noncomputable def volume (x : ℝ) : ℝ :=
  (48 - 2*x)^2 * x

theorem maximum_volume :
  (∀ x : ℝ, (0 < x) ∧ (x < 24) → volume x ≤ volume 8) ∧ (volume 8 = 8192) :=
by
  sorry

end maximum_volume_l62_62722


namespace isosceles_triangle_angle_measure_l62_62121

theorem isosceles_triangle_angle_measure:
  ∀ (α β : ℝ), (α = 112.5) → (2 * β + α = 180) → β = 33.75 :=
by
  intros α β hα h_sum
  sorry

end isosceles_triangle_angle_measure_l62_62121


namespace prob_one_side_of_tri_in_decagon_is_half_l62_62267

noncomputable def probability_one_side_of_tri_in_decagon : ℚ :=
  let num_vertices := 10
  let total_triangles := Nat.choose num_vertices 3
  let favorable_triangles := 10 * 6
  favorable_triangles / total_triangles

theorem prob_one_side_of_tri_in_decagon_is_half :
  probability_one_side_of_tri_in_decagon = 1 / 2 := by
  sorry

end prob_one_side_of_tri_in_decagon_is_half_l62_62267


namespace simplify_expr_l62_62333

-- Define the expression
def expr := |-4^2 + 7|

-- State the theorem
theorem simplify_expr : expr = 9 :=
by sorry

end simplify_expr_l62_62333


namespace factorial_last_two_digits_sum_eq_l62_62996

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l62_62996


namespace min_value_f_l62_62137

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 24 * x + 128 / x^3

theorem min_value_f : ∃ x > 0, f x = 168 :=
by
  sorry

end min_value_f_l62_62137


namespace simplify_expression_l62_62045

-- Define the algebraic expressions
def expr1 (x : ℝ) := (3 * x - 4) * (x + 9)
def expr2 (x : ℝ) := (x + 6) * (3 * x + 2)
def combined_expr (x : ℝ) := expr1 x + expr2 x
def result_expr (x : ℝ) := 6 * x^2 + 43 * x - 24

-- Theorem stating the equivalence
theorem simplify_expression (x : ℝ) : combined_expr x = result_expr x := 
by 
  sorry

end simplify_expression_l62_62045


namespace minimum_value_proof_l62_62749

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x^2 / (x + 2)) + (y^2 / (y + 1))

theorem minimum_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  min_value x y = 1 / 4 :=
  sorry

end minimum_value_proof_l62_62749


namespace hcf_of_210_and_671_l62_62683

theorem hcf_of_210_and_671 :
  let lcm := 2310
  let a := 210
  let b := 671
  gcd a b = 61 :=
by
  let lcm := 2310
  let a := 210
  let b := 671
  let hcf := gcd a b
  have rel : lcm * hcf = a * b := by sorry
  have hcf_eq : hcf = 61 := by sorry
  exact hcf_eq

end hcf_of_210_and_671_l62_62683


namespace ratio_of_amount_spent_on_movies_to_weekly_allowance_l62_62313

-- Define weekly allowance
def weekly_allowance : ℕ := 10

-- Define final amount after all transactions
def final_amount : ℕ := 11

-- Define earnings from washing the car
def earnings : ℕ := 6

-- Define amount left before washing the car
def amount_left_before_wash : ℕ := final_amount - earnings

-- Define amount spent on movies
def amount_spent_on_movies : ℕ := weekly_allowance - amount_left_before_wash

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- Prove the required ratio
theorem ratio_of_amount_spent_on_movies_to_weekly_allowance :
  ratio amount_spent_on_movies weekly_allowance = 1 / 2 :=
by
  sorry

end ratio_of_amount_spent_on_movies_to_weekly_allowance_l62_62313


namespace twentieth_term_arithmetic_sequence_eq_neg49_l62_62210

-- Definitions based on the conditions
def a1 : ℤ := 8
def d : ℤ := 5 - 8
def a (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The proof statement
theorem twentieth_term_arithmetic_sequence_eq_neg49 : a 20 = -49 :=
by 
  -- Proof will be inserted here
  sorry

end twentieth_term_arithmetic_sequence_eq_neg49_l62_62210


namespace rest_area_milepost_l62_62680

theorem rest_area_milepost 
  (milepost_fifth_exit : ℕ) 
  (milepost_fifteenth_exit : ℕ) 
  (rest_area_milepost : ℕ)
  (h1 : milepost_fifth_exit = 50)
  (h2 : milepost_fifteenth_exit = 350)
  (h3 : rest_area_milepost = (milepost_fifth_exit + (milepost_fifteenth_exit - milepost_fifth_exit) / 2)) :
  rest_area_milepost = 200 := 
by
  intros
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end rest_area_milepost_l62_62680


namespace find_a_l62_62925

theorem find_a (a : ℝ) (h : (2 - -3) / (1 - a) = Real.tan (135 * Real.pi / 180)) : a = 6 :=
sorry

end find_a_l62_62925


namespace usual_time_is_42_l62_62694

noncomputable def usual_time_to_school (R T : ℝ) := T * R
noncomputable def improved_time_to_school (R T : ℝ) := ((7/6) * R) * (T - 6)

theorem usual_time_is_42 (R T : ℝ) :
  (usual_time_to_school R T) = (improved_time_to_school R T) → T = 42 :=
by
  sorry

end usual_time_is_42_l62_62694


namespace smallest_angle_of_triangle_l62_62964

theorem smallest_angle_of_triangle (k : ℕ) (h : 4 * k + 5 * k + 9 * k = 180) : 4 * k = 40 :=
by {
  sorry
}

end smallest_angle_of_triangle_l62_62964


namespace solution_set_of_inequality_l62_62837

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -0.5 ∨ x > 2} := 
sorry

end solution_set_of_inequality_l62_62837


namespace single_discount_equivalent_l62_62873

theorem single_discount_equivalent :
  ∀ (original final: ℝ) (d1 d2 d3 total_discount: ℝ),
  original = 800 →
  d1 = 0.15 →
  d2 = 0.10 →
  d3 = 0.05 →
  final = original * (1 - d1) * (1 - d2) * (1 - d3) →
  total_discount = 1 - (final / original) →
  total_discount = 0.27325 :=
by
  intros original final d1 d2 d3 total_discount h1 h2 h3 h4 h5 h6
  sorry

end single_discount_equivalent_l62_62873


namespace divides_by_3_l62_62496

theorem divides_by_3 (a b c : ℕ) (h : 9 ∣ a ^ 3 + b ^ 3 + c ^ 3) : 3 ∣ a ∨ 3 ∣ b ∨ 3 ∣ c :=
sorry

end divides_by_3_l62_62496


namespace jose_share_is_correct_l62_62983

noncomputable def total_profit : ℝ := 
  5000 - 2000 + 7000 + 1000 - 3000 + 10000 + 500 + 4000 - 2500 + 6000 + 8000 - 1000

noncomputable def tom_investment_ratio : ℝ := 30000 * 12
noncomputable def jose_investment_ratio : ℝ := 45000 * 10
noncomputable def maria_investment_ratio : ℝ := 60000 * 8

noncomputable def total_investment_ratio : ℝ := tom_investment_ratio + jose_investment_ratio + maria_investment_ratio

noncomputable def jose_share : ℝ := (jose_investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_is_correct : jose_share = 14658 := 
by 
  sorry

end jose_share_is_correct_l62_62983


namespace solve_numRedBalls_l62_62772

-- Condition (1): There are a total of 10 balls in the bag
def totalBalls : ℕ := 10

-- Condition (2): The probability of drawing a black ball is 2/5
-- This means the number of black balls is 4
def numBlackBalls : ℕ := 4

-- Condition (3): The probability of drawing at least 1 white ball when drawing 2 balls is 7/9
def probAtLeastOneWhiteBall : ℚ := 7 / 9

-- The number of red balls in the bag is calculated based on the given conditions
def numRedBalls (totalBalls numBlackBalls : ℕ) (probAtLeastOneWhiteBall : ℚ) : ℕ := 
  let totalWhiteAndRedBalls := totalBalls - numBlackBalls
  let probTwoNonWhiteBalls := 1 - probAtLeastOneWhiteBall
  let comb (n k : ℕ) := Nat.choose n k
  let equation := comb totalWhiteAndRedBalls 2 * comb (totalBalls - 2) 0 / comb totalBalls 2
  if equation = probTwoNonWhiteBalls then totalWhiteAndRedBalls else 0

theorem solve_numRedBalls : numRedBalls totalBalls numBlackBalls probAtLeastOneWhiteBall = 1 := by
  sorry

end solve_numRedBalls_l62_62772


namespace triangle_solution_condition_l62_62613

-- Definitions of segments
variables {A B D E : Type}
variables (c f g : Real)

-- Allow noncomputable definitions for geometric constraints
noncomputable def triangle_construction (c f g : Real) : String :=
  if c > f then "more than one solution"
  else if c = f then "exactly one solution"
  else "no solution"

-- The proof problem statement
theorem triangle_solution_condition (c f g : Real) :
  (c > f → triangle_construction c f g = "more than one solution") ∧
  (c = f → triangle_construction c f g = "exactly one solution") ∧
  (c < f → triangle_construction c f g = "no solution") :=
by
  sorry

end triangle_solution_condition_l62_62613


namespace smallest_integer_rel_prime_to_1020_l62_62082

theorem smallest_integer_rel_prime_to_1020 : ∃ n : ℕ, n > 1 ∧ n = 7 ∧ gcd n 1020 = 1 := by
  -- Here we state the theorem
  sorry

end smallest_integer_rel_prime_to_1020_l62_62082


namespace eccentricity_of_hyperbola_l62_62322

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) : ℝ :=
  (1 + b^2 / a^2) ^ (1/2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 h3 = Real.sqrt 3 := 
by
  unfold hyperbola_eccentricity
  rw [h3]
  simp
  sorry

end eccentricity_of_hyperbola_l62_62322


namespace distance_to_airport_l62_62135

theorem distance_to_airport
  (t : ℝ)
  (d : ℝ)
  (h1 : 45 * (t + 1) + 20 = d)
  (h2 : d - 65 = 65 * (t - 1))
  : d = 390 := by
  sorry

end distance_to_airport_l62_62135


namespace estate_problem_l62_62029

def totalEstateValue (E a b : ℝ) : Prop :=
  (a + b = (3/5) * E) ∧ 
  (a = 2 * b) ∧ 
  (3 * b = (3/5) * E) ∧ 
  (E = a + b + (3 * b) + 4000)

theorem estate_problem (E : ℝ) (a b : ℝ) :
  totalEstateValue E a b → E = 20000 :=
by
  -- The proof will be filled here
  sorry

end estate_problem_l62_62029


namespace kylie_first_hour_apples_l62_62193

variable (A : ℕ) -- The number of apples picked in the first hour

-- Definitions based on the given conditions
def applesInFirstHour := A
def applesInSecondHour := 2 * A
def applesInThirdHour := A / 3

-- Total number of apples picked in all three hours
def totalApplesPicked := applesInFirstHour + applesInSecondHour + applesInThirdHour

-- The given condition that the total number of apples picked is 220
axiom total_is_220 : totalApplesPicked = 220

-- Proving that the number of apples picked in the first hour is 66
theorem kylie_first_hour_apples : A = 66 := by
  sorry

end kylie_first_hour_apples_l62_62193


namespace current_short_trees_l62_62845

theorem current_short_trees (S : ℕ) (S_planted : ℕ) (S_total : ℕ) 
  (H1 : S_planted = 105) 
  (H2 : S_total = 217) 
  (H3 : S + S_planted = S_total) :
  S = 112 :=
by
  sorry

end current_short_trees_l62_62845


namespace AM_GM_inequality_example_l62_62741

theorem AM_GM_inequality_example (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) : 
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_inequality_example_l62_62741


namespace set_union_proof_l62_62299

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end set_union_proof_l62_62299


namespace T_gt_S_for_n_gt_5_l62_62290

-- Define the arithmetic sequence a_n
def a : ℕ → ℕ
| n => 2 * n + 3

-- Define the sequence b_n
def b : ℕ → ℕ
| n => if n % 2 = 1 then a n - 6 else 2 * (a n)

-- Sum of first n terms of a
def S (n : ℕ) : ℕ := (List.range n).sum (λ k => a (k + 1))

-- Sum of first n terms of b
def T (n : ℕ) : ℕ := (List.range n).sum (λ k => b (k + 1))

-- Given conditions in Lean
axiom S_4 : S 4 = 32
axiom T_3 : T 3 = 16

-- The theorem to prove
theorem T_gt_S_for_n_gt_5 (n : ℕ) (hn : n > 5) : T n > S n :=
  sorry

end T_gt_S_for_n_gt_5_l62_62290


namespace largest_decimal_number_l62_62618

theorem largest_decimal_number :
  max (0.9123 : ℝ) (max (0.9912 : ℝ) (max (0.9191 : ℝ) (max (0.9301 : ℝ) (0.9091 : ℝ)))) = 0.9912 :=
by
  sorry

end largest_decimal_number_l62_62618


namespace adult_tickets_sold_l62_62063

open Nat

theorem adult_tickets_sold (A C : ℕ) (h₁ : A + C = 522) (h₂ : 15 * A + 8 * C = 5086) :
  A = 130 :=
by
  sorry

end adult_tickets_sold_l62_62063


namespace find_median_and_mode_l62_62009

def data_set := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.qsort (· ≤ ·)
  sorted.get! ((sorted.length - 1) / 2)

def mode (l : List ℕ) : ℕ :=
  l.foldl
    (λ counts n => counts.insert n (counts.find n |>.getD 0 + 1))
    (Std.HashMap.empty ℕ ℕ)
  |>.toList
  |>.foldl (λ acc (k, v) => if v > acc.2 then (k, v) else acc) (0, 0)
  |>.1

theorem find_median_and_mode :
  median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end find_median_and_mode_l62_62009


namespace probability_of_alternating_draws_l62_62101

theorem probability_of_alternating_draws :
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls in
  let successful_orders : ℕ := 2,
      total_arrangements : ℕ := Nat.choose total_balls white_balls,
      probability : ℚ := successful_orders / total_arrangements in
  probability = 1 / 126 :=
by
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls
  let successful_orders : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let probability : ℚ := successful_orders / total_arrangements
  sorry

end probability_of_alternating_draws_l62_62101


namespace largest_non_formable_amount_l62_62517

-- Definitions and conditions from the problem
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def cannot_be_formed (n a b : ℕ) : Prop :=
  ∀ x y : ℕ, n ≠ a * x + b * y

-- The statement to prove
theorem largest_non_formable_amount :
  is_coprime 8 15 ∧ cannot_be_formed 97 8 15 :=
by
  sorry

end largest_non_formable_amount_l62_62517


namespace correct_option_b_l62_62546

theorem correct_option_b (a : ℝ) : 
  (-2 * a) ^ 3 = -8 * a ^ 3 :=
by sorry

end correct_option_b_l62_62546


namespace single_discount_equivalence_l62_62870

theorem single_discount_equivalence (original_price : ℝ) (first_discount second_discount : ℝ) (final_price : ℝ) :
  original_price = 50 →
  first_discount = 0.30 →
  second_discount = 0.10 →
  final_price = original_price * (1 - first_discount) * (1 - second_discount) →
  ((original_price - final_price) / original_price) * 100 = 37 := by
  sorry

end single_discount_equivalence_l62_62870


namespace multiple_of_three_l62_62744

theorem multiple_of_three (a b : ℤ) : ∃ k : ℤ, (a + b = 3 * k) ∨ (ab = 3 * k) ∨ (a - b = 3 * k) :=
sorry

end multiple_of_three_l62_62744


namespace number_of_solutions_l62_62891

theorem number_of_solutions : 
  ∃ n : ℕ, (∀ x y : ℕ, 3 * x + 4 * y = 766 → x % 2 = 0 → x > 0 → y > 0 → x = n * 2) ∧ n = 127 := 
by
  sorry

end number_of_solutions_l62_62891


namespace new_boarders_joined_l62_62070

theorem new_boarders_joined (initial_boarders new_boarders initial_day_students total_boarders total_day_students: ℕ)
  (h1: initial_boarders = 60)
  (h2: initial_day_students = 150)
  (h3: total_boarders = initial_boarders + new_boarders)
  (h4: total_day_students = initial_day_students)
  (h5: 2 * initial_day_students = 5 * initial_boarders)
  (h6: 2 * total_boarders = total_day_students) :
  new_boarders = 15 :=
by
  sorry

end new_boarders_joined_l62_62070


namespace div_five_times_eight_by_ten_l62_62538

theorem div_five_times_eight_by_ten : (5 * 8) / 10 = 4 := by
  sorry

end div_five_times_eight_by_ten_l62_62538


namespace multiple_of_Roseville_population_l62_62050

noncomputable def Willowdale_population : ℕ := 2000

noncomputable def Roseville_population : ℕ :=
  (3 * Willowdale_population) - 500

noncomputable def SunCity_population : ℕ := 12000

theorem multiple_of_Roseville_population :
  ∃ m : ℕ, SunCity_population = (m * Roseville_population) + 1000 ∧ m = 2 :=
by
  sorry

end multiple_of_Roseville_population_l62_62050


namespace sandy_marbles_correct_l62_62486

namespace MarbleProblem

-- Define the number of dozens Jessica has
def jessica_dozens : ℕ := 3

-- Define the conversion from dozens to individual marbles
def dozens_to_marbles (d : ℕ) : ℕ := 12 * d

-- Calculate the number of marbles Jessica has
def jessica_marbles : ℕ := dozens_to_marbles jessica_dozens

-- Define the multiplier for Sandy's marbles
def sandy_multiplier : ℕ := 4

-- Define the number of marbles Sandy has
def sandy_marbles : ℕ := sandy_multiplier * jessica_marbles

theorem sandy_marbles_correct : sandy_marbles = 144 :=
by
  sorry

end MarbleProblem

end sandy_marbles_correct_l62_62486


namespace three_point_five_six_as_fraction_l62_62225

theorem three_point_five_six_as_fraction : (356 / 100 : ℝ) = (89 / 25 : ℝ) :=
begin
  sorry
end

end three_point_five_six_as_fraction_l62_62225


namespace geometric_progression_identity_l62_62955

theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a * c) : 
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := 
by
  sorry

end geometric_progression_identity_l62_62955


namespace mean_age_of_seven_friends_l62_62181

theorem mean_age_of_seven_friends 
  (mean_age_group1: ℕ)
  (mean_age_group2: ℕ)
  (n1: ℕ)
  (n2: ℕ)
  (total_friends: ℕ) :
  mean_age_group1 = 147 → 
  mean_age_group2 = 161 →
  n1 = 3 → 
  n2 = 4 →
  total_friends = 7 →
  (mean_age_group1 * n1 + mean_age_group2 * n2) / total_friends = 155 := by
  sorry

end mean_age_of_seven_friends_l62_62181


namespace general_formula_a_n_T_n_greater_than_S_n_l62_62279

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n_l62_62279


namespace side_length_a_cosine_A_l62_62753

variable (A B C : Real)
variable (a b c : Real)
variable (triangle_inequality : a + b + c = 10)
variable (sine_equation : Real.sin B + Real.sin C = 4 * Real.sin A)
variable (bc_product : b * c = 16)

theorem side_length_a :
  a = 2 :=
  sorry

theorem cosine_A :
  b + c = 8 → 
  a = 2 → 
  b * c = 16 →
  Real.cos A = 7 / 8 :=
  sorry

end side_length_a_cosine_A_l62_62753


namespace find_a_l62_62643

theorem find_a (a : ℝ) : (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 12 * x + a = (2 * x + b) ^ 2)) → a = 9 :=
by
  intro h
  sorry

end find_a_l62_62643


namespace base_n_representation_of_b_l62_62918

theorem base_n_representation_of_b (n a b : ℕ) (hn : n > 8) 
  (h_n_solution : ∃ m, m ≠ n ∧ n * m = b ∧ n + m = a) 
  (h_a_base_n : 1 * n + 8 = a) :
  (b = 8 * n) :=
by
  sorry

end base_n_representation_of_b_l62_62918


namespace gcd_1248_1001_l62_62141

theorem gcd_1248_1001 : Nat.gcd 1248 1001 = 13 := by
  sorry

end gcd_1248_1001_l62_62141


namespace mr_johnson_total_volunteers_l62_62808

theorem mr_johnson_total_volunteers (students_per_class : ℕ) (classes : ℕ) (teachers : ℕ) (additional_volunteers : ℕ) :
  students_per_class = 5 → classes = 6 → teachers = 13 → additional_volunteers = 7 →
  (students_per_class * classes + teachers + additional_volunteers) = 50 :=
by intros; simp [*]

end mr_johnson_total_volunteers_l62_62808


namespace relationship_y1_y2_l62_62034

theorem relationship_y1_y2
    (b : ℝ) 
    (y1 y2 : ℝ)
    (h1 : y1 = - (1 / 2) * (-2) + b) 
    (h2 : y2 = - (1 / 2) * 3 + b) : 
    y1 > y2 :=
sorry

end relationship_y1_y2_l62_62034


namespace jaeho_got_most_notebooks_l62_62666

-- Define the number of notebooks each friend received
def notebooks_jaehyuk : ℕ := 12
def notebooks_kyunghwan : ℕ := 3
def notebooks_jaeho : ℕ := 15

-- Define the statement proving that Jaeho received the most notebooks
theorem jaeho_got_most_notebooks : notebooks_jaeho > notebooks_jaehyuk ∧ notebooks_jaeho > notebooks_kyunghwan :=
by {
  sorry -- this is where the proof would go
}

end jaeho_got_most_notebooks_l62_62666


namespace initial_cherry_sweets_30_l62_62377

/-!
# Problem Statement
A packet of candy sweets has some cherry-flavored sweets (C), 40 strawberry-flavored sweets, 
and 50 pineapple-flavored sweets. Aaron eats half of each type of sweet and then gives away 
5 cherry-flavored sweets to his friend. There are still 55 sweets in the packet of candy.
Prove that the initial number of cherry-flavored sweets was 30.
-/

noncomputable def initial_cherry_sweets (C : ℕ) : Prop :=
  let remaining_cherry_sweets := C / 2 - 5
  let remaining_strawberry_sweets := 40 / 2
  let remaining_pineapple_sweets := 50 / 2
  remaining_cherry_sweets + remaining_strawberry_sweets + remaining_pineapple_sweets = 55

theorem initial_cherry_sweets_30 : initial_cherry_sweets 30 :=
  sorry

end initial_cherry_sweets_30_l62_62377


namespace quadratic_real_solutions_l62_62644

theorem quadratic_real_solutions (p : ℝ) : (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 :=
sorry

end quadratic_real_solutions_l62_62644


namespace projectile_reaches_50_first_at_0point5_l62_62969

noncomputable def height_at_time (t : ℝ) : ℝ := -16 * t^2 + 100 * t

theorem projectile_reaches_50_first_at_0point5 :
  ∃ t : ℝ, (height_at_time t = 50) ∧ (t = 0.5) :=
sorry

end projectile_reaches_50_first_at_0point5_l62_62969


namespace santino_total_fruits_l62_62512

theorem santino_total_fruits :
  let p := 2
  let m := 3
  let a := 4
  let o := 5
  let fp := 10
  let fm := 20
  let fa := 15
  let fo := 25
  p * fp + m * fm + a * fa + o * fo = 265 := by
  sorry

end santino_total_fruits_l62_62512


namespace incorrect_operation_B_l62_62083

variables (a b c : ℝ)

theorem incorrect_operation_B : (c - 2 * (a + b)) ≠ (c - 2 * a + 2 * b) := by
  sorry

end incorrect_operation_B_l62_62083


namespace clothing_discount_l62_62373

theorem clothing_discount (P : ℝ) :
  let first_sale_price := (4 / 5) * P
  let second_sale_price := first_sale_price * 0.60
  second_sale_price = (12 / 25) * P :=
by
  sorry

end clothing_discount_l62_62373


namespace probability_units_digit_l62_62109

noncomputable def probability_units_digit_condition : ℚ :=
  let total_outcomes := 10
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_units_digit {n : ℕ} (hl : 10000 ≤ n) (hr : n ≤ 99999) :
  probability_units_digit_condition = 1 / 2 :=
by
  sorry

end probability_units_digit_l62_62109


namespace find_angle_measure_l62_62881

theorem find_angle_measure (x : ℝ) (h : x = 2 * (90 - x) + 30) : x = 70 :=
by
  exact sorry

end find_angle_measure_l62_62881


namespace sequence_divisibility_l62_62592

theorem sequence_divisibility (g : ℕ → ℕ) (h₁ : g 1 = 1) 
(h₂ : ∀ n : ℕ, g (n + 1) = g n ^ 2 + g n + 1) 
(n : ℕ) : g n ^ 2 + 1 ∣ g (n + 1) ^ 2 + 1 :=
sorry

end sequence_divisibility_l62_62592


namespace negation_correct_l62_62685

-- Define the original proposition
def original_proposition (x : ℝ) : Prop :=
  ∀ x > 0, x^2 - 2 * x + 1 ≥ 0

-- Define what it means to negate the proposition
def negated_proposition (x : ℝ) : Prop :=
  ∃ x > 0, x^2 - 2 * x + 1 < 0

-- Main statement: the negation of the original proposition equals the negated proposition
theorem negation_correct : (¬original_proposition x) = (negated_proposition x) :=
  sorry

end negation_correct_l62_62685


namespace pencils_ratio_l62_62892

theorem pencils_ratio 
  (cindi_pencils : ℕ := 60)
  (marcia_mul_cindi : ℕ := 2)
  (total_pencils : ℕ := 480)
  (marcia_pencils : ℕ := marcia_mul_cindi * cindi_pencils) 
  (donna_pencils : ℕ := total_pencils - marcia_pencils) :
  donna_pencils / marcia_pencils = 3 := by
  sorry

end pencils_ratio_l62_62892


namespace students_admitted_to_universities_l62_62075

open Finset

theorem students_admitted_to_universities :
  let students := {A, B, C, D}
  let universities := {1, 2, 3, 4}
  (students.card = 4) → 
  (universities.card = 4) →
  (∃ s : Finset (Finset (Σ u : universities, students)), s.card = 144) := 
by
  sorry

end students_admitted_to_universities_l62_62075


namespace base8_plus_base16_as_base10_l62_62416

lemma base8_to_base10 : ∀ (n : ℕ), nat.digits 8 n = [7, 3, 5] → n = 351 :=
by simp [nat.digits_eq_foldr_reverse, nat.of_digits, nat.base_repr_le]

lemma base16_to_base10 : ∀ (n : ℕ), nat.digits 16 n = [14, 2, 12, 1] → n = 7214 :=
by simp [nat.digits_eq_foldr_reverse, nat.of_digits, nat.base_repr_le]

theorem base8_plus_base16_as_base10 : nat.of_digits 8 [5, 3, 7] + nat.of_digits 16 [1, 12, 2, 14] = 7565 :=
by {
  have h1 : nat.of_digits 8 [5, 3, 7] = 351 := by simp [nat.of_digits],
  have h2 : nat.of_digits 16 [1, 12, 2, 14] = 7214 := by simp [nat.of_digits],
  simp [h1, h2]
}

end base8_plus_base16_as_base10_l62_62416


namespace fish_count_total_l62_62520

def Jerk_Tuna_fish : ℕ := 144
def Tall_Tuna_fish : ℕ := 2 * Jerk_Tuna_fish
def Total_fish_together : ℕ := Jerk_Tuna_fish + Tall_Tuna_fish

theorem fish_count_total :
  Total_fish_together = 432 :=
by
  sorry

end fish_count_total_l62_62520


namespace book_pages_l62_62312

theorem book_pages (books sheets pages_per_sheet pages_per_book : ℕ)
  (hbooks : books = 2)
  (hpages_per_sheet : pages_per_sheet = 8)
  (hsheets : sheets = 150)
  (htotal_pages : pages_per_sheet * sheets = 1200)
  (hpages_per_book : pages_per_book = 1200 / books) :
  pages_per_book = 600 :=
by
  -- Proof goes here
  sorry

end book_pages_l62_62312


namespace tan_diff_pi_over_4_l62_62898

theorem tan_diff_pi_over_4 (α : ℝ) (hα1 : π < α) (hα2 : α < 3 / 2 * π) (hcos : Real.cos α = -4 / 5) :
  Real.tan (π / 4 - α) = 1 / 7 := by
  sorry

end tan_diff_pi_over_4_l62_62898


namespace second_valve_rate_difference_l62_62555

theorem second_valve_rate_difference (V1 V2 : ℝ) 
  (h1 : V1 = 12000 / 120)
  (h2 : V1 + V2 = 12000 / 48) :
  V2 - V1 = 50 :=
by
  -- Since h1: V1 = 100
  -- And V1 + V2 = 250 from h2
  -- Therefore V2 = 250 - 100 = 150
  -- And V2 - V1 = 150 - 100 = 50
  sorry

end second_valve_rate_difference_l62_62555


namespace minimize_f_sin_65_sin_40_l62_62274

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := m^2 * x^2 + (n + 1) * x + 1

theorem minimize_f_sin_65_sin_40 (m n : ℝ) (h₁ : m = Real.sin (65 * Real.pi / 180))
  (h₂ : n = Real.sin (40 * Real.pi / 180)) : 
  ∃ x, x = -1 ∧ (∀ y, f y m n ≥ f (-1) m n) :=
by
  -- Proof to be completed
  sorry

end minimize_f_sin_65_sin_40_l62_62274


namespace ratio_water_to_orange_juice_l62_62326

variable (O W : ℝ)

-- Conditions:
-- 1. Amount of orange juice is O for both days.
-- 2. Amount of water is W on the first day and 2W on the second day.
-- 3. Price per glass is $0.60 on the first day and $0.40 on the second day.

theorem ratio_water_to_orange_juice 
  (h : (O + W) * 0.60 = (O + 2 * W) * 0.40) : 
  W / O = 1 := 
by 
  -- The proof is skipped
  sorry

end ratio_water_to_orange_juice_l62_62326


namespace inequality_proof_l62_62195

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c) ^ 2 :=
sorry

end inequality_proof_l62_62195


namespace calculation_correct_l62_62544

theorem calculation_correct : -2 + 3 = 1 :=
by
  sorry

end calculation_correct_l62_62544


namespace slices_with_all_toppings_l62_62866

-- Definitions
def slices_with_pepperoni (x y w : ℕ) : ℕ := 15 - x - y + w
def slices_with_mushrooms (x z w : ℕ) : ℕ := 16 - x - z + w
def slices_with_olives (y z w : ℕ) : ℕ := 10 - y - z + w

-- Problem's total validation condition
axiom total_slices_with_at_least_one_topping (x y z w : ℕ) :
  15 + 16 + 10 - x - y - z - 2 * w = 24

-- Statement to prove
theorem slices_with_all_toppings (x y z w : ℕ) (h : 15 + 16 + 10 - x - y - z - 2 * w = 24) : w = 2 :=
sorry

end slices_with_all_toppings_l62_62866


namespace ratio_of_women_to_men_l62_62567

theorem ratio_of_women_to_men (M W : ℕ) 
  (h1 : M + W = 72) 
  (h2 : M - 16 = W + 8) : 
  W / M = 1 / 2 :=
sorry

end ratio_of_women_to_men_l62_62567


namespace second_valve_rate_difference_l62_62553

theorem second_valve_rate_difference (V1 V2 : ℝ) 
  (h1 : V1 = 12000 / 120)
  (h2 : V1 + V2 = 12000 / 48) :
  V2 - V1 = 50 :=
by
  -- Since h1: V1 = 100
  -- And V1 + V2 = 250 from h2
  -- Therefore V2 = 250 - 100 = 150
  -- And V2 - V1 = 150 - 100 = 50
  sorry

end second_valve_rate_difference_l62_62553


namespace certain_number_l62_62624

theorem certain_number (G : ℕ) (N : ℕ) (H1 : G = 129) 
  (H2 : N % G = 9) (H3 : 2206 % G = 13) : N = 2202 :=
by
  sorry

end certain_number_l62_62624


namespace solve_equation_l62_62841

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end solve_equation_l62_62841


namespace union_example_l62_62632

open Set

variable (A B : Set ℤ)
variable (AB : Set ℤ)

theorem union_example (hA : A = {-3, 1, 2})
                      (hB : B = {0, 1, 2, 3}) :
                      A ∪ B = {-3, 0, 1, 2, 3} :=
by
  rw [hA, hB]
  ext
  simp
  sorry

end union_example_l62_62632


namespace max_positive_root_eq_l62_62133

theorem max_positive_root_eq (b c : ℝ) (h_b : |b| ≤ 3) (h_c : |c| ≤ 3) : 
  ∃ x, x = (3 + Real.sqrt 21) / 2 ∧ x^2 + b * x + c = 0 ∧ x ≥ 0 :=
by
  sorry

end max_positive_root_eq_l62_62133


namespace compound_interest_rate_is_10_percent_l62_62536

theorem compound_interest_rate_is_10_percent
  (P : ℝ) (CI : ℝ) (t : ℝ) (A : ℝ) (n : ℝ) (r : ℝ)
  (hP : P = 4500) (hCI : CI = 945.0000000000009) (ht : t = 2) (hn : n = 1) (hA : A = P + CI)
  (h_eq : A = P * (1 + r / n)^(n * t)) :
  r = 0.1 :=
by
  sorry

end compound_interest_rate_is_10_percent_l62_62536


namespace cyclicity_A_H_P_K_l62_62565

variables {A B C H D E H' F P K : Point}
variables {Hcircle : Circle H A}
variables {tri_abc : Triangle A B C}
variables {tri_ade : Triangle A D E}
variables {tri_pde : Triangle P D E}
variables {tri_pbc : Triangle P B C}
variables {line_ah' : Line A H'}
variables {line_de : Line D E}
variables {line_hh' : Line H H'}
variables {line_pf : Line P F}
variables {quad_bcde : Quadrilateral B C D E}

noncomputable def H_is_orthocenter_triang_abc : IsOrthocenter H tri_abc := sorry
noncomputable def H_circle : Circle_with_center H A := sorry
noncomputable def H'_is_orthocenter_triang_ade : IsOrthocenter H' tri_ade := sorry
noncomputable def AH'_intersects_DE_at_F : Intersects_At line_ah' line_de F := sorry
noncomputable def P_inside_quad_bcde : Inside_Quadrilateral P quad_bcde := sorry
noncomputable def Triangles_similar : SimilarTriangles tri_pde tri_pbc := sorry
noncomputable def K_is_intersection : Intersection line_hh' line_pf K := sorry

theorem cyclicity_A_H_P_K :
  Cyclic {A, H, P, K} :=
sorry

end cyclicity_A_H_P_K_l62_62565


namespace sphere_surface_area_l62_62877

theorem sphere_surface_area (V : ℝ) (hV : V = 72 * Real.pi) : 
  ∃ S : ℝ, S = 36 * 2^(2/3) * Real.pi :=
by
  sorry

end sphere_surface_area_l62_62877


namespace fraction_zero_solve_l62_62909

theorem fraction_zero_solve (x : ℝ) (h : (x^2 - 49) / (x + 7) = 0) : x = 7 :=
by
  sorry

end fraction_zero_solve_l62_62909


namespace total_weight_of_compound_l62_62081

variable (molecular_weight : ℕ) (moles : ℕ)

theorem total_weight_of_compound (h1 : molecular_weight = 72) (h2 : moles = 4) :
  moles * molecular_weight = 288 :=
by
  sorry

end total_weight_of_compound_l62_62081


namespace circle_chord_segments_l62_62476

theorem circle_chord_segments (r : ℝ) (ch : ℝ) (a : ℝ) :
  (r = 8) ∧ (ch = 12) ∧ (r^2 - a^2 = 36) →
  a = 2 * Real.sqrt 7 → ∃ (ak bk : ℝ), ak = 8 - 2 * Real.sqrt 7 ∧ bk = 8 + 2 * Real.sqrt 7 :=
by
  sorry

end circle_chord_segments_l62_62476


namespace compute_ratio_l62_62499

theorem compute_ratio (x y z a : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : z ≠ x) (h4 : x + y + z = a) (h5 : a ≠ 0) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = 1 / 3 :=
by
  -- Proof will be filled in here
  sorry

end compute_ratio_l62_62499


namespace inequality_subtract_l62_62438

-- Definitions of the main variables and conditions
variables {a b : ℝ}
-- Condition that should hold
axiom h : a > b

-- Expected conclusion
theorem inequality_subtract : a - 1 > b - 2 :=
by
  sorry

end inequality_subtract_l62_62438


namespace geometric_sequence_product_l62_62788

theorem geometric_sequence_product {a : ℕ → ℝ} 
(h₁ : a 1 = 2) 
(h₂ : a 5 = 8) 
(h_geom : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
a 2 * a 3 * a 4 = 64 := 
sorry

end geometric_sequence_product_l62_62788


namespace equilateral_triangle_black_area_l62_62386

theorem equilateral_triangle_black_area :
  let initial_black_area := 1
  let change_fraction := 5/6 * 9/10
  let area_after_n_changes (n : Nat) : ℚ := initial_black_area * (change_fraction ^ n)
  area_after_n_changes 3 = 27/64 := 
by
  sorry

end equilateral_triangle_black_area_l62_62386


namespace binomial_coeff_divisibility_l62_62204

theorem binomial_coeff_divisibility (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : n ∣ (Nat.choose n k) * Nat.gcd n k :=
sorry

end binomial_coeff_divisibility_l62_62204


namespace soccer_team_selection_l62_62721

-- Definitions of the problem
def total_members := 16
def utility_exclusion_cond := total_members - 1

-- Lean statement for the proof problem, using the conditions and answer:
theorem soccer_team_selection :
  (utility_exclusion_cond) * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4) = 409500 :=
by
  sorry

end soccer_team_selection_l62_62721


namespace second_car_avg_mpg_l62_62012

theorem second_car_avg_mpg 
  (x y : ℝ) 
  (h1 : x + y = 75) 
  (h2 : 25 * x + 35 * y = 2275) : 
  y = 40 := 
by sorry

end second_car_avg_mpg_l62_62012


namespace problem_solution_l62_62647

theorem problem_solution (a b : ℝ) (h1 : 2 + 3 = -b) (h2 : 2 * 3 = -2 * a) : a + b = -8 :=
by
  sorry

end problem_solution_l62_62647


namespace find_value_of_N_l62_62914

theorem find_value_of_N 
  (N : ℝ) 
  (h : (20 / 100) * N = (30 / 100) * 2500) 
  : N = 3750 := 
sorry

end find_value_of_N_l62_62914


namespace range_of_a_l62_62963

theorem range_of_a (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (a : ℝ) (heqn : 3 * x + a * (2 * y - 4 * real.exp 1 * x) * (real.log y - real.log x) = 0) : 
a < 0 ∨ a ≥ 3 / (2 * real.exp 1) := 
sorry

end range_of_a_l62_62963


namespace composite_of_squares_l62_62917

theorem composite_of_squares (n : ℕ) (h1 : 8 * n + 1 = x^2) (h2 : 24 * n + 1 = y^2) (h3 : n > 1) : ∃ a b : ℕ, a ∣ (8 * n + 3) ∧ b ∣ (8 * n + 3) ∧ a ≠ 1 ∧ b ≠ 1 ∧ a ≠ (8 * n + 3) ∧ b ≠ (8 * n + 3) := by
  sorry

end composite_of_squares_l62_62917


namespace inequality_proof_l62_62977

variable {a b c d : ℝ}

theorem inequality_proof
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (h_sum : a + b + c + d = 3) :
  1 / a^3 + 1 / b^3 + 1 / c^3 + 1 / d^3 ≤ 1 / (a^3 * b^3 * c^3 * d^3) :=
sorry

end inequality_proof_l62_62977


namespace y_completes_work_in_seventy_days_l62_62705

def work_days (mahesh_days : ℕ) (mahesh_work_days : ℕ) (rajesh_days : ℕ) (y_days : ℕ) : Prop :=
  let mahesh_rate := (1:ℝ) / mahesh_days
  let rajesh_rate := (1:ℝ) / rajesh_days
  let work_done_by_mahesh := mahesh_rate * mahesh_work_days
  let remaining_work := (1:ℝ) - work_done_by_mahesh
  let rajesh_remaining_work_days := remaining_work / rajesh_rate
  let y_rate := (1:ℝ) / y_days
  y_rate = rajesh_rate

theorem y_completes_work_in_seventy_days :
  work_days 35 20 30 70 :=
by
  -- This is where the proof would go
  sorry

end y_completes_work_in_seventy_days_l62_62705


namespace valve_rate_difference_l62_62551

section ValveRates

-- Conditions
variables (V1 V2 : ℝ) (t1 t2 : ℝ) (C : ℝ)
-- Given Conditions
-- The first valve alone would fill the pool in 2 hours (120 minutes)
def valve1_rate := V1 = 12000 / 120
-- With both valves open, the pool will be filled with water in 48 minutes
def combined_rate := V1 + V2 = 12000 / 48
-- Capacity of the pool is 12000 cubic meters
def pool_capacity := C = 12000

-- The Proof of the question
theorem valve_rate_difference : V1 = 100 → V2 = 150 → (V2 - V1) = 50 :=
by
  intros hV1 hV2
  rw [hV1, hV2]
  norm_num

end ValveRates

end valve_rate_difference_l62_62551


namespace min_max_value_sum_l62_62800

variable (a b c d e : ℝ)

theorem min_max_value_sum :
  a + b + c + d + e = 10 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 30 →
  let expr := 5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)
  let m := 42
  let M := 52
  m + M = 94 := sorry

end min_max_value_sum_l62_62800


namespace probability_different_topics_l62_62586

theorem probability_different_topics (topics : ℕ) (choices : Finset ℕ) (A B : choices) 
(h_topic_count : topics = 6)
(h_totals : choices.card = topics) :
  (probability A B choosing_different := (choices.card - 1) * choices.card = 30) → 
  (total_possible_outcomes := choices.card * choices.card = 36) →
  (probability_different := 30 / 36 = 5 / 6) :=
sorry

end probability_different_topics_l62_62586


namespace b_should_pay_l62_62368

-- Definitions for the number of horses and their duration in months
def horses_of_a := 12
def months_of_a := 8

def horses_of_b := 16
def months_of_b := 9

def horses_of_c := 18
def months_of_c := 6

-- Total rent
def total_rent := 870

-- Shares in horse-months for each person
def share_of_a := horses_of_a * months_of_a
def share_of_b := horses_of_b * months_of_b
def share_of_c := horses_of_c * months_of_c

-- Total share in horse-months
def total_share := share_of_a + share_of_b + share_of_c

-- Fraction for b
def fraction_for_b := share_of_b / total_share

-- Amount b should pay
def amount_for_b := total_rent * fraction_for_b

-- Theorem to verify the amount b should pay
theorem b_should_pay : amount_for_b = 360 := by
  -- The steps of the proof would go here
  sorry

end b_should_pay_l62_62368


namespace problem_statement_l62_62750

theorem problem_statement (x : ℝ) (h₀ : x > 0) (n : ℕ) (hn : n > 0) :
  (x + (n^n : ℝ) / x^n) ≥ (n + 1) :=
sorry

end problem_statement_l62_62750


namespace solve_for_a_l62_62972

noncomputable def parabola (a b c : ℚ) (x : ℚ) := a * x^2 + b * x + c

theorem solve_for_a (a b c : ℚ) (h1 : parabola a b c 2 = 5) (h2 : parabola a b c 1 = 2) : 
  a = -3 :=
by
  -- Given: y = ax^2 + bx + c with vertex (2,5) and point (1,2)
  have eq1 : a * (2:ℚ)^2 + b * (2:ℚ) + c = 5 := h1
  have eq2 : a * (1:ℚ)^2 + b * (1:ℚ) + c = 2 := h2

  -- Combine information to find a
  sorry

end solve_for_a_l62_62972


namespace sum_PS_TV_l62_62480

theorem sum_PS_TV 
  (P V : ℝ) 
  (hP : P = 3) 
  (hV : V = 33)
  (n : ℕ) 
  (hn : n = 6) 
  (Q R S T U : ℝ) 
  (hPR : P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U < V)
  (h_divide : ∀ i : ℕ, i ≤ n → P + i * (V - P) / n = P + i * 5) :
  (P, V, Q, R, S, T, U) = (3, 33, 8, 13, 18, 23, 28) → (S - P) + (V - T) = 25 :=
by {
  sorry
}

end sum_PS_TV_l62_62480


namespace dinner_cost_per_kid_l62_62375

theorem dinner_cost_per_kid
  (row_ears : ℕ)
  (seeds_bag : ℕ)
  (seeds_ear : ℕ)
  (pay_row : ℝ)
  (bags_used : ℕ)
  (dinner_fraction : ℝ)
  (h1 : row_ears = 70)
  (h2 : seeds_bag = 48)
  (h3 : seeds_ear = 2)
  (h4 : pay_row = 1.5)
  (h5 : bags_used = 140)
  (h6 : dinner_fraction = 0.5) :
  ∃ (dinner_cost : ℝ), dinner_cost = 36 :=
by
  sorry

end dinner_cost_per_kid_l62_62375


namespace unique_2_digit_cyclic_permutation_divisible_l62_62160

def is_cyclic_permutation (n : ℕ) (M : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → M i = M j

def M (a : Fin 2 → ℕ) : ℕ := a 0 * 10 + a 1

theorem unique_2_digit_cyclic_permutation_divisible (a : Fin 2 → ℕ) (h0 : ∀ i, a i ≠ 0) :
  (M a) % (a 1 * 10 + a 0) = 0 → 
  (M a = 11) :=
by
  sorry

end unique_2_digit_cyclic_permutation_divisible_l62_62160


namespace crocodile_collection_l62_62574

noncomputable def expected_number_of_canes (n : ℕ) (p : ℝ) : ℝ :=
  1 + (Finset.range (n - 3 + 1)).sum (λ k, 1.0 / (k + 1))

theorem crocodile_collection (n : ℕ) (p : ℝ) (h_n : n = 10) (h_p : p = 0.1) :
  expected_number_of_canes n p = 3.59 :=
by
  rw [h_n, h_p]
  sorry

end crocodile_collection_l62_62574


namespace prob_KH_then_Ace_l62_62850

noncomputable def probability_KH_then_Ace_drawn_in_sequence : ℚ :=
  let prob_first_card_is_KH := 1 / 52
  let prob_second_card_is_Ace := 4 / 51
  prob_first_card_is_KH * prob_second_card_is_Ace

theorem prob_KH_then_Ace : probability_KH_then_Ace_drawn_in_sequence = 1 / 663 := by
  sorry

end prob_KH_then_Ace_l62_62850


namespace maria_total_cost_l62_62665

-- Define the costs of the items
def pencil_cost : ℕ := 8
def pen_cost : ℕ := pencil_cost / 2
def eraser_cost : ℕ := 2 * pen_cost

-- Define the total cost
def total_cost : ℕ := pen_cost + pencil_cost + eraser_cost

-- The theorem to prove
theorem maria_total_cost : total_cost = 20 := by
  sorry

end maria_total_cost_l62_62665


namespace minimize_g_function_l62_62264

noncomputable def g (x : ℝ) : ℝ := (9 * x^2 + 18 * x + 29) / (8 * (2 + x))

theorem minimize_g_function : ∀ x : ℝ, x ≥ -1 → g x = 29 / 8 :=
sorry

end minimize_g_function_l62_62264


namespace sum_of_two_digit_numbers_with_gcd_lcm_l62_62973

theorem sum_of_two_digit_numbers_with_gcd_lcm (x y : ℕ) (h1 : Nat.gcd x y = 8) (h2 : Nat.lcm x y = 96)
  (h3 : 10 ≤ x ∧ x < 100) (h4 : 10 ≤ y ∧ y < 100) : x + y = 56 :=
sorry

end sum_of_two_digit_numbers_with_gcd_lcm_l62_62973


namespace inequality_iff_l62_62745

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end inequality_iff_l62_62745


namespace simplify_expression_l62_62131

variable (a b : ℝ)

theorem simplify_expression : -3 * a * (2 * a - 4 * b + 2) + 6 * a = -6 * a ^ 2 + 12 * a * b := by
  sorry

end simplify_expression_l62_62131


namespace hyperbola_asymptote_slope_l62_62448

theorem hyperbola_asymptote_slope (m : ℝ) :
  (∀ x y : ℝ, mx^2 + y^2 = 1) →
  (∀ x y : ℝ, y = 2 * x) →
  m = -4 :=
by
  sorry

end hyperbola_asymptote_slope_l62_62448


namespace part1_and_part2_l62_62276

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l62_62276


namespace find_f_at_2_l62_62628

def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

theorem find_f_at_2 (a b c : ℝ) (h : f (-2) a b c = 10) : f 2 a b c = -26 :=
by
  sorry

end find_f_at_2_l62_62628


namespace johns_share_is_1100_l62_62864

def total_amount : ℕ := 6600
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6
def total_parts : ℕ := ratio_john + ratio_jose + ratio_binoy
def value_per_part : ℚ := total_amount / total_parts
def amount_received_by_john : ℚ := value_per_part * ratio_john

theorem johns_share_is_1100 : amount_received_by_john = 1100 := by
  sorry

end johns_share_is_1100_l62_62864


namespace no_solution_iff_n_eq_minus_half_l62_62170

theorem no_solution_iff_n_eq_minus_half (n x y z : ℝ) :
  (¬∃ x y z : ℝ, 2 * n * x + y = 2 ∧ n * y + z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1 / 2 :=
by
  sorry

end no_solution_iff_n_eq_minus_half_l62_62170


namespace sandwiches_lunch_monday_l62_62668

-- Define the conditions
variables (L : ℕ) 
variables (sandwiches_monday sandwiches_tuesday : ℕ)
variables (h1 : sandwiches_monday = L + 2 * L)
variables (h2 : sandwiches_tuesday = 1)

-- Define the fact that he ate 8 more sandwiches on Monday compared to Tuesday.
variables (h3 : sandwiches_monday = sandwiches_tuesday + 8)

theorem sandwiches_lunch_monday : L = 3 := 
by
  -- We need to prove L = 3 given the conditions (h1, h2, h3)
  -- Here is where the necessary proof would be constructed
  -- This placeholder indicates a proof needs to be inserted here
  sorry

end sandwiches_lunch_monday_l62_62668


namespace next_perfect_cube_l62_62458

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) : 
  ∃ m : ℕ, m^3 = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
by
  sorry

end next_perfect_cube_l62_62458


namespace hexagonal_pyramid_ratio_l62_62778

noncomputable def ratio_of_radii {R r : ℝ} (h1 : 2 * r + R) (h2 : R = r * (1 + (Real.sqrt 21)/3)) : ℝ :=
R / r

theorem hexagonal_pyramid_ratio :
  ∀ (R r : ℝ), 
  (h : 2 * r + R = h) ∧ (r > 0) ∧ (R > 0) ∧ (R = r * (1 + (Real.sqrt 21)/3)) →
  ratio_of_radii := sorry

end hexagonal_pyramid_ratio_l62_62778


namespace a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l62_62748

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l62_62748


namespace part1_and_part2_l62_62275

-- Define the arithmetic sequence {a_n}
def a (n : Nat) : Nat := 2 * n + 3

-- Define the sequence {b_n}
def b (n : Nat) : Nat :=
  if n % 2 = 0 then 4 * n + 6 else 2 * n - 3

-- Define the sum of the first n terms of a sequence
def summation (seq : Nat → Nat) (n : Nat) : Nat :=
  (List.range n).map seq |>.sum

-- Define S_n as the sum of the first n terms of {a_n}
def S (n : Nat) : Nat := summation a n

-- Define T_n as the sum of the first n terms of {b_n}
def T (n : Nat) : Nat := summation b n

-- Given conditions
axiom S4_eq_32 : S 4 = 32
axiom T3_eq_16 : T 3 = 16

-- Prove the general formula for {a_n} and that T_n > S_n for n > 5
theorem part1_and_part2 (n : Nat) (h : n > 5) : a n = 2 * n + 3 ∧ T n > S n :=
  by
  sorry

end part1_and_part2_l62_62275


namespace gcd_90_150_l62_62422

theorem gcd_90_150 : Int.gcd 90 150 = 30 := 
by sorry

end gcd_90_150_l62_62422


namespace mark_more_hours_l62_62199

-- Definitions based on the conditions
variables (Pat Kate Mark Alex : ℝ)
variables (total_hours : ℝ)
variables (h1 : Pat + Kate + Mark + Alex = 350)
variables (h2 : Pat = 2 * Kate)
variables (h3 : Pat = (1 / 3) * Mark)
variables (h4 : Alex = 1.5 * Kate)

-- Theorem statement with the desired proof target
theorem mark_more_hours (Pat Kate Mark Alex : ℝ) (h1 : Pat + Kate + Mark + Alex = 350) 
(h2 : Pat = 2 * Kate) (h3 : Pat = (1 / 3) * Mark) (h4 : Alex = 1.5 * Kate) : 
Mark - (Kate + Alex) = 116.66666666666667 := sorry

end mark_more_hours_l62_62199


namespace find_smaller_integer_l62_62347

theorem find_smaller_integer
  (x y : ℤ)
  (h1 : x + y = 30)
  (h2 : 2 * y = 5 * x - 10) :
  x = 10 :=
by
  -- proof would go here
  sorry

end find_smaller_integer_l62_62347


namespace sufficient_condition_not_necessary_condition_l62_62429

variable (a b : ℝ)

theorem sufficient_condition (hab : (a - b) * a^2 < 0) : a < b :=
by
  sorry

theorem not_necessary_condition (h : a < b) : (a - b) * a^2 < 0 ∨ (a - b) * a^2 = 0 :=
by
  sorry

end sufficient_condition_not_necessary_condition_l62_62429


namespace prob_different_topics_l62_62579

theorem prob_different_topics (T : ℕ) (hT : T = 6) :
  let total_outcomes := T * T,
      favorable_outcomes := T * (T - 1),
      probability_different := favorable_outcomes / total_outcomes
  in probability_different = 5 / 6 :=
by
  have : total_outcomes = 36 := by rw [hT]; norm_num
  have : favorable_outcomes = 30 := by rw [hT]; norm_num
  have : probability_different = 5 / 6 := by norm_num
  sorry

end prob_different_topics_l62_62579


namespace system_of_equations_solution_l62_62209

theorem system_of_equations_solution (x y : ℝ) (h1 : |y - x| - (|x| / x) + 1 = 0) (h2 : |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0) (hx : x ≠ 0) :
  (0 < x ∧ x ≤ 0.5 ∧ y = x) :=
by
  sorry

end system_of_equations_solution_l62_62209


namespace common_difference_is_1_over_10_l62_62904

open Real

noncomputable def a_n (a₁ d: ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

noncomputable def S_n (a₁ d : ℝ) (n : ℕ) : ℝ := 
  n * a₁ + (n * (n - 1)) * d / 2

theorem common_difference_is_1_over_10 (a₁ d : ℝ) 
  (h : (S_n a₁ d 2017 / 2017) - (S_n a₁ d 17 / 17) = 100) : 
  d = 1 / 10 :=
by
  sorry

end common_difference_is_1_over_10_l62_62904


namespace soldiers_in_groups_l62_62089

theorem soldiers_in_groups (x : ℕ) (h1 : x % 2 = 1) (h2 : x % 3 = 2) (h3 : x % 5 = 3) : x % 30 = 23 :=
by
  sorry

end soldiers_in_groups_l62_62089


namespace concert_cost_l62_62041

def ticket_cost : ℕ := 50
def number_of_people : ℕ := 2
def processing_fee_rate : ℝ := 0.15
def parking_fee : ℕ := 10
def per_person_entrance_fee : ℕ := 5

def total_cost : ℝ :=
  let tickets := (ticket_cost * number_of_people : ℕ)
  let processing_fee := tickets * processing_fee_rate
  let entrance_fee := per_person_entrance_fee * number_of_people
  (tickets : ℝ) + processing_fee + (parking_fee : ℝ) + (entrance_fee : ℝ)

theorem concert_cost :
  total_cost = 135 := by
  sorry

end concert_cost_l62_62041


namespace simplify_fraction_l62_62406

theorem simplify_fraction : (81000 ^ 3) / (27000 ^ 3) = 27 := by
  sorry

end simplify_fraction_l62_62406


namespace even_number_less_than_its_square_l62_62642

theorem even_number_less_than_its_square (m : ℕ) (h1 : 2 ∣ m) (h2 : m > 1) : m < m^2 :=
by
sorry

end even_number_less_than_its_square_l62_62642


namespace expand_expression_l62_62414

variable (y : ℝ)

theorem expand_expression : 5 * (6 * y^2 - 3 * y + 2) = 30 * y^2 - 15 * y + 10 := by
  sorry

end expand_expression_l62_62414


namespace find_m_value_l62_62152

def vectors_parallel (a1 a2 b1 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

theorem find_m_value (m : ℝ) :
  let a := (6, 3)
  let b := (m, 2)
  vectors_parallel a.1 a.2 b.1 b.2 ↔ m = 4 :=
by
  intro H
  obtain ⟨_, _⟩ := H
  sorry

end find_m_value_l62_62152


namespace derivative_at_1_l62_62758

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem derivative_at_1 : deriv f 1 = 1 + Real.cos 1 :=
by
  sorry

end derivative_at_1_l62_62758


namespace trig_identity_l62_62269

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + Real.sin (-2 * π / 3 + x) ^ 2 = 1 / 4 :=
by
  sorry

end trig_identity_l62_62269


namespace g_sum_1_2_3_2_l62_62798

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then
    (a * b + a - 3) / (3 * a)
  else
    (a * b + b + 3) / (-3 * b)

theorem g_sum_1_2_3_2 : g 1 2 + g 3 2 = -11 / 6 :=
by sorry

end g_sum_1_2_3_2_l62_62798


namespace parabola_translation_l62_62479

theorem parabola_translation :
  ∀ (x y : ℝ), y = 3 * x^2 →
  ∃ (new_x new_y : ℝ), new_y = 3 * (new_x + 3)^2 - 3 :=
by {
  sorry
}

end parabola_translation_l62_62479


namespace instrument_accuracy_confidence_l62_62566

noncomputable def instrument_accuracy (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ) : ℝ × ℝ :=
  let lower := s * (1 - q)
  let upper := s * (1 + q)
  (lower, upper)

theorem instrument_accuracy_confidence :
  ∀ (n : ℕ) (s : ℝ) (gamma : ℝ) (q : ℝ),
    n = 12 →
    s = 0.6 →
    gamma = 0.99 →
    q = 0.9 →
    0.06 < (instrument_accuracy n s gamma q).fst ∧
    (instrument_accuracy n s gamma q).snd < 1.14 :=
by
  intros n s gamma q h_n h_s h_gamma h_q
  -- proof would go here
  sorry

end instrument_accuracy_confidence_l62_62566


namespace maddie_watched_138_on_monday_l62_62200

-- Define the constants and variables from the problem statement
def total_episodes : ℕ := 8
def minutes_per_episode : ℕ := 44
def watched_thursday : ℕ := 21
def watched_friday_episodes : ℕ := 2
def watched_weekend : ℕ := 105

-- Calculate the total minutes watched from all episodes
def total_minutes : ℕ := total_episodes * minutes_per_episode

-- Calculate the minutes watched on Friday
def watched_friday : ℕ := watched_friday_episodes * minutes_per_episode

-- Calculate the total minutes watched on weekdays excluding Monday
def watched_other_days : ℕ := watched_thursday + watched_friday + watched_weekend

-- Statement to prove that Maddie watched 138 minutes on Monday
def minutes_watched_on_monday : ℕ := total_minutes - watched_other_days

-- The final statement for proof in Lean 4
theorem maddie_watched_138_on_monday : minutes_watched_on_monday = 138 := by
  -- This theorem should be proved using the above definitions and calculations, proof skipped with sorry
  sorry

end maddie_watched_138_on_monday_l62_62200


namespace math_problem_example_l62_62495

theorem math_problem_example (m n : ℤ) (h0 : m > 0) (h1 : n > 0)
    (h2 : 3 * m + 2 * n = 225) (h3 : Int.gcd m n = 15) : m + n = 105 :=
sorry

end math_problem_example_l62_62495


namespace product_of_coordinates_of_D_l62_62743

theorem product_of_coordinates_of_D (Mx My Cx Cy Dx Dy : ℝ) (M : (Mx, My) = (4, 8)) (C : (Cx, Cy) = (5, 4)) 
  (midpoint : (Mx, My) = ((Cx + Dx) / 2, (Cy + Dy) / 2)) : (Dx * Dy) = 36 := 
by
  sorry

end product_of_coordinates_of_D_l62_62743


namespace determine_k_l62_62263

theorem determine_k (k : ℕ) : 2^2004 - 2^2003 - 2^2002 + 2^2001 = k * 2^2001 → k = 3 :=
by
  intro h
  -- now we would proceed to prove it, but we'll skip proof here
  sorry

end determine_k_l62_62263


namespace range_of_a_l62_62445

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > a then x + 2 else x^2 + 5 * x + 2

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
f x a - 2 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g x a = 0 → (x = 2 ∨ x = -1 ∨ x = -2)) ↔ (-1 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l62_62445


namespace solve_fractional_equation_l62_62958

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x^2 + 4 * x - 4) = (3 * x) / (3 * x - 2) ↔ x = 1 / 3 ∨ x = -2 := by
  sorry

end solve_fractional_equation_l62_62958


namespace inversely_proportional_ratios_l62_62337

theorem inversely_proportional_ratios (x y x₁ x₂ y₁ y₂ : ℝ) (hx_inv : ∀ x y, x * y = 1)
  (hx_ratio : x₁ / x₂ = 3 / 5) :
  y₁ / y₂ = 5 / 3 :=
sorry

end inversely_proportional_ratios_l62_62337


namespace egg_laying_hens_l62_62505

theorem egg_laying_hens (total_chickens : ℕ) (num_roosters : ℕ) (non_egg_laying_hens : ℕ)
  (h1 : total_chickens = 325)
  (h2 : num_roosters = 28)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - num_roosters - non_egg_laying_hens = 277 :=
by sorry

end egg_laying_hens_l62_62505


namespace prove_percentage_cats_adopted_each_month_l62_62491

noncomputable def percentage_cats_adopted_each_month
    (initial_dogs : ℕ)
    (initial_cats : ℕ)
    (initial_lizards : ℕ)
    (adopted_dogs_percent : ℕ)
    (adopted_lizards_percent : ℕ)
    (new_pets_each_month : ℕ)
    (total_pets_after_month : ℕ)
    (adopted_cats_percent : ℕ) : Prop :=
  initial_dogs = 30 ∧
  initial_cats = 28 ∧
  initial_lizards = 20 ∧
  adopted_dogs_percent = 50 ∧
  adopted_lizards_percent = 20 ∧
  new_pets_each_month = 13 ∧
  total_pets_after_month = 65 →
  adopted_cats_percent = 25

-- The condition to prove
theorem prove_percentage_cats_adopted_each_month :
  percentage_cats_adopted_each_month 30 28 20 50 20 13 65 25 :=
by 
  sorry

end prove_percentage_cats_adopted_each_month_l62_62491


namespace system_of_equations_solution_l62_62516

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x - 2 * y = 1)
  (h2 : 3 * x + 4 * y = 23) :
  x = 5 ∧ y = 2 :=
sorry

end system_of_equations_solution_l62_62516


namespace heads_not_consecutive_probability_l62_62353

theorem heads_not_consecutive_probability :
  (∃ n m : ℕ, n = 2^4 ∧ m = 1 + Nat.choose 4 1 + Nat.choose 3 2 ∧ (m / n : ℚ) = 1 / 2) :=
by
  use 16     -- n
  use 8      -- m
  sorry

end heads_not_consecutive_probability_l62_62353


namespace stratified_sampling_l62_62242

theorem stratified_sampling 
  (students_first_grade : ℕ)
  (students_second_grade : ℕ)
  (selected_first_grade : ℕ)
  (x : ℕ)
  (h1 : students_first_grade = 400)
  (h2 : students_second_grade = 360)
  (h3 : selected_first_grade = 60)
  (h4 : (selected_first_grade / students_first_grade : ℚ) = (x / students_second_grade : ℚ)) :
  x = 54 :=
sorry

end stratified_sampling_l62_62242


namespace sandy_red_marbles_l62_62489

theorem sandy_red_marbles (jessica_marbles : ℕ) (sandy_marbles : ℕ) 
  (h₀ : jessica_marbles = 3 * 12)
  (h₁ : sandy_marbles = 4 * jessica_marbles) : 
  sandy_marbles = 144 :=
by
  sorry

end sandy_red_marbles_l62_62489


namespace AgathaAdditionalAccessories_l62_62725

def AgathaBudget : ℕ := 250
def Frame : ℕ := 85
def FrontWheel : ℕ := 35
def RearWheel : ℕ := 40
def Seat : ℕ := 25
def HandlebarTape : ℕ := 15
def WaterBottleCage : ℕ := 10
def BikeLock : ℕ := 20
def FutureExpenses : ℕ := 10

theorem AgathaAdditionalAccessories :
  AgathaBudget - (Frame + FrontWheel + RearWheel + Seat + HandlebarTape + WaterBottleCage + BikeLock + FutureExpenses) = 10 := by
  sorry

end AgathaAdditionalAccessories_l62_62725


namespace no_maximum_y_coordinate_for_hyperbola_l62_62610

theorem no_maximum_y_coordinate_for_hyperbola :
  ∀ y : ℝ, ∃ x : ℝ, y = 3 + (3 / 5) * x :=
by
  sorry

end no_maximum_y_coordinate_for_hyperbola_l62_62610


namespace river_flow_speed_l62_62779

theorem river_flow_speed (v : ℝ) :
  (6 - v ≠ 0) ∧ (6 + v ≠ 0) ∧ ((48 / (6 - v)) + (48 / (6 + v)) = 18) → v = 2 := 
by
  sorry

end river_flow_speed_l62_62779


namespace sandy_red_marbles_l62_62488

theorem sandy_red_marbles (jessica_marbles : ℕ) (sandy_marbles : ℕ) 
  (h₀ : jessica_marbles = 3 * 12)
  (h₁ : sandy_marbles = 4 * jessica_marbles) : 
  sandy_marbles = 144 :=
by
  sorry

end sandy_red_marbles_l62_62488


namespace total_cost_correct_l62_62882

noncomputable def cost_pencils (price : ℝ) (quantity : ℕ) (discount : ℝ) : ℝ :=
  if quantity > 15 then (price * quantity * (1 - discount)) else (price * quantity)

noncomputable def cost_folders (price : ℝ) (quantity : ℕ) (discount : ℝ) : ℝ :=
  if quantity > 10 then (price * quantity * (1 - discount)) else (price * quantity)

noncomputable def cost_notebooks (price : ℝ) (quantity : ℕ) : ℝ :=
  let paid_quantity := (quantity / 3) * 2 + (quantity % 3)
  in price * paid_quantity

noncomputable def cost_staplers (price : ℝ) (quantity : ℕ) : ℝ :=
  price * quantity

theorem total_cost_correct :
  let pencils_cost := cost_pencils 0.5 24 0.1 in
  let folders_cost := cost_folders 0.9 20 0.15 in
  let notebooks_cost := cost_notebooks 1.2 15 in
  let staplers_cost := cost_staplers 2.5 10 in
  pencils_cost + folders_cost + notebooks_cost + staplers_cost = 63.1 :=
by {
  sorry
}

end total_cost_correct_l62_62882


namespace candy_pebbles_l62_62887

theorem candy_pebbles (C L : ℕ) 
  (h1 : L = 3 * C)
  (h2 : L = C + 8) :
  C = 4 :=
by
  sorry

end candy_pebbles_l62_62887


namespace terminal_side_quadrant_l62_62216

theorem terminal_side_quadrant (α : ℝ) (h : α = 2) : 
  90 < α * (180 / Real.pi) ∧ α * (180 / Real.pi) < 180 := 
by
  sorry

end terminal_side_quadrant_l62_62216


namespace solve_furniture_factory_l62_62626

variable (num_workers : ℕ) (tables_per_worker : ℕ) (legs_per_worker : ℕ) 
variable (tabletop_workers legs_workers : ℕ)

axiom worker_capacity : tables_per_worker = 3 ∧ legs_per_worker = 6
axiom total_workers : num_workers = 60
axiom table_leg_ratio : ∀ (x : ℕ), tabletop_workers = x → legs_workers = (num_workers - x)
axiom daily_production_eq : ∀ (x : ℕ), (4 * tables_per_worker * x = 6 * legs_per_worker * (num_workers - x))

theorem solve_furniture_factory : 
  ∃ (x y : ℕ), num_workers = x + y ∧ 
            4 * 3 * x = 6 * (num_workers - x) ∧ 
            x = 20 ∧ y = (num_workers - 20) := by
  sorry

end solve_furniture_factory_l62_62626


namespace evaluate_expression_l62_62413

theorem evaluate_expression : 1234562 - (12 * 3 * (2 + 7)) = 1234238 :=
by 
  sorry

end evaluate_expression_l62_62413


namespace range_of_a_l62_62155

-- Defining the function f(x) = x^2 + 2ax - 1
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x - 1

-- Conditions: x1, x2 ∈ [1, +∞) and x1 < x2
variables (x1 x2 a : ℝ)
variables (h1 : 1 ≤ x1) (h2 : 1 ≤ x2) (h3 : x1 < x2)

-- Statement of the proof problem:
theorem range_of_a (hf_ineq : x2 * f x1 a - x1 * f x2 a < a * (x1 - x2)) : a ≤ 2 :=
sorry

end range_of_a_l62_62155


namespace probability_different_topics_l62_62585

theorem probability_different_topics (topics : ℕ) (choices : Finset ℕ) (A B : choices) 
(h_topic_count : topics = 6)
(h_totals : choices.card = topics) :
  (probability A B choosing_different := (choices.card - 1) * choices.card = 30) → 
  (total_possible_outcomes := choices.card * choices.card = 36) →
  (probability_different := 30 / 36 = 5 / 6) :=
sorry

end probability_different_topics_l62_62585


namespace johnson_vincent_work_together_l62_62793

theorem johnson_vincent_work_together (work : Type) (time_johnson : ℕ) (time_vincent : ℕ) (combined_time : ℕ) :
  time_johnson = 10 → time_vincent = 40 → combined_time = 8 → 
  (1 / time_johnson + 1 / time_vincent) = 1 / combined_time :=
by
  intros h_johnson h_vincent h_combined
  sorry

end johnson_vincent_work_together_l62_62793


namespace probability_correct_l62_62589

-- Definition for the total number of ways to select topics
def total_ways : ℕ := 6 * 6

-- Definition for the number of ways two students select different topics
def different_topics_ways : ℕ := 6 * 5

-- Definition for the probability of selecting different topics
def probability_different_topics : ℚ := different_topics_ways / total_ways

-- The statement to be proved in Lean
theorem probability_correct :
  probability_different_topics = 5 / 6 := 
sorry

end probability_correct_l62_62589


namespace determine_BD_l62_62478

def quadrilateral (AB BC CD DA BD : ℕ) : Prop :=
AB = 6 ∧ BC = 15 ∧ CD = 8 ∧ DA = 12 ∧ (7 < BD ∧ BD < 18)

theorem determine_BD : ∃ BD : ℕ, quadrilateral 6 15 8 12 BD ∧ 8 ≤ BD ∧ BD ≤ 17 :=
by
  sorry

end determine_BD_l62_62478


namespace a2022_value_l62_62688

theorem a2022_value 
  (a : Fin 2022 → ℤ)
  (h : ∀ n k : Fin 2022, a n - a k ≥ n.1^3 - k.1^3)
  (a1011 : a 1010 = 0) :
  a 2021 = 2022^3 - 1011^3 :=
by
  sorry

end a2022_value_l62_62688


namespace valve_rate_difference_l62_62550

section ValveRates

-- Conditions
variables (V1 V2 : ℝ) (t1 t2 : ℝ) (C : ℝ)
-- Given Conditions
-- The first valve alone would fill the pool in 2 hours (120 minutes)
def valve1_rate := V1 = 12000 / 120
-- With both valves open, the pool will be filled with water in 48 minutes
def combined_rate := V1 + V2 = 12000 / 48
-- Capacity of the pool is 12000 cubic meters
def pool_capacity := C = 12000

-- The Proof of the question
theorem valve_rate_difference : V1 = 100 → V2 = 150 → (V2 - V1) = 50 :=
by
  intros hV1 hV2
  rw [hV1, hV2]
  norm_num

end ValveRates

end valve_rate_difference_l62_62550


namespace vladimir_can_invest_more_profitably_l62_62093

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end vladimir_can_invest_more_profitably_l62_62093


namespace q_implies_not_p_l62_62319

-- Define the conditions p and q
def p (x : ℝ) := x < -1
def q (x : ℝ) := x^2 - x - 2 > 0

-- Prove that q implies ¬p
theorem q_implies_not_p (x : ℝ) : q x → ¬ p x := by
  intros hq hp
  -- Provide the steps of logic here
  sorry

end q_implies_not_p_l62_62319


namespace max_area_rect_l62_62067

theorem max_area_rect (x y : ℝ) (h_perimeter : 2 * x + 2 * y = 40) : 
  x * y ≤ 100 :=
by
  sorry

end max_area_rect_l62_62067


namespace median_and_mode_of_successful_shots_l62_62008

theorem median_and_mode_of_successful_shots :
  let shots := [3, 6, 4, 6, 4, 3, 6, 5, 7]
  let sorted_shots := [3, 3, 4, 4, 5, 6, 6, 6, 7]
  let median := sorted_shots[4]  -- 4 is the index for the 5th element (0-based indexing)
  let mode := 6  -- determined by the number that appears most frequently
  median = 5 ∧ mode = 6 :=
by
  sorry

end median_and_mode_of_successful_shots_l62_62008


namespace change_in_responses_max_min_diff_l62_62127

open Classical

theorem change_in_responses_max_min_diff :
  let initial_yes := 40
  let initial_no := 40
  let initial_undecided := 20
  let end_yes := 60
  let end_no := 30
  let end_undecided := 10
  let min_change := 20
  let max_change := 80
  max_change - min_change = 60 := by
  intros; sorry

end change_in_responses_max_min_diff_l62_62127


namespace jason_attended_games_l62_62191

-- Define the conditions as given in the problem
def games_planned_this_month : ℕ := 11
def games_planned_last_month : ℕ := 17
def games_missed : ℕ := 16

-- Define the total number of games planned
def games_planned_total : ℕ := games_planned_this_month + games_planned_last_month

-- Define the number of games attended
def games_attended : ℕ := games_planned_total - games_missed

-- Prove that Jason attended 12 games
theorem jason_attended_games : games_attended = 12 := by
  -- The proof is omitted, but the theorem statement is required
  sorry

end jason_attended_games_l62_62191


namespace initial_boys_l62_62980

-- Define the initial conditions
def initial_girls := 4
def final_children := 8
def boys_left := 3
def girls_entered := 2

-- Define the statement to be proved
theorem initial_boys : 
  ∃ (B : ℕ), (B - boys_left) + (initial_girls + girls_entered) = final_children ∧ B = 5 :=
by
  -- Placeholder for the proof
  sorry

end initial_boys_l62_62980


namespace find_halls_per_floor_l62_62027

theorem find_halls_per_floor
  (H : ℤ)
  (floors_first_wing : ℤ := 9)
  (rooms_per_hall_first_wing : ℤ := 32)
  (floors_second_wing : ℤ := 7)
  (halls_per_floor_second_wing : ℤ := 9)
  (rooms_per_hall_second_wing : ℤ := 40)
  (total_rooms : ℤ := 4248) :
  9 * H * 32 + 7 * 9 * 40 = 4248 → H = 6 :=
by
  sorry

end find_halls_per_floor_l62_62027


namespace bouquet_cost_l62_62125

theorem bouquet_cost (c : ℕ) : (c / 25 = 30 / 15) → c = 50 := by
  sorry

end bouquet_cost_l62_62125


namespace geometric_sequence_15th_term_l62_62614

theorem geometric_sequence_15th_term :
  let a_1 := 27
  let r := (1 : ℚ) / 6
  let a_15 := a_1 * r ^ 14
  a_15 = 1 / 14155776 := by
  sorry

end geometric_sequence_15th_term_l62_62614


namespace y_increase_by_18_when_x_increases_by_12_l62_62771

theorem y_increase_by_18_when_x_increases_by_12
  (h_slope : ∀ x y: ℝ, (4 * y = 6 * x) ↔ (3 * y = 2 * x)) :
  ∀ Δx : ℝ, Δx = 12 → ∃ Δy : ℝ, Δy = 18 :=
by
  sorry

end y_increase_by_18_when_x_increases_by_12_l62_62771


namespace polygon_side_count_l62_62001

theorem polygon_side_count (n : ℕ) (h : n - 3 ≤ 5) : n = 8 :=
by {
  sorry
}

end polygon_side_count_l62_62001


namespace min_value_expression_l62_62751

theorem min_value_expression (x1 x2 x3 x4 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1) ^ 2 + 1 / (Real.sin x1) ^ 2) *
  (2 * (Real.sin x2) ^ 2 + 1 / (Real.sin x2) ^ 2) *
  (2 * (Real.sin x3) ^ 2 + 1 / (Real.sin x3) ^ 2) *
  (2 * (Real.sin x4) ^ 2 + 1 / (Real.sin x4) ^ 2) = 81 := 
sorry

end min_value_expression_l62_62751


namespace seashells_given_to_brothers_l62_62119

theorem seashells_given_to_brothers :
  ∃ B : ℕ, 180 - 40 - B = 2 * 55 ∧ B = 30 := by
  sorry

end seashells_given_to_brothers_l62_62119


namespace real_life_distance_between_cities_l62_62825

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end real_life_distance_between_cities_l62_62825


namespace lattice_points_on_hyperbola_l62_62761

theorem lattice_points_on_hyperbola : 
  ∃ n, (∀ x y : ℤ, x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | 
  ∃ a b : ℤ, x = 2 * a + b ∧ y = 2 * a - b}) ∧ n = 250 := 
by {
  sorry
}

end lattice_points_on_hyperbola_l62_62761


namespace profitable_allocation_2015_l62_62091

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end profitable_allocation_2015_l62_62091


namespace lottery_probability_l62_62805

theorem lottery_probability (x_1 x_2 x_3 x_4 : ℝ) (p : ℝ) (h0 : 0 < p ∧ p < 1) : 
  x_1 = p * x_3 → 
  x_2 = p * x_4 + (1 - p) * x_1 → 
  x_3 = p + (1 - p) * x_2 → 
  x_4 = p + (1 - p) * x_3 → 
  x_2 = 0.19 :=
by
  sorry

end lottery_probability_l62_62805


namespace range_of_m_l62_62634

open Real

/-- The equation (sin x + cos x)^2 + cos(2x) = m has two roots x₁ and x₂ in the interval [0, π)
    with |x₁ - x₂| ≥ π / 4, and we need to prove that the range of m is [0, 2). -/
theorem range_of_m (m : ℝ)
  (h : ∃ (x₁ x₂ : ℝ), 0 ≤ x₁ ∧ x₁ < π ∧ 0 ≤ x₂ ∧ x₂ < π ∧ (sin x₁ + cos x₁)^2 + cos (2 * x₁) = m
    ∧ (sin x₂ + cos x₂)^2 + cos (2 * x₂) = m ∧ |x₁ - x₂| ≥ π / 4) :
  0 ≤ m ∧ m < 2 := sorry

end range_of_m_l62_62634


namespace proof_problem_l62_62957

noncomputable def sqrt_repeated (x : ℕ) (y : ℕ) : ℕ :=
Nat.sqrt x ^ y

theorem proof_problem (x y z : ℕ) :
  (sqrt_repeated x y = z) ↔ 
  ((∃ t : ℕ, x = t^2 ∧ y = 1 ∧ z = t) ∨ (x = 0 ∧ z = 0 ∧ y ≠ 0)) :=
sorry

end proof_problem_l62_62957


namespace cricket_team_average_age_l62_62088

open Real

-- Definitions based on the conditions given
def team_size := 11
def captain_age := 27
def wicket_keeper_age := 30
def remaining_players_size := team_size - 2

-- The mathematically equivalent proof problem in Lean statement
theorem cricket_team_average_age :
  ∃ A : ℝ,
    (A - 1) * remaining_players_size = (A * team_size) - (captain_age + wicket_keeper_age) ∧
    A = 24 :=
by
  sorry

end cricket_team_average_age_l62_62088


namespace roots_in_interval_l62_62120

theorem roots_in_interval (P : Polynomial ℝ) (h : ∀ i, P.coeff i = 1 ∨ P.coeff i = 0 ∨ P.coeff i = -1) : 
  ∀ x : ℝ, P.eval x = 0 → -2 ≤ x ∧ x ≤ 2 :=
by {
  -- Proof omitted
  sorry
}

end roots_in_interval_l62_62120


namespace cary_needs_6_weekends_l62_62402

variable (shoe_cost : ℕ)
variable (current_savings : ℕ)
variable (earn_per_lawn : ℕ)
variable (lawns_per_weekend : ℕ)
variable (w : ℕ)

theorem cary_needs_6_weekends
    (h1 : shoe_cost = 120)
    (h2 : current_savings = 30)
    (h3 : earn_per_lawn = 5)
    (h4 : lawns_per_weekend = 3)
    (h5 : w * (earn_per_lawn * lawns_per_weekend) = shoe_cost - current_savings) :
    w = 6 :=
by sorry

end cary_needs_6_weekends_l62_62402


namespace inequalities_hold_l62_62223

variable {a b c x y z : ℝ}

theorem inequalities_hold 
  (h1 : x ≤ a)
  (h2 : y ≤ b)
  (h3 : z ≤ c) :
  x * y + y * z + z * x ≤ a * b + b * c + c * a ∧
  x^2 + y^2 + z^2 ≤ a^2 + b^2 + c^2 ∧
  x * y * z ≤ a * b * c :=
sorry

end inequalities_hold_l62_62223


namespace median_and_mode_correct_l62_62007

noncomputable def data_set : List ℕ := [3, 6, 4, 6, 4, 3, 6, 5, 7]

def median (l : List ℕ) : ℕ :=
  let sorted := l.sorted
  sorted.nthLe (sorted.length / 2) sorry

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ (acc, freq) x =>
    if l.count x > freq then (x, l.count x)
    else acc) (0, 0)

theorem median_and_mode_correct : median data_set = 5 ∧ mode data_set = 6 :=
by
  sorry

end median_and_mode_correct_l62_62007


namespace general_formula_a_n_T_n_greater_S_n_l62_62282

-- Define the conditions for the problem
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ (n : ℕ), a n = a1 + (n - 1) * d

def b_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ (n : ℕ), b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_of_first_n_terms (a S : ℕ → ℕ) :=
  ∀ (n k : ℕ), k ≤ n → S n = (n * (a 1 + a n)) / 2

-- Given S4 = 32 and T3 = 16
def S_4_equals_32 (S : ℕ → ℕ) : Prop := S 4 = 32
def T_3_equals_16 (T : ℕ → ℕ) : Prop := T 3 = 16

-- Prove that the general formula for a_n is 2n + 3
theorem general_formula_a_n (a : ℕ → ℕ) (h : arithmetic_sequence a) : 
  a = λ n, 2 * n + 3 := 
sorry

-- Prove that for n > 5, T_n > S_n
theorem T_n_greater_S_n (a b S T: ℕ → ℕ) 
  (ha : arithmetic_sequence a) 
  (hb : b_sequence a b)
  (hS : sum_of_first_n_terms a S)
  (hT : sum_of_first_n_terms b T)
  (hS4 : S_4_equals_32 S)
  (hT3 : T_3_equals_16 T) 
  (n : ℕ) : n > 5 → T n > S n :=
sorry

end general_formula_a_n_T_n_greater_S_n_l62_62282


namespace determine_b_l62_62979

theorem determine_b (N a b c : ℤ) (h1 : a > 1 ∧ b > 1 ∧ c > 1) (h2 : N ≠ 1)
  (h3 : (N : ℝ) ^ (1 / a + 1 / (a * b) + 1 / (a * b * c) + 1 / (a * b * c ^ 2)) = N ^ (49 / 60)) :
  b = 4 :=
sorry

end determine_b_l62_62979


namespace man_older_than_son_l62_62376

theorem man_older_than_son (S M : ℕ) (h1 : S = 18) (h2 : M + 2 = 2 * (S + 2)) : M - S = 20 :=
by
  sorry

end man_older_than_son_l62_62376


namespace three_digit_number_is_112_l62_62117

theorem three_digit_number_is_112 (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 1 ≤ c ∧ c ≤ 9) (h4 : 100 * a + 10 * b + c = 56 * c) :
  100 * a + 10 * b + c = 112 :=
by sorry

end three_digit_number_is_112_l62_62117


namespace digital_earth_sustainable_development_l62_62254

theorem digital_earth_sustainable_development :
  (after_realization_digital_earth : Prop) → (scientists_can : Prop) :=
sorry

end digital_earth_sustainable_development_l62_62254


namespace circles_positional_relationship_l62_62638

theorem circles_positional_relationship
  (r1 r2 d : ℝ)
  (h1 : r1 = 1)
  (h2 : r2 = 5)
  (h3 : d = 3) :
  d < r2 - r1 := 
by
  sorry

end circles_positional_relationship_l62_62638


namespace magic_square_sum_l62_62311

-- Given conditions
def magic_square (S : ℕ) (a b c d e : ℕ) :=
  (30 + b + 27 = S) ∧
  (30 + 33 + a = S) ∧
  (33 + c + d = S) ∧
  (a + 18 + e = S) ∧
  (30 + c + e = S)

-- Prove that the sum a + d is 38 given the sums of the 3x3 magic square are equivalent
theorem magic_square_sum (a b c d e S : ℕ) (h : magic_square S a b c d e) : a + d = 38 :=
  sorry

end magic_square_sum_l62_62311


namespace cost_difference_l62_62022

theorem cost_difference (joy_pencils : ℕ) (colleen_pencils : ℕ) 
  (price_per_pencil_joy : ℝ) (price_per_pencil_colleen : ℝ) :
  joy_pencils = 30 →
  colleen_pencils = 50 →
  price_per_pencil_joy = 4 →
  price_per_pencil_colleen = 3.5 →
  (colleen_pencils * price_per_pencil_colleen - joy_pencils * price_per_pencil_joy) = 55 :=
by
  intros h_joy_pencils h_colleen_pencils h_price_joy h_price_colleen
  rw [h_joy_pencils, h_colleen_pencils, h_price_joy, h_price_colleen]
  norm_num
  repeat { sorry }

end cost_difference_l62_62022


namespace problem1_problem2_l62_62708

-- First problem
theorem problem1 (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := 
by sorry

-- Second problem
theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : ∃ k, a^x = k ∧ b^y = k ∧ c^z = k) (h_sum : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := 
by sorry

end problem1_problem2_l62_62708


namespace walter_exceptional_days_l62_62893

variable (b w : Nat)

-- Definitions of the conditions
def total_days (b w : Nat) : Prop := b + w = 10
def total_earnings (b w : Nat) : Prop := 3 * b + 6 * w = 42

-- The theorem states that given the conditions, the number of days Walter did his chores exceptionally well is 4
theorem walter_exceptional_days : total_days b w → total_earnings b w → w = 4 := 
  by
    sorry

end walter_exceptional_days_l62_62893


namespace prime_p_in_range_l62_62306

theorem prime_p_in_range (p : ℕ) (prime_p : Nat.Prime p) 
    (h : ∃ a b : ℤ, a * b = -530 * p ∧ a + b = p) : 43 < p ∧ p ≤ 53 := 
sorry

end prime_p_in_range_l62_62306


namespace line_representation_l62_62157

variable {R : Type*} [Field R]
variable (f : R → R → R)
variable (x0 y0 : R)

def not_on_line (P : R × R) (f : R → R → R) : Prop :=
  f P.1 P.2 ≠ 0

theorem line_representation (P : R × R) (hP : not_on_line P f) :
  ∃ l : R → R → Prop, (∀ x y, l x y ↔ f x y - f P.1 P.2 = 0) ∧ (l P.1 P.2) ∧ 
  ∀ x y, f x y = 0 → ∃ n : R, ∀ x1 y1, (l x1 y1 → f x1 y1 = n * (f x y)) :=
sorry

end line_representation_l62_62157


namespace total_tiles_l62_62114

theorem total_tiles (n : ℕ) (h : 2 * n - 1 = 133) : n^2 = 4489 :=
by
  sorry

end total_tiles_l62_62114


namespace quadratic_solutions_l62_62959

theorem quadratic_solutions:
  (2 * (x : ℝ)^2 - 5 * x + 2 = 0) ↔ (x = 2 ∨ x = 1 / 2) :=
sorry

end quadratic_solutions_l62_62959


namespace cary_mow_weekends_l62_62395

theorem cary_mow_weekends :
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  remaining_amount / earn_per_weekend = 6 :=
by
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  have needed_weekends : remaining_amount / earn_per_weekend = 6 :=
    sorry
  exact needed_weekends

end cary_mow_weekends_l62_62395


namespace number_of_students_l62_62185

-- Define parameters and conditions
variables (B G : ℕ) -- number of boys and girls

-- Condition: each boy is friends with exactly two girls
axiom boys_to_girls : ∀ (B G : ℕ), 2 * B = 3 * G

-- Condition: total number of children in the class
axiom total_children : ∀ (B G : ℕ), B + G = 31

-- Define the theorem that proves the correct number of students
theorem number_of_students : (B G : ℕ) → 2 * B = 3 * G → B + G = 31 → B + G = 35 :=
by
  sorry

end number_of_students_l62_62185


namespace find_z_l62_62630

open Complex

theorem find_z (z : ℂ) : (1 + 2*I) * z = 3 - I → z = (1/5) - (7/5)*I :=
by
  intro h
  sorry

end find_z_l62_62630


namespace increase_in_difference_between_strawberries_and_blueberries_l62_62853

theorem increase_in_difference_between_strawberries_and_blueberries :
  ∀ (B S : ℕ), B = 32 → S = B + 12 → (S - B) = 12 :=
by
  intros B S hB hS
  sorry

end increase_in_difference_between_strawberries_and_blueberries_l62_62853


namespace cary_mow_weekends_l62_62397

theorem cary_mow_weekends :
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  remaining_amount / earn_per_weekend = 6 :=
by
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  have needed_weekends : remaining_amount / earn_per_weekend = 6 :=
    sorry
  exact needed_weekends

end cary_mow_weekends_l62_62397


namespace length_of_shorter_leg_l62_62178

variable (h x : ℝ)

theorem length_of_shorter_leg 
  (h_med : h / 2 = 5 * Real.sqrt 3) 
  (hypotenuse_relation : h = 2 * x) 
  (median_relation : h / 2 = h / 2) :
  x = 5 := by sorry

end length_of_shorter_leg_l62_62178


namespace factorial_last_two_digits_sum_eq_l62_62995

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def last_two_digits (n : ℕ) : ℕ :=
n % 100

def sum_of_factorials_last_two_digits : ℕ :=
(last_two_digits(factorial 1) +
 last_two_digits(factorial 2) +
 last_two_digits(factorial 3) +
 last_two_digits(factorial 4) +
 last_two_digits(factorial 5) +
 last_two_digits(factorial 6) +
 last_two_digits(factorial 7) +
 last_two_digits(factorial 8) +
 last_two_digits(factorial 9)) % 100

theorem factorial_last_two_digits_sum_eq :
  sum_of_factorials_last_two_digits = 13 :=
by
  sorry

end factorial_last_two_digits_sum_eq_l62_62995


namespace negation_of_forall_ge_zero_l62_62062

theorem negation_of_forall_ge_zero :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 := by
  sorry

end negation_of_forall_ge_zero_l62_62062


namespace find_n_l62_62645

-- Define the operation €
def operation (x y : ℕ) : ℕ := 2 * x * y

-- State the theorem
theorem find_n (n : ℕ) (h : operation 8 (operation 4 n) = 640) : n = 5 :=
  by
  sorry

end find_n_l62_62645


namespace min_AC_plus_BD_l62_62058

theorem min_AC_plus_BD (k : ℝ) (h : k ≠ 0) :
  (8 + 8 / k^2) + (8 + 2 * k^2) ≥ 24 :=
by
  sorry -- skipping the proof

end min_AC_plus_BD_l62_62058


namespace quadratic_expression_value_l62_62691

theorem quadratic_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 * x2 = 2) (hx : x1^2 - 4 * x1 + 2 = 0) :
  x1^2 - 4 * x1 + 2 * x1 * x2 = 2 :=
sorry

end quadratic_expression_value_l62_62691


namespace find_added_number_l62_62309

theorem find_added_number (R D Q X : ℕ) (hR : R = 5) (hD : D = 3 * Q) (hDiv : 113 = D * Q + R) (hD_def : D = 3 * R + X) : 
  X = 3 :=
by
  -- Provide the conditions as assumptions
  sorry

end find_added_number_l62_62309


namespace positive_integer_pairs_count_l62_62271

theorem positive_integer_pairs_count :
  ∃ (pairs : List (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs → a > 0 ∧ b > 0 ∧ (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / 2021) ∧ 
    pairs.length = 4 :=
by sorry

end positive_integer_pairs_count_l62_62271


namespace general_formula_sums_inequality_for_large_n_l62_62278

def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def modified_seq (a b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, b n = if n % 2 = 1 then a n - 6 else 2 * a n

def sum_seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = (finset.range n).sum a

variables {a b S T : ℕ → ℕ}

-- Given conditions
axiom h_arith_seq : arithmetic_seq a
axiom h_modified_seq : modified_seq a b
axiom h_sum_a : sum_seq a S
axiom h_sum_b : sum_seq b T
axiom h_S4 : S 4 = 32
axiom h_T3 : T 3 = 16

-- Proof Statements
theorem general_formula :
  ∀ n : ℕ, a n = 2 * n + 3 := sorry

theorem sums_inequality_for_large_n :
  ∀ n : ℕ, n > 5 → T n > S n := sorry

end general_formula_sums_inequality_for_large_n_l62_62278


namespace problem_equivalent_to_l62_62617

theorem problem_equivalent_to (x : ℝ)
  (A : x^2 = 5*x - 6 ↔ x = 2 ∨ x = 3)
  (B : x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3)
  (C : x = x + 1 ↔ false)
  (D : x^2 - 5*x + 7 = 1 ↔ x = 2 ∨ x = 3)
  (E : x^2 - 1 = 5*x - 7 ↔ x = 2 ∨ x = 3) :
  ¬ (x = x + 1) :=
by sorry

end problem_equivalent_to_l62_62617


namespace line_and_circle_condition_l62_62651

theorem line_and_circle_condition (P Q : ℝ × ℝ) (radius : ℝ) 
  (x y m : ℝ) (n : ℝ) (l : ℝ × ℝ → Prop)
  (hPQ : P = (4, -2)) 
  (hPQ' : Q = (-1, 3)) 
  (hC : ∀ (x y : ℝ), (x - 1)^2 + y^2 = radius) 
  (hr : radius < 5) 
  (h_y_segment : ∃ (k : ℝ), |k - 0| = 4 * Real.sqrt 3) 
  : (∀ (x y : ℝ), x + y = 2) ∧ 
    ((∀ (x y : ℝ), l (x, y) ↔ x + y + m = 0 ∨ x + y = 0) 
    ∧ (m = 3 ∨ m = -4) 
    ∧ (∀ A B : ℝ × ℝ, l A → l B → (A.1 - B.1)^2 + (A.2 - B.2)^2 = radius)) := 
  by
  sorry

end line_and_circle_condition_l62_62651


namespace boys_chairs_problem_l62_62967

theorem boys_chairs_problem :
  ∃ (n k : ℕ), n * k = 123 ∧ (∀ p q : ℕ, p * q = 123 → p = n ∧ q = k ∨ p = k ∧ q = n) :=
by
  sorry

end boys_chairs_problem_l62_62967


namespace terry_mary_same_combination_l62_62716

noncomputable def probability_same_combination : ℚ :=
  let total_candies := 12 + 8
  let terry_red := (12.choose 2) / (total_candies.choose 2) 
  let remaining_red := 12 - 2
  let remaining_total := total_candies - 2
  let mary_red := (remaining_red.choose 2) / (remaining_total.choose 2)
  let both_red := terry_red * mary_red

  let terry_diff := (12.choose 1 * 8.choose 1) / (total_candies.choose 2)
  let mary_diff_red := (11.choose 1 * 7.choose 1) / (remaining_total.choose 2)
  let both_diff := terry_diff * mary_diff_red
  
  let combined := 2 * both_red + both_diff
  combined

theorem terry_mary_same_combination:
  probability_same_combination = (143 / 269) := by
  sorry

end terry_mary_same_combination_l62_62716


namespace evaluate_function_at_neg_one_l62_62294

def f (x : ℝ) : ℝ := -2 * x^2 + 1

theorem evaluate_function_at_neg_one : f (-1) = -1 :=
by
  sorry

end evaluate_function_at_neg_one_l62_62294


namespace congruent_triangles_have_equal_perimeters_and_areas_l62_62084

-- Definitions based on the conditions
structure Triangle :=
  (a b c : ℝ) -- sides of the triangle
  (A B C : ℝ) -- angles of the triangle

def congruent_triangles (Δ1 Δ2 : Triangle) : Prop :=
  Δ1.a = Δ2.a ∧ Δ1.b = Δ2.b ∧ Δ1.c = Δ2.c ∧
  Δ1.A = Δ2.A ∧ Δ1.B = Δ2.B ∧ Δ1.C = Δ2.C

-- perimeters and areas (assuming some function calc_perimeter and calc_area for simplicity)
def perimeter (Δ : Triangle) : ℝ := Δ.a + Δ.b + Δ.c
def area (Δ : Triangle) : ℝ := sorry -- implement area calculation, e.g., using Heron's formula

-- Statement to be proved
theorem congruent_triangles_have_equal_perimeters_and_areas (Δ1 Δ2 : Triangle) :
  congruent_triangles Δ1 Δ2 →
  perimeter Δ1 = perimeter Δ2 ∧ area Δ1 = area Δ2 :=
sorry

end congruent_triangles_have_equal_perimeters_and_areas_l62_62084


namespace binomial_coefficients_sum_l62_62928

theorem binomial_coefficients_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 := by
  sorry

end binomial_coefficients_sum_l62_62928


namespace train_speed_is_72_kmph_l62_62723

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 112
noncomputable def crossing_time : ℝ := 11.099112071034318

theorem train_speed_is_72_kmph :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_kmph := speed_m_per_s * 3.6
  speed_kmph = 72 :=
by
  sorry

end train_speed_is_72_kmph_l62_62723


namespace correct_calculation_l62_62548

variable (a : ℝ)

theorem correct_calculation : (-2 * a) ^ 3 = -8 * a ^ 3 := by
  sorry

end correct_calculation_l62_62548


namespace conor_total_vegetables_weekly_l62_62609

def conor_vegetables_daily (e c p o z : ℕ) : ℕ :=
  e + c + p + o + z

def conor_vegetables_weekly (vegetables_daily days_worked : ℕ) : ℕ :=
  vegetables_daily * days_worked

theorem conor_total_vegetables_weekly :
  conor_vegetables_weekly (conor_vegetables_daily 12 9 8 15 7) 6 = 306 := by
  sorry

end conor_total_vegetables_weekly_l62_62609


namespace lattice_points_hyperbola_count_l62_62762

theorem lattice_points_hyperbola_count : 
  {p : ℤ × ℤ | p.fst^2 - p.snd^2 = 1800^2}.to_finset.card = 150 :=
sorry

end lattice_points_hyperbola_count_l62_62762


namespace is_rectangle_l62_62437

-- Define the points A, B, C, and D.
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 6)
def C : ℝ × ℝ := (5, 4)
def D : ℝ × ℝ := (2, -2)

-- Define the vectors AB, DC, AD.
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
def AB := vec A B
def DC := vec D C
def AD := vec A D

-- Function to compute dot product of two vectors.
def dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that quadrilateral ABCD is a rectangle.
theorem is_rectangle : AB = DC ∧ dot AB AD = 0 := by
  sorry

end is_rectangle_l62_62437


namespace problem_statement_l62_62409

theorem problem_statement :
  (81000 ^ 3) / (27000 ^ 3) = 27 :=
by sorry

end problem_statement_l62_62409


namespace solve_fruit_juice_problem_l62_62663

open Real

noncomputable def fruit_juice_problem : Prop :=
  ∃ x, ((0.12 * 3 + x) / (3 + x) = 0.185) ∧ (x = 0.239)

theorem solve_fruit_juice_problem : fruit_juice_problem :=
sorry

end solve_fruit_juice_problem_l62_62663


namespace sum_of_squares_of_roots_l62_62261

theorem sum_of_squares_of_roots :
  let a := 10
  let b := 16
  let c := -18
  let sum_of_roots := -b / a
  let product_of_roots := c / a
  sum_of_roots ^ 2 - 2 * product_of_roots = 244 / 25 := by
  sorry

end sum_of_squares_of_roots_l62_62261


namespace min_cylinder_volume_eq_surface_area_l62_62307

theorem min_cylinder_volume_eq_surface_area (r h V S : ℝ) (hr : r > 0) (hh : h > 0)
  (hV : V = π * r^2 * h) (hS : S = 2 * π * r^2 + 2 * π * r * h) (heq : V = S) :
  V = 54 * π :=
by
  -- Placeholder for the actual proof
  sorry

end min_cylinder_volume_eq_surface_area_l62_62307


namespace range_of_m_l62_62163

noncomputable def f (x : ℝ) : ℝ := -x^3 + 6 * x^2 - 9 * x

def tangents_condition (m : ℝ) : Prop := ∃ x : ℝ, (-3 * x^2 + 12 * x - 9) * (x + 1) + m = -x^3 + 6 * x^2 - 9 * x

theorem range_of_m (m : ℝ) : tangents_condition m → -11 < m ∧ m < 16 :=
sorry

end range_of_m_l62_62163


namespace prob_different_topics_l62_62580

theorem prob_different_topics (T : ℕ) (hT : T = 6) :
  let total_outcomes := T * T,
      favorable_outcomes := T * (T - 1),
      probability_different := favorable_outcomes / total_outcomes
  in probability_different = 5 / 6 :=
by
  have : total_outcomes = 36 := by rw [hT]; norm_num
  have : favorable_outcomes = 30 := by rw [hT]; norm_num
  have : probability_different = 5 / 6 := by norm_num
  sorry

end prob_different_topics_l62_62580


namespace find_third_smallest_three_digit_palindromic_prime_l62_62835

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def second_smallest_three_digit_palindromic_prime : ℕ :=
  131 -- Given in the problem statement

noncomputable def third_smallest_three_digit_palindromic_prime : ℕ :=
  151 -- Answer obtained from the solution

theorem find_third_smallest_three_digit_palindromic_prime :
  ∃ n, is_palindrome n ∧ is_prime n ∧ 100 ≤ n ∧ n < 1000 ∧
  (n ≠ 101) ∧ (n ≠ 131) ∧ (∀ m, is_palindrome m ∧ is_prime m ∧ 100 ≤ m ∧ m < 1000 → second_smallest_three_digit_palindromic_prime < m → m = n) :=
by
  sorry -- This is where the proof would be, but it is not needed as per instructions.

end find_third_smallest_three_digit_palindromic_prime_l62_62835


namespace projected_percent_increase_l62_62323

theorem projected_percent_increase (R : ℝ) (p : ℝ) 
  (h1 : 0.7 * R = R * 0.7) 
  (h2 : 0.7 * R = 0.5 * (R + p * R)) : 
  p = 0.4 :=
by
  sorry

end projected_percent_increase_l62_62323


namespace cost_per_set_l62_62205

variable {C : ℝ} -- Define the variable cost per set.

theorem cost_per_set
  (initial_outlay : ℝ := 10000) -- Initial outlay for manufacturing.
  (revenue_per_set : ℝ := 50) -- Revenue per set sold.
  (sets_sold : ℝ := 500) -- Sets produced and sold.
  (profit : ℝ := 5000) -- Profit from selling 500 sets.

  (h_profit_eq : profit = (revenue_per_set * sets_sold) - (initial_outlay + C * sets_sold)) :
  C = 20 :=
by
  -- Proof to be filled in later.
  sorry

end cost_per_set_l62_62205


namespace flower_nectar_water_content_l62_62855

/-- Given that to yield 1 kg of honey, 1.6 kg of flower-nectar must be processed,
    and the honey obtained from this nectar contains 20% water,
    prove that the flower-nectar contains 50% water. --/
theorem flower_nectar_water_content :
  (1.6 : ℝ) * (0.2 / 1) = (50 / 100) * (1.6 : ℝ) := by
  sorry

end flower_nectar_water_content_l62_62855
