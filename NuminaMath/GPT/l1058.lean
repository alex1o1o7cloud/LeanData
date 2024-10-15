import Mathlib

namespace NUMINAMATH_GPT_add_zero_eq_self_l1058_105858

theorem add_zero_eq_self (n x : ℤ) (h : n + x = n) : x = 0 := 
sorry

end NUMINAMATH_GPT_add_zero_eq_self_l1058_105858


namespace NUMINAMATH_GPT_find_b_l1058_105857

theorem find_b (a b : ℝ) (f : ℝ → ℝ) (df : ℝ → ℝ) (x₀ : ℝ)
  (h₁ : ∀ x, f x = a * x + Real.log x)
  (h₂ : ∀ x, f x = 2 * x + b)
  (h₃ : x₀ = 1)
  (h₄ : f x₀ = a) :
  b = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_b_l1058_105857


namespace NUMINAMATH_GPT_jordan_wins_two_games_l1058_105882

theorem jordan_wins_two_games 
  (Peter_wins : ℕ) 
  (Peter_losses : ℕ)
  (Emma_wins : ℕ) 
  (Emma_losses : ℕ)
  (Jordan_losses : ℕ) 
  (hPeter : Peter_wins = 5)
  (hPeterL : Peter_losses = 4)
  (hEmma : Emma_wins = 4)
  (hEmmaL : Emma_losses = 5)
  (hJordanL : Jordan_losses = 2) : ∃ (J : ℕ), J = 2 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_jordan_wins_two_games_l1058_105882


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1058_105894

theorem solution_set_of_quadratic_inequality :
  {x : ℝ | 2 - x - x^2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1058_105894


namespace NUMINAMATH_GPT_distinct_rationals_count_l1058_105831

theorem distinct_rationals_count : ∃ N : ℕ, (N = 40) ∧ ∀ k : ℚ, (|k| < 100) → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) :=
by
  sorry

end NUMINAMATH_GPT_distinct_rationals_count_l1058_105831


namespace NUMINAMATH_GPT_q_computation_l1058_105852

def q : ℤ → ℤ → ℤ :=
  λ x y =>
    if x ≥ 0 ∧ y ≥ 0 then x + 2 * y
    else if x < 0 ∧ y < 0 then x - 3 * y
    else 2 * x + y

theorem q_computation : q (q 2 (-2)) (q (-4) (-1)) = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_q_computation_l1058_105852


namespace NUMINAMATH_GPT_exponent_solver_l1058_105840

theorem exponent_solver (x : ℕ) : 3^x + 3^x + 3^x + 3^x = 19683 → x = 7 := sorry

end NUMINAMATH_GPT_exponent_solver_l1058_105840


namespace NUMINAMATH_GPT_vector_orthogonality_solution_l1058_105821

theorem vector_orthogonality_solution :
  let a := (3, -2)
  let b := (x, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  x = 2 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_vector_orthogonality_solution_l1058_105821


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t2_l1058_105849

noncomputable def s (t : ℝ) : ℝ := t^3 - t^2 + 2 * t

theorem instantaneous_velocity_at_t2 : 
  deriv s 2 = 10 := 
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t2_l1058_105849


namespace NUMINAMATH_GPT_sin_double_angle_plus_pi_div_two_l1058_105822

open Real

theorem sin_double_angle_plus_pi_div_two (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) (h₂ : sin θ = 1 / 3) :
  sin (2 * θ + π / 2) = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_plus_pi_div_two_l1058_105822


namespace NUMINAMATH_GPT_value_of_x_after_z_doubled_l1058_105830

theorem value_of_x_after_z_doubled (x y z : ℕ) (hz : z = 48) (hz_d : z_d = 2 * z) (hy : y = z / 4) (hx : x = y / 3) :
  x = 8 := by
  -- Proof goes here (skipped as instructed)
  sorry

end NUMINAMATH_GPT_value_of_x_after_z_doubled_l1058_105830


namespace NUMINAMATH_GPT_ratio_large_to_small_l1058_105867

-- Definitions of the conditions
def total_fries_sold : ℕ := 24
def small_fries_sold : ℕ := 4
def large_fries_sold : ℕ := total_fries_sold - small_fries_sold

-- The proof goal
theorem ratio_large_to_small : large_fries_sold / small_fries_sold = 5 :=
by
  -- Mathematical steps would go here, but we skip with sorry
  sorry

end NUMINAMATH_GPT_ratio_large_to_small_l1058_105867


namespace NUMINAMATH_GPT_proportional_function_property_l1058_105883

theorem proportional_function_property :
  (∀ x, ∃ y, y = -3 * x ∧
  (x = 0 → y = 0) ∧
  (x > 0 → y < 0) ∧
  (x < 0 → y > 0) ∧
  (x = 1 → y = -3) ∧
  (∀ x, y = -3 * x → (x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0))) :=
by
  sorry

end NUMINAMATH_GPT_proportional_function_property_l1058_105883


namespace NUMINAMATH_GPT_expression_C_eq_seventeen_l1058_105887

theorem expression_C_eq_seventeen : (3 + 4 * 5 - 6) = 17 := 
by 
  sorry

end NUMINAMATH_GPT_expression_C_eq_seventeen_l1058_105887


namespace NUMINAMATH_GPT_find_a_l1058_105853

theorem find_a (a : ℤ) : 0 ≤ a ∧ a ≤ 13 ∧ (51^2015 + a) % 13 = 0 → a = 1 :=
by { sorry }

end NUMINAMATH_GPT_find_a_l1058_105853


namespace NUMINAMATH_GPT_problem_statement_l1058_105870

variable (a b c : ℝ)

theorem problem_statement
  (h1 : a + b = 100)
  (h2 : b + c = 140) :
  c - a = 40 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1058_105870


namespace NUMINAMATH_GPT_unique_polynomial_l1058_105888

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x
noncomputable def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

theorem unique_polynomial 
  (a b c : ℝ) 
  (extremes : f' a b c 1 = 0 ∧ f' a b c (-1) = 0) 
  (tangent_slope : f' a b c 0 = -3)
  : f a b c = f 1 0 (-3) := sorry

end NUMINAMATH_GPT_unique_polynomial_l1058_105888


namespace NUMINAMATH_GPT_stratified_sampling_male_athletes_l1058_105811

theorem stratified_sampling_male_athletes : 
  ∀ (total_males total_females total_to_sample : ℕ), 
    total_males = 20 → 
    total_females = 10 → 
    total_to_sample = 6 → 
    20 * (total_to_sample / (total_males + total_females)) = 4 :=
by
  intros total_males total_females total_to_sample h_males h_females h_sample
  rw [h_males, h_females, h_sample]
  sorry

end NUMINAMATH_GPT_stratified_sampling_male_athletes_l1058_105811


namespace NUMINAMATH_GPT_superchess_no_attacks_l1058_105805

open Finset

theorem superchess_no_attacks (board_size : ℕ) (num_pieces : ℕ)  (attack_limit : ℕ) 
  (h_board_size : board_size = 100) (h_num_pieces : num_pieces = 20) 
  (h_attack_limit : attack_limit = 20) : 
  ∃ (placements : Finset (ℕ × ℕ)), placements.card = num_pieces ∧
  ∀ {p1 p2 : ℕ × ℕ}, p1 ≠ p2 → p1 ∈ placements → p2 ∈ placements → 
  ¬(∃ (attack_positions : Finset (ℕ × ℕ)), attack_positions.card ≤ attack_limit ∧ 
  ∃ piece_pos : ℕ × ℕ, piece_pos ∈ placements ∧ attack_positions ⊆ placements ∧ p1 ∈ attack_positions ∧ p2 ∈ attack_positions) :=
sorry

end NUMINAMATH_GPT_superchess_no_attacks_l1058_105805


namespace NUMINAMATH_GPT_calculate_k_l1058_105814

variable (A B C D k : ℕ)

def workers_time : Prop :=
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (A - 8 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (B - 2 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (3 / (C : ℚ))

theorem calculate_k (h : workers_time A B C D) : k = 16 :=
  sorry

end NUMINAMATH_GPT_calculate_k_l1058_105814


namespace NUMINAMATH_GPT_evaluate_expression_l1058_105813

theorem evaluate_expression (a b : ℤ) (h_a : a = 4) (h_b : b = -3) : -a - b^3 + a * b = 11 :=
by
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1058_105813


namespace NUMINAMATH_GPT_miles_walked_on_Tuesday_l1058_105842

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end NUMINAMATH_GPT_miles_walked_on_Tuesday_l1058_105842


namespace NUMINAMATH_GPT_point_in_first_quadrant_l1058_105873

-- Define the system of equations
def equations (x y : ℝ) : Prop :=
  x + y = 2 ∧ x - y = 1

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : equations x y) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_GPT_point_in_first_quadrant_l1058_105873


namespace NUMINAMATH_GPT_quadractic_b_value_l1058_105898

def quadratic_coefficients (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem quadractic_b_value :
  ∀ (a b c : ℝ), quadratic_coefficients 1 (-2) (-3) (x : ℝ) → 
  b = -2 := by
  sorry

end NUMINAMATH_GPT_quadractic_b_value_l1058_105898


namespace NUMINAMATH_GPT_jose_share_of_profit_correct_l1058_105832

noncomputable def jose_share_of_profit (total_profit : ℝ) : ℝ :=
  let tom_investment_time := 30000 * 12
  let jose_investment_time := 45000 * 10
  let angela_investment_time := 60000 * 8
  let rebecca_investment_time := 75000 * 6
  let total_investment_time := tom_investment_time + jose_investment_time + angela_investment_time + rebecca_investment_time
  (jose_investment_time / total_investment_time) * total_profit

theorem jose_share_of_profit_correct : 
  ∀ (total_profit : ℝ), total_profit = 72000 -> jose_share_of_profit total_profit = 18620.69 := 
by
  intro total_profit
  sorry

end NUMINAMATH_GPT_jose_share_of_profit_correct_l1058_105832


namespace NUMINAMATH_GPT_solve_problem_l1058_105824

noncomputable def problem_statement (x : ℝ) : Prop :=
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * Real.cos (3 * x / 2) ^ 2

theorem solve_problem (x : ℝ) :
  problem_statement x ↔
  (∃ k : ℤ, x = (Real.pi / 8) * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = (Real.pi / 4) * (4 * n - 1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_problem_l1058_105824


namespace NUMINAMATH_GPT_fraction_arithmetic_proof_l1058_105826

theorem fraction_arithmetic_proof :
  (7 / 6) + (5 / 4) - (3 / 2) = 11 / 12 :=
by sorry

end NUMINAMATH_GPT_fraction_arithmetic_proof_l1058_105826


namespace NUMINAMATH_GPT_petya_cannot_have_equal_coins_l1058_105854

def petya_initial_two_kopeck_coins : Nat := 1
def petya_initial_ten_kopeck_coins : Nat := 0
def petya_use_ten_kopeck (T G : Nat) : Nat := G - 1 + T + 5
def petya_use_two_kopeck (T G : Nat) : Nat := T - 1 + G + 5

theorem petya_cannot_have_equal_coins : ¬ (∃ n : Nat, 
  ∃ T G : Nat, 
    T = G ∧ 
    (n = petya_use_ten_kopeck T G ∨ n = petya_use_two_kopeck T G ∨ n = petya_initial_two_kopeck_coins + petya_initial_ten_kopeck_coins)) := 
by
  sorry

end NUMINAMATH_GPT_petya_cannot_have_equal_coins_l1058_105854


namespace NUMINAMATH_GPT_solutions_of_quadratic_l1058_105869

theorem solutions_of_quadratic 
  (p q : ℚ) 
  (h₁ : 2 * p * p + 11 * p - 21 = 0) 
  (h₂ : 2 * q * q + 11 * q - 21 = 0) : 
  (p - q) * (p - q) = 289 / 4 := 
sorry

end NUMINAMATH_GPT_solutions_of_quadratic_l1058_105869


namespace NUMINAMATH_GPT_range_for_a_l1058_105879

theorem range_for_a (f : ℝ → ℝ) (a : ℝ) (n : ℝ) :
  (∀ x, f x = x^n) →
  f 8 = 1/4 →
  f (a+1) < f 2 →
  a < -3 ∨ a > 1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_range_for_a_l1058_105879


namespace NUMINAMATH_GPT_negation_proposition_l1058_105891

theorem negation_proposition :
  (¬(∀ x : ℝ, x^2 - x + 2 < 0) ↔ ∃ x : ℝ, x^2 - x + 2 ≥ 0) :=
sorry

end NUMINAMATH_GPT_negation_proposition_l1058_105891


namespace NUMINAMATH_GPT_eggs_in_each_basket_l1058_105844

theorem eggs_in_each_basket :
  ∃ x : ℕ, x ∣ 30 ∧ x ∣ 42 ∧ x ≥ 5 ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_eggs_in_each_basket_l1058_105844


namespace NUMINAMATH_GPT_total_good_vegetables_l1058_105889

theorem total_good_vegetables :
  let carrots_day1 := 23
  let carrots_day2 := 47
  let tomatoes_day1 := 34
  let cucumbers_day1 := 42
  let tomatoes_day2 := 50
  let cucumbers_day2 := 38
  let rotten_carrots_day1 := 10
  let rotten_carrots_day2 := 15
  let rotten_tomatoes_day1 := 5
  let rotten_cucumbers_day1 := 7
  let rotten_tomatoes_day2 := 7
  let rotten_cucumbers_day2 := 12
  let good_carrots := (carrots_day1 - rotten_carrots_day1) + (carrots_day2 - rotten_carrots_day2)
  let good_tomatoes := (tomatoes_day1 - rotten_tomatoes_day1) + (tomatoes_day2 - rotten_tomatoes_day2)
  let good_cucumbers := (cucumbers_day1 - rotten_cucumbers_day1) + (cucumbers_day2 - rotten_cucumbers_day2)
  good_carrots + good_tomatoes + good_cucumbers = 178 := 
  sorry

end NUMINAMATH_GPT_total_good_vegetables_l1058_105889


namespace NUMINAMATH_GPT_billboard_dimensions_l1058_105864

theorem billboard_dimensions (photo_width_cm : ℕ) (photo_length_dm : ℕ) (billboard_area_m2 : ℕ)
  (h1 : photo_width_cm = 30) (h2 : photo_length_dm = 4) (h3 : billboard_area_m2 = 48) :
  ∃ photo_length_cm : ℕ, photo_length_cm = 40 ∧
  ∃ k : ℕ, k = 20 ∧
  ∃ billboard_width_m billboard_length_m : ℕ,
    billboard_width_m = photo_width_cm * k / 100 ∧ 
    billboard_length_m = photo_length_cm * k / 100 ∧ 
    billboard_width_m = 6 ∧ 
    billboard_length_m = 8 := by
  sorry

end NUMINAMATH_GPT_billboard_dimensions_l1058_105864


namespace NUMINAMATH_GPT_mrs_jensens_preschool_l1058_105833

theorem mrs_jensens_preschool (total_students students_with_both students_with_neither students_with_green_eyes students_with_red_hair : ℕ) 
(h1 : total_students = 40) 
(h2 : students_with_red_hair = 3 * students_with_green_eyes) 
(h3 : students_with_both = 8) 
(h4 : students_with_neither = 4) :
students_with_green_eyes = 12 := 
sorry

end NUMINAMATH_GPT_mrs_jensens_preschool_l1058_105833


namespace NUMINAMATH_GPT_num_students_l1058_105817

theorem num_students (n : ℕ) 
    (average_marks_wrong : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (average_marks_correct : ℕ) :
    average_marks_wrong = 100 →
    wrong_mark = 90 →
    correct_mark = 10 →
    average_marks_correct = 92 →
    n = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_num_students_l1058_105817


namespace NUMINAMATH_GPT_probability_not_overcoming_is_half_l1058_105886

/-- Define the five elements. -/
inductive Element
| metal | wood | water | fire | earth

open Element

/-- Define the overcoming relation. -/
def overcomes : Element → Element → Prop
| metal, wood => true
| wood, earth => true
| earth, water => true
| water, fire => true
| fire, metal => true
| _, _ => false

/-- Define the probability calculation. -/
def probability_not_overcoming : ℚ :=
  let total_combinations := 10    -- C(5, 2)
  let overcoming_combinations := 5
  let not_overcoming_combinations := total_combinations - overcoming_combinations
  not_overcoming_combinations / total_combinations

/-- The proof problem statement. -/
theorem probability_not_overcoming_is_half : probability_not_overcoming = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_overcoming_is_half_l1058_105886


namespace NUMINAMATH_GPT_coplanar_lines_l1058_105801

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 5 - k * s, 3 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * t, 4 + 2 * t, 6 - 2 * t)

theorem coplanar_lines (k : ℝ) :
  (exists s t : ℝ, line1 s k = line2 t) ∨ line1 1 k = (1, -k, k) ∧ line2 1 = (2, 2, -2) → k = -1 :=
by sorry

end NUMINAMATH_GPT_coplanar_lines_l1058_105801


namespace NUMINAMATH_GPT_quadratic_solution_value_l1058_105839

open Real

theorem quadratic_solution_value (a b : ℝ) (h1 : 2 + b = -a) (h2 : 2 * b = -6) :
  (2 * a + b)^2023 = -1 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_value_l1058_105839


namespace NUMINAMATH_GPT_toll_for_18_wheel_truck_l1058_105859

-- Define the number of wheels on the front axle and the other axles
def front_axle_wheels : ℕ := 2
def other_axle_wheels : ℕ := 4
def total_wheels : ℕ := 18

-- Define the toll formula
def toll (x : ℕ) : ℝ := 3.50 + 0.50 * (x - 2)

-- Calculate the number of axles for the 18-wheel truck
def num_axles : ℕ := 1 + (total_wheels - front_axle_wheels) / other_axle_wheels

-- Define the expected toll for the given number of axles
def expected_toll : ℝ := 5.00

-- State the theorem
theorem toll_for_18_wheel_truck : toll num_axles = expected_toll := by
    sorry

end NUMINAMATH_GPT_toll_for_18_wheel_truck_l1058_105859


namespace NUMINAMATH_GPT_wholesale_cost_is_200_l1058_105809

variable (W R E : ℝ)

def retail_price (W : ℝ) : ℝ := 1.20 * W

def employee_price (R : ℝ) : ℝ := 0.75 * R

-- Main theorem stating that given the retail and employee price formulas and the employee paid amount,
-- the wholesale cost W is equal to 200.
theorem wholesale_cost_is_200
  (hR : R = retail_price W)
  (hE : E = employee_price R)
  (heq : E = 180) :
  W = 200 :=
by
  sorry

end NUMINAMATH_GPT_wholesale_cost_is_200_l1058_105809


namespace NUMINAMATH_GPT_pizzas_in_park_l1058_105865

-- Define the conditions and the proof problem
def pizza_cost : ℕ := 12
def delivery_charge : ℕ := 2
def park_distance : ℕ := 100  -- in meters
def building_distance : ℕ := 2000  -- in meters
def pizzas_delivered_to_building : ℕ := 2
def total_payment_received : ℕ := 64

-- Prove the number of pizzas delivered in the park
theorem pizzas_in_park : (64 - (pizzas_delivered_to_building * pizza_cost + delivery_charge)) / pizza_cost = 3 :=
by
  sorry -- Proof not required

end NUMINAMATH_GPT_pizzas_in_park_l1058_105865


namespace NUMINAMATH_GPT_solve_system_l1058_105810

theorem solve_system (x y z : ℝ) 
  (h1 : x^3 - y = 6)
  (h2 : y^3 - z = 6)
  (h3 : z^3 - x = 6) :
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end NUMINAMATH_GPT_solve_system_l1058_105810


namespace NUMINAMATH_GPT_num_people_is_8_l1058_105838

-- Define the known conditions
def bill_amt : ℝ := 314.16
def person_amt : ℝ := 34.91
def total_amt : ℝ := 314.19

-- Prove that the number of people is 8
theorem num_people_is_8 : ∃ num_people : ℕ, num_people = total_amt / person_amt ∧ num_people = 8 :=
by
  sorry

end NUMINAMATH_GPT_num_people_is_8_l1058_105838


namespace NUMINAMATH_GPT_solve_equation_l1058_105871

theorem solve_equation (x : ℝ) (h : 2 * x + 6 = 2 + 3 * x) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1058_105871


namespace NUMINAMATH_GPT_evaluate_expression_l1058_105851

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem evaluate_expression : spadesuit 3 (spadesuit 6 5) = -112 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1058_105851


namespace NUMINAMATH_GPT_father_age_difference_l1058_105861

variables (F S X : ℕ)
variable (h1 : F = 33)
variable (h2 : F = 3 * S + X)
variable (h3 : F + 3 = 2 * (S + 3) + 10)

theorem father_age_difference : X = 3 :=
by
  sorry

end NUMINAMATH_GPT_father_age_difference_l1058_105861


namespace NUMINAMATH_GPT_product_of_smallest_primes_l1058_105884

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def smallest_one_digit_primes : List ℕ := [2, 3]
def smallest_two_digit_prime : ℕ := 11

theorem product_of_smallest_primes : 
  (smallest_one_digit_primes.prod * smallest_two_digit_prime) = 66 :=
by
  sorry

end NUMINAMATH_GPT_product_of_smallest_primes_l1058_105884


namespace NUMINAMATH_GPT_gabor_can_cross_l1058_105860

open Real

-- Definitions based on conditions
def river_width : ℝ := 100
def total_island_perimeter : ℝ := 800
def banks_parallel : Prop := true

theorem gabor_can_cross (w : ℝ) (p : ℝ) (bp : Prop) : 
  w = river_width → 
  p = total_island_perimeter → 
  bp = banks_parallel → 
  ∃ d : ℝ, d ≤ 300 := 
by
  sorry

end NUMINAMATH_GPT_gabor_can_cross_l1058_105860


namespace NUMINAMATH_GPT_water_settles_at_34_cm_l1058_105868

-- Conditions definitions
def h : ℝ := 40 -- Initial height of the liquids in cm
def ρ_w : ℝ := 1000 -- Density of water in kg/m^3
def ρ_o : ℝ := 700  -- Density of oil in kg/m^3

-- Given the conditions provided above,
-- prove that the new height level of water in the first vessel is 34 cm
theorem water_settles_at_34_cm :
  (40 / (1 + (ρ_o / ρ_w))) = 34 := 
sorry

end NUMINAMATH_GPT_water_settles_at_34_cm_l1058_105868


namespace NUMINAMATH_GPT_assumption_for_contradiction_l1058_105819

theorem assumption_for_contradiction (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (h : 5 ∣ a * b) : 
  ¬ (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) := 
sorry

end NUMINAMATH_GPT_assumption_for_contradiction_l1058_105819


namespace NUMINAMATH_GPT_solve_for_a_l1058_105855

theorem solve_for_a (a : ℝ) (y : ℝ) (h1 : 4 * 2 + y = a) (h2 : 2 * 2 + 5 * y = 3 * a) : a = 18 :=
  sorry

end NUMINAMATH_GPT_solve_for_a_l1058_105855


namespace NUMINAMATH_GPT_neither_plaid_nor_purple_l1058_105841

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end NUMINAMATH_GPT_neither_plaid_nor_purple_l1058_105841


namespace NUMINAMATH_GPT_julien_swims_50_meters_per_day_l1058_105892

-- Definitions based on given conditions
def distance_julien_swims_per_day : ℕ := 50
def distance_sarah_swims_per_day (J : ℕ) : ℕ := 2 * J
def distance_jamir_swims_per_day (J : ℕ) : ℕ := distance_sarah_swims_per_day J + 20
def combined_distance_per_day (J : ℕ) : ℕ := J + distance_sarah_swims_per_day J + distance_jamir_swims_per_day J
def combined_distance_per_week (J : ℕ) : ℕ := 7 * combined_distance_per_day J

-- Proof statement 
theorem julien_swims_50_meters_per_day :
  combined_distance_per_week distance_julien_swims_per_day = 1890 :=
by
  -- We are formulating the proof without solving it, to be proven formally in Lean
  sorry

end NUMINAMATH_GPT_julien_swims_50_meters_per_day_l1058_105892


namespace NUMINAMATH_GPT_find_roots_l1058_105877

-- Given the conditions:
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given points (x, y)
def points := [(-5, 6), (-4, 0), (-2, -6), (0, -4), (2, 6)] 

-- Prove that the roots of the quadratic equation are -4 and 1
theorem find_roots (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : quadratic_function a b c (-5) = 6)
  (h₂ : quadratic_function a b c (-4) = 0)
  (h₃ : quadratic_function a b c (-2) = -6)
  (h₄ : quadratic_function a b c (0) = -4)
  (h₅ : quadratic_function a b c (2) = 6) :
  ∃ x₁ x₂ : ℝ, quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0 ∧ x₁ = -4 ∧ x₂ = 1 := 
sorry

end NUMINAMATH_GPT_find_roots_l1058_105877


namespace NUMINAMATH_GPT_boy_current_age_l1058_105802

theorem boy_current_age (x : ℕ) (h : 5 ≤ x) (age_statement : x = 2 * (x - 5)) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_boy_current_age_l1058_105802


namespace NUMINAMATH_GPT_cos_pi_minus_alpha_l1058_105806

theorem cos_pi_minus_alpha (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : 0 < α ∧ α < π / 2) :
  Real.cos (π - α) = -12 / 13 :=
sorry

end NUMINAMATH_GPT_cos_pi_minus_alpha_l1058_105806


namespace NUMINAMATH_GPT_area_bounded_region_l1058_105803

theorem area_bounded_region :
  (∃ (x y : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) →
  ∃ (area : ℝ), area = 1250 :=
by
  sorry

end NUMINAMATH_GPT_area_bounded_region_l1058_105803


namespace NUMINAMATH_GPT_solution_l1058_105881

open Set

theorem solution (A B : Set ℤ) :
  (∀ x, x ∈ A ∨ x ∈ B) →
  (∀ x, x ∈ A → (x - 1) ∈ B) →
  (∀ x y, x ∈ B ∧ y ∈ B → (x + y) ∈ A) →
  A = { z | ∃ n, z = 2 * n } ∧ B = { z | ∃ n, z = 2 * n + 1 } :=
by
  sorry

end NUMINAMATH_GPT_solution_l1058_105881


namespace NUMINAMATH_GPT_inequality_sum_l1058_105876

theorem inequality_sum
  (x y z : ℝ)
  (h : abs (x * y * z) = 1) :
  (1 / (x^2 + x + 1) + 1 / (x^2 - x + 1)) +
  (1 / (y^2 + y + 1) + 1 / (y^2 - y + 1)) +
  (1 / (z^2 + z + 1) + 1 / (z^2 - z + 1)) ≤ 4 := 
sorry

end NUMINAMATH_GPT_inequality_sum_l1058_105876


namespace NUMINAMATH_GPT_chad_total_spend_on_ice_l1058_105816

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end NUMINAMATH_GPT_chad_total_spend_on_ice_l1058_105816


namespace NUMINAMATH_GPT_mike_payments_total_months_l1058_105899

-- Definitions based on conditions
def lower_rate := 295
def higher_rate := 310
def lower_payments := 5
def higher_payments := 7
def total_paid := 3615

-- The statement to prove
theorem mike_payments_total_months : lower_payments + higher_payments = 12 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mike_payments_total_months_l1058_105899


namespace NUMINAMATH_GPT_mixed_groups_count_l1058_105823

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_mixed_groups_count_l1058_105823


namespace NUMINAMATH_GPT_probability_of_5_blue_marbles_l1058_105829

/--
Jane has a bag containing 9 blue marbles and 6 red marbles. 
She draws a marble, records its color, returns it to the bag, and repeats this process 8 times. 
We aim to prove that the probability that she draws exactly 5 blue marbles is \(0.279\).
-/
theorem probability_of_5_blue_marbles :
  let blue_probability := 9 / 15 
  let red_probability := 6 / 15
  let single_combination_prob := (blue_probability^5) * (red_probability^3)
  let combinations := (Nat.choose 8 5)
  let total_probability := combinations * single_combination_prob
  (Float.round (total_probability.toFloat * 1000) / 1000) = 0.279 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_5_blue_marbles_l1058_105829


namespace NUMINAMATH_GPT_cistern_water_breadth_l1058_105880

theorem cistern_water_breadth (length width total_area : ℝ) (h : ℝ) 
  (h_length : length = 10) 
  (h_width : width = 6) 
  (h_area : total_area = 103.2) : 
  (60 + 20*h + 12*h = total_area) → h = 1.35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cistern_water_breadth_l1058_105880


namespace NUMINAMATH_GPT_area_of_shape_l1058_105834

def points := [(0, 1), (1, 2), (3, 2), (4, 1), (2, 0)]

theorem area_of_shape : 
  let I := 6 -- Number of interior points
  let B := 5 -- Number of boundary points
  ∃ (A : ℝ), A = I + B / 2 - 1 ∧ A = 7.5 := 
  by
    use 7.5
    simp
    sorry

end NUMINAMATH_GPT_area_of_shape_l1058_105834


namespace NUMINAMATH_GPT_isosceles_triangle_area_l1058_105836

theorem isosceles_triangle_area (a b h : ℝ) (h_eq : h = a / (2 * Real.sqrt 3)) :
  (1 / 2 * a * h) = (a^2 * Real.sqrt 3) / 12 :=
by
  -- Define the necessary parameters and conditions
  let area := (1 / 2) * a * h
  have h := h_eq
  -- Substitute and prove the calculated area
  sorry

end NUMINAMATH_GPT_isosceles_triangle_area_l1058_105836


namespace NUMINAMATH_GPT_sum_of_digits_eq_11_l1058_105856

-- Define the problem conditions
variables (p q r : ℕ)
variables (h1 : 1 ≤ p ∧ p ≤ 9)
variables (h2 : 1 ≤ q ∧ q ≤ 9)
variables (h3 : 1 ≤ r ∧ r ≤ 9)
variables (h4 : p ≠ q ∧ p ≠ r ∧ q ≠ r)
variables (h5 : (10 * p + q) * (10 * p + r) = 221)

-- Define the theorem
theorem sum_of_digits_eq_11 : p + q + r = 11 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_eq_11_l1058_105856


namespace NUMINAMATH_GPT_work_completion_time_l1058_105896

-- Definitions for work rates
def work_rate_B : ℚ := 1 / 7
def work_rate_A : ℚ := 1 / 10

-- Statement to prove
theorem work_completion_time (W : ℚ) : 
  (1 / work_rate_A + 1 / work_rate_B) = 70 / 17 := 
by 
  sorry

end NUMINAMATH_GPT_work_completion_time_l1058_105896


namespace NUMINAMATH_GPT_simplify_expression_l1058_105893

theorem simplify_expression (a b c d : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) :
  -5 * a + 2017 * c * d - 5 * b = 2017 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1058_105893


namespace NUMINAMATH_GPT_contingency_table_proof_l1058_105807

noncomputable def probability_of_mistake (K_squared : ℝ) : ℝ :=
if K_squared > 3.841 then 0.05 else 1.0 -- placeholder definition to be refined

theorem contingency_table_proof :
  probability_of_mistake 4.013 ≤ 0.05 :=
by sorry

end NUMINAMATH_GPT_contingency_table_proof_l1058_105807


namespace NUMINAMATH_GPT_train_length_is_sixteenth_mile_l1058_105847

theorem train_length_is_sixteenth_mile
  (train_speed : ℕ)
  (bridge_length : ℕ)
  (man_speed : ℕ)
  (cross_time : ℚ)
  (man_distance : ℚ)
  (length_of_train : ℚ)
  (h1 : train_speed = 80)
  (h2 : bridge_length = 1)
  (h3 : man_speed = 5)
  (h4 : cross_time = bridge_length / train_speed)
  (h5 : man_distance = man_speed * cross_time)
  (h6 : length_of_train = man_distance) :
  length_of_train = 1 / 16 :=
by sorry

end NUMINAMATH_GPT_train_length_is_sixteenth_mile_l1058_105847


namespace NUMINAMATH_GPT_mahmoud_gets_at_least_two_heads_l1058_105828

def probability_of_at_least_two_heads := 1 - ((1/2)^5 + 5 * (1/2)^5)

theorem mahmoud_gets_at_least_two_heads (n : ℕ) (hn : n = 5) :
  probability_of_at_least_two_heads = 13 / 16 :=
by
  simp only [probability_of_at_least_two_heads, hn]
  sorry

end NUMINAMATH_GPT_mahmoud_gets_at_least_two_heads_l1058_105828


namespace NUMINAMATH_GPT_perimeter_of_sector_l1058_105808

theorem perimeter_of_sector (r : ℝ) (area : ℝ) (perimeter : ℝ) 
  (hr : r = 1) (ha : area = π / 3) : perimeter = (2 * π / 3) + 2 :=
by
  -- You can start the proof here
  sorry

end NUMINAMATH_GPT_perimeter_of_sector_l1058_105808


namespace NUMINAMATH_GPT_expenditure_of_negative_l1058_105890

def income := 5000
def expenditure (x : Int) : Int := -x

theorem expenditure_of_negative (x : Int) : expenditure (-x) = x :=
by
  sorry

example : expenditure (-400) = 400 :=
by 
  exact expenditure_of_negative 400

end NUMINAMATH_GPT_expenditure_of_negative_l1058_105890


namespace NUMINAMATH_GPT_equation_of_trisection_line_l1058_105850

/-- Let P be the point (1, 2) and let A and B be the points (2, 3) and (-3, 0), respectively. 
    One of the lines through point P and a trisection point of the line segment joining A and B has 
    the equation 3x + 7y = 17. -/
theorem equation_of_trisection_line :
  let P : ℝ × ℝ := (1, 2)
  let A : ℝ × ℝ := (2, 3)
  let B : ℝ × ℝ := (-3, 0)
  -- Definition of the trisection points
  let T1 : ℝ × ℝ := ((2 + (-3 - 2) / 3) / 1, (3 + (0 - 3) / 3) / 1) -- First trisection point
  let T2 : ℝ × ℝ := ((2 + 2 * (-3 - 2) / 3) / 1, (3 + 2 * (0 - 3) / 3) / 1) -- Second trisection point
  -- Equation of the line through P and T2 is 3x + 7y = 17
  3 * (P.1 + P.2) + 7 * (P.2 + T2.2) = 17 :=
sorry

end NUMINAMATH_GPT_equation_of_trisection_line_l1058_105850


namespace NUMINAMATH_GPT_ratio_fraction_l1058_105843

variable (X Y Z : ℝ)
variable (k : ℝ) (hk : k > 0)

-- Given conditions
def ratio_condition := (3 * Y = 2 * X) ∧ (6 * Y = 2 * Z)

-- Statement
theorem ratio_fraction (h : ratio_condition X Y Z) : 
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_fraction_l1058_105843


namespace NUMINAMATH_GPT_fluorescent_tubes_count_l1058_105897

theorem fluorescent_tubes_count 
  (x y : ℕ)
  (h1 : x + y = 13)
  (h2 : x / 3 + y / 2 = 5) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_fluorescent_tubes_count_l1058_105897


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1058_105872

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 15) 
  (h2 : a + 11 * d = 21) :
  a + 4 * d = 0 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l1058_105872


namespace NUMINAMATH_GPT_triangle_inequality_for_f_l1058_105815

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 3 * x + m

theorem triangle_inequality_for_f (a b c m : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) 
(h₆ : 6 < m) :
  ∃ u v w, u = f a m ∧ v = f b m ∧ w = f c m ∧ u + v > w ∧ u + w > v ∧ v + w > u := 
sorry

end NUMINAMATH_GPT_triangle_inequality_for_f_l1058_105815


namespace NUMINAMATH_GPT_ploughing_problem_l1058_105820

theorem ploughing_problem
  (hours_per_day_group1 : ℕ)
  (days_group1 : ℕ)
  (bulls_group1 : ℕ)
  (total_fields_group2 : ℕ)
  (hours_per_day_group2 : ℕ)
  (days_group2 : ℕ)
  (bulls_group2 : ℕ)
  (fields_group1 : ℕ)
  (fields_group2 : ℕ) :
    hours_per_day_group1 = 10 →
    days_group1 = 3 →
    bulls_group1 = 10 →
    hours_per_day_group2 = 8 →
    days_group2 = 2 →
    bulls_group2 = 30 →
    fields_group2 = 32 →
    480 * fields_group1 = 300 * fields_group2 →
    fields_group1 = 20 := by
  sorry

end NUMINAMATH_GPT_ploughing_problem_l1058_105820


namespace NUMINAMATH_GPT_length_of_bridge_l1058_105818

def length_of_train : ℝ := 135  -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 45  -- Speed of the train in km/hr
def speed_of_train_m_per_s : ℝ := 12.5  -- Speed of the train in m/s
def time_to_cross_bridge : ℝ := 30  -- Time to cross the bridge in seconds
def distance_covered : ℝ := speed_of_train_m_per_s * time_to_cross_bridge  -- Total distance covered

theorem length_of_bridge :
  distance_covered - length_of_train = 240 :=
by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1058_105818


namespace NUMINAMATH_GPT_min_rubles_for_1001_l1058_105825

def min_rubles_needed (n : ℕ) : ℕ :=
  let side_cells := (n + 1) * 4
  let inner_cells := (n - 1) * (n - 1)
  let total := inner_cells * 4 + side_cells
  total / 2 -- since each side is shared by two cells

theorem min_rubles_for_1001 : min_rubles_needed 1001 = 503000 := by
  sorry

end NUMINAMATH_GPT_min_rubles_for_1001_l1058_105825


namespace NUMINAMATH_GPT_flower_beds_l1058_105835

theorem flower_beds (seeds_per_bed total_seeds flower_beds : ℕ) 
  (h1 : seeds_per_bed = 10) (h2 : total_seeds = 60) : 
  flower_beds = total_seeds / seeds_per_bed := by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_flower_beds_l1058_105835


namespace NUMINAMATH_GPT_dad_caught_more_trouts_l1058_105862

-- Definitions based on conditions
def caleb_trouts : ℕ := 2
def dad_trouts : ℕ := 3 * caleb_trouts

-- The proof problem: proving dad caught 4 more trouts than Caleb
theorem dad_caught_more_trouts : dad_trouts = caleb_trouts + 4 :=
by
  sorry

end NUMINAMATH_GPT_dad_caught_more_trouts_l1058_105862


namespace NUMINAMATH_GPT_preimage_of_8_is_5_image_of_8_is_64_l1058_105846

noncomputable def f (x : ℝ) : ℝ := 2^(x - 2)

theorem preimage_of_8_is_5 : ∃ x, f x = 8 := by
  use 5
  sorry

theorem image_of_8_is_64 : f 8 = 64 := by
  sorry

end NUMINAMATH_GPT_preimage_of_8_is_5_image_of_8_is_64_l1058_105846


namespace NUMINAMATH_GPT_train_length_is_correct_l1058_105848

noncomputable def convert_speed (speed_kmh : ℕ) : ℝ :=
  (speed_kmh : ℝ) * 5 / 18

noncomputable def relative_speed (train_speed_kmh man's_speed_kmh : ℕ) : ℝ :=
  convert_speed train_speed_kmh + convert_speed man's_speed_kmh

noncomputable def length_of_train (train_speed_kmh man's_speed_kmh : ℕ) (time_seconds : ℝ) : ℝ := 
  relative_speed train_speed_kmh man's_speed_kmh * time_seconds

theorem train_length_is_correct :
  length_of_train 60 6 29.997600191984645 = 550 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1058_105848


namespace NUMINAMATH_GPT_two_legged_birds_count_l1058_105837

-- Definitions and conditions
variables {x y z : ℕ}
variables (heads_eq : x + y + z = 200) (legs_eq : 2 * x + 3 * y + 4 * z = 558)

-- The statement to prove
theorem two_legged_birds_count : x = 94 :=
sorry

end NUMINAMATH_GPT_two_legged_birds_count_l1058_105837


namespace NUMINAMATH_GPT_correct_factorization_l1058_105812

theorem correct_factorization :
  (∀ a b : ℝ, ¬ (a^2 + b^2 = (a + b) * (a - b))) ∧
  (∀ a : ℝ, ¬ (a^4 - 1 = (a^2 + 1) * (a^2 - 1))) ∧
  (∀ x : ℝ, ¬ (x^2 + 2 * x + 4 = (x + 2)^2)) ∧
  (∀ x : ℝ, x^2 - 3 * x + 2 = (x - 1) * (x - 2)) :=
by
  sorry

end NUMINAMATH_GPT_correct_factorization_l1058_105812


namespace NUMINAMATH_GPT_subtraction_of_negatives_l1058_105874

theorem subtraction_of_negatives : (-7) - (-5) = -2 := 
by {
  -- sorry replaces the actual proof steps.
  sorry
}

end NUMINAMATH_GPT_subtraction_of_negatives_l1058_105874


namespace NUMINAMATH_GPT_constant_for_odd_m_l1058_105845

theorem constant_for_odd_m (constant : ℝ) (f : ℕ → ℝ)
  (h1 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k + 1) → f m = constant * m)
  (h2 : ∀ m : ℕ, m > 0 → (∃ k : ℕ, m = 2 * k) → f m = (1/2 : ℝ) * m)
  (h3 : f 5 * f 6 = 15) : constant = 1 :=
by
  sorry

end NUMINAMATH_GPT_constant_for_odd_m_l1058_105845


namespace NUMINAMATH_GPT_find_r_amount_l1058_105875

theorem find_r_amount (p q r : ℝ) (h_total : p + q + r = 8000) (h_r_fraction : r = 2 / 3 * (p + q)) : r = 3200 :=
by 
  -- Proof is not required, hence we use sorry
  sorry

end NUMINAMATH_GPT_find_r_amount_l1058_105875


namespace NUMINAMATH_GPT_triangle_area_l1058_105804

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ :=
(x, y, z)

noncomputable def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.2.1 * w.2.2 - v.2.2 * w.2.1,
 v.2.2 * w.1 - v.1 * w.2.2,
 v.1 * w.2.1 - v.2.1 * w.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem triangle_area :
  let A := vector 2 1 (-1)
  let B := vector 3 0 3
  let C := vector 7 3 2
  let AB := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
  let AC := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)
  0.5 * magnitude (cross_product AB AC) = (1 / 2) * Real.sqrt 459 :=
by
  -- All the steps needed to prove the theorem here
  sorry

end NUMINAMATH_GPT_triangle_area_l1058_105804


namespace NUMINAMATH_GPT_min_c_value_l1058_105863

theorem min_c_value 
  (a b c d e : ℕ) 
  (h1 : a < b)
  (h2 : b < c)
  (h3 : c < d)
  (h4 : d < e)
  (h5 : a + 1 = b)
  (h6 : b + 1 = c)
  (h7 : c + 1 = d)
  (h8 : d + 1 = e)
  (h9 : ∃ k : ℕ, k ^ 2 = b + c + d)
  (h10 : ∃ m : ℕ, m ^ 3 = a + b + c + d + e) : 
  c = 675 := 
sorry

end NUMINAMATH_GPT_min_c_value_l1058_105863


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_six_l1058_105885

open Nat

noncomputable def sum_first_six_terms (a : ℕ → ℚ) : ℚ :=
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  3 * (2 * a1 + 5 * d) / 3

theorem arithmetic_sequence_sum_six (a : ℕ → ℚ) (h : a 2 + a 5 = 2 / 3) : sum_first_six_terms a = 2 :=
by
  let a1 : ℚ := a 1
  let d : ℚ := a 2 - a1
  have eq1 : a 5 = a1 + 4 * d := by sorry
  have eq2 : 3 * (2 * a1 + 5 * d) / 3 = (2 : ℚ) := by sorry
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_six_l1058_105885


namespace NUMINAMATH_GPT_largest_k_for_right_triangle_l1058_105866

noncomputable def k : ℝ := (3 * Real.sqrt 2 - 4) / 2

theorem largest_k_for_right_triangle (a b c : ℝ) (h : c^2 = a^2 + b^2) :
    a^3 + b^3 + c^3 ≥ k * (a + b + c)^3 :=
sorry

end NUMINAMATH_GPT_largest_k_for_right_triangle_l1058_105866


namespace NUMINAMATH_GPT_steve_speed_back_home_l1058_105827

-- Definitions based on conditions
def distance := 20 -- distance from house to work in km
def total_time := 6 -- total time on the road in hours
def speed_to_work (v : ℝ) := v -- speed to work in km/h
def speed_back_home (v : ℝ) := 2 * v -- speed back home in km/h

-- Theorem to assert the proof
theorem steve_speed_back_home (v : ℝ) (h : distance / v + distance / (2 * v) = total_time) :
  speed_back_home v = 10 := by
  -- Proof goes here but we just state sorry to skip it
  sorry

end NUMINAMATH_GPT_steve_speed_back_home_l1058_105827


namespace NUMINAMATH_GPT_polynomial_root_condition_l1058_105895

noncomputable def polynomial_q (q x : ℝ) : ℝ :=
  x^6 + 3 * q * x^4 + 3 * x^4 + 3 * q * x^2 + x^2 + 3 * q + 1

theorem polynomial_root_condition (q : ℝ) :
  (∃ x > 0, polynomial_q q x = 0) ↔ (q ≥ 3 / 2) :=
sorry

end NUMINAMATH_GPT_polynomial_root_condition_l1058_105895


namespace NUMINAMATH_GPT_roses_in_vase_now_l1058_105878

-- Definitions of initial conditions and variables
def initial_roses : ℕ := 12
def initial_orchids : ℕ := 2
def orchids_cut : ℕ := 19
def orchids_now : ℕ := 21

-- The proof problem to show that the number of roses now is still the same as initially.
theorem roses_in_vase_now : initial_roses = 12 :=
by
  -- The proof itself is left as an exercise (add proof here)
  sorry

end NUMINAMATH_GPT_roses_in_vase_now_l1058_105878


namespace NUMINAMATH_GPT_maximum_x_plus_7y_exists_Q_locus_l1058_105800

noncomputable def Q_locus (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem maximum_x_plus_7y (M : ℝ × ℝ) (h : Q_locus M.fst M.snd) : 
  ∃ max_value, max_value = 18 :=
  sorry

theorem exists_Q_locus (x y : ℝ) : 
  (∃ (Q : ℝ × ℝ), Q_locus Q.fst Q.snd) :=
  sorry

end NUMINAMATH_GPT_maximum_x_plus_7y_exists_Q_locus_l1058_105800
