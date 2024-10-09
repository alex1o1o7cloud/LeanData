import Mathlib

namespace bottle_and_beverage_weight_l1008_100815

theorem bottle_and_beverage_weight 
  (B : ℝ)  -- Weight of the bottle in kilograms
  (x : ℝ)  -- Original weight of the beverage in kilograms
  (h1 : B + 2 * x = 5)  -- Condition: double the beverage weight total
  (h2 : B + 4 * x = 9)  -- Condition: quadruple the beverage weight total
: x = 2 ∧ B = 1 := 
by
  sorry

end bottle_and_beverage_weight_l1008_100815


namespace find_y_l1008_100863

theorem find_y (y : ℕ) (hy1 : y % 9 = 0) (hy2 : y^2 > 200) (hy3 : y < 30) : y = 18 :=
sorry

end find_y_l1008_100863


namespace smallest_pos_int_ending_in_9_divisible_by_13_l1008_100854

theorem smallest_pos_int_ending_in_9_divisible_by_13 :
  ∃ (n : ℕ), n % 10 = 9 ∧ n % 13 = 0 ∧ ∀ m, m % 10 = 9 ∧ m % 13 = 0 → n ≤ m :=
by
  sorry

end smallest_pos_int_ending_in_9_divisible_by_13_l1008_100854


namespace quad_factor_value_l1008_100801

theorem quad_factor_value (c d : ℕ) (h1 : c + d = 14) (h2 : c * d = 40) (h3 : c > d) : 4 * d - c = 6 :=
sorry

end quad_factor_value_l1008_100801


namespace profit_eqn_65_to_75_maximize_profit_with_discount_l1008_100892

-- Definitions for the conditions
def total_pieces (x y : ℕ) : Prop := x + y = 100

def total_cost (x y : ℕ) : Prop := 80 * x + 60 * y ≤ 7500

def min_pieces_A (x : ℕ) : Prop := x ≥ 65

def profit_without_discount (x : ℕ) : ℕ := 10 * x + 3000

def profit_with_discount (x a : ℕ) (h1 : 0 < a) (h2 : a < 20): ℕ := (10 - a) * x + 3000

-- Proof statement
theorem profit_eqn_65_to_75 (x: ℕ) (h1: total_pieces x (100 - x)) (h2: total_cost x (100 - x)) (h3: min_pieces_A x) :
  65 ≤ x ∧ x ≤ 75 → profit_without_discount x = 10 * x + 3000 :=
by
  sorry

theorem maximize_profit_with_discount (x a : ℕ) (h1 : total_pieces x (100 - x)) (h2 : total_cost x (100 - x)) (h3 : min_pieces_A x) (h4 : 0 < a) (h5 : a < 20) :
  if a < 10 then x = 75 ∧ profit_with_discount 75 a h4 h5 = (10 - a) * 75 + 3000
  else if a = 10 then 65 ≤ x ∧ x ≤ 75 ∧ profit_with_discount x a h4 h5 = 3000
  else x = 65 ∧ profit_with_discount 65 a h4 h5 = (10 - a) * 65 + 3000 :=
by
  sorry

end profit_eqn_65_to_75_maximize_profit_with_discount_l1008_100892


namespace find_y_l1008_100818

noncomputable def a := (3/5) * 2500
noncomputable def b := (2/7) * ((5/8) * 4000 + (1/4) * 3600 - (11/20) * 7200)
noncomputable def c (y : ℚ) := (3/10) * y
def result (a b c : ℚ) := a * b / c

theorem find_y : ∃ y : ℚ, result a b (c y) = 25000 ∧ y = -4/21 := 
by
  sorry

end find_y_l1008_100818


namespace abs_inequality_solution_l1008_100810

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 2| + |x + 3| < 8) → x ∈ Set.Ioo (-4.5) (3.5) :=
by
  sorry

end abs_inequality_solution_l1008_100810


namespace total_animals_hunted_l1008_100859

theorem total_animals_hunted :
  let sam_hunts := 6
  let rob_hunts := sam_hunts / 2
  let total_sam_rob := sam_hunts + rob_hunts
  let mark_hunts := total_sam_rob / 3
  let peter_hunts := mark_hunts * 3
  sam_hunts + rob_hunts + mark_hunts + peter_hunts = 21 :=
by
  sorry

end total_animals_hunted_l1008_100859


namespace square_value_zero_l1008_100874

variable {a b : ℝ}

theorem square_value_zero (h1 : a > b) (h2 : -2 * a - 1 < -2 * b + 0) : 0 = 0 := 
by
  sorry

end square_value_zero_l1008_100874


namespace ramu_profit_percent_is_21_64_l1008_100845

-- Define the costs and selling price as constants
def cost_of_car : ℕ := 42000
def cost_of_repairs : ℕ := 13000
def selling_price : ℕ := 66900

-- Define the total cost and profit
def total_cost : ℕ := cost_of_car + cost_of_repairs
def profit : ℕ := selling_price - total_cost

-- Define the profit percent formula
def profit_percent : ℚ := ((profit : ℚ) / (total_cost : ℚ)) * 100

-- State the theorem we want to prove
theorem ramu_profit_percent_is_21_64 : profit_percent = 21.64 := by
  sorry

end ramu_profit_percent_is_21_64_l1008_100845


namespace part1_part2_l1008_100896

def quadratic_inequality_A (x m : ℝ) := -x^2 + 2 * m * x + 4 - m^2 ≥ 0
def quadratic_inequality_B (x : ℝ) := 2 * x^2 - 5 * x - 7 < 0

theorem part1 (m : ℝ) :
  (∀ x, quadratic_inequality_A x m ∧ quadratic_inequality_B x ↔ 0 ≤ x ∧ x < 7 / 2) →
  m = 2 := by sorry

theorem part2 (m : ℝ) :
  (∀ x, quadratic_inequality_B x → ¬ quadratic_inequality_A x m) →
  m ≤ -3 ∨ 11 / 2 ≤ m := by sorry

end part1_part2_l1008_100896


namespace polygon_interior_angles_l1008_100828

theorem polygon_interior_angles {n : ℕ} (h : (n - 2) * 180 = 900) : n = 7 :=
by
  sorry

end polygon_interior_angles_l1008_100828


namespace range_of_squared_sum_l1008_100839

theorem range_of_squared_sum (x y : ℝ) (h : x^2 + 1 / y^2 = 2) : ∃ z, z = x^2 + y^2 ∧ z ≥ 1 / 2 :=
by
  sorry

end range_of_squared_sum_l1008_100839


namespace general_formula_l1008_100808

theorem general_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, a (n + 1) = 2^(n + 1) - 1 :=
by
  sorry

end general_formula_l1008_100808


namespace polynomial_value_l1008_100856

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by sorry

end polynomial_value_l1008_100856


namespace train_crossing_time_l1008_100877

theorem train_crossing_time
  (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ)
  (train_length_eq : length_train = 720)
  (bridge_length_eq : length_bridge = 320)
  (speed_eq : speed_kmh = 90) :
  (length_train + length_bridge) / (speed_kmh * (1000 / 3600)) = 41.6 := by
  sorry

end train_crossing_time_l1008_100877


namespace infinite_x_differs_from_two_kth_powers_l1008_100827

theorem infinite_x_differs_from_two_kth_powers (k : ℕ) (h : k > 1) : 
  ∃ (f : ℕ → ℕ), (∀ n, f n = (2^(n+1))^k - (2^n)^k) ∧ (∀ n, ∀ a b : ℕ, ¬ f n = a^k + b^k) :=
sorry

end infinite_x_differs_from_two_kth_powers_l1008_100827


namespace combined_stickers_l1008_100840

def initial_stickers_june : ℕ := 76
def initial_stickers_bonnie : ℕ := 63
def birthday_stickers : ℕ := 25

theorem combined_stickers : 
  (initial_stickers_june + birthday_stickers) + (initial_stickers_bonnie + birthday_stickers) = 189 := 
by
  sorry

end combined_stickers_l1008_100840


namespace number_of_ways_to_cut_pipe_l1008_100803

theorem number_of_ways_to_cut_pipe : 
  (∃ (x y: ℕ), 2 * x + 3 * y = 15) ∧ 
  (∃! (x y: ℕ), 2 * x + 3 * y = 15) :=
by
  sorry

end number_of_ways_to_cut_pipe_l1008_100803


namespace units_digit_b_l1008_100880

theorem units_digit_b (a b : ℕ) (h1 : a % 10 = 9) (h2 : a * b = 34^8) : b % 10 = 4 :=
by
  sorry

end units_digit_b_l1008_100880


namespace profit_ratio_l1008_100891

theorem profit_ratio (SP CP : ℝ) (h : SP / CP = 3) : (SP - CP) / CP = 2 :=
by
  sorry

end profit_ratio_l1008_100891


namespace xyz_sum_l1008_100834

theorem xyz_sum (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y) : x + y + z = 16 * x :=
by
  sorry

end xyz_sum_l1008_100834


namespace find_a_of_exponential_inverse_l1008_100804

theorem find_a_of_exponential_inverse (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) (h₂ : ∀ x, a^x = 9 ↔ x = 2) : a = 3 := 
by
  sorry

end find_a_of_exponential_inverse_l1008_100804


namespace bc_over_ad_eq_50_point_4_l1008_100890

theorem bc_over_ad_eq_50_point_4 :
  let B := (2, 2, 5)
  let S (r : ℝ) (B : ℝ × ℝ × ℝ) := {p | dist p B ≤ r }
  let d := (20 : ℝ)
  let c := (48 : ℝ)
  let b := (28 * Real.pi : ℝ)
  let a := ((4 * Real.pi) / 3 : ℝ)
  let bc := b * c
  let ad := a * d
  bc / ad = 50.4 := by
    sorry

end bc_over_ad_eq_50_point_4_l1008_100890


namespace percentage_commute_l1008_100893

variable (x : Real)
variable (h : 0.20 * 0.10 * x = 12)

theorem percentage_commute :
  0.10 * 0.20 * x = 12 :=
by
  sorry

end percentage_commute_l1008_100893


namespace lcm_5_6_8_9_l1008_100806

theorem lcm_5_6_8_9 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := by
  sorry

end lcm_5_6_8_9_l1008_100806


namespace picnic_problem_l1008_100820

variables (M W C A : ℕ)

theorem picnic_problem
  (H1 : M + W + C = 200)
  (H2 : A = C + 20)
  (H3 : M = 65)
  (H4 : A = M + W) :
  M - W = 20 :=
by sorry

end picnic_problem_l1008_100820


namespace ratio_of_fixing_times_is_two_l1008_100825

noncomputable def time_per_shirt : ℝ := 1.5
noncomputable def number_of_shirts : ℕ := 10
noncomputable def number_of_pants : ℕ := 12
noncomputable def hourly_rate : ℝ := 30
noncomputable def total_cost : ℝ := 1530

theorem ratio_of_fixing_times_is_two :
  let total_hours := total_cost / hourly_rate
  let shirt_hours := number_of_shirts * time_per_shirt
  let pant_hours := total_hours - shirt_hours
  let time_per_pant := pant_hours / number_of_pants
  (time_per_pant / time_per_shirt) = 2 :=
by
  sorry

end ratio_of_fixing_times_is_two_l1008_100825


namespace simplify_fraction_l1008_100817

noncomputable def a : ℝ := 2
noncomputable def b : ℝ := Real.sqrt 3

theorem simplify_fraction :
    (1 / (a + b)) * (1 / (a - b)) = 1 := by
  sorry

end simplify_fraction_l1008_100817


namespace find_N_l1008_100829

theorem find_N (N p q : ℝ) 
  (h1 : N / p = 4) 
  (h2 : N / q = 18) 
  (h3 : p - q = 0.5833333333333334) :
  N = 3 := 
sorry

end find_N_l1008_100829


namespace inequality_solution_set_l1008_100862

theorem inequality_solution_set :
  {x : ℝ | (3 - x) * (1 + x) > 0} = {x : ℝ | -1 < x ∧ x < 3} :=
sorry

end inequality_solution_set_l1008_100862


namespace problem_proof_l1008_100832

def delta (a b : ℕ) : ℕ := a^2 + b

theorem problem_proof :
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  delta u v = 5^88 + 7^18 :=
by
  let x := 6
  let y := 8
  let z := 4
  let w := 2
  let u := 5^delta x y
  let v := 7^delta z w
  have h1: delta x y = 44 := by sorry
  have h2: delta z w = 18 := by sorry
  have hu: u = 5^44 := by sorry
  have hv: v = 7^18 := by sorry
  have hdelta: delta u v = 5^88 + 7^18 := by sorry
  exact hdelta

end problem_proof_l1008_100832


namespace pipe_cistern_problem_l1008_100823

theorem pipe_cistern_problem:
  ∀ (rate_p rate_q : ℝ),
    rate_p = 1 / 10 →
    rate_q = 1 / 15 →
    ∀ (filled_in_4_minutes : ℝ),
      filled_in_4_minutes = 4 * (rate_p + rate_q) →
      ∀ (remaining : ℝ),
        remaining = 1 - filled_in_4_minutes →
        ∀ (time_to_fill : ℝ),
          time_to_fill = remaining / rate_q →
          time_to_fill = 5 :=
by
  intros rate_p rate_q Hp Hq filled_in_4_minutes H4 remaining Hr time_to_fill Ht
  sorry

end pipe_cistern_problem_l1008_100823


namespace range_of_k_l1008_100816

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - x

theorem range_of_k :
  (∀ (x₁ x₂ : ℝ), x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → |f x₁ - f x₂| ≤ k) → k ≥ Real.exp 1 - 1 :=
by
  sorry

end range_of_k_l1008_100816


namespace theo_eggs_needed_l1008_100838

def customers_first_hour : ℕ := 5
def customers_second_hour : ℕ := 7
def customers_third_hour : ℕ := 3
def customers_fourth_hour : ℕ := 8
def eggs_per_3_egg_omelette : ℕ := 3
def eggs_per_4_egg_omelette : ℕ := 4

theorem theo_eggs_needed :
  (customers_first_hour * eggs_per_3_egg_omelette) +
  (customers_second_hour * eggs_per_4_egg_omelette) +
  (customers_third_hour * eggs_per_3_egg_omelette) +
  (customers_fourth_hour * eggs_per_4_egg_omelette) = 84 := by
  sorry

end theo_eggs_needed_l1008_100838


namespace age_difference_l1008_100858

theorem age_difference 
  (a b : ℕ) 
  (h1 : 0 ≤ a ∧ a < 10) 
  (h2 : 0 ≤ b ∧ b < 10) 
  (h3 : 10 * a + b + 5 = 3 * (10 * b + a + 5)) : 
  (10 * a + b) - (10 * b + a) = 63 := 
by
  sorry

end age_difference_l1008_100858


namespace A_walking_speed_l1008_100807

-- Definition for the conditions
def A_speed (v : ℝ) : Prop := 
  ∃ (t : ℝ), 120 = 20 * (t - 6) ∧ 120 = v * t

-- The main theorem to prove the question
theorem A_walking_speed : ∀ (v : ℝ), A_speed v → v = 10 :=
by
  intros v h
  sorry

end A_walking_speed_l1008_100807


namespace not_perfect_square_l1008_100894

open Nat

theorem not_perfect_square (m n : ℕ) : ¬∃ k : ℕ, k^2 = 1 + 3^m + 3^n :=
by
  sorry

end not_perfect_square_l1008_100894


namespace term_2005_is_1004th_l1008_100879

-- Define the first term and the common difference
def a1 : Int := -1
def d : Int := 2

-- Define the general term formula of the arithmetic sequence
def a_n (n : Nat) : Int :=
  a1 + (n - 1) * d

-- State the theorem that the year 2005 is the 1004th term in the sequence
theorem term_2005_is_1004th : ∃ n : Nat, a_n n = 2005 ∧ n = 1004 := by
  sorry

end term_2005_is_1004th_l1008_100879


namespace percentage_deposit_l1008_100812

theorem percentage_deposit (deposited : ℝ) (initial_amount : ℝ) (amount_deposited : ℝ) (P : ℝ) 
  (h1 : deposited = 750) 
  (h2 : initial_amount = 50000)
  (h3 : amount_deposited = 0.20 * (P / 100) * (0.25 * initial_amount))
  (h4 : amount_deposited = deposited) : 
  P = 30 := 
sorry

end percentage_deposit_l1008_100812


namespace area_of_OBEC_is_25_l1008_100819

noncomputable def area_OBEC : ℝ :=
  let A := (20 / 3, 0)
  let B := (0, 20)
  let C := (10, 0)
  let E := (5, 5)
  let O := (0, 0)
  let area_triangle (P Q R : ℝ × ℝ) : ℝ :=
    (1 / 2) * abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2))
  area_triangle O B E - area_triangle O E C

theorem area_of_OBEC_is_25 :
  area_OBEC = 25 := 
by
  sorry

end area_of_OBEC_is_25_l1008_100819


namespace water_percentage_in_fresh_mushrooms_l1008_100872

theorem water_percentage_in_fresh_mushrooms
  (fresh_mushrooms_mass : ℝ)
  (dried_mushrooms_mass : ℝ)
  (dried_mushrooms_water_percentage : ℝ)
  (dried_mushrooms_non_water_mass : ℝ)
  (fresh_mushrooms_dry_percentage : ℝ)
  (fresh_mushrooms_water_percentage : ℝ)
  (h1 : fresh_mushrooms_mass = 22)
  (h2 : dried_mushrooms_mass = 2.5)
  (h3 : dried_mushrooms_water_percentage = 12 / 100)
  (h4 : dried_mushrooms_non_water_mass = dried_mushrooms_mass * (1 - dried_mushrooms_water_percentage))
  (h5 : fresh_mushrooms_dry_percentage = dried_mushrooms_non_water_mass / fresh_mushrooms_mass * 100)
  (h6 : fresh_mushrooms_water_percentage = 100 - fresh_mushrooms_dry_percentage) :
  fresh_mushrooms_water_percentage = 90 := 
by
  sorry

end water_percentage_in_fresh_mushrooms_l1008_100872


namespace simplify_expr_l1008_100844

theorem simplify_expr (x : ℝ) : 
  2 * x * (4 * x ^ 3 - 3 * x + 1) - 7 * (x ^ 3 - x ^ 2 + 3 * x - 4) = 
  8 * x ^ 4 - 7 * x ^ 3 + x ^ 2 - 19 * x + 28 := 
by
  sorry

end simplify_expr_l1008_100844


namespace completing_the_square_l1008_100853

theorem completing_the_square {x : ℝ} : x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := 
sorry

end completing_the_square_l1008_100853


namespace hall_mat_expenditure_l1008_100868

theorem hall_mat_expenditure
  (length width height cost_per_sq_meter : ℕ)
  (H_length : length = 20)
  (H_width : width = 15)
  (H_height : height = 5)
  (H_cost_per_sq_meter : cost_per_sq_meter = 50) :
  (2 * (length * width) + 2 * (length * height) + 2 * (width * height)) * cost_per_sq_meter = 47500 :=
by
  sorry

end hall_mat_expenditure_l1008_100868


namespace time_for_pipe_a_to_fill_l1008_100899

noncomputable def pipe_filling_time (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) : ℝ := 
  (1 / a_rate)

theorem time_for_pipe_a_to_fill (a_rate b_rate c_rate : ℝ) (fill_time_together : ℝ) 
  (h1 : b_rate = 2 * a_rate) 
  (h2 : c_rate = 2 * b_rate) 
  (h3 : (a_rate + b_rate + c_rate) * fill_time_together = 1) : 
  pipe_filling_time a_rate b_rate c_rate fill_time_together = 42 :=
sorry

end time_for_pipe_a_to_fill_l1008_100899


namespace last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l1008_100847

theorem last_number_of_nth_row (n : ℕ) : 
    let last_number := 2^n - 1
    last_number = 2^n - 1 := 
sorry

theorem sum_of_numbers_in_nth_row (n : ℕ) :
    let sum := (3 * 2^(n-3)) - 2^(n-2)
    sum = (3 * 2^(n-3)) - 2^(n-2) :=
sorry

theorem position_of_2008 : 
    let position := 985
    position = 985 :=
sorry

end last_number_of_nth_row_sum_of_numbers_in_nth_row_position_of_2008_l1008_100847


namespace analytical_expression_f_l1008_100870

def f : ℝ → ℝ := sorry

theorem analytical_expression_f :
  (∀ x : ℝ, f (x + 2) = x^2 - x + 1) →
  (∀ y : ℝ, f y = y^2 - 5*y + 7) :=
by
  sorry

end analytical_expression_f_l1008_100870


namespace complement_set_unique_l1008_100866

-- Define the universal set U
def U : Set ℕ := {1,2,3,4,5,6,7,8}

-- Define the complement of B with respect to U
def complement_B : Set ℕ := {1,3}

-- The set B that we need to prove
def B : Set ℕ := {2,4,5,6,7,8}

-- State that B is the set with the given complement in U
theorem complement_set_unique (U : Set ℕ) (complement_B : Set ℕ) :
    (U \ complement_B = {2,4,5,6,7,8}) :=
by
    -- We need to prove B is the set {2,4,5,6,7,8}
    sorry

end complement_set_unique_l1008_100866


namespace find_k_l1008_100821

theorem find_k
  (t k : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : t = 20) :
  k = 68 := 
by
  sorry

end find_k_l1008_100821


namespace steven_apples_minus_peaches_l1008_100888

-- Define the number of apples and peaches Steven has.
def steven_apples : ℕ := 19
def steven_peaches : ℕ := 15

-- Problem statement: Prove that the number of apples minus the number of peaches is 4.
theorem steven_apples_minus_peaches : steven_apples - steven_peaches = 4 := by
  sorry

end steven_apples_minus_peaches_l1008_100888


namespace max_tan_alpha_l1008_100835

theorem max_tan_alpha (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h : Real.tan (α + β) = 9 * Real.tan β) : Real.tan α ≤ 4 / 3 :=
by
  sorry

end max_tan_alpha_l1008_100835


namespace problem_prove_ω_and_delta_l1008_100861

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem problem_prove_ω_and_delta (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) 
    (h_sym_axis : ∀ x, f ω φ x = f ω φ (-(x + π))) 
    (h_center_sym : ∃ c : ℝ, (c = π / 2) ∧ (f ω φ c = 0)) 
    (h_monotone_increasing : ∀ x, -π ≤ x ∧ x ≤ -π / 2 → f ω φ x < f ω φ (x + 1)) :
    (ω = 1 / 3) ∧ (∀ δ : ℝ, (∀ x : ℝ, f ω φ (x + δ) = f ω φ (-x + δ)) → ∃ k : ℤ, δ = 2 * π + 3 * k * π) :=
by
  sorry

end problem_prove_ω_and_delta_l1008_100861


namespace new_person_weight_l1008_100867

theorem new_person_weight (weights : List ℝ) (len_weights : weights.length = 8) (replace_weight : ℝ) (new_weight : ℝ)
  (weight_diff :  (weights.sum - replace_weight + new_weight) / 8 = (weights.sum / 8) + 3) 
  (replace_weight_eq : replace_weight = 70):
  new_weight = 94 :=
sorry

end new_person_weight_l1008_100867


namespace scientific_notation_80000000_l1008_100884

-- Define the given number
def number : ℕ := 80000000

-- Define the scientific notation form
def scientific_notation (n k : ℕ) (a : ℝ) : Prop :=
  n = (a * (10 : ℝ) ^ k)

-- The theorem to prove scientific notation of 80,000,000
theorem scientific_notation_80000000 : scientific_notation number 7 8 :=
by {
  sorry
}

end scientific_notation_80000000_l1008_100884


namespace pool_capacity_l1008_100836

theorem pool_capacity
  (pump_removes : ∀ (x : ℝ), x > 0 → (2 / 3) * x / 7.5 = (4 / 15) * x)
  (working_time : 0.15 * 60 = 9)
  (remaining_water : ∀ (x : ℝ), x > 0 → x - (0.8 * x) = 25) :
  ∃ x : ℝ, x = 125 :=
by
  sorry

end pool_capacity_l1008_100836


namespace unique_acute_triangulation_l1008_100865

-- Definitions for the proof problem
def is_convex (polygon : Type) : Prop := sorry
def is_acute_triangle (triangle : Type) : Prop := sorry
def is_triangulation (polygon : Type) (triangulation : List Type) : Prop := sorry
def is_acute_triangulation (polygon : Type) (triangulation : List Type) : Prop :=
  is_triangulation polygon triangulation ∧ ∀ triangle ∈ triangulation, is_acute_triangle triangle

-- Proposition to be proved
theorem unique_acute_triangulation (n : ℕ) (polygon : Type) 
  (h₁ : is_convex polygon) (h₂ : n ≥ 3) :
  ∃! triangulation : List Type, is_acute_triangulation polygon triangulation := 
sorry

end unique_acute_triangulation_l1008_100865


namespace squared_expression_l1008_100841

variable (x : ℝ)

theorem squared_expression (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := by
  sorry

end squared_expression_l1008_100841


namespace quadratic_points_range_l1008_100833

theorem quadratic_points_range (a : ℝ) (y1 y2 y3 y4 : ℝ) :
  (∀ (x : ℝ), 
    (x = -4 → y1 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = -3 → y2 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 0 → y3 = a * x^2 + 4 * a * x - 6) ∧ 
    (x = 2 → y4 = a * x^2 + 4 * a * x - 6)) →
  (∃! (y : ℝ), y > 0 ∧ (y = y1 ∨ y = y2 ∨ y = y3 ∨ y = y4)) →
  (a < -2 ∨ a > 1 / 2) :=
by
  sorry

end quadratic_points_range_l1008_100833


namespace goods_purchase_solutions_l1008_100809

theorem goods_purchase_solutions (a : ℕ) (h1 : 0 < a ∧ a ≤ 45) :
  ∃ x : ℝ, 45 - 20 * (x - 1) = a * x :=
by sorry

end goods_purchase_solutions_l1008_100809


namespace yard_length_l1008_100889

theorem yard_length :
  let num_trees := 11
  let distance_between_trees := 18
  (num_trees - 1) * distance_between_trees = 180 :=
by
  let num_trees := 11
  let distance_between_trees := 18
  sorry

end yard_length_l1008_100889


namespace quadratic_grid_fourth_column_l1008_100850

theorem quadratic_grid_fourth_column 
  (grid : ℕ → ℕ → ℝ)
  (row_quadratic : ∀ i : ℕ, (∃ a b c : ℝ, ∀ n : ℕ, grid i n = a * n^2 + b * n + c))
  (col_quadratic : ∀ j : ℕ, j ≤ 3 → (∃ a b c : ℝ, ∀ n : ℕ, grid n j = a * n^2 + b * n + c)) :
  ∃ a b c : ℝ, ∀ n : ℕ, grid n 4 = a * n^2 + b * n + c := 
sorry

end quadratic_grid_fourth_column_l1008_100850


namespace determine_f_l1008_100800

theorem determine_f (d e f : ℝ) 
  (h_eq : ∀ y : ℝ, (-3) = d * y^2 + e * y + f)
  (h_vertex : ∀ k : ℝ, -1 = d * (3 - k)^2 + e * (3 - k) + f) :
  f = -5 / 2 :=
sorry

end determine_f_l1008_100800


namespace total_tickets_sold_l1008_100813

theorem total_tickets_sold (x y : ℕ) (h1 : 12 * x + 8 * y = 3320) (h2 : y = x + 240) : 
  x + y = 380 :=
by -- proof
  sorry

end total_tickets_sold_l1008_100813


namespace solve_equation_l1008_100898

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (3 - x^2) / (x + 2) + (2 * x^2 - 8) / (x^2 - 4) = 3 ↔ 
  x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
by
  sorry

end solve_equation_l1008_100898


namespace juniors_involved_in_sports_l1008_100822

theorem juniors_involved_in_sports 
    (total_students : ℕ) (percentage_juniors : ℝ) (percentage_sports : ℝ) 
    (H1 : total_students = 500) 
    (H2 : percentage_juniors = 0.40) 
    (H3 : percentage_sports = 0.70) : 
    total_students * percentage_juniors * percentage_sports = 140 := 
by
  sorry

end juniors_involved_in_sports_l1008_100822


namespace simplify_and_evaluate_l1008_100878

theorem simplify_and_evaluate (x : ℝ) (h : x^2 - 3*x - 2 = 0) :
  (x + 1) * (x - 1) - (x + 3)^2 + 2 * x^2 = -6 := 
by {
  sorry
}

end simplify_and_evaluate_l1008_100878


namespace exists_a_satisfying_f_l1008_100837

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + 1 else x - 1

theorem exists_a_satisfying_f (a : ℝ) : 
  f (a + 1) = f a ↔ (a = -1/2 ∨ a = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end exists_a_satisfying_f_l1008_100837


namespace age_proof_l1008_100882

theorem age_proof (A B C D : ℕ) 
  (h1 : A = D + 16)
  (h2 : B = D + 8)
  (h3 : C = D + 4)
  (h4 : A - 6 = 3 * (D - 6))
  (h5 : A - 6 = 2 * (B - 6))
  (h6 : A - 6 = (C - 6) + 4) 
  : A = 30 ∧ B = 22 ∧ C = 18 ∧ D = 14 :=
sorry

end age_proof_l1008_100882


namespace simplify_fraction_l1008_100814

variable {x y : ℝ}

theorem simplify_fraction (h : x ≠ y) : (x^6 - y^6) / (x^3 - y^3) = x^3 + y^3 := by
  sorry

end simplify_fraction_l1008_100814


namespace range_of_b_l1008_100875

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) := x^2 + b * x + c

def A (b c : ℝ) := {x : ℝ | f x b c = 0}
def B (b c : ℝ) := {x : ℝ | f (f x b c) b c = 0}

theorem range_of_b (b c : ℝ) (h : ∃ x₀ : ℝ, x₀ ∈ B b c ∧ x₀ ∉ A b c) :
  b < 0 ∨ b ≥ 4 := 
sorry

end range_of_b_l1008_100875


namespace electronic_items_stock_l1008_100842

-- Define the base statements
def all_in_stock (S : Type) (p : S → Prop) : Prop := ∀ x, p x
def some_not_in_stock (S : Type) (p : S → Prop) : Prop := ∃ x, ¬ p x

-- Define the main theorem statement
theorem electronic_items_stock (S : Type) (p : S → Prop) :
  ¬ all_in_stock S p → some_not_in_stock S p :=
by
  intros
  sorry

end electronic_items_stock_l1008_100842


namespace total_bees_approx_l1008_100871

-- Define a rectangular garden with given width and length
def garden_width : ℝ := 450
def garden_length : ℝ := 550

-- Define the average density of bees per square foot
def bee_density : ℝ := 2.5

-- Define the area of the garden in square feet
def garden_area : ℝ := garden_width * garden_length

-- Define the total number of bees in the garden
def total_bees : ℝ := bee_density * garden_area

-- Prove that the total number of bees approximately equals 620,000
theorem total_bees_approx : abs (total_bees - 620000) < 1000 :=
by
  sorry

end total_bees_approx_l1008_100871


namespace total_toothpicks_needed_l1008_100851

theorem total_toothpicks_needed (length width : ℕ) (hl : length = 50) (hw : width = 40) : 
  (length + 1) * width + (width + 1) * length = 4090 := 
by
  -- proof omitted, replace this line with actual proof
  sorry

end total_toothpicks_needed_l1008_100851


namespace remainder_12345678901_mod_101_l1008_100848

theorem remainder_12345678901_mod_101 : 12345678901 % 101 = 24 :=
by
  sorry

end remainder_12345678901_mod_101_l1008_100848


namespace prime_bounds_l1008_100843

noncomputable def is_prime (p : ℕ) : Prop := 2 ≤ p ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

theorem prime_bounds (n : ℕ) (h1 : 2 ≤ n) 
  (h2 : ∀ k, 0 ≤ k → k ≤ Nat.sqrt (n / 3) → is_prime (k^2 + k + n)) : 
  ∀ k, 0 ≤ k → k ≤ n - 2 → is_prime (k^2 + k + n) :=
by
  sorry

end prime_bounds_l1008_100843


namespace average_investment_per_km_in_scientific_notation_l1008_100864

-- Definitions based on the conditions of the problem
def total_investment : ℝ := 29.6 * 10^9
def upgraded_distance : ℝ := 6000

-- A theorem to be proven
theorem average_investment_per_km_in_scientific_notation :
  (total_investment / upgraded_distance) = 4.9 * 10^6 :=
by
  sorry

end average_investment_per_km_in_scientific_notation_l1008_100864


namespace inversely_proportional_l1008_100869

theorem inversely_proportional (x y : ℕ) (c : ℕ) 
  (h1 : x * y = c)
  (hx1 : x = 40) 
  (hy1 : y = 5) 
  (hy2 : y = 10) : x = 20 :=
by
  sorry

end inversely_proportional_l1008_100869


namespace find_x_l1008_100846

-- Define vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (2, 5)
def c (x : ℝ) : ℝ × ℝ := (3, x)

-- Dot product function
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Compute 8a - b
def sum_vec : ℝ × ℝ :=
  (8 * a.1 - b.1, 8 * a.2 - b.2)

-- Prove that x = 4 given condition
theorem find_x (x : ℝ) (h : dot_product sum_vec (c x) = 30) : x = 4 :=
by
  sorry

end find_x_l1008_100846


namespace Carol_optimal_choice_l1008_100805

noncomputable def Alice_choices := Set.Icc 0 (1 : ℝ)
noncomputable def Bob_choices := Set.Icc (1 / 3) (3 / 4 : ℝ)

theorem Carol_optimal_choice : 
  ∀ (c : ℝ), c ∈ Set.Icc 0 1 → 
  (∃! c, c = 7 / 12) := 
sorry

end Carol_optimal_choice_l1008_100805


namespace fixed_point_through_ellipse_l1008_100887

-- Define the ellipse and the points
def C (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def P2 : ℝ × ℝ := (0, 1)

-- Define the condition for a line not passing through P2 and intersecting the ellipse
def line_l_intersects_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  ∃ (x1 x2 b k : ℝ), l (x1, k * x1 + b) ∧ l (x2, k * x2 + b) ∧
  (C x1 (k * x1 + b)) ∧ (C x2 (k * x2 + b)) ∧
  ((x1, k * x1 + b) ≠ P2 ∧ (x2, k * x2 + b) ≠ P2) ∧
  ((k * x1 + b ≠ 1) ∧ (k * x2 + b ≠ 1)) ∧ 
  (∃ (kA kB : ℝ), kA = (k * x1 + b - 1) / x1 ∧ kB = (k * x2 + b - 1) / x2 ∧ kA + kB = -1)

-- Prove there exists a fixed point (2, -1) through which all such lines must pass
theorem fixed_point_through_ellipse (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) :
  line_l_intersects_ellipse A B l → l (2, -1) :=
sorry

end fixed_point_through_ellipse_l1008_100887


namespace common_chord_equation_l1008_100885

-- Definition of the first circle
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

-- Definition of the second circle
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4 * y = 0

-- Proposition stating we need to prove the line equation
theorem common_chord_equation (x y : ℝ) : circle1 x y → circle2 x y → x - y = 0 :=
by
  intros h1 h2
  sorry

end common_chord_equation_l1008_100885


namespace ne_of_P_l1008_100857

-- Define the initial proposition P
def P : Prop := ∀ m : ℝ, (0 ≤ m → 4^m ≥ 4 * m)

-- Define the negation of P
def not_P : Prop := ∃ m : ℝ, (0 ≤ m ∧ 4^m < 4 * m)

-- The theorem we need to prove
theorem ne_of_P : ¬P ↔ not_P :=
by
  sorry

end ne_of_P_l1008_100857


namespace solve_inequality_l1008_100897

theorem solve_inequality :
  {x : ℝ | (x^2 - 9) / (x - 3) > 0} = { x : ℝ | (-3 < x ∧ x < 3) ∨ (x > 3)} :=
by {
  sorry
}

end solve_inequality_l1008_100897


namespace solved_work_problem_l1008_100881

noncomputable def work_problem : Prop :=
  ∃ (m w x : ℝ), 
  (3 * m + 8 * w = 6 * m + x * w) ∧ 
  (4 * m + 5 * w = 0.9285714285714286 * (3 * m + 8 * w)) ∧
  (x = 14)

theorem solved_work_problem : work_problem := sorry

end solved_work_problem_l1008_100881


namespace min_value_problem_l1008_100852

theorem min_value_problem (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + 3 * b = 1) :
    (1 / a) + (3 / b) ≥ 16 :=
sorry

end min_value_problem_l1008_100852


namespace stream_speed_is_2_l1008_100830

variable (v : ℝ) -- Let v be the speed of the stream in km/h

-- Condition 1: Man's swimming speed in still water
def swimming_speed_still : ℝ := 6

-- Condition 2: It takes him twice as long to swim upstream as downstream
def condition : Prop := (swimming_speed_still + v) / (swimming_speed_still - v) = 2

theorem stream_speed_is_2 : condition v → v = 2 := by
  intro h
  -- Proof goes here
  sorry

end stream_speed_is_2_l1008_100830


namespace simplify_expression_l1008_100873

open Real

theorem simplify_expression (α : ℝ) : 
  (cos (4 * α - π / 2) * sin (5 * π / 2 + 2 * α)) / ((1 + cos (2 * α)) * (1 + cos (4 * α))) = tan α :=
by
  sorry

end simplify_expression_l1008_100873


namespace find_smallest_angle_b1_l1008_100826

-- Definitions and conditions
def smallest_angle_in_sector (b1 e : ℕ) (k : ℕ := 5) : Prop :=
  2 * b1 + (k - 1) * k * e = 360 ∧ b1 + 2 * e = 36

theorem find_smallest_angle_b1 (b1 e : ℕ) : smallest_angle_in_sector b1 e → b1 = 30 :=
  sorry

end find_smallest_angle_b1_l1008_100826


namespace true_discount_l1008_100886

theorem true_discount (BD PV TD : ℝ) (h1 : BD = 36) (h2 : PV = 180) :
  TD = 30 :=
by
  sorry

end true_discount_l1008_100886


namespace remainder_of_1999_pow_81_mod_7_eq_1_l1008_100849

/-- 
  Prove the remainder R when 1999^81 is divided by 7 is equal to 1.
  Conditions:
  - number: 1999
  - divisor: 7
-/
theorem remainder_of_1999_pow_81_mod_7_eq_1 : (1999 ^ 81) % 7 = 1 := 
by 
  sorry

end remainder_of_1999_pow_81_mod_7_eq_1_l1008_100849


namespace minimum_seedlings_needed_l1008_100824

theorem minimum_seedlings_needed (n : ℕ) (h1 : 75 ≤ n) (h2 : n ≤ 80) (H : 1200 * 100 / n = 1500) : n = 80 :=
sorry

end minimum_seedlings_needed_l1008_100824


namespace distribution_scheme_count_l1008_100883

-- Define the people and communities
inductive Person
| A | B | C
deriving DecidableEq, Repr

inductive Community
| C1 | C2 | C3 | C4 | C5 | C6 | C7
deriving DecidableEq, Repr

-- Define a function to count the number of valid distribution schemes
def countDistributionSchemes : Nat :=
  -- This counting is based on recognizing the problem involves permutations and combinations,
  -- the specific detail logic is omitted since we are only writing the statement, no proof.
  336

-- The main theorem statement
theorem distribution_scheme_count :
  countDistributionSchemes = 336 :=
sorry

end distribution_scheme_count_l1008_100883


namespace lightsaber_ratio_l1008_100802

theorem lightsaber_ratio (T L : ℕ) (hT : T = 1000) (hTotal : L + T = 3000) : L / T = 2 :=
by
  sorry

end lightsaber_ratio_l1008_100802


namespace balloon_difference_l1008_100895

theorem balloon_difference (your_balloons : ℕ) (friend_balloons : ℕ) (h1 : your_balloons = 7) (h2 : friend_balloons = 5) : your_balloons - friend_balloons = 2 :=
by
  sorry

end balloon_difference_l1008_100895


namespace cistern_capacity_l1008_100876

theorem cistern_capacity (C : ℝ) (h1 : C / 20 > 0) (h2 : C / 24 > 0) (h3 : 4 - C / 20 = C / 24) : C = 480 / 11 :=
by sorry

end cistern_capacity_l1008_100876


namespace sequence_general_term_l1008_100831

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : ∀ n, S n = n^2) 
    (h_a₁ : S 1 = 1) (h_an : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) : 
  ∀ n, a n = 2 * n - 1 := 
by
  sorry

end sequence_general_term_l1008_100831


namespace solve_inequality_system_l1008_100855

theorem solve_inequality_system (x : ℝ) (h1 : x - 2 ≤ 0) (h2 : (x - 1) / 2 < x) : -1 < x ∧ x ≤ 2 := 
sorry

end solve_inequality_system_l1008_100855


namespace jamie_minimum_4th_quarter_score_l1008_100811

-- Define the conditions for Jamie's scores and the average requirement
def qualifying_score := 85
def first_quarter_score := 80
def second_quarter_score := 85
def third_quarter_score := 78

-- The function to determine the required score in the 4th quarter
def minimum_score_for_quarter (N : ℕ) := first_quarter_score + second_quarter_score + third_quarter_score + N ≥ 4 * qualifying_score

-- The main statement to be proved
theorem jamie_minimum_4th_quarter_score (N : ℕ) : minimum_score_for_quarter N ↔ N ≥ 97 :=
by
  sorry

end jamie_minimum_4th_quarter_score_l1008_100811


namespace original_number_l1008_100860

theorem original_number (x y : ℕ) (h1 : x + y = 859560) (h2 : y = 859560 % 456) : x = 859376 ∧ 456 ∣ x :=
by
  sorry

end original_number_l1008_100860
