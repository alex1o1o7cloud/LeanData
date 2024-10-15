import Mathlib

namespace NUMINAMATH_GPT_production_average_l1517_151732

theorem production_average (n : ℕ) (P : ℕ) (hP : P = n * 50)
  (h1 : (P + 95) / (n + 1) = 55) : n = 8 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_production_average_l1517_151732


namespace NUMINAMATH_GPT_minimum_shirts_to_save_money_l1517_151706

-- Definitions for the costs
def EliteCost (n : ℕ) : ℕ := 30 + 8 * n
def OmegaCost (n : ℕ) : ℕ := 10 + 12 * n

-- Theorem to prove the given solution
theorem minimum_shirts_to_save_money : ∃ n : ℕ, 30 + 8 * n < 10 + 12 * n ∧ n = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimum_shirts_to_save_money_l1517_151706


namespace NUMINAMATH_GPT_min_value_fraction_ineq_l1517_151791

-- Define the conditions and statement to be proved
theorem min_value_fraction_ineq (x : ℝ) (hx : x > 4) : 
  ∃ M, M = 4 * Real.sqrt 5 ∧ ∀ y : ℝ, y > 4 → (y + 16) / Real.sqrt (y - 4) ≥ M := 
sorry

end NUMINAMATH_GPT_min_value_fraction_ineq_l1517_151791


namespace NUMINAMATH_GPT_coin_flips_heads_l1517_151712

theorem coin_flips_heads (H T : ℕ) (flip_condition : H + T = 211) (tail_condition : T = H + 81) :
    H = 65 :=
by
  sorry

end NUMINAMATH_GPT_coin_flips_heads_l1517_151712


namespace NUMINAMATH_GPT_inequality_xy_yz_zx_l1517_151762

theorem inequality_xy_yz_zx {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) <= 1 / 4 * (Real.sqrt 33 + 1) :=
sorry

end NUMINAMATH_GPT_inequality_xy_yz_zx_l1517_151762


namespace NUMINAMATH_GPT_combined_rate_mpg_900_over_41_l1517_151754

-- Declare the variables and conditions
variables {d : ℕ} (h_d_pos : d > 0)

def combined_mpg (d : ℕ) : ℚ :=
  let anna_car_gasoline := (d : ℚ) / 50
  let ben_car_gasoline  := (d : ℚ) / 20
  let carl_car_gasoline := (d : ℚ) / 15
  let total_gasoline    := anna_car_gasoline + ben_car_gasoline + carl_car_gasoline
  ((3 : ℚ) * d) / total_gasoline

-- Define the theorem statement
theorem combined_rate_mpg_900_over_41 :
  ∀ d : ℕ, d > 0 → combined_mpg d = 900 / 41 :=
by
  intros d h_d_pos
  rw [combined_mpg]
  -- Steps following the solution
  sorry -- proof omitted

end NUMINAMATH_GPT_combined_rate_mpg_900_over_41_l1517_151754


namespace NUMINAMATH_GPT_find_n_l1517_151788

theorem find_n (n : ℝ) (h1 : ∀ x y : ℝ, (n + 1) * x^(n^2 - 5) = y) 
               (h2 : ∀ x > 0, (n + 1) * x^(n^2 - 5) > 0) :
               n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1517_151788


namespace NUMINAMATH_GPT_probability_of_sphere_in_cube_l1517_151776

noncomputable def cube_volume : Real :=
  (4 : Real)^3

noncomputable def sphere_volume : Real :=
  (4 / 3) * Real.pi * (2 : Real)^3

noncomputable def probability : Real :=
  sphere_volume / cube_volume

theorem probability_of_sphere_in_cube : probability = Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_probability_of_sphere_in_cube_l1517_151776


namespace NUMINAMATH_GPT_total_price_of_order_l1517_151769

-- Define the price of each item
def price_ice_cream_bar : ℝ := 0.60
def price_sundae : ℝ := 1.40

-- Define the quantity of each item
def quantity_ice_cream_bar : ℕ := 125
def quantity_sundae : ℕ := 125

-- Calculate the costs
def cost_ice_cream_bar := quantity_ice_cream_bar * price_ice_cream_bar
def cost_sundae := quantity_sundae * price_sundae

-- Calculate the total cost
def total_cost := cost_ice_cream_bar + cost_sundae

-- Statement of the theorem
theorem total_price_of_order : total_cost = 250 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_price_of_order_l1517_151769


namespace NUMINAMATH_GPT_compare_sums_of_square_roots_l1517_151741

theorem compare_sums_of_square_roots
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (M : ℝ := Real.sqrt a + Real.sqrt b) 
  (N : ℝ := Real.sqrt (a + b)) :
  M > N :=
by
  sorry

end NUMINAMATH_GPT_compare_sums_of_square_roots_l1517_151741


namespace NUMINAMATH_GPT_no_egg_arrangements_possible_l1517_151781

noncomputable def num_egg_arrangements 
  (total_eggs : ℕ) 
  (type_A_eggs : ℕ) 
  (type_B_eggs : ℕ)
  (type_C_eggs : ℕ)
  (groups : ℕ)
  (ratio_A : ℕ) 
  (ratio_B : ℕ) 
  (ratio_C : ℕ) : ℕ :=
if (total_eggs = type_A_eggs + type_B_eggs + type_C_eggs) ∧ 
   (type_A_eggs / groups = ratio_A) ∧ 
   (type_B_eggs / groups = ratio_B) ∧ 
   (type_C_eggs / groups = ratio_C) then 0 else 0

theorem no_egg_arrangements_possible :
  num_egg_arrangements 35 15 12 8 5 2 3 1 = 0 := 
by sorry

end NUMINAMATH_GPT_no_egg_arrangements_possible_l1517_151781


namespace NUMINAMATH_GPT_problem_statement_l1517_151789

-- Define the power function f and the property that it is odd
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_cond : f 3 < f 2)

-- The statement we need to prove
theorem problem_statement : f (-3) > f (-2) := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1517_151789


namespace NUMINAMATH_GPT_min_value_correct_l1517_151721

noncomputable def min_value (x y : ℝ) : ℝ :=
x * y / (x^2 + y^2)

theorem min_value_correct :
  ∃ x y : ℝ,
    (2 / 5 : ℝ) ≤ x ∧ x ≤ (1 / 2 : ℝ) ∧
    (1 / 3 : ℝ) ≤ y ∧ y ≤ (3 / 8 : ℝ) ∧
    min_value x y = (6 / 13 : ℝ) :=
by sorry

end NUMINAMATH_GPT_min_value_correct_l1517_151721


namespace NUMINAMATH_GPT_ratio_of_shares_l1517_151733

-- Definitions for the given conditions
def capital_A : ℕ := 4500
def capital_B : ℕ := 16200
def months_A : ℕ := 12
def months_B : ℕ := 5 -- B joined after 7 months

-- Effective capital contributions
def effective_capital_A : ℕ := capital_A * months_A
def effective_capital_B : ℕ := capital_B * months_B

-- Defining the statement to prove
theorem ratio_of_shares : effective_capital_A / Nat.gcd effective_capital_A effective_capital_B = 2 ∧ effective_capital_B / Nat.gcd effective_capital_A effective_capital_B = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_shares_l1517_151733


namespace NUMINAMATH_GPT_quadrant_of_angle_l1517_151703

theorem quadrant_of_angle (θ : ℝ) (h1 : Real.cos θ = -3 / 5) (h2 : Real.tan θ = 4 / 3) :
    θ ∈ Set.Icc (π : ℝ) (3 * π / 2) := sorry

end NUMINAMATH_GPT_quadrant_of_angle_l1517_151703


namespace NUMINAMATH_GPT_trader_profit_percent_equal_eight_l1517_151770

-- Defining the initial conditions
def original_price (P : ℝ) := P
def purchased_price (P : ℝ) := 0.60 * original_price P
def selling_price (P : ℝ) := 1.80 * purchased_price P

-- Statement to be proved
theorem trader_profit_percent_equal_eight (P : ℝ) (h : P > 0) :
  ((selling_price P - original_price P) / original_price P) * 100 = 8 :=
by
  sorry

end NUMINAMATH_GPT_trader_profit_percent_equal_eight_l1517_151770


namespace NUMINAMATH_GPT_total_distance_is_3_miles_l1517_151764

-- Define conditions
def running_speed := 6   -- mph
def walking_speed := 2   -- mph
def running_time := 20 / 60   -- hours
def walking_time := 30 / 60   -- hours

-- Define total distance
def total_distance := (running_speed * running_time) + (walking_speed * walking_time)

theorem total_distance_is_3_miles : total_distance = 3 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_is_3_miles_l1517_151764


namespace NUMINAMATH_GPT_polynomial_q_correct_l1517_151710

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end NUMINAMATH_GPT_polynomial_q_correct_l1517_151710


namespace NUMINAMATH_GPT_sale_price_same_as_original_l1517_151787

theorem sale_price_same_as_original (x : ℝ) :
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sale_price = x := 
by
  let increased_price := 1.25 * x
  let sale_price := 0.8 * increased_price
  sorry

end NUMINAMATH_GPT_sale_price_same_as_original_l1517_151787


namespace NUMINAMATH_GPT_triangle_inequality_l1517_151756

variable (a b c : ℝ)
variable (h1 : a * b + b * c + c * a = 18)
variable (h2 : 1 < a)
variable (h3 : 1 < b)
variable (h4 : 1 < c)

theorem triangle_inequality :
  (1 / (a - 1)^3 + 1 / (b - 1)^3 + 1 / (c - 1)^3) > (1 / (a + b + c - 3)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1517_151756


namespace NUMINAMATH_GPT_determine_m_l1517_151745

variable {x y z : ℝ}

theorem determine_m (h : (5 / (x + y)) = (m / (x + z)) ∧ (m / (x + z)) = (13 / (z - y))) : m = 18 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l1517_151745


namespace NUMINAMATH_GPT_negation_of_proposition_l1517_151739

theorem negation_of_proposition (x : ℝ) : 
  ¬ (∀ x : ℝ, x^2 - x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 1 < 0) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1517_151739


namespace NUMINAMATH_GPT_denomination_of_remaining_coins_l1517_151708

/-
There are 324 coins total.
The total value of the coins is Rs. 70.
There are 220 coins of 20 paise each.
Find the denomination of the remaining coins.
-/

def total_coins := 324
def total_value := 7000 -- Rs. 70 converted into paise
def num_20_paise_coins := 220
def value_20_paise_coin := 20
  
theorem denomination_of_remaining_coins :
  let total_remaining_value := total_value - (num_20_paise_coins * value_20_paise_coin)
  let num_remaining_coins := total_coins - num_20_paise_coins
  num_remaining_coins > 0 →
  total_remaining_value / num_remaining_coins = 25 :=
by
  sorry

end NUMINAMATH_GPT_denomination_of_remaining_coins_l1517_151708


namespace NUMINAMATH_GPT_monthly_salary_equals_l1517_151713

-- Define the base salary
def base_salary : ℝ := 1600

-- Define the commission rate
def commission_rate : ℝ := 0.04

-- Define the sales amount for which the salaries are equal
def sales_amount : ℝ := 5000

-- Define the total earnings with a base salary and commission for 5000 worth of sales
def total_earnings : ℝ := base_salary + (commission_rate * sales_amount)

-- Define the monthly salary from Furniture by Design
def monthly_salary : ℝ := 1800

-- Prove that the monthly salary S is equal to 1800
theorem monthly_salary_equals :
  total_earnings = monthly_salary :=
by
  -- The proof is skipped with sorry.
  sorry

end NUMINAMATH_GPT_monthly_salary_equals_l1517_151713


namespace NUMINAMATH_GPT_determine_y_l1517_151726

def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

theorem determine_y (y : ℝ) (h : diamond 4 y = 30) : y = 5 / 3 :=
by sorry

end NUMINAMATH_GPT_determine_y_l1517_151726


namespace NUMINAMATH_GPT_tenth_term_arith_seq_l1517_151758

variable (a1 d : Int) -- Initial term and common difference
variable (n : Nat) -- nth term

-- Definition of the nth term in an arithmetic sequence
def arithmeticSeq (a1 d : Int) (n : Nat) : Int :=
  a1 + (n - 1) * d

-- Specific values for the problem
def a_10 : Int :=
  arithmeticSeq 10 (-3) 10

-- The theorem we want to prove
theorem tenth_term_arith_seq : a_10 = -17 := by
  sorry

end NUMINAMATH_GPT_tenth_term_arith_seq_l1517_151758


namespace NUMINAMATH_GPT_park_area_correct_l1517_151740

noncomputable def rect_park_area (speed_km_hr : ℕ) (time_min : ℕ) (ratio_l_b : ℕ) : ℕ := by
  let speed_m_min := speed_km_hr * 1000 / 60
  let perimeter := speed_m_min * time_min
  let B := perimeter * 3 / 8
  let L := B / 3
  let area := L * B
  exact area

theorem park_area_correct : rect_park_area 12 8 3 = 120000 := by
  sorry

end NUMINAMATH_GPT_park_area_correct_l1517_151740


namespace NUMINAMATH_GPT_cos_diff_identity_l1517_151700

variable {α : ℝ}

def sin_alpha := -3 / 5

def alpha_interval (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi)

theorem cos_diff_identity (h1 : Real.sin α = sin_alpha) (h2 : alpha_interval α) :
  Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 10 :=
  sorry

end NUMINAMATH_GPT_cos_diff_identity_l1517_151700


namespace NUMINAMATH_GPT_graph_passes_through_point_l1517_151792

theorem graph_passes_through_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ p : ℝ × ℝ, p = (2, 0) ∧ ∀ x, (x = 2 → a ^ (x - 2) - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_point_l1517_151792


namespace NUMINAMATH_GPT_calc_expr_l1517_151744

noncomputable def expr_val : ℝ :=
  Real.sqrt 4 - |(-(1 / 4 : ℝ))| + (Real.pi - 2)^0 + 2^(-2 : ℝ)

theorem calc_expr : expr_val = 3 := by
  sorry

end NUMINAMATH_GPT_calc_expr_l1517_151744


namespace NUMINAMATH_GPT_min_sum_of_squares_l1517_151796

theorem min_sum_of_squares 
  (x_1 x_2 x_3 : ℝ)
  (h1: x_1 + 3 * x_2 + 4 * x_3 = 72)
  (h2: x_1 = 3 * x_2)
  (h3: 0 < x_1)
  (h4: 0 < x_2)
  (h5: 0 < x_3) : 
  x_1^2 + x_2^2 + x_3^2 = 347.04 := 
sorry

end NUMINAMATH_GPT_min_sum_of_squares_l1517_151796


namespace NUMINAMATH_GPT_systematic_sample_contains_18_l1517_151722

theorem systematic_sample_contains_18 (employees : Finset ℕ) (sample : Finset ℕ)
    (h1 : employees = Finset.range 52)
    (h2 : sample.card = 4)
    (h3 : ∀ n ∈ sample, n ∈ employees)
    (h4 : 5 ∈ sample)
    (h5 : 31 ∈ sample)
    (h6 : 44 ∈ sample) :
  18 ∈ sample :=
sorry

end NUMINAMATH_GPT_systematic_sample_contains_18_l1517_151722


namespace NUMINAMATH_GPT_sum_of_legs_l1517_151747

theorem sum_of_legs (x : ℕ) (h : x^2 + (x + 1)^2 = 41^2) : x + (x + 1) = 57 :=
sorry

end NUMINAMATH_GPT_sum_of_legs_l1517_151747


namespace NUMINAMATH_GPT_symmetrical_circle_l1517_151768

-- Defining the given circle's equation
def given_circle_eq (x y: ℝ) : Prop := (x + 2)^2 + y^2 = 5

-- Defining the equation of the symmetrical circle
def symmetrical_circle_eq (x y: ℝ) : Prop := (x - 2)^2 + y^2 = 5

-- Proving the symmetry property
theorem symmetrical_circle (x y : ℝ) : 
  (given_circle_eq x y) → (symmetrical_circle_eq (-x) (-y)) :=
by
  sorry

end NUMINAMATH_GPT_symmetrical_circle_l1517_151768


namespace NUMINAMATH_GPT_average_age_new_students_l1517_151798

theorem average_age_new_students (O A_old A_new_avg A_new : ℕ) 
  (hO : O = 8) 
  (hA_old : A_old = 40) 
  (hA_new_avg : A_new_avg = 36)
  (h_total_age_before : O * A_old = 8 * 40)
  (h_total_age_after : (O + 8) * A_new_avg = 16 * 36)
  (h_age_new_students : (16 * 36) - (8 * 40) = A_new * 8) :
  A_new = 32 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_new_students_l1517_151798


namespace NUMINAMATH_GPT_inequality_solution_l1517_151711

theorem inequality_solution (a : ℝ) (h : ∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) : a < -1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1517_151711


namespace NUMINAMATH_GPT_caller_wins_both_at_35_l1517_151705

theorem caller_wins_both_at_35 (n : ℕ) :
  ∀ n, (n % 5 = 0 ∧ n % 7 = 0) ↔ n = 35 :=
by
  sorry

end NUMINAMATH_GPT_caller_wins_both_at_35_l1517_151705


namespace NUMINAMATH_GPT_common_difference_arithmetic_seq_l1517_151765

theorem common_difference_arithmetic_seq (S n a1 d : ℕ) (h_sum : S = 650) (h_n : n = 20) (h_a1 : a1 = 4) :
  S = (n / 2) * (2 * a1 + (n - 1) * d) → d = 3 := by
  intros h_formula
  sorry

end NUMINAMATH_GPT_common_difference_arithmetic_seq_l1517_151765


namespace NUMINAMATH_GPT_exists_four_numbers_with_equal_sum_l1517_151775

theorem exists_four_numbers_with_equal_sum (S : Finset ℕ) (hS : S.card = 16) (h_range : ∀ n ∈ S, n ≤ 100) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ c ≠ d ∧ a ≠ c ∧ b ≠ d ∧ a + b = c + d :=
by
  sorry

end NUMINAMATH_GPT_exists_four_numbers_with_equal_sum_l1517_151775


namespace NUMINAMATH_GPT_min_ab_diff_value_l1517_151727

noncomputable def min_ab_diff (x y z : ℝ) : ℝ :=
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2

theorem min_ab_diff_value : ∀ (x y z : ℝ),
  0 ≤ x → 0 ≤ y → 0 ≤ z → min_ab_diff x y z = 36 :=
by
  intros x y z hx hy hz
  sorry

end NUMINAMATH_GPT_min_ab_diff_value_l1517_151727


namespace NUMINAMATH_GPT_relationship_between_A_and_B_l1517_151734

theorem relationship_between_A_and_B (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  let A := a^2
  let B := 2 * a - 1
  A > B :=
by
  let A := a^2
  let B := 2 * a - 1
  sorry

end NUMINAMATH_GPT_relationship_between_A_and_B_l1517_151734


namespace NUMINAMATH_GPT_math_problem_l1517_151746

def otimes (a b : ℚ) : ℚ := (a^3) / (b^2)

theorem math_problem : ((otimes (otimes 2 4) 6) - (otimes 2 (otimes 4 6))) = -23327 / 288 := by sorry

end NUMINAMATH_GPT_math_problem_l1517_151746


namespace NUMINAMATH_GPT_tile_area_l1517_151774

-- Define the properties and conditions of the tile

structure Tile where
  sides : Fin 9 → ℝ 
  six_of_length_1 : ∀ i : Fin 6, sides i = 1 
  congruent_quadrilaterals : Fin 3 → Quadrilateral

structure Quadrilateral where
  length : ℝ
  width : ℝ

-- Given the tile structure, calculate the area
noncomputable def area_of_tile (t: Tile) : ℝ := sorry

-- Statement: Prove the area of the tile given the conditions
theorem tile_area (t : Tile) : area_of_tile t = (4 * Real.sqrt 3 / 3) :=
  sorry

end NUMINAMATH_GPT_tile_area_l1517_151774


namespace NUMINAMATH_GPT_divides_floor_factorial_div_l1517_151718

theorem divides_floor_factorial_div {m n : ℕ} (h1 : 1 < m) (h2 : m < n + 2) (h3 : 3 < n) :
  (m - 1) ∣ (n! / m) :=
sorry

end NUMINAMATH_GPT_divides_floor_factorial_div_l1517_151718


namespace NUMINAMATH_GPT_As_annual_income_l1517_151714

theorem As_annual_income :
  let Cm := 14000
  let Bm := Cm + 0.12 * Cm
  let Am := (5 / 2) * Bm
  Am * 12 = 470400 := by
  sorry

end NUMINAMATH_GPT_As_annual_income_l1517_151714


namespace NUMINAMATH_GPT_sum_of_subsets_l1517_151729

theorem sum_of_subsets (a1 a2 a3 : ℝ) (h : (a1 + a2 + a3) + (a1 + a2 + a1 + a3 + a2 + a3) = 12) : 
  a1 + a2 + a3 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_subsets_l1517_151729


namespace NUMINAMATH_GPT_simplify_sum_l1517_151748

theorem simplify_sum : 
  (-1: ℤ)^(2010) + (-1: ℤ)^(2011) + (1: ℤ)^(2012) + (-1: ℤ)^(2013) = -2 := by
  sorry

end NUMINAMATH_GPT_simplify_sum_l1517_151748


namespace NUMINAMATH_GPT_expression_simplifies_to_neg_seven_l1517_151717

theorem expression_simplifies_to_neg_seven (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
(h₃ : a + b + c = 0) (h₄ : ab + ac + bc ≠ 0) : 
    (a^7 + b^7 + c^7) / (abc * (ab + ac + bc)) = -7 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplifies_to_neg_seven_l1517_151717


namespace NUMINAMATH_GPT_find_a17_a18_a19_a20_l1517_151736

variable {α : Type*} [Field α]

-- Definitions based on the given conditions:
def geometric_sequence (a : ℕ → α) : Prop :=
  ∃ r : α, ∀ n : ℕ, a n = a 0 * r ^ n

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (Finset.range n).sum a

-- Problem statement based on the question and conditions:
theorem find_a17_a18_a19_a20 (a S : ℕ → α) (h_geom : geometric_sequence a)
  (h_sum : sum_of_first_n_terms a S) (hS4 : S 4 = 1) (hS8 : S 8 = 3) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
sorry

end NUMINAMATH_GPT_find_a17_a18_a19_a20_l1517_151736


namespace NUMINAMATH_GPT_find_discriminant_l1517_151783

variables {a b c : ℝ}
variables (P : ℝ → ℝ)
def is_quadratic_polynomial (P : ℝ → ℝ) : Prop := ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, P x = a * x^2 + b * x + c)

theorem find_discriminant (h1 : is_quadratic_polynomial P)
  (h2 : ∃ x, P x = x - 2)
  (h3 : ∃ y, P y = 1 - y / 2)
  : ∃ D, D = -1/2 := 
sorry

end NUMINAMATH_GPT_find_discriminant_l1517_151783


namespace NUMINAMATH_GPT_draw_from_unit_D_l1517_151761

variable (d : ℕ)

-- Variables representing the number of questionnaires drawn from A, B, C, and D
def QA : ℕ := 30 - d
def QB : ℕ := 30
def QC : ℕ := 30 + d
def QD : ℕ := 30 + 2 * d

-- Total number of questionnaires drawn
def TotalDrawn : ℕ := QA d + QB + QC d + QD d

theorem draw_from_unit_D :
  (TotalDrawn d = 150) →
  QD d = 60 := sorry

end NUMINAMATH_GPT_draw_from_unit_D_l1517_151761


namespace NUMINAMATH_GPT_transform_fraction_l1517_151786

theorem transform_fraction (x : ℝ) (h₁ : x ≠ 3) : - (1 / (3 - x)) = (1 / (x - 3)) := 
    sorry

end NUMINAMATH_GPT_transform_fraction_l1517_151786


namespace NUMINAMATH_GPT_Caitlin_age_l1517_151782

theorem Caitlin_age (Aunt_Anna_age : ℕ) (h1 : Aunt_Anna_age = 54) (Brianna_age : ℕ) (h2 : Brianna_age = (2 * Aunt_Anna_age) / 3) (Caitlin_age : ℕ) (h3 : Caitlin_age = Brianna_age - 7) : 
  Caitlin_age = 29 := 
  sorry

end NUMINAMATH_GPT_Caitlin_age_l1517_151782


namespace NUMINAMATH_GPT_gcd_seq_coprime_l1517_151784

def seq (n : ℕ) : ℕ := 2^(2^n) + 1

theorem gcd_seq_coprime (n k : ℕ) (hnk : n ≠ k) : Nat.gcd (seq n) (seq k) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_seq_coprime_l1517_151784


namespace NUMINAMATH_GPT_slope_of_line_passing_through_MN_l1517_151795

theorem slope_of_line_passing_through_MN :
  let M := (-2, 1)
  let N := (1, 4)
  ∃ m : ℝ, m = (N.2 - M.2) / (N.1 - M.1) ∧ m = 1 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_passing_through_MN_l1517_151795


namespace NUMINAMATH_GPT_radius_calculation_l1517_151799

noncomputable def radius_of_circle (n : ℕ) : ℝ :=
if 2 ≤ n ∧ n ≤ 11 then
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61
else
  0  -- Outside the specified range

theorem radius_calculation (n : ℕ) (hn : 2 ≤ n ∧ n ≤ 11) :
  radius_of_circle n =
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61 :=
sorry

end NUMINAMATH_GPT_radius_calculation_l1517_151799


namespace NUMINAMATH_GPT_team_order_l1517_151778

-- Define the points of teams
variables (A B C D : ℕ)

-- State the conditions
def condition1 := A + C = B + D
def condition2 := B + A + 5 ≤ D + C
def condition3 := B + C ≥ A + D + 3

-- Statement of the theorem
theorem team_order (h1 : condition1 A B C D) (h2 : condition2 A B C D) (h3 : condition3 A B C D) :
  C > D ∧ D > B ∧ B > A :=
sorry

end NUMINAMATH_GPT_team_order_l1517_151778


namespace NUMINAMATH_GPT_difference_is_three_l1517_151755

-- Define the range for two-digit numbers
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- Define whether a number is a multiple of three
def is_multiple_of_three (n : ℕ) : Prop := n % 3 = 0

-- Identify the smallest and largest two-digit multiples of three
def smallest_two_digit_multiple_of_three : ℕ := 12
def largest_two_digit_multiple_of_three : ℕ := 99

-- Identify the smallest and largest two-digit non-multiples of three
def smallest_two_digit_non_multiple_of_three : ℕ := 10
def largest_two_digit_non_multiple_of_three : ℕ := 98

-- Calculate Joey's sum
def joeys_sum : ℕ := smallest_two_digit_multiple_of_three + largest_two_digit_multiple_of_three

-- Calculate Zoë's sum
def zoes_sum : ℕ := smallest_two_digit_non_multiple_of_three + largest_two_digit_non_multiple_of_three

-- Prove the difference between Joey's and Zoë's sums is 3
theorem difference_is_three : joeys_sum - zoes_sum = 3 :=
by
  -- The proof is not given, so we use sorry here
  sorry

end NUMINAMATH_GPT_difference_is_three_l1517_151755


namespace NUMINAMATH_GPT_probability_of_a_b_c_l1517_151777

noncomputable def probability_condition : ℚ :=
  5 / 6 * 5 / 6 * 7 / 8

theorem probability_of_a_b_c : 
  let a_outcome := 6
  let b_outcome := 6
  let c_outcome := 8
  (1 / a_outcome) * (1 / b_outcome) * (1 / c_outcome) = probability_condition :=
sorry

end NUMINAMATH_GPT_probability_of_a_b_c_l1517_151777


namespace NUMINAMATH_GPT_find_cost_price_l1517_151759

variable (CP : ℝ)

def selling_price (CP : ℝ) := CP * 1.40

theorem find_cost_price (h : selling_price CP = 1680) : CP = 1200 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l1517_151759


namespace NUMINAMATH_GPT_sin_alpha_value_l1517_151767

open Real

theorem sin_alpha_value (α β : ℝ) 
  (h1 : cos (α - β) = 3 / 5) 
  (h2 : sin β = -5 / 13) 
  (h3 : 0 < α ∧ α < π / 2) 
  (h4 : -π / 2 < β ∧ β < 0) 
  : sin α = 33 / 65 :=
sorry

end NUMINAMATH_GPT_sin_alpha_value_l1517_151767


namespace NUMINAMATH_GPT_three_zeros_of_f_l1517_151704

noncomputable def f (a x b : ℝ) : ℝ := (1/2) * a * x^2 - (a^2 + a + 2) * x + (2 * a + 2) * (Real.log x) + b

theorem three_zeros_of_f (a b : ℝ) (h1 : a > 3) (h2 : a^2 + a + 1 < b) (h3 : b < 2 * a^2 - 2 * a + 2) : 
  ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f a x1 b = 0 ∧ f a x2 b = 0 ∧ f a x3 b = 0 :=
by
  sorry

end NUMINAMATH_GPT_three_zeros_of_f_l1517_151704


namespace NUMINAMATH_GPT_replace_digits_and_check_divisibility_l1517_151716

theorem replace_digits_and_check_divisibility (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 ≠ 0 ∧ 
     (30 * 10^5 + a * 10^4 + b * 10^2 + 3) % 13 = 0) ↔ 
    (30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3000803 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3020303 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3030703 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3050203 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3060603 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3080103 ∨ 
     30 * 10^5 + a * 10^4 + b * 10^2 + 3 = 3090503) := sorry

end NUMINAMATH_GPT_replace_digits_and_check_divisibility_l1517_151716


namespace NUMINAMATH_GPT_percent_of_z_l1517_151719

variable {x y z : ℝ}

theorem percent_of_z (h₁ : x = 1.20 * y) (h₂ : y = 0.50 * z) : x = 0.60 * z :=
by
  sorry

end NUMINAMATH_GPT_percent_of_z_l1517_151719


namespace NUMINAMATH_GPT_bananas_to_pears_l1517_151780

theorem bananas_to_pears : ∀ (cost_banana cost_apple cost_pear : ℚ),
  (5 * cost_banana = 3 * cost_apple) →
  (9 * cost_apple = 6 * cost_pear) →
  (25 * cost_banana = 10 * cost_pear) :=
by
  intros cost_banana cost_apple cost_pear h1 h2
  sorry

end NUMINAMATH_GPT_bananas_to_pears_l1517_151780


namespace NUMINAMATH_GPT_tank_capacity_correctness_l1517_151731

noncomputable def tankCapacity : ℝ := 77.65

theorem tank_capacity_correctness (T : ℝ) 
  (h_initial: T * (5 / 8) + 11 = T * (23 / 30)) : 
  T = tankCapacity := 
by
  sorry

end NUMINAMATH_GPT_tank_capacity_correctness_l1517_151731


namespace NUMINAMATH_GPT_min_distance_PQ_l1517_151743

theorem min_distance_PQ :
  ∀ (P Q : ℝ × ℝ), (P.1 - P.2 - 4 = 0) → (Q.1^2 = 4 * Q.2) →
  ∃ (d : ℝ), d = dist P Q ∧ d = 3 * Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_min_distance_PQ_l1517_151743


namespace NUMINAMATH_GPT_find_f_of_2011_l1517_151794

-- Define the function f
def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 7

-- The main statement we need to prove
theorem find_f_of_2011 (a b c : ℝ) (h : f (-2011) a b c = -17) : f 2011 a b c = 31 :=
by
  sorry

end NUMINAMATH_GPT_find_f_of_2011_l1517_151794


namespace NUMINAMATH_GPT_sum_of_abc_is_12_l1517_151752

theorem sum_of_abc_is_12 (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abc_is_12_l1517_151752


namespace NUMINAMATH_GPT_sum_of_bases_l1517_151766

theorem sum_of_bases (R₁ R₂ : ℕ) 
    (h1 : (4 * R₁ + 5) / (R₁^2 - 1) = (3 * R₂ + 4) / (R₂^2 - 1))
    (h2 : (5 * R₁ + 4) / (R₁^2 - 1) = (4 * R₂ + 3) / (R₂^2 - 1)) : 
    R₁ + R₂ = 23 := 
sorry

end NUMINAMATH_GPT_sum_of_bases_l1517_151766


namespace NUMINAMATH_GPT_sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l1517_151707

theorem sqrt_49_mul_sqrt_25_eq_7_sqrt_5 : (Real.sqrt (49 * Real.sqrt 25)) = 7 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_49_mul_sqrt_25_eq_7_sqrt_5_l1517_151707


namespace NUMINAMATH_GPT_find_room_dimension_l1517_151751

noncomputable def unknown_dimension_of_room 
  (cost_per_sq_ft : ℕ)
  (total_cost : ℕ)
  (w : ℕ)
  (l : ℕ)
  (h : ℕ)
  (door_h : ℕ)
  (door_w : ℕ)
  (window_h : ℕ)
  (window_w : ℕ)
  (num_windows : ℕ) : ℕ := sorry

theorem find_room_dimension :
  unknown_dimension_of_room 10 9060 25 15 12 6 3 4 3 3 = 25 :=
sorry

end NUMINAMATH_GPT_find_room_dimension_l1517_151751


namespace NUMINAMATH_GPT_perpendicular_vectors_x_value_l1517_151702

-- Define the vectors a and b
def a : ℝ × ℝ := (3, -1)
def b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define the dot product function for vectors in ℝ^2
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- The mathematical statement to prove
theorem perpendicular_vectors_x_value (x : ℝ) (h : dot_product a (b x) = 0) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_value_l1517_151702


namespace NUMINAMATH_GPT_point_satisfies_equation_l1517_151772

theorem point_satisfies_equation (x y : ℝ) :
  (-1 ≤ x ∧ x ≤ 3) ∧ (-5 ≤ y ∧ y ≤ 1) ∧
  ((3 * x + 2 * y = 5) ∨ (-3 * x + 2 * y = -1) ∨ (3 * x - 2 * y = 13) ∨ (-3 * x - 2 * y = 7))
  → 3 * |x - 1| + 2 * |y + 2| = 6 := 
by 
  sorry

end NUMINAMATH_GPT_point_satisfies_equation_l1517_151772


namespace NUMINAMATH_GPT_cars_without_paying_l1517_151749

theorem cars_without_paying (total_cars : ℕ) (percent_with_tickets : ℚ) (fraction_with_passes : ℚ)
  (h1 : total_cars = 300)
  (h2 : percent_with_tickets = 0.75)
  (h3 : fraction_with_passes = 1/5) :
  let cars_with_tickets := percent_with_tickets * total_cars
  let cars_with_passes := fraction_with_passes * cars_with_tickets
  total_cars - (cars_with_tickets + cars_with_passes) = 30 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_cars_without_paying_l1517_151749


namespace NUMINAMATH_GPT_notebook_cost_l1517_151773

theorem notebook_cost (s n c : ℕ) (h1 : s > 20) (h2 : n > 2) (h3 : c > 2 * n) (h4 : s * c * n = 4515) : c = 35 :=
sorry

end NUMINAMATH_GPT_notebook_cost_l1517_151773


namespace NUMINAMATH_GPT_min_value_proof_l1517_151750

noncomputable def min_value_expression (a b : ℝ) : ℝ :=
  (1 / (12 * a + 1)) + (1 / (8 * b + 1))

theorem min_value_proof (a b : ℝ) (h1 : 3 * a + 2 * b = 1) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  min_value_expression a b = 2 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l1517_151750


namespace NUMINAMATH_GPT_find_pairs_of_real_numbers_l1517_151715

theorem find_pairs_of_real_numbers (x y : ℝ) :
  (∀ n : ℕ, n > 0 → x * ⌊n * y⌋ = y * ⌊n * x⌋) →
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_of_real_numbers_l1517_151715


namespace NUMINAMATH_GPT_positive_diff_solutions_abs_eq_12_l1517_151709

theorem positive_diff_solutions_abs_eq_12 : 
  ∀ (x1 x2 : ℤ), (|x1 - 4| = 12) ∧ (|x2 - 4| = 12) ∧ (x1 > x2) → (x1 - x2 = 24) :=
by
  sorry

end NUMINAMATH_GPT_positive_diff_solutions_abs_eq_12_l1517_151709


namespace NUMINAMATH_GPT_average_score_of_entire_class_l1517_151742

theorem average_score_of_entire_class :
  ∀ (num_students num_boys : ℕ) (avg_score_girls avg_score_boys : ℝ),
  num_students = 50 →
  num_boys = 20 →
  avg_score_girls = 85 →
  avg_score_boys = 80 →
  (avg_score_boys * num_boys + avg_score_girls * (num_students - num_boys)) / num_students = 83 :=
by
  intros num_students num_boys avg_score_girls avg_score_boys
  sorry

end NUMINAMATH_GPT_average_score_of_entire_class_l1517_151742


namespace NUMINAMATH_GPT_series_sum_correct_l1517_151793

open Classical

noncomputable def series_sum : ℝ :=
  ∑' k : ℕ, (2 * (k+1)) / 4^(k+1)

theorem series_sum_correct :
  series_sum = 8 / 9 :=
by
  sorry

end NUMINAMATH_GPT_series_sum_correct_l1517_151793


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1517_151728

theorem parabola_focus_coordinates (y x : ℝ) (h : y^2 = 8 * x) : (x, y) = (2, 0) :=
sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1517_151728


namespace NUMINAMATH_GPT_islanders_liars_l1517_151738

theorem islanders_liars (n : ℕ) (h : n = 450) : (∃ L : ℕ, (L = 150 ∨ L = 450)) :=
sorry

end NUMINAMATH_GPT_islanders_liars_l1517_151738


namespace NUMINAMATH_GPT_problem_M_plus_N_l1517_151785

theorem problem_M_plus_N (M N : ℝ) (H1 : 4/7 = M/77) (H2 : 4/7 = 98/(N^2)) : M + N = 57.1 := 
sorry

end NUMINAMATH_GPT_problem_M_plus_N_l1517_151785


namespace NUMINAMATH_GPT_range_of_ab_l1517_151725

theorem range_of_ab (a b : ℝ) 
  (h1: ∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 1 = 0 → (2 * a * x - b * y + 2 = 0)) : 
  ab ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_ab_l1517_151725


namespace NUMINAMATH_GPT_prove_composite_k_l1517_151737

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := ∃ p q, p > 1 ∧ q > 1 ∧ n = p * q

def problem_statement (a b c d : ℕ) (h : a * b = c * d) : Prop :=
  is_composite (a^1984 + b^1984 + c^1984 + d^1984)

-- The theorem to prove
theorem prove_composite_k (a b c d : ℕ) (h : a * b = c * d) : 
  problem_statement a b c d h := sorry

end NUMINAMATH_GPT_prove_composite_k_l1517_151737


namespace NUMINAMATH_GPT_four_numbers_are_perfect_squares_l1517_151735

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem four_numbers_are_perfect_squares (a b c d : ℕ) (h1 : is_perfect_square (a * b * c))
                                                      (h2 : is_perfect_square (a * c * d))
                                                      (h3 : is_perfect_square (b * c * d))
                                                      (h4 : is_perfect_square (a * b * d)) : 
                                                      is_perfect_square a ∧
                                                      is_perfect_square b ∧
                                                      is_perfect_square c ∧
                                                      is_perfect_square d :=
by
  sorry

end NUMINAMATH_GPT_four_numbers_are_perfect_squares_l1517_151735


namespace NUMINAMATH_GPT_original_number_l1517_151797

-- Define the three-digit number and its permutations under certain conditions.
-- Prove the original number given the specific conditions stated.
theorem original_number (a b c : ℕ)
  (ha : a % 2 = 1) -- a being odd
  (m : ℕ := 100 * a + 10 * b + c)
  (sum_permutations : 100*a + 10*b + c + 100*a + 10*c + b + 100*b + 10*c + a + 
                      100*c + 10*a + b + 100*b + 10*a + c + 100*c + 10*b + a = 3300) :
  m = 192 := 
sorry

end NUMINAMATH_GPT_original_number_l1517_151797


namespace NUMINAMATH_GPT_max_pieces_of_pie_l1517_151753

theorem max_pieces_of_pie : ∃ (PIE PIECE : ℕ), 10000 ≤ PIE ∧ PIE < 100000
  ∧ 10000 ≤ PIECE ∧ PIECE < 100000
  ∧ ∃ (n : ℕ), n = 7 ∧ PIE = n * PIECE := by
  sorry

end NUMINAMATH_GPT_max_pieces_of_pie_l1517_151753


namespace NUMINAMATH_GPT_TomTotalWeight_l1517_151760

def TomWeight : ℝ := 150
def HandWeight (personWeight: ℝ) : ℝ := 1.5 * personWeight
def VestWeight (personWeight: ℝ) : ℝ := 0.5 * personWeight
def TotalHandWeight (handWeight: ℝ) : ℝ := 2 * handWeight
def TotalWeight (totalHandWeight vestWeight: ℝ) : ℝ := totalHandWeight + vestWeight

theorem TomTotalWeight : TotalWeight (TotalHandWeight (HandWeight TomWeight)) (VestWeight TomWeight) = 525 := 
by
  sorry

end NUMINAMATH_GPT_TomTotalWeight_l1517_151760


namespace NUMINAMATH_GPT_units_digit_of_42_pow_3_add_24_pow_3_l1517_151771

theorem units_digit_of_42_pow_3_add_24_pow_3 :
    (42 ^ 3 + 24 ^ 3) % 10 = 2 :=
by
    have units_digit_42 := (42 % 10 = 2)
    have units_digit_24 := (24 % 10 = 4)
    sorry

end NUMINAMATH_GPT_units_digit_of_42_pow_3_add_24_pow_3_l1517_151771


namespace NUMINAMATH_GPT_boys_without_notebooks_l1517_151730

theorem boys_without_notebooks
  (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (h1 : total_boys = 24) (h2 : students_with_notebooks = 30) (h3 : girls_with_notebooks = 17) :
  total_boys - (students_with_notebooks - girls_with_notebooks) = 11 :=
by
  sorry

end NUMINAMATH_GPT_boys_without_notebooks_l1517_151730


namespace NUMINAMATH_GPT_total_swim_distance_five_weeks_total_swim_time_five_weeks_l1517_151763

-- Definitions of swim distances and times based on Jasmine's routine 
def monday_laps : ℕ := 10
def tuesday_laps : ℕ := 15
def tuesday_aerobics_time : ℕ := 20
def wednesday_laps : ℕ := 12
def wednesday_time_per_lap : ℕ := 2
def thursday_laps : ℕ := 18
def friday_laps : ℕ := 20

-- Proving total swim distance for five weeks
theorem total_swim_distance_five_weeks : (5 * (monday_laps + tuesday_laps + wednesday_laps + thursday_laps + friday_laps)) = 375 := 
by 
  sorry

-- Proving total swim time for five weeks (partially solvable)
theorem total_swim_time_five_weeks : (5 * (tuesday_aerobics_time + wednesday_laps * wednesday_time_per_lap)) = 220 := 
by 
  sorry

end NUMINAMATH_GPT_total_swim_distance_five_weeks_total_swim_time_five_weeks_l1517_151763


namespace NUMINAMATH_GPT_triangle_area_l1517_151701

/-- Define the area of a triangle with one side of length 13, an opposite angle of 60 degrees, and side ratio 4:3. -/
theorem triangle_area (a b c : ℝ) (A : ℝ) (S : ℝ) 
  (h_a : a = 13)
  (h_A : A = Real.pi / 3)
  (h_bc_ratio : b / c = 4 / 3)
  (h_cos_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  (h_area : S = 1 / 2 * b * c * Real.sin A) :
  S = 39 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1517_151701


namespace NUMINAMATH_GPT_tangent_line_value_l1517_151790

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem tangent_line_value (a b : ℝ) (h : a ≤ 0) 
  (h_tangent : ∀ x : ℝ, f a x = 2 * x + b) : a - 2 * b = 2 :=
sorry

end NUMINAMATH_GPT_tangent_line_value_l1517_151790


namespace NUMINAMATH_GPT_carl_marbles_l1517_151757

-- Define initial conditions
def initial_marbles : ℕ := 12
def lost_marbles : ℕ := initial_marbles / 2
def remaining_marbles : ℕ := initial_marbles - lost_marbles
def additional_marbles : ℕ := 10
def new_marbles_from_mother : ℕ := 25

-- Define the final number of marbles Carl will put back in the jar
def total_marbles_put_back : ℕ := remaining_marbles + additional_marbles + new_marbles_from_mother

-- Statement to be proven
theorem carl_marbles : total_marbles_put_back = 41 :=
by
  sorry

end NUMINAMATH_GPT_carl_marbles_l1517_151757


namespace NUMINAMATH_GPT_range_of_a_l1517_151779

def set_A (a : ℝ) : Set ℝ := {-1, 0, a}
def set_B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : (set_A a) ∩ set_B ≠ ∅) : 1/3 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1517_151779


namespace NUMINAMATH_GPT_find_z_l1517_151720

theorem find_z (a b p q : ℝ) (z : ℝ) 
  (cond : (z + a + b = q * (p * z - a - b))) : 
  z = (a + b) * (q + 1) / (p * q - 1) :=
sorry

end NUMINAMATH_GPT_find_z_l1517_151720


namespace NUMINAMATH_GPT_prob_point_closer_to_six_than_zero_l1517_151724

theorem prob_point_closer_to_six_than_zero : 
  let interval_start := 0
  let interval_end := 7
  let closer_to_six := fun x => x > ((interval_start + 6) / 2)
  let total_length := interval_end - interval_start
  let length_closer_to_six := interval_end - (interval_start + 6) / 2
  total_length > 0 -> length_closer_to_six / total_length = 4 / 7 :=
by
  sorry

end NUMINAMATH_GPT_prob_point_closer_to_six_than_zero_l1517_151724


namespace NUMINAMATH_GPT_part1_part2_l1517_151723

-- Define the constants based on given conditions
def cost_price : ℕ := 5
def initial_selling_price : ℕ := 9
def initial_sales_volume : ℕ := 32
def price_increment : ℕ := 2
def sales_decrement : ℕ := 8

-- Part 1: Define the elements 
def selling_price_part1 : ℕ := 11
def profit_per_item_part1 : ℕ := 6
def daily_sales_volume_part1 : ℕ := 24

theorem part1 :
  (selling_price_part1 - cost_price = profit_per_item_part1) ∧ 
  (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price_part1 - initial_selling_price) = daily_sales_volume_part1) := 
by
  sorry

-- Part 2: Define the elements 
def target_daily_profit : ℕ := 140
def selling_price1_part2 : ℕ := 12
def selling_price2_part2 : ℕ := 10

theorem part2 :
  (((selling_price1_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price1_part2 - initial_selling_price)) = target_daily_profit) ∨
  ((selling_price2_part2 - cost_price) *
    (initial_sales_volume - (sales_decrement / price_increment) * 
    (selling_price2_part2 - initial_selling_price)) = target_daily_profit)) :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1517_151723
