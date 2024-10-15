import Mathlib

namespace NUMINAMATH_GPT_value_of_stocks_l409_40959

def initial_investment (bonus : ℕ) (stocks : ℕ) : ℕ := bonus / stocks
def final_value_stock_A (initial : ℕ) : ℕ := initial * 2
def final_value_stock_B (initial : ℕ) : ℕ := initial * 2
def final_value_stock_C (initial : ℕ) : ℕ := initial / 2

theorem value_of_stocks 
    (bonus : ℕ) (stocks : ℕ) (h_bonus : bonus = 900) (h_stocks : stocks = 3) : 
    initial_investment bonus stocks * 2 + initial_investment bonus stocks * 2 + initial_investment bonus stocks / 2 = 1350 :=
by
    sorry

end NUMINAMATH_GPT_value_of_stocks_l409_40959


namespace NUMINAMATH_GPT_original_price_of_books_l409_40991

theorem original_price_of_books (purchase_cost : ℝ) (original_price : ℝ) :
  (purchase_cost = 162) →
  (original_price ≤ 100) ∨ 
  (100 < original_price ∧ original_price ≤ 200 ∧ purchase_cost = original_price * 0.9) ∨ 
  (original_price > 200 ∧ purchase_cost = original_price * 0.8) →
  (original_price = 180 ∨ original_price = 202.5) :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_books_l409_40991


namespace NUMINAMATH_GPT_must_be_true_l409_40943

noncomputable def f (x : ℝ) := |Real.log x|

theorem must_be_true (a b c : ℝ) (h0 : 0 < a) (h1 : a < b) (h2 : b < c) 
                     (h3 : f b < f a) (h4 : f a < f c) :
                     (c > 1) ∧ (1 / c < a) ∧ (a < 1) ∧ (a < b) ∧ (b < 1 / a) :=
by
  sorry

end NUMINAMATH_GPT_must_be_true_l409_40943


namespace NUMINAMATH_GPT_selling_price_correct_l409_40906

theorem selling_price_correct (C P_rate : ℝ) (hC : C = 50) (hP_rate : P_rate = 0.40) : 
  C + (P_rate * C) = 70 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l409_40906


namespace NUMINAMATH_GPT_trapezoid_perimeter_is_183_l409_40931

-- Declare the lengths of the sides of the trapezoid
def EG : ℕ := 35
def FH : ℕ := 40
def GH : ℕ := 36

-- Declare the relation between the bases EF and GH
def EF : ℕ := 2 * GH

-- The statement of the problem
theorem trapezoid_perimeter_is_183 : EF = 72 ∧ (EG + GH + FH + EF) = 183 := by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_is_183_l409_40931


namespace NUMINAMATH_GPT_problem_l409_40970

-- Define i as the imaginary unit
def i : ℂ := Complex.I

-- The statement to be proved
theorem problem : i * (1 - i) ^ 2 = 2 := by
  sorry

end NUMINAMATH_GPT_problem_l409_40970


namespace NUMINAMATH_GPT_find_naturals_divisibility_l409_40985

theorem find_naturals_divisibility :
  {n : ℕ | (2^n + n) ∣ (8^n + n)} = {1, 2, 4, 6} :=
by sorry

end NUMINAMATH_GPT_find_naturals_divisibility_l409_40985


namespace NUMINAMATH_GPT_addition_of_decimals_l409_40937

theorem addition_of_decimals : (0.3 + 0.03 : ℝ) = 0.33 := by
  sorry

end NUMINAMATH_GPT_addition_of_decimals_l409_40937


namespace NUMINAMATH_GPT_mean_temperature_is_correct_l409_40936

def temperatures : List ℤ := [-8, -6, -3, -3, 0, 4, -1]
def mean_temperature (temps : List ℤ) : ℚ := (temps.sum : ℚ) / temps.length

theorem mean_temperature_is_correct :
  mean_temperature temperatures = -17 / 7 :=
by
  sorry

end NUMINAMATH_GPT_mean_temperature_is_correct_l409_40936


namespace NUMINAMATH_GPT_max_value_at_x0_l409_40996

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (1 + x) - Real.log x

theorem max_value_at_x0 {x0 : ℝ} (h : ∃ x0, ∀ x, f x ≤ f x0) : 
  f x0 = x0 :=
sorry

end NUMINAMATH_GPT_max_value_at_x0_l409_40996


namespace NUMINAMATH_GPT_divide_value_l409_40950

def divide (a b c : ℝ) : ℝ := |b^2 - 5 * a * c|

theorem divide_value : divide 2 (-3) 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_divide_value_l409_40950


namespace NUMINAMATH_GPT_leading_coefficient_of_f_l409_40941

noncomputable def polynomial : Type := ℕ → ℝ

def satisfies_condition (f : polynomial) : Prop :=
  ∀ (x : ℕ), f (x + 1) - f x = 6 * x + 4

theorem leading_coefficient_of_f (f : polynomial) (h : satisfies_condition f) : 
  ∃ a b c : ℝ, (∀ (x : ℕ), f x = a * (x^2) + b * x + c) ∧ a = 3 := 
by
  sorry

end NUMINAMATH_GPT_leading_coefficient_of_f_l409_40941


namespace NUMINAMATH_GPT_casey_marathon_time_l409_40904

theorem casey_marathon_time (C : ℝ) (h : (C + (4 / 3) * C) / 2 = 7) : C = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_casey_marathon_time_l409_40904


namespace NUMINAMATH_GPT_bob_pays_more_than_samantha_l409_40911

theorem bob_pays_more_than_samantha
  (total_slices : ℕ := 12)
  (cost_plain_pizza : ℝ := 12)
  (cost_olives : ℝ := 3)
  (slices_one_third_pizza : ℕ := total_slices / 3)
  (total_cost : ℝ := cost_plain_pizza + cost_olives)
  (cost_per_slice : ℝ := total_cost / total_slices)
  (bob_slices_total : ℕ := slices_one_third_pizza + 3)
  (samantha_slices_total : ℕ := total_slices - bob_slices_total)
  (bob_total_cost : ℝ := bob_slices_total * cost_per_slice)
  (samantha_total_cost : ℝ := samantha_slices_total * cost_per_slice) :
  bob_total_cost - samantha_total_cost = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_bob_pays_more_than_samantha_l409_40911


namespace NUMINAMATH_GPT_smallest_number_of_ten_consecutive_natural_numbers_l409_40917

theorem smallest_number_of_ten_consecutive_natural_numbers 
  (x : ℕ) 
  (h : 6 * x + 39 = 2 * (4 * x + 6) + 15) : 
  x = 6 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_number_of_ten_consecutive_natural_numbers_l409_40917


namespace NUMINAMATH_GPT_range_of_k_l409_40954

open Set

variable {k : ℝ}

def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < k + 1}

theorem range_of_k (h : (compl A) ∩ B k ≠ ∅) : 0 < k ∧ k < 3 := sorry

end NUMINAMATH_GPT_range_of_k_l409_40954


namespace NUMINAMATH_GPT_chess_competition_l409_40960

theorem chess_competition (W M : ℕ) 
  (hW : W * (W - 1) / 2 = 45) 
  (hM : M * 10 = 200) :
  M * (M - 1) / 2 = 190 :=
by
  sorry

end NUMINAMATH_GPT_chess_competition_l409_40960


namespace NUMINAMATH_GPT_find_p_l409_40994

variables {m n p : ℚ}

theorem find_p (h1 : m = 3 * n + 5) (h2 : (m + 2) = 3 * (n + p) + 5) : p = 2 / 3 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_find_p_l409_40994


namespace NUMINAMATH_GPT_product_of_roots_l409_40956

theorem product_of_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 3 / x = y + 3 / y) : x * y = 3 :=
sorry

end NUMINAMATH_GPT_product_of_roots_l409_40956


namespace NUMINAMATH_GPT_problem_min_value_problem_inequality_range_l409_40984

theorem problem_min_value (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) :
  (1 / a + 4 / b) ≥ 9 :=
sorry

theorem problem_inequality_range (a b : ℝ) (h : a + b = 1) (ha : a > 0) (hb : b > 0) (x : ℝ) :
  (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| ↔ -7 ≤ x ∧ x ≤ 11 :=
sorry

end NUMINAMATH_GPT_problem_min_value_problem_inequality_range_l409_40984


namespace NUMINAMATH_GPT_largest_possible_difference_l409_40905

theorem largest_possible_difference 
  (weight_A weight_B weight_C : ℝ)
  (hA : 24.9 ≤ weight_A ∧ weight_A ≤ 25.1)
  (hB : 24.8 ≤ weight_B ∧ weight_B ≤ 25.2)
  (hC : 24.7 ≤ weight_C ∧ weight_C ≤ 25.3) :
  ∃ w1 w2 : ℝ, (w1 = weight_C ∧ w2 = weight_C ∧ abs (w1 - w2) = 0.6) :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_difference_l409_40905


namespace NUMINAMATH_GPT_solution_xy_l409_40900

noncomputable def find_xy (x y : ℚ) : Prop :=
  (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1 / 3

theorem solution_xy :
  find_xy (10 + 1 / 3) (10 + 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_xy_l409_40900


namespace NUMINAMATH_GPT_ram_account_balance_increase_l409_40940

theorem ram_account_balance_increase 
  (initial_deposit : ℕ := 500)
  (first_year_balance : ℕ := 600)
  (second_year_percentage_increase : ℕ := 32)
  (second_year_balance : ℕ := initial_deposit + initial_deposit * second_year_percentage_increase / 100) 
  (second_year_increase : ℕ := second_year_balance - first_year_balance) 
  : (second_year_increase * 100 / first_year_balance) = 10 := 
sorry

end NUMINAMATH_GPT_ram_account_balance_increase_l409_40940


namespace NUMINAMATH_GPT_football_combinations_l409_40975

theorem football_combinations : 
  ∃ (W D L : ℕ), W + D + L = 15 ∧ 3 * W + D = 33 ∧ 
  (9 ≤ W ∧ W ≤ 11) ∧
  (W = 9 → D = 6 ∧ L = 0) ∧
  (W = 10 → D = 3 ∧ L = 2) ∧
  (W = 11 → D = 0 ∧ L = 4) :=
sorry

end NUMINAMATH_GPT_football_combinations_l409_40975


namespace NUMINAMATH_GPT_blender_customers_l409_40983

variable (p_t p_b : ℕ) (c_t c_b : ℕ) (k : ℕ)

-- Define the conditions
def condition_toaster_popularity : p_t = 20 := sorry
def condition_toaster_cost : c_t = 300 := sorry
def condition_blender_cost : c_b = 450 := sorry
def condition_inverse_proportionality : p_t * c_t = k := sorry

-- Proof goal: number of customers who would buy the blender
theorem blender_customers : p_b = 13 :=
by
  have h1 : p_t * c_t = 6000 := by sorry -- Using the given conditions
  have h2 : p_b * c_b = 6000 := by sorry -- Assumption for the same constant k
  have h3 : c_b = 450 := sorry
  have h4 : p_b = 6000 / 450 := by sorry
  have h5 : p_b = 13 := by sorry
  exact h5

end NUMINAMATH_GPT_blender_customers_l409_40983


namespace NUMINAMATH_GPT_solve_inequality_l409_40965

theorem solve_inequality (x : ℝ) : 
  (-9 * x^2 + 6 * x + 15 > 0) ↔ (x > -1 ∧ x < 5/3) := 
sorry

end NUMINAMATH_GPT_solve_inequality_l409_40965


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l409_40903

theorem gcd_of_three_numbers (a b c : ℕ) (h1 : a = 15378) (h2 : b = 21333) (h3 : c = 48906) :
  Nat.gcd (Nat.gcd a b) c = 3 :=
by
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l409_40903


namespace NUMINAMATH_GPT_count_angles_l409_40946

open Real

noncomputable def isGeometricSequence (a b c : ℝ) : Prop :=
(a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ (a / b = b / c ∨ b / a = a / c ∨ c / a = a / b)

theorem count_angles (h1 : ∀ θ : ℝ, 0 < θ ∧ θ < 2 * π → (sin θ * cos θ = tan θ) ∨ (sin θ ^ 3 = cos θ ^ 2)) :
  ∃ n : ℕ, 
    (∀ θ : ℝ, 0 < θ ∧ θ < 2 * π ∧ (θ % (π/2) ≠ 0) → isGeometricSequence (sin θ) (cos θ) (tan θ) ) → 
    n = 6 := 
sorry

end NUMINAMATH_GPT_count_angles_l409_40946


namespace NUMINAMATH_GPT_solve_system_of_equations_l409_40998

theorem solve_system_of_equations :
  ∃ (x y: ℝ), (x - y - 1 = 0) ∧ (4 * (x - y) - y = 0) ∧ (x = 5) ∧ (y = 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l409_40998


namespace NUMINAMATH_GPT_jerry_clock_reading_l409_40992

noncomputable def clock_reading_after_pills (pills : ℕ) (start_time : ℕ) (interval : ℕ) : ℕ :=
(start_time + (pills - 1) * interval) % 12

theorem jerry_clock_reading :
  clock_reading_after_pills 150 12 5 = 1 :=
by
  sorry

end NUMINAMATH_GPT_jerry_clock_reading_l409_40992


namespace NUMINAMATH_GPT_quadratic_solution_l409_40921

noncomputable def g (x : ℝ) : ℝ := x^2 + 2021 * x + 18

theorem quadratic_solution : ∀ x : ℝ, g (g x + x + 1) / g x = x^2 + 2023 * x + 2040 :=
by
  intros
  sorry

end NUMINAMATH_GPT_quadratic_solution_l409_40921


namespace NUMINAMATH_GPT_part1_part2_l409_40974

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

-- Statement 1: If f(x) is an odd function, then a = 1.
theorem part1 (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) → a = 1 :=
sorry

-- Statement 2: If f(x) is defined on [-4, +∞), and for all x in the domain, 
-- f(cos(x) + b + 1/4) ≥ f(sin^2(x) - b - 3), then b ∈ [-1,1].
theorem part2 (a : ℝ) (b : ℝ) :
  (∀ x : ℝ, f a (Real.cos x + b + 1/4) ≥ f a (Real.sin x ^ 2 - b - 3)) ∧
  (∀ x : ℝ, -4 ≤ x) ∧ -4 ≤ a ∧ a = 1 → -1 ≤ b ∧ b ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_l409_40974


namespace NUMINAMATH_GPT_g_at_3_eq_19_l409_40919

def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem g_at_3_eq_19 : g 3 = 19 := by
  sorry

end NUMINAMATH_GPT_g_at_3_eq_19_l409_40919


namespace NUMINAMATH_GPT_remainder_T10_mod_5_l409_40942

noncomputable def T : ℕ → ℕ
| 0     => 1
| 1     => 2
| (n+2) => T (n+1) + T n + T n

theorem remainder_T10_mod_5 :
  (T 10) % 5 = 4 :=
sorry

end NUMINAMATH_GPT_remainder_T10_mod_5_l409_40942


namespace NUMINAMATH_GPT_lifting_to_bodyweight_ratio_l409_40987

variable (t : ℕ) (w : ℕ) (p : ℕ) (delta_w : ℕ)

def lifting_total_after_increase (t : ℕ) (p : ℕ) : ℕ :=
  t + (t * p / 100)

def bodyweight_after_increase (w : ℕ) (delta_w : ℕ) : ℕ :=
  w + delta_w

theorem lifting_to_bodyweight_ratio (h_t : t = 2200) (h_w : w = 245) (h_p : p = 15) (h_delta_w : delta_w = 8) :
  lifting_total_after_increase t p / bodyweight_after_increase w delta_w = 10 :=
  by
    -- Use the given conditions
    rw [h_t, h_w, h_p, h_delta_w]
    -- Calculation steps are omitted, directly providing the final assertion
    sorry

end NUMINAMATH_GPT_lifting_to_bodyweight_ratio_l409_40987


namespace NUMINAMATH_GPT_complex_expression_identity_l409_40928

noncomputable section

variable (x y : ℂ) 

theorem complex_expression_identity (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x^2 + x*y + y^2 = 0) : 
  (x / (x + y)) ^ 1990 + (y / (x + y)) ^ 1990 = -1 := 
by 
  sorry

end NUMINAMATH_GPT_complex_expression_identity_l409_40928


namespace NUMINAMATH_GPT_find_base_b_l409_40952

theorem find_base_b :
  ∃ b : ℕ, (b > 7) ∧ (b > 10) ∧ (b > 8) ∧ (b > 12) ∧ 
    (4 + 3 = 7) ∧ ((2 + 7 + 1) % b = 3) ∧ ((3 + 4 + 1) % b = 5) ∧ 
    ((5 + 6 + 1) % b = 2) ∧ (1 + 1 = 2)
    ∧ b = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_base_b_l409_40952


namespace NUMINAMATH_GPT_find_f_100_l409_40909

theorem find_f_100 (f : ℝ → ℝ) (k : ℝ) (h_nonzero : k ≠ 0) 
(h_func : ∀ x y : ℝ, 0 < x → 0 < y → k * (x * f y - y * f x) = f (x / y)) : 
f 100 = 0 := 
by
  sorry

end NUMINAMATH_GPT_find_f_100_l409_40909


namespace NUMINAMATH_GPT_parabola_maximum_value_l409_40976

noncomputable def maximum_parabola (a b c : ℝ) (h := -b / (2*a)) (k := a * h^2 + b * h + c) : Prop :=
  ∀ (x : ℝ), a ≠ 0 → b = 12 → c = 4 → a = -3 → k = 16

theorem parabola_maximum_value : maximum_parabola (-3) 12 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_maximum_value_l409_40976


namespace NUMINAMATH_GPT_triangle_height_l409_40968

theorem triangle_height (area base : ℝ) (h : ℝ) (h_area : area = 46) (h_base : base = 10) 
  (h_formula : area = (base * h) / 2) : 
  h = 9.2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l409_40968


namespace NUMINAMATH_GPT_pizza_topping_combinations_l409_40927

theorem pizza_topping_combinations :
  (Nat.choose 7 3) = 35 :=
sorry

end NUMINAMATH_GPT_pizza_topping_combinations_l409_40927


namespace NUMINAMATH_GPT_vessel_reaches_boat_in_shortest_time_l409_40981

-- Define the given conditions as hypotheses
variable (dist_AC : ℝ) (angle_C : ℝ) (speed_CB : ℝ) (angle_B : ℝ) (speed_A : ℝ)

-- Assign values to variables based on the problem statement
def vessel_distress_boat_condition : Prop :=
  dist_AC = 10 ∧ angle_C = 45 ∧ speed_CB = 9 ∧ angle_B = 105 ∧ speed_A = 21

-- Define the time (in minutes) for the vessel to reach the fishing boat
noncomputable def shortest_time_to_reach_boat : ℝ :=
  25

-- The theorem that we need to prove given the conditions
theorem vessel_reaches_boat_in_shortest_time :
  vessel_distress_boat_condition dist_AC angle_C speed_CB angle_B speed_A → 
  shortest_time_to_reach_boat = 25 := by
    intros
    sorry

end NUMINAMATH_GPT_vessel_reaches_boat_in_shortest_time_l409_40981


namespace NUMINAMATH_GPT_saree_blue_stripes_l409_40948

theorem saree_blue_stripes (brown_stripes gold_stripes blue_stripes : ℕ) 
    (h1 : brown_stripes = 4)
    (h2 : gold_stripes = 3 * brown_stripes)
    (h3 : blue_stripes = 5 * gold_stripes) : 
    blue_stripes = 60 := 
by
  sorry

end NUMINAMATH_GPT_saree_blue_stripes_l409_40948


namespace NUMINAMATH_GPT_girl_needs_120_oranges_l409_40961

-- Define the cost and selling prices per pack
def cost_per_pack : ℤ := 15   -- cents
def oranges_per_pack_cost : ℤ := 4
def sell_per_pack : ℤ := 30   -- cents
def oranges_per_pack_sell : ℤ := 6

-- Define the target profit
def target_profit : ℤ := 150  -- cents

-- Calculate the cost price per orange
def cost_per_orange : ℚ := cost_per_pack / oranges_per_pack_cost

-- Calculate the selling price per orange
def sell_per_orange : ℚ := sell_per_pack / oranges_per_pack_sell

-- Calculate the profit per orange
def profit_per_orange : ℚ := sell_per_orange - cost_per_orange

-- Calculate the number of oranges needed to achieve the target profit
def oranges_needed : ℚ := target_profit / profit_per_orange

-- Lean theorem statement
theorem girl_needs_120_oranges :
  oranges_needed = 120 :=
  sorry

end NUMINAMATH_GPT_girl_needs_120_oranges_l409_40961


namespace NUMINAMATH_GPT_volunteers_allocation_scheme_count_l409_40993

theorem volunteers_allocation_scheme_count :
  let volunteers := 6
  let groups_of_two := 2
  let groups_of_one := 2
  let pavilions := 4
  let calculate_combinations (n k : ℕ) := Nat.choose n k
  calculate_combinations volunteers 2 * calculate_combinations (volunteers - 2) 2 * 
  calculate_combinations pavilions 2 * Nat.factorial pavilions = 1080 := by
sorry

end NUMINAMATH_GPT_volunteers_allocation_scheme_count_l409_40993


namespace NUMINAMATH_GPT_ab_squared_non_positive_l409_40930

theorem ab_squared_non_positive (a b : ℝ) (h : 7 * a + 9 * |b| = 0) : a * b^2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_ab_squared_non_positive_l409_40930


namespace NUMINAMATH_GPT_eight_and_five_l409_40924

def my_and (a b : ℕ) : ℕ := (a + b) ^ 2 * (a - b)

theorem eight_and_five : my_and 8 5 = 507 := 
  by sorry

end NUMINAMATH_GPT_eight_and_five_l409_40924


namespace NUMINAMATH_GPT_quadratic_point_value_l409_40990

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h_min : ∀ x : ℝ, a * x^2 + b * x + c ≥ a * (-1)^2 + b * (-1) + c) 
  (h_at_min : a * (-1)^2 + b * (-1) + c = -3)
  (h_point : a * (1)^2 + b * (1) + c = 7) : 
  a * (3)^2 + b * (3) + c = 37 :=
sorry

end NUMINAMATH_GPT_quadratic_point_value_l409_40990


namespace NUMINAMATH_GPT_kim_shoes_l409_40964

variable (n : ℕ)

theorem kim_shoes : 
  (∀ n, 2 * n = 6 → (1 : ℚ) / (2 * n - 1) = (1 : ℚ) / 5 → n = 3) := 
sorry

end NUMINAMATH_GPT_kim_shoes_l409_40964


namespace NUMINAMATH_GPT_tens_digit_of_binary_result_l409_40944

def digits_tens_digit_subtraction (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) : ℕ :=
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let difference := original_number - reversed_number
  (difference % 100) / 10

theorem tens_digit_of_binary_result (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) :
  digits_tens_digit_subtraction a b c h1 h2 = 9 :=
sorry

end NUMINAMATH_GPT_tens_digit_of_binary_result_l409_40944


namespace NUMINAMATH_GPT_triangle_cosines_identity_l409_40902

theorem triangle_cosines_identity 
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * Real.cos (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))) / a) + 
  (c^2 * Real.cos (Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))) / b) + 
  (a^2 * Real.cos (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / c) = 
  (a^4 + b^4 + c^4) / (2 * a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_triangle_cosines_identity_l409_40902


namespace NUMINAMATH_GPT_find_linear_function_l409_40926

theorem find_linear_function (a : ℝ) (a_pos : 0 < a) :
  ∃ (b : ℝ), ∀ (f : ℕ → ℝ),
  (∀ (k m : ℕ), (a * m ≤ k ∧ k < (a + 1) * m) → f (k + m) = f k + f m) →
  ∀ n : ℕ, f n = b * n :=
sorry

end NUMINAMATH_GPT_find_linear_function_l409_40926


namespace NUMINAMATH_GPT_chinese_chess_sets_l409_40951

theorem chinese_chess_sets (x y : ℕ) 
  (h1 : 24 * x + 18 * y = 300) 
  (h2 : x + y = 14) : 
  y = 6 := 
sorry

end NUMINAMATH_GPT_chinese_chess_sets_l409_40951


namespace NUMINAMATH_GPT_sum_of_digits_625_base5_l409_40938

def sum_of_digits_base_5 (n : ℕ) : ℕ :=
  let rec sum_digits n :=
    if n = 0 then 0
    else (n % 5) + sum_digits (n / 5)
  sum_digits n

theorem sum_of_digits_625_base5 : sum_of_digits_base_5 625 = 5 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_625_base5_l409_40938


namespace NUMINAMATH_GPT_train_speed_correct_l409_40963

def train_length : ℝ := 250  -- length of the train in meters
def time_to_pass : ℝ := 18  -- time to pass a tree in seconds
def speed_of_train_km_hr : ℝ := 50  -- speed of the train in km/hr

theorem train_speed_correct :
  (train_length / time_to_pass) * (3600 / 1000) = speed_of_train_km_hr :=
by
  sorry

end NUMINAMATH_GPT_train_speed_correct_l409_40963


namespace NUMINAMATH_GPT_intersecting_lines_a_plus_b_l409_40986

theorem intersecting_lines_a_plus_b :
  ∃ (a b : ℝ), (∀ x y : ℝ, (x = 1 / 3 * y + a) ∧ (y = 1 / 3 * x + b) → (x = 3 ∧ y = 4)) ∧ a + b = 14 / 3 :=
sorry

end NUMINAMATH_GPT_intersecting_lines_a_plus_b_l409_40986


namespace NUMINAMATH_GPT_max_product_two_integers_sum_300_l409_40982

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end NUMINAMATH_GPT_max_product_two_integers_sum_300_l409_40982


namespace NUMINAMATH_GPT_unique_solution_cond_l409_40962

open Real

theorem unique_solution_cond (a c : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = c * x + 2) ↔ c ≠ 4 :=
by sorry

end NUMINAMATH_GPT_unique_solution_cond_l409_40962


namespace NUMINAMATH_GPT_range_g_l409_40997

noncomputable def g (x : Real) : Real := (Real.sin x)^6 + (Real.cos x)^4

theorem range_g :
  ∃ (a : Real), 
    (∀ x : Real, g x ≥ a ∧ g x ≤ 1) ∧
    (∀ y : Real, y < a → ¬∃ x : Real, g x = y) :=
sorry

end NUMINAMATH_GPT_range_g_l409_40997


namespace NUMINAMATH_GPT_move_point_right_l409_40914

theorem move_point_right (x y : ℝ) (h₁ : x = 1) (h₂ : y = 1) (dx : ℝ) (h₃ : dx = 2) : (x + dx, y) = (3, 1) :=
by
  rw [h₁, h₂, h₃]
  simp
  sorry

end NUMINAMATH_GPT_move_point_right_l409_40914


namespace NUMINAMATH_GPT_number_of_zero_points_l409_40957

theorem number_of_zero_points (f : ℝ → ℝ) (h_odd : ∀ x, f x = -f (-x)) (h_period : ∀ x, f (x - π) = f (x + π)) :
  ∃ (points : Finset ℝ), (∀ x ∈ points, 0 ≤ x ∧ x ≤ 8 ∧ f x = 0) ∧ points.card = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_zero_points_l409_40957


namespace NUMINAMATH_GPT_fraction_representing_repeating_decimal_l409_40916

theorem fraction_representing_repeating_decimal (x a b : ℕ) (h : x = 35) (h1 : 100 * x - x = 35) 
(h2 : ∃ (a b : ℕ), x = a / b ∧ gcd a b = 1 ∧ a + b = 134) : a + b = 134 := 
sorry

end NUMINAMATH_GPT_fraction_representing_repeating_decimal_l409_40916


namespace NUMINAMATH_GPT_albrecht_correct_substitution_l409_40973

theorem albrecht_correct_substitution (a b : ℕ) (h : (a + 2 * b - 3)^2 = a^2 + 4 * b^2 - 9) :
  (a = 2 ∧ b = 15) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 3) ∨ (a = 15 ∧ b = 2) :=
by
  -- The proof will be filled in here
  sorry

end NUMINAMATH_GPT_albrecht_correct_substitution_l409_40973


namespace NUMINAMATH_GPT_simplify_expression_l409_40910

theorem simplify_expression :
  5 * (18 / 7) * (21 / -45) = -6 / 5 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l409_40910


namespace NUMINAMATH_GPT_sarahs_loan_amount_l409_40988

theorem sarahs_loan_amount 
  (down_payment : ℕ := 10000)
  (monthly_payment : ℕ := 600)
  (repayment_years : ℕ := 5)
  (interest_rate : ℚ := 0) : down_payment + (monthly_payment * (12 * repayment_years)) = 46000 :=
by
  sorry

end NUMINAMATH_GPT_sarahs_loan_amount_l409_40988


namespace NUMINAMATH_GPT_prove_percent_liquid_X_in_new_solution_l409_40901

variable (initial_solution total_weight_x total_weight_y total_weight_new)

def percent_liquid_X_in_new_solution : Prop :=
  let liquid_X_in_initial := 0.45 * 12
  let water_in_initial := 0.55 * 12
  let remaining_liquid_X := liquid_X_in_initial
  let remaining_water := water_in_initial - 5
  let liquid_X_in_added := 0.45 * 7
  let water_in_added := 0.55 * 7
  let total_liquid_X := remaining_liquid_X + liquid_X_in_added
  let total_water := remaining_water + water_in_added
  let total_weight := total_liquid_X + total_water
  (total_liquid_X / total_weight) * 100 = 61.07

theorem prove_percent_liquid_X_in_new_solution :
  percent_liquid_X_in_new_solution := by
  sorry

end NUMINAMATH_GPT_prove_percent_liquid_X_in_new_solution_l409_40901


namespace NUMINAMATH_GPT_sqrt_one_div_four_is_one_div_two_l409_40908

theorem sqrt_one_div_four_is_one_div_two : Real.sqrt (1 / 4) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_one_div_four_is_one_div_two_l409_40908


namespace NUMINAMATH_GPT_lambda_property_l409_40995
open Int

noncomputable def lambda : ℝ := 1 + Real.sqrt 2

theorem lambda_property (n : ℕ) (hn : n > 0) :
  2 * ⌊lambda * n⌋ = 1 - n + ⌊lambda * ⌊lambda * n⌋⌋ :=
sorry

end NUMINAMATH_GPT_lambda_property_l409_40995


namespace NUMINAMATH_GPT_length_of_flat_terrain_l409_40958

theorem length_of_flat_terrain (total_time : ℚ)
  (total_distance : ℕ)
  (speed_uphill speed_flat speed_downhill : ℚ)
  (distance_uphill distance_flat : ℕ) :
  total_time = 116 / 60 ∧
  total_distance = distance_uphill + distance_flat + (total_distance - distance_uphill - distance_flat) ∧
  speed_uphill = 4 ∧
  speed_flat = 5 ∧
  speed_downhill = 6 ∧
  distance_uphill ≥ 0 ∧
  distance_flat ≥ 0 ∧
  distance_uphill + distance_flat ≤ total_distance →
  distance_flat = 3 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_flat_terrain_l409_40958


namespace NUMINAMATH_GPT_negative_three_degrees_below_zero_l409_40966

-- Definitions based on conditions
def positive_temperature (t : ℤ) : Prop := t > 0
def negative_temperature (t : ℤ) : Prop := t < 0
def above_zero (t : ℤ) : Prop := positive_temperature t
def below_zero (t : ℤ) : Prop := negative_temperature t

-- Example given in conditions
def ten_degrees_above_zero := above_zero 10

-- Lean 4 statement for the proof
theorem negative_three_degrees_below_zero : below_zero (-3) :=
by
  sorry

end NUMINAMATH_GPT_negative_three_degrees_below_zero_l409_40966


namespace NUMINAMATH_GPT_smooth_transition_l409_40979

theorem smooth_transition (R : ℝ) (x₀ y₀ : ℝ) :
  ∃ m : ℝ, ∀ (x y : ℝ), (x - x₀)^2 + (y - y₀)^2 = R^2 → y - y₀ = m * (x - x₀) :=
sorry

end NUMINAMATH_GPT_smooth_transition_l409_40979


namespace NUMINAMATH_GPT_unique_reconstruction_l409_40922

-- Definition of the sums on the edges given the face values
variables (a b c d e f : ℤ)

-- The 12 edge sums
variables (e₁ e₂ e₃ e₄ e₅ e₆ e₇ e₈ e₉ e₁₀ e₁₁ e₁₂ : ℤ)
variables (h₁ : e₁ = a + b) (h₂ : e₂ = a + c) (h₃ : e₃ = a + d) 
          (h₄ : e₄ = a + e) (h₅ : e₅ = b + c) (h₆ : e₆ = b + f) 
          (h₇ : e₇ = c + f) (h₈ : e₈ = d + f) (h₉ : e₉ = d + e)
          (h₁₀ : e₁₀ = e + f) (h₁₁ : e₁₁ = b + d) (h₁₂ : e₁₂ = c + e)

-- Proving that the face values can be uniquely determined given the edge sums
theorem unique_reconstruction :
  ∃ a' b' c' d' e' f' : ℤ, 
    (e₁ = a' + b') ∧ (e₂ = a' + c') ∧ (e₃ = a' + d') ∧ (e₄ = a' + e') ∧ 
    (e₅ = b' + c') ∧ (e₆ = b' + f') ∧ (e₇ = c' + f') ∧ (e₈ = d' + f') ∧ 
    (e₉ = d' + e') ∧ (e₁₀ = e' + f') ∧ (e₁₁ = b' + d') ∧ (e₁₂ = c' + e') ∧ 
    (a = a') ∧ (b = b') ∧ (c = c') ∧ (d = d') ∧ (e = e') ∧ (f = f') := by
  sorry

end NUMINAMATH_GPT_unique_reconstruction_l409_40922


namespace NUMINAMATH_GPT_number_of_proper_subsets_l409_40915

theorem number_of_proper_subsets (S : Finset ℕ) (h : S = {1, 2, 3, 4}) : S.powerset.card - 1 = 15 := by
  sorry

end NUMINAMATH_GPT_number_of_proper_subsets_l409_40915


namespace NUMINAMATH_GPT_jerry_stickers_l409_40969

variable (G F J : ℕ)

theorem jerry_stickers (h1 : F = 18) (h2 : G = F - 6) (h3 : J = 3 * G) : J = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_jerry_stickers_l409_40969


namespace NUMINAMATH_GPT_find_tangent_points_l409_40935

-- Step a: Define the curve and the condition for the tangent line parallel to y = 4x.
def curve (x : ℝ) : ℝ := x^3 + x - 2
def tangent_slope : ℝ := 4

-- Step d: Provide the statement that the coordinates of P₀ are (1, 0) and (-1, -4).
theorem find_tangent_points : 
  ∃ (P₀ : ℝ × ℝ), (curve P₀.1 = P₀.2) ∧ 
                 ((P₀ = (1, 0)) ∨ (P₀ = (-1, -4))) := 
by
  sorry

end NUMINAMATH_GPT_find_tangent_points_l409_40935


namespace NUMINAMATH_GPT_mixed_number_division_l409_40933

theorem mixed_number_division :
  (5 + 1 / 2) / (2 / 11) = 121 / 4 :=
by sorry

end NUMINAMATH_GPT_mixed_number_division_l409_40933


namespace NUMINAMATH_GPT_arithmetic_sequence_ratios_l409_40971

noncomputable def a_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {a_n}
noncomputable def b_n : ℕ → ℚ := sorry -- definition of the arithmetic sequence {b_n}
noncomputable def S_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {a_n}
noncomputable def T_n (n : ℕ) : ℚ := sorry -- definition of the sum of the first n terms of {b_n}

theorem arithmetic_sequence_ratios :
  (∀ n : ℕ, 0 < n → S_n n / T_n n = (7 * n + 1) / (4 * n + 27)) →
  (a_n 7 / b_n 7 = 92 / 79) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratios_l409_40971


namespace NUMINAMATH_GPT_Tile_in_rectangle_R_l409_40949

structure Tile :=
  (top : ℕ)
  (right : ℕ)
  (bottom : ℕ)
  (left : ℕ)

def X : Tile := ⟨5, 3, 6, 2⟩
def Y : Tile := ⟨3, 6, 2, 5⟩
def Z : Tile := ⟨6, 0, 1, 5⟩
def W : Tile := ⟨2, 5, 3, 0⟩

theorem Tile_in_rectangle_R : 
  X.top = 5 ∧ X.right = 3 ∧ X.bottom = 6 ∧ X.left = 2 ∧ 
  Y.top = 3 ∧ Y.right = 6 ∧ Y.bottom = 2 ∧ Y.left = 5 ∧ 
  Z.top = 6 ∧ Z.right = 0 ∧ Z.bottom = 1 ∧ Z.left = 5 ∧ 
  W.top = 2 ∧ W.right = 5 ∧ W.bottom = 3 ∧ W.left = 0 → 
  (∀ rectangle_R : Tile, rectangle_R = W) :=
by sorry

end NUMINAMATH_GPT_Tile_in_rectangle_R_l409_40949


namespace NUMINAMATH_GPT_area_of_rectangle_l409_40932

theorem area_of_rectangle (w d : ℝ) (h_w : w = 4) (h_d : d = 5) : ∃ l : ℝ, (w^2 + l^2 = d^2) ∧ (w * l = 12) :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l409_40932


namespace NUMINAMATH_GPT_triangle_is_isosceles_l409_40939

/-- Given triangle ABC with angles A, B, and C, where C = π - (A + B),
    if 2 * sin A * cos B = sin C, then triangle ABC is an isosceles triangle -/
theorem triangle_is_isosceles
  (A B C : ℝ)
  (hC : C = π - (A + B))
  (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  A = B :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l409_40939


namespace NUMINAMATH_GPT_gcd_1260_924_l409_40999

theorem gcd_1260_924 : Nat.gcd 1260 924 = 84 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1260_924_l409_40999


namespace NUMINAMATH_GPT_distinct_arith_prog_triangles_l409_40929

theorem distinct_arith_prog_triangles (n : ℕ) (h10 : n % 10 = 0) : 
  (3 * n = 180 → ∃ d : ℕ, ∀ a b c, a = n - d ∧ b = n ∧ c = n + d 
  →  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d < 60) :=
by
  sorry

end NUMINAMATH_GPT_distinct_arith_prog_triangles_l409_40929


namespace NUMINAMATH_GPT_geometric_series_second_term_l409_40945

theorem geometric_series_second_term (a : ℝ) (r : ℝ) (sum : ℝ) 
  (h1 : r = 1/4) 
  (h2 : sum = 40) 
  (sum_formula : sum = a / (1 - r)) : a * r = 7.5 :=
by {
  -- Proof to be filled in later
  sorry
}

end NUMINAMATH_GPT_geometric_series_second_term_l409_40945


namespace NUMINAMATH_GPT_arc_length_calculation_l409_40972

theorem arc_length_calculation (C θ : ℝ) (hC : C = 72) (hθ : θ = 45) :
  (θ / 360) * C = 9 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_calculation_l409_40972


namespace NUMINAMATH_GPT_prove_f_2_eq_3_l409_40977

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 1 then 3 * a ^ x else Real.log (2 * x + 4) / Real.log a

theorem prove_f_2_eq_3 (a : ℝ) (h1 : f 1 a = 6) : f 2 a = 3 :=
by
  -- Define the conditions
  have h1 : 3 * a = 6 := by simp [f] at h1; assumption
  -- Two subcases: x <= 1 and x > 1
  have : a = 2 := by linarith
  simp [f, this]
  sorry

end NUMINAMATH_GPT_prove_f_2_eq_3_l409_40977


namespace NUMINAMATH_GPT_seahawks_field_goals_l409_40918

-- Defining the conditions as hypotheses
def final_score_seahawks : ℕ := 37
def points_per_touchdown : ℕ := 7
def points_per_fieldgoal : ℕ := 3
def touchdowns_seahawks : ℕ := 4

-- Stating the goal to prove
theorem seahawks_field_goals : 
  (final_score_seahawks - touchdowns_seahawks * points_per_touchdown) / points_per_fieldgoal = 3 := 
by 
  sorry

end NUMINAMATH_GPT_seahawks_field_goals_l409_40918


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l409_40955

def M : Set ℝ := {x | 0 < x ∧ x ≤ 2}

def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ∈ M → a ∈ N) ∧ (∃ a, a ∈ N ∧ a ∉ M) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l409_40955


namespace NUMINAMATH_GPT_graph_fixed_point_l409_40978

theorem graph_fixed_point (f : ℝ → ℝ) (h : f 1 = 1) : f 1 = 1 :=
by
  sorry

end NUMINAMATH_GPT_graph_fixed_point_l409_40978


namespace NUMINAMATH_GPT_solution_set_inequality_l409_40912

theorem solution_set_inequality (t : ℝ) (ht : 0 < t ∧ t < 1) :
  {x : ℝ | x^2 - (t + t⁻¹) * x + 1 < 0} = {x : ℝ | t < x ∧ x < t⁻¹} :=
sorry

end NUMINAMATH_GPT_solution_set_inequality_l409_40912


namespace NUMINAMATH_GPT_find_ordered_pair_l409_40920

theorem find_ordered_pair (x y : ℤ) 
  (h1 : x + y = (7 - x) + (7 - y))
  (h2 : x - y = (x - 2) + (y - 2))
  : (x, y) = (5, 2) := 
sorry

end NUMINAMATH_GPT_find_ordered_pair_l409_40920


namespace NUMINAMATH_GPT_ratio_of_sheep_to_horses_l409_40967

theorem ratio_of_sheep_to_horses (H : ℕ) (hH : 230 * H = 12880) (n_sheep : ℕ) (h_sheep : n_sheep = 56) :
  (n_sheep / H) = 1 := by
  sorry

end NUMINAMATH_GPT_ratio_of_sheep_to_horses_l409_40967


namespace NUMINAMATH_GPT_rex_cards_remaining_l409_40923

theorem rex_cards_remaining
  (nicole_cards : ℕ)
  (cindy_cards : ℕ)
  (rex_cards : ℕ)
  (cards_per_person : ℕ)
  (h1 : nicole_cards = 400)
  (h2 : cindy_cards = 2 * nicole_cards)
  (h3 : rex_cards = (nicole_cards + cindy_cards) / 2)
  (h4 : cards_per_person = rex_cards / 4) :
  cards_per_person = 150 :=
by
  sorry

end NUMINAMATH_GPT_rex_cards_remaining_l409_40923


namespace NUMINAMATH_GPT_triangle_area_l409_40953

theorem triangle_area (a b : ℝ) (C : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : C = π / 3) : 
  (1/2 * a * b * Real.sin C) = (3 * Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l409_40953


namespace NUMINAMATH_GPT_simplify_expression_l409_40934

theorem simplify_expression :
  ((9 * 10^8) * 2^2) / (3 * 2^3 * 10^3) = 150000 := by sorry

end NUMINAMATH_GPT_simplify_expression_l409_40934


namespace NUMINAMATH_GPT_sum_of_undefined_values_l409_40913

theorem sum_of_undefined_values (y : ℝ) :
  (y^2 - 7 * y + 12 = 0) → y = 3 ∨ y = 4 → (3 + 4 = 7) :=
by
  intro hy
  intro hy'
  sorry

end NUMINAMATH_GPT_sum_of_undefined_values_l409_40913


namespace NUMINAMATH_GPT_cars_15th_time_l409_40989

noncomputable def minutes_since_8am (hour : ℕ) (minute : ℕ) : ℕ :=
  hour * 60 + minute

theorem cars_15th_time :
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  total_time = expected_time :=
by
  let initial_time := minutes_since_8am 8 0
  let interval := 5
  let obstacles_time := 3 * 10
  let minutes_passed := (15 - 1) * interval + obstacles_time
  let total_time := initial_time + minutes_passed
  let expected_time := minutes_since_8am 9 40
  show total_time = expected_time
  sorry

end NUMINAMATH_GPT_cars_15th_time_l409_40989


namespace NUMINAMATH_GPT_opposite_of_half_l409_40980

theorem opposite_of_half : -(- (1/2)) = (1/2) := 
by 
  sorry

end NUMINAMATH_GPT_opposite_of_half_l409_40980


namespace NUMINAMATH_GPT_find_factor_l409_40925

-- Definitions based on the conditions
def n : ℤ := 155
def result : ℤ := 110
def constant : ℤ := 200

-- Statement to be proved
theorem find_factor (f : ℤ) (h : n * f - constant = result) : f = 2 := by
  sorry

end NUMINAMATH_GPT_find_factor_l409_40925


namespace NUMINAMATH_GPT_prove_b_eq_d_and_c_eq_e_l409_40947

variable (a b c d e f : ℕ)

-- Define the expressions for A and B as per the problem statement
def A := 10^5 * a + 10^4 * b + 10^3 * c + 10^2 * d + 10 * e + f
def B := 10^5 * f + 10^4 * d + 10^3 * e + 10^2 * b + 10 * c + a

-- Define the condition that A - B is divisible by 271
def divisible_by_271 (n : ℕ) : Prop := ∃ k : ℕ, n = 271 * k

-- Define the main theorem to prove b = d and c = e under the given conditions
theorem prove_b_eq_d_and_c_eq_e
    (h1 : divisible_by_271 (A a b c d e f - B a b c d e f)) :
    b = d ∧ c = e :=
sorry

end NUMINAMATH_GPT_prove_b_eq_d_and_c_eq_e_l409_40947


namespace NUMINAMATH_GPT_sum_g_h_l409_40907

theorem sum_g_h (d g h : ℝ) 
  (h1 : (8 * d^2 - 4 * d + g) * (4 * d^2 + h * d + 7) = 32 * d^4 + (4 * h - 16) * d^3 - (14 * d^2 - 28 * d - 56)) :
  g + h = -8 :=
sorry

end NUMINAMATH_GPT_sum_g_h_l409_40907
