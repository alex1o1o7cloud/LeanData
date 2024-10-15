import Mathlib

namespace NUMINAMATH_GPT_max_sum_of_four_numbers_l2363_236395

theorem max_sum_of_four_numbers : 
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ (2 * a + 3 * b + 2 * c + 3 * d = 2017) ∧ 
    (a + b + c + d = 806) :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_four_numbers_l2363_236395


namespace NUMINAMATH_GPT_fraction_of_jam_eaten_for_dinner_l2363_236319

-- Define the problem
theorem fraction_of_jam_eaten_for_dinner :
  ∃ (J : ℝ) (x : ℝ), 
  J > 0 ∧
  (1 / 3) * J + (x * (2 / 3) * J) + (4 / 7) * J = J ∧
  x = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_jam_eaten_for_dinner_l2363_236319


namespace NUMINAMATH_GPT_group_value_21_le_a_lt_41_l2363_236328

theorem group_value_21_le_a_lt_41 : 
  (∀ a: ℤ, 21 ≤ a ∧ a < 41 → (21 + 41) / 2 = 31) :=
by 
  sorry

end NUMINAMATH_GPT_group_value_21_le_a_lt_41_l2363_236328


namespace NUMINAMATH_GPT_total_cost_accurate_l2363_236342

def price_iphone: ℝ := 800
def price_iwatch: ℝ := 300
def price_ipad: ℝ := 500

def discount_iphone: ℝ := 0.15
def discount_iwatch: ℝ := 0.10
def discount_ipad: ℝ := 0.05

def tax_iphone: ℝ := 0.07
def tax_iwatch: ℝ := 0.05
def tax_ipad: ℝ := 0.06

def cashback: ℝ := 0.02

theorem total_cost_accurate:
  let discounted_auction (price: ℝ) (discount: ℝ) := price * (1 - discount)
  let taxed_auction (price: ℝ) (tax: ℝ) := price * (1 + tax)
  let total_cost :=
    let discount_iphone_cost := discounted_auction price_iphone discount_iphone
    let discount_iwatch_cost := discounted_auction price_iwatch discount_iwatch
    let discount_ipad_cost := discounted_auction price_ipad discount_ipad
    
    let tax_iphone_cost := taxed_auction discount_iphone_cost tax_iphone
    let tax_iwatch_cost := taxed_auction discount_iwatch_cost tax_iwatch
    let tax_ipad_cost := taxed_auction discount_ipad_cost tax_ipad
    
    let total_price := tax_iphone_cost + tax_iwatch_cost + tax_ipad_cost
    total_price * (1 - cashback)
  total_cost = 1484.31 := 
  by sorry

end NUMINAMATH_GPT_total_cost_accurate_l2363_236342


namespace NUMINAMATH_GPT_symmetric_line_l2363_236315

theorem symmetric_line (y : ℝ → ℝ) (h : ∀ x, y x = 2 * x + 1) :
  ∀ x, y (-x) = -2 * x + 1 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_symmetric_line_l2363_236315


namespace NUMINAMATH_GPT_eval_expr_l2363_236320

-- Define the expression
def expr : ℚ := 2 + 3 / (2 + 1 / (2 + 1 / 2))

-- The theorem to prove the evaluation of the expression
theorem eval_expr : expr = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_eval_expr_l2363_236320


namespace NUMINAMATH_GPT_total_revenue_l2363_236370

def price_federal := 50
def price_state := 30
def price_quarterly := 80
def num_federal := 60
def num_state := 20
def num_quarterly := 10

theorem total_revenue : (num_federal * price_federal + num_state * price_state + num_quarterly * price_quarterly) = 4400 := by
  sorry

end NUMINAMATH_GPT_total_revenue_l2363_236370


namespace NUMINAMATH_GPT_hoseok_subtraction_result_l2363_236307

theorem hoseok_subtraction_result:
  ∃ x : ℤ, 15 * x = 45 ∧ x - 1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_hoseok_subtraction_result_l2363_236307


namespace NUMINAMATH_GPT_divisors_form_60k_l2363_236377

-- Define the conditions in Lean
def is_positive_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

def satisfies_conditions (n a b c : ℕ) : Prop :=
  is_positive_divisor n a ∧
  is_positive_divisor n b ∧
  is_positive_divisor n c ∧
  a > b ∧ b > c ∧
  is_positive_divisor n (a^2 - b^2) ∧
  is_positive_divisor n (b^2 - c^2) ∧
  is_positive_divisor n (a^2 - c^2)

-- State the theorem to be proven in Lean
theorem divisors_form_60k (n : ℕ) (a b c : ℕ) (h1 : satisfies_conditions n a b c) : 
  ∃ k : ℕ, n = 60 * k :=
sorry

end NUMINAMATH_GPT_divisors_form_60k_l2363_236377


namespace NUMINAMATH_GPT_incorrect_statement_is_A_l2363_236367

theorem incorrect_statement_is_A :
  (∀ (w h : ℝ), w * (2 * h) ≠ 3 * (w * h)) ∧
  (∀ (s : ℝ), (2 * s) ^ 2 = 4 * (s ^ 2)) ∧
  (∀ (s : ℝ), (2 * s) ^ 3 = 8 * (s ^ 3)) ∧
  (∀ (w h : ℝ), (w / 2) * (3 * h) = (3 / 2) * (w * h)) ∧
  (∀ (l w : ℝ), (2 * l) * (3 * w) = 6 * (l * w)) →
  ∃ (incorrect_statement : String), incorrect_statement = "A" := 
by 
  sorry

end NUMINAMATH_GPT_incorrect_statement_is_A_l2363_236367


namespace NUMINAMATH_GPT_angle_PDO_45_degrees_l2363_236310

-- Define the square configuration
variables (A B C D L P Q M N O : Type)
variables (a : ℝ) -- side length of the square ABCD

-- Conditions as hypothesized in the problem
def is_square (v₁ v₂ v₃ v₄ : Type) := true -- Placeholder for the square property
def on_diagonal_AC (L : Type) := true -- Placeholder for L being on diagonal AC
def common_vertex_L (sq1_v1 sq1_v2 sq1_v3 sq1_v4 sq2_v1 sq2_v2 sq2_v3 sq2_v4 : Type) := true -- Placeholder for common vertex L
def point_on_side (P AB_side: Type) := true -- Placeholder for P on side AB of ABCD
def square_center (center sq_v1 sq_v2 sq_v3 sq_v4 : Type) := true -- Placeholder for square's center

-- Prove the angle PDO is 45 degrees
theorem angle_PDO_45_degrees 
  (h₁ : is_square A B C D)
  (h₂ : on_diagonal_AC L)
  (h₃ : is_square A P L Q)
  (h₄ : is_square C M L N)
  (h₅ : common_vertex_L A P L Q C M L N)
  (h₆ : point_on_side P B)
  (h₇ : square_center O C M L N)
  : ∃ θ : ℝ, θ = 45 := 
  sorry

end NUMINAMATH_GPT_angle_PDO_45_degrees_l2363_236310


namespace NUMINAMATH_GPT_range_of_a_circle_C_intersects_circle_D_l2363_236336

/-- Definitions of circles C and D --/
def circle_C_eq (x y : ℝ) := (x - 1)^2 + (y - 1)^2 = 1
def circle_D_eq (x y m : ℝ) := x^2 + y^2 - 2 * m * x = 0

/-- Condition for the line intersecting Circle C --/
def line_intersects_circle_C (a : ℝ) := (∃ x y : ℝ, circle_C_eq x y ∧ (x + y = a))

/-- Proof of range for a --/
theorem range_of_a (a : ℝ) : line_intersects_circle_C a → (2 - Real.sqrt 2 ≤ a ∧ a ≤ 2 + Real.sqrt 2) :=
sorry

/-- Proposition for point A lying on circle C and satisfying the inequality --/
def point_A_on_circle_C_and_inequality (m : ℝ) (x y : ℝ) :=
  circle_C_eq x y ∧ x^2 + y^2 - (m + Real.sqrt 2 / 2) * x - (m + Real.sqrt 2 / 2) * y ≤ 0

/-- Proof that Circle C intersects Circle D --/
theorem circle_C_intersects_circle_D (m : ℝ) (a : ℝ) : 
  (∀ (x y : ℝ), point_A_on_circle_C_and_inequality m x y) →
  (1 ≤ m ∧
   ∃ (x y : ℝ), (circle_D_eq x y m ∧ (Real.sqrt ((m - 1)^2 + 1) < m + 1 ∧ Real.sqrt ((m - 1)^2 + 1) > m - 1))) :=
sorry

end NUMINAMATH_GPT_range_of_a_circle_C_intersects_circle_D_l2363_236336


namespace NUMINAMATH_GPT_pages_per_day_l2363_236381

-- Define the given conditions
def total_pages : ℕ := 957
def total_days : ℕ := 47

-- State the theorem based on the conditions and the required proof
theorem pages_per_day (p : ℕ) (d : ℕ) (h1 : p = total_pages) (h2 : d = total_days) :
  p / d = 20 := by
  sorry

end NUMINAMATH_GPT_pages_per_day_l2363_236381


namespace NUMINAMATH_GPT_max_elements_of_S_l2363_236339

-- Define the relation on set S and the conditions given
variable {S : Type} (R : S → S → Prop)

-- Lean translation of the conditions
def condition_1 (a b : S) : Prop :=
  (R a b ∨ R b a) ∧ ¬ (R a b ∧ R b a)

def condition_2 (a b c : S) : Prop :=
  R a b ∧ R b c → R c a

-- Define the problem statement:
theorem max_elements_of_S (h1 : ∀ a b : S, condition_1 R a b)
                          (h2 : ∀ a b c : S, condition_2 R a b c) :
  ∃ (n : ℕ), (∀ T : Finset S, T.card ≤ n) ∧ (∃ T : Finset S, T.card = 3) :=
sorry

end NUMINAMATH_GPT_max_elements_of_S_l2363_236339


namespace NUMINAMATH_GPT_total_cost_correct_l2363_236393

def cost_first_day : Nat := 4 + 5 + 3 + 2
def cost_second_day : Nat := 5 + 6 + 4
def total_cost : Nat := cost_first_day + cost_second_day

theorem total_cost_correct : total_cost = 29 := by
  sorry

end NUMINAMATH_GPT_total_cost_correct_l2363_236393


namespace NUMINAMATH_GPT_num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l2363_236383

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def sum_of_tens_and_units_is_ten (n : ℕ) : Prop :=
  (n / 10 % 10) + (n % 10) = 10

theorem num_even_three_digit_numbers_with_sum_of_tens_and_units_10 : 
  ∃! (N : ℕ), (N = 36) ∧ 
               (∀ n : ℕ, is_three_digit n → is_even n → sum_of_tens_and_units_is_ten n →
                         n = 36) := 
sorry

end NUMINAMATH_GPT_num_even_three_digit_numbers_with_sum_of_tens_and_units_10_l2363_236383


namespace NUMINAMATH_GPT_daily_sales_volume_80_sales_volume_function_price_for_profit_l2363_236351

-- Define all relevant conditions
def cost_price : ℝ := 70
def max_price : ℝ := 99
def initial_price : ℝ := 95
def initial_sales : ℕ := 50
def price_reduction_effect : ℕ := 2

-- Part 1: Proving daily sales volume at 80 yuan
theorem daily_sales_volume_80 : 
  (initial_price - 80) * price_reduction_effect + initial_sales = 80 := 
by sorry

-- Part 2: Proving functional relationship
theorem sales_volume_function (x : ℝ) (h₁ : 70 ≤ x) (h₂ : x ≤ 99) : 
  (initial_sales + price_reduction_effect * (initial_price - x) = -2 * x + 240) :=
by sorry

-- Part 3: Proving price for 1200 yuan daily profit
theorem price_for_profit (profit_target : ℝ) (h : profit_target = 1200) :
  ∃ x, (x - cost_price) * (initial_sales + price_reduction_effect * (initial_price - x)) = profit_target ∧ x ≤ max_price :=
by sorry

end NUMINAMATH_GPT_daily_sales_volume_80_sales_volume_function_price_for_profit_l2363_236351


namespace NUMINAMATH_GPT_rearrange_rooks_possible_l2363_236327

theorem rearrange_rooks_possible (board : Fin 8 × Fin 8 → Prop) (rooks : Fin 8 → Fin 8 × Fin 8) (painted : Fin 8 × Fin 8 → Prop) :
  (∀ i j : Fin 8, i ≠ j → (rooks i).1 ≠ (rooks j).1 ∧ (rooks i).2 ≠ (rooks j).2) → -- no two rooks are in the same row or column
  (∃ (unpainted_count : ℕ), (unpainted_count = 64 - 27)) → -- 27 squares are painted red
  (∃ new_rooks : Fin 8 → Fin 8 × Fin 8,
    (∀ i : Fin 8, ¬painted (new_rooks i)) ∧ -- all rooks are on unpainted squares
    (∀ i j : Fin 8, i ≠ j → (new_rooks i).1 ≠ (new_rooks j).1 ∧ (new_rooks i).2 ≠ (new_rooks j).2) ∧ -- no two rooks are in the same row or column
    (∃ i : Fin 8, rooks i ≠ new_rooks i)) -- at least one rook has moved
:=
sorry

end NUMINAMATH_GPT_rearrange_rooks_possible_l2363_236327


namespace NUMINAMATH_GPT_point_in_first_or_third_quadrant_l2363_236330

-- Definitions based on conditions
variables {x y : ℝ}

-- The proof statement
theorem point_in_first_or_third_quadrant (h : x * y > 0) : 
  (0 < x ∧ 0 < y) ∨ (x < 0 ∧ y < 0) :=
  sorry

end NUMINAMATH_GPT_point_in_first_or_third_quadrant_l2363_236330


namespace NUMINAMATH_GPT_basketball_games_l2363_236361

theorem basketball_games (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 3 * N + 4 * M = 88) : 3 * N = 48 :=
by sorry

end NUMINAMATH_GPT_basketball_games_l2363_236361


namespace NUMINAMATH_GPT_grade_point_average_one_third_l2363_236397

theorem grade_point_average_one_third :
  ∃ (x : ℝ), 55 = (1/3) * x + (2/3) * 60 ∧ x = 45 :=
by
  sorry

end NUMINAMATH_GPT_grade_point_average_one_third_l2363_236397


namespace NUMINAMATH_GPT_rectangle_area_ratio_is_three_l2363_236388

variables {a b : ℝ}

-- Rectangle ABCD with midpoint F on CD, BC = 3 * BE
def rectangle_midpoint_condition (CD_length : ℝ) (BC_length : ℝ) (BE_length : ℝ) (F_midpoint : Prop) :=
  F_midpoint ∧ BC_length = 3 * BE_length

-- Areas and the ratio
def area_rectangle (CD_length BC_length : ℝ) : ℝ :=
  CD_length * BC_length

def area_shaded (a b : ℝ) : ℝ :=
  2 * a * b

theorem rectangle_area_ratio_is_three (h : rectangle_midpoint_condition (2 * a) (3 * b) b (F_midpoint := True)) :
  area_rectangle (2 * a) (3 * b) = 3 * area_shaded a b :=
by
  unfold rectangle_midpoint_condition at h
  unfold area_rectangle area_shaded
  rw [←mul_assoc, ←mul_assoc]
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_is_three_l2363_236388


namespace NUMINAMATH_GPT_min_shift_for_even_function_l2363_236376

theorem min_shift_for_even_function :
  ∃ (m : ℝ), (m > 0) ∧ (∀ x : ℝ, (Real.sin (x + m) + Real.cos (x + m)) = (Real.sin (-x + m) + Real.cos (-x + m))) ∧ m = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_shift_for_even_function_l2363_236376


namespace NUMINAMATH_GPT_num_br_atoms_l2363_236318

theorem num_br_atoms (num_br : ℕ) : 
  (1 * 1 + num_br * 80 + 3 * 16 = 129) → num_br = 1 :=
  by
    intro h
    sorry

end NUMINAMATH_GPT_num_br_atoms_l2363_236318


namespace NUMINAMATH_GPT_neg_p_l2363_236338

-- Define the sets A and B
def is_odd (x : ℤ) : Prop := x % 2 = 1
def is_even (x : ℤ) : Prop := x % 2 = 0

-- Define the proposition p
def p : Prop := ∀ x : ℤ, is_odd x → is_even (2 * x)

-- State the negation of proposition p
theorem neg_p : ¬ p ↔ ∃ x : ℤ, is_odd x ∧ ¬ is_even (2 * x) := by sorry

end NUMINAMATH_GPT_neg_p_l2363_236338


namespace NUMINAMATH_GPT_manufacturing_department_percentage_l2363_236378

theorem manufacturing_department_percentage (total_degrees mfg_degrees : ℝ)
  (h1 : total_degrees = 360)
  (h2 : mfg_degrees = 162) : (mfg_degrees / total_degrees) * 100 = 45 :=
by 
  sorry

end NUMINAMATH_GPT_manufacturing_department_percentage_l2363_236378


namespace NUMINAMATH_GPT_base_6_conversion_l2363_236355

-- Define the conditions given in the problem
def base_6_to_10 (a b c : ℕ) : ℕ := a * 6^2 + b * 6^1 + c * 6^0

-- given that 524_6 = 2cd_10 and c, d are base-10 digits, prove that (c * d) / 12 = 3/4
theorem base_6_conversion (c d : ℕ) (h1 : base_6_to_10 5 2 4 = 196) (h2 : 2 * 10 * c + d = 196) :
  (c * d) / 12 = 3 / 4 :=
sorry

end NUMINAMATH_GPT_base_6_conversion_l2363_236355


namespace NUMINAMATH_GPT_Ponchik_week_day_l2363_236349

theorem Ponchik_week_day (n s : ℕ) (h1 : s = 20) (h2 : s * (4 * n + 1) = 1360) : n = 4 :=
by
  sorry

end NUMINAMATH_GPT_Ponchik_week_day_l2363_236349


namespace NUMINAMATH_GPT_remainder_approx_l2363_236312

def x : ℝ := 74.99999999999716 * 96
def y : ℝ := 74.99999999999716
def quotient : ℝ := 96
def expected_remainder : ℝ := 0.4096

theorem remainder_approx (x y : ℝ) (quotient : ℝ) (h1 : y = 74.99999999999716)
  (h2 : quotient = 96) (h3 : x = y * quotient) :
  x - y * quotient = expected_remainder :=
by
  sorry

end NUMINAMATH_GPT_remainder_approx_l2363_236312


namespace NUMINAMATH_GPT_coins_in_pockets_l2363_236375

theorem coins_in_pockets : (Nat.choose (5 + 3 - 1) (3 - 1)) = 21 := by
  sorry

end NUMINAMATH_GPT_coins_in_pockets_l2363_236375


namespace NUMINAMATH_GPT_Namjoon_walk_extra_l2363_236352

-- Define the usual distance Namjoon walks to school
def usual_distance := 1.2

-- Define the distance Namjoon walked to the intermediate point
def intermediate_distance := 0.3

-- Define the total distance Namjoon walked today
def total_distance_today := (intermediate_distance * 2) + usual_distance

-- Define the extra distance walked today compared to usual
def extra_distance := total_distance_today - usual_distance

-- State the theorem to prove that the extra distance walked today is 0.6 km
theorem Namjoon_walk_extra : extra_distance = 0.6 := 
by
  sorry

end NUMINAMATH_GPT_Namjoon_walk_extra_l2363_236352


namespace NUMINAMATH_GPT_largest_constant_c_l2363_236301

theorem largest_constant_c (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 1) : 
  x^6 + y^6 ≥ (1 / 2) * x * y :=
sorry

end NUMINAMATH_GPT_largest_constant_c_l2363_236301


namespace NUMINAMATH_GPT_product_of_digits_of_nondivisible_by_5_number_is_30_l2363_236360

-- Define the four-digit numbers
def numbers : List ℕ := [4825, 4835, 4845, 4855, 4865]

-- Define units and tens digit function
def units_digit (n : ℕ) := n % 10
def tens_digit (n : ℕ) := (n / 10) % 10

-- Assertion that 4865 is the number that is not divisible by 5
def not_divisible_by_5 (n : ℕ) : Prop := ¬ (units_digit n = 5 ∨ units_digit n = 0)

-- Lean 4 statement to prove the product of units and tens digit of the number not divisible by 5 is 30
theorem product_of_digits_of_nondivisible_by_5_number_is_30 :
  ∃ n ∈ numbers, not_divisible_by_5 n ∧ (units_digit n) * (tens_digit n) = 30 :=
by
  sorry

end NUMINAMATH_GPT_product_of_digits_of_nondivisible_by_5_number_is_30_l2363_236360


namespace NUMINAMATH_GPT_sum_coefficients_equals_l2363_236334

theorem sum_coefficients_equals :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ), 
  (∀ x : ℤ, (2 * x + 1) ^ 5 = 
    a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) →
  a_0 = 1 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 3^5 - 1 :=
by
  intros a_0 a_1 a_2 a_3 a_4 a_5 h h0
  sorry

end NUMINAMATH_GPT_sum_coefficients_equals_l2363_236334


namespace NUMINAMATH_GPT_cauliflower_area_l2363_236371

theorem cauliflower_area
  (s : ℕ) (a : ℕ) 
  (H1 : s * s / a = 40401)
  (H2 : s * s / a = 40000) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_cauliflower_area_l2363_236371


namespace NUMINAMATH_GPT_courier_cost_formula_l2363_236350

def cost (P : ℕ) : ℕ :=
if P = 0 then 0 else max 50 (30 + 7 * (P - 1))

theorem courier_cost_formula (P : ℕ) : cost P = 
  if P = 0 then 0 else max 50 (30 + 7 * (P - 1)) :=
by
  sorry

end NUMINAMATH_GPT_courier_cost_formula_l2363_236350


namespace NUMINAMATH_GPT_find_a_l2363_236373

theorem find_a (a : ℝ) :
  (∃x y : ℝ, x^2 + y^2 + 2 * x - 2 * y + a = 0 ∧ x + y + 4 = 0) →
  ∃c : ℝ, c = 2 ∧ a = -7 :=
by
  -- proof to be filled in
  sorry

end NUMINAMATH_GPT_find_a_l2363_236373


namespace NUMINAMATH_GPT_mobius_total_trip_time_l2363_236345

theorem mobius_total_trip_time :
  ∀ (d1 d2 v1 v2 : ℝ) (n r : ℕ),
  d1 = 143 → d2 = 143 → 
  v1 = 11 → v2 = 13 → 
  n = 4 → r = (30:ℝ)/60 →
  d1 / v1 + d2 / v2 + n * r = 26 :=
by
  intros d1 d2 v1 v2 n r h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num

end NUMINAMATH_GPT_mobius_total_trip_time_l2363_236345


namespace NUMINAMATH_GPT_min_value_expr_least_is_nine_l2363_236394

noncomputable def minimum_value_expression (a b c d : ℝ) : ℝ :=
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2

theorem min_value_expr_least_is_nine (a b c d : ℝ)
  (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  minimum_value_expression a b c d = 9 := 
sorry

end NUMINAMATH_GPT_min_value_expr_least_is_nine_l2363_236394


namespace NUMINAMATH_GPT_max_markers_with_20_dollars_l2363_236303

theorem max_markers_with_20_dollars (single_marker_cost : ℕ) (four_pack_cost : ℕ) (eight_pack_cost : ℕ) :
  single_marker_cost = 2 → four_pack_cost = 6 → eight_pack_cost = 10 → (∃ n, n = 16) := by
    intros h1 h2 h3
    existsi 16
    sorry

end NUMINAMATH_GPT_max_markers_with_20_dollars_l2363_236303


namespace NUMINAMATH_GPT_fourth_equation_l2363_236335

theorem fourth_equation :
  (5 * 6 * 7 * 8) = (2^4) * 1 * 3 * 5 * 7 :=
by
  sorry

end NUMINAMATH_GPT_fourth_equation_l2363_236335


namespace NUMINAMATH_GPT_overall_percent_change_in_stock_l2363_236322

noncomputable def stock_change (initial_value : ℝ) : ℝ :=
  let value_after_first_day := 0.85 * initial_value
  let value_after_second_day := 1.25 * value_after_first_day
  (value_after_second_day - initial_value) / initial_value * 100

theorem overall_percent_change_in_stock (x : ℝ) : stock_change x = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_overall_percent_change_in_stock_l2363_236322


namespace NUMINAMATH_GPT_production_today_l2363_236302

theorem production_today (n : ℕ) (P T : ℕ) 
  (h1 : n = 4) 
  (h2 : (P + T) / (n + 1) = 58) 
  (h3 : P = n * 50) : 
  T = 90 := 
by
  sorry

end NUMINAMATH_GPT_production_today_l2363_236302


namespace NUMINAMATH_GPT_largest_number_of_hcf_lcm_l2363_236358

theorem largest_number_of_hcf_lcm (a b c : ℕ) (h : Nat.gcd (Nat.gcd a b) c = 42)
  (factor1 : 10 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor2 : 20 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor3 : 25 ∣ Nat.lcm (Nat.lcm a b) c)
  (factor4 : 30 ∣ Nat.lcm (Nat.lcm a b) c) :
  max (max a b) c = 1260 := 
  sorry

end NUMINAMATH_GPT_largest_number_of_hcf_lcm_l2363_236358


namespace NUMINAMATH_GPT_cooper_savings_l2363_236391

theorem cooper_savings :
  let daily_savings := 34
  let days_in_year := 365
  daily_savings * days_in_year = 12410 :=
by
  sorry

end NUMINAMATH_GPT_cooper_savings_l2363_236391


namespace NUMINAMATH_GPT_min_value_a_l2363_236306

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ (Real.sqrt 2) / 2 → x^3 - 2 * x * Real.log x / Real.log a ≤ 0) ↔ a ≥ 1 / 4 := 
sorry

end NUMINAMATH_GPT_min_value_a_l2363_236306


namespace NUMINAMATH_GPT_contrapositive_of_proposition_l2363_236324

theorem contrapositive_of_proposition :
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_proposition_l2363_236324


namespace NUMINAMATH_GPT_minimum_bottles_needed_l2363_236357

theorem minimum_bottles_needed (fl_oz_needed : ℝ) (bottle_size_ml : ℝ) (fl_oz_per_liter : ℝ) (ml_per_liter : ℝ)
  (h1 : fl_oz_needed = 60)
  (h2 : bottle_size_ml = 250)
  (h3 : fl_oz_per_liter = 33.8)
  (h4 : ml_per_liter = 1000) :
  ∃ n : ℕ, n = 8 ∧ fl_oz_needed * ml_per_liter / fl_oz_per_liter / bottle_size_ml ≤ n :=
by
  sorry

end NUMINAMATH_GPT_minimum_bottles_needed_l2363_236357


namespace NUMINAMATH_GPT_x_squared_y_plus_xy_squared_l2363_236325

-- Define the variables and their conditions
variables {x y : ℝ}

-- Define the theorem stating that if xy = 3 and x + y = 5, then x^2y + xy^2 = 15
theorem x_squared_y_plus_xy_squared (h1 : x * y = 3) (h2 : x + y = 5) : x^2 * y + x * y^2 = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_x_squared_y_plus_xy_squared_l2363_236325


namespace NUMINAMATH_GPT_proof_problem_l2363_236359

/-- 
  Given:
  - r, j, z are Ryan's, Jason's, and Zachary's earnings respectively.
  - Zachary sold 40 games at $5 each.
  - Jason received 30% more money than Zachary.
  - The total amount of money received by all three is $770.
  Prove:
  - Ryan received $50 more than Jason.
--/
def problem_statement : Prop :=
  ∃ (r j z : ℕ), 
    z = 40 * 5 ∧
    j = z + z * 30 / 100 ∧
    r + j + z = 770 ∧ 
    r - j = 50

theorem proof_problem : problem_statement :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem_l2363_236359


namespace NUMINAMATH_GPT_number_of_men_at_picnic_l2363_236308

theorem number_of_men_at_picnic (total persons W M A C : ℕ) (h1 : total = 200) 
  (h2 : M = W + 20) (h3 : A = C + 20) (h4 : A = M + W) : M = 65 :=
by
  -- Proof can be filled in here
  sorry

end NUMINAMATH_GPT_number_of_men_at_picnic_l2363_236308


namespace NUMINAMATH_GPT_total_cost_after_discounts_l2363_236340

-- Definition of the cost function with applicable discounts
def pencil_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

def pen_cost (price: ℝ) (count: ℕ) (discount_threshold: ℕ) (discount_rate: ℝ) :=
  let initial_cost := count * price
  if count > discount_threshold then
    initial_cost - (initial_cost * discount_rate)
  else initial_cost

-- The statement to be proved
theorem total_cost_after_discounts :
  let pencil_price := 2.50
  let pen_price := 3.50
  let pencil_count := 38
  let pen_count := 56
  let pencil_discount_threshold := 30
  let pencil_discount_rate := 0.10
  let pen_discount_threshold := 50
  let pen_discount_rate := 0.15
  let total_cost := pencil_cost pencil_price pencil_count pencil_discount_threshold pencil_discount_rate
                   + pen_cost pen_price pen_count pen_discount_threshold pen_discount_rate
  total_cost = 252.10 := 
by 
  sorry

end NUMINAMATH_GPT_total_cost_after_discounts_l2363_236340


namespace NUMINAMATH_GPT_total_tiles_144_l2363_236326

-- Define the dimensions of the dining room
def diningRoomLength : ℕ := 15
def diningRoomWidth : ℕ := 20

-- Define the border width using 1x1 tiles
def borderWidth : ℕ := 2

-- Area of each 3x3 tile
def tileArea : ℕ := 9

-- Calculate the dimensions of the inner area after the border
def innerAreaLength : ℕ := diningRoomLength - 2 * borderWidth
def innerAreaWidth : ℕ := diningRoomWidth - 2 * borderWidth

-- Calculate the area of the inner region
def innerArea : ℕ := innerAreaLength * innerAreaWidth

-- Calculate the number of 3x3 tiles
def numThreeByThreeTiles : ℕ := (innerArea + tileArea - 1) / tileArea -- rounded up division

-- Calculate the number of 1x1 tiles for the border
def numOneByOneTiles : ℕ :=
  2 * (innerAreaLength + innerAreaWidth + 4 * borderWidth)

-- Total number of tiles
def totalTiles : ℕ := numOneByOneTiles + numThreeByThreeTiles

-- Prove that the total number of tiles is 144
theorem total_tiles_144 : totalTiles = 144 := by
  sorry

end NUMINAMATH_GPT_total_tiles_144_l2363_236326


namespace NUMINAMATH_GPT_perimeter_with_new_tiles_l2363_236396

theorem perimeter_with_new_tiles (p_original : ℕ) (num_original_tiles : ℕ) (num_new_tiles : ℕ)
  (h1 : p_original = 16)
  (h2 : num_original_tiles = 9)
  (h3 : num_new_tiles = 3) :
  ∃ p_new : ℕ, p_new = 17 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_with_new_tiles_l2363_236396


namespace NUMINAMATH_GPT_solve_quadratic_equation_l2363_236354

theorem solve_quadratic_equation (x : ℝ) (h : x^2 = x) : x = 0 ∨ x = 1 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l2363_236354


namespace NUMINAMATH_GPT_namjoon_rank_l2363_236313

theorem namjoon_rank (total_students : ℕ) (fewer_than_namjoon : ℕ) (rank_of_namjoon : ℕ) 
  (h1 : total_students = 13) (h2 : fewer_than_namjoon = 4) : rank_of_namjoon = 9 :=
sorry

end NUMINAMATH_GPT_namjoon_rank_l2363_236313


namespace NUMINAMATH_GPT_roses_in_centerpiece_l2363_236309

variable (r : ℕ)

theorem roses_in_centerpiece (h : 6 * 15 * (3 * r + 6) = 2700) : r = 8 := 
  sorry

end NUMINAMATH_GPT_roses_in_centerpiece_l2363_236309


namespace NUMINAMATH_GPT_inverse_of_f_at_2_l2363_236364

noncomputable def f (x : ℝ) : ℝ := x^2 - 1

theorem inverse_of_f_at_2 : ∀ x, x ≥ 0 → f x = 2 → x = Real.sqrt 3 :=
by
  intro x hx heq
  sorry

end NUMINAMATH_GPT_inverse_of_f_at_2_l2363_236364


namespace NUMINAMATH_GPT_find_the_number_l2363_236382

theorem find_the_number (x : ℝ) (h : 8 * x + 64 = 336) : x = 34 :=
by
  sorry

end NUMINAMATH_GPT_find_the_number_l2363_236382


namespace NUMINAMATH_GPT_circle_area_pi_div_2_l2363_236348

open Real EuclideanGeometry

variable (x y : ℝ)

def circleEquation : Prop := 3 * x^2 + 3 * y^2 - 15 * x + 9 * y + 27 = 0

theorem circle_area_pi_div_2
  (h : circleEquation x y) : 
  ∃ (r : ℝ), r = sqrt 0.5 ∧ π * r * r = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_area_pi_div_2_l2363_236348


namespace NUMINAMATH_GPT_permutation_6_4_l2363_236316

theorem permutation_6_4 : (Nat.factorial 6) / (Nat.factorial (6 - 4)) = 360 := by
  sorry

end NUMINAMATH_GPT_permutation_6_4_l2363_236316


namespace NUMINAMATH_GPT_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l2363_236387

variable {a b : ℝ}

theorem correct_calculation : a ^ 3 * a = a ^ 4 := 
by
  sorry

theorem incorrect_calculation_A : a ^ 3 + a ^ 3 ≠ 2 * a ^ 6 := 
by
  sorry

theorem incorrect_calculation_B : (a ^ 3) ^ 3 ≠ a ^ 6 :=
by
  sorry

theorem incorrect_calculation_D : (a - b) ^ 2 ≠ a ^ 2 - b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_incorrect_calculation_A_incorrect_calculation_B_incorrect_calculation_D_l2363_236387


namespace NUMINAMATH_GPT_scientific_notation_of_393000_l2363_236392

theorem scientific_notation_of_393000 : 
  ∃ (a : ℝ) (n : ℤ), a = 3.93 ∧ n = 5 ∧ (393000 = a * 10^n) := 
by
  use 3.93
  use 5
  sorry

end NUMINAMATH_GPT_scientific_notation_of_393000_l2363_236392


namespace NUMINAMATH_GPT_weight_of_steel_rod_l2363_236317

theorem weight_of_steel_rod (length1 : ℝ) (weight1 : ℝ) (length2 : ℝ) (weight2 : ℝ) 
  (h1 : length1 = 9) (h2 : weight1 = 34.2) (h3 : length2 = 11.25) : 
  weight2 = (weight1 / length1) * length2 :=
by
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end NUMINAMATH_GPT_weight_of_steel_rod_l2363_236317


namespace NUMINAMATH_GPT_no_real_k_for_distinct_roots_l2363_236369

theorem no_real_k_for_distinct_roots (k : ℝ) : ¬ ( -8 * k^2 > 0 ) := 
by
  sorry

end NUMINAMATH_GPT_no_real_k_for_distinct_roots_l2363_236369


namespace NUMINAMATH_GPT_good_apples_count_l2363_236353

def total_apples : ℕ := 14
def unripe_apples : ℕ := 6

theorem good_apples_count : total_apples - unripe_apples = 8 :=
by
  unfold total_apples unripe_apples
  sorry

end NUMINAMATH_GPT_good_apples_count_l2363_236353


namespace NUMINAMATH_GPT_problem_a_lt_c_lt_b_l2363_236362

noncomputable def a : ℝ := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : ℝ := Real.sqrt 6 / 2

theorem problem_a_lt_c_lt_b : a < c ∧ c < b := 
by {
  sorry
}

end NUMINAMATH_GPT_problem_a_lt_c_lt_b_l2363_236362


namespace NUMINAMATH_GPT_probability_two_red_books_l2363_236344

theorem probability_two_red_books (total_books red_books blue_books selected_books : ℕ)
  (h_total: total_books = 8)
  (h_red: red_books = 4)
  (h_blue: blue_books = 4)
  (h_selected: selected_books = 2) :
  (Nat.choose red_books selected_books : ℚ) / (Nat.choose total_books selected_books) = 3 / 14 := by
  sorry

end NUMINAMATH_GPT_probability_two_red_books_l2363_236344


namespace NUMINAMATH_GPT_solve_stamps_l2363_236368

noncomputable def stamps_problem : Prop :=
  ∃ (A B C D : ℝ), 
    A + B + C + D = 251 ∧
    A = 2 * B + 2 ∧
    A = 3 * C + 6 ∧
    A = 4 * D - 16 ∧
    D = 32

theorem solve_stamps : stamps_problem :=
sorry

end NUMINAMATH_GPT_solve_stamps_l2363_236368


namespace NUMINAMATH_GPT_full_batches_needed_l2363_236365

def students : Nat := 150
def cookies_per_student : Nat := 3
def cookies_per_batch : Nat := 20
def attendance_rate : Rat := 0.70

theorem full_batches_needed : 
  let attendees := (students : Rat) * attendance_rate
  let total_cookies_needed := attendees * (cookies_per_student : Rat)
  let batches_needed := total_cookies_needed / (cookies_per_batch : Rat)
  batches_needed.ceil = 16 :=
by
  sorry

end NUMINAMATH_GPT_full_batches_needed_l2363_236365


namespace NUMINAMATH_GPT_roots_sum_product_l2363_236300

variable {a b : ℝ}

theorem roots_sum_product (ha : a + b = 6) (hp : a * b = 8) : 
  a^4 + b^4 + a^3 * b + a * b^3 = 432 :=
by
  sorry

end NUMINAMATH_GPT_roots_sum_product_l2363_236300


namespace NUMINAMATH_GPT_mowed_times_in_spring_l2363_236323

-- Definition of the problem conditions
def total_mowed_times : ℕ := 11
def summer_mowed_times : ℕ := 5

-- The theorem to prove
theorem mowed_times_in_spring : (total_mowed_times - summer_mowed_times = 6) :=
by
  sorry

end NUMINAMATH_GPT_mowed_times_in_spring_l2363_236323


namespace NUMINAMATH_GPT_valentine_problem_l2363_236329

def initial_valentines : ℕ := 30
def given_valentines : ℕ := 8
def remaining_valentines : ℕ := 22

theorem valentine_problem : initial_valentines - given_valentines = remaining_valentines := by
  sorry

end NUMINAMATH_GPT_valentine_problem_l2363_236329


namespace NUMINAMATH_GPT_max_value_of_f_l2363_236398

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_value_of_f : ∀ x : ℝ, f x ≤ Real.sin 1 + 1 ∧ (f 0 = Real.sin 1 + 1) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2363_236398


namespace NUMINAMATH_GPT_vectors_parallel_opposite_directions_l2363_236385

theorem vectors_parallel_opposite_directions
  (a b : ℝ × ℝ)
  (h₁ : a = (-1, 2))
  (h₂ : b = (2, -4)) :
  b = (-2 : ℝ) • a ∧ b = -2 • a :=
by
  sorry

end NUMINAMATH_GPT_vectors_parallel_opposite_directions_l2363_236385


namespace NUMINAMATH_GPT_multiply_binomials_l2363_236314

theorem multiply_binomials (x : ℝ) : (4 * x + 3) * (2 * x - 7) = 8 * x^2 - 22 * x - 21 :=
by 
  -- Proof is to be filled here
  sorry

end NUMINAMATH_GPT_multiply_binomials_l2363_236314


namespace NUMINAMATH_GPT_slope_parallel_line_l2363_236372

theorem slope_parallel_line (x y : ℝ) (a b c : ℝ) (h : 3 * x - 6 * y = 15) : 
  ∃ m : ℝ, m = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_slope_parallel_line_l2363_236372


namespace NUMINAMATH_GPT_days_at_grandparents_l2363_236346

theorem days_at_grandparents
  (total_vacation_days : ℕ)
  (travel_to_gp : ℕ)
  (travel_to_brother : ℕ)
  (days_at_brother : ℕ)
  (travel_to_sister : ℕ)
  (days_at_sister : ℕ)
  (travel_home : ℕ)
  (total_days : total_vacation_days = 21) :
  total_vacation_days - (travel_to_gp + travel_to_brother + days_at_brother + travel_to_sister + days_at_sister + travel_home) = 5 :=
by
  sorry -- proof to be constructed

end NUMINAMATH_GPT_days_at_grandparents_l2363_236346


namespace NUMINAMATH_GPT_peanut_price_is_correct_l2363_236379

noncomputable def price_per_pound_of_peanuts : ℝ := 
  let total_weight := 100
  let mixed_price_per_pound := 2.5
  let cashew_weight := 60
  let cashew_price_per_pound := 4
  let peanut_weight := total_weight - cashew_weight
  let total_revenue := total_weight * mixed_price_per_pound
  let cashew_cost := cashew_weight * cashew_price_per_pound
  let peanut_cost := total_revenue - cashew_cost
  peanut_cost / peanut_weight

theorem peanut_price_is_correct :
  price_per_pound_of_peanuts = 0.25 := 
by sorry

end NUMINAMATH_GPT_peanut_price_is_correct_l2363_236379


namespace NUMINAMATH_GPT_function_domain_l2363_236363

noncomputable def domain_function (x : ℝ) : Prop :=
  x > 0 ∧ (Real.log x / Real.log 2)^2 - 1 > 0

theorem function_domain :
  { x : ℝ | domain_function x } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

end NUMINAMATH_GPT_function_domain_l2363_236363


namespace NUMINAMATH_GPT_ways_to_select_at_least_one_defective_l2363_236337

open Finset

-- Define basic combinatorial selection functions
def combination (n k : ℕ) := Nat.choose n k

-- Given conditions
def total_products : ℕ := 100
def defective_products : ℕ := 6
def selected_products : ℕ := 3
def non_defective_products : ℕ := total_products - defective_products

-- The question to prove: the number of ways to select at least one defective product
theorem ways_to_select_at_least_one_defective :
  (combination total_products selected_products) - (combination non_defective_products selected_products) =
  (combination 100 3) - (combination 94 3) := by
  sorry

end NUMINAMATH_GPT_ways_to_select_at_least_one_defective_l2363_236337


namespace NUMINAMATH_GPT_remainder_492381_div_6_l2363_236356

theorem remainder_492381_div_6 : 492381 % 6 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_492381_div_6_l2363_236356


namespace NUMINAMATH_GPT_annie_start_crayons_l2363_236321

def start_crayons (end_crayons : ℕ) (added_crayons : ℕ) : ℕ := end_crayons - added_crayons

theorem annie_start_crayons (added_crayons end_crayons : ℕ) (h1 : added_crayons = 36) (h2 : end_crayons = 40) :
  start_crayons end_crayons added_crayons = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_eq_of_eq_add sorry  -- skips the detailed proof

end NUMINAMATH_GPT_annie_start_crayons_l2363_236321


namespace NUMINAMATH_GPT_range_of_a_l2363_236386

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2363_236386


namespace NUMINAMATH_GPT_divisible_digit_B_l2363_236384

-- Define the digit type as natural numbers within the range 0 to 9.
def digit := {n : ℕ // n <= 9}

-- Define what it means for a number to be even.
def even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Define what it means for a number to be divisible by 3.
def divisible_by_3 (n : ℕ) : Prop := ∃ k, n = 3 * k

-- Define our problem in Lean as properties of the digit B.
theorem divisible_digit_B (B : digit) (h_even : even B.1) (h_div_by_3 : divisible_by_3 (14 + B.1)) : B.1 = 4 :=
sorry

end NUMINAMATH_GPT_divisible_digit_B_l2363_236384


namespace NUMINAMATH_GPT_jed_change_l2363_236347

theorem jed_change :
  ∀ (num_games : ℕ) (cost_per_game : ℕ) (payment : ℕ) (bill_value : ℕ),
  num_games = 6 →
  cost_per_game = 15 →
  payment = 100 →
  bill_value = 5 →
  (payment - num_games * cost_per_game) / bill_value = 2 :=
by
  intros num_games cost_per_game payment bill_value
  sorry

end NUMINAMATH_GPT_jed_change_l2363_236347


namespace NUMINAMATH_GPT_triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l2363_236304

-- Define the properties and variables of the given obtuse triangle
variables (a b c : ℝ) (A C : ℝ)
-- Given conditions
axiom ha : a = 7
axiom hb : b = 3
axiom hcosC : Real.cos C = 11 / 14

-- Prove the values of c and angle A
theorem triangle_ABC_c_and_A_value (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : c = 5 ∧ A = 2 * Real.pi / 3 :=
sorry

-- Prove the value of sin(2C - π / 6)
theorem sin_2C_minus_pi_6 (ha : a = 7) (hb : b = 3) (hcosC : Real.cos C = 11 / 14) : Real.sin (2 * C - Real.pi / 6) = 71 / 98 :=
sorry

end NUMINAMATH_GPT_triangle_ABC_c_and_A_value_sin_2C_minus_pi_6_l2363_236304


namespace NUMINAMATH_GPT_customer_paid_correct_amount_l2363_236332

noncomputable def cost_price : ℝ := 5565.217391304348
noncomputable def markup_percentage : ℝ := 0.15
noncomputable def markup_amount (cost : ℝ) : ℝ := cost * markup_percentage
noncomputable def final_price (cost : ℝ) (markup : ℝ) : ℝ := cost + markup

theorem customer_paid_correct_amount :
  final_price cost_price (markup_amount cost_price) = 6400 := sorry

end NUMINAMATH_GPT_customer_paid_correct_amount_l2363_236332


namespace NUMINAMATH_GPT_number_of_children_l2363_236366

def cost_of_adult_ticket := 19
def cost_of_child_ticket := cost_of_adult_ticket - 6
def number_of_adults := 2
def total_cost := 77

theorem number_of_children : 
  ∃ (x : ℕ), cost_of_child_ticket * x + cost_of_adult_ticket * number_of_adults = total_cost ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l2363_236366


namespace NUMINAMATH_GPT_numerator_in_second_fraction_l2363_236331

theorem numerator_in_second_fraction (p q x: ℚ) (h1 : p / q = 4 / 5) (h2 : 11 / 7 + x / (2 * q + p) = 2) : x = 6 :=
sorry

end NUMINAMATH_GPT_numerator_in_second_fraction_l2363_236331


namespace NUMINAMATH_GPT_find_larger_number_l2363_236374

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l2363_236374


namespace NUMINAMATH_GPT_gcd_polynomial_l2363_236389

theorem gcd_polynomial (b : ℤ) (h : 1820 ∣ b) : Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
sorry

end NUMINAMATH_GPT_gcd_polynomial_l2363_236389


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l2363_236380

open Real

theorem eccentricity_of_ellipse 
  (O B F : ℝ × ℝ)
  (a b : ℝ) 
  (h_a_gt_b: a > b)
  (h_b_gt_0: b > 0)
  (ellipse_eq : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h_OB_eq_OF : dist O B = dist O F)
  (O_is_origin : O = (0,0))
  (B_is_upper_vertex : B = (0, b))
  (F_is_right_focus : F = (c, 0) ∧ c = Real.sqrt (a^2 - b^2)) :
 (c / a = sqrt 2 / 2)
:=
sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l2363_236380


namespace NUMINAMATH_GPT_quadrilateral_area_inequality_l2363_236311

theorem quadrilateral_area_inequality (a b c d : ℝ) :
  ∃ (S_ABCD : ℝ), S_ABCD ≤ (1 / 4) * (a + c) ^ 2 + b * d :=
sorry

end NUMINAMATH_GPT_quadrilateral_area_inequality_l2363_236311


namespace NUMINAMATH_GPT_infinite_series_sum_l2363_236399

theorem infinite_series_sum (a r : ℝ) (h₀ : -1 < r) (h₁ : r < 1) :
    (∑' n, if (n % 2 = 0) then a * r^(n/2) else a^2 * r^((n+1)/2)) = (a * (1 + a * r))/(1 - r^2) :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_sum_l2363_236399


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2363_236343

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x > 1) : x < 1 := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2363_236343


namespace NUMINAMATH_GPT_number_of_students_increased_l2363_236305

theorem number_of_students_increased
  (original_number_of_students : ℕ) (increase_in_expenses : ℕ) (diminshed_average_expenditure : ℕ)
  (original_expenditure : ℕ) (increase_in_students : ℕ) :
  original_number_of_students = 35 →
  increase_in_expenses = 42 →
  diminshed_average_expenditure = 1 →
  original_expenditure = 420 →
  (35 + increase_in_students) * (12 - 1) - 420 = 42 →
  increase_in_students = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_students_increased_l2363_236305


namespace NUMINAMATH_GPT_problem_statement_l2363_236390

noncomputable def solveProblem : ℝ :=
  let a := 2
  let b := -3
  let c := 1
  a + b + c

-- The theorem statement to ensure a + b + c equals 0
theorem problem_statement : solveProblem = 0 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2363_236390


namespace NUMINAMATH_GPT_diagonal_of_square_l2363_236341

theorem diagonal_of_square (length_rect width_rect : ℝ) (h1 : length_rect = 45) (h2 : width_rect = 40)
  (area_rect : ℝ) (h3 : area_rect = length_rect * width_rect) (area_square : ℝ) (h4 : area_square = area_rect)
  (side_square : ℝ) (h5 : side_square^2 = area_square) (diagonal_square : ℝ) (h6 : diagonal_square = side_square * Real.sqrt 2) :
  diagonal_square = 60 := by
  sorry

end NUMINAMATH_GPT_diagonal_of_square_l2363_236341


namespace NUMINAMATH_GPT_part1_part2_l2363_236333

def f (x : ℝ) (t : ℝ) : ℝ := x^2 + 2 * t * x + t - 1

theorem part1 (hf : ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3) : 
  ∀ x ∈ Set.Icc (-(3 : ℝ)) (1 : ℝ), f x 2 ≤ 6 ∧ f x 2 ≥ -3 :=
by 
  sorry
  
theorem part2 (ht : ∀ x ∈ Set.Icc (1 : ℝ) (2 : ℝ), f x t > 0) : 
  t ∈ Set.Ioi (0 : ℝ) :=
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l2363_236333
