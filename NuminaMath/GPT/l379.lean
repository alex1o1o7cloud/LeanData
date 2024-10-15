import Mathlib

namespace NUMINAMATH_GPT_bell_rings_count_l379_37996

-- Defining the conditions
def bell_rings_per_class : ℕ := 2
def total_classes_before_music : ℕ := 4
def bell_rings_during_music_start : ℕ := 1

-- The main proof statement
def total_bell_rings : ℕ :=
  total_classes_before_music * bell_rings_per_class + bell_rings_during_music_start

theorem bell_rings_count : total_bell_rings = 9 := by
  sorry

end NUMINAMATH_GPT_bell_rings_count_l379_37996


namespace NUMINAMATH_GPT_no_solutions_l379_37902

theorem no_solutions (x : ℝ) (hx : x ≠ 0): ¬ (12 * Real.sin x + 5 * Real.cos x = 13 + 1 / |x|) := 
by 
  sorry

end NUMINAMATH_GPT_no_solutions_l379_37902


namespace NUMINAMATH_GPT_number_of_books_Ryan_l379_37941

structure LibraryProblem :=
  (Total_pages_Ryan : ℕ)
  (Total_days : ℕ)
  (Pages_per_book_brother : ℕ)
  (Extra_pages_Ryan : ℕ)

def calculate_books_received (p : LibraryProblem) : ℕ :=
  let Total_pages_brother := p.Pages_per_book_brother * p.Total_days
  let Ryan_daily_average := (Total_pages_brother / p.Total_days) + p.Extra_pages_Ryan
  p.Total_pages_Ryan / Ryan_daily_average

theorem number_of_books_Ryan (p : LibraryProblem) (h1 : p.Total_pages_Ryan = 2100)
  (h2 : p.Total_days = 7) (h3 : p.Pages_per_book_brother = 200) (h4 : p.Extra_pages_Ryan = 100) :
  calculate_books_received p = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_books_Ryan_l379_37941


namespace NUMINAMATH_GPT_no_3_digit_numbers_sum_27_even_l379_37916

-- Define the conditions
def is_digit_sum_27 (n : ℕ) : Prop :=
  (n ≥ 100 ∧ n < 1000) ∧ ((n / 100) + (n / 10 % 10) + (n % 10) = 27)

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

-- Define the theorem
theorem no_3_digit_numbers_sum_27_even :
  ¬ ∃ n : ℕ, is_digit_sum_27 n ∧ is_even n :=
by
  sorry

end NUMINAMATH_GPT_no_3_digit_numbers_sum_27_even_l379_37916


namespace NUMINAMATH_GPT_total_cost_of_replacing_floor_l379_37950

-- Dimensions of the first rectangular section
def length1 : ℕ := 8
def width1 : ℕ := 7

-- Dimensions of the second rectangular section
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Cost to remove the old flooring
def cost_removal : ℕ := 50

-- Cost of new flooring per square foot
def cost_per_sqft : ℝ := 1.25

-- Total cost to replace the floor in both sections of the L-shaped room
theorem total_cost_of_replacing_floor 
  (A1 : ℕ := length1 * width1)
  (A2 : ℕ := length2 * width2)
  (total_area : ℕ := A1 + A2)
  (cost_flooring : ℝ := total_area * cost_per_sqft)
  : cost_removal + cost_flooring = 150 :=
sorry

end NUMINAMATH_GPT_total_cost_of_replacing_floor_l379_37950


namespace NUMINAMATH_GPT_number_of_two_digit_primes_with_digit_sum_12_l379_37951

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def is_prime (n : ℕ) : Prop :=
  Nat.Prime n

theorem number_of_two_digit_primes_with_digit_sum_12 : 
  ∃! n, is_two_digit n ∧ is_prime n ∧ sum_of_digits n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_two_digit_primes_with_digit_sum_12_l379_37951


namespace NUMINAMATH_GPT_sum_of_squares_of_rates_equals_536_l379_37968

-- Define the biking, jogging, and swimming rates as integers.
variables (b j s : ℤ)

-- Condition: Ed's total distance equation.
def ed_distance_eq : Prop := 3 * b + 2 * j + 4 * s = 80

-- Condition: Sue's total distance equation.
def sue_distance_eq : Prop := 4 * b + 3 * j + 2 * s = 98

-- The main statement to prove.
theorem sum_of_squares_of_rates_equals_536 (hb : b ≥ 0) (hj : j ≥ 0) (hs : s ≥ 0) 
  (h1 : ed_distance_eq b j s) (h2 : sue_distance_eq b j s) :
  b^2 + j^2 + s^2 = 536 :=
by sorry

end NUMINAMATH_GPT_sum_of_squares_of_rates_equals_536_l379_37968


namespace NUMINAMATH_GPT_part_i_part_ii_l379_37922

noncomputable def f (x a : ℝ) := |x - a|

theorem part_i :
  (∀ (x : ℝ), (f x 1) ≥ (|x + 1| + 1) ↔ x ≤ -0.5) :=
sorry

theorem part_ii :
  (∀ (x a : ℝ), (f x a) + 3 * x ≤ 0 → { x | x ≤ -1 } ⊆ { x | (f x a) + 3 * x ≤ 0 }) →
  (∀ (a : ℝ), (0 ≤ a ∧ a ≤ 2) ∨ (-4 ≤ a ∧ a < 0)) :=
sorry

end NUMINAMATH_GPT_part_i_part_ii_l379_37922


namespace NUMINAMATH_GPT_total_games_l379_37977

theorem total_games (N : ℕ) (p : ℕ)
  (hPetya : 2 ∣ N)
  (hKolya : 3 ∣ N)
  (hVasya : 5 ∣ N)
  (hGamesNotInvolving : 2 ≤ N - (N / 2 + N / 3 + N / 5)) :
  N = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_games_l379_37977


namespace NUMINAMATH_GPT_monotone_increasing_function_range_l379_37926

theorem monotone_increasing_function_range (a : ℝ) :
  (∀ x ∈ Set.Ioo (1 / 2 : ℝ) (3 : ℝ), (1 / x + 2 * a * x - 3) ≥ 0) ↔ a ≥ 9 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_monotone_increasing_function_range_l379_37926


namespace NUMINAMATH_GPT_solve_quadratic_eq_l379_37983

theorem solve_quadratic_eq (x : ℝ) : (x^2 - 2*x + 1 = 9) → (x = 4 ∨ x = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l379_37983


namespace NUMINAMATH_GPT_remainder_zero_l379_37944

theorem remainder_zero :
  ∀ (a b c d : ℕ),
  a % 53 = 47 →
  b % 53 = 4 →
  c % 53 = 10 →
  d % 53 = 14 →
  (((a * b * c) % 53) * d) % 47 = 0 := 
by 
  intros a b c d h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_remainder_zero_l379_37944


namespace NUMINAMATH_GPT_problem_statement_l379_37952

open Real

variables {f : ℝ → ℝ} {a b c : ℝ}

-- f is twice differentiable on ℝ
axiom hf : ∀ x : ℝ, Differentiable ℝ f
axiom hf' : ∀ x : ℝ, Differentiable ℝ (deriv f)

-- ∃ c ∈ ℝ, such that (f(b) - f(a)) / (b - a) ≠ f'(c) for all a ≠ b
axiom hc : ∃ c : ℝ, ∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c

-- Prove that f''(c) = 0
theorem problem_statement : ∃ c : ℝ, (∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c) → deriv (deriv f) c = 0 := sorry

end NUMINAMATH_GPT_problem_statement_l379_37952


namespace NUMINAMATH_GPT_best_fitting_model_l379_37966

-- Define the \(R^2\) values for each model
def R2_Model1 : ℝ := 0.75
def R2_Model2 : ℝ := 0.90
def R2_Model3 : ℝ := 0.25
def R2_Model4 : ℝ := 0.55

-- State that Model 2 is the best fitting model
theorem best_fitting_model : R2_Model2 = max (max R2_Model1 R2_Model2) (max R2_Model3 R2_Model4) :=
by -- Proof skipped
  sorry

end NUMINAMATH_GPT_best_fitting_model_l379_37966


namespace NUMINAMATH_GPT_initial_population_l379_37947

theorem initial_population (P : ℝ) (h1 : 1.20 * P = P_1) (h2 : 0.96 * P = P_2) (h3 : P_2 = 9600) : P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l379_37947


namespace NUMINAMATH_GPT_investment_months_l379_37965

theorem investment_months (i_a i_b i_c a_gain total_gain : ℝ) (m : ℝ) :
  i_a = 1 ∧ i_b = 2 * i_a ∧ i_c = 3 * i_a ∧ a_gain = 6100 ∧ total_gain = 18300 ∧ m * i_b * (12 - m) + i_c * 3 * 4 = 12200 →
  a_gain / total_gain = i_a * 12 / (i_a * 12 + i_b * (12 - m) + i_c * 4) → m = 6 :=
by
  intros h1 h2
  obtain ⟨ha, hb, hc, hag, htg, h⟩ := h1
  -- proof omitted
  sorry

end NUMINAMATH_GPT_investment_months_l379_37965


namespace NUMINAMATH_GPT_contradiction_assumption_l379_37911

theorem contradiction_assumption (a b c : ℕ) :
  (∃ k : ℕ, (k = a ∨ k = b ∨ k = c) ∧ ∃ n : ℕ, k = 2 * n + 1) →
  (∃ k1 k2 : ℕ, (k1 = a ∨ k1 = b ∨ k1 = c) ∧ (k2 = a ∨ k2 = b ∨ k2 = c) ∧ k1 ≠ k2 ∧ ∃ n1 n2 : ℕ, k1 = 2 * n1 ∧ k2 = 2 * n2) ∨
  (∀ k : ℕ, (k = a ∨ k = b ∨ k = c) → ∃ n : ℕ, k = 2 * n + 1) :=
sorry

end NUMINAMATH_GPT_contradiction_assumption_l379_37911


namespace NUMINAMATH_GPT_range_of_m_l379_37999

noncomputable def range_m (a b : ℝ) (m : ℝ) : Prop :=
  (3 * a + 4 / b = 1) ∧ a > 0 ∧ b > 0 → (1 / a + 3 * b > m)

theorem range_of_m (m : ℝ) : (∀ a b : ℝ, (range_m a b m)) ↔ m < 27 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l379_37999


namespace NUMINAMATH_GPT_employee_payment_l379_37914

theorem employee_payment 
    (total_pay : ℕ)
    (pay_A : ℕ)
    (pay_B : ℕ)
    (h1 : total_pay = 560)
    (h2 : pay_A = 3 * pay_B / 2)
    (h3 : pay_A + pay_B = total_pay) :
    pay_B = 224 :=
sorry

end NUMINAMATH_GPT_employee_payment_l379_37914


namespace NUMINAMATH_GPT_tiling_polygons_l379_37908

theorem tiling_polygons (n : ℕ) (h1 : 2 < n) (h2 : ∃ x : ℕ, x * (((n - 2) * 180 : ℝ) / n) = 360) :
  n = 3 ∨ n = 4 ∨ n = 6 := 
by
  sorry

end NUMINAMATH_GPT_tiling_polygons_l379_37908


namespace NUMINAMATH_GPT_ott_fraction_part_l379_37987

noncomputable def fractional_part_of_group_money (x : ℝ) (M L N P : ℝ) :=
  let total_initial := M + L + N + P + 2
  let money_received_by_ott := 4 * x
  let ott_final_money := 2 + money_received_by_ott
  let total_final := total_initial + money_received_by_ott
  (ott_final_money / total_final) = (3 / 14)

theorem ott_fraction_part (x : ℝ) (M L N P : ℝ)
    (hM : M = 6 * x) (hL : L = 5 * x) (hN : N = 4 * x) (hP : P = 7 * x) :
    fractional_part_of_group_money x M L N P :=
by
  sorry

end NUMINAMATH_GPT_ott_fraction_part_l379_37987


namespace NUMINAMATH_GPT_parts_supplier_total_amount_received_l379_37929

noncomputable def total_amount_received (total_packages: ℕ) (price_per_package: ℚ) (discount_factor: ℚ)
  (X_percentage: ℚ) (Y_percentage: ℚ) : ℚ :=
  let X_packages := X_percentage * total_packages
  let Y_packages := Y_percentage * total_packages
  let Z_packages := total_packages - X_packages - Y_packages
  let discounted_price := discount_factor * price_per_package
  let cost_X := X_packages * price_per_package
  let cost_Y := Y_packages * price_per_package
  let cost_Z := 10 * price_per_package + (Z_packages - 10) * discounted_price
  cost_X + cost_Y + cost_Z

-- Given conditions
def total_packages : ℕ := 60
def price_per_package : ℚ := 20
def discount_factor : ℚ := 4 / 5
def X_percentage : ℚ := 0.20
def Y_percentage : ℚ := 0.15

theorem parts_supplier_total_amount_received :
  total_amount_received total_packages price_per_package discount_factor X_percentage Y_percentage = 1084 := 
by 
  -- Here we need the proof, but we put sorry to skip it as per instructions
  sorry

end NUMINAMATH_GPT_parts_supplier_total_amount_received_l379_37929


namespace NUMINAMATH_GPT_range_of_a_l379_37940

theorem range_of_a (x y a : ℝ): 
  (x + 3 * y = 3 - a) ∧ (2 * x + y = 1 + 3 * a) ∧ (x + y > 3 * a + 4) ↔ (a < -3 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l379_37940


namespace NUMINAMATH_GPT_positions_after_196_moves_l379_37912

def cat_position (n : ℕ) : ℕ :=
  n % 4

def mouse_position (n : ℕ) : ℕ :=
  n % 8

def cat_final_position : ℕ := 0 -- top left based on the reverse order cycle
def mouse_final_position : ℕ := 3 -- bottom middle based on the reverse order cycle

theorem positions_after_196_moves :
  cat_position 196 = cat_final_position ∧ mouse_position 196 = mouse_final_position :=
by
  sorry

end NUMINAMATH_GPT_positions_after_196_moves_l379_37912


namespace NUMINAMATH_GPT_sum_of_exponents_l379_37919

def power_sum_2021 (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ r → (a k = 1 ∨ a k = -1)) ∧
  (a 1 * 3 ^ n 1 + a 2 * 3 ^ n 2 + a 3 * 3 ^ n 3 + a 4 * 3 ^ n 4 + a 5 * 3 ^ n 5 + a 6 * 3 ^ n 6 = 2021) ∧
  (n 1 = 7 ∧ n 2 = 5 ∧ n 3 = 4 ∧ n 4 = 2 ∧ n 5 = 1 ∧ n 6 = 0) ∧
  (a 1 = 1 ∧ a 2 = -1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 1 ∧ a 6 = -1)

theorem sum_of_exponents : ∃ (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ), power_sum_2021 a n r ∧ (n 1 + n 2 + n 3 + n 4 + n 5 + n 6 = 19) :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_exponents_l379_37919


namespace NUMINAMATH_GPT_max_seq_value_l379_37936

def is_arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + a m

variables (a : ℕ → ℤ)
variables (S : ℕ → ℤ)

axiom distinct_terms (h : is_arithmetic_seq a) : ∀ n m, n ≠ m → a n ≠ a m
axiom condition_1 : ∀ n, a (2 * n) = 2 * a n - 3
axiom condition_2 : a 6 * a 6 = a 1 * a 21
axiom sum_of_first_n_terms : ∀ n, S n = n * (n + 4)

noncomputable def seq (n : ℕ) : ℤ := S n / 2^(n - 1)

theorem max_seq_value : 
  (∀ n, seq n >= seq (n - 1) ∧ seq n >= seq (n + 1)) → 
  (∃ n, seq n = 6) :=
sorry

end NUMINAMATH_GPT_max_seq_value_l379_37936


namespace NUMINAMATH_GPT_isosceles_triangle_base_l379_37943

variable (a b : ℕ)

theorem isosceles_triangle_base 
  (h_isosceles : a = 7 ∧ b = 3)
  (triangle_inequality : 7 + 7 > 3) : b = 3 := by
-- Begin of the proof
sorry
-- End of the proof

end NUMINAMATH_GPT_isosceles_triangle_base_l379_37943


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_identity_l379_37939

theorem arithmetic_geometric_mean_identity (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 96) : x^2 + y^2 = 1408 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_identity_l379_37939


namespace NUMINAMATH_GPT_total_marbles_l379_37935

-- Define the number of marbles Mary and Joan have respectively
def mary_marbles := 9
def joan_marbles := 3

-- Prove that the total number of marbles is 12
theorem total_marbles : mary_marbles + joan_marbles = 12 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l379_37935


namespace NUMINAMATH_GPT_fraction_condition_l379_37963

theorem fraction_condition (x : ℚ) :
  (3 + 2 * x) / (4 + 3 * x) = 5 / 9 ↔ x = -7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_condition_l379_37963


namespace NUMINAMATH_GPT_truck_capacity_cost_function_minimum_cost_l379_37956

theorem truck_capacity :
  ∃ (m n : ℕ),
    3 * m + 4 * n = 27 ∧ 
    4 * m + 5 * n = 35 ∧
    m = 5 ∧ 
    n = 3 :=
by {
  sorry
}

theorem cost_function (a : ℕ) (h : a ≤ 5) :
  ∃ (w : ℕ),
    w = 50 * a + 2250 :=
by {
  sorry
}

theorem minimum_cost :
  ∃ (w : ℕ),
    w = 2250 ∧ 
    ∀ (a : ℕ), a ≤ 5 → (50 * a + 2250) ≥ 2250 :=
by {
  sorry
}

end NUMINAMATH_GPT_truck_capacity_cost_function_minimum_cost_l379_37956


namespace NUMINAMATH_GPT_row_time_14_24_l379_37960

variable (d c s r : ℝ)

-- Assumptions
def swim_with_current (d c s : ℝ) := s + c = d / 40
def swim_against_current (d c s : ℝ) := s - c = d / 45
def row_against_current (d c r : ℝ) := r - c = d / 15

-- Expected result
def time_to_row_harvard_mit (d c r : ℝ) := d / (r + c) = 14 + 24 / 60

theorem row_time_14_24 :
  swim_with_current d c s ∧
  swim_against_current d c s ∧
  row_against_current d c r →
  time_to_row_harvard_mit d c r :=
by
  sorry

end NUMINAMATH_GPT_row_time_14_24_l379_37960


namespace NUMINAMATH_GPT_complex_identity_l379_37997

theorem complex_identity (a b : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 - 2 * i) * i = a + b * i) : a * b = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_identity_l379_37997


namespace NUMINAMATH_GPT_luke_good_games_l379_37962

-- Definitions
def bought_from_friend : ℕ := 2
def bought_from_garage_sale : ℕ := 2
def defective_games : ℕ := 2

-- The theorem we want to prove
theorem luke_good_games :
  bought_from_friend + bought_from_garage_sale - defective_games = 2 := 
by 
  sorry

end NUMINAMATH_GPT_luke_good_games_l379_37962


namespace NUMINAMATH_GPT_find_missing_number_l379_37955

theorem find_missing_number (x : ℕ) (h : 10111 - 10 * 2 * x = 10011) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_missing_number_l379_37955


namespace NUMINAMATH_GPT_evaluate_expression_l379_37959

theorem evaluate_expression : 
  3 * (-3)^4 + 3 * (-3)^3 + 3 * (-3)^2 + 3 * 3^2 + 3 * 3^3 + 3 * 3^4 = 540 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l379_37959


namespace NUMINAMATH_GPT_circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l379_37924

noncomputable def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 1)^2 = 10

theorem circle_through_points (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (h_center : ∃ C: ℝ × ℝ, C.1 - C.2 - 4 = 0 ∧ (circle_eq C.1 C.2)) : ∀ x y, circle_eq x y ↔ (x - 3) ^ 2 + (y + 1) ^ 2 = 10 := 
by sorry

theorem line_perpendicular_and_tangent_to_circle (line_slope : ℝ) (tangent : ∀ x y, circle_eq x y → (x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0)) : ∀ x, x + 3*y + 10 = 0 ∨ x + 3*y - 10 = 0 :=
by sorry

theorem max_area_triangle (A B P : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (6, 0)) (hP : circle_eq P.1 P.2) : ∃ area : ℝ, area = 5 + 5 * Real.sqrt 2
:= 
by sorry

end NUMINAMATH_GPT_circle_through_points_line_perpendicular_and_tangent_to_circle_max_area_triangle_l379_37924


namespace NUMINAMATH_GPT_product_of_x_y_l379_37976

-- Assume the given conditions
variables (EF GH FG HE : ℝ)
variables (x y : ℝ)
variable (EFGH : Type)

-- Conditions given
axiom h1 : EF = 58
axiom h2 : GH = 3 * x + 1
axiom h3 : FG = 2 * y^2
axiom h4 : HE = 36
-- It is given that EFGH forms a parallelogram
axiom h5 : EF = GH
axiom h6 : FG = HE

-- The product of x and y is determined by the conditions
theorem product_of_x_y : x * y = 57 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_product_of_x_y_l379_37976


namespace NUMINAMATH_GPT_num_four_digit_snappy_numbers_divisible_by_25_l379_37971

def is_snappy (n : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 = d4 ∧ d2 = d3

def is_divisible_by_25 (n : ℕ) : Prop :=
  let last_two_digits := n % 100
  last_two_digits = 0 ∨ last_two_digits = 25 ∨ last_two_digits = 50 ∨ last_two_digits = 75

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem num_four_digit_snappy_numbers_divisible_by_25 : 
  ∃ n, n = 3 ∧ (∀ x, is_four_digit x ∧ is_snappy x ∧ is_divisible_by_25 x ↔ x = 5225 ∨ x = 0550 ∨ x = 5775)
:=
sorry

end NUMINAMATH_GPT_num_four_digit_snappy_numbers_divisible_by_25_l379_37971


namespace NUMINAMATH_GPT_number_of_crosswalks_per_intersection_l379_37949

theorem number_of_crosswalks_per_intersection 
  (num_intersections : Nat) 
  (total_lines : Nat) 
  (lines_per_crosswalk : Nat) 
  (h1 : num_intersections = 5) 
  (h2 : total_lines = 400) 
  (h3 : lines_per_crosswalk = 20) :
  (total_lines / lines_per_crosswalk) / num_intersections = 4 :=
by
  -- Proof steps can be inserted here
  sorry

end NUMINAMATH_GPT_number_of_crosswalks_per_intersection_l379_37949


namespace NUMINAMATH_GPT_candle_cost_correct_l379_37967

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end NUMINAMATH_GPT_candle_cost_correct_l379_37967


namespace NUMINAMATH_GPT_annual_interest_rate_equivalent_l379_37973

noncomputable def quarterly_compound_rate : ℝ := 1 + 0.02
noncomputable def annual_compound_amount : ℝ := quarterly_compound_rate ^ 4

theorem annual_interest_rate_equivalent : 
  (annual_compound_amount - 1) * 100 = 8.24 := 
by
  sorry

end NUMINAMATH_GPT_annual_interest_rate_equivalent_l379_37973


namespace NUMINAMATH_GPT_odd_function_symmetry_l379_37930

def f (x : ℝ) : ℝ := x^3 + x

-- Prove that f(-x) = -f(x)
theorem odd_function_symmetry : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end NUMINAMATH_GPT_odd_function_symmetry_l379_37930


namespace NUMINAMATH_GPT_laura_has_435_dollars_l379_37907

-- Define the monetary values and relationships
def darwin_money := 45
def mia_money := 2 * darwin_money + 20
def combined_money := mia_money + darwin_money
def laura_money := 3 * combined_money - 30

-- The theorem to prove: Laura's money is $435
theorem laura_has_435_dollars : laura_money = 435 := by
  sorry

end NUMINAMATH_GPT_laura_has_435_dollars_l379_37907


namespace NUMINAMATH_GPT_error_in_area_l379_37992

theorem error_in_area (s : ℝ) (h : s > 0) :
  let s_measured := 1.02 * s
  let A_actual := s^2
  let A_measured := s_measured^2
  let error := (A_measured - A_actual) / A_actual * 100
  error = 4.04 := by
  sorry

end NUMINAMATH_GPT_error_in_area_l379_37992


namespace NUMINAMATH_GPT_sum_of_coordinates_of_D_l379_37934

theorem sum_of_coordinates_of_D (P C D : ℝ × ℝ)
  (hP : P = (4, 9))
  (hC : C = (10, 5))
  (h_mid : P = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) :
  D.1 + D.2 = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_D_l379_37934


namespace NUMINAMATH_GPT_sum_of_100th_group_is_1010100_l379_37974

theorem sum_of_100th_group_is_1010100 : (100 + 100^2 + 100^3) = 1010100 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_100th_group_is_1010100_l379_37974


namespace NUMINAMATH_GPT_N_even_for_all_permutations_l379_37932

noncomputable def N (a b : Fin 2013 → ℕ) : ℕ :=
  Finset.prod (Finset.univ : Finset (Fin 2013)) (λ i => a i - b i)

theorem N_even_for_all_permutations {a : Fin 2013 → ℕ}
  (h_distinct : Function.Injective a) :
  ∀ b : Fin 2013 → ℕ,
  (∀ i, b i ∈ Finset.univ.image a) →
  ∃ n, n = N a b ∧ Even n :=
by
  -- This is where the proof would go, using the given conditions.
  sorry

end NUMINAMATH_GPT_N_even_for_all_permutations_l379_37932


namespace NUMINAMATH_GPT_cos_value_of_angle_l379_37990

theorem cos_value_of_angle (α : ℝ) (h : Real.sin (α + Real.pi / 6) = 1 / 3) :
  Real.cos (2 * α - 2 * Real.pi / 3) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cos_value_of_angle_l379_37990


namespace NUMINAMATH_GPT_find_m_l379_37942

theorem find_m (m : ℝ) : (m - 2) * (0 : ℝ)^2 + 4 * (0 : ℝ) + 2 - |m| = 0 → m = -2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_m_l379_37942


namespace NUMINAMATH_GPT_functions_not_necessarily_equal_l379_37904

-- Define the domain and range
variables {α β : Type*}

-- Define two functions f and g with the same domain and range
variables (f g : α → β)

-- Lean statement for the given mathematical problem
theorem functions_not_necessarily_equal (h_domain : ∀ x : α, (∃ x : α, true))
  (h_range : ∀ y : β, (∃ y : β, true)) : ¬(f = g) :=
sorry

end NUMINAMATH_GPT_functions_not_necessarily_equal_l379_37904


namespace NUMINAMATH_GPT_smallest_possible_stamps_l379_37981

theorem smallest_possible_stamps (M : ℕ) : 
  ((M % 5 = 2) ∧ (M % 7 = 2) ∧ (M % 9 = 2) ∧ (M > 2)) → M = 317 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_possible_stamps_l379_37981


namespace NUMINAMATH_GPT_first_point_x_coord_l379_37975

variables (m n : ℝ)

theorem first_point_x_coord (h1 : m = 2 * n + 5) (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m = 2 * n + 5 :=
by 
  sorry

end NUMINAMATH_GPT_first_point_x_coord_l379_37975


namespace NUMINAMATH_GPT_pencils_needed_l379_37970

theorem pencils_needed (pencilsA : ℕ) (pencilsB : ℕ) (classroomsA : ℕ) (classroomsB : ℕ) (total_shortage : ℕ)
  (hA : pencilsA = 480)
  (hB : pencilsB = 735)
  (hClassA : classroomsA = 6)
  (hClassB : classroomsB = 9)
  (hShortage : total_shortage = 85) 
  : 90 = 6 + 5 * ((total_shortage / (classroomsA + classroomsB)) + 1) * classroomsB :=
by {
  sorry
}

end NUMINAMATH_GPT_pencils_needed_l379_37970


namespace NUMINAMATH_GPT_min_value_PA_d_l379_37923

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_value_PA_d :
  let A : ℝ × ℝ := (3, 4)
  let parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
  let distance_to_line (P : ℝ × ℝ) (line_x : ℝ) : ℝ := abs (P.1 - line_x)
  let d : ℝ := distance_to_line P (-1)
  ∀ P : ℝ × ℝ, parabola P → (distance P A + d) ≥ 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_min_value_PA_d_l379_37923


namespace NUMINAMATH_GPT_triangle_area_l379_37928

noncomputable def area_of_triangle (a b c α β γ : ℝ) :=
  (1 / 2) * a * b * Real.sin γ

theorem triangle_area 
  (a b c A B C : ℝ)
  (h1 : b * Real.cos C = 3 * a * Real.cos B - c * Real.cos B)
  (h2 : (a * b * Real.cos C) / (a * b) = 2) :
  area_of_triangle a b c A B C = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l379_37928


namespace NUMINAMATH_GPT_choose_officers_ways_l379_37918

theorem choose_officers_ways :
  let members := 12
  let vp_candidates := 4
  let remaining_after_president := members - 1
  let remaining_after_vice_president := remaining_after_president - 1
  let remaining_after_secretary := remaining_after_vice_president - 1
  let remaining_after_treasurer := remaining_after_secretary - 1
  (members * vp_candidates * (remaining_after_vice_president) *
   (remaining_after_secretary) * (remaining_after_treasurer)) = 34560 := by
  -- Calculation here
  sorry

end NUMINAMATH_GPT_choose_officers_ways_l379_37918


namespace NUMINAMATH_GPT_both_selected_probability_l379_37991

-- Define the probabilities of selection for X and Y
def P_X := 1 / 7
def P_Y := 2 / 9

-- Statement to prove that the probability of both being selected is 2 / 63
theorem both_selected_probability :
  (P_X * P_Y) = (2 / 63) :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_both_selected_probability_l379_37991


namespace NUMINAMATH_GPT_train_speed_l379_37921

theorem train_speed (distance time : ℝ) (h₀ : distance = 180) (h₁ : time = 9) : 
  ((distance / 1000) / (time / 3600)) = 72 :=
by 
  -- below statement will bring the remainder of the setup and will be proved without the steps
  sorry

end NUMINAMATH_GPT_train_speed_l379_37921


namespace NUMINAMATH_GPT_empty_set_negation_l379_37986

open Set

theorem empty_set_negation (α : Type) : ¬ (∀ s : Set α, ∅ ⊆ s) ↔ (∃ s : Set α, ¬(∅ ⊆ s)) :=
by
  sorry

end NUMINAMATH_GPT_empty_set_negation_l379_37986


namespace NUMINAMATH_GPT_factor_expression_l379_37910

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l379_37910


namespace NUMINAMATH_GPT_mono_increasing_intervals_l379_37953

noncomputable def f : ℝ → ℝ :=
by sorry

theorem mono_increasing_intervals (f : ℝ → ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_sym : ∀ x, f x = f (-2 - x))
  (h_decr1 : ∀ x y, -2 ≤ x ∧ x < y ∧ y ≤ -1 → f y ≤ f x) :
  (∀ x y, 1 ≤ x ∧ x < y ∧ y ≤ 2 → f x ≤ f y) ∧
  (∀ x y, 3 ≤ x ∧ x < y ∧ y ≤ 4 → f x ≤ f y) :=
sorry

end NUMINAMATH_GPT_mono_increasing_intervals_l379_37953


namespace NUMINAMATH_GPT_amount_spent_on_candy_l379_37917

-- Define the given conditions
def amount_from_mother := 80
def amount_from_father := 40
def amount_from_uncle := 70
def final_amount := 140 

-- Define the initial amount
def initial_amount := amount_from_mother + amount_from_father 

-- Prove the amount spent on candy
theorem amount_spent_on_candy : 
  initial_amount - (final_amount - amount_from_uncle) = 50 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_amount_spent_on_candy_l379_37917


namespace NUMINAMATH_GPT_number_of_boys_l379_37906

noncomputable def numGirls : Nat := 46
noncomputable def numGroups : Nat := 8
noncomputable def groupSize : Nat := 9
noncomputable def totalMembers : Nat := numGroups * groupSize
noncomputable def numBoys : Nat := totalMembers - numGirls

theorem number_of_boys :
  numBoys = 26 := by
  sorry

end NUMINAMATH_GPT_number_of_boys_l379_37906


namespace NUMINAMATH_GPT_decagon_perimeter_l379_37957

theorem decagon_perimeter (num_sides : ℕ) (side_length : ℝ) (h_num_sides : num_sides = 10) (h_side_length : side_length = 3) : 
  (num_sides * side_length = 30) :=
by
  sorry

end NUMINAMATH_GPT_decagon_perimeter_l379_37957


namespace NUMINAMATH_GPT_run_faster_l379_37927

theorem run_faster (v_B k : ℝ) (h1 : ∀ (t : ℝ), 96 / (k * v_B) = t → 24 / v_B = t) : k = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_run_faster_l379_37927


namespace NUMINAMATH_GPT_compute_y_series_l379_37938

theorem compute_y_series :
  (∑' n : ℕ, (1 / 3) ^ n) + (∑' n : ℕ, ((-1) ^ n) / (4 ^ n)) = ∑' n : ℕ, (1 / (23 / 13) ^ n) :=
by
  sorry

end NUMINAMATH_GPT_compute_y_series_l379_37938


namespace NUMINAMATH_GPT_geometric_sequence_product_l379_37933

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a 1 * (a 2 / a 1) ^ n)
  (h1 : a 1 * a 4 = -3) : a 2 * a 3 = -3 :=
by
  -- sorry is placed here to indicate the proof is not provided.
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l379_37933


namespace NUMINAMATH_GPT_part_I_part_II_l379_37995

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log x) / x

theorem part_I (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  ∃ x ∈ Set.Ioo m (m + 1), ∀ y ∈ Set.Ioo m (m + 1), f y ≤ f x := sorry

theorem part_II (x : ℝ) (h : 1 < x) :
  (x + 1) * (x + Real.exp (-x)) * f x > 2 * (1 + 1 / Real.exp 1) := sorry

end NUMINAMATH_GPT_part_I_part_II_l379_37995


namespace NUMINAMATH_GPT_correct_operation_l379_37913

theorem correct_operation (a b : ℝ) : (a * b) - 2 * (a * b) = - (a * b) :=
sorry

end NUMINAMATH_GPT_correct_operation_l379_37913


namespace NUMINAMATH_GPT_find_divisor_l379_37985

-- Definitions based on the conditions
def is_divisor (d : ℕ) (a b k : ℕ) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ (b - a) / n = k ∧ k = d

-- Problem statement
theorem find_divisor (a b k : ℕ) (H : b = 43 ∧ a = 10 ∧ k = 11) : ∃ d, d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l379_37985


namespace NUMINAMATH_GPT_floor_sqrt_sum_eq_floor_sqrt_expr_l379_37903

-- Proof problem definition
theorem floor_sqrt_sum_eq_floor_sqrt_expr (n : ℕ) : 
  (Int.floor (Real.sqrt n + Real.sqrt (n + 1))) = (Int.floor (Real.sqrt (4 * n + 2))) := 
sorry

end NUMINAMATH_GPT_floor_sqrt_sum_eq_floor_sqrt_expr_l379_37903


namespace NUMINAMATH_GPT_least_value_xy_l379_37988

theorem least_value_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(3*y) = 1/9) : x*y = 108 :=
sorry

end NUMINAMATH_GPT_least_value_xy_l379_37988


namespace NUMINAMATH_GPT_middle_number_consecutive_even_l379_37905

theorem middle_number_consecutive_even (a b c : ℤ) 
  (h1 : a = b - 2) 
  (h2 : c = b + 2) 
  (h3 : a + b = 18) 
  (h4 : a + c = 22) 
  (h5 : b + c = 28) : 
  b = 11 :=
by sorry

end NUMINAMATH_GPT_middle_number_consecutive_even_l379_37905


namespace NUMINAMATH_GPT_smallest_c_inv_l379_37980

def f (x : ℝ) : ℝ := (x + 3)^2 - 7

theorem smallest_c_inv (c : ℝ) : (∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) →
  c = -3 :=
sorry

end NUMINAMATH_GPT_smallest_c_inv_l379_37980


namespace NUMINAMATH_GPT_roots_of_polynomial_inequality_l379_37961

theorem roots_of_polynomial_inequality :
  (∃ (p q r s : ℂ), (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) ∧
  (p * q * r * s = 3) ∧ (p*q + p*r + p*s + q*r + q*s + r*s = 11)) →
  (1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = 11/3) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_inequality_l379_37961


namespace NUMINAMATH_GPT_ratio_a_to_d_l379_37946

theorem ratio_a_to_d (a b c d : ℕ) 
  (h1 : a * 4 = b * 3) 
  (h2 : b * 9 = c * 7) 
  (h3 : c * 7 = d * 5) : 
  a * 3 = d := 
sorry

end NUMINAMATH_GPT_ratio_a_to_d_l379_37946


namespace NUMINAMATH_GPT_quarterly_business_tax_cost_l379_37900

theorem quarterly_business_tax_cost
    (price_federal : ℕ := 50)
    (price_state : ℕ := 30)
    (Q : ℕ)
    (num_federal : ℕ := 60)
    (num_state : ℕ := 20)
    (num_quart_business : ℕ := 10)
    (total_revenue : ℕ := 4400)
    (revenue_equation : num_federal * price_federal + num_state * price_state + num_quart_business * Q = total_revenue) :
    Q = 80 :=
by 
  sorry

end NUMINAMATH_GPT_quarterly_business_tax_cost_l379_37900


namespace NUMINAMATH_GPT_octagon_area_l379_37994

noncomputable def regular_octagon_area_inscribed_circle_radius3 : ℝ :=
  18 * Real.sqrt 2

theorem octagon_area
  (r : ℝ)
  (h : r = 3)
  (octagon_inscribed : ∀ (x : ℝ), x = r * 3 * Real.sin (π / 8)): 
  regular_octagon_area_inscribed_circle_radius3 = 18 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_octagon_area_l379_37994


namespace NUMINAMATH_GPT_range_of_x_l379_37998

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  1 - x > 0

theorem range_of_x (x : ℝ) : range_of_independent_variable x → x < 1 :=
by sorry

end NUMINAMATH_GPT_range_of_x_l379_37998


namespace NUMINAMATH_GPT_intersection_A_B_l379_37937

def A : Set ℝ := {x | 1 < x}
def B : Set ℝ := {y | y ≤ 2}
def expected_intersection : Set ℝ := {z | 1 < z ∧ z ≤ 2}

theorem intersection_A_B : (A ∩ B) = expected_intersection :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_intersection_A_B_l379_37937


namespace NUMINAMATH_GPT_range_m_single_solution_l379_37920

-- Statement expressing the conditions and conclusion.
theorem range_m_single_solution :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → x^3 - 3 * x + m = 0 → ∃! x, 0 ≤ x ∧ x ≤ 2) ↔ m ∈ (Set.Ico (-2 : ℝ) 0) ∪ {2} := 
sorry

end NUMINAMATH_GPT_range_m_single_solution_l379_37920


namespace NUMINAMATH_GPT_solve_inequality_system_simplify_expression_l379_37909

-- Part 1: System of Inequalities

theorem solve_inequality_system : 
  ∀ (x : ℝ), (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x → 1 ≤ x ∧ x < 3 :=  by
  sorry

-- Part 2: Expression Simplification

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) : 
  (m - 1 / m) * ((m^2 - m) / (m^2 - 2 * m + 1)) = m + 1 :=
  by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_simplify_expression_l379_37909


namespace NUMINAMATH_GPT_calculate_a3_b3_l379_37989

theorem calculate_a3_b3 (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by
  sorry

end NUMINAMATH_GPT_calculate_a3_b3_l379_37989


namespace NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_l379_37948

theorem surface_area_of_circumscribed_sphere (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : 
  ∃ S : ℝ, S = 29 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_l379_37948


namespace NUMINAMATH_GPT_modulo_power_l379_37915

theorem modulo_power (a n : ℕ) (p : ℕ) (hn_pos : 0 < n) (hp_odd : p % 2 = 1)
  (hp_prime : Nat.Prime p) (h : a^p ≡ 1 [MOD p^n]) : a ≡ 1 [MOD p^(n-1)] :=
by
  sorry

end NUMINAMATH_GPT_modulo_power_l379_37915


namespace NUMINAMATH_GPT_minimum_value_of_expression_l379_37972

theorem minimum_value_of_expression :
  ∀ x y : ℝ, x^2 - x * y + y^2 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l379_37972


namespace NUMINAMATH_GPT_chess_club_girls_l379_37969

theorem chess_club_girls (B G : ℕ) (h1 : B + G = 32) (h2 : (1 / 2 : ℝ) * G + B = 20) : G = 24 :=
by
  -- proof
  sorry

end NUMINAMATH_GPT_chess_club_girls_l379_37969


namespace NUMINAMATH_GPT_intersection_A_B_l379_37978

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 8}
def B := {x : ℝ | x^2 - 3 * x - 4 < 0}
def expected := {x : ℝ | 2 ≤ x ∧ x < 4 }

theorem intersection_A_B : (A ∩ B) = expected := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l379_37978


namespace NUMINAMATH_GPT_find_constants_l379_37982

theorem find_constants (A B C : ℝ) (hA : A = 7) (hB : B = -9) (hC : C = 5) :
  (∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 → 
    ( -2 * x ^ 2 + 5 * x - 7) / (x ^ 3 - x) = A / x + (B * x + C) / (x ^ 2 - 1) ) :=
by
  intros x hx
  rw [hA, hB, hC]
  sorry

end NUMINAMATH_GPT_find_constants_l379_37982


namespace NUMINAMATH_GPT_find_c_value_l379_37925

variable {x: ℝ}

theorem find_c_value (d e c : ℝ) (h₁ : 6 * d = 18) (h₂ : -15 + 6 * e = -5)
(h₃ : (10 / 3) * c = 15) :
  c = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_find_c_value_l379_37925


namespace NUMINAMATH_GPT_xy_value_l379_37901

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := 
by 
  sorry

end NUMINAMATH_GPT_xy_value_l379_37901


namespace NUMINAMATH_GPT_calc_one_calc_two_calc_three_l379_37954

theorem calc_one : (54 + 38) * 15 = 1380 := by
  sorry

theorem calc_two : 1500 - 32 * 45 = 60 := by
  sorry

theorem calc_three : 157 * (70 / 35) = 314 := by
  sorry

end NUMINAMATH_GPT_calc_one_calc_two_calc_three_l379_37954


namespace NUMINAMATH_GPT_common_divisors_greatest_l379_37993

theorem common_divisors_greatest (n : ℕ) (h₁ : ∀ d, d ∣ 120 ∧ d ∣ n ↔ d = 1 ∨ d = 3 ∨ d = 9) : 9 = Nat.gcd 120 n := by
  sorry

end NUMINAMATH_GPT_common_divisors_greatest_l379_37993


namespace NUMINAMATH_GPT_sum_of_all_numbers_after_n_steps_l379_37958

def initial_sum : ℕ := 2

def sum_after_step (n : ℕ) : ℕ :=
  2 * 3^n

theorem sum_of_all_numbers_after_n_steps (n : ℕ) : 
  sum_after_step n = 2 * 3^n :=
by sorry

end NUMINAMATH_GPT_sum_of_all_numbers_after_n_steps_l379_37958


namespace NUMINAMATH_GPT_blue_hat_cost_l379_37945

variable (B : ℕ)
variable (totalHats : ℕ := 85)
variable (greenHatCost : ℕ := 7)
variable (greenHatsBought : ℕ := 38)
variable (totalCost : ℕ := 548)

theorem blue_hat_cost 
(h1 : greenHatsBought = 38) 
(h2 : totalHats = 85) 
(h3 : greenHatCost = 7)
(h4 : totalCost = 548) :
  let totalGreenHatCost := greenHatCost * greenHatsBought
  let totalBlueHatCost := totalCost - totalGreenHatCost
  let totalBlueHatsBought := totalHats - greenHatsBought
  B = totalBlueHatCost / totalBlueHatsBought := by
  sorry

end NUMINAMATH_GPT_blue_hat_cost_l379_37945


namespace NUMINAMATH_GPT_simplify_expression_l379_37931

variable (a b : ℝ)

theorem simplify_expression : -3 * a * (2 * a - 4 * b + 2) + 6 * a = -6 * a ^ 2 + 12 * a * b := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l379_37931


namespace NUMINAMATH_GPT_no_pairs_exist_l379_37964

theorem no_pairs_exist (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) : (1/a + 1/b = 2/(a+b)) → False :=
by
  sorry

end NUMINAMATH_GPT_no_pairs_exist_l379_37964


namespace NUMINAMATH_GPT_show_revenue_l379_37984

variable (tickets_first_show : Nat) (tickets_cost : Nat) (multiplicator : Nat)
variable (tickets_second_show : Nat := multiplicator * tickets_first_show)
variable (total_tickets : Nat := tickets_first_show + tickets_second_show)
variable (total_revenue : Nat := total_tickets * tickets_cost)

theorem show_revenue :
    tickets_first_show = 200 ∧ tickets_cost = 25 ∧ multiplicator = 3 →
    total_revenue = 20000 := 
by
    intros h
    sorry

end NUMINAMATH_GPT_show_revenue_l379_37984


namespace NUMINAMATH_GPT_fraction_exponent_evaluation_l379_37979

theorem fraction_exponent_evaluation : 
  (3 ^ 10 + 3 ^ 8) / (3 ^ 10 - 3 ^ 8) = 5 / 4 :=
by sorry

end NUMINAMATH_GPT_fraction_exponent_evaluation_l379_37979
