import Mathlib

namespace NUMINAMATH_GPT_row_number_sum_l1609_160934

theorem row_number_sum (n : ℕ) (h : (2 * n - 1) ^ 2 = 2015 ^ 2) : n = 1008 :=
by
  sorry

end NUMINAMATH_GPT_row_number_sum_l1609_160934


namespace NUMINAMATH_GPT_photograph_perimeter_is_23_l1609_160923

noncomputable def photograph_perimeter (w h m : ℝ) : ℝ :=
if (w + 4) * (h + 4) = m ∧ (w + 8) * (h + 8) = m + 94 then 2 * (w + h) else 0

theorem photograph_perimeter_is_23 (w h m : ℝ) 
    (h₁ : (w + 4) * (h + 4) = m) 
    (h₂ : (w + 8) * (h + 8) = m + 94) : 
    photograph_perimeter w h m = 23 := 
by 
  sorry

end NUMINAMATH_GPT_photograph_perimeter_is_23_l1609_160923


namespace NUMINAMATH_GPT_evaluate_polynomial_at_two_l1609_160972

def f (x : ℝ) : ℝ := x^5 + 2 * x^3 + 3 * x^2 + x + 1

theorem evaluate_polynomial_at_two : f 2 = 41 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_two_l1609_160972


namespace NUMINAMATH_GPT_ratio_of_green_to_blue_l1609_160955

-- Definitions of the areas and the circles
noncomputable def red_area : ℝ := Real.pi * (1 : ℝ) ^ 2
noncomputable def middle_area : ℝ := Real.pi * (2 : ℝ) ^ 2
noncomputable def large_area: ℝ := Real.pi * (3 : ℝ) ^ 2

noncomputable def blue_area : ℝ := middle_area - red_area
noncomputable def green_area : ℝ := large_area - middle_area

-- The proof that the ratio of the green area to the blue area is 5/3
theorem ratio_of_green_to_blue : green_area / blue_area = 5 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_green_to_blue_l1609_160955


namespace NUMINAMATH_GPT_harmonyNumbersWithFirstDigit2_l1609_160918

def isHarmonyNumber (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  digits.sum = 6

def startsWithDigit (d n : ℕ) : Prop :=
  n / 1000 = d

theorem harmonyNumbersWithFirstDigit2 :
  ∃ c : ℕ, c = 15 ∧ ∀ n : ℕ, (1000 ≤ n ∧ n < 10000) → isHarmonyNumber n → startsWithDigit 2 n → ∃ m : ℕ, m < c ∧ m = n :=
sorry

end NUMINAMATH_GPT_harmonyNumbersWithFirstDigit2_l1609_160918


namespace NUMINAMATH_GPT_employed_males_percentage_l1609_160932

theorem employed_males_percentage (p_employed : ℝ) (p_employed_females : ℝ) : 
  (64 / 100) * (1 - 21.875 / 100) * 100 = 49.96 :=
by
  sorry

end NUMINAMATH_GPT_employed_males_percentage_l1609_160932


namespace NUMINAMATH_GPT_red_socks_l1609_160927

variable {R : ℕ}

theorem red_socks (h1 : 2 * R + R + 6 * R = 90) : R = 10 := 
by
  sorry

end NUMINAMATH_GPT_red_socks_l1609_160927


namespace NUMINAMATH_GPT_hypotenuse_right_triangle_l1609_160980

theorem hypotenuse_right_triangle (a b : ℕ) (h1 : a = 15) (h2 : b = 36) :
  ∃ c, c ^ 2 = a ^ 2 + b ^ 2 ∧ c = 39 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_right_triangle_l1609_160980


namespace NUMINAMATH_GPT_james_out_of_pocket_l1609_160994

-- Definitions based on conditions
def old_car_value : ℝ := 20000
def old_car_sold_for : ℝ := 0.80 * old_car_value
def new_car_sticker_price : ℝ := 30000
def new_car_bought_for : ℝ := 0.90 * new_car_sticker_price

-- Question and proof statement
def amount_out_of_pocket : ℝ := new_car_bought_for - old_car_sold_for

theorem james_out_of_pocket : amount_out_of_pocket = 11000 := by
  sorry

end NUMINAMATH_GPT_james_out_of_pocket_l1609_160994


namespace NUMINAMATH_GPT_original_apples_l1609_160957

-- Define the conditions
def sells_50_percent (initial remaining : ℕ) : Prop :=
  (initial / 2) = remaining

-- Define the goal
theorem original_apples (remaining : ℕ) (initial : ℕ) (h : sells_50_percent initial remaining) : initial = 10000 :=
by
  sorry

end NUMINAMATH_GPT_original_apples_l1609_160957


namespace NUMINAMATH_GPT_equal_distances_l1609_160933

theorem equal_distances (c : ℝ) (distance : ℝ) :
  abs (2 - -4) = distance ∧ (abs (c - -4) = distance ∨ abs (c - 2) = distance) ↔ (c = -10 ∨ c = 8) :=
by
  sorry

end NUMINAMATH_GPT_equal_distances_l1609_160933


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1609_160988

theorem solution_set_of_inequality :
  {x : ℝ | x^2 * (x - 4) ≥ 0} = {x : ℝ | x = 0 ∨ x ≥ 4} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1609_160988


namespace NUMINAMATH_GPT_value_of_a_l1609_160900

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem value_of_a (a : ℝ) (ha : a > 1) (h : f (g a) = 12) : 
  a = Real.sqrt (Real.sqrt 10 - 2) :=
by sorry

end NUMINAMATH_GPT_value_of_a_l1609_160900


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1609_160943

theorem arithmetic_sequence_problem 
  (a : ℕ → ℚ) 
  (a1 : a 1 = 1 / 3) 
  (a2_a5 : a 2 + a 5 = 4) 
  (an : ∃ n, a n = 33) :
  ∃ n, a n = 33 ∧ n = 50 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1609_160943


namespace NUMINAMATH_GPT_inequality_holds_l1609_160908

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) :
  ( (2 + x) / (1 + x) )^2 + ( (2 + y) / (1 + y) )^2 ≥ 9 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1609_160908


namespace NUMINAMATH_GPT_parabolic_arch_height_l1609_160921

/-- Define the properties of the parabolic arch -/
def parabolic_arch (a k x : ℝ) : ℝ := a * x^2 + k

/-- Define the conditions of the problem -/
def conditions (a k : ℝ) : Prop :=
  (parabolic_arch a k 25 = 0) ∧ (parabolic_arch a k 0 = 20)

theorem parabolic_arch_height (a k : ℝ) (condition_a_k : conditions a k) :
  parabolic_arch a k 10 = 16.8 :=
by
  unfold conditions at condition_a_k
  cases' condition_a_k with h1 h2
  sorry

end NUMINAMATH_GPT_parabolic_arch_height_l1609_160921


namespace NUMINAMATH_GPT_complex_quadrant_l1609_160968

theorem complex_quadrant (z : ℂ) (h : z = (↑0 + 1*I) / (1 + 1*I)) : z.re > 0 ∧ z.im > 0 := 
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1609_160968


namespace NUMINAMATH_GPT_range_of_b_l1609_160928

theorem range_of_b (a b : ℝ) : 
  (∀ x : ℝ, -3 < x ∧ x < 1 → (1 - a) * x^2 - 4 * x + 6 > 0) ∧
  (∀ x : ℝ, 3 * x^2 + b * x + 3 ≥ 0) →
  (-6 ≤ b ∧ b ≤ 6) :=
by
  sorry

end NUMINAMATH_GPT_range_of_b_l1609_160928


namespace NUMINAMATH_GPT_min_value_of_n_l1609_160926

theorem min_value_of_n 
  (n k : ℕ) 
  (h1 : 8 * n = 225 * k + 3)
  (h2 : k ≡ 5 [MOD 8]) : 
  n = 141 := 
  sorry

end NUMINAMATH_GPT_min_value_of_n_l1609_160926


namespace NUMINAMATH_GPT_binomial_variance_is_one_l1609_160944

noncomputable def binomial_variance (n : ℕ) (p : ℚ) : ℚ := n * p * (1 - p)

theorem binomial_variance_is_one :
  binomial_variance 4 (1 / 2) = 1 := by
  sorry

end NUMINAMATH_GPT_binomial_variance_is_one_l1609_160944


namespace NUMINAMATH_GPT_beverage_price_l1609_160953

theorem beverage_price (P : ℝ) :
  (3 * 2.25 + 4 * P + 4 * 1.00) / 6 = 2.79 → P = 1.50 :=
by
  intro h -- Introduce the hypothesis.
  sorry  -- Proof is omitted.

end NUMINAMATH_GPT_beverage_price_l1609_160953


namespace NUMINAMATH_GPT_compute_expression_l1609_160999

theorem compute_expression :
    ( (2 / 3) * Real.sqrt 15 - Real.sqrt 20 ) / ( (1 / 3) * Real.sqrt 5 ) = 2 * Real.sqrt 3 - 6 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1609_160999


namespace NUMINAMATH_GPT_exactly_one_divisible_by_5_l1609_160982

def a (n : ℕ) : ℕ := 2^(2*n + 1) - 2^(n + 1) + 1
def b (n : ℕ) : ℕ := 2^(2*n + 1) + 2^(n + 1) + 1

theorem exactly_one_divisible_by_5 (n : ℕ) (hn : 0 < n) : (a n % 5 = 0 ∧ b n % 5 ≠ 0) ∨ (a n % 5 ≠ 0 ∧ b n % 5 = 0) :=
  sorry

end NUMINAMATH_GPT_exactly_one_divisible_by_5_l1609_160982


namespace NUMINAMATH_GPT_wall_width_l1609_160901

theorem wall_width
  (w h l : ℝ)
  (h_eq : h = 6 * w)
  (l_eq : l = 7 * h)
  (volume_eq : w * h * l = 6804) :
  w = 3 :=
by
  sorry

end NUMINAMATH_GPT_wall_width_l1609_160901


namespace NUMINAMATH_GPT_Peter_can_guarantee_victory_l1609_160914

structure Board :=
  (size : ℕ)
  (cells : Fin size × Fin size → Option Color)

inductive Player
  | Peter
  | Victor
deriving DecidableEq

inductive Color
  | Red
  | Green
  | White
deriving DecidableEq

structure Move :=
  (player : Player)
  (rectangle : Fin 2 × Fin 2)
  (position : Fin 7 × Fin 7)

def isValidMove (board : Board) (move : Move) : Prop := sorry

def applyMove (board : Board) (move : Move) : Board := sorry

def allCellsColored (board : Board) : Prop := sorry

theorem Peter_can_guarantee_victory :
  ∀ (initialBoard : Board),
    (∀ (move : Move), move.player = Player.Victor → isValidMove initialBoard move) →
    Player.Peter = Player.Peter →
    (∃ finalBoard : Board,
       allCellsColored finalBoard ∧ 
       ¬ (∃ (move : Move), move.player = Player.Victor ∧ isValidMove finalBoard move)) :=
sorry

end NUMINAMATH_GPT_Peter_can_guarantee_victory_l1609_160914


namespace NUMINAMATH_GPT_playground_area_l1609_160975

open Real

theorem playground_area (l w : ℝ) (h1 : 2*l + 2*w = 100) (h2 : l = 2*w) : l * w = 5000 / 9 :=
by
  sorry

end NUMINAMATH_GPT_playground_area_l1609_160975


namespace NUMINAMATH_GPT_parallel_lines_iff_a_eq_1_l1609_160967

theorem parallel_lines_iff_a_eq_1 (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y - 1 = 0 ↔ x + 2*y + 4 = 0) ↔ (a = 1) := 
sorry

end NUMINAMATH_GPT_parallel_lines_iff_a_eq_1_l1609_160967


namespace NUMINAMATH_GPT_find_F2_l1609_160993

-- Set up the conditions as definitions
def m : ℝ := 1 -- in kg
def R1 : ℝ := 0.5 -- in meters
def R2 : ℝ := 1 -- in meters
def F1 : ℝ := 1 -- in Newtons

-- Rotational inertia I formula
def I (R : ℝ) : ℝ := m * R^2

-- Equality of angular accelerations
def alpha_eq (F1 F2 R1 R2 : ℝ) : Prop :=
  (F1 * R1) / (I R1) = (F2 * R2) / (I R2)

-- The proof goal
theorem find_F2 (F2 : ℝ) : 
  alpha_eq F1 F2 R1 R2 → F2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_F2_l1609_160993


namespace NUMINAMATH_GPT_license_count_l1609_160945

def num_licenses : ℕ :=
  let num_letters := 3
  let num_digits := 10
  let num_digit_slots := 6
  num_letters * num_digits ^ num_digit_slots

theorem license_count :
  num_licenses = 3000000 := by
  sorry

end NUMINAMATH_GPT_license_count_l1609_160945


namespace NUMINAMATH_GPT_daisies_per_bouquet_l1609_160905

def total_bouquets := 20
def rose_bouquets := 10
def roses_per_rose_bouquet := 12
def total_flowers_sold := 190

def total_roses_sold := rose_bouquets * roses_per_rose_bouquet
def daisy_bouquets := total_bouquets - rose_bouquets
def total_daisies_sold := total_flowers_sold - total_roses_sold

theorem daisies_per_bouquet :
  (total_daisies_sold / daisy_bouquets = 7) := sorry

end NUMINAMATH_GPT_daisies_per_bouquet_l1609_160905


namespace NUMINAMATH_GPT_max_profit_l1609_160977

variables (x y : ℝ)

def profit (x y : ℝ) : ℝ := 50000 * x + 30000 * y

theorem max_profit :
  (3 * x + y ≤ 13) ∧ (2 * x + 3 * y ≤ 18) ∧ (x ≥ 0) ∧ (y ≥ 0) →
  (∃ x y, profit x y = 390000) :=
by
  sorry

end NUMINAMATH_GPT_max_profit_l1609_160977


namespace NUMINAMATH_GPT_nathalie_cake_fraction_l1609_160983

theorem nathalie_cake_fraction
    (cake_weight : ℕ)
    (pierre_ate : ℕ)
    (double_what_nathalie_ate : pierre_ate = 2 * (pierre_ate / 2))
    (pierre_ate_correct : pierre_ate = 100) :
    (pierre_ate / 2) / cake_weight = 1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_nathalie_cake_fraction_l1609_160983


namespace NUMINAMATH_GPT_find_x_l1609_160938

theorem find_x (x : ℝ) (h : (3 * x) / 7 = 15) : x = 35 :=
sorry

end NUMINAMATH_GPT_find_x_l1609_160938


namespace NUMINAMATH_GPT_problem_proof_l1609_160935

noncomputable def problem : Prop :=
  ∀ x : ℝ, (x ≠ 2 ∧ (x-2)/(x-4) ≤ 3) ↔ (4 < x ∧ x < 5)

theorem problem_proof : problem := sorry

end NUMINAMATH_GPT_problem_proof_l1609_160935


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1609_160915

variable {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ d : ℝ, a (n + 1) = a n + d

-- Condition 2: Given property
def property (a : ℕ → ℝ) : Prop :=
a 7 + a 13 = 20

theorem arithmetic_sequence_sum (h_seq : is_arithmetic_sequence a) (h_prop : property a) :
  a 9 + a 10 + a 11 = 30 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1609_160915


namespace NUMINAMATH_GPT_ratios_of_square_areas_l1609_160984

variable (x : ℝ)

def square_area (side_length : ℝ) : ℝ := side_length^2

theorem ratios_of_square_areas (hA : square_area x = x^2)
                               (hB : square_area (5 * x) = 25 * x^2)
                               (hC : square_area (2 * x) = 4 * x^2) :
  (square_area x / square_area (5 * x) = 1 / 25 ∧
   square_area (2 * x) / square_area (5 * x) = 4 / 25) := 
by {
  sorry
}

end NUMINAMATH_GPT_ratios_of_square_areas_l1609_160984


namespace NUMINAMATH_GPT_probability_no_defective_pencils_l1609_160922

theorem probability_no_defective_pencils : 
  let total_pencils := 9
  let defective_pencils := 2
  let chosen_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils chosen_pencils
  let non_defective_ways := Nat.choose non_defective_pencils chosen_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 12 := 
by
  sorry

end NUMINAMATH_GPT_probability_no_defective_pencils_l1609_160922


namespace NUMINAMATH_GPT_quadratic_roots_k_relation_l1609_160907

theorem quadratic_roots_k_relation (k a b k1 k2 : ℝ) 
    (h_eq : k * (a^2 - a) + 2 * a + 7 = 0)
    (h_eq_b : k * (b^2 - b) + 2 * b + 7 = 0)
    (h_ratio : a / b + b / a = 3)
    (h_k : k = k1 ∨ k = k2)
    (h_vieta_sum : k1 + k2 = 39)
    (h_vieta_product : k1 * k2 = 4) :
    k1 / k2 + k2 / k1 = 1513 / 4 := 
    sorry

end NUMINAMATH_GPT_quadratic_roots_k_relation_l1609_160907


namespace NUMINAMATH_GPT_negation_of_there_exists_l1609_160998

theorem negation_of_there_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 3 = 0) ↔ ∀ x : ℝ, x^2 - x + 3 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_negation_of_there_exists_l1609_160998


namespace NUMINAMATH_GPT_group_booking_cost_correct_l1609_160920

-- Definitions based on the conditions of the problem
def weekday_rate_first_week : ℝ := 18.00
def weekend_rate_first_week : ℝ := 20.00
def weekday_rate_additional_weeks : ℝ := 11.00
def weekend_rate_additional_weeks : ℝ := 13.00
def security_deposit : ℝ := 50.00
def discount_rate : ℝ := 0.10
def group_size : ℝ := 5
def stay_duration : ℕ := 23

-- Computation of total cost
def total_cost (first_week_weekdays : ℕ) (first_week_weekends : ℕ) 
  (additional_week_weekdays : ℕ) (additional_week_weekends : ℕ) 
  (additional_days_weekdays : ℕ) : ℝ := 
  let cost_first_weekdays := first_week_weekdays * weekday_rate_first_week
  let cost_first_weekends := first_week_weekends * weekend_rate_first_week
  let cost_additional_weeks := 2 * (additional_week_weekdays * weekday_rate_additional_weeks + 
                                    additional_week_weekends * weekend_rate_additional_weeks)
  let cost_additional_days := additional_days_weekdays * weekday_rate_additional_weeks
  let total_before_deposit := cost_first_weekdays + cost_first_weekends + 
                              cost_additional_weeks + cost_additional_days
  let total_before_discount := total_before_deposit + security_deposit
  let total_discount := discount_rate * total_before_discount
  total_before_discount - total_discount

-- Proof setup
theorem group_booking_cost_correct :
  total_cost 5 2 5 2 2 = 327.60 :=
by
  -- Placeholder for the proof; steps not required for Lean statement
  sorry

end NUMINAMATH_GPT_group_booking_cost_correct_l1609_160920


namespace NUMINAMATH_GPT_x_squared_plus_y_squared_l1609_160940

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 16) : x^2 + y^2 = 356 := 
by
  sorry

end NUMINAMATH_GPT_x_squared_plus_y_squared_l1609_160940


namespace NUMINAMATH_GPT_selling_price_correct_l1609_160909

variable (CostPrice GainPercent : ℝ)
variables (Profit SellingPrice : ℝ)

noncomputable def calculateProfit : ℝ := (GainPercent / 100) * CostPrice

noncomputable def calculateSellingPrice : ℝ := CostPrice + calculateProfit CostPrice GainPercent

theorem selling_price_correct 
  (h1 : CostPrice = 900) 
  (h2 : GainPercent = 30)
  : calculateSellingPrice CostPrice GainPercent = 1170 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l1609_160909


namespace NUMINAMATH_GPT_cost_price_of_cloths_l1609_160903

-- Definitions based on conditions
def SP_A := 8500 / 85
def Profit_A := 15
def CP_A := SP_A - Profit_A

def SP_B := 10200 / 120
def Profit_B := 12
def CP_B := SP_B - Profit_B

def SP_C := 4200 / 60
def Profit_C := 10
def CP_C := SP_C - Profit_C

-- Theorem to prove the cost prices
theorem cost_price_of_cloths :
    CP_A = 85 ∧
    CP_B = 73 ∧
    CP_C = 60 :=
by
    sorry

end NUMINAMATH_GPT_cost_price_of_cloths_l1609_160903


namespace NUMINAMATH_GPT_simplify_expression_l1609_160946

theorem simplify_expression (a b : ℝ) :
  ((3 * a^3 * b - 12 * a^2 * b^2 - 6 * a * b^3) / (-3 * a * b) - 4 * a * b) = (-a^2 + 2 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1609_160946


namespace NUMINAMATH_GPT_find_A_l1609_160954

theorem find_A (A a b : ℝ) (h1 : 3^a = A) (h2 : 5^b = A) (h3 : 1/a + 1/b = 2) : A = Real.sqrt 15 :=
by
  /- Proof omitted -/
  sorry

end NUMINAMATH_GPT_find_A_l1609_160954


namespace NUMINAMATH_GPT_even_function_a_equals_one_l1609_160936

theorem even_function_a_equals_one 
  (a : ℝ) 
  (h : ∀ x : ℝ, 2^(-x) + a * 2^x = 2^x + a * 2^(-x)) : 
  a = 1 := 
by
  sorry

end NUMINAMATH_GPT_even_function_a_equals_one_l1609_160936


namespace NUMINAMATH_GPT_price_of_adult_ticket_eq_32_l1609_160997

theorem price_of_adult_ticket_eq_32 
  (num_adults : ℕ)
  (num_children : ℕ)
  (price_child_ticket : ℕ)
  (price_adult_ticket : ℕ)
  (total_collected : ℕ)
  (h1 : num_adults = 400)
  (h2 : num_children = 200)
  (h3 : price_adult_ticket = 2 * price_child_ticket)
  (h4 : total_collected = 16000)
  (h5 : total_collected = num_adults * price_adult_ticket + num_children * price_child_ticket)
  : price_adult_ticket = 32 := 
by
  sorry

end NUMINAMATH_GPT_price_of_adult_ticket_eq_32_l1609_160997


namespace NUMINAMATH_GPT_fred_baseball_cards_l1609_160970

theorem fred_baseball_cards :
  ∀ (fred_cards_initial melanie_bought : ℕ), fred_cards_initial = 5 → melanie_bought = 3 → fred_cards_initial - melanie_bought = 2 :=
by
  intros fred_cards_initial melanie_bought h1 h2
  sorry

end NUMINAMATH_GPT_fred_baseball_cards_l1609_160970


namespace NUMINAMATH_GPT_greatest_discarded_oranges_l1609_160992

theorem greatest_discarded_oranges (n : ℕ) : n % 7 ≤ 6 := 
by 
  sorry

end NUMINAMATH_GPT_greatest_discarded_oranges_l1609_160992


namespace NUMINAMATH_GPT_current_speed_correct_l1609_160959

noncomputable def speed_of_current : ℝ :=
  let rowing_speed_still_water := 10 -- speed of rowing in still water in kmph
  let distance_meters := 60 -- distance covered in meters
  let time_seconds := 17.998560115190788 -- time taken in seconds
  let distance_km := distance_meters / 1000 -- converting distance to kilometers
  let time_hours := time_seconds / 3600 -- converting time to hours
  let downstream_speed := distance_km / time_hours -- calculating downstream speed
  downstream_speed - rowing_speed_still_water -- calculating and returning the speed of the current

theorem current_speed_correct : speed_of_current = 2.00048 := by
  -- The proof is not provided in this statement as per the requirements.
  sorry

end NUMINAMATH_GPT_current_speed_correct_l1609_160959


namespace NUMINAMATH_GPT_bottom_row_bricks_l1609_160902

theorem bottom_row_bricks (x : ℕ) 
    (h : x + (x - 1) + (x - 2) + (x - 3) + (x - 4) = 200) : x = 42 :=
sorry

end NUMINAMATH_GPT_bottom_row_bricks_l1609_160902


namespace NUMINAMATH_GPT_factorization_a_minus_b_l1609_160924

theorem factorization_a_minus_b (a b : ℤ) (h1 : 3 * b + a = -7) (h2 : a * b = -6) : a - b = 7 :=
sorry

end NUMINAMATH_GPT_factorization_a_minus_b_l1609_160924


namespace NUMINAMATH_GPT_inscribed_sphere_radius_in_regular_octahedron_l1609_160976

theorem inscribed_sphere_radius_in_regular_octahedron (a : ℝ) (r : ℝ) 
  (h1 : a = 6)
  (h2 : let V := 72 * Real.sqrt 2; V = (1 / 3) * ((8 * (3 * Real.sqrt 3)) * r)) : 
  r = Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_in_regular_octahedron_l1609_160976


namespace NUMINAMATH_GPT_additional_birds_flew_up_l1609_160912

-- Defining the conditions from the problem
def original_birds : ℕ := 179
def total_birds : ℕ := 217

-- Defining the question to be proved as a theorem
theorem additional_birds_flew_up : 
  total_birds - original_birds = 38 :=
by
  sorry

end NUMINAMATH_GPT_additional_birds_flew_up_l1609_160912


namespace NUMINAMATH_GPT_tori_original_height_l1609_160925

-- Definitions for given conditions
def current_height : ℝ := 7.26
def height_gained : ℝ := 2.86

-- Theorem statement
theorem tori_original_height : current_height - height_gained = 4.40 :=
by sorry

end NUMINAMATH_GPT_tori_original_height_l1609_160925


namespace NUMINAMATH_GPT_value_of_5_l1609_160941

def q' (q : ℤ) : ℤ := 3 * q - 3

theorem value_of_5'_prime : q' (q' 5) = 33 :=
by
  sorry

end NUMINAMATH_GPT_value_of_5_l1609_160941


namespace NUMINAMATH_GPT_distance_between_foci_l1609_160986

theorem distance_between_foci (a b : ℝ) (h₁ : a^2 = 18) (h₂ : b^2 = 2) :
  2 * (Real.sqrt (a^2 + b^2)) = 4 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_l1609_160986


namespace NUMINAMATH_GPT_bagel_spending_l1609_160974

variable (B D : ℝ)

theorem bagel_spending (h1 : B - D = 12.50) (h2 : D = B * 0.75) : B + D = 87.50 := 
sorry

end NUMINAMATH_GPT_bagel_spending_l1609_160974


namespace NUMINAMATH_GPT_find_original_list_size_l1609_160949

theorem find_original_list_size
  (n m : ℤ)
  (h1 : (m + 3) * (n + 1) = m * n + 20)
  (h2 : (m + 1) * (n + 2) = m * n + 22):
  n = 7 :=
sorry

end NUMINAMATH_GPT_find_original_list_size_l1609_160949


namespace NUMINAMATH_GPT_solution_set_of_x_abs_x_lt_x_l1609_160985

theorem solution_set_of_x_abs_x_lt_x :
  {x : ℝ | x * |x| < x} = {x : ℝ | 0 < x ∧ x < 1} ∪ {x : ℝ | x < -1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_x_abs_x_lt_x_l1609_160985


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1609_160996

def f (x : ℝ) : ℝ := (x - 3) ^ 3 + x - 1

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) = a n + d

-- Problem Statement
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1609_160996


namespace NUMINAMATH_GPT_initial_tax_rate_l1609_160948

theorem initial_tax_rate 
  (income : ℝ)
  (differential_savings : ℝ)
  (final_tax_rate : ℝ)
  (initial_tax_rate : ℝ) 
  (h1 : income = 42400) 
  (h2 : differential_savings = 4240) 
  (h3 : final_tax_rate = 32)
  (h4 : differential_savings = (initial_tax_rate / 100) * income - (final_tax_rate / 100) * income) :
  initial_tax_rate = 42 :=
sorry

end NUMINAMATH_GPT_initial_tax_rate_l1609_160948


namespace NUMINAMATH_GPT_geometric_sequence_solution_l1609_160956

variables (a : ℕ → ℝ) (q : ℝ)
-- Given conditions
def condition1 : Prop := abs (a 1) = 1
def condition2 : Prop := a 5 = -8 * a 2
def condition3 : Prop := a 5 > a 2
-- Proof statement
theorem geometric_sequence_solution :
  condition1 a → condition2 a → condition3 a → ∀ n, a n = (-2)^(n - 1) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l1609_160956


namespace NUMINAMATH_GPT_tangent_lines_through_P_l1609_160911

noncomputable def curve_eq (x : ℝ) : ℝ := 1/3 * x^3 + 4/3

theorem tangent_lines_through_P (x y : ℝ) :
  ((4 * x - y - 4 = 0 ∨ y = x + 2) ∧ (curve_eq 2 = 4)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_lines_through_P_l1609_160911


namespace NUMINAMATH_GPT_problem_solution_l1609_160989

def count_valid_n : ℕ :=
  let count_mult_3 := (3000 / 3)
  let count_mult_6 := (3000 / 6)
  count_mult_3 - count_mult_6

theorem problem_solution : count_valid_n = 500 := 
sorry

end NUMINAMATH_GPT_problem_solution_l1609_160989


namespace NUMINAMATH_GPT_rhombus_area_2400_l1609_160950

noncomputable def area_of_rhombus (x y : ℝ) : ℝ :=
  2 * x * y

theorem rhombus_area_2400 (x y : ℝ) 
  (hx : x = 15) 
  (hy : y = (16 / 3) * x) 
  (rx : 18.75 * 4 * x * y = x * y * (78.75)) 
  (ry : 50 * 4 * x * y = x * y * (200)) : 
  area_of_rhombus 15 80 = 2400 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_2400_l1609_160950


namespace NUMINAMATH_GPT_total_height_of_tower_l1609_160971

theorem total_height_of_tower :
  let S₃₅ : ℕ := (35 * (35 + 1)) / 2
  let S₆₅ : ℕ := (65 * (65 + 1)) / 2
  S₃₅ + S₆₅ = 2775 :=
by
  let S₃₅ := (35 * (35 + 1)) / 2
  let S₆₅ := (65 * (65 + 1)) / 2
  sorry

end NUMINAMATH_GPT_total_height_of_tower_l1609_160971


namespace NUMINAMATH_GPT_tan_ratio_l1609_160952

theorem tan_ratio (a b : ℝ)
  (h1 : Real.cos (a + b) = 1 / 3)
  (h2 : Real.cos (a - b) = 1 / 2) :
  (Real.tan a) / (Real.tan b) = 5 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l1609_160952


namespace NUMINAMATH_GPT_inequality_has_solutions_l1609_160910

theorem inequality_has_solutions (a : ℝ) :
  (∃ x : ℝ, |x + 3| + |x - 1| < a^2 - 3 * a) ↔ (a < -1 ∨ 4 < a) := 
by
  sorry

end NUMINAMATH_GPT_inequality_has_solutions_l1609_160910


namespace NUMINAMATH_GPT_sally_orange_balloons_l1609_160937

def initial_orange_balloons : ℝ := 9.0
def found_orange_balloons : ℝ := 2.0

theorem sally_orange_balloons :
  initial_orange_balloons + found_orange_balloons = 11.0 := 
by
  sorry

end NUMINAMATH_GPT_sally_orange_balloons_l1609_160937


namespace NUMINAMATH_GPT_sum_a6_a7_a8_l1609_160939

-- Sequence definition and sum of the first n terms
def S (n : ℕ) : ℕ := n^2 + 3 * n

theorem sum_a6_a7_a8 : S 8 - S 5 = 48 :=
by
  -- Definition and proof details are skipped
  sorry

end NUMINAMATH_GPT_sum_a6_a7_a8_l1609_160939


namespace NUMINAMATH_GPT_gift_spending_l1609_160942

def total_amount : ℝ := 700.00
def wrapping_expenses : ℝ := 139.00
def amount_spent_on_gifts : ℝ := 700.00 - 139.00

theorem gift_spending :
  (total_amount - wrapping_expenses) = 561.00 :=
by
  sorry

end NUMINAMATH_GPT_gift_spending_l1609_160942


namespace NUMINAMATH_GPT_units_digit_quotient_4_l1609_160904

theorem units_digit_quotient_4 (n : ℕ) (h₁ : n ≥ 1) :
  (5^1994 + 6^1994) % 10 = 1 ∧ (5^1994 + 6^1994) % 7 = 5 → 
  (5^1994 + 6^1994) / 7 % 10 = 4 := 
sorry

end NUMINAMATH_GPT_units_digit_quotient_4_l1609_160904


namespace NUMINAMATH_GPT_strawberries_left_correct_l1609_160995

-- Define the initial and given away amounts in kilograms and grams
def initial_strawberries_kg : Int := 3
def initial_strawberries_g : Int := 300
def given_strawberries_kg : Int := 1
def given_strawberries_g : Int := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : Int) : Int := kg * 1000

-- Calculate the total strawberries initially and given away in grams
def total_initial_strawberries_g : Int :=
  (kg_to_g initial_strawberries_kg) + initial_strawberries_g

def total_given_strawberries_g : Int :=
  (kg_to_g given_strawberries_kg) + given_strawberries_g

-- The amount of strawberries left after giving some away
def strawberries_left : Int :=
  total_initial_strawberries_g - total_given_strawberries_g

-- The statement to prove
theorem strawberries_left_correct :
  strawberries_left = 1400 :=
by
  sorry

end NUMINAMATH_GPT_strawberries_left_correct_l1609_160995


namespace NUMINAMATH_GPT_axis_of_symmetry_l1609_160931

variables (a : ℝ) (x : ℝ)

def parabola := a * (x + 1) * (x - 3)

theorem axis_of_symmetry (h : a ≠ 0) : x = 1 := 
sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1609_160931


namespace NUMINAMATH_GPT_incorrect_expression_among_options_l1609_160990

theorem incorrect_expression_among_options :
  ¬(0.75 ^ (-0.3) < 0.75 ^ (0.1)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_expression_among_options_l1609_160990


namespace NUMINAMATH_GPT_increase_by_one_unit_l1609_160958

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 + 3 * x

-- State the theorem
theorem increase_by_one_unit (x : ℝ) : regression_eq (x + 1) - regression_eq x = 3 := by
  sorry

end NUMINAMATH_GPT_increase_by_one_unit_l1609_160958


namespace NUMINAMATH_GPT_total_fruit_punch_l1609_160930

/-- Conditions -/
def orange_punch : ℝ := 4.5
def cherry_punch : ℝ := 2 * orange_punch
def apple_juice : ℝ := cherry_punch - 1.5
def pineapple_juice : ℝ := 3
def grape_punch : ℝ := 1.5 * apple_juice

/-- Proof that total fruit punch is 35.25 liters -/
theorem total_fruit_punch :
  orange_punch + cherry_punch + apple_juice + pineapple_juice + grape_punch = 35.25 := by
  sorry

end NUMINAMATH_GPT_total_fruit_punch_l1609_160930


namespace NUMINAMATH_GPT_rosalina_gifts_l1609_160913

theorem rosalina_gifts (Emilio_gifts Jorge_gifts Pedro_gifts : ℕ) 
  (hEmilio : Emilio_gifts = 11) 
  (hJorge : Jorge_gifts = 6) 
  (hPedro : Pedro_gifts = 4) : 
  Emilio_gifts + Jorge_gifts + Pedro_gifts = 21 :=
by
  sorry

end NUMINAMATH_GPT_rosalina_gifts_l1609_160913


namespace NUMINAMATH_GPT_sqrt_diff_approx_l1609_160973

noncomputable def x : ℝ := Real.sqrt 50 - Real.sqrt 48

theorem sqrt_diff_approx : abs (x - 0.14) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_diff_approx_l1609_160973


namespace NUMINAMATH_GPT_arithmetic_seq_a4_l1609_160991

theorem arithmetic_seq_a4 (a : ℕ → ℕ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 4) 
  (h3 : a 3 = 6) : 
  a 4 = 8 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a4_l1609_160991


namespace NUMINAMATH_GPT_mode_of_gold_medals_is_8_l1609_160961

def countries : List String := ["Norway", "Germany", "China", "USA", "Sweden", "Netherlands", "Austria"]

def gold_medals : List Nat := [16, 12, 9, 8, 8, 8, 7]

def mode (lst : List Nat) : Nat :=
  lst.foldr
    (fun (x : Nat) acc =>
      if lst.count x > lst.count acc then x else acc)
    lst.head!

theorem mode_of_gold_medals_is_8 :
  mode gold_medals = 8 :=
by sorry

end NUMINAMATH_GPT_mode_of_gold_medals_is_8_l1609_160961


namespace NUMINAMATH_GPT_internal_diagonal_cubes_l1609_160906

theorem internal_diagonal_cubes :
  let A := (120, 360, 400)
  let gcd_xy := gcd 120 360
  let gcd_yz := gcd 360 400
  let gcd_zx := gcd 400 120
  let gcd_xyz := gcd (gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz
  new_cubes = 720 :=
by
  -- Definitions
  let A := (120, 360, 400)
  let gcd_xy := Int.gcd 120 360
  let gcd_yz := Int.gcd 360 400
  let gcd_zx := Int.gcd 400 120
  let gcd_xyz := Int.gcd (Int.gcd 120 360) 400
  let new_cubes := 120 + 360 + 400 - (gcd_xy + gcd_yz + gcd_zx) + gcd_xyz

  -- Assertion
  exact Eq.refl new_cubes

end NUMINAMATH_GPT_internal_diagonal_cubes_l1609_160906


namespace NUMINAMATH_GPT_area_of_region_between_semicircles_l1609_160917

/-- Given a region between two semicircles with the same center and parallel diameters,
where the farthest distance between two points with a clear line of sight is 12 meters,
prove that the area of the region is 18π square meters. -/
theorem area_of_region_between_semicircles :
  ∃ (R r : ℝ), R > r ∧ (R - r = 6) ∧ 18 * Real.pi = (Real.pi / 2) * (R^2 - r^2) ∧ (R^2 - r^2 = 144) :=
sorry

end NUMINAMATH_GPT_area_of_region_between_semicircles_l1609_160917


namespace NUMINAMATH_GPT_inequality_always_true_l1609_160919

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_inequality_always_true_l1609_160919


namespace NUMINAMATH_GPT_find_two_digit_number_l1609_160947

theorem find_two_digit_number : ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 10 * x + y = x^3 + y^2 ∧ 10 * x + y = 24 := by
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l1609_160947


namespace NUMINAMATH_GPT_part1_part2_l1609_160916

variable (x y z : ℕ)

theorem part1 (h1 : 3 * x + 5 * y = 98) (h2 : 8 * x + 3 * y = 158) : x = 16 ∧ y = 10 :=
sorry

theorem part2 (hx : x = 16) (hy : y = 10) (hz : 16 * z + 10 * (40 - z) ≤ 550) : z ≤ 25 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1609_160916


namespace NUMINAMATH_GPT_fraction_of_students_received_B_l1609_160962

theorem fraction_of_students_received_B {total_students : ℝ}
  (fraction_A : ℝ)
  (fraction_A_or_B : ℝ)
  (h_fraction_A : fraction_A = 0.7)
  (h_fraction_A_or_B : fraction_A_or_B = 0.9) :
  fraction_A_or_B - fraction_A = 0.2 :=
by
  rw [h_fraction_A, h_fraction_A_or_B]
  sorry

end NUMINAMATH_GPT_fraction_of_students_received_B_l1609_160962


namespace NUMINAMATH_GPT_divisible_by_factorial_l1609_160981

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, _ => 0
| _, 0 => 0
| n + 1, k + 1 => (n + 1) * (f (n + 1) k + f n k)

theorem divisible_by_factorial (n k : ℕ) : n! ∣ f n k := by sorry

end NUMINAMATH_GPT_divisible_by_factorial_l1609_160981


namespace NUMINAMATH_GPT_paint_cost_is_200_l1609_160965

-- Define the basic conditions and parameters
def side_length : ℕ := 5
def faces_of_cube : ℕ := 6
def area_per_face (side : ℕ) : ℕ := side * side
def total_surface_area (side : ℕ) (faces : ℕ) : ℕ := faces * area_per_face side
def coverage_per_kg : ℕ := 15
def cost_per_kg : ℕ := 20

-- Calculate total cost
def total_cost (side : ℕ) (faces : ℕ) (coverage : ℕ) (cost : ℕ) : ℕ :=
  let total_area := total_surface_area side faces
  let kgs_required := total_area / coverage
  kgs_required * cost

theorem paint_cost_is_200 :
  total_cost side_length faces_of_cube coverage_per_kg cost_per_kg = 200 :=
by
  sorry

end NUMINAMATH_GPT_paint_cost_is_200_l1609_160965


namespace NUMINAMATH_GPT_pairs_with_green_shirts_l1609_160964

theorem pairs_with_green_shirts (red_shirts green_shirts total_pairs red_pairs : ℕ) 
    (h1 : red_shirts = 70) 
    (h2 : green_shirts = 58) 
    (h3 : total_pairs = 64) 
    (h4 : red_pairs = 34) 
    : (∃ green_pairs : ℕ, green_pairs = 28) := 
by 
    sorry

end NUMINAMATH_GPT_pairs_with_green_shirts_l1609_160964


namespace NUMINAMATH_GPT_solve_inequalities_solve_fruit_purchase_l1609_160978

-- Part 1: Inequalities
theorem solve_inequalities {x : ℝ} : 
  (2 * x < 16) ∧ (3 * x > 2 * x + 3) → (3 < x ∧ x < 8) := by
  sorry

-- Part 2: Fruit Purchase
theorem solve_fruit_purchase {x y : ℝ} : 
  (x + y = 7) ∧ (5 * x + 8 * y = 41) → (x = 5 ∧ y = 2) := by
  sorry

end NUMINAMATH_GPT_solve_inequalities_solve_fruit_purchase_l1609_160978


namespace NUMINAMATH_GPT_max_rectangle_area_l1609_160951

theorem max_rectangle_area (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) (h1 : l + w = 20) (hlw : l = 10 ∨ w = 10) : 
(l = 10 ∧ w = 10 ∧ l * w = 100) :=
by sorry

end NUMINAMATH_GPT_max_rectangle_area_l1609_160951


namespace NUMINAMATH_GPT_part1_part2_l1609_160969

variable {x : ℝ}

/-- Prove that the range of the function f(x) = (sqrt(1+x) + sqrt(1-x) + 2) * (sqrt(1-x^2) + 1) for 0 ≤ x ≤ 1 is (0, 8]. -/
theorem part1 (hx : 0 ≤ x ∧ x ≤ 1) :
  0 < ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ∧ 
  ((Real.sqrt (1 + x) + Real.sqrt (1 - x) + 2) * (Real.sqrt (1 - x^2) + 1)) ≤ 8 :=
sorry

/-- Prove that for 0 ≤ x ≤ 1, there exists a positive number β such that sqrt(1+x) + sqrt(1-x) ≤ 2 - x^2 / β, with the minimal β = 4. -/
theorem part2 (hx : 0 ≤ x ∧ x ≤ 1) :
  ∃ β : ℝ, β > 0 ∧ β = 4 ∧ (Real.sqrt (1 + x) + Real.sqrt (1 - x) ≤ 2 - x^2 / β) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1609_160969


namespace NUMINAMATH_GPT_find_m_l1609_160987

def A : Set ℕ := {1, 3}
def B (m : ℕ) : Set ℕ := {1, 2, m}

theorem find_m (m : ℕ) (h : A ⊆ B m) : m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l1609_160987


namespace NUMINAMATH_GPT_polynomial_evaluation_qin_jiushao_l1609_160979

theorem polynomial_evaluation_qin_jiushao :
  let x := 3
  let V0 := 7
  let V1 := V0 * x + 6
  let V2 := V1 * x + 5
  let V3 := V2 * x + 4
  let V4 := V3 * x + 3
  V4 = 789 :=
by
  -- placeholder for proof
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_qin_jiushao_l1609_160979


namespace NUMINAMATH_GPT_percentage_increase_in_llama_cost_l1609_160960

def cost_of_goat : ℕ := 400
def number_of_goats : ℕ := 3
def total_cost : ℕ := 4800

def llamas_cost (x : ℕ) : Prop :=
  let total_cost_goats := number_of_goats * cost_of_goat
  let total_cost_llamas := total_cost - total_cost_goats
  let number_of_llamas := 2 * number_of_goats
  let cost_per_llama := total_cost_llamas / number_of_llamas
  let increase := cost_per_llama - cost_of_goat
  ((increase / cost_of_goat) * 100) = x

theorem percentage_increase_in_llama_cost :
  llamas_cost 50 :=
sorry

end NUMINAMATH_GPT_percentage_increase_in_llama_cost_l1609_160960


namespace NUMINAMATH_GPT_percentage_of_class_are_men_proof_l1609_160929

/-- Definition of the problem using the conditions provided. -/
def percentage_of_class_are_men (W M : ℝ) : Prop :=
  -- Conditions based on the problem statement
  M + W = 100 ∧
  0.10 * W + 0.85 * M = 40

/-- The proof statement we need to show: Under the given conditions, the percentage of men (M) is 40. -/
theorem percentage_of_class_are_men_proof (W M : ℝ) :
  percentage_of_class_are_men W M → M = 40 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_class_are_men_proof_l1609_160929


namespace NUMINAMATH_GPT_new_volume_increased_dimensions_l1609_160966

theorem new_volume_increased_dimensions (l w h : ℝ) 
  (h_vol : l * w * h = 5000) 
  (h_sa : l * w + w * h + h * l = 900) 
  (h_sum_edges : l + w + h = 60) : 
  (l + 2) * (w + 2) * (h + 2) = 7048 := 
by 
  sorry

end NUMINAMATH_GPT_new_volume_increased_dimensions_l1609_160966


namespace NUMINAMATH_GPT_baker_cakes_remaining_l1609_160963

theorem baker_cakes_remaining (initial_cakes: ℕ) (fraction_sold: ℚ) (sold_cakes: ℕ) (cakes_remaining: ℕ) :
  initial_cakes = 149 ∧ fraction_sold = 2/5 ∧ sold_cakes = 59 ∧ cakes_remaining = initial_cakes - sold_cakes → cakes_remaining = 90 :=
by
  sorry

end NUMINAMATH_GPT_baker_cakes_remaining_l1609_160963
