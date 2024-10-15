import Mathlib

namespace NUMINAMATH_GPT_aunt_gave_each_20_l1608_160856

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end NUMINAMATH_GPT_aunt_gave_each_20_l1608_160856


namespace NUMINAMATH_GPT_annual_interest_approx_l1608_160864

noncomputable def P : ℝ := 10000
noncomputable def r : ℝ := 0.05
noncomputable def t : ℝ := 1
noncomputable def e : ℝ := Real.exp 1

theorem annual_interest_approx :
  let A := P * Real.exp (r * t)
  let interest := A - P
  abs (interest - 512.71) < 0.01 := sorry

end NUMINAMATH_GPT_annual_interest_approx_l1608_160864


namespace NUMINAMATH_GPT_binom_20_4_plus_10_l1608_160879

open Nat

noncomputable def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_20_4_plus_10 :
  binom 20 4 + 10 = 4855 := by
  sorry

end NUMINAMATH_GPT_binom_20_4_plus_10_l1608_160879


namespace NUMINAMATH_GPT_average_goods_per_hour_l1608_160843

-- Define the conditions
def morning_goods : ℕ := 64
def morning_hours : ℕ := 4
def afternoon_rate : ℕ := 23
def afternoon_hours : ℕ := 3

-- Define the target statement to be proven
theorem average_goods_per_hour : (morning_goods + afternoon_rate * afternoon_hours) / (morning_hours + afternoon_hours) = 19 := by
  -- Add proof steps here
  sorry

end NUMINAMATH_GPT_average_goods_per_hour_l1608_160843


namespace NUMINAMATH_GPT_boat_license_combinations_l1608_160847

theorem boat_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let digit_positions := 5
  (letter_choices * (digit_choices ^ digit_positions)) = 300000 :=
  sorry

end NUMINAMATH_GPT_boat_license_combinations_l1608_160847


namespace NUMINAMATH_GPT_largest_difference_l1608_160889

noncomputable def A := 3 * 2023^2024
noncomputable def B := 2023^2024
noncomputable def C := 2022 * 2023^2023
noncomputable def D := 3 * 2023^2023
noncomputable def E := 2023^2023
noncomputable def F := 2023^2022

theorem largest_difference :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B := 
sorry

end NUMINAMATH_GPT_largest_difference_l1608_160889


namespace NUMINAMATH_GPT_point_on_hyperbola_l1608_160886

theorem point_on_hyperbola : 
  (∃ x y : ℝ, (x, y) = (3, -2) ∧ y = -6 / x) :=
by
  sorry

end NUMINAMATH_GPT_point_on_hyperbola_l1608_160886


namespace NUMINAMATH_GPT_four_transformations_of_1989_l1608_160849

-- Definition of the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Initial number
def initial_number : ℕ := 1989

-- Theorem statement
theorem four_transformations_of_1989 : 
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits initial_number))) = 9 :=
by
  sorry

end NUMINAMATH_GPT_four_transformations_of_1989_l1608_160849


namespace NUMINAMATH_GPT_mrs_heine_dogs_l1608_160868

theorem mrs_heine_dogs (total_biscuits biscuits_per_dog : ℕ) (h1 : total_biscuits = 6) (h2 : biscuits_per_dog = 3) :
  total_biscuits / biscuits_per_dog = 2 :=
by
  sorry

end NUMINAMATH_GPT_mrs_heine_dogs_l1608_160868


namespace NUMINAMATH_GPT_incorrect_operation_l1608_160815

theorem incorrect_operation (a : ℝ) : ¬ (a^3 + a^3 = 2 * a^6) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_operation_l1608_160815


namespace NUMINAMATH_GPT_boxes_filled_l1608_160881

theorem boxes_filled (total_toys toys_per_box : ℕ) (h1 : toys_per_box = 8) (h2 : total_toys = 32) :
  total_toys / toys_per_box = 4 :=
by
  sorry

end NUMINAMATH_GPT_boxes_filled_l1608_160881


namespace NUMINAMATH_GPT_geometric_sequence_an_l1608_160800

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 3 else 3 * (2:ℝ)^(n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  if n = 1 then 3 else (3 * (2:ℝ)^n - 3)

theorem geometric_sequence_an (n : ℕ) (h1 : a 1 = 3) (h2 : S 2 = 9) :
  a n = 3 * 2^(n-1) ∧ S n = 3 * (2^n - 1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_an_l1608_160800


namespace NUMINAMATH_GPT_t_of_polynomial_has_factor_l1608_160870

theorem t_of_polynomial_has_factor (t : ℤ) :
  (∃ a b : ℤ, x ^ 3 - x ^ 2 - 7 * x + t = (x + 1) * (x ^ 2 + a * x + b)) → t = -5 :=
by
  sorry

end NUMINAMATH_GPT_t_of_polynomial_has_factor_l1608_160870


namespace NUMINAMATH_GPT_golden_chest_diamonds_rubies_l1608_160867

theorem golden_chest_diamonds_rubies :
  ∀ (diamonds rubies : ℕ), diamonds = 421 → rubies = 377 → diamonds - rubies = 44 :=
by
  intros diamonds rubies
  sorry

end NUMINAMATH_GPT_golden_chest_diamonds_rubies_l1608_160867


namespace NUMINAMATH_GPT_driver_average_speed_l1608_160893

theorem driver_average_speed (v t : ℝ) (h1 : ∀ d : ℝ, d = v * t → (d / (v + 10)) = (3 / 4) * t) : v = 30 := by
  sorry

end NUMINAMATH_GPT_driver_average_speed_l1608_160893


namespace NUMINAMATH_GPT_sum_valid_two_digit_integers_l1608_160821

theorem sum_valid_two_digit_integers :
  ∃ S : ℕ, S = 36 ∧ (∀ n, 10 ≤ n ∧ n < 100 →
    (∃ a b, n = 10 * a + b ∧ a + b ∣ n ∧ 2 * a * b ∣ n → n = 36)) :=
by
  sorry

end NUMINAMATH_GPT_sum_valid_two_digit_integers_l1608_160821


namespace NUMINAMATH_GPT_pears_picked_l1608_160876

def Jason_pears : ℕ := 46
def Keith_pears : ℕ := 47
def Mike_pears : ℕ := 12
def total_pears : ℕ := 105

theorem pears_picked :
  Jason_pears + Keith_pears + Mike_pears = total_pears :=
by
  exact rfl

end NUMINAMATH_GPT_pears_picked_l1608_160876


namespace NUMINAMATH_GPT_find_apartment_number_l1608_160882

open Nat

def is_apartment_number (x a b : ℕ) : Prop :=
  x = 10 * a + b ∧ x = 17 * b

theorem find_apartment_number : ∃ x a b : ℕ, is_apartment_number x a b ∧ x = 85 :=
by
  sorry

end NUMINAMATH_GPT_find_apartment_number_l1608_160882


namespace NUMINAMATH_GPT_couple_slices_each_l1608_160802

noncomputable def slices_for_couple (total_slices children_slices people_in_couple : ℕ) : ℕ :=
  (total_slices - children_slices) / people_in_couple

theorem couple_slices_each (people_in_couple children slices_per_pizza num_pizzas : ℕ) (H1 : people_in_couple = 2) (H2 : children = 6) (H3 : slices_per_pizza = 4) (H4 : num_pizzas = 3) :
  slices_for_couple (num_pizzas * slices_per_pizza) (children * 1) people_in_couple = 3 := 
  by
  rw [H1, H2, H3, H4]
  show slices_for_couple (3 * 4) (6 * 1) 2 = 3
  rfl

end NUMINAMATH_GPT_couple_slices_each_l1608_160802


namespace NUMINAMATH_GPT_jacket_total_selling_price_l1608_160852

theorem jacket_total_selling_price :
  let original_price := 120
  let discount_rate := 0.30
  let tax_rate := 0.08
  let processing_fee := 5
  let discounted_price := original_price * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  let total_price := discounted_price + tax + processing_fee
  total_price = 95.72 := by
  sorry

end NUMINAMATH_GPT_jacket_total_selling_price_l1608_160852


namespace NUMINAMATH_GPT_principal_amount_l1608_160828

/-- Given:
 - 820 = P + (P * R * 2) / 100
 - 1020 = P + (P * R * 6) / 100
Prove:
 - P = 720
--/

theorem principal_amount (P R : ℝ) (h1 : 820 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 6) / 100) : P = 720 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l1608_160828


namespace NUMINAMATH_GPT_test_takers_percent_correct_l1608_160811

theorem test_takers_percent_correct 
  (n : Set ℕ → ℝ) 
  (A B : Set ℕ) 
  (hB : n B = 0.75) 
  (hAB : n (A ∩ B) = 0.60) 
  (hneither : n (Set.univ \ (A ∪ B)) = 0.05) 
  : n A = 0.80 := by
  sorry

end NUMINAMATH_GPT_test_takers_percent_correct_l1608_160811


namespace NUMINAMATH_GPT_system_solution_l1608_160812

theorem system_solution (x y : ℚ) (h1 : 2 * x - 3 * y = 1) (h2 : (y + 1) / 4 + 1 = (x + 2) / 3) : x = 3 ∧ y = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l1608_160812


namespace NUMINAMATH_GPT_wrapping_paper_area_correct_l1608_160801

noncomputable def wrapping_paper_area (l w h : ℝ) (hlw : l ≥ w) : ℝ :=
  (l + 2*h)^2

theorem wrapping_paper_area_correct (l w h : ℝ) (hlw : l ≥ w) :
  wrapping_paper_area l w h hlw = (l + 2*h)^2 :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_correct_l1608_160801


namespace NUMINAMATH_GPT_neg_p_sufficient_but_not_necessary_for_q_l1608_160830

variable {x : ℝ}

def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

theorem neg_p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, ¬ p x → q x) ∧ ¬ (∀ x : ℝ, q x → ¬ p x) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_sufficient_but_not_necessary_for_q_l1608_160830


namespace NUMINAMATH_GPT_sum_of_solutions_l1608_160816

theorem sum_of_solutions (a b c : ℚ) (h : a ≠ 0) (eq : 2 * x^2 - 7 * x - 9 = 0) : 
  (-b / a) = (7 / 2) := 
sorry

end NUMINAMATH_GPT_sum_of_solutions_l1608_160816


namespace NUMINAMATH_GPT_compare_abc_l1608_160885

noncomputable def a : ℝ := (1 / 4) * Real.logb 2 3
noncomputable def b : ℝ := 1 / 2
noncomputable def c : ℝ := (1 / 2) * Real.logb 5 3

theorem compare_abc : c < a ∧ a < b := sorry

end NUMINAMATH_GPT_compare_abc_l1608_160885


namespace NUMINAMATH_GPT_bob_initial_pennies_l1608_160820

-- Definitions of conditions
variables (a b : ℕ)
def condition1 : Prop := b + 2 = 4 * (a - 2)
def condition2 : Prop := b - 2 = 3 * (a + 2)

-- Goal: Proving that b = 62
theorem bob_initial_pennies (h1 : condition1 a b) (h2 : condition2 a b) : b = 62 :=
by {
  sorry
}

end NUMINAMATH_GPT_bob_initial_pennies_l1608_160820


namespace NUMINAMATH_GPT_tanya_dan_error_l1608_160841

theorem tanya_dan_error 
  (a b c d e f g : ℤ)
  (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g)
  (h₇ : a % 2 = 1) (h₈ : b % 2 = 1) (h₉ : c % 2 = 1) (h₁₀ : d % 2 = 1) 
  (h₁₁ : e % 2 = 1) (h₁₂ : f % 2 = 1) (h₁₃ : g % 2 = 1)
  (h₁₄ : (a + b + c + d + e + f + g) / 7 - d = 3 / 7) :
  false :=
by sorry

end NUMINAMATH_GPT_tanya_dan_error_l1608_160841


namespace NUMINAMATH_GPT_ticTacToeWinningDiagonals_l1608_160825

-- Define the tic-tac-toe board and the conditions
def ticTacToeBoard : Type := Fin 3 × Fin 3
inductive Player | X | O

def isWinningDiagonal (board : ticTacToeBoard → Option Player) : Prop :=
  (board (0, 0) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 2) = some Player.O) ∨
  (board (0, 2) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 0) = some Player.O)

-- Define the main problem statement
theorem ticTacToeWinningDiagonals : ∃ (n : ℕ), n = 40 :=
  sorry

end NUMINAMATH_GPT_ticTacToeWinningDiagonals_l1608_160825


namespace NUMINAMATH_GPT_exceed_1000_cents_l1608_160810

def total_amount (n : ℕ) : ℕ :=
  3 * (3 ^ n - 1) / (3 - 1)

theorem exceed_1000_cents : 
  ∃ n : ℕ, total_amount n ≥ 1000 ∧ (n + 7) % 7 = 6 := 
by
  sorry

end NUMINAMATH_GPT_exceed_1000_cents_l1608_160810


namespace NUMINAMATH_GPT_existence_of_nonnegative_value_l1608_160837

theorem existence_of_nonnegative_value :
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0 := 
by
  sorry

end NUMINAMATH_GPT_existence_of_nonnegative_value_l1608_160837


namespace NUMINAMATH_GPT_problem_statement_l1608_160846

theorem problem_statement : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1608_160846


namespace NUMINAMATH_GPT_find_distance_l1608_160807

-- Definitions based on conditions
def speed : ℝ := 75 -- in km/hr
def time : ℝ := 4 -- in hr

-- Statement to be proved
theorem find_distance : speed * time = 300 := by
  sorry

end NUMINAMATH_GPT_find_distance_l1608_160807


namespace NUMINAMATH_GPT_rachel_brought_16_brownies_l1608_160895

def total_brownies : ℕ := 40
def brownies_left_at_home : ℕ := 24

def brownies_brought_to_school : ℕ :=
  total_brownies - brownies_left_at_home

theorem rachel_brought_16_brownies :
  brownies_brought_to_school = 16 :=
by
  sorry

end NUMINAMATH_GPT_rachel_brought_16_brownies_l1608_160895


namespace NUMINAMATH_GPT_find_c_l1608_160862

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x, g_inv (g x c) = x) ↔ c = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_find_c_l1608_160862


namespace NUMINAMATH_GPT_gcd_ab_a2b2_eq_one_or_two_l1608_160891

-- Definitions and conditions
def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Problem statement
theorem gcd_ab_a2b2_eq_one_or_two (a b : ℕ) (h : coprime a b) : 
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_gcd_ab_a2b2_eq_one_or_two_l1608_160891


namespace NUMINAMATH_GPT_smallest_n_l1608_160873

theorem smallest_n (n : ℕ) (h1 : n % 6 = 4) (h2 : n % 7 = 3) (h3 : 15 < n) : n = 52 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1608_160873


namespace NUMINAMATH_GPT_total_pairs_of_shoes_equivalence_l1608_160863

variable (Scott Anthony Jim Melissa Tim: ℕ)

theorem total_pairs_of_shoes_equivalence
    (h1 : Scott = 7)
    (h2 : Anthony = 3 * Scott)
    (h3 : Jim = Anthony - 2)
    (h4 : Jim = 2 * Melissa)
    (h5 : Tim = (Anthony + Melissa) / 2):

  Scott + Anthony + Jim + Melissa + Tim = 71 :=
  by
  sorry

end NUMINAMATH_GPT_total_pairs_of_shoes_equivalence_l1608_160863


namespace NUMINAMATH_GPT_total_price_of_books_l1608_160818

theorem total_price_of_books (total_books: ℕ) (math_books: ℕ) (math_book_cost: ℕ) (history_book_cost: ℕ) (price: ℕ) 
  (h1 : total_books = 90) 
  (h2 : math_books = 54) 
  (h3 : math_book_cost = 4) 
  (h4 : history_book_cost = 5)
  (h5 : price = 396) :
  let history_books := total_books - math_books
  let math_books_price := math_books * math_book_cost
  let history_books_price := history_books * history_book_cost
  let total_price := math_books_price + history_books_price
  total_price = price := 
  by
    sorry

end NUMINAMATH_GPT_total_price_of_books_l1608_160818


namespace NUMINAMATH_GPT_lottery_win_probability_l1608_160804

theorem lottery_win_probability :
  let MegaBall_prob := 1 / 30
  let WinnerBall_prob := 1 / Nat.choose 50 5
  let BonusBall_prob := 1 / 15
  let Total_prob := MegaBall_prob * WinnerBall_prob * BonusBall_prob
  Total_prob = 1 / 953658000 :=
by
  sorry

end NUMINAMATH_GPT_lottery_win_probability_l1608_160804


namespace NUMINAMATH_GPT_amount_distributed_l1608_160829

theorem amount_distributed (A : ℕ) (h : A / 14 = A / 18 + 80) : A = 5040 :=
sorry

end NUMINAMATH_GPT_amount_distributed_l1608_160829


namespace NUMINAMATH_GPT_street_trees_one_side_number_of_street_trees_l1608_160824

-- Conditions
def road_length : ℕ := 2575
def interval : ℕ := 25
def trees_at_endpoints : ℕ := 2

-- Question: number of street trees on one side of the road
theorem street_trees_one_side (road_length interval : ℕ) (trees_at_endpoints : ℕ) : ℕ :=
  (road_length / interval) + 1

-- Proof of the provided problem
theorem number_of_street_trees : street_trees_one_side road_length interval trees_at_endpoints = 104 :=
by
  sorry

end NUMINAMATH_GPT_street_trees_one_side_number_of_street_trees_l1608_160824


namespace NUMINAMATH_GPT_polynomial_condition_l1608_160827

noncomputable def polynomial_of_degree_le (n : ℕ) (P : Polynomial ℝ) :=
  P.degree ≤ n

noncomputable def has_nonneg_coeff (P : Polynomial ℝ) :=
  ∀ i, 0 ≤ P.coeff i

theorem polynomial_condition
  (n : ℕ) (P : Polynomial ℝ)
  (h1 : polynomial_of_degree_le n P)
  (h2 : has_nonneg_coeff P)
  (h3 : ∀ x : ℝ, x > 0 → P.eval x * P.eval (1 / x) ≤ (P.eval 1) ^ 2) : 
  ∃ a_n : ℝ, 0 ≤ a_n ∧ P = Polynomial.C a_n * Polynomial.X^n :=
sorry

end NUMINAMATH_GPT_polynomial_condition_l1608_160827


namespace NUMINAMATH_GPT_carmen_burning_candles_l1608_160817

theorem carmen_burning_candles (candle_hours_per_night: ℕ) (nights_per_candle: ℕ) (candles_used: ℕ) (total_nights: ℕ) : 
  candle_hours_per_night = 2 →
  nights_per_candle = 8 / candle_hours_per_night →
  candles_used = 6 →
  total_nights = candles_used * (nights_per_candle / candle_hours_per_night) →
  total_nights = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_carmen_burning_candles_l1608_160817


namespace NUMINAMATH_GPT_original_weight_of_apples_l1608_160806

theorem original_weight_of_apples (x : ℕ) (h1 : 5 * (x - 30) = 2 * x) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_original_weight_of_apples_l1608_160806


namespace NUMINAMATH_GPT_zmod_field_l1608_160888

theorem zmod_field (p : ℕ) [Fact (Nat.Prime p)] : Field (ZMod p) :=
sorry

end NUMINAMATH_GPT_zmod_field_l1608_160888


namespace NUMINAMATH_GPT_find_value_of_f3_l1608_160854

variable {R : Type} [LinearOrderedField R]

/-- f is an odd function -/
def is_odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

/-- f is symmetric about the line x = 1 -/
def is_symmetric_about (f : R → R) (a : R) : Prop := ∀ x : R, f (a + x) = f (a - x)

variable (f : R → R)
variable (Hodd : is_odd_function f)
variable (Hsymmetric : is_symmetric_about f 1)
variable (Hf1 : f 1 = 2)

theorem find_value_of_f3 : f 3 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_f3_l1608_160854


namespace NUMINAMATH_GPT_find_certain_number_l1608_160809

theorem find_certain_number (x : ℝ) 
  (h : 3889 + x - 47.95000000000027 = 3854.002) : x = 12.95200000000054 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1608_160809


namespace NUMINAMATH_GPT_sum_of_squares_l1608_160866

theorem sum_of_squares (a d : Int) : 
  ∃ y1 y2 : Int, a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (3*a + y1*d)^2 + (a + y2*d)^2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1608_160866


namespace NUMINAMATH_GPT_fish_caught_in_second_catch_l1608_160860

theorem fish_caught_in_second_catch {N x : ℕ} (hN : N = 1750) (hx1 : 70 * x = 2 * N) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_fish_caught_in_second_catch_l1608_160860


namespace NUMINAMATH_GPT_compute_expression_l1608_160826

theorem compute_expression : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1608_160826


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l1608_160803

theorem simplify_and_evaluate_expr 
  (x : ℝ) 
  (h : x = 1/2) : 
  (2 * x - 1) ^ 2 - (3 * x + 1) * (3 * x - 1) + 5 * x * (x - 1) = -5 / 2 := 
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l1608_160803


namespace NUMINAMATH_GPT_travel_time_l1608_160835

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end NUMINAMATH_GPT_travel_time_l1608_160835


namespace NUMINAMATH_GPT_find_values_l1608_160844

theorem find_values (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 = 4 * a * b) :
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_values_l1608_160844


namespace NUMINAMATH_GPT_radius_ratio_l1608_160861

variable (VL VS rL rS : ℝ)
variable (hVL : VL = 432 * Real.pi)
variable (hVS : VS = 0.275 * VL)

theorem radius_ratio (h1 : (4 / 3) * Real.pi * rL^3 = VL)
                     (h2 : (4 / 3) * Real.pi * rS^3 = VS) :
  rS / rL = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_radius_ratio_l1608_160861


namespace NUMINAMATH_GPT_quarters_given_by_mom_l1608_160883

theorem quarters_given_by_mom :
  let dimes := 4
  let quarters := 4
  let nickels := 7
  let value_dimes := 0.10 * dimes
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let initial_total := value_dimes + value_quarters + value_nickels
  let final_total := 3.00
  let additional_amount := final_total - initial_total
  additional_amount / 0.25 = 5 :=
by
  sorry

end NUMINAMATH_GPT_quarters_given_by_mom_l1608_160883


namespace NUMINAMATH_GPT_largest_whole_number_lt_div_l1608_160851

theorem largest_whole_number_lt_div {x : ℕ} (hx : 8 * x < 80) : x ≤ 9 :=
by
  sorry

end NUMINAMATH_GPT_largest_whole_number_lt_div_l1608_160851


namespace NUMINAMATH_GPT_range_of_m_l1608_160877

theorem range_of_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / (x + 1) + 4 / y = 1) (m : ℝ) :
  x + y / 4 > m^2 - 5 * m - 3 ↔ -1 < m ∧ m < 6 := sorry

end NUMINAMATH_GPT_range_of_m_l1608_160877


namespace NUMINAMATH_GPT_max_value_x2_y3_z_l1608_160842

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  if x + y + z = 3 then x^2 * y^3 * z else 0

theorem max_value_x2_y3_z
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x + y + z = 3) :
  maximum_value x y z ≤ 9 / 16 := sorry

end NUMINAMATH_GPT_max_value_x2_y3_z_l1608_160842


namespace NUMINAMATH_GPT_john_learns_vowels_in_fifteen_days_l1608_160836

def days_to_learn_vowels (days_per_vowel : ℕ) (num_vowels : ℕ) : ℕ :=
  days_per_vowel * num_vowels

theorem john_learns_vowels_in_fifteen_days :
  days_to_learn_vowels 3 5 = 15 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_john_learns_vowels_in_fifteen_days_l1608_160836


namespace NUMINAMATH_GPT_soccer_team_players_l1608_160808

theorem soccer_team_players
  (first_half_starters : ℕ)
  (first_half_subs : ℕ)
  (second_half_mult : ℕ)
  (did_not_play : ℕ)
  (players_prepared : ℕ) :
  first_half_starters = 11 →
  first_half_subs = 2 →
  second_half_mult = 2 →
  did_not_play = 7 →
  players_prepared = 20 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_soccer_team_players_l1608_160808


namespace NUMINAMATH_GPT_problem_l1608_160878

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1608_160878


namespace NUMINAMATH_GPT_amount_paid_l1608_160872

theorem amount_paid (lemonade_price_per_cup sandwich_price_per_item change_received : ℝ) 
    (num_lemonades num_sandwiches : ℕ)
    (h1 : lemonade_price_per_cup = 2) 
    (h2 : sandwich_price_per_item = 2.50) 
    (h3 : change_received = 11) 
    (h4 : num_lemonades = 2) 
    (h5 : num_sandwiches = 2) : 
    (lemonade_price_per_cup * num_lemonades + sandwich_price_per_item * num_sandwiches + change_received = 20) :=
by
  sorry

end NUMINAMATH_GPT_amount_paid_l1608_160872


namespace NUMINAMATH_GPT_unit_prices_max_toys_l1608_160839

-- For question 1
theorem unit_prices (x y : ℕ)
  (h₁ : y = x + 25)
  (h₂ : 2*y + x = 200) : x = 50 ∧ y = 75 :=
by {
  sorry
}

-- For question 2
theorem max_toys (cost_a cost_b q_a q_b : ℕ)
  (h₁ : cost_a = 50)
  (h₂ : cost_b = 75)
  (h₃ : q_b = 2 * q_a)
  (h₄ : 50 * q_a + 75 * q_b ≤ 20000) : q_a ≤ 100 :=
by {
  sorry
}

end NUMINAMATH_GPT_unit_prices_max_toys_l1608_160839


namespace NUMINAMATH_GPT_determine_positive_integers_l1608_160892

theorem determine_positive_integers (x y z : ℕ) (h : x^2 + y^2 - 15 = 2^z) :
  (x = 0 ∧ y = 4 ∧ z = 0) ∨ (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 1 ∧ y = 4 ∧ z = 1) ∨ (x = 4 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_determine_positive_integers_l1608_160892


namespace NUMINAMATH_GPT_number_of_ways_to_form_divisible_number_l1608_160871

def valid_digits : List ℕ := [0, 2, 4, 7, 8, 9]

def is_divisible_by_4 (d1 d2 : ℕ) : Prop :=
  (d1 * 10 + d2) % 4 = 0

def is_divisible_by_3 (sum_of_digits : ℕ) : Prop :=
  sum_of_digits % 3 = 0

def replace_asterisks_to_form_divisible_number : Prop :=
  ∃ (a1 a2 a3 a4 a5 l : ℕ), a1 ∈ valid_digits ∧ a2 ∈ valid_digits ∧ a3 ∈ valid_digits ∧ a4 ∈ valid_digits ∧ a5 ∈ valid_digits ∧
  l ∈ [0, 2, 4, 8] ∧
  is_divisible_by_4 0 l ∧
  is_divisible_by_3 (11 + a1 + a2 + a3 + a4 + a5) ∧
  (4 * 324 = 1296)

theorem number_of_ways_to_form_divisible_number :
  replace_asterisks_to_form_divisible_number :=
  sorry

end NUMINAMATH_GPT_number_of_ways_to_form_divisible_number_l1608_160871


namespace NUMINAMATH_GPT_half_abs_sum_diff_squares_cubes_l1608_160890

theorem half_abs_sum_diff_squares_cubes (a b : ℤ) (h1 : a = 21) (h2 : b = 15) :
  (|a^2 - b^2| + |a^3 - b^3|) / 2 = 3051 := by
  sorry

end NUMINAMATH_GPT_half_abs_sum_diff_squares_cubes_l1608_160890


namespace NUMINAMATH_GPT_prob_XYZ_wins_l1608_160884

-- Define probabilities as given in the conditions
def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_Z : ℚ := 1 / 12

-- Define the probability that one of X, Y, or Z wins, assuming events are mutually exclusive
def P_XYZ_wins : ℚ := P_X + P_Y + P_Z

theorem prob_XYZ_wins : P_XYZ_wins = 11 / 24 := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_prob_XYZ_wins_l1608_160884


namespace NUMINAMATH_GPT_num_men_in_boat_l1608_160819

theorem num_men_in_boat 
  (n : ℕ) (W : ℝ)
  (h1 : (W / n : ℝ) = W / n)
  (h2 : (W + 8) / n = W / n + 1)
  : n = 8 := 
sorry

end NUMINAMATH_GPT_num_men_in_boat_l1608_160819


namespace NUMINAMATH_GPT_find_prime_pairs_l1608_160899

open Nat

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the problem as a theorem in Lean
theorem find_prime_pairs :
  ∀ (p n : ℕ), is_prime p ∧ n > 0 ∧ p^3 - 2*p^2 + p + 1 = 3^n ↔ (p = 2 ∧ n = 1) ∨ (p = 5 ∧ n = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_prime_pairs_l1608_160899


namespace NUMINAMATH_GPT_additional_discount_during_sale_l1608_160834

theorem additional_discount_during_sale:
  ∀ (list_price : ℝ) (max_typical_discount_pct : ℝ) (lowest_possible_sale_pct : ℝ),
  30 ≤ max_typical_discount_pct ∧ max_typical_discount_pct ≤ 50 ∧
  lowest_possible_sale_pct = 40 ∧ 
  list_price = 80 →
  ((max_typical_discount_pct * list_price / 100) - (lowest_possible_sale_pct * list_price / 100)) * 100 / 
    (max_typical_discount_pct * list_price / 100) = 20 :=
by
  sorry

end NUMINAMATH_GPT_additional_discount_during_sale_l1608_160834


namespace NUMINAMATH_GPT_combined_selling_price_l1608_160898

theorem combined_selling_price (C_c : ℕ) (C_s : ℕ) (C_m : ℕ) (L_c L_s L_m : ℕ)
  (hc : C_c = 1600)
  (hs : C_s = 12000)
  (hm : C_m = 45000)
  (hlc : L_c = 15)
  (hls : L_s = 10)
  (hlm : L_m = 5) :
  85 * C_c / 100 + 90 * C_s / 100 + 95 * C_m / 100 = 54910 := by
  sorry

end NUMINAMATH_GPT_combined_selling_price_l1608_160898


namespace NUMINAMATH_GPT_range_of_a_if_inequality_holds_l1608_160822

noncomputable def satisfies_inequality_for_all_xy_pos (a : ℝ) :=
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9

theorem range_of_a_if_inequality_holds :
  (∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9) → (a ≥ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_if_inequality_holds_l1608_160822


namespace NUMINAMATH_GPT_factorize_expression_l1608_160869

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1608_160869


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l1608_160875

theorem sqrt_meaningful_range (x : ℝ) (h : 2 * x - 3 ≥ 0) : x ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l1608_160875


namespace NUMINAMATH_GPT_students_neither_class_l1608_160814

theorem students_neither_class : 
  let total_students := 1500
  let music_students := 300
  let art_students := 200
  let dance_students := 100
  let theater_students := 50
  let music_art_students := 80
  let music_dance_students := 40
  let music_theater_students := 30
  let art_dance_students := 25
  let art_theater_students := 20
  let dance_theater_students := 10
  let music_art_dance_students := 50
  let music_art_theater_students := 30
  let art_dance_theater_students := 20
  let music_dance_theater_students := 10
  let all_four_students := 5
  total_students - 
    (music_students + 
     art_students + 
     dance_students + 
     theater_students - 
     (music_art_students + 
      music_dance_students + 
      music_theater_students + 
      art_dance_students + 
      art_theater_students + 
      dance_theater_students) + 
     (music_art_dance_students + 
      music_art_theater_students + 
      art_dance_theater_students + 
      music_dance_theater_students) - 
     all_four_students) = 950 :=
sorry

end NUMINAMATH_GPT_students_neither_class_l1608_160814


namespace NUMINAMATH_GPT_total_robodinos_in_shipment_l1608_160813

-- Definitions based on the conditions:
def percentage_on_display : ℝ := 0.30
def percentage_in_storage : ℝ := 0.70
def stored_robodinos : ℕ := 168

-- The main statement to prove:
theorem total_robodinos_in_shipment (T : ℝ) : (percentage_in_storage * T = stored_robodinos) → T = 240 := by
  sorry

end NUMINAMATH_GPT_total_robodinos_in_shipment_l1608_160813


namespace NUMINAMATH_GPT_petya_series_sum_l1608_160840

theorem petya_series_sum (n k : ℕ) (h1 : (n + k) * (k + 1) = 20 * (n + 2 * k)) 
                                      (h2 : (n + k) * (k + 1) = 60 * n) :
  n = 29 ∧ k = 29 :=
by
  sorry

end NUMINAMATH_GPT_petya_series_sum_l1608_160840


namespace NUMINAMATH_GPT_gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l1608_160838

theorem gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4 (k : Int) :
  Int.gcd ((360 * k)^2 + 6 * (360 * k) + 8) (360 * k + 4) = 4 := 
sorry

end NUMINAMATH_GPT_gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l1608_160838


namespace NUMINAMATH_GPT_puzzle_piece_total_l1608_160865

theorem puzzle_piece_total :
  let p1 := 1000
  let p2 := p1 + 0.30 * p1
  let p3 := 2 * p2
  let p4 := (p1 + p3) + 0.50 * (p1 + p3)
  let p5 := 3 * p4
  let p6 := p1 + p2 + p3 + p4 + p5
  p1 + p2 + p3 + p4 + p5 + p6 = 55000
:= sorry

end NUMINAMATH_GPT_puzzle_piece_total_l1608_160865


namespace NUMINAMATH_GPT_find_a1_a7_l1608_160894

-- Definitions based on the problem conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def a_3_5_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 5 = -6

def a_2_6_condition (a : ℕ → ℝ) : Prop :=
  a 2 * a 6 = 8

-- The theorem we need to prove
theorem find_a1_a7 (a : ℕ → ℝ) (ha : is_geometric_sequence a) (h35 : a_3_5_condition a) (h26 : a_2_6_condition a) :
  a 1 + a 7 = -9 :=
sorry

end NUMINAMATH_GPT_find_a1_a7_l1608_160894


namespace NUMINAMATH_GPT_number_of_ways_to_feed_animals_l1608_160845

-- Definitions for the conditions
def pairs_of_animals := 5
def alternating_feeding (start_with_female : Bool) (remaining_pairs : ℕ) : ℕ :=
if start_with_female then
  (pairs_of_animals.factorial / 2 ^ pairs_of_animals)
else
  0 -- we can ignore this case as it is not needed

-- Theorem statement
theorem number_of_ways_to_feed_animals :
  alternating_feeding true pairs_of_animals = 2880 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_feed_animals_l1608_160845


namespace NUMINAMATH_GPT_ravi_refrigerator_purchase_price_l1608_160850

theorem ravi_refrigerator_purchase_price (purchase_price_mobile : ℝ) (sold_mobile : ℝ)
  (profit : ℝ) (loss : ℝ) (overall_profit : ℝ)
  (H1 : purchase_price_mobile = 8000)
  (H2 : loss = 0.04)
  (H3 : profit = 0.10)
  (H4 : overall_profit = 200) :
  ∃ R : ℝ, 0.96 * R + sold_mobile = R + purchase_price_mobile + overall_profit ∧ R = 15000 :=
by
  use 15000
  sorry

end NUMINAMATH_GPT_ravi_refrigerator_purchase_price_l1608_160850


namespace NUMINAMATH_GPT_solve_equation_l1608_160858

theorem solve_equation (a b : ℚ) : 
  ((b = 0) → false) ∧ 
  ((4 * a - 3 = 0) → ((5 * b - 1 = 0) → a = 3 / 4 ∧ b = 1 / 5)) ∧ 
  ((4 * a - 3 ≠ 0) → (∃ x : ℚ, x = (5 * b - 1) / (4 * a - 3))) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1608_160858


namespace NUMINAMATH_GPT_remainder_when_sum_div_by_3_l1608_160832

theorem remainder_when_sum_div_by_3 
  (m n p q : ℕ)
  (a : ℕ := 6 * m + 4)
  (b : ℕ := 6 * n + 4)
  (c : ℕ := 6 * p + 4)
  (d : ℕ := 6 * q + 4)
  : (a + b + c + d) % 3 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_sum_div_by_3_l1608_160832


namespace NUMINAMATH_GPT_vectors_parallel_l1608_160848

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end NUMINAMATH_GPT_vectors_parallel_l1608_160848


namespace NUMINAMATH_GPT_find_m_value_l1608_160855

theorem find_m_value (m : ℤ) (h : (∀ x : ℤ, (x-5)*(x+7) = x^2 - mx - 35)) : m = -2 :=
by sorry

end NUMINAMATH_GPT_find_m_value_l1608_160855


namespace NUMINAMATH_GPT_total_cookies_l1608_160880

def total_chocolate_chip_batches := 5
def cookies_per_chocolate_chip_batch := 8
def total_oatmeal_batches := 3
def cookies_per_oatmeal_batch := 7
def total_sugar_batches := 1
def cookies_per_sugar_batch := 10
def total_double_chocolate_batches := 1
def cookies_per_double_chocolate_batch := 6

theorem total_cookies : 
  (total_chocolate_chip_batches * cookies_per_chocolate_chip_batch) +
  (total_oatmeal_batches * cookies_per_oatmeal_batch) +
  (total_sugar_batches * cookies_per_sugar_batch) +
  (total_double_chocolate_batches * cookies_per_double_chocolate_batch) = 77 :=
by sorry

end NUMINAMATH_GPT_total_cookies_l1608_160880


namespace NUMINAMATH_GPT_abs_of_neg_one_third_l1608_160833

theorem abs_of_neg_one_third : abs (- (1 / 3)) = (1 / 3) := by
  sorry

end NUMINAMATH_GPT_abs_of_neg_one_third_l1608_160833


namespace NUMINAMATH_GPT_gain_percent_is_30_l1608_160857

-- Given conditions
def CostPrice : ℕ := 100
def SellingPrice : ℕ := 130
def Gain : ℕ := SellingPrice - CostPrice
def GainPercent : ℕ := (Gain * 100) / CostPrice

-- The theorem to be proven
theorem gain_percent_is_30 :
  GainPercent = 30 := sorry

end NUMINAMATH_GPT_gain_percent_is_30_l1608_160857


namespace NUMINAMATH_GPT_find_x8_l1608_160831

theorem find_x8 (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 :=
by sorry

end NUMINAMATH_GPT_find_x8_l1608_160831


namespace NUMINAMATH_GPT_probability_N_taller_than_L_l1608_160823

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end NUMINAMATH_GPT_probability_N_taller_than_L_l1608_160823


namespace NUMINAMATH_GPT_jamies_shoes_cost_l1608_160874

-- Define the costs of items and the total cost.
def cost_total : ℤ := 110
def cost_coat : ℤ := 40
def cost_one_pair_jeans : ℤ := 20

-- Define the number of pairs of jeans.
def num_pairs_jeans : ℕ := 2

-- Define the cost of Jamie's shoes (to be proved).
def cost_jamies_shoes : ℤ := cost_total - (cost_coat + num_pairs_jeans * cost_one_pair_jeans)

theorem jamies_shoes_cost : cost_jamies_shoes = 30 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_jamies_shoes_cost_l1608_160874


namespace NUMINAMATH_GPT_james_coursework_materials_expense_l1608_160805

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end NUMINAMATH_GPT_james_coursework_materials_expense_l1608_160805


namespace NUMINAMATH_GPT_equal_area_division_l1608_160853

theorem equal_area_division (d : ℝ) : 
  (∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 
   (x = d ∨ x = 4) ∧ (y = 4 ∨ y = 0) ∧ 
   (2 : ℝ) * (4 - d) = 4) ↔ d = 2 :=
by
  sorry

end NUMINAMATH_GPT_equal_area_division_l1608_160853


namespace NUMINAMATH_GPT_nicky_catchup_time_l1608_160896

-- Definitions related to the problem
def head_start : ℕ := 12
def speed_cristina : ℕ := 5
def speed_nicky : ℕ := 3
def time_to_catchup : ℕ := 36
def nicky_runtime_before_catchup : ℕ := head_start + time_to_catchup

-- Theorem to prove the correct runtime for Nicky before Cristina catches up
theorem nicky_catchup_time : nicky_runtime_before_catchup = 48 := by
  sorry

end NUMINAMATH_GPT_nicky_catchup_time_l1608_160896


namespace NUMINAMATH_GPT_find_prime_triplets_l1608_160887

def is_prime (n : ℕ) : Prop := Nat.Prime n

def valid_triplet (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ (p * (r + 1) = q * (r + 5))

theorem find_prime_triplets :
  { (p, q, r) | valid_triplet p q r } = {(3, 2, 7), (5, 3, 5), (7, 3, 2)} :=
by {
  sorry -- Proof is to be completed
}

end NUMINAMATH_GPT_find_prime_triplets_l1608_160887


namespace NUMINAMATH_GPT_correct_statements_B_and_C_l1608_160897

variable {a b c : ℝ}

-- Definitions from the conditions
def conditionB (a b c : ℝ) : Prop := a > b ∧ b > 0 ∧ c < 0
def conclusionB (a b c : ℝ) : Prop := c / a^2 > c / b^2

def conditionC (a b c : ℝ) : Prop := c > a ∧ a > b ∧ b > 0
def conclusionC (a b c : ℝ) : Prop := a / (c - a) > b / (c - b)

theorem correct_statements_B_and_C (a b c : ℝ) : 
  (conditionB a b c → conclusionB a b c) ∧ 
  (conditionC a b c → conclusionC a b c) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_B_and_C_l1608_160897


namespace NUMINAMATH_GPT_g_at_2_l1608_160859

def g (x : ℝ) : ℝ := x^3 - x

theorem g_at_2 : g 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_g_at_2_l1608_160859
