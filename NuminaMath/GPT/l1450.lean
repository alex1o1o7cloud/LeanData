import Mathlib

namespace NUMINAMATH_GPT_compare_x_y_l1450_145083

theorem compare_x_y :
  let x := 123456789 * 123456786
  let y := 123456788 * 123456787
  x < y := sorry

end NUMINAMATH_GPT_compare_x_y_l1450_145083


namespace NUMINAMATH_GPT_watch_loss_percentage_l1450_145012

theorem watch_loss_percentage 
  (cost_price : ℕ) (gain_percent : ℕ) (extra_amount : ℕ) (selling_price_loss : ℕ)
  (h_cost_price : cost_price = 2500)
  (h_gain_percent : gain_percent = 10)
  (h_extra_amount : extra_amount = 500)
  (h_gain_condition : cost_price + gain_percent * cost_price / 100 = selling_price_loss + extra_amount) :
  (cost_price - selling_price_loss) * 100 / cost_price = 10 := 
by 
  sorry

end NUMINAMATH_GPT_watch_loss_percentage_l1450_145012


namespace NUMINAMATH_GPT_head_start_fraction_of_length_l1450_145009

-- Define the necessary variables and assumptions.
variables (Va Vb L H : ℝ)

-- Given conditions
def condition_speed_relation : Prop := Va = (22 / 19) * Vb
def condition_dead_heat : Prop := (L / Va) = ((L - H) / Vb)

-- The statement to be proven
theorem head_start_fraction_of_length (h_speed_relation: condition_speed_relation Va Vb) (h_dead_heat: condition_dead_heat L Va H Vb) : 
  H = (3 / 22) * L :=
sorry

end NUMINAMATH_GPT_head_start_fraction_of_length_l1450_145009


namespace NUMINAMATH_GPT_inequality_solution_l1450_145048

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 2 ↔ (0 < x ∧ x ≤ 0.5) ∨ (6 ≤ x) :=
by { sorry }

end NUMINAMATH_GPT_inequality_solution_l1450_145048


namespace NUMINAMATH_GPT_remainder_2017_div_89_l1450_145096

theorem remainder_2017_div_89 : 2017 % 89 = 59 :=
by
  sorry

end NUMINAMATH_GPT_remainder_2017_div_89_l1450_145096


namespace NUMINAMATH_GPT_original_number_exists_l1450_145030

theorem original_number_exists : 
  ∃ (t o : ℕ), (10 * t + o = 74) ∧ (t = o * o - 9) ∧ (10 * o + t = 10 * t + o - 27) :=
by
  sorry

end NUMINAMATH_GPT_original_number_exists_l1450_145030


namespace NUMINAMATH_GPT_product_of_coprime_numbers_l1450_145075

variable {a b c : ℕ}

def coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem product_of_coprime_numbers (h1 : coprime a b) (h2 : a * b = c) : Nat.lcm a b = c := by
  sorry

end NUMINAMATH_GPT_product_of_coprime_numbers_l1450_145075


namespace NUMINAMATH_GPT_task_D_is_suitable_l1450_145093

-- Definitions of the tasks
def task_A := "Investigating the age distribution of your classmates"
def task_B := "Understanding the ratio of male to female students in the eighth grade of your school"
def task_C := "Testing the urine samples of athletes who won championships at the Olympics"
def task_D := "Investigating the sleeping conditions of middle school students in Lishui City"

-- Definition of suitable_for_sampling_survey condition
def suitable_for_sampling_survey (task : String) : Prop :=
  task = task_D

-- Theorem statement
theorem task_D_is_suitable : suitable_for_sampling_survey task_D := by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_task_D_is_suitable_l1450_145093


namespace NUMINAMATH_GPT_cannot_tile_10x10_board_l1450_145060

-- Define the tiling board problem
def typeA_piece (i j : ℕ) : Prop := 
  ((i ≤ 98) ∧ (j ≤ 98) ∧ (i % 2 = 0) ∧ (j % 2 = 0))

def typeB_piece (i j : ℕ) : Prop := 
  ((i + 2 < 10) ∧ (j + 2 < 10))

def typeC_piece (i j : ℕ) : Prop := 
  ((i % 4 = 0 ∨ i % 4 = 2) ∧ (j % 4 = 0 ∨ j % 4 = 2))

-- Main theorem statement
theorem cannot_tile_10x10_board : 
  ¬ (∃ f : Fin 25 → Fin 10 × Fin 10, 
    (∀ k : Fin 25, typeA_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeB_piece (f k).1 (f k).2) ∨ 
    (∀ k : Fin 25, typeC_piece (f k).1 (f k).2)) :=
sorry

end NUMINAMATH_GPT_cannot_tile_10x10_board_l1450_145060


namespace NUMINAMATH_GPT_save_percentage_l1450_145091

theorem save_percentage (I S : ℝ) 
  (h1 : 1.5 * I - 2 * S + (I - S) = 2 * (I - S))
  (h2 : I ≠ 0) : 
  S / I = 0.5 :=
by sorry

end NUMINAMATH_GPT_save_percentage_l1450_145091


namespace NUMINAMATH_GPT_number_of_shirts_proof_l1450_145005

def regular_price := 50
def discount_percentage := 20
def total_paid := 240

def sale_price (rp : ℕ) (dp : ℕ) : ℕ := rp * (100 - dp) / 100

def number_of_shirts (tp : ℕ) (sp : ℕ) : ℕ := tp / sp

theorem number_of_shirts_proof : 
  number_of_shirts total_paid (sale_price regular_price discount_percentage) = 6 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_shirts_proof_l1450_145005


namespace NUMINAMATH_GPT_Harkamal_purchase_grapes_l1450_145002

theorem Harkamal_purchase_grapes
  (G : ℕ) -- The number of kilograms of grapes
  (cost_grapes_per_kg : ℕ := 70)
  (kg_mangoes : ℕ := 9)
  (cost_mangoes_per_kg : ℕ := 55)
  (total_paid : ℕ := 1195) :
  70 * G + 55 * 9 = 1195 → G = 10 := 
by
  sorry

end NUMINAMATH_GPT_Harkamal_purchase_grapes_l1450_145002


namespace NUMINAMATH_GPT_phone_cost_l1450_145010

theorem phone_cost (C : ℝ) (h1 : 0.40 * C + 780 = C) : C = 1300 := by
  sorry

end NUMINAMATH_GPT_phone_cost_l1450_145010


namespace NUMINAMATH_GPT_melanie_dimes_l1450_145038

variable (initial_dimes : ℕ) -- initial dimes Melanie had
variable (dimes_from_dad : ℕ) -- dimes given by dad
variable (dimes_to_mother : ℕ) -- dimes given to mother

def final_dimes (initial_dimes dimes_from_dad dimes_to_mother : ℕ) : ℕ :=
  initial_dimes + dimes_from_dad - dimes_to_mother

theorem melanie_dimes :
  initial_dimes = 7 →
  dimes_from_dad = 8 →
  dimes_to_mother = 4 →
  final_dimes initial_dimes dimes_from_dad dimes_to_mother = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end NUMINAMATH_GPT_melanie_dimes_l1450_145038


namespace NUMINAMATH_GPT_one_eighth_of_N_l1450_145070

theorem one_eighth_of_N
  (N : ℝ)
  (h : (6 / 11) * N = 48) : (1 / 8) * N = 11 :=
sorry

end NUMINAMATH_GPT_one_eighth_of_N_l1450_145070


namespace NUMINAMATH_GPT_find_a_l1450_145064

theorem find_a (a : ℝ) :
  {x : ℝ | (x + a) / ((x + 1) * (x + 3)) > 0} = {x : ℝ | x > -3 ∧ x ≠ -1} →
  a = 1 := 
by sorry

end NUMINAMATH_GPT_find_a_l1450_145064


namespace NUMINAMATH_GPT_determine_BD_l1450_145014

def quadrilateral (AB BC CD DA BD : ℕ) : Prop :=
AB = 6 ∧ BC = 15 ∧ CD = 8 ∧ DA = 12 ∧ (7 < BD ∧ BD < 18)

theorem determine_BD : ∃ BD : ℕ, quadrilateral 6 15 8 12 BD ∧ 8 ≤ BD ∧ BD ≤ 17 :=
by
  sorry

end NUMINAMATH_GPT_determine_BD_l1450_145014


namespace NUMINAMATH_GPT_probability_two_blue_marbles_l1450_145085

theorem probability_two_blue_marbles (h_red: ℕ := 3) (h_blue: ℕ := 4) (h_white: ℕ := 9) :
  (h_blue / (h_red + h_blue + h_white)) * ((h_blue - 1) / ((h_red + h_blue + h_white) - 1)) = 1 / 20 :=
by sorry

end NUMINAMATH_GPT_probability_two_blue_marbles_l1450_145085


namespace NUMINAMATH_GPT_beads_left_in_container_l1450_145041

theorem beads_left_in_container 
  (initial_beads green brown red total_beads taken_beads remaining_beads : Nat) 
  (h1 : green = 1) (h2 : brown = 2) (h3 : red = 3) 
  (h4 : total_beads = green + brown + red)
  (h5 : taken_beads = 2) 
  (h6 : remaining_beads = total_beads - taken_beads) : 
  remaining_beads = 4 := 
by
  sorry

end NUMINAMATH_GPT_beads_left_in_container_l1450_145041


namespace NUMINAMATH_GPT_minimize_S_n_at_7_l1450_145065

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n + 1) - a n = a 2 - a 1

def conditions (a : ℕ → ℤ) : Prop :=
arithmetic_sequence a ∧ a 2 = -11 ∧ (a 5 + a 9 = -2)

-- Define the sum of first n terms of the sequence
def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
(n * (a 1 + a n)) / 2

-- Define the minimum S_n and that it occurs at n = 7
theorem minimize_S_n_at_7 (a : ℕ → ℤ) (n : ℕ) (h : conditions a) :
  ∀ m, S a m ≥ S a 7 := sorry

end NUMINAMATH_GPT_minimize_S_n_at_7_l1450_145065


namespace NUMINAMATH_GPT_total_amount_correct_l1450_145098

noncomputable def total_amount (p_a r_a t_a p_b r_b t_b p_c r_c t_c : ℚ) : ℚ :=
  let final_price (p r t : ℚ) := p - (p * r / 100) + ((p - (p * r / 100)) * t / 100)
  final_price p_a r_a t_a + final_price p_b r_b t_b + final_price p_c r_c t_c

theorem total_amount_correct :
  total_amount 2500 6 10 3150 8 12 1000 5 7 = 6847.26 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_correct_l1450_145098


namespace NUMINAMATH_GPT_value_of_b_l1450_145025

theorem value_of_b (a b c : ℤ) : 
  (∃ d : ℤ, a = 17 + d ∧ b = 17 + 2 * d ∧ c = 17 + 3 * d ∧ 41 = 17 + 4 * d) → b = 29 :=
by
  intros h
  sorry


end NUMINAMATH_GPT_value_of_b_l1450_145025


namespace NUMINAMATH_GPT_Marta_books_directly_from_bookstore_l1450_145081

theorem Marta_books_directly_from_bookstore :
  let total_books_sale := 5
  let price_per_book_sale := 10
  let total_books_online := 2
  let total_cost_online := 40
  let total_spent := 210
  let cost_of_books_directly := 3 * total_cost_online
  let total_cost_sale := total_books_sale * price_per_book_sale
  let cost_per_book_directly := cost_of_books_directly / (total_cost_online / total_books_online)
  total_spent = total_cost_sale + total_cost_online + cost_of_books_directly ∧ (cost_of_books_directly / cost_per_book_directly) = 2 :=
by
  sorry

end NUMINAMATH_GPT_Marta_books_directly_from_bookstore_l1450_145081


namespace NUMINAMATH_GPT_difference_in_amount_paid_l1450_145024

variable (P Q : ℝ)

theorem difference_in_amount_paid (hP : P > 0) (hQ : Q > 0) :
  (1.10 * P * 0.80 * Q - P * Q) = -0.12 * (P * Q) := 
by 
  sorry

end NUMINAMATH_GPT_difference_in_amount_paid_l1450_145024


namespace NUMINAMATH_GPT_geometric_series_first_term_l1450_145067

theorem geometric_series_first_term 
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 24)
  (h_sum : S = a / (1 - r)) : 
  a = 18 :=
by {
  -- valid proof body goes here
  sorry
}

end NUMINAMATH_GPT_geometric_series_first_term_l1450_145067


namespace NUMINAMATH_GPT_no_internal_angle_less_than_60_l1450_145084

-- Define the concept of a Δ-curve
def delta_curve (K : Type) : Prop := sorry

-- Define the concept of a bicentric Δ-curve
def bicentric_delta_curve (K : Type) : Prop := sorry

-- Define the concept of internal angles of a Δ-curve
def has_internal_angle (K : Type) (A : ℝ) : Prop := sorry

-- The Lean statement for the problem
theorem no_internal_angle_less_than_60 (K : Type) 
  (h1 : delta_curve K) 
  (h2 : has_internal_angle K 60 ↔ bicentric_delta_curve K) :
  (∀ A < 60, ¬has_internal_angle K A) ∧ (has_internal_angle K 60 → bicentric_delta_curve K) := 
sorry

end NUMINAMATH_GPT_no_internal_angle_less_than_60_l1450_145084


namespace NUMINAMATH_GPT_max_gold_coins_l1450_145059

theorem max_gold_coins (n : ℕ) (k : ℕ) (H1 : n = 13 * k + 3) (H2 : n < 150) : n ≤ 146 := 
by
  sorry

end NUMINAMATH_GPT_max_gold_coins_l1450_145059


namespace NUMINAMATH_GPT_chip_final_balance_l1450_145071

noncomputable def finalBalance : ℝ := 
  let initialBalance := 50.0
  let month1InterestRate := 0.20
  let month2NewCharges := 20.0
  let month2InterestRate := 0.20
  let month3NewCharges := 30.0
  let month3Payment := 10.0
  let month3InterestRate := 0.25
  let month4NewCharges := 40.0
  let month4Payment := 20.0
  let month4InterestRate := 0.15

  -- Month 1
  let month1InterestFee := initialBalance * month1InterestRate
  let balanceMonth1 := initialBalance + month1InterestFee

  -- Month 2
  let balanceMonth2BeforeInterest := balanceMonth1 + month2NewCharges
  let month2InterestFee := balanceMonth2BeforeInterest * month2InterestRate
  let balanceMonth2 := balanceMonth2BeforeInterest + month2InterestFee

  -- Month 3
  let balanceMonth3BeforeInterest := balanceMonth2 + month3NewCharges
  let balanceMonth3AfterPayment := balanceMonth3BeforeInterest - month3Payment
  let month3InterestFee := balanceMonth3AfterPayment * month3InterestRate
  let balanceMonth3 := balanceMonth3AfterPayment + month3InterestFee

  -- Month 4
  let balanceMonth4BeforeInterest := balanceMonth3 + month4NewCharges
  let balanceMonth4AfterPayment := balanceMonth4BeforeInterest - month4Payment
  let month4InterestFee := balanceMonth4AfterPayment * month4InterestRate
  let balanceMonth4 := balanceMonth4AfterPayment + month4InterestFee

  balanceMonth4

theorem chip_final_balance : finalBalance = 189.75 := by sorry

end NUMINAMATH_GPT_chip_final_balance_l1450_145071


namespace NUMINAMATH_GPT_problem1_problem2_l1450_145063

theorem problem1 : (82 - 15) * (32 + 18) = 3350 :=
by
  sorry

theorem problem2 : (25 + 4) * 75 = 2175 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1450_145063


namespace NUMINAMATH_GPT_functional_relationship_optimizing_profit_l1450_145086

-- Define the scope of the problem with conditions and proof statements

variables (x : ℝ) (y : ℝ)

-- Conditions
def price_condition := 44 ≤ x ∧ x ≤ 52
def sales_function := y = -10 * x + 740
def profit_function (x : ℝ) := -10 * x^2 + 1140 * x - 29600

-- Lean statement to prove the first part
theorem functional_relationship (h₁ : 44 ≤ x) (h₂ : x ≤ 52) : y = -10 * x + 740 := by
  sorry

-- Lean statement to prove the second part
theorem optimizing_profit (h₃ : 44 ≤ x) (h₄ : x ≤ 52) : (profit_function 52 = 2640 ∧ (∀ x, (44 ≤ x ∧ x ≤ 52) → profit_function x ≤ 2640)) := by
  sorry

end NUMINAMATH_GPT_functional_relationship_optimizing_profit_l1450_145086


namespace NUMINAMATH_GPT_tangency_condition_l1450_145037

-- Definitions based on conditions
def ellipse (x y : ℝ) : Prop := 2 * x^2 + 3 * y^2 = 6
def hyperbola (x y n : ℝ) : Prop := 3 * x^2 - n * (y - 1)^2 = 3

-- The theorem statement based on the question and correct answer:
theorem tangency_condition (n : ℝ) (x y : ℝ) : 
  ellipse x y → hyperbola x y n → n = -6 :=
sorry

end NUMINAMATH_GPT_tangency_condition_l1450_145037


namespace NUMINAMATH_GPT_jar_filled_fraction_l1450_145074

variable (S L : ℝ)

-- Conditions
axiom h1 : S * (1/3) = L * (1/2)

-- Statement of the problem
theorem jar_filled_fraction :
  (L * (1/2)) + (S * (1/3)) = L := by
sorry

end NUMINAMATH_GPT_jar_filled_fraction_l1450_145074


namespace NUMINAMATH_GPT_each_friend_pays_6413_l1450_145033

noncomputable def amount_each_friend_pays (total_bill : ℝ) (friends : ℕ) (first_discount : ℝ) (second_discount : ℝ) : ℝ :=
  let bill_after_first_coupon := total_bill * (1 - first_discount)
  let bill_after_second_coupon := bill_after_first_coupon * (1 - second_discount)
  bill_after_second_coupon / friends

theorem each_friend_pays_6413 :
  amount_each_friend_pays 600 8 0.10 0.05 = 64.13 :=
by
  sorry

end NUMINAMATH_GPT_each_friend_pays_6413_l1450_145033


namespace NUMINAMATH_GPT_eval_expression_l1450_145020

theorem eval_expression : |-3| - (Real.sqrt 7 + 1)^0 - 2^2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1450_145020


namespace NUMINAMATH_GPT_digit_d_for_5678d_is_multiple_of_9_l1450_145090

def is_multiple_of_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem digit_d_for_5678d_is_multiple_of_9 : 
  ∃ d : ℕ, d < 10 ∧ is_multiple_of_9 (56780 + d) ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_digit_d_for_5678d_is_multiple_of_9_l1450_145090


namespace NUMINAMATH_GPT_product_is_solution_quotient_is_solution_l1450_145013

-- Definitions and conditions from the problem statement
variable (a b c d : ℤ)

-- The conditions
axiom h1 : a^2 - 5 * b^2 = 1
axiom h2 : c^2 - 5 * d^2 = 1

-- Lean 4 statement for the first part: the product
theorem product_is_solution :
  ∃ (m n : ℤ), ((m + n * (5:ℚ)) = (a + b * (5:ℚ)) * (c + d * (5:ℚ))) ∧ (m^2 - 5 * n^2 = 1) :=
sorry

-- Lean 4 statement for the second part: the quotient
theorem quotient_is_solution :
  ∃ (p q : ℤ), ((p + q * (5:ℚ)) = (a + b * (5:ℚ)) / (c + d * (5:ℚ))) ∧ (p^2 - 5 * q^2 = 1) :=
sorry

end NUMINAMATH_GPT_product_is_solution_quotient_is_solution_l1450_145013


namespace NUMINAMATH_GPT_find_y_value_l1450_145019

theorem find_y_value (k : ℝ) (x y : ℝ) (h1 : y = k * x^(1/5)) (h2 : y = 4) (h3 : x = 32) :
  y = 6 := by
  sorry

end NUMINAMATH_GPT_find_y_value_l1450_145019


namespace NUMINAMATH_GPT_predicted_whales_l1450_145057

theorem predicted_whales (num_last_year num_this_year num_next_year : ℕ)
  (h1 : num_this_year = 2 * num_last_year)
  (h2 : num_last_year = 4000)
  (h3 : num_next_year = 8800) :
  num_next_year - num_this_year = 800 :=
by
  sorry

end NUMINAMATH_GPT_predicted_whales_l1450_145057


namespace NUMINAMATH_GPT_infinite_series_sum_l1450_145054

noncomputable def sum_geometric_series (a b : ℝ) (h : ∑' n : ℕ, a / b ^ (n + 1) = 3) : ℝ :=
  ∑' n : ℕ, a / b ^ (n + 1)

theorem infinite_series_sum (a b c : ℝ) (h : sum_geometric_series a b (by sorry) = 3) :
  ∑' n : ℕ, (c * a) / (a + b) ^ (n + 1) = 3 * c / 4 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l1450_145054


namespace NUMINAMATH_GPT_normal_vector_proof_l1450_145023

-- Define the 3D vector type
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a specific normal vector n
def n : Vector3D := ⟨1, -2, 2⟩

-- Define the vector v we need to prove is a normal vector of the same plane
def v : Vector3D := ⟨2, -4, 4⟩

-- Define the statement (without the proof)
theorem normal_vector_proof : v = ⟨2 * n.x, 2 * n.y, 2 * n.z⟩ :=
by
  sorry

end NUMINAMATH_GPT_normal_vector_proof_l1450_145023


namespace NUMINAMATH_GPT_increasing_log_condition_range_of_a_l1450_145050

noncomputable def t (x a : ℝ) := x^2 - a*x + 3*a

theorem increasing_log_condition :
  (∀ x ≥ 2, 2 * x - a ≥ 0) ∧ a > -4 ∧ a ≤ 4 →
  ∀ x ≥ 2, x^2 - a*x + 3*a > 0 :=
by
  sorry

theorem range_of_a
  (h1 : ∀ x ≥ 2, 2 * x - a ≥ 0)
  (h2 : 4 - 2 * a + 3 * a > 0)
  (h3 : ∀ x ≥ 2, t x a > 0)
  : a > -4 ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_increasing_log_condition_range_of_a_l1450_145050


namespace NUMINAMATH_GPT_log_geometric_sequence_l1450_145049

theorem log_geometric_sequence :
  ∀ (a : ℕ → ℝ), (∀ n, 0 < a n) → (∃ r : ℝ, ∀ n, a (n + 1) = a n * r) →
  a 2 * a 18 = 16 → Real.logb 2 (a 10) = 2 :=
by
  intros a h_positive h_geometric h_condition
  sorry

end NUMINAMATH_GPT_log_geometric_sequence_l1450_145049


namespace NUMINAMATH_GPT_initialPersonsCount_l1450_145094

noncomputable def numberOfPersonsInitially (increaseInAverageWeight kg_diff : ℝ) : ℝ :=
  kg_diff / increaseInAverageWeight

theorem initialPersonsCount :
  numberOfPersonsInitially 2.5 20 = 8 := by
  sorry

end NUMINAMATH_GPT_initialPersonsCount_l1450_145094


namespace NUMINAMATH_GPT_probability_open_lock_l1450_145087

/-- Given 5 keys and only 2 can open the lock, the probability of opening the lock by selecting one key randomly is 0.4. -/
theorem probability_open_lock (k : Finset ℕ) (h₁ : k.card = 5) (s : Finset ℕ) (h₂ : s.card = 2 ∧ s ⊆ k) :
  ∃ p : ℚ, p = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_probability_open_lock_l1450_145087


namespace NUMINAMATH_GPT_log_49_48_in_terms_of_a_and_b_l1450_145076

-- Define the constants and hypotheses
variable (a b : ℝ)
variable (h1 : a = Real.logb 7 3)
variable (h2 : b = Real.logb 7 4)

-- Define the statement to be proved
theorem log_49_48_in_terms_of_a_and_b (a b : ℝ) (h1 : a = Real.logb 7 3) (h2 : b = Real.logb 7 4) :
  Real.logb 49 48 = (a + 2 * b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_log_49_48_in_terms_of_a_and_b_l1450_145076


namespace NUMINAMATH_GPT_percentage_of_boys_l1450_145072

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (total_students_eq : total_students = 42)
  (ratio_eq : boy_ratio = 3 ∧ girl_ratio = 4) :
  (boy_ratio + girl_ratio) = 7 ∧ (total_students / 7 * boy_ratio * 100 / total_students : ℚ) = 42.86 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_l1450_145072


namespace NUMINAMATH_GPT_decoded_word_is_correct_l1450_145052

-- Assume that we have a way to represent figures and encoded words
structure Figure1
structure Figure2

-- Assume the existence of a key that maps arrow patterns to letters
def decode (f1 : Figure1) (f2 : Figure2) : String := sorry

theorem decoded_word_is_correct (f1 : Figure1) (f2 : Figure2) :
  decode f1 f2 = "КОМПЬЮТЕР" :=
by
  sorry

end NUMINAMATH_GPT_decoded_word_is_correct_l1450_145052


namespace NUMINAMATH_GPT_sequence_problem_l1450_145039

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n - a (n - 1) = a 1 - a 0

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b n * b (n - 1) = b 1 * b 0

theorem sequence_problem
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : a 0 = -9) (ha1 : a 3 = -1) (ha_seq : arithmetic_sequence a)
  (hb : b 0 = -9) (hb4 : b 4 = -1) (hb_seq : geometric_sequence b) :
  b 2 * (a 2 - a 1) = -8 :=
sorry

end NUMINAMATH_GPT_sequence_problem_l1450_145039


namespace NUMINAMATH_GPT_minimize_expression_l1450_145088

theorem minimize_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (cond1 : x + y > z) (cond2 : y + z > x) (cond3 : z + x > y) :
  (x + y + z) * (1 / (x + y - z) + 1 / (y + z - x) + 1 / (z + x - y)) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_minimize_expression_l1450_145088


namespace NUMINAMATH_GPT_bucket_full_weight_l1450_145029

variables (x y p q : Real)

theorem bucket_full_weight (h1 : x + (1 / 4) * y = p)
                           (h2 : x + (3 / 4) * y = q) :
    x + y = 3 * q - p :=
by
  sorry

end NUMINAMATH_GPT_bucket_full_weight_l1450_145029


namespace NUMINAMATH_GPT_sprint_time_l1450_145053

def speed (Mark : Type) : ℝ := 6.0
def distance (Mark : Type) : ℝ := 144.0

theorem sprint_time (Mark : Type) : (distance Mark) / (speed Mark) = 24 := by
  sorry

end NUMINAMATH_GPT_sprint_time_l1450_145053


namespace NUMINAMATH_GPT_simplify_expression_l1450_145011

noncomputable def simplify_fraction (x : ℝ) (h : x ≠ 2) : ℝ :=
  (1 + (1 / (x - 2))) / ((x - x^2) / (x - 2))

theorem simplify_expression (x : ℝ) (h : x ≠ 2) : simplify_fraction x h = -(x - 1) / x :=
  sorry

end NUMINAMATH_GPT_simplify_expression_l1450_145011


namespace NUMINAMATH_GPT_number_divisible_by_7_last_digits_l1450_145000

theorem number_divisible_by_7_last_digits :
  ∀ d : ℕ, d ≤ 9 → ∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d :=
by
  sorry

end NUMINAMATH_GPT_number_divisible_by_7_last_digits_l1450_145000


namespace NUMINAMATH_GPT_mark_leftover_amount_l1450_145006

-- Definitions
def raise_percentage : ℝ := 0.05
def old_hourly_wage : ℝ := 40
def hours_per_week : ℝ := 8 * 5
def old_weekly_expenses : ℝ := 600
def new_expense : ℝ := 100

-- Calculate new hourly wage
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)

-- Calculate weekly earnings at the new wage
def weekly_earnings : ℝ := new_hourly_wage * hours_per_week

-- Calculate new total weekly expenses
def total_weekly_expenses : ℝ := old_weekly_expenses + new_expense

-- Calculate leftover amount
def leftover_per_week : ℝ := weekly_earnings - total_weekly_expenses

theorem mark_leftover_amount : leftover_per_week = 980 := by
  sorry

end NUMINAMATH_GPT_mark_leftover_amount_l1450_145006


namespace NUMINAMATH_GPT_radius_of_smaller_molds_l1450_145034

noncomputable def hemisphereVolume (r : ℝ) : ℝ :=
  (2 / 3) * Real.pi * r ^ 3

theorem radius_of_smaller_molds :
  ∀ (R r : ℝ), R = 2 ∧ (64 * hemisphereVolume r) = hemisphereVolume R → r = 1 / 2 :=
by
  intros R r h
  sorry

end NUMINAMATH_GPT_radius_of_smaller_molds_l1450_145034


namespace NUMINAMATH_GPT_average_six_consecutive_integers_starting_with_d_l1450_145035

theorem average_six_consecutive_integers_starting_with_d (c : ℝ) (d : ℝ)
  (h₁ : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  (d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5)) / 6 = c + 5 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_average_six_consecutive_integers_starting_with_d_l1450_145035


namespace NUMINAMATH_GPT_arithmetic_square_root_problem_l1450_145044

open Real

theorem arithmetic_square_root_problem 
  (a b c : ℝ)
  (ha : 5 * a - 2 = -27)
  (hb : b = ⌊sqrt 22⌋)
  (hc : c = -sqrt (4 / 25)) :
  sqrt (4 * a * c + 7 * b) = 6 := by
  sorry

end NUMINAMATH_GPT_arithmetic_square_root_problem_l1450_145044


namespace NUMINAMATH_GPT_exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l1450_145021

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared :
  ∃ (n : ℕ), sum_of_digits n = 1000 ∧ sum_of_digits (n ^ 2) = 1000000 := sorry

end NUMINAMATH_GPT_exists_nat_with_digit_sum_1000_and_square_digit_sum_1000_squared_l1450_145021


namespace NUMINAMATH_GPT_tim_total_money_raised_l1450_145055

-- Definitions based on conditions
def maxDonation : ℤ := 1200
def numMaxDonors : ℤ := 500
def numHalfDonors : ℤ := 3 * numMaxDonors
def halfDonation : ℤ := maxDonation / 2
def totalPercent : ℚ := 0.4

def totalDonationFromMaxDonors : ℤ := numMaxDonors * maxDonation
def totalDonationFromHalfDonors : ℤ := numHalfDonors * halfDonation
def totalDonation : ℤ := totalDonationFromMaxDonors + totalDonationFromHalfDonors

-- Proposition that Tim's total money raised is $3,750,000
theorem tim_total_money_raised : (totalDonation : ℚ) / totalPercent = 3750000 := by
  -- Verified in the proof steps
  sorry

end NUMINAMATH_GPT_tim_total_money_raised_l1450_145055


namespace NUMINAMATH_GPT_children_tickets_sold_l1450_145066

-- Given conditions
variables (A C : ℕ) -- A represents the number of adult tickets, C the number of children tickets.
variables (total_money total_tickets price_adult price_children : ℕ)
variables (total_money_eq : total_money = 104)
variables (total_tickets_eq : total_tickets = 21)
variables (price_adult_eq : price_adult = 6)
variables (price_children_eq : price_children = 4)
variables (money_eq : price_adult * A + price_children * C = total_money)
variables (tickets_eq : A + C = total_tickets)

-- Problem statement: prove that C = 11
theorem children_tickets_sold : C = 11 :=
by
  -- Necessary Lean code to handle proof here (omitting proof details as instructed)
  sorry

end NUMINAMATH_GPT_children_tickets_sold_l1450_145066


namespace NUMINAMATH_GPT_max_chords_l1450_145007

noncomputable def max_closed_chords (n : ℕ) (h : n ≥ 3) : ℕ :=
  n

/-- Given an integer number n ≥ 3 and n distinct points on a circle, labeled 1 through n,
prove that the maximum number of closed chords [ij], i ≠ j, having pairwise non-empty intersections is n. -/
theorem max_chords {n : ℕ} (h : n ≥ 3) :
  max_closed_chords n h = n := 
sorry

end NUMINAMATH_GPT_max_chords_l1450_145007


namespace NUMINAMATH_GPT_james_meditation_sessions_l1450_145036

theorem james_meditation_sessions (minutes_per_session : ℕ) (hours_per_week : ℕ) (days_per_week : ℕ) (h1 : minutes_per_session = 30) (h2 : hours_per_week = 7) (h3 : days_per_week = 7) : 
  (hours_per_week * 60 / days_per_week / minutes_per_session) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_james_meditation_sessions_l1450_145036


namespace NUMINAMATH_GPT_div_problem_l1450_145018

theorem div_problem (a b c : ℝ) (h1 : a / (b * c) = 4) (h2 : (a / b) / c = 12) : a / b = 4 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_div_problem_l1450_145018


namespace NUMINAMATH_GPT_student_history_score_l1450_145051

theorem student_history_score 
  (math : ℕ) 
  (third : ℕ) 
  (average : ℕ) 
  (H : ℕ) 
  (h_math : math = 74)
  (h_third : third = 67)
  (h_avg : average = 75)
  (h_overall_avg : (math + third + H) / 3 = average) : 
  H = 84 :=
by
  sorry

end NUMINAMATH_GPT_student_history_score_l1450_145051


namespace NUMINAMATH_GPT_evaluate_expression_l1450_145032

theorem evaluate_expression (a : ℝ) (h : a = 4 / 3) : 
  (4 * a^2 - 12 * a + 9) * (3 * a - 4) = 0 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1450_145032


namespace NUMINAMATH_GPT_maci_school_supplies_cost_l1450_145047

theorem maci_school_supplies_cost :
  let blue_pen_cost := 0.10
  let red_pen_cost := 2 * blue_pen_cost
  let pencil_cost := red_pen_cost / 2
  let notebook_cost := 10 * blue_pen_cost
  let blue_pen_count := 10
  let red_pen_count := 15
  let pencil_count := 5
  let notebook_count := 3
  let total_pen_count := blue_pen_count + red_pen_count
  let total_cost_before_discount := 
      blue_pen_count * blue_pen_cost + 
      red_pen_count * red_pen_cost + 
      pencil_count * pencil_cost + 
      notebook_count * notebook_cost
  let pen_discount_rate := if total_pen_count > 12 then 0.10 else 0
  let notebook_discount_rate := if notebook_count > 4 then 0.20 else 0
  let pen_discount := pen_discount_rate * (blue_pen_count * blue_pen_cost + red_pen_count * red_pen_cost)
  let total_cost_after_discount := 
      total_cost_before_discount - pen_discount
  total_cost_after_discount = 7.10 :=
by
  sorry

end NUMINAMATH_GPT_maci_school_supplies_cost_l1450_145047


namespace NUMINAMATH_GPT_quadratic_residues_count_l1450_145003

theorem quadratic_residues_count (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) :
  ∃ (q_residues : Finset (ZMod p)), q_residues.card = (p - 1) / 2 ∧
  ∃ (nq_residues : Finset (ZMod p)), nq_residues.card = (p - 1) / 2 ∧
  ∀ d ∈ q_residues, ∃ x y : ZMod p, x^2 = d ∧ y^2 = d ∧ x ≠ y :=
by
  sorry

end NUMINAMATH_GPT_quadratic_residues_count_l1450_145003


namespace NUMINAMATH_GPT_marys_remaining_money_l1450_145062

def drinks_cost (p : ℝ) := 4 * p
def medium_pizzas_cost (p : ℝ) := 3 * (3 * p)
def large_pizzas_cost (p : ℝ) := 2 * (5 * p)
def total_initial_money := 50

theorem marys_remaining_money (p : ℝ) : 
  total_initial_money - (drinks_cost p + medium_pizzas_cost p + large_pizzas_cost p) = 50 - 23 * p :=
by
  sorry

end NUMINAMATH_GPT_marys_remaining_money_l1450_145062


namespace NUMINAMATH_GPT_pascal_triangle_ratio_l1450_145017

theorem pascal_triangle_ratio (n r : ℕ) (hn1 : 5 * r = 2 * n - 3) (hn2 : 7 * r = 3 * n - 11) : n = 34 :=
by
  -- The proof steps will fill here eventually
  sorry

end NUMINAMATH_GPT_pascal_triangle_ratio_l1450_145017


namespace NUMINAMATH_GPT_trig_identity_simplify_l1450_145068

-- Define the problem in Lean 4
theorem trig_identity_simplify (α : Real) : (Real.sin (α - Real.pi / 2) * Real.tan (Real.pi - α)) = Real.sin α :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_simplify_l1450_145068


namespace NUMINAMATH_GPT_inequality_solution_l1450_145022

noncomputable def f (a b x : ℝ) : ℝ := 1 / Real.sqrt x + 1 / Real.sqrt (a + b - x)

theorem inequality_solution 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (x : ℝ) 
  (hx : x ∈ Set.Ioo (min a b) (max a b)) : 
  f a b x < f a b a ∧ f a b x < f a b b := 
sorry

end NUMINAMATH_GPT_inequality_solution_l1450_145022


namespace NUMINAMATH_GPT_midpoint_one_sixth_one_twelfth_l1450_145015

theorem midpoint_one_sixth_one_twelfth : (1 : ℚ) / 8 = (1 / 6 + 1 / 12) / 2 := by
  sorry

end NUMINAMATH_GPT_midpoint_one_sixth_one_twelfth_l1450_145015


namespace NUMINAMATH_GPT_find_number_l1450_145058

theorem find_number (n : ℕ) (h1 : 45 = 11 * n + 1) : n = 4 :=
  sorry

end NUMINAMATH_GPT_find_number_l1450_145058


namespace NUMINAMATH_GPT_ones_digit_power_sum_l1450_145082

noncomputable def ones_digit_of_power_sum_is_5 : Prop :=
  (1^2010 + 2^2010 + 3^2010 + 4^2010 + 5^2010 + 6^2010 + 7^2010 + 8^2010 + 9^2010 + 10^2010) % 10 = 5

theorem ones_digit_power_sum : ones_digit_of_power_sum_is_5 :=
  sorry

end NUMINAMATH_GPT_ones_digit_power_sum_l1450_145082


namespace NUMINAMATH_GPT_function_has_zero_in_interval_l1450_145016

   theorem function_has_zero_in_interval (fA fB fC fD : ℝ → ℝ) (hA : ∀ x, fA x = x - 3)
       (hB : ∀ x, fB x = 2^x) (hC : ∀ x, fC x = x^2) (hD : ∀ x, fD x = Real.log x) :
       ∃ x, 0 < x ∧ x < 2 ∧ fD x = 0 :=
   by
       sorry
   
end NUMINAMATH_GPT_function_has_zero_in_interval_l1450_145016


namespace NUMINAMATH_GPT_lemonade_syrup_parts_l1450_145097

theorem lemonade_syrup_parts (L : ℝ) :
  (L = 2 / 0.75) →
  (L = 2.6666666666666665) :=
by
  sorry

end NUMINAMATH_GPT_lemonade_syrup_parts_l1450_145097


namespace NUMINAMATH_GPT_max_profit_price_l1450_145046

-- Define the conditions
def hotel_rooms : ℕ := 50
def base_price : ℕ := 180
def price_increase : ℕ := 10
def expense_per_room : ℕ := 20

-- Define the price as a function of x
def room_price (x : ℕ) : ℕ := base_price + price_increase * x

-- Define the number of occupied rooms as a function of x
def occupied_rooms (x : ℕ) : ℕ := hotel_rooms - x

-- Define the profit function
def profit (x : ℕ) : ℕ := (room_price x - expense_per_room) * occupied_rooms x

-- The statement to be proven:
theorem max_profit_price : ∃ (x : ℕ), room_price x = 350 ∧ ∀ y : ℕ, profit y ≤ profit x :=
by
  sorry

end NUMINAMATH_GPT_max_profit_price_l1450_145046


namespace NUMINAMATH_GPT_x_coordinate_point_P_l1450_145040

theorem x_coordinate_point_P (x y : ℝ) (h_on_parabola : y^2 = 4 * x) 
  (h_distance : dist (x, y) (1, 0) = 3) : x = 2 :=
sorry

end NUMINAMATH_GPT_x_coordinate_point_P_l1450_145040


namespace NUMINAMATH_GPT_card_at_position_52_l1450_145027

def cards_order : List String := ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]

theorem card_at_position_52 : cards_order[(52 % 13)] = "A" :=
by
  -- proof will be added here
  sorry

end NUMINAMATH_GPT_card_at_position_52_l1450_145027


namespace NUMINAMATH_GPT_words_difference_l1450_145073

-- Definitions based on conditions.
def right_hand_speed (words_per_minute : ℕ) := 10
def left_hand_speed (words_per_minute : ℕ) := 7
def time_duration (minutes : ℕ) := 5

-- Problem statement
theorem words_difference :
  let right_hand_words := right_hand_speed 0 * time_duration 0
  let left_hand_words := left_hand_speed 0 * time_duration 0
  (right_hand_words - left_hand_words) = 15 :=
by
  sorry

end NUMINAMATH_GPT_words_difference_l1450_145073


namespace NUMINAMATH_GPT_cost_of_antibiotics_for_a_week_l1450_145045

noncomputable def antibiotic_cost : ℕ := 3
def doses_per_day : ℕ := 3
def days_in_week : ℕ := 7

theorem cost_of_antibiotics_for_a_week : doses_per_day * days_in_week * antibiotic_cost = 63 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_antibiotics_for_a_week_l1450_145045


namespace NUMINAMATH_GPT_find_n_l1450_145056

noncomputable def n : ℕ := sorry -- Explicitly define n as a variable, but the value is not yet provided.

theorem find_n (h₁ : n > 0)
    (h₂ : Real.sqrt 3 > (n + 4) / (n + 1))
    (h₃ : Real.sqrt 3 < (n + 3) / n) : 
    n = 4 :=
sorry

end NUMINAMATH_GPT_find_n_l1450_145056


namespace NUMINAMATH_GPT_Ava_watch_minutes_l1450_145095

theorem Ava_watch_minutes (hours_watched : ℕ) (minutes_per_hour : ℕ) (h : hours_watched = 4) (m : minutes_per_hour = 60) : 
  hours_watched * minutes_per_hour = 240 :=
by
  sorry

end NUMINAMATH_GPT_Ava_watch_minutes_l1450_145095


namespace NUMINAMATH_GPT_max_cigarettes_with_staggered_packing_l1450_145069

theorem max_cigarettes_with_staggered_packing :
  ∃ n : ℕ, n > 160 ∧ n = 176 :=
by
  let diameter := 2
  let rows_initial := 8
  let cols_initial := 20
  let total_initial := rows_initial * cols_initial
  have h1 : total_initial = 160 := by norm_num
  let alternative_packing_capacity := 176
  have h2 : alternative_packing_capacity > total_initial := by norm_num
  use alternative_packing_capacity
  exact ⟨h2, rfl⟩

end NUMINAMATH_GPT_max_cigarettes_with_staggered_packing_l1450_145069


namespace NUMINAMATH_GPT_apricot_tea_calories_l1450_145026

theorem apricot_tea_calories :
  let apricot_juice_weight := 150
  let apricot_juice_calories_per_100g := 30
  let honey_weight := 50
  let honey_calories_per_100g := 304
  let water_weight := 300
  let apricot_tea_weight := apricot_juice_weight + honey_weight + water_weight
  let apricot_juice_calories := apricot_juice_weight * apricot_juice_calories_per_100g / 100
  let honey_calories := honey_weight * honey_calories_per_100g / 100
  let total_calories := apricot_juice_calories + honey_calories
  let caloric_density := total_calories / apricot_tea_weight
  let tea_weight := 250
  let calories_in_250g_tea := tea_weight * caloric_density
  calories_in_250g_tea = 98.5 := by
  sorry

end NUMINAMATH_GPT_apricot_tea_calories_l1450_145026


namespace NUMINAMATH_GPT_evaluate_expression_l1450_145061

theorem evaluate_expression : (164^2 - 148^2) / 16 = 312 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1450_145061


namespace NUMINAMATH_GPT_required_run_rate_is_correct_l1450_145008

open Nat

noncomputable def requiredRunRate (initialRunRate : ℝ) (initialOvers : ℕ) (targetRuns : ℕ) (totalOvers : ℕ) : ℝ :=
  let runsScored := initialRunRate * initialOvers
  let runsNeeded := targetRuns - runsScored
  let remainingOvers := totalOvers - initialOvers
  runsNeeded / (remainingOvers : ℝ)

theorem required_run_rate_is_correct :
  (requiredRunRate 3.6 10 282 50 = 6.15) :=
by
  sorry

end NUMINAMATH_GPT_required_run_rate_is_correct_l1450_145008


namespace NUMINAMATH_GPT_cube_face_parallel_probability_l1450_145092

theorem cube_face_parallel_probability :
  ∃ (n m : ℕ), (n = 15) ∧ (m = 3) ∧ (m / n = (1 / 5 : ℝ)) := 
sorry

end NUMINAMATH_GPT_cube_face_parallel_probability_l1450_145092


namespace NUMINAMATH_GPT_inscribed_circle_radius_l1450_145078

noncomputable def radius_inscribed_circle (DE DF EF : ℝ) : ℝ := 
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem inscribed_circle_radius :
  radius_inscribed_circle 8 5 9 = 6 * Real.sqrt 11 / 11 :=
by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_l1450_145078


namespace NUMINAMATH_GPT_angle_measure_l1450_145042

theorem angle_measure (x : ℝ) (h1 : (180 - x) = 3*x - 2) : x = 45.5 :=
by
  sorry

end NUMINAMATH_GPT_angle_measure_l1450_145042


namespace NUMINAMATH_GPT_triangle_perimeter_l1450_145080

theorem triangle_perimeter (a b c : ℝ) (h1 : a = 2) (h2 : (b-2)^2 + |c-3| = 0) : a + b + c = 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l1450_145080


namespace NUMINAMATH_GPT_mean_visits_between_200_and_300_l1450_145077

def monday_visits := 300
def tuesday_visits := 400
def wednesday_visits := 300
def thursday_visits := 200
def friday_visits := 200

def total_visits := monday_visits + tuesday_visits + wednesday_visits + thursday_visits + friday_visits
def number_of_days := 5
def mean_visits_per_day := total_visits / number_of_days

theorem mean_visits_between_200_and_300 : 200 ≤ mean_visits_per_day ∧ mean_visits_per_day ≤ 300 :=
by sorry

end NUMINAMATH_GPT_mean_visits_between_200_and_300_l1450_145077


namespace NUMINAMATH_GPT_solve_for_a_b_and_extrema_l1450_145028

noncomputable def f (a b x : ℝ) := -2 * a * Real.sin (2 * x + (Real.pi / 6)) + 2 * a + b

theorem solve_for_a_b_and_extrema:
  ∃ (a b : ℝ), a > 0 ∧ 
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 2), -5 ≤ f a b x ∧ f a b x ≤ 1) ∧ 
  a = 2 ∧ b = -5 ∧
  (∀ x ∈ Set.Icc (0:ℝ) (Real.pi / 4),
    (f a b (Real.pi / 6) = -5 ∨ f a b 0 = -3)) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_b_and_extrema_l1450_145028


namespace NUMINAMATH_GPT_number_of_games_X_l1450_145043

variable (x : ℕ) -- Total number of games played by team X
variable (y : ℕ) -- Wins by team Y
variable (ly : ℕ) -- Losses by team Y
variable (dy : ℕ) -- Draws by team Y
variable (wx : ℕ) -- Wins by team X
variable (lx : ℕ) -- Losses by team X
variable (dx : ℕ) -- Draws by team X

axiom wins_ratio_X : wx = 3 * x / 4
axiom wins_ratio_Y : y = 2 * (x + 12) / 3
axiom wins_difference : y = wx + 4
axiom losses_difference : ly = lx + 5
axiom draws_difference : dy = dx + 3
axiom eq_losses_draws : lx + dx = (x - wx)

theorem number_of_games_X : x = 48 :=
by
  sorry

end NUMINAMATH_GPT_number_of_games_X_l1450_145043


namespace NUMINAMATH_GPT_best_fitting_model_is_model3_l1450_145089

-- Definitions of the coefficients of determination for the models
def R2_model1 : ℝ := 0.60
def R2_model2 : ℝ := 0.90
def R2_model3 : ℝ := 0.98
def R2_model4 : ℝ := 0.25

-- The best fitting effect corresponds to the highest R^2 value
theorem best_fitting_model_is_model3 :
  R2_model3 = max (max R2_model1 R2_model2) (max R2_model3 R2_model4) :=
by {
  -- Proofblock is skipped, using sorry
  sorry
}

end NUMINAMATH_GPT_best_fitting_model_is_model3_l1450_145089


namespace NUMINAMATH_GPT_average_speed_v2_l1450_145099

theorem average_speed_v2 (v1 : ℝ) (t : ℝ) (S1 : ℝ) (S2 : ℝ) : 
  (v1 = 30) → (t = 30) → (S1 = 800) → (S2 = 200) → 
  (v2 = (v1 - (S1 - S2) / t) ∨ v2 = (v1 + (S1 - S2) / t)) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_average_speed_v2_l1450_145099


namespace NUMINAMATH_GPT_intersection_complement_l1450_145031

open Set

def U : Set ℝ := univ
def A : Set ℤ := {x : ℤ | -3 < x ∧ x ≤ 1}
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement (U A B : Set ℝ) : A ∩ (U \ B) = {0, 1} := sorry

end NUMINAMATH_GPT_intersection_complement_l1450_145031


namespace NUMINAMATH_GPT_distribution_difference_l1450_145001

theorem distribution_difference 
  (total_amnt : ℕ)
  (p_amnt : ℕ) 
  (q_amnt : ℕ) 
  (r_amnt : ℕ)
  (s_amnt : ℕ)
  (h_total : total_amnt = 1000)
  (h_p : p_amnt = 2 * q_amnt)
  (h_s : s_amnt = 4 * r_amnt)
  (h_qr : q_amnt = r_amnt) :
  s_amnt - p_amnt = 250 := 
sorry

end NUMINAMATH_GPT_distribution_difference_l1450_145001


namespace NUMINAMATH_GPT_proof_l1450_145079

noncomputable def question := ∀ x : ℝ, (0.12 * x = 36) → (0.5 * (0.4 * 0.3 * x) = 18) 

theorem proof : question :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_proof_l1450_145079


namespace NUMINAMATH_GPT_Marnie_can_make_9_bracelets_l1450_145004

def number_of_beads : Nat :=
  (5 * 50) + (2 * 100)

def beads_per_bracelet : Nat := 50

def total_bracelets (total_beads : Nat) (beads_per_bracelet : Nat) : Nat :=
  total_beads / beads_per_bracelet

theorem Marnie_can_make_9_bracelets :
  total_bracelets number_of_beads beads_per_bracelet = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Marnie_can_make_9_bracelets_l1450_145004
