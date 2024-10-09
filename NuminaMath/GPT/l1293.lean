import Mathlib

namespace division_result_l1293_129338

theorem division_result : (108 * 3 - (108 + 92)) / (92 * 7 - (45 * 3)) = 124 / 509 := 
by
  sorry

end division_result_l1293_129338


namespace tax_rate_calculation_l1293_129337

theorem tax_rate_calculation (price_before_tax total_price : ℝ) 
  (h_price_before_tax : price_before_tax = 92) 
  (h_total_price : total_price = 98.90) : 
  (total_price - price_before_tax) / price_before_tax * 100 = 7.5 := 
by 
  -- Proof will be provided here.
  sorry

end tax_rate_calculation_l1293_129337


namespace area_of_circle_l1293_129358

noncomputable def point : Type := ℝ × ℝ

def A : point := (8, 15)
def B : point := (14, 9)

def is_on_circle (P : point) (r : ℝ) (C : point) : Prop :=
  (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = r ^ 2

def tangent_intersects_x_axis (tangent_point : point) (circle_center : point) : Prop :=
  ∃ x : ℝ, ∃ C : point, C.2 = 0 ∧ tangent_point = C ∧ circle_center = (x, 0)

theorem area_of_circle :
  ∃ C : point, ∃ r : ℝ,
    is_on_circle A r C ∧ 
    is_on_circle B r C ∧ 
    tangent_intersects_x_axis A C ∧ 
    tangent_intersects_x_axis B C ∧ 
    (↑(π * r ^ 2) = (117 * π) / 8) :=
sorry

end area_of_circle_l1293_129358


namespace age_difference_l1293_129332

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end age_difference_l1293_129332


namespace james_total_pay_l1293_129369

def original_prices : List ℝ := [15, 20, 25, 18, 22, 30]
def discounts : List ℝ := [0.30, 0.50, 0.40, 0.20, 0.45, 0.25]

def discounted_price (price discount : ℝ) : ℝ :=
  price * (1 - discount)

def total_price_after_discount (prices discounts : List ℝ) : ℝ :=
  (List.zipWith discounted_price prices discounts).sum

theorem james_total_pay :
  total_price_after_discount original_prices discounts = 84.50 :=
  by sorry

end james_total_pay_l1293_129369


namespace man_speed_is_correct_l1293_129344

noncomputable def train_length : ℝ := 165
noncomputable def train_speed_kmph : ℝ := 60
noncomputable def time_seconds : ℝ := 9

-- Function to convert speed from kmph to m/s
noncomputable def kmph_to_mps (speed_kmph: ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

-- Function to convert speed from m/s to kmph
noncomputable def mps_to_kmph (speed_mps: ℝ) : ℝ :=
  speed_mps * 3600 / 1000

-- The speed of the train in m/s
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph

-- The relative speed of the train with respect to the man in m/s
noncomputable def relative_speed_mps : ℝ := train_length / time_seconds

-- The speed of the man in m/s
noncomputable def man_speed_mps : ℝ := relative_speed_mps - train_speed_mps

-- The speed of the man in kmph
noncomputable def man_speed_kmph : ℝ := mps_to_kmph man_speed_mps

-- The statement to be proved
theorem man_speed_is_correct : man_speed_kmph = 5.976 := 
sorry

end man_speed_is_correct_l1293_129344


namespace part_1_part_2_l1293_129320

noncomputable def prob_pass_no_fee : ℚ :=
  (3 / 4) * (2 / 3) +
  (1 / 4) * (3 / 4) * (2 / 3) +
  (3 / 4) * (1 / 3) * (2 / 3) +
  (1 / 4) * (3 / 4) * (1 / 3) * (2 / 3)

noncomputable def prob_pass_200_fee : ℚ :=
  (1 / 4) * (1 / 4) * (3 / 4) * ((2 / 3) + (1 / 3) * (2 / 3)) +
  (1 / 3) * (1 / 3) * (2 / 3) * ((3 / 4) + (1 / 4) * (3 / 4))

theorem part_1 : prob_pass_no_fee = 5 / 6 := by
  sorry

theorem part_2 : prob_pass_200_fee = 1 / 9 := by
  sorry

end part_1_part_2_l1293_129320


namespace bags_bought_l1293_129370

theorem bags_bought (initial_bags : ℕ) (bags_given : ℕ) (final_bags : ℕ) (bags_bought : ℕ) :
  initial_bags = 20 → 
  bags_given = 4 → 
  final_bags = 22 → 
  bags_bought = final_bags - (initial_bags - bags_given) → 
  bags_bought = 6 := 
by
  intros h_initial h_given h_final h_buy
  rw [h_initial, h_given, h_final] at h_buy
  exact h_buy

#check bags_bought

end bags_bought_l1293_129370


namespace john_money_left_l1293_129313

-- Definitions for initial conditions
def initial_amount : ℤ := 100
def cost_roast : ℤ := 17
def cost_vegetables : ℤ := 11

-- Total spent calculation
def total_spent : ℤ := cost_roast + cost_vegetables

-- Remaining money calculation
def remaining_money : ℤ := initial_amount - total_spent

-- Theorem stating that John has €72 left
theorem john_money_left : remaining_money = 72 := by
  sorry

end john_money_left_l1293_129313


namespace calc_result_l1293_129397

theorem calc_result (a : ℤ) : 3 * a - 5 * a + a = -a := by
  sorry

end calc_result_l1293_129397


namespace divisibility_of_difference_by_9_l1293_129394

theorem divisibility_of_difference_by_9 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) :
  9 ∣ ((10 * a + b) - (10 * b + a)) :=
by {
  -- The problem statement
  sorry
}

end divisibility_of_difference_by_9_l1293_129394


namespace find_total_income_l1293_129333

theorem find_total_income (I : ℝ) (H : (0.27 * I = 35000)) : I = 129629.63 :=
by
  sorry

end find_total_income_l1293_129333


namespace maria_total_cost_l1293_129331

def price_pencil: ℕ := 8
def price_pen: ℕ := price_pencil / 2
def total_price: ℕ := price_pencil + price_pen

theorem maria_total_cost: total_price = 12 := by
  sorry

end maria_total_cost_l1293_129331


namespace probability_of_drawing_two_black_two_white_l1293_129362

noncomputable def probability_two_black_two_white : ℚ :=
  let total_ways := (Nat.choose 18 4)
  let ways_black := (Nat.choose 10 2)
  let ways_white := (Nat.choose 8 2)
  let favorable_ways := ways_black * ways_white
  favorable_ways / total_ways

theorem probability_of_drawing_two_black_two_white :
  probability_two_black_two_white = 7 / 17 := sorry

end probability_of_drawing_two_black_two_white_l1293_129362


namespace sum_of_fourth_powers_of_solutions_l1293_129367

theorem sum_of_fourth_powers_of_solutions (x y : ℝ)
  (h : |x^2 - 2 * x + 1/1004| = 1/1004 ∨ |y^2 - 2 * y + 1/1004| = 1/1004) :
  x^4 + y^4 = 20160427280144 / 12600263001 :=
sorry

end sum_of_fourth_powers_of_solutions_l1293_129367


namespace sum_of_first_nine_terms_l1293_129355

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def sum_of_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = (n / 2) * (a 0 + a (n - 1))

theorem sum_of_first_nine_terms 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_of_terms a S)
  (h_sum_terms : a 2 + a 3 + a 4 + a 5 + a 6 = 20) :
  S 9 = 36 :=
sorry

end sum_of_first_nine_terms_l1293_129355


namespace percentage_increase_of_numerator_l1293_129380

theorem percentage_increase_of_numerator (N D : ℝ) (P : ℝ) (h1 : N / D = 0.75)
  (h2 : (N + (P / 100) * N) / (D - (8 / 100) * D) = 15 / 16) :
  P = 15 :=
sorry

end percentage_increase_of_numerator_l1293_129380


namespace power_mod_l1293_129346

theorem power_mod (n : ℕ) : (3 ^ 2017) % 17 = 3 := 
by
  sorry

end power_mod_l1293_129346


namespace money_left_after_shopping_l1293_129392

def initial_budget : ℝ := 999.00
def shoes_price : ℝ := 165.00
def yoga_mat_price : ℝ := 85.00
def sports_watch_price : ℝ := 215.00
def hand_weights_price : ℝ := 60.00
def sales_tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.10

def total_cost_before_discount : ℝ :=
  shoes_price + yoga_mat_price + sports_watch_price + hand_weights_price

def discount_on_watch : ℝ := sports_watch_price * discount_rate

def discounted_watch_price : ℝ := sports_watch_price - discount_on_watch

def total_cost_after_discount : ℝ :=
  shoes_price + yoga_mat_price + discounted_watch_price + hand_weights_price

def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

def total_cost_including_tax : ℝ := total_cost_after_discount + sales_tax

def money_left : ℝ := initial_budget - total_cost_including_tax

theorem money_left_after_shopping : 
  money_left = 460.25 :=
by
  sorry

end money_left_after_shopping_l1293_129392


namespace correct_statement_C_l1293_129371

theorem correct_statement_C
  (a : ℚ) : a < 0 → |a| = -a := 
by
  sorry

end correct_statement_C_l1293_129371


namespace effective_price_l1293_129316

-- Definitions based on conditions
def upfront_payment (C : ℝ) := 0.20 * C = 240
def cashback (C : ℝ) := 0.10 * C

-- Problem statement
theorem effective_price (C : ℝ) (h₁ : upfront_payment C) : C - cashback C = 1080 :=
by
  sorry

end effective_price_l1293_129316


namespace part1_part2_l1293_129377

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := -x^2 + a * x - 3

theorem part1 {a : ℝ} :
  (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
sorry

theorem part2 :
  ∀ x > 0, Real.log x > (1 / Real.exp x) - (2 / (Real.exp 1) * x) :=
sorry

end part1_part2_l1293_129377


namespace find_sum_of_integers_l1293_129322

theorem find_sum_of_integers (w x y z : ℤ)
  (h1 : w - x + y = 7)
  (h2 : x - y + z = 8)
  (h3 : y - z + w = 4)
  (h4 : z - w + x = 3) : w + x + y + z = 11 :=
by
  sorry

end find_sum_of_integers_l1293_129322


namespace Sam_has_38_dollars_l1293_129317

theorem Sam_has_38_dollars (total_money erica_money sam_money : ℕ) 
  (h1 : total_money = 91)
  (h2 : erica_money = 53) 
  (h3 : total_money = erica_money + sam_money) : 
  sam_money = 38 := 
by 
  sorry

end Sam_has_38_dollars_l1293_129317


namespace servant_cash_received_l1293_129335

theorem servant_cash_received (annual_cash : ℕ) (turban_price : ℕ) (served_months : ℕ) (total_months : ℕ) (cash_received : ℕ) :
  annual_cash = 90 → turban_price = 50 → served_months = 9 → total_months = 12 → 
  cash_received = (annual_cash + turban_price) * served_months / total_months - turban_price → 
  cash_received = 55 :=
by {
  intros;
  sorry
}

end servant_cash_received_l1293_129335


namespace sum_a4_a5_a6_l1293_129302

variable (a : ℕ → ℤ)
variable (d : ℤ)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 21)

theorem sum_a4_a5_a6 : a 4 + a 5 + a 6 = 63 := by
  sorry

end sum_a4_a5_a6_l1293_129302


namespace trapezoid_diagonals_l1293_129361

theorem trapezoid_diagonals {BC AD AB CD AC BD : ℝ} (h b1 b2 : ℝ) 
  (hBC : BC = b1) (hAD : AD = b2) (hAB : AB = h) (hCD : CD = h) 
  (hAC : AC^2 = AB^2 + BC^2) (hBD : BD^2 = CD^2 + AD^2) :
  BD^2 - AC^2 = b2^2 - b1^2 := 
by 
  -- proof is omitted
  sorry

end trapezoid_diagonals_l1293_129361


namespace necessary_but_not_sufficient_for_q_implies_range_of_a_l1293_129330

variable (a : ℝ)

def p (x : ℝ) := |4*x - 3| ≤ 1
def q (x : ℝ) := x^2 - (2*a+1)*x + a*(a+1) ≤ 0

theorem necessary_but_not_sufficient_for_q_implies_range_of_a :
  (∀ x : ℝ, q a x → p x) → (0 ≤ a ∧ a ≤ 1/2) :=
by
  sorry

end necessary_but_not_sufficient_for_q_implies_range_of_a_l1293_129330


namespace compare_doubling_l1293_129379

theorem compare_doubling (a b : ℝ) (h : a > b) : 2 * a > 2 * b :=
  sorry

end compare_doubling_l1293_129379


namespace number_of_elements_in_S_l1293_129382

def S : Set ℕ := { n : ℕ | ∃ k : ℕ, n > 1 ∧ (10^10 - 1) % n = 0 }

theorem number_of_elements_in_S (h1 : Nat.Prime 9091) :
  ∃ T : Finset ℕ, T.card = 127 ∧ ∀ n, n ∈ T ↔ n ∈ S :=
sorry

end number_of_elements_in_S_l1293_129382


namespace relationship_between_y1_y2_l1293_129375

theorem relationship_between_y1_y2 (y1 y2 : ℝ) :
    (y1 = -3 * 2 + 4 ∧ y2 = -3 * (-1) + 4) → y1 < y2 :=
by
  sorry

end relationship_between_y1_y2_l1293_129375


namespace part1_part2_l1293_129381

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^4 - 4 * x^3 + (3 + m) * x^2 - 12 * x + 12

theorem part1 (m : ℤ) : 
  (∀ x : ℝ, f x m - f (1 - x) m + 4 * x^3 = 0) ↔ (m = 8 ∨ m = 12) := 
sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f x m ≥ 0) ↔ (4 ≤ m) := 
sorry

end part1_part2_l1293_129381


namespace correct_rounded_result_l1293_129342

-- Definition of rounding to the nearest hundred
def rounded_to_nearest_hundred (n : ℕ) : ℕ :=
  if n % 100 < 50 then n / 100 * 100 else (n / 100 + 1) * 100

-- Given conditions
def sum : ℕ := 68 + 57

-- The theorem to prove
theorem correct_rounded_result : rounded_to_nearest_hundred sum = 100 :=
by
  -- Proof skipped
  sorry

end correct_rounded_result_l1293_129342


namespace find_point_B_l1293_129325

theorem find_point_B (A B : ℝ) (h1 : A = 2) (h2 : abs (B - A) = 5) : B = -3 ∨ B = 7 :=
by
  -- This is where the proof steps would go, but we can skip it with sorry.
  sorry

end find_point_B_l1293_129325


namespace negation_universal_proposition_l1293_129372

theorem negation_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) → ∃ x : ℝ, x^2 - 2 * x + 1 < 0 :=
by sorry

end negation_universal_proposition_l1293_129372


namespace union_example_l1293_129389

theorem union_example (P Q : Set ℕ) (hP : P = {1, 2, 3, 4}) (hQ : Q = {2, 4}) :
  P ∪ Q = {1, 2, 3, 4} :=
by
  sorry

end union_example_l1293_129389


namespace total_flower_petals_l1293_129388

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end total_flower_petals_l1293_129388


namespace sqrt_range_l1293_129307

theorem sqrt_range (x : ℝ) (h : 5 - x ≥ 0) : x ≤ 5 :=
sorry

end sqrt_range_l1293_129307


namespace symmetry_about_x2_symmetry_about_2_0_l1293_129318

-- Define the conditions and their respective conclusions.
theorem symmetry_about_x2 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) : 
  ∀ x, f (x) = f (4 - x) := 
sorry

theorem symmetry_about_2_0 (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = -f (3 + x)) : 
  ∀ x, f (x) = -f (4 - x) := 
sorry

end symmetry_about_x2_symmetry_about_2_0_l1293_129318


namespace minimum_value_is_8_l1293_129374

noncomputable def minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_is_8 :
  ∃ (x y : ℝ) (hx : 0 < x) (hy : 0 < y), minimum_value x y hx hy = 8 :=
by
  sorry

end minimum_value_is_8_l1293_129374


namespace initial_number_of_numbers_is_five_l1293_129321

-- Define the conditions and the given problem
theorem initial_number_of_numbers_is_five
  (n : ℕ) (S : ℕ)
  (h1 : S / n = 27)
  (h2 : (S - 35) / (n - 1) = 25) : n = 5 :=
by
  sorry

end initial_number_of_numbers_is_five_l1293_129321


namespace firstGradeMuffins_l1293_129384

-- Define the conditions as the number of muffins baked by each class
def mrsBrierMuffins : ℕ := 18
def mrsMacAdamsMuffins : ℕ := 20
def mrsFlanneryMuffins : ℕ := 17

-- Define the total number of muffins baked
def totalMuffins : ℕ := mrsBrierMuffins + mrsMacAdamsMuffins + mrsFlanneryMuffins

-- Prove that the total number of muffins baked is 55
theorem firstGradeMuffins : totalMuffins = 55 := by
  sorry

end firstGradeMuffins_l1293_129384


namespace exists_n_l1293_129363

def F_n (a n : ℕ) : ℕ :=
  let q := a ^ (1 / n)
  let r := a % n
  q + r

noncomputable def largest_A : ℕ :=
  53590

theorem exists_n (a : ℕ) (h : a ≤ largest_A) :
  ∃ n1 n2 n3 n4 n5 n6 : ℕ, 
    F_n (F_n (F_n (F_n (F_n (F_n a n1) n2) n3) n4) n5) n6 = 1 := 
sorry

end exists_n_l1293_129363


namespace units_digit_k_squared_plus_2_k_is_7_l1293_129373

def k : ℕ := 2012^2 + 2^2012

theorem units_digit_k_squared_plus_2_k_is_7 : (k^2 + 2^k) % 10 = 7 :=
by sorry

end units_digit_k_squared_plus_2_k_is_7_l1293_129373


namespace total_candies_is_829_l1293_129306

-- Conditions as definitions
def Adam : ℕ := 6
def James : ℕ := 3 * Adam
def Rubert : ℕ := 4 * James
def Lisa : ℕ := 2 * Rubert
def Chris : ℕ := Lisa + 5
def Emily : ℕ := 3 * Chris - 7

-- Total candies
def total_candies : ℕ := Adam + James + Rubert + Lisa + Chris + Emily

-- Theorem to prove
theorem total_candies_is_829 : total_candies = 829 :=
by
  -- skipping the proof
  sorry

end total_candies_is_829_l1293_129306


namespace percentage_value_l1293_129368

theorem percentage_value (M : ℝ) (h : (25 / 100) * M = (55 / 100) * 1500) : M = 3300 :=
by
  sorry

end percentage_value_l1293_129368


namespace scientific_notation_to_standard_form_l1293_129383

theorem scientific_notation_to_standard_form :
  - 3.96 * 10^5 = -396000 :=
sorry

end scientific_notation_to_standard_form_l1293_129383


namespace inequality_ab_bc_ca_l1293_129378

open Real

theorem inequality_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b / (a + b) + b * c / (b + c) + c * a / (c + a)) ≤ (3 * (a * b + b * c + c * a) / (2 * (a + b + c))) := by
sorry

end inequality_ab_bc_ca_l1293_129378


namespace find_prime_b_l1293_129345

-- Define the polynomial function f
def f (n a : ℕ) : ℕ := n^3 - 4 * a * n^2 - 12 * n + 144

-- Define b as a prime number
def b (n : ℕ) (a : ℕ) : ℕ := f n a

-- Theorem statement
theorem find_prime_b (n : ℕ) (a : ℕ) (h : n = 7) (ha : a = 2) (hb : ∃ p : ℕ, Nat.Prime p ∧ p = b n a) :
  b n a = 11 :=
by
  sorry

end find_prime_b_l1293_129345


namespace fraction_power_equality_l1293_129354

theorem fraction_power_equality :
  (72000 ^ 4) / (24000 ^ 4) = 81 := 
by
  sorry

end fraction_power_equality_l1293_129354


namespace sandy_spent_on_shorts_l1293_129301

variable (amount_on_shirt amount_on_jacket total_amount amount_on_shorts : ℝ)

theorem sandy_spent_on_shorts :
  amount_on_shirt = 12.14 →
  amount_on_jacket = 7.43 →
  total_amount = 33.56 →
  amount_on_shorts = total_amount - amount_on_shirt - amount_on_jacket →
  amount_on_shorts = 13.99 :=
by
  intros h_shirt h_jacket h_total h_computation
  sorry

end sandy_spent_on_shorts_l1293_129301


namespace more_bottles_of_regular_soda_l1293_129326

theorem more_bottles_of_regular_soda (reg_soda diet_soda : ℕ) (h1 : reg_soda = 79) (h2 : diet_soda = 53) :
  reg_soda - diet_soda = 26 :=
by
  sorry

end more_bottles_of_regular_soda_l1293_129326


namespace farm_problem_l1293_129348

variable (H R : ℕ)

-- Conditions
def initial_relation : Prop := R = H + 6
def hens_updated : Prop := H + 8 = 20
def current_roosters (H R : ℕ) : ℕ := R + 4

-- Theorem statement
theorem farm_problem (H R : ℕ)
  (h1 : initial_relation H R)
  (h2 : hens_updated H) :
  current_roosters H R = 22 :=
by
  sorry

end farm_problem_l1293_129348


namespace reciprocal_of_minus_one_half_l1293_129310

theorem reciprocal_of_minus_one_half : (1 / (-1 / 2)) = -2 := 
by sorry

end reciprocal_of_minus_one_half_l1293_129310


namespace find_percentage_l1293_129319

theorem find_percentage (P : ℝ) : 
  (P / 100) * 700 = 210 ↔ P = 30 := by
  sorry

end find_percentage_l1293_129319


namespace molecular_weight_bleach_l1293_129343

theorem molecular_weight_bleach :
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  molecular_weight = 74.44
:=
by
  let Na := 22.99
  let O := 16.00
  let Cl := 35.45
  let molecular_weight := Na + O + Cl
  sorry

end molecular_weight_bleach_l1293_129343


namespace type_b_quantity_l1293_129386

theorem type_b_quantity 
  (x : ℕ)
  (hx : x + 2 * x + 4 * x = 140) : 
  2 * x = 40 := 
sorry

end type_b_quantity_l1293_129386


namespace greatest_integer_value_l1293_129315

theorem greatest_integer_value (x : ℤ) (h : ∃ x : ℤ, x = 29 ∧ ∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*(x+6) + 26)) :
  (∀ x : ℤ, (x ≠ 3 → ∃ k : ℤ, (x^2 + 3*x + 8) = (x-3)*k + 26) → x = 29) :=
by
  sorry

end greatest_integer_value_l1293_129315


namespace candy_cost_l1293_129360

theorem candy_cost (tickets_whack_a_mole : ℕ) (tickets_skee_ball : ℕ) 
  (total_tickets : ℕ) (candies : ℕ) (cost_per_candy : ℕ) 
  (h1 : tickets_whack_a_mole = 8) (h2 : tickets_skee_ball = 7)
  (h3 : total_tickets = tickets_whack_a_mole + tickets_skee_ball)
  (h4 : candies = 3) (h5 : total_tickets = candies * cost_per_candy) :
  cost_per_candy = 5 :=
by
  sorry

end candy_cost_l1293_129360


namespace determine_abcd_l1293_129339

-- Define a 4-digit natural number abcd in terms of its digits a, b, c, d
def four_digit_number (abcd a b c d : ℕ) :=
  abcd = 1000 * a + 100 * b + 10 * c + d

-- Define the condition given in the problem
def satisfies_condition (abcd a b c d : ℕ) :=
  abcd - (100 * a + 10 * b + c) - (10 * a + b) - a = 1995

-- Define the main theorem statement proving the number is 2243
theorem determine_abcd : ∃ (a b c d abcd : ℕ), four_digit_number abcd a b c d ∧ satisfies_condition abcd a b c d ∧ abcd = 2243 :=
by
  sorry

end determine_abcd_l1293_129339


namespace value_of_v_3_l1293_129349

-- Defining the polynomial
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Given evaluation point
def eval_point : ℝ := -2

-- Horner's method intermediate value v_3
def v_3_using_horner_method (x : ℝ) : ℝ :=
  let V0 := 1
  let V1 := x * V0 - 5
  let V2 := x * V1 + 6
  let V3 := x * V2 -- x^3 term is zero
  V3

-- Statement to prove
theorem value_of_v_3 :
  v_3_using_horner_method eval_point = -40 :=
by 
  -- Proof to be completed later
  sorry

end value_of_v_3_l1293_129349


namespace donovan_lap_time_l1293_129304

-- Definitions based on problem conditions
def lap_time_michael := 40  -- Michael's lap time in seconds
def laps_michael := 9       -- Laps completed by Michael to pass Donovan
def laps_donovan := 8       -- Laps completed by Donovan in the same time

-- Condition based on the solution
def race_duration := laps_michael * lap_time_michael

-- define the conjecture
theorem donovan_lap_time : 
  (race_duration = laps_donovan * 45) := 
sorry

end donovan_lap_time_l1293_129304


namespace instantaneous_velocity_at_2_l1293_129300

def displacement (t : ℝ) : ℝ := 2 * t^2 + 3

theorem instantaneous_velocity_at_2 : (deriv displacement 2) = 8 :=
by 
  -- Proof would go here
  sorry

end instantaneous_velocity_at_2_l1293_129300


namespace manny_problem_l1293_129376

noncomputable def num_slices_left (num_pies : Nat) (slices_per_pie : Nat) (num_classmates : Nat) (num_teachers : Nat) (num_slices_per_person : Nat) : Nat :=
  let total_slices := num_pies * slices_per_pie
  let total_people := 1 + num_classmates + num_teachers
  let slices_taken := total_people * num_slices_per_person
  total_slices - slices_taken

theorem manny_problem : num_slices_left 3 10 24 1 1 = 4 := by
  sorry

end manny_problem_l1293_129376


namespace common_terms_count_l1293_129399

theorem common_terms_count (β : ℕ) (h1 : β = 55) (h2 : β + 1 = 56) : 
  ∃ γ : ℕ, γ = 6 :=
by
  sorry

end common_terms_count_l1293_129399


namespace industrial_lubricants_percentage_l1293_129341

theorem industrial_lubricants_percentage :
  let a := 12   -- percentage for microphotonics
  let b := 24   -- percentage for home electronics
  let c := 15   -- percentage for food additives
  let d := 29   -- percentage for genetically modified microorganisms
  let angle_basic_astrophysics := 43.2 -- degrees for basic astrophysics
  let total_angle := 360              -- total degrees in a circle
  let total_budget := 100             -- total budget in percentage
  let e := (angle_basic_astrophysics / total_angle) * total_budget -- percentage for basic astrophysics
  a + b + c + d + e = 92 → total_budget - (a + b + c + d + e) = 8 :=
by
  intros
  sorry

end industrial_lubricants_percentage_l1293_129341


namespace extreme_values_l1293_129347

noncomputable def f (x : ℝ) : ℝ := x + 4 / x

theorem extreme_values (x : ℝ) (hx : x ≠ 0) :
  (x = -2 → f x = -4 ∧ ∀ y, y > -2 → f y > -4) ∧
  (x = 2 → f x = 4 ∧ ∀ y, y < 2 → f y > 4) :=
sorry

end extreme_values_l1293_129347


namespace num_triangles_from_decagon_l1293_129328

-- Define the number of vertices in the regular decagon
def num_vertices : Nat := 10

-- Define the combination function
def combination (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- main statement to be proved
theorem num_triangles_from_decagon : combination num_vertices 3 = 120 := by
  sorry

end num_triangles_from_decagon_l1293_129328


namespace minimum_value_l1293_129365

open Real

theorem minimum_value (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) : 
  1/m + 4/n ≥ 9 :=
by
  sorry

end minimum_value_l1293_129365


namespace DanielCandies_l1293_129329

noncomputable def initialCandies (x : ℝ) : Prop :=
  (3 / 8) * x - (3 / 2) - 16 = 10

theorem DanielCandies : ∃ x : ℝ, initialCandies x ∧ x = 93 :=
by
  use 93
  simp [initialCandies]
  norm_num
  sorry

end DanielCandies_l1293_129329


namespace calculate_length_of_train_l1293_129395

noncomputable def length_of_train (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := (relative_speed_kmh : ℝ) * 1000 / 3600
  relative_speed_ms * time_seconds

theorem calculate_length_of_train :
  length_of_train 50 5 7.2 = 110 := by
  -- This is where the actual proof would go, but it's omitted for now as per instructions.
  sorry

end calculate_length_of_train_l1293_129395


namespace exists_equal_subinterval_l1293_129396

open Set Metric Function

variable {a b : ℝ}
variable {f : ℕ → ℝ → ℝ}
variable {n m : ℕ}

-- Define the conditions
def continuous_on_interval (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ n, ContinuousOn (f n) (Icc a b)

def root_cond (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ x ∈ Icc a b, ∃ m n, m ≠ n ∧ f m x = f n x

-- The main theorem statement
theorem exists_equal_subinterval (f : ℕ → ℝ → ℝ) (a b : ℝ) 
  (h_cont : continuous_on_interval f a b) 
  (h_root : root_cond f a b) : 
  ∃ (c d : ℝ), c < d ∧ Icc c d ⊆ Icc a b ∧ ∃ m n, m ≠ n ∧ ∀ x ∈ Icc c d, f m x = f n x := 
sorry

end exists_equal_subinterval_l1293_129396


namespace five_digit_number_count_l1293_129314

theorem five_digit_number_count : ∃ n, n = 1134 ∧ ∀ (a b c d e : ℕ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧ 
  (a < b ∧ b < c ∧ c > d ∧ d > e) → n = 1134 :=
by 
  sorry

end five_digit_number_count_l1293_129314


namespace jars_needed_l1293_129336

def hives : ℕ := 5
def honey_per_hive : ℕ := 20
def jar_capacity : ℝ := 0.5
def friend_ratio : ℝ := 0.5

theorem jars_needed : (hives * honey_per_hive) / 2 / jar_capacity = 100 := 
by sorry

end jars_needed_l1293_129336


namespace cost_of_notebook_l1293_129324

theorem cost_of_notebook (num_students : ℕ) (more_than_half_bought : ℕ) (num_notebooks : ℕ) 
                         (cost_per_notebook : ℕ) (total_cost : ℕ) 
                         (half_students : more_than_half_bought > 18) 
                         (more_than_one_notebook : num_notebooks > 1) 
                         (cost_gt_notebooks : cost_per_notebook > num_notebooks) 
                         (calc_total_cost : more_than_half_bought * cost_per_notebook * num_notebooks = 2310) :
  cost_per_notebook = 11 := 
sorry

end cost_of_notebook_l1293_129324


namespace remainder_of_product_mod_7_l1293_129353

theorem remainder_of_product_mod_7
  (a b c : ℕ)
  (ha : a ≡ 2 [MOD 7])
  (hb : b ≡ 3 [MOD 7])
  (hc : c ≡ 4 [MOD 7]) :
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_mod_7_l1293_129353


namespace joans_remaining_kittens_l1293_129334

theorem joans_remaining_kittens (initial_kittens given_away : ℕ) (h1 : initial_kittens = 15) (h2 : given_away = 7) : initial_kittens - given_away = 8 := sorry

end joans_remaining_kittens_l1293_129334


namespace solve_cubed_root_equation_l1293_129308

theorem solve_cubed_root_equation :
  (∃ x : ℚ, (5 - 2 / x) ^ (1 / 3) = -3) ↔ x = 1 / 16 := 
by
  sorry

end solve_cubed_root_equation_l1293_129308


namespace sandy_red_marbles_l1293_129398

theorem sandy_red_marbles (jessica_marbles : ℕ) (sandy_marbles : ℕ) 
  (h₀ : jessica_marbles = 3 * 12)
  (h₁ : sandy_marbles = 4 * jessica_marbles) : 
  sandy_marbles = 144 :=
by
  sorry

end sandy_red_marbles_l1293_129398


namespace general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l1293_129350

-- Part 1: Finding the general term of the arithmetic sequence
theorem general_term_arithmetic_seq (a : ℕ → ℤ) (h1 : a 1 = 25) (h4 : a 4 = 16) :
  ∃ d : ℤ, a n = 28 - 3 * n := 
sorry

-- Part 2: Finding the value of n that maximizes the sum of the first n terms
theorem max_sum_of_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (h1 : a 1 = 25)
  (h4 : a 4 = 16) 
  (ha : ∀ n, a n = 28 - 3 * n) -- Using the result from part 1
  (h_sum : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n : ℕ, S n < S (n + 1)) →
  9 = 9 :=
sorry

end general_term_arithmetic_seq_max_sum_of_arithmetic_seq_l1293_129350


namespace cannot_form_triangle_l1293_129323

theorem cannot_form_triangle {a b c : ℝ} (h1 : a = 2) (h2 : b = 3) (h3 : c = 6) : 
  ¬ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :=
by
  sorry

end cannot_form_triangle_l1293_129323


namespace A_beats_B_by_14_meters_l1293_129327

theorem A_beats_B_by_14_meters :
  let distance := 70
  let time_A := 20
  let time_B := 25
  let speed_A := distance / time_A
  let speed_B := distance / time_B
  let distance_B_in_A_time := speed_B * time_A
  (distance - distance_B_in_A_time) = 14 :=
by
  sorry

end A_beats_B_by_14_meters_l1293_129327


namespace set_diff_example_l1293_129311

-- Definitions of sets A and B
def A : Set ℕ := {2, 3}
def B : Set ℕ := {1, 3, 4}

-- Definition of set difference
def set_diff (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- The mathematically equivalent proof problem statement
theorem set_diff_example :
  set_diff A B = {2} :=
sorry

end set_diff_example_l1293_129311


namespace largest_n_crates_same_orange_count_l1293_129356

theorem largest_n_crates_same_orange_count :
  ∀ (num_crates : ℕ) (min_oranges max_oranges : ℕ),
    num_crates = 200 →
    min_oranges = 100 →
    max_oranges = 130 →
    (∃ (n : ℕ), n = 7 ∧ (∃ (distribution : ℕ → ℕ), 
      (∀ x, min_oranges ≤ x ∧ x ≤ max_oranges) ∧ 
      (∀ x, distribution x ≤ num_crates ∧ 
          ∃ y, distribution y ≥ n))) := sorry

end largest_n_crates_same_orange_count_l1293_129356


namespace evaluate_expression_l1293_129351

theorem evaluate_expression :
  10 - 10.5 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.3 :=
by 
  sorry

end evaluate_expression_l1293_129351


namespace inverse_function_l1293_129357

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

noncomputable def f_inv (y : ℝ) : ℝ := -2 - Real.sqrt ((1 + y) / 2)

theorem inverse_function :
  ∀ (x : ℝ), x < -2 → f_inv (f x) = x ∧ ∀ (y : ℝ), y > -1 → f (f_inv y) = y :=
by
  sorry

end inverse_function_l1293_129357


namespace rotated_D_coords_l1293_129309

-- Definitions of the points used in the problem
def point (x y : ℤ) : ℤ × ℤ := (x, y)

-- Definitions of the vertices of the triangle DEF
def D : ℤ × ℤ := point 2 (-3)
def E : ℤ × ℤ := point 2 0
def F : ℤ × ℤ := point 5 (-3)

-- Definition of the rotation center
def center : ℤ × ℤ := point 3 (-2)

-- Function to rotate a point (x, y) by 180 degrees around (h, k)
def rotate_180 (p c : ℤ × ℤ) : ℤ × ℤ := 
  let (x, y) := p
  let (h, k) := c
  (2 * h - x, 2 * k - y)

-- Statement to prove the required coordinates after rotation
theorem rotated_D_coords : rotate_180 D center = point 4 (-1) :=
  sorry

end rotated_D_coords_l1293_129309


namespace opposite_of_2023_is_neg2023_l1293_129390

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l1293_129390


namespace q_r_share_difference_l1293_129352

theorem q_r_share_difference
  (T : ℝ) -- Total amount of money
  (x : ℝ) -- Common multiple of shares
  (p_share q_share r_share s_share : ℝ) -- Shares before tax
  (p_tax q_tax r_tax s_tax : ℝ) -- Tax percentages
  (h_ratio : p_share = 3 * x ∧ q_share = 7 * x ∧ r_share = 12 * x ∧ s_share = 5 * x) -- Ratio condition
  (h_tax : p_tax = 0.10 ∧ q_tax = 0.15 ∧ r_tax = 0.20 ∧ s_tax = 0.25) -- Tax condition
  (h_difference_pq : q_share * (1 - q_tax) - p_share * (1 - p_tax) = 2400) -- Difference between p and q after tax
  : (r_share * (1 - r_tax) - q_share * (1 - q_tax)) = 2695.38 := sorry

end q_r_share_difference_l1293_129352


namespace couscous_problem_l1293_129387

def total_couscous (S1 S2 S3 : ℕ) : ℕ :=
  S1 + S2 + S3

def couscous_per_dish (total : ℕ) (dishes : ℕ) : ℕ :=
  total / dishes

theorem couscous_problem 
  (S1 S2 S3 : ℕ) (dishes : ℕ) 
  (h1 : S1 = 7) (h2 : S2 = 13) (h3 : S3 = 45) (h4 : dishes = 13) :
  couscous_per_dish (total_couscous S1 S2 S3) dishes = 5 := by  
  sorry

end couscous_problem_l1293_129387


namespace find_m_l1293_129366

-- Define the condition for m to be within the specified range
def valid_range (m : ℤ) : Prop := -180 < m ∧ m < 180

-- Define the relationship with the trigonometric equation to be proven
def tan_eq (m : ℤ) : Prop := Real.tan (m * Real.pi / 180) = Real.tan (1500 * Real.pi / 180)

-- State the main theorem to be proved
theorem find_m (m : ℤ) (h1 : valid_range m) (h2 : tan_eq m) : m = 60 :=
sorry

end find_m_l1293_129366


namespace journey_duration_l1293_129391

theorem journey_duration
  (distance : ℕ) (speed : ℕ) (h1 : distance = 48) (h2 : speed = 8) :
  distance / speed = 6 := 
by
  sorry

end journey_duration_l1293_129391


namespace points_per_member_correct_l1293_129312

noncomputable def points_per_member (total_members: ℝ) (absent_members: ℝ) (total_points: ℝ) :=
  (total_points / (total_members - absent_members))

theorem points_per_member_correct:
  points_per_member 5.0 2.0 6.0 = 2.0 :=
by 
  sorry

end points_per_member_correct_l1293_129312


namespace min_re_z4_re_z4_l1293_129393

theorem min_re_z4_re_z4 (z : ℂ) (h : z.re ≠ 0) : 
  ∃ t : ℝ, (t = (z.im / z.re)) ∧ ((1 - 6 * (t^2) + (t^4)) = -8) := sorry

end min_re_z4_re_z4_l1293_129393


namespace range_of_m_l1293_129359

theorem range_of_m (m : ℝ) :
  (∃ ρ θ : ℝ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 0 ∧
    (∃ ρ₀ θ₀ : ℝ, ∀ ρ θ, m * ρ * (Real.cos θ)^2 + 3 * ρ * (Real.sin θ)^2 - 6 * (Real.cos θ) = 
      m * ρ₀ * (Real.cos θ₀)^2 + 3 * ρ₀ * (Real.sin θ₀)^2 - 6 * (Real.cos θ₀))) →
  m > 0 ∧ m ≠ 3 := sorry

end range_of_m_l1293_129359


namespace width_of_box_is_correct_l1293_129385

noncomputable def length_of_box : ℝ := 62
noncomputable def height_lowered : ℝ := 0.5
noncomputable def volume_removed_in_gallons : ℝ := 5812.5
noncomputable def gallons_to_cubic_feet : ℝ := 1 / 7.48052

theorem width_of_box_is_correct :
  let volume_removed_in_cubic_feet := volume_removed_in_gallons * gallons_to_cubic_feet
  let area_of_base := length_of_box * W
  let needed_volume := area_of_base * height_lowered
  volume_removed_in_cubic_feet = needed_volume →
  W = 25.057 :=
by
  sorry

end width_of_box_is_correct_l1293_129385


namespace inverse_proportion_graph_l1293_129303

theorem inverse_proportion_graph (k : ℝ) (x : ℝ) (y : ℝ) (h1 : y = k / x) (h2 : (3, -4) ∈ {p : ℝ × ℝ | p.snd = k / p.fst}) :
  k < 0 → ∀ x1 x2 : ℝ, x1 < x2 → y1 = k / x1 → y2 = k / x2 → y1 < y2 := by
  sorry

end inverse_proportion_graph_l1293_129303


namespace max_food_cost_l1293_129305

theorem max_food_cost (total_cost : ℝ) (food_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_allowable : ℝ)
  (h1 : tax_rate = 0.07) (h2 : tip_rate = 0.15) (h3 : max_allowable = 75) (h4 : total_cost = food_cost * (1 + tax_rate + tip_rate)) :
  food_cost ≤ 61.48 :=
sorry

end max_food_cost_l1293_129305


namespace intersection_points_form_line_slope_l1293_129364

theorem intersection_points_form_line_slope (s : ℝ) :
  ∃ (m : ℝ), m = 1/18 ∧ ∀ (x y : ℝ),
    (3 * x + y = 5 * s + 6) ∧ (2 * x - 3 * y = 3 * s - 5) →
    ∃ k : ℝ, (y = m * x + k) :=
by
  sorry

end intersection_points_form_line_slope_l1293_129364


namespace line_through_intersection_points_of_circles_l1293_129340

theorem line_through_intersection_points_of_circles :
  (∀ x y : ℝ, x^2 + y^2 = 9 ∧ (x + 4)^2 + (y + 3)^2 = 8 → 4 * x + 3 * y + 13 = 0) :=
by
  sorry

end line_through_intersection_points_of_circles_l1293_129340
