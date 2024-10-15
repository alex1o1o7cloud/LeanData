import Mathlib

namespace NUMINAMATH_GPT_employee_price_l1514_151483

theorem employee_price (wholesale_cost retail_markup employee_discount : ℝ) 
    (h₁ : wholesale_cost = 200) 
    (h₂ : retail_markup = 0.20) 
    (h₃ : employee_discount = 0.25) : 
    (wholesale_cost * (1 + retail_markup)) * (1 - employee_discount) = 180 := 
by
  sorry

end NUMINAMATH_GPT_employee_price_l1514_151483


namespace NUMINAMATH_GPT_cube_root_sum_is_integer_l1514_151450

theorem cube_root_sum_is_integer :
  let a := (2 + (10 / 9) * Real.sqrt 3)^(1/3)
  let b := (2 - (10 / 9) * Real.sqrt 3)^(1/3)
  a + b = 2 := by
  sorry

end NUMINAMATH_GPT_cube_root_sum_is_integer_l1514_151450


namespace NUMINAMATH_GPT_tangent_intersection_locus_l1514_151445

theorem tangent_intersection_locus :
  ∀ (l : ℝ → ℝ) (C : ℝ → ℝ), 
  (∀ x > 0, C x = x + 1/x) →
  (∃ k : ℝ, ∀ x, l x = k * x + 1) →
  ∃ (P : ℝ × ℝ), (P = (2, 2)) ∨ (P = (2, 5/2)) :=
by sorry

end NUMINAMATH_GPT_tangent_intersection_locus_l1514_151445


namespace NUMINAMATH_GPT_wages_of_one_man_l1514_151494

variable (R : Type) [DivisionRing R] [DecidableEq R]
variable (money : R)
variable (num_men : ℕ := 5)
variable (num_women : ℕ := 8)
variable (total_wages : R := 180)
variable (wages_men : R := 36)

axiom equal_women : num_men = num_women
axiom total_earnings (wages : ℕ → R) :
  (wages num_men) + (wages num_women) + (wages 8) = total_wages

theorem wages_of_one_man :
  wages_men = total_wages / num_men := by
  sorry

end NUMINAMATH_GPT_wages_of_one_man_l1514_151494


namespace NUMINAMATH_GPT_smallest_k_square_divisible_l1514_151461

theorem smallest_k_square_divisible (k : ℤ) (n : ℤ) (h1 : k = 60)
    (h2 : ∀ m : ℤ, m < k → ∃ d : ℤ, d ∣ (k^2) → m = d ) : n = 3600 :=
sorry

end NUMINAMATH_GPT_smallest_k_square_divisible_l1514_151461


namespace NUMINAMATH_GPT_participants_count_l1514_151455

theorem participants_count (F M : ℕ)
  (hF2 : F / 2 = 110)
  (hM4 : M / 4 = 330 - F - M / 3)
  (hFm : (F + M) / 3 = F / 2 + M / 4) :
  F + M = 330 :=
sorry

end NUMINAMATH_GPT_participants_count_l1514_151455


namespace NUMINAMATH_GPT_problem_statement_l1514_151438

theorem problem_statement (h1 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2) :
  (Real.pi / (Real.sqrt 3 - 1))^0 - (Real.cos (Real.pi / 6))^2 = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l1514_151438


namespace NUMINAMATH_GPT_simplify_fraction_l1514_151414

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1514_151414


namespace NUMINAMATH_GPT_problem_solution_l1514_151404

noncomputable def solution_set : Set ℝ :=
  { x : ℝ | x ∈ (Set.Ioo 0 (5 - Real.sqrt 10)) ∨ x ∈ (Set.Ioi (5 + Real.sqrt 10)) }

theorem problem_solution (x : ℝ) : (x^3 - 10*x^2 + 15*x > 0) ↔ x ∈ solution_set :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1514_151404


namespace NUMINAMATH_GPT_find_nsatisfy_l1514_151430

-- Define the function S(n) that denotes the sum of the digits of n
def S (n : ℕ) : ℕ := n.digits 10 |>.sum

-- State the main theorem
theorem find_nsatisfy {n : ℕ} : n = 2 * (S n)^2 → n = 50 ∨ n = 162 ∨ n = 392 ∨ n = 648 := 
sorry

end NUMINAMATH_GPT_find_nsatisfy_l1514_151430


namespace NUMINAMATH_GPT_intersection_complement_eq_l1514_151474

-- Definitions as per given conditions
def U : Set ℕ := { x | x > 0 ∧ x < 9 }
def A : Set ℕ := { 1, 2, 3, 4 }
def B : Set ℕ := { 3, 4, 5, 6 }

-- Complement of B with respect to U
def C_U_B : Set ℕ := U \ B

-- Statement of the theorem to be proved
theorem intersection_complement_eq : A ∩ C_U_B = { 1, 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_eq_l1514_151474


namespace NUMINAMATH_GPT_budget_equality_year_l1514_151422

theorem budget_equality_year :
  let budget_q_1990 := 540000
  let budget_v_1990 := 780000
  let annual_increase_q := 30000
  let annual_decrease_v := 10000

  let budget_q (n : ℕ) := budget_q_1990 + n * annual_increase_q
  let budget_v (n : ℕ) := budget_v_1990 - n * annual_decrease_v

  (∃ n : ℕ, budget_q n = budget_v n ∧ 1990 + n = 1996) :=
by
  sorry

end NUMINAMATH_GPT_budget_equality_year_l1514_151422


namespace NUMINAMATH_GPT_power_modulo_l1514_151431

theorem power_modulo (a b c n : ℕ) (h1 : a = 17) (h2 : b = 1999) (h3 : c = 29) (h4 : n = a^b % c) : 
  n = 17 := 
by
  -- Note: Additional assumptions and intermediate calculations could be provided as needed
  sorry

end NUMINAMATH_GPT_power_modulo_l1514_151431


namespace NUMINAMATH_GPT_minimum_value_of_a_l1514_151442

theorem minimum_value_of_a (x : ℝ) (a : ℝ) (hx : 0 ≤ x) (hx2 : x ≤ 20) (ha : 0 < a) (h : (20 - x) / 4 + a / 2 * Real.sqrt x ≥ 5) : 
  a ≥ Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_a_l1514_151442


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1514_151448

theorem geometric_sequence_sum (q : ℝ) (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_a1 : a 1 = 1)
  (h_sum : a 1 + a 3 + a 5 = 21) :
  a 2 + a 4 + a 6 = 42 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1514_151448


namespace NUMINAMATH_GPT_new_group_size_l1514_151405

theorem new_group_size (N : ℕ) (h1 : 20 < N) (h2 : N < 50) (h3 : (N - 5) % 6 = 0) (h4 : (N - 5) % 7 = 0) (h5 : (N % (N - 7)) = 7) : (N - 7).gcd (N) = 8 :=
by
  sorry

end NUMINAMATH_GPT_new_group_size_l1514_151405


namespace NUMINAMATH_GPT_balls_to_boxes_l1514_151432

theorem balls_to_boxes (balls boxes : ℕ) (h1 : balls = 5) (h2 : boxes = 3) :
  ∃ ways : ℕ, ways = 150 := by
  sorry

end NUMINAMATH_GPT_balls_to_boxes_l1514_151432


namespace NUMINAMATH_GPT_total_apples_l1514_151487

-- Definitions based on the problem conditions
def marin_apples : ℕ := 8
def david_apples : ℕ := (3 * marin_apples) / 4
def amanda_apples : ℕ := (3 * david_apples) / 2 + 2

-- The statement that we need to prove
theorem total_apples : marin_apples + david_apples + amanda_apples = 25 := by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_total_apples_l1514_151487


namespace NUMINAMATH_GPT_minimum_value_quot_l1514_151481

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem minimum_value_quot (a b : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_quot_l1514_151481


namespace NUMINAMATH_GPT_open_door_within_time_l1514_151424

-- Define the initial conditions
def device := ℕ → ℕ

-- Constraint: Each device has 5 toggle switches ("0" or "1") and a three-digit display.
def valid_configuration (d : device) (k : ℕ) : Prop :=
  d k < 32 ∧ d k <= 999

def system_configuration (A B : device) (k : ℕ) : Prop :=
  A k = B k

-- Constraint: The devices can be synchronized to display the same number simultaneously to open the door.
def open_door (A B : device) : Prop :=
  ∃ k, system_configuration A B k

-- The main theorem: Devices A and B can be synchronized within the given time constraints to open the door.
theorem open_door_within_time (A B : device) (notebook : ℕ) : 
  (∀ k, valid_configuration A k ∧ valid_configuration B k) →
  open_door A B :=
by sorry

end NUMINAMATH_GPT_open_door_within_time_l1514_151424


namespace NUMINAMATH_GPT_max_value_inequality_l1514_151421

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 5 * x + 3 * y < 90) : 
  x * y * (90 - 5 * x - 3 * y) ≤ 1800 := 
sorry

end NUMINAMATH_GPT_max_value_inequality_l1514_151421


namespace NUMINAMATH_GPT_Miss_Darlington_total_blueberries_l1514_151457

-- Conditions
def initial_basket := 20
def additional_baskets := 9

-- Definition and statement to be proved
theorem Miss_Darlington_total_blueberries :
  initial_basket + additional_baskets * initial_basket = 200 :=
by
  sorry

end NUMINAMATH_GPT_Miss_Darlington_total_blueberries_l1514_151457


namespace NUMINAMATH_GPT_correct_option_l1514_151446

def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k / x

theorem correct_option :
  inverse_proportion x y → 
  (y = x + 3 ∨ y = x / 3 ∨ y = 3 / (x ^ 2) ∨ y = 3 / x) → 
  y = 3 / x :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1514_151446


namespace NUMINAMATH_GPT_complement_union_l1514_151401

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {2, 3}

theorem complement_union (U : Set ℕ) (M : Set ℕ) (N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {1, 2, 4}) (hN : N = {2, 3}) :
  (U \ M) ∪ N = {0, 2, 3} :=
by
  rw [hU, hM, hN] -- Substitute U, M, N definitions
  sorry -- Proof omitted

end NUMINAMATH_GPT_complement_union_l1514_151401


namespace NUMINAMATH_GPT_initial_men_in_garrison_l1514_151444

variable (x : ℕ)

theorem initial_men_in_garrison (h1 : x * 65 = x * 50 + (x + 3000) * 20) : x = 2000 :=
  sorry

end NUMINAMATH_GPT_initial_men_in_garrison_l1514_151444


namespace NUMINAMATH_GPT_conic_section_focus_l1514_151495

theorem conic_section_focus {m : ℝ} (h_non_zero : m ≠ 0) (h_non_five : m ≠ 5)
  (h_focus : ∃ (x_focus y_focus : ℝ), (x_focus, y_focus) = (2, 0) 
  ∧ (x_focus = c ∧ x_focus^2 / 4 = 5 * (1 - c^2 / m))) : m = 9 := 
by
  sorry

end NUMINAMATH_GPT_conic_section_focus_l1514_151495


namespace NUMINAMATH_GPT_simplify_fraction_l1514_151463

theorem simplify_fraction : 
  (1 - (1 / 4)) / (1 - (1 / 3)) = 9 / 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1514_151463


namespace NUMINAMATH_GPT_cost_per_bundle_l1514_151464

-- Condition: each rose costs 500 won
def rose_price := 500

-- Condition: total number of roses
def total_roses := 200

-- Condition: number of bundles
def bundles := 25

-- Question: Prove the cost per bundle
theorem cost_per_bundle (rp : ℕ) (tr : ℕ) (b : ℕ) : rp = 500 → tr = 200 → b = 25 → (rp * tr) / b = 4000 :=
by
  intros h0 h1 h2
  sorry

end NUMINAMATH_GPT_cost_per_bundle_l1514_151464


namespace NUMINAMATH_GPT_ratio_of_doctors_to_lawyers_l1514_151453

variable (d l : ℕ) -- number of doctors and lawyers
variable (h1 : (40 * d + 55 * l) / (d + l) = 45) -- overall average age condition

theorem ratio_of_doctors_to_lawyers : d = 2 * l :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_doctors_to_lawyers_l1514_151453


namespace NUMINAMATH_GPT_mary_money_left_l1514_151412

variable (p : ℝ)

theorem mary_money_left :
  have cost_drinks := 3 * p
  have cost_medium_pizza := 2 * p
  have cost_large_pizza := 3 * p
  let total_cost := cost_drinks + cost_medium_pizza + cost_large_pizza
  30 - total_cost = 30 - 8 * p := by {
    sorry
  }

end NUMINAMATH_GPT_mary_money_left_l1514_151412


namespace NUMINAMATH_GPT_caleb_caught_trouts_l1514_151434

theorem caleb_caught_trouts (C : ℕ) (h1 : 3 * C = C + 4) : C = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_caleb_caught_trouts_l1514_151434


namespace NUMINAMATH_GPT_min_value_frac_inverse_l1514_151416

theorem min_value_frac_inverse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : 
  (1 / a + 1 / b) >= 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_inverse_l1514_151416


namespace NUMINAMATH_GPT_problem_l1514_151415

theorem problem 
  (a b A B : ℝ)
  (h : ∀ x : ℝ, 1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x) ≥ 0) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
by sorry

end NUMINAMATH_GPT_problem_l1514_151415


namespace NUMINAMATH_GPT_smallest_value_of_M_l1514_151440

theorem smallest_value_of_M :
  ∀ (a b c d e f g M : ℕ), a > 0 → b > 0 → c > 0 → d > 0 → e > 0 → f > 0 → g > 0 →
  a + b + c + d + e + f + g = 2024 →
  M = max (a + b) (max (b + c) (max (c + d) (max (d + e) (max (e + f) (f + g))))) →
  M = 338 :=
by
  intro a b c d e f g M ha hb hc hd he hf hg hsum hmax
  sorry

end NUMINAMATH_GPT_smallest_value_of_M_l1514_151440


namespace NUMINAMATH_GPT_reciprocal_inequality_reciprocal_inequality_opposite_l1514_151400

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : ab > 0) : (1 / a < 1 / b) := 
sorry

theorem reciprocal_inequality_opposite (a b : ℝ) (h1 : a > b) (h2 : ab < 0) : (1 / a > 1 / b) := 
sorry

end NUMINAMATH_GPT_reciprocal_inequality_reciprocal_inequality_opposite_l1514_151400


namespace NUMINAMATH_GPT_binomial_expansion_b_value_l1514_151433

theorem binomial_expansion_b_value (a b x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x ^ 2 + a^5 * x ^ 5) : b = 40 := 
sorry

end NUMINAMATH_GPT_binomial_expansion_b_value_l1514_151433


namespace NUMINAMATH_GPT_smart_charging_piles_growth_l1514_151429

-- Define the conditions
variables {x : ℝ}

-- First month charging piles
def first_month_piles : ℝ := 301

-- Third month charging piles
def third_month_piles : ℝ := 500

-- The theorem stating the relationship between the first and third month
theorem smart_charging_piles_growth : 
  first_month_piles * (1 + x) ^ 2 = third_month_piles :=
by
  sorry

end NUMINAMATH_GPT_smart_charging_piles_growth_l1514_151429


namespace NUMINAMATH_GPT_optimal_fruit_combination_l1514_151498

structure FruitPrices :=
  (price_2_apples : ℕ)
  (price_6_apples : ℕ)
  (price_12_apples : ℕ)
  (price_2_oranges : ℕ)
  (price_6_oranges : ℕ)
  (price_12_oranges : ℕ)

def minCostFruits : ℕ :=
  sorry

theorem optimal_fruit_combination (fp : FruitPrices) (total_fruits : ℕ)
  (mult_2_or_3 : total_fruits = 15) :
  fp.price_2_apples = 48 →
  fp.price_6_apples = 126 →
  fp.price_12_apples = 224 →
  fp.price_2_oranges = 60 →
  fp.price_6_oranges = 164 →
  fp.price_12_oranges = 300 →
  minCostFruits = 314 :=
by
  sorry

end NUMINAMATH_GPT_optimal_fruit_combination_l1514_151498


namespace NUMINAMATH_GPT_pats_and_mats_numbers_l1514_151406

theorem pats_and_mats_numbers (x y : ℕ) (hxy : x ≠ y) (hx_gt_hy : x > y) 
    (h_sum : (x + y) + (x - y) + x * y + (x / y) = 98) : x = 12 ∧ y = 6 :=
by
  sorry

end NUMINAMATH_GPT_pats_and_mats_numbers_l1514_151406


namespace NUMINAMATH_GPT_candy_ratio_l1514_151489

theorem candy_ratio 
  (tabitha_candy : ℕ)
  (stan_candy : ℕ)
  (julie_candy : ℕ)
  (carlos_candy : ℕ)
  (total_candy : ℕ)
  (h1 : tabitha_candy = 22)
  (h2 : stan_candy = 13)
  (h3 : julie_candy = tabitha_candy / 2)
  (h4 : total_candy = 72)
  (h5 : tabitha_candy + stan_candy + julie_candy + carlos_candy = total_candy) :
  carlos_candy / stan_candy = 2 :=
by
  sorry

end NUMINAMATH_GPT_candy_ratio_l1514_151489


namespace NUMINAMATH_GPT_value_of_b_l1514_151488

variable (a b c : ℕ)
variable (h_a_nonzero : a ≠ 0)
variable (h_a : a < 8)
variable (h_b : b < 8)
variable (h_c : c < 8)
variable (h_square : ∃ k, k^2 = a * 8^3 + 3 * 8^2 + b * 8 + c)

theorem value_of_b : b = 1 :=
by sorry

end NUMINAMATH_GPT_value_of_b_l1514_151488


namespace NUMINAMATH_GPT_chords_even_arcs_even_l1514_151428

theorem chords_even_arcs_even (N : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ N → ¬ ((k : ℤ) % 2 = 1)) : 
  N % 2 = 0 := 
sorry

end NUMINAMATH_GPT_chords_even_arcs_even_l1514_151428


namespace NUMINAMATH_GPT_divisibility_condition_l1514_151462

theorem divisibility_condition (M C D U A q1 q2 q3 r1 r2 r3 : ℕ)
  (h1 : 10 = A * q1 + r1)
  (h2 : 10 * r1 = A * q2 + r2)
  (h3 : 10 * r2 = A * q3 + r3) :
  (U + D * r1 + C * r2 + M * r3) % A = 0 ↔ (1000 * M + 100 * C + 10 * D + U) % A = 0 :=
sorry

end NUMINAMATH_GPT_divisibility_condition_l1514_151462


namespace NUMINAMATH_GPT_correct_calculation_is_7_88_l1514_151496

theorem correct_calculation_is_7_88 (x : ℝ) (h : x * 8 = 56) : (x / 8) + 7 = 7.88 :=
by
  have hx : x = 7 := by
    linarith [h]
  rw [hx]
  norm_num
  sorry

end NUMINAMATH_GPT_correct_calculation_is_7_88_l1514_151496


namespace NUMINAMATH_GPT_average_of_last_four_numbers_l1514_151482

theorem average_of_last_four_numbers
  (seven_avg : ℝ)
  (first_three_avg : ℝ)
  (seven_avg_is_62 : seven_avg = 62)
  (first_three_avg_is_58 : first_three_avg = 58) :
  (7 * seven_avg - 3 * first_three_avg) / 4 = 65 :=
by
  rw [seven_avg_is_62, first_three_avg_is_58]
  sorry

end NUMINAMATH_GPT_average_of_last_four_numbers_l1514_151482


namespace NUMINAMATH_GPT_Jose_age_correct_l1514_151492

variable (Jose Zack Inez : ℕ)

-- Define the conditions
axiom Inez_age : Inez = 15
axiom Zack_age : Zack = Inez + 3
axiom Jose_age : Jose = Zack - 4

-- The proof statement
theorem Jose_age_correct : Jose = 14 :=
by
  -- Proof will be filled in later
  sorry

end NUMINAMATH_GPT_Jose_age_correct_l1514_151492


namespace NUMINAMATH_GPT_total_tiles_l1514_151470

theorem total_tiles (n : ℕ) (h : 2 * n - 1 = 133) : n^2 = 4489 :=
by
  sorry

end NUMINAMATH_GPT_total_tiles_l1514_151470


namespace NUMINAMATH_GPT_cost_of_eight_CDs_l1514_151443

theorem cost_of_eight_CDs (cost_of_two_CDs : ℕ) (h : cost_of_two_CDs = 36) : 8 * (cost_of_two_CDs / 2) = 144 := by
  sorry

end NUMINAMATH_GPT_cost_of_eight_CDs_l1514_151443


namespace NUMINAMATH_GPT_y_intercept_of_line_eq_l1514_151491

theorem y_intercept_of_line_eq (x y : ℝ) (h : x + y - 1 = 0) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_eq_l1514_151491


namespace NUMINAMATH_GPT_no_triangle_formed_l1514_151459

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := 4 * x + 3 * y + 5 = 0
def line3 (m : ℝ) (x y : ℝ) := m * x - y - 1 = 0

theorem no_triangle_formed (m : ℝ) :
  (∀ x y, line1 x y → line3 m x y) ∨
  (∀ x y, line2 x y → line3 m x y) ∨
  (∃ x y, line1 x y ∧ line2 x y ∧ line3 m x y) ↔
  (m = -4/3 ∨ m = 2/3 ∨ m = 4/3) :=
sorry -- Proof to be provided

end NUMINAMATH_GPT_no_triangle_formed_l1514_151459


namespace NUMINAMATH_GPT_marriage_year_proof_l1514_151441

-- Definitions based on conditions
def marriage_year : ℕ := sorry
def child1_birth_year : ℕ := 1982
def child2_birth_year : ℕ := 1984
def reference_year : ℕ := 1986

-- Age calculations based on reference year
def age_in_1986 (birth_year : ℕ) : ℕ := reference_year - birth_year

-- Combined ages in the reference year
def combined_ages_in_1986 : ℕ := age_in_1986 child1_birth_year + age_in_1986 child2_birth_year

-- The main theorem to prove
theorem marriage_year_proof :
  combined_ages_in_1986 = reference_year - marriage_year →
  marriage_year = 1980 := by
  sorry

end NUMINAMATH_GPT_marriage_year_proof_l1514_151441


namespace NUMINAMATH_GPT_handshakes_correct_l1514_151437

-- Definitions based on conditions
def num_gremlins : ℕ := 25
def num_imps : ℕ := 20
def num_imps_shaking_hands_among_themselves : ℕ := num_imps / 2
def comb (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate the total handshakes
def total_handshakes : ℕ :=
  (comb num_gremlins 2) + -- Handshakes among gremlins
  (comb num_imps_shaking_hands_among_themselves 2) + -- Handshakes among half the imps
  (num_gremlins * num_imps) -- Handshakes between all gremlins and all imps

-- The theorem to be proved
theorem handshakes_correct : total_handshakes = 845 := by
  sorry

end NUMINAMATH_GPT_handshakes_correct_l1514_151437


namespace NUMINAMATH_GPT_total_baseball_fans_l1514_151407

variable (Y M R : ℕ)

open Nat

theorem total_baseball_fans (h1 : 3 * M = 2 * Y) 
    (h2 : 4 * R = 5 * M) 
    (h3 : M = 96) : Y + M + R = 360 := by
  sorry

end NUMINAMATH_GPT_total_baseball_fans_l1514_151407


namespace NUMINAMATH_GPT_find_mistaken_number_l1514_151473

theorem find_mistaken_number : 
  ∃! x : ℕ, (x ∈ {n : ℕ | n ≥ 10 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 0)} ∧ 
  (10 + 15 + 20 + 25 + 30 + 35 + 40 + 45 + 50 + 55 + 60 + 65 + 70 + 75 + 80 + 85 + 90 + 95) + 2 * x = 1035) :=
sorry

end NUMINAMATH_GPT_find_mistaken_number_l1514_151473


namespace NUMINAMATH_GPT_blue_marbles_difference_l1514_151417

theorem blue_marbles_difference  (a b : ℚ) 
  (h1 : 3 * a + 2 * b = 80)
  (h2 : 2 * a = b) :
  (7 * a - 3 * b) = 80 / 7 := by
  sorry

end NUMINAMATH_GPT_blue_marbles_difference_l1514_151417


namespace NUMINAMATH_GPT_initial_passengers_l1514_151435

theorem initial_passengers (P : ℝ) :
  (1/2 * (2/3 * P + 280) + 12 = 242) → P = 270 :=
by
  sorry

end NUMINAMATH_GPT_initial_passengers_l1514_151435


namespace NUMINAMATH_GPT_product_of_four_consecutive_naturals_is_square_l1514_151486

theorem product_of_four_consecutive_naturals_is_square (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2) := 
by
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_naturals_is_square_l1514_151486


namespace NUMINAMATH_GPT_max_x_plus_2y_l1514_151493

theorem max_x_plus_2y (x y : ℝ) 
  (h1 : 4 * x + 3 * y ≤ 9) 
  (h2 : 3 * x + 5 * y ≤ 15) : 
  x + 2 * y ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_x_plus_2y_l1514_151493


namespace NUMINAMATH_GPT_find_m_solve_inequality_l1514_151419

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := m - |x - 2|

theorem find_m (m : ℝ) : (∀ x : ℝ, m - |x| ≥ 0 ↔ x ∈ [-1, 1]) → m = 1 :=
by
  sorry

theorem solve_inequality (x : ℝ) : |x + 1| + |x - 2| > 4 * 1 ↔ x < -3 / 2 ∨ x > 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_solve_inequality_l1514_151419


namespace NUMINAMATH_GPT_angle_ABC_measure_l1514_151452

theorem angle_ABC_measure
  (CBD : ℝ)
  (ABC ABD : ℝ)
  (h1 : CBD = 90)
  (h2 : ABC + ABD + CBD = 270)
  (h3 : ABD = 100) : 
  ABC = 80 :=
by
  -- Given:
  -- CBD = 90
  -- ABC + ABD + CBD = 270
  -- ABD = 100
  sorry

end NUMINAMATH_GPT_angle_ABC_measure_l1514_151452


namespace NUMINAMATH_GPT_age_difference_l1514_151420

theorem age_difference (A B : ℕ) (h1 : B = 39) (h2 : A + 10 = 2 * (B - 10)) : A - B = 9 := by
  sorry

end NUMINAMATH_GPT_age_difference_l1514_151420


namespace NUMINAMATH_GPT_no_positive_integer_solution_exists_l1514_151460

theorem no_positive_integer_solution_exists :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * x^2 + 2 * x + 2 = y^2 :=
by
  -- The proof steps will go here.
  sorry

end NUMINAMATH_GPT_no_positive_integer_solution_exists_l1514_151460


namespace NUMINAMATH_GPT_num_students_l1514_151426

theorem num_students (x : ℕ) (h1 : ∃ z : ℕ, z = 10 * x + 6) (h2 : ∃ z : ℕ, z = 12 * x - 6) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_num_students_l1514_151426


namespace NUMINAMATH_GPT_intersection_of_A_and_B_eq_C_l1514_151436

noncomputable def A (x : ℝ) : Prop := x^2 - 4*x + 3 < 0
noncomputable def B (x : ℝ) : Prop := 2 - x > 0
noncomputable def A_inter_B (x : ℝ) : Prop := A x ∧ B x

theorem intersection_of_A_and_B_eq_C :
  {x : ℝ | A_inter_B x} = {x : ℝ | 1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_eq_C_l1514_151436


namespace NUMINAMATH_GPT_min_value_of_parabola_in_interval_l1514_151497

theorem min_value_of_parabola_in_interval :
  ∀ x : ℝ, -10 ≤ x ∧ x ≤ 0 → (x^2 + 12 * x + 35) ≥ -1 := by
  sorry

end NUMINAMATH_GPT_min_value_of_parabola_in_interval_l1514_151497


namespace NUMINAMATH_GPT_boys_on_soccer_team_l1514_151467

theorem boys_on_soccer_team (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : B = 15 :=
sorry

end NUMINAMATH_GPT_boys_on_soccer_team_l1514_151467


namespace NUMINAMATH_GPT_yogurt_combinations_l1514_151413

theorem yogurt_combinations : (4 * Nat.choose 8 3) = 224 := by
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l1514_151413


namespace NUMINAMATH_GPT_factor_64_minus_16y_squared_l1514_151409

theorem factor_64_minus_16y_squared (y : ℝ) : 
  64 - 16 * y^2 = 16 * (2 - y) * (2 + y) :=
by
  -- skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_factor_64_minus_16y_squared_l1514_151409


namespace NUMINAMATH_GPT_find_divisors_of_10_pow_10_sum_157_l1514_151484

theorem find_divisors_of_10_pow_10_sum_157 
  (x y : ℕ) 
  (hx₁ : 0 < x) 
  (hy₁ : 0 < y) 
  (hx₂ : x ∣ 10^10) 
  (hy₂ : y ∣ 10^10) 
  (hxy₁ : x ≠ y) 
  (hxy₂ : x + y = 157) : 
  (x = 32 ∧ y = 125) ∨ (x = 125 ∧ y = 32) := 
by
  sorry

end NUMINAMATH_GPT_find_divisors_of_10_pow_10_sum_157_l1514_151484


namespace NUMINAMATH_GPT_symmetric_circle_eqn_l1514_151468

theorem symmetric_circle_eqn :
  ∀ (x y : ℝ),
  ((x + 1)^2 + (y - 1)^2 = 1) ∧ (x - y - 1 = 0) →
  (∀ (x' y' : ℝ), (x' = y + 1) ∧ (y' = x - 1) → (x' + 1)^2 + (y' - 1)^2 = 1) →
  (x - 2)^2 + (y + 2)^2 = 1 :=
by
  intros x y h h_sym
  sorry

end NUMINAMATH_GPT_symmetric_circle_eqn_l1514_151468


namespace NUMINAMATH_GPT_range_of_a_l1514_151478

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
if h : x < 1 then a * x^2 - 6 * x + a^2 + 1 else x^(5 - 2 * a)

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≥ f a y) ↔ (5/2 < a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1514_151478


namespace NUMINAMATH_GPT_multiply_and_simplify_l1514_151477
open Classical

theorem multiply_and_simplify (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end NUMINAMATH_GPT_multiply_and_simplify_l1514_151477


namespace NUMINAMATH_GPT_parabola_coeff_sum_l1514_151423

def parabola_vertex_form (a b c : ℚ) : Prop :=
  (∀ y : ℚ, y = 2 → (-3) = a * (y - 2)^2 + b * (y - 2) + c) ∧
  (∀ x y : ℚ, x = 1 ∧ y = -1 → x = a * y^2 + b * y + c) ∧
  (a < 0)  -- Since the parabola opens to the left, implying the coefficient 'a' is positive.

theorem parabola_coeff_sum (a b c : ℚ) :
  parabola_vertex_form a b c → a + b + c = -23 / 9 :=
by
  sorry

end NUMINAMATH_GPT_parabola_coeff_sum_l1514_151423


namespace NUMINAMATH_GPT_students_walk_fraction_l1514_151418

theorem students_walk_fraction :
  (1 - (1/3 + 1/5 + 1/10 + 1/15)) = 3/10 :=
by sorry

end NUMINAMATH_GPT_students_walk_fraction_l1514_151418


namespace NUMINAMATH_GPT_proof_problem_l1514_151471

-- Definitions of the function and conditions:
def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodicity_f : ∀ x, f (x + 2) = -f x
axiom f_def_on_interval : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2^x - 1

-- The theorem statement:
theorem proof_problem :
  f 6 < f (11 / 2) ∧ f (11 / 2) < f (-7) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1514_151471


namespace NUMINAMATH_GPT_chord_square_l1514_151499

/-- 
Circles with radii 3 and 6 are externally tangent and are internally tangent to a circle with radius 9. 
The circle with radius 9 has a chord that is a common external tangent of the other two circles. Prove that 
the square of the length of this chord is 72.
-/
theorem chord_square (O₁ O₂ O₃ : Type) 
  (r₁ r₂ r₃ : ℝ) 
  (O₁_tangent_O₂ : r₁ + r₂ = 9) 
  (O₃_tangent_O₁ : r₃ - r₁ = 6) 
  (O₃_tangent_O₂ : r₃ - r₂ = 3) 
  (tangent_chord : ℝ) : 
  tangent_chord^2 = 72 :=
by sorry

end NUMINAMATH_GPT_chord_square_l1514_151499


namespace NUMINAMATH_GPT_Shyam_money_l1514_151472

theorem Shyam_money (r g k s : ℕ) 
  (h1 : 7 * g = 17 * r) 
  (h2 : 7 * k = 17 * g)
  (h3 : 11 * s = 13 * k)
  (hr : r = 735) : 
  s = 2119 := 
by
  sorry

end NUMINAMATH_GPT_Shyam_money_l1514_151472


namespace NUMINAMATH_GPT_capacity_ratio_proof_l1514_151454

noncomputable def capacity_ratio :=
  ∀ (C_X C_Y : ℝ), 
    (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y →
    (C_Y / C_X) = (1 / 2)

-- includes a statement without proof
theorem capacity_ratio_proof (C_X C_Y : ℝ) (h : (1 / 2) * C_X + (2 / 5) * C_Y = (65 / 100) * C_Y) : 
  (C_Y / C_X) = (1 / 2) :=
  by
    sorry

end NUMINAMATH_GPT_capacity_ratio_proof_l1514_151454


namespace NUMINAMATH_GPT_hyperbola_representation_l1514_151439

variable (x y : ℝ)

/--
Given the equation (x - y)^2 = 3(x^2 - y^2), we prove that
the resulting graph represents a hyperbola.
-/
theorem hyperbola_representation :
  (x - y)^2 = 3 * (x^2 - y^2) →
  ∃ A B C : ℝ, A ≠ 0 ∧ (x^2 + x * y - 2 * y^2 = 0) ∧ (A = 1) ∧ (B = 1) ∧ (C = -2) ∧ (B^2 - 4*A*C > 0) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_representation_l1514_151439


namespace NUMINAMATH_GPT_probability_at_least_two_students_succeeding_l1514_151410

-- The probabilities of each student succeeding
def p1 : ℚ := 1 / 2
def p2 : ℚ := 1 / 4
def p3 : ℚ := 1 / 5

/-- Calculation of the total probability that at least two out of the three students succeed -/
theorem probability_at_least_two_students_succeeding : 
  (p1 * p2 * (1 - p3)) + (p1 * (1 - p2) * p3) + ((1 - p1) * p2 * p3) + (p1 * p2 * p3) = 9 / 40 :=
  sorry

end NUMINAMATH_GPT_probability_at_least_two_students_succeeding_l1514_151410


namespace NUMINAMATH_GPT_number_equation_form_l1514_151485

variable (a : ℝ)

theorem number_equation_form :
  3 * a + 5 = 4 * a := 
sorry

end NUMINAMATH_GPT_number_equation_form_l1514_151485


namespace NUMINAMATH_GPT_solve_for_x_add_y_l1514_151402

theorem solve_for_x_add_y (x y : ℤ) 
  (h1 : y = 245) 
  (h2 : x - y = 200) : 
  x + y = 690 :=
by {
  -- Here we would provide the proof if needed
  sorry
}

end NUMINAMATH_GPT_solve_for_x_add_y_l1514_151402


namespace NUMINAMATH_GPT_triangle_shape_l1514_151425

theorem triangle_shape (a b c : ℝ) (A B C : ℝ) (h1 : a * Real.cos A = b * Real.cos B) :
  (a = b ∨ c = a ∨ c = b ∨ A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_triangle_shape_l1514_151425


namespace NUMINAMATH_GPT_total_pages_in_book_l1514_151479

theorem total_pages_in_book (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 22) (h2 : days = 569) : total_pages = 12518 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l1514_151479


namespace NUMINAMATH_GPT_necklace_wire_length_l1514_151458

theorem necklace_wire_length
  (spools : ℕ)
  (feet_per_spool : ℕ)
  (total_necklaces : ℕ)
  (h1 : spools = 3)
  (h2 : feet_per_spool = 20)
  (h3 : total_necklaces = 15) :
  (spools * feet_per_spool) / total_necklaces = 4 := by
  sorry

end NUMINAMATH_GPT_necklace_wire_length_l1514_151458


namespace NUMINAMATH_GPT_pizzas_ordered_l1514_151475

def number_of_people : ℝ := 8.0
def slices_per_person : ℝ := 2.625
def slices_per_pizza : ℝ := 8.0

theorem pizzas_ordered : ⌈number_of_people * slices_per_person / slices_per_pizza⌉ = 3 := 
by
  sorry

end NUMINAMATH_GPT_pizzas_ordered_l1514_151475


namespace NUMINAMATH_GPT_max_catch_up_distance_l1514_151469

/-- 
Given:
  - The total length of the race is 5000 feet.
  - Alex and Max are even for the first 200 feet, so the initial distance between them is 0 feet.
  - On the uphill slope, Alex gets ahead by 300 feet.
  - On the downhill slope, Max gains a lead of 170 feet over Alex, reducing Alex's lead.
  - On the flat section, Alex pulls ahead by 440 feet.

Prove:
  - The distance left for Max to catch up to Alex is 4430 feet.
--/
theorem max_catch_up_distance :
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  total_distance - final_distance = 4430 :=
by
  let total_distance := 5000
  let initial_distance := 0
  let alex_uphill_lead := 300
  let max_downhill_gain := 170
  let alex_flat_gain := 440
  let final_distance := initial_distance + alex_uphill_lead - max_downhill_gain + alex_flat_gain
  have final_distance_calc : final_distance = 570
  sorry
  show total_distance - final_distance = 4430
  sorry

end NUMINAMATH_GPT_max_catch_up_distance_l1514_151469


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1514_151451

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1514_151451


namespace NUMINAMATH_GPT_no_square_number_divisible_by_six_in_range_l1514_151408

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ (x : ℕ), (x ^ 2) % 6 = 0 ∧ 39 < x ^ 2 ∧ x ^ 2 < 120 :=
by
  sorry

end NUMINAMATH_GPT_no_square_number_divisible_by_six_in_range_l1514_151408


namespace NUMINAMATH_GPT_probability_not_red_is_two_thirds_l1514_151490

-- Given conditions as definitions
def number_of_orange_marbles : ℕ := 4
def number_of_purple_marbles : ℕ := 7
def number_of_red_marbles : ℕ := 8
def number_of_yellow_marbles : ℕ := 5

-- Define the total number of marbles
def total_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_red_marbles + 
  number_of_yellow_marbles

def number_of_non_red_marbles : ℕ :=
  number_of_orange_marbles + 
  number_of_purple_marbles + 
  number_of_yellow_marbles

-- Define the probability
def probability_not_red : ℚ :=
  number_of_non_red_marbles / total_marbles

-- The theorem that states the probability of not picking a red marble is 2/3
theorem probability_not_red_is_two_thirds :
  probability_not_red = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_red_is_two_thirds_l1514_151490


namespace NUMINAMATH_GPT_goods_train_length_is_470_l1514_151403

noncomputable section

def speed_kmph := 72
def platform_length := 250
def crossing_time := 36

def speed_mps := speed_kmph * 5 / 18
def distance_covered := speed_mps * crossing_time

def length_of_train := distance_covered - platform_length

theorem goods_train_length_is_470 :
  length_of_train = 470 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_length_is_470_l1514_151403


namespace NUMINAMATH_GPT_cone_base_circumference_l1514_151447

-- Definitions of the problem
def radius : ℝ := 5
def angle_sector_degree : ℝ := 120
def full_circle_degree : ℝ := 360

-- Proof statement
theorem cone_base_circumference 
  (r : ℝ) (angle_sector : ℝ) (full_angle : ℝ) 
  (h1 : r = radius) 
  (h2 : angle_sector = angle_sector_degree) 
  (h3 : full_angle = full_circle_degree) : 
  (angle_sector / full_angle) * (2 * π * r) = (10 * π) / 3 := 
by sorry

end NUMINAMATH_GPT_cone_base_circumference_l1514_151447


namespace NUMINAMATH_GPT_find_orig_denominator_l1514_151449

-- Definitions as per the conditions
def orig_numer : ℕ := 2
def mod_numer : ℕ := orig_numer + 3

-- The modified fraction yields 1/3
def new_fraction (d : ℕ) : Prop :=
  (mod_numer : ℚ) / (d + 4) = 1 / 3

-- Proof Problem Statement
theorem find_orig_denominator (d : ℕ) : new_fraction d → d = 11 :=
  sorry

end NUMINAMATH_GPT_find_orig_denominator_l1514_151449


namespace NUMINAMATH_GPT_calc_expression_l1514_151480

theorem calc_expression :
  (12^4 + 375) * (24^4 + 375) * (36^4 + 375) * (48^4 + 375) * (60^4 + 375) /
  ((6^4 + 375) * (18^4 + 375) * (30^4 + 375) * (42^4 + 375) * (54^4 + 375)) = 159 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1514_151480


namespace NUMINAMATH_GPT_tamara_diff_3kim_height_l1514_151427

variables (K T X : ℕ) -- Kim's height, Tamara's height, and the difference inches respectively

-- Conditions
axiom ht_Tamara : T = 68
axiom combined_ht : T + K = 92
axiom diff_eqn : T = 3 * K - X

theorem tamara_diff_3kim_height (h₁ : T = 68) (h₂ : T + K = 92) (h₃ : T = 3 * K - X) : X = 4 :=
by
  sorry

end NUMINAMATH_GPT_tamara_diff_3kim_height_l1514_151427


namespace NUMINAMATH_GPT_diophantine_solution_l1514_151411

theorem diophantine_solution (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (n : ℕ) (h_n : n > a * b) :
  ∃ x y : ℕ, n = a * x + b * y :=
by
  sorry

end NUMINAMATH_GPT_diophantine_solution_l1514_151411


namespace NUMINAMATH_GPT_find_Xe_minus_Ye_l1514_151476

theorem find_Xe_minus_Ye (e X Y : ℕ) (h1 : 8 < e) (h2 : e^2*X + e*Y + e*X + X + e^2*X + X = 243 * e^2):
  X - Y = (2 * e^2 + 4 * e - 726) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_Xe_minus_Ye_l1514_151476


namespace NUMINAMATH_GPT_eating_time_l1514_151466

-- Defining the terms based on the conditions provided
def rate_mr_swift := 1 / 15 -- Mr. Swift eats 1 pound in 15 minutes
def rate_mr_slow := 1 / 45  -- Mr. Slow eats 1 pound in 45 minutes

-- Combined eating rate of Mr. Swift and Mr. Slow
def combined_rate := rate_mr_swift + rate_mr_slow

-- Total amount of cereal to be consumed
def total_cereal := 4 -- pounds

-- Proving the total time to eat the cereal
theorem eating_time :
  (total_cereal / combined_rate) = 45 :=
by
  sorry

end NUMINAMATH_GPT_eating_time_l1514_151466


namespace NUMINAMATH_GPT_price_reduction_relationship_l1514_151465

variable (a : ℝ) -- original price a in yuan
variable (b : ℝ) -- final price b in yuan

-- condition: price decreased by 10% first
def priceAfterFirstReduction := a * (1 - 0.10)

-- condition: price decreased by 20% on the result of the first reduction
def finalPrice := priceAfterFirstReduction a * (1 - 0.20)

-- theorem: relationship between original price a and final price b
theorem price_reduction_relationship (h : b = finalPrice a) : 
  b = a * (1 - 0.10) * (1 - 0.20) :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_price_reduction_relationship_l1514_151465


namespace NUMINAMATH_GPT_coefficient_of_x_100_l1514_151456

-- Define the polynomial P
noncomputable def P : Polynomial ℤ :=
  (Polynomial.C (-1) + Polynomial.X) *
  (Polynomial.C (-2) + Polynomial.X^2) *
  (Polynomial.C (-3) + Polynomial.X^3) *
  (Polynomial.C (-4) + Polynomial.X^4) *
  (Polynomial.C (-5) + Polynomial.X^5) *
  (Polynomial.C (-6) + Polynomial.X^6) *
  (Polynomial.C (-7) + Polynomial.X^7) *
  (Polynomial.C (-8) + Polynomial.X^8) *
  (Polynomial.C (-9) + Polynomial.X^9) *
  (Polynomial.C (-10) + Polynomial.X^10) *
  (Polynomial.C (-11) + Polynomial.X^11) *
  (Polynomial.C (-12) + Polynomial.X^12) *
  (Polynomial.C (-13) + Polynomial.X^13) *
  (Polynomial.C (-14) + Polynomial.X^14) *
  (Polynomial.C (-15) + Polynomial.X^15)

-- State the theorem
theorem coefficient_of_x_100 : P.coeff 100 = 445 :=
  by sorry

end NUMINAMATH_GPT_coefficient_of_x_100_l1514_151456
