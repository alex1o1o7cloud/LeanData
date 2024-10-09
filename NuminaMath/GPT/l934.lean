import Mathlib

namespace subtract_angles_l934_93486

theorem subtract_angles :
  (90 * 60 * 60 - (78 * 60 * 60 + 28 * 60 + 56)) = (11 * 60 * 60 + 31 * 60 + 4) :=
by
  sorry

end subtract_angles_l934_93486


namespace total_mustard_bottles_l934_93472

theorem total_mustard_bottles : 
  let table1 : ℝ := 0.25
  let table2 : ℝ := 0.25
  let table3 : ℝ := 0.38
  table1 + table2 + table3 = 0.88 :=
by
  sorry

end total_mustard_bottles_l934_93472


namespace empty_subset_of_disjoint_and_nonempty_l934_93423

variable {α : Type*} (A B : Set α)

theorem empty_subset_of_disjoint_and_nonempty (h₁ : A ≠ ∅) (h₂ : A ∩ B = ∅) : ∅ ⊆ B :=
by
  sorry

end empty_subset_of_disjoint_and_nonempty_l934_93423


namespace domain_f_domain_g_intersection_M_N_l934_93497

namespace MathProof

open Set

def M : Set ℝ := { x | -2 < x ∧ x < 4 }
def N : Set ℝ := { x | x < 1 ∨ x ≥ 3 }

theorem domain_f :
  (M = { x : ℝ | -2 < x ∧ x < 4 }) := by
  sorry

theorem domain_g :
  (N = { x : ℝ | x < 1 ∨ x ≥ 3 }) := by
  sorry

theorem intersection_M_N : 
  (M ∩ N = { x : ℝ | (-2 < x ∧ x < 1) ∨ (3 ≤ x ∧ x < 4) }) := by
  sorry

end MathProof

end domain_f_domain_g_intersection_M_N_l934_93497


namespace aaron_erasers_l934_93456

theorem aaron_erasers (initial_erasers erasers_given_to_Doris erasers_given_to_Ethan erasers_given_to_Fiona : ℕ) 
  (h1 : initial_erasers = 225) 
  (h2 : erasers_given_to_Doris = 75) 
  (h3 : erasers_given_to_Ethan = 40) 
  (h4 : erasers_given_to_Fiona = 50) : 
  initial_erasers - (erasers_given_to_Doris + erasers_given_to_Ethan + erasers_given_to_Fiona) = 60 :=
by sorry

end aaron_erasers_l934_93456


namespace find_g_30_l934_93469

def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) = x * g y

axiom g_one : g 1 = 10

theorem find_g_30 : g 30 = 300 := by
  sorry

end find_g_30_l934_93469


namespace correct_option_l934_93485

-- Definitions
def option_A (a : ℕ) : Prop := a^2 * a^3 = a^5
def option_B (a : ℕ) : Prop := a^6 / a^2 = a^3
def option_C (a b : ℕ) : Prop := (a * b^3) ^ 2 = a^2 * b^9
def option_D (a : ℕ) : Prop := 5 * a - 2 * a = 3

-- Theorem statement
theorem correct_option :
  (∃ (a : ℕ), option_A a) ∧
  (∀ (a : ℕ), ¬option_B a) ∧
  (∀ (a b : ℕ), ¬option_C a b) ∧
  (∀ (a : ℕ), ¬option_D a) :=
by
  sorry

end correct_option_l934_93485


namespace eccentricity_of_hyperbola_l934_93444

noncomputable def find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) : ℝ :=
Real.sqrt (1 + (b / a)^2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) :
  find_eccentricity a b h1 h2 h3 = 3 / 2 := by
  sorry

end eccentricity_of_hyperbola_l934_93444


namespace number_of_roses_per_set_l934_93468

-- Define the given conditions
def total_days : ℕ := 7
def sets_per_day : ℕ := 2
def total_roses : ℕ := 168

-- Define the statement to be proven
theorem number_of_roses_per_set : 
  (sets_per_day * total_days * (total_roses / (sets_per_day * total_days)) = total_roses) ∧ 
  (total_roses / (sets_per_day * total_days) = 12) :=
by 
  sorry

end number_of_roses_per_set_l934_93468


namespace remainder_5x_div_9_l934_93455

theorem remainder_5x_div_9 {x : ℕ} (h : x % 9 = 5) : (5 * x) % 9 = 7 :=
sorry

end remainder_5x_div_9_l934_93455


namespace find_g_60_l934_93453

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_func_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y^2
axiom g_45 : g 45 = 15

theorem find_g_60 : g 60 = 8.4375 := sorry

end find_g_60_l934_93453


namespace medal_winners_combinations_l934_93419

theorem medal_winners_combinations:
  ∀ n k : ℕ, (n = 6) → (k = 3) → (n.choose k = 20) :=
by
  intros n k hn hk
  simp [hn, hk]
  -- We can continue the proof using additional math concepts if necessary.
  sorry

end medal_winners_combinations_l934_93419


namespace domain_of_sqrt_2_cos_x_minus_1_l934_93427

theorem domain_of_sqrt_2_cos_x_minus_1 :
  {x : ℝ | ∃ k : ℤ, - (Real.pi / 3) + 2 * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + 2 * k * Real.pi } =
  {x : ℝ | 2 * Real.cos x - 1 ≥ 0 } :=
sorry

end domain_of_sqrt_2_cos_x_minus_1_l934_93427


namespace julius_wins_probability_l934_93415

noncomputable def probability_julius_wins (p_julius p_larry : ℚ) : ℚ :=
  (p_julius / (1 - p_larry ^ 2))

theorem julius_wins_probability :
  probability_julius_wins (2/3) (1/3) = 3/4 :=
by
  sorry

end julius_wins_probability_l934_93415


namespace find_value_l934_93482

theorem find_value (a b : ℝ) (h1 : 2 * a - 3 * b = 1) : 5 - 4 * a + 6 * b = 3 := 
by
  sorry

end find_value_l934_93482


namespace total_rocks_l934_93434

-- Definitions of variables based on the conditions
variables (igneous shiny_igneous : ℕ) (sedimentary : ℕ) (metamorphic : ℕ) (comet shiny_comet : ℕ)
variables (h1 : 1 / 4 * igneous = 15) (h2 : 1 / 2 * comet = 20)
variables (h3 : comet = 2 * metamorphic) (h4 : igneous = 3 * metamorphic)
variables (h5 : sedimentary = 2 * igneous)

-- The statement to be proved: the total number of rocks is 240
theorem total_rocks (igneous sedimentary metamorphic comet : ℕ) 
  (h1 : igneous = 4 * 15) 
  (h2 : comet = 2 * 20)
  (h3 : comet = 2 * metamorphic) 
  (h4 : igneous = 3 * metamorphic) 
  (h5 : sedimentary = 2 * igneous) : 
  igneous + sedimentary + metamorphic + comet = 240 :=
sorry

end total_rocks_l934_93434


namespace nonnegative_expr_interval_l934_93426

noncomputable def expr (x : ℝ) : ℝ := (2 * x - 15 * x ^ 2 + 56 * x ^ 3) / (9 - x ^ 3)

theorem nonnegative_expr_interval (x : ℝ) :
  expr x ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end nonnegative_expr_interval_l934_93426


namespace remainder_r15_minus_1_l934_93436

theorem remainder_r15_minus_1 (r : ℝ) : 
    (r^15 - 1) % (r - 1) = 0 :=
sorry

end remainder_r15_minus_1_l934_93436


namespace arithmetic_sequence_solution_l934_93473

theorem arithmetic_sequence_solution (a : ℕ → ℝ) (d : ℝ) 
(h1 : d ≠ 0) 
(h2 : a 1 = 2) 
(h3 : a 1 * a 4 = (a 2) ^ 2) :
∀ n, a n = 2 * n :=
by 
  sorry

end arithmetic_sequence_solution_l934_93473


namespace total_cost_fencing_l934_93409

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end total_cost_fencing_l934_93409


namespace find_constant_l934_93452

theorem find_constant
  {x : ℕ} (f : ℕ → ℕ)
  (h1 : ∀ x, f x = x^2 + 2*x + c)
  (h2 : f 2 = 12) :
  c = 4 :=
by sorry

end find_constant_l934_93452


namespace necessary_but_not_sufficient_l934_93487

variable (a : ℝ)

theorem necessary_but_not_sufficient (h : a ≥ 2) : (a = 2 ∨ a > 2) ∧ ¬(a > 2 → a ≥ 2) := by
  sorry

end necessary_but_not_sufficient_l934_93487


namespace sales_in_third_month_is_6855_l934_93429

noncomputable def sales_in_third_month : ℕ :=
  let sale_1 := 6435
  let sale_2 := 6927
  let sale_4 := 7230
  let sale_5 := 6562
  let sale_6 := 6791
  let total_sales := 6800 * 6
  total_sales - (sale_1 + sale_2 + sale_4 + sale_5 + sale_6)

theorem sales_in_third_month_is_6855 : sales_in_third_month = 6855 := by
  sorry

end sales_in_third_month_is_6855_l934_93429


namespace pictures_per_album_l934_93431

theorem pictures_per_album (phone_pics camera_pics albums : ℕ) (h_phone : phone_pics = 22) (h_camera : camera_pics = 2) (h_albums : albums = 4) (h_total_pics : phone_pics + camera_pics = 24) : (phone_pics + camera_pics) / albums = 6 :=
by
  sorry

end pictures_per_album_l934_93431


namespace george_reels_per_day_l934_93402

theorem george_reels_per_day
  (days : ℕ := 5)
  (jackson_per_day : ℕ := 6)
  (jonah_per_day : ℕ := 4)
  (total_fishes : ℕ := 90) :
  (∃ george_per_day : ℕ, george_per_day = 8) :=
by
  -- Calculation steps are skipped here; they would need to be filled in for a complete proof.
  sorry

end george_reels_per_day_l934_93402


namespace find_a11_l934_93457

-- Defining the sequence a_n and its properties
def seq (a : ℕ → ℝ) : Prop :=
  (a 3 = 2) ∧ 
  (a 5 = 1) ∧ 
  (∃ d, ∀ n, (1 / (1 + a n)) = (1 / (1 + a 1)) + (n - 1) * d)

-- The goal is to prove that the value of a_{11} is 0
theorem find_a11 (a : ℕ → ℝ) (h : seq a) : a 11 = 0 :=
sorry

end find_a11_l934_93457


namespace find_f_sum_l934_93454

noncomputable def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom functional_eq : ∀ x : ℝ, f (2 + x) + f (2 - x) = 0
axiom f_at_one : f 1 = 9

theorem find_f_sum :
  f 2010 + f 2011 + f 2012 = -9 :=
sorry

end find_f_sum_l934_93454


namespace range_of_m_l934_93470

noncomputable def f (x : ℝ) (m : ℝ) :=
if x ≤ 2 then x^2 - m * (2 * x - 1) + m^2 else 2^(x + 1)

theorem range_of_m {m : ℝ} :
  (∀ x, f x m ≥ f 2 m) → (2 ≤ m ∧ m ≤ 4) :=
by
  sorry

end range_of_m_l934_93470


namespace first_grade_frequency_is_correct_second_grade_frequency_is_correct_l934_93420

def total_items : ℕ := 400
def second_grade_items : ℕ := 20
def first_grade_items : ℕ := total_items - second_grade_items

def frequency_first_grade : ℚ := first_grade_items / total_items
def frequency_second_grade : ℚ := second_grade_items / total_items

theorem first_grade_frequency_is_correct : frequency_first_grade = 0.95 := 
 by
 sorry

theorem second_grade_frequency_is_correct : frequency_second_grade = 0.05 := 
 by 
 sorry

end first_grade_frequency_is_correct_second_grade_frequency_is_correct_l934_93420


namespace shoes_total_price_l934_93493

-- Define the variables involved
variables (S J : ℝ)

-- Define the conditions
def condition1 : Prop := J = (1 / 4) * S
def condition2 : Prop := 6 * S + 4 * J = 560

-- Define the total price calculation
def total_price : ℝ := 6 * S

-- State the theorem and proof goal
theorem shoes_total_price (h1 : condition1 S J) (h2 : condition2 S J) : total_price S = 480 := 
sorry

end shoes_total_price_l934_93493


namespace maximum_value_expression_l934_93433

-- Defining the variables and the main condition
variables (x y z : ℝ)

-- Assuming the non-negativity and sum of squares conditions
variables (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x^2 + y^2 + z^2 = 1)

-- Main statement about the maximum value
theorem maximum_value_expression : 
  4 * x * y * Real.sqrt 2 + 5 * y * z + 3 * x * z * Real.sqrt 3 ≤ 
  (44 * Real.sqrt 2 + 110 + 9 * Real.sqrt 3) / 3 :=
sorry

end maximum_value_expression_l934_93433


namespace difference_between_percent_and_value_is_five_l934_93428

def hogs : ℕ := 75
def ratio : ℕ := 3

def num_of_cats (hogs : ℕ) (ratio : ℕ) : ℕ := hogs / ratio

def cats : ℕ := num_of_cats hogs ratio

def percent_of_cats (cats : ℕ) : ℝ := 0.60 * cats
def value_to_subtract : ℕ := 10

def difference (percent : ℝ) (value : ℕ) : ℝ := percent - value

theorem difference_between_percent_and_value_is_five
    (hogs : ℕ)
    (ratio : ℕ)
    (cats : ℕ := num_of_cats hogs ratio)
    (percent : ℝ := percent_of_cats cats)
    (value : ℕ := value_to_subtract)
    :
    difference percent value = 5 :=
by {
    sorry
}

end difference_between_percent_and_value_is_five_l934_93428


namespace minimum_score_118_l934_93445

noncomputable def minimum_score (μ σ : ℝ) (p : ℝ) : ℝ :=
  sorry

theorem minimum_score_118 :
  minimum_score 98 10 (9100 / 400000) = 118 :=
by sorry

end minimum_score_118_l934_93445


namespace part1_part2_l934_93492

-- Define the cost price, current selling price, sales per week, and change in sales per reduction in price.
def cost_price : ℝ := 50
def current_price : ℝ := 80
def current_sales : ℝ := 200
def sales_increase_per_yuan : ℝ := 20

-- Define the weekly profit calculation.
def weekly_profit (price : ℝ) : ℝ :=
(price - cost_price) * (current_sales + sales_increase_per_yuan * (current_price - price))

-- Part 1: Selling price for a weekly profit of 7500 yuan while maximizing customer benefits.
theorem part1 (price : ℝ) : 
  (weekly_profit price = 7500) →  -- Given condition for weekly profit
  (price = 65) := sorry  -- Conclude that the price must be 65 yuan for maximizing customer benefits

-- Part 2: Selling price to maximize the weekly profit and the maximum profit
theorem part2 : 
  ∃ price : ℝ, (price = 70 ∧ weekly_profit price = 8000) := sorry  -- Conclude that the price is 70 yuan and max profit is 8000 yuan

end part1_part2_l934_93492


namespace find_s_l934_93442

variable (x t s : ℝ)

-- Conditions
#check (0.75 * x) / 60  -- Time for the first part of the trip
#check 0.25 * x  -- Distance for the remaining part of the trip
#check t - (0.75 * x) / 60  -- Time for the remaining part of the trip
#check 40 * t  -- Solving for x from average speed relation

-- Prove the value of s
theorem find_s (h1 : x = 40 * t) (h2 : s = (0.25 * x) / (t - (0.75 * x) / 60)) : s = 20 := by sorry

end find_s_l934_93442


namespace inequality_nonneg_real_l934_93494

theorem inequality_nonneg_real (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ (2 / (1 + a * b)) ∧ ((1 / (1 + a^2)) + (1 / (1 + b^2)) = (2 / (1 + a * b)) ↔ a = b) :=
sorry

end inequality_nonneg_real_l934_93494


namespace secret_spread_reaches_3280_on_saturday_l934_93476

theorem secret_spread_reaches_3280_on_saturday :
  (∃ n : ℕ, 4 * ( 3^n - 1) / 2 + 1 = 3280 ) ∧ n = 7  :=
sorry

end secret_spread_reaches_3280_on_saturday_l934_93476


namespace remainder_is_20_l934_93460

def N := 220020
def a := 555
def b := 445
def d := a + b
def q := 2 * (a - b)

theorem remainder_is_20 : N % d = 20 := by
  sorry

end remainder_is_20_l934_93460


namespace power_of_two_has_half_nines_l934_93466

theorem power_of_two_has_half_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, (∃ m : ℕ, (k / 2 < m) ∧ 
            (10^k ∣ (2^n + m + 1)) ∧ 
            (2^n % (10^k) = 10^k - 1)) :=
sorry

end power_of_two_has_half_nines_l934_93466


namespace total_subjects_l934_93464

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end total_subjects_l934_93464


namespace probability_before_third_ring_l934_93404

-- Definitions of the conditions
def prob_first_ring : ℝ := 0.2
def prob_second_ring : ℝ := 0.3

-- Theorem stating that the probability of being answered before the third ring is 0.5
theorem probability_before_third_ring : prob_first_ring + prob_second_ring = 0.5 :=
by
  sorry

end probability_before_third_ring_l934_93404


namespace value_of_r_when_m_eq_3_l934_93432

theorem value_of_r_when_m_eq_3 :
  ∀ (r t m : ℕ),
  r = 5^t - 2*t →
  t = 3^m + 2 →
  m = 3 →
  r = 5^29 - 58 :=
by
  intros r t m h1 h2 h3
  rw [h3] at h2
  rw [Nat.pow_succ] at h2
  sorry

end value_of_r_when_m_eq_3_l934_93432


namespace part1_part2_l934_93465

variable {f : ℝ → ℝ}

theorem part1 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : f 1 = 0 :=
by sorry

theorem part2 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f (-x) + f (3 - x) ≥ 2 :=
by sorry

end part1_part2_l934_93465


namespace product_simplification_l934_93471

theorem product_simplification :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) = 7 :=
by
  sorry

end product_simplification_l934_93471


namespace product_of_consecutive_integers_eq_255_l934_93406

theorem product_of_consecutive_integers_eq_255 (x : ℕ) (h : x * (x + 1) = 255) : x + (x + 1) = 31 := 
sorry

end product_of_consecutive_integers_eq_255_l934_93406


namespace three_digit_numbers_containing_2_and_exclude_6_l934_93489

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end three_digit_numbers_containing_2_and_exclude_6_l934_93489


namespace find_other_number_l934_93435

theorem find_other_number
  (a b : ℕ)  -- Define the numbers as natural numbers
  (h1 : a = 300)             -- Condition stating the certain number is 300
  (h2 : a = 150 * b)         -- Condition stating the ratio is 150:1
  : b = 2 :=                 -- Goal stating the other number should be 2
  by
    sorry                    -- Placeholder for the proof steps

end find_other_number_l934_93435


namespace number_division_l934_93498

theorem number_division (x : ℚ) (h : x / 6 = 1 / 10) : (x / (3 / 25)) = 5 :=
by {
  sorry
}

end number_division_l934_93498


namespace part1_part2_l934_93440

variable {a x y : ℝ} 

-- Conditions
def condition_1 (a x y : ℝ) := x - y = 1 + 3 * a
def condition_2 (a x y : ℝ) := x + y = -7 - a
def condition_3 (x : ℝ) := x ≤ 0
def condition_4 (y : ℝ) := y < 0

-- Part 1: Range for a
theorem part1 (a : ℝ) : 
  (∀ x y, condition_1 a x y ∧ condition_2 a x y ∧ condition_3 x ∧ condition_4 y → (-2 < a ∧ a ≤ 3)) :=
sorry

-- Part 2: Specific integer value for a
theorem part2 (a : ℝ) :
  (-2 < a ∧ a ≤ 3 → (∃ (x : ℝ), (2 * a + 1) * x > 2 * a + 1 ∧ x < 1) → a = -1) :=
sorry

end part1_part2_l934_93440


namespace find_smaller_number_l934_93425

-- Define the conditions
def sum_of_numbers (x y : ℕ) := x + y = 70
def second_number_relation (x y : ℕ) := y = 3 * x + 10

-- Define the problem statement
theorem find_smaller_number (x y : ℕ) (h1 : sum_of_numbers x y) (h2 : second_number_relation x y) : x = 15 :=
sorry

end find_smaller_number_l934_93425


namespace find_sum_l934_93410

theorem find_sum (A B : ℕ) (h1 : B = 278 + 365 * 3) (h2 : A = 20 * 100 + 87 * 10) : A + B = 4243 := by
    sorry

end find_sum_l934_93410


namespace initial_percentage_of_water_l934_93437

variable (P : ℚ) -- Initial percentage of water

theorem initial_percentage_of_water (h : P / 100 * 40 + 5 = 9) : P = 10 := 
  sorry

end initial_percentage_of_water_l934_93437


namespace new_time_between_maintenance_checks_l934_93499

-- Definitions based on the conditions
def original_time : ℝ := 25
def percentage_increase : ℝ := 0.20

-- Statement to be proved
theorem new_time_between_maintenance_checks : original_time * (1 + percentage_increase) = 30 := by
  sorry

end new_time_between_maintenance_checks_l934_93499


namespace find_n_from_t_l934_93481

theorem find_n_from_t (n t : ℕ) (h1 : t = n * (n - 1) * (n + 1) + n) (h2 : t = 64) : n = 4 := by
  sorry

end find_n_from_t_l934_93481


namespace canoes_built_by_April_l934_93491

theorem canoes_built_by_April :
  (∃ (c1 c2 c3 c4 : ℕ), 
    c1 = 5 ∧ 
    c2 = 3 * c1 ∧ 
    c3 = 3 * c2 ∧ 
    c4 = 3 * c3 ∧
    (c1 + c2 + c3 + c4) = 200) :=
sorry

end canoes_built_by_April_l934_93491


namespace option_a_option_b_option_d_l934_93467

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end option_a_option_b_option_d_l934_93467


namespace amount_invested_l934_93416

theorem amount_invested (P : ℝ) :
  P * (1.03)^2 - P = 0.08 * P + 6 → P = 314.136 := by
  sorry

end amount_invested_l934_93416


namespace time_addition_sum_l934_93461

theorem time_addition_sum (A B C : ℕ) (h1 : A = 7) (h2 : B = 59) (h3 : C = 59) : A + B + C = 125 :=
sorry

end time_addition_sum_l934_93461


namespace proposition_false_l934_93405

theorem proposition_false (x y : ℤ) (h : x + y = 5) : ¬ (x = 1 ∧ y = 4) := by 
  sorry

end proposition_false_l934_93405


namespace area_of_quadrilateral_l934_93417

theorem area_of_quadrilateral (d o1 o2 : ℝ) (h1 : d = 24) (h2 : o1 = 9) (h3 : o2 = 6) :
  (1 / 2 * d * o1) + (1 / 2 * d * o2) = 180 :=
by {
  sorry
}

end area_of_quadrilateral_l934_93417


namespace service_fee_correct_l934_93413
open Nat -- Open the natural number namespace

-- Define the conditions
def ticket_price : ℕ := 44
def num_tickets : ℕ := 3
def total_paid : ℕ := 150

-- Define the cost of tickets
def cost_of_tickets : ℕ := ticket_price * num_tickets

-- Define the service fee calculation
def service_fee : ℕ := total_paid - cost_of_tickets

-- The proof problem statement
theorem service_fee_correct : service_fee = 18 :=
by
  -- Omits the proof, providing a placeholder.
  sorry

end service_fee_correct_l934_93413


namespace tens_digit_of_3_pow_100_l934_93438

-- Definition: The cyclic behavior of the last two digits of 3^n.
def last_two_digits_cycle : List ℕ := [03, 09, 27, 81, 43, 29, 87, 61, 83, 49, 47, 41, 23, 69, 07, 21, 63, 89, 67, 01]

-- Condition: The length of the cycle of the last two digits of 3^n.
def cycle_length : ℕ := 20

-- Assertion: The last two digits of 3^20 is 01.
def last_two_digits_3_pow_20 : ℕ := 1

-- Given n = 100, the tens digit of 3^n when n is expressed in decimal notation
theorem tens_digit_of_3_pow_100 : (3 ^ 100 / 10) % 10 = 0 := by
  let n := 100
  let position_in_cycle := (n % cycle_length)
  have cycle_repeat : (n % cycle_length = 0) := rfl
  have digits_3_pow_20 : (3^20 % 100 = 1) := by sorry
  show (3 ^ 100 / 10) % 10 = 0
  sorry

end tens_digit_of_3_pow_100_l934_93438


namespace least_integer_solution_l934_93495

theorem least_integer_solution :
  ∃ x : ℤ, (abs (3 * x - 4) ≤ 25) ∧ (∀ y : ℤ, (abs (3 * y - 4) ≤ 25) → x ≤ y) :=
sorry

end least_integer_solution_l934_93495


namespace domain_of_f_l934_93407

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = x} = {x : ℝ | x ≠ 6} := by
  sorry

end domain_of_f_l934_93407


namespace dogs_not_eat_either_l934_93441

-- Definitions for our conditions
variable (dogs_total : ℕ) (dogs_watermelon : ℕ) (dogs_salmon : ℕ) (dogs_both : ℕ)

-- Specific values of our conditions
def dogs_total_value : ℕ := 60
def dogs_watermelon_value : ℕ := 9
def dogs_salmon_value : ℕ := 48
def dogs_both_value : ℕ := 5

-- The theorem we need to prove
theorem dogs_not_eat_either : 
    dogs_total = dogs_total_value → 
    dogs_watermelon = dogs_watermelon_value → 
    dogs_salmon = dogs_salmon_value → 
    dogs_both = dogs_both_value → 
    (dogs_total - (dogs_watermelon + dogs_salmon - dogs_both) = 8) :=
by
  intros
  sorry

end dogs_not_eat_either_l934_93441


namespace symmetric_point_P_l934_93443

-- Define the point P
def P : ℝ × ℝ := (1, -2)

-- Define the function to get the symmetric point with respect to the origin
def symmetric_point (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.1, -point.2)

-- State the theorem that proves the symmetric point of P is (-1, 2)
theorem symmetric_point_P :
  symmetric_point P = (-1, 2) :=
  sorry

end symmetric_point_P_l934_93443


namespace number_of_tangent_lines_through_origin_l934_93490

def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

def f_prime (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := f x₀ + f_prime x₀ * (x - x₀)

theorem number_of_tangent_lines_through_origin : 
  ∃! (x₀ : ℝ), x₀^3 - 3*x₀^2 + 4 = 0 := 
sorry

end number_of_tangent_lines_through_origin_l934_93490


namespace symphony_orchestra_has_260_members_l934_93475

def symphony_orchestra_member_count (n : ℕ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4

theorem symphony_orchestra_has_260_members : symphony_orchestra_member_count 260 :=
by {
  sorry
}

end symphony_orchestra_has_260_members_l934_93475


namespace arith_general_formula_geom_general_formula_geom_sum_formula_l934_93459

-- Arithmetic Sequence Conditions
def arith_seq (a₈ a₁₀ : ℕ → ℝ) := a₈ = 6 ∧ a₁₀ = 0

-- General formula for arithmetic sequence
theorem arith_general_formula (a₁ : ℝ) (d : ℝ) (h₈ : 6 = a₁ + 7 * d) (h₁₀ : 0 = a₁ + 9 * d) :
  ∀ n : ℕ, aₙ = 30 - 3 * (n - 1) :=
sorry

-- General formula for geometric sequence
def geom_seq (a₁ a₄ : ℕ → ℝ) := a₁ = 1/2 ∧ a₄ = 4

theorem geom_general_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, aₙ = 2^(n-2) :=
sorry

-- Sum of the first n terms of geometric sequence
theorem geom_sum_formula (a₁ : ℝ) (q : ℝ) (h₁ : a₁ = 1 / 2) (h₄ : 4 = a₁ * q ^ 3) :
  ∀ n : ℕ, Sₙ = 2^(n-1) - 1 / 2 :=
sorry

end arith_general_formula_geom_general_formula_geom_sum_formula_l934_93459


namespace min_value_h_l934_93408

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end min_value_h_l934_93408


namespace geometric_progression_solution_l934_93446

theorem geometric_progression_solution (p : ℝ) :
  (3 * p + 1)^2 = (9 * p + 10) * |p - 3| ↔ p = -1 ∨ p = 29 / 18 :=
by
  sorry

end geometric_progression_solution_l934_93446


namespace quadratic_even_coeff_l934_93424

theorem quadratic_even_coeff (a b c : ℤ) (h : a ≠ 0) (hq : ∃ x : ℚ, a * x^2 + b * x + c = 0) : ¬ (∀ x : ℤ, (x ≠ 0 → (x % 2 = 1))) := 
sorry

end quadratic_even_coeff_l934_93424


namespace sine_five_l934_93458

noncomputable def sine_value (x : ℝ) : ℝ :=
  Real.sin (5 * x)

theorem sine_five : sine_value 1 = -0.959 := 
  by
  sorry

end sine_five_l934_93458


namespace least_integer_k_l934_93412

theorem least_integer_k (k : ℕ) (h : k ^ 3 ∣ 336) : k = 84 :=
sorry

end least_integer_k_l934_93412


namespace percentage_difference_is_50_percent_l934_93483

-- Definitions of hourly wages
def Mike_hourly_wage : ℕ := 14
def Phil_hourly_wage : ℕ := 7

-- Calculating the percentage difference
theorem percentage_difference_is_50_percent :
  (Mike_hourly_wage - Phil_hourly_wage) * 100 / Mike_hourly_wage = 50 :=
by
  sorry

end percentage_difference_is_50_percent_l934_93483


namespace percentage_markup_l934_93484

theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 4800) (h₂ : cost_price = 3840) :
  (selling_price - cost_price) / cost_price * 100 = 25 :=
by
  sorry

end percentage_markup_l934_93484


namespace jameson_total_medals_l934_93462

-- Define the number of track, swimming, and badminton medals
def track_medals := 5
def swimming_medals := 2 * track_medals
def badminton_medals := 5

-- Define the total number of medals
def total_medals := track_medals + swimming_medals + badminton_medals

-- Theorem statement
theorem jameson_total_medals : total_medals = 20 := 
by
  sorry

end jameson_total_medals_l934_93462


namespace fraction_value_l934_93474

theorem fraction_value : (5 * 7) / 10.0 = 3.5 := by
  sorry

end fraction_value_l934_93474


namespace no_real_solutions_of_quadratic_eq_l934_93421

theorem no_real_solutions_of_quadratic_eq
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  ∀ x : ℝ, ¬ (b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 = 0) :=
by
  sorry

end no_real_solutions_of_quadratic_eq_l934_93421


namespace find_b_value_l934_93496

theorem find_b_value
  (b : ℝ) :
  (∃ x y : ℝ, x = 3 ∧ y = -5 ∧ b * x + (b + 2) * y = b - 1) → b = -3 :=
by
  sorry

end find_b_value_l934_93496


namespace number_in_parentheses_l934_93488

theorem number_in_parentheses (x : ℤ) (h : x - (-2) = 3) : x = 1 :=
by {
  sorry
}

end number_in_parentheses_l934_93488


namespace ellipse_focal_distance_correct_l934_93430

noncomputable def ellipse_focal_distance (x y : ℝ) (θ : ℝ) : ℝ :=
  let a := 5 -- semi-major axis
  let b := 2 -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2) -- calculate focal distance
  2 * c -- return 2c

theorem ellipse_focal_distance_correct (θ : ℝ) :
  ellipse_focal_distance (-4 + 2 * Real.cos θ) (1 + 5 * Real.sin θ) θ = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_focal_distance_correct_l934_93430


namespace log_problem_l934_93439

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_problem :
  let x := (log_base 8 2) ^ (log_base 2 8)
  log_base 3 x = -3 :=
by
  sorry

end log_problem_l934_93439


namespace first_day_more_than_200_paperclips_l934_93414

def paperclips_after_days (k : ℕ) : ℕ :=
  3 * 2^k

theorem first_day_more_than_200_paperclips : (∀ k, 3 * 2^k <= 200) → k <= 7 → 3 * 2^7 > 200 → k = 7 :=
by
  intro h_le h_lt h_gt
  sorry

end first_day_more_than_200_paperclips_l934_93414


namespace selling_price_with_discount_l934_93403

variable (a : ℝ)

theorem selling_price_with_discount (h : a ≥ 0) : (a * 1.2 * 0.91) = (a * 1.2 * 0.91) :=
by
  sorry

end selling_price_with_discount_l934_93403


namespace athlete_a_catches_up_and_race_duration_l934_93451

-- Track is 1000 meters
def track_length : ℕ := 1000

-- Athlete A's speed: first minute, increasing until 5th minute and decreasing until 600 meters/min
def athlete_A_speed (minute : ℕ) : ℕ :=
  match minute with
  | 0 => 1000
  | 1 => 1000
  | 2 => 1200
  | 3 => 1400
  | 4 => 1600
  | 5 => 1400
  | 6 => 1200
  | 7 => 1000
  | 8 => 800
  | 9 => 600
  | _ => 600

-- Athlete B's constant speed
def athlete_B_speed : ℕ := 1200

-- Function to compute distance covered in given minutes, assuming starts at 0
def total_distance (speed : ℕ → ℕ) (minutes : ℕ) : ℕ :=
  (List.range minutes).map speed |>.sum

-- Defining the maximum speed moment for A
def athlete_A_max_speed_distance : ℕ := total_distance athlete_A_speed 4
def athlete_B_max_speed_distance : ℕ := athlete_B_speed * 4

-- Proof calculation for target time 10 2/3 minutes
def time_catch : ℚ := 10 + 2 / 3

-- Defining the theorem to be proven
theorem athlete_a_catches_up_and_race_duration :
  athlete_A_max_speed_distance > athlete_B_max_speed_distance ∧ time_catch = 32 / 3 :=
by
  -- Place holder for the proof's details
  sorry

end athlete_a_catches_up_and_race_duration_l934_93451


namespace range_of_m_l934_93449

noncomputable def system_of_equations (x y m : ℝ) : Prop :=
  (x + 2 * y = 1 - m) ∧ (2 * x + y = 3)

variable (x y m : ℝ)

theorem range_of_m (h : system_of_equations x y m) (hxy : x + y > 0) : m < 4 :=
by
  sorry

end range_of_m_l934_93449


namespace determine_a_minus_b_l934_93400

theorem determine_a_minus_b (a b : ℤ) 
  (h1 : 2009 * a + 2013 * b = 2021) 
  (h2 : 2011 * a + 2015 * b = 2023) : 
  a - b = -5 :=
sorry

end determine_a_minus_b_l934_93400


namespace cos_diff_simplify_l934_93447

theorem cos_diff_simplify (x : ℝ) (y : ℝ) (h1 : x = Real.cos (Real.pi / 10)) (h2 : y = Real.cos (3 * Real.pi / 10)) : 
  x - y = 4 * x * (1 - x^2) := 
sorry

end cos_diff_simplify_l934_93447


namespace solve_expression_l934_93418

def evaluation_inside_parentheses : ℕ := 3 - 3

def power_of_zero : ℝ := (5 : ℝ) ^ evaluation_inside_parentheses

theorem solve_expression :
  (3 : ℝ) - power_of_zero = 2 := by
  -- Utilize the conditions defined above
  sorry

end solve_expression_l934_93418


namespace min_value_condition_l934_93477

noncomputable def poly_min_value (a b : ℝ) : ℝ := a^2 + b^2

theorem min_value_condition (a b : ℝ) (h: ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  ∃ a b : ℝ, poly_min_value a b = 4 := 
by sorry

end min_value_condition_l934_93477


namespace stable_k_digit_number_l934_93479

def is_stable (a k : ℕ) : Prop :=
  ∀ m n : ℕ, (10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a))

theorem stable_k_digit_number (k : ℕ) (h_pos : k > 0) : ∃ (a : ℕ) (h : ∀ m n : ℕ, 10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a)), (10^(k-1)) ≤ a ∧ a < 10^k ∧ ∀ b : ℕ, (∀ m n : ℕ, 10^k ∣ ((m * 10^k + b) * (n * 10^k + b) - b)) → (10^(k-1)) ≤ b ∧ b < 10^k → a = b :=
by
  sorry

end stable_k_digit_number_l934_93479


namespace problem1_problem2_l934_93448

-- Problem 1: Prove f(x) ≥ 3 implies x ≤ -1 or x ≥ 1 given f(x) = |x + 1| + |2x - 1| and m = 1
theorem problem1 (x : ℝ) : (|x + 1| + |2 * x - 1| >= 3) ↔ (x <= -1 ∨ x >= 1) :=
by
 sorry

-- Problem 2: Prove ½ f(x) ≤ |x + 1| holds for x ∈ [m, 2m²] implies ½ < m ≤ 1 given f(x) = |x + m| + |2x - 1| and m > 0
theorem problem2 (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : m ≤ x ∧ x ≤ 2 * m^2) : (1/2 * (|x + m| + |2 * x - 1|) ≤ |x + 1|) ↔ (1/2 < m ∧ m ≤ 1) :=
by
 sorry

end problem1_problem2_l934_93448


namespace union_of_sets_l934_93422

open Set

-- Define the sets A and B
def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

-- Prove that the union of A and B equals {–2, 0, 3}
theorem union_of_sets : A ∪ B = {-2, 0, 3} := by
  sorry

end union_of_sets_l934_93422


namespace cube_edges_after_cuts_l934_93450

theorem cube_edges_after_cuts (V E : ℕ) (hV : V = 8) (hE : E = 12) : 
  12 + 24 = 36 := by
  sorry

end cube_edges_after_cuts_l934_93450


namespace probability_blue_face_l934_93463

theorem probability_blue_face :
  (3 / 6 : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end probability_blue_face_l934_93463


namespace student_entrepreneur_profit_l934_93480

theorem student_entrepreneur_profit {x y a: ℝ} 
  (h1 : a * (y - x) = 1000) 
  (h2 : (ay / x) * y - ay = 1500)
  (h3 : y = 3 / 2 * x) : a * x = 2000 := 
sorry

end student_entrepreneur_profit_l934_93480


namespace loan_payment_period_years_l934_93411

noncomputable def house_cost := 480000
noncomputable def trailer_cost := 120000
noncomputable def monthly_difference := 1500

theorem loan_payment_period_years:
  ∃ N : ℕ, (house_cost = (trailer_cost / N + monthly_difference) * N ∧
            N = 240) →
            N / 12 = 20 :=
sorry

end loan_payment_period_years_l934_93411


namespace discount_is_100_l934_93401

-- Define the constants for the problem conditions
def suit_cost : ℕ := 430
def shoes_cost : ℕ := 190
def amount_paid : ℕ := 520

-- Total cost before discount
def total_cost_before_discount (a b : ℕ) : ℕ := a + b

-- Discount amount
def discount_amount (total paid : ℕ) : ℕ := total - paid

-- Main theorem statement
theorem discount_is_100 : discount_amount (total_cost_before_discount suit_cost shoes_cost) amount_paid = 100 := 
by
sorry

end discount_is_100_l934_93401


namespace express_y_in_terms_of_x_and_p_l934_93478

theorem express_y_in_terms_of_x_and_p (x p : ℚ) (h : x = (1 + p / 100) * (1 / y)) : 
  y = (100 + p) / (100 * x) := 
sorry

end express_y_in_terms_of_x_and_p_l934_93478
