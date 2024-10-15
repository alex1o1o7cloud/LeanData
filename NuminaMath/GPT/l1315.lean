import Mathlib

namespace NUMINAMATH_GPT_solve_y_pos_in_arithmetic_seq_l1315_131508

-- Define the first term as 4
def first_term : ℕ := 4

-- Define the third term as 36
def third_term : ℕ := 36

-- Basing on the properties of an arithmetic sequence, 
-- we solve for the positive second term (y) such that its square equals to 20
theorem solve_y_pos_in_arithmetic_seq : ∃ y : ℝ, y > 0 ∧ y ^ 2 = 20 := by
  sorry

end NUMINAMATH_GPT_solve_y_pos_in_arithmetic_seq_l1315_131508


namespace NUMINAMATH_GPT_tan_alpha_l1315_131568

variable (α : ℝ)
variable (H_cos : Real.cos α = 12/13)
variable (H_quadrant : 3 * Real.pi / 2 < α ∧ α < 2 * Real.pi)

theorem tan_alpha :
  Real.tan α = -5/12 :=
sorry

end NUMINAMATH_GPT_tan_alpha_l1315_131568


namespace NUMINAMATH_GPT_fifth_term_sequence_l1315_131571

theorem fifth_term_sequence : 2^5 * 1 * 3 * 5 * 7 * 9 = 6 * 7 * 8 * 9 * 10 := 
by 
  sorry

end NUMINAMATH_GPT_fifth_term_sequence_l1315_131571


namespace NUMINAMATH_GPT_not_possible_one_lies_other_not_l1315_131505

-- Variable definitions: Jean is lying (J), Pierre is lying (P)
variable (J P : Prop)

-- Conditions from the problem
def Jean_statement : Prop := P → J
def Pierre_statement : Prop := P → J

-- Theorem statement
theorem not_possible_one_lies_other_not (h1 : Jean_statement J P) (h2 : Pierre_statement J P) : ¬ ((J ∨ ¬ J) ∧ (P ∨ ¬ P) ∧ ((J ∧ ¬ P) ∨ (¬ J ∧ P))) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_one_lies_other_not_l1315_131505


namespace NUMINAMATH_GPT_sequence_geometric_sum_bn_l1315_131556

theorem sequence_geometric (a : ℕ → ℕ) (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) : 
  (∀ n, a n = 2^n) :=
by sorry

theorem sum_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) 
  (h_gen : ∀ n, a n = 2^n) (h_bn : ∀ n, b n = n * a n) :
  (∀ n, S n = (n-1) * 2^(n+1) + 2) :=
by sorry

end NUMINAMATH_GPT_sequence_geometric_sum_bn_l1315_131556


namespace NUMINAMATH_GPT_min_length_intersection_l1315_131564

def set_with_length (a b : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ b}
def length_of_set (a b : ℝ) := b - a
def M (m : ℝ) := set_with_length m (m + 3/4)
def N (n : ℝ) := set_with_length (n - 1/3) n

theorem min_length_intersection (m n : ℝ) (h₁ : 0 ≤ m) (h₂ : m + 3/4 ≤ 1) (h₃ : 0 ≤ n - 1/3) (h₄ : n ≤ 1) : 
  length_of_set (max m (n - 1/3)) (min (m + 3/4) n) = 1/12 :=
by
  sorry

end NUMINAMATH_GPT_min_length_intersection_l1315_131564


namespace NUMINAMATH_GPT_translation_preserves_parallel_and_equal_length_l1315_131548

theorem translation_preserves_parallel_and_equal_length
    (A B C D : ℝ)
    (after_translation : (C - A) = (D - B))
    (connecting_parallel : C - A = D - B) :
    (C - A = D - B) ∧ (C - A = D - B) :=
by
  sorry

end NUMINAMATH_GPT_translation_preserves_parallel_and_equal_length_l1315_131548


namespace NUMINAMATH_GPT_larger_number_is_72_l1315_131520

theorem larger_number_is_72 (a b : ℕ) (h1 : 5 * b = 6 * a) (h2 : b - a = 12) : b = 72 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_72_l1315_131520


namespace NUMINAMATH_GPT_quadratic_roots_l1315_131517

theorem quadratic_roots (r s : ℝ) (A : ℝ) (B : ℝ) (C : ℝ) (p q : ℝ) 
  (h1 : A = 3) (h2 : B = 4) (h3 : C = 5) 
  (h4 : r + s = -B / A) (h5 : rs = C / A) 
  (h6 : 4 * rs = q) :
  p = 56 / 9 :=
by 
  -- We assume the correct answer is given as we skip the proof details here.
  sorry

end NUMINAMATH_GPT_quadratic_roots_l1315_131517


namespace NUMINAMATH_GPT_new_average_doubled_l1315_131552

theorem new_average_doubled
  (average : ℕ)
  (num_students : ℕ)
  (h_avg : average = 45)
  (h_num_students : num_students = 30)
  : (2 * average * num_students / num_students) = 90 := by
  sorry

end NUMINAMATH_GPT_new_average_doubled_l1315_131552


namespace NUMINAMATH_GPT_not_perfect_square_l1315_131530

theorem not_perfect_square (p : ℕ) (hp : Nat.Prime p) : ¬ ∃ t : ℕ, 7 * p + 3^p - 4 = t^2 :=
sorry

end NUMINAMATH_GPT_not_perfect_square_l1315_131530


namespace NUMINAMATH_GPT_event_complementary_and_mutually_exclusive_l1315_131599

def students : Finset (String × String) := 
  { ("boy", "1"), ("boy", "2"), ("boy", "3"), ("girl", "1"), ("girl", "2") }

def event_at_least_one_girl (s : Finset (String × String)) : Prop :=
  ∃ x ∈ s, (x.1 = "girl")

def event_all_boys (s : Finset (String × String)) : Prop :=
  ∀ x ∈ s, (x.1 = "boy")

def two_students (s : Finset (String × String)) : Prop :=
  s.card = 2

theorem event_complementary_and_mutually_exclusive :
  ∀ s: Finset (String × String), two_students s → 
  (event_at_least_one_girl s ↔ ¬ event_all_boys s) ∧ 
  (event_all_boys s ↔ ¬ event_at_least_one_girl s) :=
sorry

end NUMINAMATH_GPT_event_complementary_and_mutually_exclusive_l1315_131599


namespace NUMINAMATH_GPT_gamin_difference_calculation_l1315_131544

def largest_number : ℕ := 532
def smallest_number : ℕ := 406
def difference : ℕ := 126

theorem gamin_difference_calculation : largest_number - smallest_number = difference :=
by
  -- The solution proves that the difference between the largest and smallest numbers is 126.
  sorry

end NUMINAMATH_GPT_gamin_difference_calculation_l1315_131544


namespace NUMINAMATH_GPT_jessica_flowers_problem_l1315_131578

theorem jessica_flowers_problem
(initial_roses initial_daisies : ℕ)
(thrown_roses thrown_daisies : ℕ)
(current_roses current_daisies : ℕ)
(cut_roses cut_daisies : ℕ)
(h_initial_roses : initial_roses = 21)
(h_initial_daisies : initial_daisies = 17)
(h_thrown_roses : thrown_roses = 34)
(h_thrown_daisies : thrown_daisies = 25)
(h_current_roses : current_roses = 15)
(h_current_daisies : current_daisies = 10)
(h_cut_roses : cut_roses = (thrown_roses - initial_roses) + current_roses)
(h_cut_daisies : cut_daisies = (thrown_daisies - initial_daisies) + current_daisies) :
thrown_roses + thrown_daisies - (cut_roses + cut_daisies) = 13 := by
  sorry

end NUMINAMATH_GPT_jessica_flowers_problem_l1315_131578


namespace NUMINAMATH_GPT_count_valid_numbers_l1315_131545

theorem count_valid_numbers : 
  let count_A := 10 
  let count_B := 2 
  count_A * count_B = 20 :=
by 
  let count_A := 10
  let count_B := 2
  have : count_A * count_B = 20 := by norm_num
  exact this

end NUMINAMATH_GPT_count_valid_numbers_l1315_131545


namespace NUMINAMATH_GPT_joe_paint_problem_l1315_131515

theorem joe_paint_problem (f : ℝ) (h₁ : 360 * f + (1 / 6) * (360 - 360 * f) = 135) : f = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_joe_paint_problem_l1315_131515


namespace NUMINAMATH_GPT_integer_cubed_fraction_l1315_131565

theorem integer_cubed_fraction
  (a b : ℕ)
  (hab : 0 < b ∧ 0 < a)
  (h : (a^2 + b^2) % (a - b)^2 = 0) :
  (a^3 + b^3) % (a - b)^3 = 0 :=
by sorry

end NUMINAMATH_GPT_integer_cubed_fraction_l1315_131565


namespace NUMINAMATH_GPT_find_integer_pairs_l1315_131549

theorem find_integer_pairs (x y : ℤ) (h_xy : x ≤ y) (h_eq : (1 : ℚ)/x + (1 : ℚ)/y = 1/4) :
  (x, y) = (5, 20) ∨ (x, y) = (6, 12) ∨ (x, y) = (8, 8) ∨ (x, y) = (-4, 2) ∨ (x, y) = (-12, 3) :=
sorry

end NUMINAMATH_GPT_find_integer_pairs_l1315_131549


namespace NUMINAMATH_GPT_number_of_integers_satisfying_l1315_131536

theorem number_of_integers_satisfying (n : ℤ) : 
    (25 < n^2 ∧ n^2 < 144) → Finset.card (Finset.filter (fun n => 25 < n^2 ∧ n^2 < 144) (Finset.range 25)) = 12 := by
  sorry

end NUMINAMATH_GPT_number_of_integers_satisfying_l1315_131536


namespace NUMINAMATH_GPT_sum_of_solutions_eq_minus_2_l1315_131540

-- Defining the equation and the goal
theorem sum_of_solutions_eq_minus_2 (x1 x2 : ℝ) (floor : ℝ → ℤ) (h1 : floor (3 * x1 + 1) = 2 * x1 - 1 / 2)
(h2 : floor (3 * x2 + 1) = 2 * x2 - 1 / 2) :
  x1 + x2 = -2 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_minus_2_l1315_131540


namespace NUMINAMATH_GPT_cohen_saw_1300_fish_eater_birds_l1315_131551

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end NUMINAMATH_GPT_cohen_saw_1300_fish_eater_birds_l1315_131551


namespace NUMINAMATH_GPT_expenditure_record_l1315_131597

/-- Lean function to represent the condition and the proof problem -/
theorem expenditure_record (income expenditure : Int) (h_income : income = 500) (h_recorded_income : income = 500) (h_expenditure : expenditure = 200) : expenditure = -200 := 
by
  sorry

end NUMINAMATH_GPT_expenditure_record_l1315_131597


namespace NUMINAMATH_GPT_kerry_age_l1315_131582

theorem kerry_age (cost_per_box : ℝ) (boxes_bought : ℕ) (candles_per_box : ℕ) (cakes : ℕ) 
  (total_cost : ℝ) (total_candles : ℕ) (candles_per_cake : ℕ) (age : ℕ) :
  cost_per_box = 2.5 →
  boxes_bought = 2 →
  candles_per_box = 12 →
  cakes = 3 →
  total_cost = 5 →
  total_cost = boxes_bought * cost_per_box →
  total_candles = boxes_bought * candles_per_box →
  candles_per_cake = total_candles / cakes →
  age = candles_per_cake →
  age = 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_kerry_age_l1315_131582


namespace NUMINAMATH_GPT_john_age_l1315_131569

/-!
# John’s Current Age Proof
Given the following condition:
1. 9 years from now, John will be 3 times as old as he was 11 years ago.
Prove that John is currently 21 years old.
-/

def john_current_age (x : ℕ) : Prop :=
  (x + 9 = 3 * (x - 11)) → (x = 21)

-- Proof Statement
theorem john_age : john_current_age 21 :=
by
  sorry

end NUMINAMATH_GPT_john_age_l1315_131569


namespace NUMINAMATH_GPT_green_chameleon_increase_l1315_131559

variable (initial_yellow: ℕ) (initial_green: ℕ)

-- Number of sunny and cloudy days
def sunny_days: ℕ := 18
def cloudy_days: ℕ := 12

-- Change in yellow chameleons
def delta_yellow: ℕ := 5

theorem green_chameleon_increase 
  (initial_yellow: ℕ) (initial_green: ℕ) 
  (sunny_days: ℕ := 18) 
  (cloudy_days: ℕ := 12) 
  (delta_yellow: ℕ := 5): 
  (initial_green + 11) = initial_green + delta_yellow + (sunny_days - cloudy_days) := 
by
  sorry

end NUMINAMATH_GPT_green_chameleon_increase_l1315_131559


namespace NUMINAMATH_GPT_ratio_length_to_width_l1315_131509

-- Define the given conditions and values
def width : ℕ := 75
def perimeter : ℕ := 360

-- Define the proof problem statement
theorem ratio_length_to_width (L : ℕ) (P_eq : perimeter = 2 * L + 2 * width) :
  (L / width : ℚ) = 7 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_length_to_width_l1315_131509


namespace NUMINAMATH_GPT_sum_bases_l1315_131557

theorem sum_bases (R1 R2 : ℕ) (F1 F2 : ℚ)
  (h1 : F1 = (4 * R1 + 5) / (R1 ^ 2 - 1))
  (h2 : F2 = (5 * R1 + 4) / (R1 ^ 2 - 1))
  (h3 : F1 = (3 * R2 + 8) / (R2 ^ 2 - 1))
  (h4 : F2 = (6 * R2 + 1) / (R2 ^ 2 - 1)) :
  R1 + R2 = 19 :=
sorry

end NUMINAMATH_GPT_sum_bases_l1315_131557


namespace NUMINAMATH_GPT_panteleimon_twos_l1315_131521

-- Define the variables
variables (P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2 : ℕ)

-- Define the conditions
def conditions :=
  P_5 + P_4 + P_3 + P_2 = 20 ∧
  G_5 + G_4 + G_3 + G_2 = 20 ∧
  P_5 = G_4 ∧
  P_4 = G_3 ∧
  P_3 = G_2 ∧
  P_2 = G_5 ∧
  (5 * P_5 + 4 * P_4 + 3 * P_3 + 2 * P_2 = 5 * G_5 + 4 * G_4 + 3 * G_3 + 2 * G_2)

-- The proof goal
theorem panteleimon_twos (h : conditions P_5 P_4 P_3 P_2 G_5 G_4 G_3 G_2) : P_2 = 5 :=
sorry

end NUMINAMATH_GPT_panteleimon_twos_l1315_131521


namespace NUMINAMATH_GPT_continuous_func_unique_l1315_131511

theorem continuous_func_unique (f : ℝ → ℝ) (hf_cont : Continuous f)
  (hf_eqn : ∀ x : ℝ, f x + f (x^2) = 2) :
  ∀ x : ℝ, f x = 1 :=
by
  sorry

end NUMINAMATH_GPT_continuous_func_unique_l1315_131511


namespace NUMINAMATH_GPT_initial_tests_count_l1315_131589

theorem initial_tests_count (n S : ℕ)
  (h1 : S = 35 * n)
  (h2 : (S - 20) / (n - 1) = 40) :
  n = 4 := 
sorry

end NUMINAMATH_GPT_initial_tests_count_l1315_131589


namespace NUMINAMATH_GPT_cheapest_store_for_60_balls_l1315_131598

def cost_store_A (n : ℕ) (price_per_ball : ℕ) (free_per_10 : ℕ) : ℕ :=
  if n < 10 then n * price_per_ball
  else (n / 10) * 10 * price_per_ball + (n % 10) * price_per_ball * (n / (10 + free_per_10))

def cost_store_B (n : ℕ) (discount : ℕ) (price_per_ball : ℕ) : ℕ :=
  n * (price_per_ball - discount)

def cost_store_C (n : ℕ) (price_per_ball : ℕ) (cashback_threshold cashback_amt : ℕ) : ℕ :=
  let initial_cost := n * price_per_ball
  let cashback := (initial_cost / cashback_threshold) * cashback_amt
  initial_cost - cashback

theorem cheapest_store_for_60_balls
  (price_per_ball discount free_per_10 cashback_threshold cashback_amt : ℕ) :
  cost_store_A 60 price_per_ball free_per_10 = 1250 →
  cost_store_B 60 discount price_per_ball = 1200 →
  cost_store_C 60 price_per_ball cashback_threshold cashback_amt = 1290 →
  min (cost_store_A 60 price_per_ball free_per_10) (min (cost_store_B 60 discount price_per_ball) (cost_store_C 60 price_per_ball cashback_threshold cashback_amt))
  = 1200 :=
by
  sorry

end NUMINAMATH_GPT_cheapest_store_for_60_balls_l1315_131598


namespace NUMINAMATH_GPT_remainder_7_pow_4_div_100_l1315_131591

theorem remainder_7_pow_4_div_100 : (7 ^ 4) % 100 = 1 := 
by
  sorry

end NUMINAMATH_GPT_remainder_7_pow_4_div_100_l1315_131591


namespace NUMINAMATH_GPT_new_person_weight_l1315_131537

theorem new_person_weight (avg_increase : ℝ) (num_persons : ℕ) (initial_person_weight : ℝ) 
  (weight_increase : ℝ) (final_person_weight : ℝ) : 
  avg_increase = 2.5 ∧ num_persons = 8 ∧ initial_person_weight = 65 ∧ 
  weight_increase = num_persons * avg_increase ∧ final_person_weight = initial_person_weight + weight_increase 
  → final_person_weight = 85 :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_new_person_weight_l1315_131537


namespace NUMINAMATH_GPT_sequence_a4_eq_neg3_l1315_131512

theorem sequence_a4_eq_neg3 (a : ℕ → ℤ) (h1 : a 1 = 3) (h2 : a 2 = 6)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) : a 4 = -3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a4_eq_neg3_l1315_131512


namespace NUMINAMATH_GPT_min_value_expression_l1315_131516

theorem min_value_expression (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  (x^2 + 8 * x * y + 25 * y^2 + 16 * y * z + 9 * z^2) ≥ 403 / 9 := by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1315_131516


namespace NUMINAMATH_GPT_negate_proposition_l1315_131586

def p (x : ℝ) : Prop := x^2 + x - 6 > 0
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

def neg_p (x : ℝ) : Prop := x^2 + x - 6 ≤ 0
def neg_q (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 2

theorem negate_proposition (x : ℝ) :
  (¬ (p x → q x)) ↔ (neg_p x → neg_q x) :=
by unfold p q neg_p neg_q; apply sorry

end NUMINAMATH_GPT_negate_proposition_l1315_131586


namespace NUMINAMATH_GPT_zayne_total_revenue_l1315_131522

-- Defining the constants and conditions
def price_per_bracelet := 5
def deal_price := 8
def initial_bracelets := 30
def revenue_from_five_dollar_sales := 60

-- Calculating number of bracelets sold for $5 each
def bracelets_sold_five_dollars := revenue_from_five_dollar_sales / price_per_bracelet

-- Calculating remaining bracelets after selling some for $5 each
def remaining_bracelets := initial_bracelets - bracelets_sold_five_dollars

-- Calculating number of pairs sold at two for $8
def pairs_sold := remaining_bracelets / 2

-- Calculating revenue from selling pairs
def revenue_from_deal_sales := pairs_sold * deal_price

-- Total revenue calculation
def total_revenue := revenue_from_five_dollar_sales + revenue_from_deal_sales

-- Theorem to prove the total revenue is $132
theorem zayne_total_revenue : total_revenue = 132 := by
  sorry

end NUMINAMATH_GPT_zayne_total_revenue_l1315_131522


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1315_131513

theorem negation_of_universal_proposition {f : ℝ → ℝ} :
  (¬ (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0)) ↔
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1315_131513


namespace NUMINAMATH_GPT_only_function_l1315_131503

def divides (a b : ℕ) : Prop := ∃ k, b = k * a

def satisfies_condition (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, divides (f m + f n) (m + n)

theorem only_function (f : ℕ → ℕ) (h : satisfies_condition f) : f = id :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_only_function_l1315_131503


namespace NUMINAMATH_GPT_sum_xyz_eq_11sqrt5_l1315_131566

noncomputable def x : ℝ :=
sorry

noncomputable def y : ℝ :=
sorry

noncomputable def z : ℝ :=
sorry

axiom pos_x : x > 0
axiom pos_y : y > 0
axiom pos_z : z > 0

axiom xy_eq_30 : x * y = 30
axiom xz_eq_60 : x * z = 60
axiom yz_eq_90 : y * z = 90

theorem sum_xyz_eq_11sqrt5 : x + y + z = 11 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_sum_xyz_eq_11sqrt5_l1315_131566


namespace NUMINAMATH_GPT_money_distribution_l1315_131595

theorem money_distribution (A B C : ℝ) (h1 : A + B + C = 1000) (h2 : B + C = 600) (h3 : C = 300) : A + C = 700 := by
  sorry

end NUMINAMATH_GPT_money_distribution_l1315_131595


namespace NUMINAMATH_GPT_cost_to_feed_turtles_l1315_131554

theorem cost_to_feed_turtles :
  (∀ (weight : ℚ), weight > 0 → (food_per_half_pound_ounces = 1) →
    (total_weight_pounds = 30) →
    (jar_contents_ounces = 15 ∧ jar_cost = 2) →
    (total_cost_dollars = 8)) :=
by
  sorry

variables
  (food_per_half_pound_ounces : ℚ := 1)
  (total_weight_pounds : ℚ := 30)
  (jar_contents_ounces : ℚ := 15)
  (jar_cost : ℚ := 2)
  (total_cost_dollars : ℚ := 8)

-- Assuming all variables needed to state the theorem exist and are meaningful
/-!
  Given:
  - Each turtle needs 1 ounce of food per 1/2 pound of body weight
  - Total turtles' weight is 30 pounds
  - Each jar of food contains 15 ounces and costs $2

  Prove:
  - The total cost to feed Sylvie's turtles is $8
-/

end NUMINAMATH_GPT_cost_to_feed_turtles_l1315_131554


namespace NUMINAMATH_GPT_bathroom_width_l1315_131588

def length : ℝ := 4
def area : ℝ := 8
def width : ℝ := 2

theorem bathroom_width :
  area = length * width :=
by
  sorry

end NUMINAMATH_GPT_bathroom_width_l1315_131588


namespace NUMINAMATH_GPT_find_integers_a_b_c_l1315_131529

theorem find_integers_a_b_c :
  ∃ a b c : ℤ, ((x - a) * (x - 12) + 1 = (x + b) * (x + c)) ∧ 
  ((b + 12) * (c + 12) = 1 → ((b = -11 ∧ c = -11) → a = 10) ∧ 
  ((b = -13 ∧ c = -13) → a = 14)) :=
by
  sorry

end NUMINAMATH_GPT_find_integers_a_b_c_l1315_131529


namespace NUMINAMATH_GPT_volume_of_cut_pyramid_l1315_131572

theorem volume_of_cut_pyramid
  (base_length : ℝ)
  (slant_length : ℝ)
  (cut_height : ℝ)
  (original_base_area : ℝ)
  (original_height : ℝ)
  (new_base_area : ℝ)
  (volume : ℝ)
  (h_base_length : base_length = 8 * Real.sqrt 2)
  (h_slant_length : slant_length = 10)
  (h_cut_height : cut_height = 3)
  (h_original_base_area : original_base_area = (base_length ^ 2) / 2)
  (h_original_height : original_height = Real.sqrt (slant_length ^ 2 - (base_length / Real.sqrt 2) ^ 2))
  (h_new_base_area : new_base_area = original_base_area / 4)
  (h_volume : volume = (1 / 3) * new_base_area * cut_height) :
  volume = 32 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cut_pyramid_l1315_131572


namespace NUMINAMATH_GPT_expression_evaluation_l1315_131587

theorem expression_evaluation (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x^2 = 1 / y^2) :
  (x^2 - 4 / x^2) * (y^2 + 4 / y^2) = x^4 - 16 / x^4 :=
by
  sorry

end NUMINAMATH_GPT_expression_evaluation_l1315_131587


namespace NUMINAMATH_GPT_domain_of_function_l1315_131502

theorem domain_of_function :
  ∀ x : ℝ, ⌊x^2 - 8 * x + 18⌋ ≠ 0 :=
sorry

end NUMINAMATH_GPT_domain_of_function_l1315_131502


namespace NUMINAMATH_GPT_number_of_outcomes_for_champions_l1315_131592

def num_events : ℕ := 3
def num_competitors : ℕ := 6
def total_possible_outcomes : ℕ := num_competitors ^ num_events

theorem number_of_outcomes_for_champions :
  total_possible_outcomes = 216 :=
by
  sorry

end NUMINAMATH_GPT_number_of_outcomes_for_champions_l1315_131592


namespace NUMINAMATH_GPT_trigonometric_expression_eval_l1315_131500

theorem trigonometric_expression_eval :
  2 * (Real.cos (5 * Real.pi / 16))^6 +
  2 * (Real.sin (11 * Real.pi / 16))^6 +
  (3 * Real.sqrt 2 / 8) = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_eval_l1315_131500


namespace NUMINAMATH_GPT_find_b_l1315_131514

theorem find_b {a b : ℝ} (h₁ : 2 * 2 + b = 1 - 2 * a) (h₂ : -2 * 2 + b = -15 + 2 * a) : 
  b = -7 := sorry

end NUMINAMATH_GPT_find_b_l1315_131514


namespace NUMINAMATH_GPT_find_last_even_number_l1315_131580

theorem find_last_even_number (n : ℕ) (h : 4 * (n * (n + 1) * (2 * n + 1) / 6) = 560) : 2 * n = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_last_even_number_l1315_131580


namespace NUMINAMATH_GPT_jean_pages_written_l1315_131574

theorem jean_pages_written:
  (∀ d : ℕ, 150 * d = 900 → d * 2 = 12) :=
by
  sorry

end NUMINAMATH_GPT_jean_pages_written_l1315_131574


namespace NUMINAMATH_GPT_probability_to_buy_ticket_l1315_131535

def p : ℝ := 0.1
def q : ℝ := 0.9
def initial_money : ℝ := 20
def target_money : ℝ := 45
def ticket_cost : ℝ := 10
def prize : ℝ := 30

noncomputable def equation_lhs : ℝ := p^2 * (1 + 2 * q)
noncomputable def equation_rhs : ℝ := 1 - 2 * p * q^2

noncomputable def x2 : ℝ := equation_lhs / equation_rhs

theorem probability_to_buy_ticket : x2 = 0.033 := sorry

end NUMINAMATH_GPT_probability_to_buy_ticket_l1315_131535


namespace NUMINAMATH_GPT_find_speed_from_p_to_q_l1315_131581

noncomputable def speed_from_p_to_q (v : ℝ) (d : ℝ) : Prop :=
  let return_speed := 1.5 * v
  let avg_speed := 75
  let total_distance := 2 * d
  let total_time := d / v + d / return_speed
  avg_speed = total_distance / total_time

theorem find_speed_from_p_to_q (v : ℝ) (d : ℝ) : speed_from_p_to_q v d → v = 62.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_speed_from_p_to_q_l1315_131581


namespace NUMINAMATH_GPT_integer_roots_of_polynomial_l1315_131590

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  {x : ℤ | (x^3 + a₂ * x^2 + a₁ * x - 18 = 0)} ⊆ {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18} :=
by sorry

end NUMINAMATH_GPT_integer_roots_of_polynomial_l1315_131590


namespace NUMINAMATH_GPT_total_operations_l1315_131563

-- Define the process of iterative multiplication and division as described in the problem
def process (start : Nat) : Nat :=
  let m1 := 3 * start
  let m2 := 3 * m1
  let m3 := 3 * m2
  let m4 := 3 * m3
  let m5 := 3 * m4
  let d1 := m5 / 2
  let d2 := d1 / 2
  let d3 := d2 / 2
  let d4 := d3 / 2
  let d5 := d4 / 2
  let d6 := d5 / 2
  let d7 := d6 / 2
  d7

theorem total_operations : process 1 = 1 ∧ 5 + 7 = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_operations_l1315_131563


namespace NUMINAMATH_GPT_phone_answered_within_two_rings_l1315_131519

def probability_of_first_ring : ℝ := 0.5
def probability_of_second_ring : ℝ := 0.3
def probability_of_within_two_rings : ℝ := 0.8

theorem phone_answered_within_two_rings :
  probability_of_first_ring + probability_of_second_ring = probability_of_within_two_rings :=
by
  sorry

end NUMINAMATH_GPT_phone_answered_within_two_rings_l1315_131519


namespace NUMINAMATH_GPT_validate_shots_statistics_l1315_131575

-- Define the scores and their frequencies
def scores : List ℕ := [6, 7, 8, 9, 10]
def times : List ℕ := [4, 10, 11, 9, 6]

-- Condition 1: Calculate the mode
def mode := 8

-- Condition 2: Calculate the median
def median := 8

-- Condition 3: Calculate the 35th percentile
def percentile_35 := ¬(35 * 40 / 100 = 7)

-- Condition 4: Calculate the average
def average := 8.075

theorem validate_shots_statistics :
  mode = 8
  ∧ median = 8
  ∧ percentile_35
  ∧ average = 8.075 :=
by
  sorry

end NUMINAMATH_GPT_validate_shots_statistics_l1315_131575


namespace NUMINAMATH_GPT_polynomial_mod_p_zero_l1315_131585

def is_zero_mod_p (p : ℕ) [Fact (Nat.Prime p)] (f : (List ℕ → ℤ)) : Prop :=
  ∀ (x : List ℕ), f x % p = 0

theorem polynomial_mod_p_zero
  (p : ℕ) [Fact (Nat.Prime p)]
  (n : ℕ) 
  (f : (List ℕ → ℤ)) 
  (h : ∀ (x : List ℕ), f x % p = 0) 
  (g : (List ℕ → ℤ)) :
  (∀ (x : List ℕ), g x % p = 0) := sorry

end NUMINAMATH_GPT_polynomial_mod_p_zero_l1315_131585


namespace NUMINAMATH_GPT_max_sin_a_l1315_131562

theorem max_sin_a (a b : ℝ)
  (h1 : b = Real.pi / 2 - a)
  (h2 : Real.cos (a + b) = Real.cos a + Real.cos b) :
  Real.sin a ≤ Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_max_sin_a_l1315_131562


namespace NUMINAMATH_GPT_fraction_grades_C_l1315_131542

def fraction_grades_A (students : ℕ) : ℕ := (1 / 5) * students
def fraction_grades_B (students : ℕ) : ℕ := (1 / 4) * students
def num_grades_D : ℕ := 5
def total_students : ℕ := 100

theorem fraction_grades_C :
  (total_students - (fraction_grades_A total_students + fraction_grades_B total_students + num_grades_D)) / total_students = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_grades_C_l1315_131542


namespace NUMINAMATH_GPT_power_inequality_l1315_131526

theorem power_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) : 
  a^5 + b^5 > a^2 * b^3 + a^3 * b^2 :=
sorry

end NUMINAMATH_GPT_power_inequality_l1315_131526


namespace NUMINAMATH_GPT_composite_prop_true_l1315_131527

def p : Prop := ∀ (x : ℝ), x > 0 → x + (1/(2*x)) ≥ 1

def q : Prop := ∀ (x : ℝ), x > 1 → (x^2 + 2*x - 3 > 0)

theorem composite_prop_true : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_composite_prop_true_l1315_131527


namespace NUMINAMATH_GPT_minimum_rectangle_length_l1315_131596

theorem minimum_rectangle_length (a x y : ℝ) (h : x * y = a^2) : x ≥ a ∨ y ≥ a :=
sorry

end NUMINAMATH_GPT_minimum_rectangle_length_l1315_131596


namespace NUMINAMATH_GPT_quadratic_equation_unique_solution_l1315_131560

theorem quadratic_equation_unique_solution (a b x k : ℝ) (h : a = 8) (h₁ : b = 36) (h₂ : k = 40.5) : 
  (8*x^2 + 36*x + 40.5 = 0) ∧ x = -2.25 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_equation_unique_solution_l1315_131560


namespace NUMINAMATH_GPT_product_evaluation_l1315_131525

-- Define the conditions and the target expression
def product (a : ℕ) : ℕ := (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a

-- Main theorem statement
theorem product_evaluation : product 7 = 5040 :=
by
  -- Lean usually requires some import from the broader Mathlib to support arithmetic simplifications
  sorry

end NUMINAMATH_GPT_product_evaluation_l1315_131525


namespace NUMINAMATH_GPT_train_length_is_50_meters_l1315_131576

theorem train_length_is_50_meters
  (L : ℝ)
  (equal_length : ∀ (a b : ℝ), a = L ∧ b = L → a + b = 2 * L)
  (speed_faster_train : ℝ := 46) -- km/hr
  (speed_slower_train : ℝ := 36) -- km/hr
  (relative_speed : ℝ := speed_faster_train - speed_slower_train)
  (relative_speed_km_per_sec : ℝ := relative_speed / 3600) -- converting km/hr to km/sec
  (time : ℝ := 36) -- seconds
  (distance_covered : ℝ := 2 * L)
  (distance_eq : distance_covered = relative_speed_km_per_sec * time):
  L = 50 / 1000 :=
by 
  -- We will prove it as per the derived conditions
  sorry

end NUMINAMATH_GPT_train_length_is_50_meters_l1315_131576


namespace NUMINAMATH_GPT_dagger_simplified_l1315_131501

def dagger (m n p q : ℚ) : ℚ := (m^2) * p * (q / n)

theorem dagger_simplified :
  dagger (5:ℚ) (9:ℚ) (4:ℚ) (6:ℚ) = (200:ℚ) / (3:ℚ) :=
by
  sorry

end NUMINAMATH_GPT_dagger_simplified_l1315_131501


namespace NUMINAMATH_GPT_dogwood_trees_tomorrow_l1315_131573

def initial_dogwood_trees : Nat := 7
def trees_planted_today : Nat := 3
def final_total_dogwood_trees : Nat := 12

def trees_after_today : Nat := initial_dogwood_trees + trees_planted_today
def trees_planted_tomorrow : Nat := final_total_dogwood_trees - trees_after_today

theorem dogwood_trees_tomorrow :
  trees_planted_tomorrow = 2 :=
by
  sorry

end NUMINAMATH_GPT_dogwood_trees_tomorrow_l1315_131573


namespace NUMINAMATH_GPT_dvd_sold_168_l1315_131579

/-- 
Proof that the number of DVDs sold (D) is 168 given the conditions:
1) D = 1.6 * C
2) D + C = 273 
-/
theorem dvd_sold_168 (C D : ℝ) (h1 : D = 1.6 * C) (h2 : D + C = 273) : D = 168 := 
sorry

end NUMINAMATH_GPT_dvd_sold_168_l1315_131579


namespace NUMINAMATH_GPT_train_length_is_correct_l1315_131543

noncomputable def length_of_train (t : ℝ) (v_train : ℝ) (v_man : ℝ) : ℝ :=
  let relative_speed : ℝ := (v_train - v_man) * (5/18)
  relative_speed * t

theorem train_length_is_correct :
  length_of_train 23.998 63 3 = 400 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l1315_131543


namespace NUMINAMATH_GPT_candy_received_l1315_131555

theorem candy_received (pieces_eaten : ℕ) (piles : ℕ) (pieces_per_pile : ℕ) 
  (h_eaten : pieces_eaten = 12) (h_piles : piles = 4) (h_pieces_per_pile : pieces_per_pile = 5) :
  pieces_eaten + piles * pieces_per_pile = 32 := 
by
  sorry

end NUMINAMATH_GPT_candy_received_l1315_131555


namespace NUMINAMATH_GPT_possible_values_of_product_l1315_131547

theorem possible_values_of_product 
  (P_A P_B P_C P_D P_E : ℕ)
  (H1 : P_A = P_B + P_C + P_D + P_E)
  (H2 : ∃ n1 n2 n3 n4, 
          ((P_B = n1 * (n1 + 1)) ∨ (P_B = n2 * (n2 + 1) * (n2 + 2)) ∨ 
           (P_B = n3 * (n3 + 1) * (n3 + 2) * (n3 + 3)) ∨ (P_B = n4 * (n4 + 1) * (n4 + 2) * (n4 + 3) * (n4 + 4))) ∧
          ∃ m1 m2 m3 m4, 
          ((P_C = m1 * (m1 + 1)) ∨ (P_C = m2 * (m2 + 1) * (m2 + 2)) ∨ 
           (P_C = m3 * (m3 + 1) * (m3 + 2) * (m3 + 3)) ∨ (P_C = m4 * (m4 + 1) * (m4 + 2) * (m4 + 3) * (m4 + 4))) ∧
          ∃ o1 o2 o3 o4, 
          ((P_D = o1 * (o1 + 1)) ∨ (P_D = o2 * (o2 + 1) * (o2 + 2)) ∨ 
           (P_D = o3 * (o3 + 1) * (o3 + 2) * (o3 + 3)) ∨ (P_D = o4 * (o4 + 1) * (o4 + 2) * (o4 + 3) * (o4 + 4))) ∧
          ∃ p1 p2 p3 p4, 
          ((P_E = p1 * (p1 + 1)) ∨ (P_E = p2 * (p2 + 1) * (p2 + 2)) ∨ 
           (P_E = p3 * (p3 + 1) * (p3 + 2) * (p3 + 3)) ∨ (P_E = p4 * (p4 + 1) * (p4 + 2) * (p4 + 3) * (p4 + 4))) ∧ 
          ∃ q1 q2 q3 q4, 
          ((P_A = q1 * (q1 + 1)) ∨ (P_A = q2 * (q2 + 1) * (q2 + 2)) ∨ 
           (P_A = q3 * (q3 + 1) * (q3 + 2) * (q3 + 3)) ∨ (P_A = q4 * (q4 + 1) * (q4 + 2) * (q4 + 3) * (q4 + 4)))) :
  P_A = 6 ∨ P_A = 24 :=
by sorry

end NUMINAMATH_GPT_possible_values_of_product_l1315_131547


namespace NUMINAMATH_GPT_exp_eval_l1315_131510

theorem exp_eval : (3^2)^4 = 6561 := by
sorry

end NUMINAMATH_GPT_exp_eval_l1315_131510


namespace NUMINAMATH_GPT_total_emails_vacation_l1315_131567

def day_1_emails : ℕ := 16
def day_2_emails : ℕ := day_1_emails / 2
def day_3_emails : ℕ := day_2_emails / 2
def day_4_emails : ℕ := day_3_emails / 2

def total_emails : ℕ := day_1_emails + day_2_emails + day_3_emails + day_4_emails

theorem total_emails_vacation : total_emails = 30 := by
  -- Use "sorry" to skip the proof as per instructions.
  sorry

end NUMINAMATH_GPT_total_emails_vacation_l1315_131567


namespace NUMINAMATH_GPT_geometric_sequence_product_l1315_131523

theorem geometric_sequence_product 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_log_sum : Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6) :
  a 1 * a 15 = 10000 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1315_131523


namespace NUMINAMATH_GPT_cubic_polynomial_value_at_3_and_neg3_l1315_131561

variable (Q : ℝ → ℝ)
variable (a b c d m : ℝ)
variable (h1 : Q 1 = 5 * m)
variable (h0 : Q 0 = 2 * m)
variable (h_1 : Q (-1) = 6 * m)
variable (hQ : ∀ x, Q x = a * x^3 + b * x^2 + c * x + d)

theorem cubic_polynomial_value_at_3_and_neg3 :
  Q 3 + Q (-3) = 67 * m := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_cubic_polynomial_value_at_3_and_neg3_l1315_131561


namespace NUMINAMATH_GPT_problem1_l1315_131528

theorem problem1 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y > 2) : 
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 := 
sorry

end NUMINAMATH_GPT_problem1_l1315_131528


namespace NUMINAMATH_GPT_boat_distance_against_stream_l1315_131539

/-- 
  Given:
  1. The boat goes 13 km along the stream in one hour.
  2. The speed of the boat in still water is 11 km/hr.

  Prove:
  The distance the boat goes against the stream in one hour is 9 km.
-/
theorem boat_distance_against_stream (v_s : ℝ) (distance_along_stream time : ℝ) (v_still : ℝ) :
  distance_along_stream = 13 ∧ time = 1 ∧ v_still = 11 ∧ (v_still + v_s) = 13 → 
  (v_still - v_s) * time = 9 := by
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_l1315_131539


namespace NUMINAMATH_GPT_license_plate_combinations_l1315_131504

def number_of_license_plates : ℕ :=
  10^5 * 26^3 * 20

theorem license_plate_combinations :
  number_of_license_plates = 35152000000 := by
  -- Here's where the proof would go
  sorry

end NUMINAMATH_GPT_license_plate_combinations_l1315_131504


namespace NUMINAMATH_GPT_sum_of_star_tips_l1315_131553

theorem sum_of_star_tips :
  let n := 9
  let alpha := 80  -- in degrees
  let total := n * alpha
  total = 720 := by sorry

end NUMINAMATH_GPT_sum_of_star_tips_l1315_131553


namespace NUMINAMATH_GPT_circle_sum_condition_l1315_131533

theorem circle_sum_condition (n : ℕ) (n_ge_1 : n ≥ 1)
  (x : Fin n → ℝ) (sum_x : (Finset.univ.sum x) = n - 1) :
  ∃ j : Fin n, ∀ k : ℕ, k ≥ 1 → k ≤ n → (Finset.range k).sum (fun i => x ⟨(j + i) % n, sorry⟩) ≥ k - 1 :=
sorry

end NUMINAMATH_GPT_circle_sum_condition_l1315_131533


namespace NUMINAMATH_GPT_Richard_Orlando_ratio_l1315_131538

def Jenny_cards : ℕ := 6
def Orlando_more_cards : ℕ := 2
def Total_cards : ℕ := 38

theorem Richard_Orlando_ratio :
  let Orlando_cards := Jenny_cards + Orlando_more_cards
  let Richard_cards := Total_cards - (Jenny_cards + Orlando_cards)
  let ratio := Richard_cards / Orlando_cards
  ratio = 3 :=
by
  sorry

end NUMINAMATH_GPT_Richard_Orlando_ratio_l1315_131538


namespace NUMINAMATH_GPT_inequality_proof_l1315_131507

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b)) ≥ 1 / 2 * (a + b + c) := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1315_131507


namespace NUMINAMATH_GPT_frosting_sugar_l1315_131541

-- Define the conditions as constants
def total_sugar : ℝ := 0.8
def cake_sugar : ℝ := 0.2

-- The theorem stating that the sugar required for the frosting is 0.6 cups
theorem frosting_sugar : total_sugar - cake_sugar = 0.6 := by
  sorry

end NUMINAMATH_GPT_frosting_sugar_l1315_131541


namespace NUMINAMATH_GPT_pollutant_decay_l1315_131506

noncomputable def p (t : ℝ) (p0 : ℝ) := p0 * 2^(-t / 30)

theorem pollutant_decay : 
  ∃ p0 : ℝ, p0 = 300 ∧ p 60 p0 = 75 * Real.log 2 := 
by
  sorry

end NUMINAMATH_GPT_pollutant_decay_l1315_131506


namespace NUMINAMATH_GPT_inequality_correct_l1315_131534

theorem inequality_correct (a b : ℝ) (h1 : a > b) (h2 : b > 0) : (1 / a) < (1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_correct_l1315_131534


namespace NUMINAMATH_GPT_carrots_picked_by_mother_l1315_131583

-- Define the conditions
def faye_picked : ℕ := 23
def good_carrots : ℕ := 12
def bad_carrots : ℕ := 16

-- Define the problem of the total number of carrots
def total_carrots : ℕ := good_carrots + bad_carrots

-- Define the mother's picked carrots
def mother_picked (total_faye : ℕ) (total : ℕ) := total - total_faye

-- State the theorem
theorem carrots_picked_by_mother (faye_picked : ℕ) (total_carrots : ℕ) : mother_picked faye_picked total_carrots = 5 := by
  sorry

end NUMINAMATH_GPT_carrots_picked_by_mother_l1315_131583


namespace NUMINAMATH_GPT_penny_difference_l1315_131593

variables (p : ℕ)

/-- Liam and Mia have certain numbers of fifty-cent coins. This theorem proves the difference 
    in their total value in pennies. 
-/
theorem penny_difference:
  (3 * p + 2) * 50 - (2 * p + 7) * 50 = 50 * p - 250 :=
by
  sorry

end NUMINAMATH_GPT_penny_difference_l1315_131593


namespace NUMINAMATH_GPT_lcm_of_15_18_20_is_180_l1315_131558

theorem lcm_of_15_18_20_is_180 : Nat.lcm (Nat.lcm 15 18) 20 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_of_15_18_20_is_180_l1315_131558


namespace NUMINAMATH_GPT_tart_fill_l1315_131550

theorem tart_fill (cherries blueberries total : ℚ) (h_cherries : cherries = 0.08) (h_blueberries : blueberries = 0.75) (h_total : total = 0.91) :
  total - (cherries + blueberries) = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_tart_fill_l1315_131550


namespace NUMINAMATH_GPT_stratified_sampling_female_students_l1315_131577

-- Definitions from conditions
def male_students : ℕ := 800
def female_students : ℕ := 600
def drawn_male_students : ℕ := 40
def total_students : ℕ := 1400

-- Proof statement
theorem stratified_sampling_female_students : 
  (female_students * drawn_male_students) / male_students = 30 :=
by
  -- substitute and simplify
  sorry

end NUMINAMATH_GPT_stratified_sampling_female_students_l1315_131577


namespace NUMINAMATH_GPT_difference_length_width_l1315_131532

-- Definition of variables and conditions
variables (L W : ℝ)
def hall_width_half_length : Prop := W = (1/2) * L
def hall_area_578 : Prop := L * W = 578

-- Theorem to prove the desired result
theorem difference_length_width (h1 : hall_width_half_length L W) (h2 : hall_area_578 L W) : L - W = 17 :=
sorry

end NUMINAMATH_GPT_difference_length_width_l1315_131532


namespace NUMINAMATH_GPT_total_eggs_l1315_131518

-- Define the number of eggs eaten in each meal
def breakfast_eggs : ℕ := 2
def lunch_eggs : ℕ := 3
def dinner_eggs : ℕ := 1

-- Prove the total number of eggs eaten is 6
theorem total_eggs : breakfast_eggs + lunch_eggs + dinner_eggs = 6 :=
by
  sorry

end NUMINAMATH_GPT_total_eggs_l1315_131518


namespace NUMINAMATH_GPT_probability_diagonals_intersect_hexagon_l1315_131524

theorem probability_diagonals_intersect_hexagon:
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2 -- Total number of diagonals in a convex polygon
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2 -- Total number of ways to choose 2 diagonals
  let non_principal_intersections := 3 * 6 -- Each of 6 non-principal diagonals intersects 3 others
  let principal_intersections := 4 * 3 -- Each of 3 principal diagonals intersects 4 others
  let total_intersections := (non_principal_intersections + principal_intersections) / 2 -- Correcting for double-counting
  let probability := total_intersections / total_pairs -- Probability of intersection inside the hexagon
  probability = 5 / 12 := by
  let n : ℕ := 6
  let total_diagonals := (n * (n - 3)) / 2
  let total_pairs := (total_diagonals * (total_diagonals - 1)) / 2
  let non_principal_intersections := 3 * 6
  let principal_intersections := 4 * 3
  let total_intersections := (non_principal_intersections + principal_intersections) / 2
  let probability := total_intersections / total_pairs
  have h : total_diagonals = 9 := by norm_num
  have h_pairs : total_pairs = 36 := by norm_num
  have h_intersections : total_intersections = 15 := by norm_num
  have h_prob : probability = 5 / 12 := by norm_num
  exact h_prob

end NUMINAMATH_GPT_probability_diagonals_intersect_hexagon_l1315_131524


namespace NUMINAMATH_GPT_negation_equivalence_l1315_131570

-- Declare the condition for real solutions of a quadratic equation
def has_real_solutions (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a * x + 1 = 0

-- Define the proposition p
def prop_p : Prop :=
  ∀ a : ℝ, a ≥ 0 → has_real_solutions a

-- Define the negation of p
def neg_prop_p : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ ¬ has_real_solutions a

-- The theorem stating the equivalence of p's negation to its formulated negation.
theorem negation_equivalence : neg_prop_p = ¬ prop_p := by
  sorry

end NUMINAMATH_GPT_negation_equivalence_l1315_131570


namespace NUMINAMATH_GPT_system_solution_l1315_131584

theorem system_solution (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0)
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_system_solution_l1315_131584


namespace NUMINAMATH_GPT_smallest_total_cashews_l1315_131594

noncomputable def first_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  (2 * c1) / 3 + c2 / 6 + (4 * c3) / 18

noncomputable def second_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + (4 * c3) / 18

noncomputable def third_monkey_final (c1 c2 c3 : ℕ) : ℕ :=
  c1 / 6 + (2 * c2) / 6 + c3 / 9

theorem smallest_total_cashews : ∃ (c1 c2 c3 : ℕ), ∃ y : ℕ,
  3 * y = first_monkey_final c1 c2 c3 ∧
  2 * y = second_monkey_final c1 c2 c3 ∧
  y = third_monkey_final c1 c2 c3 ∧
  c1 + c2 + c3 = 630 :=
sorry

end NUMINAMATH_GPT_smallest_total_cashews_l1315_131594


namespace NUMINAMATH_GPT_bianca_picture_books_shelves_l1315_131546

theorem bianca_picture_books_shelves (total_shelves : ℕ) (books_per_shelf : ℕ) (mystery_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 8 →
  mystery_shelves = 5 →
  total_books = 72 →
  total_shelves = (total_books - (mystery_shelves * books_per_shelf)) / books_per_shelf →
  total_shelves = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_bianca_picture_books_shelves_l1315_131546


namespace NUMINAMATH_GPT_least_number_divisible_remainder_l1315_131531

theorem least_number_divisible_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 := 
sorry

end NUMINAMATH_GPT_least_number_divisible_remainder_l1315_131531
