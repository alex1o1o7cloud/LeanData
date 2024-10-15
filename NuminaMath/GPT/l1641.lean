import Mathlib

namespace NUMINAMATH_GPT_chests_content_l1641_164160

-- Define the chests and their labels
inductive CoinContent where
  | gold : CoinContent
  | silver : CoinContent
  | copper : CoinContent

structure Chest where
  label : CoinContent
  contents : CoinContent

-- Given conditions and incorrect labels
def chest1 : Chest := { label := CoinContent.gold, contents := sorry }
def chest2 : Chest := { label := CoinContent.silver, contents := sorry }
def chest3 : Chest := { label := CoinContent.gold, contents := sorry }

-- The proof problem
theorem chests_content :
  chest1.contents ≠ CoinContent.gold ∧
  chest2.contents ≠ CoinContent.silver ∧
  chest3.contents ≠ CoinContent.gold ∨ chest3.contents ≠ CoinContent.silver →
  chest1.contents = CoinContent.silver ∧
  chest2.contents = CoinContent.gold ∧
  chest3.contents = CoinContent.copper := by
  sorry

end NUMINAMATH_GPT_chests_content_l1641_164160


namespace NUMINAMATH_GPT_sum_of_coefficients_l1641_164106

noncomputable def simplify (x : ℝ) : ℝ := 
  (x^3 + 11 * x^2 + 38 * x + 40) / (x + 3)

theorem sum_of_coefficients : 
  (∀ x : ℝ, (x ≠ -3) → (simplify x = x^2 + 8 * x + 14)) ∧
  (1 + 8 + 14 + -3 = 20) :=
by      
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1641_164106


namespace NUMINAMATH_GPT_inequality_of_positive_numbers_l1641_164130

theorem inequality_of_positive_numbers (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) : 
  (a^2 * b^2 + b^2 * c^2 + c^2 * a^2) / (a + b + c) ≥ a * b * c :=
sorry

end NUMINAMATH_GPT_inequality_of_positive_numbers_l1641_164130


namespace NUMINAMATH_GPT_largest_among_abc_l1641_164144

variable {a b c : ℝ}

theorem largest_among_abc 
  (hn1 : a < 0) 
  (hn2 : b < 0) 
  (hn3 : c < 0) 
  (h : (c / (a + b)) < (a / (b + c)) ∧ (a / (b + c)) < (b / (c + a))) : c > a ∧ c > b :=
by
  sorry

end NUMINAMATH_GPT_largest_among_abc_l1641_164144


namespace NUMINAMATH_GPT_total_apples_picked_l1641_164163

theorem total_apples_picked (benny_apples : ℕ) (dan_apples : ℕ) (h_benny : benny_apples = 2) (h_dan : dan_apples = 9) :
  benny_apples + dan_apples = 11 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_picked_l1641_164163


namespace NUMINAMATH_GPT_max_blocks_l1641_164121

theorem max_blocks (box_height box_width box_length : ℝ) 
  (typeA_height typeA_width typeA_length typeB_height typeB_width typeB_length : ℝ) 
  (h_box : box_height = 8) (w_box : box_width = 10) (l_box : box_length = 12) 
  (h_typeA : typeA_height = 3) (w_typeA : typeA_width = 2) (l_typeA : typeA_length = 4) 
  (h_typeB : typeB_height = 4) (w_typeB : typeB_width = 3) (l_typeB : typeB_length = 5) : 
  max (⌊box_height / typeA_height⌋ * ⌊box_width / typeA_width⌋ * ⌊box_length / typeA_length⌋)
      (⌊box_height / typeB_height⌋ * ⌊box_width / typeB_width⌋ * ⌊box_length / typeB_length⌋) = 30 := 
  by
  sorry

end NUMINAMATH_GPT_max_blocks_l1641_164121


namespace NUMINAMATH_GPT_jaco_payment_l1641_164105

theorem jaco_payment :
  let cost_shoes : ℝ := 74
  let cost_socks : ℝ := 2 * 2
  let cost_bag : ℝ := 42
  let total_cost_before_discount : ℝ := cost_shoes + cost_socks + cost_bag
  let discount_threshold : ℝ := 100
  let discount_rate : ℝ := 0.10
  let amount_exceeding_threshold : ℝ := total_cost_before_discount - discount_threshold
  let discount : ℝ := if amount_exceeding_threshold > 0 then discount_rate * amount_exceeding_threshold else 0
  let final_amount : ℝ := total_cost_before_discount - discount
  final_amount = 118 :=
by
  sorry

end NUMINAMATH_GPT_jaco_payment_l1641_164105


namespace NUMINAMATH_GPT_max_ab_l1641_164100

theorem max_ab (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a + 4 * b = 8) :
  ab ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_ab_l1641_164100


namespace NUMINAMATH_GPT_deepak_walking_speed_l1641_164111

noncomputable def speed_deepak (circumference: ℕ) (wife_speed_kmph: ℚ) (meet_time_min: ℚ) : ℚ :=
  let meet_time_hr := meet_time_min / 60
  let wife_speed_mpm := wife_speed_kmph * 1000 / 60
  let distance_wife := wife_speed_mpm * meet_time_min
  let distance_deepak := circumference - distance_wife
  let deepak_speed_mpm := distance_deepak / meet_time_min
  deepak_speed_mpm * 60 / 1000

theorem deepak_walking_speed
  (circumference: ℕ) 
  (wife_speed_kmph: ℚ)
  (meet_time_min: ℚ)
  (H1: circumference = 627)
  (H2: wife_speed_kmph = 3.75)
  (H3: meet_time_min = 4.56) :
  speed_deepak circumference wife_speed_kmph meet_time_min = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_deepak_walking_speed_l1641_164111


namespace NUMINAMATH_GPT_ratio_of_construction_paper_packs_l1641_164123

-- Definitions for conditions
def marie_glue_sticks : Nat := 15
def marie_construction_paper : Nat := 30
def allison_total_items : Nat := 28
def allison_additional_glue_sticks : Nat := 8

-- Define the main quantity to prove
def allison_glue_sticks : Nat := marie_glue_sticks + allison_additional_glue_sticks
def allison_construction_paper : Nat := allison_total_items - allison_glue_sticks

-- The ratio should be of type Rat or Nat
theorem ratio_of_construction_paper_packs : (marie_construction_paper : Nat) / allison_construction_paper = 6 / 1 := by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_ratio_of_construction_paper_packs_l1641_164123


namespace NUMINAMATH_GPT_cubic_polynomial_a_value_l1641_164191

theorem cubic_polynomial_a_value (a b c d y₁ y₂ : ℝ)
  (h₁ : y₁ = a + b + c + d)
  (h₂ : y₂ = -a + b - c + d)
  (h₃ : y₁ - y₂ = -8) : a = -4 :=
by
  sorry

end NUMINAMATH_GPT_cubic_polynomial_a_value_l1641_164191


namespace NUMINAMATH_GPT_angle_between_apothems_correct_l1641_164192

noncomputable def angle_between_apothems (n : ℕ) (α : ℝ) : ℝ :=
  2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2))

theorem angle_between_apothems_correct (n : ℕ) (α : ℝ) (h1 : 0 < n) (h2 : 0 < α) (h3 : α < 2 * Real.pi) :
  angle_between_apothems n α = 2 * Real.arcsin (Real.cos (Real.pi / n) * Real.tan (α / 2)) :=
by
  sorry

end NUMINAMATH_GPT_angle_between_apothems_correct_l1641_164192


namespace NUMINAMATH_GPT_function_increasing_l1641_164114

noncomputable def f (a x : ℝ) := x^2 + a * x + 1 / x

theorem function_increasing (a : ℝ) :
  (∀ x, (1 / 3) < x → 0 ≤ (2 * x + a - 1 / x^2)) → a ≥ 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_function_increasing_l1641_164114


namespace NUMINAMATH_GPT_sum_of_coefficients_l1641_164199

theorem sum_of_coefficients :
  (∃ a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 : ℤ,
    (1 - 2 * x)^9 = a_9 * x^9 + a_8 * x^8 + a_7 * x^7 + a_6 * x^6 + a_5 * x^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
    a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 = -2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l1641_164199


namespace NUMINAMATH_GPT_george_stickers_l1641_164142

theorem george_stickers :
  let bob_stickers := 12
  let tom_stickers := 3 * bob_stickers
  let dan_stickers := 2 * tom_stickers
  let george_stickers := 5 * dan_stickers
  george_stickers = 360 := by
  sorry

end NUMINAMATH_GPT_george_stickers_l1641_164142


namespace NUMINAMATH_GPT_find_c_l1641_164101

variables {α : Type*} [LinearOrderedField α]

def p (x : α) : α := 3 * x - 9
def q (x : α) (c : α) : α := 4 * x - c

-- We aim to prove that if p(q(3,c)) = 6, then c = 7
theorem find_c (c : α) : p (q 3 c) = 6 → c = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l1641_164101


namespace NUMINAMATH_GPT_solve_for_x_l1641_164196

-- Define the problem
def equation (x : ℝ) : Prop := x + 2 * x + 12 = 500 - (3 * x + 4 * x)

-- State the theorem that we want to prove
theorem solve_for_x : ∃ (x : ℝ), equation x ∧ x = 48.8 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1641_164196


namespace NUMINAMATH_GPT_find_x_plus_y_l1641_164197

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3005) (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 3004 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l1641_164197


namespace NUMINAMATH_GPT_remainder_when_12_plus_a_div_by_31_l1641_164174

open Int

theorem remainder_when_12_plus_a_div_by_31 (a : ℤ) (ha : 0 < a) (h : 17 * a % 31 = 1) : (12 + a) % 31 = 23 := by
  sorry

end NUMINAMATH_GPT_remainder_when_12_plus_a_div_by_31_l1641_164174


namespace NUMINAMATH_GPT_max_difference_and_max_value_of_multiple_of_5_l1641_164108

theorem max_difference_and_max_value_of_multiple_of_5:
  ∀ (N : ℕ), 
  (∃ (d : ℕ), d = 0 ∨ d = 5 ∧ N = 740 + d) →
  (∃ (diff : ℕ), diff = 5) ∧ (∃ (max_num : ℕ), max_num = 745) :=
by
  intro N
  rintro ⟨d, (rfl | rfl), rfl⟩
  apply And.intro
  use 5
  use 745
  sorry

end NUMINAMATH_GPT_max_difference_and_max_value_of_multiple_of_5_l1641_164108


namespace NUMINAMATH_GPT_book_price_l1641_164162

theorem book_price (x : ℕ) (h1 : x - 1 = 1 + (x - 1)) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_book_price_l1641_164162


namespace NUMINAMATH_GPT_movie_production_cost_l1641_164139

-- Definitions based on the conditions
def opening_revenue : ℝ := 120 -- in million dollars
def total_revenue : ℝ := 3.5 * opening_revenue -- movie made during its entire run
def kept_revenue : ℝ := 0.60 * total_revenue -- production company keeps 60% of total revenue
def profit : ℝ := 192 -- in million dollars

-- Theorem stating the cost to produce the movie
theorem movie_production_cost : 
  (kept_revenue - 60) = profit :=
by
  sorry

end NUMINAMATH_GPT_movie_production_cost_l1641_164139


namespace NUMINAMATH_GPT_economist_winning_strategy_l1641_164186

-- Conditions setup
variables {n a b x1 x2 y1 y2 : ℕ}

-- Definitions according to the conditions
def valid_initial_division (n a b : ℕ) : Prop :=
  n > 4 ∧ n % 2 = 1 ∧ 2 ≤ a ∧ 2 ≤ b ∧ a + b = n ∧ a < b

def valid_further_division (a b x1 x2 y1 y2 : ℕ) : Prop :=
  x1 + x2 = a ∧ x1 ≥ 1 ∧ x2 ≥ 1 ∧ y1 + y2 = b ∧ y1 ≥ 1 ∧ y2 ≥ 1 ∧ x1 ≤ x2 ∧ y1 ≤ y2

-- Methods defined: Assumptions about which parts the economist takes
def method_1 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max x2 y2 + min x1 y1

def method_2 (x1 x2 y1 y2 : ℕ) : ℕ :=
  (x1 + y1) / 2 + (x2 + y2) / 2

def method_3 (x1 x2 y1 y2 : ℕ) : ℕ :=
  max (method_1 x1 x2 y1 y2 - 1) (method_2 x1 x2 y1 y2 - 1) + 1

-- The statement to prove that the economist would choose method 1
theorem economist_winning_strategy :
  ∀ n a b x1 x2 y1 y2,
    valid_initial_division n a b →
    valid_further_division a b x1 x2 y1 y2 →
    n > 4 → n % 2 = 1 →
    (method_1 x1 x2 y1 y2) > (method_2 x1 x2 y1 y2) →
    (method_1 x1 x2 y1 y2) > (method_3 x1 x2 y1 y2) →
    method_1 x1 x2 y1 y2 = max (method_1 x1 x2 y1 y2) (method_2 x1 x2 y1 y2) :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_economist_winning_strategy_l1641_164186


namespace NUMINAMATH_GPT_number_of_B_eq_l1641_164168

variable (a b : ℝ)
variable (B : ℝ)

theorem number_of_B_eq : 3 * B = a + b → B = (a + b) / 3 :=
by sorry

end NUMINAMATH_GPT_number_of_B_eq_l1641_164168


namespace NUMINAMATH_GPT_mean_of_samantha_scores_l1641_164147

noncomputable def arithmetic_mean (l : List ℝ) : ℝ := l.sum / l.length

theorem mean_of_samantha_scores :
  arithmetic_mean [93, 87, 90, 96, 88, 94] = 91.333 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_samantha_scores_l1641_164147


namespace NUMINAMATH_GPT_find_x_parallel_l1641_164190

def m : ℝ × ℝ := (-2, 4)
def n (x : ℝ) : ℝ × ℝ := (x, -1)

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u.1 = k * v.1 ∧ u.2 = k * v.2

theorem find_x_parallel :
  parallel m (n x) → x = 1 / 2 := by 
sorry

end NUMINAMATH_GPT_find_x_parallel_l1641_164190


namespace NUMINAMATH_GPT_Megan_total_earnings_two_months_l1641_164189

-- Define the conditions
def hours_per_day : ℕ := 8
def wage_per_hour : ℝ := 7.50
def days_per_month : ℕ := 20

-- Define the main question and correct answer
theorem Megan_total_earnings_two_months : 
  (2 * (days_per_month * (hours_per_day * wage_per_hour))) = 2400 := 
by
  -- In the problem statement, we are given conditions so we just state sorry because the focus is on the statement, not the solution steps.
  sorry

end NUMINAMATH_GPT_Megan_total_earnings_two_months_l1641_164189


namespace NUMINAMATH_GPT_sandy_paid_for_pants_l1641_164113

-- Define the costs and change as constants
def cost_of_shirt : ℝ := 8.25
def amount_paid_with : ℝ := 20.00
def change_received : ℝ := 2.51

-- Define the amount paid for pants
def amount_paid_for_pants : ℝ := 9.24

-- The theorem stating the problem
theorem sandy_paid_for_pants : 
  amount_paid_with - (cost_of_shirt + change_received) = amount_paid_for_pants := 
by 
  -- proof is required here
  sorry

end NUMINAMATH_GPT_sandy_paid_for_pants_l1641_164113


namespace NUMINAMATH_GPT_calculation_is_zero_l1641_164151

theorem calculation_is_zero : 
  20062006 * 2007 + 20072007 * 2008 - 2006 * 20072007 - 2007 * 20082008 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_calculation_is_zero_l1641_164151


namespace NUMINAMATH_GPT_complement_A_eq_B_subset_complement_A_l1641_164124

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 + 4 * x > 0 }
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1 }

-- The universal set U is the set of all real numbers
def U : Set ℝ := Set.univ

-- Complement of A in U
def complement_U_A : Set ℝ := {x | -4 ≤ x ∧ x ≤ 0}

-- Proof statement for part (1)
theorem complement_A_eq : complement_U_A = {x | -4 ≤ x ∧ x ≤ 0} :=
  sorry 

-- Proof statement for part (2)
theorem B_subset_complement_A (a : ℝ) : B a ⊆ complement_U_A ↔ -3 ≤ a ∧ a ≤ -1 :=
  sorry 

end NUMINAMATH_GPT_complement_A_eq_B_subset_complement_A_l1641_164124


namespace NUMINAMATH_GPT_actual_diameter_of_tissue_l1641_164148

variable (magnified_diameter : ℝ) (magnification_factor : ℝ)

theorem actual_diameter_of_tissue 
    (h1 : magnified_diameter = 0.2) 
    (h2 : magnification_factor = 1000) : 
    magnified_diameter / magnification_factor = 0.0002 := 
  by
    sorry

end NUMINAMATH_GPT_actual_diameter_of_tissue_l1641_164148


namespace NUMINAMATH_GPT_complement_union_l1641_164183

-- Definitions of sets A and B based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≤ 0}

def B : Set ℝ := {x | x ≥ 1}

-- Theorem to prove the complement of the union of sets A and B within U
theorem complement_union (x : ℝ) : x ∉ (A ∪ B) ↔ (0 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_GPT_complement_union_l1641_164183


namespace NUMINAMATH_GPT_find_s_of_2_l1641_164149

def t (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) : ℝ := x^2 + 4 * x - 1

theorem find_s_of_2 : s (2) = 281 / 16 :=
by
  sorry

end NUMINAMATH_GPT_find_s_of_2_l1641_164149


namespace NUMINAMATH_GPT_smallest_perfect_square_4_10_18_l1641_164127

theorem smallest_perfect_square_4_10_18 :
  ∃ n : ℕ, (∃ k : ℕ, n = k^2) ∧ (4 ∣ n) ∧ (10 ∣ n) ∧ (18 ∣ n) ∧ n = 900 := 
  sorry

end NUMINAMATH_GPT_smallest_perfect_square_4_10_18_l1641_164127


namespace NUMINAMATH_GPT_first_term_of_new_ratio_l1641_164141

-- Given conditions as definitions
def original_ratio : ℚ := 6 / 7
def x (n : ℕ) : Prop := n ≥ 3

-- Prove that the first term of the ratio that the new ratio should be less than is 4
theorem first_term_of_new_ratio (n : ℕ) (h1 : x n) : ∃ b, (6 - n) / (7 - n) < 4 / b :=
by
  exists 5
  sorry

end NUMINAMATH_GPT_first_term_of_new_ratio_l1641_164141


namespace NUMINAMATH_GPT_integer_solutions_of_polynomial_l1641_164182

theorem integer_solutions_of_polynomial :
  ∀ n : ℤ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = -1 ∨ n = 3 := 
by 
  sorry

end NUMINAMATH_GPT_integer_solutions_of_polynomial_l1641_164182


namespace NUMINAMATH_GPT_sin_pi_plus_alpha_l1641_164145

open Real

-- Define the given conditions
variable (α : ℝ) (hα1 : sin (π / 2 + α) = 3 / 5) (hα2 : 0 < α ∧ α < π / 2)

-- The theorem statement that must be proved
theorem sin_pi_plus_alpha : sin (π + α) = -4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sin_pi_plus_alpha_l1641_164145


namespace NUMINAMATH_GPT_interest_rate_difference_l1641_164152

theorem interest_rate_difference:
  ∀ (R H: ℝ),
    (300 * (H / 100) * 5 = 300 * (R / 100) * 5 + 90) →
    (H - R = 6) :=
by
  intros R H h
  sorry

end NUMINAMATH_GPT_interest_rate_difference_l1641_164152


namespace NUMINAMATH_GPT_compute_expression_l1641_164115

theorem compute_expression : 3034 - (1002 / 200.4) = 3029 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l1641_164115


namespace NUMINAMATH_GPT_ratio_of_age_difference_l1641_164154

theorem ratio_of_age_difference (R J K : ℕ) 
  (h1 : R = J + 6) 
  (h2 : R + 4 = 2 * (J + 4)) 
  (h3 : (R + 4) * (K + 4) = 108) : 
  (R - J) / (R - K) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_age_difference_l1641_164154


namespace NUMINAMATH_GPT_sum_of_square_and_divisor_not_square_l1641_164187

theorem sum_of_square_and_divisor_not_square {A B : ℕ} (hA : A ≠ 0) (hA_square : ∃ k : ℕ, A = k * k) (hB_divisor : B ∣ A) : ¬ (∃ m : ℕ, A + B = m * m) := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sum_of_square_and_divisor_not_square_l1641_164187


namespace NUMINAMATH_GPT_sqrt_of_square_neg_l1641_164131

variable {a : ℝ}

theorem sqrt_of_square_neg (h : a < 0) : Real.sqrt (a^2) = -a := 
sorry

end NUMINAMATH_GPT_sqrt_of_square_neg_l1641_164131


namespace NUMINAMATH_GPT_midpoint_to_plane_distance_l1641_164107

noncomputable def distance_to_plane (A B P: ℝ) (dA dB: ℝ) : ℝ :=
if h : A = B then |dA|
else if h1 : dA + dB = (2 : ℝ) * (dA + dB) / 2 then (dA + dB) / 2
else if h2 : |dB - dA| = (2 : ℝ) * |dB - dA| / 2 then |dB - dA| / 2
else 0

theorem midpoint_to_plane_distance
  (α : Type*)
  (A B P: ℝ)
  {dA dB : ℝ}
  (h_dA : dA = 3)
  (h_dB : dB = 5) :
  distance_to_plane A B P dA dB = 4 ∨ distance_to_plane A B P dA dB = 1 :=
by sorry

end NUMINAMATH_GPT_midpoint_to_plane_distance_l1641_164107


namespace NUMINAMATH_GPT_disjoint_subsets_mod_1000_l1641_164159

open Nat

theorem disjoint_subsets_mod_1000 :
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  m % 1000 = 625 := 
by
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  have : m % 1000 = 625 := sorry
  exact this

end NUMINAMATH_GPT_disjoint_subsets_mod_1000_l1641_164159


namespace NUMINAMATH_GPT_Aiyanna_has_more_cookies_l1641_164198

theorem Aiyanna_has_more_cookies (cookies_Alyssa : ℕ) (cookies_Aiyanna : ℕ) (h1 : cookies_Alyssa = 129) (h2 : cookies_Aiyanna = cookies_Alyssa + 11) : cookies_Aiyanna = 140 := by
  sorry

end NUMINAMATH_GPT_Aiyanna_has_more_cookies_l1641_164198


namespace NUMINAMATH_GPT_rectangular_plot_area_l1641_164156

theorem rectangular_plot_area (Breadth Length Area : ℕ): 
  (Length = 3 * Breadth) → 
  (Breadth = 30) → 
  (Area = Length * Breadth) → 
  Area = 2700 :=
by 
  intros h_length h_breadth h_area
  rw [h_breadth] at h_length
  rw [h_length, h_breadth] at h_area
  exact h_area

end NUMINAMATH_GPT_rectangular_plot_area_l1641_164156


namespace NUMINAMATH_GPT_students_suggested_bacon_l1641_164184

-- Defining the conditions
def total_students := 310
def mashed_potatoes_students := 185

-- Lean statement for proving the equivalent problem
theorem students_suggested_bacon : total_students - mashed_potatoes_students = 125 := by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_students_suggested_bacon_l1641_164184


namespace NUMINAMATH_GPT_tangent_line_slope_l1641_164194

/-- Given the line y = mx is tangent to the circle x^2 + y^2 - 4x + 2 = 0, 
    the slope m must be ±1. -/
theorem tangent_line_slope (m : ℝ) :
  (∃ x y : ℝ, y = m * x ∧ (x ^ 2 + y ^ 2 - 4 * x + 2 = 0)) →
  (m = 1 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_slope_l1641_164194


namespace NUMINAMATH_GPT_remainder_1493827_div_4_l1641_164188

theorem remainder_1493827_div_4 : 1493827 % 4 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_1493827_div_4_l1641_164188


namespace NUMINAMATH_GPT_find_g_5_l1641_164109

variable (g : ℝ → ℝ)

axiom func_eqn : ∀ x y : ℝ, x * g y = y * g x
axiom g_10 : g 10 = 15

theorem find_g_5 : g 5 = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_g_5_l1641_164109


namespace NUMINAMATH_GPT_equal_contribution_expense_split_l1641_164181

theorem equal_contribution_expense_split (Mitch_expense Jam_expense Jay_expense Jordan_expense total_expense each_contribution : ℕ)
  (hmitch : Mitch_expense = 4 * 7)
  (hjam : Jam_expense = (2 * 15) / 10 + 4) -- note: 1.5 dollar per box interpreted as 15/10 to avoid float in Lean
  (hjay : Jay_expense = 3 * 3)
  (hjordan : Jordan_expense = 4 * 2)
  (htotal : total_expense = Mitch_expense + Jam_expense + Jay_expense + Jordan_expense)
  (hequal_split : each_contribution = total_expense / 4) :
  each_contribution = 13 :=
by
  sorry

end NUMINAMATH_GPT_equal_contribution_expense_split_l1641_164181


namespace NUMINAMATH_GPT_tan_315_eq_neg1_l1641_164155

theorem tan_315_eq_neg1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_315_eq_neg1_l1641_164155


namespace NUMINAMATH_GPT_solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l1641_164129

def solve_inequality (a x : ℝ) : Prop :=
  a^2 * x - 6 < 4 * x + 3 * a

theorem solution_set_a_eq_2 :
  ∀ x : ℝ, solve_inequality 2 x ↔ true :=
sorry

theorem solution_set_a_eq_neg_2 :
  ∀ x : ℝ, ¬ solve_inequality (-2) x :=
sorry

theorem solution_set_neg_2_lt_a_lt_2 (a : ℝ) (h : -2 < a ∧ a < 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x > 3 / (a - 2) :=
sorry

theorem solution_set_a_lt_neg_2_or_a_gt_2 (a : ℝ) (h : a < -2 ∨ a > 2) :
  ∀ x : ℝ, solve_inequality a x ↔ x < 3 / (a - 2) :=
sorry

end NUMINAMATH_GPT_solution_set_a_eq_2_solution_set_a_eq_neg_2_solution_set_neg_2_lt_a_lt_2_solution_set_a_lt_neg_2_or_a_gt_2_l1641_164129


namespace NUMINAMATH_GPT_radian_measure_of_neg_300_degrees_l1641_164132

theorem radian_measure_of_neg_300_degrees : (-300 : ℝ) * (Real.pi / 180) = -5 * Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_radian_measure_of_neg_300_degrees_l1641_164132


namespace NUMINAMATH_GPT_probability_top_card_heart_l1641_164175

def specially_designed_deck (n_cards n_ranks n_suits cards_per_suit : ℕ) : Prop :=
  n_cards = 60 ∧ n_ranks = 15 ∧ n_suits = 4 ∧ cards_per_suit = n_ranks

theorem probability_top_card_heart (n_cards n_ranks n_suits cards_per_suit : ℕ)
  (h_deck : specially_designed_deck n_cards n_ranks n_suits cards_per_suit) :
  (15 / 60 : ℝ) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_top_card_heart_l1641_164175


namespace NUMINAMATH_GPT_operation_B_correct_operation_C_correct_l1641_164136

theorem operation_B_correct (x y : ℝ) : (-3 * x * y) ^ 2 = 9 * x ^ 2 * y ^ 2 :=
  sorry

theorem operation_C_correct (x y : ℝ) (h : x ≠ y) : 
  (x - y) / (2 * x * y - x ^ 2 - y ^ 2) = 1 / (y - x) :=
  sorry

end NUMINAMATH_GPT_operation_B_correct_operation_C_correct_l1641_164136


namespace NUMINAMATH_GPT_zachary_additional_money_needed_l1641_164164

noncomputable def total_cost : ℝ := 3.756 + 2 * 2.498 + 11.856 + 4 * 1.329 + 7.834
noncomputable def zachary_money : ℝ := 24.042
noncomputable def money_needed : ℝ := total_cost - zachary_money

theorem zachary_additional_money_needed : money_needed = 9.716 := 
by 
  sorry

end NUMINAMATH_GPT_zachary_additional_money_needed_l1641_164164


namespace NUMINAMATH_GPT_arithmetic_sequence_of_condition_l1641_164169

variables {R : Type*} [LinearOrderedRing R]

theorem arithmetic_sequence_of_condition (x y z : R) (h : (z-x)^2 - 4*(x-y)*(y-z) = 0) : 2*y = x + z :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_of_condition_l1641_164169


namespace NUMINAMATH_GPT_range_of_x_in_second_quadrant_l1641_164153

theorem range_of_x_in_second_quadrant (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_in_second_quadrant_l1641_164153


namespace NUMINAMATH_GPT_max_area_of_triangle_l1641_164119

-- Define the problem conditions and the maximum area S
theorem max_area_of_triangle
  (A B C : ℝ)
  (a b c S : ℝ)
  (h1 : 4 * S = a^2 - (b - c)^2)
  (h2 : b + c = 8) :
  S ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_area_of_triangle_l1641_164119


namespace NUMINAMATH_GPT_min_cost_of_packaging_l1641_164176

def packaging_problem : Prop :=
  ∃ (x y : ℕ), 35 * x + 24 * y = 106 ∧ 140 * x + 120 * y = 500

theorem min_cost_of_packaging : packaging_problem :=
sorry

end NUMINAMATH_GPT_min_cost_of_packaging_l1641_164176


namespace NUMINAMATH_GPT_sale_in_fourth_month_l1641_164133

-- Given conditions
def sales_first_month : ℕ := 5266
def sales_second_month : ℕ := 5768
def sales_third_month : ℕ := 5922
def sales_sixth_month : ℕ := 4937
def required_average_sales : ℕ := 5600
def number_of_months : ℕ := 6

-- Sum of the first, second, third, and sixth month's sales
def total_sales_without_fourth_fifth : ℕ := sales_first_month + sales_second_month + sales_third_month + sales_sixth_month

-- Total sales required to achieve the average required
def required_total_sales : ℕ := required_average_sales * number_of_months

-- The sale in the fourth month should be calculated as follows
def sales_fourth_month : ℕ := required_total_sales - total_sales_without_fourth_fifth

-- Proof statement
theorem sale_in_fourth_month :
  sales_fourth_month = 11707 := by
  sorry

end NUMINAMATH_GPT_sale_in_fourth_month_l1641_164133


namespace NUMINAMATH_GPT_max_value_expression_l1641_164193

theorem max_value_expression (x y : ℝ) (h : x * y > 0) : 
  ∃ (max_val : ℝ), max_val = 4 - 2 * Real.sqrt 2 ∧ 
  (∀ a b : ℝ, a * b > 0 → (a / (a + b) + 2 * b / (a + 2 * b)) ≤ max_val) := 
sorry

end NUMINAMATH_GPT_max_value_expression_l1641_164193


namespace NUMINAMATH_GPT_regular_milk_cartons_l1641_164137

variable (R C : ℕ)
variable (h1 : C + R = 24)
variable (h2 : C = 7 * R)

theorem regular_milk_cartons : R = 3 :=
by
  sorry

end NUMINAMATH_GPT_regular_milk_cartons_l1641_164137


namespace NUMINAMATH_GPT_cone_height_l1641_164185

theorem cone_height (r : ℝ) (n : ℕ) (circumference : ℝ) 
  (sector_circumference : ℝ) (base_radius : ℝ) (slant_height : ℝ) 
  (h : ℝ) : 
  r = 8 →
  n = 4 →
  circumference = 2 * Real.pi * r →
  sector_circumference = circumference / n →
  base_radius = sector_circumference / (2 * Real.pi) →
  slant_height = r →
  h = Real.sqrt (slant_height^2 - base_radius^2) →
  h = 2 * Real.sqrt 15 := 
by
  intros
  sorry

end NUMINAMATH_GPT_cone_height_l1641_164185


namespace NUMINAMATH_GPT_fraction_half_l1641_164116

theorem fraction_half {A : ℕ} (h : 8 * (A + 8) - 8 * (A - 8) = 128) (age_eq : A = 64) :
  (64 : ℚ) / (128 : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_half_l1641_164116


namespace NUMINAMATH_GPT_union_of_M_and_N_l1641_164122

namespace SetOperations

def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {1, 3, 4}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3, 4} :=
sorry

end SetOperations

end NUMINAMATH_GPT_union_of_M_and_N_l1641_164122


namespace NUMINAMATH_GPT_equation_of_circle_l1641_164173

-- Defining the problem conditions directly
variables (a : ℝ) (x y: ℝ)

-- Assume a ≠ 0
variable (h : a ≠ 0)

-- Prove that the circle passing through the origin with center (a, a) has the equation (x - a)^2 + (y - a)^2 = 2a^2.
theorem equation_of_circle (h : a ≠ 0) :
  (x - a)^2 + (y - a)^2 = 2 * a^2 :=
sorry

end NUMINAMATH_GPT_equation_of_circle_l1641_164173


namespace NUMINAMATH_GPT_solve_inequality_l1641_164157

theorem solve_inequality (x : ℝ) : 
  2 ≤ |x - 3| ∧ |x - 3| ≤ 8 ↔ (-5 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 11) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1641_164157


namespace NUMINAMATH_GPT_total_ticket_cost_l1641_164120

theorem total_ticket_cost :
  ∀ (A : ℝ), 
  -- Conditions
  (6 : ℝ) * (5 : ℝ) + (2 : ℝ) * A = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_ticket_cost_l1641_164120


namespace NUMINAMATH_GPT_paint_canvas_cost_ratio_l1641_164150

theorem paint_canvas_cost_ratio (C P : ℝ) (hc : 0.6 * C = C - 0.4 * C) (hp : 0.4 * P = P - 0.6 * P)
 (total_cost_reduction : 0.4 * P + 0.6 * C = 0.44 * (P + C)) :
  P / C = 4 :=
by
  sorry

end NUMINAMATH_GPT_paint_canvas_cost_ratio_l1641_164150


namespace NUMINAMATH_GPT_factor_tree_value_l1641_164112

theorem factor_tree_value :
  ∀ (X Y Z F G : ℕ),
  X = Y * Z → 
  Y = 7 * F → 
  F = 2 * 5 → 
  Z = 11 * G → 
  G = 7 * 3 → 
  X = 16170 := 
by
  intros X Y Z F G
  sorry

end NUMINAMATH_GPT_factor_tree_value_l1641_164112


namespace NUMINAMATH_GPT_last_score_entered_is_75_l1641_164143

theorem last_score_entered_is_75 (scores : List ℕ) (h : scores = [62, 75, 83, 90]) :
  ∃ last_score, last_score ∈ scores ∧ 
    (∀ (num list : List ℕ), list ≠ [] → list.length ≤ scores.length → 
    ¬ list.sum % list.length ≠ 0) → 
  last_score = 75 :=
by
  sorry

end NUMINAMATH_GPT_last_score_entered_is_75_l1641_164143


namespace NUMINAMATH_GPT_cubic_solution_unique_real_l1641_164172

theorem cubic_solution_unique_real (x : ℝ) : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3 → x = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_cubic_solution_unique_real_l1641_164172


namespace NUMINAMATH_GPT_area_of_triangle_LMN_l1641_164177

-- Define the vertices
def point := ℝ × ℝ
def L: point := (2, 3)
def M: point := (5, 1)
def N: point := (3, 5)

-- Shoelace formula for the area of a triangle
noncomputable def triangle_area (A B C : point) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2))

-- Statement to prove the area
theorem area_of_triangle_LMN : triangle_area L M N = 4 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_area_of_triangle_LMN_l1641_164177


namespace NUMINAMATH_GPT_points_per_correct_answer_hard_round_l1641_164178

theorem points_per_correct_answer_hard_round (total_points easy_points_per average_points_per hard_correct : ℕ) 
(easy_correct average_correct : ℕ) : 
  (total_points = (easy_correct * easy_points_per + average_correct * average_points_per) + (hard_correct * 5)) →
  (easy_correct = 6) →
  (easy_points_per = 2) →
  (average_correct = 2) →
  (average_points_per = 3) →
  (hard_correct = 4) →
  (total_points = 38) →
  5 = 5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_points_per_correct_answer_hard_round_l1641_164178


namespace NUMINAMATH_GPT_honor_students_count_l1641_164138

noncomputable def number_of_students_in_class_is_less_than_30 := ∃ n, n < 30
def probability_girl_honor_student (G E_G : ℕ) := E_G / G = (3 : ℚ) / 13
def probability_boy_honor_student (B E_B : ℕ) := E_B / B = (4 : ℚ) / 11

theorem honor_students_count (G B E_G E_B : ℕ) 
  (hG_cond : probability_girl_honor_student G E_G) 
  (hB_cond : probability_boy_honor_student B E_B) 
  (h_total_students : G + B < 30) 
  (hE_G_def : E_G = 3 * G / 13) 
  (hE_B_def : E_B = 4 * B / 11) 
  (hG_nonneg : G >= 13)
  (hB_nonneg : B >= 11):
  E_G + E_B = 7 := 
sorry

end NUMINAMATH_GPT_honor_students_count_l1641_164138


namespace NUMINAMATH_GPT_seats_shortage_l1641_164128

-- Definitions of the conditions
def children := 52
def adults := 29
def seniors := 15
def pets := 3
def total_seats := 95

-- Theorem statement to prove the number of people and pets without seats
theorem seats_shortage : children + adults + seniors + pets - total_seats = 4 :=
by
  sorry

end NUMINAMATH_GPT_seats_shortage_l1641_164128


namespace NUMINAMATH_GPT_rectangle_area_correct_l1641_164179

theorem rectangle_area_correct (l r s : ℝ) (b : ℝ := 10) (h1 : l = (1 / 4) * r) (h2 : r = s) (h3 : s^2 = 1225) :
  l * b = 87.5 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_correct_l1641_164179


namespace NUMINAMATH_GPT_village_connection_possible_l1641_164195

variable (V : Type) -- Type of villages
variable (Villages : List V) -- List of 26 villages
variable (connected_by_tractor connected_by_train : V → V → Prop) -- Connections

-- Define the hypothesis
variable (bidirectional_connections : ∀ (v1 v2 : V), v1 ≠ v2 → (connected_by_tractor v1 v2 ∨ connected_by_train v1 v2))

-- Main theorem statement
theorem village_connection_possible :
  ∃ (mode : V → V → Prop), (∀ v1 v2 : V, v1 ≠ v2 → v1 ∈ Villages → v2 ∈ Villages → mode v1 v2) ∧
  (∀ v1 v2 : V, v1 ∈ Villages → v2 ∈ Villages → ∃ (path : List (V × V)), (∀ edge ∈ path, mode edge.fst edge.snd) ∧ path ≠ []) :=
by
  sorry

end NUMINAMATH_GPT_village_connection_possible_l1641_164195


namespace NUMINAMATH_GPT_MinValue_x3y2z_l1641_164110

theorem MinValue_x3y2z (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h : 1/x + 1/y + 1/z = 6) : x^3 * y^2 * z ≥ 1 / 108 :=
by
  sorry

end NUMINAMATH_GPT_MinValue_x3y2z_l1641_164110


namespace NUMINAMATH_GPT_planted_fraction_l1641_164103

theorem planted_fraction (a b c : ℕ) (x h : ℝ) 
  (h_right_triangle : a = 5 ∧ b = 12)
  (h_hypotenuse : c = 13)
  (h_square_dist : x = 3) : 
  (h * ((a * b) - (x^2))) / (a * b / 2) = (7 : ℝ) / 10 :=
by
  sorry

end NUMINAMATH_GPT_planted_fraction_l1641_164103


namespace NUMINAMATH_GPT_final_configuration_l1641_164170

def initial_configuration : (String × String) :=
  ("bottom-right", "bottom-left")

def first_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("bottom-right", "bottom-left") => ("top-right", "top-left")
  | _ => conf

def second_transformation (conf : (String × String)) : (String × String) :=
  match conf with
  | ("top-right", "top-left") => ("top-left", "top-right")
  | _ => conf

theorem final_configuration :
  second_transformation (first_transformation initial_configuration) =
  ("top-left", "top-right") :=
by
  sorry

end NUMINAMATH_GPT_final_configuration_l1641_164170


namespace NUMINAMATH_GPT_q_can_do_work_in_10_days_l1641_164134

theorem q_can_do_work_in_10_days (R_p R_q R_pq: ℝ)
  (h1 : R_p = 1 / 15)
  (h2 : R_pq = 1 / 6)
  (h3 : R_p + R_q = R_pq) :
  1 / R_q = 10 :=
by
  -- Proof steps go here.
  sorry

end NUMINAMATH_GPT_q_can_do_work_in_10_days_l1641_164134


namespace NUMINAMATH_GPT_sum_of_percentages_l1641_164140

theorem sum_of_percentages : 
  let x := 80 + (0.2 * 80)
  let y := 60 - (0.3 * 60)
  let z := 40 + (0.5 * 40)
  x + y + z = 198 := by
  sorry

end NUMINAMATH_GPT_sum_of_percentages_l1641_164140


namespace NUMINAMATH_GPT_machines_needed_l1641_164146

theorem machines_needed (original_machines : ℕ) (original_days : ℕ) (additional_machines : ℕ) :
  original_machines = 12 → original_days = 40 → 
  additional_machines = ((original_machines * original_days) / (3 * original_days / 4)) - original_machines →
  additional_machines = 4 :=
by
  intros h_machines h_days h_additional
  rw [h_machines, h_days] at h_additional
  sorry

end NUMINAMATH_GPT_machines_needed_l1641_164146


namespace NUMINAMATH_GPT_cubic_identity_l1641_164158

theorem cubic_identity (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end NUMINAMATH_GPT_cubic_identity_l1641_164158


namespace NUMINAMATH_GPT_hike_up_time_eq_l1641_164118

variable (t : ℝ)
variable (h_rate_up : ℝ := 4)
variable (h_rate_down : ℝ := 6)
variable (total_time : ℝ := 3)

theorem hike_up_time_eq (h_rate_up_eq : h_rate_up = 4) 
                        (h_rate_down_eq : h_rate_down = 6) 
                        (total_time_eq : total_time = 3) 
                        (dist_eq : h_rate_up * t = h_rate_down * (total_time - t)) :
  t = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_hike_up_time_eq_l1641_164118


namespace NUMINAMATH_GPT_shortest_altitude_l1641_164171

theorem shortest_altitude (a b c : ℝ) (h : a = 9 ∧ b = 12 ∧ c = 15) (h_right : a^2 + b^2 = c^2) : 
  ∃ x : ℝ, x = 7.2 ∧ (1/2) * c * x = (1/2) * a * b := 
by
  sorry

end NUMINAMATH_GPT_shortest_altitude_l1641_164171


namespace NUMINAMATH_GPT_factorize_poly_l1641_164180

open Polynomial

theorem factorize_poly : 
  (X ^ 15 + X ^ 7 + 1 : Polynomial ℤ) =
    (X^2 + X + 1) * (X^13 - X^12 + X^10 - X^9 + X^7 - X^6 + X^4 - X^3 + X - 1) := 
  by
  sorry

end NUMINAMATH_GPT_factorize_poly_l1641_164180


namespace NUMINAMATH_GPT_yellow_balls_l1641_164117

theorem yellow_balls (total_balls : ℕ) (prob_yellow : ℚ) (x : ℕ) :
  total_balls = 40 ∧ prob_yellow = 0.30 → (x : ℚ) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_yellow_balls_l1641_164117


namespace NUMINAMATH_GPT_range_of_a_l1641_164135

variable {f : ℝ → ℝ}
variable {a : ℝ}

-- Define the conditions given:
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

def monotonic_increasing_on_nonnegative_reals (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, (0 ≤ x1) → (0 ≤ x2) → (x1 < x2) → (f x1 < f x2)

def inequality_in_interval (f : ℝ → ℝ) (a : ℝ) :=
  ∀ x, (1 / 2 ≤ x) → (x ≤ 1) → (f (a * x + 1) ≤ f (x - 2))

-- The theorem we want to prove
theorem range_of_a (h1 : even_function f)
                   (h2 : monotonic_increasing_on_nonnegative_reals f)
                   (h3 : inequality_in_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := sorry

end NUMINAMATH_GPT_range_of_a_l1641_164135


namespace NUMINAMATH_GPT_checkerboard_probability_l1641_164161

-- Define the number of squares in the checkerboard and the number on the perimeter
def total_squares : Nat := 10 * 10
def perimeter_squares : Nat := 10 + 10 + (10 - 2) + (10 - 2)

-- The number of squares not on the perimeter
def inner_squares : Nat := total_squares - perimeter_squares

-- The probability that a randomly chosen square does not touch the outer edge
def probability_not_on_perimeter : ℚ := inner_squares / total_squares

theorem checkerboard_probability :
  probability_not_on_perimeter = 16 / 25 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_checkerboard_probability_l1641_164161


namespace NUMINAMATH_GPT_solve_tangents_equation_l1641_164165

open Real

def is_deg (x : ℝ) : Prop := ∃ k : ℤ, x = 30 + 180 * k

theorem solve_tangents_equation (x : ℝ) (h : tan (x * π / 180) * tan (20 * π / 180) + tan (20 * π / 180) * tan (40 * π / 180) + tan (40 * π / 180) * tan (x * π / 180) = 1) :
  is_deg x :=
sorry

end NUMINAMATH_GPT_solve_tangents_equation_l1641_164165


namespace NUMINAMATH_GPT_intersect_at_single_point_l1641_164125

theorem intersect_at_single_point :
  (∃ (x y : ℝ), y = 3 * x + 5 ∧ y = -5 * x + 20 ∧ y = 4 * x + p) → p = 25 / 8 :=
by
  sorry

end NUMINAMATH_GPT_intersect_at_single_point_l1641_164125


namespace NUMINAMATH_GPT_fraction_value_l1641_164167

theorem fraction_value (a b c d : ℝ) (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  (a - d) * (b - c) / ((a - b) * (c - d)) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_fraction_value_l1641_164167


namespace NUMINAMATH_GPT_no_positive_integer_satisfies_conditions_l1641_164126

theorem no_positive_integer_satisfies_conditions : 
  ¬ ∃ (n : ℕ), (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by
  -- Proof will go here.
  sorry

end NUMINAMATH_GPT_no_positive_integer_satisfies_conditions_l1641_164126


namespace NUMINAMATH_GPT_algebraic_expression_value_zero_l1641_164104

theorem algebraic_expression_value_zero (a b : ℝ) (h : a - b = 2) : (a^3 - 2 * a^2 * b + a * b^2 - 4 * a = 0) :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_zero_l1641_164104


namespace NUMINAMATH_GPT_find_x_l1641_164102

theorem find_x (x : ℤ) (h : 2 * x = (26 - x) + 19) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1641_164102


namespace NUMINAMATH_GPT_ball_bounce_height_l1641_164166

theorem ball_bounce_height (n : ℕ) : (512 * (1/2)^n < 20) → n = 8 := 
sorry

end NUMINAMATH_GPT_ball_bounce_height_l1641_164166
