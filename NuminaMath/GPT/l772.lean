import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l772_77207

open Real

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l772_77207


namespace NUMINAMATH_GPT_unknown_number_l772_77287

theorem unknown_number (x : ℝ) (h : 7^8 - 6/x + 9^3 + 3 + 12 = 95) : x = 1 / 960908.333 :=
sorry

end NUMINAMATH_GPT_unknown_number_l772_77287


namespace NUMINAMATH_GPT_fractional_expression_evaluation_l772_77252

theorem fractional_expression_evaluation
  (m n r t : ℚ)
  (h1 : m / n = 4 / 3)
  (h2 : r / t = 9 / 14) :
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -11 / 14 := by
  sorry

end NUMINAMATH_GPT_fractional_expression_evaluation_l772_77252


namespace NUMINAMATH_GPT_intersection_value_unique_l772_77237

theorem intersection_value_unique (x : ℝ) :
  (∃ y : ℝ, y = 8 / (x^2 + 4) ∧ x + y = 2) → x = 0 :=
by
  sorry

end NUMINAMATH_GPT_intersection_value_unique_l772_77237


namespace NUMINAMATH_GPT_Xia_shared_stickers_l772_77264

def stickers_shared (initial remaining sheets_per_sheet : ℕ) : ℕ :=
  initial - (remaining * sheets_per_sheet)

theorem Xia_shared_stickers :
  stickers_shared 150 5 10 = 100 :=
by
  sorry

end NUMINAMATH_GPT_Xia_shared_stickers_l772_77264


namespace NUMINAMATH_GPT_oreos_total_l772_77232

variable (Jordan : ℕ)
variable (James : ℕ := 4 * Jordan + 7)

theorem oreos_total (h : James = 43) : 43 + Jordan = 52 :=
sorry

end NUMINAMATH_GPT_oreos_total_l772_77232


namespace NUMINAMATH_GPT_max_positive_integers_l772_77220

theorem max_positive_integers (f : Fin 2018 → ℤ) (h : ∀ i : Fin 2018, f i > f (i - 1) + f (i - 2)) : 
  ∃ n: ℕ, n = 2016 ∧ (∀ i : ℕ, i < 2018 → f i > 0) ∧ (∀ i : ℕ, i < 2 → f i < 0) := 
sorry

end NUMINAMATH_GPT_max_positive_integers_l772_77220


namespace NUMINAMATH_GPT_max_expression_value_l772_77256

theorem max_expression_value : 
  ∃ a b c d e f : ℕ, 1 ≤ a ∧ a ≤ 6 ∧
                   1 ≤ b ∧ b ≤ 6 ∧
                   1 ≤ c ∧ c ≤ 6 ∧
                   1 ≤ d ∧ d ≤ 6 ∧
                   1 ≤ e ∧ e ≤ 6 ∧
                   1 ≤ f ∧ f ≤ 6 ∧
                   a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
                   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
                   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
                   d ≠ e ∧ d ≠ f ∧
                   e ≠ f ∧
                   (f * (a * d + b * c) / (b * d * e) = 14) :=
sorry

end NUMINAMATH_GPT_max_expression_value_l772_77256


namespace NUMINAMATH_GPT_initial_amount_of_money_l772_77201

variable (X : ℕ) -- Initial amount of money Lily had in her account

-- Conditions
def spent_on_shirt : ℕ := 7
def spent_in_second_shop : ℕ := 3 * spent_on_shirt
def remaining_after_purchases : ℕ := 27

-- Proof problem: prove that the initial amount of money X is 55 given the conditions
theorem initial_amount_of_money (h : X - spent_on_shirt - spent_in_second_shop = remaining_after_purchases) : X = 55 :=
by
  -- Placeholder to indicate that steps will be worked out in Lean
  sorry

end NUMINAMATH_GPT_initial_amount_of_money_l772_77201


namespace NUMINAMATH_GPT_number_of_truthful_dwarfs_l772_77273

def total_dwarfs := 10
def hands_raised_vanilla := 10
def hands_raised_chocolate := 5
def hands_raised_fruit := 1
def total_hands_raised := hands_raised_vanilla + hands_raised_chocolate + hands_raised_fruit
def extra_hands := total_hands_raised - total_dwarfs
def liars := extra_hands
def truthful := total_dwarfs - liars

theorem number_of_truthful_dwarfs : truthful = 4 :=
by sorry

end NUMINAMATH_GPT_number_of_truthful_dwarfs_l772_77273


namespace NUMINAMATH_GPT_original_number_l772_77292

theorem original_number (x : ℝ) (h : 1.40 * x = 1680) : x = 1200 :=
by {
  sorry -- We will skip the actual proof steps here.
}

end NUMINAMATH_GPT_original_number_l772_77292


namespace NUMINAMATH_GPT_pythagorean_theorem_l772_77296

-- Definitions from the conditions
variables {a b c : ℝ}
-- Assuming a right triangle with legs a, b and hypotenuse c
def is_right_triangle (a b c : ℝ) : Prop :=
  c^2 = a^2 + b^2

-- Statement of the theorem:
theorem pythagorean_theorem (a b c : ℝ) (h : is_right_triangle a b c) : c^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_pythagorean_theorem_l772_77296


namespace NUMINAMATH_GPT_largest_unrepresentable_n_l772_77267

theorem largest_unrepresentable_n (a b : ℕ) (ha : 1 < a) (hb : 1 < b) : ∃ n, ¬ ∃ x y : ℕ, n = 7 * a + 5 * b ∧ n = 47 :=
  sorry

end NUMINAMATH_GPT_largest_unrepresentable_n_l772_77267


namespace NUMINAMATH_GPT_max_probability_sum_15_l772_77228

-- Context and Definitions based on conditions
def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The assertion to be proved:
theorem max_probability_sum_15 (n : ℕ) (h : n ∈ S) :
  n = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_probability_sum_15_l772_77228


namespace NUMINAMATH_GPT_number_of_new_bricks_l772_77251

-- Definitions from conditions
def edge_length_original_brick : ℝ := 0.3
def edge_length_new_brick : ℝ := 0.5
def number_original_bricks : ℕ := 600

-- The classroom volume is unchanged, so we set up a proportion problem
-- Assuming the classroom is fully paved
theorem number_of_new_bricks :
  let volume_original_bricks := number_original_bricks * (edge_length_original_brick ^ 2)
  let volume_new_bricks := x * (edge_length_new_brick ^ 2)
  volume_original_bricks = volume_new_bricks → x = 216 := 
by
  sorry

end NUMINAMATH_GPT_number_of_new_bricks_l772_77251


namespace NUMINAMATH_GPT_problem_statement_l772_77235

noncomputable def f (x : ℝ) : ℝ := x + 1 / x - Real.sqrt 2

theorem problem_statement (x : ℝ) (h₁ : x ∈ Set.Ioc (Real.sqrt 2 / 2) 1) :
  Real.sqrt 2 / 2 < f (f x) ∧ f (f x) < x :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l772_77235


namespace NUMINAMATH_GPT_sum_of_relatively_prime_integers_l772_77247

theorem sum_of_relatively_prime_integers (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (h3 : x * y + x + y = 154) (h4 : Nat.gcd x y = 1) (h5 : x < 30) (h6 : y < 30) : 
  x + y = 34 :=
sorry -- proof

end NUMINAMATH_GPT_sum_of_relatively_prime_integers_l772_77247


namespace NUMINAMATH_GPT_zoe_remaining_pictures_l772_77213

-- Definitions based on the conditions
def total_pictures : Nat := 88
def colored_pictures : Nat := 20

-- Proof statement
theorem zoe_remaining_pictures : total_pictures - colored_pictures = 68 := by
  sorry

end NUMINAMATH_GPT_zoe_remaining_pictures_l772_77213


namespace NUMINAMATH_GPT_arithmetic_difference_l772_77253

variables (p q r : ℝ)

theorem arithmetic_difference (h1 : (p + q) / 2 = 10) (h2 : (q + r) / 2 = 27) : r - p = 34 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_difference_l772_77253


namespace NUMINAMATH_GPT_opposite_of_six_is_neg_six_l772_77214

-- Define the condition that \( a \) is the opposite of \( 6 \)
def is_opposite_of_six (a : Int) : Prop := a = -6

-- Prove that \( a = -6 \) given that \( a \) is the opposite of \( 6 \)
theorem opposite_of_six_is_neg_six (a : Int) (h : is_opposite_of_six a) : a = -6 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_six_is_neg_six_l772_77214


namespace NUMINAMATH_GPT_total_students_in_school_l772_77219

theorem total_students_in_school : 
  ∀ (number_of_deaf_students number_of_blind_students : ℕ), 
  (number_of_deaf_students = 180) → 
  (number_of_deaf_students = 3 * number_of_blind_students) → 
  (number_of_deaf_students + number_of_blind_students = 240) :=
by 
  sorry

end NUMINAMATH_GPT_total_students_in_school_l772_77219


namespace NUMINAMATH_GPT_ratio_of_years_taught_l772_77245

-- Definitions based on given conditions
def C : ℕ := 4
def A : ℕ := 2 * C
def total_years (S : ℕ) : Prop := C + A + S = 52

-- Proof statement
theorem ratio_of_years_taught (S : ℕ) (h : total_years S) : 
  S / A = 5 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_years_taught_l772_77245


namespace NUMINAMATH_GPT_lucas_earnings_l772_77259

-- Declare constants and definitions given in the problem
def dollars_per_window : ℕ := 3
def windows_per_floor : ℕ := 5
def floors : ℕ := 4
def penalty_amount : ℕ := 2
def days_per_period : ℕ := 4
def total_days : ℕ := 12

-- Definition of the number of total windows
def total_windows : ℕ := windows_per_floor * floors

-- Initial earnings before penalties
def initial_earnings : ℕ := total_windows * dollars_per_window

-- Number of penalty periods
def penalty_periods : ℕ := total_days / days_per_period

-- Total penalty amount
def total_penalty : ℕ := penalty_periods * penalty_amount

-- Final earnings after penalties
def final_earnings : ℕ := initial_earnings - total_penalty

-- Proof problem: correct amount Lucas' father will pay
theorem lucas_earnings : final_earnings = 54 :=
by
  sorry

end NUMINAMATH_GPT_lucas_earnings_l772_77259


namespace NUMINAMATH_GPT_initial_number_is_11_l772_77255

theorem initial_number_is_11 :
  ∃ (N : ℤ), ∃ (k : ℤ), N - 11 = 17 * k ∧ N = 11 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_is_11_l772_77255


namespace NUMINAMATH_GPT_probability_exactly_one_second_class_product_l772_77229

open Nat

/-- Proof problem -/
theorem probability_exactly_one_second_class_product :
  let n := 100 -- total products
  let k := 4   -- number of selected products
  let first_class := 90 -- first-class products
  let second_class := 10 -- second-class products
  let C (n k : ℕ) := Nat.choose n k
  (C second_class 1 * C first_class 3 : ℚ) / C n k = 
  (Nat.choose second_class 1 * Nat.choose first_class 3 : ℚ) / Nat.choose n k :=
by
  -- Mathematically equivalent proof
  sorry

end NUMINAMATH_GPT_probability_exactly_one_second_class_product_l772_77229


namespace NUMINAMATH_GPT_range_of_m_l772_77277

def P (m : ℝ) : Prop := m^2 - 4 > 0
def Q (m : ℝ) : Prop := 16 * (m - 2)^2 - 16 < 0

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) ↔ (m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l772_77277


namespace NUMINAMATH_GPT_lesser_of_two_numbers_l772_77258

theorem lesser_of_two_numbers (a b : ℕ) (h₁ : a + b = 55) (h₂ : a - b = 7) (h₃ : a > b) : b = 24 :=
by
  sorry

end NUMINAMATH_GPT_lesser_of_two_numbers_l772_77258


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l772_77239

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℤ) (h₁ : a 2 = 9) (h₂ : a 5 = 33) :
  ∀ d : ℤ, (∀ n : ℕ, a n = a 1 + (n - 1) * d) → d = 8 :=
by
  -- We state the theorem and provide a "sorry" proof placeholder
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l772_77239


namespace NUMINAMATH_GPT_fraction_to_decimal_l772_77257

theorem fraction_to_decimal : (7 / 16 : ℝ) = 0.4375 := by
  sorry

end NUMINAMATH_GPT_fraction_to_decimal_l772_77257


namespace NUMINAMATH_GPT_river_flow_rate_l772_77279

variables (d w : ℝ) (V : ℝ)

theorem river_flow_rate (h₁ : d = 4) (h₂ : w = 40) (h₃ : V = 10666.666666666666) :
  ((V / 60) / (d * w) * 3.6) = 4 :=
by sorry

end NUMINAMATH_GPT_river_flow_rate_l772_77279


namespace NUMINAMATH_GPT_parabola_focus_l772_77280

theorem parabola_focus (a : ℝ) (p : ℝ) (x y : ℝ) :
  a = -3 ∧ p = 6 →
  (y^2 = -2 * p * x) → 
  (y^2 = -12 * x) := 
by sorry

end NUMINAMATH_GPT_parabola_focus_l772_77280


namespace NUMINAMATH_GPT_Susan_roses_ratio_l772_77248

theorem Susan_roses_ratio (total_roses given_roses vase_roses remaining_roses : ℕ) 
  (H1 : total_roses = 3 * 12)
  (H2 : vase_roses = total_roses - given_roses)
  (H3 : remaining_roses = vase_roses * 2 / 3)
  (H4 : remaining_roses = 12) :
  given_roses / gcd given_roses total_roses = 1 ∧ total_roses / gcd given_roses total_roses = 2 :=
by
  sorry

end NUMINAMATH_GPT_Susan_roses_ratio_l772_77248


namespace NUMINAMATH_GPT_john_apartment_number_l772_77234

variable (k d m : ℕ)

theorem john_apartment_number (h1 : k = m) (h2 : d + m = 239) (h3 : 10 * (k - 1) + 1 ≤ d) (h4 : d ≤ 10 * k) : d = 217 := 
by 
  sorry

end NUMINAMATH_GPT_john_apartment_number_l772_77234


namespace NUMINAMATH_GPT_determine_b_l772_77265

noncomputable def has_exactly_one_real_solution (b : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - b*x^3 - 3*b*x + b^2 - 2 = 0 ∧ ∀ y : ℝ, y ≠ x → y^4 - b*y^3 - 3*b*y + b^2 - 2 ≠ 0

theorem determine_b (b : ℝ) :
  has_exactly_one_real_solution b → b < 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_b_l772_77265


namespace NUMINAMATH_GPT_calc_difference_l772_77263

theorem calc_difference :
  let a := (7/12 : ℚ) * 450
  let b := (3/5 : ℚ) * 320
  let c := (5/9 : ℚ) * 540
  let d := b + c
  d - a = 229.5 := by
  -- declare the variables and provide their values
  sorry

end NUMINAMATH_GPT_calc_difference_l772_77263


namespace NUMINAMATH_GPT_sally_cut_red_orchids_l772_77261

-- Definitions and conditions
def initial_red_orchids := 9
def orchids_in_vase_after_cutting := 15

-- Problem statement
theorem sally_cut_red_orchids : (orchids_in_vase_after_cutting - initial_red_orchids) = 6 := by
  sorry

end NUMINAMATH_GPT_sally_cut_red_orchids_l772_77261


namespace NUMINAMATH_GPT_percentage_sold_is_80_l772_77276

-- Definitions corresponding to conditions
def first_day_houses : Nat := 20
def items_per_house : Nat := 2
def total_items_sold : Nat := 104

-- Calculate the houses visited on the second day
def second_day_houses : Nat := 2 * first_day_houses

-- Calculate items sold on the first day
def items_sold_first_day : Nat := first_day_houses * items_per_house

-- Calculate items sold on the second day
def items_sold_second_day : Nat := total_items_sold - items_sold_first_day

-- Calculate houses sold to on the second day
def houses_sold_to_second_day : Nat := items_sold_second_day / items_per_house

-- Percentage calculation
def percentage_sold_second_day : Nat := (houses_sold_to_second_day * 100) / second_day_houses

-- Theorem proving that James sold to 80% of the houses on the second day
theorem percentage_sold_is_80 : percentage_sold_second_day = 80 := by
  sorry

end NUMINAMATH_GPT_percentage_sold_is_80_l772_77276


namespace NUMINAMATH_GPT_gcd_of_given_lengths_l772_77233

def gcd_of_lengths_is_eight : Prop :=
  let lengths := [48, 64, 80, 120]
  ∃ d, d = 8 ∧ (∀ n ∈ lengths, d ∣ n)

theorem gcd_of_given_lengths : gcd_of_lengths_is_eight := 
  sorry

end NUMINAMATH_GPT_gcd_of_given_lengths_l772_77233


namespace NUMINAMATH_GPT_jill_braids_dancers_l772_77275

def dancers_on_team (braids_per_dancer : ℕ) (seconds_per_braid : ℕ) (total_time_seconds : ℕ) : ℕ :=
  total_time_seconds / seconds_per_braid / braids_per_dancer

theorem jill_braids_dancers (h1 : braids_per_dancer = 5) (h2 : seconds_per_braid = 30)
                             (h3 : total_time_seconds = 20 * 60) : 
  dancers_on_team braids_per_dancer seconds_per_braid total_time_seconds = 8 :=
by
  sorry

end NUMINAMATH_GPT_jill_braids_dancers_l772_77275


namespace NUMINAMATH_GPT_empty_boxes_count_l772_77268

-- Definitions based on conditions:
def large_box_contains (B : Type) : ℕ := 1
def initial_small_boxes (B : Type) : ℕ := 10
def non_empty_boxes (B : Type) : ℕ := 6
def additional_smaller_boxes_in_non_empty (B : Type) (b : B) : ℕ := 10
def non_empty_small_boxes := 5

-- Proving that the number of empty boxes is 55 given the conditions:
theorem empty_boxes_count (B : Type) : 
  large_box_contains B = 1 ∧
  initial_small_boxes B = 10 ∧
  non_empty_boxes B = 6 ∧
  (∃ b : B, additional_smaller_boxes_in_non_empty B b = 10) →
  (initial_small_boxes B - non_empty_small_boxes + non_empty_small_boxes * additional_smaller_boxes_in_non_empty B) = 55 :=
by 
  sorry

end NUMINAMATH_GPT_empty_boxes_count_l772_77268


namespace NUMINAMATH_GPT_total_juice_sold_3_days_l772_77297

def juice_sales_problem (V_L V_M V_S : ℕ) (d1 d2 d3 : ℕ) :=
  (d1 = V_L + 4 * V_M) ∧ 
  (d2 = 2 * V_L + 6 * V_S) ∧ 
  (d3 = V_L + 3 * V_M + 3 * V_S) ∧
  (d1 = d2) ∧
  (d2 = d3)

theorem total_juice_sold_3_days (V_L V_M V_S d1 d2 d3 : ℕ) 
  (h : juice_sales_problem V_L V_M V_S d1 d2 d3) 
  (h_VM : V_M = 3) 
  (h_VL : V_L = 6) : 
  3 * d1 = 54 := 
by 
  -- Proof will be filled in
  sorry

end NUMINAMATH_GPT_total_juice_sold_3_days_l772_77297


namespace NUMINAMATH_GPT_sequence_general_term_l772_77238

theorem sequence_general_term 
  (x : ℕ → ℝ)
  (h1 : x 1 = 2)
  (h2 : x 2 = 3)
  (h3 : ∀ m ≥ 1, x (2*m+1) = x (2*m) + x (2*m-1))
  (h4 : ∀ m ≥ 2, x (2*m) = x (2*m-1) + 2*x (2*m-2)) :
  ∀ m, (x (2*m-1) = ((3 - Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((3 + Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m ∧ 
          x (2*m) = ((1 + 2 * Real.sqrt 2) / 4) * (2 + Real.sqrt 2) ^ m + ((1 - 2 * Real.sqrt 2) / 4) * (2 - Real.sqrt 2) ^ m) :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l772_77238


namespace NUMINAMATH_GPT_problem_statement_l772_77205

open Set

-- Definitions based on the problem's conditions
def U : Set ℕ := { x | 0 < x ∧ x ≤ 8 }
def S : Set ℕ := {1, 2, 4, 5}
def T : Set ℕ := {3, 5, 7}
def complement_U_T : Set ℕ := U \ T

-- The Lean 4 statement to prove
theorem problem_statement : S ∩ complement_U_T = {1, 2, 4} :=
by sorry

end NUMINAMATH_GPT_problem_statement_l772_77205


namespace NUMINAMATH_GPT_better_fit_model_l772_77240

-- Define the residual sums of squares
def RSS_1 : ℝ := 152.6
def RSS_2 : ℝ := 159.8

-- Define the statement that the model with RSS_1 is the better fit
theorem better_fit_model : RSS_1 < RSS_2 → RSS_1 = 152.6 :=
by
  sorry

end NUMINAMATH_GPT_better_fit_model_l772_77240


namespace NUMINAMATH_GPT_tetrahedron_labeling_impossible_l772_77216

/-- Suppose each vertex of a tetrahedron needs to be labeled with an integer from 1 to 4, each integer being used exactly once.
We need to prove that there are no such arrangements in which the sum of the numbers on the vertices of each face is the same for all four faces.
Arrangements that can be rotated into each other are considered identical. -/
theorem tetrahedron_labeling_impossible :
  ∀ (label : Fin 4 → Fin 5) (h_unique : ∀ v1 v2 : Fin 4, v1 ≠ v2 → label v1 ≠ label v2),
  ∃ (sum_faces : ℕ), sum_faces = 7 ∧ sum_faces % 3 = 1 → False :=
by
  sorry

end NUMINAMATH_GPT_tetrahedron_labeling_impossible_l772_77216


namespace NUMINAMATH_GPT_geometric_series_sum_y_equals_nine_l772_77243

theorem geometric_series_sum_y_equals_nine : 
  (∑' n : ℕ, (1 / 3) ^ n) * (∑' n : ℕ, (-1 / 3) ^ n) = ∑' n : ℕ, (1 / (9 ^ n)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_y_equals_nine_l772_77243


namespace NUMINAMATH_GPT_acute_triangle_l772_77203

theorem acute_triangle (a b c : ℝ) (h : a^π + b^π = c^π) : a^2 + b^2 > c^2 := sorry

end NUMINAMATH_GPT_acute_triangle_l772_77203


namespace NUMINAMATH_GPT_park_trees_after_planting_l772_77244

theorem park_trees_after_planting (current_trees trees_today trees_tomorrow : ℕ)
  (h1 : current_trees = 7)
  (h2 : trees_today = 5)
  (h3 : trees_tomorrow = 4) :
  current_trees + trees_today + trees_tomorrow = 16 :=
by
  sorry

end NUMINAMATH_GPT_park_trees_after_planting_l772_77244


namespace NUMINAMATH_GPT_find_angle_CDB_l772_77286

variables (A B C D E : Type)
variables [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C] [LinearOrderedField D] [LinearOrderedField E]

noncomputable def angle := ℝ -- Define type for angles

variables (AB AD AC ACB ACD : angle)
variables (BAD BEA CDB : ℝ)

-- Define the given angles and conditions in Lean
axiom AB_eq_AD : AB = AD
axiom angle_ACD_eq_angle_ACB : AC = ACD
axiom angle_BAD_eq_140 : BAD = 140
axiom angle_BEA_eq_110 : BEA = 110

theorem find_angle_CDB (AB_eq_AD : AB = AD)
                       (angle_ACD_eq_angle_ACB : AC = ACD)
                       (angle_BAD_eq_140 : BAD = 140)
                       (angle_BEA_eq_110 : BEA = 110) :
                       CDB = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_CDB_l772_77286


namespace NUMINAMATH_GPT_tall_mirror_passes_l772_77225

theorem tall_mirror_passes (T : ℕ)
    (s_tall_ref : ℕ)
    (s_wide_ref : ℕ)
    (e_tall_ref : ℕ)
    (e_wide_ref : ℕ)
    (wide_passes : ℕ)
    (total_reflections : ℕ)
    (H1 : s_tall_ref = 10)
    (H2 : s_wide_ref = 5)
    (H3 : e_tall_ref = 6)
    (H4 : e_wide_ref = 3)
    (H5 : wide_passes = 5)
    (H6 : s_tall_ref * T + s_wide_ref * wide_passes + e_tall_ref * T + e_wide_ref * wide_passes = 88) : 
    T = 3 := 
by sorry

end NUMINAMATH_GPT_tall_mirror_passes_l772_77225


namespace NUMINAMATH_GPT_main_inequality_l772_77211

theorem main_inequality (m : ℝ) : (∀ x : ℝ, |2 * x - m| ≤ |3 * x + 6|) ↔ m = -4 := by
  sorry

end NUMINAMATH_GPT_main_inequality_l772_77211


namespace NUMINAMATH_GPT_triangle_inequality_inequality_l772_77250

theorem triangle_inequality_inequality {a b c : ℝ}
  (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  3 * (b + c - a) * (c + a - b) * (a + b - c) ≤ a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_inequality_l772_77250


namespace NUMINAMATH_GPT_visible_steps_on_escalator_l772_77210

variable (steps_visible : ℕ) -- The number of steps visible on the escalator
variable (al_steps : ℕ := 150) -- Al walks down 150 steps
variable (bob_steps : ℕ := 75) -- Bob walks up 75 steps
variable (al_speed : ℕ := 3) -- Al's walking speed
variable (bob_speed : ℕ := 1) -- Bob's walking speed
variable (escalator_speed : ℚ) -- The speed of the escalator

theorem visible_steps_on_escalator : steps_visible = 120 :=
by
  -- Define times taken by Al and Bob
  let al_time := al_steps / al_speed
  let bob_time := bob_steps / bob_speed

  -- Define effective speeds considering escalator speed 'escalator_speed'
  let al_effective_speed := al_speed - escalator_speed
  let bob_effective_speed := bob_speed + escalator_speed

  -- Calculate the total steps walked if the escalator was stopped (same total steps)
  have al_total_steps := al_effective_speed * al_time
  have bob_total_steps := bob_effective_speed * bob_time

  -- Set up the equation
  have eq := al_total_steps = bob_total_steps

  -- Substitute and solve for escalator_speed
  sorry

end NUMINAMATH_GPT_visible_steps_on_escalator_l772_77210


namespace NUMINAMATH_GPT_certain_event_birthday_example_l772_77288
-- Import the necessary library

-- Define the problem with conditions
def certain_event_people_share_birthday (num_days : ℕ) (num_people : ℕ) : Prop :=
  num_people > num_days

-- Define a specific instance based on the given problem
theorem certain_event_birthday_example : certain_event_people_share_birthday 365 400 :=
by
  sorry

end NUMINAMATH_GPT_certain_event_birthday_example_l772_77288


namespace NUMINAMATH_GPT_original_average_is_24_l772_77204

theorem original_average_is_24
  (A : ℝ)
  (h1 : ∀ n : ℕ, n = 7 → 35 * A = 7 * 120) :
  A = 24 :=
by
  sorry

end NUMINAMATH_GPT_original_average_is_24_l772_77204


namespace NUMINAMATH_GPT_phillip_remaining_money_l772_77231

def initial_money : ℝ := 95
def cost_oranges : ℝ := 14
def cost_apples : ℝ := 25
def cost_candy : ℝ := 6
def cost_eggs : ℝ := 12
def cost_milk : ℝ := 8
def discount_apples_rate : ℝ := 0.15
def discount_milk_rate : ℝ := 0.10

def discounted_cost_apples : ℝ := cost_apples * (1 - discount_apples_rate)
def discounted_cost_milk : ℝ := cost_milk * (1 - discount_milk_rate)

def total_spent : ℝ := cost_oranges + discounted_cost_apples + cost_candy + cost_eggs + discounted_cost_milk

def remaining_money : ℝ := initial_money - total_spent

theorem phillip_remaining_money : remaining_money = 34.55 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_phillip_remaining_money_l772_77231


namespace NUMINAMATH_GPT_cyclist_average_rate_l772_77281

noncomputable def average_rate_round_trip (D : ℝ) : ℝ :=
  let time_to_travel := D / 10
  let time_to_return := D / 9
  let total_distance := 2 * D
  let total_time := time_to_travel + time_to_return
  (total_distance / total_time)

theorem cyclist_average_rate (D : ℝ) (hD : D > 0) :
  average_rate_round_trip D = 180 / 19 :=
by
  sorry

end NUMINAMATH_GPT_cyclist_average_rate_l772_77281


namespace NUMINAMATH_GPT_trigonometric_identity_l772_77215

theorem trigonometric_identity
  (α : Real)
  (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + 2 * Real.sin α * Real.cos α) = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l772_77215


namespace NUMINAMATH_GPT_paper_pieces_l772_77271

theorem paper_pieces (n : ℕ) (h1 : 20 = 2 * n - 8) : n^2 + 20 = 216 := 
by
  sorry

end NUMINAMATH_GPT_paper_pieces_l772_77271


namespace NUMINAMATH_GPT_ratio_fraction_l772_77254

theorem ratio_fraction (A B C : ℕ) (h1 : 7 * B = 3 * A) (h2 : 6 * C = 5 * B) :
  (C : ℚ) / (A : ℚ) = 5 / 14 ∧ (A : ℚ) / (C : ℚ) = 14 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_fraction_l772_77254


namespace NUMINAMATH_GPT_sqrt_product_simplification_l772_77218

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_product_simplification_l772_77218


namespace NUMINAMATH_GPT_simplify_expr_l772_77206

variable {x y : ℝ}

theorem simplify_expr (hx : x ≠ 0) (hy : y ≠ 0) :
  ((x^3 + 1) / x) * ((y^3 + 1) / y) - ((x^3 - 1) / y) * ((y^3 - 1) / x) = 2 * x^2 + 2 * y^2 :=
by sorry

end NUMINAMATH_GPT_simplify_expr_l772_77206


namespace NUMINAMATH_GPT_buildings_collapsed_l772_77212

theorem buildings_collapsed (B : ℕ) (h₁ : 2 * B = X) (h₂ : 4 * B = Y) (h₃ : 8 * B = Z) (h₄ : B + 2 * B + 4 * B + 8 * B = 60) : B = 4 :=
by
  sorry

end NUMINAMATH_GPT_buildings_collapsed_l772_77212


namespace NUMINAMATH_GPT_student_average_marks_l772_77241

theorem student_average_marks 
(P C M : ℕ) 
(h1 : (P + M) / 2 = 90) 
(h2 : (P + C) / 2 = 70) 
(h3 : P = 65) : 
  (P + C + M) / 3 = 85 :=
  sorry

end NUMINAMATH_GPT_student_average_marks_l772_77241


namespace NUMINAMATH_GPT_sally_sours_total_l772_77272

theorem sally_sours_total (cherry_sours lemon_sours orange_sours total_sours : ℕ) 
    (h1 : cherry_sours = 32)
    (h2 : 5 * cherry_sours = 4 * lemon_sours)
    (h3 : orange_sours = total_sours / 4)
    (h4 : cherry_sours + lemon_sours + orange_sours = total_sours) : 
    total_sours = 96 :=
by
  rw [h1] at h2
  have h5 : lemon_sours = 40 := by linarith
  rw [h1, h5] at h4
  have h6 : orange_sours = total_sours / 4 := by assumption
  rw [h6] at h4
  have h7 : 72 + total_sours / 4 = total_sours := by linarith
  sorry

end NUMINAMATH_GPT_sally_sours_total_l772_77272


namespace NUMINAMATH_GPT_ratio_sum_odd_even_divisors_l772_77249

def M : ℕ := 33 * 38 * 58 * 462

theorem ratio_sum_odd_even_divisors : 
  let sum_odd_divisors := 
    (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_all_divisors := 
    (1 + 2 + 4 + 8) * (1 + 3 + 3^2) * (1 + 7) * (1 + 11 + 11^2) * (1 + 19) * (1 + 29)
  let sum_even_divisors := sum_all_divisors - sum_odd_divisors
  (sum_odd_divisors : ℚ) / sum_even_divisors = 1 / 14 :=
by sorry

end NUMINAMATH_GPT_ratio_sum_odd_even_divisors_l772_77249


namespace NUMINAMATH_GPT_slope_intercept_parallel_line_l772_77269

def is_parallel (m1 m2 : ℝ) : Prop :=
  m1 = m2

theorem slope_intercept_parallel_line (A : ℝ × ℝ) (hA₁ : A.1 = 3) (hA₂ : A.2 = 2) 
  (m : ℝ) (h_parallel : is_parallel m (-4)) : ∃ b : ℝ, ∀ x y : ℝ, y = -4 * x + b :=
by
  use 14
  intro x y
  sorry

end NUMINAMATH_GPT_slope_intercept_parallel_line_l772_77269


namespace NUMINAMATH_GPT_solve_inequality_system_l772_77284

theorem solve_inequality_system (x : ℝ) :
  (8 * x - 3 ≤ 13) ∧ ((x - 1) / 3 - 2 < x - 1) → -2 < x ∧ x ≤ 2 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l772_77284


namespace NUMINAMATH_GPT_find_discounts_l772_77293

variables (a b c : ℝ)
variables (x y z : ℝ)

theorem find_discounts (h1 : 1.1 * a - x * a = 0.99 * a)
                       (h2 : 1.12 * b - y * b = 0.99 * b)
                       (h3 : 1.15 * c - z * c = 0.99 * c) : 
x = 0.11 ∧ y = 0.13 ∧ z = 0.16 := 
sorry

end NUMINAMATH_GPT_find_discounts_l772_77293


namespace NUMINAMATH_GPT_penelope_mandm_candies_l772_77208

theorem penelope_mandm_candies (m n : ℕ) (r : ℝ) :
  (m / n = 5 / 3) → (n = 15) → (m = 25) :=
by
  sorry

end NUMINAMATH_GPT_penelope_mandm_candies_l772_77208


namespace NUMINAMATH_GPT_smallest_x_absolute_value_l772_77246

theorem smallest_x_absolute_value : ∃ x : ℤ, |x + 3| = 15 ∧ ∀ y : ℤ, |y + 3| = 15 → x ≤ y :=
sorry

end NUMINAMATH_GPT_smallest_x_absolute_value_l772_77246


namespace NUMINAMATH_GPT_total_number_of_soccer_games_l772_77294

theorem total_number_of_soccer_games (teams : ℕ)
  (regular_games_per_team : ℕ)
  (promotional_games_per_team : ℕ)
  (h1 : teams = 15)
  (h2 : regular_games_per_team = 14)
  (h3 : promotional_games_per_team = 2) :
  ((teams * regular_games_per_team) / 2 + (teams * promotional_games_per_team) / 2) = 120 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_soccer_games_l772_77294


namespace NUMINAMATH_GPT_tangent_line_inv_g_at_0_l772_77260

noncomputable def g (x : ℝ) := Real.log x

theorem tangent_line_inv_g_at_0 
  (h₁ : ∀ x, g x = Real.log x) 
  (h₂ : ∀ x, x > 0): 
  ∃ m b, (∀ x y, y = g⁻¹ x → y - m * x = b) ∧ 
         (m = 1) ∧ 
         (b = 1) ∧ 
         (∀ x y, x - y + 1 = 0) := 
by
  sorry

end NUMINAMATH_GPT_tangent_line_inv_g_at_0_l772_77260


namespace NUMINAMATH_GPT_number_of_unlocked_cells_l772_77209

-- Establish the conditions from the problem description.
def total_cells : ℕ := 2004

-- Helper function to determine if a number is a perfect square.
def is_perfect_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = n

-- Counting the number of perfect squares in the range from 1 to total_cells.
def perfect_squares_up_to (n : ℕ) : ℕ :=
  (Nat.sqrt n)

-- The theorem that needs to be proved.
theorem number_of_unlocked_cells : perfect_squares_up_to total_cells = 44 :=
by
  sorry

end NUMINAMATH_GPT_number_of_unlocked_cells_l772_77209


namespace NUMINAMATH_GPT_negation_of_forall_geq_l772_77224

theorem negation_of_forall_geq {x : ℝ} : ¬ (∀ x : ℝ, x^2 - x ≥ 0) ↔ ∃ x : ℝ, x^2 - x < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_forall_geq_l772_77224


namespace NUMINAMATH_GPT_average_wage_correct_l772_77295

def male_workers : ℕ := 20
def female_workers : ℕ := 15
def child_workers : ℕ := 5

def male_wage : ℕ := 25
def female_wage : ℕ := 20
def child_wage : ℕ := 8

def total_amount_paid_per_day : ℕ := 
  (male_workers * male_wage) + (female_workers * female_wage) + (child_workers * child_wage)

def total_number_of_workers : ℕ := 
  male_workers + female_workers + child_workers

def average_wage_per_day : ℕ := 
  total_amount_paid_per_day / total_number_of_workers

theorem average_wage_correct : 
  average_wage_per_day = 21 := by 
  sorry

end NUMINAMATH_GPT_average_wage_correct_l772_77295


namespace NUMINAMATH_GPT_find_a_values_l772_77278

theorem find_a_values (a t t₁ t₂ : ℝ) :
  (t^2 + (a - 6) * t + (9 - 3 * a) = 0) ∧
  (t₁ = 4 * t₂) ∧
  (t₁ + t₂ = 6 - a) ∧
  (t₁ * t₂ = 9 - 3 * a)
  ↔ (a = -2 ∨ a = 2) := sorry

end NUMINAMATH_GPT_find_a_values_l772_77278


namespace NUMINAMATH_GPT_frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l772_77230

-- Definitions of conditions
def grasshopper_jump : ℕ := 19
def mouse_jump_frog (frog_jump : ℕ) : ℕ := frog_jump + 20
def mouse_jump_grasshopper : ℕ := grasshopper_jump + 30

-- The proof problem statement
theorem frog_jumps_10_inches_more_than_grasshopper (frog_jump : ℕ) :
  mouse_jump_frog frog_jump = mouse_jump_grasshopper → frog_jump = 29 :=
by
  sorry

-- The ultimate question in the problem
theorem frog_jumps_10_inches_farther_than_grasshopper : 
  (∃ (frog_jump : ℕ), frog_jump = 29) → (frog_jump - grasshopper_jump = 10) :=
by
  sorry

end NUMINAMATH_GPT_frog_jumps_10_inches_more_than_grasshopper_frog_jumps_10_inches_farther_than_grasshopper_l772_77230


namespace NUMINAMATH_GPT_greatest_value_of_b_l772_77298

noncomputable def solution : ℝ :=
  (3 + Real.sqrt 21) / 2

theorem greatest_value_of_b :
  ∀ b : ℝ, b^2 - 4 * b + 3 < -b + 6 → b ≤ solution :=
by
  intro b
  intro h
  sorry

end NUMINAMATH_GPT_greatest_value_of_b_l772_77298


namespace NUMINAMATH_GPT_triangle_angle_ratio_l772_77290

theorem triangle_angle_ratio (a b c : ℝ) (h₁ : a + b + c = 180)
  (h₂ : b = 2 * a) (h₃ : c = 3 * a) : a = 30 ∧ b = 60 ∧ c = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_ratio_l772_77290


namespace NUMINAMATH_GPT_trigonometric_value_existence_l772_77283

noncomputable def can_be_value_of_tan (n : ℝ) : Prop :=
∃ θ : ℝ, Real.tan θ = n

noncomputable def can_be_value_of_cot (n : ℝ) : Prop :=
∃ θ : ℝ, 1 / Real.tan θ = n

def can_be_value_of_sin (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.sin θ = n

def can_be_value_of_cos (n : ℝ) : Prop :=
|n| ≤ 1 ∧ ∃ θ : ℝ, Real.cos θ = n

def can_be_value_of_sec (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.cos θ = n

def can_be_value_of_csc (n : ℝ) : Prop :=
|n| ≥ 1 ∧ ∃ θ : ℝ, 1 / Real.sin θ = n

theorem trigonometric_value_existence (n : ℝ) : 
  can_be_value_of_tan n ∧ 
  can_be_value_of_cot n ∧ 
  can_be_value_of_sin n ∧ 
  can_be_value_of_cos n ∧ 
  can_be_value_of_sec n ∧ 
  can_be_value_of_csc n := 
sorry

end NUMINAMATH_GPT_trigonometric_value_existence_l772_77283


namespace NUMINAMATH_GPT_inequality_positives_l772_77202

theorem inequality_positives (x1 x2 x3 x4 x5 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) (hx4 : 0 < x4) (hx5 : 0 < x5) : 
  (x1 + x2 + x3 + x4 + x5)^2 ≥ 4 * (x1 * x2 + x3 * x4 + x5 * x1 + x2 * x3 + x4 * x5) :=
sorry

end NUMINAMATH_GPT_inequality_positives_l772_77202


namespace NUMINAMATH_GPT_smallest_consecutive_natural_number_sum_l772_77266

theorem smallest_consecutive_natural_number_sum (a n : ℕ) (hn : n > 1) (h : n * a + (n * (n - 1)) / 2 = 2016) :
  ∃ a, a = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_consecutive_natural_number_sum_l772_77266


namespace NUMINAMATH_GPT_circle_radius_l772_77274

theorem circle_radius (a c r : ℝ) (h₁ : a = π * r^2) (h₂ : c = 2 * π * r) (h₃ : a + c = 100 * π) : 
  r = 9.05 := 
sorry

end NUMINAMATH_GPT_circle_radius_l772_77274


namespace NUMINAMATH_GPT_exists_equal_sum_pairs_l772_77262

theorem exists_equal_sum_pairs (n : ℕ) (hn : n > 2009) :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a ≤ 2009 ∧ b ≤ 2009 ∧ c ≤ 2009 ∧ d ≤ 2009 ∧
  (1 / a + 1 / b : ℝ) = 1 / c + 1 / d :=
sorry

end NUMINAMATH_GPT_exists_equal_sum_pairs_l772_77262


namespace NUMINAMATH_GPT_find_larger_number_l772_77221

theorem find_larger_number (hc_f : ℕ) (factor1 factor2 : ℕ)
(h_hcf : hc_f = 63)
(h_factor1 : factor1 = 11)
(h_factor2 : factor2 = 17)
(lcm := hc_f * factor1 * factor2)
(A := hc_f * factor1)
(B := hc_f * factor2) :
max A B = 1071 := by
  sorry

end NUMINAMATH_GPT_find_larger_number_l772_77221


namespace NUMINAMATH_GPT_percent_increase_perimeter_third_triangle_l772_77270

noncomputable def side_length_first : ℝ := 4
noncomputable def side_length_second : ℝ := 2 * side_length_first
noncomputable def side_length_third : ℝ := 2 * side_length_second

noncomputable def perimeter (s : ℝ) : ℝ := 3 * s

noncomputable def percent_increase (initial_perimeter final_perimeter : ℝ) : ℝ := 
  ((final_perimeter - initial_perimeter) / initial_perimeter) * 100

theorem percent_increase_perimeter_third_triangle :
  percent_increase (perimeter side_length_first) (perimeter side_length_third) = 300 := 
sorry

end NUMINAMATH_GPT_percent_increase_perimeter_third_triangle_l772_77270


namespace NUMINAMATH_GPT_simplify_fraction_l772_77291

open Complex

theorem simplify_fraction :
  (7 + 9 * I) / (3 - 4 * I) = 2.28 + 2.2 * I := 
by {
    -- We know that this should be true based on the provided solution,
    -- but we will place a placeholder here for the actual proof.
    sorry
}

end NUMINAMATH_GPT_simplify_fraction_l772_77291


namespace NUMINAMATH_GPT_max_varphi_l772_77285

noncomputable def f (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ)
noncomputable def g (x φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ + (2 * Real.pi / 3))

theorem max_varphi (φ : ℝ) (h : φ < 0) (hE : ∀ x, g x φ = g (-x) φ) : φ = -Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_max_varphi_l772_77285


namespace NUMINAMATH_GPT_distance_covered_l772_77299

-- Definitions
def speed : ℕ := 150  -- Speed in km/h
def time : ℕ := 8  -- Time in hours

-- Proof statement
theorem distance_covered : speed * time = 1200 := 
by
  sorry

end NUMINAMATH_GPT_distance_covered_l772_77299


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l772_77227

theorem average_gas_mileage_round_trip :
  let distance_to_city := 150
  let mpg_sedan := 25
  let mpg_rental := 15
  let total_distance := 2 * distance_to_city
  let gas_used_outbound := distance_to_city / mpg_sedan
  let gas_used_return := distance_to_city / mpg_rental
  let total_gas_used := gas_used_outbound + gas_used_return
  let avg_gas_mileage := total_distance / total_gas_used
  avg_gas_mileage = 18.75 := by
{
  sorry
}

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l772_77227


namespace NUMINAMATH_GPT_grandma_can_give_cherry_exists_better_grand_strategy_l772_77236

variable (Packet1 : Finset String) (Packet2 : Finset String) (Packet3 : Finset String)
variable (isCabbage : String → Prop) (isCherry : String → Prop)
variable (wholePie : String → Prop)

-- Conditions
axiom Packet1_cond : ∀ p ∈ Packet1, isCabbage p
axiom Packet2_cond : ∀ p ∈ Packet2, isCherry p
axiom Packet3_cond_cabbage : ∃ p ∈ Packet3, isCabbage p
axiom Packet3_cond_cherry : ∃ p ∈ Packet3, isCherry p

-- Question (a)
theorem grandma_can_give_cherry (h1 : ∃ p1 ∈ Packet3, wholePie p1 ∧ isCherry p1 ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy = (2 / 3) * (1 : ℝ) :=
by
  sorry

-- Question (b)
theorem exists_better_grand_strategy (h2 : ∃ p ∈ Packet3, wholePie p ∧ isCherry p ∨
    ∃ p2 ∈ Packet1, wholePie p2 ∧ (∃ q ∈ Packet2 ∪ Packet3, isCherry q ∧ wholePie q) ∨
    ∃ p3 ∈ Packet2, wholePie p3 ∧ isCherry p3) :
  ∃ grand_strategy, grand_strategy > (2 / 3) * (1 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_grandma_can_give_cherry_exists_better_grand_strategy_l772_77236


namespace NUMINAMATH_GPT_sum_when_max_power_less_500_l772_77222

theorem sum_when_max_power_less_500 :
  ∃ (a b : ℕ), a > 0 ∧ b > 1 ∧ a^b < 500 ∧
  (∀ (a' b' : ℕ), a' > 0 → b' > 1 → a'^b' < 500 → a^b ≥ a'^b') ∧ (a + b = 24) :=
  sorry

end NUMINAMATH_GPT_sum_when_max_power_less_500_l772_77222


namespace NUMINAMATH_GPT_annual_parking_savings_l772_77226

theorem annual_parking_savings :
  let weekly_rate := 10
  let monthly_rate := 40
  let weeks_in_year := 52
  let months_in_year := 12
  let annual_weekly_cost := weekly_rate * weeks_in_year
  let annual_monthly_cost := monthly_rate * months_in_year
  let savings := annual_weekly_cost - annual_monthly_cost
  savings = 40 := by
{
  sorry
}

end NUMINAMATH_GPT_annual_parking_savings_l772_77226


namespace NUMINAMATH_GPT_universal_negation_example_l772_77242

theorem universal_negation_example :
  (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) →
  (¬ (∀ x : ℝ, x^2 - 3 * x + 1 ≤ 0) = (∃ x : ℝ, x^2 - 3 * x + 1 > 0)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_universal_negation_example_l772_77242


namespace NUMINAMATH_GPT_evaluate_expression_l772_77217

theorem evaluate_expression : -30 + 5 * (9 / (3 + 3)) = -22.5 := sorry

end NUMINAMATH_GPT_evaluate_expression_l772_77217


namespace NUMINAMATH_GPT_geometric_sequence_characterization_l772_77289

theorem geometric_sequence_characterization (a : ℕ → ℝ) :
  (∀ n, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_characterization_l772_77289


namespace NUMINAMATH_GPT_min_value_of_f_l772_77200

-- Define the function f
def f (a b c x y z : ℤ) : ℤ := a * x + b * y + c * z

-- Define the gcd function for three integers
def gcd3 (a b c : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) c

-- Define the main theorem to prove
theorem min_value_of_f (a b c : ℕ) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) :
  ∃ (x y z : ℤ), f a b c x y z = gcd3 a b c := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l772_77200


namespace NUMINAMATH_GPT_smallest_gcd_value_l772_77282

theorem smallest_gcd_value (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : Nat.gcd m n = 8) : Nat.gcd (8 * m) (12 * n) = 32 :=
by
  sorry

end NUMINAMATH_GPT_smallest_gcd_value_l772_77282


namespace NUMINAMATH_GPT_cylinder_volume_l772_77223

theorem cylinder_volume (r h : ℝ) (hrh : 2 * Real.pi * r * h = 100 * Real.pi) (h_diag : 4 * r^2 + h^2 = 200) :
  Real.pi * r^2 * h = 250 * Real.pi :=
sorry

end NUMINAMATH_GPT_cylinder_volume_l772_77223
