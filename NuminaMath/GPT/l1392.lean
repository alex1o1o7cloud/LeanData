import Mathlib

namespace NUMINAMATH_GPT_like_terms_eq_l1392_139255

theorem like_terms_eq : 
  ∀ (x y : ℕ), 
  (x + 2 * y = 3) → 
  (2 * x + y = 9) → 
  (x + y = 4) :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_like_terms_eq_l1392_139255


namespace NUMINAMATH_GPT_bathing_suits_total_l1392_139268

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969
def total_bathing_suits : ℕ := 19766

theorem bathing_suits_total :
  men_bathing_suits + women_bathing_suits = total_bathing_suits := by
  sorry

end NUMINAMATH_GPT_bathing_suits_total_l1392_139268


namespace NUMINAMATH_GPT_union_complement_eq_l1392_139233

def U : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 1, 2}
def N : Set ℕ := {0, 2, 3}

theorem union_complement_eq : M ∪ (U \ N) = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_union_complement_eq_l1392_139233


namespace NUMINAMATH_GPT_weight_of_replaced_person_l1392_139292

/-- The weight of the person who was replaced is calculated given the average weight increase for 8 persons and the weight of the new person. --/
theorem weight_of_replaced_person
  (avg_weight_increase : ℝ)
  (num_persons : ℕ)
  (weight_new_person : ℝ) :
  avg_weight_increase = 3 → 
  num_persons = 8 →
  weight_new_person = 89 →
  weight_new_person - avg_weight_increase * num_persons = 65 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_weight_of_replaced_person_l1392_139292


namespace NUMINAMATH_GPT_light_flash_fraction_l1392_139279

theorem light_flash_fraction (flash_interval : ℕ) (total_flashes : ℕ) (seconds_in_hour : ℕ) (fraction_of_hour : ℚ) :
  flash_interval = 6 →
  total_flashes = 600 →
  seconds_in_hour = 3600 →
  fraction_of_hour = 1 →
  (total_flashes * flash_interval) / seconds_in_hour = fraction_of_hour := by
  sorry

end NUMINAMATH_GPT_light_flash_fraction_l1392_139279


namespace NUMINAMATH_GPT_find_k_l1392_139261

theorem find_k (x y k : ℝ) (h₁ : x = 2) (h₂ : y = -1) (h₃ : y - k * x = 7) : k = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1392_139261


namespace NUMINAMATH_GPT_total_time_marco_6_laps_total_time_in_minutes_and_seconds_l1392_139250

noncomputable def marco_running_time : ℕ :=
  let distance_1 := 150
  let speed_1 := 5
  let time_1 := distance_1 / speed_1

  let distance_2 := 300
  let speed_2 := 4
  let time_2 := distance_2 / speed_2

  let time_per_lap := time_1 + time_2
  let total_laps := 6
  let total_time_seconds := time_per_lap * total_laps

  total_time_seconds

theorem total_time_marco_6_laps : marco_running_time = 630 := sorry

theorem total_time_in_minutes_and_seconds : 10 * 60 + 30 = 630 := sorry

end NUMINAMATH_GPT_total_time_marco_6_laps_total_time_in_minutes_and_seconds_l1392_139250


namespace NUMINAMATH_GPT_quarterly_production_growth_l1392_139263

theorem quarterly_production_growth (P_A P_Q2 : ℕ) (x : ℝ)
  (hA : P_A = 500000)
  (hQ2 : P_Q2 = 1820000) :
  50 + 50 * (1 + x) + 50 * (1 + x)^2 = 182 :=
by 
  sorry

end NUMINAMATH_GPT_quarterly_production_growth_l1392_139263


namespace NUMINAMATH_GPT_heptagon_divisibility_impossible_l1392_139226

theorem heptagon_divisibility_impossible (a b c d e f g : ℕ) :
  (b ∣ a ∨ a ∣ b) ∧ (c ∣ b ∨ b ∣ c) ∧ (d ∣ c ∨ c ∣ d) ∧ (e ∣ d ∨ d ∣ e) ∧
  (f ∣ e ∨ e ∣ f) ∧ (g ∣ f ∨ f ∣ g) ∧ (a ∣ g ∨ g ∣ a) →
  ¬((a ∣ c ∨ c ∣ a) ∧ (a ∣ d ∨ d ∣ a) ∧ (a ∣ e ∨ e ∣ a) ∧ (a ∣ f ∨ f ∣ a) ∧
    (a ∣ g ∨ g ∣ a) ∧ (b ∣ d ∨ d ∣ b) ∧ (b ∣ e ∨ e ∣ b) ∧ (b ∣ f ∨ f ∣ b) ∧
    (b ∣ g ∨ g ∣ b) ∧ (c ∣ e ∨ e ∣ c) ∧ (c ∣ f ∨ f ∣ c) ∧ (c ∣ g ∨ g ∣ c) ∧
    (d ∣ f ∨ f ∣ d) ∧ (d ∣ g ∨ g ∣ d) ∧ (e ∣ g ∨ g ∣ e)) :=
 by
  sorry

end NUMINAMATH_GPT_heptagon_divisibility_impossible_l1392_139226


namespace NUMINAMATH_GPT_distance_from_P_to_AD_l1392_139277

-- Definitions of points and circles
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 0, y := 4}
def D : Point := {x := 0, y := 0}
def C : Point := {x := 4, y := 0}
def M : Point := {x := 2, y := 0}
def radiusM : ℝ := 2
def radiusA : ℝ := 4

-- Definition of the circles
def circleM (P : Point) : Prop := (P.x - M.x)^2 + P.y^2 = radiusM^2
def circleA (P : Point) : Prop := P.x^2 + (P.y - A.y)^2 = radiusA^2

-- Definition of intersection point \(P\) of the two circles
def is_intersection (P : Point) : Prop := circleM P ∧ circleA P

-- Distance from point \(P\) to line \(\overline{AD}\) computed as the x-coordinate
def distance_to_line_AD (P : Point) : ℝ := P.x

-- The theorem to prove
theorem distance_from_P_to_AD :
  ∃ P : Point, is_intersection P ∧ distance_to_line_AD P = 16/5 :=
by {
  -- Use "sorry" as the proof placeholder
  sorry
}

end NUMINAMATH_GPT_distance_from_P_to_AD_l1392_139277


namespace NUMINAMATH_GPT_biscuit_dimensions_l1392_139290

theorem biscuit_dimensions (sheet_length : ℝ) (sheet_width : ℝ) (num_biscuits : ℕ) 
  (h₁ : sheet_length = 12) (h₂ : sheet_width = 12) (h₃ : num_biscuits = 16) :
  ∃ biscuit_length : ℝ, biscuit_length = 3 :=
by
  sorry

end NUMINAMATH_GPT_biscuit_dimensions_l1392_139290


namespace NUMINAMATH_GPT_correct_multiple_l1392_139220

theorem correct_multiple (n : ℝ) (m : ℝ) (h1 : n = 6) (h2 : m * n - 6 = 2 * n) : m * n = 18 :=
by
  sorry

end NUMINAMATH_GPT_correct_multiple_l1392_139220


namespace NUMINAMATH_GPT_green_chips_count_l1392_139214

def total_chips : ℕ := 60
def fraction_blue_chips : ℚ := 1 / 6
def num_red_chips : ℕ := 34

theorem green_chips_count :
  let num_blue_chips := total_chips * fraction_blue_chips
  let chips_not_green := num_blue_chips + num_red_chips
  let num_green_chips := total_chips - chips_not_green
  num_green_chips = 16 := by
    let num_blue_chips := total_chips * fraction_blue_chips
    let chips_not_green := num_blue_chips + num_red_chips
    let num_green_chips := total_chips - chips_not_green
    show num_green_chips = 16
    sorry

end NUMINAMATH_GPT_green_chips_count_l1392_139214


namespace NUMINAMATH_GPT_subset_interval_l1392_139215

theorem subset_interval (a : ℝ) : 
  (∀ x : ℝ, (-a-1 < x ∧ x < -a+1 → -3 < x ∧ x < 1)) ↔ (0 ≤ a ∧ a ≤ 2) := 
by
  sorry

end NUMINAMATH_GPT_subset_interval_l1392_139215


namespace NUMINAMATH_GPT_ratio_of_middle_angle_l1392_139241

theorem ratio_of_middle_angle (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : C = 5 * A)
  (h3 : A = 20) :
  B / A = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_middle_angle_l1392_139241


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l1392_139295

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l1392_139295


namespace NUMINAMATH_GPT_polynomial_sum_is_integer_l1392_139222

-- Define the integer polynomial and the integers a and b
variables (f : ℤ[X]) (a b : ℤ)

-- The theorem statement
theorem polynomial_sum_is_integer :
  ∃ c : ℤ, f.eval (a - real.sqrt b) + f.eval (a + real.sqrt b) = c :=
sorry

end NUMINAMATH_GPT_polynomial_sum_is_integer_l1392_139222


namespace NUMINAMATH_GPT_lines_intersect_at_point_l1392_139242

theorem lines_intersect_at_point :
  ∃ (x y : ℝ), (3 * x + 4 * y + 7 = 0) ∧ (x - 2 * y - 1 = 0) ∧ (x = -1) ∧ (y = -1) :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_at_point_l1392_139242


namespace NUMINAMATH_GPT_parallel_lines_slope_l1392_139208

theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2 * y + 1 = 0 → ∀ x y : ℝ, x + y - 2 = 0 → True) → 
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_l1392_139208


namespace NUMINAMATH_GPT_product_of_square_roots_l1392_139289
-- Importing the necessary Lean library

-- Declare the mathematical problem in Lean 4
theorem product_of_square_roots (x : ℝ) (hx : 0 ≤ x) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) :=
by
  sorry

end NUMINAMATH_GPT_product_of_square_roots_l1392_139289


namespace NUMINAMATH_GPT_gwen_spent_zero_l1392_139275

theorem gwen_spent_zero 
  (m : ℕ) 
  (d : ℕ) 
  (S : ℕ) 
  (h1 : m = 8) 
  (h2 : d = 5)
  (h3 : (m - S) = (d - S) + 3) : 
  S = 0 :=
by
  sorry

end NUMINAMATH_GPT_gwen_spent_zero_l1392_139275


namespace NUMINAMATH_GPT_opposite_sides_of_line_l1392_139247

theorem opposite_sides_of_line (m : ℝ) 
  (ha : (m + 0 - 1) * (2 + m - 1) < 0): 
  -1 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_opposite_sides_of_line_l1392_139247


namespace NUMINAMATH_GPT_sum_of_3digit_numbers_remainder_2_l1392_139265

-- Define the smallest and largest three-digit numbers leaving remainder 2 when divided by 5
def smallest : ℕ := 102
def largest  : ℕ := 997
def common_diff : ℕ := 5

-- Define the arithmetic sequence
def seq_length : ℕ := ((largest - smallest) / common_diff) + 1
def sequence_sum : ℕ := seq_length * (smallest + largest) / 2

-- The theorem to be proven
theorem sum_of_3digit_numbers_remainder_2 : sequence_sum = 98910 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_3digit_numbers_remainder_2_l1392_139265


namespace NUMINAMATH_GPT_symmetric_point_l1392_139236

theorem symmetric_point (A B C : ℝ) (hA : A = Real.sqrt 7) (hB : B = 1) :
  C = 2 - Real.sqrt 7 ↔ (A + C) / 2 = B :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_l1392_139236


namespace NUMINAMATH_GPT_snail_distance_round_100_l1392_139294

def snail_distance (n : ℕ) : ℕ :=
  if n = 0 then 100 else (100 * (n + 2)) / (n + 1)

theorem snail_distance_round_100 : snail_distance 100 = 5050 :=
  sorry

end NUMINAMATH_GPT_snail_distance_round_100_l1392_139294


namespace NUMINAMATH_GPT_pencils_per_row_l1392_139216

-- Define the conditions as parameters
variables (total_pencils : Int) (rows : Int) 

-- State the proof problem using the conditions and the correct answer
theorem pencils_per_row (h₁ : total_pencils = 12) (h₂ : rows = 3) : total_pencils / rows = 4 := 
by 
  sorry

end NUMINAMATH_GPT_pencils_per_row_l1392_139216


namespace NUMINAMATH_GPT_puppies_adopted_each_day_l1392_139207

variable (initial_puppies additional_puppies days total_puppies puppies_per_day : ℕ)

axiom initial_puppies_ax : initial_puppies = 9
axiom additional_puppies_ax : additional_puppies = 12
axiom days_ax : days = 7
axiom total_puppies_ax : total_puppies = initial_puppies + additional_puppies
axiom adoption_rate_ax : total_puppies / days = puppies_per_day

theorem puppies_adopted_each_day : 
  initial_puppies = 9 → additional_puppies = 12 → days = 7 → total_puppies = initial_puppies + additional_puppies → total_puppies / days = puppies_per_day → puppies_per_day = 3 :=
by
  intro initial_puppies_ax additional_puppies_ax days_ax total_puppies_ax adoption_rate_ax
  sorry

end NUMINAMATH_GPT_puppies_adopted_each_day_l1392_139207


namespace NUMINAMATH_GPT_bird_count_l1392_139274

def initial_birds : ℕ := 12
def new_birds : ℕ := 8
def total_birds : ℕ := initial_birds + new_birds

theorem bird_count : total_birds = 20 := by
  sorry

end NUMINAMATH_GPT_bird_count_l1392_139274


namespace NUMINAMATH_GPT_queenie_worked_4_days_l1392_139240

-- Conditions
def daily_earning : ℕ := 150
def overtime_rate : ℕ := 5
def overtime_hours : ℕ := 4
def total_pay : ℕ := 770

-- Question
def number_of_days_worked (d : ℕ) : Prop := 
  daily_earning * d + overtime_rate * overtime_hours * d = total_pay

-- Theorem statement
theorem queenie_worked_4_days : ∃ d : ℕ, number_of_days_worked d ∧ d = 4 := 
by 
  use 4
  unfold number_of_days_worked 
  sorry

end NUMINAMATH_GPT_queenie_worked_4_days_l1392_139240


namespace NUMINAMATH_GPT_max_S_value_max_S_value_achievable_l1392_139204

theorem max_S_value (x y z w : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) (hw : 0 ≤ w ∧ w ≤ 1) :
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) ≤ 8 / 27 :=
sorry

theorem max_S_value_achievable :
  ∃ (x y z w : ℝ), (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) ∧ (0 ≤ w ∧ w ≤ 1) ∧ 
  (x^2 * y + y^2 * z + z^2 * w + w^2 * x - x * y^2 - y * z^2 - z * w^2 - w * x^2) = 8 / 27 :=
sorry

end NUMINAMATH_GPT_max_S_value_max_S_value_achievable_l1392_139204


namespace NUMINAMATH_GPT_find_S9_l1392_139239

variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Condition: an arithmetic sequence with the sum of first n terms S_n.
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition: a_3 + a_4 + a_5 + a_6 + a_7 = 20.
def given_condition (a : ℕ → ℝ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 20

-- The sum of the first n terms.
def sum_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n / 2 : ℝ) * (a 1 + a n)

-- Prove that S_9 = 36.
theorem find_S9 (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_arithmetic_sequence : arithmetic_sequence a) 
  (h_given_condition : given_condition a)
  (h_sum_terms : sum_terms S a) : 
  S 9 = 36 :=
sorry

end NUMINAMATH_GPT_find_S9_l1392_139239


namespace NUMINAMATH_GPT_crossed_out_number_is_29_l1392_139221

theorem crossed_out_number_is_29 : 
  ∀ n : ℕ, (11 * n + 66 - (325 - (12 * n + 66 - 325))) = 29 :=
by sorry

end NUMINAMATH_GPT_crossed_out_number_is_29_l1392_139221


namespace NUMINAMATH_GPT_sample_size_calculation_l1392_139272

/--
A factory produces three different models of products: A, B, and C. The ratio of their quantities is 2:3:5.
Using stratified sampling, a sample of size n is drawn, and it contains 16 units of model A.
We need to prove that the sample size n is 80.
-/
theorem sample_size_calculation
  (k : ℕ)
  (hk : 2 * k = 16)
  (n : ℕ)
  (hn : n = (2 + 3 + 5) * k) :
  n = 80 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_calculation_l1392_139272


namespace NUMINAMATH_GPT_arithmetic_sequence_and_sum_l1392_139252

noncomputable def a_n (n : ℕ) : ℤ := 2 * n + 10

def S_n (n : ℕ) : ℤ := n * (12 + 2 * n + 10) / 2

theorem arithmetic_sequence_and_sum :
    (a_n 10 = 30) ∧ 
    (a_n 20 = 50) ∧ 
    (∀ n, S_n n = 11 * n + n^2) ∧ 
    (S_n 3 = 42) :=
by {
    -- a_n 10 = 2 * 10 + 10 = 30
    -- a_n 20 = 2 * 20 + 10 = 50
    -- S_n n = n * (2n + 22) / 2 = 11n + n^2
    -- S_n 3 = 3 * 14 = 42
    sorry
}

end NUMINAMATH_GPT_arithmetic_sequence_and_sum_l1392_139252


namespace NUMINAMATH_GPT_distinct_positive_roots_l1392_139259

noncomputable def f (a x : ℝ) : ℝ := x^4 - x^3 + 8 * a * x^2 - a * x + a^2

theorem distinct_positive_roots (a : ℝ) :
  0 < a ∧ a < 1/24 → (∀ x1 x2 x3 x4 : ℝ, f a x1 = 0 ∧ 0 < x1 ∧ f a x2 = 0 ∧ 0 < x2 ∧ f a x3 = 0 ∧ 0 < x3 ∧ f a x4 = 0 ∧ 0 < x4 ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ↔ (1/25 < a ∧ a < 1/24) :=
sorry

end NUMINAMATH_GPT_distinct_positive_roots_l1392_139259


namespace NUMINAMATH_GPT_total_revenue_proof_l1392_139249

-- Define constants for the problem
def original_price_per_case : ℝ := 25
def first_group_customers : ℕ := 8
def first_group_cases_per_customer : ℕ := 3
def first_group_discount_percentage : ℝ := 0.15
def second_group_customers : ℕ := 4
def second_group_cases_per_customer : ℕ := 2
def second_group_discount_percentage : ℝ := 0.10
def third_group_customers : ℕ := 8
def third_group_cases_per_customer : ℕ := 1

-- Calculate the prices after discount
def discounted_price_first_group : ℝ := original_price_per_case * (1 - first_group_discount_percentage)
def discounted_price_second_group : ℝ := original_price_per_case * (1 - second_group_discount_percentage)
def regular_price : ℝ := original_price_per_case

-- Calculate the total revenue
def total_revenue_first_group : ℝ := first_group_customers * first_group_cases_per_customer * discounted_price_first_group
def total_revenue_second_group : ℝ := second_group_customers * second_group_cases_per_customer * discounted_price_second_group
def total_revenue_third_group : ℝ := third_group_customers * third_group_cases_per_customer * regular_price

def total_revenue : ℝ := total_revenue_first_group + total_revenue_second_group + total_revenue_third_group

-- Prove that the total revenue is $890
theorem total_revenue_proof : total_revenue = 890 := by
  sorry

end NUMINAMATH_GPT_total_revenue_proof_l1392_139249


namespace NUMINAMATH_GPT_tan_105_degree_l1392_139217

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_105_degree_l1392_139217


namespace NUMINAMATH_GPT_cube_diagonal_length_l1392_139266

theorem cube_diagonal_length (s : ℝ) 
    (h₁ : 6 * s^2 = 54) 
    (h₂ : 12 * s = 36) :
    ∃ d : ℝ, d = 3 * Real.sqrt 3 ∧ d = Real.sqrt (3 * s^2) :=
by
  sorry

end NUMINAMATH_GPT_cube_diagonal_length_l1392_139266


namespace NUMINAMATH_GPT_easter_egg_problem_l1392_139213

-- Define the conditions as assumptions
def total_eggs : Nat := 63
def helen_eggs (H : Nat) := H
def hannah_eggs (H : Nat) := 2 * H
def harry_eggs (H : Nat) := 2 * H + 3

-- The theorem stating the proof problem
theorem easter_egg_problem (H : Nat) (hh : hannah_eggs H + helen_eggs H + harry_eggs H = total_eggs) : 
    helen_eggs H = 12 ∧ hannah_eggs H = 24 ∧ harry_eggs H = 27 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_easter_egg_problem_l1392_139213


namespace NUMINAMATH_GPT_libby_quarters_left_after_payment_l1392_139235

noncomputable def quarters_needed (usd_target : ℝ) (usd_per_quarter : ℝ) : ℝ := 
  usd_target / usd_per_quarter

noncomputable def quarters_left (initial_quarters : ℝ) (used_quarters : ℝ) : ℝ := 
  initial_quarters - used_quarters

theorem libby_quarters_left_after_payment
  (initial_quarters : ℝ) (usd_target : ℝ) (usd_per_quarter : ℝ) 
  (h_initial : initial_quarters = 160) 
  (h_usd_target : usd_target = 35) 
  (h_usd_per_quarter : usd_per_quarter = 0.25) : 
  quarters_left initial_quarters (quarters_needed usd_target usd_per_quarter) = 20 := 
by
  sorry

end NUMINAMATH_GPT_libby_quarters_left_after_payment_l1392_139235


namespace NUMINAMATH_GPT_max_volume_of_pyramid_l1392_139238

theorem max_volume_of_pyramid
  (a b c : ℝ)
  (h1 : a + b + c = 9)
  (h2 : ∀ (α β : ℝ), α = 30 ∧ β = 45)
  : ∃ V, V = (9 * Real.sqrt 2) / 4 ∧ V = (1/6) * (Real.sqrt 2 / 2) * a * b * c :=
by
  sorry

end NUMINAMATH_GPT_max_volume_of_pyramid_l1392_139238


namespace NUMINAMATH_GPT_total_pages_in_storybook_l1392_139288

theorem total_pages_in_storybook
  (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) (n : ℕ) (Sₙ : ℕ) 
  (h₁ : a₁ = 12)
  (h₂ : d = 1)
  (h₃ : aₙ = 26)
  (h₄ : aₙ = a₁ + (n - 1) * d)
  (h₅ : Sₙ = n * (a₁ + aₙ) / 2) :
  Sₙ = 285 :=
by
  sorry

end NUMINAMATH_GPT_total_pages_in_storybook_l1392_139288


namespace NUMINAMATH_GPT_hexagon_coloring_l1392_139286

def valid_coloring_hexagon : Prop :=
  ∃ (A B C D E F : Fin 8), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ A ≠ D ∧ B ≠ D ∧ C ≠ D ∧
    B ≠ E ∧ C ≠ E ∧ D ≠ E ∧ A ≠ F ∧ C ≠ F ∧ E ≠ F

theorem hexagon_coloring : ∃ (n : Nat), valid_coloring_hexagon ∧ n = 20160 := 
sorry

end NUMINAMATH_GPT_hexagon_coloring_l1392_139286


namespace NUMINAMATH_GPT_circle_radius_increase_l1392_139234

-- Defining the problem conditions and the resulting proof
theorem circle_radius_increase (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) : r = n * (Real.sqrt 3 - 1) / 2 :=
sorry  -- Proof is left as an exercise

end NUMINAMATH_GPT_circle_radius_increase_l1392_139234


namespace NUMINAMATH_GPT_evaluate_expression_l1392_139270

theorem evaluate_expression (b : ℝ) (hb : b = -3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 :=
by 
  -- Lean code typically begins the proof block here
  sorry  -- The proof itself is omitted

end NUMINAMATH_GPT_evaluate_expression_l1392_139270


namespace NUMINAMATH_GPT_mass_percentage_of_O_in_CaCO3_l1392_139251

-- Assuming the given conditions as definitions
def molar_mass_Ca : ℝ := 40.08
def molar_mass_C : ℝ := 12.01
def molar_mass_O : ℝ := 16.00
def formula_CaCO3 : (ℕ × ℝ) := (1, molar_mass_Ca) -- 1 atom of Calcium
def formula_CaCO3_C : (ℕ × ℝ) := (1, molar_mass_C) -- 1 atom of Carbon
def formula_CaCO3_O : (ℕ × ℝ) := (3, molar_mass_O) -- 3 atoms of Oxygen

-- Desired result
def mass_percentage_O_CaCO3 : ℝ := 47.95

-- The theorem statement to be proven
theorem mass_percentage_of_O_in_CaCO3 :
  let molar_mass_CaCO3 := formula_CaCO3.2 + formula_CaCO3_C.2 + (formula_CaCO3_O.1 * formula_CaCO3_O.2)
  let mass_percentage_O := (formula_CaCO3_O.1 * formula_CaCO3_O.2 / molar_mass_CaCO3) * 100
  mass_percentage_O = mass_percentage_O_CaCO3 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_O_in_CaCO3_l1392_139251


namespace NUMINAMATH_GPT_altitude_difference_l1392_139276

theorem altitude_difference 
  (alt_A : ℤ) (alt_B : ℤ) (alt_C : ℤ)
  (hA : alt_A = -102) (hB : alt_B = -80) (hC : alt_C = -25) :
  (max (max alt_A alt_B) alt_C) - (min (min alt_A alt_B) alt_C) = 77 := 
by 
  sorry

end NUMINAMATH_GPT_altitude_difference_l1392_139276


namespace NUMINAMATH_GPT_subtraction_of_twos_from_ones_l1392_139244

theorem subtraction_of_twos_from_ones (n : ℕ) : 
  let ones := (10^n - 1) * 10^n + (10^n - 1)
  let twos := 2 * (10^n - 1)
  ones - twos = (10^n - 1) * (10^n - 1) :=
by
  sorry

end NUMINAMATH_GPT_subtraction_of_twos_from_ones_l1392_139244


namespace NUMINAMATH_GPT_smallest_constant_N_l1392_139280

theorem smallest_constant_N (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a + b > c) (h₅ : b + c > a) (h₆ : c + a > b) :
  (a^2 + b^2 + c^2) / (a * b + b * c + c * a) > 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_constant_N_l1392_139280


namespace NUMINAMATH_GPT_inequality_int_part_l1392_139248

theorem inequality_int_part (a : ℝ) (n : ℕ) (h1 : 1 ≤ a) (h2 : (0 : ℝ) ≤ n ∧ (n : ℝ) ≤ a) : 
  ⌊a⌋ > (n / (n + 1 : ℝ)) * a := 
by 
  sorry

end NUMINAMATH_GPT_inequality_int_part_l1392_139248


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1392_139237

def set_A : Set ℝ := {x : ℝ | x^2 - 5 * x + 6 > 0}
def set_B : Set ℝ := {x : ℝ | x < 1}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | x < 1} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1392_139237


namespace NUMINAMATH_GPT_container_solution_exists_l1392_139225

theorem container_solution_exists (x y : ℕ) (h : 130 * x + 160 * y = 3000) : 
  (x = 12) ∧ (y = 9) :=
by sorry

end NUMINAMATH_GPT_container_solution_exists_l1392_139225


namespace NUMINAMATH_GPT_fare_midpoint_to_b_l1392_139299

-- Define the conditions
def initial_fare : ℕ := 5
def initial_distance : ℕ := 2
def additional_fare_per_km : ℕ := 2
def total_fare : ℕ := 35
def walked_distance_meters : ℕ := 800

-- Define the correct answer
def fare_from_midpoint_to_b : ℕ := 19

-- Prove that the fare from the midpoint between A and B to B is 19 yuan
theorem fare_midpoint_to_b (y : ℝ) (h1 : 16.8 < y ∧ y ≤ 17) : 
  let half_distance := y / 2
  let total_taxi_distance := half_distance - 2
  let total_additional_fare := ⌈total_taxi_distance⌉ * additional_fare_per_km
  initial_fare + total_additional_fare = fare_from_midpoint_to_b := 
by
  sorry

end NUMINAMATH_GPT_fare_midpoint_to_b_l1392_139299


namespace NUMINAMATH_GPT_bus_trip_length_l1392_139271

theorem bus_trip_length (v T : ℝ) 
    (h1 : 2 * v + (T - 2 * v) * (3 / (2 * v)) + 1 = T / v + 5)
    (h2 : 2 + 30 / v + (T - (2 * v + 30)) * (3 / (2 * v)) + 1 = T / v + 4) : 
    T = 180 :=
    sorry

end NUMINAMATH_GPT_bus_trip_length_l1392_139271


namespace NUMINAMATH_GPT_new_person_weight_is_55_l1392_139253

variable (W : ℝ) -- Total weight of the original 8 people
variable (new_person_weight : ℝ) -- Weight of the new person
variable (avg_increase : ℝ := 2.5) -- The average weight increase

-- Given conditions
def condition (W new_person_weight : ℝ) : Prop :=
  new_person_weight = W + (8 * avg_increase) + 35 - W

-- The proof statement
theorem new_person_weight_is_55 (W : ℝ) : (new_person_weight = 55) :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_is_55_l1392_139253


namespace NUMINAMATH_GPT_solve_system_eqn_l1392_139285

theorem solve_system_eqn (x y : ℚ) (h₁ : 3*y - 4*x = 8) (h₂ : 2*y + x = -1) :
  x = -19/11 ∧ y = 4/11 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_eqn_l1392_139285


namespace NUMINAMATH_GPT_rate_of_rainfall_on_Monday_l1392_139200

theorem rate_of_rainfall_on_Monday (R : ℝ) :
  7 * R + 4 * 2 + 2 * (2 * 2) = 23 → R = 1 := 
by
  sorry

end NUMINAMATH_GPT_rate_of_rainfall_on_Monday_l1392_139200


namespace NUMINAMATH_GPT_remaining_area_l1392_139287

-- Given a regular hexagon and a rhombus composed of two equilateral triangles.
-- Hexagon area is 135 square centimeters.

variable (hexagon_area : ℝ) (rhombus_area : ℝ)
variable (is_regular_hexagon : Prop) (is_composed_of_two_equilateral_triangles : Prop)

-- The conditions
def correct_hexagon_area := hexagon_area = 135
def rhombus_is_composed := is_composed_of_two_equilateral_triangles = true
def hexagon_is_regular := is_regular_hexagon = true

-- Goal: Remaining area after cutting out the rhombus should be 75 square centimeters
theorem remaining_area : 
  correct_hexagon_area hexagon_area →
  hexagon_is_regular is_regular_hexagon →
  rhombus_is_composed is_composed_of_two_equilateral_triangles →
  hexagon_area - rhombus_area = 75 :=
by
  sorry

end NUMINAMATH_GPT_remaining_area_l1392_139287


namespace NUMINAMATH_GPT_fraction_identity_l1392_139212

theorem fraction_identity (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1392_139212


namespace NUMINAMATH_GPT_sqrt_of_square_neg_five_eq_five_l1392_139209

theorem sqrt_of_square_neg_five_eq_five :
  Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_of_square_neg_five_eq_five_l1392_139209


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l1392_139282

-- Define the vertices P, Q, and R
def P : (Int × Int) := (-3, 2)
def Q : (Int × Int) := (1, 7)
def R : (Int × Int) := (3, -1)

-- Define the formula for the area of a triangle given vertices
def triangle_area (A B C : Int × Int) : Real :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Define the statement to prove
theorem area_of_triangle_PQR : triangle_area P Q R = 21 := 
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l1392_139282


namespace NUMINAMATH_GPT_math_problem_l1392_139223

def letters := "MATHEMATICS".toList

def vowels := "AAEII".toList
def consonants := "MTHMTCS".toList
def fixed_t := 'T'

def factorial (n : Nat) : Nat := 
  if n = 0 then 1 
  else n * factorial (n - 1)

def arrangements (n : Nat) (reps : List Nat) : Nat := 
  factorial n / reps.foldr (fun r acc => factorial r * acc) 1

noncomputable def vowel_arrangements := arrangements 5 [2, 2]
noncomputable def consonant_arrangements := arrangements 6 [2]

noncomputable def total_arrangements := vowel_arrangements * consonant_arrangements

theorem math_problem : total_arrangements = 10800 := by
  sorry

end NUMINAMATH_GPT_math_problem_l1392_139223


namespace NUMINAMATH_GPT_percentage_of_number_l1392_139246

theorem percentage_of_number (N : ℕ) (P : ℕ) (h1 : N = 120) (h2 : (3 * N) / 5 = 72) (h3 : (P * 72) / 100 = 36) : P = 50 :=
sorry

end NUMINAMATH_GPT_percentage_of_number_l1392_139246


namespace NUMINAMATH_GPT_inequality_solution_set_l1392_139260

theorem inequality_solution_set (x : ℝ) : 
  (x + 2) / (x - 1) ≤ 0 ↔ -2 ≤ x ∧ x < 1 := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1392_139260


namespace NUMINAMATH_GPT_randy_brother_ate_l1392_139258

-- Definitions
def initial_biscuits : ℕ := 32
def biscuits_from_father : ℕ := 13
def biscuits_from_mother : ℕ := 15
def remaining_biscuits : ℕ := 40

-- Theorem to prove
theorem randy_brother_ate : 
  initial_biscuits + biscuits_from_father + biscuits_from_mother - remaining_biscuits = 20 :=
by
  sorry

end NUMINAMATH_GPT_randy_brother_ate_l1392_139258


namespace NUMINAMATH_GPT_balls_in_base_l1392_139205

theorem balls_in_base (n k : ℕ) (h1 : 165 = (n * (n + 1) * (n + 2)) / 6) (h2 : k = n * (n + 1) / 2) : k = 45 := 
by 
  sorry

end NUMINAMATH_GPT_balls_in_base_l1392_139205


namespace NUMINAMATH_GPT_certain_number_is_47_l1392_139218

theorem certain_number_is_47 (x : ℤ) (h : 34 + x - 53 = 28) : x = 47 :=
by
  sorry

end NUMINAMATH_GPT_certain_number_is_47_l1392_139218


namespace NUMINAMATH_GPT_probability_of_usable_gas_pipe_l1392_139269

theorem probability_of_usable_gas_pipe (x y : ℝ)
  (hx : 75 ≤ x) 
  (hy : 75 ≤ y)
  (hxy : x + y ≤ 225) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x < 300 ∧ y < 300 ∧ x + y > 75 ∧ (300 - x - y) ≥ 75) → 
  ((150 * 150) / (300 * 300 / 2) = (1 / 4)) :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_usable_gas_pipe_l1392_139269


namespace NUMINAMATH_GPT_no_half_dimension_cuboid_l1392_139293

theorem no_half_dimension_cuboid
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ) (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0) :
  ¬ (a' * b' * c' = (1 / 2) * a * b * c ∧ 2 * (a' * b' + b' * c' + c' * a') = a * b + b * c + c * a) :=
by
  sorry

end NUMINAMATH_GPT_no_half_dimension_cuboid_l1392_139293


namespace NUMINAMATH_GPT_car_rental_cost_l1392_139284

theorem car_rental_cost (D R M P C : ℝ) (hD : D = 5) (hR : R = 30) (hM : M = 500) (hP : P = 0.25) 
(hC : C = (R * D) + (P * M)) : C = 275 :=
by
  rw [hD, hR, hM, hP] at hC
  sorry

end NUMINAMATH_GPT_car_rental_cost_l1392_139284


namespace NUMINAMATH_GPT_charlie_paints_60_sqft_l1392_139224

theorem charlie_paints_60_sqft (A B C : ℕ) (total_sqft : ℕ) (h_ratio : A = 3 ∧ B = 5 ∧ C = 2) (h_total : total_sqft = 300) : 
  C * (total_sqft / (A + B + C)) = 60 :=
by
  rcases h_ratio with ⟨rfl, rfl, rfl⟩
  rcases h_total with rfl
  sorry

end NUMINAMATH_GPT_charlie_paints_60_sqft_l1392_139224


namespace NUMINAMATH_GPT_math_time_more_than_science_l1392_139298

section ExamTimes

-- Define the number of questions and time in minutes for each subject
def num_english_questions := 60
def num_math_questions := 25
def num_science_questions := 35

def time_english_minutes := 100
def time_math_minutes := 120
def time_science_minutes := 110

-- Define the time per question for each subject
def time_per_question (total_time : ℕ) (num_questions : ℕ) : ℚ :=
  total_time / num_questions

def time_english_per_question := time_per_question time_english_minutes num_english_questions
def time_math_per_question := time_per_question time_math_minutes num_math_questions
def time_science_per_question := time_per_question time_science_minutes num_science_questions

-- Prove the additional time per Math question compared to Science question
theorem math_time_more_than_science : 
  (time_math_per_question - time_science_per_question) = 1.6571 := 
sorry

end ExamTimes

end NUMINAMATH_GPT_math_time_more_than_science_l1392_139298


namespace NUMINAMATH_GPT_picture_books_count_l1392_139262

-- Definitions based on the given conditions
def total_books : ℕ := 35
def fiction_books : ℕ := 5
def non_fiction_books : ℕ := fiction_books + 4
def autobiographies : ℕ := 2 * fiction_books
def total_non_picture_books : ℕ := fiction_books + non_fiction_books + autobiographies
def picture_books : ℕ := total_books - total_non_picture_books

-- Statement of the problem
theorem picture_books_count : picture_books = 11 :=
by sorry

end NUMINAMATH_GPT_picture_books_count_l1392_139262


namespace NUMINAMATH_GPT_multiply_of_Mari_buttons_l1392_139227

-- Define the variables and constants from the problem
def Mari_buttons : ℕ := 8
def Sue_buttons : ℕ := 22
def Kendra_buttons : ℕ := 2 * Sue_buttons

-- Statement that we need to prove
theorem multiply_of_Mari_buttons : ∃ (x : ℕ), Kendra_buttons = 8 * x + 4 ∧ x = 5 := by
  sorry

end NUMINAMATH_GPT_multiply_of_Mari_buttons_l1392_139227


namespace NUMINAMATH_GPT_zumish_12_words_remainder_l1392_139283

def zumishWords n :=
  if n < 2 then (0, 0, 0)
  else if n == 2 then (4, 4, 4)
  else let (a, b, c) := zumishWords (n - 1)
       (2 * (a + c) % 1000, 2 * a % 1000, 2 * b % 1000)

def countZumishWords (n : Nat) :=
  let (a, b, c) := zumishWords n
  (a + b + c) % 1000

theorem zumish_12_words_remainder :
  countZumishWords 12 = 322 :=
by
  intros
  sorry

end NUMINAMATH_GPT_zumish_12_words_remainder_l1392_139283


namespace NUMINAMATH_GPT_no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l1392_139264

theorem no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018 (m n : ℕ) : ¬ (m^2 = n^2 + 2018) :=
sorry

end NUMINAMATH_GPT_no_nat_numbers_m_n_m_squared_eq_n_squared_plus_2018_l1392_139264


namespace NUMINAMATH_GPT_find_vertex_D_l1392_139297

structure Point where
  x : ℤ
  y : ℤ

def vector_sub (a b : Point) : Point :=
  Point.mk (a.x - b.x) (a.y - b.y)

def vector_add (a b : Point) : Point :=
  Point.mk (a.x + b.x) (a.y + b.y)

def is_parallelogram (A B C D : Point) : Prop :=
  vector_sub B A = vector_sub D C

theorem find_vertex_D (A B C D : Point)
  (hA : A = Point.mk (-1) (-2))
  (hB : B = Point.mk 3 (-1))
  (hC : C = Point.mk 5 6)
  (hParallelogram: is_parallelogram A B C D) :
  D = Point.mk 1 5 :=
sorry

end NUMINAMATH_GPT_find_vertex_D_l1392_139297


namespace NUMINAMATH_GPT_hyperbola_equation_sum_l1392_139257

theorem hyperbola_equation_sum (h k a c b : ℝ) (h_h : h = 1) (h_k : k = 1) (h_a : a = 3) (h_c : c = 9) (h_c2 : c^2 = a^2 + b^2) :
    h + k + a + b = 5 + 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_sum_l1392_139257


namespace NUMINAMATH_GPT_sum_of_first_five_terms_l1392_139210

theorem sum_of_first_five_terms 
  (a₂ a₃ a₄ : ℤ)
  (h1 : a₂ = 4)
  (h2 : a₃ = 7)
  (h3 : a₄ = 10) :
  ∃ a1 a5, a1 + a₂ + a₃ + a₄ + a5 = 35 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_five_terms_l1392_139210


namespace NUMINAMATH_GPT_repeating_decimals_product_fraction_l1392_139203

theorem repeating_decimals_product_fraction : 
  let x := 1 / 33
  let y := 9 / 11
  x * y = 9 / 363 := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_product_fraction_l1392_139203


namespace NUMINAMATH_GPT_fractional_product_l1392_139281

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end NUMINAMATH_GPT_fractional_product_l1392_139281


namespace NUMINAMATH_GPT_prime_power_condition_l1392_139278

open Nat

theorem prime_power_condition (u v : ℕ) :
  (∃ p n : ℕ, p.Prime ∧ p^n = (u * v^3) / (u^2 + v^2)) ↔ ∃ k : ℕ, k ≥ 1 ∧ u = 2^k ∧ v = 2^k := by {
  sorry
}

end NUMINAMATH_GPT_prime_power_condition_l1392_139278


namespace NUMINAMATH_GPT_small_to_large_circle_ratio_l1392_139230

theorem small_to_large_circle_ratio (a b : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : π * b^2 - π * a^2 = 5 * π * a^2) :
  a / b = 1 / Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_small_to_large_circle_ratio_l1392_139230


namespace NUMINAMATH_GPT_cindy_envelopes_l1392_139232

theorem cindy_envelopes (h₁ : ℕ := 4) (h₂ : ℕ := 7) (h₃ : ℕ := 5) (h₄ : ℕ := 10) (h₅ : ℕ := 3) (initial : ℕ := 137) :
  initial - (h₁ + h₂ + h₃ + h₄ + h₅) = 108 :=
by
  sorry

end NUMINAMATH_GPT_cindy_envelopes_l1392_139232


namespace NUMINAMATH_GPT_exists_n_for_dvd_ka_pow_n_add_n_l1392_139256

theorem exists_n_for_dvd_ka_pow_n_add_n 
  (a k : ℕ) (a_pos : 0 < a) (k_pos : 0 < k) (d : ℕ) (d_pos : 0 < d) :
  ∃ n : ℕ, 0 < n ∧ d ∣ k * (a ^ n) + n :=
by
  sorry

end NUMINAMATH_GPT_exists_n_for_dvd_ka_pow_n_add_n_l1392_139256


namespace NUMINAMATH_GPT_inverse_proportionality_l1392_139273

theorem inverse_proportionality (a b c k a1 a2 b1 b2 c1 c2 : ℝ)
    (h1 : a * b * c = k)
    (h2 : a1 / a2 = 3 / 4)
    (h3 : b1 = 2 * b2)
    (h4 : c1 ≠ 0 ∧ c2 ≠ 0) :
    c1 / c2 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_inverse_proportionality_l1392_139273


namespace NUMINAMATH_GPT_alcohol_mixture_l1392_139243

theorem alcohol_mixture:
  ∃ (x y z: ℝ), 
    0.10 * x + 0.30 * y + 0.50 * z = 157.5 ∧
    x + y + z = 450 ∧
    x = y ∧
    x = 112.5 ∧
    y = 112.5 ∧
    z = 225 :=
sorry

end NUMINAMATH_GPT_alcohol_mixture_l1392_139243


namespace NUMINAMATH_GPT_A_times_B_is_correct_l1392_139245

noncomputable def A : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}
noncomputable def B : Set ℝ := {x : ℝ | x ≥ 0}

noncomputable def A_union_B : Set ℝ := {x : ℝ | x ≥ 0}
noncomputable def A_inter_B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

noncomputable def A_times_B : Set ℝ := {x : ℝ | x ∈ A_union_B ∧ x ∉ A_inter_B}

theorem A_times_B_is_correct :
  A_times_B = {x : ℝ | x > 2} := sorry

end NUMINAMATH_GPT_A_times_B_is_correct_l1392_139245


namespace NUMINAMATH_GPT_sum_term_ratio_equals_four_l1392_139291

variable {a_n : ℕ → ℝ} -- The arithmetic sequence a_n
variable {S_n : ℕ → ℝ} -- The sum of the first n terms S_n
variable {d : ℝ} -- The common difference of the sequence
variable {a_1 : ℝ} -- The first term of the sequence

-- The conditions as hypotheses
axiom a_n_formula (n : ℕ) : a_n n = a_1 + (n - 1) * d
axiom S_n_formula (n : ℕ) : S_n n = n * (a_1 + (n - 1) * d / 2)
axiom non_zero_d : d ≠ 0
axiom condition_a10_S4 : a_n 10 = S_n 4

-- The proof statement
theorem sum_term_ratio_equals_four : (S_n 8) / (a_n 9) = 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_term_ratio_equals_four_l1392_139291


namespace NUMINAMATH_GPT_donuts_purchased_l1392_139211

/-- John goes to a bakery every day for a four-day workweek and chooses between a 
    60-cent croissant or a 90-cent donut. At the end of the week, he spent a whole 
    number of dollars. Prove that he must have purchased 2 donuts. -/
theorem donuts_purchased (d c : ℕ) (h1 : d + c = 4) (h2 : 90 * d + 60 * c % 100 = 0) : d = 2 :=
sorry

end NUMINAMATH_GPT_donuts_purchased_l1392_139211


namespace NUMINAMATH_GPT_zebra_difference_is_zebra_l1392_139229

/-- 
A zebra number is a non-negative integer in which the digits strictly alternate between even and odd.
Given two 100-digit zebra numbers, prove that their difference is still a 100-digit zebra number.
-/
theorem zebra_difference_is_zebra 
  (A B : ℕ) 
  (hA : (∀ i, (A / 10^i % 10) % 2 = i % 2) ∧ (A / 10^100 = 0) ∧ (A > 10^99))
  (hB : (∀ i, (B / 10^i % 10) % 2 = i % 2) ∧ (B / 10^100 = 0) ∧ (B > 10^99)) 
  : (∀ j, (((A - B) / 10^j) % 10) % 2 = j % 2) ∧ ((A - B) / 10^100 = 0) ∧ ((A - B) > 10^99) :=
sorry

end NUMINAMATH_GPT_zebra_difference_is_zebra_l1392_139229


namespace NUMINAMATH_GPT_min_t_value_l1392_139296

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value : 
  ∀ (x y : ℝ), x ∈ Set.Icc (-3 : ℝ) (2 : ℝ) → y ∈ Set.Icc (-3 : ℝ) (2 : ℝ)
  → |f (x) - f (y)| ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_min_t_value_l1392_139296


namespace NUMINAMATH_GPT_black_haired_girls_count_l1392_139267

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end NUMINAMATH_GPT_black_haired_girls_count_l1392_139267


namespace NUMINAMATH_GPT_average_weight_of_whole_class_l1392_139228

def num_students_a : ℕ := 50
def num_students_b : ℕ := 70
def avg_weight_a : ℚ := 50
def avg_weight_b : ℚ := 70

theorem average_weight_of_whole_class :
  (num_students_a * avg_weight_a + num_students_b * avg_weight_b) / (num_students_a + num_students_b) = 61.67 := by
  sorry

end NUMINAMATH_GPT_average_weight_of_whole_class_l1392_139228


namespace NUMINAMATH_GPT_marble_probability_l1392_139206

theorem marble_probability (g w r b : ℕ) (h_g : g = 4) (h_w : w = 3) (h_r : r = 5) (h_b : b = 6) :
  (g + w + r + b = 18) → (g + w = 7) → (7 / 18 = 7 / 18) :=
by
  sorry

end NUMINAMATH_GPT_marble_probability_l1392_139206


namespace NUMINAMATH_GPT_time_at_simple_interest_l1392_139201

theorem time_at_simple_interest 
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * (R + 5) / 100) * T = (P * (R / 100) * T) + 150) : 
  T = 10 := 
by 
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_time_at_simple_interest_l1392_139201


namespace NUMINAMATH_GPT_john_needs_2_sets_l1392_139219

-- Definition of the conditions
def num_bars_per_set : ℕ := 7
def total_bars : ℕ := 14

-- The corresponding proof problem statement
theorem john_needs_2_sets : total_bars / num_bars_per_set = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_needs_2_sets_l1392_139219


namespace NUMINAMATH_GPT_tank_filling_time_l1392_139254

noncomputable def netWaterPerCycle (rateA rateB rateC : ℕ) : ℕ := rateA + rateB - rateC

noncomputable def totalTimeToFill (tankCapacity rateA rateB rateC cycleDuration : ℕ) : ℕ :=
  let netWater := netWaterPerCycle rateA rateB rateC
  let cyclesNeeded := tankCapacity / netWater
  cyclesNeeded * cycleDuration

theorem tank_filling_time :
  totalTimeToFill 750 40 30 20 3 = 45 :=
by
  -- replace "sorry" with the actual proof if required
  sorry

end NUMINAMATH_GPT_tank_filling_time_l1392_139254


namespace NUMINAMATH_GPT_economical_refuel_l1392_139231

theorem economical_refuel (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) : 
  (x + y) / 2 > (2 * x * y) / (x + y) :=
sorry -- Proof omitted

end NUMINAMATH_GPT_economical_refuel_l1392_139231


namespace NUMINAMATH_GPT_sum_exterior_angles_const_l1392_139202

theorem sum_exterior_angles_const (n : ℕ) (h : n ≥ 3) : 
  ∃ s : ℝ, s = 360 :=
by
  sorry

end NUMINAMATH_GPT_sum_exterior_angles_const_l1392_139202
