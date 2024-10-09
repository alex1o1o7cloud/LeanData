import Mathlib

namespace large_number_exponent_l812_81286

theorem large_number_exponent (h : 10000 = 10 ^ 4) : 10000 ^ 50 * 10 ^ 5 = 10 ^ 205 := 
by
  sorry

end large_number_exponent_l812_81286


namespace john_paid_more_than_jane_by_540_l812_81274

noncomputable def original_price : ℝ := 36.000000000000036
noncomputable def discount_percentage : ℝ := 0.10
noncomputable def tip_percentage : ℝ := 0.15

noncomputable def discounted_price : ℝ := original_price * (1 - discount_percentage)
noncomputable def john_tip : ℝ := original_price * tip_percentage
noncomputable def jane_tip : ℝ := discounted_price * tip_percentage

noncomputable def john_total_payment : ℝ := discounted_price + john_tip
noncomputable def jane_total_payment : ℝ := discounted_price + jane_tip

noncomputable def difference : ℝ := john_total_payment - jane_total_payment

theorem john_paid_more_than_jane_by_540 :
  difference = 0.5400000000000023 := sorry

end john_paid_more_than_jane_by_540_l812_81274


namespace geo_seq_fifth_term_l812_81298

theorem geo_seq_fifth_term (a : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a 3 = 3) :
  a 5 = 12 := 
sorry

end geo_seq_fifth_term_l812_81298


namespace least_xy_l812_81200

theorem least_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 1 / (3 * y) = 1 / 8) : x * y = 96 :=
sorry

end least_xy_l812_81200


namespace StacyBoughtPacks_l812_81225

theorem StacyBoughtPacks (sheets_per_pack days daily_printed_sheets total_packs : ℕ) 
  (h1 : sheets_per_pack = 240)
  (h2 : days = 6)
  (h3 : daily_printed_sheets = 80) 
  (h4 : total_packs = (days * daily_printed_sheets) / sheets_per_pack) : total_packs = 2 :=
by 
  sorry

end StacyBoughtPacks_l812_81225


namespace quotient_remainder_l812_81218

theorem quotient_remainder (x y : ℕ) (hx : 0 ≤ x) (hy : 0 < y) : 
  ∃ q r : ℕ, q ≥ 0 ∧ 0 ≤ r ∧ r < y ∧ x = q * y + r := by
  sorry

end quotient_remainder_l812_81218


namespace find_a1_range_a1_l812_81221

variables (a_1 : ℤ) (d : ℤ := -1) (S : ℕ → ℤ)

-- Definition of sum of first n terms in an arithmetic sequence
def arithmetic_sum (n : ℕ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

-- Definition of nth term in an arithmetic sequence
def arithmetic_nth_term (n : ℕ) : ℤ := a_1 + (n - 1) * d

-- Given conditions for the problems
axiom S_def : ∀ n, S n = arithmetic_sum a_1 d n

-- Problem 1: Proving a1 = 1 given S_5 = -5
theorem find_a1 (h : S 5 = -5) : a_1 = 1 :=
by
  sorry

-- Problem 2: Proving range of a1 given S_n ≤ a_n for any positive integer n
theorem range_a1 (h : ∀ n : ℕ, n > 0 → S n ≤ arithmetic_nth_term a_1 d n) : a_1 ≤ 0 :=
by
  sorry

end find_a1_range_a1_l812_81221


namespace stairs_climbed_l812_81237

theorem stairs_climbed (s v r : ℕ) 
  (h_s: s = 318) 
  (h_v: v = 18 + s / 2) 
  (h_r: r = 2 * v) 
  : s + v + r = 849 :=
by {
  sorry
}

end stairs_climbed_l812_81237


namespace find_larger_number_l812_81284

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2415) (h2 : L = 21 * S + 15) : L = 2535 := 
by
  sorry

end find_larger_number_l812_81284


namespace minimum_value_of_f_l812_81213

def f (x a : ℝ) : ℝ := abs (x + 1) + abs (a * x + 1)

theorem minimum_value_of_f (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3 / 2) →
  (∃ x : ℝ, f x a = 3 / 2) →
  (a = -1 / 2 ∨ a = -2) :=
by
  intros h1 h2
  sorry

end minimum_value_of_f_l812_81213


namespace min_groups_with_conditions_l812_81262

theorem min_groups_with_conditions (n a b m : ℕ) (h_n : n = 8) (h_a : a = 4) (h_b : b = 1) :
  m ≥ 2 :=
sorry

end min_groups_with_conditions_l812_81262


namespace sum_of_integers_eq_17_l812_81270

theorem sum_of_integers_eq_17 (a b : ℕ) (h1 : a * b + a + b = 87) 
  (h2 : Nat.gcd a b = 1) (h3 : a < 15) (h4 : b < 15) (h5 : Even a ∨ Even b) :
  a + b = 17 := 
sorry

end sum_of_integers_eq_17_l812_81270


namespace roots_abs_less_than_one_l812_81251

theorem roots_abs_less_than_one {a b : ℝ} 
    (h : |a| + |b| < 1) 
    (x1 x2 : ℝ) 
    (h_roots : x1 * x1 + a * x1 + b = 0) 
    (h_roots' : x2 * x2 + a * x2 + b = 0) 
    : |x1| < 1 ∧ |x2| < 1 := 
sorry

end roots_abs_less_than_one_l812_81251


namespace equation_D_is_linear_l812_81248

-- Definitions according to the given conditions
def equation_A (x y : ℝ) := x + 2 * y = 3
def equation_B (x : ℝ) := 3 * x - 2
def equation_C (x : ℝ) := x^2 + x = 6
def equation_D (x : ℝ) := (1 / 3) * x - 2 = 3

-- Properties of a linear equation
def is_linear (eq : ℝ → Prop) : Prop :=
∃ a b c : ℝ, (∃ x : ℝ, eq x = (a * x + b = c)) ∧ a ≠ 0

-- Specifying that equation_D is linear
theorem equation_D_is_linear : is_linear equation_D :=
by
  sorry

end equation_D_is_linear_l812_81248


namespace exists_x_for_integer_conditions_l812_81249

-- Define the conditions as functions in Lean
def is_int_div (a b : Int) : Prop := ∃ k : Int, a = b * k

-- The target statement in Lean 4
theorem exists_x_for_integer_conditions :
  ∃ t_1 : Int, ∃ x : Int, (x = 105 * t_1 + 52) ∧ 
    (is_int_div (x - 3) 7) ∧ 
    (is_int_div (x - 2) 5) ∧ 
    (is_int_div (x - 4) 3) :=
by 
  sorry

end exists_x_for_integer_conditions_l812_81249


namespace cos_alpha_value_l812_81276
open Real

theorem cos_alpha_value (α : ℝ) (h0 : 0 < α ∧ α < π / 2) 
  (h1 : sin (α - π / 6) = 1 / 3) : 
  cos α = (2 * sqrt 6 - 1) / 6 := 
by 
  sorry

end cos_alpha_value_l812_81276


namespace fgf_one_l812_81207

/-- Define the function f(x) = 5x + 2 --/
def f (x : ℝ) := 5 * x + 2

/-- Define the function g(x) = 3x - 1 --/
def g (x : ℝ) := 3 * x - 1

/-- Prove that f(g(f(1))) = 102 given the definitions of f and g --/
theorem fgf_one : f (g (f 1)) = 102 := by
  sorry

end fgf_one_l812_81207


namespace buns_distribution_not_equal_for_all_cases_l812_81253

theorem buns_distribution_not_equal_for_all_cases :
  ∀ (initial_buns : Fin 30 → ℕ),
  (∃ (p : ℕ → Fin 30 → Fin 30), 
    (∀ t, 
      (∀ i, 
        (initial_buns (p t i) = initial_buns i ∨ 
         initial_buns (p t i) = initial_buns i + 2 ∨ 
         initial_buns (p t i) = initial_buns i - 2))) → 
    ¬ ∀ n : Fin 30, initial_buns n = 2) := 
sorry

end buns_distribution_not_equal_for_all_cases_l812_81253


namespace floor_of_sum_eq_l812_81285

theorem floor_of_sum_eq (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0)
  (hxy : x^2 + y^2 = 2500) (hzw : z^2 + w^2 = 2500) (hxz : x * z = 1200) (hyw : y * w = 1200) :
  ⌊x + y + z + w⌋ = 140 := by
  sorry

end floor_of_sum_eq_l812_81285


namespace max_value_expression_l812_81246

theorem max_value_expression (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 * (a + b))) +
   Real.sqrt (Real.sqrt (b^2 * (b + c))) +
   Real.sqrt (Real.sqrt (c^2 * (c + d))) +
   Real.sqrt (Real.sqrt (d^2 * (d + a)))) ≤ 4 * Real.sqrt (Real.sqrt 2) := by
  sorry

end max_value_expression_l812_81246


namespace product_of_decimals_l812_81263

theorem product_of_decimals :
  0.5 * 0.8 = 0.40 :=
by
  -- Proof will go here; using sorry to skip for now
  sorry

end product_of_decimals_l812_81263


namespace inverse_proportion_increasing_implication_l812_81219

theorem inverse_proportion_increasing_implication (m x : ℝ) (h1 : x > 0) (h2 : ∀ x1 x2, x1 > 0 → x2 > 0 → x1 < x2 → (m + 3) / x1 < (m + 3) / x2) : m < -3 :=
by
  sorry

end inverse_proportion_increasing_implication_l812_81219


namespace area_of_sector_l812_81291

theorem area_of_sector {R θ: ℝ} (hR: R = 2) (hθ: θ = (2 * Real.pi) / 3) :
  (1 / 2) * R^2 * θ = (4 / 3) * Real.pi :=
by
  simp [hR, hθ]
  norm_num
  linarith

end area_of_sector_l812_81291


namespace geometric_series_common_ratio_l812_81279

theorem geometric_series_common_ratio (a S r : ℝ) (h1 : a = 512) (h2 : S = 3072) 
(h3 : S = a / (1 - r)) : r = 5/6 := 
sorry

end geometric_series_common_ratio_l812_81279


namespace digits_sum_is_15_l812_81255

theorem digits_sum_is_15 (f o g : ℕ) (h1 : f * 100 + o * 10 + g = 366) (h2 : 4 * (f * 100 + o * 10 + g) = 1464) (h3 : f < 10 ∧ o < 10 ∧ g < 10) :
  f + o + g = 15 :=
sorry

end digits_sum_is_15_l812_81255


namespace t_range_inequality_l812_81277

theorem t_range_inequality (t : ℝ) :
  (1/8) * (2 * t - t^2) ≤ -1/4 ∧ 3 - t^2 ≥ 2 ↔ -1 ≤ t ∧ t ≤ 1 - Real.sqrt 3 :=
by
  sorry

end t_range_inequality_l812_81277


namespace general_term_formula_sum_of_first_n_terms_l812_81260

noncomputable def a (n : ℕ) : ℕ :=
(n + 2^n)^2

theorem general_term_formula :
  ∀ n : ℕ, a n = n^2 + n * 2^(n+1) + 4^n :=
sorry

noncomputable def S (n : ℕ) : ℕ :=
(n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3

theorem sum_of_first_n_terms :
  ∀ n : ℕ, S n = (n-1) * 2^(n+2) + 4 + (4^(n+1) - 4) / 3 :=
sorry

end general_term_formula_sum_of_first_n_terms_l812_81260


namespace find_sqrt_abc_sum_l812_81205

theorem find_sqrt_abc_sum (a b c : ℝ) (h1 : b + c = 20) (h2 : c + a = 22) (h3 : a + b = 24) :
    Real.sqrt (a * b * c * (a + b + c)) = 206.1 := by
  sorry

end find_sqrt_abc_sum_l812_81205


namespace sum_of_roots_l812_81292

theorem sum_of_roots (r s t : ℝ) (h : 3 * r * s * t - 9 * (r * s + s * t + t * r) - 28 * (r + s + t) + 12 = 0) : r + s + t = 3 :=
by sorry

end sum_of_roots_l812_81292


namespace ratio_of_medians_to_sides_l812_81201

theorem ratio_of_medians_to_sides (a b c : ℝ) (m_a m_b m_c : ℝ) 
  (h1: m_a = 1/2 * (2 * b^2 + 2 * c^2 - a^2)^(1/2))
  (h2: m_b = 1/2 * (2 * a^2 + 2 * c^2 - b^2)^(1/2))
  (h3: m_c = 1/2 * (2 * a^2 + 2 * b^2 - c^2)^(1/2)) :
  (m_a*m_a + m_b*m_b + m_c*m_c) / (a*a + b*b + c*c) = 3/4 := 
by 
  sorry

end ratio_of_medians_to_sides_l812_81201


namespace geometric_progression_sum_eq_l812_81242

theorem geometric_progression_sum_eq
  (a q b : ℝ) (n : ℕ)
  (hq : q ≠ 1)
  (h : (a * (q^2^n - 1)) / (q - 1) = (b * (q^(2*n) - 1)) / (q^2 - 1)) :
  b = a + a * q :=
by
  sorry

end geometric_progression_sum_eq_l812_81242


namespace check_correct_l812_81222

-- Given the conditions
variable (x y : ℕ) (H1 : 10 ≤ x ∧ x ≤ 81) (H2 : y = x + 18)

-- Rewrite the problem and correct answer for verification in Lean
theorem check_correct (Hx : 10 ≤ x ∧ x ≤ 81) (Hy : y = x + 18) : 
  y = 2 * x ↔ x = 18 := 
by
  sorry

end check_correct_l812_81222


namespace percentage_passed_eng_students_l812_81227

variable (total_male_students : ℕ := 120)
variable (total_female_students : ℕ := 100)
variable (total_international_students : ℕ := 70)
variable (total_disabilities_students : ℕ := 30)

variable (male_eng_percentage : ℕ := 25)
variable (female_eng_percentage : ℕ := 20)
variable (intern_eng_percentage : ℕ := 15)
variable (disab_eng_percentage : ℕ := 10)

variable (male_pass_percentage : ℕ := 20)
variable (female_pass_percentage : ℕ := 25)
variable (intern_pass_percentage : ℕ := 30)
variable (disab_pass_percentage : ℕ := 35)

def total_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100) +
  (total_female_students * female_eng_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100)

def total_passed_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100 * male_pass_percentage / 100) +
  (total_female_students * female_eng_percentage / 100 * female_pass_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100 * intern_pass_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100 * disab_pass_percentage / 100)

def passed_eng_students_percentage : ℕ :=
  total_passed_engineering_students * 100 / total_engineering_students

theorem percentage_passed_eng_students :
  passed_eng_students_percentage = 23 :=
sorry

end percentage_passed_eng_students_l812_81227


namespace production_increase_l812_81204

theorem production_increase (h1 : ℝ) (h2 : ℝ) (h3 : h1 = 0.75) (h4 : h2 = 0.5) :
  (h1 + h2 - 1) = 0.25 := by
  sorry

end production_increase_l812_81204


namespace Gwen_walking_and_elevation_gain_l812_81281

theorem Gwen_walking_and_elevation_gain :
  ∀ (jogging_time walking_time total_time elevation_gain : ℕ)
    (jogging_feet total_feet : ℤ),
    jogging_time = 15 ∧ jogging_feet = 500 ∧ (jogging_time + walking_time = total_time) ∧
    (5 * walking_time = 3 * jogging_time) ∧ (total_time * jogging_feet = 15 * total_feet)
    → walking_time = 9 ∧ total_feet = 800 := by 
  sorry

end Gwen_walking_and_elevation_gain_l812_81281


namespace factorize_expression_l812_81224

theorem factorize_expression (a : ℝ) : 2 * a^2 - 8 = 2 * (a + 2) * (a - 2) :=
by
  -- skipping the proof
  sorry

end factorize_expression_l812_81224


namespace john_has_22_quarters_l812_81297

-- Definitions based on conditions
def number_of_quarters (Q : ℕ) : ℕ := Q
def number_of_dimes (Q : ℕ) : ℕ := Q + 3
def number_of_nickels (Q : ℕ) : ℕ := Q - 6

-- Total number of coins condition
def total_number_of_coins (Q : ℕ) : Prop := 
  (number_of_quarters Q) + (number_of_dimes Q) + (number_of_nickels Q) = 63

-- Goal: Proving the number of quarters is 22
theorem john_has_22_quarters : ∃ Q : ℕ, total_number_of_coins Q ∧ Q = 22 :=
by
  -- Proof skipped 
  sorry

end john_has_22_quarters_l812_81297


namespace star_value_when_c_2_d_3_l812_81258

def star (c d : ℕ) : ℕ := c^3 + 3*c^2*d + 3*c*d^2 + d^3

theorem star_value_when_c_2_d_3 :
  star 2 3 = 125 :=
by
  sorry

end star_value_when_c_2_d_3_l812_81258


namespace perfect_square_factors_450_l812_81261

theorem perfect_square_factors_450 :
  ∃ (n : ℕ), n = 4 ∧ ∀ (d : ℕ), d ∣ 450 → ∃ (k : ℕ), k^2 = d ↔ d = 1 ∨ d = 9 ∨ d = 25 ∨ d = 225 :=
by
  sorry

end perfect_square_factors_450_l812_81261


namespace distance_from_origin_is_correct_l812_81212

-- Define the point (x, y) with given conditions
variables (x y : ℝ)

-- Given conditions
axiom h1 : y = 20
axiom h2 : dist (x, y) (2, 15) = 15
axiom h3 : x > 2

-- The theorem to prove
theorem distance_from_origin_is_correct :
  dist (x, y) (0, 0) = Real.sqrt (604 + 40 * Real.sqrt 2) :=
by
  -- Set h1, h2, and h3 as our constraints
  sorry

end distance_from_origin_is_correct_l812_81212


namespace _l812_81229

noncomputable def X := 0
noncomputable def Y := 1
noncomputable def Z := 2

noncomputable def angle_XYZ (X Y Z : ℝ) : ℝ := 90 -- Triangle XYZ where ∠X = 90°

noncomputable def length_YZ := 10 -- YZ = 10 units
noncomputable def length_XY := 6 -- XY = 6 units
noncomputable def length_XZ : ℝ := Real.sqrt (length_YZ^2 - length_XY^2) -- Pythagorean theorem to find XZ
noncomputable def cos_Z : ℝ := length_XZ / length_YZ -- cos Z = adjacent/hypotenuse

example : cos_Z = 0.8 :=
by {
  sorry
}

end _l812_81229


namespace donovan_lap_time_is_45_l812_81264

-- Definitions based on the conditions
def circular_track_length : ℕ := 600
def michael_lap_time : ℕ := 40
def michael_laps_to_pass_donovan : ℕ := 9

-- The theorem to prove
theorem donovan_lap_time_is_45 : ∃ D : ℕ, 8 * D = michael_laps_to_pass_donovan * michael_lap_time ∧ D = 45 := by
  sorry

end donovan_lap_time_is_45_l812_81264


namespace ribbon_tying_length_l812_81217

theorem ribbon_tying_length :
  let l1 := 36
  let l2 := 42
  let l3 := 48
  let cut1 := l1 / 6
  let cut2 := l2 / 6
  let cut3 := l3 / 6
  let rem1 := l1 - cut1
  let rem2 := l2 - cut2
  let rem3 := l3 - cut3
  let total_rem := rem1 + rem2 + rem3
  let final_length := 97
  let tying_length := total_rem - final_length
  tying_length = 8 :=
by
  sorry

end ribbon_tying_length_l812_81217


namespace stratified_sampling_example_l812_81208

noncomputable def sample_proportion := 70 / 3500
noncomputable def total_students := 3500 + 1500
noncomputable def sample_size := total_students * sample_proportion

theorem stratified_sampling_example 
  (high_school_students : ℕ := 3500)
  (junior_high_students : ℕ := 1500)
  (sampled_high_school_students : ℕ := 70)
  (proportion_of_sampling : ℝ := sampled_high_school_students / high_school_students)
  (total_number_of_students : ℕ := high_school_students + junior_high_students)
  (calculated_sample_size : ℝ := total_number_of_students * proportion_of_sampling) :
  calculated_sample_size = 100 :=
by
  sorry

end stratified_sampling_example_l812_81208


namespace choir_members_minimum_l812_81271

theorem choir_members_minimum (n : ℕ) : (∃ n, n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 ∧ ∀ m, (m % 8 = 0 ∧ m % 9 = 0 ∧ m % 10 = 0) → n ≤ m) → n = 360 :=
by
  sorry

end choir_members_minimum_l812_81271


namespace find_original_number_l812_81203

variable (x : ℕ)

theorem find_original_number (h : 3 * (2 * x + 9) = 69) : x = 7 :=
by
  sorry

end find_original_number_l812_81203


namespace jack_evening_emails_l812_81220

theorem jack_evening_emails
  (emails_afternoon : ℕ := 3)
  (emails_morning : ℕ := 6)
  (emails_total : ℕ := 10) :
  emails_total - emails_afternoon - emails_morning = 1 :=
by
  sorry

end jack_evening_emails_l812_81220


namespace find_c_l812_81233

-- Define the polynomial f(x)
def f (c : ℚ) (x : ℚ) : ℚ := 2 * c * x^3 + 14 * x^2 - 6 * c * x + 25

-- State the problem in Lean 4
theorem find_c (c : ℚ) : (∀ x : ℚ, f c x = 0 ↔ x = (-5)) → c = 75 / 44 := 
by sorry

end find_c_l812_81233


namespace patrol_streets_in_one_hour_l812_81247

-- Definitions of the given conditions
def streets_patrolled_by_A := 36
def hours_by_A := 4
def rate_A := streets_patrolled_by_A / hours_by_A

def streets_patrolled_by_B := 55
def hours_by_B := 5
def rate_B := streets_patrolled_by_B / hours_by_B

def streets_patrolled_by_C := 42
def hours_by_C := 6
def rate_C := streets_patrolled_by_C / hours_by_C

-- Proof statement 
theorem patrol_streets_in_one_hour : rate_A + rate_B + rate_C = 27 := by
  sorry

end patrol_streets_in_one_hour_l812_81247


namespace average_speed_l812_81282

def s (t : ℝ) : ℝ := 3 + t^2

theorem average_speed {t1 t2 : ℝ} (h1 : t1 = 2) (h2: t2 = 2.1) :
  (s t2 - s t1) / (t2 - t1) = 4.1 :=
by
  sorry

end average_speed_l812_81282


namespace walnut_trees_planted_today_l812_81259

-- Define the number of walnut trees before planting
def walnut_trees_before_planting : ℕ := 22

-- Define the number of walnut trees after planting
def walnut_trees_after_planting : ℕ := 55

-- Define a theorem to prove the number of walnut trees planted
theorem walnut_trees_planted_today : 
  walnut_trees_after_planting - walnut_trees_before_planting = 33 :=
by
  -- The proof will be inserted here.
  sorry

end walnut_trees_planted_today_l812_81259


namespace cylinder_ellipse_eccentricity_l812_81223

noncomputable def eccentricity_of_ellipse (diameter : ℝ) (angle : ℝ) : ℝ :=
  let r := diameter / 2
  let b := r
  let a := r / (Real.cos angle)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem cylinder_ellipse_eccentricity :
  eccentricity_of_ellipse 12 (Real.pi / 6) = 1 / 2 :=
by
  sorry

end cylinder_ellipse_eccentricity_l812_81223


namespace plywood_cut_difference_l812_81244

theorem plywood_cut_difference :
  ∀ (length width : ℕ) (n : ℕ) (perimeter_greatest perimeter_least : ℕ),
    length = 8 ∧ width = 4 ∧ n = 4 ∧
    (∀ l w, (l = (length / 2) ∧ w = width) ∨ (l = length ∧ w = (width / 2)) → (perimeter_greatest = 2 * (l + w))) ∧
    (∀ l w, (l = (length / n) ∧ w = width) ∨ (l = length ∧ w = (width / n)) → (perimeter_least = 2 * (l + w))) →
    length = 8 ∧ width = 4 ∧ n = 4 ∧ perimeter_greatest = 18 ∧ perimeter_least = 12 →
    (perimeter_greatest - perimeter_least) = 6 :=
by
  intros length width n perimeter_greatest perimeter_least h1 h2
  sorry

end plywood_cut_difference_l812_81244


namespace smallest_nat_mul_47_last_four_digits_l812_81280

theorem smallest_nat_mul_47_last_four_digits (N : ℕ) :
  (47 * N) % 10000 = 1969 ↔ N = 8127 :=
sorry

end smallest_nat_mul_47_last_four_digits_l812_81280


namespace find_absolute_cd_l812_81234

noncomputable def polynomial_solution (c d : ℤ) (root1 root2 root3 : ℤ) : Prop :=
  c ≠ 0 ∧ d ≠ 0 ∧ 
  root1 = root2 ∧
  (root3 ≠ root1 ∨ root3 ≠ root2) ∧
  (root1^3 + root2^2 * root3 + (c * root1^2) + (d * root1) + 16 * c = 0) ∧ 
  (root2^3 + root1^2 * root3 + (c * root2^2) + (d * root2) + 16 * c = 0) ∧
  (root3^3 + root1^2 * root3 + (c * root3^2) + (d * root3) + 16 * c = 0)

theorem find_absolute_cd : ∃ c d root1 root2 root3 : ℤ,
  polynomial_solution c d root1 root2 root3 ∧ (|c * d| = 2560) :=
sorry

end find_absolute_cd_l812_81234


namespace range_of_m_l812_81238

noncomputable def proposition_p (x m : ℝ) := (x - m) ^ 2 > 3 * (x - m)
noncomputable def proposition_q (x : ℝ) := x ^ 2 + 3 * x - 4 < 0

theorem range_of_m (m : ℝ) : 
  (∀ x, proposition_p x m → proposition_q x) → 
  (1 ≤ m ∨ m ≤ -7) :=
sorry

end range_of_m_l812_81238


namespace range_a_l812_81266

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (x - 2))

def domain_A : Set ℝ := { x | x < -1 ∨ x > 2 }

def solution_set_B (a : ℝ) : Set ℝ := { x | x < a ∨ x > a + 1 }

theorem range_a (a : ℝ)
  (h : (domain_A ∪ solution_set_B a) = solution_set_B a) :
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end range_a_l812_81266


namespace members_who_play_both_l812_81230

theorem members_who_play_both (N B T Neither : ℕ) (hN : N = 30) (hB : B = 16) (hT : T = 19) (hNeither : Neither = 2) : 
  B + T - (N - Neither) = 7 :=
by
  sorry

end members_who_play_both_l812_81230


namespace perimeter_ratio_l812_81209

theorem perimeter_ratio (w l : ℕ) (hfold : w = 8) (lfold : l = 6) 
(folded_w : w / 2 = 4) (folded_l : l / 2 = 3) 
(hcut : w / 4 = 1) (lcut : l / 2 = 3) 
(perimeter_small : ℕ) (perimeter_large : ℕ)
(hperim_small : perimeter_small = 2 * (3 + 4)) 
(hperim_large : perimeter_large = 2 * (6 + 4)) :
(perimeter_small : ℕ) / (perimeter_large : ℕ) = 7 / 10 := sorry

end perimeter_ratio_l812_81209


namespace subset_singleton_zero_l812_81267

def X : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}

theorem subset_singleton_zero : {0} ⊆ X :=
by
  sorry

end subset_singleton_zero_l812_81267


namespace division_problem_l812_81275

variables (a b c : ℤ)

theorem division_problem 
  (h1 : a ∣ b * c - 1)
  (h2 : b ∣ c * a - 1)
  (h3 : c ∣ a * b - 1) : 
  abc ∣ ab + bc + ca - 1 := 
sorry

end division_problem_l812_81275


namespace symmetric_point_with_respect_to_x_axis_l812_81216

-- Definition of point M
def point_M : ℝ × ℝ := (3, -4)

-- Define the symmetry condition with respect to the x-axis
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

-- Statement that the symmetric point to point M with respect to the x-axis is (3, 4)
theorem symmetric_point_with_respect_to_x_axis : symmetric_x point_M = (3, 4) :=
by
  -- This is the statement of the theorem; the proof will be added here.
  sorry

end symmetric_point_with_respect_to_x_axis_l812_81216


namespace intersection_points_of_parabolas_l812_81265

/-- Let P1 be the equation of the first parabola: y = 3x^2 - 8x + 2 -/
def P1 (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 2

/-- Let P2 be the equation of the second parabola: y = 6x^2 + 4x + 2 -/
def P2 (x : ℝ) : ℝ := 6 * x^2 + 4 * x + 2

/-- Prove that the intersection points of P1 and P2 are (-4, 82) and (0, 2) -/
theorem intersection_points_of_parabolas : 
  {p : ℝ × ℝ | ∃ x, p = (x, P1 x) ∧ P1 x = P2 x} = 
    {(-4, 82), (0, 2)} :=
sorry

end intersection_points_of_parabolas_l812_81265


namespace ratio_w_y_l812_81254

theorem ratio_w_y (w x y z : ℝ) 
  (h1 : w / x = 5 / 4) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 4) 
  (h4 : w + x + y + z = 60) : 
  w / y = 10 / 3 :=
sorry

end ratio_w_y_l812_81254


namespace expression_for_an_l812_81289

noncomputable def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  2 + (n - 1) * d

theorem expression_for_an (d : ℕ) (n : ℕ) 
  (h1 : d > 0)
  (h2 : (arithmetic_sequence d 1) = 2)
  (h3 : (arithmetic_sequence d 1) < (arithmetic_sequence d 2))
  (h4 : (arithmetic_sequence d 2)^2 = 2 * (arithmetic_sequence d 4)) :
  arithmetic_sequence d n = 2 * n := sorry

end expression_for_an_l812_81289


namespace negation_of_exists_prop_l812_81210

variable (n : ℕ)

theorem negation_of_exists_prop :
  (¬ (∃ n : ℕ, n^2 > 2^n)) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by
  sorry

end negation_of_exists_prop_l812_81210


namespace min_value_of_f_l812_81293

def f (x y : ℝ) : ℝ := x^3 + y^3 + x^2 * y + x * y^2 - 3 * (x^2 + y^2 + x * y) + 3 * (x + y)

theorem min_value_of_f : ∀ x y : ℝ, x ≥ 1/2 → y ≥ 1/2 → f x y ≥ 1
    := by
      intros x y hx hy
      -- Rest of the proof would go here
      sorry

end min_value_of_f_l812_81293


namespace value_of_expression_l812_81243

theorem value_of_expression (x : ℝ) (h : x + 3 = 10) : 5 * x + 15 = 50 := by
  sorry

end value_of_expression_l812_81243


namespace shaded_regions_area_sum_l812_81296

theorem shaded_regions_area_sum (side_len : ℚ) (radius : ℚ) (a b c : ℤ) :
  side_len = 16 → radius = side_len / 2 →
  a = (64 / 3) ∧ b = 32 ∧ c = 3 →
  (∃ x : ℤ, x = a + b + c ∧ x = 99) :=
by
  intros hside_len hradius h_constituents
  sorry

end shaded_regions_area_sum_l812_81296


namespace chess_tournament_max_N_l812_81245

theorem chess_tournament_max_N :
  ∃ (N : ℕ), N = 120 ∧
  ∀ (S T : Finset ℕ), S.card = 15 ∧ T.card = 15 ∧
  (∀ s ∈ S, ∀ t ∈ T, (s, t) ∈ (S.product T)) ∧
  (∀ s, ∃! t, (s, t) ∈ (S.product T)) → 
  ∃ (ways_one_game : ℕ), ways_one_game = N ∧ ways_one_game = 120 :=
by
  sorry

end chess_tournament_max_N_l812_81245


namespace max_area_of_fencing_l812_81206

theorem max_area_of_fencing (P : ℕ) (hP : P = 150) 
  (x y : ℕ) (h1 : x + y = P / 2) : (x * y) ≤ 1406 :=
sorry

end max_area_of_fencing_l812_81206


namespace find_lisa_speed_l812_81214

theorem find_lisa_speed (Distance : ℕ) (Time : ℕ) (h1 : Distance = 256) (h2 : Time = 8) : Distance / Time = 32 := 
by {
  sorry
}

end find_lisa_speed_l812_81214


namespace power_expression_l812_81211

variable {a b : ℝ}

theorem power_expression : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := 
by 
  sorry

end power_expression_l812_81211


namespace number_of_integer_solutions_l812_81215

theorem number_of_integer_solutions (x : ℤ) :
  (∃ n : ℤ, n^2 = x^4 + 8*x^3 + 18*x^2 + 8*x + 36) ↔ x = -1 :=
sorry

end number_of_integer_solutions_l812_81215


namespace number_of_pencils_l812_81268

-- Definitions based on the conditions
def ratio_pens_pencils (P L : ℕ) : Prop := P * 6 = 5 * L
def pencils_more_than_pens (P L : ℕ) : Prop := L = P + 4

-- Statement to prove the number of pencils
theorem number_of_pencils : ∃ L : ℕ, (∃ P : ℕ, ratio_pens_pencils P L ∧ pencils_more_than_pens P L) ∧ L = 24 :=
by
  sorry

end number_of_pencils_l812_81268


namespace tan_alpha_plus_pi_over_4_l812_81250

theorem tan_alpha_plus_pi_over_4 
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  Real.tan (α + Real.pi / 4) = 3 / 22 :=
sorry

end tan_alpha_plus_pi_over_4_l812_81250


namespace wrapping_paper_l812_81235

theorem wrapping_paper (total_used_per_roll : ℚ) (number_of_presents : ℕ) (fraction_used : ℚ) (fraction_left : ℚ) 
  (h1 : total_used_per_roll = 2 / 5) 
  (h2 : number_of_presents = 5) 
  (h3 : fraction_used = total_used_per_roll / number_of_presents) 
  (h4 : fraction_left = 1 - total_used_per_roll) : 
  fraction_used = 2 / 25 ∧ fraction_left = 3 / 5 := 
by 
  sorry

end wrapping_paper_l812_81235


namespace range_of_a_l812_81294

variable (a : ℝ)
def f (x : ℝ) : ℝ := a * x^2 - 2 * a * x - 4

theorem range_of_a :
  (∀ x : ℝ, f a x < 0) → (-4 < a ∧ a ≤ 0) :=
by
  sorry

end range_of_a_l812_81294


namespace smallest_PR_minus_QR_l812_81239

theorem smallest_PR_minus_QR :
  ∃ (PQ QR PR : ℤ), 
    PQ + QR + PR = 2023 ∧ PQ ≤ QR ∧ QR < PR ∧ PR - QR = 13 :=
by
  sorry

end smallest_PR_minus_QR_l812_81239


namespace lowest_price_is_six_l812_81252

def single_package_cost : ℝ := 7
def eight_oz_package_cost : ℝ := 4
def four_oz_package_original_cost : ℝ := 2
def discount_rate : ℝ := 0.5

theorem lowest_price_is_six
  (cost_single : single_package_cost = 7)
  (cost_eight : eight_oz_package_cost = 4)
  (cost_four : four_oz_package_original_cost = 2)
  (discount : discount_rate = 0.5) :
  min single_package_cost (eight_oz_package_cost + 2 * (four_oz_package_original_cost * discount_rate)) = 6 := by
  sorry

end lowest_price_is_six_l812_81252


namespace minimum_force_to_submerge_cube_l812_81231

-- Definitions and given conditions
def volume_cube : ℝ := 10e-6 -- 10 cm^3 in m^3
def density_cube : ℝ := 700 -- in kg/m^3
def density_water : ℝ := 1000 -- in kg/m^3
def gravity : ℝ := 10 -- in m/s^2

-- Prove the minimum force required to submerge the cube completely
theorem minimum_force_to_submerge_cube : 
  (density_water * volume_cube * gravity - density_cube * volume_cube * gravity) = 0.03 :=
by
  sorry

end minimum_force_to_submerge_cube_l812_81231


namespace sufficient_condition_for_inequality_l812_81226

theorem sufficient_condition_for_inequality (x : ℝ) : (1 - 1/x > 0) → (x > 1) :=
by
  sorry

end sufficient_condition_for_inequality_l812_81226


namespace not_unique_equilateral_by_one_angle_and_opposite_side_l812_81228

-- Definitions related to triangles
structure Triangle :=
  (a b c : ℝ) -- sides
  (alpha beta gamma : ℝ) -- angles

-- Definition of triangle types
def isIsosceles (t : Triangle) : Prop :=
  (t.a = t.b ∨ t.b = t.c ∨ t.a = t.c)

def isRight (t : Triangle) : Prop :=
  (t.alpha = 90 ∨ t.beta = 90 ∨ t.gamma = 90)

def isEquilateral (t : Triangle) : Prop :=
  (t.a = t.b ∧ t.b = t.c ∧ t.alpha = 60 ∧ t.beta = 60 ∧ t.gamma = 60)

def isScalene (t : Triangle) : Prop :=
  (t.a ≠ t.b ∧ t.b ≠ t.c ∧ t.a ≠ t.c)

-- Proof that having one angle and the side opposite it does not determine an equilateral triangle.
theorem not_unique_equilateral_by_one_angle_and_opposite_side :
  ¬ ∀ (t1 t2 : Triangle), (isEquilateral t1 ∧ isEquilateral t2 →
    t1.alpha = t2.alpha ∧ t1.a = t2.a →
    t1 = t2) := sorry

end not_unique_equilateral_by_one_angle_and_opposite_side_l812_81228


namespace unique_zero_of_function_l812_81256

theorem unique_zero_of_function (a : ℝ) :
  (∃! x : ℝ, e^(abs x) + 2 * a - 1 = 0) ↔ a = 0 := 
by 
  sorry

end unique_zero_of_function_l812_81256


namespace num_cages_l812_81283

-- Define the conditions as given
def parrots_per_cage : ℕ := 8
def parakeets_per_cage : ℕ := 2
def total_birds_in_store : ℕ := 40

-- Prove that the number of bird cages is 4
theorem num_cages (x : ℕ) (h : 10 * x = total_birds_in_store) : x = 4 :=
sorry

end num_cages_l812_81283


namespace tangent_line_at_point_P_l812_81287

-- Definitions from Conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5
def point_on_circle : Prop := circle_eq 1 2

-- Statement to Prove
theorem tangent_line_at_point_P : 
  point_on_circle → ∃ (m : ℝ) (b : ℝ), (m = -1/2) ∧ (b = 5/2) ∧ (∀ x y : ℝ, y = m * x + b ↔ x + 2 * y - 5 = 0) :=
by
  sorry

end tangent_line_at_point_P_l812_81287


namespace average_hidden_primes_l812_81278

theorem average_hidden_primes
  (visible_card1 visible_card2 visible_card3 : ℕ)
  (hidden_card1 hidden_card2 hidden_card3 : ℕ)
  (h1 : visible_card1 = 68)
  (h2 : visible_card2 = 39)
  (h3 : visible_card3 = 57)
  (prime1 : Nat.Prime hidden_card1)
  (prime2 : Nat.Prime hidden_card2)
  (prime3 : Nat.Prime hidden_card3)
  (common_sum : ℕ)
  (h4 : visible_card1 + hidden_card1 = common_sum)
  (h5 : visible_card2 + hidden_card2 = common_sum)
  (h6 : visible_card3 + hidden_card3 = common_sum) :
  (hidden_card1 + hidden_card2 + hidden_card3) / 3 = 15 + 1/3 :=
sorry

end average_hidden_primes_l812_81278


namespace min_dancers_l812_81272

theorem min_dancers (N : ℕ) (h1 : N % 4 = 0) (h2 : N % 9 = 0) (h3 : N % 10 = 0) (h4 : N > 50) : N = 180 :=
  sorry

end min_dancers_l812_81272


namespace intersection_points_l812_81299

theorem intersection_points (a : ℝ) :
  (∀ x y : ℝ, (x^2 + y^2 = a^2) ↔ (y = x^2 - 2 * a)) ↔ (0 < a ∧ a < 1) :=
sorry

end intersection_points_l812_81299


namespace sum_first_n_terms_l812_81240

variable (a : ℕ → ℕ)

axiom a1_condition : a 1 = 2
axiom diff_condition : ∀ n : ℕ, a (n + 1) - a n = 2^n

-- Define the sum of the first n terms of the sequence
noncomputable def S : ℕ → ℕ
| 0 => 0
| (n + 1) => S n + a (n + 1)

theorem sum_first_n_terms (n : ℕ) : S a n = 2^(n + 1) - 2 :=
by
  sorry

end sum_first_n_terms_l812_81240


namespace total_prep_time_l812_81232

-- Definitions:
def jack_time_to_put_shoes_on : ℕ := 4
def additional_time_per_toddler : ℕ := 3
def number_of_toddlers : ℕ := 2

-- Total time calculation
def total_time : ℕ :=
  let time_per_toddler := jack_time_to_put_shoes_on + additional_time_per_toddler
  let total_toddler_time := time_per_toddler * number_of_toddlers
  total_toddler_time + jack_time_to_put_shoes_on

-- Theorem:
theorem total_prep_time :
  total_time = 18 :=
sorry

end total_prep_time_l812_81232


namespace annual_yield_range_l812_81241

-- Here we set up the conditions as definitions in Lean 4
def last_year_range : ℝ := 10000
def improvement_rate : ℝ := 0.15

-- Theorems that are based on the conditions and need proving
theorem annual_yield_range (last_year_range : ℝ) (improvement_rate : ℝ) : 
  last_year_range * (1 + improvement_rate) = 11500 := 
sorry

end annual_yield_range_l812_81241


namespace proof_a_squared_plus_1_l812_81269

theorem proof_a_squared_plus_1 (a : ℤ) (h1 : 3 < a) (h2 : a < 5) : a^2 + 1 = 17 :=
  by
  sorry

end proof_a_squared_plus_1_l812_81269


namespace range_of_a_l812_81288

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l812_81288


namespace E_runs_is_20_l812_81202

-- Definitions of runs scored by each batsman as multiples of 4
def a := 28
def e := 20
def d := e + 12
def b := d + e
def c := 107 - b
def total_runs := a + b + c + d + e

-- Adding conditions
axiom A_max: a > b ∧ a > c ∧ a > d ∧ a > e
axiom runs_multiple_of_4: ∀ (x : ℕ), x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e → x % 4 = 0
axiom average_runs: total_runs = 180
axiom d_condition: d = e + 12
axiom e_condition: e = a - 8
axiom b_condition: b = d + e
axiom bc_condition: b + c = 107

theorem E_runs_is_20 : e = 20 := by
  sorry

end E_runs_is_20_l812_81202


namespace correct_bushes_needed_l812_81290

def yield_per_bush := 10
def containers_per_zucchini := 3
def zucchinis_needed := 36
def bushes_needed (yield_per_bush containers_per_zucchini zucchinis_needed : ℕ) : ℕ :=
  Nat.ceil ((zucchinis_needed * containers_per_zucchini : ℕ) / yield_per_bush)

theorem correct_bushes_needed : bushes_needed yield_per_bush containers_per_zucchini zucchinis_needed = 11 := 
by
  sorry

end correct_bushes_needed_l812_81290


namespace total_payroll_l812_81295

theorem total_payroll 
  (heavy_operator_pay : ℕ) 
  (laborer_pay : ℕ) 
  (total_people : ℕ) 
  (laborers : ℕ)
  (heavy_operators : ℕ)
  (total_payroll : ℕ)
  (h1: heavy_operator_pay = 140)
  (h2: laborer_pay = 90)
  (h3: total_people = 35)
  (h4: laborers = 19)
  (h5: heavy_operators = total_people - laborers)
  (h6: total_payroll = (heavy_operators * heavy_operator_pay) + (laborers * laborer_pay)) :
  total_payroll = 3950 :=
by sorry

end total_payroll_l812_81295


namespace smallest_n_for_symmetry_property_l812_81236

-- Define the setup for the problem
def has_required_symmetry (n : ℕ) : Prop :=
∀ (S : Finset (Fin n)), S.card = 5 →
∃ (l : Fin n → Fin n), (∀ v ∈ S, l v ≠ v) ∧ (∀ v ∈ S, l v ∉ S)

-- The main lemma we are proving
theorem smallest_n_for_symmetry_property : ∃ n : ℕ, (∀ m < n, ¬ has_required_symmetry m) ∧ has_required_symmetry 14 :=
by
  sorry

end smallest_n_for_symmetry_property_l812_81236


namespace sara_no_ingredients_pies_l812_81257

theorem sara_no_ingredients_pies:
  ∀ (total_pies : ℕ) (berries_pies : ℕ) (cream_pies : ℕ) (nuts_pies : ℕ) (coconut_pies : ℕ),
  total_pies = 60 →
  berries_pies = 1/3 * total_pies →
  cream_pies = 1/2 * total_pies →
  nuts_pies = 3/5 * total_pies →
  coconut_pies = 1/5 * total_pies →
  (total_pies - nuts_pies) = 24 :=
by
  intros total_pies berries_pies cream_pies nuts_pies coconut_pies ht hb hc hn hcoc
  sorry

end sara_no_ingredients_pies_l812_81257


namespace max_full_pikes_l812_81273

theorem max_full_pikes (initial_pikes : ℕ) (pike_full_condition : ℕ → Prop) (remaining_pikes : ℕ) 
  (h_initial : initial_pikes = 30)
  (h_condition : ∀ n, pike_full_condition n → n ≥ 3)
  (h_remaining : remaining_pikes ≥ 1) :
    ∃ max_full : ℕ, max_full ≤ 9 := 
sorry

end max_full_pikes_l812_81273
