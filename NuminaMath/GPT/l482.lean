import Mathlib

namespace smallest_7_digit_number_divisible_by_all_l482_48262

def smallest_7_digit_number : ℕ := 7207200

theorem smallest_7_digit_number_divisible_by_all :
  smallest_7_digit_number >= 1000000 ∧ smallest_7_digit_number < 10000000 ∧
  smallest_7_digit_number % 35 = 0 ∧ 
  smallest_7_digit_number % 112 = 0 ∧ 
  smallest_7_digit_number % 175 = 0 ∧ 
  smallest_7_digit_number % 288 = 0 ∧ 
  smallest_7_digit_number % 429 = 0 ∧ 
  smallest_7_digit_number % 528 = 0 :=
by
  sorry

end smallest_7_digit_number_divisible_by_all_l482_48262


namespace liars_positions_l482_48220

structure Islander :=
  (position : Nat)
  (statement : String)

-- Define our islanders
def A : Islander := { position := 1, statement := "My closest tribesman in this line is 3 meters away from me." }
def D : Islander := { position := 4, statement := "My closest tribesman in this line is 1 meter away from me." }
def E : Islander := { position := 5, statement := "My closest tribesman in this line is 2 meters away from me." }

-- Define the other islanders with dummy statements
def B : Islander := { position := 2, statement := "" }
def C : Islander := { position := 3, statement := "" }
def F : Islander := { position := 6, statement := "" }

-- Define the main theorem
theorem liars_positions (knights_count : Nat) (liars_count : Nat) (is_knight : Islander → Bool)
  (is_lair : Islander → Bool) : 
  ( ∀ x, is_knight x ↔ ¬is_lair x ) → -- Knight and liar are mutually exclusive
  knights_count = 3 → 
  liars_count = 3 →
  is_knight A = false → 
  is_knight D = false → 
  is_knight E = false → 
  is_lair A = true ∧
  is_lair D = true ∧
  is_lair E = true := by
  sorry

end liars_positions_l482_48220


namespace arithmetic_sequence_sum_l482_48223

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h_arith : ∀ k, S (k + 1) - S k = S 1 - S 0)
  (h_S5 : S 5 = 10) (h_S10 : S 10 = 18) : S 15 = 26 :=
by
  -- Rest of the proof goes here
  sorry

end arithmetic_sequence_sum_l482_48223


namespace fractional_product_l482_48210

theorem fractional_product :
  ((3/4) * (4/5) * (5/6) * (6/7) * (7/8)) = 3/8 :=
by
  sorry

end fractional_product_l482_48210


namespace age_ratio_in_years_l482_48233

theorem age_ratio_in_years (p c x : ℕ) 
  (H1 : p - 2 = 3 * (c - 2)) 
  (H2 : p - 4 = 4 * (c - 4)) 
  (H3 : (p + x) / (c + x) = 2) : 
  x = 4 :=
sorry

end age_ratio_in_years_l482_48233


namespace find_vertex_D_l482_48245

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

end find_vertex_D_l482_48245


namespace total_pages_in_storybook_l482_48238

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

end total_pages_in_storybook_l482_48238


namespace prime_power_condition_l482_48208

open Nat

theorem prime_power_condition (u v : ℕ) :
  (∃ p n : ℕ, p.Prime ∧ p^n = (u * v^3) / (u^2 + v^2)) ↔ ∃ k : ℕ, k ≥ 1 ∧ u = 2^k ∧ v = 2^k := by {
  sorry
}

end prime_power_condition_l482_48208


namespace geometric_series_common_ratio_l482_48299

theorem geometric_series_common_ratio (a r : ℝ) (n : ℕ) 
(h1 : a = 7 / 3) 
(h2 : r = 49 / 21)
(h3 : r = 343 / 147):
  r = 7 / 3 :=
by
  sorry

end geometric_series_common_ratio_l482_48299


namespace no_half_dimension_cuboid_l482_48227

theorem no_half_dimension_cuboid
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (a' b' c' : ℝ) (ha' : a' > 0) (hb' : b' > 0) (hc' : c' > 0) :
  ¬ (a' * b' * c' = (1 / 2) * a * b * c ∧ 2 * (a' * b' + b' * c' + c' * a') = a * b + b * c + c * a) :=
by
  sorry

end no_half_dimension_cuboid_l482_48227


namespace digit_difference_l482_48270

theorem digit_difference (X Y : ℕ) (h_digits : 0 ≤ X ∧ X < 10 ∧ 0 ≤ Y ∧ Y < 10) (h_diff :  (10 * X + Y) - (10 * Y + X) = 45) : X - Y = 5 :=
sorry

end digit_difference_l482_48270


namespace weight_of_replaced_person_l482_48216

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

end weight_of_replaced_person_l482_48216


namespace minimum_value_of_z_l482_48272

theorem minimum_value_of_z :
  ∀ (x y : ℝ), ∃ z : ℝ, z = 2*x^2 + 3*y^2 + 8*x - 6*y + 35 ∧ z ≥ 24 := by
  sorry

end minimum_value_of_z_l482_48272


namespace total_time_marco_6_laps_total_time_in_minutes_and_seconds_l482_48209

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

end total_time_marco_6_laps_total_time_in_minutes_and_seconds_l482_48209


namespace product_of_square_roots_l482_48231
-- Importing the necessary Lean library

-- Declare the mathematical problem in Lean 4
theorem product_of_square_roots (x : ℝ) (hx : 0 ≤ x) :
  Real.sqrt (40 * x) * Real.sqrt (5 * x) * Real.sqrt (18 * x) = 60 * x * Real.sqrt (3 * x) :=
by
  sorry

end product_of_square_roots_l482_48231


namespace sum_of_digits_nine_ab_l482_48279

noncomputable def sum_digits_base_10 (n : ℕ) : ℕ :=
-- Function to compute the sum of digits of a number in base 10
sorry

def a : ℕ := 6 * ((10^1500 - 1) / 9)

def b : ℕ := 3 * ((10^1500 - 1) / 9)

def nine_ab : ℕ := 9 * a * b

theorem sum_of_digits_nine_ab :
  sum_digits_base_10 nine_ab = 13501 :=
sorry

end sum_of_digits_nine_ab_l482_48279


namespace percentage_speaking_both_langs_l482_48290

def diplomats_total : ℕ := 100
def diplomats_french : ℕ := 22
def diplomats_not_russian : ℕ := 32
def diplomats_neither : ℕ := 20

theorem percentage_speaking_both_langs
  (h1 : 20% diplomats_total = diplomats_neither)
  (h2 : diplomats_total - diplomats_not_russian = 68)
  (h3 : diplomats_total ≠ 0) :
  (22 + 68 - 80) / diplomats_total * 100 = 10 :=
by
  sorry

end percentage_speaking_both_langs_l482_48290


namespace necessary_but_not_sufficient_condition_l482_48225

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  ((a > 2) ∧ (b > 2) → (a + b > 4)) ∧ ¬((a + b > 4) → (a > 2) ∧ (b > 2)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l482_48225


namespace math_time_more_than_science_l482_48212

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

end math_time_more_than_science_l482_48212


namespace TrigPowerEqualsOne_l482_48294

theorem TrigPowerEqualsOne : ((Real.cos (160 * Real.pi / 180) + Real.sin (160 * Real.pi / 180) * Complex.I)^36 = 1) :=
by
  sorry

end TrigPowerEqualsOne_l482_48294


namespace milk_for_18_cookies_l482_48217

def milk_needed_to_bake_cookies (cookies : ℕ) (milk_per_24_cookies : ℚ) (quarts_to_pints : ℚ) : ℚ :=
  (milk_per_24_cookies * quarts_to_pints) * (cookies / 24)

theorem milk_for_18_cookies :
  milk_needed_to_bake_cookies 18 4.5 2 = 6.75 :=
by
  sorry

end milk_for_18_cookies_l482_48217


namespace n_digit_numbers_modulo_3_l482_48213

def a (i : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then if i = 0 then 1 else 0 else 2 * a i (n - 1) + a ((i + 1) % 3) (n - 1) + a ((i + 2) % 3) (n - 1)

theorem n_digit_numbers_modulo_3 (n : ℕ) (h : 0 < n) : 
  (a 0 n) = (4^n + 2) / 3 :=
sorry

end n_digit_numbers_modulo_3_l482_48213


namespace gaussian_guardians_total_points_l482_48237

theorem gaussian_guardians_total_points :
  let daniel := 7
  let curtis := 8
  let sid := 2
  let emily := 11
  let kalyn := 6
  let hyojeong := 12
  let ty := 1
  let winston := 7
  daniel + curtis + sid + emily + kalyn + hyojeong + ty + winston = 54 := by
  sorry

end gaussian_guardians_total_points_l482_48237


namespace evaluate_64_pow_7_over_6_l482_48232

theorem evaluate_64_pow_7_over_6 : (64 : ℝ)^(7 / 6) = 128 := by
  have h : (64 : ℝ) = 2^6 := by norm_num
  rw [h]
  norm_num
  sorry

end evaluate_64_pow_7_over_6_l482_48232


namespace postcards_remainder_l482_48239

theorem postcards_remainder :
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  total % 15 = 3 :=
by
  let amelia := 45
  let ben := 55
  let charles := 23
  let total := amelia + ben + charles
  show total % 15 = 3
  sorry

end postcards_remainder_l482_48239


namespace perpendicular_line_sum_l482_48241

theorem perpendicular_line_sum (a b c : ℝ) (h1 : a + 4 * c - 2 = 0) (h2 : 2 - 5 * c + b = 0) 
  (perpendicular : (a / -4) * (2 / 5) = -1) : a + b + c = -4 := 
sorry

end perpendicular_line_sum_l482_48241


namespace problem_statement_l482_48258

variable (x : ℝ)

theorem problem_statement (h : x^2 - x - 1 = 0) : 1995 + 2 * x - x^3 = 1994 := by
  sorry

end problem_statement_l482_48258


namespace julia_played_with_kids_on_tuesday_l482_48246

theorem julia_played_with_kids_on_tuesday (total: ℕ) (monday: ℕ) (tuesday: ℕ) 
  (h1: total = 18) (h2: monday = 4) : 
  tuesday = (total - monday) :=
by
  sorry

end julia_played_with_kids_on_tuesday_l482_48246


namespace cubic_sum_div_pqr_eq_three_l482_48254

theorem cubic_sum_div_pqr_eq_three (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p + q + r = 6) :
  (p^3 + q^3 + r^3) / (p * q * r) = 3 := 
by
  sorry

end cubic_sum_div_pqr_eq_three_l482_48254


namespace gcd_360_128_is_8_l482_48219

def gcd_360_128 : ℕ :=
  gcd 360 128

theorem gcd_360_128_is_8 : gcd_360_128 = 8 :=
  by
    -- Proof goes here (use sorry for now)
    sorry

end gcd_360_128_is_8_l482_48219


namespace hyperbola_equation_sum_l482_48202

theorem hyperbola_equation_sum (h k a c b : ℝ) (h_h : h = 1) (h_k : k = 1) (h_a : a = 3) (h_c : c = 9) (h_c2 : c^2 = a^2 + b^2) :
    h + k + a + b = 5 + 6 * Real.sqrt 2 :=
by
  sorry

end hyperbola_equation_sum_l482_48202


namespace wallpaper_job_completion_l482_48243

theorem wallpaper_job_completion (x : ℝ) (y : ℝ) 
  (h1 : ∀ a b : ℝ, (a = 1.5) → (7/x + (7-a)/(x-3) = 1)) 
  (h2 : y = x - 3) 
  (h3 : x - y = 3) : 
  (x = 14) ∧ (y = 11) :=
sorry

end wallpaper_job_completion_l482_48243


namespace jensen_miles_city_l482_48278

theorem jensen_miles_city (total_gallons : ℕ) (highway_miles : ℕ) (highway_mpg : ℕ)
  (city_mpg : ℕ) (highway_gallons : ℕ) (city_gallons : ℕ) (city_miles : ℕ) :
  total_gallons = 9 ∧ highway_miles = 210 ∧ highway_mpg = 35 ∧ city_mpg = 18 ∧
  highway_gallons = highway_miles / highway_mpg ∧
  city_gallons = total_gallons - highway_gallons ∧
  city_miles = city_gallons * city_mpg → city_miles = 54 :=
by
  sorry

end jensen_miles_city_l482_48278


namespace maintenance_check_days_l482_48273

theorem maintenance_check_days (x : ℝ) (hx : x + 0.20 * x = 60) : x = 50 :=
by
  -- this is where the proof would go
  sorry

end maintenance_check_days_l482_48273


namespace initial_rows_l482_48222

theorem initial_rows (r T : ℕ) (h1 : T = 42 * r) (h2 : T = 28 * (r + 12)) : r = 24 :=
by
  sorry

end initial_rows_l482_48222


namespace distance_from_P_to_AD_l482_48207

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

end distance_from_P_to_AD_l482_48207


namespace keiko_speed_l482_48235

theorem keiko_speed (a b : ℝ) (s : ℝ) (h1 : (2 * a + 2 * π * (b + 8)) / s = (2 * a + 2 * π * b) / s + 48) : s = π / 3 :=
by {
  sorry -- proof is not required
}

end keiko_speed_l482_48235


namespace percent_of_liquidX_in_solutionB_l482_48291

theorem percent_of_liquidX_in_solutionB (P : ℝ) (h₁ : 0.8 / 100 = 0.008) 
(h₂ : 1.5 / 100 = 0.015) 
(h₃ : 300 * 0.008 = 2.4) 
(h₄ : 1000 * 0.015 = 15) 
(h₅ : 15 - 2.4 = 12.6) 
(h₆ : 12.6 / 700 = P) : 
P * 100 = 1.8 :=
by sorry

end percent_of_liquidX_in_solutionB_l482_48291


namespace remainder_when_squared_l482_48266

theorem remainder_when_squared (n : ℤ) (h : n % 5 = 3) : (n^2) % 5 = 4 := by
  sorry

end remainder_when_squared_l482_48266


namespace sum_of_squares_of_consecutive_integers_l482_48250

theorem sum_of_squares_of_consecutive_integers (b : ℕ) (h : (b-1) * b * (b+1) = 12 * ((b-1) + b + (b+1))) : 
  (b - 1) * (b - 1) + b * b + (b + 1) * (b + 1) = 110 := 
by sorry

end sum_of_squares_of_consecutive_integers_l482_48250


namespace conditions_necessary_sufficient_l482_48259

variables (p q r s : Prop)

theorem conditions_necessary_sufficient :
  ((p → r) ∧ (¬ (r → p)) ∧ (q → r) ∧ (s → r) ∧ (q → s)) →
  ((s ↔ q) ∧ ((p → q) ∧ ¬ (q → p)) ∧ ((¬ p → ¬ s) ∧ ¬ (¬ s → ¬ p))) := by
  sorry

end conditions_necessary_sufficient_l482_48259


namespace carl_additional_gift_bags_l482_48260

theorem carl_additional_gift_bags (definite_visitors additional_visitors extravagant_bags average_bags total_bags_needed : ℕ) :
  definite_visitors = 50 →
  additional_visitors = 40 →
  extravagant_bags = 10 →
  average_bags = 20 →
  total_bags_needed = 90 →
  (total_bags_needed - (extravagant_bags + average_bags)) = 60 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end carl_additional_gift_bags_l482_48260


namespace right_triangle_medians_right_triangle_l482_48269

theorem right_triangle_medians_right_triangle (a b c s_a s_b s_c : ℝ)
  (hyp_a_lt_b : a < b) (hyp_b_lt_c : b < c)
  (h_c_hypotenuse : c = Real.sqrt (a^2 + b^2))
  (h_sa : s_a^2 = b^2 + (a / 2)^2)
  (h_sb : s_b^2 = a^2 + (b / 2)^2)
  (h_sc : s_c^2 = (a^2 + b^2) / 4) :
  b = a * Real.sqrt 2 :=
by
  sorry

end right_triangle_medians_right_triangle_l482_48269


namespace percentage_of_number_l482_48203

theorem percentage_of_number (N : ℕ) (P : ℕ) (h1 : N = 120) (h2 : (3 * N) / 5 = 72) (h3 : (P * 72) / 100 = 36) : P = 50 :=
sorry

end percentage_of_number_l482_48203


namespace correct_transformation_l482_48229

theorem correct_transformation (a b c : ℝ) (h : (b / (a^2 + 1)) > (c / (a^2 + 1))) : b > c :=
by {
  -- Placeholder proof
  sorry
}

end correct_transformation_l482_48229


namespace height_ratio_l482_48283

theorem height_ratio (C : ℝ) (h_o : ℝ) (V_s : ℝ) (h_s : ℝ) (r : ℝ) :
  C = 18 * π →
  h_o = 20 →
  V_s = 270 * π →
  C = 2 * π * r →
  V_s = 1 / 3 * π * r^2 * h_s →
  h_s / h_o = 1 / 2 :=
by
  sorry

end height_ratio_l482_48283


namespace volume_rectangular_solid_l482_48244

theorem volume_rectangular_solid
  (a b c : ℝ) 
  (h1 : a * b = 12)
  (h2 : b * c = 8)
  (h3 : a * c = 6) :
  a * b * c = 24 :=
sorry

end volume_rectangular_solid_l482_48244


namespace octal_to_decimal_l482_48221

theorem octal_to_decimal (n_octal : ℕ) (h : n_octal = 123) : 
  let d0 := 3 * 8^0
  let d1 := 2 * 8^1
  let d2 := 1 * 8^2
  n_octal = 64 + 16 + 3 :=
by
  sorry

end octal_to_decimal_l482_48221


namespace sum_first_n_terms_arithmetic_sequence_l482_48268

/-- Define the arithmetic sequence with common difference d and a given term a₄. -/
def arithmetic_sequence (n : ℕ) (a₁ d : ℤ) : ℤ :=
  a₁ + (n - 1) * d

/-- Define the sum of the first n terms of an arithmetic sequence. -/
def sum_of_arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * ((2 * a₁ + (n - 1) * d) / 2)

theorem sum_first_n_terms_arithmetic_sequence :
  ∀ n : ℕ, 
  ∀ a₁ : ℤ, 
  (∀ d, d = 2 → (∀ a₁, (a₁ + 3 * d = 8) → sum_of_arithmetic_sequence a₁ d n = (n : ℤ) * ((n : ℤ) + 1))) :=
by
  intros n a₁ d hd h₁
  sorry

end sum_first_n_terms_arithmetic_sequence_l482_48268


namespace nesbitt_inequality_l482_48234

theorem nesbitt_inequality {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ (3 / 2) ∧ (a = b ∧ b = c → a / (b + c) + b / (c + a) + c / (a + b) = 3 / 2) :=
sorry

end nesbitt_inequality_l482_48234


namespace fare_midpoint_to_b_l482_48240

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

end fare_midpoint_to_b_l482_48240


namespace total_population_l482_48265

variable (b g t : ℕ)

-- Conditions: 
axiom boys_to_girls (h1 : b = 4 * g) : Prop
axiom girls_to_teachers (h2 : g = 8 * t) : Prop

theorem total_population (h1 : b = 4 * g) (h2 : g = 8 * t) : b + g + t = 41 * b / 32 :=
sorry

end total_population_l482_48265


namespace rigged_coin_probability_l482_48295

theorem rigged_coin_probability (p : ℝ) (h1 : p < 1 / 2) (h2 : 20 * (p ^ 3) * ((1 - p) ^ 3) = 1 / 12) :
  p = (1 - Real.sqrt 0.86) / 2 :=
by
  sorry

end rigged_coin_probability_l482_48295


namespace reflect_over_x_axis_l482_48287

def coords (P : ℝ × ℝ) : ℝ × ℝ :=
  (P.1, -P.2)

theorem reflect_over_x_axis :
  coords (-6, -9) = (-6, 9) :=
by
  sorry

end reflect_over_x_axis_l482_48287


namespace cindy_envelopes_l482_48204

theorem cindy_envelopes (h₁ : ℕ := 4) (h₂ : ℕ := 7) (h₃ : ℕ := 5) (h₄ : ℕ := 10) (h₅ : ℕ := 3) (initial : ℕ := 137) :
  initial - (h₁ + h₂ + h₃ + h₄ + h₅) = 108 :=
by
  sorry

end cindy_envelopes_l482_48204


namespace sum_of_3digit_numbers_remainder_2_l482_48206

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

end sum_of_3digit_numbers_remainder_2_l482_48206


namespace min_t_value_l482_48256

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x - 1

theorem min_t_value : 
  ∀ (x y : ℝ), x ∈ Set.Icc (-3 : ℝ) (2 : ℝ) → y ∈ Set.Icc (-3 : ℝ) (2 : ℝ)
  → |f (x) - f (y)| ≤ 20 :=
by
  sorry

end min_t_value_l482_48256


namespace range_of_m_l482_48297

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, abs (x - m) < 1 ↔ (1/3 < x ∧ x < 1/2)) ↔ (-1/2 ≤ m ∧ m ≤ 4/3) :=
by
  sorry

end range_of_m_l482_48297


namespace count_integers_expression_negative_l482_48282

theorem count_integers_expression_negative :
  ∃ n : ℕ, n = 4 ∧ 
  ∀ x : ℤ, x^4 - 60 * x^2 + 144 < 0 → n = 4 := by
  -- Placeholder for the proof
  sorry

end count_integers_expression_negative_l482_48282


namespace percentage_increase_second_year_is_20_l482_48263

noncomputable def find_percentage_increase_second_year : ℕ :=
  let P₀ := 1000
  let P₁ := P₀ + (10 * P₀) / 100
  let Pf := 1320
  let P := (Pf - P₁) * 100 / P₁
  P

theorem percentage_increase_second_year_is_20 :
  find_percentage_increase_second_year = 20 :=
by
  sorry

end percentage_increase_second_year_is_20_l482_48263


namespace pollen_allergy_expected_count_l482_48285

theorem pollen_allergy_expected_count : 
  ∀ (sample_size : ℕ) (pollen_allergy_ratio : ℚ), 
  pollen_allergy_ratio = 1/4 ∧ sample_size = 400 → sample_size * pollen_allergy_ratio = 100 :=
  by 
    intros
    sorry

end pollen_allergy_expected_count_l482_48285


namespace trigonometric_identity_proof_l482_48255

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 2 * Real.tan (Real.pi / 5)) :
  (Real.cos (α - 3 * Real.pi / 10)) / (Real.sin (α - Real.pi / 5)) = 3 :=
by
  sorry

end trigonometric_identity_proof_l482_48255


namespace larry_wins_probability_l482_48211

noncomputable def probability_larry_wins (p_L : ℚ) (p_J : ℚ) : ℚ :=
  let q_L := 1 - p_L
  let q_J := 1 - p_J
  let r := q_L * q_J
  p_L / (1 - r)

theorem larry_wins_probability
  (p_L : ℚ) (p_J : ℚ) (h1 : p_L = 3 / 5) (h2 : p_J = 1 / 3) :
  probability_larry_wins p_L p_J = 9 / 11 :=
by 
  sorry

end larry_wins_probability_l482_48211


namespace fruit_seller_gain_l482_48292

-- Define necessary variables
variables {C S : ℝ} (G : ℝ)

-- Given conditions
def selling_price_def (C : ℝ) : ℝ := 1.25 * C
def total_cost_price (C : ℝ) : ℝ := 150 * C
def total_selling_price (C : ℝ) : ℝ := 150 * (selling_price_def C)
def gain (C : ℝ) : ℝ := total_selling_price C - total_cost_price C

-- Statement to prove: number of apples' selling price gained by the fruit-seller is 30
theorem fruit_seller_gain : G = 30 ↔ gain C = G * (selling_price_def C) :=
by
  sorry

end fruit_seller_gain_l482_48292


namespace evaluate_expression_l482_48293

theorem evaluate_expression :
  ∀ (a b c : ℚ),
  c = b + 1 →
  b = a + 5 →
  a = 3 →
  (a + 2 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (a + 3) * (b + 1) * (c + 9) / ((a + 2) * (b - 3) * (c + 7)) = 2.43 := 
by
  intros a b c hc hb ha h1 h2 h3
  sorry

end evaluate_expression_l482_48293


namespace trapezoid_shaded_fraction_l482_48253

theorem trapezoid_shaded_fraction (total_strips : ℕ) (shaded_strips : ℕ)
  (h_total : total_strips = 7) (h_shaded : shaded_strips = 4) :
  (shaded_strips : ℚ) / (total_strips : ℚ) = 4 / 7 := 
by
  sorry

end trapezoid_shaded_fraction_l482_48253


namespace parallelogram_height_l482_48247

theorem parallelogram_height
  (area : ℝ)
  (base : ℝ)
  (h_area : area = 375)
  (h_base : base = 25) :
  (area / base) = 15 :=
by
  sorry

end parallelogram_height_l482_48247


namespace find_distance_PQ_of_polar_coords_l482_48249

theorem find_distance_PQ_of_polar_coords (α β : ℝ) (h : β - α = 2 * Real.pi / 3) :
  let P := (5, α)
  let Q := (12, β)
  dist P Q = Real.sqrt 229 :=
by
  sorry

end find_distance_PQ_of_polar_coords_l482_48249


namespace Alyssa_total_spent_l482_48277

/-- Definition of fruit costs -/
def cost_grapes : ℝ := 12.08
def cost_cherries : ℝ := 9.85
def cost_mangoes : ℝ := 7.50
def cost_pineapple : ℝ := 4.25
def cost_starfruit : ℝ := 3.98

/-- Definition of tax and discount -/
def tax_rate : ℝ := 0.10
def discount : ℝ := 3.00

/-- Calculation of the total cost Alyssa spent after applying tax and discount -/
def total_spent : ℝ := 
  let total_cost_before_tax := cost_grapes + cost_cherries + cost_mangoes + cost_pineapple + cost_starfruit
  let tax := tax_rate * total_cost_before_tax
  let total_cost_with_tax := total_cost_before_tax + tax
  total_cost_with_tax - discount

/-- Statement that needs to be proven -/
theorem Alyssa_total_spent : total_spent = 38.43 := by 
  sorry

end Alyssa_total_spent_l482_48277


namespace b2009_value_l482_48284

noncomputable def b (n : ℕ) : ℝ := sorry

axiom b_recursion (n : ℕ) (hn : 2 ≤ n) : b n = b (n - 1) * b (n + 1)

axiom b1_value : b 1 = 2 + Real.sqrt 3
axiom b1776_value : b 1776 = 10 + Real.sqrt 3

theorem b2009_value : b 2009 = -4 + 8 * Real.sqrt 3 := 
by sorry

end b2009_value_l482_48284


namespace conditional_probability_l482_48215

variable (pA pB pAB : ℝ)
variable (h1 : pA = 0.2)
variable (h2 : pB = 0.18)
variable (h3 : pAB = 0.12)

theorem conditional_probability : (pAB / pB = 2 / 3) :=
by
  -- sorry is used to skip the proof
  sorry

end conditional_probability_l482_48215


namespace sufficient_but_not_necessary_condition_l482_48248

theorem sufficient_but_not_necessary_condition (b : ℝ) :
  (∀ x : ℝ, b * x^2 - b * x + 1 > 0) ↔ (b = 0 ∨ (0 < b ∧ b < 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l482_48248


namespace prime_odd_sum_l482_48288

theorem prime_odd_sum (a b : ℕ) (h1 : Prime a) (h2 : Odd b) (h3 : a^2 + b = 2001) : a + b = 1999 :=
sorry

end prime_odd_sum_l482_48288


namespace interval_between_doses_l482_48218

noncomputable def dose_mg : ℕ := 2 * 375

noncomputable def total_mg_per_day : ℕ := 3000

noncomputable def hours_in_day : ℕ := 24

noncomputable def doses_per_day := total_mg_per_day / dose_mg

noncomputable def hours_between_doses := hours_in_day / doses_per_day

theorem interval_between_doses : hours_between_doses = 6 :=
by
  sorry

end interval_between_doses_l482_48218


namespace black_haired_girls_count_l482_48200

theorem black_haired_girls_count (initial_total_girls : ℕ) (initial_blonde_girls : ℕ) (added_blonde_girls : ℕ) (final_blonde_girls total_girls : ℕ) 
    (h1 : initial_total_girls = 80) 
    (h2 : initial_blonde_girls = 30) 
    (h3 : added_blonde_girls = 10) 
    (h4 : final_blonde_girls = initial_blonde_girls + added_blonde_girls) 
    (h5 : total_girls = initial_total_girls) : 
    total_girls - final_blonde_girls = 40 :=
by
  sorry

end black_haired_girls_count_l482_48200


namespace tennis_racket_weight_l482_48298

theorem tennis_racket_weight 
  (r b : ℝ)
  (h1 : 10 * r = 8 * b)
  (h2 : 4 * b = 120) :
  r = 24 :=
by
  sorry

end tennis_racket_weight_l482_48298


namespace ron_spends_on_chocolate_bars_l482_48271

/-- Ron is hosting a camp for 15 scouts where each scout needs 2 s'mores.
    Each chocolate bar costs $1.50 and can be broken into 3 sections to make 3 s'mores.
    A discount of 15% applies if 10 or more chocolate bars are purchased.
    Calculate the total amount Ron will spend on chocolate bars after applying the discount if applicable. -/
theorem ron_spends_on_chocolate_bars :
  let cost_per_bar := 1.5
  let s'mores_per_bar := 3
  let scouts := 15
  let s'mores_per_scout := 2
  let total_s'mores := scouts * s'mores_per_scout
  let bars_needed := total_s'mores / s'mores_per_bar
  let discount := 0.15
  let total_cost := bars_needed * cost_per_bar
  let discount_amount := if bars_needed >= 10 then discount * total_cost else 0
  let final_cost := total_cost - discount_amount
  final_cost = 12.75 := by sorry

end ron_spends_on_chocolate_bars_l482_48271


namespace even_sum_probability_l482_48264

theorem even_sum_probability :
  let wheel1 := (2/6, 3/6, 1/6)   -- (probability of even, odd, zero) for the first wheel
  let wheel2 := (2/4, 2/4)        -- (probability of even, odd) for the second wheel
  let both_even := (1/3) * (1/2)  -- probability of both numbers being even
  let both_odd := (1/2) * (1/2)   -- probability of both numbers being odd
  let zero_and_even := (1/6) * (1/2)  -- probability of one number being zero and the other even
  let total_probability := both_even + both_odd + zero_and_even
  total_probability = 1/2 := by sorry

end even_sum_probability_l482_48264


namespace students_taking_neither_l482_48261

variable (total_students math_students physics_students both_students : ℕ)
variable (h1 : total_students = 80)
variable (h2 : math_students = 50)
variable (h3 : physics_students = 40)
variable (h4 : both_students = 25)

theorem students_taking_neither (h1 : total_students = 80)
    (h2 : math_students = 50)
    (h3 : physics_students = 40)
    (h4 : both_students = 25) :
    total_students - (math_students - both_students + physics_students - both_students + both_students) = 15 :=
by
    sorry

end students_taking_neither_l482_48261


namespace blue_paint_amount_l482_48214

theorem blue_paint_amount
  (blue_white_ratio : ℚ := 4 / 5)
  (white_paint : ℚ := 15)
  (blue_paint : ℚ) :
  blue_paint = 12 :=
by
  sorry

end blue_paint_amount_l482_48214


namespace prime_product_2002_l482_48252

theorem prime_product_2002 {a b c d : ℕ} (ha_prime : Prime a) (hb_prime : Prime b) (hc_prime : Prime c) (hd_prime : Prime d)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h1 : a + c = d)
  (h2 : a * (a + b + c + d) = c * (d - b))
  (h3 : 1 + b * c + d = b * d) :
  a * b * c * d = 2002 := 
by 
  sorry

end prime_product_2002_l482_48252


namespace remainder_of_9876543210_div_101_l482_48224

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 :=
  sorry

end remainder_of_9876543210_div_101_l482_48224


namespace exists_n_for_dvd_ka_pow_n_add_n_l482_48205

theorem exists_n_for_dvd_ka_pow_n_add_n 
  (a k : ℕ) (a_pos : 0 < a) (k_pos : 0 < k) (d : ℕ) (d_pos : 0 < d) :
  ∃ n : ℕ, 0 < n ∧ d ∣ k * (a ^ n) + n :=
by
  sorry

end exists_n_for_dvd_ka_pow_n_add_n_l482_48205


namespace total_vegetables_l482_48296

-- Define the initial conditions
def potatoes : Nat := 560
def cucumbers : Nat := potatoes - 132
def tomatoes : Nat := 3 * cucumbers
def peppers : Nat := tomatoes / 2
def carrots : Nat := cucumbers + tomatoes

-- State the theorem to prove the total number of vegetables
theorem total_vegetables :
  560 + (560 - 132) + (3 * (560 - 132)) + ((3 * (560 - 132)) / 2) + ((560 - 132) + (3 * (560 - 132))) = 4626 := by
  sorry

end total_vegetables_l482_48296


namespace yellow_balls_in_bag_l482_48236

open Classical

theorem yellow_balls_in_bag (Y : ℕ) (hY1 : (Y/(Y+2): ℝ) * ((Y-1)/(Y+1): ℝ) = 0.5) : Y = 5 := by
  sorry

end yellow_balls_in_bag_l482_48236


namespace product_xyz_42_l482_48276

theorem product_xyz_42 (x y z : ℝ) 
  (h1 : (x - 2)^2 + (y - 3)^2 + (z - 4)^2 = 9)
  (h2 : x + y + z = 12) : x * y * z = 42 :=
by
  sorry

end product_xyz_42_l482_48276


namespace expression_equivalence_l482_48274

theorem expression_equivalence:
  let a := 10006 - 8008
  let b := 10000 - 8002
  a = b :=
by {
  sorry
}

end expression_equivalence_l482_48274


namespace number_of_flute_players_l482_48275

theorem number_of_flute_players (F T B D C H : ℕ)
  (hT : T = 3 * F)
  (hB : B = T - 8)
  (hD : D = B + 11)
  (hC : C = 2 * F)
  (hH : H = B + 3)
  (h_total : F + T + B + D + C + H = 65) :
  F = 6 :=
by
  sorry

end number_of_flute_players_l482_48275


namespace second_number_is_255_l482_48289

theorem second_number_is_255 (x : ℝ) (n : ℝ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 90) 
  (h2 : (128 + n + 511 + 1023 + x) / 5 = 423) : 
  n = 255 :=
sorry

end second_number_is_255_l482_48289


namespace bathing_suits_total_l482_48201

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969
def total_bathing_suits : ℕ := 19766

theorem bathing_suits_total :
  men_bathing_suits + women_bathing_suits = total_bathing_suits := by
  sorry

end bathing_suits_total_l482_48201


namespace equivalence_of_statements_l482_48228

variable (P Q : Prop)

theorem equivalence_of_statements (h : P → Q) :
  (P → Q) ∧ (¬Q → ¬P) ∧ (¬P ∨ Q) := by
  sorry

end equivalence_of_statements_l482_48228


namespace sum_term_ratio_equals_four_l482_48226

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

end sum_term_ratio_equals_four_l482_48226


namespace snail_distance_round_100_l482_48257

def snail_distance (n : ℕ) : ℕ :=
  if n = 0 then 100 else (100 * (n + 2)) / (n + 1)

theorem snail_distance_round_100 : snail_distance 100 = 5050 :=
  sorry

end snail_distance_round_100_l482_48257


namespace multiplication_problems_l482_48280

theorem multiplication_problems :
  (30 * 30 = 900) ∧
  (30 * 40 = 1200) ∧
  (40 * 70 = 2800) ∧
  (50 * 70 = 3500) ∧
  (60 * 70 = 4200) ∧
  (4 * 90 = 360) :=
by sorry

end multiplication_problems_l482_48280


namespace sum_xyz_zero_l482_48267

theorem sum_xyz_zero 
  (x y z : ℝ)
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : y = 6 * z) : 
  x + y + z = 0 := by
  sorry

end sum_xyz_zero_l482_48267


namespace biscuit_dimensions_l482_48242

theorem biscuit_dimensions (sheet_length : ℝ) (sheet_width : ℝ) (num_biscuits : ℕ) 
  (h₁ : sheet_length = 12) (h₂ : sheet_width = 12) (h₃ : num_biscuits = 16) :
  ∃ biscuit_length : ℝ, biscuit_length = 3 :=
by
  sorry

end biscuit_dimensions_l482_48242


namespace lacy_percentage_correct_l482_48251

variable (x : ℕ)

-- Definitions from the conditions
def total_problems := 8 * x
def missed_problems := 2 * x
def answered_problems := total_problems - missed_problems
def bonus_problems := x
def bonus_points := 2 * bonus_problems
def regular_points := answered_problems - bonus_problems
def total_points_scored := bonus_points + regular_points
def total_available_points := 8 * x + 2 * x

theorem lacy_percentage_correct :
  total_points_scored / total_available_points * 100 = 90 := by
  -- Proof steps would go here, but are not required per instructions.
  sorry

end lacy_percentage_correct_l482_48251


namespace running_speed_l482_48281

theorem running_speed (R : ℝ) (walking_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) (half_distance : ℝ) (walking_time : ℝ) (running_time : ℝ)
  (h1 : walking_speed = 4)
  (h2 : total_distance = 16)
  (h3 : total_time = 3)
  (h4 : half_distance = total_distance / 2)
  (h5 : walking_time = half_distance / walking_speed)
  (h6 : running_time = half_distance / R)
  (h7 : walking_time + running_time = total_time) :
  R = 8 := 
sorry

end running_speed_l482_48281


namespace sum_of_cubes_l482_48286

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
by
  sorry

end sum_of_cubes_l482_48286


namespace eq_perp_bisector_BC_area_triangle_ABC_l482_48230

section Triangle_ABC

open Real

-- Define the vertices A, B, and C
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (-1, 1)
def C : ℝ × ℝ := (3, 3)

-- Define the equation of the perpendicular bisector
theorem eq_perp_bisector_BC : ∀ x y : ℝ, 2 * x + y - 4 = 0 :=
sorry

-- Define the area of the triangle ABC
noncomputable def triangle_area : ℝ :=
1 / 2 * (abs ((-1 * 3 + 3 * (-2) + 3 * 4) - (3 * 4 + 1 * (-2) + 3*(-1))))

theorem area_triangle_ABC : triangle_area = 7 :=
sorry

end Triangle_ABC

end eq_perp_bisector_BC_area_triangle_ABC_l482_48230
