import Mathlib

namespace NUMINAMATH_GPT_xyz_value_l2201_220183

theorem xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 30) 
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) : x * y * z = 7 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l2201_220183


namespace NUMINAMATH_GPT_boat_distance_against_stream_l2201_220170

variable (v_s : ℝ)
variable (effective_speed_stream : ℝ := 15)
variable (speed_still_water : ℝ := 10)
variable (distance_along_stream : ℝ := 15)

theorem boat_distance_against_stream : 
  distance_along_stream / effective_speed_stream = 1 ∧ effective_speed_stream = speed_still_water + v_s →
  10 - v_s = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boat_distance_against_stream_l2201_220170


namespace NUMINAMATH_GPT_remainder_is_three_l2201_220113

def eleven_div_four_has_remainder_three (A : ℕ) : Prop :=
  11 = 4 * 2 + A

theorem remainder_is_three : eleven_div_four_has_remainder_three 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_is_three_l2201_220113


namespace NUMINAMATH_GPT_digit_possibilities_757_l2201_220121

theorem digit_possibilities_757
  (N : ℕ)
  (h : N < 10) :
  (∃ d₀ d₁ d₂ : ℕ, (d₀ = 2 ∨ d₀ = 5 ∨ d₀ = 8) ∧
  (d₁ = 2 ∨ d₁ = 5 ∨ d₁ = 8) ∧
  (d₂ = 2 ∨ d₂ = 5 ∨ d₂ = 8) ∧
  (d₀ ≠ d₁) ∧
  (d₀ ≠ d₂) ∧
  (d₁ ≠ d₂)) :=
by
  sorry

end NUMINAMATH_GPT_digit_possibilities_757_l2201_220121


namespace NUMINAMATH_GPT_f1_min_max_f2_min_max_l2201_220199

-- Define the first function and assert its max and min values
def f1 (x : ℝ) : ℝ := x^3 + 2 * x

theorem f1_min_max : ∀ x ∈ Set.Icc (-1 : ℝ) 1,
  (∃ x_min x_max, x_min = -1 ∧ x_max = 1 ∧ f1 x_min = -3 ∧ f1 x_max = 3) := by
  sorry

-- Define the second function and assert its max and min values
def f2 (x : ℝ) : ℝ := (x - 1) * (x - 2)^2

theorem f2_min_max : ∀ x ∈ Set.Icc (0 : ℝ) 3,
  (∃ x_min x_max, x_min = 0 ∧ x_max = 3 ∧ (f2 x_min = -4) ∧ f2 x_max = 2) := by
  sorry

end NUMINAMATH_GPT_f1_min_max_f2_min_max_l2201_220199


namespace NUMINAMATH_GPT_sum_of_a_and_b_l2201_220111

theorem sum_of_a_and_b (a b : ℤ) (h1 : a + 2 * b = 8) (h2 : 2 * a + b = 4) : a + b = 4 := by
  sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l2201_220111


namespace NUMINAMATH_GPT_find_n_l2201_220102

theorem find_n (n : ℕ) (m : ℕ) (h_pos_n : n > 0) (h_pos_m : m > 0) (h_div : (2^n - 1) ∣ (m^2 + 81)) : 
  ∃ k : ℕ, n = 2^k := 
sorry

end NUMINAMATH_GPT_find_n_l2201_220102


namespace NUMINAMATH_GPT_jill_present_age_l2201_220136

-- Define the main proof problem
theorem jill_present_age (H J : ℕ) (h1 : H + J = 33) (h2 : H - 6 = 2 * (J - 6)) : J = 13 :=
by
  sorry

end NUMINAMATH_GPT_jill_present_age_l2201_220136


namespace NUMINAMATH_GPT_students_still_in_school_l2201_220192

def total_students := 5000
def students_to_beach := total_students / 2
def remaining_after_beach := total_students - students_to_beach
def students_to_art_museum := remaining_after_beach / 3
def remaining_after_art_museum := remaining_after_beach - students_to_art_museum
def students_to_science_fair := remaining_after_art_museum / 4
def remaining_after_science_fair := remaining_after_art_museum - students_to_science_fair
def students_to_music_workshop := 200
def remaining_students := remaining_after_science_fair - students_to_music_workshop

theorem students_still_in_school : remaining_students = 1051 := by
  sorry

end NUMINAMATH_GPT_students_still_in_school_l2201_220192


namespace NUMINAMATH_GPT_blue_pigment_percentage_l2201_220123

-- Define weights and pigments in the problem
variables (S G : ℝ)
-- Conditions
def sky_blue_paint := 0.9 * S = 4.5
def total_weight := S + G = 10
def sky_blue_blue_pigment := 0.1
def green_blue_pigment := 0.7

-- Prove the percentage of blue pigment in brown paint is 40%
theorem blue_pigment_percentage :
  sky_blue_paint S →
  total_weight S G →
  (0.1 * (4.5 / 0.9) + 0.7 * (10 - (4.5 / 0.9))) / 10 * 100 = 40 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_blue_pigment_percentage_l2201_220123


namespace NUMINAMATH_GPT_chromium_percentage_l2201_220159

noncomputable def chromium_percentage_in_new_alloy 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) : ℝ :=
    (((chromium_percentage_first * weight_first / 100) + (chromium_percentage_second * weight_second / 100)) 
    / (weight_first + weight_second)) * 100

theorem chromium_percentage 
    (chromium_percentage_first: ℝ) 
    (weight_first: ℝ) 
    (chromium_percentage_second: ℝ) 
    (weight_second: ℝ) 
    (h1 : chromium_percentage_first = 10) 
    (h2 : weight_first = 15) 
    (h3 : chromium_percentage_second = 8) 
    (h4 : weight_second = 35) :
    chromium_percentage_in_new_alloy chromium_percentage_first weight_first chromium_percentage_second weight_second = 8.6 :=
by 
  rw [h1, h2, h3, h4]
  simp [chromium_percentage_in_new_alloy]
  norm_num


end NUMINAMATH_GPT_chromium_percentage_l2201_220159


namespace NUMINAMATH_GPT_total_people_waiting_in_line_l2201_220186

-- Conditions
def people_fitting_in_ferris_wheel : ℕ := 56
def people_not_getting_on : ℕ := 36

-- Definition: Number of people waiting in line
def number_of_people_waiting_in_line : ℕ := people_fitting_in_ferris_wheel + people_not_getting_on

-- Theorem to prove
theorem total_people_waiting_in_line : number_of_people_waiting_in_line = 92 := by
  -- This is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_total_people_waiting_in_line_l2201_220186


namespace NUMINAMATH_GPT_inequality_sufficient_condition_l2201_220168

theorem inequality_sufficient_condition (x : ℝ) (h : 1 < x ∧ x < 2) : 
  (x+1)/(x-1) > 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_sufficient_condition_l2201_220168


namespace NUMINAMATH_GPT_ratio_of_fifth_terms_l2201_220100

theorem ratio_of_fifth_terms (a_n b_n : ℕ → ℕ) (S T : ℕ → ℕ)
  (hs : ∀ n, S n = n * (a_n 1 + a_n n) / 2)
  (ht : ∀ n, T n = n * (b_n 1 + b_n n) / 2)
  (h : ∀ n, S n / T n = (7 * n + 2) / (n + 3)) :
  a_n 5 / b_n 5 = 65 / 12 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_fifth_terms_l2201_220100


namespace NUMINAMATH_GPT_largest_perfect_square_factor_of_1764_l2201_220125

theorem largest_perfect_square_factor_of_1764 : ∃ m, m * m = 1764 ∧ ∀ n, n * n ∣ 1764 → n * n ≤ 1764 :=
by
  sorry

end NUMINAMATH_GPT_largest_perfect_square_factor_of_1764_l2201_220125


namespace NUMINAMATH_GPT_total_handshakes_l2201_220161

theorem total_handshakes (gremlins imps unfriendly_gremlins : ℕ) 
    (handshakes_among_friendly : ℕ) (handshakes_friendly_with_unfriendly : ℕ) 
    (handshakes_between_imps_and_gremlins : ℕ) 
    (h_friendly : gremlins = 30) (h_imps : imps = 20) 
    (h_unfriendly : unfriendly_gremlins = 10) 
    (h_handshakes_among_friendly : handshakes_among_friendly = 190) 
    (h_handshakes_friendly_with_unfriendly : handshakes_friendly_with_unfriendly = 200)
    (h_handshakes_between_imps_and_gremlins : handshakes_between_imps_and_gremlins = 600) : 
    handshakes_among_friendly + handshakes_friendly_with_unfriendly + handshakes_between_imps_and_gremlins = 990 := 
by 
    sorry

end NUMINAMATH_GPT_total_handshakes_l2201_220161


namespace NUMINAMATH_GPT_fruit_display_l2201_220133

theorem fruit_display (bananas : ℕ) (Oranges : ℕ) (Apples : ℕ) (hBananas : bananas = 5)
  (hOranges : Oranges = 2 * bananas) (hApples : Apples = 2 * Oranges) :
  bananas + Oranges + Apples = 35 :=
by sorry

end NUMINAMATH_GPT_fruit_display_l2201_220133


namespace NUMINAMATH_GPT_least_positive_integer_condition_l2201_220178

theorem least_positive_integer_condition :
  ∃ (n : ℕ), n > 0 ∧ (n % 2 = 1) ∧ (n % 5 = 4) ∧ (n % 7 = 6) ∧ n = 69 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_integer_condition_l2201_220178


namespace NUMINAMATH_GPT_marble_count_calculation_l2201_220148

theorem marble_count_calculation (y b g : ℕ) (x : ℕ)
  (h1 : y = 2 * x)
  (h2 : b = 3 * x)
  (h3 : g = 4 * x)
  (h4 : g = 32) : y + b + g = 72 :=
by
  sorry

end NUMINAMATH_GPT_marble_count_calculation_l2201_220148


namespace NUMINAMATH_GPT_stable_scores_l2201_220128

theorem stable_scores (S_A S_B S_C S_D : ℝ) (hA : S_A = 2.2) (hB : S_B = 6.6) (hC : S_C = 7.4) (hD : S_D = 10.8) : 
  S_A ≤ S_B ∧ S_A ≤ S_C ∧ S_A ≤ S_D :=
by
  sorry

end NUMINAMATH_GPT_stable_scores_l2201_220128


namespace NUMINAMATH_GPT_odd_function_value_l2201_220138

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^3 - Real.sin x + b + 2

theorem odd_function_value (a b : ℝ) (h1 : ∀ x, f x b = -f (-x) b) (h2 : a - 4 + 2 * a - 2 = 0) : f a b + f (2 * -a) b = 0 := by
  sorry

end NUMINAMATH_GPT_odd_function_value_l2201_220138


namespace NUMINAMATH_GPT_min_value_of_f_range_of_a_l2201_220143

def f (x : ℝ) : ℝ := 2 * |x - 2| - x + 5

theorem min_value_of_f : ∃ (m : ℝ), m = 3 ∧ ∀ x : ℝ, f x ≥ m :=
by
  use 3
  sorry

theorem range_of_a (a : ℝ) : (|a + 2| ≥ 3 ↔ a ≤ -5 ∨ a ≥ 1) :=
sorry

end NUMINAMATH_GPT_min_value_of_f_range_of_a_l2201_220143


namespace NUMINAMATH_GPT_common_ratio_value_l2201_220180

variable (a : ℕ → ℝ) -- defining the geometric sequence as a function ℕ → ℝ
variable (q : ℝ) -- defining the common ratio

-- conditions from the problem
def geo_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

axiom h1 : geo_seq a q
axiom h2 : a 2020 = 8 * a 2017

-- main statement to be proved
theorem common_ratio_value : q = 2 :=
sorry

end NUMINAMATH_GPT_common_ratio_value_l2201_220180


namespace NUMINAMATH_GPT_least_number_to_add_l2201_220156

theorem least_number_to_add (n d : ℕ) (h₁ : n = 1054) (h₂ : d = 23) : ∃ x, (n + x) % d = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_least_number_to_add_l2201_220156


namespace NUMINAMATH_GPT_graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l2201_220110

theorem graph_of_3x2_minus_12y2_is_pair_of_straight_lines :
  ∀ (x y : ℝ), 3 * x^2 - 12 * y^2 = 0 ↔ (x = 2 * y ∨ x = -2 * y) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_3x2_minus_12y2_is_pair_of_straight_lines_l2201_220110


namespace NUMINAMATH_GPT_probability_male_is_2_5_l2201_220151

variable (num_male_students num_female_students : ℕ)

def total_students (num_male_students num_female_students : ℕ) : ℕ :=
  num_male_students + num_female_students

def probability_of_male (num_male_students num_female_students : ℕ) : ℚ :=
  num_male_students / (total_students num_male_students num_female_students : ℚ)

theorem probability_male_is_2_5 :
  probability_of_male 2 3 = 2 / 5 := by
    sorry

end NUMINAMATH_GPT_probability_male_is_2_5_l2201_220151


namespace NUMINAMATH_GPT_gcd_91_72_l2201_220176

/-- Prove that the greatest common divisor of 91 and 72 is 1. -/
theorem gcd_91_72 : Nat.gcd 91 72 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_91_72_l2201_220176


namespace NUMINAMATH_GPT_exists_triangle_with_side_lengths_l2201_220103

theorem exists_triangle_with_side_lengths (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_ineq : 6 * (a * b + b * c + c * a) > 5 * (a^2 + b^2 + c^2)) : 
  (a + b > c ∧ b + c > a ∧ c + a > b) :=
sorry

end NUMINAMATH_GPT_exists_triangle_with_side_lengths_l2201_220103


namespace NUMINAMATH_GPT_meet_without_contact_probability_l2201_220126

noncomputable def prob_meet_without_contact : ℝ :=
  let total_area := 1
  let outside_area := (1 / 8) * 2
  total_area - outside_area

theorem meet_without_contact_probability :
  prob_meet_without_contact = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_meet_without_contact_probability_l2201_220126


namespace NUMINAMATH_GPT_abc_eq_ab_bc_ca_l2201_220119

variable {u v w A B C : ℝ}
variable (Huvw : u * v * w = 1)
variable (HA : A = u * v + u + 1)
variable (HB : B = v * w + v + 1)
variable (HC : C = w * u + w + 1)

theorem abc_eq_ab_bc_ca 
  (Huvw : u * v * w = 1)
  (HA : A = u * v + u + 1)
  (HB : B = v * w + v + 1)
  (HC : C = w * u + w + 1) : 
  A * B * C = A * B + B * C + C * A := 
by
  sorry

end NUMINAMATH_GPT_abc_eq_ab_bc_ca_l2201_220119


namespace NUMINAMATH_GPT_even_function_of_shift_sine_l2201_220158

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := (x - 6)^2 * Real.sin (ω * x)

theorem even_function_of_shift_sine :
  ∃ ω : ℝ, (∀ x : ℝ, f x ω = f (-x) ω) → ω = π / 4 :=
by
  sorry

end NUMINAMATH_GPT_even_function_of_shift_sine_l2201_220158


namespace NUMINAMATH_GPT_hyperbola_eccentricity_sqrt2_l2201_220147

theorem hyperbola_eccentricity_sqrt2
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c = Real.sqrt (a^2 + b^2))
  (h : (c + a)^2 + (b^2 / a)^2 = 2 * c * (c + a)) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_sqrt2_l2201_220147


namespace NUMINAMATH_GPT_initial_men_count_l2201_220139

theorem initial_men_count (x : ℕ) 
  (h1 : ∀ t : ℕ, t = 25 * x) 
  (h2 : ∀ t : ℕ, t = 12 * 75) : 
  x = 36 := 
by
  sorry

end NUMINAMATH_GPT_initial_men_count_l2201_220139


namespace NUMINAMATH_GPT_min_weighings_to_find_heaviest_l2201_220114

-- Given conditions
variable (n : ℕ) (hn : n > 2)
variables (coins : Fin n) -- Representing coins with distinct masses
variables (scales : Fin n) -- Representing n scales where one is faulty

-- Theorem statement: Minimum number of weighings to find the heaviest coin
theorem min_weighings_to_find_heaviest : ∃ m, m = 2 * n - 1 := 
by
  existsi (2 * n - 1)
  rfl

end NUMINAMATH_GPT_min_weighings_to_find_heaviest_l2201_220114


namespace NUMINAMATH_GPT_number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l2201_220115

/- Definitions for each number's expression using five eights -/
def number1 : Int := (8 / 8) ^ (8 / 8) * (8 / 8)
def number2 : Int := 8 / 8 + 8 / 8
def number3 : Int := (8 + 8 + 8) / 8
def number4 : Int := 8 / 8 + 8 / 8 + 8 / 8 + 8 / 8
def number5 : Int := (8 * 8 - 8) / 8 + 8 / 8

/- Theorem statements to be proven -/
theorem number1_is_1 : number1 = 1 := by
  sorry

theorem number2_is_2 : number2 = 2 := by
  sorry

theorem number3_is_3 : number3 = 3 := by
  sorry

theorem number4_is_4 : number4 = 4 := by
  sorry

theorem number5_is_5 : number5 = 5 := by
  sorry

end NUMINAMATH_GPT_number1_is_1_number2_is_2_number3_is_3_number4_is_4_number5_is_5_l2201_220115


namespace NUMINAMATH_GPT_nonnegative_poly_sum_of_squares_l2201_220169

open Polynomial

theorem nonnegative_poly_sum_of_squares (P : Polynomial ℝ) 
    (hP : ∀ x : ℝ, 0 ≤ P.eval x) 
    : ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 := 
by
  sorry

end NUMINAMATH_GPT_nonnegative_poly_sum_of_squares_l2201_220169


namespace NUMINAMATH_GPT_min_surface_area_l2201_220198

/-- Defining the conditions and the problem statement -/
def solid (volume : ℝ) (face1 face2 : ℝ) : Prop := 
  ∃ x y z, x * y * z = volume ∧ (x * y = face1 ∨ y * z = face1 ∨ z * x = face1)
                      ∧ (x * y = face2 ∨ y * z = face2 ∨ z * x = face2)

def juan_solids (face1 face2 face3 face4 face5 face6 : ℝ) : Prop :=
  solid 128 4 32 ∧ solid 128 64 16 ∧ solid 128 8 32

theorem min_surface_area {volume : ℝ} {face1 face2 face3 face4 face5 face6 : ℝ} 
  (h : juan_solids 4 32 64 16 8 32) : 
  ∃ area : ℝ, area = 688 :=
sorry

end NUMINAMATH_GPT_min_surface_area_l2201_220198


namespace NUMINAMATH_GPT_hyperbola_standard_equation_l2201_220152

theorem hyperbola_standard_equation
  (passes_through : ∀ {x y : ℝ}, (x, y) = (1, 1) → 2 * x + y = 0 ∨ 2 * x - y = 0)
  (asymptote1 : ∀ {x y : ℝ}, 2 * x + y = 0 → y = -2 * x)
  (asymptote2 : ∀ {x y : ℝ}, 2 * x - y = 0 → y = 2 * x) :
  ∃ a b : ℝ, a = 4 / 3 ∧ b = 1 / 3 ∧ ∀ x y : ℝ, (x, y) = (1, 1) → (x^2 / a - y^2 / b = 1) := 
sorry

end NUMINAMATH_GPT_hyperbola_standard_equation_l2201_220152


namespace NUMINAMATH_GPT_value_of_gg_neg1_l2201_220189

def g (x : ℝ) : ℝ := 4 * x^2 + 3

theorem value_of_gg_neg1 : g (g (-1)) = 199 := by
  sorry

end NUMINAMATH_GPT_value_of_gg_neg1_l2201_220189


namespace NUMINAMATH_GPT_common_altitude_l2201_220162

theorem common_altitude (A1 A2 b1 b2 h : ℝ)
    (hA1 : A1 = 800)
    (hA2 : A2 = 1200)
    (hb1 : b1 = 40)
    (hb2 : b2 = 60)
    (h1 : A1 = 1 / 2 * b1 * h)
    (h2 : A2 = 1 / 2 * b2 * h) :
    h = 40 := 
sorry

end NUMINAMATH_GPT_common_altitude_l2201_220162


namespace NUMINAMATH_GPT_makeup_exam_probability_l2201_220197

theorem makeup_exam_probability (total_students : ℕ) (students_in_makeup_exam : ℕ)
  (h1 : total_students = 42) (h2 : students_in_makeup_exam = 3) :
  (students_in_makeup_exam : ℚ) / total_students = 1 / 14 := by
  sorry

end NUMINAMATH_GPT_makeup_exam_probability_l2201_220197


namespace NUMINAMATH_GPT_number_of_solutions_eq_l2201_220174

open Nat

theorem number_of_solutions_eq (n : ℕ) : 
  ∃ N, (∀ (x : ℝ), 1 ≤ x ∧ x ≤ n → x^2 - ⌊x^2⌋ = (x - ⌊x⌋)^2) → N = n^2 - n + 1 :=
by sorry

end NUMINAMATH_GPT_number_of_solutions_eq_l2201_220174


namespace NUMINAMATH_GPT_fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l2201_220164

def fractional_eq_solution_1 (x : ℝ) : Prop :=
  x + 5 / x = -6

theorem fractional_eq_solutions_1 : fractional_eq_solution_1 (-1) ∧ fractional_eq_solution_1 (-5) := sorry

def fractional_eq_solution_2 (x : ℝ) : Prop :=
  x - 3 / x = 4

theorem fractional_eq_reciprocal_sum
  (m n : ℝ) (h₀ : fractional_eq_solution_2 m) (h₁ : fractional_eq_solution_2 n) :
  m * n = -3 → m + n = 4 → (1 / m + 1 / n = -4 / 3) := sorry

def fractional_eq_solution_3 (x : ℝ) (a : ℝ) : Prop :=
  x + (a^2 + 2 * a) / (x + 1) = 2 * a + 1

theorem fractional_eq_solution_diff_square (a : ℝ) (h₀ : a ≠ 0)
  (x1 x2 : ℝ) (hx1 : fractional_eq_solution_3 x1 a) (hx2 : fractional_eq_solution_3 x2 a) :
  x1 + 1 = a → x2 + 1 = a + 2 → (x1 - x2) ^ 2 = 4 := sorry

end NUMINAMATH_GPT_fractional_eq_solutions_1_fractional_eq_reciprocal_sum_fractional_eq_solution_diff_square_l2201_220164


namespace NUMINAMATH_GPT_quadratic_roots_unique_l2201_220145

theorem quadratic_roots_unique (p q : ℚ) :
  (∀ x : ℚ, x^2 + p * x + q = 0 ↔ (x = 2 * p ∨ x = p + q)) →
  p = 2 / 3 ∧ q = -8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_unique_l2201_220145


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l2201_220181

theorem sufficient_but_not_necessary (a b : ℝ) : 
  (a > b + 1) → (a > b) ∧ (¬(a > b) → ¬(a > b + 1)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l2201_220181


namespace NUMINAMATH_GPT_frac_add_eq_seven_halves_l2201_220166

theorem frac_add_eq_seven_halves {x y : ℝ} (h : x / y = 5 / 2) : (x + y) / y = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_frac_add_eq_seven_halves_l2201_220166


namespace NUMINAMATH_GPT_not_possible_1006_2012_gons_l2201_220108

theorem not_possible_1006_2012_gons :
  ∀ (n : ℕ), (∀ (k : ℕ), k ≤ 2011 → 2 * n ≤ k) → n ≠ 1006 :=
by
  intro n h
  -- Here goes the skipped proof part
  sorry

end NUMINAMATH_GPT_not_possible_1006_2012_gons_l2201_220108


namespace NUMINAMATH_GPT_circle_radius_l2201_220160

theorem circle_radius (r₂ : ℝ) : 
  (∃ r₁ : ℝ, r₁ = 5 ∧ (∀ d : ℝ, d = 7 → (d = r₁ + r₂ ∨ d = abs (r₁ - r₂)))) → (r₂ = 2 ∨ r₂ = 12) :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2201_220160


namespace NUMINAMATH_GPT_mows_in_summer_l2201_220191

theorem mows_in_summer (S : ℕ) (h1 : 8 - S = 3) : S = 5 :=
sorry

end NUMINAMATH_GPT_mows_in_summer_l2201_220191


namespace NUMINAMATH_GPT_foldable_polygons_count_l2201_220167

def isValidFolding (base_positions : Finset Nat) (additional_position : Nat) : Prop :=
  ∃ (valid_positions : Finset Nat), valid_positions = {4, 5, 6, 7, 8, 9} ∧ additional_position ∈ valid_positions

theorem foldable_polygons_count : 
  ∃ (valid_additional_positions : Finset Nat), valid_additional_positions = {4, 5, 6, 7, 8, 9} ∧ valid_additional_positions.card = 6 := 
by
  sorry

end NUMINAMATH_GPT_foldable_polygons_count_l2201_220167


namespace NUMINAMATH_GPT_pairs_of_positive_integers_l2201_220135

theorem pairs_of_positive_integers (x y : ℕ) (hx : x > 0) (hy : y > 0) :
    (∃ (m : ℕ), m ≥ 2 ∧ (x = m^3 + 2*m^2 - m - 1 ∧ y = m^3 + m^2 - 2*m - 1 ∨ 
                        x = m^3 + m^2 - 2*m - 1 ∧ y = m^3 + 2*m^2 - m - 1)) ∨
    (x = 1 ∧ y = 1) ↔ 
    (∃ n : ℝ, n^3 = 7*x^2 - 13*x*y + 7*y^2) ∧ (Int.natAbs (x - y) - 1 = n) :=
by
  sorry

end NUMINAMATH_GPT_pairs_of_positive_integers_l2201_220135


namespace NUMINAMATH_GPT_value_of_b_l2201_220187

theorem value_of_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := 
by
  sorry

end NUMINAMATH_GPT_value_of_b_l2201_220187


namespace NUMINAMATH_GPT_two_digit_numbers_div_by_7_with_remainder_1_l2201_220130

theorem two_digit_numbers_div_by_7_with_remainder_1 :
  {n : ℕ | ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 10 * a + b ∧ (10 * a + b) % 7 = 1 ∧ (10 * b + a) % 7 = 1} 
  = {22, 29, 92, 99} := 
by
  sorry

end NUMINAMATH_GPT_two_digit_numbers_div_by_7_with_remainder_1_l2201_220130


namespace NUMINAMATH_GPT_largest_n_for_ap_interior_angles_l2201_220153

theorem largest_n_for_ap_interior_angles (n : ℕ) (d : ℤ) (a : ℤ) :
  (∀ i ∈ Finset.range n, a + i * d < 180) → 720 = d * (n - 1) * n → n ≤ 27 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_for_ap_interior_angles_l2201_220153


namespace NUMINAMATH_GPT_smallest_positive_divisor_l2201_220146

theorem smallest_positive_divisor
  (a b x₀ y₀ : ℤ)
  (h₀ : a ≠ 0 ∨ b ≠ 0)
  (h₁ : ∀ x y, a * x₀ + b * y₀ ≤ 0 ∨ a * x + b * y ≥ a * x₀ + b * y₀)
  (h₂ : 0 < a * x₀ + b * y₀):
  ∀ x y : ℤ, a * x₀ + b * y₀ ∣ a * x + b * y := 
sorry

end NUMINAMATH_GPT_smallest_positive_divisor_l2201_220146


namespace NUMINAMATH_GPT_z_gets_amount_per_unit_l2201_220184

-- Define the known conditions
variables (x y z : ℝ)
variables (x_share : ℝ)
variables (y_share : ℝ)
variables (z_share : ℝ)
variables (total : ℝ)

-- Assume the conditions given in the problem
axiom h1 : y_share = 54
axiom h2 : total = 234
axiom h3 : (y / x) = 0.45
axiom h4 : total = x_share + y_share + z_share

-- Prove the target statement
theorem z_gets_amount_per_unit : ((z_share / x_share) = 0.50) :=
by
  sorry

end NUMINAMATH_GPT_z_gets_amount_per_unit_l2201_220184


namespace NUMINAMATH_GPT_max_nested_fraction_value_l2201_220194

-- Define the problem conditions
def numbers := (List.range 100).map (λ n => n + 1)

-- Define the nested fraction function
noncomputable def nested_fraction (l : List ℕ) : ℚ :=
  l.foldr (λ x acc => x / acc) 1

-- Prove that the maximum value of the nested fraction from 1 to 100 is 100! / 4
theorem max_nested_fraction_value :
  nested_fraction numbers = (Nat.factorial 100) / 4 :=
sorry

end NUMINAMATH_GPT_max_nested_fraction_value_l2201_220194


namespace NUMINAMATH_GPT_total_number_of_cats_l2201_220104

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_cats_l2201_220104


namespace NUMINAMATH_GPT_distance_behind_C_l2201_220157

-- Conditions based on the problem
def distance_race : ℕ := 1000
def distance_B_when_A_finishes : ℕ := 50
def distance_C_when_B_finishes : ℕ := 100

-- Derived condition based on given problem details
def distance_run_by_B_when_A_finishes : ℕ := distance_race - distance_B_when_A_finishes
def distance_run_by_C_when_B_finishes : ℕ := distance_race - distance_C_when_B_finishes

-- Ratios
def ratio_B_to_A : ℚ := distance_run_by_B_when_A_finishes / distance_race
def ratio_C_to_B : ℚ := distance_run_by_C_when_B_finishes / distance_race

-- Combined ratio
def ratio_C_to_A : ℚ := ratio_C_to_B * ratio_B_to_A

-- Distance run by C when A finishes
def distance_run_by_C_when_A_finishes : ℚ := distance_race * ratio_C_to_A

-- Distance C is behind the finish line when A finishes
def distance_C_behind_when_A_finishes : ℚ := distance_race - distance_run_by_C_when_A_finishes

theorem distance_behind_C (d_race : ℕ) (d_BA : ℕ) (d_CB : ℕ)
  (hA : d_race = 1000) (hB : d_BA = 50) (hC : d_CB = 100) :
  distance_C_behind_when_A_finishes = 145 :=
  by sorry

end NUMINAMATH_GPT_distance_behind_C_l2201_220157


namespace NUMINAMATH_GPT_cement_percentage_of_second_concrete_l2201_220141

theorem cement_percentage_of_second_concrete 
  (total_weight : ℝ) (final_percentage : ℝ) (partial_weight : ℝ) 
  (percentage_first_concrete : ℝ) :
  total_weight = 4500 →
  final_percentage = 0.108 →
  partial_weight = 1125 →
  percentage_first_concrete = 0.108 →
  ∃ percentage_second_concrete : ℝ, 
    percentage_second_concrete = 0.324 :=
by
  intros h1 h2 h3 h4
  let total_cement := total_weight * final_percentage
  let cement_first_concrete := partial_weight * percentage_first_concrete
  let cement_second_concrete := total_cement - cement_first_concrete
  let percentage_second_concrete := cement_second_concrete / partial_weight
  use percentage_second_concrete
  sorry

end NUMINAMATH_GPT_cement_percentage_of_second_concrete_l2201_220141


namespace NUMINAMATH_GPT_image_of_center_l2201_220137

-- Define the initial coordinates
def initial_coordinate : ℝ × ℝ := (-3, 4)

-- Function to reflect a point across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Function to translate a point up
def translate_up (p : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + units)

-- Definition of the final coordinate
noncomputable def final_coordinate : ℝ × ℝ :=
  translate_up (reflect_x initial_coordinate) 5

-- Theorem stating the final coordinate after transformations
theorem image_of_center : final_coordinate = (-3, 1) := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_image_of_center_l2201_220137


namespace NUMINAMATH_GPT_L_like_reflexive_l2201_220131

-- Definitions of the shapes and condition of being an "L-like shape"
inductive Shape
| A | B | C | D | E | LLike : Shape → Shape

-- reflection_equiv function representing reflection equivalence across a vertical dashed line
def reflection_equiv (s1 s2 : Shape) : Prop :=
sorry -- This would be defined according to the exact conditions of the shapes and reflection logic.

-- Given the shapes
axiom L_like : Shape
axiom A : Shape
axiom B : Shape
axiom C : Shape
axiom D : Shape
axiom E : Shape

-- The proof problem: Shape D is the mirrored reflection of the given "L-like shape" across a vertical dashed line
theorem L_like_reflexive :
  reflection_equiv L_like D :=
sorry

end NUMINAMATH_GPT_L_like_reflexive_l2201_220131


namespace NUMINAMATH_GPT_find_x_l2201_220196

theorem find_x (x : ℕ) (hcf lcm : ℕ):
  (hcf = Nat.gcd x 18) → 
  (lcm = Nat.lcm x 18) → 
  (lcm - hcf = 120) → 
  x = 42 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l2201_220196


namespace NUMINAMATH_GPT_olya_candies_l2201_220155

theorem olya_candies (P M T O : ℕ) (h1 : P + M + T + O = 88) (h2 : 1 ≤ P) (h3 : 1 ≤ M) (h4 : 1 ≤ T) (h5 : 1 ≤ O) (h6 : M + T = 57) (h7 : P > M) (h8 : P > T) (h9 : P > O) : O = 1 :=
by
  sorry

end NUMINAMATH_GPT_olya_candies_l2201_220155


namespace NUMINAMATH_GPT_european_confidence_95_european_teams_not_face_l2201_220171

-- Definitions for the conditions
def european_teams_round_of_16 := 44
def european_teams_not_round_of_16 := 22
def other_regions_round_of_16 := 36
def other_regions_not_round_of_16 := 58
def total_teams := 160

-- Formula for K^2 calculation
def k_value : ℚ := 3.841
def k_squared (n a_d_diff b_c_diff a b c d : ℚ) : ℚ :=
  n * ((a_d_diff - b_c_diff)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Definitions and calculation of K^2
def n1 := (european_teams_round_of_16 + other_regions_round_of_16 : ℚ)
def a_d_diff1 := (european_teams_round_of_16 * other_regions_not_round_of_16 : ℚ)
def b_c_diff1 := (european_teams_not_round_of_16 * other_regions_round_of_16 : ℚ)
def k_squared_result := k_squared n1 a_d_diff1 b_c_diff1
                                 (european_teams_round_of_16 + european_teams_not_round_of_16)
                                 (other_regions_round_of_16 + other_regions_not_round_of_16)
                                 total_teams total_teams

-- Theorem for 95% confidence derived
theorem european_confidence_95 :
  k_squared_result > k_value := sorry

-- Probability calculation setup
def total_ways_to_pair_teams : ℚ := 15
def ways_european_teams_not_face : ℚ := 6
def probability_european_teams_not_face := ways_european_teams_not_face / total_ways_to_pair_teams

-- Theorem for probability
theorem european_teams_not_face :
  probability_european_teams_not_face = 2 / 5 := sorry

end NUMINAMATH_GPT_european_confidence_95_european_teams_not_face_l2201_220171


namespace NUMINAMATH_GPT_average_movers_per_hour_l2201_220172

-- Define the main problem parameters
def total_people : ℕ := 3200
def days : ℕ := 4
def hours_per_day : ℕ := 24
def total_hours : ℕ := hours_per_day * days
def average_people_per_hour := total_people / total_hours

-- State the theorem to prove
theorem average_movers_per_hour :
  average_people_per_hour = 33 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_average_movers_per_hour_l2201_220172


namespace NUMINAMATH_GPT_expression_for_neg_x_l2201_220195

def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem expression_for_neg_x (f : ℝ → ℝ) (h_odd : odd_function f) (h_nonneg : ∀ (x : ℝ), 0 ≤ x → f x = x^2 - 2 * x) :
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2 * x :=
by 
  intros x hx 
  have hx_pos : -x > 0 := by linarith 
  have h_fx_neg : f (-x) = -f x := h_odd x
  rw [h_nonneg (-x) (by linarith)] at h_fx_neg
  linarith

end NUMINAMATH_GPT_expression_for_neg_x_l2201_220195


namespace NUMINAMATH_GPT_presidency_meeting_ways_l2201_220106

theorem presidency_meeting_ways : 
  ∃ (ways : ℕ), ways = 4 * 6 * 3 * 225 := sorry

end NUMINAMATH_GPT_presidency_meeting_ways_l2201_220106


namespace NUMINAMATH_GPT_car_dealership_sales_l2201_220142

theorem car_dealership_sales (x : ℕ)
  (h1 : 5 * x = 30 * 8)
  (h2 : 30 + x = 78) : 
  x = 48 :=
sorry

end NUMINAMATH_GPT_car_dealership_sales_l2201_220142


namespace NUMINAMATH_GPT_smaller_number_of_two_digits_product_3774_l2201_220154

theorem smaller_number_of_two_digits_product_3774 (a b : ℕ) (ha : 9 < a ∧ a < 100) (hb : 9 < b ∧ b < 100) (h : a * b = 3774) : a = 51 ∨ b = 51 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_of_two_digits_product_3774_l2201_220154


namespace NUMINAMATH_GPT_chalkboard_area_l2201_220163

theorem chalkboard_area (width : ℝ) (h₁ : width = 3.5) (length : ℝ) (h₂ : length = 2.3 * width) : 
  width * length = 28.175 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end NUMINAMATH_GPT_chalkboard_area_l2201_220163


namespace NUMINAMATH_GPT_problem_sufficient_necessary_condition_l2201_220140

open Set

variable {x : ℝ}

def P (x : ℝ) : Prop := abs (x - 2) < 3
def Q (x : ℝ) : Prop := x^2 - 8 * x + 15 < 0

theorem problem_sufficient_necessary_condition :
    (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬ Q x) :=
by
  sorry

end NUMINAMATH_GPT_problem_sufficient_necessary_condition_l2201_220140


namespace NUMINAMATH_GPT_cos_squared_formula_15deg_l2201_220122

theorem cos_squared_formula_15deg :
  (Real.cos (15 * Real.pi / 180))^2 - (1 / 2) = (Real.sqrt 3) / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_squared_formula_15deg_l2201_220122


namespace NUMINAMATH_GPT_vans_needed_l2201_220177

-- Definitions of conditions
def students : Nat := 2
def adults : Nat := 6
def capacity_per_van : Nat := 4

-- Main theorem to prove
theorem vans_needed : (students + adults) / capacity_per_van = 2 := by
  sorry

end NUMINAMATH_GPT_vans_needed_l2201_220177


namespace NUMINAMATH_GPT_boys_in_2nd_l2201_220120

def students_in_3rd := 19
def students_in_4th := 2 * students_in_3rd
def girls_in_2nd := 19
def total_students := 86
def students_in_2nd := total_students - students_in_3rd - students_in_4th

theorem boys_in_2nd : students_in_2nd - girls_in_2nd = 10 := by
  sorry

end NUMINAMATH_GPT_boys_in_2nd_l2201_220120


namespace NUMINAMATH_GPT_n_congruence_mod_9_l2201_220185

def n : ℕ := 2 + 333 + 5555 + 77777 + 999999 + 2222222 + 44444444 + 666666666

theorem n_congruence_mod_9 : n % 9 = 4 :=
by
  sorry

end NUMINAMATH_GPT_n_congruence_mod_9_l2201_220185


namespace NUMINAMATH_GPT_volume_expression_correct_l2201_220107

variable (x : ℝ)

def volume (x : ℝ) := x * (30 - 2 * x) * (20 - 2 * x)

theorem volume_expression_correct (h : x < 10) :
  volume x = 4 * x^3 - 100 * x^2 + 600 * x :=
by sorry

end NUMINAMATH_GPT_volume_expression_correct_l2201_220107


namespace NUMINAMATH_GPT_min_value_of_a_plus_b_minus_c_l2201_220132

theorem min_value_of_a_plus_b_minus_c (a b c : ℝ)
  (h : ∀ x y : ℝ, 3 * x + 4 * y - 5 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ 3 * x + 4 * y + 5) :
  a = 3 ∧ b = 4 ∧ -5 ≤ c ∧ c ≤ 5 ∧ a + b - c = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_of_a_plus_b_minus_c_l2201_220132


namespace NUMINAMATH_GPT_abc_product_l2201_220109

theorem abc_product (A B C D : ℕ) 
  (h1 : A + B + C + D = 64)
  (h2 : A + 3 = B - 3)
  (h3 : A + 3 = C * 3)
  (h4 : A + 3 = D / 3) :
  A * B * C * D = 19440 := 
by
  sorry

end NUMINAMATH_GPT_abc_product_l2201_220109


namespace NUMINAMATH_GPT_standard_equation_hyperbola_l2201_220116

-- Define necessary conditions
def condition_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :=
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)

def condition_asymptote (a b : ℝ) :=
  b / a = Real.sqrt 3

def condition_focus_hyperbola_parabola (a b : ℝ) :=
  (a^2 + b^2).sqrt = 4

-- Define the proof problem
theorem standard_equation_hyperbola (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h_asymptote : condition_asymptote a b)
  (h_focus : condition_focus_hyperbola_parabola a b) :
  ∀ x y : ℝ, (x^2 / 4 - y^2 / 12 = 1) :=
sorry

end NUMINAMATH_GPT_standard_equation_hyperbola_l2201_220116


namespace NUMINAMATH_GPT_sean_bought_two_soups_l2201_220193

theorem sean_bought_two_soups :
  ∃ (number_of_soups : ℕ),
    let soda_cost := 1
    let total_soda_cost := 3 * soda_cost
    let soup_cost := total_soda_cost
    let sandwich_cost := 3 * soup_cost
    let total_cost := 3 * soda_cost + sandwich_cost + soup_cost * number_of_soups
    total_cost = 18 ∧ number_of_soups = 2 :=
by
  sorry

end NUMINAMATH_GPT_sean_bought_two_soups_l2201_220193


namespace NUMINAMATH_GPT_cos_diff_l2201_220188

theorem cos_diff (α : ℝ) (h1 : Real.cos α = (Real.sqrt 2) / 10) (h2 : α > -π ∧ α < 0) :
  Real.cos (α - π / 4) = -3 / 5 :=
sorry

end NUMINAMATH_GPT_cos_diff_l2201_220188


namespace NUMINAMATH_GPT_marbles_remainder_l2201_220112

theorem marbles_remainder (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 6) : (r + p) % 8 = 3 :=
by sorry

end NUMINAMATH_GPT_marbles_remainder_l2201_220112


namespace NUMINAMATH_GPT_set_D_not_right_triangle_l2201_220150

theorem set_D_not_right_triangle :
  let a := 11
  let b := 12
  let c := 15
  a ^ 2 + b ^ 2 ≠ c ^ 2
:=
by
  let a := 11
  let b := 12
  let c := 15
  sorry

end NUMINAMATH_GPT_set_D_not_right_triangle_l2201_220150


namespace NUMINAMATH_GPT_base_8_addition_l2201_220101

theorem base_8_addition (X Y : ℕ) (h1 : Y + 2 % 8 = X % 8) (h2 : X + 3 % 8 = 2 % 8) : X + Y = 12 := by
  sorry

end NUMINAMATH_GPT_base_8_addition_l2201_220101


namespace NUMINAMATH_GPT_student_weight_l2201_220117

variable (S W : ℕ)

theorem student_weight (h1 : S - 5 = 2 * W) (h2 : S + W = 110) : S = 75 :=
by
  sorry

end NUMINAMATH_GPT_student_weight_l2201_220117


namespace NUMINAMATH_GPT_area_white_portion_l2201_220134

/-- The dimensions of the sign --/
def sign_width : ℝ := 7
def sign_height : ℝ := 20

/-- The areas of letters "S", "A", "V", and "E" --/
def area_S : ℝ := 14
def area_A : ℝ := 16
def area_V : ℝ := 12
def area_E : ℝ := 12

/-- Calculate the total area of the sign --/
def total_area_sign : ℝ := sign_width * sign_height

/-- Calculate the total area covered by the letters --/
def total_area_letters : ℝ := area_S + area_A + area_V + area_E

/-- Calculate the area of the white portion of the sign --/
theorem area_white_portion : total_area_sign - total_area_letters = 86 := by
  sorry

end NUMINAMATH_GPT_area_white_portion_l2201_220134


namespace NUMINAMATH_GPT_equal_naturals_of_infinite_divisibility_l2201_220149

theorem equal_naturals_of_infinite_divisibility
  (a b : ℕ)
  (h : ∀ᶠ n in Filter.atTop, (a^(n + 1) + b^(n + 1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end NUMINAMATH_GPT_equal_naturals_of_infinite_divisibility_l2201_220149


namespace NUMINAMATH_GPT_inheritance_amount_l2201_220118

-- Definitions of conditions
def inheritance (y : ℝ) : Prop :=
  let federalTaxes := 0.25 * y
  let remainingAfterFederal := 0.75 * y
  let stateTaxes := 0.1125 * y
  let totalTaxes := federalTaxes + stateTaxes
  totalTaxes = 12000

-- Theorem statement
theorem inheritance_amount (y : ℝ) (h : inheritance y) : y = 33103 :=
sorry

end NUMINAMATH_GPT_inheritance_amount_l2201_220118


namespace NUMINAMATH_GPT_cone_base_circumference_l2201_220124

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (C : ℝ) : 
  r = 5 → θ = 300 → C = (θ / 360) * (2 * Real.pi * r) → C = (25 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l2201_220124


namespace NUMINAMATH_GPT_even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l2201_220173

theorem even_ngon_parallel_edges (n : ℕ) (h : n % 2 = 0) :
  ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

theorem odd_ngon_no_two_parallel_edges (n : ℕ) (h : n % 2 = 1) :
  ¬ ∃ i j, i ≠ j ∧ (i + 1) % n + i % n = (j + 1) % n + j % n :=
sorry

end NUMINAMATH_GPT_even_ngon_parallel_edges_odd_ngon_no_two_parallel_edges_l2201_220173


namespace NUMINAMATH_GPT_original_commercial_length_l2201_220182

theorem original_commercial_length (x : ℝ) (h : 0.70 * x = 21) : x = 30 := sorry

end NUMINAMATH_GPT_original_commercial_length_l2201_220182


namespace NUMINAMATH_GPT_probability_two_females_one_male_l2201_220165

theorem probability_two_females_one_male
  (total_contestants : ℕ)
  (female_contestants : ℕ)
  (male_contestants : ℕ)
  (choose_count : ℕ)
  (total_combinations : ℕ)
  (female_combinations : ℕ)
  (male_combinations : ℕ)
  (favorable_outcomes : ℕ)
  (probability : ℚ)
  (h1 : total_contestants = 8)
  (h2 : female_contestants = 5)
  (h3 : male_contestants = 3)
  (h4 : choose_count = 3)
  (h5 : total_combinations = Nat.choose total_contestants choose_count)
  (h6 : female_combinations = Nat.choose female_contestants 2)
  (h7 : male_combinations = Nat.choose male_contestants 1)
  (h8 : favorable_outcomes = female_combinations * male_combinations)
  (h9 : probability = favorable_outcomes / total_combinations) :
  probability = 15 / 28 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_females_one_male_l2201_220165


namespace NUMINAMATH_GPT_distribute_weights_l2201_220144

theorem distribute_weights (max_weight : ℕ) (w_gbeans w_milk w_carrots w_apples w_bread w_rice w_oranges w_pasta : ℕ)
  (h_max_weight : max_weight = 20)
  (h_w_gbeans : w_gbeans = 4)
  (h_w_milk : w_milk = 6)
  (h_w_carrots : w_carrots = 2 * w_gbeans)
  (h_w_apples : w_apples = 3)
  (h_w_bread : w_bread = 1)
  (h_w_rice : w_rice = 5)
  (h_w_oranges : w_oranges = 2)
  (h_w_pasta : w_pasta = 3)
  : (w_gbeans + w_milk + w_carrots + w_apples + w_bread - 2 = max_weight) ∧ 
    (w_rice + w_oranges + w_pasta + 2 ≤ max_weight) :=
by
  sorry

end NUMINAMATH_GPT_distribute_weights_l2201_220144


namespace NUMINAMATH_GPT_percent_increase_l2201_220129

theorem percent_increase (original new : ℕ) (h1 : original = 30) (h2 : new = 60) :
  ((new - original) / original) * 100 = 100 := 
by
  sorry

end NUMINAMATH_GPT_percent_increase_l2201_220129


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l2201_220179

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l2201_220179


namespace NUMINAMATH_GPT_keith_attended_games_l2201_220105

def total_games : ℕ := 8
def missed_games : ℕ := 4
def attended_games (total : ℕ) (missed : ℕ) : ℕ := total - missed

theorem keith_attended_games : attended_games total_games missed_games = 4 := by
  sorry

end NUMINAMATH_GPT_keith_attended_games_l2201_220105


namespace NUMINAMATH_GPT_probability_no_shaded_in_2_by_2004_l2201_220190

noncomputable def probability_no_shaded_rectangle (total_rectangles shaded_rectangles : Nat) : ℚ :=
  1 - (shaded_rectangles : ℚ) / (total_rectangles : ℚ)

theorem probability_no_shaded_in_2_by_2004 :
  let rows := 2
  let cols := 2004
  let total_rectangles := (cols + 1) * cols / 2 * rows
  let shaded_rectangles := 501 * 2507 
  probability_no_shaded_rectangle total_rectangles shaded_rectangles = 1501 / 4008 :=
by
  sorry

end NUMINAMATH_GPT_probability_no_shaded_in_2_by_2004_l2201_220190


namespace NUMINAMATH_GPT_probability_non_defective_pencils_l2201_220175

theorem probability_non_defective_pencils :
  let total_pencils := 8
  let defective_pencils := 2
  let selected_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_combinations := Nat.choose total_pencils selected_pencils
  let non_defective_combinations := Nat.choose non_defective_pencils selected_pencils
  (non_defective_combinations:ℚ) / (total_combinations:ℚ) = 5 / 14 := by
  sorry

end NUMINAMATH_GPT_probability_non_defective_pencils_l2201_220175


namespace NUMINAMATH_GPT_solve_inequality_l2201_220127

theorem solve_inequality (x : ℝ) : (|2 * x - 1| < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l2201_220127
