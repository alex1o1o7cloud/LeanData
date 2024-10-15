import Mathlib

namespace NUMINAMATH_GPT_find_n_l2302_230242

-- Define the parameters of the arithmetic sequence
def a1 : ℤ := 1
def d : ℤ := 3
def a_n : ℤ := 298

-- The general formula for the nth term in an arithmetic sequence
def an (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The theorem to prove that n equals 100 given the conditions
theorem find_n (n : ℕ) (h : an n = a_n) : n = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2302_230242


namespace NUMINAMATH_GPT_divisibility_of_2b_by_a_l2302_230250

theorem divisibility_of_2b_by_a (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b)
  (h_cond : ∃ᶠ m in at_top, ∃ᶠ n in at_top, (∃ k₁ : ℕ, m^2 + a * n + b = k₁^2) ∧ (∃ k₂ : ℕ, n^2 + a * m + b = k₂^2)) :
  a ∣ 2 * b :=
sorry

end NUMINAMATH_GPT_divisibility_of_2b_by_a_l2302_230250


namespace NUMINAMATH_GPT_prove_trig_inequality_l2302_230239

noncomputable def trig_inequality : Prop :=
  (0 < 1 / 2) ∧ (1 / 2 < Real.pi / 6) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.sin x < Real.sin y) ∧
  (∀ x y : ℝ, (0 < x) ∧ (x < y) ∧ (y < Real.pi / 6) → Real.cos x > Real.cos y) →
  (Real.cos (1 / 2) > Real.tan (1 / 2) ∧ Real.tan (1 / 2) > Real.sin (1 / 2))

theorem prove_trig_inequality : trig_inequality :=
by
  sorry

end NUMINAMATH_GPT_prove_trig_inequality_l2302_230239


namespace NUMINAMATH_GPT_find_equation_for_second_machine_l2302_230294

theorem find_equation_for_second_machine (x : ℝ) : 
  (1 / 6) + (1 / x) = 1 / 3 ↔ (x = 6) := 
by 
  sorry

end NUMINAMATH_GPT_find_equation_for_second_machine_l2302_230294


namespace NUMINAMATH_GPT_required_fencing_l2302_230215

-- Definitions from conditions
def length_uncovered : ℝ := 30
def area : ℝ := 720

-- Prove that the amount of fencing required is 78 feet
theorem required_fencing : 
  ∃ (W : ℝ), (area = length_uncovered * W) ∧ (2 * W + length_uncovered = 78) := 
sorry

end NUMINAMATH_GPT_required_fencing_l2302_230215


namespace NUMINAMATH_GPT_locus_of_midpoint_of_tangents_l2302_230233

theorem locus_of_midpoint_of_tangents 
  (P Q Q1 Q2 : ℝ × ℝ)
  (L : P.2 = P.1 + 2)
  (C : ∀ p, p = Q1 ∨ p = Q2 → p.2 ^ 2 = 4 * p.1)
  (Q_is_midpoint : Q = ((Q1.1 + Q2.1) / 2, (Q1.2 + Q2.2) / 2)) :
  ∃ x y, (y - 1)^2 = 2 * (x - 3 / 2) := sorry

end NUMINAMATH_GPT_locus_of_midpoint_of_tangents_l2302_230233


namespace NUMINAMATH_GPT_area_of_support_is_15_l2302_230226

-- Define the given conditions
def initial_mass : ℝ := 60
def reduced_mass : ℝ := initial_mass - 10
def area_reduction : ℝ := 5
def mass_per_area_increase : ℝ := 1

-- Define the area of the support and prove that it is 15 dm^2
theorem area_of_support_is_15 (x : ℝ) 
  (initial_mass_eq : initial_mass / x = initial_mass / x) 
  (new_mass_eq : reduced_mass / (x - area_reduction) = initial_mass / x + mass_per_area_increase) : 
  x = 15 :=
  sorry

end NUMINAMATH_GPT_area_of_support_is_15_l2302_230226


namespace NUMINAMATH_GPT_height_of_first_podium_l2302_230228

noncomputable def height_of_podium_2_cm := 53.0
noncomputable def height_of_podium_2_mm := 7.0
noncomputable def height_on_podium_2_cm := 190.0
noncomputable def height_on_podium_1_cm := 232.0
noncomputable def height_on_podium_1_mm := 5.0

def expected_height_of_podium_1_cm := 96.2

theorem height_of_first_podium :
  let height_podium_2 := height_of_podium_2_cm + height_of_podium_2_mm / 10.0
  let height_podium_1 := height_on_podium_1_cm + height_on_podium_1_mm / 10.0
  let hyeonjoo_height := height_on_podium_2_cm - height_podium_2
  height_podium_1 - hyeonjoo_height = expected_height_of_podium_1_cm :=
by sorry

end NUMINAMATH_GPT_height_of_first_podium_l2302_230228


namespace NUMINAMATH_GPT_least_number_of_colors_needed_l2302_230247

-- Define the tessellation of hexagons
structure HexagonalTessellation :=
(adjacent : (ℕ × ℕ) → (ℕ × ℕ) → Prop)
(symm : ∀ {a b : ℕ × ℕ}, adjacent a b → adjacent b a)
(irrefl : ∀ a : ℕ × ℕ, ¬ adjacent a a)
(hex_property : ∀ a : ℕ × ℕ, ∃ b1 b2 b3 b4 b5 b6,
  adjacent a b1 ∧ adjacent a b2 ∧ adjacent a b3 ∧ adjacent a b4 ∧ adjacent a b5 ∧ adjacent a b6)

-- Define a coloring function for a HexagonalTessellation
def coloring (T : HexagonalTessellation) (colors : ℕ) :=
(∀ (a b : ℕ × ℕ), T.adjacent a b → a ≠ b → colors ≥ 1 → colors ≤ 3)

-- Statement to prove the minimum number of colors required
theorem least_number_of_colors_needed (T : HexagonalTessellation) :
  ∃ colors, coloring T colors ∧ colors = 3 :=
sorry

end NUMINAMATH_GPT_least_number_of_colors_needed_l2302_230247


namespace NUMINAMATH_GPT_greatest_num_of_coins_l2302_230272

-- Define the total amount of money Carlos has in U.S. coins.
def total_value : ℝ := 5.45

-- Define the value of each type of coin.
def quarter_value : ℝ := 0.25
def dime_value : ℝ := 0.10
def nickel_value : ℝ := 0.05

-- Define the number of quarters, dimes, and nickels Carlos has.
def num_coins (q : ℕ) := quarter_value * q + dime_value * q + nickel_value * q

-- The main theorem: Carlos can have at most 13 quarters, dimes, and nickels.
theorem greatest_num_of_coins (q : ℕ) :
  num_coins q = total_value → q ≤ 13 :=
sorry

end NUMINAMATH_GPT_greatest_num_of_coins_l2302_230272


namespace NUMINAMATH_GPT_rectangle_height_l2302_230216

-- Defining the conditions
def base : ℝ := 9
def area : ℝ := 33.3

-- Stating the proof problem
theorem rectangle_height : (area / base) = 3.7 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_height_l2302_230216


namespace NUMINAMATH_GPT_keys_per_lock_l2302_230210

-- Define the given conditions
def num_complexes := 2
def apartments_per_complex := 12
def total_keys := 72

-- Calculate the total number of apartments
def total_apartments := num_complexes * apartments_per_complex

-- The theorem statement to prove
theorem keys_per_lock : total_keys / total_apartments = 3 := 
by
  sorry

end NUMINAMATH_GPT_keys_per_lock_l2302_230210


namespace NUMINAMATH_GPT_problem_statement_l2302_230277

noncomputable def given_function (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - 2 * x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem problem_statement :
  is_even_function given_function ∧ smallest_positive_period given_function Real.pi :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l2302_230277


namespace NUMINAMATH_GPT_part1_l2302_230203

variables (a c : ℝ × ℝ)
variables (a_parallel_c : ∃ k : ℝ, c = (k * a.1, k * a.2))
variables (a_value : a = (1,2))
variables (c_magnitude : (c.1 ^ 2 + c.2 ^ 2) = (3 * Real.sqrt 5) ^ 2)

theorem part1: c = (3, 6) ∨ c = (-3, -6) :=
by
  sorry

end NUMINAMATH_GPT_part1_l2302_230203


namespace NUMINAMATH_GPT_digits_making_number_divisible_by_4_l2302_230253

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end NUMINAMATH_GPT_digits_making_number_divisible_by_4_l2302_230253


namespace NUMINAMATH_GPT_integer_solutions_of_quadratic_l2302_230206

theorem integer_solutions_of_quadratic (k : ℤ) :
  ∀ x : ℤ, (6 - k) * (9 - k) * x^2 - (117 - 15 * k) * x + 54 = 0 ↔
  k = 3 ∨ k = 7 ∨ k = 15 ∨ k = 6 ∨ k = 9 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_of_quadratic_l2302_230206


namespace NUMINAMATH_GPT_g_is_odd_l2302_230227

noncomputable def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_g_is_odd_l2302_230227


namespace NUMINAMATH_GPT_three_collinear_points_l2302_230237

theorem three_collinear_points (f : ℝ → Prop) (h_black_or_white : ∀ (x : ℝ), f x = true ∨ f x = false)
: ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (b = (a + c) / 2) ∧ ((f a = f b) ∧ (f b = f c)) :=
sorry

end NUMINAMATH_GPT_three_collinear_points_l2302_230237


namespace NUMINAMATH_GPT_value_of_a_l2302_230205

theorem value_of_a (x : ℝ) (n : ℕ) (h : x > 0) (h_n : n > 0) :
  (∀ k : ℕ, 1 ≤ k → k ≤ n → x + k ≥ k + 1) → a = n^n :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l2302_230205


namespace NUMINAMATH_GPT_lemon_cookies_amount_l2302_230271

def cookies_problem 
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) : Prop :=
  jenny_pb_cookies = 40 ∧
  jenny_cc_cookies = 50 ∧
  marcus_pb_cookies = 30 ∧
  total_pb_cookies = jenny_pb_cookies + marcus_pb_cookies ∧
  total_pb_cookies = 70 ∧
  total_non_pb_cookies = jenny_cc_cookies + marcus_lemon_cookies ∧
  total_pb_cookies = total_non_pb_cookies

theorem lemon_cookies_amount
  (jenny_pb_cookies : ℕ) (jenny_cc_cookies : ℕ) (marcus_pb_cookies : ℕ) (marcus_lemon_cookies : ℕ)
  (total_pb_cookies : ℕ) (total_non_pb_cookies : ℕ) :
  cookies_problem jenny_pb_cookies jenny_cc_cookies marcus_pb_cookies marcus_lemon_cookies total_pb_cookies total_non_pb_cookies →
  marcus_lemon_cookies = 20 :=
by
  sorry

end NUMINAMATH_GPT_lemon_cookies_amount_l2302_230271


namespace NUMINAMATH_GPT_find_value_of_a_minus_b_l2302_230288

variable (a b : ℝ)

theorem find_value_of_a_minus_b (h1 : |a| = 2) (h2 : b^2 = 9) (h3 : a < b) :
  a - b = -1 ∨ a - b = -5 := 
sorry

end NUMINAMATH_GPT_find_value_of_a_minus_b_l2302_230288


namespace NUMINAMATH_GPT_find_f_neg_2010_6_l2302_230204

noncomputable def f : ℝ → ℝ := sorry

axiom f_add_one (x : ℝ) : f (x + 1) + f x = 3

axiom f_on_interval (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = 2 - x

theorem find_f_neg_2010_6 : f (-2010.6) = 1.4 := by {
  sorry
}

end NUMINAMATH_GPT_find_f_neg_2010_6_l2302_230204


namespace NUMINAMATH_GPT_closest_integers_to_2013_satisfy_trig_eq_l2302_230219

noncomputable def closestIntegersSatisfyingTrigEq (x : ℝ) : Prop := 
  (2^(Real.sin x)^2 + 2^(Real.cos x)^2 = 2 * Real.sqrt 2)

theorem closest_integers_to_2013_satisfy_trig_eq : closestIntegersSatisfyingTrigEq (1935 * (Real.pi / 180)) ∧ closestIntegersSatisfyingTrigEq (2025 * (Real.pi / 180)) :=
sorry

end NUMINAMATH_GPT_closest_integers_to_2013_satisfy_trig_eq_l2302_230219


namespace NUMINAMATH_GPT_min_value_expression_l2302_230265

theorem min_value_expression (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 
  = 5^(5/4) - 10 * Real.sqrt (5^(1/4)) + 5 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l2302_230265


namespace NUMINAMATH_GPT_highest_power_of_3_divides_N_l2302_230298

-- Define the range of two-digit numbers and the concatenation function
def concatTwoDigitIntegers : ℕ := sorry  -- Placeholder for the concatenation implementation

-- Integer N formed by concatenating integers from 31 to 68
def N := concatTwoDigitIntegers

-- The statement proving the highest power of 3 dividing N is 3^1
theorem highest_power_of_3_divides_N :
  (∃ k : ℕ, 3^k ∣ N ∧ ¬ 3^(k+1) ∣ N) ∧ 3^1 ∣ N ∧ ¬ 3^2 ∣ N :=
by
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_highest_power_of_3_divides_N_l2302_230298


namespace NUMINAMATH_GPT_avg_of_two_numbers_l2302_230275

theorem avg_of_two_numbers (a b c d : ℕ) (h_different: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_positive: a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0)
  (h_average: (a + b + c + d) / 4 = 4)
  (h_max_diff: ∀ x y : ℕ, (x ≠ y ∧ x > 0 ∧ y > 0 ∧ x ≠ a ∧ x ≠ b ∧ x ≠ c ∧ x ≠ d ∧ y ≠ a ∧ y ≠ b ∧ y ≠ c ∧ y ≠ d) → (max x y - min x y <= max a d - min a d)) : 
  (a + b + c + d - min a (min b (min c d)) - max a (max b (max c d))) / 2 = 5 / 2 :=
by sorry

end NUMINAMATH_GPT_avg_of_two_numbers_l2302_230275


namespace NUMINAMATH_GPT_sqrt_div_l2302_230268

theorem sqrt_div (a b : ℝ) (h1 : a = 28) (h2 : b = 7) :
  Real.sqrt a / Real.sqrt b = 2 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_div_l2302_230268


namespace NUMINAMATH_GPT_train_speeds_proof_l2302_230256

-- Defining the initial conditions
variables (v_g v_p v_e : ℝ)
variables (t_g t_p t_e : ℝ) -- t_g, t_p, t_e are the times for goods, passenger, and express trains respectively

-- Conditions given in the problem
def goods_train_speed := v_g 
def passenger_train_speed := 90 
def express_train_speed := 1.5 * 90

-- Passenger train catches up with the goods train after 4 hours
def passenger_goods_catchup := 90 * 4 = v_g * (t_g + 4) - v_g * t_g

-- Express train catches up with the passenger train after 3 hours
def express_passenger_catchup := 1.5 * 90 * 3 = 90 * (3 + 4)

-- Theorem to prove the speeds of each train
theorem train_speeds_proof (h1 : 90 * 4 = v_g * (t_g + 4) - v_g * t_g)
                           (h2 : 1.5 * 90 * 3 = 90 * (3 + 4)) :
    v_g = 90 ∧ v_p = 90 ∧ v_e = 135 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speeds_proof_l2302_230256


namespace NUMINAMATH_GPT_correct_options_l2302_230289

theorem correct_options (a b c : ℝ) (h1 : ∀ x : ℝ, (a*x^2 + b*x + c > 0) ↔ (-3 < x ∧ x < 2)) :
  (a < 0) ∧ (a + b + c > 0) ∧ (∀ x, (b*x + c > 0) ↔ x > 6) = False ∧ (∀ x, (c*x^2 + b*x + a < 0) ↔ (-1/3 < x ∧ x < 1/2)) :=
by 
  sorry

end NUMINAMATH_GPT_correct_options_l2302_230289


namespace NUMINAMATH_GPT_division_value_l2302_230236

theorem division_value (x y : ℝ) (h1 : (x - 5) / y = 7) (h2 : (x - 14) / 10 = 4) : y = 7 :=
sorry

end NUMINAMATH_GPT_division_value_l2302_230236


namespace NUMINAMATH_GPT_smallest_a_gcd_77_88_l2302_230222

theorem smallest_a_gcd_77_88 :
  ∃ (a : ℕ), a > 0 ∧ (∀ b, b > 0 → b < a → (gcd b 77 > 1 ∧ gcd b 88 > 1) → false) ∧ gcd a 77 > 1 ∧ gcd a 88 > 1 ∧ a = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_gcd_77_88_l2302_230222


namespace NUMINAMATH_GPT_number_of_extreme_value_points_l2302_230263

noncomputable def f (x : ℝ) : ℝ := x^2 + x - Real.log x

theorem number_of_extreme_value_points : ∃! c : ℝ, c > 0 ∧ (deriv f c = 0) :=
by
  sorry

end NUMINAMATH_GPT_number_of_extreme_value_points_l2302_230263


namespace NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2302_230287

theorem distance_between_foci_of_hyperbola :
  ∀ (x y : ℝ), x^2 - 6 * x - 4 * y^2 - 8 * y = 27 → (4 * Real.sqrt 10) = 4 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_foci_of_hyperbola_l2302_230287


namespace NUMINAMATH_GPT_volleyball_count_l2302_230291

theorem volleyball_count (x y z : ℕ) (h1 : x + y + z = 20) (h2 : 6 * x + 3 * y + z = 33) : z = 15 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_count_l2302_230291


namespace NUMINAMATH_GPT_james_total_catch_l2302_230270

def pounds_of_trout : ℕ := 200
def pounds_of_salmon : ℕ := pounds_of_trout + (pounds_of_trout / 2)
def pounds_of_tuna : ℕ := 2 * pounds_of_salmon
def total_pounds_of_fish : ℕ := pounds_of_trout + pounds_of_salmon + pounds_of_tuna

theorem james_total_catch : total_pounds_of_fish = 1100 := by
  sorry

end NUMINAMATH_GPT_james_total_catch_l2302_230270


namespace NUMINAMATH_GPT_number_of_bottle_caps_put_inside_l2302_230224

-- Definitions according to the conditions
def initial_bottle_caps : ℕ := 7
def final_bottle_caps : ℕ := 14
def additional_bottle_caps (initial final : ℕ) := final - initial

-- The main theorem to prove
theorem number_of_bottle_caps_put_inside : additional_bottle_caps initial_bottle_caps final_bottle_caps = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bottle_caps_put_inside_l2302_230224


namespace NUMINAMATH_GPT_total_hours_watched_l2302_230201

/-- Given a 100-hour long video, Lila watches it at twice the average speed, and Roger watches it at the average speed. Both watched six such videos. We aim to prove that the total number of hours watched by Lila and Roger together is 900 hours. -/
theorem total_hours_watched {video_length lila_speed_multiplier roger_speed_multiplier num_videos : ℕ} 
  (h1 : video_length = 100)
  (h2 : lila_speed_multiplier = 2) 
  (h3 : roger_speed_multiplier = 1)
  (h4 : num_videos = 6) :
  (num_videos * (video_length / lila_speed_multiplier) + num_videos * (video_length / roger_speed_multiplier)) = 900 := 
sorry

end NUMINAMATH_GPT_total_hours_watched_l2302_230201


namespace NUMINAMATH_GPT_proof_inequality_l2302_230283

noncomputable def inequality (a b c : ℝ) : Prop :=
  a + 2 * b + c = 1 ∧ a^2 + b^2 + c^2 = 1 → -2/3 ≤ c ∧ c ≤ 1

theorem proof_inequality (a b c : ℝ) (h : a + 2 * b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : -2/3 ≤ c ∧ c ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_inequality_l2302_230283


namespace NUMINAMATH_GPT_bridge_length_l2302_230230

theorem bridge_length (train_length : ℕ) (crossing_time : ℕ) (train_speed_kmh : ℕ) :
  train_length = 500 → crossing_time = 45 → train_speed_kmh = 64 → 
  ∃ (bridge_length : ℝ), bridge_length = 300.1 :=
by
  intros h1 h2 h3
  have speed_mps := (train_speed_kmh * 1000) / 3600
  have total_distance := speed_mps * crossing_time
  have bridge_length_calculated := total_distance - train_length
  use bridge_length_calculated
  sorry

end NUMINAMATH_GPT_bridge_length_l2302_230230


namespace NUMINAMATH_GPT_students_not_in_biology_l2302_230243

theorem students_not_in_biology (total_students : ℕ) (percent_enrolled : ℝ) (students_enrolled : ℕ) (students_not_enrolled : ℕ) : 
  total_students = 880 ∧ percent_enrolled = 32.5 ∧ total_students - students_enrolled = students_not_enrolled ∧ students_enrolled = 286 ∧ students_not_enrolled = 594 :=
by
  sorry

end NUMINAMATH_GPT_students_not_in_biology_l2302_230243


namespace NUMINAMATH_GPT_greatest_overlap_l2302_230257

-- Defining the conditions based on the problem statement
def percentage_internet (n : ℕ) : Prop := n = 35
def percentage_snacks (m : ℕ) : Prop := m = 70

-- The theorem to prove the greatest possible overlap
theorem greatest_overlap (n m k : ℕ) (hn : percentage_internet n) (hm : percentage_snacks m) : 
  k ≤ 35 :=
by sorry

end NUMINAMATH_GPT_greatest_overlap_l2302_230257


namespace NUMINAMATH_GPT_minimum_ellipse_area_l2302_230218

theorem minimum_ellipse_area (a b : ℝ) (h₁ : 4 * (a : ℝ) ^ 2 * b ^ 2 = a ^ 2 + b ^ 4)
  (h₂ : (∀ x y : ℝ, ((x - 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1)) 
       ∧ (∀ x y : ℝ, ((x + 2) ^ 2 + y ^ 2 ≤ 4 → x ^ 2 / (4 * a ^ 2) + y ^ 2 / (4 * b ^ 2) ≤ 1))) : 
  ∃ k : ℝ, (k = 16) ∧ (π * (4 * a * b) = k * π) :=
by sorry

end NUMINAMATH_GPT_minimum_ellipse_area_l2302_230218


namespace NUMINAMATH_GPT_probability_heads_twice_in_three_flips_l2302_230241

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_heads_twice_in_three_flips :
  let p := 0.5
  let n := 3
  let k := 2
  (binomial_coefficient n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_probability_heads_twice_in_three_flips_l2302_230241


namespace NUMINAMATH_GPT_jonah_profit_l2302_230217

def cost_per_pineapple (quantity : ℕ) : ℝ :=
  if quantity > 50 then 1.60 else if quantity > 40 then 1.80 else 2.00

def total_cost (quantity : ℕ) : ℝ :=
  cost_per_pineapple quantity * quantity

def bundle_revenue (bundles : ℕ) : ℝ :=
  bundles * 20

def single_ring_revenue (rings : ℕ) : ℝ :=
  rings * 4

def total_revenue (bundles : ℕ) (rings : ℕ) : ℝ :=
  bundle_revenue bundles + single_ring_revenue rings

noncomputable def profit (quantity bundles rings : ℕ) : ℝ :=
  total_revenue bundles rings - total_cost quantity

theorem jonah_profit : profit 60 35 150 = 1204 := by
  sorry

end NUMINAMATH_GPT_jonah_profit_l2302_230217


namespace NUMINAMATH_GPT_ken_situps_l2302_230267

variable (K : ℕ)

theorem ken_situps (h1 : Nathan = 2 * K)
                   (h2 : Bob = 3 * K / 2)
                   (h3 : Bob = K + 10) : 
                   K = 20 := 
by
  sorry

end NUMINAMATH_GPT_ken_situps_l2302_230267


namespace NUMINAMATH_GPT_treble_of_doubled_and_increased_l2302_230202

theorem treble_of_doubled_and_increased (initial_number : ℕ) (result : ℕ) : 
  initial_number = 15 → (initial_number * 2 + 5) * 3 = result → result = 105 := 
by 
  intros h1 h2
  rw [h1] at h2
  linarith

end NUMINAMATH_GPT_treble_of_doubled_and_increased_l2302_230202


namespace NUMINAMATH_GPT_sum_of_neg_ints_l2302_230208

theorem sum_of_neg_ints (xs : List Int) (h₁ : ∀ x ∈ xs, x < 0)
  (h₂ : ∀ x ∈ xs, 3 < |x| ∧ |x| < 6) : xs.sum = -9 :=
sorry

end NUMINAMATH_GPT_sum_of_neg_ints_l2302_230208


namespace NUMINAMATH_GPT_umbrella_cost_l2302_230214

theorem umbrella_cost (number_of_umbrellas : Nat) (total_cost : Nat) (h1 : number_of_umbrellas = 3) (h2 : total_cost = 24) :
  (total_cost / number_of_umbrellas) = 8 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_umbrella_cost_l2302_230214


namespace NUMINAMATH_GPT_organic_fertilizer_prices_l2302_230266

theorem organic_fertilizer_prices
  (x y : ℝ)
  (h1 : x - y = 100)
  (h2 : 2 * x + y = 1700) :
  x = 600 ∧ y = 500 :=
by {
  sorry
}

end NUMINAMATH_GPT_organic_fertilizer_prices_l2302_230266


namespace NUMINAMATH_GPT_number_of_matches_is_85_l2302_230299

open Nat

/-- This definition calculates combinations of n taken k at a time. -/
def binom (n k : ℕ) : ℕ := n.choose k

/-- The calculation of total number of matches in the entire tournament. -/
def total_matches (groups teams_per_group : ℕ) : ℕ :=
  let matches_per_group := binom teams_per_group 2
  let total_matches_first_round := groups * matches_per_group
  let matches_final_round := binom groups 2
  total_matches_first_round + matches_final_round

/-- Theorem proving the total number of matches played is 85, given 5 groups with 6 teams each. -/
theorem number_of_matches_is_85 : total_matches 5 6 = 85 :=
  by
  sorry

end NUMINAMATH_GPT_number_of_matches_is_85_l2302_230299


namespace NUMINAMATH_GPT_pizzas_served_during_lunch_l2302_230232

theorem pizzas_served_during_lunch {total_pizzas dinner_pizzas lunch_pizzas: ℕ} 
(h_total: total_pizzas = 15) (h_dinner: dinner_pizzas = 6) (h_eq: total_pizzas = dinner_pizzas + lunch_pizzas) : 
lunch_pizzas = 9 := by
  sorry

end NUMINAMATH_GPT_pizzas_served_during_lunch_l2302_230232


namespace NUMINAMATH_GPT_roots_odd_even_l2302_230285

theorem roots_odd_even (n : ℤ) (x1 x2 : ℤ) (h_eqn : x1^2 + (4 * n + 1) * x1 + 2 * n = 0) (h_eqn' : x2^2 + (4 * n + 1) * x2 + 2 * n = 0) :
  ((x1 % 2 = 0 ∧ x2 % 2 ≠ 0) ∨ (x1 % 2 ≠ 0 ∧ x2 % 2 = 0)) :=
sorry

end NUMINAMATH_GPT_roots_odd_even_l2302_230285


namespace NUMINAMATH_GPT_ratio_bisector_circumradius_l2302_230235

theorem ratio_bisector_circumradius (h_a h_b h_c : ℝ) (ha_val : h_a = 1/3) (hb_val : h_b = 1/4) (hc_val : h_c = 1/5) :
  ∃ (CD R : ℝ), CD / R = 24 * Real.sqrt 2 / 35 :=
by
  sorry

end NUMINAMATH_GPT_ratio_bisector_circumradius_l2302_230235


namespace NUMINAMATH_GPT_janet_percentage_l2302_230259

-- Define the number of snowballs made by Janet and her brother
def janet_snowballs : ℕ := 50
def brother_snowballs : ℕ := 150

-- Define the total number of snowballs
def total_snowballs : ℕ := janet_snowballs + brother_snowballs

-- Define the percentage function
def percentage (part : ℕ) (whole : ℕ) : ℕ := (part * 100) / whole

-- State the theorem to be proved
theorem janet_percentage : percentage janet_snowballs total_snowballs = 25 := by
  sorry

end NUMINAMATH_GPT_janet_percentage_l2302_230259


namespace NUMINAMATH_GPT_tileable_contains_domino_l2302_230251

theorem tileable_contains_domino {m n a b : ℕ} (h_m : m ≥ a) (h_n : n ≥ b) :
  (∀ (x : ℕ) (y : ℕ), x + a ≤ m → y + b ≤ n → ∃ (p : ℕ) (q : ℕ), p = x ∧ q = y) :=
sorry

end NUMINAMATH_GPT_tileable_contains_domino_l2302_230251


namespace NUMINAMATH_GPT_quadrilateral_sides_l2302_230273

noncomputable def circle_radius : ℝ := 25
noncomputable def diagonal1_length : ℝ := 48
noncomputable def diagonal2_length : ℝ := 40

theorem quadrilateral_sides :
  ∃ (a b c d : ℝ),
    (a = 5 * Real.sqrt 10 ∧ 
    b = 9 * Real.sqrt 10 ∧ 
    c = 13 * Real.sqrt 10 ∧ 
    d = 15 * Real.sqrt 10) ∧ 
    (diagonal1_length = 48 ∧ 
    diagonal2_length = 40 ∧ 
    circle_radius = 25) :=
sorry

end NUMINAMATH_GPT_quadrilateral_sides_l2302_230273


namespace NUMINAMATH_GPT_product_not_power_of_two_l2302_230297

theorem product_not_power_of_two (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ∃ k : ℕ, (36 * a + b) * (a + 36 * b) ≠ 2^k :=
by
  sorry

end NUMINAMATH_GPT_product_not_power_of_two_l2302_230297


namespace NUMINAMATH_GPT_polygon_diagonals_formula_l2302_230278

theorem polygon_diagonals_formula (n : ℕ) (h₁ : n = 5) (h₂ : 2 * n = (n * (n - 3)) / 2) :
  ∃ D : ℕ, D = n * (n - 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_polygon_diagonals_formula_l2302_230278


namespace NUMINAMATH_GPT_percentile_75_eq_95_l2302_230261

def seventy_fifth_percentile (data : List ℕ) : ℕ := sorry

theorem percentile_75_eq_95 : seventy_fifth_percentile [92, 93, 88, 99, 89, 95] = 95 := 
sorry

end NUMINAMATH_GPT_percentile_75_eq_95_l2302_230261


namespace NUMINAMATH_GPT_KodyAgeIs32_l2302_230229

-- Definition for Mohamed's current age
def mohamedCurrentAge : ℕ := 2 * 30

-- Definition for Mohamed's age four years ago
def mohamedAgeFourYrsAgo : ℕ := mohamedCurrentAge - 4

-- Definition for Kody's age four years ago
def kodyAgeFourYrsAgo : ℕ := mohamedAgeFourYrsAgo / 2

-- Definition to check Kody's current age
def kodyCurrentAge : ℕ := kodyAgeFourYrsAgo + 4

theorem KodyAgeIs32 : kodyCurrentAge = 32 := by
  sorry

end NUMINAMATH_GPT_KodyAgeIs32_l2302_230229


namespace NUMINAMATH_GPT_find_phi_l2302_230220

open Real

theorem find_phi (φ : ℝ) (hφ : |φ| < π / 2)
  (h_symm : ∀ x, sin (2 * x + φ) = sin (2 * ((2 * π / 3 - x) / 2) + φ)) :
  φ = -π / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_phi_l2302_230220


namespace NUMINAMATH_GPT_min_value_expr_l2302_230274

theorem min_value_expr (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (hxyz : x * y * z = 1) :
  2 * x^2 + 8 * x * y + 6 * y^2 + 16 * y * z + 3 * z^2 ≥ 24 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expr_l2302_230274


namespace NUMINAMATH_GPT_range_of_a_l2302_230244

theorem range_of_a:
  (∃ x : ℝ, 1 ≤ x ∧ |x - a| + x - 4 ≤ 0) → (-2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2302_230244


namespace NUMINAMATH_GPT_completely_factored_form_l2302_230245

theorem completely_factored_form (x : ℤ) :
  (12 * x ^ 3 + 95 * x - 6) - (-3 * x ^ 3 + 5 * x - 6) = 15 * x * (x ^ 2 + 6) :=
by
  sorry

end NUMINAMATH_GPT_completely_factored_form_l2302_230245


namespace NUMINAMATH_GPT_nut_game_winning_strategy_l2302_230269

theorem nut_game_winning_strategy (N : ℕ) (h : N > 2) : ∃ second_player_wins : Prop, second_player_wins :=
sorry

end NUMINAMATH_GPT_nut_game_winning_strategy_l2302_230269


namespace NUMINAMATH_GPT_find_y_l2302_230293

theorem find_y (y : ℝ) (h : (15 + 28 + y) / 3 = 25) : y = 32 := by
  sorry

end NUMINAMATH_GPT_find_y_l2302_230293


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l2302_230262

theorem problem1_solution (x : ℝ) : (x^2 - 4 * x = 5) → (x = 5 ∨ x = -1) :=
by sorry

theorem problem2_solution (x : ℝ) : (2 * x^2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1/2) :=
by sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l2302_230262


namespace NUMINAMATH_GPT_cos_seven_pi_over_six_l2302_230209

open Real

theorem cos_seven_pi_over_six : cos (7 * π / 6) = - (sqrt 3 / 2) := 
by
  sorry

end NUMINAMATH_GPT_cos_seven_pi_over_six_l2302_230209


namespace NUMINAMATH_GPT_relationship_between_mode_median_mean_l2302_230255

def data_set : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode : ℕ := 60 -- derived from the problem conditions
def median : ℕ := 50 -- derived from the problem conditions
def mean : ℚ := 330 / 7 -- derived from the problem conditions

theorem relationship_between_mode_median_mean :
  mode > median ∧ median > mean :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_mode_median_mean_l2302_230255


namespace NUMINAMATH_GPT_promotional_rate_ratio_is_one_third_l2302_230252

-- Define the conditions
def normal_monthly_charge : ℕ := 30
def extra_fee : ℕ := 15
def total_paid : ℕ := 175

-- Define the total data plan amount equation
def calculate_total (P : ℕ) : ℕ :=
  P + 2 * normal_monthly_charge + (normal_monthly_charge + extra_fee) + 2 * normal_monthly_charge

theorem promotional_rate_ratio_is_one_third (P : ℕ) (hP : calculate_total P = total_paid) :
  P * 3 = normal_monthly_charge :=
by sorry

end NUMINAMATH_GPT_promotional_rate_ratio_is_one_third_l2302_230252


namespace NUMINAMATH_GPT_find_number_l2302_230284

theorem find_number (x : ℕ) (h : 695 - 329 = x - 254) : x = 620 :=
sorry

end NUMINAMATH_GPT_find_number_l2302_230284


namespace NUMINAMATH_GPT_blue_beads_count_l2302_230223

-- Define variables and conditions
variables (r b : ℕ)

-- Define the conditions
def condition1 : Prop := r = 30
def condition2 : Prop := r / 3 = b / 2

-- State the theorem
theorem blue_beads_count (h1 : condition1 r) (h2 : condition2 r b) : b = 20 :=
sorry

end NUMINAMATH_GPT_blue_beads_count_l2302_230223


namespace NUMINAMATH_GPT_zero_point_in_range_l2302_230295

theorem zero_point_in_range (a : ℝ) (x1 x2 x3 : ℝ) (h1 : 0 < a) (h2 : a < 2) (h3 : x1 < x2) (h4 : x2 < x3)
  (hx1 : (x1^3 - 4*x1 + a) = 0) (hx2 : (x2^3 - 4*x2 + a) = 0) (hx3 : (x3^3 - 4*x3 + a) = 0) :
  0 < x2 ∧ x2 < 1 :=
by
  sorry

end NUMINAMATH_GPT_zero_point_in_range_l2302_230295


namespace NUMINAMATH_GPT_largest_x_to_floor_ratio_l2302_230260

theorem largest_x_to_floor_ratio : ∃ x : ℝ, (⌊x⌋ / x = 9 / 10 ∧ ∀ y : ℝ, (⌊y⌋ / y = 9 / 10 → y ≤ x)) :=
sorry

end NUMINAMATH_GPT_largest_x_to_floor_ratio_l2302_230260


namespace NUMINAMATH_GPT_parabola_sum_coefficients_l2302_230234

theorem parabola_sum_coefficients :
  ∃ (a b c : ℤ), 
    (∀ x : ℝ, (x = 0 → a * (x^2) + b * x + c = 1)) ∧
    (∀ x : ℝ, (x = 2 → a * (x^2) + b * x + c = 9)) ∧
    (a * (1^2) + b * 1 + c = 4)
  → a + b + c = 4 :=
by sorry

end NUMINAMATH_GPT_parabola_sum_coefficients_l2302_230234


namespace NUMINAMATH_GPT_sum_of_first_11_terms_l2302_230249

theorem sum_of_first_11_terms (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 4 + a 8 = 16) : (11 / 2) * (a 1 + a 11) = 88 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_11_terms_l2302_230249


namespace NUMINAMATH_GPT_sum_of_numbers_in_ratio_with_lcm_l2302_230207

theorem sum_of_numbers_in_ratio_with_lcm
  (x : ℕ)
  (h1 : Nat.lcm (2 * x) (Nat.lcm (3 * x) (5 * x)) = 120) :
  (2 * x) + (3 * x) + (5 * x) = 40 := 
sorry

end NUMINAMATH_GPT_sum_of_numbers_in_ratio_with_lcm_l2302_230207


namespace NUMINAMATH_GPT_medium_kite_area_l2302_230246

-- Define the points and the spacing on the grid
structure Point :=
mk :: (x : ℕ) (y : ℕ)

def medium_kite_vertices : List Point :=
[Point.mk 0 4, Point.mk 4 10, Point.mk 12 4, Point.mk 4 0]

def grid_spacing : ℕ := 2

-- Function to calculate the area of a kite given list of vertices and spacing
noncomputable def area_medium_kite (vertices : List Point) (spacing : ℕ) : ℕ := sorry

-- The theorem to be proved
theorem medium_kite_area (vertices : List Point) (spacing : ℕ) :
  vertices = medium_kite_vertices ∧ spacing = grid_spacing → area_medium_kite vertices spacing = 288 := 
by {
  -- The detailed proof would go here
  sorry
}

end NUMINAMATH_GPT_medium_kite_area_l2302_230246


namespace NUMINAMATH_GPT_find_a_l2302_230240

-- Assuming the existence of functions and variables as per conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Defining the given conditions
axiom cond1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x - 5
axiom cond2 : f a = 6

-- Now stating the proof goal
theorem find_a : a = 7 / 4 := by
  sorry

end NUMINAMATH_GPT_find_a_l2302_230240


namespace NUMINAMATH_GPT_number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l2302_230248

theorem number_of_two_digit_factors_2_pow_18_minus_1_is_zero :
  (∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ n ∣ (2^18 - 1)) = false :=
by sorry

end NUMINAMATH_GPT_number_of_two_digit_factors_2_pow_18_minus_1_is_zero_l2302_230248


namespace NUMINAMATH_GPT_election_vote_percentage_l2302_230296

theorem election_vote_percentage 
  (total_students : ℕ)
  (winner_percentage : ℝ)
  (loser_percentage : ℝ)
  (vote_difference : ℝ)
  (P : ℝ)
  (H1 : total_students = 2000)
  (H2 : winner_percentage = 0.55)
  (H3 : loser_percentage = 0.45)
  (H4 : vote_difference = 50)
  (H5 : 0.1 * P * (total_students / 100) = vote_difference) :
  P = 25 := 
sorry

end NUMINAMATH_GPT_election_vote_percentage_l2302_230296


namespace NUMINAMATH_GPT_ramu_profit_percent_l2302_230211

noncomputable def profitPercent
  (purchase_price : ℝ)
  (repair_cost : ℝ)
  (selling_price : ℝ) : ℝ :=
  ((selling_price - (purchase_price + repair_cost)) / (purchase_price + repair_cost)) * 100

theorem ramu_profit_percent :
  profitPercent 42000 13000 61900 = 12.55 :=
by
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l2302_230211


namespace NUMINAMATH_GPT_quadratic_square_binomial_l2302_230290

theorem quadratic_square_binomial (a : ℝ) :
  (∃ d : ℝ, 9 * x ^ 2 - 18 * x + a = (3 * x + d) ^ 2) → a = 9 :=
by
  intro h
  match h with
  | ⟨d, h_eq⟩ => sorry

end NUMINAMATH_GPT_quadratic_square_binomial_l2302_230290


namespace NUMINAMATH_GPT_value_of_c_l2302_230200

theorem value_of_c (c : ℝ) : (∀ x : ℝ, x * (4 * x + 1) < c ↔ x > -5 / 2 ∧ x < 3) → c = 27 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_value_of_c_l2302_230200


namespace NUMINAMATH_GPT_min_value_expression_l2302_230231

theorem min_value_expression :
  ∃ x : ℝ, (x+2) * (x+3) * (x+5) * (x+6) + 2024 = 2021.75 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l2302_230231


namespace NUMINAMATH_GPT_polygon_sides_from_diagonals_l2302_230238

theorem polygon_sides_from_diagonals (D : ℕ) (hD : D = 16) : 
  ∃ n : ℕ, 2 * D = n * (n - 3) ∧ n = 7 :=
by
  use 7
  simp [hD]
  norm_num
  sorry

end NUMINAMATH_GPT_polygon_sides_from_diagonals_l2302_230238


namespace NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l2302_230225

-- Definitions representing the given conditions
def CagneyRate := 15 -- seconds per cupcake
def LaceyRate := 40 -- seconds per cupcake
def JessieRate := 30 -- seconds per cupcake
def initialDuration := 3 * 60 -- 3 minutes in seconds
def totalDuration := 10 * 60 -- 10 minutes in seconds
def afterJessieDuration := totalDuration - initialDuration -- 7 minutes in seconds

-- Proof statement
theorem cupcakes_frosted_in_10_minutes : 
  let combinedRateBefore := (CagneyRate * LaceyRate) / (CagneyRate + LaceyRate)
  let combinedRateAfter := (CagneyRate * LaceyRate * JessieRate) / (CagneyRate * LaceyRate + LaceyRate * JessieRate + JessieRate * CagneyRate)
  let cupcakesBefore := initialDuration / combinedRateBefore
  let cupcakesAfter := afterJessieDuration / combinedRateAfter
  cupcakesBefore + cupcakesAfter = 68 :=
by
  sorry

end NUMINAMATH_GPT_cupcakes_frosted_in_10_minutes_l2302_230225


namespace NUMINAMATH_GPT_equation_b_not_symmetric_about_x_axis_l2302_230213

def equationA (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equationB (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equationC (x y : ℝ) : Prop := 2 * x^2 - y^2 = 1
def equationD (x y : ℝ) : Prop := x + y^2 = -1

def symmetric_about_x_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, f x y ↔ f x (-y)

theorem equation_b_not_symmetric_about_x_axis : 
  ¬ symmetric_about_x_axis (equationB) :=
sorry

end NUMINAMATH_GPT_equation_b_not_symmetric_about_x_axis_l2302_230213


namespace NUMINAMATH_GPT_students_selected_juice_l2302_230254

def fraction_of_students_choosing_juice (students_selected_juice_ratio students_selected_soda_ratio : ℚ) : ℚ :=
  students_selected_juice_ratio / students_selected_soda_ratio

def num_students_selecting (students_selected_soda : ℕ) (fraction_juice : ℚ) : ℚ :=
  fraction_juice * students_selected_soda

theorem students_selected_juice (students_selected_soda : ℕ) : students_selected_soda = 120 ∧
    (fraction_of_students_choosing_juice 0.15 0.75) = 1/5 →
    num_students_selecting students_selected_soda (fraction_of_students_choosing_juice 0.15 0.75) = 24 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_students_selected_juice_l2302_230254


namespace NUMINAMATH_GPT_skittles_distribution_l2302_230212

theorem skittles_distribution :
  let initial_skittles := 14
  let additional_skittles := 22
  let total_skittles := initial_skittles + additional_skittles
  let number_of_people := 7
  (total_skittles / number_of_people = 5) :=
by
  sorry

end NUMINAMATH_GPT_skittles_distribution_l2302_230212


namespace NUMINAMATH_GPT_order_of_6_l2302_230279

def f (x : ℤ) : ℤ := (x^2) % 13

theorem order_of_6 :
  ∀ n : ℕ, (∀ k < n, f^[k] 6 ≠ 6) → f^[n] 6 = 6 → n = 72 :=
by
  sorry

end NUMINAMATH_GPT_order_of_6_l2302_230279


namespace NUMINAMATH_GPT_intersection_correct_union_correct_intersection_complement_correct_l2302_230258

def U := ℝ
def A : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def B : Set ℝ := {x | x < -3 ∨ x > 1}
def C_U_A : Set ℝ := {x | x ≤ 0 ∨ x > 2}
def C_U_B : Set ℝ := {x | -3 ≤ x ∧ x ≤ 1}

theorem intersection_correct : (A ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 2} :=
sorry

theorem union_correct : (A ∪ B) = {x : ℝ | x < -3 ∨ x > 0} :=
sorry

theorem intersection_complement_correct : (C_U_A ∩ C_U_B) = {x : ℝ | -3 ≤ x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_GPT_intersection_correct_union_correct_intersection_complement_correct_l2302_230258


namespace NUMINAMATH_GPT_total_turnips_l2302_230276

theorem total_turnips (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : 
  melanie_turnips + benny_turnips = 252 := 
by sorry

end NUMINAMATH_GPT_total_turnips_l2302_230276


namespace NUMINAMATH_GPT_solutions_exist_l2302_230221

theorem solutions_exist (k : ℤ) : ∃ x y : ℤ, (x = 3 * k + 2) ∧ (y = 7 * k + 4) ∧ (7 * x - 3 * y = 2) :=
by {
  -- Proof will be filled in here
  sorry
}

end NUMINAMATH_GPT_solutions_exist_l2302_230221


namespace NUMINAMATH_GPT_rocket_coaster_total_cars_l2302_230292

theorem rocket_coaster_total_cars (C_4 C_6 : ℕ) (h1 : C_4 = 9) (h2 : 4 * C_4 + 6 * C_6 = 72) :
  C_4 + C_6 = 15 :=
sorry

end NUMINAMATH_GPT_rocket_coaster_total_cars_l2302_230292


namespace NUMINAMATH_GPT_car_b_speed_l2302_230280

theorem car_b_speed (v : ℕ) (h1 : ∀ (v : ℕ), CarA_speed = 3 * v)
                   (h2 : ∀ (time : ℕ), CarA_time = 6)
                   (h3 : ∀ (time : ℕ), CarB_time = 2)
                   (h4 : Car_total_distance = 1000) :
    v = 50 :=
by
  sorry

end NUMINAMATH_GPT_car_b_speed_l2302_230280


namespace NUMINAMATH_GPT_goldfish_to_pretzels_ratio_l2302_230286

theorem goldfish_to_pretzels_ratio :
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := kids * items_per_baggie
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  ratio = 4 :=
by
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := 16 * 22 -- or kids * items_per_baggie for clarity
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  show ratio = 4
  · sorry

end NUMINAMATH_GPT_goldfish_to_pretzels_ratio_l2302_230286


namespace NUMINAMATH_GPT_product_value_l2302_230281

theorem product_value :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
    -- Skipping the actual proof
    sorry

end NUMINAMATH_GPT_product_value_l2302_230281


namespace NUMINAMATH_GPT_max_dominoes_l2302_230282

theorem max_dominoes (m n : ℕ) (h : n ≥ m) :
  ∃ k, k = m * n - (m / 2 : ℕ) :=
by sorry

end NUMINAMATH_GPT_max_dominoes_l2302_230282


namespace NUMINAMATH_GPT_line_intersects_y_axis_l2302_230264

-- Define the points
def P1 : ℝ × ℝ := (3, 18)
def P2 : ℝ × ℝ := (-9, -6)

-- State that the line passing through P1 and P2 intersects the y-axis at (0, 12)
theorem line_intersects_y_axis :
  ∃ y : ℝ, (∃ m b : ℝ, ∀ x : ℝ, y = m * x + b ∧ (m = (P2.2 - P1.2) / (P2.1 - P1.1)) ∧ (P1.2 = m * P1.1 + b) ∧ (x = 0) ∧ y = 12) :=
sorry

end NUMINAMATH_GPT_line_intersects_y_axis_l2302_230264
