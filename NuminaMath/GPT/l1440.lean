import Mathlib

namespace find_n_l1440_144020

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem find_n (n : ℤ) (h : ∃ x, n < x ∧ x < n+1 ∧ f x = 0) : n = 2 :=
sorry

end find_n_l1440_144020


namespace ABCD_area_is_correct_l1440_144043

-- Define rectangle ABCD with the given conditions
def ABCD_perimeter (x : ℝ) : Prop :=
  2 * (4 * x + x) = 160

-- Define the area to be proved
def ABCD_area (x : ℝ) : ℝ :=
  4 * (x ^ 2)

-- The proof problem: given the conditions, the area should be 1024 square centimeters
theorem ABCD_area_is_correct (x : ℝ) (h : ABCD_perimeter x) : 
  ABCD_area x = 1024 := 
by {
  sorry
}

end ABCD_area_is_correct_l1440_144043


namespace gain_percent_l1440_144015

-- Definitions for the problem
variables (MP CP SP : ℝ)
def cost_price := CP = 0.64 * MP
def selling_price := SP = 0.88 * MP

-- The statement to prove
theorem gain_percent (h1 : cost_price MP CP) (h2 : selling_price MP SP) :
  (SP - CP) / CP * 100 = 37.5 := 
sorry

end gain_percent_l1440_144015


namespace find_a_l1440_144078

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x + 1)

theorem find_a : 
  ( ∀ a : ℝ, 
    (∀ x : ℝ,  0 ≤ x ∧ x ≤ 1 → f a 0 + f a 1 = a) → a = 1/2 ) :=
sorry

end find_a_l1440_144078


namespace three_digit_number_count_l1440_144041

def total_three_digit_numbers : ℕ := 900

def count_ABA : ℕ := 9 * 9  -- 81

def count_ABC : ℕ := 9 * 9 * 8  -- 648

def valid_three_digit_numbers : ℕ := total_three_digit_numbers - (count_ABA + count_ABC)

theorem three_digit_number_count :
  valid_three_digit_numbers = 171 := by
  sorry

end three_digit_number_count_l1440_144041


namespace abs_gt_two_nec_but_not_suff_l1440_144063

theorem abs_gt_two_nec_but_not_suff (x : ℝ) : (|x| > 2 → x < -2) ∧ (¬ (|x| > 2 ↔ x < -2)) := 
sorry

end abs_gt_two_nec_but_not_suff_l1440_144063


namespace speed_of_faster_train_l1440_144049

noncomputable def speed_of_slower_train_kmph := 36
def time_to_cross_seconds := 12
def length_of_faster_train_meters := 120

-- Speed of train V_f in kmph 
theorem speed_of_faster_train 
  (relative_speed_mps : ℝ := length_of_faster_train_meters / time_to_cross_seconds)
  (speed_of_slower_train_mps : ℝ := speed_of_slower_train_kmph * (1000 / 3600))
  (speed_of_faster_train_mps : ℝ := relative_speed_mps + speed_of_slower_train_mps)
  (speed_of_faster_train_kmph : ℝ := speed_of_faster_train_mps * (3600 / 1000) )
  : speed_of_faster_train_kmph = 72 := 
sorry

end speed_of_faster_train_l1440_144049


namespace max_value_of_f_l1440_144068

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^4 + 2*x^2 + 3

-- State the theorem: the maximum value of f(x) is 4
theorem max_value_of_f : ∃ x : ℝ, f x = 4 := sorry

end max_value_of_f_l1440_144068


namespace find_c_and_d_l1440_144048

theorem find_c_and_d :
  ∀ (y c d : ℝ), (y^2 - 5 * y + 5 / y + 1 / (y^2) = 17) ∧ (y = c - Real.sqrt d) ∧ (0 < c) ∧ (0 < d) → (c + d = 106) :=
by
  intros y c d h
  sorry

end find_c_and_d_l1440_144048


namespace max_value_of_S_l1440_144016

-- Define the sequence sum function
def S (n : ℕ) : ℤ :=
  -2 * (n : ℤ) ^ 3 + 21 * (n : ℤ) ^ 2 + 23 * (n : ℤ)

theorem max_value_of_S :
  ∃ (n : ℕ), S n = 504 ∧ 
             (∀ k : ℕ, S k ≤ 504) :=
sorry

end max_value_of_S_l1440_144016


namespace fermats_little_theorem_for_q_plus_1_l1440_144080

theorem fermats_little_theorem_for_q_plus_1 (q : ℕ) (h1 : Nat.Prime q) (h2 : q % 2 = 1) :
  (q + 1)^(q - 1) % q = 1 := by
  sorry

end fermats_little_theorem_for_q_plus_1_l1440_144080


namespace larger_number_is_1590_l1440_144093

theorem larger_number_is_1590 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 :=
by
  sorry

end larger_number_is_1590_l1440_144093


namespace solution_exists_l1440_144027

theorem solution_exists (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 :=
by
  sorry

end solution_exists_l1440_144027


namespace median_squared_formula_l1440_144060

theorem median_squared_formula (a b c m : ℝ) (AC_is_median : 2 * m^2 + c^2 = a^2 + b^2) : 
  m^2 = (1/4) * (2 * a^2 + 2 * b^2 - c^2) := 
by
  sorry

end median_squared_formula_l1440_144060


namespace find_b_l1440_144094

theorem find_b (a b c d : ℝ) (h : ∃ k : ℝ, 2 * k = π ∧ k * (b / 2) = π) : b = 4 :=
by
  sorry

end find_b_l1440_144094


namespace bisections_needed_l1440_144083

theorem bisections_needed (ε : ℝ) (ε_pos : ε = 0.01) (h : 0 < ε) : 
  ∃ n : ℕ, n ≤ 7 ∧ 1 / (2^n) < ε :=
by
  sorry

end bisections_needed_l1440_144083


namespace solution_set_inequality_l1440_144001

variable (a b c : ℝ)
variable (condition1 : ∀ x : ℝ, ax^2 + bx + c < 0 ↔ x < -1 ∨ 2 < x)

theorem solution_set_inequality (h : a < 0 ∧ b = -a ∧ c = -2 * a) :
  ∀ x : ℝ, (bx^2 + ax - c ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by
  intro x
  sorry

end solution_set_inequality_l1440_144001


namespace arrange_athletes_l1440_144035

theorem arrange_athletes :
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  (Nat.choose athletes country_athletes) *
  (Nat.choose (athletes - country_athletes) country_athletes) *
  (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
  (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520 :=
by
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  show (Nat.choose athletes country_athletes) *
       (Nat.choose (athletes - country_athletes) country_athletes) *
       (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
       (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520
  sorry

end arrange_athletes_l1440_144035


namespace ink_cartridge_15th_month_l1440_144077

def months_in_year : ℕ := 12
def first_change_month : ℕ := 1   -- January is the first month

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + (3 * (n - 1))) % months_in_year

theorem ink_cartridge_15th_month : nth_change_month 15 = 7 := by
  -- This is where the proof would go
  sorry

end ink_cartridge_15th_month_l1440_144077


namespace correct_system_of_equations_l1440_144097

theorem correct_system_of_equations (x y : ℕ) : 
  (x / 3 = y - 2) ∧ ((x - 9) / 2 = y) ↔ 
  (x / 3 = y - 2) ∧ (x / 2 - 9 = y) := sorry

end correct_system_of_equations_l1440_144097


namespace average_speed_return_trip_l1440_144057

def speed1 : ℝ := 12 -- Speed for the first part of the trip in miles per hour
def distance1 : ℝ := 18 -- Distance for the first part of the trip in miles
def speed2 : ℝ := 10 -- Speed for the second part of the trip in miles per hour
def distance2 : ℝ := 18 -- Distance for the second part of the trip in miles
def total_round_trip_time : ℝ := 7.3 -- Total time for the round trip in hours

theorem average_speed_return_trip :
  let time1 := distance1 / speed1 -- Time taken for the first part of the trip
  let time2 := distance2 / speed2 -- Time taken for the second part of the trip
  let total_time_to_destination := time1 + time2 -- Total time for the trip to the destination
  let time_return_trip := total_round_trip_time - total_time_to_destination -- Time for the return trip
  let return_trip_distance := distance1 + distance2 -- Distance for the return trip (same as to the destination)
  let avg_speed_return_trip := return_trip_distance / time_return_trip -- Average speed for the return trip
  avg_speed_return_trip = 9 := 
by
  sorry

end average_speed_return_trip_l1440_144057


namespace other_factor_computation_l1440_144037

theorem other_factor_computation (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :
  a = 11 → b = 43 → c = 2 → d = 31 → e = 1311 → 33 ∣ 363 →
  a * b * c * d * e = 38428986 :=
by
  intros ha hb hc hd he hdiv
  rw [ha, hb, hc, hd, he]
  -- proof steps go here if required
  sorry

end other_factor_computation_l1440_144037


namespace stratified_sampling_third_year_students_l1440_144089

theorem stratified_sampling_third_year_students 
  (N : ℕ) (N_1 : ℕ) (P_sophomore : ℝ) (n : ℕ) (N_2 : ℕ) :
  N = 2000 →
  N_1 = 760 →
  P_sophomore = 0.37 →
  n = 20 →
  N_2 = Nat.ceil (N - N_1 - P_sophomore * N) →
  Nat.floor ((n : ℝ) / (N : ℝ) * (N_2 : ℝ)) = 5 :=
by
  sorry

end stratified_sampling_third_year_students_l1440_144089


namespace sqrt_fraction_arith_sqrt_16_l1440_144099

-- Prove that the square root of 4/9 is ±2/3
theorem sqrt_fraction (a b : ℕ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) (h_a : a = 4) (h_b : b = 9) : 
    (Real.sqrt (a / (b : ℝ)) = abs (Real.sqrt a / Real.sqrt b)) :=
by
    rw [h_a, h_b]
    sorry

-- Prove that the arithmetic square root of √16 is 4.
theorem arith_sqrt_16 : Real.sqrt (Real.sqrt 16) = 4 :=
by
    sorry

end sqrt_fraction_arith_sqrt_16_l1440_144099


namespace equivalent_shaded_areas_l1440_144039

/- 
  Definitions and parameters:
  - l_sq: the side length of the larger square.
  - s_sq: the side length of the smaller square.
-/
variables (l_sq s_sq : ℝ)
  
-- The area of the larger square
def area_larger_square : ℝ := l_sq * l_sq
  
-- The area of the smaller square
def area_smaller_square : ℝ := s_sq * s_sq
  
-- The shaded area in diagram i
def shaded_area_diagram_i : ℝ := area_larger_square l_sq - area_smaller_square s_sq

-- The polygonal areas in diagrams ii and iii
variables (polygon_area_ii polygon_area_iii : ℝ)

-- The theorem to prove the equivalence of the areas
theorem equivalent_shaded_areas :
  polygon_area_ii = shaded_area_diagram_i l_sq s_sq ∧ polygon_area_iii = shaded_area_diagram_i l_sq s_sq :=
sorry

end equivalent_shaded_areas_l1440_144039


namespace Jasmine_total_weight_in_pounds_l1440_144032

-- Definitions for the conditions provided
def weight_chips_ounces : ℕ := 20
def weight_cookies_ounces : ℕ := 9
def bags_chips : ℕ := 6
def tins_cookies : ℕ := 4 * bags_chips
def total_weight_ounces : ℕ := (weight_chips_ounces * bags_chips) + (weight_cookies_ounces * tins_cookies)
def total_weight_pounds : ℕ := total_weight_ounces / 16

-- The proof problem statement
theorem Jasmine_total_weight_in_pounds : total_weight_pounds = 21 := 
by
  sorry

end Jasmine_total_weight_in_pounds_l1440_144032


namespace find_constants_monotonicity_range_of_k_l1440_144009

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (b - 2 ^ x) / (2 ^ (x + 1) + a)

theorem find_constants (h_odd : ∀ x : ℝ, f x a b = - f (-x) a b) :
  a = 2 ∧ b = 1 :=
sorry

theorem monotonicity (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1) :
  ∀ x y : ℝ, x < y → f y a b ≤ f x a b :=
sorry

theorem range_of_k (a : ℝ) (b : ℝ) (h_constants : a = 2 ∧ b = 1)
  (h_pos : ∀ x : ℝ, x ≥ 1 → f (k * 3^x) a b + f (3^x - 9^x + 2) a b > 0) :
  k < 4 / 3 :=
sorry

end find_constants_monotonicity_range_of_k_l1440_144009


namespace find_M_value_l1440_144046

def distinct_positive_integers (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_M_value (C y M A : ℕ) 
  (h1 : distinct_positive_integers C y M A) 
  (h2 : C + y + 2 * M + A = 11) : M = 1 :=
sorry

end find_M_value_l1440_144046


namespace simplify_expression_l1440_144044

theorem simplify_expression (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 5) * (2 * x - 1) - 
  (2 * x - 1) * (x^2 + 2 * x - 8) + 
  (x^2 - 2 * x + 3) * (2 * x - 1) * (x - 2) = 
  8 * x^4 - 2 * x^3 - 5 * x^2 + 32 * x - 15 := 
  sorry

end simplify_expression_l1440_144044


namespace smallest_positive_n_l1440_144024

theorem smallest_positive_n (x y z : ℕ) (n : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x ∣ y^3) → (y ∣ z^3) → (z ∣ x^3) → (xyz ∣ (x + y + z)^13) :=
by
  sorry

end smallest_positive_n_l1440_144024


namespace percentage_of_girls_who_like_basketball_l1440_144029

theorem percentage_of_girls_who_like_basketball 
  (total_students : ℕ)
  (percentage_girls : ℝ)
  (percentage_boys_basketball : ℝ)
  (factor_girls_to_boys_not_basketball : ℝ)
  (total_students_eq : total_students = 25)
  (percentage_girls_eq : percentage_girls = 0.60)
  (percentage_boys_basketball_eq : percentage_boys_basketball = 0.40)
  (factor_girls_to_boys_not_basketball_eq : factor_girls_to_boys_not_basketball = 2) 
  : 
  ((factor_girls_to_boys_not_basketball * (total_students * (1 - percentage_girls) * (1 - percentage_boys_basketball))) / 
  (total_students * percentage_girls)) * 100 = 80 :=
by
  sorry

end percentage_of_girls_who_like_basketball_l1440_144029


namespace range_of_set_is_8_l1440_144004

theorem range_of_set_is_8 (a b c : ℕ) 
  (h1 : (a + b + c) / 3 = 6) 
  (h2 : b = 6) 
  (h3 : a = 2) 
  : max a (max b c) - min a (min b c) = 8 := 
by sorry

end range_of_set_is_8_l1440_144004


namespace water_usage_eq_13_l1440_144059

theorem water_usage_eq_13 (m x : ℝ) (h : 16 * m = 10 * m + (x - 10) * 2 * m) : x = 13 :=
by sorry

end water_usage_eq_13_l1440_144059


namespace Jeremy_songs_l1440_144000

theorem Jeremy_songs (songs_yesterday : ℕ) (songs_difference : ℕ) (songs_today : ℕ) (total_songs : ℕ) :
  songs_yesterday = 9 ∧ songs_difference = 5 ∧ songs_today = songs_yesterday + songs_difference ∧ 
  total_songs = songs_yesterday + songs_today → total_songs = 23 :=
by
  intros h
  sorry

end Jeremy_songs_l1440_144000


namespace john_bought_3_croissants_l1440_144011

variable (c k : ℕ)

theorem john_bought_3_croissants
  (h1 : c + k = 5)
  (h2 : ∃ n : ℕ, 88 * c + 44 * k = 100 * n) :
  c = 3 :=
by
-- Proof omitted
sorry

end john_bought_3_croissants_l1440_144011


namespace factorization_correct_l1440_144066

noncomputable def factor_expression (y : ℝ) : ℝ :=
  3 * y * (y - 5) + 4 * (y - 5)

theorem factorization_correct (y : ℝ) : factor_expression y = (3 * y + 4) * (y - 5) :=
by sorry

end factorization_correct_l1440_144066


namespace line_intersects_hyperbola_l1440_144065

variables (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0)

def line (x y : ℝ) := a * x - y + b = 0

def hyperbola (x y : ℝ) := x^2 / (|a| / |b|) - y^2 / (|b| / |a|) = 1

theorem line_intersects_hyperbola :
  ∃ x y : ℝ, line a b x y ∧ hyperbola a b x y := 
sorry

end line_intersects_hyperbola_l1440_144065


namespace probability_of_selecting_girl_l1440_144098

theorem probability_of_selecting_girl (boys girls : ℕ) (total_students : ℕ) (prob : ℚ) 
  (h1 : boys = 3) 
  (h2 : girls = 2) 
  (h3 : total_students = boys + girls) 
  (h4 : prob = girls / total_students) : 
  prob = 2 / 5 := 
sorry

end probability_of_selecting_girl_l1440_144098


namespace find_m_l1440_144082

noncomputable def hex_to_dec (m : ℕ) : ℕ :=
  3 * 6^4 + m * 6^3 + 5 * 6^2 + 2

theorem find_m (m : ℕ) : hex_to_dec m = 4934 ↔ m = 4 := 
by
  sorry

end find_m_l1440_144082


namespace nat_pairs_solution_l1440_144054

theorem nat_pairs_solution (x y : ℕ) :
  2^(2*x+1) + 2^x + 1 = y^2 → (x = 0 ∧ y = 2) ∨ (x = 4 ∧ y = 23) :=
by
  sorry

end nat_pairs_solution_l1440_144054


namespace infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l1440_144069

-- Problem 1: Infinitely many primes congruent to 3 modulo 4
theorem infinite_primes_congruent_3_mod_4 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 4 = 3) → ∃ q, Nat.Prime q ∧ q % 4 = 3 ∧ q ∉ ps :=
by
  sorry

-- Problem 2: Infinitely many primes congruent to 5 modulo 6
theorem infinite_primes_congruent_5_mod_6 :
  ∀ (ps : Finset ℕ), (∀ p ∈ ps, Nat.Prime p ∧ p % 6 = 5) → ∃ q, Nat.Prime q ∧ q % 6 = 5 ∧ q ∉ ps :=
by
  sorry

end infinite_primes_congruent_3_mod_4_infinite_primes_congruent_5_mod_6_l1440_144069


namespace solution_set_g_lt_6_range_of_values_a_l1440_144002

-- Definitions
def f (a x : ℝ) : ℝ := 3 * |x - a| + |3 * x + 1|
def g (x : ℝ) : ℝ := |4 * x - 1| - |x + 2|

-- First part: solution set for g(x) < 6
theorem solution_set_g_lt_6 :
  {x : ℝ | g x < 6} = {x : ℝ | -7/5 < x ∧ x < 3} :=
sorry

-- Second part: range of values for a such that f(x1) and g(x2) are opposite numbers
theorem range_of_values_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f a x1 = -g x2) → -13/12 ≤ a ∧ a ≤ 5/12 :=
sorry

end solution_set_g_lt_6_range_of_values_a_l1440_144002


namespace find_a5_l1440_144019

noncomputable def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
a₁ + (n - 1) * d

theorem find_a5 (a₁ d : ℚ) (h₁ : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 5 - arithmetic_sequence a₁ d 8 = 1)
(h₂ : arithmetic_sequence a₁ d 9 - arithmetic_sequence a₁ d 2 = 5) :
arithmetic_sequence a₁ d 5 = 6 :=
sorry

end find_a5_l1440_144019


namespace max_min_difference_abc_l1440_144006

theorem max_min_difference_abc (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
    let M := 1
    let m := -1/2
    M - m = 3/2 :=
by
  sorry

end max_min_difference_abc_l1440_144006


namespace max_of_four_expressions_l1440_144073

theorem max_of_four_expressions :
  996 * 996 > 995 * 997 ∧ 996 * 996 > 994 * 998 ∧ 996 * 996 > 993 * 999 :=
by
  sorry

end max_of_four_expressions_l1440_144073


namespace total_time_over_weekend_l1440_144021

def time_per_round : ℕ := 30
def rounds_saturday : ℕ := 11
def rounds_sunday : ℕ := 15

theorem total_time_over_weekend :
  (rounds_saturday * time_per_round) + (rounds_sunday * time_per_round) = 780 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end total_time_over_weekend_l1440_144021


namespace find_y_values_l1440_144052

theorem find_y_values (x : ℝ) (y : ℝ) 
  (h : x^2 + 4 * ((x / (x + 3))^2) = 64) : 
  y = (x + 3)^2 * (x - 2) / (2 * x + 3) → 
  y = 250 / 3 :=
sorry

end find_y_values_l1440_144052


namespace abc_def_ratio_l1440_144053

theorem abc_def_ratio (a b c d e f : ℝ)
    (h1 : a / b = 1 / 3)
    (h2 : b / c = 2)
    (h3 : c / d = 1 / 2)
    (h4 : d / e = 3)
    (h5 : e / f = 1 / 8) :
    (a * b * c) / (d * e * f) = 1 / 8 :=
by
  sorry

end abc_def_ratio_l1440_144053


namespace solve_arithmetic_sequence_l1440_144033

theorem solve_arithmetic_sequence :
  ∀ (x : ℝ), x > 0 ∧ x^2 = (2^2 + 5^2) / 2 → x = Real.sqrt (29 / 2) :=
by
  intro x
  intro hx
  sorry

end solve_arithmetic_sequence_l1440_144033


namespace west_of_1km_l1440_144071

def east_direction (d : Int) : Int :=
  d

def west_direction (d : Int) : Int :=
  -d

theorem west_of_1km :
  east_direction (2) = 2 →
  west_direction (1) = -1 := by
  sorry

end west_of_1km_l1440_144071


namespace interior_triangles_from_chords_l1440_144091

theorem interior_triangles_from_chords (h₁ : ∀ p₁ p₂ p₃ : Prop, ¬(p₁ ∧ p₂ ∧ p₃)) : 
  ∀ (nine_points_on_circle : Finset ℝ) (h₂ : nine_points_on_circle.card = 9), 
    ∃ (triangles : ℕ), triangles = 210 := 
by 
  sorry

end interior_triangles_from_chords_l1440_144091


namespace perpendicular_line_plane_l1440_144061

variables {m : ℝ}

theorem perpendicular_line_plane (h : (4 / 2) = (2 / 1) ∧ (2 / 1) = (m / -1)) : m = -2 :=
by
  sorry

end perpendicular_line_plane_l1440_144061


namespace smallest_triangle_perimeter_l1440_144034

theorem smallest_triangle_perimeter : ∃ (a b c : ℕ), a = 3 ∧ b = a + 1 ∧ c = b + 1 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 12 := by
  sorry

end smallest_triangle_perimeter_l1440_144034


namespace initial_players_round_robin_l1440_144051

-- Definitions of conditions
def num_matches_round_robin (x : ℕ) : ℕ := x * (x - 1) / 2
def num_matches_after_drop_out (x : ℕ) : ℕ := num_matches_round_robin x - 2 * (x - 4) + 1

-- The theorem statement
theorem initial_players_round_robin (x : ℕ) 
  (two_players_dropped : num_matches_after_drop_out x = 84) 
  (round_robin_condition : num_matches_round_robin x - 2 * (x - 4) + 1 = 84 ∨ num_matches_round_robin x - 2 * (x - 4) = 84) :
  x = 15 :=
sorry

end initial_players_round_robin_l1440_144051


namespace maria_total_distance_in_miles_l1440_144017

theorem maria_total_distance_in_miles :
  ∀ (steps_per_mile : ℕ) (full_cycles : ℕ) (remaining_steps : ℕ),
    steps_per_mile = 1500 →
    full_cycles = 50 →
    remaining_steps = 25000 →
    (100000 * full_cycles + remaining_steps) / steps_per_mile = 3350 := by
  intros
  sorry

end maria_total_distance_in_miles_l1440_144017


namespace find_investment_sum_l1440_144062

theorem find_investment_sum (P : ℝ)
  (h1 : SI_15 = P * (15 / 100) * 2)
  (h2 : SI_12 = P * (12 / 100) * 2)
  (h3 : SI_15 - SI_12 = 420) :
  P = 7000 :=
by
  sorry

end find_investment_sum_l1440_144062


namespace whiteboards_per_class_is_10_l1440_144087

-- Definitions from conditions
def classes : ℕ := 5
def ink_per_whiteboard_ml : ℕ := 20
def cost_per_ml_cents : ℕ := 50
def total_cost_cents : ℕ := 100 * 100  -- converting $100 to cents

-- Following the solution, define other useful constants
def cost_per_whiteboard_cents : ℕ := ink_per_whiteboard_ml * cost_per_ml_cents
def total_cost_all_classes_cents : ℕ := classes * total_cost_cents
def total_whiteboards : ℕ := total_cost_all_classes_cents / cost_per_whiteboard_cents
def whiteboards_per_class : ℕ := total_whiteboards / classes

-- We want to prove that each class uses 10 whiteboards.
theorem whiteboards_per_class_is_10 : whiteboards_per_class = 10 :=
  sorry

end whiteboards_per_class_is_10_l1440_144087


namespace cube_mono_increasing_l1440_144025

theorem cube_mono_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

end cube_mono_increasing_l1440_144025


namespace determine_common_ratio_l1440_144028

variable (a : ℕ → ℝ) (q : ℝ)

-- Given conditions
axiom a2 : a 2 = 1 / 2
axiom a5 : a 5 = 4
axiom geom_seq_def : ∀ n, a n = a 1 * q ^ (n - 1)

-- Prove the common ratio q == 2
theorem determine_common_ratio : q = 2 :=
by
  -- here we should unfold the proof steps given in the solution
  sorry

end determine_common_ratio_l1440_144028


namespace four_digit_number_l1440_144095

theorem four_digit_number : ∃ (a b c d : ℕ), 
  a + b + c + d = 16 ∧ 
  b + c = 10 ∧ 
  a - d = 2 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) % 9 = 0 ∧ 
  (10^3 * a + 10^2 * b + 10 * c + d) = 4622 :=
by
  sorry

end four_digit_number_l1440_144095


namespace smallest_w_l1440_144040

theorem smallest_w (w : ℕ) (h : 2^5 ∣ 936 * w ∧ 3^3 ∣ 936 * w ∧ 11^2 ∣ 936 * w) : w = 4356 :=
sorry

end smallest_w_l1440_144040


namespace part_I_part_II_l1440_144005

noncomputable def f (a x : ℝ) : ℝ := |a * x - 1| + |x + 2|

theorem part_I (h₁ : ∀ x : ℝ, f 1 x ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) : True :=
by sorry

theorem part_II (h₂ : ∃ a : ℝ, a > 0 ∧ (∀ x, f a x ≥ 2) ∧ (∀ b : ℝ, b > 0 ∧ (∀ x, f b x ≥ 2) → a ≤ b) ) : True :=
by sorry

end part_I_part_II_l1440_144005


namespace christine_sales_value_l1440_144055

variable {X : ℝ}

def commission_rate : ℝ := 0.12
def personal_needs_percent : ℝ := 0.60
def savings_amount : ℝ := 1152
def savings_percent : ℝ := 0.40

theorem christine_sales_value:
  (savings_percent * (commission_rate * X) = savings_amount) → 
  (X = 24000) := 
by
  intro h
  sorry

end christine_sales_value_l1440_144055


namespace yule_log_surface_area_increase_l1440_144030

noncomputable def yuleLogIncreaseSurfaceArea : ℝ := 
  let h := 10
  let d := 5
  let r := d / 2
  let n := 9
  let initialSurfaceArea := 2 * Real.pi * r * h + 2 * Real.pi * r^2
  let sliceHeight := h / n
  let sliceSurfaceArea := 2 * Real.pi * r * sliceHeight + 2 * Real.pi * r^2
  let totalSlicesSurfaceArea := n * sliceSurfaceArea
  let increaseSurfaceArea := totalSlicesSurfaceArea - initialSurfaceArea
  increaseSurfaceArea

theorem yule_log_surface_area_increase : yuleLogIncreaseSurfaceArea = 100 * Real.pi := by
  sorry

end yule_log_surface_area_increase_l1440_144030


namespace chocolate_game_winner_l1440_144075

theorem chocolate_game_winner (m n : ℕ) (h_m : m = 6) (h_n : n = 8) :
  (∃ k : ℕ, (48 - 1) - 2 * k = 0) ↔ true :=
by
  sorry

end chocolate_game_winner_l1440_144075


namespace school_students_l1440_144042

theorem school_students
  (total_students : ℕ)
  (students_in_both : ℕ)
  (students_chemistry : ℕ)
  (students_biology : ℕ)
  (students_only_chemistry : ℕ)
  (students_only_biology : ℕ)
  (h1 : total_students = students_only_chemistry + students_only_biology + students_in_both)
  (h2 : students_chemistry = 3 * students_biology)
  (students_in_both_eq : students_in_both = 5)
  (total_students_eq : total_students = 43) :
  students_only_chemistry + students_in_both = 36 :=
by
  sorry

end school_students_l1440_144042


namespace find_sum_of_abs_roots_l1440_144070

variable {p q r n : ℤ}

theorem find_sum_of_abs_roots (h1 : p + q + r = 0) (h2 : p * q + q * r + r * p = -2024) (h3 : p * q * r = -n) :
  |p| + |q| + |r| = 100 :=
  sorry

end find_sum_of_abs_roots_l1440_144070


namespace range_of_a_l1440_144092

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + Real.exp x - Real.exp (-x)

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l1440_144092


namespace eqn_abs_3x_minus_2_solution_l1440_144067

theorem eqn_abs_3x_minus_2_solution (x : ℝ) :
  (|x + 5| = 3 * x - 2) ↔ x = 7 / 2 :=
by
  sorry

end eqn_abs_3x_minus_2_solution_l1440_144067


namespace lion_weight_l1440_144031

theorem lion_weight :
  ∃ (L : ℝ), 
    (∃ (T P : ℝ), 
      L + T + P = 106.6 ∧ 
      P = T - 7.7 ∧ 
      T = L - 4.8) ∧ 
    L = 41.3 :=
by
  sorry

end lion_weight_l1440_144031


namespace largest_divisor_if_n_sq_div_72_l1440_144064

theorem largest_divisor_if_n_sq_div_72 (n : ℕ) (h : n > 0) (h72 : 72 ∣ n^2) : ∃ m, m = 12 ∧ m ∣ n :=
by { sorry }

end largest_divisor_if_n_sq_div_72_l1440_144064


namespace smallest_natural_number_l1440_144096

theorem smallest_natural_number :
  ∃ n : ℕ, (n > 0) ∧ (7 * n % 10000 = 2012) ∧ ∀ m : ℕ, (7 * m % 10000 = 2012) → (n ≤ m) :=
sorry

end smallest_natural_number_l1440_144096


namespace volume_of_four_cubes_l1440_144038

theorem volume_of_four_cubes (edge_length : ℕ) (num_cubes : ℕ) (h_edge : edge_length = 5) (h_num : num_cubes = 4) :
  num_cubes * (edge_length ^ 3) = 500 :=
by 
  sorry

end volume_of_four_cubes_l1440_144038


namespace find_ratio_of_hyperbola_l1440_144074

noncomputable def hyperbola (x y a b : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

theorem find_ratio_of_hyperbola (a b : ℝ) (h : a > b) 
  (h_asymptote_angle : ∀ α : ℝ, (y = ↑(b / a) * x -> α = 45)) :
  a / b = 1 :=
sorry

end find_ratio_of_hyperbola_l1440_144074


namespace kevin_started_with_cards_l1440_144085

-- The definitions corresponding to the conditions in the problem
def ended_with : Nat := 54
def found_cards : Nat := 47
def started_with (ended_with found_cards : Nat) : Nat := ended_with - found_cards

-- The Lean statement for the proof problem itself
theorem kevin_started_with_cards : started_with ended_with found_cards = 7 := by
  sorry

end kevin_started_with_cards_l1440_144085


namespace arccos_one_half_eq_pi_div_three_l1440_144050

theorem arccos_one_half_eq_pi_div_three : Real.arccos (1/2) = Real.pi / 3 :=
sorry

end arccos_one_half_eq_pi_div_three_l1440_144050


namespace min_value_product_expression_l1440_144013

theorem min_value_product_expression (x : ℝ) : ∃ m, m = -2746.25 ∧ (∀ y : ℝ, (13 - y) * (8 - y) * (13 + y) * (8 + y) ≥ m) :=
sorry

end min_value_product_expression_l1440_144013


namespace handshake_problem_l1440_144081

-- Define the remainder operation
def r_mod (n : ℕ) (k : ℕ) : ℕ := n % k

-- Define the function F
def F (t : ℕ) : ℕ := r_mod (t^3) 5251

-- The lean theorem statement with the given conditions and expected results
theorem handshake_problem :
  ∃ (x y : ℕ),
    F x = 506 ∧
    F (x + 1) = 519 ∧
    F y = 229 ∧
    F (y + 1) = 231 ∧
    x = 102 ∧
    y = 72 :=
by
  sorry

end handshake_problem_l1440_144081


namespace chicks_increased_l1440_144022

theorem chicks_increased (chicks_day1 chicks_day2: ℕ) (H1 : chicks_day1 = 23) (H2 : chicks_day2 = 12) : 
  chicks_day1 + chicks_day2 = 35 :=
by
  sorry

end chicks_increased_l1440_144022


namespace probability_of_specific_individual_drawn_on_third_attempt_l1440_144026

theorem probability_of_specific_individual_drawn_on_third_attempt :
  let population_size := 6
  let sample_size := 3
  let prob_not_drawn_first_attempt := 5 / 6
  let prob_not_drawn_second_attempt := 4 / 5
  let prob_drawn_third_attempt := 1 / 4
  (prob_not_drawn_first_attempt * prob_not_drawn_second_attempt * prob_drawn_third_attempt) = 1 / 6 :=
by sorry

end probability_of_specific_individual_drawn_on_third_attempt_l1440_144026


namespace powerjet_30_minutes_500_gallons_per_hour_l1440_144014

theorem powerjet_30_minutes_500_gallons_per_hour:
  ∀ (rate : ℝ) (time : ℝ), rate = 500 → time = 30 → (rate * (time / 60) = 250) := by
  intros rate time rate_eq time_eq
  sorry

end powerjet_30_minutes_500_gallons_per_hour_l1440_144014


namespace students_who_did_not_receive_an_A_l1440_144058

def total_students : ℕ := 40
def a_in_literature : ℕ := 10
def a_in_science : ℕ := 18
def a_in_both : ℕ := 6

theorem students_who_did_not_receive_an_A :
  total_students - ((a_in_literature + a_in_science) - a_in_both) = 18 :=
by
  sorry

end students_who_did_not_receive_an_A_l1440_144058


namespace final_output_M_l1440_144084

-- Definitions of the steps in the conditions
def initial_M : ℕ := 1
def increment_M1 (M : ℕ) : ℕ := M + 1
def increment_M2 (M : ℕ) : ℕ := M + 2

-- Define the final value of M after performing the operations
def final_M : ℕ := increment_M2 (increment_M1 initial_M)

-- The statement to prove
theorem final_output_M : final_M = 4 :=
by
  -- Placeholder for the actual proof
  sorry

end final_output_M_l1440_144084


namespace arithmetic_sequence_find_side_length_l1440_144010

variable (A B C a b c : ℝ)

-- Condition: Given that b(1 + cos(C)) = c(2 - cos(B))
variable (h : b * (1 + Real.cos C) = c * (2 - Real.cos B))

-- Question I: Prove that a + b = 2 * c
theorem arithmetic_sequence (h : b * (1 + Real.cos C) = c * (2 - Real.cos B)) : a + b = 2 * c :=
sorry

-- Additional conditions for Question II
variable (C_eq : C = Real.pi / 3)
variable (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3)

-- Question II: Find c
theorem find_side_length (C_eq : C = Real.pi / 3) (area : (1 / 2) * a * b * Real.sin C = 4 * Real.sqrt 3) : c = 4 :=
sorry

end arithmetic_sequence_find_side_length_l1440_144010


namespace correct_average_calculation_l1440_144047

theorem correct_average_calculation (n : ℕ) (incorrect_avg correct_num wrong_num : ℕ) (incorrect_avg_eq : incorrect_avg = 21) (n_eq : n = 10) (correct_num_eq : correct_num = 36) (wrong_num_eq : wrong_num = 26) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 22 := by
  sorry

end correct_average_calculation_l1440_144047


namespace meeting_time_l1440_144076

noncomputable def start_time : ℕ := 13 -- 1 pm in 24-hour format
noncomputable def speed_A : ℕ := 5 -- in kmph
noncomputable def speed_B : ℕ := 7 -- in kmph
noncomputable def initial_distance : ℕ := 24 -- in km

theorem meeting_time : start_time + (initial_distance / (speed_A + speed_B)) = 15 :=
by
  sorry

end meeting_time_l1440_144076


namespace gcd_1734_816_l1440_144090

theorem gcd_1734_816 : Nat.gcd 1734 816 = 102 := by
  sorry

end gcd_1734_816_l1440_144090


namespace function_d_is_odd_l1440_144079

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Given function
def f (x : ℝ) : ℝ := x^3

-- Proof statement
theorem function_d_is_odd : is_odd_function f := 
by sorry

end function_d_is_odd_l1440_144079


namespace square_perimeter_l1440_144018

theorem square_perimeter (s : ℝ) (h1 : (2 * (s + s / 4)) = 40) :
  4 * s = 64 :=
by
  sorry

end square_perimeter_l1440_144018


namespace solve_inequality_l1440_144008

variables (a b c x α β : ℝ)

theorem solve_inequality 
  (h1 : ∀ x, a * x^2 + b * x + c > 0 ↔ α < x ∧ x < β)
  (h2 : β > α)
  (ha : a < 0)
  (h3 : α + β = -b / a)
  (h4 : α * β = c / a) :
  ∀ x, (c * x^2 + b * x + a < 0 ↔ x < 1 / β ∨ x > 1 / α) := 
  by
    -- A detailed proof would follow here.
    sorry

end solve_inequality_l1440_144008


namespace permutations_of_BANANA_l1440_144045

theorem permutations_of_BANANA : 
  let word := ["B", "A", "N", "A", "N", "A"]
  let total_letters := 6
  let repeated_A := 3
  (total_letters.factorial / repeated_A.factorial) = 120 :=
by
  sorry

end permutations_of_BANANA_l1440_144045


namespace people_in_each_bus_l1440_144003

-- Definitions and conditions
def num_vans : ℕ := 2
def num_buses : ℕ := 3
def people_per_van : ℕ := 8
def total_people : ℕ := 76

-- Theorem statement to prove the number of people in each bus
theorem people_in_each_bus : (total_people - num_vans * people_per_van) / num_buses = 20 :=
by
    -- The actual proof would go here
    sorry

end people_in_each_bus_l1440_144003


namespace sum_of_squares_l1440_144072

variable {x y z a b c : Real}
variable (h₁ : x * y = a) (h₂ : x * z = b) (h₃ : y * z = c)
variable (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem sum_of_squares : x^2 + y^2 + z^2 = (a * b)^2 / (a * b * c) + (a * c)^2 / (a * b * c) + (b * c)^2 / (a * b * c) := 
sorry

end sum_of_squares_l1440_144072


namespace exists_unique_representation_l1440_144088

theorem exists_unique_representation (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y)^2 + 3 * x + y) / 2 :=
sorry

end exists_unique_representation_l1440_144088


namespace simplify_and_evaluate_expression_l1440_144012

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = 2):
  ( ( (2 * m + 1) / m - 1 ) / ( (m^2 - 1) / m ) ) = 1 :=
by
  rw [h] -- Replace m by 2
  sorry

end simplify_and_evaluate_expression_l1440_144012


namespace range_of_a_l1440_144056

theorem range_of_a (a : ℝ) :
  (0 + 0 + a) * (2 - 1 + a) < 0 ↔ (-1 < a ∧ a < 0) :=
by sorry

end range_of_a_l1440_144056


namespace ellipse_equation_l1440_144007

theorem ellipse_equation
  (P : ℝ × ℝ)
  (a b c : ℝ)
  (h1 : a > b ∧ b > 0)
  (h2 : 2 * a = 5 + 3)
  (h3 : (2 * c) ^ 2 = 5 ^ 2 - 3 ^ 2)
  (h4 : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1 ∨ P.2 ^ 2 / a ^ 2 + P.1 ^ 2 / b ^ 2 = 1)
  : ((a = 4) ∧ (c = 2) ∧ (b ^ 2 = 12) ∧
    (P.1 ^ 2 / 16 + P.2 ^ 2 / 12 = 1) ∨
    (P.2 ^ 2 / 16 + P.1 ^ 2 / 12 = 1)) :=
sorry

end ellipse_equation_l1440_144007


namespace sin_subtract_pi_over_6_l1440_144036

theorem sin_subtract_pi_over_6 (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) 
  (hcos : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end sin_subtract_pi_over_6_l1440_144036


namespace exists_multiple_of_10_of_three_distinct_integers_l1440_144086

theorem exists_multiple_of_10_of_three_distinct_integers
    (a b c : ℤ) 
    (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    ∃ x y : ℤ, (x = a ∨ x = b ∨ x = c) ∧ (y = a ∨ y = b ∨ y = c) ∧ x ≠ y ∧ (10 ∣ (x^5 * y^3 - x^3 * y^5)) :=
by
  sorry

end exists_multiple_of_10_of_three_distinct_integers_l1440_144086


namespace moving_circle_passes_through_focus_l1440_144023

-- Given conditions
def is_on_parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def is_tangent_to_line (circle_center_x : ℝ) : Prop :=
  circle_center_x + 2 = 0

-- Prove that the point (2,0) lies on the moving circle
theorem moving_circle_passes_through_focus (circle_center_x circle_center_y : ℝ) :
  is_on_parabola circle_center_x circle_center_y →
  is_tangent_to_line circle_center_x →
  (circle_center_x - 2)^2 + circle_center_y^2 = (circle_center_x + 2)^2 :=
by
  -- Proof skipped with sorry.
  sorry

end moving_circle_passes_through_focus_l1440_144023
