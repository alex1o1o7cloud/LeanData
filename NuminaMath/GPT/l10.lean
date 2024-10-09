import Mathlib

namespace tickets_won_in_skee_ball_l10_1038

-- Define the conditions as Lean definitions
def tickets_from_whack_a_mole : ℕ := 8
def ticket_cost_per_candy : ℕ := 5
def candies_bought : ℕ := 3

-- We now state the conjecture (mathematical proof problem) 
-- Prove that the number of tickets won in skee ball is 7.
theorem tickets_won_in_skee_ball :
  (candies_bought * ticket_cost_per_candy) - tickets_from_whack_a_mole = 7 :=
by
  sorry

end tickets_won_in_skee_ball_l10_1038


namespace daily_rate_problem_l10_1053

noncomputable def daily_rate : ℝ := 126.19 -- Correct answer

theorem daily_rate_problem
  (days : ℕ := 14)
  (pet_fee : ℝ := 100)
  (service_fee_rate : ℝ := 0.20)
  (security_deposit : ℝ := 1110)
  (deposit_rate : ℝ := 0.50)
  (x : ℝ) : x = daily_rate :=
by
  have total_cost := days * x + pet_fee + service_fee_rate * (days * x)
  have total_cost_with_fees := days * x * (1 + service_fee_rate) + pet_fee
  have security_deposit_cost := deposit_rate * total_cost_with_fees
  have eq_security : security_deposit_cost = security_deposit := sorry
  sorry

end daily_rate_problem_l10_1053


namespace area_dodecagon_equals_rectangle_l10_1004

noncomputable def area_regular_dodecagon (r : ℝ) : ℝ := 3 * r^2

theorem area_dodecagon_equals_rectangle (r : ℝ) :
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  area_dodecagon = area_rectangle :=
by
  let area_dodecagon := area_regular_dodecagon r
  let area_rectangle := r * (3 * r)
  show area_dodecagon = area_rectangle
  sorry

end area_dodecagon_equals_rectangle_l10_1004


namespace possible_remainders_of_a2_l10_1016

theorem possible_remainders_of_a2 (p : ℕ) (k : ℕ) (hp : Nat.Prime p) (hk : 0 < k) 
  (hresidue : ∀ i : ℕ, i < p → ∃ j : ℕ, j < p ∧ ((j^k+j) % p = i)) :
  ∃ s : Finset ℕ, s = Finset.range p ∧ (2^k + 2) % p ∈ s := 
sorry

end possible_remainders_of_a2_l10_1016


namespace largest_x_to_floor_ratio_l10_1080

theorem largest_x_to_floor_ratio : ∃ x : ℝ, (⌊x⌋ / x = 9 / 10 ∧ ∀ y : ℝ, (⌊y⌋ / y = 9 / 10 → y ≤ x)) :=
sorry

end largest_x_to_floor_ratio_l10_1080


namespace neg_p_sufficient_not_necessary_q_l10_1022

theorem neg_p_sufficient_not_necessary_q (p q : Prop) 
  (h₁ : p → ¬q) 
  (h₂ : ¬(¬q → p)) : (q → ¬p) ∧ ¬(¬p → q) :=
sorry

end neg_p_sufficient_not_necessary_q_l10_1022


namespace beatrice_tv_ratio_l10_1059

theorem beatrice_tv_ratio (T1 T2 T Ttotal : ℕ)
  (h1 : T1 = 8)
  (h2 : T2 = 10)
  (h_total : Ttotal = 42)
  (h_T : T = Ttotal - T1 - T2) :
  (T / gcd T T1, T1 / gcd T T1) = (3, 1) :=
by {
  sorry
}

end beatrice_tv_ratio_l10_1059


namespace fraction_is_one_third_l10_1020

theorem fraction_is_one_third :
  (3 + 9 - 27 + 81 + 243 - 729) / (9 + 27 - 81 + 243 + 729 - 2187) = 1 / 3 :=
by
  sorry

end fraction_is_one_third_l10_1020


namespace meal_cost_is_25_l10_1023

def total_cost_samosas : ℕ := 3 * 2
def total_cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2
def tip_percentage : ℝ := 0.25

def total_food_cost : ℕ := total_cost_samosas + total_cost_pakoras + cost_mango_lassi
def tip_amount : ℝ := total_food_cost * tip_percentage
def total_meal_cost : ℝ := total_food_cost + tip_amount

theorem meal_cost_is_25 : total_meal_cost = 25 := by
    sorry

end meal_cost_is_25_l10_1023


namespace percentage_of_life_in_accounting_jobs_l10_1034

-- Define the conditions
def years_as_accountant : ℕ := 25
def years_as_manager : ℕ := 15
def lifespan : ℕ := 80

-- Define the proof problem statement
theorem percentage_of_life_in_accounting_jobs :
  (years_as_accountant + years_as_manager) / lifespan * 100 = 50 := 
by sorry

end percentage_of_life_in_accounting_jobs_l10_1034


namespace gcf_3150_7350_l10_1002

theorem gcf_3150_7350 : Nat.gcd 3150 7350 = 525 := by
  sorry

end gcf_3150_7350_l10_1002


namespace john_weekly_earnings_increase_l10_1070

theorem john_weekly_earnings_increase :
  let earnings_before := 60 + 100
  let earnings_after := 78 + 120
  let increase := earnings_after - earnings_before
  (increase / earnings_before : ℚ) * 100 = 23.75 :=
by
  -- Definitions
  let earnings_before := (60 : ℚ) + 100
  let earnings_after := (78 : ℚ) + 120
  let increase := earnings_after - earnings_before

  -- Calculation of percentage increase
  let percentage_increase : ℚ := (increase / earnings_before) * 100

  -- Expected result
  have expected_result : percentage_increase = 23.75 := by sorry
  exact expected_result

end john_weekly_earnings_increase_l10_1070


namespace quadratic_sum_eq_504_l10_1009

theorem quadratic_sum_eq_504 :
  ∃ (a b c : ℝ), (∀ x : ℝ, 20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧ a + b + c = 504 :=
by sorry

end quadratic_sum_eq_504_l10_1009


namespace medium_kite_area_l10_1089

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

end medium_kite_area_l10_1089


namespace difference_between_c_and_a_l10_1011

variables (a b c : ℝ)

theorem difference_between_c_and_a
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

end difference_between_c_and_a_l10_1011


namespace how_much_milk_did_joey_drink_l10_1041

theorem how_much_milk_did_joey_drink (initial_milk : ℚ) (rachel_fraction : ℚ) (joey_fraction : ℚ) :
  initial_milk = 3/4 →
  rachel_fraction = 1/3 →
  joey_fraction = 1/2 →
  joey_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/4 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  sorry

end how_much_milk_did_joey_drink_l10_1041


namespace order_of_6_l10_1096

def f (x : ℤ) : ℤ := (x^2) % 13

theorem order_of_6 :
  ∀ n : ℕ, (∀ k < n, f^[k] 6 ≠ 6) → f^[n] 6 = 6 → n = 72 :=
by
  sorry

end order_of_6_l10_1096


namespace midpoint_x_sum_l10_1008

variable {p q r s : ℝ}

theorem midpoint_x_sum (h : p + q + r + s = 20) :
  ((p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2) = 20 :=
by
  sorry

end midpoint_x_sum_l10_1008


namespace contrapositive_l10_1074

-- Definitions based on the conditions
def original_proposition (a b : ℝ) : Prop := a^2 + b^2 = 0 → a = 0 ∧ b = 0

-- The theorem to prove the contrapositive
theorem contrapositive (a b : ℝ) : original_proposition a b ↔ (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) :=
sorry

end contrapositive_l10_1074


namespace first_term_of_geometric_series_l10_1058

variable (a r : ℝ)
variable (h1 : a / (1 - r) = 20)
variable (h2 : a^2 / (1 - r^2) = 80)

theorem first_term_of_geometric_series (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : 
  a = 20 / 3 :=
  sorry

end first_term_of_geometric_series_l10_1058


namespace discounted_price_l10_1042

theorem discounted_price (P : ℝ) (original_price : ℝ) (discount_rate : ℝ)
  (h1 : original_price = 975)
  (h2 : discount_rate = 0.20)
  (h3 : P = original_price - discount_rate * original_price) : 
  P = 780 := 
by
  sorry

end discounted_price_l10_1042


namespace percentile_75_eq_95_l10_1081

def seventy_fifth_percentile (data : List ℕ) : ℕ := sorry

theorem percentile_75_eq_95 : seventy_fifth_percentile [92, 93, 88, 99, 89, 95] = 95 := 
sorry

end percentile_75_eq_95_l10_1081


namespace polygon_diagonals_formula_l10_1094

theorem polygon_diagonals_formula (n : ℕ) (h₁ : n = 5) (h₂ : 2 * n = (n * (n - 3)) / 2) :
  ∃ D : ℕ, D = n * (n - 3) / 2 :=
by
  sorry

end polygon_diagonals_formula_l10_1094


namespace roots_odd_even_l10_1082

theorem roots_odd_even (n : ℤ) (x1 x2 : ℤ) (h_eqn : x1^2 + (4 * n + 1) * x1 + 2 * n = 0) (h_eqn' : x2^2 + (4 * n + 1) * x2 + 2 * n = 0) :
  ((x1 % 2 = 0 ∧ x2 % 2 ≠ 0) ∨ (x1 % 2 ≠ 0 ∧ x2 % 2 = 0)) :=
sorry

end roots_odd_even_l10_1082


namespace Alyssa_puppies_l10_1062

theorem Alyssa_puppies (initial_puppies : ℕ) (given_puppies : ℕ)
  (h_initial : initial_puppies = 7) (h_given : given_puppies = 5) :
  initial_puppies - given_puppies = 2 :=
by
  sorry

end Alyssa_puppies_l10_1062


namespace horner_eval_at_minus_point_two_l10_1065

def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_eval_at_minus_point_two :
  f (-0.2) = 0.81873 :=
by 
  sorry

end horner_eval_at_minus_point_two_l10_1065


namespace cricket_player_average_increase_l10_1018

theorem cricket_player_average_increase (total_innings initial_innings next_run : ℕ) (initial_average desired_increase : ℕ) 
(h1 : initial_innings = 10) (h2 : initial_average = 32) (h3 : next_run = 76) : desired_increase = 4 :=
by
  sorry

end cricket_player_average_increase_l10_1018


namespace area_of_triangle_ABC_l10_1046

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1424233, 2848467)
def C : point := (1424234, 2848469)

def triangle_area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem area_of_triangle_ABC : triangle_area A B C = 0.50 := by
  sorry

end area_of_triangle_ABC_l10_1046


namespace find_y_l10_1087

theorem find_y (y : ℝ) (h : (15 + 28 + y) / 3 = 25) : y = 32 := by
  sorry

end find_y_l10_1087


namespace length_of_train_l10_1054

-- Definitions for the given conditions:
def speed : ℝ := 60   -- in kmph
def time : ℝ := 20    -- in seconds
def platform_length : ℝ := 213.36  -- in meters

-- Conversion factor from km/h to m/s
noncomputable def kmph_to_mps (kmph : ℝ) : ℝ := kmph * (1000 / 3600)

-- Total distance covered by train while crossing the platform
noncomputable def total_distance (speed_in_kmph : ℝ) (time_in_seconds : ℝ) : ℝ := 
  (kmph_to_mps speed_in_kmph) * time_in_seconds

-- Length of the train
noncomputable def train_length (total_distance_covered : ℝ) (platform_len : ℝ) : ℝ :=
  total_distance_covered - platform_len

-- Expected length of the train
def expected_train_length : ℝ := 120.04

-- Theorem to prove the length of the train given the conditions
theorem length_of_train : 
  train_length (total_distance speed time) platform_length = expected_train_length :=
by 
  sorry

end length_of_train_l10_1054


namespace organic_fertilizer_prices_l10_1097

theorem organic_fertilizer_prices
  (x y : ℝ)
  (h1 : x - y = 100)
  (h2 : 2 * x + y = 1700) :
  x = 600 ∧ y = 500 :=
by {
  sorry
}

end organic_fertilizer_prices_l10_1097


namespace fill_parentheses_l10_1069

variable (a b : ℝ)

theorem fill_parentheses :
  1 - a^2 + 2 * a * b - b^2 = 1 - (a^2 - 2 * a * b + b^2) :=
by
  sorry

end fill_parentheses_l10_1069


namespace rocket_coaster_total_cars_l10_1077

theorem rocket_coaster_total_cars (C_4 C_6 : ℕ) (h1 : C_4 = 9) (h2 : 4 * C_4 + 6 * C_6 = 72) :
  C_4 + C_6 = 15 :=
sorry

end rocket_coaster_total_cars_l10_1077


namespace find_n_l10_1078

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

end find_n_l10_1078


namespace probability_x_lt_2y_is_2_over_5_l10_1013

noncomputable def rectangle_area : ℝ :=
  5 * 2

noncomputable def triangle_area : ℝ :=
  1 / 2 * 4 * 2

noncomputable def probability_x_lt_2y : ℝ :=
  triangle_area / rectangle_area

theorem probability_x_lt_2y_is_2_over_5 :
  probability_x_lt_2y = 2 / 5 := by
  sorry

end probability_x_lt_2y_is_2_over_5_l10_1013


namespace third_generation_tail_length_l10_1032

theorem third_generation_tail_length (tail_length : ℕ → ℕ) (h0 : tail_length 0 = 16)
    (h_next : ∀ n, tail_length (n + 1) = tail_length n + (25 * tail_length n) / 100) :
    tail_length 2 = 25 :=
by
  sorry

end third_generation_tail_length_l10_1032


namespace problem_statement_l10_1093

noncomputable def given_function (x : ℝ) : ℝ := Real.sin (Real.pi / 2 - 2 * x)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) : Prop :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem problem_statement :
  is_even_function given_function ∧ smallest_positive_period given_function Real.pi :=
by
  sorry

end problem_statement_l10_1093


namespace find_quadratic_eq_l10_1033

theorem find_quadratic_eq (x y : ℝ) (hx : x + y = 10) (hy : |x - y| = 12) :
    ∃ a b c : ℝ, a = 1 ∧ b = -10 ∧ c = -11 ∧ (x^2 + b * x + c = 0) ∧ (y^2 + b * y + c = 0) := by
  sorry

end find_quadratic_eq_l10_1033


namespace purely_imaginary_a_eq_1_fourth_quadrant_a_range_l10_1045

-- Definitions based on given conditions
def z (a : ℝ) := (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I

-- Purely imaginary proof statement
theorem purely_imaginary_a_eq_1 (a : ℝ) 
  (hz : (a^2 - 7 * a + 6) + (a^2 - 5 * a - 6) * Complex.I = (0 : ℂ) + (a^2 - 5 * a - 6) * Complex.I) :
  a = 1 := by 
  sorry

-- Fourth quadrant proof statement
theorem fourth_quadrant_a_range (a : ℝ) 
  (hz1 : a^2 - 7 * a + 6 > 0) 
  (hz2 : a^2 - 5 * a - 6 < 0) : 
  -1 < a ∧ a < 1 := by 
  sorry

end purely_imaginary_a_eq_1_fourth_quadrant_a_range_l10_1045


namespace factorization_example_l10_1047

theorem factorization_example :
  (4 : ℤ) * x^2 - 1 = (2 * x + 1) * (2 * x - 1) := 
by
  sorry

end factorization_example_l10_1047


namespace inequality_proof_l10_1049

theorem inequality_proof
  (a b c : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : c^2 + a * b = a^2 + b^2) :
  c^2 + a * b ≤ a * c + b * c := sorry

end inequality_proof_l10_1049


namespace sum_of_roots_l10_1075

theorem sum_of_roots (a b c d : ℝ) (h : ∀ x : ℝ, 
  a * (x ^ 3 - x) ^ 3 + b * (x ^ 3 - x) ^ 2 + c * (x ^ 3 - x) + d 
  ≥ a * (x ^ 2 + x + 1) ^ 3 + b * (x ^ 2 + x + 1) ^ 2 + c * (x ^ 2 + x + 1) + d) :
  b / a = -6 :=
sorry

end sum_of_roots_l10_1075


namespace goldfish_to_pretzels_ratio_l10_1083

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

end goldfish_to_pretzels_ratio_l10_1083


namespace find_x_l10_1030

/-- 
Prove that the value of x is 25 degrees, given the following conditions:
1. The sum of the angles in triangle BAC: angle_BAC + 50° + 55° = 180°
2. The angles forming a straight line DAE: 80° + angle_BAC + x = 180°
-/
theorem find_x (angle_BAC : ℝ) (x : ℝ)
  (h1 : angle_BAC + 50 + 55 = 180)
  (h2 : 80 + angle_BAC + x = 180) :
  x = 25 :=
  sorry

end find_x_l10_1030


namespace car_b_speed_l10_1091

theorem car_b_speed (v : ℕ) (h1 : ∀ (v : ℕ), CarA_speed = 3 * v)
                   (h2 : ∀ (time : ℕ), CarA_time = 6)
                   (h3 : ∀ (time : ℕ), CarB_time = 2)
                   (h4 : Car_total_distance = 1000) :
    v = 50 :=
by
  sorry

end car_b_speed_l10_1091


namespace proportion_of_segments_l10_1048

theorem proportion_of_segments
  (a b c d : ℝ)
  (h1 : b = 3)
  (h2 : c = 4)
  (h3 : d = 6)
  (h4 : a / b = c / d) :
  a = 2 :=
by
  sorry

end proportion_of_segments_l10_1048


namespace evaluate_expression_l10_1028

noncomputable def a := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def b := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def c := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2
noncomputable def d := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2

theorem evaluate_expression : (1 / a + 1 / b + 1 / c + 1 / d)^2 = 39 / 140 := 
by
  sorry

end evaluate_expression_l10_1028


namespace not_integer_fraction_l10_1027

theorem not_integer_fraction (a b : ℕ) (h1 : a > b) (h2 : b > 2) : ¬ (∃ k : ℤ, (2^a + 1) = k * (2^b - 1)) :=
sorry

end not_integer_fraction_l10_1027


namespace people_dislike_both_radio_and_music_l10_1035

theorem people_dislike_both_radio_and_music (N : ℕ) (p_r p_rm : ℝ) (hN : N = 2000) (hp_r : p_r = 0.25) (hp_rm : p_rm = 0.15) : 
  N * p_r * p_rm = 75 :=
by {
  sorry
}

end people_dislike_both_radio_and_music_l10_1035


namespace relationship_between_mode_median_mean_l10_1095

def data_set : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode : ℕ := 60 -- derived from the problem conditions
def median : ℕ := 50 -- derived from the problem conditions
def mean : ℚ := 330 / 7 -- derived from the problem conditions

theorem relationship_between_mode_median_mean :
  mode > median ∧ median > mean :=
by
  sorry

end relationship_between_mode_median_mean_l10_1095


namespace geometric_sequence_a2_value_l10_1010

theorem geometric_sequence_a2_value
  (a : ℕ → ℝ)
  (a1 a2 a3 : ℝ)
  (h1 : a 1 = a1)
  (h2 : a 2 = a2)
  (h3 : a 3 = a3)
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, a (n + 1) = a 1 * (a 2) ^ n)
  (h_sum : a1 + a2 + a3 = 18)
  (h_inverse_sum : 1/a1 + 1/a2 + 1/a3 = 2)
  : a2 = 3 :=
sorry

end geometric_sequence_a2_value_l10_1010


namespace problem_1_problem_2_l10_1012

def setA (x : ℝ) : Prop := 2 ≤ x ∧ x < 7
def setB (x : ℝ) : Prop := 3 < x ∧ x ≤ 10
def setC (a : ℝ) (x : ℝ) : Prop := a - 5 < x ∧ x < a

theorem problem_1 (x : ℝ) :
  (setA x ∧ setB x ↔ 3 < x ∧ x < 7) ∧
  (setA x ∨ setB x ↔ 2 ≤ x ∧ x ≤ 10) := 
by sorry

theorem problem_2 (a : ℝ) :
  (∀ x, setC a x → (2 ≤ x ∧ x ≤ 10)) ↔ (7 ≤ a ∧ a ≤ 10) :=
by sorry

end problem_1_problem_2_l10_1012


namespace regular_polygon_sides_l10_1052

theorem regular_polygon_sides (h : ∀ n : ℕ, (120 * n) = 180 * (n - 2)) : 6 = 6 :=
by
  sorry

end regular_polygon_sides_l10_1052


namespace greatest_num_of_coins_l10_1086

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

end greatest_num_of_coins_l10_1086


namespace find_number_l10_1084

theorem find_number (x : ℕ) (h : 695 - 329 = x - 254) : x = 620 :=
sorry

end find_number_l10_1084


namespace ratio_of_customers_third_week_l10_1073

def ratio_of_customers (c1 c3 : ℕ) (s k t : ℕ) : Prop := s = 500 ∧ k = 50 ∧ t = 760 ∧ c1 = 35 ∧ c3 = 105 ∧ (t - s - k) - (35 + 70) = c1 ∧ c3 = 105 ∧ (c3 / c1 = 3)

theorem ratio_of_customers_third_week (c1 c3 : ℕ) (s k t : ℕ)
  (h1 : s = 500)
  (h2 : k = 50)
  (h3 : t = 760)
  (h4 : c1 = 35)
  (h5 : c3 = 105)
  (h6 : (t - s - k) - (35 + 70) = c1)
  (h7 : c3 = 105) :
  (c3 / c1) = 3 :=
  sorry

end ratio_of_customers_third_week_l10_1073


namespace union_complement_l10_1076

universe u

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1}

theorem union_complement (U A B : Set ℕ) (hU : U = {0, 2, 4, 6, 8, 10}) (hA : A = {2, 4, 6}) (hB : B = {1}) :
  (U \ A) ∪ B = {0, 1, 8, 10} :=
by
  -- The proof is omitted.
  sorry

end union_complement_l10_1076


namespace semicircle_parametric_equation_correct_l10_1007

-- Define the conditions of the problem in terms of Lean definitions and propositions.

def semicircle_parametric_equation : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ (Real.pi / 2) →
    ∃ α : ℝ, α = 2 * θ ∧ 0 ≤ α ∧ α ≤ Real.pi ∧
    (∃ (x y : ℝ), x = 1 + Real.cos α ∧ y = Real.sin α)

-- Statement that we will prove
theorem semicircle_parametric_equation_correct : semicircle_parametric_equation :=
  sorry

end semicircle_parametric_equation_correct_l10_1007


namespace equation_of_hyperbola_l10_1061

variable (a b c : ℝ)
variable (x y : ℝ)

theorem equation_of_hyperbola :
  (0 < a) ∧ (0 < b) ∧ (c / a = Real.sqrt 3) ∧ (a^2 / c = 1) ∧ (c = 3) ∧ (b = Real.sqrt 6)
  → (x^2 / 3 - y^2 / 6 = 1) :=
by
  sorry

end equation_of_hyperbola_l10_1061


namespace find_denominator_x_l10_1037

noncomputable def sum_fractions : ℝ := 
    3.0035428163476343

noncomputable def fraction1 (x : ℝ) : ℝ :=
    2007 / x

noncomputable def fraction2 : ℝ :=
    8001 / 5998

noncomputable def fraction3 : ℝ :=
    2001 / 3999

-- Problem statement in Lean
theorem find_denominator_x (x : ℝ) :
  sum_fractions = fraction1 x + fraction2 + fraction3 ↔ x = 1717 :=
by sorry

end find_denominator_x_l10_1037


namespace incorrect_transformation_is_not_valid_l10_1015

-- Define the system of linear equations
def eq1 (x y : ℝ) := 2 * x + y = 5
def eq2 (x y : ℝ) := 3 * x + 4 * y = 7

-- The definition of the correct transformation for x from equation eq2
def correct_transformation (x y : ℝ) := x = (7 - 4 * y) / 3

-- The definition of the incorrect transformation for x from equation eq2
def incorrect_transformation (x y : ℝ) := x = (7 + 4 * y) / 3

theorem incorrect_transformation_is_not_valid (x y : ℝ) 
  (h1 : eq1 x y) 
  (h2 : eq2 x y) :
  ¬ incorrect_transformation x y := 
by
  sorry

end incorrect_transformation_is_not_valid_l10_1015


namespace ken_situps_l10_1098

variable (K : ℕ)

theorem ken_situps (h1 : Nathan = 2 * K)
                   (h2 : Bob = 3 * K / 2)
                   (h3 : Bob = K + 10) : 
                   K = 20 := 
by
  sorry

end ken_situps_l10_1098


namespace log_base_function_inequalities_l10_1039

/-- 
Given the function y = log_(1/(sqrt(2))) (1/(x + 3)),
prove that:
1. for y > 0, x ∈ (-2, +∞)
2. for y < 0, x ∈ (-3, -2)
-/
theorem log_base_function_inequalities :
  let y (x : ℝ) := Real.logb (1 / Real.sqrt 2) (1 / (x + 3))
  ∀ x : ℝ, (y x > 0 ↔ x > -2) ∧ (y x < 0 ↔ -3 < x ∧ x < -2) :=
by
  intros
  -- Proof steps would go here
  sorry

end log_base_function_inequalities_l10_1039


namespace greatest_overlap_l10_1099

-- Defining the conditions based on the problem statement
def percentage_internet (n : ℕ) : Prop := n = 35
def percentage_snacks (m : ℕ) : Prop := m = 70

-- The theorem to prove the greatest possible overlap
theorem greatest_overlap (n m k : ℕ) (hn : percentage_internet n) (hm : percentage_snacks m) : 
  k ≤ 35 :=
by sorry

end greatest_overlap_l10_1099


namespace axis_of_symmetry_and_vertex_l10_1067

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 1

theorem axis_of_symmetry_and_vertex :
  (∃ (a : ℝ), (f a = -2 * (a - 1)^2 + 3) ∧ a = 1) ∧ ∃ v, (v = (1, 3) ∧ ∀ x, f x = -2 * (x - 1)^2 + 3) :=
sorry

end axis_of_symmetry_and_vertex_l10_1067


namespace solve_for_x_l10_1024

variable (x : ℝ)

-- Define the condition: 20% of x = 300
def twenty_percent_eq_300 := (0.20 * x = 300)

-- Define the goal: 120% of x = 1800
def one_twenty_percent_eq_1800 := (1.20 * x = 1800)

theorem solve_for_x (h : twenty_percent_eq_300 x) : one_twenty_percent_eq_1800 x :=
sorry

end solve_for_x_l10_1024


namespace eval_expression_l10_1005

theorem eval_expression : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := 
by
  sorry

end eval_expression_l10_1005


namespace range_of_m_l10_1066

theorem range_of_m (m : ℝ) :
  (∃ (x1 x2 : ℝ), (2*x1^2 - 2*x1 + 3*m - 1 = 0 ∧ 2*x2^2 - 2*x2 + 3*m - 1 = 0) ∧ (x1 * x2 > x1 + x2 - 4)) →
  -5/3 < m ∧ m ≤ 1/2 :=
by
  sorry

end range_of_m_l10_1066


namespace Jake_weight_is_118_l10_1063

-- Define the current weights of Jake, his sister, and Mark
variable (J S M : ℕ)

-- Define the given conditions
axiom h1 : J - 12 = 2 * (S + 4)
axiom h2 : M = J + S + 50
axiom h3 : J + S + M = 385

theorem Jake_weight_is_118 : J = 118 :=
by
  sorry

end Jake_weight_is_118_l10_1063


namespace train_speeds_proof_l10_1092

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

end train_speeds_proof_l10_1092


namespace janet_percentage_l10_1079

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

end janet_percentage_l10_1079


namespace same_terminal_side_l10_1036

theorem same_terminal_side (k : ℤ) : 
  ((2 * k + 1) * 180) % 360 = ((4 * k + 1) * 180) % 360 ∨ ((2 * k + 1) * 180) % 360 = ((4 * k - 1) * 180) % 360 := 
sorry

end same_terminal_side_l10_1036


namespace distance_to_post_office_l10_1001

variable (D : ℚ)
variable (rate_to_post : ℚ := 25)
variable (rate_back : ℚ := 4)
variable (total_time : ℚ := 5 + 48 / 60)

theorem distance_to_post_office : (D / rate_to_post + D / rate_back = total_time) → D = 20 := by
  sorry

end distance_to_post_office_l10_1001


namespace isosceles_triangle_perimeter_l10_1014

theorem isosceles_triangle_perimeter (a b : ℕ) (h_isosceles : a = 3 ∨ a = 7 ∨ b = 3 ∨ b = 7) (h_ineq1 : 3 + 3 ≤ b ∨ b + b ≤ 3) (h_ineq2 : 7 + 7 ≥ a ∨ a + a ≥ 7) :
  (a = 3 ∧ b = 7) → 3 + 7 + 7 = 17 :=
by
  -- To be completed
  sorry

end isosceles_triangle_perimeter_l10_1014


namespace digits_making_number_divisible_by_4_l10_1088

theorem digits_making_number_divisible_by_4 (N : ℕ) (hN : N < 10) :
  (∃ n0 n4 n8, n0 = 0 ∧ n4 = 4 ∧ n8 = 8 ∧ N = n0 ∨ N = n4 ∨ N = n8) :=
by
  sorry

end digits_making_number_divisible_by_4_l10_1088


namespace comic_books_ratio_l10_1055

variable (S : ℕ)

def initial_comics := 22
def remaining_comics := 17
def comics_bought := 6

theorem comic_books_ratio (h1 : initial_comics - S + comics_bought = remaining_comics) :
  (S : ℚ) / initial_comics = 1 / 2 := by
  sorry

end comic_books_ratio_l10_1055


namespace probability_not_buy_l10_1068

-- Define the given probability of Sam buying a new book
def P_buy : ℚ := 5 / 8

-- Theorem statement: The probability that Sam will not buy a new book is 3 / 8
theorem probability_not_buy : 1 - P_buy = 3 / 8 :=
by
  -- Proof omitted
  sorry

end probability_not_buy_l10_1068


namespace range_of_a_l10_1025

noncomputable def f (a x : ℝ) := a * x - x^2 - Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ 2*x₁*x₁ - a*x₁ + 1 = 0 ∧ 
  2*x₂*x₂ - a*x₂ + 1 = 0 ∧ f a x₁ + f a x₂ ≥ 4 + Real.log 2) ↔ 
  a ∈ Set.Ici (2 * Real.sqrt 3) := 
sorry

end range_of_a_l10_1025


namespace greatest_integer_a_for_domain_of_expression_l10_1000

theorem greatest_integer_a_for_domain_of_expression :
  ∃ a : ℤ, (a^2 < 60 ∧ (∀ b : ℤ, b^2 < 60 → b ≤ a)) :=
  sorry

end greatest_integer_a_for_domain_of_expression_l10_1000


namespace text_message_cost_eq_l10_1060

theorem text_message_cost_eq (x : ℝ) (CA CB : ℝ) : 
  (CA = 0.25 * x + 9) → (CB = 0.40 * x) → CA = CB → x = 60 :=
by
  intros hCA hCB heq
  sorry

end text_message_cost_eq_l10_1060


namespace additional_matches_l10_1057

theorem additional_matches 
  (avg_runs_first_25 : ℕ → ℚ) 
  (avg_runs_additional : ℕ → ℚ) 
  (avg_runs_all : ℚ) 
  (total_matches_first_25 : ℕ) 
  (total_matches_all : ℕ) 
  (total_runs_first_25 : ℚ) 
  (total_runs_all : ℚ) 
  (x : ℕ)
  (h1 : avg_runs_first_25 25 = 45)
  (h2 : avg_runs_additional x = 15)
  (h3 : avg_runs_all = 38.4375)
  (h4 : total_matches_first_25 = 25)
  (h5 : total_matches_all = 32)
  (h6 : total_runs_first_25 = avg_runs_first_25 25 * 25)
  (h7 : total_runs_all = avg_runs_all * 32)
  (h8 : total_runs_first_25 + avg_runs_additional x * x = total_runs_all) :
  x = 7 :=
sorry

end additional_matches_l10_1057


namespace intersection_A_B_intersection_A_complementB_l10_1019

-- Definitions of the sets A and B
def setA : Set ℝ := { x | -5 ≤ x ∧ x ≤ 3 }
def setB : Set ℝ := { x | x < -2 ∨ x > 4 }

-- Proof problem 1: A ∩ B = { x | -5 ≤ x < -2 }
theorem intersection_A_B:
  setA ∩ setB = { x : ℝ | -5 ≤ x ∧ x < -2 } :=
sorry

-- Definition of the complement of B
def complB : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

-- Proof problem 2: A ∩ (complB) = { x | -2 ≤ x ≤ 3 }
theorem intersection_A_complementB:
  setA ∩ complB = { x : ℝ | -2 ≤ x ∧ x ≤ 3 } :=
sorry

end intersection_A_B_intersection_A_complementB_l10_1019


namespace arithmetic_sequence_a10_l10_1006

theorem arithmetic_sequence_a10 
  (S : ℕ → ℕ)
  (a : ℕ → ℕ)
  (h_seq : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h_S4 : S 4 = 10)
  (h_S9 : S 9 = 45) :
  a 10 = 10 :=
sorry

end arithmetic_sequence_a10_l10_1006


namespace product_of_numbers_l10_1029

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 85) : x * y = 18 := by
  sorry

end product_of_numbers_l10_1029


namespace growth_rate_double_l10_1031

noncomputable def lake_coverage (days : ℕ) : ℝ := if days = 39 then 1 else if days = 38 then 0.5 else 0  -- Simplified condition statement

theorem growth_rate_double (days : ℕ) : 
  (lake_coverage 39 = 1) → (lake_coverage 38 = 0.5) → (∀ n, lake_coverage (n + 1) = 2 * lake_coverage n) := 
  by 
  intros h39 h38 
  apply sorry  -- Proof not required

end growth_rate_double_l10_1031


namespace number_of_space_diagonals_l10_1064

theorem number_of_space_diagonals (V E F tF qF : ℕ)
    (hV : V = 30) (hE : E = 72) (hF : F = 44) (htF : tF = 34) (hqF : qF = 10) : 
    V * (V - 1) / 2 - E - qF * 2 = 343 :=
by
  sorry

end number_of_space_diagonals_l10_1064


namespace number_of_matches_is_85_l10_1090

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

end number_of_matches_is_85_l10_1090


namespace running_speed_l10_1072

theorem running_speed (side : ℕ) (time_seconds : ℕ) (speed_result : ℕ) 
  (h1 : side = 50) (h2 : time_seconds = 60) (h3 : speed_result = 12) : 
  (4 * side * 3600) / (time_seconds * 1000) = speed_result :=
by
  sorry

end running_speed_l10_1072


namespace final_expression_l10_1021

theorem final_expression (y : ℝ) : (3 * (1 / 2 * (12 * y + 3))) = 18 * y + 4.5 :=
by
  sorry

end final_expression_l10_1021


namespace rose_days_to_complete_work_l10_1044

theorem rose_days_to_complete_work (R : ℝ) (h1 : 1 / 10 + 1 / R = 1 / 8) : R = 40 := 
sorry

end rose_days_to_complete_work_l10_1044


namespace find_number_l10_1003

theorem find_number
  (a b c : ℕ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 328 - (100 * a + 10 * b + c) = a + b + c) :
  100 * a + 10 * b + c = 317 :=
sorry

end find_number_l10_1003


namespace base7_digit_divisibility_l10_1026

-- Define base-7 digit integers
notation "digit" => Fin 7

-- Define conversion from base-7 to base-10 for the form 3dd6_7
def base7_to_base10 (d : digit) : ℤ := 3 * (7^3) + (d:ℤ) * (7^2) + (d:ℤ) * 7 + 6

-- Define the property of being divisible by 13
def is_divisible_by_13 (n : ℤ) : Prop := ∃ k : ℤ, n = 13 * k

-- Formalize the theorem
theorem base7_digit_divisibility (d : digit) :
  is_divisible_by_13 (base7_to_base10 d) ↔ d = 4 :=
sorry

end base7_digit_divisibility_l10_1026


namespace A_and_B_finish_together_in_11_25_days_l10_1071

theorem A_and_B_finish_together_in_11_25_days (A_rate B_rate : ℝ)
    (hA : A_rate = 1/18) (hB : B_rate = 1/30) :
    1 / (A_rate + B_rate) = 11.25 := by
  sorry

end A_and_B_finish_together_in_11_25_days_l10_1071


namespace probability_of_same_color_is_correct_l10_1050

def probability_same_color (blue_balls yellow_balls : ℕ) : ℚ :=
  let total_balls := blue_balls + yellow_balls
  let prob_blue := (blue_balls / total_balls : ℚ)
  let prob_yellow := (yellow_balls / total_balls : ℚ)
  (prob_blue ^ 2) + (prob_yellow ^ 2)

theorem probability_of_same_color_is_correct :
  probability_same_color 8 5 = 89 / 169 :=
by 
  sorry

end probability_of_same_color_is_correct_l10_1050


namespace proof_inequality_l10_1085

noncomputable def inequality (a b c : ℝ) : Prop :=
  a + 2 * b + c = 1 ∧ a^2 + b^2 + c^2 = 1 → -2/3 ≤ c ∧ c ≤ 1

theorem proof_inequality (a b c : ℝ) (h : a + 2 * b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : -2/3 ≤ c ∧ c ≤ 1 :=
by {
  sorry
}

end proof_inequality_l10_1085


namespace all_sets_B_l10_1043

open Set

theorem all_sets_B (B : Set ℕ) :
  { B | {1, 2} ∪ B = {1, 2, 3} } =
  ({ {3}, {1, 3}, {2, 3}, {1, 2, 3} } : Set (Set ℕ)) :=
sorry

end all_sets_B_l10_1043


namespace jovana_shells_l10_1051

def initial_weight : ℕ := 5
def added_weight : ℕ := 23
def total_weight : ℕ := 28

theorem jovana_shells :
  initial_weight + added_weight = total_weight :=
by
  sorry

end jovana_shells_l10_1051


namespace intersection_points_eq_one_l10_1040

-- Definitions for the equations of the circles
def circle1 (x y : ℝ) : ℝ := x^2 + (y - 3)^2
def circle2 (x y : ℝ) : ℝ := x^2 + (y + 2)^2

-- The proof problem statement
theorem intersection_points_eq_one : 
  ∃ p : ℝ × ℝ, (circle1 p.1 p.2 = 9) ∧ (circle2 p.1 p.2 = 4) ∧
  (∀ q : ℝ × ℝ, (circle1 q.1 q.2 = 9) ∧ (circle2 q.1 q.2 = 4) → q = p) :=
sorry

end intersection_points_eq_one_l10_1040


namespace vacation_hours_per_week_l10_1056

open Nat

theorem vacation_hours_per_week :
  let planned_hours_per_week := 25
  let total_weeks := 15
  let total_money_needed := 4500
  let sick_weeks := 3
  let hourly_rate := total_money_needed / (planned_hours_per_week * total_weeks)
  let remaining_weeks := total_weeks - sick_weeks
  let total_hours_needed := total_money_needed / hourly_rate
  let required_hours_per_week := total_hours_needed / remaining_weeks
  required_hours_per_week = 31.25 := by
sorry

end vacation_hours_per_week_l10_1056


namespace evening_sales_l10_1017

theorem evening_sales
  (remy_bottles_morning : ℕ := 55)
  (nick_bottles_fewer : ℕ := 6)
  (price_per_bottle : ℚ := 0.50)
  (evening_sales_more : ℚ := 3) :
  let nick_bottles_morning := remy_bottles_morning - nick_bottles_fewer
  let remy_sales_morning := remy_bottles_morning * price_per_bottle
  let nick_sales_morning := nick_bottles_morning * price_per_bottle
  let total_morning_sales := remy_sales_morning + nick_sales_morning
  let total_evening_sales := total_morning_sales + evening_sales_more
  total_evening_sales = 55 :=
by
  sorry

end evening_sales_l10_1017
