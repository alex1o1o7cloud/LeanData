import Mathlib

namespace NUMINAMATH_GPT_juan_more_marbles_l1420_142050

theorem juan_more_marbles (connie_marbles : ℕ) (juan_marbles : ℕ) (h1 : connie_marbles = 323) (h2 : juan_marbles = 498) :
  juan_marbles - connie_marbles = 175 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_juan_more_marbles_l1420_142050


namespace NUMINAMATH_GPT_minimum_perimeter_l1420_142002

-- Define the area condition
def area_condition (l w : ℝ) : Prop := l * w = 64

-- Define the perimeter function
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w

-- The theorem statement based on the conditions and the correct answer
theorem minimum_perimeter (l w : ℝ) (h : area_condition l w) : 
  perimeter l w ≥ 32 := by
sorry

end NUMINAMATH_GPT_minimum_perimeter_l1420_142002


namespace NUMINAMATH_GPT_geom_seq_sum_five_terms_l1420_142047

theorem geom_seq_sum_five_terms (a : ℕ → ℝ) (q : ℝ) 
    (h_pos : ∀ n, 0 < a n)
    (h_a2 : a 2 = 8) 
    (h_arith : 2 * a 4 - a 3 = a 3 - 4 * a 5) :
    a 1 * (1 - q^5) / (1 - q) = 31 :=
by
    sorry

end NUMINAMATH_GPT_geom_seq_sum_five_terms_l1420_142047


namespace NUMINAMATH_GPT_milk_price_per_liter_l1420_142014

theorem milk_price_per_liter (M : ℝ) 
  (price_fruit_per_kg : ℝ) (price_each_fruit_kg_eq_2: price_fruit_per_kg = 2)
  (milk_liters_per_batch : ℝ) (milk_liters_per_batch_eq_10: milk_liters_per_batch = 10)
  (fruit_kg_per_batch : ℝ) (fruit_kg_per_batch_eq_3 : fruit_kg_per_batch = 3)
  (cost_three_batches : ℝ) (cost_three_batches_eq_63: cost_three_batches = 63) :
  M = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_milk_price_per_liter_l1420_142014


namespace NUMINAMATH_GPT_factor_polynomial_l1420_142071

theorem factor_polynomial (x y : ℝ) : 
  (x^2 - 2*x*y + y^2 - 16) = (x - y + 4) * (x - y - 4) :=
sorry

end NUMINAMATH_GPT_factor_polynomial_l1420_142071


namespace NUMINAMATH_GPT_part1_part2_l1420_142096

section PartOne

variables (x y : ℕ)
def condition1 := x + y = 360
def condition2 := x - y = 110

theorem part1 (h1 : condition1 x y) (h2 : condition2 x y) : x = 235 ∧ y = 125 := by {
  sorry
}

end PartOne

section PartTwo

variables (t W : ℕ)
def tents_capacity (t : ℕ) := 40 * t + 20 * (9 - t)
def food_capacity (t : ℕ) := 10 * t + 20 * (9 - t)
def transportation_cost (t : ℕ) := 4000 * t + 3600 * (9 - t)

theorem part2 
  (htents : tents_capacity t ≥ 235) 
  (hfood : food_capacity t ≥ 125) : 
  W = transportation_cost t → t = 3 ∧ W = 33600 := by {
  sorry
}

end PartTwo

end NUMINAMATH_GPT_part1_part2_l1420_142096


namespace NUMINAMATH_GPT_easter_eggs_total_l1420_142058

theorem easter_eggs_total (h he total : ℕ)
 (hannah_eggs : h = 42) 
 (twice_he : h = 2 * he) 
 (total_eggs : total = h + he) : 
 total = 63 := 
sorry

end NUMINAMATH_GPT_easter_eggs_total_l1420_142058


namespace NUMINAMATH_GPT_sum_of_first_six_terms_l1420_142023

theorem sum_of_first_six_terms 
  (a₁ : ℝ) 
  (r : ℝ) 
  (h_ratio : r = 2) 
  (h_sum_three : a₁ + 2*a₁ + 4*a₁ = 3) 
  : a₁ * (r^6 - 1) / (r - 1) = 27 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_first_six_terms_l1420_142023


namespace NUMINAMATH_GPT_range_of_a_l1420_142070

noncomputable def f (a x : ℝ) : ℝ := x^2 - a * x + a + 3
noncomputable def g (a x : ℝ) : ℝ := a * x - 2 * a

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, f a x₀ < 0 ∧ g a x₀ < 0) → 7 < a :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1420_142070


namespace NUMINAMATH_GPT_parabola_latus_rectum_equation_l1420_142037

theorem parabola_latus_rectum_equation :
  (∃ (y x : ℝ), y^2 = 4 * x) → (∀ x, x = -1) :=
by
  sorry

end NUMINAMATH_GPT_parabola_latus_rectum_equation_l1420_142037


namespace NUMINAMATH_GPT_simplify_expression_l1420_142010

theorem simplify_expression (x : ℝ) : (3 * x)^5 + (5 * x) * (x^4) - 7 * x^5 = 241 * x^5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1420_142010


namespace NUMINAMATH_GPT_remainder_proof_l1420_142069

noncomputable def problem (n : ℤ) : Prop :=
  n % 9 = 4

noncomputable def solution (n : ℤ) : ℤ :=
  (4 * n - 11) % 9

theorem remainder_proof (n : ℤ) (h : problem n) : solution n = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_proof_l1420_142069


namespace NUMINAMATH_GPT_total_books_l1420_142042

def numberOfMysteryShelves := 6
def numberOfPictureShelves := 2
def booksPerShelf := 9

theorem total_books (hMystery : numberOfMysteryShelves = 6) 
                    (hPicture : numberOfPictureShelves = 2) 
                    (hBooksPerShelf : booksPerShelf = 9) :
  numberOfMysteryShelves * booksPerShelf + numberOfPictureShelves * booksPerShelf = 72 :=
  by 
  sorry

end NUMINAMATH_GPT_total_books_l1420_142042


namespace NUMINAMATH_GPT_quadratic_inequality_solution_range_l1420_142056

theorem quadratic_inequality_solution_range (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + 1 > 0) ↔ -2 < m ∧ m < 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_range_l1420_142056


namespace NUMINAMATH_GPT_number_of_triangles_2016_30_l1420_142060

def f (m n : ℕ) : ℕ :=
  2 * m - n - 2

theorem number_of_triangles_2016_30 :
  f 2016 30 = 4000 := 
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_2016_30_l1420_142060


namespace NUMINAMATH_GPT_paddington_more_goats_l1420_142001

theorem paddington_more_goats (W P total : ℕ) (hW : W = 140) (hTotal : total = 320) (hTotalGoats : W + P = total) : P - W = 40 :=
by
  sorry

end NUMINAMATH_GPT_paddington_more_goats_l1420_142001


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1420_142025

theorem no_positive_integer_solutions (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : x^4 * y^4 - 14 * x^2 * y^2 + 49 ≠ 0 := 
by sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1420_142025


namespace NUMINAMATH_GPT_train_speed_l1420_142005

theorem train_speed
    (length_train : ℝ) (length_platform : ℝ) (time_seconds : ℝ)
    (h_train : length_train = 250)
    (h_platform : length_platform = 250.04)
    (h_time : time_seconds = 25) :
    (length_train + length_platform) / time_seconds * 3.6 = 72.006 :=
by sorry

end NUMINAMATH_GPT_train_speed_l1420_142005


namespace NUMINAMATH_GPT_smallest_integer_to_make_perfect_square_l1420_142017

def y : ℕ := 2^3 * 3^4 * 4^5 * 5^6 * 6^7 * 7^8 * 8^9 * 9^10

theorem smallest_integer_to_make_perfect_square :
  ∃ n : ℕ, n > 0 ∧ (∃ k : ℕ, (n * y) = k^2) ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_to_make_perfect_square_l1420_142017


namespace NUMINAMATH_GPT_weight_of_mixture_is_112_5_l1420_142084

noncomputable def weight_of_mixture (W : ℝ) : Prop :=
  (5 / 14) * W + (3 / 10) * W + (2 / 9) * W + (1 / 7) * W + 2.5 = W

theorem weight_of_mixture_is_112_5 : ∃ W : ℝ, weight_of_mixture W ∧ W = 112.5 :=
by {
  use 112.5,
  sorry
}

end NUMINAMATH_GPT_weight_of_mixture_is_112_5_l1420_142084


namespace NUMINAMATH_GPT_total_distance_is_75_l1420_142094

def distance1 : ℕ := 30
def distance2 : ℕ := 20
def distance3 : ℕ := 25

def total_distance : ℕ := distance1 + distance2 + distance3

theorem total_distance_is_75 : total_distance = 75 := by
  sorry

end NUMINAMATH_GPT_total_distance_is_75_l1420_142094


namespace NUMINAMATH_GPT_inverse_proposition_l1420_142062

-- Define the variables m, n, and a^2
variables (m n : ℝ) (a : ℝ)

-- State the proof problem
theorem inverse_proposition
  (h1 : m > n)
: m * a^2 > n * a^2 :=
sorry

end NUMINAMATH_GPT_inverse_proposition_l1420_142062


namespace NUMINAMATH_GPT_ratio_of_bike_to_tractor_speed_l1420_142063

theorem ratio_of_bike_to_tractor_speed (d_tr: ℝ) (t_tr: ℝ) (d_car: ℝ) (t_car: ℝ) (k: ℝ) (β: ℝ) 
  (h1: d_tr / t_tr = 25) 
  (h2: d_car / t_car = 90)
  (h3: 90 = 9 / 5 * β)
: β / (d_tr / t_tr) = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_bike_to_tractor_speed_l1420_142063


namespace NUMINAMATH_GPT_total_distance_run_l1420_142083

-- Given conditions
def number_of_students : Nat := 18
def distance_per_student : Nat := 106

-- Prove that the total distance run by the students equals 1908 meters.
theorem total_distance_run : number_of_students * distance_per_student = 1908 := by
  sorry

end NUMINAMATH_GPT_total_distance_run_l1420_142083


namespace NUMINAMATH_GPT_circles_intersect_and_common_chord_l1420_142059

theorem circles_intersect_and_common_chord :
  (∃ P : ℝ × ℝ, P.1 ^ 2 + P.2 ^ 2 - P.1 + P.2 - 2 = 0 ∧
                P.1 ^ 2 + P.2 ^ 2 = 5) ∧
  (∀ x y : ℝ, (x ^ 2 + y ^ 2 - x + y - 2 = 0 ∧ x ^ 2 + y ^ 2 = 5) →
              x - y - 3 = 0) ∧
  (∃ A B : ℝ × ℝ, A.1 ^ 2 + A.2 ^ 2 - A.1 + A.2 - 2 = 0 ∧
                   A.1 ^ 2 + A.2 ^ 2 = 5 ∧
                   B.1 ^ 2 + B.2 ^ 2 - B.1 + B.2 - 2 = 0 ∧
                   B.1 ^ 2 + B.2 ^ 2 = 5 ∧
                   (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 2) := sorry

end NUMINAMATH_GPT_circles_intersect_and_common_chord_l1420_142059


namespace NUMINAMATH_GPT_evaluate_72_squared_minus_48_squared_l1420_142097

theorem evaluate_72_squared_minus_48_squared :
  (72:ℤ)^2 - (48:ℤ)^2 = 2880 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_72_squared_minus_48_squared_l1420_142097


namespace NUMINAMATH_GPT_simplify_trig_expression_l1420_142038

theorem simplify_trig_expression :
  (2 - Real.sin 21 * Real.sin 21 - Real.cos 21 * Real.cos 21 + 
  (Real.sin 17 * Real.sin 17) * (Real.sin 17 * Real.sin 17) + 
  (Real.sin 17 * Real.sin 17) * (Real.cos 17 * Real.cos 17) + 
  (Real.cos 17 * Real.cos 17)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_trig_expression_l1420_142038


namespace NUMINAMATH_GPT_max_ski_trips_l1420_142028

/--
The ski lift carries skiers from the bottom of the mountain to the top, taking 15 minutes each way, 
and it takes 5 minutes to ski back down the mountain. 
Given that the total available time is 2 hours, prove that the maximum number of trips 
down the mountain in that time is 6.
-/
theorem max_ski_trips (ride_up_time : ℕ) (ski_down_time : ℕ) (total_time : ℕ) :
  ride_up_time = 15 →
  ski_down_time = 5 →
  total_time = 120 →
  (total_time / (ride_up_time + ski_down_time) = 6) :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_max_ski_trips_l1420_142028


namespace NUMINAMATH_GPT_add_to_both_num_and_denom_l1420_142031

theorem add_to_both_num_and_denom (n : ℕ) : (4 + n) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end NUMINAMATH_GPT_add_to_both_num_and_denom_l1420_142031


namespace NUMINAMATH_GPT_units_digit_7_pow_l1420_142016

theorem units_digit_7_pow (n : ℕ) : 
  ∃ k, 7^n % 10 = k ∧ ((7^1 % 10 = 7) ∧ (7^2 % 10 = 9) ∧ (7^3 % 10 = 3) ∧ (7^4 % 10 = 1) ∧ (7^5 % 10 = 7)) → 
  7^2010 % 10 = 9 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_7_pow_l1420_142016


namespace NUMINAMATH_GPT_time_first_tap_to_fill_cistern_l1420_142018

-- Defining the conditions
axiom second_tap_empty_time : ℝ
axiom combined_tap_fill_time : ℝ
axiom second_tap_rate : ℝ
axiom combined_tap_rate : ℝ

-- Specifying the given conditions
def problem_conditions :=
  second_tap_empty_time = 8 ∧
  combined_tap_fill_time = 8 ∧
  second_tap_rate = 1 / 8 ∧
  combined_tap_rate = 1 / 8

-- Defining the problem statement
theorem time_first_tap_to_fill_cistern :
  problem_conditions →
  (∃ T : ℝ, (1 / T - 1 / 8 = 1 / 8) ∧ T = 4) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_time_first_tap_to_fill_cistern_l1420_142018


namespace NUMINAMATH_GPT_gravitational_force_solution_l1420_142073

noncomputable def gravitational_force_proportionality (d d' : ℕ) (f f' k : ℝ) : Prop :=
  (f * (d:ℝ)^2 = k) ∧
  d = 6000 ∧
  f = 800 ∧
  d' = 36000 ∧
  f' * (d':ℝ)^2 = k

theorem gravitational_force_solution : ∃ k, gravitational_force_proportionality 6000 36000 800 (1/45) k :=
by
  sorry

end NUMINAMATH_GPT_gravitational_force_solution_l1420_142073


namespace NUMINAMATH_GPT_calculate_value_l1420_142057

theorem calculate_value : 15 * (1 / 3) + 45 * (2 / 3) = 35 := 
by
simp -- We use simp to simplify the expression
sorry -- We put sorry as we are skipping the full proof

end NUMINAMATH_GPT_calculate_value_l1420_142057


namespace NUMINAMATH_GPT_xyz_value_l1420_142091

variable {x y z : ℝ}

theorem xyz_value (h1 : (x + y + z) * (x * y + x * z + y * z) = 27)
                  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9) :
                x * y * z = 6 := by
  sorry

end NUMINAMATH_GPT_xyz_value_l1420_142091


namespace NUMINAMATH_GPT_solve_for_x_l1420_142033

def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

theorem solve_for_x (x : ℝ) : star 6 x = 45 ↔ x = 19 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1420_142033


namespace NUMINAMATH_GPT_range_of_a_l1420_142090

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 4 * x + a - 3 > 0) ↔ (0 ≤ a ∧ a ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1420_142090


namespace NUMINAMATH_GPT_predict_sales_amount_l1420_142078

theorem predict_sales_amount :
  let x_data := [2, 4, 5, 6, 8]
  let y_data := [30, 40, 50, 60, 70]
  let b := 7
  let x := 10 -- corresponding to 10,000 yuan investment
  let a := 15 -- \hat{a} calculated from the regression equation and data points
  let regression (x : ℝ) := b * x + a
  regression x = 85 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_predict_sales_amount_l1420_142078


namespace NUMINAMATH_GPT_robie_initial_cards_l1420_142000

def total_initial_boxes : Nat := 2 + 5
def cards_per_box : Nat := 10
def unboxed_cards : Nat := 5

theorem robie_initial_cards :
  (total_initial_boxes * cards_per_box + unboxed_cards) = 75 :=
by
  sorry

end NUMINAMATH_GPT_robie_initial_cards_l1420_142000


namespace NUMINAMATH_GPT_john_saving_yearly_l1420_142065

def old_monthly_cost : ℕ := 1200
def increase_percentage : ℕ := 40
def split_count : ℕ := 3

def old_annual_cost (monthly_cost : ℕ) := monthly_cost * 12
def new_monthly_cost (monthly_cost : ℕ) (percentage : ℕ) := monthly_cost * (100 + percentage) / 100
def new_monthly_share (new_cost : ℕ) (split : ℕ) := new_cost / split
def new_annual_cost (monthly_share : ℕ) := monthly_share * 12
def annual_savings (old_annual : ℕ) (new_annual : ℕ) := old_annual - new_annual

theorem john_saving_yearly 
  (old_cost : ℕ := old_monthly_cost)
  (increase : ℕ := increase_percentage)
  (split : ℕ := split_count) :
  annual_savings (old_annual_cost old_cost) 
                 (new_annual_cost (new_monthly_share (new_monthly_cost old_cost increase) split)) 
  = 7680 :=
by
  sorry

end NUMINAMATH_GPT_john_saving_yearly_l1420_142065


namespace NUMINAMATH_GPT_probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l1420_142095

-- Defining the conditions
def p : ℚ := 4 / 5
def n : ℕ := 5
def k1 : ℕ := 2
def k2 : ℕ := 1

-- Binomial coefficient function
def binomial (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

-- Binomial probability function
def binom_prob (k n : ℕ) (p : ℚ) : ℚ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- The first proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate is 0.05 given the accuracy rate
theorem probability_of_2_out_of_5_accurate :
  binom_prob k1 n p = 0.05 := by
  sorry

-- The second proof problem:
-- Prove that the probability of exactly 2 out of 5 forecasts being accurate, with the third forecast being one of the accurate ones, is 0.02 given the accuracy rate
theorem probability_of_2_out_of_5_with_third_accurate :
  binom_prob k2 (n - 1) p = 0.02 := by
  sorry

end NUMINAMATH_GPT_probability_of_2_out_of_5_accurate_probability_of_2_out_of_5_with_third_accurate_l1420_142095


namespace NUMINAMATH_GPT_circle_k_range_l1420_142046

def circle_equation (k x y : ℝ) : Prop :=
  x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0

theorem circle_k_range (k : ℝ) (h : ∃ x y, circle_equation k x y) : k > 4 ∨ k < -1 :=
by
  sorry

end NUMINAMATH_GPT_circle_k_range_l1420_142046


namespace NUMINAMATH_GPT_solve_equation_l1420_142039

theorem solve_equation (x : ℝ) (h : x ≠ 3) : (x + 6) / (x - 3) = 4 ↔ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1420_142039


namespace NUMINAMATH_GPT_ratio_of_female_to_male_members_l1420_142007

theorem ratio_of_female_to_male_members 
  (f m : ℕ)
  (avg_age_female avg_age_male avg_age_membership : ℕ)
  (hf : avg_age_female = 35)
  (hm : avg_age_male = 30)
  (ha : avg_age_membership = 32)
  (h_avg : (35 * f + 30 * m) / (f + m) = 32) : 
  f / m = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_of_female_to_male_members_l1420_142007


namespace NUMINAMATH_GPT_dave_paid_for_6_candy_bars_l1420_142067

-- Given conditions
def number_of_candy_bars : ℕ := 20
def cost_per_candy_bar : ℝ := 1.50
def amount_paid_by_john : ℝ := 21

-- Correct answer
def number_of_candy_bars_paid_by_dave : ℝ := 6

-- The proof problem in Lean statement
theorem dave_paid_for_6_candy_bars (H : number_of_candy_bars * cost_per_candy_bar - amount_paid_by_john = 9) :
  number_of_candy_bars_paid_by_dave = 6 := by
sorry

end NUMINAMATH_GPT_dave_paid_for_6_candy_bars_l1420_142067


namespace NUMINAMATH_GPT_robin_cut_hair_l1420_142006

-- Definitions as per the given conditions
def initial_length := 17
def current_length := 13

-- Statement of the proof problem
theorem robin_cut_hair : initial_length - current_length = 4 := 
by 
  sorry

end NUMINAMATH_GPT_robin_cut_hair_l1420_142006


namespace NUMINAMATH_GPT_sector_area_l1420_142061

theorem sector_area (r : ℝ) (θ : ℝ) (arc_area : ℝ) : 
  r = 24 ∧ θ = 110 ∧ arc_area = 176 * Real.pi → 
  arc_area = (θ / 360) * (Real.pi * r ^ 2) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sector_area_l1420_142061


namespace NUMINAMATH_GPT_number_of_yellow_parrots_l1420_142081

theorem number_of_yellow_parrots (total_parrots : ℕ) (red_fraction : ℚ) 
  (h_total_parrots : total_parrots = 108) 
  (h_red_fraction : red_fraction = 5 / 6) : 
  ∃ (yellow_parrots : ℕ), yellow_parrots = total_parrots * (1 - red_fraction) ∧ yellow_parrots = 18 := 
by
  sorry

end NUMINAMATH_GPT_number_of_yellow_parrots_l1420_142081


namespace NUMINAMATH_GPT_glass_heavier_than_plastic_l1420_142048

-- Define the conditions
def condition1 (G : ℕ) : Prop := 3 * G = 600
def condition2 (G P : ℕ) : Prop := 4 * G + 5 * P = 1050

-- Define the theorem to prove
theorem glass_heavier_than_plastic (G P : ℕ) (h1 : condition1 G) (h2 : condition2 G P) : G - P = 150 :=
by
  sorry

end NUMINAMATH_GPT_glass_heavier_than_plastic_l1420_142048


namespace NUMINAMATH_GPT_solve_cryptarithm_l1420_142068

-- Definitions for digits mapped to letters
def C : ℕ := 9
def H : ℕ := 3
def U : ℕ := 5
def K : ℕ := 4
def T : ℕ := 1
def R : ℕ := 2
def I : ℕ := 0
def G : ℕ := 6
def N : ℕ := 8
def S : ℕ := 7

-- Function to evaluate the cryptarithm sum
def cryptarithm_sum : ℕ :=
  (C*10000 + H*1000 + U*100 + C*10 + K) +
  (T*10000 + R*1000 + I*100 + G*10 + G) +
  (T*10000 + U*1000 + R*100 + N*10 + S)

-- Equation checking the result
def cryptarithm_correct : Prop :=
  cryptarithm_sum = T*100000 + R*10000 + I*1000 + C*100 + K*10 + S

-- The theorem we want to prove
theorem solve_cryptarithm : cryptarithm_correct :=
by
  -- Proof steps would be filled here
  -- but for now, we just acknowledge it is a theorem
  sorry

end NUMINAMATH_GPT_solve_cryptarithm_l1420_142068


namespace NUMINAMATH_GPT_jill_peaches_l1420_142029

open Nat

theorem jill_peaches (Jake Steven Jill : ℕ)
  (h1 : Jake = Steven - 6)
  (h2 : Steven = Jill + 18)
  (h3 : Jake = 17) :
  Jill = 5 := 
by
  sorry

end NUMINAMATH_GPT_jill_peaches_l1420_142029


namespace NUMINAMATH_GPT_total_amount_earned_l1420_142008

-- Definitions of the conditions.
def work_done_per_day (days : ℕ) : ℚ := 1 / days

def total_work_done_per_day : ℚ :=
  work_done_per_day 6 + work_done_per_day 8 + work_done_per_day 12

def b_share : ℚ := work_done_per_day 8

def total_amount (b_earnings : ℚ) : ℚ := b_earnings * (total_work_done_per_day / b_share)

-- Main theorem stating that the total amount earned is $1170 if b's share is $390.
theorem total_amount_earned (h_b : b_share * 390 = 390) : total_amount 390 = 1170 := by sorry

end NUMINAMATH_GPT_total_amount_earned_l1420_142008


namespace NUMINAMATH_GPT_two_digit_number_system_l1420_142066

theorem two_digit_number_system (x y : ℕ) :
  (10 * x + y - 3 * (x + y) = 13) ∧ (10 * x + y - 6 = 4 * (x + y)) :=
by sorry

end NUMINAMATH_GPT_two_digit_number_system_l1420_142066


namespace NUMINAMATH_GPT_negation_proposition_l1420_142093

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ ∀ x : ℝ, x^3 + 5*x - 2 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_negation_proposition_l1420_142093


namespace NUMINAMATH_GPT_original_price_increased_by_total_percent_l1420_142082

noncomputable def percent_increase_sequence (P : ℝ) : ℝ :=
  let step1 := P * 1.15
  let step2 := step1 * 1.40
  let step3 := step2 * 1.20
  let step4 := step3 * 0.90
  let step5 := step4 * 1.25
  (step5 - P) / P * 100

theorem original_price_increased_by_total_percent (P : ℝ) : percent_increase_sequence P = 117.35 :=
by
  -- Sorry is used here for simplicity, but the automated proof will involve calculating the exact percentage increase step-by-step.
  sorry

end NUMINAMATH_GPT_original_price_increased_by_total_percent_l1420_142082


namespace NUMINAMATH_GPT_range_a_if_no_solution_l1420_142054

def f (x : ℝ) : ℝ := abs (x - abs (2 * x - 4))

theorem range_a_if_no_solution (a : ℝ) :
  (∀ x : ℝ, f x > 0 → false) → a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_a_if_no_solution_l1420_142054


namespace NUMINAMATH_GPT_rectangle_area_increase_l1420_142080

variable {L W : ℝ} -- Define variables for length and width

theorem rectangle_area_increase (p : ℝ) (hW : W' = 0.4 * W) (hA : A' = 1.36 * (L * W)) :
  L' = L + (240 / 100) * L :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l1420_142080


namespace NUMINAMATH_GPT_work_equivalence_l1420_142035

variable (m d r : ℕ)

theorem work_equivalence (h : d > 0) : (m * d) / (m + r^2) = d := sorry

end NUMINAMATH_GPT_work_equivalence_l1420_142035


namespace NUMINAMATH_GPT_cone_to_sphere_ratio_l1420_142086

-- Prove the ratio of the cone's altitude to its base radius
theorem cone_to_sphere_ratio (r h : ℝ) (h_r_pos : 0 < r) 
  (vol_cone : ℝ) (vol_sphere : ℝ) 
  (hyp_vol_relation : vol_cone = (1 / 3) * vol_sphere)
  (vol_sphere_def : vol_sphere = (4 / 3) * π * r^3)
  (vol_cone_def : vol_cone = (1 / 3) * π * r^2 * h) :
  h / r = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_to_sphere_ratio_l1420_142086


namespace NUMINAMATH_GPT_cheap_feed_amount_l1420_142032

theorem cheap_feed_amount (x y : ℝ) (h1 : x + y = 27) (h2 : 0.17 * x + 0.36 * y = 7.02) : 
  x = 14.21 :=
sorry

end NUMINAMATH_GPT_cheap_feed_amount_l1420_142032


namespace NUMINAMATH_GPT_volume_of_quadrilateral_pyramid_l1420_142051

theorem volume_of_quadrilateral_pyramid (m α : ℝ) : 
  ∃ (V : ℝ), V = (2 / 3) * m^3 * (Real.cos α) * (Real.sin (2 * α)) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_quadrilateral_pyramid_l1420_142051


namespace NUMINAMATH_GPT_intersection_A_B_complement_A_in_U_complement_B_in_U_l1420_142011

-- Definitions and conditions
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {5, 6, 7, 8}
def B : Set ℕ := {2, 4, 6, 8}

-- Problems to prove
theorem intersection_A_B : A ∩ B = {6, 8} := by
  sorry

theorem complement_A_in_U : U \ A = {1, 2, 3, 4} := by
  sorry

theorem complement_B_in_U : U \ B = {1, 3, 5, 7} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_complement_A_in_U_complement_B_in_U_l1420_142011


namespace NUMINAMATH_GPT_MathContestMeanMedianDifference_l1420_142024

theorem MathContestMeanMedianDifference :
  (15 / 100 * 65 + 20 / 100 * 85 + 40 / 100 * 95 + 25 / 100 * 110) - 95 = -3 := 
by
  sorry

end NUMINAMATH_GPT_MathContestMeanMedianDifference_l1420_142024


namespace NUMINAMATH_GPT_email_difference_l1420_142085

def morning_emails_early : ℕ := 10
def morning_emails_late : ℕ := 15
def afternoon_emails_early : ℕ := 7
def afternoon_emails_late : ℕ := 12

theorem email_difference :
  (morning_emails_early + morning_emails_late) - (afternoon_emails_early + afternoon_emails_late) = 6 :=
by
  sorry

end NUMINAMATH_GPT_email_difference_l1420_142085


namespace NUMINAMATH_GPT_range_of_a_if_solution_non_empty_l1420_142027

variable (f : ℝ → ℝ) (a : ℝ)

/-- Given that the solution set of f(x) < | -1 | is non-empty,
    we need to prove that |a| ≥ 4. -/
theorem range_of_a_if_solution_non_empty (h : ∃ x, f x < 1) : |a| ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_if_solution_non_empty_l1420_142027


namespace NUMINAMATH_GPT_plane_equation_l1420_142092

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + s + 2 * t, 4 - 2 * s, 1 - s + t)

def normal_vector : ℝ × ℝ × ℝ :=
  (-2, -3, 4)

def point_on_plane : ℝ × ℝ × ℝ :=
  (3, 4, 1)

theorem plane_equation : ∀ (x y z : ℝ),
  (∃ (s t : ℝ), (x, y, z) = parametric_plane s t) ↔
  2 * x + 3 * y - 4 * z - 14 = 0 :=
sorry

end NUMINAMATH_GPT_plane_equation_l1420_142092


namespace NUMINAMATH_GPT_y_in_terms_of_x_l1420_142043

theorem y_in_terms_of_x (p x y : ℝ) (hx : x = 1 + 2^p) (hy : y = 1 + 2^(-p)) : y = x / (x - 1) := 
by 
  sorry

end NUMINAMATH_GPT_y_in_terms_of_x_l1420_142043


namespace NUMINAMATH_GPT_abs_eq_cases_l1420_142076

theorem abs_eq_cases (a b : ℝ) : (|a| = |b|) → (a = b ∨ a = -b) :=
sorry

end NUMINAMATH_GPT_abs_eq_cases_l1420_142076


namespace NUMINAMATH_GPT_sum_x_coordinates_intersection_mod_9_l1420_142020

theorem sum_x_coordinates_intersection_mod_9 :
  ∃ x y : ℤ, (y ≡ 3 * x + 4 [ZMOD 9]) ∧ (y ≡ 7 * x + 2 [ZMOD 9]) ∧ x ≡ 5 [ZMOD 9] := sorry

end NUMINAMATH_GPT_sum_x_coordinates_intersection_mod_9_l1420_142020


namespace NUMINAMATH_GPT_randy_final_amount_l1420_142034

-- Conditions as definitions
def initial_dollars : ℝ := 30
def initial_euros : ℝ := 20
def lunch_cost : ℝ := 10
def ice_cream_percentage : ℝ := 0.25
def snack_percentage : ℝ := 0.10
def conversion_rate : ℝ := 0.85

-- Main proof statement without the proof body
theorem randy_final_amount :
  let euros_in_dollars := initial_euros / conversion_rate
  let total_dollars := initial_dollars + euros_in_dollars
  let dollars_after_lunch := total_dollars - lunch_cost
  let ice_cream_cost := dollars_after_lunch * ice_cream_percentage
  let dollars_after_ice_cream := dollars_after_lunch - ice_cream_cost
  let snack_euros := initial_euros * snack_percentage
  let snack_dollars := snack_euros / conversion_rate
  let final_dollars := dollars_after_ice_cream - snack_dollars
  final_dollars = 30.30 :=
by
  sorry

end NUMINAMATH_GPT_randy_final_amount_l1420_142034


namespace NUMINAMATH_GPT_probability_of_even_product_l1420_142049

-- Each die has faces numbered from 1 to 8.
def faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Calculate the number of outcomes where the product of two rolls is even.
def num_even_product_outcomes : ℕ := (64 - 16)

-- Calculate the total number of outcomes when two eight-sided dice are rolled.
def total_outcomes : ℕ := 64

-- The probability that the product is even.
def probability_even_product : ℚ := num_even_product_outcomes / total_outcomes

theorem probability_of_even_product :
  probability_even_product = 3 / 4 :=
  by
    sorry

end NUMINAMATH_GPT_probability_of_even_product_l1420_142049


namespace NUMINAMATH_GPT_find_distance_city_A_B_l1420_142088

-- Variables and givens
variable (D : ℝ)

-- Conditions from the problem
variable (JohnSpeed : ℝ := 40) (LewisSpeed : ℝ := 60)
variable (MeetDistance : ℝ := 160)
variable (TimeJohn : ℝ := (D - MeetDistance) / JohnSpeed)
variable (TimeLewis : ℝ := (D + MeetDistance) / LewisSpeed)

-- Lean 4 theorem statement for the proof
theorem find_distance_city_A_B :
  TimeJohn = TimeLewis → D = 800 :=
by
  sorry

end NUMINAMATH_GPT_find_distance_city_A_B_l1420_142088


namespace NUMINAMATH_GPT_smallest_integer_divisibility_conditions_l1420_142098

theorem smallest_integer_divisibility_conditions :
  ∃ n : ℕ, n > 0 ∧ (24 ∣ n^2) ∧ (900 ∣ n^3) ∧ (1024 ∣ n^4) ∧ n = 120 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_divisibility_conditions_l1420_142098


namespace NUMINAMATH_GPT_total_length_of_visible_edges_l1420_142021

theorem total_length_of_visible_edges (shortest_side : ℕ) (removed_side : ℕ) (longest_side : ℕ) (new_visible_sides_sum : ℕ) 
  (h1 : shortest_side = 4) 
  (h2 : removed_side = 2 * shortest_side) 
  (h3 : removed_side = longest_side / 2) 
  (h4 : longest_side = 16) 
  (h5 : new_visible_sides_sum = shortest_side + removed_side + removed_side) : 
  new_visible_sides_sum = 20 := by 
sorry

end NUMINAMATH_GPT_total_length_of_visible_edges_l1420_142021


namespace NUMINAMATH_GPT_problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l1420_142044

-- Define Lean goals for the true statements
theorem problem_part_a (x : ℝ) (h : x < 0) : x^3 < x := sorry
theorem problem_part_b (x : ℝ) (h : x^3 > 0) : x > 0 := sorry
theorem problem_part_e (x : ℝ) (h : x > 1) : x^3 > x := sorry

-- Disprove the false statements by showing the negation
theorem problem_part_c_disproof (x : ℝ) (h : x^3 < x) : ¬ (|x| > 1) := sorry
theorem problem_part_d_disproof (x : ℝ) (h : x^3 > x) : ¬ (x > 1) := sorry

end NUMINAMATH_GPT_problem_part_a_problem_part_b_problem_part_e_problem_part_c_disproof_problem_part_d_disproof_l1420_142044


namespace NUMINAMATH_GPT_find_number_l1420_142036

-- Define the problem statement
theorem find_number (n : ℕ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n) / 5 = 27) : n = 9 :=
sorry

end NUMINAMATH_GPT_find_number_l1420_142036


namespace NUMINAMATH_GPT_range_of_x_l1420_142004

noncomputable def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
noncomputable def specific_function (f : ℝ → ℝ) := ∀ x : ℝ, x ≥ 0 → f x = 2^x

theorem range_of_x (f : ℝ → ℝ)  
  (hf_even : even_function f) 
  (hf_specific : specific_function f) : {x : ℝ | f (1 - 2 * x) < f 3} = {x : ℝ | -1 < x ∧ x < 2} := 
by
  sorry

end NUMINAMATH_GPT_range_of_x_l1420_142004


namespace NUMINAMATH_GPT_perfect_square_trinomial_l1420_142072

theorem perfect_square_trinomial :
  120^2 - 40 * 120 + 20^2 = 10000 := sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l1420_142072


namespace NUMINAMATH_GPT_arc_length_of_sector_l1420_142099

noncomputable def central_angle := 36
noncomputable def radius := 15

theorem arc_length_of_sector : (central_angle * Real.pi * radius / 180 = 3 * Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l1420_142099


namespace NUMINAMATH_GPT_simplify_fraction_l1420_142053

theorem simplify_fraction :
  (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1420_142053


namespace NUMINAMATH_GPT_negation_example_l1420_142075

theorem negation_example : ¬ (∃ x : ℤ, x^2 + 2 * x + 1 ≤ 0) ↔ ∀ x : ℤ, x^2 + 2 * x + 1 > 0 := 
by 
  sorry

end NUMINAMATH_GPT_negation_example_l1420_142075


namespace NUMINAMATH_GPT_team_average_typing_speed_l1420_142074

-- Definitions of typing speeds of each team member
def typing_speed_rudy := 64
def typing_speed_joyce := 76
def typing_speed_gladys := 91
def typing_speed_lisa := 80
def typing_speed_mike := 89

-- Number of team members
def number_of_team_members := 5

-- Total typing speed calculation
def total_typing_speed := typing_speed_rudy + typing_speed_joyce + typing_speed_gladys + typing_speed_lisa + typing_speed_mike

-- Average typing speed calculation
def average_typing_speed := total_typing_speed / number_of_team_members

-- Theorem statement
theorem team_average_typing_speed : average_typing_speed = 80 := by
  sorry

end NUMINAMATH_GPT_team_average_typing_speed_l1420_142074


namespace NUMINAMATH_GPT_distinct_pairs_count_l1420_142022

theorem distinct_pairs_count :
  ∃ (n : ℕ), n = 2 ∧ 
  ∀ (x y : ℝ), (x = 3 * x^2 + y^2) ∧ (y = 3 * x * y) → 
    ((x = 0 ∧ y = 0) ∨ (x = 1 / 3 ∧ y = 0)) :=
by
  sorry

end NUMINAMATH_GPT_distinct_pairs_count_l1420_142022


namespace NUMINAMATH_GPT_A_3_2_eq_29_l1420_142013

-- Define the recursive function A(m, n).
def A : Nat → Nat → Nat
| 0, n => n + 1
| (m + 1), 0 => A m 1
| (m + 1), (n + 1) => A m (A (m + 1) n)

-- Prove that A(3, 2) = 29
theorem A_3_2_eq_29 : A 3 2 = 29 := by 
  sorry

end NUMINAMATH_GPT_A_3_2_eq_29_l1420_142013


namespace NUMINAMATH_GPT_marys_balloons_l1420_142055

theorem marys_balloons (x y : ℝ) (h1 : x = 4 * y) (h2 : x = 7.0) : y = 1.75 := by
  sorry

end NUMINAMATH_GPT_marys_balloons_l1420_142055


namespace NUMINAMATH_GPT_scientific_notation_of_0point0000025_l1420_142015

theorem scientific_notation_of_0point0000025 : ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * 10 ^ n ∧ a = 2.5 ∧ n = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_scientific_notation_of_0point0000025_l1420_142015


namespace NUMINAMATH_GPT_half_MN_correct_l1420_142009

noncomputable def OM : ℝ × ℝ := (-2, 3)
noncomputable def ON : ℝ × ℝ := (-1, -5)
noncomputable def MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
noncomputable def half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)

theorem half_MN_correct : half_MN = (1 / 2, -4) :=
by
  -- define the values of OM and ON
  let OM : ℝ × ℝ := (-2, 3)
  let ON : ℝ × ℝ := (-1, -5)
  -- calculate MN
  let MN : ℝ × ℝ := (ON.1 - OM.1, ON.2 - OM.2)
  -- calculate half of MN
  let half_MN : ℝ × ℝ := (MN.1 / 2, MN.2 / 2)
  -- assert the expected value
  exact sorry

end NUMINAMATH_GPT_half_MN_correct_l1420_142009


namespace NUMINAMATH_GPT_greatest_sum_of_consecutive_odd_integers_lt_500_l1420_142012

-- Define the consecutive odd integers and their conditions
def consecutive_odd_integers (n : ℤ) : Prop :=
  n % 2 = 1 ∧ (n + 2) % 2 = 1

-- Define the condition that their product must be less than 500
def prod_less_500 (n : ℤ) : Prop :=
  n * (n + 2) < 500

-- The theorem statement
theorem greatest_sum_of_consecutive_odd_integers_lt_500 : 
  ∃ n : ℤ, consecutive_odd_integers n ∧ prod_less_500 n ∧ ∀ m : ℤ, consecutive_odd_integers m ∧ prod_less_500 m → n + (n + 2) ≥ m + (m + 2) :=
sorry

end NUMINAMATH_GPT_greatest_sum_of_consecutive_odd_integers_lt_500_l1420_142012


namespace NUMINAMATH_GPT_area_ratio_XYZ_PQR_l1420_142003

theorem area_ratio_XYZ_PQR 
  (PR PQ QR : ℝ)
  (p q r : ℝ) 
  (hPR : PR = 15) 
  (hPQ : PQ = 20) 
  (hQR : QR = 25)
  (hPX : p * PR = PR * p)
  (hQY : q * QR = QR * q) 
  (hPZ : r * PQ = PQ * r) 
  (hpq_sum : p + q + r = 3 / 4) 
  (hpq_sq_sum : p^2 + q^2 + r^2 = 9 / 16) : 
  (area_triangle_XYZ / area_triangle_PQR = 1 / 4) :=
sorry

end NUMINAMATH_GPT_area_ratio_XYZ_PQR_l1420_142003


namespace NUMINAMATH_GPT_numMilkmen_rented_pasture_l1420_142087

def cowMonths (cows: ℕ) (months: ℕ) : ℕ := cows * months

def totalCowMonths (a: ℕ) (b: ℕ) (c: ℕ) (d: ℕ) : ℕ := a + b + c + d

noncomputable def rentPerCowMonth (share: ℕ) (cowMonths: ℕ) : ℕ := 
  share / cowMonths

theorem numMilkmen_rented_pasture 
  (a_cows: ℕ) (a_months: ℕ) (b_cows: ℕ) (b_months: ℕ) (c_cows: ℕ) (c_months: ℕ) (d_cows: ℕ) (d_months: ℕ)
  (a_share: ℕ) (total_rent: ℕ) 
  (ha: a_cows = 24) (hma: a_months = 3) 
  (hb: b_cows = 10) (hmb: b_months = 5)
  (hc: c_cows = 35) (hmc: c_months = 4)
  (hd: d_cows = 21) (hmd: d_months = 3)
  (ha_share: a_share = 720) (htotal_rent: total_rent = 3250)
  : 4 = 4 := by
  sorry

end NUMINAMATH_GPT_numMilkmen_rented_pasture_l1420_142087


namespace NUMINAMATH_GPT_box_prices_l1420_142089

theorem box_prices (a b c : ℝ) 
  (h1 : a + b + c = 9) 
  (h2 : 3 * a + 2 * b + c = 16) : 
  c - a = 2 := 
by 
  sorry

end NUMINAMATH_GPT_box_prices_l1420_142089


namespace NUMINAMATH_GPT_expected_score_two_free_throws_is_correct_l1420_142030

noncomputable def expected_score_two_free_throws (p : ℝ) (n : ℕ) : ℝ :=
n * p

theorem expected_score_two_free_throws_is_correct : expected_score_two_free_throws 0.7 2 = 1.4 :=
by
  -- Proof will be written here.
  sorry

end NUMINAMATH_GPT_expected_score_two_free_throws_is_correct_l1420_142030


namespace NUMINAMATH_GPT_tens_digit_N_to_20_l1420_142045

theorem tens_digit_N_to_20 (N : ℕ) (h1 : Even N) (h2 : ¬(∃ k : ℕ, N = 10 * k)) : 
  ((N ^ 20) / 10) % 10 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_tens_digit_N_to_20_l1420_142045


namespace NUMINAMATH_GPT_solution_set_f_le_1_l1420_142064

variable {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def monotone_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x y⦄, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

theorem solution_set_f_le_1 :
  is_even_function f →
  monotone_on_nonneg f →
  f (-2) = 1 →
  {x : ℝ | f x ≤ 1} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} :=
by
  intros h_even h_mono h_f_neg_2
  sorry

end NUMINAMATH_GPT_solution_set_f_le_1_l1420_142064


namespace NUMINAMATH_GPT_thomas_lost_pieces_l1420_142077

theorem thomas_lost_pieces (audrey_lost : ℕ) (total_pieces_left : ℕ) (initial_pieces_each : ℕ) (total_pieces_initial : ℕ) (audrey_remaining_pieces : ℕ) (thomas_remaining_pieces : ℕ) : 
  audrey_lost = 6 → total_pieces_left = 21 → initial_pieces_each = 16 → total_pieces_initial = 32 → 
  audrey_remaining_pieces = initial_pieces_each - audrey_lost → 
  thomas_remaining_pieces = total_pieces_left - audrey_remaining_pieces → 
  initial_pieces_each - thomas_remaining_pieces = 5 :=
by
  sorry

end NUMINAMATH_GPT_thomas_lost_pieces_l1420_142077


namespace NUMINAMATH_GPT_one_third_of_five_times_seven_l1420_142026

theorem one_third_of_five_times_seven:
  (1/3 : ℝ) * (5 * 7) = 35 / 3 := 
by
  -- Definitions and calculations go here
  sorry

end NUMINAMATH_GPT_one_third_of_five_times_seven_l1420_142026


namespace NUMINAMATH_GPT_soccer_balls_are_20_l1420_142052

variable (S : ℕ)
variable (num_baseballs : ℕ) (num_volleyballs : ℕ)
variable (condition_baseballs : num_baseballs = 5 * S)
variable (condition_volleyballs : num_volleyballs = 3 * S)
variable (condition_total : num_baseballs + num_volleyballs = 160)

theorem soccer_balls_are_20 :
  S = 20 :=
by
  sorry

end NUMINAMATH_GPT_soccer_balls_are_20_l1420_142052


namespace NUMINAMATH_GPT_triangle_A1B1C1_sides_l1420_142019

theorem triangle_A1B1C1_sides
  (a b c x y z R : ℝ) 
  (h_positive_a : a > 0)
  (h_positive_b : b > 0)
  (h_positive_c : c > 0)
  (h_positive_x : x > 0)
  (h_positive_y : y > 0)
  (h_positive_z : z > 0)
  (h_positive_R : R > 0) :
  (↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R), ↑c * ↑x / (2 * ↑R)) = (↑c * ↑x / (2 * ↑R), ↑a * ↑y / (2 * ↑R), ↑b * ↑z / (2 * ↑R)) :=
by sorry

end NUMINAMATH_GPT_triangle_A1B1C1_sides_l1420_142019


namespace NUMINAMATH_GPT_solve_for_a_l1420_142079

theorem solve_for_a (x a : ℝ) (h : 3 * x + 2 * a = 3) (hx : x = 5) : a = -6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l1420_142079


namespace NUMINAMATH_GPT_degree_of_k_l1420_142040

open Polynomial

theorem degree_of_k (h k : Polynomial ℝ) 
  (h_def : h = -5 * X^5 + 4 * X^3 - 2 * X^2 + C 8)
  (deg_sum : (h + k).degree = 2) : k.degree = 5 :=
sorry

end NUMINAMATH_GPT_degree_of_k_l1420_142040


namespace NUMINAMATH_GPT_bill_due_months_l1420_142041

theorem bill_due_months {TD A: ℝ} (R: ℝ) : 
  TD = 189 → A = 1764 → R = 16 → 
  ∃ M: ℕ, A - TD * (1 + (R/100) * (M/12)) = 1764 - 189 * (1 + (16/100) * (10/12)) ∧ M = 10 :=
by
  intro hTD hA hR
  use 10
  sorry

end NUMINAMATH_GPT_bill_due_months_l1420_142041
