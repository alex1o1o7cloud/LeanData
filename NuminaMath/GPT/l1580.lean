import Mathlib

namespace sin_2pi_minus_alpha_l1580_158032

theorem sin_2pi_minus_alpha (α : ℝ) (h₁ : Real.cos (α + Real.pi) = Real.sqrt 3 / 2) (h₂ : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
    Real.sin (2 * Real.pi - α) = -1 / 2 := 
sorry

end sin_2pi_minus_alpha_l1580_158032


namespace coordinates_of_point_in_fourth_quadrant_l1580_158019

-- Define the conditions as separate hypotheses
def point_in_fourth_quadrant (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- State the main theorem
theorem coordinates_of_point_in_fourth_quadrant
  (x y : ℝ) (h1 : point_in_fourth_quadrant x y) (h2 : |x| = 3) (h3 : |y| = 5) :
  (x = 3) ∧ (y = -5) :=
by
  sorry

end coordinates_of_point_in_fourth_quadrant_l1580_158019


namespace restaurant_A2_probability_l1580_158021

noncomputable def prob_A2 (P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 : ℝ) : ℝ :=
  P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1

theorem restaurant_A2_probability :
  let P_A1 := 0.4
  let P_B1 := 0.6
  let P_A2_given_A1 := 0.6
  let P_A2_given_B1 := 0.5
  prob_A2 P_A1 P_B1 P_A2_given_A1 P_A2_given_B1 = 0.54 :=
by
  sorry

end restaurant_A2_probability_l1580_158021


namespace correct_operation_l1580_158007

variable (a : ℕ)

theorem correct_operation :
  (3 * a + 2 * a ≠ 5 * a^2) ∧
  (3 * a - 2 * a ≠ 1) ∧
  a^2 * a^3 = a^5 ∧
  (a / a^2 ≠ a) :=
by
  sorry

end correct_operation_l1580_158007


namespace days_to_finish_together_l1580_158022

-- Define the work rate of B
def work_rate_B : ℚ := 1 / 12

-- Define the work rate of A
def work_rate_A : ℚ := 2 * work_rate_B

-- Combined work rate of A and B
def combined_work_rate : ℚ := work_rate_A + work_rate_B

-- Prove that the number of days required for A and B to finish the work together is 4
theorem days_to_finish_together : (1 / combined_work_rate) = 4 := 
by
  sorry

end days_to_finish_together_l1580_158022


namespace fg_sum_at_2_l1580_158086

noncomputable def f (x : ℚ) : ℚ := (5 * x^3 + 4 * x^2 - 2 * x + 3) / (x^3 - 2 * x^2 + 3 * x + 1)
noncomputable def g (x : ℚ) : ℚ := x^2 - 2

theorem fg_sum_at_2 : f (g 2) + g (f 2) = 468 / 7 := by
  sorry

end fg_sum_at_2_l1580_158086


namespace nine_fifths_sum_l1580_158011

open Real

theorem nine_fifths_sum (a b: ℝ) (ha: a > 0) (hb: b > 0)
    (h1: a * (sqrt a) + b * (sqrt b) = 183) 
    (h2: a * (sqrt b) + b * (sqrt a) = 182) : 
    9 / 5 * (a + b) = 657 := 
by 
    sorry

end nine_fifths_sum_l1580_158011


namespace minimize_m_at_l1580_158017

noncomputable def m (x y : ℝ) : ℝ := 4 * x ^ 2 - 12 * x * y + 10 * y ^ 2 + 4 * y + 9

theorem minimize_m_at (x y : ℝ) : m x y = 5 ↔ (x = -3 ∧ y = -2) := 
sorry

end minimize_m_at_l1580_158017


namespace sqrt_prod_simplified_l1580_158055

open Real

variable (x : ℝ)

theorem sqrt_prod_simplified (hx : 0 ≤ x) : sqrt (50 * x) * sqrt (18 * x) * sqrt (8 * x) = 30 * x * sqrt (2 * x) :=
by
  sorry

end sqrt_prod_simplified_l1580_158055


namespace truth_of_q_l1580_158016

variable {p q : Prop}

theorem truth_of_q (hnp : ¬ p) (hpq : p ∨ q) : q :=
  by
  sorry

end truth_of_q_l1580_158016


namespace hall_breadth_is_12_l1580_158045

/-- Given a hall with length 15 meters, if the sum of the areas of the floor and the ceiling 
    is equal to the sum of the areas of the four walls and the volume of the hall is 1200 
    cubic meters, then the breadth of the hall is 12 meters. -/
theorem hall_breadth_is_12 (b h : ℝ) (h1 : 15 * b * h = 1200)
  (h2 : 2 * (15 * b) = 2 * (15 * h) + 2 * (b * h)) : b = 12 :=
sorry

end hall_breadth_is_12_l1580_158045


namespace partners_count_l1580_158023

theorem partners_count (P A : ℕ) (h1 : P / A = 2 / 63) (h2 : P / (A + 50) = 1 / 34) : P = 20 :=
sorry

end partners_count_l1580_158023


namespace A_n_plus_B_n_eq_2n_cubed_l1580_158033

-- Definition of A_n given the grouping of positive integers
def A_n (n : ℕ) : ℕ :=
  let sum_first_n_squared := n * n * (n * n + 1) / 2
  let sum_first_n_minus_1_squared := (n - 1) * (n - 1) * ((n - 1) * (n - 1) + 1) / 2
  sum_first_n_squared - sum_first_n_minus_1_squared

-- Definition of B_n given the array of cubes of natural numbers
def B_n (n : ℕ) : ℕ := n * n * n - (n - 1) * (n - 1) * (n - 1)

-- The theorem to prove that A_n + B_n = 2n^3
theorem A_n_plus_B_n_eq_2n_cubed (n : ℕ) : A_n n + B_n n = 2 * n^3 := by
  sorry

end A_n_plus_B_n_eq_2n_cubed_l1580_158033


namespace fraction_distinctly_marked_l1580_158001

theorem fraction_distinctly_marked 
  (area_large_rectangle : ℕ)
  (fraction_shaded : ℚ)
  (fraction_further_marked : ℚ)
  (h_area_large_rectangle : area_large_rectangle = 15 * 24)
  (h_fraction_shaded : fraction_shaded = 1/3)
  (h_fraction_further_marked : fraction_further_marked = 1/2) :
  (fraction_further_marked * fraction_shaded = 1/6) :=
by
  sorry

end fraction_distinctly_marked_l1580_158001


namespace calculate_b_50_l1580_158066

def sequence_b : ℕ → ℤ
| 0 => sorry -- This case is not used.
| 1 => 3
| (n + 2) => sequence_b (n + 1) + 3 * (n + 1) + 1

theorem calculate_b_50 : sequence_b 50 = 3727 := 
by
    sorry

end calculate_b_50_l1580_158066


namespace sphere_surface_area_ratio_l1580_158099

theorem sphere_surface_area_ratio (V1 V2 : ℝ) (h1 : V1 = (4 / 3) * π * (r1^3))
  (h2 : V2 = (4 / 3) * π * (r2^3)) (h3 : V1 / V2 = 1 / 27) :
  (4 * π * r1^2) / (4 * π * r2^2) = 1 / 9 := 
sorry

end sphere_surface_area_ratio_l1580_158099


namespace robert_total_interest_l1580_158061

theorem robert_total_interest
  (inheritance : ℕ)
  (part1 part2 : ℕ)
  (rate1 rate2 : ℝ)
  (time : ℝ) :
  inheritance = 4000 →
  part2 = 1800 →
  part1 = inheritance - part2 →
  rate1 = 0.05 →
  rate2 = 0.065 →
  time = 1 →
  (part1 * rate1 * time + part2 * rate2 * time) = 227 :=
by
  intros
  sorry

end robert_total_interest_l1580_158061


namespace range_of_m_l1580_158054

/-- The quadratic equation x^2 + (2m - 1)x + 4 - 2m = 0 has one root 
greater than 2 and the other less than 2 if and only if m < -3. -/
theorem range_of_m (m : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1 ^ 2 + (2 * m - 1) * x1 + 4 - 2 * m = 0 ∧
    x2 ^ 2 + (2 * m - 1) * x2 + 4 - 2 * m = 0) ↔
    m < -3 := by
  sorry

end range_of_m_l1580_158054


namespace gray_area_l1580_158002

def center_C : (ℝ × ℝ) := (6, 5)
def center_D : (ℝ × ℝ) := (14, 5)
def radius_C : ℝ := 3
def radius_D : ℝ := 3

theorem gray_area :
  let area_rectangle := 8 * 5
  let area_sector_C := (1 / 2) * π * radius_C^2
  let area_sector_D := (1 / 2) * π * radius_D^2
  area_rectangle - (area_sector_C + area_sector_D) = 40 - 9 * π :=
by
  sorry

end gray_area_l1580_158002


namespace probability_of_perfect_square_is_correct_l1580_158076

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def probability_perfect_square (p : ℚ) : ℚ :=
  let less_than_equal_60 := 7 * p
  let greater_than_60 := 4 * 4 * p
  less_than_equal_60 + greater_than_60

theorem probability_of_perfect_square_is_correct :
  let p : ℚ := 1 / 300
  probability_perfect_square p = 23 / 300 :=
sorry

end probability_of_perfect_square_is_correct_l1580_158076


namespace area_of_triangle_AMN_is_correct_l1580_158029

noncomputable def area_triangle_AMN : ℝ :=
  let A := (120 + 56 * Real.sqrt 3) / 3
  let M := (12 + 20 * Real.sqrt 3) / 3
  let N := 4 * Real.sqrt 3 + 20
  (A * N) / 2

theorem area_of_triangle_AMN_is_correct :
  area_triangle_AMN = (224 * Real.sqrt 3 + 240) / 3 := sorry

end area_of_triangle_AMN_is_correct_l1580_158029


namespace remainder_5n_minus_12_l1580_158062

theorem remainder_5n_minus_12 (n : ℤ) (hn : n % 9 = 4) : (5 * n - 12) % 9 = 8 := 
by sorry

end remainder_5n_minus_12_l1580_158062


namespace radii_of_circles_l1580_158072

theorem radii_of_circles
  (r s : ℝ)
  (h_ratio : r / s = 9 / 4)
  (h_right_triangle : ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2)
  (h_tangent : (r + s)^2 = (r - s)^2 + 12^2) :
   r = 20 / 47 ∧ s = 45 / 47 :=
by
  sorry

end radii_of_circles_l1580_158072


namespace solve_for_x_l1580_158098

theorem solve_for_x (x : ℝ) (h : (x^2 - 36) / 3 = (x^2 + 3 * x + 9) / 6) : x = 9 ∨ x = -9 := 
by 
  sorry

end solve_for_x_l1580_158098


namespace total_amount_of_money_l1580_158040

theorem total_amount_of_money (P1 : ℝ) (interest_total : ℝ)
  (hP1 : P1 = 299.99999999999994) (hInterest : interest_total = 144) :
  ∃ T : ℝ, T = 3000 :=
by
  sorry

end total_amount_of_money_l1580_158040


namespace jordan_rectangle_length_l1580_158064

variables (L : ℝ)

-- Condition: Carol's rectangle measures 12 inches by 15 inches.
def carol_area : ℝ := 12 * 15

-- Condition: Jordan's rectangle has the same area as Carol's rectangle.
def jordan_area : ℝ := carol_area

-- Condition: Jordan's rectangle is 20 inches wide.
def jordan_width : ℝ := 20

-- Proposition: Length of Jordan's rectangle == 9 inches.
theorem jordan_rectangle_length : L * jordan_width = jordan_area → L = 9 := 
by
  intros h
  sorry

end jordan_rectangle_length_l1580_158064


namespace ratio_is_7_to_10_l1580_158037

-- Given conditions in the problem translated to Lean definitions
def snakes : ℕ := 100
def arctic_foxes : ℕ := 80
def leopards : ℕ := 20
def bee_eaters : ℕ := 10 * leopards
def alligators : ℕ := 2 * (arctic_foxes + leopards)
def total_animals : ℕ := 670
def other_animals : ℕ := snakes + arctic_foxes + leopards + bee_eaters + alligators
def cheetahs : ℕ := total_animals - other_animals

-- The ratio of cheetahs to snakes to be proven
def ratio_cheetahs_to_snakes (cheetahs snakes : ℕ) : ℚ := cheetahs / snakes

theorem ratio_is_7_to_10 : ratio_cheetahs_to_snakes cheetahs snakes = 7 / 10 :=
by
  sorry

end ratio_is_7_to_10_l1580_158037


namespace card_deck_initial_count_l1580_158079

theorem card_deck_initial_count 
  (r b : ℕ)
  (h1 : r / (r + b) = 1 / 4)
  (h2 : r / (r + (b + 6)) = 1 / 5) : 
  r + b = 24 :=
by
  sorry

end card_deck_initial_count_l1580_158079


namespace GCD_of_n_pow_13_sub_n_l1580_158038

theorem GCD_of_n_pow_13_sub_n :
  ∀ n : ℤ, gcd (n^13 - n) 2730 = gcd (n^13 - n) n := sorry

end GCD_of_n_pow_13_sub_n_l1580_158038


namespace find_N_l1580_158070

theorem find_N (N x : ℝ) (h1 : N / (1 + 4 / x) = 1) (h2 : x = 0.5) : N = 9 := 
by 
  sorry

end find_N_l1580_158070


namespace largest_value_of_number_l1580_158088

theorem largest_value_of_number 
  (v w x y z : ℝ)
  (h1 : v + w + x + y + z = 8)
  (h2 : v^2 + w^2 + x^2 + y^2 + z^2 = 16) :
  ∃ (m : ℝ), m = 2.4 ∧ (m = v ∨ m = w ∨ m = x ∨ m = y ∨ m = z) :=
sorry

end largest_value_of_number_l1580_158088


namespace sandy_initial_payment_l1580_158060

theorem sandy_initial_payment (P : ℝ) (repairs cost: ℝ) (selling_price gain: ℝ) 
  (hc : repairs = 300)
  (hs : selling_price = 1260) 
  (hg : gain = 5)
  (h : selling_price = (P + repairs) * (1 + gain / 100)) : 
  P = 900 :=
sorry

end sandy_initial_payment_l1580_158060


namespace range_of_a_l1580_158027

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 1 else a * x^2 - x + 2

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ -1) ↔ (a ≥ 1/12) :=
by
  sorry

end range_of_a_l1580_158027


namespace gcd_positive_ints_l1580_158080

theorem gcd_positive_ints (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (hdiv : (a^2 + b^2) ∣ (a * c + b * d)) : 
  Nat.gcd (c^2 + d^2) (a^2 + b^2) > 1 := 
sorry

end gcd_positive_ints_l1580_158080


namespace train_speed_l1580_158095

theorem train_speed (distance_meters : ℝ) (time_seconds : ℝ) :
  distance_meters = 180 →
  time_seconds = 17.998560115190784 →
  ((distance_meters / 1000) / (time_seconds / 3600)) = 36.00360072014403 :=
by 
  intros h1 h2
  rw [h1, h2]
  sorry

end train_speed_l1580_158095


namespace product_of_two_numbers_l1580_158004

theorem product_of_two_numbers (a b : ℕ) (H1 : Nat.gcd a b = 20) (H2 : Nat.lcm a b = 128) : a * b = 2560 :=
by
  sorry

end product_of_two_numbers_l1580_158004


namespace flat_fee_first_night_l1580_158096

-- Given conditions
variable (f n : ℝ)
axiom alice_cost : f + 3 * n = 245
axiom bob_cost : f + 5 * n = 350

-- Main theorem to prove
theorem flat_fee_first_night : f = 87.5 := by sorry

end flat_fee_first_night_l1580_158096


namespace blue_marbles_count_l1580_158073

theorem blue_marbles_count
  (total_marbles : ℕ)
  (yellow_marbles : ℕ)
  (red_marbles : ℕ)
  (blue_marbles : ℕ)
  (yellow_probability : ℚ)
  (total_marbles_eq : yellow_marbles = 6)
  (yellow_probability_eq : yellow_probability = 1 / 4)
  (red_marbles_eq : red_marbles = 11)
  (total_marbles_def : total_marbles = yellow_marbles * 4)
  (blue_marbles_def : blue_marbles = total_marbles - red_marbles - yellow_marbles) :
  blue_marbles = 7 :=
sorry

end blue_marbles_count_l1580_158073


namespace calc_result_l1580_158003

theorem calc_result : (377 / 13 / 29 * 1 / 4 / 2) = 0.125 := 
by sorry

end calc_result_l1580_158003


namespace alice_paper_cranes_l1580_158044

theorem alice_paper_cranes : 
  ∀ (total : ℕ) (half : ℕ) (one_fifth : ℕ) (thirty_percent : ℕ),
    total = 1000 →
    half = total / 2 →
    one_fifth = (total - half) / 5 →
    thirty_percent = ((total - half) - one_fifth) * 3 / 10 →
    total - (half + one_fifth + thirty_percent) = 280 :=
by
  intros total half one_fifth thirty_percent h_total h_half h_one_fifth h_thirty_percent
  sorry

end alice_paper_cranes_l1580_158044


namespace cookie_revenue_l1580_158059

theorem cookie_revenue :
  let robyn_day1_packs := 25
  let robyn_day1_price := 4.0
  let lucy_day1_packs := 17
  let lucy_day1_price := 5.0
  let robyn_day2_packs := 15
  let robyn_day2_price := 3.5
  let lucy_day2_packs := 9
  let lucy_day2_price := 4.5
  let robyn_day3_packs := 23
  let robyn_day3_price := 4.5
  let lucy_day3_packs := 20
  let lucy_day3_price := 3.5
  let robyn_day1_revenue := robyn_day1_packs * robyn_day1_price
  let lucy_day1_revenue := lucy_day1_packs * lucy_day1_price
  let robyn_day2_revenue := robyn_day2_packs * robyn_day2_price
  let lucy_day2_revenue := lucy_day2_packs * lucy_day2_price
  let robyn_day3_revenue := robyn_day3_packs * robyn_day3_price
  let lucy_day3_revenue := lucy_day3_packs * lucy_day3_price
  let robyn_total_revenue := robyn_day1_revenue + robyn_day2_revenue + robyn_day3_revenue
  let lucy_total_revenue := lucy_day1_revenue + lucy_day2_revenue + lucy_day3_revenue
  let total_revenue := robyn_total_revenue + lucy_total_revenue
  total_revenue = 451.5 := 
by
  sorry

end cookie_revenue_l1580_158059


namespace estimated_white_balls_is_correct_l1580_158034

-- Define the total number of balls
def total_balls : ℕ := 10

-- Define the number of trials
def trials : ℕ := 100

-- Define the number of times a red ball is drawn
def red_draws : ℕ := 80

-- Define the function to estimate the number of red balls based on the frequency
def estimated_red_balls (total_balls : ℕ) (red_draws : ℕ) (trials : ℕ) : ℕ :=
  total_balls * red_draws / trials

-- Define the function to estimate the number of white balls
def estimated_white_balls (total_balls : ℕ) (estimated_red_balls : ℕ) : ℕ :=
  total_balls - estimated_red_balls

-- State the theorem to prove the estimated number of white balls
theorem estimated_white_balls_is_correct : 
  estimated_white_balls total_balls (estimated_red_balls total_balls red_draws trials) = 2 :=
by
  sorry

end estimated_white_balls_is_correct_l1580_158034


namespace sum_of_perimeters_triangles_l1580_158049

theorem sum_of_perimeters_triangles (a : ℕ → ℕ) (side_length : ℕ) (P : ℕ → ℕ):
  (∀ n : ℕ, a 0 = side_length ∧ P 0 = 3 * a 0) →
  (∀ n : ℕ, a (n + 1) = a n / 2 ∧ P (n + 1) = 3 * a (n + 1)) →
  (side_length = 45) →
  ∑' n, P n = 270 :=
by
  -- the proof would continue here
  sorry

end sum_of_perimeters_triangles_l1580_158049


namespace original_volume_l1580_158075

variable (V : ℝ)

theorem original_volume (h1 : (1/4) * V = V₁)
                       (h2 : (1/4) * V₁ = V₂)
                       (h3 : (1/3) * V₂ = 0.4) : 
                       V = 19.2 := 
by 
  sorry

end original_volume_l1580_158075


namespace product_of_two_numbers_l1580_158071

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 27) (h2 : x - y = 9) : x * y = 162 := 
by {
  sorry
}

end product_of_two_numbers_l1580_158071


namespace candy_vs_chocolate_l1580_158042

theorem candy_vs_chocolate
  (candy1 candy2 chocolate : ℕ)
  (h1 : candy1 = 38)
  (h2 : candy2 = 36)
  (h3 : chocolate = 16) :
  (candy1 + candy2) - chocolate = 58 :=
by
  sorry

end candy_vs_chocolate_l1580_158042


namespace units_digit_of_result_is_eight_l1580_158056

def three_digit_number_reverse_subtract (a b c : ℕ) : ℕ :=
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  original - reversed

theorem units_digit_of_result_is_eight (a b c : ℕ) (h : a = c + 2) :
  (three_digit_number_reverse_subtract a b c) % 10 = 8 :=
by
  sorry

end units_digit_of_result_is_eight_l1580_158056


namespace trigonometric_inequalities_l1580_158091

theorem trigonometric_inequalities (θ : ℝ) (h1 : Real.sin (θ + Real.pi) < 0) (h2 : Real.cos (θ - Real.pi) > 0) : 
  Real.sin θ > 0 ∧ Real.cos θ < 0 :=
sorry

end trigonometric_inequalities_l1580_158091


namespace probability_rain_at_most_3_days_in_july_l1580_158030

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_rain_at_most_3_days_in_july :
  let p := 1 / 5
  let n := 31
  let sum_prob := binomial_probability n 0 p + binomial_probability n 1 p + binomial_probability n 2 p + binomial_probability n 3 p
  abs (sum_prob - 0.125) < 0.001 :=
by
  sorry

end probability_rain_at_most_3_days_in_july_l1580_158030


namespace no_monochromatic_10_term_progression_l1580_158025

def can_color_without_monochromatic_progression (n k : ℕ) (c : Fin n → Fin k) : Prop :=
  ∀ (a d : ℕ), (a < n) → (a + (9 * d) < n) → (∀ i : ℕ, i < 10 → c ⟨a + (i * d), sorry⟩ = c ⟨a, sorry⟩) → 
    (∃ j i : ℕ, j < 10 ∧ i < 10 ∧ c ⟨a + (i * d), sorry⟩ ≠ c ⟨a + (j * d), sorry⟩)

theorem no_monochromatic_10_term_progression :
  ∃ c : Fin 2008 → Fin 4, can_color_without_monochromatic_progression 2008 4 c :=
sorry

end no_monochromatic_10_term_progression_l1580_158025


namespace tangent_line_y_intercept_at_P_1_12_is_9_l1580_158028

noncomputable def curve (x : ℝ) : ℝ := x^3 + 11

noncomputable def tangent_slope_at (x : ℝ) : ℝ := 3 * x^2

noncomputable def tangent_line_y_intercept : ℝ :=
  let P : ℝ × ℝ := (1, curve 1)
  let slope := tangent_slope_at 1
  P.snd - slope * P.fst

theorem tangent_line_y_intercept_at_P_1_12_is_9 :
  tangent_line_y_intercept = 9 :=
sorry

end tangent_line_y_intercept_at_P_1_12_is_9_l1580_158028


namespace remainder_of_37_div_8_is_5_l1580_158020

theorem remainder_of_37_div_8_is_5 : ∃ A B : ℤ, 37 = 8 * A + B ∧ 0 ≤ B ∧ B < 8 ∧ B = 5 := 
by
  sorry

end remainder_of_37_div_8_is_5_l1580_158020


namespace smallest_bottles_needed_l1580_158083

/-- Christine needs at least 60 fluid ounces of milk, the store sells milk in 250 milliliter bottles,
and there are 32 fluid ounces in 1 liter. The smallest number of bottles Christine should purchase
is 8. -/
theorem smallest_bottles_needed
  (fl_oz_needed : ℕ := 60)
  (ml_per_bottle : ℕ := 250)
  (fl_oz_per_liter : ℕ := 32) :
  let liters_needed := fl_oz_needed / fl_oz_per_liter
  let ml_needed := liters_needed * 1000
  let bottles := (ml_needed + ml_per_bottle - 1) / ml_per_bottle
  bottles = 8 :=
by
  sorry

end smallest_bottles_needed_l1580_158083


namespace necessary_and_sufficient_condition_l1580_158052

theorem necessary_and_sufficient_condition (p q : Prop) 
  (hpq : p → q) (hqp : q → p) : 
  (p ↔ q) :=
by 
  sorry

end necessary_and_sufficient_condition_l1580_158052


namespace average_employees_per_week_l1580_158026

theorem average_employees_per_week (x : ℝ)
  (h1 : ∀ (x : ℝ), ∃ y : ℝ, y = x + 200)
  (h2 : ∀ (x : ℝ), ∃ z : ℝ, z = x + 150)
  (h3 : ∀ (x : ℝ), ∃ w : ℝ, w = 2 * (x + 150))
  (h4 : ∀ (w : ℝ), w = 400) :
  (250 + 50 + 200 + 400) / 4 = 225 :=
by 
  sorry

end average_employees_per_week_l1580_158026


namespace ellipse_equation_line_AC_l1580_158090

noncomputable def ellipse_eq (x y a b : ℝ) : Prop := 
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def foci_distance (a c : ℝ) : Prop := 
  a - c = 1 ∧ a + c = 3

noncomputable def b_value (a c b : ℝ) : Prop :=
  b = Real.sqrt (a^2 - c^2)

noncomputable def rhombus_on_line (m : ℝ) : Prop := 
  7 * (2 * m / 7) + 1 - 7 * (3 * m / 7) = 0

theorem ellipse_equation (a b c : ℝ) (h1 : foci_distance a c) (h2 : b_value a c b) :
  ellipse_eq x y a b :=
sorry

theorem line_AC (a b c x y x1 y1 x2 y2 : ℝ) 
  (h1 : ellipse_eq x1 y1 a b)
  (h2 : ellipse_eq x2 y2 a b)
  (h3 : 7 * x1 - 7 * y1 + 1 = 0)
  (h4 : 7 * x2 - 7 * y2 + 1 = 0)
  (h5 : rhombus_on_line y) :
  x + y + 1 = 0 :=
sorry

end ellipse_equation_line_AC_l1580_158090


namespace eccentricity_of_ellipse_l1580_158094

variables {a b c e : ℝ}

-- Definition of geometric progression condition for the ellipse axes and focal length
def geometric_progression_condition (a b c : ℝ) : Prop :=
  (2 * b) ^ 2 = 2 * c * 2 * a

-- Eccentricity calculation
def eccentricity {a c : ℝ} (e : ℝ) : Prop :=
  e = (a^2 - c^2) / a^2

-- Theorem that states the eccentricity under the given condition
theorem eccentricity_of_ellipse (h : geometric_progression_condition a b c) : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_ellipse_l1580_158094


namespace find_a_range_for_two_distinct_roots_l1580_158068

def f (x : ℝ) : ℝ := x^3 - 3 * x + 5

theorem find_a_range_for_two_distinct_roots :
  ∀ (a : ℝ), 3 ≤ a ∧ a ≤ 7 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = a ∧ f x2 = a :=
by
  -- The proof will be here
  sorry

end find_a_range_for_two_distinct_roots_l1580_158068


namespace total_books_in_week_l1580_158000

def books_read (n : ℕ) : ℕ :=
  if n = 0 then 2 -- day 1 (indexed by 0)
  else if n = 1 then 2 -- day 2
  else 2 + n -- starting from day 3 (indexed by 2)

-- Summing the books read from day 1 to day 7 (indexed from 0 to 6)
theorem total_books_in_week : (List.sum (List.map books_read [0, 1, 2, 3, 4, 5, 6])) = 29 := by
  sorry

end total_books_in_week_l1580_158000


namespace acme_vowel_soup_sequences_l1580_158014

-- Define the vowels and their frequencies
def vowels : List (Char × ℕ) := [('A', 6), ('E', 6), ('I', 6), ('O', 4), ('U', 4)]

-- Noncomputable definition to calculate the total number of sequences
noncomputable def number_of_sequences : ℕ :=
  let single_vowel_choices := 6 + 6 + 6 + 4 + 4
  single_vowel_choices^5

-- Theorem stating the number of five-letter sequences
theorem acme_vowel_soup_sequences : number_of_sequences = 11881376 := by
  sorry

end acme_vowel_soup_sequences_l1580_158014


namespace symmetry_axis_l1580_158050

noncomputable def y_func (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 4)

theorem symmetry_axis : ∃ a : ℝ, (∀ x : ℝ, y_func (a - x) = y_func (a + x)) ∧ a = Real.pi / 8 :=
by
  sorry

end symmetry_axis_l1580_158050


namespace negation_example_l1580_158063

variable (x : ℤ)

theorem negation_example : (¬ ∀ x : ℤ, |x| ≠ 3) ↔ (∃ x : ℤ, |x| = 3) :=
by
  sorry

end negation_example_l1580_158063


namespace remaining_problems_l1580_158039

-- Define the conditions
def worksheets_total : ℕ := 15
def worksheets_graded : ℕ := 7
def problems_per_worksheet : ℕ := 3

-- Define the proof goal
theorem remaining_problems : (worksheets_total - worksheets_graded) * problems_per_worksheet = 24 :=
by
  sorry

end remaining_problems_l1580_158039


namespace solve_for_diamond_l1580_158036

theorem solve_for_diamond (d : ℕ) (h1 : d * 9 + 6 = d * 10 + 3) (h2 : d < 10) : d = 3 :=
by
  sorry

end solve_for_diamond_l1580_158036


namespace circle_radius_squared_l1580_158018

-- Let r be the radius of the circle.
-- Let AB and CD be chords of the circle with lengths 10 and 7 respectively.
-- Let the extensions of AB and CD intersect at a point P outside the circle.
-- Let ∠APD be 60 degrees.
-- Let BP be 8.

theorem circle_radius_squared
  (r : ℝ)       -- radius of the circle
  (AB : ℝ)     -- length of chord AB
  (CD : ℝ)     -- length of chord CD
  (APD : ℝ)    -- angle APD
  (BP : ℝ)     -- length of segment BP
  (hAB : AB = 10)
  (hCD : CD = 7)
  (hAPD : APD = 60)
  (hBP : BP = 8)
  : r^2 = 73 := 
  sorry

end circle_radius_squared_l1580_158018


namespace condition_for_equation_l1580_158082

theorem condition_for_equation (a b c : ℕ) (ha : 0 < a ∧ a < 20) (hb : 0 < b ∧ b < 20) (hc : 0 < c ∧ c < 20) :
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 :=
by
  sorry

end condition_for_equation_l1580_158082


namespace general_term_formula_not_arithmetic_sequence_l1580_158006

noncomputable def geometric_sequence (n : ℕ) : ℕ := 2^n

theorem general_term_formula :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    (∃ (q : ℕ),
      ∀ n, a n = 2^n) :=
by
  sorry

theorem not_arithmetic_sequence :
  ∀ (a : ℕ → ℕ),
    (∀ n, a n = geometric_sequence n) →
    ¬(∃ m n p : ℕ, m < n ∧ n < p ∧ (2 * a n = a m + a p)) :=
by
  sorry

end general_term_formula_not_arithmetic_sequence_l1580_158006


namespace clare_bought_loaves_l1580_158092

-- Define the given conditions
def initial_amount : ℕ := 47
def remaining_amount : ℕ := 35
def cost_per_loaf : ℕ := 2
def cost_per_carton : ℕ := 2
def number_of_cartons : ℕ := 2

-- Required to prove the number of loaves of bread bought by Clare
theorem clare_bought_loaves (initial_amount remaining_amount cost_per_loaf cost_per_carton number_of_cartons : ℕ) 
    (h1 : initial_amount = 47) 
    (h2 : remaining_amount = 35) 
    (h3 : cost_per_loaf = 2) 
    (h4 : cost_per_carton = 2) 
    (h5 : number_of_cartons = 2) : 
    (initial_amount - remaining_amount - cost_per_carton * number_of_cartons) / cost_per_loaf = 4 :=
by sorry

end clare_bought_loaves_l1580_158092


namespace calculate_expression_l1580_158041

theorem calculate_expression : (3.65 - 1.25) * 2 = 4.80 := 
by 
  sorry

end calculate_expression_l1580_158041


namespace problem_l1580_158058

open Real

theorem problem (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  (2 + a) * (2 + b) ≥ c * d := 
sorry

end problem_l1580_158058


namespace lucille_total_revenue_l1580_158024

theorem lucille_total_revenue (salary_ratio stock_ratio : ℕ) (salary_amount : ℝ) (h_ratio : salary_ratio / stock_ratio = 4 / 11) (h_salary : salary_amount = 800) : 
  ∃ total_revenue : ℝ, total_revenue = 3000 :=
by
  sorry

end lucille_total_revenue_l1580_158024


namespace nonnegative_integer_with_divisors_is_multiple_of_6_l1580_158069

-- Definitions as per conditions in (a)
def has_two_distinct_divisors_with_distance (n : ℕ) : Prop := ∃ d1 d2 : ℕ,
  d1 ≠ d2 ∧ d1 ∣ n ∧ d2 ∣ n ∧
  (d1:ℚ) - n / 3 = n / 3 - (d2:ℚ)

-- Main statement to prove as derived in (c)
theorem nonnegative_integer_with_divisors_is_multiple_of_6 (n : ℕ) :
  n > 0 ∧ has_two_distinct_divisors_with_distance n → ∃ k : ℕ, n = 6 * k :=
by
  sorry

end nonnegative_integer_with_divisors_is_multiple_of_6_l1580_158069


namespace solution_set_of_inequality_l1580_158087

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x^2 - x else Real.log (x + 1) / Real.log 2

theorem solution_set_of_inequality :
  { x : ℝ | f x ≥ 2 } = { x : ℝ | x ∈ Set.Iic (-1) } ∪ { x : ℝ | x ∈ Set.Ici 3 } :=
by
  sorry

end solution_set_of_inequality_l1580_158087


namespace range_of_m_l1580_158015

-- Definitions of the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x + 1| > m
def q (m : ℝ) : Prop := ∀ x > 2, 2 * x - 2 * m > 0

-- The main theorem statement
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) → 1 ≤ m ∧ m ≤ 2 :=
by
  sorry

end range_of_m_l1580_158015


namespace cos_identity_l1580_158089

open Real

theorem cos_identity
  (θ : ℝ)
  (h1 : cos ((5 * π) / 12 + θ) = 3 / 5)
  (h2 : -π < θ ∧ θ < -π / 2) :
  cos ((π / 12) - θ) = -4 / 5 :=
by
  sorry

end cos_identity_l1580_158089


namespace aaron_earnings_l1580_158008

def time_worked_monday := 75 -- in minutes
def time_worked_tuesday := 50 -- in minutes
def time_worked_wednesday := 145 -- in minutes
def time_worked_friday := 30 -- in minutes
def hourly_rate := 3 -- dollars per hour

def total_minutes_worked := 
  time_worked_monday + time_worked_tuesday + 
  time_worked_wednesday + time_worked_friday

def total_hours_worked := total_minutes_worked / 60

def total_earnings := total_hours_worked * hourly_rate

theorem aaron_earnings :
  total_earnings = 15 := by
  sorry

end aaron_earnings_l1580_158008


namespace find_x_to_be_2_l1580_158057

variable (x : ℝ)

def a := (2, x)
def b := (3, x + 1)

theorem find_x_to_be_2 (h : a x = b x) : x = 2 := by
  sorry

end find_x_to_be_2_l1580_158057


namespace sum_of_original_numbers_l1580_158053

theorem sum_of_original_numbers :
  ∃ a b : ℚ, a = b + 12 ∧ a^2 + b^2 = 169 / 2 ∧ (a^2)^2 - (b^2)^2 = 5070 ∧ a + b = 5 :=
by
  sorry

end sum_of_original_numbers_l1580_158053


namespace max_value_seq_l1580_158074

theorem max_value_seq : 
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 2 = 4 ∧ 
    (∀ n ≥ 2, 2 * a n = (n - 1) / n * a (n - 1) + (n + 1) / n * a (n + 1)) ∧ 
    ∀ n : ℕ, n > 0 → 
      ∃ m : ℕ, m > 0 ∧ 
        ∀ k : ℕ, k > 0 → (a k) / k ≤ 2 ∧ (a 2) / 2 = 2 :=
sorry

end max_value_seq_l1580_158074


namespace intersection_A_B_l1580_158077

def A : Set ℝ := { x : ℝ | -1 < x ∧ x < 3 }
def B : Set ℝ := { x : ℝ | x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -1 < x ∧ x < 2 } :=
by
  sorry

end intersection_A_B_l1580_158077


namespace oak_trees_remaining_l1580_158031

theorem oak_trees_remaining (initial_trees cut_down_trees remaining_trees : ℕ)
  (h1 : initial_trees = 9)
  (h2 : cut_down_trees = 2)
  (h3 : remaining_trees = initial_trees - cut_down_trees) :
  remaining_trees = 7 :=
by 
  sorry

end oak_trees_remaining_l1580_158031


namespace hannah_stocking_stuffers_l1580_158009

theorem hannah_stocking_stuffers (candy_caness : ℕ) (beanie_babies : ℕ) (books : ℕ) (kids : ℕ) : 
  candy_caness = 4 → 
  beanie_babies = 2 → 
  books = 1 → 
  kids = 3 → 
  candy_caness + beanie_babies + books = 7 → 
  7 * kids = 21 := 
by sorry

end hannah_stocking_stuffers_l1580_158009


namespace unique_solution_j_l1580_158005

theorem unique_solution_j (j : ℝ) : (∀ x : ℝ, (2 * x + 7) * (x - 5) = -43 + j * x) → (j = 5 ∨ j = -11) :=
by
  sorry

end unique_solution_j_l1580_158005


namespace tangent_parallel_and_point_P_l1580_158012

noncomputable def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_parallel_and_point_P (P : ℝ × ℝ) (hP1 : P = (1, f 1)) (hP2 : P = (-1, f (-1))) :
  (f 1 = 3 ∧ f (-1) = 3) ∧ (deriv f 1 = 2 ∧ deriv f (-1) = 2) :=
by
  sorry

end tangent_parallel_and_point_P_l1580_158012


namespace simplify_expression_l1580_158085

theorem simplify_expression : 
  8 - (-3) + (-5) + (-7) = 3 + 8 - 7 - 5 := 
by
  sorry

end simplify_expression_l1580_158085


namespace no_possible_values_for_b_l1580_158046

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_possible_values_for_b_l1580_158046


namespace second_reduction_is_18_point_1_percent_l1580_158048

noncomputable def second_reduction_percentage (P : ℝ) : ℝ :=
  let first_price := 0.91 * P
  let second_price := 0.819 * P
  let R := (first_price - second_price) / first_price
  R * 100

theorem second_reduction_is_18_point_1_percent (P : ℝ) : second_reduction_percentage P = 18.1 :=
by
  -- Proof omitted
  sorry

end second_reduction_is_18_point_1_percent_l1580_158048


namespace average_15_19_x_eq_20_l1580_158043

theorem average_15_19_x_eq_20 (x : ℝ) : (15 + 19 + x) / 3 = 20 → x = 26 :=
by
  sorry

end average_15_19_x_eq_20_l1580_158043


namespace remaining_volume_is_21_l1580_158013

-- Definitions of edge lengths and volumes
def edge_length_original : ℕ := 3
def edge_length_small : ℕ := 1
def volume (a : ℕ) : ℕ := a ^ 3

-- Volumes of the original cube and the small cubes
def volume_original : ℕ := volume edge_length_original
def volume_small : ℕ := volume edge_length_small
def number_of_faces : ℕ := 6
def total_volume_cut : ℕ := number_of_faces * volume_small

-- Volume of the remaining part
def volume_remaining : ℕ := volume_original - total_volume_cut

-- Proof statement
theorem remaining_volume_is_21 : volume_remaining = 21 := by
  sorry

end remaining_volume_is_21_l1580_158013


namespace ending_number_condition_l1580_158084

theorem ending_number_condition (h : ∃ k : ℕ, k < 21 ∧ 100 < 19 * k) : ∃ n, 21.05263157894737 * 19 = n → n = 399 :=
by
  sorry  -- this is where the proof would go

end ending_number_condition_l1580_158084


namespace circle_eq_l1580_158047

theorem circle_eq (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (4, 0))
  (hC : C = (0, 2)) :
  ∃ (h: ℝ), (x - 3) ^ 2 + (y - 3) ^ 2 = h :=
by 
  use 10
  -- additional steps to rigorously prove the result would go here
  sorry

end circle_eq_l1580_158047


namespace grandfather_grandson_ages_l1580_158067

theorem grandfather_grandson_ages :
  ∃ (x y a b : ℕ), 
    70 < x ∧ 
    x < 80 ∧ 
    x - a = 10 * (y - a) ∧ 
    x + b = 8 * (y + b) ∧ 
    x = 71 ∧ 
    y = 8 :=
by
  sorry

end grandfather_grandson_ages_l1580_158067


namespace percent_nonunion_part_time_women_l1580_158051

noncomputable def percent (part: ℚ) (whole: ℚ) : ℚ := part / whole * 100

def employees : ℚ := 100
def men_ratio : ℚ := 54 / 100
def women_ratio : ℚ := 46 / 100
def full_time_men_ratio : ℚ := 70 / 100
def part_time_men_ratio : ℚ := 30 / 100
def full_time_women_ratio : ℚ := 60 / 100
def part_time_women_ratio : ℚ := 40 / 100
def union_full_time_ratio : ℚ := 60 / 100
def union_part_time_ratio : ℚ := 50 / 100

def men := employees * men_ratio
def women := employees * women_ratio
def full_time_men := men * full_time_men_ratio
def part_time_men := men * part_time_men_ratio
def full_time_women := women * full_time_women_ratio
def part_time_women := women * part_time_women_ratio
def total_full_time := full_time_men + full_time_women
def total_part_time := part_time_men + part_time_women

def union_full_time := total_full_time * union_full_time_ratio
def union_part_time := total_part_time * union_part_time_ratio
def nonunion_full_time := total_full_time - union_full_time
def nonunion_part_time := total_part_time - union_part_time

def nonunion_part_time_women_ratio : ℚ := 50 / 100
def nonunion_part_time_women := part_time_women * nonunion_part_time_women_ratio

theorem percent_nonunion_part_time_women : 
  percent nonunion_part_time_women nonunion_part_time = 52.94 :=
by
  sorry

end percent_nonunion_part_time_women_l1580_158051


namespace misread_weight_l1580_158010

theorem misread_weight (n : ℕ) (average_incorrect : ℚ) (average_correct : ℚ) (corrected_weight : ℚ) (incorrect_total correct_total diff : ℚ)
  (h1 : n = 20)
  (h2 : average_incorrect = 58.4)
  (h3 : average_correct = 59)
  (h4 : corrected_weight = 68)
  (h5 : incorrect_total = n * average_incorrect)
  (h6 : correct_total = n * average_correct)
  (h7 : diff = correct_total - incorrect_total)
  (h8 : diff = corrected_weight - x) : x = 56 := 
sorry

end misread_weight_l1580_158010


namespace trig_proof_l1580_158065

noncomputable def trig_problem (α : ℝ) (h : Real.tan α = 3) : Prop :=
  Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5

theorem trig_proof (α : ℝ) (h : Real.tan α = 3) : Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5 :=
by
  sorry

end trig_proof_l1580_158065


namespace speed_conversion_l1580_158093

theorem speed_conversion (speed_m_s : ℚ) (conversion_factor : ℚ) :
  speed_m_s = 8 / 26 → conversion_factor = 3.6 →
  speed_m_s * conversion_factor = 1.1077 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end speed_conversion_l1580_158093


namespace part_a_first_player_wins_part_b_first_player_wins_l1580_158097

/-- Define the initial state of the game -/
structure GameState :=
(pile1 : Nat) (pile2 : Nat)

/-- Define the moves allowed in Part a) -/
inductive MoveA
| take_from_pile1 : MoveA
| take_from_pile2 : MoveA
| take_from_both  : MoveA

/-- Define the moves allowed in Part b) -/
inductive MoveB
| take_from_pile1 : MoveB
| take_from_pile2 : MoveB
| take_from_both  : MoveB
| transfer_to_pile2 : MoveB

/-- Define what it means for the first player to have a winning strategy in part a) -/
def first_player_wins_a (initial_state : GameState) : Prop := sorry

/-- Define what it means for the first player to have a winning strategy in part b) -/
def first_player_wins_b (initial_state : GameState) : Prop := sorry

/-- Theorem statement for part a) -/
theorem part_a_first_player_wins :
  first_player_wins_a ⟨7, 7⟩ :=
sorry

/-- Theorem statement for part b) -/
theorem part_b_first_player_wins :
  first_player_wins_b ⟨7, 7⟩ :=
sorry

end part_a_first_player_wins_part_b_first_player_wins_l1580_158097


namespace total_money_tshirts_l1580_158078

-- Conditions
def price_per_tshirt : ℕ := 62
def num_tshirts_sold : ℕ := 183

-- Question: prove the total money made from selling the t-shirts
theorem total_money_tshirts :
  num_tshirts_sold * price_per_tshirt = 11346 := 
by
  -- Proof goes here
  sorry

end total_money_tshirts_l1580_158078


namespace cat_can_pass_through_gap_l1580_158081

theorem cat_can_pass_through_gap (R : ℝ) (h : ℝ) (π : ℝ) (hπ : π = Real.pi)
  (L₀ : ℝ) (L₁ : ℝ)
  (hL₀ : L₀ = 2 * π * R)
  (hL₁ : L₁ = L₀ + 1)
  (hL₁' : L₁ = 2 * π * (R + h)) :
  h = 1 / (2 * π) :=
by
  sorry

end cat_can_pass_through_gap_l1580_158081


namespace isosceles_trapezoid_l1580_158035

-- Define a type for geometric points
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define structures for geometric properties
structure Trapezoid :=
  (A B C D M N : Point)
  (is_midpoint_M : 2 * M.x = A.x + B.x ∧ 2 * M.y = A.y + B.y)
  (is_midpoint_N : 2 * N.x = C.x + D.x ∧ 2 * N.y = C.y + D.y)
  (AB_parallel_CD : (B.y - A.y) * (D.x - C.x) = (B.x - A.x) * (D.y - C.y)) -- AB || CD
  (MN_perpendicular_AB_CD : (N.y - M.y) * (B.y - A.y) + (N.x - M.x) * (B.x - A.x) = 0 ∧
                            (N.y - M.y) * (D.y - C.y) + (N.x - M.x) * (D.x - C.x) = 0) -- MN ⊥ AB && MN ⊥ CD

-- The isosceles condition
def is_isosceles (T : Trapezoid) : Prop :=
  ((T.A.x - T.D.x) ^ 2 + (T.A.y - T.D.y) ^ 2) = ((T.B.x - T.C.x) ^ 2 + (T.B.y - T.C.y) ^ 2)

-- The theorem statement
theorem isosceles_trapezoid (T : Trapezoid) : is_isosceles T :=
by
  sorry

end isosceles_trapezoid_l1580_158035
