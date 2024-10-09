import Mathlib

namespace find_n_for_divisibility_l1280_128079

def digit_sum_odd_positions := 8 + 4 + 5 + 6 -- The sum of the digits in odd positions
def digit_sum_even_positions (n : ℕ) := 5 + n + 2 -- The sum of the digits in even positions

def is_divisible_by_11 (n : ℕ) := (digit_sum_odd_positions - digit_sum_even_positions n) % 11 = 0

theorem find_n_for_divisibility : is_divisible_by_11 5 :=
by
  -- Proof would go here (but according to the instructions, we'll insert a placeholder)
  sorry

end find_n_for_divisibility_l1280_128079


namespace mr_william_land_percentage_l1280_128047

def total_tax_collected : ℝ := 3840
def mr_william_tax_paid : ℝ := 480
def expected_percentage : ℝ := 12.5

theorem mr_william_land_percentage :
  (mr_william_tax_paid / total_tax_collected) * 100 = expected_percentage := 
sorry

end mr_william_land_percentage_l1280_128047


namespace ball_radius_l1280_128068

theorem ball_radius (x r : ℝ) 
  (h1 : (15 : ℝ) ^ 2 + x ^ 2 = r ^ 2) 
  (h2 : x + 12 = r) : 
  r = 15.375 := 
sorry

end ball_radius_l1280_128068


namespace sin_330_deg_l1280_128023

theorem sin_330_deg : Real.sin (330 * Real.pi / 180) = -1 / 2 := by
  sorry

end sin_330_deg_l1280_128023


namespace find_y_of_series_eq_92_l1280_128043

theorem find_y_of_series_eq_92 (y : ℝ) (h : (∑' n, (2 + 5 * n) * y^n) = 92) (converge : abs y < 1) : y = 18 / 23 :=
sorry

end find_y_of_series_eq_92_l1280_128043


namespace total_value_is_correct_l1280_128090

-- We will define functions that convert base 7 numbers to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let digits := (n.digits 7)
  digits.enum.foldr (λ ⟨i, d⟩ acc => acc + d * 7^i) 0

-- Define the specific numbers in base 7
def silver_value_base7 : Nat := 5326
def gemstone_value_base7 : Nat := 3461
def spice_value_base7 : Nat := 656

-- Define the combined total in base 10
def total_value_base10 : Nat := base7_to_base10 silver_value_base7 + base7_to_base10 gemstone_value_base7 + base7_to_base10 spice_value_base7

theorem total_value_is_correct :
  total_value_base10 = 3485 :=
by
  sorry

end total_value_is_correct_l1280_128090


namespace ratio_of_triangle_BFD_to_square_ABCE_l1280_128009

def is_square (ABCE : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ a b c e : ℝ, ABCE a b c e → a = b ∧ b = c ∧ c = e

def ratio_of_areas (AF FE CD DE : ℝ) (ratio : ℝ) : Prop :=
  AF = 3 * FE ∧ CD = 3 * DE ∧ ratio = 1 / 2

theorem ratio_of_triangle_BFD_to_square_ABCE (AF FE CD DE ratio : ℝ) (ABCE : ℝ → ℝ → ℝ → ℝ → Prop)
  (h1 : is_square ABCE)
  (h2 : AF = 3 * FE) (h3 : CD = 3 * DE) : ratio_of_areas AF FE CD DE (1 / 2) :=
by
  sorry

end ratio_of_triangle_BFD_to_square_ABCE_l1280_128009


namespace book_cost_l1280_128011

theorem book_cost (x y : ℝ) (h₁ : 2 * y = x) (h₂ : 100 + y = x - 100) : x = 200 := by
  sorry

end book_cost_l1280_128011


namespace arithmetic_sequence_y_value_l1280_128085

theorem arithmetic_sequence_y_value (y : ℚ) :
  ∃ y : ℚ, 
    (y - 2) - (2/3) = (4 * y - 1) - (y - 2) → 
    y = 11/6 := by
  sorry

end arithmetic_sequence_y_value_l1280_128085


namespace pastries_and_cost_correct_l1280_128048

def num_pastries_lola := 13 + 10 + 8 + 6
def cost_lola := 13 * 0.50 + 10 * 1.00 + 8 * 3.00 + 6 * 2.00

def num_pastries_lulu := 16 + 12 + 14 + 9
def cost_lulu := 16 * 0.50 + 12 * 1.00 + 14 * 3.00 + 9 * 2.00

def num_pastries_lila := 22 + 15 + 10 + 12
def cost_lila := 22 * 0.50 + 15 * 1.00 + 10 * 3.00 + 12 * 2.00

def num_pastries_luka := 18 + 20 + 7 + 14 + 25
def cost_luka := 18 * 0.50 + 20 * 1.00 + 7 * 3.00 + 14 * 2.00 + 25 * 1.50

def total_pastries := num_pastries_lola + num_pastries_lulu + num_pastries_lila + num_pastries_luka
def total_cost := cost_lola + cost_lulu + cost_lila + cost_luka

theorem pastries_and_cost_correct :
  total_pastries = 231 ∧ total_cost = 328.00 :=
by
  sorry

end pastries_and_cost_correct_l1280_128048


namespace smallest_sum_of_exterior_angles_l1280_128095

open Real

theorem smallest_sum_of_exterior_angles 
  (p q r : ℕ) 
  (hp : p > 2) 
  (hq : q > 2) 
  (hr : r > 2) 
  (hpq : p ≠ q) 
  (hqr : q ≠ r) 
  (hrp : r ≠ p) 
  : (360 / p + 360 / q + 360 / r) ≥ 282 ∧ 
    (360 / p + 360 / q + 360 / r) = 282 → 
    360 / p = 120 ∧ 360 / q = 90 ∧ 360 / r = 72 := 
sorry

end smallest_sum_of_exterior_angles_l1280_128095


namespace intersection_dist_general_l1280_128074

theorem intersection_dist_general {a b : ℝ} 
  (h1 : (a^2 + 1) * (a^2 + 4 * (b + 1)) = 34)
  (h2 : (a^2 + 1) * (a^2 + 4 * (b + 2)) = 42) : 
  ∀ x1 x2 : ℝ, 
  x1 ≠ x2 → 
  (x1 * x1 = a * x1 + b - 1 ∧ x2 * x2 = a * x2 + b - 1) → 
  |x2 - x1| = 3 * Real.sqrt 2 :=
by
  sorry

end intersection_dist_general_l1280_128074


namespace partI_inequality_solution_partII_minimum_value_l1280_128025

-- Part (I)
theorem partI_inequality_solution (x : ℝ) : 
  (abs (x + 1) + abs (2 * x - 1) ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 1) :=
sorry

-- Part (II)
theorem partII_minimum_value (a b c : ℝ) (h1 : a + b + c = 2) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) :
  (∀ a b c : ℝ, a + b + c = 2 ->  a > 0 -> b > 0 -> c > 0 -> 
    (1 / a + 1 / b + 1 / c) = (9 / 2)) :=
sorry

end partI_inequality_solution_partII_minimum_value_l1280_128025


namespace wendy_baked_29_cookies_l1280_128053

variables (cupcakes : ℕ) (pastries_taken_home : ℕ) (pastries_sold : ℕ)

def total_initial_pastries (cupcakes pastries_taken_home pastries_sold : ℕ) : ℕ :=
  pastries_taken_home + pastries_sold

def cookies_baked (total_initial_pastries cupcakes : ℕ) : ℕ :=
  total_initial_pastries - cupcakes

theorem wendy_baked_29_cookies :
  cupcakes = 4 →
  pastries_taken_home = 24 →
  pastries_sold = 9 →
  cookies_baked (total_initial_pastries cupcakes pastries_taken_home pastries_sold) cupcakes = 29 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact sorry

end wendy_baked_29_cookies_l1280_128053


namespace sausage_more_than_pepperoni_l1280_128063

noncomputable def pieces_of_meat_per_slice : ℕ := 22
noncomputable def slices : ℕ := 6
noncomputable def total_pieces_of_meat : ℕ := pieces_of_meat_per_slice * slices

noncomputable def pieces_of_pepperoni : ℕ := 30
noncomputable def pieces_of_ham : ℕ := 2 * pieces_of_pepperoni

noncomputable def total_pieces_of_meat_without_sausage : ℕ := pieces_of_pepperoni + pieces_of_ham
noncomputable def pieces_of_sausage : ℕ := total_pieces_of_meat - total_pieces_of_meat_without_sausage

theorem sausage_more_than_pepperoni : (pieces_of_sausage - pieces_of_pepperoni) = 12 := by
  sorry

end sausage_more_than_pepperoni_l1280_128063


namespace max_m_value_l1280_128002

variables {x y m : ℝ}

theorem max_m_value (h1 : 4 * x + 3 * y = 4 * m + 5)
                     (h2 : 3 * x - y = m - 1)
                     (h3 : x + 4 * y ≤ 3) :
                     m ≤ -1 :=
sorry

end max_m_value_l1280_128002


namespace checkered_triangle_division_l1280_128088

theorem checkered_triangle_division :
  ∀ (triangle : List ℕ), triangle.sum = 63 →
  ∃ (part1 part2 part3 : List ℕ),
    part1.sum = 21 ∧ part2.sum = 21 ∧ part3.sum = 21 ∧
    part1 ≠ part2 ∧ part2 ≠ part3 ∧ part1 ≠ part3 ∧
    (part1 ++ part2 ++ part3).length = triangle.length ∧
    (∃ (area1 area2 area3 : ℕ), area1 ≠ area2 ∧ area2 ≠ area3 ∧ area1 ≠ area3) :=
by
  sorry

end checkered_triangle_division_l1280_128088


namespace not_recurring_decimal_l1280_128035

-- Definitions based on the provided conditions
def is_recurring_decimal (x : ℝ) : Prop :=
  ∃ d m n : ℕ, d ≠ 0 ∧ (x * d) % 10 ^ n = m

-- Condition: 0.89898989
def number_0_89898989 : ℝ := 0.89898989

-- Proof statement to show 0.89898989 is not a recurring decimal
theorem not_recurring_decimal : ¬ is_recurring_decimal number_0_89898989 :=
sorry

end not_recurring_decimal_l1280_128035


namespace find_original_cost_price_l1280_128065

variable (C : ℝ)

-- Conditions
def first_discount (C : ℝ) : ℝ := 0.95 * C
def second_discount (C : ℝ) : ℝ := 0.9215 * C
def loss_price (C : ℝ) : ℝ := 0.90 * C
def gain_price_before_tax (C : ℝ) : ℝ := 1.08 * C
def gain_price_after_tax (C : ℝ) : ℝ := 1.20 * C

-- Prove that original cost price is 1800
theorem find_original_cost_price 
  (h1 : first_discount C = loss_price C)
  (h2 : gain_price_after_tax C - loss_price C = 540) : 
  C = 1800 := 
sorry

end find_original_cost_price_l1280_128065


namespace prob1_prob2_prob3_prob4_l1280_128087

theorem prob1 : (3^3)^2 = 3^6 := by
  sorry

theorem prob2 : (-4 * x * y^3) * (-2 * x^2) = 8 * x^3 * y^3 := by
  sorry

theorem prob3 : 2 * x * (3 * y - x^2) + 2 * x * x^2 = 6 * x * y := by
  sorry

theorem prob4 : (20 * x^3 * y^5 - 10 * x^4 * y^4 - 20 * x^3 * y^2) / (-5 * x^3 * y^2) = -4 * y^3 + 2 * x * y^2 + 4 := by
  sorry

end prob1_prob2_prob3_prob4_l1280_128087


namespace cosine_periodicity_l1280_128077

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l1280_128077


namespace skee_ball_tickets_l1280_128005

-- Represent the given conditions as Lean definitions
def whack_a_mole_tickets : ℕ := 33
def candy_cost_per_piece : ℕ := 6
def candies_bought : ℕ := 7
def total_candy_tickets : ℕ := candies_bought * candy_cost_per_piece

-- Goal: Prove the number of tickets won playing 'skee ball'
theorem skee_ball_tickets (h : 42 = total_candy_tickets): whack_a_mole_tickets + 9 = total_candy_tickets :=
by {
  sorry
}

end skee_ball_tickets_l1280_128005


namespace range_of_a_l1280_128039

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) ∧
  ¬(∀ x : ℝ, (4 * x - 3) ^ 2 ≤ 1 → (x ^ 2 - (2 * a + 1) * x + a * (a + 1)) ≤ 0) →
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end range_of_a_l1280_128039


namespace total_rocks_is_300_l1280_128055

-- Definitions of rock types in Cliff's collection
variables (I S M : ℕ) -- I: number of igneous rocks, S: number of sedimentary rocks, M: number of metamorphic rocks
variables (shinyI shinyS shinyM : ℕ) -- shinyI: shiny igneous rocks, shinyS: shiny sedimentary rocks, shinyM: shiny metamorphic rocks

-- Given conditions
def igneous_one_third_shiny (I shinyI : ℕ) := 2 * shinyI = 3 * I
def sedimentary_two_ig_as_sed (S I : ℕ) := S = 2 * I
def metamorphic_twice_as_ig (M I : ℕ) := M = 2 * I
def shiny_igneous_is_40 (shinyI : ℕ) := shinyI = 40
def one_fifth_sed_shiny (S shinyS : ℕ) := 5 * shinyS = S
def three_quarters_met_shiny (M shinyM : ℕ) := 4 * shinyM = 3 * M

-- Theorem statement
theorem total_rocks_is_300 (I S M shinyI shinyS shinyM : ℕ)
  (h1 : igneous_one_third_shiny I shinyI)
  (h2 : sedimentary_two_ig_as_sed S I)
  (h3 : metamorphic_twice_as_ig M I)
  (h4 : shiny_igneous_is_40 shinyI)
  (h5 : one_fifth_sed_shiny S shinyS)
  (h6 : three_quarters_met_shiny M shinyM) :
  (I + S + M) = 300 :=
sorry -- Proof to be completed

end total_rocks_is_300_l1280_128055


namespace part1_part2_l1280_128066

theorem part1 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h_imag : z.re = 0) : a = 1 :=
sorry

theorem part2 (a : ℝ) (ha : z = Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6))
  (h4thQuad : z.re > 0 ∧ z.im < 0) : -1 < a ∧ a < 1 :=
sorry

end part1_part2_l1280_128066


namespace train_pass_time_l1280_128058

-- Assuming conversion factor, length of the train, and speed in km/hr
def conversion_factor := 1000 / 3600
def train_length := 280
def speed_km_hr := 36

-- Defining speed in m/s
def speed_m_s := speed_km_hr * conversion_factor

-- Defining the time to pass a tree
def time_to_pass_tree := train_length / speed_m_s

-- Theorem statement
theorem train_pass_time : time_to_pass_tree = 28 := by
  sorry

end train_pass_time_l1280_128058


namespace probability_of_choosing_gulongzhong_l1280_128034

def num_attractions : Nat := 4
def num_ways_gulongzhong : Nat := 1
def probability_gulongzhong : ℚ := num_ways_gulongzhong / num_attractions

theorem probability_of_choosing_gulongzhong : probability_gulongzhong = 1 / 4 := 
by 
  sorry

end probability_of_choosing_gulongzhong_l1280_128034


namespace algebraic_identity_l1280_128022

theorem algebraic_identity (a b : ℝ) : a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

example : (2011 : ℝ)^2 - (2010 : ℝ)^2 = 4021 := 
by
  have h := algebraic_identity 2011 2010
  rw [h]
  norm_num

end algebraic_identity_l1280_128022


namespace job_completion_days_l1280_128000

theorem job_completion_days :
  let days_total := 150
  let workers_initial := 25
  let workers_less_efficient := 15
  let workers_more_efficient := 10
  let days_elapsed := 40
  let efficiency_less := 1
  let efficiency_more := 1.5
  let work_fraction_completed := 1/3
  let workers_fired_less := 4
  let workers_fired_more := 3
  let units_per_day_initial := (workers_less_efficient * efficiency_less) + (workers_more_efficient * efficiency_more)
  let work_completed := units_per_day_initial * days_elapsed
  let total_work := work_completed / work_fraction_completed
  let workers_remaining_less := workers_less_efficient - workers_fired_less
  let workers_remaining_more := workers_more_efficient - workers_fired_more
  let units_per_day_new := (workers_remaining_less * efficiency_less) + (workers_remaining_more * efficiency_more)
  let work_remaining := total_work * (2/3)
  let remaining_days := work_remaining / units_per_day_new
  remaining_days.ceil = 112 :=
by
  sorry

end job_completion_days_l1280_128000


namespace major_premise_is_wrong_l1280_128064

-- Definitions of the conditions
def line_parallel_to_plane (l : Type) (p : Type) : Prop := sorry
def line_contained_in_plane (l : Type) (p : Type) : Prop := sorry

-- Stating the main problem: the major premise is wrong
theorem major_premise_is_wrong :
  ∀ (a b : Type) (α : Type), line_contained_in_plane a α → line_parallel_to_plane b α → ¬ (line_parallel_to_plane b a) := 
by 
  intros a b α h1 h2
  sorry

end major_premise_is_wrong_l1280_128064


namespace driving_scenario_l1280_128040

theorem driving_scenario (x : ℝ) (h1 : x > 0) :
  (240 / x) - (240 / (1.5 * x)) = 1 :=
by
  sorry

end driving_scenario_l1280_128040


namespace simplify_and_evaluate_expression_l1280_128033

theorem simplify_and_evaluate_expression (x : ℝ) (hx : x = 6) :
  (1 + (2 / (x + 1))) * ((x^2 + x) / (x^2 - 9)) = 2 := by
  sorry

end simplify_and_evaluate_expression_l1280_128033


namespace binary_calculation_l1280_128056

-- Binary arithmetic definition
def binary_mul (a b : Nat) : Nat := a * b
def binary_div (a b : Nat) : Nat := a / b

-- Binary numbers in Nat (representing binary literals by their decimal equivalent)
def b110010 := 50   -- 110010_2 in decimal
def b101000 := 40   -- 101000_2 in decimal
def b100 := 4       -- 100_2 in decimal
def b10 := 2        -- 10_2 in decimal
def b10111000 := 184-- 10111000_2 in decimal

theorem binary_calculation :
  binary_div (binary_div (binary_mul b110010 b101000) b100) b10 = b10111000 :=
by
  sorry

end binary_calculation_l1280_128056


namespace randolph_age_l1280_128021

theorem randolph_age (R Sy S : ℕ) 
  (h1 : R = Sy + 5) 
  (h2 : Sy = 2 * S) 
  (h3 : S = 25) : 
  R = 55 :=
by 
  sorry

end randolph_age_l1280_128021


namespace geometric_sequence_general_term_l1280_128059

noncomputable def geometric_sequence (a : ℕ → ℝ) := 
  ∃ q : ℝ, q > 0 ∧ (∀ n, a (n + 1) = q * a n)

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_S3 : a 1 * (1 + (a 2 / a 1) + (a 3 / a 1)) = 21) 
  (h_condition : 2 * a 2 = a 3) :
  ∃ c : ℝ, c = 3 ∧ ∀ n, a n = 3 * 2^(n - 1) := sorry

end geometric_sequence_general_term_l1280_128059


namespace trains_length_difference_eq_zero_l1280_128046

theorem trains_length_difference_eq_zero
  (T1_pole_time : ℕ) (T1_platform_time : ℕ) (T2_pole_time : ℕ) (T2_platform_time : ℕ) (platform_length : ℕ)
  (h1 : T1_pole_time = 11)
  (h2 : T1_platform_time = 22)
  (h3 : T2_pole_time = 15)
  (h4 : T2_platform_time = 30)
  (h5 : platform_length = 120) :
  let L1 := T1_pole_time * platform_length / (T1_platform_time - T1_pole_time)
  let L2 := T2_pole_time * platform_length / (T2_platform_time - T2_pole_time)
  L1 = L2 :=
by
  sorry

end trains_length_difference_eq_zero_l1280_128046


namespace paint_cost_is_correct_l1280_128004

-- Definition of known conditions
def costPerKg : ℕ := 50
def coveragePerKg : ℕ := 20
def sideOfCube : ℕ := 20

-- Definition of correct answer
def totalCost : ℕ := 6000

-- Theorem statement
theorem paint_cost_is_correct : (6 * (sideOfCube * sideOfCube) / coveragePerKg) * costPerKg = totalCost :=
by
  sorry

end paint_cost_is_correct_l1280_128004


namespace four_digit_number_exists_l1280_128031

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), A = B / 3 ∧ C = A + B ∧ D = 3 * B ∧
  A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1349) :=
by
  sorry

end four_digit_number_exists_l1280_128031


namespace second_monkey_took_20_peaches_l1280_128020

theorem second_monkey_took_20_peaches (total_peaches : ℕ) 
  (h1 : total_peaches > 0)
  (eldest_share : ℕ)
  (middle_share : ℕ)
  (youngest_share : ℕ)
  (h3 : total_peaches = eldest_share + middle_share + youngest_share)
  (h4 : eldest_share = (total_peaches * 5) / 9)
  (second_total : ℕ := total_peaches - eldest_share)
  (h5 : middle_share = (second_total * 5) / 9)
  (h6 : youngest_share = second_total - middle_share)
  (h7 : eldest_share - youngest_share = 29) :
  middle_share = 20 :=
by
  sorry

end second_monkey_took_20_peaches_l1280_128020


namespace max_value_sqrt_expr_l1280_128051

open Real

theorem max_value_sqrt_expr (x y z : ℝ)
  (h1 : x + y + z = 1)
  (h2 : x ≥ -1/3)
  (h3 : y ≥ -1)
  (h4 : z ≥ -5/3) :
  (sqrt (3 * x + 1) + sqrt (3 * y + 3) + sqrt (3 * z + 5)) ≤ 6 :=
  sorry

end max_value_sqrt_expr_l1280_128051


namespace PR_length_right_triangle_l1280_128029

theorem PR_length_right_triangle
  (P Q R : Type)
  (cos_R : ℝ)
  (PQ PR : ℝ)
  (h1 : cos_R = 5 * Real.sqrt 34 / 34)
  (h2 : PQ = Real.sqrt 34)
  (h3 : cos_R = PR / PQ) : PR = 5 := by
  sorry

end PR_length_right_triangle_l1280_128029


namespace simplify_expression_l1280_128014

theorem simplify_expression (n : ℕ) (hn : 0 < n) :
  (3^(n+5) - 3 * 3^n) / (3 * 3^(n+4) - 6) = 80 / 81 :=
by
  sorry

end simplify_expression_l1280_128014


namespace becky_packs_lunch_days_l1280_128069

-- Definitions of conditions
def school_days := 180
def aliyah_packing_fraction := 1 / 2
def becky_relative_fraction := 1 / 2

-- Derived quantities from conditions
def aliyah_pack_days := school_days * aliyah_packing_fraction
def becky_pack_days := aliyah_pack_days * becky_relative_fraction

-- Statement to prove
theorem becky_packs_lunch_days : becky_pack_days = 45 := by
  sorry

end becky_packs_lunch_days_l1280_128069


namespace alpha_value_l1280_128032

noncomputable def alpha (x : ℝ) := Real.arccos x

theorem alpha_value (h1 : Real.cos α = -1/6) (h2 : 0 < α ∧ α < Real.pi) : 
  α = Real.pi - alpha (1/6) :=
by
  sorry

end alpha_value_l1280_128032


namespace compute_expression_l1280_128041

theorem compute_expression : (46 + 15)^2 - (46 - 15)^2 = 2760 :=
by
  sorry

end compute_expression_l1280_128041


namespace photos_on_last_page_l1280_128086

noncomputable def total_photos : ℕ := 10 * 35 * 4
noncomputable def photos_per_page_after_reorganization : ℕ := 8
noncomputable def total_pages_needed : ℕ := (total_photos + photos_per_page_after_reorganization - 1) / photos_per_page_after_reorganization
noncomputable def pages_filled_in_first_6_albums : ℕ := 6 * 35
noncomputable def last_page_photos : ℕ := if total_pages_needed ≤ pages_filled_in_first_6_albums then 0 else total_photos % photos_per_page_after_reorganization

theorem photos_on_last_page : last_page_photos = 0 :=
by
  sorry

end photos_on_last_page_l1280_128086


namespace karen_cases_picked_up_l1280_128081

theorem karen_cases_picked_up (total_boxes : ℤ) (boxes_per_case : ℤ) (h1 : total_boxes = 36) (h2 : boxes_per_case = 12) : (total_boxes / boxes_per_case) = 3 := by
  sorry

end karen_cases_picked_up_l1280_128081


namespace cartesian_equation_of_circle_c2_positional_relationship_between_circles_l1280_128024
noncomputable def circle_c1 := {p : ℝ × ℝ | (p.1)^2 - 2*p.1 + (p.2)^2 = 0}
noncomputable def circle_c2_polar (theta : ℝ) : ℝ × ℝ := (2 * Real.sin theta * Real.cos theta, 2 * Real.sin theta * Real.sin theta)
noncomputable def circle_c2_cartesian := {p : ℝ × ℝ | (p.1)^2 + (p.2 - 1)^2 = 1}

theorem cartesian_equation_of_circle_c2 :
  ∀ p : ℝ × ℝ, (∃ θ : ℝ, p = circle_c2_polar θ) ↔ p ∈ circle_c2_cartesian :=
by
  sorry

theorem positional_relationship_between_circles :
  ∃ p : ℝ × ℝ, p ∈ circle_c1 ∧ p ∈ circle_c2_cartesian :=
by
  sorry

end cartesian_equation_of_circle_c2_positional_relationship_between_circles_l1280_128024


namespace find_a4_plus_b4_l1280_128036

-- Variables representing the given conditions
variables {a b : ℝ}

-- The theorem statement to prove
theorem find_a4_plus_b4 (h1 : a^2 - b^2 = 8) (h2 : a * b = 2) : a^4 + b^4 = 56 :=
sorry

end find_a4_plus_b4_l1280_128036


namespace exists_consecutive_with_square_factors_l1280_128097

theorem exists_consecutive_with_square_factors (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ∃ m : ℕ, m^2 ∣ (k + i) ∧ m > 1 :=
by {
  sorry
}

end exists_consecutive_with_square_factors_l1280_128097


namespace exists_equilateral_triangle_same_color_l1280_128038

-- Define a type for colors
inductive Color
| red : Color
| blue : Color

-- Define our statement
-- Given each point in the plane is colored either red or blue,
-- there exists an equilateral triangle with vertices of the same color.
theorem exists_equilateral_triangle_same_color (coloring : ℝ × ℝ → Color) : 
  ∃ (p₁ p₂ p₃ : ℝ × ℝ), 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧ 
    dist p₁ p₂ = dist p₂ p₃ ∧ dist p₂ p₃ = dist p₃ p₁ ∧ 
    (coloring p₁ = coloring p₂ ∧ coloring p₂ = coloring p₃) :=
by
  sorry

end exists_equilateral_triangle_same_color_l1280_128038


namespace rainfall_hydroville_2012_l1280_128028

-- Define the average monthly rainfall for each year
def avg_rainfall_2010 : ℝ := 37.2
def avg_rainfall_2011 : ℝ := avg_rainfall_2010 + 3.5
def avg_rainfall_2012 : ℝ := avg_rainfall_2011 - 1.2

-- Define the total rainfall for 2012
def total_rainfall_2012 : ℝ := 12 * avg_rainfall_2012

-- The theorem to be proved
theorem rainfall_hydroville_2012 : total_rainfall_2012 = 474 := by
  sorry

end rainfall_hydroville_2012_l1280_128028


namespace correct_option_l1280_128094

theorem correct_option 
  (A_false : ¬ (-6 - (-9)) = -3)
  (B_false : ¬ (-2 * (-5)) = -7)
  (C_false : ¬ (-x^2 + 3 * x^2) = 2)
  (D_true : (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b) :
  (4 * a^2 * b - 2 * b * a^2) = 2 * a^2 * b :=
by sorry

end correct_option_l1280_128094


namespace find_C_l1280_128030

theorem find_C (A B C : ℕ) (h1 : (8 + 4 + A + 7 + 3 + B + 2) % 3 = 0)
  (h2 : (5 + 2 + 9 + A + B + 4 + C) % 3 = 0) : C = 2 :=
by
  sorry

end find_C_l1280_128030


namespace parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l1280_128075

theorem parabola_right_shift_unique_intersection (p : ℚ) :
  let y := 2 * (x - p)^2;
  (x * x - 4) = 0 →
  p = 31 / 8 := sorry

theorem parabola_down_shift_unique_intersection (q : ℚ) :
  let y := 2 * x^2 - q;
  (x * x - 4) = 0 →
  q = 31 / 8 := sorry

end parabola_right_shift_unique_intersection_parabola_down_shift_unique_intersection_l1280_128075


namespace find_number_l1280_128001

theorem find_number (x q : ℕ) (h1 : x = 3 * q) (h2 : q + x + 3 = 63) : x = 45 :=
sorry

end find_number_l1280_128001


namespace addition_example_l1280_128091

theorem addition_example : 300 + 2020 + 10001 = 12321 := 
by 
  sorry

end addition_example_l1280_128091


namespace hermans_breakfast_cost_l1280_128052

-- Define the conditions
def meals_per_day : Nat := 4
def days_per_week : Nat := 5
def cost_per_meal : Nat := 4
def total_weeks : Nat := 16

-- Define the statement to prove
theorem hermans_breakfast_cost :
  (meals_per_day * days_per_week * cost_per_meal * total_weeks) = 1280 := by
  sorry

end hermans_breakfast_cost_l1280_128052


namespace geometric_sequence_not_sufficient_nor_necessary_l1280_128084

theorem geometric_sequence_not_sufficient_nor_necessary (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) → 
  (¬ (q > 1 → ∀ n : ℕ, a n < a (n + 1))) ∧ (¬ (∀ n : ℕ, a n < a (n + 1) → q > 1)) :=
by
  sorry

end geometric_sequence_not_sufficient_nor_necessary_l1280_128084


namespace square_perimeter_ratio_l1280_128026

theorem square_perimeter_ratio (a b : ℝ) (h : (a^2 / b^2) = (49 / 64)) : (4 * a) / (4 * b) = 7 / 8 :=
by
  -- Given that the areas are in the ratio 49:64, we have (a / b)^2 = 49 / 64.
  -- Therefore, (a / b) = sqrt (49 / 64) = 7 / 8.
  -- Thus, the ratio of the perimeters 4a / 4b = 7 / 8.
  sorry

end square_perimeter_ratio_l1280_128026


namespace hyperbola_parameters_sum_l1280_128054

theorem hyperbola_parameters_sum :
  ∃ (h k a b : ℝ), 
    (h = 2 ∧ k = 0 ∧ a = 3 ∧ b = 3 * Real.sqrt 3) ∧
    h + k + a + b = 3 * Real.sqrt 3 + 5 := by
  sorry

end hyperbola_parameters_sum_l1280_128054


namespace largest_angle_of_scalene_triangle_l1280_128093

-- Define the problem statement in Lean
theorem largest_angle_of_scalene_triangle (x : ℝ) (hx : x = 30) : 3 * x = 90 :=
by {
  -- Given that the smallest angle is x and x = 30 degrees
  sorry
}

end largest_angle_of_scalene_triangle_l1280_128093


namespace geometric_sequence_a6_l1280_128016

theorem geometric_sequence_a6 (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 4) (h2 : a 4 = 2) 
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q) :
  a 6 = 4 :=
sorry

end geometric_sequence_a6_l1280_128016


namespace largest_number_among_options_l1280_128007

theorem largest_number_among_options :
  let A := 8.12366
  let B := 8.1236666666666 -- Repeating decimal 8.123\overline{6}
  let C := 8.1236363636363 -- Repeating decimal 8.12\overline{36}
  let D := 8.1236236236236 -- Repeating decimal 8.1\overline{236}
  let E := 8.1236123612361 -- Repeating decimal 8.\overline{1236}
  B > A ∧ B > C ∧ B > D ∧ B > E :=
by
  let A := 8.12366
  let B := 8.12366666666666
  let C := 8.12363636363636
  let D := 8.12362362362362
  let E := 8.12361236123612
  sorry

end largest_number_among_options_l1280_128007


namespace no_prime_divisible_by_42_l1280_128098

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end no_prime_divisible_by_42_l1280_128098


namespace intersection_of_A_and_B_l1280_128092

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l1280_128092


namespace average_beef_sales_l1280_128050

def ground_beef_sales.Thur : ℕ := 210
def ground_beef_sales.Fri : ℕ := 2 * ground_beef_sales.Thur
def ground_beef_sales.Sat : ℕ := 150
def ground_beef_sales.total : ℕ := ground_beef_sales.Thur + ground_beef_sales.Fri + ground_beef_sales.Sat
def ground_beef_sales.days : ℕ := 3
def ground_beef_sales.average : ℕ := ground_beef_sales.total / ground_beef_sales.days

theorem average_beef_sales (thur : ℕ) (fri : ℕ) (sat : ℕ) (days : ℕ) (total : ℕ) (avg : ℕ) :
  thur = 210 → 
  fri = 2 * thur → 
  sat = 150 → 
  total = thur + fri + sat → 
  days = 3 → 
  avg = total / days → 
  avg = 260 := by
    sorry

end average_beef_sales_l1280_128050


namespace original_phone_number_eq_l1280_128027

theorem original_phone_number_eq :
  ∃ (a b c d e f : ℕ), 
    (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 282500) ∧
    (1000000 * 2 + 100000 * a + 10000 * 8 + 1000 * b + 100 * c + 10 * d + e = 81 * (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) ∧
    (0 ≤ a ∧ a ≤ 9) ∧
    (0 ≤ b ∧ b ≤ 9) ∧
    (0 ≤ c ∧ c ≤ 9) ∧
    (0 ≤ d ∧ d ≤ 9) ∧
    (0 ≤ e ∧ e ≤ 9) ∧
    (0 ≤ f ∧ f ≤ 9) :=
sorry

end original_phone_number_eq_l1280_128027


namespace projection_areas_are_correct_l1280_128018

noncomputable def S1 := 1/2 * 2 * 2
noncomputable def S2 := 1/2 * 2 * Real.sqrt 2
noncomputable def S3 := 1/2 * 2 * Real.sqrt 2

theorem projection_areas_are_correct :
  S3 = S2 ∧ S3 ≠ S1 :=
by
  sorry

end projection_areas_are_correct_l1280_128018


namespace race_problem_l1280_128061

theorem race_problem
  (total_distance : ℕ)
  (A_time : ℕ)
  (B_extra_time : ℕ)
  (A_speed B_speed : ℕ)
  (A_distance B_distance : ℕ)
  (H1 : total_distance = 120)
  (H2 : A_time = 8)
  (H3 : B_extra_time = 7)
  (H4 : A_speed = total_distance / A_time)
  (H5 : B_speed = total_distance / (A_time + B_extra_time))
  (H6 : A_distance = total_distance)
  (H7 : B_distance = B_speed * A_time) :
  A_distance - B_distance = 56 := 
sorry

end race_problem_l1280_128061


namespace interval_solution_l1280_128062

theorem interval_solution (x : ℝ) : 
  (1 < 5 * x ∧ 5 * x < 3) ∧ (2 < 8 * x ∧ 8 * x < 4) ↔ (1/4 < x ∧ x < 1/2) := 
by
  sorry

end interval_solution_l1280_128062


namespace max_value_A_l1280_128012

theorem max_value_A (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ( ( (x - y) * Real.sqrt (x^2 + y^2) + (y - z) * Real.sqrt (y^2 + z^2) + (z - x) * Real.sqrt (z^2 + x^2) + Real.sqrt 2 ) / 
    ( (x - y)^2 + (y - z)^2 + (z - x)^2 + 2 ) ) ≤ 1 / Real.sqrt 2 :=
sorry

end max_value_A_l1280_128012


namespace combination_15_5_l1280_128019

theorem combination_15_5 : 
  ∀ (n r : ℕ), n = 15 → r = 5 → n.choose r = 3003 :=
by
  intro n r h1 h2
  rw [h1, h2]
  exact Nat.choose_eq_factorial_div_factorial (by norm_num)

end combination_15_5_l1280_128019


namespace floor_eq_correct_l1280_128015

theorem floor_eq_correct (y : ℝ) (h : ⌊y⌋ + y = 17 / 4) : y = 9 / 4 :=
sorry

end floor_eq_correct_l1280_128015


namespace cubic_roots_number_l1280_128083

noncomputable def determinant_cubic (a b c d : ℝ) (x : ℝ) : ℝ :=
  x * (x^2 + a^2) + c * (b * x + a * b) - b * (c * a - b * x)

theorem cubic_roots_number (a b c d : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) :
  ∃ roots : ℕ, (roots = 1 ∨ roots = 3) :=
  sorry

end cubic_roots_number_l1280_128083


namespace tax_rate_correct_l1280_128006

def total_value : ℝ := 1720
def non_taxable_amount : ℝ := 600
def tax_paid : ℝ := 89.6

def taxable_amount : ℝ := total_value - non_taxable_amount

theorem tax_rate_correct : (tax_paid / taxable_amount) * 100 = 8 := by
  sorry

end tax_rate_correct_l1280_128006


namespace total_cost_of_cable_l1280_128008

-- Defining the conditions as constants
def east_west_streets := 18
def east_west_length := 2
def north_south_streets := 10
def north_south_length := 4
def cable_per_mile_street := 5
def cost_per_mile_cable := 2000

-- The theorem contains the problem statement and asserts the answer
theorem total_cost_of_cable :
  (east_west_streets * east_west_length + north_south_streets * north_south_length) * cable_per_mile_street * cost_per_mile_cable = 760000 := 
  sorry

end total_cost_of_cable_l1280_128008


namespace problem_l1280_128089

noncomputable def a : Real := Real.sin (14 * Real.pi / 180) + Real.cos (14 * Real.pi / 180)
noncomputable def b : Real := Real.sin (16 * Real.pi / 180) + Real.cos (16 * Real.pi / 180)
noncomputable def c : Real := Real.sqrt 6 / 2

theorem problem :
  a < c ∧ c < b := by
  sorry

end problem_l1280_128089


namespace largest_n_for_factorable_poly_l1280_128072

theorem largest_n_for_factorable_poly :
  ∃ n : ℤ, (∀ A B : ℤ, (3 * B + A = n) ∧ (A * B = 72) → (A = 1 ∧ B = 72 ∧ n = 217)) ∧
           (∀ A B : ℤ, A * B = 72 → 3 * B + A ≤ 217) :=
by
  sorry

end largest_n_for_factorable_poly_l1280_128072


namespace calculate_selling_price_l1280_128003

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 20
noncomputable def profit_percent : ℝ := 22.448979591836732

noncomputable def total_cost : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := (profit_percent / 100) * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem calculate_selling_price : selling_price = 300 := by
  sorry

end calculate_selling_price_l1280_128003


namespace minute_hand_position_l1280_128080

theorem minute_hand_position (t : ℕ) (h_start : t = 2022) :
  let cycle_minutes := 8
  let net_movement_per_cycle := 2
  let full_cycles := t / cycle_minutes
  let remaining_minutes := t % cycle_minutes
  let full_cycles_movement := full_cycles * net_movement_per_cycle
  let extra_movement := if remaining_minutes <= 5 then remaining_minutes else 5 - (remaining_minutes - 5)
  let total_movement := full_cycles_movement + extra_movement
  (total_movement % 60) = 28 :=
by {
  sorry
}

end minute_hand_position_l1280_128080


namespace sum_of_arithmetic_sequence_l1280_128049

theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : S n = n * (a 1 + a n) / 2)
  (h3 : a 2 + a 5 + a 11 = 6) :
  S 11 = 22 :=
sorry

end sum_of_arithmetic_sequence_l1280_128049


namespace solve_system_l1280_128060

theorem solve_system (x y : ℚ) (h1 : 6 * x = -9 - 3 * y) (h2 : 4 * x = 5 * y - 34) : x = 1/2 ∧ y = -4 :=
by
  sorry

end solve_system_l1280_128060


namespace simplest_fraction_sum_l1280_128037

theorem simplest_fraction_sum (c d : ℕ) (h1 : 0.325 = (c:ℚ)/d) (h2 : Int.gcd c d = 1) : c + d = 53 :=
by sorry

end simplest_fraction_sum_l1280_128037


namespace loaf_slices_l1280_128082

theorem loaf_slices (S : ℕ) (T : ℕ) : 
  (S - 7 = 2 * T + 3) ∧ (S ≥ 20) → S = 20 :=
by
  sorry

end loaf_slices_l1280_128082


namespace translate_right_one_unit_l1280_128017

theorem translate_right_one_unit (x y : ℤ) (hx : x = 4) (hy : y = -3) : (x + 1, y) = (5, -3) :=
by
  -- The proof would go here
  sorry

end translate_right_one_unit_l1280_128017


namespace smallest_number_l1280_128044

-- Define the numbers
def A := 5.67823
def B := 5.67833333333 -- repeating decimal
def C := 5.67838383838 -- repeating decimal
def D := 5.67837837837 -- repeating decimal
def E := 5.6783678367  -- repeating decimal

-- The Lean statement to prove that E is the smallest
theorem smallest_number : E < A ∧ E < B ∧ E < C ∧ E < D :=
by
  sorry

end smallest_number_l1280_128044


namespace min_packs_126_l1280_128045

-- Define the sizes of soda packs
def pack_sizes : List ℕ := [6, 12, 24, 48]

-- Define the total number of cans required
def total_cans : ℕ := 126

-- Define a function to calculate the minimum number of packs required
noncomputable def min_packs_to_reach_target (target : ℕ) (sizes : List ℕ) : ℕ :=
sorry -- Implementation will be complex dynamic programming or greedy algorithm

-- The main theorem statement to prove
theorem min_packs_126 (P : ℕ) (h1 : (min_packs_to_reach_target total_cans pack_sizes) = P) : P = 4 :=
sorry -- Proof not required

end min_packs_126_l1280_128045


namespace horses_eat_oats_twice_a_day_l1280_128010

-- Define the main constants and assumptions
def number_of_horses : ℕ := 4
def oats_per_meal : ℕ := 4
def grain_per_day : ℕ := 3
def total_food : ℕ := 132
def duration_in_days : ℕ := 3

-- Main theorem statement
theorem horses_eat_oats_twice_a_day (x : ℕ) (h : duration_in_days * number_of_horses * (oats_per_meal * x + grain_per_day) = total_food) : x = 2 := 
sorry

end horses_eat_oats_twice_a_day_l1280_128010


namespace brother_pays_correct_amount_l1280_128067

-- Definition of constants and variables
def friend_per_day := 5
def cousin_per_day := 4
def total_amount_collected := 119
def days := 7
def brother_per_day := 8

-- Statement of the theorem to be proven
theorem brother_pays_correct_amount :
  friend_per_day * days + cousin_per_day * days + brother_per_day * days = total_amount_collected :=
by {
  sorry
}

end brother_pays_correct_amount_l1280_128067


namespace fraction_shaded_area_l1280_128057

theorem fraction_shaded_area (PX XQ : ℝ) (PA PR PQ : ℝ) (h1 : PX = 1) (h2 : 3 * XQ = PX) (h3 : PQ = PR) (h4 : PA = 1) (h5 : PA + AR = PR) (h6 : PR = 4):
  (3 / 16 : ℝ) = 0.375 :=
by
  -- proof here
  sorry

end fraction_shaded_area_l1280_128057


namespace carrots_thrown_out_l1280_128013

variable (x : ℕ)

theorem carrots_thrown_out :
  let initial_carrots := 23
  let picked_later := 47
  let total_carrots := 60
  initial_carrots - x + picked_later = total_carrots → x = 10 :=
by
  intros
  sorry

end carrots_thrown_out_l1280_128013


namespace cylinder_volume_transformation_l1280_128099

variable (r h : ℝ)
variable (V_original : ℝ)
variable (V_new : ℝ)

noncomputable def original_volume : ℝ := Real.pi * r^2 * h

noncomputable def new_volume : ℝ := Real.pi * (3 * r)^2 * (2 * h)

theorem cylinder_volume_transformation 
  (h_original : original_volume r h = 15) :
  new_volume r h = 270 :=
by
  unfold original_volume at h_original
  unfold new_volume
  sorry

end cylinder_volume_transformation_l1280_128099


namespace range_of_a_l1280_128078

theorem range_of_a 
  (x y a : ℝ) 
  (hx_pos : 0 < x) 
  (hy_pos : 0 < y) 
  (hxy : x + y + 3 = x * y) 
  (h_a : ∀ x y : ℝ, (x + y)^2 - a * (x + y) + 1 ≥ 0) :
  a ≤ 37 / 6 := 
sorry

end range_of_a_l1280_128078


namespace roots_modulus_less_than_one_l1280_128071

theorem roots_modulus_less_than_one
  (A B C D : ℝ)
  (h1 : ∀ x, x^2 + A * x + B = 0 → |x| < 1)
  (h2 : ∀ x, x^2 + C * x + D = 0 → |x| < 1) :
  ∀ x, x^2 + (A + C) / 2 * x + (B + D) / 2 = 0 → |x| < 1 :=
by
  sorry

end roots_modulus_less_than_one_l1280_128071


namespace number_divided_by_005_l1280_128076

theorem number_divided_by_005 (number : ℝ) (h : number / 0.05 = 1500) : number = 75 :=
sorry

end number_divided_by_005_l1280_128076


namespace periodic_even_l1280_128073

noncomputable def f : ℝ → ℝ := sorry  -- We assume the existence of such a function.

variables {α β : ℝ}  -- acute angles of a right triangle

-- Function properties
theorem periodic_even (h_periodic: ∀ x: ℝ, f (x + 2) = f x)
  (h_even: ∀ x: ℝ, f (-x) = f x)
  (h_decreasing: ∀ x y: ℝ, -3 ≤ x ∧ x < y ∧ y ≤ -2 → f x > f y)
  (h_inc_interval_0_1: ∀ x y: ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y)
  (ha: 0 < α ∧ α < π / 2)
  (hb: 0 < β ∧ β < π / 2)
  (h_sum_right_triangle: α + β = π / 2): f (Real.sin α) > f (Real.cos β) :=
sorry

end periodic_even_l1280_128073


namespace sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l1280_128096

noncomputable def compare_sq_roots_sum : Prop := 
  (Real.sqrt 11 + Real.sqrt 3) < (Real.sqrt 9 + Real.sqrt 5)

theorem sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5 :
  compare_sq_roots_sum :=
sorry

end sqrt11_plus_sqrt3_lt_sqrt9_plus_sqrt5_l1280_128096


namespace selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l1280_128042

-- Problem conditions
def cost_price : ℕ := 70
def max_price : ℕ := 99
def initial_price : ℕ := 110
def initial_sales : ℕ := 20
def price_drop_rate : ℕ := 1
def sales_increase_rate : ℕ := 2
def sales_increase_per_yuan : ℕ := 2
def profit_target : ℕ := 1200

-- Selling price for given sales volume
def selling_price_for_sales_volume (sales_volume : ℕ) : ℕ :=
  initial_price - (sales_volume - initial_sales) / sales_increase_per_yuan

-- Functional relationship between sales volume (y) and price (x)
def sales_volume_function (x : ℕ) : ℕ :=
  initial_sales + sales_increase_rate * (initial_price - x)

-- Profit for given price and resulting sales volume
def daily_profit (x : ℕ) : ℤ :=
  (x - cost_price) * (sales_volume_function x)

-- Part 1: Selling price for 30 items sold
theorem selling_price_30_items : selling_price_for_sales_volume 30 = 105 :=
by
  sorry

-- Part 2: Functional relationship between sales volume and selling price
theorem sales_volume_functional_relationship (x : ℕ) (hx : 70 ≤ x ∧ x ≤ 99) :
  sales_volume_function x = 240 - 2 * x :=
by
  sorry

-- Part 3: Selling price for a daily profit of 1200 yuan
theorem selling_price_for_1200_profit {x : ℕ} (hx : 70 ≤ x ∧ x ≤ 99) :
  daily_profit x = 1200 → x = 90 :=
by
  sorry

end selling_price_30_items_sales_volume_functional_relationship_selling_price_for_1200_profit_l1280_128042


namespace marble_probability_correct_l1280_128070

noncomputable def marble_probability : ℚ :=
  let total_ways := (Nat.choose 20 4 : ℚ)
  let ways_two_red := (Nat.choose 12 2 : ℚ)
  let ways_two_blue := (Nat.choose 8 2 : ℚ)
  (ways_two_red * ways_two_blue) / total_ways

theorem marble_probability_correct : marble_probability = 56 / 147 :=
by
  -- Note: the proof is omitted as per instructions
  sorry

end marble_probability_correct_l1280_128070
