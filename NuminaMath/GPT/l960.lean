import Mathlib

namespace inequality_proof_l960_96027

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 1) :
  ((1 / x^2 - x) * (1 / y^2 - y) * (1 / z^2 - z) ≥ (26 / 3)^3) :=
by sorry

end inequality_proof_l960_96027


namespace fish_total_after_transfer_l960_96024

-- Definitions of the initial conditions
def lilly_initial : ℕ := 10
def rosy_initial : ℕ := 9
def jack_initial : ℕ := 15
def fish_transferred : ℕ := 2

-- Total fish after Lilly transfers 2 fish to Jack
theorem fish_total_after_transfer : (lilly_initial - fish_transferred) + rosy_initial + (jack_initial + fish_transferred) = 34 := by
  sorry

end fish_total_after_transfer_l960_96024


namespace quantity_of_milk_in_original_mixture_l960_96029

variable (M W : ℕ)

-- Conditions
def ratio_original : Prop := M = 2 * W
def ratio_after_adding_water : Prop := M * 5 = 6 * (W + 10)

theorem quantity_of_milk_in_original_mixture
  (h1 : ratio_original M W)
  (h2 : ratio_after_adding_water M W) :
  M = 30 := by
  sorry

end quantity_of_milk_in_original_mixture_l960_96029


namespace find_n_l960_96038

theorem find_n (x y n : ℝ) (h1 : 2 * x - 5 * y = 3 * n + 7) (h2 : x - 3 * y = 4) 
  (h3 : x = y):
  n = -1 / 3 := 
by 
  sorry

end find_n_l960_96038


namespace ratio_of_girls_to_boys_l960_96021

theorem ratio_of_girls_to_boys (x y : ℕ) (h1 : x + y = 28) (h2 : x - y = 4) : x = 16 ∧ y = 12 ∧ x / y = 4 / 3 :=
by
  sorry

end ratio_of_girls_to_boys_l960_96021


namespace find_m_l960_96072

theorem find_m (x y m : ℤ) (h1 : x = 3) (h2 : y = 1) (h3 : x - m * y = 1) : m = 2 :=
by
  -- Proof goes here
  sorry

end find_m_l960_96072


namespace time_to_install_rest_of_windows_l960_96034

-- Definition of the given conditions:
def num_windows_needed : ℕ := 10
def num_windows_installed : ℕ := 6
def install_time_per_window : ℕ := 5

-- Statement that we aim to prove:
theorem time_to_install_rest_of_windows :
  install_time_per_window * (num_windows_needed - num_windows_installed) = 20 := by
  sorry

end time_to_install_rest_of_windows_l960_96034


namespace correct_calculation_l960_96014

variable (a b : ℝ)

theorem correct_calculation : (-a^3)^2 = a^6 := 
by 
  sorry

end correct_calculation_l960_96014


namespace positive_difference_of_two_numbers_l960_96025

theorem positive_difference_of_two_numbers (x y : ℝ)
  (h1 : x + y = 10) (h2 : x ^ 2 - y ^ 2 = 40) : abs (x - y) = 4 := 
by
  sorry

end positive_difference_of_two_numbers_l960_96025


namespace sarah_reads_40_words_per_minute_l960_96001

-- Define the conditions as constants
def words_per_page := 100
def pages_per_book := 80
def reading_hours := 20
def number_of_books := 6

-- Convert hours to minutes
def total_reading_time := reading_hours * 60

-- Calculate the total number of words in one book
def words_per_book := words_per_page * pages_per_book

-- Calculate the total number of words in all books
def total_words := words_per_book * number_of_books

-- Define the words read per minute
def words_per_minute := total_words / total_reading_time

-- Theorem statement: Sarah reads 40 words per minute
theorem sarah_reads_40_words_per_minute : words_per_minute = 40 :=
by
  sorry

end sarah_reads_40_words_per_minute_l960_96001


namespace solve_inequality_l960_96026

open Set

variable {f : ℝ → ℝ}
open Function

theorem solve_inequality (h_inc : ∀ x y, 0 < x → 0 < y → x < y → f x < f y)
  (h_func_eq : ∀ x y, 0 < x → 0 < y → f (x / y) = f x - f y)
  (h_f3 : f 3 = 1)
  (x : ℝ) (hx_pos : 0 < x)
  (hx_ge : x > 5)
  (h_ineq : f x - f (1 / (x - 5)) ≥ 2) :
  x ≥ (5 + Real.sqrt 61) / 2 := sorry

end solve_inequality_l960_96026


namespace clock_correction_time_l960_96049

theorem clock_correction_time :
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  correction = 138.75 :=
by
  let time_loss_per_day : ℝ := 15 / 60
  let days_elapsed : ℝ := 9 + 6 / 24
  let total_time_loss : ℝ := (15 / 1440) * (days_elapsed * 24)
  let correction : ℝ := total_time_loss * 60
  have : correction = 138.75 := sorry
  exact this

end clock_correction_time_l960_96049


namespace product_of_three_numbers_l960_96022

theorem product_of_three_numbers:
  ∃ (a b c : ℚ), 
    a + b + c = 30 ∧ 
    a = 2 * (b + c) ∧ 
    b = 5 * c ∧ 
    a * b * c = 2500 / 9 :=
by {
  sorry
}

end product_of_three_numbers_l960_96022


namespace cos_beta_l960_96062

variable (α β : ℝ)
variable (hα : 0 < α ∧ α < π / 2)
variable (hβ : 0 < β ∧ β < π / 2)
variable (h1 : Real.sin α = 3 / 5)
variable (h2 : Real.cos (α + β) = 5 / 13)

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 3 / 5) (h2 : Real.cos (α + β) = 5 / 13) : 
  Real.cos β = 56 / 65 := by
  sorry

end cos_beta_l960_96062


namespace stream_speed_l960_96061

variable (v : ℝ)

def effective_speed_downstream (v : ℝ) : ℝ := 7.5 + v
def effective_speed_upstream (v : ℝ) : ℝ := 7.5 - v 

theorem stream_speed : (7.5 - v) / (7.5 + v) = 1 / 2 → v = 2.5 :=
by
  intro h
  -- Proof will be resolved here
  sorry

end stream_speed_l960_96061


namespace seq_period_3_l960_96074

def seq (a : ℕ → ℚ) := ∀ n, 
  (0 ≤ a n ∧ a n < 1) ∧ (
  (0 ≤ a n ∧ a n < 1/2 → a (n+1) = 2 * a n) ∧ 
  (1/2 ≤ a n ∧ a n < 1 → a (n+1) = 2 * a n - 1))

theorem seq_period_3 (a : ℕ → ℚ) (h : seq a) (h1 : a 1 = 6 / 7) : 
  a 2016 = 3 / 7 := 
sorry

end seq_period_3_l960_96074


namespace smallest_percent_increase_l960_96007

-- Define the values of each question
def question_values : List ℕ :=
  [150, 250, 400, 600, 1100, 2300, 4700, 9500, 19000, 38000, 76000, 150000, 300000, 600000, 1200000]

-- Define a function to calculate the percent increase between two questions
def percent_increase (v1 v2 : ℕ) : Float :=
  ((v2 - v1).toFloat / v1.toFloat) * 100

-- Define the specific question transitions and their percent increases
def percent_increase_1_to_4 : Float := percent_increase question_values[0] question_values[3]  -- Question 1 to 4
def percent_increase_2_to_6 : Float := percent_increase question_values[1] question_values[5]  -- Question 2 to 6
def percent_increase_5_to_10 : Float := percent_increase question_values[4] question_values[9]  -- Question 5 to 10
def percent_increase_9_to_15 : Float := percent_increase question_values[8] question_values[14] -- Question 9 to 15

-- Prove that the smallest percent increase is from Question 1 to 4
theorem smallest_percent_increase :
  percent_increase_1_to_4 < percent_increase_2_to_6 ∧
  percent_increase_1_to_4 < percent_increase_5_to_10 ∧
  percent_increase_1_to_4 < percent_increase_9_to_15 :=
by
  sorry

end smallest_percent_increase_l960_96007


namespace perfect_cube_divisor_l960_96094

theorem perfect_cube_divisor (a b : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a^2 + 3*a*b + 3*b^2 - 1 ∣ a + b^3) :
  ∃ k > 1, ∃ m : ℕ, a^2 + 3*a*b + 3*b^2 - 1 = k^3 * m := 
sorry

end perfect_cube_divisor_l960_96094


namespace min_value_of_expression_l960_96068

noncomputable def min_expression_value (y : ℝ) (hy : y > 2) : ℝ :=
  (y^2 + y + 1) / Real.sqrt (y - 2)

theorem min_value_of_expression (y : ℝ) (hy : y > 2) :
  min_expression_value y hy = 3 * Real.sqrt 35 :=
sorry

end min_value_of_expression_l960_96068


namespace sector_area_is_2_l960_96013

-- Definition of the sector's properties
def sector_perimeter (r : ℝ) (θ : ℝ) : ℝ := r * θ + 2 * r

def sector_area (r : ℝ) (θ : ℝ) : ℝ := 0.5 * r^2 * θ

-- Theorem stating that the area of the sector is 2 cm² given the conditions
theorem sector_area_is_2 (r θ : ℝ) (h1 : sector_perimeter r θ = 6) (h2 : θ = 1) : sector_area r θ = 2 :=
by
  sorry

end sector_area_is_2_l960_96013


namespace chosen_number_l960_96080

theorem chosen_number (x : ℕ) (h : 5 * x - 138 = 102) : x = 48 :=
sorry

end chosen_number_l960_96080


namespace sum_series_eq_one_third_l960_96015

theorem sum_series_eq_one_third :
  ∑' n : ℕ, (if h : n > 0 then (2^n / (1 + 2^n + 2^(n + 1) + 2^(2 * n + 1))) else 0) = 1 / 3 :=
by
  sorry

end sum_series_eq_one_third_l960_96015


namespace solve_for_x_l960_96079

noncomputable def solve_equation (x : ℝ) : Prop := 
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = 3 * x / (3 * x - 2) ∧ x ≠ 2 / 3

theorem solve_for_x (x : ℝ) (h : solve_equation x) : x = (Real.sqrt 6) / 3 ∨ x = - (Real.sqrt 6) / 3 := 
  sorry

end solve_for_x_l960_96079


namespace area_reflected_arcs_l960_96035

theorem area_reflected_arcs (s : ℝ) (h : s = 2) : 
  ∃ A, A = 2 * Real.pi * Real.sqrt 2 - 8 :=
by
  -- constants
  let r := Real.sqrt (2 * Real.sqrt 2)
  let sector_area := Real.pi * r^2 / 8
  let triangle_area := 1 -- Equilateral triangle properties
  let reflected_arc_area := sector_area - triangle_area
  let total_area := 8 * reflected_arc_area
  use total_area
  sorry

end area_reflected_arcs_l960_96035


namespace no_ingredient_pies_max_l960_96023

theorem no_ingredient_pies_max :
  ∃ (total apple blueberry cream chocolate no_ingredient : ℕ),
    total = 48 ∧
    apple = 24 ∧
    blueberry = 16 ∧
    cream = 18 ∧
    chocolate = 12 ∧
    no_ingredient = total - (apple + blueberry + chocolate - min apple blueberry - min apple chocolate - min blueberry chocolate) - cream ∧
    no_ingredient = 10 := sorry

end no_ingredient_pies_max_l960_96023


namespace capacity_of_first_bucket_is_3_l960_96063

variable (C : ℝ)

theorem capacity_of_first_bucket_is_3 
  (h1 : 48 / C = 48 / 3 - 4) : 
  C = 3 := 
  sorry

end capacity_of_first_bucket_is_3_l960_96063


namespace prob_draw_l960_96003

-- Define the probabilities as constants
def prob_A_winning : ℝ := 0.4
def prob_A_not_losing : ℝ := 0.9

-- Prove that the probability of a draw is 0.5
theorem prob_draw : prob_A_not_losing - prob_A_winning = 0.5 :=
by sorry

end prob_draw_l960_96003


namespace solve_for_a_l960_96067

theorem solve_for_a (a : ℝ) : 
  (2 * a + 16 + 3 * a - 8) / 2 = 69 → a = 26 :=
by
  sorry

end solve_for_a_l960_96067


namespace equal_share_candy_l960_96028

theorem equal_share_candy :
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  total_candy / number_of_people = 7 :=
by
  let hugh : ℕ := 8
  let tommy : ℕ := 6
  let melany : ℕ := 7
  let total_candy := hugh + tommy + melany
  let number_of_people := 3
  show total_candy / number_of_people = 7
  sorry

end equal_share_candy_l960_96028


namespace find_a_if_g_even_l960_96091

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then x - 1 else if -2 ≤ x ∧ x ≤ 0 then -1 else 0

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x) + a * x

theorem find_a_if_g_even (a : ℝ) : (∀ x : ℝ, f x + a * x = f (-x) + a * (-x)) → a = -1/2 :=
by
  intro h
  sorry

end find_a_if_g_even_l960_96091


namespace shaded_total_area_l960_96058

theorem shaded_total_area:
  ∀ (r₁ r₂ r₃ : ℝ),
  π * r₁ ^ 2 = 100 * π →
  r₂ = r₁ / 2 →
  r₃ = r₂ / 2 →
  (1 / 2) * (π * r₁ ^ 2) + (1 / 2) * (π * r₂ ^ 2) + (1 / 2) * (π * r₃ ^ 2) = 65.625 * π :=
by
  intro r₁ r₂ r₃ h₁ h₂ h₃
  sorry

end shaded_total_area_l960_96058


namespace percentage_of_original_solution_l960_96039

-- Define the problem and conditions
variable (P : ℝ)
variable (h1 : (0.5 * P + 0.5 * 60) = 55)

-- The theorem to prove
theorem percentage_of_original_solution : P = 50 :=
by
  -- Proof will go here
  sorry

end percentage_of_original_solution_l960_96039


namespace lcm_is_only_function_l960_96018

noncomputable def f (x y : ℕ) : ℕ := Nat.lcm x y

theorem lcm_is_only_function 
    (f : ℕ → ℕ → ℕ)
    (h1 : ∀ x : ℕ, f x x = x) 
    (h2 : ∀ x y : ℕ, f x y = f y x) 
    (h3 : ∀ x y : ℕ, (x + y) * f x y = y * f x (x + y)) : 
  ∀ x y : ℕ, f x y = Nat.lcm x y := 
sorry

end lcm_is_only_function_l960_96018


namespace impossible_to_save_one_minute_for_60kmh_l960_96075

theorem impossible_to_save_one_minute_for_60kmh (v : ℝ) (h : v = 60) :
  ¬ ∃ (new_v : ℝ), 1 / new_v = (1 / 60) - 1 :=
by
  sorry

end impossible_to_save_one_minute_for_60kmh_l960_96075


namespace equivalent_solution_l960_96031

theorem equivalent_solution (c x : ℤ) 
    (h1 : 3 * x + 9 = 6)
    (h2 : c * x - 15 = -5)
    (hx : x = -1) :
    c = -10 :=
sorry

end equivalent_solution_l960_96031


namespace m_range_and_simplification_l960_96060

theorem m_range_and_simplification (x y m : ℝ)
  (h1 : (3 * (x + 1) / 2) + y = 2)
  (h2 : 3 * x - m = 2 * y)
  (hx : x ≤ 1)
  (hy : y ≤ 1) :
  (-3 ≤ m) ∧ (m ≤ 5) ∧ (|x - 1| + |y - 1| + |m + 3| + |m - 5| - |x + y - 2| = 8) := 
by sorry

end m_range_and_simplification_l960_96060


namespace orchard_total_mass_l960_96096

def num_gala_trees := 20
def yield_gala_tree := 120
def num_fuji_trees := 10
def yield_fuji_tree := 180
def num_redhaven_trees := 30
def yield_redhaven_tree := 55
def num_elberta_trees := 15
def yield_elberta_tree := 75

def total_mass_gala := num_gala_trees * yield_gala_tree
def total_mass_fuji := num_fuji_trees * yield_fuji_tree
def total_mass_redhaven := num_redhaven_trees * yield_redhaven_tree
def total_mass_elberta := num_elberta_trees * yield_elberta_tree

def total_mass_fruit := total_mass_gala + total_mass_fuji + total_mass_redhaven + total_mass_elberta

theorem orchard_total_mass : total_mass_fruit = 6975 := by
  sorry

end orchard_total_mass_l960_96096


namespace cube_volume_l960_96089

theorem cube_volume (A : ℝ) (h : A = 24) : 
  ∃ V : ℝ, V = 8 :=
by
  sorry

end cube_volume_l960_96089


namespace area_of_square_field_l960_96050

theorem area_of_square_field (side_length : ℕ) (h : side_length = 25) :
  side_length * side_length = 625 := by
  sorry

end area_of_square_field_l960_96050


namespace probability_of_blue_ball_l960_96095

theorem probability_of_blue_ball 
(P_red P_yellow P_blue : ℝ) 
(h_red : P_red = 0.48)
(h_yellow : P_yellow = 0.35) 
(h_prob : P_red + P_yellow + P_blue = 1) 
: P_blue = 0.17 := 
sorry

end probability_of_blue_ball_l960_96095


namespace arithmetic_sequence_second_term_l960_96004

theorem arithmetic_sequence_second_term (a d : ℝ) (h : a + (a + 2 * d) = 8) : a + d = 4 :=
sorry

end arithmetic_sequence_second_term_l960_96004


namespace cell_phone_plan_cost_l960_96069

theorem cell_phone_plan_cost:
  let base_cost : ℕ := 25
  let text_cost : ℕ := 8
  let extra_min_cost : ℕ := 12
  let texts_sent : ℕ := 150
  let hours_talked : ℕ := 27
  let extra_minutes := (hours_talked - 25) * 60
  let total_cost := (base_cost * 100) + (texts_sent * text_cost) + (extra_minutes * extra_min_cost)
  (total_cost = 5140) :=
by
  sorry

end cell_phone_plan_cost_l960_96069


namespace min_lines_to_separate_points_l960_96086

theorem min_lines_to_separate_points (m n : ℕ) (h_m : m = 8) (h_n : n = 8) : 
  (m - 1) + (n - 1) = 14 := by
  sorry

end min_lines_to_separate_points_l960_96086


namespace find_candy_bars_per_week_l960_96099

-- Define the conditions
variables (x : ℕ)

-- Condition: Kim's dad buys Kim x candy bars each week
def candies_bought := 16 * x

-- Condition: Kim eats one candy bar every 4 weeks
def candies_eaten := 16 / 4

-- Condition: After 16 weeks, Kim has saved 28 candy bars
def saved_candies := 28

-- The theorem we want to prove
theorem find_candy_bars_per_week : (16 * x - (16 / 4) = 28) → x = 2 := by
  -- We will skip the actual proof for now.
  sorry

end find_candy_bars_per_week_l960_96099


namespace hexagon_label_count_l960_96076

def hexagon_label (s : Finset ℕ) (a b c d e f g : ℕ) : Prop :=
  s = Finset.range 8 ∧ 
  (a ∈ s) ∧ (b ∈ s) ∧ (c ∈ s) ∧ (d ∈ s) ∧ (e ∈ s) ∧ (f ∈ s) ∧ (g ∈ s) ∧
  a + b + c + d + e + f + g = 28 ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ 
  d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
  e ≠ f ∧ e ≠ g ∧
  f ≠ g ∧
  a + g + d = b + g + e ∧ b + g + e = c + g + f

theorem hexagon_label_count : ∃ s a b c d e f g, hexagon_label s a b c d e f g ∧ 
  (s.card = 8) ∧ (a + g + d = 10) ∧ (b + g + e = 10) ∧ (c + g + f = 10) ∧ 
  144 = 3 * 48 :=
sorry

end hexagon_label_count_l960_96076


namespace trig_identity_proof_l960_96088

noncomputable def value_expr : ℝ :=
  (2 * Real.cos (10 * Real.pi / 180) - Real.sin (20 * Real.pi / 180)) / Real.sin (70 * Real.pi / 180)

theorem trig_identity_proof : value_expr = Real.sqrt 3 :=
by
  sorry

end trig_identity_proof_l960_96088


namespace dilation_image_l960_96093

theorem dilation_image :
  let z_0 := (1 : ℂ) + 2 * I
  let k := (2 : ℂ)
  let z_1 := (3 : ℂ) + I
  let z := z_0 + k * (z_1 - z_0)
  z = 5 :=
by
  sorry

end dilation_image_l960_96093


namespace parallel_vectors_x_value_l960_96040

variable {x : ℝ}

theorem parallel_vectors_x_value (h : (1 / x) = (2 / -6)) : x = -3 := sorry

end parallel_vectors_x_value_l960_96040


namespace exponentiation_and_division_l960_96020

theorem exponentiation_and_division (a b c : ℕ) (h : a = 6) (h₂ : b = 3) (h₃ : c = 15) :
  9^a * 3^b / 3^c = 1 := by
  sorry

end exponentiation_and_division_l960_96020


namespace tailoring_cost_is_200_l960_96006

variables 
  (cost_first_suit : ℕ := 300)
  (total_paid : ℕ := 1400)

def cost_of_second_suit (tailoring_cost : ℕ) := 3 * cost_first_suit + tailoring_cost

theorem tailoring_cost_is_200 (T : ℕ) (h1 : cost_first_suit = 300) (h2 : total_paid = 1400) 
  (h3 : total_paid = cost_first_suit + cost_of_second_suit T) : 
  T = 200 := 
by 
  sorry

end tailoring_cost_is_200_l960_96006


namespace inequality_proof_l960_96082

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a + b + c ≤ (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ∧ 
    (a^2 + b^2) / (2 * c) + (a^2 + c^2) / (2 * b) + (b^2 + c^2) / (2 * a) ≤ (a^3 / (b * c)) + (b^3 / (a * c)) + (c^3 / (a * b)) := 
by
  sorry

end inequality_proof_l960_96082


namespace rachel_bought_3_tables_l960_96009

-- Definitions from conditions
def chairs := 7
def minutes_per_furniture := 4
def total_minutes := 40

-- Define the number of tables Rachel bought
def number_of_tables (chairs : ℕ) (minutes_per_furniture : ℕ) (total_minutes : ℕ) : ℕ :=
  (total_minutes - (chairs * minutes_per_furniture)) / minutes_per_furniture

-- Lean theorem stating the proof problem
theorem rachel_bought_3_tables : number_of_tables chairs minutes_per_furniture total_minutes = 3 :=
by
  sorry

end rachel_bought_3_tables_l960_96009


namespace inequality_solution_value_l960_96046

theorem inequality_solution_value 
  (a : ℝ)
  (h : ∀ x, (1 < x ∧ x < 2) ↔ (ax / (x - 1) > 1)) :
  a = 1 / 2 :=
sorry

end inequality_solution_value_l960_96046


namespace simplify_expression_l960_96081

theorem simplify_expression (x : ℝ) (h : x ≠ 1) : 
  ((x^2 + 1) / (x - 1) - 2 * x / (x - 1)) = x - 1 :=
by
  -- Proof goes here.
  sorry

end simplify_expression_l960_96081


namespace range_of_a_extrema_of_y_l960_96057

variable {a b c : ℝ}

def setA (a b c : ℝ) : Prop := a^2 - b * c - 8 * a + 7 = 0
def setB (a b c : ℝ) : Prop := b^2 + c^2 + b * c - b * a + b = 0

theorem range_of_a (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) : 1 ≤ a ∧ a ≤ 9 :=
sorry

theorem extrema_of_y (h: ∃ a b c : ℝ, setA a b c ∧ setB a b c) 
  (y : ℝ) 
  (hy1 : y = a * b + b * c + a * c)
  (hy2 : ∀ x y z : ℝ, setA x y z → setB x y z → y = x * y + y * z + x * z) : 
  y = 88 ∨ y = -56 :=
sorry

end range_of_a_extrema_of_y_l960_96057


namespace even_function_cos_sin_l960_96000

theorem even_function_cos_sin {f : ℝ → ℝ}
  (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = Real.cos (3 * x) + Real.sin (2 * x)) :
  ∀ x, x > 0 → f x = Real.cos (3 * x) - Real.sin (2 * x) := by
  sorry

end even_function_cos_sin_l960_96000


namespace picture_area_l960_96083

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1) 
  (h : (2 * x + 5) * (y + 4) - x * y = 84) : x * y = 15 :=
by
  sorry

end picture_area_l960_96083


namespace total_course_selection_schemes_l960_96048

theorem total_course_selection_schemes :
  let PE_courses := 4
  let Art_courses := 4
  let total_courses := PE_courses + Art_courses
  let choose_two :=
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 1)
  let choose_three :=
    (Nat.choose PE_courses 2) * (Nat.choose Art_courses 1) +
    (Nat.choose PE_courses 1) * (Nat.choose Art_courses 2)
  total_courses = 8
  ∧ (choose_two + choose_three = 64) :=
by
  sorry

end total_course_selection_schemes_l960_96048


namespace probability_scoring_80_or_above_probability_failing_exam_l960_96011

theorem probability_scoring_80_or_above (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C) = 0.69 :=
by
  intros hB hC hD hE
  sorry

theorem probability_failing_exam (P : Set ℝ → ℝ) (B C D E : Set ℝ) :
  P B = 0.18 →
  P C = 0.51 →
  P D = 0.15 →
  P E = 0.09 →
  P (B ∪ C ∪ D ∪ E) = 0.93 →
  1 - P (B ∪ C ∪ D ∪ E) = 0.07 :=
by
  intros hB hC hD hE hBCDE
  sorry

end probability_scoring_80_or_above_probability_failing_exam_l960_96011


namespace ratio_length_to_width_is_3_l960_96017

-- Define the conditions given in the problem
def area_of_garden : ℕ := 768
def width_of_garden : ℕ := 16

-- Define the length calculated from the area and width
def length_of_garden := area_of_garden / width_of_garden

-- Define the ratio to be proven
def ratio_of_length_to_width := length_of_garden / width_of_garden

-- Prove that the ratio is 3:1
theorem ratio_length_to_width_is_3 :
  ratio_of_length_to_width = 3 := by
  sorry

end ratio_length_to_width_is_3_l960_96017


namespace bathroom_visits_time_l960_96043

variable (t_8 : ℕ) (n8 : ℕ) (n6 : ℕ)

theorem bathroom_visits_time (h1 : t_8 = 20) (h2 : n8 = 8) (h3 : n6 = 6) :
  (t_8 / n8) * n6 = 15 := by
  sorry

end bathroom_visits_time_l960_96043


namespace infinite_double_perfect_squares_l960_96070

-- Definition of a double number
def is_double_number (n : ℕ) : Prop :=
  ∃ (k : ℕ), ∃ (d : ℕ), d ≠ 0 ∧ 10^k * d + d = n ∧ 10^k ≤ d ∧ d < 10^(k+1)

-- The theorem statement
theorem infinite_double_perfect_squares :
  ∃ (S : Set ℕ), (∀ n ∈ S, is_double_number n ∧ ∃ m, m * m = n) ∧
  Set.Infinite S :=
sorry

end infinite_double_perfect_squares_l960_96070


namespace water_loss_per_jump_l960_96090

def pool_capacity : ℕ := 2000 -- in liters
def jump_limit : ℕ := 1000
def clean_threshold : ℝ := 0.80

theorem water_loss_per_jump :
  (pool_capacity * (1 - clean_threshold)) * 1000 / jump_limit = 400 :=
by
  -- We prove that the water lost per jump in mL is 400
  sorry

end water_loss_per_jump_l960_96090


namespace arithmetic_sequence_a6_l960_96092

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1)
  (h_a1 : a 1 = 5) (h_a5 : a 5 = 1) : a 6 = 0 :=
by
  -- Definitions derived from conditions in the problem:
  -- 1. a : ℕ → ℤ : Sequence defined on ℕ with integer values.
  -- 2. h_arith : ∀ n, a (n+1) - a n = a 2 - a 1 : Arithmetic sequence property
  -- 3. h_a1 : a 1 = 5 : First term of the sequence is 5.
  -- 4. h_a5 : a 5 = 1 : Fifth term of the sequence is 1.
  sorry

end arithmetic_sequence_a6_l960_96092


namespace geometric_sequence_sum_l960_96053

theorem geometric_sequence_sum (a : ℕ → ℤ) (r : ℤ) (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 0 + a 1 + a 2 = 8)
  (h2 : a 3 + a 4 + a 5 = -4) :
  a 6 + a 7 + a 8 = 2 := 
sorry

end geometric_sequence_sum_l960_96053


namespace Mr_A_Mrs_A_are_normal_l960_96077

def is_knight (person : Type) : Prop := sorry
def is_liar (person : Type) : Prop := sorry
def is_normal (person : Type) : Prop := sorry

variable (Mr_A Mrs_A : Type)

axiom Mr_A_statement : is_normal Mrs_A → False
axiom Mrs_A_statement : is_normal Mr_A → False

theorem Mr_A_Mrs_A_are_normal :
  is_normal Mr_A ∧ is_normal Mrs_A :=
sorry

end Mr_A_Mrs_A_are_normal_l960_96077


namespace base3_to_base10_conversion_l960_96066

theorem base3_to_base10_conversion : ∀ n : ℕ, n = 120102 → (1 * 3^5 + 2 * 3^4 + 0 * 3^3 + 1 * 3^2 + 0 * 3^1 + 2 * 3^0) = 416 :=
by
  intro n hn
  sorry

end base3_to_base10_conversion_l960_96066


namespace father_age_l960_96045

theorem father_age (F D : ℕ) (h1 : F = 4 * D) (h2 : (F + 5) + (D + 5) = 50) : F = 32 :=
by
  sorry

end father_age_l960_96045


namespace find_intersection_point_l960_96042

theorem find_intersection_point :
  ∃ (x y z : ℝ), 
    ((∃ t : ℝ, x = 1 + 2 * t ∧ y = 1 - t ∧ z = -2 + 3 * t) ∧ 
    (4 * x + 2 * y - z - 11 = 0)) ∧ 
    (x = 3 ∧ y = 0 ∧ z = 1) :=
by
  sorry

end find_intersection_point_l960_96042


namespace total_seeds_l960_96010

-- Define the conditions given in the problem
def morningMikeTomato := 50
def morningMikePepper := 30

def morningTedTomato := 2 * morningMikeTomato
def morningTedPepper := morningMikePepper / 2

def morningSarahTomato := morningMikeTomato + 30
def morningSarahPepper := morningMikePepper + 30

def afternoonMikeTomato := 60
def afternoonMikePepper := 40

def afternoonTedTomato := afternoonMikeTomato - 20
def afternoonTedPepper := afternoonMikePepper

def afternoonSarahTomato := morningSarahTomato + 20
def afternoonSarahPepper := morningSarahPepper + 10

-- Prove that the total number of seeds planted is 685
theorem total_seeds (total: Nat) : 
    total = (
        (morningMikeTomato + afternoonMikeTomato) + 
        (morningTedTomato + afternoonTedTomato) + 
        (morningSarahTomato + afternoonSarahTomato) +
        (morningMikePepper + afternoonMikePepper) + 
        (morningTedPepper + afternoonTedPepper) + 
        (morningSarahPepper + afternoonSarahPepper)
    ) := 
    by 
        have tomato_seeds := (
            morningMikeTomato + afternoonMikeTomato +
            morningTedTomato + afternoonTedTomato + 
            morningSarahTomato + afternoonSarahTomato
        )
        have pepper_seeds := (
            morningMikePepper + afternoonMikePepper +
            morningTedPepper + afternoonTedPepper + 
            morningSarahPepper + afternoonSarahPepper
        )
        have total_seeds := tomato_seeds + pepper_seeds
        sorry

end total_seeds_l960_96010


namespace single_elimination_games_l960_96019

theorem single_elimination_games (n : ℕ) (h : n = 512) : (n - 1) = 511 :=
by
  sorry

end single_elimination_games_l960_96019


namespace lines_not_intersecting_may_be_parallel_or_skew_l960_96030

theorem lines_not_intersecting_may_be_parallel_or_skew (a b : ℝ × ℝ → Prop) 
  (h : ∀ x, ¬ (a x ∧ b x)) : 
  (∃ c d : ℝ × ℝ → Prop, a = c ∧ b = d) := 
sorry

end lines_not_intersecting_may_be_parallel_or_skew_l960_96030


namespace gcd_possible_values_l960_96036

theorem gcd_possible_values (a b : ℕ) (hab : a * b = 288) : 
  ∃ S : Finset ℕ, (∀ g : ℕ, g ∈ S ↔ ∃ p q r s : ℕ, p + r = 5 ∧ q + s = 2 ∧ g = 2^min p r * 3^min q s) 
  ∧ S.card = 14 := 
sorry

end gcd_possible_values_l960_96036


namespace expressions_not_equal_l960_96044

theorem expressions_not_equal (x : ℝ) (hx : x > 0) : 
  3 * x^x ≠ 2 * x^x + x^(2 * x) ∧ 
  x^(3 * x) ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^x ≠ 2 * x^x + x^(2 * x) ∧ 
  (3 * x)^(3 * x) ≠ 2 * x^x + x^(2 * x) :=
by 
  sorry

end expressions_not_equal_l960_96044


namespace central_angle_l960_96041

-- Definition: percentage corresponds to central angle
def percentage_equal_ratio (P : ℝ) (θ : ℝ) : Prop :=
  P = θ / 360

-- Theorem statement: Given that P = θ / 360, we want to prove θ = 360 * P
theorem central_angle (P θ : ℝ) (h : percentage_equal_ratio P θ) : θ = 360 * P :=
sorry

end central_angle_l960_96041


namespace parallel_resistor_problem_l960_96052

theorem parallel_resistor_problem
  (x : ℝ)
  (r : ℝ := 2.2222222222222223)
  (y : ℝ := 5) : 
  (1 / r = 1 / x + 1 / y) → x = 4 :=
by sorry

end parallel_resistor_problem_l960_96052


namespace clarinet_fraction_l960_96078

theorem clarinet_fraction 
  (total_flutes total_clarinets total_trumpets total_pianists total_band: ℕ)
  (percent_flutes : ℚ) (fraction_trumpets fraction_pianists : ℚ)
  (total_persons_in_band: ℚ)
  (flutes_got_in : total_flutes = 20)
  (clarinets_got_in : total_clarinets = 30)
  (trumpets_got_in : total_trumpets = 60)
  (pianists_got_in : total_pianists = 20)
  (band_got_in : total_band = 53)
  (percent_flutes_got_in: percent_flutes = 0.8)
  (fraction_trumpets_got_in: fraction_trumpets = 1/3)
  (fraction_pianists_got_in: fraction_pianists = 1/10)
  (persons_in_band: total_persons_in_band = 53) :
  (15 / 30 : ℚ) = (1 / 2 : ℚ) := 
by
  sorry

end clarinet_fraction_l960_96078


namespace balloons_lost_is_correct_l960_96097

def original_balloons : ℕ := 8
def current_balloons : ℕ := 6
def lost_balloons : ℕ := original_balloons - current_balloons

theorem balloons_lost_is_correct : lost_balloons = 2 := by
  sorry

end balloons_lost_is_correct_l960_96097


namespace least_subtraction_for_divisibility_l960_96087

def original_number : ℕ := 5474827

def required_subtraction : ℕ := 7

theorem least_subtraction_for_divisibility :
  ∃ k : ℕ, (original_number - required_subtraction) = 12 * k :=
sorry

end least_subtraction_for_divisibility_l960_96087


namespace find_k_value_l960_96037

theorem find_k_value :
  (∃ p q : ℝ → ℝ,
    (∀ x, p x = 3 * x + 5) ∧
    (∃ k : ℝ, (∀ x, q x = k * x + 3) ∧
      (p (-4) = -7) ∧ (q (-4) = -7) ∧ k = 2.5)) :=
by
  sorry

end find_k_value_l960_96037


namespace kamals_salary_change_l960_96073

theorem kamals_salary_change : 
  ∀ (S : ℝ), ((S * 0.5 * 1.3 * 0.8 - S) / S) * 100 = -48 :=
by
  intro S
  sorry

end kamals_salary_change_l960_96073


namespace marbles_end_of_day_l960_96055

theorem marbles_end_of_day :
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54 :=
by
  let initial_marbles := 40
  let lost_marbles_at_breakfast := 3
  let given_to_Susie_at_lunch := 5
  let new_marbles_from_mom := 12
  let returned_by_Susie := 2 * given_to_Susie_at_lunch
  show initial_marbles - lost_marbles_at_breakfast - given_to_Susie_at_lunch + new_marbles_from_mom + returned_by_Susie = 54
  sorry

end marbles_end_of_day_l960_96055


namespace roots_relationship_l960_96051

variable {a b c : ℝ} (h : a ≠ 0)

theorem roots_relationship (x y : ℝ) :
  (x = (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a) ∨ x = (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)) →
  (y = (-b + Real.sqrt (b^2 - 4*a*c)) / 2 ∨ y = (-b - Real.sqrt (b^2 - 4*a*c)) / 2) →
  (x = y / a) :=
by
  sorry

end roots_relationship_l960_96051


namespace max_sum_of_first_n_terms_l960_96033

variable {a : ℕ → ℝ} -- Define sequence a with index ℕ and real values
variable {d : ℝ}      -- Common difference for the arithmetic sequence

-- Conditions and question are formulated into the theorem statement
theorem max_sum_of_first_n_terms (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_diff_neg : d < 0)
  (h_a4_eq_a12 : (a 4)^2 = (a 12)^2) :
  n = 7 ∨ n = 8 := 
sorry

end max_sum_of_first_n_terms_l960_96033


namespace unit_circle_solution_l960_96016

noncomputable def unit_circle_point_x (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) : ℝ :=
  1 / 26

theorem unit_circle_solution (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) :
  unit_circle_point_x α hα hcos = 1 / 26 :=
by
  sorry

end unit_circle_solution_l960_96016


namespace total_students_in_Lansing_l960_96071

theorem total_students_in_Lansing :
  let num_schools_300 := 20
  let num_schools_350 := 30
  let num_schools_400 := 15
  let students_per_school_300 := 300
  let students_per_school_350 := 350
  let students_per_school_400 := 400
  (num_schools_300 * students_per_school_300 + num_schools_350 * students_per_school_350 + num_schools_400 * students_per_school_400 = 22500) := 
  sorry

end total_students_in_Lansing_l960_96071


namespace rectangular_prism_diagonals_l960_96059

theorem rectangular_prism_diagonals :
  let l := 3
  let w := 4
  let h := 5
  let face_diagonals := 6 * 2
  let space_diagonals := 4
  face_diagonals + space_diagonals = 16 := 
by
  sorry

end rectangular_prism_diagonals_l960_96059


namespace part1_part2_l960_96032

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (m n : ℝ) : f (m + n) = f m * f n
axiom positive_property (x : ℝ) (h : x > 0) : 0 < f x ∧ f x < 1

theorem part1 (x : ℝ) : f 0 = 1 ∧ (x < 0 → f x > 1) := by
  sorry

theorem part2 (x : ℝ) : 
  f (2 * x^2 - 4 * x - 1) < 1 ∧ f (x - 1) < 1 → x < -1/2 ∨ x > 2 := by
  sorry

end part1_part2_l960_96032


namespace find_S12_l960_96005

theorem find_S12 (S : ℕ → ℕ) (h1 : S 3 = 6) (h2 : S 9 = 15) : S 12 = 18 :=
by
  sorry

end find_S12_l960_96005


namespace area_parallelogram_l960_96098

theorem area_parallelogram (AE EB : ℝ) (SAEF SCEF SAEC SBEC SABC SABCD : ℝ) (h1 : SAE = 2 * EB)
  (h2 : SCEF = 1) (h3 : SAE == 2 * SCEF / 3) (h4 : SAEC == SAE + SCEF) 
  (h5 : SBEC == 1/2 * SAEC) (h6 : SABC == SAEC + SBEC) (h7 : SABCD == 2 * SABC) :
  SABCD = 5 := sorry

end area_parallelogram_l960_96098


namespace max_subjects_per_teacher_l960_96064

theorem max_subjects_per_teacher (maths physics chemistry : ℕ) (min_teachers : ℕ)
  (h_math : maths = 6) (h_physics : physics = 5) (h_chemistry : chemistry = 5) (h_min_teachers : min_teachers = 4) :
  (maths + physics + chemistry) / min_teachers = 4 :=
by
  -- the proof will be here
  sorry

end max_subjects_per_teacher_l960_96064


namespace valid_outfits_count_l960_96008

-- Definitions based on problem conditions
def shirts : Nat := 5
def pants : Nat := 6
def invalid_combination : Nat := 1

-- Problem statement
theorem valid_outfits_count : shirts * pants - invalid_combination = 29 := by 
  sorry

end valid_outfits_count_l960_96008


namespace problem_solution_l960_96065

theorem problem_solution (m n p : ℝ) 
  (h1 : 1 * m + 4 * p - 2 = 0) 
  (h2 : 2 * 1 - 5 * p + n = 0) 
  (h3 : (m / (-4)) * (2 / 5) = -1) :
  n = -12 :=
sorry

end problem_solution_l960_96065


namespace total_cost_of_digging_well_l960_96085

noncomputable def cost_of_digging (depth : ℝ) (diameter : ℝ) (cost_per_cubic_meter : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := Real.pi * (radius^2) * depth
  volume * cost_per_cubic_meter

theorem total_cost_of_digging_well :
  cost_of_digging 14 3 15 = 1484.4 :=
by
  sorry

end total_cost_of_digging_well_l960_96085


namespace average_salary_correct_l960_96002

def salary_A : ℕ := 8000
def salary_B : ℕ := 5000
def salary_C : ℕ := 15000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def number_of_people : ℕ := 5

def average_salary : ℕ := total_salary / number_of_people

theorem average_salary_correct : average_salary = 9000 := by
  -- proof is skipped
  sorry

end average_salary_correct_l960_96002


namespace karl_savings_proof_l960_96047

-- Definitions based on the conditions
def original_price_per_notebook : ℝ := 3.00
def sale_discount : ℝ := 0.25
def extra_discount_threshold : ℝ := 10
def extra_discount_rate : ℝ := 0.05

-- The number of notebooks Karl could have purchased instead
def notebooks_purchased : ℝ := 12

-- The total savings calculation
noncomputable def total_savings : ℝ := 
  let original_total := notebooks_purchased * original_price_per_notebook
  let discounted_price_per_notebook := original_price_per_notebook * (1 - sale_discount)
  let extra_discount := if notebooks_purchased > extra_discount_threshold then discounted_price_per_notebook * extra_discount_rate else 0
  let total_price_after_discounts := notebooks_purchased * discounted_price_per_notebook - notebooks_purchased * extra_discount
  original_total - total_price_after_discounts

-- Formal statement to prove
theorem karl_savings_proof : total_savings = 10.35 := 
  sorry

end karl_savings_proof_l960_96047


namespace remainder_of_hx10_divided_by_hx_is_6_l960_96056

noncomputable def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_hx10_divided_by_hx_is_6 : 
  let q := h (x ^ 10);
  q % h (x) = 6 := by
  sorry

end remainder_of_hx10_divided_by_hx_is_6_l960_96056


namespace solve_for_x_and_y_l960_96054

theorem solve_for_x_and_y (x y : ℝ) (h1 : x + y = 15) (h2 : x - y = 5) : x = 10 ∧ y = 5 :=
by
  sorry

end solve_for_x_and_y_l960_96054


namespace power_product_l960_96084

theorem power_product (m n : ℕ) (hm : 2 < m) (hn : 0 < n) : 
  (2^m - 1) * (2^n + 1) > 0 :=
by 
  sorry

end power_product_l960_96084


namespace number_of_clients_l960_96012

theorem number_of_clients (cars_clients_selects : ℕ)
                          (cars_selected_per_client : ℕ)
                          (each_car_selected_times : ℕ)
                          (total_cars : ℕ)
                          (h1 : total_cars = 18)
                          (h2 : cars_clients_selects = total_cars * each_car_selected_times)
                          (h3 : each_car_selected_times = 3)
                          (h4 : cars_selected_per_client = 3)
                          : total_cars * each_car_selected_times / cars_selected_per_client = 18 :=
by {
  sorry
}

end number_of_clients_l960_96012
