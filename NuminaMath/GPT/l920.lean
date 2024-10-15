import Mathlib

namespace NUMINAMATH_GPT_measured_weight_loss_l920_92054

variable (W : ℝ) (hW : W > 0)

noncomputable def final_weigh_in (initial_weight : ℝ) : ℝ :=
  (0.90 * initial_weight) * 1.02

theorem measured_weight_loss :
  final_weigh_in W = 0.918 * W → (W - final_weigh_in W) / W * 100 = 8.2 := 
by
  intro h
  unfold final_weigh_in at h
  -- skip detailed proof steps, focus on the statement
  sorry

end NUMINAMATH_GPT_measured_weight_loss_l920_92054


namespace NUMINAMATH_GPT_isosceles_base_length_l920_92045

theorem isosceles_base_length (x b : ℕ) (h1 : 2 * x + b = 40) (h2 : x = 15) : b = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_base_length_l920_92045


namespace NUMINAMATH_GPT_value_added_to_half_is_five_l920_92001

theorem value_added_to_half_is_five (n V : ℕ) (h₁ : n = 16) (h₂ : (1 / 2 : ℝ) * n + V = 13) : V = 5 := 
by 
  sorry

end NUMINAMATH_GPT_value_added_to_half_is_five_l920_92001


namespace NUMINAMATH_GPT_betty_eggs_per_teaspoon_vanilla_l920_92094

theorem betty_eggs_per_teaspoon_vanilla
  (sugar_cream_cheese_ratio : ℚ)
  (vanilla_cream_cheese_ratio : ℚ)
  (sugar_in_cups : ℚ)
  (eggs_used : ℕ)
  (expected_ratio : ℚ) :
  sugar_cream_cheese_ratio = 1/4 →
  vanilla_cream_cheese_ratio = 1/2 →
  sugar_in_cups = 2 →
  eggs_used = 8 →
  expected_ratio = 2 →
  (eggs_used / (sugar_in_cups * 4 * vanilla_cream_cheese_ratio)) = expected_ratio :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_betty_eggs_per_teaspoon_vanilla_l920_92094


namespace NUMINAMATH_GPT_incorrect_inequality_l920_92025

theorem incorrect_inequality (a b : ℝ) (h : a < b) : ¬ (-4 * a < -4 * b) :=
by sorry

end NUMINAMATH_GPT_incorrect_inequality_l920_92025


namespace NUMINAMATH_GPT_cubic_roots_identity_l920_92031

theorem cubic_roots_identity (p q r : ℝ) 
  (h1 : p + q + r = 0) 
  (h2 : p * q + q * r + r * p = -3) 
  (h3 : p * q * r = -2) : 
  p * (q - r) ^ 2 + q * (r - p) ^ 2 + r * (p - q) ^ 2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_cubic_roots_identity_l920_92031


namespace NUMINAMATH_GPT_grocer_initial_stock_l920_92022

noncomputable def initial_coffee_stock (x : ℝ) : Prop :=
  let initial_decaf := 0.20 * x
  let additional_coffee := 100
  let additional_decaf := 0.50 * additional_coffee
  let total_coffee := x + additional_coffee
  let total_decaf := initial_decaf + additional_decaf
  0.26 * total_coffee = total_decaf

theorem grocer_initial_stock :
  ∃ x : ℝ, initial_coffee_stock x ∧ x = 400 :=
by
  sorry

end NUMINAMATH_GPT_grocer_initial_stock_l920_92022


namespace NUMINAMATH_GPT_evaluate_expression_l920_92049

theorem evaluate_expression : (20 * 3 + 10) / (5 + 3) = 9 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l920_92049


namespace NUMINAMATH_GPT_line_equation_from_point_normal_l920_92089

theorem line_equation_from_point_normal :
  let M1 : ℝ × ℝ := (7, -8)
  let n : ℝ × ℝ := (-2, 3)
  ∃ C : ℝ, ∀ x y : ℝ, 2 * x - 3 * y + C = 0 ↔ (C = -38) := 
by
  sorry

end NUMINAMATH_GPT_line_equation_from_point_normal_l920_92089


namespace NUMINAMATH_GPT_sum_a_eq_9_l920_92072

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sum_a_eq_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 0 ≤ a2 ∧ a2 < 2) (h2 : 0 ≤ a3 ∧ a3 < 3) (h3 : 0 ≤ a4 ∧ a4 < 4)
  (h4 : 0 ≤ a5 ∧ a5 < 5) (h5 : 0 ≤ a6 ∧ a6 < 6) (h6 : 0 ≤ a7 ∧ a7 < 7)
  (h_eq : (5 : ℚ) / 7 = (a2 : ℚ) / factorial 2 + (a3 : ℚ) / factorial 3 + (a4 : ℚ) / factorial 4 + 
                         (a5 : ℚ) / factorial 5 + (a6 : ℚ) / factorial 6 + (a7 : ℚ) / factorial 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 := 
sorry

end NUMINAMATH_GPT_sum_a_eq_9_l920_92072


namespace NUMINAMATH_GPT_remainder_4032_125_l920_92081

theorem remainder_4032_125 : 4032 % 125 = 32 := by
  sorry

end NUMINAMATH_GPT_remainder_4032_125_l920_92081


namespace NUMINAMATH_GPT_mean_of_five_numbers_l920_92077

theorem mean_of_five_numbers (a b c d e : ℚ) (h : a + b + c + d + e = 2/3) : 
  (a + b + c + d + e) / 5 = 2 / 15 := 
by 
  -- This is where the proof would go, but we'll omit it as per instructions
  sorry

end NUMINAMATH_GPT_mean_of_five_numbers_l920_92077


namespace NUMINAMATH_GPT_product_zero_when_a_is_three_l920_92007

theorem product_zero_when_a_is_three (a : ℤ) (h : a = 3) :
  (a - 9) * (a - 8) * (a - 7) * (a - 6) * (a - 5) * (a - 4) * (a - 3) * (a - 2) * (a - 1) * a = 0 :=
by
  cases h
  sorry

end NUMINAMATH_GPT_product_zero_when_a_is_three_l920_92007


namespace NUMINAMATH_GPT_carlson_handkerchief_usage_l920_92086

def problem_statement : Prop :=
  let handkerchief_area := 25 * 25 -- Area in cm²
  let total_fabric_area := 3 * 10000 -- Total fabric area in cm²
  let days := 8
  let total_handkerchiefs := total_fabric_area / handkerchief_area
  let handkerchiefs_per_day := total_handkerchiefs / days
  handkerchiefs_per_day = 6

theorem carlson_handkerchief_usage : problem_statement := by
  sorry

end NUMINAMATH_GPT_carlson_handkerchief_usage_l920_92086


namespace NUMINAMATH_GPT_divisible_by_13_l920_92040

theorem divisible_by_13 (a : ℤ) (h₀ : 0 ≤ a) (h₁ : a ≤ 13) : (51^2015 + a) % 13 = 0 → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_13_l920_92040


namespace NUMINAMATH_GPT_downstream_speed_l920_92012

variable (V_u V_s V_d : ℝ)

theorem downstream_speed (h1 : V_u = 22) (h2 : V_s = 32) (h3 : V_s = (V_u + V_d) / 2) : V_d = 42 :=
sorry

end NUMINAMATH_GPT_downstream_speed_l920_92012


namespace NUMINAMATH_GPT_cyclic_ABCD_l920_92014

variable {Point : Type}
variable {Angle LineCircle : Type → Type}
variable {cyclicQuadrilateral : List (Point) → Prop}
variable {convexQuadrilateral : List (Point) → Prop}
variable {lineSegment : Point → Point → LineCircle Point}
variable {onSegment : Point → LineCircle Point → Prop}
variable {angle : Point → Point → Point → Angle Point}

theorem cyclic_ABCD (A B C D P Q E : Point)
  (h1 : convexQuadrilateral [A, B, C, D])
  (h2 : cyclicQuadrilateral [P, Q, D, A])
  (h3 : cyclicQuadrilateral [Q, P, B, C])
  (h4 : onSegment E (lineSegment P Q))
  (h5 : angle P A E = angle Q D E)
  (h6 : angle P B E = angle Q C E) :
  cyclicQuadrilateral [A, B, C, D] :=
  sorry

end NUMINAMATH_GPT_cyclic_ABCD_l920_92014


namespace NUMINAMATH_GPT_percentage_both_colors_l920_92063

theorem percentage_both_colors
  (total_flags : ℕ)
  (even_flags : total_flags % 2 = 0)
  (C : ℕ)
  (total_flags_eq : total_flags = 2 * C)
  (blue_percent : ℕ)
  (blue_percent_eq : blue_percent = 60)
  (red_percent : ℕ)
  (red_percent_eq : red_percent = 65) :
  ∃ both_colors_percent : ℕ, both_colors_percent = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_both_colors_l920_92063


namespace NUMINAMATH_GPT_tracy_initial_candies_l920_92078

noncomputable def initial_candies : Nat := 80

theorem tracy_initial_candies
  (x : Nat)
  (hx1 : ∃ y : Nat, (1 ≤ y ∧ y ≤ 6) ∧ x = (5 * (44 + y)) / 3)
  (hx2 : x % 20 = 0) : x = initial_candies := by
  sorry

end NUMINAMATH_GPT_tracy_initial_candies_l920_92078


namespace NUMINAMATH_GPT_prove_d_minus_r_eq_1_l920_92009

theorem prove_d_minus_r_eq_1 
  (d r : ℕ) 
  (h_d1 : d > 1)
  (h1 : 1122 % d = r)
  (h2 : 1540 % d = r)
  (h3 : 2455 % d = r) :
  d - r = 1 :=
by sorry

end NUMINAMATH_GPT_prove_d_minus_r_eq_1_l920_92009


namespace NUMINAMATH_GPT_sum_of_remainders_mod_l920_92080

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_remainders_mod_l920_92080


namespace NUMINAMATH_GPT_middle_group_frequency_l920_92079

theorem middle_group_frequency (capacity : ℕ) (n_rectangles : ℕ) (A_mid A_other : ℝ) 
  (h_capacity : capacity = 300)
  (h_rectangles : n_rectangles = 9)
  (h_areas : A_mid = 1 / 5 * A_other)
  (h_total_area : A_mid + A_other = 1) : 
  capacity * A_mid = 50 := by
  sorry

end NUMINAMATH_GPT_middle_group_frequency_l920_92079


namespace NUMINAMATH_GPT_stratified_sampling_students_l920_92035

theorem stratified_sampling_students :
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  f + s = 70 :=
by
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  sorry

end NUMINAMATH_GPT_stratified_sampling_students_l920_92035


namespace NUMINAMATH_GPT_charley_pencils_lost_l920_92024

theorem charley_pencils_lost :
  ∃ x : ℕ, (30 - x - (1/3 : ℝ) * (30 - x) = 16) ∧ x = 6 :=
by
  -- Since x must be an integer and the equations naturally produce whole numbers,
  -- we work within the context of natural numbers, then cast to real as needed.
  use 6
  -- Express the main condition in terms of x
  have h: (30 - 6 - (1/3 : ℝ) * (30 - 6) = 16) := by sorry
  exact ⟨h, rfl⟩

end NUMINAMATH_GPT_charley_pencils_lost_l920_92024


namespace NUMINAMATH_GPT_find_valid_pairs_l920_92020

theorem find_valid_pairs :
  ∃ (a b c : ℕ), 
    (a = 33 ∧ b = 22 ∧ c = 1111) ∨
    (a = 66 ∧ b = 88 ∧ c = 4444) ∨
    (a = 88 ∧ b = 33 ∧ c = 7777) ∧
    (11 ≤ a ∧ a ≤ 99) ∧ (11 ≤ b ∧ b ≤ 99) ∧ (1111 ≤ c ∧ c ≤ 9999) ∧
    (a % 11 = 0) ∧ (b % 11 = 0) ∧ (c % 1111 = 0) ∧
    (a * a + b = c) := sorry

end NUMINAMATH_GPT_find_valid_pairs_l920_92020


namespace NUMINAMATH_GPT_complement_intersection_M_N_l920_92015

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x > -1}
def U : Set ℝ := Set.univ

theorem complement_intersection_M_N :
  U \ (M ∩ N) = {x | x ≤ -1} ∪ {x | x ≥ 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_M_N_l920_92015


namespace NUMINAMATH_GPT_faster_train_length_is_150_l920_92066

def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def time_seconds : ℝ := 15

noncomputable def length_faster_train : ℝ :=
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  relative_speed_mps * time_seconds

theorem faster_train_length_is_150 :
  length_faster_train = 150 := by
sorry

end NUMINAMATH_GPT_faster_train_length_is_150_l920_92066


namespace NUMINAMATH_GPT_machine_production_time_difference_undetermined_l920_92011

theorem machine_production_time_difference_undetermined :
  ∀ (machineP_machineQ_440_hours_diff : ℝ)
    (machineQ_production_rate : ℝ)
    (machineA_production_rate : ℝ),
    machineA_production_rate = 4.000000000000005 →
    machineQ_production_rate = machineA_production_rate * 1.1 →
    machineP_machineQ_440_hours_diff > 0 →
    machineQ_production_rate * machineP_machineQ_440_hours_diff = 440 →
    ∃ machineP_production_rate, 
    ¬(∃ hours_diff : ℝ, hours_diff = 440 / machineP_production_rate - 440 / machineQ_production_rate) := sorry

end NUMINAMATH_GPT_machine_production_time_difference_undetermined_l920_92011


namespace NUMINAMATH_GPT_position_of_21_over_19_in_sequence_l920_92032

def sequence_term (n : ℕ) : ℚ := (n + 3) / (n + 1)

theorem position_of_21_over_19_in_sequence :
  ∃ n : ℕ, sequence_term n = 21 / 19 ∧ n = 18 :=
by sorry

end NUMINAMATH_GPT_position_of_21_over_19_in_sequence_l920_92032


namespace NUMINAMATH_GPT_checkerboard_sums_l920_92069

-- Define the dimensions and the arrangement of the checkerboard
def n : ℕ := 10
def board (i j : ℕ) : ℕ := i * n + j + 1

-- Define corner positions
def top_left_corner : ℕ := board 0 0
def top_right_corner : ℕ := board 0 (n - 1)
def bottom_left_corner : ℕ := board (n - 1) 0
def bottom_right_corner : ℕ := board (n - 1) (n - 1)

-- Sum of the corners
def corner_sum : ℕ := top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner

-- Define the positions of the main diagonals
def main_diagonal (i : ℕ) : ℕ := board i i
def anti_diagonal (i : ℕ) : ℕ := board i (n - 1 - i)

-- Sum of the main diagonals
def diagonal_sum : ℕ := (Finset.range n).sum main_diagonal + (Finset.range n).sum anti_diagonal - (main_diagonal 0 + main_diagonal (n - 1))

-- Statement to prove
theorem checkerboard_sums : corner_sum = 202 ∧ diagonal_sum = 101 :=
by
-- Proof is not required as per the instructions
sorry

end NUMINAMATH_GPT_checkerboard_sums_l920_92069


namespace NUMINAMATH_GPT_special_blend_probability_l920_92050

/-- Define the probability variables and conditions -/
def visit_count : ℕ := 6
def special_blend_prob : ℚ := 3 / 4
def non_special_blend_prob : ℚ := 1 / 4

/-- The binomial coefficient for choosing 5 days out of 6 -/
def choose_6_5 : ℕ := Nat.choose 6 5

/-- The probability of serving the special blend exactly 5 times out of 6 -/
def prob_special_blend_5 : ℚ := (choose_6_5 : ℚ) * (special_blend_prob ^ 5) * (non_special_blend_prob ^ 1)

/-- Statement to prove the desired probability -/
theorem special_blend_probability :
  prob_special_blend_5 = 1458 / 4096 :=
by
  sorry

end NUMINAMATH_GPT_special_blend_probability_l920_92050


namespace NUMINAMATH_GPT_find_x_for_g_l920_92002

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5) / 3) : ℝ)^(1/3 : ℝ)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ↔ x = -65 / 12 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_g_l920_92002


namespace NUMINAMATH_GPT_coin_exchange_impossible_l920_92043

theorem coin_exchange_impossible :
  ∀ (n : ℕ), (n % 4 = 1) → (¬ (∃ k : ℤ, n + 4 * k = 26)) :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_coin_exchange_impossible_l920_92043


namespace NUMINAMATH_GPT_non_neg_sum_sq_inequality_l920_92044

theorem non_neg_sum_sq_inequality (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
sorry

end NUMINAMATH_GPT_non_neg_sum_sq_inequality_l920_92044


namespace NUMINAMATH_GPT_abs_monotonic_increasing_even_l920_92075

theorem abs_monotonic_increasing_even :
  (∀ x : ℝ, |x| = |(-x)|) ∧ (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → |x1| ≤ |x2|) :=
by
  sorry

end NUMINAMATH_GPT_abs_monotonic_increasing_even_l920_92075


namespace NUMINAMATH_GPT_dark_chocolate_bars_sold_l920_92000

theorem dark_chocolate_bars_sold (W D : ℕ) (h₁ : 4 * D = 3 * W) (h₂ : W = 20) : D = 15 :=
by
  sorry

end NUMINAMATH_GPT_dark_chocolate_bars_sold_l920_92000


namespace NUMINAMATH_GPT_rafael_earnings_l920_92082

theorem rafael_earnings 
  (hours_monday : ℕ) 
  (hours_tuesday : ℕ) 
  (hours_left : ℕ) 
  (rate_per_hour : ℕ) 
  (h_monday : hours_monday = 10) 
  (h_tuesday : hours_tuesday = 8) 
  (h_left : hours_left = 20) 
  (h_rate : rate_per_hour = 20) : 
  (hours_monday + hours_tuesday + hours_left) * rate_per_hour = 760 := 
by
  sorry

end NUMINAMATH_GPT_rafael_earnings_l920_92082


namespace NUMINAMATH_GPT_land_plot_side_length_l920_92085

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end NUMINAMATH_GPT_land_plot_side_length_l920_92085


namespace NUMINAMATH_GPT_minimum_button_presses_to_exit_l920_92016

def arms_after (r y : ℕ) : ℕ := 3 + r - 2 * y
def doors_after (y g : ℕ) : ℕ := 3 + y - 2 * g

theorem minimum_button_presses_to_exit :
  ∃ r y g : ℕ, arms_after r y = 0 ∧ doors_after y g = 0 ∧ r + y + g = 9 :=
sorry

end NUMINAMATH_GPT_minimum_button_presses_to_exit_l920_92016


namespace NUMINAMATH_GPT_rectangular_container_volume_l920_92096

theorem rectangular_container_volume (a b c : ℝ) 
  (h1 : a * b = 30) 
  (h2 : b * c = 20) 
  (h3 : c * a = 12) : 
  a * b * c = 60 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_container_volume_l920_92096


namespace NUMINAMATH_GPT_geometric_sequence_a3_l920_92070

theorem geometric_sequence_a3 (
  a : ℕ → ℝ
) 
(h1 : a 1 = 1)
(h5 : a 5 = 16)
(h_geometric : ∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) :
a 3 = 4 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a3_l920_92070


namespace NUMINAMATH_GPT_find_three_numbers_l920_92074

theorem find_three_numbers (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : a + b - c = 10) 
  (h3 : a - b + c = 8) : 
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_three_numbers_l920_92074


namespace NUMINAMATH_GPT_rectangle_area_l920_92056

-- Define the vertices of the rectangle
def V1 : ℝ × ℝ := (-7, 1)
def V2 : ℝ × ℝ := (1, 1)
def V3 : ℝ × ℝ := (1, -6)
def V4 : ℝ × ℝ := (-7, -6)

-- Define the function to compute the area of the rectangle given the vertices
noncomputable def area_of_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  let length := abs (B.1 - A.1)
  let width := abs (A.2 - D.2)
  length * width

-- The statement to prove
theorem rectangle_area : area_of_rectangle V1 V2 V3 V4 = 56 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l920_92056


namespace NUMINAMATH_GPT_order_of_abc_l920_92004

noncomputable def a : ℚ := 1 / 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 6 - 2

theorem order_of_abc : a > c ∧ c > b := by
  sorry

end NUMINAMATH_GPT_order_of_abc_l920_92004


namespace NUMINAMATH_GPT_interval_contains_solution_l920_92041

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 2

theorem interval_contains_solution :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_interval_contains_solution_l920_92041


namespace NUMINAMATH_GPT_find_a_l920_92046

-- Define the necessary variables
variables (a b : ℝ) (t : ℝ)

-- Given conditions
def b_val : ℝ := 2120
def t_val : ℝ := 0.5

-- The statement we need to prove
theorem find_a (h: b = b_val) (h2: t = t_val) (h3: t = a / b) : a = 1060 := by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_find_a_l920_92046


namespace NUMINAMATH_GPT_blueberries_cartons_proof_l920_92036

def total_needed_cartons : ℕ := 26
def strawberries_cartons : ℕ := 10
def cartons_to_buy : ℕ := 7

theorem blueberries_cartons_proof :
  strawberries_cartons + cartons_to_buy + 9 = total_needed_cartons :=
by
  sorry

end NUMINAMATH_GPT_blueberries_cartons_proof_l920_92036


namespace NUMINAMATH_GPT_total_points_sum_l920_92097

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls := [6, 2, 5, 3, 4]
def carlos_rolls := [3, 2, 2, 6, 1]

def score (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem total_points_sum :
  score allie_rolls + score carlos_rolls = 44 :=
by
  sorry

end NUMINAMATH_GPT_total_points_sum_l920_92097


namespace NUMINAMATH_GPT_gcd_2023_1991_l920_92019

theorem gcd_2023_1991 : Nat.gcd 2023 1991 = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_2023_1991_l920_92019


namespace NUMINAMATH_GPT_tim_age_difference_l920_92092

theorem tim_age_difference (j_turned_23_j_turned_35 : ∃ (j_age_when_james_23 : ℕ) (john_age_when_james_23 : ℕ), 
                                          j_age_when_james_23 = 23 ∧ john_age_when_james_23 = 35)
                           (tim_age : ℕ) (tim_age_eq : tim_age = 79)
                           (tim_age_twice_john_age_less_X : ∃ (X : ℕ) (john_age : ℕ), tim_age = 2 * john_age - X) :
  ∃ (X : ℕ), X = 15 :=
by
  sorry

end NUMINAMATH_GPT_tim_age_difference_l920_92092


namespace NUMINAMATH_GPT_joan_seashells_correct_l920_92027

/-- Joan originally found 70 seashells -/
def joan_original_seashells : ℕ := 70

/-- Sam gave Joan 27 seashells -/
def seashells_given_by_sam : ℕ := 27

/-- The total number of seashells Joan has now -/
def joan_total_seashells : ℕ := joan_original_seashells + seashells_given_by_sam

theorem joan_seashells_correct : joan_total_seashells = 97 :=
by
  unfold joan_total_seashells
  unfold joan_original_seashells seashells_given_by_sam
  sorry

end NUMINAMATH_GPT_joan_seashells_correct_l920_92027


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l920_92057

def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0
def condition_q (x : ℝ) : Prop := |x - 2| < 1

theorem sufficient_but_not_necessary_condition : 
  (∀ x : ℝ, condition_p x → condition_q x) ∧ ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l920_92057


namespace NUMINAMATH_GPT_find_f_2011_l920_92073

open Function

variable {R : Type} [Field R]

def functional_equation (f : R → R) : Prop :=
  ∀ a b : R, f (a * f b) = a * b

theorem find_f_2011 (f : ℝ → ℝ) (h : functional_equation f) : f 2011 = 2011 :=
sorry

end NUMINAMATH_GPT_find_f_2011_l920_92073


namespace NUMINAMATH_GPT_average_weight_increase_l920_92084

variable {W : ℝ} -- Total weight before replacement
variable {n : ℝ} -- Number of men in the group

theorem average_weight_increase
  (h1 : (W - 58 + 83) / n - W / n = 2.5) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_weight_increase_l920_92084


namespace NUMINAMATH_GPT_opposite_of_negative_a_is_a_l920_92090

-- Define the problem:
theorem opposite_of_negative_a_is_a (a : ℝ) : -(-a) = a :=
by 
  sorry

end NUMINAMATH_GPT_opposite_of_negative_a_is_a_l920_92090


namespace NUMINAMATH_GPT_least_number_subtracted_l920_92021

-- Define the original number and the divisor
def original_number : ℕ := 427398
def divisor : ℕ := 14

-- Define the least number to be subtracted
def remainder := original_number % divisor
def least_number := remainder

-- The statement to be proven
theorem least_number_subtracted : least_number = 6 :=
by
  sorry

end NUMINAMATH_GPT_least_number_subtracted_l920_92021


namespace NUMINAMATH_GPT_nonnegative_integer_solutions_l920_92053

theorem nonnegative_integer_solutions (x : ℕ) :
  2 * x - 1 < 5 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
sorry

end NUMINAMATH_GPT_nonnegative_integer_solutions_l920_92053


namespace NUMINAMATH_GPT_companyA_sold_bottles_l920_92029

-- Let CompanyA and CompanyB be the prices per bottle for the respective companies
def CompanyA_price : ℝ := 4
def CompanyB_price : ℝ := 3.5

-- Company B sold 350 bottles
def CompanyB_bottles : ℕ := 350

-- Total revenue of Company B
def CompanyB_revenue : ℝ := CompanyB_price * CompanyB_bottles

-- Additional condition that the revenue difference is $25
def revenue_difference : ℝ := 25

-- Define the total revenue equations for both scenarios
def revenue_scenario1 (x : ℕ) : Prop :=
  CompanyA_price * x = CompanyB_revenue + revenue_difference

def revenue_scenario2 (x : ℕ) : Prop :=
  CompanyA_price * x + revenue_difference = CompanyB_revenue

-- The problem translates to finding x such that either of these conditions hold
theorem companyA_sold_bottles : ∃ x : ℕ, revenue_scenario2 x ∧ x = 300 :=
by
  sorry

end NUMINAMATH_GPT_companyA_sold_bottles_l920_92029


namespace NUMINAMATH_GPT_ratio_of_patients_l920_92083

def one_in_four_zx (current_patients : ℕ) : ℕ :=
  current_patients / 4

def previous_patients : ℕ :=
  26

def diagnosed_patients : ℕ :=
  13

def current_patients : ℕ :=
  diagnosed_patients * 4

theorem ratio_of_patients : 
  one_in_four_zx current_patients = diagnosed_patients → 
  (current_patients / previous_patients) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_patients_l920_92083


namespace NUMINAMATH_GPT_initial_ratio_of_stamps_l920_92071

theorem initial_ratio_of_stamps (P Q : ℕ) (h1 : ((P - 8 : ℤ) : ℚ) / (Q + 8) = 6 / 5) (h2 : P - 8 = Q + 8) : P / Q = 6 / 5 :=
sorry

end NUMINAMATH_GPT_initial_ratio_of_stamps_l920_92071


namespace NUMINAMATH_GPT_max_1x2_rectangles_in_3x3_grid_l920_92023

theorem max_1x2_rectangles_in_3x3_grid : 
  ∀ unit_squares rectangles_1x2 : ℕ, unit_squares + rectangles_1x2 = 9 → 
  (∃ max_rectangles : ℕ, max_rectangles = rectangles_1x2 ∧ max_rectangles = 5) :=
by
  sorry

end NUMINAMATH_GPT_max_1x2_rectangles_in_3x3_grid_l920_92023


namespace NUMINAMATH_GPT_shopkeeper_profit_percentage_l920_92028

theorem shopkeeper_profit_percentage
  (C : ℝ) -- The cost price of one article
  (cost_price_50 : ℝ := 50 * C) -- The cost price of 50 articles
  (cost_price_70 : ℝ := 70 * C) -- The cost price of 70 articles
  (selling_price_50 : ℝ := 70 * C) -- Selling price of 50 articles is the cost price of 70 articles
  :
  ∃ (P : ℝ), P = 40 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percentage_l920_92028


namespace NUMINAMATH_GPT_system_has_two_distinct_solutions_for_valid_a_l920_92005

noncomputable def log_eq (x y a : ℝ) : Prop := 
  Real.log (a * x + 4 * a) / Real.log (abs (x + 3)) = 
  2 * Real.log (x + y) / Real.log (abs (x + 3))

noncomputable def original_system (x y a : ℝ) : Prop :=
  log_eq x y a ∧ (x + 1 + Real.sqrt (x^2 + 2 * x + y - 4) = 0)

noncomputable def valid_range (a : ℝ) : Prop := 
  (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16 / 3)

theorem system_has_two_distinct_solutions_for_valid_a (a : ℝ) :
  valid_range a → 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ original_system x₁ 5 a ∧ original_system x₂ 5 a ∧ (-5 < x₁ ∧ x₁ ≤ -1) ∧ (-5 < x₂ ∧ x₂ ≤ -1) := 
sorry

end NUMINAMATH_GPT_system_has_two_distinct_solutions_for_valid_a_l920_92005


namespace NUMINAMATH_GPT_range_of_k_condition_l920_92087

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end NUMINAMATH_GPT_range_of_k_condition_l920_92087


namespace NUMINAMATH_GPT_rabbits_distribution_l920_92010

def num_ways_to_distribute : ℕ :=
  20 + 390 + 150

theorem rabbits_distribution :
  num_ways_to_distribute = 560 := by
  sorry

end NUMINAMATH_GPT_rabbits_distribution_l920_92010


namespace NUMINAMATH_GPT_frac_m_over_q_l920_92017

variable (m n p q : ℚ)

theorem frac_m_over_q (h1 : m / n = 10) (h2 : p / n = 2) (h3 : p / q = 1 / 5) : m / q = 1 :=
by
  sorry

end NUMINAMATH_GPT_frac_m_over_q_l920_92017


namespace NUMINAMATH_GPT_antonov_packs_remaining_l920_92099

theorem antonov_packs_remaining (total_candies : ℕ) (pack_size : ℕ) (packs_given : ℕ) (candies_remaining : ℕ) (packs_remaining : ℕ) :
  total_candies = 60 →
  pack_size = 20 →
  packs_given = 1 →
  candies_remaining = total_candies - pack_size * packs_given →
  packs_remaining = candies_remaining / pack_size →
  packs_remaining = 2 := by
  sorry

end NUMINAMATH_GPT_antonov_packs_remaining_l920_92099


namespace NUMINAMATH_GPT_geom_progression_vertex_ad_l920_92051

theorem geom_progression_vertex_ad
  (a b c d : ℝ)
  (geom_prog : a * c = b * b ∧ b * d = c * c)
  (vertex : (b, c) = (1, 3)) :
  a * d = 3 :=
sorry

end NUMINAMATH_GPT_geom_progression_vertex_ad_l920_92051


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l920_92033

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (r : ℝ)
    (h₁ : a₁ = 5^(3/4))
    (h₂ : a₂ = 5^(1/2))
    (h₃ : a₃ = 5^(1/4))
    (geometric_seq : a₂ = a₁ * r ∧ a₃ = a₂ * r) :
    a₃ * r = 1 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l920_92033


namespace NUMINAMATH_GPT_intersection_A_B_l920_92059

def setA : Set ℝ := {x | x^2 - 1 < 0}
def setB : Set ℝ := {x | x > 0}

theorem intersection_A_B : setA ∩ setB = {x | 0 < x ∧ x < 1} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l920_92059


namespace NUMINAMATH_GPT_range_of_S_l920_92058

variable {a b x : ℝ}
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem range_of_S (h1 : ∀ x ∈ Set.Icc 0 1, |f x a b| ≤ 1) :
  ∃ l u, -2 ≤ l ∧ u ≤ 9 / 4 ∧ ∀ (S : ℝ), (S = (a + 1) * (b + 1)) → l ≤ S ∧ S ≤ u :=
by
  sorry

end NUMINAMATH_GPT_range_of_S_l920_92058


namespace NUMINAMATH_GPT_difference_Q_R_l920_92047

variable (P Q R : ℝ) (x : ℝ)

theorem difference_Q_R (h1 : 11 * x - 5 * x = 12100) : 19 * x - 11 * x = 16133.36 :=
by
  sorry

end NUMINAMATH_GPT_difference_Q_R_l920_92047


namespace NUMINAMATH_GPT_vector_perpendicular_sets_l920_92039

-- Define the problem in Lean
theorem vector_perpendicular_sets (x : ℝ) : 
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sin x + Real.cos x, Real.sin x - Real.cos x)
  a.1 * b.1 + a.2 * b.2 = 0 ↔ ∃ (k : ℤ), x = k * (π / 2) + (π / 8) :=
sorry

end NUMINAMATH_GPT_vector_perpendicular_sets_l920_92039


namespace NUMINAMATH_GPT_dennis_years_taught_l920_92034

theorem dennis_years_taught (A V D : ℕ) (h1 : V + A + D = 75) (h2 : V = A + 9) (h3 : V = D - 9) : D = 34 :=
sorry

end NUMINAMATH_GPT_dennis_years_taught_l920_92034


namespace NUMINAMATH_GPT_speed_of_second_half_l920_92048

theorem speed_of_second_half (t d s1 d1 d2 : ℝ) (h_t : t = 30) (h_d : d = 672) (h_s1 : s1 = 21)
  (h_d1 : d1 = d / 2) (h_d2 : d2 = d / 2) (h_t1 : d1 / s1 = 16) (h_t2 : t - d1 / s1 = 14) :
  d2 / 14 = 24 :=
by sorry

end NUMINAMATH_GPT_speed_of_second_half_l920_92048


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l920_92038

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
                                     (h2 : a 3 + a 11 = 40) :
  a 6 - a 7 + a 8 = 20 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l920_92038


namespace NUMINAMATH_GPT_parts_production_equation_l920_92037

theorem parts_production_equation (x : ℝ) : 
  let apr := 50
  let may := 50 * (1 + x)
  let jun := 50 * (1 + x) * (1 + x)
  (apr + may + jun = 182) :=
sorry

end NUMINAMATH_GPT_parts_production_equation_l920_92037


namespace NUMINAMATH_GPT_problem1_problem2_l920_92030

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x)
  else x - 4 / x

theorem problem1 (h : ∀ x : ℝ, f 1 x = 3 → x = 4) : ∃ x : ℝ, f 1 x = 3 ∧ x = 4 :=
sorry

theorem problem2 (h : ∀ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) →
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a ≤ -1 → 
  a = -11 / 6) : ∃ a : ℝ, a ≤ -1 ∧ (∃ x1 x2 x3 : ℝ, 
  (x1 < x2 ∧ x2 < x3 ∧ x2 - x1 = x3 - x2) ∧ 
  f a x1 = 3 ∧ f a x2 = 3 ∧ f a x3 = 3 ∧ a = -11 / 6) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l920_92030


namespace NUMINAMATH_GPT_log_travel_time_24_l920_92008

noncomputable def time_for_log_to_travel (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) : ℝ :=
  D / v

theorem log_travel_time_24 (D u v : ℝ) (h1 : D / (u + v) = 4) (h2 : D / (u - v) = 6) :
  time_for_log_to_travel D u v h1 h2 = 24 :=
sorry

end NUMINAMATH_GPT_log_travel_time_24_l920_92008


namespace NUMINAMATH_GPT_population_scientific_notation_l920_92068

theorem population_scientific_notation : 
  (1.41: ℝ) * (10 ^ 9) = 1.41 * 10 ^ 9 := 
by
  sorry

end NUMINAMATH_GPT_population_scientific_notation_l920_92068


namespace NUMINAMATH_GPT_integral_solution_unique_l920_92018

theorem integral_solution_unique (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_integral_solution_unique_l920_92018


namespace NUMINAMATH_GPT_fraction_difference_l920_92064

theorem fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := 
  sorry

end NUMINAMATH_GPT_fraction_difference_l920_92064


namespace NUMINAMATH_GPT_solve_for_y_l920_92091

theorem solve_for_y : 
  ∀ (y : ℚ), y = 45 / (8 - 3 / 7) → y = 315 / 53 :=
by
  intro y
  intro h
  -- proof steps would be placed here
  sorry

end NUMINAMATH_GPT_solve_for_y_l920_92091


namespace NUMINAMATH_GPT_regular_polygon_sides_l920_92076

theorem regular_polygon_sides (ex_angle : ℝ) (hne_zero : ex_angle ≠ 0)
  (sum_ext_angles : ∀ (n : ℕ), n > 2 → n * ex_angle = 360) :
  ∃ (n : ℕ), n * 15 = 360 ∧ n = 24 :=
by 
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l920_92076


namespace NUMINAMATH_GPT_find_number_l920_92060

theorem find_number (x : ℝ) : x = 7 ∧ x^2 + 95 = (x - 19)^2 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l920_92060


namespace NUMINAMATH_GPT_combined_average_yield_l920_92093

theorem combined_average_yield (yield_A : ℝ) (price_A : ℝ) (yield_B : ℝ) (price_B : ℝ) (yield_C : ℝ) (price_C : ℝ) :
  yield_A = 0.20 → price_A = 100 → yield_B = 0.12 → price_B = 200 → yield_C = 0.25 → price_C = 300 →
  (yield_A * price_A + yield_B * price_B + yield_C * price_C) / (price_A + price_B + price_C) = 0.1983 :=
by
  intros hYA hPA hYB hPB hYC hPC
  sorry

end NUMINAMATH_GPT_combined_average_yield_l920_92093


namespace NUMINAMATH_GPT_jon_and_mary_frosting_l920_92095

-- Jon frosts a cupcake every 40 seconds
def jon_frost_rate : ℚ := 1 / 40

-- Mary frosts a cupcake every 24 seconds
def mary_frost_rate : ℚ := 1 / 24

-- Combined frosting rate of Jon and Mary
def combined_frost_rate : ℚ := jon_frost_rate + mary_frost_rate

-- Total time in seconds for 12 minutes
def total_time_seconds : ℕ := 12 * 60

-- Calculate the total number of cupcakes frosted in 12 minutes
def total_cupcakes_frosted (time_seconds : ℕ) (rate : ℚ) : ℚ :=
  time_seconds * rate

theorem jon_and_mary_frosting : total_cupcakes_frosted total_time_seconds combined_frost_rate = 48 := by
  sorry

end NUMINAMATH_GPT_jon_and_mary_frosting_l920_92095


namespace NUMINAMATH_GPT_sum_of_abs_first_10_terms_l920_92052

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ := n^2 - 5 * n + 2

theorem sum_of_abs_first_10_terms : 
  let S := sum_of_first_n_terms 10
  let S3 := sum_of_first_n_terms 3
  (S - 2 * S3) = 60 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_abs_first_10_terms_l920_92052


namespace NUMINAMATH_GPT_opposite_sides_line_l920_92055

theorem opposite_sides_line (m : ℝ) :
  ( (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ) → (-7 < m ∧ m < 24) :=
by sorry

end NUMINAMATH_GPT_opposite_sides_line_l920_92055


namespace NUMINAMATH_GPT_difference_of_squares_example_l920_92042

theorem difference_of_squares_example : 625^2 - 375^2 = 250000 :=
by sorry

end NUMINAMATH_GPT_difference_of_squares_example_l920_92042


namespace NUMINAMATH_GPT_salad_cost_is_correct_l920_92003

-- Definitions of costs according to the given conditions
def muffin_cost : ℝ := 2
def coffee_cost : ℝ := 4
def soup_cost : ℝ := 3
def lemonade_cost : ℝ := 0.75

def breakfast_cost : ℝ := muffin_cost + coffee_cost
def lunch_cost : ℝ := breakfast_cost + 3

def salad_cost : ℝ := lunch_cost - (soup_cost + lemonade_cost)

-- Statement to prove
theorem salad_cost_is_correct : salad_cost = 5.25 :=
by
  sorry

end NUMINAMATH_GPT_salad_cost_is_correct_l920_92003


namespace NUMINAMATH_GPT_remainder_modulo_l920_92065

theorem remainder_modulo (y : ℕ) (hy : 5 * y ≡ 1 [MOD 17]) : (7 + y) % 17 = 14 :=
sorry

end NUMINAMATH_GPT_remainder_modulo_l920_92065


namespace NUMINAMATH_GPT_minimum_distance_l920_92098

noncomputable def point_on_curve (x : ℝ) : ℝ := -x^2 + 3 * Real.log x

noncomputable def point_on_line (x : ℝ) : ℝ := x + 2

theorem minimum_distance 
  (a b c d : ℝ) 
  (hP : b = point_on_curve a) 
  (hQ : d = point_on_line c) 
  : (a - c)^2 + (b - d)^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_distance_l920_92098


namespace NUMINAMATH_GPT_ball_box_distribution_l920_92013

theorem ball_box_distribution :
  ∃ (distinct_ways : ℕ), distinct_ways = 7 :=
by
  let num_balls := 5
  let num_boxes := 4
  sorry

end NUMINAMATH_GPT_ball_box_distribution_l920_92013


namespace NUMINAMATH_GPT_sufficient_condition_inequalities_l920_92062

theorem sufficient_condition_inequalities (x a : ℝ) :
  (¬ (a-4 < x ∧ x < a+4) → ¬ (1 < x ∧ x < 2)) ↔ -2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_inequalities_l920_92062


namespace NUMINAMATH_GPT_gcd_seven_factorial_ten_fact_div_5_fact_l920_92026

def factorial (n : ℕ) : ℕ := Nat.factorial n

-- Define 7!
def seven_factorial := factorial 7

-- Define 10! / 5!
def ten_fact_div_5_fact := factorial 10 / factorial 5

-- Prove that the GCD of 7! and (10! / 5!) is 2520
theorem gcd_seven_factorial_ten_fact_div_5_fact :
  Nat.gcd seven_factorial ten_fact_div_5_fact = 2520 := by
sorry

end NUMINAMATH_GPT_gcd_seven_factorial_ten_fact_div_5_fact_l920_92026


namespace NUMINAMATH_GPT_arcsin_sqrt_three_over_two_l920_92061

theorem arcsin_sqrt_three_over_two : 
  ∃ θ, θ = Real.arcsin (Real.sqrt 3 / 2) ∧ θ = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_sqrt_three_over_two_l920_92061


namespace NUMINAMATH_GPT_Toms_swimming_speed_is_2_l920_92006

theorem Toms_swimming_speed_is_2
  (S : ℝ)
  (h1 : 2 * S + 4 * S = 12) :
  S = 2 :=
by
  sorry

end NUMINAMATH_GPT_Toms_swimming_speed_is_2_l920_92006


namespace NUMINAMATH_GPT_squirrel_spring_acorns_l920_92067

/--
A squirrel had stashed 210 acorns to last him the three winter months. 
It divided the pile into thirds, one for each month, and then took some 
from each third, leaving 60 acorns for each winter month. The squirrel 
combined the ones it took to eat in the first cold month of spring. 
Prove that the number of acorns the squirrel has for the beginning of spring 
is 30.
-/
theorem squirrel_spring_acorns :
  ∀ (initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month : ℕ),
    initial_acorns = 210 →
    acorns_per_month = initial_acorns / 3 →
    remaining_acorns_per_month = 60 →
    acorns_taken_per_month = acorns_per_month - remaining_acorns_per_month →
    3 * acorns_taken_per_month = 30 :=
by
  intros initial_acorns acorns_per_month remaining_acorns_per_month acorns_taken_per_month
  sorry

end NUMINAMATH_GPT_squirrel_spring_acorns_l920_92067


namespace NUMINAMATH_GPT_probability_not_within_square_B_l920_92088

theorem probability_not_within_square_B {A B : Type} 
  (area_A : ℝ) (perimeter_B : ℝ) (area_B : ℝ) (not_covered : ℝ) 
  (h1 : area_A = 30) 
  (h2 : perimeter_B = 16) 
  (h3 : area_B = 16) 
  (h4 : not_covered = area_A - area_B) :
  (not_covered / area_A) = 7 / 15 := by sorry

end NUMINAMATH_GPT_probability_not_within_square_B_l920_92088
