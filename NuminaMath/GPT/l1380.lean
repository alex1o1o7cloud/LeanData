import Mathlib

namespace sum_of_three_numbers_l1380_138076

theorem sum_of_three_numbers : 3.15 + 0.014 + 0.458 = 3.622 :=
by sorry

end sum_of_three_numbers_l1380_138076


namespace Todd_time_correct_l1380_138083

theorem Todd_time_correct :
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  Todd_time = 88 :=
by
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  sorry

end Todd_time_correct_l1380_138083


namespace batsman_average_l1380_138008

theorem batsman_average (avg_20 : ℕ) (avg_10 : ℕ) (total_matches_20 : ℕ) (total_matches_10 : ℕ) :
  avg_20 = 40 → avg_10 = 20 → total_matches_20 = 20 → total_matches_10 = 10 →
  (800 + 200) / 30 = 33.33 :=
by
  sorry

end batsman_average_l1380_138008


namespace share_of_a_l1380_138078

variables (A B C : ℝ)

def conditions :=
  A = (2 / 3) * (B + C) ∧
  B = (2 / 3) * (A + C) ∧
  A + B + C = 700

theorem share_of_a (h : conditions A B C) : A = 280 :=
by { sorry }

end share_of_a_l1380_138078


namespace find_original_sales_tax_percentage_l1380_138011

noncomputable def original_sales_tax_percentage (x : ℝ) : Prop :=
∃ (x : ℝ),
  let reduced_tax := 10 / 3 / 100;
  let market_price := 9000;
  let difference := 14.999999999999986;
  (x / 100 * market_price - reduced_tax * market_price = difference) ∧ x = 0.5

theorem find_original_sales_tax_percentage : original_sales_tax_percentage 0.5 :=
sorry

end find_original_sales_tax_percentage_l1380_138011


namespace middle_school_soccer_league_l1380_138004

theorem middle_school_soccer_league (n : ℕ) (h : (n * (n - 1)) / 2 = 36) : n = 9 := 
  sorry

end middle_school_soccer_league_l1380_138004


namespace sequence_diff_n_l1380_138022

theorem sequence_diff_n {a : ℕ → ℕ} (h1 : a 1 = 1) 
(h2 : ∀ n : ℕ, a (n + 1) ≤ 2 * n) (n : ℕ) :
  ∃ p q : ℕ, a p - a q = n :=
sorry

end sequence_diff_n_l1380_138022


namespace minimum_possible_area_l1380_138034

theorem minimum_possible_area (l w l_min w_min : ℝ) (hl : l = 5) (hw : w = 7) 
  (hl_min : l_min = l - 0.5) (hw_min : w_min = w - 0.5) : 
  l_min * w_min = 29.25 :=
by
  sorry

end minimum_possible_area_l1380_138034


namespace juan_ran_80_miles_l1380_138077

def speed : Real := 10 -- miles per hour
def time : Real := 8   -- hours

theorem juan_ran_80_miles :
  speed * time = 80 := 
by
  sorry

end juan_ran_80_miles_l1380_138077


namespace arithmetic_mean_is_b_l1380_138075

variable (x a b : ℝ)
variable (hx : x ≠ 0)
variable (hb : b ≠ 0)

theorem arithmetic_mean_is_b : (1 / 2 : ℝ) * ((x * b + a) / x + (x * b - a) / x) = b :=
by
  sorry

end arithmetic_mean_is_b_l1380_138075


namespace stewart_farm_sheep_count_l1380_138067

theorem stewart_farm_sheep_count 
  (S H : ℕ) 
  (ratio : S * 7 = 4 * H)
  (food_per_horse : H * 230 = 12880) : 
  S = 32 := 
sorry

end stewart_farm_sheep_count_l1380_138067


namespace ribbons_purple_l1380_138090

theorem ribbons_purple (total_ribbons : ℕ) (yellow_ribbons purple_ribbons orange_ribbons black_ribbons : ℕ)
  (h1 : yellow_ribbons = total_ribbons / 4)
  (h2 : purple_ribbons = total_ribbons / 3)
  (h3 : orange_ribbons = total_ribbons / 6)
  (h4 : black_ribbons = 40)
  (h5 : yellow_ribbons + purple_ribbons + orange_ribbons + black_ribbons = total_ribbons) :
  purple_ribbons = 53 :=
by
  sorry

end ribbons_purple_l1380_138090


namespace part_a_part_b_l1380_138057

-- Define the system of equations
def system_of_equations (x y z p : ℝ) :=
  x^2 - 3 * y + p = z ∧ y^2 - 3 * z + p = x ∧ z^2 - 3 * x + p = y

-- Part (a) proof problem statement
theorem part_a (p : ℝ) (hp : p ≥ 4) :
  (p > 4 → ¬ ∃ (x y z : ℝ), system_of_equations x y z p) ∧
  (p = 4 → ∀ (x y z : ℝ), system_of_equations x y z 4 → x = 2 ∧ y = 2 ∧ z = 2) :=
by sorry

-- Part (b) proof problem statement
theorem part_b (p : ℝ) (hp : 1 < p ∧ p < 4) :
  ∀ (x y z : ℝ), system_of_equations x y z p → x = y ∧ y = z :=
by sorry

end part_a_part_b_l1380_138057


namespace cost_of_5_dozen_l1380_138080

noncomputable def price_per_dozen : ℝ :=
  24 / 3

noncomputable def cost_before_tax (num_dozen : ℝ) : ℝ :=
  num_dozen * price_per_dozen

noncomputable def cost_after_tax (num_dozen : ℝ) : ℝ :=
  (1 + 0.10) * cost_before_tax num_dozen

theorem cost_of_5_dozen :
  cost_after_tax 5 = 44 := 
sorry

end cost_of_5_dozen_l1380_138080


namespace arithmetic_sequence_sum_l1380_138054

-- Given {a_n} is an arithmetic sequence, and a_2 + a_3 + a_{10} + a_{11} = 40, prove a_6 + a_7 = 20
theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 40) :
  a 6 + a 7 = 20 :=
sorry

end arithmetic_sequence_sum_l1380_138054


namespace part1_values_correct_estimated_students_correct_l1380_138089

def students_data : List ℕ :=
  [30, 60, 70, 10, 30, 115, 70, 60, 75, 90, 15, 70, 40, 75, 105, 80, 60, 30, 70, 45]

def total_students := 200

def categorized_counts := (2, 5, 10, 3) -- (0 ≤ t < 30, 30 ≤ t < 60, 60 ≤ t < 90, 90 ≤ t < 120)

def mean := 60

def median := 65

def mode := 70

theorem part1_values_correct :
  let a := 5
  let b := 3
  let c := 65
  let d := 70
  categorized_counts = (2, a, 10, b) ∧ mean = 60 ∧ median = c ∧ mode = d := by {
  -- Proof will be provided here
  sorry
}

theorem estimated_students_correct :
  let at_least_avg := 130
  at_least_avg = (total_students * 13 / 20) := by {
  -- Proof will be provided here
  sorry
}

end part1_values_correct_estimated_students_correct_l1380_138089


namespace total_water_filled_jars_l1380_138020

theorem total_water_filled_jars (x : ℕ) (h : 4 * x + 2 * x + x = 14 * 4) : 3 * x = 24 :=
by
  sorry

end total_water_filled_jars_l1380_138020


namespace trig_signs_l1380_138051

-- The conditions formulated as hypotheses
theorem trig_signs (h1 : Real.pi / 2 < 2 ∧ 2 < 3 ∧ 3 < Real.pi ∧ Real.pi < 4 ∧ 4 < 3 * Real.pi / 2) : 
  Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 := 
sorry

end trig_signs_l1380_138051


namespace minimum_reciprocal_sum_l1380_138086

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) - 2

theorem minimum_reciprocal_sum (a m n : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1)
  (h₃ : f a (-1) = -1) (h₄ : m + n = 2) (h₅ : 0 < m) (h₆ : 0 < n) :
  (1 / m) + (1 / n) = 2 :=
by
  sorry

end minimum_reciprocal_sum_l1380_138086


namespace students_didnt_like_food_l1380_138081

theorem students_didnt_like_food (total_students : ℕ) (liked_food : ℕ) (didnt_like_food : ℕ) 
  (h1 : total_students = 814) (h2 : liked_food = 383) 
  : didnt_like_food = total_students - liked_food := 
by 
  rw [h1, h2]
  sorry

end students_didnt_like_food_l1380_138081


namespace mean_of_remaining_l1380_138037

variable (a b c : ℝ)
variable (mean_of_four : ℝ := 90)
variable (largest : ℝ := 105)

theorem mean_of_remaining (h1 : (a + b + c + largest) / 4 = mean_of_four) : (a + b + c) / 3 = 85 := by
  sorry

end mean_of_remaining_l1380_138037


namespace walking_speed_of_A_l1380_138021

-- Given conditions
def B_speed := 20 -- kmph
def start_delay := 10 -- hours
def distance_covered := 200 -- km

-- Prove A's walking speed
theorem walking_speed_of_A (v : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance_covered = v * time_A ∧ distance_covered = B_speed * time_B ∧ time_B = time_A - start_delay → v = 10 :=
by
  intro h
  sorry

end walking_speed_of_A_l1380_138021


namespace players_per_group_l1380_138040

theorem players_per_group (new_players : ℕ) (returning_players : ℕ) (groups : ℕ) 
  (h1 : new_players = 48) 
  (h2 : returning_players = 6) 
  (h3 : groups = 9) : 
  (new_players + returning_players) / groups = 6 :=
by
  sorry

end players_per_group_l1380_138040


namespace max_rows_l1380_138048

theorem max_rows (m : ℕ) : (∀ T : Matrix (Fin m) (Fin 8) (Fin 4), 
  ∀ i j : Fin m, ∀ k l : Fin 8, i ≠ j ∧ T i k = T j k ∧ T i l = T j l → k ≠ l) → m ≤ 28 :=
sorry

end max_rows_l1380_138048


namespace sandy_paints_area_l1380_138074

-- Definition of the dimensions
def wall_height : ℝ := 10
def wall_length : ℝ := 15
def window_height : ℝ := 3
def window_length : ℝ := 5
def door_height : ℝ := 1
def door_length : ℝ := 6.5

-- Areas computation
def wall_area : ℝ := wall_height * wall_length
def window_area : ℝ := window_height * window_length
def door_area : ℝ := door_height * door_length

-- Area to be painted
def area_not_painted : ℝ := window_area + door_area
def area_to_be_painted : ℝ := wall_area - area_not_painted

-- The theorem to prove
theorem sandy_paints_area : area_to_be_painted = 128.5 := by
  -- The proof is omitted
  sorry

end sandy_paints_area_l1380_138074


namespace final_price_calculation_l1380_138016

theorem final_price_calculation 
  (ticket_price : ℝ)
  (initial_discount : ℝ)
  (additional_discount : ℝ)
  (sales_tax : ℝ)
  (final_price : ℝ) 
  (h1 : ticket_price = 200) 
  (h2 : initial_discount = 0.25) 
  (h3 : additional_discount = 0.15) 
  (h4 : sales_tax = 0.07)
  (h5 : final_price = (ticket_price * (1 - initial_discount)) * (1 - additional_discount) * (1 + sales_tax)):
  final_price = 136.43 :=
by
  sorry

end final_price_calculation_l1380_138016


namespace birch_tree_taller_than_pine_tree_l1380_138014

theorem birch_tree_taller_than_pine_tree :
  let pine_tree_height := (49 : ℚ) / 4
  let birch_tree_height := (37 : ℚ) / 2
  birch_tree_height - pine_tree_height = 25 / 4 :=
by
  sorry

end birch_tree_taller_than_pine_tree_l1380_138014


namespace elderly_sample_correct_l1380_138070

-- Conditions
def young_employees : ℕ := 300
def middle_aged_employees : ℕ := 150
def elderly_employees : ℕ := 100
def total_employees : ℕ := young_employees + middle_aged_employees + elderly_employees
def sample_size : ℕ := 33
def elderly_sample (total : ℕ) (elderly : ℕ) (sample : ℕ) : ℕ := (sample * elderly) / total

-- Statement to prove
theorem elderly_sample_correct :
  elderly_sample total_employees elderly_employees sample_size = 6 := 
by
  sorry

end elderly_sample_correct_l1380_138070


namespace prob_draw_correct_l1380_138019

-- Given conditions
def prob_A_wins : ℝ := 0.40
def prob_A_not_lose : ℝ := 0.90

-- Definition to be proved
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem prob_draw_correct : prob_draw = 0.50 := by
  sorry

end prob_draw_correct_l1380_138019


namespace zero_not_in_range_of_g_l1380_138066

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈1 / (x + 3)⌉
  else ⌊1 / (x + 3)⌋

theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l1380_138066


namespace find_min_value_l1380_138063

theorem find_min_value (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y = 2) : 
  (∃ (c : ℝ), ∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → x + y = 2 → c = 1 / 2 ∧ (8 / ((x + 2) * (y + 4))) ≥ c) :=
  sorry

end find_min_value_l1380_138063


namespace usual_time_is_12_l1380_138069

variable (S T : ℕ)

theorem usual_time_is_12 (h1: S > 0) (h2: 5 * (T + 3) = 4 * T) : T = 12 := 
by 
  sorry

end usual_time_is_12_l1380_138069


namespace elements_author_is_euclid_l1380_138046

def author_of_elements := "Euclid"

theorem elements_author_is_euclid : author_of_elements = "Euclid" :=
by
  rfl -- Reflexivity of equality, since author_of_elements is defined to be "Euclid".

end elements_author_is_euclid_l1380_138046


namespace value_of_fraction_l1380_138038

-- Lean 4 statement
theorem value_of_fraction (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 8) : (x + y) / 3 = 5 / 3 :=
by
  sorry

end value_of_fraction_l1380_138038


namespace greatest_b_not_in_range_l1380_138059

theorem greatest_b_not_in_range (b : ℤ) : ∀ x : ℝ, ¬ (x^2 + (b : ℝ) * x + 20 = -9) ↔ b ≤ 10 :=
by
  sorry

end greatest_b_not_in_range_l1380_138059


namespace find_b_perpendicular_l1380_138001

theorem find_b_perpendicular (a b : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 1 = 0 ∧ 3 * x + b * y + 5 = 0 → 
  - (a / 2) * - (3 / b) = -1) → b = -3 := 
sorry

end find_b_perpendicular_l1380_138001


namespace find_y_l1380_138068

theorem find_y (y : ℕ) (h : 4 ^ 12 = 64 ^ y) : y = 4 :=
sorry

end find_y_l1380_138068


namespace wang_hao_height_is_158_l1380_138015

/-- Yao Ming's height in meters. -/
def yao_ming_height : ℝ := 2.29

/-- Wang Hao is 0.71 meters shorter than Yao Ming. -/
def height_difference : ℝ := 0.71

/-- Wang Hao's height in meters. -/
def wang_hao_height : ℝ := yao_ming_height - height_difference

theorem wang_hao_height_is_158 :
  wang_hao_height = 1.58 :=
by
  sorry

end wang_hao_height_is_158_l1380_138015


namespace original_population_divisor_l1380_138079

theorem original_population_divisor (a b c : ℕ) (ha : ∃ a, ∃ b, ∃ c, a^2 + 120 = b^2 ∧ b^2 + 80 = c^2) :
  7 ∣ a :=
by
  sorry

end original_population_divisor_l1380_138079


namespace sum_of_x_intersections_l1380_138088

theorem sum_of_x_intersections (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 + x) = f (3 - x))
  (m : ℕ) (xs : Fin m → ℝ) (ys : Fin m → ℝ)
  (h_intersection : ∀ i : Fin m, f (xs i) = |(xs i)^2 - 4 * (xs i) - 3|) :
  (Finset.univ.sum fun i => xs i) = 2 * m :=
by
  sorry

end sum_of_x_intersections_l1380_138088


namespace adam_first_half_correct_l1380_138039

-- Define the conditions
def second_half_correct := 2
def points_per_question := 8
def final_score := 80

-- Define the number of questions Adam answered correctly in the first half
def first_half_correct :=
  (final_score - (second_half_correct * points_per_question)) / points_per_question

-- Statement to prove
theorem adam_first_half_correct : first_half_correct = 8 :=
by
  -- skipping the proof
  sorry

end adam_first_half_correct_l1380_138039


namespace terms_of_sequence_are_equal_l1380_138049

theorem terms_of_sequence_are_equal
    (n : ℤ)
    (h_n : n ≥ 2018)
    (a b : ℕ → ℕ)
    (h_a_distinct : ∀ i j, i ≠ j → a i ≠ a j)
    (h_b_distinct : ∀ i j, i ≠ j → b i ≠ b j)
    (h_a_bounds : ∀ i, a i ≤ 5 * n)
    (h_b_bounds : ∀ i, b i ≤ 5 * n)
    (h_arith_seq : ∀ i, (a (i + 1) * b i - a i * b (i + 1)) = (a 1 * b 0 - a 0 * b 1) * i) :
    ∀ i j, (a i * b j = a j * b i) := 
by 
  sorry

end terms_of_sequence_are_equal_l1380_138049


namespace notebook_cost_l1380_138042

theorem notebook_cost (s n c : ℕ) (h1 : s > 17) (h2 : n > 2 ∧ n % 2 = 0) (h3 : c > n) (h4 : s * c * n = 2013) : c = 61 :=
sorry

end notebook_cost_l1380_138042


namespace sum_of_arithmetic_sequence_9_terms_l1380_138096

-- Define the odd function and its properties
variables {f : ℝ → ℝ} (h1 : ∀ x, f (-x) = -f (x)) 
          (h2 : ∀ x y, x < y → f x < f y)

-- Define the shifted function g
noncomputable def g (x : ℝ) := f (x - 5)

-- Define the arithmetic sequence with non-zero common difference
variables {a : ℕ → ℝ} (d : ℝ) (h3 : d ≠ 0) 
          (h4 : ∀ n, a (n + 1) = a n + d)

-- Condition given by the problem
variable (h5 : g (a 1) + g (a 9) = 0)

-- Proof obligation
theorem sum_of_arithmetic_sequence_9_terms :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 45 :=
sorry

end sum_of_arithmetic_sequence_9_terms_l1380_138096


namespace solution_set_l1380_138099

theorem solution_set :
  {p : ℝ × ℝ | (p.1^2 + 3 * p.1 * p.2 + 2 * p.2^2) * (p.1^2 * p.2^2 - 1) = 0} =
  {p : ℝ × ℝ | p.2 = -p.1 / 2} ∪
  {p : ℝ × ℝ | p.2 = -p.1} ∪
  {p : ℝ × ℝ | p.2 = -1 / p.1} ∪
  {p : ℝ × ℝ | p.2 = 1 / p.1} :=
by sorry

end solution_set_l1380_138099


namespace crayons_received_l1380_138036

theorem crayons_received (crayons_left : ℕ) (crayons_lost_given_away : ℕ) (lost_twice_given : ∃ (G L : ℕ), L = 2 * G ∧ L + G = crayons_lost_given_away) :
  crayons_left = 2560 →
  crayons_lost_given_away = 9750 →
  ∃ (total_crayons_received : ℕ), total_crayons_received = 12310 :=
by
  intros h1 h2
  obtain ⟨G, L, hL, h_sum⟩ := lost_twice_given
  sorry -- Proof goes here

end crayons_received_l1380_138036


namespace slices_left_for_phill_correct_l1380_138058

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end slices_left_for_phill_correct_l1380_138058


namespace similar_triangle_leg_l1380_138082

theorem similar_triangle_leg (x : Real) : 
  (12 / x = 9 / 7) → x = 84 / 9 := by
  intro h
  sorry

end similar_triangle_leg_l1380_138082


namespace line_parallel_not_coincident_l1380_138072

theorem line_parallel_not_coincident (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) →
  ¬ ∃ k : ℝ, (∀ x y : ℝ, a * x + 2 * y + 6 = k * (x + (a - 1) * y + (a^2 - 1))) →
  a = -1 :=
by
  sorry

end line_parallel_not_coincident_l1380_138072


namespace total_students_correct_l1380_138028

def third_grade_students := 203
def fourth_grade_students := third_grade_students + 125
def total_students := third_grade_students + fourth_grade_students

theorem total_students_correct :
  total_students = 531 :=
by
  -- We state that the total number of students is 531
  sorry

end total_students_correct_l1380_138028


namespace Cody_spent_25_tickets_on_beanie_l1380_138055

-- Introducing the necessary definitions and assumptions
variable (x : ℕ)

-- Define the conditions translated from the problem statement
def initial_tickets := 49
def tickets_left (x : ℕ) := initial_tickets - x + 6

-- State the main problem as Theorem
theorem Cody_spent_25_tickets_on_beanie (H : tickets_left x = 30) : x = 25 := by
  sorry

end Cody_spent_25_tickets_on_beanie_l1380_138055


namespace find_g_3_l1380_138002

-- Definitions and conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g (x - 1) = 2 * x + 6)

-- Theorem: Proof problem corresponding to the problem
theorem find_g_3 : g 3 = 14 :=
by
  -- Insert proof here
  sorry

end find_g_3_l1380_138002


namespace quadratic_equation_unique_solution_l1380_138029

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  16 - 4 * a * c = 0 ∧ a + c = 5 ∧ a < c → (a, c) = (1, 4) :=
by
  sorry

end quadratic_equation_unique_solution_l1380_138029


namespace find_number_l1380_138026

theorem find_number (n : ℝ) (h : n / 0.04 = 400.90000000000003) : n = 16.036 := 
by
  sorry

end find_number_l1380_138026


namespace train_speed_l1380_138091

/-- Define the lengths of the train and the bridge and the time taken to cross the bridge. --/
def len_train : ℕ := 360
def len_bridge : ℕ := 240
def time_minutes : ℕ := 4
def time_seconds : ℕ := 240 -- 4 minutes converted to seconds

/-- Define the speed calculation based on the given domain. --/
def total_distance : ℕ := len_train + len_bridge
def speed (distance : ℕ) (time : ℕ) : ℚ := distance / time

/-- The statement to prove that the speed of the train is 2.5 m/s. --/
theorem train_speed :
  speed total_distance time_seconds = 2.5 := sorry

end train_speed_l1380_138091


namespace no_real_b_for_line_to_vertex_of_parabola_l1380_138095

theorem no_real_b_for_line_to_vertex_of_parabola : 
  ¬ ∃ b : ℝ, ∃ x : ℝ, y = x + b ∧ y = x^2 + b^2 + 1 :=
by
  sorry

end no_real_b_for_line_to_vertex_of_parabola_l1380_138095


namespace stratified_sampling_pines_l1380_138071

def total_saplings : ℕ := 30000
def pine_saplings : ℕ := 4000
def sample_size : ℕ := 150

theorem stratified_sampling_pines :
  sample_size * pine_saplings / total_saplings = 20 := by
  sorry

end stratified_sampling_pines_l1380_138071


namespace find_h_l1380_138094

theorem find_h (h : ℝ) :
  (∃ x : ℝ, 3 ≤ x ∧ x ≤ 7 ∧ -(x - h)^2 = -1) → (h = 2 ∨ h = 8) :=
by sorry

end find_h_l1380_138094


namespace find_missing_number_l1380_138065

theorem find_missing_number (x : ℝ)
  (h1 : (x + 42 + 78 + 104) / 4 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) :
  x = 74 :=
sorry

end find_missing_number_l1380_138065


namespace chris_money_before_birthday_l1380_138017

/-- Chris's total money now is $279 -/
def money_now : ℕ := 279

/-- Money received from Chris's grandmother is $25 -/
def money_grandmother : ℕ := 25

/-- Money received from Chris's aunt and uncle is $20 -/
def money_aunt_uncle : ℕ := 20

/-- Money received from Chris's parents is $75 -/
def money_parents : ℕ := 75

/-- Total money received for his birthday -/
def money_received : ℕ := money_grandmother + money_aunt_uncle + money_parents

/-- Money Chris had before his birthday -/
def money_before_birthday : ℕ := money_now - money_received

theorem chris_money_before_birthday : money_before_birthday = 159 := by
  sorry

end chris_money_before_birthday_l1380_138017


namespace smallest_and_largest_values_l1380_138087

theorem smallest_and_largest_values (x : ℕ) (h : x < 100) :
  (x ≡ 2 [MOD 3]) ∧ (x ≡ 2 [MOD 4]) ∧ (x ≡ 2 [MOD 5]) ↔ (x = 2 ∨ x = 62) :=
by
  sorry

end smallest_and_largest_values_l1380_138087


namespace solve_inequality_system_l1380_138084

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l1380_138084


namespace manager_monthly_salary_l1380_138010

theorem manager_monthly_salary (average_salary_20 : ℝ) (new_average_salary_21 : ℝ) (m : ℝ) 
  (h1 : average_salary_20 = 1300) 
  (h2 : new_average_salary_21 = 1400) 
  (h3 : 20 * average_salary_20 + m = 21 * new_average_salary_21) : 
  m = 3400 := 
by 
  -- Proof is omitted
  sorry

end manager_monthly_salary_l1380_138010


namespace paint_needed_for_720_statues_l1380_138000

noncomputable def paint_for_similar_statues (n : Nat) (h₁ h₂ : ℝ) (p₁ : ℝ) : ℝ :=
  let ratio := (h₂ / h₁) ^ 2
  n * (ratio * p₁)

theorem paint_needed_for_720_statues :
  paint_for_similar_statues 720 12 2 1 = 20 :=
by
  sorry

end paint_needed_for_720_statues_l1380_138000


namespace find_x_l1380_138032

theorem find_x (x : ℝ) (h : 0.75 * x = (1 / 3) * x + 110) : x = 264 :=
sorry

end find_x_l1380_138032


namespace find_a_and_b_l1380_138023

def star (a b : ℕ) : ℕ := a^b + a * b

theorem find_a_and_b (a b : ℕ) (h1 : 2 ≤ a) (h2 : 2 ≤ b) (h3 : star a b = 40) : a + b = 7 :=
by
  sorry

end find_a_and_b_l1380_138023


namespace expression_simplifies_to_36_l1380_138062

theorem expression_simplifies_to_36 (x : ℝ) : (x + 1)^2 + 2 * (x + 1) * (5 - x) + (5 - x)^2 = 36 :=
by
  sorry

end expression_simplifies_to_36_l1380_138062


namespace medal_awarding_ways_l1380_138045

def num_sprinters := 10
def num_americans := 4
def num_kenyans := 2
def medal_positions := 3 -- gold, silver, bronze

-- The main statement to be proven
theorem medal_awarding_ways :
  let ways_case1 := 2 * 3 * 5 * 4
  let ways_case2 := 4 * 3 * 2 * 2 * 5
  ways_case1 + ways_case2 = 360 :=
by
  sorry

end medal_awarding_ways_l1380_138045


namespace chord_bisect_angle_l1380_138043

theorem chord_bisect_angle (AB AC : ℝ) (angle_CAB : ℝ) (h1 : AB = 2) (h2 : AC = 1) (h3 : angle_CAB = 120) : 
  ∃ x : ℝ, x = 3 := 
by
  -- Proof goes here
  sorry

end chord_bisect_angle_l1380_138043


namespace falsity_of_proposition_implies_a_range_l1380_138007

theorem falsity_of_proposition_implies_a_range (a : ℝ) : 
  (¬ ∃ x₀ : ℝ, a * Real.sin x₀ + Real.cos x₀ ≥ 2) →
  a ∈ Set.Ioo (-Real.sqrt 3) (Real.sqrt 3) :=
by 
  sorry

end falsity_of_proposition_implies_a_range_l1380_138007


namespace problem_l1380_138098

theorem problem (a : ℝ) :
  (∀ x : ℝ, (x > 1 ↔ (x - 1 > 0 ∧ 2 * x - a > 0))) → a ≤ 2 :=
by
  sorry

end problem_l1380_138098


namespace sum_of_divisor_and_quotient_is_correct_l1380_138009

theorem sum_of_divisor_and_quotient_is_correct (divisor quotient : ℕ)
  (h1 : 1000 ≤ divisor ∧ divisor < 10000) -- Divisor is a four-digit number.
  (h2 : quotient * divisor + remainder = original_number) -- Division condition (could be more specific)
  (h3 : remainder < divisor) -- Remainder condition
  (h4 : original_number = 82502) -- Given original number
  : divisor + quotient = 723 := 
sorry

end sum_of_divisor_and_quotient_is_correct_l1380_138009


namespace simplify_expression_correct_l1380_138003

def simplify_expression (y : ℝ) : ℝ :=
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + y ^ 8)

theorem simplify_expression_correct (y : ℝ) :
  simplify_expression y = 15 * y ^ 13 - y ^ 12 + 6 * y ^ 11 + 5 * y ^ 10 - 7 * y ^ 9 - 2 * y ^ 8 :=
by
  sorry

end simplify_expression_correct_l1380_138003


namespace parallel_and_through_point_l1380_138085

-- Defining the given line
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0

-- Defining the target line passing through the point (0, 4)
def line2 (x y : ℝ) : Prop := 2 * x - y + 4 = 0

-- Define the point (0, 4)
def point : ℝ × ℝ := (0, 4)

-- Prove that line2 passes through the point (0, 4) and is parallel to line1
theorem parallel_and_through_point (x y : ℝ) 
  (h1 : line1 x y) 
  : line2 (point.fst) (point.snd) := by
  sorry

end parallel_and_through_point_l1380_138085


namespace rides_ratio_l1380_138024

theorem rides_ratio (total_money rides_spent dessert_spent money_left : ℕ) 
  (h1 : total_money = 30) 
  (h2 : dessert_spent = 5) 
  (h3 : money_left = 10) 
  (h4 : total_money - money_left = rides_spent + dessert_spent) : 
  (rides_spent : ℚ) / total_money = 1 / 2 := 
sorry

end rides_ratio_l1380_138024


namespace compute_expr1_factorize_expr2_l1380_138013

-- Definition for Condition 1: None explicitly stated.

-- Theorem for Question 1
theorem compute_expr1 (y : ℝ) : (y - 1) * (y + 5) = y^2 + 4*y - 5 :=
by sorry

-- Definition for Condition 2: None explicitly stated.

-- Theorem for Question 2
theorem factorize_expr2 (x y : ℝ) : -x^2 + 4*x*y - 4*y^2 = -((x - 2*y)^2) :=
by sorry

end compute_expr1_factorize_expr2_l1380_138013


namespace construct_quad_root_of_sums_l1380_138041

theorem construct_quad_root_of_sums (a b : ℝ) : ∃ c : ℝ, c = (a^4 + b^4)^(1/4) := 
by
  sorry

end construct_quad_root_of_sums_l1380_138041


namespace variance_of_scores_l1380_138064

open Real

def scores : List ℝ := [30, 26, 32, 27, 35]
noncomputable def average (s : List ℝ) : ℝ := s.sum / s.length
noncomputable def variance (s : List ℝ) : ℝ :=
  (s.map (λ x => (x - average s) ^ 2)).sum / s.length

theorem variance_of_scores :
  variance scores = 54 / 5 := 
by
  sorry

end variance_of_scores_l1380_138064


namespace total_pencils_l1380_138035

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 9) : pencils_per_child * children = 18 :=
sorry

end total_pencils_l1380_138035


namespace adjacent_angles_l1380_138006

theorem adjacent_angles (α β : ℝ) (h1 : α = β + 30) (h2 : α + β = 180) : α = 105 ∧ β = 75 := by
  sorry

end adjacent_angles_l1380_138006


namespace number_of_birdhouses_l1380_138056

-- Definitions for the conditions
def cost_per_nail : ℝ := 0.05
def cost_per_plank : ℝ := 3.0
def planks_per_birdhouse : ℕ := 7
def nails_per_birdhouse : ℕ := 20
def total_cost : ℝ := 88.0

-- Total cost calculation per birdhouse
def cost_per_birdhouse := planks_per_birdhouse * cost_per_plank + nails_per_birdhouse * cost_per_nail

-- Proving that the number of birdhouses is 4
theorem number_of_birdhouses : total_cost / cost_per_birdhouse = 4 := by
  sorry

end number_of_birdhouses_l1380_138056


namespace arctan_sum_l1380_138025

theorem arctan_sum (x y : ℝ) (hx : x = 3) (hy : y = 7) :
  Real.arctan (x / y) + Real.arctan (y / x) = Real.pi / 2 := 
by
  rw [hx, hy]
  sorry

end arctan_sum_l1380_138025


namespace total_distance_of_drive_l1380_138005

theorem total_distance_of_drive :
  let christina_speed := 30
  let christina_time_minutes := 180
  let christina_time_hours := christina_time_minutes / 60
  let friend_speed := 40
  let friend_time := 3
  let distance_christina := christina_speed * christina_time_hours
  let distance_friend := friend_speed * friend_time
  let total_distance := distance_christina + distance_friend
  total_distance = 210 :=
by
  sorry

end total_distance_of_drive_l1380_138005


namespace value_of_q_when_p_is_smallest_l1380_138027

-- Definitions of primality
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m > 1, m < n → ¬ (n % m = 0)

-- smallest prime number
def smallest_prime : ℕ := 2

-- Given conditions
def p : ℕ := 3
def q : ℕ := 2 + 13 * p

-- The theorem to prove
theorem value_of_q_when_p_is_smallest :
  is_prime smallest_prime →
  is_prime q →
  smallest_prime = 2 →
  p = 3 →
  q = 41 :=
by sorry

end value_of_q_when_p_is_smallest_l1380_138027


namespace expansion_abs_coeff_sum_l1380_138097

theorem expansion_abs_coeff_sum :
  ∀ (a a_1 a_2 a_3 a_4 a_5 : ℤ),
  (1 - x)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 32 :=
by
  sorry

end expansion_abs_coeff_sum_l1380_138097


namespace find_unknown_rate_l1380_138044

def cost_with_discount_and_tax (original_price : ℝ) (count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := (original_price * count) * (1 - discount)
  discounted_price * (1 + tax)

theorem find_unknown_rate :
  let total_blankets := 10
  let average_price := 160
  let total_cost := total_blankets * average_price
  let cost_100_blankets := cost_with_discount_and_tax 100 3 0.05 0.12
  let cost_150_blankets := cost_with_discount_and_tax 150 5 0.10 0.15
  let cost_unknown_blankets := 2 * x
  total_cost = cost_100_blankets + cost_150_blankets + cost_unknown_blankets →
  x = 252.275 :=
by
  sorry

end find_unknown_rate_l1380_138044


namespace total_paintable_wall_area_l1380_138050

/-- 
  Conditions:
  - John's house has 4 bedrooms.
  - Each bedroom is 15 feet long, 12 feet wide, and 10 feet high.
  - Doorways, windows, and a fireplace occupy 85 square feet per bedroom.
  Question: Prove that the total paintable wall area is 1820 square feet.
--/
theorem total_paintable_wall_area 
  (num_bedrooms : ℕ)
  (length width height non_paintable_area : ℕ)
  (h_num_bedrooms : num_bedrooms = 4)
  (h_length : length = 15)
  (h_width : width = 12)
  (h_height : height = 10)
  (h_non_paintable_area : non_paintable_area = 85) :
  (num_bedrooms * ((2 * (length * height) + 2 * (width * height)) - non_paintable_area) = 1820) :=
by
  sorry

end total_paintable_wall_area_l1380_138050


namespace problem_l1380_138093

theorem problem (a b : ℝ) (h1 : ∀ x : ℝ, 1 < x ∧ x < 2 → ax^2 - bx + 2 < 0) : a + b = 4 :=
sorry

end problem_l1380_138093


namespace red_trace_larger_sphere_area_l1380_138061

-- Defining the parameters and the given conditions
variables {R1 R2 : ℝ} (A1 : ℝ) (A2 : ℝ)
def smaller_sphere_radius := 4
def larger_sphere_radius := 6
def red_trace_smaller_sphere_area := 37

theorem red_trace_larger_sphere_area :
  R1 = smaller_sphere_radius → R2 = larger_sphere_radius → 
  A1 = red_trace_smaller_sphere_area → 
  A2 = A1 * (R2 / R1) ^ 2 → 
  A2 = 83.25 := 
  by
  intros hR1 hR2 hA1 hA2
  -- Use the given values and solve the assertion
  sorry

end red_trace_larger_sphere_area_l1380_138061


namespace class_distances_l1380_138047

theorem class_distances (x y z : ℕ) 
  (h1 : y = x + 8)
  (h2 : z = 3 * x)
  (h3 : x + y + z = 108) : 
  x = 20 ∧ y = 28 ∧ z = 60 := 
  by sorry

end class_distances_l1380_138047


namespace equal_cost_l1380_138033

theorem equal_cost (x : ℝ) : (2.75 * x + 125 = 1.50 * x + 140) ↔ (x = 12) := 
by sorry

end equal_cost_l1380_138033


namespace present_population_l1380_138053

-- Definitions
def initial_population : ℕ := 1200
def first_year_increase_rate : ℝ := 0.25
def second_year_increase_rate : ℝ := 0.30

-- Problem Statement
theorem present_population (initial_population : ℕ) 
    (first_year_increase_rate second_year_increase_rate : ℝ) : 
    initial_population = 1200 → 
    first_year_increase_rate = 0.25 → 
    second_year_increase_rate = 0.30 →
    ∃ current_population : ℕ, current_population = 1950 :=
by
  intros h₁ h₂ h₃
  sorry

end present_population_l1380_138053


namespace solve_for_A_l1380_138031

def diamond (A B : ℝ) := 4 * A + 3 * B + 7

theorem solve_for_A : diamond A 5 = 71 → A = 12.25 := by
  intro h
  unfold diamond at h
  sorry

end solve_for_A_l1380_138031


namespace man_was_absent_for_days_l1380_138060

theorem man_was_absent_for_days
  (x y : ℕ)
  (h1 : x + y = 30)
  (h2 : 10 * x - 2 * y = 216) :
  y = 7 :=
by
  sorry

end man_was_absent_for_days_l1380_138060


namespace average_cd_e_l1380_138073

theorem average_cd_e (c d e : ℝ) (h : (4 + 6 + 9 + c + d + e) / 6 = 20) : 
    (c + d + e) / 3 = 101 / 3 :=
by
  sorry

end average_cd_e_l1380_138073


namespace find_prime_p_l1380_138012

theorem find_prime_p (p : ℕ) (hp : Nat.Prime p) (hp_plus_10 : Nat.Prime (p + 10)) (hp_plus_14 : Nat.Prime (p + 14)) : p = 3 := 
sorry

end find_prime_p_l1380_138012


namespace even_n_if_fraction_is_integer_l1380_138030

theorem even_n_if_fraction_is_integer (n : ℕ) (h_pos : 0 < n) :
  (∃ a b : ℕ, 0 < b ∧ (a^2 + n^2) % (b^2 - n^2) = 0) → n % 2 = 0 := 
sorry

end even_n_if_fraction_is_integer_l1380_138030


namespace moles_of_C6H6_l1380_138092

def balanced_reaction (a b c d : ℕ) : Prop :=
  a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧ a + b + c + d = 4

theorem moles_of_C6H6 (a b c d : ℕ) (h_balanced : balanced_reaction a b c d) :
  a = 1 := 
by 
  sorry

end moles_of_C6H6_l1380_138092


namespace arithmetic_progression_l1380_138052

-- Define the general formula for the nth term of an arithmetic progression
def nth_term (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the conditions given in the problem
def condition1 (a1 d : ℤ) : Prop := nth_term a1 d 13 = 3 * nth_term a1 d 3
def condition2 (a1 d : ℤ) : Prop := nth_term a1 d 18 = 2 * nth_term a1 d 7 + 8

-- The main proof problem statement
theorem arithmetic_progression (a1 d : ℤ) (h1 : condition1 a1 d) (h2 : condition2 a1 d) : a1 = 12 ∧ d = 4 :=
by
  sorry

end arithmetic_progression_l1380_138052


namespace janet_initial_crayons_proof_l1380_138018

-- Define the initial number of crayons Michelle has
def michelle_initial_crayons : ℕ := 2

-- Define the final number of crayons Michelle will have after receiving Janet's crayons
def michelle_final_crayons : ℕ := 4

-- Define the function that calculates Janet's initial crayons
def janet_initial_crayons (m_i m_f : ℕ) : ℕ := m_f - m_i

-- The Lean statement to prove Janet's initial number of crayons
theorem janet_initial_crayons_proof : janet_initial_crayons michelle_initial_crayons michelle_final_crayons = 2 :=
by
  -- Proof steps go here (we use sorry to skip the proof)
  sorry

end janet_initial_crayons_proof_l1380_138018
