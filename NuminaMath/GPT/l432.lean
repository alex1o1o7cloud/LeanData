import Mathlib

namespace NUMINAMATH_GPT_find_root_and_m_l432_43210

theorem find_root_and_m {x : ℝ} {m : ℝ} (h : ∃ x1 x2 : ℝ, (x1 = 1) ∧ (x1 + x2 = -m) ∧ (x1 * x2 = 3)) :
  ∃ x2 : ℝ, (x2 = 3) ∧ (m = -4) :=
by
  obtain ⟨x1, x2, h1, h_sum, h_product⟩ := h
  have hx1 : x1 = 1 := h1
  rw [hx1] at h_product
  have hx2 : x2 = 3 := by linarith [h_product]
  have hm : m = -4 := by
    rw [hx1, hx2] at h_sum
    linarith
  exact ⟨x2, hx2, hm⟩

end NUMINAMATH_GPT_find_root_and_m_l432_43210


namespace NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_l432_43280

-- Define the number 32767
def number := 32767

-- Find the greatest prime divisor of 32767
def greatest_prime_divisor : ℕ :=
  127

-- Prove the sum of the digits of the greatest prime divisor is 10
theorem sum_of_digits_of_greatest_prime_divisor (h : greatest_prime_divisor = 127) : (1 + 2 + 7) = 10 :=
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_greatest_prime_divisor_l432_43280


namespace NUMINAMATH_GPT_identify_counterfeit_in_three_weighings_l432_43276

def CoinType := {x // x = "gold" ∨ x = "silver"}

structure Coins where
  golds: Fin 13
  silvers: Fin 14
  is_counterfeit: CoinType
  counterfeit_weight: Int

def is_lighter (c1 c2: Coins): Prop := sorry
def is_heavier (c1 c2: Coins): Prop := sorry
def balance (c1 c2: Coins): Prop := sorry

def find_counterfeit_coin (coins: Coins): Option Coins := sorry

theorem identify_counterfeit_in_three_weighings (coins: Coins) :
  ∃ (identify: Coins → Option Coins),
  ∀ coins, ( identify coins ≠ none ) :=
sorry

end NUMINAMATH_GPT_identify_counterfeit_in_three_weighings_l432_43276


namespace NUMINAMATH_GPT_tree_growth_l432_43271

theorem tree_growth (x : ℝ) : 4*x + 4*2*x + 4*2 + 4*3 = 32 → x = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tree_growth_l432_43271


namespace NUMINAMATH_GPT_quadratic_decreases_after_vertex_l432_43208

theorem quadratic_decreases_after_vertex :
  ∀ x : ℝ, (x > 2) → (y = -(x - 2)^2 + 3) → ∃ k : ℝ, k < 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_decreases_after_vertex_l432_43208


namespace NUMINAMATH_GPT_cos_neg_pi_over_3_l432_43253

theorem cos_neg_pi_over_3 : Real.cos (-π / 3) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_neg_pi_over_3_l432_43253


namespace NUMINAMATH_GPT_max_area_of_triangle_l432_43224

noncomputable def max_triangle_area (v1 v2 v3 : ℝ) (S : ℝ) : Prop :=
  2 * S + Real.sqrt 3 * (v1 * v2 + v3) = 0 ∧ v3 = Real.sqrt 3 → S ≤ Real.sqrt 3 / 4

theorem max_area_of_triangle (v1 v2 v3 S : ℝ) :
  max_triangle_area v1 v2 v3 S :=
by
  sorry

end NUMINAMATH_GPT_max_area_of_triangle_l432_43224


namespace NUMINAMATH_GPT_son_age_is_18_l432_43246

theorem son_age_is_18
  (S F : ℕ)
  (h1 : F = S + 20)
  (h2 : F + 2 = 2 * (S + 2)) :
  S = 18 :=
by sorry

end NUMINAMATH_GPT_son_age_is_18_l432_43246


namespace NUMINAMATH_GPT_maria_first_stop_distance_is_280_l432_43282

noncomputable def maria_travel_distance : ℝ := 560
noncomputable def first_stop_distance (x : ℝ) : ℝ := x
noncomputable def distance_after_first_stop (x : ℝ) : ℝ := maria_travel_distance - first_stop_distance x
noncomputable def second_stop_distance (x : ℝ) : ℝ := (1 / 4) * distance_after_first_stop x
noncomputable def remaining_distance : ℝ := 210

theorem maria_first_stop_distance_is_280 :
  ∃ x, first_stop_distance x = 280 ∧ second_stop_distance x + remaining_distance = distance_after_first_stop x := sorry

end NUMINAMATH_GPT_maria_first_stop_distance_is_280_l432_43282


namespace NUMINAMATH_GPT_steven_needs_more_seeds_l432_43201

theorem steven_needs_more_seeds :
  let total_seeds_needed := 60
  let seeds_per_apple := 6
  let seeds_per_pear := 2
  let seeds_per_grape := 3
  let apples_collected := 4
  let pears_collected := 3
  let grapes_collected := 9
  total_seeds_needed - (apples_collected * seeds_per_apple + pears_collected * seeds_per_pear + grapes_collected * seeds_per_grape) = 3 :=
by
  sorry

end NUMINAMATH_GPT_steven_needs_more_seeds_l432_43201


namespace NUMINAMATH_GPT_determine_a_values_l432_43204

theorem determine_a_values (a b c : ℤ) :
  (∀ x : ℤ, (x - a) * (x - 5) + 2 = (x + b) * (x + c)) ↔ a = 2 ∨ a = 8 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_values_l432_43204


namespace NUMINAMATH_GPT_correct_range_a_l432_43289

noncomputable def proposition_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
noncomputable def proposition_q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem correct_range_a (a : ℝ) :
  (¬ ∃ x, proposition_p a x → ¬ ∃ x, proposition_q x) →
  (a ≤ -4 ∨ a ≥ 2 ∨ a = 0) :=
sorry

end NUMINAMATH_GPT_correct_range_a_l432_43289


namespace NUMINAMATH_GPT_relationship_among_abc_l432_43213

noncomputable def a : ℝ := Real.sqrt 6 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 5 + Real.sqrt 8
def c : ℝ := 5

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l432_43213


namespace NUMINAMATH_GPT_lcm_is_multiple_of_230_l432_43212

theorem lcm_is_multiple_of_230 (d n : ℕ) (h1 : n = 230) (h2 : ¬ (3 ∣ n)) (h3 : ¬ (2 ∣ d)) : ∃ m : ℕ, Nat.lcm d n = 230 * m :=
by
  exists 1 -- Placeholder for demonstration purposes
  sorry

end NUMINAMATH_GPT_lcm_is_multiple_of_230_l432_43212


namespace NUMINAMATH_GPT_cristine_lemons_left_l432_43256

theorem cristine_lemons_left (initial_lemons : ℕ) (given_fraction : ℚ) (exchanged_lemons : ℕ) (h1 : initial_lemons = 12) (h2 : given_fraction = 1/4) (h3 : exchanged_lemons = 2) : 
  initial_lemons - initial_lemons * given_fraction - exchanged_lemons = 7 :=
by 
  sorry

end NUMINAMATH_GPT_cristine_lemons_left_l432_43256


namespace NUMINAMATH_GPT_intersection_M_N_l432_43226

open Set

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

theorem intersection_M_N :
  M ∩ N = {-1, 3} := 
sorry

end NUMINAMATH_GPT_intersection_M_N_l432_43226


namespace NUMINAMATH_GPT_Claire_plans_to_buy_five_cookies_l432_43294

theorem Claire_plans_to_buy_five_cookies :
  let initial_amount := 100
  let latte_cost := 3.75
  let croissant_cost := 3.50
  let days := 7
  let cookie_cost := 1.25
  let remaining_amount := 43
  let daily_expense := latte_cost + croissant_cost
  let weekly_expense := daily_expense * days
  let total_spent := initial_amount - remaining_amount
  let cookie_spent := total_spent - weekly_expense
  let cookies := cookie_spent / cookie_cost
  cookies = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_Claire_plans_to_buy_five_cookies_l432_43294


namespace NUMINAMATH_GPT_number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l432_43273

def Jungkook_cards : Real := 0.8
def Yoongi_cards : Real := 0.5

theorem number_of_people_with_cards_leq_0_point_3 : 
  (Jungkook_cards <= 0.3 ∨ Yoongi_cards <= 0.3) = False := 
by 
  -- neither Jungkook nor Yoongi has number cards less than or equal to 0.3
  sorry

theorem number_of_people_with_cards_leq_0_point_3_count :
  (if (Jungkook_cards <= 0.3) then 1 else 0) + (if (Yoongi_cards <= 0.3) then 1 else 0) = 0 :=
by 
  -- calculate number of people with cards less than or equal to 0.3
  sorry

end NUMINAMATH_GPT_number_of_people_with_cards_leq_0_point_3_number_of_people_with_cards_leq_0_point_3_count_l432_43273


namespace NUMINAMATH_GPT_fraction_equality_l432_43268

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_equality :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := 
by
  sorry

end NUMINAMATH_GPT_fraction_equality_l432_43268


namespace NUMINAMATH_GPT_Joan_pays_139_20_l432_43257

noncomputable def JKL : Type := ℝ × ℝ × ℝ

def conditions (J K L : ℝ) : Prop :=
  J + K + L = 600 ∧
  2 * J = K + 74 ∧
  L = K + 52

theorem Joan_pays_139_20 (J K L : ℝ) (h : conditions J K L) : J = 139.20 :=
by
  sorry

end NUMINAMATH_GPT_Joan_pays_139_20_l432_43257


namespace NUMINAMATH_GPT_total_study_time_is_60_l432_43211

-- Define the times Elizabeth studied for each test
def science_time : ℕ := 25
def math_time : ℕ := 35

-- Define the total study time
def total_study_time : ℕ := science_time + math_time

-- Proposition that the total study time equals 60 minutes
theorem total_study_time_is_60 : total_study_time = 60 := by
  /-
  Here we would provide the proof steps, but since the task is to write the statement only,
  we add 'sorry' to indicate the missing proof.
  -/
  sorry

end NUMINAMATH_GPT_total_study_time_is_60_l432_43211


namespace NUMINAMATH_GPT_problem_a_l432_43299

theorem problem_a : (1038^2 % 1000) ≠ 4 := by
  sorry

end NUMINAMATH_GPT_problem_a_l432_43299


namespace NUMINAMATH_GPT_arithmetic_sequence_S_15_l432_43237

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ}

theorem arithmetic_sequence_S_15 :
  (a 1 - a 4 - a 8 - a 12 + a 15 = 2) →
  (a 1 + a 15 = 2 * a 8) →
  (a 4 + a 12 = 2 * a 8) →
  S 15 a = -30 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S_15_l432_43237


namespace NUMINAMATH_GPT_Andy_more_white_socks_than_black_l432_43222

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end NUMINAMATH_GPT_Andy_more_white_socks_than_black_l432_43222


namespace NUMINAMATH_GPT_no_prime_divisible_by_77_l432_43255

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_divisible_by_77 : ∀ p : ℕ, is_prime p → ¬ (77 ∣ p) :=
by
  sorry

end NUMINAMATH_GPT_no_prime_divisible_by_77_l432_43255


namespace NUMINAMATH_GPT_patio_length_four_times_width_l432_43297

theorem patio_length_four_times_width (w l : ℕ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 :=
by
  sorry

end NUMINAMATH_GPT_patio_length_four_times_width_l432_43297


namespace NUMINAMATH_GPT_expected_value_l432_43292

theorem expected_value (p1 p2 p3 p4 p5 p6 : ℕ) (hp1 : p1 = 1) (hp2 : p2 = 5) (hp3 : p3 = 10) 
(hp4 : p4 = 25) (hp5 : p5 = 50) (hp6 : p6 = 100) :
  (p1 / 2 + p2 / 2 + p3 / 2 + p4 / 2 + p5 / 2 + p6 / 2 : ℝ) = 95.5 := by
  sorry

end NUMINAMATH_GPT_expected_value_l432_43292


namespace NUMINAMATH_GPT_max_distance_is_15_l432_43281

noncomputable def max_distance_between_cars (v_A v_B: ℝ) (a: ℝ) (D: ℝ) : ℝ :=
  if v_A > v_B ∧ D = a + 60 then (a * (1 - a / 60)) else 0

theorem max_distance_is_15 (v_A v_B: ℝ) (a: ℝ) (D: ℝ) :
  v_A > v_B ∧ D = a + 60 → max_distance_between_cars v_A v_B a D = 15 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_is_15_l432_43281


namespace NUMINAMATH_GPT_find_f_2021_l432_43216

variable (f : ℝ → ℝ)

axiom functional_equation : ∀ a b : ℝ, f ( (a + 2 * b) / 3) = (f a + 2 * f b) / 3
axiom f_one : f 1 = 1
axiom f_four : f 4 = 7

theorem find_f_2021 : f 2021 = 4041 := by
  sorry

end NUMINAMATH_GPT_find_f_2021_l432_43216


namespace NUMINAMATH_GPT_min_xy_l432_43236

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : xy = x + 4 * y + 5) : xy ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_xy_l432_43236


namespace NUMINAMATH_GPT_incorrect_simplification_l432_43242

theorem incorrect_simplification :
  (-(1 + 1/2) ≠ 1 + 1/2) := 
by sorry

end NUMINAMATH_GPT_incorrect_simplification_l432_43242


namespace NUMINAMATH_GPT_coeff_sum_zero_l432_43293

theorem coeff_sum_zero (a₀ a₁ a₂ a₃ a₄ : ℝ) (h : ∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) :
  a₁ + a₂ + a₃ + a₄ = 0 :=
by
  sorry

end NUMINAMATH_GPT_coeff_sum_zero_l432_43293


namespace NUMINAMATH_GPT_cars_cost_between_15000_and_20000_l432_43263

theorem cars_cost_between_15000_and_20000 (total_cars : ℕ) (p1 p2 : ℕ) :
    total_cars = 3000 → 
    p1 = 15 → 
    p2 = 40 → 
    (p1 * total_cars / 100 + p2 * total_cars / 100 + x = total_cars) → 
    x = 1350 :=
by
  intro h_total
  intro h_p1
  intro h_p2
  intro h_eq
  sorry

end NUMINAMATH_GPT_cars_cost_between_15000_and_20000_l432_43263


namespace NUMINAMATH_GPT_final_value_of_A_l432_43202

theorem final_value_of_A (A : ℝ) (h1: A = 15) (h2: A = -A + 5) : A = -10 :=
sorry

end NUMINAMATH_GPT_final_value_of_A_l432_43202


namespace NUMINAMATH_GPT_wage_recovery_l432_43244

theorem wage_recovery (W : ℝ) (h : W > 0) : (1 - 0.3) * W * (1 + 42.86 / 100) = W :=
by
  sorry

end NUMINAMATH_GPT_wage_recovery_l432_43244


namespace NUMINAMATH_GPT_line_quadrants_condition_l432_43248

theorem line_quadrants_condition (m n : ℝ) (h : m * n < 0) :
  (m > 0 ∧ n < 0) :=
sorry

end NUMINAMATH_GPT_line_quadrants_condition_l432_43248


namespace NUMINAMATH_GPT_cat_finishes_food_on_next_wednesday_l432_43279

def cat_food_consumption_per_day : ℚ :=
  (1 / 4) + (1 / 6)

def total_food_on_day (n : ℕ) : ℚ :=
  n * cat_food_consumption_per_day

def total_cans : ℚ := 8

theorem cat_finishes_food_on_next_wednesday :
  total_food_on_day 10 = total_cans := sorry

end NUMINAMATH_GPT_cat_finishes_food_on_next_wednesday_l432_43279


namespace NUMINAMATH_GPT_smallest_x_l432_43206

theorem smallest_x (x: ℕ) (hx: x > 0) (h: 11^2021 ∣ 5^(3*x) - 3^(4*x)) : 
  x = 11^2020 := sorry

end NUMINAMATH_GPT_smallest_x_l432_43206


namespace NUMINAMATH_GPT_decrease_percent_in_revenue_l432_43230

-- Definitions based on the conditions
def original_tax (T : ℝ) := T
def original_consumption (C : ℝ) := C
def new_tax (T : ℝ) := 0.70 * T
def new_consumption (C : ℝ) := 1.20 * C

-- Theorem statement for the decrease percent in revenue
theorem decrease_percent_in_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  100 * ((original_tax T * original_consumption C - new_tax T * new_consumption C) / (original_tax T * original_consumption C)) = 16 :=
by
  sorry

end NUMINAMATH_GPT_decrease_percent_in_revenue_l432_43230


namespace NUMINAMATH_GPT_calc_expression_l432_43245

theorem calc_expression :
  (3 * Real.sqrt 48 - 2 * Real.sqrt 12) / Real.sqrt 3 = 8 :=
sorry

end NUMINAMATH_GPT_calc_expression_l432_43245


namespace NUMINAMATH_GPT_min_operations_to_reach_goal_l432_43235

-- Define the initial and final configuration of the letters
structure Configuration where
  A : Char := 'A'
  B : Char := 'B'
  C : Char := 'C'
  D : Char := 'D'
  E : Char := 'E'
  F : Char := 'F'
  G : Char := 'G'

-- Define a valid rotation operation
inductive Rotation
| rotate_ABC : Rotation
| rotate_ABD : Rotation
| rotate_DEF : Rotation
| rotate_EFC : Rotation

-- Function representing a single rotation
def applyRotation : Configuration -> Rotation -> Configuration
| config, Rotation.rotate_ABC => 
  { A := config.C, B := config.A, C := config.B, D := config.D, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_ABD => 
  { A := config.B, B := config.D, D := config.A, C := config.C, E := config.E, F := config.F, G := config.G }
| config, Rotation.rotate_DEF => 
  { D := config.E, E := config.F, F := config.D, A := config.A, B := config.B, C := config.C, G := config.G }
| config, Rotation.rotate_EFC => 
  { E := config.F, F := config.C, C := config.E, A := config.A, B := config.B, D := config.D, G := config.G }

-- Define the goal configuration
def goalConfiguration : Configuration := 
  { A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G' }

-- Function to apply multiple rotations
def applyRotations (config : Configuration) (rotations : List Rotation) : Configuration :=
  rotations.foldl applyRotation config

-- Main theorem statement 
theorem min_operations_to_reach_goal : 
  ∃ rotations : List Rotation, rotations.length = 3 ∧ applyRotations {A := 'A', B := 'B', C := 'C', D := 'D', E := 'E', F := 'F', G := 'G'} rotations = goalConfiguration :=
sorry

end NUMINAMATH_GPT_min_operations_to_reach_goal_l432_43235


namespace NUMINAMATH_GPT_passing_percentage_is_correct_l432_43243

theorem passing_percentage_is_correct :
  ∀ (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ),
    marks_obtained = 59 →
    marks_failed_by = 40 →
    max_marks = 300 →
    (marks_obtained + marks_failed_by) / max_marks * 100 = 33 :=
by
  intros marks_obtained marks_failed_by max_marks h1 h2 h3
  sorry

end NUMINAMATH_GPT_passing_percentage_is_correct_l432_43243


namespace NUMINAMATH_GPT_rectangles_in_square_rectangles_in_three_squares_l432_43228

-- Given conditions as definitions
def positive_integer (n : ℕ) : Prop := n > 0

-- Part a
theorem rectangles_in_square (n : ℕ) (h : positive_integer n) :
  (n * (n + 1) / 2) ^ 2 = (n * (n + 1) / 2) ^ 2 :=
by sorry

-- Part b
theorem rectangles_in_three_squares (n : ℕ) (h : positive_integer n) :
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 = 
  n^2 * (2 * n + 1)^2 - n^4 - n^3 * (n + 1) - (n * (n + 1) / 2)^2 :=
by sorry

end NUMINAMATH_GPT_rectangles_in_square_rectangles_in_three_squares_l432_43228


namespace NUMINAMATH_GPT_function_decreasing_on_interval_l432_43209

noncomputable def g (x : ℝ) := -(1 / 3) * Real.sin (4 * x - Real.pi / 3)
noncomputable def f (x : ℝ) := -(1 / 3) * Real.sin (4 * x)

theorem function_decreasing_on_interval :
  ∀ x y : ℝ, (-Real.pi / 8 ≤ x ∧ x ≤ Real.pi / 8) → (-Real.pi / 8 ≤ y ∧ y ≤ Real.pi / 8) → x < y → f x > f y :=
sorry

end NUMINAMATH_GPT_function_decreasing_on_interval_l432_43209


namespace NUMINAMATH_GPT_find_speed_A_l432_43277

-- Defining the distance between the two stations as 155 km.
def distance := 155

-- Train A starts from station A at 7 a.m. and meets Train B at 11 a.m.
-- Therefore, Train A travels for 4 hours.
def time_A := 4

-- Train B starts from station B at 8 a.m. and meets Train A at 11 a.m.
-- Therefore, Train B travels for 3 hours.
def time_B := 3

-- Speed of Train B is given as 25 km/h.
def speed_B := 25

-- Condition that the total distance covered by both trains equals the distance between the two stations.
def meet_condition (v_A : ℕ) := (time_A * v_A) + (time_B * speed_B) = distance

-- The Lean theorem statement to be proved
theorem find_speed_A (v_A := 20) : meet_condition v_A :=
by
  -- Using 'sorrry' to skip the proof
  sorry

end NUMINAMATH_GPT_find_speed_A_l432_43277


namespace NUMINAMATH_GPT_geometric_series_sum_l432_43266

theorem geometric_series_sum :
  let a := (1 / 4 : ℚ)
  let r := (-1 / 4 : ℚ)
  let n := 6
  let sum := a * ((1 - r ^ n) / (1 - r))
  sum = (4095 / 5120 : ℚ) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l432_43266


namespace NUMINAMATH_GPT_max_a2_b2_c2_d2_l432_43250

-- Define the conditions for a, b, c, d
variables (a b c d : ℝ) 

-- Define the hypotheses from the problem
variables (h₁ : a + b = 17)
variables (h₂ : ab + c + d = 94)
variables (h₃ : ad + bc = 195)
variables (h₄ : cd = 120)

-- Define the final statement to be proved
theorem max_a2_b2_c2_d2 : ∃ (a b c d : ℝ), a + b = 17 ∧ ab + c + d = 94 ∧ ad + bc = 195 ∧ cd = 120 ∧ (a^2 + b^2 + c^2 + d^2) = 918 :=
by sorry

end NUMINAMATH_GPT_max_a2_b2_c2_d2_l432_43250


namespace NUMINAMATH_GPT_suitable_survey_is_D_l432_43278

-- Define the surveys
def survey_A := "Survey on the viewing of the movie 'The Long Way Home' by middle school students in our city"
def survey_B := "Survey on the germination rate of a batch of rose seeds"
def survey_C := "Survey on the water quality of the Jialing River"
def survey_D := "Survey on the health codes of students during the epidemic"

-- Define what it means for a survey to be suitable for a comprehensive census
def suitable_for_census (survey : String) : Prop :=
  survey = survey_D

-- Define the main theorem statement
theorem suitable_survey_is_D : suitable_for_census survey_D :=
by
  -- We assume sorry here to skip the proof
  sorry

end NUMINAMATH_GPT_suitable_survey_is_D_l432_43278


namespace NUMINAMATH_GPT_min_balls_to_ensure_20_l432_43286

theorem min_balls_to_ensure_20 (red green yellow blue purple white black : ℕ) (Hred : red = 30) (Hgreen : green = 25) (Hyellow : yellow = 18) (Hblue : blue = 15) (Hpurple : purple = 12) (Hwhite : white = 10) (Hblack : black = 7) :
  ∀ n, n ≥ 101 → (∃ r g y b p w bl, r + g + y + b + p + w + bl = n ∧ (r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ p ≥ 20 ∨ w ≥ 20 ∨ bl ≥ 20)) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_min_balls_to_ensure_20_l432_43286


namespace NUMINAMATH_GPT_range_of_a_l432_43285

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, 
    1 ≤ x ∧ x ≤ 2 ∧ 
    2 ≤ y ∧ y ≤ 3 → 
    x * y ≤ a * x^2 + 2 * y^2) ↔ 
  a ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l432_43285


namespace NUMINAMATH_GPT_sum_abs_arithmetic_sequence_l432_43249

variable (n : ℕ)

def S_n (n : ℕ) : ℚ :=
  - ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n

def T_n (n : ℕ) : ℚ :=
  if n ≤ 34 then
    -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
  else
    ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502

theorem sum_abs_arithmetic_sequence :
  T_n n = (if n ≤ 34 then -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
           else ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502) :=
by sorry

end NUMINAMATH_GPT_sum_abs_arithmetic_sequence_l432_43249


namespace NUMINAMATH_GPT_probability_no_adjacent_same_color_l432_43254

-- Define the problem space
def total_beads : ℕ := 9
def red_beads : ℕ := 4
def white_beads : ℕ := 3
def blue_beads : ℕ := 2

-- Define the total number of arrangements
def total_arrangements := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- State the probability computation theorem
theorem probability_no_adjacent_same_color :
  (∃ valid_arrangements : ℕ,
     valid_arrangements / total_arrangements = 1 / 63) := sorry

end NUMINAMATH_GPT_probability_no_adjacent_same_color_l432_43254


namespace NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l432_43231

theorem smallest_three_digit_multiple_of_17 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 17 = 0 → n ≤ m := sorry

end NUMINAMATH_GPT_smallest_three_digit_multiple_of_17_l432_43231


namespace NUMINAMATH_GPT_Jake_has_8_peaches_l432_43288

variables (Jake Steven Jill : ℕ)

-- The conditions
def condition1 : Steven = 15 := sorry
def condition2 : Steven = Jill + 14 := sorry
def condition3 : Jake = Steven - 7 := sorry

-- The proof statement
theorem Jake_has_8_peaches 
  (h1 : Steven = 15) 
  (h2 : Steven = Jill + 14) 
  (h3 : Jake = Steven - 7) : Jake = 8 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_Jake_has_8_peaches_l432_43288


namespace NUMINAMATH_GPT_three_pow_zero_l432_43283

theorem three_pow_zero : 3^0 = 1 :=
by sorry

end NUMINAMATH_GPT_three_pow_zero_l432_43283


namespace NUMINAMATH_GPT_jungkook_biggest_l432_43252

noncomputable def jungkook_number : ℕ := 6 * 3
def yoongi_number : ℕ := 4
def yuna_number : ℕ := 5

theorem jungkook_biggest :
  jungkook_number > yoongi_number ∧ jungkook_number > yuna_number :=
by
  unfold jungkook_number yoongi_number yuna_number
  sorry

end NUMINAMATH_GPT_jungkook_biggest_l432_43252


namespace NUMINAMATH_GPT_smallest_positive_value_of_expression_l432_43264

theorem smallest_positive_value_of_expression :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a^3 + b^3 + c^3 - 3 * a * b * c = 4) :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_value_of_expression_l432_43264


namespace NUMINAMATH_GPT_area_of_field_l432_43275

-- Define the given conditions and the problem
theorem area_of_field (L W A : ℝ) (hL : L = 20) (hFencing : 2 * W + L = 88) (hA : A = L * W) : 
  A = 680 :=
by
  sorry

end NUMINAMATH_GPT_area_of_field_l432_43275


namespace NUMINAMATH_GPT_value_preserving_interval_of_g_l432_43234

noncomputable def g (x : ℝ) (m : ℝ) : ℝ :=
  x + m - Real.log x

theorem value_preserving_interval_of_g
  (m : ℝ)
  (h_increasing : ∀ x, x ∈ Set.Ici 2 → 1 - 1 / x > 0)
  (h_range : ∀ y, y ∈ Set.Ici 2): 
  (2 + m - Real.log 2 = 2) → 
  m = Real.log 2 :=
by 
  sorry

end NUMINAMATH_GPT_value_preserving_interval_of_g_l432_43234


namespace NUMINAMATH_GPT_volunteer_assignment_correct_l432_43270

def volunteerAssignment : ℕ := 5
def pavilions : ℕ := 4

def numberOfWays (volunteers pavilions : ℕ) : ℕ := 72 -- This is based on the provided correct answer.

theorem volunteer_assignment_correct : 
  numberOfWays volunteerAssignment pavilions = 72 := 
by
  sorry

end NUMINAMATH_GPT_volunteer_assignment_correct_l432_43270


namespace NUMINAMATH_GPT_fourth_power_nested_sqrt_l432_43267

noncomputable def nested_sqrt : ℝ := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))

theorem fourth_power_nested_sqrt : nested_sqrt ^ 4 = 16 := by
  sorry

end NUMINAMATH_GPT_fourth_power_nested_sqrt_l432_43267


namespace NUMINAMATH_GPT_tea_blend_gain_percent_l432_43229

theorem tea_blend_gain_percent :
  let cost_18 := 18
  let cost_20 := 20
  let ratio_5_to_3 := (5, 3)
  let selling_price := 21
  let total_cost := (ratio_5_to_3.1 * cost_18) + (ratio_5_to_3.2 * cost_20)
  let total_weight := ratio_5_to_3.1 + ratio_5_to_3.2
  let cost_price_per_kg := total_cost / total_weight
  let gain_percent := ((selling_price - cost_price_per_kg) / cost_price_per_kg) * 100
  gain_percent = 12 :=
by
  sorry

end NUMINAMATH_GPT_tea_blend_gain_percent_l432_43229


namespace NUMINAMATH_GPT_george_correct_possible_change_sum_l432_43233

noncomputable def george_possible_change_sum : ℕ :=
if h : ∃ (change : ℕ), change < 100 ∧
  ((change % 25 == 7) ∨ (change % 25 == 32) ∨ (change % 25 == 57) ∨ (change % 25 == 82)) ∧
  ((change % 10 == 2) ∨ (change % 10 == 12) ∨ (change % 10 == 22) ∨
   (change % 10 == 32) ∨ (change % 10 == 42) ∨ (change % 10 == 52) ∨
   (change % 10 == 62) ∨ (change % 10 == 72) ∨ (change % 10 == 82) ∨ (change % 10 == 92)) ∧
  ((change % 5 == 9) ∨ (change % 5 == 14) ∨ (change % 5 == 19) ∨
   (change % 5 == 24) ∨ (change % 5 == 29) ∨ (change % 5 == 34) ∨
   (change % 5 == 39) ∨ (change % 5 == 44) ∨ (change % 5 == 49) ∨
   (change % 5 == 54) ∨ (change % 5 == 59) ∨ (change % 5 == 64) ∨
   (change % 5 == 69) ∨ (change % 5 == 74) ∨ (change % 5 == 79) ∨
   (change % 5 == 84) ∨ (change % 5 == 89) ∨ (change % 5 == 94) ∨ (change % 5 == 99)) then
  114
else 0

theorem george_correct_possible_change_sum :
  george_possible_change_sum = 114 :=
by
  sorry

end NUMINAMATH_GPT_george_correct_possible_change_sum_l432_43233


namespace NUMINAMATH_GPT_find_weeks_period_l432_43215

def weekly_addition : ℕ := 3
def bikes_sold : ℕ := 18
def bikes_in_stock : ℕ := 45
def initial_stock : ℕ := 51

theorem find_weeks_period (x : ℕ) :
  initial_stock + weekly_addition * x - bikes_sold = bikes_in_stock ↔ x = 4 := 
by 
  sorry

end NUMINAMATH_GPT_find_weeks_period_l432_43215


namespace NUMINAMATH_GPT_geometric_sequence_sum_l432_43205

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 0 = 3)
(h_sum : a 0 + a 1 + a 2 = 21) (hq : ∀ n, a (n + 1) = a n * q) : a 2 + a 3 + a 4 = 84 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l432_43205


namespace NUMINAMATH_GPT_minimum_F_l432_43274

noncomputable def F (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + (0.5 * x)

theorem minimum_F : ∃ x : ℝ, x ≥ 0 ∧ F x = 57.5 ∧ ∀ y ≥ 0, F y ≥ F x := by
  use 55
  sorry

end NUMINAMATH_GPT_minimum_F_l432_43274


namespace NUMINAMATH_GPT_evaluate_sum_of_powers_of_i_l432_43260

-- Definition of the imaginary unit i with property i^2 = -1.
def i : ℂ := Complex.I

lemma i_pow_2 : i^2 = -1 := by
  sorry

lemma i_pow_4n (n : ℤ) : i^(4 * n) = 1 := by
  sorry

-- Problem statement: Evaluate i^13 + i^18 + i^23 + i^28 + i^33 + i^38.
theorem evaluate_sum_of_powers_of_i : 
  i^13 + i^18 + i^23 + i^28 + i^33 + i^38 = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_sum_of_powers_of_i_l432_43260


namespace NUMINAMATH_GPT_age_problem_solution_l432_43296

theorem age_problem_solution 
  (x : ℕ) 
  (xiaoxiang_age : ℕ := 5) 
  (father_age : ℕ := 48) 
  (mother_age : ℕ := 42) 
  (h : (father_age + x) + (mother_age + x) = 6 * (xiaoxiang_age + x)) : 
  x = 15 :=
by {
  -- To be proved
  sorry
}

end NUMINAMATH_GPT_age_problem_solution_l432_43296


namespace NUMINAMATH_GPT_scientific_notation_of_2102000_l432_43269

theorem scientific_notation_of_2102000 : ∃ (x : ℝ) (n : ℤ), 2102000 = x * 10 ^ n ∧ x = 2.102 ∧ n = 6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_2102000_l432_43269


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l432_43220

/--
Given a hyperbola with the following properties:
1. Point \( P \) is on the left branch of the hyperbola \( C \): \(\frac{x^2}{a^2} - \frac{y^2}{b^2} = 1\), where \( a > 0 \) and \( b > 0 \).
2. \( F_2 \) is the right focus of the hyperbola.
3. One of the asymptotes of the hyperbola is perpendicular to the line segment \( PF_2 \).

Prove that the eccentricity \( e \) of the hyperbola is \( \sqrt{5} \).
-/
theorem hyperbola_eccentricity (a b e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (P_on_hyperbola : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (F2_is_focus : True) -- Placeholder for focus-related condition
  (asymptote_perpendicular : True) -- Placeholder for asymptote perpendicular condition
  : e = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l432_43220


namespace NUMINAMATH_GPT_probability_at_least_one_defective_is_correct_l432_43221

noncomputable def probability_at_least_one_defective : ℚ :=
  let total_bulbs := 23
  let defective_bulbs := 4
  let non_defective_bulbs := total_bulbs - defective_bulbs
  let probability_neither_defective :=
    (non_defective_bulbs / total_bulbs) * ((non_defective_bulbs - 1) / (total_bulbs - 1))
  1 - probability_neither_defective

theorem probability_at_least_one_defective_is_correct :
  probability_at_least_one_defective = 164 / 506 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_defective_is_correct_l432_43221


namespace NUMINAMATH_GPT_dart_not_land_in_circle_probability_l432_43261

theorem dart_not_land_in_circle_probability :
  let side_length := 1
  let radius := side_length / 2
  let area_square := side_length * side_length
  let area_circle := π * radius * radius
  let prob_inside_circle := area_circle / area_square
  let prob_outside_circle := 1 - prob_inside_circle
  prob_outside_circle = 1 - (π / 4) :=
by
  sorry

end NUMINAMATH_GPT_dart_not_land_in_circle_probability_l432_43261


namespace NUMINAMATH_GPT_probability_odd_80_heads_l432_43259

noncomputable def coin_toss_probability_odd (n : ℕ) (p : ℝ) : ℝ :=
  (1 / 2) * (1 - (1 / 3^n))

theorem probability_odd_80_heads :
  coin_toss_probability_odd 80 (3 / 4) = (1 / 2) * (1 - 1 / 3^80) :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_80_heads_l432_43259


namespace NUMINAMATH_GPT_factorial_division_l432_43258

-- Conditions: definition for factorial
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Statement of the problem: Proving the equality
theorem factorial_division :
  (factorial 10) / ((factorial 5) * (factorial 2)) = 15120 :=
by
  sorry

end NUMINAMATH_GPT_factorial_division_l432_43258


namespace NUMINAMATH_GPT_solution_set_of_inequality_l432_43295

theorem solution_set_of_inequality (x : ℝ) : ((x - 1) * (2 - x) ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l432_43295


namespace NUMINAMATH_GPT_five_integers_sum_to_first_set_impossible_second_set_sum_l432_43247

theorem five_integers_sum_to_first_set :
  ∃ (a b c d e : ℤ), 
    (a + b = 0) ∧ (a + c = 2) ∧ (b + c = 4) ∧ (a + d = 4) ∧ (b + d = 6) ∧
    (a + e = 8) ∧ (b + e = 9) ∧ (c + d = 11) ∧ (c + e = 13) ∧ (d + e = 15) ∧ 
    (a + b + c + d + e = 18) := 
sorry

theorem impossible_second_set_sum : 
  ¬∃ (a b c d e : ℤ), 
    (a + b = 12) ∧ (a + c = 13) ∧ (a + d = 14) ∧ (a + e = 15) ∧ (b + c = 16) ∧
    (b + d = 16) ∧ (b + e = 17) ∧ (c + d = 17) ∧ (c + e = 18) ∧ (d + e = 20) ∧
    (a + b + c + d + e = 39) :=
sorry

end NUMINAMATH_GPT_five_integers_sum_to_first_set_impossible_second_set_sum_l432_43247


namespace NUMINAMATH_GPT_functional_eq_solution_l432_43262

theorem functional_eq_solution (f : ℤ → ℤ) (h : ∀ x y : ℤ, x ≠ 0 →
  x * f (2 * f y - x) + y^2 * f (2 * x - f y) = (f x ^ 2) / x + f (y * f y)) :
  (∀ x: ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end NUMINAMATH_GPT_functional_eq_solution_l432_43262


namespace NUMINAMATH_GPT_red_pill_cost_l432_43272

theorem red_pill_cost :
  ∃ (r : ℚ) (b : ℚ), (∀ (d : ℕ), d = 21 → 3 * r - 2 = 39) ∧
                      (1 ≤ d → r = b + 1) ∧
                      (21 * (r + 2 * b) = 819) → 
                      r = 41 / 3 :=
by sorry

end NUMINAMATH_GPT_red_pill_cost_l432_43272


namespace NUMINAMATH_GPT_proof_problem_l432_43298

theorem proof_problem (f g g_inv : ℝ → ℝ) (hinv : ∀ x, f (x ^ 4 - 1) = g x)
  (hginv : ∀ y, g (g_inv y) = y) (h : ∀ y, f (g_inv y) = g (g_inv y)) :
  g_inv (f 15) = 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l432_43298


namespace NUMINAMATH_GPT_smallest_possible_QNNN_l432_43217

theorem smallest_possible_QNNN :
  ∃ (Q N : ℕ), (N = 1 ∨ N = 5 ∨ N = 6) ∧ (NN = 10 * N + N) ∧ (Q * 1000 + NN * 10 + N = NN * N) ∧ (Q * 1000 + NN * 10 + N) = 275 :=
sorry

end NUMINAMATH_GPT_smallest_possible_QNNN_l432_43217


namespace NUMINAMATH_GPT_find_AD_length_l432_43203

noncomputable def triangle_AD (A B C : Type) (AB AC : ℝ) (ratio_BD_CD : ℝ) (AD : ℝ) : Prop :=
  AB = 13 ∧ AC = 20 ∧ ratio_BD_CD = 3 / 4 → AD = 8 * Real.sqrt 2

theorem find_AD_length {A B C : Type} :
  triangle_AD A B C 13 20 (3/4) (8 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_find_AD_length_l432_43203


namespace NUMINAMATH_GPT_sum_of_fifth_powers_52070424_l432_43225

noncomputable def sum_of_fifth_powers (n : ℤ) : ℤ :=
  (n-1)^5 + n^5 + (n+1)^5

theorem sum_of_fifth_powers_52070424 :
  ∃ (n : ℤ), (n-1)^2 + n^2 + (n+1)^2 = 2450 ∧ sum_of_fifth_powers n = 52070424 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fifth_powers_52070424_l432_43225


namespace NUMINAMATH_GPT_area_gray_region_in_terms_of_pi_l432_43251

variable (r : ℝ)

theorem area_gray_region_in_terms_of_pi 
    (h1 : ∀ (r : ℝ), ∃ (outer_r : ℝ), outer_r = r + 3)
    (h2 : width_gray_region = 3)
    : ∃ (area_gray : ℝ), area_gray = π * (6 * r + 9) := 
sorry

end NUMINAMATH_GPT_area_gray_region_in_terms_of_pi_l432_43251


namespace NUMINAMATH_GPT_exam_full_marks_l432_43241

variables {A B C D F : ℝ}

theorem exam_full_marks
  (hA : A = 0.90 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.80 * D)
  (hA_val : A = 360)
  (hD : D = 0.80 * F) 
  : F = 500 :=
sorry

end NUMINAMATH_GPT_exam_full_marks_l432_43241


namespace NUMINAMATH_GPT_find_locus_of_M_l432_43291

variables {P : Type*} [MetricSpace P] 
variables (A B C M : P)

def on_perpendicular_bisector (A B M : P) : Prop := 
  dist A M = dist B M

def on_circle (center : P) (radius : ℝ) (M : P) : Prop := 
  dist center M = radius

def M_AB (A B M : P) : Prop :=
  (on_perpendicular_bisector A B M) ∨ (on_circle A (dist A B) M) ∨ (on_circle B (dist A B) M)

def M_BC (B C M : P) : Prop :=
  (on_perpendicular_bisector B C M) ∨ (on_circle B (dist B C) M) ∨ (on_circle C (dist B C) M)

theorem find_locus_of_M :
  {M : P | M_AB A B M} ∩ {M : P | M_BC B C M} = {M : P | M_AB A B M ∧ M_BC B C M} :=
by sorry

end NUMINAMATH_GPT_find_locus_of_M_l432_43291


namespace NUMINAMATH_GPT_cone_volume_l432_43219

theorem cone_volume (V_cylinder : ℝ) (V_cone : ℝ) (h : V_cylinder = 81 * Real.pi) :
  V_cone = 27 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l432_43219


namespace NUMINAMATH_GPT_find_n_l432_43227

theorem find_n (n : ℕ) (h : 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012) : n = 1005 :=
sorry

end NUMINAMATH_GPT_find_n_l432_43227


namespace NUMINAMATH_GPT_circumference_of_tank_a_l432_43238

def is_circumference_of_tank_a (h_A h_B C_B : ℝ) (V_A_eq : ℝ → Prop) : Prop :=
  ∃ (C_A : ℝ), 
    C_B = 10 ∧ 
    h_A = 10 ∧
    h_B = 7 ∧
    V_A_eq 0.7 ∧ 
    C_A = 7

theorem circumference_of_tank_a (h_A : ℝ) (h_B : ℝ) (C_B : ℝ) (V_A_eq : ℝ → Prop) : 
  is_circumference_of_tank_a h_A h_B C_B V_A_eq := 
by
  sorry

end NUMINAMATH_GPT_circumference_of_tank_a_l432_43238


namespace NUMINAMATH_GPT_day_after_exponential_days_l432_43232

noncomputable def days_since_monday (n : ℕ) : String :=
  let days := ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  days.get! (n % 7)

theorem day_after_exponential_days :
  days_since_monday (2^20) = "Friday" :=
by
  sorry

end NUMINAMATH_GPT_day_after_exponential_days_l432_43232


namespace NUMINAMATH_GPT_abs_diff_squares_105_95_l432_43200

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end NUMINAMATH_GPT_abs_diff_squares_105_95_l432_43200


namespace NUMINAMATH_GPT_stream_speed_l432_43265

variables (v : ℝ) (swimming_speed : ℝ) (ratio : ℝ)

theorem stream_speed (hs : swimming_speed = 4.5) (hr : ratio = 2) (h : (swimming_speed - v) / (swimming_speed + v) = 1 / ratio) :
  v = 1.5 :=
sorry

end NUMINAMATH_GPT_stream_speed_l432_43265


namespace NUMINAMATH_GPT_third_altitude_is_less_than_15_l432_43287

variable (a b c : ℝ)
variable (ha hb hc : ℝ)
variable (A : ℝ)

def triangle_area (side : ℝ) (height : ℝ) : ℝ := 0.5 * side * height

axiom ha_eq : ha = 10
axiom hb_eq : hb = 6

theorem third_altitude_is_less_than_15 : hc < 15 :=
sorry

end NUMINAMATH_GPT_third_altitude_is_less_than_15_l432_43287


namespace NUMINAMATH_GPT_solve_linear_system_l432_43214

theorem solve_linear_system :
  ∃ x y : ℚ, (3 * x - y = 4) ∧ (6 * x - 3 * y = 10) ∧ (x = 2 / 3) ∧ (y = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_linear_system_l432_43214


namespace NUMINAMATH_GPT_find_length_of_train_l432_43239

def speed_kmh : Real := 60
def time_to_cross_bridge : Real := 26.997840172786177
def length_of_bridge : Real := 340

noncomputable def speed_ms : Real := speed_kmh * (1000 / 3600)
noncomputable def total_distance : Real := speed_ms * time_to_cross_bridge
noncomputable def length_of_train : Real := total_distance - length_of_bridge

theorem find_length_of_train :
  length_of_train = 109.9640028797695 := 
sorry

end NUMINAMATH_GPT_find_length_of_train_l432_43239


namespace NUMINAMATH_GPT_james_music_listening_hours_l432_43284

theorem james_music_listening_hours (BPM : ℕ) (beats_per_week : ℕ) (hours_per_day : ℕ) 
  (h1 : BPM = 200) 
  (h2 : beats_per_week = 168000) 
  (h3 : hours_per_day * 200 * 60 * 7 = beats_per_week) : 
  hours_per_day = 2 := 
by
  sorry

end NUMINAMATH_GPT_james_music_listening_hours_l432_43284


namespace NUMINAMATH_GPT_triangle_angle_area_l432_43207

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.sqrt 3 * Real.cos x
variables {A B C : ℝ}
variables {BC : ℝ}
variables {S : ℝ}

theorem triangle_angle_area (hABC : A + B + C = π) (hBC : BC = 2) (h_fA : f A = 0) 
  (hA : A = π / 3) : S = Real.sqrt 3 :=
by
  -- Sorry, proof skipped
  sorry

end NUMINAMATH_GPT_triangle_angle_area_l432_43207


namespace NUMINAMATH_GPT_inverse_value_of_f_l432_43240

theorem inverse_value_of_f (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2^x - 2) : f⁻¹ 2 = 3 :=
sorry

end NUMINAMATH_GPT_inverse_value_of_f_l432_43240


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l432_43290

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l432_43290


namespace NUMINAMATH_GPT_find_f_g_2_l432_43218

def f (x : ℝ) : ℝ := x + 3
def g (x : ℝ) : ℝ := x^2 - 6

theorem find_f_g_2 : f (g 2) = 1 := 
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_f_g_2_l432_43218


namespace NUMINAMATH_GPT_halve_second_column_l432_43223

-- Definitions of given matrices
variable (f g h i : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := ![![f, g], ![h, i]])
variable (N : Matrix (Fin 2) (Fin 2) ℝ := ![![1, 0], ![0, (1/2)]])

-- Proof statement to be proved
theorem halve_second_column (hf : f ≠ 0) (hh : h ≠ 0) : N * A = ![![f, (1/2) * g], ![h, (1/2) * i]] := by
  sorry

end NUMINAMATH_GPT_halve_second_column_l432_43223
