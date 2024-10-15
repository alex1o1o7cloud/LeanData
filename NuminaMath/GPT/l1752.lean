import Mathlib

namespace NUMINAMATH_GPT_area_of_large_rectangle_l1752_175284

noncomputable def areaEFGH : ℕ :=
  let shorter_side := 3
  let longer_side := 2 * shorter_side
  let width_EFGH := shorter_side + shorter_side
  let length_EFGH := longer_side + longer_side
  width_EFGH * length_EFGH

theorem area_of_large_rectangle :
  areaEFGH = 72 := by
  sorry

end NUMINAMATH_GPT_area_of_large_rectangle_l1752_175284


namespace NUMINAMATH_GPT_min_floor_sum_l1752_175215

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ (n : ℕ), n = 4 ∧ n = 
  ⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(2 * c + a) / b⌋ := 
sorry

end NUMINAMATH_GPT_min_floor_sum_l1752_175215


namespace NUMINAMATH_GPT_intersection_is_singleton_l1752_175245

-- Definitions of sets M and N
def M : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2}
def N : Set (ℝ × ℝ) := {p | p.1 - p.2 = 4}

-- The stated proposition we need to prove
theorem intersection_is_singleton :
  M ∩ N = {(3, -1)} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_is_singleton_l1752_175245


namespace NUMINAMATH_GPT_river_current_speed_l1752_175217

/--
Given conditions:
- The rower realized the hat was missing 15 minutes after passing under the bridge.
- The rower caught the hat 15 minutes later.
- The total distance the hat traveled from the bridge is 1 kilometer.
Prove that the speed of the river current is 2 km/h.
-/
theorem river_current_speed (t1 t2 d : ℝ) (h_t1 : t1 = 15 / 60) (h_t2 : t2 = 15 / 60) (h_d : d = 1) : 
  d / (t1 + t2) = 2 := by
sorry

end NUMINAMATH_GPT_river_current_speed_l1752_175217


namespace NUMINAMATH_GPT_unique_root_when_abs_t_gt_2_l1752_175244

theorem unique_root_when_abs_t_gt_2 (t : ℝ) (h : |t| > 2) :
  ∃! x : ℝ, x^3 - 3 * x = t ∧ |x| > 2 :=
sorry

end NUMINAMATH_GPT_unique_root_when_abs_t_gt_2_l1752_175244


namespace NUMINAMATH_GPT_square_area_l1752_175224

-- Definition of the vertices' coordinates
def y_coords := ({-3, 2, 2, -3} : Set ℤ)
def x_coords_when_y2 := ({0, 5} : Set ℤ)

-- The statement we need to prove
theorem square_area (h1 : y_coords = {-3, 2, 2, -3}) 
                     (h2 : x_coords_when_y2 = {0, 5}) : 
                     ∃ s : ℤ, s^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l1752_175224


namespace NUMINAMATH_GPT_cesaro_sum_100_terms_l1752_175268

noncomputable def cesaro_sum (A : List ℝ) : ℝ :=
  let n := A.length
  (List.sum A) / n

theorem cesaro_sum_100_terms :
  ∀ (A : List ℝ), A.length = 99 →
  cesaro_sum A = 1000 →
  cesaro_sum (1 :: A) = 991 :=
by
  intros A h1 h2
  sorry

end NUMINAMATH_GPT_cesaro_sum_100_terms_l1752_175268


namespace NUMINAMATH_GPT_valentines_count_l1752_175248

theorem valentines_count (x y : ℕ) (h : x * y = x + y + 42) : x * y = 88 := by
  sorry

end NUMINAMATH_GPT_valentines_count_l1752_175248


namespace NUMINAMATH_GPT_construct_segment_length_l1752_175266

theorem construct_segment_length (a b : ℝ) (h : a > b) : 
  ∃ c : ℝ, c = (a^2 + b^2) / (a - b) :=
by
  sorry

end NUMINAMATH_GPT_construct_segment_length_l1752_175266


namespace NUMINAMATH_GPT_length_AB_eight_l1752_175292

-- Define parabola and line
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (x y k : ℝ) : Prop := y = k * x - k

-- Define intersection points A and B
def intersects (p1 p2 : ℝ × ℝ) (k : ℝ) : Prop :=
  parabola p1.1 p1.2 ∧ line p1.1 p1.2 k ∧
  parabola p2.1 p2.2 ∧ line p2.1 p2.2 k

-- Define midpoint distance condition
def midpoint_condition (p1 p2 : ℝ × ℝ) : Prop :=
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  midpoint.1 = 3

-- The main theorem statement
theorem length_AB_eight (k : ℝ) (A B : ℝ × ℝ) (h1 : intersects A B k)
  (h2 : midpoint_condition A B) : abs ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 64 := 
sorry

end NUMINAMATH_GPT_length_AB_eight_l1752_175292


namespace NUMINAMATH_GPT_bowling_ball_weight_l1752_175255

theorem bowling_ball_weight (b c : ℕ) 
  (h1 : 5 * b = 3 * c) 
  (h2 : 3 * c = 105) : 
  b = 21 := 
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l1752_175255


namespace NUMINAMATH_GPT_quadratic_roots_sum_product_l1752_175251

theorem quadratic_roots_sum_product :
  ∀ (x1 x2 : ℝ), (x1^2 - 2 * x1 - 8 = 0 ∧ x2^2 - 2 * x2 - 8 = 0 ∧ x1 ≠ x2) →
    (x1 + x2) / (x1 * x2) = -1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_sum_product_l1752_175251


namespace NUMINAMATH_GPT_deer_distribution_l1752_175218

theorem deer_distribution :
  ∃ a : ℕ → ℚ,
    (a 1 + a 2 + a 3 + a 4 + a 5 = 5) ∧
    (a 4 = 2 / 3) ∧ 
    (a 3 = 1) ∧ 
    (a 1 = 5 / 3) :=
by
  sorry

end NUMINAMATH_GPT_deer_distribution_l1752_175218


namespace NUMINAMATH_GPT_purchased_both_books_l1752_175204

theorem purchased_both_books: 
  ∀ (A B AB C : ℕ), A = 2 * B → AB = 2 * (B - AB) → C = 1000 → C = A - AB → AB = 500 := 
by
  intros A B AB C h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_purchased_both_books_l1752_175204


namespace NUMINAMATH_GPT_fraction_in_orange_tin_l1752_175231

variables {C : ℕ} -- assume total number of cookies as a natural number

theorem fraction_in_orange_tin (h1 : 11 / 12 = (1 / 6) + (5 / 12) + w)
  (h2 : 1 - (11 / 12) = 1 / 12) :
  w = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_in_orange_tin_l1752_175231


namespace NUMINAMATH_GPT_single_point_graph_l1752_175272

theorem single_point_graph (d : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 8 * y + d = 0 → x = -1 ∧ y = 4) → d = 19 :=
by
  sorry

end NUMINAMATH_GPT_single_point_graph_l1752_175272


namespace NUMINAMATH_GPT_Rams_monthly_salary_l1752_175297

variable (R S A : ℝ)
variable (annual_salary : ℝ)
variable (monthly_salary_conversion : annual_salary / 12 = A)
variable (ram_shyam_condition : 0.10 * R = 0.08 * S)
variable (shyam_abhinav_condition : S = 2 * A)
variable (abhinav_annual_salary : annual_salary = 192000)

theorem Rams_monthly_salary 
  (annual_salary : ℝ)
  (ram_shyam_condition : 0.10 * R = 0.08 * S)
  (shyam_abhinav_condition : S = 2 * A)
  (abhinav_annual_salary : annual_salary = 192000)
  (monthly_salary_conversion: annual_salary / 12 = A): 
  R = 25600 := by
  sorry

end NUMINAMATH_GPT_Rams_monthly_salary_l1752_175297


namespace NUMINAMATH_GPT_min_x2_y2_eq_16_then_product_zero_l1752_175202

theorem min_x2_y2_eq_16_then_product_zero
  (x y : ℝ)
  (h1 : ∃ x y : ℝ, (x^2 + y^2 = 16 ∧ ∀ a b : ℝ, a^2 + b^2 ≥ 16) ) :
  (x + 4) * (y - 4) = 0 := 
sorry

end NUMINAMATH_GPT_min_x2_y2_eq_16_then_product_zero_l1752_175202


namespace NUMINAMATH_GPT_sequence_a_100_l1752_175252

theorem sequence_a_100 (a : ℕ → ℤ) (h₁ : a 1 = 3) (h₂ : ∀ n : ℕ, a (n + 1) = a n - 2) : a 100 = -195 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_100_l1752_175252


namespace NUMINAMATH_GPT_travel_time_without_paddles_l1752_175288

variables (A B : Type) (v v_r S : ℝ)
noncomputable def time_to_travel (distance velocity : ℝ) := distance / velocity

-- Condition: The travel time from A to B is 3 times the travel time from B to A
axiom travel_condition : (time_to_travel S (v + v_r)) = 3 * (time_to_travel S (v - v_r))

-- Condition: We are considering travel from B to A by canoe without paddles
noncomputable def time_without_paddles := time_to_travel S v_r

-- Proving that without paddles it takes 3 times longer than usual (using canoes with paddles)
theorem travel_time_without_paddles :
  time_without_paddles S v_r = 3 * (time_to_travel S (v - v_r)) :=
sorry

end NUMINAMATH_GPT_travel_time_without_paddles_l1752_175288


namespace NUMINAMATH_GPT_diamonds_count_l1752_175277

-- Definitions based on the conditions given in the problem
def totalGems : Nat := 5155
def rubies : Nat := 5110
def diamonds (total rubies : Nat) : Nat := total - rubies

-- Statement of the proof problem
theorem diamonds_count : diamonds totalGems rubies = 45 := by
  sorry

end NUMINAMATH_GPT_diamonds_count_l1752_175277


namespace NUMINAMATH_GPT_carolyn_sum_correct_l1752_175280

def initial_sequence := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def carolyn_removes : List ℕ := [4, 8, 10, 9]

theorem carolyn_sum_correct : carolyn_removes.sum = 31 :=
by
  sorry

end NUMINAMATH_GPT_carolyn_sum_correct_l1752_175280


namespace NUMINAMATH_GPT_anand_income_l1752_175228

theorem anand_income
  (x y : ℕ)
  (h1 : 5 * x - 3 * y = 800)
  (h2 : 4 * x - 2 * y = 800) : 
  5 * x = 2000 := 
sorry

end NUMINAMATH_GPT_anand_income_l1752_175228


namespace NUMINAMATH_GPT_cost_per_kg_after_30_l1752_175290

theorem cost_per_kg_after_30 (l m : ℝ) 
  (hl : l = 20) 
  (h1 : 30 * l + 3 * m = 663) 
  (h2 : 30 * l + 6 * m = 726) : 
  m = 21 :=
by
  -- Proof will be written here
  sorry

end NUMINAMATH_GPT_cost_per_kg_after_30_l1752_175290


namespace NUMINAMATH_GPT_playerA_winning_conditions_l1752_175260

def playerA_has_winning_strategy (n : ℕ) : Prop :=
  (n % 4 = 0) ∨ (n % 4 = 3)

theorem playerA_winning_conditions (n : ℕ) (h : n ≥ 2) : 
  playerA_has_winning_strategy n ↔ (n % 4 = 0 ∨ n % 4 = 3) :=
by sorry

end NUMINAMATH_GPT_playerA_winning_conditions_l1752_175260


namespace NUMINAMATH_GPT_average_bacterial_count_closest_to_true_value_l1752_175285

-- Define the conditions
variables (dilution_spread_plate_method : Prop)
          (count_has_randomness : Prop)
          (count_not_uniform : Prop)

-- State the theorem
theorem average_bacterial_count_closest_to_true_value
  (h1: dilution_spread_plate_method)
  (h2: count_has_randomness)
  (h3: count_not_uniform)
  : true := sorry

end NUMINAMATH_GPT_average_bacterial_count_closest_to_true_value_l1752_175285


namespace NUMINAMATH_GPT_preimage_of_5_1_is_2_3_l1752_175287

-- Define the mapping function
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, 2*p.1 - p.2)

-- Define the pre-image condition for (5, 1)
theorem preimage_of_5_1_is_2_3 : ∃ p : ℝ × ℝ, f p = (5, 1) ∧ p = (2, 3) :=
by
  -- Here we state that such a point p exists with the required properties.
  sorry

end NUMINAMATH_GPT_preimage_of_5_1_is_2_3_l1752_175287


namespace NUMINAMATH_GPT_base10_to_base7_conversion_l1752_175259

theorem base10_to_base7_conversion : 2023 = 5 * 7^3 + 6 * 7^2 + 2 * 7^1 + 0 * 7^0 :=
  sorry

end NUMINAMATH_GPT_base10_to_base7_conversion_l1752_175259


namespace NUMINAMATH_GPT_main_theorem_l1752_175282

-- Define the function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Condition: f is symmetric about x = 1
def symmetric_about_one (a b c : ℝ) : Prop := 
  ∀ x : ℝ, f a b c (1 - x) = f a b c (1 + x)

-- Main statement
theorem main_theorem (a b c : ℝ) (h₁ : 0 < a) (h₂ : symmetric_about_one a b c) :
  ∀ x : ℝ, f a b c (2^x) > f a b c (3^x) :=
sorry

end NUMINAMATH_GPT_main_theorem_l1752_175282


namespace NUMINAMATH_GPT_cube_properties_l1752_175225

theorem cube_properties (s y : ℝ) (h1 : s^3 = 8 * y) (h2 : 6 * s^2 = 6 * y) : y = 64 := by
  sorry

end NUMINAMATH_GPT_cube_properties_l1752_175225


namespace NUMINAMATH_GPT_solve_system_of_equations_l1752_175249

theorem solve_system_of_equations (x y : ℝ) (hx : x + y + Real.sqrt (x * y) = 28)
  (hy : x^2 + y^2 + x * y = 336) : (x = 4 ∧ y = 16) ∨ (x = 16 ∧ y = 4) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1752_175249


namespace NUMINAMATH_GPT_b_k_divisible_by_11_is_5_l1752_175246

def b (n : ℕ) : ℕ :=
  -- Function to concatenate numbers from 1 to n
  let digits := List.join (List.map (λ x => Nat.digits 10 x) (List.range' 1 n.succ))
  digits.foldl (λ acc d => acc * 10 + d) 0

def g (n : ℕ) : ℤ :=
  let digits := Nat.digits 10 n
  digits.enum.foldl (λ acc ⟨i, d⟩ => if i % 2 = 0 then acc + Int.ofNat d else acc - Int.ofNat d) 0

def isDivisibleBy11 (n : ℕ) : Bool :=
  g n % 11 = 0

def count_b_k_divisible_by_11 : ℕ :=
  List.length (List.filter isDivisibleBy11 (List.map b (List.range' 1 51)))

theorem b_k_divisible_by_11_is_5 : count_b_k_divisible_by_11 = 5 := by
  sorry

end NUMINAMATH_GPT_b_k_divisible_by_11_is_5_l1752_175246


namespace NUMINAMATH_GPT_question_solution_l1752_175293

theorem question_solution 
  (hA : -(-1) = abs (-1))
  (hB : ¬ (∃ n : ℤ, ∀ m : ℤ, n < m ∧ m < 0))
  (hC : (-2)^3 = -2^3)
  (hD : ∃ q : ℚ, q = 0) :
  ¬ (∀ q : ℚ, q > 0 ∨ q < 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_question_solution_l1752_175293


namespace NUMINAMATH_GPT_drawing_two_black_balls_probability_equals_half_l1752_175296

noncomputable def total_number_of_events : ℕ := 6

noncomputable def number_of_black_draw_events : ℕ := 3

noncomputable def probability_of_drawing_two_black_balls : ℚ :=
  number_of_black_draw_events / total_number_of_events

theorem drawing_two_black_balls_probability_equals_half :
  probability_of_drawing_two_black_balls = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_drawing_two_black_balls_probability_equals_half_l1752_175296


namespace NUMINAMATH_GPT_arccos_cos_three_l1752_175206

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi :=
  sorry

end NUMINAMATH_GPT_arccos_cos_three_l1752_175206


namespace NUMINAMATH_GPT_largest_five_digit_number_with_product_120_l1752_175261

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def prod_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldr (· * ·) 1

def max_five_digit_prod_120 : ℕ := 85311

theorem largest_five_digit_number_with_product_120 :
  is_five_digit max_five_digit_prod_120 ∧ prod_of_digits max_five_digit_prod_120 = 120 :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_number_with_product_120_l1752_175261


namespace NUMINAMATH_GPT_tan_13pi_div_3_eq_sqrt_3_l1752_175239

theorem tan_13pi_div_3_eq_sqrt_3 : Real.tan (13 * Real.pi / 3) = Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_tan_13pi_div_3_eq_sqrt_3_l1752_175239


namespace NUMINAMATH_GPT_count_solutions_cos2x_plus_3sin2x_eq_1_l1752_175243

open Real

theorem count_solutions_cos2x_plus_3sin2x_eq_1 :
  ∀ x : ℝ, (-10 < x ∧ x < 45 → cos x ^ 2 + 3 * sin x ^ 2 = 1) → 
  ∃! n : ℕ, n = 18 := 
by
  intro x hEq
  sorry

end NUMINAMATH_GPT_count_solutions_cos2x_plus_3sin2x_eq_1_l1752_175243


namespace NUMINAMATH_GPT_average_chemistry_mathematics_l1752_175279

-- Define the conditions 
variable {P C M : ℝ} -- Marks in Physics, Chemistry, and Mathematics

-- The given condition in the problem
theorem average_chemistry_mathematics (h : P + C + M = P + 130) : (C + M) / 2 = 65 := 
by
  -- This will be the main proof block (we use 'sorry' to omit the actual proof)
  sorry

end NUMINAMATH_GPT_average_chemistry_mathematics_l1752_175279


namespace NUMINAMATH_GPT_a_finishes_race_in_t_seconds_l1752_175216

theorem a_finishes_race_in_t_seconds 
  (time_B : ℝ := 45)
  (dist_B : ℝ := 100)
  (dist_A_wins_by : ℝ := 20)
  (total_dist : ℝ := 100)
  : ∃ t : ℝ, t = 36 := 
  sorry

end NUMINAMATH_GPT_a_finishes_race_in_t_seconds_l1752_175216


namespace NUMINAMATH_GPT_y_intercept_of_line_l1752_175247

theorem y_intercept_of_line : ∀ (x y : ℝ), (3 * x - 4 * y = 12) → (x = 0) → (y = -3) :=
by
  intros x y h_eq h_x0
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1752_175247


namespace NUMINAMATH_GPT_speed_increase_percentage_l1752_175257

variable (T : ℚ)  -- usual travel time in minutes
variable (v : ℚ)  -- usual speed

-- Conditions
-- Ivan usually arrives at 9:00 AM, traveling for T minutes at speed v.
-- When Ivan leaves 40 minutes late and drives 1.6 times his usual speed, he arrives at 8:35 AM
def usual_arrival_time : ℚ := 9 * 60  -- 9:00 AM in minutes

def time_when_late : ℚ := (9 * 60) + 40 - (25 + 40)  -- 8:35 AM in minutes

def increased_speed := 1.6 * v -- 60% increase in speed

def time_taken_with_increased_speed := T - 65

theorem speed_increase_percentage :
  ((T / (T - 40)) = 1.3) :=
by
-- assume the equation for usual time T in terms of increased speed is known
-- Use provided conditions and solve the equation to derive the result.
  sorry

end NUMINAMATH_GPT_speed_increase_percentage_l1752_175257


namespace NUMINAMATH_GPT_exponent_multiplication_l1752_175253

theorem exponent_multiplication (a : ℝ) (m n : ℝ) (h1 : a^m = 2) (h2 : a^n = 4) :
  a^(m + n) = 8 := by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l1752_175253


namespace NUMINAMATH_GPT_survey_participants_l1752_175230

-- Total percentage for option A and option B in bytes
def percent_A : ℝ := 0.50
def percent_B : ℝ := 0.30

-- Number of participants who chose option A
def participants_A : ℕ := 150

-- Target number of participants who chose option B (to be proved)
def participants_B : ℕ := 90

-- The theorem to prove the number of participants who chose option B
theorem survey_participants :
  (participants_B : ℝ) = participants_A * (percent_B / percent_A) :=
by
  sorry

end NUMINAMATH_GPT_survey_participants_l1752_175230


namespace NUMINAMATH_GPT_best_chart_for_temperature_changes_l1752_175278

def Pie_chart := "Represent the percentage of parts in the whole."
def Line_chart := "Represent changes over time."
def Bar_chart := "Show the specific number of each item."

theorem best_chart_for_temperature_changes : 
  "The best statistical chart to use for understanding temperature changes throughout a day" = Line_chart :=
by
  sorry

end NUMINAMATH_GPT_best_chart_for_temperature_changes_l1752_175278


namespace NUMINAMATH_GPT_true_statement_l1752_175207

def statement_i (i : ℕ) (n : ℕ) : Prop := 
  (i = (n - 1))

theorem true_statement :
  ∃! n : ℕ, (n ≤ 100 ∧ ∀ i, (i ≠ n - 1) → statement_i i n = false) ∧ statement_i (n - 1) n = true :=
by
  sorry

end NUMINAMATH_GPT_true_statement_l1752_175207


namespace NUMINAMATH_GPT_opposite_of_neg_five_is_five_l1752_175283

-- Define the condition for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that the opposite of -5 is 5
theorem opposite_of_neg_five_is_five : is_opposite (-5) 5 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_opposite_of_neg_five_is_five_l1752_175283


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1752_175241

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) 
(h_d2 : d2 = 18) (h_area : area = 126) (h_formula : area = (d1 * d2) / 2) : 
d1 = 14 :=
by
  -- We're skipping the proof steps.
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1752_175241


namespace NUMINAMATH_GPT_shortest_track_length_l1752_175209

open Nat

def Melanie_track_length := 8
def Martin_track_length := 20

theorem shortest_track_length :
  Nat.lcm Melanie_track_length Martin_track_length = 40 :=
by
  sorry

end NUMINAMATH_GPT_shortest_track_length_l1752_175209


namespace NUMINAMATH_GPT_find_term_of_sequence_l1752_175229

theorem find_term_of_sequence :
  ∀ (a d n : ℤ), a = -5 → d = -4 → (-4)*n + 1 = -401 → n = 100 :=
by
  intros a d n h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_find_term_of_sequence_l1752_175229


namespace NUMINAMATH_GPT_integer_between_squares_l1752_175250

theorem integer_between_squares (a b c d: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) (h₃: 0 < d) (h₄: c * d = 1) : 
  ∃ n : ℤ, ab ≤ n^2 ∧ n^2 ≤ (a + c) * (b + d) := 
by 
  sorry

end NUMINAMATH_GPT_integer_between_squares_l1752_175250


namespace NUMINAMATH_GPT_no_int_coeffs_l1752_175213

def P (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_int_coeffs (a b c d : ℤ) : 
  ¬ (P a b c d 19 = 1 ∧ P a b c d 62 = 2) :=
by sorry

end NUMINAMATH_GPT_no_int_coeffs_l1752_175213


namespace NUMINAMATH_GPT_profit_percentage_before_decrease_l1752_175275

-- Defining the conditions as Lean definitions
def newManufacturingCost : ℝ := 50
def oldManufacturingCost : ℝ := 80
def profitPercentageNew : ℝ := 0.5

-- Defining the problem as a theorem in Lean
theorem profit_percentage_before_decrease
  (P : ℝ)
  (hP : profitPercentageNew * P = P - newManufacturingCost) :
  ((P - oldManufacturingCost) / P) * 100 = 20 := 
by
  sorry

end NUMINAMATH_GPT_profit_percentage_before_decrease_l1752_175275


namespace NUMINAMATH_GPT_rectangle_is_square_l1752_175219

theorem rectangle_is_square
  (a b: ℝ)  -- rectangle side lengths
  (h: a ≠ b)  -- initial assumption: rectangle not a square
  (shift_perpendicular: ∀ (P Q R S: ℝ × ℝ), (P ≠ Q → Q ≠ R → R ≠ S → S ≠ P) → (∀ (shift: ℝ × ℝ → ℝ × ℝ), ∀ (P₁: ℝ × ℝ), shift P₁ = P₁ + (0, 1) ∨ shift P₁ = P₁ + (1, 0)) → false):
  False := sorry

end NUMINAMATH_GPT_rectangle_is_square_l1752_175219


namespace NUMINAMATH_GPT_vat_percentage_is_15_l1752_175214

def original_price : ℝ := 1700
def final_price : ℝ := 1955
def tax_amount := final_price - original_price

theorem vat_percentage_is_15 :
  (tax_amount / original_price) * 100 = 15 := 
sorry

end NUMINAMATH_GPT_vat_percentage_is_15_l1752_175214


namespace NUMINAMATH_GPT_initial_strawberries_l1752_175254

-- Define the conditions
def strawberries_eaten : ℝ := 42.0
def strawberries_left : ℝ := 36.0

-- State the theorem
theorem initial_strawberries :
  strawberries_eaten + strawberries_left = 78 :=
by
  sorry

end NUMINAMATH_GPT_initial_strawberries_l1752_175254


namespace NUMINAMATH_GPT_council_counts_l1752_175276

theorem council_counts 
    (total_classes : ℕ := 20)
    (students_per_class : ℕ := 5)
    (total_students : ℕ := 100)
    (petya_class_council : ℕ × ℕ := (1, 4))  -- (boys, girls)
    (equal_boys_girls : 2 * 50 = total_students)  -- Equal number of boys and girls
    (more_girls_classes : ℕ := 15)
    (min_girls_each : ℕ := 3)
    (remaining_classes : ℕ := 4)
    (remaining_students : ℕ := 20)
    : (19, 1) = (19, 1) :=
by
    -- actual proof goes here
    sorry

end NUMINAMATH_GPT_council_counts_l1752_175276


namespace NUMINAMATH_GPT_exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l1752_175205

-- Definitions based on the given conditions
def male_students := 3
def female_students := 2
def total_students := male_students + female_students

def at_least_1_male_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ male_students
def all_female_event := ∀ (n : ℕ), n ≤ female_students
def at_least_1_female_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ female_students
def all_male_event := ∀ (n : ℕ), n ≤ male_students
def exactly_1_male_event := ∃ (n : ℕ), n = 1 ∧ n ≤ male_students
def exactly_2_female_event := ∃ (n : ℕ), n = 2 ∧ n ≤ female_students

def mutually_exclusive (e1 e2 : Prop) : Prop := ¬ (e1 ∧ e2)
def complementary (e1 e2 : Prop) : Prop := e1 ∧ ¬ e2 ∨ ¬ e1 ∧ e2

-- Statement of the problem
theorem exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary :
  mutually_exclusive exactly_1_male_event exactly_2_female_event ∧ 
  ¬ complementary exactly_1_male_event exactly_2_female_event :=
by
  sorry

end NUMINAMATH_GPT_exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l1752_175205


namespace NUMINAMATH_GPT_probability_black_then_red_l1752_175263

/-- Definition of a standard deck -/
def standard_deck := {cards : Finset (Fin 52) // cards.card = 52}

/-- Definition of black cards in the deck -/
def black_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Definition of red cards in the deck -/
def red_cards := {cards : Finset (Fin 52) // cards.card = 26}

/-- Probability of drawing the top card as black and the second card as red -/
def prob_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) : ℚ :=
  (26 * 26) / (52 * 51)

theorem probability_black_then_red (deck : standard_deck) (black : black_cards) (red : red_cards) :
  prob_black_then_red deck black red = 13 / 51 :=
sorry

end NUMINAMATH_GPT_probability_black_then_red_l1752_175263


namespace NUMINAMATH_GPT_giselle_initial_doves_l1752_175299

theorem giselle_initial_doves (F : ℕ) (h1 : ∀ F, F > 0) (h2 : 3 * F * 3 / 4 + F = 65) : F = 20 :=
sorry

end NUMINAMATH_GPT_giselle_initial_doves_l1752_175299


namespace NUMINAMATH_GPT_quadrilateral_pyramid_volume_l1752_175234

theorem quadrilateral_pyramid_volume (h Q : ℝ) : 
  ∃ V : ℝ, V = (2 / 3 : ℝ) * h * (Real.sqrt (h^2 + 4 * Q^2) - h^2) :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_pyramid_volume_l1752_175234


namespace NUMINAMATH_GPT_option_c_is_incorrect_l1752_175264

/-- Define the temperature data -/
def temps : List Int := [-20, -10, 0, 10, 20, 30]

/-- Define the speed of sound data corresponding to the temperatures -/
def speeds : List Int := [318, 324, 330, 336, 342, 348]

/-- The speed of sound at 10 degrees Celsius -/
def speed_at_10 : Int := 336

/-- The incorrect claim in option C -/
def incorrect_claim : Prop := (speed_at_10 * 4 ≠ 1334)

/-- Prove that the claim in option C is incorrect -/
theorem option_c_is_incorrect : incorrect_claim :=
by {
  sorry
}

end NUMINAMATH_GPT_option_c_is_incorrect_l1752_175264


namespace NUMINAMATH_GPT_mr_blues_yard_expectation_l1752_175223

noncomputable def calculate_expected_harvest (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let area := length_feet * width_feet
  let total_yield := area * yield_per_sqft
  total_yield

theorem mr_blues_yard_expectation : calculate_expected_harvest 18 25 2.5 (3 / 4) = 2109.375 :=
by
  sorry

end NUMINAMATH_GPT_mr_blues_yard_expectation_l1752_175223


namespace NUMINAMATH_GPT_domain_shift_l1752_175270

theorem domain_shift (f : ℝ → ℝ) (dom_f : ∀ x, 1 ≤ x ∧ x ≤ 4 → f x = f x) :
  ∀ x, -1 ≤ x ∧ x ≤ 2 ↔ (1 ≤ x + 2 ∧ x + 2 ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_domain_shift_l1752_175270


namespace NUMINAMATH_GPT_num_integers_satisfy_inequality_l1752_175222

theorem num_integers_satisfy_inequality : ∃ (s : Finset ℤ), (∀ x ∈ s, |7 * x - 5| ≤ 15) ∧ s.card = 5 :=
by
  sorry

end NUMINAMATH_GPT_num_integers_satisfy_inequality_l1752_175222


namespace NUMINAMATH_GPT_BD_value_l1752_175208

noncomputable def triangleBD (AC BC AD CD : ℝ) : ℝ :=
  let θ := Real.arccos ((3 ^ 2 + 9 ^ 2 - 7 ^ 2) / (2 * 3 * 9))
  let ψ := Real.pi - θ
  let cosψ := Real.cos ψ
  let x := (-1.026 + Real.sqrt ((1.026 ^ 2) + 4 * 40)) / 2
  if x > 0 then x else 5.8277 -- confirmed manually as positive root.

theorem BD_value : (triangleBD 7 7 9 3) = 5.8277 :=
by
  apply sorry

end NUMINAMATH_GPT_BD_value_l1752_175208


namespace NUMINAMATH_GPT_range_of_m_value_of_x_l1752_175235

noncomputable def a : ℝ := 3 / 2

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log a

-- Statement for the range of m
theorem range_of_m :
  ∀ m : ℝ, f (3 * m - 2) < f (2 * m + 5) ↔ (2 / 3) < m ∧ m < 7 :=
by
  intro m
  sorry

-- Value of x
theorem value_of_x :
  ∃ x : ℝ, f (x - 2 / x) = Real.log (7 / 2) / Real.log (3 / 2) ∧ x > 0 ∧ x = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_range_of_m_value_of_x_l1752_175235


namespace NUMINAMATH_GPT_greatest_of_consecutive_even_numbers_l1752_175269

theorem greatest_of_consecutive_even_numbers (n : ℤ) (h : ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 35) : n + 4 = 39 :=
by
  sorry

end NUMINAMATH_GPT_greatest_of_consecutive_even_numbers_l1752_175269


namespace NUMINAMATH_GPT_problem_statement_l1752_175221

theorem problem_statement :
  (1 / 3 * 1 / 6 * P = (1 / 4 * 1 / 8 * 64) + (1 / 5 * 1 / 10 * 100)) → 
  P = 72 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1752_175221


namespace NUMINAMATH_GPT_part1_part2_l1752_175289

def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

theorem part1 (x : ℝ) : f x ≥ 3 ↔ x ≤ -3 / 2 ∨ x ≥ 3 / 2 := 
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a) ↔ -1 ≤ a ∧ a ≤ 2 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1752_175289


namespace NUMINAMATH_GPT_engineer_days_l1752_175281

theorem engineer_days (x : ℕ) (k : ℕ) (d : ℕ) (n : ℕ) (m : ℕ) (e : ℕ)
  (h1 : k = 10) -- Length of the road in km
  (h2 : d = 15) -- Total days to complete the project
  (h3 : n = 30) -- Initial number of men
  (h4 : m = 2) -- Length of the road completed in x days
  (h5 : e = n + 30) -- New number of men
  (h6 : (4 : ℚ) / x = (8 : ℚ) / (d - x)) : x = 5 :=
by
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_engineer_days_l1752_175281


namespace NUMINAMATH_GPT_sphere_volume_from_surface_area_l1752_175265

theorem sphere_volume_from_surface_area (S : ℝ) (V : ℝ) (R : ℝ) (h1 : S = 36 * Real.pi) (h2 : S = 4 * Real.pi * R ^ 2) (h3 : V = (4 / 3) * Real.pi * R ^ 3) : V = 36 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_from_surface_area_l1752_175265


namespace NUMINAMATH_GPT_radius_of_larger_circle_15_l1752_175271

def radius_larger_circle (r1 r2 r3 r : ℝ) : Prop :=
  ∃ (A B C O : EuclideanSpace ℝ (Fin 2)), 
    dist A B = r1 + r2 ∧
    dist B C = r2 + r3 ∧
    dist A C = r1 + r3 ∧
    dist O A = r - r1 ∧
    dist O B = r - r2 ∧
    dist O C = r - r3 ∧
    (dist O A + r1 = r ∧
    dist O B + r2 = r ∧
    dist O C + r3 = r)

theorem radius_of_larger_circle_15 :
  radius_larger_circle 10 3 2 15 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_15_l1752_175271


namespace NUMINAMATH_GPT_least_N_no_square_l1752_175201

theorem least_N_no_square (N : ℕ) : 
  (∀ k, (1000 * N) ≤ k ∧ k ≤ (1000 * N + 999) → 
  ∃ m, ¬ (k = m^2)) ↔ N = 282 :=
by
  sorry

end NUMINAMATH_GPT_least_N_no_square_l1752_175201


namespace NUMINAMATH_GPT_ac_lt_bc_of_a_gt_b_and_c_lt_0_l1752_175212

theorem ac_lt_bc_of_a_gt_b_and_c_lt_0 {a b c : ℝ} (h1 : a > b) (h2 : c < 0) : a * c < b * c :=
  sorry

end NUMINAMATH_GPT_ac_lt_bc_of_a_gt_b_and_c_lt_0_l1752_175212


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1752_175242

theorem solution_set_of_inequality :
  ∀ x : ℝ, (x^2 - 3*x - 4 ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 4) :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1752_175242


namespace NUMINAMATH_GPT_negation_example_l1752_175203

open Real

theorem negation_example : 
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n ≥ x^2) ↔ ∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2 := 
  sorry

end NUMINAMATH_GPT_negation_example_l1752_175203


namespace NUMINAMATH_GPT_black_piece_is_option_C_l1752_175286

-- Definitions for the problem conditions
def rectangular_prism (cubes : Nat) := cubes = 16
def block (small_cubes : Nat) := small_cubes = 4
def piece_containing_black_shape_is_partially_seen (rows : Nat) := rows = 2

-- Hypotheses and conditions
variable (rect_prism : Nat) (block1 block2 block3 block4 : Nat)
variable (visibility_block1 visibility_block2 visibility_block3 : Bool)
variable (visible_in_back_row : Bool)

-- Given conditions based on the problem statement
axiom h1 : rectangular_prism rect_prism
axiom h2 : block block1
axiom h3 : block block2
axiom h4 : block block3
axiom h5 : block block4
axiom h6 : visibility_block1 = true
axiom h7 : visibility_block2 = true
axiom h8 : visibility_block3 = true
axiom h9 : visible_in_back_row = true

-- Prove the configuration matches Option C
theorem black_piece_is_option_C :
  ∀ (config : Char), (config = 'C') :=
by
  intros
  -- Proof incomplete intentionally.
  sorry

end NUMINAMATH_GPT_black_piece_is_option_C_l1752_175286


namespace NUMINAMATH_GPT_sum_is_integer_l1752_175298

theorem sum_is_integer (x y z : ℝ) (h1 : x ^ 2 = y + 2) (h2 : y ^ 2 = z + 2) (h3 : z ^ 2 = x + 2) : ∃ n : ℤ, x + y + z = n :=
  sorry

end NUMINAMATH_GPT_sum_is_integer_l1752_175298


namespace NUMINAMATH_GPT_q_minus_r_l1752_175220

noncomputable def problem (x : ℝ) : Prop :=
  (5 * x - 15) / (x^2 + x - 20) = x + 3

def q_and_r (q r : ℝ) : Prop :=
  q ≠ r ∧ problem q ∧ problem r ∧ q > r

theorem q_minus_r (q r : ℝ) (h : q_and_r q r) : q - r = 2 :=
  sorry

end NUMINAMATH_GPT_q_minus_r_l1752_175220


namespace NUMINAMATH_GPT_pavan_total_distance_l1752_175267

theorem pavan_total_distance:
  ∀ (D : ℝ),
  (∃ Time1 Time2,
    Time1 = (D / 2) / 30 ∧
    Time2 = (D / 2) / 25 ∧
    Time1 + Time2 = 11)
  → D = 150 :=
by
  intros D h
  sorry

end NUMINAMATH_GPT_pavan_total_distance_l1752_175267


namespace NUMINAMATH_GPT_find_multiplier_l1752_175291

theorem find_multiplier (n k : ℤ) (h1 : n + 4 = 15) (h2 : 3 * n = k * (n + 4) + 3) : k = 2 :=
  sorry

end NUMINAMATH_GPT_find_multiplier_l1752_175291


namespace NUMINAMATH_GPT_ribbon_tape_needed_l1752_175295

theorem ribbon_tape_needed 
  (total_length : ℝ) (num_boxes : ℕ) (ribbon_per_box : ℝ)
  (h1 : total_length = 82.04)
  (h2 : num_boxes = 28)
  (h3 : total_length / num_boxes = ribbon_per_box)
  : ribbon_per_box = 2.93 :=
sorry

end NUMINAMATH_GPT_ribbon_tape_needed_l1752_175295


namespace NUMINAMATH_GPT_total_lives_l1752_175226

theorem total_lives (initial_players additional_players lives_per_player : ℕ) (h1 : initial_players = 4) (h2 : additional_players = 5) (h3 : lives_per_player = 3) :
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end NUMINAMATH_GPT_total_lives_l1752_175226


namespace NUMINAMATH_GPT_five_digit_palindromes_count_l1752_175238

theorem five_digit_palindromes_count : ∃ n : ℕ, n = 900 ∧
  ∀ (a b c : ℕ),
    1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    ∃ (x : ℕ), x = a * 10^4 + b * 10^3 + c * 10^2 + b * 10 + a := sorry

end NUMINAMATH_GPT_five_digit_palindromes_count_l1752_175238


namespace NUMINAMATH_GPT_average_age_of_women_l1752_175211

theorem average_age_of_women (A : ℕ) (W1 W2 : ℕ) 
  (h1 : 7 * A - 26 - 30 + W1 + W2 = 7 * (A + 4)) : 
  (W1 + W2) / 2 = 42 := 
by 
  sorry

end NUMINAMATH_GPT_average_age_of_women_l1752_175211


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l1752_175227

open Real

-- Defining the conditions as per the problem
def triangle_side_equality (AB AC : ℝ) : Prop := AB = AC
def angle_relation (angleBAC angleBTC : ℝ) : Prop := angleBAC = 2 * angleBTC
def side_length_BT (BT : ℝ) : Prop := BT = 70
def side_length_AT (AT : ℝ) : Prop := AT = 37

-- Proving the area of triangle ABC given the conditions
theorem area_of_triangle_ABC
  (AB AC : ℝ)
  (angleBAC angleBTC : ℝ)
  (BT AT : ℝ)
  (h1 : triangle_side_equality AB AC)
  (h2 : angle_relation angleBAC angleBTC)
  (h3 : side_length_BT BT)
  (h4 : side_length_AT AT) 
  : ∃ area : ℝ, area = 420 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l1752_175227


namespace NUMINAMATH_GPT_each_bug_ate_1_5_flowers_l1752_175236

-- Define the conditions given in the problem
def bugs : ℝ := 2.0
def flowers : ℝ := 3.0

-- The goal is to prove that the number of flowers each bug ate is 1.5
theorem each_bug_ate_1_5_flowers : (flowers / bugs) = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_each_bug_ate_1_5_flowers_l1752_175236


namespace NUMINAMATH_GPT_A_empty_iff_a_gt_9_over_8_A_one_element_l1752_175262

-- Definition of A based on a given condition
def A (a : ℝ) : Set ℝ := {x | a * x^2 - 3 * x + 2 = 0}

-- Problem 1: Prove that if A is empty, then a > 9/8
theorem A_empty_iff_a_gt_9_over_8 {a : ℝ} : 
  (A a = ∅) ↔ (a > 9 / 8) := 
sorry

-- Problem 2: Prove the elements in A when it contains only one element
theorem A_one_element {a : ℝ} : 
  (∃! x, x ∈ A a) ↔ (a = 0 ∧ (A a = {2 / 3})) ∨ (a = 9 / 8 ∧ (A a = {4 / 3})) := 
sorry

end NUMINAMATH_GPT_A_empty_iff_a_gt_9_over_8_A_one_element_l1752_175262


namespace NUMINAMATH_GPT_find_water_and_bucket_weight_l1752_175210

-- Define the original amount of water (x) and the weight of the bucket (y)
variables (x y : ℝ)

-- Given conditions described as hypotheses
def conditions (x y : ℝ) : Prop :=
  4 * x + y = 16 ∧ 6 * x + y = 22

-- The goal is to prove the values of x and y
theorem find_water_and_bucket_weight (h : conditions x y) : x = 3 ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_water_and_bucket_weight_l1752_175210


namespace NUMINAMATH_GPT_chord_cos_theta_condition_l1752_175258

open Real

-- Translation of the given conditions and proof problem
theorem chord_cos_theta_condition
  (a b x y θ : ℝ)
  (h1 : a^2 = b^2 + 2) :
  x * y = cos θ := 
sorry

end NUMINAMATH_GPT_chord_cos_theta_condition_l1752_175258


namespace NUMINAMATH_GPT_angle_sum_triangle_l1752_175200

theorem angle_sum_triangle (A B C : Type) (angle_A angle_B angle_C : ℝ) 
(h1 : angle_A = 45) (h2 : angle_B = 25) 
(h3 : angle_A + angle_B + angle_C = 180) : 
angle_C = 110 := 
sorry

end NUMINAMATH_GPT_angle_sum_triangle_l1752_175200


namespace NUMINAMATH_GPT_function_relationship_value_of_x_when_y_is_1_l1752_175294

variable (x y : ℝ) (k : ℝ)

-- Conditions
def inverse_proportion (x y : ℝ) : Prop :=
  ∃ k : ℝ, y = k / (x - 3)

axiom condition_1 : inverse_proportion x y
axiom condition_2 : y = 5 ∧ x = 4

-- Statements to be proved
theorem function_relationship :
  ∃ k : ℝ, (y = k / (x - 3)) ∧ (y = 5 ∧ x = 4 → k = 5) :=
by
  sorry

theorem value_of_x_when_y_is_1 (hy : y = 1) :
  ∃ x : ℝ, (y = 5 / (x - 3)) ∧ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_function_relationship_value_of_x_when_y_is_1_l1752_175294


namespace NUMINAMATH_GPT_find_x_when_perpendicular_l1752_175233

def a : ℝ × ℝ := (1, -2)
def b (x: ℝ) : ℝ × ℝ := (x, 1)
def are_perpendicular (a b: ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_x_when_perpendicular (x: ℝ) (h: are_perpendicular a (b x)) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_when_perpendicular_l1752_175233


namespace NUMINAMATH_GPT_non_congruent_triangles_count_l1752_175274

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def count_non_congruent_triangles : ℕ :=
  let a_values := [1, 2]
  let b_values := [2, 3]
  let triangles := [(1, 2, 2), (2, 2, 2), (2, 2, 3)]
  triangles.length

theorem non_congruent_triangles_count : count_non_congruent_triangles = 3 :=
  by
    -- Proof would go here
    sorry

end NUMINAMATH_GPT_non_congruent_triangles_count_l1752_175274


namespace NUMINAMATH_GPT_find_coprime_pairs_l1752_175273

theorem find_coprime_pairs :
  ∀ (x y : ℕ), x > 0 → y > 0 → x.gcd y = 1 →
    (x ∣ y^2 + 210) →
    (y ∣ x^2 + 210) →
    (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) ∨ 
    (∃ n : ℕ, n > 0 ∧ n = 1 ∧ n = 1 ∧ 
      (x = 212*n - n - 1 ∨ y = 212*n - n - 1)) := sorry

end NUMINAMATH_GPT_find_coprime_pairs_l1752_175273


namespace NUMINAMATH_GPT_initial_rotations_l1752_175232

-- Given conditions as Lean definitions
def rotations_per_block : ℕ := 200
def blocks_to_ride : ℕ := 8
def additional_rotations_needed : ℕ := 1000

-- Question translated to proof statement
theorem initial_rotations (rotations : ℕ) :
  rotations + additional_rotations_needed = rotations_per_block * blocks_to_ride → rotations = 600 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_initial_rotations_l1752_175232


namespace NUMINAMATH_GPT_cos_315_deg_l1752_175240

noncomputable def cos_315 : ℝ :=
  Real.cos (315 * Real.pi / 180)

theorem cos_315_deg : cos_315 = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_315_deg_l1752_175240


namespace NUMINAMATH_GPT_salon_revenue_l1752_175256

noncomputable def revenue (num_customers first_visit second_visit third_visit : ℕ) (first_charge second_charge : ℕ) : ℕ :=
  num_customers * first_charge + second_visit * second_charge + third_visit * second_charge

theorem salon_revenue : revenue 100 100 30 10 10 8 = 1320 :=
by
  unfold revenue
  -- The proof will continue here.
  sorry

end NUMINAMATH_GPT_salon_revenue_l1752_175256


namespace NUMINAMATH_GPT_total_instruments_correct_l1752_175237

def numberOfFlutesCharlie : ℕ := 1
def numberOfHornsCharlie : ℕ := 2
def numberOfHarpsCharlie : ℕ := 1
def numberOfDrumsCharlie : ℕ := 5

def numberOfFlutesCarli : ℕ := 3 * numberOfFlutesCharlie
def numberOfHornsCarli : ℕ := numberOfHornsCharlie / 2
def numberOfDrumsCarli : ℕ := 2 * numberOfDrumsCharlie
def numberOfHarpsCarli : ℕ := 0

def numberOfFlutesNick : ℕ := 2 * numberOfFlutesCarli - 1
def numberOfHornsNick : ℕ := numberOfHornsCharlie + numberOfHornsCarli
def numberOfDrumsNick : ℕ := 4 * numberOfDrumsCarli - 2
def numberOfHarpsNick : ℕ := 0

def numberOfFlutesDaisy : ℕ := numberOfFlutesNick * numberOfFlutesNick
def numberOfHornsDaisy : ℕ := (numberOfHornsNick - numberOfHornsCarli) / 2
def numberOfDrumsDaisy : ℕ := (numberOfDrumsCharlie + numberOfDrumsCarli + numberOfDrumsNick) / 3
def numberOfHarpsDaisy : ℕ := numberOfHarpsCharlie

def numberOfInstrumentsCharlie : ℕ := numberOfFlutesCharlie + numberOfHornsCharlie + numberOfHarpsCharlie + numberOfDrumsCharlie
def numberOfInstrumentsCarli : ℕ := numberOfFlutesCarli + numberOfHornsCarli + numberOfDrumsCarli
def numberOfInstrumentsNick : ℕ := numberOfFlutesNick + numberOfHornsNick + numberOfDrumsNick
def numberOfInstrumentsDaisy : ℕ := numberOfFlutesDaisy + numberOfHornsDaisy + numberOfHarpsDaisy + numberOfDrumsDaisy

def totalInstruments : ℕ := numberOfInstrumentsCharlie + numberOfInstrumentsCarli + numberOfInstrumentsNick + numberOfInstrumentsDaisy

theorem total_instruments_correct : totalInstruments = 113 := by
  sorry

end NUMINAMATH_GPT_total_instruments_correct_l1752_175237
