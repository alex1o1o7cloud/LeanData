import Mathlib

namespace number_of_players_l82_8211

theorem number_of_players (x y z : ℕ) 
  (h1 : x + y + z = 10)
  (h2 : x * y + y * z + z * x = 31) : 
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ (x = 2 ∧ y = 5 ∧ z = 3) ∨ (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ (x = 5 ∧ y = 2 ∧ z = 3) ∨ (x = 5 ∧ y = 3 ∧ z = 2) :=
sorry

end number_of_players_l82_8211


namespace proof_problem_l82_8210

-- Definitions of the propositions
def p : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → y = 5 - 3 * x
def q : Prop := ∀ (x y : ℝ), 6 * x + 2 * y - 1 = 0 → 2 * x + 6 * y - 4 = 0

-- Translate the mathematical proof problem into a Lean theorem
theorem proof_problem : 
  (p ∧ ¬q) ∧ ¬((¬p) ∧ q) :=
by
  -- You can fill in the exact proof steps here
  sorry

end proof_problem_l82_8210


namespace cell_division_relationship_l82_8265

noncomputable def number_of_cells_after_divisions (x : ℕ) : ℕ :=
  2^x

theorem cell_division_relationship (x : ℕ) : 
  number_of_cells_after_divisions x = 2^x := 
by 
  sorry

end cell_division_relationship_l82_8265


namespace average_words_per_hour_l82_8242

/-- Prove that given a total of 50,000 words written in 100 hours with the 
writing output increasing by 10% each subsequent hour, the average number 
of words written per hour is 500. -/
theorem average_words_per_hour 
(words_total : ℕ) 
(hours_total : ℕ) 
(increase : ℝ) :
  words_total = 50000 ∧ hours_total = 100 ∧ increase = 0.1 →
  (words_total / hours_total : ℝ) = 500 :=
by 
  intros h
  sorry

end average_words_per_hour_l82_8242


namespace root_of_quadratic_l82_8239

theorem root_of_quadratic (a b c : ℝ) :
  (4 * a + 2 * b + c = 0) ↔ (a * 2^2 + b * 2 + c = 0) :=
by
  sorry

end root_of_quadratic_l82_8239


namespace solve_inequality_find_m_range_l82_8205

noncomputable def f (x : ℝ) : ℝ := |x - 2|
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m

theorem solve_inequality (a : ℝ) : 
  ∀ x : ℝ, f x + a - 1 > 0 ↔ 
    (a = 1 ∧ x ≠ 2) ∨ 
    (a > 1) ∨ 
    (a < 1 ∧ (x > 3 - a ∨ x < a + 1)) :=
sorry

theorem find_m_range (m : ℝ) : 
  (∀ x : ℝ, f x > g x m) ↔ m < 5 :=
sorry

end solve_inequality_find_m_range_l82_8205


namespace find_m_for_one_real_solution_l82_8263

variables {m x : ℝ}

-- Given condition
def equation := (x + 4) * (x + 1) = m + 2 * x

-- The statement to prove
theorem find_m_for_one_real_solution : (∃ m : ℝ, m = 7 / 4 ∧ ∀ (x : ℝ), (x + 4) * (x + 1) = m + 2 * x) :=
by
  -- The proof starts here, which we will skip with sorry
  sorry

end find_m_for_one_real_solution_l82_8263


namespace aaron_walking_speed_l82_8298

-- Definitions of the conditions
def distance_jog : ℝ := 3 -- in miles
def speed_jog : ℝ := 2 -- in miles/hour
def total_time : ℝ := 3 -- in hours

-- The problem statement
theorem aaron_walking_speed :
  ∃ (v : ℝ), v = (distance_jog / (total_time - (distance_jog / speed_jog))) ∧ v = 2 :=
by
  sorry

end aaron_walking_speed_l82_8298


namespace nigella_base_salary_is_3000_l82_8290

noncomputable def nigella_base_salary : ℝ :=
  let house_A_cost := 60000
  let house_B_cost := 3 * house_A_cost
  let house_C_cost := (2 * house_A_cost) - 110000
  let commission_A := 0.02 * house_A_cost
  let commission_B := 0.02 * house_B_cost
  let commission_C := 0.02 * house_C_cost
  let total_earnings := 8000
  let total_commission := commission_A + commission_B + commission_C
  total_earnings - total_commission

theorem nigella_base_salary_is_3000 : 
  nigella_base_salary = 3000 :=
by sorry

end nigella_base_salary_is_3000_l82_8290


namespace coat_shirt_ratio_l82_8218

variable (P S C k : ℕ)

axiom h1 : P + S = 100
axiom h2 : P + C = 244
axiom h3 : C = k * S
axiom h4 : C = 180

theorem coat_shirt_ratio (P S C k : ℕ) (h1 : P + S = 100) (h2 : P + C = 244) (h3 : C = k * S) (h4 : C = 180) :
  C / S = 5 :=
sorry

end coat_shirt_ratio_l82_8218


namespace average_first_21_multiples_of_17_l82_8233

theorem average_first_21_multiples_of_17:
  let n := 21
  let a1 := 17
  let a21 := 17 * n
  let sum := n / 2 * (a1 + a21)
  (sum / n = 187) :=
by
  sorry

end average_first_21_multiples_of_17_l82_8233


namespace extra_marks_15_l82_8270

theorem extra_marks_15 {T P : ℝ} (h1 : 0.30 * T = P - 30) (h2 : 0.45 * T = P + 15) (h3 : P = 120) : 
  0.45 * T - P = 15 := 
by
  sorry

end extra_marks_15_l82_8270


namespace true_statements_count_l82_8269

def reciprocal (n : ℕ) : ℚ := 1 / n

theorem true_statements_count :
  let s1 := reciprocal 4 + reciprocal 8 = reciprocal 12
  let s2 := reciprocal 8 - reciprocal 5 = reciprocal 3
  let s3 := reciprocal 3 * reciprocal 9 = reciprocal 27
  let s4 := reciprocal 15 / reciprocal 3 = reciprocal 5
  (if s1 then 1 else 0) + 
  (if s2 then 1 else 0) + 
  (if s3 then 1 else 0) + 
  (if s4 then 1 else 0) = 2 :=
by
  sorry

end true_statements_count_l82_8269


namespace no_extreme_value_at_5_20_l82_8212

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 4 * x ^ 2 - k * x - 8

theorem no_extreme_value_at_5_20 (k : ℝ) :
  ¬ (∃ (c : ℝ), (forall (x : ℝ), f k x = f k c + (4 * (x - c) ^ 2 - 8 - 20)) ∧ c = 5) ↔ (k ≤ 40 ∨ k ≥ 160) := sorry

end no_extreme_value_at_5_20_l82_8212


namespace problem1_eval_problem2_eval_l82_8238

-- Problem 1
theorem problem1_eval :
  (1 : ℚ) * (-4.5) - (-5.6667) - (2.5) - 7.6667 = -9 := 
by
  sorry

-- Problem 2
theorem problem2_eval :
  (-(4^2) / (-2)^3) - ((4 / 9) * ((-3 / 2)^2)) = 1 := 
by
  sorry

end problem1_eval_problem2_eval_l82_8238


namespace value_of_a_l82_8249
noncomputable def find_a (a b c : ℝ) : ℝ :=
if 2 * b = a + c ∧ (a * c) * (b * c) = ((a * b) ^ 2) ∧ a + b + c = 6 then a else 0

theorem value_of_a (a b c : ℝ) :
  (2 * b = a + c) ∧ ((a * c) * (b * c) = (a * b) ^ 2) ∧ (a + b + c = 6) ∧ (a ≠ c) ∧ (a ≠ b) ∧ (b ≠ c) → a = 4 :=
by sorry

end value_of_a_l82_8249


namespace circle_tangent_to_x_axis_at_origin_l82_8278

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h_circle : ∀ x y : ℝ, x^2 + y^2 + Dx + Ey + F = 0)
  (h_tangent : ∃ x, x^2 + (0 : ℝ)^2 + Dx + E * 0 + F = 0 ∧ ∃ r : ℝ, ∀ x y, x^2 + (y - r)^2 = r^2) :
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 :=
by
  sorry

end circle_tangent_to_x_axis_at_origin_l82_8278


namespace krios_population_limit_l82_8297

theorem krios_population_limit (initial_population : ℕ) (acre_per_person : ℕ) (total_acres : ℕ) (doubling_years : ℕ) :
  initial_population = 150 →
  acre_per_person = 2 →
  total_acres = 35000 →
  doubling_years = 30 →
  ∃ (years_from_2005 : ℕ), years_from_2005 = 210 ∧ (initial_population * 2^(years_from_2005 / doubling_years)) ≥ total_acres / acre_per_person :=
by
  intros
  sorry

end krios_population_limit_l82_8297


namespace clock_correct_time_fraction_l82_8272

/-- 
  A 24-hour digital clock displays the hour and minute of a day, 
  counting from 00:00 to 23:59. However, due to a glitch, whenever 
  the clock is supposed to display a '2', it mistakenly displays a '5'.

  Prove that the fraction of a day during which the clock shows the correct 
  time is 23/40.
-/
theorem clock_correct_time_fraction :
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  (correct_hours / total_hours) * (correct_minutes / total_minutes) = 23 / 40 :=
by
  let total_hours := 24
  let affected_hours := 6
  let correct_hours := total_hours - affected_hours
  let total_minutes := 60
  let affected_minutes := 14
  let correct_minutes := total_minutes - affected_minutes
  have h1 : correct_hours = 18 := rfl
  have h2 : correct_minutes = 46 := rfl
  have h3 : 18 / 24 = 3 / 4 := by norm_num
  have h4 : 46 / 60 = 23 / 30 := by norm_num
  have h5 : (3 / 4) * (23 / 30) = 23 / 40 := by norm_num
  exact h5

end clock_correct_time_fraction_l82_8272


namespace quadratic_inequality_has_real_solution_l82_8293

-- Define the quadratic function and the inequality
def quadratic (a x : ℝ) : ℝ := x^2 - 8 * x + a
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, quadratic a x < 0

-- Define the condition for 'a' within the interval (0, 16)
def condition_on_a (a : ℝ) : Prop := 0 < a ∧ a < 16

-- The main statement to prove
theorem quadratic_inequality_has_real_solution (a : ℝ) (h : condition_on_a a) : quadratic_inequality a :=
sorry

end quadratic_inequality_has_real_solution_l82_8293


namespace find_a_b_l82_8281

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_a_b : 
  (∀ x : ℝ, f (g x a b) = 9 * x^2 + 6 * x + 1) ↔ ((a = 3 ∧ b = 1) ∨ (a = -3 ∧ b = -1)) :=
by
  sorry

end find_a_b_l82_8281


namespace investment_time_period_l82_8296

theorem investment_time_period :
  ∀ (A P : ℝ) (R : ℝ) (T : ℝ),
  A = 896 → P = 799.9999999999999 → R = 5 →
  (A - P) = (P * R * T / 100) → T = 2.4 :=
by
  intros A P R T hA hP hR hSI
  sorry

end investment_time_period_l82_8296


namespace real_roots_quadratic_l82_8273

theorem real_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 3 * x - 1 = 0) ↔ (m ≥ -5/4 ∧ m ≠ 1) := by
  sorry

end real_roots_quadratic_l82_8273


namespace smallest_positive_b_l82_8246

theorem smallest_positive_b (b : ℤ) :
  b % 5 = 1 ∧ b % 4 = 2 ∧ b % 7 = 3 → b = 86 :=
by
  sorry

end smallest_positive_b_l82_8246


namespace persons_attended_total_l82_8295

theorem persons_attended_total (p q : ℕ) (a : ℕ) (c : ℕ) (total_amount : ℕ) (adult_ticket : ℕ) (child_ticket : ℕ) 
  (h1 : adult_ticket = 60) (h2 : child_ticket = 25) (h3 : total_amount = 14000) 
  (h4 : a = 200) (h5 : p = a + c)
  (h6 : a * adult_ticket + c * child_ticket = total_amount):
  p = 280 :=
by
  sorry

end persons_attended_total_l82_8295


namespace product_of_consecutive_integers_even_l82_8260

theorem product_of_consecutive_integers_even (n : ℤ) : Even (n * (n + 1)) :=
sorry

end product_of_consecutive_integers_even_l82_8260


namespace mark_lloyd_ratio_l82_8202

theorem mark_lloyd_ratio (M L C : ℕ) (h1 : M = L) (h2 : M = C - 10) (h3 : C = 100) (h4 : M + L + C + 80 = 300) : M = L :=
by {
  sorry -- proof steps go here
}

end mark_lloyd_ratio_l82_8202


namespace actual_diameter_layer_3_is_20_micrometers_l82_8253

noncomputable def magnified_diameter_to_actual (magnified_diameter_cm : ℕ) (magnification_factor : ℕ) : ℕ :=
  (magnified_diameter_cm * 10000) / magnification_factor

def layer_3_magnified_diameter_cm : ℕ := 3
def layer_3_magnification_factor : ℕ := 1500

theorem actual_diameter_layer_3_is_20_micrometers :
  magnified_diameter_to_actual layer_3_magnified_diameter_cm layer_3_magnification_factor = 20 :=
by
  sorry

end actual_diameter_layer_3_is_20_micrometers_l82_8253


namespace value_of_F_l82_8247

theorem value_of_F (D E F : ℕ) (hD : D < 10) (hE : E < 10) (hF : F < 10)
    (h1 : (8 + 5 + D + 7 + 3 + E + 2) % 3 = 0)
    (h2 : (4 + 1 + 7 + D + E + 6 + F) % 3 = 0) : 
    F = 6 :=
by
  sorry

end value_of_F_l82_8247


namespace readers_all_three_l82_8255

def total_readers : ℕ := 500
def readers_science_fiction : ℕ := 320
def readers_literary_works : ℕ := 200
def readers_non_fiction : ℕ := 150
def readers_sf_and_lw : ℕ := 120
def readers_sf_and_nf : ℕ := 80
def readers_lw_and_nf : ℕ := 60

theorem readers_all_three :
  total_readers = readers_science_fiction + readers_literary_works + readers_non_fiction - (readers_sf_and_lw + readers_sf_and_nf + readers_lw_and_nf) + 90 :=
by
  sorry

end readers_all_three_l82_8255


namespace range_of_a_for_three_distinct_real_roots_l82_8243

theorem range_of_a_for_three_distinct_real_roots (a : ℝ) :
  (∃ (f : ℝ → ℝ), ∀ x, f x = x^3 - 3*x^2 - a ∧ ∃ r1 r2 r3 : ℝ, r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧ f r1 = 0 ∧ f r2 = 0 ∧ f r3 = 0) ↔ (-4 < a ∧ a < 0) :=
by
  sorry

end range_of_a_for_three_distinct_real_roots_l82_8243


namespace eq_solution_set_l82_8251

theorem eq_solution_set :
  {x : ℝ | (2 / (x + 2)) + (4 / (x + 8)) ≥ 3 / 4} = {x : ℝ | -2 < x ∧ x ≤ 2} :=
by {
  sorry
}

end eq_solution_set_l82_8251


namespace sum_of_solutions_eq_zero_l82_8299

noncomputable def f (x : ℝ) : ℝ := 2 ^ |x| + 5 * |x|

theorem sum_of_solutions_eq_zero (x : ℝ) (hx : f x = 28) :
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_eq_zero_l82_8299


namespace function_properties_and_k_range_l82_8206

theorem function_properties_and_k_range :
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 ^ x) ∧ (∀ y, y > 0)) ∧
  (∀ k : ℝ, (∃ t : ℝ, t > 0 ∧ (t^2 - 2*t + k = 0)) ↔ (0 < k ∧ k < 1)) :=
by sorry

end function_properties_and_k_range_l82_8206


namespace round_robin_matches_l82_8237

-- Define the number of players in the tournament
def numPlayers : ℕ := 10

-- Define a function to calculate the number of matches in a round-robin tournament
def calculateMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

-- Theorem statement to prove that the number of matches in a 10-person round-robin chess tournament is 45
theorem round_robin_matches : calculateMatches numPlayers = 45 := by
  sorry

end round_robin_matches_l82_8237


namespace quiz_scores_dropped_students_l82_8227

theorem quiz_scores_dropped_students (T S : ℝ) :
  T = 30 * 60.25 →
  T - S = 26 * 63.75 →
  S = 150 :=
by
  intros hT h_rem
  -- Additional steps would be implemented here.
  sorry

end quiz_scores_dropped_students_l82_8227


namespace find_f_neg_two_l82_8222

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x^2 - 1 else sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)

axiom f_odd : is_odd_function f
axiom f_pos : ∀ x, x > 0 → f x = x^2 - 1

theorem find_f_neg_two : f (-2) = -3 :=
by
  sorry

end find_f_neg_two_l82_8222


namespace period_of_f_l82_8254

noncomputable def f (x : ℝ) : ℝ := sorry

theorem period_of_f (a : ℝ) (h : a ≠ 0) (H : ∀ x : ℝ, f (x + a) = (1 + f x) / (1 - f x)) : 
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = 4 * |a| :=
by
  sorry

end period_of_f_l82_8254


namespace maximum_weekly_hours_l82_8230

-- Conditions
def regular_rate : ℝ := 8 -- $8 per hour for the first 20 hours
def overtime_rate : ℝ := regular_rate * 1.25 -- 25% higher than the regular rate
def max_weekly_earnings : ℝ := 460 -- Maximum of $460 in a week
def regular_hours : ℕ := 20 -- First 20 hours are regular hours
def regular_earnings : ℝ := regular_hours * regular_rate -- Earnings for regular hours
def max_overtime_earnings : ℝ := max_weekly_earnings - regular_earnings -- Maximum overtime earnings

-- Proof problem statement
theorem maximum_weekly_hours : regular_hours + (max_overtime_earnings / overtime_rate) = 50 := by
  sorry

end maximum_weekly_hours_l82_8230


namespace cos_double_angle_l82_8282

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 3/5) : Real.cos (2 * a) = 7/25 :=
by
  sorry

end cos_double_angle_l82_8282


namespace average_score_l82_8258

variable (score : Fin 5 → ℤ)
variable (actual_score : ℤ)
variable (rank : Fin 5)
variable (average : ℤ)

def students_scores_conditions := 
  score 0 = 10 ∧ score 1 = -5 ∧ score 2 = 0 ∧ score 3 = 8 ∧ score 4 = -3 ∧
  actual_score = 90 ∧ rank.val = 2

theorem average_score (h : students_scores_conditions score actual_score rank) :
  average = 92 :=
sorry

end average_score_l82_8258


namespace selling_price_of_book_l82_8220

theorem selling_price_of_book (cost_price : ℕ) (profit_rate : ℕ) (profit : ℕ) (selling_price : ℕ) :
  cost_price = 50 → profit_rate = 80 → profit = (profit_rate * cost_price) / 100 → selling_price = cost_price + profit → selling_price = 90 :=
by
  intros h_cost_price h_profit_rate h_profit h_selling_price
  rw [h_cost_price, h_profit_rate] at h_profit
  simp at h_profit
  rw [h_cost_price, h_profit] at h_selling_price
  exact h_selling_price

end selling_price_of_book_l82_8220


namespace area_of_gray_region_is_27pi_l82_8219

-- Define the conditions
def concentric_circles (inner_radius outer_radius : ℝ) :=
  2 * inner_radius = outer_radius

def width_of_gray_region (inner_radius outer_radius width : ℝ) :=
  width = outer_radius - inner_radius

-- Define the proof problem
theorem area_of_gray_region_is_27pi
(inner_radius outer_radius : ℝ) 
(h1 : concentric_circles inner_radius outer_radius)
(h2 : width_of_gray_region inner_radius outer_radius 3) :
π * outer_radius^2 - π * inner_radius^2 = 27 * π :=
by
  -- Proof goes here, but it is not required as per instructions
  sorry

end area_of_gray_region_is_27pi_l82_8219


namespace product_of_three_numbers_l82_8201

theorem product_of_three_numbers 
  (a b c : ℕ) 
  (h1 : a + b + c = 300) 
  (h2 : 9 * a = b - 11) 
  (h3 : 9 * a = c + 15) : 
  a * b * c = 319760 := 
  sorry

end product_of_three_numbers_l82_8201


namespace minimum_sum_of_box_dimensions_l82_8228

theorem minimum_sum_of_box_dimensions :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 ∧ 0 < a ∧ 0 < b ∧ 0 < c :=
sorry

end minimum_sum_of_box_dimensions_l82_8228


namespace rational_sum_of_squares_is_square_l82_8240

theorem rational_sum_of_squares_is_square (a b c : ℚ) :
  ∃ r : ℚ, r ^ 2 = (1 / (b - c) ^ 2 + 1 / (c - a) ^ 2 + 1 / (a - b) ^ 2) :=
by
  sorry

end rational_sum_of_squares_is_square_l82_8240


namespace parabola_intersects_x_axis_l82_8276

theorem parabola_intersects_x_axis {p q x₀ x₁ x₂ : ℝ} (h : ∀ (x : ℝ), x ^ 2 + p * x + q ≠ 0)
    (M_below_x_axis : x₀ ^ 2 + p * x₀ + q < 0)
    (M_at_1_neg2 : x₀ = 1 ∧ (1 ^ 2 + p * 1 + q = -2)) :
    (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₀ < x₁ → x₁ < x₂) ∧ x₁ = -1 ∧ x₂ = 2 ∨ x₁ = 0 ∧ x₂ = 3) :=
by
  sorry

end parabola_intersects_x_axis_l82_8276


namespace restroom_students_l82_8280

theorem restroom_students (R : ℕ) (h1 : 4 * 6 = 24) (h2 : (2/3 : ℚ) * 24 = 16)
  (h3 : 23 = 16 + (3 * R - 1) + R) : R = 2 :=
by
  sorry

end restroom_students_l82_8280


namespace algebra_expression_value_l82_8288

theorem algebra_expression_value (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 11) : a^2 - a * b + b^2 = 67 :=
by
  sorry

end algebra_expression_value_l82_8288


namespace fractionSpentOnMachinery_l82_8266

-- Given conditions
def companyCapital (C : ℝ) : Prop := 
  ∃ remainingCapital, remainingCapital = 0.675 * C ∧ 
  ∃ rawMaterial, rawMaterial = (1/4) * C ∧ 
  ∃ remainingAfterRaw, remainingAfterRaw = (3/4) * C ∧ 
  ∃ spentOnMachinery, spentOnMachinery = remainingAfterRaw - remainingCapital

-- Question translated to Lean statement
theorem fractionSpentOnMachinery (C : ℝ) (h : companyCapital C) : 
  ∃ remainingAfterRaw spentOnMachinery,
    spentOnMachinery / remainingAfterRaw = 1/10 :=
by 
  sorry

end fractionSpentOnMachinery_l82_8266


namespace six_digit_permutation_reverse_div_by_11_l82_8229

theorem six_digit_permutation_reverse_div_by_11 
  (a b c : ℕ)
  (h_a : 1 ≤ a ∧ a ≤ 9)
  (h_b : 0 ≤ b ∧ b ≤ 9)
  (h_c : 0 ≤ c ∧ c ≤ 9)
  (X : ℕ)
  (h_X : X = 100001 * a + 10010 * b + 1100 * c) :
  11 ∣ X :=
by 
  sorry

end six_digit_permutation_reverse_div_by_11_l82_8229


namespace total_amount_returned_l82_8234

noncomputable def continuous_compounding_interest : ℝ :=
  let P : ℝ := 325 / (Real.exp 0.12 - 1)
  let A1 : ℝ := P * Real.exp 0.04
  let A2 : ℝ := A1 * Real.exp 0.05
  let A3 : ℝ := A2 * Real.exp 0.03
  let total_interest : ℝ := 325
  let total_amount : ℝ := P + total_interest
  total_amount

theorem total_amount_returned :
  continuous_compounding_interest = 2874.02 :=
by
  sorry

end total_amount_returned_l82_8234


namespace sum_of_numbers_l82_8203

noncomputable def mean (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem sum_of_numbers (a b c : ℕ) (h1 : mean a b c = a + 8)
  (h2 : mean a b c = c - 20) (h3 : b = 7) (h_le1 : a ≤ b) (h_le2 : b ≤ c) :
  a + b + c = 57 :=
by {
  sorry
}

end sum_of_numbers_l82_8203


namespace find_y_value_l82_8287

theorem find_y_value 
  (k : ℝ) 
  (y : ℝ) 
  (hx81 : y = 3 * Real.sqrt 2)
  (h_eq : ∀ (x : ℝ), y = k * x ^ (1 / 4)) 
  : (∃ y, y = 2 ∧ y = k * 4 ^ (1 / 4))
:= sorry

end find_y_value_l82_8287


namespace exponent_multiplication_l82_8200

-- Define the core condition: the base 625
def base := 625

-- Define the exponents
def exp1 := 0.08
def exp2 := 0.17
def combined_exp := exp1 + exp2

-- The mathematical goal to prove
theorem exponent_multiplication (b : ℝ) (e1 e2 : ℝ) (h1 : b = 625) (h2 : e1 = 0.08) (h3 : e2 = 0.17) :
  (b ^ e1 * b ^ e2) = 5 :=
by {
  -- Sorry is added to skip the actual proof steps.
  sorry
}

end exponent_multiplication_l82_8200


namespace appropriate_sampling_method_l82_8224

-- Definitions and conditions
def total_products : ℕ := 40
def first_class_products : ℕ := 10
def second_class_products : ℕ := 25
def defective_products : ℕ := 5
def samples_needed : ℕ := 8

-- Theorem statement
theorem appropriate_sampling_method : 
  (first_class_products + second_class_products + defective_products = total_products) ∧ 
  (2 ≤ first_class_products ∧ 2 ≤ second_class_products ∧ 1 ≤ defective_products) → 
  "Stratified Sampling" = "The appropriate sampling method for quality analysis" :=
  sorry

end appropriate_sampling_method_l82_8224


namespace joan_gave_melanie_apples_l82_8204

theorem joan_gave_melanie_apples (original_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : original_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  sorry

end joan_gave_melanie_apples_l82_8204


namespace find_xy_l82_8291

noncomputable def star (a b c d : ℝ) : ℝ × ℝ :=
  (a * c + b * d, a * d + b * c)

theorem find_xy (a b x y : ℝ) (h : star a b x y = (a, b)) (h' : a^2 ≠ b^2) : (x, y) = (1, 0) :=
  sorry

end find_xy_l82_8291


namespace kiley_slices_eaten_l82_8283

def slices_of_cheesecake (total_calories_per_cheesecake calories_per_slice : ℕ) : ℕ :=
  total_calories_per_cheesecake / calories_per_slice

def slices_eaten (total_slices percentage_ate : ℚ) : ℚ :=
  total_slices * percentage_ate

theorem kiley_slices_eaten :
  ∀ (total_calories_per_cheesecake calories_per_slice : ℕ) (percentage_ate : ℚ),
  total_calories_per_cheesecake = 2800 →
  calories_per_slice = 350 →
  percentage_ate = (25 / 100 : ℚ) →
  slices_eaten (slices_of_cheesecake total_calories_per_cheesecake calories_per_slice) percentage_ate = 2 :=
by
  intros total_calories_per_cheesecake calories_per_slice percentage_ate h1 h2 h3
  rw [h1, h2, h3]
  sorry

end kiley_slices_eaten_l82_8283


namespace polynomial_coefficients_sum_l82_8279

theorem polynomial_coefficients_sum :
  let p := (5 * x^3 - 3 * x^2 + x - 8) * (8 - 3 * x)
  let a := -15
  let b := 49
  let c := -27
  let d := 32
  let e := -64
  16 * a + 8 * b + 4 * c + 2 * d + e = 44 := 
by
  sorry

end polynomial_coefficients_sum_l82_8279


namespace max_figures_in_grid_l82_8256

-- Definition of the grid size
def grid_size : ℕ := 9

-- Definition of the figure coverage
def figure_coverage : ℕ := 4

-- The total number of unit squares in the grid is 9 * 9 = 81
def total_unit_squares : ℕ := grid_size * grid_size

-- Each figure covers exactly 4 unit squares
def units_per_figure : ℕ := figure_coverage

-- The number of such 2x2 blocks that can be formed in 9x9 grid.
def maximal_figures_possible : ℕ := (grid_size / 2) * (grid_size / 2)

-- The main theorem to be proved
theorem max_figures_in_grid : 
  maximal_figures_possible = total_unit_squares / units_per_figure := by
  sorry

end max_figures_in_grid_l82_8256


namespace line_tangent_72_l82_8223

theorem line_tangent_72 (k : ℝ) : 4 * x + 6 * y + k = 0 → y^2 = 32 * x → (48^2 - 4 * (8 * k) = 0 ↔ k = 72) :=
by
  sorry

end line_tangent_72_l82_8223


namespace squares_with_center_25_60_l82_8216

theorem squares_with_center_25_60 :
  let center_x := 25
  let center_y := 60
  let non_neg_int_coords (x : ℤ) (y : ℤ) := x ≥ 0 ∧ y ≥ 0
  let is_center (x : ℤ) (y : ℤ) := x = center_x ∧ y = center_y
  let num_squares := 650
  ∃ n : ℤ, (n = num_squares) ∧ ∀ (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℤ), 
    non_neg_int_coords x₁ y₁ ∧ non_neg_int_coords x₂ y₂ ∧ 
    non_neg_int_coords x₃ y₃ ∧ non_neg_int_coords x₄ y₄ ∧ 
    is_center ((x₁ + x₂ + x₃ + x₄) / 4) ((y₁ + y₂ + y₃ + y₄) / 4) → 
    ∃ (k : ℤ), n = 650 :=
sorry

end squares_with_center_25_60_l82_8216


namespace eighteenth_entry_l82_8208

def r_8 (n : ℕ) : ℕ := n % 8

theorem eighteenth_entry (n : ℕ) (h : r_8 (3 * n) ≤ 3) : n = 17 :=
sorry

end eighteenth_entry_l82_8208


namespace mt_product_l82_8225

def g : ℝ → ℝ := sorry

axiom func_eqn (x y : ℝ) : g (x * g y + 2 * x) = 2 * x * y + g x

axiom g3_value : g 3 = 6

def m : ℕ := 1

def t : ℝ := 6

theorem mt_product : m * t = 6 :=
by 
  sorry

end mt_product_l82_8225


namespace find_y_value_l82_8277

theorem find_y_value (x y : ℝ) 
    (h1 : x^2 + 3 * x + 6 = y - 2) 
    (h2 : x = -5) : 
    y = 18 := 
  by 
  sorry

end find_y_value_l82_8277


namespace stops_time_proof_l82_8264

variable (departure_time arrival_time driving_time stop_time_in_minutes : ℕ)
variable (h_departure : departure_time = 7 * 60)
variable (h_arrival : arrival_time = 20 * 60)
variable (h_driving : driving_time = 12 * 60)
variable (total_minutes := arrival_time - departure_time)

theorem stops_time_proof :
  stop_time_in_minutes = (total_minutes - driving_time) := by
  sorry

end stops_time_proof_l82_8264


namespace enrique_shredder_Y_feeds_l82_8221

theorem enrique_shredder_Y_feeds :
  let typeB_contracts := 350
  let pages_per_TypeB := 10
  let shredderY_capacity := 8
  let total_pages_TypeB := typeB_contracts * pages_per_TypeB
  let feeds_ShredderY := (total_pages_TypeB + shredderY_capacity - 1) / shredderY_capacity
  feeds_ShredderY = 438 := sorry

end enrique_shredder_Y_feeds_l82_8221


namespace quadratic_has_one_solution_l82_8268

theorem quadratic_has_one_solution (k : ℝ) : (4 : ℝ) * (4 : ℝ) - k ^ 2 = 0 → k = 8 ∨ k = -8 := by
  sorry

end quadratic_has_one_solution_l82_8268


namespace first_class_rate_l82_8267

def pass_rate : ℝ := 0.95
def cond_first_class_rate : ℝ := 0.20

theorem first_class_rate :
  (pass_rate * cond_first_class_rate) = 0.19 :=
by
  -- The proof is omitted as we're not required to provide it.
  sorry

end first_class_rate_l82_8267


namespace age_difference_l82_8217

-- Defining the age variables as fractions
variables (x y : ℚ)

-- Given conditions
axiom ratio1 : 2 * x / y = 2 / y
axiom ratio2 : (5 * x + 20) / (y + 20) = 8 / 3

-- The main theorem to prove the difference between Mahesh's and Suresh's ages.
theorem age_difference : 5 * x - y = (125 / 8) := sorry

end age_difference_l82_8217


namespace p_correct_l82_8285

noncomputable def p : ℝ → ℝ := sorry

axiom p_at_3 : p 3 = 10

axiom p_condition (x y : ℝ) : p x * p y = p x + p y + p (x * y) - 2

theorem p_correct : ∀ x, p x = x^2 + 1 :=
sorry

end p_correct_l82_8285


namespace end_of_month_books_count_l82_8226

theorem end_of_month_books_count:
  ∀ (initial_books : ℝ) (loaned_out_books : ℝ) (return_rate : ℝ)
    (rounded_loaned_out_books : ℝ) (returned_books : ℝ)
    (not_returned_books : ℝ) (end_of_month_books : ℝ),
    initial_books = 75 →
    loaned_out_books = 60.00000000000001 →
    return_rate = 0.65 →
    rounded_loaned_out_books = 60 →
    returned_books = return_rate * rounded_loaned_out_books →
    not_returned_books = rounded_loaned_out_books - returned_books →
    end_of_month_books = initial_books - not_returned_books →
    end_of_month_books = 54 :=
by
  intros initial_books loaned_out_books return_rate
         rounded_loaned_out_books returned_books
         not_returned_books end_of_month_books
  intros h_initial_books h_loaned_out_books h_return_rate
         h_rounded_loaned_out_books h_returned_books
         h_not_returned_books h_end_of_month_books
  sorry

end end_of_month_books_count_l82_8226


namespace find_y_l82_8231

theorem find_y (x y : ℕ) (h1 : x % y = 7) (h2 : (x : ℚ) / y = 86.1) (h3 : Nat.Prime (x + y)) : y = 70 :=
sorry

end find_y_l82_8231


namespace smallest_multiple_of_7_greater_than_500_l82_8241

theorem smallest_multiple_of_7_greater_than_500 : ∃ n : ℤ, (∃ k : ℤ, n = 7 * k) ∧ n > 500 ∧ n = 504 := 
by
  sorry

end smallest_multiple_of_7_greater_than_500_l82_8241


namespace continuous_iff_integral_condition_l82_8232

open Real 

noncomputable section

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

def integral_condition (f : ℝ → ℝ) (a : ℝ) (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (∫ x in a..(a + a_seq n), f x) + (∫ x in (a - a_seq n)..a, f x) ≤ (a_seq n) / n

theorem continuous_iff_integral_condition (a : ℝ) (f : ℝ → ℝ)
  (h_nondec : is_non_decreasing f) :
  ContinuousAt f a ↔ ∃ (a_seq : ℕ → ℝ), (∀ n, 0 < a_seq n) ∧ integral_condition f a a_seq := sorry

end continuous_iff_integral_condition_l82_8232


namespace combined_frosting_rate_l82_8275

theorem combined_frosting_rate (time_Cagney time_Lacey total_time : ℕ) (Cagney_rate Lacey_rate : ℚ) :
  (time_Cagney = 20) →
  (time_Lacey = 30) →
  (total_time = 5 * 60) →
  (Cagney_rate = 1 / time_Cagney) →
  (Lacey_rate = 1 / time_Lacey) →
  ((Cagney_rate + Lacey_rate) * total_time) = 25 :=
by
  intros
  -- conditions are given and used in the statement.
  -- proof follows from these conditions. 
  sorry

end combined_frosting_rate_l82_8275


namespace foldable_shape_is_axisymmetric_l82_8284

def is_axisymmetric_shape (shape : Type) : Prop :=
  (∃ l : (shape → shape), (∀ x, l x = x))

theorem foldable_shape_is_axisymmetric (shape : Type) (l : shape → shape) 
  (h1 : ∀ x, l x = x) : is_axisymmetric_shape shape := by
  sorry

end foldable_shape_is_axisymmetric_l82_8284


namespace guppies_eaten_by_moray_eel_l82_8257

-- Definitions based on conditions
def moray_eel_guppies_per_day : ℕ := sorry -- Number of guppies the moray eel eats per day

def number_of_betta_fish : ℕ := 5

def guppies_per_betta : ℕ := 7

def total_guppies_needed_per_day : ℕ := 55

-- Theorem based on the question
theorem guppies_eaten_by_moray_eel :
  moray_eel_guppies_per_day = total_guppies_needed_per_day - (number_of_betta_fish * guppies_per_betta) :=
sorry

end guppies_eaten_by_moray_eel_l82_8257


namespace exists_two_points_same_color_l82_8259

theorem exists_two_points_same_color :
  ∀ (x : ℝ), ∀ (color : ℝ × ℝ → Prop),
  (∀ (p : ℝ × ℝ), color p = red ∨ color p = blue) →
  (∃ (p1 p2 : ℝ × ℝ), dist p1 p2 = x ∧ color p1 = color p2) :=
by
  intro x color color_prop
  sorry

end exists_two_points_same_color_l82_8259


namespace y_power_x_equals_49_l82_8215

theorem y_power_x_equals_49 (x y : ℝ) (h : |x - 2| = -(y + 7)^2) : y ^ x = 49 := by
  sorry

end y_power_x_equals_49_l82_8215


namespace fenced_area_with_cutout_l82_8294

def rectangle_area (length width : ℝ) : ℝ := length * width

def square_area (side : ℝ) : ℝ := side * side

theorem fenced_area_with_cutout :
  rectangle_area 20 18 - square_area 4 = 344 :=
by
  -- This is where the proof would go, but it is omitted as per instructions.
  sorry

end fenced_area_with_cutout_l82_8294


namespace smallest_expression_l82_8235

theorem smallest_expression (a b : ℝ) (h : b < 0) : a + b < a ∧ a < a - b :=
by
  sorry

end smallest_expression_l82_8235


namespace find_f_5_l82_8250

def f (x : ℝ) : ℝ := sorry -- we need to create a function under our condition

theorem find_f_5 : f 5 = 0 :=
sorry

end find_f_5_l82_8250


namespace chessboard_fraction_sum_l82_8261

theorem chessboard_fraction_sum (r s m n : ℕ) (h_r : r = 1296) (h_s : s = 204) (h_frac : (17 : ℚ) / 108 = (s : ℕ) / (r : ℕ)) : m + n = 125 :=
sorry

end chessboard_fraction_sum_l82_8261


namespace evaluation_at_x_4_l82_8236

noncomputable def simplified_expression (x : ℝ) :=
  (x - 1 - (3 / (x + 1))) / ((x^2 + 2 * x) / (x + 1))

theorem evaluation_at_x_4 : simplified_expression 4 = 1 / 2 :=
by
  sorry

end evaluation_at_x_4_l82_8236


namespace find_b_l82_8245

theorem find_b (b p : ℚ) :
  (∀ x : ℚ, (2 * x^3 + b * x + 7 = (x^2 + p * x + 1) * (2 * x + 7))) →
  b = -45 / 2 :=
sorry

end find_b_l82_8245


namespace venus_speed_mph_l82_8292

theorem venus_speed_mph (speed_mps : ℝ) (seconds_per_hour : ℝ) (mph : ℝ) 
  (h1 : speed_mps = 21.9) 
  (h2 : seconds_per_hour = 3600)
  (h3 : mph = speed_mps * seconds_per_hour) : 
  mph = 78840 := 
  by 
  sorry

end venus_speed_mph_l82_8292


namespace rationalize_sqrt_three_sub_one_l82_8209

theorem rationalize_sqrt_three_sub_one :
  (1 / (Real.sqrt 3 - 1)) = ((Real.sqrt 3 + 1) / 2) :=
by
  sorry

end rationalize_sqrt_three_sub_one_l82_8209


namespace cost_of_45_lilies_l82_8274

-- Defining the conditions
def price_per_lily (n : ℕ) : ℝ :=
  if n <= 30 then 2
  else 1.8

-- Stating the problem in Lean 4
theorem cost_of_45_lilies :
  price_per_lily 15 * 15 = 30 → (price_per_lily 45 * 45 = 81) :=
by
  intro h
  sorry

end cost_of_45_lilies_l82_8274


namespace complex_number_solution_l82_8248

open Complex

theorem complex_number_solution (z : ℂ) (h : z^2 = -99 - 40 * I) : z = 2 - 10 * I ∨ z = -2 + 10 * I :=
sorry

end complex_number_solution_l82_8248


namespace math_problem_l82_8289

theorem math_problem :
  (Real.pi - 3.14)^0 + Real.sqrt ((Real.sqrt 2 - 1)^2) = Real.sqrt 2 :=
by
  sorry

end math_problem_l82_8289


namespace first_day_of_month_l82_8213

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l82_8213


namespace area_of_region_inside_circle_outside_rectangle_l82_8286

theorem area_of_region_inside_circle_outside_rectangle
  (EF FH : ℝ)
  (hEF : EF = 6)
  (hFH : FH = 5)
  (r : ℝ)
  (h_radius : r = (EF^2 + FH^2).sqrt) :
  π * r^2 - EF * FH = 61 * π - 30 :=
by
  sorry

end area_of_region_inside_circle_outside_rectangle_l82_8286


namespace find_a1_l82_8252

open Nat

theorem find_a1 (a : ℕ → ℕ) (h1 : ∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n)
  (h2 : a 3 = 12) : a 1 = 3 :=
sorry

end find_a1_l82_8252


namespace bob_height_in_inches_l82_8244

theorem bob_height_in_inches (tree_height shadow_tree bob_shadow : ℝ)
  (h1 : tree_height = 50)
  (h2 : shadow_tree = 25)
  (h3 : bob_shadow = 6) :
  (12 * (tree_height / shadow_tree) * bob_shadow) = 144 :=
by sorry

end bob_height_in_inches_l82_8244


namespace boys_and_girls_in_class_l82_8214

theorem boys_and_girls_in_class (m d : ℕ)
  (A : (m - 1 = 10 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)) ∨ 
       (m - 1 = 14 - 4 ∧ (d - 1 = 14 ∨ d - 1 = 10 + 4 ∨ d - 1 = 10 - 4)))
  (B : (m - 1 = 13 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)) ∨ 
       (m - 1 = 11 - 4 ∧ (d - 1 = 11 ∨ d - 1 = 11 + 4 ∨ d - 1 = 11 - 4)))
  (C : (m - 1 = 13 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4)) ∨ 
       (m - 1 = 19 - 4 ∧ (d - 1 = 19 ∨ d - 1 = 19 + 4 ∨ d - 1 = 19 - 4))) : 
  m = 14 ∧ d = 15 := 
sorry

end boys_and_girls_in_class_l82_8214


namespace find_XY_in_306090_triangle_l82_8271

-- Definitions of the problem
def angleZ := 90
def angleX := 60
def hypotenuseXZ := 12
def isRightTriangle (XYZ : Type) (angleZ : ℕ) : Prop := angleZ = 90
def is306090Triangle (XYZ : Type) (angleX : ℕ) (angleZ : ℕ) : Prop := (angleX = 60) ∧ (angleZ = 90)

-- Lean theorem statement
theorem find_XY_in_306090_triangle 
  (XYZ : Type)
  (hypotenuseXZ : ℕ)
  (h1 : isRightTriangle XYZ angleZ)
  (h2 : is306090Triangle XYZ angleX angleZ) :
  XY = 8 := 
sorry

end find_XY_in_306090_triangle_l82_8271


namespace pages_already_read_l82_8262

theorem pages_already_read (total_pages : ℕ) (pages_left : ℕ) (h_total : total_pages = 563) (h_left : pages_left = 416) :
  total_pages - pages_left = 147 :=
by
  sorry

end pages_already_read_l82_8262


namespace range_of_m_l82_8207

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : 1 / x + 4 / y = 1) (H : x + y > m^2 + 8 * m) : -9 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l82_8207
