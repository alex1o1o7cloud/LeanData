import Mathlib

namespace NUMINAMATH_GPT_history_book_cost_l972_97277

def total_books : ℕ := 90
def cost_math_book : ℕ := 4
def total_price : ℕ := 397
def math_books_bought : ℕ := 53

theorem history_book_cost :
  ∃ (H : ℕ), H = (total_price - (math_books_bought * cost_math_book)) / (total_books - math_books_bought) ∧ H = 5 :=
by
  sorry

end NUMINAMATH_GPT_history_book_cost_l972_97277


namespace NUMINAMATH_GPT_contrapositive_question_l972_97210

theorem contrapositive_question (x : ℝ) :
  (x = 2 → x^2 - 3 * x + 2 = 0) ↔ (x^2 - 3 * x + 2 ≠ 0 → x ≠ 2) := 
sorry

end NUMINAMATH_GPT_contrapositive_question_l972_97210


namespace NUMINAMATH_GPT_range_of_a_l972_97231

noncomputable def f : ℝ → ℝ := sorry
variable (f_even : ∀ x : ℝ, f x = f (-x))
variable (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
variable (a : ℝ) (h : f a ≤ f 2)

theorem range_of_a (f_even : ∀ x : ℝ, f x = f (-x))
                   (f_increasing : ∀ x y : ℝ, x ≤ y ∧ y ≤ 0 → f x ≤ f y)
                   (h : f a ≤ f 2) :
                   a ≤ -2 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l972_97231


namespace NUMINAMATH_GPT_pool_depth_is_10_feet_l972_97235

-- Definitions based on conditions
def hoseRate := 60 -- cubic feet per minute
def poolWidth := 80 -- feet
def poolLength := 150 -- feet
def drainingTime := 2000 -- minutes

-- Proof goal: the depth of the pool is 10 feet
theorem pool_depth_is_10_feet :
  ∃ (depth : ℝ), depth = 10 ∧ (hoseRate * drainingTime) = (poolWidth * poolLength * depth) :=
by
  use 10
  sorry

end NUMINAMATH_GPT_pool_depth_is_10_feet_l972_97235


namespace NUMINAMATH_GPT_score_standard_deviation_l972_97270

theorem score_standard_deviation (mean std_dev : ℝ)
  (h1 : mean = 76)
  (h2 : mean - 2 * std_dev = 60) :
  100 = mean + 3 * std_dev :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_score_standard_deviation_l972_97270


namespace NUMINAMATH_GPT_initial_salty_cookies_l972_97223

theorem initial_salty_cookies
  (initial_sweet_cookies : ℕ) 
  (ate_sweet_cookies : ℕ) 
  (ate_salty_cookies : ℕ) 
  (ate_diff : ℕ) 
  (H1 : initial_sweet_cookies = 39)
  (H2 : ate_sweet_cookies = 32)
  (H3 : ate_salty_cookies = 23)
  (H4 : ate_diff = 9) :
  initial_sweet_cookies - ate_diff = 30 :=
by sorry

end NUMINAMATH_GPT_initial_salty_cookies_l972_97223


namespace NUMINAMATH_GPT_smallest_number_conditions_l972_97225

theorem smallest_number_conditions :
  ∃ m : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], m % k = 2) ∧ (m % 8 = 0) ∧ ( ∀ n : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], n % k = 2) ∧ (n % 8 = 0) → m ≤ n ) :=
sorry

end NUMINAMATH_GPT_smallest_number_conditions_l972_97225


namespace NUMINAMATH_GPT_tangent_length_external_tangent_length_internal_l972_97288

noncomputable def tangent_length_ext (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R + r) / R)

noncomputable def tangent_length_int (R r a : ℝ) (h : R > r) (hAB : AB = a) : ℝ :=
  a * Real.sqrt ((R - r) / R)

theorem tangent_length_external (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_ext R r a h hAB :=
sorry

theorem tangent_length_internal (R r a T : ℝ) (h : R > r) (hAB : AB = a) :
  T = tangent_length_int R r a h hAB :=
sorry

end NUMINAMATH_GPT_tangent_length_external_tangent_length_internal_l972_97288


namespace NUMINAMATH_GPT_total_amount_paid_l972_97296

-- Definitions based on the conditions in step a)
def ring_cost : ℕ := 24
def ring_quantity : ℕ := 2

-- Statement to prove that the total cost is $48.
theorem total_amount_paid : ring_quantity * ring_cost = 48 := 
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l972_97296


namespace NUMINAMATH_GPT_difference_in_sums_l972_97239

def sum_of_digits (n : ℕ) : ℕ := (toString n).foldl (λ acc c => acc + (c.toNat - '0'.toNat)) 0

def Petrov_numbers := List.range' 1 2014 |>.filter (λ n => n % 2 = 1)
def Vasechkin_numbers := List.range' 2 2012 |>.filter (λ n => n % 2 = 0)

def sum_of_digits_Petrov := (Petrov_numbers.map sum_of_digits).sum
def sum_of_digits_Vasechkin := (Vasechkin_numbers.map sum_of_digits).sum

theorem difference_in_sums : sum_of_digits_Petrov - sum_of_digits_Vasechkin = 1007 := by
  sorry

end NUMINAMATH_GPT_difference_in_sums_l972_97239


namespace NUMINAMATH_GPT_find_x_l972_97232

-- Define the operation "※" as given
def star (a b : ℕ) : ℚ := (a + 2 * b) / 3

-- Given that 6 ※ x = 22 / 3, prove that x = 8
theorem find_x : ∃ x : ℕ, star 6 x = 22 / 3 ↔ x = 8 :=
by
  sorry -- Proof not required

end NUMINAMATH_GPT_find_x_l972_97232


namespace NUMINAMATH_GPT_deepak_and_wife_meet_time_l972_97276

noncomputable def deepak_speed_kmph : ℝ := 20
noncomputable def wife_speed_kmph : ℝ := 12
noncomputable def track_circumference_m : ℝ := 1000

noncomputable def speed_to_m_per_min (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 60

noncomputable def deepak_speed_m_per_min : ℝ := speed_to_m_per_min deepak_speed_kmph
noncomputable def wife_speed_m_per_min : ℝ := speed_to_m_per_min wife_speed_kmph

noncomputable def combined_speed_m_per_min : ℝ :=
  deepak_speed_m_per_min + wife_speed_m_per_min

noncomputable def meeting_time_minutes : ℝ :=
  track_circumference_m / combined_speed_m_per_min

theorem deepak_and_wife_meet_time :
  abs (meeting_time_minutes - 1.875) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_deepak_and_wife_meet_time_l972_97276


namespace NUMINAMATH_GPT_larger_number_is_1617_l972_97294

-- Given conditions
variables (L S : ℤ)
axiom condition1 : L - S = 1515
axiom condition2 : L = 16 * S + 15

-- To prove
theorem larger_number_is_1617 : L = 1617 := by
  sorry

end NUMINAMATH_GPT_larger_number_is_1617_l972_97294


namespace NUMINAMATH_GPT_geom_sequence_product_l972_97243

theorem geom_sequence_product (q a1 : ℝ) (h1 : a1 * (a1 * q) * (a1 * q^2) = 3) (h2 : (a1 * q^9) * (a1 * q^10) * (a1 * q^11) = 24) :
  (a1 * q^12) * (a1 * q^13) * (a1 * q^14) = 48 :=
by
  sorry

end NUMINAMATH_GPT_geom_sequence_product_l972_97243


namespace NUMINAMATH_GPT_linear_combination_harmonic_l972_97282

-- Define the harmonic property for a function
def is_harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x, y) = (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1)) / 4

-- The main statement to be proven in Lean
theorem linear_combination_harmonic
  (f g : ℤ × ℤ → ℝ) (a b : ℝ) (hf : is_harmonic f) (hg : is_harmonic g) :
  is_harmonic (fun p => a * f p + b * g p) :=
by
  sorry

end NUMINAMATH_GPT_linear_combination_harmonic_l972_97282


namespace NUMINAMATH_GPT_car_service_month_l972_97230

-- Define the conditions
def first_service_month : ℕ := 3 -- Representing March as the 3rd month
def service_interval : ℕ := 7
def total_services : ℕ := 13

-- Define an auxiliary function to calculate months and reduce modulo 12
def nth_service_month (first_month : ℕ) (interval : ℕ) (n : ℕ) : ℕ :=
  (first_month + (interval * (n - 1))) % 12

-- The theorem statement
theorem car_service_month : nth_service_month first_service_month service_interval total_services = 3 :=
by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_car_service_month_l972_97230


namespace NUMINAMATH_GPT_find_a_l972_97201

theorem find_a (a b c : ℕ) (h₁ : a + b = c) (h₂ : b + 2 * c = 10) (h₃ : c = 4) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_l972_97201


namespace NUMINAMATH_GPT_find_perpendicular_line_l972_97253

-- Define the point P
structure Point where
  x : ℤ
  y : ℤ

-- Define the line
structure Line where
  a : ℤ
  b : ℤ
  c : ℤ

-- The given problem conditions
def P : Point := { x := -1, y := 3 }

def given_line : Line := { a := 1, b := -2, c := 3 }

def perpendicular_line (line : Line) (point : Point) : Line :=
  ⟨ -line.b, line.a, -(line.a * point.y - line.b * point.x) ⟩

-- Theorem statement to prove
theorem find_perpendicular_line :
  perpendicular_line given_line P = { a := 2, b := 1, c := -1 } :=
by
  sorry

end NUMINAMATH_GPT_find_perpendicular_line_l972_97253


namespace NUMINAMATH_GPT_result_after_subtraction_l972_97268

theorem result_after_subtraction (x : ℕ) (h : x = 125) : 2 * x - 138 = 112 :=
by
  sorry

end NUMINAMATH_GPT_result_after_subtraction_l972_97268


namespace NUMINAMATH_GPT_range_of_a_l972_97261

theorem range_of_a (x a : ℝ) (h1 : -2 < x) (h2 : x ≤ 1) (h3 : |x - 2| < a) : a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l972_97261


namespace NUMINAMATH_GPT_probability_red_or_black_probability_red_black_or_white_l972_97208

-- We define the probabilities of events A, B, and C
def P_A : ℚ := 5 / 12
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 6

-- Define the probability of event D for completeness
def P_D : ℚ := 1 / 12

-- 1. Statement for the probability of drawing a red or black ball (P(A ⋃ B))
theorem probability_red_or_black :
  (P_A + P_B = 3 / 4) :=
by
  sorry

-- 2. Statement for the probability of drawing a red, black, or white ball (P(A ⋃ B ⋃ C))
theorem probability_red_black_or_white :
  (P_A + P_B + P_C = 11 / 12) :=
by
  sorry

end NUMINAMATH_GPT_probability_red_or_black_probability_red_black_or_white_l972_97208


namespace NUMINAMATH_GPT_annual_raise_l972_97248

-- Definitions based on conditions
def new_hourly_rate := 20
def new_weekly_hours := 40
def old_hourly_rate := 16
def old_weekly_hours := 25
def weeks_in_year := 52

-- Statement of the theorem
theorem annual_raise (new_hourly_rate new_weekly_hours old_hourly_rate old_weekly_hours weeks_in_year : ℕ) : 
  new_hourly_rate * new_weekly_hours * weeks_in_year - old_hourly_rate * old_weekly_hours * weeks_in_year = 20800 := 
  sorry -- Proof is omitted

end NUMINAMATH_GPT_annual_raise_l972_97248


namespace NUMINAMATH_GPT_ac_work_time_l972_97274

theorem ac_work_time (W : ℝ) (a_work_rate : ℝ) (b_work_rate : ℝ) (bc_work_rate : ℝ) (t : ℝ) : 
  (a_work_rate = W / 4) ∧ 
  (b_work_rate = W / 12) ∧ 
  (bc_work_rate = W / 3) → 
  t = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ac_work_time_l972_97274


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l972_97206

-- Problem 1
theorem simplify_expr1 (x y : ℝ) : x^2 - 5 * y - 4 * x^2 + y - 1 = -3 * x^2 - 4 * y - 1 :=
by sorry

-- Problem 2
theorem simplify_expr2 (a b : ℝ) : 7 * a + 3 * (a - 3 * b) - 2 * (b - 3 * a) = 16 * a - 11 * b :=
by sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l972_97206


namespace NUMINAMATH_GPT_winning_candidate_votes_l972_97280

def total_votes : ℕ := 100000
def winning_percentage : ℚ := 42 / 100
def expected_votes : ℚ := 42000

theorem winning_candidate_votes : winning_percentage * total_votes = expected_votes := by
  sorry

end NUMINAMATH_GPT_winning_candidate_votes_l972_97280


namespace NUMINAMATH_GPT_sin_beta_value_l972_97292

theorem sin_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.cos α = 4 / 5) 
  (h4 : Real.cos (α + β) = 5 / 13) : 
  Real.sin β = 33 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_sin_beta_value_l972_97292


namespace NUMINAMATH_GPT_Hans_current_age_l972_97263

variable {H : ℕ} -- Hans' current age

-- Conditions
def Josiah_age (H : ℕ) := 3 * H
def Hans_age_in_3_years (H : ℕ) := H + 3
def Josiah_age_in_3_years (H : ℕ) := Josiah_age H + 3
def sum_of_ages_in_3_years (H : ℕ) := Hans_age_in_3_years H + Josiah_age_in_3_years H

-- Theorem to prove
theorem Hans_current_age : sum_of_ages_in_3_years H = 66 → H = 15 :=
by
  sorry

end NUMINAMATH_GPT_Hans_current_age_l972_97263


namespace NUMINAMATH_GPT_car_body_mass_l972_97264

theorem car_body_mass (m_model : ℕ) (scale : ℕ) : 
  m_model = 1 → scale = 11 → m_car = 1331 :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_car_body_mass_l972_97264


namespace NUMINAMATH_GPT_mike_seashells_l972_97269

theorem mike_seashells (initial total : ℕ) (h1 : initial = 79) (h2 : total = 142) :
    total - initial = 63 :=
by
  sorry

end NUMINAMATH_GPT_mike_seashells_l972_97269


namespace NUMINAMATH_GPT_quadratic_rewriting_l972_97207

theorem quadratic_rewriting (d e : ℤ) (f : ℤ) : 
  (16 * x^2 - 40 * x - 24) = (d * x + e)^2 + f → 
  d^2 = 16 → 
  2 * d * e = -40 → 
  d * e = -20 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_quadratic_rewriting_l972_97207


namespace NUMINAMATH_GPT_students_taking_either_not_both_l972_97262

theorem students_taking_either_not_both (students_both : ℕ) (students_physics : ℕ) (students_only_chemistry : ℕ) :
  students_both = 12 →
  students_physics = 22 →
  students_only_chemistry = 9 →
  students_physics - students_both + students_only_chemistry = 19 :=
by
  intros h_both h_physics h_chemistry
  rw [h_both, h_physics, h_chemistry]
  repeat { sorry }

end NUMINAMATH_GPT_students_taking_either_not_both_l972_97262


namespace NUMINAMATH_GPT_least_possible_area_of_square_l972_97237

theorem least_possible_area_of_square (s : ℝ) (h₁ : 4.5 ≤ s) (h₂ : s < 5.5) : 
  s * s ≥ 20.25 :=
sorry

end NUMINAMATH_GPT_least_possible_area_of_square_l972_97237


namespace NUMINAMATH_GPT_amount_subtracted_l972_97228

theorem amount_subtracted (N A : ℝ) (h1 : N = 100) (h2 : 0.80 * N - A = 60) : A = 20 :=
by 
  sorry

end NUMINAMATH_GPT_amount_subtracted_l972_97228


namespace NUMINAMATH_GPT_no_nat_exists_perfect_cubes_l972_97297

theorem no_nat_exists_perfect_cubes : ¬ ∃ n : ℕ, ∃ a b : ℤ, 2^(n + 1) - 1 = a^3 ∧ 2^(n - 1)*(2^n - 1) = b^3 := 
by
  sorry

end NUMINAMATH_GPT_no_nat_exists_perfect_cubes_l972_97297


namespace NUMINAMATH_GPT_ice_cream_bar_price_l972_97200

theorem ice_cream_bar_price 
  (num_bars num_sundaes : ℕ)
  (total_cost : ℝ)
  (sundae_price ice_cream_bar_price : ℝ)
  (h1 : num_bars = 125)
  (h2 : num_sundaes = 125)
  (h3 : total_cost = 250.00)
  (h4 : sundae_price = 1.40)
  (total_price_condition : num_bars * ice_cream_bar_price + num_sundaes * sundae_price = total_cost) :
  ice_cream_bar_price = 0.60 :=
sorry

end NUMINAMATH_GPT_ice_cream_bar_price_l972_97200


namespace NUMINAMATH_GPT_bus_total_people_l972_97281

def number_of_boys : ℕ := 50
def additional_girls (b : ℕ) : ℕ := (2 * b) / 5
def number_of_girls (b : ℕ) : ℕ := b + additional_girls b
def total_people (b g : ℕ) : ℕ := b + g + 3  -- adding 3 for the driver, assistant, and teacher

theorem bus_total_people : total_people number_of_boys (number_of_girls number_of_boys) = 123 :=
by
  sorry

end NUMINAMATH_GPT_bus_total_people_l972_97281


namespace NUMINAMATH_GPT_recycling_program_earnings_l972_97250

-- Define conditions
def signup_earning : ℝ := 5.00
def referral_earning_tier1 : ℝ := 8.00
def referral_earning_tier2 : ℝ := 1.50
def friend_earning_signup : ℝ := 5.00
def friend_earning_tier2 : ℝ := 2.00

def initial_friend_count : ℕ := 5
def initial_friend_tier1_referrals_day1 : ℕ := 3
def initial_friend_tier1_referrals_week : ℕ := 2

def additional_friend_count : ℕ := 2
def additional_friend_tier1_referrals : ℕ := 1

-- Calculate Katrina's total earnings
def katrina_earnings : ℝ :=
  signup_earning +
  (initial_friend_count * referral_earning_tier1) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * referral_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * referral_earning_tier2) +
  (additional_friend_count * referral_earning_tier1) +
  (additional_friend_count * additional_friend_tier1_referrals * referral_earning_tier2)

-- Calculate friends' total earnings
def friends_earnings : ℝ :=
  (initial_friend_count * friend_earning_signup) +
  (initial_friend_count * initial_friend_tier1_referrals_day1 * friend_earning_tier2) +
  (initial_friend_count * initial_friend_tier1_referrals_week * friend_earning_tier2) +
  (additional_friend_count * friend_earning_signup) +
  (additional_friend_count * additional_friend_tier1_referrals * friend_earning_tier2)

-- Calculate combined total earnings
def combined_earnings : ℝ := katrina_earnings + friends_earnings

-- The proof assertion
theorem recycling_program_earnings : combined_earnings = 190.50 :=
by sorry

end NUMINAMATH_GPT_recycling_program_earnings_l972_97250


namespace NUMINAMATH_GPT_problem_f_increasing_l972_97293

theorem problem_f_increasing (a : ℝ) 
  (h1 : ∀ x, 2 ≤ x → 0 < x^2 - a * x + 3 * a) 
  (h2 : ∀ x, 2 ≤ x → 0 ≤ 2 * x - a) : 
  -4 < a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_GPT_problem_f_increasing_l972_97293


namespace NUMINAMATH_GPT_range_of_a_l972_97218

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → (x^2 - 2 * a * x + 2) ≥ a) ↔ (-3 ≤ a ∧ a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l972_97218


namespace NUMINAMATH_GPT_vector_addition_l972_97284

def v1 : ℝ × ℝ := (3, -6)
def v2 : ℝ × ℝ := (2, -9)
def v3 : ℝ × ℝ := (-1, 3)
def c1 : ℝ := 4
def c2 : ℝ := 5
def result : ℝ × ℝ := (23, -72)

theorem vector_addition :
  c1 • v1 + c2 • v2 - v3 = result :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_l972_97284


namespace NUMINAMATH_GPT_consecutive_int_sqrt_l972_97241

theorem consecutive_int_sqrt (m n : ℤ) (h1 : m < n) (h2 : m < Real.sqrt 13) (h3 : Real.sqrt 13 < n) (h4 : n = m + 1) : m * n = 12 :=
sorry

end NUMINAMATH_GPT_consecutive_int_sqrt_l972_97241


namespace NUMINAMATH_GPT_num_distinct_x_intercepts_l972_97202

def f (x : ℝ) : ℝ := (x - 5) * (x^3 + 5*x^2 + 9*x + 9)

theorem num_distinct_x_intercepts : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 = 0 ∧ f x2 = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x1 ∨ x = x2) :=
sorry

end NUMINAMATH_GPT_num_distinct_x_intercepts_l972_97202


namespace NUMINAMATH_GPT_sculpture_paint_area_correct_l972_97246

def sculpture_exposed_area (edge_length : ℝ) (num_cubes_layer1 : ℕ) (num_cubes_layer2 : ℕ) (num_cubes_layer3 : ℕ) : ℝ :=
  let area_top_layer1 := num_cubes_layer1 * edge_length ^ 2
  let area_side_layer1 := 8 * 3 * edge_length ^ 2
  let area_top_layer2 := num_cubes_layer2 * edge_length ^ 2
  let area_side_layer2 := 10 * edge_length ^ 2
  let area_top_layer3 := num_cubes_layer3 * edge_length ^ 2
  let area_side_layer3 := num_cubes_layer3 * 4 * edge_length ^ 2
  area_top_layer1 + area_side_layer1 + area_top_layer2 + area_side_layer2 + area_top_layer3 + area_side_layer3

theorem sculpture_paint_area_correct :
  sculpture_exposed_area 1 12 6 2 = 62 := by
  sorry

end NUMINAMATH_GPT_sculpture_paint_area_correct_l972_97246


namespace NUMINAMATH_GPT_diagonal_lt_half_perimeter_l972_97242

theorem diagonal_lt_half_perimeter (AB BC CD DA AC : ℝ) (h1 : AB > 0) (h2 : BC > 0) (h3 : CD > 0) (h4 : DA > 0) 
  (h_triangle1 : AC < AB + BC) (h_triangle2 : AC < AD + DC) :
  AC < (AB + BC + CD + DA) / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_diagonal_lt_half_perimeter_l972_97242


namespace NUMINAMATH_GPT_num_solutions_l972_97272

-- Define the problem and the condition
def matrix_eq (x : ℝ) : Prop :=
  3 * x^2 - 4 * x = 7

-- Define the main theorem to prove the number of solutions
theorem num_solutions : ∃! x : ℝ, matrix_eq x :=
sorry

end NUMINAMATH_GPT_num_solutions_l972_97272


namespace NUMINAMATH_GPT_compare_y_values_l972_97254

variable (a : ℝ) (y₁ y₂ : ℝ)
variable (h : a > 0)
variable (p1 : y₁ = a * (-1 : ℝ)^2 - 4 * a * (-1 : ℝ) + 2)
variable (p2 : y₂ = a * (1 : ℝ)^2 - 4 * a * (1 : ℝ) + 2)

theorem compare_y_values : y₁ > y₂ :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_y_values_l972_97254


namespace NUMINAMATH_GPT_triangle_weight_l972_97290

variables (S C T : ℕ)

def scale1 := (S + C = 8)
def scale2 := (S + 2 * C = 11)
def scale3 := (C + 2 * T = 15)

theorem triangle_weight (h1 : scale1 S C) (h2 : scale2 S C) (h3 : scale3 C T) : T = 6 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_weight_l972_97290


namespace NUMINAMATH_GPT_distance_MF_l972_97291

-- Define the conditions for the problem
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

def focus : (ℝ × ℝ) := (2, 0)

def lies_on_parabola (M : ℝ × ℝ) : Prop :=
  parabola M.1 M.2

def distance_to_line (M : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  abs (M.1 - line_x)

def point_M_conditions (M : ℝ × ℝ) : Prop :=
  distance_to_line M (-3) = 6 ∧ lies_on_parabola M

-- The final proof problem statement in Lean
theorem distance_MF (M : ℝ × ℝ) (h : point_M_conditions M) : dist M focus = 5 :=
by sorry

end NUMINAMATH_GPT_distance_MF_l972_97291


namespace NUMINAMATH_GPT_solve_system_correct_l972_97285

noncomputable def solve_system (n : ℕ) (x : ℕ → ℝ) : Prop :=
  n > 2 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k + x (k + 1) = x (k + 2) ^ 2) ∧ 
  x (n + 1) = x 1 ∧ x (n + 2) = x 2 →
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → x i = 2

theorem solve_system_correct (n : ℕ) (x : ℕ → ℝ) : solve_system n x := 
sorry

end NUMINAMATH_GPT_solve_system_correct_l972_97285


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l972_97279

theorem solve_arithmetic_sequence :
  ∃ x > 0, (x * x = (4 + 25) / 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l972_97279


namespace NUMINAMATH_GPT_money_sister_gave_l972_97249

theorem money_sister_gave (months_saved : ℕ) (savings_per_month : ℕ) (total_paid : ℕ) 
  (h1 : months_saved = 3) 
  (h2 : savings_per_month = 70) 
  (h3 : total_paid = 260) : 
  (total_paid - (months_saved * savings_per_month) = 50) :=
by {
  sorry
}

end NUMINAMATH_GPT_money_sister_gave_l972_97249


namespace NUMINAMATH_GPT_walking_rate_ratio_l972_97213

variables (R R' : ℝ)

theorem walking_rate_ratio (h₁ : R * 21 = R' * 18) : R' / R = 7 / 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_walking_rate_ratio_l972_97213


namespace NUMINAMATH_GPT_quadratic_roots_integer_sum_eq_198_l972_97267

theorem quadratic_roots_integer_sum_eq_198 (x p q x1 x2 : ℤ) 
  (h_eqn : x^2 + p * x + q = 0)
  (h_roots : (x - x1) * (x - x2) = 0)
  (h_pq_sum : p + q = 198) :
  (x1 = 2 ∧ x2 = 200) ∨ (x1 = 0 ∧ x2 = -198) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_integer_sum_eq_198_l972_97267


namespace NUMINAMATH_GPT_max_elements_in_S_l972_97251

theorem max_elements_in_S : ∀ (S : Finset ℕ), 
  (∀ (a b : ℕ), a ∈ S → b ∈ S → 
    (∃ c ∈ S, Nat.Coprime c a ∧ Nat.Coprime c b) ∧
    (∃ d ∈ S, ∃ x y : ℕ, x ∣ a ∧ x ∣ b ∧ x ∣ d ∧ y ∣ a ∧ y ∣ b ∧ y ∣ d)) →
  S.card ≤ 72 :=
by sorry

end NUMINAMATH_GPT_max_elements_in_S_l972_97251


namespace NUMINAMATH_GPT_problem_1_problem_2_l972_97286

-- Problem (1)
theorem problem_1 (x a : ℝ) (h_a : a = 1) (hP : x^2 - 4*a*x + 3*a^2 < 0) (hQ1 : x^2 - x - 6 ≤ 0) (hQ2 : x^2 + 2*x - 8 > 0) :
  2 < x ∧ x < 3 := sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (h_a_pos : 0 < a) (h_suff_neccess : (¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) → ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) ∧
                   ¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) ≠ ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) :
  1 < a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l972_97286


namespace NUMINAMATH_GPT_maximum_n_for_dart_probability_l972_97258

theorem maximum_n_for_dart_probability (n : ℕ) (h : n ≥ 1) :
  (∃ r : ℝ, r = 1 ∧
  ∃ A_square A_circles : ℝ, A_square = n^2 ∧ A_circles = n * π * r^2 ∧
  (A_circles / A_square) ≥ 1 / 2) → n ≤ 6 := by
  sorry

end NUMINAMATH_GPT_maximum_n_for_dart_probability_l972_97258


namespace NUMINAMATH_GPT_maurice_rides_before_visit_l972_97204

-- Defining all conditions in Lean
variables
  (M : ℕ) -- Number of times Maurice had been horseback riding before visiting Matt
  (Matt_rides_with_M : ℕ := 8 * 2) -- Number of times Matt rode with Maurice (8 times, 2 horses each time)
  (Matt_rides_alone : ℕ := 16) -- Number of times Matt rode solo
  (total_Matt_rides : ℕ := Matt_rides_with_M + Matt_rides_alone) -- Total rides by Matt
  (three_times_M : ℕ := 3 * M) -- Three times the number of times Maurice rode before visiting
  (unique_horses_M : ℕ := 8) -- Total number of unique horses Maurice rode during his visit

-- Main theorem
theorem maurice_rides_before_visit  
  (h1: total_Matt_rides = three_times_M) 
  (h2: unique_horses_M = M) 
  : M = 10 := sorry

end NUMINAMATH_GPT_maurice_rides_before_visit_l972_97204


namespace NUMINAMATH_GPT_tank_never_fills_l972_97259

structure Pipe :=
(rate1 : ℕ) (rate2 : ℕ)

def net_flow (pA pB pC pD : Pipe) (time1 time2 : ℕ) : ℤ :=
  let fillA := pA.rate1 * time1 + pA.rate2 * time2
  let fillB := pB.rate1 * time1 + pB.rate2 * time2
  let drainC := pC.rate1 * time1 + pC.rate2 * time2
  let drainD := pD.rate1 * (time1 + time2)
  (fillA + fillB) - (drainC + drainD)

theorem tank_never_fills (pA pB pC pD : Pipe) (time1 time2 : ℕ)
  (hA : pA = Pipe.mk 40 20) (hB : pB = Pipe.mk 20 40) 
  (hC : pC = Pipe.mk 20 40) (hD : pD = Pipe.mk 30 30) 
  (hTime : time1 = 30 ∧ time2 = 30): 
  net_flow pA pB pC pD time1 time2 = 0 := by
  sorry

end NUMINAMATH_GPT_tank_never_fills_l972_97259


namespace NUMINAMATH_GPT_square_area_l972_97299

theorem square_area (p : ℝ → ℝ) (a b : ℝ) (h₁ : ∀ x, p x = x^2 + 3 * x + 2) (h₂ : p a = 5) (h₃ : p b = 5) (h₄ : a ≠ b) : (b - a)^2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_square_area_l972_97299


namespace NUMINAMATH_GPT_number_of_blue_faces_l972_97265

theorem number_of_blue_faces (n : ℕ) (h : (6 * n^2) / (6 * n^3) = 1 / 3) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_blue_faces_l972_97265


namespace NUMINAMATH_GPT_percentage_of_men_not_speaking_french_or_spanish_l972_97247

theorem percentage_of_men_not_speaking_french_or_spanish 
  (total_employees : ℕ) 
  (men_percent women_percent : ℝ)
  (men_french percent men_spanish_percent men_other_percent : ℝ)
  (women_french_percent women_spanish_percent women_other_percent : ℝ)
  (h1 : men_percent = 60)
  (h2 : women_percent = 40)
  (h3 : men_french_percent = 55)
  (h4 : men_spanish_percent = 35)
  (h5 : men_other_percent = 10)
  (h6 : women_french_percent = 45)
  (h7 : women_spanish_percent = 25)
  (h8 : women_other_percent = 30) :
  men_other_percent = 10 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_men_not_speaking_french_or_spanish_l972_97247


namespace NUMINAMATH_GPT_choose_starters_l972_97222

theorem choose_starters :
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  Nat.choose totalPlayers 6 - Nat.choose playersExcludingTwins 6 = 5005 :=
by
  let totalPlayers := 16
  let numberOfTwins := 2
  let playersExcludingTwins := totalPlayers - numberOfTwins
  sorry

end NUMINAMATH_GPT_choose_starters_l972_97222


namespace NUMINAMATH_GPT_greatest_integer_l972_97287

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end NUMINAMATH_GPT_greatest_integer_l972_97287


namespace NUMINAMATH_GPT_eq_has_unique_solution_l972_97211

theorem eq_has_unique_solution : 
  ∃! x : ℝ, (x ≠ 0)
    ∧ ((x < 0 → false) ∧ 
      (x > 0 → (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9)) :=
by sorry

end NUMINAMATH_GPT_eq_has_unique_solution_l972_97211


namespace NUMINAMATH_GPT_num_trailing_zeroes_500_factorial_l972_97220

-- Define the function to count factors of a prime p in n!
def count_factors_in_factorial (n p : ℕ) : ℕ :=
  if p = 0 then 0 else
    (n / p) + (n / (p ^ 2)) + (n / (p ^ 3)) + (n / (p ^ 4))

theorem num_trailing_zeroes_500_factorial : 
  count_factors_in_factorial 500 5 = 124 :=
sorry

end NUMINAMATH_GPT_num_trailing_zeroes_500_factorial_l972_97220


namespace NUMINAMATH_GPT_snail_returns_to_starting_point_l972_97278

-- Define the variables and conditions
variables (a1 a2 b1 b2 : ℕ)

-- Prove that snail can return to starting point after whole number of hours
theorem snail_returns_to_starting_point (h1 : a1 = a2) (h2 : b1 = b2) : (a1 + b1 : ℕ) = (a1 + b1 : ℕ) :=
by sorry

end NUMINAMATH_GPT_snail_returns_to_starting_point_l972_97278


namespace NUMINAMATH_GPT_numerator_equals_denominator_l972_97229

theorem numerator_equals_denominator (x : ℝ) (h : 4 * x - 3 = 5 * x + 2) : x = -5 :=
  by
    sorry

end NUMINAMATH_GPT_numerator_equals_denominator_l972_97229


namespace NUMINAMATH_GPT_common_root_quadratic_l972_97289

theorem common_root_quadratic (a x1: ℝ) :
  (x1^2 + a * x1 + 1 = 0) ∧ (x1^2 + x1 + a = 0) ↔ a = -2 :=
sorry

end NUMINAMATH_GPT_common_root_quadratic_l972_97289


namespace NUMINAMATH_GPT_relatively_prime_positive_integers_l972_97205

theorem relatively_prime_positive_integers (a b : ℕ) (h1 : a > b) (h2 : gcd a b = 1) (h3 : (a^3 - b^3) / (a - b)^3 = 91 / 7) : a - b = 1 := 
by 
  sorry

end NUMINAMATH_GPT_relatively_prime_positive_integers_l972_97205


namespace NUMINAMATH_GPT_race_distance_correct_l972_97226

noncomputable def solve_race_distance : ℝ :=
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs

  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  Dp

theorem race_distance_correct :
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs
  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  time_p = time_q := by
  sorry

end NUMINAMATH_GPT_race_distance_correct_l972_97226


namespace NUMINAMATH_GPT_domain_sqrt_tan_x_sub_sqrt3_l972_97227

open Real

noncomputable def domain := {x : ℝ | ∃ k : ℤ, k * π + π / 3 ≤ x ∧ x < k * π + π / 2}

theorem domain_sqrt_tan_x_sub_sqrt3 :
  {x | ∃ y : ℝ, y = sqrt (tan x - sqrt 3)} = domain :=
by
  sorry

end NUMINAMATH_GPT_domain_sqrt_tan_x_sub_sqrt3_l972_97227


namespace NUMINAMATH_GPT_trajectory_midpoint_eq_C2_length_CD_l972_97298

theorem trajectory_midpoint_eq_C2 {x y x' y' : ℝ} :
  (x' - 0)^2 + (y' - 4)^2 = 16 →
  x = (x' + 4) / 2 →
  y = y' / 2 →
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  sorry

theorem length_CD {x y Cx Cy Dx Dy : ℝ} :
  ((x - 2)^2 + (y - 2)^2 = 4) →
  (x^2 + (y - 4)^2 = 16) →
  ((Cx - Dx)^2 + (Cy - Dy)^2 = 14) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_midpoint_eq_C2_length_CD_l972_97298


namespace NUMINAMATH_GPT_number_of_cars_l972_97224

variable (C B : ℕ)

-- Define the conditions
def number_of_bikes : Prop := B = 2
def total_number_of_wheels : Prop := 4 * C + 2 * B = 44

-- State the theorem
theorem number_of_cars (hB : number_of_bikes B) (hW : total_number_of_wheels C B) : C = 10 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_cars_l972_97224


namespace NUMINAMATH_GPT_line_passes_through_first_and_fourth_quadrants_l972_97214

theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) (H : b * k < 0) :
  (∃x₁, k * x₁ + b > 0) ∧ (∃x₂, k * x₂ + b < 0) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_first_and_fourth_quadrants_l972_97214


namespace NUMINAMATH_GPT_sequence_converges_l972_97221

theorem sequence_converges (a : ℕ → ℝ) (h_nonneg : ∀ n, 0 ≤ a n) (h_condition : ∀ m n, a (n + m) ≤ a n * a m) : 
    ∃ l : ℝ, ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |(a n)^ (1/n) - l| < ε :=
by
  sorry

end NUMINAMATH_GPT_sequence_converges_l972_97221


namespace NUMINAMATH_GPT_allocation_schemes_correct_l972_97209

def numWaysToAllocateVolunteers : ℕ :=
  let n := 5 -- number of volunteers
  let m := 4 -- number of events
  Nat.choose n 2 * Nat.factorial m

theorem allocation_schemes_correct :
  numWaysToAllocateVolunteers = 240 :=
by
  sorry

end NUMINAMATH_GPT_allocation_schemes_correct_l972_97209


namespace NUMINAMATH_GPT_expression_evaluation_correct_l972_97283

theorem expression_evaluation_correct (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  ( ( ( (x - 2) ^ 2 * (x ^ 2 + x + 1) ^ 2 ) / (x ^ 3 - 1) ^ 2 ) ^ 2 *
    ( ( (x + 2) ^ 2 * (x ^ 2 - x + 1) ^ 2 ) / (x ^ 3 + 1) ^ 2 ) ^ 2 ) 
  = (x^2 - 4)^4 := 
sorry

end NUMINAMATH_GPT_expression_evaluation_correct_l972_97283


namespace NUMINAMATH_GPT_evaluate_expression_l972_97245

theorem evaluate_expression (a b c : ℝ) (h1 : a = 4) (h2 : b = -4) (h3 : c = 3) : (3 / (a + b + c) = 1) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l972_97245


namespace NUMINAMATH_GPT_simplify_expression_l972_97219

theorem simplify_expression (x : ℕ) : (5 * x^4)^3 = 125 * x^(12) := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l972_97219


namespace NUMINAMATH_GPT_boxes_containing_neither_l972_97257

-- Define the conditions
def total_boxes : ℕ := 15
def boxes_with_pencils : ℕ := 8
def boxes_with_pens : ℕ := 5
def boxes_with_markers : ℕ := 3
def boxes_with_pencils_and_pens : ℕ := 2
def boxes_with_pencils_and_markers : ℕ := 1
def boxes_with_pens_and_markers : ℕ := 1
def boxes_with_all_three : ℕ := 0

-- The proof problem
theorem boxes_containing_neither (h: total_boxes = 15) : 
  total_boxes - ((boxes_with_pencils - boxes_with_pencils_and_pens - boxes_with_pencils_and_markers) + 
  (boxes_with_pens - boxes_with_pencils_and_pens - boxes_with_pens_and_markers) + 
  (boxes_with_markers - boxes_with_pencils_and_markers - boxes_with_pens_and_markers) + 
  boxes_with_pencils_and_pens + boxes_with_pencils_and_markers + boxes_with_pens_and_markers) = 3 := 
by
  -- Specify that we want to use the equality of the number of boxes
  sorry

end NUMINAMATH_GPT_boxes_containing_neither_l972_97257


namespace NUMINAMATH_GPT_greater_number_l972_97203

theorem greater_number (x y : ℕ) (h_sum : x + y = 50) (h_diff : x - y = 16) : x = 33 :=
by
  sorry

end NUMINAMATH_GPT_greater_number_l972_97203


namespace NUMINAMATH_GPT_hannah_bananas_l972_97260

theorem hannah_bananas (B : ℕ) (h1 : B / 4 = 15 / 3) : B = 20 :=
by
  sorry

end NUMINAMATH_GPT_hannah_bananas_l972_97260


namespace NUMINAMATH_GPT_shaded_region_area_computed_correctly_l972_97271

noncomputable def side_length : ℝ := 15
noncomputable def quarter_circle_radius : ℝ := side_length / 3
noncomputable def square_area : ℝ := side_length ^ 2
noncomputable def circle_area : ℝ := Real.pi * (quarter_circle_radius ^ 2)
noncomputable def shaded_region_area : ℝ := square_area - circle_area

theorem shaded_region_area_computed_correctly : 
  shaded_region_area = 225 - 25 * Real.pi := 
by 
  -- This statement only defines the proof problem.
  sorry

end NUMINAMATH_GPT_shaded_region_area_computed_correctly_l972_97271


namespace NUMINAMATH_GPT_buses_dispatched_theorem_l972_97255

-- Define the conditions and parameters
def buses_dispatched (buses: ℕ) (hours: ℕ) : ℕ :=
  buses * hours

-- Define the specific problem
noncomputable def buses_from_6am_to_4pm : ℕ :=
  let buses_per_hour := 5 / 2
  let hours         := 16 - 6
  buses_dispatched (buses_per_hour : ℕ) hours

-- State the theorem that needs to be proven
theorem buses_dispatched_theorem : buses_from_6am_to_4pm = 25 := 
by {
  -- This 'sorry' is a placeholder for the actual proof.
  sorry
}

end NUMINAMATH_GPT_buses_dispatched_theorem_l972_97255


namespace NUMINAMATH_GPT_fish_population_estimate_l972_97234

theorem fish_population_estimate
  (N : ℕ) 
  (tagged_initial : ℕ)
  (caught_again : ℕ)
  (tagged_again : ℕ)
  (h1 : tagged_initial = 60)
  (h2 : caught_again = 60)
  (h3 : tagged_again = 2)
  (h4 : (tagged_initial : ℚ) / N = (tagged_again : ℚ) / caught_again) :
  N = 1800 :=
by
  sorry

end NUMINAMATH_GPT_fish_population_estimate_l972_97234


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l972_97216

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop := ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of the sum of the first n terms
def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

-- Problem statement in Lean 4
theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (S : ℕ → ℤ)
  (h1 : arithmetic_sequence a)
  (h2 : sum_first_n_terms a S)
  (h3 : S 9 = a 4 + a 5 + a 6 + 66) :
  a 2 + a 8 = 22 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l972_97216


namespace NUMINAMATH_GPT_range_of_m_for_inequality_l972_97256

theorem range_of_m_for_inequality (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 := 
sorry

end NUMINAMATH_GPT_range_of_m_for_inequality_l972_97256


namespace NUMINAMATH_GPT_jog_to_coffee_shop_l972_97238

def constant_pace_jogging (time_to_park : ℕ) (dist_to_park : ℝ) (dist_to_coffee_shop : ℝ) : Prop :=
  time_to_park / dist_to_park * dist_to_coffee_shop = 6

theorem jog_to_coffee_shop
  (time_to_park : ℕ)
  (dist_to_park : ℝ)
  (dist_to_coffee_shop : ℝ)
  (h1 : time_to_park = 12)
  (h2 : dist_to_park = 1.5)
  (h3 : dist_to_coffee_shop = 0.75)
: constant_pace_jogging time_to_park dist_to_park dist_to_coffee_shop :=
by sorry

end NUMINAMATH_GPT_jog_to_coffee_shop_l972_97238


namespace NUMINAMATH_GPT_pete_total_blocks_traveled_l972_97295

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end NUMINAMATH_GPT_pete_total_blocks_traveled_l972_97295


namespace NUMINAMATH_GPT_tan_diff_identity_l972_97266

theorem tan_diff_identity (α : ℝ) (hα : 0 < α ∧ α < π) (h : Real.sin α = 4 / 5) :
  Real.tan (π / 4 - α) = -1 / 7 ∨ Real.tan (π / 4 - α) = -7 :=
sorry

end NUMINAMATH_GPT_tan_diff_identity_l972_97266


namespace NUMINAMATH_GPT_height_of_picture_frame_l972_97212

-- Define the given conditions
def width : ℕ := 6
def perimeter : ℕ := 30
def perimeter_formula (w h : ℕ) : ℕ := 2 * (w + h)

-- Prove that the height of the picture frame is 9 inches
theorem height_of_picture_frame : ∃ height : ℕ, height = 9 ∧ perimeter_formula width height = perimeter :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_height_of_picture_frame_l972_97212


namespace NUMINAMATH_GPT_smallest_natural_number_k_l972_97273

theorem smallest_natural_number_k :
  ∃ k : ℕ, k = 4 ∧ ∀ (a : ℝ) (n : ℕ), 0 ≤ a ∧ a ≤ 1 ∧ 1 ≤ n → a^(k) * (1 - a)^(n) < 1 / (n + 1)^3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_k_l972_97273


namespace NUMINAMATH_GPT_trigonometric_identity_l972_97240

theorem trigonometric_identity
  (α : Real)
  (hcos : Real.cos α = -4/5)
  (hquad : π/2 < α ∧ α < π) :
  (-Real.sin (2 * α) / Real.cos α) = -6/5 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l972_97240


namespace NUMINAMATH_GPT_watermelon_yield_increase_l972_97252

noncomputable def yield_increase (initial_yield final_yield annual_increase_rate : ℝ) (years : ℕ) : ℝ :=
  initial_yield * (1 + annual_increase_rate) ^ years

theorem watermelon_yield_increase :
  ∀ (x : ℝ),
    (yield_increase 20 28.8 x 2 = 28.8) →
    (yield_increase 28.8 40 x 2 > 40) :=
by
  intros x hx
  have incEq : 20 * (1 + x) ^ 2 = 28.8 := hx
  sorry

end NUMINAMATH_GPT_watermelon_yield_increase_l972_97252


namespace NUMINAMATH_GPT_simplify_expression_l972_97215

-- Define the algebraic expression
def algebraic_expr (x : ℚ) : ℚ := (3 / (x - 1) - x - 1) * (x - 1) / (x^2 - 4 * x + 4)

theorem simplify_expression : algebraic_expr 0 = 1 :=
by
  -- The proof is skipped using sorry
  sorry

end NUMINAMATH_GPT_simplify_expression_l972_97215


namespace NUMINAMATH_GPT_factorization_of_m_squared_minus_4_l972_97233

theorem factorization_of_m_squared_minus_4 (m : ℝ) : m^2 - 4 = (m + 2) * (m - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_m_squared_minus_4_l972_97233


namespace NUMINAMATH_GPT_angles_proof_l972_97275

-- Definitions (directly from the conditions)
variable {θ₁ θ₂ θ₃ θ₄ : ℝ}

def complementary (θ₁ θ₂ : ℝ) : Prop := θ₁ + θ₂ = 90
def supplementary (θ₃ θ₄ : ℝ) : Prop := θ₃ + θ₄ = 180

-- Theorem statement
theorem angles_proof (h1 : complementary θ₁ θ₂) (h2 : supplementary θ₃ θ₄) (h3 : θ₁ = θ₃) :
  θ₂ + 90 = θ₄ :=
by
  sorry

end NUMINAMATH_GPT_angles_proof_l972_97275


namespace NUMINAMATH_GPT_find_m_if_z_is_pure_imaginary_l972_97217

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_m_if_z_is_pure_imaginary (m : ℝ) (z : ℂ) (i : ℂ) (h_i_unit : i^2 = -1) (h_z : z = (1 + i) / (1 - i) + m * (1 - i)) :
  is_pure_imaginary z → m = 0 := 
by
  sorry

end NUMINAMATH_GPT_find_m_if_z_is_pure_imaginary_l972_97217


namespace NUMINAMATH_GPT_susan_hourly_rate_l972_97244

-- Definitions based on conditions
def vacation_workdays : ℕ := 10 -- Susan is taking a two-week vacation equivalent to 10 workdays

def weekly_workdays : ℕ := 5 -- Susan works 5 days a week

def paid_vacation_days : ℕ := 6 -- Susan has 6 days of paid vacation

def hours_per_day : ℕ := 8 -- Susan works 8 hours a day

def missed_pay_total : ℕ := 480 -- Susan will miss $480 pay on her unpaid vacation days

-- Calculations
def unpaid_vacation_days : ℕ := vacation_workdays - paid_vacation_days

def daily_lost_pay : ℕ := missed_pay_total / unpaid_vacation_days

def hourly_rate : ℕ := daily_lost_pay / hours_per_day

theorem susan_hourly_rate :
  hourly_rate = 15 := by sorry

end NUMINAMATH_GPT_susan_hourly_rate_l972_97244


namespace NUMINAMATH_GPT_coefficient_of_x9_in_polynomial_is_240_l972_97236

-- Define the polynomial (1 + 3x - 2x^2)^5
noncomputable def polynomial : ℕ → ℝ := (fun x => (1 + 3*x - 2*x^2)^5)

-- Define the term we are interested in (x^9)
def term := 9

-- The coefficient we want to prove
def coefficient := 240

-- The goal is to prove that the coefficient of x^9 in the expansion of (1 + 3x - 2x^2)^5 is 240
theorem coefficient_of_x9_in_polynomial_is_240 : polynomial 9 = coefficient := sorry

end NUMINAMATH_GPT_coefficient_of_x9_in_polynomial_is_240_l972_97236
