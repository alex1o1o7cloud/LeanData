import Mathlib

namespace NUMINAMATH_GPT_problem_l442_44237

open BigOperators

variables {p q : ℝ} {n : ℕ}

theorem problem 
  (h : p + q = 1) : 
  ∑ r in Finset.range (n / 2 + 1), (-1 : ℝ) ^ r * (Nat.choose (n - r) r) * p^r * q^r = (p ^ (n + 1) - q ^ (n + 1)) / (p - q) :=
by
  sorry

end NUMINAMATH_GPT_problem_l442_44237


namespace NUMINAMATH_GPT_probability_team_B_wins_third_game_l442_44270

theorem probability_team_B_wins_third_game :
  ∀ (A B : ℕ → Prop),
    (∀ n, A n ∨ B n) ∧ -- Each game is won by either A or B
    (∀ n, A n ↔ ¬ B n) ∧ -- No ties, outcomes are independent
    (A 0) ∧ -- Team A wins the first game
    (B 1) ∧ -- Team B wins the second game
    (∃ n1 n2 n3, A n1 ∧ A n2 ∧ A n3 ∧ n1 ≠ n2 ∧ n2 ≠ n3 ∧ n1 ≠ n3) -- Team A wins three games
    → (∃ S, ((A 0) ∧ (B 1) ∧ (B 2)) ↔ (S = 1/3)) := sorry

end NUMINAMATH_GPT_probability_team_B_wins_third_game_l442_44270


namespace NUMINAMATH_GPT_populations_equal_after_years_l442_44283

-- Defining the initial population and rates of change
def initial_population_X : ℕ := 76000
def rate_of_decrease_X : ℕ := 1200
def initial_population_Y : ℕ := 42000
def rate_of_increase_Y : ℕ := 800

-- Define the number of years for which we need to find the populations to be equal
def years (n : ℕ) : Prop :=
  (initial_population_X - rate_of_decrease_X * n) = (initial_population_Y + rate_of_increase_Y * n)

-- Theorem stating that the populations will be equal at n = 17
theorem populations_equal_after_years {n : ℕ} (h : n = 17) : years n :=
by
  sorry

end NUMINAMATH_GPT_populations_equal_after_years_l442_44283


namespace NUMINAMATH_GPT_solve_complex_eq_l442_44276

open Complex

theorem solve_complex_eq (z : ℂ) (h : (3 - 4 * I) * z = 5) : z = (3 / 5) + (4 / 5) * I :=
by
  sorry

end NUMINAMATH_GPT_solve_complex_eq_l442_44276


namespace NUMINAMATH_GPT_discount_each_book_l442_44215

-- Definition of conditions
def original_price : ℝ := 5
def num_books : ℕ := 10
def total_paid : ℝ := 45

-- Theorem statement to prove the discount
theorem discount_each_book (d : ℝ) 
  (h1 : original_price * (num_books : ℝ) - d * (num_books : ℝ) = total_paid) : 
  d = 0.5 := 
sorry

end NUMINAMATH_GPT_discount_each_book_l442_44215


namespace NUMINAMATH_GPT_sum_of_ages_l442_44219

variable (J L : ℝ)
variable (h1 : J = L + 8)
variable (h2 : J + 10 = 5 * (L - 5))

theorem sum_of_ages (J L : ℝ) (h1 : J = L + 8) (h2 : J + 10 = 5 * (L - 5)) : J + L = 29.5 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l442_44219


namespace NUMINAMATH_GPT_sandwiches_left_l442_44200

theorem sandwiches_left (S G K L : ℕ) (h1 : S = 20) (h2 : G = 4) (h3 : K = 2 * G) (h4 : L = S - G - K) : L = 8 :=
sorry

end NUMINAMATH_GPT_sandwiches_left_l442_44200


namespace NUMINAMATH_GPT_frustum_volume_l442_44264

theorem frustum_volume (m : ℝ) (α : ℝ) (k : ℝ) : 
  m = 3/π ∧ 
  α = 43 + 40/60 + 42.2/3600 ∧ 
  k = 1 →
  frustumVolume = 0.79 := 
sorry

end NUMINAMATH_GPT_frustum_volume_l442_44264


namespace NUMINAMATH_GPT_find_number_l442_44249

theorem find_number (x : ℕ) : x * 9999 = 4691130840 → x = 469200 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_number_l442_44249


namespace NUMINAMATH_GPT_simplify_and_evaluate_l442_44223

noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := 2 - Real.sqrt 2

theorem simplify_and_evaluate : 
  let expr := (a / (a^2 - b^2) - 1 / (a + b)) / (b / (b - a))
  expr = -1 / 2 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l442_44223


namespace NUMINAMATH_GPT_speed_in_still_water_l442_44261

theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 25) (h_downstream : downstream_speed = 65) : 
  (upstream_speed + downstream_speed) / 2 = 45 :=
by
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l442_44261


namespace NUMINAMATH_GPT_comparison_of_a_b_c_l442_44240

theorem comparison_of_a_b_c : 
  let a := (1/3)^(2/5)
  let b := 2^(4/3)
  let c := Real.logb 2 (1/3)
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_comparison_of_a_b_c_l442_44240


namespace NUMINAMATH_GPT_ratio_of_areas_l442_44271

theorem ratio_of_areas (AB CD AH BG CF DG S_ABCD S_KLMN : ℕ)
  (h1 : AB = 15)
  (h2 : CD = 19)
  (h3 : DG = 17)
  (condition1 : S_ABCD = 17 * (AH + BG))
  (midpoints_AH_CF : AH = BG)
  (midpoints_CF_CD : CF = CD/2)
  (condition2 : (∃ h₁ h₂ : ℕ, S_KLMN = h₁ * AH + h₂ * CF / 2))
  (h_case1 : (S_KLMN = (AH + BG + CD)))
  (h_case2 : (S_KLMN = (AB + (CD - DG)))) :
  (S_ABCD / S_KLMN = 2 / 3 ∨ S_ABCD / S_KLMN = 2) :=
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l442_44271


namespace NUMINAMATH_GPT_circle_representation_l442_44285

theorem circle_representation (a : ℝ): 
  (∃ (x y : ℝ), (x^2 + y^2 + 2*x + a = 0) ∧ (∃ D E F, D = 2 ∧ E = 0 ∧ F = -a ∧ (D^2 + E^2 - 4*F > 0))) ↔ (a > -1) :=
by 
  sorry

end NUMINAMATH_GPT_circle_representation_l442_44285


namespace NUMINAMATH_GPT_find_a_of_line_slope_l442_44247

theorem find_a_of_line_slope (a : ℝ) (h1 : a > 0)
  (h2 : ∃ (b : ℝ), (a, 5) = (b * 1, b * 2) ∧ (2, a) = (b * 1, 2 * b) ∧ b = 1) 
  : a = 3 := 
sorry

end NUMINAMATH_GPT_find_a_of_line_slope_l442_44247


namespace NUMINAMATH_GPT_quadratic_solution_pair_l442_44256

open Real

noncomputable def solution_pair : ℝ × ℝ :=
  ((45 - 15 * sqrt 5) / 2, (45 + 15 * sqrt 5) / 2)

theorem quadratic_solution_pair (a c : ℝ) 
  (h1 : (∃ x : ℝ, a * x^2 + 30 * x + c = 0 ∧ ∀ y : ℝ, y ≠ x → a * y^2 + 30 * y + c ≠ 0))
  (h2 : a + c = 45)
  (h3 : a < c) :
  (a, c) = solution_pair :=
sorry

end NUMINAMATH_GPT_quadratic_solution_pair_l442_44256


namespace NUMINAMATH_GPT_find_x_set_eq_l442_44296

noncomputable def f : ℝ → ℝ :=
sorry -- The actual definition of f according to its properties is omitted

lemma odd_function (x : ℝ) : f (-x) = -f x :=
sorry

lemma periodic_function (x : ℝ) : f (x + 2) = -f x :=
sorry

lemma f_definition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = 1 / 2 * x :=
sorry

theorem find_x_set_eq (x : ℝ) : (f x = -1 / 2) ↔ (∃ k : ℤ, x = 4 * k - 1) :=
sorry

end NUMINAMATH_GPT_find_x_set_eq_l442_44296


namespace NUMINAMATH_GPT_sum_of_integers_l442_44288

variable (p q r s : ℤ)

theorem sum_of_integers :
  (p - q + r = 7) →
  (q - r + s = 8) →
  (r - s + p = 4) →
  (s - p + q = 1) →
  p + q + r + s = 20 := by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sum_of_integers_l442_44288


namespace NUMINAMATH_GPT_probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l442_44263

noncomputable def P_n (n : ℕ) : ℚ :=
  if n = 3 then 1 / 4
  else if n = 4 then 3 / 4
  else 0

theorem probability_center_in_convex_hull_3_points :
  P_n 3 = 1 / 4 :=
by
  sorry

theorem probability_center_in_convex_hull_4_points :
  P_n 4 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_center_in_convex_hull_3_points_probability_center_in_convex_hull_4_points_l442_44263


namespace NUMINAMATH_GPT_arithmetic_prog_sum_bound_l442_44287

noncomputable def Sn (n : ℕ) (a1 : ℝ) (d : ℝ) : ℝ := n * a1 + (n * (n - 1) / 2) * d

theorem arithmetic_prog_sum_bound (n : ℕ) (a1 an : ℝ) (d : ℝ) (h_d_neg : d < 0) 
  (ha_n : an = a1 + (n - 1) * d) :
  n * an < Sn n a1 d ∧ Sn n a1 d < n * a1 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_prog_sum_bound_l442_44287


namespace NUMINAMATH_GPT_greatest_integer_solution_l442_44278

theorem greatest_integer_solution :
  ∃ n : ℤ, (n^2 - 17 * n + 72 ≤ 0) ∧ (∀ m : ℤ, (m^2 - 17 * m + 72 ≤ 0) → m ≤ n) ∧ n = 9 :=
sorry

end NUMINAMATH_GPT_greatest_integer_solution_l442_44278


namespace NUMINAMATH_GPT_least_overlap_coffee_tea_l442_44231

open BigOperators

-- Define the percentages in a way that's compatible in Lean
def percentage (n : ℕ) := n / 100

noncomputable def C := percentage 75
noncomputable def T := percentage 80
noncomputable def B := percentage 55

-- The theorem statement
theorem least_overlap_coffee_tea : C + T - 1 = B := sorry

end NUMINAMATH_GPT_least_overlap_coffee_tea_l442_44231


namespace NUMINAMATH_GPT_kaleb_toys_l442_44277

def initial_savings : ℕ := 21
def allowance : ℕ := 15
def cost_per_toy : ℕ := 6

theorem kaleb_toys : (initial_savings + allowance) / cost_per_toy = 6 :=
by
  sorry

end NUMINAMATH_GPT_kaleb_toys_l442_44277


namespace NUMINAMATH_GPT_candy_system_of_equations_l442_44243

-- Definitions based on conditions
def candy_weight := 100
def candy_price1 := 36
def candy_price2 := 20
def mixed_candy_price := 28

theorem candy_system_of_equations (x y: ℝ):
  (x + y = candy_weight) ∧ (candy_price1 * x + candy_price2 * y = mixed_candy_price * candy_weight) :=
sorry

end NUMINAMATH_GPT_candy_system_of_equations_l442_44243


namespace NUMINAMATH_GPT_circumscribed_circles_intersect_l442_44269

noncomputable def circumcircle (a b c : Point) : Set Point := sorry

noncomputable def intersect_at_single_point (circles : List (Set Point)) : Option Point := sorry

variables {A1 A2 A3 B1 B2 B3 : Point}

theorem circumscribed_circles_intersect
  (h1 : ∃ P, ∀ circle ∈ [
    circumcircle A1 A2 B3, 
    circumcircle A1 B2 A3, 
    circumcircle B1 A2 A3
  ], P ∈ circle) :
  ∃ Q, ∀ circle ∈ [
    circumcircle B1 B2 A3, 
    circumcircle B1 A2 B3, 
    circumcircle A1 B2 B3
  ], Q ∈ circle :=
sorry

end NUMINAMATH_GPT_circumscribed_circles_intersect_l442_44269


namespace NUMINAMATH_GPT_Bowen_total_spent_l442_44260

def pencil_price : ℝ := 0.25
def pen_price : ℝ := 0.15
def num_pens : ℕ := 40

def num_pencils := num_pens + (2 / 5) * num_pens

theorem Bowen_total_spent : num_pencils * pencil_price + num_pens * pen_price = 20 := by
  sorry

end NUMINAMATH_GPT_Bowen_total_spent_l442_44260


namespace NUMINAMATH_GPT_find_x_l442_44259

theorem find_x (x : ℝ) (h1 : (x - 1) / (x + 2) = 0) (h2 : x ≠ -2) : x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l442_44259


namespace NUMINAMATH_GPT_amount_paid_for_peaches_l442_44291

def total_spent := 23.86
def cherries_spent := 11.54
def peaches_spent := 12.32

theorem amount_paid_for_peaches :
  total_spent - cherries_spent = peaches_spent :=
sorry

end NUMINAMATH_GPT_amount_paid_for_peaches_l442_44291


namespace NUMINAMATH_GPT_power_mod_eq_nine_l442_44227

theorem power_mod_eq_nine :
  ∃ n : ℕ, 13^6 ≡ n [MOD 11] ∧ 0 ≤ n ∧ n < 11 ∧ n = 9 :=
by
  sorry

end NUMINAMATH_GPT_power_mod_eq_nine_l442_44227


namespace NUMINAMATH_GPT_identify_perfect_square_is_689_l442_44232

-- Definitions of the conditions
def natural_numbers (n : ℕ) : Prop := True -- All natural numbers are accepted
def digits_in_result (n m : ℕ) (d : ℕ) : Prop := (n * m) % 1000 = d

-- Theorem to be proved
theorem identify_perfect_square_is_689 (n : ℕ) :
  (∀ m, natural_numbers m → digits_in_result m m 689 ∨ digits_in_result m m 759) →
  ∃ m, natural_numbers m ∧ digits_in_result m m 689 :=
sorry

end NUMINAMATH_GPT_identify_perfect_square_is_689_l442_44232


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_l442_44281

noncomputable def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_formula {a : ℕ → ℤ} (h_seq : arithmetic_seq a) 
  (h_a1 : a 1 = 6) (h_a3a5 : a 3 + a 5 = 0) : 
  ∀ n, a n = 8 - 2 * n :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_l442_44281


namespace NUMINAMATH_GPT_log_bounds_sum_l442_44222

theorem log_bounds_sum : (∀ a b : ℕ, a = 18 ∧ b = 19 → 18 < Real.log 537800 / Real.log 2 ∧ Real.log 537800 / Real.log 2 < 19 → a + b = 37) := 
sorry

end NUMINAMATH_GPT_log_bounds_sum_l442_44222


namespace NUMINAMATH_GPT_miranda_can_stuff_10_pillows_l442_44244

def feathers_needed_per_pillow : ℕ := 2
def goose_feathers_per_pound : ℕ := 300
def duck_feathers_per_pound : ℕ := 500
def goose_total_feathers : ℕ := 3600
def duck_total_feathers : ℕ := 4000

theorem miranda_can_stuff_10_pillows :
  (goose_total_feathers / goose_feathers_per_pound + duck_total_feathers / duck_feathers_per_pound) / feathers_needed_per_pillow = 10 :=
by
  sorry

end NUMINAMATH_GPT_miranda_can_stuff_10_pillows_l442_44244


namespace NUMINAMATH_GPT_min_number_of_students_l442_44229

theorem min_number_of_students 
  (n : ℕ)
  (h1 : 25 ≡ 99 [MOD n])
  (h2 : 8 ≡ 119 [MOD n]) : 
  n = 37 :=
by sorry

end NUMINAMATH_GPT_min_number_of_students_l442_44229


namespace NUMINAMATH_GPT_parabola_focus_distance_l442_44251

theorem parabola_focus_distance (p m : ℝ) (hp : p > 0)
  (P_on_parabola : m^2 = 2 * p)
  (PF_dist : (1 + p / 2) = 3) : p = 4 := 
  sorry

end NUMINAMATH_GPT_parabola_focus_distance_l442_44251


namespace NUMINAMATH_GPT_gallons_of_gas_l442_44216

-- Define the conditions
def mpg : ℕ := 19
def d1 : ℕ := 15
def d2 : ℕ := 6
def d3 : ℕ := 2
def d4 : ℕ := 4
def d5 : ℕ := 11

-- The theorem to prove
theorem gallons_of_gas : (d1 + d2 + d3 + d4 + d5) / mpg = 2 := 
by {
    sorry
}

end NUMINAMATH_GPT_gallons_of_gas_l442_44216


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l442_44228

theorem negation_of_universal_proposition :
  (¬ ∀ x > 1, (1 / 2)^x < 1 / 2) ↔ (∃ x > 1, (1 / 2)^x ≥ 1 / 2) :=
sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l442_44228


namespace NUMINAMATH_GPT_rationalize_denominator_l442_44201

theorem rationalize_denominator : (1 / (Real.sqrt 3 + 1)) = ((Real.sqrt 3 - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l442_44201


namespace NUMINAMATH_GPT_fixed_monthly_costs_l442_44280

theorem fixed_monthly_costs
  (cost_per_component : ℕ) (shipping_cost : ℕ) 
  (num_components : ℕ) (selling_price : ℚ)
  (F : ℚ) :
  cost_per_component = 80 →
  shipping_cost = 6 →
  num_components = 150 →
  selling_price = 196.67 →
  F = (num_components * selling_price) - (num_components * (cost_per_component + shipping_cost)) →
  F = 16600.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fixed_monthly_costs_l442_44280


namespace NUMINAMATH_GPT_Mongolian_Mathematical_Olympiad_54th_l442_44254

theorem Mongolian_Mathematical_Olympiad_54th {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^4 + b^4 + c^4 + (a^2 / (b + c)^2) + (b^2 / (c + a)^2) + (c^2 / (a + b)^2) ≥ a * b + b * c + c * a :=
sorry

end NUMINAMATH_GPT_Mongolian_Mathematical_Olympiad_54th_l442_44254


namespace NUMINAMATH_GPT_speed_of_train_l442_44210

def distance : ℝ := 80
def time : ℝ := 6
def expected_speed : ℝ := 13.33

theorem speed_of_train : distance / time = expected_speed :=
by
  sorry

end NUMINAMATH_GPT_speed_of_train_l442_44210


namespace NUMINAMATH_GPT_determine_value_of_m_l442_44289

noncomputable def conics_same_foci (m : ℝ) : Prop :=
  let c1 := Real.sqrt (4 - m^2)
  let c2 := Real.sqrt (m + 2)
  (∀ (x y : ℝ),
    (x^2 / 4 + y^2 / m^2 = 1) → (x^2 / m - y^2 / 2 = 1) → c1 = c2) → 
  m = 1

theorem determine_value_of_m : ∃ (m : ℝ), conics_same_foci m :=
sorry

end NUMINAMATH_GPT_determine_value_of_m_l442_44289


namespace NUMINAMATH_GPT_paint_cans_needed_l442_44209

theorem paint_cans_needed
    (num_bedrooms : ℕ)
    (num_other_rooms : ℕ)
    (total_rooms : ℕ)
    (gallons_per_room : ℕ)
    (color_paint_cans_per_gallon : ℕ)
    (white_paint_cans_per_gallon : ℕ)
    (total_paint_needed : ℕ)
    (color_paint_cans_needed : ℕ)
    (white_paint_cans_needed : ℕ)
    (total_paint_cans : ℕ)
    (h1 : num_bedrooms = 3)
    (h2 : num_other_rooms = 2 * num_bedrooms)
    (h3 : total_rooms = num_bedrooms + num_other_rooms)
    (h4 : gallons_per_room = 2)
    (h5 : total_paint_needed = total_rooms * gallons_per_room)
    (h6 : color_paint_cans_per_gallon = 1)
    (h7 : white_paint_cans_per_gallon = 3)
    (h8 : color_paint_cans_needed = num_bedrooms * gallons_per_room * color_paint_cans_per_gallon)
    (h9 : white_paint_cans_needed = (num_other_rooms * gallons_per_room) / white_paint_cans_per_gallon)
    (h10 : total_paint_cans = color_paint_cans_needed + white_paint_cans_needed) :
    total_paint_cans = 10 :=
by sorry

end NUMINAMATH_GPT_paint_cans_needed_l442_44209


namespace NUMINAMATH_GPT_remainder_987654_div_8_l442_44235

theorem remainder_987654_div_8 : 987654 % 8 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_987654_div_8_l442_44235


namespace NUMINAMATH_GPT_mean_of_remaining_quiz_scores_l442_44292

theorem mean_of_remaining_quiz_scores (k : ℕ) (hk : k > 12) 
  (mean_k : ℝ) (mean_12 : ℝ) 
  (mean_class : mean_k = 8) 
  (mean_12_group : mean_12 = 14) 
  (mean_correct : mean_12 * 12 + mean_k * (k - 12) = 8 * k) :
  mean_k * (k - 12) = (8 * k - 168) := 
by {
  sorry
}

end NUMINAMATH_GPT_mean_of_remaining_quiz_scores_l442_44292


namespace NUMINAMATH_GPT_cary_mow_weekends_l442_44220

theorem cary_mow_weekends :
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  remaining_amount / earn_per_weekend = 6 :=
by
  let cost_shoes := 120
  let saved_amount := 30
  let earn_per_lawn := 5
  let lawns_per_weekend := 3
  let remaining_amount := cost_shoes - saved_amount
  let earn_per_weekend := lawns_per_weekend * earn_per_lawn
  have needed_weekends : remaining_amount / earn_per_weekend = 6 :=
    sorry
  exact needed_weekends

end NUMINAMATH_GPT_cary_mow_weekends_l442_44220


namespace NUMINAMATH_GPT_school_pupils_l442_44236

def girls : ℕ := 868
def difference : ℕ := 281
def boys (g b : ℕ) : Prop := g = b + difference
def total_pupils (g b t : ℕ) : Prop := t = g + b

theorem school_pupils : 
  ∃ b t, boys girls b ∧ total_pupils girls b t ∧ t = 1455 :=
by
  sorry

end NUMINAMATH_GPT_school_pupils_l442_44236


namespace NUMINAMATH_GPT_cyclists_meeting_time_l442_44295

theorem cyclists_meeting_time :
  ∃ t : ℕ, t = Nat.lcm 7 (Nat.lcm 12 9) ∧ t = 252 :=
by
  use 252
  have h1 : Nat.lcm 7 (Nat.lcm 12 9) = 252 := sorry
  exact ⟨rfl, h1⟩

end NUMINAMATH_GPT_cyclists_meeting_time_l442_44295


namespace NUMINAMATH_GPT_gain_percent_is_150_l442_44233

theorem gain_percent_is_150 (CP SP : ℝ) (hCP : CP = 10) (hSP : SP = 25) : (SP - CP) / CP * 100 = 150 := by
  sorry

end NUMINAMATH_GPT_gain_percent_is_150_l442_44233


namespace NUMINAMATH_GPT_diana_total_extra_video_game_time_l442_44250

-- Definitions from the conditions
def minutesPerHourReading := 30
def raisePercent := 20
def choresToMinutes := 10
def maxChoresBonusMinutes := 60
def sportsPracticeHours := 8
def homeworkHours := 4
def totalWeekHours := 24
def readingHours := 8
def choresCompleted := 10

-- Deriving some necessary facts
def baseVideoGameTime := readingHours * minutesPerHourReading
def raiseMinutes := baseVideoGameTime * (raisePercent / 100)
def videoGameTimeWithRaise := baseVideoGameTime + raiseMinutes

def bonusesFromChores := (choresCompleted / 2) * choresToMinutes
def limitedChoresBonus := min bonusesFromChores maxChoresBonusMinutes

-- Total extra video game time
def totalExtraVideoGameTime := videoGameTimeWithRaise + limitedChoresBonus

-- The proof problem
theorem diana_total_extra_video_game_time : totalExtraVideoGameTime = 338 := by
  sorry

end NUMINAMATH_GPT_diana_total_extra_video_game_time_l442_44250


namespace NUMINAMATH_GPT_probability_of_9_heads_in_12_l442_44204

def coin_flip_probability_9_heads_in_12_flips : Prop :=
  let total_outcomes := 2^12
  let success_outcomes := Nat.choose 12 9
  success_outcomes / total_outcomes = 220 / 4096

theorem probability_of_9_heads_in_12 : coin_flip_probability_9_heads_in_12_flips :=
  sorry

end NUMINAMATH_GPT_probability_of_9_heads_in_12_l442_44204


namespace NUMINAMATH_GPT_max_sum_of_squares_eq_50_l442_44266

theorem max_sum_of_squares_eq_50 :
  ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ (∀ x' y' : ℤ, x'^2 + y'^2 = 50 → x + y ≥ x' + y') ∧ x + y = 10 := 
sorry

end NUMINAMATH_GPT_max_sum_of_squares_eq_50_l442_44266


namespace NUMINAMATH_GPT_rohan_age_is_25_l442_44207

-- Define the current age of Rohan
def rohan_current_age (x : ℕ) : Prop :=
  x + 15 = 4 * (x - 15)

-- The goal is to prove that Rohan's current age is 25 years old
theorem rohan_age_is_25 : ∃ x : ℕ, rohan_current_age x ∧ x = 25 :=
by
  existsi (25 : ℕ)
  -- Proof is omitted since this is a statement only
  sorry

end NUMINAMATH_GPT_rohan_age_is_25_l442_44207


namespace NUMINAMATH_GPT_pages_after_break_l442_44217

-- Formalize the conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.70

-- Define the proof problem
theorem pages_after_break : 
  let pages_read_before_break := (break_percentage * total_pages)
  let pages_remaining := total_pages - pages_read_before_break
  pages_remaining = 9 :=
by
  sorry

end NUMINAMATH_GPT_pages_after_break_l442_44217


namespace NUMINAMATH_GPT_find_a16_l442_44211

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n ≥ 1, a (n + 1) = 1 - 1 / a n

theorem find_a16 (a : ℕ → ℝ) (h : seq a) : a 16 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a16_l442_44211


namespace NUMINAMATH_GPT_smallest_nonprime_with_large_prime_factors_l442_44208

/-- 
The smallest nonprime integer greater than 1 with no prime factor less than 15
falls in the range 260 < m ≤ 270.
-/
theorem smallest_nonprime_with_large_prime_factors :
  ∃ m : ℕ, 2 < m ∧ ¬ Nat.Prime m ∧ (∀ p : ℕ, Nat.Prime p → p ∣ m → 15 ≤ p) ∧ 260 < m ∧ m ≤ 270 :=
by
  sorry

end NUMINAMATH_GPT_smallest_nonprime_with_large_prime_factors_l442_44208


namespace NUMINAMATH_GPT_profit_percentage_for_unspecified_weight_l442_44224

-- Definitions to align with the conditions
def total_sugar : ℝ := 1000
def profit_400_kg : ℝ := 0.08
def unspecified_weight : ℝ := 600
def overall_profit : ℝ := 0.14
def total_400_kg := total_sugar - unspecified_weight
def total_overall_profit := total_sugar * overall_profit
def total_400_kg_profit := total_400_kg * profit_400_kg
def total_unspecified_weight_profit (profit_percentage : ℝ) := unspecified_weight * profit_percentage

-- The theorem statement
theorem profit_percentage_for_unspecified_weight : 
  ∃ (profit_percentage : ℝ), total_400_kg_profit + total_unspecified_weight_profit profit_percentage = total_overall_profit ∧ profit_percentage = 0.18 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_for_unspecified_weight_l442_44224


namespace NUMINAMATH_GPT_sum_digits_350_1350_base2_l442_44298

def binary_sum_digits (n : ℕ) : ℕ :=
  (Nat.digits 2 n).sum

theorem sum_digits_350_1350_base2 :
  binary_sum_digits 350 + binary_sum_digits 1350 = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_digits_350_1350_base2_l442_44298


namespace NUMINAMATH_GPT_question_solution_l442_44297

theorem question_solution
  (f : ℝ → ℝ)
  (h_decreasing : ∀ ⦃x y : ℝ⦄, -3 < x ∧ x < 0 → -3 < y ∧ y < 0 → x < y → f y < f x)
  (h_symmetry : ∀ x : ℝ, f (x) = f (-x + 6)) :
  f (-5) < f (-3/2) ∧ f (-3/2) < f (-7/2) :=
sorry

end NUMINAMATH_GPT_question_solution_l442_44297


namespace NUMINAMATH_GPT_number_of_sides_sum_of_interior_angles_l442_44257

-- Condition: each exterior angle of the regular polygon is 18 degrees.
def exterior_angle (n : ℕ) : Prop :=
  360 / n = 18

-- Question 1: Determine the number of sides the polygon has.
theorem number_of_sides : ∃ n, n > 2 ∧ exterior_angle n :=
  sorry

-- Question 2: Calculate the sum of the interior angles.
theorem sum_of_interior_angles {n : ℕ} (h : 360 / n = 18) : 
  180 * (n - 2) = 3240 :=
  sorry

end NUMINAMATH_GPT_number_of_sides_sum_of_interior_angles_l442_44257


namespace NUMINAMATH_GPT_number_of_commonly_used_structures_is_3_l442_44226

def commonly_used_algorithm_structures : Nat := 3
theorem number_of_commonly_used_structures_is_3 
  (structures : Nat)
  (h : structures = 1 ∨ structures = 2 ∨ structures = 3 ∨ structures = 4) :
  commonly_used_algorithm_structures = 3 :=
by
  -- Proof to be added
  sorry

end NUMINAMATH_GPT_number_of_commonly_used_structures_is_3_l442_44226


namespace NUMINAMATH_GPT_jessica_total_money_after_activities_l442_44268

-- Definitions for given conditions
def weekly_allowance : ℕ := 10
def spent_on_movies : ℕ := weekly_allowance / 2
def earned_from_washing_car : ℕ := 6

-- Theorem statement
theorem jessica_total_money_after_activities : 
  (weekly_allowance - spent_on_movies) + earned_from_washing_car = 11 :=
by 
  sorry

end NUMINAMATH_GPT_jessica_total_money_after_activities_l442_44268


namespace NUMINAMATH_GPT_rectangle_area_y_l442_44225

theorem rectangle_area_y (y : ℚ) (h_pos: y > 0) 
  (h_area: ((6 : ℚ) - (-2)) * (y - 2) = 64) : y = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_y_l442_44225


namespace NUMINAMATH_GPT_quadratic_completion_l442_44282

theorem quadratic_completion (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 26 * x + 81 = (x + b)^2 + c) → b + c = -101 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_completion_l442_44282


namespace NUMINAMATH_GPT_helen_oranges_l442_44218

def initial_oranges := 9
def oranges_from_ann := 29
def oranges_taken_away := 14

def final_oranges (initial : Nat) (add : Nat) (taken : Nat) : Nat :=
  initial + add - taken

theorem helen_oranges :
  final_oranges initial_oranges oranges_from_ann oranges_taken_away = 24 :=
by
  sorry

end NUMINAMATH_GPT_helen_oranges_l442_44218


namespace NUMINAMATH_GPT_votes_cast_l442_44206

theorem votes_cast (V : ℝ) (h1 : ∃ Vc, Vc = 0.25 * V) (h2 : ∃ Vr, Vr = 0.25 * V + 4000) : V = 8000 :=
sorry

end NUMINAMATH_GPT_votes_cast_l442_44206


namespace NUMINAMATH_GPT_height_of_trapezoid_l442_44274

-- Define the condition that a trapezoid has diagonals of given lengths and a given midline.
def trapezoid_conditions (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : Prop := 
  AC = 6 ∧ BD = 8 ∧ ML = 5

-- Define the height of the trapezoid.
def trapezoid_height (AC BD ML : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : ℝ :=
  4.8

-- The theorem statement
theorem height_of_trapezoid (AC BD ML h : ℝ) (h_d1 : AC = 6) (h_d2 : BD = 8) (h_ml : ML = 5) : 
  trapezoid_conditions AC BD ML h_d1 h_d2 h_ml 
  → trapezoid_height AC BD ML h_d1 h_d2 h_ml = 4.8 := 
by
  intros
  sorry

end NUMINAMATH_GPT_height_of_trapezoid_l442_44274


namespace NUMINAMATH_GPT_greatest_possible_large_chips_l442_44212

theorem greatest_possible_large_chips :
  ∃ l s : ℕ, ∃ p : ℕ, s + l = 61 ∧ s = l + p ∧ Nat.Prime p ∧ l = 29 :=
sorry

end NUMINAMATH_GPT_greatest_possible_large_chips_l442_44212


namespace NUMINAMATH_GPT_probability_line_through_cube_faces_l442_44284

def prob_line_intersects_cube_faces : ℚ :=
  1 / 7

theorem probability_line_through_cube_faces :
  let cube_vertices := 8
  let total_selections := Nat.choose cube_vertices 2
  let body_diagonals := 4
  let probability := (body_diagonals : ℚ) / total_selections
  probability = prob_line_intersects_cube_faces :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_line_through_cube_faces_l442_44284


namespace NUMINAMATH_GPT_share_expenses_l442_44267

theorem share_expenses (h l : ℕ) : 
  let henry_paid := 120
  let linda_paid := 150
  let jack_paid := 210
  let total_paid := henry_paid + linda_paid + jack_paid
  let each_should_pay := total_paid / 3
  let henry_owes := each_should_pay - henry_paid
  let linda_owes := each_should_pay - linda_paid
  (h = henry_owes) → 
  (l = linda_owes) → 
  h - l = 30 := by
  sorry

end NUMINAMATH_GPT_share_expenses_l442_44267


namespace NUMINAMATH_GPT_largest_d_l442_44262

theorem largest_d (a b c d : ℝ) (h1 : a + b + c + d = 10) 
  (h2 : ab + ac + ad + bc + bd + cd = 20) : 
  d ≤ (5 + Real.sqrt 105) / 2 := sorry

end NUMINAMATH_GPT_largest_d_l442_44262


namespace NUMINAMATH_GPT_find_a3_l442_44273

def sequence_sum (n : ℕ) : ℕ := n^2 + n

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem find_a3 : a 3 = 6 := by
  sorry

end NUMINAMATH_GPT_find_a3_l442_44273


namespace NUMINAMATH_GPT_find_CM_of_trapezoid_l442_44252

noncomputable def trapezoid_CM (AD BC : ℝ) (M : ℝ) : ℝ :=
  if (AD = 12) ∧ (BC = 8) ∧ (M = 2.4)
  then M
  else 0

theorem find_CM_of_trapezoid (trapezoid_ABCD : Type) (AD BC CM : ℝ) (AM_divides_eq_areas : Prop) :
  AD = 12 → BC = 8 → AM_divides_eq_areas → CM = 2.4 := 
by
  intros h1 h2 h3
  have : AD = 12 := h1
  have : BC = 8 := h2
  have : CM = 2.4 := sorry
  exact this

end NUMINAMATH_GPT_find_CM_of_trapezoid_l442_44252


namespace NUMINAMATH_GPT_find_prices_find_min_money_spent_l442_44234

-- Define the prices of volleyball and soccer ball
def prices (pv ps : ℕ) : Prop :=
  pv + 20 = ps ∧ 500 / ps = 400 / pv

-- Define the quantity constraint
def quantity_constraint (a : ℕ) : Prop :=
  a ≥ 25 ∧ a < 50

-- Define the minimum amount spent problem
def min_money_spent (a : ℕ) (pv ps : ℕ) : Prop :=
  prices pv ps → quantity_constraint a → 100 * a + 80 * (50 - a) = 4500

-- Prove the price of each volleyball and soccer ball
theorem find_prices : ∃ (pv ps : ℕ), prices pv ps ∧ pv = 80 ∧ ps = 100 :=
by {sorry}

-- Prove the minimum amount of money spent
theorem find_min_money_spent : ∃ (a pv ps : ℕ), min_money_spent a pv ps :=
by {sorry}

end NUMINAMATH_GPT_find_prices_find_min_money_spent_l442_44234


namespace NUMINAMATH_GPT_fewer_seats_right_side_l442_44245

theorem fewer_seats_right_side
  (left_seats : ℕ)
  (people_per_seat : ℕ)
  (back_seat_capacity : ℕ)
  (total_capacity : ℕ)
  (h1 : left_seats = 15)
  (h2 : people_per_seat = 3)
  (h3 : back_seat_capacity = 12)
  (h4 : total_capacity = 93)
  : left_seats - (total_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat = 3 :=
  by sorry

end NUMINAMATH_GPT_fewer_seats_right_side_l442_44245


namespace NUMINAMATH_GPT_convert_50_to_base_3_l442_44242

-- Define a function to convert decimal to ternary (base-3)
def convert_to_ternary (n : ℕ) : ℕ := sorry

-- Main theorem statement
theorem convert_50_to_base_3 : convert_to_ternary 50 = 1212 :=
sorry

end NUMINAMATH_GPT_convert_50_to_base_3_l442_44242


namespace NUMINAMATH_GPT_M_subset_N_l442_44265

def M : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def N : Set ℝ := { y | ∃ x : ℝ, y = x^2 }

theorem M_subset_N : M ⊆ N :=
by
  sorry

end NUMINAMATH_GPT_M_subset_N_l442_44265


namespace NUMINAMATH_GPT_sqrt_inequality_l442_44202

theorem sqrt_inequality (x y z : ℝ) (hx : 1 < x) (hy : 1 < y) (hz : 1 < z)
  (h : 1 / x + 1 / y + 1 / z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := 
by
  sorry

end NUMINAMATH_GPT_sqrt_inequality_l442_44202


namespace NUMINAMATH_GPT_find_gain_percent_l442_44203

-- Definitions based on the conditions
def CP : ℕ := 900
def SP : ℕ := 1170

-- Calculation of gain
def Gain := SP - CP

-- Calculation of gain percent
def GainPercent := (Gain * 100) / CP

-- The theorem to prove the gain percent is 30%
theorem find_gain_percent : GainPercent = 30 := 
by
  sorry -- Proof to be filled in.

end NUMINAMATH_GPT_find_gain_percent_l442_44203


namespace NUMINAMATH_GPT_sum_of_angles_of_inscribed_quadrilateral_l442_44214

/--
Given a quadrilateral EFGH inscribed in a circle, and the measures of ∠EGH = 50° and ∠GFE = 70°,
then the sum of the angles ∠EFG + ∠EHG is 60°.
-/
theorem sum_of_angles_of_inscribed_quadrilateral
  (E F G H : Type)
  (circumscribed : True) -- This is just a place holder for the circle condition
  (angle_EGH : ℝ) (angle_GFE : ℝ)
  (h1 : angle_EGH = 50)
  (h2 : angle_GFE = 70) :
  ∃ (angle_EFG angle_EHG : ℝ), angle_EFG + angle_EHG = 60 := sorry

end NUMINAMATH_GPT_sum_of_angles_of_inscribed_quadrilateral_l442_44214


namespace NUMINAMATH_GPT_arithmetic_seq_sum_is_110_l442_44213

noncomputable def S₁₀ (a_1 : ℝ) : ℝ :=
  10 / 2 * (2 * a_1 + 9 * (-2))

theorem arithmetic_seq_sum_is_110 (a1 a3 a7 a9 : ℝ) 
  (h_diff3 : a3 = a1 - 4)
  (h_diff7 : a7 = a1 - 12)
  (h_diff9 : a9 = a1 - 16)
  (h_geom : (a1 - 12) ^ 2 = (a1 - 4) * (a1 - 16)) :
  S₁₀ a1 = 110 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_is_110_l442_44213


namespace NUMINAMATH_GPT_arithmetic_sequence_formula_sum_Tn_formula_l442_44253

variable {a : ℕ → ℤ} -- The sequence a_n
variable {S : ℕ → ℤ} -- The sum S_n
variable {a₃ : ℤ} (h₁ : a₃ = 20)
variable {S₃ S₄ : ℤ} (h₂ : 2 * S₃ = S₄ + 8)

/- The general formula for the arithmetic sequence a_n -/
theorem arithmetic_sequence_formula (d : ℤ) (a₁ : ℤ)
  (h₃ : (a₃ = a₁ + 2 * d))
  (h₄ : (S₃ = 3 * a₁ + 3 * d))
  (h₅ : (S₄ = 4 * a₁ + 6 * d)) :
  ∀ n : ℕ, a n = 8 * n - 4 :=
by
  sorry

variable {b : ℕ → ℚ} -- Define b_n
variable {T : ℕ → ℚ} -- Define T_n
variable {S_general : ℕ → ℚ} (h₆ : ∀ n, S n = 4 * n ^ 2)
variable {b_general : ℚ → ℚ} (h₇ : ∀ n, b n = 1 / (S n - 1))
variable {T_general : ℕ → ℚ} -- Define T_n

/- The formula for T_n given b_n -/
theorem sum_Tn_formula :
  ∀ n : ℕ, T n = n / (2 * n + 1) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_sum_Tn_formula_l442_44253


namespace NUMINAMATH_GPT_accommodate_students_l442_44299

-- Define the parameters
def number_of_classrooms := 15
def one_third_classrooms := number_of_classrooms / 3
def desks_per_classroom_30 := 30
def desks_per_classroom_25 := 25

-- Define the number of classrooms for each type
def classrooms_with_30_desks := one_third_classrooms
def classrooms_with_25_desks := number_of_classrooms - classrooms_with_30_desks

-- Calculate total number of students that can be accommodated
def total_students : ℕ := 
  (classrooms_with_30_desks * desks_per_classroom_30) +
  (classrooms_with_25_desks * desks_per_classroom_25)

-- Prove that total number of students that the school can accommodate is 400
theorem accommodate_students : total_students = 400 := sorry

end NUMINAMATH_GPT_accommodate_students_l442_44299


namespace NUMINAMATH_GPT_man_l442_44286

theorem man's_speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h_current_speed : current_speed = 5) (h_against_current_speed : against_current_speed = 12) 
  (h_v : v - current_speed = against_current_speed) : 
  v + current_speed = 22 := 
by
  sorry

end NUMINAMATH_GPT_man_l442_44286


namespace NUMINAMATH_GPT_min_increase_velocity_correct_l442_44238

noncomputable def min_increase_velocity (V_A V_B V_C V_D : ℝ) (dist_AC dist_CD : ℝ) : ℝ :=
  let t_AC := dist_AC / (V_A + V_C)
  let t_AB := 30 / (V_A - V_B)
  let t_AD := (dist_AC + dist_CD) / (V_A + V_D)
  let new_velocity_A := (dist_AC + dist_CD) / t_AC - V_D
  new_velocity_A - V_A

theorem min_increase_velocity_correct :
  min_increase_velocity 80 50 70 60 300 400 = 210 :=
by
  sorry

end NUMINAMATH_GPT_min_increase_velocity_correct_l442_44238


namespace NUMINAMATH_GPT_days_to_empty_tube_l442_44230

-- Define the conditions
def gelInTube : ℕ := 128
def dailyUsage : ℕ := 4

-- Define the proof statement
theorem days_to_empty_tube : gelInTube / dailyUsage = 32 := 
by 
  sorry

end NUMINAMATH_GPT_days_to_empty_tube_l442_44230


namespace NUMINAMATH_GPT_inverse_f_1_l442_44290

noncomputable def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

theorem inverse_f_1 : ∃ x : ℝ, f x = 1 ∧ x = 2 := by
sorry

end NUMINAMATH_GPT_inverse_f_1_l442_44290


namespace NUMINAMATH_GPT_find_x_l442_44221

-- Define the vectors a, b, and c
def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, 1)

-- Define the collinearity condition
def collinear_with_3a_plus_b (x : ℝ) : Prop :=
  ∃ k : ℝ, c x = k • (3 • a + b)

theorem find_x :
  ∀ x : ℝ, collinear_with_3a_plus_b x → x = -4 := 
sorry

end NUMINAMATH_GPT_find_x_l442_44221


namespace NUMINAMATH_GPT_perfect_square_expression_l442_44241

theorem perfect_square_expression : 
    ∀ x : ℝ, (11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02)^2) → (x = 0.4792) :=
by
  intros x h
  -- sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_perfect_square_expression_l442_44241


namespace NUMINAMATH_GPT_coconut_grove_yield_l442_44258

theorem coconut_grove_yield (x Y : ℕ) (h1 : x = 10)
  (h2 : (x + 2) * 30 + x * Y + (x - 2) * 180 = 3 * x * 100) : Y = 120 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_coconut_grove_yield_l442_44258


namespace NUMINAMATH_GPT_negation_of_proposition_l442_44246

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x^2 + 2 * x + 3 ≥ 0)) ↔ (∃ x : ℝ, x^2 + 2 * x + 3 < 0) :=
by sorry

end NUMINAMATH_GPT_negation_of_proposition_l442_44246


namespace NUMINAMATH_GPT_solution_set_of_xf_l442_44255

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

theorem solution_set_of_xf (f : ℝ → ℝ) (hf_odd : is_odd_function f) (hf_one : f 1 = 0)
    (h_derivative : ∀ x > 0, (x * (deriv f x) - f x) / (x^2) > 0) :
    {x : ℝ | x * f x > 0} = {x : ℝ | x < -1 ∨ x > 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_xf_l442_44255


namespace NUMINAMATH_GPT_volume_Q3_l442_44294

noncomputable def sequence_of_polyhedra (n : ℕ) : ℚ :=
match n with
| 0     => 1
| 1     => 3 / 2
| 2     => 45 / 32
| 3     => 585 / 128
| _     => 0 -- for n > 3 not defined

theorem volume_Q3 : sequence_of_polyhedra 3 = 585 / 128 :=
by
  -- Placeholder for the theorem proof
  sorry

end NUMINAMATH_GPT_volume_Q3_l442_44294


namespace NUMINAMATH_GPT_intersection_A_B_l442_44205

def A : Set ℝ := { x | |x| > 1 }
def B : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | 1 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l442_44205


namespace NUMINAMATH_GPT_expected_points_A_correct_prob_A_B_same_points_correct_l442_44272

-- Conditions
def game_is_independent := true

def prob_A_B_win := 2/5
def prob_A_B_draw := 1/5

def prob_A_C_win := 1/3
def prob_A_C_draw := 1/3

def prob_B_C_win := 1/2
def prob_B_C_draw := 1/6

noncomputable def prob_A_B_lose := 1 - prob_A_B_win - prob_A_B_draw
noncomputable def prob_A_C_lose := 1 - prob_A_C_win - prob_A_C_draw
noncomputable def prob_B_C_lose := 1 - prob_B_C_win - prob_B_C_draw

noncomputable def expected_points_A : ℚ := 0 * (prob_A_B_lose * prob_A_C_lose)        /- P(ξ=0) = 2/15 -/
                                       + 1 * ((prob_A_B_draw * prob_A_C_lose) +
                                              (prob_A_B_lose * prob_A_C_draw))        /- P(ξ=1) = 1/5 -/
                                       + 2 * (prob_A_B_draw * prob_A_C_draw)         /- P(ξ=2) = 1/15 -/
                                       + 3 * ((prob_A_B_win * prob_A_C_lose) + 
                                              (prob_A_B_win * prob_A_C_draw) + 
                                              (prob_A_C_win * prob_A_B_lose))        /- P(ξ=3) = 4/15 -/
                                       + 4 * ((prob_A_B_draw * prob_A_C_win) +
                                              (prob_A_B_win * prob_A_C_win))         /- P(ξ=4) = 1/5 -/
                                       + 6 * (prob_A_B_win * prob_A_C_win)           /- P(ξ=6) = 2/15 -/

theorem expected_points_A_correct : expected_points_A = 41 / 15 :=
by
  sorry

noncomputable def prob_A_B_same_points: ℚ := ((prob_A_B_draw * prob_A_C_lose) * prob_B_C_lose)  /- both 1 point -/
                                            + ((prob_A_B_draw * prob_A_C_draw) * prob_B_C_draw)/- both 2 points -/
                                            + ((prob_A_B_win * prob_B_C_win) * prob_A_C_lose)  /- both 3 points -/
                                            + ((prob_A_B_win * prob_A_C_lose) * prob_B_C_win)  /- both 3 points -/
                                            + ((prob_A_B_draw * prob_A_C_win) * prob_B_C_win)  /- both 4 points -/

theorem prob_A_B_same_points_correct : prob_A_B_same_points = 8 / 45 :=
by
  sorry

end NUMINAMATH_GPT_expected_points_A_correct_prob_A_B_same_points_correct_l442_44272


namespace NUMINAMATH_GPT_eval_expression_l442_44275

theorem eval_expression : abs (-6) - (-4) + (-7) = 3 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l442_44275


namespace NUMINAMATH_GPT_find_an_l442_44248

def sequence_sum (k : ℝ) (n : ℕ) : ℝ :=
  k * n ^ 2 + n

def term_of_sequence (k : ℝ) (n : ℕ) (S_n : ℝ) (S_nm1 : ℝ) : ℝ :=
  S_n - S_nm1

theorem find_an (k : ℝ) (n : ℕ) (h₁ : n > 0) :
  term_of_sequence k n (sequence_sum k n) (sequence_sum k (n - 1)) = 2 * k * n - k + 1 :=
by
  sorry

end NUMINAMATH_GPT_find_an_l442_44248


namespace NUMINAMATH_GPT_inverse_var_y_l442_44293

theorem inverse_var_y (k : ℝ) (y x : ℝ)
  (h1 : 5 * y = k / x^2)
  (h2 : y = 16) (h3 : x = 1) (h4 : k = 80) :
  y = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inverse_var_y_l442_44293


namespace NUMINAMATH_GPT_brendas_age_l442_44239

theorem brendas_age (A B J : ℕ) 
  (h1 : A = 4 * B) 
  (h2 : J = B + 8) 
  (h3 : A = J) 
: B = 8 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_brendas_age_l442_44239


namespace NUMINAMATH_GPT_min_value_x_plus_y_l442_44279

theorem min_value_x_plus_y {x y : ℝ} (hx : 0 < x) (hy : 0 < y) 
  (h : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_x_plus_y_l442_44279
