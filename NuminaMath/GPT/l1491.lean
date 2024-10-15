import Mathlib

namespace NUMINAMATH_GPT_bakery_made_muffins_l1491_149130

-- Definitions based on conditions
def muffins_per_box : ℕ := 5
def available_boxes : ℕ := 10
def additional_boxes_needed : ℕ := 9

-- Theorem statement
theorem bakery_made_muffins :
  (available_boxes * muffins_per_box) + (additional_boxes_needed * muffins_per_box) = 95 := 
by
  sorry

end NUMINAMATH_GPT_bakery_made_muffins_l1491_149130


namespace NUMINAMATH_GPT_inverse_g_neg1_l1491_149167

noncomputable def g (c d x : ℝ) : ℝ := 1 / (c * x + d)

theorem inverse_g_neg1 (c d : ℝ) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ y : ℝ, g c d y = -1 ∧ y = (-1 - d) / c := 
by
  unfold g
  sorry

end NUMINAMATH_GPT_inverse_g_neg1_l1491_149167


namespace NUMINAMATH_GPT_prob_even_heads_40_l1491_149124

noncomputable def probability_even_heads (n : ℕ) : ℚ :=
  if n = 0 then 1 else
  (1/2) * (1 + (2/5) ^ n)

theorem prob_even_heads_40 :
  probability_even_heads 40 = 1/2 * (1 + (2/5) ^ 40) :=
by {
  sorry
}

end NUMINAMATH_GPT_prob_even_heads_40_l1491_149124


namespace NUMINAMATH_GPT_expand_expression_l1491_149107

variable (x y : ℝ)

theorem expand_expression :
  12 * (3 * x + 4 * y - 2) = 36 * x + 48 * y - 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1491_149107


namespace NUMINAMATH_GPT_total_price_correct_l1491_149105

-- Definitions based on given conditions
def basic_computer_price : ℝ := 2125
def enhanced_computer_price : ℝ := 2125 + 500
def printer_price (P : ℝ) := P = 1/8 * (enhanced_computer_price + P)

-- Statement to prove the total price of the basic computer and printer
theorem total_price_correct (P : ℝ) (h : printer_price P) : 
  basic_computer_price + P = 2500 :=
by
  sorry

end NUMINAMATH_GPT_total_price_correct_l1491_149105


namespace NUMINAMATH_GPT_crayons_left_l1491_149184

theorem crayons_left (initial_crayons erasers_left more_crayons_than_erasers : ℕ)
    (H1 : initial_crayons = 531)
    (H2 : erasers_left = 38)
    (H3 : more_crayons_than_erasers = 353) :
    (initial_crayons - (initial_crayons - (erasers_left + more_crayons_than_erasers)) = 391) :=
by 
  sorry

end NUMINAMATH_GPT_crayons_left_l1491_149184


namespace NUMINAMATH_GPT_chameleons_cannot_all_turn_to_single_color_l1491_149161

theorem chameleons_cannot_all_turn_to_single_color
  (W : ℕ) (B : ℕ)
  (hW : W = 20)
  (hB : B = 25)
  (h_interaction: ∀ t : ℕ, ∃ W' B' : ℕ,
    W' + B' = W + B ∧
    (W - B) % 3 = (W' - B') % 3) :
  ∀ t : ℕ, (W - B) % 3 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_chameleons_cannot_all_turn_to_single_color_l1491_149161


namespace NUMINAMATH_GPT_margarita_vs_ricciana_l1491_149109

-- Ricciana's distances
def ricciana_run : ℕ := 20
def ricciana_jump : ℕ := 4
def ricciana_total : ℕ := ricciana_run + ricciana_jump

-- Margarita's distances
def margarita_run : ℕ := 18
def margarita_jump : ℕ := (2 * ricciana_jump) - 1
def margarita_total : ℕ := margarita_run + margarita_jump

-- Statement to prove Margarita ran and jumped 1 more foot than Ricciana
theorem margarita_vs_ricciana : margarita_total = ricciana_total + 1 := by
  sorry

end NUMINAMATH_GPT_margarita_vs_ricciana_l1491_149109


namespace NUMINAMATH_GPT_equilateral_triangle_lines_l1491_149156

-- Define the properties of an equilateral triangle
structure EquilateralTriangle :=
(sides_length : ℝ) -- All sides are of equal length
(angle : ℝ := 60)  -- All internal angles are 60 degrees

-- Define the concept that altitudes, medians, and angle bisectors coincide
structure CoincidingLines (T : EquilateralTriangle) :=
(altitude : T.angle = 60)
(median : T.angle = 60)
(angle_bisector : T.angle = 60)

-- Define a statement that proves the number of distinct lines in the equilateral triangle
theorem equilateral_triangle_lines (T : EquilateralTriangle) (L : CoincidingLines T) :  
  -- The total number of distinct lines consisting of altitudes, medians, and angle bisectors
  (3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_lines_l1491_149156


namespace NUMINAMATH_GPT_car_speed_l1491_149177

theorem car_speed (distance time : ℝ) (h₁ : distance = 50) (h₂ : time = 5) : (distance / time) = 10 :=
by
  rw [h₁, h₂]
  norm_num

end NUMINAMATH_GPT_car_speed_l1491_149177


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l1491_149155

theorem perfect_square_trinomial_m (m : ℤ) :
  ∀ y : ℤ, ∃ a : ℤ, (y^2 - m * y + 1 = (y + a) ^ 2) ∨ (y^2 - m * y + 1 = (y - a) ^ 2) → (m = 2 ∨ m = -2) :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l1491_149155


namespace NUMINAMATH_GPT_project_inflation_cost_increase_l1491_149195

theorem project_inflation_cost_increase :
  let original_lumber_cost := 450
  let original_nails_cost := 30
  let original_fabric_cost := 80
  let lumber_inflation := 0.2
  let nails_inflation := 0.1
  let fabric_inflation := 0.05
  
  let new_lumber_cost := original_lumber_cost * (1 + lumber_inflation)
  let new_nails_cost := original_nails_cost * (1 + nails_inflation)
  let new_fabric_cost := original_fabric_cost * (1 + fabric_inflation)
  
  let total_increased_cost := (new_lumber_cost - original_lumber_cost) 
                            + (new_nails_cost - original_nails_cost) 
                            + (new_fabric_cost - original_fabric_cost)
  total_increased_cost = 97 := sorry

end NUMINAMATH_GPT_project_inflation_cost_increase_l1491_149195


namespace NUMINAMATH_GPT_find_b_for_perpendicular_lines_l1491_149168

theorem find_b_for_perpendicular_lines:
  (∃ b : ℝ, ∀ (x y : ℝ), (3 * x + y - 5 = 0) ∧ (b * x + y + 2 = 0) → b = -1/3) :=
by
  sorry

end NUMINAMATH_GPT_find_b_for_perpendicular_lines_l1491_149168


namespace NUMINAMATH_GPT_earnings_total_l1491_149198

-- Define the earnings for each day based on given conditions
def Monday_earnings : ℝ := 0.20 * 10 * 3
def Tuesday_earnings : ℝ := 0.25 * 12 * 4
def Wednesday_earnings : ℝ := 0.10 * 15 * 5
def Thursday_earnings : ℝ := 0.15 * 8 * 6
def Friday_earnings : ℝ := 0.30 * 20 * 2

-- Compute total earnings over the five days
def total_earnings : ℝ :=
  Monday_earnings + Tuesday_earnings + Wednesday_earnings + Thursday_earnings + Friday_earnings

-- Lean statement to prove the total earnings
theorem earnings_total :
  total_earnings = 44.70 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_earnings_total_l1491_149198


namespace NUMINAMATH_GPT_union_of_A_and_B_l1491_149131

variables (A B : Set ℤ)
variable (a : ℤ)
theorem union_of_A_and_B : (A = {4, a^2}) → (B = {a-6, 1+a, 9}) → (A ∩ B = {9}) → (A ∪ B = {-9, -2, 4, 9}) :=
by
  intros hA hB hInt
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1491_149131


namespace NUMINAMATH_GPT_reciprocal_neg_six_l1491_149193

-- Define the concept of reciprocal
def reciprocal (a : ℤ) (h : a ≠ 0) : ℚ := 1 / a

theorem reciprocal_neg_six : reciprocal (-6) (by norm_num) = -1 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_reciprocal_neg_six_l1491_149193


namespace NUMINAMATH_GPT_symmetry_center_l1491_149113

theorem symmetry_center {φ : ℝ} (hφ : |φ| < Real.pi / 2) (h : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ x : ℝ, 2 * Real.sin (2 * x + φ) = 2 * Real.sin (- (2 * x + φ)) ∧ x = -Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_symmetry_center_l1491_149113


namespace NUMINAMATH_GPT_abby_and_damon_weight_l1491_149101

variables {a b c d : ℝ}

theorem abby_and_damon_weight (h1 : a + b = 260) (h2 : b + c = 245) 
(h3 : c + d = 270) (h4 : a + c = 220) : a + d = 285 := 
by 
  sorry

end NUMINAMATH_GPT_abby_and_damon_weight_l1491_149101


namespace NUMINAMATH_GPT_find_possible_y_values_l1491_149110

noncomputable def validYValues (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) : Set ℝ :=
  { y | y = (x - 3)^2 * (x + 4) / (2 * x - 4) }

theorem find_possible_y_values (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 90) :
  validYValues x hx = {39, 6} :=
sorry

end NUMINAMATH_GPT_find_possible_y_values_l1491_149110


namespace NUMINAMATH_GPT_amoeba_count_14_l1491_149119

noncomputable def amoeba_count (day : ℕ) : ℕ :=
  if day = 1 then 1
  else if day = 2 then 2
  else 2^(day - 3) * 5

theorem amoeba_count_14 : amoeba_count 14 = 10240 := by
  sorry

end NUMINAMATH_GPT_amoeba_count_14_l1491_149119


namespace NUMINAMATH_GPT_vampire_conversion_l1491_149186

theorem vampire_conversion (x : ℕ) 
  (h_population : village_population = 300)
  (h_initial_vampires : initial_vampires = 2)
  (h_two_nights_vampires : 2 + 2 * x + x * (2 + 2 * x) = 72) :
  x = 5 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_vampire_conversion_l1491_149186


namespace NUMINAMATH_GPT_range_of_y_l1491_149129

noncomputable def y (x : ℝ) : ℝ := (Real.log x / Real.log 2 + 2) * (2 * (Real.log x / (2 * Real.log 2)) - 4)

theorem range_of_y :
  (1 ≤ x ∧ x ≤ 8) →
  (∀ t : ℝ, t = Real.log x / Real.log 2 → y x = t^2 - 2 * t - 8 ∧ 0 ≤ t ∧ t ≤ 3) →
  ∃ ymin ymax, (ymin ≤ y x ∧ y x ≤ ymax) ∧ ymin = -9 ∧ ymax = -5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_y_l1491_149129


namespace NUMINAMATH_GPT_find_total_amount_l1491_149134

noncomputable def total_amount (a b c : ℕ) : Prop :=
  a = 3 * b ∧ b = c + 25 ∧ b = 134 ∧ a + b + c = 645

theorem find_total_amount : ∃ a b c, total_amount a b c :=
by
  sorry

end NUMINAMATH_GPT_find_total_amount_l1491_149134


namespace NUMINAMATH_GPT_computation_l1491_149150

theorem computation :
  ( ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * 
    ( (7^3 - 1) / (7^3 + 1) ) * ( (8^3 - 1) / (8^3 + 1) ) 
  ) = (73 / 312) :=
by
  sorry

end NUMINAMATH_GPT_computation_l1491_149150


namespace NUMINAMATH_GPT_larry_channels_l1491_149143

-- Initial conditions
def init_channels : ℕ := 150
def channels_taken_away : ℕ := 20
def channels_replaced : ℕ := 12
def channels_reduce_request : ℕ := 10
def sports_package : ℕ := 8
def supreme_sports_package : ℕ := 7

-- Calculation representing the overall change step-by-step
theorem larry_channels : 
  init_channels - channels_taken_away + channels_replaced - channels_reduce_request + sports_package + supreme_sports_package = 147 :=
by sorry

end NUMINAMATH_GPT_larry_channels_l1491_149143


namespace NUMINAMATH_GPT_choir_average_age_l1491_149128

theorem choir_average_age :
  let num_females := 10
  let avg_age_females := 32
  let num_males := 18
  let avg_age_males := 35
  let num_people := num_females + num_males
  let sum_ages_females := avg_age_females * num_females
  let sum_ages_males := avg_age_males * num_males
  let total_sum_ages := sum_ages_females + sum_ages_males
  let avg_age := (total_sum_ages : ℚ) / num_people
  avg_age = 33.92857 := by
  sorry

end NUMINAMATH_GPT_choir_average_age_l1491_149128


namespace NUMINAMATH_GPT_find_f_neg2_l1491_149152

noncomputable def f : ℝ → ℝ := sorry

axiom cond1 : ∀ a b : ℝ, f (a + b) = f a * f b
axiom cond2 : ∀ x : ℝ, f x > 0
axiom cond3 : f 1 = 1 / 3

theorem find_f_neg2 : f (-2) = 9 := sorry

end NUMINAMATH_GPT_find_f_neg2_l1491_149152


namespace NUMINAMATH_GPT_deadlift_weight_loss_is_200_l1491_149170

def initial_squat : ℕ := 700
def initial_bench : ℕ := 400
def initial_deadlift : ℕ := 800
def lost_squat_percent : ℕ := 30
def new_total : ℕ := 1490

theorem deadlift_weight_loss_is_200 : initial_deadlift - (new_total - ((initial_squat * (100 - lost_squat_percent)) / 100 + initial_bench)) = 200 :=
by
  sorry

end NUMINAMATH_GPT_deadlift_weight_loss_is_200_l1491_149170


namespace NUMINAMATH_GPT_uranus_appears_7_minutes_after_6AM_l1491_149187

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end NUMINAMATH_GPT_uranus_appears_7_minutes_after_6AM_l1491_149187


namespace NUMINAMATH_GPT_regina_has_20_cows_l1491_149118

theorem regina_has_20_cows (C P : ℕ)
  (h1 : P = 4 * C)
  (h2 : 400 * P + 800 * C = 48000) :
  C = 20 :=
by
  sorry

end NUMINAMATH_GPT_regina_has_20_cows_l1491_149118


namespace NUMINAMATH_GPT_real_number_a_value_l1491_149126

open Set

variable {a : ℝ}

theorem real_number_a_value (A B : Set ℝ) (hA : A = {-1, 1, 3}) (hB : B = {a + 2, a^2 + 4}) (hAB : A ∩ B = {3}) : a = 1 := 
by 
-- Step proof will be here
sorry

end NUMINAMATH_GPT_real_number_a_value_l1491_149126


namespace NUMINAMATH_GPT_total_water_heaters_l1491_149115

-- Define the conditions
variables (W C : ℕ) -- W: capacity of Wallace's water heater, C: capacity of Catherine's water heater
variable (wallace_3over4_full : W = 40 ∧ W * 3 / 4 ∧ C = W / 2 ∧ C * 3 / 4)

-- The proof problem
theorem total_water_heaters (wallace_3over4_full : W = 40 ∧ (W * 3 / 4 = 30) ∧ C = W / 2 ∧ (C * 3 / 4 = 15)) : W * 3 / 4 + C * 3 / 4 = 45 :=
sorry

end NUMINAMATH_GPT_total_water_heaters_l1491_149115


namespace NUMINAMATH_GPT_sum_lent_is_3000_l1491_149136

noncomputable def principal_sum (P : ℕ) : Prop :=
  let R := 5
  let T := 5
  let SI := (P * R * T) / 100
  SI = P - 2250

theorem sum_lent_is_3000 : ∃ (P : ℕ), principal_sum P ∧ P = 3000 :=
by
  use 3000
  unfold principal_sum
  -- The following are the essential parts
  sorry

end NUMINAMATH_GPT_sum_lent_is_3000_l1491_149136


namespace NUMINAMATH_GPT_a3_eq_5_l1491_149149

-- Define the geometric sequence and its properties
variables {a : ℕ → ℝ} {q : ℝ}

-- Assumptions
def geom_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n : ℕ, a (n + 1) = a 1 * (q ^ n)
axiom a1_pos : a 1 > 0
axiom a2a4_eq_25 : a 2 * a 4 = 25
axiom geom : geom_seq a q

-- Statement to prove
theorem a3_eq_5 : a 3 = 5 :=
by sorry

end NUMINAMATH_GPT_a3_eq_5_l1491_149149


namespace NUMINAMATH_GPT_smaller_successive_number_l1491_149173

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 9506) : n = 97 :=
sorry

end NUMINAMATH_GPT_smaller_successive_number_l1491_149173


namespace NUMINAMATH_GPT_ming_wins_inequality_l1491_149178

variables (x : ℕ)

def remaining_distance (x : ℕ) : ℕ := 10000 - 200 * x
def ming_remaining_distance (x : ℕ) : ℕ := remaining_distance x - 200

-- Ensure that Xiao Ming's winning inequality holds:
theorem ming_wins_inequality (h1 : 0 < x) :
  (ming_remaining_distance x) / 250 > (remaining_distance x) / 300 :=
sorry

end NUMINAMATH_GPT_ming_wins_inequality_l1491_149178


namespace NUMINAMATH_GPT_cos_alpha_value_l1491_149171

theorem cos_alpha_value (α : ℝ) (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : Real.cos α = 1 / 5 :=
sorry

end NUMINAMATH_GPT_cos_alpha_value_l1491_149171


namespace NUMINAMATH_GPT_base8_to_decimal_l1491_149127

theorem base8_to_decimal (n : ℕ) (h : n = 54321) : 
  (5 * 8^4 + 4 * 8^3 + 3 * 8^2 + 2 * 8^1 + 1 * 8^0) = 22737 := 
by
  sorry

end NUMINAMATH_GPT_base8_to_decimal_l1491_149127


namespace NUMINAMATH_GPT_five_digit_numbers_with_4_or_5_l1491_149108

theorem five_digit_numbers_with_4_or_5 : 
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  total_five_digit - without_4_or_5 = 61328 :=
by
  let total_five_digit := 99999 - 10000 + 1
  let without_4_or_5 := 7 * 8^4
  have h : total_five_digit - without_4_or_5 = 61328 := by sorry
  exact h

end NUMINAMATH_GPT_five_digit_numbers_with_4_or_5_l1491_149108


namespace NUMINAMATH_GPT_simplify_and_evaluate_expr_l1491_149121

noncomputable def a : ℝ := 3 + Real.sqrt 5
noncomputable def b : ℝ := 3 - Real.sqrt 5

theorem simplify_and_evaluate_expr : 
  (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expr_l1491_149121


namespace NUMINAMATH_GPT_f_periodic_analytic_expression_f_distinct_real_roots_l1491_149163

noncomputable def f (x : ℝ) (k : ℤ) : ℝ := (x - 2 * k)^2

def I_k (k : ℤ) : Set ℝ := { x | 2 * k - 1 < x ∧ x ≤ 2 * k + 1 }

def M_k (k : ℕ) : Set ℝ := { a | 0 < a ∧ a ≤ 1 / (2 * ↑k + 1) }

theorem f_periodic (x : ℝ) (k : ℤ) : f x k = f (x - 2 * k) 0 := by
  sorry

theorem analytic_expression_f (x : ℝ) (k : ℤ) (hx : x ∈ I_k k) : f x k = (x - 2 * k)^2 := by
  sorry

theorem distinct_real_roots (k : ℕ) (a : ℝ) (h : a ∈ M_k k) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ I_k k ∧ x2 ∈ I_k k ∧ f x1 k = a * x1 ∧ f x2 k = a * x2 := by
  sorry

end NUMINAMATH_GPT_f_periodic_analytic_expression_f_distinct_real_roots_l1491_149163


namespace NUMINAMATH_GPT_equiv_or_neg_equiv_l1491_149142

theorem equiv_or_neg_equiv (x y : ℤ) (h : (x^2) % 239 = (y^2) % 239) :
  (x % 239 = y % 239) ∨ (x % 239 = (-y) % 239) :=
by
  sorry

end NUMINAMATH_GPT_equiv_or_neg_equiv_l1491_149142


namespace NUMINAMATH_GPT_second_month_interest_l1491_149165

def compounded_interest (initial_loan : ℝ) (rate_per_month : ℝ) : ℝ :=
  initial_loan * rate_per_month

theorem second_month_interest :
  let initial_loan := 200
  let rate_per_month := 0.10
  compounded_interest (initial_loan + compounded_interest initial_loan rate_per_month) rate_per_month = 22 :=
by
  sorry

end NUMINAMATH_GPT_second_month_interest_l1491_149165


namespace NUMINAMATH_GPT_greatest_number_of_consecutive_integers_sum_to_91_l1491_149160

theorem greatest_number_of_consecutive_integers_sum_to_91 :
  ∃ N, (∀ (a : ℤ), (N : ℕ) > 0 → (N * (2 * a + N - 1) = 182)) ∧ (N = 182) :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_number_of_consecutive_integers_sum_to_91_l1491_149160


namespace NUMINAMATH_GPT_polynomial_divisibility_l1491_149140

theorem polynomial_divisibility (A B : ℝ)
  (h: ∀ (x : ℂ), x^2 + x + 1 = 0 → x^104 + A * x^3 + B * x = 0) :
  A + B = 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l1491_149140


namespace NUMINAMATH_GPT_fraction_of_men_married_is_two_thirds_l1491_149145

-- Define the total number of faculty members
def total_faculty_members : ℕ := 100

-- Define the number of women as 70% of the faculty members
def women : ℕ := (70 * total_faculty_members) / 100

-- Define the number of men as 30% of the faculty members
def men : ℕ := (30 * total_faculty_members) / 100

-- Define the number of married faculty members as 40% of the faculty members
def married_faculty : ℕ := (40 * total_faculty_members) / 100

-- Define the number of single men as 1/3 of the men
def single_men : ℕ := men / 3

-- Define the number of married men as 2/3 of the men
def married_men : ℕ := (2 * men) / 3

-- Define the fraction of men who are married
def fraction_married_men : ℚ := married_men / men

-- The proof statement
theorem fraction_of_men_married_is_two_thirds : fraction_married_men = 2 / 3 := 
by sorry

end NUMINAMATH_GPT_fraction_of_men_married_is_two_thirds_l1491_149145


namespace NUMINAMATH_GPT_zinc_to_copper_ratio_l1491_149176

theorem zinc_to_copper_ratio (total_weight zinc_weight copper_weight : ℝ) 
  (h1 : total_weight = 64) 
  (h2 : zinc_weight = 28.8) 
  (h3 : copper_weight = total_weight - zinc_weight) : 
  (zinc_weight / 0.4) / (copper_weight / 0.4) = 9 / 11 :=
by
  sorry

end NUMINAMATH_GPT_zinc_to_copper_ratio_l1491_149176


namespace NUMINAMATH_GPT_solve_inequality_l1491_149180

def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

def given_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x >= 0 → f x = x^3 - 8

theorem solve_inequality (f : ℝ → ℝ) (h_even : even_function f) (h_given : given_function f) :
  {x | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1491_149180


namespace NUMINAMATH_GPT_walk_fraction_correct_l1491_149146

def bus_fraction := 1/3
def automobile_fraction := 1/5
def bicycle_fraction := 1/8
def metro_fraction := 1/15

def total_transport_fraction := bus_fraction + automobile_fraction + bicycle_fraction + metro_fraction

def walk_fraction := 1 - total_transport_fraction

theorem walk_fraction_correct : walk_fraction = 11/40 := by
  sorry

end NUMINAMATH_GPT_walk_fraction_correct_l1491_149146


namespace NUMINAMATH_GPT_prove_seq_formula_l1491_149139

noncomputable def seq (a : ℕ → ℝ) : ℕ → ℝ
| 0     => 1
| 1     => 5
| n + 2 => (2 * (seq a (n + 1))^2 - 3 * (seq a (n + 1)) - 9) / (2 * (seq a n))

theorem prove_seq_formula : ∀ (n : ℕ), seq a n = 2^(n + 2) - 3 :=
by
  sorry  -- Proof not needed for the mathematical translation

end NUMINAMATH_GPT_prove_seq_formula_l1491_149139


namespace NUMINAMATH_GPT_no_valid_arithmetic_operation_l1491_149158

-- Definition for arithmetic operations
inductive Operation
| div : Operation
| mul : Operation
| add : Operation
| sub : Operation

open Operation

-- Given conditions
def equation (op : Operation) : Prop :=
  match op with
  | div => (8 / 2) + 5 - (3 - 2) = 12
  | mul => (8 * 2) + 5 - (3 - 2) = 12
  | add => (8 + 2) + 5 - (3 - 2) = 12
  | sub => (8 - 2) + 5 - (3 - 2) = 12

-- Statement to prove
theorem no_valid_arithmetic_operation : ∀ op : Operation, ¬ equation op := by
  sorry

end NUMINAMATH_GPT_no_valid_arithmetic_operation_l1491_149158


namespace NUMINAMATH_GPT_police_coverage_l1491_149114

-- Define the intersections and streets
inductive Intersection : Type
| A | B | C | D | E | F | G | H | I | J | K

open Intersection

-- Define each street as a set of intersections
def horizontal_streets : List (List Intersection) :=
  [[A, B, C, D], [E, F, G], [H, I, J, K]]

def vertical_streets : List (List Intersection) :=
  [[A, E, H], [B, F, I], [D, G, J]]

def diagonal_streets : List (List Intersection) :=
  [[H, F, C], [C, G, K]]

def all_streets : List (List Intersection) :=
  horizontal_streets ++ vertical_streets ++ diagonal_streets

-- Define the set of police officers' placements
def police_officers : List Intersection := [B, G, H]

-- Check if each street is covered by at least one police officer
def is_covered (street : List Intersection) (officers : List Intersection) : Prop :=
  ∃ i, i ∈ street ∧ i ∈ officers

-- Define the proof problem statement
theorem police_coverage :
  ∀ street ∈ all_streets, is_covered street police_officers :=
by sorry

end NUMINAMATH_GPT_police_coverage_l1491_149114


namespace NUMINAMATH_GPT_base_sum_correct_l1491_149175

theorem base_sum_correct :
  let C := 12
  let a := 3 * 9^2 + 5 * 9^1 + 7 * 9^0
  let b := 4 * 13^2 + C * 13^1 + 2 * 13^0
  a + b = 1129 :=
by
  sorry

end NUMINAMATH_GPT_base_sum_correct_l1491_149175


namespace NUMINAMATH_GPT_range_of_m_l1491_149169

noncomputable def p (m : ℝ) : Prop := ∀ x : ℝ, -m * x ^ 2 + 2 * x - m > 0
noncomputable def q (m : ℝ) : Prop := ∀ x > 0, (4 / x + x - m + 1) > 2

theorem range_of_m : 
  (∃ (m : ℝ), (p m ∨ q m) ∧ ¬(p m ∧ q m)) → (∃ (m : ℝ), -1 ≤ m ∧ m < 3) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_range_of_m_l1491_149169


namespace NUMINAMATH_GPT_distinct_values_count_l1491_149192

noncomputable def f : ℕ → ℤ := sorry -- The actual function definition is not required

theorem distinct_values_count :
  ∃! n, n = 3 ∧ 
  (∀ x : ℕ, 
    (f x = f (x - 1) + f (x + 1) ∧ 
     (x = 1 → f x = 2009) ∧ 
     (x = 3 → f x = 0))) := 
sorry

end NUMINAMATH_GPT_distinct_values_count_l1491_149192


namespace NUMINAMATH_GPT_apples_sold_by_noon_l1491_149135

theorem apples_sold_by_noon 
  (k g c l : ℕ) 
  (hk : k = 23) 
  (hg : g = 37) 
  (hc : c = 14) 
  (hl : l = 38) :
  k + g + c - l = 36 := 
by
  -- k = 23
  -- g = 37
  -- c = 14
  -- l = 38
  -- k + g + c - l = 36

  sorry

end NUMINAMATH_GPT_apples_sold_by_noon_l1491_149135


namespace NUMINAMATH_GPT_yanni_paintings_l1491_149120

theorem yanni_paintings
  (total_area : ℤ)
  (painting1 : ℕ → ℤ × ℤ)
  (painting2 : ℤ × ℤ)
  (painting3 : ℤ × ℤ)
  (num_paintings : ℕ) :
  total_area = 200
  → painting1 1 = (5, 5)
  → painting1 2 = (5, 5)
  → painting1 3 = (5, 5)
  → painting2 = (10, 8)
  → painting3 = (5, 9)
  → num_paintings = 5 := 
by
  sorry

end NUMINAMATH_GPT_yanni_paintings_l1491_149120


namespace NUMINAMATH_GPT_profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l1491_149185

theorem profitability_when_x_gt_94 (A : ℕ) (x : ℕ) (hx : x > 94) : 
  1/3 * x * A - (2/3 * x * (A / 2)) = 0 := 
sorry

theorem daily_profit_when_x_le_94 (A : ℕ) (x : ℕ) (hx : 1 ≤ x ∧ x ≤ 94) : 
  ∃ T : ℕ, T = (x - 3 * x / (2 * (96 - x))) * A := 
sorry

theorem max_profit_occurs_at_84 (A : ℕ) : 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 94 ∧ 
  (∀ y : ℕ, 1 ≤ y ∧ y ≤ 94 → 
    (y - 3 * y / (2 * (96 - y))) * A ≤ (84 - 3 * 84 / (2 * (96 - 84))) * A) := 
sorry

end NUMINAMATH_GPT_profitability_when_x_gt_94_daily_profit_when_x_le_94_max_profit_occurs_at_84_l1491_149185


namespace NUMINAMATH_GPT_solve_equation_l1491_149182

theorem solve_equation (x : ℝ) : (x + 3) * (x - 1) = 12 ↔ (x = -5 ∨ x = 3) := sorry

end NUMINAMATH_GPT_solve_equation_l1491_149182


namespace NUMINAMATH_GPT_smallest_value_l1491_149181

theorem smallest_value : 54 * Real.sqrt 3 < 144 ∧ 54 * Real.sqrt 3 < 108 * Real.sqrt 6 - 108 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_smallest_value_l1491_149181


namespace NUMINAMATH_GPT_total_bees_in_colony_l1491_149111

def num_bees_in_hive_after_changes (initial_bees : ℕ) (bees_in : ℕ) (bees_out : ℕ) : ℕ :=
  initial_bees + bees_in - bees_out

theorem total_bees_in_colony :
  let hive1 := num_bees_in_hive_after_changes 45 12 8
  let hive2 := num_bees_in_hive_after_changes 60 15 20
  let hive3 := num_bees_in_hive_after_changes 75 10 5
  hive1 + hive2 + hive3 = 184 :=
by
  sorry

end NUMINAMATH_GPT_total_bees_in_colony_l1491_149111


namespace NUMINAMATH_GPT_min_positive_d_l1491_149125

theorem min_positive_d (a b t d : ℤ) (h1 : 3 * t = 2 * a + 2 * b + 2016)
                                       (h2 : t - a = d)
                                       (h3 : t - b = 2 * d)
                                       (h4 : 2 * a + 2 * b > 0) :
    ∃ d : ℤ, d > 0 ∧ (505 ≤ d ∧ ∀ e : ℤ, e > 0 → 3 * (a + d) = 2 * (b + 2 * e) + 2016 → 505 ≤ e) := 
sorry

end NUMINAMATH_GPT_min_positive_d_l1491_149125


namespace NUMINAMATH_GPT_decreasing_function_l1491_149183

-- Define the functions
noncomputable def fA (x : ℝ) : ℝ := 3^x
noncomputable def fB (x : ℝ) : ℝ := Real.logb 0.5 x
noncomputable def fC (x : ℝ) : ℝ := Real.sqrt x
noncomputable def fD (x : ℝ) : ℝ := 1/x

-- Define the domains
def domainA : Set ℝ := Set.univ
def domainB : Set ℝ := {x | x > 0}
def domainC : Set ℝ := {x | x ≥ 0}
def domainD : Set ℝ := {x | x < 0} ∪ {x | x > 0}

-- Prove that fB is the only decreasing function in its domain
theorem decreasing_function:
  (∀ x y, x ∈ domainA → y ∈ domainA → x < y → fA x > fA y) = false ∧
  (∀ x y, x ∈ domainB → y ∈ domainB → x < y → fB x > fB y) ∧
  (∀ x y, x ∈ domainC → y ∈ domainC → x < y → fC x > fC y) = false ∧
  (∀ x y, x ∈ domainD → y ∈ domainD → x < y → fD x > fD y) = false :=
  sorry

end NUMINAMATH_GPT_decreasing_function_l1491_149183


namespace NUMINAMATH_GPT_intersection_A_B_complement_l1491_149148

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ 1}
def B_complement : Set ℝ := U \ B

theorem intersection_A_B_complement : A ∩ B_complement = {x | x > 1} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_complement_l1491_149148


namespace NUMINAMATH_GPT_positive_integers_satisfy_l1491_149133

theorem positive_integers_satisfy (n : ℕ) (h1 : 25 - 5 * n > 15) : n = 1 :=
by sorry

end NUMINAMATH_GPT_positive_integers_satisfy_l1491_149133


namespace NUMINAMATH_GPT_susie_investment_l1491_149132

theorem susie_investment :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2000 ∧
  (x * 1.04 + (2000 - x) * 1.06 = 2120) → (x = 0) :=
by
  sorry

end NUMINAMATH_GPT_susie_investment_l1491_149132


namespace NUMINAMATH_GPT_roots_equality_l1491_149138

variable {α β p q : ℝ}

theorem roots_equality (h1 : α ≠ β)
    (h2 : α * α + p * α + q = 0 ∧ β * β + p * β + q = 0)
    (h3 : α^3 - α^2 * β - α * β^2 + β^3 = 0) : 
  p = 0 ∧ q < 0 :=
by 
  sorry

end NUMINAMATH_GPT_roots_equality_l1491_149138


namespace NUMINAMATH_GPT_inequality_proof_equality_conditions_l1491_149189

theorem inequality_proof
  (x y : ℝ)
  (h1 : x ≥ y)
  (h2 : y ≥ 1) :
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) ≥
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

theorem equality_conditions
  (x y : ℝ) :
  (x = y ∨ x = 1 ∨ y = 1) ↔
  (x / Real.sqrt (x + y) + y / Real.sqrt (y + 1) + 1 / Real.sqrt (x + 1) =
   y / Real.sqrt (x + y) + x / Real.sqrt (x + 1) + 1 / Real.sqrt (y + 1)) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_equality_conditions_l1491_149189


namespace NUMINAMATH_GPT_num_male_students_selected_l1491_149104

def total_students := 220
def male_students := 60
def selected_female_students := 32

def selected_male_students (total_students male_students selected_female_students : Nat) : Nat :=
  (selected_female_students * male_students) / (total_students - male_students)

theorem num_male_students_selected : selected_male_students total_students male_students selected_female_students = 12 := by
  unfold selected_male_students
  sorry

end NUMINAMATH_GPT_num_male_students_selected_l1491_149104


namespace NUMINAMATH_GPT_probability_no_physics_and_chemistry_l1491_149106

-- Define the probabilities for the conditions
def P_physics : ℚ := 5 / 8
def P_no_physics : ℚ := 1 - P_physics
def P_chemistry_given_no_physics : ℚ := 2 / 3

-- Define the theorem we want to prove
theorem probability_no_physics_and_chemistry :
  P_no_physics * P_chemistry_given_no_physics = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_probability_no_physics_and_chemistry_l1491_149106


namespace NUMINAMATH_GPT_gcd_of_6Tn2_and_nplus1_eq_2_l1491_149153

theorem gcd_of_6Tn2_and_nplus1_eq_2 (n : ℕ) (h_pos : 0 < n) :
  Nat.gcd (6 * ((n * (n + 1) / 2)^2)) (n + 1) = 2 :=
sorry

end NUMINAMATH_GPT_gcd_of_6Tn2_and_nplus1_eq_2_l1491_149153


namespace NUMINAMATH_GPT_gasoline_price_april_l1491_149102

theorem gasoline_price_april (P₀ : ℝ) (P₁ P₂ P₃ P₄ : ℝ) (x : ℝ)
  (h₁ : P₁ = P₀ * 1.20)  -- Price after January's increase
  (h₂ : P₂ = P₁ * 0.80)  -- Price after February's decrease
  (h₃ : P₃ = P₂ * 1.25)  -- Price after March's increase
  (h₄ : P₄ = P₃ * (1 - x / 100))  -- Price after April's decrease
  (h₅ : P₄ = P₀)  -- Price at the end of April equals the initial price
  : x = 17 := 
by
  sorry

end NUMINAMATH_GPT_gasoline_price_april_l1491_149102


namespace NUMINAMATH_GPT_john_spent_half_on_fruits_and_vegetables_l1491_149141

theorem john_spent_half_on_fruits_and_vegetables (M : ℝ) (F : ℝ) 
  (spent_on_meat : ℝ) (spent_on_bakery : ℝ) (spent_on_candy : ℝ) :
  (M = 120) → 
  (spent_on_meat = (1 / 3) * M) → 
  (spent_on_bakery = (1 / 10) * M) → 
  (spent_on_candy = 8) → 
  (F * M + spent_on_meat + spent_on_bakery + spent_on_candy = M) → 
  (F = 1 / 2) := 
  by 
    sorry

end NUMINAMATH_GPT_john_spent_half_on_fruits_and_vegetables_l1491_149141


namespace NUMINAMATH_GPT_sequence_a8_l1491_149144

theorem sequence_a8 (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a n + a (n + 1)) 
  (h2 : a 7 = 120) : 
  a 8 = 194 :=
sorry

end NUMINAMATH_GPT_sequence_a8_l1491_149144


namespace NUMINAMATH_GPT_trajectory_equation_l1491_149154

-- Definitions and conditions
noncomputable def tangent_to_x_axis (M : ℝ × ℝ) := M.snd = 0
noncomputable def internally_tangent (M : ℝ × ℝ) := ∃ (r : ℝ), 0 < r ∧ M.1^2 + (M.2 - r)^2 = 4

-- The theorem stating the proof problem
theorem trajectory_equation (M : ℝ × ℝ) (h_tangent : tangent_to_x_axis M) (h_internal_tangent : internally_tangent M) :
  (∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧ M.fst^2 = 4 * (y - 1)) :=
sorry

end NUMINAMATH_GPT_trajectory_equation_l1491_149154


namespace NUMINAMATH_GPT_compare_rental_fees_l1491_149194

namespace HanfuRental

def store_A_rent_price : ℝ := 120
def store_B_rent_price : ℝ := 160
def store_A_discount : ℝ := 0.20
def store_B_discount_limit : ℕ := 6
def store_B_excess_rate : ℝ := 0.50
def x : ℕ := 40 -- number of Hanfu costumes

def y₁ (x : ℕ) : ℝ := (store_A_rent_price * (1 - store_A_discount)) * x

def y₂ (x : ℕ) : ℝ :=
  if x ≤ store_B_discount_limit then store_B_rent_price * x
  else store_B_rent_price * store_B_discount_limit + store_B_excess_rate * store_B_rent_price * (x - store_B_discount_limit)

theorem compare_rental_fees (x : ℕ) (hx : x = 40) :
  y₂ x ≤ y₁ x :=
sorry

end HanfuRental

end NUMINAMATH_GPT_compare_rental_fees_l1491_149194


namespace NUMINAMATH_GPT_max_value_of_expression_l1491_149100

-- We have three nonnegative real numbers a, b, and c,
-- such that a + b + c = 3.
def nonnegative (x : ℝ) := x ≥ 0

theorem max_value_of_expression (a b c : ℝ) (h1 : nonnegative a) (h2 : nonnegative b) (h3 : nonnegative c) (h4 : a + b + c = 3) :
  a + b^2 + c^4 ≤ 3 :=
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l1491_149100


namespace NUMINAMATH_GPT_cone_lateral_area_l1491_149147

theorem cone_lateral_area (C l r A : ℝ) (hC : C = 4 * Real.pi) (hl : l = 3) 
  (hr : 2 * Real.pi * r = 4 * Real.pi) (hA : A = Real.pi * r * l) : A = 6 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l1491_149147


namespace NUMINAMATH_GPT_H_iterated_l1491_149197

variable (H : ℝ → ℝ)

-- Conditions as hypotheses
axiom H_2 : H 2 = -4
axiom H_neg4 : H (-4) = 6
axiom H_6 : H 6 = 6

-- The theorem we want to prove
theorem H_iterated (H : ℝ → ℝ) (h1 : H 2 = -4) (h2 : H (-4) = 6) (h3 : H 6 = 6) : 
  H (H (H (H (H 2)))) = 6 := by
  sorry

end NUMINAMATH_GPT_H_iterated_l1491_149197


namespace NUMINAMATH_GPT_apples_total_l1491_149199

theorem apples_total (lexie_apples : ℕ) (tom_apples : ℕ) (h1 : lexie_apples = 12) (h2 : tom_apples = 2 * lexie_apples) : lexie_apples + tom_apples = 36 :=
by
  sorry

end NUMINAMATH_GPT_apples_total_l1491_149199


namespace NUMINAMATH_GPT_compute_expression_l1491_149103

theorem compute_expression : 42 * 52 + 48 * 42 = 4200 :=
by sorry

end NUMINAMATH_GPT_compute_expression_l1491_149103


namespace NUMINAMATH_GPT_agent_007_encryption_l1491_149190

theorem agent_007_encryption : ∃ (m n : ℕ), (0.07 : ℝ) = (1 / m : ℝ) + (1 / n : ℝ) := 
sorry

end NUMINAMATH_GPT_agent_007_encryption_l1491_149190


namespace NUMINAMATH_GPT_total_pennies_l1491_149166

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end NUMINAMATH_GPT_total_pennies_l1491_149166


namespace NUMINAMATH_GPT_max_value_of_z_l1491_149188

theorem max_value_of_z : ∀ x : ℝ, (x^2 - 14 * x + 10 ≤ 0 - 39) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_z_l1491_149188


namespace NUMINAMATH_GPT_greatest_k_inequality_l1491_149112

theorem greatest_k_inequality :
  ∃ k : ℕ, k = 13 ∧ ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → a * b * c = 1 → 
  (1 / a + 1 / b + 1 / c + k / (a + b + c + 1) ≥ 3 + k / 4) :=
sorry

end NUMINAMATH_GPT_greatest_k_inequality_l1491_149112


namespace NUMINAMATH_GPT_linda_savings_l1491_149137

theorem linda_savings (S : ℝ) (h : (1 / 2) * S = 300) : S = 600 :=
sorry

end NUMINAMATH_GPT_linda_savings_l1491_149137


namespace NUMINAMATH_GPT_income_increase_by_parental_support_l1491_149122

variables (a b c S : ℝ)

theorem income_increase_by_parental_support 
  (h1 : S = a + b + c)
  (h2 : 2 * a + b + c = 1.05 * S)
  (h3 : a + 2 * b + c = 1.15 * S) :
  (a + b + 2 * c) = 1.8 * S :=
sorry

end NUMINAMATH_GPT_income_increase_by_parental_support_l1491_149122


namespace NUMINAMATH_GPT_ages_of_Xs_sons_l1491_149162

def ages_problem (x y : ℕ) : Prop :=
x ≠ y ∧ x ≤ 10 ∧ y ≤ 10 ∧
∀ u v : ℕ, u * v = x * y → u ≤ 10 ∧ v ≤ 10 → (u, v) = (x, y) ∨ (u, v) = (y, x) ∨
(∀ z w : ℕ, z / w = x / y → z = x ∧ w = y ∨ z = y ∧ w = x → u ≠ z ∧ v ≠ w) →
(∀ a b : ℕ, a - b = (x - y) ∨ b - a = (y - x) → (x, y) = (a, b) ∨ (x, y) = (b, a))

theorem ages_of_Xs_sons : ages_problem 8 2 := 
by {
  sorry
}


end NUMINAMATH_GPT_ages_of_Xs_sons_l1491_149162


namespace NUMINAMATH_GPT_sum_of_odd_integers_l1491_149159

theorem sum_of_odd_integers (n : ℕ) (h : n * (n + 1) = 4970) : (n * n = 4900) :=
by sorry

end NUMINAMATH_GPT_sum_of_odd_integers_l1491_149159


namespace NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l1491_149123

-- Define vectors
def vec (x y : ℝ) : ℝ × ℝ := (x, y)
def OA : ℝ × ℝ := vec 3 (-4)
def OB : ℝ × ℝ := vec 6 (-3)
def OC (m : ℝ) : ℝ × ℝ := vec (2 * m) (m + 1)

-- Define the condition that AB is parallel to OC
def is_parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 = k * w.1 ∧ v.2 = k * w.2

-- Calculate AB
def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

-- The theorem to prove
theorem parallel_vectors_m_eq_neg3 (m : ℝ) :
  is_parallel AB (OC m) → m = -3 := by
  sorry

end NUMINAMATH_GPT_parallel_vectors_m_eq_neg3_l1491_149123


namespace NUMINAMATH_GPT_perimeter_of_triangle_l1491_149191

def point (x y : ℝ) := (x, y)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

noncomputable def perimeter_triangle (a b c : ℝ × ℝ) : ℝ :=
  distance a b + distance b c + distance c a

theorem perimeter_of_triangle :
  let A := point 1 2
  let B := point 6 8
  let C := point 1 5
  perimeter_triangle A B C = Real.sqrt 61 + Real.sqrt 34 + 3 :=
by
  -- proof steps can be provided here
  sorry

end NUMINAMATH_GPT_perimeter_of_triangle_l1491_149191


namespace NUMINAMATH_GPT_cone_slant_height_l1491_149157

noncomputable def slant_height (r : ℝ) (CSA : ℝ) : ℝ := CSA / (Real.pi * r)

theorem cone_slant_height : slant_height 10 628.3185307179587 = 20 :=
by
  sorry

end NUMINAMATH_GPT_cone_slant_height_l1491_149157


namespace NUMINAMATH_GPT_prove_expression_value_l1491_149172

theorem prove_expression_value (m n : ℝ) (h : m^2 + 3 * n - 1 = 2) : 2 * m^2 + 6 * n + 1 = 7 := by
  sorry

end NUMINAMATH_GPT_prove_expression_value_l1491_149172


namespace NUMINAMATH_GPT_final_price_of_coat_after_discounts_l1491_149151

def original_price : ℝ := 120
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.20

theorem final_price_of_coat_after_discounts : 
    (1 - second_discount) * (1 - first_discount) * original_price = 72 := 
by
    sorry

end NUMINAMATH_GPT_final_price_of_coat_after_discounts_l1491_149151


namespace NUMINAMATH_GPT_central_angle_of_sector_l1491_149117

-- Define the given conditions
def radius : ℝ := 10
def area : ℝ := 100

-- The statement to be proved
theorem central_angle_of_sector (α : ℝ) (h : area = (1 / 2) * α * radius ^ 2) : α = 2 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l1491_149117


namespace NUMINAMATH_GPT_tan_double_angle_l1491_149179

theorem tan_double_angle (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : Real.tan (2 * α) = 4 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1491_149179


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1491_149196

-- Define a sequence of positive terms
def is_positive_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∀ i, 0 < seq i

-- Define what it means for a sequence to be geometric
def is_geometric_sequence (seq : Fin 8 → ℝ) : Prop :=
  ∃ q > 0, q ≠ 1 ∧ ∀ i j, i < j → seq j = (q ^ (j - i : ℤ)) * seq i

-- State the theorem
theorem sufficient_but_not_necessary_condition (seq : Fin 8 → ℝ) (h_pos : is_positive_sequence seq) :
  ¬is_geometric_sequence seq → seq 0 + seq 7 < seq 3 + seq 4 ∧ 
  (seq 0 + seq 7 < seq 3 + seq 4 → ¬is_geometric_sequence seq) ∧
  (¬is_geometric_sequence seq → ¬(seq 0 + seq 7 < seq 3 + seq 4) -> ¬ is_geometric_sequence seq) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1491_149196


namespace NUMINAMATH_GPT_soccer_league_teams_l1491_149116

theorem soccer_league_teams (n : ℕ) (h : n * (n - 1) / 2 = 105) : n = 15 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_soccer_league_teams_l1491_149116


namespace NUMINAMATH_GPT_percent_problem_l1491_149164

theorem percent_problem (x : ℝ) (h : 0.3 * 0.15 * x = 18) : 0.15 * 0.3 * x = 18 :=
by
  sorry

end NUMINAMATH_GPT_percent_problem_l1491_149164


namespace NUMINAMATH_GPT_triangle_inequality_l1491_149174

variable {a b c : ℝ}

theorem triangle_inequality (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  (a / Real.sqrt (2*b^2 + 2*c^2 - a^2)) + (b / Real.sqrt (2*c^2 + 2*a^2 - b^2)) + 
  (c / Real.sqrt (2*a^2 + 2*b^2 - c^2)) ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1491_149174
