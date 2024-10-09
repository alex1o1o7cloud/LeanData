import Mathlib

namespace prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l1171_117128

-- Definitions
def groupA : List ℕ := [2, 4, 6]
def groupB : List ℕ := [3, 5]
def card_count_A : ℕ := groupA.length
def card_count_B : ℕ := groupB.length

-- Condition 1: Probability of drawing the card with number 2 from group A
def prob_draw_2_groupA : ℚ := 1 / card_count_A

-- Condition 2: Game Rule Outcomes
def is_multiple_of_3 (n : ℕ) : Bool := n % 3 == 0

def outcomes : List (ℕ × ℕ) := [(2, 3), (2, 5), (4, 3), (4, 5), (6, 3), (6, 5)]

def winning_outcomes_A : List (ℕ × ℕ) :=List.filter (λ p => is_multiple_of_3 (p.1 * p.2)) outcomes
def winning_outcomes_B : List (ℕ × ℕ) := List.filter (λ p => ¬ is_multiple_of_3 (p.1 * p.2)) outcomes

def prob_win_A : ℚ := winning_outcomes_A.length / outcomes.length
def prob_win_B : ℚ := winning_outcomes_B.length / outcomes.length

-- Proof problems
theorem prob_draw_2_groupA_is_one_third : prob_draw_2_groupA = 1 / 3 := sorry

theorem game_rule_is_unfair : prob_win_A ≠ prob_win_B := sorry

end prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l1171_117128


namespace time_between_shark_sightings_l1171_117131

def earnings_per_photo : ℕ := 15
def fuel_cost_per_hour : ℕ := 50
def hunting_hours : ℕ := 5
def expected_profit : ℕ := 200

theorem time_between_shark_sightings :
  (hunting_hours * 60) / ((expected_profit + (fuel_cost_per_hour * hunting_hours)) / earnings_per_photo) = 10 :=
by 
  sorry

end time_between_shark_sightings_l1171_117131


namespace quadratic_eq_has_real_root_l1171_117129

theorem quadratic_eq_has_real_root (a b : ℝ) :
  ¬(∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 :=
by
  sorry

end quadratic_eq_has_real_root_l1171_117129


namespace find_number_l1171_117123

theorem find_number (x : ℝ) (h : 0.7 * x = 48 + 22) : x = 100 :=
by
  sorry

end find_number_l1171_117123


namespace largest_class_students_l1171_117150

theorem largest_class_students :
  ∃ x : ℕ, (x + (x - 4) + (x - 8) + (x - 12) + (x - 16) + (x - 20) + (x - 24) +
  (x - 28) + (x - 32) + (x - 36) = 100) ∧ x = 28 :=
by
  sorry

end largest_class_students_l1171_117150


namespace total_boxes_sold_l1171_117167

-- Define the number of boxes of plain cookies
def P : ℝ := 793.375

-- Define the combined value of cookies sold
def total_value : ℝ := 1586.75

-- Define the cost per box of each type of cookie
def cost_chocolate_chip : ℝ := 1.25
def cost_plain : ℝ := 0.75

-- State the theorem to prove
theorem total_boxes_sold :
  ∃ C : ℝ, cost_chocolate_chip * C + cost_plain * P = total_value ∧ C + P = 1586.75 :=
by
  sorry

end total_boxes_sold_l1171_117167


namespace time_to_cross_first_platform_l1171_117158

-- Define the given conditions
def length_first_platform : ℕ := 140
def length_second_platform : ℕ := 250
def length_train : ℕ := 190
def time_cross_second_platform : Nat := 20
def speed := (length_train + length_second_platform) / time_cross_second_platform

-- The theorem to be proved
theorem time_to_cross_first_platform : 
  (length_train + length_first_platform) / speed = 15 :=
sorry

end time_to_cross_first_platform_l1171_117158


namespace differential_savings_l1171_117121

theorem differential_savings (income : ℝ) (tax_rate1 tax_rate2 : ℝ) 
                            (old_tax_rate_eq : tax_rate1 = 0.40) 
                            (new_tax_rate_eq : tax_rate2 = 0.33) 
                            (income_eq : income = 45000) :
    ((tax_rate1 - tax_rate2) * income) = 3150 :=
by
  rw [old_tax_rate_eq, new_tax_rate_eq, income_eq]
  norm_num

end differential_savings_l1171_117121


namespace road_trip_cost_l1171_117148

theorem road_trip_cost 
  (x : ℝ)
  (initial_cost_per_person: ℝ) 
  (redistributed_cost_per_person: ℝ)
  (cost_difference: ℝ) :
  initial_cost_per_person = x / 4 →
  redistributed_cost_per_person = x / 7 →
  cost_difference = 8 →
  initial_cost_per_person - redistributed_cost_per_person = cost_difference →
  x = 74.67 :=
by
  intro h1 h2 h3 h4
  -- starting the proof
  rw [h1, h2] at h4
  sorry

end road_trip_cost_l1171_117148


namespace geometric_sequence_common_ratio_l1171_117142

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → α)
  (q : α)
  (h1 : is_geometric_sequence a q)
  (h2 : a 3 = 6)
  (h3 : a 0 + a 1 + a 2 = 18) :
  q = 1 ∨ q = - (1 / 2) := 
sorry

end geometric_sequence_common_ratio_l1171_117142


namespace at_least_one_miss_l1171_117149

variables (p q : Prop)

-- Proposition stating the necessary and sufficient condition.
theorem at_least_one_miss : ¬(p ∧ q) ↔ (¬p ∨ ¬q) :=
by sorry

end at_least_one_miss_l1171_117149


namespace x0_y0_sum_eq_31_l1171_117136

theorem x0_y0_sum_eq_31 :
  ∃ x0 y0 : ℕ, (0 ≤ x0 ∧ x0 < 37) ∧ (0 ≤ y0 ∧ y0 < 37) ∧ 
  (2 * x0 ≡ 1 [MOD 37]) ∧ (3 * y0 ≡ 36 [MOD 37]) ∧ 
  (x0 + y0 = 31) :=
sorry

end x0_y0_sum_eq_31_l1171_117136


namespace smallest_positive_integer_divisible_l1171_117100

theorem smallest_positive_integer_divisible (n : ℕ) (h1 : 15 = 3 * 5) (h2 : 16 = 2 ^ 4) (h3 : 18 = 2 * 3 ^ 2) :
  n = Nat.lcm (Nat.lcm 15 16) 18 ↔ n = 720 :=
by
  sorry

end smallest_positive_integer_divisible_l1171_117100


namespace fuel_capacity_ratio_l1171_117155

noncomputable def oldCost : ℝ := 200
noncomputable def newCost : ℝ := 480
noncomputable def priceIncreaseFactor : ℝ := 1.20

theorem fuel_capacity_ratio (C C_new : ℝ) (h1 : newCost = C_new * oldCost * priceIncreaseFactor / C) : 
  C_new / C = 2 :=
sorry

end fuel_capacity_ratio_l1171_117155


namespace total_games_attended_l1171_117113

theorem total_games_attended 
  (games_this_month : ℕ)
  (games_last_month : ℕ)
  (games_next_month : ℕ)
  (total_games : ℕ) 
  (h : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : games_next_month = 16) 
  (htotal : total_games = 44) :
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end total_games_attended_l1171_117113


namespace min_third_side_of_right_triangle_l1171_117138

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l1171_117138


namespace trihedral_sphere_radius_l1171_117173

noncomputable def sphere_radius 
  (α r : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  : ℝ :=
r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3)

theorem trihedral_sphere_radius 
  (α r R : ℝ) 
  (hα : 0 < α ∧ α < (Real.pi / 2)) 
  (hR : R = sphere_radius α r hα) 
  : R = r * Real.sqrt ((4 * (Real.cos (α / 2)) ^ 2 + 3) / 3) :=
by
  sorry

end trihedral_sphere_radius_l1171_117173


namespace latoya_call_duration_l1171_117185

theorem latoya_call_duration
  (initial_credit remaining_credit : ℝ) (cost_per_minute : ℝ) (t : ℝ)
  (h1 : initial_credit = 30)
  (h2 : remaining_credit = 26.48)
  (h3 : cost_per_minute = 0.16)
  (h4 : initial_credit - remaining_credit = t * cost_per_minute) :
  t = 22 := 
sorry

end latoya_call_duration_l1171_117185


namespace trishul_investment_less_than_raghu_l1171_117124

noncomputable def VishalInvestment (T : ℝ) : ℝ := 1.10 * T

noncomputable def TotalInvestment (T : ℝ) (R : ℝ) : ℝ :=
  T + VishalInvestment T + R

def RaghuInvestment : ℝ := 2100

def TotalSumInvested : ℝ := 6069

theorem trishul_investment_less_than_raghu :
  ∃ T : ℝ, TotalInvestment T RaghuInvestment = TotalSumInvested → (RaghuInvestment - T) / RaghuInvestment * 100 = 10 := by
  sorry

end trishul_investment_less_than_raghu_l1171_117124


namespace find_m_l1171_117199

theorem find_m (m : ℕ) (h1 : (3 * m - 7) % 2 = 0) (h2 : 3 * m - 7 < 0) : m = 1 := 
by
  sorry

end find_m_l1171_117199


namespace intersection_two_sets_l1171_117139

theorem intersection_two_sets (M N : Set ℤ) (h1 : M = {1, 2, 3, 4}) (h2 : N = {-2, 2}) :
  M ∩ N = {2} := 
by
  sorry

end intersection_two_sets_l1171_117139


namespace find_g2_l1171_117107

-- Define the conditions of the problem
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x

-- Prove the desired value of g(2)
theorem find_g2 (g : ℝ → ℝ) (h : satisfies_condition g) : g 2 = -19 / 6 :=
by
  sorry

end find_g2_l1171_117107


namespace tenth_square_tiles_more_than_ninth_l1171_117178

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ := 2 * n - 1

-- Calculate the number of tiles used in the nth square
def tiles_count (n : ℕ) : ℕ := (side_length n) ^ 2

-- State the theorem that the tenth square requires 72 more tiles than the ninth square
theorem tenth_square_tiles_more_than_ninth : tiles_count 10 - tiles_count 9 = 72 :=
by
  sorry

end tenth_square_tiles_more_than_ninth_l1171_117178


namespace weather_forecast_minutes_l1171_117168

theorem weather_forecast_minutes 
  (total_duration : ℕ) 
  (national_news : ℕ) 
  (international_news : ℕ) 
  (sports : ℕ) 
  (advertising : ℕ) 
  (wf : ℕ) :
  total_duration = 30 →
  national_news = 12 →
  international_news = 5 →
  sports = 5 →
  advertising = 6 →
  total_duration - (national_news + international_news + sports + advertising) = wf →
  wf = 2 :=
by
  intros
  sorry

end weather_forecast_minutes_l1171_117168


namespace compute_expression_at_4_l1171_117162

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end compute_expression_at_4_l1171_117162


namespace total_people_present_l1171_117184

theorem total_people_present (A B : ℕ) 
  (h1 : 2 * A + B = 10) 
  (h2 : A + 2 * B = 14) :
  A + B = 8 :=
sorry

end total_people_present_l1171_117184


namespace term_five_eq_nine_l1171_117137

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- The sum of the first n terms of the sequence equals n^2.
axiom sum_formula : ∀ n, S n = n^2

-- Definition of the nth term in terms of the sequence sum.
def a_n (n : ℕ) : ℕ := S n - S (n - 1)

-- Goal: Prove that the 5th term, a(5), equals 9.
theorem term_five_eq_nine : a_n S 5 = 9 :=
by
  sorry

end term_five_eq_nine_l1171_117137


namespace fraction_irreducible_l1171_117144

open Nat

theorem fraction_irreducible (m n : ℕ) : Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 :=
  sorry

end fraction_irreducible_l1171_117144


namespace divisors_of_64n4_l1171_117166

theorem divisors_of_64n4 (n : ℕ) (hn : 0 < n) (hdiv : ∃ d, d = (120 * n^3) ∧ d.divisors.card = 120) : (64 * n^4).divisors.card = 375 := 
by 
  sorry

end divisors_of_64n4_l1171_117166


namespace doubled_sum_of_squares_l1171_117127

theorem doubled_sum_of_squares (a b : ℝ) : 
  2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := 
by
  sorry

end doubled_sum_of_squares_l1171_117127


namespace x_plus_y_possible_values_l1171_117151

theorem x_plus_y_possible_values (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < 20) (h4 : y < 20) (h5 : x + y + x * y = 99) : 
  x + y = 23 ∨ x + y = 18 :=
by
  sorry

end x_plus_y_possible_values_l1171_117151


namespace commission_percentage_is_4_l1171_117120

-- Define the given conditions
def commission := 12.50
def total_sales := 312.5

-- The problem is to prove the commission percentage
theorem commission_percentage_is_4 :
  (commission / total_sales) * 100 = 4 := by
  sorry

end commission_percentage_is_4_l1171_117120


namespace determine_A_plus_B_l1171_117106

theorem determine_A_plus_B :
  ∃ (A B : ℚ), ((∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → 
  (Bx - 23) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) ∧
  (A + B = 11 / 9)) :=
sorry

end determine_A_plus_B_l1171_117106


namespace system_real_solution_conditions_l1171_117179

theorem system_real_solution_conditions (a b c x y z : ℝ) (h1 : a * x + b * y = c * z) (h2 : a * Real.sqrt (1 - x^2) + b * Real.sqrt (1 - y^2) = c * Real.sqrt (1 - z^2)) :
  abs a ≤ abs b + abs c ∧ abs b ≤ abs a + abs c ∧ abs c ≤ abs a + abs b ∧
  (a * b >= 0 ∨ a * c >= 0 ∨ b * c >= 0) :=
sorry

end system_real_solution_conditions_l1171_117179


namespace fraction_meaningful_l1171_117164

theorem fraction_meaningful (x : ℝ) : (∃ z, z = 3 / (x - 4)) ↔ x ≠ 4 :=
by
  sorry

end fraction_meaningful_l1171_117164


namespace LitterPatrol_pickup_l1171_117172

theorem LitterPatrol_pickup :
  ∃ n : ℕ, n = 10 + 8 :=
sorry

end LitterPatrol_pickup_l1171_117172


namespace odd_function_periodicity_l1171_117169

noncomputable def f : ℝ → ℝ := sorry

theorem odd_function_periodicity (f_odd : ∀ x, f (-x) = -f x)
  (f_periodic : ∀ x, f (x + 2) = -f x) (f_val : f 1 = 2) : f 2011 = -2 :=
by
  sorry

end odd_function_periodicity_l1171_117169


namespace charitable_woman_l1171_117188

theorem charitable_woman (initial_pennies : ℕ) 
  (farmer_share : ℕ) (beggar_share : ℕ) (boy_share : ℕ) (left_pennies : ℕ) 
  (h1 : initial_pennies = 42)
  (h2 : farmer_share = (initial_pennies / 2 + 1))
  (h3 : beggar_share = ((initial_pennies - farmer_share) / 2 + 2))
  (h4 : boy_share = ((initial_pennies - farmer_share - beggar_share) / 2 + 3))
  (h5 : left_pennies = initial_pennies - farmer_share - beggar_share - boy_share) : 
  left_pennies = 1 :=
by
  sorry

end charitable_woman_l1171_117188


namespace parabola_axis_of_symmetry_range_l1171_117177

theorem parabola_axis_of_symmetry_range
  (a b c m n t : ℝ)
  (h₀ : 0 < a)
  (h₁ : m = a * 1^2 + b * 1 + c)
  (h₂ : n = a * 3^2 + b * 3 + c)
  (h₃ : m < n)
  (h₄ : n < c)
  (h_t : t = -b / (2 * a)) :
  (3 / 2) < t ∧ t < 2 :=
sorry

end parabola_axis_of_symmetry_range_l1171_117177


namespace product_is_approximately_9603_l1171_117176

noncomputable def smaller_number : ℝ := 97.49871794028884
noncomputable def successive_number : ℝ := smaller_number + 1
noncomputable def product_of_numbers : ℝ := smaller_number * successive_number

theorem product_is_approximately_9603 : abs (product_of_numbers - 9603) < 10e-3 := 
sorry

end product_is_approximately_9603_l1171_117176


namespace min_value_of_function_l1171_117196

theorem min_value_of_function (h : 0 < x ∧ x < 1) : 
  ∃ (y : ℝ), (∀ z : ℝ, z = (4 / x + 1 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end min_value_of_function_l1171_117196


namespace probability_of_sum_16_with_duplicates_l1171_117125

namespace DiceProbability

def is_valid_die_roll (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 6

def is_valid_combination (x y z : ℕ) : Prop :=
  x + y + z = 16 ∧ 
  is_valid_die_roll x ∧ 
  is_valid_die_roll y ∧ 
  is_valid_die_roll z ∧ 
  (x = y ∨ y = z ∨ z = x)

theorem probability_of_sum_16_with_duplicates (P : ℚ) :
  (∃ x y z : ℕ, is_valid_combination x y z) → 
  P = 1 / 36 :=
sorry

end DiceProbability

end probability_of_sum_16_with_duplicates_l1171_117125


namespace value_expression_l1171_117109

noncomputable def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_expression (p q r s t : ℝ) (h : g p q r s t (-3) = 9) : 
  16 * p - 8 * q + 4 * r - 2 * s + t = -9 := 
by
  sorry

end value_expression_l1171_117109


namespace deschamps_cows_l1171_117157

theorem deschamps_cows (p v : ℕ) (h1 : p + v = 160) (h2 : 2 * p + 4 * v = 400) : v = 40 :=
by sorry

end deschamps_cows_l1171_117157


namespace arithmetic_sequence_30th_term_l1171_117182

def arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

theorem arithmetic_sequence_30th_term :
  arithmetic_sequence 3 6 30 = 177 :=
by
  -- Proof steps go here
  sorry

end arithmetic_sequence_30th_term_l1171_117182


namespace population_of_missing_village_l1171_117110

theorem population_of_missing_village 
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ) 
  (avg_pop : ℕ) 
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1023)
  (h4 : pop4 = 945)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000) :
  ∃ (pop_missing : ℕ), pop_missing = 1100 := 
by
  -- Placeholder for proof
  sorry

end population_of_missing_village_l1171_117110


namespace margaret_score_l1171_117145

theorem margaret_score (average_score marco_score margaret_score : ℝ)
  (h1: average_score = 90)
  (h2: marco_score = average_score - 0.10 * average_score)
  (h3: margaret_score = marco_score + 5) : 
  margaret_score = 86 := 
by
  sorry

end margaret_score_l1171_117145


namespace Stephen_total_distance_l1171_117135

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end Stephen_total_distance_l1171_117135


namespace ratio_of_areas_of_two_concentric_circles_l1171_117197

theorem ratio_of_areas_of_two_concentric_circles
  (C₁ C₂ : ℝ)
  (h1 : ∀ θ₁ θ₂, θ₁ = 30 ∧ θ₂ = 24 →
      (θ₁ / 360) * C₁ = (θ₂ / 360) * C₂):
  (C₁ / C₂) ^ 2 = (16 / 25) := by
  sorry

end ratio_of_areas_of_two_concentric_circles_l1171_117197


namespace line_intersects_circle_l1171_117141

theorem line_intersects_circle (r d : ℝ) (hr : r = 5) (hd : d = 3 * Real.sqrt 2) : d < r :=
by
  rw [hr, hd]
  exact sorry

end line_intersects_circle_l1171_117141


namespace triangle_is_isosceles_l1171_117171

variable (a b c : ℝ)
variable (h : a^2 - b * c = a * (b - c))

theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - b * c = a * (b - c)) : a = b ∨ b = c ∨ c = a := by
  sorry

end triangle_is_isosceles_l1171_117171


namespace points_equidistant_from_circle_and_tangents_l1171_117153

noncomputable def circle_radius := 4
noncomputable def tangent_distance := 6

theorem points_equidistant_from_circle_and_tangents :
  ∃! (P : ℝ × ℝ), dist P (0, 0) = circle_radius ∧
                 dist P (0, tangent_distance) = tangent_distance - circle_radius ∧
                 dist P (0, -tangent_distance) = tangent_distance - circle_radius :=
by {
  sorry
}

end points_equidistant_from_circle_and_tangents_l1171_117153


namespace slope_range_l1171_117102

noncomputable def directed_distance (a b c x0 y0 : ℝ) : ℝ :=
  (a * x0 + b * y0 + c) / (Real.sqrt (a^2 + b^2))

theorem slope_range {A B P : ℝ × ℝ} (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : P = (3, 0))
                   {C : ℝ × ℝ} (hC : ∃ θ : ℝ, C = (9 * Real.cos θ, 18 + 9 * Real.sin θ))
                   {a b c : ℝ} (h_line : c = -3 * a)
                   (h_sum_distances : directed_distance a b c (-1) 0 +
                                      directed_distance a b c 1 0 +
                                      directed_distance a b c (9 * Real.cos θ) (18 + 9 * Real.sin θ) = 0) :
  -3 ≤ - (a / b) ∧ - (a / b) ≤ -1 := sorry

end slope_range_l1171_117102


namespace original_avg_age_is_fifty_l1171_117191

-- Definitions based on conditions
variable (N : ℕ) -- original number of students
variable (A : ℕ) -- original average age
variable (new_students : ℕ) -- number of new students
variable (new_avg_age : ℕ) -- average age of new students
variable (decreased_avg_age : ℕ) -- new average age after new students join

-- Conditions given in the problem
def original_avg_age_condition : Prop := A = 50
def new_students_condition : Prop := new_students = 12
def avg_age_new_students_condition : Prop := new_avg_age = 32
def decreased_avg_age_condition : Prop := decreased_avg_age = 46

-- Final Mathematical Equivalent Proof Problem
theorem original_avg_age_is_fifty
  (h1 : original_avg_age_condition A)
  (h2 : new_students_condition new_students)
  (h3 : avg_age_new_students_condition new_avg_age)
  (h4 : decreased_avg_age_condition decreased_avg_age) :
  A = 50 :=
by sorry

end original_avg_age_is_fifty_l1171_117191


namespace math_books_count_l1171_117118

theorem math_books_count (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 396) : M = 54 :=
sorry

end math_books_count_l1171_117118


namespace randy_biscuits_l1171_117108

theorem randy_biscuits (F : ℕ) (initial_biscuits mother_biscuits brother_ate remaining_biscuits : ℕ) 
  (h_initial : initial_biscuits = 32)
  (h_mother : mother_biscuits = 15)
  (h_brother : brother_ate = 20)
  (h_remaining : remaining_biscuits = 40)
  : ((initial_biscuits + mother_biscuits + F) - brother_ate) = remaining_biscuits → F = 13 := 
by
  intros h_eq
  sorry

end randy_biscuits_l1171_117108


namespace remainder_of_14_pow_53_mod_7_l1171_117180

theorem remainder_of_14_pow_53_mod_7 : (14 ^ 53) % 7 = 0 := by
  sorry

end remainder_of_14_pow_53_mod_7_l1171_117180


namespace least_three_digit_product_of_digits_is_8_l1171_117103

theorem least_three_digit_product_of_digits_is_8 :
  ∃ n : ℕ, n ≥ 100 ∧ n < 1000 ∧ (n.digits 10).prod = 8 ∧ ∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m.digits 10).prod = 8 → n ≤ m :=
sorry

end least_three_digit_product_of_digits_is_8_l1171_117103


namespace jen_lisa_spent_l1171_117190

theorem jen_lisa_spent (J L : ℝ) 
  (h1 : L = 0.8 * J) 
  (h2 : J = L + 15) : 
  J + L = 135 := 
by
  sorry

end jen_lisa_spent_l1171_117190


namespace determine_p_l1171_117161

theorem determine_p (p : ℝ) (h : (2 * p - 1) * (-1)^2 + 2 * (1 - p) * (-1) + 3 * p = 0) : p = 3 / 7 := by
  sorry

end determine_p_l1171_117161


namespace leaves_blew_away_correct_l1171_117140

-- Definitions based on conditions
def original_leaves : ℕ := 356
def leaves_left : ℕ := 112
def leaves_blew_away : ℕ := original_leaves - leaves_left

-- Theorem statement based on the question and correct answer
theorem leaves_blew_away_correct : leaves_blew_away = 244 := by {
  -- Proof goes here (omitted for now)
  sorry
}

end leaves_blew_away_correct_l1171_117140


namespace farmer_ear_count_l1171_117111

theorem farmer_ear_count
    (seeds_per_ear : ℕ)
    (price_per_ear : ℝ)
    (cost_per_bag : ℝ)
    (seeds_per_bag : ℕ)
    (profit : ℝ)
    (target_profit : ℝ) :
  seeds_per_ear = 4 →
  price_per_ear = 0.1 →
  cost_per_bag = 0.5 →
  seeds_per_bag = 100 →
  target_profit = 40 →
  profit = price_per_ear - ((cost_per_bag / seeds_per_bag) * seeds_per_ear) →
  target_profit / profit = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end farmer_ear_count_l1171_117111


namespace nathan_final_temperature_l1171_117117

theorem nathan_final_temperature : ∃ (final_temp : ℝ), final_temp = 77.4 :=
  let initial_temp : ℝ := 50
  let type_a_increase : ℝ := 2
  let type_b_increase : ℝ := 3.5
  let type_c_increase : ℝ := 4.8
  let type_d_increase : ℝ := 7.2
  let type_a_quantity : ℚ := 6
  let type_b_quantity : ℚ := 5
  let type_c_quantity : ℚ := 9
  let type_d_quantity : ℚ := 3
  let temp_after_a := initial_temp + 3 * type_a_increase
  let temp_after_b := temp_after_a + 2 * type_b_increase
  let temp_after_c := temp_after_b + 3 * type_c_increase
  let final_temp := temp_after_c
  ⟨final_temp, sorry⟩

end nathan_final_temperature_l1171_117117


namespace complement_union_l1171_117126

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union : 
  U = {0, 1, 2, 3, 4} →
  (U \ A = {1, 2}) →
  B = {1, 3} →
  (A ∪ B = {0, 1, 3, 4}) :=
by
  intros hU hA hB
  sorry

end complement_union_l1171_117126


namespace dealer_purchase_fraction_l1171_117187

theorem dealer_purchase_fraction (P C : ℝ) (h1 : ∃ S, S = 1.5 * P) (h2 : ∃ S, S = 2 * C) :
  C / P = 3 / 8 :=
by
  -- The statement of the theorem has been generated based on the problem conditions.
  sorry

end dealer_purchase_fraction_l1171_117187


namespace find_interest_rate_l1171_117160
noncomputable def annualInterestRate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  P * (1 + r / n)^(n * t) = A

theorem find_interest_rate :
  annualInterestRate 5000 6050.000000000001 1 2 0.1 :=
by
  -- The proof goes here
  sorry

end find_interest_rate_l1171_117160


namespace saltwater_solution_l1171_117189

theorem saltwater_solution (x : ℝ) (h1 : ∃ v : ℝ, v = x ∧ v * 0.2 = 0.20 * x)
(h2 : 3 / 4 * x = 3 / 4 * x)
(h3 : ∃ v' : ℝ, v' = 3 / 4 * x + 6 + 12)
(h4 : (0.20 * x + 12) / (3 / 4 * x + 18) = 1 / 3) : x = 120 :=
by 
  sorry

end saltwater_solution_l1171_117189


namespace solve_fraction_l1171_117104

theorem solve_fraction (x : ℝ) (h₁ : x^2 - 1 = 0) (h₂ : (x - 2) * (x + 1) ≠ 0) : x = 1 := 
sorry

end solve_fraction_l1171_117104


namespace number_534n_divisible_by_12_l1171_117186

theorem number_534n_divisible_by_12 (n : ℕ) : (5340 + n) % 12 = 0 ↔ n = 0 := by sorry

end number_534n_divisible_by_12_l1171_117186


namespace line_relation_in_perpendicular_planes_l1171_117175

-- Let's define the notions of planes and lines being perpendicular/parallel
variables {α β : Plane} {a : Line}

def plane_perpendicular (α β : Plane) : Prop := sorry -- definition of perpendicular planes
def line_perpendicular_plane (a : Line) (β : Plane) : Prop := sorry -- definition of a line being perpendicular to a plane
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line being parallel to a plane
def line_in_plane (a : Line) (α : Plane) : Prop := sorry -- definition of a line lying in a plane

-- The theorem stating the relationship given the conditions
theorem line_relation_in_perpendicular_planes 
  (h1 : plane_perpendicular α β) 
  (h2 : line_perpendicular_plane a β) : 
  line_parallel_plane a α ∨ line_in_plane a α :=
sorry

end line_relation_in_perpendicular_planes_l1171_117175


namespace sqrt_one_half_eq_sqrt_two_over_two_l1171_117154

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end sqrt_one_half_eq_sqrt_two_over_two_l1171_117154


namespace adults_not_wearing_blue_l1171_117170

-- Conditions
def children : ℕ := 45
def adults : ℕ := children / 3
def adults_wearing_blue : ℕ := adults / 3

-- Theorem Statement
theorem adults_not_wearing_blue :
  adults - adults_wearing_blue = 10 :=
sorry

end adults_not_wearing_blue_l1171_117170


namespace remaining_paint_needed_l1171_117147

-- Define the conditions
def total_paint_needed : ℕ := 70
def paint_bought : ℕ := 23
def paint_already_have : ℕ := 36

-- Lean theorem statement
theorem remaining_paint_needed : (total_paint_needed - (paint_already_have + paint_bought)) = 11 := by
  sorry

end remaining_paint_needed_l1171_117147


namespace cost_of_jeans_and_shirts_l1171_117122

theorem cost_of_jeans_and_shirts 
  (S : ℕ) (J : ℕ) (X : ℕ)
  (hS : S = 18)
  (h2J3S : 2 * J + 3 * S = 76)
  (h3J2S : 3 * J + 2 * S = X) :
  X = 69 :=
by
  sorry

end cost_of_jeans_and_shirts_l1171_117122


namespace smallest_value_of_n_l1171_117152

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end smallest_value_of_n_l1171_117152


namespace smallest_fraction_division_l1171_117165

theorem smallest_fraction_division (a b : ℕ) (h_coprime : Nat.gcd a b = 1) 
(h1 : ∃ n, (25 * a = n * 21 * b)) (h2 : ∃ m, (15 * a = m * 14 * b)) : (a = 42) ∧ (b = 5) := 
sorry

end smallest_fraction_division_l1171_117165


namespace range_of_a_l1171_117112

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a 
  (a : ℝ) 
  (h : ∀ x : ℝ, 1 < x → 2 * a * Real.log x ≤ 2 * x^2 + f a (2 * x - 1)) :
  a ≤ 2 :=
sorry

end range_of_a_l1171_117112


namespace reduce_to_one_l1171_117198

theorem reduce_to_one (n : ℕ) : ∃ k, (k = 1) :=
by
  sorry

end reduce_to_one_l1171_117198


namespace percentage_enclosed_by_hexagons_is_50_l1171_117114

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def square_area (s : ℝ) : ℝ :=
  s^2

noncomputable def total_tiling_unit_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * square_area s

noncomputable def percentage_enclosed_by_hexagons (s : ℝ) : ℝ :=
  (hexagon_area s / total_tiling_unit_area s) * 100

theorem percentage_enclosed_by_hexagons_is_50 (s : ℝ) : percentage_enclosed_by_hexagons s = 50 := by
  sorry

end percentage_enclosed_by_hexagons_is_50_l1171_117114


namespace inequality_a_b_c_d_l1171_117133

theorem inequality_a_b_c_d
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h₄ : a * b + b * c + c * d + d * a = 1) :
  (a ^ 3 / (b + c + d) + b ^ 3 / (c + d + a) + c ^ 3 / (a + b + d) + d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_a_b_c_d_l1171_117133


namespace pure_imaginary_condition_l1171_117130

variable (a : ℝ)

def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition :
  isPureImaginary (a - 17 / (4 - (i : ℂ))) → a = 4 := 
by
  sorry

end pure_imaginary_condition_l1171_117130


namespace M1M2_product_l1171_117116

theorem M1M2_product :
  ∀ (M1 M2 : ℝ),
  (∀ x : ℝ, x^2 - 5 * x + 6 ≠ 0 →
    (45 * x - 55) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) →
  (M1 + M2 = 45) →
  (3 * M1 + 2 * M2 = 55) →
  M1 * M2 = 200 :=
by
  sorry

end M1M2_product_l1171_117116


namespace find_h_l1171_117181

theorem find_h (x : ℝ) : 
  ∃ a k : ℝ, 3 * x^2 + 9 * x + 20 = a * (x - (-3 / 2))^2 + k :=
sorry

end find_h_l1171_117181


namespace calc_diagonal_of_rectangle_l1171_117193

variable (a : ℕ) (A : ℕ)

theorem calc_diagonal_of_rectangle (h_a : a = 6) (h_A : A = 48) (H : a * a' = A) :
  ∃ d : ℕ, d = 10 :=
by
 sorry

end calc_diagonal_of_rectangle_l1171_117193


namespace miles_traveled_correct_l1171_117163

def initial_odometer_reading := 212.3
def odometer_reading_at_lunch := 372.0
def miles_traveled := odometer_reading_at_lunch - initial_odometer_reading

theorem miles_traveled_correct : miles_traveled = 159.7 :=
by
  sorry

end miles_traveled_correct_l1171_117163


namespace latus_rectum_parabola_l1171_117143

theorem latus_rectum_parabola : 
  ∀ (x y : ℝ), (x = 4 * y^2) → (x = -1/16) :=
by 
  sorry

end latus_rectum_parabola_l1171_117143


namespace proof_x_square_ab_a_square_l1171_117101

variable {x b a : ℝ}

/-- Given that x < b < a < 0 where x, b, and a are real numbers, we need to prove x^2 > ab > a^2. -/
theorem proof_x_square_ab_a_square (hx : x < b) (hb : b < a) (ha : a < 0) :
  x^2 > ab ∧ ab > a^2 := 
by
  sorry

end proof_x_square_ab_a_square_l1171_117101


namespace min_value_of_F_on_neg_infinity_l1171_117146

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions provided in the problem
axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom g_odd : ∀ x : ℝ, g (-x) = - g x
noncomputable def F (x : ℝ) := a * f x + b * g x + 2
axiom F_max_on_pos : ∃ x ∈ (Set.Ioi 0), F x = 5

-- Prove the conclusion of the problem
theorem min_value_of_F_on_neg_infinity : ∃ y ∈ (Set.Iio 0), F y = -1 :=
sorry

end min_value_of_F_on_neg_infinity_l1171_117146


namespace yangyang_departure_time_l1171_117119

noncomputable def departure_time : Nat := 373 -- 6:13 in minutes from midnight (6 * 60 + 13)

theorem yangyang_departure_time :
  let arrival_at_60_mpm := 413 -- 6:53 in minutes from midnight
  let arrival_at_75_mpm := 405 -- 6:45 in minutes from midnight
  let difference := arrival_at_60_mpm - arrival_at_75_mpm -- time difference
  let x := 40 -- time taken to walk to school at 60 meters per minute
  departure_time = arrival_at_60_mpm - x :=
by
  -- Definitions
  let arrival_at_60_mpm := 413
  let arrival_at_75_mpm := 405
  let difference := 8
  let x := 40
  have h : departure_time = (413 - 40) := rfl
  sorry

end yangyang_departure_time_l1171_117119


namespace point_2023_0_cannot_lie_on_line_l1171_117134

-- Define real numbers a and c with the condition ac > 0
variables (a c : ℝ)

-- The condition ac > 0
def ac_positive := (a * c > 0)

-- The statement that (2023, 0) cannot be on the line y = ax + c given the condition a * c > 0
theorem point_2023_0_cannot_lie_on_line (h : ac_positive a c) : ¬ (0 = 2023 * a + c) :=
sorry

end point_2023_0_cannot_lie_on_line_l1171_117134


namespace is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l1171_117194

-- Problem 1: If \(2^{n} - 1\) is prime, then \(n\) is prime.
theorem is_prime_if_two_pow_n_minus_one_is_prime (n : ℕ) (hn : Prime (2^n - 1)) : Prime n :=
sorry

-- Problem 2: If \(2^{n} + 1\) is prime, then \(n\) is a power of 2.
theorem is_power_of_two_if_two_pow_n_plus_one_is_prime (n : ℕ) (hn : Prime (2^n + 1)) : ∃ k : ℕ, n = 2^k :=
sorry

end is_prime_if_two_pow_n_minus_one_is_prime_is_power_of_two_if_two_pow_n_plus_one_is_prime_l1171_117194


namespace mike_picked_32_limes_l1171_117156

theorem mike_picked_32_limes (total_limes : ℕ) (alyssa_limes : ℕ) (mike_limes : ℕ) 
  (h1 : total_limes = 57) (h2 : alyssa_limes = 25) (h3 : mike_limes = total_limes - alyssa_limes) : 
  mike_limes = 32 :=
by
  sorry

end mike_picked_32_limes_l1171_117156


namespace ratio_of_x_l1171_117115

theorem ratio_of_x (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
    ((x + 1 / x) / (x - 1 / x)) = (Real.sqrt 7 / Real.sqrt 6) :=
by
  sorry

end ratio_of_x_l1171_117115


namespace sandy_comic_books_l1171_117195

-- Define Sandy's initial number of comic books
def initial_comic_books : ℕ := 14

-- Define the number of comic books Sandy sold
def sold_comic_books (n : ℕ) : ℕ := n / 2

-- Define the number of comic books Sandy bought
def bought_comic_books : ℕ := 6

-- Define the number of comic books Sandy has now
def final_comic_books (initial : ℕ) (sold : ℕ) (bought : ℕ) : ℕ :=
  initial - sold + bought

-- The theorem statement to prove the final number of comic books
theorem sandy_comic_books : final_comic_books initial_comic_books (sold_comic_books initial_comic_books) bought_comic_books = 13 := by
  sorry

end sandy_comic_books_l1171_117195


namespace smallest_value_expression_l1171_117183

theorem smallest_value_expression
    (a b c : ℝ) 
    (h1 : c > b)
    (h2 : b > a)
    (h3 : c ≠ 0) : 
    ∃ z : ℝ, z = 0 ∧ z = (a + b)^2 / c^2 + (b - c)^2 / c^2 + (c - b)^2 / c^2 :=
by
  sorry

end smallest_value_expression_l1171_117183


namespace expected_value_of_draws_before_stopping_l1171_117174

noncomputable def totalBalls := 10
noncomputable def redBalls := 2
noncomputable def whiteBalls := 8

noncomputable def prob_one_draw_white : ℚ := whiteBalls / totalBalls
noncomputable def prob_two_draws_white : ℚ := (redBalls / totalBalls) * (whiteBalls / (totalBalls - 1))
noncomputable def prob_three_draws_white : ℚ := (redBalls / (totalBalls - redBalls + 1)) * ((redBalls - 1) / (totalBalls - 1)) * (whiteBalls / (totalBalls - 2))

noncomputable def expected_draws_before_white : ℚ :=
  1 * prob_one_draw_white + 2 * prob_two_draws_white + 3 * prob_three_draws_white

theorem expected_value_of_draws_before_stopping : expected_draws_before_white = 11 / 9 := by
  sorry

end expected_value_of_draws_before_stopping_l1171_117174


namespace probability_sum_eight_l1171_117105

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 5

theorem probability_sum_eight :
  (favorable_outcomes : ℚ) / total_outcomes = 5 / 36 := by
  sorry

end probability_sum_eight_l1171_117105


namespace valid_triangle_side_l1171_117192

theorem valid_triangle_side (x : ℕ) (h_pos : 0 < x) (h1 : x + 6 > 15) (h2 : 21 > x) :
  10 ≤ x ∧ x ≤ 20 :=
by {
  sorry
}

end valid_triangle_side_l1171_117192


namespace find_larger_number_l1171_117132

theorem find_larger_number (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := 
sorry

end find_larger_number_l1171_117132


namespace square_of_binomial_l1171_117159

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (9:ℝ) * x^2 + 24 * x + a = (3 * x + b)^2) → a = 16 :=
by
  sorry

end square_of_binomial_l1171_117159
