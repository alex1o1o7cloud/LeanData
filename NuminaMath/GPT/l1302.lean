import Mathlib

namespace pythagorean_triangle_inscribed_circle_radius_is_integer_l1302_130236

theorem pythagorean_triangle_inscribed_circle_radius_is_integer 
  (a b c : ℕ)
  (h1 : c^2 = a^2 + b^2) 
  (h2 : r = (a + b - c) / 2) :
  ∃ (r : ℕ), r = (a + b - c) / 2 :=
sorry

end pythagorean_triangle_inscribed_circle_radius_is_integer_l1302_130236


namespace train_crossing_time_l1302_130210

/-- A train 400 m long traveling at a speed of 36 km/h crosses an electric pole in 40 seconds. -/
theorem train_crossing_time (length : ℝ) (speed_kmph : ℝ) (speed_mps : ℝ) (time : ℝ) 
  (h1 : length = 400)
  (h2 : speed_kmph = 36)
  (h3 : speed_mps = speed_kmph * 1000 / 3600)
  (h4 : time = length / speed_mps) :
  time = 40 :=
by {
  sorry
}

end train_crossing_time_l1302_130210


namespace probability_student_less_than_25_l1302_130226

-- Defining the problem conditions
def total_students : ℕ := 100
def percent_male : ℕ := 40
def percent_female : ℕ := 100 - percent_male
def percent_male_25_or_older : ℕ := 40
def percent_female_25_or_older : ℕ := 30

-- Calculation based on the conditions
def num_male_students := (percent_male * total_students) / 100
def num_female_students := (percent_female * total_students) / 100
def num_male_25_or_older := (percent_male_25_or_older * num_male_students) / 100
def num_female_25_or_older := (percent_female_25_or_older * num_female_students) / 100

def num_25_or_older := num_male_25_or_older + num_female_25_or_older
def num_less_than_25 := total_students - num_25_or_older
def probability_less_than_25 := (num_less_than_25: ℚ) / total_students

-- Define the theorem
theorem probability_student_less_than_25 :
  probability_less_than_25 = 0.66 := by
  sorry

end probability_student_less_than_25_l1302_130226


namespace blue_books_count_l1302_130239

def number_of_blue_books (R B : ℕ) (p : ℚ) : Prop :=
  R = 4 ∧ p = 3/14 → B^2 + 7 * B - 44 = 0

theorem blue_books_count :
  ∃ B : ℕ, number_of_blue_books 4 B (3/14) ∧ B = 4 :=
by
  sorry

end blue_books_count_l1302_130239


namespace find_smaller_angle_l1302_130246

theorem find_smaller_angle (x : ℝ) (h1 : (x + (x + 18) = 180)) : x = 81 := 
by 
  sorry

end find_smaller_angle_l1302_130246


namespace inequality_problem_l1302_130279

-- Define the conditions and the problem statement
theorem inequality_problem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_problem_l1302_130279


namespace die_total_dots_l1302_130264

theorem die_total_dots :
  ∀ (face1 face2 face3 face4 face5 face6 : ℕ),
    face1 < face2 ∧ face2 < face3 ∧ face3 < face4 ∧ face4 < face5 ∧ face5 < face6 ∧
    (face2 - face1 ≥ 2) ∧ (face3 - face2 ≥ 2) ∧ (face4 - face3 ≥ 2) ∧ (face5 - face4 ≥ 2) ∧ (face6 - face5 ≥ 2) ∧
    (face3 ≠ face1 + 2) ∧ (face4 ≠ face2 + 2) ∧ (face5 ≠ face3 + 2) ∧ (face6 ≠ face4 + 2)
    → face1 + face2 + face3 + face4 + face5 + face6 = 27 :=
by {
  sorry
}

end die_total_dots_l1302_130264


namespace change_combinations_50_cents_l1302_130208

-- Define the conditions for creating 50 cents using standard coins
def ways_to_make_change (pennies nickels dimes : ℕ) : ℕ :=
  pennies + 5 * nickels + 10 * dimes

theorem change_combinations_50_cents : 
  ∃ num_ways, 
    num_ways = 28 ∧
    ∀ (pennies nickels dimes : ℕ), 
      pennies + 5 * nickels + 10 * dimes = 50 → 
      -- Exclude using only a single half-dollar
      ¬(num_ways = if (pennies = 0 ∧ nickels = 0 ∧ dimes = 0) then 1 else 28) := 
sorry

end change_combinations_50_cents_l1302_130208


namespace subset_condition_l1302_130271

def A : Set ℝ := {2, 0, 1, 6}
def B (a : ℝ) : Set ℝ := {x | x + a > 0}

theorem subset_condition (a : ℝ) (h : A ⊆ B a) : a > 0 :=
sorry

end subset_condition_l1302_130271


namespace least_positive_integer_satifies_congruences_l1302_130280

theorem least_positive_integer_satifies_congruences :
  ∃ x : ℕ, x ≡ 1 [MOD 4] ∧ x ≡ 2 [MOD 5] ∧ x ≡ 3 [MOD 6] ∧ x = 17 :=
sorry

end least_positive_integer_satifies_congruences_l1302_130280


namespace non_negative_integer_solutions_of_inequality_system_l1302_130217

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end non_negative_integer_solutions_of_inequality_system_l1302_130217


namespace min_distance_l1302_130278

open Complex

theorem min_distance (z : ℂ) (hz : abs (z + 2 - 2*I) = 1) : abs (z - 2 - 2*I) = 3 :=
sorry

end min_distance_l1302_130278


namespace factorization_problem_l1302_130291

theorem factorization_problem :
  ∃ (a b : ℤ), (25 * x^2 - 130 * x - 120 = (5 * x + a) * (5 * x + b)) ∧ (a + 3 * b = -86) := by
  sorry

end factorization_problem_l1302_130291


namespace ratio_siblings_l1302_130292

theorem ratio_siblings (M J C : ℕ) 
  (hM : M = 60)
  (hJ : J = 4 * M - 60)
  (hJ_C : J = C + 135) :
  (C : ℚ) / M = 3 / 4 :=
by
  sorry

end ratio_siblings_l1302_130292


namespace sum_of_squares_of_roots_l1302_130285

theorem sum_of_squares_of_roots :
  ∃ x1 x2 : ℝ, (10 * x1 ^ 2 + 15 * x1 - 20 = 0) ∧ (10 * x2 ^ 2 + 15 * x2 - 20 = 0) ∧ (x1 ≠ x2) ∧ x1^2 + x2^2 = 25/4 :=
sorry

end sum_of_squares_of_roots_l1302_130285


namespace Jake_weight_correct_l1302_130221

def Mildred_weight : ℕ := 59
def Carol_weight : ℕ := Mildred_weight + 9
def Jake_weight : ℕ := 2 * Carol_weight

theorem Jake_weight_correct : Jake_weight = 136 := by
  sorry

end Jake_weight_correct_l1302_130221


namespace robert_books_l1302_130290

/-- Given that Robert reads at a speed of 75 pages per hour, books have 300 pages, and Robert reads for 9 hours,
    he can read 2 complete 300-page books in that time. -/
theorem robert_books (reading_speed : ℤ) (pages_per_book : ℤ) (hours_available : ℤ) 
(h1 : reading_speed = 75) 
(h2 : pages_per_book = 300) 
(h3 : hours_available = 9) : 
  hours_available / (pages_per_book / reading_speed) = 2 := 
by {
  -- adding placeholder for proof
  sorry
}

end robert_books_l1302_130290


namespace determine_k_l1302_130242

theorem determine_k (k : ℝ) : (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 :=
by
  intro h
  sorry

end determine_k_l1302_130242


namespace complement_B_in_U_l1302_130262

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {x | x = 1}
def U : Set ℕ := A ∪ B

theorem complement_B_in_U : (U \ B) = {2, 3} := by
  sorry

end complement_B_in_U_l1302_130262


namespace concentration_proof_l1302_130268

noncomputable def newConcentration (vol1 vol2 vol3 : ℝ) (perc1 perc2 perc3 : ℝ) (totalVol : ℝ) (finalVol : ℝ) :=
  (vol1 * perc1 + vol2 * perc2 + vol3 * perc3) / finalVol

theorem concentration_proof : 
  newConcentration 2 6 4 0.2 0.55 0.35 (12 : ℝ) (15 : ℝ) = 0.34 := 
by 
  sorry

end concentration_proof_l1302_130268


namespace complement_intersection_l1302_130255

def M : Set ℝ := {x | x ≥ 1}
def N : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

theorem complement_intersection : (M ∩ N)ᶜ = { x : ℝ | x < 1 ∨ x > 3 } :=
  sorry

end complement_intersection_l1302_130255


namespace ratio_D_E_equal_l1302_130235

variable (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ)

def mary_story_conditions (total_characters : ℕ) (initial_A : ℕ) (initial_C : ℕ) (initial_D : ℕ) (initial_E : ℕ) : Prop :=
  total_characters = 60 ∧
  initial_A = 1 / 2 * total_characters ∧
  initial_C = 1 / 2 * initial_A ∧
  initial_D + initial_E = total_characters - (initial_A + initial_C)

theorem ratio_D_E_equal (total_characters initial_A initial_C initial_D initial_E : ℕ) :
  mary_story_conditions total_characters initial_A initial_C initial_D initial_E →
  initial_D = initial_E :=
sorry

end ratio_D_E_equal_l1302_130235


namespace cos_sum_identity_cosine_30_deg_l1302_130238

theorem cos_sum_identity : 
  (Real.cos (Real.pi * 43 / 180) * Real.cos (Real.pi * 13 / 180) + 
   Real.sin (Real.pi * 43 / 180) * Real.sin (Real.pi * 13 / 180)) = 
   (Real.cos (Real.pi * 30 / 180)) :=
sorry

theorem cosine_30_deg : 
  Real.cos (Real.pi * 30 / 180) = (Real.sqrt 3 / 2) :=
sorry

end cos_sum_identity_cosine_30_deg_l1302_130238


namespace line_divides_circle_1_3_l1302_130266

noncomputable def circle_equidistant_from_origin : Prop := 
  ∃ l : ℝ → ℝ, ∀ x y : ℝ, ((x-1)^2 + (y-1)^2 = 2) → 
                     (l 0 = 0 ∧ (l x = l y) ∧ 
                     ((x = 0) ∨ (y = 0)))

theorem line_divides_circle_1_3 (x y : ℝ) : 
  (x - 1)^2 + (y - 1)^2 = 2 → 
  (x = 0 ∨ y = 0) :=
by
  sorry

end line_divides_circle_1_3_l1302_130266


namespace middle_number_l1302_130201

theorem middle_number {a b c : ℚ} 
  (h1 : a + b = 15) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) 
  (h4 : c = 2 * a) : 
  b = 25 / 3 := 
by 
  sorry

end middle_number_l1302_130201


namespace min_value_of_expression_l1302_130276

theorem min_value_of_expression : 
  ∃ x y : ℝ, (z = x^2 + 2*x*y + 2*y^2 + 2*x + 4*y + 3) ∧ z = 1 ∧ x = 0 ∧ y = -1 :=
by
  sorry

end min_value_of_expression_l1302_130276


namespace prob_qualified_bulb_factory_a_l1302_130244

-- Define the given probability of a light bulb being produced by Factory A
def prob_factory_a : ℝ := 0.7

-- Define the given pass rate (conditional probability) of Factory A's light bulbs
def pass_rate_factory_a : ℝ := 0.95

-- The goal is to prove that the probability of getting a qualified light bulb produced by Factory A is 0.665
theorem prob_qualified_bulb_factory_a : prob_factory_a * pass_rate_factory_a = 0.665 :=
by
  -- This is where the proof would be, but we'll use sorry to skip the proof
  sorry

end prob_qualified_bulb_factory_a_l1302_130244


namespace parabola_sum_l1302_130252

theorem parabola_sum (a b c : ℝ)
  (h1 : 4 = a * 1^2 + b * 1 + c)
  (h2 : -1 = a * (-2)^2 + b * (-2) + c)
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c = a * (x + 1)^2 - 2)
  : a + b + c = 5 := by
  sorry

end parabola_sum_l1302_130252


namespace monkey2_peach_count_l1302_130259

noncomputable def total_peaches : ℕ := 81
def monkey1_share (p : ℕ) : ℕ := (5 * p) / 6
def remaining_after_monkey1 (p : ℕ) : ℕ := p - monkey1_share p
def monkey2_share (p : ℕ) : ℕ := (5 * remaining_after_monkey1 p) / 9
def remaining_after_monkey2 (p : ℕ) : ℕ := remaining_after_monkey1 p - monkey2_share p
def monkey3_share (p : ℕ) : ℕ := remaining_after_monkey2 p

theorem monkey2_peach_count : monkey2_share total_peaches = 20 :=
by
  sorry

end monkey2_peach_count_l1302_130259


namespace find_angle_l1302_130241

variable (a b : ℝ × ℝ) (α : ℝ)
variable (θ : ℝ)

-- Conditions provided in the problem
def condition1 := (a.1^2 + a.2^2 = 4)
def condition2 := (b = (4 * Real.cos α, -4 * Real.sin α))
def condition3 := (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0)

-- Desired result
theorem find_angle (h1 : condition1 a) (h2 : condition2 b α) (h3 : condition3 a b) :
  θ = Real.pi / 3 :=
sorry

end find_angle_l1302_130241


namespace distance_from_pole_to_line_l1302_130209

-- Definitions based on the problem condition
def polar_equation_line (ρ θ : ℝ) := ρ * (Real.cos θ + Real.sin θ) = Real.sqrt 3

-- The statement of the proof problem
theorem distance_from_pole_to_line (ρ θ : ℝ) (h : polar_equation_line ρ θ) :
  ρ = Real.sqrt 6 / 2 := sorry

end distance_from_pole_to_line_l1302_130209


namespace f_f_neg1_l1302_130263

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_f_neg1 : f (f (-1)) = 5 :=
  by
    sorry

end f_f_neg1_l1302_130263


namespace arithmetic_sequence_sum_is_18_l1302_130232

variable (a : ℕ → ℕ)

theorem arithmetic_sequence_sum_is_18
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 2) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 18 := 
sorry

end arithmetic_sequence_sum_is_18_l1302_130232


namespace jessica_walks_distance_l1302_130267

theorem jessica_walks_distance (rate time : ℝ) (h_rate : rate = 4) (h_time : time = 2) :
  rate * time = 8 :=
by 
  rw [h_rate, h_time]
  norm_num

end jessica_walks_distance_l1302_130267


namespace num_members_in_league_l1302_130286

theorem num_members_in_league :
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  num_members = 150 :=
by
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  sorry

end num_members_in_league_l1302_130286


namespace x0_in_M_implies_x0_in_N_l1302_130204

def M : Set ℝ := {x | ∃ (k : ℤ), x = k + 1 / 2}
def N : Set ℝ := {x | ∃ (k : ℤ), x = k / 2 + 1}

theorem x0_in_M_implies_x0_in_N (x0 : ℝ) (h : x0 ∈ M) : x0 ∈ N := 
sorry

end x0_in_M_implies_x0_in_N_l1302_130204


namespace find_m_l1302_130281

theorem find_m (m : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (h : a = (2, -4) ∧ b = (-3, m) ∧ (‖a‖ * ‖b‖ + (a.1 * b.1 + a.2 * b.2)) = 0) : m = 6 := 
by 
  sorry

end find_m_l1302_130281


namespace first_discount_percentage_l1302_130296

-- Given conditions
def initial_price : ℝ := 390
def final_price : ℝ := 285.09
def second_discount : ℝ := 0.15

-- Definition for the first discount percentage
noncomputable def first_discount (D : ℝ) : ℝ :=
initial_price * (1 - D / 100) * (1 - second_discount)

-- Theorem statement
theorem first_discount_percentage : ∃ D : ℝ, first_discount D = final_price ∧ D = 13.99 :=
by
  sorry

end first_discount_percentage_l1302_130296


namespace trivia_team_total_points_l1302_130295

def totalPoints : Nat := 182

def points_member_A : Nat := 3 * 2
def points_member_B : Nat := 5 * 4 + 1 * 6
def points_member_C : Nat := 2 * 6
def points_member_D : Nat := 4 * 2 + 2 * 4
def points_member_E : Nat := 1 * 2 + 3 * 4
def points_member_F : Nat := 5 * 6
def points_member_G : Nat := 2 * 4 + 1 * 2
def points_member_H : Nat := 3 * 6 + 2 * 2
def points_member_I : Nat := 1 * 4 + 4 * 6
def points_member_J : Nat := 7 * 2 + 1 * 4

theorem trivia_team_total_points : 
  points_member_A + points_member_B + points_member_C + points_member_D + points_member_E + 
  points_member_F + points_member_G + points_member_H + points_member_I + points_member_J = totalPoints := 
by
  repeat { sorry }

end trivia_team_total_points_l1302_130295


namespace factorial_mod_prime_l1302_130215
-- Import all necessary libraries

-- State the conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- The main problem statement
theorem factorial_mod_prime (n : ℕ) (h : n = 10) : factorial n % 13 = 7 := by
  sorry

end factorial_mod_prime_l1302_130215


namespace tangent_segments_area_l1302_130225

theorem tangent_segments_area (r : ℝ) (l : ℝ) (area : ℝ) :
  r = 4 ∧ l = 6 → area = 9 * Real.pi :=
by
  sorry

end tangent_segments_area_l1302_130225


namespace find_n_l1302_130253

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 103 ∧ 100 * n % 103 = 65 % 103 ∧ n = 68 :=
by
  sorry

end find_n_l1302_130253


namespace repayment_correct_l1302_130258

noncomputable def repayment_amount (a γ : ℝ) : ℝ :=
  a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1)

theorem repayment_correct (a γ : ℝ) (γ_pos : γ > 0) : 
  repayment_amount a γ = a * γ * (1 + γ) ^ 5 / ((1 + γ) ^ 5 - 1) :=
by
   sorry

end repayment_correct_l1302_130258


namespace circle_area_l1302_130245

theorem circle_area : 
    (∃ x y : ℝ, 3 * x^2 + 3 * y^2 - 9 * x + 12 * y + 27 = 0) →
    (∃ A : ℝ, A = (7 / 4) * Real.pi) :=
by
  sorry

end circle_area_l1302_130245


namespace Yella_last_week_usage_l1302_130222

/-- 
Yella's computer usage last week was some hours. If she plans to use the computer 8 hours a day for this week, 
her computer usage for this week is 35 hours less. Given these conditions, prove that Yella's computer usage 
last week was 91 hours.
-/
theorem Yella_last_week_usage (daily_usage : ℕ) (days_in_week : ℕ) (difference : ℕ)
  (h1: daily_usage = 8)
  (h2: days_in_week = 7)
  (h3: difference = 35) :
  daily_usage * days_in_week + difference = 91 := 
by
  sorry

end Yella_last_week_usage_l1302_130222


namespace mechanism_parts_l1302_130283

theorem mechanism_parts (L S : ℕ) (h1 : L + S = 30) (h2 : L ≤ 11) (h3 : S ≤ 19) :
  L = 11 ∧ S = 19 :=
by
  sorry

end mechanism_parts_l1302_130283


namespace increasing_interval_l1302_130272

noncomputable def f (x : ℝ) := Real.log x / Real.log (1 / 2)

def is_monotonically_increasing (f : ℝ → ℝ) (s : Set ℝ) :=
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x < f y

def h (x : ℝ) : ℝ := x^2 + x - 2

theorem increasing_interval :
  is_monotonically_increasing (f ∘ h) {x : ℝ | x < -2} :=
sorry

end increasing_interval_l1302_130272


namespace geometric_seq_sum_identity_l1302_130212

noncomputable def is_geometric_seq (a : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, a (n + 1) = q * a n

theorem geometric_seq_sum_identity (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0)
  (hgeom : is_geometric_seq a) 
  (h1 : a 2 + a 6 = 3) 
  (h2 : a 6 + a 10 = 12) : 
  a 8 + a 12 = 24 :=
sorry

end geometric_seq_sum_identity_l1302_130212


namespace original_price_of_car_l1302_130287

theorem original_price_of_car (P : ℝ) 
  (h₁ : 0.561 * P + 200 = 7500) : 
  P = 13012.48 := 
sorry

end original_price_of_car_l1302_130287


namespace common_ratio_of_gp_l1302_130231

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
if r = 1 then a * n else a * (1 - r^n) / (1 - r)

theorem common_ratio_of_gp (a : ℝ) (r : ℝ) (h : geometric_sum a r 6 / geometric_sum a r 3 = 28) : r = 3 :=
by
  sorry

end common_ratio_of_gp_l1302_130231


namespace johns_ratio_l1302_130233

-- Definitions for initial counts
def initial_pink := 26
def initial_green := 15
def initial_yellow := 24
def initial_total := initial_pink + initial_green + initial_yellow

-- Definitions for Carl's and John's actions
def carl_pink_taken := 4
def john_pink_taken := 6
def remaining_pink := initial_pink - carl_pink_taken - john_pink_taken

-- Definition for remaining hard hats
def total_remaining := 43

-- Compute John's green hat withdrawal
def john_green_taken := (initial_total - carl_pink_taken - john_pink_taken) - total_remaining
def ratio := john_green_taken / john_pink_taken

theorem johns_ratio : ratio = 2 :=
by
  -- Proof details omitted
  sorry

end johns_ratio_l1302_130233


namespace find_divisor_l1302_130293

theorem find_divisor (D : ℕ) : 
  (242 % D = 15) ∧ 
  (698 % D = 27) ∧ 
  ((242 + 698) % D = 5) → 
  D = 42 := 
by 
  sorry

end find_divisor_l1302_130293


namespace third_price_reduction_l1302_130284

theorem third_price_reduction (original_price final_price : ℝ) (x : ℝ) 
  (h1 : (original_price * (1 - x)^2 = final_price))
  (h2 : final_price = 100)
  (h3 : original_price = 100 / (1 - 0.19)) :
  (original_price * (1 - x)^3 = 90) :=
by
  sorry

end third_price_reduction_l1302_130284


namespace john_spent_30_l1302_130261

/-- At a supermarket, John spent 1/5 of his money on fresh fruits and vegetables, 1/3 on meat products, and 1/10 on bakery products. If he spent the remaining $11 on candy, how much did John spend at the supermarket? -/
theorem john_spent_30 (X : ℝ) (h1 : X * (1/5) + X * (1/3) + X * (1/10) + 11 = X) : X = 30 := 
by 
  sorry

end john_spent_30_l1302_130261


namespace hypotenuse_length_l1302_130250

theorem hypotenuse_length (a b c : ℝ) (h1 : c^2 = a^2 + b^2) (h2 : a^2 + b^2 + c^2 = 2500) : c = 25 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_length_l1302_130250


namespace original_work_days_l1302_130298

-- Definitions based on conditions
noncomputable def L : ℕ := 7  -- Number of laborers originally employed
noncomputable def A : ℕ := 3  -- Number of absent laborers
noncomputable def t : ℕ := 14 -- Number of days it took the remaining laborers to finish the work

-- Theorem statement to prove
theorem original_work_days : (L - A) * t = L * 8 := by
  sorry

end original_work_days_l1302_130298


namespace find_max_value_l1302_130260

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x + a

theorem find_max_value (a x : ℝ) (h_min : f 1 a = 1) : 
  ∃ x : ℝ, f (-1/3) 2 = 59/27 :=
by {
  sorry
}

end find_max_value_l1302_130260


namespace quadratic_roots_real_find_m_value_l1302_130288

theorem quadratic_roots_real (m : ℝ) (h_roots : ∃ x1 x2 : ℝ, x1 * x1 + 4 * x1 + (m - 1) = 0 ∧ x2 * x2 + 4 * x2 + (m - 1) = 0) :
  m ≤ 5 :=
by {
  sorry
}

theorem find_m_value (m : ℝ) (x1 x2 : ℝ) (h_eq1 : x1 * x1 + 4 * x1 + (m - 1) = 0) (h_eq2 : x2 * x2 + 4 * x2 + (m - 1) = 0) (h_cond : 2 * (x1 + x2) + x1 * x2 + 10 = 0) :
  m = -1 :=
by {
  sorry
}

end quadratic_roots_real_find_m_value_l1302_130288


namespace min_value_expression_l1302_130243

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  ∃ a b, a > 0 ∧ b > 0 ∧ (∀ x y, x > 0 ∧ y > 0 → (1 / x + x / y^2 + y ≥ 2 * Real.sqrt 2)) := 
sorry

end min_value_expression_l1302_130243


namespace triangle_angle_and_side_ratio_l1302_130206

theorem triangle_angle_and_side_ratio
  (A B C : Real)
  (a b c : Real)
  (h1 : a / Real.sin A = b / Real.sin B)
  (h2 : b / Real.sin B = c / Real.sin C)
  (h3 : (a + c) / b = (Real.sin A - Real.sin B) / (Real.sin A - Real.sin C)) :
  C = Real.pi / 3 ∧ (1 < (a + b) / c ∧ (a + b) / c < 2) :=
by
  sorry


end triangle_angle_and_side_ratio_l1302_130206


namespace max_crystalline_polyhedron_volume_l1302_130249

theorem max_crystalline_polyhedron_volume (n : ℕ) (R : ℝ) (h_n : n > 1) :
  ∃ V : ℝ, 
    V = (32 / 81) * (n - 1) * (R ^ 3) * Real.sin (2 * Real.pi / (n - 1)) :=
sorry

end max_crystalline_polyhedron_volume_l1302_130249


namespace divisible_by_a_minus_one_squared_l1302_130248

theorem divisible_by_a_minus_one_squared (a n : ℕ) (h : n > 0) :
  (a^(n+1) - n * (a - 1) - a) % (a - 1)^2 = 0 :=
by
  sorry

end divisible_by_a_minus_one_squared_l1302_130248


namespace blue_length_is_2_l1302_130228

-- Define the lengths of the parts
def total_length : ℝ := 4
def purple_length : ℝ := 1.5
def black_length : ℝ := 0.5

-- Define the length of the blue part with the given conditions
def blue_length : ℝ := total_length - (purple_length + black_length)

-- State the theorem we need to prove
theorem blue_length_is_2 : blue_length = 2 :=
by 
  sorry

end blue_length_is_2_l1302_130228


namespace simplify_expression_l1302_130269

theorem simplify_expression (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 2) * 4 + (5 - 2 / 2) * (8 * p - 12) = 36 * p - 36 :=
by
  sorry

end simplify_expression_l1302_130269


namespace greatest_possible_bxa_l1302_130224

-- Define the property of the number being divisible by 35
def div_by_35 (n : ℕ) : Prop :=
  n % 35 = 0

-- Define the main proof problem
theorem greatest_possible_bxa :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ div_by_35 (10 * a + b) ∧ (∀ (a' b' : ℕ), a' < 10 → b' < 10 → div_by_35 (10 * a' + b') → b * a ≥ b' * a') :=
sorry

end greatest_possible_bxa_l1302_130224


namespace geometric_seq_ratio_l1302_130207

theorem geometric_seq_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : a 5 - a 3 = 12) 
  (h2 : a 6 - a 4 = 24) 
  (ha : ∃ a₁ : ℝ, (∀ n : ℕ, a n = a₁ * q ^ (n - 1)))
  (hS : ∀ n : ℕ, S n = a₁ * (1 - q ^ n) / (1 - q)) :
  ∀ n : ℕ, S n / a n = 2 - 2 ^ (1 - n) :=
sorry

end geometric_seq_ratio_l1302_130207


namespace percentage_distance_l1302_130270

theorem percentage_distance (start : ℝ) (end_point : ℝ) (point : ℝ) (total_distance : ℝ)
  (distance_from_start : ℝ) :
  start = -55 → end_point = 55 → point = 5.5 → total_distance = end_point - start →
  distance_from_start = point - start →
  (distance_from_start / total_distance) * 100 = 55 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_distance_l1302_130270


namespace intersect_sphere_circle_l1302_130211

-- Define the given sphere equation
def sphere (h k l R : ℝ) (x y z : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 + (z - l)^2 = R^2

-- Define the equation of a circle in the plane x = x0 parallel to the yz-plane
def circle_in_plane (x0 y0 z0 r : ℝ) (y z : ℝ) : Prop :=
  (y - y0)^2 + (z - z0)^2 = r^2

-- Define the intersecting circle from the sphere equation in the x = c plane
def intersecting_circle (h k l c R : ℝ) (y z : ℝ) : Prop :=
  (y - k)^2 + (z - l)^2 = R^2 - (h - c)^2

-- The main proof statement
theorem intersect_sphere_circle (h k l R c x0 y0 z0 r: ℝ) :
  ∀ y z, intersecting_circle h k l c R y z ↔ circle_in_plane x0 y0 z0 r y z :=
sorry

end intersect_sphere_circle_l1302_130211


namespace maximum_revenue_l1302_130237

def ticket_price (x : ℕ) (y : ℤ) : Prop :=
  (6 ≤ x ∧ x ≤ 10 ∧ y = 1000 * x - 5750) ∨
  (10 < x ∧ x ≤ 38 ∧ y = -30 * x^2 + 1300 * x - 5750)

theorem maximum_revenue :
  ∃ x y, ticket_price x y ∧ y = 8830 ∧ x = 22 :=
by {
  sorry
}

end maximum_revenue_l1302_130237


namespace find_m_from_intersection_l1302_130223

-- Define the sets A and B
def A : Set ℕ := {1, 2, 3}
def B (m : ℕ) : Set ℕ := {2, m, 4}

-- Prove the relationship given the conditions
theorem find_m_from_intersection (m : ℕ) (h : A ∩ B m = {2, 3}) : m = 3 := 
by 
  sorry

end find_m_from_intersection_l1302_130223


namespace max_smart_winners_min_total_prize_l1302_130282

-- Define relevant constants and conditions
def total_winners := 25
def prize_smart : ℕ := 15
def prize_comprehensive : ℕ := 30

-- Problem 1: Maximum number of winners in "Smartest Brain" competition
theorem max_smart_winners (x : ℕ) (h1 : total_winners = 25)
  (h2 : total_winners - x ≥ 5 * x) : x ≤ 4 :=
sorry

-- Problem 2: Minimum total prize amount
theorem min_total_prize (y : ℕ) (h1 : y ≤ 4)
  (h2 : total_winners = 25)
  (h3 : (total_winners - y) ≥ 5 * y)
  (h4 : prize_smart = 15)
  (h5 : prize_comprehensive = 30) :
  15 * y + 30 * (25 - y) = 690 :=
sorry

end max_smart_winners_min_total_prize_l1302_130282


namespace initial_total_quantity_l1302_130257

theorem initial_total_quantity
  (x : ℝ)
  (milk_water_ratio : 5 / 9 = 5 * x / (3 * x + 12))
  (milk_juice_ratio : 5 / 8 = 5 * x / (4 * x + 6)) :
  5 * x + 3 * x + 4 * x = 24 :=
by
  sorry

end initial_total_quantity_l1302_130257


namespace arithmetic_sequence_third_term_l1302_130227

theorem arithmetic_sequence_third_term (a d : ℤ) 
  (h20 : a + 19 * d = 17) (h21 : a + 20 * d = 20) : a + 2 * d = -34 := 
sorry

end arithmetic_sequence_third_term_l1302_130227


namespace lara_additional_miles_needed_l1302_130240

theorem lara_additional_miles_needed :
  ∀ (d1 d2 d_total t1 speed1 speed2 avg_speed : ℝ),
    d1 = 20 →
    speed1 = 25 →
    speed2 = 40 →
    avg_speed = 35 →
    t1 = d1 / speed1 →
    d_total = d1 + d2 →
    avg_speed = (d_total) / (t1 + d2 / speed2) →
    d2 = 64 :=
by sorry

end lara_additional_miles_needed_l1302_130240


namespace coolers_total_capacity_l1302_130214

theorem coolers_total_capacity :
  ∃ (C1 C2 C3 : ℕ), 
    C1 = 100 ∧ 
    C2 = C1 + (C1 / 2) ∧ 
    C3 = C2 / 2 ∧ 
    (C1 + C2 + C3 = 325) :=
sorry

end coolers_total_capacity_l1302_130214


namespace total_shaded_area_l1302_130251

/-- 
Given a 6-foot by 12-foot floor tiled with 1-foot by 1-foot tiles,
where each tile has four white quarter circles of radius 1/3 foot at its corners,
prove that the total shaded area of the floor is 72 - 8π square feet.
-/
theorem total_shaded_area :
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  total_shaded_area = 72 - 8 * Real.pi :=
by
  let floor_length := 6
  let floor_width := 12
  let tile_size := 1
  let radius := 1 / 3
  let area_of_tile := tile_size * tile_size
  let white_area_per_tile := (Real.pi * radius^2 / 4) * 4
  let shaded_area_per_tile := area_of_tile - white_area_per_tile
  let number_of_tiles := floor_length * floor_width
  let total_shaded_area := number_of_tiles * shaded_area_per_tile
  sorry

end total_shaded_area_l1302_130251


namespace slope_of_given_line_eq_l1302_130294

theorem slope_of_given_line_eq : (∀ x y : ℝ, (4 / x + 5 / y = 0) → (x ≠ 0 ∧ y ≠ 0) → ∀ y x : ℝ, y = - (5 * x / 4) → ∃ m, m = -5/4) :=
by
  sorry

end slope_of_given_line_eq_l1302_130294


namespace largest_inscribed_circle_radius_l1302_130274

theorem largest_inscribed_circle_radius (k : ℝ) (h_perimeter : 0 < k) :
  ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2) :=
by
  have h_r : ∃ (r : ℝ), r = (k / 2) * (3 - 2 * Real.sqrt 2)
  exact ⟨(k / 2) * (3 - 2 * Real.sqrt 2), rfl⟩
  exact h_r

end largest_inscribed_circle_radius_l1302_130274


namespace courtyard_width_l1302_130297

theorem courtyard_width (length : ℕ) (brick_length brick_width : ℕ) (num_bricks : ℕ) (W : ℕ)
  (H1 : length = 25)
  (H2 : brick_length = 20)
  (H3 : brick_width = 10)
  (H4 : num_bricks = 18750)
  (H5 : 2500 * (W * 100) = num_bricks * (brick_length * brick_width)) :
  W = 15 :=
by sorry

end courtyard_width_l1302_130297


namespace total_trapezoid_area_l1302_130256

def large_trapezoid_area (AB CD altitude_L : ℝ) : ℝ :=
  0.5 * (AB + CD) * altitude_L

def small_trapezoid_area (EF GH altitude_S : ℝ) : ℝ :=
  0.5 * (EF + GH) * altitude_S

def total_area (large_area small_area : ℝ) : ℝ :=
  large_area + small_area

theorem total_trapezoid_area :
  large_trapezoid_area 60 30 15 + small_trapezoid_area 25 10 5 = 762.5 :=
by
  -- proof goes here
  sorry

end total_trapezoid_area_l1302_130256


namespace find_value_of_a_l1302_130247

-- Definitions based on the conditions
def x (k : ℕ) : ℕ := 3 * k
def y (k : ℕ) : ℕ := 4 * k
def z (k : ℕ) : ℕ := 6 * k

-- Setting up the sum equation
def sum_eq_52 (k : ℕ) : Prop := x k + y k + z k = 52

-- Defining the y equation
def y_eq (a : ℚ) (k : ℕ) : Prop := y k = 15 * a + 5

-- Stating the main problem
theorem find_value_of_a (a : ℚ) (k : ℕ) : sum_eq_52 k → y_eq a k → a = 11 / 15 := by
  sorry

end find_value_of_a_l1302_130247


namespace determine_k_l1302_130275

theorem determine_k (k : ℚ) (h_collinear : ∃ (f : ℚ → ℚ), 
  f 0 = 3 ∧ f 7 = k ∧ f 21 = 2) : k = 8 / 3 :=
by
  sorry

end determine_k_l1302_130275


namespace rectangle_length_l1302_130234

theorem rectangle_length (L W : ℝ) (h1 : L = 4 * W) (h2 : L * W = 100) : L = 20 :=
by
  sorry

end rectangle_length_l1302_130234


namespace original_cost_of_horse_l1302_130229

theorem original_cost_of_horse (x : ℝ) (h : x - x^2 / 100 = 24) : x = 40 ∨ x = 60 := 
by 
  sorry

end original_cost_of_horse_l1302_130229


namespace vector_BC_l1302_130277

def vector_subtraction (v1 v2 : ℤ × ℤ) : ℤ × ℤ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem vector_BC (BA CA BC : ℤ × ℤ) (hBA : BA = (2, 3)) (hCA : CA = (4, 7)) :
  BC = vector_subtraction BA CA → BC = (-2, -4) :=
by
  intro hBC
  rw [vector_subtraction, hBA, hCA] at hBC
  simpa using hBC

end vector_BC_l1302_130277


namespace cos_diff_of_symmetric_sines_l1302_130202

theorem cos_diff_of_symmetric_sines (a β : Real) (h1 : Real.sin a = 1 / 3) 
  (h2 : Real.sin β = 1 / 3) (h3 : Real.cos a = -Real.cos β) : 
  Real.cos (a - β) = -7 / 9 := by
  sorry

end cos_diff_of_symmetric_sines_l1302_130202


namespace remaining_sweet_potatoes_l1302_130254

def harvested_sweet_potatoes : ℕ := 80
def sold_sweet_potatoes_mrs_adams : ℕ := 20
def sold_sweet_potatoes_mr_lenon : ℕ := 15
def traded_sweet_potatoes : ℕ := 10
def donated_sweet_potatoes : ℕ := 5

theorem remaining_sweet_potatoes :
  harvested_sweet_potatoes - (sold_sweet_potatoes_mrs_adams + sold_sweet_potatoes_mr_lenon + traded_sweet_potatoes + donated_sweet_potatoes) = 30 :=
by
  sorry

end remaining_sweet_potatoes_l1302_130254


namespace jelly_sold_l1302_130220

theorem jelly_sold (G S R P : ℕ) (h1 : G = 2 * S) (h2 : R = 2 * P) (h3 : R = G / 3) (h4 : P = 6) : S = 18 := by
  sorry

end jelly_sold_l1302_130220


namespace popsicles_consumed_l1302_130289

def total_minutes (hours : ℕ) (additional_minutes : ℕ) : ℕ :=
  hours * 60 + additional_minutes

def popsicles_in_time (total_time : ℕ) (interval : ℕ) : ℕ :=
  total_time / interval

theorem popsicles_consumed : popsicles_in_time (total_minutes 4 30) 15 = 18 :=
by
  -- The proof is omitted
  sorry

end popsicles_consumed_l1302_130289


namespace polynomial_degree_one_condition_l1302_130205

theorem polynomial_degree_one_condition (P : ℝ → ℝ) (c : ℝ) :
  (∀ a b : ℝ, a < b → (P = fun x => x + c) ∨ (P = fun x => -x + c)) ∧
  (∀ a b : ℝ, a < b →
    (max (P a) (P b) - min (P a) (P b) = b - a)) :=
sorry

end polynomial_degree_one_condition_l1302_130205


namespace abs_diff_of_two_numbers_l1302_130230

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : |x - y| = 6 := 
sorry

end abs_diff_of_two_numbers_l1302_130230


namespace ice_cream_ratio_l1302_130218

-- Definitions based on the conditions
def oli_scoops : ℕ := 4
def victoria_scoops : ℕ := oli_scoops + 4

-- Statement to prove the ratio
theorem ice_cream_ratio :
  victoria_scoops / oli_scoops = 2 :=
by
  -- The exact proof strategy here is omitted with 'sorry'
  sorry

end ice_cream_ratio_l1302_130218


namespace slope_of_line_l1302_130200

noncomputable def slope_range : Set ℝ :=
  {α | (5 * Real.pi / 6) ≤ α ∧ α < Real.pi}

theorem slope_of_line (x a : ℝ) :
  let k := -1 / (a^2 + Real.sqrt 3)
  ∃ α ∈ slope_range, k = Real.tan α :=
sorry

end slope_of_line_l1302_130200


namespace interval_solution_l1302_130216

-- Let the polynomial be defined
def polynomial (x : ℝ) : ℝ := x^3 - 12 * x^2 + 30 * x

-- Prove the inequality for the specified intervals
theorem interval_solution :
  { x : ℝ | polynomial x > 0 } = { x : ℝ | (0 < x ∧ x < 5) ∨ x > 6 } :=
by
  sorry

end interval_solution_l1302_130216


namespace find_integers_l1302_130273

theorem find_integers (x y : ℤ) 
  (h1 : x * y + (x + y) = 95) 
  (h2 : x * y - (x + y) = 59) : 
  (x = 11 ∧ y = 7) ∨ (x = 7 ∧ y = 11) :=
by
  sorry

end find_integers_l1302_130273


namespace jane_nail_polish_drying_time_l1302_130219

theorem jane_nail_polish_drying_time :
  let base_coat := 4
  let color_coat_1 := 5
  let color_coat_2 := 6
  let color_coat_3 := 7
  let index_finger_1 := 8
  let index_finger_2 := 10
  let middle_finger := 12
  let ring_finger := 11
  let pinky_finger := 14
  let top_coat := 9
  base_coat + color_coat_1 + color_coat_2 + color_coat_3 + index_finger_1 + index_finger_2 + middle_finger + ring_finger + pinky_finger + top_coat = 86 :=
by sorry

end jane_nail_polish_drying_time_l1302_130219


namespace total_donation_l1302_130299

theorem total_donation {carwash_proceeds bake_sale_proceeds mowing_lawn_proceeds : ℝ}
    (hc : carwash_proceeds = 100)
    (hb : bake_sale_proceeds = 80)
    (hl : mowing_lawn_proceeds = 50)
    (carwash_donation : ℝ := 0.9 * carwash_proceeds)
    (bake_sale_donation : ℝ := 0.75 * bake_sale_proceeds)
    (mowing_lawn_donation : ℝ := 1.0 * mowing_lawn_proceeds) :
    carwash_donation + bake_sale_donation + mowing_lawn_donation = 200 := by
  sorry

end total_donation_l1302_130299


namespace atLeastOneNotLessThanTwo_l1302_130213

open Real

theorem atLeastOneNotLessThanTwo (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + 1 / b < 2 ∧ b + 1 / c < 2 ∧ c + 1 / a < 2) → False := 
by
  sorry

end atLeastOneNotLessThanTwo_l1302_130213


namespace union_of_complements_l1302_130203

def U : Set ℕ := {x | 0 ≤ x ∧ x < 6}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {x | x^2 + 4 = 5 * x}
def complement_U (S : Set ℕ) : Set ℕ := {x ∈ U | x ∉ S}

theorem union_of_complements :
  complement_U A ∪ complement_U B = {0, 2, 3, 4, 5} := by
sorry

end union_of_complements_l1302_130203


namespace interest_group_selections_l1302_130265

-- Define the number of students and the number of interest groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem statement: The total number of different possible selections of interest groups is 81.
theorem interest_group_selections : num_groups ^ num_students = 81 := by
  sorry

end interest_group_selections_l1302_130265
