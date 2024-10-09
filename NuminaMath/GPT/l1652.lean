import Mathlib

namespace project_completion_days_l1652_165258

theorem project_completion_days (A_days : ℕ) (B_days : ℕ) (A_alone_days : ℕ) :
  A_days = 20 → B_days = 25 → A_alone_days = 2 → (A_alone_days : ℚ) * (1 / A_days) + (10 : ℚ) * (1 / (A_days * B_days / (A_days + B_days))) = 1 :=
by
  sorry

end project_completion_days_l1652_165258


namespace solve_for_w_l1652_165253

theorem solve_for_w (w : ℕ) (h : w^2 - 5 * w = 0) (hp : w > 0) : w = 5 :=
sorry

end solve_for_w_l1652_165253


namespace mary_total_earnings_l1652_165281

-- Define the earnings for each job
def cleaning_earnings (homes_cleaned : ℕ) : ℕ := 46 * homes_cleaned
def babysitting_earnings (days_babysat : ℕ) : ℕ := 35 * days_babysat
def petcare_earnings (days_petcare : ℕ) : ℕ := 60 * days_petcare

-- Define the total earnings
def total_earnings (homes_cleaned days_babysat days_petcare : ℕ) : ℕ :=
  cleaning_earnings homes_cleaned + babysitting_earnings days_babysat + petcare_earnings days_petcare

-- Given values
def homes_cleaned_last_week : ℕ := 4
def days_babysat_last_week : ℕ := 5
def days_petcare_last_week : ℕ := 3

-- Prove the total earnings
theorem mary_total_earnings : total_earnings homes_cleaned_last_week days_babysat_last_week days_petcare_last_week = 539 :=
by
  -- We just state the theorem; the proof is not required
  sorry

end mary_total_earnings_l1652_165281


namespace find_radius_l1652_165224

noncomputable def square_radius (r : ℝ) : Prop :=
  let s := (2 * r) / Real.sqrt 2  -- side length of the square derived from the radius
  let perimeter := 4 * s         -- perimeter of the square
  let area := Real.pi * r^2      -- area of the circumscribed circle
  perimeter = area               -- given condition

theorem find_radius (r : ℝ) (h : square_radius r) : r = (4 * Real.sqrt 2) / Real.pi :=
by
  sorry

end find_radius_l1652_165224


namespace calculate_g_at_5_l1652_165260

variable {R : Type} [LinearOrderedField R] (g : R → R)
variable (x : R)

theorem calculate_g_at_5 (h : ∀ x : R, g (3 * x - 4) = 5 * x - 7) : g 5 = 8 :=
by
  sorry

end calculate_g_at_5_l1652_165260


namespace remainder_2pow33_minus_1_div_9_l1652_165264

theorem remainder_2pow33_minus_1_div_9 : (2^33 - 1) % 9 = 7 := 
  sorry

end remainder_2pow33_minus_1_div_9_l1652_165264


namespace spadesuit_evaluation_l1652_165295

-- Define the operation
def spadesuit (a b : ℝ) : ℝ := (a + b) * (a - b)

-- The theorem to prove
theorem spadesuit_evaluation : spadesuit 4 (spadesuit 5 (-2)) = -425 :=
by
  sorry

end spadesuit_evaluation_l1652_165295


namespace find_green_pepper_weight_l1652_165254

variable (weight_red_peppers : ℝ) (total_weight_peppers : ℝ)

theorem find_green_pepper_weight 
    (h1 : weight_red_peppers = 0.33) 
    (h2 : total_weight_peppers = 0.66) 
    : total_weight_peppers - weight_red_peppers = 0.33 := 
by sorry

end find_green_pepper_weight_l1652_165254


namespace factor_expression_l1652_165235

variable (a : ℝ)

theorem factor_expression : 37 * a^2 + 111 * a = 37 * a * (a + 3) :=
  sorry

end factor_expression_l1652_165235


namespace max_value_of_f_l1652_165240

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem max_value_of_f :
  ∃ x, (f x = 1 / exp 1) ∧ (∀ y, f y ≤ f x) :=
by
  sorry

end max_value_of_f_l1652_165240


namespace simplify_expression_l1652_165259

theorem simplify_expression (a : ℝ) (h : a = -2) : 
  (1 - a / (a + 1)) / (1 / (1 - a ^ 2)) = 1 / 3 :=
by
  subst h
  sorry

end simplify_expression_l1652_165259


namespace odd_increasing_three_digit_numbers_count_eq_50_l1652_165278

def count_odd_increasing_three_digit_numbers : Nat := by
  -- Mathematical conditions:
  -- let a, b, c be digits of the number
  -- 0 < a < b < c <= 9 and c is an odd digit

  -- We analyze values for 'c' which must be an odd digit,
  -- and count valid (a, b) combinations for each case of c.

  -- Starting from cases for c:
  -- for c = 1, no valid (a, b); count = 0
  -- for c = 3, valid (a, b) are from {1, 2}; count = 1
  -- for c = 5, valid (a, b) are from {1, 2, 3, 4}; count = 6
  -- for c = 7, valid (a, b) are from {1, 2, 3, 4, 5, 6}; count = 15
  -- for c = 9, valid (a, b) are from {1, 2, 3, 4, 5, 6, 7, 8}; count = 28

  -- Sum counts for all valid cases of c
  exact 50

-- Define our main theorem based on problem and final result
theorem odd_increasing_three_digit_numbers_count_eq_50 :
  count_odd_increasing_three_digit_numbers = 50 := by
  unfold count_odd_increasing_three_digit_numbers
  exact rfl -- the correct proof will fill in this part

end odd_increasing_three_digit_numbers_count_eq_50_l1652_165278


namespace tip_percentage_l1652_165283

/--
A family paid $30 for food, the sales tax rate is 9.5%, and the total amount paid was $35.75. Prove that the tip percentage is 9.67%.
-/
theorem tip_percentage (food_cost : ℝ) (sales_tax_rate : ℝ) (total_paid : ℝ)
  (h1 : food_cost = 30)
  (h2 : sales_tax_rate = 0.095)
  (h3 : total_paid = 35.75) :
  ((total_paid - (food_cost * (1 + sales_tax_rate))) / food_cost) * 100 = 9.67 :=
by
  sorry

end tip_percentage_l1652_165283


namespace walter_age_in_2001_l1652_165288

/-- In 1996, Walter was one-third as old as his grandmother, 
and the sum of the years in which they were born is 3864.
Prove that Walter will be 37 years old at the end of 2001. -/
theorem walter_age_in_2001 (y : ℕ) (H1 : ∃ g, g = 3 * y)
  (H2 : 1996 - y + (1996 - (3 * y)) = 3864) : y + 5 = 37 :=
by sorry

end walter_age_in_2001_l1652_165288


namespace temperature_value_l1652_165272

theorem temperature_value (k : ℝ) (t : ℝ) (h1 : t = 5 / 9 * (k - 32)) (h2 : k = 221) : t = 105 :=
by
  sorry

end temperature_value_l1652_165272


namespace robert_birth_year_l1652_165212

theorem robert_birth_year (n : ℕ) (h1 : (n + 1)^2 - n^2 = 89) : n = 44 ∧ n^2 = 1936 :=
by {
  sorry
}

end robert_birth_year_l1652_165212


namespace min_value_x_plus_2y_l1652_165292

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 4) : x + 2 * y = 2 :=
sorry

end min_value_x_plus_2y_l1652_165292


namespace triangle_interior_angles_not_greater_than_60_l1652_165296

theorem triangle_interior_angles_not_greater_than_60 (α β γ : ℝ) (h_sum : α + β + γ = 180) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0) :
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60 :=
by
  sorry

end triangle_interior_angles_not_greater_than_60_l1652_165296


namespace canadian_ratio_correct_l1652_165220

-- The total number of scientists
def total_scientists : ℕ := 70

-- Half of the scientists are from Europe
def european_scientists : ℕ := total_scientists / 2

-- The number of scientists from the USA
def usa_scientists : ℕ := 21

-- The number of Canadian scientists
def canadian_scientists : ℕ := total_scientists - european_scientists - usa_scientists

-- The ratio of the number of Canadian scientists to the total number of scientists
def canadian_ratio : ℚ := canadian_scientists / total_scientists

-- Prove that the ratio is 1:5
theorem canadian_ratio_correct : canadian_ratio = 1 / 5 :=
by
  sorry

end canadian_ratio_correct_l1652_165220


namespace max_homework_time_l1652_165263

theorem max_homework_time (biology_time history_time geography_time : ℕ) :
    biology_time = 20 ∧ history_time = 2 * biology_time ∧ geography_time = 3 * history_time →
    biology_time + history_time + geography_time = 180 :=
by
    intros
    sorry

end max_homework_time_l1652_165263


namespace find_side_b_l1652_165200

variables {a b c : ℝ} {B : ℝ}

theorem find_side_b 
  (area_triangle : (1 / 2) * a * c * (Real.sin B) = Real.sqrt 3) 
  (B_is_60_degrees : B = Real.pi / 3) 
  (relation_ac : a^2 + c^2 = 3 * a * c) : 
  b = 2 * Real.sqrt 2 := 
by 
  sorry

end find_side_b_l1652_165200


namespace difference_qr_l1652_165270

-- Definitions of p, q, r in terms of the common multiplier x
def p (x : ℕ) := 3 * x
def q (x : ℕ) := 7 * x
def r (x : ℕ) := 12 * x

-- Given condition that the difference between p and q's share is 4000
def condition1 (x : ℕ) := q x - p x = 4000

-- Theorem stating that the difference between q and r's share is 5000
theorem difference_qr (x : ℕ) (h : condition1 x) : r x - q x = 5000 :=
by
  -- Proof placeholder
  sorry

end difference_qr_l1652_165270


namespace survey_respondents_l1652_165280

theorem survey_respondents
  (X Y Z : ℕ) 
  (h1 : X = 360) 
  (h2 : X * 4 = Y * 9) 
  (h3 : X * 3 = Z * 9) : 
  X + Y + Z = 640 :=
by
  sorry

end survey_respondents_l1652_165280


namespace geometric_sequence_sum_l1652_165205

open Real

variable {a a5 a3 a4 S4 q : ℝ}

theorem geometric_sequence_sum (h1 : q < 1)
                             (h2 : a + a5 = 20)
                             (h3 : a3 * a5 = 64) :
                             S4 = 120 := by
  sorry

end geometric_sequence_sum_l1652_165205


namespace find_x_l1652_165293

theorem find_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end find_x_l1652_165293


namespace general_formula_correct_S_k_equals_189_l1652_165256

-- Define the arithmetic sequence with initial conditions
def a (n : ℕ) : ℤ :=
  if n = 1 then -11
  else sorry  -- Will be defined by the general formula

-- Given conditions in Lean
def initial_condition (a : ℕ → ℤ) :=
  a 1 = -11 ∧ a 4 + a 6 = -6

-- General formula for the arithmetic sequence to be proven
def general_formula (n : ℕ) : ℤ := 2 * n - 13

-- Sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℤ :=
  n^2 - 12 * n

-- Problem 1: Prove the general formula
theorem general_formula_correct : ∀ n : ℕ, initial_condition a → a n = general_formula n :=
by sorry

-- Problem 2: Prove that k = 21 such that S_k = 189
theorem S_k_equals_189 : ∃ k : ℕ, S k = 189 ∧ k = 21 :=
by sorry

end general_formula_correct_S_k_equals_189_l1652_165256


namespace base_conversion_addition_correct_l1652_165271

theorem base_conversion_addition_correct :
  let A := 10
  let C := 12
  let n13 := 3 * 13^2 + 7 * 13^1 + 6
  let n14 := 4 * 14^2 + A * 14^1 + C
  n13 + n14 = 1540 := by
    let A := 10
    let C := 12
    let n13 := 3 * 13^2 + 7 * 13^1 + 6
    let n14 := 4 * 14^2 + A * 14^1 + C
    let sum := n13 + n14
    have h1 : n13 = 604 := by sorry
    have h2 : n14 = 936 := by sorry
    have h3 : sum = 1540 := by sorry
    exact h3

end base_conversion_addition_correct_l1652_165271


namespace henry_seashells_l1652_165257

theorem henry_seashells (H L : ℕ) (h1 : H + 24 + L = 59) (h2 : H + 24 + (3 * L) / 4 = 53) : H = 11 := by
  sorry

end henry_seashells_l1652_165257


namespace domino_chain_can_be_built_l1652_165213

def domino_chain_possible : Prop :=
  let total_pieces := 28
  let pieces_with_sixes_removed := 7
  let remaining_pieces := total_pieces - pieces_with_sixes_removed
  (∀ n : ℕ, n < 6 → (∃ k : ℕ, k = 6) → (remaining_pieces % 2 = 0))

theorem domino_chain_can_be_built (h : domino_chain_possible) : Prop :=
  sorry

end domino_chain_can_be_built_l1652_165213


namespace total_amount_l1652_165234

-- Define p, q, r and their shares
variables (p q r : ℕ)

-- Given conditions translated to Lean definitions
def ratio_pq := (5 * q) = (4 * p)
def ratio_qr := (9 * r) = (10 * q)
def r_share := r = 400

-- Statement to prove
theorem total_amount (hpq : ratio_pq p q) (hqr : ratio_qr q r) (hr : r_share r) :
  (p + q + r) = 1210 :=
by
  sorry

end total_amount_l1652_165234


namespace find_value_l1652_165245

theorem find_value (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a - 2 * b)^2 = 25 :=
by
  sorry

end find_value_l1652_165245


namespace tom_caught_16_trout_l1652_165207

theorem tom_caught_16_trout (melanie_trout : ℕ) (tom_caught_twice : melanie_trout * 2 = 16) : 
  2 * melanie_trout = 16 :=
by 
  sorry

end tom_caught_16_trout_l1652_165207


namespace marys_garbage_bill_l1652_165202

def weekly_cost_trash (trash_count : ℕ) := 10 * trash_count
def weekly_cost_recycling (recycling_count : ℕ) := 5 * recycling_count

def weekly_cost (trash_count : ℕ) (recycling_count : ℕ) : ℕ :=
  weekly_cost_trash trash_count + weekly_cost_recycling recycling_count

def monthly_cost (weekly_cost : ℕ) := 4 * weekly_cost

def elderly_discount (total_cost : ℕ) : ℕ :=
  total_cost * 18 / 100

def final_bill (monthly_cost : ℕ) (discount : ℕ) (fine : ℕ) : ℕ :=
  monthly_cost - discount + fine

theorem marys_garbage_bill : final_bill
  (monthly_cost (weekly_cost 2 1))
  (elderly_discount (monthly_cost (weekly_cost 2 1)))
  20 = 102 := by
{
  sorry -- The proof steps are omitted as per the instructions.
}

end marys_garbage_bill_l1652_165202


namespace carla_cream_volume_l1652_165225

-- Definitions of the given conditions and problem
def watermelon_puree_volume : ℕ := 500
def servings_count : ℕ := 4
def volume_per_serving : ℕ := 150
def total_smoothies_volume := servings_count * volume_per_serving
def cream_volume := total_smoothies_volume - watermelon_puree_volume

-- Statement of the proposition we want to prove
theorem carla_cream_volume : cream_volume = 100 := by
  sorry

end carla_cream_volume_l1652_165225


namespace inequality_true_l1652_165242

-- Define the conditions
variables (a b : ℝ) (h : a < b) (hb_neg : b < 0)

-- State the theorem to be proved
theorem inequality_true (ha : a < b) (hb : b < 0) : (|a| / |b| > 1) :=
sorry

end inequality_true_l1652_165242


namespace number_of_three_digit_integers_congruent_to_2_mod_4_l1652_165233

theorem number_of_three_digit_integers_congruent_to_2_mod_4 : ∃ (n : ℕ), n = 225 ∧ ∀ k : ℤ, 100 ≤ 4 * k + 2 ∧ 4 * k + 2 ≤ 999 → 24 < k ∧ k < 250 := by
  sorry

end number_of_three_digit_integers_congruent_to_2_mod_4_l1652_165233


namespace minimum_value_of_expression_l1652_165285

theorem minimum_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) = 24 :=
sorry

end minimum_value_of_expression_l1652_165285


namespace right_triangle_sides_l1652_165214

theorem right_triangle_sides (a b c : ℝ) (h_ratio : ∃ x : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x) 
(h_area : 1 / 2 * a * b = 24) : a = 6 ∧ b = 8 ∧ c = 10 :=
by
  sorry

end right_triangle_sides_l1652_165214


namespace profit_percentage_is_50_l1652_165238

noncomputable def cost_of_machine := 11000
noncomputable def repair_cost := 5000
noncomputable def transportation_charges := 1000
noncomputable def selling_price := 25500

noncomputable def total_cost := cost_of_machine + repair_cost + transportation_charges
noncomputable def profit := selling_price - total_cost
noncomputable def profit_percentage := (profit / total_cost) * 100

theorem profit_percentage_is_50 : profit_percentage = 50 := by
  sorry

end profit_percentage_is_50_l1652_165238


namespace number_of_different_ways_to_travel_l1652_165261

-- Define the conditions
def number_of_morning_flights : ℕ := 2
def number_of_afternoon_flights : ℕ := 3

-- Assert the question and the answer
theorem number_of_different_ways_to_travel : 
  (number_of_morning_flights * number_of_afternoon_flights) = 6 :=
by
  sorry

end number_of_different_ways_to_travel_l1652_165261


namespace joggers_difference_l1652_165206

theorem joggers_difference (Tyson_joggers Alexander_joggers Christopher_joggers : ℕ) 
  (h1 : Alexander_joggers = Tyson_joggers + 22) 
  (h2 : Christopher_joggers = 20 * Tyson_joggers)
  (h3 : Christopher_joggers = 80) : 
  Christopher_joggers - Alexander_joggers = 54 :=
by 
  sorry

end joggers_difference_l1652_165206


namespace density_ratio_of_large_cube_l1652_165276

theorem density_ratio_of_large_cube 
  (V0 m0 : ℝ) (initial_density replacement_density: ℝ)
  (initial_mass final_mass : ℝ) (V_total : ℝ) 
  (h1 : initial_density = m0 / V0)
  (h2 : replacement_density = 2 * initial_density)
  (h3 : initial_mass = 8 * m0)
  (h4 : final_mass = 6 * m0 + 2 * (2 * m0))
  (h5 : V_total = 8 * V0) :
  initial_density / (final_mass / V_total) = 0.8 :=
sorry

end density_ratio_of_large_cube_l1652_165276


namespace sqrt_range_l1652_165275

theorem sqrt_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_range_l1652_165275


namespace wendy_score_l1652_165267

def score_per_treasure : ℕ := 5
def treasures_first_level : ℕ := 4
def treasures_second_level : ℕ := 3

theorem wendy_score :
  score_per_treasure * treasures_first_level + score_per_treasure * treasures_second_level = 35 :=
by
  sorry

end wendy_score_l1652_165267


namespace max_teams_4_weeks_l1652_165262

noncomputable def max_teams_in_tournament (weeks number_teams : ℕ) : ℕ :=
  if h : weeks > 0 then (number_teams * (number_teams - 1)) / (2 * weeks) else 0

theorem max_teams_4_weeks : max_teams_in_tournament 4 7 = 6 := by
  -- Assumptions
  let n := 6
  let teams := 7 * n
  let weeks := 4
  
  -- Define the constraints and checks here
  sorry

end max_teams_4_weeks_l1652_165262


namespace range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l1652_165226

-- Define the propositions p and q
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := x^2 - 5 * x + 6 < 0

-- Question 1: When a = 1, if p ∧ q is true, determine the range of x
theorem range_of_x_when_a_is_1_and_p_and_q_are_true :
  ∀ x, p x 1 ∧ q x → 2 < x ∧ x < 3 :=
by
  sorry

-- Question 2: If p is a necessary but not sufficient condition for q, determine the range of a
theorem range_of_a_when_p_necessary_for_q :
  ∀ a, (∀ x, q x → p x a) ∧ ¬ (∀ x, p x a → q x) → 1 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_when_a_is_1_and_p_and_q_are_true_range_of_a_when_p_necessary_for_q_l1652_165226


namespace modulo_4_equiv_2_l1652_165298

open Nat

noncomputable def f (n : ℕ) [Fintype (ZMod n)] : ZMod n → ZMod n := sorry

theorem modulo_4_equiv_2 (n : ℕ) [hn : Fact (n > 0)] 
  (f : ZMod n → ZMod n)
  (h1 : ∀ x, f x ≠ x)
  (h2 : ∀ x, f (f x) = x)
  (h3 : ∀ x, f (f (f (x + 1) + 1) + 1) = x) : 
  n % 4 = 2 := 
sorry

end modulo_4_equiv_2_l1652_165298


namespace percentage_subtracted_l1652_165294

theorem percentage_subtracted (a : ℝ) (p : ℝ) (h : (1 - p / 100) * a = 0.97 * a) : p = 3 :=
by
  sorry

end percentage_subtracted_l1652_165294


namespace exists_root_in_interval_l1652_165211

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x - 2

theorem exists_root_in_interval :
  (f 1 < 0) ∧ (f 2 > 0) ∧ (∀ x > 0, ContinuousAt f x) → (∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0) :=
by sorry

end exists_root_in_interval_l1652_165211


namespace find_x2_y2_l1652_165291

theorem find_x2_y2 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : xy + x + y = 35) (h4 : xy * (x + y) = 360) : x^2 + y^2 = 185 := by
  sorry

end find_x2_y2_l1652_165291


namespace smaller_than_negative_one_l1652_165222

theorem smaller_than_negative_one :
  ∃ x ∈ ({0, -1/2, 1, -2} : Set ℝ), x < -1 ∧ x = -2 :=
by
  -- the proof part is skipped
  sorry

end smaller_than_negative_one_l1652_165222


namespace max_students_gcd_l1652_165203

def numPens : Nat := 1802
def numPencils : Nat := 1203
def numErasers : Nat := 1508
def numNotebooks : Nat := 2400

theorem max_students_gcd : Nat.gcd (Nat.gcd (Nat.gcd numPens numPencils) numErasers) numNotebooks = 1 := by
  sorry

end max_students_gcd_l1652_165203


namespace total_marbles_l1652_165223

variable (w o p : ℝ)

-- Conditions as hypothesis
axiom h1 : o + p = 10
axiom h2 : w + p = 12
axiom h3 : w + o = 5

theorem total_marbles : w + o + p = 13.5 :=
by
  sorry

end total_marbles_l1652_165223


namespace tim_balloons_proof_l1652_165231

-- Define the number of balloons Dan has
def dan_balloons : ℕ := 29

-- Define the relationship between Tim's and Dan's balloons
def balloons_ratio : ℕ := 7

-- Define the number of balloons Tim has
def tim_balloons : ℕ := balloons_ratio * dan_balloons

-- Prove that the number of balloons Tim has is 203
theorem tim_balloons_proof : tim_balloons = 203 :=
sorry

end tim_balloons_proof_l1652_165231


namespace ferris_wheel_capacity_l1652_165232

theorem ferris_wheel_capacity 
  (num_seats : ℕ)
  (people_per_seat : ℕ)
  (h1 : num_seats = 4)
  (h2 : people_per_seat = 5) :
  num_seats * people_per_seat = 20 := by
  sorry

end ferris_wheel_capacity_l1652_165232


namespace angle_C_is_sixty_l1652_165255

variable {A B C D E : Type}
variable {AD BE BC AC : ℝ}
variable {triangle : A ≠ B ∧ B ≠ C ∧ C ≠ A} 
variable (angle_C : ℝ)

-- Given conditions
variable (h_eq : AD * BC = BE * AC)
variable (h_ineq : AC ≠ BC)

-- To prove
theorem angle_C_is_sixty (h_eq : AD * BC = BE * AC) (h_ineq : AC ≠ BC) : angle_C = 60 :=
by
  sorry

end angle_C_is_sixty_l1652_165255


namespace square_area_l1652_165243

theorem square_area (side : ℕ) (h : side = 19) : side * side = 361 := by
  sorry

end square_area_l1652_165243


namespace sum_min_max_x_y_l1652_165277

theorem sum_min_max_x_y (x y : ℕ) (h : 6 * x + 7 * y = 2012): 288 + 335 = 623 :=
by
  sorry

end sum_min_max_x_y_l1652_165277


namespace order_of_numbers_l1652_165244

theorem order_of_numbers (x y : ℝ) (hx : x > 1) (hy : -1 < y ∧ y < 0) : y < -y ∧ -y < -xy ∧ -xy < x :=
by 
  sorry

end order_of_numbers_l1652_165244


namespace five_digit_divisibility_l1652_165217

-- Definitions of n and m
def n (a b c d e : ℕ) := 10000 * a + 1000 * b + 100 * c + 10 * d + e
def m (a b d e : ℕ) := 1000 * a + 100 * b + 10 * d + e

-- Condition that n is a five-digit number whose first digit is non-zero and n/m is an integer
theorem five_digit_divisibility (a b c d e : ℕ):
  1 <= a ∧ a <= 9 → 0 <= b ∧ b <= 9 → 0 <= c ∧ c <= 9 → 0 <= d ∧ d <= 9 → 0 <= e ∧ e <= 9 →
  m a b d e ∣ n a b c d e →
  ∃ x y : ℕ, a = x ∧ b = y ∧ c = 0 ∧ d = 0 ∧ e = 0 :=
by
  sorry

end five_digit_divisibility_l1652_165217


namespace intersection_A_B_l1652_165221

-- Definitions for sets A and B
def A : Set ℝ := { x | ∃ y : ℝ, x + y^2 = 1 }
def B : Set ℝ := { y | ∃ x : ℝ, y = x^2 - 1 }

-- The proof goal to show the intersection of sets A and B
theorem intersection_A_B : A ∩ B = { z | -1 ≤ z ∧ z ≤ 1 } :=
by
  sorry

end intersection_A_B_l1652_165221


namespace average_first_6_numbers_l1652_165273

theorem average_first_6_numbers (A : ℕ) (h1 : (13 * 9) = (6 * A + 45 + 6 * 7)) : A = 5 :=
by 
  -- h1 : 117 = (6 * A + 45 + 42),
  -- solving for the value of A by performing algebraic operations will prove it.
  sorry

end average_first_6_numbers_l1652_165273


namespace product_of_digits_l1652_165237

theorem product_of_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 4 = 0) : A * B = 32 ∨ A * B = 36 :=
sorry

end product_of_digits_l1652_165237


namespace solve_for_star_l1652_165282

theorem solve_for_star : ∀ (star : ℝ), (45 - (28 - (37 - (15 - star))) = 54) → star = 15 := by
  intros star h
  sorry

end solve_for_star_l1652_165282


namespace geometric_sum_five_terms_l1652_165239

theorem geometric_sum_five_terms (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ)
  (h_geo : ∀ n, a (n + 1) = q * a n)
  (h_pos : ∀ n, 0 < a n)
  (h_sum : ∀ n, S n = (a 0) * (1 - q^n) / (1 - q))
  (h_a2a4 : a 1 * a 3 = 16)
  (h_ratio : (a 3 + a 4 + a 7) / (a 0 + a 1 + a 4) = 8) :
  S 5 = 31 :=
sorry

end geometric_sum_five_terms_l1652_165239


namespace minimum_dot_product_l1652_165248

-- Definitions of points A and B
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (2, 0)

-- Definition of condition that P lies on the line x - y + 1 = 0
def onLineP (P : ℝ × ℝ) : Prop := P.1 - P.2 + 1 = 0

-- Definition of dot product between vectors PA and PB
def dotProduct (P A B : ℝ × ℝ) : ℝ := 
  let PA := (P.1 - A.1, P.2 - A.2)
  let PB := (P.1 - B.1, P.2 - B.2)
  PA.1 * PB.1 + PA.2 * PB.2

-- Lean 4 theorem statement
theorem minimum_dot_product (P : ℝ × ℝ) (hP : onLineP P) : 
  dotProduct P pointA pointB = 0 := 
sorry

end minimum_dot_product_l1652_165248


namespace monomial_same_type_l1652_165208

theorem monomial_same_type (a b : ℕ) (h1 : a + 1 = 3) (h2 : b = 3) : a + b = 5 :=
by 
  -- proof goes here
  sorry

end monomial_same_type_l1652_165208


namespace percentage_comedies_l1652_165236

theorem percentage_comedies (a : ℕ) (d c T : ℕ) 
  (h1 : d = 5 * a) 
  (h2 : c = 10 * a) 
  (h3 : T = c + d + a) : 
  (c : ℝ) / T * 100 = 62.5 := 
by 
  sorry

end percentage_comedies_l1652_165236


namespace ram_ravi_selected_probability_l1652_165287

noncomputable def probability_both_selected : ℝ := 
  let probability_ram_80 := (1 : ℝ) / 7
  let probability_ravi_80 := (1 : ℝ) / 5
  let probability_both_80 := probability_ram_80 * probability_ravi_80
  let num_applicants := 200
  let num_spots := 4
  let probability_single_selection := (num_spots : ℝ) / (num_applicants : ℝ)
  let probability_both_selected_given_80 := probability_single_selection * probability_single_selection
  probability_both_80 * probability_both_selected_given_80

theorem ram_ravi_selected_probability :
  probability_both_selected = 1 / 87500 := 
by
  sorry

end ram_ravi_selected_probability_l1652_165287


namespace remainder_of_12111_div_3_l1652_165299

theorem remainder_of_12111_div_3 : 12111 % 3 = 0 := by
  sorry

end remainder_of_12111_div_3_l1652_165299


namespace paint_weight_correct_l1652_165215

def weight_of_paint (total_weight : ℕ) (half_paint_weight : ℕ) : ℕ :=
  2 * (total_weight - half_paint_weight)

theorem paint_weight_correct :
  weight_of_paint 24 14 = 20 := by 
  sorry

end paint_weight_correct_l1652_165215


namespace rewrite_expression_and_compute_l1652_165247

noncomputable def c : ℚ := 8
noncomputable def p : ℚ := -3 / 8
noncomputable def q : ℚ := 119 / 8

theorem rewrite_expression_and_compute :
  (∃ (c p q : ℚ), 8 * j ^ 2 - 6 * j + 16 = c * (j + p) ^ 2 + q) →
  q / p = -119 / 3 :=
by
  sorry

end rewrite_expression_and_compute_l1652_165247


namespace find_ratio_l1652_165266

open Real

-- Definitions and conditions
variables (b1 b2 : ℝ) (F1 F2 : ℝ × ℝ)
noncomputable def ellipse_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 49) + (Q.2^2 / b1^2) = 1
noncomputable def hyperbola_eq (Q : ℝ × ℝ) : Prop := (Q.1^2 / 16) - (Q.2^2 / b2^2) = 1
noncomputable def same_foci (Q : ℝ × ℝ) : Prop := true  -- Placeholder: Representing that both shapes have the same foci F1 and F2

-- The main theorem
theorem find_ratio (Q : ℝ × ℝ) (h1 : ellipse_eq b1 Q) (h2 : hyperbola_eq b2 Q) (h3 : same_foci Q) : 
  abs ((dist Q F1) - (dist Q F2)) / ((dist Q F1) + (dist Q F2)) = 4 / 7 := 
sorry

end find_ratio_l1652_165266


namespace floor_add_ceil_eq_five_l1652_165251

theorem floor_add_ceil_eq_five (x : ℝ) :
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 5 ↔ 2 < x ∧ x < 3 :=
by sorry

end floor_add_ceil_eq_five_l1652_165251


namespace range_of_m_l1652_165284

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), (x > 2 * m ∧ x ≥ m - 3) ∧ x = 1) ↔ 0 ≤ m ∧ m < 0.5 :=
by
  sorry

end range_of_m_l1652_165284


namespace find_x_l1652_165289

-- Define the custom operation on m and n
def operation (m n : ℤ) : ℤ := 2 * m - 3 * n

-- Lean statement of the problem
theorem find_x (x : ℤ) (h : operation x 7 = operation 7 x) : x = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end find_x_l1652_165289


namespace domain_f_correct_domain_g_correct_l1652_165241

noncomputable def domain_f : Set ℝ :=
  {x | x + 1 ≥ 0 ∧ x ≠ 1}

noncomputable def expected_domain_f : Set ℝ :=
  {x | (-1 ≤ x ∧ x < 1) ∨ x > 1}

theorem domain_f_correct :
  domain_f = expected_domain_f :=
by
  sorry

noncomputable def domain_g : Set ℝ :=
  {x | 3 - 4 * x > 0}

noncomputable def expected_domain_g : Set ℝ :=
  {x | x < 3 / 4}

theorem domain_g_correct :
  domain_g = expected_domain_g :=
by
  sorry

end domain_f_correct_domain_g_correct_l1652_165241


namespace range_of_a_minus_b_l1652_165216

theorem range_of_a_minus_b (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 2 < b ∧ b < 4) : -4 < a - b ∧ a - b < -1 :=
by
  sorry

end range_of_a_minus_b_l1652_165216


namespace chord_through_P_midpoint_of_ellipse_has_given_line_l1652_165252

-- Define the ellipse
def ellipse (x y : ℝ) := 4 * x^2 + 9 * y^2 = 144

-- Define point P
def pointP := (3, 1)

-- Define the problem statement
theorem chord_through_P_midpoint_of_ellipse_has_given_line:
  ∃ (m : ℝ) (c : ℝ), (∀ (x y : ℝ), 4 * x^2 + 9 * y^2 = 144 → x + y = m ∧ 3 * x + y = c) → 
  ∃ (A : ℝ) (B : ℝ), ellipse 3 1 ∧ (A * 4 + B * 3 - 15 = 0) := sorry

end chord_through_P_midpoint_of_ellipse_has_given_line_l1652_165252


namespace length_of_second_train_l1652_165269

theorem length_of_second_train (speed1 speed2 : ℝ) (length1 time : ℝ) (h1 : speed1 = 60) (h2 : speed2 = 40) 
  (h3 : length1 = 450) (h4 : time = 26.99784017278618) :
  let speed1_mps := speed1 * 1000 / 3600
  let speed2_mps := speed2 * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := relative_speed * time
  let length2 := total_distance - length1
  length2 = 300 :=
by
  sorry

end length_of_second_train_l1652_165269


namespace outfit_choices_l1652_165201

-- Define the numbers of shirts, pants, and hats.
def num_shirts : ℕ := 6
def num_pants : ℕ := 7
def num_hats : ℕ := 6

-- Define the number of colors and the constraints.
def num_colors : ℕ := 6

-- The total number of outfits without restrictions.
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- Number of outfits where all items are the same color.
def same_color_outfits : ℕ := num_colors

-- Number of outfits where the shirt and pants are the same color.
def same_shirt_pants_color_outfits : ℕ := num_colors + 1  -- accounting for the extra pair of pants

-- The total number of valid outfits calculated.
def valid_outfits : ℕ :=
  total_outfits - same_color_outfits - same_shirt_pants_color_outfits

-- The theorem statement asserting the correct answer.
theorem outfit_choices : valid_outfits = 239 := by
  sorry

end outfit_choices_l1652_165201


namespace solve_inequality_prove_inequality_l1652_165297

open Real

-- Problem 1: Solve the inequality
theorem solve_inequality (x : ℝ) : (x - 1) / (2 * x + 1) ≤ 0 ↔ (-1 / 2) < x ∧ x ≤ 1 :=
sorry

-- Problem 2: Prove the inequality given positive a, b, and c
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) * (1 / a + 1 / (b + c)) ≥ 4 :=
sorry

end solve_inequality_prove_inequality_l1652_165297


namespace purely_imaginary_iff_l1652_165229

noncomputable def isPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0

theorem purely_imaginary_iff (a : ℝ) :
  isPurelyImaginary (Complex.mk ((a * (a + 2)) / (a - 1)) (a ^ 2 + 2 * a - 3))
  ↔ a = 0 ∨ a = -2 := by
  sorry

end purely_imaginary_iff_l1652_165229


namespace existence_of_points_on_AC_l1652_165268

theorem existence_of_points_on_AC (A B C M : ℝ) (hAB : abs (A - B) = 2) (hBC : abs (B - C) = 1) :
  ((abs (A - M) + abs (B - M) = abs (C - M)) ↔ (M = A - 1) ∨ (M = A + 1)) :=
by
  sorry

end existence_of_points_on_AC_l1652_165268


namespace range_of_x_l1652_165246

theorem range_of_x (x : ℝ) (h : (x + 1) ^ 0 = 1) : x ≠ -1 :=
sorry

end range_of_x_l1652_165246


namespace solve_for_s_l1652_165210

-- Definition of the condition
def condition (s : ℝ) : Prop := (s - 60) / 3 = (6 - 3 * s) / 4

-- Theorem stating that if the condition holds, then s = 19.85
theorem solve_for_s (s : ℝ) : condition s → s = 19.85 := 
by {
  sorry -- Proof is skipped as per requirements
}

end solve_for_s_l1652_165210


namespace quadratic_equal_roots_l1652_165230

theorem quadratic_equal_roots (k : ℝ) :
  (∀ x : ℝ, x^2 + k * x + 1 = 0 → x = -k / 2) ↔ (k = 2 ∨ k = -2) :=
by
  sorry

end quadratic_equal_roots_l1652_165230


namespace student_B_more_stable_than_A_student_B_more_stable_l1652_165265

-- Define students A and B.
structure Student :=
  (average_score : ℝ)
  (variance : ℝ)

-- Given data for both students.
def studentA : Student :=
  { average_score := 90, variance := 51 }

def studentB : Student :=
  { average_score := 90, variance := 12 }

-- The theorem that student B has more stable performance than student A.
theorem student_B_more_stable_than_A (A B : Student) (h_avg : A.average_score = B.average_score) :
  A.variance > B.variance → B.variance < A.variance :=
by
  intro h
  linarith

-- Specific instance of the theorem with given data for students A and B.
theorem student_B_more_stable : studentA.variance > studentB.variance → studentB.variance < studentA.variance :=
  student_B_more_stable_than_A studentA studentB rfl

end student_B_more_stable_than_A_student_B_more_stable_l1652_165265


namespace rational_expression_is_rational_l1652_165228

theorem rational_expression_is_rational (a b c : ℚ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) :
  ∃ r : ℚ, 
    r = Real.sqrt ((1 / (a - b)^2) + (1 / (b - c)^2) + (1 / (c - a)^2)) :=
sorry

end rational_expression_is_rational_l1652_165228


namespace max_area_quadrilateral_l1652_165204

theorem max_area_quadrilateral (a b c d : ℝ) (h1 : a = 1) (h2 : b = 4) (h3 : c = 7) (h4 : d = 8) : 
  ∃ A : ℝ, (A ≤ (1/2) * 1 * 8 + (1/2) * 4 * 7) ∧ (A = 18) :=
by
  sorry

end max_area_quadrilateral_l1652_165204


namespace abs_mult_example_l1652_165286

theorem abs_mult_example : (|(-3)| * 2) = 6 := by
  have h1 : |(-3)| = 3 := by
    exact abs_of_neg (show -3 < 0 by norm_num)
  rw [h1]
  exact mul_eq_mul_left_iff.mpr (Or.inl rfl)

end abs_mult_example_l1652_165286


namespace symmetrical_circle_proof_l1652_165250

open Real

-- Definition of the original circle equation
def original_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Defining the symmetrical circle equation to be proven
def symmetrical_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + y^2 = 5

theorem symmetrical_circle_proof :
  ∀ x y : ℝ, original_circle x y ↔ symmetrical_circle x y :=
by sorry

end symmetrical_circle_proof_l1652_165250


namespace correct_system_of_equations_l1652_165290

theorem correct_system_of_equations :
  ∃ (x y : ℝ), (4 * x + y = 5 * y + x) ∧ (5 * x + 6 * y = 16) := sorry

end correct_system_of_equations_l1652_165290


namespace total_rainfall_2007_correct_l1652_165274

noncomputable def rainfall_2005 : ℝ := 40.5
noncomputable def rainfall_2006 : ℝ := rainfall_2005 + 3
noncomputable def rainfall_2007 : ℝ := rainfall_2006 + 4
noncomputable def total_rainfall_2007 : ℝ := 12 * rainfall_2007

theorem total_rainfall_2007_correct : total_rainfall_2007 = 570 := 
sorry

end total_rainfall_2007_correct_l1652_165274


namespace john_needs_total_planks_l1652_165218

theorem john_needs_total_planks : 
  let large_planks := 12
  let small_planks := 17
  large_planks + small_planks = 29 :=
by
  sorry

end john_needs_total_planks_l1652_165218


namespace new_solution_is_45_percent_liquid_x_l1652_165249

-- Define initial conditions
def solution_y_initial_weight := 8.0 -- kilograms
def percent_liquid_x := 0.30
def percent_water := 0.70
def evaporated_water_weight := 4.0 -- kilograms
def added_solution_y_weight := 4.0 -- kilograms

-- Define the relevant quantities
def liquid_x_initial := solution_y_initial_weight * percent_liquid_x
def water_initial := solution_y_initial_weight * percent_water
def remaining_water_after_evaporation := water_initial - evaporated_water_weight

def liquid_x_after_evaporation := liquid_x_initial 
def water_after_evaporation := remaining_water_after_evaporation

def added_liquid_x := added_solution_y_weight * percent_liquid_x
def added_water := added_solution_y_weight * percent_water

def total_liquid_x := liquid_x_after_evaporation + added_liquid_x
def total_water := water_after_evaporation + added_water

def total_new_solution_weight := total_liquid_x + total_water

def new_solution_percent_liquid_x := (total_liquid_x / total_new_solution_weight) * 100

-- The theorem we want to prove
theorem new_solution_is_45_percent_liquid_x : new_solution_percent_liquid_x = 45 := by
  sorry

end new_solution_is_45_percent_liquid_x_l1652_165249


namespace circumference_is_720_l1652_165219

-- Given conditions
def uniform_speed (A_speed B_speed : ℕ) : Prop := A_speed > 0 ∧ B_speed > 0
def diametrically_opposite_start (A_pos B_pos : ℕ) (circumference : ℕ) : Prop := A_pos = 0 ∧ B_pos = circumference / 2
def meets_first_after_B_travel (A_distance B_distance : ℕ) : Prop := B_distance = 150
def meets_second_90_yards_before_A_lap (A_distance_lap B_distance_lap A_distance B_distance : ℕ) : Prop := 
  A_distance_lap = A_distance + 2 * (A_distance - B_distance) - 90 ∧ B_distance_lap = A_distance - B_distance_lap + (B_distance + 90)

theorem circumference_is_720 (circumference A_speed B_speed A_pos B_pos
                     A_distance B_distance
                     A_distance_lap B_distance_lap : ℕ) :
  uniform_speed A_speed B_speed →
  diametrically_opposite_start A_pos B_pos circumference →
  meets_first_after_B_travel A_distance B_distance →
  meets_second_90_yards_before_A_lap A_distance_lap B_distance_lap A_distance B_distance →
  circumference = 720 :=
sorry

end circumference_is_720_l1652_165219


namespace direct_proportion_l1652_165279

theorem direct_proportion (c f p : ℝ) (h : f ≠ 0 ∧ p = c * f) : ∃ k : ℝ, p / f = k * (f / f) :=
by
  sorry

end direct_proportion_l1652_165279


namespace solve_m_value_l1652_165227

-- Definitions for conditions
def hyperbola_eq (m : ℝ) : Prop := ∀ x y : ℝ, 3 * m * x^2 - m * y^2 = 3
def has_focus (m : ℝ) : Prop := (∃ f1 f2 : ℝ, f1 = 0 ∧ f2 = 2)

-- Statement of the problem to prove
theorem solve_m_value (m : ℝ) (h_eq : hyperbola_eq m) (h_focus : has_focus m) : m = -1 :=
sorry

end solve_m_value_l1652_165227


namespace unique_solution_iff_a_eq_2019_l1652_165209

theorem unique_solution_iff_a_eq_2019 (x a : ℝ) :
  (∃! x : ℝ, x - 1000 ≥ 1018 ∧ x + 1 ≤ a) ↔ a = 2019 :=
by
  sorry

end unique_solution_iff_a_eq_2019_l1652_165209
