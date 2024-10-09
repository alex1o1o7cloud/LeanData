import Mathlib

namespace negation_of_prop1_equiv_l1442_144283

-- Given proposition: if x > 1 then x > 0
def prop1 (x : ℝ) : Prop := x > 1 → x > 0

-- Negation of the given proposition: if x ≤ 1 then x ≤ 0
def neg_prop1 (x : ℝ) : Prop := x ≤ 1 → x ≤ 0

-- The theorem to prove that the negation of the proposition "If x > 1, then x > 0" 
-- is "If x ≤ 1, then x ≤ 0"
theorem negation_of_prop1_equiv (x : ℝ) : ¬(prop1 x) ↔ neg_prop1 x :=
by
  sorry

end negation_of_prop1_equiv_l1442_144283


namespace percentage_caught_customers_l1442_144232

noncomputable def total_sampling_percentage : ℝ := 0.25
noncomputable def caught_percentage : ℝ := 0.88

theorem percentage_caught_customers :
  total_sampling_percentage * caught_percentage = 0.22 :=
by
  sorry

end percentage_caught_customers_l1442_144232


namespace line_through_point_equal_intercepts_l1442_144251

theorem line_through_point_equal_intercepts (a b : ℝ) : 
  ((∃ (k : ℝ), k ≠ 0 ∧ (3 = 2 * k) ∧ b = k) ∨ ((a ≠ 0) ∧ (5/a = 1))) → 
  (a = 1 ∧ b = 1) ∨ (3 * a - 2 * b = 0) := 
by 
  sorry

end line_through_point_equal_intercepts_l1442_144251


namespace find_angle_C_l1442_144271

theorem find_angle_C (A B C : ℝ) (h1 : A = 88) (h2 : B - C = 20) (angle_sum : A + B + C = 180) : C = 36 :=
by
  sorry

end find_angle_C_l1442_144271


namespace max_sides_three_obtuse_l1442_144201

theorem max_sides_three_obtuse (n : ℕ) (convex : Prop) (obtuse_angles : ℕ) :
  (convex = true ∧ obtuse_angles = 3) → n ≤ 6 :=
by
  sorry

end max_sides_three_obtuse_l1442_144201


namespace Goat_guilty_l1442_144224

-- Condition definitions
def Goat_lied : Prop := sorry
def Beetle_testimony_true : Prop := sorry
def Mosquito_testimony_true : Prop := sorry
def Goat_accused_Beetle_or_Mosquito : Prop := sorry
def Beetle_accused_Goat_or_Mosquito : Prop := sorry
def Mosquito_accused_Beetle_or_Goat : Prop := sorry

-- Theorem: The Goat is guilty
theorem Goat_guilty (G_lied : Goat_lied) 
    (B_true : Beetle_testimony_true) 
    (M_true : Mosquito_testimony_true)
    (G_accuse : Goat_accused_Beetle_or_Mosquito)
    (B_accuse : Beetle_accused_Goat_or_Mosquito)
    (M_accuse : Mosquito_accused_Beetle_or_Goat) : 
  Prop :=
  sorry

end Goat_guilty_l1442_144224


namespace triangle_area_and_angle_l1442_144234

theorem triangle_area_and_angle (a b c A B C : ℝ) 
  (habc: A + B + C = Real.pi)
  (h1: (2*a + b)*Real.cos C + c*Real.cos B = 0)
  (h2: c = 2*Real.sqrt 6 / 3)
  (h3: Real.sin A * Real.cos B = (Real.sqrt 3 - 1)/4) :
  (C = 2*Real.pi / 3) ∧ (1/2 * b * c * Real.sin A = (6 - 2 * Real.sqrt 3)/9) :=
by
  sorry

end triangle_area_and_angle_l1442_144234


namespace inequality_of_triangle_tangents_l1442_144244

theorem inequality_of_triangle_tangents
  (a b c x y z : ℝ)
  (h1 : a = y + z)
  (h2 : b = x + z)
  (h3 : c = x + y)
  (h_order : a ≥ b ∧ b ≥ c)
  (h_tangents : z ≥ y ∧ y ≥ x) :
  (a * z + b * y + c * x ≥ (a^2 + b^2 + c^2) / 2) ∧
  ((a^2 + b^2 + c^2) / 2 ≥ a * x + b * y + c * z) :=
sorry

end inequality_of_triangle_tangents_l1442_144244


namespace hydrochloric_acid_moles_l1442_144272

theorem hydrochloric_acid_moles (amyl_alcohol moles_required : ℕ) 
  (h_ratio : amyl_alcohol = moles_required) 
  (h_balanced : amyl_alcohol = 3) :
  moles_required = 3 :=
by
  sorry

end hydrochloric_acid_moles_l1442_144272


namespace proof_problem_l1442_144208

-- Given definitions
def A := { y : ℝ | ∃ x : ℝ, y = x^2 + 1 }
def B := { p : ℝ × ℝ | ∃ x : ℝ, p.snd = x^2 + 1 }

-- Theorem to prove 1 ∉ B and 2 ∈ A
theorem proof_problem : 1 ∉ B ∧ 2 ∈ A :=
by
  sorry

end proof_problem_l1442_144208


namespace contradiction_problem_l1442_144226

theorem contradiction_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → False := 
by
  sorry

end contradiction_problem_l1442_144226


namespace intersection_M_N_l1442_144203

-- Define set M
def M : Set ℝ := {x : ℝ | ∃ t : ℝ, x = 2^(-t) }

-- Define set N
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = Real.sin x }

-- Theorem stating the intersection of M and N
theorem intersection_M_N :
  (M ∩ N) = {y : ℝ | 0 < y ∧ y ≤ 1} :=
by
  sorry

end intersection_M_N_l1442_144203


namespace fraction_value_l1442_144278

def op_at (a b : ℤ) : ℤ := a * b - b ^ 2
def op_sharp (a b : ℤ) : ℤ := a + b - a * b ^ 2

theorem fraction_value : (op_at 7 3) / (op_sharp 7 3) = -12 / 53 :=
by
  sorry

end fraction_value_l1442_144278


namespace ratio_pentagon_rectangle_l1442_144233

-- Definitions of conditions.
def pentagon_side_length (p : ℕ) : Prop := 5 * p = 30
def rectangle_width (w : ℕ) : Prop := 6 * w = 30

-- The theorem to prove.
theorem ratio_pentagon_rectangle (p w : ℕ) (h1 : pentagon_side_length p) (h2 : rectangle_width w) :
  p / w = 6 / 5 :=
by sorry

end ratio_pentagon_rectangle_l1442_144233


namespace find_y_l1442_144222

theorem find_y (y : ℝ) (h : (15 + 25 + y) / 3 = 23) : y = 29 :=
sorry

end find_y_l1442_144222


namespace smallest_n_l1442_144277

theorem smallest_n (o y v : ℕ) (h1 : 18 * o = 21 * y) (h2 : 21 * y = 10 * v) (h3 : 10 * v = 30 * n) : 
  n = 21 := by
  sorry

end smallest_n_l1442_144277


namespace S13_is_52_l1442_144205

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {n : ℕ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ n, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem S13_is_52 (h1 : is_arithmetic_sequence a)
                  (h2 : a 3 + a 7 + a 11 = 12)
                  (h3 : sum_of_first_n_terms S a) :
  S 13 = 52 :=
by sorry

end S13_is_52_l1442_144205


namespace triangle_area_from_altitudes_l1442_144281

noncomputable def triangleArea (altitude1 altitude2 altitude3 : ℝ) : ℝ :=
  sorry

theorem triangle_area_from_altitudes
  (h1 : altitude1 = 15)
  (h2 : altitude2 = 21)
  (h3 : altitude3 = 35) :
  triangleArea 15 21 35 = 245 * Real.sqrt 3 :=
sorry

end triangle_area_from_altitudes_l1442_144281


namespace no_integer_solution_l1442_144296

theorem no_integer_solution (x y : ℤ) : ¬(x^4 + y^2 = 4 * y + 4) :=
by
  sorry

end no_integer_solution_l1442_144296


namespace opposite_of_neg_two_thirds_l1442_144294

theorem opposite_of_neg_two_thirds : - (- (2 / 3) : ℚ) = (2 / 3 : ℚ) :=
by
  sorry

end opposite_of_neg_two_thirds_l1442_144294


namespace parabola_solution_l1442_144223

theorem parabola_solution (a b : ℝ) : 
  (∃ y : ℝ, y = 2^2 + 2 * a + b ∧ y = 20) ∧ 
  (∃ y : ℝ, y = (-2)^2 + (-2) * a + b ∧ y = 0) ∧ 
  b = (0^2 + 0 * a + b) → 
  a = 5 ∧ b = 6 := 
by {
  sorry
}

end parabola_solution_l1442_144223


namespace ratio_a_d_l1442_144287

theorem ratio_a_d (a b c d : ℕ) 
  (hab : a * 4 = b * 3) 
  (hbc : b * 9 = c * 7) 
  (hcd : c * 7 = d * 5) : 
  a * 12 = d :=
sorry

end ratio_a_d_l1442_144287


namespace returned_books_percentage_is_correct_l1442_144204

-- This function takes initial_books, end_books, and loaned_books and computes the percentage of books returned.
noncomputable def percent_books_returned (initial_books : ℕ) (end_books : ℕ) (loaned_books : ℕ) : ℚ :=
  let books_out_on_loan := initial_books - end_books
  let books_returned := loaned_books - books_out_on_loan
  (books_returned : ℚ) / (loaned_books : ℚ) * 100

-- The main theorem that states the percentage of books returned is 70%
theorem returned_books_percentage_is_correct :
  percent_books_returned 75 57 60 = 70 := by
  sorry

end returned_books_percentage_is_correct_l1442_144204


namespace jack_jill_same_speed_l1442_144265

-- Definitions for Jack and Jill's conditions
def jacks_speed (x : ℝ) : ℝ := x^2 - 13*x - 48
def jills_distance (x : ℝ) : ℝ := x^2 - 5*x - 84
def jills_time (x : ℝ) : ℝ := x + 8

-- Theorem stating the same walking speed given the conditions
theorem jack_jill_same_speed (x : ℝ) (h : jacks_speed x = jills_distance x / jills_time x) : 
  jacks_speed x = 6 :=
by
  sorry

end jack_jill_same_speed_l1442_144265


namespace quadrilateral_possible_with_2_2_2_l1442_144228

theorem quadrilateral_possible_with_2_2_2 :
  ∀ (s1 s2 s3 s4 : ℕ), (s1 = 2) → (s2 = 2) → (s3 = 2) → (s4 = 5) →
  s1 + s2 + s3 > s4 :=
by
  intros s1 s2 s3 s4 h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Proof omitted
  sorry

end quadrilateral_possible_with_2_2_2_l1442_144228


namespace jason_borrowed_amount_l1442_144220

theorem jason_borrowed_amount :
  let cycle := [1, 3, 5, 7, 9, 11]
  let total_chores := 48
  let chores_per_cycle := cycle.length
  let earnings_one_cycle := cycle.sum
  let complete_cycles := total_chores / chores_per_cycle
  let total_earnings := complete_cycles * earnings_one_cycle
  total_earnings = 288 :=
by
  sorry

end jason_borrowed_amount_l1442_144220


namespace alice_current_age_l1442_144282

def alice_age_twice_eve (a b : Nat) : Prop := a = 2 * b

def eve_age_after_10_years (a b : Nat) : Prop := a = b + 10

theorem alice_current_age (a b : Nat) (h1 : alice_age_twice_eve a b) (h2 : eve_age_after_10_years a b) : a = 20 := by
  sorry

end alice_current_age_l1442_144282


namespace indira_cricket_minutes_l1442_144268

def totalMinutesSeanPlayed (sean_minutes_per_day : ℕ) (days : ℕ) : ℕ :=
  sean_minutes_per_day * days

def totalMinutesIndiraPlayed (total_minutes_together : ℕ) (total_minutes_sean : ℕ) : ℕ :=
  total_minutes_together - total_minutes_sean

theorem indira_cricket_minutes :
  totalMinutesIndiraPlayed 1512 (totalMinutesSeanPlayed 50 14) = 812 :=
by
  sorry

end indira_cricket_minutes_l1442_144268


namespace coin_selection_probability_l1442_144209

noncomputable def probability_at_least_50_cents : ℚ := 
  let total_ways := Nat.choose 12 6 -- total ways to choose 6 coins out of 12
  let case1 := 1 -- 6 dimes
  let case2 := (Nat.choose 6 5) * (Nat.choose 4 1) -- 5 dimes and 1 nickel
  let case3 := (Nat.choose 6 4) * (Nat.choose 4 2) -- 4 dimes and 2 nickels
  let successful_ways := case1 + case2 + case3 -- total successful outcomes
  successful_ways / total_ways

theorem coin_selection_probability : 
  probability_at_least_50_cents = 127 / 924 := by 
  sorry

end coin_selection_probability_l1442_144209


namespace fraction_meaningful_range_l1442_144236

variable (x : ℝ)

theorem fraction_meaningful_range (h : x - 2 ≠ 0) : x ≠ 2 :=
by
  sorry

end fraction_meaningful_range_l1442_144236


namespace find_balloons_given_to_Fred_l1442_144256

variable (x : ℝ)
variable (Sam_initial_balance : ℝ := 46.0)
variable (Dan_balance : ℝ := 16.0)
variable (total_balance : ℝ := 52.0)

theorem find_balloons_given_to_Fred
  (h : Sam_initial_balance - x + Dan_balance = total_balance) :
  x = 10.0 :=
by
  sorry

end find_balloons_given_to_Fred_l1442_144256


namespace initial_cost_of_smartphone_l1442_144246

theorem initial_cost_of_smartphone 
(C : ℝ) 
(h : 0.85 * C = 255) : 
C = 300 := 
sorry

end initial_cost_of_smartphone_l1442_144246


namespace glue_needed_l1442_144285

-- Definitions based on conditions
def num_friends : ℕ := 7
def clippings_per_friend : ℕ := 3
def drops_per_clipping : ℕ := 6

-- Calculation
def total_clippings : ℕ := num_friends * clippings_per_friend
def total_drops_of_glue : ℕ := drops_per_clipping * total_clippings

-- Theorem statement
theorem glue_needed : total_drops_of_glue = 126 := by
  sorry

end glue_needed_l1442_144285


namespace initial_hair_length_l1442_144213

-- Definitions based on the conditions
def hair_cut_off : ℕ := 13
def current_hair_length : ℕ := 1

-- The problem statement to be proved
theorem initial_hair_length : (current_hair_length + hair_cut_off = 14) :=
by
  sorry

end initial_hair_length_l1442_144213


namespace correct_option_is_C_l1442_144280

-- Define the polynomial expressions and their expected values as functions
def optionA (x : ℝ) : Prop := (x + 2) * (x - 5) = x^2 - 2 * x - 3
def optionB (x : ℝ) : Prop := (x + 3) * (x - 1 / 3) = x^2 + x - 1
def optionC (x : ℝ) : Prop := (x - 2 / 3) * (x + 1 / 2) = x^2 - 1 / 6 * x - 1 / 3
def optionD (x : ℝ) : Prop := (x - 2) * (-x - 2) = x^2 - 4

-- Problem Statement: Verify that the polynomial multiplication in Option C is correct
theorem correct_option_is_C (x : ℝ) : optionC x :=
by
  -- Statement indicating the proof goes here
  sorry

end correct_option_is_C_l1442_144280


namespace moles_C2H6_for_HCl_l1442_144245

theorem moles_C2H6_for_HCl 
  (form_HCl : ℕ)
  (moles_Cl2 : ℕ)
  (reaction : ℕ) : 
  (6 * (reaction * moles_Cl2)) = form_HCl * (6 * reaction) :=
by
  -- The necessary proof steps will go here
  sorry

end moles_C2H6_for_HCl_l1442_144245


namespace probability_three_non_red_purple_balls_l1442_144273

def total_balls : ℕ := 150
def prob_white : ℝ := 0.15
def prob_green : ℝ := 0.20
def prob_yellow : ℝ := 0.30
def prob_red : ℝ := 0.30
def prob_purple : ℝ := 0.05
def prob_not_red_purple : ℝ := 1 - (prob_red + prob_purple)

theorem probability_three_non_red_purple_balls :
  (prob_not_red_purple * prob_not_red_purple * prob_not_red_purple) = 0.274625 :=
by
  sorry

end probability_three_non_red_purple_balls_l1442_144273


namespace a_sufficient_not_necessary_for_a_squared_eq_b_squared_l1442_144227

theorem a_sufficient_not_necessary_for_a_squared_eq_b_squared
  (a b : ℝ) :
  (a = b) → (a^2 = b^2) ∧ ¬ ((a^2 = b^2) → (a = b)) :=
  sorry

end a_sufficient_not_necessary_for_a_squared_eq_b_squared_l1442_144227


namespace solve_equation_l1442_144211

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ 0) :
  (x / (x - 2) - 3 / x = 1) → x = 6 :=
by
  sorry

end solve_equation_l1442_144211


namespace decreasing_hyperbola_l1442_144215

theorem decreasing_hyperbola (m : ℝ) (x : ℝ) (hx : x > 0) (y : ℝ) (h_eq : y = (1 - m) / x) :
  (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > x₁ → (1 - m) / x₂ < (1 - m) / x₁) ↔ m < 1 :=
by
  sorry

end decreasing_hyperbola_l1442_144215


namespace fraction_comparison_l1442_144225

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end fraction_comparison_l1442_144225


namespace find_k_l1442_144269

theorem find_k (a b : ℤ × ℤ) (k : ℤ) 
  (h₁ : a = (2, 1)) 
  (h₂ : a.1 + b.1 = 1 ∧ a.2 + b.2 = k)
  (h₃ : a.1 * b.1 + a.2 * b.2 = 0) : k = 3 :=
sorry

end find_k_l1442_144269


namespace number_of_attendees_choosing_water_l1442_144263

variables {total_attendees : ℕ} (juice_percent water_percent : ℚ)

-- Conditions
def attendees_juice (total_attendees : ℕ) : ℚ := 0.7 * total_attendees
def attendees_water (total_attendees : ℕ) : ℚ := 0.3 * total_attendees
def attendees_juice_given := (attendees_juice total_attendees) = 140

-- Theorem statement
theorem number_of_attendees_choosing_water 
  (h1 : juice_percent = 0.7) 
  (h2 : water_percent = 0.3) 
  (h3 : attendees_juice total_attendees = 140) : 
  attendees_water total_attendees = 60 :=
sorry

end number_of_attendees_choosing_water_l1442_144263


namespace count_positive_integers_x_satisfying_inequality_l1442_144242

theorem count_positive_integers_x_satisfying_inequality :
  ∃ n : ℕ, n = 6 ∧ (∀ x : ℕ, (144 ≤ x^2 ∧ x^2 ≤ 289) → (x = 12 ∨ x = 13 ∨ x = 14 ∨ x = 15 ∨ x = 16 ∨ x = 17)) :=
sorry

end count_positive_integers_x_satisfying_inequality_l1442_144242


namespace deposit_is_500_l1442_144216

-- Definitions corresponding to the conditions
def janet_saved : ℕ := 2225
def rent_per_month : ℕ := 1250
def advance_months : ℕ := 2
def extra_needed : ℕ := 775

-- Definition that encapsulates the deposit calculation
def deposit_required (saved rent_monthly months_advance extra : ℕ) : ℕ :=
  let total_rent := months_advance * rent_monthly
  let total_needed := saved + extra
  total_needed - total_rent

-- Theorem statement for the proof problem
theorem deposit_is_500 : deposit_required janet_saved rent_per_month advance_months extra_needed = 500 :=
by
  sorry

end deposit_is_500_l1442_144216


namespace maximize_triangle_areas_l1442_144217

theorem maximize_triangle_areas (L W : ℝ) (h1 : 2 * L + 2 * W = 80) (h2 : L ≤ 25) : W = 15 :=
by 
  sorry

end maximize_triangle_areas_l1442_144217


namespace power_simplification_l1442_144253

theorem power_simplification :
  (1 / ((-5) ^ 4) ^ 2) * (-5) ^ 9 = -5 :=
by 
  sorry

end power_simplification_l1442_144253


namespace determine_ab_l1442_144238

noncomputable def f (a b : ℕ) (x : ℝ) : ℝ := x^2 + 2 * a * x + b * 2^x

theorem determine_ab (a b : ℕ) (h : ∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) :
  (a, b) = (0, 0) ∨ (a, b) = (1, 0) :=
by
  sorry

end determine_ab_l1442_144238


namespace largest_triangle_perimeter_l1442_144221

theorem largest_triangle_perimeter (x : ℤ) (hx1 : 7 + 11 > x) (hx2 : 7 + x > 11) (hx3 : 11 + x > 7) (hx4 : 5 ≤ x) (hx5 : x < 18) : 
  7 + 11 + x = 35 :=
sorry

end largest_triangle_perimeter_l1442_144221


namespace possible_values_l1442_144259

def expression (m n : ℕ) : ℤ :=
  (m^2 + m * n + n^2) / (m * n - 1)

theorem possible_values (m n : ℕ) (h : m * n ≠ 1) : 
  ∃ (N : ℤ), N = expression m n → N = 0 ∨ N = 4 ∨ N = 7 :=
by
  sorry

end possible_values_l1442_144259


namespace number_of_customers_per_month_l1442_144262

-- Define the constants and conditions
def price_lettuce_per_head : ℝ := 1
def price_tomato_per_piece : ℝ := 0.5
def num_lettuce_per_customer : ℕ := 2
def num_tomato_per_customer : ℕ := 4
def monthly_sales : ℝ := 2000

-- Calculate the cost per customer
def cost_per_customer : ℝ := 
  (num_lettuce_per_customer * price_lettuce_per_head) + 
  (num_tomato_per_customer * price_tomato_per_piece)

-- Prove the number of customers per month
theorem number_of_customers_per_month : monthly_sales / cost_per_customer = 500 :=
  by
    -- Here, we would write the proof steps
    sorry

end number_of_customers_per_month_l1442_144262


namespace problem_statement_l1442_144248

theorem problem_statement (n : ℕ) (h1 : 0 < n) (h2 : ∃ k : ℤ, (1/2 + 1/3 + 1/11 + 1/n : ℚ) = k) : ¬ (n > 66) := 
sorry

end problem_statement_l1442_144248


namespace find_particular_number_l1442_144255

theorem find_particular_number (x : ℤ) (h : x - 7 = 2) : x = 9 :=
by {
  -- The proof will be written here.
  sorry
}

end find_particular_number_l1442_144255


namespace marker_cost_l1442_144235

theorem marker_cost (s n c : ℕ) (h_majority : s > 20) (h_markers : n > 1) (h_cost : c > n) (h_total_cost : s * n * c = 3388) : c = 11 :=
by {
  sorry
}

end marker_cost_l1442_144235


namespace num_two_digit_multiples_5_and_7_l1442_144247

/-- 
    Theorem: There are exactly 2 positive two-digit integers that are multiples of both 5 and 7.
-/
theorem num_two_digit_multiples_5_and_7 : 
  ∃ (count : ℕ), count = 2 ∧ ∀ (n : ℕ), (10 ≤ n ∧ n ≤ 99) → 
    (n % 5 = 0 ∧ n % 7 = 0) ↔ (n = 35 ∨ n = 70) := 
by
  sorry

end num_two_digit_multiples_5_and_7_l1442_144247


namespace find_x_given_conditions_l1442_144230

variables {x y z : ℝ}

theorem find_x_given_conditions (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = (576 : ℝ)^(1/7) := 
sorry

end find_x_given_conditions_l1442_144230


namespace difference_of_numbers_l1442_144286

theorem difference_of_numbers (x y : ℕ) (h1 : x + y = 64) (h2 : y = 26) : x - y = 12 :=
sorry

end difference_of_numbers_l1442_144286


namespace atomic_number_cannot_be_x_plus_4_l1442_144206

-- Definitions for atomic numbers and elements in the same main group
def in_same_main_group (A B : Type) (atomic_num_A atomic_num_B : ℕ) : Prop :=
  atomic_num_B ≠ atomic_num_A + 4

-- Noncomputable definition is likely needed as the problem involves non-algorithmic aspects.
noncomputable def periodic_table_condition (A B : Type) (x : ℕ) : Prop :=
  in_same_main_group A B x (x + 4)

-- Main theorem stating the mathematical proof problem
theorem atomic_number_cannot_be_x_plus_4
  (A B : Type)
  (x : ℕ)
  (h : periodic_table_condition A B x) : false :=
  by
    sorry

end atomic_number_cannot_be_x_plus_4_l1442_144206


namespace eval_7_star_3_l1442_144290

def operation (a b : ℕ) : ℕ := (4 * a + 5 * b - a * b)

theorem eval_7_star_3 : operation 7 3 = 22 :=
  by {
    -- substitution and calculation steps
    sorry
  }

end eval_7_star_3_l1442_144290


namespace simplify_expression_l1442_144237

theorem simplify_expression (x : ℝ) : 
  ( ( (x^(16/8))^(1/4) )^3 * ( (x^(16/4))^(1/8) )^5 ) = x^4 := 
by 
  sorry

end simplify_expression_l1442_144237


namespace calc1_calc2_calc3_calc4_l1442_144218

theorem calc1 : 327 + 46 - 135 = 238 := by sorry
theorem calc2 : 1000 - 582 - 128 = 290 := by sorry
theorem calc3 : (124 - 62) * 6 = 372 := by sorry
theorem calc4 : 500 - 400 / 5 = 420 := by sorry

end calc1_calc2_calc3_calc4_l1442_144218


namespace dorothy_age_relation_l1442_144254

theorem dorothy_age_relation (D S : ℕ) (h1: S = 5) (h2: D + 5 = 2 * (S + 5)) : D = 3 * S :=
by
  -- implement the proof here
  sorry

end dorothy_age_relation_l1442_144254


namespace greatest_large_chips_l1442_144291

theorem greatest_large_chips :
  ∃ (l : ℕ), (∃ (s : ℕ), ∃ (p : ℕ), s + l = 70 ∧ s = l + p ∧ Nat.Prime p) ∧ 
  (∀ (l' : ℕ), (∃ (s' : ℕ), ∃ (p' : ℕ), s' + l' = 70 ∧ s' = l' + p' ∧ Nat.Prime p') → l' ≤ 34) :=
sorry

end greatest_large_chips_l1442_144291


namespace probability_of_multiple_of_3_is_1_5_l1442_144257

-- Definition of the problem conditions
def digits : List ℕ := [1, 2, 3, 4, 5]

-- Function to calculate the probability
noncomputable def probability_of_multiple_of_3 : ℚ := 
  let total_permutations := (Nat.factorial 5) / (Nat.factorial (5 - 4))  -- i.e., 120
  let valid_permutations := Nat.factorial 4  -- i.e., 24, for the valid combination
  valid_permutations / total_permutations 

-- Statement to be proved
theorem probability_of_multiple_of_3_is_1_5 :
  probability_of_multiple_of_3 = 1 / 5 := 
by
  -- Skeleton for the proof
  sorry

end probability_of_multiple_of_3_is_1_5_l1442_144257


namespace number_of_observations_is_14_l1442_144202

theorem number_of_observations_is_14
  (mean_original : ℚ) (mean_new : ℚ) (original_sum : ℚ) 
  (corrected_sum : ℚ) (n : ℚ)
  (h1 : mean_original = 36)
  (h2 : mean_new = 36.5)
  (h3 : corrected_sum = original_sum + 7)
  (h4 : mean_new = corrected_sum / n)
  (h5 : original_sum = mean_original * n) :
  n = 14 :=
by
  -- Here goes the proof
  sorry

end number_of_observations_is_14_l1442_144202


namespace carrots_thrown_out_l1442_144299

def initial_carrots := 19
def additional_carrots := 46
def total_current_carrots := 61

def total_picked := initial_carrots + additional_carrots

theorem carrots_thrown_out : total_picked - total_current_carrots = 4 := by
  sorry

end carrots_thrown_out_l1442_144299


namespace least_grapes_in_heap_l1442_144212

theorem least_grapes_in_heap :
  ∃ n : ℕ, (n % 19 = 1) ∧ (n % 23 = 1) ∧ (n % 29 = 1) ∧ n = 12209 :=
by
  sorry

end least_grapes_in_heap_l1442_144212


namespace rectangular_prism_inequalities_l1442_144214

variable {a b c : ℝ}

noncomputable def p (a b c : ℝ) := 4 * (a + b + c)
noncomputable def S (a b c : ℝ) := 2 * (a * b + b * c + c * a)
noncomputable def d (a b c : ℝ) := Real.sqrt (a^2 + b^2 + c^2)

theorem rectangular_prism_inequalities (h : a > b) (h1 : b > c) :
  a > (1 / 3) * (p a b c / 4 + Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) ∧
  c < (1 / 3) * (p a b c / 4 - Real.sqrt (d a b c ^ 2 - (1 / 2) * S a b c)) :=
by
  sorry

end rectangular_prism_inequalities_l1442_144214


namespace complex_calculation_l1442_144258

def complex_add (a b : ℂ) : ℂ := a + b
def complex_mul (a b : ℂ) : ℂ := a * b

theorem complex_calculation :
  let z1 := (⟨2, -3⟩ : ℂ)
  let z2 := (⟨4, 6⟩ : ℂ)
  let z3 := (⟨-1, 2⟩ : ℂ)
  complex_mul (complex_add z1 z2) z3 = (⟨-12, 9⟩ : ℂ) :=
by 
  sorry

end complex_calculation_l1442_144258


namespace smallest_value_x_squared_plus_six_x_plus_nine_l1442_144200

theorem smallest_value_x_squared_plus_six_x_plus_nine : ∀ x : ℝ, x^2 + 6 * x + 9 ≥ 0 :=
by sorry

end smallest_value_x_squared_plus_six_x_plus_nine_l1442_144200


namespace find_e_value_l1442_144274

theorem find_e_value : (14 ^ 2) * (5 ^ 3) * 568 = 13916000 := by
  sorry

end find_e_value_l1442_144274


namespace intersection_of_A_and_B_l1442_144267

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l1442_144267


namespace opposite_of_neg_nine_is_nine_l1442_144284

-- Define the predicate for the opposite of a number
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- State the theorem that needs to be proved
theorem opposite_of_neg_nine_is_nine : ∃ x, is_opposite (-9) x ∧ x = 9 := 
by {
  -- Use "sorry" to indicate the proof is not provided
  sorry
}

end opposite_of_neg_nine_is_nine_l1442_144284


namespace exist_six_subsets_of_six_elements_l1442_144249

theorem exist_six_subsets_of_six_elements (n m : ℕ) (X : Finset ℕ) (A : Fin m → Finset ℕ) :
    n > 6 →
    X.card = n →
    (∀ i, (A i).card = 5 ∧ (A i ⊆ X)) →
    m > (n * (n-1) * (n-2) * (n-3) * (4*n-15)) / 600 →
    ∃ i1 i2 i3 i4 i5 i6 : Fin m,
      i1 < i2 ∧ i2 < i3 ∧ i3 < i4 ∧ i4 < i5 ∧ i5 < i6 ∧
      (A i1 ∪ A i2 ∪ A i3 ∪ A i4 ∪ A i5 ∪ A i6).card = 6 := 
sorry

end exist_six_subsets_of_six_elements_l1442_144249


namespace acute_angle_of_parallel_vectors_l1442_144219
open Real

theorem acute_angle_of_parallel_vectors (α : ℝ) (h₁ : abs (α * π / 180) < π / 2) :
  let a := (3 / 2, sin (α * π / 180))
  let b := (sin (α * π / 180), 1 / 6) 
  a.1 * b.2 = a.2 * b.1 → α = 30 :=
by
  sorry

end acute_angle_of_parallel_vectors_l1442_144219


namespace B_monthly_income_is_correct_l1442_144243

variable (A_m B_m C_m : ℝ)
variable (A_annual C_m_value : ℝ)
variable (ratio_A_to_B : ℝ)

-- Given conditions
def conditions :=
  A_annual = 537600 ∧
  C_m_value = 16000 ∧
  ratio_A_to_B = 5 / 2 ∧
  A_m = A_annual / 12 ∧
  B_m = (2 / 5) * A_m ∧
  B_m = 1.12 * C_m ∧
  C_m = C_m_value

-- Prove that B's monthly income is Rs. 17920
theorem B_monthly_income_is_correct (h : conditions A_m B_m C_m A_annual C_m_value ratio_A_to_B) : 
  B_m = 17920 :=
by 
  sorry

end B_monthly_income_is_correct_l1442_144243


namespace minValue_is_9_minValue_achieves_9_l1442_144276

noncomputable def minValue (x y : ℝ) : ℝ :=
  (x^2 + 1/(y^2)) * (1/(x^2) + 4 * y^2)

theorem minValue_is_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) : minValue x y ≥ 9 :=
  sorry

theorem minValue_achieves_9 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 1/2) : minValue x y = 9 :=
  sorry

end minValue_is_9_minValue_achieves_9_l1442_144276


namespace total_weight_proof_l1442_144298

-- Define molar masses
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008

-- Define moles of elements in each compound
def moles_C4H10 : ℕ := 8
def moles_C3H8 : ℕ := 5
def moles_CH4 : ℕ := 3

-- Define the molar masses of each compound
def molar_mass_C4H10 : ℝ := 4 * molar_mass_C + 10 * molar_mass_H
def molar_mass_C3H8 : ℝ := 3 * molar_mass_C + 8 * molar_mass_H
def molar_mass_CH4 : ℝ := 1 * molar_mass_C + 4 * molar_mass_H

-- Define the total weight
def total_weight : ℝ :=
  moles_C4H10 * molar_mass_C4H10 +
  moles_C3H8 * molar_mass_C3H8 +
  moles_CH4 * molar_mass_CH4

theorem total_weight_proof :
  total_weight = 733.556 := by
  sorry

end total_weight_proof_l1442_144298


namespace solve_fractional_equation_l1442_144239

theorem solve_fractional_equation (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) → (x = 4) :=
by 
  intro h
  -- proof steps would go here
  sorry

end solve_fractional_equation_l1442_144239


namespace div_operation_example_l1442_144297

theorem div_operation_example : ((180 / 6) / 3) = 10 := by
  sorry

end div_operation_example_l1442_144297


namespace maximizing_sum_of_arithmetic_sequence_l1442_144270

theorem maximizing_sum_of_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_decreasing : ∀ n, a n > a (n + 1))
  (h_sum : S 5 = S 10) :
  (S 7 >= S n ∧ S 8 >= S n) := sorry

end maximizing_sum_of_arithmetic_sequence_l1442_144270


namespace billy_music_book_songs_l1442_144295

theorem billy_music_book_songs (can_play : ℕ) (needs_to_learn : ℕ) (total_songs : ℕ) 
  (h1 : can_play = 24) (h2 : needs_to_learn = 28) : 
  total_songs = can_play + needs_to_learn ↔ total_songs = 52 :=
by
  sorry

end billy_music_book_songs_l1442_144295


namespace bicycle_speed_B_l1442_144293

theorem bicycle_speed_B (v_A v_B : ℝ) (d : ℝ) (h1 : d = 12) (h2 : v_A = 1.2 * v_B) (h3 : d / v_B - d / v_A = 1 / 6) : v_B = 12 :=
by
  sorry

end bicycle_speed_B_l1442_144293


namespace total_baseball_cards_l1442_144275

-- Define the number of baseball cards each person has
def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

-- The total number of baseball cards they have
theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards + john_cards + sarah_cards + emma_cards = 100 :=
by
  sorry

end total_baseball_cards_l1442_144275


namespace find_present_ratio_l1442_144231

noncomputable def present_ratio_of_teachers_to_students : Prop :=
  ∃ (S T S' T' : ℕ),
    (T = 3) ∧
    (S = 50 * T) ∧
    (S' = S + 50) ∧
    (T' = T + 5) ∧
    (S' / T' = 25 / 1) ∧ 
    (T / S = 1 / 50)

theorem find_present_ratio : present_ratio_of_teachers_to_students :=
by
  sorry

end find_present_ratio_l1442_144231


namespace complement_set_example_l1442_144240

open Set

variable (U M : Set ℕ)

def complement (U M : Set ℕ) := U \ M

theorem complement_set_example :
  (U = {1, 2, 3, 4, 5, 6}) → 
  (M = {1, 3, 5}) → 
  (complement U M = {2, 4, 6}) := by
  intros hU hM
  rw [complement, hU, hM]
  sorry

end complement_set_example_l1442_144240


namespace find_f2_l1442_144229

-- A condition of the problem is the specific form of the function
def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

-- Given condition
theorem find_f2 (a b : ℝ) (h : f (-2) a b = 3) : f 2 a b = -19 :=
by
  sorry

end find_f2_l1442_144229


namespace journey_time_calculation_l1442_144241

theorem journey_time_calculation (dist totalDistance : ℝ) (rate1 rate2 : ℝ)
  (firstHalfDistance secondHalfDistance : ℝ) (time1 time2 totalTime : ℝ) :
  totalDistance = 224 ∧ rate1 = 21 ∧ rate2 = 24 ∧
  firstHalfDistance = totalDistance / 2 ∧ secondHalfDistance = totalDistance / 2 ∧
  time1 = firstHalfDistance / rate1 ∧ time2 = secondHalfDistance / rate2 ∧
  totalTime = time1 + time2 →
  totalTime = 10 :=
sorry

end journey_time_calculation_l1442_144241


namespace number_of_possible_values_of_s_l1442_144266

noncomputable def s := {s : ℚ | ∃ w x y z : ℕ, s = w / 1000 + x / 10000 + y / 100000 + z / 1000000 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10}

theorem number_of_possible_values_of_s (s_approx : s → ℚ → Prop) (h_s_approx : ∀ s, s_approx s (3 / 11)) :
  ∃ n : ℕ, n = 266 :=
by
  sorry

end number_of_possible_values_of_s_l1442_144266


namespace polynomial_simplification_l1442_144261

variable (x : ℝ)

theorem polynomial_simplification :
  (3*x^3 + 4*x^2 + 12)*(x + 1) - (x + 1)*(2*x^3 + 6*x^2 - 42) + (6*x^2 - 28)*(x + 1)*(x - 2) = 
  7*x^4 - 7*x^3 - 42*x^2 + 82*x + 110 :=
by
  sorry

end polynomial_simplification_l1442_144261


namespace valid_integer_pairs_l1442_144252

theorem valid_integer_pairs :
  ∀ a b : ℕ, 1 ≤ a → 1 ≤ b → a ^ (b ^ 2) = b ^ a → (a, b) = (1, 1) ∨ (a, b) = (16, 2) ∨ (a, b) = (27, 3) :=
by
  sorry

end valid_integer_pairs_l1442_144252


namespace mass_of_man_l1442_144207

theorem mass_of_man (L B : ℝ) (h : ℝ) (ρ : ℝ) (V : ℝ) : L = 8 ∧ B = 3 ∧ h = 0.01 ∧ ρ = 1 ∧ V = L * 100 * B * 100 * h → V / 1000 = 240 :=
by
  sorry

end mass_of_man_l1442_144207


namespace a_n_divisible_by_11_l1442_144250

-- Define the sequence
def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ 
  a 2 = 3 ∧ 
  ∀ n, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n

-- Main statement
theorem a_n_divisible_by_11 (a : ℕ → ℤ) (h : seq a) :
  ∀ n, ∃ k : ℕ, a n % 11 = 0 ↔ n = 4 + 11 * k :=
sorry

end a_n_divisible_by_11_l1442_144250


namespace sequence_problem_proof_l1442_144264

-- Define the sequence terms, using given conditions
def a_1 : ℕ := 1
def a_2 : ℕ := 2
def a_3 : ℕ := a_1 + a_2
def a_4 : ℕ := a_2 + a_3
def x : ℕ := a_3 + a_4

-- Prove that x = 8
theorem sequence_problem_proof : x = 8 := 
by
  sorry

end sequence_problem_proof_l1442_144264


namespace ratio_of_areas_of_circles_l1442_144279

-- Given conditions
variables (R_C R_D : ℝ) -- Radii of circles C and D respectively
variables (L : ℝ) -- Common length of the arcs

-- Equivalent arc condition
def arc_length_condition : Prop :=
  (60 / 360) * (2 * Real.pi * R_C) = L ∧ (40 / 360) * (2 * Real.pi * R_D) = L

-- Goal: ratio of areas
def area_ratio : Prop :=
  (Real.pi * R_C^2) / (Real.pi * R_D^2) = (9 / 4)

-- Problem statement
theorem ratio_of_areas_of_circles (R_C R_D L : ℝ) (hc : arc_length_condition R_C R_D L) :
  area_ratio R_C R_D :=
by
  sorry

end ratio_of_areas_of_circles_l1442_144279


namespace units_digit_7_pow_3_pow_4_l1442_144260

theorem units_digit_7_pow_3_pow_4 :
  (7 ^ (3 ^ 4)) % 10 = 7 :=
by
  -- Here's the proof placeholder
  sorry

end units_digit_7_pow_3_pow_4_l1442_144260


namespace consumption_reduction_l1442_144210

variable (P C : ℝ)

theorem consumption_reduction (h : P > 0 ∧ C > 0) : 
  (1.25 * P * (0.8 * C) = P * C) :=
by
  -- Conditions: original price P, original consumption C
  -- New price 1.25 * P, New consumption 0.8 * C
  sorry

end consumption_reduction_l1442_144210


namespace min_value_of_a_and_b_l1442_144288

theorem min_value_of_a_and_b (a b : ℝ) (h : a ^ 2 + 2 * b ^ 2 = 6) : 
  ∃ (m : ℝ), (∀ (x y : ℝ), x ^ 2 + 2 * y ^ 2 = 6 → x + y ≥ m) ∧ (a + b = m) :=
sorry

end min_value_of_a_and_b_l1442_144288


namespace heaviest_tv_l1442_144292

theorem heaviest_tv :
  let area (width height : ℝ) := width * height
  let weight (area : ℝ) := area * 4
  let weight_in_pounds (weight : ℝ) := weight / 16
  let bill_area := area 48 100
  let bob_area := area 70 60
  let steve_area := area 84 92
  let bill_weight := weight bill_area
  let bob_weight := weight bob_area
  let steve_weight := weight steve_area
  let bill_weight_pounds := weight_in_pounds (weight bill_area)
  let bob_weight_pounds := weight_in_pounds (weight bob_area)
  let steve_weight_pounds := weight_in_pounds (weight steve_area)
  bob_weight_pounds + bill_weight_pounds < steve_weight_pounds
  ∧ abs ((steve_weight_pounds) - (bill_weight_pounds + bob_weight_pounds)) = 318 :=
by
  sorry

end heaviest_tv_l1442_144292


namespace family_trip_eggs_l1442_144289

theorem family_trip_eggs (adults girls boys : ℕ)
  (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) (extra_eggs_for_boy : ℕ) :
  adults = 3 →
  eggs_per_adult = 3 →
  girls = 7 →
  eggs_per_girl = 1 →
  boys = 10 →
  extra_eggs_for_boy = 1 →
  (adults * eggs_per_adult + girls * eggs_per_girl + boys * (eggs_per_girl + extra_eggs_for_boy)) = 36 :=
by
  intros
  sorry

end family_trip_eggs_l1442_144289
