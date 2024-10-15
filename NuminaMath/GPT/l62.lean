import Mathlib

namespace NUMINAMATH_GPT_contractor_pays_male_worker_rs_35_l62_6282

theorem contractor_pays_male_worker_rs_35
  (num_male_workers : ℕ)
  (num_female_workers : ℕ)
  (num_child_workers : ℕ)
  (female_worker_wage : ℕ)
  (child_worker_wage : ℕ)
  (average_wage_per_day : ℕ)
  (total_workers : ℕ := num_male_workers + num_female_workers + num_child_workers)
  (total_wage : ℕ := average_wage_per_day * total_workers)
  (total_female_wage : ℕ := num_female_workers * female_worker_wage)
  (total_child_wage : ℕ := num_child_workers * child_worker_wage)
  (total_male_wage : ℕ := total_wage - total_female_wage - total_child_wage) :
  num_male_workers = 20 →
  num_female_workers = 15 →
  num_child_workers = 5 →
  female_worker_wage = 20 →
  child_worker_wage = 8 →
  average_wage_per_day = 26 →
  total_male_wage / num_male_workers = 35 :=
by
  intros h20 h15 h5 h20w h8w h26
  sorry

end NUMINAMATH_GPT_contractor_pays_male_worker_rs_35_l62_6282


namespace NUMINAMATH_GPT_sum_of_numbers_l62_6297

def a : ℝ := 217
def b : ℝ := 2.017
def c : ℝ := 0.217
def d : ℝ := 2.0017

theorem sum_of_numbers :
  a + b + c + d = 221.2357 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l62_6297


namespace NUMINAMATH_GPT_solve_equation_l62_6212

theorem solve_equation (a b : ℤ) (ha : a ≥ 0) (hb : b ≥ 0) :
  a^2 = b * (b + 7) ↔ (a = 0 ∧ b = 0) ∨ (a = 12 ∧ b = 9) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l62_6212


namespace NUMINAMATH_GPT_proof_true_proposition_l62_6299

open Classical

def P : Prop := ∀ x : ℝ, x^2 ≥ 0
def Q : Prop := ∃ x : ℚ, x^2 ≠ 3
def true_proposition (p q : Prop) := p ∨ ¬q

theorem proof_true_proposition : P ∧ ¬Q → true_proposition P Q :=
by
  intro h
  sorry

end NUMINAMATH_GPT_proof_true_proposition_l62_6299


namespace NUMINAMATH_GPT_domain_sqrt_l62_6218

noncomputable def domain_of_function := {x : ℝ | x ≥ 0 ∧ x - 1 ≥ 0}

theorem domain_sqrt : domain_of_function = {x : ℝ | 1 ≤ x} := by {
  sorry
}

end NUMINAMATH_GPT_domain_sqrt_l62_6218


namespace NUMINAMATH_GPT_domain_of_log_sqrt_l62_6223

noncomputable def domain_of_function := {x : ℝ | (2 * x - 1 > 0) ∧ (2 * x - 1 ≠ 1) ∧ (3 * x - 2 > 0)}

theorem domain_of_log_sqrt : domain_of_function = {x : ℝ | (2 / 3 < x ∧ x < 1) ∨ (1 < x)} :=
by sorry

end NUMINAMATH_GPT_domain_of_log_sqrt_l62_6223


namespace NUMINAMATH_GPT_common_property_of_rectangles_rhombuses_and_squares_l62_6221

-- Definitions of shapes and properties

-- Assume properties P1 = "Diagonals are equal", P2 = "Diagonals bisect each other", 
-- P3 = "Diagonals are perpendicular to each other", and P4 = "Diagonals bisect each other and are equal"

def is_rectangle (R : Type) : Prop := sorry
def is_rhombus (R : Type) : Prop := sorry
def is_square (R : Type) : Prop := sorry

def diagonals_bisect_each_other (R : Type) : Prop := sorry

-- Theorem stating the common property
theorem common_property_of_rectangles_rhombuses_and_squares 
  (R : Type)
  (H_rect : is_rectangle R)
  (H_rhomb : is_rhombus R)
  (H_square : is_square R) :
  diagonals_bisect_each_other R := 
  sorry

end NUMINAMATH_GPT_common_property_of_rectangles_rhombuses_and_squares_l62_6221


namespace NUMINAMATH_GPT_min_value_ineq_l62_6240

open Real

theorem min_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5 * x + 2) * (y^2 + 5 * y + 2) * (z^2 + 5 * z + 2) / (x * y * z) ≥ 512 :=
by sorry

noncomputable def optimal_min_value : ℝ := 512

end NUMINAMATH_GPT_min_value_ineq_l62_6240


namespace NUMINAMATH_GPT_simplify_expression_l62_6234

-- Define the conditions as parameters
variable (x y : ℕ)

-- State the theorem with the required conditions and proof goal
theorem simplify_expression (hx : x = 2) (hy : y = 3) :
  (8 * x * y^2) / (6 * x^2 * y) = 2 := by
  -- We'll provide the outline and leave the proof as sorry
  sorry

end NUMINAMATH_GPT_simplify_expression_l62_6234


namespace NUMINAMATH_GPT_sum_of_coefficients_l62_6213

noncomputable def P (x : ℤ) : ℤ := (x ^ 2 - 3 * x + 1) ^ 100

theorem sum_of_coefficients : P 1 = 1 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l62_6213


namespace NUMINAMATH_GPT_painting_problem_equation_l62_6278

def dougPaintingRate := 1 / 3
def davePaintingRate := 1 / 4
def combinedPaintingRate := dougPaintingRate + davePaintingRate
def timeRequiredToComplete (t : ℝ) : Prop := 
  (t - 1) * combinedPaintingRate = 2 / 3

theorem painting_problem_equation : ∃ t : ℝ, timeRequiredToComplete t :=
sorry

end NUMINAMATH_GPT_painting_problem_equation_l62_6278


namespace NUMINAMATH_GPT_some_athletes_not_members_honor_society_l62_6205

universe u

variable {U : Type u} -- Assume U is our universe of discourse, e.g., individuals.
variables (Athletes Disciplined HonorSociety : U → Prop)

-- Conditions
def some_athletes_not_disciplined := ∃ x, Athletes x ∧ ¬Disciplined x
def all_honor_society_disciplined := ∀ x, HonorSociety x → Disciplined x

-- Correct Answer
theorem some_athletes_not_members_honor_society :
  some_athletes_not_disciplined Athletes Disciplined →
  all_honor_society_disciplined HonorSociety Disciplined →
  ∃ y, Athletes y ∧ ¬HonorSociety y :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_some_athletes_not_members_honor_society_l62_6205


namespace NUMINAMATH_GPT_xy_difference_l62_6286

theorem xy_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by {
    sorry
}

end NUMINAMATH_GPT_xy_difference_l62_6286


namespace NUMINAMATH_GPT_min_max_x_l62_6225

theorem min_max_x (n : ℕ) (hn : 0 < n) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = n * x + n * y) : 
  n + 1 ≤ x ∧ x ≤ n * (n + 1) :=
by {
  sorry  -- Proof goes here
}

end NUMINAMATH_GPT_min_max_x_l62_6225


namespace NUMINAMATH_GPT_cos_sum_of_angles_l62_6285

theorem cos_sum_of_angles (α β : Real) (h1 : Real.sin α = 4/5) (h2 : (π/2) < α ∧ α < π) 
(h3 : Real.cos β = -5/13) (h4 : 0 < β ∧ β < π/2) : 
  Real.cos (α + β) = -33/65 := 
by
  sorry

end NUMINAMATH_GPT_cos_sum_of_angles_l62_6285


namespace NUMINAMATH_GPT_find_largest_number_l62_6241

theorem find_largest_number (w x y z : ℕ) 
  (h1 : w + x + y = 190) 
  (h2 : w + x + z = 210) 
  (h3 : w + y + z = 220) 
  (h4 : x + y + z = 235) : 
  max (max w x) (max y z) = 95 := 
sorry

end NUMINAMATH_GPT_find_largest_number_l62_6241


namespace NUMINAMATH_GPT_ordered_pairs_count_l62_6217

theorem ordered_pairs_count : 
  (∃ s : Finset (ℕ × ℕ), (∀ p ∈ s, p.1 > 0 ∧ p.2 > 0 ∧ p.1 + p.2 ≤ 6) ∧ s.card = 15) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_ordered_pairs_count_l62_6217


namespace NUMINAMATH_GPT_max_distance_curve_line_l62_6261

noncomputable def curve_param_x (θ : ℝ) : ℝ := 1 + Real.cos θ
noncomputable def curve_param_y (θ : ℝ) : ℝ := Real.sin θ
noncomputable def line (x y : ℝ) : Prop := x + y + 2 = 0

theorem max_distance_curve_line 
  (θ : ℝ) 
  (x := curve_param_x θ) 
  (y := curve_param_y θ) :
  ∃ (d : ℝ), 
    (∀ t : ℝ, curve_param_x t = x ∧ curve_param_y t = y → d ≤ (abs (x + y + 2)) / Real.sqrt (1^2 + 1^2)) 
    ∧ d = (3 * Real.sqrt 2) / 2 + 1 :=
sorry

end NUMINAMATH_GPT_max_distance_curve_line_l62_6261


namespace NUMINAMATH_GPT_find_profits_maximize_profit_week3_l62_6260

-- Defining the conditions of the problems
def week1_sales_A := 10
def week1_sales_B := 12
def week1_profit := 2000

def week2_sales_A := 20
def week2_sales_B := 15
def week2_profit := 3100

def total_sales_week3 := 25

-- Condition: Sales of type B exceed sales of type A but do not exceed twice the sales of type A
def sales_condition (x : ℕ) := (total_sales_week3 - x) > x ∧ (total_sales_week3 - x) ≤ 2 * x

-- Define the profits for types A and B
def profit_A (a b : ℕ) := week1_sales_A * a + week1_sales_B * b = week1_profit
def profit_B (a b : ℕ) := week2_sales_A * a + week2_sales_B * b = week2_profit

-- Define the profit function for week 3
def profit_week3 (a b x : ℕ) := a * x + b * (total_sales_week3 - x)

theorem find_profits : ∃ a b, profit_A a b ∧ profit_B a b :=
by
  use 80, 100
  sorry

theorem maximize_profit_week3 : 
  ∃ x y, 
  sales_condition x ∧ 
  x + y = total_sales_week3 ∧ 
  profit_week3 80 100 x = 2320 :=
by
  use 9, 16
  sorry

end NUMINAMATH_GPT_find_profits_maximize_profit_week3_l62_6260


namespace NUMINAMATH_GPT_total_games_in_season_l62_6203

theorem total_games_in_season :
  let num_teams := 14
  let teams_per_division := 7
  let games_within_division_per_team := 6 * 3
  let games_against_other_division_per_team := 7
  let games_per_team := games_within_division_per_team + games_against_other_division_per_team
  let total_initial_games := games_per_team * num_teams
  let total_games := total_initial_games / 2
  total_games = 175 :=
by
  sorry

end NUMINAMATH_GPT_total_games_in_season_l62_6203


namespace NUMINAMATH_GPT_income_of_A_l62_6259

theorem income_of_A (x y : ℝ) (hx₁ : 5 * x - 3 * y = 1600) (hx₂ : 4 * x - 2 * y = 1600) : 
  5 * x = 4000 :=
by
  sorry

end NUMINAMATH_GPT_income_of_A_l62_6259


namespace NUMINAMATH_GPT_find_f_l62_6204

theorem find_f (f : ℝ → ℝ) (h₀ : f 0 = 1) (h₁ : ∀ x y, f (x * y) = f ((x^2 + y^2) / 2) + (x - y)^2) : 
  ∀ x, f x = 1 - 2 * x :=
by
  sorry  -- Proof not required

end NUMINAMATH_GPT_find_f_l62_6204


namespace NUMINAMATH_GPT_positive_integers_sum_reciprocal_l62_6281

theorem positive_integers_sum_reciprocal (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_sum : a + b + c = 2010) (h_recip : (1 : ℚ)/a + (1 : ℚ)/b + (1 : ℚ)/c = 1/58) :
  (a = 1740 ∧ b = 180 ∧ c = 90) ∨ 
  (a = 1740 ∧ b = 90 ∧ c = 180) ∨ 
  (a = 180 ∧ b = 90 ∧ c = 1740) ∨ 
  (a = 180 ∧ b = 1740 ∧ c = 90) ∨ 
  (a = 90 ∧ b = 1740 ∧ c = 180) ∨ 
  (a = 90 ∧ b = 180 ∧ c = 1740) := 
sorry

end NUMINAMATH_GPT_positive_integers_sum_reciprocal_l62_6281


namespace NUMINAMATH_GPT_inequality_abc_l62_6267

theorem inequality_abc 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_abc_l62_6267


namespace NUMINAMATH_GPT_carly_practice_time_l62_6283

-- conditions
def practice_time_butterfly_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def practice_time_backstroke_weekly (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  hours_per_day * days_per_week

def total_weekly_practice (butterfly_hours : ℕ) (backstroke_hours : ℕ) : ℕ :=
  butterfly_hours + backstroke_hours

def monthly_practice (weekly_hours : ℕ) (weeks_per_month : ℕ) : ℕ :=
  weekly_hours * weeks_per_month

-- Proof Problem Statement
theorem carly_practice_time :
  practice_time_butterfly_weekly 3 4 + practice_time_backstroke_weekly 2 6 * 4 = 96 :=
by
  sorry

end NUMINAMATH_GPT_carly_practice_time_l62_6283


namespace NUMINAMATH_GPT_perp_a_beta_l62_6288

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry
noncomputable def Incident (l : line) (p : plane) : Prop := sorry
noncomputable def Perpendicular (l1 l2 : line) : Prop := sorry
noncomputable def Parallel (l1 l2 : line) : Prop := sorry

variables {α β : plane} {a AB : line}

-- Conditions extracted from the problem
axiom condition1 : Perpendicular α β
axiom condition2 : Incident AB β ∧ Incident AB α
axiom condition3 : Parallel a α
axiom condition4 : Perpendicular a AB

-- The statement that needs to be proved
theorem perp_a_beta : Perpendicular a β :=
  sorry

end NUMINAMATH_GPT_perp_a_beta_l62_6288


namespace NUMINAMATH_GPT_amanda_speed_l62_6209

-- Defining the conditions
def distance : ℝ := 6 -- 6 miles
def time : ℝ := 3 -- 3 hours

-- Stating the question with the conditions and the correct answer
theorem amanda_speed : (distance / time) = 2 :=
by 
  -- the proof is skipped as instructed
  sorry

end NUMINAMATH_GPT_amanda_speed_l62_6209


namespace NUMINAMATH_GPT_value_of_y_l62_6273

theorem value_of_y (x y : ℝ) (h1 : x - y = 16) (h2 : x + y = 4) : y = -6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l62_6273


namespace NUMINAMATH_GPT_square_side_length_equals_nine_l62_6222

-- Definitions based on the conditions
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 8
def rectangle_perimeter (length width : ℕ) : ℕ := 2 * length + 2 * width
def side_length_of_square (perimeter : ℕ) : ℕ := perimeter / 4

-- The theorem we want to prove
theorem square_side_length_equals_nine : 
  side_length_of_square (rectangle_perimeter rectangle_length rectangle_width) = 9 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_square_side_length_equals_nine_l62_6222


namespace NUMINAMATH_GPT_even_three_digit_numbers_l62_6210

theorem even_three_digit_numbers (n : ℕ) :
  (n >= 100 ∧ n < 1000) ∧
  (n % 2 = 0) ∧
  ((n % 100) / 10 + (n % 10) = 12) →
  n = 12 :=
sorry

end NUMINAMATH_GPT_even_three_digit_numbers_l62_6210


namespace NUMINAMATH_GPT_geom_sequence_sum_correct_l62_6294

noncomputable def geom_sequence_sum (a₁ a₄ : ℕ) (S₅ : ℕ) :=
  ∃ q : ℕ, a₁ = 1 ∧ a₄ = a₁ * q ^ 3 ∧ S₅ = (a₁ * (1 - q ^ 5)) / (1 - q)

theorem geom_sequence_sum_correct : geom_sequence_sum 1 8 31 :=
by {
  sorry
}

end NUMINAMATH_GPT_geom_sequence_sum_correct_l62_6294


namespace NUMINAMATH_GPT_problem_statement_l62_6289

theorem problem_statement :
  ∃ (a b c : ℕ), gcd a (gcd b c) = 1 ∧
  (∃ x y : ℝ, 2 * y = 8 * x - 7) ∧
  a ^ 2 + b ^ 2 + (c:ℤ) ^ 2 = 117 :=
sorry

end NUMINAMATH_GPT_problem_statement_l62_6289


namespace NUMINAMATH_GPT_savings_percentage_first_year_l62_6200

noncomputable def savings_percentage (I S : ℝ) : ℝ := (S / I) * 100

theorem savings_percentage_first_year (I S : ℝ) (h1 : S = 0.20 * I) :
  savings_percentage I S = 20 :=
by
  unfold savings_percentage
  rw [h1]
  field_simp
  norm_num
  sorry

end NUMINAMATH_GPT_savings_percentage_first_year_l62_6200


namespace NUMINAMATH_GPT_remainder_of_square_l62_6219

variable (N X : Set ℤ)
variable (k : ℤ)

/-- Given any n in set N and any x in set X, where dividing n by x gives a remainder of 3,
prove that the remainder of n^2 divided by x is 9 mod x. -/
theorem remainder_of_square (n x : ℤ) (hn : n ∈ N) (hx : x ∈ X)
  (h : ∃ k, n = k * x + 3) : (n^2) % x = 9 % x :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_square_l62_6219


namespace NUMINAMATH_GPT_Q_lies_in_third_quadrant_l62_6287

theorem Q_lies_in_third_quadrant (b : ℝ) (P_in_fourth_quadrant : 2 > 0 ∧ b < 0) :
    b < 0 ∧ -2 < 0 ↔
    (b < 0 ∧ -2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_Q_lies_in_third_quadrant_l62_6287


namespace NUMINAMATH_GPT_neg_number_among_set_l62_6295

theorem neg_number_among_set :
  ∃ n ∈ ({5, 1, -2, 0} : Set ℤ), n < 0 ∧ n = -2 :=
by
  sorry

end NUMINAMATH_GPT_neg_number_among_set_l62_6295


namespace NUMINAMATH_GPT_value_of_x_l62_6248

theorem value_of_x (x y : ℕ) (h1 : x / y = 3) (h2 : y = 25) : x = 75 := by
  sorry

end NUMINAMATH_GPT_value_of_x_l62_6248


namespace NUMINAMATH_GPT_scientific_notation_29150000_l62_6247

theorem scientific_notation_29150000 :
  29150000 = 2.915 * 10^7 := sorry

end NUMINAMATH_GPT_scientific_notation_29150000_l62_6247


namespace NUMINAMATH_GPT_alma_carrots_leftover_l62_6275

/-- Alma has 47 baby carrots and wishes to distribute them equally among 4 goats.
    We need to prove that the number of leftover carrots after such distribution is 3. -/
theorem alma_carrots_leftover (total_carrots : ℕ) (goats : ℕ) (leftover : ℕ) 
  (h1 : total_carrots = 47) (h2 : goats = 4) (h3 : leftover = total_carrots % goats) : 
  leftover = 3 :=
by
  sorry

end NUMINAMATH_GPT_alma_carrots_leftover_l62_6275


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l62_6229

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x^2 > 1 → 1 / x < 1) ∧ (¬(1 / x < 1 → x^2 > 1)) :=
by sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l62_6229


namespace NUMINAMATH_GPT_count_bottom_right_arrows_l62_6242

/-!
# Problem Statement
Each blank cell on the edge is to be filled with an arrow. The number in each square indicates the number of arrows pointing to that number. The arrows can point in the following directions: up, down, left, right, top-left, top-right, bottom-left, and bottom-right. Each arrow must point to a number. Figure 3 is provided and based on this, determine the number of arrows pointing to the bottom-right direction.
-/

def bottom_right_arrows_count : Nat :=
  2

theorem count_bottom_right_arrows :
  bottom_right_arrows_count = 2 :=
by
  sorry

end NUMINAMATH_GPT_count_bottom_right_arrows_l62_6242


namespace NUMINAMATH_GPT_total_volume_of_cubes_l62_6256

theorem total_volume_of_cubes (Jim_cubes : Nat) (Jim_side_length : Nat) 
    (Laura_cubes : Nat) (Laura_side_length : Nat)
    (h1 : Jim_cubes = 7) (h2 : Jim_side_length = 3) 
    (h3 : Laura_cubes = 4) (h4 : Laura_side_length = 4) : 
    (Jim_cubes * Jim_side_length^3 + Laura_cubes * Laura_side_length^3 = 445) :=
by
  sorry

end NUMINAMATH_GPT_total_volume_of_cubes_l62_6256


namespace NUMINAMATH_GPT_vectors_form_basis_l62_6262

-- Define the vectors in set B
def e1 : ℝ × ℝ := (-1, 2)
def e2 : ℝ × ℝ := (3, 7)

-- Define a function that checks if two vectors form a basis
def form_basis (v1 v2 : ℝ × ℝ) : Prop :=
  let det := v1.1 * v2.2 - v1.2 * v2.1
  det ≠ 0

-- State the theorem that vectors e1 and e2 form a basis
theorem vectors_form_basis : form_basis e1 e2 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_vectors_form_basis_l62_6262


namespace NUMINAMATH_GPT_bmw_cars_sold_l62_6263

def percentage_non_bmw (ford_pct nissan_pct chevrolet_pct : ℕ) : ℕ :=
  ford_pct + nissan_pct + chevrolet_pct

def percentage_bmw (total_pct non_bmw_pct : ℕ) : ℕ :=
  total_pct - non_bmw_pct

def number_of_bmws (total_cars bmw_pct : ℕ) : ℕ :=
  (total_cars * bmw_pct) / 100

theorem bmw_cars_sold (total_cars ford_pct nissan_pct chevrolet_pct : ℕ)
  (h_total_cars : total_cars = 300)
  (h_ford_pct : ford_pct = 20)
  (h_nissan_pct : nissan_pct = 25)
  (h_chevrolet_pct : chevrolet_pct = 10) :
  number_of_bmws total_cars (percentage_bmw 100 (percentage_non_bmw ford_pct nissan_pct chevrolet_pct)) = 135 := by
  sorry

end NUMINAMATH_GPT_bmw_cars_sold_l62_6263


namespace NUMINAMATH_GPT_find_f_neg2_l62_6206

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - 3 else -(2^(-x) - 3)

theorem find_f_neg2 : f (-2) = -1 :=
sorry

end NUMINAMATH_GPT_find_f_neg2_l62_6206


namespace NUMINAMATH_GPT_problem_statement_l62_6250

variables {totalBuyers : ℕ}
variables {C M K CM CK MK CMK : ℕ}

-- Given conditions
def conditions (totalBuyers : ℕ) (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : Prop :=
  totalBuyers = 150 ∧
  C = 70 ∧
  M = 60 ∧
  K = 50 ∧
  CM = 25 ∧
  CK = 15 ∧
  MK = 10 ∧
  CMK = 5

-- Number of buyers who purchase at least one mixture
def buyersAtLeastOne (C : ℕ) (M : ℕ) (K : ℕ)
  (CM : ℕ) (CK : ℕ) (MK : ℕ) (CMK : ℕ) : ℕ :=
  C + M + K - CM - CK - MK + CMK

-- Number of buyers who purchase none
def buyersNone (totalBuyers : ℕ) (buyersAtLeastOne : ℕ) : ℕ :=
  totalBuyers - buyersAtLeastOne

-- Probability computation
def probabilityNone (totalBuyers : ℕ) (buyersNone : ℕ) : ℚ :=
  buyersNone / totalBuyers

-- Theorem statement
theorem problem_statement : conditions totalBuyers C M K CM CK MK CMK →
  probabilityNone totalBuyers (buyersNone totalBuyers (buyersAtLeastOne C M K CM CK MK CMK)) = 0.1 :=
by
  intros h
  -- Assumptions from the problem
  have h_total : totalBuyers = 150 := h.left
  have hC : C = 70 := h.right.left
  have hM : M = 60 := h.right.right.left
  have hK : K = 50 := h.right.right.right.left
  have hCM : CM = 25 := h.right.right.right.right.left
  have hCK : CK = 15 := h.right.right.right.right.right.left
  have hMK : MK = 10 := h.right.right.right.right.right.right.left
  have hCMK : CMK = 5 := h.right.right.right.right.right.right.right
  sorry

end NUMINAMATH_GPT_problem_statement_l62_6250


namespace NUMINAMATH_GPT_value_of_y_when_x_is_zero_l62_6265

noncomputable def quadratic_y (h x : ℝ) : ℝ := -(x + h)^2

theorem value_of_y_when_x_is_zero :
  ∀ (h : ℝ), (∀ x, x < -3 → quadratic_y h x < quadratic_y h (-3)) →
            (∀ x, x > -3 → quadratic_y h x < quadratic_y h (-3)) →
            quadratic_y h 0 = -9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_when_x_is_zero_l62_6265


namespace NUMINAMATH_GPT_car_b_speed_l62_6202

theorem car_b_speed
  (v_A v_B : ℝ) (d_A d_B d : ℝ)
  (h1 : v_A = 5 / 3 * v_B)
  (h2 : d_A = v_A * 5)
  (h3 : d_B = v_B * 5)
  (h4 : d = d_A + d_B)
  (h5 : d_A = d / 2 + 25) :
  v_B = 15 := 
sorry

end NUMINAMATH_GPT_car_b_speed_l62_6202


namespace NUMINAMATH_GPT_values_of_x_l62_6214

theorem values_of_x (x : ℝ) (h1 : x^2 - 3 * x - 10 < 0) (h2 : 1 < x) : 1 < x ∧ x < 5 := 
sorry

end NUMINAMATH_GPT_values_of_x_l62_6214


namespace NUMINAMATH_GPT_pages_per_inch_l62_6228

theorem pages_per_inch (number_of_books : ℕ) (average_pages_per_book : ℕ) (total_thickness : ℕ) 
                        (H1 : number_of_books = 6)
                        (H2 : average_pages_per_book = 160)
                        (H3 : total_thickness = 12) :
  (number_of_books * average_pages_per_book) / total_thickness = 80 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_pages_per_inch_l62_6228


namespace NUMINAMATH_GPT_find_common_difference_l62_6215

variable (a an Sn d : ℚ)
variable (n : ℕ)

def arithmetic_sequence (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

def sum_arithmetic_sequence (a : ℚ) (an : ℚ) (n : ℕ) : ℚ :=
  n * (a + an) / 2

theorem find_common_difference
  (h1 : a = 3)
  (h2 : an = 50)
  (h3 : Sn = 318)
  (h4 : an = arithmetic_sequence a d n)
  (h5 : Sn = sum_arithmetic_sequence a an n) :
  d = 47 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_common_difference_l62_6215


namespace NUMINAMATH_GPT_price_of_book_l62_6216

variables (D B : ℝ)

def younger_brother : ℝ := 10

theorem price_of_book 
  (h1 : D = 1/2 * (B + younger_brother))
  (h2 : B = 1/3 * (D + younger_brother)) : 
  D + B + younger_brother = 24 := 
sorry

end NUMINAMATH_GPT_price_of_book_l62_6216


namespace NUMINAMATH_GPT_total_number_of_members_l62_6266

-- Define the basic setup
def committees := Fin 5
def members := {m : Finset committees // m.card = 2}

-- State the theorem
theorem total_number_of_members :
  (∃ s : Finset members, s.card = 10) :=
sorry

end NUMINAMATH_GPT_total_number_of_members_l62_6266


namespace NUMINAMATH_GPT_marks_chemistry_l62_6208

-- Definitions based on conditions
def marks_english : ℕ := 96
def marks_math : ℕ := 98
def marks_physics : ℕ := 99
def marks_biology : ℕ := 98
def average_marks : ℝ := 98.2
def num_subjects : ℕ := 5

-- Statement to prove
theorem marks_chemistry :
  ((marks_english + marks_math + marks_physics + marks_biology : ℕ) + (x : ℕ)) / num_subjects = average_marks →
  x = 100 :=
by
  sorry

end NUMINAMATH_GPT_marks_chemistry_l62_6208


namespace NUMINAMATH_GPT_ninth_term_geometric_sequence_l62_6277

noncomputable def geometric_seq (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem ninth_term_geometric_sequence (a r : ℝ) (h_positive : ∀ n, 0 < geometric_seq a r n)
  (h_fifth_term : geometric_seq a r 5 = 32)
  (h_eleventh_term : geometric_seq a r 11 = 2) :
  geometric_seq a r 9 = 2 :=
by
{
  sorry
}

end NUMINAMATH_GPT_ninth_term_geometric_sequence_l62_6277


namespace NUMINAMATH_GPT_problem_statement_l62_6224

theorem problem_statement (a b : ℝ) (h : a > b) : a - 1 > b - 1 :=
sorry

end NUMINAMATH_GPT_problem_statement_l62_6224


namespace NUMINAMATH_GPT_champagne_bottles_needed_l62_6292

-- Define the initial conditions of the problem
def num_guests : ℕ := 120
def glasses_per_guest : ℕ := 2
def servings_per_bottle : ℕ := 6

-- The statement we need to prove
theorem champagne_bottles_needed : 
  (num_guests * glasses_per_guest) / servings_per_bottle = 40 := 
by
  sorry

end NUMINAMATH_GPT_champagne_bottles_needed_l62_6292


namespace NUMINAMATH_GPT_circumradius_of_consecutive_triangle_l62_6272

theorem circumradius_of_consecutive_triangle
  (a b c : ℕ)
  (h : a = b - 1)
  (h1 : c = b + 1)
  (r : ℝ)
  (h2 : r = 4)
  (h3 : a + b > c)
  (h4 : a + c > b)
  (h5 : b + c > a)
  : ∃ R : ℝ, R = 65 / 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_circumradius_of_consecutive_triangle_l62_6272


namespace NUMINAMATH_GPT_first_discount_is_20_percent_l62_6245

-- Define the problem parameters
def original_price : ℝ := 200
def final_price : ℝ := 152
def second_discount : ℝ := 0.05

-- Define the function to compute the price after two discounts
def price_after_discounts (first_discount : ℝ) : ℝ := 
  original_price * (1 - first_discount) * (1 - second_discount)

-- Define the statement that we need to prove
theorem first_discount_is_20_percent : 
  ∃ (first_discount : ℝ), price_after_discounts first_discount = final_price ∧ first_discount = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_is_20_percent_l62_6245


namespace NUMINAMATH_GPT_xiaoming_grandfather_age_l62_6238

def grandfather_age (x xm_diff : ℕ) :=
  xm_diff = 60 ∧ x > 7 * (x - xm_diff) ∧ x < 70

theorem xiaoming_grandfather_age (x : ℕ) (h_cond : grandfather_age x 60) : x = 69 :=
by
  sorry

end NUMINAMATH_GPT_xiaoming_grandfather_age_l62_6238


namespace NUMINAMATH_GPT_ivy_has_20_collectors_dolls_l62_6232

theorem ivy_has_20_collectors_dolls
  (D : ℕ) (I : ℕ) (C : ℕ)
  (h1 : D = 60)
  (h2 : D = 2 * I)
  (h3 : C = 2 * I / 3) 
  : C = 20 :=
by sorry

end NUMINAMATH_GPT_ivy_has_20_collectors_dolls_l62_6232


namespace NUMINAMATH_GPT_div_condition_for_lcm_l62_6220

theorem div_condition_for_lcm (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h : Nat.lcm (x + 2) (y + 2) - Nat.lcm (x + 1) (y + 1) = Nat.lcm (x + 1) (y + 1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x :=
sorry

end NUMINAMATH_GPT_div_condition_for_lcm_l62_6220


namespace NUMINAMATH_GPT_find_all_solutions_l62_6226

def is_solution (f : ℕ → ℝ) : Prop :=
  (∀ n ≥ 1, f (n + 1) ≥ f n) ∧
  (∀ m n, Nat.gcd m n = 1 → f (m * n) = f m * f n)

theorem find_all_solutions :
  ∀ f : ℕ → ℝ, is_solution f →
    (∀ n, f n = 0) ∨ (∃ a ≥ 0, ∀ n, f n = n ^ a) :=
sorry

end NUMINAMATH_GPT_find_all_solutions_l62_6226


namespace NUMINAMATH_GPT_range_of_k_l62_6274

theorem range_of_k
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : a^2 + c^2 = 16)
  (h2 : b^2 + c^2 = 25) : 
  9 < a^2 + b^2 ∧ a^2 + b^2 < 41 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l62_6274


namespace NUMINAMATH_GPT_factorize_x_squared_minus_one_l62_6257

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_one_l62_6257


namespace NUMINAMATH_GPT_fraction_value_condition_l62_6236

theorem fraction_value_condition (m n : ℚ) (h : m / n = 2 / 3) : m / (m + n) = 2 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_value_condition_l62_6236


namespace NUMINAMATH_GPT_paul_lives_on_story_5_l62_6231

/-- 
Given:
1. Each story is 10 feet tall.
2. Paul makes 3 trips out from and back to his apartment each day.
3. Over a week (7 days), he travels 2100 feet vertically in total.

Prove that the story on which Paul lives \( S \) is 5.
-/
theorem paul_lives_on_story_5 (height_per_story : ℕ)
  (trips_per_day : ℕ)
  (number_of_days : ℕ)
  (total_feet_travelled : ℕ)
  (S : ℕ) :
  height_per_story = 10 → 
  trips_per_day = 3 → 
  number_of_days = 7 → 
  total_feet_travelled = 2100 → 
  2 * height_per_story * trips_per_day * number_of_days * S = total_feet_travelled → 
  S = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_paul_lives_on_story_5_l62_6231


namespace NUMINAMATH_GPT_division_value_of_712_5_by_12_5_is_57_l62_6268

theorem division_value_of_712_5_by_12_5_is_57 : 712.5 / 12.5 = 57 :=
  by
    sorry

end NUMINAMATH_GPT_division_value_of_712_5_by_12_5_is_57_l62_6268


namespace NUMINAMATH_GPT_find_f_three_l62_6246

def f : ℝ → ℝ := sorry

theorem find_f_three (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_find_f_three_l62_6246


namespace NUMINAMATH_GPT_trivia_team_students_l62_6251

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (h_not_picked : not_picked = 9) 
(h_groups : groups = 3) (h_students_per_group : students_per_group = 9) :
    not_picked + (groups * students_per_group) = 36 := by
  sorry

end NUMINAMATH_GPT_trivia_team_students_l62_6251


namespace NUMINAMATH_GPT_goldfish_cost_graph_is_finite_set_of_points_l62_6235

theorem goldfish_cost_graph_is_finite_set_of_points :
  ∀ (n : ℤ), (1 ≤ n ∧ n ≤ 12) → ∃ (C : ℤ), C = 15 * n ∧ ∀ m ≠ n, C ≠ 15 * m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_goldfish_cost_graph_is_finite_set_of_points_l62_6235


namespace NUMINAMATH_GPT_total_visitors_over_two_days_l62_6255

-- Definitions of the conditions
def visitors_on_Saturday : ℕ := 200
def additional_visitors_on_Sunday : ℕ := 40

-- Statement of the problem
theorem total_visitors_over_two_days :
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  total_visitors = 440 :=
by
  let visitors_on_Sunday := visitors_on_Saturday + additional_visitors_on_Sunday
  let total_visitors := visitors_on_Saturday + visitors_on_Sunday
  sorry

end NUMINAMATH_GPT_total_visitors_over_two_days_l62_6255


namespace NUMINAMATH_GPT_rectangle_dimensions_l62_6254

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : 2 * l + 2 * w = 150) 
  (h2 : l = w + 15) : 
  w = 30 ∧ l = 45 := 
  by 
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l62_6254


namespace NUMINAMATH_GPT_find_integer_l62_6249

-- Definition of the given conditions
def conditions (x : ℤ) (r : ℤ) : Prop :=
  (0 ≤ r ∧ r < 7) ∧ ((x - 77) * 8 = 259 + r)

-- Statement of the theorem to be proved
theorem find_integer : ∃ x : ℤ, ∃ r : ℤ, conditions x r ∧ (x = 110) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_l62_6249


namespace NUMINAMATH_GPT_find_four_digit_number_l62_6270

-- Definitions of the digit variables a, b, c, d, and their constraints.
def four_digit_expressions_meet_condition (abcd abc ab : ℕ) (a : ℕ) :=
  ∃ (b c d : ℕ), abcd = (1000 * a + 100 * b + 10 * c + d)
  ∧ abc = (100 * a + 10 * b + c)
  ∧ ab = (10 * a + b)
  ∧ abcd - abc - ab - a = 1787

-- Main statement to be proven.
theorem find_four_digit_number
: ∀ a b c d : ℕ, 
  four_digit_expressions_meet_condition (1000 * a + 100 * b + 10 * c + d) (100 * a + 10 * b + c) (10 * a + b) a
  → (a = 2 ∧ b = 0 ∧ ((c = 0 ∧ d = 9) ∨ (c = 1 ∧ d = 0))) :=
sorry

end NUMINAMATH_GPT_find_four_digit_number_l62_6270


namespace NUMINAMATH_GPT_sets_are_equal_l62_6233

def setA : Set ℤ := {x | ∃ a b : ℤ, x = 12 * a + 8 * b}
def setB : Set ℤ := {y | ∃ c d : ℤ, y = 20 * c + 16 * d}

theorem sets_are_equal : setA = setB := 
by
  sorry

end NUMINAMATH_GPT_sets_are_equal_l62_6233


namespace NUMINAMATH_GPT_obtain_26_kg_of_sand_l62_6211

theorem obtain_26_kg_of_sand :
  ∃ (x y : ℕ), (37 - x = x + 3) ∧ (20 - y = y + 2) ∧ (x + y = 26) := by
  sorry

end NUMINAMATH_GPT_obtain_26_kg_of_sand_l62_6211


namespace NUMINAMATH_GPT_triangle_area_squared_l62_6269

theorem triangle_area_squared
  (R : ℝ)
  (A : ℝ)
  (AC_minus_AB : ℝ)
  (area : ℝ)
  (hx : R = 4)
  (hy : A = 60)
  (hz : AC_minus_AB = 4)
  (area_eq : area = 8 * Real.sqrt 3) :
  area^2 = 192 :=
by
  -- We include the conditions 
  have hR := hx
  have hA := hy
  have hAC_AB := hz
  have harea := area_eq
  -- We will use these to construct the required proof 
  sorry

end NUMINAMATH_GPT_triangle_area_squared_l62_6269


namespace NUMINAMATH_GPT_part1_inequality_l62_6291

theorem part1_inequality (a b x y : ℝ) (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) 
    (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_a_ge_x : a ≥ x) : 
    (a - x) ^ 2 + (b - y) ^ 2 ≤ (a + b - x) ^ 2 + y ^ 2 := 
by 
  sorry

end NUMINAMATH_GPT_part1_inequality_l62_6291


namespace NUMINAMATH_GPT_find_a10_l62_6290

variable {q : ℝ}
variable {a : ℕ → ℝ}

-- Sequence conditions
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
axiom positive_ratio : 0 < q
axiom condition_1 : a 2 = 1
axiom condition_2 : a 4 * a 8 = 2 * (a 5) ^ 2

theorem find_a10 : a 10 = 16 := by
  sorry

end NUMINAMATH_GPT_find_a10_l62_6290


namespace NUMINAMATH_GPT_largest_5_digit_integer_congruent_to_19_mod_26_l62_6237

theorem largest_5_digit_integer_congruent_to_19_mod_26 :
  ∃ n : ℕ, 10000 ≤ 26 * n + 19 ∧ 26 * n + 19 < 100000 ∧ (26 * n + 19 ≡ 19 [MOD 26]) ∧ 26 * n + 19 = 99989 :=
by
  sorry

end NUMINAMATH_GPT_largest_5_digit_integer_congruent_to_19_mod_26_l62_6237


namespace NUMINAMATH_GPT_max_f_on_interval_l62_6230

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.sqrt 3) * Real.sin x * Real.cos x

theorem max_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f y ≤ f x ∧ f x = 3 / 2 :=
  sorry

end NUMINAMATH_GPT_max_f_on_interval_l62_6230


namespace NUMINAMATH_GPT_sector_central_angle_l62_6284

theorem sector_central_angle (r l θ : ℝ) (h_perimeter : 2 * r + l = 8) (h_area : (1 / 2) * l * r = 4) : θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l62_6284


namespace NUMINAMATH_GPT_inequality_proof_l62_6264

variable (a b c : ℝ)

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c)) +
    (b / Real.sqrt (b^2 + 8 * a * c)) +
    (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l62_6264


namespace NUMINAMATH_GPT_cost_of_1000_gums_in_dollars_l62_6298

theorem cost_of_1000_gums_in_dollars :
  let cost_per_piece_in_cents := 1
  let pieces := 1000
  let cents_per_dollar := 100
  ∃ cost_in_dollars : ℝ, cost_in_dollars = (cost_per_piece_in_cents * pieces) / cents_per_dollar :=
sorry

end NUMINAMATH_GPT_cost_of_1000_gums_in_dollars_l62_6298


namespace NUMINAMATH_GPT_trapezoid_not_isosceles_l62_6258

noncomputable def is_trapezoid (BC AD AC : ℝ) : Prop :=
BC = 3 ∧ AD = 4 ∧ AC = 6

def is_isosceles_trapezoid_not_possible (BC AD AC : ℝ) : Prop :=
is_trapezoid BC AD AC → ¬(BC = AD)

theorem trapezoid_not_isosceles (BC AD AC : ℝ) :
  is_isosceles_trapezoid_not_possible BC AD AC :=
sorry

end NUMINAMATH_GPT_trapezoid_not_isosceles_l62_6258


namespace NUMINAMATH_GPT_wood_burned_in_afternoon_l62_6293

theorem wood_burned_in_afternoon 
  (burned_morning : ℕ) 
  (start_bundles : ℕ) 
  (end_bundles : ℕ) 
  (burned_afternoon : ℕ) 
  (h1 : burned_morning = 4) 
  (h2 : start_bundles = 10) 
  (h3 : end_bundles = 3) 
  (h4 : burned_morning + burned_afternoon = start_bundles - end_bundles) :
  burned_afternoon = 3 := 
sorry

end NUMINAMATH_GPT_wood_burned_in_afternoon_l62_6293


namespace NUMINAMATH_GPT_single_elimination_games_l62_6280

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = n - 1 :=
by
  have h1 : n = 512 := h
  use 511
  sorry

end NUMINAMATH_GPT_single_elimination_games_l62_6280


namespace NUMINAMATH_GPT_min_value_expr_l62_6279

theorem min_value_expr (x y : ℝ) (hx : x > 1) (hy : y > 1) : (x^3 / (y - 1)) + (y^3 / (x - 1)) ≥ 24 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l62_6279


namespace NUMINAMATH_GPT_range_of_a_l62_6296

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l62_6296


namespace NUMINAMATH_GPT_second_month_sale_l62_6201

theorem second_month_sale 
  (sale_1st: ℕ) (sale_3rd: ℕ) (sale_4th: ℕ) (sale_5th: ℕ) (sale_6th: ℕ) (avg_sale: ℕ)
  (h1: sale_1st = 5266) (h3: sale_3rd = 5864)
  (h4: sale_4th = 6122) (h5: sale_5th = 6588)
  (h6: sale_6th = 4916) (h_avg: avg_sale = 5750) :
  ∃ sale_2nd, (sale_1st + sale_2nd + sale_3rd + sale_4th + sale_5th + sale_6th) / 6 = avg_sale :=
by
  sorry

end NUMINAMATH_GPT_second_month_sale_l62_6201


namespace NUMINAMATH_GPT_find_x_coordinate_l62_6271

-- Define the center and radius of the circle
structure Circle where
  center : (ℝ × ℝ)
  radius : ℝ

-- Define the points on the circle
def lies_on_circle (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (x_c, y_c) := C.center
  let (x_p, y_p) := P
  (x_p - x_c)^2 + (y_p - y_c)^2 = C.radius^2

-- Lean 4 statement
theorem find_x_coordinate :
  ∀ (C : Circle), C.radius = 2 → lies_on_circle C (2, 0) ∧ lies_on_circle C (-2, 0) → 2 = 2 := by
  intro C h_radius ⟨h_lies_on_2_0, h_lies_on__2_0⟩
  sorry

end NUMINAMATH_GPT_find_x_coordinate_l62_6271


namespace NUMINAMATH_GPT_work_duration_l62_6227

theorem work_duration (work_rate_x work_rate_y : ℚ) (time_x : ℕ) (total_work : ℚ) :
  work_rate_x = (1 / 20) → 
  work_rate_y = (1 / 12) → 
  time_x = 4 → 
  total_work = 1 →
  ((time_x * work_rate_x) + ((total_work - (time_x * work_rate_x)) / (work_rate_x + work_rate_y))) = 10 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_work_duration_l62_6227


namespace NUMINAMATH_GPT_find_two_numbers_l62_6207

theorem find_two_numbers (x y : ℕ) : 
  (x + y = 20) ∧
  (x * y = 96) ↔ 
  ((x = 12 ∧ y = 8) ∨ (x = 8 ∧ y = 12)) := 
by
  sorry

end NUMINAMATH_GPT_find_two_numbers_l62_6207


namespace NUMINAMATH_GPT_zed_to_wyes_l62_6239

theorem zed_to_wyes (value_ex: ℝ) (value_wye: ℝ) (value_zed: ℝ)
  (h1: 2 * value_ex = 29 * value_wye)
  (h2: value_zed = 16 * value_ex) : value_zed = 232 * value_wye := by
  sorry

end NUMINAMATH_GPT_zed_to_wyes_l62_6239


namespace NUMINAMATH_GPT_gail_working_hours_x_l62_6253

theorem gail_working_hours_x (x : ℕ) (hx : x < 12) : 
  let hours_am := 12 - x
  let hours_pm := x
  hours_am + hours_pm = 12 := 
by {
  sorry
}

end NUMINAMATH_GPT_gail_working_hours_x_l62_6253


namespace NUMINAMATH_GPT_equilibrium_possible_l62_6252

variables {a b θ : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : (b / 2) < a) (h4 : a ≤ b)

theorem equilibrium_possible :
  θ = 0 ∨ θ = Real.arccos ((b^2 + 2 * a^2) / (3 * a * b)) → 
  (b / 2) < a ∧ a ≤ b ∧ (0 ≤ θ ∧ θ ≤ π) :=
sorry

end NUMINAMATH_GPT_equilibrium_possible_l62_6252


namespace NUMINAMATH_GPT_max_product_of_sum_2020_l62_6243

/--
  Prove that the maximum product of two integers whose sum is 2020 is 1020100.
-/
theorem max_product_of_sum_2020 : 
  ∃ x : ℤ, (x + (2020 - x) = 2020) ∧ (x * (2020 - x) = 1020100) :=
by
  sorry

end NUMINAMATH_GPT_max_product_of_sum_2020_l62_6243


namespace NUMINAMATH_GPT_sum_powers_l62_6244

open Complex

theorem sum_powers (ω : ℂ) (h₁ : ω^5 = 1) (h₂ : ω ≠ 1) : 
  ω^10 + ω^12 + ω^14 + ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 = 1 := sorry

end NUMINAMATH_GPT_sum_powers_l62_6244


namespace NUMINAMATH_GPT_num_employees_excluding_manager_l62_6276

/-- 
If the average monthly salary of employees is Rs. 1500, 
and adding a manager with salary Rs. 14100 increases 
the average salary by Rs. 600, prove that the number 
of employees (excluding the manager) is 20.
-/
theorem num_employees_excluding_manager 
  (avg_salary : ℕ) 
  (manager_salary : ℕ) 
  (new_avg_increase : ℕ) : 
  (∃ n : ℕ, 
    avg_salary = 1500 ∧ 
    manager_salary = 14100 ∧ 
    new_avg_increase = 600 ∧ 
    n = 20) := 
sorry

end NUMINAMATH_GPT_num_employees_excluding_manager_l62_6276
