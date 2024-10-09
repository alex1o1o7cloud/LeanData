import Mathlib

namespace evaluate_expression_l182_18262

theorem evaluate_expression : (18 * 3 + 6) / (6 - 3) = 20 := by
  sorry

end evaluate_expression_l182_18262


namespace angles_arithmetic_progression_l182_18277

theorem angles_arithmetic_progression (A B C : ℝ) (h_sum : A + B + C = 180) :
  (B = 60) ↔ (A + C = 2 * B) :=
by
  sorry

end angles_arithmetic_progression_l182_18277


namespace perpendicular_bisector_correct_vertex_C_correct_l182_18247

-- Define the vertices A, B, and the coordinates of the angle bisector line
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := -1, y := -1 }

-- The angle bisector CD equation
def angle_bisector_CD (p : Point) : Prop :=
  p.x + p.y - 1 = 0

-- The perpendicular bisector equation of side AB
def perpendicular_bisector_AB (p : Point) : Prop :=
  4 * p.x + 6 * p.y - 3 = 0

-- Coordinates of vertex C
def C_coordinates (c : Point) : Prop :=
  c.x = -1 ∧ c.y = 2

theorem perpendicular_bisector_correct :
  ∀ (M : Point), M.x = 0 ∧ M.y = 1/2 →
  ∀ (p : Point), perpendicular_bisector_AB p :=
sorry

theorem vertex_C_correct :
  ∃ (C : Point), angle_bisector_CD C ∧ (C : Point) = { x := -1, y := 2 } :=
sorry

end perpendicular_bisector_correct_vertex_C_correct_l182_18247


namespace no_int_solutions_for_equation_l182_18287

theorem no_int_solutions_for_equation :
  ¬ ∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3 * y^2 = x^4 + x := 
sorry

end no_int_solutions_for_equation_l182_18287


namespace original_cost_of_meal_l182_18291

-- Definitions for conditions
def meal_cost (initial_cost : ℝ) : ℝ :=
  initial_cost + 0.085 * initial_cost + 0.18 * initial_cost

-- The theorem we aim to prove
theorem original_cost_of_meal (total_cost : ℝ) (h : total_cost = 35.70) :
  ∃ initial_cost : ℝ, initial_cost = 28.23 ∧ meal_cost initial_cost = total_cost :=
by
  use 28.23
  rw [meal_cost, h]
  sorry

end original_cost_of_meal_l182_18291


namespace vertex_of_parabola_l182_18295

theorem vertex_of_parabola (a b c : ℝ) (h k : ℝ) (x y : ℝ) :
  (∀ x, y = (1/2) * (x - 1)^2 + 2) → (h, k) = (1, 2) :=
by
  intro hy
  exact sorry

end vertex_of_parabola_l182_18295


namespace cirrus_clouds_count_l182_18208

theorem cirrus_clouds_count 
  (cirrus cumulus cumulonimbus : ℕ)
  (h1 : cirrus = 4 * cumulus)
  (h2 : cumulus = 12 * cumulonimbus)
  (h3 : cumulonimbus = 3) : 
  cirrus = 144 := 
by
  sorry

end cirrus_clouds_count_l182_18208


namespace problem_statement_l182_18231

variable (f : ℝ → ℝ)

theorem problem_statement (h : ∀ x : ℝ, 2 * (f x) + x * (deriv f x) > x^2) :
  ∀ x : ℝ, x^2 * f x ≥ 0 :=
by
  sorry

end problem_statement_l182_18231


namespace mike_office_visits_per_day_l182_18212

-- Define the constants from the conditions
def pull_ups_per_visit : ℕ := 2
def total_pull_ups_per_week : ℕ := 70
def days_per_week : ℕ := 7

-- Calculate total office visits per week
def office_visits_per_week : ℕ := total_pull_ups_per_week / pull_ups_per_visit

-- Lean statement that states Mike goes into his office 5 times a day
theorem mike_office_visits_per_day : office_visits_per_week / days_per_week = 5 := by
  sorry

end mike_office_visits_per_day_l182_18212


namespace max_z_value_l182_18290

theorem max_z_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x * y + y * z + z * x = 3) : z = 13 / 3 := 
sorry

end max_z_value_l182_18290


namespace x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l182_18272

theorem x_eq_1_sufficient_not_necessary_for_x_sq_eq_1 (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ((x^2 = 1) → (x = 1 ∨ x = -1)) :=
by 
  sorry

end x_eq_1_sufficient_not_necessary_for_x_sq_eq_1_l182_18272


namespace martha_bedroom_size_l182_18270

theorem martha_bedroom_size (x jenny_size total_size : ℤ) (h₁ : jenny_size = x + 60) (h₂ : total_size = x + jenny_size) (h_total : total_size = 300) : x = 120 :=
by
  -- Adding conditions and the ultimate goal
  sorry


end martha_bedroom_size_l182_18270


namespace men_in_second_group_l182_18223

theorem men_in_second_group (M : ℕ) (W : ℝ) (h1 : 15 * 25 = W) (h2 : M * 18.75 = W) : M = 20 :=
sorry

end men_in_second_group_l182_18223


namespace phase_shift_of_sine_l182_18207

theorem phase_shift_of_sine :
  let B := 5
  let C := (3 * Real.pi) / 2
  let phase_shift := C / B
  phase_shift = (3 * Real.pi) / 10 := by
    sorry

end phase_shift_of_sine_l182_18207


namespace stone_hitting_ground_time_l182_18237

noncomputable def equation (s : ℝ) : ℝ := -4.5 * s^2 - 12 * s + 48

theorem stone_hitting_ground_time :
  ∃ s : ℝ, equation s = 0 ∧ s = (-8 + 16 * Real.sqrt 7) / 6 :=
by
  sorry

end stone_hitting_ground_time_l182_18237


namespace technology_courses_correct_l182_18248

variable (m : ℕ)

def subject_courses := m
def arts_courses := subject_courses + 9
def technology_courses := 1 / 3 * arts_courses + 5

theorem technology_courses_correct : technology_courses = 1 / 3 * m + 8 := by
  sorry

end technology_courses_correct_l182_18248


namespace additional_track_length_l182_18234

theorem additional_track_length (rise : ℝ) (grade1 grade2 : ℝ) (h1 : grade1 = 0.04) (h2 : grade2 = 0.02) (h3 : rise = 800) :
  ∃ (additional_length : ℝ), additional_length = (rise / grade2 - rise / grade1) ∧ additional_length = 20000 :=
by
  sorry

end additional_track_length_l182_18234


namespace ratio_of_u_to_v_l182_18241

theorem ratio_of_u_to_v (b u v : ℝ) (Hu : u = -b/12) (Hv : v = -b/8) : 
  u / v = 2 / 3 := 
sorry

end ratio_of_u_to_v_l182_18241


namespace train_speed_l182_18204

theorem train_speed (distance time : ℝ) (h1 : distance = 450) (h2 : time = 8) : distance / time = 56.25 := by
  sorry

end train_speed_l182_18204


namespace sum21_exists_l182_18281

theorem sum21_exists (S : Finset ℕ) (h_size : S.card = 11) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 20) :
  ∃ a b, a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ a + b = 21 :=
by
  sorry

end sum21_exists_l182_18281


namespace largest_even_number_in_series_l182_18298

/-- 
  If the sum of 25 consecutive even numbers is 10,000,
  what is the largest number among these 25 consecutive even numbers? 
-/
theorem largest_even_number_in_series (n : ℤ) (S : ℤ) (h : S = 25 * (n - 24)) (h_sum : S = 10000) :
  n = 424 :=
by {
  sorry -- proof goes here
}

end largest_even_number_in_series_l182_18298


namespace closest_integer_to_10_minus_sqrt_12_l182_18258

theorem closest_integer_to_10_minus_sqrt_12 (a b c d : ℤ) (h_a : a = 4) (h_b : b = 5) (h_c : c = 6) (h_d : d = 7) :
  d = 7 :=
by
  sorry

end closest_integer_to_10_minus_sqrt_12_l182_18258


namespace average_of_ABC_l182_18202

theorem average_of_ABC (A B C : ℤ)
  (h1 : 101 * C - 202 * A = 404)
  (h2 : 101 * B + 303 * A = 505)
  (h3 : 101 * A + 101 * B + 101 * C = 303) :
  (A + B + C) / 3 = 3 :=
by
  sorry

end average_of_ABC_l182_18202


namespace keith_missed_games_l182_18227

-- Define the total number of football games
def total_games : ℕ := 8

-- Define the number of games Keith attended
def attended_games : ℕ := 4

-- Define the number of games played at night (although it is not directly necessary for the proof)
def night_games : ℕ := 4

-- Define the number of games Keith missed
def missed_games : ℕ := total_games - attended_games

-- Prove that the number of games Keith missed is 4
theorem keith_missed_games : missed_games = 4 := by
  sorry

end keith_missed_games_l182_18227


namespace projectile_height_time_l182_18250

theorem projectile_height_time (h : ∀ t : ℝ, -16 * t^2 + 100 * t = 64 → t = 1) : (∃ t : ℝ, -16 * t^2 + 100 * t = 64 ∧ t = 1) :=
by sorry

end projectile_height_time_l182_18250


namespace find_a_l182_18278

theorem find_a (a : ℝ) : 3 * a + 150 = 360 → a = 70 := 
by 
  intro h
  sorry

end find_a_l182_18278


namespace regular_price_of_polo_shirt_l182_18273

/--
Zane purchases 2 polo shirts from the 40% off rack at the men's store. 
The polo shirts are priced at a certain amount at the regular price. 
He paid $60 for the shirts. 
Prove that the regular price of each polo shirt is $50.
-/
theorem regular_price_of_polo_shirt (P : ℝ) 
  (h1 : ∀ (x : ℝ), x = 0.6 * P → 2 * x = 60) : 
  P = 50 :=
sorry

end regular_price_of_polo_shirt_l182_18273


namespace area_of_given_trapezium_l182_18209

def area_of_trapezium (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem area_of_given_trapezium :
  area_of_trapezium 20 18 25 = 475 :=
by
  sorry

end area_of_given_trapezium_l182_18209


namespace correct_calculation_l182_18205

theorem correct_calculation (x : ℝ) (h : 63 + x = 69) : 36 / x = 6 :=
by
  sorry

end correct_calculation_l182_18205


namespace complement_A_inter_B_l182_18252

def A : Set ℝ := {x | abs (x - 2) ≤ 2}

def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -1 ≤ x ∧ x ≤ 2}

def A_inter_B : Set ℝ := A ∩ B

def C_R (s : Set ℝ) : Set ℝ := {x | x ∉ s}

theorem complement_A_inter_B :
  C_R A_inter_B = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end complement_A_inter_B_l182_18252


namespace percentage_increase_l182_18236

theorem percentage_increase (x : ℝ) (h1 : 75 + 0.75 * x * 0.8 = 72) : x = 20 :=
by
  sorry

end percentage_increase_l182_18236


namespace total_years_l182_18299

variable (T D : ℕ)
variable (Tom_years : T = 50)
variable (Devin_years : D = 25 - 5)

theorem total_years (hT : T = 50) (hD : D = 25 - 5) : T + D = 70 := by
  sorry

end total_years_l182_18299


namespace smallest_w_value_l182_18244

theorem smallest_w_value (w : ℕ) (hw : w > 0) :
  (∀ k : ℕ, (2^5 ∣ 936 * w) ∧ (3^3 ∣ 936 * w) ∧ (10^2 ∣ 936 * w)) ↔ w = 900 := 
sorry

end smallest_w_value_l182_18244


namespace base_133_not_perfect_square_l182_18265

theorem base_133_not_perfect_square (b : ℤ) : ¬ ∃ k : ℤ, b^2 + 3 * b + 3 = k^2 := by
  sorry

end base_133_not_perfect_square_l182_18265


namespace polynomial_coeff_fraction_eq_neg_122_div_121_l182_18214

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end polynomial_coeff_fraction_eq_neg_122_div_121_l182_18214


namespace sophie_total_spending_l182_18245

-- Definitions based on conditions
def num_cupcakes : ℕ := 5
def price_per_cupcake : ℝ := 2
def num_doughnuts : ℕ := 6
def price_per_doughnut : ℝ := 1
def num_slices_apple_pie : ℕ := 4
def price_per_slice_apple_pie : ℝ := 2
def num_cookies : ℕ := 15
def price_per_cookie : ℝ := 0.60

-- Total cost calculation
def total_cost : ℝ :=
  num_cupcakes * price_per_cupcake +
  num_doughnuts * price_per_doughnut +
  num_slices_apple_pie * price_per_slice_apple_pie +
  num_cookies * price_per_cookie

-- Theorem stating the total cost is 33
theorem sophie_total_spending : total_cost = 33 := by
  sorry

end sophie_total_spending_l182_18245


namespace sum_of_cousins_ages_l182_18261

theorem sum_of_cousins_ages :
  ∃ (a b c d e : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 ∧
    a * b = 36 ∧ c * d = 40 ∧ a + b + c + d + e = 33 :=
by
  sorry

end sum_of_cousins_ages_l182_18261


namespace total_toucans_l182_18211

def initial_toucans : Nat := 2

def new_toucans : Nat := 1

theorem total_toucans : initial_toucans + new_toucans = 3 := by
  sorry

end total_toucans_l182_18211


namespace line_contains_point_l182_18271

theorem line_contains_point {
    k : ℝ
} :
  (2 - k * 3 = -4 * 1) → k = 2 :=
by
  sorry

end line_contains_point_l182_18271


namespace bob_max_candies_l182_18216

theorem bob_max_candies (b : ℕ) (h : b + 2 * b = 30) : b = 10 := 
sorry

end bob_max_candies_l182_18216


namespace range_of_m_l182_18284

-- Definitions of vectors a and b
def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (m, 4)

-- Dot product function for two 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Condition for acute angle
def is_acute (m : ℝ) : Prop := dot_product a (b m) > 0

-- Definition of the range of m
def m_range : Set ℝ := {m | m > -12 ∧ m ≠ 4/3}

-- The theorem to prove
theorem range_of_m (m : ℝ) : is_acute m → m ∈ m_range :=
by
  sorry

end range_of_m_l182_18284


namespace tens_digit_of_23_pow_2057_l182_18289

theorem tens_digit_of_23_pow_2057 : (23^2057 % 100) / 10 % 10 = 6 := 
by
  sorry

end tens_digit_of_23_pow_2057_l182_18289


namespace range_of_a_l182_18219

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < 2 → (a+1)*x > 2*a+2) → a < -1 :=
by
  sorry

end range_of_a_l182_18219


namespace certain_number_eq_neg_thirteen_over_two_l182_18226

noncomputable def CertainNumber (w : ℝ) : ℝ := 13 * w / (1 - w)

theorem certain_number_eq_neg_thirteen_over_two (w : ℝ) (h : w ^ 2 = 1) (hz : 1 - w ≠ 0) :
  CertainNumber w = -13 / 2 :=
sorry

end certain_number_eq_neg_thirteen_over_two_l182_18226


namespace grandpa_rank_l182_18233

theorem grandpa_rank (mom dad grandpa : ℕ) 
  (h1 : mom < dad) 
  (h2 : dad < grandpa) : 
  ∀ rank: ℕ, rank = 3 := 
by
  sorry

end grandpa_rank_l182_18233


namespace daily_harvest_sacks_l182_18210

theorem daily_harvest_sacks (sacks_per_section : ℕ) (num_sections : ℕ) (total_sacks : ℕ) :
  sacks_per_section = 65 → num_sections = 12 → total_sacks = sacks_per_section * num_sections → total_sacks = 780 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end daily_harvest_sacks_l182_18210


namespace base8_subtraction_l182_18285

theorem base8_subtraction : (53 - 26 : ℕ) = 25 :=
by sorry

end base8_subtraction_l182_18285


namespace book_count_l182_18294

theorem book_count (P C B : ℕ) (h1 : P = 3 * C / 2) (h2 : B = 3 * C / 4) (h3 : P + C + B > 3000) : 
  P + C + B = 3003 := by
  sorry

end book_count_l182_18294


namespace number_of_four_digit_numbers_with_two_identical_digits_l182_18255

-- Define the conditions
def starts_with_nine (n : ℕ) : Prop := n / 1000 = 9
def has_exactly_two_identical_digits (n : ℕ) : Prop := 
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d2) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d2 ∧ (n % 100) / 10 = d1 ∧ n % 10 = d1) ∨
  (∃ d1 d2, d1 ≠ d2 ∧ (n % 1000) / 100 = d1 ∧ (n % 100) / 10 = d2 ∧ n % 10 = d1)

-- Define the proof problem
theorem number_of_four_digit_numbers_with_two_identical_digits : 
  ∃ n, starts_with_nine n ∧ has_exactly_two_identical_digits n ∧ n = 432 := 
sorry

end number_of_four_digit_numbers_with_two_identical_digits_l182_18255


namespace circle_area_x2_y2_eq_102_l182_18229

theorem circle_area_x2_y2_eq_102 :
  ∀ (x y : ℝ), (x + 9)^2 + (y - 3)^2 = 102 → π * 102 = 102 * π :=
by
  intros
  sorry

end circle_area_x2_y2_eq_102_l182_18229


namespace find_a_l182_18228

def star (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : star a 5 = 9) : a = 17 := by
  sorry

end find_a_l182_18228


namespace increased_expenses_percent_l182_18286

theorem increased_expenses_percent (S : ℝ) (hS : S = 6250) (initial_save_percent : ℝ) (final_savings : ℝ) 
  (initial_save_percent_def : initial_save_percent = 20) 
  (final_savings_def : final_savings = 250) : 
  (initial_save_percent / 100 * S - final_savings) / (S - initial_save_percent / 100 * S) * 100 = 20 := by
  sorry

end increased_expenses_percent_l182_18286


namespace xiaoming_accuracy_l182_18296

theorem xiaoming_accuracy :
  ∀ (correct already_wrong extra_needed : ℕ),
  correct = 30 →
  already_wrong = 6 →
  (correct + extra_needed).toFloat / (correct + already_wrong + extra_needed).toFloat = 0.85 →
  extra_needed = 4 := by
  intros correct already_wrong extra_needed h_correct h_wrong h_accuracy
  sorry

end xiaoming_accuracy_l182_18296


namespace john_profit_l182_18242

theorem john_profit (cost price : ℕ) (n : ℕ) (h1 : cost = 4) (h2 : price = 8) (h3 : n = 30) : 
  n * (price - cost) = 120 :=
by
  -- The proof goes here
  sorry

end john_profit_l182_18242


namespace problem_solution_l182_18254

-- Definitions for the digits and arithmetic conditions
def is_digit (n : ℕ) : Prop := n < 10

-- Problem conditions stated in Lean
variables (A B C D E : ℕ)

-- Define the conditions
axiom digits_A : is_digit A
axiom digits_B : is_digit B
axiom digits_C : is_digit C
axiom digits_D : is_digit D
axiom digits_E : is_digit E

-- Subtraction result for second equation
axiom sub_eq : A - C = A

-- Additional conditions derived from the problem
axiom add_eq : (E + E = D)

-- Now, state the problem in Lean
theorem problem_solution : D = 8 :=
sorry

end problem_solution_l182_18254


namespace cost_price_equals_selling_price_l182_18280

theorem cost_price_equals_selling_price (C : ℝ) (x : ℝ) (hp : C > 0) (profit : ℝ := 0.25) (h : 30 * C = (1 + profit) * C * x) : x = 24 :=
by
  sorry

end cost_price_equals_selling_price_l182_18280


namespace problem_1_problem_2_l182_18221

def f (a : ℝ) (x : ℝ) : ℝ := abs (a * x + 1)

def g (a : ℝ) (x : ℝ) : ℝ := f a x - abs (x + 1)

theorem problem_1 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 ↔ f a x ≤ 3) → a = 2 := by
  intro h
  sorry

theorem problem_2 (a : ℝ) : a = 2 → (∃ x : ℝ, ∀ y : ℝ, g a y ≥ g a x ∧ g a x = -1/2) := by
  intro ha2
  use -1/2
  sorry

end problem_1_problem_2_l182_18221


namespace number_less_than_one_is_correct_l182_18215

theorem number_less_than_one_is_correct : (1 - 5 = -4) :=
by
  sorry

end number_less_than_one_is_correct_l182_18215


namespace Q_mul_P_plus_Q_eq_one_l182_18201

noncomputable def sqrt5_plus_2_pow (n : ℕ) :=
  (Real.sqrt 5 + 2)^(2 * n + 1)

noncomputable def P (n : ℕ) :=
  Int.floor (sqrt5_plus_2_pow n)

noncomputable def Q (n : ℕ) :=
  sqrt5_plus_2_pow n - P n

theorem Q_mul_P_plus_Q_eq_one (n : ℕ) : Q n * (P n + Q n) = 1 := by
  sorry

end Q_mul_P_plus_Q_eq_one_l182_18201


namespace max_disks_l182_18235

theorem max_disks (n k : ℕ) (h1: n ≥ 1) (h2: k ≥ 1) :
  (∃ (d : ℕ), d = if n > 1 ∧ k > 1 then 2 * (n + k) - 4 else max n k) ∧
  (∀ (p q : ℕ), (p <= n → q <= k → ¬∃ (x y : ℕ), x + 1 = y ∨ x - 1 = y ∨ x + 1 = p ∨ x - 1 = p)) :=
sorry

end max_disks_l182_18235


namespace average_age_of_cricket_team_l182_18256

theorem average_age_of_cricket_team
  (A : ℝ)
  (captain_age : ℝ) (wicket_keeper_age : ℝ)
  (team_size : ℕ) (remaining_players : ℕ)
  (captain_age_eq : captain_age = 24)
  (wicket_keeper_age_eq : wicket_keeper_age = 27)
  (remaining_players_eq : remaining_players = team_size - 2)
  (average_age_condition : (team_size * A - (captain_age + wicket_keeper_age)) = remaining_players * (A - 1)) : 
  A = 21 := by
  sorry

end average_age_of_cricket_team_l182_18256


namespace incorrect_eqn_x9_y9_neg1_l182_18243

theorem incorrect_eqn_x9_y9_neg1 (x y : ℂ) 
  (hx : x = (-1 + Complex.I * Real.sqrt 3) / 2) 
  (hy : y = (-1 - Complex.I * Real.sqrt 3) / 2) : 
  x^9 + y^9 ≠ -1 :=
sorry

end incorrect_eqn_x9_y9_neg1_l182_18243


namespace fourth_rectangle_area_is_112_l182_18275

def area_of_fourth_rectangle (length : ℕ) (width : ℕ) (area1 : ℕ) (area2 : ℕ) (area3 : ℕ) : ℕ :=
  length * width - area1 - area2 - area3

theorem fourth_rectangle_area_is_112 :
  area_of_fourth_rectangle 20 12 24 48 36 = 112 :=
by
  sorry

end fourth_rectangle_area_is_112_l182_18275


namespace points_on_opposite_sides_l182_18203

theorem points_on_opposite_sides (a : ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
    (hA : A = (a, 1)) 
    (hB : B = (2, a)) 
    (opposite_sides : A.1 < 0 ∧ B.1 > 0 ∨ A.1 > 0 ∧ B.1 < 0) 
    : a < 0 := 
  sorry

end points_on_opposite_sides_l182_18203


namespace special_sale_day_price_l182_18222

-- Define the original price
def original_price : ℝ := 250

-- Define the first discount rate
def first_discount_rate : ℝ := 0.40

-- Calculate the price after the first discount
def price_after_first_discount (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

-- Define the second discount rate (special sale day)
def second_discount_rate : ℝ := 0.10

-- Calculate the price after the second discount
def price_after_second_discount (discounted_price : ℝ) (discount_rate : ℝ) : ℝ :=
  discounted_price * (1 - discount_rate)

-- Theorem statement
theorem special_sale_day_price :
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 := by
  sorry

end special_sale_day_price_l182_18222


namespace compound_interest_rate_l182_18246

theorem compound_interest_rate :
  ∀ (P A : ℝ) (t n : ℕ) (r : ℝ),
  P = 12000 →
  A = 21500 →
  t = 5 →
  n = 1 →
  A = P * (1 + r / n) ^ (n * t) →
  r = 0.121898 :=
by
  intros P A t n r hP hA ht hn hCompound
  sorry

end compound_interest_rate_l182_18246


namespace nancy_spelling_problems_l182_18264

structure NancyProblems where
  math_problems : ℝ
  rate : ℝ
  hours : ℝ
  total_problems : ℝ

noncomputable def calculate_spelling_problems (n : NancyProblems) : ℝ :=
  n.total_problems - n.math_problems

theorem nancy_spelling_problems :
  ∀ (n : NancyProblems), n.math_problems = 17.0 ∧ n.rate = 8.0 ∧ n.hours = 4.0 ∧ n.total_problems = 32.0 →
  calculate_spelling_problems n = 15.0 :=
by
  intros
  sorry

end nancy_spelling_problems_l182_18264


namespace no_nat_p_prime_and_p6_plus_6_prime_l182_18200

theorem no_nat_p_prime_and_p6_plus_6_prime (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (p^6 + 6)) : False := 
sorry

end no_nat_p_prime_and_p6_plus_6_prime_l182_18200


namespace final_alcohol_percentage_l182_18297

noncomputable def initial_volume : ℝ := 6
noncomputable def initial_percentage : ℝ := 0.25
noncomputable def added_alcohol : ℝ := 3
noncomputable def final_volume : ℝ := initial_volume + added_alcohol
noncomputable def final_percentage : ℝ := (initial_volume * initial_percentage + added_alcohol) / final_volume * 100

theorem final_alcohol_percentage :
  final_percentage = 50 := by
  sorry

end final_alcohol_percentage_l182_18297


namespace poly_coefficients_sum_l182_18238

theorem poly_coefficients_sum :
  ∀ (x A B C D : ℝ),
  (x - 3) * (4 * x^2 + 2 * x - 7) = A * x^3 + B * x^2 + C * x + D →
  A + B + C + D = 2 :=
by sorry

end poly_coefficients_sum_l182_18238


namespace geometric_series_sum_l182_18225

theorem geometric_series_sum 
  (a : ℝ) (r : ℝ) (s : ℝ)
  (h_a : a = 9)
  (h_r : r = -2/3)
  (h_abs_r : |r| < 1)
  (h_s : s = a / (1 - r)) : 
  s = 5.4 := by
  sorry

end geometric_series_sum_l182_18225


namespace proof_problem_l182_18263

noncomputable def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 3 else 3 * n

noncomputable def sequence_b (n : ℕ) : ℕ :=
  3 ^ n

noncomputable def sequence_c (n : ℕ) : ℕ :=
  sequence_b (sequence_a n)

theorem proof_problem :
  sequence_c 2017 = 27 ^ 2017 :=
by sorry

end proof_problem_l182_18263


namespace max_fraction_value_l182_18230

theorem max_fraction_value :
  ∀ (x y : ℝ), (1/4 ≤ x ∧ x ≤ 3/5) ∧ (1/5 ≤ y ∧ y ≤ 1/2) → 
    xy / (x^2 + y^2) ≤ 2/5 :=
by
  sorry

end max_fraction_value_l182_18230


namespace math_problem_solution_l182_18274

noncomputable def a_range : Set ℝ := {a : ℝ | (0 < a ∧ a ≤ 1) ∨ (5 ≤ a ∧ a < 6)}

theorem math_problem_solution (a : ℝ) :
  (1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∨ ((a - 3)^2 - 4 < 0)
  ∧ ¬((1 - 4 * (a^2 - 6 * a) > 0 ∧ a^2 - 6 * a < 0) ∧ ((a - 3)^2 - 4 < 0)) →
  a ∈ a_range :=
sorry

end math_problem_solution_l182_18274


namespace spadesuit_eval_l182_18257

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem spadesuit_eval :
  spadesuit 5 (spadesuit 6 3) = -704 := by
  sorry

end spadesuit_eval_l182_18257


namespace cole_drive_time_to_work_l182_18288

theorem cole_drive_time_to_work :
  ∀ (D : ℝ),
    (D / 80 + D / 120 = 3) → (D / 80 * 60 = 108) :=
by
  intro D h
  sorry

end cole_drive_time_to_work_l182_18288


namespace dice_probability_l182_18260

theorem dice_probability :
  let prob_roll_less_than_four := 3 / 6
  let prob_roll_even := 3 / 6
  let prob_roll_greater_than_four := 2 / 6
  prob_roll_less_than_four * prob_roll_even * prob_roll_greater_than_four = 1 / 12 :=
by
  sorry

end dice_probability_l182_18260


namespace max_M_l182_18279

noncomputable def conditions (x y z u : ℝ) : Prop :=
  (x - 2 * y = z - 2 * u) ∧ (2 * y * z = u * x) ∧ (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < u) ∧ (z ≥ y)

theorem max_M (x y z u : ℝ) : conditions x y z u → ∃ M : ℝ, M = 6 + 4 * Real.sqrt 2 ∧ M ≤ z / y :=
by {
  sorry
}

end max_M_l182_18279


namespace polynomial_divisible_by_square_l182_18253

def f (x : ℝ) (a1 a2 a3 a4 : ℝ) : ℝ := x^4 + a1 * x^3 + a2 * x^2 + a3 * x + a4
def f' (x : ℝ) (a1 a2 a3 : ℝ) : ℝ := 4 * x^3 + 3 * a1 * x^2 + 2 * a2 * x + a3

theorem polynomial_divisible_by_square (x0 a1 a2 a3 a4 : ℝ) 
  (h1 : f x0 a1 a2 a3 a4 = 0) 
  (h2 : f' x0 a1 a2 a3 = 0) : 
  ∃ g : ℝ → ℝ, ∀ x : ℝ, f x a1 a2 a3 a4 = (x - x0)^2 * (g x) :=
sorry

end polynomial_divisible_by_square_l182_18253


namespace pencils_per_box_l182_18240

theorem pencils_per_box:
  ∀ (red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes : ℕ),
  red_pencils = 20 →
  blue_pencils = 2 * red_pencils →
  yellow_pencils = 40 →
  green_pencils = red_pencils + blue_pencils →
  total_pencils = red_pencils + blue_pencils + yellow_pencils + green_pencils →
  num_boxes = 8 →
  total_pencils / num_boxes = 20 :=
by
  intros red_pencils blue_pencils yellow_pencils green_pencils total_pencils num_boxes
  intros h1 h2 h3 h4 h5 h6
  sorry

end pencils_per_box_l182_18240


namespace find_z_l182_18224

noncomputable def w : ℝ := sorry
noncomputable def x : ℝ := (5 * w) / 4
noncomputable def y : ℝ := 1.40 * w

theorem find_z (z : ℝ) : x = (1 - z / 100) * y → z = 10.71 :=
by
  sorry

end find_z_l182_18224


namespace triangle_inequality_l182_18259

theorem triangle_inequality (a b c : ℝ) (S : ℝ) (hS : S = (1/4) * Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S :=
sorry

end triangle_inequality_l182_18259


namespace find_t_l182_18268

-- Define the elements and the conditions
def vector_a : ℝ × ℝ := (1, -1)
def vector_b (t : ℝ) : ℝ × ℝ := (t, 1)

def add_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def sub_vectors (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

def parallel (u v : ℝ × ℝ) : Prop := u.1 * v.2 = u.2 * v.1

-- Lean statement of the problem
theorem find_t (t : ℝ) : 
  parallel (add_vectors vector_a (vector_b t)) (sub_vectors vector_a (vector_b t)) → t = -1 :=
by
  sorry

end find_t_l182_18268


namespace fraction_historical_fiction_new_releases_l182_18292

-- Define constants for book categories and new releases
def historical_fiction_percentage : ℝ := 0.40
def science_fiction_percentage : ℝ := 0.25
def biographies_percentage : ℝ := 0.15
def mystery_novels_percentage : ℝ := 0.20

def historical_fiction_new_releases : ℝ := 0.45
def science_fiction_new_releases : ℝ := 0.30
def biographies_new_releases : ℝ := 0.50
def mystery_novels_new_releases : ℝ := 0.35

-- Statement of the problem to prove
theorem fraction_historical_fiction_new_releases :
  (historical_fiction_percentage * historical_fiction_new_releases) /
    (historical_fiction_percentage * historical_fiction_new_releases +
     science_fiction_percentage * science_fiction_new_releases +
     biographies_percentage * biographies_new_releases +
     mystery_novels_percentage * mystery_novels_new_releases) = 9 / 20 :=
by
  sorry

end fraction_historical_fiction_new_releases_l182_18292


namespace complement_set_A_in_U_l182_18218

-- Given conditions
def U : Set ℤ := {-1, 0, 1, 2}
def A : Set ℤ := {x | x ∈ U ∧ x^2 < 1}

-- Theorem to prove complement
theorem complement_set_A_in_U :
  U \ A = {-1, 1, 2} :=
by
  sorry

end complement_set_A_in_U_l182_18218


namespace rectangle_area_l182_18269

variable (L B : ℕ)

theorem rectangle_area :
  (L - B = 23) ∧ (2 * L + 2 * B = 166) → (L * B = 1590) :=
by
  sorry

end rectangle_area_l182_18269


namespace candy_distribution_l182_18283

theorem candy_distribution (n : Nat) : ∃ k : Nat, n = 2 ^ k :=
sorry

end candy_distribution_l182_18283


namespace sum_of_consecutive_integers_is_33_l182_18220

theorem sum_of_consecutive_integers_is_33 :
  ∃ (x : ℕ), x * (x + 1) = 272 ∧ x + (x + 1) = 33 :=
by
  sorry

end sum_of_consecutive_integers_is_33_l182_18220


namespace blocks_remaining_l182_18217

def initial_blocks : ℕ := 55
def blocks_eaten : ℕ := 29

theorem blocks_remaining : initial_blocks - blocks_eaten = 26 := by
  sorry

end blocks_remaining_l182_18217


namespace ratio_size12_to_size6_l182_18251

-- Definitions based on conditions
def cheerleaders_size2 : ℕ := 4
def cheerleaders_size6 : ℕ := 10
def total_cheerleaders : ℕ := 19
def cheerleaders_size12 : ℕ := total_cheerleaders - (cheerleaders_size2 + cheerleaders_size6)

-- Proof statement
theorem ratio_size12_to_size6 : cheerleaders_size12.toFloat / cheerleaders_size6.toFloat = 1 / 2 := sorry

end ratio_size12_to_size6_l182_18251


namespace max_dist_2_minus_2i_l182_18232

open Complex

noncomputable def max_dist (z1 : ℂ) : ℝ :=
  Complex.abs 1 + Complex.abs z1

theorem max_dist_2_minus_2i :
  max_dist (2 - 2*I) = 1 + 2 * Real.sqrt 2 := by
  sorry

end max_dist_2_minus_2i_l182_18232


namespace part1_part2_l182_18266

open Set

/-- Define sets A and B as per given conditions --/
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

/-- Part 1: Prove the intersection and union with complements --/
theorem part1 :
  A ∩ B = {x | 3 ≤ x ∧ x < 6} ∧ (compl B) ∪ A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by {
  sorry
}

/-- Part 2: Given C ⊆ B, prove the constraints on a --/
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem part2 (a : ℝ) (h : C a ⊆ B) : 2 ≤ a ∧ a ≤ 8 :=
by {
  sorry
}

end part1_part2_l182_18266


namespace post_spacing_change_l182_18276

theorem post_spacing_change :
  ∀ (posts : ℕ → ℝ) (constant_spacing : ℝ), 
  (∀ n, 1 ≤ n ∧ n < 16 → posts (n + 1) - posts n = constant_spacing) →
  posts 16 - posts 1 = 48 → 
  posts 28 - posts 16 = 36 →
  ∃ (k : ℕ), 16 < k ∧ k ≤ 28 ∧ posts (k + 1) - posts k ≠ constant_spacing ∧ posts (k + 1) - posts k = 2.9 ∧ k = 20 := 
  sorry

end post_spacing_change_l182_18276


namespace solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l182_18206

-- Problem 1
theorem solve_quadratic_1 (x : ℝ) : (x - 1) ^ 2 - 4 = 0 ↔ (x = -1 ∨ x = 3) :=
by
  sorry

-- Problem 2
theorem solve_quadratic_2 (x : ℝ) : (2 * x - 1) * (x + 3) = 4 ↔ (x = -7 / 2 ∨ x = 1) :=
by
  sorry

-- Problem 3
theorem solve_quadratic_3 (x : ℝ) : 2 * x ^ 2 - 5 * x + 2 = 0 ↔ (x = 2 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_1_solve_quadratic_2_solve_quadratic_3_l182_18206


namespace smallest_area_of_2020th_square_l182_18239

theorem smallest_area_of_2020th_square :
  ∃ (S : ℤ) (A : ℕ), 
    (S * S - 2019 = A) ∧ 
    (∃ k : ℕ, k * k = A) ∧ 
    (∀ (T : ℤ) (B : ℕ), ((T * T - 2019 = B) ∧ (∃ l : ℕ, l * l = B)) → (A ≤ B)) :=
sorry

end smallest_area_of_2020th_square_l182_18239


namespace line_does_not_pass_second_quadrant_l182_18267

-- Definitions of conditions
variables (k b x y : ℝ)
variable  (h₁ : k > 0) -- condition k > 0
variable  (h₂ : b < 0) -- condition b < 0


theorem line_does_not_pass_second_quadrant : 
  ¬∃ (x y : ℝ), (x < 0 ∧ y > 0) ∧ (y = k * x + b) :=
sorry

end line_does_not_pass_second_quadrant_l182_18267


namespace area_of_common_part_geq_3484_l182_18213

theorem area_of_common_part_geq_3484 :
  ∀ (R : ℝ) (S T : ℝ → Prop), 
  (R = 1) →
  (∀ x y, S x ↔ (x * x + y * y = R * R) ∧ T y) →
  ∃ (S_common : ℝ) (T_common : ℝ),
    (S_common + T_common > 3.484) :=
by
  sorry

end area_of_common_part_geq_3484_l182_18213


namespace correct_average_l182_18249

theorem correct_average (n : ℕ) (wrong_avg : ℕ) (wrong_num correct_num : ℕ) (correct_avg : ℕ)
  (h1 : n = 10) 
  (h2 : wrong_avg = 21)
  (h3 : wrong_num = 26)
  (h4 : correct_num = 36)
  (h5 : correct_avg = 22) :
  (wrong_avg * n + (correct_num - wrong_num)) / n = correct_avg :=
by
  sorry

end correct_average_l182_18249


namespace solution_correctness_l182_18282

def is_prime (n : ℕ) : Prop := Nat.Prime n

def problem_statement (a b c : ℕ) : Prop :=
  (a * b * c = 56) ∧
  (a * b + b * c + a * c = 311) ∧
  is_prime a ∧ is_prime b ∧ is_prime c

theorem solution_correctness (a b c : ℕ) (h : problem_statement a b c) :
  a = 2 ∨ a = 13 ∨ a = 19 ∧
  b = 2 ∨ b = 13 ∨ b = 19 ∧
  c = 2 ∨ c = 13 ∨ c = 19 :=
by
  sorry

end solution_correctness_l182_18282


namespace polynomial_coeff_diff_l182_18293

theorem polynomial_coeff_diff (a b c d e f : ℝ) :
  ((3*x + 1)^5 = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) →
  (a - b + c - d + e - f = 32) :=
by
  sorry

end polynomial_coeff_diff_l182_18293
