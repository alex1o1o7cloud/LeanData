import Mathlib

namespace replace_star_l1962_196255

theorem replace_star (x : ℕ) : 2 * 18 * 14 = 6 * x * 7 → x = 12 :=
sorry

end replace_star_l1962_196255


namespace Isabella_hair_length_l1962_196278

-- Define the conditions using variables
variables (h_current h_cut_off h_initial : ℕ)

-- The proof problem statement
theorem Isabella_hair_length :
  h_current = 9 → h_cut_off = 9 → h_initial = h_current + h_cut_off → h_initial = 18 :=
by
  intros hc hc' hi
  rw [hc, hc'] at hi
  exact hi


end Isabella_hair_length_l1962_196278


namespace find_a_l1962_196290

-- Define the variables
variables (m d a b : ℝ)

-- State the main theorem with conditions
theorem find_a (h : m = d * a * b / (a - b)) (h_ne : m ≠ d * b) : a = m * b / (m - d * b) :=
sorry

end find_a_l1962_196290


namespace range_of_y_under_conditions_l1962_196247

theorem range_of_y_under_conditions :
  (∀ x : ℝ, (x - y) * (x + y) < 1) → (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
  intro h
  have h' : ∀ x : ℝ, (x - y) * (1 - x - y) < 1 := by
    sorry
  have g_min : ∀ x : ℝ, y^2 - y < x^2 - x + 1 := by
    sorry
  have min_value : y^2 - y < 3/4 := by
    sorry
  have range_y : (-1/2 : ℝ) < y ∧ y < (3/2 : ℝ) := by
    sorry
  exact range_y

end range_of_y_under_conditions_l1962_196247


namespace find_correct_speed_l1962_196229

-- Definitions for given conditions
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Given conditions as definitions
def condition1 (d t : ℝ) : Prop := distance_traveled 35 (t + (5 / 60)) = d
def condition2 (d t : ℝ) : Prop := distance_traveled 55 (t - (5 / 60)) = d

-- Statement to prove
theorem find_correct_speed (d t r : ℝ) (h1 : condition1 d t) (h2 : condition2 d t) :
  r = (d / t) ∧ r = 42.78 :=
by sorry

end find_correct_speed_l1962_196229


namespace determine_n_l1962_196277

theorem determine_n (n : ℕ) (hn : 0 < n) :
  (∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 3 * x + 2 * y + z = n) → (n = 15 ∨ n = 16) :=
  by sorry

end determine_n_l1962_196277


namespace eval_expression_at_neg_one_l1962_196252

variable (x : ℤ)

theorem eval_expression_at_neg_one : x = -1 → 3 * x ^ 2 + 2 * x - 1 = 0 := by
  intro h
  rw [h]
  show 3 * (-1) ^ 2 + 2 * (-1) - 1 = 0
  sorry

end eval_expression_at_neg_one_l1962_196252


namespace calculate_f_at_2_l1962_196257

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

theorem calculate_f_at_2
  (a b : ℝ)
  (h_extremum : 3 + 2 * a + b = 0)
  (h_f1 : f 1 a b = 10) :
  f 2 a b = 18 :=
sorry

end calculate_f_at_2_l1962_196257


namespace rotated_square_vertical_distance_is_correct_l1962_196254

-- Define a setup with four 1-inch squares in a straight line
-- and the second square rotated 45 degrees around its center

-- Noncomputable setup
noncomputable def rotated_square_vert_distance : ℝ :=
  let side_length := 1
  let diagonal := side_length * Real.sqrt 2
  -- Calculate the required vertical distance according to given conditions
  Real.sqrt 2 + side_length / 2

-- Theorem statement confirming the calculated vertical distance
theorem rotated_square_vertical_distance_is_correct :
  rotated_square_vert_distance = Real.sqrt 2 + 1 / 2 :=
by
  sorry

end rotated_square_vertical_distance_is_correct_l1962_196254


namespace find_f2_l1962_196231

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1)
  (H2 : f 8 = 15) :
  f 2 = 3 := 
sorry

end find_f2_l1962_196231


namespace fraction_of_foreign_males_l1962_196262

theorem fraction_of_foreign_males
  (total_students : ℕ)
  (female_ratio : ℚ)
  (non_foreign_males : ℕ)
  (foreign_male_fraction : ℚ)
  (h1 : total_students = 300)
  (h2 : female_ratio = 2/3)
  (h3 : non_foreign_males = 90) :
  foreign_male_fraction = 1/10 :=
by
  sorry

end fraction_of_foreign_males_l1962_196262


namespace find_ratio_l1962_196207

theorem find_ratio (x y c d : ℝ) (h₁ : 4 * x - 2 * y = c) (h₂ : 5 * y - 10 * x = d) (h₃ : d ≠ 0) : c / d = 0 :=
sorry

end find_ratio_l1962_196207


namespace intersection_A_B_l1962_196217

def setA : Set ℝ := {x | 0 < x}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}
def intersectionAB : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = intersectionAB := by
  sorry

end intersection_A_B_l1962_196217


namespace no_valid_prime_angles_l1962_196218

def is_prime (n : ℕ) : Prop := Prime n

theorem no_valid_prime_angles :
  ∀ (x : ℕ), (x < 30) ∧ is_prime x ∧ is_prime (3 * x) → False :=
by sorry

end no_valid_prime_angles_l1962_196218


namespace solution_set_of_inequality_l1962_196240

variable {f : ℝ → ℝ}

theorem solution_set_of_inequality (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_deriv_neg : ∀ x : ℝ, 0 < x → (x^2 + 1) * deriv f x + 2 * x * f x < 0)
  (h_f_neg1_zero : f (-1) = 0) :
  { x : ℝ | f x > 0 } = { x | x < -1 } ∪ { x | 0 < x ∧ x < 1 } := by
  sorry

end solution_set_of_inequality_l1962_196240


namespace xyz_inequality_l1962_196215

theorem xyz_inequality (x y z : ℝ) (h : x^2 + y^2 + z^2 = 2) : x + y + z ≤ x * y * z + 2 := by
  sorry

end xyz_inequality_l1962_196215


namespace reams_for_haley_correct_l1962_196242

-- Definitions: 
-- total reams = 5
-- reams for sister = 3
-- reams for Haley = ?

def total_reams : Nat := 5
def reams_for_sister : Nat := 3
def reams_for_haley : Nat := total_reams - reams_for_sister

-- The proof problem: prove reams_for_haley = 2 given the conditions.
theorem reams_for_haley_correct : reams_for_haley = 2 := by 
  sorry

end reams_for_haley_correct_l1962_196242


namespace coordinates_of_M_l1962_196245

theorem coordinates_of_M :
  -- Given the function f(x) = 2x^2 + 1
  let f : Real → Real := λ x => 2 * x^2 + 1
  -- And its derivative
  let f' : Real → Real := λ x => 4 * x
  -- The coordinates of point M where the instantaneous rate of change is -8 are (-2, 9)
  (∃ x0 : Real, f' x0 = -8 ∧ f x0 = y0 ∧ x0 = -2 ∧ y0 = 9) := by
    sorry

end coordinates_of_M_l1962_196245


namespace sum_of_distances_l1962_196219

theorem sum_of_distances (d_1 d_2 : ℝ) (h1 : d_1 = 1 / 9 * d_2) (h2 : d_1 + d_2 = 6) : d_1 + d_2 + 6 = 20 :=
by
  sorry

end sum_of_distances_l1962_196219


namespace log_40_cannot_be_directly_calculated_l1962_196282

theorem log_40_cannot_be_directly_calculated (log_3 log_5 : ℝ) (h1 : log_3 = 0.4771) (h2 : log_5 = 0.6990) : 
  ¬ (exists (log_40 : ℝ), (log_40 = (log_3 + log_5) + log_40)) :=
by {
  sorry
}

end log_40_cannot_be_directly_calculated_l1962_196282


namespace schools_participation_l1962_196256

-- Definition of the problem conditions
def school_teams : ℕ := 3

-- Paula's rank p must satisfy this
def total_participants (p : ℕ) : ℕ := 2 * p - 1

-- Predicate indicating the number of participants condition:
def participants_condition (p : ℕ) : Prop := total_participants p ≥ 75

-- Translation of number of participants to number of schools
def number_of_schools (n : ℕ) : ℕ := 3 * n

-- The statement to prove:
theorem schools_participation : ∃ (n p : ℕ), participants_condition p ∧ p = 38 ∧ number_of_schools n = total_participants p ∧ n = 25 := 
by 
  sorry

end schools_participation_l1962_196256


namespace seventh_fifth_tiles_difference_l1962_196246

def side_length (n : ℕ) : ℕ := 2 * n - 1
def number_of_tiles (n : ℕ) : ℕ := (side_length n) ^ 2
def tiles_difference (n m : ℕ) : ℕ := number_of_tiles n - number_of_tiles m

theorem seventh_fifth_tiles_difference : tiles_difference 7 5 = 88 := by
  sorry

end seventh_fifth_tiles_difference_l1962_196246


namespace tetrahedron_volume_is_zero_l1962_196236

noncomputable def volume_of_tetrahedron (p q r : ℝ) : ℝ :=
  (1 / 6) * p * q * r

theorem tetrahedron_volume_is_zero (p q r : ℝ)
  (hpq : p^2 + q^2 = 36)
  (hqr : q^2 + r^2 = 64)
  (hrp : r^2 + p^2 = 100) :
  volume_of_tetrahedron p q r = 0 := by
  sorry

end tetrahedron_volume_is_zero_l1962_196236


namespace ratio_diamond_brace_ring_l1962_196213

theorem ratio_diamond_brace_ring
  (cost_ring : ℤ) (cost_car : ℤ) (total_worth : ℤ) (cost_diamond_brace : ℤ)
  (h1 : cost_ring = 4000) (h2 : cost_car = 2000) (h3 : total_worth = 14000)
  (h4 : cost_diamond_brace = total_worth - (cost_ring + cost_car)) :
  cost_diamond_brace / cost_ring = 2 :=
by
  sorry

end ratio_diamond_brace_ring_l1962_196213


namespace arccos_neg_one_eq_pi_l1962_196266

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi := by
  sorry

end arccos_neg_one_eq_pi_l1962_196266


namespace arith_seq_sum_first_four_terms_l1962_196295

noncomputable def sum_first_four_terms_arith_seq (a1 : ℤ) (d : ℤ) : ℤ :=
  4 * a1 + 6 * d

theorem arith_seq_sum_first_four_terms (a1 a3 : ℤ) 
  (h1 : a3 = a1 + 2 * 3)
  (h2 : a1 + a3 = 8) 
  (d : ℤ := 3) :
  sum_first_four_terms_arith_seq a1 d = 22 := by
  unfold sum_first_four_terms_arith_seq
  sorry

end arith_seq_sum_first_four_terms_l1962_196295


namespace find_k_l1962_196269

theorem find_k : ∃ k : ℝ, (3 * k - 4) / (k + 7) = 2 / 5 ∧ k = 34 / 13 :=
by
  use 34 / 13
  sorry

end find_k_l1962_196269


namespace sequence_value_at_20_l1962_196275

open Nat

def arithmetic_sequence (a : ℕ → ℤ) : Prop := 
  a 1 = 1 ∧ ∀ n, a (n + 1) - a n = 4

theorem sequence_value_at_20 (a : ℕ → ℤ) (h : arithmetic_sequence a) : a 20 = 77 :=
sorry

end sequence_value_at_20_l1962_196275


namespace laura_owes_amount_l1962_196292

noncomputable def calculate_amount_owed (P R T : ℝ) : ℝ :=
  let I := P * R * T 
  P + I

theorem laura_owes_amount (P : ℝ) (R : ℝ) (T : ℝ) (hP : P = 35) (hR : R = 0.09) (hT : T = 1) :
  calculate_amount_owed P R T = 38.15 := by
  -- Prove that the total amount owed calculated by the formula matches the correct answer
  sorry

end laura_owes_amount_l1962_196292


namespace Brian_traveled_60_miles_l1962_196285

theorem Brian_traveled_60_miles (mpg gallons : ℕ) (hmpg : mpg = 20) (hgallons : gallons = 3) :
    mpg * gallons = 60 := by
  sorry

end Brian_traveled_60_miles_l1962_196285


namespace boxes_difference_l1962_196216

theorem boxes_difference (white_balls red_balls balls_per_box : ℕ)
  (h_white : white_balls = 30)
  (h_red : red_balls = 18)
  (h_box : balls_per_box = 6) :
  (white_balls / balls_per_box) - (red_balls / balls_per_box) = 2 :=
by 
  sorry

end boxes_difference_l1962_196216


namespace scientific_notation_932700_l1962_196200

theorem scientific_notation_932700 : 932700 = 9.327 * 10^5 :=
sorry

end scientific_notation_932700_l1962_196200


namespace ratio_of_areas_of_triangle_and_trapezoid_l1962_196228

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

theorem ratio_of_areas_of_triangle_and_trapezoid :
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  (a_small / a_trapezoid) = (1 / 3) :=
by
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  have h : (a_small / a_trapezoid) = (1 / 3) := 
    by sorry  -- Here would be the proof steps, but we're skipping
  exact h

end ratio_of_areas_of_triangle_and_trapezoid_l1962_196228


namespace find_sum_of_squares_l1962_196249

theorem find_sum_of_squares (x y z : ℝ)
  (h1 : x^2 + 3 * y = 8)
  (h2 : y^2 + 5 * z = -9)
  (h3 : z^2 + 7 * x = -16) : x^2 + y^2 + z^2 = 20.75 :=
sorry

end find_sum_of_squares_l1962_196249


namespace graph_of_equation_l1962_196281

theorem graph_of_equation (x y : ℝ) : (x - y)^2 = x^2 + y^2 ↔ (x = 0 ∨ y = 0) := by
  sorry

end graph_of_equation_l1962_196281


namespace derivative_at_one_l1962_196225

noncomputable def f (x : ℝ) : ℝ := 1 / x

theorem derivative_at_one : deriv f 1 = -1 := sorry

end derivative_at_one_l1962_196225


namespace employees_excluding_manager_l1962_196235

theorem employees_excluding_manager (average_salary average_increase manager_salary n : ℕ)
  (h_avg_salary : average_salary = 2400)
  (h_avg_increase : average_increase = 100)
  (h_manager_salary : manager_salary = 4900)
  (h_new_avg_salary : average_salary + average_increase = 2500)
  (h_total_salary : (n + 1) * (average_salary + average_increase) = n * average_salary + manager_salary) :
  n = 24 :=
by
  sorry

end employees_excluding_manager_l1962_196235


namespace max_value_fraction_l1962_196221

theorem max_value_fraction (a b x y : ℝ) (h1 : a > 1) (h2 : b > 1) (h3 : a^x = 3) (h4 : b^y = 3) (h5 : a + b = 2 * Real.sqrt 3) :
  1/x + 1/y ≤ 1 :=
sorry

end max_value_fraction_l1962_196221


namespace order_of_t_t2_neg_t_l1962_196239

theorem order_of_t_t2_neg_t (t : ℝ) (h : t^2 + t < 0) : t < t^2 ∧ t^2 < -t :=
by
  sorry

end order_of_t_t2_neg_t_l1962_196239


namespace problem_solution_l1962_196209

theorem problem_solution (a b c d x : ℚ) 
  (h1 : 2 * a + 2 = x) 
  (h2 : 3 * b + 3 = x) 
  (h3 : 4 * c + 4 = x) 
  (h4 : 5 * d + 5 = x) 
  (h5 : 2 * a + 3 * b + 4 * c + 5 * d + 6 = x) 
  : 2 * a + 3 * b + 4 * c + 5 * d = -10 / 3 := 
by 
  sorry

end problem_solution_l1962_196209


namespace three_digit_number_division_l1962_196267

theorem three_digit_number_division :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ m : ℕ, 10 ≤ m ∧ m < 100 ∧ n / m = 8 ∧ n % m = 6) → n = 342 :=
by
  sorry

end three_digit_number_division_l1962_196267


namespace percentage_transform_l1962_196268

theorem percentage_transform (n : ℝ) (h : 0.3 * 0.4 * n = 36) : 0.4 * 0.3 * n = 36 :=
by
  sorry

end percentage_transform_l1962_196268


namespace ratio_of_ducks_to_total_goats_and_chickens_l1962_196271

theorem ratio_of_ducks_to_total_goats_and_chickens 
    (goats chickens ducks pigs : ℕ) 
    (h1 : goats = 66)
    (h2 : chickens = 2 * goats)
    (h3 : pigs = ducks / 3)
    (h4 : goats = pigs + 33) :
    (ducks : ℚ) / (goats + chickens : ℚ) = 1 / 2 := 
by
  sorry

end ratio_of_ducks_to_total_goats_and_chickens_l1962_196271


namespace small_bottles_initial_l1962_196212

theorem small_bottles_initial
  (S : ℤ)
  (big_bottles_initial : ℤ := 15000)
  (sold_small_bottles_percentage : ℚ := 0.11)
  (sold_big_bottles_percentage : ℚ := 0.12)
  (remaining_bottles_in_storage : ℤ := 18540)
  (remaining_small_bottles : ℚ := 0.89 * S)
  (remaining_big_bottles : ℚ := 0.88 * big_bottles_initial)
  (h : remaining_small_bottles + remaining_big_bottles = remaining_bottles_in_storage)
  : S = 6000 :=
by
  sorry

end small_bottles_initial_l1962_196212


namespace expansion_three_times_expansion_six_times_l1962_196244

-- Definition for the rule of expansion
def expand (a b : Nat) : Nat := a * b + a + b

-- Problem 1: Expansion with a = 1, b = 3 for 3 times results in 255.
theorem expansion_three_times : expand (expand (expand 1 3) 7) 31 = 255 := sorry

-- Problem 2: After 6 operations, the expanded number matches the given pattern.
theorem expansion_six_times (p q : ℕ) (hp : p > q) (hq : q > 0) : 
  ∃ m n, m = 8 ∧ n = 13 ∧ (expand (expand (expand (expand (expand (expand q (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) (expand p q)) = (q + 1) ^ m * (p + 1) ^ n - 1 :=
sorry

end expansion_three_times_expansion_six_times_l1962_196244


namespace right_triangle_area_inscribed_circle_l1962_196259

theorem right_triangle_area_inscribed_circle (r a b c : ℝ)
  (h_c : c = 6 + 7)
  (h_a : a = 6 + r)
  (h_b : b = 7 + r)
  (h_pyth : (6 + r)^2 + (7 + r)^2 = 13^2):
  (1 / 2) * (a * b) = 42 :=
by
  -- The necessary calculations have already been derived and verified
  sorry

end right_triangle_area_inscribed_circle_l1962_196259


namespace min_sum_x_y_l1962_196226

theorem min_sum_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y ≥ 9 :=
by sorry

example (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h_xy : 4 * x + y = x * y) : x + y = 9 ↔ (x = 3 ∧ y = 6) :=
by sorry

end min_sum_x_y_l1962_196226


namespace polygon_sides_l1962_196223

theorem polygon_sides (n : ℕ) (h : n - 3 = 5) : n = 8 :=
by {
  sorry
}

end polygon_sides_l1962_196223


namespace total_time_simultaneous_l1962_196286

def total_time_bread1 : Nat := 30 + 120 + 20 + 120 + 10 + 30 + 30 + 15
def total_time_bread2 : Nat := 90 + 15 + 20 + 25 + 10
def total_time_bread3 : Nat := 40 + 100 + 5 + 110 + 15 + 5 + 25 + 20

theorem total_time_simultaneous :
  max (max total_time_bread1 total_time_bread2) total_time_bread3 = 375 :=
by
  sorry

end total_time_simultaneous_l1962_196286


namespace intersection_line_through_circles_l1962_196237

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 = 10
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 + 2 * x + 2 * y - 14 = 0

theorem intersection_line_through_circles : 
  (∀ x y : ℝ, circle1_equation x y → circle2_equation x y → x + y - 2 = 0) :=
by
  intros x y h1 h2
  sorry

end intersection_line_through_circles_l1962_196237


namespace max_value_of_a_l1962_196214

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 + |2 * x - 6| ≥ a) → a ≤ 5 :=
by sorry

end max_value_of_a_l1962_196214


namespace quadratic_rewrite_as_square_of_binomial_plus_integer_l1962_196227

theorem quadratic_rewrite_as_square_of_binomial_plus_integer :
    ∃ a b, ∀ x, x^2 + 16 * x + 72 = (x + a)^2 + b ∧ b = 8 :=
by
  sorry

end quadratic_rewrite_as_square_of_binomial_plus_integer_l1962_196227


namespace half_angle_in_quadrant_l1962_196201

theorem half_angle_in_quadrant (α : ℝ) (k : ℤ) (h : k * 360 + 90 < α ∧ α < k * 360 + 180) :
  ∃ n : ℤ, (n * 360 + 45 < α / 2 ∧ α / 2 < n * 360 + 90) ∨ (n * 360 + 225 < α / 2 ∧ α / 2 < n * 360 + 270) :=
by sorry

end half_angle_in_quadrant_l1962_196201


namespace sales_tax_difference_l1962_196272

-- Definitions for the price and tax rates
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.065
def tax_rate2 : ℝ := 0.06
def tax_rate3 : ℝ := 0.07

-- Sales tax amounts derived from the given rates and item price
def tax_amount (rate : ℝ) (price : ℝ) : ℝ := rate * price

-- Calculate the individual tax amounts
def tax_amount1 : ℝ := tax_amount tax_rate1 item_price
def tax_amount2 : ℝ := tax_amount tax_rate2 item_price
def tax_amount3 : ℝ := tax_amount tax_rate3 item_price

-- Proposition stating the proof problem
theorem sales_tax_difference :
  max tax_amount1 (max tax_amount2 tax_amount3) - min tax_amount1 (min tax_amount2 tax_amount3) = 0.50 :=
by 
  sorry

end sales_tax_difference_l1962_196272


namespace find_original_number_of_men_l1962_196299

variable (M : ℕ) (W : ℕ)

-- Given conditions translated to Lean
def condition1 := M * 10 = W -- M men complete work W in 10 days
def condition2 := (M - 10) * 20 = W -- (M - 10) men complete work W in 20 days

theorem find_original_number_of_men (h1 : condition1 M W) (h2 : condition2 M W) : M = 20 :=
sorry

end find_original_number_of_men_l1962_196299


namespace range_of_a_l1962_196273

theorem range_of_a (a : ℝ) (x : ℝ) : (x^2 + 2*x > 3) → (x > a) → (¬ (x^2 + 2*x > 3) → ¬ (x > a)) → a ≥ 1 :=
by
  intros hp hq hr
  sorry

end range_of_a_l1962_196273


namespace solve_equation_l1962_196253

def equation (x : ℝ) : Prop := (2 / x + 3 * (4 / x / (8 / x)) = 1.2)

theorem solve_equation : 
  ∃ x : ℝ, equation x ∧ x = - 20 / 3 :=
by
  sorry

end solve_equation_l1962_196253


namespace median_of_consecutive_integers_l1962_196289

theorem median_of_consecutive_integers (sum_n : ℤ) (n : ℤ) 
  (h1 : sum_n = 6^4) (h2 : n = 36) : 
  (sum_n / n) = 36 :=
by
  sorry

end median_of_consecutive_integers_l1962_196289


namespace parallelogram_base_length_l1962_196205

theorem parallelogram_base_length (area height : ℝ) (h_area : area = 108) (h_height : height = 9) : 
  ∃ base : ℝ, base = area / height ∧ base = 12 := 
  by sorry

end parallelogram_base_length_l1962_196205


namespace clever_value_points_l1962_196203

def clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = (deriv f) x₀

theorem clever_value_points :
  (clever_value_point (fun x : ℝ => x^2)) ∧
  (clever_value_point (fun x : ℝ => Real.log x)) ∧
  (clever_value_point (fun x : ℝ => x + (1 / x))) :=
by
  -- Proof omitted
  sorry

end clever_value_points_l1962_196203


namespace solution_set_of_inequality_l1962_196211

theorem solution_set_of_inequality (x : ℝ) : 
  (|x| * (1 - 2 * x) > 0) ↔ (x ∈ ((Set.Iio 0) ∪ (Set.Ioo 0 (1/2)))) :=
by
  sorry

end solution_set_of_inequality_l1962_196211


namespace part_cost_l1962_196210

theorem part_cost (hours : ℕ) (hourly_rate total_paid : ℕ) 
  (h1 : hours = 2)
  (h2 : hourly_rate = 75)
  (h3 : total_paid = 300) : 
  total_paid - (hours * hourly_rate) = 150 := 
by
  sorry

end part_cost_l1962_196210


namespace age_of_eldest_child_l1962_196241

theorem age_of_eldest_child (age_sum : ∀ (x : ℕ), x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 40) :
  ∃ x, x + 8 = 12 :=
by {
  sorry
}

end age_of_eldest_child_l1962_196241


namespace percent_first_shift_participating_l1962_196258

variable (total_employees_in_company : ℕ)
variable (first_shift_employees : ℕ)
variable (second_shift_employees : ℕ)
variable (third_shift_employees : ℕ)
variable (second_shift_percent_participating : ℚ)
variable (third_shift_percent_participating : ℚ)
variable (overall_percent_participating : ℚ)
variable (first_shift_percent_participating : ℚ)

theorem percent_first_shift_participating :
  total_employees_in_company = 150 →
  first_shift_employees = 60 →
  second_shift_employees = 50 →
  third_shift_employees = 40 →
  second_shift_percent_participating = 0.40 →
  third_shift_percent_participating = 0.10 →
  overall_percent_participating = 0.24 →
  first_shift_percent_participating = (12 / 60) →
  first_shift_percent_participating = 0.20 := 
by 
  intros t_e f_s_e s_s_e t_s_e s_s_p_p t_s_p_p o_p_p f_s_p_p
  -- Sorry, here would be the place for the actual proof
  sorry

end percent_first_shift_participating_l1962_196258


namespace isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l1962_196243

-- Part 1 proof
theorem isosceles_triangle_sides_part1 (x : ℝ) (h1 : x + 2 * x + 2 * x = 20) : 
  x = 4 ∧ 2 * x = 8 :=
by
  sorry

-- Part 2 proof
theorem isosceles_triangle_sides_part2 (a b : ℝ) (h2 : a = 5) (h3 : 2 * b + a = 20) :
  b = 7.5 :=
by
  sorry

end isosceles_triangle_sides_part1_isosceles_triangle_sides_part2_l1962_196243


namespace stella_annual_income_l1962_196274

-- Define the conditions
def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def total_months : ℕ := 12

-- The question: What is Stella's annual income last year?
def annual_income (monthly_income : ℕ) (worked_months : ℕ) : ℕ :=
  monthly_income * worked_months

-- Prove that Stella's annual income last year was $49190
theorem stella_annual_income : annual_income monthly_income (total_months - unpaid_leave_months) = 49190 :=
by
  sorry

end stella_annual_income_l1962_196274


namespace five_star_three_eq_ten_l1962_196287

def operation (a b : ℝ) : ℝ := b^2 + 1

theorem five_star_three_eq_ten : operation 5 3 = 10 := by
  sorry

end five_star_three_eq_ten_l1962_196287


namespace choir_grouping_l1962_196238

theorem choir_grouping (sopranos altos tenors basses : ℕ)
  (h_sopranos : sopranos = 10)
  (h_altos : altos = 15)
  (h_tenors : tenors = 12)
  (h_basses : basses = 18)
  (ratio : ℕ) :
  ratio = 1 →
  ∃ G : ℕ, G ≤ 10 ∧ G ≤ 15 ∧ G ≤ 12 ∧ 2 * G ≤ 18 ∧ G = 9 :=
by sorry

end choir_grouping_l1962_196238


namespace cleaning_time_is_100_l1962_196206

def time_hosing : ℕ := 10
def time_shampoo_per : ℕ := 15
def num_shampoos : ℕ := 3
def time_drying : ℕ := 20
def time_brushing : ℕ := 25

def total_time : ℕ :=
  time_hosing + (num_shampoos * time_shampoo_per) + time_drying + time_brushing

theorem cleaning_time_is_100 :
  total_time = 100 :=
by
  sorry

end cleaning_time_is_100_l1962_196206


namespace find_starting_number_l1962_196202

theorem find_starting_number (n : ℤ) (h1 : ∀ k : ℤ, n ≤ k ∧ k ≤ 38 → k % 4 = 0) (h2 : (n + 38) / 2 = 22) : n = 8 :=
sorry

end find_starting_number_l1962_196202


namespace certain_number_l1962_196288

theorem certain_number (x y z : ℕ) 
  (h1 : x + y = 15) 
  (h2 : y = 7) 
  (h3 : 3 * x = z * y - 11) : 
  z = 5 :=
by sorry

end certain_number_l1962_196288


namespace remaining_funds_correct_l1962_196204

def david_initial_funds : ℝ := 1800
def emma_initial_funds : ℝ := 2400
def john_initial_funds : ℝ := 1200

def david_spent_percentage : ℝ := 0.60
def emma_spent_percentage : ℝ := 0.75
def john_spent_percentage : ℝ := 0.50

def david_remaining_funds : ℝ := david_initial_funds * (1 - david_spent_percentage)
def emma_spent : ℝ := emma_initial_funds * emma_spent_percentage
def emma_remaining_funds : ℝ := emma_spent - 800
def john_remaining_funds : ℝ := john_initial_funds * (1 - john_spent_percentage)

theorem remaining_funds_correct :
  david_remaining_funds = 720 ∧
  emma_remaining_funds = 1400 ∧
  john_remaining_funds = 600 :=
by
  sorry

end remaining_funds_correct_l1962_196204


namespace impossible_arrangement_of_300_numbers_in_circle_l1962_196280

theorem impossible_arrangement_of_300_numbers_in_circle :
  ¬ ∃ (nums : Fin 300 → ℕ), (∀ i : Fin 300, nums i > 0) ∧
    ∃ unique_exception : Fin 300,
      ∀ i : Fin 300, i ≠ unique_exception → nums i = Int.natAbs (nums (Fin.mod (i.val - 1) 300) - nums (Fin.mod (i.val + 1) 300)) := 
sorry

end impossible_arrangement_of_300_numbers_in_circle_l1962_196280


namespace additional_people_required_l1962_196224

-- Given condition: Four people can mow a lawn in 6 hours
def work_rate: ℕ := 4 * 6

-- New condition: Number of people needed to mow the lawn in 3 hours
def people_required_in_3_hours: ℕ := work_rate / 3

-- Statement: Number of additional people required
theorem additional_people_required : people_required_in_3_hours - 4 = 4 :=
by
  -- Proof would go here
  sorry

end additional_people_required_l1962_196224


namespace necessary_but_not_sufficient_ellipse_l1962_196291

def is_ellipse (m : ℝ) : Prop := 
  1 < m ∧ m < 3 ∧ m ≠ 2

theorem necessary_but_not_sufficient_ellipse (m : ℝ) :
  (1 < m ∧ m < 3) → (m ≠ 2) → is_ellipse m :=
by
  intros h₁ h₂
  have h : 1 < m ∧ m < 3 ∧ m ≠ 2 := ⟨h₁.left, h₁.right, h₂⟩
  exact h

end necessary_but_not_sufficient_ellipse_l1962_196291


namespace probability_of_no_adjacent_standing_is_123_over_1024_l1962_196222

def total_outcomes : ℕ := 2 ^ 10

 -- Define the recursive sequence a_n
def a : ℕ → ℕ 
| 0 => 1
| 1 => 1
| n + 2 => a (n + 1) + a n

lemma a_10_val : a 10 = 123 := by
  sorry

def probability_no_adjacent_standing (n : ℕ): ℚ :=
  a n / total_outcomes

theorem probability_of_no_adjacent_standing_is_123_over_1024 :
  probability_no_adjacent_standing 10 = 123 / 1024 := by
  rw [probability_no_adjacent_standing, total_outcomes, a_10_val]
  norm_num

end probability_of_no_adjacent_standing_is_123_over_1024_l1962_196222


namespace quadratic_decreasing_right_of_axis_of_symmetry_l1962_196263

theorem quadratic_decreasing_right_of_axis_of_symmetry :
  ∀ x : ℝ, -2 * (x - 1)^2 < -2 * (x + 1 - 1)^2 →
  (∀ x' : ℝ, x' > 1 → -2 * (x' - 1)^2 < -2 * (x + 1 - 1)^2) :=
by
  sorry

end quadratic_decreasing_right_of_axis_of_symmetry_l1962_196263


namespace fg_of_3_eq_79_l1962_196233

def g (x : ℤ) : ℤ := x ^ 3
def f (x : ℤ) : ℤ := 3 * x - 2

theorem fg_of_3_eq_79 : f (g 3) = 79 := by
  sorry

end fg_of_3_eq_79_l1962_196233


namespace system_solutions_l1962_196261

theorem system_solutions (a b : ℝ) :
  (∃ (x y : ℝ), x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := 
sorry

end system_solutions_l1962_196261


namespace car_turns_proof_l1962_196296

def turns_opposite_direction (angle1 angle2 : ℝ) : Prop :=
  angle1 + angle2 = 180

theorem car_turns_proof
  (angle1 angle2 : ℝ)
  (h1 : (angle1 = 50 ∧ angle2 = 130) ∨ (angle1 = -50 ∧ angle2 = 130) ∨ 
       (angle1 = 50 ∧ angle2 = -130) ∨ (angle1 = 30 ∧ angle2 = -30)) :
  turns_opposite_direction angle1 angle2 ↔ (angle1 = 50 ∧ angle2 = 130) :=
by
  sorry

end car_turns_proof_l1962_196296


namespace find_k_l1962_196279

theorem find_k (k : ℕ) : (1 / 3)^32 * (1 / 125)^k = 1 / 27^32 → k = 0 :=
by {
  sorry
}

end find_k_l1962_196279


namespace koala_fiber_intake_l1962_196276

theorem koala_fiber_intake 
  (absorption_rate : ℝ) 
  (absorbed_fiber : ℝ) 
  (eaten_fiber : ℝ) 
  (h1 : absorption_rate = 0.40) 
  (h2 : absorbed_fiber = 16)
  (h3 : absorbed_fiber = absorption_rate * eaten_fiber) :
  eaten_fiber = 40 := 
  sorry

end koala_fiber_intake_l1962_196276


namespace moe_share_of_pie_l1962_196208

-- Definitions based on conditions
def leftover_pie : ℚ := 8 / 9
def num_people : ℚ := 3

-- Theorem to prove the amount of pie Moe took home
theorem moe_share_of_pie : (leftover_pie / num_people) = 8 / 27 := by
  sorry

end moe_share_of_pie_l1962_196208


namespace ratio_is_l1962_196250

noncomputable def volume_dodecahedron (s : ℝ) : ℝ := (15 + 7 * Real.sqrt 5) / 4 * s ^ 3

noncomputable def volume_tetrahedron (s : ℝ) : ℝ := Real.sqrt 2 / 12 * ((Real.sqrt 3 / 2) * s) ^ 3

noncomputable def ratio_volumes (s : ℝ) : ℝ := volume_dodecahedron s / volume_tetrahedron s

theorem ratio_is (s : ℝ) : ratio_volumes s = (60 + 28 * Real.sqrt 5) / Real.sqrt 6 :=
by
  sorry

end ratio_is_l1962_196250


namespace find_number_l1962_196294

theorem find_number (x : ℝ) : (x + 1) / (x + 5) = (x + 5) / (x + 13) → x = 3 :=
sorry

end find_number_l1962_196294


namespace math_problem_l1962_196234

variable (a : ℝ)
noncomputable def problem := a = Real.sqrt 11 - 1
noncomputable def target := a^2 + 2 * a + 1 = 11

theorem math_problem (h : problem a) : target a :=
  sorry

end math_problem_l1962_196234


namespace inequality_proof_l1962_196284

theorem inequality_proof (a b : ℝ) (h : a < b) : -a - 1 > -b - 1 :=
sorry

end inequality_proof_l1962_196284


namespace trigonometric_identity_l1962_196264

theorem trigonometric_identity
  (α β : Real)
  (h : Real.cos α * Real.cos β - Real.sin α * Real.sin β = 0) :
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = 1 ∨
  Real.sin α * Real.cos β + Real.cos α * Real.sin β = -1 := by
  sorry

end trigonometric_identity_l1962_196264


namespace speed_of_goods_train_l1962_196248

theorem speed_of_goods_train 
  (t₁ t₂ v_express : ℝ)
  (h1 : v_express = 90) 
  (h2 : t₁ = 6) 
  (h3 : t₂ = 4)
  (h4 : v_express * t₂ = v * (t₁ + t₂)) : 
  v = 36 :=
by
  sorry

end speed_of_goods_train_l1962_196248


namespace bridget_apples_l1962_196220

theorem bridget_apples (x : ℕ) (h1 : x - 2 ≥ 0) (h2 : (x - 2) / 3 = 0 → false)
    (h3 : (2 * (x - 2) / 3) - 5 = 6) : x = 20 :=
by
  sorry

end bridget_apples_l1962_196220


namespace number_of_sequences_with_at_least_two_reds_l1962_196270

theorem number_of_sequences_with_at_least_two_reds (n : ℕ) (h : n ≥ 2) :
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2 :=
by
  intros
  let T_n := 3 * 2^(n - 1)
  let R_0 := 2
  let R_1n := 4 * n - 4
  show T_n - R_0 - R_1n = 3 * 2^(n - 1) - 4 * n + 2
  sorry

end number_of_sequences_with_at_least_two_reds_l1962_196270


namespace original_number_of_people_l1962_196260

theorem original_number_of_people (x : ℕ) (h1 : 3 ∣ x) (h2 : 6 ∣ x) (h3 : (x / 2) = 18) : x = 36 :=
by
  sorry

end original_number_of_people_l1962_196260


namespace correct_option_l1962_196232

-- Definition of the conditions
def conditionA : Prop := (Real.sqrt ((-1 : ℝ)^2) = 1)
def conditionB : Prop := (Real.sqrt ((-1 : ℝ)^2) = -1)
def conditionC : Prop := (Real.sqrt (-(1^2) : ℝ) = 1)
def conditionD : Prop := (Real.sqrt (-(1^2) : ℝ) = -1)

-- Proving the correct condition
theorem correct_option : conditionA := by
  sorry

end correct_option_l1962_196232


namespace depression_comparative_phrase_l1962_196283

def correct_comparative_phrase (phrase : String) : Prop :=
  phrase = "twice as…as"

theorem depression_comparative_phrase :
  correct_comparative_phrase "twice as…as" :=
by
  sorry

end depression_comparative_phrase_l1962_196283


namespace least_value_of_x_l1962_196265

theorem least_value_of_x (x p : ℕ) (h1 : (x / (11 * p)) = 3) (h2 : x > 0) (h3 : Nat.Prime p) : x = 66 := by
  sorry

end least_value_of_x_l1962_196265


namespace find_all_f_l1962_196298

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_all_f :
  (∀ x : ℝ, f x ≥ 0) ∧
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x + 2 * y^2) →
  ∃ a c : ℝ, (∀ x : ℝ, f x = x^2 + a * x + c) ∧ (a^2 - 4 * c ≤ 0) := sorry

end find_all_f_l1962_196298


namespace part1_part2_l1962_196251

noncomputable def f (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp x
noncomputable def h (x : ℝ) : ℝ := -3 * Real.log x + x^3 + (2 * x^2 - 4 * x) * Real.exp x + 7

theorem part1 (a : ℤ) : 
  (∀ x, (a : ℝ) < x ∧ x < a + 5 → ∀ y, (a : ℝ) < y ∧ y < a + 5 → f x ≤ f y) →
  a = -6 ∨ a = -5 ∨ a = -4 :=
sorry

theorem part2 (x : ℝ) (hx : 0 < x) : 
  f x < h x :=
sorry

end part1_part2_l1962_196251


namespace find_b_l1962_196297

noncomputable def complex_b_value (i : ℂ) (b : ℝ) : Prop :=
(1 + b * i) * i = 1 + i

theorem find_b (i : ℂ) (b : ℝ) (hi : i^2 = -1) (h : complex_b_value i b) : b = -1 :=
by {
  sorry
}

end find_b_l1962_196297


namespace inequality_positive_reals_l1962_196293

open Real

variable (x y : ℝ)

theorem inequality_positive_reals (hx : 0 < x) (hy : 0 < y) : x^2 + (8 / (x * y)) + y^2 ≥ 8 :=
by
  sorry

end inequality_positive_reals_l1962_196293


namespace integer_roots_p_l1962_196230

theorem integer_roots_p (p x1 x2 : ℤ) (h1 : x1 * x2 = p + 4) (h2 : x1 + x2 = -p) : p = 8 ∨ p = -4 := 
sorry

end integer_roots_p_l1962_196230
