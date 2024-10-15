import Mathlib

namespace NUMINAMATH_GPT_order_fractions_l700_70097

theorem order_fractions : (16/13 : ℚ) < 21/17 ∧ 21/17 < 20/15 :=
by {
  -- use cross-multiplication:
  -- 16*17 < 21*13 -> 272 < 273 -> true
  -- 16*15 < 20*13 -> 240 < 260 -> true
  -- 21*15 < 20*17 -> 315 < 340 -> true
  sorry
}

end NUMINAMATH_GPT_order_fractions_l700_70097


namespace NUMINAMATH_GPT_divides_number_of_ones_l700_70022

theorem divides_number_of_ones (n : ℕ) (h1 : ¬(2 ∣ n)) (h2 : ¬(5 ∣ n)) : ∃ k : ℕ, n ∣ ((10^k - 1) / 9) :=
by
  sorry

end NUMINAMATH_GPT_divides_number_of_ones_l700_70022


namespace NUMINAMATH_GPT_max_f_angle_A_of_triangle_l700_70069

noncomputable def f (x : ℝ) : ℝ := (Real.cos (2 * x - 4 * Real.pi / 3)) + 2 * (Real.cos x)^2

theorem max_f : ∃ x : ℝ, f x = 2 := sorry

theorem angle_A_of_triangle (A B C : ℝ) (h : A + B + C = Real.pi)
  (h2 : f (B + C) = 3 / 2) : A = Real.pi / 3 := sorry

end NUMINAMATH_GPT_max_f_angle_A_of_triangle_l700_70069


namespace NUMINAMATH_GPT_larger_number_is_400_l700_70090

def problem_statement : Prop :=
  ∃ (a b hcf lcm num1 num2 : ℕ),
  hcf = 25 ∧
  a = 14 ∧
  b = 16 ∧
  lcm = hcf * a * b ∧
  num1 = hcf * a ∧
  num2 = hcf * b ∧
  num1 < num2 ∧
  num2 = 400

theorem larger_number_is_400 : problem_statement :=
  sorry

end NUMINAMATH_GPT_larger_number_is_400_l700_70090


namespace NUMINAMATH_GPT_regular_polygon_sides_l700_70021

theorem regular_polygon_sides (n : ℕ) (h1 : 180 * (n - 2) = 144 * n) : n = 10 := 
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l700_70021


namespace NUMINAMATH_GPT_sum_abs_binom_coeff_l700_70014

theorem sum_abs_binom_coeff (a a1 a2 a3 a4 a5 a6 a7 : ℤ)
    (h : (1 - 2 * x) ^ 7 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) :
    |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3 ^ 7 - 1 := sorry

end NUMINAMATH_GPT_sum_abs_binom_coeff_l700_70014


namespace NUMINAMATH_GPT_inequality_proof_l700_70024

noncomputable def inequality_holds (a b : ℝ) (ha : a > 1) (hb : b > 1) : Prop :=
  (a ^ 2) / (b - 1) + (b ^ 2) / (a - 1) ≥ 8

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  inequality_holds a b ha hb :=
sorry

end NUMINAMATH_GPT_inequality_proof_l700_70024


namespace NUMINAMATH_GPT_average_price_per_book_l700_70039

theorem average_price_per_book (books1_cost : ℕ) (books1_count : ℕ)
    (books2_cost : ℕ) (books2_count : ℕ)
    (h1 : books1_cost = 6500) (h2 : books1_count = 65)
    (h3 : books2_cost = 2000) (h4 : books2_count = 35) :
    (books1_cost + books2_cost) / (books1_count + books2_count) = 85 :=
by
    sorry

end NUMINAMATH_GPT_average_price_per_book_l700_70039


namespace NUMINAMATH_GPT_table_length_l700_70041

theorem table_length (L : ℕ) (H1 : ∃ n : ℕ, 80 = n * L)
  (H2 : L ≥ 16) (H3 : ∃ m : ℕ, 16 = m * 4)
  (H4 : L % 4 = 0) : L = 20 := by 
sorry

end NUMINAMATH_GPT_table_length_l700_70041


namespace NUMINAMATH_GPT_value_of_m_plus_n_l700_70002

-- Conditions
variables (m n : ℤ)
def P_symmetric_Q_x_axis := (m - 1 = 2 * m - 4) ∧ (n + 2 = -2)

-- Proof Problem Statement
theorem value_of_m_plus_n (h : P_symmetric_Q_x_axis m n) : (m + n) ^ 2023 = -1 := sorry

end NUMINAMATH_GPT_value_of_m_plus_n_l700_70002


namespace NUMINAMATH_GPT_Tim_placed_rulers_l700_70020

variable (initial_rulers final_rulers : ℕ)
variable (placed_rulers : ℕ)

-- Given conditions
def initial_rulers_def : initial_rulers = 11 := sorry
def final_rulers_def : final_rulers = 25 := sorry

-- Goal
theorem Tim_placed_rulers : placed_rulers = final_rulers - initial_rulers :=
  by
  sorry

end NUMINAMATH_GPT_Tim_placed_rulers_l700_70020


namespace NUMINAMATH_GPT_original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l700_70075

-- Definition of the quadrilateral being a rhombus
def is_rhombus (quad : Type) : Prop := 
-- A quadrilateral is a rhombus if and only if all its sides are equal in length
sorry

-- Definition of the diagonals of quadrilateral being perpendicular
def diagonals_are_perpendicular (quad : Type) : Prop := 
-- The diagonals of a quadrilateral are perpendicular
sorry

-- Original proposition: If a quadrilateral is a rhombus, then its diagonals are perpendicular to each other
theorem original_proposition (quad : Type) : is_rhombus quad → diagonals_are_perpendicular quad :=
sorry

-- Converse proposition: If the diagonals of a quadrilateral are perpendicular to each other, then it is a rhombus, which is False
theorem converse_proposition_false (quad : Type) : diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

-- Inverse proposition: If a quadrilateral is not a rhombus, then its diagonals are not perpendicular, which is False
theorem inverse_proposition_false (quad : Type) : ¬ is_rhombus quad → ¬ diagonals_are_perpendicular quad :=
sorry

-- Contrapositive proposition: If the diagonals of a quadrilateral are not perpendicular, then it is not a rhombus, which is True
theorem contrapositive_proposition_true (quad : Type) : ¬ diagonals_are_perpendicular quad → ¬ is_rhombus quad :=
sorry

end NUMINAMATH_GPT_original_proposition_converse_proposition_false_inverse_proposition_false_contrapositive_proposition_true_l700_70075


namespace NUMINAMATH_GPT_men_in_first_group_l700_70087

theorem men_in_first_group
  (M : ℕ) -- number of men in the first group
  (h1 : M * 8 * 24 = 12 * 8 * 16) : M = 8 :=
sorry

end NUMINAMATH_GPT_men_in_first_group_l700_70087


namespace NUMINAMATH_GPT_find_b_squared_l700_70072

theorem find_b_squared :
  let ellipse_eq := ∀ x y : ℝ, x^2 / 25 + y^2 / b^2 = 1
  let hyperbola_eq := ∀ x y : ℝ, x^2 / 225 - y^2 / 144 = 1 / 36
  let coinciding_foci := 
    let c_ellipse := Real.sqrt (25 - b^2)
    let c_hyperbola := Real.sqrt ((225 / 36) + (144 / 36))
    c_ellipse = c_hyperbola
  ellipse_eq ∧ hyperbola_eq ∧ coinciding_foci → b^2 = 14.75
:= by sorry

end NUMINAMATH_GPT_find_b_squared_l700_70072


namespace NUMINAMATH_GPT_travel_time_correct_l700_70045

def luke_bus_to_work : ℕ := 70
def paula_bus_to_work : ℕ := (70 * 3) / 5
def jane_train_to_work : ℕ := 120
def michael_cycle_to_work : ℕ := 120 / 4

def luke_bike_back_home : ℕ := 70 * 5
def paula_bus_back_home: ℕ := paula_bus_to_work
def jane_train_back_home : ℕ := 120 * 2
def michael_cycle_back_home : ℕ := michael_cycle_to_work

def luke_total_travel : ℕ := luke_bus_to_work + luke_bike_back_home
def paula_total_travel : ℕ := paula_bus_to_work + paula_bus_back_home
def jane_total_travel : ℕ := jane_train_to_work + jane_train_back_home
def michael_total_travel : ℕ := michael_cycle_to_work + michael_cycle_back_home

def total_travel_time : ℕ := luke_total_travel + paula_total_travel + jane_total_travel + michael_total_travel

theorem travel_time_correct : total_travel_time = 924 :=
by sorry

end NUMINAMATH_GPT_travel_time_correct_l700_70045


namespace NUMINAMATH_GPT_ball_bounce_height_l700_70098

theorem ball_bounce_height (b : ℕ) (h₀: ℝ) (r: ℝ) (h_final: ℝ) :
  h₀ = 200 ∧ r = 3 / 4 ∧ h_final = 25 →
  200 * (3 / 4) ^ b < 25 ↔ b ≥ 25 := by
  sorry

end NUMINAMATH_GPT_ball_bounce_height_l700_70098


namespace NUMINAMATH_GPT_poodle_barks_count_l700_70076

-- Define the conditions as hypothesis
variables (poodle_barks terrier_barks terrier_hushes : ℕ)

-- Define the conditions
def condition1 : Prop :=
  poodle_barks = 2 * terrier_barks

def condition2 : Prop :=
  terrier_hushes = terrier_barks / 2

def condition3 : Prop :=
  terrier_hushes = 6

-- The theorem we need to prove
theorem poodle_barks_count (poodle_barks terrier_barks terrier_hushes : ℕ)
  (h1 : condition1 poodle_barks terrier_barks)
  (h2 : condition2 terrier_barks terrier_hushes)
  (h3 : condition3 terrier_hushes) :
  poodle_barks = 24 :=
by
  -- Proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_poodle_barks_count_l700_70076


namespace NUMINAMATH_GPT_fermats_little_theorem_l700_70080

theorem fermats_little_theorem (n p : ℕ) [hp : Fact p.Prime] : p ∣ (n^p - n) :=
sorry

end NUMINAMATH_GPT_fermats_little_theorem_l700_70080


namespace NUMINAMATH_GPT_volume_of_cuboid_l700_70077

-- Define the edges of the cuboid
def edge1 : ℕ := 6
def edge2 : ℕ := 5
def edge3 : ℕ := 6

-- Define the volume formula for a cuboid
def volume (a b c : ℕ) : ℕ := a * b * c

-- State the theorem
theorem volume_of_cuboid : volume edge1 edge2 edge3 = 180 := by
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_l700_70077


namespace NUMINAMATH_GPT_circular_seat_coloring_l700_70085

def count_colorings (n : ℕ) : ℕ :=
  sorry

theorem circular_seat_coloring :
  count_colorings 6 = 66 :=
by
  sorry

end NUMINAMATH_GPT_circular_seat_coloring_l700_70085


namespace NUMINAMATH_GPT_count_squares_below_line_l700_70046

theorem count_squares_below_line (units : ℕ) :
  let intercept_x := 221;
  let intercept_y := 7;
  let total_squares := intercept_x * intercept_y;
  let diagonal_squares := intercept_x - 1 + intercept_y - 1 + 1; 
  let non_diag_squares := total_squares - diagonal_squares;
  let below_line := non_diag_squares / 2;
  below_line = 660 :=
by
  sorry

end NUMINAMATH_GPT_count_squares_below_line_l700_70046


namespace NUMINAMATH_GPT_product_of_digits_of_N_l700_70029

theorem product_of_digits_of_N (N : ℕ) (h : N * (N + 1) / 2 = 2485) : 
  (N.digits 10).prod = 0 :=
sorry

end NUMINAMATH_GPT_product_of_digits_of_N_l700_70029


namespace NUMINAMATH_GPT_percentage_of_a_l700_70058

theorem percentage_of_a (a : ℕ) (x : ℕ) (h1 : a = 190) (h2 : (x * a) / 100 = 95) : x = 50 := by
  sorry

end NUMINAMATH_GPT_percentage_of_a_l700_70058


namespace NUMINAMATH_GPT_cheryl_distance_walked_l700_70057

theorem cheryl_distance_walked :
  let s1 := 2  -- speed during the first segment in miles per hour
  let t1 := 3  -- time during the first segment in hours
  let s2 := 4  -- speed during the second segment in miles per hour
  let t2 := 2  -- time during the second segment in hours
  let s3 := 1  -- speed during the third segment in miles per hour
  let t3 := 3  -- time during the third segment in hours
  let s4 := 3  -- speed during the fourth segment in miles per hour
  let t4 := 5  -- time during the fourth segment in hours
  let d1 := s1 * t1  -- distance for the first segment
  let d2 := s2 * t2  -- distance for the second segment
  let d3 := s3 * t3  -- distance for the third segment
  let d4 := s4 * t4  -- distance for the fourth segment
  d1 + d2 + d3 + d4 = 32 :=
by
  sorry

end NUMINAMATH_GPT_cheryl_distance_walked_l700_70057


namespace NUMINAMATH_GPT_value_of_expr_l700_70088

theorem value_of_expr : (365^2 - 349^2) / 16 = 714 := by
  sorry

end NUMINAMATH_GPT_value_of_expr_l700_70088


namespace NUMINAMATH_GPT_locus_midpoint_l700_70026

/-- Given a fixed point A (4, -2) and a moving point B on the curve x^2 + y^2 = 4,
    prove that the locus of the midpoint P of the line segment AB satisfies the equation 
    (x - 2)^2 + (y + 1)^2 = 1. -/
theorem locus_midpoint (A B P : ℝ × ℝ)
  (hA : A = (4, -2))
  (hB : ∃ (x y : ℝ), B = (x, y) ∧ x^2 + y^2 = 4)
  (hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (P.1 - 2)^2 + (P.2 + 1)^2 = 1 :=
sorry

end NUMINAMATH_GPT_locus_midpoint_l700_70026


namespace NUMINAMATH_GPT_hannah_age_double_july_age_20_years_ago_l700_70009

/-- Define the current ages of July (J) and her husband (H) -/
def current_age_july : ℕ := 23
def current_age_husband : ℕ := 25

/-- Assertion that July's husband is 2 years older than her -/
axiom husband_older : current_age_husband = current_age_july + 2

/-- We denote the ages 20 years ago -/
def age_july_20_years_ago := current_age_july - 20
def age_hannah_20_years_ago := current_age_husband - 20 - 2 * (current_age_july - 20)

theorem hannah_age_double_july_age_20_years_ago :
  age_hannah_20_years_ago = 6 :=
by sorry

end NUMINAMATH_GPT_hannah_age_double_july_age_20_years_ago_l700_70009


namespace NUMINAMATH_GPT_find_x_l700_70052

-- Definition of logarithm in Lean
noncomputable def log (b a: ℝ) : ℝ := Real.log a / Real.log b

-- Problem statement in Lean
theorem find_x (x : ℝ) (h : log 64 4 = 1 / 3) : log x 8 = 1 / 3 → x = 512 :=
by sorry

end NUMINAMATH_GPT_find_x_l700_70052


namespace NUMINAMATH_GPT_nathan_tomato_plants_l700_70062

theorem nathan_tomato_plants (T: ℕ) : 
  5 * 14 + T * 16 = 186 * 7 / 6 + 9 * 10 :=
  sorry

end NUMINAMATH_GPT_nathan_tomato_plants_l700_70062


namespace NUMINAMATH_GPT_problem_statement_l700_70086

-- Define the constants and variables
variables (x y z a b c : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := x / a + y / b + z / c = 4
def condition2 : Prop := a / x + b / y + c / z = 1

-- State the theorem that proves the question equals the correct answer
theorem problem_statement (h1 : condition1 x y z a b c) (h2 : condition2 x y z a b c) :
    x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 12 :=
sorry

end NUMINAMATH_GPT_problem_statement_l700_70086


namespace NUMINAMATH_GPT_valid_call_time_at_15_l700_70038

def time_difference := 5 -- Beijing is 5 hours ahead of Moscow

def beijing_start_time := 14 -- Start time in Beijing corresponding to 9:00 in Moscow
def beijing_end_time := 17  -- End time in Beijing corresponding to 17:00 in Beijing

-- Define the call time in Beijing
def call_time_beijing := 15

-- The time window during which they can start the call in Beijing
def valid_call_time (t : ℕ) : Prop :=
  beijing_start_time <= t ∧ t <= beijing_end_time

-- The theorem to prove that 15:00 is a valid call time in Beijing
theorem valid_call_time_at_15 : valid_call_time call_time_beijing :=
by
  sorry

end NUMINAMATH_GPT_valid_call_time_at_15_l700_70038


namespace NUMINAMATH_GPT_elevator_max_weight_capacity_l700_70096

theorem elevator_max_weight_capacity 
  (num_adults : ℕ)
  (weight_adult : ℕ)
  (num_children : ℕ)
  (weight_child : ℕ)
  (max_next_person_weight : ℕ) 
  (H_adults : num_adults = 3)
  (H_weight_adult : weight_adult = 140)
  (H_children : num_children = 2)
  (H_weight_child : weight_child = 64)
  (H_max_next : max_next_person_weight = 52) : 
  num_adults * weight_adult + num_children * weight_child + max_next_person_weight = 600 := 
by
  sorry

end NUMINAMATH_GPT_elevator_max_weight_capacity_l700_70096


namespace NUMINAMATH_GPT_parabola_standard_equation_l700_70010

theorem parabola_standard_equation (h : ∀ y, y = 1/2) : ∃ c : ℝ, c = -2 ∧ (∀ x y, x^2 = c * y) :=
by
  -- Considering 'h' provides the condition for the directrix
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_l700_70010


namespace NUMINAMATH_GPT_locus_centers_of_tangent_circles_l700_70056

theorem locus_centers_of_tangent_circles (a b : ℝ) :
  (x^2 + y^2 = 1) ∧ ((x - 1)^2 + (y -1)^2 = 81) →
  (a^2 + b^2 - (2 * a * b) / 63 - (66 * a) / 63 - (66 * b) / 63 + 17 = 0) :=
by
  sorry

end NUMINAMATH_GPT_locus_centers_of_tangent_circles_l700_70056


namespace NUMINAMATH_GPT_three_digit_numbers_divide_26_l700_70066

def divides (d n : ℕ) : Prop := ∃ k, n = d * k

theorem three_digit_numbers_divide_26 (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (divides 26 (a^2 + b^2 + c^2)) ↔ 
    ((a = 1 ∧ b = 0 ∧ c = 0) ∨
     (a = 1 ∧ b = 1 ∧ c = 0) ∨
     (a = 3 ∧ b = 2 ∧ c = 0) ∨
     (a = 5 ∧ b = 1 ∧ c = 0) ∨
     (a = 4 ∧ b = 3 ∧ c = 1)) :=
by 
  sorry

end NUMINAMATH_GPT_three_digit_numbers_divide_26_l700_70066


namespace NUMINAMATH_GPT_probability_point_in_region_l700_70013

theorem probability_point_in_region (x y : ℝ) 
  (h1 : 0 ≤ x ∧ x ≤ 2010) 
  (h2 : 0 ≤ y ∧ y ≤ 2009) 
  (h3 : ∃ (u v : ℝ), (u, v) = (x, y) ∧ x > 2 * y ∧ y > 500) : 
  ∃ p : ℚ, p = 1505 / 4018 := 
sorry

end NUMINAMATH_GPT_probability_point_in_region_l700_70013


namespace NUMINAMATH_GPT_triangle_shape_area_l700_70054

theorem triangle_shape_area (a b : ℕ) (area_small area_middle area_large : ℕ) :
  a = 2 →
  b = 2 →
  area_small = (1 / 2) * a * b →
  area_middle = 2 * area_small →
  area_large = 2 * area_middle →
  area_small + area_middle + area_large = 14 :=
by
  intros
  sorry

end NUMINAMATH_GPT_triangle_shape_area_l700_70054


namespace NUMINAMATH_GPT_least_number_to_be_added_l700_70047

theorem least_number_to_be_added (k : ℕ) (h₁ : Nat.Prime 29) (h₂ : Nat.Prime 37) (H : Nat.gcd 29 37 = 1) : 
  (433124 + k) % Nat.lcm 29 37 = 0 → k = 578 :=
by 
  sorry

end NUMINAMATH_GPT_least_number_to_be_added_l700_70047


namespace NUMINAMATH_GPT_karen_wrong_questions_l700_70028

theorem karen_wrong_questions (k l n : ℕ) (h1 : k + l = 6 + n) (h2 : k + n = l + 9) : k = 6 := 
by
  sorry

end NUMINAMATH_GPT_karen_wrong_questions_l700_70028


namespace NUMINAMATH_GPT_remainder_2365947_div_8_l700_70055

theorem remainder_2365947_div_8 : (2365947 % 8) = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_2365947_div_8_l700_70055


namespace NUMINAMATH_GPT_percentage_increase_soda_price_l700_70074

theorem percentage_increase_soda_price
  (C_new : ℝ) (S_new : ℝ) (C_increase : ℝ) (C_total_before : ℝ)
  (h1 : C_new = 20)
  (h2: S_new = 6)
  (h3: C_increase = 0.25)
  (h4: C_new * (1 - C_increase) + S_new * (1 + (S_new / (S_new * (1 + (S_new / (S_new * 0.5)))))) = C_total_before) : 
  (S_new - S_new * (1 - C_increase) * 100 / (S_new * (1 + 0.5)) * C_total_before) = 50 := 
by 
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_percentage_increase_soda_price_l700_70074


namespace NUMINAMATH_GPT_wall_area_160_l700_70071

noncomputable def wall_area (small_tile_area : ℝ) (fraction_small : ℝ) : ℝ :=
  small_tile_area / fraction_small

theorem wall_area_160 (small_tile_area : ℝ) (fraction_small : ℝ) (h1 : small_tile_area = 80) (h2 : fraction_small = 1 / 2) :
  wall_area small_tile_area fraction_small = 160 :=
by
  rw [wall_area, h1, h2]
  norm_num

end NUMINAMATH_GPT_wall_area_160_l700_70071


namespace NUMINAMATH_GPT_value_of_some_number_l700_70050

theorem value_of_some_number (a : ℤ) (h : a = 105) :
  (a ^ 3 = 3 * (5 ^ 3) * (3 ^ 2) * (7 ^ 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_some_number_l700_70050


namespace NUMINAMATH_GPT_yellow_mugs_count_l700_70027

variables (R B Y O : ℕ)
variables (B_eq_3R : B = 3 * R)
variables (R_eq_Y_div_2 : R = Y / 2)
variables (O_eq_4 : O = 4)
variables (mugs_eq_40 : R + B + Y + O = 40)

theorem yellow_mugs_count : Y = 12 :=
by 
  sorry

end NUMINAMATH_GPT_yellow_mugs_count_l700_70027


namespace NUMINAMATH_GPT_trajectory_range_k_l700_70079

-- Condition Definitions
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def N (x : ℝ) : ℝ × ℝ := (x, 0)
def vector_MN (x y : ℝ) : ℝ × ℝ := (0, -y)
def vector_AN (x : ℝ) : ℝ × ℝ := (x + 1, 0)
def vector_BN (x : ℝ) : ℝ × ℝ := (x - 1, 0)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Problem 1: Prove the trajectory equation
theorem trajectory (x y : ℝ) (h : (vector_MN x y).1^2 + (vector_MN x y).2^2 = dot_product (vector_AN x) (vector_BN x)) :
  x^2 - y^2 = 1 :=
sorry

-- Problem 2: Prove the range of k
theorem range_k (k : ℝ) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 - y^2 = 1) ↔ -Real.sqrt 2 ≤ k ∧ k ≤ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_trajectory_range_k_l700_70079


namespace NUMINAMATH_GPT_value_of_m_sub_n_l700_70067

theorem value_of_m_sub_n (m n : ℤ) (h1 : |m| = 5) (h2 : n^2 = 36) (h3 : m * n < 0) : m - n = 11 ∨ m - n = -11 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_m_sub_n_l700_70067


namespace NUMINAMATH_GPT_find_third_number_l700_70042

theorem find_third_number : ∃ (x : ℝ), 0.3 * 0.8 + x * 0.5 = 0.29 ∧ x = 0.1 :=
by
  use 0.1
  sorry

end NUMINAMATH_GPT_find_third_number_l700_70042


namespace NUMINAMATH_GPT_three_digit_number_ends_with_same_three_digits_l700_70037

theorem three_digit_number_ends_with_same_three_digits (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, k ≥ 1 → N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) := 
sorry

end NUMINAMATH_GPT_three_digit_number_ends_with_same_three_digits_l700_70037


namespace NUMINAMATH_GPT_max_cube_side_length_max_rect_parallelepiped_dimensions_l700_70051

-- Part (a)
theorem max_cube_side_length (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ s : ℝ, s = a * b * c / (a * b + b * c + a * c) :=
sorry

-- Part (b)
theorem max_rect_parallelepiped_dimensions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ x y z : ℝ, x = a / 3 ∧ y = b / 3 ∧ z = c / 3 :=
sorry

end NUMINAMATH_GPT_max_cube_side_length_max_rect_parallelepiped_dimensions_l700_70051


namespace NUMINAMATH_GPT_find_x_l700_70089

theorem find_x (x : ℝ) (h : (3 * x - 7) / 4 = 14) : x = 21 :=
sorry

end NUMINAMATH_GPT_find_x_l700_70089


namespace NUMINAMATH_GPT_gcd_C_D_eq_6_l700_70023

theorem gcd_C_D_eq_6
  (C D : ℕ)
  (h_lcm : Nat.lcm C D = 180)
  (h_ratio : C = 5 * D / 6) :
  Nat.gcd C D = 6 := 
by
  sorry

end NUMINAMATH_GPT_gcd_C_D_eq_6_l700_70023


namespace NUMINAMATH_GPT_dinner_cost_per_kid_l700_70081

theorem dinner_cost_per_kid
  (row_ears : ℕ)
  (seeds_bag : ℕ)
  (seeds_ear : ℕ)
  (pay_row : ℝ)
  (bags_used : ℕ)
  (dinner_fraction : ℝ)
  (h1 : row_ears = 70)
  (h2 : seeds_bag = 48)
  (h3 : seeds_ear = 2)
  (h4 : pay_row = 1.5)
  (h5 : bags_used = 140)
  (h6 : dinner_fraction = 0.5) :
  ∃ (dinner_cost : ℝ), dinner_cost = 36 :=
by
  sorry

end NUMINAMATH_GPT_dinner_cost_per_kid_l700_70081


namespace NUMINAMATH_GPT_gcd_polynomial_l700_70003

theorem gcd_polynomial (b : ℤ) (k : ℤ) (hk : k % 2 = 1) (h_b : b = 1193 * k) :
  Int.gcd (2 * b^2 + 31 * b + 73) (b + 17) = 1 := 
  sorry

end NUMINAMATH_GPT_gcd_polynomial_l700_70003


namespace NUMINAMATH_GPT_line_through_intersection_parallel_to_y_axis_l700_70082

theorem line_through_intersection_parallel_to_y_axis:
  ∃ x, (∃ y, 3 * x + 2 * y - 5 = 0 ∧ x - 3 * y + 2 = 0) ∧
       (x = 1) :=
sorry

end NUMINAMATH_GPT_line_through_intersection_parallel_to_y_axis_l700_70082


namespace NUMINAMATH_GPT_horse_distance_traveled_l700_70064

theorem horse_distance_traveled :
  let r2 := 12
  let n2 := 120
  let D2 := n2 * 2 * Real.pi * r2
  D2 = 2880 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_horse_distance_traveled_l700_70064


namespace NUMINAMATH_GPT_true_propositions_count_l700_70007

theorem true_propositions_count :
  (∃ x₀ : ℤ, x₀^3 < 0) ∧
  ((∀ a : ℝ, (∃ x : ℝ, a*x^2 + 2*x + 1 = 0 ∧ x < 0) ↔ a ≤ 1) → false) ∧ 
  (¬ (∀ x : ℝ, x^2 = 1/4 * x^2 → y = 1 → false)) →
  true_prop_count = 1 := 
sorry

end NUMINAMATH_GPT_true_propositions_count_l700_70007


namespace NUMINAMATH_GPT_population_net_increase_l700_70060

-- Define the birth rate and death rate conditions
def birth_rate := 4 / 2 -- people per second
def death_rate := 2 / 2 -- people per second
def net_increase_per_sec := birth_rate - death_rate -- people per second

-- Define the duration of one day in seconds
def seconds_in_a_day := 24 * 3600 -- seconds

-- Define the problem to prove
theorem population_net_increase :
  net_increase_per_sec * seconds_in_a_day = 86400 :=
by
  sorry

end NUMINAMATH_GPT_population_net_increase_l700_70060


namespace NUMINAMATH_GPT_cordelia_bleach_time_l700_70018

theorem cordelia_bleach_time
    (H : ℕ)
    (total_time : H + 2 * H = 9) :
    H = 3 :=
by
  sorry

end NUMINAMATH_GPT_cordelia_bleach_time_l700_70018


namespace NUMINAMATH_GPT_fifth_equation_sum_first_17_even_sum_even_28_to_50_l700_70017

-- Define a function to sum the first n even numbers
def sum_even (n : ℕ) : ℕ := n * (n + 1)

-- Part (1) According to the pattern, write down the ⑤th equation
theorem fifth_equation : sum_even 5 = 30 := by
  sorry

-- Part (2) Calculate according to this pattern:
-- ① Sum of first 17 even numbers
theorem sum_first_17_even : sum_even 17 = 306 := by
  sorry

-- ② Sum of even numbers from 28 to 50
theorem sum_even_28_to_50 : 
  let sum_even_50 := sum_even 25
  let sum_even_26 := sum_even 13
  sum_even_50 - sum_even_26 = 468 := by
  sorry

end NUMINAMATH_GPT_fifth_equation_sum_first_17_even_sum_even_28_to_50_l700_70017


namespace NUMINAMATH_GPT_multiplication_problem_l700_70092

theorem multiplication_problem :
  250 * 24.98 * 2.498 * 1250 = 19484012.5 := by
  sorry

end NUMINAMATH_GPT_multiplication_problem_l700_70092


namespace NUMINAMATH_GPT_percent_students_elected_to_learn_from_home_l700_70000

theorem percent_students_elected_to_learn_from_home (H : ℕ) : 
  (100 - H) / 2 = 30 → H = 40 := 
by
  sorry

end NUMINAMATH_GPT_percent_students_elected_to_learn_from_home_l700_70000


namespace NUMINAMATH_GPT_part1_part2_l700_70030

noncomputable def y (a x : ℝ) : ℝ := a * x^2 + (1 - a) * x + a - 2

-- Part (1)
theorem part1 (a : ℝ) : (∀ x : ℝ, y a x ≥ -2) ↔ a ∈ Set.Ici (1 / 3) :=
sorry

-- Part (2)
theorem part2 (a x : ℝ) :
  (a ≠ 0 → ( a > 0 ↔ -1/a < x ∧ x < 1)
  ∧ (a = 0 ↔ x < 1)
  ∧ (-1 < a ∧ a < 0 ↔ x < 1 ∨ x > -1/a)
  ∧ (a = -1 ↔ x ≠ 1)
  ∧ (a < -1 ↔ x < -1/a ∨ x > 1)) :=
sorry

end NUMINAMATH_GPT_part1_part2_l700_70030


namespace NUMINAMATH_GPT_candies_taken_away_per_incorrect_answer_eq_2_l700_70035

/-- Define constants and assumptions --/
def candy_per_correct := 3
def correct_answers := 7
def extra_correct_answers := 2
def total_candies_if_extra_correct := 31

/-- The number of candies taken away per incorrect answer --/
def x : ℤ := sorry

/-- Prove that the number of candies taken away for each incorrect answer is 2. --/
theorem candies_taken_away_per_incorrect_answer_eq_2 : 
  ∃ x : ℤ, ((correct_answers + extra_correct_answers) * candy_per_correct - total_candies_if_extra_correct = x + (extra_correct_answers * candy_per_correct - (total_candies_if_extra_correct - correct_answers * candy_per_correct))) ∧ x = 2 := 
by
  exists 2
  sorry

end NUMINAMATH_GPT_candies_taken_away_per_incorrect_answer_eq_2_l700_70035


namespace NUMINAMATH_GPT_positive_integer_solutions_condition_l700_70061

theorem positive_integer_solutions_condition (a : ℕ) (A B : ℝ) :
  (∃ (x y z : ℕ), x^2 + y^2 + z^2 = (13 * a)^2 ∧
  x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = (1/4) * (2 * A + B) * (13 * a)^4)
  ↔ A = (1 / 2) * B := 
sorry

end NUMINAMATH_GPT_positive_integer_solutions_condition_l700_70061


namespace NUMINAMATH_GPT_min_absolute_difference_l700_70063

open Int

theorem min_absolute_difference (x y : ℤ) (hx : 0 < x) (hy : 0 < y) (h : x * y - 4 * x + 3 * y = 215) : |x - y| = 15 :=
sorry

end NUMINAMATH_GPT_min_absolute_difference_l700_70063


namespace NUMINAMATH_GPT_camp_boys_count_l700_70034

/-- The ratio of boys to girls and total number of individuals in the camp including teachers
is given, we prove the number of boys is 26. -/
theorem camp_boys_count 
  (b g t : ℕ) -- b = number of boys, g = number of girls, t = number of teachers
  (h1 : b = 3 * (t - 5))  -- boys count related to some integer "t" minus teachers
  (h2 : g = 4 * (t - 5))  -- girls count related to some integer "t" minus teachers
  (total_individuals : t = 65) : 
  b = 26 :=
by
  have h : 3 * (t - 5) + 4 * (t - 5) + 5 = 65 := sorry
  sorry

end NUMINAMATH_GPT_camp_boys_count_l700_70034


namespace NUMINAMATH_GPT_no_such_function_exists_l700_70031

-- Let's define the assumptions as conditions
def condition1 (f : ℝ → ℝ) := ∀ x : ℝ, f (x^2) - (f x)^2 ≥ 1 / 4
def distinct_values (f : ℝ → ℝ) := ∀ x y : ℝ, x ≠ y → f x ≠ f y

-- Now we state the main theorem
theorem no_such_function_exists : ¬ ∃ f : ℝ → ℝ, condition1 f ∧ distinct_values f :=
sorry

end NUMINAMATH_GPT_no_such_function_exists_l700_70031


namespace NUMINAMATH_GPT_find_angle_A_l700_70001

noncomputable def angle_A (a b c S : ℝ) := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))

theorem find_angle_A (a b c S : ℝ) (hb : 0 < b) (hc : 0 < c) (hS : S = (1/2) * b * c * Real.sin (angle_A a b c S)) 
    (h_eq : b^2 + c^2 = (1/3) * a^2 + (4 * Real.sqrt 3 / 3) * S) : 
    angle_A a b c S = π / 6 := by 
  sorry

end NUMINAMATH_GPT_find_angle_A_l700_70001


namespace NUMINAMATH_GPT_number_of_second_graders_l700_70005

-- Define the number of kindergartners
def kindergartners : ℕ := 34

-- Define the number of first graders
def first_graders : ℕ := 48

-- Define the total number of students
def total_students : ℕ := 120

-- Define the proof statement
theorem number_of_second_graders : total_students - (kindergartners + first_graders) = 38 := by
  -- omit the proof details
  sorry

end NUMINAMATH_GPT_number_of_second_graders_l700_70005


namespace NUMINAMATH_GPT_avg_mpg_sum_l700_70049

def first_car_gallons : ℕ := 25
def second_car_gallons : ℕ := 35
def total_miles : ℕ := 2275
def first_car_mpg : ℕ := 40

noncomputable def sum_of_avg_mpg_of_two_cars : ℝ := 76.43

theorem avg_mpg_sum :
  let first_car_miles := (first_car_gallons * first_car_mpg : ℕ)
  let second_car_miles := total_miles - first_car_miles
  let second_car_mpg := (second_car_miles : ℝ) / second_car_gallons
  let sum_avg_mpg := (first_car_mpg : ℝ) + second_car_mpg
  sum_avg_mpg = sum_of_avg_mpg_of_two_cars :=
by
  sorry

end NUMINAMATH_GPT_avg_mpg_sum_l700_70049


namespace NUMINAMATH_GPT_log_fraction_identity_l700_70095

theorem log_fraction_identity (a b : ℝ) (h2 : Real.log 2 = a) (h3 : Real.log 3 = b) :
  (Real.log 12 / Real.log 15) = (2 * a + b) / (1 - a + b) := 
  sorry

end NUMINAMATH_GPT_log_fraction_identity_l700_70095


namespace NUMINAMATH_GPT_sum_of_num_and_denom_l700_70025

-- Define the repeating decimal G
def G : ℚ := 739 / 999

-- State the theorem
theorem sum_of_num_and_denom (a b : ℕ) (hb : b ≠ 0) (h : G = a / b) : a + b = 1738 := sorry

end NUMINAMATH_GPT_sum_of_num_and_denom_l700_70025


namespace NUMINAMATH_GPT_factorize_x_squared_minus_one_l700_70036

theorem factorize_x_squared_minus_one (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_one_l700_70036


namespace NUMINAMATH_GPT_symmetric_circle_equation_l700_70084

noncomputable def equation_of_symmetric_circle (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, C₁ x y ↔ x^2 + y^2 - 4 * x - 8 * y + 19 = 0

theorem symmetric_circle_equation :
  ∀ (C₁ : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop),
  equation_of_symmetric_circle C₁ l →
  (∀ x y, l x y ↔ x + 2 * y - 5 = 0) →
  ∃ C₂ : ℝ → ℝ → Prop, (∀ x y, C₂ x y ↔ x^2 + y^2 = 1) :=
by
  intros C₁ l hC₁ hₗ
  sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l700_70084


namespace NUMINAMATH_GPT_total_conference_games_scheduled_l700_70083

-- Definitions of the conditions
def num_divisions : ℕ := 2
def teams_per_division : ℕ := 6
def intradivision_games_per_pair : ℕ := 3
def interdivision_games_per_pair : ℕ := 2

-- The statement to prove the total number of conference games
theorem total_conference_games_scheduled : 
  (num_divisions * (teams_per_division * (teams_per_division - 1) * intradivision_games_per_pair) / 2) 
  + (teams_per_division * teams_per_division * interdivision_games_per_pair) = 162 := 
by
  sorry

end NUMINAMATH_GPT_total_conference_games_scheduled_l700_70083


namespace NUMINAMATH_GPT_archer_hits_less_than_8_l700_70011

variables (P10 P9 P8 : ℝ)

-- Conditions
def hitting10_ring := P10 = 0.3
def hitting9_ring := P9 = 0.3
def hitting8_ring := P8 = 0.2

-- Statement to prove
theorem archer_hits_less_than_8 (P10 P9 P8 : ℝ)
  (h10 : hitting10_ring P10)
  (h9 : hitting9_ring P9)
  (h8 : hitting8_ring P8)
  (mutually_exclusive: P10 + P9 + P8 <= 1):
  1 - (P10 + P9 + P8) = 0.2 :=
by
  -- Here goes the proof 
  sorry

end NUMINAMATH_GPT_archer_hits_less_than_8_l700_70011


namespace NUMINAMATH_GPT_round_robin_tournament_participant_can_mention_all_l700_70008

theorem round_robin_tournament_participant_can_mention_all :
  ∀ (n : ℕ) (participants : Fin n → Fin n → Prop),
  (∀ i j : Fin n, i ≠ j → (participants i j ∨ participants j i)) →
  (∃ A : Fin n, ∀ (B : Fin n), B ≠ A → (participants A B ∨ ∃ C : Fin n, participants A C ∧ participants C B)) := by
  sorry

end NUMINAMATH_GPT_round_robin_tournament_participant_can_mention_all_l700_70008


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l700_70059

theorem geometric_sequence_first_term 
  (T : ℕ → ℝ) 
  (h1 : T 5 = 243) 
  (h2 : T 6 = 729) 
  (hr : ∃ r : ℝ, ∀ n : ℕ, T n = T 1 * r^(n - 1)) :
  T 1 = 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l700_70059


namespace NUMINAMATH_GPT_number_doubled_is_12_l700_70091

theorem number_doubled_is_12 (A B C D E : ℝ) (h1 : (A + B + C + D + E) / 5 = 6.8)
  (X : ℝ) (h2 : ((A + B + C + D + E - X) + 2 * X) / 5 = 9.2) : X = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_doubled_is_12_l700_70091


namespace NUMINAMATH_GPT_solution_set_of_inequality_l700_70004

theorem solution_set_of_inequality :
  {x : ℝ | (x + 3) * (x - 2) < 0} = {x | -3 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l700_70004


namespace NUMINAMATH_GPT_selection_ways_l700_70093

/-- 
A math interest group in a vocational school consists of 4 boys and 3 girls. 
If 3 students are randomly selected from these 7 students to participate in a math competition, 
and the selection must include both boys and girls, then the number of different ways to select the 
students is 30.
-/
theorem selection_ways (B G : ℕ) (students : ℕ) (selections : ℕ) (condition_boys_girls : B = 4 ∧ G = 3)
  (condition_students : students = B + G) (condition_selections : selections = 3) :
  (B = 4 ∧ G = 3 ∧ students = 7 ∧ selections = 3) → 
  ∃ (res : ℕ), res = 30 :=
by
  sorry

end NUMINAMATH_GPT_selection_ways_l700_70093


namespace NUMINAMATH_GPT_rectangle_area_1600_l700_70012

theorem rectangle_area_1600
  (l w : ℝ)
  (h1 : l = 4 * w)
  (h2 : 2 * l + 2 * w = 200) :
  l * w = 1600 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_1600_l700_70012


namespace NUMINAMATH_GPT_chapters_ratio_l700_70073

theorem chapters_ratio
  (c1 : ℕ) (c2 : ℕ) (total : ℕ) (x : ℕ)
  (h1 : c1 = 20)
  (h2 : c2 = 15)
  (h3 : total = 75)
  (h4 : x = (c1 + 2 * c2) / 2)
  (h5 : c1 + 2 * c2 + x = total) :
  (x : ℚ) / (c1 + 2 * c2 : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_chapters_ratio_l700_70073


namespace NUMINAMATH_GPT_range_of_a_l700_70044

def S : Set ℝ := {x | (x - 2) ^ 2 > 9 }
def T (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 8 }

theorem range_of_a (a : ℝ) : (S ∪ T a) = Set.univ ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l700_70044


namespace NUMINAMATH_GPT_cube_volume_l700_70019

theorem cube_volume (s : ℝ) (h : 12 * s = 96) : s^3 = 512 :=
by
  sorry

end NUMINAMATH_GPT_cube_volume_l700_70019


namespace NUMINAMATH_GPT_solve_for_x_l700_70015

theorem solve_for_x : ∀ (x : ℝ), (x = 3 / 4) →
  3 - (1 / (4 * (1 - x))) = 2 * (1 / (4 * (1 - x))) :=
by
  intros x h
  rw [h]
  sorry

end NUMINAMATH_GPT_solve_for_x_l700_70015


namespace NUMINAMATH_GPT_arrangement_of_mississippi_no_adjacent_s_l700_70006

-- Conditions: The word "MISSISSIPPI" has 11 letters with specific frequencies: 1 M, 4 I's, 4 S's, 2 P's.
-- No two S's can be adjacent.
def ways_to_arrange_mississippi_no_adjacent_s: Nat :=
  let total_non_s_arrangements := Nat.factorial 7 / (Nat.factorial 4 * Nat.factorial 2)
  let gaps_for_s := Nat.choose 8 4
  total_non_s_arrangements * gaps_for_s

theorem arrangement_of_mississippi_no_adjacent_s : ways_to_arrange_mississippi_no_adjacent_s = 7350 :=
by
  unfold ways_to_arrange_mississippi_no_adjacent_s
  sorry

end NUMINAMATH_GPT_arrangement_of_mississippi_no_adjacent_s_l700_70006


namespace NUMINAMATH_GPT_positive_difference_is_zero_l700_70078

-- Definitions based on conditions
def jo_sum (n : ℕ) : ℕ := (n * (n + 1)) / 2

def rounded_to_nearest_5 (x : ℕ) : ℕ :=
  if x % 5 = 0 then x
  else (x / 5) * 5 + (if x % 5 >= 3 then 5 else 0)

def alan_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map rounded_to_nearest_5 |>.sum

-- Theorem based on question and correct answer
theorem positive_difference_is_zero :
  jo_sum 120 - alan_sum 120 = 0 := sorry

end NUMINAMATH_GPT_positive_difference_is_zero_l700_70078


namespace NUMINAMATH_GPT_initial_cd_count_l700_70068

variable (X : ℕ)

theorem initial_cd_count (h1 : (2 / 3 : ℝ) * X + 8 = 22) : X = 21 :=
by
  sorry

end NUMINAMATH_GPT_initial_cd_count_l700_70068


namespace NUMINAMATH_GPT_correctly_calculated_value_l700_70043

theorem correctly_calculated_value (x : ℝ) (hx : x + 0.42 = 0.9) : (x - 0.42) + 0.5 = 0.56 := by
  -- proof to be provided
  sorry

end NUMINAMATH_GPT_correctly_calculated_value_l700_70043


namespace NUMINAMATH_GPT_min_value_l700_70094

-- Defining the conditions
variables {x y z : ℝ}

-- Problem statement translating the conditions
theorem min_value (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 5) : 
  ∃ (minval : ℝ), minval = 36/5 ∧ ∀ w, w = (1/x + 4/y + 9/z) → w ≥ minval :=
by
  sorry

end NUMINAMATH_GPT_min_value_l700_70094


namespace NUMINAMATH_GPT_euler_totient_divisibility_l700_70065

def euler_totient (n : ℕ) : ℕ := sorry

theorem euler_totient_divisibility (n : ℕ) (hn : 0 < n) : 2^(n * (n + 1)) ∣ 32 * euler_totient (2^(2^n) - 1) := 
sorry

end NUMINAMATH_GPT_euler_totient_divisibility_l700_70065


namespace NUMINAMATH_GPT_find_number_l700_70048

theorem find_number (x : ℝ) (h : 0.95 * x - 12 = 178) : x = 200 :=
sorry

end NUMINAMATH_GPT_find_number_l700_70048


namespace NUMINAMATH_GPT_ratio_of_puzzle_times_l700_70099

def total_time := 70
def warmup_time := 10
def remaining_puzzles := 60 / 2

theorem ratio_of_puzzle_times : (remaining_puzzles / warmup_time) = 3 := by
  -- Given Conditions
  have H1 : 70 = 10 + 2 * (60 / 2) := by sorry
  -- Simplification and Calculation
  have H2 : (remaining_puzzles = 30) := by sorry
  -- Ratio Calculation
  have ratio_calculation: (30 / 10) = 3 := by sorry
  exact ratio_calculation

end NUMINAMATH_GPT_ratio_of_puzzle_times_l700_70099


namespace NUMINAMATH_GPT_solve_equation_l700_70033

theorem solve_equation (x y z : ℕ) :
  (∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 2) ↔ (x^2 + 3 * y^2 = 2^z) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l700_70033


namespace NUMINAMATH_GPT_calculate_gross_income_l700_70040
noncomputable def gross_income (net_income : ℝ) (tax_rate : ℝ) : ℝ := net_income / (1 - tax_rate)

theorem calculate_gross_income : gross_income 20000 0.13 = 22989 :=
by
  sorry

end NUMINAMATH_GPT_calculate_gross_income_l700_70040


namespace NUMINAMATH_GPT_solve_quadratic_l700_70016

theorem solve_quadratic : ∀ x, x^2 - 4 * x + 3 = 0 ↔ x = 3 ∨ x = 1 := 
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l700_70016


namespace NUMINAMATH_GPT_r_at_5_l700_70053

def r (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) + x^2 - 1

theorem r_at_5 :
  r 5 = 48 := by
  sorry

end NUMINAMATH_GPT_r_at_5_l700_70053


namespace NUMINAMATH_GPT_larger_fraction_of_two_l700_70032

theorem larger_fraction_of_two (x y : ℚ) (h1 : x + y = 7/8) (h2 : x * y = 1/4) : max x y = 1/2 :=
sorry

end NUMINAMATH_GPT_larger_fraction_of_two_l700_70032


namespace NUMINAMATH_GPT_cosine_sine_difference_identity_l700_70070

theorem cosine_sine_difference_identity :
  (Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)
  - Real.sin (255 * Real.pi / 180) * Real.sin (165 * Real.pi / 180)) = 1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cosine_sine_difference_identity_l700_70070
