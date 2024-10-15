import Mathlib

namespace NUMINAMATH_GPT_time_to_cover_length_correct_l1840_184024

-- Given conditions
def speed_escalator := 20 -- ft/sec
def length_escalator := 210 -- feet
def speed_person := 4 -- ft/sec

-- Time is distance divided by speed
def time_to_cover_length : ℚ :=
  length_escalator / (speed_escalator + speed_person)

theorem time_to_cover_length_correct :
  time_to_cover_length = 8.75 := by
  sorry

end NUMINAMATH_GPT_time_to_cover_length_correct_l1840_184024


namespace NUMINAMATH_GPT_minimal_total_distance_l1840_184037

variable (A B : ℝ) -- Coordinates of houses A and B on a straight road
variable (h_dist : B - A = 50) -- The distance between A and B is 50 meters

-- Define a point X on the road
variable (X : ℝ)

-- Define the function that calculates the total distance from point X to A and B
def total_distance (A B X : ℝ) := abs (X - A) + abs (X - B)

-- The theorem stating that the total distance is minimized if X lies on the line segment AB
theorem minimal_total_distance : A ≤ X ∧ X ≤ B ↔ total_distance A B X = B - A :=
by
  sorry

end NUMINAMATH_GPT_minimal_total_distance_l1840_184037


namespace NUMINAMATH_GPT_arrange_in_order_l1840_184003

noncomputable def x1 : ℝ := Real.sin (Real.cos (3 * Real.pi / 8))
noncomputable def x2 : ℝ := Real.cos (Real.sin (3 * Real.pi / 8))
noncomputable def x3 : ℝ := Real.cos (Real.cos (3 * Real.pi / 8))
noncomputable def x4 : ℝ := Real.sin (Real.sin (3 * Real.pi / 8))

theorem arrange_in_order : 
  x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 := 
by 
  sorry

end NUMINAMATH_GPT_arrange_in_order_l1840_184003


namespace NUMINAMATH_GPT_positive_real_solutions_l1840_184073

noncomputable def x1 := (75 + Real.sqrt 5773) / 2
noncomputable def x2 := (-50 + Real.sqrt 2356) / 2

theorem positive_real_solutions :
  ∀ x : ℝ, 
  0 < x → 
  (1/2 * (4*x^2 - 1) = (x^2 - 75*x - 15) * (x^2 + 50*x + 10)) ↔ 
  (x = x1 ∨ x = x2) :=
by
  sorry

end NUMINAMATH_GPT_positive_real_solutions_l1840_184073


namespace NUMINAMATH_GPT_intersection_points_count_l1840_184076

open Real

theorem intersection_points_count :
  (∃ (x y : ℝ), ((x - ⌊x⌋)^2 + y^2 = x - ⌊x⌋) ∧ (y = 1/3 * x + 1)) →
  (∃ (n : ℕ), n = 8) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_intersection_points_count_l1840_184076


namespace NUMINAMATH_GPT_income_increase_l1840_184014

variable (a : ℝ)

theorem income_increase (h : ∃ a : ℝ, a > 0):
  a * 1.142 = a * 1 + a * 0.142 :=
by
  sorry

end NUMINAMATH_GPT_income_increase_l1840_184014


namespace NUMINAMATH_GPT_fantasy_gala_handshakes_l1840_184007

theorem fantasy_gala_handshakes
    (gremlins imps : ℕ)
    (gremlin_handshakes : ℕ)
    (imp_handshakes : ℕ)
    (imp_gremlin_handshakes : ℕ)
    (total_handshakes : ℕ)
    (h1 : gremlins = 30)
    (h2 : imps = 20)
    (h3 : gremlin_handshakes = (30 * 29) / 2)
    (h4 : imp_handshakes = (20 * 5) / 2)
    (h5 : imp_gremlin_handshakes = 20 * 30)
    (h6 : total_handshakes = gremlin_handshakes + imp_handshakes + imp_gremlin_handshakes) :
    total_handshakes = 1085 := by
    sorry

end NUMINAMATH_GPT_fantasy_gala_handshakes_l1840_184007


namespace NUMINAMATH_GPT_smallest_cut_length_l1840_184021

theorem smallest_cut_length (x : ℕ) (h₁ : 9 ≥ x) (h₂ : 12 ≥ x) (h₃ : 15 ≥ x)
  (h₄ : x ≥ 6) (h₅ : x ≥ 12) (h₆ : x ≥ 18) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_cut_length_l1840_184021


namespace NUMINAMATH_GPT_number_of_distinct_intersection_points_l1840_184089

theorem number_of_distinct_intersection_points :
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 16}
  let line := {p : ℝ × ℝ | p.1 = 4}
  let intersection_points := circle ∩ line
  ∃! p : ℝ × ℝ, p ∈ intersection_points :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_intersection_points_l1840_184089


namespace NUMINAMATH_GPT_six_digit_numbers_with_at_least_one_zero_correct_l1840_184084

def total_six_digit_numbers : ℕ := 9 * 10^5

def total_six_digit_numbers_with_no_zero : ℕ := 9^6

def six_digit_numbers_with_at_least_one_zero : ℕ := 
  total_six_digit_numbers - total_six_digit_numbers_with_no_zero

theorem six_digit_numbers_with_at_least_one_zero_correct : 
  six_digit_numbers_with_at_least_one_zero = 368559 := by
  sorry

end NUMINAMATH_GPT_six_digit_numbers_with_at_least_one_zero_correct_l1840_184084


namespace NUMINAMATH_GPT_remainder_three_l1840_184032

theorem remainder_three (n : ℕ) (h1 : Nat.Prime (n + 3)) (h2 : Nat.Prime (n + 7)) : n % 3 = 1 :=
sorry

end NUMINAMATH_GPT_remainder_three_l1840_184032


namespace NUMINAMATH_GPT_soccer_ball_price_l1840_184030

theorem soccer_ball_price 
  (B S V : ℕ) 
  (h1 : (B + S + V) / 3 = 36)
  (h2 : B = V + 10)
  (h3 : S = V + 8) : 
  S = 38 := 
by 
  sorry

end NUMINAMATH_GPT_soccer_ball_price_l1840_184030


namespace NUMINAMATH_GPT_djibo_age_sum_years_ago_l1840_184035

theorem djibo_age_sum_years_ago (x : ℕ) (h₁: 17 - x + 28 - x = 35) : x = 5 :=
by
  -- proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_djibo_age_sum_years_ago_l1840_184035


namespace NUMINAMATH_GPT_lucy_crayons_correct_l1840_184001

-- Define the number of crayons Willy has.
def willyCrayons : ℕ := 5092

-- Define the number of extra crayons Willy has compared to Lucy.
def extraCrayons : ℕ := 1121

-- Define the number of crayons Lucy has.
def lucyCrayons : ℕ := willyCrayons - extraCrayons

-- Statement to prove
theorem lucy_crayons_correct : lucyCrayons = 3971 := 
by
  -- The proof is omitted as per instructions
  sorry

end NUMINAMATH_GPT_lucy_crayons_correct_l1840_184001


namespace NUMINAMATH_GPT_exists_polynomials_Q_R_l1840_184015

theorem exists_polynomials_Q_R (P : Polynomial ℝ) (hP : ∀ x > 0, P.eval x > 0) :
  ∃ (Q R : Polynomial ℝ), (∀ a, 0 ≤ a → ∀ b, 0 ≤ b → Q.coeff a ≥ 0 ∧ R.coeff b ≥ 0) ∧ ∀ x > 0, P.eval x = (Q.eval x) / (R.eval x) := 
by
  sorry

end NUMINAMATH_GPT_exists_polynomials_Q_R_l1840_184015


namespace NUMINAMATH_GPT_constants_A_B_C_l1840_184027

theorem constants_A_B_C (A B C : ℝ) (h₁ : ∀ x : ℝ, (x^2 + 5 * x - 6) / (x^4 + x^2) = A / x^2 + (B * x + C) / (x^2 + 1)) :
  A = -6 ∧ B = 0 ∧ C = 7 :=
by
  sorry

end NUMINAMATH_GPT_constants_A_B_C_l1840_184027


namespace NUMINAMATH_GPT_mult_mod_7_zero_l1840_184020

theorem mult_mod_7_zero :
  (2007 ≡ 5 [MOD 7]) →
  (2008 ≡ 6 [MOD 7]) →
  (2009 ≡ 0 [MOD 7]) →
  (2010 ≡ 1 [MOD 7]) →
  (2007 * 2008 * 2009 * 2010 ≡ 0 [MOD 7]) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_mult_mod_7_zero_l1840_184020


namespace NUMINAMATH_GPT_ab_value_l1840_184079

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 152) : a * b = 15 := by
  sorry

end NUMINAMATH_GPT_ab_value_l1840_184079


namespace NUMINAMATH_GPT_find_multiplier_l1840_184016

-- Define the numbers and the equation based on the conditions
def n : ℝ := 3.0
def m : ℝ := 7

-- State the problem in Lean 4
theorem find_multiplier : m * n = 3 * n + 12 := by
  -- Specific steps skipped; only structure is needed
  sorry

end NUMINAMATH_GPT_find_multiplier_l1840_184016


namespace NUMINAMATH_GPT_cos_difference_identity_l1840_184069

theorem cos_difference_identity (α : ℝ)
  (h : Real.sin (α + π / 6) + Real.cos α = - (Real.sqrt 3) / 3) :
  Real.cos (π / 6 - α) = -1 / 3 := 
sorry

end NUMINAMATH_GPT_cos_difference_identity_l1840_184069


namespace NUMINAMATH_GPT_simplify_fraction_90_150_l1840_184080

theorem simplify_fraction_90_150 :
  let num := 90
  let denom := 150
  let gcd := 30
  2 * 3^2 * 5 = num →
  2 * 3 * 5^2 = denom →
  (num / gcd) = 3 →
  (denom / gcd) = 5 →
  num / denom = (3 / 5) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_simplify_fraction_90_150_l1840_184080


namespace NUMINAMATH_GPT_sum_of_digits_of_valid_n_eq_seven_l1840_184040

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_valid_n (n : ℕ) : Prop :=
  (500 < n) ∧ (Nat.gcd 70 (n + 150) = 35) ∧ (Nat.gcd (n + 70) 150 = 50)

theorem sum_of_digits_of_valid_n_eq_seven :
  ∃ n : ℕ, is_valid_n n ∧ sum_of_digits n = 7 := by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_valid_n_eq_seven_l1840_184040


namespace NUMINAMATH_GPT_triangle_area_AC_1_AD_BC_circumcircle_l1840_184041

noncomputable def area_triangle_ABC (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_AC_1_AD_BC_circumcircle (A B C D E : ℝ × ℝ) (hAC : dist A C = 1)
  (hAD : dist A D = (2 / 3) * dist A B)
  (hMidE : E = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (hCircum : dist E ((A.1 + C.1) / 2, (A.2 + C.2) / 2) = 1 / 2) :
  area_triangle_ABC A B C = (Real.sqrt 5) / 6 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_AC_1_AD_BC_circumcircle_l1840_184041


namespace NUMINAMATH_GPT_stones_required_to_pave_hall_l1840_184072

noncomputable def hall_length_meters : ℝ := 36
noncomputable def hall_breadth_meters : ℝ := 15
noncomputable def stone_length_dms : ℝ := 4
noncomputable def stone_breadth_dms : ℝ := 5

theorem stones_required_to_pave_hall :
  let hall_length_dms := hall_length_meters * 10
  let hall_breadth_dms := hall_breadth_meters * 10
  let hall_area_dms_squared := hall_length_dms * hall_breadth_dms
  let stone_area_dms_squared := stone_length_dms * stone_breadth_dms
  let number_of_stones := hall_area_dms_squared / stone_area_dms_squared
  number_of_stones = 2700 :=
by
  sorry

end NUMINAMATH_GPT_stones_required_to_pave_hall_l1840_184072


namespace NUMINAMATH_GPT_part_a_part_b_l1840_184038

def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem part_a : ∃ n : ℕ, is_multiple_of_9 n ∧ digit_sum n = 81 ∧ (n / 9) = 111111111 := 
sorry

theorem part_b : ∃ n1 n2 n3 n4 : ℕ,
  is_multiple_of_9 n1 ∧
  is_multiple_of_9 n2 ∧
  is_multiple_of_9 n3 ∧
  is_multiple_of_9 n4 ∧
  digit_sum n1 = 27 ∧ digit_sum n2 = 27 ∧ digit_sum n3 = 27 ∧ digit_sum n4 = 27 ∧
  (n1 / 9) + 1 = (n2 / 9) ∧ 
  (n2 / 9) + 1 = (n3 / 9) ∧ 
  (n3 / 9) + 1 = (n4 / 9) ∧ 
  (n4 / 9) < 1111 := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l1840_184038


namespace NUMINAMATH_GPT_find_f_10_l1840_184094

def f : ℕ → ℚ := sorry
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = f x / (1 + f x)
axiom f_initial : f 1 = 1

theorem find_f_10 : f 10 = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_find_f_10_l1840_184094


namespace NUMINAMATH_GPT_positive_number_sum_square_l1840_184036

theorem positive_number_sum_square (n : ℝ) (h : n^2 + n = 210) (h_pos : n > 0) : n = 14 :=
sorry

end NUMINAMATH_GPT_positive_number_sum_square_l1840_184036


namespace NUMINAMATH_GPT_value_of_expression_l1840_184009

theorem value_of_expression (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 53) :
  x^3 - y^3 - 2 * (x + y) + 10 = 2011 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1840_184009


namespace NUMINAMATH_GPT_oliver_siblings_l1840_184048

structure Child :=
  (name : String)
  (eyeColor : String)
  (hairColor : String)

def oliver := Child.mk "Oliver" "Gray" "Brown"
def charles := Child.mk "Charles" "Gray" "Red"
def diana := Child.mk "Diana" "Green" "Brown"
def olivia := Child.mk "Olivia" "Green" "Red"
def ethan := Child.mk "Ethan" "Green" "Red"
def fiona := Child.mk "Fiona" "Green" "Brown"

def sharesCharacteristic (c1 c2 : Child) : Prop :=
  c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor

def sameFamily (c1 c2 c3 : Child) : Prop :=
  sharesCharacteristic c1 c2 ∧
  sharesCharacteristic c2 c3 ∧
  sharesCharacteristic c3 c1

theorem oliver_siblings : 
  sameFamily oliver charles diana :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_oliver_siblings_l1840_184048


namespace NUMINAMATH_GPT_astroid_area_l1840_184018

-- Definitions coming from the conditions
noncomputable def x (t : ℝ) := 4 * (Real.cos t)^3
noncomputable def y (t : ℝ) := 4 * (Real.sin t)^3

-- The theorem stating the area of the astroid
theorem astroid_area : (∫ t in (0 : ℝ)..(Real.pi / 2), y t * (deriv x t)) * 4 = 24 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_astroid_area_l1840_184018


namespace NUMINAMATH_GPT_raviraj_cycle_distance_l1840_184088

theorem raviraj_cycle_distance :
  ∃ (d : ℝ), d = Real.sqrt ((425: ℝ)^2 + (200: ℝ)^2) ∧ d = 470 := 
by
  sorry

end NUMINAMATH_GPT_raviraj_cycle_distance_l1840_184088


namespace NUMINAMATH_GPT_Kira_breakfast_time_l1840_184065

theorem Kira_breakfast_time :
  let sausages := 3
  let eggs := 6
  let time_per_sausage := 5
  let time_per_egg := 4
  (sausages * time_per_sausage + eggs * time_per_egg) = 39 :=
by
  sorry

end NUMINAMATH_GPT_Kira_breakfast_time_l1840_184065


namespace NUMINAMATH_GPT_natural_eq_rational_exists_diff_l1840_184006

-- Part (a)
theorem natural_eq (x y : ℕ) (h : x^3 + y = y^3 + x) : x = y := 
by sorry

-- Part (b)
theorem rational_exists_diff (x y : ℚ) (h : x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) : ∃ (x y : ℚ), x ≠ y ∧ x^3 + y = y^3 + x := 
by sorry

end NUMINAMATH_GPT_natural_eq_rational_exists_diff_l1840_184006


namespace NUMINAMATH_GPT_sin_value_l1840_184077

theorem sin_value (α : ℝ) (h : Real.cos (α + π / 6) = - (Real.sqrt 2) / 10) : 
  Real.sin (2 * α - π / 6) = 24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_value_l1840_184077


namespace NUMINAMATH_GPT_probability_neither_perfect_square_nor_cube_l1840_184075

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n
noncomputable def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

theorem probability_neither_perfect_square_nor_cube :
  let total_numbers := 200
  let count_squares := (14 : ℕ)  -- Corresponds to the number of perfect squares
  let count_cubes := (5 : ℕ)  -- Corresponds to the number of perfect cubes
  let count_sixth_powers := (2 : ℕ)  -- Corresponds to the number of sixth powers
  let count_ineligible := count_squares + count_cubes - count_sixth_powers
  let count_eligible := total_numbers - count_ineligible
  (count_eligible : ℚ) / (total_numbers : ℚ) = 183 / 200 :=
by sorry

end NUMINAMATH_GPT_probability_neither_perfect_square_nor_cube_l1840_184075


namespace NUMINAMATH_GPT_score_of_29_impossible_l1840_184071

theorem score_of_29_impossible :
  ¬ ∃ (c u w : ℕ), c + u + w = 10 ∧ 3 * c + u = 29 :=
by {
  sorry
}

end NUMINAMATH_GPT_score_of_29_impossible_l1840_184071


namespace NUMINAMATH_GPT_log_one_plus_x_sq_lt_x_sq_l1840_184045

theorem log_one_plus_x_sq_lt_x_sq {x : ℝ} (hx : 0 < x) : 
  Real.log (1 + x^2) < x^2 := 
sorry

end NUMINAMATH_GPT_log_one_plus_x_sq_lt_x_sq_l1840_184045


namespace NUMINAMATH_GPT_hyperbolas_same_asymptotes_l1840_184025

-- Define the given hyperbolas
def hyperbola1 (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
def hyperbola2 (x y M : ℝ) : Prop := (y^2 / 25) - (x^2 / M) = 1

-- The main theorem statement
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, hyperbola1 x y → hyperbola2 x y M) ↔ M = 225/16 :=
by
  sorry

end NUMINAMATH_GPT_hyperbolas_same_asymptotes_l1840_184025


namespace NUMINAMATH_GPT_simplify_expr_l1840_184056

theorem simplify_expr (x : ℕ) (h : x = 2018) : x^2 + 2 * x - x * (x + 1) = x := by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1840_184056


namespace NUMINAMATH_GPT_points_opposite_side_of_line_l1840_184059

theorem points_opposite_side_of_line :
  (∀ a : ℝ, ((2 * 2 - 3 * 1 + a) * (2 * 4 - 3 * 3 + a) < 0) ↔ -1 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_GPT_points_opposite_side_of_line_l1840_184059


namespace NUMINAMATH_GPT_dilation_at_origin_neg3_l1840_184031

-- Define the dilation matrix centered at the origin with scale factor -3
def dilation_matrix (scale_factor : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![scale_factor, 0], ![0, scale_factor]]

-- The theorem stating that a dilation with scale factor -3 results in the specified matrix
theorem dilation_at_origin_neg3 :
  dilation_matrix (-3) = ![![(-3 : ℝ), 0], ![0, -3]] :=
sorry

end NUMINAMATH_GPT_dilation_at_origin_neg3_l1840_184031


namespace NUMINAMATH_GPT_cricketer_wickets_l1840_184044

noncomputable def initial_average (R W : ℝ) : ℝ := R / W

noncomputable def new_average (R W : ℝ) (additional_runs additional_wickets : ℝ) : ℝ :=
  (R + additional_runs) / (W + additional_wickets)

theorem cricketer_wickets (R W : ℝ) 
(h1 : initial_average R W = 12.4) 
(h2 : new_average R W 26 5 = 12.0) : 
  W = 85 :=
sorry

end NUMINAMATH_GPT_cricketer_wickets_l1840_184044


namespace NUMINAMATH_GPT_vehicle_height_limit_l1840_184051

theorem vehicle_height_limit (h : ℝ) (sign : String) (cond : sign = "Height Limit 4.5 meters") : h ≤ 4.5 :=
sorry

end NUMINAMATH_GPT_vehicle_height_limit_l1840_184051


namespace NUMINAMATH_GPT_tan_of_cos_first_quadrant_l1840_184047

-- Define the angle α in the first quadrant and its cosine value
variable (α : ℝ) (h1 : 0 < α ∧ α < π/2) (hcos : Real.cos α = 2 / 3)

-- State the theorem
theorem tan_of_cos_first_quadrant : Real.tan α = Real.sqrt 5 / 2 := 
by
  sorry

end NUMINAMATH_GPT_tan_of_cos_first_quadrant_l1840_184047


namespace NUMINAMATH_GPT_limit_fraction_l1840_184091

theorem limit_fraction :
  ∀ ε > 0, ∃ (N : ℕ), ∀ n ≥ N, |((4 * n - 1) / (2 * n + 1) : ℚ) - 2| < ε := 
  by sorry

end NUMINAMATH_GPT_limit_fraction_l1840_184091


namespace NUMINAMATH_GPT_mass_ratio_speed_ratio_l1840_184017

variable {m1 m2 : ℝ} -- masses of the two balls
variable {V0 V : ℝ} -- velocities before and after collision
variable (h1 : V = 4 * V0) -- speed of m2 is four times that of m1 after collision

theorem mass_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                   (h3 : m1 * V0 = m1 * V + 4 * m2 * V) :
  m2 / m1 = 1 / 2 := sorry

theorem speed_ratio (h2 :  m1 * V0^2 = m1 * V^2 + 16 * m2 * V^2)
                    (h3 : m1 * V0 = m1 * V + 4 * m2 * V)
                    (h4 : m2 / m1 = 1 / 2) :
  V0 / V = 3 := sorry

end NUMINAMATH_GPT_mass_ratio_speed_ratio_l1840_184017


namespace NUMINAMATH_GPT_triangle_side_ratio_eq_one_l1840_184099

theorem triangle_side_ratio_eq_one
    (a b c C : ℝ)
    (h1 : a = 2 * b * Real.cos C)
    (cosine_rule : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
    (b / c = 1) := 
by 
    sorry

end NUMINAMATH_GPT_triangle_side_ratio_eq_one_l1840_184099


namespace NUMINAMATH_GPT_min_value_reciprocal_sum_l1840_184098

theorem min_value_reciprocal_sum (m n : ℝ) (hmn : m + n = 1) (hm_pos : m > 0) (hn_pos : n > 0) :
  1 / m + 1 / n ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_reciprocal_sum_l1840_184098


namespace NUMINAMATH_GPT_number_of_cows_brought_l1840_184055

/--
A certain number of cows and 10 goats are brought for Rs. 1500. 
If the average price of a goat is Rs. 70, and the average price of a cow is Rs. 400, 
then the number of cows brought is 2.
-/
theorem number_of_cows_brought : 
  ∃ c : ℕ, ∃ g : ℕ, g = 10 ∧ (70 * g + 400 * c = 1500) ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_number_of_cows_brought_l1840_184055


namespace NUMINAMATH_GPT_find_integer_pairs_l1840_184096

theorem find_integer_pairs :
  ∃ (n : ℤ) (a : ℤ) (b : ℤ),
    (∀ a b : ℤ, (∃ m : ℤ, a^2 - 4*b = m^2) ∧ (∃ k : ℤ, b^2 - 4*a = k^2) ↔ 
    (a = 0 ∧ ∃ n : ℤ, b = n^2) ∨
    (b = 0 ∧ ∃ n : ℤ, a = n^2) ∨
    (b > 0 ∧ ∃ a : ℤ, a^2 > 0 ∧ b = -1 - a) ∨
    (a > 0 ∧ ∃ b : ℤ, b^2 > 0 ∧ a = -1 - b) ∨
    (a = 4 ∧ b = 4) ∨
    (a = 5 ∧ b = 6) ∨
    (a = 6 ∧ b = 5)) :=
sorry

end NUMINAMATH_GPT_find_integer_pairs_l1840_184096


namespace NUMINAMATH_GPT_sin_960_eq_sqrt3_over_2_neg_l1840_184011

-- Conditions
axiom sine_periodic : ∀ θ, Real.sin (θ + 360 * Real.pi / 180) = Real.sin θ

-- Theorem to prove
theorem sin_960_eq_sqrt3_over_2_neg : Real.sin (960 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_sin_960_eq_sqrt3_over_2_neg_l1840_184011


namespace NUMINAMATH_GPT_train_crossing_signal_pole_l1840_184057

theorem train_crossing_signal_pole
  (length_train : ℕ)
  (same_length_platform : ℕ)
  (time_crossing_platform : ℕ)
  (h_train_platform : length_train = 420)
  (h_platform : same_length_platform = 420)
  (h_time_platform : time_crossing_platform = 60) : 
  (length_train / (length_train + same_length_platform / time_crossing_platform)) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_train_crossing_signal_pole_l1840_184057


namespace NUMINAMATH_GPT_field_dimension_area_l1840_184092

theorem field_dimension_area (m : ℝ) : (3 * m + 8) * (m - 3) = 120 → m = 7 :=
by
  sorry

end NUMINAMATH_GPT_field_dimension_area_l1840_184092


namespace NUMINAMATH_GPT_range_of_values_for_k_l1840_184002

theorem range_of_values_for_k (k : ℝ) (h : k ≠ 0) :
  (1 : ℝ) ∈ { x : ℝ | k^2 * x^2 - 6 * k * x + 8 ≥ 0 } ↔ (k ≥ 4 ∨ k ≤ 2) := 
by
  -- proof 
  sorry

end NUMINAMATH_GPT_range_of_values_for_k_l1840_184002


namespace NUMINAMATH_GPT_correct_operation_l1840_184083

theorem correct_operation (x : ℝ) (f : ℝ → ℝ) (h : ∀ x, (x / 10) = 0.01 * f x) : 
  f x = 10 * x :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l1840_184083


namespace NUMINAMATH_GPT_cube_inequality_of_greater_l1840_184005

variable (a b : ℝ)

theorem cube_inequality_of_greater (h : a > b) : a^3 > b^3 :=
sorry

end NUMINAMATH_GPT_cube_inequality_of_greater_l1840_184005


namespace NUMINAMATH_GPT_linda_age_l1840_184063

theorem linda_age
  (j k l : ℕ)       -- Ages of Jane, Kevin, and Linda respectively
  (h1 : j + k + l = 36)    -- Condition 1: j + k + l = 36
  (h2 : l - 3 = j)         -- Condition 2: l - 3 = j
  (h3 : k + 4 = (1 / 2 : ℝ) * (l + 4))  -- Condition 3: k + 4 = 1/2 * (l + 4)
  : l = 16 := 
sorry

end NUMINAMATH_GPT_linda_age_l1840_184063


namespace NUMINAMATH_GPT_students_in_either_but_not_both_l1840_184022

-- Definitions and conditions
def both : ℕ := 18
def geom : ℕ := 35
def only_stats : ℕ := 16

-- Correct answer to prove
def total_not_both : ℕ := geom - both + only_stats

theorem students_in_either_but_not_both : total_not_both = 33 := by
  sorry

end NUMINAMATH_GPT_students_in_either_but_not_both_l1840_184022


namespace NUMINAMATH_GPT_range_of_a_l1840_184012

noncomputable def f (x : ℝ) : ℝ := x + 1 / x
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x - a / x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x - g x a

theorem range_of_a (e : ℝ) (a : ℝ) (H : ∀ x ∈ Set.Icc 1 e, f x ≥ g x a) :
  -2 ≤ a ∧ a ≤ (2 * e) / (e - 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1840_184012


namespace NUMINAMATH_GPT_no_odd_integer_trinomial_has_root_1_over_2022_l1840_184082

theorem no_odd_integer_trinomial_has_root_1_over_2022 :
  ¬ ∃ (a b c : ℤ), (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1 ∧ (a * (1 / 2022)^2 + b * (1 / 2022) + c = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_odd_integer_trinomial_has_root_1_over_2022_l1840_184082


namespace NUMINAMATH_GPT_sum_of_squares_l1840_184046

theorem sum_of_squares (a b n : ℕ) (h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2) : 
  ∃ e f : ℕ, a^2 + n * b^2 = e^2 + f^2 :=
by
  sorry

-- Theorem parameters and logical flow explained:

-- a, b, n : ℕ                  -- Natural number inputs
-- h : ∃ k : ℕ, a^2 + 2 * n * b^2 = k^2  -- Condition given in the problem that a^2 + 2nb^2 is a perfect square
-- Prove that there exist natural numbers e and f such that a^2 + nb^2 = e^2 + f^2

end NUMINAMATH_GPT_sum_of_squares_l1840_184046


namespace NUMINAMATH_GPT_only_ten_perfect_square_l1840_184026

theorem only_ten_perfect_square (n : ℤ) :
  ∃ k : ℤ, n^4 + 6 * n^3 + 11 * n^2 + 3 * n + 31 = k^2 ↔ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_only_ten_perfect_square_l1840_184026


namespace NUMINAMATH_GPT_abs_sum_inequality_solution_l1840_184054

theorem abs_sum_inequality_solution (x : ℝ) : 
  (|x - 5| + |x + 1| < 8) ↔ (-2 < x ∧ x < 6) :=
sorry

end NUMINAMATH_GPT_abs_sum_inequality_solution_l1840_184054


namespace NUMINAMATH_GPT_distance_to_station_is_6_l1840_184013

noncomputable def distance_man_walks (walking_speed1 walking_speed2 time_diff: ℝ) : ℝ :=
  let D := (time_diff * walking_speed1 * walking_speed2) / (walking_speed1 - walking_speed2)
  D

theorem distance_to_station_is_6 :
  distance_man_walks 5 6 (12 / 60) = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_station_is_6_l1840_184013


namespace NUMINAMATH_GPT_find_a_minus_inverse_l1840_184033

-- Definition for the given condition
def condition (a : ℝ) : Prop := a + a⁻¹ = 6

-- Definition for the target value to be proven
def target_value (x : ℝ) : Prop := x = 4 * Real.sqrt 2 ∨ x = -4 * Real.sqrt 2

-- Theorem statement to be proved
theorem find_a_minus_inverse (a : ℝ) (ha : condition a) : target_value (a - a⁻¹) :=
by
  sorry

end NUMINAMATH_GPT_find_a_minus_inverse_l1840_184033


namespace NUMINAMATH_GPT_athlete_B_more_stable_l1840_184010

variable (average_scores_A average_scores_B : ℝ)
variable (s_A_squared s_B_squared : ℝ)

theorem athlete_B_more_stable
  (h_avg : average_scores_A = average_scores_B)
  (h_var_A : s_A_squared = 1.43)
  (h_var_B : s_B_squared = 0.82) :
  s_A_squared > s_B_squared :=
by 
  rw [h_var_A, h_var_B]
  sorry

end NUMINAMATH_GPT_athlete_B_more_stable_l1840_184010


namespace NUMINAMATH_GPT_incorrect_statement_l1840_184070

noncomputable def first_line_of_defense := "Skin and mucous membranes"
noncomputable def second_line_of_defense := "Antimicrobial substances and phagocytic cells in body fluids"
noncomputable def third_line_of_defense := "Immune organs and immune cells"
noncomputable def non_specific_immunity := "First and second line of defense"
noncomputable def specific_immunity := "Third line of defense"
noncomputable def d_statement := "The defensive actions performed by the three lines of defense in the human body are called non-specific immunity"

theorem incorrect_statement : d_statement ≠ specific_immunity ∧ d_statement ≠ non_specific_immunity := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_l1840_184070


namespace NUMINAMATH_GPT_total_clouds_counted_l1840_184081

def clouds_counted (carson_clouds : ℕ) (brother_factor : ℕ) : ℕ :=
  carson_clouds + (carson_clouds * brother_factor)

theorem total_clouds_counted (carson_clouds brother_factor total_clouds : ℕ) 
  (h₁ : carson_clouds = 6) (h₂ : brother_factor = 3) (h₃ : total_clouds = 24) :
  clouds_counted carson_clouds brother_factor = total_clouds :=
by
  sorry

end NUMINAMATH_GPT_total_clouds_counted_l1840_184081


namespace NUMINAMATH_GPT_position_of_seventeen_fifteen_in_sequence_l1840_184062

theorem position_of_seventeen_fifteen_in_sequence :
  ∃ n : ℕ, (17 : ℚ) / 15 = (n + 3 : ℚ) / (n + 1) :=
sorry

end NUMINAMATH_GPT_position_of_seventeen_fifteen_in_sequence_l1840_184062


namespace NUMINAMATH_GPT_cos_value_l1840_184052

-- Given condition
axiom sin_condition (α : ℝ) : Real.sin (Real.pi / 6 + α) = 2 / 3

-- The theorem we need to prove
theorem cos_value (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = 2 / 3) : 
  Real.cos (Real.pi / 3 - α) = 2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_cos_value_l1840_184052


namespace NUMINAMATH_GPT_max_correct_answers_l1840_184019

-- Definitions based on the conditions
def total_problems : ℕ := 12
def points_per_correct : ℕ := 6
def points_per_incorrect : ℕ := 3
def max_score : ℤ := 37 -- Final score, using ℤ to handle potential negatives in deducting points

-- The statement to prove
theorem max_correct_answers :
  ∃ (c w : ℕ), c + w = total_problems ∧ points_per_correct * c - points_per_incorrect * (total_problems - c) = max_score ∧ c = 8 :=
by
  sorry

end NUMINAMATH_GPT_max_correct_answers_l1840_184019


namespace NUMINAMATH_GPT_min_value_a_squared_plus_b_squared_l1840_184039

theorem min_value_a_squared_plus_b_squared :
  ∃ (a b : ℝ), (b = 3 * a - 6) → (a^2 + b^2 = 18 / 5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_a_squared_plus_b_squared_l1840_184039


namespace NUMINAMATH_GPT_min_additional_packs_needed_l1840_184097

-- Defining the problem conditions
def total_sticker_packs : ℕ := 40
def packs_per_basket : ℕ := 7

-- The statement to prove
theorem min_additional_packs_needed : 
  ∃ (additional_packs : ℕ), 
    (total_sticker_packs + additional_packs) % packs_per_basket = 0 ∧ 
    (total_sticker_packs + additional_packs) / packs_per_basket = 6 ∧ 
    additional_packs = 2 :=
by 
  sorry

end NUMINAMATH_GPT_min_additional_packs_needed_l1840_184097


namespace NUMINAMATH_GPT_chess_tournament_participants_l1840_184093

theorem chess_tournament_participants (n : ℕ) (h : n * (n - 1) / 2 = 378) : n = 28 :=
sorry

end NUMINAMATH_GPT_chess_tournament_participants_l1840_184093


namespace NUMINAMATH_GPT_quadratic_relationship_l1840_184085

theorem quadratic_relationship (a b c : ℝ) (α : ℝ) (h₁ : α + α^2 = -b / a) (h₂ : α^3 = c / a) : b^2 = 3 * a * c + c^2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_relationship_l1840_184085


namespace NUMINAMATH_GPT_increasing_function_in_interval_l1840_184000

noncomputable def y₁ (x : ℝ) : ℝ := abs (x + 1)
noncomputable def y₂ (x : ℝ) : ℝ := 3 - x
noncomputable def y₃ (x : ℝ) : ℝ := 1 / x
noncomputable def y₄ (x : ℝ) : ℝ := -x^2 + 4

theorem increasing_function_in_interval : ∀ x, (0 < x ∧ x < 1) → 
  y₁ x > y₁ (x - 0.1) ∧ y₂ x < y₂ (x - 0.1) ∧ y₃ x < y₃ (x - 0.1) ∧ y₄ x < y₄ (x - 0.1) :=
by {
  sorry
}

end NUMINAMATH_GPT_increasing_function_in_interval_l1840_184000


namespace NUMINAMATH_GPT_max_ratio_1099_l1840_184049

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_ratio_1099 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 → (sum_of_digits n : ℚ) / n ≤ (sum_of_digits 1099 : ℚ) / 1099 :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_max_ratio_1099_l1840_184049


namespace NUMINAMATH_GPT_expression_in_parentheses_l1840_184061

theorem expression_in_parentheses (x y : ℝ) (h₀ : x ≠ 0) (h₁ : y ≠ 0) :
  ∃ expr : ℝ, xy * expr = -x^3 * y^2 ∧ expr = -x^2 * y :=
by
  sorry

end NUMINAMATH_GPT_expression_in_parentheses_l1840_184061


namespace NUMINAMATH_GPT_min_value_of_b_plus_2_div_a_l1840_184042

theorem min_value_of_b_plus_2_div_a (a : ℝ) (b : ℝ) (h₁ : 0 < a) 
  (h₂ : ∀ x : ℝ, 0 < x → (ax - 1) * (x^2 + bx - 4) ≥ 0) : 
  ∃ a' b', (a' > 0 ∧ b' = 4 * a' - 1 / a') ∧ b' + 2 / a' = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_b_plus_2_div_a_l1840_184042


namespace NUMINAMATH_GPT_cost_price_l1840_184053

namespace ClothingDiscount

variables (x : ℝ)

def loss_condition (x : ℝ) : ℝ := 0.5 * x + 20
def profit_condition (x : ℝ) : ℝ := 0.8 * x - 40

def marked_price := { x : ℝ // loss_condition x = profit_condition x }

noncomputable def clothing_price : marked_price := 
    ⟨200, sorry⟩

theorem cost_price : loss_condition 200 = 120 :=
sorry

end ClothingDiscount

end NUMINAMATH_GPT_cost_price_l1840_184053


namespace NUMINAMATH_GPT_smallest_positive_integer_between_101_and_200_l1840_184086

theorem smallest_positive_integer_between_101_and_200 :
  ∃ n : ℕ, n > 1 ∧ n % 6 = 1 ∧ n % 7 = 1 ∧ n % 8 = 1 ∧ 101 ≤ n ∧ n ≤ 200 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_between_101_and_200_l1840_184086


namespace NUMINAMATH_GPT_oxen_eat_as_much_as_buffaloes_or_cows_l1840_184028

theorem oxen_eat_as_much_as_buffaloes_or_cows
  (B C O : ℝ)
  (h1 : 3 * B = 4 * C)
  (h2 : (15 * B + 8 * O + 24 * C) * 36 = (30 * B + 8 * O + 64 * C) * 18) :
  3 * B = 4 * O :=
by sorry

end NUMINAMATH_GPT_oxen_eat_as_much_as_buffaloes_or_cows_l1840_184028


namespace NUMINAMATH_GPT_triangle_side_lengths_inequality_iff_l1840_184078

theorem triangle_side_lengths_inequality_iff :
  {x : ℕ | 7 < x^2 ∧ x^2 < 17} = {3, 4} :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_inequality_iff_l1840_184078


namespace NUMINAMATH_GPT_kaleb_boxes_required_l1840_184068

/-- Kaleb's Games Packing Problem -/
theorem kaleb_boxes_required (initial_games sold_games box_capacity : ℕ) (h1 : initial_games = 76) (h2 : sold_games = 46) (h3 : box_capacity = 5) :
  ((initial_games - sold_games) / box_capacity) = 6 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_kaleb_boxes_required_l1840_184068


namespace NUMINAMATH_GPT_distance_from_center_to_line_l1840_184087

-- Define the conditions 
def circle_polar_eq (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def line_polar_eq (ρ θ : ℝ) : Prop := ρ * Real.sin θ + 2 * ρ * Real.cos θ = 1

-- Define the assertion that we want to prove
theorem distance_from_center_to_line (ρ θ : ℝ) 
  (h_circle: circle_polar_eq ρ θ) 
  (h_line: line_polar_eq ρ θ) : 
  ∃ d : ℝ, d = (Real.sqrt 5) / 5 := 
sorry

end NUMINAMATH_GPT_distance_from_center_to_line_l1840_184087


namespace NUMINAMATH_GPT_num_arithmetic_sequences_l1840_184029

theorem num_arithmetic_sequences (d : ℕ) (x : ℕ)
  (h_sum : 8 * x + 28 * d = 1080)
  (h_no180 : ∀ i, x + i * d ≠ 180)
  (h_pos : ∀ i, 0 < x + i * d)
  (h_less160 : ∀ i, x + i * d < 160)
  (h_not_equiangular : d ≠ 0) :
  ∃ n : ℕ, n = 3 :=
by sorry

end NUMINAMATH_GPT_num_arithmetic_sequences_l1840_184029


namespace NUMINAMATH_GPT_first_box_weight_l1840_184060

theorem first_box_weight (X : ℕ) 
  (h1 : 11 + 5 + X = 18) : X = 2 := 
by
  sorry

end NUMINAMATH_GPT_first_box_weight_l1840_184060


namespace NUMINAMATH_GPT_eval_expression_l1840_184067

def x : ℤ := 18 / 3 * 7^2 - 80 + 4 * 7

theorem eval_expression : -x = -242 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l1840_184067


namespace NUMINAMATH_GPT_at_least_one_inequality_holds_l1840_184095

theorem at_least_one_inequality_holds
    (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) :
    (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
by
  sorry

end NUMINAMATH_GPT_at_least_one_inequality_holds_l1840_184095


namespace NUMINAMATH_GPT__l1840_184034

lemma right_triangle_angles (AB BC AC : ℝ) (α β : ℝ)
  (h1 : AB = 1) 
  (h2 : BC = Real.sin α)
  (h3 : AC = Real.cos α)
  (h4 : AB^2 = BC^2 + AC^2) -- Pythagorean theorem for the right triangle
  (h5 : α = (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1))) :
  β = 90 - (1 / 2) * Real.arcsin (2 * (Real.sqrt 2 - 1)) :=
sorry

end NUMINAMATH_GPT__l1840_184034


namespace NUMINAMATH_GPT_solve_m_n_l1840_184023

theorem solve_m_n (m n : ℤ) (h : m^2 - 2 * m * n + 2 * n^2 - 8 * n + 16 = 0) : m = 4 ∧ n = 4 :=
sorry

end NUMINAMATH_GPT_solve_m_n_l1840_184023


namespace NUMINAMATH_GPT_first_part_eq_19_l1840_184050

theorem first_part_eq_19 (x y : ℕ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 :=
by sorry

end NUMINAMATH_GPT_first_part_eq_19_l1840_184050


namespace NUMINAMATH_GPT_correct_operation_l1840_184090

variable (a b m : ℕ)

theorem correct_operation :
  (3 * a^2 * 2 * a^2 ≠ 5 * a^2) ∧
  ((2 * a^2)^3 = 8 * a^6) ∧
  (m^6 / m^3 ≠ m^2) ∧
  ((a + b)^2 ≠ a^2 + b^2) →
  ((2 * a^2)^3 = 8 * a^6) :=
by
  intros
  sorry

end NUMINAMATH_GPT_correct_operation_l1840_184090


namespace NUMINAMATH_GPT_complex_pow_sub_eq_zero_l1840_184074

namespace complex_proof

open Complex

def i : ℂ := Complex.I -- Defining i to be the imaginary unit

-- Stating the conditions as definitions
def condition := i^2 = -1

-- Stating the goal as a theorem
theorem complex_pow_sub_eq_zero (cond : condition) :
  (1 + 2 * i) ^ 24 - (1 - 2 * i) ^ 24 = 0 := 
by
  sorry

end complex_proof

end NUMINAMATH_GPT_complex_pow_sub_eq_zero_l1840_184074


namespace NUMINAMATH_GPT_find_principal_l1840_184008

/-- Given that the simple interest SI is Rs. 90, the rate R is 3.5 percent, and the time T is 4 years,
prove that the principal P is approximately Rs. 642.86 using the simple interest formula. -/
theorem find_principal
  (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) 
  (h1 : SI = 90) (h2 : R = 3.5) (h3 : T = 4) 
  : P = 90 * 100 / (3.5 * 4) :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l1840_184008


namespace NUMINAMATH_GPT_a_value_intersection_l1840_184064

open Set

noncomputable def a_intersection_problem (a : ℝ) : Prop :=
  let A := { x : ℝ | x^2 < a^2 }
  let B := { x : ℝ | 1 < x ∧ x < 3 }
  let C := { x : ℝ | 1 < x ∧ x < 2 }
  A ∩ B = C → (a = 2 ∨ a = -2)

-- The theorem statement corresponding to the problem
theorem a_value_intersection (a : ℝ) :
  a_intersection_problem a :=
sorry

end NUMINAMATH_GPT_a_value_intersection_l1840_184064


namespace NUMINAMATH_GPT_michelle_travel_distance_l1840_184058

-- Define the conditions
def initial_fee : ℝ := 2
def charge_per_mile : ℝ := 2.5
def total_paid : ℝ := 12

-- Define the theorem to prove the distance Michelle traveled
theorem michelle_travel_distance : (total_paid - initial_fee) / charge_per_mile = 4 := by
  sorry

end NUMINAMATH_GPT_michelle_travel_distance_l1840_184058


namespace NUMINAMATH_GPT_Zoe_given_card_6_l1840_184004

-- Define the cards and friends
def cards : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
def friends : List String := ["Eliza", "Miguel", "Naomi", "Ivan", "Zoe"]

-- Define scores 
def scores (name : String) : ℕ :=
  match name with
  | "Eliza"  => 15
  | "Miguel" => 11
  | "Naomi"  => 9
  | "Ivan"   => 13
  | "Zoe"    => 10
  | _ => 0

-- Each friend is given a pair of cards
def cardAssignments (name : String) : List (ℕ × ℕ) :=
  match name with
  | "Eliza"  => [(6,9), (7,8), (5,10), (4,11), (3,12)]
  | "Miguel" => [(1,10), (2,9), (3,8), (4,7), (5,6)]
  | "Naomi"  => [(1,8), (2,7), (3,6), (4,5)]
  | "Ivan"   => [(1,12), (2,11), (3,10), (4,9), (5,8), (6,7)]
  | "Zoe"    => [(1,9), (2,8), (3,7), (4,6)]
  | _ => []

-- The proof statement
theorem Zoe_given_card_6 : ∃ c1 c2, (c1, c2) ∈ cardAssignments "Zoe" ∧ (c1 = 6 ∨ c2 = 6)
:= by
  sorry -- Proof omitted as per the instructions

end NUMINAMATH_GPT_Zoe_given_card_6_l1840_184004


namespace NUMINAMATH_GPT_tetrahedron_distance_sum_eq_l1840_184043

-- Defining the necessary conditions
variables {V K : ℝ}
variables {S_1 S_2 S_3 S_4 H_1 H_2 H_3 H_4 : ℝ}

axiom ratio_eq (i : ℕ) (Si : ℝ) (K : ℝ) : (Si / i = K)
axiom volume_eq : S_1 * H_1 + S_2 * H_2 + S_3 * H_3 + S_4 * H_4 = 3 * V

-- Main theorem stating that the desired result holds under the given conditions
theorem tetrahedron_distance_sum_eq :
  H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4 = 3 * V / K :=
by
have h1 : S_1 = K * 1 := by sorry
have h2 : S_2 = K * 2 := by sorry
have h3 : S_3 = K * 3 := by sorry
have h4 : S_4 = K * 4 := by sorry
have sum_eq : K * (H_1 + 2 * H_2 + 3 * H_3 + 4 * H_4) = 3 * V := by sorry
exact sorry

end NUMINAMATH_GPT_tetrahedron_distance_sum_eq_l1840_184043


namespace NUMINAMATH_GPT_apples_count_l1840_184066

theorem apples_count : (23 - 20 + 6 = 9) :=
by
  sorry

end NUMINAMATH_GPT_apples_count_l1840_184066
