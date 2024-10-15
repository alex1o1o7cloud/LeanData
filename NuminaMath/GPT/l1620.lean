import Mathlib

namespace NUMINAMATH_GPT_students_in_each_class_l1620_162010

-- Define the conditions
def sheets_per_student : ℕ := 5
def total_sheets : ℕ := 400
def number_of_classes : ℕ := 4

-- Define the main proof theorem
theorem students_in_each_class : (total_sheets / sheets_per_student) / number_of_classes = 20 := by
  sorry -- Proof goes here

end NUMINAMATH_GPT_students_in_each_class_l1620_162010


namespace NUMINAMATH_GPT_deers_distribution_l1620_162002

theorem deers_distribution (a_1 d a_2 a_5 : ℚ) 
  (h1 : a_2 = a_1 + d)
  (h2 : 5 * a_1 + 10 * d = 5)
  (h3 : a_2 = 2 / 3) :
  a_5 = 1 / 3 :=
sorry

end NUMINAMATH_GPT_deers_distribution_l1620_162002


namespace NUMINAMATH_GPT_cos_inequality_l1620_162019

open Real

-- Given angles of a triangle A, B, C

theorem cos_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hTriangle : A + B + C = π) :
  1 / (1 + cos B ^ 2 + cos C ^ 2) + 1 / (1 + cos C ^ 2 + cos A ^ 2) + 1 / (1 + cos A ^ 2 + cos B ^ 2) ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_inequality_l1620_162019


namespace NUMINAMATH_GPT_P_plus_Q_is_expected_l1620_162038

-- defining the set P
def P : Set ℝ := { x | x ^ 2 - 3 * x - 4 ≤ 0 }

-- defining the set Q
def Q : Set ℝ := { x | x ^ 2 - 2 * x - 15 > 0 }

-- defining the set P + Q
def P_plus_Q : Set ℝ := { x | (x ∈ P ∨ x ∈ Q) ∧ ¬(x ∈ P ∧ x ∈ Q) }

-- the expected result
def expected_P_plus_Q : Set ℝ := { x | x < -3 } ∪ { x | -1 ≤ x ∧ x ≤ 4 } ∪ { x | x > 5 }

-- theorem stating that P + Q equals the expected result
theorem P_plus_Q_is_expected : P_plus_Q = expected_P_plus_Q := by
  sorry

end NUMINAMATH_GPT_P_plus_Q_is_expected_l1620_162038


namespace NUMINAMATH_GPT_find_k_l1620_162061

theorem find_k (x y k : ℝ) (h1 : x = 1) (h2 : y = 4) (h3 : k * x + y = 3) : k = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1620_162061


namespace NUMINAMATH_GPT_water_needed_in_pints_l1620_162082

-- Define the input data
def parts_water : ℕ := 5
def parts_lemon : ℕ := 2
def pints_per_gallon : ℕ := 8
def total_gallons : ℕ := 3

-- Define the total parts of the mixture
def total_parts : ℕ := parts_water + parts_lemon

-- Define the total pints of lemonade
def total_pints : ℕ := total_gallons * pints_per_gallon

-- Define the pints per part of the mixture
def pints_per_part : ℚ := total_pints / total_parts

-- Define the total pints of water needed
def pints_water : ℚ := parts_water * pints_per_part

-- The theorem stating what we need to prove
theorem water_needed_in_pints : pints_water = 17 + 1 / 7 := by
  sorry

end NUMINAMATH_GPT_water_needed_in_pints_l1620_162082


namespace NUMINAMATH_GPT_extreme_point_of_f_l1620_162022

open Real

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - log x

theorem extreme_point_of_f : ∃ x₀ > 0, f x₀ = f (sqrt 3 / 3) ∧ 
  (∀ x < sqrt 3 / 3, f x > f (sqrt 3 / 3)) ∧
  (∀ x > sqrt 3 / 3, f x > f (sqrt 3 / 3)) :=
sorry

end NUMINAMATH_GPT_extreme_point_of_f_l1620_162022


namespace NUMINAMATH_GPT_find_f_zero_function_decreasing_find_range_x_l1620_162047

noncomputable def f : ℝ → ℝ := sorry

-- Define the main conditions as hypotheses
axiom additivity : ∀ x1 x2 : ℝ, f (x1 + x2) = f x1 + f x2
axiom negativity : ∀ x : ℝ, x > 0 → f x < 0

-- First theorem: proving f(0) = 0
theorem find_f_zero : f 0 = 0 := sorry

-- Second theorem: proving the function is decreasing over (-∞, ∞)
theorem function_decreasing : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2 := sorry

-- Third theorem: finding the range of x such that f(x) + f(2-3x) < 0
theorem find_range_x (x : ℝ) : f x + f (2 - 3 * x) < 0 → x < 1 := sorry

end NUMINAMATH_GPT_find_f_zero_function_decreasing_find_range_x_l1620_162047


namespace NUMINAMATH_GPT_difference_between_x_and_y_is_36_l1620_162037

theorem difference_between_x_and_y_is_36 (x y : ℤ) (h1 : x + y = 20) (h2 : x = 28) : x - y = 36 := 
by 
  sorry

end NUMINAMATH_GPT_difference_between_x_and_y_is_36_l1620_162037


namespace NUMINAMATH_GPT_percentage_dogs_and_video_games_l1620_162007

theorem percentage_dogs_and_video_games (total_students : ℕ)
  (students_dogs_movies : ℕ)
  (students_prefer_dogs : ℕ) :
  total_students = 30 →
  students_dogs_movies = 3 →
  students_prefer_dogs = 18 →
  (students_prefer_dogs - students_dogs_movies) * 100 / total_students = 50 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_dogs_and_video_games_l1620_162007


namespace NUMINAMATH_GPT_compute_expression_l1620_162014

theorem compute_expression (x : ℝ) (h : x = 8) : 
  (x^6 - 64 * x^3 + 1024) / (x^3 - 16) = 480 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_compute_expression_l1620_162014


namespace NUMINAMATH_GPT_grain_spilled_l1620_162085

def original_grain : ℕ := 50870
def remaining_grain : ℕ := 918

theorem grain_spilled : (original_grain - remaining_grain) = 49952 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_grain_spilled_l1620_162085


namespace NUMINAMATH_GPT_sum_of_ages_l1620_162063

variable (P_years Q_years : ℝ) (D_years : ℝ)

-- conditions
def condition_1 : Prop := Q_years = 37.5
def condition_2 : Prop := P_years = 3 * (Q_years - D_years)
def condition_3 : Prop := P_years - Q_years = D_years

-- statement to prove
theorem sum_of_ages (h1 : condition_1 Q_years) (h2 : condition_2 P_years Q_years D_years) (h3 : condition_3 P_years Q_years D_years) :
  P_years + Q_years = 93.75 :=
by sorry

end NUMINAMATH_GPT_sum_of_ages_l1620_162063


namespace NUMINAMATH_GPT_simplify_expression_l1620_162031

variable (x : Int)

theorem simplify_expression : 3 * x + 5 * x + 7 * x = 15 * x :=
  by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1620_162031


namespace NUMINAMATH_GPT_other_root_l1620_162052

theorem other_root (k : ℝ) : 
  5 * (2:ℝ)^2 + k * (2:ℝ) - 8 = 0 → 
  ∃ q : ℝ, 5 * q^2 + k * q - 8 = 0 ∧ q ≠ 2 ∧ q = -4/5 :=
by {
  sorry
}

end NUMINAMATH_GPT_other_root_l1620_162052


namespace NUMINAMATH_GPT_maximum_p_l1620_162051

noncomputable def p (a b c : ℝ) : ℝ :=
  (2 / (a ^ 2 + 1)) - (2 / (b ^ 2 + 1)) + (3 / (c ^ 2 + 1))

theorem maximum_p (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : abc + a + c = b) : 
  p a b c ≤ 10 / 3 ∧ ∃ a b c, abc + a + c = b ∧ p a b c = 10 / 3 :=
sorry

end NUMINAMATH_GPT_maximum_p_l1620_162051


namespace NUMINAMATH_GPT_point_on_x_axis_l1620_162008

theorem point_on_x_axis (m : ℝ) (h : 3 * m + 1 = 0) : m = -1 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_point_on_x_axis_l1620_162008


namespace NUMINAMATH_GPT_p_plus_q_l1620_162045

-- Define the circles w1 and w2
def circle1 (x y : ℝ) := x^2 + y^2 + 10*x - 20*y - 77 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 10*x - 20*y + 193 = 0

-- Define the line condition
def line (a x y : ℝ) := y = a * x

-- Prove that p + q = 85, where m^2 = p / q and m is the smallest positive a
theorem p_plus_q : ∃ p q : ℕ, (p.gcd q = 1) ∧ (m^2 = (p : ℝ)/(q : ℝ)) ∧ (p + q = 85) :=
  sorry

end NUMINAMATH_GPT_p_plus_q_l1620_162045


namespace NUMINAMATH_GPT_smallest_positive_z_l1620_162024

open Real

theorem smallest_positive_z (x y z : ℝ) (m k n : ℤ) 
  (h1 : cos x = 0) 
  (h2 : sin y = 1) 
  (h3 : cos (x + z) = -1 / 2) :
  z = 5 * π / 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_z_l1620_162024


namespace NUMINAMATH_GPT_spencer_sessions_per_day_l1620_162036

theorem spencer_sessions_per_day :
  let jumps_per_minute := 4
  let minutes_per_session := 10
  let jumps_per_session := jumps_per_minute * minutes_per_session
  let total_jumps := 400
  let days := 5
  let jumps_per_day := total_jumps / days
  let sessions_per_day := jumps_per_day / jumps_per_session
  sessions_per_day = 2 :=
by
  sorry

end NUMINAMATH_GPT_spencer_sessions_per_day_l1620_162036


namespace NUMINAMATH_GPT_determinant_of_A_l1620_162091

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![ -5, 8],
    ![ 3, -4]]

theorem determinant_of_A : A.det = -4 := by
  sorry

end NUMINAMATH_GPT_determinant_of_A_l1620_162091


namespace NUMINAMATH_GPT_product_of_fractions_l1620_162066

theorem product_of_fractions (a b c d e f : ℚ) (h_a : a = 1) (h_b : b = 2) (h_c : c = 3) 
  (h_d : d = 2) (h_e : e = 3) (h_f : f = 4) :
  (a / b) * (d / e) * (c / f) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l1620_162066


namespace NUMINAMATH_GPT_billion_to_scientific_notation_l1620_162080

theorem billion_to_scientific_notation : 
  (98.36 * 10^9) = 9.836 * 10^10 := 
by
  sorry

end NUMINAMATH_GPT_billion_to_scientific_notation_l1620_162080


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1620_162046

theorem quadratic_no_real_roots :
  ∀ x : ℝ, ¬(x^2 - 2 * x + 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1620_162046


namespace NUMINAMATH_GPT_distance_to_fourth_buoy_l1620_162072

theorem distance_to_fourth_buoy
  (buoy_interval_distance : ℕ)
  (total_distance_to_third_buoy : ℕ)
  (h : total_distance_to_third_buoy = buoy_interval_distance * 3) :
  (buoy_interval_distance * 4 = 96) :=
by
  sorry

end NUMINAMATH_GPT_distance_to_fourth_buoy_l1620_162072


namespace NUMINAMATH_GPT_volume_of_box_with_ratio_125_l1620_162020

def volumes : Finset ℕ := {60, 80, 100, 120, 200}

theorem volume_of_box_with_ratio_125 : 80 ∈ volumes ∧ ∃ (x : ℕ), 10 * x^3 = 80 :=
by {
  -- Skipping the proof, as only the statement is required.
  sorry
}

end NUMINAMATH_GPT_volume_of_box_with_ratio_125_l1620_162020


namespace NUMINAMATH_GPT_gcd_of_two_powers_l1620_162050

-- Define the expressions
def two_pow_1015_minus_1 : ℤ := 2^1015 - 1
def two_pow_1024_minus_1 : ℤ := 2^1024 - 1

-- Define the gcd function and the target value
noncomputable def gcd_expr : ℤ := Int.gcd (2^1015 - 1) (2^1024 - 1)
def target : ℤ := 511

-- The statement we want to prove
theorem gcd_of_two_powers : gcd_expr = target := by 
  sorry

end NUMINAMATH_GPT_gcd_of_two_powers_l1620_162050


namespace NUMINAMATH_GPT_inequality_abc_l1620_162049

theorem inequality_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  abs (b / a - b / c) + abs (c / a - c / b) + abs (b * c + 1) > 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_abc_l1620_162049


namespace NUMINAMATH_GPT_polar_distance_l1620_162075

/-
Problem:
In the polar coordinate system, it is known that A(2, π / 6), B(4, 5π / 6). Then, the distance between points A and B is 2√7.

Conditions:
- Point A in polar coordinates: A(2, π / 6)
- Point B in polar coordinates: B(4, 5π / 6)
-/

/-- The distance between two points in the polar coordinate system A(2, π / 6) and B(4, 5π / 6) is 2√7. -/
theorem polar_distance :
  let A_ρ := 2
  let A_θ := π / 6
  let B_ρ := 4
  let B_θ := 5 * π / 6
  let A_x := A_ρ * Real.cos A_θ
  let A_y := A_ρ * Real.sin A_θ
  let B_x := B_ρ * Real.cos B_θ
  let B_y := B_ρ * Real.sin B_θ
  let distance := Real.sqrt ((B_x - A_x)^2 + (B_y - A_y)^2)
  distance = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_polar_distance_l1620_162075


namespace NUMINAMATH_GPT_find_number_l1620_162027

theorem find_number : ∃ (x : ℤ), 45 + 3 * x = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_GPT_find_number_l1620_162027


namespace NUMINAMATH_GPT_product_equality_l1620_162092

theorem product_equality : (2.05 * 4.1 = 20.5 * 0.41) :=
by
  sorry

end NUMINAMATH_GPT_product_equality_l1620_162092


namespace NUMINAMATH_GPT_calculate_exponent_l1620_162028

theorem calculate_exponent (m : ℝ) : (243 : ℝ)^(1 / 3) = 3^m → m = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_exponent_l1620_162028


namespace NUMINAMATH_GPT_triangle_angles_l1620_162086

-- Defining a structure for a triangle with angles
structure Triangle :=
(angleA angleB angleC : ℝ)

-- Define the condition for the triangle mentioned in the problem
def triangle_condition (t : Triangle) : Prop :=
  ∃ (α : ℝ), α = 22.5 ∧ t.angleA = 90 ∧ t.angleB = α ∧ t.angleC = 67.5

theorem triangle_angles :
  ∃ (t : Triangle), triangle_condition t :=
by
  -- The proof outline
  -- We need to construct a triangle with the given angle conditions
  -- angleA = 90°, angleB = 22.5°, angleC = 67.5°
  sorry

end NUMINAMATH_GPT_triangle_angles_l1620_162086


namespace NUMINAMATH_GPT_carl_first_to_roll_six_l1620_162096

-- Definitions based on problem conditions
def prob_six := 1 / 6
def prob_not_six := 5 / 6

-- Define geometric series sum formula for the given context
theorem carl_first_to_roll_six :
  ∑' n : ℕ, (prob_not_six^(3*n+1) * prob_six) = 25 / 91 :=
by
  sorry

end NUMINAMATH_GPT_carl_first_to_roll_six_l1620_162096


namespace NUMINAMATH_GPT_sandy_hourly_wage_l1620_162015

theorem sandy_hourly_wage (x : ℝ)
    (h1 : 10 * x + 6 * x + 14 * x = 450) : x = 15 :=
by
    sorry

end NUMINAMATH_GPT_sandy_hourly_wage_l1620_162015


namespace NUMINAMATH_GPT_average_weight_of_a_and_b_is_40_l1620_162087

variable (A B C : ℝ)

-- Conditions
def condition1 : Prop := (A + B + C) / 3 = 42
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 40

-- Theorem statement
theorem average_weight_of_a_and_b_is_40 (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 B) : 
    (A + B) / 2 = 40 := by
  sorry

end NUMINAMATH_GPT_average_weight_of_a_and_b_is_40_l1620_162087


namespace NUMINAMATH_GPT_intersection_point_l1620_162011

structure Point3D : Type where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨8, -9, 5⟩
def B : Point3D := ⟨18, -19, 15⟩
def C : Point3D := ⟨2, 5, -8⟩
def D : Point3D := ⟨4, -3, 12⟩

/-- Prove that the intersection point of lines AB and CD is (16, -19, 13) -/
theorem intersection_point :
  ∃ (P : Point3D), 
  (∃ t : ℝ, P = ⟨A.x + t * (B.x - A.x), A.y + t * (B.y - A.y), A.z + t * (B.z - A.z)⟩) ∧
  (∃ s : ℝ, P = ⟨C.x + s * (D.x - C.x), C.y + s * (D.y - C.y), C.z + s * (D.z - C.z)⟩) ∧
  P = ⟨16, -19, 13⟩ :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l1620_162011


namespace NUMINAMATH_GPT_kids_go_to_camp_l1620_162077

-- Define the total number of kids in Lawrence County
def total_kids : ℕ := 1059955

-- Define the number of kids who stay home
def stay_home : ℕ := 495718

-- Define the expected number of kids who go to camp
def expected_go_to_camp : ℕ := 564237

-- The theorem to prove the number of kids who go to camp
theorem kids_go_to_camp :
  total_kids - stay_home = expected_go_to_camp :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_kids_go_to_camp_l1620_162077


namespace NUMINAMATH_GPT_no_product_equal_remainder_l1620_162089

theorem no_product_equal_remainder (n : ℤ) : 
  ¬ (n = (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 1) = n * (n + 2) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 2) = n * (n + 1) * (n + 3) * (n + 4) * (n + 5) ∨
     (n + 3) = n * (n + 1) * (n + 2) * (n + 4) * (n + 5) ∨
     (n + 4) = n * (n + 1) * (n + 2) * (n + 3) * (n + 5) ∨
     (n + 5) = n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end NUMINAMATH_GPT_no_product_equal_remainder_l1620_162089


namespace NUMINAMATH_GPT_find_b_l1620_162090

theorem find_b (a b c : ℕ) (h1 : a + b + c = 99) (h2 : a + 6 = b - 6) (h3 : b - 6 = 5 * c) : b = 51 :=
sorry

end NUMINAMATH_GPT_find_b_l1620_162090


namespace NUMINAMATH_GPT_number_of_children_tickets_l1620_162042

theorem number_of_children_tickets 
    (x y : ℤ) 
    (h1 : x + y = 225) 
    (h2 : 6 * x + 9 * y = 1875) : 
    x = 50 := 
  sorry

end NUMINAMATH_GPT_number_of_children_tickets_l1620_162042


namespace NUMINAMATH_GPT_price_per_box_l1620_162003

theorem price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : 
  total_apples = 10000 → apples_per_box = 50 → total_revenue = 7000 → 
  total_revenue / (total_apples / apples_per_box) = 35 :=
by
  intros h1 h2 h3
  -- we can skip the actual proof with sorry. This indicates that the proof is not provided,
  -- but the statement is what needs to be proven.
  sorry

end NUMINAMATH_GPT_price_per_box_l1620_162003


namespace NUMINAMATH_GPT_probability_fourth_ball_black_l1620_162035

theorem probability_fourth_ball_black :
  let total_balls := 6
  let red_balls := 3
  let black_balls := 3
  let prob_black_first_draw := black_balls / total_balls
  (prob_black_first_draw = 1 / 2) ->
  (prob_black_first_draw = (black_balls / total_balls)) ->
  (black_balls / total_balls = 1 / 2) ->
  1 / 2 = 1 / 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_fourth_ball_black_l1620_162035


namespace NUMINAMATH_GPT_cakesServedDuringDinner_today_is_6_l1620_162043

def cakesServedDuringDinner (x : ℕ) : Prop :=
  5 + x + 3 = 14

theorem cakesServedDuringDinner_today_is_6 : cakesServedDuringDinner 6 :=
by
  unfold cakesServedDuringDinner
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cakesServedDuringDinner_today_is_6_l1620_162043


namespace NUMINAMATH_GPT_tom_total_spent_correct_l1620_162064

-- Definitions for discount calculations
def original_price_skateboard : ℝ := 9.46
def discount_rate_skateboard : ℝ := 0.10
def discounted_price_skateboard : ℝ := original_price_skateboard * (1 - discount_rate_skateboard)

def original_price_marbles : ℝ := 9.56
def discount_rate_marbles : ℝ := 0.10
def discounted_price_marbles : ℝ := original_price_marbles * (1 - discount_rate_marbles)

def price_shorts : ℝ := 14.50

def original_price_action_figures : ℝ := 12.60
def discount_rate_action_figures : ℝ := 0.20
def discounted_price_action_figures : ℝ := original_price_action_figures * (1 - discount_rate_action_figures)

-- Total for all discounted items
def total_discounted_items : ℝ := 
  discounted_price_skateboard + discounted_price_marbles + price_shorts + discounted_price_action_figures

-- Currency conversion for video game
def price_video_game_eur : ℝ := 20.50
def exchange_rate_eur_to_usd : ℝ := 1.12
def price_video_game_usd : ℝ := price_video_game_eur * exchange_rate_eur_to_usd

-- Total amount spent including the video game
def total_spent : ℝ := total_discounted_items + price_video_game_usd

-- Lean proof statement
theorem tom_total_spent_correct :
  total_spent = 64.658 :=
by {
  -- This is a placeholder "by sorry" which means the proof is missing.
  sorry
}

end NUMINAMATH_GPT_tom_total_spent_correct_l1620_162064


namespace NUMINAMATH_GPT_fraction_multiplication_l1620_162013

theorem fraction_multiplication :
  (3 / 4 : ℚ) * (1 / 2) * (2 / 5) * 5000 = 750 :=
by
  norm_num
  done

end NUMINAMATH_GPT_fraction_multiplication_l1620_162013


namespace NUMINAMATH_GPT_train_length_l1620_162097

theorem train_length :
  (∃ (L : ℝ), (L / 30 = (L + 2500) / 120) ∧ L = 75000 / 90) :=
sorry

end NUMINAMATH_GPT_train_length_l1620_162097


namespace NUMINAMATH_GPT_total_blue_balloons_l1620_162073

theorem total_blue_balloons (joan_balloons : ℕ) (melanie_balloons : ℕ) (h_joan : joan_balloons = 40) (h_melanie : melanie_balloons = 41) : joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_GPT_total_blue_balloons_l1620_162073


namespace NUMINAMATH_GPT_intersection_complement_l1620_162074

open Set

noncomputable def U := ℝ
noncomputable def A := {x : ℝ | x^2 + 2 * x < 3}
noncomputable def B := {x : ℝ | x - 2 ≤ 0 ∧ x ≠ 0}

theorem intersection_complement :
  A ∩ -B = {x : ℝ | -3 < x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_GPT_intersection_complement_l1620_162074


namespace NUMINAMATH_GPT_overall_percent_supporters_l1620_162044

theorem overall_percent_supporters
  (percent_A : ℝ) (percent_B : ℝ)
  (members_A : ℕ) (members_B : ℕ)
  (supporters_A : ℕ)
  (supporters_B : ℕ)
  (total_supporters : ℕ)
  (total_members : ℕ)
  (overall_percent : ℝ) 
  (h1 : percent_A = 0.70) 
  (h2 : percent_B = 0.75)
  (h3 : members_A = 200) 
  (h4 : members_B = 800) 
  (h5 : supporters_A = percent_A * members_A) 
  (h6 : supporters_B = percent_B * members_B) 
  (h7 : total_supporters = supporters_A + supporters_B) 
  (h8 : total_members = members_A + members_B) 
  (h9 : overall_percent = (total_supporters : ℝ) / total_members * 100) :
  overall_percent = 74 := by
  sorry

end NUMINAMATH_GPT_overall_percent_supporters_l1620_162044


namespace NUMINAMATH_GPT_margin_in_terms_of_ratio_l1620_162095

variable (S m : ℝ)

theorem margin_in_terms_of_ratio (h1 : M = (1/m) * S) (h2 : C = S - M) : M = (1/m) * S :=
sorry

end NUMINAMATH_GPT_margin_in_terms_of_ratio_l1620_162095


namespace NUMINAMATH_GPT_rational_function_nonnegative_l1620_162029

noncomputable def rational_function (x : ℝ) : ℝ :=
  (x - 8 * x^2 + 16 * x^3) / (9 - x^3)

theorem rational_function_nonnegative :
  ∀ x, 0 ≤ x ∧ x < 3 → 0 ≤ rational_function x :=
sorry

end NUMINAMATH_GPT_rational_function_nonnegative_l1620_162029


namespace NUMINAMATH_GPT_time_to_reach_rest_area_l1620_162048

variable (rate_per_minute : ℕ) (remaining_distance_yards : ℕ)

theorem time_to_reach_rest_area (h_rate : rate_per_minute = 2) (h_distance : remaining_distance_yards = 50) :
  (remaining_distance_yards * 3) / rate_per_minute = 75 := by
  sorry

end NUMINAMATH_GPT_time_to_reach_rest_area_l1620_162048


namespace NUMINAMATH_GPT_third_number_l1620_162000

theorem third_number (x : ℝ) 
    (h : 217 + 2.017 + 2.0017 + x = 221.2357) : 
    x = 0.217 :=
sorry

end NUMINAMATH_GPT_third_number_l1620_162000


namespace NUMINAMATH_GPT_fuel_consumption_l1620_162093

-- Define the initial conditions based on the problem
variable (s Q : ℝ)

-- Distance and fuel data points
def data_points : List (ℝ × ℝ) := [(0, 50), (100, 42), (200, 34), (300, 26), (400, 18)]

-- Define the function Q and required conditions
theorem fuel_consumption :
  (∀ p ∈ data_points, ∃ k b, Q = k * s + b ∧
    ((p.1 = 0 → b = 50) ∧
     (p.1 = 100 → Q = 42 → k = -0.08))) :=
by
  sorry

end NUMINAMATH_GPT_fuel_consumption_l1620_162093


namespace NUMINAMATH_GPT_chess_tournament_l1620_162054

theorem chess_tournament :
  ∀ (n : ℕ), (∃ (players : ℕ) (total_games : ℕ),
  players = 8 ∧ total_games = 56 ∧ total_games = (players * (players - 1) * n) / 2) →
  n = 2 :=
by
  intros n h
  rcases h with ⟨players, total_games, h_players, h_total_games, h_eq⟩
  have := h_eq
  sorry

end NUMINAMATH_GPT_chess_tournament_l1620_162054


namespace NUMINAMATH_GPT_bricks_in_chimney_l1620_162069

-- Define the conditions
def brenda_rate (h : ℕ) : ℚ := h / 8
def brandon_rate (h : ℕ) : ℚ := h / 12
def combined_rate (h : ℕ) : ℚ := (brenda_rate h + brandon_rate h) - 15
def total_bricks_in_6_hours (h : ℕ) : ℚ := 6 * combined_rate h

-- The proof statement
theorem bricks_in_chimney : ∃ h : ℕ, total_bricks_in_6_hours h = h ∧ h = 360 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bricks_in_chimney_l1620_162069


namespace NUMINAMATH_GPT_correct_equation_l1620_162006

theorem correct_equation (x : ℝ) :
  232 + x = 3 * (146 - x) :=
sorry

end NUMINAMATH_GPT_correct_equation_l1620_162006


namespace NUMINAMATH_GPT_roots_reciprocal_sum_eq_three_halves_l1620_162071

theorem roots_reciprocal_sum_eq_three_halves
  {a b : ℝ}
  (h1 : a^2 - 6 * a + 4 = 0)
  (h2 : b^2 - 6 * b + 4 = 0)
  (h_roots : a ≠ b) :
  1/a + 1/b = 3/2 := by
  sorry

end NUMINAMATH_GPT_roots_reciprocal_sum_eq_three_halves_l1620_162071


namespace NUMINAMATH_GPT_inverse_h_l1620_162056

-- Define the functions f, g, and h as given in the conditions
def f (x : ℝ) := 4 * x - 3
def g (x : ℝ) := 3 * x + 2
def h (x : ℝ) := f (g x)

-- State the problem of proving the inverse of h
theorem inverse_h : ∀ x, h⁻¹ (x : ℝ) = (x - 5) / 12 :=
sorry

end NUMINAMATH_GPT_inverse_h_l1620_162056


namespace NUMINAMATH_GPT_dreams_ratio_l1620_162021

variable (N : ℕ) (D_total : ℕ) (D_per_day : ℕ)

-- Conditions
def days_per_year : Prop := N = 365
def dreams_per_day : Prop := D_per_day = 4
def total_dreams : Prop := D_total = 4380

-- Derived definitions
def dreams_this_year := D_per_day * N
def dreams_last_year := D_total - dreams_this_year

-- Theorem to prove
theorem dreams_ratio 
  (h1 : days_per_year N)
  (h2 : dreams_per_day D_per_day)
  (h3 : total_dreams D_total)
  : dreams_last_year N D_total D_per_day / dreams_this_year N D_per_day = 2 :=
by
  sorry

end NUMINAMATH_GPT_dreams_ratio_l1620_162021


namespace NUMINAMATH_GPT_cos_150_eq_neg_sqrt3_over_2_l1620_162060

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_cos_150_eq_neg_sqrt3_over_2_l1620_162060


namespace NUMINAMATH_GPT_units_digit_2019_pow_2019_l1620_162017

theorem units_digit_2019_pow_2019 : (2019^2019) % 10 = 9 := 
by {
  -- The statement of the problem is proved below
  sorry  -- Solution to be filled in
}

end NUMINAMATH_GPT_units_digit_2019_pow_2019_l1620_162017


namespace NUMINAMATH_GPT_sue_shoes_probability_l1620_162001

def sueShoes : List (String × ℕ) := [("black", 7), ("brown", 3), ("gray", 2)]

def total_shoes := 24

def prob_same_color (color : String) (pairs : List (String × ℕ)) : ℚ :=
  let total_pairs := pairs.foldr (λ p acc => acc + p.snd) 0
  let matching_pair := pairs.filter (λ p => p.fst = color)
  if matching_pair.length = 1 then
   let n := matching_pair.head!.snd * 2
   (n / total_shoes) * ((n / 2) / (total_shoes - 1))
  else 0

def prob_total (pairs : List (String × ℕ)) : ℚ :=
  (prob_same_color "black" pairs) + (prob_same_color "brown" pairs) + (prob_same_color "gray" pairs)

theorem sue_shoes_probability :
  prob_total sueShoes = 31 / 138 := by
  sorry

end NUMINAMATH_GPT_sue_shoes_probability_l1620_162001


namespace NUMINAMATH_GPT_find_the_number_l1620_162053

theorem find_the_number (x : ℕ) : (220040 = (x + 445) * (2 * (x - 445)) + 40) → x = 555 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_the_number_l1620_162053


namespace NUMINAMATH_GPT_probability_square_not_touching_vertex_l1620_162084

theorem probability_square_not_touching_vertex :
  let total_squares := 64
  let squares_touching_vertices := 16
  let squares_not_touching_vertices := total_squares - squares_touching_vertices
  let probability := (squares_not_touching_vertices : ℚ) / total_squares
  probability = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_square_not_touching_vertex_l1620_162084


namespace NUMINAMATH_GPT_range_of_x_l1620_162023

theorem range_of_x (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) (x : ℝ) : 
  (x ^ 2 + (a - 4) * x + 4 - 2 * a > 0) ↔ (x < 1 ∨ x > 3) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l1620_162023


namespace NUMINAMATH_GPT_peter_has_read_more_books_l1620_162005

theorem peter_has_read_more_books
  (total_books : ℕ)
  (peter_percentage : ℚ)
  (brother_percentage : ℚ)
  (sarah_percentage : ℚ)
  (peter_books : ℚ := (peter_percentage / 100) * total_books)
  (brother_books : ℚ := (brother_percentage / 100) * total_books)
  (sarah_books : ℚ := (sarah_percentage / 100) * total_books)
  (combined_books : ℚ := brother_books + sarah_books)
  (difference : ℚ := peter_books - combined_books) :
  total_books = 50 → peter_percentage = 60 → brother_percentage = 25 → sarah_percentage = 15 → difference = 10 :=
by
  sorry

end NUMINAMATH_GPT_peter_has_read_more_books_l1620_162005


namespace NUMINAMATH_GPT_ValleyFalcons_all_items_l1620_162081

noncomputable def num_fans_receiving_all_items (capacity : ℕ) (tshirt_interval : ℕ) 
  (cap_interval : ℕ) (wristband_interval : ℕ) : ℕ :=
  (capacity / Nat.lcm (Nat.lcm tshirt_interval cap_interval) wristband_interval)

theorem ValleyFalcons_all_items:
  num_fans_receiving_all_items 3000 50 25 60 = 10 :=
by
  -- This is where the mathematical proof would go
  sorry

end NUMINAMATH_GPT_ValleyFalcons_all_items_l1620_162081


namespace NUMINAMATH_GPT_find_x_l1620_162098

theorem find_x (x : ℤ) (h_pos : x > 0) 
  (n := x^2 + 2 * x + 17) 
  (d := 2 * x + 5)
  (h_div : n = d * x + 7) : x = 2 := 
sorry

end NUMINAMATH_GPT_find_x_l1620_162098


namespace NUMINAMATH_GPT_quadratic_root_four_times_another_l1620_162078

theorem quadratic_root_four_times_another (a : ℝ) :
  (∃ x1 x2 : ℝ, x^2 + a * x + 2 * a = 0 ∧ x2 = 4 * x1) → a = 25 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_four_times_another_l1620_162078


namespace NUMINAMATH_GPT_cheating_percentage_l1620_162025

theorem cheating_percentage (x : ℝ) :
  (∀ cost_price : ℝ, cost_price = 100 →
   let received_when_buying : ℝ := cost_price * (1 + x / 100)
   let given_when_selling : ℝ := cost_price * (1 - x / 100)
   let profit : ℝ := received_when_buying - given_when_selling
   let profit_percentage : ℝ := profit / cost_price
   profit_percentage = 2 / 9) →
  x = 22.22222222222222 := 
by
  sorry

end NUMINAMATH_GPT_cheating_percentage_l1620_162025


namespace NUMINAMATH_GPT_sum_largest_smallest_5_6_7_l1620_162058

/--
Given the digits 5, 6, and 7, if we form all possible three-digit numbers using each digit exactly once, 
then the sum of the largest and smallest of these numbers is 1332.
-/
theorem sum_largest_smallest_5_6_7 : 
  let d1 := 5
  let d2 := 6
  let d3 := 7
  let smallest := 100 * d1 + 10 * d2 + d3
  let largest := 100 * d3 + 10 * d2 + d1
  smallest + largest = 1332 := 
by
  sorry

end NUMINAMATH_GPT_sum_largest_smallest_5_6_7_l1620_162058


namespace NUMINAMATH_GPT_problem_statement_l1620_162079

def f : ℝ → ℝ :=
  sorry

lemma even_function (x : ℝ) : f (-x) = f x :=
  sorry

lemma periodicity (x : ℝ) (hx : 0 ≤ x) : f (x + 2) = -f x :=
  sorry

lemma value_in_interval (x : ℝ) (hx : 0 ≤ x ∧ x < 2) : f x = Real.log (x + 1) :=
  sorry

theorem problem_statement : f (-2001) + f 2012 = 1 :=
  sorry

end NUMINAMATH_GPT_problem_statement_l1620_162079


namespace NUMINAMATH_GPT_number_of_ways_to_select_book_l1620_162012

-- Definitions directly from the problem's conditions
def numMathBooks : Nat := 3
def numChineseBooks : Nat := 5
def numEnglishBooks : Nat := 8

-- The proof problem statement in Lean 4
theorem number_of_ways_to_select_book : numMathBooks + numChineseBooks + numEnglishBooks = 16 := 
by
  show 3 + 5 + 8 = 16
  sorry

end NUMINAMATH_GPT_number_of_ways_to_select_book_l1620_162012


namespace NUMINAMATH_GPT_train_length_correct_l1620_162016

def train_length (speed_kph : ℕ) (time_sec : ℕ) : ℕ :=
  let speed_mps := speed_kph * 1000 / 3600
  speed_mps * time_sec

theorem train_length_correct :
  train_length 90 10 = 250 := by
  sorry

end NUMINAMATH_GPT_train_length_correct_l1620_162016


namespace NUMINAMATH_GPT_inversely_proportional_decrease_l1620_162009

theorem inversely_proportional_decrease :
  ∀ {x y q c : ℝ}, 
  0 < x ∧ 0 < y ∧ 0 < c ∧ 0 < q →
  (x * y = c) →
  (((1 + q / 100) * x) * ((100 / (100 + q)) * y) = c) →
  ((y - (100 / (100 + q)) * y) / y) * 100 = 100 * q / (100 + q) :=
by
  intros x y q c hb hxy hxy'
  sorry

end NUMINAMATH_GPT_inversely_proportional_decrease_l1620_162009


namespace NUMINAMATH_GPT_acute_triangle_inequality_l1620_162032

variable (f : ℝ → ℝ)
variable {A B : ℝ}
variable (h₁ : ∀ x : ℝ, x * (f'' x) - 2 * (f x) > 0)
variable (h₂ : A + B < Real.pi / 2 ∧ 0 < A ∧ 0 < B)

theorem acute_triangle_inequality :
  f (Real.cos A) * (Real.sin B) ^ 2 < f (Real.sin B) * (Real.cos A) ^ 2 := 
  sorry

end NUMINAMATH_GPT_acute_triangle_inequality_l1620_162032


namespace NUMINAMATH_GPT_neg_ln_gt_zero_l1620_162004

theorem neg_ln_gt_zero {x : ℝ} : (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ ∃ x : ℝ, Real.log (x^2 + 1) ≤ 0 := by
  sorry

end NUMINAMATH_GPT_neg_ln_gt_zero_l1620_162004


namespace NUMINAMATH_GPT_inverse_variation_l1620_162033

theorem inverse_variation (x y : ℝ) (h1 : 7 * y = 1400 / x^3) (h2 : x = 4) : y = 25 / 8 :=
  by
  sorry

end NUMINAMATH_GPT_inverse_variation_l1620_162033


namespace NUMINAMATH_GPT_calculate_expression_l1620_162076

theorem calculate_expression : 5 * 12 + 6 * 11 - 2 * 15 + 7 * 9 = 159 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1620_162076


namespace NUMINAMATH_GPT_log_expression_correct_l1620_162088

-- The problem involves logarithms and exponentials
theorem log_expression_correct : 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 50) + (Real.log 25) + Real.exp (Real.log 3) = 5 := 
  by 
    sorry

end NUMINAMATH_GPT_log_expression_correct_l1620_162088


namespace NUMINAMATH_GPT_second_newly_inserted_number_eq_l1620_162041

theorem second_newly_inserted_number_eq : 
  ∃ q : ℝ, (q ^ 12 = 2) ∧ (1 * (q ^ 2) = 2 ^ (1 / 6)) := 
by
  sorry

end NUMINAMATH_GPT_second_newly_inserted_number_eq_l1620_162041


namespace NUMINAMATH_GPT_sampling_is_systematic_l1620_162099

-- Conditions
def production_line (units_per_day : ℕ) : Prop := units_per_day = 128

def sampling_inspection (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  samples_per_day = 8 ∧ inspection_time = 30 ∧ inspection_days = 7

-- Question
def sampling_method (method : String) (units_per_day : ℕ) (samples_per_day : ℕ) (inspection_time : ℕ) (inspection_days : ℕ) : Prop :=
  production_line units_per_day ∧ sampling_inspection samples_per_day inspection_time inspection_days → method = "systematic sampling"

-- Theorem stating the question == answer given conditions
theorem sampling_is_systematic : sampling_method "systematic sampling" 128 8 30 7 :=
by
  sorry

end NUMINAMATH_GPT_sampling_is_systematic_l1620_162099


namespace NUMINAMATH_GPT_money_sum_l1620_162057

theorem money_sum (A B C : ℕ) (h1 : A + C = 300) (h2 : B + C = 600) (h3 : C = 200) : A + B + C = 700 :=
by
  sorry

end NUMINAMATH_GPT_money_sum_l1620_162057


namespace NUMINAMATH_GPT_side_length_S2_l1620_162065

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end NUMINAMATH_GPT_side_length_S2_l1620_162065


namespace NUMINAMATH_GPT_inconsistent_conditions_l1620_162070

-- Definitions based on the given conditions
def B : Nat := 59
def C : Nat := 27
def D : Nat := 31
def A := B * C + D

theorem inconsistent_conditions (A_is_factor : ∃ k : Nat, 4701 = k * A) : false := by
  sorry

end NUMINAMATH_GPT_inconsistent_conditions_l1620_162070


namespace NUMINAMATH_GPT_part_a_l1620_162083

theorem part_a (n : ℤ) (m : ℤ) (h : m = n + 2) : 
  n * m + 1 = (n + 1) ^ 2 := by
  sorry

end NUMINAMATH_GPT_part_a_l1620_162083


namespace NUMINAMATH_GPT_attendees_count_l1620_162026

def n_students_seated : ℕ := 300
def n_students_standing : ℕ := 25
def n_teachers_seated : ℕ := 30

def total_attendees : ℕ :=
  n_students_seated + n_students_standing + n_teachers_seated

theorem attendees_count :
  total_attendees = 355 := by
  sorry

end NUMINAMATH_GPT_attendees_count_l1620_162026


namespace NUMINAMATH_GPT_tangent_product_le_one_third_l1620_162067

theorem tangent_product_le_one_third (α β : ℝ) (h : α + β = π / 3) (hα : 0 < α) (hβ : 0 < β) : 
  Real.tan α * Real.tan β ≤ 1 / 3 :=
sorry

end NUMINAMATH_GPT_tangent_product_le_one_third_l1620_162067


namespace NUMINAMATH_GPT_smallest_integer_in_ratio_l1620_162030

theorem smallest_integer_in_ratio {a b c : ℕ} (h1 : a = 2 * b / 3) (h2 : c = 5 * b / 3) (h3 : a + b + c = 60) : b = 12 := 
  sorry

end NUMINAMATH_GPT_smallest_integer_in_ratio_l1620_162030


namespace NUMINAMATH_GPT_student_survey_l1620_162039

-- Define the conditions given in the problem
theorem student_survey (S F : ℝ) (h1 : F = 25 + 65) (h2 : F = 0.45 * S) : S = 200 :=
by
  sorry

end NUMINAMATH_GPT_student_survey_l1620_162039


namespace NUMINAMATH_GPT_pupils_sent_up_exam_l1620_162034

theorem pupils_sent_up_exam (average_marks : ℕ) (specific_scores : List ℕ) (new_average : ℕ) : 
  (average_marks = 39) → 
  (specific_scores = [25, 12, 15, 19]) → 
  (new_average = 44) → 
  ∃ n : ℕ, (n > 4) ∧ (average_marks * n) = 39 * n ∧ ((39 * n - specific_scores.sum) / (n - specific_scores.length)) = new_average →
  n = 21 :=
by
  intros h_avg h_scores h_new_avg
  sorry

end NUMINAMATH_GPT_pupils_sent_up_exam_l1620_162034


namespace NUMINAMATH_GPT_max_ab_l1620_162040

theorem max_ab (a b : ℝ) (h1 : 1 ≤ a - b ∧ a - b ≤ 2) (h2 : 3 ≤ a + b ∧ a + b ≤ 4) : ab ≤ 15 / 4 :=
sorry

end NUMINAMATH_GPT_max_ab_l1620_162040


namespace NUMINAMATH_GPT_f_odd_and_increasing_l1620_162055

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) := sorry

end NUMINAMATH_GPT_f_odd_and_increasing_l1620_162055


namespace NUMINAMATH_GPT_sticker_arrangement_l1620_162068

theorem sticker_arrangement : 
  ∀ (n : ℕ), n = 35 → 
  (∀ k : ℕ, k = 8 → 
    ∃ m : ℕ, m = 5 ∧ (n + m) % k = 0) := 
by sorry

end NUMINAMATH_GPT_sticker_arrangement_l1620_162068


namespace NUMINAMATH_GPT_correct_number_of_three_digit_numbers_l1620_162062

def count_valid_three_digit_numbers : Nat :=
  let hundreds := [1, 2, 3, 4, 6, 7, 9].length
  let tens_units := [0, 1, 2, 3, 4, 6, 7, 9].length
  hundreds * tens_units * tens_units

theorem correct_number_of_three_digit_numbers :
  count_valid_three_digit_numbers = 448 :=
by
  unfold count_valid_three_digit_numbers
  sorry

end NUMINAMATH_GPT_correct_number_of_three_digit_numbers_l1620_162062


namespace NUMINAMATH_GPT_express_in_scientific_notation_l1620_162018

theorem express_in_scientific_notation 
  (A : 149000000 = 149 * 10^6)
  (B : 149000000 = 1.49 * 10^8)
  (C : 149000000 = 14.9 * 10^7)
  (D : 149000000 = 1.5 * 10^8) :
  149000000 = 1.49 * 10^8 := 
by
  sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l1620_162018


namespace NUMINAMATH_GPT_cannot_form_set_l1620_162094

/-- Define the set of non-negative real numbers not exceeding 20 --/
def setA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 20}

/-- Define the set of solutions of the equation x^2 - 9 = 0 within the real numbers --/
def setB : Set ℝ := {x | x^2 - 9 = 0}

/-- Define the set of all students taller than 170 cm enrolled in a certain school in the year 2013 --/
def setC : Type := sorry

/-- Define the (pseudo) set of all approximate values of sqrt(3) --/
def pseudoSetD : Set ℝ := {x | x = Real.sqrt 3}

/-- Main theorem stating that setD cannot form a mathematically valid set --/
theorem cannot_form_set (x : ℝ) : x ∈ pseudoSetD → False := sorry

end NUMINAMATH_GPT_cannot_form_set_l1620_162094


namespace NUMINAMATH_GPT_students_without_favorite_subject_l1620_162059

theorem students_without_favorite_subject (total_students : ℕ) (like_math : ℕ) (like_english : ℕ) (like_science : ℕ) :
  total_students = 30 →
  like_math = total_students * 1 / 5 →
  like_english = total_students * 1 / 3 →
  like_science = (total_students - (like_math + like_english)) * 1 / 7 →
  total_students - (like_math + like_english + like_science) = 12 :=
by
  intro h_total h_math h_english h_science
  sorry

end NUMINAMATH_GPT_students_without_favorite_subject_l1620_162059
