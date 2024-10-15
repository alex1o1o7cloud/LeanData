import Mathlib

namespace NUMINAMATH_GPT_average_reading_days_l1583_158370

def emery_days : ℕ := 20
def serena_days : ℕ := 5 * emery_days
def average_days (e s : ℕ) : ℕ := (e + s) / 2

theorem average_reading_days 
  (e s : ℕ) 
  (h1 : e = emery_days)
  (h2 : s = serena_days) :
  average_days e s = 60 :=
by
  rw [h1, h2, emery_days, serena_days]
  sorry

end NUMINAMATH_GPT_average_reading_days_l1583_158370


namespace NUMINAMATH_GPT_find_cost_price_l1583_158371

noncomputable def original_cost_price (C S C_new S_new : ℝ) : Prop :=
  S = 1.25 * C ∧
  C_new = 0.80 * C ∧
  S_new = 1.25 * C - 10.50 ∧
  S_new = 1.04 * C

theorem find_cost_price (C S C_new S_new : ℝ) :
  original_cost_price C S C_new S_new → C = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l1583_158371


namespace NUMINAMATH_GPT_combined_resistance_l1583_158378

theorem combined_resistance (x y : ℝ) (r : ℝ) (hx : x = 4) (hy : y = 6) :
  (1 / r) = (1 / x) + (1 / y) → r = 12 / 5 :=
by
  sorry

end NUMINAMATH_GPT_combined_resistance_l1583_158378


namespace NUMINAMATH_GPT_ellipse_eccentricity_l1583_158362

theorem ellipse_eccentricity (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : b = (4/3) * c) (h4 : a^2 - b^2 = c^2) : 
  c / a = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l1583_158362


namespace NUMINAMATH_GPT_find_f2_l1583_158351

-- Definitions based on the given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable (f g : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_g_def : ∀ x, g x = f x + 9)
variable (h_g_val : g (-2) = 3)

-- Prove the required goal
theorem find_f2 : f 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l1583_158351


namespace NUMINAMATH_GPT_jesse_needs_more_carpet_l1583_158377

def additional_carpet_needed (carpet : ℕ) (length : ℕ) (width : ℕ) : ℕ :=
  let room_area := length * width
  room_area - carpet

theorem jesse_needs_more_carpet
  (carpet : ℕ) (length : ℕ) (width : ℕ)
  (h_carpet : carpet = 18)
  (h_length : length = 4)
  (h_width : width = 20) :
  additional_carpet_needed carpet length width = 62 :=
by {
  -- the proof goes here
  sorry
}

end NUMINAMATH_GPT_jesse_needs_more_carpet_l1583_158377


namespace NUMINAMATH_GPT_cookies_number_l1583_158327

-- Define all conditions in the problem
def number_of_chips_per_cookie := 7
def number_of_cookies_per_dozen := 12
def number_of_uneaten_chips := 168

-- Define D as the number of dozens of cookies
variable (D : ℕ)

-- Prove the Lean theorem
theorem cookies_number (h : 7 * 6 * D = 168) : D = 4 :=
by
  sorry

end NUMINAMATH_GPT_cookies_number_l1583_158327


namespace NUMINAMATH_GPT_rectangular_prism_volume_increase_l1583_158301

theorem rectangular_prism_volume_increase (L B H : ℝ) :
  let V_original := L * B * H
  let L_new := L * 1.07
  let B_new := B * 1.18
  let H_new := H * 1.25
  let V_new := L_new * B_new * H_new
  let increase_in_volume := (V_new - V_original) / V_original * 100
  increase_in_volume = 56.415 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_increase_l1583_158301


namespace NUMINAMATH_GPT_solve_for_b_l1583_158396

theorem solve_for_b :
  (∀ (x y : ℝ), 4 * y - 3 * x + 2 = 0) →
  (∀ (x y : ℝ), 2 * y + b * x - 1 = 0) →
  (∃ b : ℝ, b = 8 / 3) := 
by
  sorry

end NUMINAMATH_GPT_solve_for_b_l1583_158396


namespace NUMINAMATH_GPT_sin_double_theta_l1583_158379

-- Given condition
def given_condition (θ : ℝ) : Prop :=
  Real.cos (Real.pi / 4 - θ) = 1 / 2

-- The statement we want to prove: sin(2θ) = -1/2
theorem sin_double_theta (θ : ℝ) (h : given_condition θ) : Real.sin (2 * θ) = -1 / 2 :=
sorry

end NUMINAMATH_GPT_sin_double_theta_l1583_158379


namespace NUMINAMATH_GPT_solve_inequality_system_l1583_158349

theorem solve_inequality_system : 
  (∀ x : ℝ, (1 / 3 * x - 1 ≤ 1 / 2 * x + 1) ∧ ((3 * x - (x - 2) ≥ 6) ∧ (x + 1 > (4 * x - 1) / 3)) → (2 ≤ x ∧ x < 4)) := 
by
  intro x h
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1583_158349


namespace NUMINAMATH_GPT_morning_registration_count_l1583_158353

variable (M : ℕ) -- Number of students registered for the morning session
variable (MorningAbsentees : ℕ := 3) -- Absentees in the morning session
variable (AfternoonRegistered : ℕ := 24) -- Students registered for the afternoon session
variable (AfternoonAbsentees : ℕ := 4) -- Absentees in the afternoon session

theorem morning_registration_count :
  (M - MorningAbsentees) + (AfternoonRegistered - AfternoonAbsentees) = 42 → M = 25 :=
by
  sorry

end NUMINAMATH_GPT_morning_registration_count_l1583_158353


namespace NUMINAMATH_GPT_zero_function_solution_l1583_158324

theorem zero_function_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = f (x^3) + 3 * x^2 * f (x) * f (y) + 3 * (f (x) * f (y))^2 + y^6 * f (y)) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_function_solution_l1583_158324


namespace NUMINAMATH_GPT_train_length_l1583_158318

theorem train_length (speed_faster speed_slower : ℝ) (time_sec : ℝ) (length_each_train : ℝ) :
  speed_faster = 47 ∧ speed_slower = 36 ∧ time_sec = 36 ∧ 
  (length_each_train = 55 ↔ 2 * length_each_train = ((speed_faster - speed_slower) * (1000/3600) * time_sec)) :=
by {
  -- We declare the speeds in km/hr and convert the relative speed to m/s for calculation.
  sorry
}

end NUMINAMATH_GPT_train_length_l1583_158318


namespace NUMINAMATH_GPT_floor_x_floor_x_eq_44_iff_l1583_158350

theorem floor_x_floor_x_eq_44_iff (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 44) ↔ (7.333 ≤ x ∧ x < 7.5) :=
by
  sorry

end NUMINAMATH_GPT_floor_x_floor_x_eq_44_iff_l1583_158350


namespace NUMINAMATH_GPT_correct_statements_are_two_l1583_158352

def statement1 : Prop := 
  ∀ (data : Type) (eq : data → data → Prop), 
    (∃ (t : data), eq t t) → 
    (∀ (d1 d2 : data), eq d1 d2 → d1 = d2)

def statement2 : Prop := 
  ∀ (samplevals : Type) (regress_eqn : samplevals → samplevals → Prop), 
    (∃ (s : samplevals), regress_eqn s s) → 
    (∀ (sv1 sv2 : samplevals), regress_eqn sv1 sv2 → sv1 = sv2)

def statement3 : Prop := 
  ∀ (predvals : Type) (pred_eqn : predvals → predvals → Prop), 
    (∃ (p : predvals), pred_eqn p p) → 
    (∀ (pp1 pp2 : predvals), pred_eqn pp1 pp2 → pp1 = pp2)

def statement4 : Prop := 
  ∀ (observedvals : Type) (linear_eqn : observedvals → observedvals → Prop), 
    (∃ (o : observedvals), linear_eqn o o) → 
    (∀ (ov1 ov2 : observedvals), linear_eqn ov1 ov2 → ov1 = ov2)

def correct_statements_count : ℕ := 2

theorem correct_statements_are_two : 
  (statement1 ∧ statement2 ∧ ¬ statement3 ∧ ¬ statement4) → 
  correct_statements_count = 2 := by
  sorry

end NUMINAMATH_GPT_correct_statements_are_two_l1583_158352


namespace NUMINAMATH_GPT_diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l1583_158338

noncomputable def diagonals_in_regular_polygon (n : ℕ) : ℕ :=
  n * (n - 3) / 2

noncomputable def exterior_angle (n : ℕ) : ℝ :=
  360.0 / n

theorem diagonals_of_60_sided_polygon :
  diagonals_in_regular_polygon 60 = 1710 :=
by
  sorry

theorem exterior_angle_of_60_sided_polygon :
  exterior_angle 60 = 6.0 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_of_60_sided_polygon_exterior_angle_of_60_sided_polygon_l1583_158338


namespace NUMINAMATH_GPT_x_share_of_profit_l1583_158369

-- Define the problem conditions
def investment_x : ℕ := 5000
def investment_y : ℕ := 15000
def total_profit : ℕ := 1600

-- Define the ratio simplification
def ratio_x : ℕ := 1
def ratio_y : ℕ := 3
def total_ratio_parts : ℕ := ratio_x + ratio_y

-- Define the profit division per part
def profit_per_part : ℕ := total_profit / total_ratio_parts

-- Lean 4 statement to prove
theorem x_share_of_profit : profit_per_part * ratio_x = 400 := sorry

end NUMINAMATH_GPT_x_share_of_profit_l1583_158369


namespace NUMINAMATH_GPT_area_above_the_line_l1583_158373

-- Definitions of the circle and the line equations
def circle_eqn (x y : ℝ) := (x - 5)^2 + (y - 3)^2 = 1
def line_eqn (x y : ℝ) := y = x - 5

-- The main statement to prove
theorem area_above_the_line : 
  ∃ (A : ℝ), A = (3 / 4) * Real.pi ∧ 
  ∀ (x y : ℝ), 
    circle_eqn x y ∧ y > x - 5 → 
    A > 0 := 
sorry

end NUMINAMATH_GPT_area_above_the_line_l1583_158373


namespace NUMINAMATH_GPT_women_attended_l1583_158309

theorem women_attended (m w : ℕ) 
  (h_danced_with_4_women : ∀ (k : ℕ), k < m → k * 4 = 60)
  (h_danced_with_3_men : ∀ (k : ℕ), k < w → 3 * (k * (m / 3)) = 60)
  (h_men_count : m = 15) : 
  w = 20 := 
sorry

end NUMINAMATH_GPT_women_attended_l1583_158309


namespace NUMINAMATH_GPT_nancy_carrots_l1583_158304

def carrots_total 
  (initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) : ℕ :=
  initial - thrown_out + picked_next_day

theorem nancy_carrots : 
  carrots_total 12 2 21 = 31 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_nancy_carrots_l1583_158304


namespace NUMINAMATH_GPT_ratio_lcm_gcf_l1583_158384

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 252) (h₂ : b = 675) : 
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  (lcm_ab / gcf_ab) = 2100 :=
by
  sorry

end NUMINAMATH_GPT_ratio_lcm_gcf_l1583_158384


namespace NUMINAMATH_GPT_perpendicular_lines_solve_b_l1583_158387

theorem perpendicular_lines_solve_b (b : ℝ) : (∀ x y : ℝ, y = 3 * x + 7 →
                                                    ∃ y1 : ℝ, y1 = ( - b / 4 ) * x + 3 ∧
                                                               3 * ( - b / 4 ) = -1) → 
                                               b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_solve_b_l1583_158387


namespace NUMINAMATH_GPT_b_minus_a_l1583_158314

theorem b_minus_a :
  ∃ (a b : ℝ), (2 + 4 = -a) ∧ (2 * 4 = b) ∧ (b - a = 14) :=
by
  use (-6 : ℝ)
  use (8 : ℝ)
  simp
  sorry

end NUMINAMATH_GPT_b_minus_a_l1583_158314


namespace NUMINAMATH_GPT_ideal_type_circle_D_l1583_158302

-- Define the line equation
def line_l (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0

-- Define the distance condition for circles
def ideal_type_circle (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), 
    line_l P.1 P.2 ∧ line_l Q.1 Q.2 ∧
    dist P (0, 0) = radius ∧
    dist Q (0, 0) = radius ∧
    dist (P, Q) = 1

-- Definition of given circles A, B, C, D
def circle_A (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_B (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 1
def circle_D (x y : ℝ) : Prop := (x - 4)^2 + (y - 4)^2 = 16

-- Define circle centers and radii for A, B, C, D
def center_A : ℝ × ℝ := (0, 0)
def radius_A : ℝ := 1
def center_B : ℝ × ℝ := (0, 0)
def radius_B : ℝ := 4
def center_C : ℝ × ℝ := (4, 4)
def radius_C : ℝ := 1
def center_D : ℝ × ℝ := (4, 4)
def radius_D : ℝ := 4

-- Problem Statement: Prove that option D is the "ideal type" circle
theorem ideal_type_circle_D : 
  ideal_type_circle center_D radius_D :=
sorry

end NUMINAMATH_GPT_ideal_type_circle_D_l1583_158302


namespace NUMINAMATH_GPT_f_odd_f_decreasing_f_extremum_l1583_158360

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_val : f 1 = -2
axiom f_neg : ∀ x > 0, f x < 0

theorem f_odd : ∀ x : ℝ, f (-x) = -f x :=
sorry

theorem f_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
sorry

theorem f_extremum : ∃ (max min : ℝ), max = f (-3) ∧ min = f 3 :=
sorry

end NUMINAMATH_GPT_f_odd_f_decreasing_f_extremum_l1583_158360


namespace NUMINAMATH_GPT_basketball_cost_l1583_158392

-- Initial conditions
def initial_amount : Nat := 50
def cost_jerseys (n price_per_jersey : Nat) : Nat := n * price_per_jersey
def cost_shorts : Nat := 8
def remaining_amount : Nat := 14

-- Derived total spent calculation
def total_spent (initial remaining : Nat) : Nat := initial - remaining
def known_cost (jerseys shorts : Nat) : Nat := jerseys + shorts

-- Prove the cost of the basketball
theorem basketball_cost :
  let jerseys := cost_jerseys 5 2
  let shorts := cost_shorts
  let total_spent := total_spent initial_amount remaining_amount
  let known_cost := known_cost jerseys shorts
  total_spent - known_cost = 18 := 
by
  sorry

end NUMINAMATH_GPT_basketball_cost_l1583_158392


namespace NUMINAMATH_GPT_gcd_540_180_diminished_by_2_eq_178_l1583_158367

theorem gcd_540_180_diminished_by_2_eq_178 : gcd 540 180 - 2 = 178 := by
  sorry

end NUMINAMATH_GPT_gcd_540_180_diminished_by_2_eq_178_l1583_158367


namespace NUMINAMATH_GPT_b_alone_work_time_l1583_158381

def work_rate_combined (a_rate b_rate : ℝ) : ℝ := a_rate + b_rate

theorem b_alone_work_time
  (a_rate b_rate : ℝ)
  (h1 : work_rate_combined a_rate b_rate = 1/16)
  (h2 : a_rate = 1/20) :
  b_rate = 1/80 := by
  sorry

end NUMINAMATH_GPT_b_alone_work_time_l1583_158381


namespace NUMINAMATH_GPT_largest_angle_in_ratio_3_4_5_l1583_158306

theorem largest_angle_in_ratio_3_4_5 : ∃ (A B C : ℝ), (A / 3 = B / 4 ∧ B / 4 = C / 5) ∧ (A + B + C = 180) ∧ (C = 75) :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_in_ratio_3_4_5_l1583_158306


namespace NUMINAMATH_GPT_tom_saves_80_dollars_l1583_158319

def normal_doctor_cost : ℝ := 200
def discount_percentage : ℝ := 0.7
def discount_clinic_cost_per_visit : ℝ := normal_doctor_cost * (1 - discount_percentage)
def number_of_visits : ℝ := 2
def total_discount_clinic_cost : ℝ := discount_clinic_cost_per_visit * number_of_visits
def savings : ℝ := normal_doctor_cost - total_discount_clinic_cost

theorem tom_saves_80_dollars : savings = 80 := by
  sorry

end NUMINAMATH_GPT_tom_saves_80_dollars_l1583_158319


namespace NUMINAMATH_GPT_derivative_of_f_l1583_158372

noncomputable def f (x : ℝ) : ℝ := 2^x - Real.log x / Real.log 3

theorem derivative_of_f (x : ℝ) : (deriv f x) = 2^x * Real.log 2 - 1 / (x * Real.log 3) :=
by
  -- This statement skips the proof details
  sorry

end NUMINAMATH_GPT_derivative_of_f_l1583_158372


namespace NUMINAMATH_GPT_infinite_series_computation_l1583_158366

noncomputable def infinite_series_sum (a b : ℝ) : ℝ :=
  ∑' n : ℕ, if n = 0 then (0 : ℝ) else
    (1 : ℝ) / ((2 * (n - 1 : ℕ) * a - (n - 2 : ℕ) * b) * (2 * n * a - (n - 1 : ℕ) * b))

theorem infinite_series_computation (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ineq : a > b) :
  infinite_series_sum a b = 1 / ((2 * a - b) * (2 * b)) :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_computation_l1583_158366


namespace NUMINAMATH_GPT_sin_A_value_of_triangle_l1583_158332

theorem sin_A_value_of_triangle 
  (a b : ℝ) (A B C : ℝ) (h_triangle : a = 2) (h_b : b = 3) (h_tanB : Real.tan B = 3) :
  Real.sin A = Real.sqrt 10 / 5 :=
sorry

end NUMINAMATH_GPT_sin_A_value_of_triangle_l1583_158332


namespace NUMINAMATH_GPT_value_of_expression_l1583_158357

theorem value_of_expression (x : ℤ) (h : x = 5) : 3 * x + 4 = 19 :=
by
  rw [h]
  exact rfl

end NUMINAMATH_GPT_value_of_expression_l1583_158357


namespace NUMINAMATH_GPT_log_12_eq_2a_plus_b_l1583_158307

variable (lg : ℝ → ℝ)
variable (lg_2_eq_a : lg 2 = a)
variable (lg_3_eq_b : lg 3 = b)

theorem log_12_eq_2a_plus_b : lg 12 = 2 * a + b :=
by
  sorry

end NUMINAMATH_GPT_log_12_eq_2a_plus_b_l1583_158307


namespace NUMINAMATH_GPT_arithmetic_sequence_general_formula_inequality_satisfaction_l1583_158316

namespace Problem

-- Definitions for the sequences and the sum of terms
def a (n : ℕ) : ℕ := sorry -- define based on conditions
def S (n : ℕ) : ℕ := sorry -- sum of first n terms of {a_n}
def b (n : ℕ) : ℕ := 2 * (S (n + 1) - S n) * S n - n * (S (n + 1) + S n)

-- Part 1: Prove the general formula for the arithmetic sequence
theorem arithmetic_sequence_general_formula :
  (∀ n : ℕ, b n = 0) → (∀ n : ℕ, a n = 0 ∨ a n = n) :=
sorry

-- Part 2: Conditions for geometric sequences and inequality
def a_2n_minus_1 (n : ℕ) : ℕ := 2 ^ n
def a_2n (n : ℕ) : ℕ := 3 * 2 ^ (n - 1)
def b_2n (n : ℕ) : ℕ := sorry -- compute based on conditions
def b_2n_minus_1 (n : ℕ) : ℕ := sorry -- compute based on conditions

def b_condition (n : ℕ) : Prop := b_2n n < b_2n_minus_1 n

-- Prove the set of all positive integers n that satisfy the inequality
theorem inequality_satisfaction :
  { n : ℕ | b_condition n } = {1, 2, 3, 4, 5, 6} :=
sorry

end Problem

end NUMINAMATH_GPT_arithmetic_sequence_general_formula_inequality_satisfaction_l1583_158316


namespace NUMINAMATH_GPT_jake_present_weight_l1583_158346

theorem jake_present_weight (J S B : ℝ) (h1 : J - 20 = 2 * S) (h2 : B = 0.5 * J) (h3 : J + S + B = 330) :
  J = 170 :=
by sorry

end NUMINAMATH_GPT_jake_present_weight_l1583_158346


namespace NUMINAMATH_GPT_unpainted_unit_cubes_l1583_158347

theorem unpainted_unit_cubes (total_cubes painted_faces edge_overlaps corner_overlaps : ℕ) :
  total_cubes = 6 * 6 * 6 ∧
  painted_faces = 6 * (2 * 6) ∧
  edge_overlaps = 12 * 3 / 2 ∧
  corner_overlaps = 8 ∧
  total_cubes - (painted_faces - edge_overlaps - corner_overlaps) = 170 :=
by
  sorry

end NUMINAMATH_GPT_unpainted_unit_cubes_l1583_158347


namespace NUMINAMATH_GPT_chocolate_distribution_l1583_158341

theorem chocolate_distribution
  (total_chocolate : ℚ)
  (num_piles : ℕ)
  (piles_given_to_shaina : ℕ)
  (weight_each_pile : ℚ)
  (weight_of_shaina_piles : ℚ)
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_given_to_shaina = 2)
  (h4 : weight_each_pile = total_chocolate / num_piles)
  (h5 : weight_of_shaina_piles = piles_given_to_shaina * weight_each_pile) :
  weight_of_shaina_piles = 24 / 7 := by
  sorry

end NUMINAMATH_GPT_chocolate_distribution_l1583_158341


namespace NUMINAMATH_GPT_fill_time_difference_correct_l1583_158386

-- Define the time to fill one barrel in normal conditions
def normal_fill_time_per_barrel : ℕ := 3

-- Define the time to fill one barrel with a leak
def leak_fill_time_per_barrel : ℕ := 5

-- Define the number of barrels to fill
def barrels_to_fill : ℕ := 12

-- Define the time to fill 12 barrels in normal conditions
def normal_fill_time : ℕ := normal_fill_time_per_barrel * barrels_to_fill

-- Define the time to fill 12 barrels with a leak
def leak_fill_time : ℕ := leak_fill_time_per_barrel * barrels_to_fill

-- Define the time difference
def time_difference : ℕ := leak_fill_time - normal_fill_time

theorem fill_time_difference_correct : time_difference = 24 := by
  sorry

end NUMINAMATH_GPT_fill_time_difference_correct_l1583_158386


namespace NUMINAMATH_GPT_geometric_sequence_product_l1583_158389

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_cond : a 2 * a 4 = 16) : a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1583_158389


namespace NUMINAMATH_GPT_max_type_A_pieces_max_profit_l1583_158397

noncomputable def type_A_cost := 80
noncomputable def type_A_sell := 120
noncomputable def type_B_cost := 60
noncomputable def type_B_sell := 90
noncomputable def total_clothes := 100
noncomputable def min_type_A := 65
noncomputable def max_cost := 7500

/-- The maximum number of type A clothing pieces that can be purchased --/
theorem max_type_A_pieces (x : ℕ) : 
  type_A_cost * x + type_B_cost * (total_clothes - x) ≤ max_cost → 
  x ≤ 75 := by 
sorry

variable (a : ℝ) (h_a : 0 < a ∧ a < 10)

/-- The optimal purchase strategy to maximize profit --/
theorem max_profit (x y : ℕ) : 
  (x + y = total_clothes) ∧ 
  (type_A_cost * x + type_B_cost * y ≤ max_cost) ∧
  (min_type_A ≤ x) ∧ 
  (x ≤ 75) → 
  (type_A_sell - type_A_cost - a) * x + (type_B_sell - type_B_cost) * y 
  ≤ (type_A_sell - type_A_cost - a) * 75 + (type_B_sell - type_B_cost) * 25 := by 
sorry

end NUMINAMATH_GPT_max_type_A_pieces_max_profit_l1583_158397


namespace NUMINAMATH_GPT_binom_60_3_eq_34220_l1583_158343

def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_60_3_eq_34220 : binom 60 3 = 34220 := by
  sorry

end NUMINAMATH_GPT_binom_60_3_eq_34220_l1583_158343


namespace NUMINAMATH_GPT_tangent_perpendicular_point_l1583_158394

open Real

noncomputable def f (x : ℝ) : ℝ := exp x - (1 / 2) * x^2

theorem tangent_perpendicular_point :
  ∃ x0, (f x0 = 1) ∧ (x0 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_perpendicular_point_l1583_158394


namespace NUMINAMATH_GPT_time_taken_by_alex_l1583_158376

-- Define the conditions
def distance_per_lap : ℝ := 500 -- distance per lap in meters
def distance_first_part : ℝ := 150 -- first part of the distance in meters
def speed_first_part : ℝ := 3 -- speed for the first part in meters per second
def distance_second_part : ℝ := 350 -- remaining part of the distance in meters
def speed_second_part : ℝ := 4 -- speed for the remaining part in meters per second
def num_laps : ℝ := 4 -- number of laps run by Alex

-- Target time, expressed in seconds
def target_time : ℝ := 550 -- 9 minutes and 10 seconds is 550 seconds

-- Prove that given the conditions, the total time Alex takes to run 4 laps is 550 seconds
theorem time_taken_by_alex :
  (distance_first_part / speed_first_part + distance_second_part / speed_second_part) * num_laps = target_time :=
by
  sorry

end NUMINAMATH_GPT_time_taken_by_alex_l1583_158376


namespace NUMINAMATH_GPT_exists_valid_configuration_l1583_158391

-- Define the nine circles
def circles : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

-- Define the connections (adjacency list) where each connected pair must sum to 23
def lines : List (ℕ × ℕ) := [(1, 8), (8, 6), (8, 9), (9, 2), (2, 7), (7, 6), (7, 4), (4, 1), (4, 5), (5, 6), (5, 3), (6, 3)]

-- The main theorem that we need to prove: there exists a permutation of circles satisfying the line sum condition
theorem exists_valid_configuration: 
  ∃ (f : ℕ → ℕ), 
    (∀ x ∈ circles, f x ∈ circles) ∧ 
    (∀ (a b : ℕ), (a, b) ∈ lines → f a + f b = 23) :=
sorry

end NUMINAMATH_GPT_exists_valid_configuration_l1583_158391


namespace NUMINAMATH_GPT_no_positive_solution_l1583_158315

theorem no_positive_solution (a : ℕ → ℝ) (h1 : ∀ n, a n > 0) :
  ¬ (∀ n ≥ 2, a (n + 2) = a n - a (n - 1)) :=
sorry

end NUMINAMATH_GPT_no_positive_solution_l1583_158315


namespace NUMINAMATH_GPT_find_second_number_l1583_158380

theorem find_second_number (x y z : ℚ) (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 240 / 7 := by
  sorry

end NUMINAMATH_GPT_find_second_number_l1583_158380


namespace NUMINAMATH_GPT_cube_faces_edges_vertices_sum_l1583_158382

theorem cube_faces_edges_vertices_sum :
  let faces := 6
  let edges := 12
  let vertices := 8
  faces + edges + vertices = 26 :=
by
  sorry

end NUMINAMATH_GPT_cube_faces_edges_vertices_sum_l1583_158382


namespace NUMINAMATH_GPT_c_share_of_profit_l1583_158305

theorem c_share_of_profit
  (a_investment : ℝ)
  (b_investment : ℝ)
  (c_investment : ℝ)
  (total_profit : ℝ)
  (ha : a_investment = 30000)
  (hb : b_investment = 45000)
  (hc : c_investment = 50000)
  (hp : total_profit = 90000) :
  (c_investment / (a_investment + b_investment + c_investment)) * total_profit = 36000 := 
by
  sorry

end NUMINAMATH_GPT_c_share_of_profit_l1583_158305


namespace NUMINAMATH_GPT_moles_of_water_l1583_158340

-- Definitions related to the reaction conditions.
def HCl : Type := sorry
def NaHCO3 : Type := sorry
def NaCl : Type := sorry
def H2O : Type := sorry
def CO2 : Type := sorry

def reaction (h : HCl) (n : NaHCO3) : Nat := sorry -- Represents the balanced reaction

-- Given conditions in Lean.
axiom one_mole_HCl : HCl
axiom one_mole_NaHCO3 : NaHCO3
axiom balanced_equation : reaction one_mole_HCl one_mole_NaHCO3 = 1 -- 1 mole of water is produced

-- The theorem to prove.
theorem moles_of_water : reaction one_mole_HCl one_mole_NaHCO3 = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_moles_of_water_l1583_158340


namespace NUMINAMATH_GPT_smallest_base10_integer_l1583_158303

theorem smallest_base10_integer :
  ∃ a b : ℕ, a > 3 ∧ b > 3 ∧ (2 * a + 2 = 3 * b + 3) ∧ (2 * a + 2 = 18) :=
by
  existsi 8 -- assign specific solutions to a
  existsi 5 -- assign specific solutions to b
  exact sorry -- follows from the validations done above

end NUMINAMATH_GPT_smallest_base10_integer_l1583_158303


namespace NUMINAMATH_GPT_initial_amount_saved_l1583_158375

noncomputable section

def cost_of_couch : ℝ := 750
def cost_of_table : ℝ := 100
def cost_of_lamp : ℝ := 50
def amount_still_owed : ℝ := 400

def total_cost : ℝ := cost_of_couch + cost_of_table + cost_of_lamp

theorem initial_amount_saved (initial_amount : ℝ) :
  initial_amount = total_cost - amount_still_owed ↔ initial_amount = 500 :=
by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_initial_amount_saved_l1583_158375


namespace NUMINAMATH_GPT_total_study_hours_during_semester_l1583_158345

-- Definitions of the given conditions
def semester_weeks : ℕ := 15
def weekday_study_hours_per_day : ℕ := 3
def saturday_study_hours : ℕ := 4
def sunday_study_hours : ℕ := 5

-- Theorem statement to prove the total study hours during the semester
theorem total_study_hours_during_semester : 
  (semester_weeks * ((5 * weekday_study_hours_per_day) + saturday_study_hours + sunday_study_hours)) = 360 := by
  -- We are skipping the proof step and adding a placeholder
  sorry

end NUMINAMATH_GPT_total_study_hours_during_semester_l1583_158345


namespace NUMINAMATH_GPT_find_other_endpoint_of_diameter_l1583_158336

theorem find_other_endpoint_of_diameter 
    (center endpoint : ℝ × ℝ) 
    (h_center : center = (5, -2)) 
    (h_endpoint : endpoint = (2, 3))
    : (center.1 + (center.1 - endpoint.1), center.2 + (center.2 - endpoint.2)) = (8, -7) := 
by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_of_diameter_l1583_158336


namespace NUMINAMATH_GPT_minimum_value_of_a_l1583_158356

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp (3 * Real.log x - x)) - x^2 - (a - 4) * x - 4

theorem minimum_value_of_a (h : ∀ x > 0, f x ≤ 0) : a ≥ 4 / Real.exp 2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_a_l1583_158356


namespace NUMINAMATH_GPT_charity_meaning_l1583_158312

theorem charity_meaning (noun_charity : String) (h : noun_charity = "charity") : 
  (noun_charity = "charity" → "charity" = "charitable organization") :=
by
  sorry

end NUMINAMATH_GPT_charity_meaning_l1583_158312


namespace NUMINAMATH_GPT_find_a_l1583_158325

theorem find_a (a : ℝ) (A : Set ℝ) (hA : A = {a - 2, a^2 + 4*a, 10}) (h : -3 ∈ A) : a = -3 := 
by
  -- placeholder proof
  sorry

end NUMINAMATH_GPT_find_a_l1583_158325


namespace NUMINAMATH_GPT_audit_sampling_is_systematic_l1583_158339

def is_systematic_sampling (population_size : Nat) (step : Nat) (initial_index : Nat) : Prop :=
  ∃ (k : Nat), ∀ (n : Nat), n ≠ 0 → initial_index + step * (n - 1) ≤ population_size

theorem audit_sampling_is_systematic :
  ∀ (population_size : Nat) (random_index : Nat),
  population_size = 50 * 50 →  -- This represents the total number of invoices (50% of a larger population segment)
  random_index < 50 →         -- Randomly selected index from the first 50 invoices
  is_systematic_sampling population_size 50 random_index := 
by
  intros
  sorry

end NUMINAMATH_GPT_audit_sampling_is_systematic_l1583_158339


namespace NUMINAMATH_GPT_multiply_by_11_l1583_158355

theorem multiply_by_11 (A B k : ℕ) (h1 : 10 * A + B < 100) (h2 : A + B = 10 + k) :
  (10 * A + B) * 11 = 100 * (A + 1) + 10 * k + B :=
by 
  sorry

end NUMINAMATH_GPT_multiply_by_11_l1583_158355


namespace NUMINAMATH_GPT_josh_initial_marbles_l1583_158322

def marbles_initial (lost : ℕ) (left : ℕ) : ℕ := lost + left

theorem josh_initial_marbles :
  marbles_initial 5 4 = 9 :=
by sorry

end NUMINAMATH_GPT_josh_initial_marbles_l1583_158322


namespace NUMINAMATH_GPT_intersection_M_N_eq_neg2_l1583_158330

open Set

-- Definitions of the sets M and N
def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x * x - x - 6 ≥ 0}

-- Proof statement that M ∩ N = {-2}
theorem intersection_M_N_eq_neg2 : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_neg2_l1583_158330


namespace NUMINAMATH_GPT_woman_complete_time_l1583_158388

-- Define the work rate of one man
def man_rate := 1 / 100

-- Define the combined work rate equation for 10 men and 15 women completing work in 5 days
def combined_work_rate (W : ℝ) : Prop :=
  10 * man_rate + 15 * W = 1 / 5

-- Prove that given the combined work rate equation, one woman alone takes 150 days to complete the work
theorem woman_complete_time (W : ℝ) : combined_work_rate W → W = 1 / 150 :=
by
  intro h
  have h1 : 10 * man_rate + 15 * W = 1 / 5 := h
  rw [man_rate] at h1
  sorry -- Proof steps would go here

end NUMINAMATH_GPT_woman_complete_time_l1583_158388


namespace NUMINAMATH_GPT_largest_room_length_l1583_158361

theorem largest_room_length (L : ℕ) (w_large w_small l_small diff_area : ℕ)
  (h1 : w_large = 45)
  (h2 : w_small = 15)
  (h3 : l_small = 8)
  (h4 : diff_area = 1230)
  (h5 : w_large * L - (w_small * l_small) = diff_area) :
  L = 30 :=
by sorry

end NUMINAMATH_GPT_largest_room_length_l1583_158361


namespace NUMINAMATH_GPT_initial_money_l1583_158364

/-
We had $3500 left after spending 30% of our money on clothing, 
25% on electronics, and saving 15% in a bank account. 
How much money (X) did we start with before shopping and saving?
-/

theorem initial_money (M : ℝ) 
  (h_clothing : 0.30 * M ≠ 0) 
  (h_electronics : 0.25 * M ≠ 0) 
  (h_savings : 0.15 * M ≠ 0) 
  (remaining_money : 0.30 * M = 3500) : 
  M = 11666.67 := 
sorry

end NUMINAMATH_GPT_initial_money_l1583_158364


namespace NUMINAMATH_GPT_Lisa_pay_per_hour_is_15_l1583_158333

-- Given conditions:
def Greta_hours : ℕ := 40
def Greta_pay_per_hour : ℕ := 12
def Lisa_hours : ℕ := 32

-- Define Greta's earnings based on the given conditions:
def Greta_earnings : ℕ := Greta_hours * Greta_pay_per_hour

-- The main statement to prove:
theorem Lisa_pay_per_hour_is_15 (h1 : Greta_earnings = Greta_hours * Greta_pay_per_hour) 
                                (h2 : Greta_earnings = Lisa_hours * L) :
  L = 15 :=
by sorry

end NUMINAMATH_GPT_Lisa_pay_per_hour_is_15_l1583_158333


namespace NUMINAMATH_GPT_imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l1583_158308

theorem imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half :
  (Complex.exp (-Complex.I * Real.pi / 6)).im = -1/2 := by
sorry

end NUMINAMATH_GPT_imaginary_part_of_exp_neg_pi_div_6_eq_neg_one_half_l1583_158308


namespace NUMINAMATH_GPT_probability_same_plane_l1583_158398

-- Define the number of vertices in a cube
def num_vertices : ℕ := 8

-- Define the number of vertices to be selected
def selection : ℕ := 4

-- Define the total number of ways to select 4 vertices out of 8
def total_ways : ℕ := Nat.choose num_vertices selection

-- Define the number of favorable ways to have 4 vertices lie in the same plane
def favorable_ways : ℕ := 12

-- Define the probability that the 4 selected vertices lie in the same plane
def probability : ℚ := favorable_ways / total_ways

-- The statement we need to prove
theorem probability_same_plane : probability = 6 / 35 := by
  sorry

end NUMINAMATH_GPT_probability_same_plane_l1583_158398


namespace NUMINAMATH_GPT_least_n_for_factorial_multiple_10080_l1583_158374

theorem least_n_for_factorial_multiple_10080 (n : ℕ) 
  (h₁ : 0 < n) 
  (h₂ : ∀ m, m > 0 → (n ≠ m → n! % 10080 ≠ 0)) 
  : n = 8 := 
sorry

end NUMINAMATH_GPT_least_n_for_factorial_multiple_10080_l1583_158374


namespace NUMINAMATH_GPT_biography_increase_l1583_158310

theorem biography_increase (B N : ℝ) (hN : N = 0.35 * (B + N) - 0.20 * B):
  (N / (0.20 * B) * 100) = 115.38 :=
by
  sorry

end NUMINAMATH_GPT_biography_increase_l1583_158310


namespace NUMINAMATH_GPT_find_angle_A_l1583_158348

theorem find_angle_A (a b : ℝ) (B A : ℝ) (ha : a = Real.sqrt 3) (hb : b = Real.sqrt 2) (hB : B = Real.pi / 4) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l1583_158348


namespace NUMINAMATH_GPT_sqrt_x_minus_2_real_iff_x_ge_2_l1583_158365

theorem sqrt_x_minus_2_real_iff_x_ge_2 (x : ℝ) : (∃ r : ℝ, r * r = x - 2) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_x_minus_2_real_iff_x_ge_2_l1583_158365


namespace NUMINAMATH_GPT_multiples_of_six_l1583_158399

theorem multiples_of_six (a b : ℕ) (h₁ : a = 5) (h₂ : b = 127) :
  ∃ n : ℕ, n = 21 ∧ ∀ x : ℕ, (a < 6 * x ∧ 6 * x < b) ↔ (1 ≤ x ∧ x ≤ 21) :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_six_l1583_158399


namespace NUMINAMATH_GPT_find_vertex_A_l1583_158331

variables (B C: ℝ × ℝ × ℝ)

-- Defining midpoints conditions
def midpoint_BC : ℝ × ℝ × ℝ := (1, 5, -1)
def midpoint_AC : ℝ × ℝ × ℝ := (0, 4, -2)
def midpoint_AB : ℝ × ℝ × ℝ := (2, 3, 4)

-- The coordinates of point A we need to prove
def target_A : ℝ × ℝ × ℝ := (1, 2, 3)

-- Lean statement proving the coordinates of A
theorem find_vertex_A (A B C : ℝ × ℝ × ℝ)
  (hBC : midpoint_BC = (1, 5, -1))
  (hAC : midpoint_AC = (0, 4, -2))
  (hAB : midpoint_AB = (2, 3, 4)) :
  A = (1, 2, 3) := 
sorry

end NUMINAMATH_GPT_find_vertex_A_l1583_158331


namespace NUMINAMATH_GPT_original_savings_eq_920_l1583_158311

variable (S : ℝ) -- Define S as a real number representing Linda's savings
variable (h1 : S * (1 / 4) = 230) -- Given condition

theorem original_savings_eq_920 :
  S = 920 :=
by
  sorry

end NUMINAMATH_GPT_original_savings_eq_920_l1583_158311


namespace NUMINAMATH_GPT_sin_product_eq_one_sixteenth_l1583_158337

theorem sin_product_eq_one_sixteenth : 
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * 
  (Real.sin (78 * Real.pi / 180)) = 
  1 / 16 := 
sorry

end NUMINAMATH_GPT_sin_product_eq_one_sixteenth_l1583_158337


namespace NUMINAMATH_GPT_contrapositive_l1583_158334

variable (P Q : Prop)

theorem contrapositive (h : P → Q) : ¬Q → ¬P :=
sorry

end NUMINAMATH_GPT_contrapositive_l1583_158334


namespace NUMINAMATH_GPT_cost_per_quart_l1583_158342

theorem cost_per_quart (paint_cost : ℝ) (coverage : ℝ) (cost_to_paint_cube : ℝ) (cube_edge : ℝ) 
    (h_coverage : coverage = 1200) (h_cost_to_paint_cube : cost_to_paint_cube = 1.60) 
    (h_cube_edge : cube_edge = 10) : paint_cost = 3.20 := by 
  sorry

end NUMINAMATH_GPT_cost_per_quart_l1583_158342


namespace NUMINAMATH_GPT_fly_flies_more_than_10_meters_l1583_158383

theorem fly_flies_more_than_10_meters :
  ∃ (fly_path_length : ℝ), 
  (∃ (c : ℝ) (a b : ℝ), c = 5 ∧ a^2 + b^2 = c^2) →
  (fly_path_length > 10) := 
by
  sorry

end NUMINAMATH_GPT_fly_flies_more_than_10_meters_l1583_158383


namespace NUMINAMATH_GPT_algae_plants_in_milford_lake_l1583_158358

theorem algae_plants_in_milford_lake (original : ℕ) (increase : ℕ) : (original = 809) → (increase = 2454) → (original + increase = 3263) :=
by
  sorry

end NUMINAMATH_GPT_algae_plants_in_milford_lake_l1583_158358


namespace NUMINAMATH_GPT_problem_solution_l1583_158335

theorem problem_solution (x : ℝ) : (∃ (x : ℝ), 5 < x ∧ x ≤ 6) ↔ (∃ (x : ℝ), (x - 3) / (x - 5) ≥ 3) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1583_158335


namespace NUMINAMATH_GPT_crystal_meals_count_l1583_158354

def num_entrees : ℕ := 4
def num_drinks : ℕ := 4
def num_desserts : ℕ := 2

theorem crystal_meals_count : num_entrees * num_drinks * num_desserts = 32 := by
  sorry

end NUMINAMATH_GPT_crystal_meals_count_l1583_158354


namespace NUMINAMATH_GPT_total_chocolate_bars_proof_l1583_158385

def large_box_contains := 17
def first_10_boxes_contains := 10
def medium_boxes_per_small := 4
def chocolate_bars_per_medium := 26

def remaining_7_boxes := 7
def first_two_boxes := 2
def first_two_bars := 18
def next_three_boxes := 3
def next_three_bars := 22
def last_two_boxes := 2
def last_two_bars := 30

noncomputable def total_chocolate_bars_in_large_box : Nat :=
  let chocolate_in_first_10 := first_10_boxes_contains * medium_boxes_per_small * chocolate_bars_per_medium
  let chocolate_in_remaining_7 :=
    (first_two_boxes * first_two_bars) +
    (next_three_boxes * next_three_bars) +
    (last_two_boxes * last_two_bars)
  chocolate_in_first_10 + chocolate_in_remaining_7

theorem total_chocolate_bars_proof :
  total_chocolate_bars_in_large_box = 1202 :=
by
  -- Detailed calculation is skipped
  sorry

end NUMINAMATH_GPT_total_chocolate_bars_proof_l1583_158385


namespace NUMINAMATH_GPT_sample_quantities_and_probability_l1583_158395

-- Define the given quantities from each workshop
def q_A := 10
def q_B := 20
def q_C := 30

-- Total sample size
def n := 6

-- Given conditions, the total quantity and sample ratio
def total_quantity := q_A + q_B + q_C
def ratio := n / total_quantity

-- Derived quantities in the samples based on the proportion
def sample_A := q_A * ratio
def sample_B := q_B * ratio
def sample_C := q_C * ratio

-- Combinatorial calculations
def C (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def total_combinations := C 6 2
def workshop_C_combinations := C 3 2
def probability_C_samples := workshop_C_combinations / total_combinations

-- Theorem to prove the quantities and probability
theorem sample_quantities_and_probability :
  sample_A = 1 ∧ sample_B = 2 ∧ sample_C = 3 ∧ probability_C_samples = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sample_quantities_and_probability_l1583_158395


namespace NUMINAMATH_GPT_combination_lock_code_l1583_158326

theorem combination_lock_code :
  ∀ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ (x + y + x * y = 10 * x + y) →
  10 * x + y = 19 ∨ 10 * x + y = 29 ∨ 10 * x + y = 39 ∨ 10 * x + y = 49 ∨
  10 * x + y = 59 ∨ 10 * x + y = 69 ∨ 10 * x + y = 79 ∨ 10 * x + y = 89 ∨
  10 * x + y = 99 :=
by
  sorry

end NUMINAMATH_GPT_combination_lock_code_l1583_158326


namespace NUMINAMATH_GPT_coffee_mix_price_per_pound_l1583_158320

-- Definitions based on conditions
def total_weight : ℝ := 100
def columbian_price_per_pound : ℝ := 8.75
def brazilian_price_per_pound : ℝ := 3.75
def columbian_weight : ℝ := 52
def brazilian_weight : ℝ := total_weight - columbian_weight

-- Goal to prove
theorem coffee_mix_price_per_pound :
  (columbian_weight * columbian_price_per_pound + brazilian_weight * brazilian_price_per_pound) / total_weight = 6.35 :=
by
  sorry

end NUMINAMATH_GPT_coffee_mix_price_per_pound_l1583_158320


namespace NUMINAMATH_GPT_inequality_squares_l1583_158313

theorem inequality_squares (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_inequality_squares_l1583_158313


namespace NUMINAMATH_GPT_probability_of_diff_by_three_is_one_eighth_l1583_158390

-- Define the problem within a namespace
namespace DiceRoll

-- Define the probability of rolling two integers that differ by 3 on an 8-sided die
noncomputable def prob_diff_by_three : ℚ :=
  let successful_outcomes := 8
  let total_outcomes := 8 * 8
  successful_outcomes / total_outcomes

-- The main theorem
theorem probability_of_diff_by_three_is_one_eighth :
  prob_diff_by_three = 1 / 8 := by
  sorry

end DiceRoll

end NUMINAMATH_GPT_probability_of_diff_by_three_is_one_eighth_l1583_158390


namespace NUMINAMATH_GPT_quadratic_sum_l1583_158317

theorem quadratic_sum (a b c : ℝ) (h : ∀ x : ℝ, 5 * x^2 - 30 * x - 45 = a * (x + b)^2 + c) :
  a + b + c = -88 := by
  sorry

end NUMINAMATH_GPT_quadratic_sum_l1583_158317


namespace NUMINAMATH_GPT_trigonometric_identity_l1583_158393

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 + Real.sin α * Real.cos α = 6 / 5 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1583_158393


namespace NUMINAMATH_GPT_find_p_l1583_158300

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p q r : ℕ) (hp : is_prime p) (hq : is_prime q) (hr : is_prime r) (h1 : p + q = r + 2) (h2 : 1 < p) (h3 : p < q) :
  p = 2 := 
sorry

end NUMINAMATH_GPT_find_p_l1583_158300


namespace NUMINAMATH_GPT_adam_walks_distance_l1583_158359

/-- The side length of the smallest squares is 20 cm. --/
def smallest_square_side : ℕ := 20

/-- The side length of the middle-sized square is 2 times the smallest square. --/
def middle_square_side : ℕ := 2 * smallest_square_side

/-- The side length of the largest square is 3 times the smallest square. --/
def largest_square_side : ℕ := 3 * smallest_square_side

/-- The number of smallest squares Adam encounters. --/
def num_smallest_squares : ℕ := 5

/-- The number of middle-sized squares Adam encounters. --/
def num_middle_squares : ℕ := 5

/-- The number of largest squares Adam encounters. --/
def num_largest_squares : ℕ := 2

/-- The total distance Adam walks from P to Q. --/
def total_distance : ℕ :=
  num_smallest_squares * smallest_square_side +
  num_middle_squares * middle_square_side +
  num_largest_squares * largest_square_side

/-- Proof that the total distance Adam walks is 420 cm. --/
theorem adam_walks_distance : total_distance = 420 := by
  sorry

end NUMINAMATH_GPT_adam_walks_distance_l1583_158359


namespace NUMINAMATH_GPT_odd_integer_solution_l1583_158344

theorem odd_integer_solution
  (y : ℤ) (hy_odd : y % 2 = 1)
  (h : ∃ x : ℤ, x^2 + 2*y^2 = y*x^2 + y + 1) :
  y = 1 :=
sorry

end NUMINAMATH_GPT_odd_integer_solution_l1583_158344


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1583_158329

theorem arithmetic_sequence_properties
    (a_1 : ℕ)
    (d : ℕ)
    (sequence : Fin 240 → ℕ)
    (h1 : ∀ n, sequence n = a_1 + n * d)
    (h2 : sequence 0 = a_1)
    (h3 : 1 ≤ a_1 ∧ a_1 ≤ 9)
    (h4 : ∃ n₁, sequence n₁ = 100)
    (h5 : ∃ n₂, sequence n₂ = 3103) :
  (a_1 = 9 ∧ d = 13) ∨ (a_1 = 1 ∧ d = 33) ∨ (a_1 = 9 ∧ d = 91) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1583_158329


namespace NUMINAMATH_GPT_range_of_a_l1583_158321

-- Define the inequality condition
def condition (a : ℝ) (x : ℝ) : Prop := abs (a - 2 * x) > x - 1

-- Define the range for x
def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the main theorem statement
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, in_range x → condition a x) ↔ (a < 2 ∨ 5 < a) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1583_158321


namespace NUMINAMATH_GPT_f_neg_one_f_monotonic_decreasing_solve_inequality_l1583_158368

-- Definitions based on conditions in part a)
variables {f : ℝ → ℝ}
axiom f_add : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂ - 2
axiom f_one : f 1 = 0
axiom f_neg : ∀ x > 1, f x < 0

-- Proof statement for the value of f(-1)
theorem f_neg_one : f (-1) = 4 := by
  sorry

-- Proof statement for the monotonicity of f(x)
theorem f_monotonic_decreasing : ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂ := by
  sorry

-- Proof statement for the inequality solution
theorem solve_inequality (x : ℝ) :
  ∀ t, t = f (x^2 - 2*x) →
  t^2 + 2*t - 8 < 0 → (-1 < x ∧ x < 0) ∨ (2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_GPT_f_neg_one_f_monotonic_decreasing_solve_inequality_l1583_158368


namespace NUMINAMATH_GPT_GoldenRabbitCards_count_l1583_158363

theorem GoldenRabbitCards_count :
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  golden_cards = 5904 :=
by
  let total_cards := 10000
  let non_golden_combinations := 8 * 8 * 8 * 8
  let golden_cards := total_cards - non_golden_combinations
  sorry

end NUMINAMATH_GPT_GoldenRabbitCards_count_l1583_158363


namespace NUMINAMATH_GPT_ratio_greater_than_two_ninths_l1583_158323

-- Define the conditions
def M : ℕ := 8
def N : ℕ := 36

-- State the theorem
theorem ratio_greater_than_two_ninths : (M : ℚ) / (N : ℚ) > 2 / 9 := 
by {
    -- skipping the proof with sorry
    sorry
}

end NUMINAMATH_GPT_ratio_greater_than_two_ninths_l1583_158323


namespace NUMINAMATH_GPT_goods_train_speed_l1583_158328

theorem goods_train_speed (man_train_speed_kmh : Float) 
    (goods_train_length_m : Float) 
    (passing_time_s : Float) 
    (kmh_to_ms : Float := 1000 / 3600) : 
    man_train_speed_kmh = 50 → 
    goods_train_length_m = 280 → 
    passing_time_s = 9 → 
    Float.round ((goods_train_length_m / passing_time_s + man_train_speed_kmh * kmh_to_ms) * 3600 / 1000) = 61.99
:= by
  sorry

end NUMINAMATH_GPT_goods_train_speed_l1583_158328
