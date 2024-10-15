import Mathlib

namespace NUMINAMATH_GPT_trajectory_of_M_lines_perpendicular_l1872_187269

-- Define the given conditions
def parabola (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 = P.2

def midpoint_condition (P M : ℝ × ℝ) : Prop :=
  P.1 = 1/2 * M.1 ∧ P.2 = M.2

def trajectory_condition (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 = 4 * M.2

theorem trajectory_of_M (P M : ℝ × ℝ) (H1 : parabola P) (H2 : midpoint_condition P M) : 
  trajectory_condition M :=
sorry

-- Define the conditions for the second part
def line_through_F (A B : ℝ × ℝ) (F : ℝ × ℝ): Prop :=
  ∃ k : ℝ, A.2 = k * A.1 + F.2 ∧ B.2 = k * B.1 + F.2

def perpendicular_feet (A B A1 B1 : ℝ × ℝ) : Prop :=
  A1 = (A.1, -1) ∧ B1 = (B.1, -1)

def perpendicular_lines (A1 B1 F : ℝ × ℝ) : Prop :=
  let v1 := (-A1.1, F.2 - A1.2)
  let v2 := (-B1.1, F.2 - B1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem lines_perpendicular (A B A1 B1 F : ℝ × ℝ) (H1 : trajectory_condition A) (H2 : trajectory_condition B) 
(H3 : line_through_F A B F) (H4 : perpendicular_feet A B A1 B1) :
  perpendicular_lines A1 B1 F :=
sorry

end NUMINAMATH_GPT_trajectory_of_M_lines_perpendicular_l1872_187269


namespace NUMINAMATH_GPT_usable_parking_lot_percentage_l1872_187299

theorem usable_parking_lot_percentage
  (length width : ℝ) (area_per_car : ℝ) (number_of_cars : ℝ)
  (h_len : length = 400)
  (h_wid : width = 500)
  (h_area_car : area_per_car = 10)
  (h_cars : number_of_cars = 16000) :
  ((number_of_cars * area_per_car) / (length * width) * 100) = 80 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_usable_parking_lot_percentage_l1872_187299


namespace NUMINAMATH_GPT_largest_by_changing_first_digit_l1872_187286

def value_with_digit_changed (d : Nat) : Float :=
  match d with
  | 1 => 0.86123
  | 2 => 0.78123
  | 3 => 0.76823
  | 4 => 0.76183
  | 5 => 0.76128
  | _ => 0.76123 -- default case

theorem largest_by_changing_first_digit :
  ∀ d : Nat, d ∈ [1, 2, 3, 4, 5] → value_with_digit_changed 1 ≥ value_with_digit_changed d :=
by
  intro d hd_list
  sorry

end NUMINAMATH_GPT_largest_by_changing_first_digit_l1872_187286


namespace NUMINAMATH_GPT_option_D_is_correct_l1872_187212

noncomputable def correct_operation : Prop := 
  (∀ x : ℝ, x + x ≠ 2 * x^2) ∧
  (∀ y : ℝ, 2 * y^3 + 3 * y^2 ≠ 5 * y^5) ∧
  (∀ x : ℝ, 2 * x - x ≠ 1) ∧
  (∀ x y : ℝ, 4 * x^3 * y^2 - (-2)^2 * x^3 * y^2 = 0)

theorem option_D_is_correct : correct_operation :=
by {
  -- We'll complete the proofs later
  sorry
}

end NUMINAMATH_GPT_option_D_is_correct_l1872_187212


namespace NUMINAMATH_GPT_line_parallel_slope_l1872_187260

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_line_parallel_slope_l1872_187260


namespace NUMINAMATH_GPT_hyperbola_focus_coordinates_l1872_187297

open Real

theorem hyperbola_focus_coordinates :
  ∃ x y : ℝ, (2 * x^2 - y^2 + 8 * x + 4 * y - 28 = 0) ∧
           ((x = -2 - 4 * sqrt 3 ∧ y = 2) ∨ (x = -2 + 4 * sqrt 3 ∧ y = 2)) := by sorry

end NUMINAMATH_GPT_hyperbola_focus_coordinates_l1872_187297


namespace NUMINAMATH_GPT_red_yellow_flowers_l1872_187201

theorem red_yellow_flowers
  (total : ℕ)
  (yellow_white : ℕ)
  (red_white : ℕ)
  (extra_red_over_white : ℕ)
  (H1 : total = 44)
  (H2 : yellow_white = 13)
  (H3 : red_white = 14)
  (H4 : extra_red_over_white = 4) :
  ∃ (red_yellow : ℕ), red_yellow = 17 := by
  sorry

end NUMINAMATH_GPT_red_yellow_flowers_l1872_187201


namespace NUMINAMATH_GPT_find_m_value_l1872_187282

theorem find_m_value (f : ℝ → ℝ) (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3) (h2 : f m = 6) : m = -(1 / 4) :=
sorry

end NUMINAMATH_GPT_find_m_value_l1872_187282


namespace NUMINAMATH_GPT_polynomial_value_at_minus_two_l1872_187239

def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

theorem polynomial_value_at_minus_two : f (-2) = 12 := by 
  sorry

end NUMINAMATH_GPT_polynomial_value_at_minus_two_l1872_187239


namespace NUMINAMATH_GPT_net_rate_of_pay_equals_39_dollars_per_hour_l1872_187215

-- Definitions of the conditions
def hours_travelled : ℕ := 3
def speed_per_hour : ℕ := 60
def car_consumption_rate : ℕ := 30
def earnings_per_mile : ℕ := 75  -- expressing $0.75 as 75 cents to avoid floating-point
def gasoline_cost_per_gallon : ℕ := 300  -- expressing $3.00 as 300 cents to avoid floating-point

-- Proof statement
theorem net_rate_of_pay_equals_39_dollars_per_hour : 
  (earnings_per_mile * (speed_per_hour * hours_travelled) - gasoline_cost_per_gallon * ((speed_per_hour * hours_travelled) / car_consumption_rate)) / hours_travelled = 3900 := 
by 
  -- The statement below essentially expresses 39 dollars per hour in cents (i.e., 3900 cents per hour).
  sorry

end NUMINAMATH_GPT_net_rate_of_pay_equals_39_dollars_per_hour_l1872_187215


namespace NUMINAMATH_GPT_bailey_towel_set_cost_l1872_187256

def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def cost_per_guest_set : ℝ := 40.00
def cost_per_master_set : ℝ := 50.00
def discount_rate : ℝ := 0.20

def total_cost_before_discount : ℝ := 
  (guest_bathroom_sets * cost_per_guest_set) + (master_bathroom_sets * cost_per_master_set)

def discount_amount : ℝ := total_cost_before_discount * discount_rate

def final_amount_spent : ℝ := total_cost_before_discount - discount_amount

theorem bailey_towel_set_cost : final_amount_spent = 224.00 := by sorry

end NUMINAMATH_GPT_bailey_towel_set_cost_l1872_187256


namespace NUMINAMATH_GPT_largest_integer_condition_l1872_187219

theorem largest_integer_condition (m a b : ℤ) 
  (h1 : m < 150) 
  (h2 : m > 50) 
  (h3 : m = 9 * a - 2) 
  (h4 : m = 6 * b - 4) : 
  m = 106 := 
sorry

end NUMINAMATH_GPT_largest_integer_condition_l1872_187219


namespace NUMINAMATH_GPT_cost_of_fencing_is_8750_rsquare_l1872_187225

variable (l w : ℝ)
variable (area : ℝ := 7500)
variable (cost_per_meter : ℝ := 0.25)
variable (ratio_lw : ℝ := 4/3)

theorem cost_of_fencing_is_8750_rsquare :
  (l / w = ratio_lw) → 
  (l * w = area) → 
  (2 * (l + w) * cost_per_meter = 87.50) :=
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_cost_of_fencing_is_8750_rsquare_l1872_187225


namespace NUMINAMATH_GPT_narrow_black_stripes_l1872_187254

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end NUMINAMATH_GPT_narrow_black_stripes_l1872_187254


namespace NUMINAMATH_GPT_car_speed_l1872_187207

theorem car_speed (distance time speed : ℝ)
  (h_const_speed : ∀ t : ℝ, t = time → speed = distance / t)
  (h_distance : distance = 48)
  (h_time : time = 8) :
  speed = 6 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_l1872_187207


namespace NUMINAMATH_GPT_sum_divisible_by_100_l1872_187238

theorem sum_divisible_by_100 (S : Finset ℤ) (hS : S.card = 200) : 
  ∃ T : Finset ℤ, T ⊆ S ∧ T.card = 100 ∧ (T.sum id) % 100 = 0 := 
  sorry

end NUMINAMATH_GPT_sum_divisible_by_100_l1872_187238


namespace NUMINAMATH_GPT_volunteer_arrangement_l1872_187245

theorem volunteer_arrangement (volunteers : Fin 5) (elderly : Fin 2) 
  (h1 : elderly.1 ≠ 0 ∧ elderly.1 ≠ 6) : 
  ∃ arrangements : ℕ, arrangements = 960 := 
sorry

end NUMINAMATH_GPT_volunteer_arrangement_l1872_187245


namespace NUMINAMATH_GPT_num_students_earning_B_l1872_187264

variables (nA nB nC nF : ℕ)

-- Conditions from the problem
def condition1 := nA = 6 * nB / 10
def condition2 := nC = 15 * nB / 10
def condition3 := nF = 4 * nB / 10
def condition4 := nA + nB + nC + nF = 50

-- The theorem to prove
theorem num_students_earning_B (nA nB nC nF : ℕ) : 
  condition1 nA nB → 
  condition2 nC nB → 
  condition3 nF nB → 
  condition4 nA nB nC nF → 
  nB = 14 :=
by
  sorry

end NUMINAMATH_GPT_num_students_earning_B_l1872_187264


namespace NUMINAMATH_GPT_part1_part2_l1872_187291

-- Part 1
theorem part1 (x y : ℝ) 
  (h1 : x + 2 * y = 9) 
  (h2 : 2 * x + y = 6) :
  (x - y = -3) ∧ (x + y = 5) :=
sorry

-- Part 2
theorem part2 (x y : ℝ) 
  (h1 : x + 2 = 5) 
  (h2 : y - 1 = 4) :
  x = 3 ∧ y = 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1872_187291


namespace NUMINAMATH_GPT_find_x_l1872_187267

theorem find_x (x : ℕ) (h1 : (31 : ℕ) ≤ 100) (h2 : (58 : ℕ) ≤ 100) (h3 : (98 : ℕ) ≤ 100) (h4 : 0 < x) (h5 : x ≤ 100)
               (h_mean_mode : ((31 + 58 + 98 + x + x) / 5 : ℚ) = 1.5 * x) : x = 34 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1872_187267


namespace NUMINAMATH_GPT_gcd_seq_consecutive_l1872_187289

-- Define the sequence b_n
def seq (n : ℕ) : ℕ := n.factorial + 2 * n

-- Main theorem statement
theorem gcd_seq_consecutive (n : ℕ) : n ≥ 0 → Nat.gcd (seq n) (seq (n + 1)) = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_gcd_seq_consecutive_l1872_187289


namespace NUMINAMATH_GPT_flower_beds_fraction_l1872_187227

-- Define the main problem parameters
def leg_length := (30 - 18) / 2
def triangle_area := (1 / 2) * (leg_length ^ 2)
def total_flower_bed_area := 2 * triangle_area
def yard_area := 30 * 6
def fraction_of_yard_occupied := total_flower_bed_area / yard_area

-- The theorem to be proved
theorem flower_beds_fraction :
  fraction_of_yard_occupied = 1/5 := by
  sorry

end NUMINAMATH_GPT_flower_beds_fraction_l1872_187227


namespace NUMINAMATH_GPT_measure_of_angle_B_and_area_of_triangle_l1872_187272

theorem measure_of_angle_B_and_area_of_triangle 
    (a b c : ℝ) 
    (A B C : ℝ) 
    (condition : 2 * c = a + (Real.cos A * (b / (Real.cos B))))
    (sum_sides : a + c = 3 * Real.sqrt 2)
    (side_b : b = 4)
    (angle_B : B = Real.pi / 3) :
    B = Real.pi / 3 ∧ 
    (1/2 * a * c * (Real.sin B) = Real.sqrt 3 / 6) :=
by
    sorry

end NUMINAMATH_GPT_measure_of_angle_B_and_area_of_triangle_l1872_187272


namespace NUMINAMATH_GPT_min_value_a_l1872_187290

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end NUMINAMATH_GPT_min_value_a_l1872_187290


namespace NUMINAMATH_GPT_arccos_cos_eq_l1872_187280

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end NUMINAMATH_GPT_arccos_cos_eq_l1872_187280


namespace NUMINAMATH_GPT_percentage_increase_l1872_187235

theorem percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1.1025 → x = 5.024 := 
sorry

end NUMINAMATH_GPT_percentage_increase_l1872_187235


namespace NUMINAMATH_GPT_partner_q_investment_time_l1872_187242

theorem partner_q_investment_time 
  (P Q R : ℝ)
  (Profit_p Profit_q Profit_r : ℝ)
  (Tp Tq Tr : ℝ)
  (h1 : P / Q = 7 / 5)
  (h2 : Q / R = 5 / 3)
  (h3 : Profit_p / Profit_q = 7 / 14)
  (h4 : Profit_q / Profit_r = 14 / 9)
  (h5 : Tp = 5)
  (h6 : Tr = 9) :
  Tq = 14 :=
by
  sorry

end NUMINAMATH_GPT_partner_q_investment_time_l1872_187242


namespace NUMINAMATH_GPT_fraction_value_l1872_187270

theorem fraction_value : (2 + 3 + 4 : ℚ) / (2 * 3 * 4) = 3 / 8 := 
by sorry

end NUMINAMATH_GPT_fraction_value_l1872_187270


namespace NUMINAMATH_GPT_function_monotonicity_l1872_187220

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_monotonicity :
  ∀ x₁ x₂, -Real.pi / 6 ≤ x₁ → x₁ ≤ x₂ → x₂ ≤ Real.pi / 3 → f x₁ ≤ f x₂ :=
by
  sorry

end NUMINAMATH_GPT_function_monotonicity_l1872_187220


namespace NUMINAMATH_GPT_polynomial_operation_correct_l1872_187234

theorem polynomial_operation_correct :
    ∀ (s t : ℝ), (s * t + 0.25 * s * t = 0) :=
by
  intros s t
  sorry

end NUMINAMATH_GPT_polynomial_operation_correct_l1872_187234


namespace NUMINAMATH_GPT_symmetric_point_l1872_187276

theorem symmetric_point (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : |y| = 3) : 
  (2, -3) = (-x, -y) :=
sorry

end NUMINAMATH_GPT_symmetric_point_l1872_187276


namespace NUMINAMATH_GPT_intersection_of_S_and_T_l1872_187294

noncomputable def S := {x : ℝ | x ≥ 2}
noncomputable def T := {x : ℝ | x ≤ 5}

theorem intersection_of_S_and_T : S ∩ T = {x : ℝ | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_S_and_T_l1872_187294


namespace NUMINAMATH_GPT_ratio_of_green_to_yellow_l1872_187206

def envelopes_problem (B Y G X : ℕ) : Prop :=
  B = 14 ∧
  Y = B - 6 ∧
  G = X * Y ∧
  B + Y + G = 46 ∧
  G / Y = 3

theorem ratio_of_green_to_yellow :
  ∃ B Y G X : ℕ, envelopes_problem B Y G X :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_green_to_yellow_l1872_187206


namespace NUMINAMATH_GPT_find_missing_square_l1872_187252

-- Defining the sequence as a list of natural numbers' squares
def square_sequence (n: ℕ) : ℕ := n * n

-- Proving the missing element in the given sequence is 36
theorem find_missing_square :
  (square_sequence 0 = 1) ∧ 
  (square_sequence 1 = 4) ∧ 
  (square_sequence 2 = 9) ∧ 
  (square_sequence 3 = 16) ∧ 
  (square_sequence 4 = 25) ∧ 
  (square_sequence 6 = 49) →
  square_sequence 5 = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_missing_square_l1872_187252


namespace NUMINAMATH_GPT_bees_hatch_every_day_l1872_187221

   /-- 
   Given:
   - The queen loses 900 bees every day.
   - The initial number of bees is 12500.
   - After 7 days, the total number of bees is 27201.
   
   Prove:
   - The number of bees hatching from the queen's eggs every day is 3001.
   -/
   
   theorem bees_hatch_every_day :
     ∃ x : ℕ, 12500 + 7 * (x - 900) = 27201 → x = 3001 :=
   sorry
   
end NUMINAMATH_GPT_bees_hatch_every_day_l1872_187221


namespace NUMINAMATH_GPT_min_value_of_x_plus_y_l1872_187262

-- Define the conditions
variables (x y : ℝ)
variables (h1 : x > 0) (h2 : y > 0) (h3 : y + 9 * x = x * y)

-- The statement of the problem
theorem min_value_of_x_plus_y : x + y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_value_of_x_plus_y_l1872_187262


namespace NUMINAMATH_GPT_contrapositive_equivalence_l1872_187277

-- Define the original proposition and its contrapositive
def original_proposition (q p : Prop) := q → p
def contrapositive (q p : Prop) := ¬q → ¬p

-- The theorem to prove
theorem contrapositive_equivalence (q p : Prop) :
  (original_proposition q p) ↔ (contrapositive q p) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_equivalence_l1872_187277


namespace NUMINAMATH_GPT_simplify_fraction_l1872_187217

theorem simplify_fraction (a b : ℕ) (h : a = 150) (hb : b = 450) : a / b = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1872_187217


namespace NUMINAMATH_GPT_value_of_expression_l1872_187278

theorem value_of_expression
  (a b : ℝ)
  (h₁ : a = 2 + Real.sqrt 3)
  (h₂ : b = 2 - Real.sqrt 3) :
  a^2 + 2 * a * b - b * (3 * a - b) = 13 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1872_187278


namespace NUMINAMATH_GPT_digit_after_decimal_is_4_l1872_187296

noncomputable def sum_fractions : ℚ := (2 / 9) + (3 / 11)

theorem digit_after_decimal_is_4 :
  (sum_fractions - sum_fractions.floor) * 10 = 4 :=
by
  sorry

end NUMINAMATH_GPT_digit_after_decimal_is_4_l1872_187296


namespace NUMINAMATH_GPT_measure_of_angle_C_l1872_187265

-- Define the conditions using Lean 4 constructs
variable (a b c : ℝ)
variable (A B C : ℝ) -- Measures of angles in triangle ABC
variable (triangle_ABC : (a * a + b * b - c * c = a * b))

-- Statement of the proof problem
theorem measure_of_angle_C (h : a^2 + b^2 - c^2 = ab) (h2 : 0 < C ∧ C < π) : C = π / 3 :=
by
  -- Proof will go here but is omitted with sorry
  sorry

end NUMINAMATH_GPT_measure_of_angle_C_l1872_187265


namespace NUMINAMATH_GPT_trigonometric_identity_l1872_187213

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.tan α = Real.sqrt 2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = (5 - Real.sqrt 2) / 3 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1872_187213


namespace NUMINAMATH_GPT_regular_rate_survey_l1872_187248

theorem regular_rate_survey (R : ℝ) 
  (total_surveys : ℕ := 50)
  (rate_increase : ℝ := 0.30)
  (cellphone_surveys : ℕ := 35)
  (total_earnings : ℝ := 605) :
  35 * (1.30 * R) + 15 * R = 605 → R = 10 :=
by
  sorry

end NUMINAMATH_GPT_regular_rate_survey_l1872_187248


namespace NUMINAMATH_GPT_virginia_ends_up_with_93_eggs_l1872_187210

-- Define the initial and subtracted number of eggs as conditions
def initial_eggs : ℕ := 96
def taken_eggs : ℕ := 3

-- The theorem we want to prove
theorem virginia_ends_up_with_93_eggs : (initial_eggs - taken_eggs) = 93 :=
by
  sorry

end NUMINAMATH_GPT_virginia_ends_up_with_93_eggs_l1872_187210


namespace NUMINAMATH_GPT_max_cars_per_div_100_is_20_l1872_187236

theorem max_cars_per_div_100_is_20 :
  let m : ℕ := Nat.succ (Nat.succ 0) -- represents m going to infinity
  let car_length : ℕ := 5
  let speed_factor : ℕ := 10
  let sensor_distance_per_hour : ℕ := speed_factor * 1000 * m
  let separation_distance : ℕ := car_length * (m + 1)
  let max_cars : ℕ := (sensor_distance_per_hour / separation_distance) * m
  Nat.floor ((2 * (max_cars : ℝ)) / 100) = 20 :=
by
  sorry

end NUMINAMATH_GPT_max_cars_per_div_100_is_20_l1872_187236


namespace NUMINAMATH_GPT_complex_transformation_l1872_187251

open Complex

def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

def rotation90 (z : ℂ) : ℂ :=
  z * I

theorem complex_transformation (z : ℂ) (center : ℂ) (scale : ℝ) :
  center = -1 + 2 * I → scale = 2 → z = 3 + I →
  rotation90 (dilation z center scale) = 4 + 7 * I :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  dsimp [dilation]
  dsimp [rotation90]
  sorry

end NUMINAMATH_GPT_complex_transformation_l1872_187251


namespace NUMINAMATH_GPT_Elina_garden_area_l1872_187231

theorem Elina_garden_area :
  ∀ (L W: ℝ),
    (30 * L = 1500) →
    (12 * (2 * (L + W)) = 1500) →
    (L * W = 625) :=
by
  intros L W h1 h2
  sorry

end NUMINAMATH_GPT_Elina_garden_area_l1872_187231


namespace NUMINAMATH_GPT_b_income_percentage_increase_l1872_187266

theorem b_income_percentage_increase (A_m B_m C_m : ℕ) (annual_income_A : ℕ)
  (C_income : C_m = 15000)
  (annual_income_A_cond : annual_income_A = 504000)
  (ratio_cond : A_m / B_m = 5 / 2)
  (A_m_cond : A_m = annual_income_A / 12) :
  ((B_m - C_m) * 100 / C_m) = 12 :=
by
  sorry

end NUMINAMATH_GPT_b_income_percentage_increase_l1872_187266


namespace NUMINAMATH_GPT_collinear_probability_in_rectangular_array_l1872_187298

noncomputable def prob_collinear (total_dots chosen_dots favorable_sets : ℕ) : ℚ :=
  favorable_sets / (Nat.choose total_dots chosen_dots)

theorem collinear_probability_in_rectangular_array :
  prob_collinear 20 4 2 = 2 / 4845 :=
by
  sorry

end NUMINAMATH_GPT_collinear_probability_in_rectangular_array_l1872_187298


namespace NUMINAMATH_GPT_second_order_derivative_l1872_187224

-- Define the parameterized functions x and y
noncomputable def x (t : ℝ) : ℝ := 1 / t
noncomputable def y (t : ℝ) : ℝ := 1 / (1 + t ^ 2)

-- Define the second-order derivative of y with respect to x
noncomputable def d2y_dx2 (t : ℝ) : ℝ := (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3

-- Prove the relationship based on given conditions
theorem second_order_derivative :
  ∀ t : ℝ, (∃ x y : ℝ, x = 1 / t ∧ y = 1 / (1 + t ^ 2)) → 
    (d2y_dx2 t) = (2 * (t^2 - 3) * t^4) / (1 + t^2) ^ 3 :=
by
  intros t ht
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_second_order_derivative_l1872_187224


namespace NUMINAMATH_GPT_monotonically_increasing_range_of_a_l1872_187283

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotonically_increasing_range_of_a :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-1 / 3 : ℝ) ≤ a ∧ a ≤ (1 / 3 : ℝ) :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_range_of_a_l1872_187283


namespace NUMINAMATH_GPT_selection_methods_count_l1872_187214

-- Define a function to compute combinations (n choose r)
def combination (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement
theorem selection_methods_count :
  combination 5 2 * combination 3 1 * combination 2 1 = 60 :=
by
  sorry

end NUMINAMATH_GPT_selection_methods_count_l1872_187214


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1872_187261

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific terms in arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Conditions given in the problem
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The proof goal
theorem arithmetic_sequence_problem : a 9 - 1/3 * a 11 = 16 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1872_187261


namespace NUMINAMATH_GPT_angle_measure_supplement_complement_l1872_187240

theorem angle_measure_supplement_complement (x : ℝ) 
    (h1 : 180 - x = 7 * (90 - x)) : 
    x = 75 := by
  sorry

end NUMINAMATH_GPT_angle_measure_supplement_complement_l1872_187240


namespace NUMINAMATH_GPT_reservoir_fullness_before_storm_l1872_187243

-- Definition of the conditions as Lean definitions
def storm_deposits : ℝ := 120 -- in billion gallons
def reservoir_percentage_after_storm : ℝ := 85 -- percentage
def original_contents : ℝ := 220 -- in billion gallons

-- The proof statement
theorem reservoir_fullness_before_storm (storm_deposits reservoir_percentage_after_storm original_contents : ℝ) : 
    (169 / 340) * 100 = 49.7 := 
  sorry

end NUMINAMATH_GPT_reservoir_fullness_before_storm_l1872_187243


namespace NUMINAMATH_GPT_triangle_table_distinct_lines_l1872_187230

theorem triangle_table_distinct_lines (a : ℕ) (h : a > 1) : 
  ∀ (n : ℕ) (line : ℕ → ℕ), 
  (line 0 = a) → 
  (∀ k, line (2*k + 1) = line k ^ 2 ∧ line (2*k + 2) = line k + 1) → 
  ∀ i j, i < 2^n → j < 2^n → (i ≠ j → line i ≠ line j) := 
by {
  sorry
}

end NUMINAMATH_GPT_triangle_table_distinct_lines_l1872_187230


namespace NUMINAMATH_GPT_graph_of_equation_is_two_intersecting_lines_l1872_187200

theorem graph_of_equation_is_two_intersecting_lines :
  ∀ x y : ℝ, (x + 3 * y) ^ 3 = x ^ 3 + 9 * y ^ 3 ↔ (x = 0 ∨ y = 0 ∨ x + 3 * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_graph_of_equation_is_two_intersecting_lines_l1872_187200


namespace NUMINAMATH_GPT_inequality_of_sums_l1872_187287

theorem inequality_of_sums (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_ineq : a > b ∧ b > c ∧ c > d) :
  (a + b + c + d)^2 > a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_of_sums_l1872_187287


namespace NUMINAMATH_GPT_find_a_l1872_187229

theorem find_a (a : ℝ) (h : ∃ x, x = -1 ∧ 4 * x^3 + 2 * a * x = 8) : a = -6 :=
sorry

end NUMINAMATH_GPT_find_a_l1872_187229


namespace NUMINAMATH_GPT_transform_expression_to_product_l1872_187244

open Real

noncomputable def transform_expression (α : ℝ) : ℝ :=
  4.66 * sin (5 * π / 2 + 4 * α) - (sin (5 * π / 2 + 2 * α)) ^ 6 + (cos (7 * π / 2 - 2 * α)) ^ 6

theorem transform_expression_to_product (α : ℝ) :
  transform_expression α = (1 / 8) * sin (4 * α) * sin (8 * α) :=
by
  sorry

end NUMINAMATH_GPT_transform_expression_to_product_l1872_187244


namespace NUMINAMATH_GPT_simple_interest_rate_l1872_187271

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 15000) (hSI : SI = 6000) (hT : T = 8) :
  ∃ R : ℝ, (SI = P * R * T / 100) ∧ R = 5 :=
by
  use 5
  field_simp [hP, hSI, hT]
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1872_187271


namespace NUMINAMATH_GPT_largest_quotient_is_25_l1872_187232

def largest_quotient_set : Set ℤ := {-25, -4, -1, 1, 3, 9}

theorem largest_quotient_is_25 :
  ∃ (a b : ℤ), a ∈ largest_quotient_set ∧ b ∈ largest_quotient_set ∧ b ≠ 0 ∧ (a : ℚ) / b = 25 := by
  sorry

end NUMINAMATH_GPT_largest_quotient_is_25_l1872_187232


namespace NUMINAMATH_GPT_cost_of_iphone_l1872_187241

def trade_in_value : ℕ := 240
def weekly_earnings : ℕ := 80
def weeks_worked : ℕ := 7
def total_earnings := weekly_earnings * weeks_worked
def total_money := total_earnings + trade_in_value
def new_iphone_cost : ℕ := 800

theorem cost_of_iphone :
  total_money = new_iphone_cost := by
  sorry

end NUMINAMATH_GPT_cost_of_iphone_l1872_187241


namespace NUMINAMATH_GPT_mabel_shark_ratio_l1872_187293

variables (F1 F2 sharks_total sharks_day1 sharks_day2 ratio : ℝ)
variables (fish_day1 := 15)
variables (shark_percentage := 0.25)
variables (total_sharks := 15)

noncomputable def ratio_of_fish_counts := (F2 / F1)

theorem mabel_shark_ratio 
    (fish_day1 : ℝ := 15)
    (shark_percentage : ℝ := 0.25)
    (total_sharks : ℝ := 15)
    (sharks_day1 := 0.25 * fish_day1)
    (sharks_day2 := total_sharks - sharks_day1)
    (F2 := sharks_day2 / shark_percentage)
    (ratio := F2 / fish_day1):
    ratio = 16 / 5 :=
by
  sorry

end NUMINAMATH_GPT_mabel_shark_ratio_l1872_187293


namespace NUMINAMATH_GPT_tangent_slope_through_origin_l1872_187292

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^a + 1

theorem tangent_slope_through_origin (a : ℝ) (h : curve a 1 = 2) 
  (tangent_passing_through_origin : ∀ y, (y - 2 = a * (1 - 0)) → y = 0): a = 2 := 
sorry

end NUMINAMATH_GPT_tangent_slope_through_origin_l1872_187292


namespace NUMINAMATH_GPT_liam_birthday_next_monday_2018_l1872_187205

-- Define year advancement rules
def is_leap_year (y : ℕ) : Prop :=
  (y % 4 = 0 ∧ y % 100 ≠ 0) ∨ (y % 400 = 0)

-- Define function to calculate next weekday
def next_weekday (current_day : ℕ) (years_elapsed : ℕ) : ℕ :=
  let advance := (years_elapsed / 4) * 2 + (years_elapsed % 4)
  (current_day + advance) % 7

theorem liam_birthday_next_monday_2018 :
  (next_weekday 4 3 = 0) :=
sorry

end NUMINAMATH_GPT_liam_birthday_next_monday_2018_l1872_187205


namespace NUMINAMATH_GPT_exists_a_star_b_eq_a_l1872_187246

variable {S : Type*} [CommSemigroup S]

def exists_element_in_S (star : S → S → S) : Prop :=
  ∃ a : S, ∀ b : S, star a b = a

theorem exists_a_star_b_eq_a
  (star : S → S → S)
  (comm : ∀ a b : S, star a b = star b a)
  (assoc : ∀ a b c : S, star (star a b) c = star a (star b c))
  (exists_a : ∃ a : S, star a a = a) :
  exists_element_in_S star := sorry

end NUMINAMATH_GPT_exists_a_star_b_eq_a_l1872_187246


namespace NUMINAMATH_GPT_gdp_scientific_notation_l1872_187204

theorem gdp_scientific_notation (gdp : ℝ) (h : gdp = 338.8 * 10^9) : gdp = 3.388 * 10^10 :=
by sorry

end NUMINAMATH_GPT_gdp_scientific_notation_l1872_187204


namespace NUMINAMATH_GPT_car_average_speed_l1872_187249

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end NUMINAMATH_GPT_car_average_speed_l1872_187249


namespace NUMINAMATH_GPT_min_value_of_trig_expression_l1872_187237

open Real

theorem min_value_of_trig_expression (α : ℝ) (h₁ : sin α ≠ 0) (h₂ : cos α ≠ 0) : 
  (9 / (sin α)^2 + 1 / (cos α)^2) ≥ 16 :=
  sorry

end NUMINAMATH_GPT_min_value_of_trig_expression_l1872_187237


namespace NUMINAMATH_GPT_total_amount_distributed_l1872_187218

def number_of_persons : ℕ := 22
def amount_per_person : ℕ := 1950

theorem total_amount_distributed : (number_of_persons * amount_per_person) = 42900 := by
  sorry

end NUMINAMATH_GPT_total_amount_distributed_l1872_187218


namespace NUMINAMATH_GPT_cats_sold_during_sale_l1872_187247

-- Definitions based on conditions in a)
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def cats_left : ℕ := 8
def total_cats := siamese_cats + house_cats

-- Proof statement
theorem cats_sold_during_sale : total_cats - cats_left = 10 := by
  sorry

end NUMINAMATH_GPT_cats_sold_during_sale_l1872_187247


namespace NUMINAMATH_GPT_smallest_k_divides_polynomial_l1872_187202

theorem smallest_k_divides_polynomial :
  ∃ (k : ℕ), k > 0 ∧ (∀ z : ℂ, z ≠ 0 → 
    (z ^ 11 + z ^ 9 + z ^ 7 + z ^ 6 + z ^ 5 + z ^ 2 + 1) ∣ (z ^ k - 1)) ∧ k = 11 := by
  sorry

end NUMINAMATH_GPT_smallest_k_divides_polynomial_l1872_187202


namespace NUMINAMATH_GPT_distance_after_3rd_turn_l1872_187226

theorem distance_after_3rd_turn (d1 d2 d4 total_distance : ℕ) 
  (h1 : d1 = 5) 
  (h2 : d2 = 8) 
  (h4 : d4 = 0) 
  (h_total : total_distance = 23) : 
  total_distance - (d1 + d2 + d4) = 10 := 
  sorry

end NUMINAMATH_GPT_distance_after_3rd_turn_l1872_187226


namespace NUMINAMATH_GPT_percent_of_100_is_30_l1872_187250

theorem percent_of_100_is_30 : (30 / 100) * 100 = 30 := 
by
  sorry

end NUMINAMATH_GPT_percent_of_100_is_30_l1872_187250


namespace NUMINAMATH_GPT_highest_red_ball_probability_l1872_187279

theorem highest_red_ball_probability :
  ∀ (total balls red yellow black : ℕ),
    total = 10 →
    red = 7 →
    yellow = 2 →
    black = 1 →
    (red / total) > (yellow / total) ∧ (red / total) > (black / total) :=
by
  intro total balls red yellow black
  intro h_total h_red h_yellow h_black
  sorry

end NUMINAMATH_GPT_highest_red_ball_probability_l1872_187279


namespace NUMINAMATH_GPT_total_birds_from_monday_to_wednesday_l1872_187228

def birds_monday := 70
def birds_tuesday := birds_monday / 2
def birds_wednesday := birds_tuesday + 8
def total_birds := birds_monday + birds_tuesday + birds_wednesday

theorem total_birds_from_monday_to_wednesday : total_birds = 148 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end NUMINAMATH_GPT_total_birds_from_monday_to_wednesday_l1872_187228


namespace NUMINAMATH_GPT_proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l1872_187274

variable (x y : ℤ)

def proposition_A := (x ≠ 1000 ∨ y ≠ 1002)
def proposition_B := (x + y ≠ 2002)

theorem proposition_A_necessary_for_B : proposition_B x y → proposition_A x y := by
  sorry

theorem proposition_A_not_sufficient_for_B : ¬ (proposition_A x y → proposition_B x y) := by
  sorry

end NUMINAMATH_GPT_proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l1872_187274


namespace NUMINAMATH_GPT_a_share_is_2500_l1872_187273

theorem a_share_is_2500
  (x : ℝ)
  (h1 : 4 * x = 3 * x + 500)
  (h2 : 6 * x = 2 * 2 * x) : 5 * x = 2500 :=
by 
  sorry

end NUMINAMATH_GPT_a_share_is_2500_l1872_187273


namespace NUMINAMATH_GPT_PetrovFamilySavings_l1872_187223

def parents_salary : ℕ := 56000
def grandmothers_pension : ℕ := 14300
def sons_scholarship : ℕ := 2500

def communal_services : ℕ := 9800
def food_expenses : ℕ := 21000
def transport_expenses : ℕ := 3200
def leisure_expenses : ℕ := 5200
def other_expenses : ℕ := 15000

def total_income : ℕ := parents_salary + grandmothers_pension + sons_scholarship
def total_expenses : ℕ := communal_services + food_expenses + transport_expenses + leisure_expenses + other_expenses

def surplus : ℕ := total_income - total_expenses
def deposit : ℕ := surplus / 10

def amount_set_aside : ℕ := surplus - deposit

theorem PetrovFamilySavings : amount_set_aside = 16740 := by
  sorry

end NUMINAMATH_GPT_PetrovFamilySavings_l1872_187223


namespace NUMINAMATH_GPT_sharon_distance_to_mothers_house_l1872_187255

noncomputable def total_distance (x : ℝ) :=
  x / 240

noncomputable def adjusted_speed (x : ℝ) :=
  x / 240 - 1 / 4

theorem sharon_distance_to_mothers_house (x : ℝ) (h1 : x / 240 = total_distance x) 
(h2 : adjusted_speed x = x / 240 - 1 / 4) 
(h3 : 120 + 120 * x / (x - 60) = 330) : 
x = 140 := 
by 
  sorry

end NUMINAMATH_GPT_sharon_distance_to_mothers_house_l1872_187255


namespace NUMINAMATH_GPT_converse_not_true_without_negatives_l1872_187211

theorem converse_not_true_without_negatives (a b c d : ℕ) (h : a + d = b + c) : ¬(a - c = b - d) :=
by
  sorry

end NUMINAMATH_GPT_converse_not_true_without_negatives_l1872_187211


namespace NUMINAMATH_GPT_sequence_properties_l1872_187216

-- Define the sequences a_n and b_n
noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def b (n : ℕ) : ℕ := sorry

-- Define the conditions
axiom h1 : a 1 = 1
axiom h2 : b 1 = 1
axiom h3 : ∀ n, b (n + 1) ^ 2 = b n * b (n + 2)
axiom h4 : 9 * (b 3) ^ 2 = b 2 * b 6
axiom h5 : ∀ n, b (n + 1) / a (n + 1) = b n / (a n + 2 * b n)

-- Define the theorem to prove
theorem sequence_properties :
  (∀ n, a n = (2 * n - 1) * 3 ^ (n - 1)) ∧
  (∀ n, (a n) / (b n) = (a (n + 1)) / (b (n + 1)) + 2) := by
  sorry

end NUMINAMATH_GPT_sequence_properties_l1872_187216


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1872_187263

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1872_187263


namespace NUMINAMATH_GPT_find_angle_degree_l1872_187203

theorem find_angle_degree (x : ℝ) (h : 90 - x = (1 / 3) * (180 - x) + 20) : x = 75 := by
    sorry

end NUMINAMATH_GPT_find_angle_degree_l1872_187203


namespace NUMINAMATH_GPT_johnson_family_seating_l1872_187295

def johnson_family_boys : ℕ := 5
def johnson_family_girls : ℕ := 4
def total_chairs : ℕ := 9
def total_arrangements : ℕ := Nat.factorial total_chairs

noncomputable def seating_arrangements_with_at_least_3_boys : ℕ :=
  let three_boys_block_ways := 7 * (5 * 4 * 3) * Nat.factorial 6
  total_arrangements - three_boys_block_ways

theorem johnson_family_seating : seating_arrangements_with_at_least_3_boys = 60480 := by
  unfold seating_arrangements_with_at_least_3_boys
  sorry

end NUMINAMATH_GPT_johnson_family_seating_l1872_187295


namespace NUMINAMATH_GPT_wire_ratio_l1872_187209

theorem wire_ratio (a b : ℝ) (h : (a / 4) ^ 2 = (b / (2 * Real.pi)) ^ 2 * Real.pi) : a / b = 2 / Real.sqrt Real.pi := by
  sorry

end NUMINAMATH_GPT_wire_ratio_l1872_187209


namespace NUMINAMATH_GPT_cost_of_cheaper_feed_l1872_187259

theorem cost_of_cheaper_feed (C : ℝ) 
  (h1 : 35 * 0.36 = 12.6)
  (h2 : 18 * 0.53 = 9.54)
  (h3 : 17 * C + 9.54 = 12.6) :
  C = 0.18 := sorry

end NUMINAMATH_GPT_cost_of_cheaper_feed_l1872_187259


namespace NUMINAMATH_GPT_expressions_equal_iff_l1872_187257

theorem expressions_equal_iff (x y z : ℝ) : x + y + z = 0 ↔ x + yz = (x + y) * (x + z) :=
by
  sorry

end NUMINAMATH_GPT_expressions_equal_iff_l1872_187257


namespace NUMINAMATH_GPT_sides_of_figures_intersection_l1872_187222

theorem sides_of_figures_intersection (n p q : ℕ) (h1 : p ≠ 0) (h2 : q ≠ 0) :
  p + q ≤ n + 4 :=
by sorry

end NUMINAMATH_GPT_sides_of_figures_intersection_l1872_187222


namespace NUMINAMATH_GPT_altitude_of_triangle_l1872_187275

theorem altitude_of_triangle (b h_t h_p : ℝ) (hb : b ≠ 0) 
  (area_eq : b * h_p = (1/2) * b * h_t) 
  (h_p_def : h_p = 100) : h_t = 200 :=
by
  sorry

end NUMINAMATH_GPT_altitude_of_triangle_l1872_187275


namespace NUMINAMATH_GPT_set_union_complement_eq_l1872_187268

def P : Set ℝ := {x | x^2 - 4 * x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}
def R_complement_Q : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem set_union_complement_eq :
  P ∪ R_complement_Q = {x | x ≤ -2} ∪ {x | x ≥ 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_set_union_complement_eq_l1872_187268


namespace NUMINAMATH_GPT_total_amount_for_gifts_l1872_187284

theorem total_amount_for_gifts (workers_per_block : ℕ) (worth_per_gift : ℕ) (number_of_blocks : ℕ)
  (h1 : workers_per_block = 100) (h2 : worth_per_gift = 4) (h3 : number_of_blocks = 10) :
  (workers_per_block * worth_per_gift * number_of_blocks = 4000) := by
  sorry

end NUMINAMATH_GPT_total_amount_for_gifts_l1872_187284


namespace NUMINAMATH_GPT_g_neither_even_nor_odd_l1872_187233

noncomputable def g (x : ℝ) : ℝ := (1 / (3^x - 2)) + 1

theorem g_neither_even_nor_odd : ¬ (∀ x : ℝ, g x = g (-x)) ∧ ¬ (∀ x : ℝ, g x = -g (-x)) := 
by sorry

end NUMINAMATH_GPT_g_neither_even_nor_odd_l1872_187233


namespace NUMINAMATH_GPT_bob_weight_l1872_187253

theorem bob_weight (j b : ℝ) (h1 : j + b = 200) (h2 : b - j = b / 3) : b = 120 :=
sorry

end NUMINAMATH_GPT_bob_weight_l1872_187253


namespace NUMINAMATH_GPT_total_travel_time_l1872_187258

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_l1872_187258


namespace NUMINAMATH_GPT_original_concentration_A_l1872_187285

-- Definitions of initial conditions and parameters
def mass_A : ℝ := 2000 -- 2 kg in grams
def mass_B : ℝ := 3000 -- 3 kg in grams
def pour_out_A : ℝ := 0.15 -- 15% poured out from bottle A
def pour_out_B : ℝ := 0.30 -- 30% poured out from bottle B
def mixed_concentration1 : ℝ := 27.5 -- 27.5% concentration after first mix
def pour_out_restored : ℝ := 0.40 -- 40% poured out again

-- Using the calculated remaining mass and concentration to solve the proof
theorem original_concentration_A (x y : ℝ) 
  (h1 : 300 * x + 900 * y = 27.5 * (300 + 900)) 
  (h2 : (1700 * x + 300 * 27.5) * 0.4 / (2000 * 0.4) + (2100 * y + 900 * 27.5) * 0.4 / (3000 * 0.4) = 26) : 
  x = 20 :=
by 
  -- Skipping the proof. The proof should involve solving the system of equations.
  sorry

end NUMINAMATH_GPT_original_concentration_A_l1872_187285


namespace NUMINAMATH_GPT_Cindy_correct_answer_l1872_187288

theorem Cindy_correct_answer (x : ℕ) (h : (x - 14) / 4 = 28) : ((x - 5) / 7) * 4 = 69 := by
  sorry

end NUMINAMATH_GPT_Cindy_correct_answer_l1872_187288


namespace NUMINAMATH_GPT_average_payment_debt_l1872_187281

theorem average_payment_debt :
  let total_payments := 65
  let first_20_payment := 410
  let increment := 65
  let remaining_payment := first_20_payment + increment
  let first_20_total := 20 * first_20_payment
  let remaining_total := 45 * remaining_payment
  let total_paid := first_20_total + remaining_total
  let average_payment := total_paid / total_payments
  average_payment = 455 := by sorry

end NUMINAMATH_GPT_average_payment_debt_l1872_187281


namespace NUMINAMATH_GPT_percentage_discount_four_friends_l1872_187208

theorem percentage_discount_four_friends 
  (num_friends : ℕ)
  (original_price : ℝ)
  (total_spent : ℝ)
  (item_per_friend : ℕ)
  (total_items : ℕ)
  (each_spent : ℝ)
  (discount_percentage : ℝ):
  num_friends = 4 →
  original_price = 20 →
  total_spent = 40 →
  item_per_friend = 1 →
  total_items = num_friends * item_per_friend →
  each_spent = total_spent / num_friends →
  discount_percentage = ((original_price - each_spent) / original_price) * 100 →
  discount_percentage = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_discount_four_friends_l1872_187208
