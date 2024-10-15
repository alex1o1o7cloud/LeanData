import Mathlib

namespace NUMINAMATH_GPT_gloria_money_left_l383_38332

theorem gloria_money_left 
  (cost_of_cabin : ℕ) (cash : ℕ)
  (num_cypress_trees num_pine_trees num_maple_trees : ℕ)
  (price_per_cypress_tree price_per_pine_tree price_per_maple_tree : ℕ)
  (money_left : ℕ)
  (h_cost_of_cabin : cost_of_cabin = 129000)
  (h_cash : cash = 150)
  (h_num_cypress_trees : num_cypress_trees = 20)
  (h_num_pine_trees : num_pine_trees = 600)
  (h_num_maple_trees : num_maple_trees = 24)
  (h_price_per_cypress_tree : price_per_cypress_tree = 100)
  (h_price_per_pine_tree : price_per_pine_tree = 200)
  (h_price_per_maple_tree : price_per_maple_tree = 300)
  (h_money_left : money_left = (num_cypress_trees * price_per_cypress_tree + 
                                num_pine_trees * price_per_pine_tree + 
                                num_maple_trees * price_per_maple_tree + 
                                cash) - cost_of_cabin)
  : money_left = 350 :=
by
  sorry

end NUMINAMATH_GPT_gloria_money_left_l383_38332


namespace NUMINAMATH_GPT_students_not_playing_either_game_l383_38325

theorem students_not_playing_either_game
  (total_students : ℕ) -- There are 20 students in the class
  (play_basketball : ℕ) -- Half of them play basketball
  (play_volleyball : ℕ) -- Two-fifths of them play volleyball
  (play_both : ℕ) -- One-tenth of them play both basketball and volleyball
  (h_total : total_students = 20)
  (h_basketball : play_basketball = 10)
  (h_volleyball : play_volleyball = 8)
  (h_both : play_both = 2) :
  total_students - (play_basketball + play_volleyball - play_both) = 4 := by
  sorry

end NUMINAMATH_GPT_students_not_playing_either_game_l383_38325


namespace NUMINAMATH_GPT_sum_of_cubes_of_roots_l383_38308

theorem sum_of_cubes_of_roots (P : Polynomial ℝ)
  (hP : P = Polynomial.C (-1) + Polynomial.X ^ 3 - Polynomial.C 3 * Polynomial.X) 
  (x1 x2 x3 : ℝ) 
  (hr : P.eval x1 = 0 ∧ P.eval x2 = 0 ∧ P.eval x3 = 0) :
  x1^3 + x2^3 + x3^3 = 3 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_of_roots_l383_38308


namespace NUMINAMATH_GPT_correct_calculation_l383_38304

theorem correct_calculation (x : ℕ) (h : x / 9 = 30) : x - 37 = 233 :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l383_38304


namespace NUMINAMATH_GPT_fraction_given_to_classmates_l383_38399

theorem fraction_given_to_classmates
  (total_boxes : ℕ) (pens_per_box : ℕ)
  (percentage_to_friends : ℝ) (pens_left_after_classmates : ℕ) :
  total_boxes = 20 →
  pens_per_box = 5 →
  percentage_to_friends = 0.40 →
  pens_left_after_classmates = 45 →
  (15 / (total_boxes * pens_per_box - percentage_to_friends * total_boxes * pens_per_box)) = 1 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fraction_given_to_classmates_l383_38399


namespace NUMINAMATH_GPT_final_score_is_89_l383_38366

def final_score (s_e s_l s_b : ℝ) (p_e p_l p_b : ℝ) : ℝ :=
  s_e * p_e + s_l * p_l + s_b * p_b

theorem final_score_is_89 :
  final_score 95 92 80 0.4 0.25 0.35 = 89 := 
by
  sorry

end NUMINAMATH_GPT_final_score_is_89_l383_38366


namespace NUMINAMATH_GPT_max_value_f_l383_38334

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem max_value_f (h : ∀ ε > (0 : ℝ), ∃ x : ℝ, x < 1 ∧ ε < f x) : ∀ x : ℝ, x < 1 → f x ≤ -1 :=
by
  intros x hx
  dsimp [f]
  -- Proof steps are omitted.
  sorry

example (h: ∀ ε > 0, ∃ x : ℝ, x < 1 ∧ ε < f x) : ∃ x : ℝ, x < 1 ∧ f x = -1 :=
by
  use 0
  -- Proof steps are omitted.
  sorry

end NUMINAMATH_GPT_max_value_f_l383_38334


namespace NUMINAMATH_GPT_price_of_cookie_cookie_price_verification_l383_38348

theorem price_of_cookie 
  (total_spent : ℝ) 
  (cost_per_cupcake : ℝ)
  (num_cupcakes : ℕ)
  (cost_per_doughnut : ℝ)
  (num_doughnuts : ℕ)
  (cost_per_pie_slice : ℝ)
  (num_pie_slices : ℕ)
  (num_cookies : ℕ)
  (total_cookies_cost : ℝ)
  (total_cost : ℝ) :
  (num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  + num_cookies * total_cookies_cost = total_spent) → 
  total_cookies_cost = 0.60 :=
by
  sorry

noncomputable def sophie_cookies_price : ℝ := 
  let total_cost := 33
  let num_cupcakes := 5
  let cost_per_cupcake := 2
  let num_doughnuts := 6
  let cost_per_doughnut := 1
  let num_pie_slices := 4
  let cost_per_pie_slice := 2
  let num_cookies := 15
  let total_spent_on_other_items := 
    num_cupcakes * cost_per_cupcake + num_doughnuts * cost_per_doughnut + num_pie_slices * cost_per_pie_slice 
  let remaining_cost := total_cost - total_spent_on_other_items 
  remaining_cost / num_cookies

theorem cookie_price_verification :
  sophie_cookies_price = 0.60 :=
by
  sorry

end NUMINAMATH_GPT_price_of_cookie_cookie_price_verification_l383_38348


namespace NUMINAMATH_GPT_find_b_l383_38337

theorem find_b 
    (x1 x2 b c : ℝ)
    (h_distinct : x1 ≠ x2)
    (h_root_x : ∀ x, (x^2 + 5 * b * x + c = 0) → x = x1 ∨ x = x2)
    (h_common_root : ∃ y, (y^2 + 2 * x1 * y + 2 * x2 = 0) ∧ (y^2 + 2 * x2 * y + 2 * x1 = 0)) :
  b = 1 / 10 := 
sorry

end NUMINAMATH_GPT_find_b_l383_38337


namespace NUMINAMATH_GPT_toy_playing_dogs_ratio_l383_38377

theorem toy_playing_dogs_ratio
  (d_t : ℕ) (d_r : ℕ) (d_n : ℕ) (d_b : ℕ) (d_p : ℕ)
  (h1 : d_t = 88)
  (h2 : d_r = 12)
  (h3 : d_n = 10)
  (h4 : d_b = d_t / 4)
  (h5 : d_p = d_t - d_r - d_b - d_n) :
  d_p / d_t = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_toy_playing_dogs_ratio_l383_38377


namespace NUMINAMATH_GPT_maximum_value_of_n_l383_38380

noncomputable def max_n (a b c : ℝ) (n : ℕ) :=
  a > b ∧ b > c ∧ (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2

theorem maximum_value_of_n (a b c : ℝ) (n : ℕ) : 
  a > b → b > c → (∀ (n : ℕ), (1 / (a - b) + 1 / (b - c) ≥ n^2 / (a - c))) → n ≤ 2 :=
  by sorry

end NUMINAMATH_GPT_maximum_value_of_n_l383_38380


namespace NUMINAMATH_GPT_quiz_probability_l383_38386

theorem quiz_probability :
  let probMCQ := 1/3
  let probTF1 := 1/2
  let probTF2 := 1/2
  probMCQ * probTF1 * probTF2 = 1/12 := by
  sorry

end NUMINAMATH_GPT_quiz_probability_l383_38386


namespace NUMINAMATH_GPT_mortgage_payoff_months_l383_38378

-- Declare the initial payment (P), the common ratio (r), and the total amount (S)
def initial_payment : ℕ := 100
def common_ratio : ℕ := 3
def total_amount : ℕ := 12100

-- Define a function that calculates the sum of a geometric series
noncomputable def geom_series_sum (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * (1 - r ^ n) / (1 - r)

-- The statement we need to prove
theorem mortgage_payoff_months : ∃ n : ℕ, geom_series_sum initial_payment common_ratio n = total_amount :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_mortgage_payoff_months_l383_38378


namespace NUMINAMATH_GPT_cylinder_not_occupied_volume_l383_38374

theorem cylinder_not_occupied_volume :
  let r := 10
  let h_cylinder := 30
  let h_full_cone := 10
  let volume_cylinder := π * r^2 * h_cylinder
  let volume_full_cone := (1 / 3) * π * r^2 * h_full_cone
  let volume_half_cone := (1 / 2) * volume_full_cone
  let volume_unoccupied := volume_cylinder - (volume_full_cone + volume_half_cone)
  volume_unoccupied = 2500 * π := 
by
  sorry

end NUMINAMATH_GPT_cylinder_not_occupied_volume_l383_38374


namespace NUMINAMATH_GPT_ratio_of_sides_l383_38303

theorem ratio_of_sides 
  (a b c d : ℝ) 
  (h1 : (a * b) / (c * d) = 0.16) 
  (h2 : b / d = 2 / 5) : 
  a / c = 0.4 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_sides_l383_38303


namespace NUMINAMATH_GPT_apple_order_for_month_l383_38323

def Chandler_apples (week : ℕ) : ℕ :=
  23 + 2 * week

def Lucy_apples (week : ℕ) : ℕ :=
  19 - week

def Ross_apples : ℕ :=
  15

noncomputable def total_apples : ℕ :=
  (Chandler_apples 0 + Chandler_apples 1 + Chandler_apples 2 + Chandler_apples 3) +
  (Lucy_apples 0 + Lucy_apples 1 + Lucy_apples 2 + Lucy_apples 3) +
  (Ross_apples * 4)

theorem apple_order_for_month : total_apples = 234 := by
  sorry

end NUMINAMATH_GPT_apple_order_for_month_l383_38323


namespace NUMINAMATH_GPT_households_both_brands_l383_38339

theorem households_both_brands
  (T : ℕ) (N : ℕ) (A : ℕ) (B : ℕ)
  (hT : T = 300) (hN : N = 80) (hA : A = 60) (hB : ∃ X : ℕ, B = 3 * X ∧ T = N + A + B + X) :
  ∃ X : ℕ, X = 40 :=
by
  -- Upon extracting values from conditions, solving for both brand users X = 40
  sorry

end NUMINAMATH_GPT_households_both_brands_l383_38339


namespace NUMINAMATH_GPT_largest_five_digit_palindromic_number_l383_38317

theorem largest_five_digit_palindromic_number (a b c d e : ℕ)
  (h1 : ∃ a b c, 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧
                 ∃ d e, 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
                 (10001 * a + 1010 * b + 100 * c = 45 * (1001 * d + 110 * e))) :
  10001 * 5 + 1010 * 9 + 100 * 8 = 59895 :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_palindromic_number_l383_38317


namespace NUMINAMATH_GPT_roots_of_f_l383_38369

noncomputable def f (a x : ℝ) : ℝ := x - Real.log (a * x)

theorem roots_of_f (a : ℝ) :
  (a < 0 → ¬∃ x : ℝ, f a x = 0) ∧
  (0 < a ∧ a < Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a = Real.exp 1 → ∃! x : ℝ, f a x = 0) ∧
  (a > Real.exp 1 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :=
sorry

end NUMINAMATH_GPT_roots_of_f_l383_38369


namespace NUMINAMATH_GPT_snooker_tournament_total_cost_l383_38359

def VIP_cost : ℝ := 45
def GA_cost : ℝ := 20
def total_tickets_sold : ℝ := 320
def vip_and_general_admission_relationship := 276

def total_cost_of_tickets : ℝ := 6950

theorem snooker_tournament_total_cost 
  (V G : ℝ)
  (h1 : VIP_cost * V + GA_cost * G = total_cost_of_tickets)
  (h2 : V + G = total_tickets_sold)
  (h3 : V = G - vip_and_general_admission_relationship) : 
  VIP_cost * V + GA_cost * G = total_cost_of_tickets := 
by {
  sorry
}

end NUMINAMATH_GPT_snooker_tournament_total_cost_l383_38359


namespace NUMINAMATH_GPT_robot_cost_max_units_A_l383_38354

noncomputable def cost_price_A (x : ℕ) := 1600
noncomputable def cost_price_B (x : ℕ) := 2800

theorem robot_cost (x : ℕ) (y : ℕ) (a : ℕ) (b : ℕ) :
  y = 2 * x - 400 →
  a = 96000 →
  b = 168000 →
  a / x = 6000 →
  b / y = 6000 →
  (x = 1600 ∧ y = 2800) :=
by sorry

theorem max_units_A (m n total_units : ℕ) : 
  total_units = 100 →
  m + n = 100 →
  m ≤ 2 * n →
  m ≤ 66 :=
by sorry

end NUMINAMATH_GPT_robot_cost_max_units_A_l383_38354


namespace NUMINAMATH_GPT_correct_option_is_B_l383_38345

-- Define the total number of balls
def total_black_balls : ℕ := 3
def total_red_balls : ℕ := 7
def total_balls : ℕ := total_black_balls + total_red_balls

-- Define the event of drawing balls
def drawing_balls (n : ℕ) : Prop := n = 3

-- Define what a random variable is within this context
def is_random_variable (n : ℕ) : Prop :=
  n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3

-- The main statement to prove
theorem correct_option_is_B (n : ℕ) :
  drawing_balls n → is_random_variable n :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l383_38345


namespace NUMINAMATH_GPT_men_left_hostel_l383_38384

variable (x : ℕ)
variable (h1 : 250 * 36 = (250 - x) * 45)

theorem men_left_hostel : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_men_left_hostel_l383_38384


namespace NUMINAMATH_GPT_find_unknown_number_l383_38342

theorem find_unknown_number (x : ℝ) (h : (28 + 48 / x) * x = 1980) : x = 69 :=
sorry

end NUMINAMATH_GPT_find_unknown_number_l383_38342


namespace NUMINAMATH_GPT_p_q_r_inequality_l383_38362

theorem p_q_r_inequality (p q r : ℝ) (h₁ : ∀ x, (x < -6 ∨ (3 ≤ x ∧ x ≤ 8)) ↔ (x - p) * (x - q) ≤ 0) (h₂ : p < q) : p + 2 * q + 3 * r = 1 :=
by
  sorry

end NUMINAMATH_GPT_p_q_r_inequality_l383_38362


namespace NUMINAMATH_GPT_average_marks_l383_38361

theorem average_marks (P C M : ℝ) (h1 : P = 95) (h2 : (P + M) / 2 = 90) (h3 : (P + C) / 2 = 70) :
  (P + C + M) / 3 = 75 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_l383_38361


namespace NUMINAMATH_GPT_hyperbola_eccentricity_correct_l383_38311

noncomputable def hyperbola_eccentricity : ℝ := 2

variables {a b : ℝ}
variables (ha_pos : 0 < a) (hb_pos : 0 < b)
variables (h_hyperbola : ∃ x y, x^2/a^2 - y^2/b^2 = 1)
variables (h_circle_chord_len : ∃ d, d = 2 ∧ ∃ x y, ((x - 2)^2 + y^2 = 4) ∧ (x * b/a = -y))

theorem hyperbola_eccentricity_correct :
  ∀ (a b : ℝ), 0 < a → 0 < b → (∃ x y, x^2 / a^2 - y^2 / b^2 = 1) 
  ∧ (∃ d, d = 2 ∧ ∃ x y, (x - 2)^2 + y^2 = 4 ∧ (x * b / a = -y)) →
  (eccentricity = 2) :=
by
  intro a b ha_pos hb_pos h_conditions
  have e := hyperbola_eccentricity
  sorry


end NUMINAMATH_GPT_hyperbola_eccentricity_correct_l383_38311


namespace NUMINAMATH_GPT_expression_value_l383_38321

theorem expression_value 
  (a b c : ℕ) 
  (ha : a = 12) 
  (hb : b = 2) 
  (hc : c = 7) :
  (a - (b - c)) - ((a - b) - c) = 14 := 
by 
  sorry

end NUMINAMATH_GPT_expression_value_l383_38321


namespace NUMINAMATH_GPT_alcohol_to_water_ratio_l383_38358

theorem alcohol_to_water_ratio (p q r : ℝ) :
  let alcohol := (p / (p + 1) + q / (q + 1) + r / (r + 1))
  let water := (1 / (p + 1) + 1 / (q + 1) + 1 / (r + 1))
  (alcohol / water) = (p * q * r + p * q + p * r + q * r + p + q + r) / (p * q + p * r + q * r + p + q + r + 1) :=
sorry

end NUMINAMATH_GPT_alcohol_to_water_ratio_l383_38358


namespace NUMINAMATH_GPT_product_of_repeating_decimal_l383_38315

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_l383_38315


namespace NUMINAMATH_GPT_reflected_rectangle_has_no_point_neg_3_4_l383_38385

structure Point where
  x : ℤ
  y : ℤ
  deriving DecidableEq, Repr

def reflect_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def is_not_vertex (pts: List Point) (p: Point) : Prop :=
  ¬ (p ∈ pts)

theorem reflected_rectangle_has_no_point_neg_3_4 :
  let initial_pts := [ Point.mk 1 3, Point.mk 1 1, Point.mk 4 1, Point.mk 4 3 ]
  let reflected_pts := initial_pts.map reflect_y
  is_not_vertex reflected_pts (Point.mk (-3) 4) :=
by
  sorry

end NUMINAMATH_GPT_reflected_rectangle_has_no_point_neg_3_4_l383_38385


namespace NUMINAMATH_GPT_largest_number_of_square_plots_l383_38338

/-- A rectangular field measures 30 meters by 60 meters with 2268 meters of internal fencing to partition into congruent, square plots. The entire field must be partitioned with sides of squares parallel to the edges. Prove the largest number of square plots is 722. -/
theorem largest_number_of_square_plots (s n : ℕ) (h_length : 60 = n * s) (h_width : 30 = s * 2 * n) (h_fence : 120 * n - 90 ≤ 2268) :
(s * 2 * n) = 722 :=
sorry

end NUMINAMATH_GPT_largest_number_of_square_plots_l383_38338


namespace NUMINAMATH_GPT_natasha_average_speed_climbing_l383_38346

theorem natasha_average_speed_climbing :
  ∀ D : ℝ,
    (total_time = 3 + 2) →
    (total_distance = 2 * D) →
    (average_speed = total_distance / total_time) →
    (average_speed = 3) →
    (D = 7.5) →
    (climb_speed = D / 3) →
    (climb_speed = 2.5) :=
by
  intros D total_time_eq total_distance_eq average_speed_eq average_speed_is_3 D_is_7_5 climb_speed_eq
  sorry

end NUMINAMATH_GPT_natasha_average_speed_climbing_l383_38346


namespace NUMINAMATH_GPT_sin_double_angle_given_sum_identity_l383_38392

theorem sin_double_angle_given_sum_identity {α : ℝ} 
  (h : Real.sin (Real.pi / 4 + α) = Real.sqrt 5 / 5) : 
  Real.sin (2 * α) = -3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_given_sum_identity_l383_38392


namespace NUMINAMATH_GPT_algebraic_expression_value_l383_38394

variable (x y : ℝ)

def condition1 : Prop := y - x = -1
def condition2 : Prop := x * y = 2

def expression : ℝ := -2 * x^3 * y + 4 * x^2 * y^2 - 2 * x * y^3

theorem algebraic_expression_value (h1 : condition1 x y) (h2 : condition2 x y) : expression x y = -4 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l383_38394


namespace NUMINAMATH_GPT_smaller_rectangle_ratio_l383_38383

theorem smaller_rectangle_ratio
  (length_large : ℝ) (width_large : ℝ) (area_small : ℝ)
  (h_length : length_large = 40)
  (h_width : width_large = 20)
  (h_area : area_small = 200) : 
  ∃ r : ℝ, (length_large * r) * (width_large * r) = area_small ∧ r = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_rectangle_ratio_l383_38383


namespace NUMINAMATH_GPT_quadratic_vertex_property_l383_38376

variable {a b c x0 y0 m n : ℝ}

-- Condition 1: (x0, y0) is a fixed point on the graph of the quadratic function y = ax^2 + bx + c
axiom fixed_point_on_graph : y0 = a * x0^2 + b * x0 + c

-- Condition 2: (m, n) is a moving point on the graph of the quadratic function
axiom moving_point_on_graph : n = a * m^2 + b * m + c

-- Condition 3: For any real number m, a(y0 - n) ≤ 0
axiom inequality_condition : ∀ m : ℝ, a * (y0 - (a * m^2 + b * m + c)) ≤ 0

-- Statement to prove
theorem quadratic_vertex_property : 2 * a * x0 + b = 0 := 
sorry

end NUMINAMATH_GPT_quadratic_vertex_property_l383_38376


namespace NUMINAMATH_GPT_depth_of_channel_l383_38350

theorem depth_of_channel (a b A : ℝ) (h : ℝ) (h_area : A = (1 / 2) * (a + b) * h)
  (ha : a = 12) (hb : b = 6) (hA : A = 630) : h = 70 :=
by
  sorry

end NUMINAMATH_GPT_depth_of_channel_l383_38350


namespace NUMINAMATH_GPT_derivative_at_1_derivative_at_neg_2_derivative_at_x0_l383_38344

noncomputable def f (x : ℝ) : ℝ := 2 / x + x

theorem derivative_at_1 : (deriv f 1) = -1 :=
sorry

theorem derivative_at_neg_2 : (deriv f (-2)) = 1 / 2 :=
sorry

theorem derivative_at_x0 (x0 : ℝ) : (deriv f x0) = -2 / (x0^2) + 1 :=
sorry

end NUMINAMATH_GPT_derivative_at_1_derivative_at_neg_2_derivative_at_x0_l383_38344


namespace NUMINAMATH_GPT_series_sum_l383_38388

theorem series_sum :
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  -- We define the sequence in list form for clarity
  (s.sum = 29) :=
by
  let a_1 := 2
  let d := 3
  let s := [2, -5, 8, -11, 14, -17, 20, -23, 26, -29, 32, -35, 38, -41, 44, -47, 50, -53, 56]
  sorry

end NUMINAMATH_GPT_series_sum_l383_38388


namespace NUMINAMATH_GPT_not_valid_mapping_circle_triangle_l383_38389

inductive Point
| mk : ℝ → ℝ → Point

inductive Circle
| mk : ℝ → ℝ → ℝ → Circle

inductive Triangle
| mk : Point → Point → Point → Triangle

open Point (mk)
open Circle (mk)
open Triangle (mk)

def valid_mapping (A B : Type) (f : A → B) := ∀ a₁ a₂ : A, f a₁ = f a₂ → a₁ = a₂

def inscribed_triangle_mapping (c : Circle) : Triangle := sorry -- map a circle to one of its inscribed triangles

theorem not_valid_mapping_circle_triangle :
  ¬ valid_mapping Circle Triangle inscribed_triangle_mapping :=
sorry

end NUMINAMATH_GPT_not_valid_mapping_circle_triangle_l383_38389


namespace NUMINAMATH_GPT_custom_op_diff_l383_38370

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_diff : custom_op 8 5 - custom_op 5 8 = -12 :=
by
  sorry

end NUMINAMATH_GPT_custom_op_diff_l383_38370


namespace NUMINAMATH_GPT_range_of_k_l383_38398

theorem range_of_k (k : ℝ) :
  (∃ (x : ℝ), 2 < x ∧ x < 3 ∧ x^2 + (1 - k) * x - 2 * (k + 1) = 0) →
  1 < k ∧ k < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_k_l383_38398


namespace NUMINAMATH_GPT_find_k_l383_38351

theorem find_k (k : ℝ) :
  let a := (3, 1)
  let b := (1, 3)
  let c := (k, 7)
  ((a.1 - c.1) * b.2 - (a.2 - c.2) * b.1 = 0) → k = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l383_38351


namespace NUMINAMATH_GPT_circular_garden_area_l383_38322

theorem circular_garden_area (AD DB DC R : ℝ) 
  (h1 : AD = 10) 
  (h2 : DB = 10) 
  (h3 : DC = 12) 
  (h4 : AD^2 + DC^2 = R^2) : 
  π * R^2 = 244 * π := 
  by 
    sorry

end NUMINAMATH_GPT_circular_garden_area_l383_38322


namespace NUMINAMATH_GPT_opposite_reciprocal_abs_value_l383_38382

theorem opposite_reciprocal_abs_value (a b c d m : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) (h3 : abs m = 3) : 
  (a + b) / m + c * d + m = 4 ∨ (a + b) / m + c * d + m = -2 := by 
  sorry

end NUMINAMATH_GPT_opposite_reciprocal_abs_value_l383_38382


namespace NUMINAMATH_GPT_new_year_season_markup_l383_38379

variable {C : ℝ} (hC : 0 < C)

theorem new_year_season_markup (h1 : ∀ C, C > 0 → ∃ P1, P1 = 1.20 * C)
                              (h2 : ∀ (P1 M : ℝ), M >= 0 → ∃ P2, P2 = P1 * (1 + M / 100))
                              (h3 : ∀ P2, ∃ P3, P3 = P2 * 0.91)
                              (h4 : ∃ P3, P3 = 1.365 * C) :
  ∃ M, M = 25 := 
by 
  sorry

end NUMINAMATH_GPT_new_year_season_markup_l383_38379


namespace NUMINAMATH_GPT_weighted_mean_is_correct_l383_38352

-- Define the given values
def dollar_from_aunt : ℝ := 9
def euros_from_uncle : ℝ := 9
def dollar_from_sister : ℝ := 7
def dollar_from_friends_1 : ℝ := 22
def dollar_from_friends_2 : ℝ := 23
def euros_from_friends_3 : ℝ := 18
def pounds_from_friends_4 : ℝ := 15
def dollar_from_friends_5 : ℝ := 22

-- Define the exchange rates
def exchange_rate_euro_to_usd : ℝ := 1.20
def exchange_rate_pound_to_usd : ℝ := 1.38

-- Calculate the amounts in USD
def dollar_from_uncle : ℝ := euros_from_uncle * exchange_rate_euro_to_usd
def dollar_from_friends_3_converted : ℝ := euros_from_friends_3 * exchange_rate_euro_to_usd
def dollar_from_friends_4_converted : ℝ := pounds_from_friends_4 * exchange_rate_pound_to_usd

-- Define total amounts from family and friends in USD
def family_total : ℝ := dollar_from_aunt + dollar_from_uncle + dollar_from_sister
def friends_total : ℝ := dollar_from_friends_1 + dollar_from_friends_2 + dollar_from_friends_3_converted + dollar_from_friends_4_converted + dollar_from_friends_5

-- Define weights
def family_weight : ℝ := 0.40
def friends_weight : ℝ := 0.60

-- Calculate the weighted mean
def weighted_mean : ℝ := (family_total * family_weight) + (friends_total * friends_weight)

theorem weighted_mean_is_correct : weighted_mean = 76.30 := by
  sorry

end NUMINAMATH_GPT_weighted_mean_is_correct_l383_38352


namespace NUMINAMATH_GPT_find_divisor_l383_38396

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_dividend : dividend = 190) (h_quotient : quotient = 9) (h_remainder : remainder = 1) :
  ∃ divisor : ℕ, dividend = divisor * quotient + remainder ∧ divisor = 21 := 
by
  sorry

end NUMINAMATH_GPT_find_divisor_l383_38396


namespace NUMINAMATH_GPT_factor_sum_l383_38395

theorem factor_sum :
  ∃ d e f : ℤ, (∀ x : ℤ, x^2 + 11 * x + 24 = (x + d) * (x + e)) ∧
              (∀ x : ℤ, x^2 + 9 * x - 36 = (x + e) * (x - f)) ∧
              d + e + f = 14 := by
  sorry

end NUMINAMATH_GPT_factor_sum_l383_38395


namespace NUMINAMATH_GPT_Pythagorean_triple_example_1_Pythagorean_triple_example_2_l383_38397

theorem Pythagorean_triple_example_1 : 3^2 + 4^2 = 5^2 := by
  sorry

theorem Pythagorean_triple_example_2 : 5^2 + 12^2 = 13^2 := by
  sorry

end NUMINAMATH_GPT_Pythagorean_triple_example_1_Pythagorean_triple_example_2_l383_38397


namespace NUMINAMATH_GPT_length_of_pond_l383_38368

-- Define the problem conditions
variables (W L S : ℝ)
variables (h1 : L = 2 * W) (h2 : L = 24) 
variables (A_field A_pond : ℝ)
variables (h3 : A_pond = 1 / 8 * A_field)

-- State the theorem
theorem length_of_pond :
  A_field = L * W ∧ A_pond = S^2 ∧ A_pond = 1 / 8 * A_field ∧ L = 24 ∧ L = 2 * W → 
  S = 6 :=
by
  sorry

end NUMINAMATH_GPT_length_of_pond_l383_38368


namespace NUMINAMATH_GPT_symmetric_circle_equation_l383_38360

theorem symmetric_circle_equation (x y : ℝ) :
  (x - 1)^2 + (y + 2)^2 = 5 → (x + 1)^2 + (y - 2)^2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l383_38360


namespace NUMINAMATH_GPT_golf_tees_per_member_l383_38363

theorem golf_tees_per_member (T : ℕ) : 
  (∃ (t : ℕ), 
     t = 4 * T ∧ 
     (∀ (g : ℕ), g ≤ 2 → g * 12 + 28 * 2 = t)
  ) → T = 20 :=
by
  intros h
  -- problem statement is enough for this example
  sorry

end NUMINAMATH_GPT_golf_tees_per_member_l383_38363


namespace NUMINAMATH_GPT_Cheryl_total_distance_l383_38375

theorem Cheryl_total_distance :
  let speed := 2
  let duration := 3
  let distance_away := speed * duration
  let distance_home := distance_away
  let total_distance := distance_away + distance_home
  total_distance = 12 := by
  sorry

end NUMINAMATH_GPT_Cheryl_total_distance_l383_38375


namespace NUMINAMATH_GPT_standard_deviation_of_distribution_l383_38336

theorem standard_deviation_of_distribution (μ σ : ℝ) 
    (h₁ : μ = 15) (h₂ : μ - 2 * σ = 12) : σ = 1.5 := by
  sorry

end NUMINAMATH_GPT_standard_deviation_of_distribution_l383_38336


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l383_38356

theorem remainder_when_divided_by_7 (n : ℕ) (h1 : n % 2 = 1) (h2 : ∃ p : ℕ, p > 0 ∧ (n + p) % 10 = 0 ∧ p = 5) : n % 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l383_38356


namespace NUMINAMATH_GPT_tangent_product_eq_three_l383_38335

noncomputable def tangent (θ : ℝ) : ℝ := Real.sin θ / Real.cos θ

theorem tangent_product_eq_three : 
  let θ1 := π / 9
  let θ2 := 2 * π / 9
  let θ3 := 4 * π / 9
  tangent θ1 * tangent θ2 * tangent θ3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_tangent_product_eq_three_l383_38335


namespace NUMINAMATH_GPT_soccer_teams_participation_l383_38306

theorem soccer_teams_participation (total_games : ℕ) (teams_play : ℕ → ℕ) (x : ℕ) :
  (total_games = 20) → (teams_play x = x * (x - 1)) → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_soccer_teams_participation_l383_38306


namespace NUMINAMATH_GPT_total_number_of_students_l383_38367

theorem total_number_of_students (girls boys : ℕ) 
  (h_ratio : 8 * girls = 5 * boys) 
  (h_girls : girls = 160) : 
  girls + boys = 416 := 
sorry

end NUMINAMATH_GPT_total_number_of_students_l383_38367


namespace NUMINAMATH_GPT_simplify_sqrt_sum_l383_38302

theorem simplify_sqrt_sum : (Real.sqrt 72 + Real.sqrt 32) = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_simplify_sqrt_sum_l383_38302


namespace NUMINAMATH_GPT_probability_no_two_boys_same_cinema_l383_38313

-- Definitions
def total_cinemas := 10
def total_boys := 7

def total_arrangements : ℕ := total_cinemas ^ total_boys
def favorable_arrangements : ℕ := 10 * 9 * 8 * 7 * 6 * 5 * 4
def probability := (favorable_arrangements : ℚ) / total_arrangements

-- Mathematical proof problem
theorem probability_no_two_boys_same_cinema : 
  probability = 0.06048 := 
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_probability_no_two_boys_same_cinema_l383_38313


namespace NUMINAMATH_GPT_range_of_a_l383_38328

noncomputable def f (x a : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1
def intersects_at_single_point (f : ℝ → ℝ → ℝ) (a : ℝ) : Prop :=
∃! x, f x a = 3

theorem range_of_a (a : ℝ) :
  intersects_at_single_point f a ↔ -1 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l383_38328


namespace NUMINAMATH_GPT_flagpole_height_l383_38393

/-
A flagpole is of certain height. It breaks, folding over in half, such that what was the tip of the flagpole is now dangling two feet above the ground. 
The flagpole broke 7 feet from the base. Prove that the height of the flagpole is 16 feet.
-/

theorem flagpole_height (H : ℝ) (h1 : H > 0) (h2 : H - 7 > 0) (h3 : H - 9 = 7) : H = 16 :=
by
  /- the proof is omitted -/
  sorry

end NUMINAMATH_GPT_flagpole_height_l383_38393


namespace NUMINAMATH_GPT_exists_digit_a_l383_38309

theorem exists_digit_a : 
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (1111 * a - 1 = (a - 1) ^ (a - 2)) :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_digit_a_l383_38309


namespace NUMINAMATH_GPT_kendalls_total_distance_l383_38314

-- Definitions of the conditions
def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5

-- The theorem to prove the total distance
theorem kendalls_total_distance : distance_with_mother + distance_with_father = 0.67 :=
by
  sorry

end NUMINAMATH_GPT_kendalls_total_distance_l383_38314


namespace NUMINAMATH_GPT_set_operations_l383_38312

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | 0 < x ∧ x < 5 }
def U : Set ℝ := Set.univ  -- Universal set ℝ
def complement (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem set_operations :
  (A ∩ B = { x | 0 < x ∧ x < 2 }) ∧ 
  (complement A ∪ B = { x | 0 < x }) :=
by {
  sorry
}

end NUMINAMATH_GPT_set_operations_l383_38312


namespace NUMINAMATH_GPT_other_position_in_arithmetic_progression_l383_38372

theorem other_position_in_arithmetic_progression 
  (a d : ℝ) (x : ℕ)
  (h1 : a + (4 - 1) * d + a + (x - 1) * d = 20)
  (h2 : 5 * (2 * a + 9 * d) = 100) :
  x = 7 := by
  sorry

end NUMINAMATH_GPT_other_position_in_arithmetic_progression_l383_38372


namespace NUMINAMATH_GPT_hyperbola_distance_to_left_focus_l383_38373

theorem hyperbola_distance_to_left_focus (P : ℝ × ℝ)
  (h1 : (P.1^2) / 9 - (P.2^2) / 16 = 1)
  (dPF2 : dist P (4, 0) = 4) : dist P (-4, 0) = 10 := 
sorry

end NUMINAMATH_GPT_hyperbola_distance_to_left_focus_l383_38373


namespace NUMINAMATH_GPT_shoes_multiple_l383_38381

-- Define the number of shoes each has
variables (J E B : ℕ)

-- Conditions
axiom h1 : B = 22
axiom h2 : J = E / 2
axiom h3 : J + E + B = 121

-- Prove the multiple of E to B is 3
theorem shoes_multiple : E / B = 3 :=
by
  -- Inject the provisional proof
  sorry

end NUMINAMATH_GPT_shoes_multiple_l383_38381


namespace NUMINAMATH_GPT_even_sum_probability_correct_l383_38310

-- Definition: Calculate probabilities based on the given wheels
def even_probability_wheel_one : ℚ := 1/3
def odd_probability_wheel_one : ℚ := 2/3
def even_probability_wheel_two : ℚ := 1/4
def odd_probability_wheel_two : ℚ := 3/4

-- Probability of both numbers being even
def both_even_probability : ℚ := even_probability_wheel_one * even_probability_wheel_two

-- Probability of both numbers being odd
def both_odd_probability : ℚ := odd_probability_wheel_one * odd_probability_wheel_two

-- Final probability of the sum being even
def even_sum_probability : ℚ := both_even_probability + both_odd_probability

theorem even_sum_probability_correct : even_sum_probability = 7/12 := 
sorry

end NUMINAMATH_GPT_even_sum_probability_correct_l383_38310


namespace NUMINAMATH_GPT_factorize_expression_l383_38365

theorem factorize_expression (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by sorry

end NUMINAMATH_GPT_factorize_expression_l383_38365


namespace NUMINAMATH_GPT_expression_result_l383_38333

-- We denote k as a natural number representing the number of digits in A, B, C, and D.
variable (k : ℕ)

-- Definitions of the numbers A, B, C, D, and E based on the problem statement.
def A : ℕ := 3 * ((10 ^ (k - 1) - 1) / 9)
def B : ℕ := 4 * ((10 ^ (k - 1) - 1) / 9)
def C : ℕ := 6 * ((10 ^ (k - 1) - 1) / 9)
def D : ℕ := 7 * ((10 ^ (k - 1) - 1) / 9)
def E : ℕ := 5 * ((10 ^ (2 * k) - 1) / 9)

-- The statement we want to prove.
theorem expression_result :
  E - A * D - B * C + 1 = (10 ^ (k + 1) - 1) / 9 :=
by
  sorry

end NUMINAMATH_GPT_expression_result_l383_38333


namespace NUMINAMATH_GPT_sabi_share_removed_l383_38307

theorem sabi_share_removed :
  ∀ (N S M x : ℝ), N - 5 = 2 * (S - x) / 8 ∧ S - x = 4 * (6 * (M - 4)) / 16 ∧ M = 102 ∧ N + S + M = 1100 
  → x = 829.67 := by
  sorry

end NUMINAMATH_GPT_sabi_share_removed_l383_38307


namespace NUMINAMATH_GPT_region_area_l383_38320

theorem region_area (x y : ℝ) : 
  (x^2 + y^2 + 14 * x + 18 * y = 0) → 
  (π * 130) = 130 * π :=
by 
  sorry

end NUMINAMATH_GPT_region_area_l383_38320


namespace NUMINAMATH_GPT_park_shape_l383_38341

def cost_of_fencing (side_count : ℕ) (side_cost : ℕ) := side_count * side_cost

theorem park_shape (total_cost : ℕ) (side_cost : ℕ) (h_total : total_cost = 224) (h_side : side_cost = 56) : 
  (∃ sides : ℕ, sides = total_cost / side_cost ∧ sides = 4) ∧ (∀ (sides : ℕ),  cost_of_fencing sides side_cost = total_cost → sides = 4 → sides = 4 ∧ (∀ (x y z w : ℕ), x = y → y = z → z = w → w = x)) :=
by
  sorry

end NUMINAMATH_GPT_park_shape_l383_38341


namespace NUMINAMATH_GPT_arithmetic_sequence_terms_l383_38349

theorem arithmetic_sequence_terms (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 3 + a 4 = 10) 
  (h2 : a (n - 3) + a (n - 2) = 30) 
  (h3 : (n * (a 1 + a n)) / 2 = 100) : 
  n = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_terms_l383_38349


namespace NUMINAMATH_GPT_problem_l383_38347

variable (f g h : ℕ → ℕ)

-- Define the conditions as hypotheses
axiom h1 : ∀ (n m : ℕ), n ≠ m → h n ≠ h m
axiom h2 : ∀ y, ∃ x, g x = y
axiom h3 : ∀ n, f n = g n - h n + 1

theorem problem : ∀ n, f n = 1 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l383_38347


namespace NUMINAMATH_GPT_rise_in_water_level_l383_38357

-- Define the conditions related to the cube and the vessel
def edge_length := 15 -- in cm
def base_length := 20 -- in cm
def base_width := 15 -- in cm

-- Calculate volumes and areas
def V_cube := edge_length ^ 3
def A_base := base_length * base_width

-- Declare the mathematical proof problem statement
theorem rise_in_water_level : 
  (V_cube / A_base : ℝ) = 11.25 :=
by
  -- edge_length, V_cube, A_base are all already defined
  -- This particularly proves (15^3) / (20 * 15) = 11.25
  sorry

end NUMINAMATH_GPT_rise_in_water_level_l383_38357


namespace NUMINAMATH_GPT_subtraction_of_decimals_l383_38316

theorem subtraction_of_decimals : (3.75 - 0.48) = 3.27 :=
by
  sorry

end NUMINAMATH_GPT_subtraction_of_decimals_l383_38316


namespace NUMINAMATH_GPT_solve_equation_l383_38353

def equation_params (a x : ℝ) : Prop :=
  a * (1 / (Real.cos x) - Real.tan x) = 1

def valid_solutions (a x : ℝ) (k : ℤ) : Prop :=
  (a ≠ 0) ∧ (Real.cos x ≠ 0) ∧ (
    (|a| ≥ 1 ∧ x = Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k) ∨
    ((-1 < a ∧ a < 0) ∨ (0 < a ∧ a < 1) ∧ x = - Real.arccos (a / Real.sqrt (a * a + 1)) + 2 * Real.pi * k)
  )

theorem solve_equation (a x : ℝ) (k : ℤ) :
  equation_params a x → valid_solutions a x k := by
  sorry

end NUMINAMATH_GPT_solve_equation_l383_38353


namespace NUMINAMATH_GPT_circle_intersection_l383_38319

theorem circle_intersection (m : ℝ) :
  (x^2 + y^2 - 2*m*x + m^2 - 4 = 0 ∧ x^2 + y^2 + 2*x - 4*m*y + 4*m^2 - 8 = 0) →
  (-12/5 < m ∧ m < -2/5) ∨ (0 < m ∧ m < 2) :=
by sorry

end NUMINAMATH_GPT_circle_intersection_l383_38319


namespace NUMINAMATH_GPT_largest_difference_l383_38355

noncomputable def A : ℕ := 3 * 2010 ^ 2011
noncomputable def B : ℕ := 2010 ^ 2011
noncomputable def C : ℕ := 2009 * 2010 ^ 2010
noncomputable def D : ℕ := 3 * 2010 ^ 2010
noncomputable def E : ℕ := 2010 ^ 2010
noncomputable def F : ℕ := 2010 ^ 2009

theorem largest_difference :
  (A - B = 2 * 2010 ^ 2011) ∧ 
  (B - C = 2010 ^ 2010) ∧ 
  (C - D = 2006 * 2010 ^ 2010) ∧ 
  (D - E = 2 * 2010 ^ 2010) ∧ 
  (E - F = 2009 * 2010 ^ 2009) ∧ 
  (2 * 2010 ^ 2011 > 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2006 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2 * 2010 ^ 2010) ∧
  (2 * 2010 ^ 2011 > 2009 * 2010 ^ 2009) :=
by
  sorry

end NUMINAMATH_GPT_largest_difference_l383_38355


namespace NUMINAMATH_GPT_greatest_m_div_36_and_7_l383_38331

def reverse_digits (m : ℕ) : ℕ :=
  let d1 := (m / 1000) % 10
  let d2 := (m / 100) % 10
  let d3 := (m / 10) % 10
  let d4 := m % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1

theorem greatest_m_div_36_and_7
  (m : ℕ) (n : ℕ := reverse_digits m)
  (h1 : 1000 ≤ m ∧ m < 10000)
  (h2 : 1000 ≤ n ∧ n < 10000)
  (h3 : 36 ∣ m ∧ 36 ∣ n)
  (h4 : 7 ∣ m) :
  m = 9828 := 
sorry

end NUMINAMATH_GPT_greatest_m_div_36_and_7_l383_38331


namespace NUMINAMATH_GPT_probability_htth_l383_38326

def probability_of_sequence_HTTH := (1 / 2) * (1 / 2) * (1 / 2) * (1 / 2)

theorem probability_htth : probability_of_sequence_HTTH = 1 / 16 := by
  sorry

end NUMINAMATH_GPT_probability_htth_l383_38326


namespace NUMINAMATH_GPT_temperature_decrease_l383_38371

theorem temperature_decrease (T : ℝ) 
    (h1 : T * (3 / 4) = T - 21)
    (h2 : T > 0) : 
    T = 84 := 
  sorry

end NUMINAMATH_GPT_temperature_decrease_l383_38371


namespace NUMINAMATH_GPT_paint_fence_together_time_l383_38390

-- Define the times taken by Jamshid and Taimour
def Taimour_time := 18 -- Taimour takes 18 hours to paint the fence
def Jamshid_time := Taimour_time / 2 -- Jamshid takes half the time Taimour takes

-- Define the work rates
def Taimour_rate := 1 / Taimour_time
def Jamshid_rate := 1 / Jamshid_time

-- Define the combined work rate
def combined_rate := Taimour_rate + Jamshid_rate

-- Define the total time taken when working together
def together_time := 1 / combined_rate

-- State the main theorem
theorem paint_fence_together_time : together_time = 6 := 
sorry

end NUMINAMATH_GPT_paint_fence_together_time_l383_38390


namespace NUMINAMATH_GPT_parallelogram_perimeter_eq_60_l383_38364

-- Given conditions from the problem
variables (P Q R M N O : Type*)
variables (PQ PR QR PM MN NO PO : ℝ)
variables {PQ_eq_PR : PQ = PR}
variables {PQ_val : PQ = 30}
variables {PR_val : PR = 30}
variables {QR_val : QR = 28}
variables {MN_parallel_PR : true}  -- Parallel condition we can treat as true for simplification
variables {NO_parallel_PQ : true}  -- Another parallel condition treated as true

-- Statement of the problem to be proved
theorem parallelogram_perimeter_eq_60 :
  PM + MN + NO + PO = 60 :=
sorry

end NUMINAMATH_GPT_parallelogram_perimeter_eq_60_l383_38364


namespace NUMINAMATH_GPT_percent_of_div_l383_38391

theorem percent_of_div (P: ℝ) (Q: ℝ) (R: ℝ) : ( ( P / 100 ) * Q ) / R = 354.2 :=
by
  -- Given P = 168, Q = 1265, R = 6
  let P := 168
  let Q := 1265
  let R := 6
  -- sorry to skip the actual proof.
  sorry

end NUMINAMATH_GPT_percent_of_div_l383_38391


namespace NUMINAMATH_GPT_interest_difference_correct_l383_38327

noncomputable def principal : ℝ := 1000
noncomputable def rate : ℝ := 0.10
noncomputable def time : ℝ := 4

noncomputable def simple_interest (P r t : ℝ) : ℝ := P * r * t
noncomputable def compound_interest (P r t : ℝ) : ℝ := P * (1 + r)^t - P

noncomputable def interest_difference (P r t : ℝ) : ℝ := 
  compound_interest P r t - simple_interest P r t

theorem interest_difference_correct :
  interest_difference principal rate time = 64.10 :=
by
  sorry

end NUMINAMATH_GPT_interest_difference_correct_l383_38327


namespace NUMINAMATH_GPT_sum_of_square_roots_l383_38330

theorem sum_of_square_roots : 
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) + 
  Real.sqrt (1 + 3 + 5 + 7 + 9 + 11)) = 21 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_sum_of_square_roots_l383_38330


namespace NUMINAMATH_GPT_girls_bought_balloons_l383_38305

theorem girls_bought_balloons (initial_balloons boys_bought girls_bought remaining_balloons : ℕ)
  (h1 : initial_balloons = 36)
  (h2 : boys_bought = 3)
  (h3 : remaining_balloons = 21)
  (h4 : initial_balloons - remaining_balloons = boys_bought + girls_bought) :
  girls_bought = 12 := by
  sorry

end NUMINAMATH_GPT_girls_bought_balloons_l383_38305


namespace NUMINAMATH_GPT_solve_sys_eqns_l383_38300

def sys_eqns_solution (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

theorem solve_sys_eqns :
  ∃ (x y : ℝ),
  (sys_eqns_solution x y ∧
  ((x = 0 ∧ y = 0) ∨
  (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
  (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2))) :=
by
  sorry

end NUMINAMATH_GPT_solve_sys_eqns_l383_38300


namespace NUMINAMATH_GPT_average_weight_increase_l383_38329

theorem average_weight_increase
  (A : ℝ) -- Average weight of the two persons
  (w1 : ℝ) (h1 : w1 = 65) -- One person's weight is 65 kg 
  (w2 : ℝ) (h2 : w2 = 74) -- The new person's weight is 74 kg
  :
  ((A * 2 - w1 + w2) / 2 - A = 4.5) :=
by
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_average_weight_increase_l383_38329


namespace NUMINAMATH_GPT_cubic_eq_one_complex_solution_l383_38340

theorem cubic_eq_one_complex_solution (k : ℂ) :
  (∃ (x : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0) ∧
  (∀ (x y z : ℂ), 8 * x^3 + 12 * x^2 + k * x + 1 = 0 → 8 * y^3 + 12 * y^2 + k * y + 1 = 0
    → 8 * z^3 + 12 * z^2 + k * z + 1 = 0 → x = y ∧ y = z) →
  k = 6 :=
sorry

end NUMINAMATH_GPT_cubic_eq_one_complex_solution_l383_38340


namespace NUMINAMATH_GPT_probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l383_38324

def probability_of_at_least_one_head (p : ℚ) (n : ℕ) : ℚ := 
  1 - (1 - p)^n

theorem probability_of_at_least_one_head_in_three_tosses_is_7_over_8 :
  probability_of_at_least_one_head (1/2) 3 = 7/8 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_at_least_one_head_in_three_tosses_is_7_over_8_l383_38324


namespace NUMINAMATH_GPT_slope_of_line_l383_38387

theorem slope_of_line 
  (p l t : ℝ) (p_pos : p > 0)
  (h_parabola : (2:ℝ)*p = 4) -- Since the parabola passes through M(l,2)
  (h_incircle_center : ∃ (k m : ℝ), (k + 1 = 0) ∧ (k^2 - k - 2 = 0)) :
  ∃ (k : ℝ), k = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_slope_of_line_l383_38387


namespace NUMINAMATH_GPT_factor_expression_l383_38343

theorem factor_expression (x : ℝ) : 46 * x^3 - 115 * x^7 = -23 * x^3 * (5 * x^4 - 2) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l383_38343


namespace NUMINAMATH_GPT_ratio_of_length_to_width_l383_38318

-- Definitions of conditions
def width := 5
def area := 75

-- Theorem statement proving the ratio is 3
theorem ratio_of_length_to_width {l : ℕ} (h1 : l * width = area) : l / width = 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_length_to_width_l383_38318


namespace NUMINAMATH_GPT_probability_four_coins_l383_38301

-- Define four fair coin flips, having 2 possible outcomes for each coin
def four_coin_flips_outcomes : ℕ := 2 ^ 4

-- Define the favorable outcomes: all heads or all tails
def favorable_outcomes : ℕ := 2

-- The probability of getting all heads or all tails
def probability_all_heads_or_tails : ℚ := favorable_outcomes / four_coin_flips_outcomes

-- The theorem stating the answer to the problem
theorem probability_four_coins:
  probability_all_heads_or_tails = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_probability_four_coins_l383_38301
