import Mathlib

namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l860_86063

noncomputable def find_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) : ℝ :=
Real.sqrt (1 + (b / a)^2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b / a = Real.sqrt 5 / 2) :
  find_eccentricity a b h1 h2 h3 = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l860_86063


namespace NUMINAMATH_GPT_hexagon_unique_intersection_points_are_45_l860_86042

-- Definitions related to hexagon for the proof problem
def hexagon_vertices : ℕ := 6
def sides_of_hexagon : ℕ := 6
def diagonals_of_hexagon : ℕ := 9
def total_line_segments : ℕ := 15
def total_intersections : ℕ := 105
def vertex_intersections_per_vertex : ℕ := 10
def total_vertex_intersections : ℕ := 60

-- Final Proof Statement that needs to be proved
theorem hexagon_unique_intersection_points_are_45 :
  total_intersections - total_vertex_intersections = 45 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_unique_intersection_points_are_45_l860_86042


namespace NUMINAMATH_GPT_sector_area_is_nine_l860_86003

-- Given the conditions: the perimeter of the sector is 12 cm and the central angle is 2 radians
def sector_perimeter_radius (r : ℝ) :=
  4 * r = 12

def sector_angle : ℝ := 2

-- Prove that the area of the sector is 9 cm²
theorem sector_area_is_nine (r : ℝ) (s : ℝ) (h : sector_perimeter_radius r) (h_angle : sector_angle = 2) :
  s = 9 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_is_nine_l860_86003


namespace NUMINAMATH_GPT_lcm_of_18_and_20_l860_86035

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_of_18_and_20_l860_86035


namespace NUMINAMATH_GPT_number_of_tangent_lines_through_origin_l860_86067

def f (x : ℝ) : ℝ := -x^3 + 6*x^2 - 9*x + 8

def f_prime (x : ℝ) : ℝ := -3*x^2 + 12*x - 9

def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ := f x₀ + f_prime x₀ * (x - x₀)

theorem number_of_tangent_lines_through_origin : 
  ∃! (x₀ : ℝ), x₀^3 - 3*x₀^2 + 4 = 0 := 
sorry

end NUMINAMATH_GPT_number_of_tangent_lines_through_origin_l860_86067


namespace NUMINAMATH_GPT_number_in_parentheses_l860_86079

theorem number_in_parentheses (x : ℤ) (h : x - (-2) = 3) : x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_in_parentheses_l860_86079


namespace NUMINAMATH_GPT_problem1_problem2_l860_86093

-- Problem 1: Prove f(x) ≥ 3 implies x ≤ -1 or x ≥ 1 given f(x) = |x + 1| + |2x - 1| and m = 1
theorem problem1 (x : ℝ) : (|x + 1| + |2 * x - 1| >= 3) ↔ (x <= -1 ∨ x >= 1) :=
by
 sorry

-- Problem 2: Prove ½ f(x) ≤ |x + 1| holds for x ∈ [m, 2m²] implies ½ < m ≤ 1 given f(x) = |x + m| + |2x - 1| and m > 0
theorem problem2 (m : ℝ) (x : ℝ) (h_m : 0 < m) (h_x : m ≤ x ∧ x ≤ 2 * m^2) : (1/2 * (|x + m| + |2 * x - 1|) ≤ |x + 1|) ↔ (1/2 < m ∧ m ≤ 1) :=
by
 sorry

end NUMINAMATH_GPT_problem1_problem2_l860_86093


namespace NUMINAMATH_GPT_calculate_plot_size_in_acres_l860_86009

theorem calculate_plot_size_in_acres :
  let bottom_edge_cm : ℝ := 15
  let top_edge_cm : ℝ := 10
  let height_cm : ℝ := 10
  let cm_to_miles : ℝ := 3
  let miles_to_acres : ℝ := 640
  let trapezoid_area_cm2 := (bottom_edge_cm + top_edge_cm) * height_cm / 2
  let trapezoid_area_miles2 := trapezoid_area_cm2 * (cm_to_miles ^ 2)
  (trapezoid_area_miles2 * miles_to_acres) = 720000 :=
by
  sorry

end NUMINAMATH_GPT_calculate_plot_size_in_acres_l860_86009


namespace NUMINAMATH_GPT_geometric_progression_solution_l860_86055

theorem geometric_progression_solution (p : ℝ) :
  (3 * p + 1)^2 = (9 * p + 10) * |p - 3| ↔ p = -1 ∨ p = 29 / 18 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l860_86055


namespace NUMINAMATH_GPT_expression_value_l860_86047

theorem expression_value (a b : ℝ) (h₁ : a - 2 * b = 0) (h₂ : b ≠ 0) : 
  ( (b / (a - b) + 1) * (a^2 - b^2) / a^2 ) = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_expression_value_l860_86047


namespace NUMINAMATH_GPT_jameson_total_medals_l860_86097

-- Define the number of track, swimming, and badminton medals
def track_medals := 5
def swimming_medals := 2 * track_medals
def badminton_medals := 5

-- Define the total number of medals
def total_medals := track_medals + swimming_medals + badminton_medals

-- Theorem statement
theorem jameson_total_medals : total_medals = 20 := 
by
  sorry

end NUMINAMATH_GPT_jameson_total_medals_l860_86097


namespace NUMINAMATH_GPT_minimum_ceiling_height_l860_86006

def is_multiple_of_0_1 (h : ℝ) : Prop := ∃ (k : ℤ), h = k / 10

def football_field_illuminated (h : ℝ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 80 →
  (x^2 + y^2 ≤ h^2) ∨ ((x - 100)^2 + y^2 ≤ h^2) ∨
  (x^2 + (y - 80)^2 ≤ h^2) ∨ ((x - 100)^2 + (y - 80)^2 ≤ h^2)

theorem minimum_ceiling_height :
  ∃ (h : ℝ), football_field_illuminated h ∧ is_multiple_of_0_1 h ∧ h = 32.1 :=
sorry

end NUMINAMATH_GPT_minimum_ceiling_height_l860_86006


namespace NUMINAMATH_GPT_percentage_difference_is_50_percent_l860_86071

-- Definitions of hourly wages
def Mike_hourly_wage : ℕ := 14
def Phil_hourly_wage : ℕ := 7

-- Calculating the percentage difference
theorem percentage_difference_is_50_percent :
  (Mike_hourly_wage - Phil_hourly_wage) * 100 / Mike_hourly_wage = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_is_50_percent_l860_86071


namespace NUMINAMATH_GPT_find_g_30_l860_86074

def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) = x * g y

axiom g_one : g 1 = 10

theorem find_g_30 : g 30 = 300 := by
  sorry

end NUMINAMATH_GPT_find_g_30_l860_86074


namespace NUMINAMATH_GPT_solve_expression_l860_86050

def evaluation_inside_parentheses : ℕ := 3 - 3

def power_of_zero : ℝ := (5 : ℝ) ^ evaluation_inside_parentheses

theorem solve_expression :
  (3 : ℝ) - power_of_zero = 2 := by
  -- Utilize the conditions defined above
  sorry

end NUMINAMATH_GPT_solve_expression_l860_86050


namespace NUMINAMATH_GPT_attendance_difference_l860_86012

theorem attendance_difference :
  let a := 65899
  let b := 66018
  b - a = 119 :=
sorry

end NUMINAMATH_GPT_attendance_difference_l860_86012


namespace NUMINAMATH_GPT_no_real_solutions_of_quadratic_eq_l860_86029

theorem no_real_solutions_of_quadratic_eq
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  ∀ x : ℝ, ¬ (b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_of_quadratic_eq_l860_86029


namespace NUMINAMATH_GPT_pairs_satisfy_ineq_l860_86004

theorem pairs_satisfy_ineq (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y ≤ 0 ↔
  ∃ n m : ℤ, x = n * Real.pi ∧ y = m * Real.pi := 
sorry

end NUMINAMATH_GPT_pairs_satisfy_ineq_l860_86004


namespace NUMINAMATH_GPT_selling_price_with_discount_l860_86000

variable (a : ℝ)

theorem selling_price_with_discount (h : a ≥ 0) : (a * 1.2 * 0.91) = (a * 1.2 * 0.91) :=
by
  sorry

end NUMINAMATH_GPT_selling_price_with_discount_l860_86000


namespace NUMINAMATH_GPT_secret_spread_reaches_3280_on_saturday_l860_86069

theorem secret_spread_reaches_3280_on_saturday :
  (∃ n : ℕ, 4 * ( 3^n - 1) / 2 + 1 = 3280 ) ∧ n = 7  :=
sorry

end NUMINAMATH_GPT_secret_spread_reaches_3280_on_saturday_l860_86069


namespace NUMINAMATH_GPT_find_value_l860_86073

theorem find_value (a b : ℝ) (h1 : 2 * a - 3 * b = 1) : 5 - 4 * a + 6 * b = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_value_l860_86073


namespace NUMINAMATH_GPT_inequality_nonneg_real_l860_86089

theorem inequality_nonneg_real (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ (2 / (1 + a * b)) ∧ ((1 / (1 + a^2)) + (1 / (1 + b^2)) = (2 / (1 + a * b)) ↔ a = b) :=
sorry

end NUMINAMATH_GPT_inequality_nonneg_real_l860_86089


namespace NUMINAMATH_GPT_first_grade_frequency_is_correct_second_grade_frequency_is_correct_l860_86028

def total_items : ℕ := 400
def second_grade_items : ℕ := 20
def first_grade_items : ℕ := total_items - second_grade_items

def frequency_first_grade : ℚ := first_grade_items / total_items
def frequency_second_grade : ℚ := second_grade_items / total_items

theorem first_grade_frequency_is_correct : frequency_first_grade = 0.95 := 
 by
 sorry

theorem second_grade_frequency_is_correct : frequency_second_grade = 0.05 := 
 by 
 sorry

end NUMINAMATH_GPT_first_grade_frequency_is_correct_second_grade_frequency_is_correct_l860_86028


namespace NUMINAMATH_GPT_remainder_3_pow_20_mod_7_l860_86040

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end NUMINAMATH_GPT_remainder_3_pow_20_mod_7_l860_86040


namespace NUMINAMATH_GPT_nonnegative_expr_interval_l860_86014

noncomputable def expr (x : ℝ) : ℝ := (2 * x - 15 * x ^ 2 + 56 * x ^ 3) / (9 - x ^ 3)

theorem nonnegative_expr_interval (x : ℝ) :
  expr x ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_nonnegative_expr_interval_l860_86014


namespace NUMINAMATH_GPT_increased_hypotenuse_length_l860_86033

theorem increased_hypotenuse_length :
  let AB := 24
  let BC := 10
  let AB' := AB + 6
  let BC' := BC + 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by
  sorry

end NUMINAMATH_GPT_increased_hypotenuse_length_l860_86033


namespace NUMINAMATH_GPT_find_positive_integer_x_l860_86001

theorem find_positive_integer_x :
  ∃ x : ℕ, x > 0 ∧ (5 * x + 1) / (x - 1) > 2 * x + 2 ∧
  ∀ y : ℕ, y > 0 ∧ (5 * y + 1) / (y - 1) > 2 * x + 2 → y = 2 :=
sorry

end NUMINAMATH_GPT_find_positive_integer_x_l860_86001


namespace NUMINAMATH_GPT_find_sum_l860_86032

theorem find_sum (A B : ℕ) (h1 : B = 278 + 365 * 3) (h2 : A = 20 * 100 + 87 * 10) : A + B = 4243 := by
    sorry

end NUMINAMATH_GPT_find_sum_l860_86032


namespace NUMINAMATH_GPT_part1_part2_l860_86056

-- Define the cost price, current selling price, sales per week, and change in sales per reduction in price.
def cost_price : ℝ := 50
def current_price : ℝ := 80
def current_sales : ℝ := 200
def sales_increase_per_yuan : ℝ := 20

-- Define the weekly profit calculation.
def weekly_profit (price : ℝ) : ℝ :=
(price - cost_price) * (current_sales + sales_increase_per_yuan * (current_price - price))

-- Part 1: Selling price for a weekly profit of 7500 yuan while maximizing customer benefits.
theorem part1 (price : ℝ) : 
  (weekly_profit price = 7500) →  -- Given condition for weekly profit
  (price = 65) := sorry  -- Conclude that the price must be 65 yuan for maximizing customer benefits

-- Part 2: Selling price to maximize the weekly profit and the maximum profit
theorem part2 : 
  ∃ price : ℝ, (price = 70 ∧ weekly_profit price = 8000) := sorry  -- Conclude that the price is 70 yuan and max profit is 8000 yuan

end NUMINAMATH_GPT_part1_part2_l860_86056


namespace NUMINAMATH_GPT_symphony_orchestra_has_260_members_l860_86059

def symphony_orchestra_member_count (n : ℕ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 6 = 2 ∧ n % 8 = 3 ∧ n % 9 = 4

theorem symphony_orchestra_has_260_members : symphony_orchestra_member_count 260 :=
by {
  sorry
}

end NUMINAMATH_GPT_symphony_orchestra_has_260_members_l860_86059


namespace NUMINAMATH_GPT_number_of_roses_per_set_l860_86099

-- Define the given conditions
def total_days : ℕ := 7
def sets_per_day : ℕ := 2
def total_roses : ℕ := 168

-- Define the statement to be proven
theorem number_of_roses_per_set : 
  (sets_per_day * total_days * (total_roses / (sets_per_day * total_days)) = total_roses) ∧ 
  (total_roses / (sets_per_day * total_days) = 12) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_roses_per_set_l860_86099


namespace NUMINAMATH_GPT_first_day_more_than_200_paperclips_l860_86034

def paperclips_after_days (k : ℕ) : ℕ :=
  3 * 2^k

theorem first_day_more_than_200_paperclips : (∀ k, 3 * 2^k <= 200) → k <= 7 → 3 * 2^7 > 200 → k = 7 :=
by
  intro h_le h_lt h_gt
  sorry

end NUMINAMATH_GPT_first_day_more_than_200_paperclips_l860_86034


namespace NUMINAMATH_GPT_service_fee_correct_l860_86016
open Nat -- Open the natural number namespace

-- Define the conditions
def ticket_price : ℕ := 44
def num_tickets : ℕ := 3
def total_paid : ℕ := 150

-- Define the cost of tickets
def cost_of_tickets : ℕ := ticket_price * num_tickets

-- Define the service fee calculation
def service_fee : ℕ := total_paid - cost_of_tickets

-- The proof problem statement
theorem service_fee_correct : service_fee = 18 :=
by
  -- Omits the proof, providing a placeholder.
  sorry

end NUMINAMATH_GPT_service_fee_correct_l860_86016


namespace NUMINAMATH_GPT_canoes_built_by_April_l860_86084

theorem canoes_built_by_April :
  (∃ (c1 c2 c3 c4 : ℕ), 
    c1 = 5 ∧ 
    c2 = 3 * c1 ∧ 
    c3 = 3 * c2 ∧ 
    c4 = 3 * c3 ∧
    (c1 + c2 + c3 + c4) = 200) :=
sorry

end NUMINAMATH_GPT_canoes_built_by_April_l860_86084


namespace NUMINAMATH_GPT_subtract_angles_l860_86080

theorem subtract_angles :
  (90 * 60 * 60 - (78 * 60 * 60 + 28 * 60 + 56)) = (11 * 60 * 60 + 31 * 60 + 4) :=
by
  sorry

end NUMINAMATH_GPT_subtract_angles_l860_86080


namespace NUMINAMATH_GPT_minimum_score_118_l860_86078

noncomputable def minimum_score (μ σ : ℝ) (p : ℝ) : ℝ :=
  sorry

theorem minimum_score_118 :
  minimum_score 98 10 (9100 / 400000) = 118 :=
by sorry

end NUMINAMATH_GPT_minimum_score_118_l860_86078


namespace NUMINAMATH_GPT_finite_S_k_iff_k_power_of_2_l860_86008

def S_k_finite (k : ℕ) : Prop :=
  ∃ (n a b : ℕ), (n ≠ 0 ∧ n % 2 = 1) ∧ (a + b = k) ∧ (Nat.gcd a b = 1) ∧ (n ∣ (a^n + b^n))

theorem finite_S_k_iff_k_power_of_2 (k : ℕ) (h : k > 1) : 
  (∀ n a b, n ≠ 0 → n % 2 = 1 → a + b = k → Nat.gcd a b = 1 → n ∣ (a^n + b^n) → false) ↔ 
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end NUMINAMATH_GPT_finite_S_k_iff_k_power_of_2_l860_86008


namespace NUMINAMATH_GPT_range_of_m_l860_86090

noncomputable def system_of_equations (x y m : ℝ) : Prop :=
  (x + 2 * y = 1 - m) ∧ (2 * x + y = 3)

variable (x y m : ℝ)

theorem range_of_m (h : system_of_equations x y m) (hxy : x + y > 0) : m < 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l860_86090


namespace NUMINAMATH_GPT_maximum_value_expression_l860_86018

-- Defining the variables and the main condition
variables (x y z : ℝ)

-- Assuming the non-negativity and sum of squares conditions
variables (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x^2 + y^2 + z^2 = 1)

-- Main statement about the maximum value
theorem maximum_value_expression : 
  4 * x * y * Real.sqrt 2 + 5 * y * z + 3 * x * z * Real.sqrt 3 ≤ 
  (44 * Real.sqrt 2 + 110 + 9 * Real.sqrt 3) / 3 :=
sorry

end NUMINAMATH_GPT_maximum_value_expression_l860_86018


namespace NUMINAMATH_GPT_min_value_h_l860_86027

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 4
noncomputable def g (x : ℝ) : ℝ := abs (Real.sin x) + 4 / abs (Real.sin x)
noncomputable def h (x : ℝ) : ℝ := 2^x + 2^(2-x)
noncomputable def j (x : ℝ) : ℝ := Real.log x + 4 / Real.log x

theorem min_value_h : ∃ x, h x = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_h_l860_86027


namespace NUMINAMATH_GPT_julius_wins_probability_l860_86017

noncomputable def probability_julius_wins (p_julius p_larry : ℚ) : ℚ :=
  (p_julius / (1 - p_larry ^ 2))

theorem julius_wins_probability :
  probability_julius_wins (2/3) (1/3) = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_julius_wins_probability_l860_86017


namespace NUMINAMATH_GPT_student_entrepreneur_profit_l860_86058

theorem student_entrepreneur_profit {x y a: ℝ} 
  (h1 : a * (y - x) = 1000) 
  (h2 : (ay / x) * y - ay = 1500)
  (h3 : y = 3 / 2 * x) : a * x = 2000 := 
sorry

end NUMINAMATH_GPT_student_entrepreneur_profit_l860_86058


namespace NUMINAMATH_GPT_find_b_value_l860_86066

theorem find_b_value
  (b : ℝ) :
  (∃ x y : ℝ, x = 3 ∧ y = -5 ∧ b * x + (b + 2) * y = b - 1) → b = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l860_86066


namespace NUMINAMATH_GPT_remainder_r15_minus_1_l860_86038

theorem remainder_r15_minus_1 (r : ℝ) : 
    (r^15 - 1) % (r - 1) = 0 :=
sorry

end NUMINAMATH_GPT_remainder_r15_minus_1_l860_86038


namespace NUMINAMATH_GPT_probability_blue_face_l860_86052

theorem probability_blue_face :
  (3 / 6 : ℝ) = (1 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_face_l860_86052


namespace NUMINAMATH_GPT_domain_of_sqrt_2_cos_x_minus_1_l860_86025

theorem domain_of_sqrt_2_cos_x_minus_1 :
  {x : ℝ | ∃ k : ℤ, - (Real.pi / 3) + 2 * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + 2 * k * Real.pi } =
  {x : ℝ | 2 * Real.cos x - 1 ≥ 0 } :=
sorry

end NUMINAMATH_GPT_domain_of_sqrt_2_cos_x_minus_1_l860_86025


namespace NUMINAMATH_GPT_ellipse_focal_distance_correct_l860_86022

noncomputable def ellipse_focal_distance (x y : ℝ) (θ : ℝ) : ℝ :=
  let a := 5 -- semi-major axis
  let b := 2 -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2) -- calculate focal distance
  2 * c -- return 2c

theorem ellipse_focal_distance_correct (θ : ℝ) :
  ellipse_focal_distance (-4 + 2 * Real.cos θ) (1 + 5 * Real.sin θ) θ = 2 * Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_focal_distance_correct_l860_86022


namespace NUMINAMATH_GPT_total_rocks_l860_86019

-- Definitions of variables based on the conditions
variables (igneous shiny_igneous : ℕ) (sedimentary : ℕ) (metamorphic : ℕ) (comet shiny_comet : ℕ)
variables (h1 : 1 / 4 * igneous = 15) (h2 : 1 / 2 * comet = 20)
variables (h3 : comet = 2 * metamorphic) (h4 : igneous = 3 * metamorphic)
variables (h5 : sedimentary = 2 * igneous)

-- The statement to be proved: the total number of rocks is 240
theorem total_rocks (igneous sedimentary metamorphic comet : ℕ) 
  (h1 : igneous = 4 * 15) 
  (h2 : comet = 2 * 20)
  (h3 : comet = 2 * metamorphic) 
  (h4 : igneous = 3 * metamorphic) 
  (h5 : sedimentary = 2 * igneous) : 
  igneous + sedimentary + metamorphic + comet = 240 :=
sorry

end NUMINAMATH_GPT_total_rocks_l860_86019


namespace NUMINAMATH_GPT_math_quiz_l860_86046

theorem math_quiz (x : ℕ) : 
  (∃ x ≥ 14, (∃ y : ℕ, 16 = x + y + 1) → (6 * x - 2 * y ≥ 75)) → 
  x ≥ 14 :=
by
  sorry

end NUMINAMATH_GPT_math_quiz_l860_86046


namespace NUMINAMATH_GPT_option_a_option_b_option_d_l860_86076

theorem option_a (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 - b^2 ≤ 4 := 
sorry

theorem option_b (a b : ℝ) (h1 : a > 0) (h2 : a^2 = 4 * b) : a^2 + 1 / b ≥ 4 :=
sorry

theorem option_d (a b c x1 x2 : ℝ) 
  (h1 : a > 0) 
  (h2 : a^2 = 4 * b) 
  (h3 : (x1 - x2) = 4)
  (h4 : (x1 + x2)^2 - 4 * (x1 * x2 + c) = (x1 - x2)^2) : c = 4 :=
sorry

end NUMINAMATH_GPT_option_a_option_b_option_d_l860_86076


namespace NUMINAMATH_GPT_proposition_false_l860_86002

theorem proposition_false (x y : ℤ) (h : x + y = 5) : ¬ (x = 1 ∧ y = 4) := by 
  sorry

end NUMINAMATH_GPT_proposition_false_l860_86002


namespace NUMINAMATH_GPT_product_of_consecutive_integers_eq_255_l860_86044

theorem product_of_consecutive_integers_eq_255 (x : ℕ) (h : x * (x + 1) = 255) : x + (x + 1) = 31 := 
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_eq_255_l860_86044


namespace NUMINAMATH_GPT_three_digit_numbers_containing_2_and_exclude_6_l860_86064

def three_digit_numbers_exclude_2_6 := 7 * (8 * 8)
def three_digit_numbers_exclude_6 := 8 * (9 * 9)
def three_digit_numbers_include_2_exclude_6 := three_digit_numbers_exclude_6 - three_digit_numbers_exclude_2_6

theorem three_digit_numbers_containing_2_and_exclude_6 :
  three_digit_numbers_include_2_exclude_6 = 200 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_containing_2_and_exclude_6_l860_86064


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l860_86070

variable (a : ℝ)

theorem necessary_but_not_sufficient (h : a ≥ 2) : (a = 2 ∨ a > 2) ∧ ¬(a > 2 → a ≥ 2) := by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l860_86070


namespace NUMINAMATH_GPT_least_integer_k_l860_86015

theorem least_integer_k (k : ℕ) (h : k ^ 3 ∣ 336) : k = 84 :=
sorry

end NUMINAMATH_GPT_least_integer_k_l860_86015


namespace NUMINAMATH_GPT_find_g_60_l860_86051

noncomputable def g (x : ℝ) : ℝ := sorry

axiom g_func_eq (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : g (x * y) = g x / y^2
axiom g_45 : g 45 = 15

theorem find_g_60 : g 60 = 8.4375 := sorry

end NUMINAMATH_GPT_find_g_60_l860_86051


namespace NUMINAMATH_GPT_find_s_l860_86054

variable (x t s : ℝ)

-- Conditions
#check (0.75 * x) / 60  -- Time for the first part of the trip
#check 0.25 * x  -- Distance for the remaining part of the trip
#check t - (0.75 * x) / 60  -- Time for the remaining part of the trip
#check 40 * t  -- Solving for x from average speed relation

-- Prove the value of s
theorem find_s (h1 : x = 40 * t) (h2 : s = (0.25 * x) / (t - (0.75 * x) / 60)) : s = 20 := by sorry

end NUMINAMATH_GPT_find_s_l860_86054


namespace NUMINAMATH_GPT_aaron_erasers_l860_86081

theorem aaron_erasers (initial_erasers erasers_given_to_Doris erasers_given_to_Ethan erasers_given_to_Fiona : ℕ) 
  (h1 : initial_erasers = 225) 
  (h2 : erasers_given_to_Doris = 75) 
  (h3 : erasers_given_to_Ethan = 40) 
  (h4 : erasers_given_to_Fiona = 50) : 
  initial_erasers - (erasers_given_to_Doris + erasers_given_to_Ethan + erasers_given_to_Fiona) = 60 :=
by sorry

end NUMINAMATH_GPT_aaron_erasers_l860_86081


namespace NUMINAMATH_GPT_loan_payment_period_years_l860_86037

noncomputable def house_cost := 480000
noncomputable def trailer_cost := 120000
noncomputable def monthly_difference := 1500

theorem loan_payment_period_years:
  ∃ N : ℕ, (house_cost = (trailer_cost / N + monthly_difference) * N ∧
            N = 240) →
            N / 12 = 20 :=
sorry

end NUMINAMATH_GPT_loan_payment_period_years_l860_86037


namespace NUMINAMATH_GPT_domain_of_f_l860_86045

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : {x : ℝ | ∃ y : ℝ, f y = x} = {x : ℝ | x ≠ 6} := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l860_86045


namespace NUMINAMATH_GPT_dogs_not_eat_either_l860_86053

-- Definitions for our conditions
variable (dogs_total : ℕ) (dogs_watermelon : ℕ) (dogs_salmon : ℕ) (dogs_both : ℕ)

-- Specific values of our conditions
def dogs_total_value : ℕ := 60
def dogs_watermelon_value : ℕ := 9
def dogs_salmon_value : ℕ := 48
def dogs_both_value : ℕ := 5

-- The theorem we need to prove
theorem dogs_not_eat_either : 
    dogs_total = dogs_total_value → 
    dogs_watermelon = dogs_watermelon_value → 
    dogs_salmon = dogs_salmon_value → 
    dogs_both = dogs_both_value → 
    (dogs_total - (dogs_watermelon + dogs_salmon - dogs_both) = 8) :=
by
  intros
  sorry

end NUMINAMATH_GPT_dogs_not_eat_either_l860_86053


namespace NUMINAMATH_GPT_product_simplification_l860_86095

theorem product_simplification :
  (1 + (1 / 1)) * (1 + (1 / 2)) * (1 + (1 / 3)) * (1 + (1 / 4)) * (1 + (1 / 5)) * (1 + (1 / 6)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_product_simplification_l860_86095


namespace NUMINAMATH_GPT_cos_diff_simplify_l860_86092

theorem cos_diff_simplify (x : ℝ) (y : ℝ) (h1 : x = Real.cos (Real.pi / 10)) (h2 : y = Real.cos (3 * Real.pi / 10)) : 
  x - y = 4 * x * (1 - x^2) := 
sorry

end NUMINAMATH_GPT_cos_diff_simplify_l860_86092


namespace NUMINAMATH_GPT_stable_k_digit_number_l860_86096

def is_stable (a k : ℕ) : Prop :=
  ∀ m n : ℕ, (10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a))

theorem stable_k_digit_number (k : ℕ) (h_pos : k > 0) : ∃ (a : ℕ) (h : ∀ m n : ℕ, 10^k ∣ ((m * 10^k + a) * (n * 10^k + a) - a)), (10^(k-1)) ≤ a ∧ a < 10^k ∧ ∀ b : ℕ, (∀ m n : ℕ, 10^k ∣ ((m * 10^k + b) * (n * 10^k + b) - b)) → (10^(k-1)) ≤ b ∧ b < 10^k → a = b :=
by
  sorry

end NUMINAMATH_GPT_stable_k_digit_number_l860_86096


namespace NUMINAMATH_GPT_intersection_A_B_l860_86049

def is_defined (x : ℝ) : Prop := x^2 - 1 ≥ 0

def range_of_y (y : ℝ) : Prop := y ≥ 0

def A_set : Set ℝ := { x | is_defined x }
def B_set : Set ℝ := { y | range_of_y y }

theorem intersection_A_B : A_set ∩ B_set = { x | 1 ≤ x } := 
sorry

end NUMINAMATH_GPT_intersection_A_B_l860_86049


namespace NUMINAMATH_GPT_percentage_markup_l860_86091

theorem percentage_markup (selling_price cost_price : ℝ) (h₁ : selling_price = 4800) (h₂ : cost_price = 3840) :
  (selling_price - cost_price) / cost_price * 100 = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_markup_l860_86091


namespace NUMINAMATH_GPT_find_cost_price_l860_86010

theorem find_cost_price (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (h1 : SP = 715) (h2 : profit_percent = 0.10) (h3 : SP = CP * (1 + profit_percent)) : 
  CP = 650 :=
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l860_86010


namespace NUMINAMATH_GPT_tens_digit_of_3_pow_100_l860_86062

-- Definition: The cyclic behavior of the last two digits of 3^n.
def last_two_digits_cycle : List ℕ := [03, 09, 27, 81, 43, 29, 87, 61, 83, 49, 47, 41, 23, 69, 07, 21, 63, 89, 67, 01]

-- Condition: The length of the cycle of the last two digits of 3^n.
def cycle_length : ℕ := 20

-- Assertion: The last two digits of 3^20 is 01.
def last_two_digits_3_pow_20 : ℕ := 1

-- Given n = 100, the tens digit of 3^n when n is expressed in decimal notation
theorem tens_digit_of_3_pow_100 : (3 ^ 100 / 10) % 10 = 0 := by
  let n := 100
  let position_in_cycle := (n % cycle_length)
  have cycle_repeat : (n % cycle_length = 0) := rfl
  have digits_3_pow_20 : (3^20 % 100 = 1) := by sorry
  show (3 ^ 100 / 10) % 10 = 0
  sorry

end NUMINAMATH_GPT_tens_digit_of_3_pow_100_l860_86062


namespace NUMINAMATH_GPT_probability_before_third_ring_l860_86036

-- Definitions of the conditions
def prob_first_ring : ℝ := 0.2
def prob_second_ring : ℝ := 0.3

-- Theorem stating that the probability of being answered before the third ring is 0.5
theorem probability_before_third_ring : prob_first_ring + prob_second_ring = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_probability_before_third_ring_l860_86036


namespace NUMINAMATH_GPT_total_cost_fencing_l860_86024

/-
  Given conditions:
  1. Length of the plot (l) = 55 meters
  2. Length is 10 meters more than breadth (b): l = b + 10
  3. Cost of fencing per meter (cost_per_meter) = 26.50
  
  Prove that the total cost of fencing the plot is 5300 currency units.
-/
def length : ℕ := 55
def breadth : ℕ := length - 10
def cost_per_meter : ℝ := 26.50
def perimeter : ℕ := 2 * (length + breadth)
def total_cost : ℝ := cost_per_meter * perimeter

theorem total_cost_fencing : total_cost = 5300 := by
  sorry

end NUMINAMATH_GPT_total_cost_fencing_l860_86024


namespace NUMINAMATH_GPT_min_value_condition_l860_86083

noncomputable def poly_min_value (a b : ℝ) : ℝ := a^2 + b^2

theorem min_value_condition (a b : ℝ) (h: ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : 
  ∃ a b : ℝ, poly_min_value a b = 4 := 
by sorry

end NUMINAMATH_GPT_min_value_condition_l860_86083


namespace NUMINAMATH_GPT_no_such_function_exists_l860_86048

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f (f n) = n + 1987 := 
sorry

end NUMINAMATH_GPT_no_such_function_exists_l860_86048


namespace NUMINAMATH_GPT_power_of_two_has_half_nines_l860_86086

theorem power_of_two_has_half_nines (k : ℕ) (h : k > 1) :
  ∃ n : ℕ, (∃ m : ℕ, (k / 2 < m) ∧ 
            (10^k ∣ (2^n + m + 1)) ∧ 
            (2^n % (10^k) = 10^k - 1)) :=
sorry

end NUMINAMATH_GPT_power_of_two_has_half_nines_l860_86086


namespace NUMINAMATH_GPT_find_other_number_l860_86020

theorem find_other_number
  (a b : ℕ)  -- Define the numbers as natural numbers
  (h1 : a = 300)             -- Condition stating the certain number is 300
  (h2 : a = 150 * b)         -- Condition stating the ratio is 150:1
  : b = 2 :=                 -- Goal stating the other number should be 2
  by
    sorry                    -- Placeholder for the proof steps

end NUMINAMATH_GPT_find_other_number_l860_86020


namespace NUMINAMATH_GPT_remainder_5x_div_9_l860_86077

theorem remainder_5x_div_9 {x : ℕ} (h : x % 9 = 5) : (5 * x) % 9 = 7 :=
sorry

end NUMINAMATH_GPT_remainder_5x_div_9_l860_86077


namespace NUMINAMATH_GPT_sales_in_third_month_is_6855_l860_86021

noncomputable def sales_in_third_month : ℕ :=
  let sale_1 := 6435
  let sale_2 := 6927
  let sale_4 := 7230
  let sale_5 := 6562
  let sale_6 := 6791
  let total_sales := 6800 * 6
  total_sales - (sale_1 + sale_2 + sale_4 + sale_5 + sale_6)

theorem sales_in_third_month_is_6855 : sales_in_third_month = 6855 := by
  sorry

end NUMINAMATH_GPT_sales_in_third_month_is_6855_l860_86021


namespace NUMINAMATH_GPT_find_a11_l860_86082

-- Defining the sequence a_n and its properties
def seq (a : ℕ → ℝ) : Prop :=
  (a 3 = 2) ∧ 
  (a 5 = 1) ∧ 
  (∃ d, ∀ n, (1 / (1 + a n)) = (1 / (1 + a 1)) + (n - 1) * d)

-- The goal is to prove that the value of a_{11} is 0
theorem find_a11 (a : ℕ → ℝ) (h : seq a) : a 11 = 0 :=
sorry

end NUMINAMATH_GPT_find_a11_l860_86082


namespace NUMINAMATH_GPT_total_mustard_bottles_l860_86068

theorem total_mustard_bottles : 
  let table1 : ℝ := 0.25
  let table2 : ℝ := 0.25
  let table3 : ℝ := 0.38
  table1 + table2 + table3 = 0.88 :=
by
  sorry

end NUMINAMATH_GPT_total_mustard_bottles_l860_86068


namespace NUMINAMATH_GPT_shoes_total_price_l860_86088

-- Define the variables involved
variables (S J : ℝ)

-- Define the conditions
def condition1 : Prop := J = (1 / 4) * S
def condition2 : Prop := 6 * S + 4 * J = 560

-- Define the total price calculation
def total_price : ℝ := 6 * S

-- State the theorem and proof goal
theorem shoes_total_price (h1 : condition1 S J) (h2 : condition2 S J) : total_price S = 480 := 
sorry

end NUMINAMATH_GPT_shoes_total_price_l860_86088


namespace NUMINAMATH_GPT_find_n_from_t_l860_86072

theorem find_n_from_t (n t : ℕ) (h1 : t = n * (n - 1) * (n + 1) + n) (h2 : t = 64) : n = 4 := by
  sorry

end NUMINAMATH_GPT_find_n_from_t_l860_86072


namespace NUMINAMATH_GPT_difference_between_percent_and_value_is_five_l860_86026

def hogs : ℕ := 75
def ratio : ℕ := 3

def num_of_cats (hogs : ℕ) (ratio : ℕ) : ℕ := hogs / ratio

def cats : ℕ := num_of_cats hogs ratio

def percent_of_cats (cats : ℕ) : ℝ := 0.60 * cats
def value_to_subtract : ℕ := 10

def difference (percent : ℝ) (value : ℕ) : ℝ := percent - value

theorem difference_between_percent_and_value_is_five
    (hogs : ℕ)
    (ratio : ℕ)
    (cats : ℕ := num_of_cats hogs ratio)
    (percent : ℝ := percent_of_cats cats)
    (value : ℕ := value_to_subtract)
    :
    difference percent value = 5 :=
by {
    sorry
}

end NUMINAMATH_GPT_difference_between_percent_and_value_is_five_l860_86026


namespace NUMINAMATH_GPT_confidence_level_unrelated_l860_86041

noncomputable def chi_squared_value : ℝ := 8.654

theorem confidence_level_unrelated :
  chi_squared_value > 6.635 →
  (100 - 99) = 1 :=
by
  sorry

end NUMINAMATH_GPT_confidence_level_unrelated_l860_86041


namespace NUMINAMATH_GPT_initial_percentage_of_water_l860_86039

variable (P : ℚ) -- Initial percentage of water

theorem initial_percentage_of_water (h : P / 100 * 40 + 5 = 9) : P = 10 := 
  sorry

end NUMINAMATH_GPT_initial_percentage_of_water_l860_86039


namespace NUMINAMATH_GPT_total_subjects_l860_86061

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end NUMINAMATH_GPT_total_subjects_l860_86061


namespace NUMINAMATH_GPT_angle_in_parallelogram_l860_86011

theorem angle_in_parallelogram (EFGH : Parallelogram) (angle_EFG angle_FGH : ℝ)
  (h1 : angle_EFG = angle_FGH + 90) : angle_EHG = 45 :=
by sorry

end NUMINAMATH_GPT_angle_in_parallelogram_l860_86011


namespace NUMINAMATH_GPT_area_of_quadrilateral_l860_86043

theorem area_of_quadrilateral (d o1 o2 : ℝ) (h1 : d = 24) (h2 : o1 = 9) (h3 : o2 = 6) :
  (1 / 2 * d * o1) + (1 / 2 * d * o2) = 180 :=
by {
  sorry
}

end NUMINAMATH_GPT_area_of_quadrilateral_l860_86043


namespace NUMINAMATH_GPT_find_smaller_number_l860_86013

-- Define the conditions
def sum_of_numbers (x y : ℕ) := x + y = 70
def second_number_relation (x y : ℕ) := y = 3 * x + 10

-- Define the problem statement
theorem find_smaller_number (x y : ℕ) (h1 : sum_of_numbers x y) (h2 : second_number_relation x y) : x = 15 :=
sorry

end NUMINAMATH_GPT_find_smaller_number_l860_86013


namespace NUMINAMATH_GPT_range_of_m_l860_86075

noncomputable def f (x : ℝ) (m : ℝ) :=
if x ≤ 2 then x^2 - m * (2 * x - 1) + m^2 else 2^(x + 1)

theorem range_of_m {m : ℝ} :
  (∀ x, f x m ≥ f 2 m) → (2 ≤ m ∧ m ≤ 4) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l860_86075


namespace NUMINAMATH_GPT_part1_part2_l860_86085

variable {f : ℝ → ℝ}

theorem part1 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : f 1 = 0 :=
by sorry

theorem part2 (h1 : ∀ a b : ℝ, 0 < a ∧ 0 < b → f (a * b) = f a + f b)
              (h2 : ∀ a b : ℝ, 0 < a ∧ 0 < b ∧ a ≤ b → f a ≥ f b)
              (h3 : f 2 = 1) : ∀ x : ℝ, -1 ≤ x ∧ x < 0 → f (-x) + f (3 - x) ≥ 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l860_86085


namespace NUMINAMATH_GPT_empty_subset_of_disjoint_and_nonempty_l860_86005

variable {α : Type*} (A B : Set α)

theorem empty_subset_of_disjoint_and_nonempty (h₁ : A ≠ ∅) (h₂ : A ∩ B = ∅) : ∅ ⊆ B :=
by
  sorry

end NUMINAMATH_GPT_empty_subset_of_disjoint_and_nonempty_l860_86005


namespace NUMINAMATH_GPT_remainder_is_20_l860_86087

def N := 220020
def a := 555
def b := 445
def d := a + b
def q := 2 * (a - b)

theorem remainder_is_20 : N % d = 20 := by
  sorry

end NUMINAMATH_GPT_remainder_is_20_l860_86087


namespace NUMINAMATH_GPT_num_black_squares_in_37th_row_l860_86007

-- Define the total number of squares in the n-th row
def total_squares_in_row (n : ℕ) : ℕ := 2 * n - 1

-- Define the number of black squares in the n-th row
def black_squares_in_row (n : ℕ) : ℕ := (total_squares_in_row n - 1) / 2

theorem num_black_squares_in_37th_row : black_squares_in_row 37 = 36 :=
by
  sorry

end NUMINAMATH_GPT_num_black_squares_in_37th_row_l860_86007


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_and_p_l860_86057

theorem express_y_in_terms_of_x_and_p (x p : ℚ) (h : x = (1 + p / 100) * (1 / y)) : 
  y = (100 + p) / (100 * x) := 
sorry

end NUMINAMATH_GPT_express_y_in_terms_of_x_and_p_l860_86057


namespace NUMINAMATH_GPT_cube_edges_after_cuts_l860_86094

theorem cube_edges_after_cuts (V E : ℕ) (hV : V = 8) (hE : E = 12) : 
  12 + 24 = 36 := by
  sorry

end NUMINAMATH_GPT_cube_edges_after_cuts_l860_86094


namespace NUMINAMATH_GPT_part1_part2_l860_86060

variable {a x y : ℝ} 

-- Conditions
def condition_1 (a x y : ℝ) := x - y = 1 + 3 * a
def condition_2 (a x y : ℝ) := x + y = -7 - a
def condition_3 (x : ℝ) := x ≤ 0
def condition_4 (y : ℝ) := y < 0

-- Part 1: Range for a
theorem part1 (a : ℝ) : 
  (∀ x y, condition_1 a x y ∧ condition_2 a x y ∧ condition_3 x ∧ condition_4 y → (-2 < a ∧ a ≤ 3)) :=
sorry

-- Part 2: Specific integer value for a
theorem part2 (a : ℝ) :
  (-2 < a ∧ a ≤ 3 → (∃ (x : ℝ), (2 * a + 1) * x > 2 * a + 1 ∧ x < 1) → a = -1) :=
sorry

end NUMINAMATH_GPT_part1_part2_l860_86060


namespace NUMINAMATH_GPT_find_constant_l860_86098

theorem find_constant
  {x : ℕ} (f : ℕ → ℕ)
  (h1 : ∀ x, f x = x^2 + 2*x + c)
  (h2 : f 2 = 12) :
  c = 4 :=
by sorry

end NUMINAMATH_GPT_find_constant_l860_86098


namespace NUMINAMATH_GPT_Liz_needs_more_money_l860_86031

theorem Liz_needs_more_money (P : ℝ) (h1 : P = 30000 + 2500) (h2 : 0.80 * P = 26000) : 30000 - (0.80 * P) = 4000 :=
by
  sorry

end NUMINAMATH_GPT_Liz_needs_more_money_l860_86031


namespace NUMINAMATH_GPT_least_integer_solution_l860_86065

theorem least_integer_solution :
  ∃ x : ℤ, (abs (3 * x - 4) ≤ 25) ∧ (∀ y : ℤ, (abs (3 * y - 4) ≤ 25) → x ≤ y) :=
sorry

end NUMINAMATH_GPT_least_integer_solution_l860_86065


namespace NUMINAMATH_GPT_pictures_per_album_l860_86023

theorem pictures_per_album (phone_pics camera_pics albums : ℕ) (h_phone : phone_pics = 22) (h_camera : camera_pics = 2) (h_albums : albums = 4) (h_total_pics : phone_pics + camera_pics = 24) : (phone_pics + camera_pics) / albums = 6 :=
by
  sorry

end NUMINAMATH_GPT_pictures_per_album_l860_86023


namespace NUMINAMATH_GPT_union_of_sets_l860_86030

open Set

-- Define the sets A and B
def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

-- Prove that the union of A and B equals {–2, 0, 3}
theorem union_of_sets : A ∪ B = {-2, 0, 3} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l860_86030
