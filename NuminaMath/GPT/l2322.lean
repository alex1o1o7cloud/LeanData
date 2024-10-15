import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_inequality_l2322_232269

theorem solution_set_of_inequality :
  {x : ℝ // (2 < x ∨ x < 2) ∧ x ≠ 3} =
  {x : ℝ // x < 2 ∨ 3 < x } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2322_232269


namespace NUMINAMATH_GPT_combined_tickets_l2322_232205

-- Definitions for the initial conditions
def stuffedTigerPrice : ℝ := 43
def keychainPrice : ℝ := 5.5
def discount1 : ℝ := 0.20 * stuffedTigerPrice
def discountedTigerPrice : ℝ := stuffedTigerPrice - discount1
def ticketsLeftDave : ℝ := 55
def spentDave : ℝ := discountedTigerPrice + keychainPrice
def initialTicketsDave : ℝ := spentDave + ticketsLeftDave

def dinoToyPrice : ℝ := 65
def discount2 : ℝ := 0.15 * dinoToyPrice
def discountedDinoToyPrice : ℝ := dinoToyPrice - discount2
def ticketsLeftAlex : ℝ := 42
def spentAlex : ℝ := discountedDinoToyPrice
def initialTicketsAlex : ℝ := spentAlex + ticketsLeftAlex

-- Lean statement proving the combined number of tickets at the start
theorem combined_tickets {dave_alex_combined : ℝ} 
    (h1 : dave_alex_combined = initialTicketsDave + initialTicketsAlex) : 
    dave_alex_combined = 192.15 := 
by 
    -- Placeholder for the actual proof
    sorry

end NUMINAMATH_GPT_combined_tickets_l2322_232205


namespace NUMINAMATH_GPT_pencils_per_box_l2322_232243

theorem pencils_per_box (total_pencils : ℝ) (num_boxes : ℝ) (pencils_per_box : ℝ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4.0) 
  (h3 : pencils_per_box = total_pencils / num_boxes) : 
  pencils_per_box = 648 :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_box_l2322_232243


namespace NUMINAMATH_GPT_min_value_2a_3b_equality_case_l2322_232285

theorem min_value_2a_3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) : 
  2 * a + 3 * b ≥ 25 :=
sorry

theorem equality_case (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 3 / b = 1) :
  (a = 5) ∧ (b = 5) → 2 * a + 3 * b = 25 :=
sorry

end NUMINAMATH_GPT_min_value_2a_3b_equality_case_l2322_232285


namespace NUMINAMATH_GPT_triangle_area_is_24_l2322_232283

-- Define the vertices
def vertex1 : ℝ × ℝ := (3, 2)
def vertex2 : ℝ × ℝ := (3, -4)
def vertex3 : ℝ × ℝ := (11, -4)

-- Define a function to calculate the area of a triangle given its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

-- Prove the area of the triangle with the given vertices is 24 square units
theorem triangle_area_is_24 : triangle_area vertex1 vertex2 vertex3 = 24 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_24_l2322_232283


namespace NUMINAMATH_GPT_greatest_sum_of_int_pairs_squared_eq_64_l2322_232256

theorem greatest_sum_of_int_pairs_squared_eq_64 :
  ∃ (x y : ℤ), x^2 + y^2 = 64 ∧ (∀ (a b : ℤ), a^2 + b^2 = 64 → a + b ≤ 8) ∧ x + y = 8 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_sum_of_int_pairs_squared_eq_64_l2322_232256


namespace NUMINAMATH_GPT_quadratic_real_roots_l2322_232220

theorem quadratic_real_roots (m : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ (x^2 - m * x + (m - 1) = 0) ∧ (y^2 - m * y + (m - 1) = 0) 
  ∨ ∃ z : ℝ, (z^2 - m * z + (m - 1) = 0) := 
sorry

end NUMINAMATH_GPT_quadratic_real_roots_l2322_232220


namespace NUMINAMATH_GPT_speed_ratio_l2322_232245

-- Define the speeds of A and B
variables (v_A v_B : ℝ)

-- Assume the conditions of the problem
axiom h1 : 200 / v_A = 400 / v_B

-- Prove the ratio of the speeds
theorem speed_ratio : v_A / v_B = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_speed_ratio_l2322_232245


namespace NUMINAMATH_GPT_locus_of_point_parabola_l2322_232261

/-- If the distance from point P to the point F (4, 0) is one unit less than its distance to the line x + 5 = 0, then the equation of the locus of point P is y^2 = 16x. -/
theorem locus_of_point_parabola :
  ∀ P : ℝ × ℝ, dist P (4, 0) + 1 = abs (P.1 + 5) → P.2^2 = 16 * P.1 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_point_parabola_l2322_232261


namespace NUMINAMATH_GPT_odd_function_period_2pi_l2322_232213

noncomputable def f (x : ℝ) : ℝ := Real.tan (x / 2)

theorem odd_function_period_2pi (x : ℝ) : 
  f (-x) = -f (x) ∧ 
  ∃ p > 0, p = 2 * Real.pi ∧ ∀ x, f (x + p) = f (x) := 
by
  sorry

end NUMINAMATH_GPT_odd_function_period_2pi_l2322_232213


namespace NUMINAMATH_GPT_probability_at_least_one_white_ball_stall_owner_monthly_earning_l2322_232206

noncomputable def prob_at_least_one_white_ball : ℚ :=
1 - (3 / 10)

theorem probability_at_least_one_white_ball : prob_at_least_one_white_ball = 9 / 10 :=
sorry

noncomputable def expected_monthly_earnings (daily_draws : ℕ) (days_in_month : ℕ) : ℤ :=
(days_in_month * (90 * 1 - 10 * 5))

theorem stall_owner_monthly_earning (daily_draws : ℕ) (days_in_month : ℕ) :
  daily_draws = 100 → days_in_month = 30 →
  expected_monthly_earnings daily_draws days_in_month = 1200 :=
sorry

end NUMINAMATH_GPT_probability_at_least_one_white_ball_stall_owner_monthly_earning_l2322_232206


namespace NUMINAMATH_GPT_tim_investment_l2322_232296

noncomputable def initial_investment_required 
  (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / ((1 + r / n) ^ (n * t))

theorem tim_investment :
  initial_investment_required 100000 0.10 2 3 = 74622 :=
by
  sorry

end NUMINAMATH_GPT_tim_investment_l2322_232296


namespace NUMINAMATH_GPT_intersecting_graphs_l2322_232255

theorem intersecting_graphs (a b c d : ℝ) (h₁ : (3, 6) = (3, -|3 - a| + b))
  (h₂ : (9, 2) = (9, -|9 - a| + b))
  (h₃ : (3, 6) = (3, |3 - c| + d))
  (h₄ : (9, 2) = (9, |9 - c| + d)) : 
  a + c = 12 := 
sorry

end NUMINAMATH_GPT_intersecting_graphs_l2322_232255


namespace NUMINAMATH_GPT_expected_value_eight_sided_die_win_l2322_232201

/-- The expected value of winning with a fair 8-sided die, where the win is \( n^3 \) dollars if \( n \) is rolled, is 162 dollars. -/
theorem expected_value_eight_sided_die_win :
  (1 / 8) * (1^3) + (1 / 8) * (2^3) + (1 / 8) * (3^3) + (1 / 8) * (4^3) +
  (1 / 8) * (5^3) + (1 / 8) * (6^3) + (1 / 8) * (7^3) + (1 / 8) * (8^3) = 162 := 
by
  -- Simplification and calculation here
  sorry

end NUMINAMATH_GPT_expected_value_eight_sided_die_win_l2322_232201


namespace NUMINAMATH_GPT_average_marks_is_25_l2322_232207

variable (M P C : ℕ)

def average_math_chemistry (M C : ℕ) : ℕ :=
  (M + C) / 2

theorem average_marks_is_25 (M P C : ℕ) 
  (h₁ : M + P = 30)
  (h₂ : C = P + 20) : 
  average_math_chemistry M C = 25 :=
by
  sorry

end NUMINAMATH_GPT_average_marks_is_25_l2322_232207


namespace NUMINAMATH_GPT_sum_interior_angles_polygon_l2322_232265

theorem sum_interior_angles_polygon (n : ℕ) (h : 180 * (n - 2) = 1440) :
  180 * ((n + 3) - 2) = 1980 := by
  sorry

end NUMINAMATH_GPT_sum_interior_angles_polygon_l2322_232265


namespace NUMINAMATH_GPT_smallest_number_is_3_l2322_232210

theorem smallest_number_is_3 (a b c : ℝ) (h1 : (a + b + c) / 3 = 7) (h2 : a = 9 ∨ b = 9 ∨ c = 9) : min (min a b) c = 3 := 
sorry

end NUMINAMATH_GPT_smallest_number_is_3_l2322_232210


namespace NUMINAMATH_GPT_smallest_number_condition_l2322_232222

theorem smallest_number_condition
  (x : ℕ)
  (h1 : (x - 24) % 5 = 0)
  (h2 : (x - 24) % 10 = 0)
  (h3 : (x - 24) % 15 = 0)
  (h4 : (x - 24) / 30 = 84)
  : x = 2544 := 
sorry

end NUMINAMATH_GPT_smallest_number_condition_l2322_232222


namespace NUMINAMATH_GPT_total_short_trees_l2322_232286

def short_trees_initial := 41
def short_trees_planted := 57

theorem total_short_trees : short_trees_initial + short_trees_planted = 98 := by
  sorry

end NUMINAMATH_GPT_total_short_trees_l2322_232286


namespace NUMINAMATH_GPT_units_digit_n_l2322_232254

theorem units_digit_n (m n : ℕ) (h₁ : m * n = 14^8) (hm : m % 10 = 6) : n % 10 = 1 :=
sorry

end NUMINAMATH_GPT_units_digit_n_l2322_232254


namespace NUMINAMATH_GPT_solve_for_m_l2322_232251

-- Define the conditions as hypotheses
def hyperbola_equation (x y : Real) (m : Real) : Prop :=
  (x^2)/(m+9) + (y^2)/9 = 1

def eccentricity (e : Real) (a b : Real) : Prop :=
  e = 2 ∧ e^2 = 1 + (b^2)/(a^2)

-- Prove that m = -36 given the conditions
theorem solve_for_m (m : Real) (h : hyperbola_equation x y m) (h_ecc : eccentricity 2 3 (Real.sqrt (-(m+9)))) :
  m = -36 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l2322_232251


namespace NUMINAMATH_GPT_jack_morning_emails_l2322_232228

theorem jack_morning_emails (x : ℕ) (aft_mails eve_mails total_morn_eve : ℕ) (h1: aft_mails = 4) (h2: eve_mails = 8) (h3: total_morn_eve = 11) :
  x = total_morn_eve - eve_mails :=
by 
  sorry

end NUMINAMATH_GPT_jack_morning_emails_l2322_232228


namespace NUMINAMATH_GPT_factory_sample_capacity_l2322_232240

theorem factory_sample_capacity (n : ℕ) (a_ratio b_ratio c_ratio : ℕ) 
  (total_ratio : a_ratio + b_ratio + c_ratio = 10) (a_sample : ℕ)
  (h : a_sample = 16) (h_ratio : a_ratio = 2) :
  n = 80 :=
by
  -- sample calculations proof would normally be here
  sorry

end NUMINAMATH_GPT_factory_sample_capacity_l2322_232240


namespace NUMINAMATH_GPT_find_a_and_b_l2322_232216

-- Define the line equation
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the curve equation
def curve (a b x : ℝ) : ℝ := x^3 + a * x + b

-- Define the derivative of the curve
def curve_derivative (a x : ℝ) : ℝ := 3 * x^2 + a

-- Main theorem to prove a = -1 and b = 3 given tangency conditions
theorem find_a_and_b 
  (k : ℝ) (a b : ℝ) (tangent_point : ℝ × ℝ)
  (h_tangent : tangent_point = (1, 3))
  (h_line : line k tangent_point.1 = tangent_point.2)
  (h_curve : curve a b tangent_point.1 = tangent_point.2)
  (h_slope : curve_derivative a tangent_point.1 = k) : 
  a = -1 ∧ b = 3 := 
by
  sorry

end NUMINAMATH_GPT_find_a_and_b_l2322_232216


namespace NUMINAMATH_GPT_potato_bag_weight_l2322_232274

theorem potato_bag_weight (w : ℕ) (h₁ : w = 36) : w = 36 :=
by
  sorry

end NUMINAMATH_GPT_potato_bag_weight_l2322_232274


namespace NUMINAMATH_GPT_exists_multiple_with_digits_0_or_1_l2322_232271

theorem exists_multiple_with_digits_0_or_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k % n = 0) ∧ (∀ digit ∈ k.digits 10, digit = 0 ∨ digit = 1) ∧ (k.digits 10).length ≤ n :=
sorry

end NUMINAMATH_GPT_exists_multiple_with_digits_0_or_1_l2322_232271


namespace NUMINAMATH_GPT_shaded_region_area_l2322_232238

theorem shaded_region_area
  (R r : ℝ)
  (h : r^2 = R^2 - 2500)
  : π * (R^2 - r^2) = 2500 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2322_232238


namespace NUMINAMATH_GPT_complement_of_A_in_U_l2322_232293

noncomputable def U : Set ℤ := {x : ℤ | x^2 ≤ 2*x + 3}
def A : Set ℤ := {0, 1, 2}

theorem complement_of_A_in_U : (U \ A) = {-1, 3} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l2322_232293


namespace NUMINAMATH_GPT_workdays_ride_l2322_232297

-- Define the conditions
def work_distance : ℕ := 20
def weekend_ride : ℕ := 200
def speed : ℕ := 25
def hours_per_week : ℕ := 16

-- Define the question
def total_distance : ℕ := speed * hours_per_week
def distance_during_workdays : ℕ := total_distance - weekend_ride
def round_trip_distance : ℕ := 2 * work_distance

theorem workdays_ride : 
  (distance_during_workdays / round_trip_distance) = 5 :=
by
  sorry

end NUMINAMATH_GPT_workdays_ride_l2322_232297


namespace NUMINAMATH_GPT_compare_log_inequalities_l2322_232234

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem compare_log_inequalities (a x1 x2 : ℝ) 
  (ha_pos : a > 0) (ha_neq_one : a ≠ 1) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (a > 1 → 1 / 2 * (f a x1 + f a x2) ≤ f a ((x1 + x2) / 2)) ∧
  (0 < a ∧ a < 1 → 1 / 2 * (f a x1 + f a x2) ≥ f a ((x1 + x2) / 2)) :=
by { sorry }

end NUMINAMATH_GPT_compare_log_inequalities_l2322_232234


namespace NUMINAMATH_GPT_sum_of_squares_l2322_232266

theorem sum_of_squares (a b c d e f : ℤ)
  (h : ∀ x : ℤ, 512 * x ^ 3 + 125 = (a * x ^ 2 + b * x + c) * (d * x ^ 2 + e * x + f)) :
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 6410 := 
sorry

end NUMINAMATH_GPT_sum_of_squares_l2322_232266


namespace NUMINAMATH_GPT_find_a_pure_imaginary_l2322_232214

theorem find_a_pure_imaginary (a : ℝ) (i : ℂ) (h1 : i = (0 : ℝ) + I) :
  (∃ b : ℝ, a - (17 / (4 - i)) = (0 + b*I)) → a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_pure_imaginary_l2322_232214


namespace NUMINAMATH_GPT_find_AM_l2322_232277

-- Definitions (conditions)
variables {A M B : ℝ}
variable  (collinear : A ≤ M ∧ M ≤ B ∨ B ≤ M ∧ M ≤ A ∨ A ≤ B ∧ B ≤ M)
          (h1 : abs (M - A) = 2 * abs (M - B)) 
          (h2 : abs (A - B) = 6)

-- Proof problem statement
theorem find_AM : (abs (M - A) = 4) ∨ (abs (M - A) = 12) :=
by 
  sorry

end NUMINAMATH_GPT_find_AM_l2322_232277


namespace NUMINAMATH_GPT_a_11_is_12_l2322_232258

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def a_2 (a : ℕ → ℝ) := a 2 = 3
def a_6 (a : ℕ → ℝ) := a 6 = 7

-- The statement to prove
theorem a_11_is_12 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a2 : a_2 a) (h_a6 : a_6 a) : a 11 = 12 :=
  sorry

end NUMINAMATH_GPT_a_11_is_12_l2322_232258


namespace NUMINAMATH_GPT_base_height_ratio_l2322_232264

-- Define the conditions
def cultivation_cost : ℝ := 333.18
def rate_per_hectare : ℝ := 24.68
def base_of_field : ℝ := 300
def height_of_field : ℝ := 300

-- Prove the ratio of base to height is 1
theorem base_height_ratio (b h : ℝ) (cost rate : ℝ)
  (h1 : cost = 333.18) (h2 : rate = 24.68) 
  (h3 : b = 300) (h4 : h = 300) : b / h = 1 :=
by
  sorry

end NUMINAMATH_GPT_base_height_ratio_l2322_232264


namespace NUMINAMATH_GPT_solve_for_x_l2322_232211

theorem solve_for_x (x : ℚ) (h : (x - 3) / (x + 2) + (3 * x - 9) / (x - 3) = 2) : x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2322_232211


namespace NUMINAMATH_GPT_find_a4_l2322_232280

variable (a_1 d : ℝ)

def a_n (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

axiom condition1 : (a_n a_1 d 2 + a_n a_1 d 6) / 2 = 5 * Real.sqrt 3
axiom condition2 : (a_n a_1 d 3 + a_n a_1 d 7) / 2 = 7 * Real.sqrt 3

theorem find_a4 : a_n a_1 d 4 = 5 * Real.sqrt 3 :=
by
  -- Proof should go here, but we insert "sorry" to mark it as incomplete.
  sorry

end NUMINAMATH_GPT_find_a4_l2322_232280


namespace NUMINAMATH_GPT_pyramid_volume_l2322_232236

def area_SAB : ℝ := 9
def area_SBC : ℝ := 9
def area_SCD : ℝ := 27
def area_SDA : ℝ := 27
def area_ABCD : ℝ := 36
def dihedral_angle_equal := ∀ (α β γ δ: ℝ), α = β ∧ β = γ ∧ γ = δ

theorem pyramid_volume (h_eq_dihedral : dihedral_angle_equal)
  (area_conditions : area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27)
  (area_quadrilateral : area_ABCD = 36) :
  (1 / 3 * area_ABCD * 4.5) = 54 :=
sorry

end NUMINAMATH_GPT_pyramid_volume_l2322_232236


namespace NUMINAMATH_GPT_find_n_l2322_232242

def C (k : ℕ) : ℕ :=
  if k = 1 then 0
  else (Nat.factors k).eraseDup.foldr (· + ·) 0

theorem find_n (n : ℕ) : 
  (∀ n, (C (2 ^ n + 1) = C n) ↔ n = 3) := 
by
  sorry

end NUMINAMATH_GPT_find_n_l2322_232242


namespace NUMINAMATH_GPT_find_x_l2322_232290

theorem find_x (n : ℕ) (h_odd : n % 2 = 1)
  (h_three_primes : ∃ (p1 p2 p3 : ℕ), p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ Nat.Prime p1 ∧ Nat.Prime p2 ∧ Nat.Prime p3 ∧ 
    11 = p1 ∧ (7 ^ n + 1) = p1 * p2 * p3) :
  (7 ^ n + 1) = 16808 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2322_232290


namespace NUMINAMATH_GPT_polynomial_sum_l2322_232246

def f (x : ℝ) : ℝ := -6 * x^2 + 2 * x - 7
def g (x : ℝ) : ℝ := -4 * x^2 + 4 * x - 3
def h (x : ℝ) : ℝ := 10 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : 
  f x + g x + (h x)^2 = 100 * x^4 + 120 * x^3 + 34 * x^2 + 30 * x - 6 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l2322_232246


namespace NUMINAMATH_GPT_cos_60_eq_half_l2322_232279

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_60_eq_half_l2322_232279


namespace NUMINAMATH_GPT_cyclic_sum_nonneg_l2322_232253

theorem cyclic_sum_nonneg 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (k : ℝ) (hk1 : 0 ≤ k) (hk2 : k < 2) :
  (a^2 - b * c) / (b^2 + c^2 + k * a^2)
  + (b^2 - c * a) / (c^2 + a^2 + k * b^2)
  + (c^2 - a * b) / (a^2 + b^2 + k * c^2) ≥ 0 :=
sorry

end NUMINAMATH_GPT_cyclic_sum_nonneg_l2322_232253


namespace NUMINAMATH_GPT_translated_point_is_correct_l2322_232291

-- Cartesian Point definition
structure Point where
  x : Int
  y : Int

-- Define the translation function
def translate (p : Point) (dx dy : Int) : Point :=
  Point.mk (p.x + dx) (p.y - dy)

-- Define the initial point A and the translation amounts
def A : Point := ⟨-3, 2⟩
def dx : Int := 3
def dy : Int := 2

-- The proof goal
theorem translated_point_is_correct :
  translate A dx dy = ⟨0, 0⟩ :=
by
  -- This is where the proof would normally go
  sorry

end NUMINAMATH_GPT_translated_point_is_correct_l2322_232291


namespace NUMINAMATH_GPT_trains_cross_time_l2322_232231

noncomputable def time_to_cross_trains : ℝ :=
  200 / (89.992800575953935 * (1000 / 3600))

theorem trains_cross_time :
  abs (time_to_cross_trains - 8) < 1e-7 :=
by
  sorry

end NUMINAMATH_GPT_trains_cross_time_l2322_232231


namespace NUMINAMATH_GPT_fraction_eval_l2322_232230

theorem fraction_eval : 1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 89 :=
by
  sorry

end NUMINAMATH_GPT_fraction_eval_l2322_232230


namespace NUMINAMATH_GPT_value_of_M_l2322_232239

theorem value_of_M (M : ℝ) (h : 0.25 * M = 0.35 * 4025) : M = 5635 :=
sorry

end NUMINAMATH_GPT_value_of_M_l2322_232239


namespace NUMINAMATH_GPT_solve_system_l2322_232288

theorem solve_system :
    (∃ x y z : ℝ, 5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0 ∧
                49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 =0
                ∧ ((x = 0 ∧ y = 1 ∧ z = -2)
                   ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7))) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2322_232288


namespace NUMINAMATH_GPT_art_club_artworks_l2322_232259

-- Define the conditions
def students := 25
def artworks_per_student_per_quarter := 3
def quarters_per_year := 4
def years := 3

-- Calculate total artworks
theorem art_club_artworks : 
  students * artworks_per_student_per_quarter * quarters_per_year * years = 900 :=
by
  sorry

end NUMINAMATH_GPT_art_club_artworks_l2322_232259


namespace NUMINAMATH_GPT_function_symmetry_implies_even_l2322_232281

theorem function_symmetry_implies_even (f : ℝ → ℝ) (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x = y ↔ -f (-x) = -y) : ∀ x, f x = f (-x) :=
by
  sorry

end NUMINAMATH_GPT_function_symmetry_implies_even_l2322_232281


namespace NUMINAMATH_GPT_maximize_revenue_l2322_232204

noncomputable def revenue (p : ℝ) : ℝ := 100 * p - 4 * p^2

theorem maximize_revenue : ∃ p : ℝ, 0 ≤ p ∧ p ≤ 20 ∧ (∀ q : ℝ, 0 ≤ q ∧ q ≤ 20 → revenue q ≤ revenue p) ∧ p = 12.5 := by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l2322_232204


namespace NUMINAMATH_GPT_cost_price_of_article_l2322_232282

theorem cost_price_of_article (x : ℝ) (h : 57 - x = x - 43) : x = 50 := 
by 
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l2322_232282


namespace NUMINAMATH_GPT_problem_statement_l2322_232260

theorem problem_statement (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2322_232260


namespace NUMINAMATH_GPT_find_m_l2322_232226

open Real

noncomputable def f (x m : ℝ) : ℝ :=
  2 * (sin x ^ 4 + cos x ^ 4) + m * (sin x + cos x) ^ 4

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 → f x m ≤ 5) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x m = 5) :=
sorry

end NUMINAMATH_GPT_find_m_l2322_232226


namespace NUMINAMATH_GPT_height_of_tank_A_l2322_232273

theorem height_of_tank_A (C_A C_B h_B : ℝ) (capacity_ratio : ℝ) :
  C_A = 8 → C_B = 10 → h_B = 8 → capacity_ratio = 0.4800000000000001 →
  ∃ h_A : ℝ, h_A = 6 := by
  intros hCA hCB hHB hCR
  sorry

end NUMINAMATH_GPT_height_of_tank_A_l2322_232273


namespace NUMINAMATH_GPT_rectangle_area_ratio_l2322_232217

theorem rectangle_area_ratio (l b : ℕ) (h1 : l = b + 10) (h2 : b = 8) : (l * b) / b = 18 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l2322_232217


namespace NUMINAMATH_GPT_find_a_l2322_232227

-- Define the conditions
def parabola_equation (a : ℝ) (x : ℝ) : ℝ := a * x^2
def axis_of_symmetry : ℝ := -2

-- The main theorem: proving the value of a
theorem find_a (a : ℝ) : (axis_of_symmetry = - (1 / (4 * a))) → a = 1/8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l2322_232227


namespace NUMINAMATH_GPT_largest_increase_is_2007_2008_l2322_232224

-- Define the number of students each year
def students_2005 : ℕ := 50
def students_2006 : ℕ := 55
def students_2007 : ℕ := 60
def students_2008 : ℕ := 70
def students_2009 : ℕ := 72
def students_2010 : ℕ := 80

-- Define the percentage increase function
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old) : ℚ) / old * 100

-- Define percentage increases for each pair of consecutive years
def increase_2005_2006 := percentage_increase students_2005 students_2006
def increase_2006_2007 := percentage_increase students_2006 students_2007
def increase_2007_2008 := percentage_increase students_2007 students_2008
def increase_2008_2009 := percentage_increase students_2008 students_2009
def increase_2009_2010 := percentage_increase students_2009 students_2010

-- State the theorem
theorem largest_increase_is_2007_2008 :
  (max (max increase_2005_2006 (max increase_2006_2007 increase_2008_2009))
       increase_2009_2010) < increase_2007_2008 := 
by
  -- Add proof steps if necessary.
  sorry

end NUMINAMATH_GPT_largest_increase_is_2007_2008_l2322_232224


namespace NUMINAMATH_GPT_projection_correct_l2322_232229

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def P : Point3D := ⟨-1, 3, -4⟩

def projection_yOz_plane (P : Point3D) : Point3D :=
  ⟨0, P.y, P.z⟩

theorem projection_correct :
  projection_yOz_plane P = ⟨0, 3, -4⟩ :=
by
  -- The theorem proof is omitted.
  sorry

end NUMINAMATH_GPT_projection_correct_l2322_232229


namespace NUMINAMATH_GPT_no_such_function_exists_l2322_232233

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ+ → ℕ+), ∀ (n : ℕ+), f^[n] n = n + 1 :=
by
  sorry

end NUMINAMATH_GPT_no_such_function_exists_l2322_232233


namespace NUMINAMATH_GPT_james_total_payment_is_correct_l2322_232267

-- Define the constants based on the conditions
def numDirtBikes : Nat := 3
def costPerDirtBike : Nat := 150
def numOffRoadVehicles : Nat := 4
def costPerOffRoadVehicle : Nat := 300
def numTotalVehicles : Nat := numDirtBikes + numOffRoadVehicles
def registrationCostPerVehicle : Nat := 25

-- Define the total calculation using the given conditions
def totalPaidByJames : Nat :=
  (numDirtBikes * costPerDirtBike) +
  (numOffRoadVehicles * costPerOffRoadVehicle) +
  (numTotalVehicles * registrationCostPerVehicle)

-- State the proof problem
theorem james_total_payment_is_correct : totalPaidByJames = 1825 := by
  sorry

end NUMINAMATH_GPT_james_total_payment_is_correct_l2322_232267


namespace NUMINAMATH_GPT_male_percentage_l2322_232257

theorem male_percentage (total_employees : ℕ)
  (males_below_50 : ℕ)
  (percentage_males_at_least_50 : ℝ)
  (male_percentage : ℝ) :
  total_employees = 2200 →
  males_below_50 = 616 →
  percentage_males_at_least_50 = 0.3 → 
  male_percentage = 40 :=
by
  sorry

end NUMINAMATH_GPT_male_percentage_l2322_232257


namespace NUMINAMATH_GPT_C_neither_necessary_nor_sufficient_for_A_l2322_232221

theorem C_neither_necessary_nor_sufficient_for_A 
  (A B C : Prop) 
  (h1 : B → C)
  (h2 : B → A) : 
  ¬(A → C) ∧ ¬(C → A) :=
by
  sorry

end NUMINAMATH_GPT_C_neither_necessary_nor_sufficient_for_A_l2322_232221


namespace NUMINAMATH_GPT_ab_a4_b4_divisible_by_30_l2322_232208

theorem ab_a4_b4_divisible_by_30 (a b : Int) : 30 ∣ a * b * (a^4 - b^4) := 
by
  sorry

end NUMINAMATH_GPT_ab_a4_b4_divisible_by_30_l2322_232208


namespace NUMINAMATH_GPT_sum_smallest_largest_even_integers_l2322_232275

theorem sum_smallest_largest_even_integers (m b z : ℕ) (hm_even : m % 2 = 0)
  (h_mean : z = (b + (b + 2 * (m - 1))) / 2) :
  (b + (b + 2 * (m - 1))) = 2 * z :=
by
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_even_integers_l2322_232275


namespace NUMINAMATH_GPT_smallest_positive_integer_in_linear_combination_l2322_232212

theorem smallest_positive_integer_in_linear_combination :
  ∃ m n : ℤ, 2016 * m + 43200 * n = 24 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_in_linear_combination_l2322_232212


namespace NUMINAMATH_GPT_incorrect_neg_p_l2322_232203

theorem incorrect_neg_p (p : ∀ x : ℝ, x ≥ 1) : ¬ (∀ x : ℝ, x < 1) :=
sorry

end NUMINAMATH_GPT_incorrect_neg_p_l2322_232203


namespace NUMINAMATH_GPT_age_ratio_l2322_232268

theorem age_ratio (S M : ℕ) (h₁ : M = S + 35) (h₂ : S = 33) : 
  (M + 2) / (S + 2) = 2 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_age_ratio_l2322_232268


namespace NUMINAMATH_GPT_symmetric_point_xOz_l2322_232241

theorem symmetric_point_xOz (x y z : ℝ) : (x, y, z) = (-1, 2, 1) → (x, -y, z) = (-1, -2, 1) :=
by
  intros h
  cases h
  sorry

end NUMINAMATH_GPT_symmetric_point_xOz_l2322_232241


namespace NUMINAMATH_GPT_original_sugar_amount_l2322_232295

theorem original_sugar_amount (f : ℕ) (s t r : ℕ) (h1 : f = 5) (h2 : r = 10) (h3 : t = 14) (h4 : f * 2 = r):
  s = t / 2 := sorry

end NUMINAMATH_GPT_original_sugar_amount_l2322_232295


namespace NUMINAMATH_GPT_one_fourth_of_eight_point_four_l2322_232289

theorem one_fourth_of_eight_point_four : (8.4 / 4) = (21 / 10) :=
by
  -- The expected proof would go here
  sorry

end NUMINAMATH_GPT_one_fourth_of_eight_point_four_l2322_232289


namespace NUMINAMATH_GPT_election_win_percentage_l2322_232252

theorem election_win_percentage (total_votes : ℕ) (james_percentage : ℝ) (additional_votes_needed : ℕ) (votes_needed_to_win_percentage : ℝ) :
    total_votes = 2000 →
    james_percentage = 0.005 →
    additional_votes_needed = 991 →
    votes_needed_to_win_percentage = (1001 / 2000) * 100 →
    votes_needed_to_win_percentage > 50.05 :=
by
  intros h_total_votes h_james_percentage h_additional_votes_needed h_votes_needed_to_win_percentage
  sorry

end NUMINAMATH_GPT_election_win_percentage_l2322_232252


namespace NUMINAMATH_GPT_award_medals_at_most_one_canadian_l2322_232235

/-- Definition of conditions -/
def sprinter_count := 10 -- Total number of sprinters
def canadian_sprinter_count := 4 -- Number of Canadian sprinters
def medals := ["Gold", "Silver", "Bronze"] -- Types of medals

/-- Definition stating the requirement of the problem -/
def atMostOneCanadianMedal (total_sprinters : Nat) (canadian_sprinters : Nat) 
    (medal_types : List String) : Bool := 
  if total_sprinters = sprinter_count ∧ canadian_sprinters = canadian_sprinter_count ∧ medal_types = medals then
    true
  else
    false

/-- Statement to prove the number of ways to award the medals -/
theorem award_medals_at_most_one_canadian :
  (atMostOneCanadianMedal sprinter_count canadian_sprinter_count medals) →
  ∃ (ways : Nat), ways = 480 :=
by
  sorry

end NUMINAMATH_GPT_award_medals_at_most_one_canadian_l2322_232235


namespace NUMINAMATH_GPT_pencils_per_pack_l2322_232270

def packs := 28
def rows := 42
def pencils_per_row := 16

theorem pencils_per_pack (total_pencils : ℕ) : total_pencils = rows * pencils_per_row → total_pencils / packs = 24 :=
by
  sorry

end NUMINAMATH_GPT_pencils_per_pack_l2322_232270


namespace NUMINAMATH_GPT_q_can_be_true_or_false_l2322_232250

-- Define the propositions p and q
variables (p q : Prop)

-- The assumptions given in the problem
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ p

-- The statement we want to prove
theorem q_can_be_true_or_false : ∀ q, q ∨ ¬ q :=
by
  intro q
  exact em q -- Use the principle of excluded middle

end NUMINAMATH_GPT_q_can_be_true_or_false_l2322_232250


namespace NUMINAMATH_GPT_a_minus_b_is_perfect_square_l2322_232232
-- Import necessary libraries

-- Define the problem in Lean
theorem a_minus_b_is_perfect_square (a b c : ℕ) (h1: Nat.gcd a (Nat.gcd b c) = 1) 
    (h2: (ab : ℚ) / (a - b) = c) : ∃ k : ℕ, a - b = k * k :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_is_perfect_square_l2322_232232


namespace NUMINAMATH_GPT_proposition_true_l2322_232219

theorem proposition_true (x y : ℝ) : x + 2 * y ≠ 5 → (x ≠ 1 ∨ y ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_proposition_true_l2322_232219


namespace NUMINAMATH_GPT_parallel_vectors_l2322_232202

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (0, 1)
def c (k : ℝ) : ℝ × ℝ := (-2, k)

theorem parallel_vectors (k : ℝ) (h : (1, 4) = c k) : k = -8 :=
sorry

end NUMINAMATH_GPT_parallel_vectors_l2322_232202


namespace NUMINAMATH_GPT_total_legs_correct_l2322_232223

-- Define the number of animals
def num_dogs : ℕ := 2
def num_chickens : ℕ := 1

-- Define the number of legs per animal
def legs_per_dog : ℕ := 4
def legs_per_chicken : ℕ := 2

-- Define the total number of legs from dogs and chickens
def total_legs : ℕ := num_dogs * legs_per_dog + num_chickens * legs_per_chicken

theorem total_legs_correct : total_legs = 10 :=
by
  -- this is where the proof would go, but we add sorry for now to skip it
  sorry

end NUMINAMATH_GPT_total_legs_correct_l2322_232223


namespace NUMINAMATH_GPT_arithmetic_evaluation_l2322_232248

theorem arithmetic_evaluation :
  (3.2 - 2.95) / (0.25 * 2 + 1/4) + (2 * 0.3) / (2.3 - (1 + 2/5)) = 1 := by
  sorry

end NUMINAMATH_GPT_arithmetic_evaluation_l2322_232248


namespace NUMINAMATH_GPT_roy_consumes_tablets_in_225_minutes_l2322_232272

variables 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ)

def total_time_to_consume_all_tablets 
  (total_tablets : ℕ) 
  (time_per_tablet : ℕ) : ℕ :=
  (total_tablets - 1) * time_per_tablet

theorem roy_consumes_tablets_in_225_minutes 
  (h1 : total_tablets = 10) 
  (h2 : time_per_tablet = 25) : 
  total_time_to_consume_all_tablets total_tablets time_per_tablet = 225 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_roy_consumes_tablets_in_225_minutes_l2322_232272


namespace NUMINAMATH_GPT_factorization_correct_l2322_232244

theorem factorization_correct (x y : ℝ) :
  x^4 - 2*x^2*y - 3*y^2 + 8*y - 4 = (x^2 + y - 2) * (x^2 - 3*y + 2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l2322_232244


namespace NUMINAMATH_GPT_circle_area_l2322_232262

theorem circle_area (x y : ℝ) : 
  x^2 + y^2 - 18 * x + 8 * y = -72 → 
  ∃ r : ℝ, r = 5 ∧ π * r ^ 2 = 25 * π := 
by
  sorry

end NUMINAMATH_GPT_circle_area_l2322_232262


namespace NUMINAMATH_GPT_roots_of_quadratic_l2322_232247

theorem roots_of_quadratic (x : ℝ) : x^2 + x = 0 ↔ (x = 0 ∨ x = -1) :=
by sorry

end NUMINAMATH_GPT_roots_of_quadratic_l2322_232247


namespace NUMINAMATH_GPT_right_angle_triangle_probability_l2322_232215

def vertex_count : ℕ := 16
def ways_to_choose_3_points : ℕ := Nat.choose vertex_count 3
def number_of_rectangles : ℕ := 36
def right_angle_triangles_per_rectangle : ℕ := 4
def total_right_angle_triangles : ℕ := number_of_rectangles * right_angle_triangles_per_rectangle
def probability_right_angle_triangle : ℚ := total_right_angle_triangles / ways_to_choose_3_points

theorem right_angle_triangle_probability :
  probability_right_angle_triangle = (9 / 35 : ℚ) := by
  sorry

end NUMINAMATH_GPT_right_angle_triangle_probability_l2322_232215


namespace NUMINAMATH_GPT_product_of_solutions_l2322_232218

theorem product_of_solutions : 
  ∀ x : ℝ, 5 = -2 * x^2 + 6 * x → (∃ α β : ℝ, (α ≠ β ∧ (α * β = 5 / 2))) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l2322_232218


namespace NUMINAMATH_GPT_alpha_beta_square_l2322_232298

-- Statement of the problem in Lean 4
theorem alpha_beta_square :
  ∀ (α β : ℝ), (α ≠ β ∧ ∀ x : ℝ, x^2 - 2 * x - 1 = 0 → (x = α ∨ x = β)) → (α - β)^2 = 8 := 
by
  intros α β h
  sorry

end NUMINAMATH_GPT_alpha_beta_square_l2322_232298


namespace NUMINAMATH_GPT_f_half_l2322_232284

theorem f_half (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - 2 * x) = 1 / (x ^ 2)) :
  f (1 / 2) = 16 :=
sorry

end NUMINAMATH_GPT_f_half_l2322_232284


namespace NUMINAMATH_GPT_point_in_third_quadrant_l2322_232249

theorem point_in_third_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : 
  (-b < 0) ∧ (a < 0) ∧ (-b > a) :=
by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l2322_232249


namespace NUMINAMATH_GPT_point_in_third_quadrant_l2322_232209

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end NUMINAMATH_GPT_point_in_third_quadrant_l2322_232209


namespace NUMINAMATH_GPT_average_speed_car_y_l2322_232292

-- Defining the constants based on the problem conditions
def speedX : ℝ := 35
def timeDifference : ℝ := 1.2  -- This is 72 minutes converted to hours
def distanceFromStartOfY : ℝ := 294

-- Defining the main statement
theorem average_speed_car_y : 
  ( ∀ timeX timeY distanceX distanceY : ℝ, 
      timeX = timeY + timeDifference ∧
      distanceX = speedX * timeX ∧
      distanceY = distanceFromStartOfY ∧
      distanceX = distanceFromStartOfY + speedX * timeDifference
  → distanceY / timeX = 30.625) :=
sorry

end NUMINAMATH_GPT_average_speed_car_y_l2322_232292


namespace NUMINAMATH_GPT_smallest_N_for_circular_table_l2322_232299

/--
  Given a circular table with 60 chairs, prove that the smallest number of people, N,
  such that any additional person must sit next to someone already seated is 20.
-/
theorem smallest_N_for_circular_table (N : ℕ) (h : N = 20) : 
  ∀ (next_seated : ℕ), next_seated ≤ N → (∃ i : ℕ, i < N ∧ next_seated = i + 1 ∨ next_seated = i - 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_N_for_circular_table_l2322_232299


namespace NUMINAMATH_GPT_compare_cubics_l2322_232225

variable {a b : ℝ}

theorem compare_cubics (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_GPT_compare_cubics_l2322_232225


namespace NUMINAMATH_GPT_contrapositive_inverse_converse_negation_false_l2322_232276

theorem contrapositive (a b : ℤ) : (a ≤ b) → (a - 2 ≤ b - 2) :=
sorry

theorem inverse (a b : ℤ) : (a - 2 ≤ b - 2) → (a ≤ b) :=
sorry

theorem converse (a b : ℤ) : (a - 2 > b - 2) → (a > b) :=
sorry

theorem negation_false (a b : ℤ) : ¬ ((a > b) → (a - 2 ≤ b - 2)) :=
sorry

end NUMINAMATH_GPT_contrapositive_inverse_converse_negation_false_l2322_232276


namespace NUMINAMATH_GPT_triangle_inequality_l2322_232200

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a / (b + c - a) + b / (c + a - b) + c / (a + b - c) ≥ 3 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l2322_232200


namespace NUMINAMATH_GPT_find_the_number_l2322_232263

-- Define the number we are trying to find
variable (x : ℝ)

-- Define the main condition from the problem
def main_condition : Prop := 0.7 * x - 40 = 30

-- Formalize the goal to prove
theorem find_the_number (h : main_condition x) : x = 100 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_the_number_l2322_232263


namespace NUMINAMATH_GPT_arithmetic_sequence_a4_possible_values_l2322_232237

theorem arithmetic_sequence_a4_possible_values (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∀ n, a (n + 1) = a n + d) 
  (h2 : a 1 * a 5 = 9)
  (h3 : a 2 = 3) : 
  a 4 = 3 ∨ a 4 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a4_possible_values_l2322_232237


namespace NUMINAMATH_GPT_min_m_n_l2322_232294

theorem min_m_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 108 * m = n^3) : m + n = 8 :=
sorry

end NUMINAMATH_GPT_min_m_n_l2322_232294


namespace NUMINAMATH_GPT_gcd_1080_920_is_40_l2322_232278

theorem gcd_1080_920_is_40 : Nat.gcd 1080 920 = 40 :=
by
  sorry

end NUMINAMATH_GPT_gcd_1080_920_is_40_l2322_232278


namespace NUMINAMATH_GPT_smallest_possible_recording_l2322_232287

theorem smallest_possible_recording :
  ∃ (A B C : ℤ), 
      (0 ≤ A ∧ A ≤ 10) ∧ 
      (0 ≤ B ∧ B ≤ 10) ∧ 
      (0 ≤ C ∧ C ≤ 10) ∧ 
      (A + B + C = 12) ∧ 
      (A + B + C) % 5 = 0 ∧ 
      A = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_recording_l2322_232287
