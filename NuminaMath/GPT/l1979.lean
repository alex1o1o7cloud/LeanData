import Mathlib

namespace NUMINAMATH_GPT_number_of_pounds_of_vegetables_l1979_197984

-- Defining the conditions
def beef_cost_per_pound : ℕ := 6  -- Beef costs $6 per pound
def vegetable_cost_per_pound : ℕ := 2  -- Vegetables cost $2 per pound
def beef_pounds : ℕ := 4  -- Troy buys 4 pounds of beef
def total_cost : ℕ := 36  -- The total cost of everything is $36

-- Prove the number of pounds of vegetables Troy buys is 6
theorem number_of_pounds_of_vegetables (V : ℕ) :
  beef_cost_per_pound * beef_pounds + vegetable_cost_per_pound * V = total_cost → V = 6 :=
by
  sorry  -- Proof to be filled in later

end NUMINAMATH_GPT_number_of_pounds_of_vegetables_l1979_197984


namespace NUMINAMATH_GPT_find_abc_sum_l1979_197910

theorem find_abc_sum (A B C : ℤ) (h : ∀ x : ℝ, x^3 + A * x^2 + B * x + C = (x + 1) * (x - 3) * (x - 4)) : A + B + C = 11 :=
by {
  -- This statement asserts that, given the conditions, the sum A + B + C equals 11
  sorry
}

end NUMINAMATH_GPT_find_abc_sum_l1979_197910


namespace NUMINAMATH_GPT_triangle_area_l1979_197918

theorem triangle_area (a b c : ℝ) (K : ℝ) (m n p : ℕ) (h1 : a = 10) (h2 : b = 12) (h3 : c = 15)
  (h4 : K = 240 * Real.sqrt 7 / 7)
  (h5 : Int.gcd m p = 1) -- m and p are relatively prime
  (h6 : n ≠ 1 ∧ ¬ (∃ x, x^2 ∣ n ∧ x > 1)) -- n is not divisible by the square of any prime
  : m + n + p = 254 := sorry

end NUMINAMATH_GPT_triangle_area_l1979_197918


namespace NUMINAMATH_GPT_new_batting_average_l1979_197938

def initial_runs (A : ℕ) := 16 * A
def additional_runs := 85
def increased_average := 3
def runs_in_5_innings := 100 + 120 + 45 + 75 + 65
def total_runs_17_innings (A : ℕ) := 17 * (A + increased_average)
def A : ℕ := 34
def total_runs_22_innings := total_runs_17_innings A + runs_in_5_innings
def number_of_innings := 22
def new_average := total_runs_22_innings / number_of_innings

theorem new_batting_average : new_average = 47 :=
by sorry

end NUMINAMATH_GPT_new_batting_average_l1979_197938


namespace NUMINAMATH_GPT_eval_custom_op_l1979_197997

def custom_op (a b : ℤ) : ℤ := 2 * b + 5 * a - a^2 - b

theorem eval_custom_op : custom_op 3 4 = 10 :=
by
  sorry

end NUMINAMATH_GPT_eval_custom_op_l1979_197997


namespace NUMINAMATH_GPT_part1_part2_l1979_197930

def f (x a : ℝ) : ℝ := |x - a| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x 2 > 5 ↔ x < - 1 / 3 ∨ x > 3 :=
by sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ |a - 2|) → a ≤ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1979_197930


namespace NUMINAMATH_GPT_fraction_irreducible_l1979_197920

theorem fraction_irreducible (n : ℕ) (hn : 0 < n) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
by sorry

end NUMINAMATH_GPT_fraction_irreducible_l1979_197920


namespace NUMINAMATH_GPT_find_radius_l1979_197947

-- Defining the conditions as given in the math problem
def sectorArea (r : ℝ) (L : ℝ) : ℝ := 0.5 * r * L

theorem find_radius (h1 : sectorArea r 5.5 = 13.75) : r = 5 :=
by sorry

end NUMINAMATH_GPT_find_radius_l1979_197947


namespace NUMINAMATH_GPT_PQR_product_l1979_197998

def PQR_condition (P Q R S : ℕ) : Prop :=
  P + Q + R + S = 100 ∧
  ∃ x : ℕ, P = x - 4 ∧ Q = x + 4 ∧ R = x / 4 ∧ S = 4 * x

theorem PQR_product (P Q R S : ℕ) (h : PQR_condition P Q R S) : P * Q * R * S = 61440 :=
by 
  sorry

end NUMINAMATH_GPT_PQR_product_l1979_197998


namespace NUMINAMATH_GPT_reflect_curve_maps_onto_itself_l1979_197915

theorem reflect_curve_maps_onto_itself (a b c : ℝ) :
    ∃ (x0 y0 : ℝ), 
    x0 = -a / 3 ∧ 
    y0 = 2 * a^3 / 27 - a * b / 3 + c ∧
    ∀ x y x' y', 
    y = x^3 + a * x^2 + b * x + c → 
    x' = 2 * x0 - x → 
    y' = 2 * y0 - y → 
    y' = x'^3 + a * x'^2 + b * x' + c := 
    by sorry

end NUMINAMATH_GPT_reflect_curve_maps_onto_itself_l1979_197915


namespace NUMINAMATH_GPT_factor_expression_l1979_197944

theorem factor_expression (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1979_197944


namespace NUMINAMATH_GPT_fill_bathtub_time_l1979_197914

theorem fill_bathtub_time
  (r_cold : ℚ := 1/10)
  (r_hot : ℚ := 1/15)
  (r_empty : ℚ := -1/12)
  (net_rate : ℚ := r_cold + r_hot + r_empty) :
  net_rate = 1/12 → 
  t = 12 :=
by
  sorry

end NUMINAMATH_GPT_fill_bathtub_time_l1979_197914


namespace NUMINAMATH_GPT_play_role_assignments_l1979_197919

def specific_role_assignments (men women remaining either_gender_roles : ℕ) : ℕ :=
  men * women * Nat.choose remaining either_gender_roles

theorem play_role_assignments :
  specific_role_assignments 6 7 11 4 = 13860 := by
  -- The given problem statement implies evaluating the specific role assignments
  sorry

end NUMINAMATH_GPT_play_role_assignments_l1979_197919


namespace NUMINAMATH_GPT_boat_downstream_distance_l1979_197977

theorem boat_downstream_distance 
  (Vb Vr T D U : ℝ)
  (h1 : Vb + Vr = 21)
  (h2 : Vb - Vr = 12)
  (h3 : U = 48)
  (h4 : T = 4)
  (h5 : D = 20) :
  (Vb + Vr) * D = 420 :=
by
  sorry

end NUMINAMATH_GPT_boat_downstream_distance_l1979_197977


namespace NUMINAMATH_GPT_expression_equals_33_l1979_197968

noncomputable def calculate_expression : ℚ :=
  let part1 := 25 * 52
  let part2 := 46 * 15
  let diff := part1 - part2
  (2013 / diff) * 10

theorem expression_equals_33 : calculate_expression = 33 := sorry

end NUMINAMATH_GPT_expression_equals_33_l1979_197968


namespace NUMINAMATH_GPT_convex_quadrilateral_max_two_obtuse_l1979_197983

theorem convex_quadrilateral_max_two_obtuse (a b c d : ℝ)
  (h1 : a + b + c + d = 360)
  (h2 : a < 180) (h3 : b < 180) (h4 : c < 180) (h5 : d < 180)
  : (∃ A1 A2, a = A1 ∧ b = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ c < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ d < 90) ∨
    (∃ A1 A2, a = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ b < 90 ∧ c < 90) ∨
    (∃ A1 A2, b = A1 ∧ c = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ d < 90) ∨
    (∃ A1 A2, b = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ c < 90) ∨
    (∃ A1 A2, c = A1 ∧ d = A2 ∧ A1 > 90 ∧ A2 > 90 ∧ a < 90 ∧ b < 90) ∨
    (¬∃ x y z, (x > 90) ∧ (y > 90) ∧ (z > 90) ∧ x + y + z ≤ 360) := sorry

end NUMINAMATH_GPT_convex_quadrilateral_max_two_obtuse_l1979_197983


namespace NUMINAMATH_GPT_polygon_sides_l1979_197900

theorem polygon_sides (interior_angle: ℝ) (sum_exterior_angles: ℝ) (n: ℕ) (h: interior_angle = 108) (h1: sum_exterior_angles = 360): n = 5 :=
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_l1979_197900


namespace NUMINAMATH_GPT_find_a_5_l1979_197963

theorem find_a_5 (a : ℕ → ℤ) (h₁ : ∀ n : ℕ, n > 0 → a (n + 1) = a n - 1)
  (h₂ : a 2 + a 4 + a 6 = 18) : a 5 = 5 := 
sorry

end NUMINAMATH_GPT_find_a_5_l1979_197963


namespace NUMINAMATH_GPT_Kishore_misc_expense_l1979_197902

theorem Kishore_misc_expense:
  let savings := 2400
  let percent_saved := 0.10
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let total_salary := savings / percent_saved 
  let total_spent := rent + milk + groceries + education + petrol
  total_salary - (total_spent + savings) = 6100 := 
by
  sorry

end NUMINAMATH_GPT_Kishore_misc_expense_l1979_197902


namespace NUMINAMATH_GPT_part_a_part_b_l1979_197950

noncomputable def volume_of_prism (V : ℝ) : ℝ :=
  (9 / 250) * V

noncomputable def max_volume_of_prism (V : ℝ) : ℝ :=
  (1 / 12) * V

theorem part_a (V : ℝ) :
  volume_of_prism V = (9 / 250) * V :=
  by sorry

theorem part_b (V : ℝ) :
  max_volume_of_prism V = (1 / 12) * V :=
  by sorry

end NUMINAMATH_GPT_part_a_part_b_l1979_197950


namespace NUMINAMATH_GPT_sum_of_m_and_n_l1979_197954

noncomputable section

variable {a b m n : ℕ}

theorem sum_of_m_and_n 
  (h1 : a = n * b)
  (h2 : (a + b) = m * (a - b)) :
  m + n = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_m_and_n_l1979_197954


namespace NUMINAMATH_GPT_value_of_x_m_minus_n_l1979_197913

variables {x : ℝ} {m n : ℝ}

theorem value_of_x_m_minus_n (hx_m : x^m = 6) (hx_n : x^n = 3) : x^(m - n) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_x_m_minus_n_l1979_197913


namespace NUMINAMATH_GPT_meena_cookies_left_l1979_197912

def dozen : ℕ := 12

def baked_cookies : ℕ := 5 * dozen
def mr_stone_buys : ℕ := 2 * dozen
def brock_buys : ℕ := 7
def katy_buys : ℕ := 2 * brock_buys
def total_sold : ℕ := mr_stone_buys + brock_buys + katy_buys
def cookies_left : ℕ := baked_cookies - total_sold

theorem meena_cookies_left : cookies_left = 15 := by
  sorry

end NUMINAMATH_GPT_meena_cookies_left_l1979_197912


namespace NUMINAMATH_GPT_cylinder_height_same_volume_as_cone_l1979_197939

theorem cylinder_height_same_volume_as_cone
    (r_cone : ℝ) (h_cone : ℝ) (r_cylinder : ℝ) (V : ℝ)
    (h_volume_cone_eq : V = (1 / 3) * Real.pi * r_cone ^ 2 * h_cone)
    (r_cone_val : r_cone = 2)
    (h_cone_val : h_cone = 6)
    (r_cylinder_val : r_cylinder = 1) :
    ∃ h_cylinder : ℝ, (V = Real.pi * r_cylinder ^ 2 * h_cylinder) ∧ h_cylinder = 8 :=
by
  -- Here you would provide the proof for the theorem.
  sorry

end NUMINAMATH_GPT_cylinder_height_same_volume_as_cone_l1979_197939


namespace NUMINAMATH_GPT_orthocenter_of_triangle_l1979_197994

theorem orthocenter_of_triangle (A : ℝ × ℝ) (x y : ℝ) 
  (h₁ : x + y = 0) (h₂ : 2 * x - 3 * y + 1 = 0) : 
  A = (1, 2) → (x, y) = (-1 / 5, 1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_orthocenter_of_triangle_l1979_197994


namespace NUMINAMATH_GPT_total_surface_area_l1979_197911

theorem total_surface_area (a b c : ℝ)
    (h1 : a + b + c = 40)
    (h2 : a^2 + b^2 + c^2 = 625)
    (h3 : a * b * c = 600) : 
    2 * (a * b + b * c + c * a) = 975 :=
by
  sorry

end NUMINAMATH_GPT_total_surface_area_l1979_197911


namespace NUMINAMATH_GPT_pos_int_solutions_3x_2y_841_l1979_197961

theorem pos_int_solutions_3x_2y_841 :
  {n : ℕ // ∃ (x y : ℕ), 3 * x + 2 * y = 841 ∧ x > 0 ∧ y > 0} =
  {n : ℕ // n = 140} := 
sorry

end NUMINAMATH_GPT_pos_int_solutions_3x_2y_841_l1979_197961


namespace NUMINAMATH_GPT_foma_should_give_ierema_55_coins_l1979_197905

theorem foma_should_give_ierema_55_coins (F E Y : ℤ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)
  (h3 : E + 70 = Y) :
  F - E = 110 → F - E = 55 :=
by
  sorry

end NUMINAMATH_GPT_foma_should_give_ierema_55_coins_l1979_197905


namespace NUMINAMATH_GPT_construction_rates_construction_cost_l1979_197932

-- Defining the conditions as Lean hypotheses

def length := 1650
def diff_rate := 30
def time_ratio := 3/2

-- Daily construction rates (questions answered as hypotheses as well)
def daily_rate_A := 60
def daily_rate_B := 90

-- Additional conditions for cost calculations
def cost_A_per_day := 90000
def cost_B_per_day := 120000
def total_days := 14
def alone_days_A := 5

-- Problem stated as proofs to be completed
theorem construction_rates :
  (∀ (x : ℕ), x = daily_rate_A ∧ (x + diff_rate) = daily_rate_B ∧ 
  (1650 / (x + diff_rate)) * (3/2) = (1650 / x) → 
  60 = daily_rate_A ∧ (60 + 30) = daily_rate_B ) :=
by sorry

theorem construction_cost :
  (∀ (m : ℕ), m = alone_days_A ∧ 
  (cost_A_per_day * total_days + cost_B_per_day * (total_days - alone_days_A)) / 1000 = 2340) :=
by sorry

end NUMINAMATH_GPT_construction_rates_construction_cost_l1979_197932


namespace NUMINAMATH_GPT_linear_inequalities_solution_range_l1979_197946

theorem linear_inequalities_solution_range (m : ℝ) :
  (∃ x : ℝ, x - 2 * m < 0 ∧ x + m > 2) ↔ m > 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_linear_inequalities_solution_range_l1979_197946


namespace NUMINAMATH_GPT_solve_equation_l1979_197956

theorem solve_equation (x : ℝ) (h : 16 * x^2 = 81) : x = 9 / 4 ∨ x = - (9 / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1979_197956


namespace NUMINAMATH_GPT_minimize_expression_l1979_197958

theorem minimize_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 30) :
  (a, b) = (15 / 4, 15) ↔ (∀ x y : ℝ, 0 < x → 0 < y → (4 * x + y = 30) → (1 / x + 4 / y) ≥ (1 / (15 / 4) + 4 / 15)) := by
sorry

end NUMINAMATH_GPT_minimize_expression_l1979_197958


namespace NUMINAMATH_GPT_total_cost_john_paid_l1979_197989

theorem total_cost_john_paid 
  (meters_of_cloth : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ)
  (h1 : meters_of_cloth = 9.25)
  (h2 : cost_per_meter = 48)
  (h3 : total_cost = meters_of_cloth * cost_per_meter) :
  total_cost = 444 :=
sorry

end NUMINAMATH_GPT_total_cost_john_paid_l1979_197989


namespace NUMINAMATH_GPT_chairs_to_remove_l1979_197935

-- Defining the conditions
def chairs_per_row : Nat := 15
def total_chairs : Nat := 180
def expected_attendees : Nat := 125

-- Main statement to prove
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ) : 
  chairs_per_row = 15 → 
  total_chairs = 180 → 
  expected_attendees = 125 → 
  ∃ n, total_chairs - (chairs_per_row * n) = 45 ∧ n * chairs_per_row ≥ expected_attendees := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_chairs_to_remove_l1979_197935


namespace NUMINAMATH_GPT_expression_divisible_by_41_l1979_197917

theorem expression_divisible_by_41 (n : ℕ) : 41 ∣ (5 * 7^(2*(n+1)) + 2^(3*n)) :=
  sorry

end NUMINAMATH_GPT_expression_divisible_by_41_l1979_197917


namespace NUMINAMATH_GPT_sufficient_condition_for_inequality_l1979_197995

theorem sufficient_condition_for_inequality (a b : ℝ) (h_nonzero : a * b ≠ 0) : (a < b ∧ b < 0) → (1 / a ^ 2 > 1 / b ^ 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_inequality_l1979_197995


namespace NUMINAMATH_GPT_compute_expression_l1979_197975

theorem compute_expression :
  45 * 72 + 28 * 45 = 4500 :=
  sorry

end NUMINAMATH_GPT_compute_expression_l1979_197975


namespace NUMINAMATH_GPT_larger_number_is_23_l1979_197992

-- Definitions for the two conditions
variables (x y : ℝ)

-- The conditions given in the problem
def sum_condition (x y : ℝ) : Prop := x + y = 40
def difference_condition (x y : ℝ) : Prop := x - y = 6

-- The proof statement
theorem larger_number_is_23 (x y : ℝ) (h1 : sum_condition x y) (h2 : difference_condition x y) : x = 23 :=
by
  sorry

end NUMINAMATH_GPT_larger_number_is_23_l1979_197992


namespace NUMINAMATH_GPT_midterm_exam_2022_option_probabilities_l1979_197933

theorem midterm_exam_2022_option_probabilities :
  let no_option := 4
  let prob_distribution := (1 : ℚ) / 3
  let combs_with_4_correct := 1
  let combs_with_3_correct := 4
  let combs_with_2_correct := 6
  let prob_4_correct := prob_distribution
  let prob_3_correct := prob_distribution / combs_with_3_correct
  let prob_2_correct := prob_distribution / combs_with_2_correct
  
  let prob_B_correct := combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct
  let prob_C_given_event_A := combs_with_3_correct * prob_3_correct / (combs_with_2_correct * prob_2_correct + combs_with_3_correct * prob_3_correct + prob_4_correct)
  
  (prob_B_correct > 1 / 2) ∧ (prob_C_given_event_A = 1 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_midterm_exam_2022_option_probabilities_l1979_197933


namespace NUMINAMATH_GPT_largest_n_for_inequality_l1979_197964

theorem largest_n_for_inequality :
  ∃ n : ℕ, 3 * n^2007 < 3^4015 ∧ ∀ m : ℕ, 3 * m^2007 < 3^4015 → m ≤ 8 ∧ n = 8 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_for_inequality_l1979_197964


namespace NUMINAMATH_GPT_simplification_qrt_1_simplification_qrt_2_l1979_197922

-- Problem 1
theorem simplification_qrt_1 : (2 * Real.sqrt 12 + 3 * Real.sqrt 3 - Real.sqrt 27) = 4 * Real.sqrt 3 :=
by
  sorry

-- Problem 2
theorem simplification_qrt_2 : (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2 * 12) + Real.sqrt 24) = 4 + Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_simplification_qrt_1_simplification_qrt_2_l1979_197922


namespace NUMINAMATH_GPT_trajectory_of_point_inside_square_is_conic_or_degenerates_l1979_197907

noncomputable def is_conic_section (a : ℝ) (P : ℝ × ℝ) : Prop :=
  ∃ (m n l : ℝ) (x y : ℝ), 
    x = P.1 ∧ y = P.2 ∧ 
    (m^2 + n^2) * x^2 - 2 * n * (l + m) * x * y + (l^2 + n^2) * y^2 = (l * m - n^2)^2 ∧
    4 * n^2 * (l + m)^2 - 4 * (m^2 + n^2) * (l^2 + n^2) ≤ 0

theorem trajectory_of_point_inside_square_is_conic_or_degenerates
  (a : ℝ) (P : ℝ × ℝ)
  (h1 : 0 < P.1) (h2 : P.1 < 2 * a)
  (h3 : 0 < P.2) (h4 : P.2 < 2 * a)
  : is_conic_section a P :=
sorry

end NUMINAMATH_GPT_trajectory_of_point_inside_square_is_conic_or_degenerates_l1979_197907


namespace NUMINAMATH_GPT_hot_drink_sales_l1979_197978

theorem hot_drink_sales (x y : ℝ) (h : y = -2.35 * x + 147.7) (hx : x = 2) : y = 143 := 
by sorry

end NUMINAMATH_GPT_hot_drink_sales_l1979_197978


namespace NUMINAMATH_GPT_arithmetic_sequences_count_l1979_197921

noncomputable def countArithmeticSequences (n : ℕ) : ℕ :=
  if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4

theorem arithmetic_sequences_count :
  ∀ n : ℕ, countArithmeticSequences n = if n % 2 = 0 then (n^2) / 4 else (n^2 - 1) / 4 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequences_count_l1979_197921


namespace NUMINAMATH_GPT_smallest_number_of_cubes_l1979_197901

noncomputable def container_cubes (length_ft : ℕ) (height_ft : ℕ) (width_ft : ℕ) (prime_inch : ℕ) : ℕ :=
  let length_inch := length_ft * 12
  let height_inch := height_ft * 12
  let width_inch := width_ft * 12
  (length_inch / prime_inch) * (height_inch / prime_inch) * (width_inch / prime_inch)

theorem smallest_number_of_cubes :
  container_cubes 60 24 30 3 = 2764800 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_cubes_l1979_197901


namespace NUMINAMATH_GPT_inclination_angle_l1979_197949

theorem inclination_angle (θ : ℝ) (h : 0 ≤ θ ∧ θ < 180) :
  (∀ x y : ℝ, x - y + 3 = 0 → θ = 45) :=
sorry

end NUMINAMATH_GPT_inclination_angle_l1979_197949


namespace NUMINAMATH_GPT_scientific_notation_example_l1979_197937

theorem scientific_notation_example : (8485000 : ℝ) = 8.485 * 10 ^ 6 := 
by 
  sorry

end NUMINAMATH_GPT_scientific_notation_example_l1979_197937


namespace NUMINAMATH_GPT_drone_height_l1979_197980

theorem drone_height (r s h : ℝ) 
  (h_distance_RS : r^2 + s^2 = 160^2)
  (h_DR : h^2 + r^2 = 170^2) 
  (h_DS : h^2 + s^2 = 150^2) : 
  h = 30 * Real.sqrt 43 :=
by 
  sorry

end NUMINAMATH_GPT_drone_height_l1979_197980


namespace NUMINAMATH_GPT_min_inquiries_for_parity_l1979_197959

-- Define the variables and predicates
variables (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n)

-- Define the main theorem we need to prove
theorem min_inquiries_for_parity (m n : ℕ) (h_m : m > 2) (h_n : n > 2) (h_meven : Even m) (h_neven : Even n) : 
  ∃ k, (k = m + n - 4) := 
sorry

end NUMINAMATH_GPT_min_inquiries_for_parity_l1979_197959


namespace NUMINAMATH_GPT_candies_eaten_l1979_197934

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end NUMINAMATH_GPT_candies_eaten_l1979_197934


namespace NUMINAMATH_GPT_incorrect_statement_B_l1979_197996

-- Define the plane vector operation "☉".
def vector_operation (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

-- Define the mathematical problem based on the given conditions.
theorem incorrect_statement_B (a b : ℝ × ℝ) : vector_operation a b ≠ vector_operation b a := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_B_l1979_197996


namespace NUMINAMATH_GPT_income_recording_l1979_197955

theorem income_recording (exp_200 : Int := -200) (income_60 : Int := 60) : exp_200 = -200 → income_60 = 60 →
  (income_60 > 0) :=
by
  intro h_exp h_income
  sorry

end NUMINAMATH_GPT_income_recording_l1979_197955


namespace NUMINAMATH_GPT_range_of_a_for_propositions_p_and_q_l1979_197924

theorem range_of_a_for_propositions_p_and_q :
  {a : ℝ | ∃ x, (x^2 + 2 * a * x + 4 = 0) ∧ (3 - 2 * a > 1)} = {a | a ≤ -2} := sorry

end NUMINAMATH_GPT_range_of_a_for_propositions_p_and_q_l1979_197924


namespace NUMINAMATH_GPT_red_crayons_count_l1979_197986

variable (R : ℕ) -- Number of red crayons
variable (B : ℕ) -- Number of blue crayons
variable (Y : ℕ) -- Number of yellow crayons

-- Conditions
axiom h1 : B = R + 5
axiom h2 : Y = 2 * B - 6
axiom h3 : Y = 32

-- Statement to prove
theorem red_crayons_count : R = 14 :=
by
  sorry

end NUMINAMATH_GPT_red_crayons_count_l1979_197986


namespace NUMINAMATH_GPT_exists_real_A_l1979_197987

theorem exists_real_A (t : ℝ) (n : ℕ) (h_root: t^2 - 10 * t + 1 = 0) :
  ∃ A : ℝ, (A = t) ∧ ∀ n : ℕ, ∃ k : ℕ, A^n + 1/(A^n) - k^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_exists_real_A_l1979_197987


namespace NUMINAMATH_GPT_net_income_calculation_l1979_197940

-- Definitions based on conditions
def rent_per_hour := 20
def monday_hours := 8
def wednesday_hours := 8
def friday_hours := 6
def sunday_hours := 5
def maintenance_cost := 35
def insurance_fee := 15
def rental_days := 4

-- Derived values based on conditions
def total_income_per_week :=
  (monday_hours + wednesday_hours) * rent_per_hour * 2 + 
  friday_hours * rent_per_hour + 
  sunday_hours * rent_per_hour

def total_expenses_per_week :=
  maintenance_cost + 
  insurance_fee * rental_days

def net_income_per_week := 
  total_income_per_week - total_expenses_per_week

-- The final proof statement
theorem net_income_calculation : net_income_per_week = 445 := by
  sorry

end NUMINAMATH_GPT_net_income_calculation_l1979_197940


namespace NUMINAMATH_GPT_anna_age_l1979_197965

-- Define the conditions as given in the problem
variable (x : ℕ)
variable (m n : ℕ)

-- Translate the problem statement into Lean
axiom perfect_square_condition : x - 4 = m^2
axiom perfect_cube_condition : x + 3 = n^3

-- The proof problem statement in Lean 4
theorem anna_age : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_anna_age_l1979_197965


namespace NUMINAMATH_GPT_range_of_g_l1979_197993

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arccos x)^2 - (Real.arcsin x)^2

theorem range_of_g :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → -((Real.pi^2) / 4) ≤ g x ∧ g x ≤ (3 * (Real.pi^2)) / 4 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_range_of_g_l1979_197993


namespace NUMINAMATH_GPT_tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l1979_197929

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + x * Real.exp x - Real.exp 1

-- Part (Ⅰ)
theorem tangent_line_at_one (h_a : a = 0) : ∃ m b : ℝ, ∀ x : ℝ, 2 * Real.exp 1 * x - y - 2 * Real.exp 1 = 0 := sorry

-- Part (Ⅱ)
theorem unique_zero_of_f (h_a : a > 0) : ∃! t : ℝ, f a t = 0 := sorry

-- Part (Ⅲ)
theorem exists_lower_bound_of_f (h_a : a < 0) : ∃ m : ℝ, ∀ x : ℝ, f a x ≥ m := sorry

end NUMINAMATH_GPT_tangent_line_at_one_unique_zero_of_f_exists_lower_bound_of_f_l1979_197929


namespace NUMINAMATH_GPT_twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l1979_197999

theorem twenty_five_percent_less_than_80_is_twenty_five_percent_more_of (n : ℝ) (h : 1.25 * n = 80 - 0.25 * 80) : n = 48 :=
by
  sorry

end NUMINAMATH_GPT_twenty_five_percent_less_than_80_is_twenty_five_percent_more_of_l1979_197999


namespace NUMINAMATH_GPT_dealer_gross_profit_l1979_197906

theorem dealer_gross_profit
  (purchase_price : ℝ)
  (markup_rate : ℝ)
  (discount_rate : ℝ)
  (initial_selling_price : ℝ)
  (final_selling_price : ℝ)
  (gross_profit : ℝ)
  (h0 : purchase_price = 150)
  (h1 : markup_rate = 0.5)
  (h2 : discount_rate = 0.2)
  (h3 : initial_selling_price = purchase_price + markup_rate * initial_selling_price)
  (h4 : final_selling_price = initial_selling_price - discount_rate * initial_selling_price)
  (h5 : gross_profit = final_selling_price - purchase_price) :
  gross_profit = 90 :=
sorry

end NUMINAMATH_GPT_dealer_gross_profit_l1979_197906


namespace NUMINAMATH_GPT_parents_years_in_america_before_aziz_birth_l1979_197927

noncomputable def aziz_birth_year (current_year : ℕ) (aziz_age : ℕ) : ℕ :=
  current_year - aziz_age

noncomputable def years_parents_in_america_before_aziz_birth (arrival_year : ℕ) (aziz_birth_year : ℕ) : ℕ :=
  aziz_birth_year - arrival_year

theorem parents_years_in_america_before_aziz_birth 
  (current_year : ℕ := 2021) 
  (aziz_age : ℕ := 36) 
  (arrival_year : ℕ := 1982) 
  (expected_years : ℕ := 3) :
  years_parents_in_america_before_aziz_birth arrival_year (aziz_birth_year current_year aziz_age) = expected_years :=
by 
  sorry

end NUMINAMATH_GPT_parents_years_in_america_before_aziz_birth_l1979_197927


namespace NUMINAMATH_GPT_quadratic_roots_distinct_real_l1979_197928

theorem quadratic_roots_distinct_real (a b c : ℝ) (h : a = 1 ∧ b = -2 ∧ c = 0)
    (Δ : ℝ := b^2 - 4 * a * c) (hΔ : Δ > 0) :
    (∀ r1 r2 : ℝ, r1 ≠ r2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_distinct_real_l1979_197928


namespace NUMINAMATH_GPT_perpendicular_slope_of_line_l1979_197909

theorem perpendicular_slope_of_line (x y : ℤ) : 
    (5 * x - 4 * y = 20) → 
    ∃ m : ℚ, m = -4 / 5 := 
by 
    sorry

end NUMINAMATH_GPT_perpendicular_slope_of_line_l1979_197909


namespace NUMINAMATH_GPT_supplement_of_double_complement_l1979_197971

def angle : ℝ := 30

def complement (θ : ℝ) : ℝ :=
  90 - θ

def double_complement (θ : ℝ) : ℝ :=
  2 * (complement θ)

def supplement (θ : ℝ) : ℝ :=
  180 - θ

theorem supplement_of_double_complement (θ : ℝ) (h : θ = angle) : supplement (double_complement θ) = 60 :=
by
  sorry

end NUMINAMATH_GPT_supplement_of_double_complement_l1979_197971


namespace NUMINAMATH_GPT_measure_six_pints_l1979_197926
-- Importing the necessary library

-- Defining the problem conditions
def total_wine : ℕ := 12
def capacity_8_pint_vessel : ℕ := 8
def capacity_5_pint_vessel : ℕ := 5

-- The problem to prove: it is possible to measure 6 pints into the 8-pint container
theorem measure_six_pints :
  ∃ (n : ℕ), n = 6 ∧ n ≤ capacity_8_pint_vessel := 
sorry

end NUMINAMATH_GPT_measure_six_pints_l1979_197926


namespace NUMINAMATH_GPT_evaluate_expression_l1979_197982

variable (x y : ℚ)

theorem evaluate_expression 
  (hx : x = 2) 
  (hy : y = -1 / 5) : 
  (2 * x - 3)^2 - (x + 2 * y) * (x - 2 * y) - 3 * y^2 + 3 = 1 / 25 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1979_197982


namespace NUMINAMATH_GPT_different_lists_count_l1979_197916

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end NUMINAMATH_GPT_different_lists_count_l1979_197916


namespace NUMINAMATH_GPT_nine_div_one_plus_four_div_x_eq_one_l1979_197903

theorem nine_div_one_plus_four_div_x_eq_one (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end NUMINAMATH_GPT_nine_div_one_plus_four_div_x_eq_one_l1979_197903


namespace NUMINAMATH_GPT_f_iterated_result_l1979_197943

def f (x : ℕ) : ℕ :=
  if Even x then 3 * x / 2 else 2 * x + 1

theorem f_iterated_result : f (f (f (f 1))) = 31 := by
  sorry

end NUMINAMATH_GPT_f_iterated_result_l1979_197943


namespace NUMINAMATH_GPT_bakery_combinations_l1979_197948

theorem bakery_combinations 
  (total_breads : ℕ) (bread_types : Finset ℕ) (purchases : Finset ℕ)
  (h_total : total_breads = 8)
  (h_bread_types : bread_types.card = 5)
  (h_purchases : purchases.card = 2) : 
  ∃ (combinations : ℕ), combinations = 70 := 
sorry

end NUMINAMATH_GPT_bakery_combinations_l1979_197948


namespace NUMINAMATH_GPT_number_of_extremum_points_of_f_l1979_197967

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (x + 1)^3 * Real.exp (x + 1) else (-(x + 1))^3 * Real.exp (-(x + 1))

theorem number_of_extremum_points_of_f :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ((f (x1 - epsilon) < f x1 ∧ f x1 > f (x1 + epsilon)) ∨ (f (x1 - epsilon) > f x1 ∧ f x1 < f (x1 + epsilon))) ∧
    ((f (x2 - epsilon) < f x2 ∧ f x2 > f (x2 + epsilon)) ∨ (f (x2 - epsilon) > f x2 ∧ f x2 < f (x2 + epsilon))) ∧
    ((f (x3 - epsilon) < f x3 ∧ f x3 > f (x3 + epsilon)) ∨ (f (x3 - epsilon) > f x3 ∧ f x3 < f (x3 + epsilon)))) :=
sorry

end NUMINAMATH_GPT_number_of_extremum_points_of_f_l1979_197967


namespace NUMINAMATH_GPT_external_angle_at_C_l1979_197953

-- Definitions based on conditions
def angleA : ℝ := 40
def B := 2 * angleA
def sum_of_angles_in_triangle (A B C : ℝ) : Prop := A + B + C = 180
def external_angle (C : ℝ) : ℝ := 180 - C

-- Theorem statement
theorem external_angle_at_C :
  ∃ C : ℝ, sum_of_angles_in_triangle angleA B C ∧ external_angle C = 120 :=
sorry

end NUMINAMATH_GPT_external_angle_at_C_l1979_197953


namespace NUMINAMATH_GPT_rational_expression_simplification_l1979_197972

theorem rational_expression_simplification
  (a b c : ℚ) 
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( ((a^2 * b^2) / c^2 - (2 / c) + (1 / (a^2 * b^2)) + (2 * a * b) / c^2 - (2 / (a * b * c))) 
      / ((2 / (a * b)) - (2 * a * b) / c) ) 
      / (101 / c) = - (1 / 202) :=
by sorry

end NUMINAMATH_GPT_rational_expression_simplification_l1979_197972


namespace NUMINAMATH_GPT_max_value_2ac_minus_abc_l1979_197991

theorem max_value_2ac_minus_abc (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 7) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c <= 4) : 
  2 * a * c - a * b * c ≤ 28 :=
sorry

end NUMINAMATH_GPT_max_value_2ac_minus_abc_l1979_197991


namespace NUMINAMATH_GPT_michael_age_multiple_l1979_197908

theorem michael_age_multiple (M Y O k : ℤ) (hY : Y = 5) (hO : O = 3 * Y) (h_combined : M + O + Y = 28) (h_relation : O = k * (M - 1) + 1) : k = 2 :=
by
  -- Definitions and given conditions are provided:
  have hY : Y = 5 := hY
  have hO : O = 3 * Y := hO
  have h_combined : M + O + Y = 28 := h_combined
  have h_relation : O = k * (M - 1) + 1 := h_relation
  
  -- Begin the proof by using the provided conditions
  sorry

end NUMINAMATH_GPT_michael_age_multiple_l1979_197908


namespace NUMINAMATH_GPT_geometric_sequence_a_sequence_b_l1979_197976

theorem geometric_sequence_a (a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60) :
  ∀ n, a n = 4 * 3^(n - 1) :=
sorry

theorem sequence_b (b a : ℕ → ℤ) (h1 : a 1 = 4) (h2 : 2 * a 2 + a 3 = 60)
  (h3 : ∀ n, b (n + 1) = b n + a n) (h4 : b 1 = a 2) :
  ∀ n, b n = 2 * 3^n + 10 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a_sequence_b_l1979_197976


namespace NUMINAMATH_GPT_fred_washing_cars_l1979_197936

theorem fred_washing_cars :
  ∀ (initial_amount final_amount money_made : ℕ),
  initial_amount = 23 →
  final_amount = 86 →
  money_made = final_amount - initial_amount →
  money_made = 63 := by
    intros initial_amount final_amount money_made h_initial h_final h_calc
    rw [h_initial, h_final] at h_calc
    exact h_calc

end NUMINAMATH_GPT_fred_washing_cars_l1979_197936


namespace NUMINAMATH_GPT_tom_age_ratio_l1979_197966

-- Define the constants T and N with the given conditions
variables (T N : ℕ)
-- Tom's age T years, sum of three children's ages is also T
-- N years ago, Tom's age was three times the sum of children's ages then

-- We need to prove that T / N = 4 under these conditions
theorem tom_age_ratio (h1 : T = 3 * T - 8 * N) : T / N = 4 :=
sorry

end NUMINAMATH_GPT_tom_age_ratio_l1979_197966


namespace NUMINAMATH_GPT_sum_of_decimals_l1979_197974

theorem sum_of_decimals :
  5.467 + 2.349 + 3.785 = 11.751 :=
sorry

end NUMINAMATH_GPT_sum_of_decimals_l1979_197974


namespace NUMINAMATH_GPT_janet_initial_stickers_l1979_197985

variable (x : ℕ)

theorem janet_initial_stickers (h : x + 53 = 56) : x = 3 := by
  sorry

end NUMINAMATH_GPT_janet_initial_stickers_l1979_197985


namespace NUMINAMATH_GPT_packs_needed_l1979_197931

-- Define the problem conditions
def bulbs_bedroom : ℕ := 2
def bulbs_bathroom : ℕ := 1
def bulbs_kitchen : ℕ := 1
def bulbs_basement : ℕ := 4
def bulbs_pack : ℕ := 2

def total_bulbs_main_areas : ℕ := bulbs_bedroom + bulbs_bathroom + bulbs_kitchen + bulbs_basement
def bulbs_garage : ℕ := total_bulbs_main_areas / 2

def total_bulbs : ℕ := total_bulbs_main_areas + bulbs_garage

def total_packs : ℕ := total_bulbs / bulbs_pack

-- The proof statement
theorem packs_needed : total_packs = 6 :=
by
  sorry

end NUMINAMATH_GPT_packs_needed_l1979_197931


namespace NUMINAMATH_GPT_max_unsuccessful_attempts_l1979_197951

theorem max_unsuccessful_attempts (n_rings letters_per_ring : ℕ) (h_rings : n_rings = 3) (h_letters : letters_per_ring = 6) : 
  (letters_per_ring ^ n_rings) - 1 = 215 := 
by 
  -- conditions
  rw [h_rings, h_letters]
  -- necessary imports and proof generation
  sorry

end NUMINAMATH_GPT_max_unsuccessful_attempts_l1979_197951


namespace NUMINAMATH_GPT_distance_between_chords_l1979_197970

-- Definitions based on the conditions
structure CircleGeometry where
  radius: ℝ
  d1: ℝ -- distance from the center to the closest chord (34 units)
  d2: ℝ -- distance from the center to the second chord (38 units)
  d3: ℝ -- distance from the center to the outermost chord (38 units)

-- The problem itself
theorem distance_between_chords (circle: CircleGeometry) (h1: circle.d2 = 3) (h2: circle.d1 = 3 * circle.d2) (h3: circle.d3 = circle.d2) :
  2 * circle.d2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_chords_l1979_197970


namespace NUMINAMATH_GPT_fixed_fee_1430_l1979_197945

def fixed_monthly_fee (f p : ℝ) : Prop :=
  f + p = 20.60 ∧ f + 3 * p = 33.20

theorem fixed_fee_1430 (f p: ℝ) (h : fixed_monthly_fee f p) : 
  f = 14.30 :=
by
  sorry

end NUMINAMATH_GPT_fixed_fee_1430_l1979_197945


namespace NUMINAMATH_GPT_SmartMart_science_kits_l1979_197969

theorem SmartMart_science_kits (sc pz : ℕ) (h1 : pz = sc - 9) (h2 : pz = 36) : sc = 45 := by
  sorry

end NUMINAMATH_GPT_SmartMart_science_kits_l1979_197969


namespace NUMINAMATH_GPT_kite_area_correct_l1979_197981

open Real

structure Point where
  x : ℝ
  y : ℝ

def Kite (p1 p2 p3 p4 : Point) : Prop :=
  let triangle_area (a b c : Point) : ℝ :=
    abs (0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)))
  triangle_area p1 p2 p4 + triangle_area p1 p3 p4 = 102

theorem kite_area_correct : ∃ (p1 p2 p3 p4 : Point), 
  p1 = Point.mk 0 10 ∧ 
  p2 = Point.mk 6 14 ∧ 
  p3 = Point.mk 12 10 ∧ 
  p4 = Point.mk 6 0 ∧ 
  Kite p1 p2 p3 p4 :=
by
  sorry

end NUMINAMATH_GPT_kite_area_correct_l1979_197981


namespace NUMINAMATH_GPT_cos_x_minus_pi_over_3_l1979_197923

theorem cos_x_minus_pi_over_3 (x : ℝ) (h : Real.sin (x + π / 6) = 4 / 5) :
  Real.cos (x - π / 3) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_x_minus_pi_over_3_l1979_197923


namespace NUMINAMATH_GPT_polynomial_product_l1979_197957

theorem polynomial_product (x : ℝ) : (x - 1) * (x + 3) * (x + 5) = x^3 + 7*x^2 + 7*x - 15 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_product_l1979_197957


namespace NUMINAMATH_GPT_smallest_d_l1979_197952

theorem smallest_d (c d : ℕ) (h1 : c - d = 8)
  (h2 : Nat.gcd ((c^3 + d^3) / (c + d)) (c * d) = 16) : d = 4 := by
  sorry

end NUMINAMATH_GPT_smallest_d_l1979_197952


namespace NUMINAMATH_GPT_xiaoying_school_trip_l1979_197990

theorem xiaoying_school_trip :
  ∃ (x y : ℝ), 
    (1200 / 1000) = (3 / 60) * x + (5 / 60) * y ∧ 
    x + y = 16 :=
by
  sorry

end NUMINAMATH_GPT_xiaoying_school_trip_l1979_197990


namespace NUMINAMATH_GPT_range_of_a_for_negative_root_l1979_197960

theorem range_of_a_for_negative_root (a : ℝ) :
  (∃ x : ℝ, x < 0 ∧ 4^x - 2^(x-1) + a = 0) →
  - (1/2 : ℝ) < a ∧ a ≤ (1/16 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_negative_root_l1979_197960


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1979_197988

-- Definition of operation T
def T (x y m n : ℚ) := (m * x + n * y) * (x + 2 * y)

-- Problem 1: Given T(1, -1) = 0 and T(0, 2) = 8, prove m = 1 and n = 1
theorem problem1 (m n : ℚ) (h1 : T 1 (-1) m n = 0) (h2 : T 0 2 m n = 8) : m = 1 ∧ n = 1 := by
  sorry

-- Problem 2: Given the system of inequalities in terms of p and knowing T(x, y) = (mx + ny)(x + 2y) with m = 1 and n = 1
--            has exactly 3 integer solutions, prove the range of values for a is 42 ≤ a < 54
theorem problem2 (a : ℚ) 
  (h1 : ∃ p : ℚ, T (2 * p) (2 - p) 1 1 > 4 ∧ T (4 * p) (3 - 2 * p) 1 1 ≤ a)
  (h2 : ∃! p : ℤ, -1 < p ∧ p ≤ (a - 18) / 12) : 42 ≤ a ∧ a < 54 := by
  sorry

-- Problem 3: Given T(x, y) = T(y, x) when x^2 ≠ y^2, prove m = 2n
theorem problem3 (m n : ℚ) 
  (h : ∀ x y : ℚ, x^2 ≠ y^2 → T x y m n = T y x m n) : m = 2 * n := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1979_197988


namespace NUMINAMATH_GPT_total_carrots_l1979_197925

def sally_carrots : ℕ := 6
def fred_carrots : ℕ := 4
def mary_carrots : ℕ := 10

theorem total_carrots : sally_carrots + fred_carrots + mary_carrots = 20 := by
  sorry

end NUMINAMATH_GPT_total_carrots_l1979_197925


namespace NUMINAMATH_GPT_m_range_iff_four_distinct_real_roots_l1979_197962

noncomputable def four_distinct_real_roots (m : ℝ) : Prop :=
∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
(x1^2 - 4 * |x1| + 5 = m) ∧
(x2^2 - 4 * |x2| + 5 = m) ∧
(x3^2 - 4 * |x3| + 5 = m) ∧
(x4^2 - 4 * |x4| + 5 = m)

theorem m_range_iff_four_distinct_real_roots (m : ℝ) :
  four_distinct_real_roots m ↔ 1 < m ∧ m < 5 :=
sorry

end NUMINAMATH_GPT_m_range_iff_four_distinct_real_roots_l1979_197962


namespace NUMINAMATH_GPT_find_first_month_sale_l1979_197979

/-- Given the sales for months two to six and the average sales over six months,
    prove the sale in the first month. -/
theorem find_first_month_sale
  (sales_2 : ℤ) (sales_3 : ℤ) (sales_4 : ℤ) (sales_5 : ℤ) (sales_6 : ℤ)
  (avg_sales : ℤ)
  (h2 : sales_2 = 5468) (h3 : sales_3 = 5568) (h4 : sales_4 = 6088)
  (h5 : sales_5 = 6433) (h6 : sales_6 = 5922) (h_avg : avg_sales = 5900) : 
  ∃ (sale_1 : ℤ), sale_1 = 5921 := 
by
  have total_sales : ℤ := avg_sales * 6
  have known_sales_sum : ℤ := sales_2 + sales_3 + sales_4 + sales_5
  use total_sales - known_sales_sum - sales_6
  sorry

end NUMINAMATH_GPT_find_first_month_sale_l1979_197979


namespace NUMINAMATH_GPT_inradius_of_equal_area_and_perimeter_l1979_197942

theorem inradius_of_equal_area_and_perimeter
  (a b c : ℝ)
  (A : ℝ)
  (h1 : A = a + b + c)
  (s : ℝ := (a + b + c) / 2)
  (h2 : A = s * (2 * A / (a + b + c))) :
  ∃ r : ℝ, r = 2 := by
  sorry

end NUMINAMATH_GPT_inradius_of_equal_area_and_perimeter_l1979_197942


namespace NUMINAMATH_GPT_tangent_line_eq_max_f_val_in_interval_a_le_2_l1979_197941

-- Definitions based on given conditions
def f (x : ℝ) (a : ℝ) : ℝ := x ^ 3 - a * x ^ 2

def f_prime (x : ℝ) (a : ℝ) : ℝ := 3 * x ^ 2 - 2 * a * x

-- (I) (i) Proof that the tangent line equation is y = 3x - 2 at (1, f(1))
theorem tangent_line_eq (a : ℝ) (h : f_prime 1 a = 3) : y = 3 * x - 2 :=
by sorry

-- (I) (ii) Proof that the max value of f(x) in [0,2] is 8
theorem max_f_val_in_interval : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x 0 ≤ f 2 0 :=
by sorry

-- (II) Proof that a ≤ 2 if f(x) + x ≥ 0 for all x ∈ [0,2]
theorem a_le_2 (a : ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → f x a + x ≥ 0) : a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_tangent_line_eq_max_f_val_in_interval_a_le_2_l1979_197941


namespace NUMINAMATH_GPT_composite_proposition_l1979_197973

noncomputable def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 5 ≤ 4

noncomputable def q : Prop := ∀ x : ℝ, 0 < x ∧ x < Real.pi / 2 → ¬ (∀ v : ℝ, v = (Real.sin x + 4 / Real.sin x) → v = 4)

theorem composite_proposition : p ∧ ¬q := 
by 
  sorry

end NUMINAMATH_GPT_composite_proposition_l1979_197973


namespace NUMINAMATH_GPT_find_x_of_equation_l1979_197904

theorem find_x_of_equation :
  ∃ x : ℕ, 16^5 + 16^5 + 16^5 = 4^x ∧ x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_of_equation_l1979_197904
