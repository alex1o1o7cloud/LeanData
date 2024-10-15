import Mathlib

namespace NUMINAMATH_GPT_walking_speed_10_mph_l1862_186241

theorem walking_speed_10_mph 
  (total_minutes : ℕ)
  (distance : ℕ)
  (rest_per_segment : ℕ)
  (rest_time : ℕ)
  (segments : ℕ)
  (walk_time : ℕ)
  (walk_time_hours : ℕ) :
  total_minutes = 328 → 
  distance = 50 → 
  rest_per_segment = 7 → 
  segments = 4 →
  rest_time = segments * rest_per_segment →
  walk_time = total_minutes - rest_time →
  walk_time_hours = walk_time / 60 →
  distance / walk_time_hours = 10 :=
by
  sorry

end NUMINAMATH_GPT_walking_speed_10_mph_l1862_186241


namespace NUMINAMATH_GPT_distance_traveled_downstream_l1862_186250

noncomputable def speed_boat : ℝ := 20  -- Speed of the boat in still water in km/hr
noncomputable def rate_current : ℝ := 5  -- Rate of current in km/hr
noncomputable def time_minutes : ℝ := 24  -- Time traveled downstream in minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert time to hours
noncomputable def effective_speed_downstream : ℝ := speed_boat + rate_current  -- Effective speed downstream

theorem distance_traveled_downstream :
  effective_speed_downstream * time_hours = 10 := by {
  sorry
}

end NUMINAMATH_GPT_distance_traveled_downstream_l1862_186250


namespace NUMINAMATH_GPT_polynomial_remainder_l1862_186245

theorem polynomial_remainder (x : ℤ) : 
  (2 * x + 3) ^ 504 % (x^2 - x + 1) = (16 * x + 5) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_remainder_l1862_186245


namespace NUMINAMATH_GPT_point_on_parabola_touching_x_axis_l1862_186228

theorem point_on_parabola_touching_x_axis (a b c : ℤ) (h : ∃ r : ℤ, a * (r * r) + b * r + c = 0 ∧ (r * r) = 0) :
  ∃ (a' b' : ℤ), ∃ k : ℤ, (k * k) + a' * k + b' = 0 ∧ (k * k) = 0 :=
sorry

end NUMINAMATH_GPT_point_on_parabola_touching_x_axis_l1862_186228


namespace NUMINAMATH_GPT_max_area_of_rectangular_pen_l1862_186273

theorem max_area_of_rectangular_pen (P : ℕ) (hP : P = 60) : 
  ∃ A : ℕ, A = 225 ∧ (∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A) := 
by
  sorry

end NUMINAMATH_GPT_max_area_of_rectangular_pen_l1862_186273


namespace NUMINAMATH_GPT_count_of_squares_difference_l1862_186281

theorem count_of_squares_difference (h_range : ∀ n, 1 ≤ n ∧ n ≤ 2000) :
  ∃ count, (∀ n, 1 ≤ n ∧ n ≤ 2000 → 
             (∃ a b, n = a^2 - b^2)) ↔ count = 1500 :=
by sorry

end NUMINAMATH_GPT_count_of_squares_difference_l1862_186281


namespace NUMINAMATH_GPT_wholesale_cost_per_bag_l1862_186231

theorem wholesale_cost_per_bag (W : ℝ) (h1 : 1.12 * W = 28) : W = 25 :=
sorry

end NUMINAMATH_GPT_wholesale_cost_per_bag_l1862_186231


namespace NUMINAMATH_GPT_birthday_friends_count_l1862_186286

theorem birthday_friends_count (n : ℕ) 
    (h1 : ∃ T, T = 12 * (n + 2)) 
    (h2 : ∃ T', T' = 16 * n) 
    (h3 : (∃ T, T = 12 * (n + 2)) → ∃ T', T' = 16 * n) : 
    n = 6 := 
by
    sorry

end NUMINAMATH_GPT_birthday_friends_count_l1862_186286


namespace NUMINAMATH_GPT_midpoint_set_of_segments_eq_circle_l1862_186234

-- Define the existence of skew perpendicular lines with given properties
variable (a d : ℝ)

-- Conditions: Distance between lines is a, segment length is d
-- The coordinates system configuration
-- Point on the first line: (x, 0, 0)
-- Point on the second line: (0, y, a)
def are_midpoints_of_segments_of_given_length
  (p : ℝ × ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), 
    p = (x / 2, y / 2, a / 2) ∧ 
    x^2 + y^2 = d^2 - a^2

-- Proof statement
theorem midpoint_set_of_segments_eq_circle :
  { p : ℝ × ℝ × ℝ | are_midpoints_of_segments_of_given_length a d p } =
  { p : ℝ × ℝ × ℝ | ∃ (r : ℝ), p = (r * (d^2 - a^2) / (2*d), r * (d^2 - a^2) / (2*d), a / 2)
    ∧ r^2 * (d^2 - a^2) = (d^2 - a^2) } :=
sorry

end NUMINAMATH_GPT_midpoint_set_of_segments_eq_circle_l1862_186234


namespace NUMINAMATH_GPT_range_of_m_l1862_186267

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / (x - 3) - 1 = x / (3 - x)) →
  m > 3 ∧ m ≠ 9 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1862_186267


namespace NUMINAMATH_GPT_ratio_fraction_l1862_186203

theorem ratio_fraction (x : ℚ) : x = 2 / 9 ↔ (2 / 6) / x = (3 / 4) / (1 / 2) := by
  sorry

end NUMINAMATH_GPT_ratio_fraction_l1862_186203


namespace NUMINAMATH_GPT_average_annual_reduction_10_percent_l1862_186252

theorem average_annual_reduction_10_percent :
  ∀ x : ℝ, (1 - x) ^ 2 = 1 - 0.19 → x = 0.1 :=
by
  intros x h
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_average_annual_reduction_10_percent_l1862_186252


namespace NUMINAMATH_GPT_kamal_average_marks_l1862_186240

theorem kamal_average_marks :
  (76 / 120) * 0.2 + 
  (60 / 110) * 0.25 + 
  (82 / 100) * 0.15 + 
  (67 / 90) * 0.2 + 
  (85 / 100) * 0.15 + 
  (78 / 95) * 0.05 = 0.70345 :=
by 
  sorry

end NUMINAMATH_GPT_kamal_average_marks_l1862_186240


namespace NUMINAMATH_GPT_rose_bushes_in_park_l1862_186265

theorem rose_bushes_in_park (current_bushes : ℕ) (newly_planted : ℕ) (h1 : current_bushes = 2) (h2 : newly_planted = 4) : current_bushes + newly_planted = 6 :=
by
  sorry

end NUMINAMATH_GPT_rose_bushes_in_park_l1862_186265


namespace NUMINAMATH_GPT_arithmetic_sequence_third_eighth_term_sum_l1862_186297

variable {α : Type*} [AddCommGroup α] [Module ℚ α]

def arith_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_third_eighth_term_sum {a : ℕ → ℚ} {S : ℕ → ℚ} 
  (h_seq: ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum: arith_sequence_sum a S) 
  (h_S10 : S 10 = 4) : 
  a 3 + a 8 = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_third_eighth_term_sum_l1862_186297


namespace NUMINAMATH_GPT_fred_found_28_more_seashells_l1862_186279

theorem fred_found_28_more_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (h_tom : tom_seashells = 15) (h_fred : fred_seashells = 43) : 
  fred_seashells - tom_seashells = 28 := 
by 
  sorry

end NUMINAMATH_GPT_fred_found_28_more_seashells_l1862_186279


namespace NUMINAMATH_GPT_green_chips_correct_l1862_186213

-- Definitions
def total_chips : ℕ := 120
def blue_chips : ℕ := total_chips / 4
def red_chips : ℕ := total_chips * 20 / 100
def yellow_chips : ℕ := total_chips / 10
def non_green_chips : ℕ := blue_chips + red_chips + yellow_chips
def green_chips : ℕ := total_chips - non_green_chips

-- Statement to prove
theorem green_chips_correct : green_chips = 54 := by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_green_chips_correct_l1862_186213


namespace NUMINAMATH_GPT_fred_total_earnings_l1862_186293

def fred_earnings (earnings_per_hour hours_worked : ℝ) : ℝ := earnings_per_hour * hours_worked

theorem fred_total_earnings :
  fred_earnings 12.5 8 = 100 := by
sorry

end NUMINAMATH_GPT_fred_total_earnings_l1862_186293


namespace NUMINAMATH_GPT_no_such_function_l1862_186276

noncomputable def no_such_function_exists : Prop :=
  ¬∃ f : ℝ → ℝ, 
    (∀ y : ℝ, ∃ x : ℝ, f x = y) ∧ 
    (∀ x : ℝ, f (f x) = (x - 1) * f x + 2)

-- Here's the theorem statement to be proved
theorem no_such_function : no_such_function_exists :=
sorry

end NUMINAMATH_GPT_no_such_function_l1862_186276


namespace NUMINAMATH_GPT_find_real_num_l1862_186288

noncomputable def com_num (a : ℝ) : ℂ := (a + 3 * Complex.I) / (1 + 2 * Complex.I)

theorem find_real_num (a : ℝ) : (∃ b : ℝ, com_num a = b * Complex.I) → a = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_real_num_l1862_186288


namespace NUMINAMATH_GPT_samantha_total_cost_l1862_186214

noncomputable def daily_rental_rate : ℝ := 30
noncomputable def daily_rental_days : ℝ := 3
noncomputable def cost_per_mile : ℝ := 0.15
noncomputable def miles_driven : ℝ := 500

theorem samantha_total_cost :
  (daily_rental_rate * daily_rental_days) + (cost_per_mile * miles_driven) = 165 :=
by
  sorry

end NUMINAMATH_GPT_samantha_total_cost_l1862_186214


namespace NUMINAMATH_GPT_michael_initial_fish_l1862_186247

-- Define the conditions
def benGave : ℝ := 18.0
def totalFish : ℝ := 67

-- Define the statement to be proved
theorem michael_initial_fish :
  (totalFish - benGave) = 49 := by
  sorry

end NUMINAMATH_GPT_michael_initial_fish_l1862_186247


namespace NUMINAMATH_GPT_exp_add_l1862_186257

theorem exp_add (z w : Complex) : Complex.exp z * Complex.exp w = Complex.exp (z + w) := 
by 
  sorry

end NUMINAMATH_GPT_exp_add_l1862_186257


namespace NUMINAMATH_GPT_second_integer_value_l1862_186295

-- Definitions of conditions directly from a)
def consecutive_integers (a b c : ℤ) : Prop :=
  b = a + 1 ∧ c = b + 1

def sum_of_first_and_third (a c : ℤ) (sum : ℤ) : Prop :=
  a + c = sum

-- Translated proof problem
theorem second_integer_value (n: ℤ) (h1: consecutive_integers (n - 1) n (n + 1))
  (h2: sum_of_first_and_third (n - 1) (n + 1) 118) : 
  n = 59 :=
by
  sorry

end NUMINAMATH_GPT_second_integer_value_l1862_186295


namespace NUMINAMATH_GPT_minimize_AC_plus_BC_l1862_186264

noncomputable def minimize_distance (k : ℝ) : Prop :=
  let A := (5, 5)
  let B := (2, 1)
  let C := (0, k)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  let AC := dist A C
  let BC := dist B C
  ∀ k', dist (0, k') A + dist (0, k') B ≥ AC + BC

theorem minimize_AC_plus_BC : minimize_distance (15 / 7) :=
sorry

end NUMINAMATH_GPT_minimize_AC_plus_BC_l1862_186264


namespace NUMINAMATH_GPT_price_of_jumbo_pumpkin_l1862_186216

theorem price_of_jumbo_pumpkin (total_pumpkins : ℕ) (total_revenue : ℝ)
  (regular_pumpkins : ℕ) (price_regular : ℝ)
  (sold_jumbo_pumpkins : ℕ) (revenue_jumbo : ℝ): 
  total_pumpkins = 80 →
  total_revenue = 395.00 →
  regular_pumpkins = 65 →
  price_regular = 4.00 →
  sold_jumbo_pumpkins = total_pumpkins - regular_pumpkins →
  revenue_jumbo = total_revenue - (price_regular * regular_pumpkins) →
  revenue_jumbo / sold_jumbo_pumpkins = 9.00 :=
by
  intro h_total_pumpkins
  intro h_total_revenue
  intro h_regular_pumpkins
  intro h_price_regular
  intro h_sold_jumbo_pumpkins
  intro h_revenue_jumbo
  sorry

end NUMINAMATH_GPT_price_of_jumbo_pumpkin_l1862_186216


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_primes_l1862_186284

theorem smallest_four_digit_divisible_by_primes : 
  let p1 := 2 
  let p2 := 3 
  let p3 := 5 
  let p4 := 7 
  let p5 := 11 
  let lcm_val := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm p1 p2) p3) p4) p5 
  1000 ≤ lcm_val ∧ lcm_val < 10000 :=
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_primes_l1862_186284


namespace NUMINAMATH_GPT_sum_of_exponents_of_1985_eq_40_l1862_186271

theorem sum_of_exponents_of_1985_eq_40 :
  ∃ (e₀ e₁ e₂ e₃ e₄ e₅ : ℕ), 1985 = 2^e₀ + 2^e₁ + 2^e₂ + 2^e₃ + 2^e₄ + 2^e₅ 
  ∧ e₀ ≠ e₁ ∧ e₀ ≠ e₂ ∧ e₀ ≠ e₃ ∧ e₀ ≠ e₄ ∧ e₀ ≠ e₅
  ∧ e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₁ ≠ e₅
  ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₂ ≠ e₅
  ∧ e₃ ≠ e₄ ∧ e₃ ≠ e₅
  ∧ e₄ ≠ e₅
  ∧ e₀ + e₁ + e₂ + e₃ + e₄ + e₅ = 40 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_exponents_of_1985_eq_40_l1862_186271


namespace NUMINAMATH_GPT_sum_of_diagonals_l1862_186296

-- Definitions of the given lengths
def AB := 5
def CD := 5
def BC := 12
def DE := 12
def AE := 18

-- Variables for the diagonal lengths
variables (AC BD CE : ℚ)

-- The Lean 4 theorem statement
theorem sum_of_diagonals (hAC : AC = 723 / 44) (hBD : BD = 44 / 3) (hCE : CE = 351 / 22) :
  AC + BD + CE = 6211 / 132 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_diagonals_l1862_186296


namespace NUMINAMATH_GPT_restaurant_meal_cost_l1862_186249

/--
Each adult meal costs $8 and kids eat free. 
If there is a group of 11 people, out of which 2 are kids, 
prove that the total cost for the group to eat is $72.
-/
theorem restaurant_meal_cost (cost_per_adult : ℕ) (group_size : ℕ) (kids : ℕ) 
  (all_free_kids : ℕ → Prop) (total_cost : ℕ)  
  (h1 : cost_per_adult = 8) 
  (h2 : group_size = 11) 
  (h3 : kids = 2) 
  (h4 : all_free_kids kids) 
  (h5 : total_cost = (group_size - kids) * cost_per_adult) : 
  total_cost = 72 := 
by 
  sorry

end NUMINAMATH_GPT_restaurant_meal_cost_l1862_186249


namespace NUMINAMATH_GPT_cos_transformation_l1862_186220

variable {θ a : ℝ}

theorem cos_transformation (h : Real.sin (θ + π / 12) = a) :
  Real.cos (θ + 7 * π / 12) = -a := 
sorry

end NUMINAMATH_GPT_cos_transformation_l1862_186220


namespace NUMINAMATH_GPT_intercept_form_l1862_186246

theorem intercept_form (x y : ℝ) : 2 * x - 3 * y - 4 = 0 ↔ x / 2 + y / (-4/3) = 1 := sorry

end NUMINAMATH_GPT_intercept_form_l1862_186246


namespace NUMINAMATH_GPT_calculation_correct_l1862_186277

def grid_coloring_probability : ℚ := 591 / 1024

theorem calculation_correct : (m + n = 1615) ↔ (∃ m n : ℕ, m + n = 1615 ∧ gcd m n = 1 ∧ grid_coloring_probability = m / n) := sorry

end NUMINAMATH_GPT_calculation_correct_l1862_186277


namespace NUMINAMATH_GPT_volume_of_water_overflow_l1862_186256

-- Definitions based on given conditions
def mass_of_ice : ℝ := 50
def density_of_fresh_ice : ℝ := 0.9
def density_of_salt_ice : ℝ := 0.95
def density_of_fresh_water : ℝ := 1
def density_of_salt_water : ℝ := 1.03

-- Theorem statement corresponding to the problem
theorem volume_of_water_overflow
  (m : ℝ := mass_of_ice) 
  (rho_n : ℝ := density_of_fresh_ice) 
  (rho_c : ℝ := density_of_salt_ice) 
  (rho_fw : ℝ := density_of_fresh_water) 
  (rho_sw : ℝ := density_of_salt_water) :
  ∃ (ΔV : ℝ), ΔV = 2.63 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_water_overflow_l1862_186256


namespace NUMINAMATH_GPT_river_width_l1862_186275

theorem river_width 
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
  (h_depth : depth = 2) 
  (h_flow_rate: flow_rate_kmph = 3) 
  (h_volume : volume_per_minute = 4500) : 
  the_width_of_the_river = 45 :=
by
  sorry 

end NUMINAMATH_GPT_river_width_l1862_186275


namespace NUMINAMATH_GPT_remainder_549547_div_7_l1862_186210

theorem remainder_549547_div_7 : 549547 % 7 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_549547_div_7_l1862_186210


namespace NUMINAMATH_GPT_fraction_of_y_l1862_186253

theorem fraction_of_y (w x y : ℝ) (h1 : wx = y) 
  (h2 : (w + x) / 2 = 0.5) : 
  (2 / w + 2 / x = 2 / y) := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_y_l1862_186253


namespace NUMINAMATH_GPT_no_such_abc_exists_l1862_186226

theorem no_such_abc_exists : ¬ ∃ (a b c : ℝ), ∀ (x y : ℝ),
  |x + a| + |x + y + b| + |y + c| > |x| + |x + y| + |y| :=
by
  sorry

end NUMINAMATH_GPT_no_such_abc_exists_l1862_186226


namespace NUMINAMATH_GPT_class_students_l1862_186217

theorem class_students (A B : ℕ) 
  (h1 : A + B = 85) 
  (h2 : (3 * A) / 8 + (3 * B) / 5 = 42) : 
  A = 40 ∧ B = 45 :=
by
  sorry

end NUMINAMATH_GPT_class_students_l1862_186217


namespace NUMINAMATH_GPT_no_real_solution_for_x_l1862_186282

theorem no_real_solution_for_x
  (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/x = 1/3) :
  false :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_for_x_l1862_186282


namespace NUMINAMATH_GPT_bowling_ball_surface_area_l1862_186287

theorem bowling_ball_surface_area (diameter : ℝ) (h : diameter = 9) :
    let r := diameter / 2
    let surface_area := 4 * Real.pi * r^2
    surface_area = 81 * Real.pi := by
  sorry

end NUMINAMATH_GPT_bowling_ball_surface_area_l1862_186287


namespace NUMINAMATH_GPT_calculate_expression_l1862_186206

theorem calculate_expression (x y : ℚ) (hx : x = 5 / 6) (hy : y = 6 / 5) : 
  (1 / 3) * (x ^ 8) * (y ^ 9) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1862_186206


namespace NUMINAMATH_GPT_math_problem_l1862_186221

theorem math_problem (a b : ℝ) (h : a / (1 + a) + b / (1 + b) = 1) : 
  a / (1 + b^2) - b / (1 + a^2) = a - b := 
sorry

end NUMINAMATH_GPT_math_problem_l1862_186221


namespace NUMINAMATH_GPT_determine_g_l1862_186200

noncomputable def g : ℝ → ℝ := sorry 

lemma g_functional_equation (x y : ℝ) : g (x * y) = g ((x^2 + y^2 + 1) / 3) + (x - y)^2 :=
sorry

lemma g_at_zero : g 0 = 1 :=
sorry

theorem determine_g (x : ℝ) : g x = 2 - 2 * x :=
sorry

end NUMINAMATH_GPT_determine_g_l1862_186200


namespace NUMINAMATH_GPT_probability_adjacent_vertices_decagon_l1862_186258

noncomputable def probability_adjacent_vertices : ℚ :=
  let num_vertices := 10
  let adjacent := 2
  let remaining := num_vertices - 1 -- 9
  adjacent / remaining

theorem probability_adjacent_vertices_decagon :
  probability_adjacent_vertices = 2 / 9 :=
by
  unfold probability_adjacent_vertices
  simp
  sorry

end NUMINAMATH_GPT_probability_adjacent_vertices_decagon_l1862_186258


namespace NUMINAMATH_GPT_inconsistency_proof_l1862_186283

-- Let TotalBoys be the number of boys, which is 120
def TotalBoys := 120

-- Let AverageMarks be the average marks obtained by 120 boys, which is 40
def AverageMarks := 40

-- Let PassedBoys be the number of boys who passed, which is 125
def PassedBoys := 125

-- Let AverageMarksFailed be the average marks of failed boys, which is 15
def AverageMarksFailed := 15

-- We need to prove the inconsistency
theorem inconsistency_proof :
  ∀ (P : ℝ), 
    (TotalBoys * AverageMarks = PassedBoys * P + (TotalBoys - PassedBoys) * AverageMarksFailed) →
    False :=
by
  intro P h
  sorry

end NUMINAMATH_GPT_inconsistency_proof_l1862_186283


namespace NUMINAMATH_GPT_quadratic_expression_evaluation_l1862_186266

theorem quadratic_expression_evaluation (x y : ℝ) (h1 : 3 * x + y = 10) (h2 : x + 3 * y = 14) :
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_quadratic_expression_evaluation_l1862_186266


namespace NUMINAMATH_GPT_median_length_angle_bisector_length_l1862_186278

variable (a b c : ℝ) (ma n : ℝ)

theorem median_length (h1 : ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4)) : 
  ma = Real.sqrt ((b^2 + c^2) / 2 - a^2 / 4) :=
by
  sorry

theorem angle_bisector_length (h2 : n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2)) :
  n = b * c * Real.sqrt (((b + c)^2 - a^2) / (b + c)^2) :=
by
  sorry

end NUMINAMATH_GPT_median_length_angle_bisector_length_l1862_186278


namespace NUMINAMATH_GPT_emily_art_supplies_l1862_186233

theorem emily_art_supplies (total_spent skirts_cost skirt_quantity : ℕ) 
  (total_spent_eq : total_spent = 50) 
  (skirt_cost_eq : skirts_cost = 15) 
  (skirt_quantity_eq : skirt_quantity = 2) :
  total_spent - skirt_quantity * skirts_cost = 20 :=
by
  sorry

end NUMINAMATH_GPT_emily_art_supplies_l1862_186233


namespace NUMINAMATH_GPT_other_number_is_31_l1862_186242

namespace LucasProblem

-- Definitions of the integers a and b and the condition on their sum
variables (a b : ℤ)
axiom h_sum : 3 * a + 4 * b = 161
axiom h_one_is_17 : a = 17 ∨ b = 17

-- The theorem we need to prove
theorem other_number_is_31 (h_one_is_17 : a = 17 ∨ b = 17) : 
  (b = 17 → a = 31) ∧ (a = 17 → false) :=
by
  sorry

end LucasProblem

end NUMINAMATH_GPT_other_number_is_31_l1862_186242


namespace NUMINAMATH_GPT_find_m_l1862_186244

-- Definitions based on conditions
def Point (α : Type) := α × α

def A : Point ℝ := (2, -3)
def B : Point ℝ := (4, 3)
def C (m : ℝ) : Point ℝ := (5, m)

-- The collinearity condition
def collinear (p1 p2 p3 : Point ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

-- The proof problem
theorem find_m (m : ℝ) : collinear A B (C m) → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1862_186244


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l1862_186211

theorem boat_speed_in_still_water :
  ∀ (V_b V_s : ℝ) (distance time : ℝ),
  V_s = 5 →
  time = 4 →
  distance = 84 →
  (distance / time) = V_b + V_s →
  V_b = 16 :=
by
  -- Given definitions and values
  intros V_b V_s distance time
  intro hV_s
  intro htime
  intro hdistance
  intro heq
  sorry -- Placeholder for the actual proof

end NUMINAMATH_GPT_boat_speed_in_still_water_l1862_186211


namespace NUMINAMATH_GPT_cost_per_square_meter_l1862_186215

-- Definitions from conditions
def lawn_length : ℝ := 80
def lawn_breadth : ℝ := 50
def road_width : ℝ := 10
def total_cost : ℝ := 3600

-- Theorem to prove the cost per square meter of traveling the roads
theorem cost_per_square_meter :
  total_cost / 
  ((lawn_length * road_width) + (lawn_breadth * road_width) - (road_width * road_width)) = 3 := by
  sorry

end NUMINAMATH_GPT_cost_per_square_meter_l1862_186215


namespace NUMINAMATH_GPT_gerald_pfennigs_left_l1862_186292

theorem gerald_pfennigs_left (cost_of_pie : ℕ) (farthings_initial : ℕ) (farthings_per_pfennig : ℕ) :
  cost_of_pie = 2 → farthings_initial = 54 → farthings_per_pfennig = 6 → 
  (farthings_initial / farthings_per_pfennig) - cost_of_pie = 7 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_gerald_pfennigs_left_l1862_186292


namespace NUMINAMATH_GPT_proposition_B_proposition_D_l1862_186280

open Real

variable (a b : ℝ)

theorem proposition_B (h : a^2 ≠ b^2) : a ≠ b := 
sorry

theorem proposition_D (h : a > abs b) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_proposition_B_proposition_D_l1862_186280


namespace NUMINAMATH_GPT_no_term_in_sequence_is_3_alpha_5_beta_l1862_186208

theorem no_term_in_sequence_is_3_alpha_5_beta :
  ∀ (v : ℕ → ℕ),
    v 0 = 0 →
    v 1 = 1 →
    (∀ n, 1 ≤ n → v (n + 1) = 8 * v n * v (n - 1)) →
    ∀ n, ∀ (α β : ℕ), α > 0 → β > 0 → v n ≠ 3^α * 5^β := by
  intros v h0 h1 recurrence n α β hα hβ
  sorry

end NUMINAMATH_GPT_no_term_in_sequence_is_3_alpha_5_beta_l1862_186208


namespace NUMINAMATH_GPT_focus_coordinates_of_parabola_l1862_186263

def parabola_focus_coordinates (x y : ℝ) : Prop :=
  x^2 + y = 0 ∧ (0, -1/4) = (0, y)

theorem focus_coordinates_of_parabola (x y : ℝ) :
  parabola_focus_coordinates x y →
  (0, y) = (0, -1/4) := by
  sorry

end NUMINAMATH_GPT_focus_coordinates_of_parabola_l1862_186263


namespace NUMINAMATH_GPT_value_of_q_l1862_186259

open Real

theorem value_of_q (p q : ℝ) (hpq_cond1 : 1 < p ∧ p < q) 
  (hpq_cond2 : 1 / p + 1 / q = 1) (hpq_cond3 : p * q = 8) : q = 4 + 2 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_q_l1862_186259


namespace NUMINAMATH_GPT_cost_of_four_enchiladas_and_five_tacos_l1862_186230

-- Define the cost of an enchilada and a taco
variables (e t : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := e + 4 * t = 2.30
def condition2 : Prop := 4 * e + t = 3.10

-- Define the final cost of four enchiladas and five tacos
def cost : ℝ := 4 * e + 5 * t

-- State the theorem we need to prove
theorem cost_of_four_enchiladas_and_five_tacos 
  (h1 : condition1 e t) 
  (h2 : condition2 e t) : 
  cost e t = 4.73 := 
sorry

end NUMINAMATH_GPT_cost_of_four_enchiladas_and_five_tacos_l1862_186230


namespace NUMINAMATH_GPT_equal_sum_sequence_S_9_l1862_186260

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Conditions taken from the problem statement
def equal_sum_sequence (a : ℕ → ℕ) (c : ℕ) :=
  ∀ n : ℕ, a n + a (n + 1) = c

def sum_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Lean statement of the problem
theorem equal_sum_sequence_S_9
  (h1 : equal_sum_sequence a 5)
  (h2 : a 1 = 2)
  : sum_first_n_terms a 9 = 22 :=
sorry

end NUMINAMATH_GPT_equal_sum_sequence_S_9_l1862_186260


namespace NUMINAMATH_GPT_garage_sale_items_count_l1862_186229

theorem garage_sale_items_count :
  (16 + 22) + 1 = 38 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_garage_sale_items_count_l1862_186229


namespace NUMINAMATH_GPT_tan_prod_eq_sqrt_seven_l1862_186223

theorem tan_prod_eq_sqrt_seven : 
  let x := (Real.pi / 7) 
  let y := (2 * Real.pi / 7)
  let z := (3 * Real.pi / 7)
  Real.tan x * Real.tan y * Real.tan z = Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_prod_eq_sqrt_seven_l1862_186223


namespace NUMINAMATH_GPT_half_is_greater_than_third_by_one_sixth_l1862_186222

theorem half_is_greater_than_third_by_one_sixth : (0.5 : ℝ) - (1 / 3 : ℝ) = 1 / 6 := by
  sorry

end NUMINAMATH_GPT_half_is_greater_than_third_by_one_sixth_l1862_186222


namespace NUMINAMATH_GPT_xiaohong_money_l1862_186236

def cost_kg_pears (x : ℝ) := x

def cost_kg_apples (x : ℝ) := x + 1.1

theorem xiaohong_money (x : ℝ) (hx : 6 * x - 3 = 5 * (x + 1.1) - 4) : 6 * x - 3 = 24 :=
by sorry

end NUMINAMATH_GPT_xiaohong_money_l1862_186236


namespace NUMINAMATH_GPT_solve_polynomial_l1862_186294

theorem solve_polynomial (z : ℂ) :
    z^5 - 5 * z^3 + 6 * z = 0 ↔ 
    z = 0 ∨ z = -Real.sqrt 2 ∨ z = Real.sqrt 2 ∨ z = -Real.sqrt 3 ∨ z = Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_solve_polynomial_l1862_186294


namespace NUMINAMATH_GPT_determine_a_l1862_186212

theorem determine_a 
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x - 2| < 3 ↔ - 5 / 3 < x ∧ x < 1 / 3) : 
  a = -3 := by 
  sorry

end NUMINAMATH_GPT_determine_a_l1862_186212


namespace NUMINAMATH_GPT_max_value_quadratic_expression_l1862_186238

theorem max_value_quadratic_expression : ∃ x : ℝ, -3 * x^2 + 18 * x - 5 ≤ 22 ∧ ∀ y : ℝ, -3 * y^2 + 18 * y - 5 ≤ 22 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_quadratic_expression_l1862_186238


namespace NUMINAMATH_GPT_distance_car_to_stream_l1862_186209

theorem distance_car_to_stream (total_distance : ℝ) (stream_to_meadow : ℝ) (meadow_to_campsite : ℝ) (h1 : total_distance = 0.7) (h2 : stream_to_meadow = 0.4) (h3 : meadow_to_campsite = 0.1) :
  (total_distance - (stream_to_meadow + meadow_to_campsite) = 0.2) :=
by
  sorry

end NUMINAMATH_GPT_distance_car_to_stream_l1862_186209


namespace NUMINAMATH_GPT_number_count_two_digit_property_l1862_186248

open Nat

theorem number_count_two_digit_property : 
  (∃ (n : Finset ℕ), (∀ (x : ℕ), x ∈ n ↔ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 11 * a + 2 * b ≡ 7 [MOD 10] ∧ x = 10 * a + b) ∧ n.card = 5) :=
by
  sorry

end NUMINAMATH_GPT_number_count_two_digit_property_l1862_186248


namespace NUMINAMATH_GPT_max_stickers_single_player_l1862_186201

noncomputable def max_stickers (num_players : ℕ) (average_stickers : ℕ) : ℕ :=
  let total_stickers := num_players * average_stickers
  let min_stickers_one_player := 1
  let min_stickers_others := (num_players - 1) * min_stickers_one_player
  total_stickers - min_stickers_others

theorem max_stickers_single_player : 
  ∀ (num_players average_stickers : ℕ), 
    num_players = 25 → 
    average_stickers = 4 →
    ∀ player_stickers : ℕ, player_stickers ≤ max_stickers num_players average_stickers → player_stickers = 76 :=
    by
      intro num_players average_stickers players_eq avg_eq player_stickers player_le_max
      sorry

end NUMINAMATH_GPT_max_stickers_single_player_l1862_186201


namespace NUMINAMATH_GPT_paving_stone_width_l1862_186269

theorem paving_stone_width :
  ∀ (L₁ L₂ : ℝ) (n : ℕ) (length width : ℝ), 
    L₁ = 30 → L₂ = 16 → length = 2 → n = 240 →
    (L₁ * L₂ = n * (length * width)) → width = 1 :=
by
  sorry

end NUMINAMATH_GPT_paving_stone_width_l1862_186269


namespace NUMINAMATH_GPT_hyperbola_asymptote_product_l1862_186239

theorem hyperbola_asymptote_product (k1 k2 : ℝ) (h1 : k1 = 1) (h2 : k2 = -1) :
  k1 * k2 = -1 :=
by
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_hyperbola_asymptote_product_l1862_186239


namespace NUMINAMATH_GPT_abhay_speed_l1862_186235

variables (A S : ℝ)

theorem abhay_speed (h1 : 24 / A = 24 / S + 2) (h2 : 24 / (2 * A) = 24 / S - 1) : A = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_abhay_speed_l1862_186235


namespace NUMINAMATH_GPT_arcsin_neg_one_half_l1862_186298

theorem arcsin_neg_one_half : Real.arcsin (-1 / 2) = -Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_neg_one_half_l1862_186298


namespace NUMINAMATH_GPT_identify_linear_equation_l1862_186219

def is_linear_equation (eq : String) : Prop := sorry

theorem identify_linear_equation :
  is_linear_equation "2x = 0" ∧ ¬is_linear_equation "x^2 - 4x = 3" ∧ ¬is_linear_equation "x + 2y = 1" ∧ ¬is_linear_equation "x - 1 = 1 / x" :=
by 
  sorry

end NUMINAMATH_GPT_identify_linear_equation_l1862_186219


namespace NUMINAMATH_GPT_temperature_problem_product_of_possible_N_l1862_186289

theorem temperature_problem (M L : ℤ) (N : ℤ) :
  (M = L + N) →
  (M - 8 = L + N - 8) →
  (L + 4 = L + 4) →
  (|((L + N - 8) - (L + 4))| = 3) →
  N = 15 ∨ N = 9 :=
by sorry

theorem product_of_possible_N :
  (∀ M L : ℤ, ∀ N : ℤ,
    (M = L + N) →
    (M - 8 = L + N - 8) →
    (L + 4 = L + 4) →
    (|((L + N - 8) - (L + 4))| = 3) →
    N = 15 ∨ N = 9) →
    15 * 9 = 135 :=
by sorry

end NUMINAMATH_GPT_temperature_problem_product_of_possible_N_l1862_186289


namespace NUMINAMATH_GPT_triplets_of_positive_integers_l1862_186227

/-- We want to determine all positive integer triplets (a, b, c) such that
    ab - c, bc - a, and ca - b are all powers of 2.
    A power of 2 is an integer of the form 2^n, where n is a non-negative integer.-/
theorem triplets_of_positive_integers (a b c : ℕ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) :
  ((∃ k1 : ℕ, ab - c = 2^k1) ∧ (∃ k2 : ℕ, bc - a = 2^k2) ∧ (∃ k3 : ℕ, ca - b = 2^k3))
  ↔ (a = 2 ∧ b = 2 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 2) ∨ (a = 2 ∧ b = 6 ∧ c = 11) ∨ (a = 3 ∧ b = 5 ∧ c = 7) :=
sorry

end NUMINAMATH_GPT_triplets_of_positive_integers_l1862_186227


namespace NUMINAMATH_GPT_find_k_l1862_186232

theorem find_k : ∃ k : ℕ, ∀ n : ℕ, n > 0 → (2^n + 11) % (2^k - 1) = 0 ↔ k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1862_186232


namespace NUMINAMATH_GPT_meaningful_sqrt_l1862_186207

theorem meaningful_sqrt (a : ℝ) (h : a ≥ 4) : a = 6 ↔ ∃ x ∈ ({-1, 0, 2, 6} : Set ℝ), x = 6 := 
by
  sorry

end NUMINAMATH_GPT_meaningful_sqrt_l1862_186207


namespace NUMINAMATH_GPT_usual_time_catch_bus_l1862_186254

-- Define the problem context
variable (S T : ℝ)

-- Hypotheses for the conditions given
def condition1 : Prop := S * T = (4 / 5) * S * (T + 4)
def condition2 : Prop := S ≠ 0

-- Theorem that states the fact we need to prove
theorem usual_time_catch_bus (h1 : condition1 S T) (h2 : condition2 S) : T = 16 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_usual_time_catch_bus_l1862_186254


namespace NUMINAMATH_GPT_max_sin_product_proof_l1862_186291

noncomputable def max_sin_product : ℝ :=
  let A := (-8, 0)
  let B := (8, 0)
  let C (t : ℝ) := (t, 6)
  let AB : ℝ := 16
  let AC (t : ℝ) := Real.sqrt ((t + 8)^2 + 36)
  let BC (t : ℝ) := Real.sqrt ((t - 8)^2 + 36)
  let area : ℝ := 48
  let sin_ACB (t : ℝ) := 96 / Real.sqrt (((t + 8)^2 + 36) * ((t - 8)^2 + 36))
  let sin_CAB_CBA : ℝ := 3 / 8
  sin_CAB_CBA

theorem max_sin_product_proof : ∀ t : ℝ, max_sin_product = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_sin_product_proof_l1862_186291


namespace NUMINAMATH_GPT_total_value_of_coins_l1862_186202

theorem total_value_of_coins (num_quarters num_nickels : ℕ) (val_quarter val_nickel : ℝ)
  (h_quarters : num_quarters = 8) (h_nickels : num_nickels = 13)
  (h_total_coins : num_quarters + num_nickels = 21) (h_val_quarter : val_quarter = 0.25)
  (h_val_nickel : val_nickel = 0.05) :
  num_quarters * val_quarter + num_nickels * val_nickel = 2.65 := 
sorry

end NUMINAMATH_GPT_total_value_of_coins_l1862_186202


namespace NUMINAMATH_GPT_fiona_reaches_goal_l1862_186224

-- Define the set of lily pads
def pads : Finset ℕ := Finset.range 15

-- Define the start, predator, and goal pads
def start_pad : ℕ := 0
def predator_pads : Finset ℕ := {4, 8}
def goal_pad : ℕ := 13

-- Define the hop probabilities
def hop_next : ℚ := 1/3
def hop_two : ℚ := 1/3
def hop_back : ℚ := 1/3

-- Define the transition probabilities (excluding jumps to negative pads)
def transition (current next : ℕ) : ℚ :=
  if next = current + 1 ∨ next = current + 2 ∨ (next = current - 1 ∧ current > 0)
  then 1/3 else 0

-- Define the function to check if a pad is safe
def is_safe (pad : ℕ) : Prop := ¬ (pad ∈ predator_pads)

-- Define the probability that Fiona reaches pad 13 without landing on 4 or 8
noncomputable def probability_reach_13 : ℚ :=
  -- Function to recursively calculate the probability
  sorry

-- Statement to prove
theorem fiona_reaches_goal : probability_reach_13 = 16 / 177147 := 
sorry

end NUMINAMATH_GPT_fiona_reaches_goal_l1862_186224


namespace NUMINAMATH_GPT_trajectory_equation_l1862_186268

theorem trajectory_equation : ∀ (x y : ℝ),
  (x + 3)^2 + y^2 + (x - 3)^2 + y^2 = 38 → x^2 + y^2 = 10 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_trajectory_equation_l1862_186268


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1862_186299

def p (x : ℝ) : Prop := x < 1
def q (x : ℝ) : Prop := x^2 + x - 2 < 0

theorem necessary_but_not_sufficient (x : ℝ):
  (p x → q x) ∧ (q x → p x) → False ∧ (q x → p x) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1862_186299


namespace NUMINAMATH_GPT_seth_oranges_l1862_186205

def initial_boxes := 9
def boxes_given_to_mother := 1

def remaining_boxes_after_giving_to_mother := initial_boxes - boxes_given_to_mother
def boxes_given_away := remaining_boxes_after_giving_to_mother / 2
def boxes_left := remaining_boxes_after_giving_to_mother - boxes_given_away

theorem seth_oranges : boxes_left = 4 := by
  sorry

end NUMINAMATH_GPT_seth_oranges_l1862_186205


namespace NUMINAMATH_GPT_max_grapes_discarded_l1862_186270

theorem max_grapes_discarded (n : ℕ) : 
  ∃ k : ℕ, k ∣ n → 7 * k + 6 = n → ∃ m, m = 6 := by
  sorry

end NUMINAMATH_GPT_max_grapes_discarded_l1862_186270


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l1862_186285

variable {a_n : ℕ → ℝ}

-- Defining the geometric sequence and the given conditions
def is_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, a_n (n + 1) = a_n n * r

def is_increasing_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n, a_n n < a_n (n + 1)

def condition (a_n : ℕ → ℝ) : Prop := a_n 0 < a_n 1 ∧ a_n 1 < a_n 2

-- The proof statement
theorem sufficient_and_necessary_condition (a_n : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a_n) :
  condition a_n ↔ is_increasing_sequence a_n :=
sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l1862_186285


namespace NUMINAMATH_GPT_average_of_first_45_results_l1862_186251

theorem average_of_first_45_results
  (A : ℝ)
  (h1 : (45 + 25 : ℝ) = 70)
  (h2 : (25 : ℝ) * 45 = 1125)
  (h3 : (70 : ℝ) * 32.142857142857146 = 2250)
  (h4 : ∀ x y z : ℝ, 45 * x + y = z → x = 25) :
  A = 25 :=
by
  sorry

end NUMINAMATH_GPT_average_of_first_45_results_l1862_186251


namespace NUMINAMATH_GPT_cylinder_in_sphere_volume_difference_is_correct_l1862_186243

noncomputable def volume_difference (base_radius_cylinder : ℝ) (radius_sphere : ℝ) : ℝ :=
  let height_cylinder := Real.sqrt (radius_sphere^2 - base_radius_cylinder^2)
  let volume_sphere := (4 / 3) * Real.pi * radius_sphere^3
  let volume_cylinder := Real.pi * base_radius_cylinder^2 * height_cylinder
  volume_sphere - volume_cylinder

theorem cylinder_in_sphere_volume_difference_is_correct :
  volume_difference 4 7 = (1372 - 48 * Real.sqrt 33) / 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cylinder_in_sphere_volume_difference_is_correct_l1862_186243


namespace NUMINAMATH_GPT_evaluate_expression_l1862_186204

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1862_186204


namespace NUMINAMATH_GPT_complement_union_l1862_186261

def M := { x : ℝ | (x + 3) * (x - 1) < 0 }
def N := { x : ℝ | x ≤ -3 }
def union_set := M ∪ N

theorem complement_union :
  ∀ x : ℝ, x ∈ (⊤ \ union_set) ↔ x ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1862_186261


namespace NUMINAMATH_GPT_conference_problem_l1862_186262

noncomputable def exists_round_table (n : ℕ) (scientists : Finset ℕ) (acquaintance : ℕ → Finset ℕ) : Prop :=
  ∃ (A B C D : ℕ), A ∈ scientists ∧ B ∈ scientists ∧ C ∈ scientists ∧ D ∈ scientists ∧
  ((A ≠ B ∧ A ≠ C ∧ A ≠ D) ∧ (B ≠ C ∧ B ≠ D) ∧ (C ≠ D)) ∧
  (B ∈ acquaintance A ∧ C ∈ acquaintance B ∧ D ∈ acquaintance C ∧ A ∈ acquaintance D)

theorem conference_problem :
  ∀ (scientists : Finset ℕ),
  ∀ (acquaintance : ℕ → Finset ℕ),
    (scientists.card = 50) →
    (∀ s ∈ scientists, (acquaintance s).card ≥ 25) →
    exists_round_table 50 scientists acquaintance :=
sorry

end NUMINAMATH_GPT_conference_problem_l1862_186262


namespace NUMINAMATH_GPT_fraction_comparison_l1862_186225

theorem fraction_comparison : (5555553 / 5555557 : ℚ) > (6666664 / 6666669 : ℚ) :=
  sorry

end NUMINAMATH_GPT_fraction_comparison_l1862_186225


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1862_186290

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {y | y^2 + y = 0}

theorem intersection_of_A_and_B : A ∩ B = {0} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1862_186290


namespace NUMINAMATH_GPT_overall_average_speed_is_six_l1862_186274

-- Definitions of the conditions
def cycling_time := 45 / 60 -- hours
def cycling_speed := 12 -- mph
def stopping_time := 15 / 60 -- hours
def walking_time := 75 / 60 -- hours
def walking_speed := 3 -- mph

-- Problem statement: Proving that the overall average speed is 6 mph
theorem overall_average_speed_is_six : 
  (cycling_speed * cycling_time + walking_speed * walking_time) /
  (cycling_time + walking_time + stopping_time) = 6 :=
by
  sorry

end NUMINAMATH_GPT_overall_average_speed_is_six_l1862_186274


namespace NUMINAMATH_GPT_find_a_div_b_l1862_186218

theorem find_a_div_b (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 6 * b) / (b + 6 * a) = 3) : 
  a / b = (8 + Real.sqrt 46) / 6 ∨ a / b = (8 - Real.sqrt 46) / 6 :=
by 
  sorry

end NUMINAMATH_GPT_find_a_div_b_l1862_186218


namespace NUMINAMATH_GPT_find_m_l1862_186272

-- Define the given vectors and the parallel condition
def vectors_parallel (m : ℝ) : Prop :=
  let a := (1, m)
  let b := (3, 1)
  a.1 * b.2 = a.2 * b.1

-- Statement to be proved
theorem find_m (m : ℝ) : vectors_parallel m → m = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1862_186272


namespace NUMINAMATH_GPT_monotonic_function_range_l1862_186237

theorem monotonic_function_range (a : ℝ) :
  (∀ x : ℝ, -3 * x^2 + 2 * a * x - 1 ≤ 0) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_monotonic_function_range_l1862_186237


namespace NUMINAMATH_GPT_highway_length_l1862_186255

theorem highway_length 
  (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) 
  (h_speed1 : speed1 = 14)
  (h_speed2 : speed2 = 16)
  (h_time : time = 1.5) : 
  speed1 * time + speed2 * time = 45 := 
sorry

end NUMINAMATH_GPT_highway_length_l1862_186255
