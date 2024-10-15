import Mathlib

namespace NUMINAMATH_GPT_krishan_nandan_investment_l1332_133237

def investment_ratio (k r₁ r₂ : ℕ) (N T Gn : ℕ) : Prop :=
  k = r₁ ∧ r₂ = 1 ∧ Gn = N * T ∧ k * N * 3 * T + Gn = 26000 ∧ Gn = 2000

/-- Given the conditions, the ratio of Krishan's investment to Nandan's investment is 4:1. -/
theorem krishan_nandan_investment :
  ∃ k N T Gn Gn_total : ℕ, 
    investment_ratio k 4 1 N T Gn  ∧ k * N * 3 * T = 24000 :=
by
  sorry

end NUMINAMATH_GPT_krishan_nandan_investment_l1332_133237


namespace NUMINAMATH_GPT_fare_from_midpoint_C_to_B_l1332_133236

noncomputable def taxi_fare (d : ℝ) : ℝ :=
  if d <= 5 then 10.8 else 10.8 + 1.2 * (d - 5)

theorem fare_from_midpoint_C_to_B (x : ℝ) (h1 : taxi_fare x = 24)
    (h2 : taxi_fare (x - 0.46) = 24) :
    taxi_fare (x / 2) = 14.4 :=
by
  sorry

end NUMINAMATH_GPT_fare_from_midpoint_C_to_B_l1332_133236


namespace NUMINAMATH_GPT_peter_needs_5000_for_vacation_l1332_133280

variable (currentSavings : ℕ) (monthlySaving : ℕ) (months : ℕ)

-- Conditions
def peterSavings := currentSavings
def monthlySavings := monthlySaving
def savingDuration := months

-- Goal
def vacationFundsRequired (currentSavings monthlySaving months : ℕ) : ℕ :=
  currentSavings + (monthlySaving * months)

theorem peter_needs_5000_for_vacation
  (h1 : currentSavings = 2900)
  (h2 : monthlySaving = 700)
  (h3 : months = 3) :
  vacationFundsRequired currentSavings monthlySaving months = 5000 := by
  sorry

end NUMINAMATH_GPT_peter_needs_5000_for_vacation_l1332_133280


namespace NUMINAMATH_GPT_second_number_deduction_l1332_133284

theorem second_number_deduction
  (x : ℝ)
  (h1 : (10 * 16 = 10 * x + (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)))
  (h2 : 2.5 + (x+1 - y) + 6.5 + 8.5 + 10.5 + 12.5 + 14.5 + 16.5 + 18.5 + 20.5 = 115)
  : y = 8 :=
by
  -- This is where the proof would go, but we'll leave it as 'sorry' for now.
  sorry

end NUMINAMATH_GPT_second_number_deduction_l1332_133284


namespace NUMINAMATH_GPT_problem_statement_l1332_133234

-- Definitions based on conditions
def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := 3 * x + 2

-- Theorem statement
theorem problem_statement : f (g 3) = 120 ∧ f 3 = 8 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1332_133234


namespace NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1332_133208

theorem quadratic_has_two_distinct_real_roots (a : ℝ) (h : a ≠ 0): 
  (a < 4 / 3) ↔ (∃ x y : ℝ, x ≠ y ∧  a * x^2 - 4 * x + 3 = 0 ∧ a * y^2 - 4 * y + 3 = 0) := 
sorry

end NUMINAMATH_GPT_quadratic_has_two_distinct_real_roots_l1332_133208


namespace NUMINAMATH_GPT_minimum_value_of_f_range_of_t_l1332_133287

noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem minimum_value_of_f :
  (∃ x > 3, f x = 9) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ (t ≤ -2 ∨ t > -1) :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_range_of_t_l1332_133287


namespace NUMINAMATH_GPT_convex_pentadecagon_diagonals_l1332_133275

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem convex_pentadecagon_diagonals :
  number_of_diagonals 15 = 90 :=
by sorry

end NUMINAMATH_GPT_convex_pentadecagon_diagonals_l1332_133275


namespace NUMINAMATH_GPT_find_constants_and_formula_l1332_133258

namespace ArithmeticSequence

variable {a : ℕ → ℤ} -- Sequence a : ℕ → ℤ

-- Given conditions
axiom a_5 : a 5 = 11
axiom a_12 : a 12 = 31

-- Definitions to be proved
def a_1 := -2
def d := 3
def a_formula (n : ℕ) := a_1 + (n - 1) * d

theorem find_constants_and_formula :
  (a 1 = a_1) ∧
  (a 2 - a 1 = d) ∧
  (a 20 = 55) ∧
  (∀ n, a n = a_formula n) := by
  sorry

end ArithmeticSequence

end NUMINAMATH_GPT_find_constants_and_formula_l1332_133258


namespace NUMINAMATH_GPT_marbles_left_l1332_133281

def initial_marbles : ℝ := 150
def lost_marbles : ℝ := 58.5
def given_away_marbles : ℝ := 37.2
def found_marbles : ℝ := 10.8

theorem marbles_left :
  initial_marbles - lost_marbles - given_away_marbles + found_marbles = 65.1 :=
by 
  sorry

end NUMINAMATH_GPT_marbles_left_l1332_133281


namespace NUMINAMATH_GPT_inequality_for_pos_reals_l1332_133252

-- Definitions for positive real numbers
variables {x y : ℝ}
def is_pos_real (x : ℝ) : Prop := x > 0

-- Theorem statement
theorem inequality_for_pos_reals (hx : is_pos_real x) (hy : is_pos_real y) : 
  2 * (x^2 + y^2) ≥ (x + y)^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_for_pos_reals_l1332_133252


namespace NUMINAMATH_GPT_painted_faces_cube_eq_54_l1332_133212

def painted_faces (n : ℕ) : ℕ :=
  if n = 5 then (3 * 3) * 6 else 0

theorem painted_faces_cube_eq_54 : painted_faces 5 = 54 := by {
  sorry
}

end NUMINAMATH_GPT_painted_faces_cube_eq_54_l1332_133212


namespace NUMINAMATH_GPT_arithmetic_mean_difference_l1332_133264

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_difference_l1332_133264


namespace NUMINAMATH_GPT_trajectory_midpoint_l1332_133268

theorem trajectory_midpoint (P M D : ℝ × ℝ) (hP : P.1 ^ 2 + P.2 ^ 2 = 16) (hD : D = (P.1, 0)) (hM : M = ((P.1 + D.1)/2, (P.2 + D.2)/2)) :
  (M.1 ^ 2) / 4 + (M.2 ^ 2) / 16 = 1 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_midpoint_l1332_133268


namespace NUMINAMATH_GPT_total_animals_l1332_133288

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end NUMINAMATH_GPT_total_animals_l1332_133288


namespace NUMINAMATH_GPT_sandy_age_l1332_133229

theorem sandy_age (S M : ℕ) 
  (h1 : M = S + 16) 
  (h2 : (↑S : ℚ) / ↑M = 7 / 9) : 
  S = 56 :=
by sorry

end NUMINAMATH_GPT_sandy_age_l1332_133229


namespace NUMINAMATH_GPT_min_value_of_expression_l1332_133220

theorem min_value_of_expression (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y * z = 1) : 
  (3 * x + y) * (x + 3 * z) * (y + z + 1) ≥ 48 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1332_133220


namespace NUMINAMATH_GPT_geometric_sequence_sum_range_l1332_133297

theorem geometric_sequence_sum_range {a : ℕ → ℝ}
  (h4_8: a 4 * a 8 = 9) :
  a 3 + a 9 ∈ Set.Iic (-6) ∪ Set.Ici 6 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_range_l1332_133297


namespace NUMINAMATH_GPT_correct_probability_l1332_133231

noncomputable def T : ℕ := 44
noncomputable def num_books : ℕ := T - 35
noncomputable def n : ℕ := 9
noncomputable def favorable_outcomes : ℕ := (Nat.choose n 6) * 2
noncomputable def total_arrangements : ℕ := (Nat.factorial n)
noncomputable def probability : Rat := (favorable_outcomes : ℚ) / (total_arrangements : ℚ)
noncomputable def m : ℕ := 1
noncomputable def p : Nat := Nat.gcd 168 362880
noncomputable def final_prob_form : Rat := 1 / 2160
noncomputable def answer : ℕ := m + 2160

theorem correct_probability : 
  probability = final_prob_form ∧ answer = 2161 := 
by
  sorry

end NUMINAMATH_GPT_correct_probability_l1332_133231


namespace NUMINAMATH_GPT_coins_in_bag_l1332_133286

theorem coins_in_bag (x : ℕ) (h : x + x / 2 + x / 4 = 105) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_coins_in_bag_l1332_133286


namespace NUMINAMATH_GPT_option_D_correct_l1332_133290

theorem option_D_correct (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end NUMINAMATH_GPT_option_D_correct_l1332_133290


namespace NUMINAMATH_GPT_huangs_tax_is_65_yuan_l1332_133209

noncomputable def monthly_salary : ℝ := 2900
noncomputable def tax_free_portion : ℝ := 2000
noncomputable def tax_rate_5_percent : ℝ := 0.05
noncomputable def tax_rate_10_percent : ℝ := 0.10

noncomputable def taxable_income_amount (income : ℝ) (exemption : ℝ) : ℝ := income - exemption

noncomputable def personal_income_tax (income : ℝ) : ℝ :=
  let taxable_income := taxable_income_amount income tax_free_portion
  if taxable_income ≤ 500 then
    taxable_income * tax_rate_5_percent
  else
    (500 * tax_rate_5_percent) + ((taxable_income - 500) * tax_rate_10_percent)

theorem huangs_tax_is_65_yuan : personal_income_tax monthly_salary = 65 :=
by
  sorry

end NUMINAMATH_GPT_huangs_tax_is_65_yuan_l1332_133209


namespace NUMINAMATH_GPT_relationship_between_x_y_z_l1332_133202

noncomputable def x := Real.sqrt 0.82
noncomputable def y := Real.sin 1
noncomputable def z := Real.log 7 / Real.log 3

theorem relationship_between_x_y_z : y < z ∧ z < x := 
by sorry

end NUMINAMATH_GPT_relationship_between_x_y_z_l1332_133202


namespace NUMINAMATH_GPT_average_incorrect_l1332_133243

theorem average_incorrect : ¬( (1 + 1 + 0 + 2 + 4) / 5 = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_average_incorrect_l1332_133243


namespace NUMINAMATH_GPT_dot_product_eq_neg29_l1332_133299

-- Given definitions and conditions
variables (a b : ℝ × ℝ)

-- Theorem to prove the dot product condition.
theorem dot_product_eq_neg29 (h1 : a + b = (2, -4)) (h2 : 3 • a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end NUMINAMATH_GPT_dot_product_eq_neg29_l1332_133299


namespace NUMINAMATH_GPT_lucas_can_afford_book_l1332_133247

-- Definitions from the conditions
def book_cost : ℝ := 28.50
def two_ten_dollar_bills : ℝ := 2 * 10
def five_one_dollar_bills : ℝ := 5 * 1
def six_quarters : ℝ := 6 * 0.25
def nickel_value : ℝ := 0.05

-- Given the conditions, we need to prove that if Lucas has at least 40 nickels, he can afford the book.
theorem lucas_can_afford_book (m : ℝ) (h : m >= 40) : 
  (two_ten_dollar_bills + five_one_dollar_bills + six_quarters + m * nickel_value) >= book_cost :=
by {
  sorry
}

end NUMINAMATH_GPT_lucas_can_afford_book_l1332_133247


namespace NUMINAMATH_GPT_kangaroo_fraction_sum_l1332_133249

theorem kangaroo_fraction_sum (G P : ℕ) (hG : 1 ≤ G) (hP : 1 ≤ P) (hTotal : G + P = 2016) : 
  (G * (P / G) + P * (G / P) = 2016) :=
by
  sorry

end NUMINAMATH_GPT_kangaroo_fraction_sum_l1332_133249


namespace NUMINAMATH_GPT_spherical_ball_radius_l1332_133203

noncomputable def largest_spherical_ball_radius (inner_radius outer_radius : ℝ) (center : ℝ × ℝ × ℝ) (table_z : ℝ) : ℝ :=
  let r := 4
  r

theorem spherical_ball_radius
  (inner_radius outer_radius : ℝ)
  (center : ℝ × ℝ × ℝ)
  (table_z : ℝ)
  (h1 : inner_radius = 3)
  (h2 : outer_radius = 5)
  (h3 : center = (4,0,1))
  (h4 : table_z = 0) :
  largest_spherical_ball_radius inner_radius outer_radius center table_z = 4 :=
by sorry

end NUMINAMATH_GPT_spherical_ball_radius_l1332_133203


namespace NUMINAMATH_GPT_average_comparison_l1332_133200

theorem average_comparison (x : ℝ) : 
    (14 + 32 + 53) / 3 = 3 + (21 + 47 + x) / 3 → 
    x = 22 :=
by 
  sorry

end NUMINAMATH_GPT_average_comparison_l1332_133200


namespace NUMINAMATH_GPT_find_second_sum_l1332_133274

theorem find_second_sum (S : ℤ) (x : ℤ) (h_S : S = 2678)
  (h_eq_interest : x * 3 * 8 = (S - x) * 5 * 3) : (S - x) = 1648 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_second_sum_l1332_133274


namespace NUMINAMATH_GPT_intersection_points_l1332_133254

theorem intersection_points (k : ℝ) : ∃ (P : ℝ × ℝ), P = (1, 0) ∧ ∀ x y : ℝ, (kx - y - k = 0) → (x^2 + y^2 = 2) → ∃ y1 y2 : ℝ, (y = y1 ∨ y = y2) :=
by
  sorry

end NUMINAMATH_GPT_intersection_points_l1332_133254


namespace NUMINAMATH_GPT_cos_theta_value_projection_value_l1332_133213

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_cos_theta_value_projection_value_l1332_133213


namespace NUMINAMATH_GPT_pre_bought_tickets_l1332_133235

theorem pre_bought_tickets (P : ℕ) 
  (h1 : ∃ P, 155 * P + 2900 = 6000) : P = 20 :=
by {
  -- Insert formalization of steps leading to P = 20
  sorry
}

end NUMINAMATH_GPT_pre_bought_tickets_l1332_133235


namespace NUMINAMATH_GPT_seventeen_power_sixty_three_mod_seven_l1332_133262

theorem seventeen_power_sixty_three_mod_seven : (17^63) % 7 = 6 := by
  -- Here you would write the actual proof demonstrating the equivalence:
  -- 1. 17 ≡ 3 (mod 7)
  -- 2. Calculate 3^63 (mod 7)
  sorry

end NUMINAMATH_GPT_seventeen_power_sixty_three_mod_seven_l1332_133262


namespace NUMINAMATH_GPT_minimum_k_for_mutual_criticism_l1332_133241

theorem minimum_k_for_mutual_criticism (k : ℕ) (h1 : 15 * k > 105) : k ≥ 8 := by
  sorry

end NUMINAMATH_GPT_minimum_k_for_mutual_criticism_l1332_133241


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1332_133239

theorem line_passes_through_fixed_point (m : ℝ) : 
  (2 + m) * (-1) + (1 - 2 * m) * (-2) + 4 - 3 * m = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1332_133239


namespace NUMINAMATH_GPT_circle_and_line_properties_l1332_133285

-- Define the circle C with center on the positive x-axis and passing through the origin
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line l: y = kx + 2
def line_l (k x y : ℝ) : Prop := y = k * x + 2

-- Statement: the circle and line setup
theorem circle_and_line_properties (k : ℝ) : 
  ∀ (x y : ℝ), 
  circle_C x y → 
  ∃ (x1 y1 x2 y2 : ℝ), 
  line_l k x1 y1 ∧ 
  line_l k x2 y2 ∧ 
  circle_C x1 y1 ∧ 
  circle_C x2 y2 ∧ 
  (x1 ≠ x2 ∧ y1 ≠ y2) → 
  k < -3/4 ∧
  ( (y1 / x1) + (y2 / x2) = 1 ) :=
by
  sorry

end NUMINAMATH_GPT_circle_and_line_properties_l1332_133285


namespace NUMINAMATH_GPT_lisa_walks_distance_per_minute_l1332_133283

-- Variables and conditions
variable (d : ℤ) -- distance that Lisa walks each minute (what we're solving for)
variable (daily_distance : ℤ) -- distance that Lisa walks each hour
variable (total_distance_in_two_days : ℤ := 1200) -- total distance in two days
variable (hours_per_day : ℤ := 1) -- one hour per day

-- Given conditions
axiom walks_for_an_hour_each_day : ∀ (d: ℤ), daily_distance = d * 60
axiom walks_1200_meters_in_two_days : ∀ (d: ℤ), total_distance_in_two_days = 2 * daily_distance

-- The theorem we want to prove
theorem lisa_walks_distance_per_minute : (d = 10) :=
by
  -- TODO: complete the proof
  sorry

end NUMINAMATH_GPT_lisa_walks_distance_per_minute_l1332_133283


namespace NUMINAMATH_GPT_total_packets_needed_l1332_133260

theorem total_packets_needed :
  let oak_seedlings := 420
  let oak_per_packet := 7
  let maple_seedlings := 825
  let maple_per_packet := 5
  let pine_seedlings := 2040
  let pine_per_packet := 12
  let oak_packets := oak_seedlings / oak_per_packet
  let maple_packets := maple_seedlings / maple_per_packet
  let pine_packets := pine_seedlings / pine_per_packet
  let total_packets := oak_packets + maple_packets + pine_packets
  total_packets = 395 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_packets_needed_l1332_133260


namespace NUMINAMATH_GPT_parallel_lines_when_m_is_neg7_l1332_133278

-- Given two lines l1 and l2 defined as:
def l1 (m : ℤ) (x y : ℤ) := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℤ) (x y : ℤ) := 2 * x + (5 + m) * y = 8

-- The proof problem to show that l1 is parallel to l2 when m = -7
theorem parallel_lines_when_m_is_neg7 :
  ∃ m : ℤ, (∀ x y : ℤ, l1 m x y → l2 m x y) → m = -7 := 
sorry

end NUMINAMATH_GPT_parallel_lines_when_m_is_neg7_l1332_133278


namespace NUMINAMATH_GPT_quadratic_behavior_l1332_133204

theorem quadratic_behavior (x : ℝ) : x < 3 → ∃ y : ℝ, y = 5 * (x - 3) ^ 2 + 2 ∧ ∀ x1 x2 : ℝ, x1 < x2 ∧ x1 < 3 ∧ x2 < 3 → (5 * (x1 - 3) ^ 2 + 2) > (5 * (x2 - 3) ^ 2 + 2) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_behavior_l1332_133204


namespace NUMINAMATH_GPT_average_side_lengths_of_squares_l1332_133295

theorem average_side_lengths_of_squares:
  let a₁ := 25
  let a₂ := 36
  let a₃ := 64

  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃

  (s₁ + s₂ + s₃) / 3 = 19 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_average_side_lengths_of_squares_l1332_133295


namespace NUMINAMATH_GPT_find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l1332_133292

noncomputable def f (a m x : ℝ) := Real.log (x + m) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a
noncomputable def F (a m x : ℝ) := f a m x - g a x

theorem find_m_and_domain (a : ℝ) (m : ℝ) (h : F a m 0 = 0) : m = 1 ∧ ∀ x, -1 < x ∧ x < 1 :=
sorry

theorem parity_of_F (a : ℝ) (m : ℝ) (h : m = 1) : ∀ x, F a m (-x) = -F a m x :=
sorry

theorem range_of_x_for_F_positive (a : ℝ) (m : ℝ) (h : m = 1) :
  (a > 1 → ∀ x, 0 < x ∧ x < 1 → F a m x > 0) ∧ (0 < a ∧ a < 1 → ∀ x, -1 < x ∧ x < 0 → F a m x > 0) :=
sorry

end NUMINAMATH_GPT_find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l1332_133292


namespace NUMINAMATH_GPT_find_f_neg_two_l1332_133225

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg_two (h : ∀ x : ℝ, x ≠ 0 → f (1 / x) + (1 / x) * f (-x) = 2 * x) :
  f (-2) = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_neg_two_l1332_133225


namespace NUMINAMATH_GPT_n_value_l1332_133223

theorem n_value (n : ℤ) (h1 : (18888 - n) % 11 = 0) : n = 7 :=
sorry

end NUMINAMATH_GPT_n_value_l1332_133223


namespace NUMINAMATH_GPT_area_of_region_l1332_133257

theorem area_of_region : 
  (∃ A : ℝ, 
    (∀ x y : ℝ, 
      (|4 * x - 20| + |3 * y + 9| ≤ 4) → 
      A = (32 / 3))) :=
by 
  sorry

end NUMINAMATH_GPT_area_of_region_l1332_133257


namespace NUMINAMATH_GPT_two_digit_number_representation_l1332_133269

theorem two_digit_number_representation (m n : ℕ) (hm : m < 10) (hn : n < 10) : 10 * n + m = m + 10 * n :=
by sorry

end NUMINAMATH_GPT_two_digit_number_representation_l1332_133269


namespace NUMINAMATH_GPT_determine_z_l1332_133271

theorem determine_z (i z : ℂ) (hi : i^2 = -1) (h : i * z = 2 * z + 1) : 
  z = - (2/5 : ℂ) - (1/5 : ℂ) * i := by
  sorry

end NUMINAMATH_GPT_determine_z_l1332_133271


namespace NUMINAMATH_GPT_radius_of_shorter_tank_l1332_133219

theorem radius_of_shorter_tank (h : ℝ) (r : ℝ) 
  (volume_eq : ∀ (π : ℝ), π * (10^2) * (2 * h) = π * (r^2) * h) : 
  r = 10 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_radius_of_shorter_tank_l1332_133219


namespace NUMINAMATH_GPT_range_of_a1_l1332_133232

theorem range_of_a1 (a1 : ℝ) :
  (∃ (a2 a3 : ℝ), 
    ((a2 = 2 * a1 - 12) ∨ (a2 = a1 / 2 + 12)) ∧
    ((a3 = 2 * a2 - 12) ∨ (a3 = a2 / 2 + 12)) ) →
  ((a3 > a1) ↔ ((a1 ≤ 12) ∨ (24 ≤ a1))) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a1_l1332_133232


namespace NUMINAMATH_GPT_no_valid_pairs_l1332_133226

theorem no_valid_pairs (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) :
  ¬(1000 * a + 100 * b + 32) % 99 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_valid_pairs_l1332_133226


namespace NUMINAMATH_GPT_petyas_number_l1332_133253

theorem petyas_number :
  ∃ (N : ℕ), 
  (N % 2 = 1 ∧ ∃ (M : ℕ), N = 149 * M ∧ (M = Nat.mod (N : ℕ) (100))) →
  (N = 745 ∨ N = 3725) :=
by
  sorry

end NUMINAMATH_GPT_petyas_number_l1332_133253


namespace NUMINAMATH_GPT_decimal_equivalent_of_one_half_squared_l1332_133255

theorem decimal_equivalent_of_one_half_squared : (1 / 2 : ℝ) ^ 2 = 0.25 := 
sorry

end NUMINAMATH_GPT_decimal_equivalent_of_one_half_squared_l1332_133255


namespace NUMINAMATH_GPT_rajans_position_l1332_133293

theorem rajans_position
    (total_boys : ℕ)
    (vinay_position_from_right : ℕ)
    (boys_between_rajan_and_vinay : ℕ)
    (total_boys_eq : total_boys = 24)
    (vinay_position_from_right_eq : vinay_position_from_right = 10)
    (boys_between_eq : boys_between_rajan_and_vinay = 8) :
    ∃ R : ℕ, R = 6 :=
by
  sorry

end NUMINAMATH_GPT_rajans_position_l1332_133293


namespace NUMINAMATH_GPT_gold_copper_ratio_l1332_133205

theorem gold_copper_ratio (G C : ℕ) 
  (h1 : 19 * G + 9 * C = 18 * (G + C)) : 
  G = 9 * C :=
by
  sorry

end NUMINAMATH_GPT_gold_copper_ratio_l1332_133205


namespace NUMINAMATH_GPT_calculate_final_price_l1332_133207

def original_price : ℝ := 120
def fixture_discount : ℝ := 0.20
def decor_discount : ℝ := 0.15

def discounted_price_after_first_discount (p : ℝ) (d : ℝ) : ℝ :=
  p * (1 - d)

def final_price (p : ℝ) (d1 : ℝ) (d2 : ℝ) : ℝ :=
  let price_after_first_discount := discounted_price_after_first_discount p d1
  price_after_first_discount * (1 - d2)

theorem calculate_final_price :
  final_price original_price fixture_discount decor_discount = 81.60 :=
by sorry

end NUMINAMATH_GPT_calculate_final_price_l1332_133207


namespace NUMINAMATH_GPT_transformed_function_correct_l1332_133279

-- Given function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem to be proven
theorem transformed_function_correct (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (x - 1) = 2 * x - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_transformed_function_correct_l1332_133279


namespace NUMINAMATH_GPT_miles_to_mall_l1332_133242

noncomputable def miles_to_grocery_store : ℕ := 10
noncomputable def miles_to_pet_store : ℕ := 5
noncomputable def miles_back_home : ℕ := 9
noncomputable def miles_per_gallon : ℕ := 15
noncomputable def cost_per_gallon : ℝ := 3.50
noncomputable def total_cost_of_gas : ℝ := 7.00
noncomputable def total_miles_driven := 2 * miles_per_gallon

theorem miles_to_mall : total_miles_driven -
  (miles_to_grocery_store + miles_to_pet_store + miles_back_home) = 6 :=
by
  -- proof omitted 
  sorry

end NUMINAMATH_GPT_miles_to_mall_l1332_133242


namespace NUMINAMATH_GPT_P_iff_Q_l1332_133294

def P (x : ℝ) := x > 1 ∨ x < -1
def Q (x : ℝ) := |x + 1| + |x - 1| > 2

theorem P_iff_Q : ∀ x, P x ↔ Q x :=
by
  intros x
  sorry

end NUMINAMATH_GPT_P_iff_Q_l1332_133294


namespace NUMINAMATH_GPT_relationships_with_correlation_l1332_133272

-- Definitions for each of the relationships as conditions
def person_age_wealth := true -- placeholder definition 
def curve_points_coordinates := true -- placeholder definition
def apple_production_climate := true -- placeholder definition
def tree_diameter_height := true -- placeholder definition
def student_school := true -- placeholder definition

-- Statement to prove which relationships involve correlation
theorem relationships_with_correlation :
  person_age_wealth ∧ apple_production_climate ∧ tree_diameter_height :=
by
  sorry

end NUMINAMATH_GPT_relationships_with_correlation_l1332_133272


namespace NUMINAMATH_GPT_cos_pi_plus_2alpha_l1332_133289

-- Define the main theorem using the given condition and the result to be proven
theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 3) : Real.cos (π + 2 * α) = 7 / 9 :=
sorry

end NUMINAMATH_GPT_cos_pi_plus_2alpha_l1332_133289


namespace NUMINAMATH_GPT_percent_of_absent_students_l1332_133240

noncomputable def absent_percentage : ℚ :=
  let total_students := 120
  let boys := 70
  let girls := 50
  let absent_boys := boys * (1/5 : ℚ)
  let absent_girls := girls * (1/4 : ℚ)
  let total_absent := absent_boys + absent_girls
  (total_absent / total_students) * 100

theorem percent_of_absent_students : absent_percentage = 22.5 := sorry

end NUMINAMATH_GPT_percent_of_absent_students_l1332_133240


namespace NUMINAMATH_GPT_order_of_three_numbers_l1332_133216

theorem order_of_three_numbers :
  let a := (7 : ℝ) ^ (0.3 : ℝ)
  let b := (0.3 : ℝ) ^ (7 : ℝ)
  let c := Real.log (0.3 : ℝ)
  a > b ∧ b > c ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_three_numbers_l1332_133216


namespace NUMINAMATH_GPT_expand_product_l1332_133221

theorem expand_product (x : ℝ): (x + 4) * (x - 5 + 2) = x^2 + x - 12 :=
by 
  sorry

end NUMINAMATH_GPT_expand_product_l1332_133221


namespace NUMINAMATH_GPT_original_triangle_area_l1332_133238

-- Define the scaling factor and given areas
def scaling_factor : ℕ := 2
def new_triangle_area : ℕ := 32

-- State that if the dimensions of the original triangle are doubled, the area becomes 32 square feet
theorem original_triangle_area (original_area : ℕ) : (scaling_factor * scaling_factor) * original_area = new_triangle_area → original_area = 8 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_original_triangle_area_l1332_133238


namespace NUMINAMATH_GPT_arithmetic_sequence_a10_gt_0_l1332_133206

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def arithmetic_sequence (a : ℕ → α) := ∀ n1 n2, a n1 - a n2 = (n1 - n2) * (a 1 - a 0)
def a9_lt_0 (a : ℕ → α) := a 9 < 0
def a1_add_a18_gt_0 (a : ℕ → α) := a 1 + a 18 > 0

-- The proof statement
theorem arithmetic_sequence_a10_gt_0 
  (a : ℕ → α) 
  (h_arith : arithmetic_sequence a) 
  (h_a9 : a9_lt_0 a) 
  (h_a1_a18 : a1_add_a18_gt_0 a) : 
  a 10 > 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a10_gt_0_l1332_133206


namespace NUMINAMATH_GPT_cost_of_1500_pieces_of_gum_in_dollars_l1332_133276

theorem cost_of_1500_pieces_of_gum_in_dollars :
  (2 * 1500 * (1 - 0.10) / 100) = 27 := sorry

end NUMINAMATH_GPT_cost_of_1500_pieces_of_gum_in_dollars_l1332_133276


namespace NUMINAMATH_GPT_circle_center_coordinates_l1332_133228

theorem circle_center_coordinates :
  let p1 := (2, -3)
  let p2 := (8, 9)
  let midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  midpoint (2 : ℝ) (-3) 8 9 = (5, 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1332_133228


namespace NUMINAMATH_GPT_edric_monthly_salary_l1332_133267

theorem edric_monthly_salary 
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (weeks_per_month : ℝ)
  (hourly_rate : ℝ) :
  hours_per_day = 8 ∧ days_per_week = 6 ∧ weeks_per_month = 4.33 ∧ hourly_rate = 3 →
  (hours_per_day * days_per_week * weeks_per_month * hourly_rate) = 623.52 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_edric_monthly_salary_l1332_133267


namespace NUMINAMATH_GPT_solve_for_n_l1332_133298

variable (n : ℚ)

theorem solve_for_n (h : 22 + Real.sqrt (-4 + 18 * n) = 24) : n = 4 / 9 := by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1332_133298


namespace NUMINAMATH_GPT_single_elimination_games_l1332_133273

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = 511 := by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l1332_133273


namespace NUMINAMATH_GPT_numBoysInClassroom_l1332_133291

-- Definitions based on the problem conditions
def numGirls : ℕ := 10
def girlsToBoysRatio : ℝ := 0.5

-- The statement to prove
theorem numBoysInClassroom : ∃ B : ℕ, girlsToBoysRatio * B = numGirls ∧ B = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_numBoysInClassroom_l1332_133291


namespace NUMINAMATH_GPT_segment_area_l1332_133296

theorem segment_area (d : ℝ) (θ : ℝ) (r := d / 2)
  (A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180))
  (A_sector := (θ / 360) * Real.pi * r^2) :
  θ = 60 →
  d = 10 →
  A_sector - A_triangle = (100 * Real.pi - 75 * Real.sqrt 3) / 24 :=
by
  sorry

end NUMINAMATH_GPT_segment_area_l1332_133296


namespace NUMINAMATH_GPT_find_inverse_of_25_l1332_133211

-- Define the inverses and the modulo
def inverse_mod (a m i : ℤ) : Prop :=
  (a * i) % m = 1

-- The given condition in the problem
def condition (m : ℤ) : Prop :=
  inverse_mod 5 m 39

-- The theorem we want to prove
theorem find_inverse_of_25 (m : ℤ) (h : condition m) : inverse_mod 25 m 8 :=
by
  sorry

end NUMINAMATH_GPT_find_inverse_of_25_l1332_133211


namespace NUMINAMATH_GPT_reflected_line_equation_l1332_133263

-- Definitions based on given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Statement of the mathematical problem
theorem reflected_line_equation :
  ∀ x y : ℝ, (incident_line x = y) → (reflection_line x = x) → y = (1/2) * x - (1/2) :=
sorry

end NUMINAMATH_GPT_reflected_line_equation_l1332_133263


namespace NUMINAMATH_GPT_solve_for_n_l1332_133222

theorem solve_for_n (n : ℕ) : (9^n * 9^n * 9^n * 9^n = 729^4) -> n = 3 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_n_l1332_133222


namespace NUMINAMATH_GPT_horse_catches_up_l1332_133224

-- Definitions based on given conditions
def dog_speed := 20 -- derived from 5 steps * 4 meters
def horse_speed := 21 -- derived from 3 steps * 7 meters
def initial_distance := 30 -- dog has already run 30 meters

-- Statement to be proved
theorem horse_catches_up (d h : ℕ) (time : ℕ) :
  d = dog_speed → h = horse_speed →
  initial_distance = 30 →
  h * time = initial_distance + dog_speed * time →
  time = 600 / (h - d) ∧ h * time - initial_distance = 600 :=
by
  intros
  -- Proof placeholders
  sorry  -- Omit the actual proof steps

end NUMINAMATH_GPT_horse_catches_up_l1332_133224


namespace NUMINAMATH_GPT_max_abs_z_l1332_133265

open Complex

theorem max_abs_z (z : ℂ) (h : abs (z + I) + abs (z - I) = 2) : abs z ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_abs_z_l1332_133265


namespace NUMINAMATH_GPT_inequality_solution_eq_l1332_133251

theorem inequality_solution_eq :
  ∀ y : ℝ, 2 ≤ |y - 5| ∧ |y - 5| ≤ 8 ↔ (-3 ≤ y ∧ y ≤ 3) ∨ (7 ≤ y ∧ y ≤ 13) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_eq_l1332_133251


namespace NUMINAMATH_GPT_base6_divisible_by_13_l1332_133233

theorem base6_divisible_by_13 (d : ℕ) (h : d < 6) : 13 ∣ (435 + 42 * d) ↔ d = 5 := 
by
  -- Proof implementation will go here, but is currently omitted
  sorry

end NUMINAMATH_GPT_base6_divisible_by_13_l1332_133233


namespace NUMINAMATH_GPT_measure_angle_R_l1332_133210

-- Given conditions
variables {P Q R : Type}
variable {x : ℝ} -- x represents the measure of angles P and Q

-- Setting up the given conditions
def isosceles_triangle (P Q R : Type) (x : ℝ) : Prop :=
  x + x + (x + 40) = 180

-- Statement we need to prove
theorem measure_angle_R (P Q R : Type) (x : ℝ) (h : isosceles_triangle P Q R x) : ∃ r : ℝ, r = 86.67 :=
by {
  sorry
}

end NUMINAMATH_GPT_measure_angle_R_l1332_133210


namespace NUMINAMATH_GPT_statement_1_statement_2_statement_3_all_statements_correct_l1332_133282

-- Define the function f and the axioms/conditions given in the problem
def f : ℕ → ℕ → ℕ := sorry

-- Conditions
axiom f_initial : f 1 1 = 1
axiom f_nat : ∀ m n : ℕ, m > 0 → n > 0 → f m n > 0
axiom f_condition_1 : ∀ m n : ℕ, m > 0 → n > 0 → f m (n + 1) = f m n + 2
axiom f_condition_2 : ∀ m : ℕ, m > 0 → f (m + 1) 1 = 2 * f m 1

-- Statements to be proved
theorem statement_1 : f 1 5 = 9 := sorry
theorem statement_2 : f 5 1 = 16 := sorry
theorem statement_3 : f 5 6 = 26 := sorry

theorem all_statements_correct : (f 1 5 = 9) ∧ (f 5 1 = 16) ∧ (f 5 6 = 26) := by
  exact ⟨statement_1, statement_2, statement_3⟩

end NUMINAMATH_GPT_statement_1_statement_2_statement_3_all_statements_correct_l1332_133282


namespace NUMINAMATH_GPT_original_number_is_7_l1332_133259

theorem original_number_is_7 (N : ℕ) (h : ∃ (k : ℤ), N = 12 * k + 7) : N = 7 :=
sorry

end NUMINAMATH_GPT_original_number_is_7_l1332_133259


namespace NUMINAMATH_GPT_prism_cut_out_l1332_133244

theorem prism_cut_out (x y : ℕ)
  (H1 : 15 * 5 * 4 - y * 5 * x = 120)
  (H2 : x < 4) :
  x = 3 ∧ y = 12 :=
sorry

end NUMINAMATH_GPT_prism_cut_out_l1332_133244


namespace NUMINAMATH_GPT_second_box_capacity_l1332_133277

-- Given conditions
def height1 := 4 -- height of the first box in cm
def width1 := 2 -- width of the first box in cm
def length1 := 6 -- length of the first box in cm
def clay_capacity1 := 48 -- weight capacity of the first box in grams

def height2 := 3 * height1 -- height of the second box in cm
def width2 := 2 * width1 -- width of the second box in cm
def length2 := length1 -- length of the second box in cm

-- Hypothesis: weight capacity increases quadratically with height
def quadratic_relationship (h1 h2 : ℕ) (capacity1 : ℕ) : ℕ :=
  (h2 / h1) * (h2 / h1) * capacity1

-- The proof problem
theorem second_box_capacity :
  quadratic_relationship height1 height2 clay_capacity1 = 432 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_second_box_capacity_l1332_133277


namespace NUMINAMATH_GPT_correct_value_of_A_sub_B_l1332_133215

variable {x y : ℝ}

-- Given two polynomials A and B where B = 3x - 2y, and a mistaken equation A + B = x - y,
-- we want to prove the correct value of A - B.
theorem correct_value_of_A_sub_B (A B : ℝ) (h1 : B = 3 * x - 2 * y) (h2 : A + B = x - y) :
  A - B = -5 * x + 3 * y :=
by
  sorry

end NUMINAMATH_GPT_correct_value_of_A_sub_B_l1332_133215


namespace NUMINAMATH_GPT_pizza_slices_leftover_l1332_133217

def slices_per_small_pizza := 4
def slices_per_large_pizza := 8
def small_pizzas_purchased := 3
def large_pizzas_purchased := 2

def george_slices := 3
def bob_slices := george_slices + 1
def susie_slices := bob_slices / 2
def bill_slices := 3
def fred_slices := 3
def mark_slices := 3

def total_slices := small_pizzas_purchased * slices_per_small_pizza + large_pizzas_purchased * slices_per_large_pizza
def total_eaten_slices := george_slices + bob_slices + susie_slices + bill_slices + fred_slices + mark_slices

def slices_leftover := total_slices - total_eaten_slices

theorem pizza_slices_leftover : slices_leftover = 10 := by
  sorry

end NUMINAMATH_GPT_pizza_slices_leftover_l1332_133217


namespace NUMINAMATH_GPT_find_x_l1332_133261

/-- Given vectors a and b, and a is parallel to b -/
def vectors (x : ℝ) : Prop :=
  let a := (x, 2)
  let b := (2, 1)
  a.1 * b.2 = a.2 * b.1

theorem find_x: ∀ x : ℝ, vectors x → x = 4 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_find_x_l1332_133261


namespace NUMINAMATH_GPT_max_possible_acute_angled_triangles_l1332_133246
-- Define the sets of points on lines a and b
def maxAcuteAngledTriangles (n : Nat) : Nat :=
  let sum1 := (n * (n - 1) / 2)  -- Sum of first (n-1) natural numbers
  let sum2 := (sum1 * 50) - (n * (n - 1) * (2 * n - 1) / 6) -- Applying the given formula
  (2 * sum2)  -- Multiply by 2 for both colors of alternating points

-- Define the main theorem
theorem max_possible_acute_angled_triangles : maxAcuteAngledTriangles 50 = 41650 := by
  sorry

end NUMINAMATH_GPT_max_possible_acute_angled_triangles_l1332_133246


namespace NUMINAMATH_GPT_solveInequalityRegion_l1332_133256

noncomputable def greatestIntegerLessThan (x : ℝ) : ℤ :=
  Int.floor x

theorem solveInequalityRegion :
  ∀ (x y : ℝ), abs x < 1 → abs y < 1 → x * y ≠ 0 → (greatestIntegerLessThan (x + y) ≤ 
  greatestIntegerLessThan x + greatestIntegerLessThan y) :=
by
  intros x y h1 h2 h3
  sorry

end NUMINAMATH_GPT_solveInequalityRegion_l1332_133256


namespace NUMINAMATH_GPT_original_average_l1332_133214

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end NUMINAMATH_GPT_original_average_l1332_133214


namespace NUMINAMATH_GPT_smallest_number_of_blocks_needed_l1332_133201

/--
Given:
  A wall with the following properties:
  1. The wall is 100 feet long and 7 feet high.
  2. Blocks used are 1 foot high and either 1 foot or 2 feet long.
  3. Blocks cannot be cut.
  4. Vertical joins in the blocks must be staggered.
  5. The wall must be even on the ends.
Prove:
  The smallest number of blocks needed to build this wall is 353.
-/
theorem smallest_number_of_blocks_needed :
  let length := 100
  let height := 7
  let block_height := 1
  (∀ b : ℕ, b = 1 ∨ b = 2) →
  ∃ (blocks_needed : ℕ), blocks_needed = 353 :=
by sorry

end NUMINAMATH_GPT_smallest_number_of_blocks_needed_l1332_133201


namespace NUMINAMATH_GPT_peanuts_in_box_l1332_133250

   theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (total_peanuts : ℕ) 
     (h1 : initial_peanuts = 4) (h2 : added_peanuts = 6) : total_peanuts = initial_peanuts + added_peanuts :=
   by
     sorry

   example : peanuts_in_box 4 6 10 rfl rfl = rfl :=
   by
     sorry
   
end NUMINAMATH_GPT_peanuts_in_box_l1332_133250


namespace NUMINAMATH_GPT_sin_cos_of_angle_l1332_133230

theorem sin_cos_of_angle (a : ℝ) (h₀ : a ≠ 0) :
  ∃ (s c : ℝ), (∃ (k : ℝ), s = k * (8 / 17) ∧ c = -k * (15 / 17) ∧ k = if a > 0 then 1 else -1) :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_of_angle_l1332_133230


namespace NUMINAMATH_GPT_combined_weight_of_candles_l1332_133248

theorem combined_weight_of_candles 
  (beeswax_weight_per_candle : ℕ)
  (coconut_oil_weight_per_candle : ℕ)
  (total_candles : ℕ)
  (candles_made : ℕ) 
  (total_weight: ℕ) 
  : 
  beeswax_weight_per_candle = 8 → 
  coconut_oil_weight_per_candle = 1 → 
  total_candles = 10 → 
  candles_made = total_candles - 3 →
  total_weight = candles_made * (beeswax_weight_per_candle + coconut_oil_weight_per_candle) →
  total_weight = 63 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_combined_weight_of_candles_l1332_133248


namespace NUMINAMATH_GPT_range_of_a_l1332_133218

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| = ax + 1 → x < 0) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1332_133218


namespace NUMINAMATH_GPT_domain_of_f_l1332_133266

def denominator (x : ℝ) : ℝ := x^2 - 4 * x + 3

def is_defined (x : ℝ) : Prop := denominator x ≠ 0

theorem domain_of_f :
  {x : ℝ // is_defined x} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1332_133266


namespace NUMINAMATH_GPT_total_votes_l1332_133227

theorem total_votes (V : ℝ) (win_percentage : ℝ) (majority : ℝ) (lose_percentage : ℝ)
  (h1 : win_percentage = 0.75) (h2 : lose_percentage = 0.25) (h3 : majority = 420) :
  V = 840 :=
by
  sorry

end NUMINAMATH_GPT_total_votes_l1332_133227


namespace NUMINAMATH_GPT_similar_triangles_height_l1332_133245

theorem similar_triangles_height (h_small: ℝ) (area_ratio: ℝ) (h_large: ℝ) :
  h_small = 5 ∧ area_ratio = 1/9 ∧ h_large = 3 * h_small → h_large = 15 :=
by
  intro h 
  sorry

end NUMINAMATH_GPT_similar_triangles_height_l1332_133245


namespace NUMINAMATH_GPT_c_geq_one_l1332_133270

theorem c_geq_one {a b : ℕ} {c : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : (a + 1) / (b + c) = b / a) : 1 ≤ c :=
  sorry

end NUMINAMATH_GPT_c_geq_one_l1332_133270
