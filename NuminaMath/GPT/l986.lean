import Mathlib

namespace NUMINAMATH_GPT_circle_and_parabola_no_intersection_l986_98614

theorem circle_and_parabola_no_intersection (m : ℝ) (h : m ≠ 0) :
  (m > 0 ∨ m < -4) ↔
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x = 0) → (y^2 = 4 * m * x) → x ≠ -m := 
sorry

end NUMINAMATH_GPT_circle_and_parabola_no_intersection_l986_98614


namespace NUMINAMATH_GPT_problem_solution_l986_98647

theorem problem_solution : (3.242 * 14) / 100 = 0.45388 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l986_98647


namespace NUMINAMATH_GPT_john_sarah_money_total_l986_98642

theorem john_sarah_money_total (j_money s_money : ℚ) (H1 : j_money = 5/8) (H2 : s_money = 7/16) :
  (j_money + s_money : ℚ) = 1.0625 := 
by
  sorry

end NUMINAMATH_GPT_john_sarah_money_total_l986_98642


namespace NUMINAMATH_GPT_exists_integers_for_expression_l986_98627

theorem exists_integers_for_expression (n : ℤ) : 
  ∃ a b c d : ℤ, n = a^2 + b^2 - c^2 - d^2 := 
sorry

end NUMINAMATH_GPT_exists_integers_for_expression_l986_98627


namespace NUMINAMATH_GPT_smallest_a_value_l986_98630

theorem smallest_a_value {a b c : ℝ} :
  (∃ (a b c : ℝ), (∀ x, (a * (x - 1/2)^2 - 5/4 = a * x^2 + b * x + c)) ∧ a > 0 ∧ ∃ n : ℤ, a + b + c = n)
  → (∃ (a : ℝ), a = 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_a_value_l986_98630


namespace NUMINAMATH_GPT_walking_west_is_negative_l986_98655

-- Definitions based on conditions
def east (m : Int) : Int := m
def west (m : Int) : Int := -m

-- Proof statement (no proof required, so use "sorry")
theorem walking_west_is_negative (m : Int) (h : east 8 = 8) : west 10 = -10 :=
by
  sorry

end NUMINAMATH_GPT_walking_west_is_negative_l986_98655


namespace NUMINAMATH_GPT_exactly_two_pass_probability_l986_98671

theorem exactly_two_pass_probability (PA PB PC : ℚ) (hPA : PA = 2 / 3) (hPB : PB = 3 / 4) (hPC : PC = 2 / 5) :
  ((PA * PB * (1 - PC)) + (PA * (1 - PB) * PC) + ((1 - PA) * PB * PC) = 7 / 15) := by
  sorry

end NUMINAMATH_GPT_exactly_two_pass_probability_l986_98671


namespace NUMINAMATH_GPT_complex_number_solution_l986_98608

theorem complex_number_solution (a b : ℤ) (z : ℂ) (h1 : z = a + b * Complex.I) (h2 : z^3 = 2 + 11 * Complex.I) : a + b = 3 :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l986_98608


namespace NUMINAMATH_GPT_n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l986_98603

theorem n_squared_plus_m_squared_odd_implies_n_plus_m_not_even (n m : ℤ) (h : (n^2 + m^2) % 2 = 1) : (n + m) % 2 ≠ 0 := by
  sorry

end NUMINAMATH_GPT_n_squared_plus_m_squared_odd_implies_n_plus_m_not_even_l986_98603


namespace NUMINAMATH_GPT_width_of_beam_l986_98619

theorem width_of_beam (L W k : ℝ) (h1 : L = k * W) (h2 : 250 = k * 1.5) : 
  (k = 166.6667) → (583.3333 = 166.6667 * W) → W = 3.5 :=
by 
  intro hk1 
  intro h583
  sorry

end NUMINAMATH_GPT_width_of_beam_l986_98619


namespace NUMINAMATH_GPT_percent_spent_on_other_items_l986_98670

def total_amount_spent (T : ℝ) : ℝ := T
def clothing_percent (p : ℝ) : Prop := p = 0.45
def food_percent (p : ℝ) : Prop := p = 0.45
def clothing_tax (t : ℝ) (T : ℝ) : ℝ := 0.05 * (0.45 * T)
def food_tax (t : ℝ) (T : ℝ) : ℝ := 0.0 * (0.45 * T)
def other_items_tax (p : ℝ) (T : ℝ) : ℝ := 0.10 * (p * T)
def total_tax (T : ℝ) (tax : ℝ) : Prop := tax = 0.0325 * T

theorem percent_spent_on_other_items (T : ℝ) (p_clothing p_food x : ℝ) (tax : ℝ) 
  (h1 : clothing_percent p_clothing) (h2 : food_percent p_food)
  (h3 : clothing_tax tax T = 0.05 * (0.45 * T))
  (h4 : food_tax tax T = 0.0)
  (h5 : other_items_tax x T = 0.10 * (x * T))
  (h6 : total_tax T (clothing_tax tax T + food_tax tax T + other_items_tax x T)) : 
  x = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_percent_spent_on_other_items_l986_98670


namespace NUMINAMATH_GPT_fraction_addition_l986_98674

theorem fraction_addition : (2 / 5) + (3 / 8) = 31 / 40 := 
by {
  sorry
}

end NUMINAMATH_GPT_fraction_addition_l986_98674


namespace NUMINAMATH_GPT_evaluate_expression_l986_98653

theorem evaluate_expression : 
  (1 / 4 - 1 / 6) / (1 / 3 - 1 / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l986_98653


namespace NUMINAMATH_GPT_circle_radius_l986_98600

theorem circle_radius (A : ℝ) (r : ℝ) (hA : A = 121 * Real.pi) (hArea : A = Real.pi * r^2) : r = 11 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l986_98600


namespace NUMINAMATH_GPT_sarah_score_l986_98612

theorem sarah_score (j g s : ℕ) 
  (h1 : g = 2 * j) 
  (h2 : s = g + 50) 
  (h3 : (s + g + j) / 3 = 110) : 
  s = 162 := 
by 
  sorry

end NUMINAMATH_GPT_sarah_score_l986_98612


namespace NUMINAMATH_GPT_rainfall_ratio_l986_98688

theorem rainfall_ratio (rain_15_days : ℕ) (total_rain : ℕ) (days_in_month : ℕ) (rain_per_day_first_15 : ℕ) :
  rain_per_day_first_15 * 15 = rain_15_days →
  rain_15_days + (days_in_month - 15) * (rain_per_day_first_15 * 2) = total_rain →
  days_in_month = 30 →
  total_rain = 180 →
  rain_per_day_first_15 = 4 →
  2 = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_rainfall_ratio_l986_98688


namespace NUMINAMATH_GPT_sum_of_coordinates_of_B_is_zero_l986_98663

structure Point where
  x : Int
  y : Int

def translation_to_right (P : Point) (n : Int) : Point :=
  { x := P.x + n, y := P.y }

def translation_down (P : Point) (n : Int) : Point :=
  { x := P.x, y := P.y - n }

def A : Point := { x := -1, y := 2 }

def B : Point := translation_down (translation_to_right A 1) 2

theorem sum_of_coordinates_of_B_is_zero :
  B.x + B.y = 0 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_B_is_zero_l986_98663


namespace NUMINAMATH_GPT_equation_1_solution_1_equation_2_solution_l986_98679

theorem equation_1_solution_1 (x : ℝ) (h : 4 * (x - 1) ^ 2 = 25) : x = 7 / 2 ∨ x = -3 / 2 := by
  sorry

theorem equation_2_solution (x : ℝ) (h : (1 / 3) * (x + 2) ^ 3 - 9 = 0) : x = 1 := by
  sorry

end NUMINAMATH_GPT_equation_1_solution_1_equation_2_solution_l986_98679


namespace NUMINAMATH_GPT_molecular_weight_l986_98685

-- Definitions of the molar masses of the elements
def molar_mass_N : ℝ := 14.01
def molar_mass_H : ℝ := 1.01
def molar_mass_I : ℝ := 126.90
def molar_mass_Ca : ℝ := 40.08
def molar_mass_S : ℝ := 32.07
def molar_mass_O : ℝ := 16.00

-- Definition of the molar masses of the compounds
def molar_mass_NH4I : ℝ := molar_mass_N + 4 * molar_mass_H + molar_mass_I
def molar_mass_CaSO4 : ℝ := molar_mass_Ca + molar_mass_S + 4 * molar_mass_O

-- Number of moles
def moles_NH4I : ℝ := 3
def moles_CaSO4 : ℝ := 2

-- Total mass calculation
def total_mass : ℝ :=
  moles_NH4I * molar_mass_NH4I + 
  moles_CaSO4 * molar_mass_CaSO4

-- Problem statement
theorem molecular_weight : total_mass = 707.15 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_l986_98685


namespace NUMINAMATH_GPT_fraction_of_hidden_sea_is_five_over_eight_l986_98651

noncomputable def cloud_fraction := 1 / 2
noncomputable def island_uncovered_fraction := 1 / 4 
noncomputable def island_covered_fraction := island_uncovered_fraction / (1 - cloud_fraction)

-- The total island area is the sum of covered and uncovered.
noncomputable def total_island_fraction := island_uncovered_fraction + island_covered_fraction 

-- The sea area covered by the cloud is half minus the fraction of the island covered by the cloud.
noncomputable def sea_covered_by_cloud := cloud_fraction - island_covered_fraction 

-- The sea occupies the remainder of the landscape not taken by the uncoveed island.
noncomputable def total_sea_fraction := 1 - island_uncovered_fraction - cloud_fraction + island_covered_fraction 

-- The sea fraction visible and not covered by clouds
noncomputable def sea_visible_not_covered := total_sea_fraction - sea_covered_by_cloud 

-- The fraction of the sea hidden by the cloud
noncomputable def sea_fraction_hidden_by_cloud := sea_covered_by_cloud / total_sea_fraction 

theorem fraction_of_hidden_sea_is_five_over_eight : sea_fraction_hidden_by_cloud = 5 / 8 := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_hidden_sea_is_five_over_eight_l986_98651


namespace NUMINAMATH_GPT_number_of_ways_to_distribute_balls_l986_98641

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end NUMINAMATH_GPT_number_of_ways_to_distribute_balls_l986_98641


namespace NUMINAMATH_GPT_power_of_p_in_product_l986_98696

theorem power_of_p_in_product (p q : ℕ) (x : ℕ) (hp : Prime p) (hq : Prime q) 
  (h : (x + 1) * 6 = 30) : x = 4 := 
by sorry

end NUMINAMATH_GPT_power_of_p_in_product_l986_98696


namespace NUMINAMATH_GPT_trigonometric_identity_l986_98648

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l986_98648


namespace NUMINAMATH_GPT_solve_for_x_l986_98668

def delta (x : ℝ) : ℝ := 5 * x + 6
def phi (x : ℝ) : ℝ := 6 * x + 5

theorem solve_for_x : ∀ x : ℝ, delta (phi x) = -1 → x = - 16 / 15 :=
by
  intro x
  intro h
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_solve_for_x_l986_98668


namespace NUMINAMATH_GPT_average_beef_sales_l986_98606

theorem average_beef_sales 
  (thursday_sales : ℕ)
  (friday_sales : ℕ)
  (saturday_sales : ℕ)
  (h_thursday : thursday_sales = 210)
  (h_friday : friday_sales = 2 * thursday_sales)
  (h_saturday : saturday_sales = 150) :
  (thursday_sales + friday_sales + saturday_sales) / 3 = 260 :=
by sorry

end NUMINAMATH_GPT_average_beef_sales_l986_98606


namespace NUMINAMATH_GPT_problem_1_problem_2_l986_98652

def f (x a : ℝ) := |x + a| + |x + 3|
def g (x : ℝ) := |x - 1| + 2

theorem problem_1 : ∀ x : ℝ, |g x| < 3 ↔ 0 < x ∧ x < 2 := 
by
  sorry

theorem problem_2 : (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) ↔ a ≥ 5 ∨ a ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l986_98652


namespace NUMINAMATH_GPT_imaginary_part_of_fraction_l986_98628

open Complex

theorem imaginary_part_of_fraction :
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  z.im = 1 :=
by
  let z := (5 * Complex.I) / (1 + 2 * Complex.I)
  show z.im = 1
  sorry

end NUMINAMATH_GPT_imaginary_part_of_fraction_l986_98628


namespace NUMINAMATH_GPT_option2_is_cheaper_l986_98615

def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

def final_price_option1 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.10
  apply_discount price_after_second_discount 0.05

def final_price_option2 (initial_price : ℝ) : ℝ :=
  let price_after_first_discount := apply_discount initial_price 0.30
  let price_after_second_discount := apply_discount price_after_first_discount 0.05
  apply_discount price_after_second_discount 0.15

theorem option2_is_cheaper (initial_price : ℝ) (h : initial_price = 12000) :
  final_price_option2 initial_price = 6783 ∧ final_price_option1 initial_price = 7182 → 6783 < 7182 :=
by
  intros
  sorry

end NUMINAMATH_GPT_option2_is_cheaper_l986_98615


namespace NUMINAMATH_GPT_slope_of_line_l986_98658

-- Defining the parametric equations of the line
def parametric_x (t : ℝ) : ℝ := 3 + 4 * t
def parametric_y (t : ℝ) : ℝ := 4 - 5 * t

-- Stating the problem in Lean: asserting the slope of the line
theorem slope_of_line : 
  (∃ (m : ℝ), ∀ t : ℝ, parametric_y t = m * parametric_x t + (4 - 3 * m)) 
  → (∃ m : ℝ, m = -5 / 4) :=
  by sorry

end NUMINAMATH_GPT_slope_of_line_l986_98658


namespace NUMINAMATH_GPT_light_travel_distance_120_years_l986_98624

theorem light_travel_distance_120_years :
  let annual_distance : ℝ := 9.46e12
  let years : ℝ := 120
  (annual_distance * years) = 1.1352e15 := 
by
  sorry

end NUMINAMATH_GPT_light_travel_distance_120_years_l986_98624


namespace NUMINAMATH_GPT_point_P_in_first_quadrant_l986_98643

def lies_in_first_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : lies_in_first_quadrant 2 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_point_P_in_first_quadrant_l986_98643


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l986_98602

theorem problem1 : 25 - 9 + (-12) - (-7) = 4 := by
  sorry

theorem problem2 : (1 / 9) * (-2)^3 / ((2 / 3)^2) = -2 := by
  sorry

theorem problem3 : ((5 / 12) + (2 / 3) - (3 / 4)) * (-12) = -4 := by
  sorry

theorem problem4 : -(1^4) + (-2) / (-1/3) - |(-9)| = -4 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l986_98602


namespace NUMINAMATH_GPT_arithmetic_geom_sequences_l986_98694

theorem arithmetic_geom_sequences
  (a : ℕ → ℤ)
  (b : ℕ → ℤ)
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_geom : ∃ q, ∀ n, b (n + 1) = b n * q)
  (h1 : a 2 + a 3 = 14)
  (h2 : a 4 - a 1 = 6)
  (h3 : b 2 = a 1)
  (h4 : b 3 = a 3) :
  (∀ n, a n = 2 * n + 2) ∧ (∃ m, b 6 = a m ∧ m = 31) := sorry

end NUMINAMATH_GPT_arithmetic_geom_sequences_l986_98694


namespace NUMINAMATH_GPT_smallest_x_consecutive_cubes_l986_98639

theorem smallest_x_consecutive_cubes :
  ∃ (u v w x : ℕ), u < v ∧ v < w ∧ w < x ∧ u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ (u^3 + v^3 + w^3 = x^3) ∧ (x = 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_x_consecutive_cubes_l986_98639


namespace NUMINAMATH_GPT_maximum_q_minus_r_l986_98673

theorem maximum_q_minus_r : 
  ∀ q r : ℕ, (1027 = 23 * q + r) ∧ (q > 0) ∧ (r > 0) → q - r ≤ 29 := 
by
  sorry

end NUMINAMATH_GPT_maximum_q_minus_r_l986_98673


namespace NUMINAMATH_GPT_sum_of_decimals_is_fraction_l986_98616

def decimal_to_fraction_sum : ℚ :=
  (1 / 10) + (2 / 100) + (3 / 1000) + (4 / 10000) + (5 / 100000) + (6 / 1000000) + (7 / 10000000)

theorem sum_of_decimals_is_fraction :
  decimal_to_fraction_sum = 1234567 / 10000000 :=
by sorry

end NUMINAMATH_GPT_sum_of_decimals_is_fraction_l986_98616


namespace NUMINAMATH_GPT_kayak_rental_cost_l986_98660

variable (K : ℕ) -- the cost of a kayak rental per day
variable (x : ℕ) -- the number of kayaks rented

-- Conditions
def canoe_cost_per_day : ℕ := 11
def total_revenue : ℕ := 460
def canoes_more_than_kayaks : ℕ := 5

def ratio_condition : Prop := 4 * x = 3 * (x + 5)
def total_revenue_condition : Prop := canoe_cost_per_day * (x + 5) + K * x = total_revenue

-- Main statement
theorem kayak_rental_cost :
  ratio_condition x →
  total_revenue_condition K x →
  K = 16 := by sorry

end NUMINAMATH_GPT_kayak_rental_cost_l986_98660


namespace NUMINAMATH_GPT_probability_two_roads_at_least_5_miles_long_l986_98659

-- Probabilities of roads being at least 5 miles long
def prob_A_B := 3 / 4
def prob_B_C := 2 / 3
def prob_C_D := 1 / 2

-- Theorem: Probability of at least two roads being at least 5 miles long
theorem probability_two_roads_at_least_5_miles_long :
  prob_A_B * prob_B_C * (1 - prob_C_D) +
  prob_A_B * prob_C_D * (1 - prob_B_C) +
  (1 - prob_A_B) * prob_B_C * prob_C_D +
  prob_A_B * prob_B_C * prob_C_D = 11 / 24 := 
by
  sorry -- Proof goes here

end NUMINAMATH_GPT_probability_two_roads_at_least_5_miles_long_l986_98659


namespace NUMINAMATH_GPT_lower_limit_for_a_l986_98678

theorem lower_limit_for_a 
  {k : ℤ} 
  (a b : ℤ) 
  (h1 : k ≤ a) 
  (h2 : a < 17) 
  (h3 : 3 < b) 
  (h4 : b < 29) 
  (h5 : 3.75 = 4 - 0.25) 
  : (7 ≤ a) :=
sorry

end NUMINAMATH_GPT_lower_limit_for_a_l986_98678


namespace NUMINAMATH_GPT_range_of_a_l986_98676

theorem range_of_a 
  (a : ℝ)
  (H1 : ∀ x : ℝ, -2 < x ∧ x < 3 → -2 < x ∧ x < a)
  (H2 : ¬(∀ x : ℝ, -2 < x ∧ x < a → -2 < x ∧ x < 3)) :
  3 < a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l986_98676


namespace NUMINAMATH_GPT_path_bound_l986_98620

/-- Definition of P_k: the number of non-intersecting paths of length k starting from point O on a grid 
    where each cell has side length 1. -/
def P_k (k : ℕ) : ℕ := sorry  -- This would normally be defined through some combinatorial method

/-- The main theorem stating the required proof statement. -/
theorem path_bound (k : ℕ) : (P_k k : ℝ) / (3^k : ℝ) < 2 := sorry

end NUMINAMATH_GPT_path_bound_l986_98620


namespace NUMINAMATH_GPT_linear_equation_with_two_variables_l986_98622

def equation (a x y : ℝ) : ℝ := (a^2 - 4) * x^2 + (2 - 3 * a) * x + (a + 1) * y + 3 * a

theorem linear_equation_with_two_variables (a : ℝ) :
  (equation a x y = 0) ∧ (a^2 - 4 = 0) ∧ (2 - 3 * a ≠ 0) ∧ (a + 1 ≠ 0) →
  (a = 2 ∨ a = -2) :=
by sorry

end NUMINAMATH_GPT_linear_equation_with_two_variables_l986_98622


namespace NUMINAMATH_GPT_time_to_cover_same_distance_l986_98692

theorem time_to_cover_same_distance
  (a b c d : ℕ) (k : ℕ) 
  (h_k : k = 3) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_speed_eq : 3 * (a + 2 * b) = 3 * a - b) : 
  (a + 2 * b) * (c + d) / (3 * a - b) = (a + 2 * b) * (c + d) / (3 * a - b) :=
by sorry

end NUMINAMATH_GPT_time_to_cover_same_distance_l986_98692


namespace NUMINAMATH_GPT_mean_proportional_49_64_l986_98693

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end NUMINAMATH_GPT_mean_proportional_49_64_l986_98693


namespace NUMINAMATH_GPT_number_of_valid_pairs_l986_98683

theorem number_of_valid_pairs :
  ∃ (n : Nat), n = 8 ∧ 
  (∃ (a b : Int), 4 < a ∧ a < b ∧ b < 22 ∧ (4 + a + b + 22) / 4 = 13) :=
sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l986_98683


namespace NUMINAMATH_GPT_double_probability_correct_l986_98661

def is_double (a : ℕ × ℕ) : Prop := a.1 = a.2

def total_dominoes : ℕ := 13 * 13

def double_count : ℕ := 13

def double_probability := (double_count : ℚ) / total_dominoes

theorem double_probability_correct : double_probability = 13 / 169 := by
  sorry

end NUMINAMATH_GPT_double_probability_correct_l986_98661


namespace NUMINAMATH_GPT_smallest_integral_value_of_y_l986_98640

theorem smallest_integral_value_of_y :
  ∃ y : ℤ, (1 / 4 : ℝ) < y / 7 ∧ y / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 / 4 : ℝ) < z / 7 ∧ z / 7 < 2 / 3 → y ≤ z :=
by
  -- The statement is defined and the proof is left as "sorry" to illustrate that no solution steps are used directly.
  sorry

end NUMINAMATH_GPT_smallest_integral_value_of_y_l986_98640


namespace NUMINAMATH_GPT_cupcakes_for_children_l986_98644

-- Definitions for the conditions
def packs15 : Nat := 4
def packs10 : Nat := 4
def cupcakes_per_pack15 : Nat := 15
def cupcakes_per_pack10 : Nat := 10

-- Proposition to prove the total number of cupcakes is 100
theorem cupcakes_for_children :
  (packs15 * cupcakes_per_pack15) + (packs10 * cupcakes_per_pack10) = 100 := by
  sorry

end NUMINAMATH_GPT_cupcakes_for_children_l986_98644


namespace NUMINAMATH_GPT_brianna_sandwiches_l986_98645

theorem brianna_sandwiches (meats : ℕ) (cheeses : ℕ) (h_meats : meats = 8) (h_cheeses : cheeses = 7) :
  (Nat.choose meats 2) * (Nat.choose cheeses 1) = 196 := 
by
  rw [h_meats, h_cheeses]
  norm_num
  sorry

end NUMINAMATH_GPT_brianna_sandwiches_l986_98645


namespace NUMINAMATH_GPT_required_run_rate_l986_98682

def initial_run_rate : ℝ := 3.2
def overs_completed : ℝ := 10
def target_runs : ℝ := 282
def remaining_overs : ℝ := 50

theorem required_run_rate :
  (target_runs - initial_run_rate * overs_completed) / remaining_overs = 5 := 
by
  sorry

end NUMINAMATH_GPT_required_run_rate_l986_98682


namespace NUMINAMATH_GPT_complete_square_solution_l986_98672

theorem complete_square_solution (x : ℝ) (h : x^2 - 4 * x + 2 = 0) : (x - 2)^2 = 2 := 
by sorry

end NUMINAMATH_GPT_complete_square_solution_l986_98672


namespace NUMINAMATH_GPT_min_distance_from_circle_to_line_l986_98657

noncomputable def circle_center : (ℝ × ℝ) := (3, -1)
noncomputable def circle_radius : ℝ := 2

def on_circle (P : ℝ × ℝ) : Prop := (P.1 - circle_center.1) ^ 2 + (P.2 + circle_center.2) ^ 2 = circle_radius ^ 2
def on_line (Q : ℝ × ℝ) : Prop := Q.1 = -3

theorem min_distance_from_circle_to_line (P Q : ℝ × ℝ)
  (h1 : on_circle P) (h2 : on_line Q) : dist P Q = 4 := 
sorry

end NUMINAMATH_GPT_min_distance_from_circle_to_line_l986_98657


namespace NUMINAMATH_GPT_factorization_of_polynomial_l986_98665

theorem factorization_of_polynomial : 
  (x^2 + 6 * x + 9 - 64 * x^4) = (-8 * x^2 + x + 3) * (8 * x^2 + x + 3) :=
by sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l986_98665


namespace NUMINAMATH_GPT_correct_propositions_l986_98646

variable (A : Set ℝ)
variable (oplus : ℝ → ℝ → ℝ)

def condition_a1 : Prop := ∀ a b : ℝ, a ∈ A → b ∈ A → (oplus a b) ∈ A
def condition_a2 : Prop := ∀ a : ℝ, a ∈ A → (oplus a a) = 0
def condition_a3 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus (oplus a b) c) = (oplus a c) + (oplus b c) + c

def proposition_1 : Prop := 0 ∈ A
def proposition_2 : Prop := (1 ∈ A) → (oplus (oplus 1 1) 1) = 0
def proposition_3 : Prop := ∀ a : ℝ, a ∈ A → (oplus a 0) = a → a = 0
def proposition_4 : Prop := ∀ a b c : ℝ, a ∈ A → b ∈ A → c ∈ A → (oplus a 0) = a → (oplus a b) = (oplus c b) → a = c

theorem correct_propositions 
  (h1 : condition_a1 A oplus) 
  (h2 : condition_a2 A oplus)
  (h3 : condition_a3 A oplus) : 
  (proposition_1 A) ∧ (¬proposition_2 A oplus) ∧ (proposition_3 A oplus) ∧ (proposition_4 A oplus) := by
  sorry

end NUMINAMATH_GPT_correct_propositions_l986_98646


namespace NUMINAMATH_GPT_total_sours_is_123_l986_98604

noncomputable def cherry_sours := 32
noncomputable def lemon_sours := 40 -- Derived from the ratio 4/5 = 32/x
noncomputable def orange_sours := 24 -- 25% of the total sours in the bag after adding them
noncomputable def grape_sours := 27 -- Derived from the ratio 3/2 = 40/y

theorem total_sours_is_123 :
  cherry_sours + lemon_sours + orange_sours + grape_sours = 123 :=
by
  sorry

end NUMINAMATH_GPT_total_sours_is_123_l986_98604


namespace NUMINAMATH_GPT_poly_coeff_difference_l986_98634

theorem poly_coeff_difference :
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (2 + x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 →
  a = 16 →
  1 = a - a_1 + a_2 - a_3 + a_4 →
  a_2 - a_1 + a_4 - a_3 = -15 :=
by
  intros a a_1 a_2 a_3 a_4 h_poly h_a h_eq
  sorry

end NUMINAMATH_GPT_poly_coeff_difference_l986_98634


namespace NUMINAMATH_GPT_second_pirate_gets_diamond_l986_98601

theorem second_pirate_gets_diamond (coins_bag1 coins_bag2 : ℕ) :
  (coins_bag1 ≤ 1 ∧ coins_bag2 ≤ 1) ∨ (coins_bag1 > 1 ∨ coins_bag2 > 1) →
  (∃ n k : ℕ, n % 2 = 0 → (coins_bag1 + n) = (coins_bag2 + k)) :=
sorry

end NUMINAMATH_GPT_second_pirate_gets_diamond_l986_98601


namespace NUMINAMATH_GPT_union_of_A_and_B_l986_98690

noncomputable def A : Set ℝ := {1, 2, 3}
noncomputable def B : Set ℝ := {x | x < 3}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 3} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l986_98690


namespace NUMINAMATH_GPT_area_of_shaded_region_l986_98656

open Real

-- Define points and squares
structure Point (α : Type*) := (x : α) (y : α)

def A := Point.mk 0 12 -- top-left corner of large square
def G := Point.mk 0 0  -- bottom-left corner of large square
def F := Point.mk 4 0  -- bottom-right corner of small square
def E := Point.mk 4 4  -- top-right corner of small square
def C := Point.mk 12 0 -- bottom-right corner of large square
def D := Point.mk 3 0  -- intersection of AF extended with the bottom edge

-- Define the length of sides
def side_small_square : ℝ := 4
def side_large_square : ℝ := 12

-- Areas calculation
def area_square (side : ℝ) : ℝ := side * side

def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height

-- Theorem statement
theorem area_of_shaded_region : area_square side_small_square - area_triangle 3 side_small_square = 10 :=
by
  rw [area_square, area_triangle]
  -- Plug in values: 4^2 - 0.5 * 3 * 4
  norm_num
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l986_98656


namespace NUMINAMATH_GPT_max_remainder_l986_98611

-- Definition of the problem
def max_remainder_condition (x : ℕ) (y : ℕ) : Prop :=
  x % 7 = y

theorem max_remainder (y : ℕ) :
  (max_remainder_condition (7 * 102 + y) y ∧ y < 7) → (y = 6 ∧ 7 * 102 + 6 = 720) :=
by
  sorry

end NUMINAMATH_GPT_max_remainder_l986_98611


namespace NUMINAMATH_GPT_shakes_indeterminable_l986_98698

theorem shakes_indeterminable (B S C x : ℝ) (h1 : 3 * B + 7 * S + C = 120) (h2 : 4 * B + x * S + C = 164.50) : ¬ (∃ B S C, ∀ x, 4 * B + x * S + C = 164.50) → false := 
by 
  sorry

end NUMINAMATH_GPT_shakes_indeterminable_l986_98698


namespace NUMINAMATH_GPT_store_profit_is_20_percent_l986_98609

variable (C : ℝ)
variable (marked_up_price : ℝ := 1.20 * C)          -- First markup price
variable (new_year_price : ℝ := 1.50 * C)           -- Second markup price
variable (discounted_price : ℝ := 1.20 * C)         -- Discounted price in February
variable (profit : ℝ := discounted_price - C)       -- Profit on items sold in February

theorem store_profit_is_20_percent (C : ℝ) : profit = 0.20 * C := 
  sorry

end NUMINAMATH_GPT_store_profit_is_20_percent_l986_98609


namespace NUMINAMATH_GPT_angle_B_in_triangle_l986_98684

theorem angle_B_in_triangle
  (a b c : ℝ)
  (h_area : 2 * (a * c * ((a^2 + c^2 - b^2) / (2 * a * c)).sin) = (a^2 + c^2 - b^2) * (Real.sqrt 3 / 6)) :
  ∃ B : ℝ, B = π / 6 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_in_triangle_l986_98684


namespace NUMINAMATH_GPT_calc1_l986_98625

theorem calc1 : (2 - Real.sqrt 3) ^ 0 - Real.sqrt 12 + Real.tan (Real.pi / 3) = 1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_calc1_l986_98625


namespace NUMINAMATH_GPT_portion_of_money_given_to_Blake_l986_98697

theorem portion_of_money_given_to_Blake
  (initial_amount : ℝ)
  (tripled_amount : ℝ)
  (sale_amount : ℝ)
  (amount_given_to_Blake : ℝ)
  (h1 : initial_amount = 20000)
  (h2 : tripled_amount = 3 * initial_amount)
  (h3 : sale_amount = tripled_amount)
  (h4 : amount_given_to_Blake = 30000) :
  amount_given_to_Blake / sale_amount = 1 / 2 :=
sorry

end NUMINAMATH_GPT_portion_of_money_given_to_Blake_l986_98697


namespace NUMINAMATH_GPT_f_increasing_on_Ioo_l986_98610

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem f_increasing_on_Ioo : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → f x1 < f x2 :=
by sorry

end NUMINAMATH_GPT_f_increasing_on_Ioo_l986_98610


namespace NUMINAMATH_GPT_necessary_not_sufficient_cond_l986_98667

theorem necessary_not_sufficient_cond (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y < 4 → xy < 4) ∧ ¬(xy < 4 → x + y < 4) :=
  by
    sorry

end NUMINAMATH_GPT_necessary_not_sufficient_cond_l986_98667


namespace NUMINAMATH_GPT_quadratic_roots_min_value_l986_98666

theorem quadratic_roots_min_value (m α β : ℝ) (h_eq : 4 * α^2 - 4 * m * α + m + 2 = 0) (h_eq2 : 4 * β^2 - 4 * m * β + m + 2 = 0) :
  (∃ m_val : ℝ, m_val = -1 ∧ α^2 + β^2 = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_min_value_l986_98666


namespace NUMINAMATH_GPT_proof_l986_98695

noncomputable def line_standard_form (t : ℝ) : Prop :=
  let (x, y) := (t + 3, 3 - t)
  x + y = 6

noncomputable def circle_standard_form (θ : ℝ) : Prop :=
  let (x, y) := (2 * Real.cos θ, 2 * Real.sin θ + 2)
  x^2 + (y - 2)^2 = 4

noncomputable def distance_center_to_line (x1 y1 : ℝ) : ℝ :=
  let (a, b, c) := (1, 1, -6)
  let num := abs (a * x1 + b * y1 + c)
  let denom := Real.sqrt (a^2 + b^2)
  num / denom

theorem proof : 
  (∀ t, line_standard_form t) ∧ 
  (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → circle_standard_form θ) ∧ 
  distance_center_to_line 0 2 = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_l986_98695


namespace NUMINAMATH_GPT_smallest_prime_reversing_to_composite_l986_98650

theorem smallest_prime_reversing_to_composite (p : ℕ) :
  p = 23 ↔ (p < 100 ∧ p ≥ 10 ∧ Nat.Prime p ∧ 
  ∃ c, c < 100 ∧ c ≥ 10 ∧ ¬ Nat.Prime c ∧ c = (p % 10) * 10 + p / 10 ∧ (p / 10 = 2 ∨ p / 10 = 3)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_reversing_to_composite_l986_98650


namespace NUMINAMATH_GPT_sin_complementary_angle_l986_98681

theorem sin_complementary_angle (θ : ℝ) (h1 : Real.tan θ = 2) (h2 : Real.cos θ < 0) : 
  Real.sin (Real.pi / 2 - θ) = -Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_GPT_sin_complementary_angle_l986_98681


namespace NUMINAMATH_GPT_cost_of_socks_l986_98669

/-- Given initial amount of $100 and cost of shirt is $24,
    find out the cost of socks if the remaining amount is $65. --/
theorem cost_of_socks
  (initial_amount : ℕ)
  (cost_of_shirt : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 100)
  (h2 : cost_of_shirt = 24)
  (h3 : remaining_amount = 65) : 
  (initial_amount - cost_of_shirt - remaining_amount) = 11 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_socks_l986_98669


namespace NUMINAMATH_GPT_point_not_on_graph_l986_98654

theorem point_not_on_graph : 
  ∀ (k : ℝ), (k ≠ 0) → (∀ x y : ℝ, y = k * x → (x, y) = (1, 2)) → ¬ (∀ x y : ℝ, y = k * x → (x, y) = (1, -2)) :=
by
  sorry

end NUMINAMATH_GPT_point_not_on_graph_l986_98654


namespace NUMINAMATH_GPT_probability_max_roll_correct_l986_98662
open Classical

noncomputable def probability_max_roll_fourth : ℚ :=
  let six_sided_max := 1 / 6
  let eight_sided_max := 3 / 4
  let ten_sided_max := 4 / 5

  let prob_A_given_B1 := (1 / 6) ^ 3
  let prob_A_given_B2 := (3 / 4) ^ 3
  let prob_A_given_B3 := (4 / 5) ^ 3

  let prob_B1 := 1 / 3
  let prob_B2 := 1 / 3
  let prob_B3 := 1 / 3

  let prob_A := prob_A_given_B1 * prob_B1 + prob_A_given_B2 * prob_B2 + prob_A_given_B3 * prob_B3

  -- Calculate probabilities with Bayes' Theorem
  let P_B1_A := (prob_A_given_B1 * prob_B1) / prob_A
  let P_B2_A := (prob_A_given_B2 * prob_B2) / prob_A
  let P_B3_A := (prob_A_given_B3 * prob_B3) / prob_A

  -- Probability of the fourth roll showing the maximum face value
  P_B1_A * six_sided_max + P_B2_A * eight_sided_max + P_B3_A * ten_sided_max

theorem probability_max_roll_correct : 
  ∃ (p q : ℕ), probability_max_roll_fourth = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 4386 :=
by sorry

end NUMINAMATH_GPT_probability_max_roll_correct_l986_98662


namespace NUMINAMATH_GPT_intersection_complement_is_l986_98635

universe u
variable {α : Type u}

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {1, 3, 5}

theorem intersection_complement_is :
  N ∩ (U \ M) = {3, 5} :=
  sorry

end NUMINAMATH_GPT_intersection_complement_is_l986_98635


namespace NUMINAMATH_GPT_parabola_directrix_l986_98691

theorem parabola_directrix (x y : ℝ) (h : y = - (1/8) * x^2) : y = 2 :=
sorry

end NUMINAMATH_GPT_parabola_directrix_l986_98691


namespace NUMINAMATH_GPT_probability_both_A_B_selected_l986_98621

-- Define the problem with the necessary conditions.
def probability_AB_selected (total_students : ℕ) (select_students : ℕ) (A B : ℕ) : ℚ :=
  if A < total_students ∧ B < total_students ∧ total_students = 5 ∧ select_students = 3 then
    let total_ways := (Nat.factorial total_students) / ((Nat.factorial select_students) * (Nat.factorial (total_students - select_students)))
    let favorable_ways := (Nat.factorial (total_students - 2)) / ((Nat.factorial (select_students - 2)) * (Nat.factorial ((total_students - 2) - (select_students - 2))))
    favorable_ways / total_ways
  else 0

theorem probability_both_A_B_selected 
  (total_students : ℕ) (select_students : ℕ) (A B : ℕ) 
  (h1 : total_students = 5) (h2 : select_students = 3) 
  (h3 : A < total_students) (h4 : B < total_students) 
  : probability_AB_selected total_students select_students A B = 3 / 10 := 
by {
  -- Insert the delta and logic to prove the theorem here.
  sorry
}

end NUMINAMATH_GPT_probability_both_A_B_selected_l986_98621


namespace NUMINAMATH_GPT_problem_I_problem_II_l986_98689

theorem problem_I (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) :
  c / a = 2 :=
sorry

theorem problem_II (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : 2 * a - a * Real.cos B = b * Real.cos A) 
  (h3 : b = 4) (h4 : Real.cos C = 1 / 4) :
  (1 / 2) * a * b * Real.sin C = Real.sqrt 15 :=
sorry

end NUMINAMATH_GPT_problem_I_problem_II_l986_98689


namespace NUMINAMATH_GPT_polynomial_sum_l986_98636

theorem polynomial_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 1 = a + a₁*(x - 1) + a₂*(x - 1)^2 + a₃*(x - 1)^3 + a₄*(x - 1)^4 + a₅*(x - 1)^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = 33 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l986_98636


namespace NUMINAMATH_GPT_three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l986_98605

-- Problem (1)
theorem three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty : 3^40 > 4^30 ∧ 4^30 > 5^20 := 
by
  sorry

-- Problem (2)
theorem sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one : 16^31 > 8^41 ∧ 8^41 > 4^61 :=
by 
  sorry

-- Problem (3)
theorem a_lt_b (a b : ℝ) (h1 : 1 < a) (h2 : 1 < b) (h3 : a^5 = 2) (h4 : b^7 = 3) : a < b :=
by
  sorry

end NUMINAMATH_GPT_three_pow_forty_gt_four_pow_thirty_gt_five_pow_twenty_sixteen_pow_thirty_one_gt_eight_pow_forty_one_gt_four_pow_sixty_one_a_lt_b_l986_98605


namespace NUMINAMATH_GPT_tan_frac_eq_one_l986_98623

open Real

-- Conditions given in the problem
def sin_frac_cond (x y : ℝ) : Prop := (sin x / sin y) + (sin y / sin x) = 4
def cos_frac_cond (x y : ℝ) : Prop := (cos x / cos y) + (cos y / cos x) = 3

-- Statement of the theorem to be proved
theorem tan_frac_eq_one (x y : ℝ) (h1 : sin_frac_cond x y) (h2 : cos_frac_cond x y) : (tan x / tan y) + (tan y / tan x) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_frac_eq_one_l986_98623


namespace NUMINAMATH_GPT_calculate_perimeter_l986_98629

noncomputable def length_square := 8
noncomputable def breadth_square := 8 -- since it's a square, length and breadth are the same
noncomputable def length_rectangle := 8
noncomputable def breadth_rectangle := 4

noncomputable def combined_length := length_square + length_rectangle
noncomputable def combined_breadth := breadth_square 

noncomputable def perimeter := 2 * (combined_length + combined_breadth)

theorem calculate_perimeter : 
  length_square = 8 ∧ 
  breadth_square = 8 ∧ 
  length_rectangle = 8 ∧ 
  breadth_rectangle = 4 ∧ 
  perimeter = 48 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_perimeter_l986_98629


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l986_98686

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l986_98686


namespace NUMINAMATH_GPT_andrew_total_appeizers_count_l986_98613

theorem andrew_total_appeizers_count :
  let hotdogs := 30
  let cheese_pops := 20
  let chicken_nuggets := 40
  hotdogs + cheese_pops + chicken_nuggets = 90 := 
by 
  sorry

end NUMINAMATH_GPT_andrew_total_appeizers_count_l986_98613


namespace NUMINAMATH_GPT_output_of_program_l986_98633

def loop_until (i S : ℕ) : ℕ :=
if i < 9 then S
else loop_until (i - 1) (S * i)

theorem output_of_program : loop_until 11 1 = 990 :=
sorry

end NUMINAMATH_GPT_output_of_program_l986_98633


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l986_98632

theorem solution_set_of_inequality_system (x : ℝ) :
  (3 * x - 1 ≥ x + 1) ∧ (x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := 
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l986_98632


namespace NUMINAMATH_GPT_product_units_digit_of_five_consecutive_l986_98687

theorem product_units_digit_of_five_consecutive (n : ℕ) : 
  ((n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) % 10) = 0 := 
sorry

end NUMINAMATH_GPT_product_units_digit_of_five_consecutive_l986_98687


namespace NUMINAMATH_GPT_Atlantic_Call_additional_charge_is_0_20_l986_98680

def United_Telephone_base_rate : ℝ := 7.00
def United_Telephone_rate_per_minute : ℝ := 0.25
def Atlantic_Call_base_rate : ℝ := 12.00
def United_Telephone_total_charge_100_minutes : ℝ := United_Telephone_base_rate + 100 * United_Telephone_rate_per_minute
def Atlantic_Call_total_charge_100_minutes (x : ℝ) : ℝ := Atlantic_Call_base_rate + 100 * x

theorem Atlantic_Call_additional_charge_is_0_20 :
  ∃ x : ℝ, United_Telephone_total_charge_100_minutes = Atlantic_Call_total_charge_100_minutes x ∧ x = 0.20 :=
by {
  -- Since United_Telephone_total_charge_100_minutes = 32.00, we need to prove:
  -- Atlantic_Call_total_charge_100_minutes 0.20 = 32.00
  sorry
}

end NUMINAMATH_GPT_Atlantic_Call_additional_charge_is_0_20_l986_98680


namespace NUMINAMATH_GPT_tom_gave_fred_balloons_l986_98638

variable (initial_balloons : ℕ) (remaining_balloons : ℕ)

def balloons_given (initial remaining : ℕ) : ℕ :=
  initial - remaining

theorem tom_gave_fred_balloons (h₀ : initial_balloons = 30) (h₁ : remaining_balloons = 14) :
  balloons_given initial_balloons remaining_balloons = 16 :=
by
  -- Here we are skipping the proof
  sorry

end NUMINAMATH_GPT_tom_gave_fred_balloons_l986_98638


namespace NUMINAMATH_GPT_age_problem_l986_98677

theorem age_problem 
  (K S E F : ℕ)
  (h1 : K = S - 5)
  (h2 : S = 2 * E)
  (h3 : E = F + 9)
  (h4 : K = 33) : 
  F = 10 :=
by 
  sorry

end NUMINAMATH_GPT_age_problem_l986_98677


namespace NUMINAMATH_GPT_solve_system_l986_98675

theorem solve_system :
  ∃ (x1 y1 x2 y2 x3 y3 : ℚ), 
    (x1 = 0 ∧ y1 = 0) ∧ 
    (x2 = -14 ∧ y2 = 6) ∧ 
    (x3 = -85/6 ∧ y3 = 35/6) ∧ 
    ((x1 + 2*y1)*(x1 + 3*y1) = x1 + y1 ∧ (2*x1 + y1)*(3*x1 + y1) = -99*(x1 + y1)) ∧ 
    ((x2 + 2*y2)*(x2 + 3*y2) = x2 + y2 ∧ (2*x2 + y2)*(3*x2 + y2) = -99*(x2 + y2)) ∧ 
    ((x3 + 2*y3)*(x3 + 3*y3) = x3 + y3 ∧ (2*x3 + y3)*(3*x3 + y3) = -99*(x3 + y3)) :=
by
  -- skips the actual proof
  sorry

end NUMINAMATH_GPT_solve_system_l986_98675


namespace NUMINAMATH_GPT_elena_fraction_left_l986_98664

variable (M : ℝ) -- Total amount of money
variable (B : ℝ) -- Total cost of all the books

-- Condition: Elena spends one-third of her money to buy half of the books
def condition : Prop := (1 / 3) * M = (1 / 2) * B

-- Goal: Fraction of the money left after buying all the books is one-third
theorem elena_fraction_left (h : condition M B) : (M - B) / M = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_elena_fraction_left_l986_98664


namespace NUMINAMATH_GPT_gain_percent_l986_98631

variable (C S : ℝ)

theorem gain_percent 
  (h : 81 * C = 45 * S) : ((4 / 5) * 100) = 80 := 
by 
  sorry

end NUMINAMATH_GPT_gain_percent_l986_98631


namespace NUMINAMATH_GPT_transform_to_100_l986_98649

theorem transform_to_100 (a b c : ℤ) (h : Int.gcd (Int.gcd a b) c = 1) :
  ∃ f : (ℤ × ℤ × ℤ → ℤ × ℤ × ℤ), (∀ p : ℤ × ℤ × ℤ,
    ∃ q : ℕ, q ≤ 5 ∧ f^[q] p = (1, 0, 0)) :=
sorry

end NUMINAMATH_GPT_transform_to_100_l986_98649


namespace NUMINAMATH_GPT_boy_lap_time_l986_98617

noncomputable def muddy_speed : ℝ := 5 * 1000 / 3600
noncomputable def sandy_speed : ℝ := 7 * 1000 / 3600
noncomputable def uphill_speed : ℝ := 4 * 1000 / 3600

noncomputable def muddy_distance : ℝ := 10
noncomputable def sandy_distance : ℝ := 15
noncomputable def uphill_distance : ℝ := 10

noncomputable def time_for_muddy : ℝ := muddy_distance / muddy_speed
noncomputable def time_for_sandy : ℝ := sandy_distance / sandy_speed
noncomputable def time_for_uphill : ℝ := uphill_distance / uphill_speed

noncomputable def total_time_for_one_side : ℝ := time_for_muddy + time_for_sandy + time_for_uphill
noncomputable def total_time_for_lap : ℝ := 4 * total_time_for_one_side

theorem boy_lap_time : total_time_for_lap = 95.656 := by
  sorry

end NUMINAMATH_GPT_boy_lap_time_l986_98617


namespace NUMINAMATH_GPT_simplify_trig_expression_l986_98607

open Real

theorem simplify_trig_expression (α : ℝ) : 
  (cos (2 * π + α) * tan (π + α)) / cos (π / 2 - α) = 1 := 
sorry

end NUMINAMATH_GPT_simplify_trig_expression_l986_98607


namespace NUMINAMATH_GPT_hypotenuse_length_l986_98618

def triangle_hypotenuse (x : ℝ) (h : ℝ) : Prop :=
  (3 * x - 3)^2 + x^2 = h^2 ∧
  (1 / 2) * x * (3 * x - 3) = 72

theorem hypotenuse_length :
  ∃ (x h : ℝ), triangle_hypotenuse x h ∧ h = Real.sqrt 505 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l986_98618


namespace NUMINAMATH_GPT_smallest_two_digit_number_l986_98637

theorem smallest_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100)
  (h2 : ∃ k : ℕ, (N - (N / 10 + (N % 10) * 10)) = k ∧ k > 0 ∧ (∃ m : ℕ, k = m * m))
  : N = 90 := 
sorry

end NUMINAMATH_GPT_smallest_two_digit_number_l986_98637


namespace NUMINAMATH_GPT_time_to_complete_together_l986_98699

-- Definitions for the given conditions
variables (x y : ℝ) (hx : x > 0) (hy : y > 0)

-- Theorem statement for the mathematically equivalent proof problem
theorem time_to_complete_together (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
   (1 : ℝ) / ((1 / x) + (1 / y)) = x * y / (x + y) :=
sorry

end NUMINAMATH_GPT_time_to_complete_together_l986_98699


namespace NUMINAMATH_GPT_floor_sqrt_50_l986_98626

theorem floor_sqrt_50 : (⌊Real.sqrt 50⌋ = 7) :=
by
  sorry

end NUMINAMATH_GPT_floor_sqrt_50_l986_98626
