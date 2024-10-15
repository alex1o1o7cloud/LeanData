import Mathlib

namespace NUMINAMATH_GPT_soda_preference_count_eq_243_l1399_139935

def total_respondents : ℕ := 540
def soda_angle : ℕ := 162
def total_circle_angle : ℕ := 360

theorem soda_preference_count_eq_243 :
  (total_respondents * soda_angle / total_circle_angle) = 243 := 
by 
  sorry

end NUMINAMATH_GPT_soda_preference_count_eq_243_l1399_139935


namespace NUMINAMATH_GPT_athlete_difference_is_30_l1399_139964

def initial_athletes : ℕ := 600
def leaving_rate : ℕ := 35
def leaving_duration : ℕ := 6
def arrival_rate : ℕ := 20
def arrival_duration : ℕ := 9

def athletes_left : ℕ := leaving_rate * leaving_duration
def new_athletes : ℕ := arrival_rate * arrival_duration
def remaining_athletes : ℕ := initial_athletes - athletes_left
def final_athletes : ℕ := remaining_athletes + new_athletes
def athlete_difference : ℕ := initial_athletes - final_athletes

theorem athlete_difference_is_30 : athlete_difference = 30 :=
by
  show athlete_difference = 30
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_athlete_difference_is_30_l1399_139964


namespace NUMINAMATH_GPT_max_regions_divided_l1399_139903

theorem max_regions_divided (n m : ℕ) (h_n : n = 10) (h_m : m = 4) (h_m_le_n : m ≤ n) : 
  ∃ r : ℕ, r = 50 :=
by
  have non_parallel_lines := n - m
  have regions_non_parallel := (non_parallel_lines * (non_parallel_lines + 1)) / 2 + 1
  have regions_parallel := m * non_parallel_lines + m
  have total_regions := regions_non_parallel + regions_parallel
  use total_regions
  sorry

end NUMINAMATH_GPT_max_regions_divided_l1399_139903


namespace NUMINAMATH_GPT_inequality_example_l1399_139998

open Real

theorem inequality_example 
    (x y z : ℝ) 
    (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1):
    (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_example_l1399_139998


namespace NUMINAMATH_GPT_limit_tanxy_over_y_l1399_139931

theorem limit_tanxy_over_y (f : ℝ×ℝ → ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x y, abs (x - 3) < δ ∧ abs y < δ → abs (f (x, y) - 3) < ε) :=
sorry

end NUMINAMATH_GPT_limit_tanxy_over_y_l1399_139931


namespace NUMINAMATH_GPT_inequality_solution_l1399_139916

theorem inequality_solution (x : ℝ) : 
  (0 < (x + 2) / ((x - 3)^3)) ↔ (x < -2 ∨ x > 3)  :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1399_139916


namespace NUMINAMATH_GPT_product_of_two_numbers_l1399_139900

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y ≠ 0) 
  (h2 : (x + y) / (x - y) = 7)
  (h3 : xy = 24 * (x - y)) : xy = 48 := 
sorry

end NUMINAMATH_GPT_product_of_two_numbers_l1399_139900


namespace NUMINAMATH_GPT_sqrt_value_l1399_139943

theorem sqrt_value {A B C : ℝ} (x y : ℝ) 
  (h1 : A = 5 * Real.sqrt (2 * x + 1)) 
  (h2 : B = 3 * Real.sqrt (x + 3)) 
  (h3 : C = Real.sqrt (10 * x + 3 * y)) 
  (h4 : A + B = C) 
  (h5 : 2 * x + 1 = x + 3) : 
  Real.sqrt (2 * y - x^2) = 14 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_value_l1399_139943


namespace NUMINAMATH_GPT_water_to_concentrate_ratio_l1399_139972

theorem water_to_concentrate_ratio (servings : ℕ) (serving_size_oz concentrate_size_oz : ℕ)
                                (cans_of_concentrate required_juice_oz : ℕ)
                                (h_servings : servings = 280)
                                (h_serving_size : serving_size_oz = 6)
                                (h_concentrate_size : concentrate_size_oz = 12)
                                (h_cans_of_concentrate : cans_of_concentrate = 35)
                                (h_required_juice : required_juice_oz = servings * serving_size_oz)
                                (h_made_juice : required_juice_oz = 1680)
                                (h_concentrate_volume : cans_of_concentrate * concentrate_size_oz = 420)
                                (h_water_volume : required_juice_oz - (cans_of_concentrate * concentrate_size_oz) = 1260)
                                (h_water_cans : 1260 / concentrate_size_oz = 105) :
                                105 / 35 = 3 :=
by
  sorry

end NUMINAMATH_GPT_water_to_concentrate_ratio_l1399_139972


namespace NUMINAMATH_GPT_sum_of_missing_angles_l1399_139922

theorem sum_of_missing_angles (angle_sum_known : ℕ) (divisor : ℕ) (total_sides : ℕ) (missing_angles_sum : ℕ)
  (h1 : angle_sum_known = 1620)
  (h2 : divisor = 180)
  (h3 : total_sides = 12)
  (h4 : angle_sum_known + missing_angles_sum = divisor * (total_sides - 2)) :
  missing_angles_sum = 180 :=
by
  -- Skipping the proof for this theorem
  sorry

end NUMINAMATH_GPT_sum_of_missing_angles_l1399_139922


namespace NUMINAMATH_GPT_tan_195_l1399_139939

theorem tan_195 (a : ℝ) (h : Real.cos 165 = a) : Real.tan 195 = - (Real.sqrt (1 - a^2)) / a := 
sorry

end NUMINAMATH_GPT_tan_195_l1399_139939


namespace NUMINAMATH_GPT_cosine_ab_ac_l1399_139970

noncomputable def vector_a := (-2, 4, -6)
noncomputable def vector_b := (0, 2, -4)
noncomputable def vector_c := (-6, 8, -10)

noncomputable def a_b : ℝ × ℝ × ℝ := (2, -2, 2)
noncomputable def a_c : ℝ × ℝ × ℝ := (-4, 4, -4)

noncomputable def ab_dot_ac : ℝ := -24

noncomputable def mag_a_b : ℝ := 2 * Real.sqrt 3
noncomputable def mag_a_c : ℝ := 4 * Real.sqrt 3

theorem cosine_ab_ac :
  (ab_dot_ac / (mag_a_b * mag_a_c) = -1) :=
sorry

end NUMINAMATH_GPT_cosine_ab_ac_l1399_139970


namespace NUMINAMATH_GPT_communication_system_connections_l1399_139910

theorem communication_system_connections (n : ℕ) (h : ∀ k < 2001, ∃ l < 2001, l ≠ k ∧ k ≠ l) :
  (∀ k < 2001, ∃ l < 2001, k ≠ l) → (n % 2 = 0 ∧ n ≤ 2000) ∨ n = 0 :=
sorry

end NUMINAMATH_GPT_communication_system_connections_l1399_139910


namespace NUMINAMATH_GPT_slope_of_parallel_line_l1399_139917

theorem slope_of_parallel_line (x y : ℝ) :
  (∃ (b : ℝ), 3 * x - 6 * y = 12) → ∀ (m₁ x₁ y₁ x₂ y₂ : ℝ), (y₁ = (1/2) * x₁ + b) ∧ (y₂ = (1/2) * x₂ + b) → (x₁ ≠ x₂) → m₁ = 1/2 :=
by 
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l1399_139917


namespace NUMINAMATH_GPT_jellybeans_left_in_jar_l1399_139902

def original_jellybeans : ℕ := 250
def class_size : ℕ := 24
def sick_children : ℕ := 2
def sick_jellybeans_each : ℕ := 7
def first_group_size : ℕ := 12
def first_group_jellybeans_each : ℕ := 5
def second_group_size : ℕ := 10
def second_group_jellybeans_each : ℕ := 4

theorem jellybeans_left_in_jar : 
  original_jellybeans - ((first_group_size * first_group_jellybeans_each) + 
  (second_group_size * second_group_jellybeans_each)) = 150 := by
  sorry

end NUMINAMATH_GPT_jellybeans_left_in_jar_l1399_139902


namespace NUMINAMATH_GPT_factorization_identity_l1399_139952

theorem factorization_identity (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2 * m - 1)^2 :=
sorry

end NUMINAMATH_GPT_factorization_identity_l1399_139952


namespace NUMINAMATH_GPT_find_line_equation_l1399_139968

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x - 4

theorem find_line_equation :
  ∃ (x₁ y₁ : ℝ), x₁ = Real.sqrt 3 ∧ y₁ = -3 ∧ ∀ x y, (line_equation x y ↔ 
  (y + 3 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3))) :=
sorry

end NUMINAMATH_GPT_find_line_equation_l1399_139968


namespace NUMINAMATH_GPT_vector_x_value_l1399_139980

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

def perpendicular (v1 v2 : ℝ × ℝ) : Prop := 
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem vector_x_value (x : ℝ) : (perpendicular (a x) b) → x = -2 / 3 := by
  intro h
  sorry

end NUMINAMATH_GPT_vector_x_value_l1399_139980


namespace NUMINAMATH_GPT_find_missing_dimension_l1399_139974

def carton_volume (l w h : ℕ) : ℕ := l * w * h

def soapbox_base_area (l w : ℕ) : ℕ := l * w

def total_base_area (n l w : ℕ) : ℕ := n * soapbox_base_area l w

def missing_dimension (carton_volume total_base_area : ℕ) : ℕ := carton_volume / total_base_area

theorem find_missing_dimension 
  (carton_l carton_w carton_h : ℕ) 
  (soapbox_l soapbox_w : ℕ) 
  (n : ℕ) 
  (h_carton_l : carton_l = 25)
  (h_carton_w : carton_w = 48)
  (h_carton_h : carton_h = 60)
  (h_soapbox_l : soapbox_l = 8)
  (h_soapbox_w : soapbox_w = 6)
  (h_n : n = 300) :
  missing_dimension (carton_volume carton_l carton_w carton_h) (total_base_area n soapbox_l soapbox_w) = 5 := 
by 
  sorry

end NUMINAMATH_GPT_find_missing_dimension_l1399_139974


namespace NUMINAMATH_GPT_connor_cats_l1399_139988

theorem connor_cats (j : ℕ) (a : ℕ) (m : ℕ) (c : ℕ) (co : ℕ) (x : ℕ) 
  (h1 : a = j / 3)
  (h2 : m = 2 * a)
  (h3 : c = a / 2)
  (h4 : c = co + 5)
  (h5 : j = 90)
  (h6 : x = j + a + m + c + co) : 
  co = 10 := 
by
  sorry

end NUMINAMATH_GPT_connor_cats_l1399_139988


namespace NUMINAMATH_GPT_total_bottles_ordered_in_april_and_may_is_1000_l1399_139958

-- Define the conditions
def casesInApril : Nat := 20
def casesInMay : Nat := 30
def bottlesPerCase : Nat := 20

-- The total number of bottles ordered in April and May
def totalBottlesOrdered : Nat := (casesInApril + casesInMay) * bottlesPerCase

-- The main statement to be proved
theorem total_bottles_ordered_in_april_and_may_is_1000 :
  totalBottlesOrdered = 1000 :=
sorry

end NUMINAMATH_GPT_total_bottles_ordered_in_april_and_may_is_1000_l1399_139958


namespace NUMINAMATH_GPT_distance_of_coming_down_stairs_l1399_139950

noncomputable def totalTimeAscendingDescending (D : ℝ) : ℝ :=
  (D / 2) + ((D + 2) / 3)

theorem distance_of_coming_down_stairs : ∃ D : ℝ, totalTimeAscendingDescending D = 4 ∧ (D + 2) = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_of_coming_down_stairs_l1399_139950


namespace NUMINAMATH_GPT_sum_abs_coeffs_expansion_l1399_139949

theorem sum_abs_coeffs_expansion (x : ℝ) :
  (|1 - 0 * x| + |1 - 3 * x| + |1 - 3^2 * x^2| + |1 - 3^3 * x^3| + |1 - 3^4 * x^4| + |1 - 3^5 * x^5| = 1024) :=
sorry

end NUMINAMATH_GPT_sum_abs_coeffs_expansion_l1399_139949


namespace NUMINAMATH_GPT_marbles_count_l1399_139963

variables {g y : ℕ}

theorem marbles_count (h1 : (g - 1)/(g + y - 1) = 1/8)
                      (h2 : g/(g + y - 3) = 1/6) :
                      g + y = 9 :=
by
-- This is just setting up the statements we need to prove the theorem. The actual proof is to be completed.
sorry

end NUMINAMATH_GPT_marbles_count_l1399_139963


namespace NUMINAMATH_GPT_find_a1_l1399_139985

-- Defining the conditions
variables (a : ℕ → ℝ)
variable (q : ℝ)
variable (h_monotone : ∀ n, a n ≥ a (n + 1)) -- Monotonically decreasing

-- Specific values from the problem
axiom h_a3 : a 3 = 1
axiom h_a2_a4 : a 2 + a 4 = 5 / 2
axiom h_geom_seq : ∀ n, a (n + 1) = a n * q  -- Geometric sequence property

-- The goal is to prove that a 1 = 4
theorem find_a1 : a 1 = 4 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_find_a1_l1399_139985


namespace NUMINAMATH_GPT_find_a_value_l1399_139904

theorem find_a_value (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : (∃ l : ℝ, ∃ f : ℝ → ℝ, f x = a^x ∧ deriv f 0 = -1)) :
  a = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_find_a_value_l1399_139904


namespace NUMINAMATH_GPT_hyperbola_eccentricity_asymptotic_lines_l1399_139956

-- Define the conditions and the proof goal:

theorem hyperbola_eccentricity_asymptotic_lines {a b c e : ℝ} 
  (h_asym : ∀ x y : ℝ, (y = x ∨ y = -x) ↔ (a = b)) 
  (h_c : c = Real.sqrt (a ^ 2 + b ^ 2))
  (h_e : e = c / a) : e = Real.sqrt 2 := sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_asymptotic_lines_l1399_139956


namespace NUMINAMATH_GPT_not_or_false_implies_or_true_l1399_139934

variable (p q : Prop)

theorem not_or_false_implies_or_true (h : ¬(p ∨ q) = False) : p ∨ q :=
by
  sorry

end NUMINAMATH_GPT_not_or_false_implies_or_true_l1399_139934


namespace NUMINAMATH_GPT_problem_l1399_139928

def f (x : ℤ) := 3 * x + 2

theorem problem : f (f (f 3)) = 107 := by
  sorry

end NUMINAMATH_GPT_problem_l1399_139928


namespace NUMINAMATH_GPT_integer_div_product_l1399_139953

theorem integer_div_product (n : ℤ) : ∃ (k : ℤ), n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end NUMINAMATH_GPT_integer_div_product_l1399_139953


namespace NUMINAMATH_GPT_ice_cream_cones_sixth_day_l1399_139993

theorem ice_cream_cones_sixth_day (cones_day1 cones_day2 cones_day3 cones_day4 cones_day5 cones_day7 : ℝ)
  (mean : ℝ) (h1 : cones_day1 = 100) (h2 : cones_day2 = 92) 
  (h3 : cones_day3 = 109) (h4 : cones_day4 = 96) 
  (h5 : cones_day5 = 103) (h7 : cones_day7 = 105) 
  (h_mean : mean = 100.1) : 
  ∃ cones_day6 : ℝ, cones_day6 = 95.7 :=
by 
  sorry

end NUMINAMATH_GPT_ice_cream_cones_sixth_day_l1399_139993


namespace NUMINAMATH_GPT_total_signs_at_intersections_l1399_139992

-- Definitions based on the given conditions
def first_intersection_signs : ℕ := 40
def second_intersection_signs : ℕ := first_intersection_signs + first_intersection_signs / 4
def third_intersection_signs : ℕ := 2 * second_intersection_signs
def fourth_intersection_signs : ℕ := third_intersection_signs - 20

-- Prove the total number of signs at the four intersections is 270
theorem total_signs_at_intersections :
  first_intersection_signs + second_intersection_signs + third_intersection_signs + fourth_intersection_signs = 270 := by
  sorry

end NUMINAMATH_GPT_total_signs_at_intersections_l1399_139992


namespace NUMINAMATH_GPT_f_four_l1399_139965

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (a b : ℝ) : f (a + b) + f (a - b) = 2 * f a + 2 * f b
axiom f_two : f 2 = 9 
axiom not_identically_zero : ¬ ∀ x : ℝ, f x = 0

theorem f_four : f 4 = 36 :=
by sorry

end NUMINAMATH_GPT_f_four_l1399_139965


namespace NUMINAMATH_GPT_largest_divisor_of_product_of_five_consecutive_integers_l1399_139937

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_product_of_five_consecutive_integers_l1399_139937


namespace NUMINAMATH_GPT_total_handshakes_l1399_139906

-- Define the conditions
def number_of_players_per_team : Nat := 11
def number_of_referees : Nat := 3
def total_number_of_players : Nat := number_of_players_per_team * 2

-- Prove the total number of handshakes
theorem total_handshakes : 
  (number_of_players_per_team * number_of_players_per_team) + (total_number_of_players * number_of_referees) = 187 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_handshakes_l1399_139906


namespace NUMINAMATH_GPT_problem_solution_l1399_139996

theorem problem_solution (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 500 / 9 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1399_139996


namespace NUMINAMATH_GPT_quadratic_eq_solutions_l1399_139951

theorem quadratic_eq_solutions : ∃ x1 x2 : ℝ, (x^2 = x) ∨ (x = 0 ∧ x = 1) := by
  sorry

end NUMINAMATH_GPT_quadratic_eq_solutions_l1399_139951


namespace NUMINAMATH_GPT_inequality_in_triangle_l1399_139926

variables {a b c : ℝ}

namespace InequalityInTriangle

-- Define the condition that a, b, c are sides of a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem inequality_in_triangle (a b c : ℝ) (h : is_triangle a b c) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) :=
sorry

end InequalityInTriangle

end NUMINAMATH_GPT_inequality_in_triangle_l1399_139926


namespace NUMINAMATH_GPT_zach_saved_money_l1399_139947

-- Definitions of known quantities
def cost_of_bike : ℝ := 100
def weekly_allowance : ℝ := 5
def mowing_earnings : ℝ := 10
def babysitting_rate : ℝ := 7
def babysitting_hours : ℝ := 2
def additional_earnings_needed : ℝ := 6

-- Calculate total earnings for this week
def total_earnings_this_week : ℝ := weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)

-- Prove that Zach has already saved $65
theorem zach_saved_money : (cost_of_bike - total_earnings_this_week - additional_earnings_needed) = 65 :=
by
  -- Sorry used as placeholder to skip the proof
  sorry

end NUMINAMATH_GPT_zach_saved_money_l1399_139947


namespace NUMINAMATH_GPT_miles_per_dollar_l1399_139941

def car_mpg : ℝ := 32
def gas_cost_per_gallon : ℝ := 4

theorem miles_per_dollar (X : ℝ) : 
  (X / gas_cost_per_gallon) * car_mpg = 8 * X :=
by
  sorry

end NUMINAMATH_GPT_miles_per_dollar_l1399_139941


namespace NUMINAMATH_GPT_ravenswood_forest_percentage_l1399_139944

def ravenswood_gnomes (westerville_gnomes : ℕ) : ℕ := 4 * westerville_gnomes
def remaining_gnomes (total_gnomes taken_percentage: ℕ) : ℕ := (total_gnomes * (100 - taken_percentage)) / 100

theorem ravenswood_forest_percentage:
  ∀ (westerville_gnomes : ℕ) (remaining : ℕ) (total_gnomes : ℕ),
  westerville_gnomes = 20 →
  total_gnomes = ravenswood_gnomes westerville_gnomes →
  remaining = 48 →
  remaining_gnomes total_gnomes 40 = remaining :=
by
  sorry

end NUMINAMATH_GPT_ravenswood_forest_percentage_l1399_139944


namespace NUMINAMATH_GPT_puzzles_sold_eq_36_l1399_139938

def n_science_kits : ℕ := 45
def n_puzzles : ℕ := n_science_kits - 9

theorem puzzles_sold_eq_36 : n_puzzles = 36 := by
  sorry

end NUMINAMATH_GPT_puzzles_sold_eq_36_l1399_139938


namespace NUMINAMATH_GPT_movie_tickets_ratio_l1399_139995

theorem movie_tickets_ratio (R H : ℕ) (hR : R = 25) (hH : H = 93) : 
  (H / R : ℚ) = 93 / 25 :=
by
  sorry

end NUMINAMATH_GPT_movie_tickets_ratio_l1399_139995


namespace NUMINAMATH_GPT_right_triangle_cos_B_l1399_139984

theorem right_triangle_cos_B (A B C : ℝ) (hC : C = 90) (hSinA : Real.sin A = 2 / 3) :
  Real.cos B = 2 / 3 :=
sorry

end NUMINAMATH_GPT_right_triangle_cos_B_l1399_139984


namespace NUMINAMATH_GPT_goldfish_equal_in_seven_months_l1399_139959

/-- Define the growth of Alice's goldfish: they triple every month. -/
def alice_goldfish (n : ℕ) : ℕ := 3 * 3 ^ n

/-- Define the growth of Bob's goldfish: they quadruple every month. -/
def bob_goldfish (n : ℕ) : ℕ := 256 * 4 ^ n

/-- The main theorem we want to prove: For Alice and Bob's goldfish count to be equal,
    it takes 7 months. -/
theorem goldfish_equal_in_seven_months : ∃ n : ℕ, alice_goldfish n = bob_goldfish n ∧ n = 7 := 
by
  sorry

end NUMINAMATH_GPT_goldfish_equal_in_seven_months_l1399_139959


namespace NUMINAMATH_GPT_base5_minus_base8_to_base10_l1399_139940

def base5_to_base10 (n : Nat) : Nat :=
  5 * 5^5 + 4 * 5^4 + 3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 0 * 5^0

def base8_to_base10 (n : Nat) : Nat :=
  4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0

theorem base5_minus_base8_to_base10 :
  (base5_to_base10 543210 - base8_to_base10 43210) = 499 :=
by
  sorry

end NUMINAMATH_GPT_base5_minus_base8_to_base10_l1399_139940


namespace NUMINAMATH_GPT_qinJiushao_value_l1399_139923

/-- A specific function f(x) with given a and b -/
def f (x : ℤ) : ℤ :=
  x^5 + 47 * x^4 - 37 * x^2 + 1

/-- Qin Jiushao algorithm to find V3 at x = -1 -/
def qinJiushao (x : ℤ) : ℤ :=
  let V0 := 1
  let V1 := V0 * x + 47
  let V2 := V1 * x + 0
  let V3 := V2 * x - 37
  V3

theorem qinJiushao_value :
  qinJiushao (-1) = 9 :=
by
  sorry

end NUMINAMATH_GPT_qinJiushao_value_l1399_139923


namespace NUMINAMATH_GPT_total_spent_after_discount_and_tax_l1399_139966

-- Define prices for each item
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define discounts and tax rates
def discount_bracelet := 0.10
def sales_tax := 0.05

-- Define the quantity of each item purchased by Paula, Olive, and Nathan
def quantity_paula_bracelets := 3
def quantity_paula_keychains := 2
def quantity_paula_coloring_books := 1
def quantity_paula_stickers := 4

def quantity_olive_coloring_books := 1
def quantity_olive_bracelets := 2
def quantity_olive_toy_cars := 1
def quantity_olive_stickers := 3

def quantity_nathan_toy_cars := 4
def quantity_nathan_stickers := 5
def quantity_nathan_keychains := 1

-- Function to calculate total cost before discount and tax
def total_cost_before_discount_and_tax (bracelets keychains coloring_books stickers toy_cars : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) +
  Float.ofNat (keychains * price_keychain) +
  Float.ofNat (coloring_books * price_coloring_book) +
  Float.ofNat (stickers * price_sticker) +
  Float.ofNat (toy_cars * price_toy_car)

-- Function to calculate discount on bracelets
def bracelet_discount (bracelets : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) * discount_bracelet

-- Function to calculate total cost after discount and before tax
def total_cost_after_discount (total_cost discount : Float) : Float :=
  total_cost - discount

-- Function to calculate total cost after tax
def total_cost_after_tax (total_cost : Float) (tax_rate : Float) : Float :=
  total_cost * (1 + tax_rate)

-- Proof statement (no proof provided, only the statement)
theorem total_spent_after_discount_and_tax : 
  total_cost_after_tax (
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_paula_bracelets quantity_paula_keychains quantity_paula_coloring_books quantity_paula_stickers 0)
      (bracelet_discount quantity_paula_bracelets)
    +
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_olive_bracelets 0 quantity_olive_coloring_books quantity_olive_stickers quantity_olive_toy_cars)
      (bracelet_discount quantity_olive_bracelets)
    +
    total_cost_before_discount_and_tax 0 quantity_nathan_keychains 0 quantity_nathan_stickers quantity_nathan_toy_cars
  ) sales_tax = 85.05 := 
sorry

end NUMINAMATH_GPT_total_spent_after_discount_and_tax_l1399_139966


namespace NUMINAMATH_GPT_range_contains_pi_div_4_l1399_139976

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end NUMINAMATH_GPT_range_contains_pi_div_4_l1399_139976


namespace NUMINAMATH_GPT_profit_calculation_l1399_139983

def Initial_Value : ℕ := 100
def Multiplier : ℕ := 3
def New_Value : ℕ := Initial_Value * Multiplier
def Profit : ℕ := New_Value - Initial_Value

theorem profit_calculation : Profit = 200 := by
  sorry

end NUMINAMATH_GPT_profit_calculation_l1399_139983


namespace NUMINAMATH_GPT_part1_part2_l1399_139920

-- Definitions from conditions
def U := ℝ
def A := {x : ℝ | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) := {x : ℝ | 5 - a < x ∧ x < a}

-- (1) If "x ∈ A" is a necessary condition for "x ∈ B", find the range of a
theorem part1 (a : ℝ) : (∀ x : ℝ, x ∈ B a → x ∈ A) → a ≤ 3 :=
by sorry

-- (2) If A ∩ B ≠ ∅, find the range of a
theorem part2 (a : ℝ) : (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → a > 5 / 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1399_139920


namespace NUMINAMATH_GPT_john_books_per_day_l1399_139961

theorem john_books_per_day (books_total : ℕ) (total_weeks : ℕ) (days_per_week : ℕ) (total_days : ℕ)
  (read_days_eq : total_days = total_weeks * days_per_week)
  (books_per_day_eq : books_total = total_days * 4) : (books_total / total_days = 4) :=
by
  -- The conditions state the following:
  -- books_total = 48 (total books read)
  -- total_weeks = 6 (total number of weeks)
  -- days_per_week = 2 (number of days John reads per week)
  -- total_days = 12 (total number of days in which John reads books)
  -- read_days_eq :- total_days = total_weeks * days_per_week
  -- books_per_day_eq :- books_total = total_days * 4
  sorry

end NUMINAMATH_GPT_john_books_per_day_l1399_139961


namespace NUMINAMATH_GPT_correct_calculation_result_l1399_139999

theorem correct_calculation_result (x : ℤ) (h : x + 44 - 39 = 63) : x + 39 - 44 = 53 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l1399_139999


namespace NUMINAMATH_GPT_min_value_expression_l1399_139981

theorem min_value_expression (x y : ℝ) : (x^2 + y^2 - 6 * x + 4 * y + 18) ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1399_139981


namespace NUMINAMATH_GPT_tetrahedron_altitudes_l1399_139907

theorem tetrahedron_altitudes (r h₁ h₂ h₃ h₄ : ℝ)
  (h₁_def : h₁ = 3 * r)
  (h₂_def : h₂ = 4 * r)
  (h₃_def : h₃ = 4 * r)
  (altitude_sum : 1/h₁ + 1/h₂ + 1/h₃ + 1/h₄ = 1/r) : 
  h₄ = 6 * r :=
by
  rw [h₁_def, h₂_def, h₃_def] at altitude_sum
  sorry

end NUMINAMATH_GPT_tetrahedron_altitudes_l1399_139907


namespace NUMINAMATH_GPT_seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l1399_139978

-- Define the sequences
def a_sq (n : ℕ) : ℕ := n ^ 2
def a_cube (n : ℕ) : ℕ := n ^ 3

-- First proof problem statement
theorem seq_satisfies_recurrence_sq :
  (a_sq 0 = 0) ∧ (a_sq 1 = 1) ∧ (a_sq 2 = 4) ∧ (a_sq 3 = 9) ∧ (a_sq 4 = 16) →
  (∀ n : ℕ, n ≥ 3 → a_sq n = 3 * a_sq (n - 1) - 3 * a_sq (n - 2) + a_sq (n - 3)) :=
by
  sorry

-- Second proof problem statement
theorem seq_satisfies_recurrence_cube :
  (a_cube 0 = 0) ∧ (a_cube 1 = 1) ∧ (a_cube 2 = 8) ∧ (a_cube 3 = 27) ∧ (a_cube 4 = 64) →
  (∀ n : ℕ, n ≥ 4 → a_cube n = 4 * a_cube (n - 1) - 6 * a_cube (n - 2) + 4 * a_cube (n - 3) - a_cube (n - 4)) :=
by
  sorry

end NUMINAMATH_GPT_seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l1399_139978


namespace NUMINAMATH_GPT_general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l1399_139921

noncomputable def arith_seq (n : ℕ) (d : ℝ) :=
  2 + (n - 1) * d

theorem general_term_formula :
  ∃ d, ∀ n, arith_seq n d = 2 ∨ arith_seq n d = 4 * n - 2 :=
by sorry

theorem no_pos_int_for_S_n_gt_40n_plus_600 :
  ∀ n, (arith_seq n 0) * n ≤ 40 * n + 600 :=
by sorry

theorem exists_pos_int_for_S_n_gt_40n_plus_600 :
  ∃ n, (arith_seq n 4) * n > 40 * n + 600 ∧ n = 31 :=
by sorry

end NUMINAMATH_GPT_general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l1399_139921


namespace NUMINAMATH_GPT_total_expenses_l1399_139945

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end NUMINAMATH_GPT_total_expenses_l1399_139945


namespace NUMINAMATH_GPT_density_change_l1399_139962

theorem density_change (V : ℝ) (Δa : ℝ) (decrease_percent : ℝ) (initial_volume : V = 27) (edge_increase : Δa = 0.9) : 
    decrease_percent = 8 := 
by 
  sorry

end NUMINAMATH_GPT_density_change_l1399_139962


namespace NUMINAMATH_GPT_hyperbola_range_of_k_l1399_139913

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x^2)/(k + 4) + (y^2)/(k - 1) = 1) → -4 < k ∧ k < 1 :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_range_of_k_l1399_139913


namespace NUMINAMATH_GPT_complex_neither_sufficient_nor_necessary_real_l1399_139954

noncomputable def quadratic_equation_real_roots (a : ℝ) : Prop := 
  (a^2 - 4 * a ≥ 0)

noncomputable def quadratic_equation_complex_roots (a : ℝ) : Prop := 
  (a^2 - 4 * (-a) < 0)

theorem complex_neither_sufficient_nor_necessary_real (a : ℝ) :
  (quadratic_equation_complex_roots a ↔ quadratic_equation_real_roots a) = false := 
sorry

end NUMINAMATH_GPT_complex_neither_sufficient_nor_necessary_real_l1399_139954


namespace NUMINAMATH_GPT_correct_calculation_result_l1399_139991

theorem correct_calculation_result :
  ∀ (A B D : ℝ),
  C = 6 →
  E = 5 →
  (A * 10 + B) * 6 + D * E = 39.6 ∨ (A * 10 + B) * 6 * D * E = 36.9 →
  (A * 10 + B) * 6 + D * E = 26.1 :=
by
  intros A B D C_eq E_eq errors
  sorry

end NUMINAMATH_GPT_correct_calculation_result_l1399_139991


namespace NUMINAMATH_GPT_ratio_of_segments_l1399_139960

theorem ratio_of_segments (a b c r s : ℝ) (h : a / b = 1 / 4)
  (h₁ : c ^ 2 = a ^ 2 + b ^ 2)
  (h₂ : r = a ^ 2 / c)
  (h₃ : s = b ^ 2 / c) :
  r / s = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_segments_l1399_139960


namespace NUMINAMATH_GPT_inequality_for_large_n_l1399_139997

theorem inequality_for_large_n (n : ℕ) (hn : n > 1) : 
  (1 / Real.exp 1 - 1 / (n * Real.exp 1)) < (1 - 1 / n) ^ n ∧ (1 - 1 / n) ^ n < (1 / Real.exp 1 - 1 / (2 * n * Real.exp 1)) :=
sorry

end NUMINAMATH_GPT_inequality_for_large_n_l1399_139997


namespace NUMINAMATH_GPT_evaluate_expression_l1399_139948

theorem evaluate_expression : (502 * 502) - (501 * 503) = 1 := sorry

end NUMINAMATH_GPT_evaluate_expression_l1399_139948


namespace NUMINAMATH_GPT_average_of_175_results_l1399_139925

theorem average_of_175_results (x y : ℕ) (hx : x = 100) (hy : y = 75) 
(a b : ℚ) (ha : a = 45) (hb : b = 65) :
  ((x * a + y * b) / (x + y) = 53.57) :=
sorry

end NUMINAMATH_GPT_average_of_175_results_l1399_139925


namespace NUMINAMATH_GPT_probability_of_selecting_green_ball_l1399_139912

def container_I :  ℕ × ℕ := (5, 5) -- (red balls, green balls)
def container_II : ℕ × ℕ := (3, 3) -- (red balls, green balls)
def container_III : ℕ × ℕ := (4, 2) -- (red balls, green balls)
def container_IV : ℕ × ℕ := (6, 6) -- (red balls, green balls)

def total_containers : ℕ := 4

def probability_of_green_ball (red_green : ℕ × ℕ) : ℚ :=
  let (red, green) := red_green
  green / (red + green)

noncomputable def combined_probability_of_green_ball : ℚ :=
  (1 / total_containers) *
  (probability_of_green_ball container_I +
   probability_of_green_ball container_II +
   probability_of_green_ball container_III +
   probability_of_green_ball container_IV)

theorem probability_of_selecting_green_ball : 
  combined_probability_of_green_ball = 11 / 24 :=
sorry

end NUMINAMATH_GPT_probability_of_selecting_green_ball_l1399_139912


namespace NUMINAMATH_GPT_fraction_doubled_unchanged_l1399_139911

theorem fraction_doubled_unchanged (x y : ℝ) (h : x ≠ y) : 
  (2 * x) / (2 * x - 2 * y) = x / (x - y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_doubled_unchanged_l1399_139911


namespace NUMINAMATH_GPT_age_of_son_l1399_139979

theorem age_of_son (D S : ℕ) (h₁ : S = D / 4) (h₂ : D - S = 27) (h₃ : D = 36) : S = 9 :=
by
  sorry

end NUMINAMATH_GPT_age_of_son_l1399_139979


namespace NUMINAMATH_GPT_first_number_is_seven_l1399_139989

variable (x y : ℝ)

theorem first_number_is_seven (h1 : x + y = 10) (h2 : 2 * x = 3 * y + 5) : x = 7 :=
sorry

end NUMINAMATH_GPT_first_number_is_seven_l1399_139989


namespace NUMINAMATH_GPT_concentric_circles_circumference_difference_and_area_l1399_139919

theorem concentric_circles_circumference_difference_and_area {r_inner r_outer : ℝ} (h1 : r_inner = 25) (h2 : r_outer = r_inner + 15) :
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi ∧ Real.pi * r_outer^2 - Real.pi * r_inner^2 = 975 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_concentric_circles_circumference_difference_and_area_l1399_139919


namespace NUMINAMATH_GPT_min_f_value_l1399_139987

noncomputable def f (a b : ℝ) := 
  Real.sqrt (2 * a^2 - 8 * a + 10) + 
  Real.sqrt (b^2 - 6 * b + 10) + 
  Real.sqrt (2 * a^2 - 2 * a * b + b^2)

theorem min_f_value : ∃ a b : ℝ, f a b = 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_min_f_value_l1399_139987


namespace NUMINAMATH_GPT_sin_cos_product_l1399_139977

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end NUMINAMATH_GPT_sin_cos_product_l1399_139977


namespace NUMINAMATH_GPT_Mark_bill_total_l1399_139969

theorem Mark_bill_total
  (original_bill : ℝ)
  (first_late_charge_rate : ℝ)
  (second_late_charge_rate : ℝ)
  (after_first_late_charge : ℝ)
  (final_total : ℝ) :
  original_bill = 500 ∧
  first_late_charge_rate = 0.02 ∧
  second_late_charge_rate = 0.02 ∧
  after_first_late_charge = original_bill * (1 + first_late_charge_rate) ∧
  final_total = after_first_late_charge * (1 + second_late_charge_rate) →
  final_total = 520.20 := by
  sorry

end NUMINAMATH_GPT_Mark_bill_total_l1399_139969


namespace NUMINAMATH_GPT_carl_olivia_cookie_difference_l1399_139990

-- Defining the various conditions
def Carl_cookies : ℕ := 7
def Olivia_cookies : ℕ := 2

-- Stating the theorem we need to prove
theorem carl_olivia_cookie_difference : Carl_cookies - Olivia_cookies = 5 :=
by sorry

end NUMINAMATH_GPT_carl_olivia_cookie_difference_l1399_139990


namespace NUMINAMATH_GPT_player_A_winning_probability_l1399_139982

theorem player_A_winning_probability :
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  P_total - P_draw - P_B_wins = 1 / 6 :=
by
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  sorry

end NUMINAMATH_GPT_player_A_winning_probability_l1399_139982


namespace NUMINAMATH_GPT_Ivan_increases_share_more_than_six_times_l1399_139957

theorem Ivan_increases_share_more_than_six_times
  (p v s i : ℝ)
  (hp : p / (v + s + i) = 3 / 7)
  (hv : v / (p + s + i) = 1 / 3)
  (hs : s / (p + v + i) = 1 / 3) :
  ∃ k : ℝ, k > 6 ∧ i * k > 0.6 * (p + v + s + i * k) :=
by
  sorry

end NUMINAMATH_GPT_Ivan_increases_share_more_than_six_times_l1399_139957


namespace NUMINAMATH_GPT_parallel_line_slope_l1399_139915

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1 / 2 ∧ (∀ x1 y1 : ℝ, 3 * x1 - 6 * y1 = 12 → 
    ∃ k : ℝ, y1 = m * x1 + k) :=
by
  sorry

end NUMINAMATH_GPT_parallel_line_slope_l1399_139915


namespace NUMINAMATH_GPT_trigonometric_expression_evaluation_l1399_139994

theorem trigonometric_expression_evaluation :
  1 / Real.sin (70 * Real.pi / 180) - Real.sqrt 3 / Real.cos (70 * Real.pi / 180) = -4 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_evaluation_l1399_139994


namespace NUMINAMATH_GPT_total_cost_of_antibiotics_l1399_139905

-- Definitions based on the conditions
def cost_A_per_dose : ℝ := 3
def cost_B_per_dose : ℝ := 4.50
def doses_per_day_A : ℕ := 2
def days_A : ℕ := 3
def doses_per_day_B : ℕ := 1
def days_B : ℕ := 4

-- Total cost calculations
def total_cost_A : ℝ := days_A * doses_per_day_A * cost_A_per_dose
def total_cost_B : ℝ := days_B * doses_per_day_B * cost_B_per_dose

-- Final proof statement
theorem total_cost_of_antibiotics : total_cost_A + total_cost_B = 36 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_total_cost_of_antibiotics_l1399_139905


namespace NUMINAMATH_GPT_b_in_terms_of_a_l1399_139918

noncomputable def a (k : ℝ) : ℝ := 3 + 3^k
noncomputable def b (k : ℝ) : ℝ := 3 + 3^(-k)

theorem b_in_terms_of_a (k : ℝ) :
  b k = (3 * (a k) - 8) / ((a k) - 3) := 
sorry

end NUMINAMATH_GPT_b_in_terms_of_a_l1399_139918


namespace NUMINAMATH_GPT_probability_each_university_at_least_one_admission_l1399_139955

def total_students := 4
def total_universities := 3

theorem probability_each_university_at_least_one_admission :
  ∃ (p : ℚ), p = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_each_university_at_least_one_admission_l1399_139955


namespace NUMINAMATH_GPT_evaluate_expr_right_to_left_l1399_139909

variable (a b c d : ℝ)

theorem evaluate_expr_right_to_left :
  (a - b * c + d) = a - b * (c + d) :=
sorry

end NUMINAMATH_GPT_evaluate_expr_right_to_left_l1399_139909


namespace NUMINAMATH_GPT_roots_sum_of_squares_l1399_139936

theorem roots_sum_of_squares {r s : ℝ} (h : Polynomial.roots (X^2 - 3*X + 1) = {r, s}) : r^2 + s^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_roots_sum_of_squares_l1399_139936


namespace NUMINAMATH_GPT_james_trip_time_l1399_139908

def speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

theorem james_trip_time:
  (distance / speed) + stop_time = 7 := 
by
  sorry

end NUMINAMATH_GPT_james_trip_time_l1399_139908


namespace NUMINAMATH_GPT_AlissaMorePresents_l1399_139929

/-- Ethan has 31 presents -/
def EthanPresents : ℕ := 31

/-- Alissa has 53 presents -/
def AlissaPresents : ℕ := 53

/-- How many more presents does Alissa have than Ethan? -/
theorem AlissaMorePresents : AlissaPresents - EthanPresents = 22 := by
  -- Place the proof here
  sorry

end NUMINAMATH_GPT_AlissaMorePresents_l1399_139929


namespace NUMINAMATH_GPT_Noah_age_in_10_years_is_22_l1399_139942

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end NUMINAMATH_GPT_Noah_age_in_10_years_is_22_l1399_139942


namespace NUMINAMATH_GPT_vasya_numbers_l1399_139973

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end NUMINAMATH_GPT_vasya_numbers_l1399_139973


namespace NUMINAMATH_GPT_points_on_line_l1399_139971

-- Define the points
def P1 : (ℝ × ℝ) := (8, 16)
def P2 : (ℝ × ℝ) := (2, 4)

-- Define the line equation as a predicate
def on_line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- Define the given points to be checked
def P3 : (ℝ × ℝ) := (5, 10)
def P4 : (ℝ × ℝ) := (7, 14)
def P5 : (ℝ × ℝ) := (4, 7)
def P6 : (ℝ × ℝ) := (10, 20)
def P7 : (ℝ × ℝ) := (3, 6)

theorem points_on_line :
  let m := 2
  let b := 0
  on_line m b P3 ∧
  on_line m b P4 ∧
  ¬ on_line m b P5 ∧
  on_line m b P6 ∧
  on_line m b P7 :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_l1399_139971


namespace NUMINAMATH_GPT_solve_quadratic_equation_l1399_139933

theorem solve_quadratic_equation :
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ ∀ x : ℝ, (x^2 - 2*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_equation_l1399_139933


namespace NUMINAMATH_GPT_polygon_sides_arithmetic_progression_l1399_139986

theorem polygon_sides_arithmetic_progression 
  (n : ℕ) 
  (h1 : ∀ n, ∃ a_1, ∃ a_n, ∀ i, a_n = 172 ∧ (a_i = a_1 + (i - 1) * 4) ∧ (i ≤ n))
  (h2 : ∀ S, S = 180 * (n - 2)) 
  (h3 : ∀ S, S = n * ((172 - 4 * (n - 1) + 172) / 2)) 
  : n = 12 := 
by 
  sorry

end NUMINAMATH_GPT_polygon_sides_arithmetic_progression_l1399_139986


namespace NUMINAMATH_GPT_max_M_min_N_l1399_139946

noncomputable def M (x y : ℝ) : ℝ := x / (2 * x + y) + y / (x + 2 * y)
noncomputable def N (x y : ℝ) : ℝ := x / (x + 2 * y) + y / (2 * x + y)

theorem max_M_min_N (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ t : ℝ, (∀ x y, 0 < x → 0 < y → M x y ≤ t) ∧ (∀ x y, 0 < x → 0 < y → N x y ≥ t) ∧ t = 2 / 3) :=
sorry

end NUMINAMATH_GPT_max_M_min_N_l1399_139946


namespace NUMINAMATH_GPT_length_four_implies_value_twenty_four_l1399_139901

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end NUMINAMATH_GPT_length_four_implies_value_twenty_four_l1399_139901


namespace NUMINAMATH_GPT_frequency_count_third_group_l1399_139914

theorem frequency_count_third_group 
  (x n : ℕ)
  (h1 : n = 420 - x)
  (h2 : x / (n:ℚ) = 0.20) :
  x = 70 :=
by sorry

end NUMINAMATH_GPT_frequency_count_third_group_l1399_139914


namespace NUMINAMATH_GPT_predicted_height_at_age_10_l1399_139924

-- Define the regression model as a function
def regression_model (x : ℝ) : ℝ := 7.19 * x + 73.93

-- Assert the predicted height at age 10
theorem predicted_height_at_age_10 : abs (regression_model 10 - 145.83) < 0.01 := 
by
  -- Here, we would prove the calculation steps
  sorry

end NUMINAMATH_GPT_predicted_height_at_age_10_l1399_139924


namespace NUMINAMATH_GPT_koala_food_consumed_l1399_139975

theorem koala_food_consumed (x y : ℝ) (h1 : 0.40 * x = 12) (h2 : 0.20 * y = 2) : 
  x = 30 ∧ y = 10 := 
by
  sorry

end NUMINAMATH_GPT_koala_food_consumed_l1399_139975


namespace NUMINAMATH_GPT_julia_money_given_l1399_139967

-- Define the conditions
def num_snickers : ℕ := 2
def num_mms : ℕ := 3
def cost_snickers : ℚ := 1.5
def cost_mms : ℚ := 2 * cost_snickers
def change_received : ℚ := 8

-- The total cost Julia had to pay
def total_cost : ℚ := (num_snickers * cost_snickers) + (num_mms * cost_mms)

-- Julia gave this amount of money to the cashier
def money_given : ℚ := total_cost + change_received

-- The problem to prove
theorem julia_money_given : money_given = 20 := by
  sorry

end NUMINAMATH_GPT_julia_money_given_l1399_139967


namespace NUMINAMATH_GPT_find_h_l1399_139932

theorem find_h {a b c n k : ℝ} (x : ℝ) (h_val : ℝ) 
  (h_quad : a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  (4 * a) * x^2 + (4 * b) * x + (4 * c) = n * (x - h_val)^2 + k → h_val = 5 :=
sorry

end NUMINAMATH_GPT_find_h_l1399_139932


namespace NUMINAMATH_GPT_total_bedrooms_is_correct_l1399_139930

def bedrooms_second_floor : Nat := 2
def bedrooms_first_floor : Nat := 8
def total_bedrooms (b1 b2 : Nat) : Nat := b1 + b2

theorem total_bedrooms_is_correct : total_bedrooms bedrooms_second_floor bedrooms_first_floor = 10 := 
by
  sorry

end NUMINAMATH_GPT_total_bedrooms_is_correct_l1399_139930


namespace NUMINAMATH_GPT_students_errors_proof_l1399_139927

noncomputable def students (x y0 y1 y2 y3 y4 y5 : ℕ): ℕ :=
  x + y5 + y4 + y3 + y2 + y1 + y0

noncomputable def errors (x y1 y2 y3 y4 y5 : ℕ): ℕ :=
  6 * x + 5 * y5 + 4 * y4 + 3 * y3 + 2 * y2 + y1

theorem students_errors_proof
  (x y0 y1 y2 y3 y4 y5 : ℕ)
  (h1 : students x y0 y1 y2 y3 y4 y5 = 333)
  (h2 : errors x y1 y2 y3 y4 y5 ≤ 1000) :
  x ≤ y3 + y2 + y1 + y0 :=
by
  sorry

end NUMINAMATH_GPT_students_errors_proof_l1399_139927
