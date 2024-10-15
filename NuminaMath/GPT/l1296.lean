import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_T_n_bound_l1296_129620

open Nat

theorem arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) (h2 : a 2 = 6) (h3_h6 : a 3 + a 6 = 27) :
  (∀ n, a n = 3 * n) := 
by
  sorry

theorem T_n_bound (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) (m : ℝ) (h_general_term : ∀ n, a n = 3 * n) 
  (h_S_n : ∀ n, S n = n^2 + n) (h_T_n : ∀ n, T n = (S n : ℝ) / (3 * (2 : ℝ)^(n-1)))
  (h_bound : ∀ n > 0, T n ≤ m) : 
  m ≥ 3/2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_T_n_bound_l1296_129620


namespace NUMINAMATH_GPT_zero_intercept_and_distinct_roots_l1296_129670

noncomputable def Q (x a' b' c' d' : ℝ) : ℝ := x^4 + a' * x^3 + b' * x^2 + c' * x + d'

theorem zero_intercept_and_distinct_roots (a' b' c' d' : ℝ) (u v w : ℝ) (h_distinct : u ≠ v ∧ v ≠ w ∧ u ≠ w) (h_intercept_at_zero : d' = 0)
(h_Q_form : ∀ x, Q x a' b' c' d' = x * (x - u) * (x - v) * (x - w)) : c' ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_zero_intercept_and_distinct_roots_l1296_129670


namespace NUMINAMATH_GPT_fraction_of_largest_jar_filled_l1296_129664

theorem fraction_of_largest_jar_filled
  (C1 C2 C3 : ℝ)
  (h1 : C1 < C2)
  (h2 : C2 < C3)
  (h3 : C1 / 6 = C2 / 5)
  (h4 : C2 / 5 = C3 / 7) :
  (C1 / 6 + C2 / 5) / C3 = 2 / 7 := sorry

end NUMINAMATH_GPT_fraction_of_largest_jar_filled_l1296_129664


namespace NUMINAMATH_GPT_calculate_total_students_l1296_129665

/-- Define the number of students who like basketball, cricket, and soccer. -/
def likes_basketball : ℕ := 7
def likes_cricket : ℕ := 10
def likes_soccer : ℕ := 8
def likes_all_three : ℕ := 2
def likes_basketball_and_cricket : ℕ := 5
def likes_basketball_and_soccer : ℕ := 4
def likes_cricket_and_soccer : ℕ := 3

/-- Calculate the number of students who like at least one sport using the principle of inclusion-exclusion. -/
def students_who_like_at_least_one_sport (b c s bc bs cs bcs : ℕ) : ℕ :=
  b + c + s - (bc + bs + cs) + bcs

theorem calculate_total_students :
  students_who_like_at_least_one_sport likes_basketball likes_cricket likes_soccer 
    (likes_basketball_and_cricket - likes_all_three) 
    (likes_basketball_and_soccer - likes_all_three) 
    (likes_cricket_and_soccer - likes_all_three) 
    likes_all_three = 21 := 
by
  sorry

end NUMINAMATH_GPT_calculate_total_students_l1296_129665


namespace NUMINAMATH_GPT_acute_triangle_condition_l1296_129647

theorem acute_triangle_condition (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (|a^2 - b^2| < c^2 ∧ c^2 < a^2 + b^2) ↔ (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) :=
sorry

end NUMINAMATH_GPT_acute_triangle_condition_l1296_129647


namespace NUMINAMATH_GPT_point_on_x_axis_coordinates_l1296_129632

theorem point_on_x_axis_coordinates (a : ℝ) (hx : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end NUMINAMATH_GPT_point_on_x_axis_coordinates_l1296_129632


namespace NUMINAMATH_GPT_jar_marbles_difference_l1296_129676

theorem jar_marbles_difference (a b : ℕ) (h1 : 9 * a = 9 * b) (h2 : 2 * a + b = 135) : 8 * b - 7 * a = 45 := by
  sorry

end NUMINAMATH_GPT_jar_marbles_difference_l1296_129676


namespace NUMINAMATH_GPT_businesses_brandon_can_apply_to_l1296_129682

-- Definitions of the given conditions in the problem
variables (x y : ℕ)

-- Define the total, fired, and quit businesses
def total_businesses : ℕ := 72
def fired_businesses : ℕ := 36
def quit_businesses : ℕ := 24

-- Define the unique businesses Brandon can still apply to, considering common businesses and reapplications
def businesses_can_apply_to : ℕ := (12 + x) + y

-- The theorem to prove
theorem businesses_brandon_can_apply_to (x y : ℕ) : businesses_can_apply_to x y = 12 + x + y := by
  unfold businesses_can_apply_to
  sorry

end NUMINAMATH_GPT_businesses_brandon_can_apply_to_l1296_129682


namespace NUMINAMATH_GPT_circle_equation_coefficients_l1296_129685

theorem circle_equation_coefficients (a : ℝ) (x y : ℝ) : 
  (a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a = 0) → (a = -1) :=
by 
  sorry

end NUMINAMATH_GPT_circle_equation_coefficients_l1296_129685


namespace NUMINAMATH_GPT_seq_eighth_term_l1296_129672

theorem seq_eighth_term : (8^2 + 2 * 8 - 1 = 79) :=
by
  sorry

end NUMINAMATH_GPT_seq_eighth_term_l1296_129672


namespace NUMINAMATH_GPT_rectangle_ratio_l1296_129669

open Real

theorem rectangle_ratio (A B C D E : Point) (rat : ℚ) : 
  let area_rect := 1
  let area_pentagon := (7 / 10 : ℚ)
  let area_triangle_AEC := 3 / 10
  let area_triangle_ECD := 1 / 5
  let x := 3 * EA
  let y := 2 * EA
  let diag_longer_side := sqrt (5 * EA ^ 2)
  let diag_shorter_side := EA * sqrt 5
  let ratio := sqrt 5 
  ( area_pentagon == area_rect * (7 / 10) ) →
  ( area_triangle_AEC + area_pentagon = area_rect ) →
  ( area_triangle_AEC == area_rect - area_pentagon ) →
  ( ratio == diag_longer_side / diag_shorter_side ) :=
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1296_129669


namespace NUMINAMATH_GPT_asbestos_tiles_width_l1296_129646

theorem asbestos_tiles_width (n : ℕ) (h : 0 < n) :
  let width_per_tile := 60
  let overlap := 10
  let effective_width := width_per_tile - overlap
  width_per_tile + (n - 1) * effective_width = 50 * n + 10 := by
sorry

end NUMINAMATH_GPT_asbestos_tiles_width_l1296_129646


namespace NUMINAMATH_GPT_shirts_made_today_l1296_129660

def shirts_per_minute : ℕ := 6
def minutes_yesterday : ℕ := 12
def total_shirts : ℕ := 156
def shirts_yesterday : ℕ := shirts_per_minute * minutes_yesterday
def shirts_today : ℕ := total_shirts - shirts_yesterday

theorem shirts_made_today :
  shirts_today = 84 :=
by
  sorry

end NUMINAMATH_GPT_shirts_made_today_l1296_129660


namespace NUMINAMATH_GPT_mink_babies_l1296_129661

theorem mink_babies (B : ℕ) (h_coats : 7 * 15 = 105)
    (h_minks: 30 + 30 * B = 210) :
  B = 6 :=
by
  sorry

end NUMINAMATH_GPT_mink_babies_l1296_129661


namespace NUMINAMATH_GPT_slope_of_line_through_midpoints_l1296_129617

theorem slope_of_line_through_midpoints :
  let P₁ := (1, 2)
  let P₂ := (3, 8)
  let P₃ := (4, 3)
  let P₄ := (7, 9)
  let M₁ := ( (P₁.1 + P₂.1)/2, (P₁.2 + P₂.2)/2 )
  let M₂ := ( (P₃.1 + P₄.1)/2, (P₃.2 + P₄.2)/2 )
  let slope := (M₂.2 - M₁.2) / (M₂.1 - M₁.1)
  slope = 2/7 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_through_midpoints_l1296_129617


namespace NUMINAMATH_GPT_tim_out_of_pocket_cost_l1296_129655

noncomputable def totalOutOfPocketCost : ℝ :=
  let mriCost := 1200
  let xrayCost := 500
  let examinationCost := 400 * (45 / 60)
  let feeForBeingSeen := 150
  let consultationFee := 75
  let physicalTherapyCost := 100 * 8
  let totalCostBeforeInsurance := mriCost + xrayCost + examinationCost + feeForBeingSeen + consultationFee + physicalTherapyCost
  let insuranceCoverage := 0.70 * totalCostBeforeInsurance
  let outOfPocketCost := totalCostBeforeInsurance - insuranceCoverage
  outOfPocketCost

theorem tim_out_of_pocket_cost : totalOutOfPocketCost = 907.50 :=
  by
    -- Proof will be provided here
    sorry

end NUMINAMATH_GPT_tim_out_of_pocket_cost_l1296_129655


namespace NUMINAMATH_GPT_sum_of_digits_is_15_l1296_129602

theorem sum_of_digits_is_15
  (A B C D E : ℕ) 
  (h_distinct: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_digits: A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
  (h_divisible_by_9: (A * 10000 + B * 1000 + C * 100 + D * 10 + E) % 9 = 0) 
  : A + B + C + D + E = 15 := 
sorry

end NUMINAMATH_GPT_sum_of_digits_is_15_l1296_129602


namespace NUMINAMATH_GPT_magnitude_squared_l1296_129629

-- Let z be the complex number 3 + 4i
def z : ℂ := 3 + 4 * Complex.I

-- Prove that the magnitude of z squared equals 25
theorem magnitude_squared : Complex.abs z ^ 2 = 25 := by
  -- The term "by" starts the proof block, and "sorry" allows us to skip the proof details.
  sorry

end NUMINAMATH_GPT_magnitude_squared_l1296_129629


namespace NUMINAMATH_GPT_toy_poodle_height_l1296_129662

theorem toy_poodle_height 
  (SP MP TP : ℕ)
  (h1 : SP = MP + 8)
  (h2 : MP = TP + 6)
  (h3 : SP = 28) 
  : TP = 14 := 
    by sorry

end NUMINAMATH_GPT_toy_poodle_height_l1296_129662


namespace NUMINAMATH_GPT_car_speed_is_90_mph_l1296_129689

-- Define the given conditions
def distance_yards : ℚ := 22
def time_seconds : ℚ := 0.5
def yards_per_mile : ℚ := 1760

-- Define the car's speed in miles per hour
noncomputable def car_speed_mph : ℚ := (distance_yards / yards_per_mile) * (3600 / time_seconds)

-- The theorem to be proven
theorem car_speed_is_90_mph : car_speed_mph = 90 := by
  sorry

end NUMINAMATH_GPT_car_speed_is_90_mph_l1296_129689


namespace NUMINAMATH_GPT_min_sum_p_q_r_s_l1296_129614

theorem min_sum_p_q_r_s (p q r s : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hs : 0 < s)
    (h1 : 2 * p = 10 * p - 15 * q)
    (h2 : 2 * q = 6 * p - 9 * q)
    (h3 : 3 * r = 10 * r - 15 * s)
    (h4 : 3 * s = 6 * r - 9 * s) : p + q + r + s = 45 := by
  sorry

end NUMINAMATH_GPT_min_sum_p_q_r_s_l1296_129614


namespace NUMINAMATH_GPT_monkey_tree_height_l1296_129628

theorem monkey_tree_height (hours: ℕ) (hop ft_per_hour : ℕ) (slip ft_per_hour : ℕ) (net_progress : ℕ) (final_hour : ℕ) (total_height : ℕ) :
  (hours = 18) ∧
  (hop = 3) ∧
  (slip = 2) ∧
  (net_progress = hop - slip) ∧
  (net_progress = 1) ∧
  (final_hour = 1) ∧
  (total_height = (hours - 1) * net_progress + hop) ∧
  (total_height = 20) :=
by
  sorry

end NUMINAMATH_GPT_monkey_tree_height_l1296_129628


namespace NUMINAMATH_GPT_grid_blue_probability_l1296_129633

-- Define the problem in Lean
theorem grid_blue_probability :
  let n := 4
  let p_tile_blue := 1 / 2
  let invariant_prob := (p_tile_blue ^ (n / 2))
  let pair_prob := (p_tile_blue * p_tile_blue)
  let total_pairs := (n * n / 2 - n / 2)
  let final_prob := (invariant_prob ^ 2) * (pair_prob ^ total_pairs)
  final_prob = 1 / 65536 := by
  sorry

end NUMINAMATH_GPT_grid_blue_probability_l1296_129633


namespace NUMINAMATH_GPT_arithmetic_sequence_n_equals_8_l1296_129618

theorem arithmetic_sequence_n_equals_8 :
  (∀ (a b c : ℕ), a + (1 / 4) * c = 2 * (1 / 2) * b) → ∃ n : ℕ, n = 8 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_n_equals_8_l1296_129618


namespace NUMINAMATH_GPT_hanks_pancakes_needed_l1296_129680

/-- Hank's pancake calculation problem -/
theorem hanks_pancakes_needed 
    (pancakes_per_big_stack : ℕ := 5)
    (pancakes_per_short_stack : ℕ := 3)
    (big_stack_orders : ℕ := 6)
    (short_stack_orders : ℕ := 9) :
    (pancakes_per_short_stack * short_stack_orders) + (pancakes_per_big_stack * big_stack_orders) = 57 := by {
  sorry
}

end NUMINAMATH_GPT_hanks_pancakes_needed_l1296_129680


namespace NUMINAMATH_GPT_radius_of_larger_circle_l1296_129648

theorem radius_of_larger_circle (r R AC BC AB : ℝ)
  (h1 : R = 4 * r)
  (h2 : AC = 8 * r)
  (h3 : BC^2 + AB^2 = AC^2)
  (h4 : AB = 16) :
  R = 32 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_l1296_129648


namespace NUMINAMATH_GPT_polynomial_independent_of_m_l1296_129619

theorem polynomial_independent_of_m (m : ℝ) (x : ℝ) (h : 6 * x^2 + (1 - 2 * m) * x + 7 * m = 6 * x^2 + x) : 
  x = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_independent_of_m_l1296_129619


namespace NUMINAMATH_GPT_quotient_is_six_l1296_129639

-- Definition of the given conditions
def S : Int := 476
def remainder : Int := 15
def difference : Int := 2395

-- Definition of the larger number based on the given conditions
def L : Int := S + difference

-- The statement we need to prove
theorem quotient_is_six : (L = S * 6 + remainder) := by
  sorry

end NUMINAMATH_GPT_quotient_is_six_l1296_129639


namespace NUMINAMATH_GPT_transform_polynomial_to_y_l1296_129697

theorem transform_polynomial_to_y (x y : ℝ) (h : y = x + 1/x) :
  (x^6 + x^5 - 5*x^4 + x^3 + x + 1 = 0) → 
  (∃ (y_expr : ℝ), (x * y_expr = 0 ∨ (x = 0 ∧ y_expr = y_expr))) :=
sorry

end NUMINAMATH_GPT_transform_polynomial_to_y_l1296_129697


namespace NUMINAMATH_GPT_correct_fraction_simplification_l1296_129679

theorem correct_fraction_simplification (a b : ℝ) (h : a ≠ b) : 
  (∀ (c d : ℝ), (c ≠ d) → (a+2 = c → b+2 = d → (a+2)/d ≠ a/b))
  ∧ (∀ (e f : ℝ), (e ≠ f) → (a-2 = e → b-2 = f → (a-2)/f ≠ a/b))
  ∧ (∀ (g h : ℝ), (g ≠ h) → (a^2 = g → b^2 = h → a^2/h ≠ a/b))
  ∧ (a / b = ( (1/2)*a / (1/2)*b )) := 
sorry

end NUMINAMATH_GPT_correct_fraction_simplification_l1296_129679


namespace NUMINAMATH_GPT_determine_a_l1296_129688

-- Given conditions
variable {a b : ℝ}
variable (h_neg : a < 0) (h_pos : b > 0) (h_max : ∀ x, -2 ≤ a * sin (b * x) ∧ a * sin (b * x) ≤ 2)

-- Statement to prove
theorem determine_a : a = -2 := by
  sorry

end NUMINAMATH_GPT_determine_a_l1296_129688


namespace NUMINAMATH_GPT_calculate_power_l1296_129650

variable (x y : ℝ)

theorem calculate_power :
  (- (1 / 2) * x^2 * y)^3 = - (1 / 8) * x^6 * y^3 :=
sorry

end NUMINAMATH_GPT_calculate_power_l1296_129650


namespace NUMINAMATH_GPT_angle_AOC_is_45_or_15_l1296_129603

theorem angle_AOC_is_45_or_15 (A O B C : Type) (α β γ : ℝ) 
  (h1 : α = 30) (h2 : β = 15) : γ = 45 ∨ γ = 15 :=
sorry

end NUMINAMATH_GPT_angle_AOC_is_45_or_15_l1296_129603


namespace NUMINAMATH_GPT_average_capacity_is_3_65_l1296_129642

/-- Define the capacities of the jars as a list--/
def jarCapacities : List ℚ := [2, 1/4, 8, 1.5, 0.75, 3, 10]

/-- Calculate the average jar capacity --/
def averageCapacity (capacities : List ℚ) : ℚ :=
  (capacities.sum) / (capacities.length)

/-- The average jar capacity for the given list of jar capacities is 3.65 liters. --/
theorem average_capacity_is_3_65 :
  averageCapacity jarCapacities = 3.65 := 
by
  unfold averageCapacity
  dsimp [jarCapacities]
  norm_num
  sorry

end NUMINAMATH_GPT_average_capacity_is_3_65_l1296_129642


namespace NUMINAMATH_GPT_polynomial_roots_l1296_129613

-- The statement that we need to prove
theorem polynomial_roots (a b : ℚ) (h : (2 + Real.sqrt 3) ^ 3 + 4 * (2 + Real.sqrt 3) ^ 2 + a * (2 + Real.sqrt 3) + b = 0) :
  ((Polynomial.C a * Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) = Polynomial.X ^ 3 + Polynomial.C (4 : ℚ) * Polynomial.X ^ 2 + Polynomial.C a * Polynomial.X + Polynomial.C b) →
  (2 - Real.sqrt 3) ^ 3 + 4 * (2 - Real.sqrt 3) ^ 2 + a * (2 - Real.sqrt 3) + b = 0 ∧ -8 ^ 3 + 4 * (-8) ^ 2 + a * (-8) + b = 0 := sorry

end NUMINAMATH_GPT_polynomial_roots_l1296_129613


namespace NUMINAMATH_GPT_positive_integral_solution_exists_l1296_129643

theorem positive_integral_solution_exists :
  ∃ n : ℕ, n > 0 ∧
  ( (n * (n + 1) * (2 * n + 1)) * 100 = 27 * 6 * (n * (n + 1))^2 ) ∧ n = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_positive_integral_solution_exists_l1296_129643


namespace NUMINAMATH_GPT_three_digit_number_cubed_sum_l1296_129668

theorem three_digit_number_cubed_sum {n : ℕ} (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (∃ a b c : ℕ, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 100 * a + 10 * b + c ∧ n = a^3 + b^3 + c^3) ↔
  n = 153 ∨ n = 370 ∨ n = 371 ∨ n = 407 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_cubed_sum_l1296_129668


namespace NUMINAMATH_GPT_mia_stops_in_quarter_C_l1296_129690

def track_circumference : ℕ := 100 -- The circumference of the track in feet.
def total_distance_run : ℕ := 10560 -- The total distance Mia runs in feet.

-- Define the function to determine the quarter of the circle Mia stops in.
def quarter_mia_stops : ℕ :=
  let quarters := track_circumference / 4 -- Each quarter's length.
  let complete_laps := total_distance_run / track_circumference
  let remaining_distance := total_distance_run % track_circumference
  if remaining_distance < quarters then 1 -- Quarter A
  else if remaining_distance < 2 * quarters then 2 -- Quarter B
  else if remaining_distance < 3 * quarters then 3 -- Quarter C
  else 4 -- Quarter D

theorem mia_stops_in_quarter_C : quarter_mia_stops = 3 := by
  sorry

end NUMINAMATH_GPT_mia_stops_in_quarter_C_l1296_129690


namespace NUMINAMATH_GPT_inequality_holds_l1296_129659

variable (x a : ℝ)

def tensor (x y : ℝ) : ℝ :=
  (1 - x) * (1 + y)

theorem inequality_holds (h : ∀ x : ℝ, tensor (x - a) (x + a) < 1) : -2 < a ∧ a < 0 := by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1296_129659


namespace NUMINAMATH_GPT_dwarfs_truthful_count_l1296_129607

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_dwarfs_truthful_count_l1296_129607


namespace NUMINAMATH_GPT_lattice_point_count_l1296_129678

noncomputable def countLatticePoints (N : ℤ) : ℤ :=
  2 * N * (N + 1) + 1

theorem lattice_point_count (N : ℤ) (hN : 71 * N > 0) :
    ∃ P, P = countLatticePoints N := sorry

end NUMINAMATH_GPT_lattice_point_count_l1296_129678


namespace NUMINAMATH_GPT_first_discount_l1296_129658

theorem first_discount (P F : ℕ) (D₂ : ℝ) (D₁ : ℝ) 
  (hP : P = 150) 
  (hF : F = 105)
  (hD₂ : D₂ = 12.5)
  (hF_eq : F = P * (1 - D₁ / 100) * (1 - D₂ / 100)) : 
  D₁ = 20 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_l1296_129658


namespace NUMINAMATH_GPT_no_square_pair_l1296_129687

/-- 
Given integers a, b, and c, where c > 0, if a(a + 4) = c^2 and (a + 2 + c)(a + 2 - c) = 4, 
then the numbers a(a + 4) and b(b + 4) cannot both be squares.
-/
theorem no_square_pair (a b c : ℤ) (hc_pos : c > 0) (ha_eq : a * (a + 4) = c^2) 
  (hfac_eq : (a + 2 + c) * (a + 2 - c) = 4) : ¬(∃ d e : ℤ, d^2 = a * (a + 4) ∧ e^2 = b * (b + 4)) :=
by sorry

end NUMINAMATH_GPT_no_square_pair_l1296_129687


namespace NUMINAMATH_GPT_proof_main_proof_l1296_129673

noncomputable def main_proof : Prop :=
  2 * Real.logb 5 10 + Real.logb 5 0.25 = 2

theorem proof_main_proof : main_proof :=
  by
    sorry

end NUMINAMATH_GPT_proof_main_proof_l1296_129673


namespace NUMINAMATH_GPT_minimize_fraction_sum_l1296_129634

theorem minimize_fraction_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h_sum : a + b + c = 6) :
  (9 / a + 4 / b + 25 / c) ≥ 50 / 3 :=
sorry

end NUMINAMATH_GPT_minimize_fraction_sum_l1296_129634


namespace NUMINAMATH_GPT_sum_after_50_rounds_l1296_129624

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end NUMINAMATH_GPT_sum_after_50_rounds_l1296_129624


namespace NUMINAMATH_GPT_average_price_of_pencil_correct_l1296_129615

def average_price_of_pencil (n_pens n_pencils : ℕ) (total_cost pen_price : ℕ) : ℕ :=
  let pen_cost := n_pens * pen_price
  let pencil_cost := total_cost - pen_cost
  let avg_pencil_price := pencil_cost / n_pencils
  avg_pencil_price

theorem average_price_of_pencil_correct :
  average_price_of_pencil 30 75 450 10 = 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end NUMINAMATH_GPT_average_price_of_pencil_correct_l1296_129615


namespace NUMINAMATH_GPT_percent_boys_in_class_l1296_129699

-- Define the conditions given in the problem
def initial_ratio (b g : ℕ) : Prop := b = 3 * g / 4

def total_students_after_new_girls (total : ℕ) (new_girls : ℕ) : Prop :=
  total = 42 ∧ new_girls = 4

-- Define the percentage calculation correctness
def percentage_of_boys (boys total : ℕ) (percentage : ℚ) : Prop :=
  percentage = (boys : ℚ) / (total : ℚ) * 100

-- State the theorem to be proven
theorem percent_boys_in_class
  (b g : ℕ)   -- Number of boys and initial number of girls
  (total new_girls : ℕ) -- Total students after new girls joined and number of new girls
  (percentage : ℚ) -- The percentage of boys in the class
  (h_initial_ratio : initial_ratio b g)
  (h_total_students : total_students_after_new_girls total new_girls)
  (h_goals : g + new_girls = total - b)
  (h_correct_calc : percentage = 35.71) :
  percentage_of_boys b total percentage :=
by
  sorry

end NUMINAMATH_GPT_percent_boys_in_class_l1296_129699


namespace NUMINAMATH_GPT_fraction_identity_l1296_129686

variable (a b : ℚ) (h : a / b = 2 / 3)

theorem fraction_identity : a / (a - b) = -2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1296_129686


namespace NUMINAMATH_GPT_percent_value_in_quarters_l1296_129698

theorem percent_value_in_quarters (dimes quarters : ℕ) (dime_value quarter_value : ℕ) (dime_count quarter_count : ℕ) :
  dimes = 50 →
  quarters = 20 →
  dime_value = 10 →
  quarter_value = 25 →
  dime_count = dimes * dime_value →
  quarter_count = quarters * quarter_value →
  (quarter_count : ℚ) / (dime_count + quarter_count) * 100 = 50 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percent_value_in_quarters_l1296_129698


namespace NUMINAMATH_GPT_difference_largest_smallest_l1296_129663

noncomputable def ratio_2_3_5 := 2 / 3
noncomputable def ratio_3_5 := 3 / 5
noncomputable def int_sum := 90

theorem difference_largest_smallest :
  ∃ (a b c : ℝ), 
    a + b + c = int_sum ∧
    b / a = ratio_2_3_5 ∧
    c / a = 5 / 2 ∧
    b / a = 3 / 2 ∧
    c - a = 12.846 := 
by
  sorry

end NUMINAMATH_GPT_difference_largest_smallest_l1296_129663


namespace NUMINAMATH_GPT_partition_solution_l1296_129636

noncomputable def partitions (a m n x : ℝ) : Prop :=
  a = x + n * (a - m * x)

theorem partition_solution (a m n : ℝ) (h : n * m < 1) :
  partitions a m n (a * (1 - n) / (1 - n * m)) :=
by
  sorry

end NUMINAMATH_GPT_partition_solution_l1296_129636


namespace NUMINAMATH_GPT_value_of_x_minus_y_l1296_129637

theorem value_of_x_minus_y (x y : ℝ) 
    (h1 : 3015 * x + 3020 * y = 3025) 
    (h2 : 3018 * x + 3024 * y = 3030) :
    x - y = 11.1167 :=
sorry

end NUMINAMATH_GPT_value_of_x_minus_y_l1296_129637


namespace NUMINAMATH_GPT_multiply_exp_result_l1296_129608

theorem multiply_exp_result : 121 * (5 ^ 4) = 75625 :=
by
  sorry

end NUMINAMATH_GPT_multiply_exp_result_l1296_129608


namespace NUMINAMATH_GPT_cubes_sum_expr_l1296_129681

variable {a b s p : ℝ}

theorem cubes_sum_expr (h1 : s = a + b) (h2 : p = a * b) : a^3 + b^3 = s^3 - 3 * s * p := by
  sorry

end NUMINAMATH_GPT_cubes_sum_expr_l1296_129681


namespace NUMINAMATH_GPT_larger_number_l1296_129640

theorem larger_number (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 4) : x = 17 :=
by
sorry

end NUMINAMATH_GPT_larger_number_l1296_129640


namespace NUMINAMATH_GPT_sum_of_first_3030_terms_l1296_129605

-- Define geometric sequence sum for n terms
noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
axiom geom_sum_1010 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 1010 = 100
axiom geom_sum_2020 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 2020 = 190

-- Prove that the sum of the first 3030 terms is 271
theorem sum_of_first_3030_terms (a r : ℝ) (hr : r ≠ 1) :
  geom_sum a r 3030 = 271 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_3030_terms_l1296_129605


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l1296_129612

theorem system1_solution : 
  ∃ (x y : ℤ), 2 * x + 3 * y = -1 ∧ y = 4 * x - 5 ∧ x = 1 ∧ y = -1 := by 
    sorry

theorem system2_solution : 
  ∃ (x y : ℤ), 3 * x + 2 * y = 20 ∧ 4 * x - 5 * y = 19 ∧ x = 6 ∧ y = 1 := by 
    sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l1296_129612


namespace NUMINAMATH_GPT_roses_to_sister_l1296_129695

theorem roses_to_sister (total_roses roses_to_mother roses_to_grandmother roses_kept : ℕ) 
  (h1 : total_roses = 20)
  (h2 : roses_to_mother = 6)
  (h3 : roses_to_grandmother = 9)
  (h4 : roses_kept = 1) : 
  total_roses - (roses_to_mother + roses_to_grandmother + roses_kept) = 4 :=
by
  sorry

end NUMINAMATH_GPT_roses_to_sister_l1296_129695


namespace NUMINAMATH_GPT_reduced_price_l1296_129638

theorem reduced_price (P : ℝ) (hP : P = 56)
    (original_qty : ℝ := 800 / P)
    (reduced_qty : ℝ := 800 / (0.65 * P))
    (diff_qty : ℝ := reduced_qty - original_qty)
    (difference_condition : diff_qty = 5) :
  0.65 * P = 36.4 :=
by
  rw [hP]
  sorry

end NUMINAMATH_GPT_reduced_price_l1296_129638


namespace NUMINAMATH_GPT_painted_cubes_count_l1296_129649

def total_painted_cubes : ℕ := 8 + 48

theorem painted_cubes_count : total_painted_cubes = 56 :=
by 
  -- Step 1: Define the number of cubes with 3 faces painted (8 corners)
  let corners := 8
  -- Step 2: Calculate the number of edge cubes with 2 faces painted
  let edge_middle_cubes_per_edge := 2
  let edges := 12
  let edge_cubes := edge_middle_cubes_per_edge * edges -- this should be 24
  -- Step 3: Calculate the number of face-interior cubes with 2 faces painted
  let face_cubes_per_face := 4
  let faces := 6
  let face_cubes := face_cubes_per_face * faces -- this should be 24
  -- Step 4: Sum them up to get total cubes with at least two faces painted
  let total_cubes := corners + edge_cubes + face_cubes
  show total_cubes = total_painted_cubes
  sorry

end NUMINAMATH_GPT_painted_cubes_count_l1296_129649


namespace NUMINAMATH_GPT_point_inside_circle_l1296_129692

theorem point_inside_circle : 
  ∀ (x y : ℝ), 
  (x-2)^2 + (y-3)^2 = 4 → 
  (3-2)^2 + (2-3)^2 < 4 :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_point_inside_circle_l1296_129692


namespace NUMINAMATH_GPT_product_at_n_equals_three_l1296_129671

theorem product_at_n_equals_three : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) = 120 := by
  sorry

end NUMINAMATH_GPT_product_at_n_equals_three_l1296_129671


namespace NUMINAMATH_GPT_regular_hexagon_interior_angles_l1296_129667

theorem regular_hexagon_interior_angles (n : ℕ) (h : n = 6) :
  (n - 2) * 180 = 720 :=
by
  subst h
  rfl

end NUMINAMATH_GPT_regular_hexagon_interior_angles_l1296_129667


namespace NUMINAMATH_GPT_sum_of_ages_l1296_129625

variable (A1 : ℝ) (A2 : ℝ) (A3 : ℝ) (A4 : ℝ) (A5 : ℝ) (A6 : ℝ) (A7 : ℝ)

noncomputable def age_first_scroll := 4080
noncomputable def age_difference := 2040

theorem sum_of_ages :
  let r := (age_difference:ℝ) / (age_first_scroll:ℝ)
  let A2 := (age_first_scroll:ℝ) + age_difference
  let A3 := A2 + (A2 - age_first_scroll) * r
  let A4 := A3 + (A3 - A2) * r
  let A5 := A4 + (A4 - A3) * r
  let A6 := A5 + (A5 - A4) * r
  let A7 := A6 + (A6 - A5) * r
  (age_first_scroll:ℝ) + A2 + A3 + A4 + A5 + A6 + A7 = 41023.75 := 
  by sorry

end NUMINAMATH_GPT_sum_of_ages_l1296_129625


namespace NUMINAMATH_GPT_regular_polygon_sides_l1296_129666

theorem regular_polygon_sides (n : ℕ) (h1 : 360 / n = 18) : n = 20 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1296_129666


namespace NUMINAMATH_GPT_angle_A_range_find_b_l1296_129630

-- Definitions based on problem conditions
variable {a b c S : ℝ}
variable {A B C : ℝ}
variable {x : ℝ}

-- First statement: range of values for A
theorem angle_A_range (h1 : c * b * Real.cos A ≤ 2 * Real.sqrt 3 * S)
                      (h2 : S = 1/2 * b * c * Real.sin A)
                      (h3 : 0 < A ∧ A < π) : π / 6 ≤ A ∧ A < π := 
sorry

-- Second statement: value of b
theorem find_b (h1 : Real.tan A = x ∧ Real.tan B = 2 * x ∧ Real.tan C = 3 * x)
               (h2 : x = 1)
               (h3 : c = 1) : b = 2 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_GPT_angle_A_range_find_b_l1296_129630


namespace NUMINAMATH_GPT_other_student_in_sample_18_l1296_129616

theorem other_student_in_sample_18 (class_size sample_size : ℕ) (all_students : Finset ℕ) (sample_students : List ℕ)
  (h_class_size : class_size = 60)
  (h_sample_size : sample_size = 4)
  (h_all_students : all_students = Finset.range 60) -- students are numbered from 1 to 60
  (h_sample : sample_students = [3, 33, 48])
  (systematic_sampling : ℕ → ℕ → List ℕ) -- systematic_sampling function that generates the sample based on first element and k
  (k : ℕ) (h_k : k = class_size / sample_size) :
  systematic_sampling 3 k = [3, 18, 33, 48] := 
  sorry

end NUMINAMATH_GPT_other_student_in_sample_18_l1296_129616


namespace NUMINAMATH_GPT_f_of_3_is_log2_3_l1296_129677

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_condition : ∀ x : ℝ, f (2 ^ x) = x

theorem f_of_3_is_log2_3 : f 3 = Real.log 3 / Real.log 2 := sorry

end NUMINAMATH_GPT_f_of_3_is_log2_3_l1296_129677


namespace NUMINAMATH_GPT_fourth_term_of_sequence_l1296_129691

theorem fourth_term_of_sequence (x : ℤ) (h : x^2 - 2 * x - 3 < 0) (hx : x ∈ {n : ℤ | x^2 - 2 * x - 3 < 0}) :
  ∃ a_1 a_2 a_3 a_4 : ℤ, 
  (a_1 = x) ∧ (a_2 = x + 1) ∧ (a_3 = x + 2) ∧ (a_4 = x + 3) ∧ 
  (a_4 = 3 ∨ a_4 = -1) :=
by { sorry }

end NUMINAMATH_GPT_fourth_term_of_sequence_l1296_129691


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l1296_129622

theorem geometric_series_common_ratio (a r S : ℝ) 
  (hS : S = a / (1 - r)) 
  (h64 : (a * r^4) / (1 - r) = S / 64) : 
  r = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l1296_129622


namespace NUMINAMATH_GPT_rationalize_denominator_l1296_129641

theorem rationalize_denominator :
  (Real.sqrt (5 / 12)) = ((Real.sqrt 15) / 6) :=
sorry

end NUMINAMATH_GPT_rationalize_denominator_l1296_129641


namespace NUMINAMATH_GPT_rearrangement_impossible_l1296_129621

-- Define the primary problem conditions and goal
theorem rearrangement_impossible :
  ¬ ∃ (f : Fin 100 → Fin 51), 
    (∀ k : Fin 51, ∃ i j : Fin 100, 
      f i = k ∧ f j = k ∧ (i < j ∧ j.val - i.val = k.val + 1)) :=
sorry

end NUMINAMATH_GPT_rearrangement_impossible_l1296_129621


namespace NUMINAMATH_GPT_canonical_equations_of_line_l1296_129601

/-- Given two planes: 
  Plane 1: 4 * x + y + z + 2 = 0
  Plane 2: 2 * x - y - 3 * z - 8 = 0
  Prove that the canonical equations of the line formed by their intersection are:
  (x - 1) / -2 = (y + 6) / 14 = z / -6 -/
theorem canonical_equations_of_line :
  (∃ x y z : ℝ, 4 * x + y + z + 2 = 0 ∧ 2 * x - y - 3 * z - 8 = 0) →
  (∀ x y z : ℝ, ((x - 1) / -2 = (y + 6) / 14) ∧ ((y + 6) / 14 = z / -6)) :=
by
  sorry

end NUMINAMATH_GPT_canonical_equations_of_line_l1296_129601


namespace NUMINAMATH_GPT_integral_curve_has_inflection_points_l1296_129610

theorem integral_curve_has_inflection_points (x y : ℝ) (f : ℝ → ℝ → ℝ) :
  f x y = y - x^2 + 2*x - 2 →
  (∃ y' y'' : ℝ, y' = f x y ∧ y'' = y - x^2 ∧ y'' = 0) ↔ y = x^2 :=
by
  sorry

end NUMINAMATH_GPT_integral_curve_has_inflection_points_l1296_129610


namespace NUMINAMATH_GPT_tony_income_l1296_129696

-- Definitions for the given conditions
def investment : ℝ := 3200
def purchase_price : ℝ := 85
def dividend : ℝ := 6.640625

-- Theorem stating Tony's income based on the conditions
theorem tony_income : (investment / purchase_price) * dividend = 250 :=
by
  sorry

end NUMINAMATH_GPT_tony_income_l1296_129696


namespace NUMINAMATH_GPT_find_x_l1296_129623

theorem find_x (x : ℝ) (h1: x > 0) (h2 : 1 / 2 * x * (3 * x) = 72) : x = 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_find_x_l1296_129623


namespace NUMINAMATH_GPT_centroid_distance_l1296_129694

theorem centroid_distance
  (a b m : ℝ)
  (h_a_nonneg : 0 ≤ a)
  (h_b_nonneg : 0 ≤ b)
  (h_m_pos : 0 < m) :
  (∃ d : ℝ, d = m * (b + 2 * a) / (3 * (a + b))) :=
by
  sorry

end NUMINAMATH_GPT_centroid_distance_l1296_129694


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1296_129693

-- Define the terms of the geometric sequence
variable {a : ℕ → ℝ}

-- Define the conditions of the problem
def a2_cond : Prop := a 2 = 2
def a6_cond : Prop := a 6 = 32

-- Define the theorem we want to prove
theorem geometric_sequence_a4 (a2_cond : a 2 = 2) (a6_cond : a 6 = 32) : a 4 = 8 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1296_129693


namespace NUMINAMATH_GPT_original_people_count_l1296_129654

theorem original_people_count (x : ℕ) 
  (H1 : (x - x / 3) / 2 = 15) : x = 45 := by
  sorry

end NUMINAMATH_GPT_original_people_count_l1296_129654


namespace NUMINAMATH_GPT_fractions_order_l1296_129627

theorem fractions_order : (23 / 18) < (21 / 16) ∧ (21 / 16) < (25 / 19) :=
by
  sorry

end NUMINAMATH_GPT_fractions_order_l1296_129627


namespace NUMINAMATH_GPT_fractions_sum_simplified_l1296_129684

noncomputable def frac12over15 : ℚ := 12 / 15
noncomputable def frac7over9 : ℚ := 7 / 9
noncomputable def frac1and1over6 : ℚ := 1 + 1 / 6

theorem fractions_sum_simplified :
  frac12over15 + frac7over9 + frac1and1over6 = 247 / 90 :=
by
  -- This step will be left as a proof to complete.
  sorry

end NUMINAMATH_GPT_fractions_sum_simplified_l1296_129684


namespace NUMINAMATH_GPT_imo1989_q3_l1296_129644

theorem imo1989_q3 (a b : ℤ) (h1 : ¬ (∃ x : ℕ, a = x ^ 2))
                   (h2 : ¬ (∃ y : ℕ, b = y ^ 2))
                   (h3 : ∃ (x y z w : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 + a * b * w ^ 2 = 0 
                                           ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) :
                   ∃ (x y z : ℤ), x ^ 2 - a * y ^ 2 - b * z ^ 2 = 0 ∧ (x, y, z) ≠ (0, 0, 0) := 
sorry

end NUMINAMATH_GPT_imo1989_q3_l1296_129644


namespace NUMINAMATH_GPT_complement_M_l1296_129683

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_M :
  (U \ M) = {2, 3} := by
  sorry

end NUMINAMATH_GPT_complement_M_l1296_129683


namespace NUMINAMATH_GPT_value_of_x_l1296_129626

theorem value_of_x (z : ℤ) (h1 : z = 100) (y : ℤ) (h2 : y = z / 10) (x : ℤ) (h3 : x = y / 3) : 
  x = 10 / 3 := 
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_value_of_x_l1296_129626


namespace NUMINAMATH_GPT_probability_of_valid_quadrilateral_l1296_129656

-- Define a regular octagon
def regular_octagon_sides : ℕ := 8

-- Total number of ways to choose 4 sides from 8 sides
def total_ways_choose_four_sides : ℕ := Nat.choose 8 4

-- Number of ways to choose 4 adjacent sides (invalid)
def invalid_adjacent_ways : ℕ := 8

-- Number of ways to choose 4 sides with 3 adjacent unchosen sides (invalid)
def invalid_three_adjacent_unchosen_ways : ℕ := 8 * 3

-- Total number of invalid ways
def total_invalid_ways : ℕ := invalid_adjacent_ways + invalid_three_adjacent_unchosen_ways

-- Total number of valid ways
def total_valid_ways : ℕ := total_ways_choose_four_sides - total_invalid_ways

-- Probability of forming a quadrilateral that contains the octagon
def probability_valid_quadrilateral : ℚ :=
  (total_valid_ways : ℚ) / (total_ways_choose_four_sides : ℚ)

-- Theorem statement
theorem probability_of_valid_quadrilateral :
  probability_valid_quadrilateral = 19 / 35 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_valid_quadrilateral_l1296_129656


namespace NUMINAMATH_GPT_valerie_laptop_purchase_l1296_129675

/-- Valerie wants to buy a new laptop priced at $800. She receives $100 dollars from her parents,
$60 dollars from her uncle, and $40 dollars from her siblings for her graduation.
She also makes $20 dollars each week from tutoring. How many weeks must she save 
her tutoring income, along with her graduation money, to buy the laptop? -/
theorem valerie_laptop_purchase :
  let price_of_laptop : ℕ := 800
  let graduation_money : ℕ := 100 + 60 + 40
  let weekly_tutoring_income : ℕ := 20
  let remaining_amount_needed : ℕ := price_of_laptop - graduation_money
  let weeks_needed := remaining_amount_needed / weekly_tutoring_income
  weeks_needed = 30 :=
by
  sorry

end NUMINAMATH_GPT_valerie_laptop_purchase_l1296_129675


namespace NUMINAMATH_GPT_find_k_value_l1296_129653

theorem find_k_value (k : ℝ) (x y : ℝ) (h1 : -3 * x + 2 * y = k) (h2 : 0.75 * x + y = 16) (h3 : x = -6) : k = 59 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_value_l1296_129653


namespace NUMINAMATH_GPT_cos_squared_identity_l1296_129652

theorem cos_squared_identity (α : ℝ) (h : Real.tan (α + π / 4) = 3 / 4) :
    Real.cos (π / 4 - α) ^ 2 = 9 / 25 := by
  sorry

end NUMINAMATH_GPT_cos_squared_identity_l1296_129652


namespace NUMINAMATH_GPT_total_vowels_written_l1296_129600

-- Define the vowels and the condition
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def num_vowels : Nat := vowels.length
def times_written : Nat := 2

-- Assert the total number of vowels written
theorem total_vowels_written : (num_vowels * times_written) = 10 := by
  sorry

end NUMINAMATH_GPT_total_vowels_written_l1296_129600


namespace NUMINAMATH_GPT_inequality_proof_l1296_129645

theorem inequality_proof (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1296_129645


namespace NUMINAMATH_GPT_never_prime_except_three_l1296_129631

theorem never_prime_except_three (p : ℕ) (hp : Nat.Prime p) :
  p^2 + 8 = 17 ∨ ∃ k, (k ≠ 1 ∧ k ≠ p^2 + 8 ∧ k ∣ (p^2 + 8)) := by
  sorry

end NUMINAMATH_GPT_never_prime_except_three_l1296_129631


namespace NUMINAMATH_GPT_no_prime_solutions_l1296_129604

theorem no_prime_solutions (p q : ℕ) (hp : p > 5) (hq : q > 5) (pp : Nat.Prime p) (pq : Nat.Prime q)
  (h : p * q ∣ (5^p - 2^p) * (5^q - 2^q)) : False :=
sorry

end NUMINAMATH_GPT_no_prime_solutions_l1296_129604


namespace NUMINAMATH_GPT_david_profit_l1296_129611

theorem david_profit (weight : ℕ) (cost sell_price : ℝ) (h_weight : weight = 50) (h_cost : cost = 50) (h_sell_price : sell_price = 1.20) : 
  sell_price * weight - cost = 10 :=
by sorry

end NUMINAMATH_GPT_david_profit_l1296_129611


namespace NUMINAMATH_GPT_part1_part2_part3_l1296_129609

-- Part 1
theorem part1 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : (x + y) * (y + z) * (z + x) ≥ 8 * x * y * z :=
sorry

-- Part 2
theorem part2 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x :=
sorry

-- Part 3
theorem part3 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : x ^ x * y ^ y * z ^ z ≥ (x * y * z) ^ ((x + y + z) / 3) :=
sorry

#print axioms part1
#print axioms part2
#print axioms part3

end NUMINAMATH_GPT_part1_part2_part3_l1296_129609


namespace NUMINAMATH_GPT_video_minutes_per_week_l1296_129657

theorem video_minutes_per_week
  (daily_videos : ℕ := 3)
  (short_video_length : ℕ := 2)
  (long_video_multiplier : ℕ := 6)
  (days_in_week : ℕ := 7) :
  (2 * short_video_length + long_video_multiplier * short_video_length) * days_in_week = 112 := 
by 
  -- conditions
  let short_videos_per_day := 2
  let long_video_length := long_video_multiplier * short_video_length
  let daily_total := short_videos_per_day * short_video_length + long_video_length
  let weekly_total := daily_total * days_in_week
  -- proof
  sorry

end NUMINAMATH_GPT_video_minutes_per_week_l1296_129657


namespace NUMINAMATH_GPT_approximate_reading_l1296_129635

-- Define the given conditions
def arrow_location_between (a b : ℝ) : Prop := a < 42.3 ∧ 42.6 < b

-- Statement of the proof problem
theorem approximate_reading (a b : ℝ) (ha : arrow_location_between a b) :
  a = 42.3 :=
sorry

end NUMINAMATH_GPT_approximate_reading_l1296_129635


namespace NUMINAMATH_GPT_find_r_l1296_129651

theorem find_r (k r : ℝ) 
  (h1 : 7 = k * 3^r) 
  (h2 : 49 = k * 9^r) : 
  r = Real.log 7 / Real.log 3 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l1296_129651


namespace NUMINAMATH_GPT_sharks_in_Cape_May_August_l1296_129674

section
variable {D_J C_J D_A C_A : ℕ}

-- Given conditions
theorem sharks_in_Cape_May_August 
  (h1 : C_J = 2 * D_J) 
  (h2 : C_A = 5 + 3 * D_A) 
  (h3 : D_J = 23) 
  (h4 : D_A = D_J) : 
  C_A = 74 := 
by 
  -- Skipped the proof steps 
  sorry
end

end NUMINAMATH_GPT_sharks_in_Cape_May_August_l1296_129674


namespace NUMINAMATH_GPT_parabola_vertex_l1296_129606

theorem parabola_vertex (x y : ℝ) : 
  (∀ x y, y^2 - 8*y + 4*x = 12 → (x, y) = (7, 4)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_parabola_vertex_l1296_129606
