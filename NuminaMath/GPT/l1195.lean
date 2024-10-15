import Mathlib

namespace NUMINAMATH_GPT_rectangle_area_is_243_square_meters_l1195_119545

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end NUMINAMATH_GPT_rectangle_area_is_243_square_meters_l1195_119545


namespace NUMINAMATH_GPT_sum_of_m_n_l1195_119531

-- Define the setup for the problem
def side_length_of_larger_square := 3
def side_length_of_smaller_square := 1
def side_length_of_given_rectangle_l1 := 1
def side_length_of_given_rectangle_l2 := 3
def total_area_of_larger_square := side_length_of_larger_square * side_length_of_larger_square
def area_of_smaller_square := side_length_of_smaller_square * side_length_of_smaller_square
def area_of_given_rectangle := side_length_of_given_rectangle_l1 * side_length_of_given_rectangle_l2

-- Define the variable for the area of rectangle R
def area_of_R := total_area_of_larger_square - (area_of_smaller_square + area_of_given_rectangle)

-- Given the problem statement, we need to find m and n such that the area of R is m/n.
def m := 5
def n := 1

-- We need to prove that m + n = 6 given these conditions
theorem sum_of_m_n : m + n = 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_m_n_l1195_119531


namespace NUMINAMATH_GPT_largest_unattainable_sum_l1195_119599

noncomputable def largestUnattainableSum (n : ℕ) : ℕ :=
  12 * n^2 + 8 * n - 1

theorem largest_unattainable_sum (n : ℕ) :
  ∀ s, (¬∃ a b c d, s = (a * (6 * n + 1) + b * (6 * n + 3) + c * (6 * n + 5) + d * (6 * n + 7)))
  ↔ s > largestUnattainableSum n := by
  sorry

end NUMINAMATH_GPT_largest_unattainable_sum_l1195_119599


namespace NUMINAMATH_GPT_Joey_SAT_Weeks_l1195_119516

theorem Joey_SAT_Weeks
    (hours_per_night : ℕ) (nights_per_week : ℕ)
    (hours_per_weekend_day : ℕ) (days_per_weekend : ℕ)
    (total_hours : ℕ) (weekly_hours : ℕ) (weeks : ℕ)
    (h1 : hours_per_night = 2) (h2 : nights_per_week = 5)
    (h3 : hours_per_weekend_day = 3) (h4 : days_per_weekend = 2)
    (h5 : total_hours = 96) (h6 : weekly_hours = 16)
    (h7 : weekly_hours = (hours_per_night * nights_per_week) + (hours_per_weekend_day * days_per_weekend)) :
  weeks = total_hours / weekly_hours :=
sorry

end NUMINAMATH_GPT_Joey_SAT_Weeks_l1195_119516


namespace NUMINAMATH_GPT_exists_natural_sum_of_squares_l1195_119582

theorem exists_natural_sum_of_squares : ∃ n : ℕ, n^2 = 0^2 + 7^2 + 24^2 + 312^2 + 48984^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_exists_natural_sum_of_squares_l1195_119582


namespace NUMINAMATH_GPT_length_of_each_piece_l1195_119518

-- Definitions based on conditions
def total_length : ℝ := 42.5
def number_of_pieces : ℝ := 50

-- The statement that we need to prove
theorem length_of_each_piece (h1 : total_length = 42.5) (h2 : number_of_pieces = 50) : 
  total_length / number_of_pieces = 0.85 := 
by
  sorry

end NUMINAMATH_GPT_length_of_each_piece_l1195_119518


namespace NUMINAMATH_GPT_lucas_1500th_day_is_sunday_l1195_119573

def days_in_week : ℕ := 7

def start_day : ℕ := 5  -- 0: Monday, 1: Tuesday, ..., 5: Friday

def nth_day_of_life (n : ℕ) : ℕ :=
  (n - 1 + start_day) % days_in_week

theorem lucas_1500th_day_is_sunday : nth_day_of_life 1500 = 0 :=
by
  sorry

end NUMINAMATH_GPT_lucas_1500th_day_is_sunday_l1195_119573


namespace NUMINAMATH_GPT_band_weight_correct_l1195_119583

universe u

structure InstrumentGroup where
  count : ℕ
  weight_per_instrument : ℕ

def total_weight (ig : InstrumentGroup) : ℕ :=
  ig.count * ig.weight_per_instrument

def total_band_weight : ℕ :=
  (total_weight ⟨6, 5⟩) + (total_weight ⟨9, 5⟩) +
  (total_weight ⟨8, 10⟩) + (total_weight ⟨3, 20⟩) + (total_weight ⟨2, 15⟩)

theorem band_weight_correct : total_band_weight = 245 := by
  rfl

end NUMINAMATH_GPT_band_weight_correct_l1195_119583


namespace NUMINAMATH_GPT_first_discount_percentage_l1195_119588

theorem first_discount_percentage (x : ℝ) (h : 450 * (1 - x / 100) * 0.85 = 306) : x = 20 :=
sorry

end NUMINAMATH_GPT_first_discount_percentage_l1195_119588


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1195_119556

-- Define the problem
theorem min_value_of_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y : ℝ, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (2 * a * x - b * y + 2 = 0)):
  ∃ (m : ℝ), m = 4 ∧ (1 / a + 1 / b) ≥ m :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1195_119556


namespace NUMINAMATH_GPT_dividend_value_l1195_119504

def dividend (divisor quotient remainder : ℝ) := (divisor * quotient) + remainder

theorem dividend_value :
  dividend 35.8 21.65 11.3 = 786.47 :=
by
  sorry

end NUMINAMATH_GPT_dividend_value_l1195_119504


namespace NUMINAMATH_GPT_annual_interest_rate_l1195_119547

theorem annual_interest_rate 
  (P A : ℝ) 
  (hP : P = 136) 
  (hA : A = 150) 
  : (A - P) / P = 0.10 :=
by sorry

end NUMINAMATH_GPT_annual_interest_rate_l1195_119547


namespace NUMINAMATH_GPT_even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l1195_119541

-- Definitions for the conditions in Lean 4
def regular_n_gon (n : ℕ) := true -- Dummy definition; actual geometric properties not needed for statement

def connected_path_visits_each_vertex_once (n : ℕ) := true -- Dummy definition; actual path properties not needed for statement

def parallel_pair (i j p q : ℕ) (n : ℕ) : Prop := (i + j) % n = (p + q) % n

-- Statements for part (a) and (b)

theorem even_n_has_parallel_pair (n : ℕ) (h_even : n % 2 = 0) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n := 
sorry

theorem odd_n_cannot_have_exactly_one_parallel_pair (n : ℕ) (h_odd : n % 2 = 1) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ¬∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n ∧ 
  (∀ (i' j' p' q' : ℕ), (i' ≠ p' ∨ j' ≠ q') → ¬parallel_pair i' j' p' q' n) := 
sorry

end NUMINAMATH_GPT_even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l1195_119541


namespace NUMINAMATH_GPT_simplify_fraction_l1195_119567

theorem simplify_fraction (x y z : ℝ) (h : x + 2 * y + z ≠ 0) :
  (x^2 + y^2 - 4 * z^2 + 2 * x * y) / (x^2 + 4 * y^2 - z^2 + 2 * x * z) = (x + y - 2 * z) / (x + z - 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1195_119567


namespace NUMINAMATH_GPT_total_pies_l1195_119579

theorem total_pies {team1 team2 team3 total_pies : ℕ} 
  (h1 : team1 = 235) 
  (h2 : team2 = 275) 
  (h3 : team3 = 240) 
  (h4 : total_pies = team1 + team2 + team3) : 
  total_pies = 750 := by 
  sorry

end NUMINAMATH_GPT_total_pies_l1195_119579


namespace NUMINAMATH_GPT_borrowed_nickels_l1195_119560

-- Define the initial and remaining number of nickels
def initial_nickels : ℕ := 87
def remaining_nickels : ℕ := 12

-- Prove that the number of nickels borrowed is 75
theorem borrowed_nickels : initial_nickels - remaining_nickels = 75 := by
  sorry

end NUMINAMATH_GPT_borrowed_nickels_l1195_119560


namespace NUMINAMATH_GPT_greatest_C_inequality_l1195_119526

theorem greatest_C_inequality (α x y z : ℝ) (hα_pos : 0 < α) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_xyz_sum : x * y + y * z + z * x = α) : 
  16 ≤ (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) / (x / z + z / x + 2) :=
sorry

end NUMINAMATH_GPT_greatest_C_inequality_l1195_119526


namespace NUMINAMATH_GPT_calculate_T1_T2_l1195_119587

def triangle (a b c : ℤ) : ℤ := a + b - 2 * c

def T1 := triangle 3 4 5
def T2 := triangle 6 8 2

theorem calculate_T1_T2 : 2 * T1 + 3 * T2 = 24 :=
  by
    sorry

end NUMINAMATH_GPT_calculate_T1_T2_l1195_119587


namespace NUMINAMATH_GPT_cos_C_in_triangle_l1195_119517

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end NUMINAMATH_GPT_cos_C_in_triangle_l1195_119517


namespace NUMINAMATH_GPT_age_of_15th_student_l1195_119594

theorem age_of_15th_student (avg_age_15 : ℕ) (avg_age_6 : ℕ) (avg_age_8 : ℕ) (num_students_15 : ℕ) (num_students_6 : ℕ) (num_students_8 : ℕ) 
  (h_avg_15 : avg_age_15 = 15) 
  (h_avg_6 : avg_age_6 = 14) 
  (h_avg_8 : avg_age_8 = 16) 
  (h_num_15 : num_students_15 = 15) 
  (h_num_6 : num_students_6 = 6) 
  (h_num_8 : num_students_8 = 8) : 
  ∃ age_15th_student : ℕ, age_15th_student = 13 := 
by
  sorry


end NUMINAMATH_GPT_age_of_15th_student_l1195_119594


namespace NUMINAMATH_GPT_sequence_general_term_l1195_119506

-- Given a sequence {a_n} whose sum of the first n terms S_n = 2a_n - 1,
-- prove that the general formula for the n-th term a_n is 2^(n-1).

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
    (h₁ : ∀ n : ℕ, S n = 2 * a n - 1)
    (h₂ : S 1 = 1) : ∀ n : ℕ, a (n + 1) = 2 ^ n :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1195_119506


namespace NUMINAMATH_GPT_maximum_area_of_rectangular_farm_l1195_119537

theorem maximum_area_of_rectangular_farm :
  ∃ l w : ℕ, 2 * (l + w) = 160 ∧ l * w = 1600 :=
by
  sorry

end NUMINAMATH_GPT_maximum_area_of_rectangular_farm_l1195_119537


namespace NUMINAMATH_GPT_complex_square_l1195_119550

theorem complex_square (a b : ℝ) (i : ℂ) (h1 : a + b * i - 2 * i = 2 - b * i) : 
  (a + b * i) ^ 2 = 3 + 4 * i := 
by {
  -- Proof steps skipped (using sorry to indicate proof is required)
  sorry
}

end NUMINAMATH_GPT_complex_square_l1195_119550


namespace NUMINAMATH_GPT_sum_of_vertices_l1195_119512

theorem sum_of_vertices (n : ℕ) (h1 : 6 * n + 12 * n = 216) : 8 * n = 96 :=
by
  -- Proof is omitted intentionally
  sorry

end NUMINAMATH_GPT_sum_of_vertices_l1195_119512


namespace NUMINAMATH_GPT_midpoint_trajectory_of_moving_point_l1195_119546

/-- Given a fixed point A (4, -3) and a moving point B on the circle (x+1)^2 + y^2 = 4, prove that 
    the equation of the trajectory of the midpoint M of the line segment AB is 
    (x - 3/2)^2 + (y + 3/2)^2 = 1. -/
theorem midpoint_trajectory_of_moving_point {x y : ℝ} :
  (∃ (B : ℝ × ℝ), (B.1 + 1)^2 + B.2^2 = 4 ∧ 
    (x, y) = ((B.1 + 4) / 2, (B.2 - 3) / 2)) →
  (x - 3/2)^2 + (y + 3/2)^2 = 1 :=
by sorry

end NUMINAMATH_GPT_midpoint_trajectory_of_moving_point_l1195_119546


namespace NUMINAMATH_GPT_artifacts_per_wing_l1195_119551

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end NUMINAMATH_GPT_artifacts_per_wing_l1195_119551


namespace NUMINAMATH_GPT_arithmetic_sequence_a2_a8_l1195_119509

theorem arithmetic_sequence_a2_a8 (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : a 3 + a 4 + a 5 + a 6 + a 7 = 450) :
  a 2 + a 8 = 180 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a2_a8_l1195_119509


namespace NUMINAMATH_GPT_units_digit_of_3_pow_2009_l1195_119571

noncomputable def units_digit (n : ℕ) : ℕ :=
  if n % 4 = 1 then 3
  else if n % 4 = 2 then 9
  else if n % 4 = 3 then 7
  else 1

theorem units_digit_of_3_pow_2009 : units_digit (2009) = 3 :=
by
  -- Skipping the proof as instructed
  sorry

end NUMINAMATH_GPT_units_digit_of_3_pow_2009_l1195_119571


namespace NUMINAMATH_GPT_problem_eq_l1195_119555

theorem problem_eq : 
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → y = x / (x + 1) → (x - y + 4 * x * y) / (x * y) = 5 :=
by
  intros x y hx hnz hyxy
  sorry

end NUMINAMATH_GPT_problem_eq_l1195_119555


namespace NUMINAMATH_GPT_gcf_48_160_120_l1195_119548

theorem gcf_48_160_120 : Nat.gcd (Nat.gcd 48 160) 120 = 8 := by
  sorry

end NUMINAMATH_GPT_gcf_48_160_120_l1195_119548


namespace NUMINAMATH_GPT_prime_between_30_and_40_has_remainder_7_l1195_119574

theorem prime_between_30_and_40_has_remainder_7 (p : ℕ) 
  (h_prime : Nat.Prime p) 
  (h_interval : 30 < p ∧ p < 40) 
  (h_mod : p % 9 = 7) : 
  p = 34 := 
sorry

end NUMINAMATH_GPT_prime_between_30_and_40_has_remainder_7_l1195_119574


namespace NUMINAMATH_GPT_early_finish_hours_l1195_119540

theorem early_finish_hours 
  (h : Nat) 
  (total_customers : Nat) 
  (num_workers : Nat := 3)
  (service_rate : Nat := 7) 
  (full_hours : Nat := 8)
  (total_customers_served : total_customers = 154) 
  (two_workers_hours : Nat := 2 * full_hours * service_rate) 
  (early_worker_customers : Nat := h * service_rate)
  (total_service : total_customers = two_workers_hours + early_worker_customers) : 
  h = 6 :=
by
  sorry

end NUMINAMATH_GPT_early_finish_hours_l1195_119540


namespace NUMINAMATH_GPT_postage_problem_l1195_119507

theorem postage_problem (n : ℕ) (h_positive : n > 0) (h_postage : ∀ k, k ∈ List.range 121 → ∃ a b c : ℕ, 6 * a + n * b + (n + 2) * c = k) :
  6 * n * (n + 2) - (6 + n + (n + 2)) = 120 → n = 8 := 
by
  sorry

end NUMINAMATH_GPT_postage_problem_l1195_119507


namespace NUMINAMATH_GPT_remainder_when_divided_by_2000_l1195_119521

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_2000_l1195_119521


namespace NUMINAMATH_GPT_market_value_calculation_l1195_119501

variables (annual_dividend_per_share face_value yield market_value : ℝ)

axiom annual_dividend_definition : annual_dividend_per_share = 0.09 * face_value
axiom face_value_definition : face_value = 100
axiom yield_definition : yield = 0.25

theorem market_value_calculation (annual_dividend_per_share face_value yield market_value : ℝ) 
  (h1: annual_dividend_per_share = 0.09 * face_value)
  (h2: face_value = 100)
  (h3: yield = 0.25):
  market_value = annual_dividend_per_share / yield :=
sorry

end NUMINAMATH_GPT_market_value_calculation_l1195_119501


namespace NUMINAMATH_GPT_monthly_average_decrease_rate_l1195_119522

-- Conditions
def january_production : Float := 1.6 * 10^6
def march_production : Float := 0.9 * 10^6
def rate_decrease : Float := 0.25

-- Proof Statement: we need to prove that the monthly average decrease rate x = 0.25 satisfies the given condition
theorem monthly_average_decrease_rate :
  january_production * (1 - rate_decrease) * (1 - rate_decrease) = march_production := by
  sorry

end NUMINAMATH_GPT_monthly_average_decrease_rate_l1195_119522


namespace NUMINAMATH_GPT_polynomial_evaluation_l1195_119544

theorem polynomial_evaluation (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l1195_119544


namespace NUMINAMATH_GPT_maximum_ab_l1195_119519

open Real

theorem maximum_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 6 * a + 5 * b = 75) :
  ab ≤ 46.875 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_maximum_ab_l1195_119519


namespace NUMINAMATH_GPT_range_of_f_l1195_119592

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

theorem range_of_f 
  (x : ℝ) : f (x - 1) + f (x + 1) > 0 ↔ x ∈ Set.Ioi 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l1195_119592


namespace NUMINAMATH_GPT_wade_average_points_per_game_l1195_119584

variable (W : ℝ)

def teammates_average_points_per_game : ℝ := 40

def total_team_points_after_5_games : ℝ := 300

theorem wade_average_points_per_game :
  teammates_average_points_per_game * 5 + W * 5 = total_team_points_after_5_games →
  W = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_wade_average_points_per_game_l1195_119584


namespace NUMINAMATH_GPT_original_price_l1195_119598

variable (q r : ℝ)

theorem original_price (x : ℝ) (h : x * (1 + q / 100) * (1 - r / 100) = 1) :
  x = 1 / ((1 + q / 100) * (1 - r / 100)) :=
sorry

end NUMINAMATH_GPT_original_price_l1195_119598


namespace NUMINAMATH_GPT_divisible_by_six_l1195_119561

theorem divisible_by_six (n a b : ℕ) (h1 : 2^n = 10 * a + b) (h2 : n > 3) (h3 : b > 0) (h4 : b < 10) : 6 ∣ (a * b) := 
sorry

end NUMINAMATH_GPT_divisible_by_six_l1195_119561


namespace NUMINAMATH_GPT_algebra_expression_l1195_119558

theorem algebra_expression (a b : ℝ) (h : a - b = 3) : 1 + a - b = 4 :=
sorry

end NUMINAMATH_GPT_algebra_expression_l1195_119558


namespace NUMINAMATH_GPT_find_polynomial_value_l1195_119502

theorem find_polynomial_value (x y : ℝ) 
  (h1 : 3 * x + y = 12) 
  (h2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_polynomial_value_l1195_119502


namespace NUMINAMATH_GPT_Carlos_candy_share_l1195_119572

theorem Carlos_candy_share (total_candy : ℚ) (num_piles : ℕ) (piles_for_Carlos : ℕ)
  (h_total_candy : total_candy = 75 / 7)
  (h_num_piles : num_piles = 5)
  (h_piles_for_Carlos : piles_for_Carlos = 2) :
  (piles_for_Carlos * (total_candy / num_piles) = 30 / 7) :=
by
  sorry

end NUMINAMATH_GPT_Carlos_candy_share_l1195_119572


namespace NUMINAMATH_GPT_andre_total_payment_l1195_119591

def treadmill_initial_price : ℝ := 1350
def treadmill_discount : ℝ := 0.30
def plate_initial_price : ℝ := 60
def plate_discount : ℝ := 0.15
def plate_quantity : ℝ := 2

theorem andre_total_payment :
  let treadmill_discounted_price := treadmill_initial_price * (1 - treadmill_discount)
  let plates_total_initial_price := plate_quantity * plate_initial_price
  let plates_discounted_price := plates_total_initial_price * (1 - plate_discount)
  treadmill_discounted_price + plates_discounted_price = 1047 := 
by
  sorry

end NUMINAMATH_GPT_andre_total_payment_l1195_119591


namespace NUMINAMATH_GPT_expression_equals_5776_l1195_119553

-- Define constants used in the problem
def a : ℕ := 476
def b : ℕ := 424
def c : ℕ := 4

-- Define the expression using the constants
def expression : ℕ := (a + b) ^ 2 - c * a * b

-- The target proof statement
theorem expression_equals_5776 : expression = 5776 := by
  sorry

end NUMINAMATH_GPT_expression_equals_5776_l1195_119553


namespace NUMINAMATH_GPT_dice_roll_probability_bounds_l1195_119525

noncomputable def dice_roll_probability : Prop :=
  let n := 80
  let p := (1 : ℝ) / 6
  let q := 1 - p
  let epsilon := 2.58 / 24
  let lower_bound := (p - epsilon) * n
  let upper_bound := (p + epsilon) * n
  5 ≤ lower_bound ∧ upper_bound ≤ 22

theorem dice_roll_probability_bounds :
  dice_roll_probability :=
sorry

end NUMINAMATH_GPT_dice_roll_probability_bounds_l1195_119525


namespace NUMINAMATH_GPT_find_common_ratio_l1195_119536

variable {α : Type*} [LinearOrderedField α] [NormedLinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop := ∀ n, a (n+1) = q * a n

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop := ∀ n, S n = (Finset.range n).sum a

theorem find_common_ratio
  (a : ℕ → α)
  (S : ℕ → α)
  (q : α)
  (pos_terms : ∀ n, 0 < a n)
  (geometric_seq : geometric_sequence a q)
  (sum_eq : sum_first_n_terms a S)
  (eqn : S 1 + 2 * S 5 = 3 * S 3) :
  q = (2:α)^(3 / 2) / 2^(3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l1195_119536


namespace NUMINAMATH_GPT_divides_sequence_l1195_119510

theorem divides_sequence (a : ℕ → ℕ) (n k: ℕ) (h0 : a 0 = 0) (h1 : a 1 = 1) 
  (hrec : ∀ m, a (m + 2) = 2 * a (m + 1) + a m) :
  (2^k ∣ a n) ↔ (2^k ∣ n) :=
sorry

end NUMINAMATH_GPT_divides_sequence_l1195_119510


namespace NUMINAMATH_GPT_min_birthdays_on_wednesday_l1195_119549

theorem min_birthdays_on_wednesday 
  (W X : ℕ) 
  (h1 : W + 6 * X = 50) 
  (h2 : W > X) : 
  W = 8 := 
sorry

end NUMINAMATH_GPT_min_birthdays_on_wednesday_l1195_119549


namespace NUMINAMATH_GPT_find_divisor_l1195_119576

theorem find_divisor (remainder quotient dividend divisor : ℕ) 
  (h_rem : remainder = 8)
  (h_quot : quotient = 43)
  (h_div : dividend = 997)
  (h_eq : dividend = divisor * quotient + remainder) : 
  divisor = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1195_119576


namespace NUMINAMATH_GPT_correct_81st_in_set_s_l1195_119566

def is_in_set_s (x : ℕ) : Prop :=
  ∃ n : ℕ, x = 8 * n + 5

noncomputable def find_81st_in_set_s : ℕ :=
  8 * 80 + 5

theorem correct_81st_in_set_s : find_81st_in_set_s = 645 := by
  sorry

end NUMINAMATH_GPT_correct_81st_in_set_s_l1195_119566


namespace NUMINAMATH_GPT_number_of_remaining_red_points_l1195_119543

/-- 
Given a grid where the distance between any two adjacent points in a row or column is 1,
and any green point can turn points within a distance of no more than 1 into green every second.
Initial state of the grid is given. Determine the number of red points after 4 seconds.
-/
def remaining_red_points_after_4_seconds (initial_state : List (List Bool)) : Nat := 
41 -- assume this is the computed number after applying the infection rule for 4 seconds

theorem number_of_remaining_red_points (initial_state : List (List Bool)) :
  remaining_red_points_after_4_seconds initial_state = 41 := 
sorry

end NUMINAMATH_GPT_number_of_remaining_red_points_l1195_119543


namespace NUMINAMATH_GPT_problem_statement_l1195_119568

variable {f : ℝ → ℝ}

-- Condition 1: The function f satisfies (x - 1)f'(x) ≤ 0
def cond1 (f : ℝ → ℝ) : Prop := ∀ x, (x - 1) * (deriv f x) ≤ 0

-- Condition 2: The function f satisfies f(-x) = f(2 + x)
def cond2 (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f (2 + x)

theorem problem_statement (f : ℝ → ℝ) (x₁ x₂ : ℝ)
  (h_cond1 : cond1 f)
  (h_cond2 : cond2 f)
  (h_dist : abs (x₁ - 1) < abs (x₂ - 1)) :
  f (2 - x₁) > f (2 - x₂) :=
sorry

end NUMINAMATH_GPT_problem_statement_l1195_119568


namespace NUMINAMATH_GPT_sufficient_conditions_for_positive_product_l1195_119542

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1) → a * b > 0 :=
by sorry

end NUMINAMATH_GPT_sufficient_conditions_for_positive_product_l1195_119542


namespace NUMINAMATH_GPT_sum_of_squares_l1195_119585

def satisfies_conditions (x y z : ℕ) : Prop :=
  x + y + z = 24 ∧
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10

theorem sum_of_squares (x y z : ℕ) (h : satisfies_conditions x y z) :
  ∀ (x y z : ℕ), x + y + z = 24 ∧ Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 →
  x^2 + y^2 + z^2 = 216 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l1195_119585


namespace NUMINAMATH_GPT_max_n_l1195_119563

noncomputable def seq_a (n : ℕ) : ℤ := 3 * n - 1

noncomputable def seq_b (n : ℕ) : ℤ := 2 * n - 3

noncomputable def sum_T (n : ℕ) : ℤ := n * (3 * n + 1) / 2

noncomputable def sum_S (n : ℕ) : ℤ := n^2 - 2 * n

theorem max_n (n : ℕ) :
  ∃ n_max : ℕ, T_n < 20 * seq_b n ∧ (∀ m : ℕ, m > n_max → T_n ≥ 20 * seq_b n) :=
  sorry

end NUMINAMATH_GPT_max_n_l1195_119563


namespace NUMINAMATH_GPT_evaluate_expression_l1195_119534

theorem evaluate_expression (x : ℤ) (h : x = 5) : 
  3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2 = 1457 := 
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1195_119534


namespace NUMINAMATH_GPT_successive_discounts_eq_single_discount_l1195_119527

theorem successive_discounts_eq_single_discount :
  ∀ (x : ℝ), (1 - 0.15) * (1 - 0.25) * x = (1 - 0.3625) * x :=
by
  intro x
  sorry

end NUMINAMATH_GPT_successive_discounts_eq_single_discount_l1195_119527


namespace NUMINAMATH_GPT_speed_ratio_l1195_119524

theorem speed_ratio (a b v1 v2 S : ℝ) (h1 : S = a * (v1 + v2)) (h2 : S = b * (v1 - v2)) (h3 : a ≠ b) : 
  v1 / v2 = (a + b) / (b - a) :=
by
  -- proof skipped
  sorry

end NUMINAMATH_GPT_speed_ratio_l1195_119524


namespace NUMINAMATH_GPT_polynomial_coeff_sum_l1195_119596

theorem polynomial_coeff_sum {a_0 a_1 a_2 a_3 a_4 a_5 : ℝ} :
  (2 * (x : ℝ) - 3)^5 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_polynomial_coeff_sum_l1195_119596


namespace NUMINAMATH_GPT_inequality_holds_l1195_119569

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (y * z)) + (y^3 / (z * x)) + (z^3 / (x * y)) ≥ x + y + z :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l1195_119569


namespace NUMINAMATH_GPT_most_stable_yield_l1195_119500

theorem most_stable_yield (S_A S_B S_C S_D : ℝ)
  (h₁ : S_A = 3.6)
  (h₂ : S_B = 2.89)
  (h₃ : S_C = 13.4)
  (h₄ : S_D = 20.14) : 
  S_B < S_A ∧ S_B < S_C ∧ S_B < S_D :=
by {
  sorry -- Proof skipped as per instructions
}

end NUMINAMATH_GPT_most_stable_yield_l1195_119500


namespace NUMINAMATH_GPT_square_divisibility_l1195_119511

theorem square_divisibility (n : ℤ) : n^2 % 4 = 0 ∨ n^2 % 4 = 1 := sorry

end NUMINAMATH_GPT_square_divisibility_l1195_119511


namespace NUMINAMATH_GPT_minor_premise_l1195_119532

-- Definitions
def Rectangle : Type := sorry
def Square : Type := sorry
def Parallelogram : Type := sorry

axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle
axiom square_is_parallelogram : Square → Parallelogram

-- Problem statement
theorem minor_premise : ∀ (S : Square), ∃ (R : Rectangle), square_is_rectangle S = R :=
by
  sorry

end NUMINAMATH_GPT_minor_premise_l1195_119532


namespace NUMINAMATH_GPT_average_price_of_towels_l1195_119557

-- Definitions based on conditions
def cost_towel1 : ℕ := 3 * 100
def cost_towel2 : ℕ := 5 * 150
def cost_towel3 : ℕ := 2 * 600
def total_cost : ℕ := cost_towel1 + cost_towel2 + cost_towel3
def total_towels : ℕ := 3 + 5 + 2
def average_price : ℕ := total_cost / total_towels

-- Statement to be proved
theorem average_price_of_towels :
  average_price = 225 :=
by
  sorry

end NUMINAMATH_GPT_average_price_of_towels_l1195_119557


namespace NUMINAMATH_GPT_min_ab_l1195_119590

theorem min_ab {a b : ℝ} (h1 : (a^2) * (-b) + (a^2 + 1) = 0) : |a * b| = 2 :=
sorry

end NUMINAMATH_GPT_min_ab_l1195_119590


namespace NUMINAMATH_GPT_days_from_friday_l1195_119578

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end NUMINAMATH_GPT_days_from_friday_l1195_119578


namespace NUMINAMATH_GPT_gcd_lcm_product_l1195_119530

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end NUMINAMATH_GPT_gcd_lcm_product_l1195_119530


namespace NUMINAMATH_GPT_probability_not_touch_outer_edge_l1195_119520

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_touch_outer_edge_l1195_119520


namespace NUMINAMATH_GPT_product_ineq_l1195_119564

-- Define the relevant elements and conditions
variables (a b : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ)

-- Assumptions based on the conditions provided
variables (h₀ : a > 0) (h₁ : b > 0)
variables (h₂ : a + b = 1)
variables (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₃ > 0) (h₆ : x₄ > 0) (h₇ : x₅ > 0)
variables (h₈ : x₁ * x₂ * x₃ * x₄ * x₅ = 1)

-- The theorem statement to be proved
theorem product_ineq : (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 :=
sorry

end NUMINAMATH_GPT_product_ineq_l1195_119564


namespace NUMINAMATH_GPT_min_value_expression_l1195_119505

noncomputable def expression (x y : ℝ) := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

theorem min_value_expression : ∀ x y : ℝ, expression x y ≥ -14 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1195_119505


namespace NUMINAMATH_GPT_neg_pi_lt_neg_314_l1195_119535

theorem neg_pi_lt_neg_314 (h : Real.pi > 3.14) : -Real.pi < -3.14 :=
sorry

end NUMINAMATH_GPT_neg_pi_lt_neg_314_l1195_119535


namespace NUMINAMATH_GPT_number_subtract_four_l1195_119580

theorem number_subtract_four (x : ℤ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end NUMINAMATH_GPT_number_subtract_four_l1195_119580


namespace NUMINAMATH_GPT_ellipse_equation_correct_l1195_119559

noncomputable def ellipse_equation_proof : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), (x - 2 * y + 4 = 0) ∧ (∃ (f : ℝ × ℝ), f = (-4, 0)) ∧ (∃ (v : ℝ × ℝ), v = (0, 2)) → 
    (x^2 / (a^2) + y^2 / (b^2) = 1 → x^2 / 20 + y^2 / 4 = 1))

theorem ellipse_equation_correct : ellipse_equation_proof :=
  sorry

end NUMINAMATH_GPT_ellipse_equation_correct_l1195_119559


namespace NUMINAMATH_GPT_total_nominal_income_l1195_119528

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end NUMINAMATH_GPT_total_nominal_income_l1195_119528


namespace NUMINAMATH_GPT_probability_of_neither_is_correct_l1195_119552

-- Definitions of the given conditions
def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_cake_and_muffin_buyers : ℕ := 19

-- Define the probability calculation function
def probability_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  let buyers_neither := total - (cake + muffin - both)
  (buyers_neither : ℚ) / (total : ℚ)

-- State the main theorem to ensure it is equivalent to our mathematical problem
theorem probability_of_neither_is_correct :
  probability_neither total_buyers cake_buyers muffin_buyers both_cake_and_muffin_buyers = 0.29 := 
sorry

end NUMINAMATH_GPT_probability_of_neither_is_correct_l1195_119552


namespace NUMINAMATH_GPT_solve_wire_cut_problem_l1195_119565

def wire_cut_problem : Prop :=
  ∃ x y : ℝ, x + y = 35 ∧ y = (2/5) * x ∧ x = 25

theorem solve_wire_cut_problem : wire_cut_problem := by
  sorry

end NUMINAMATH_GPT_solve_wire_cut_problem_l1195_119565


namespace NUMINAMATH_GPT_kevin_ends_with_cards_l1195_119589

def cards_found : ℝ := 47.0
def cards_lost : ℝ := 7.0

theorem kevin_ends_with_cards : cards_found - cards_lost = 40.0 := by
  sorry

end NUMINAMATH_GPT_kevin_ends_with_cards_l1195_119589


namespace NUMINAMATH_GPT_volume_expansion_rate_l1195_119539

theorem volume_expansion_rate (R m : ℝ) (h1 : R = 1) (h2 : (4 * π * (m^3 - 1) / 3) / (m - 1) = 28 * π / 3) : m = 2 :=
sorry

end NUMINAMATH_GPT_volume_expansion_rate_l1195_119539


namespace NUMINAMATH_GPT_positive_integer_count_l1195_119586

/-
  Prove that the number of positive integers \( n \) for which \( \frac{n(n+1)}{2} \) divides \( 30n \) is 11.
-/

theorem positive_integer_count (n : ℕ) :
  (∃ k : ℕ, k > 0 ∧ k ≤ 11 ∧ (2 * 30 * n) % (n * (n + 1)) = 0) :=
sorry

end NUMINAMATH_GPT_positive_integer_count_l1195_119586


namespace NUMINAMATH_GPT_unique_triple_property_l1195_119503

theorem unique_triple_property (a b c : ℕ) (h1 : a ∣ b * c + 1) (h2 : b ∣ a * c + 1) (h3 : c ∣ a * b + 1) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a = 2 ∧ b = 3 ∧ c = 7) :=
by
  sorry

end NUMINAMATH_GPT_unique_triple_property_l1195_119503


namespace NUMINAMATH_GPT_complex_modulus_inequality_l1195_119570

theorem complex_modulus_inequality (z : ℂ) : (‖z‖ ^ 2 + 2 * ‖z - 1‖) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_complex_modulus_inequality_l1195_119570


namespace NUMINAMATH_GPT_range_of_m_l1195_119554

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x-1)^2 < m^2 → |1 - (x-1)/3| < 2) → (abs m ≤ 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1195_119554


namespace NUMINAMATH_GPT_total_pencils_l1195_119513

variable (C Y M D : ℕ)

-- Conditions
def cheryl_has_thrice_as_cyrus (h1 : C = 3 * Y) : Prop := true
def madeline_has_half_of_cheryl (h2 : M = 63 ∧ C = 2 * M) : Prop := true
def daniel_has_25_percent_of_total (h3 : D = (C + Y + M) / 4) : Prop := true

-- Total number of pencils for all four
theorem total_pencils (h1 : C = 3 * Y) (h2 : M = 63 ∧ C = 2 * M) (h3 : D = (C + Y + M) / 4) :
  C + Y + M + D = 289 :=
by { sorry }

end NUMINAMATH_GPT_total_pencils_l1195_119513


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1195_119529

variable (a : ℝ) (x : ℝ)

def inequality_holds_for_all_real_numbers (a : ℝ) : Prop :=
    ∀ x : ℝ, (a * x^2 - a * x + 1 > 0)

theorem necessary_but_not_sufficient_condition :
  (0 < a ∧ a < 4) ↔
  (inequality_holds_for_all_real_numbers a) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1195_119529


namespace NUMINAMATH_GPT_bitcoin_donation_l1195_119538

theorem bitcoin_donation (x : ℝ) (h : 3 * (80 - x) / 2 - 10 = 80) : x = 20 :=
sorry

end NUMINAMATH_GPT_bitcoin_donation_l1195_119538


namespace NUMINAMATH_GPT_max_value_of_trig_expr_l1195_119514

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end NUMINAMATH_GPT_max_value_of_trig_expr_l1195_119514


namespace NUMINAMATH_GPT_remainder_division_l1195_119515

theorem remainder_division (N : ℤ) (hN : N % 899 = 63) : N % 29 = 5 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_division_l1195_119515


namespace NUMINAMATH_GPT_probability_chord_length_not_less_than_radius_l1195_119595

theorem probability_chord_length_not_less_than_radius
  (R : ℝ) (M N : ℝ) (h_circle : N = 2 * π * R) : 
  (∃ P : ℝ, P = 2 / 3) :=
sorry

end NUMINAMATH_GPT_probability_chord_length_not_less_than_radius_l1195_119595


namespace NUMINAMATH_GPT_min_value_a2b3c_l1195_119593

theorem min_value_a2b3c {m : ℝ} (hm : m > 0)
  (hineq : ∀ x : ℝ, |x + 1| + |2 * x - 1| ≥ m)
  {a b c : ℝ} (habc : a^2 + 2 * b^2 + 3 * c^2 = m) :
  a + 2 * b + 3 * c ≥ -3 :=
sorry

end NUMINAMATH_GPT_min_value_a2b3c_l1195_119593


namespace NUMINAMATH_GPT_last_digits_nn_periodic_l1195_119597

theorem last_digits_nn_periodic (n : ℕ) : 
  ∃ p > 0, ∀ k, (n + k * p)^(n + k * p) % 10 = n^n % 10 := 
sorry

end NUMINAMATH_GPT_last_digits_nn_periodic_l1195_119597


namespace NUMINAMATH_GPT_compound_interest_second_year_l1195_119575

theorem compound_interest_second_year
  (P : ℝ) (r : ℝ) (CI_3 : ℝ) (CI_2 : ℝ) 
  (h1 : r = 0.08) 
  (h2 : CI_3 = 1512)
  (h3 : CI_3 = CI_2 * (1 + r)) :
  CI_2 = 1400 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_compound_interest_second_year_l1195_119575


namespace NUMINAMATH_GPT_least_area_exists_l1195_119523

-- Definition of the problem conditions
def is_rectangle (l w : ℕ) : Prop :=
  2 * (l + w) = 120

def area (l w : ℕ) := l * w

-- Statement of the proof problem
theorem least_area_exists :
  ∃ (l w : ℕ), is_rectangle l w ∧ (∀ (l' w' : ℕ), is_rectangle l' w' → area l w ≤ area l' w') ∧ area l w = 59 :=
sorry

end NUMINAMATH_GPT_least_area_exists_l1195_119523


namespace NUMINAMATH_GPT_sum_of_coefficients_at_1_l1195_119533

def P (x : ℝ) := 2 * (4 * x^8 - 3 * x^5 + 9)
def Q (x : ℝ) := 9 * (x^6 + 2 * x^3 - 8)
def R (x : ℝ) := P x + Q x

theorem sum_of_coefficients_at_1 : R 1 = -25 := by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_at_1_l1195_119533


namespace NUMINAMATH_GPT_usual_time_is_25_l1195_119508

-- Definitions 
variables {S T : ℝ} (h1 : S * T = 5 / 4 * S * (T - 5))

-- Theorem statement
theorem usual_time_is_25 (h : S * T = 5 / 4 * S * (T - 5)) : T = 25 :=
by 
-- Using the assumption h, we'll derive that T = 25
sorry

end NUMINAMATH_GPT_usual_time_is_25_l1195_119508


namespace NUMINAMATH_GPT_part_1_part_2_l1195_119581

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

-- (Part 1): Prove the value of a
theorem part_1 (a : ℝ) (P : ℝ × ℝ) (hP : P = (a, -4)) :
  (∃ t : ℝ, ∃ t₂ : ℝ, t ≠ t₂ ∧ P.2 = (2 * t^3 - 3 * t^2 + 1) + (6 * t^2 - 6 * t) * (a - t)) →
  a = -1 ∨ a = 7 / 2 :=
sorry

-- (Part 2): Prove the range of k
noncomputable def g (x k : ℝ) : ℝ := k * x + 1 - Real.log x

noncomputable def h (x k : ℝ) : ℝ := min (f x) (g x k)

theorem part_2 (k : ℝ) :
  (∀ x > 0, h x k = 0 → (x = 1 ∨ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 k = 0 ∧ h x2 k = 0)) →
  0 < k ∧ k < 1 / Real.exp 2 :=
sorry

end NUMINAMATH_GPT_part_1_part_2_l1195_119581


namespace NUMINAMATH_GPT_cards_problem_l1195_119562

theorem cards_problem : 
  ∀ (cards people : ℕ),
  cards = 60 →
  people = 8 →
  ∃ fewer_people : ℕ,
  (∀ p: ℕ, p < people → (p < fewer_people → cards/people < 8)) ∧ 
  fewer_people = 4 := 
by 
  intros cards people h_cards h_people
  use 4
  sorry

end NUMINAMATH_GPT_cards_problem_l1195_119562


namespace NUMINAMATH_GPT_mike_scored_212_l1195_119577

variable {M : ℕ}

def passing_marks (max_marks : ℕ) : ℕ := (30 * max_marks) / 100

def mike_marks (passing_marks shortfall : ℕ) : ℕ := passing_marks - shortfall

theorem mike_scored_212 (max_marks : ℕ) (shortfall : ℕ)
  (h1 : max_marks = 790)
  (h2 : shortfall = 25)
  (h3 : M = mike_marks (passing_marks max_marks) shortfall) : 
  M = 212 := 
by 
  sorry

end NUMINAMATH_GPT_mike_scored_212_l1195_119577
