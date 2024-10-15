import Mathlib

namespace NUMINAMATH_GPT_largest_number_l1595_159596

theorem largest_number (a b c d : ℝ) (h1 : a = 1/2) (h2 : b = 0) (h3 : c = 1) (h4 : d = -9) :
  max (max a b) (max c d) = c :=
by
  sorry

end NUMINAMATH_GPT_largest_number_l1595_159596


namespace NUMINAMATH_GPT_jerry_age_l1595_159549

theorem jerry_age (M J : ℤ) (h1 : M = 16) (h2 : M = 2 * J - 8) : J = 12 :=
by
  sorry

end NUMINAMATH_GPT_jerry_age_l1595_159549


namespace NUMINAMATH_GPT_arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l1595_159505

theorem arcsin_one_half_eq_pi_over_six : Real.arcsin (1/2) = Real.pi/6 :=
by 
  sorry

theorem arccos_one_half_eq_pi_over_three : Real.arccos (1/2) = Real.pi/3 :=
by 
  sorry

end NUMINAMATH_GPT_arcsin_one_half_eq_pi_over_six_arccos_one_half_eq_pi_over_three_l1595_159505


namespace NUMINAMATH_GPT_find_k_l1595_159584

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (u v : V)

theorem find_k (h : ∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ u + t • (v - u) = k • u + (5 / 8) • v) :
  k = 3 / 8 := sorry

end NUMINAMATH_GPT_find_k_l1595_159584


namespace NUMINAMATH_GPT_smallest_integer_to_make_multiple_of_five_l1595_159504

theorem smallest_integer_to_make_multiple_of_five : 
  ∃ k: ℕ, 0 < k ∧ (726 + k) % 5 = 0 ∧ k = 4 := 
by
  use 4
  sorry

end NUMINAMATH_GPT_smallest_integer_to_make_multiple_of_five_l1595_159504


namespace NUMINAMATH_GPT_coordinates_of_point_P_l1595_159510

theorem coordinates_of_point_P {x y : ℝ} (hx : |x| = 2) (hy : y = 1 ∨ y = -1) (hxy : x < 0 ∧ y > 0) : 
  (x, y) = (-2, 1) := 
by 
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l1595_159510


namespace NUMINAMATH_GPT_percent_fair_hair_l1595_159534

theorem percent_fair_hair 
  (total_employees : ℕ) 
  (percent_women_fair_hair : ℝ) 
  (percent_fair_hair_women : ℝ)
  (total_women_fair_hair : ℕ)
  (total_fair_hair : ℕ)
  (h1 : percent_women_fair_hair = 30 / 100)
  (h2 : percent_fair_hair_women = 40 / 100)
  (h3 : total_women_fair_hair = percent_women_fair_hair * total_employees)
  (h4 : percent_fair_hair_women * total_fair_hair = total_women_fair_hair)
  : total_fair_hair = 75 / 100 * total_employees := 
by
  sorry

end NUMINAMATH_GPT_percent_fair_hair_l1595_159534


namespace NUMINAMATH_GPT_find_xy_l1595_159582

theorem find_xy (x y : ℝ) (h : |x^3 - 1/8| + Real.sqrt (y - 4) = 0) : x * y = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_xy_l1595_159582


namespace NUMINAMATH_GPT_tangent_line_through_point_l1595_159595

theorem tangent_line_through_point (x y : ℝ) (tangent f : ℝ → ℝ) (M : ℝ × ℝ) :
  M = (1, 1) →
  f x = x^3 + 1 →
  tangent x = 3 * x^2 →
  (∃ a b c : ℝ, a * x + b * y + c = 0 ∧ ∀ x0 y0 : ℝ, (y0 = f x0) → (y - y0 = tangent x0 * (x - x0))) ∧
  (x, y) = M →
  (a = 0 ∧ b = 1 ∧ c = -1) ∨ (a = 27 ∧ b = -4 ∧ c = -23) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_point_l1595_159595


namespace NUMINAMATH_GPT_circle_equation_l1595_159513

/-- Given a circle passing through points P(4, -2) and Q(-1, 3), and with the length of the segment 
intercepted by the circle on the y-axis as 4, prove that the standard equation of the circle
is (x-1)^2 + y^2 = 13 or (x-5)^2 + (y-4)^2 = 37 -/
theorem circle_equation {P Q : ℝ × ℝ} {a b k : ℝ} :
  P = (4, -2) ∧ Q = (-1, 3) ∧ k = 4 →
  (∃ (r : ℝ), (∀ y : ℝ, (b - y)^2 = r^2) ∧
    ((a - 1)^2 + b^2 = 13 ∨ (a - 5)^2 + (b - 4)^2 = 37)
  ) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1595_159513


namespace NUMINAMATH_GPT_g_g_x_has_exactly_4_distinct_real_roots_l1595_159563

noncomputable def g (d x : ℝ) : ℝ := x^2 + 8*x + d

theorem g_g_x_has_exactly_4_distinct_real_roots (d : ℝ) :
  (∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ g d (g d x1) = 0 ∧ g d (g d x2) = 0 ∧ g d (g d x3) = 0 ∧ g d (g d x4) = 0) ↔ d < 4 := by {
  sorry
}

end NUMINAMATH_GPT_g_g_x_has_exactly_4_distinct_real_roots_l1595_159563


namespace NUMINAMATH_GPT_projective_transformation_is_cross_ratio_preserving_l1595_159524

theorem projective_transformation_is_cross_ratio_preserving (P : ℝ → ℝ) :
  (∃ a b c d : ℝ, (ad - bc ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))) ↔
  (∀ x1 x2 x3 x4 : ℝ, (x1 - x3) * (x2 - x4) / ((x1 - x4) * (x2 - x3)) =
       (P x1 - P x3) * (P x2 - P x4) / ((P x1 - P x4) * (P x2 - P x3))) :=
sorry

end NUMINAMATH_GPT_projective_transformation_is_cross_ratio_preserving_l1595_159524


namespace NUMINAMATH_GPT_final_pressure_of_helium_l1595_159526

theorem final_pressure_of_helium
  (p v v' : ℝ) (k : ℝ)
  (h1 : p = 4)
  (h2 : v = 3)
  (h3 : v' = 6)
  (h4 : p * v = k)
  (h5 : ∀ p' : ℝ, p' * v' = k → p' = 2) :
  p' = 2 := by
  sorry

end NUMINAMATH_GPT_final_pressure_of_helium_l1595_159526


namespace NUMINAMATH_GPT_percentage_increase_l1595_159580

variables (A B C D E : ℝ)
variables (A_inc B_inc C_inc D_inc E_inc : ℝ)

-- Conditions
def conditions (A_inc B_inc C_inc D_inc E_inc : ℝ) :=
  A_inc = 0.1 * A ∧
  B_inc = (1/15) * B ∧
  C_inc = 0.05 * C ∧
  D_inc = 0.04 * D ∧
  E_inc = (1/30) * E ∧
  B = 1.5 * A ∧
  C = 2 * A ∧
  D = 2.5 * A ∧
  E = 3 * A

-- Theorem to prove
theorem percentage_increase (A B C D E : ℝ) (A_inc B_inc C_inc D_inc E_inc : ℝ) :
  conditions A B C D E A_inc B_inc C_inc D_inc E_inc →
  (A_inc + B_inc + C_inc + D_inc + E_inc) / (A + B + C + D + E) = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1595_159580


namespace NUMINAMATH_GPT_exists_consecutive_divisible_by_cube_l1595_159570

theorem exists_consecutive_divisible_by_cube (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ j : ℕ, j < k → ∃ m : ℕ, 1 < m ∧ (n + j) % (m^3) = 0 := 
sorry

end NUMINAMATH_GPT_exists_consecutive_divisible_by_cube_l1595_159570


namespace NUMINAMATH_GPT_exclude_chairs_l1595_159544

-- Definitions
def total_chairs : ℕ := 10000
def perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Statement
theorem exclude_chairs (n : ℕ) (h₁ : n = total_chairs) :
  perfect_square n → (n - total_chairs) = 0 := 
sorry

end NUMINAMATH_GPT_exclude_chairs_l1595_159544


namespace NUMINAMATH_GPT_lyka_saving_per_week_l1595_159587

-- Definitions from the conditions
def smartphone_price : ℕ := 160
def lyka_has : ℕ := 40
def weeks_in_two_months : ℕ := 8

-- The goal (question == correct answer)
theorem lyka_saving_per_week :
  (smartphone_price - lyka_has) / weeks_in_two_months = 15 :=
sorry

end NUMINAMATH_GPT_lyka_saving_per_week_l1595_159587


namespace NUMINAMATH_GPT_log_base_change_l1595_159575

theorem log_base_change (a b : ℝ) (h₁ : Real.log 5 / Real.log 3 = a) (h₂ : Real.log 7 / Real.log 3 = b) :
    Real.log 35 / Real.log 15 = (a + b) / (1 + a) :=
by
  sorry

end NUMINAMATH_GPT_log_base_change_l1595_159575


namespace NUMINAMATH_GPT_inequality_proof_l1595_159598

variable {x₁ x₂ x₃ x₄ : ℝ}

theorem inequality_proof
  (h₁ : x₁ ≥ x₂) (h₂ : x₂ ≥ x₃) (h₃ : x₃ ≥ x₄) (h₄ : x₄ ≥ 2)
  (h₅ : x₂ + x₃ + x₄ ≥ x₁) 
  : (x₁ + x₂ + x₃ + x₄)^2 ≤ 4 * x₁ * x₂ * x₃ * x₄ := 
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l1595_159598


namespace NUMINAMATH_GPT_masha_guessed_number_l1595_159565

theorem masha_guessed_number (a b : ℕ) (h1 : a + b = 2002 ∨ a * b = 2002)
  (h2 : ∀ x y, x + y = 2002 → x ≠ 1001 → y ≠ 1001)
  (h3 : ∀ x y, x * y = 2002 → x ≠ 1001 → y ≠ 1001) :
  b = 1001 :=
by {
  sorry
}

end NUMINAMATH_GPT_masha_guessed_number_l1595_159565


namespace NUMINAMATH_GPT_ratio_of_numbers_l1595_159553

theorem ratio_of_numbers (A B : ℕ) (h_lcm : Nat.lcm A B = 48) (h_hcf : Nat.gcd A B = 4) : A / 4 = 3 ∧ B / 4 = 4 :=
sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1595_159553


namespace NUMINAMATH_GPT_infinite_squares_and_circles_difference_l1595_159531

theorem infinite_squares_and_circles_difference 
  (side_length : ℝ)
  (h₁ : side_length = 1)
  (square_area_sum : ℝ)
  (circle_area_sum : ℝ)
  (h_square_area : square_area_sum = (∑' n : ℕ, (side_length / 2^n)^2))
  (h_circle_area : circle_area_sum = (∑' n : ℕ, π * (side_length / 2^(n+1))^2 ))
  : square_area_sum - circle_area_sum = 2 - (π / 2) :=
by 
  sorry 

end NUMINAMATH_GPT_infinite_squares_and_circles_difference_l1595_159531


namespace NUMINAMATH_GPT_intersection_eq_l1595_159528

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_eq : A ∩ B = {x : ℝ | (-2 ≤ x ∧ x ≤ -1) ∨ (1 ≤ x ∧ x ≤ 2)} :=
by sorry

end NUMINAMATH_GPT_intersection_eq_l1595_159528


namespace NUMINAMATH_GPT_range_of_a_decreasing_l1595_159512

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a / x

def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≥ f y

theorem range_of_a_decreasing (a : ℝ) :
  (∃ a : ℝ, (1/6) ≤ a ∧ a < (1/3)) ↔ is_decreasing (f a) :=
sorry

end NUMINAMATH_GPT_range_of_a_decreasing_l1595_159512


namespace NUMINAMATH_GPT_books_problem_l1595_159514

theorem books_problem
  (M H : ℕ)
  (h1 : M + H = 80)
  (h2 : 4 * M + 5 * H = 390) :
  M = 10 :=
by
  sorry

end NUMINAMATH_GPT_books_problem_l1595_159514


namespace NUMINAMATH_GPT_quadratic_inequality_l1595_159573

theorem quadratic_inequality (m y1 y2 y3 : ℝ)
  (h1 : m < -2)
  (h2 : y1 = (m-1)^2 - 2*(m-1))
  (h3 : y2 = m^2 - 2*m)
  (h4 : y3 = (m+1)^2 - 2*(m+1)) :
  y3 < y2 ∧ y2 < y1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1595_159573


namespace NUMINAMATH_GPT_train_crossing_time_l1595_159522

def speed_kmph : ℝ := 90
def length_train : ℝ := 225

noncomputable def speed_mps : ℝ := speed_kmph * (1000 / 3600)

theorem train_crossing_time : (length_train / speed_mps) = 9 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1595_159522


namespace NUMINAMATH_GPT_people_later_than_yoongi_l1595_159540

variable (total_students : ℕ) (people_before_yoongi : ℕ)

theorem people_later_than_yoongi
    (h1 : total_students = 20)
    (h2 : people_before_yoongi = 11) :
    total_students - (people_before_yoongi + 1) = 8 := 
sorry

end NUMINAMATH_GPT_people_later_than_yoongi_l1595_159540


namespace NUMINAMATH_GPT_find_x_eq_nine_fourths_l1595_159561

theorem find_x_eq_nine_fourths (x n : ℚ) (β : ℚ) (h1 : x = n + β) (h2 : n = ⌊x⌋) (h3 : ⌊x⌋ + x = 17 / 4) : x = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_eq_nine_fourths_l1595_159561


namespace NUMINAMATH_GPT_solve_for_x_l1595_159578

theorem solve_for_x (x : ℝ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1595_159578


namespace NUMINAMATH_GPT_eating_time_correct_l1595_159539

-- Define the rates at which each individual eats cereal
def rate_fat : ℚ := 1 / 20
def rate_thin : ℚ := 1 / 30
def rate_medium : ℚ := 1 / 15

-- Define the combined rate of eating cereal together
def combined_rate : ℚ := rate_fat + rate_thin + rate_medium

-- Define the total pounds of cereal
def total_cereal : ℚ := 5

-- Define the time taken by everyone to eat the cereal
def time_taken : ℚ := total_cereal / combined_rate

-- Proof statement
theorem eating_time_correct :
  time_taken = 100 / 3 :=
by sorry

end NUMINAMATH_GPT_eating_time_correct_l1595_159539


namespace NUMINAMATH_GPT_find_number_l1595_159525

theorem find_number:
  ∃ x : ℝ, (3/4 * x + 9 = 1/5 * (x - 8 * x^(1/3))) ∧ x = -27 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1595_159525


namespace NUMINAMATH_GPT_bob_correct_answer_l1595_159599

theorem bob_correct_answer (y : ℕ) (h : (y - 7) / 5 = 47) : (y - 5) / 7 = 33 :=
by 
  -- assumption h and the statement to prove
  sorry

end NUMINAMATH_GPT_bob_correct_answer_l1595_159599


namespace NUMINAMATH_GPT_lines_parallel_l1595_159559

-- Define line l1 and line l2
def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

-- Prove that l1 is parallel to l2
theorem lines_parallel : ∀ x : ℝ, (l1 x - l2 x) = -4 := by
  sorry

end NUMINAMATH_GPT_lines_parallel_l1595_159559


namespace NUMINAMATH_GPT_water_overflow_amount_l1595_159548

-- Declare the conditions given in the problem
def tap_production_per_hour : ℕ := 200
def tap_run_duration_in_hours : ℕ := 24
def tank_capacity_in_ml : ℕ := 4000

-- Define the total water produced by the tap
def total_water_produced : ℕ := tap_production_per_hour * tap_run_duration_in_hours

-- Define the amount of water that overflows
def water_overflowed : ℕ := total_water_produced - tank_capacity_in_ml

-- State the theorem to prove the amount of overflowing water
theorem water_overflow_amount : water_overflowed = 800 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_water_overflow_amount_l1595_159548


namespace NUMINAMATH_GPT_exercise_felt_weight_l1595_159550

variable (n w : ℕ)
variable (p : ℝ)

def total_weight (n : ℕ) (w : ℕ) : ℕ := n * w

def felt_weight (total_weight : ℕ) (p : ℝ) : ℝ := total_weight * (1 + p)

theorem exercise_felt_weight (h1 : n = 10) (h2 : w = 30) (h3 : p = 0.20) : 
  felt_weight (total_weight n w) p = 360 :=
by 
  sorry

end NUMINAMATH_GPT_exercise_felt_weight_l1595_159550


namespace NUMINAMATH_GPT_engineers_to_designers_ratio_l1595_159567

-- Define the given conditions for the problem
variables (e d : ℕ) -- e is the number of engineers, d is the number of designers
variables (h1 : (48 * e + 60 * d) / (e + d) = 52)

-- Theorem statement: The ratio of the number of engineers to the number of designers is 2:1
theorem engineers_to_designers_ratio (h1 : (48 * e + 60 * d) / (e + d) = 52) : e = 2 * d :=
by {
  sorry  
}

end NUMINAMATH_GPT_engineers_to_designers_ratio_l1595_159567


namespace NUMINAMATH_GPT_sculptures_not_on_display_approx_400_l1595_159529

theorem sculptures_not_on_display_approx_400 (A : ℕ) (hA : A = 900) :
  (2 / 3 * A - 2 / 9 * A) = 400 := by
  sorry

end NUMINAMATH_GPT_sculptures_not_on_display_approx_400_l1595_159529


namespace NUMINAMATH_GPT_total_num_problems_eq_30_l1595_159545

-- Define the conditions
def test_points : ℕ := 100
def points_per_3_point_problem : ℕ := 3
def points_per_4_point_problem : ℕ := 4
def num_4_point_problems : ℕ := 10

-- Define the number of 3-point problems
def num_3_point_problems : ℕ :=
  (test_points - num_4_point_problems * points_per_4_point_problem) / points_per_3_point_problem

-- Prove the total number of problems is 30
theorem total_num_problems_eq_30 :
  num_3_point_problems + num_4_point_problems = 30 := 
sorry

end NUMINAMATH_GPT_total_num_problems_eq_30_l1595_159545


namespace NUMINAMATH_GPT_digit_start_l1595_159502

theorem digit_start (a n p q : ℕ) (hp : a * 10^p < 2^n) (hq : 2^n < (a + 1) * 10^p)
  (hr : a * 10^q < 5^n) (hs : 5^n < (a + 1) * 10^q) :
  a = 3 :=
by
  -- The proof goes here.
  sorry

end NUMINAMATH_GPT_digit_start_l1595_159502


namespace NUMINAMATH_GPT_system_solution_5_3_l1595_159532

variables (x y : ℤ)

theorem system_solution_5_3 :
  (x = 5) ∧ (y = 3) → (2 * x - 3 * y = 1) :=
by intros; sorry

end NUMINAMATH_GPT_system_solution_5_3_l1595_159532


namespace NUMINAMATH_GPT_max_profit_at_90_l1595_159558

-- Definitions for conditions
def fixed_cost : ℝ := 5
def price_per_unit : ℝ := 100

noncomputable def variable_cost (x : ℕ) : ℝ :=
  if h : x < 80 then
    0.5 * x^2 + 40 * x
  else
    101 * x + 8100 / x - 2180

-- Definition of the profit function
noncomputable def profit (x : ℕ) : ℝ :=
  if h : x < 80 then
    -0.5 * x^2 + 60 * x - fixed_cost
  else
    1680 - x - 8100 / x

-- Maximum profit occurs at x = 90
theorem max_profit_at_90 : ∀ x : ℕ, profit 90 ≥ profit x := 
by {
  sorry
}

end NUMINAMATH_GPT_max_profit_at_90_l1595_159558


namespace NUMINAMATH_GPT_line_y_intercept_l1595_159516

theorem line_y_intercept (t : ℝ) (h : ∃ (t : ℝ), ∀ (x y : ℝ), x - 2 * y + t = 0 → (x = 2 ∧ y = -1)) :
  ∃ y : ℝ, (0 - 2 * y + t = 0) ∧ y = -2 :=
by
  sorry

end NUMINAMATH_GPT_line_y_intercept_l1595_159516


namespace NUMINAMATH_GPT_bill_face_value_l1595_159542

theorem bill_face_value
  (TD : ℝ) (T : ℝ) (r : ℝ) (FV : ℝ)
  (h1 : TD = 210)
  (h2 : T = 0.75)
  (h3 : r = 0.16) :
  FV = 1960 :=
by 
  sorry

end NUMINAMATH_GPT_bill_face_value_l1595_159542


namespace NUMINAMATH_GPT_tyrone_gave_25_marbles_l1595_159591

/-- Given that Tyrone initially had 97 marbles and Eric had 11 marbles, and after
    giving some marbles to Eric, Tyrone ended with twice as many marbles as Eric,
    we need to find the number of marbles Tyrone gave to Eric. -/
theorem tyrone_gave_25_marbles (x : ℕ) (t0 e0 : ℕ)
  (hT0 : t0 = 97)
  (hE0 : e0 = 11)
  (hT_end : (t0 - x) = 2 * (e0 + x)) :
  x = 25 := 
  sorry

end NUMINAMATH_GPT_tyrone_gave_25_marbles_l1595_159591


namespace NUMINAMATH_GPT_find_B_l1595_159536

theorem find_B (A B C : ℝ) (h : ∀ (x : ℝ), x ≠ 7 ∧ x ≠ -1 → 
    2 / ((x-7)*(x+1)^2) = A / (x-7) + B / (x+1) + C / (x+1)^2) : 
  B = 1 / 16 :=
sorry

end NUMINAMATH_GPT_find_B_l1595_159536


namespace NUMINAMATH_GPT_smallest_non_factor_l1595_159586

-- Definitions of the conditions
def isFactorOf (m n : ℕ) : Prop := n % m = 0
def distinct (a b : ℕ) : Prop := a ≠ b

-- The main statement we need to prove.
theorem smallest_non_factor (a b : ℕ) (h_distinct : distinct a b)
  (h_a_factor : isFactorOf a 48) (h_b_factor : isFactorOf b 48)
  (h_not_factor : ¬ isFactorOf (a * b) 48) :
  a * b = 32 := 
sorry

end NUMINAMATH_GPT_smallest_non_factor_l1595_159586


namespace NUMINAMATH_GPT_area_of_ABCD_l1595_159537

theorem area_of_ABCD (area_AMOP area_CNOQ : ℝ) 
  (h1: area_AMOP = 8) (h2: area_CNOQ = 24.5) : 
  ∃ (area_ABCD : ℝ), area_ABCD = 60.5 :=
by
  sorry

end NUMINAMATH_GPT_area_of_ABCD_l1595_159537


namespace NUMINAMATH_GPT_part1_part2_l1595_159551

-- Part (1): Prove k = 3 given x = -1 is a solution
theorem part1 (k : ℝ) (h : k * (-1)^2 + 4 * (-1) + 1 = 0) : k = 3 := 
sorry

-- Part (2): Prove k ≤ 4 and k ≠ 0 for the quadratic equation to have two real roots
theorem part2 (k : ℝ) (h : 16 - 4 * k ≥ 0) : k ≤ 4 ∧ k ≠ 0 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1595_159551


namespace NUMINAMATH_GPT_regression_equation_pos_corr_l1595_159576

noncomputable def linear_regression (x y : ℝ) : ℝ := 0.4 * x + 2.5

theorem regression_equation_pos_corr (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (mean_x : ℝ := 2.5) (mean_y : ℝ := 3.5)
    (pos_corr : x * y > 0)
    (cond1 : mean_x = 2.5)
    (cond2 : mean_y = 3.5) :
    linear_regression mean_x mean_y = mean_y :=
by
  sorry

end NUMINAMATH_GPT_regression_equation_pos_corr_l1595_159576


namespace NUMINAMATH_GPT_painted_prisms_l1595_159538

theorem painted_prisms (n : ℕ) (h : n > 2) :
  2 * ((n - 2) * (n - 1) + (n - 2) * n + (n - 1) * n) = (n - 2) * (n - 1) * n ↔ n = 7 :=
by sorry

end NUMINAMATH_GPT_painted_prisms_l1595_159538


namespace NUMINAMATH_GPT_jonah_walked_8_miles_l1595_159535

def speed : ℝ := 4
def time : ℝ := 2
def distance (s t : ℝ) : ℝ := s * t

theorem jonah_walked_8_miles : distance speed time = 8 := sorry

end NUMINAMATH_GPT_jonah_walked_8_miles_l1595_159535


namespace NUMINAMATH_GPT_sum_geometric_sequence_l1595_159503

variable {α : Type*} [LinearOrderedField α]

theorem sum_geometric_sequence {S : ℕ → α} {n : ℕ} (h1 : S n = 3) (h2 : S (3 * n) = 21) :
    S (2 * n) = 9 := 
sorry

end NUMINAMATH_GPT_sum_geometric_sequence_l1595_159503


namespace NUMINAMATH_GPT_perpendicular_lines_foot_of_perpendicular_l1595_159590

theorem perpendicular_lines_foot_of_perpendicular 
  (m n p : ℝ) 
  (h1 : 2 * 2 + 3 * p - 1 = 0)
  (h2 : 3 * 2 - 2 * p + n = 0)
  (h3 : - (2 / m) * (3 / 2) = -1) 
  : p - m - n = 4 := 
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_foot_of_perpendicular_l1595_159590


namespace NUMINAMATH_GPT_fewest_trips_l1595_159572

theorem fewest_trips (total_objects : ℕ) (capacity : ℕ) (h_objects : total_objects = 17) (h_capacity : capacity = 3) : 
  (total_objects + capacity - 1) / capacity = 6 :=
by
  sorry

end NUMINAMATH_GPT_fewest_trips_l1595_159572


namespace NUMINAMATH_GPT_units_digit_6_pow_6_l1595_159557

theorem units_digit_6_pow_6 : (6 ^ 6) % 10 = 6 := 
by {
  sorry
}

end NUMINAMATH_GPT_units_digit_6_pow_6_l1595_159557


namespace NUMINAMATH_GPT_c_10_value_l1595_159560

def c : ℕ → ℤ
| 0 => 3
| 1 => 9
| (n + 1) => c n * c (n - 1)

theorem c_10_value : c 10 = 3^89 :=
by
  sorry

end NUMINAMATH_GPT_c_10_value_l1595_159560


namespace NUMINAMATH_GPT_modular_inverse_of_31_mod_35_is_1_l1595_159568

theorem modular_inverse_of_31_mod_35_is_1 :
  ∃ a : ℕ, 0 ≤ a ∧ a < 35 ∧ 31 * a % 35 = 1 := sorry

end NUMINAMATH_GPT_modular_inverse_of_31_mod_35_is_1_l1595_159568


namespace NUMINAMATH_GPT_no_integer_solution_for_Px_eq_x_l1595_159579

theorem no_integer_solution_for_Px_eq_x (P : ℤ → ℤ) (hP_int_coeff : ∀ n : ℤ, ∃ k : ℤ, P n = k * n + k) 
  (hP3 : P 3 = 4) (hP4 : P 4 = 3) :
  ¬ ∃ x : ℤ, P x = x := 
by 
  sorry

end NUMINAMATH_GPT_no_integer_solution_for_Px_eq_x_l1595_159579


namespace NUMINAMATH_GPT_cubes_sum_l1595_159515

theorem cubes_sum (a b c : ℝ) (h1 : a + b + c = 8) (h2 : a * b + a * c + b * c = 9) (h3 : a * b * c = -18) :
  a^3 + b^3 + c^3 = 242 :=
by
  sorry

end NUMINAMATH_GPT_cubes_sum_l1595_159515


namespace NUMINAMATH_GPT_distance_squared_l1595_159507

noncomputable def circumcircle_radius (R : ℝ) : Prop := sorry
noncomputable def excircle_radius (p : ℝ) : Prop := sorry
noncomputable def distance_between_centers (d : ℝ) (R : ℝ) (p : ℝ) : Prop := sorry

theorem distance_squared (R p d : ℝ) (h1 : circumcircle_radius R) (h2 : excircle_radius p) (h3 : distance_between_centers d R p) :
  d^2 = R^2 + 2 * R * p := sorry

end NUMINAMATH_GPT_distance_squared_l1595_159507


namespace NUMINAMATH_GPT_find_x_l1595_159520

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end NUMINAMATH_GPT_find_x_l1595_159520


namespace NUMINAMATH_GPT_cargo_transport_possible_l1595_159527

theorem cargo_transport_possible 
  (total_cargo_weight : ℝ) 
  (weight_limit_per_box : ℝ) 
  (number_of_trucks : ℕ) 
  (max_load_per_truck : ℝ)
  (h1 : total_cargo_weight = 13.5)
  (h2 : weight_limit_per_box = 0.35)
  (h3 : number_of_trucks = 11)
  (h4 : max_load_per_truck = 1.5) :
  ∃ (n : ℕ), n ≤ number_of_trucks ∧ (total_cargo_weight / max_load_per_truck) ≤ n :=
by
  sorry

end NUMINAMATH_GPT_cargo_transport_possible_l1595_159527


namespace NUMINAMATH_GPT_maximum_value_of_a_l1595_159554

theorem maximum_value_of_a
  (a b c d : ℝ)
  (h1 : b + c + d = 3 - a)
  (h2 : 2 * b^2 + 3 * c^2 + 6 * d^2 = 5 - a^2) :
  a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_maximum_value_of_a_l1595_159554


namespace NUMINAMATH_GPT_checkerboards_that_cannot_be_covered_l1595_159581

-- Define the dimensions of the checkerboards
def checkerboard_4x6 := (4, 6)
def checkerboard_3x7 := (3, 7)
def checkerboard_5x5 := (5, 5)
def checkerboard_7x4 := (7, 4)
def checkerboard_5x6 := (5, 6)

-- Define a function to calculate the number of squares
def num_squares (dims : Nat × Nat) : Nat := dims.1 * dims.2

-- Define a function to check if a board can be exactly covered by dominoes
def can_be_covered_by_dominoes (dims : Nat × Nat) : Bool := (num_squares dims) % 2 == 0

-- Statement to be proven
theorem checkerboards_that_cannot_be_covered :
  ¬ can_be_covered_by_dominoes checkerboard_3x7 ∧ ¬ can_be_covered_by_dominoes checkerboard_5x5 :=
by
  sorry

end NUMINAMATH_GPT_checkerboards_that_cannot_be_covered_l1595_159581


namespace NUMINAMATH_GPT_converse_not_true_prop_B_l1595_159577

noncomputable def line_in_plane (b : Type) (α : Type) : Prop := sorry
noncomputable def perp_line_plane (b : Type) (β : Type) : Prop := sorry
noncomputable def perp_planes (α : Type) (β : Type) : Prop := sorry
noncomputable def parallel_planes (α : Type) (β : Type) : Prop := sorry

variables (a b c : Type) (α β : Type)

theorem converse_not_true_prop_B :
  (line_in_plane b α) → (perp_planes α β) → ¬ (perp_line_plane b β) :=
sorry

end NUMINAMATH_GPT_converse_not_true_prop_B_l1595_159577


namespace NUMINAMATH_GPT_product_of_solutions_t_squared_eq_49_l1595_159530

theorem product_of_solutions_t_squared_eq_49 (t : ℝ) (h1 : t^2 = 49) : (7 * -7) = -49 :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_t_squared_eq_49_l1595_159530


namespace NUMINAMATH_GPT_find_y_l1595_159583

theorem find_y
  (XYZ_is_straight_line : XYZ_is_straight_line)
  (angle_XYZ : ℝ)
  (angle_YWZ : ℝ)
  (y : ℝ)
  (exterior_angle_theorem : angle_XYZ = y + angle_YWZ)
  (h1 : angle_XYZ = 150)
  (h2 : angle_YWZ = 58) :
  y = 92 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1595_159583


namespace NUMINAMATH_GPT_polynomial_approx_eq_l1595_159521

theorem polynomial_approx_eq (x : ℝ) (h : x^4 - 4*x^3 + 4*x^2 + 4 = 4.999999999999999) : x = 1 :=
sorry

end NUMINAMATH_GPT_polynomial_approx_eq_l1595_159521


namespace NUMINAMATH_GPT_complement_eq_target_l1595_159518

namespace ComplementProof

-- Define the universal set U
def U : Set ℕ := {2, 4, 6, 8, 10}

-- Define the set A
def A : Set ℕ := {2, 6, 8}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := {x ∈ U | x ∉ A}

-- Define the target set
def target_set : Set ℕ := {4, 10}

-- Prove that the complement of A with respect to U is equal to {4, 10}
theorem complement_eq_target :
  complement_U_A = target_set := by sorry

end ComplementProof

end NUMINAMATH_GPT_complement_eq_target_l1595_159518


namespace NUMINAMATH_GPT_tan_of_tan_squared_2025_deg_l1595_159585

noncomputable def tan_squared (x : ℝ) : ℝ := (Real.tan x) ^ 2

theorem tan_of_tan_squared_2025_deg : 
  Real.tan (tan_squared (2025 * Real.pi / 180)) = Real.tan (Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_tan_of_tan_squared_2025_deg_l1595_159585


namespace NUMINAMATH_GPT_football_throwing_distance_l1595_159555

theorem football_throwing_distance 
  (T : ℝ)
  (yards_per_throw_at_T : ℝ)
  (yards_per_throw_at_80 : ℝ)
  (throws_on_Saturday : ℕ)
  (throws_on_Sunday : ℕ)
  (saturday_distance sunday_distance : ℝ)
  (total_distance : ℝ) :
  yards_per_throw_at_T = 20 →
  yards_per_throw_at_80 = 40 →
  throws_on_Saturday = 20 →
  throws_on_Sunday = 30 →
  saturday_distance = throws_on_Saturday * yards_per_throw_at_T →
  sunday_distance = throws_on_Sunday * yards_per_throw_at_80 →
  total_distance = saturday_distance + sunday_distance →
  total_distance = 1600 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_football_throwing_distance_l1595_159555


namespace NUMINAMATH_GPT_not_divisible_by_5_l1595_159541

theorem not_divisible_by_5 (n : ℤ) : ¬ (n^2 - 8) % 5 = 0 :=
by sorry

end NUMINAMATH_GPT_not_divisible_by_5_l1595_159541


namespace NUMINAMATH_GPT_roots_equation_l1595_159508
-- We bring in the necessary Lean libraries

-- Define the conditions as Lean definitions
variable (x1 x2 : ℝ)
variable (h1 : x1^2 + x1 - 3 = 0)
variable (h2 : x2^2 + x2 - 3 = 0)

-- Lean 4 statement we need to prove
theorem roots_equation (x1 x2 : ℝ) (h1 : x1^2 + x1 - 3 = 0) (h2 : x2^2 + x2 - 3 = 0) : 
  x1^3 - 4 * x2^2 + 19 = 0 := 
sorry

end NUMINAMATH_GPT_roots_equation_l1595_159508


namespace NUMINAMATH_GPT_tina_total_leftover_l1595_159519

def monthly_income : ℝ := 1000

def june_savings : ℝ := 0.25 * monthly_income
def june_expenses : ℝ := 200 + 0.05 * monthly_income
def june_leftover : ℝ := monthly_income - june_savings - june_expenses

def july_savings : ℝ := 0.20 * monthly_income
def july_expenses : ℝ := 250 + 0.15 * monthly_income
def july_leftover : ℝ := monthly_income - july_savings - july_expenses

def august_savings : ℝ := 0.30 * monthly_income
def august_expenses : ℝ := 250 + 50 + 0.10 * monthly_income
def august_gift : ℝ := 50
def august_leftover : ℝ := (monthly_income - august_savings - august_expenses) + august_gift

def total_leftover : ℝ :=
  june_leftover + july_leftover + august_leftover

theorem tina_total_leftover (I : ℝ) (hI : I = 1000) :
  total_leftover = 1250 := by
  rw [←hI] at *
  show total_leftover = 1250
  sorry

end NUMINAMATH_GPT_tina_total_leftover_l1595_159519


namespace NUMINAMATH_GPT_quadratic_vertex_transform_l1595_159552

theorem quadratic_vertex_transform {p q r m k : ℝ} (h : ℝ) :
  (∀ x : ℝ, p * x^2 + q * x + r = 5 * (x + 3)^2 - 15) →
  (∀ x : ℝ, 4 * p * x^2 + 4 * q * x + 4 * r = m * (x - h)^2 + k) →
  h = -3 :=
by
  intros h1 h2
  -- The actual proof goes here
  sorry

end NUMINAMATH_GPT_quadratic_vertex_transform_l1595_159552


namespace NUMINAMATH_GPT_degree_of_divisor_l1595_159533

theorem degree_of_divisor (f q r d : Polynomial ℝ)
  (h_f : f.degree = 15)
  (h_q : q.degree = 9)
  (h_r : r = Polynomial.C 5 * X^4 + Polynomial.C 3 * X^3 - Polynomial.C 2 * X^2 + Polynomial.C 9 * X - Polynomial.C 7)
  (h_div : f = d * q + r) :
  d.degree = 6 :=
by sorry

end NUMINAMATH_GPT_degree_of_divisor_l1595_159533


namespace NUMINAMATH_GPT_larger_number_of_hcf_23_lcm_factors_13_15_l1595_159588

theorem larger_number_of_hcf_23_lcm_factors_13_15 :
  ∃ A B, (Nat.gcd A B = 23) ∧ (A * B = 23 * 13 * 15) ∧ (A = 345 ∨ B = 345) := sorry

end NUMINAMATH_GPT_larger_number_of_hcf_23_lcm_factors_13_15_l1595_159588


namespace NUMINAMATH_GPT_range_of_a_l1595_159511

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, (2 * a < x ∧ x < a + 5) → (x < 6)) ↔ (1 < a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1595_159511


namespace NUMINAMATH_GPT_two_numbers_ratio_l1595_159546

theorem two_numbers_ratio (A B : ℕ) (h_lcm : Nat.lcm A B = 30) (h_sum : A + B = 25) :
  ∃ x y : ℕ, x = 2 ∧ y = 3 ∧ A / B = x / y := 
sorry

end NUMINAMATH_GPT_two_numbers_ratio_l1595_159546


namespace NUMINAMATH_GPT_total_travel_time_l1595_159562

-- Define the given conditions
def speed_jogging : ℝ := 5
def speed_bus : ℝ := 30
def distance_to_school : ℝ := 6.857142857142858

-- State the theorem to prove
theorem total_travel_time :
  (distance_to_school / speed_jogging) + (distance_to_school / speed_bus) = 1.6 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_time_l1595_159562


namespace NUMINAMATH_GPT_expense_recording_l1595_159569

-- Define the recording of income and expenses
def record_income (amount : Int) : Int := amount
def record_expense (amount : Int) : Int := -amount

-- Given conditions
def income_example := record_income 500
def expense_example := record_expense 400

-- Prove that an expense of 400 yuan is recorded as -400 yuan
theorem expense_recording : record_expense 400 = -400 :=
  by sorry

end NUMINAMATH_GPT_expense_recording_l1595_159569


namespace NUMINAMATH_GPT_part_a_proof_part_b_proof_l1595_159589

-- Part (a) statement
def part_a_statement (n : ℕ) : Prop :=
  ∀ (m : ℕ), m = 9 → (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 10 ∨ n = 12 ∨ n = 18)

theorem part_a_proof (n : ℕ) (m : ℕ) (h : m = 9) : part_a_statement n :=
  sorry

-- Part (b) statement
def part_b_statement (n m : ℕ) : Prop :=
  (n ≤ m) ∨ (n > m ∧ ∃ d : ℕ, d ∣ m ∧ n = m + d)

theorem part_b_proof (n m : ℕ) : part_b_statement n m :=
  sorry

end NUMINAMATH_GPT_part_a_proof_part_b_proof_l1595_159589


namespace NUMINAMATH_GPT_solve_for_a_and_b_l1595_159592

noncomputable def A := {x : ℝ | (-2 < x ∧ x < -1) ∨ (x > 1)}
noncomputable def B (a b : ℝ) := {x : ℝ | a ≤ x ∧ x < b}

theorem solve_for_a_and_b (a b : ℝ) :
  (A ∪ B a b = {x : ℝ | x > -2}) ∧ (A ∩ B a b = {x : ℝ | 1 < x ∧ x < 3}) →
  a = -1 ∧ b = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_and_b_l1595_159592


namespace NUMINAMATH_GPT_M_gt_N_l1595_159593

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2 * (x + y - 1)

theorem M_gt_N : M x y > N x y := sorry

end NUMINAMATH_GPT_M_gt_N_l1595_159593


namespace NUMINAMATH_GPT_F5_div_641_Fermat_rel_prime_l1595_159564

def Fermat_number (n : ℕ) : ℕ := 2^(2^n) + 1

theorem F5_div_641 : Fermat_number 5 % 641 = 0 := 
  sorry

theorem Fermat_rel_prime (k n : ℕ) (hk: k ≠ n) : Nat.gcd (Fermat_number k) (Fermat_number n) = 1 :=
  sorry

end NUMINAMATH_GPT_F5_div_641_Fermat_rel_prime_l1595_159564


namespace NUMINAMATH_GPT_g_ln_1_div_2017_l1595_159571

open Real

-- Define the functions fulfilling the given conditions
variables (f g : ℝ → ℝ) (a : ℝ)

-- Define assumptions as required by the conditions
axiom f_property : ∀ m n : ℝ, f (m + n) = f m + f n - 1
axiom g_def : ∀ x : ℝ, g x = f x + a^x / (a^x + 1)
axiom a_property : a > 0 ∧ a ≠ 1
axiom g_ln_2017 : g (log 2017) = 2018

-- The theorem to prove
theorem g_ln_1_div_2017 : g (log (1 / 2017)) = -2015 := by
  sorry

end NUMINAMATH_GPT_g_ln_1_div_2017_l1595_159571


namespace NUMINAMATH_GPT_problem_l1595_159556

variable {a b c : ℝ} -- Introducing variables a, b, c as real numbers

-- Conditions:
-- a, b, c are distinct positive real numbers
def distinct_pos (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a 

theorem problem (h : distinct_pos a b c) : 
  (b + c - a) / a + (a + c - b) / b + (a + b - c) / c > 3 :=
sorry 

end NUMINAMATH_GPT_problem_l1595_159556


namespace NUMINAMATH_GPT_bella_items_l1595_159501

theorem bella_items (M F D : ℕ) 
  (h1 : M = 60)
  (h2 : M = 2 * F)
  (h3 : F = D + 20) :
  (7 * M + 7 * F + 7 * D) / 5 = 140 := 
by
  sorry

end NUMINAMATH_GPT_bella_items_l1595_159501


namespace NUMINAMATH_GPT_edge_length_of_cube_l1595_159566

theorem edge_length_of_cube {V_cube V_cuboid : ℝ} (base_area : ℝ) (height : ℝ)
  (h1 : base_area = 10) (h2 : height = 73) (h3 : V_cube = V_cuboid - 1)
  (h4 : V_cuboid = base_area * height) :
  ∃ (a : ℝ), a^3 = V_cube ∧ a = 9 :=
by
  /- The proof is omitted -/
  sorry

end NUMINAMATH_GPT_edge_length_of_cube_l1595_159566


namespace NUMINAMATH_GPT_find_d_from_factor_condition_l1595_159506

theorem find_d_from_factor_condition (d : ℚ) : (∀ x, x = 5 → d * x^4 + 13 * x^3 - 2 * d * x^2 - 58 * x + 65 = 0) → d = -28 / 23 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_d_from_factor_condition_l1595_159506


namespace NUMINAMATH_GPT_sums_of_integers_have_same_remainder_l1595_159517

theorem sums_of_integers_have_same_remainder (n : ℕ) (n_pos : 0 < n) : 
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ 2 * n) ∧ (1 ≤ j ∧ j ≤ 2 * n) ∧ i ≠ j ∧ ((i + i) % (2 * n) = (j + j) % (2 * n)) :=
by
  sorry

end NUMINAMATH_GPT_sums_of_integers_have_same_remainder_l1595_159517


namespace NUMINAMATH_GPT_quadratic_inequality_false_iff_l1595_159594

theorem quadratic_inequality_false_iff (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x^2 - 3 * a * x + 9 < 0) ↔ (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_false_iff_l1595_159594


namespace NUMINAMATH_GPT_lines_in_plane_l1595_159523

  -- Define the necessary objects in Lean
  structure Line (α : Type) := (equation : α → α → Prop)

  def same_plane (l1 l2 : Line ℝ) : Prop := 
  -- Here you can define what it means for l1 and l2 to be in the same plane.
  sorry

  def intersect (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to intersect.
  sorry

  def parallel (l1 l2 : Line ℝ) : Prop := 
  -- Define what it means for two lines to be parallel.
  sorry

  theorem lines_in_plane (l1 l2 : Line ℝ) (h : same_plane l1 l2) : 
    (intersect l1 l2) ∨ (parallel l1 l2) := 
  by 
      sorry
  
end NUMINAMATH_GPT_lines_in_plane_l1595_159523


namespace NUMINAMATH_GPT_overall_average_score_l1595_159509

theorem overall_average_score (first_6_avg last_4_avg : ℝ) (n_first n_last n_total : ℕ) 
    (h_matches : n_first + n_last = n_total)
    (h_first_avg : first_6_avg = 41)
    (h_last_avg : last_4_avg = 35.75)
    (h_n_first : n_first = 6)
    (h_n_last : n_last = 4)
    (h_n_total : n_total = 10) :
    ((first_6_avg * n_first + last_4_avg * n_last) / n_total) = 38.9 := by
  sorry

end NUMINAMATH_GPT_overall_average_score_l1595_159509


namespace NUMINAMATH_GPT_distance_home_to_school_l1595_159547

-- Define the variables and conditions
variables (D T : ℝ)
def boy_travel_5km_hr_late := 5 * (T + 5 / 60) = D
def boy_travel_10km_hr_early := 10 * (T - 10 / 60) = D

-- State the theorem to prove
theorem distance_home_to_school 
    (H1 : boy_travel_5km_hr_late D T) 
    (H2 : boy_travel_10km_hr_early D T) : 
  D = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_distance_home_to_school_l1595_159547


namespace NUMINAMATH_GPT_solve_for_x_l1595_159500

theorem solve_for_x : ∀ (x : ℕ), (y = 2 / (4 * x + 2)) → (y = 1 / 2) → (x = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1595_159500


namespace NUMINAMATH_GPT_smallest_n_for_roots_of_unity_l1595_159597

theorem smallest_n_for_roots_of_unity (z : ℂ) (h : z^6 - z^3 + 1 = 0) : ∃ (n : ℕ), n = 18 ∧ z^n = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_n_for_roots_of_unity_l1595_159597


namespace NUMINAMATH_GPT_sqrt_expr_equals_sum_l1595_159574

theorem sqrt_expr_equals_sum :
  ∃ x y z : ℤ,
    (x + y * Int.sqrt z = Real.sqrt (77 + 28 * Real.sqrt 3)) ∧
    (x^2 + y^2 * z = 77) ∧
    (2 * x * y = 28) ∧
    (x + y + z = 16) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expr_equals_sum_l1595_159574


namespace NUMINAMATH_GPT_emily_workers_needed_l1595_159543

noncomputable def least_workers_needed
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ) :
  ℕ :=
  (remaining_work / remaining_days) / (work_done / initial_days / total_workers) * total_workers

theorem emily_workers_needed 
  (total_days : ℕ) (initial_days : ℕ) (total_workers : ℕ) (work_done : ℕ) (remaining_work : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 40)
  (h2 : initial_days = 10)
  (h3 : total_workers = 12)
  (h4 : work_done = 40)
  (h5 : remaining_work = 60)
  (h6 : remaining_days = 30) :
  least_workers_needed total_days initial_days total_workers work_done remaining_work remaining_days = 6 := 
sorry

end NUMINAMATH_GPT_emily_workers_needed_l1595_159543
