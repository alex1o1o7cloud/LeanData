import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_of_inverse_ln_l1938_193814

/-- The inverse function of natural logarithm -/
noncomputable def f (x : ℝ) := Real.exp x

/-- The equation of the tangent line for f at x = 0 -/
def tangent_line (x y : ℝ) : Prop := x - y + 1 = 0

theorem tangent_line_of_inverse_ln :
  ∀ x y : ℝ, f x = y → x = 0 → tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_of_inverse_ln_l1938_193814


namespace NUMINAMATH_CALUDE_fourth_student_is_18_l1938_193893

/-- Represents a systematic sampling of students -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  first_student : ℕ
  h_total_positive : 0 < total_students
  h_sample_positive : 0 < sample_size
  h_sample_size : sample_size ≤ total_students
  h_first_valid : first_student ≤ total_students

/-- The sampling interval for a systematic sample -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.total_students / s.sample_size

/-- The nth student in the sample -/
def nth_student (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_student + (n - 1) * sampling_interval s

/-- Theorem: In a systematic sample of 4 from 52, if 5, 31, and 44 are sampled, then 18 is the fourth -/
theorem fourth_student_is_18 (s : SystematicSample) 
    (h_total : s.total_students = 52)
    (h_sample : s.sample_size = 4)
    (h_first : s.first_student = 5)
    (h_third : nth_student s 3 = 31)
    (h_fourth : nth_student s 4 = 44) :
    nth_student s 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_fourth_student_is_18_l1938_193893


namespace NUMINAMATH_CALUDE_log_problem_l1938_193885

theorem log_problem (x : ℝ) (h : Real.log x / Real.log 7 - Real.log 3 / Real.log 7 = 2) :
  Real.log x / Real.log 13 = Real.log 52 / Real.log 13 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l1938_193885


namespace NUMINAMATH_CALUDE_triangle_area_l1938_193846

/-- Given a triangle with perimeter 28 cm and inradius 2.5 cm, its area is 35 cm² -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ) : 
  perimeter = 28 → inradius = 2.5 → area = inradius * (perimeter / 2) → area = 35 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1938_193846


namespace NUMINAMATH_CALUDE_intersection_equals_M_l1938_193866

def M : Set ℝ := {y | ∃ x, y = 3^x}
def N : Set ℝ := {y | ∃ x, y = x^2 - 1}

theorem intersection_equals_M : M ∩ N = M := by sorry

end NUMINAMATH_CALUDE_intersection_equals_M_l1938_193866


namespace NUMINAMATH_CALUDE_sum_divisible_by_31_l1938_193876

def geometric_sum (n : ℕ) : ℕ := (2^(5*n) - 1) / (2 - 1)

theorem sum_divisible_by_31 (n : ℕ+) : 
  31 ∣ geometric_sum n.val := by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_31_l1938_193876


namespace NUMINAMATH_CALUDE_inverse_307_mod_455_l1938_193854

theorem inverse_307_mod_455 : ∃ x : ℕ, x < 455 ∧ (307 * x) % 455 = 1 :=
by
  use 81
  sorry

end NUMINAMATH_CALUDE_inverse_307_mod_455_l1938_193854


namespace NUMINAMATH_CALUDE_valid_distribution_example_l1938_193841

def is_valid_distribution (probs : List ℚ) : Prop :=
  (probs.sum = 1) ∧ (∀ p ∈ probs, 0 < p ∧ p ≤ 1)

theorem valid_distribution_example : 
  is_valid_distribution [1/2, 1/3, 1/6] := by
  sorry

end NUMINAMATH_CALUDE_valid_distribution_example_l1938_193841


namespace NUMINAMATH_CALUDE_intersection_points_form_line_l1938_193824

theorem intersection_points_form_line (s : ℝ) :
  ∃ (x y : ℝ),
    (2 * x - 3 * y = 6 * s - 5) ∧
    (3 * x + y = 9 * s + 4) ∧
    (y = 3 * x + 16 / 11) :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_form_line_l1938_193824


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l1938_193840

theorem sqrt_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 21) :
  Real.sqrt a + Real.sqrt b < 2 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l1938_193840


namespace NUMINAMATH_CALUDE_power_of_three_mod_seven_l1938_193807

theorem power_of_three_mod_seven : 3^2023 % 7 = 3 := by sorry

end NUMINAMATH_CALUDE_power_of_three_mod_seven_l1938_193807


namespace NUMINAMATH_CALUDE_prop_3_prop_4_l1938_193887

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (skew : Line → Line → Prop)

-- Define the lines and planes
variable (m n l : Line)
variable (α : Plane)

-- State the theorems
theorem prop_3 (h1 : parallel m n) (h2 : perpendicular_plane m α) :
  perpendicular_plane n α := by sorry

theorem prop_4 (h1 : skew m n) (h2 : parallel_plane m α) (h3 : parallel_plane n α)
  (h4 : perpendicular m l) (h5 : perpendicular n l) :
  perpendicular_plane l α := by sorry

end NUMINAMATH_CALUDE_prop_3_prop_4_l1938_193887


namespace NUMINAMATH_CALUDE_problem_solution_l1938_193888

-- Define the function f(x)
def f (x : ℝ) : ℝ := |2*x - 1| + |2*x + 2|

-- Theorem statement
theorem problem_solution :
  (∃ (M : ℝ), (∀ x, f x ≥ M) ∧ (∃ x, f x = M) ∧ M = 3) ∧
  ({x : ℝ | f x < 3 + |2*x + 2|} = Set.Ioo (-1) 2) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 2*b^2 = 3 → 2*a + b ≤ 3*Real.sqrt 6 / 2) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a^2 + 2*b^2 = 3 ∧ 2*a + b = 3*Real.sqrt 6 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_problem_solution_l1938_193888


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1938_193880

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -6 * x^2 + 2 * x - 4 < 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_negative_l1938_193880


namespace NUMINAMATH_CALUDE_smallest_measurement_count_l1938_193874

theorem smallest_measurement_count : ∃ N : ℕ+, 
  (∀ m : ℕ+, m < N → 
    (¬(20 * m.val % 100 = 0) ∨ 
     ¬(375 * m.val % 1000 = 0) ∨ 
     ¬(25 * m.val % 100 = 0) ∨ 
     ¬(125 * m.val % 1000 = 0) ∨ 
     ¬(5 * m.val % 100 = 0))) ∧
  (20 * N.val % 100 = 0) ∧ 
  (375 * N.val % 1000 = 0) ∧ 
  (25 * N.val % 100 = 0) ∧ 
  (125 * N.val % 1000 = 0) ∧ 
  (5 * N.val % 100 = 0) ∧
  N.val = 40 := by
sorry

end NUMINAMATH_CALUDE_smallest_measurement_count_l1938_193874


namespace NUMINAMATH_CALUDE_best_play_win_probability_best_play_win_probability_m_plays_prove_best_play_win_probability_l1938_193870

/-- The probability that the best play wins in a two-play contest -/
theorem best_play_win_probability (n : ℕ) : ℝ :=
  let total_mothers : ℕ := 2 * n
  let confident_mothers : ℕ := n
  let unconfident_mothers : ℕ := n
  let prob_vote_best : ℝ := 1 / 2
  let prob_vote_child : ℝ := 1 / 2
  1 - (1 / 2) ^ n

/-- The probability that the best play wins in a contest with m plays -/
theorem best_play_win_probability_m_plays (n m : ℕ) : ℝ :=
  let total_mothers : ℕ := m * n
  let confident_mothers : ℕ := n
  let unconfident_mothers : ℕ := (m - 1) * n
  let prob_vote_best : ℝ := 1 / 2
  let prob_vote_child : ℝ := 1 / 2
  1 - (1 / 2) ^ ((m - 1) * n)

/-- Proof of the theorems -/
theorem prove_best_play_win_probability : 
  ∀ (n m : ℕ), m ≥ 2 → 
  best_play_win_probability n = 1 - (1 / 2) ^ n ∧
  best_play_win_probability_m_plays n m = 1 - (1 / 2) ^ ((m - 1) * n) := by
  sorry

end NUMINAMATH_CALUDE_best_play_win_probability_best_play_win_probability_m_plays_prove_best_play_win_probability_l1938_193870


namespace NUMINAMATH_CALUDE_concentric_circles_chord_count_l1938_193848

theorem concentric_circles_chord_count
  (angle_ABC : ℝ)
  (is_tangent : Bool)
  (h1 : angle_ABC = 60)
  (h2 : is_tangent = true) :
  ∃ n : ℕ, n * angle_ABC = 180 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_chord_count_l1938_193848


namespace NUMINAMATH_CALUDE_oliver_puzzle_cost_l1938_193823

/-- The amount of money Oliver spent on the puzzle -/
def puzzle_cost (initial_amount savings frisbee_cost birthday_gift final_amount : ℕ) : ℕ :=
  initial_amount + savings - frisbee_cost + birthday_gift - final_amount

theorem oliver_puzzle_cost : 
  puzzle_cost 9 5 4 8 15 = 3 := by sorry

end NUMINAMATH_CALUDE_oliver_puzzle_cost_l1938_193823


namespace NUMINAMATH_CALUDE_derivative_of_sin_over_x_l1938_193889

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / x

theorem derivative_of_sin_over_x :
  deriv f = fun x => (x * Real.cos x - Real.sin x) / (x^2) :=
sorry

end NUMINAMATH_CALUDE_derivative_of_sin_over_x_l1938_193889


namespace NUMINAMATH_CALUDE_laptop_down_payment_percentage_l1938_193812

theorem laptop_down_payment_percentage
  (laptop_cost : ℝ)
  (monthly_installment : ℝ)
  (additional_down_payment : ℝ)
  (balance_after_four_months : ℝ)
  (h1 : laptop_cost = 1000)
  (h2 : monthly_installment = 65)
  (h3 : additional_down_payment = 20)
  (h4 : balance_after_four_months = 520) :
  let down_payment_percentage := 100 * (laptop_cost - balance_after_four_months - 4 * monthly_installment - additional_down_payment) / laptop_cost
  down_payment_percentage = 20 := by
sorry

end NUMINAMATH_CALUDE_laptop_down_payment_percentage_l1938_193812


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_max_l1938_193836

theorem inscribed_rectangle_area_max (R : ℝ) (x y : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : x^2 + y^2 = 4*R^2) : 
  x * y ≤ 2 * R^2 ∧ (x * y = 2 * R^2 ↔ x = y ∧ x = R * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_max_l1938_193836


namespace NUMINAMATH_CALUDE_street_lights_configuration_l1938_193886

theorem street_lights_configuration (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 12 →
  k = 4 →
  m = n - k - 1 →
  Nat.choose m k = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_street_lights_configuration_l1938_193886


namespace NUMINAMATH_CALUDE_cube_fourth_root_inverse_prop_l1938_193828

-- Define the inverse proportionality between a^3 and b^(1/4)
def inverse_prop (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^(1/4) = k

-- Define the initial condition
def initial_condition (a b : ℝ) : Prop := a = 3 ∧ b = 16

-- Define the final condition
def final_condition (a b : ℝ) : Prop := a^2 * b = 54

theorem cube_fourth_root_inverse_prop 
  (a b : ℝ) 
  (h_inv_prop : inverse_prop a b) 
  (h_init : initial_condition a b) 
  (h_final : final_condition a b) : 
  b = 54^(2/5) := by
  sorry

end NUMINAMATH_CALUDE_cube_fourth_root_inverse_prop_l1938_193828


namespace NUMINAMATH_CALUDE_movie_watching_time_l1938_193858

/-- Represents the duration of a part of the movie watching session -/
structure MoviePart where
  watch_time : Nat
  rewind_time : Nat

/-- Calculates the total time for a movie watching session -/
def total_movie_time (parts : List MoviePart) : Nat :=
  parts.foldl (fun acc part => acc + part.watch_time + part.rewind_time) 0

/-- Theorem stating that the total time to watch the movie is 120 minutes -/
theorem movie_watching_time :
  let part1 : MoviePart := { watch_time := 35, rewind_time := 5 }
  let part2 : MoviePart := { watch_time := 45, rewind_time := 15 }
  let part3 : MoviePart := { watch_time := 20, rewind_time := 0 }
  total_movie_time [part1, part2, part3] = 120 := by
  sorry


end NUMINAMATH_CALUDE_movie_watching_time_l1938_193858


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l1938_193875

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (36 - 9) = Real.sqrt 170 - 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l1938_193875


namespace NUMINAMATH_CALUDE_g_of_eight_l1938_193818

/-- Given a function g : ℝ → ℝ satisfying the equation
    g(x) + g(3x+y) + 7xy = g(4x - y) + 3x^2 + 2 for all real x and y,
    prove that g(8) = -30. -/
theorem g_of_eight (g : ℝ → ℝ) 
  (h : ∀ x y : ℝ, g x + g (3*x + y) + 7*x*y = g (4*x - y) + 3*x^2 + 2) : 
  g 8 = -30 := by
  sorry

end NUMINAMATH_CALUDE_g_of_eight_l1938_193818


namespace NUMINAMATH_CALUDE_ratio_equality_l1938_193845

theorem ratio_equality (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hbc : b/c = 2005) (hcb : c/b = 2005) :
  (b + c) / (a + b) = 2005 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l1938_193845


namespace NUMINAMATH_CALUDE_average_students_count_l1938_193873

theorem average_students_count (total : ℕ) (top_yes : ℕ) (avg_yes : ℕ) (under_yes : ℕ) :
  total = 30 →
  top_yes = 19 →
  avg_yes = 12 →
  under_yes = 9 →
  ∃ (top avg under : ℕ),
    top + avg + under = total ∧
    top = top_yes ∧
    avg = avg_yes ∧
    under = under_yes :=
by
  sorry

end NUMINAMATH_CALUDE_average_students_count_l1938_193873


namespace NUMINAMATH_CALUDE_customers_added_l1938_193838

theorem customers_added (initial : ℕ) (no_tip : ℕ) (tip : ℕ) : 
  initial = 39 → no_tip = 49 → tip = 2 → 
  (no_tip + tip) - initial = 12 := by
  sorry

end NUMINAMATH_CALUDE_customers_added_l1938_193838


namespace NUMINAMATH_CALUDE_b_95_mod_49_l1938_193829

/-- Definition of the sequence b_n -/
def b (n : ℕ) : ℕ := 5^n + 7^n

/-- The remainder of b_95 when divided by 49 is 36 -/
theorem b_95_mod_49 : b 95 % 49 = 36 := by
  sorry

end NUMINAMATH_CALUDE_b_95_mod_49_l1938_193829


namespace NUMINAMATH_CALUDE_remainder_theorem_l1938_193825

/-- The remainder when x³ - 3x + 5 is divided by x + 2 is 3 -/
theorem remainder_theorem (x : ℝ) : 
  (x^3 - 3*x + 5) % (x + 2) = 3 := by
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1938_193825


namespace NUMINAMATH_CALUDE_fib_F15_units_digit_l1938_193896

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The period of the units digit in the Fibonacci sequence -/
def fib_units_period : ℕ := 60

/-- Theorem: The units digit of F_{F_15} is 5 -/
theorem fib_F15_units_digit : fib (fib 15) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fib_F15_units_digit_l1938_193896


namespace NUMINAMATH_CALUDE_smallest_prime_12_less_than_perfect_square_l1938_193817

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_prime_12_less_than_perfect_square :
  ∃ (n : ℕ), is_perfect_square n ∧ 
             is_prime (n - 12) ∧ 
             (n - 12 = 13) ∧
             (∀ m : ℕ, is_perfect_square m → is_prime (m - 12) → m - 12 ≥ 13) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_12_less_than_perfect_square_l1938_193817


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1938_193863

theorem system_of_inequalities_solution (x : ℝ) :
  (2 * x > -1 ∧ x - 1 ≤ 0) ↔ (-1/2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1938_193863


namespace NUMINAMATH_CALUDE_jack_heart_queen_probability_l1938_193813

/-- Represents a standard deck of 52 playing cards -/
def StandardDeck : Nat := 52

/-- Number of Jacks in a standard deck -/
def NumJacks : Nat := 4

/-- Number of hearts in a standard deck -/
def NumHearts : Nat := 13

/-- Number of Queens in a standard deck -/
def NumQueens : Nat := 4

/-- Probability of drawing a Jack, then a heart, then a Queen from a standard deck -/
theorem jack_heart_queen_probability :
  (NumJacks / StandardDeck) * (NumHearts / (StandardDeck - 1)) * (NumQueens / (StandardDeck - 2)) = 1 / 663 := by
  sorry

end NUMINAMATH_CALUDE_jack_heart_queen_probability_l1938_193813


namespace NUMINAMATH_CALUDE_min_distinct_values_for_given_conditions_l1938_193803

/-- Given a list of positive integers with a unique mode, this function returns the minimum number of distinct values that can occur in the list. -/
def min_distinct_values (list_size : ℕ) (mode_frequency : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of distinct values for the given conditions -/
theorem min_distinct_values_for_given_conditions :
  min_distinct_values 2057 15 = 147 := by
  sorry

end NUMINAMATH_CALUDE_min_distinct_values_for_given_conditions_l1938_193803


namespace NUMINAMATH_CALUDE_ant_meeting_probability_l1938_193862

/-- Represents a cube with 8 vertices -/
structure Cube :=
  (vertices : Fin 8)

/-- Represents an ant on a vertex of the cube -/
structure Ant :=
  (position : Fin 8)

/-- Represents a movement of an ant along an edge -/
def AntMovement := Fin 8 → Fin 8

/-- The total number of possible movement combinations for 8 ants -/
def totalMovements : ℕ := 3^8

/-- The number of non-colliding movement configurations -/
def nonCollidingMovements : ℕ := 24

/-- The probability of ants meeting -/
def probabilityOfMeeting : ℚ := 1 - (nonCollidingMovements : ℚ) / totalMovements

theorem ant_meeting_probability (c : Cube) (ants : Fin 8 → Ant) 
  (movements : Fin 8 → AntMovement) : 
  probabilityOfMeeting = 2381/2387 :=
sorry

end NUMINAMATH_CALUDE_ant_meeting_probability_l1938_193862


namespace NUMINAMATH_CALUDE_sum_interior_ninth_row_l1938_193869

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The ninth row of Pascal's Triangle -/
def ninth_row : ℕ := 9

theorem sum_interior_ninth_row :
  sum_interior ninth_row = 254 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_ninth_row_l1938_193869


namespace NUMINAMATH_CALUDE_crayon_distribution_l1938_193868

/-- The problem of distributing crayons among Fred, Benny, Jason, and Sarah. -/
theorem crayon_distribution (total : ℕ) (fred benny jason sarah : ℕ) : 
  total = 96 →
  fred = 2 * benny →
  jason = 3 * sarah →
  benny = 12 →
  total = fred + benny + jason + sarah →
  fred = 24 ∧ benny = 12 ∧ jason = 45 ∧ sarah = 15 := by
  sorry

#check crayon_distribution

end NUMINAMATH_CALUDE_crayon_distribution_l1938_193868


namespace NUMINAMATH_CALUDE_expression_value_l1938_193884

theorem expression_value : 
  (2023^3 - 3 * 2023^2 * 2024 + 4 * 2023 * 2024^2 - 2024^3 + 2) / (2023 * 2024) = 2023 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1938_193884


namespace NUMINAMATH_CALUDE_tetrahedron_volume_l1938_193882

/-- The volume of a tetrahedron with vertices on the positive coordinate axes -/
theorem tetrahedron_volume (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + b^2 = 25) (h5 : b^2 + c^2 = 36) (h6 : c^2 + a^2 = 49) :
  (1 / 6 : ℝ) * a * b * c = Real.sqrt 95 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_l1938_193882


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1938_193857

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point) : Point :=
  { x := p.x, y := -p.y }

theorem symmetric_point_coordinates :
  let B : Point := { x := 4, y := -1 }
  let A : Point := symmetricXAxis B
  A.x = 4 ∧ A.y = 1 := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1938_193857


namespace NUMINAMATH_CALUDE_min_value_fraction_l1938_193811

theorem min_value_fraction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 + b^2) / (a*b - b^2) ≥ 2 + 2*Real.sqrt 2 ∧
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ (a^2 + b^2) / (a*b - b^2) = 2 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_fraction_l1938_193811


namespace NUMINAMATH_CALUDE_max_d_value_l1938_193860

def is_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def number_form (d e : ℕ) : ℕ := 707330 + d * 1000 + e

theorem max_d_value :
  ∃ (d e : ℕ),
    is_digit d ∧
    is_digit e ∧
    number_form d e % 33 = 0 ∧
    (∀ (d' e' : ℕ), is_digit d' ∧ is_digit e' ∧ number_form d' e' % 33 = 0 → d' ≤ d) ∧
    d = 6 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l1938_193860


namespace NUMINAMATH_CALUDE_fifteenth_prime_l1938_193859

theorem fifteenth_prime (p : ℕ → ℕ) (h : ∀ n, Prime (p n)) (h15 : p 7 = 15) : p 15 = 47 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l1938_193859


namespace NUMINAMATH_CALUDE_backpacks_sold_to_dept_store_l1938_193898

def total_backpacks : ℕ := 48
def total_cost : ℕ := 576
def swap_meet_sold : ℕ := 17
def swap_meet_price : ℕ := 18
def dept_store_price : ℕ := 25
def remainder_price : ℕ := 22
def total_profit : ℕ := 442

theorem backpacks_sold_to_dept_store :
  ∃ x : ℕ, 
    x * dept_store_price + 
    swap_meet_sold * swap_meet_price + 
    (total_backpacks - swap_meet_sold - x) * remainder_price - 
    total_cost = total_profit ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_backpacks_sold_to_dept_store_l1938_193898


namespace NUMINAMATH_CALUDE_fifth_term_is_five_l1938_193865

/-- An arithmetic sequence is represented by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * seq.common_difference

theorem fifth_term_is_five
  (seq : ArithmeticSequence)
  (h : seq.nth_term 2 + seq.nth_term 4 = 10) :
  seq.nth_term 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_is_five_l1938_193865


namespace NUMINAMATH_CALUDE_william_farm_tax_l1938_193819

/-- Calculates an individual's farm tax payment given the total tax collected and their land percentage -/
def individual_farm_tax (total_tax : ℝ) (land_percentage : ℝ) : ℝ :=
  land_percentage * total_tax

/-- Proves that given the conditions, Mr. William's farm tax payment is $960 -/
theorem william_farm_tax :
  let total_tax : ℝ := 3840
  let william_land_percentage : ℝ := 0.25
  individual_farm_tax total_tax william_land_percentage = 960 := by
sorry

end NUMINAMATH_CALUDE_william_farm_tax_l1938_193819


namespace NUMINAMATH_CALUDE_a_minus_c_equals_three_l1938_193826

theorem a_minus_c_equals_three
  (e f a b c d : ℝ)
  (h1 : e = a^2 + b^2)
  (h2 : f = c^2 + d^2)
  (h3 : a - b = c + d + 9)
  (h4 : a + b = c - d - 3)
  (h5 : f - e = 5*a + 2*b + 3*c + 4*d) :
  a - c = 3 := by
sorry

end NUMINAMATH_CALUDE_a_minus_c_equals_three_l1938_193826


namespace NUMINAMATH_CALUDE_initial_principal_is_8000_l1938_193844

/-- The compound interest formula for annual compounding -/
def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

/-- Theorem: Given the conditions of the problem, the initial principal is 8000 -/
theorem initial_principal_is_8000 :
  ∃ P : ℝ,
    compound_interest P 0.05 2 = 8820 ∧
    P = 8000 := by
  sorry

end NUMINAMATH_CALUDE_initial_principal_is_8000_l1938_193844


namespace NUMINAMATH_CALUDE_two_hour_charge_l1938_193816

/-- Represents the pricing structure for therapy sessions -/
structure TherapyPricing where
  firstHourCharge : ℕ
  additionalHourCharge : ℕ
  hourDifference : firstHourCharge = additionalHourCharge + 25

/-- Calculates the total charge for a given number of therapy hours -/
def totalCharge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  if hours = 0 then 0
  else pricing.firstHourCharge + (hours - 1) * pricing.additionalHourCharge

/-- Theorem stating the total charge for 2 hours of therapy -/
theorem two_hour_charge (pricing : TherapyPricing) 
  (h : totalCharge pricing 5 = 250) : totalCharge pricing 2 = 115 := by
  sorry

end NUMINAMATH_CALUDE_two_hour_charge_l1938_193816


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1938_193843

theorem tan_alpha_value (α β : ℝ) 
  (h1 : Real.tan (3 * α - 2 * β) = 1 / 2)
  (h2 : Real.tan (5 * α - 4 * β) = 1 / 4) : 
  Real.tan α = 13 / 16 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1938_193843


namespace NUMINAMATH_CALUDE_triangle_circle_relation_l1938_193899

theorem triangle_circle_relation 
  (AO' AO₁ AB AC t s s₁ s₂ s₃ r r₁ α : ℝ) 
  (h1 : AO' * Real.sin (α/2) = r ∧ r = t/s)
  (h2 : AO₁ * Real.sin (α/2) = r₁ ∧ r₁ = t/s₁)
  (h3 : AO' * AO₁ = t^2 / (s * s₁ * Real.sin (α/2)^2))
  (h4 : Real.sin (α/2)^2 = (s₂ * s₃) / (AB * AC)) :
  AO' * AO₁ = AB * AC := by
  sorry

end NUMINAMATH_CALUDE_triangle_circle_relation_l1938_193899


namespace NUMINAMATH_CALUDE_chocolate_chip_calculation_l1938_193849

/-- The number of cups of chocolate chips needed for one recipe -/
def cups_per_recipe : ℕ := 2

/-- The number of recipes to be made -/
def number_of_recipes : ℕ := 23

/-- The total number of cups of chocolate chips needed -/
def total_cups : ℕ := cups_per_recipe * number_of_recipes

theorem chocolate_chip_calculation : total_cups = 46 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_calculation_l1938_193849


namespace NUMINAMATH_CALUDE_tobias_allowance_l1938_193864

/-- Calculates Tobias's monthly allowance based on given conditions --/
def monthly_allowance (shoe_cost : ℕ) (saving_months : ℕ) (lawn_price : ℕ) (shovel_price : ℕ) 
                      (change : ℕ) (lawns_mowed : ℕ) (driveways_shoveled : ℕ) : ℕ :=
  let total_earned := lawn_price * lawns_mowed + shovel_price * driveways_shoveled
  let total_had := shoe_cost + change
  let allowance_total := total_had - total_earned
  allowance_total / saving_months

theorem tobias_allowance :
  monthly_allowance 95 3 15 7 15 4 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tobias_allowance_l1938_193864


namespace NUMINAMATH_CALUDE_selected_number_in_fourth_group_l1938_193808

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  totalStudents : Nat
  sampleSize : Nat
  startingNumber : Nat

/-- Calculates the selected number for a given group in the systematic sampling -/
def selectedNumber (sampling : SystematicSampling) (groupIndex : Nat) : Nat :=
  sampling.startingNumber + (groupIndex - 1) * (sampling.totalStudents / sampling.sampleSize)

theorem selected_number_in_fourth_group (sampling : SystematicSampling) 
  (h1 : sampling.totalStudents = 1200)
  (h2 : sampling.sampleSize = 80)
  (h3 : sampling.startingNumber = 6) :
  selectedNumber sampling 4 = 51 := by
  sorry

end NUMINAMATH_CALUDE_selected_number_in_fourth_group_l1938_193808


namespace NUMINAMATH_CALUDE_counterpositive_equivalence_l1938_193801

theorem counterpositive_equivalence (a b c : ℝ) :
  (a^2 + b^2 + c^2 < 3 → a + b + c ≠ 3) ↔
  ¬(a + b + c = 3 → a^2 + b^2 + c^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_counterpositive_equivalence_l1938_193801


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1938_193890

theorem integer_solutions_quadratic_equation :
  {(x, y) : ℤ × ℤ | x^2 + y^2 = x + y + 2} =
  {(2, 1), (2, 0), (-1, 1), (-1, 0)} := by sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l1938_193890


namespace NUMINAMATH_CALUDE_unique_line_configuration_l1938_193830

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ  -- number of lines
  total_intersections : ℕ  -- total number of intersection points
  triple_intersections : ℕ  -- number of points where three lines intersect

/-- The specific configuration described in the problem -/
def problem_config : LineConfiguration :=
  { n := 8,  -- This is what we want to prove
    total_intersections := 16,
    triple_intersections := 6 }

/-- Theorem stating that the problem configuration is the only valid one -/
theorem unique_line_configuration :
  ∀ (config : LineConfiguration),
    (∀ (i j : ℕ), i < config.n → j < config.n → i ≠ j → ∃ (p : ℕ), p < config.total_intersections) →  -- every pair of lines intersects
    (∀ (i j k l : ℕ), i < config.n → j < config.n → k < config.n → l < config.n → 
      i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l → 
      ¬∃ (p : ℕ), p < config.total_intersections) →  -- no four lines pass through a single point
    config.total_intersections = 16 →
    config.triple_intersections = 6 →
    config = problem_config :=
by sorry

end NUMINAMATH_CALUDE_unique_line_configuration_l1938_193830


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l1938_193855

theorem roots_sum_of_squares (x₁ x₂ : ℝ) : 
  (2 * x₁^2 - 3 * x₁ - 1 = 0) → 
  (2 * x₂^2 - 3 * x₂ - 1 = 0) → 
  x₁^2 + x₂^2 = 13/4 := by
sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l1938_193855


namespace NUMINAMATH_CALUDE_power_equation_solution_l1938_193821

theorem power_equation_solution : 2^4 - 7 = 3^3 + (-18) := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l1938_193821


namespace NUMINAMATH_CALUDE_log_equation_equivalence_l1938_193861

theorem log_equation_equivalence (x : ℝ) :
  (∀ y : ℝ, y > 0 → ∃ z : ℝ, Real.exp z = y) →  -- This ensures logarithms are defined for positive reals
  (x > (3/2) ↔ (Real.log (x+5) + Real.log (2*x-3) = Real.log (2*x^2 + x - 15))) :=
by sorry

end NUMINAMATH_CALUDE_log_equation_equivalence_l1938_193861


namespace NUMINAMATH_CALUDE_nested_squares_perimeter_difference_l1938_193883

/-- The difference between the perimeters of two nested squares -/
theorem nested_squares_perimeter_difference :
  ∀ (x : ℝ),
  x > 0 →
  let small_square_side : ℝ := x
  let large_square_side : ℝ := x + 8
  let small_perimeter : ℝ := 4 * small_square_side
  let large_perimeter : ℝ := 4 * large_square_side
  large_perimeter - small_perimeter = 32 :=
by
  sorry

#check nested_squares_perimeter_difference

end NUMINAMATH_CALUDE_nested_squares_perimeter_difference_l1938_193883


namespace NUMINAMATH_CALUDE_books_per_child_l1938_193852

theorem books_per_child (num_children : ℕ) (teacher_books : ℕ) (total_books : ℕ) :
  num_children = 10 →
  teacher_books = 8 →
  total_books = 78 →
  ∃ (books_per_child : ℕ), books_per_child * num_children + teacher_books = total_books ∧ books_per_child = 7 :=
by sorry

end NUMINAMATH_CALUDE_books_per_child_l1938_193852


namespace NUMINAMATH_CALUDE_expression_equals_seventeen_l1938_193853

theorem expression_equals_seventeen : 1-(-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_seventeen_l1938_193853


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1938_193872

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1938_193872


namespace NUMINAMATH_CALUDE_abs_cubic_inequality_l1938_193894

theorem abs_cubic_inequality (x : ℝ) : |x| ≤ 2 → |3*x - x^3| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_cubic_inequality_l1938_193894


namespace NUMINAMATH_CALUDE_petyas_class_l1938_193834

theorem petyas_class (x y : ℕ) : 
  (2 * x : ℚ) / 3 + y / 7 = (x + y : ℚ) / 3 →  -- Condition 1, 2, 3
  x + y ≤ 40 →                                -- Condition 4
  x = 12                                      -- Conclusion
  := by sorry

end NUMINAMATH_CALUDE_petyas_class_l1938_193834


namespace NUMINAMATH_CALUDE_no_integer_regular_quadrilateral_pyramid_l1938_193832

theorem no_integer_regular_quadrilateral_pyramid :
  ¬ ∃ (g h f s v : ℕ), 
    g > 0 ∧ h > 0 ∧
    f * f = h * h + g * g / 2 ∧
    s = g * g + 2 * g * Int.sqrt (h * h + g * g / 4) ∧
    3 * v = g * g * h :=
by sorry

end NUMINAMATH_CALUDE_no_integer_regular_quadrilateral_pyramid_l1938_193832


namespace NUMINAMATH_CALUDE_fraction_simplification_l1938_193897

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1938_193897


namespace NUMINAMATH_CALUDE_rectangle_circumscribed_l1938_193878

/-- Two lines form a rectangle with the coordinate axes that can be circumscribed by a circle -/
theorem rectangle_circumscribed (k : ℝ) : 
  (∃ (x y : ℝ), x + 3*y - 7 = 0 ∧ k*x - y - 2 = 0) →
  (∀ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 → (x + 3*y - 7 = 0 ∨ k*x - y - 2 = 0 ∨ x = 0 ∨ y = 0)) →
  (k = 3) := by
sorry

end NUMINAMATH_CALUDE_rectangle_circumscribed_l1938_193878


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l1938_193804

theorem sine_cosine_inequality (α : Real) (h1 : 0 < α) (h2 : α < π) :
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2) ∧
  (2 * Real.sin (2 * α) = Real.cos (α / 2) ↔ α = π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l1938_193804


namespace NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l1938_193871

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

theorem smallest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → 26 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_two_digit_with_digit_product_12_l1938_193871


namespace NUMINAMATH_CALUDE_tea_mixture_price_l1938_193815

/-- Given three varieties of tea with prices and mixing ratios, calculate the price of the mixture --/
theorem tea_mixture_price (p1 p2 p3 : ℚ) (r1 r2 r3 : ℚ) : 
  p1 = 126 → p2 = 135 → p3 = 173.5 → r1 = 1 → r2 = 1 → r3 = 2 →
  (p1 * r1 + p2 * r2 + p3 * r3) / (r1 + r2 + r3) = 152 := by
  sorry

#check tea_mixture_price

end NUMINAMATH_CALUDE_tea_mixture_price_l1938_193815


namespace NUMINAMATH_CALUDE_quadratic_solution_l1938_193851

theorem quadratic_solution (x : ℝ) (h1 : x^2 - 3*x - 6 = 0) (h2 : x ≠ 0) :
  x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l1938_193851


namespace NUMINAMATH_CALUDE_bryden_quarter_sale_l1938_193850

/-- The amount a collector pays for state quarters as a percentage of face value -/
def collector_offer_percentage : ℚ := 2500

/-- The number of state quarters Bryden has -/
def bryden_quarters : ℕ := 5

/-- The face value of a single state quarter in dollars -/
def quarter_face_value : ℚ := 1/4

/-- The amount Bryden will receive for his quarters in dollars -/
def bryden_received_amount : ℚ := 31.25

theorem bryden_quarter_sale :
  (collector_offer_percentage / 100) * (bryden_quarters : ℚ) * quarter_face_value = bryden_received_amount := by
  sorry

end NUMINAMATH_CALUDE_bryden_quarter_sale_l1938_193850


namespace NUMINAMATH_CALUDE_system_solution_l1938_193877

theorem system_solution (x y z : ℝ) : 
  x + y + z = 9 ∧ 
  1/x + 1/y + 1/z = 1 ∧ 
  x*y + x*z + y*z = 27 → 
  x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1938_193877


namespace NUMINAMATH_CALUDE_money_distribution_inconsistency_l1938_193806

/-- Prove that the given conditions about money distribution are inconsistent -/
theorem money_distribution_inconsistency :
  ¬∃ (a b c : ℤ),
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧  -- Money amounts are non-negative
    a + c = 200 ∧            -- A and C together have 200
    b + c = 350 ∧            -- B and C together have 350
    c = 250                  -- C has 250
    := by sorry

end NUMINAMATH_CALUDE_money_distribution_inconsistency_l1938_193806


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_perimeter_l1938_193891

/-- A triangle with an inscribed circle -/
structure Triangle :=
  (area : ℝ)
  (perimeter : ℝ)
  (inradius : ℝ)

/-- The theorem stating that if a triangle's area equals its perimeter, 
    then the radius of its inscribed circle is 2 -/
theorem inscribed_circle_radius_when_area_equals_perimeter 
  (t : Triangle) 
  (h : t.area = t.perimeter) : 
  t.inradius = 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_when_area_equals_perimeter_l1938_193891


namespace NUMINAMATH_CALUDE_range_of_f_l1938_193867

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define the domain
def domain : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | -1 ≤ y ∧ y ≤ 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1938_193867


namespace NUMINAMATH_CALUDE_benny_piggy_bank_l1938_193881

theorem benny_piggy_bank (january_amount february_amount total_amount : ℕ) 
  (h1 : january_amount = 19)
  (h2 : february_amount = january_amount)
  (h3 : total_amount = 46) : 
  total_amount - (january_amount + february_amount) = 8 := by
  sorry

end NUMINAMATH_CALUDE_benny_piggy_bank_l1938_193881


namespace NUMINAMATH_CALUDE_teacher_age_l1938_193833

theorem teacher_age (num_students : ℕ) (student_avg_age : ℝ) (new_avg_age : ℝ) :
  num_students = 15 →
  student_avg_age = 10 →
  new_avg_age = student_avg_age + 1 →
  (num_students : ℝ) * student_avg_age + (new_avg_age * (num_students + 1) - num_students * student_avg_age) = 26 := by
  sorry

end NUMINAMATH_CALUDE_teacher_age_l1938_193833


namespace NUMINAMATH_CALUDE_quarters_per_machine_l1938_193837

/-- Represents the number of machines in the launderette -/
def num_machines : ℕ := 3

/-- Represents the number of dimes in each machine -/
def dimes_per_machine : ℕ := 100

/-- Represents the total amount of money from all machines in cents -/
def total_money : ℕ := 9000  -- $90 in cents

/-- Represents the value of a dime in cents -/
def dime_value : ℕ := 10

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

theorem quarters_per_machine :
  ∃ (q : ℕ), 
    q * quarter_value * num_machines + 
    dimes_per_machine * dime_value * num_machines = 
    total_money ∧ 
    q = 80 := by
  sorry

end NUMINAMATH_CALUDE_quarters_per_machine_l1938_193837


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_complement_relation_l1938_193835

theorem angle_with_special_supplement_complement_relation :
  ∀ x : ℝ,
  (0 < x) ∧ (x < 180) →
  (180 - x = 3 * (90 - x)) →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_complement_relation_l1938_193835


namespace NUMINAMATH_CALUDE_min_sum_perpendicular_sides_right_triangle_l1938_193822

theorem min_sum_perpendicular_sides_right_triangle (a b : ℝ) (h_positive_a : a > 0) (h_positive_b : b > 0) (h_area : a * b / 2 = 50) :
  a + b ≥ 20 ∧ (a + b = 20 ↔ a = 10 ∧ b = 10) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_perpendicular_sides_right_triangle_l1938_193822


namespace NUMINAMATH_CALUDE_percentage_calculation_l1938_193847

theorem percentage_calculation : 
  (0.47 * 1442 - 0.36 * 1412) + 66 = 235.42 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1938_193847


namespace NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l1938_193802

/-- Given a quadrilateral ABCD inscribed in a circle with radius R,
    where φ is the angle between its diagonals,
    the area S of the quadrilateral is equal to 2R^2 * sin(A) * sin(B) * sin(φ). -/
theorem area_cyclic_quadrilateral (R : ℝ) (A B φ : ℝ) (S : ℝ) 
    (hR : R > 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hφ : 0 < φ ∧ φ < π) :
  S = 2 * R^2 * Real.sin A * Real.sin B * Real.sin φ := by
  sorry

end NUMINAMATH_CALUDE_area_cyclic_quadrilateral_l1938_193802


namespace NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l1938_193809

theorem continuous_function_zero_on_interval 
  (f : ℝ → ℝ) 
  (hf_cont : Continuous f) 
  (hf_eq : ∀ x, f (2 * x^2 - 1) = 2 * x * f x) : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x = 0 := by sorry

end NUMINAMATH_CALUDE_continuous_function_zero_on_interval_l1938_193809


namespace NUMINAMATH_CALUDE_range_of_a_l1938_193810

-- Define the conditions p and q as functions
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

-- State the theorem
theorem range_of_a (a : ℝ) (h1 : a < 0) 
  (h2 : ∀ x, ¬(p a x) → ¬(q x))  -- ¬p is necessary for ¬q
  (h3 : ∃ x, ¬(p a x) ∧ q x)     -- ¬p is not sufficient for ¬q
  : a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1938_193810


namespace NUMINAMATH_CALUDE_max_m_value_min_weighted_sum_of_squares_l1938_193895

-- Part 1
theorem max_m_value (m : ℝ) : 
  (∀ x : ℝ, |x - 3| + |x - m| ≥ 2*m) → m ≤ 1 :=
sorry

-- Part 2
theorem min_weighted_sum_of_squares :
  let f (a b c : ℝ) := 4*a^2 + 9*b^2 + c^2
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    f a b c ≥ 36/49 ∧
    (f a b c = 36/49 ↔ a = 9/49 ∧ b = 4/49 ∧ c = 36/49) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_min_weighted_sum_of_squares_l1938_193895


namespace NUMINAMATH_CALUDE_one_box_can_be_emptied_l1938_193856

/-- Represents a state of three boxes with balls -/
structure BoxState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents an operation of doubling balls in one box by transferring from another -/
inductive DoubleOperation
  | DoubleAFromB
  | DoubleAFromC
  | DoubleBFromA
  | DoubleBFromC
  | DoubleCFromA
  | DoubleCFromB

/-- Applies a single doubling operation to a BoxState -/
def applyOperation (state : BoxState) (op : DoubleOperation) : BoxState :=
  match op with
  | DoubleOperation.DoubleAFromB => ⟨state.a * 2, state.b - state.a, state.c⟩
  | DoubleOperation.DoubleAFromC => ⟨state.a * 2, state.b, state.c - state.a⟩
  | DoubleOperation.DoubleBFromA => ⟨state.a - state.b, state.b * 2, state.c⟩
  | DoubleOperation.DoubleBFromC => ⟨state.a, state.b * 2, state.c - state.b⟩
  | DoubleOperation.DoubleCFromA => ⟨state.a - state.c, state.b, state.c * 2⟩
  | DoubleOperation.DoubleCFromB => ⟨state.a, state.b - state.c, state.c * 2⟩

/-- Applies a sequence of doubling operations to a BoxState -/
def applyOperations (state : BoxState) (ops : List DoubleOperation) : BoxState :=
  ops.foldl applyOperation state

/-- Predicate to check if any box is empty -/
def isAnyBoxEmpty (state : BoxState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem stating that one box can be emptied -/
theorem one_box_can_be_emptied (initial : BoxState) :
  ∃ (ops : List DoubleOperation), isAnyBoxEmpty (applyOperations initial ops) :=
sorry

end NUMINAMATH_CALUDE_one_box_can_be_emptied_l1938_193856


namespace NUMINAMATH_CALUDE_range_of_m_l1938_193839

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| ≤ 2
def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

-- Define the condition that ¬p is sufficient but not necessary for ¬q
def sufficient_not_necessary (m : ℝ) : Prop :=
  (∀ x, ¬(p x) → ¬(q x m)) ∧ ¬(∀ x, ¬(q x m) → ¬(p x))

-- Theorem statement
theorem range_of_m :
  ∀ m, sufficient_not_necessary m ↔ (2 < m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1938_193839


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1938_193827

/-- A plane in 3D space represented by its equation coefficients -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- A point in 3D space -/
structure Point where
  x : ℤ
  y : ℤ
  z : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- Check if two planes are parallel -/
def planesParallel (plane1 : Plane) (plane2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ plane1.a = k * plane2.a ∧ plane1.b = k * plane2.b ∧ plane1.c = k * plane2.c

/-- The greatest common divisor of four integers is 1 -/
def gcdOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a.natAbs b.natAbs) c.natAbs) d.natAbs = 1

theorem plane_equation_proof (givenPlane : Plane) (point : Point) :
  givenPlane.a = 3 ∧ givenPlane.b = -2 ∧ givenPlane.c = 4 ∧ givenPlane.d = 5 →
  point.x = 2 ∧ point.y = -3 ∧ point.z = 1 →
  ∃ (soughtPlane : Plane),
    soughtPlane.a = 3 ∧
    soughtPlane.b = -2 ∧
    soughtPlane.c = 4 ∧
    soughtPlane.d = -16 ∧
    soughtPlane.a > 0 ∧
    pointOnPlane soughtPlane point ∧
    planesParallel soughtPlane givenPlane ∧
    gcdOne soughtPlane.a soughtPlane.b soughtPlane.c soughtPlane.d :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l1938_193827


namespace NUMINAMATH_CALUDE_power_function_through_point_l1938_193879

/-- If f(x) = x^n is a power function and f(2) = √2, then f(4) = 2 -/
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) : 
  (∀ x > 0, f x = x ^ n) →    -- f is a power function
  f 2 = Real.sqrt 2 →         -- f passes through (2, √2)
  f 4 = 2 := by               -- then f(4) = 2
sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1938_193879


namespace NUMINAMATH_CALUDE_no_subset_with_unique_distance_l1938_193831

theorem no_subset_with_unique_distance : 
  ¬∃ (M : Set ℝ), (Set.Nonempty M) ∧ 
    (∀ (r : ℝ) (a : ℝ), r > 0 → a ∈ M → 
      ∃! (b : ℝ), b ∈ M ∧ |a - b| = r) :=
by sorry

end NUMINAMATH_CALUDE_no_subset_with_unique_distance_l1938_193831


namespace NUMINAMATH_CALUDE_min_value_of_a_l1938_193800

theorem min_value_of_a (p : ∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 2| ≤ a) : 
  ∀ b : ℝ, b < 3 → ¬(∃ x₀ : ℝ, |x₀ + 1| + |x₀ - 2| ≤ b) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_a_l1938_193800


namespace NUMINAMATH_CALUDE_petyas_coins_l1938_193842

theorem petyas_coins (total : ℕ) (not_two : ℕ) (not_ten : ℕ) (not_one : ℕ) 
  (h_total : total = 25)
  (h_not_two : not_two = 19)
  (h_not_ten : not_ten = 20)
  (h_not_one : not_one = 16) :
  total - ((total - not_two) + (total - not_ten) + (total - not_one)) = 5 := by
  sorry

end NUMINAMATH_CALUDE_petyas_coins_l1938_193842


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1938_193805

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_sum : a 1 + a 6 = 12) 
  (h_a4 : a 4 = 7) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1938_193805


namespace NUMINAMATH_CALUDE_parabola_solutions_l1938_193820

/-- A parabola defined by y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_solutions (p : Parabola) (m : ℝ) :
  p.y (-4) = m →
  p.y 0 = m →
  p.y 2 = 1 →
  p.y 4 = 0 →
  (∀ x : ℝ, p.y x = 0 ↔ x = 4 ∨ x = -8) :=
sorry

end NUMINAMATH_CALUDE_parabola_solutions_l1938_193820


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1938_193892

theorem arithmetic_calculations :
  (5 + (-6) + 3 - 8 - (-4) = -2) ∧
  (-2^2 - 3 * (-1)^3 - (-1) / (-1/2)^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1938_193892
