import Mathlib

namespace NUMINAMATH_CALUDE_discount_comparison_l3990_399054

/-- The original bill amount in dollars -/
def original_bill : ℝ := 8000

/-- The single discount rate -/
def single_discount_rate : ℝ := 0.3

/-- The first successive discount rate -/
def first_successive_discount_rate : ℝ := 0.2

/-- The second successive discount rate -/
def second_successive_discount_rate : ℝ := 0.1

/-- The difference between the two discount scenarios -/
def discount_difference : ℝ := 160

theorem discount_comparison :
  let single_discounted := original_bill * (1 - single_discount_rate)
  let successive_discounted := original_bill * (1 - first_successive_discount_rate) * (1 - second_successive_discount_rate)
  successive_discounted - single_discounted = discount_difference := by
  sorry

end NUMINAMATH_CALUDE_discount_comparison_l3990_399054


namespace NUMINAMATH_CALUDE_candy_store_spend_l3990_399043

def weekly_allowance : ℚ := 3/2

def arcade_spend (allowance : ℚ) : ℚ := (3/5) * allowance

def toy_store_spend (remaining : ℚ) : ℚ := (1/3) * remaining

theorem candy_store_spend :
  let remaining_after_arcade := weekly_allowance - arcade_spend weekly_allowance
  let remaining_after_toy := remaining_after_arcade - toy_store_spend remaining_after_arcade
  remaining_after_toy = 2/5 := by sorry

end NUMINAMATH_CALUDE_candy_store_spend_l3990_399043


namespace NUMINAMATH_CALUDE_limit_of_a_sequence_l3990_399057

def a (n : ℕ) : ℚ := (9 - n^3) / (1 + 2*n^3)

theorem limit_of_a_sequence :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - (-1/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_of_a_sequence_l3990_399057


namespace NUMINAMATH_CALUDE_f_geq_f0_range_of_a_l3990_399065

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- Theorem 1: f(x) ≥ f(0) for all x
theorem f_geq_f0 : ∀ x : ℝ, f x ≥ f 0 := by sorry

-- Theorem 2: Given 2f(x) ≥ f(a+1) for all x, the range of a is [-4.5, 1.5]
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * f x ≥ f (a + 1)) → -4.5 ≤ a ∧ a ≤ 1.5 := by sorry

end NUMINAMATH_CALUDE_f_geq_f0_range_of_a_l3990_399065


namespace NUMINAMATH_CALUDE_john_and_sarah_money_l3990_399080

theorem john_and_sarah_money (john_money : ℚ) (sarah_money : ℚ)
  (h1 : john_money = 5 / 8)
  (h2 : sarah_money = 7 / 16) :
  john_money + sarah_money = 1.0625 := by
sorry

end NUMINAMATH_CALUDE_john_and_sarah_money_l3990_399080


namespace NUMINAMATH_CALUDE_odot_examples_l3990_399036

def odot (a b : ℚ) : ℚ := a * (a + b) - 1

theorem odot_examples :
  (odot 3 (-2) = 2) ∧ (odot (-2) (odot 3 5) = -43) := by
  sorry

end NUMINAMATH_CALUDE_odot_examples_l3990_399036


namespace NUMINAMATH_CALUDE_polynomial_not_perfect_square_l3990_399002

theorem polynomial_not_perfect_square (a b c d : ℤ) (n : ℕ+) :
  ∃ (S : Finset ℕ), 
    S.card ≥ n / 4 ∧ 
    ∀ m ∈ S, m ≤ n ∧ 
    ¬∃ (k : ℤ), (m^5 : ℤ) + d*m^4 + c*m^3 + b*m^2 + 2023*m + a = k^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_perfect_square_l3990_399002


namespace NUMINAMATH_CALUDE_janet_paperclips_used_l3990_399049

/-- The number of paper clips Janet used during the day -/
def paperclips_used (initial : ℕ) (found : ℕ) (given_per_friend : ℕ) (num_friends : ℕ) (final : ℕ) : ℕ :=
  initial + found - given_per_friend * num_friends - final

/-- Theorem stating that Janet used 64 paper clips during the day -/
theorem janet_paperclips_used :
  paperclips_used 85 20 5 3 26 = 64 := by
  sorry

end NUMINAMATH_CALUDE_janet_paperclips_used_l3990_399049


namespace NUMINAMATH_CALUDE_variance_transformed_l3990_399039

-- Define a random variable ξ
variable (ξ : ℝ → ℝ)

-- Define the variance operator D
noncomputable def D (X : ℝ → ℝ) : ℝ := sorry

-- Given condition
axiom variance_xi : D ξ = 2

-- Theorem to prove
theorem variance_transformed : D (fun ω => 2 * ξ ω + 3) = 8 := by sorry

end NUMINAMATH_CALUDE_variance_transformed_l3990_399039


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l3990_399074

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, m₁ * x + y = b₁ ↔ m₂ * x + y = b₂) ↔ m₁ = m₂

/-- Given two lines l₁ and l₂, prove that if they are parallel, then m = -2 -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, m * x + 2 * y - 3 = 0 ↔ 3 * x + (m - 1) * y + m - 6 = 0) →
  m = -2 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l3990_399074


namespace NUMINAMATH_CALUDE_wall_bricks_count_l3990_399083

/-- Represents the wall construction scenario --/
structure WallConstruction where
  /-- Total number of bricks in the wall --/
  total_bricks : ℕ
  /-- Time taken by the first bricklayer alone (in hours) --/
  time_worker1 : ℕ
  /-- Time taken by the second bricklayer alone (in hours) --/
  time_worker2 : ℕ
  /-- Reduction in combined output when working together (in bricks per hour) --/
  output_reduction : ℕ
  /-- Actual time taken to complete the wall (in hours) --/
  actual_time : ℕ

/-- Theorem stating the number of bricks in the wall --/
theorem wall_bricks_count (w : WallConstruction) 
  (h1 : w.time_worker1 = 8)
  (h2 : w.time_worker2 = 12)
  (h3 : w.output_reduction = 15)
  (h4 : w.actual_time = 6) :
  w.total_bricks = 360 :=
sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l3990_399083


namespace NUMINAMATH_CALUDE_prime_remainder_30_l3990_399006

theorem prime_remainder_30 (a : ℕ) (h_prime : Nat.Prime a) :
  ∃ (q r : ℕ), a = 30 * q + r ∧ 0 ≤ r ∧ r < 30 ∧ (Nat.Prime r ∨ r = 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_remainder_30_l3990_399006


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_eq_198_l3990_399085

/-- The polynomial for which we calculate the sum of squares of coefficients -/
def p (x : ℝ) : ℝ := 3 * (x^5 + 4*x^3 + 2*x + 1)

/-- The sum of squares of coefficients of the polynomial p -/
def sum_of_squares_of_coefficients : ℝ :=
  (3^2) + (12^2) + (6^2) + (3^2) + (0^2) + (0^2)

theorem sum_of_squares_of_coefficients_eq_198 :
  sum_of_squares_of_coefficients = 198 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_eq_198_l3990_399085


namespace NUMINAMATH_CALUDE_shortest_path_on_right_angle_polyhedron_l3990_399013

/-- A polyhedron with all dihedral angles as right angles -/
structure RightAnglePolyhedron where
  -- We don't need to define the full structure, just what we need for the theorem
  edge_length : ℝ
  all_dihedral_angles_right : True  -- placeholder for the condition

/-- The shortest path between two vertices on the surface of the polyhedron -/
def shortest_surface_path (p : RightAnglePolyhedron) (X Y : ℝ × ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would depend on how we represent the polyhedron

theorem shortest_path_on_right_angle_polyhedron 
  (p : RightAnglePolyhedron) 
  (X Y : ℝ × ℝ × ℝ) 
  (h_adjacent : True)  -- placeholder for the condition that X and Y are on adjacent faces
  (h_diagonal : True)  -- placeholder for the condition that X and Y are diagonally opposite
  (h_unit_edge : p.edge_length = 1) :
  shortest_surface_path p X Y = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_on_right_angle_polyhedron_l3990_399013


namespace NUMINAMATH_CALUDE_codes_lost_with_no_leading_zeros_l3990_399067

/-- The number of digits in each code -/
def code_length : ℕ := 5

/-- The number of possible digits (0 to 9) -/
def digit_options : ℕ := 10

/-- The number of non-zero digits (1 to 9) -/
def non_zero_digits : ℕ := 9

/-- Calculates the total number of possible codes -/
def total_codes : ℕ := digit_options ^ code_length

/-- Calculates the number of codes without leading zeros -/
def codes_without_leading_zeros : ℕ := non_zero_digits * (digit_options ^ (code_length - 1))

/-- The theorem to be proved -/
theorem codes_lost_with_no_leading_zeros :
  total_codes - codes_without_leading_zeros = 10000 := by
  sorry


end NUMINAMATH_CALUDE_codes_lost_with_no_leading_zeros_l3990_399067


namespace NUMINAMATH_CALUDE_welders_left_l3990_399031

/-- Proves that 12 welders left the project given the initial conditions and remaining time. -/
theorem welders_left (initial_welders : ℕ) (initial_days : ℝ) (remaining_days : ℝ) : 
  initial_welders = 36 →
  initial_days = 3 →
  remaining_days = 3.0000000000000004 →
  (initial_welders - (initial_welders - remaining_days * initial_welders / (initial_days + remaining_days - 1))) = 12 := by
sorry

end NUMINAMATH_CALUDE_welders_left_l3990_399031


namespace NUMINAMATH_CALUDE_line_l_equation_line_l_prime_equation_l3990_399068

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0
def l₂ (x y : ℝ) : Prop := 2 * x - 3 * y + 8 = 0

-- Define the intersection point M
def M : ℝ × ℝ := (-1, 2)

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Define the symmetry point
def sym_point : ℝ × ℝ := (1, -1)

-- Theorem for the equation of line l
theorem line_l_equation : 
  ∃ (m : ℝ), ∀ (x y : ℝ), 
    (l₁ x y ∧ l₂ x y → (x, y) = M) → 
    (∀ (a b : ℝ), perp_line a b → (y - M.2) = m * (x - M.1)) → 
    (x - 2 * y + 5 = 0) :=
sorry

-- Theorem for the equation of line l′
theorem line_l_prime_equation :
  ∀ (x y : ℝ),
    (∃ (x' y' : ℝ), l₁ x' y' ∧ 
      x' = 2 * sym_point.1 - x ∧ 
      y' = 2 * sym_point.2 - y) →
    (3 * x + 4 * y + 7 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_l_equation_line_l_prime_equation_l3990_399068


namespace NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3990_399066

/-- A parabola with equation y = x^2 - 10x + d + 4 has its vertex on the x-axis if and only if d = 21 -/
theorem parabola_vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 10*x + d + 4 = 0 ∧ 
   ∀ y : ℝ, y^2 - 10*y + d + 4 ≥ x^2 - 10*x + d + 4) ↔ 
  d = 21 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_x_axis_l3990_399066


namespace NUMINAMATH_CALUDE_kates_hair_length_l3990_399091

theorem kates_hair_length :
  ∀ (kate emily logan : ℝ),
  kate = (1/2) * emily →
  emily = logan + 6 →
  logan = 20 →
  kate = 13 := by
sorry

end NUMINAMATH_CALUDE_kates_hair_length_l3990_399091


namespace NUMINAMATH_CALUDE_investment_equation_l3990_399042

/-- Proves the equation for the investment problem -/
theorem investment_equation (x : ℝ) (h : x > 0) : (106960 / (x + 500)) - (50760 / x) = 20 := by
  sorry

end NUMINAMATH_CALUDE_investment_equation_l3990_399042


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3990_399041

/-- Given an increasing function f: ℝ → ℝ with f(0) = -1 and f(3) = 1,
    the set {x ∈ ℝ | |f(x)| < 1} is equal to the open interval (0, 3). -/
theorem solution_set_equivalence (f : ℝ → ℝ) 
    (h_increasing : ∀ x y, x < y → f x < f y)
    (h_f_0 : f 0 = -1)
    (h_f_3 : f 3 = 1) :
    {x : ℝ | |f x| < 1} = Set.Ioo 0 3 := by
  sorry


end NUMINAMATH_CALUDE_solution_set_equivalence_l3990_399041


namespace NUMINAMATH_CALUDE_average_temperature_problem_l3990_399045

/-- The average temperature problem -/
theorem average_temperature_problem 
  (temp_mon : ℝ) 
  (temp_tue : ℝ) 
  (temp_wed : ℝ) 
  (temp_thu : ℝ) 
  (temp_fri : ℝ) 
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : temp_mon = 42)
  (h3 : temp_fri = 10) :
  (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 40 := by
  sorry


end NUMINAMATH_CALUDE_average_temperature_problem_l3990_399045


namespace NUMINAMATH_CALUDE_three_cubes_volume_l3990_399090

theorem three_cubes_volume (s₁ s₂ : ℝ) (h₁ : s₁ > 0) (h₂ : s₂ > 0) : 
  6 * (s₁ + s₂)^2 = 864 → 2 * s₁^3 + s₂^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_three_cubes_volume_l3990_399090


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3990_399035

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 2
  (1 + x) / (1 - x) / (x - 2 * x / (1 - x)) = - (Real.sqrt 2 + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l3990_399035


namespace NUMINAMATH_CALUDE_calculation_proof_l3990_399053

theorem calculation_proof : (-1) * (-4) + 3^2 / (7 - 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3990_399053


namespace NUMINAMATH_CALUDE_shooter_hit_rate_l3990_399087

theorem shooter_hit_rate (p : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) 
  (h2 : (1 - (1 - p)^4) = 80/81) : p = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_shooter_hit_rate_l3990_399087


namespace NUMINAMATH_CALUDE_tripled_minus_six_l3990_399000

theorem tripled_minus_six (x : ℝ) : 3 * x - 6 = 15 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_tripled_minus_six_l3990_399000


namespace NUMINAMATH_CALUDE_log_equation_solution_l3990_399014

theorem log_equation_solution (a : ℝ) (h1 : a > 1) 
  (h2 : Real.log a / Real.log 5 + Real.log a / Real.log 3 = 
        (Real.log a / Real.log 5) * (Real.log a / Real.log 3)) : 
  a = 15 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3990_399014


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3990_399099

theorem log_equality_implies_ratio_one (p q : ℝ) (hp : p > 0) (hq : q > 0) 
  (h : Real.log p / Real.log 8 = Real.log q / Real.log 18 ∧ 
       Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) : 
  p / q = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_one_l3990_399099


namespace NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l3990_399040

/-- Represents the drinking habits of Mo --/
structure MoDrinkingHabits where
  rainyDayHotChocolate : ℚ
  nonRainyDayTea : ℕ
  totalCups : ℕ
  teaMoreThanHotChocolate : ℕ
  rainyDays : ℕ

/-- Theorem stating Mo's hot chocolate consumption on rainy mornings --/
theorem mo_hot_chocolate_consumption (mo : MoDrinkingHabits)
  (h1 : mo.nonRainyDayTea = 3)
  (h2 : mo.totalCups = 20)
  (h3 : mo.teaMoreThanHotChocolate = 10)
  (h4 : mo.rainyDays = 2)
  (h5 : (7 - mo.rainyDays) * mo.nonRainyDayTea + mo.rainyDays * mo.rainyDayHotChocolate = mo.totalCups)
  (h6 : (7 - mo.rainyDays) * mo.nonRainyDayTea = mo.rainyDays * mo.rainyDayHotChocolate + mo.teaMoreThanHotChocolate) :
  mo.rainyDayHotChocolate = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l3990_399040


namespace NUMINAMATH_CALUDE_similar_triangle_longest_side_l3990_399081

theorem similar_triangle_longest_side 
  (a b c : ℝ) 
  (h_triangle : a = 5 ∧ b = 12 ∧ c = 13) 
  (h_perimeter : ∃ k : ℝ, k > 0 ∧ k * (a + b + c) = 150) :
  ∃ s : ℝ, s = 65 ∧ s = max (k * a) (max (k * b) (k * c)) :=
sorry

end NUMINAMATH_CALUDE_similar_triangle_longest_side_l3990_399081


namespace NUMINAMATH_CALUDE_lcm_15_18_l3990_399032

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_15_18_l3990_399032


namespace NUMINAMATH_CALUDE_area_between_curves_l3990_399038

theorem area_between_curves : 
  let f (x : ℝ) := x^2
  let g (x : ℝ) := x^3
  ∫ x in (0: ℝ)..(1: ℝ), f x - g x = 1/12 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l3990_399038


namespace NUMINAMATH_CALUDE_square_areas_equal_l3990_399025

/-- Represents the configuration of squares and circles -/
structure SquareCircleConfig where
  circle_radius : ℝ
  num_small_squares : ℕ

/-- Calculates the area of the larger square -/
def larger_square_area (config : SquareCircleConfig) : ℝ :=
  4 * config.circle_radius ^ 2

/-- Calculates the total area of the smaller squares -/
def total_small_squares_area (config : SquareCircleConfig) : ℝ :=
  config.num_small_squares * (2 * config.circle_radius) ^ 2

/-- Theorem stating that the area of the larger square is equal to the total area of the smaller squares -/
theorem square_areas_equal (config : SquareCircleConfig) 
    (h1 : config.circle_radius = 3)
    (h2 : config.num_small_squares = 4) : 
  larger_square_area config = total_small_squares_area config ∧ 
  larger_square_area config = 144 := by
  sorry

#eval larger_square_area { circle_radius := 3, num_small_squares := 4 }
#eval total_small_squares_area { circle_radius := 3, num_small_squares := 4 }

end NUMINAMATH_CALUDE_square_areas_equal_l3990_399025


namespace NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3990_399022

noncomputable def f (x : ℝ) := Real.exp x * (x^2 + x + 1)

theorem f_monotonicity_and_extrema :
  (∀ x y, x < y ∧ y < -2 → f x < f y) ∧
  (∀ x y, -2 < x ∧ x < y ∧ y < -1 → f x > f y) ∧
  (∀ x y, -1 < x ∧ x < y → f x < f y) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-2)| < δ ∧ x ≠ -2 → f x < f (-2)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, |x - (-1)| < δ ∧ x ≠ -1 → f x > f (-1)) ∧
  f (-2) = 3 / Real.exp 2 ∧
  f (-1) = 1 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_f_monotonicity_and_extrema_l3990_399022


namespace NUMINAMATH_CALUDE_triangle_side_length_l3990_399017

/-- Proves that in a triangle ABC with A = 60°, B = 45°, and c = 20, the length of side a is equal to 30√2 - 10√6. -/
theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 3 → -- 60° in radians
  B = π / 4 → -- 45° in radians
  c = 20 →
  a = 30 * Real.sqrt 2 - 10 * Real.sqrt 6 :=
by sorry


end NUMINAMATH_CALUDE_triangle_side_length_l3990_399017


namespace NUMINAMATH_CALUDE_price_change_equivalence_l3990_399021

theorem price_change_equivalence :
  let initial_increase := 0.40
  let subsequent_decrease := 0.15
  let equivalent_single_increase := 0.19
  ∀ (original_price : ℝ),
    original_price > 0 →
    original_price * (1 + initial_increase) * (1 - subsequent_decrease) =
    original_price * (1 + equivalent_single_increase) := by
  sorry

end NUMINAMATH_CALUDE_price_change_equivalence_l3990_399021


namespace NUMINAMATH_CALUDE_joneal_stops_in_quarter_A_l3990_399094

/-- Represents the quarters of the circular track -/
inductive Quarter : Type
| A : Quarter
| B : Quarter
| C : Quarter
| D : Quarter

/-- Calculates the quarter in which a runner stops after running a given distance -/
def stopQuarter (trackCircumference : ℕ) (runDistance : ℕ) : Quarter :=
  match (runDistance % trackCircumference) / (trackCircumference / 4) with
  | 0 => Quarter.A
  | 1 => Quarter.B
  | 2 => Quarter.C
  | _ => Quarter.D

theorem joneal_stops_in_quarter_A :
  let trackCircumference : ℕ := 100
  let runDistance : ℕ := 10000
  stopQuarter trackCircumference runDistance = Quarter.A := by
  sorry

end NUMINAMATH_CALUDE_joneal_stops_in_quarter_A_l3990_399094


namespace NUMINAMATH_CALUDE_fraction_equality_l3990_399012

theorem fraction_equality (A B : ℤ) (x : ℝ) :
  (A / (x - 2) + B / (x^2 - 4*x + 8) = (x^2 - 4*x + 18) / (x^3 - 6*x^2 + 16*x - 16)) →
  (x ≠ 2 ∧ x ≠ 4 ∧ x^2 - 4*x + 8 ≠ 0) →
  B / A = -4 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3990_399012


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l3990_399046

theorem max_triangle_side_length :
  ∀ a b c : ℕ,
    a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different integer side lengths
    a + b + c = 30 →        -- Perimeter is 30 units
    a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
    a + b > c ∧ b + c > a ∧ a + c > b → -- Triangle inequality
    a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 -- Maximum side length is 14
  := by sorry

end NUMINAMATH_CALUDE_max_triangle_side_length_l3990_399046


namespace NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l3990_399048

/-- An angle is in the second quadrant if it's between 90° and 180° exclusive. -/
def is_in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

/-- An angle is obtuse if it's between 90° and 180° exclusive. -/
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α < 180

theorem second_quadrant_necessary_not_sufficient_for_obtuse :
  (∀ α, is_obtuse α → is_in_second_quadrant α) ∧
  (∃ α, is_in_second_quadrant α ∧ ¬is_obtuse α) :=
by sorry

end NUMINAMATH_CALUDE_second_quadrant_necessary_not_sufficient_for_obtuse_l3990_399048


namespace NUMINAMATH_CALUDE_movie_tickets_correct_l3990_399098

/-- The number of movie tickets sold for the given estimation --/
def movie_tickets : ℕ := 6

/-- The price of a pack of grain crackers --/
def cracker_price : ℚ := 2.25

/-- The price of a bottle of beverage --/
def beverage_price : ℚ := 1.5

/-- The price of a chocolate bar --/
def chocolate_price : ℚ := 1

/-- The average amount of estimated snack sales per movie ticket --/
def avg_sales_per_ticket : ℚ := 2.79

/-- Theorem stating that the number of movie tickets sold is correct --/
theorem movie_tickets_correct : 
  (3 * cracker_price + 4 * beverage_price + 4 * chocolate_price) / avg_sales_per_ticket = movie_tickets :=
by sorry

end NUMINAMATH_CALUDE_movie_tickets_correct_l3990_399098


namespace NUMINAMATH_CALUDE_star_seven_three_l3990_399024

-- Define the ⋆ operation
def star (x y : ℤ) : ℤ := 2 * x - 4 * y

-- State the theorem
theorem star_seven_three : star 7 3 = 2 := by sorry

end NUMINAMATH_CALUDE_star_seven_three_l3990_399024


namespace NUMINAMATH_CALUDE_cos_equality_l3990_399071

theorem cos_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) : 
  n = 43 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_l3990_399071


namespace NUMINAMATH_CALUDE_sqrt_2_simplest_l3990_399063

def is_simplest_sqrt (x : ℝ) (others : List ℝ) : Prop :=
  ∀ y ∈ others, ¬∃ (n : ℕ) (r : ℝ), n > 1 ∧ y = n * Real.sqrt r

theorem sqrt_2_simplest : is_simplest_sqrt (Real.sqrt 2) [Real.sqrt 8, Real.sqrt 12, Real.sqrt 18] := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_simplest_l3990_399063


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3990_399019

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 3 = 2 →
  a 4 + a 5 = 6 →
  a 5 + a 6 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3990_399019


namespace NUMINAMATH_CALUDE_expression_evaluation_l3990_399030

theorem expression_evaluation :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 - 1/3) = -13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3990_399030


namespace NUMINAMATH_CALUDE_only_d_is_odd_l3990_399084

theorem only_d_is_odd : ∀ n : ℤ,
  (n = 3 * 5 + 1 ∨ n = 2 * (3 + 5) ∨ n = 3 * (3 + 5) ∨ n = (3 + 5) / 2) → ¬(Odd n) ∧
  Odd (3 + 5 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_only_d_is_odd_l3990_399084


namespace NUMINAMATH_CALUDE_final_reflection_of_C_l3990_399010

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def C : ℝ × ℝ := (3, 2)

theorem final_reflection_of_C :
  (reflect_y_eq_x ∘ reflect_y ∘ reflect_x) C = (-2, -3) := by
  sorry

end NUMINAMATH_CALUDE_final_reflection_of_C_l3990_399010


namespace NUMINAMATH_CALUDE_infinitely_many_fixed_points_l3990_399020

def is_cyclic (f : ℕ → ℕ) : Prop :=
  ∀ n, ∃ k, k > 0 ∧ (f^[k] n = n)

theorem infinitely_many_fixed_points
  (f : ℕ → ℕ)
  (h1 : ∀ n, f n - n < 2021)
  (h2 : is_cyclic f) :
  ∀ m, ∃ n > m, f n = n :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_fixed_points_l3990_399020


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3990_399007

theorem polynomial_evaluation : 
  let x : ℕ := 2
  (x^4 + x^3 + x^2 + x + 1 : ℕ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3990_399007


namespace NUMINAMATH_CALUDE_batsman_85_run_innings_l3990_399061

/-- Represents a batsman's scoring record -/
structure Batsman where
  totalRuns : ℕ
  totalInnings : ℕ

/-- Calculate the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  (b.totalRuns : ℚ) / b.totalInnings

/-- The innings in which the batsman scored 85 -/
def targetInnings (b : Batsman) : ℕ := b.totalInnings

theorem batsman_85_run_innings (b : Batsman) 
  (h1 : average b = 37)
  (h2 : average { totalRuns := b.totalRuns - 85, totalInnings := b.totalInnings - 1 } = 34) :
  targetInnings b = 17 := by
  sorry

end NUMINAMATH_CALUDE_batsman_85_run_innings_l3990_399061


namespace NUMINAMATH_CALUDE_incorrect_expression_l3990_399060

theorem incorrect_expression (x y : ℚ) (h : x / y = 5 / 6) : 
  y / (2 * x - y) ≠ 6 / 1 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_expression_l3990_399060


namespace NUMINAMATH_CALUDE_equivalence_of_functional_equations_l3990_399047

theorem equivalence_of_functional_equations (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x + y) = f x + f y) ↔
  (∀ x y : ℝ, f (x * y + x + y) = f (x * y) + f x + f y) := by
  sorry

end NUMINAMATH_CALUDE_equivalence_of_functional_equations_l3990_399047


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l3990_399009

-- Problem 1
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  S 2 = S 6 →  -- S₂ = S₆
  a 4 = 1 →    -- a₄ = 1
  a 5 = -1 := by sorry

-- Problem 2
theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence
  a 4 - a 2 = 24 →  -- a₄ - a₂ = 24
  a 2 + a 3 = 6 →   -- a₂ + a₃ = 6
  a 1 = 1/5 ∧ q = 5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_geometric_sequence_problem_l3990_399009


namespace NUMINAMATH_CALUDE_marys_max_take_home_pay_l3990_399069

/-- Calculates Mary's take-home pay after taxes and insurance premium -/
def marys_take_home_pay (max_hours : ℕ) (regular_rate : ℚ) (overtime_rates : List ℚ) 
  (social_security_rate : ℚ) (medicare_rate : ℚ) (insurance_premium : ℚ) : ℚ :=
  sorry

/-- Theorem stating Mary's take-home pay for maximum hours worked -/
theorem marys_max_take_home_pay : 
  marys_take_home_pay 70 8 [1.25, 1.5, 1.75, 2] (8/100) (2/100) 50 = 706 := by
  sorry

end NUMINAMATH_CALUDE_marys_max_take_home_pay_l3990_399069


namespace NUMINAMATH_CALUDE_three_cats_meowing_l3990_399092

/-- The number of meows for three cats in a given time period -/
def total_meows (cat1_freq : ℕ) (time : ℕ) : ℕ :=
  let cat2_freq := 2 * cat1_freq
  let cat3_freq := cat2_freq / 3
  (cat1_freq + cat2_freq + cat3_freq) * time

/-- Theorem stating that the total number of meows for three cats in 5 minutes is 55 -/
theorem three_cats_meowing (cat1_freq : ℕ) (h : cat1_freq = 3) :
  total_meows cat1_freq 5 = 55 := by
  sorry

#eval total_meows 3 5

end NUMINAMATH_CALUDE_three_cats_meowing_l3990_399092


namespace NUMINAMATH_CALUDE_unique_function_satisfying_equations_l3990_399096

-- Define the function type
def RealFunction := ℝ → ℝ

-- State the theorem
theorem unique_function_satisfying_equations (f : RealFunction) :
  (∀ x : ℝ, f (x + 1) = 1 + f x) ∧
  (∀ x : ℝ, f (x^4 - x^2) = f x^4 - f x^2) →
  ∀ x : ℝ, f x = x :=
by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_equations_l3990_399096


namespace NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l3990_399005

theorem range_of_k_for_quadratic_inequality :
  {k : ℝ | ∀ x : ℝ, k * x^2 - k * x - 1 < 0} = {k : ℝ | -4 < k ∧ k ≤ 0} := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_for_quadratic_inequality_l3990_399005


namespace NUMINAMATH_CALUDE_min_value_expression_l3990_399095

/-- The minimum value of (s+5-3|cos t|)^2 + (s-2|sin t|)^2 is 2, where s and t are real numbers. -/
theorem min_value_expression : 
  ∃ (m : ℝ), m = 2 ∧ ∀ (s t : ℝ), (s + 5 - 3 * |Real.cos t|)^2 + (s - 2 * |Real.sin t|)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3990_399095


namespace NUMINAMATH_CALUDE_coin_game_theorem_l3990_399051

/-- Represents a pile of coins -/
structure CoinPile :=
  (count : ℕ)
  (hcount : count ≥ 2015)

/-- Represents the state of the three piles -/
structure GameState :=
  (pile1 : CoinPile)
  (pile2 : CoinPile)
  (pile3 : CoinPile)

/-- The polynomial f(x) = x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + 1 -/
def f (x : ℕ) : ℕ := x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + 1

/-- Represents a valid operation on the piles -/
inductive Operation
  | SplitEven (i : Fin 3) : Operation
  | RemoveOdd (i : Fin 3) : Operation

/-- Applies an operation to a game state -/
def applyOperation (state : GameState) (op : Operation) : GameState :=
  sorry

/-- Checks if a game state has reached the goal -/
def hasReachedGoal (state : GameState) : Prop :=
  ∃ (i : Fin 3), state.pile1.count ≥ 2017^2017 ∨ 
                 state.pile2.count ≥ 2017^2017 ∨ 
                 state.pile3.count ≥ 2017^2017

/-- The main theorem to prove -/
theorem coin_game_theorem (a b c : ℕ) 
  (ha : a ≥ 2015) (hb : b ≥ 2015) (hc : c ≥ 2015) :
  (∃ (ops : List Operation), 
    hasReachedGoal (ops.foldl applyOperation 
      { pile1 := ⟨a, ha⟩, pile2 := ⟨b, hb⟩, pile3 := ⟨c, hc⟩ })) ↔ 
  (f 2 = 2017 ∧ f 1 = 7) :=
sorry

end NUMINAMATH_CALUDE_coin_game_theorem_l3990_399051


namespace NUMINAMATH_CALUDE_min_root_of_negated_quadratic_l3990_399028

theorem min_root_of_negated_quadratic (p : ℝ) (r₁ r₂ : ℝ) :
  (∀ x, (x - 19) * (x - 83) = p ↔ x = r₁ ∨ x = r₂) →
  (∃ x, (x - r₁) * (x - r₂) = -p) →
  (∀ x, (x - r₁) * (x - r₂) = -p → x ≥ -19) ∧
  (∃ x, (x - r₁) * (x - r₂) = -p ∧ x = -19) :=
by sorry

end NUMINAMATH_CALUDE_min_root_of_negated_quadratic_l3990_399028


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3990_399058

/-- Given two circles A and B, where A is inside B, this theorem proves the diameter of A
    given the diameter of B, the distance between centers, and the ratio of areas. -/
theorem circle_diameter_ratio (dB : ℝ) (d : ℝ) (r : ℝ) : 
  dB = 20 →  -- Diameter of circle B
  d = 4 →    -- Distance between centers
  r = 5 →    -- Ratio of shaded area to area of circle A
  ∃ (dA : ℝ), dA = 2 * Real.sqrt (50 / 3) ∧ 
    (π * (dA / 2)^2) * (1 + r) = π * (dB / 2)^2 ∧ 
    d ≤ (dB - dA) / 2 :=
by sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3990_399058


namespace NUMINAMATH_CALUDE_right_triangle_sides_l3990_399086

/-- A right-angled triangle with sides in arithmetic progression and area 216 cm² -/
structure RightTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- The sides form an arithmetic progression
  arith_prog : ∃ (d : ℝ), b = a + d ∧ c = b + d
  -- The triangle is right-angled (Pythagorean theorem)
  right_angle : a^2 + b^2 = c^2
  -- The area of the triangle is 216
  area : a * b / 2 = 216

theorem right_triangle_sides (t : RightTriangle) : t.a = 18 ∧ t.b = 24 ∧ t.c = 30 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l3990_399086


namespace NUMINAMATH_CALUDE_first_term_is_two_l3990_399088

/-- A sequence of 5 terms where the differences between consecutive terms form an arithmetic sequence -/
def ArithmeticSequenceOfDifferences (a : Fin 5 → ℕ) : Prop :=
  ∃ d : ℕ, ∀ i : Fin 3, a (i + 1) - a i = d + i

theorem first_term_is_two (a : Fin 5 → ℕ) 
  (h1 : a 1 = 4) 
  (h2 : a 2 = 7)
  (h3 : a 3 = 11)
  (h4 : a 4 = 16)
  (h5 : ArithmeticSequenceOfDifferences a) : 
  a 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_two_l3990_399088


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3990_399077

theorem polynomial_simplification (x : ℝ) :
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) =
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3990_399077


namespace NUMINAMATH_CALUDE_total_interest_is_350_l3990_399079

/-- Calculates the total interest for two loans over a given time period. -/
def totalInterest (principal1 : ℝ) (rate1 : ℝ) (principal2 : ℝ) (rate2 : ℝ) (time : ℝ) : ℝ :=
  principal1 * rate1 * time + principal2 * rate2 * time

/-- Theorem stating that the total interest for the given loans and time period is 350. -/
theorem total_interest_is_350 :
  totalInterest 800 0.03 1400 0.05 3.723404255319149 = 350 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_is_350_l3990_399079


namespace NUMINAMATH_CALUDE_R_properties_l3990_399026

noncomputable def R (x : ℝ) : ℝ :=
  x^2 + 1/x^2 + (1-x)^2 + 1/(1-x)^2 + x^2/(1-x)^2 + (x-1)^2/x^2

theorem R_properties :
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = R (1/x)) ∧
  (∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = R (1-x)) ∧
  ¬ (∃ c : ℝ, ∀ x : ℝ, x ≠ 0 → x ≠ 1 → R x = c) :=
by sorry

end NUMINAMATH_CALUDE_R_properties_l3990_399026


namespace NUMINAMATH_CALUDE_car_speed_problem_l3990_399076

/-- Proves that the speed at which a car travels 1 kilometer in 12 seconds less time
    than it does at 60 km/h is 50 km/h. -/
theorem car_speed_problem (v : ℝ) : 
  (1000 / (60000 / 3600) - 1000 / (v / 3600) = 12) → v = 50 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_problem_l3990_399076


namespace NUMINAMATH_CALUDE_suraj_innings_l3990_399064

/-- The number of innings Suraj played before the last one -/
def n : ℕ := sorry

/-- Suraj's average before the last innings -/
def A : ℚ := sorry

/-- Suraj's new average after the last innings -/
def new_average : ℚ := 28

/-- The increase in Suraj's average after the last innings -/
def average_increase : ℚ := 8

/-- The runs Suraj scored in the last innings -/
def last_innings_runs : ℕ := 140

theorem suraj_innings : 
  (n : ℚ) * A + last_innings_runs = (n + 1) * new_average ∧
  new_average = A + average_increase ∧
  n = 14 := by sorry

end NUMINAMATH_CALUDE_suraj_innings_l3990_399064


namespace NUMINAMATH_CALUDE_blue_markers_count_l3990_399023

theorem blue_markers_count (total : ℝ) (red : ℝ) (blue : ℝ) : 
  total = 64.0 → red = 41.0 → blue = total - red → blue = 23.0 := by
  sorry

end NUMINAMATH_CALUDE_blue_markers_count_l3990_399023


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l3990_399003

/-- Represents a point on the grid --/
structure Point :=
  (x : Nat) (y : Nat)

/-- Represents the initial configuration of shaded squares --/
def initial_shaded : List Point :=
  [⟨2, 4⟩, ⟨3, 2⟩, ⟨5, 1⟩]

/-- The dimensions of the grid --/
def grid_width : Nat := 6
def grid_height : Nat := 5

/-- Checks if a point is within the grid --/
def is_valid_point (p : Point) : Bool :=
  p.x > 0 ∧ p.x ≤ grid_width ∧ p.y > 0 ∧ p.y ≤ grid_height

/-- Reflects a point across the vertical line of symmetry --/
def reflect_vertical (p : Point) : Point :=
  ⟨grid_width + 1 - p.x, p.y⟩

/-- Reflects a point across the horizontal line of symmetry --/
def reflect_horizontal (p : Point) : Point :=
  ⟨p.x, grid_height + 1 - p.y⟩

/-- Theorem: The minimum number of additional squares to shade for symmetry is 7 --/
theorem min_additional_squares_for_symmetry :
  ∃ (additional_shaded : List Point),
    (∀ p ∈ additional_shaded, is_valid_point p) ∧
    (∀ p ∈ initial_shaded, reflect_vertical p ∈ additional_shaded ∨ reflect_vertical p ∈ initial_shaded) ∧
    (∀ p ∈ initial_shaded, reflect_horizontal p ∈ additional_shaded ∨ reflect_horizontal p ∈ initial_shaded) ∧
    (∀ p ∈ additional_shaded, reflect_vertical p ∈ additional_shaded ∨ reflect_vertical p ∈ initial_shaded) ∧
    (∀ p ∈ additional_shaded, reflect_horizontal p ∈ additional_shaded ∨ reflect_horizontal p ∈ initial_shaded) ∧
    additional_shaded.length = 7 ∧
    (∀ other_shaded : List Point,
      (∀ p ∈ other_shaded, is_valid_point p) →
      (∀ p ∈ initial_shaded, reflect_vertical p ∈ other_shaded ∨ reflect_vertical p ∈ initial_shaded) →
      (∀ p ∈ initial_shaded, reflect_horizontal p ∈ other_shaded ∨ reflect_horizontal p ∈ initial_shaded) →
      (∀ p ∈ other_shaded, reflect_vertical p ∈ other_shaded ∨ reflect_vertical p ∈ initial_shaded) →
      (∀ p ∈ other_shaded, reflect_horizontal p ∈ other_shaded ∨ reflect_horizontal p ∈ initial_shaded) →
      other_shaded.length ≥ 7) :=
by
  sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l3990_399003


namespace NUMINAMATH_CALUDE_initial_pills_count_l3990_399050

/-- The number of pills Tony takes in the first two days -/
def pills_first_two_days : ℕ := 2 * 3 * 2

/-- The number of pills Tony takes in the next three days -/
def pills_next_three_days : ℕ := 1 * 3 * 3

/-- The number of pills Tony takes on the sixth day -/
def pills_sixth_day : ℕ := 2

/-- The number of pills left in the bottle after Tony's treatment -/
def pills_left : ℕ := 27

/-- The total number of pills Tony took during his treatment -/
def total_pills_taken : ℕ := pills_first_two_days + pills_next_three_days + pills_sixth_day

/-- Theorem: The initial number of pills in the bottle is 50 -/
theorem initial_pills_count : total_pills_taken + pills_left = 50 := by
  sorry

end NUMINAMATH_CALUDE_initial_pills_count_l3990_399050


namespace NUMINAMATH_CALUDE_unique_m_value_l3990_399004

def A (m : ℝ) : Set ℝ := {1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem unique_m_value : ∀ m : ℝ, B m ⊆ A m → m = -1 := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l3990_399004


namespace NUMINAMATH_CALUDE_polynomial_value_at_8_l3990_399052

def is_monic_degree_7 (p : ℝ → ℝ) : Prop :=
  ∃ a b c d e f : ℝ, p = λ x => x^7 + a*x^6 + b*x^5 + c*x^4 + d*x^3 + e*x^2 + f*x + (p 0)

theorem polynomial_value_at_8 
  (p : ℝ → ℝ) 
  (h_monic : is_monic_degree_7 p)
  (h1 : p 1 = 1) (h2 : p 2 = 2) (h3 : p 3 = 3) (h4 : p 4 = 4)
  (h5 : p 5 = 5) (h6 : p 6 = 6) (h7 : p 7 = 7) :
  p 8 = 5048 := by
sorry

end NUMINAMATH_CALUDE_polynomial_value_at_8_l3990_399052


namespace NUMINAMATH_CALUDE_complex_number_location_l3990_399093

theorem complex_number_location (z : ℂ) (h : (2 - 3*Complex.I)*z = 1 + Complex.I) :
  z.re < 0 ∧ z.im > 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l3990_399093


namespace NUMINAMATH_CALUDE_f_properties_l3990_399072

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 9*x

-- State the theorem
theorem f_properties :
  -- Part 1: Monotonicity of f
  (∀ x < -3, ∀ y < -3, x < y → f x < f y) ∧
  (∀ x ∈ Set.Ioo (-3) 1, ∀ y ∈ Set.Ioo (-3) 1, x < y → f x > f y) ∧
  (∀ x > 1, ∀ y > 1, x < y → f x < f y) ∧
  -- Part 2: Minimum value condition
  (∀ c : ℝ, (∀ x ∈ Set.Icc (-4) c, f x ≥ -5) ∧ (∃ x ∈ Set.Icc (-4) c, f x = -5) ↔ c ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3990_399072


namespace NUMINAMATH_CALUDE_teresa_black_pencils_l3990_399078

/-- Given Teresa's pencil distribution problem, prove she has 35 black pencils. -/
theorem teresa_black_pencils : 
  (colored_pencils : ℕ) →
  (siblings : ℕ) →
  (pencils_per_sibling : ℕ) →
  (pencils_kept : ℕ) →
  colored_pencils = 14 →
  siblings = 3 →
  pencils_per_sibling = 13 →
  pencils_kept = 10 →
  (siblings * pencils_per_sibling + pencils_kept) - colored_pencils = 35 := by
sorry

end NUMINAMATH_CALUDE_teresa_black_pencils_l3990_399078


namespace NUMINAMATH_CALUDE_parallelogram_angles_l3990_399073

/-- Represents the angles of a parallelogram -/
structure ParallelogramAngles where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- Properties of a parallelogram with one angle 50° less than the other -/
def is_valid_parallelogram (p : ParallelogramAngles) : Prop :=
  p.angle1 = p.angle3 ∧
  p.angle2 = p.angle4 ∧
  p.angle1 + p.angle2 = 180 ∧
  p.angle2 = p.angle1 + 50

/-- Theorem: The angles of a parallelogram with one angle 50° less than the other are 65°, 115°, 65°, and 115° -/
theorem parallelogram_angles :
  ∃ (p : ParallelogramAngles), is_valid_parallelogram p ∧
    p.angle1 = 65 ∧ p.angle2 = 115 ∧ p.angle3 = 65 ∧ p.angle4 = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angles_l3990_399073


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3990_399097

theorem sqrt_equation_solution : ∃ x : ℝ, x = 225 / 16 ∧ Real.sqrt x + Real.sqrt (x + 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3990_399097


namespace NUMINAMATH_CALUDE_parabola_coefficient_l3990_399034

/-- Given a parabola y = ax^2 + bx + c with vertex (q, 2q) and y-intercept (0, -3q), where q ≠ 0, 
    the value of b is 10. -/
theorem parabola_coefficient (a b c q : ℝ) : q ≠ 0 →
  (∀ x y, y = a * x^2 + b * x + c ↔ 
    (x = q ∧ y = 2*q) ∨ (x = 0 ∧ y = -3*q)) →
  b = 10 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l3990_399034


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l3990_399018

theorem min_value_sqrt_sum_squares (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 3) :
  Real.sqrt (a^2 + b^2 + c^2) ≥ Real.sqrt 3 ∧ 
  (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l3990_399018


namespace NUMINAMATH_CALUDE_boys_score_in_class_l3990_399001

theorem boys_score_in_class (boy_percentage : ℝ) (girl_percentage : ℝ) 
  (girl_score : ℝ) (class_average : ℝ) : 
  boy_percentage = 40 →
  girl_percentage = 100 - boy_percentage →
  girl_score = 90 →
  class_average = 86 →
  (boy_percentage * boy_score + girl_percentage * girl_score) / 100 = class_average →
  boy_score = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_boys_score_in_class_l3990_399001


namespace NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3990_399027

/-- Given a two-digit number where the difference between the original number
    and the number with interchanged digits is 45, prove that the difference
    between its two digits is 5. -/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 45 → x - y = 5 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_digit_difference_l3990_399027


namespace NUMINAMATH_CALUDE_translated_line_y_intercept_l3990_399033

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Translates a line horizontally and vertically -/
def translateLine (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    yIntercept := l.yIntercept - dy + l.slope * dx }

/-- The original line y = x -/
def originalLine : Line :=
  { slope := 1,
    yIntercept := 0 }

/-- The translated line -/
def translatedLine : Line :=
  translateLine originalLine 3 (-2)

theorem translated_line_y_intercept :
  translatedLine.yIntercept = -5 := by
  sorry

end NUMINAMATH_CALUDE_translated_line_y_intercept_l3990_399033


namespace NUMINAMATH_CALUDE_derivative_of_exp_neg_x_l3990_399056

theorem derivative_of_exp_neg_x (x : ℝ) : deriv (fun x => Real.exp (-x)) x = -Real.exp (-x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_neg_x_l3990_399056


namespace NUMINAMATH_CALUDE_max_third_term_is_16_l3990_399070

/-- An arithmetic sequence of four positive integers -/
structure ArithmeticSequence :=
  (a : ℕ+) -- First term
  (d : ℕ+) -- Common difference
  (sum_eq_50 : a + (a + d) + (a + 2*d) + (a + 3*d) = 50)
  (third_term_even : Even (a + 2*d))

/-- The third term of an arithmetic sequence -/
def third_term (seq : ArithmeticSequence) : ℕ := seq.a + 2*seq.d

/-- Theorem: The maximum possible value for the third term is 16 -/
theorem max_third_term_is_16 :
  ∀ seq : ArithmeticSequence, third_term seq ≤ 16 :=
sorry

end NUMINAMATH_CALUDE_max_third_term_is_16_l3990_399070


namespace NUMINAMATH_CALUDE_equation_solution_l3990_399059

theorem equation_solution (a : ℝ) (x : ℝ) : 
  a = 3 → (5 * a - x = 13 ↔ x = 2) := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3990_399059


namespace NUMINAMATH_CALUDE_password_config_exists_l3990_399082

/-- A password configuration is represented by a list of integers, 
    where each integer represents the count of a distinct character. -/
def PasswordConfig := List Nat

/-- The number of combinations for a given password configuration -/
def numCombinations (config : PasswordConfig) : Nat :=
  Nat.factorial 5 / (config.map Nat.factorial).prod

/-- Theorem: There exists a 5-character password configuration 
    that results in exactly 20 different combinations -/
theorem password_config_exists : ∃ (config : PasswordConfig), 
  config.sum = 5 ∧ numCombinations config = 20 := by
  sorry

end NUMINAMATH_CALUDE_password_config_exists_l3990_399082


namespace NUMINAMATH_CALUDE_total_amount_spent_l3990_399062

def num_pens : ℕ := 30
def num_pencils : ℕ := 75
def avg_price_pencil : ℚ := 2
def avg_price_pen : ℚ := 16

theorem total_amount_spent : 
  num_pens * avg_price_pen + num_pencils * avg_price_pencil = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_amount_spent_l3990_399062


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3990_399055

/-- Represents a hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- Checks if a point lies on the hyperbola -/
def Hyperbola.contains (h : Hyperbola) (x y : ℝ) : Prop :=
  h.equation x y

/-- Checks if a line is an asymptote of the hyperbola -/
def Hyperbola.hasAsymptote (h : Hyperbola) (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x y, h.contains x y → |y - f x| < ε ∨ |x| > δ

theorem hyperbola_equation (h : Hyperbola) :
  (h.hasAsymptote (fun x ↦ 2 * x) ∧ h.hasAsymptote (fun x ↦ -2 * x)) →
  h.contains 1 (2 * Real.sqrt 5) →
  h.equation = fun x y ↦ y^2 / 16 - x^2 / 4 = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3990_399055


namespace NUMINAMATH_CALUDE_course_failure_implies_question_failure_l3990_399011

-- Define the universe of students
variable (Student : Type)

-- Define predicates
variable (passed_course : Student → Prop)
variable (failed_no_questions : Student → Prop)

-- Ms. Johnson's statement
variable (johnsons_statement : ∀ s : Student, failed_no_questions s → passed_course s)

-- Theorem to prove
theorem course_failure_implies_question_failure :
  ∀ s : Student, ¬(passed_course s) → ¬(failed_no_questions s) :=
by sorry

-- Note: The proof is omitted as per instructions

end NUMINAMATH_CALUDE_course_failure_implies_question_failure_l3990_399011


namespace NUMINAMATH_CALUDE_polynomial_modulus_bound_l3990_399089

theorem polynomial_modulus_bound (a b c d : ℂ) 
  (ha : Complex.abs a = 1) (hb : Complex.abs b = 1) 
  (hc : Complex.abs c = 1) (hd : Complex.abs d = 1) : 
  ∃ z : ℂ, Complex.abs z = 1 ∧ 
    Complex.abs (a * z^3 + b * z^2 + c * z + d) ≥ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_modulus_bound_l3990_399089


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l3990_399008

theorem xiaolis_estimate (x y : ℝ) (h1 : x > y) (h2 : y > 0) :
  (2 * x - y) / 2 > x - y := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l3990_399008


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l3990_399044

theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 960) 
  (h2 : num_moles = 5) : 
  total_weight / num_moles = 192 := by
sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l3990_399044


namespace NUMINAMATH_CALUDE_exists_square_farther_than_V_l3990_399016

/-- Represents a square on the board --/
structure Square where
  x : Fin 19
  y : Fin 19

/-- Defines the movement of the dragon --/
def dragonMove (s : Square) : Set Square :=
  { t | (t.x = s.x + 4 ∧ t.y = s.y + 1) ∨
        (t.x = s.x + 4 ∧ t.y = s.y - 1) ∨
        (t.x = s.x - 4 ∧ t.y = s.y + 1) ∨
        (t.x = s.x - 4 ∧ t.y = s.y - 1) ∨
        (t.x = s.x + 1 ∧ t.y = s.y + 4) ∨
        (t.x = s.x + 1 ∧ t.y = s.y - 4) ∨
        (t.x = s.x - 1 ∧ t.y = s.y + 4) ∨
        (t.x = s.x - 1 ∧ t.y = s.y - 4) }

/-- Draconian distance between two squares --/
def draconianDistance (s t : Square) : ℕ :=
  sorry

/-- Corner square --/
def C : Square :=
  { x := 0, y := 0 }

/-- Diagonally adjacent square to C --/
def V : Square :=
  { x := 1, y := 1 }

/-- Main theorem --/
theorem exists_square_farther_than_V :
  ∃ X : Square, draconianDistance C X > draconianDistance C V :=
sorry

end NUMINAMATH_CALUDE_exists_square_farther_than_V_l3990_399016


namespace NUMINAMATH_CALUDE_billy_already_ahead_l3990_399037

def billy_miles : List ℝ := [2, 3, 0, 4, 1, 0]
def tiffany_miles : List ℝ := [1.5, 0, 2.5, 2.5, 3, 0]

theorem billy_already_ahead : 
  (billy_miles.sum > tiffany_miles.sum) ∧ 
  (billy_miles.length = tiffany_miles.length) := by
  sorry

end NUMINAMATH_CALUDE_billy_already_ahead_l3990_399037


namespace NUMINAMATH_CALUDE_line_slope_l3990_399029

/-- A line that returns to its original position after moving 4 units left and 1 unit up has a slope of -1/4 -/
theorem line_slope (l : ℝ → ℝ) (b : ℝ) (h : ∀ x, l x = l (x + 4) - 1) : 
  ∃ k, k = -1/4 ∧ ∀ x, l x = k * x + b := by
sorry

end NUMINAMATH_CALUDE_line_slope_l3990_399029


namespace NUMINAMATH_CALUDE_binomial_7_4_l3990_399075

theorem binomial_7_4 : Nat.choose 7 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_4_l3990_399075


namespace NUMINAMATH_CALUDE_events_B_C_complementary_l3990_399015

-- Define the sample space (faces of the cube)
def S : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define events A, B, and C
def A : Set Nat := {n ∈ S | n % 2 = 1}
def B : Set Nat := {n ∈ S | n ≤ 3}
def C : Set Nat := {n ∈ S | n ≥ 4}

-- Theorem statement
theorem events_B_C_complementary : B ∪ C = S ∧ B ∩ C = ∅ :=
sorry

end NUMINAMATH_CALUDE_events_B_C_complementary_l3990_399015
