import Mathlib

namespace NUMINAMATH_CALUDE_tan_240_degrees_l2844_284457

theorem tan_240_degrees : Real.tan (240 * π / 180) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_240_degrees_l2844_284457


namespace NUMINAMATH_CALUDE_bird_migration_difference_l2844_284498

theorem bird_migration_difference (migrating_families : ℕ) (remaining_families : ℕ)
  (avg_birds_migrating : ℕ) (avg_birds_remaining : ℕ)
  (h1 : migrating_families = 86)
  (h2 : remaining_families = 45)
  (h3 : avg_birds_migrating = 12)
  (h4 : avg_birds_remaining = 8) :
  migrating_families * avg_birds_migrating - remaining_families * avg_birds_remaining = 672 := by
  sorry

end NUMINAMATH_CALUDE_bird_migration_difference_l2844_284498


namespace NUMINAMATH_CALUDE_parabola_minimum_distance_sum_l2844_284441

/-- The minimum distance sum from a point on the parabola y² = 4x to two fixed points -/
theorem parabola_minimum_distance_sum :
  ∀ x y : ℝ,
  y^2 = 4*x →
  (∀ x' y' : ℝ, y'^2 = 4*x' →
    Real.sqrt ((x - 2)^2 + (y - 1)^2) + Real.sqrt ((x - 1)^2 + y^2) ≤
    Real.sqrt ((x' - 2)^2 + (y' - 1)^2) + Real.sqrt ((x' - 1)^2 + y'^2)) →
  Real.sqrt ((x - 2)^2 + (y - 1)^2) + Real.sqrt ((x - 1)^2 + y^2) = 3 :=
by sorry

end NUMINAMATH_CALUDE_parabola_minimum_distance_sum_l2844_284441


namespace NUMINAMATH_CALUDE_circle_equation_correct_l2844_284488

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the equation of a circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation_correct :
  let c : Circle := { center := (-1, 3), radius := 2 }
  ∀ x y : ℝ, circleEquation c x y ↔ (x + 1)^2 + (y - 3)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_correct_l2844_284488


namespace NUMINAMATH_CALUDE_inner_quadrilateral_area_ratio_l2844_284420

-- Define the quadrilateral type
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the area function for quadrilaterals
noncomputable def area (q : Quadrilateral) : ℝ := sorry

-- Define the function to get points on the sides of a quadrilateral
def getInnerPoints (q : Quadrilateral) (p : ℝ) : Quadrilateral := sorry

-- Main theorem
theorem inner_quadrilateral_area_ratio 
  (ABCD : Quadrilateral) (p : ℝ) (h : p < 0.5) :
  let A₁B₁C₁D₁ := getInnerPoints ABCD p
  area A₁B₁C₁D₁ / area ABCD = 1 - 2 * p := by sorry

end NUMINAMATH_CALUDE_inner_quadrilateral_area_ratio_l2844_284420


namespace NUMINAMATH_CALUDE_distributor_cost_distributor_cost_proof_l2844_284450

/-- The cost of an item for a distributor given online store commission, desired profit, and observed price. -/
theorem distributor_cost (commission_rate : ℝ) (profit_rate : ℝ) (observed_price : ℝ) : ℝ :=
  let selling_price := observed_price / (1 - commission_rate)
  let cost := selling_price / (1 + profit_rate)
  cost

/-- Proof that the distributor's cost is $28.125 given the specified conditions. -/
theorem distributor_cost_proof :
  distributor_cost 0.2 0.2 27 = 28.125 :=
by sorry

end NUMINAMATH_CALUDE_distributor_cost_distributor_cost_proof_l2844_284450


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2844_284445

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l2844_284445


namespace NUMINAMATH_CALUDE_count_with_four_or_five_l2844_284429

/-- The number of integers from 1 to 343 (inclusive) in base 7 that do not contain 4 or 5 as a digit -/
def count_without_four_or_five : ℕ := 125

/-- The total number of integers from 1 to 343 in base 7 -/
def total_count : ℕ := 343

theorem count_with_four_or_five :
  total_count - count_without_four_or_five = 218 :=
sorry

end NUMINAMATH_CALUDE_count_with_four_or_five_l2844_284429


namespace NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l2844_284474

/-- The number of sides in a dodecagon -/
def dodecagon_sides : ℕ := 12

/-- The sum of exterior angles of any polygon in degrees -/
def sum_exterior_angles : ℝ := 360

/-- Theorem: The sum of the exterior angles of a regular dodecagon is 360° -/
theorem sum_exterior_angles_dodecagon :
  sum_exterior_angles = 360 := by sorry

end NUMINAMATH_CALUDE_sum_exterior_angles_dodecagon_l2844_284474


namespace NUMINAMATH_CALUDE_triangle_side_length_l2844_284478

/-- Triangle DEF with given properties -/
structure Triangle where
  D : ℝ  -- Angle D
  E : ℝ  -- Angle E
  F : ℝ  -- Angle F
  d : ℝ  -- Side length opposite to angle D
  e : ℝ  -- Side length opposite to angle E
  f : ℝ  -- Side length opposite to angle F

/-- The theorem stating the properties of the triangle and the value of f -/
theorem triangle_side_length 
  (t : Triangle)
  (h1 : t.d = 7)
  (h2 : t.e = 3)
  (h3 : Real.cos (t.D - t.E) = 7/8) :
  t.f = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2844_284478


namespace NUMINAMATH_CALUDE_cubic_equation_game_strategy_l2844_284499

theorem cubic_equation_game_strategy (second_player_choice : ℤ) : 
  ∃ (a b c : ℤ), ∃ (x y z : ℤ),
    (x^3 + a*x^2 + b*x + c = 0) ∧
    (y^3 + a*y^2 + b*y + c = 0) ∧
    (z^3 + a*z^2 + b*z + c = 0) ∧
    (a = second_player_choice ∨ b = second_player_choice) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_game_strategy_l2844_284499


namespace NUMINAMATH_CALUDE_retired_staff_samples_calculation_l2844_284479

/-- Calculates the number of samples for retired staff given total samples and ratio -/
def retired_staff_samples (total_samples : ℕ) (retired_ratio current_ratio student_ratio : ℕ) : ℕ :=
  let total_ratio := retired_ratio + current_ratio + student_ratio
  let unit_value := total_samples / total_ratio
  retired_ratio * unit_value

/-- Theorem stating that given 300 total samples and a ratio of 3:7:40, 
    the number of samples from retired staff is 18 -/
theorem retired_staff_samples_calculation :
  retired_staff_samples 300 3 7 40 = 18 := by
  sorry

end NUMINAMATH_CALUDE_retired_staff_samples_calculation_l2844_284479


namespace NUMINAMATH_CALUDE_max_divisor_of_f_l2844_284446

def f (n : ℕ) : ℕ := (2 * n + 7) * 3^n + 9

theorem max_divisor_of_f :
  ∃ (m : ℕ), (∀ (n : ℕ), m ∣ f n) ∧ 
  (∀ (k : ℕ), (∀ (n : ℕ), k ∣ f n) → k ≤ 36) ∧
  (∀ (n : ℕ), 36 ∣ f n) :=
sorry

end NUMINAMATH_CALUDE_max_divisor_of_f_l2844_284446


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l2844_284490

theorem triangle_side_lengths 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h_area : (1/2) * a * b * Real.sin C = 10 * Real.sqrt 3)
  (h_sum : a + b = 13)
  (h_angle : C = π/3) :
  ((a = 5 ∧ b = 8 ∧ c = 7) ∨ (a = 8 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l2844_284490


namespace NUMINAMATH_CALUDE_min_disks_needed_l2844_284456

def total_files : ℕ := 45
def disk_capacity : ℚ := 1.44

def file_size_1 : ℚ := 0.9
def file_count_1 : ℕ := 5

def file_size_2 : ℚ := 0.6
def file_count_2 : ℕ := 15

def file_size_3 : ℚ := 0.5
def file_count_3 : ℕ := total_files - file_count_1 - file_count_2

theorem min_disks_needed : 
  ∃ (n : ℕ), n = 20 ∧ 
  (∀ m : ℕ, m < n → 
    m * disk_capacity < 
      file_count_1 * file_size_1 + 
      file_count_2 * file_size_2 + 
      file_count_3 * file_size_3) ∧
  n * disk_capacity ≥ 
    file_count_1 * file_size_1 + 
    file_count_2 * file_size_2 + 
    file_count_3 * file_size_3 :=
by sorry

end NUMINAMATH_CALUDE_min_disks_needed_l2844_284456


namespace NUMINAMATH_CALUDE_decimal_operations_l2844_284458

theorem decimal_operations (x y : ℝ) : 
  (x / 10 = 0.09 → x = 0.9) ∧ 
  (3.24 * y = 3240 → y = 1000) := by
  sorry

end NUMINAMATH_CALUDE_decimal_operations_l2844_284458


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2844_284434

theorem quadratic_real_roots (p1 p2 q1 q2 : ℝ) 
  (h : p1 * p2 = 2 * (q1 + q2)) : 
  ∃ x : ℝ, (x^2 + p1*x + q1 = 0) ∨ (x^2 + p2*x + q2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2844_284434


namespace NUMINAMATH_CALUDE_inverse_of_inverse_f_l2844_284460

-- Define the original function f
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the inverse of f^(-1)(x+1)
def g (x : ℝ) : ℝ := 2 * x + 2

-- Theorem statement
theorem inverse_of_inverse_f (x : ℝ) : 
  g (f⁻¹ (x + 1)) = x ∧ f⁻¹ (g x + 1) = x := by
  sorry


end NUMINAMATH_CALUDE_inverse_of_inverse_f_l2844_284460


namespace NUMINAMATH_CALUDE_toad_ratio_is_25_to_1_l2844_284405

/-- Represents the number of toads per acre -/
structure ToadPopulation where
  green : ℕ
  brown : ℕ
  spotted_brown : ℕ

/-- The ratio of brown toads to green toads -/
def brown_to_green_ratio (pop : ToadPopulation) : ℚ :=
  pop.brown / pop.green

theorem toad_ratio_is_25_to_1 (pop : ToadPopulation) 
  (h1 : pop.green = 8)
  (h2 : pop.spotted_brown = 50)
  (h3 : pop.spotted_brown * 4 = pop.brown) : 
  brown_to_green_ratio pop = 25 := by
  sorry

end NUMINAMATH_CALUDE_toad_ratio_is_25_to_1_l2844_284405


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2844_284482

theorem unique_quadratic_solution (m : ℝ) : 
  (∃! x : ℝ, m * x^2 + 2 * x - 1 = 0) → (m = 0 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2844_284482


namespace NUMINAMATH_CALUDE_reservoir_duration_l2844_284409

theorem reservoir_duration (x y z : ℝ) 
  (h1 : 40 * (y - x) = z)
  (h2 : 40 * (1.1 * y - 1.2 * x) = z)
  : z / (y - 1.2 * x) = 50 := by
  sorry

end NUMINAMATH_CALUDE_reservoir_duration_l2844_284409


namespace NUMINAMATH_CALUDE_solution_pairs_l2844_284421

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def satisfies_conditions (a b : ℕ) : Prop :=
  (is_divisible_by (a - b) 3 ∨
   is_prime (a + 2*b) ∧
   a = 4*b - 1 ∧
   is_divisible_by (a + 7) b) ∧
  (¬(is_divisible_by (a - b) 3) ∨
   ¬(is_prime (a + 2*b)) ∨
   ¬(a = 4*b - 1) ∨
   ¬(is_divisible_by (a + 7) b))

theorem solution_pairs :
  ∀ a b : ℕ, satisfies_conditions a b ↔ (a = 3 ∧ b = 1) ∨ (a = 7 ∧ b = 2) ∨ (a = 11 ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_solution_pairs_l2844_284421


namespace NUMINAMATH_CALUDE_triangle_inequality_range_l2844_284468

theorem triangle_inequality_range (x : ℝ) : 
  (3 : ℝ) > 0 ∧ (1 + 2*x) > 0 ∧ 8 > 0 ∧
  3 + (1 + 2*x) > 8 ∧
  3 + 8 > (1 + 2*x) ∧
  (1 + 2*x) + 8 > 3 ↔
  2 < x ∧ x < 5 := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_range_l2844_284468


namespace NUMINAMATH_CALUDE_cube_painting_cost_l2844_284496

/-- The cost to paint a cube given paint cost, coverage, and cube dimensions -/
theorem cube_painting_cost 
  (paint_cost : ℝ) 
  (paint_coverage : ℝ) 
  (cube_side : ℝ) 
  (h1 : paint_cost = 40) 
  (h2 : paint_coverage = 20) 
  (h3 : cube_side = 10) : 
  paint_cost * (6 * cube_side^2 / paint_coverage) = 1200 := by
  sorry

#check cube_painting_cost

end NUMINAMATH_CALUDE_cube_painting_cost_l2844_284496


namespace NUMINAMATH_CALUDE_system_is_linear_l2844_284461

/-- A linear equation in two variables -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The specific system of equations we want to prove is linear -/
def system : LinearSystem := {
  eq1 := { a := 1, b := 0, c := 1 }  -- Represents x = 1
  eq2 := { a := 3, b := -2, c := 6 } -- Represents 3x - 2y = 6
}

/-- Theorem stating that our system is indeed a system of two linear equations -/
theorem system_is_linear : ∃ (s : LinearSystem), s = system := by sorry

end NUMINAMATH_CALUDE_system_is_linear_l2844_284461


namespace NUMINAMATH_CALUDE_ellipse_equation_l2844_284401

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given an ellipse E with specific properties, prove its equation -/
theorem ellipse_equation (E : Ellipse) (F A B M : Point) :
  E.a > E.b ∧ E.b > 0 ∧  -- a > b > 0
  F = ⟨3, 0⟩ ∧  -- Right focus at F(3,0)
  (A.y - F.y) / (A.x - F.x) = 1/2 ∧  -- Line through F with slope 1/2
  (B.y - F.y) / (B.x - F.x) = 1/2 ∧  -- intersects E at A and B
  M = ⟨1, -1⟩ ∧  -- Midpoint of AB is (1,-1)
  M.x = (A.x + B.x) / 2 ∧
  M.y = (A.y + B.y) / 2 ∧
  (A.x^2 / E.a^2) + (A.y^2 / E.b^2) = 1 ∧  -- A and B lie on the ellipse
  (B.x^2 / E.a^2) + (B.y^2 / E.b^2) = 1 →
  E.a^2 = 18 ∧ E.b^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2844_284401


namespace NUMINAMATH_CALUDE_right_triangle_sin_q_l2844_284454

/-- In a right triangle PQR with angle R = 90° and 3sin Q = 4cos Q, sin Q = 4/5 -/
theorem right_triangle_sin_q (Q : Real) (h1 : 3 * Real.sin Q = 4 * Real.cos Q) : Real.sin Q = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_q_l2844_284454


namespace NUMINAMATH_CALUDE_last_three_digits_of_5_to_15000_l2844_284485

theorem last_three_digits_of_5_to_15000 (h : 5^500 ≡ 1 [ZMOD 1000]) :
  5^15000 ≡ 1 [ZMOD 1000] := by
  sorry

end NUMINAMATH_CALUDE_last_three_digits_of_5_to_15000_l2844_284485


namespace NUMINAMATH_CALUDE_max_value_implies_ratio_l2844_284437

/-- The function f(x) = x³ + ax² + bx - a² - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- f(x) reaches a maximum value of 10 at x = 1 -/
def max_at_one (a b : ℝ) : Prop :=
  (∀ x, f a b x ≤ f a b 1) ∧ f a b 1 = 10

theorem max_value_implies_ratio (a b : ℝ) (h : max_at_one a b) : a / b = -2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_ratio_l2844_284437


namespace NUMINAMATH_CALUDE_log_expression_simplification_l2844_284425

theorem log_expression_simplification :
  (1/2) * Real.log (32/49) - (4/3) * Real.log (Real.sqrt 8) + Real.log (Real.sqrt 245) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_simplification_l2844_284425


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l2844_284411

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 6 + 3 = 6) → initial_people = 9 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l2844_284411


namespace NUMINAMATH_CALUDE_last_digit_of_4139_power_467_l2844_284447

theorem last_digit_of_4139_power_467 : (4139^467) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_4139_power_467_l2844_284447


namespace NUMINAMATH_CALUDE_assign_four_providers_from_twentyfive_l2844_284471

/-- The number of ways to assign different service providers to children -/
def assignProviders (totalProviders : ℕ) (children : ℕ) : ℕ :=
  (List.range children).foldl (fun acc i => acc * (totalProviders - i)) 1

/-- Theorem: Assigning 4 different service providers to 4 children from 25 providers -/
theorem assign_four_providers_from_twentyfive :
  assignProviders 25 4 = 303600 := by
  sorry

end NUMINAMATH_CALUDE_assign_four_providers_from_twentyfive_l2844_284471


namespace NUMINAMATH_CALUDE_problem_solution_l2844_284410

def f (x : ℝ) := |2*x - 3| + |2*x + 3|

theorem problem_solution :
  (∃ (M : ℝ),
    (∀ x, f x ≥ M) ∧
    (∃ x, f x = M) ∧
    (M = 6)) ∧
  ({x : ℝ | f x ≤ 8} = {x : ℝ | -2 ≤ x ∧ x ≤ 2}) ∧
  (∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    1/a + 1/(2*b) + 1/(3*c) = 1 →
    a + 2*b + 3*c ≥ 9) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2844_284410


namespace NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_805_l2844_284459

/-- Calculate the total interest after 10 years given the conditions -/
theorem total_interest_calculation (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 700) 
  (h2 : P * R = 700) : 
  P * R * 5 / 100 + 3 * P * R * 5 / 100 = 805 := by
  sorry

/-- Prove that the total interest is 805 -/
theorem total_interest_is_805 (P R : ℝ) 
  (h1 : P * R * 10 / 100 = 700) 
  (h2 : P * R = 700) : 
  ∃ (total_interest : ℝ), total_interest = 805 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_calculation_total_interest_is_805_l2844_284459


namespace NUMINAMATH_CALUDE_factorization_equality_l2844_284400

theorem factorization_equality (a b x y : ℝ) :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2844_284400


namespace NUMINAMATH_CALUDE_rachel_coloring_books_l2844_284480

/-- The number of pictures remaining to be colored given the number of pictures in three coloring books and the number of pictures already colored. -/
def remaining_pictures (book1 book2 book3 colored : ℕ) : ℕ :=
  book1 + book2 + book3 - colored

/-- Theorem stating that given the specific numbers from the problem, the remaining pictures to be colored is 56. -/
theorem rachel_coloring_books : remaining_pictures 23 32 45 44 = 56 := by
  sorry

end NUMINAMATH_CALUDE_rachel_coloring_books_l2844_284480


namespace NUMINAMATH_CALUDE_complex_modulus_squared_l2844_284433

theorem complex_modulus_squared (z : ℂ) (h1 : z + Complex.abs z = 6 + 2*I) 
  (h2 : z.re ≥ 0) : Complex.abs z ^ 2 = 100 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_squared_l2844_284433


namespace NUMINAMATH_CALUDE_diamond_equation_solution_l2844_284489

-- Define the diamond operation
def diamond (a b : ℝ) : ℝ := 3 * a * b - a + b

-- Theorem statement
theorem diamond_equation_solution :
  ∃ x : ℝ, diamond 3 x = 24 ∧ x = 2.7 := by
  sorry

end NUMINAMATH_CALUDE_diamond_equation_solution_l2844_284489


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_14_l2844_284442

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ :=
  sorry

/-- A function that returns the nth positive integer whose digits sum to 14 -/
def nthNumberWithDigitSum14 (n : ℕ+) : ℕ+ :=
  sorry

/-- The theorem stating that the 11th number with digit sum 14 is 194 -/
theorem eleventh_number_with_digit_sum_14 : 
  nthNumberWithDigitSum14 11 = 194 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_14_l2844_284442


namespace NUMINAMATH_CALUDE_tan_alpha_values_l2844_284436

theorem tan_alpha_values (α : Real) 
  (h : 2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + 5 * Real.cos α ^ 2 = 3) : 
  Real.tan α = 1 ∨ Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l2844_284436


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l2844_284476

theorem quadratic_two_roots 
  (a b c α : ℝ) 
  (f : ℝ → ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c) 
  (h_exists : a * f α < 0) : 
  ∃ x₁ x₂, x₁ < α ∧ α < x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l2844_284476


namespace NUMINAMATH_CALUDE_no_valid_tournament_l2844_284466

/-- Represents a round-robin chess tournament -/
structure ChessTournament where
  num_players : ℕ
  wins : Fin num_players → ℕ
  draws : Fin num_players → ℕ
  losses : Fin num_players → ℕ

/-- Definition of a valid round-robin tournament -/
def is_valid_tournament (t : ChessTournament) : Prop :=
  t.num_players = 20 ∧
  ∀ i : Fin t.num_players, 
    t.wins i + t.draws i + t.losses i = t.num_players - 1 ∧
    t.wins i = t.draws i

/-- Theorem stating that a valid tournament as described is impossible -/
theorem no_valid_tournament : ¬∃ t : ChessTournament, is_valid_tournament t := by
  sorry


end NUMINAMATH_CALUDE_no_valid_tournament_l2844_284466


namespace NUMINAMATH_CALUDE_statement_A_necessary_not_sufficient_l2844_284408

theorem statement_A_necessary_not_sufficient :
  (∀ x y : ℝ, (x + y ≠ 5 → (x ≠ 2 ∨ y ≠ 3))) ∧
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ (x + y = 5)) := by
  sorry

end NUMINAMATH_CALUDE_statement_A_necessary_not_sufficient_l2844_284408


namespace NUMINAMATH_CALUDE_equation_has_one_solution_l2844_284418

theorem equation_has_one_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_has_one_solution_l2844_284418


namespace NUMINAMATH_CALUDE_pet_shop_dogs_l2844_284486

theorem pet_shop_dogs (dogs cats bunnies : ℕ) : 
  dogs + cats + bunnies > 0 →
  dogs * 7 = cats * 4 →
  dogs * 9 = bunnies * 4 →
  dogs + bunnies = 364 →
  dogs = 112 := by
sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_l2844_284486


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2844_284475

theorem unique_solution_to_equation :
  ∃! x : ℝ, x > 0 ∧ x^(Real.log x / Real.log 5) = x^2 / 25 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2844_284475


namespace NUMINAMATH_CALUDE_prob_and_arrangements_correct_l2844_284494

/-- The number of class officers -/
def total_officers : Nat := 6

/-- The number of boys among the officers -/
def num_boys : Nat := 3

/-- The number of girls among the officers -/
def num_girls : Nat := 3

/-- The number of people selected for voluntary labor -/
def num_selected : Nat := 3

/-- The probability of selecting at least 2 girls out of 3 people from a group of 3 boys and 3 girls -/
def prob_at_least_two_girls : ℚ := 1/2

/-- The number of ways to arrange 6 people (3 boys and 3 girls) in a row, 
    where one boy must be at an end and two specific girls must be together -/
def num_arrangements : Nat := 96

theorem prob_and_arrangements_correct : 
  (total_officers = num_boys + num_girls) →
  (prob_at_least_two_girls = 1/2) ∧ 
  (num_arrangements = 96) := by sorry

end NUMINAMATH_CALUDE_prob_and_arrangements_correct_l2844_284494


namespace NUMINAMATH_CALUDE_torus_division_theorem_l2844_284444

/-- Represents a torus surface -/
structure TorusSurface where
  -- Add necessary fields here

/-- Represents a path on the torus surface -/
structure PathOnTorus where
  -- Add necessary fields here

/-- Represents the outer equator of the torus -/
def outerEquator : PathOnTorus :=
  sorry

/-- Represents a helical line on the torus -/
def helicalLine : PathOnTorus :=
  sorry

/-- Counts the number of regions a torus surface is divided into when cut along given paths -/
def countRegions (surface : TorusSurface) (path1 path2 : PathOnTorus) : ℕ :=
  sorry

/-- Theorem stating that cutting a torus along its outer equator and a helical line divides it into 3 parts -/
theorem torus_division_theorem (surface : TorusSurface) :
  countRegions surface outerEquator helicalLine = 3 :=
sorry

end NUMINAMATH_CALUDE_torus_division_theorem_l2844_284444


namespace NUMINAMATH_CALUDE_min_n_for_inequality_l2844_284493

theorem min_n_for_inequality : 
  ∃ (n : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ n * (x^4 + y^4 + z^4)) ∧ 
  (∀ (m : ℕ), (∀ (x y z : ℝ), x^2 + y^2 + z^2 ≤ m * (x^4 + y^4 + z^4)) → n ≤ m) ∧
  n = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_n_for_inequality_l2844_284493


namespace NUMINAMATH_CALUDE_triangle_tangency_points_l2844_284428

theorem triangle_tangency_points (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) →  -- positive sides
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- triangle inequality
  (∃ (M R : ℝ), 
    0 < M ∧ M < c ∧ 
    0 < R ∧ R < c ∧
    M ≠ R ∧
    M / c = 1 / 3 ∧ 
    (c - R) / c = 1 / 3 ∧
    (R - M) / c = 1 / 3) →  -- points divide c into three equal parts
  c = 3 * abs (a - b) ∧ 
  ((b < a ∧ a < 2 * b) ∨ (a < b ∧ b < 2 * a)) := by
sorry

end NUMINAMATH_CALUDE_triangle_tangency_points_l2844_284428


namespace NUMINAMATH_CALUDE_fish_length_difference_l2844_284419

theorem fish_length_difference : 
  let first_fish_length : Real := 0.3
  let second_fish_length : Real := 0.2
  first_fish_length - second_fish_length = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_fish_length_difference_l2844_284419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2844_284463

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)
  (h_arith : ArithmeticSequence a)
  (h_cond1 : a 7 - 2 * a 4 = -1)
  (h_cond2 : a 3 = 0) :
  ∃ d : ℚ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -1/2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2844_284463


namespace NUMINAMATH_CALUDE_simplify_complex_expression_l2844_284469

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Theorem stating that the simplification of 4(2-i) + 2i(3-2i) is 12 + 2i -/
theorem simplify_complex_expression : 4 * (2 - i) + 2 * i * (3 - 2 * i) = 12 + 2 * i :=
by sorry

end NUMINAMATH_CALUDE_simplify_complex_expression_l2844_284469


namespace NUMINAMATH_CALUDE_largest_digit_sum_l2844_284462

theorem largest_digit_sum (a b c z : ℕ) : 
  (a < 10 ∧ b < 10 ∧ c < 10) →  -- a, b, c are digits
  (100 * a + 10 * b + c = 1000 / z) →  -- 0.abc = 1/z
  (0 < z ∧ z ≤ 12) →  -- 0 < z ≤ 12
  (∃ (x y w : ℕ), x + y + w ≤ 8 ∧ 
    (100 * x + 10 * y + w = 1000 / z) ∧ 
    (x < 10 ∧ y < 10 ∧ w < 10)) →
  a + b + c ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_largest_digit_sum_l2844_284462


namespace NUMINAMATH_CALUDE_prob_one_of_A_or_B_is_two_thirds_l2844_284403

/-- The number of study groups -/
def num_groups : ℕ := 4

/-- The number of groups to be selected -/
def num_selected : ℕ := 2

/-- The probability of selecting exactly one of group A and group B -/
def prob_one_of_A_or_B : ℚ := 2/3

/-- Theorem stating that the probability of selecting exactly one of group A and group B
    when randomly selecting two groups out of four groups is 2/3 -/
theorem prob_one_of_A_or_B_is_two_thirds :
  prob_one_of_A_or_B = (num_groups - 2) / (Nat.choose num_groups num_selected) := by
  sorry

end NUMINAMATH_CALUDE_prob_one_of_A_or_B_is_two_thirds_l2844_284403


namespace NUMINAMATH_CALUDE_integral_equation_solution_l2844_284427

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0:ℝ)..2, (3 * x^2 + k)) = 16 → k = 4 := by
sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l2844_284427


namespace NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2844_284495

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 28) :
  let side_length := face_perimeter / 4
  side_length ^ 3 = 343 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_perimeter_l2844_284495


namespace NUMINAMATH_CALUDE_oil_price_reduction_percentage_l2844_284424

/-- Proves that the percentage reduction in oil price is 25% given the specified conditions --/
theorem oil_price_reduction_percentage (original_price reduced_price : ℚ) : 
  reduced_price = 50 →
  (1000 / reduced_price) - (1000 / original_price) = 5 →
  (original_price - reduced_price) / original_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_percentage_l2844_284424


namespace NUMINAMATH_CALUDE_average_rate_of_change_average_rate_of_change_on_interval_l2844_284455

def f (x : ℝ) : ℝ := 2 * x + 1

theorem average_rate_of_change (a b : ℝ) (h : a < b) :
  (f b - f a) / (b - a) = 2 :=
by sorry

theorem average_rate_of_change_on_interval :
  (f 2 - f 1) / (2 - 1) = 2 :=
by sorry

end NUMINAMATH_CALUDE_average_rate_of_change_average_rate_of_change_on_interval_l2844_284455


namespace NUMINAMATH_CALUDE_expression_evaluation_l2844_284413

theorem expression_evaluation : (4 * 5 * 6) * (1/4 + 1/5 - 1/10) = 42 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2844_284413


namespace NUMINAMATH_CALUDE_border_collie_grooming_time_l2844_284417

/-- Represents the time in minutes Karen takes to groom different dog breeds -/
structure GroomingTimes where
  rottweiler : ℕ
  borderCollie : ℕ
  chihuahua : ℕ

/-- Represents the number of dogs Karen grooms in a specific session -/
structure DogCounts where
  rottweilers : ℕ
  borderCollies : ℕ
  chihuahuas : ℕ

/-- Given Karen's grooming times and dog counts, calculates the total grooming time -/
def totalGroomingTime (times : GroomingTimes) (counts : DogCounts) : ℕ :=
  times.rottweiler * counts.rottweilers +
  times.borderCollie * counts.borderCollies +
  times.chihuahua * counts.chihuahuas

/-- Theorem stating that Karen takes 10 minutes to groom a border collie -/
theorem border_collie_grooming_time :
  ∀ (times : GroomingTimes) (counts : DogCounts),
    times.rottweiler = 20 →
    times.chihuahua = 45 →
    counts.rottweilers = 6 →
    counts.borderCollies = 9 →
    counts.chihuahuas = 1 →
    totalGroomingTime times counts = 255 →
    times.borderCollie = 10 := by
  sorry

end NUMINAMATH_CALUDE_border_collie_grooming_time_l2844_284417


namespace NUMINAMATH_CALUDE_all_pairs_successful_probability_expected_successful_pairs_gt_half_l2844_284414

-- Define the number of sock pairs
variable (n : ℕ)

-- Define a successful pair
def successful_pair (pair : ℕ × ℕ) : Prop := pair.1 = pair.2

-- Define the probability of all pairs being successful
def all_pairs_successful_prob : ℚ := (2^n * n.factorial) / (2*n).factorial

-- Define the expected number of successful pairs
def expected_successful_pairs : ℚ := n / (2*n - 1)

-- Theorem 1: Probability of all pairs being successful
theorem all_pairs_successful_probability :
  all_pairs_successful_prob n = (2^n * n.factorial) / (2*n).factorial :=
sorry

-- Theorem 2: Expected number of successful pairs is greater than 0.5
theorem expected_successful_pairs_gt_half :
  expected_successful_pairs n > 1/2 :=
sorry

end NUMINAMATH_CALUDE_all_pairs_successful_probability_expected_successful_pairs_gt_half_l2844_284414


namespace NUMINAMATH_CALUDE_determine_investment_l2844_284422

/-- Represents the investment and profit share of a person -/
structure Investor where
  investment : ℕ
  profitShare : ℕ

/-- Given two investors with a specific profit sharing ratio and one known investment,
    prove that the other investor's investment can be determined -/
theorem determine_investment (p q : Investor) (h1 : p.profitShare = 2)
    (h2 : q.profitShare = 4) (h3 : p.investment = 500000) :
    q.investment = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_determine_investment_l2844_284422


namespace NUMINAMATH_CALUDE_sum_of_fifth_powers_l2844_284484

theorem sum_of_fifth_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  α^5 + β^5 + γ^5 = 47.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fifth_powers_l2844_284484


namespace NUMINAMATH_CALUDE_tan_plus_cot_l2844_284477

theorem tan_plus_cot (α : ℝ) (h : Real.sin (2 * α) = 3 / 4) :
  Real.tan α + (Real.tan α)⁻¹ = 8 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_cot_l2844_284477


namespace NUMINAMATH_CALUDE_sphere_volume_sum_l2844_284470

theorem sphere_volume_sum (π : ℝ) (h : π > 0) : 
  let sphere_volume (r : ℝ) := (4/3) * π * r^3
  sphere_volume 1 + sphere_volume 4 + sphere_volume 6 + sphere_volume 3 = (1232/3) * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_sum_l2844_284470


namespace NUMINAMATH_CALUDE_range_of_set_A_l2844_284449

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def set_A : Set ℕ := {p | 15 < p ∧ p < 36 ∧ is_prime p}

theorem range_of_set_A : 
  ∃ (min max : ℕ), min ∈ set_A ∧ max ∈ set_A ∧ 
  (∀ x ∈ set_A, min ≤ x ∧ x ≤ max) ∧
  max - min = 14 :=
sorry

end NUMINAMATH_CALUDE_range_of_set_A_l2844_284449


namespace NUMINAMATH_CALUDE_sequence_parity_l2844_284435

def T : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | 2 => 0
  | n + 3 => T (n + 2) + T (n + 1) - T n

theorem sequence_parity :
  (T 2021 % 2 = 1) ∧ (T 2022 % 2 = 0) ∧ (T 2023 % 2 = 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_parity_l2844_284435


namespace NUMINAMATH_CALUDE_intersection_equality_implies_possible_a_l2844_284430

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a*x + 2 = 0}

-- Define the set of possible values for a
def possible_a : Set ℝ := {-1, 0, 2/3}

-- Theorem statement
theorem intersection_equality_implies_possible_a :
  ∀ a : ℝ, (M ∩ N a = N a) → a ∈ possible_a :=
by sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_possible_a_l2844_284430


namespace NUMINAMATH_CALUDE_unique_solution_non_unique_solution_l2844_284402

-- Define the equation
def equation (x a b : ℝ) : Prop :=
  (x - a) / (x - 2) + (x - b) / (x - 3) = 2

-- Theorem for unique solution
theorem unique_solution (a b : ℝ) :
  (∃! x, equation x a b) ↔ (a + b ≠ 5 ∧ a ≠ 2 ∧ b ≠ 3) :=
sorry

-- Theorem for non-unique solution
theorem non_unique_solution (a b : ℝ) :
  (∃ x y, x ≠ y ∧ equation x a b ∧ equation y a b) ↔ (a = 2 ∧ b = 3) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_non_unique_solution_l2844_284402


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2844_284453

theorem prime_pairs_dividing_sum_of_powers (p q : Nat) : 
  Prime p → Prime q → (p * q ∣ 3^p + 3^q) ↔ 
  ((p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) ∨ (p = 3 ∧ q = 3) ∨ 
   (p = 3 ∧ q = 5) ∨ (p = 5 ∧ q = 3)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l2844_284453


namespace NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l2844_284464

theorem least_number_to_add_for_divisibility (n m : ℕ) (h : n = 1076 ∧ m = 23) :
  ∃ x : ℕ, (n + x) % m = 0 ∧ ∀ y : ℕ, y < x → (n + y) % m ≠ 0 ∧ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_least_number_to_add_for_divisibility_l2844_284464


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l2844_284472

theorem triangle_abc_theorem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- a, b, c are sides opposite to angles A, B, C
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  Real.cos (2 * A) + 2 * Real.sin (π + B) ^ 2 + 2 * Real.cos (π / 2 + C) ^ 2 - 1 = 2 * Real.sin B * Real.sin C →
  -- Given side lengths
  b = 4 ∧ c = 5 →
  -- Conclusions
  A = π / 3 ∧ Real.sin B = 2 * Real.sqrt 7 / 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_theorem_l2844_284472


namespace NUMINAMATH_CALUDE_triangle_radii_relation_l2844_284465

/-- Given a triangle with sides a, b, c, semi-perimeter p, inradius r, and excircle radii r_a, r_b, r_c,
    prove that 1/((p-a)(p-b)) + 1/((p-b)(p-c)) + 1/((p-c)(p-a)) = 1/r^2 -/
theorem triangle_radii_relation (a b c p r r_a r_b r_c : ℝ) 
  (h_p : p = (a + b + c) / 2)
  (h_r : r > 0)
  (h_ra : r_a > 0)
  (h_rb : r_b > 0)
  (h_rc : r_c > 0)
  (h_pbc : 1 / ((p - b) * (p - c)) = 1 / (r * r_a))
  (h_pca : 1 / ((p - c) * (p - a)) = 1 / (r * r_b))
  (h_pab : 1 / ((p - a) * (p - b)) = 1 / (r * r_c))
  (h_sum : 1 / r_a + 1 / r_b + 1 / r_c = 1 / r) :
  1 / ((p - a) * (p - b)) + 1 / ((p - b) * (p - c)) + 1 / ((p - c) * (p - a)) = 1 / r^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_radii_relation_l2844_284465


namespace NUMINAMATH_CALUDE_root_equation_problem_l2844_284481

theorem root_equation_problem (a b m p r : ℝ) : 
  (a^2 - m*a + 3 = 0) →
  (b^2 - m*b + 3 = 0) →
  ((a + 1/b)^2 - p*(a + 1/b) + r = 0) →
  ((b + 1/a)^2 - p*(b + 1/a) + r = 0) →
  r = 16/3 := by
sorry

end NUMINAMATH_CALUDE_root_equation_problem_l2844_284481


namespace NUMINAMATH_CALUDE_parabola_focus_and_directrix_l2844_284423

/-- Represents a parabola with equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola -/
def focus (p : Parabola) : ℝ × ℝ := (0, 1)

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => y = -1

theorem parabola_focus_and_directrix (p : Parabola) :
  (focus p = (0, 1)) ∧ (directrix p = fun y => y = -1) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_and_directrix_l2844_284423


namespace NUMINAMATH_CALUDE_binomial_n_minus_two_l2844_284483

theorem binomial_n_minus_two (n : ℕ+) : 
  Nat.choose n (n - 2) = n * (n - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_n_minus_two_l2844_284483


namespace NUMINAMATH_CALUDE_power_of_power_l2844_284439

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l2844_284439


namespace NUMINAMATH_CALUDE_clothing_production_solution_l2844_284432

/-- Represents the solution to the clothing production problem -/
def clothingProduction (totalFabric : ℝ) (topsPerUnit : ℝ) (pantsPerUnit : ℝ) (unitFabric : ℝ) 
  (fabricForTops : ℝ) (fabricForPants : ℝ) : Prop :=
  totalFabric > 0 ∧
  topsPerUnit > 0 ∧
  pantsPerUnit > 0 ∧
  unitFabric > 0 ∧
  fabricForTops ≥ 0 ∧
  fabricForPants ≥ 0 ∧
  fabricForTops + fabricForPants = totalFabric ∧
  (fabricForTops / unitFabric) * topsPerUnit = (fabricForPants / unitFabric) * pantsPerUnit

theorem clothing_production_solution :
  clothingProduction 600 2 3 3 360 240 := by
  sorry

end NUMINAMATH_CALUDE_clothing_production_solution_l2844_284432


namespace NUMINAMATH_CALUDE_fourth_rectangle_area_l2844_284412

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Theorem: Area of the fourth rectangle in a divided large rectangle -/
theorem fourth_rectangle_area
  (a b : ℝ)
  (r1 r2 r3 r4 : Rectangle)
  (h1 : r1.width = 2*a ∧ r1.height = b)
  (h2 : r2.width = 3*a ∧ r2.height = b)
  (h3 : r3.width = 2*a ∧ r3.height = 2*b)
  (h4 : r4.width = 3*a ∧ r4.height = 2*b)
  (area1 : area r1 = 2*a*b)
  (area2 : area r2 = 6*a*b)
  (area3 : area r3 = 4*a*b) :
  area r4 = 6*a*b :=
sorry

end NUMINAMATH_CALUDE_fourth_rectangle_area_l2844_284412


namespace NUMINAMATH_CALUDE_matrix_power_equals_fibonacci_l2844_284467

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℕ := !![1, 2; 1, 1]

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- State the theorem
theorem matrix_power_equals_fibonacci (n : ℕ) :
  A^n = !![fib (2*n + 1), fib (2*n + 2); fib (2*n), fib (2*n + 1)] := by
  sorry

end NUMINAMATH_CALUDE_matrix_power_equals_fibonacci_l2844_284467


namespace NUMINAMATH_CALUDE_inequality_proof_l2844_284497

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z ≥ 1/x + 1/y + 1/z) :
  x/y + y/z + z/x ≥ 1/(x*y) + 1/(y*z) + 1/(z*x) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2844_284497


namespace NUMINAMATH_CALUDE_quadratic_roots_product_l2844_284448

theorem quadratic_roots_product (n : ℝ) (c d : ℝ) :
  c^2 - n*c + 4 = 0 →
  d^2 - n*d + 4 = 0 →
  ∃ (s : ℝ), (c + 1/d)^2 - s*(c + 1/d) + 25/4 = 0 ∧
             (d + 1/c)^2 - s*(d + 1/c) + 25/4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_product_l2844_284448


namespace NUMINAMATH_CALUDE_subtract_negative_three_l2844_284438

theorem subtract_negative_three : 0 - (-3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_three_l2844_284438


namespace NUMINAMATH_CALUDE_train_bridge_crossing_time_l2844_284473

/-- Proves that a train of given length and speed takes 30 seconds to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) 
  (h1 : train_length = 120)
  (h2 : train_speed_kmh = 45)
  (h3 : bridge_length = 255) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_crossing_time_l2844_284473


namespace NUMINAMATH_CALUDE_percentage_relationship_l2844_284451

theorem percentage_relationship (x y : ℝ) (c : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : x = 2.5 * y) (h2 : 2 * y = c / 100 * x) : c = 80 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relationship_l2844_284451


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2844_284487

theorem simplify_complex_fraction :
  (1 / ((2 / (Real.sqrt 5 + 2)) + (3 / (Real.sqrt 7 - 2)))) =
  ((2 * Real.sqrt 5 + Real.sqrt 7 + 2) / (23 + 4 * Real.sqrt 35)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2844_284487


namespace NUMINAMATH_CALUDE_smallest_sum_of_squares_l2844_284431

theorem smallest_sum_of_squares (x y : ℕ+) : 
  (x.val * (x.val + 1) ∣ y.val * (y.val + 1)) ∧ 
  (¬ (x.val ∣ y.val) ∧ ¬ (x.val ∣ (y.val + 1)) ∧ ¬ ((x.val + 1) ∣ y.val) ∧ ¬ ((x.val + 1) ∣ (y.val + 1))) →
  (∀ a b : ℕ+, 
    (a.val * (a.val + 1) ∣ b.val * (b.val + 1)) ∧ 
    (¬ (a.val ∣ b.val) ∧ ¬ (a.val ∣ (b.val + 1)) ∧ ¬ ((a.val + 1) ∣ b.val) ∧ ¬ ((a.val + 1) ∣ (b.val + 1))) →
    x.val^2 + y.val^2 ≤ a.val^2 + b.val^2) →
  x.val^2 + y.val^2 = 1421 := by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_squares_l2844_284431


namespace NUMINAMATH_CALUDE_cricketer_average_score_l2844_284404

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (first_matches : ℕ) 
  (last_matches : ℕ) 
  (first_average : ℚ) 
  (last_average : ℚ) 
  (h1 : total_matches = first_matches + last_matches) 
  (h2 : total_matches = 10) 
  (h3 : first_matches = 6) 
  (h4 : last_matches = 4) 
  (h5 : first_average = 42) 
  (h6 : last_average = 34.25) : 
  (first_average * first_matches + last_average * last_matches) / total_matches = 38.9 := by
sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l2844_284404


namespace NUMINAMATH_CALUDE_sum_of_roots_l2844_284491

theorem sum_of_roots (a b : ℝ) : 
  a * (a - 4) = 12 → b * (b - 4) = 12 → a ≠ b → a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2844_284491


namespace NUMINAMATH_CALUDE_car_a_speed_calculation_l2844_284440

-- Define the problem parameters
def initial_distance : ℝ := 40
def overtake_distance : ℝ := 8
def car_b_speed : ℝ := 50
def overtake_time : ℝ := 6

-- Define the theorem
theorem car_a_speed_calculation :
  ∃ (speed_a : ℝ),
    speed_a * overtake_time = car_b_speed * overtake_time + initial_distance + overtake_distance ∧
    speed_a = 58 :=
by sorry

end NUMINAMATH_CALUDE_car_a_speed_calculation_l2844_284440


namespace NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l2844_284416

theorem degenerate_ellipse_max_y_coordinate :
  ∀ x y : ℝ, (x^2 / 49 + (y - 3)^2 / 25 = 0) → y ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_degenerate_ellipse_max_y_coordinate_l2844_284416


namespace NUMINAMATH_CALUDE_set_equation_solution_l2844_284407

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 1 = 0}
def B (a b : ℝ) : Set ℝ := {x : ℝ | x^2 - 2*a*x + b = 0}

-- State the theorem
theorem set_equation_solution (a b : ℝ) : 
  B a b ≠ ∅ ∧ B a b ⊆ A → 
  ((a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = 1) ∨ (a = 0 ∧ b = -1)) := by
  sorry

end NUMINAMATH_CALUDE_set_equation_solution_l2844_284407


namespace NUMINAMATH_CALUDE_second_number_less_than_twice_first_l2844_284452

theorem second_number_less_than_twice_first (x y : ℤ) : 
  x + y = 57 → y = 37 → 2 * x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_number_less_than_twice_first_l2844_284452


namespace NUMINAMATH_CALUDE_fifteenth_prime_l2844_284415

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def nth_prime (n : ℕ) : ℕ := sorry

theorem fifteenth_prime :
  (nth_prime 6 = 13) → (nth_prime 15 = 47) :=
sorry

end NUMINAMATH_CALUDE_fifteenth_prime_l2844_284415


namespace NUMINAMATH_CALUDE_final_balance_is_450_l2844_284426

/-- Calculates the final balance after withdrawal and deposit --/
def finalBalance (initialBalance : ℚ) : ℚ :=
  let remainingBalance := initialBalance - 200
  let depositAmount := remainingBalance / 2
  remainingBalance + depositAmount

/-- Theorem: The final balance is $450 given the conditions --/
theorem final_balance_is_450 :
  ∃ (initialBalance : ℚ),
    (initialBalance - 200 = initialBalance * (3/5)) ∧
    (finalBalance initialBalance = 450) :=
by sorry

end NUMINAMATH_CALUDE_final_balance_is_450_l2844_284426


namespace NUMINAMATH_CALUDE_expression_evaluation_l2844_284443

theorem expression_evaluation :
  let a : ℕ := 3
  let b : ℕ := 2
  let c : ℕ := 1
  ((a^2 + b*c) + (a*b + c))^2 - ((a^2 + b*c) - (a*b + c))^2 = 308 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2844_284443


namespace NUMINAMATH_CALUDE_first_term_of_ap_l2844_284406

def arithmetic_progression (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_progression (a₁ d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℚ) * d)

theorem first_term_of_ap (a₁ d : ℚ) :
  sum_arithmetic_progression a₁ d 22 = 1045 ∧
  sum_arithmetic_progression (arithmetic_progression a₁ d 23) d 22 = 2013 →
  a₁ = 53 / 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_ap_l2844_284406


namespace NUMINAMATH_CALUDE_divisible_by_perfect_cube_l2844_284492

theorem divisible_by_perfect_cube (a b : ℕ) (ha : a > 0) (hb : b > 0)
  (h_divides : (a^2 + 3*a*b + 3*b^2 - 1) ∣ (a + b^3)) :
  ∃ (n : ℕ), n > 1 ∧ (n^3 : ℕ) ∣ (a^2 + 3*a*b + 3*b^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_perfect_cube_l2844_284492
