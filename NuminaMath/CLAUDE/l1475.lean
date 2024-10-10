import Mathlib

namespace polynomial_identity_l1475_147593

theorem polynomial_identity (a b c d : ℝ) :
  (∀ x y : ℝ, (10*x + 6*y)^3 = a*x^3 + b*x^2*y + c*x*y^2 + d*y^3) →
  -a + 2*b - 4*c + 8*d = 8 := by
sorry

end polynomial_identity_l1475_147593


namespace shortest_distance_between_circles_l1475_147549

/-- The shortest distance between two circles -/
theorem shortest_distance_between_circles : 
  let center1 : ℝ × ℝ := (5, 3)
  let radius1 : ℝ := 12
  let center2 : ℝ × ℝ := (2, -1)
  let radius2 : ℝ := 6
  let distance_between_centers : ℝ := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  let shortest_distance : ℝ := max 0 (distance_between_centers - (radius1 + radius2))
  shortest_distance = 1 :=
by sorry

end shortest_distance_between_circles_l1475_147549


namespace equation_solution_range_l1475_147514

theorem equation_solution_range (b : ℝ) : 
  (∀ x : ℝ, x = -2 → x^2 - b*x - 5 = 5) →
  (∀ x : ℝ, x = -1 → x^2 - b*x - 5 = -1) →
  (∀ x : ℝ, x = 4 → x^2 - b*x - 5 = -1) →
  (∀ x : ℝ, x = 5 → x^2 - b*x - 5 = 5) →
  ∃ x y : ℝ, 
    (-2 < x ∧ x < -1 ∧ x^2 - b*x - 5 = 0) ∧
    (4 < y ∧ y < 5 ∧ y^2 - b*y - 5 = 0) ∧
    (∀ z : ℝ, z^2 - b*z - 5 = 0 → ((-2 < z ∧ z < -1) ∨ (4 < z ∧ z < 5))) :=
by sorry

end equation_solution_range_l1475_147514


namespace tuesday_appointment_duration_l1475_147588

/-- Amanda's hourly rate in dollars -/
def hourly_rate : ℚ := 20

/-- Duration of Monday appointments in hours -/
def monday_hours : ℚ := 5 * (3/2)

/-- Duration of Thursday appointments in hours -/
def thursday_hours : ℚ := 2 * 2

/-- Duration of Saturday appointment in hours -/
def saturday_hours : ℚ := 6

/-- Total earnings for the week in dollars -/
def total_earnings : ℚ := 410

/-- Duration of Tuesday appointment in hours -/
def tuesday_hours : ℚ := 3

theorem tuesday_appointment_duration :
  hourly_rate * (monday_hours + thursday_hours + saturday_hours + tuesday_hours) = total_earnings :=
by sorry

end tuesday_appointment_duration_l1475_147588


namespace melanie_yard_sale_books_l1475_147533

/-- The number of books Melanie bought at a yard sale -/
def books_bought (initial_books final_books : ℝ) : ℝ :=
  final_books - initial_books

/-- Proof that Melanie bought 87 books at the yard sale -/
theorem melanie_yard_sale_books : books_bought 41.0 128 = 87 := by
  sorry

end melanie_yard_sale_books_l1475_147533


namespace board_cutting_l1475_147535

theorem board_cutting (total_length shorter_length : ℝ) 
  (h1 : total_length = 120)
  (h2 : shorter_length = 35)
  (h3 : ∃ longer_length, longer_length + shorter_length = total_length ∧ 
    ∃ x, longer_length = 2 * shorter_length + x) :
  ∃ longer_length x, 
    longer_length + shorter_length = total_length ∧ 
    longer_length = 2 * shorter_length + x ∧ 
    x = 15 :=
by sorry

end board_cutting_l1475_147535


namespace extremum_conditions_another_extremum_l1475_147568

/-- The function f with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_conditions (a b : ℝ) : 
  (f a b (-1) = 8 ∧ f' a b (-1) = 0) → (a = -2 ∧ b = -7) :=
by sorry

theorem another_extremum : 
  f (-2) (-7) (7/3) = -284/27 ∧ 
  (∀ x : ℝ, x ≠ -1 ∧ x ≠ 7/3 → |f (-2) (-7) x| ≤ |f (-2) (-7) (7/3)|) :=
by sorry

end extremum_conditions_another_extremum_l1475_147568


namespace parabola_coefficient_l1475_147511

/-- Proves that for a parabola y = x^2 + bx + c passing through (1, -1) and (3, 9), c = -3 -/
theorem parabola_coefficient (b c : ℝ) : 
  (1^2 + b*1 + c = -1) → 
  (3^2 + b*3 + c = 9) → 
  c = -3 := by
  sorry

end parabola_coefficient_l1475_147511


namespace coin_toss_is_random_event_l1475_147540

/-- Represents the outcome of a coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents a random event -/
class RandomEvent (α : Type) where
  /-- The probability of the event occurring is between 0 and 1, exclusive -/
  prob_between_zero_and_one : ∃ (p : ℝ), 0 < p ∧ p < 1

/-- Definition of a coin toss -/
def coinToss : Set CoinOutcome := {CoinOutcome.Heads, CoinOutcome.Tails}

/-- Theorem: Tossing a coin is a random event -/
theorem coin_toss_is_random_event : RandomEvent coinToss := by
  sorry


end coin_toss_is_random_event_l1475_147540


namespace convex_quadrilaterals_12_points_l1475_147560

/-- The number of different convex quadrilaterals that can be drawn from n distinct points
    on the circumference of a circle, where each vertex of the quadrilateral must be one
    of these n points. -/
def convex_quadrilaterals (n : ℕ) : ℕ := Nat.choose n 4

/-- Theorem stating that the number of convex quadrilaterals from 12 points is 3960 -/
theorem convex_quadrilaterals_12_points :
  convex_quadrilaterals 12 = 3960 := by
  sorry

#eval convex_quadrilaterals 12

end convex_quadrilaterals_12_points_l1475_147560


namespace max_dot_product_l1475_147532

/-- The ellipse with equation x^2/4 + y^2/3 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + (p.2^2 / 3) = 1}

/-- The center of the ellipse -/
def O : ℝ × ℝ := (0, 0)

/-- The left focus of the ellipse -/
def F : ℝ × ℝ := (-1, 0)

/-- The dot product of vectors OP and FP -/
def dotProduct (P : ℝ × ℝ) : ℝ :=
  (P.1 * (P.1 + 1)) + (P.2 * P.2)

theorem max_dot_product :
  ∃ (M : ℝ), M = 6 ∧ ∀ P ∈ Ellipse, dotProduct P ≤ M :=
sorry

end max_dot_product_l1475_147532


namespace cos_pi_third_minus_2theta_l1475_147502

theorem cos_pi_third_minus_2theta (θ : ℝ) 
  (h : Real.sin (θ - π / 6) = Real.sqrt 3 / 3) : 
  Real.cos (π / 3 - 2 * θ) = 1 / 3 := by
  sorry

end cos_pi_third_minus_2theta_l1475_147502


namespace no_real_solutions_l1475_147529

theorem no_real_solutions : ∀ x : ℝ, ¬∃ y : ℝ, 
  (y = 3 * x - 1) ∧ (4 * y^2 + y + 3 = 3 * (8 * x^2 + 3 * y + 1)) := by
  sorry

end no_real_solutions_l1475_147529


namespace cube_surface_area_increase_l1475_147565

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_surface_area := 6 * L^2
  let new_edge_length := 1.5 * L
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area * 100 = 125 := by
sorry

end cube_surface_area_increase_l1475_147565


namespace min_value_on_line_min_value_achieved_l1475_147526

/-- The minimum value of 2/a + 3/b for points (a, b) in the first quadrant on the line 2x + 3y = 1 -/
theorem min_value_on_line (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2*a + 3*b = 1) :
  2/a + 3/b ≥ 25 := by
  sorry

/-- The minimum value 25 is achieved for some point on the line -/
theorem min_value_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2*a + 3*b = 1 ∧ |2/a + 3/b - 25| < ε := by
  sorry

end min_value_on_line_min_value_achieved_l1475_147526


namespace pams_bags_to_geralds_bags_l1475_147517

/-- Represents the number of apples in each of Gerald's bags -/
def geralds_bag_size : ℕ := 40

/-- Represents the total number of apples Pam has -/
def pams_total_apples : ℕ := 1200

/-- Represents the number of bags Pam has -/
def pams_bag_count : ℕ := 10

/-- Theorem stating that each of Pam's bags equates to 3 of Gerald's bags -/
theorem pams_bags_to_geralds_bags : 
  (pams_total_apples / pams_bag_count) / geralds_bag_size = 3 := by
  sorry

end pams_bags_to_geralds_bags_l1475_147517


namespace greatest_3digit_base8_divisible_by_7_l1475_147563

/-- Converts a base 8 number to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n / 10) % 10) * 8 + (n % 10)

/-- Checks if a number is a 3-digit base 8 number -/
def is_3digit_base8 (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∃ (n : ℕ), is_3digit_base8 n ∧ 
             base8_to_base10 n % 7 = 0 ∧
             ∀ (m : ℕ), is_3digit_base8 m ∧ base8_to_base10 m % 7 = 0 → m ≤ n :=
by
  use 777
  sorry

end greatest_3digit_base8_divisible_by_7_l1475_147563


namespace raisin_nut_cost_ratio_l1475_147587

theorem raisin_nut_cost_ratio :
  ∀ (r n : ℝ),
  r > 0 →
  n > 0 →
  (5 * r) / (5 * r + 4 * n) = 0.29411764705882354 →
  n / r = 3 :=
by sorry

end raisin_nut_cost_ratio_l1475_147587


namespace point_order_on_parabola_l1475_147590

/-- Parabola equation y = (x-1)^2 - 2 -/
def parabola (x y : ℝ) : Prop := y = (x - 1)^2 - 2

theorem point_order_on_parabola (a b c d : ℝ) :
  parabola a 2 →
  parabola b 6 →
  parabola c d →
  d < 1 →
  a < 0 →
  b > 0 →
  a < c ∧ c < b :=
sorry

end point_order_on_parabola_l1475_147590


namespace inequality_proof_l1475_147500

theorem inequality_proof (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end inequality_proof_l1475_147500


namespace triangle_value_l1475_147539

theorem triangle_value (triangle p : ℤ) 
  (eq1 : triangle + p = 73)
  (eq2 : (triangle + p) + 2*p = 157) : 
  triangle = 31 := by
sorry

end triangle_value_l1475_147539


namespace quadratic_solution_properties_l1475_147589

theorem quadratic_solution_properties :
  ∀ (y₁ y₂ : ℝ), y₁^2 - 1500*y₁ + 750 = 0 ∧ y₂^2 - 1500*y₂ + 750 = 0 →
  y₁ + y₂ = 1500 ∧ y₁ * y₂ = 750 := by
sorry

end quadratic_solution_properties_l1475_147589


namespace nina_running_distance_l1475_147594

theorem nina_running_distance : 0.08 + 0.08 + 0.67 = 0.83 := by sorry

end nina_running_distance_l1475_147594


namespace largest_b_for_divisibility_l1475_147519

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def five_digit_number (b : ℕ) : ℕ := 48000 + b * 100 + 56

theorem largest_b_for_divisibility :
  ∀ b : ℕ, b ≤ 9 →
    (is_divisible_by_4 (five_digit_number b) → b ≤ 8) ∧
    is_divisible_by_4 (five_digit_number 8) :=
by sorry

end largest_b_for_divisibility_l1475_147519


namespace intersection_M_N_l1475_147531

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 = 2*x}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end intersection_M_N_l1475_147531


namespace ben_age_is_five_l1475_147599

/-- Represents the ages of Amy, Ben, and Chris -/
structure Ages where
  amy : ℝ
  ben : ℝ
  chris : ℝ

/-- The conditions given in the problem -/
def satisfies_conditions (ages : Ages) : Prop :=
  -- The average of their ages is 12
  (ages.amy + ages.ben + ages.chris) / 3 = 12 ∧
  -- Four years ago, Chris was twice as old as Amy was then
  ages.chris - 4 = 2 * (ages.amy - 4) ∧
  -- In 5 years, Ben's age will be 3/4 of Amy's age at that time
  ages.ben + 5 = 3 / 4 * (ages.amy + 5)

/-- The theorem to be proved -/
theorem ben_age_is_five :
  ∃ (ages : Ages), satisfies_conditions ages ∧ ages.ben = 5 := by
  sorry

end ben_age_is_five_l1475_147599


namespace marble_count_l1475_147595

-- Define the number of marbles for each person
def allison_marbles : ℕ := 28
def angela_marbles : ℕ := allison_marbles + 8
def albert_marbles : ℕ := 3 * angela_marbles
def addison_marbles : ℕ := 2 * albert_marbles

-- Define the total number of marbles
def total_marbles : ℕ := allison_marbles + angela_marbles + albert_marbles + addison_marbles

-- Theorem to prove
theorem marble_count : total_marbles = 388 := by
  sorry

end marble_count_l1475_147595


namespace complex_magnitude_l1475_147555

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 4 - 2 * Complex.I) :
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_magnitude_l1475_147555


namespace problem_statement_l1475_147518

theorem problem_statement (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 2 / 5) :
  (a - c) * (b - d) / ((a - b) * (c - d)) = -3 / 2 := by
sorry

end problem_statement_l1475_147518


namespace parabola_symmetry_axis_part1_parabola_symmetry_axis_part2_l1475_147513

-- Define the parabola and its properties
def Parabola (a b c : ℝ) (h : a > 0) :=
  {f : ℝ → ℝ | ∀ x, f x = a * x^2 + b * x + c}

def AxisOfSymmetry (t : ℝ) (p : Parabola a b c h) :=
  t = -b / (2 * a)

-- Theorem for part (1)
theorem parabola_symmetry_axis_part1
  (a b c : ℝ) (h : a > 0) (p : Parabola a b c h) (t : ℝ) :
  AxisOfSymmetry t p →
  (a * 1^2 + b * 1 + c = a * 2^2 + b * 2 + c) →
  t = 3/2 := by sorry

-- Theorem for part (2)
theorem parabola_symmetry_axis_part2
  (a b c : ℝ) (h : a > 0) (p : Parabola a b c h) (t : ℝ) :
  AxisOfSymmetry t p →
  (∀ x₁ x₂, 0 < x₁ → x₁ < 1 → 1 < x₂ → x₂ < 2 →
    a * x₁^2 + b * x₁ + c < a * x₂^2 + b * x₂ + c) →
  t ≤ 1/2 := by sorry

end parabola_symmetry_axis_part1_parabola_symmetry_axis_part2_l1475_147513


namespace limit_f_derivative_at_one_l1475_147554

noncomputable def f (x : ℝ) : ℝ := (x^3 - 2*x) * Real.exp x

theorem limit_f_derivative_at_one :
  (deriv f) 1 = 0 :=
sorry

end limit_f_derivative_at_one_l1475_147554


namespace zeros_before_first_nonzero_digit_l1475_147542

theorem zeros_before_first_nonzero_digit (n : ℕ) (d : ℕ) (h : d = 2^7 * 5^9) :
  (∃ k : ℕ, (3 : ℚ) / d = (k : ℚ) / 10^9 ∧ 1 ≤ k ∧ k < 10) →
  (∃ m : ℕ, (3 : ℚ) / d = (m : ℚ) / 10^8 ∧ 10 ≤ m) →
  n = 8 :=
sorry

end zeros_before_first_nonzero_digit_l1475_147542


namespace composite_expression_l1475_147525

/-- A positive integer is composite if it can be expressed as a product of two integers,
    each greater than or equal to 2. -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ n = a * b

/-- Every composite number can be expressed as xy + xz + yz + 1,
    where x, y, and z are positive integers. -/
theorem composite_expression (c : ℕ) (h : IsComposite c) :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ c = x * y + x * z + y * z + 1 :=
sorry

end composite_expression_l1475_147525


namespace hexagon_area_is_32_l1475_147524

/-- A hexagon surrounded by four right triangles forming a rectangle -/
structure HexagonWithTriangles where
  -- Side length of the hexagon
  side_length : ℝ
  -- Height of each triangle
  triangle_height : ℝ
  -- The shape forms a rectangle
  is_rectangle : Bool
  -- There are four identical right triangles
  triangle_count : Nat
  -- The triangles are identical and right-angled
  triangles_identical_right : Bool

/-- The area of the hexagon given its structure -/
def hexagon_area (h : HexagonWithTriangles) : ℝ :=
  sorry

/-- Theorem stating the area of the hexagon is 32 square units -/
theorem hexagon_area_is_32 (h : HexagonWithTriangles) 
  (h_side : h.side_length = 2)
  (h_height : h.triangle_height = 4)
  (h_rect : h.is_rectangle = true)
  (h_count : h.triangle_count = 4)
  (h_tri : h.triangles_identical_right = true) :
  hexagon_area h = 32 := by
  sorry

end hexagon_area_is_32_l1475_147524


namespace line_ellipse_intersection_range_l1475_147596

/-- The range of k values for which the line y = kx + 2 intersects the ellipse 2x^2 + 3y^2 = 6 at two distinct points -/
theorem line_ellipse_intersection_range (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + 2 ∧ y₂ = k * x₂ + 2 ∧
    2 * x₁^2 + 3 * y₁^2 = 6 ∧ 
    2 * x₂^2 + 3 * y₂^2 = 6) ↔ 
  k < -Real.sqrt (2/3) ∨ k > Real.sqrt (2/3) :=
sorry

end line_ellipse_intersection_range_l1475_147596


namespace bacteria_count_after_six_hours_l1475_147516

/-- The number of bacteria at time n -/
def bacteria : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => bacteria (n + 1) + bacteria n

/-- The time in half-hour units after which we want to count bacteria -/
def target_time : ℕ := 12

theorem bacteria_count_after_six_hours :
  bacteria target_time = 233 := by
  sorry

end bacteria_count_after_six_hours_l1475_147516


namespace isosceles_triangle_angle_measure_l1475_147553

/-- 
Given an isosceles triangle ABC where:
- Angle A is congruent to angle C
- The measure of angle B is 40 degrees less than twice the measure of angle A
Prove that the measure of angle B is 70 degrees
-/
theorem isosceles_triangle_angle_measure (A B C : ℝ) : 
  A = C →  -- Angle A is congruent to angle C
  B = 2 * A - 40 →  -- Measure of angle B is 40 degrees less than twice the measure of angle A
  A + B + C = 180 →  -- Sum of angles in a triangle is 180 degrees
  B = 70 := by
sorry

end isosceles_triangle_angle_measure_l1475_147553


namespace negation_of_universal_proposition_l1475_147527

theorem negation_of_universal_proposition :
  (¬ ∀ n : ℕ, n^2 < 2^n) ↔ (∃ n₀ : ℕ, n₀^2 ≥ 2^n₀) :=
by sorry

end negation_of_universal_proposition_l1475_147527


namespace trig_identity_proof_l1475_147585

theorem trig_identity_proof :
  6 * Real.cos (10 * π / 180) * Real.cos (50 * π / 180) * Real.cos (70 * π / 180) +
  Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) =
  6 * (1 + Real.sqrt 3) / 8 := by
sorry

end trig_identity_proof_l1475_147585


namespace exists_non_isosceles_with_four_equal_subtriangles_l1475_147567

/-- A triangle represented by its three vertices -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- An interior point of a triangle -/
def InteriorPoint (t : Triangle) := ℝ × ℝ

/-- Predicate to check if a triangle is isosceles -/
def IsIsosceles (t : Triangle) : Prop := sorry

/-- Predicate to check if a point is inside a triangle -/
def IsInside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Function to create 6 triangles by connecting an interior point to vertices and drawing perpendiculars -/
def CreateSubTriangles (t : Triangle) (p : InteriorPoint t) : List Triangle := sorry

/-- Predicate to check if 4 out of 6 triangles in a list are equal -/
def FourOutOfSixEqual (triangles : List Triangle) : Prop := sorry

/-- Theorem stating that there exists a non-isosceles triangle with an interior point
    such that 4 out of 6 resulting triangles are equal -/
theorem exists_non_isosceles_with_four_equal_subtriangles :
  ∃ (t : Triangle) (p : InteriorPoint t),
    ¬IsIsosceles t ∧
    IsInside p t ∧
    FourOutOfSixEqual (CreateSubTriangles t p) := by
  sorry

end exists_non_isosceles_with_four_equal_subtriangles_l1475_147567


namespace workshop_workers_l1475_147586

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers (average_salary : ℝ) (technician_salary : ℝ) (other_salary : ℝ) 
  (num_technicians : ℕ) :
  average_salary = 8000 →
  technician_salary = 12000 →
  other_salary = 6000 →
  num_technicians = 7 →
  ∃ (total_workers : ℕ), 
    (total_workers : ℝ) * average_salary = 
      (num_technicians : ℝ) * technician_salary + 
      ((total_workers - num_technicians) : ℝ) * other_salary ∧
    total_workers = 21 :=
by sorry

end workshop_workers_l1475_147586


namespace vector_dot_product_result_l1475_147572

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

theorem vector_dot_product_result :
  (a.1 + 2 * b.1, a.2 + 2 * b.2) • c = -3 := by
  sorry

end vector_dot_product_result_l1475_147572


namespace unique_solution_floor_equation_l1475_147591

theorem unique_solution_floor_equation :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 72 ∧ x = 9 := by sorry

end unique_solution_floor_equation_l1475_147591


namespace fraction_who_say_dislike_but_like_l1475_147571

/-- Represents the student population at Gateway Academy -/
structure StudentPopulation where
  total : ℝ
  like_skating : ℝ
  dislike_skating : ℝ
  say_like_actually_like : ℝ
  say_dislike_actually_like : ℝ
  say_like_actually_dislike : ℝ
  say_dislike_actually_dislike : ℝ

/-- The conditions of the problem -/
def gateway_academy (pop : StudentPopulation) : Prop :=
  pop.total > 0 ∧
  pop.like_skating = 0.4 * pop.total ∧
  pop.dislike_skating = 0.6 * pop.total ∧
  pop.say_like_actually_like = 0.7 * pop.like_skating ∧
  pop.say_dislike_actually_like = 0.3 * pop.like_skating ∧
  pop.say_like_actually_dislike = 0.2 * pop.dislike_skating ∧
  pop.say_dislike_actually_dislike = 0.8 * pop.dislike_skating

/-- The theorem to be proved -/
theorem fraction_who_say_dislike_but_like (pop : StudentPopulation) 
  (h : gateway_academy pop) : 
  pop.say_dislike_actually_like / (pop.say_dislike_actually_like + pop.say_dislike_actually_dislike) = 0.2 := by
  sorry

end fraction_who_say_dislike_but_like_l1475_147571


namespace factor_count_of_M_l1475_147508

/-- The number of natural-number factors of M, where M = 2^4 * 3^3 * 5^2 * 7^1 -/
def number_of_factors (M : ℕ) : ℕ :=
  5 * 4 * 3 * 2

theorem factor_count_of_M :
  let M : ℕ := 2^4 * 3^3 * 5^2 * 7^1
  number_of_factors M = 120 := by
  sorry

end factor_count_of_M_l1475_147508


namespace elena_garden_petals_l1475_147521

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of petals in Elena's garden -/
def total_petals : ℕ := num_lilies * petals_per_lily + num_tulips * petals_per_tulip

theorem elena_garden_petals : total_petals = 63 := by
  sorry

end elena_garden_petals_l1475_147521


namespace equation_satisfied_when_m_is_34_l1475_147577

theorem equation_satisfied_when_m_is_34 :
  let m : ℕ := 34
  (((1 : ℚ) ^ (m + 1)) / ((5 : ℚ) ^ (m + 1))) * (((1 : ℚ) ^ 18) / ((4 : ℚ) ^ 18)) = 1 / (2 * ((10 : ℚ) ^ 35)) :=
by sorry

end equation_satisfied_when_m_is_34_l1475_147577


namespace complement_of_M_l1475_147552

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the set M
def M : Set ℝ := {x : ℝ | x^2 - 2*x ≤ 0}

-- State the theorem
theorem complement_of_M (x : ℝ) : x ∈ (Set.univ \ M) ↔ x < 0 ∨ x > 2 := by
  sorry

end complement_of_M_l1475_147552


namespace toms_age_ratio_l1475_147543

theorem toms_age_ratio (T N : ℝ) : 
  (T - N = 3 * (T - 4 * N)) → T / N = 11 / 2 := by
  sorry

end toms_age_ratio_l1475_147543


namespace customer_total_cost_l1475_147551

-- Define the quantities and prices of items
def riqing_quantity : ℕ := 24
def riqing_price : ℚ := 1.80
def riqing_discount : ℚ := 0.8

def kangshifu_quantity : ℕ := 6
def kangshifu_price : ℚ := 1.70
def kangshifu_discount : ℚ := 0.8

def shanlin_quantity : ℕ := 5
def shanlin_price : ℚ := 3.40
def shanlin_discount : ℚ := 1  -- No discount

def shuanghui_quantity : ℕ := 3
def shuanghui_price : ℚ := 11.20
def shuanghui_discount : ℚ := 0.9

-- Define the total cost function
def total_cost : ℚ :=
  riqing_quantity * riqing_price * riqing_discount +
  kangshifu_quantity * kangshifu_price * kangshifu_discount +
  shanlin_quantity * shanlin_price * shanlin_discount +
  shuanghui_quantity * shuanghui_price * shuanghui_discount

-- Theorem statement
theorem customer_total_cost : total_cost = 89.96 := by
  sorry

end customer_total_cost_l1475_147551


namespace correct_operation_is_subtraction_l1475_147569

-- Define the possible operations
inductive Operation
  | Add
  | Multiply
  | Divide
  | Subtract

-- Function to apply the operation
def applyOperation (op : Operation) (a b : ℤ) : ℤ :=
  match op with
  | Operation.Add => a + b
  | Operation.Multiply => a * b
  | Operation.Divide => a / b
  | Operation.Subtract => a - b

-- Theorem statement
theorem correct_operation_is_subtraction :
  ∃! op : Operation, (applyOperation op 8 4) + 6 - (3 - 2) = 9 :=
by sorry

end correct_operation_is_subtraction_l1475_147569


namespace point_trajectory_l1475_147561

/-- The trajectory of a point M(x,y) satisfying a specific distance condition -/
theorem point_trajectory (x y : ℝ) (h : Real.sqrt ((x + 5)^2 + y^2) - Real.sqrt ((x - 5)^2 + y^2) = 8) (hx : x > 0) :
  x^2 / 16 - y^2 / 9 = 1 := by
  sorry

end point_trajectory_l1475_147561


namespace min_y_l1475_147550

variable (a b c d : ℝ)
variable (x : ℝ)

def y (x : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + c*(x - d)^2

theorem min_y :
  ∃ (x_min : ℝ), (∀ (x : ℝ), y x_min ≤ y x) ∧ x_min = (a + b + c*d) / (2 + c) :=
sorry

end min_y_l1475_147550


namespace cone_volume_proof_l1475_147504

theorem cone_volume_proof (a b c r h : ℝ) : 
  a = 3 → b = 4 → c^2 = a^2 + b^2 → 
  2 * r = c → h^2 + r^2 = 3^2 →
  (1/3) * π * r^2 * h = (25 * π * Real.sqrt 11) / 24 := by
  sorry

end cone_volume_proof_l1475_147504


namespace baez_marbles_l1475_147583

theorem baez_marbles (p : ℝ) : 
  25 > 0 ∧ 0 ≤ p ∧ p ≤ 100 ∧ 2 * ((100 - p) / 100 * 25) = 60 → p = 20 :=
by sorry

end baez_marbles_l1475_147583


namespace factorization_x4_minus_64_l1475_147559

theorem factorization_x4_minus_64 (x : ℝ) : 
  x^4 - 64 = (x^2 + 8) * (x + 2 * Real.sqrt 2) * (x - 2 * Real.sqrt 2) := by
  sorry

end factorization_x4_minus_64_l1475_147559


namespace base_8_first_digit_350_l1475_147598

def base_8_first_digit (n : ℕ) : ℕ :=
  (n / 64) % 8

theorem base_8_first_digit_350 :
  base_8_first_digit 350 = 5 := by
  sorry

end base_8_first_digit_350_l1475_147598


namespace quadratic_inequality_solution_l1475_147507

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 50*x + 576 ≤ 16 ↔ 20 ≤ x ∧ x ≤ 28 := by
  sorry

end quadratic_inequality_solution_l1475_147507


namespace x_value_proof_l1475_147510

theorem x_value_proof (x : ℚ) 
  (eq1 : 8 * x^2 + 7 * x - 1 = 0)
  (eq2 : 24 * x^2 + 53 * x - 7 = 0) :
  x = 1/8 := by
sorry

end x_value_proof_l1475_147510


namespace polynomial_identity_sum_of_squares_l1475_147523

theorem polynomial_identity_sum_of_squares :
  ∀ (a b c d e f : ℤ),
  (∀ x : ℝ, 1728 * x^4 + 64 = (a * x^3 + b * x^2 + c) * (d * x^2 + e * x + f)) →
  a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 416 := by
  sorry

end polynomial_identity_sum_of_squares_l1475_147523


namespace optionC_most_suitable_l1475_147547

/-- Represents a sampling option with population size and sample size -/
structure SamplingOption where
  populationSize : ℕ
  sampleSize : ℕ

/-- Determines if a sampling option is suitable for simple random sampling -/
def isSuitableForSimpleRandomSampling (option : SamplingOption) : Prop :=
  option.populationSize ≤ 30 ∧ option.sampleSize ≤ 5

/-- The given sampling options -/
def optionA : SamplingOption := ⟨1320, 300⟩
def optionB : SamplingOption := ⟨1135, 50⟩
def optionC : SamplingOption := ⟨30, 5⟩
def optionD : SamplingOption := ⟨5000, 200⟩

/-- Theorem stating that Option C is most suitable for simple random sampling -/
theorem optionC_most_suitable :
  isSuitableForSimpleRandomSampling optionC ∧
  ¬isSuitableForSimpleRandomSampling optionA ∧
  ¬isSuitableForSimpleRandomSampling optionB ∧
  ¬isSuitableForSimpleRandomSampling optionD :=
by sorry

end optionC_most_suitable_l1475_147547


namespace fish_count_l1475_147582

def billy_fish : ℕ := 10

def tony_fish (billy : ℕ) : ℕ := 3 * billy

def sarah_fish (tony : ℕ) : ℕ := tony + 5

def bobby_fish (sarah : ℕ) : ℕ := 2 * sarah

def total_fish (billy tony sarah bobby : ℕ) : ℕ := billy + tony + sarah + bobby

theorem fish_count :
  total_fish billy_fish 
             (tony_fish billy_fish) 
             (sarah_fish (tony_fish billy_fish)) 
             (bobby_fish (sarah_fish (tony_fish billy_fish))) = 145 := by
  sorry

end fish_count_l1475_147582


namespace f_min_at_x_min_l1475_147562

/-- The quadratic function f(x) = x^2 - 12x + 36 -/
def f (x : ℝ) : ℝ := x^2 - 12*x + 36

/-- The point where the minimum of f occurs -/
def x_min : ℝ := 6

theorem f_min_at_x_min :
  ∀ x : ℝ, f x ≥ f x_min :=
sorry

end f_min_at_x_min_l1475_147562


namespace least_addition_for_divisibility_l1475_147574

theorem least_addition_for_divisibility :
  ∃! x : ℕ, x < 103 ∧ (3457 + x) % 103 = 0 ∧ ∀ y : ℕ, y < x → (3457 + y) % 103 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l1475_147574


namespace smallest_number_proof_l1475_147538

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Three positive integers
  (a + b + c) / 3 = 30 →   -- Arithmetic mean is 30
  b = 28 →                 -- Median is 28
  c = b + 6 →              -- Largest number is 6 more than median
  a < b ∧ b < c →          -- Ordering of numbers
  a = 28 :=                -- Smallest number is 28
by sorry

end smallest_number_proof_l1475_147538


namespace triangle_base_length_l1475_147536

theorem triangle_base_length (area : ℝ) (height : ℝ) (base : ℝ) : 
  area = 13.5 → height = 6 → area = (base * height) / 2 → base = 4.5 := by
  sorry

end triangle_base_length_l1475_147536


namespace inequality_one_min_value_min_point_l1475_147597

-- Define the variables and conditions
variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hab : a + b = 4)

-- Theorem 1
theorem inequality_one : 1/a + 1/(b+1) ≥ 4/5 := by sorry

-- Theorem 2
theorem min_value : ∃ (min_val : ℝ), min_val = (1 + Real.sqrt 5) / 2 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 4 → 4/(x*y) + x/y ≥ min_val := by sorry

-- Theorem for the values of a and b at the minimum point
theorem min_point : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + b = 4 ∧
  4/(a*b) + a/b = (1 + Real.sqrt 5) / 2 ∧
  a = Real.sqrt 5 - 1 ∧ b = 5 - Real.sqrt 5 := by sorry

end inequality_one_min_value_min_point_l1475_147597


namespace percentage_difference_l1475_147578

theorem percentage_difference : (0.6 * 50) - (0.42 * 30) = 17.4 := by
  sorry

end percentage_difference_l1475_147578


namespace white_ring_weight_l1475_147566

/-- Given the weights of three plastic rings (orange, purple, and white) and their total weight,
    this theorem proves that the weight of the white ring is equal to the total weight
    minus the sum of the orange and purple ring weights. -/
theorem white_ring_weight 
  (orange_weight : ℝ) 
  (purple_weight : ℝ) 
  (total_weight : ℝ) 
  (h1 : orange_weight = 0.08333333333333333)
  (h2 : purple_weight = 0.3333333333333333)
  (h3 : total_weight = 0.8333333333) :
  total_weight - (orange_weight + purple_weight) = 0.41666666663333337 := by
  sorry

#eval Float.toString (0.8333333333 - (0.08333333333333333 + 0.3333333333333333))

end white_ring_weight_l1475_147566


namespace cuboid_color_is_blue_l1475_147520

/-- Represents a cube with colored faces -/
structure ColoredCube where
  red_faces : Fin 6
  blue_faces : Fin 6
  yellow_faces : Fin 6
  face_sum : red_faces + blue_faces + yellow_faces = 6

/-- Represents the arrangement of cubes in a photo -/
structure CubeArrangement where
  red_visible : Nat
  blue_visible : Nat
  yellow_visible : Nat
  total_visible : red_visible + blue_visible + yellow_visible = 8

/-- The set of four cubes -/
def cube_set : Finset ColoredCube := sorry

/-- The three different arrangements in the colored photos -/
def arrangements : Finset CubeArrangement := sorry

theorem cuboid_color_is_blue 
  (h1 : ∀ c ∈ cube_set, c.red_faces + c.blue_faces + c.yellow_faces = 6)
  (h2 : cube_set.card = 4)
  (h3 : ∀ a ∈ arrangements, a.red_visible + a.blue_visible + a.yellow_visible = 8)
  (h4 : arrangements.card = 3)
  (h5 : (arrangements.sum (λ a => a.red_visible)) = 8)
  (h6 : (arrangements.sum (λ a => a.blue_visible)) = 8)
  (h7 : (arrangements.sum (λ a => a.yellow_visible)) = 8)
  (h8 : ∃ a ∈ arrangements, a.red_visible = 2)
  (h9 : ∃ c ∈ cube_set, c.yellow_faces = 0) :
  ∀ c ∈ cube_set, c.blue_faces ≥ 2 :=
sorry

end cuboid_color_is_blue_l1475_147520


namespace simplify_fraction_l1475_147512

theorem simplify_fraction (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) := by
  sorry

end simplify_fraction_l1475_147512


namespace seventeen_in_binary_l1475_147573

theorem seventeen_in_binary : 
  (17 : ℕ).digits 2 = [1, 0, 0, 0, 1] :=
sorry

end seventeen_in_binary_l1475_147573


namespace sum_of_seventh_powers_l1475_147556

theorem sum_of_seventh_powers (α β γ : ℂ) 
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 6)
  (h3 : α^3 + β^3 + γ^3 = 14) :
  α^7 + β^7 + γ^7 = -98 := by
  sorry

end sum_of_seventh_powers_l1475_147556


namespace lisa_photos_last_weekend_l1475_147544

/-- Calculates the number of photos Lisa took last weekend given the conditions --/
def photos_last_weekend (animal_photos : ℕ) (flower_multiplier : ℕ) (scenery_difference : ℕ) (weekend_difference : ℕ) : ℕ :=
  let flower_photos := animal_photos * flower_multiplier
  let scenery_photos := flower_photos - scenery_difference
  let total_photos := animal_photos + flower_photos + scenery_photos
  total_photos - weekend_difference

/-- Theorem stating that Lisa took 45 photos last weekend --/
theorem lisa_photos_last_weekend :
  photos_last_weekend 10 3 10 15 = 45 := by
  sorry

#eval photos_last_weekend 10 3 10 15

end lisa_photos_last_weekend_l1475_147544


namespace increasing_function_iff_a_in_range_l1475_147503

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

-- State the theorem
theorem increasing_function_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ (3/2 ≤ a ∧ a < 3) :=
by sorry

end increasing_function_iff_a_in_range_l1475_147503


namespace square_difference_of_sum_and_product_l1475_147522

theorem square_difference_of_sum_and_product (x y : ℕ+) 
  (sum_eq : x + y = 26) 
  (product_eq : x * y = 168) : 
  x^2 - y^2 = 52 := by
sorry

end square_difference_of_sum_and_product_l1475_147522


namespace radian_measure_of_15_degrees_l1475_147509

theorem radian_measure_of_15_degrees :
  let degree_to_radian (d : ℝ) := d * (Real.pi / 180)
  degree_to_radian 15 = Real.pi / 12 := by
  sorry

end radian_measure_of_15_degrees_l1475_147509


namespace grid_recoloring_theorem_l1475_147576

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents the grid -/
def Grid := Fin 99 → Fin 99 → Color

/-- Represents a row or column index -/
def Index := Fin 99

/-- Represents a recoloring operation -/
inductive RecolorOp
| Row (i : Index)
| Col (j : Index)

/-- Applies a recoloring operation to a grid -/
def applyRecolor (g : Grid) (op : RecolorOp) : Grid :=
  sorry

/-- Checks if all cells in the grid have the same color -/
def isMonochromatic (g : Grid) : Prop :=
  sorry

/-- The main theorem -/
theorem grid_recoloring_theorem (g : Grid) :
  ∃ (ops : List RecolorOp), isMonochromatic (ops.foldl applyRecolor g) :=
sorry

end grid_recoloring_theorem_l1475_147576


namespace quadratic_equation_general_form_l1475_147579

theorem quadratic_equation_general_form :
  ∀ x : ℝ, (1 + 3 * x) * (x - 3) = 2 * x^2 + 1 ↔ x^2 - 8 * x - 4 = 0 :=
by sorry

end quadratic_equation_general_form_l1475_147579


namespace contractor_problem_l1475_147570

/-- A contractor problem -/
theorem contractor_problem (daily_wage : ℚ) (daily_fine : ℚ) (total_earnings : ℚ) (absent_days : ℕ) :
  daily_wage = 25 →
  daily_fine = (15/2) →
  total_earnings = 555 →
  absent_days = 6 →
  ∃ (total_days : ℕ), total_days = 24 ∧ 
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_earnings :=
by sorry

end contractor_problem_l1475_147570


namespace function_inequality_range_l1475_147557

open Real

theorem function_inequality_range (f : ℝ → ℝ) (f' : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : f 0 = 1)
  (h3 : ∀ x, 3 * f x = f' x - 3) :
  {x | 4 * f x > f' x} = {x | x > log 2 / 3} := by sorry

end function_inequality_range_l1475_147557


namespace remainder_of_nested_star_l1475_147592

-- Define the star operation
def star (a b : ℕ) : ℕ := a * b - 2

-- Define a function to represent the nested star operations
def nested_star : ℕ → ℕ
| 0 => 9
| n + 1 => star (579 - 10 * n) (nested_star n)

-- Theorem statement
theorem remainder_of_nested_star :
  nested_star 57 % 100 = 29 := by
  sorry

end remainder_of_nested_star_l1475_147592


namespace inequality_proof_l1475_147537

theorem inequality_proof (x : ℝ) : 2 * (5 * x + 3) ≤ x - 3 * (1 - 2 * x) → x ≤ -3 := by
  sorry

end inequality_proof_l1475_147537


namespace proper_subsets_of_A_l1475_147505

def U : Finset ℕ := {0,1,2,3,4,5}

def C_U_A : Finset ℕ := {1,2,3}

def A : Finset ℕ := U \ C_U_A

theorem proper_subsets_of_A : Finset.card (Finset.powerset A \ {A}) = 7 := by
  sorry

end proper_subsets_of_A_l1475_147505


namespace nonagon_side_length_l1475_147515

/-- The length of one side of a regular nonagon with circumference 171 cm is 19 cm. -/
theorem nonagon_side_length : 
  ∀ (circumference side_length : ℝ),
  circumference = 171 →
  side_length * 9 = circumference →
  side_length = 19 := by
sorry

end nonagon_side_length_l1475_147515


namespace burger_nonfiller_percentage_l1475_147575

/-- Given a burger with a total weight and filler weight, calculate the percentage that is not filler -/
theorem burger_nonfiller_percentage
  (total_weight : ℝ)
  (filler_weight : ℝ)
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45)
  : (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end burger_nonfiller_percentage_l1475_147575


namespace p_neither_necessary_nor_sufficient_for_q_l1475_147558

theorem p_neither_necessary_nor_sufficient_for_q (a : ℝ) : 
  (∃ x, x < 0 ∧ x > x^2) ∧ 
  (∃ y, y < 0 ∧ ¬(y > y^2)) ∧ 
  (∃ z, z > z^2 ∧ ¬(z < 0)) := by
sorry

end p_neither_necessary_nor_sufficient_for_q_l1475_147558


namespace harvest_calculation_l1475_147546

/-- Represents the harvest schedule and quantities for oranges and apples -/
structure HarvestData where
  total_days : ℕ
  orange_sacks : ℕ
  apple_sacks : ℕ
  orange_interval : ℕ
  apple_interval : ℕ

/-- Calculates the number of sacks harvested per day when both fruits are harvested together -/
def sacks_per_joint_harvest_day (data : HarvestData) : ℚ :=
  let orange_days := data.total_days / data.orange_interval
  let apple_days := data.total_days / data.apple_interval
  let orange_per_day := data.orange_sacks / orange_days
  let apple_per_day := data.apple_sacks / apple_days
  orange_per_day + apple_per_day

/-- The main theorem stating the result of the harvest calculation -/
theorem harvest_calculation (data : HarvestData) 
  (h1 : data.total_days = 20)
  (h2 : data.orange_sacks = 56)
  (h3 : data.apple_sacks = 35)
  (h4 : data.orange_interval = 2)
  (h5 : data.apple_interval = 3) :
  sacks_per_joint_harvest_day data = 11.4333 := by
  sorry

end harvest_calculation_l1475_147546


namespace arithmetic_sequence_k_value_l1475_147564

/-- An arithmetic sequence -/
def ArithmeticSequence (b : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_k_value
  (b : ℕ → ℚ)
  (h_arith : ArithmeticSequence b)
  (h_sum1 : b 5 + b 8 + b 11 = 21)
  (h_sum2 : (Finset.range 11).sum (fun i => b (i + 5)) = 121)
  (h_bk : ∃ k : ℕ, b k = 23) :
  ∃ k : ℕ, b k = 23 ∧ k = 16 := by
sorry

end arithmetic_sequence_k_value_l1475_147564


namespace max_value_fraction_max_value_achievable_l1475_147548

theorem max_value_fraction (y : ℝ) :
  y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) ≤ 1/25 :=
by sorry

theorem max_value_achievable :
  ∃ y : ℝ, y^2 / (y^4 + 4*y^3 + y^2 + 8*y + 16) = 1/25 :=
by sorry

end max_value_fraction_max_value_achievable_l1475_147548


namespace perpendicular_vectors_t_value_l1475_147506

/-- Two-dimensional vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dotProduct (v w : Vector2D) : ℝ := v.x * w.x + v.y * w.y

/-- Perpendicularity of two 2D vectors -/
def isPerpendicular (v w : Vector2D) : Prop := dotProduct v w = 0

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ,
  let a : Vector2D := ⟨t, 1⟩
  let b : Vector2D := ⟨1, 2⟩
  isPerpendicular a b → t = -2 := by
  sorry

end perpendicular_vectors_t_value_l1475_147506


namespace initial_fish_caught_per_day_l1475_147528

-- Define the initial colony size
def initial_colony_size : ℕ := sorry

-- Define the colony size after the first year (doubled)
def first_year_size : ℕ := 2 * initial_colony_size

-- Define the colony size after the second year (tripled from first year)
def second_year_size : ℕ := 3 * first_year_size

-- Define the current colony size (after third year)
def current_colony_size : ℕ := 1077

-- Define the increase in the third year
def third_year_increase : ℕ := 129

-- Define the fish consumption per penguin per day
def fish_per_penguin : ℚ := 3/2

-- Theorem stating the initial number of fish caught per day
theorem initial_fish_caught_per_day :
  (initial_colony_size : ℚ) * fish_per_penguin = 237 :=
by sorry

end initial_fish_caught_per_day_l1475_147528


namespace smallest_four_digit_divisible_by_35_l1475_147501

theorem smallest_four_digit_divisible_by_35 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1050 :=
by sorry

end smallest_four_digit_divisible_by_35_l1475_147501


namespace correct_average_l1475_147584

theorem correct_average (n : Nat) (incorrect_avg : ℚ) (incorrect_num : ℚ) (correct_num : ℚ) :
  n = 10 →
  incorrect_avg = 16 →
  incorrect_num = 26 →
  correct_num = 46 →
  (n : ℚ) * incorrect_avg + (correct_num - incorrect_num) = n * 18 :=
by sorry

end correct_average_l1475_147584


namespace jackie_apples_l1475_147545

-- Define the number of apples Adam has
def adam_apples : ℕ := 8

-- Define the difference between Jackie's and Adam's apples
def difference : ℕ := 2

-- Theorem: Jackie has 10 apples
theorem jackie_apples : adam_apples + difference = 10 := by
  sorry

end jackie_apples_l1475_147545


namespace total_spending_is_638_l1475_147534

/-- The total spending of Elizabeth, Emma, and Elsa -/
def total_spending (emma_spending : ℕ) : ℕ :=
  let elsa_spending := 2 * emma_spending
  let elizabeth_spending := 4 * elsa_spending
  emma_spending + elsa_spending + elizabeth_spending

/-- Theorem: The total spending is $638 when Emma spent $58 -/
theorem total_spending_is_638 : total_spending 58 = 638 := by
  sorry

end total_spending_is_638_l1475_147534


namespace last_number_proof_l1475_147580

theorem last_number_proof (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 3)
  (h3 : A + D = 13) :
  D = 2 := by
sorry

end last_number_proof_l1475_147580


namespace root_sum_square_l1475_147530

theorem root_sum_square (α β : ℝ) : 
  (α^2 + α - 2023 = 0) → 
  (β^2 + β - 2023 = 0) → 
  α^2 + 2*α + β = 2022 := by
sorry

end root_sum_square_l1475_147530


namespace triangle_area_triangle_area_proof_l1475_147581

theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∃ (x y : ℝ),
      (x + y = 2005 ∧
       x / 2005 + y / 2006 = 1 ∧
       x / 2006 + y / 2005 = 1) →
      area = 2005^2 / (2 * 4011)

-- The proof is omitted
theorem triangle_area_proof : triangle_area (2005^2 / (2 * 4011)) :=
  sorry

end triangle_area_triangle_area_proof_l1475_147581


namespace number_division_l1475_147541

theorem number_division (x : ℚ) : x / 4 = 12 → x / 3 = 16 := by
  sorry

end number_division_l1475_147541
