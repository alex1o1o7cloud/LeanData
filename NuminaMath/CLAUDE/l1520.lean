import Mathlib

namespace NUMINAMATH_CALUDE_sandwiches_per_person_l1520_152015

theorem sandwiches_per_person (people : ℝ) (total_sandwiches : ℕ) 
  (h1 : people = 219) 
  (h2 : total_sandwiches = 657) : 
  (total_sandwiches : ℝ) / people = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwiches_per_person_l1520_152015


namespace NUMINAMATH_CALUDE_track_circumference_l1520_152006

/-- The circumference of a circular track given specific conditions -/
theorem track_circumference : 
  ∀ (circumference : ℝ) (distance_B_first_meet : ℝ) (distance_A_second_meet : ℝ),
  distance_B_first_meet = 100 →
  distance_A_second_meet = circumference - 60 →
  (circumference / 2 - distance_B_first_meet) / distance_B_first_meet = 
    distance_A_second_meet / (circumference + 60) →
  circumference = 480 := by
sorry

end NUMINAMATH_CALUDE_track_circumference_l1520_152006


namespace NUMINAMATH_CALUDE_lost_ship_depth_l1520_152058

/-- The depth of a lost ship given the diver's descent rate and time taken --/
theorem lost_ship_depth (descent_rate : ℝ) (time_taken : ℝ) (h1 : descent_rate = 35) (h2 : time_taken = 100) :
  descent_rate * time_taken = 3500 := by
  sorry

end NUMINAMATH_CALUDE_lost_ship_depth_l1520_152058


namespace NUMINAMATH_CALUDE_angle_supplementary_thrice_complementary_l1520_152009

theorem angle_supplementary_thrice_complementary (x : ℝ) :
  (180 - x = 3 * (90 - x)) → x = 45 := by
  sorry

end NUMINAMATH_CALUDE_angle_supplementary_thrice_complementary_l1520_152009


namespace NUMINAMATH_CALUDE_wage_decrease_compensation_l1520_152033

/-- Proves that a 25% increase in working hours maintains the same income after a 20% wage decrease --/
theorem wage_decrease_compensation (W H S : ℝ) (C : ℝ) (H_pos : H > 0) (W_pos : W > 0) :
  let original_income := W * H + C * S
  let new_wage := W * 0.8
  let new_hours := H * 1.25
  new_wage * new_hours + C * S = original_income := by
  sorry

end NUMINAMATH_CALUDE_wage_decrease_compensation_l1520_152033


namespace NUMINAMATH_CALUDE_factorization_equality_l1520_152064

theorem factorization_equality (a b : ℝ) : b^2 - a*b + a - b = (b - 1) * (b - a) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1520_152064


namespace NUMINAMATH_CALUDE_line_angle_and_parallel_distance_l1520_152043

/-- Line in 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line) : ℝ := sorry

/-- Distance between two parallel lines -/
def distance_between_parallel_lines (l1 l2 : Line) : ℝ := sorry

/-- Check if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop := sorry

theorem line_angle_and_parallel_distance 
  (l : Line) 
  (l1 : Line) 
  (l2 : Line) 
  (h1 : l.a = 1 ∧ l.b = -2 ∧ l.c = 1) 
  (h2 : l1.a = 2 ∧ l1.b = 1 ∧ l1.c = 1) 
  (h3 : are_parallel l l2) 
  (h4 : distance_between_parallel_lines l l2 = 1) : 
  (angle_between_lines l l1 = π / 2) ∧ 
  ((l2.a = l.a ∧ l2.b = l.b ∧ (l2.c = l.c - Real.sqrt 5 ∨ l2.c = l.c + Real.sqrt 5))) := 
by sorry

end NUMINAMATH_CALUDE_line_angle_and_parallel_distance_l1520_152043


namespace NUMINAMATH_CALUDE_horner_v3_value_l1520_152001

/-- Horner's Method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 12 + 35x - 8x^2 + 79x^3 + 6x^4 + 5x^5 + 3x^6 -/
def f : List ℝ := [12, 35, -8, 79, 6, 5, 3]

/-- The x-value at which to evaluate the polynomial -/
def x : ℝ := -4

/-- Theorem: The value of v3 in Horner's Method for f(x) at x = -4 is -57 -/
theorem horner_v3_value : 
  let v0 := f.reverse.head!
  let v1 := v0 * x + f.reverse.tail!.head!
  let v2 := v1 * x + f.reverse.tail!.tail!.head!
  let v3 := v2 * x + f.reverse.tail!.tail!.tail!.head!
  v3 = -57 := by sorry

end NUMINAMATH_CALUDE_horner_v3_value_l1520_152001


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l1520_152060

/-- Parabola type representing y^2 = ax -/
structure Parabola where
  a : ℝ
  hpos : a > 0

/-- Point type representing (x, y) coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line type representing a line with slope k -/
structure Line where
  k : ℝ

def intersect_parabola_line (p : Parabola) (l : Line) : Point × Point := sorry

def extend_line (p1 p2 : Point) : Line := sorry

def slope_of_line (p1 p2 : Point) : ℝ := sorry

theorem parabola_intersection_theorem (p : Parabola) (m : Point) 
  (h_m : m.x = 4 ∧ m.y = 0) (l : Line) (k2 : ℝ) 
  (h_k : l.k = Real.sqrt 2 * k2) :
  let f := Point.mk (p.a / 4) 0
  let (a, b) := intersect_parabola_line p l
  let c := intersect_parabola_line p (extend_line a m)
  let d := intersect_parabola_line p (extend_line b m)
  slope_of_line c.1 d.1 = k2 → p.a = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l1520_152060


namespace NUMINAMATH_CALUDE_complex_sum_powers_l1520_152034

theorem complex_sum_powers (w : ℂ) (hw : w^2 - w + 1 = 0) : 
  w^101 + w^102 + w^103 + w^104 + w^105 = 4*w - 1 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_powers_l1520_152034


namespace NUMINAMATH_CALUDE_swimmers_pass_23_times_l1520_152023

/-- Represents the number of times two swimmers pass each other in a pool --/
def swimmers_passing_count (pool_length : ℝ) (speed_a speed_b : ℝ) (total_time : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of times swimmers pass each other under given conditions --/
theorem swimmers_pass_23_times :
  swimmers_passing_count 120 4 3 (15 * 60) = 23 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_23_times_l1520_152023


namespace NUMINAMATH_CALUDE_role_assignment_count_l1520_152030

/-- The number of ways to assign roles in a play. -/
def assign_roles (num_men num_women : ℕ) : ℕ :=
  let male_role_assignments := num_men
  let female_role_assignments := num_women * (num_women - 1)
  let specific_role_assignment := 1
  let remaining_actors := (num_men - 1) + num_women
  let remaining_role_assignments := remaining_actors * (remaining_actors - 1)
  male_role_assignments * female_role_assignments * specific_role_assignment * remaining_role_assignments

/-- Theorem stating the number of ways to assign roles in the given scenario. -/
theorem role_assignment_count :
  assign_roles 6 7 = 27720 :=
by sorry

end NUMINAMATH_CALUDE_role_assignment_count_l1520_152030


namespace NUMINAMATH_CALUDE_carters_increased_baking_l1520_152028

def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_redvelvet : ℕ := 8
def tripling_factor : ℕ := 3

theorem carters_increased_baking :
  (usual_cheesecakes + usual_muffins + usual_redvelvet) * tripling_factor -
  (usual_cheesecakes + usual_muffins + usual_redvelvet) = 38 :=
by sorry

end NUMINAMATH_CALUDE_carters_increased_baking_l1520_152028


namespace NUMINAMATH_CALUDE_joohee_ate_17_chocolates_l1520_152007

-- Define the total number of chocolates
def total_chocolates : ℕ := 25

-- Define the relationship between Joo-hee's and Jun-seong's chocolates
def joohee_chocolates (junseong_chocolates : ℕ) : ℕ :=
  2 * junseong_chocolates + 1

-- Theorem statement
theorem joohee_ate_17_chocolates :
  ∃ (junseong_chocolates : ℕ),
    junseong_chocolates + joohee_chocolates junseong_chocolates = total_chocolates ∧
    joohee_chocolates junseong_chocolates = 17 :=
  sorry

end NUMINAMATH_CALUDE_joohee_ate_17_chocolates_l1520_152007


namespace NUMINAMATH_CALUDE_triangle_area_half_parallelogram_area_l1520_152012

/-- The area of a triangle with equal base and height is half the area of a parallelogram with the same base and height. -/
theorem triangle_area_half_parallelogram_area (b h : ℝ) (b_pos : 0 < b) (h_pos : 0 < h) :
  (1 / 2 * b * h) = (1 / 2) * (b * h) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_half_parallelogram_area_l1520_152012


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1520_152037

/-- A continuous function satisfying f(x) = a^x * f(x/2) for all x -/
def FunctionalEquation (f : ℝ → ℝ) (a : ℝ) : Prop :=
  Continuous f ∧ a > 0 ∧ ∀ x, f x = a^x * f (x/2)

theorem functional_equation_solution {f : ℝ → ℝ} {a : ℝ} 
  (h : FunctionalEquation f a) : 
  ∃ C : ℝ, ∀ x, f x = C * a^(2*x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1520_152037


namespace NUMINAMATH_CALUDE_number_of_people_is_fifteen_l1520_152067

theorem number_of_people_is_fifteen (x : ℕ) (y : ℕ) : 
  (12 * x + 3 = y) → 
  (13 * x - 12 = y) → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_number_of_people_is_fifteen_l1520_152067


namespace NUMINAMATH_CALUDE_digits_1498_to_1500_form_229_l1520_152063

/-- A function that generates the list of positive integers starting with 2 -/
def integerListStartingWith2 : ℕ → ℕ
| 0 => 2
| n + 1 => 
  let prev := integerListStartingWith2 n
  if prev < 10 then 20
  else if prev % 10 = 9 then prev + 11
  else prev + 1

/-- A function that returns the nth digit in the concatenated list -/
def nthDigitInList (n : ℕ) : ℕ := sorry

/-- Theorem stating that the 1498th, 1499th, and 1500th digits form 229 -/
theorem digits_1498_to_1500_form_229 : 
  (nthDigitInList 1498) * 100 + (nthDigitInList 1499) * 10 + nthDigitInList 1500 = 229 := by sorry

end NUMINAMATH_CALUDE_digits_1498_to_1500_form_229_l1520_152063


namespace NUMINAMATH_CALUDE_five_digit_division_l1520_152031

/-- A five-digit number -/
def FiveDigitNumber (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

/-- A four-digit number -/
def FourDigitNumber (m : ℕ) : Prop :=
  1000 ≤ m ∧ m ≤ 9999

/-- m is formed by removing the middle digit of n -/
def MiddleDigitRemoved (n m : ℕ) : Prop :=
  FiveDigitNumber n ∧ FourDigitNumber m ∧
  ∃ (a b c d e : ℕ), n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
                     m = a * 1000 + b * 100 + d * 10 + e

theorem five_digit_division (n m : ℕ) :
  FiveDigitNumber n → MiddleDigitRemoved n m →
  (∃ k : ℕ, n = k * m) ↔ ∃ a : ℕ, 10 ≤ a ∧ a ≤ 99 ∧ n = a * 1000 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_division_l1520_152031


namespace NUMINAMATH_CALUDE_direct_proportion_quadrants_l1520_152044

/-- A direct proportion function in a plane rectangular coordinate system -/
structure DirectProportionFunction where
  n : ℝ
  f : ℝ → ℝ
  h : ∀ x, f x = (n - 1) * x

/-- Predicate to check if a point (x, y) is in the first or third quadrant -/
def isInFirstOrThirdQuadrant (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)

/-- Predicate to check if the graph of a function passes through the first and third quadrants -/
def passesFirstAndThirdQuadrants (f : ℝ → ℝ) : Prop :=
  ∀ x, isInFirstOrThirdQuadrant x (f x)

/-- Theorem: If a direct proportion function's graph passes through the first and third quadrants,
    then n > 1 -/
theorem direct_proportion_quadrants (dpf : DirectProportionFunction)
    (h : passesFirstAndThirdQuadrants dpf.f) : dpf.n > 1 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_quadrants_l1520_152044


namespace NUMINAMATH_CALUDE_subset_of_any_implies_zero_l1520_152020

theorem subset_of_any_implies_zero (a : ℝ) : 
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_subset_of_any_implies_zero_l1520_152020


namespace NUMINAMATH_CALUDE_speedster_convertibles_l1520_152014

theorem speedster_convertibles (total : ℕ) 
  (h1 : 2 * total = 3 * (total - 60))  -- 2/3 of total are Speedsters, 60 are not
  (h2 : 5 * (total - 60) = 3 * total)  -- Restating h1 in a different form
  : (4 * (total - 60)) / 5 = 96 := by  -- 4/5 of Speedsters are convertibles
  sorry

#check speedster_convertibles

end NUMINAMATH_CALUDE_speedster_convertibles_l1520_152014


namespace NUMINAMATH_CALUDE_avg_people_moving_rounded_l1520_152099

/-- The number of people moving to Texas in two days -/
def people_moving : ℕ := 1500

/-- The number of days -/
def days : ℕ := 2

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculate the average number of people moving to Texas per minute -/
def avg_people_per_minute : ℚ :=
  people_moving / (days * hours_per_day * minutes_per_hour)

/-- Round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem avg_people_moving_rounded :
  round_to_nearest avg_people_per_minute = 1 := by sorry

end NUMINAMATH_CALUDE_avg_people_moving_rounded_l1520_152099


namespace NUMINAMATH_CALUDE_justin_flower_gathering_time_l1520_152093

/-- Proves that Justin has been gathering flowers for 1 hour given the problem conditions -/
theorem justin_flower_gathering_time :
  let classmates : ℕ := 30
  let time_per_flower : ℕ := 10  -- minutes
  let lost_flowers : ℕ := 3
  let remaining_time : ℕ := 210  -- minutes
  let total_flowers_needed : ℕ := classmates
  let remaining_flowers : ℕ := remaining_time / time_per_flower + lost_flowers
  let gathered_flowers : ℕ := total_flowers_needed - remaining_flowers
  let gathering_time : ℕ := gathered_flowers * time_per_flower
  gathering_time / 60 = 1  -- hours
  := by sorry

end NUMINAMATH_CALUDE_justin_flower_gathering_time_l1520_152093


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_l1520_152080

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- Define the square root function
noncomputable def squareRoot (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x ∧ (y ≥ 0 ∨ y ≤ 0)}

theorem cube_root_and_square_root :
  (cubeRoot (1/8) = 1/2) ∧
  (squareRoot ((-6)^2) = {6, -6}) :=
sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_l1520_152080


namespace NUMINAMATH_CALUDE_ellipse_h_plus_k_l1520_152029

/-- An ellipse with foci at (1, 2) and (4, 2), passing through (-1, 5) -/
structure Ellipse where
  /-- The first focus of the ellipse -/
  focus1 : ℝ × ℝ
  /-- The second focus of the ellipse -/
  focus2 : ℝ × ℝ
  /-- A point on the ellipse -/
  point : ℝ × ℝ
  /-- The center of the ellipse -/
  center : ℝ × ℝ
  /-- The semi-major axis length -/
  a : ℝ
  /-- The semi-minor axis length -/
  b : ℝ
  /-- Constraint: focus1 is at (1, 2) -/
  focus1_def : focus1 = (1, 2)
  /-- Constraint: focus2 is at (4, 2) -/
  focus2_def : focus2 = (4, 2)
  /-- Constraint: point is at (-1, 5) -/
  point_def : point = (-1, 5)
  /-- Constraint: center is the midpoint of foci -/
  center_def : center = ((focus1.1 + focus2.1) / 2, (focus1.2 + focus2.2) / 2)
  /-- Constraint: a is positive -/
  a_pos : a > 0
  /-- Constraint: b is positive -/
  b_pos : b > 0
  /-- Constraint: sum of distances from point to foci equals 2a -/
  sum_distances : Real.sqrt ((point.1 - focus1.1)^2 + (point.2 - focus1.2)^2) +
                  Real.sqrt ((point.1 - focus2.1)^2 + (point.2 - focus2.2)^2) = 2 * a

/-- Theorem: The sum of h and k in the standard form equation of the ellipse is 4.5 -/
theorem ellipse_h_plus_k (e : Ellipse) : e.center.1 + e.center.2 = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_h_plus_k_l1520_152029


namespace NUMINAMATH_CALUDE_solve_for_a_l1520_152005

theorem solve_for_a (a x : ℝ) : 
  (3/10) * a + (2*x + 4)/2 = 4*(x - 1) ∧ x = 3 → a = 10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l1520_152005


namespace NUMINAMATH_CALUDE_sin_difference_product_l1520_152048

theorem sin_difference_product (a b c : ℝ) :
  Real.sin (a + b) - Real.sin (a - c) = 2 * Real.cos (a + (b - c) / 2) * Real.sin ((b + c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_difference_product_l1520_152048


namespace NUMINAMATH_CALUDE_equation_is_linear_l1520_152082

/-- A linear equation in two variables is of the form Ax + By = C, where A, B, and C are constants, and A and B are not both zero. -/
def IsLinearEquationInTwoVariables (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (A B C : ℝ), (A ≠ 0 ∨ B ≠ 0) ∧ ∀ x y, f x y ↔ A * x + B * y = C

/-- The equation 3x - 1 = 2 - 5y is a linear equation in two variables. -/
theorem equation_is_linear : IsLinearEquationInTwoVariables (fun x y ↦ 3 * x - 1 = 2 - 5 * y) := by
  sorry

#check equation_is_linear

end NUMINAMATH_CALUDE_equation_is_linear_l1520_152082


namespace NUMINAMATH_CALUDE_proportional_sum_equation_l1520_152003

theorem proportional_sum_equation (x y z a : ℝ) : 
  (∃ (k : ℝ), x = 2*k ∧ y = 3*k ∧ z = 5*k) →  -- x, y, z are proportional to 2, 3, 5
  x + y + z = 100 →                           -- sum is 100
  y = a*x - 10 →                              -- equation for y
  a = 2 :=                                    -- conclusion: a = 2
by
  sorry

end NUMINAMATH_CALUDE_proportional_sum_equation_l1520_152003


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1520_152094

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 - x - 6 < 0} = {x : ℝ | -2 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l1520_152094


namespace NUMINAMATH_CALUDE_ratio_twenty_ten_l1520_152032

/-- A geometric sequence with specific properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  prop1 : a 2 * a 6 = 16
  prop2 : a 4 + a 8 = 8

/-- The ratio of the 20th term to the 10th term is 1 -/
theorem ratio_twenty_ten (seq : GeometricSequence) : seq.a 20 / seq.a 10 = 1 := by
  sorry

#check ratio_twenty_ten

end NUMINAMATH_CALUDE_ratio_twenty_ten_l1520_152032


namespace NUMINAMATH_CALUDE_air_conditioner_power_consumption_l1520_152079

/-- Power consumption of three air conditioners over specified periods -/
theorem air_conditioner_power_consumption 
  (power_A : Real) (hours_A : Real) (days_A : Real)
  (power_B : Real) (hours_B : Real) (days_B : Real)
  (power_C : Real) (hours_C : Real) (days_C : Real) :
  power_A = 7.2 →
  power_B = 9.6 →
  power_C = 12 →
  hours_A = 6 →
  hours_B = 4 →
  hours_C = 3 →
  days_A = 5 →
  days_B = 7 →
  days_C = 10 →
  (power_A / 8 * hours_A * days_A) +
  (power_B / 10 * hours_B * days_B) +
  (power_C / 12 * hours_C * days_C) = 83.88 := by
  sorry

#eval (7.2 / 8 * 6 * 5) + (9.6 / 10 * 4 * 7) + (12 / 12 * 3 * 10)

end NUMINAMATH_CALUDE_air_conditioner_power_consumption_l1520_152079


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_ending_4_l1520_152095

theorem greatest_three_digit_multiple_of_17_ending_4 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 17 = 0 ∧ n % 10 = 4 → n ≤ 204 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_ending_4_l1520_152095


namespace NUMINAMATH_CALUDE_can_form_triangle_l1520_152052

/-- Triangle Inequality Theorem: A set of three line segments can form a triangle
    if and only if the sum of the lengths of any two sides is greater than
    the length of the third side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Prove that the set of line segments (3, 4, 5) can form a triangle. -/
theorem can_form_triangle : triangle_inequality 3 4 5 := by
  sorry

end NUMINAMATH_CALUDE_can_form_triangle_l1520_152052


namespace NUMINAMATH_CALUDE_sin_center_symmetry_l1520_152010

open Real

theorem sin_center_symmetry (φ : ℝ) : 
  (0 < φ ∧ φ < π / 2) →
  (∃! x, π / 6 < x ∧ x < π / 3 ∧ ∃ k : ℤ, 2 * x + φ = k * π) →
  φ = 5 * π / 12 :=
by sorry

end NUMINAMATH_CALUDE_sin_center_symmetry_l1520_152010


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1520_152036

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  firstTerm : ℚ
  commonDiff : ℚ

/-- Sum of the first n terms of an arithmetic sequence. -/
def sumFirstNTerms (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.firstTerm + (n - 1 : ℚ) * seq.commonDiff)

/-- Sum of terms from index m to n (inclusive) of an arithmetic sequence. -/
def sumTermsMtoN (seq : ArithmeticSequence) (m n : ℕ) : ℚ :=
  sumFirstNTerms seq n - sumFirstNTerms seq (m - 1)

theorem arithmetic_sequence_first_term 
  (seq : ArithmeticSequence) 
  (h1 : sumFirstNTerms seq 30 = 450)
  (h2 : sumTermsMtoN seq 31 60 = 1650) : 
  seq.firstTerm = -13/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l1520_152036


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l1520_152024

theorem sqrt_sum_equals_abs_sum (x : ℝ) :
  Real.sqrt (x^2 + 6*x + 9) + Real.sqrt (x^2 - 6*x + 9) = |x - 3| + |x + 3| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_abs_sum_l1520_152024


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l1520_152083

theorem gcd_lcm_sum : Nat.gcd 45 75 + Nat.lcm 48 18 = 159 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l1520_152083


namespace NUMINAMATH_CALUDE_band_percentage_is_twenty_percent_l1520_152061

-- Define the number of students in the band
def students_in_band : ℕ := 168

-- Define the total number of students
def total_students : ℕ := 840

-- Define the percentage of students in the band
def percentage_in_band : ℚ := (students_in_band : ℚ) / total_students * 100

-- Theorem statement
theorem band_percentage_is_twenty_percent :
  percentage_in_band = 20 := by
  sorry

end NUMINAMATH_CALUDE_band_percentage_is_twenty_percent_l1520_152061


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1520_152078

theorem quadratic_real_roots (k d : ℝ) (h : k ≠ 0) :
  (∃ x : ℝ, x^2 + k*x + k^2 + d = 0) ↔ d ≤ -3/4 * k^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1520_152078


namespace NUMINAMATH_CALUDE_christmas_on_thursday_l1520_152017

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents dates in November and December -/
structure Date where
  month : Nat
  day : Nat

/-- Returns the day of the week for a given date, assuming November 27 is a Thursday -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry

theorem christmas_on_thursday (thanksgiving : Date)
    (h1 : thanksgiving.month = 11)
    (h2 : thanksgiving.day = 27)
    (h3 : dayOfWeek thanksgiving = DayOfWeek.Thursday) :
    dayOfWeek ⟨12, 25⟩ = DayOfWeek.Thursday :=
  sorry

end NUMINAMATH_CALUDE_christmas_on_thursday_l1520_152017


namespace NUMINAMATH_CALUDE_m_range_proof_l1520_152039

/-- Proposition p: The equation x^2+mx+1=0 has two distinct negative roots -/
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- Proposition q: The domain of the function f(x)=log_2(4x^2+4(m-2)x+1) is ℝ -/
def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

/-- The range of m given the conditions -/
def m_range : Set ℝ := {m : ℝ | m ≥ 3 ∨ (1 < m ∧ m ≤ 2)}

theorem m_range_proof (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_m_range_proof_l1520_152039


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l1520_152071

theorem smallest_sum_of_reciprocal_sum (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 10 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 10 → (a : ℤ) + b ≥ 45) ∧
  ∃ p q : ℕ+, p ≠ q ∧ (1 : ℚ) / p + (1 : ℚ) / q = (1 : ℚ) / 10 ∧ (p : ℤ) + q = 45 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocal_sum_l1520_152071


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_four_exists_148_with_gcd_four_less_than_150_max_integer_with_gcd_four_l1520_152047

theorem greatest_integer_with_gcd_four (n : ℕ) : n < 150 ∧ Nat.gcd n 12 = 4 → n ≤ 148 :=
by sorry

theorem exists_148_with_gcd_four : Nat.gcd 148 12 = 4 :=
by sorry

theorem less_than_150 : 148 < 150 :=
by sorry

theorem max_integer_with_gcd_four :
  ∀ m : ℕ, m < 150 ∧ Nat.gcd m 12 = 4 → m ≤ 148 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_four_exists_148_with_gcd_four_less_than_150_max_integer_with_gcd_four_l1520_152047


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1520_152090

/-- The slope of the line passing through the intersection points of two circles -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 8 = 0) ∧ 
  (x^2 + y^2 - 10*x + 18*y + 40 = 0) →
  (∃ m : ℚ, m = 2/7 ∧ 
   ∀ (x₁ y₁ x₂ y₂ : ℝ), 
   (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 8 = 0) ∧ 
   (x₁^2 + y₁^2 - 10*x₁ + 18*y₁ + 40 = 0) ∧
   (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 8 = 0) ∧ 
   (x₂^2 + y₂^2 - 10*x₂ + 18*y₂ + 40 = 0) ∧
   x₁ ≠ x₂ →
   m = (y₂ - y₁) / (x₂ - x₁)) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1520_152090


namespace NUMINAMATH_CALUDE_smallest_n_for_seven_numbers_l1520_152040

/-- Represents the sequence generation process -/
def generateSequence (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a number is an even square -/
def isEvenSquare (n : ℕ) : Bool :=
  sorry

/-- Finds the largest even square less than or equal to n -/
def largestEvenSquare (n : ℕ) : ℕ :=
  sorry

theorem smallest_n_for_seven_numbers : 
  (∀ m : ℕ, m < 168 → (generateSequence m).length ≠ 7) ∧ 
  (generateSequence 168).length = 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_seven_numbers_l1520_152040


namespace NUMINAMATH_CALUDE_polo_shirt_price_l1520_152046

/-- The regular price of a polo shirt -/
def regular_price : ℝ := 50

/-- The number of polo shirts purchased -/
def num_shirts : ℕ := 2

/-- The discount percentage on the shirts -/
def discount_percent : ℝ := 40

/-- The total amount paid for the shirts -/
def total_paid : ℝ := 60

/-- Theorem stating that the regular price of each polo shirt is $50 -/
theorem polo_shirt_price :
  regular_price = 50 ∧
  num_shirts * regular_price * (1 - discount_percent / 100) = total_paid :=
by sorry

end NUMINAMATH_CALUDE_polo_shirt_price_l1520_152046


namespace NUMINAMATH_CALUDE_cctv_systematic_sampling_group_size_l1520_152019

/-- Calculates the group size for systematic sampling -/
def systematicSamplingGroupSize (totalViewers : ℕ) (selectedViewers : ℕ) : ℕ :=
  totalViewers / selectedViewers

/-- Theorem: The group size for selecting 10 lucky viewers from 10000 viewers using systematic sampling is 1000 -/
theorem cctv_systematic_sampling_group_size :
  systematicSamplingGroupSize 10000 10 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_cctv_systematic_sampling_group_size_l1520_152019


namespace NUMINAMATH_CALUDE_pells_equation_unique_solution_l1520_152054

-- Define the fundamental solution
def fundamental_solution (x₀ y₀ : ℕ) : Prop :=
  x₀^2 - 2003 * y₀^2 = 1

-- Define the property that all prime factors of x divide x₀
def all_prime_factors_divide (x x₀ : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀

-- The main theorem
theorem pells_equation_unique_solution (x₀ y₀ x y : ℕ) :
  fundamental_solution x₀ y₀ →
  x^2 - 2003 * y^2 = 1 →
  x > 0 →
  y > 0 →
  all_prime_factors_divide x x₀ →
  x = x₀ ∧ y = y₀ :=
sorry

end NUMINAMATH_CALUDE_pells_equation_unique_solution_l1520_152054


namespace NUMINAMATH_CALUDE_michaels_brother_final_money_l1520_152059

/-- Given the initial conditions of Michael and his brother's money, and their subsequent actions,
    this theorem proves the final amount of money Michael's brother has. -/
theorem michaels_brother_final_money (michael_initial : ℕ) (brother_initial : ℕ) 
    (candy_cost : ℕ) (h1 : michael_initial = 42) (h2 : brother_initial = 17) 
    (h3 : candy_cost = 3) : 
    brother_initial + michael_initial / 2 - candy_cost = 35 := by
  sorry

end NUMINAMATH_CALUDE_michaels_brother_final_money_l1520_152059


namespace NUMINAMATH_CALUDE_max_gcd_of_sequence_l1520_152053

theorem max_gcd_of_sequence (n : ℕ+) : 
  Nat.gcd (99 + n^2) (99 + (n + 1)^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_of_sequence_l1520_152053


namespace NUMINAMATH_CALUDE_normal_dist_prob_l1520_152004

-- Define a random variable following normal distribution
def normal_dist (μ σ : ℝ) : Type := ℝ

-- Define the probability function
noncomputable def P (X : normal_dist 1 σ) (event : Set ℝ) : ℝ := sorry

-- State the theorem
theorem normal_dist_prob (σ : ℝ) (ξ : normal_dist 1 σ) 
  (h : P ξ {x | x < 0} = 0.4) : 
  P ξ {x | x < 2} = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_dist_prob_l1520_152004


namespace NUMINAMATH_CALUDE_large_monkey_cost_is_correct_l1520_152073

/-- The cost of a large monkey doll -/
def large_monkey_cost : ℝ := 6

/-- The total amount spent on dolls -/
def total_spent : ℝ := 300

/-- The cost difference between large and small monkey dolls -/
def small_large_diff : ℝ := 2

/-- The cost difference between elephant and large monkey dolls -/
def elephant_large_diff : ℝ := 1

/-- The number of additional dolls if buying only small monkeys -/
def small_monkey_diff : ℕ := 25

/-- The number of fewer dolls if buying only elephants -/
def elephant_diff : ℕ := 15

theorem large_monkey_cost_is_correct : 
  (total_spent / (large_monkey_cost - small_large_diff) = 
   total_spent / large_monkey_cost + small_monkey_diff) ∧
  (total_spent / (large_monkey_cost + elephant_large_diff) = 
   total_spent / large_monkey_cost - elephant_diff) := by
  sorry

end NUMINAMATH_CALUDE_large_monkey_cost_is_correct_l1520_152073


namespace NUMINAMATH_CALUDE_complex_product_equals_112_l1520_152011

theorem complex_product_equals_112 (y : ℂ) (h : y = Complex.exp (2 * Real.pi * Complex.I / 9)) :
  (3 * y + y^3) * (3 * y^3 + y^9) * (3 * y^6 + y^18) * 
  (3 * y^2 + y^6) * (3 * y^5 + y^15) * (3 * y^7 + y^21) = 112 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_equals_112_l1520_152011


namespace NUMINAMATH_CALUDE_congruent_figures_alignment_l1520_152025

/-- A plane figure represented as a set of points in ℝ² -/
def PlaneFigure : Type := Set (ℝ × ℝ)

/-- Congruence relation between two plane figures -/
def Congruent (F G : PlaneFigure) : Prop := sorry

/-- Parallel translation of a plane figure -/
def ParallelTranslation (v : ℝ × ℝ) (F : PlaneFigure) : PlaneFigure := sorry

/-- Rotation of a plane figure around a point -/
def Rotation (center : ℝ × ℝ) (angle : ℝ) (F : PlaneFigure) : PlaneFigure := sorry

theorem congruent_figures_alignment (F G : PlaneFigure) (h : Congruent F G) :
  (∃ v : ℝ × ℝ, ParallelTranslation v F = G) ∨
  (∃ center : ℝ × ℝ, ∃ angle : ℝ, Rotation center angle F = G) :=
by sorry

end NUMINAMATH_CALUDE_congruent_figures_alignment_l1520_152025


namespace NUMINAMATH_CALUDE_unique_valid_number_l1520_152000

def is_valid_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  100 ≤ n ∧ n < 1000 ∧
  tens = hundreds + 3 ∧
  units = tens - 4 ∧
  (hundreds + tens + units) / 2 = tens

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 473 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1520_152000


namespace NUMINAMATH_CALUDE_carolyn_initial_marbles_l1520_152035

/-- The number of marbles Carolyn shared with Diana -/
def marbles_shared : ℕ := 42

/-- The number of marbles Carolyn had left after sharing -/
def marbles_left : ℕ := 5

/-- The number of oranges Carolyn started with (not used in the proof, but mentioned in the problem) -/
def initial_oranges : ℕ := 6

/-- Carolyn's initial number of marbles -/
def initial_marbles : ℕ := marbles_shared + marbles_left

theorem carolyn_initial_marbles :
  initial_marbles = 47 :=
by sorry

end NUMINAMATH_CALUDE_carolyn_initial_marbles_l1520_152035


namespace NUMINAMATH_CALUDE_sum_f_equals_1326_l1520_152066

/-- The number of integer points on the line segment from (0,0) to (n, n+3), excluding endpoints -/
def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 2 else 0

/-- The sum of f(n) for n from 1 to 1990 -/
def sum_f : ℕ := (Finset.range 1990).sum f

theorem sum_f_equals_1326 : sum_f = 1326 := by sorry

end NUMINAMATH_CALUDE_sum_f_equals_1326_l1520_152066


namespace NUMINAMATH_CALUDE_arg_z1_div_z2_l1520_152018

theorem arg_z1_div_z2 (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 1) (h3 : z₂ - z₁ = -1) :
  Complex.arg (z₁ / z₂) = π / 3 ∨ Complex.arg (z₁ / z₂) = 5 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_arg_z1_div_z2_l1520_152018


namespace NUMINAMATH_CALUDE_max_value_of_expression_l1520_152041

theorem max_value_of_expression (x y : ℝ) : 
  2 * x^2 + 3 * y^2 = 22 * x + 18 * y + 20 →
  4 * x + 5 * y ≤ 110 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l1520_152041


namespace NUMINAMATH_CALUDE_no_equal_conversions_l1520_152075

def fahrenheit_to_celsius (f : ℤ) : ℤ :=
  ⌊(5 : ℚ) / 9 * (f - 32)⌋

def celsius_to_fahrenheit (c : ℤ) : ℤ :=
  ⌊(9 : ℚ) / 5 * c + 33⌋

theorem no_equal_conversions :
  ∀ f : ℤ, 34 ≤ f ∧ f ≤ 1024 →
    f ≠ celsius_to_fahrenheit (fahrenheit_to_celsius f) :=
by sorry

end NUMINAMATH_CALUDE_no_equal_conversions_l1520_152075


namespace NUMINAMATH_CALUDE_total_balloons_l1520_152042

theorem total_balloons (allan_balloons jake_balloons maria_balloons : ℕ) 
  (h1 : allan_balloons = 5)
  (h2 : jake_balloons = 7)
  (h3 : maria_balloons = 3) :
  allan_balloons + jake_balloons + maria_balloons = 15 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l1520_152042


namespace NUMINAMATH_CALUDE_bus_driver_average_hours_l1520_152049

/-- The average number of hours the bus driver drives each day -/
def average_hours : ℝ := 2

/-- The average speed from Monday to Wednesday in km/h -/
def speed_mon_wed : ℝ := 12

/-- The average speed from Thursday to Friday in km/h -/
def speed_thu_fri : ℝ := 9

/-- The total distance traveled in 5 days in km -/
def total_distance : ℝ := 108

/-- The number of days driven from Monday to Wednesday -/
def days_mon_wed : ℝ := 3

/-- The number of days driven from Thursday to Friday -/
def days_thu_fri : ℝ := 2

theorem bus_driver_average_hours :
  average_hours * speed_mon_wed * days_mon_wed +
  average_hours * speed_thu_fri * days_thu_fri = total_distance :=
sorry

end NUMINAMATH_CALUDE_bus_driver_average_hours_l1520_152049


namespace NUMINAMATH_CALUDE_soda_price_ratio_l1520_152062

/-- The ratio of unit prices between two soda brands -/
theorem soda_price_ratio 
  (v : ℝ) -- Volume of Brand Y soda
  (p : ℝ) -- Price of Brand Y soda
  (h_v_pos : v > 0) -- Assumption that volume is positive
  (h_p_pos : p > 0) -- Assumption that price is positive
  : (0.85 * p) / (1.35 * v) / (p / v) = 17 / 27 := by
  sorry

end NUMINAMATH_CALUDE_soda_price_ratio_l1520_152062


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l1520_152068

def M : Set ℝ := {x | 0 < x ∧ x ≤ 4}
def N : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l1520_152068


namespace NUMINAMATH_CALUDE_quadratic_sum_and_reciprocal_l1520_152013

theorem quadratic_sum_and_reciprocal (t : ℝ) (h : t^2 - 3*t + 1 = 0) : t + 1/t = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_and_reciprocal_l1520_152013


namespace NUMINAMATH_CALUDE_game_a_more_likely_than_game_b_l1520_152088

def prob_heads : ℚ := 3/4
def prob_tails : ℚ := 1/4

def prob_game_a : ℚ := prob_heads^4

def prob_game_b : ℚ := (prob_heads * prob_tails)^3

theorem game_a_more_likely_than_game_b : prob_game_a > prob_game_b := by
  sorry

end NUMINAMATH_CALUDE_game_a_more_likely_than_game_b_l1520_152088


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1520_152085

/-- The standard equation of a circle with center (h, k) and radius r is (x-h)^2 + (y-k)^2 = r^2 -/
def StandardCircleEquation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Prove that for a circle with center (2, -1) and radius 3, the standard equation is (x-2)^2 + (y+1)^2 = 9 -/
theorem circle_equation_proof :
  ∀ (x y : ℝ), StandardCircleEquation 2 (-1) 3 x y ↔ (x - 2)^2 + (y + 1)^2 = 9 := by
  sorry


end NUMINAMATH_CALUDE_circle_equation_proof_l1520_152085


namespace NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l1520_152022

-- Define the universal set U
def U : Set ℝ := {x | x < 3}

-- Define the subset A
def A : Set ℝ := {x | x < 1}

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem statement
theorem complement_of_A_relative_to_U :
  complement_U_A = {x | 1 ≤ x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l1520_152022


namespace NUMINAMATH_CALUDE_camillas_jelly_beans_l1520_152021

theorem camillas_jelly_beans (b c : ℕ) : 
  b = 2 * c →                     -- Initial condition: twice as many blueberry as cherry
  b - 10 = 3 * (c - 10) →         -- Condition after eating: three times as many blueberry as cherry
  b = 40                          -- Conclusion: original number of blueberry jelly beans
:= by sorry

end NUMINAMATH_CALUDE_camillas_jelly_beans_l1520_152021


namespace NUMINAMATH_CALUDE_anthony_total_pencils_l1520_152077

/-- The total number of pencils Anthony has after receiving more from Kathryn -/
theorem anthony_total_pencils (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 9 → received = 56 → total = initial + received → total = 65 := by
  sorry

end NUMINAMATH_CALUDE_anthony_total_pencils_l1520_152077


namespace NUMINAMATH_CALUDE_subset_relation_l1520_152026

theorem subset_relation (P Q : Set ℝ) : Q ⊆ P :=
  by
  -- Define sets P and Q
  have h_P : P = {x : ℝ | x < 4} := by sorry
  have h_Q : Q = {x : ℝ | x^2 < 4} := by sorry

  -- Prove that Q is a subset of P
  sorry

end NUMINAMATH_CALUDE_subset_relation_l1520_152026


namespace NUMINAMATH_CALUDE_sum_of_numbers_leq_threshold_l1520_152089

theorem sum_of_numbers_leq_threshold : 
  let numbers : List ℚ := [8/10, 1/2, 9/10]
  let threshold : ℚ := 4/10
  (numbers.filter (λ x => x ≤ threshold)).sum = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_leq_threshold_l1520_152089


namespace NUMINAMATH_CALUDE_largest_angle_of_pentagon_l1520_152038

/-- Represents the measures of interior angles of a convex pentagon --/
structure PentagonAngles where
  x : ℝ
  angle1 : ℝ := x - 3
  angle2 : ℝ := x - 2
  angle3 : ℝ := x - 1
  angle4 : ℝ := x
  angle5 : ℝ := x + 1

/-- The sum of interior angles of a pentagon is 540° --/
def sumOfPentagonAngles : ℝ := 540

theorem largest_angle_of_pentagon (p : PentagonAngles) :
  p.angle1 + p.angle2 + p.angle3 + p.angle4 + p.angle5 = sumOfPentagonAngles →
  p.angle5 = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_of_pentagon_l1520_152038


namespace NUMINAMATH_CALUDE_string_displacement_impossible_l1520_152092

/-- A rectangular parallelepiped box with strings. -/
structure StringBox where
  a : ℝ
  b : ℝ
  c : ℝ
  N : ℝ × ℝ × ℝ
  P : ℝ × ℝ × ℝ

/-- Strings cross at right angles at N and P. -/
def strings_cross_at_right_angles (box : StringBox) : Prop :=
  sorry

/-- Strings are strongly glued at N and P. -/
def strings_strongly_glued (box : StringBox) : Prop :=
  sorry

/-- Any displacement of the strings is impossible. -/
def no_displacement_possible (box : StringBox) : Prop :=
  sorry

/-- Theorem: If strings cross at right angles and are strongly glued at N and P,
    then any displacement of the strings is impossible. -/
theorem string_displacement_impossible (box : StringBox) :
  strings_cross_at_right_angles box →
  strings_strongly_glued box →
  no_displacement_possible box :=
by
  sorry

end NUMINAMATH_CALUDE_string_displacement_impossible_l1520_152092


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1520_152097

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 = 3 * y) : 
  x^2 - 6*x*y + 9*y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1520_152097


namespace NUMINAMATH_CALUDE_dice_sum_product_l1520_152076

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 360 →
  a + b + c + d ≠ 20 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l1520_152076


namespace NUMINAMATH_CALUDE_only_B_forms_grid_l1520_152070

/-- Represents a shape that can be used in the puzzle game -/
inductive Shape
  | A
  | B
  | C

/-- Represents a 4x4 grid -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a shape can form a complete 4x4 grid without gaps or overlaps -/
def canFormGrid (s : Shape) : Prop :=
  ∃ (g : Grid), ∀ (i j : Fin 4), g i j = true

/-- Theorem stating that only shape B can form a complete 4x4 grid -/
theorem only_B_forms_grid :
  (canFormGrid Shape.B) ∧ 
  (¬ canFormGrid Shape.A) ∧ 
  (¬ canFormGrid Shape.C) :=
sorry

end NUMINAMATH_CALUDE_only_B_forms_grid_l1520_152070


namespace NUMINAMATH_CALUDE_xe_exp_increasing_l1520_152055

/-- The function f(x) = xe^x is increasing for all x > 0 -/
theorem xe_exp_increasing (x : ℝ) (h : x > 0) :
  Monotone (fun x => x * Real.exp x) := by sorry

end NUMINAMATH_CALUDE_xe_exp_increasing_l1520_152055


namespace NUMINAMATH_CALUDE_max_additional_plates_l1520_152045

def first_set : Finset Char := {'B', 'F', 'J', 'M', 'S'}
def second_set : Finset Char := {'E', 'U', 'Y'}
def third_set : Finset Char := {'G', 'K', 'R', 'Z'}

theorem max_additional_plates :
  ∃ (new_first : Char) (new_third : Char),
    new_first ∉ first_set ∧
    new_third ∉ third_set ∧
    (first_set.card + 1) * second_set.card * (third_set.card + 1) -
    first_set.card * second_set.card * third_set.card = 30 ∧
    ∀ (a : Char) (c : Char),
      a ∉ first_set →
      c ∉ third_set →
      (first_set.card + 1) * second_set.card * (third_set.card + 1) -
      first_set.card * second_set.card * third_set.card ≤ 30 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_plates_l1520_152045


namespace NUMINAMATH_CALUDE_dean_books_count_l1520_152065

theorem dean_books_count (tony_books breanna_books total_different_books : ℕ)
  (tony_dean_shared all_shared : ℕ) :
  tony_books = 23 →
  breanna_books = 17 →
  tony_dean_shared = 3 →
  all_shared = 1 →
  total_different_books = 47 →
  ∃ dean_books : ℕ,
    dean_books = 16 ∧
    total_different_books =
      (tony_books - tony_dean_shared - all_shared) +
      (dean_books - tony_dean_shared - all_shared) +
      (breanna_books - all_shared) :=
by sorry

end NUMINAMATH_CALUDE_dean_books_count_l1520_152065


namespace NUMINAMATH_CALUDE_betty_garden_total_l1520_152086

/-- Represents Betty's herb garden -/
structure HerbGarden where
  basil : ℕ
  oregano : ℕ

/-- The number of oregano plants is 2 more than twice the number of basil plants -/
def oregano_rule (garden : HerbGarden) : Prop :=
  garden.oregano = 2 + 2 * garden.basil

/-- Betty's garden has 5 basil plants -/
def betty_garden : HerbGarden :=
  { basil := 5, oregano := 2 + 2 * 5 }

/-- The total number of plants in the garden -/
def total_plants (garden : HerbGarden) : ℕ :=
  garden.basil + garden.oregano

theorem betty_garden_total : total_plants betty_garden = 17 := by
  sorry

end NUMINAMATH_CALUDE_betty_garden_total_l1520_152086


namespace NUMINAMATH_CALUDE_three_possible_values_for_sum_l1520_152002

theorem three_possible_values_for_sum (x y : ℤ) 
  (h : x^2 + y^2 + 1 ≤ 2*x + 2*y) : 
  ∃ (S : Finset ℤ), (Finset.card S = 3) ∧ ((x + y) ∈ S) :=
sorry

end NUMINAMATH_CALUDE_three_possible_values_for_sum_l1520_152002


namespace NUMINAMATH_CALUDE_negation_of_existence_power_of_two_l1520_152057

theorem negation_of_existence_power_of_two (p : Prop) : 
  (p ↔ ∃ n : ℕ, 2^n > 1000) → 
  (¬p ↔ ∀ n : ℕ, 2^n ≤ 1000) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_power_of_two_l1520_152057


namespace NUMINAMATH_CALUDE_largest_in_set_l1520_152072

def a : ℝ := -4

def S : Set ℝ := {-3 * a, 4 * a, 24 / a, a^2, 2 * a + 1, 1}

theorem largest_in_set : ∀ x ∈ S, x ≤ a^2 := by sorry

end NUMINAMATH_CALUDE_largest_in_set_l1520_152072


namespace NUMINAMATH_CALUDE_birth_year_problem_l1520_152056

theorem birth_year_problem (x : ℕ) : 
  (1850 ≤ x^2 + x) ∧ (x^2 + x < 1900) → -- Born in second half of 19th century
  (x^2 + 2*x - x = x^2 + x) →           -- x years old in year x^2 + 2x
  x^2 + x = 1892                        -- Year of birth is 1892
:= by sorry

end NUMINAMATH_CALUDE_birth_year_problem_l1520_152056


namespace NUMINAMATH_CALUDE_min_value_theorem_l1520_152016

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 1/b = 2) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + 1/y = 2 → 4/x + y ≥ 9/2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + 1/y = 2 ∧ 4/x + y = 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1520_152016


namespace NUMINAMATH_CALUDE_right_triangle_equivalence_l1520_152098

theorem right_triangle_equivalence (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  (a^3 + b^3 + c^3 = a*b*(a+b) - b*c*(b+c) + a*c*(a+c)) ↔
  (a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_equivalence_l1520_152098


namespace NUMINAMATH_CALUDE_mikes_tire_spending_l1520_152081

/-- The problem of calculating Mike's spending on new tires -/
theorem mikes_tire_spending (total_spent : ℚ) (speaker_cost : ℚ) (tire_cost : ℚ) :
  total_spent = 224.87 →
  speaker_cost = 118.54 →
  tire_cost = total_spent - speaker_cost →
  tire_cost = 106.33 := by
  sorry

end NUMINAMATH_CALUDE_mikes_tire_spending_l1520_152081


namespace NUMINAMATH_CALUDE_binomial_18_10_l1520_152074

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1520_152074


namespace NUMINAMATH_CALUDE_min_t_value_l1520_152096

theorem min_t_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 2 * a + b = 1) :
  (∀ t : ℝ, 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1/2) →
  t ≥ Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_t_value_l1520_152096


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1520_152027

theorem polynomial_evaluation : 
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 12 = 0 ∧ x ≠ -2 ∧ x^3 - 3*x^2 - 12*x + 9 = -23 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1520_152027


namespace NUMINAMATH_CALUDE_brittany_age_after_vacation_l1520_152084

/-- Given that Rebecca is 25 years old and Brittany is 3 years older than Rebecca,
    prove that Brittany's age after returning from a 4-year vacation is 32 years old. -/
theorem brittany_age_after_vacation (rebecca_age : ℕ) (age_difference : ℕ) (vacation_duration : ℕ)
  (h1 : rebecca_age = 25)
  (h2 : age_difference = 3)
  (h3 : vacation_duration = 4) :
  rebecca_age + age_difference + vacation_duration = 32 :=
by sorry

end NUMINAMATH_CALUDE_brittany_age_after_vacation_l1520_152084


namespace NUMINAMATH_CALUDE_candy_boxes_problem_l1520_152091

theorem candy_boxes_problem (a b c : ℕ) : 
  a = b + c - 8 → 
  b = a + c - 12 → 
  c = 10 := by
  sorry

end NUMINAMATH_CALUDE_candy_boxes_problem_l1520_152091


namespace NUMINAMATH_CALUDE_bikers_meeting_time_l1520_152051

/-- The time (in minutes) it takes for two bikers to meet again at the starting point of a circular path -/
def meetingTime (t1 t2 : ℕ) : ℕ :=
  Nat.lcm t1 t2

/-- Theorem stating that two bikers with given round completion times will meet at the starting point after a specific time -/
theorem bikers_meeting_time :
  let t1 : ℕ := 12  -- Time for first biker to complete a round
  let t2 : ℕ := 18  -- Time for second biker to complete a round
  meetingTime t1 t2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bikers_meeting_time_l1520_152051


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1520_152050

-- Part 1: System of Equations
theorem solve_system_equations :
  ∃! (x y : ℝ), x - 2*y = 1 ∧ 3*x + 4*y = 9 ∧ x = 2.2 ∧ y = 0.6 :=
by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities :
  ∀ x : ℝ, ((x - 3) / 2 + 3 ≥ x + 1 ∧ 1 - 3*(x - 1) < 8 - x) ↔ (-2 < x ∧ x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1520_152050


namespace NUMINAMATH_CALUDE_unfair_die_expected_value_is_nine_eighths_l1520_152008

def unfair_die_expected_value (p1 p2 p3 p4 p5 : ℚ) : ℚ :=
  let p6 := 1 - (p1 + p2 + p3 + p4 + p5)
  1 * p1 + 2 * p2 + 3 * p3 + 4 * p4 + 5 * p5 + 6 * p6

theorem unfair_die_expected_value_is_nine_eighths :
  unfair_die_expected_value (1/6) (1/8) (1/12) (1/12) (1/12) = 9/8 := by
  sorry

#eval unfair_die_expected_value (1/6) (1/8) (1/12) (1/12) (1/12)

end NUMINAMATH_CALUDE_unfair_die_expected_value_is_nine_eighths_l1520_152008


namespace NUMINAMATH_CALUDE_partition_sum_condition_l1520_152087

def sum_set (s : Finset Nat) : Nat := s.sum id

theorem partition_sum_condition (k : Nat) :
  (∃ (A B : Finset Nat), A ∩ B = ∅ ∧ A ∪ B = Finset.range k ∧ sum_set A = 2 * sum_set B) ↔
  (∃ m : Nat, m > 0 ∧ (k = 3 * m ∨ k = 3 * m - 1)) :=
by sorry

end NUMINAMATH_CALUDE_partition_sum_condition_l1520_152087


namespace NUMINAMATH_CALUDE_round_trip_average_speed_average_speed_approx_31_5_l1520_152069

/-- Calculates the average speed for a round trip with given conditions -/
theorem round_trip_average_speed 
  (total_distance : ℝ) 
  (plain_speed : ℝ) 
  (uphill_increase : ℝ) 
  (uphill_decrease : ℝ) : ℝ :=
  let half_distance := total_distance / 2
  let uphill_speed := plain_speed * (1 + uphill_increase) * (1 - uphill_decrease)
  let plain_time := half_distance / plain_speed
  let uphill_time := half_distance / uphill_speed
  let total_time := plain_time + uphill_time
  total_distance / total_time

/-- Proves that the average speed for the given round trip is approximately 31.5 km/hr -/
theorem average_speed_approx_31_5 : 
  ∃ ε > 0, |round_trip_average_speed 240 30 0.3 0.15 - 31.5| < ε :=
sorry

end NUMINAMATH_CALUDE_round_trip_average_speed_average_speed_approx_31_5_l1520_152069
