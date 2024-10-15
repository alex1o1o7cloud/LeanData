import Mathlib

namespace NUMINAMATH_CALUDE_tree_growth_problem_l2607_260781

/-- A tree growth problem -/
theorem tree_growth_problem (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (target_height : ℝ) :
  initial_height = 5 →
  growth_rate = 3 →
  initial_age = 1 →
  target_height = 23 →
  ∃ (years : ℝ), 
    initial_height + growth_rate * years = target_height ∧
    years + initial_age = 7 :=
by sorry

end NUMINAMATH_CALUDE_tree_growth_problem_l2607_260781


namespace NUMINAMATH_CALUDE_area_of_triangle_MEF_l2607_260700

-- Define the circle P
def circle_P : Real := 10

-- Define the chord EF
def chord_EF : Real := 12

-- Define the segment MQ
def segment_MQ : Real := 20

-- Define the perpendicular distance from P to EF
def perpendicular_distance : Real := 8

-- Theorem statement
theorem area_of_triangle_MEF :
  let radius : Real := circle_P
  let chord_length : Real := chord_EF
  let height : Real := perpendicular_distance
  (1/2 : Real) * chord_length * height = 48 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_MEF_l2607_260700


namespace NUMINAMATH_CALUDE_rabbit_count_l2607_260704

theorem rabbit_count (total_legs : ℕ) (rabbit_chicken_diff : ℕ) : 
  total_legs = 250 → rabbit_chicken_diff = 53 → 
  ∃ (rabbits : ℕ), 
    rabbits + rabbit_chicken_diff = total_legs / 2 ∧
    4 * rabbits + 2 * (rabbits + rabbit_chicken_diff) = total_legs ∧
    rabbits = 24 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_count_l2607_260704


namespace NUMINAMATH_CALUDE_drum_oil_capacity_l2607_260784

theorem drum_oil_capacity (C : ℝ) (Y : ℝ) : 
  C > 0 → -- Capacity of Drum X is positive
  Y ≥ 0 → -- Initial amount of oil in Drum Y is non-negative
  Y + (1/2 * C) = 0.65 * (2 * C) → -- After pouring, Drum Y is filled to 0.65 capacity
  Y = 0.8 * (2 * C) -- Initial fill level of Drum Y is 0.8 of its capacity
  := by sorry

end NUMINAMATH_CALUDE_drum_oil_capacity_l2607_260784


namespace NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l2607_260728

theorem sqrt_sum_fraction_simplification :
  Real.sqrt ((9 : ℝ) / 16 + 16 / 81) = Real.sqrt 985 / 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_fraction_simplification_l2607_260728


namespace NUMINAMATH_CALUDE_estimated_area_is_10_l2607_260725

/-- The function representing the lower bound of the area -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The upper bound of the area -/
def upper_bound : ℝ := 5

/-- The total square area -/
def total_area : ℝ := 16

/-- The total number of experiments -/
def total_experiments : ℕ := 1000

/-- The number of points that fall within the desired area -/
def points_within : ℕ := 625

/-- Theorem stating that the estimated area is 10 -/
theorem estimated_area_is_10 : 
  (total_area * (points_within : ℝ) / total_experiments) = 10 := by
  sorry

end NUMINAMATH_CALUDE_estimated_area_is_10_l2607_260725


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l2607_260742

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 5488 → 
  min a b = 56 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l2607_260742


namespace NUMINAMATH_CALUDE_smallest_cube_ending_432_l2607_260765

theorem smallest_cube_ending_432 : 
  ∀ n : ℕ+, n.val^3 % 1000 = 432 → n.val ≥ 138 := by sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_432_l2607_260765


namespace NUMINAMATH_CALUDE_glove_at_midpoint_l2607_260717

/-- Represents the escalator system and Semyon's movement -/
structure EscalatorSystem where
  /-- The speed of both escalators -/
  escalator_speed : ℝ
  /-- Semyon's walking speed -/
  semyon_speed : ℝ
  /-- The total height of the escalators -/
  total_height : ℝ

/-- Theorem stating that the glove will be at the midpoint when Semyon reaches the top -/
theorem glove_at_midpoint (system : EscalatorSystem)
  (h1 : system.escalator_speed > 0)
  (h2 : system.semyon_speed = system.escalator_speed)
  (h3 : system.total_height > 0) :
  let time_to_top := system.total_height / (2 * system.escalator_speed)
  let glove_position := system.escalator_speed * time_to_top
  glove_position = system.total_height / 2 := by
  sorry


end NUMINAMATH_CALUDE_glove_at_midpoint_l2607_260717


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2607_260745

-- Define the conditions
def p (x : ℝ) : Prop := Real.log (x - 3) < 0
def q (x : ℝ) : Prop := (x - 2) / (x - 4) < 0

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2607_260745


namespace NUMINAMATH_CALUDE_petya_vasya_meet_at_64_l2607_260730

/-- The number of lampposts along the alley -/
def num_lampposts : ℕ := 100

/-- The lamppost number where Petya is observed -/
def petya_observed : ℕ := 22

/-- The lamppost number where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The function to calculate the meeting point of Petya and Vasya -/
def meeting_point : ℕ := sorry

/-- Theorem stating that Petya and Vasya meet at lamppost 64 -/
theorem petya_vasya_meet_at_64 : meeting_point = 64 := by sorry

end NUMINAMATH_CALUDE_petya_vasya_meet_at_64_l2607_260730


namespace NUMINAMATH_CALUDE_z_share_per_x_rupee_l2607_260710

/-- Proof that z gets 0.50 rupees for each rupee x gets --/
theorem z_share_per_x_rupee
  (total : ℝ)
  (y_share : ℝ)
  (y_per_x : ℝ)
  (h_total : total = 156)
  (h_y_share : y_share = 36)
  (h_y_per_x : y_per_x = 0.45)
  : ∃ (z_per_x : ℝ), z_per_x = 0.50 ∧
    ∃ (units : ℝ), units * (1 + y_per_x + z_per_x) = total ∧
                   units * y_per_x = y_share :=
by
  sorry


end NUMINAMATH_CALUDE_z_share_per_x_rupee_l2607_260710


namespace NUMINAMATH_CALUDE_perpendicular_parallel_theorem_l2607_260701

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_parallel_theorem 
  (m n l : Line) (α β γ : Plane) 
  (h1 : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h2 : α ≠ β ∧ α ≠ γ ∧ β ≠ γ) :
  perpendicularLP m α → parallelLP n β → parallelPP α β → perpendicular m n :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_theorem_l2607_260701


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2607_260796

theorem parabola_line_intersection (α : Real) :
  (∃! x, 3 * x^2 + 1 = 4 * Real.sin α * x) →
  0 < α ∧ α < π / 2 →
  α = π / 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2607_260796


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2607_260791

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + m - 12 = 0) → (n^2 + n - 12 = 0) → m^2 + 2*m + n = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2607_260791


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_third_l2607_260757

/-- A cubic function f(x) = ax^3 - x^2 + x - 5 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

/-- The derivative of f(x) with respect to x -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

/-- f is monotonically increasing if its derivative is non-negative for all x -/
def is_monotone_increasing (a : ℝ) : Prop :=
  ∀ x : ℝ, f_deriv a x ≥ 0

theorem monotone_increasing_implies_a_geq_one_third :
  ∀ a : ℝ, is_monotone_increasing a → a ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_one_third_l2607_260757


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2607_260729

theorem least_positive_integer_with_remainders : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 0 ∧ 
  n % 4 = 1 ∧ 
  n % 5 = 2 ∧ 
  ∀ m : ℕ, m > 0 ∧ m % 3 = 0 ∧ m % 4 = 1 ∧ m % 5 = 2 → n ≤ m :=
by
  use 57
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2607_260729


namespace NUMINAMATH_CALUDE_expression_evaluation_l2607_260760

theorem expression_evaluation (a b : ℝ) 
  (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3*b) * (2*a - b) - 2*(a - b)^2 = -23 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2607_260760


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2607_260732

theorem fraction_to_decimal : (21 : ℚ) / 160 = 0.13125 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2607_260732


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l2607_260731

/-- A quadratic function f(x) = ax^2 + bx + c with roots at -2 and 4, and maximum value 54 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_sum_of_coefficients 
  (a b c : ℝ) 
  (h1 : QuadraticFunction a b c (-2) = 0)
  (h2 : QuadraticFunction a b c 4 = 0)
  (h3 : ∀ x, QuadraticFunction a b c x ≤ 54)
  (h4 : ∃ x, QuadraticFunction a b c x = 54) :
  a + b + c = 54 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l2607_260731


namespace NUMINAMATH_CALUDE_percentage_of_360_l2607_260753

theorem percentage_of_360 : (42 : ℝ) / 100 * 360 = 151.2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_l2607_260753


namespace NUMINAMATH_CALUDE_train_length_l2607_260749

/-- Proves that a train passing through a tunnel under specific conditions has a length of 100 meters -/
theorem train_length (tunnel_length : ℝ) (total_time : ℝ) (inside_time : ℝ) 
  (h1 : tunnel_length = 500)
  (h2 : total_time = 30)
  (h3 : inside_time = 20)
  (h4 : total_time > 0)
  (h5 : inside_time > 0)
  (h6 : total_time > inside_time) :
  ∃ (train_length : ℝ), 
    train_length = 100 ∧ 
    (tunnel_length + train_length) / total_time = (tunnel_length - train_length) / inside_time :=
by sorry


end NUMINAMATH_CALUDE_train_length_l2607_260749


namespace NUMINAMATH_CALUDE_larger_number_proof_l2607_260775

theorem larger_number_proof (x y : ℝ) (h1 : x + y = 28) (h2 : x - y = 4) : 
  max x y = 16 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2607_260775


namespace NUMINAMATH_CALUDE_inner_rectangle_side_length_l2607_260740

/-- Given a square with side length a and four congruent right triangles removed from its corners,
    this theorem proves the relationship between the original square's side length,
    the area removed, and the resulting inner rectangle's side length. -/
theorem inner_rectangle_side_length
  (a : ℝ)
  (h1 : a ≥ 24 * Real.sqrt 3)
  (h2 : 6 * (4 * Real.sqrt 3)^2 = 288) :
  a - 24 * Real.sqrt 3 = a - 6 * (4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_inner_rectangle_side_length_l2607_260740


namespace NUMINAMATH_CALUDE_jakes_birdhouse_width_l2607_260774

/-- Sara's birdhouse dimensions in feet -/
def sara_width : ℝ := 1
def sara_height : ℝ := 2
def sara_depth : ℝ := 2

/-- Jake's birdhouse dimensions in inches -/
def jake_height : ℝ := 20
def jake_depth : ℝ := 18

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

/-- Volume difference between Jake's and Sara's birdhouses in cubic inches -/
def volume_difference : ℝ := 1152

/-- Theorem stating that Jake's birdhouse width is 22.4 inches -/
theorem jakes_birdhouse_width :
  ∃ (jake_width : ℝ),
    jake_width * jake_height * jake_depth -
    (sara_width * sara_height * sara_depth * feet_to_inches^3) =
    volume_difference ∧
    jake_width = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_jakes_birdhouse_width_l2607_260774


namespace NUMINAMATH_CALUDE_third_angle_of_triangle_l2607_260714

theorem third_angle_of_triangle (a b c : ℝ) : 
  a + b + c = 180 → a = 50 → b = 80 → c = 50 := by sorry

end NUMINAMATH_CALUDE_third_angle_of_triangle_l2607_260714


namespace NUMINAMATH_CALUDE_points_always_odd_l2607_260772

/-- Represents the number of points on the line after a certain number of operations -/
def num_points (initial : ℕ) (operations : ℕ) : ℕ :=
  if operations = 0 then
    initial
  else
    2 * num_points initial (operations - 1) - 1

/-- Theorem stating that the number of points is always odd after any number of operations -/
theorem points_always_odd (initial : ℕ) (operations : ℕ) :
  Odd (num_points initial operations) :=
by
  sorry


end NUMINAMATH_CALUDE_points_always_odd_l2607_260772


namespace NUMINAMATH_CALUDE_emma_has_eight_l2607_260736

/-- The amount of money each person has -/
structure Money where
  emma : ℝ
  daya : ℝ
  jeff : ℝ
  brenda : ℝ

/-- The conditions of the problem -/
def money_conditions (m : Money) : Prop :=
  m.daya = 1.25 * m.emma ∧
  m.jeff = 0.4 * m.daya ∧
  m.brenda = m.jeff + 4 ∧
  m.brenda = 8

/-- The theorem stating Emma has $8 -/
theorem emma_has_eight (m : Money) (h : money_conditions m) : m.emma = 8 := by
  sorry

end NUMINAMATH_CALUDE_emma_has_eight_l2607_260736


namespace NUMINAMATH_CALUDE_counting_game_result_l2607_260702

/-- Represents the counting game with students in a circle. -/
def CountingGame (n : ℕ) (start : ℕ) (last : ℕ) : Prop :=
  ∃ (process : ℕ → ℕ → ℕ), 
    (process 0 start = last) ∧ 
    (∀ k, k > 0 → process k start ≠ last → 
      process (k+1) start = process k (((process k start + 2) % n) + 1))

/-- The main theorem stating that if student 37 is the last remaining
    in a circle of 40 students, then the initial student was number 5. -/
theorem counting_game_result : CountingGame 40 5 37 := by
  sorry


end NUMINAMATH_CALUDE_counting_game_result_l2607_260702


namespace NUMINAMATH_CALUDE_assignments_count_l2607_260786

/-- The number of assignments graded per hour initially -/
def initial_rate : ℕ := 6

/-- The number of assignments graded per hour after the change -/
def changed_rate : ℕ := 8

/-- The number of hours spent grading at the initial rate -/
def initial_hours : ℕ := 2

/-- The number of hours saved compared to the original plan -/
def hours_saved : ℕ := 3

/-- The total number of assignments in the batch -/
def total_assignments : ℕ := 84

/-- Theorem stating that the total number of assignments is 84 -/
theorem assignments_count :
  ∃ (x : ℕ), 
    (initial_rate * x = total_assignments) ∧ 
    (initial_rate * initial_hours + changed_rate * (x - initial_hours - hours_saved) = total_assignments) := by
  sorry

end NUMINAMATH_CALUDE_assignments_count_l2607_260786


namespace NUMINAMATH_CALUDE_average_weight_of_boys_l2607_260752

theorem average_weight_of_boys (group1_count : ℕ) (group1_avg : ℚ) 
  (group2_count : ℕ) (group2_avg : ℚ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_count := group1_count + group2_count
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  total_weight / total_count = 48.55 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_boys_l2607_260752


namespace NUMINAMATH_CALUDE_zorg_game_threshold_l2607_260712

theorem zorg_game_threshold : ∃ (n : ℕ), n = 40 ∧ ∀ (m : ℕ), m < n → (m * (m + 1)) / 2 ≤ 20 * m :=
by sorry

end NUMINAMATH_CALUDE_zorg_game_threshold_l2607_260712


namespace NUMINAMATH_CALUDE_vases_to_arrange_l2607_260741

/-- Proves that the number of vases of flowers to be arranged is 256,
    given that Jane can arrange 16 vases per day and needs 16 days to finish all arrangements. -/
theorem vases_to_arrange (vases_per_day : ℕ) (days_needed : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days_needed = 16) : 
  vases_per_day * days_needed = 256 := by
  sorry

end NUMINAMATH_CALUDE_vases_to_arrange_l2607_260741


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2607_260773

theorem imaginary_part_of_complex_expression (i : ℂ) (h : i^2 = -1) :
  Complex.im ((3 + i) / i^2 * i) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_expression_l2607_260773


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2607_260743

theorem x_squared_plus_reciprocal (x : ℝ) (h : x ≠ 0) :
  x^4 + 1/x^4 = 47 → x^2 + 1/x^2 = 7 := by sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l2607_260743


namespace NUMINAMATH_CALUDE_subtraction_sum_l2607_260768

/-- Given a subtraction problem with digits K, L, M, and N, prove that their sum is 20 -/
theorem subtraction_sum (K L M N : Nat) : 
  (K < 10) → (L < 10) → (M < 10) → (N < 10) →
  (5000 + 100 * K + 30 + L) - (1000 * M + 400 + 10 * N + 1) = 4451 →
  K + L + M + N = 20 := by
sorry

end NUMINAMATH_CALUDE_subtraction_sum_l2607_260768


namespace NUMINAMATH_CALUDE_no_solution_iff_k_eq_two_l2607_260793

-- Define the equation
def equation (x k : ℝ) : Prop :=
  (x + 2) / (x - 3) = (x - k) / (x - 7)

-- Define the domain restriction
def valid_domain (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≠ 7

-- Theorem statement
theorem no_solution_iff_k_eq_two :
  ∀ k : ℝ, (∀ x : ℝ, valid_domain x → ¬equation x k) ↔ k = 2 :=
by sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_eq_two_l2607_260793


namespace NUMINAMATH_CALUDE_total_students_is_880_l2607_260763

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The fraction of students enrolled in biology classes -/
def biology_enrollment_rate : ℚ := 35 / 100

/-- The number of students not enrolled in a biology class -/
def students_not_in_biology : ℕ := 572

/-- Theorem stating that the total number of students is 880 -/
theorem total_students_is_880 :
  (1 - biology_enrollment_rate) * total_students = students_not_in_biology :=
sorry

end NUMINAMATH_CALUDE_total_students_is_880_l2607_260763


namespace NUMINAMATH_CALUDE_sin_cos_product_l2607_260787

theorem sin_cos_product (a : ℝ) (h : Real.sin (Real.pi - a) = -2 * Real.sin (Real.pi / 2 + a)) :
  Real.sin a * Real.cos a = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_product_l2607_260787


namespace NUMINAMATH_CALUDE_trig_identity_l2607_260789

open Real

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : a ≠ 0 ∧ b ≠ 0) :
  (sin θ ^ 6 / a ^ 2 + cos θ ^ 6 / b ^ 2 = 1 / (a ^ 2 + b ^ 2)) →
  (sin θ ^ 12 / a ^ 5 + cos θ ^ 12 / b ^ 5 = 1 / a ^ 5) :=
by sorry

end NUMINAMATH_CALUDE_trig_identity_l2607_260789


namespace NUMINAMATH_CALUDE_equation_solution_l2607_260799

theorem equation_solution : ∃ x : ℚ, (5 * (x + 30) / 3 = (4 - 3 * x) / 7) ∧ (x = -519 / 22) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2607_260799


namespace NUMINAMATH_CALUDE_three_boxes_of_five_balls_l2607_260719

/-- Calculates the total number of balls given the number of boxes and balls per box -/
def totalBalls (numBoxes : ℕ) (ballsPerBox : ℕ) : ℕ :=
  numBoxes * ballsPerBox

/-- Proves that the total number of balls is 15 when there are 3 boxes with 5 balls each -/
theorem three_boxes_of_five_balls :
  totalBalls 3 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_three_boxes_of_five_balls_l2607_260719


namespace NUMINAMATH_CALUDE_cat_food_sale_theorem_l2607_260735

/-- Calculates the total number of cat food cases sold during a sale. -/
def total_cases_sold (first_group : Nat) (second_group : Nat) (third_group : Nat)
  (first_group_cases : Nat) (second_group_cases : Nat) (third_group_cases : Nat) : Nat :=
  first_group * first_group_cases + second_group * second_group_cases + third_group * third_group_cases

/-- Proves that the total number of cat food cases sold is 40 given the specified customer groups and their purchases. -/
theorem cat_food_sale_theorem :
  total_cases_sold 8 4 8 3 2 1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_cat_food_sale_theorem_l2607_260735


namespace NUMINAMATH_CALUDE_increasing_magnitude_l2607_260723

theorem increasing_magnitude (x : ℝ) (h : 1 < x ∧ x < 1.1) : x < x^x ∧ x^x < x^(x^x) := by
  sorry

end NUMINAMATH_CALUDE_increasing_magnitude_l2607_260723


namespace NUMINAMATH_CALUDE_prize_distribution_l2607_260754

theorem prize_distribution (total_winners : ℕ) (min_award : ℚ) (max_award : ℚ) :
  total_winners = 15 →
  min_award = 15 →
  max_award = 285 →
  ∃ (total_prize : ℚ),
    (2 / 5 : ℚ) * total_prize = max_award * ((3 / 5 : ℚ) * total_winners) ∧
    total_prize = 6502.5 :=
by sorry

end NUMINAMATH_CALUDE_prize_distribution_l2607_260754


namespace NUMINAMATH_CALUDE_three_integers_sum_l2607_260738

theorem three_integers_sum (a b c : ℕ) : 
  a > 1 → b > 1 → c > 1 →
  a * b * c = 216000 →
  Nat.gcd a b = 1 → Nat.gcd a c = 1 → Nat.gcd b c = 1 →
  a + b + c = 184 :=
by sorry

end NUMINAMATH_CALUDE_three_integers_sum_l2607_260738


namespace NUMINAMATH_CALUDE_complement_of_A_l2607_260733

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x ≥ 1} ∪ {x | x ≤ 0}

theorem complement_of_A (x : ℝ) : x ∈ Aᶜ ↔ 0 < x ∧ x < 1 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2607_260733


namespace NUMINAMATH_CALUDE_product_sequence_value_l2607_260759

theorem product_sequence_value : 
  (1 / 3) * (9 / 1) * (1 / 27) * (81 / 1) * (1 / 243) * (729 / 1) * (1 / 729) * (2187 / 1) = 729 := by
  sorry

end NUMINAMATH_CALUDE_product_sequence_value_l2607_260759


namespace NUMINAMATH_CALUDE_equation_solution_denominator_never_zero_l2607_260708

theorem equation_solution (x : ℝ) : 
  (x + 5) / (x^2 + 4*x + 10) = 0 ↔ x = -5 :=
by sorry

theorem denominator_never_zero (x : ℝ) : 
  x^2 + 4*x + 10 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_denominator_never_zero_l2607_260708


namespace NUMINAMATH_CALUDE_remainder_2_pow_13_mod_3_l2607_260718

theorem remainder_2_pow_13_mod_3 : 2^13 ≡ 2 [ZMOD 3] := by sorry

end NUMINAMATH_CALUDE_remainder_2_pow_13_mod_3_l2607_260718


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2607_260767

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2607_260767


namespace NUMINAMATH_CALUDE_janice_earnings_this_week_l2607_260713

/-- Calculates Janice's weekly earnings based on her work schedule and wages -/
def janice_weekly_earnings (regular_days : ℕ) (regular_wage : ℕ) (overtime_shifts : ℕ) (overtime_bonus : ℕ) : ℕ :=
  regular_days * regular_wage + overtime_shifts * overtime_bonus

/-- Proves that Janice's weekly earnings are $195 given her work schedule -/
theorem janice_earnings_this_week :
  janice_weekly_earnings 5 30 3 15 = 195 := by
  sorry

#eval janice_weekly_earnings 5 30 3 15

end NUMINAMATH_CALUDE_janice_earnings_this_week_l2607_260713


namespace NUMINAMATH_CALUDE_range_of_m_l2607_260758

/-- The curve equation -/
def curve (x y m : ℝ) : Prop := x^2 + y^2 + y + m = 0

/-- The symmetry line equation -/
def symmetry_line (x y : ℝ) : Prop := x + 2*y - 1 = 0

/-- Predicate for having four common tangents -/
def has_four_common_tangents (m : ℝ) : Prop := sorry

/-- Theorem stating the range of m -/
theorem range_of_m : 
  ∀ m : ℝ, (∀ x y : ℝ, curve x y m → ∃ x' y' : ℝ, symmetry_line x' y' ∧ has_four_common_tangents m) 
  ↔ -11/20 < m ∧ m < 1/4 := by sorry

end NUMINAMATH_CALUDE_range_of_m_l2607_260758


namespace NUMINAMATH_CALUDE_parallelepiped_count_l2607_260762

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a set of four points in 3D space -/
structure FourPoints where
  p1 : Point3D
  p2 : Point3D
  p3 : Point3D
  p4 : Point3D

/-- Predicate to check if four points are coplanar -/
def areCoplanar (points : FourPoints) : Prop :=
  ∃ (a b c d : ℝ), a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∧
    a * points.p1.x + b * points.p1.y + c * points.p1.z + d = 0 ∧
    a * points.p2.x + b * points.p2.y + c * points.p2.z + d = 0 ∧
    a * points.p3.x + b * points.p3.y + c * points.p3.z + d = 0 ∧
    a * points.p4.x + b * points.p4.y + c * points.p4.z + d = 0

/-- Function to count the number of distinct parallelepipeds -/
def countParallelepipeds (points : FourPoints) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the number of distinct parallelepipeds is 29 -/
theorem parallelepiped_count (points : FourPoints) 
  (h : ¬ areCoplanar points) : countParallelepipeds points = 29 := by
  sorry

end NUMINAMATH_CALUDE_parallelepiped_count_l2607_260762


namespace NUMINAMATH_CALUDE_power_between_n_and_2n_smallest_m_s_for_2_and_3_l2607_260748

theorem power_between_n_and_2n (s : ℕ) (hs : s > 1) :
  ∃ (m_s : ℕ), ∀ (n : ℕ), n ≥ m_s → ∃ (k : ℕ), n < k^s ∧ k^s < 2*n :=
by sorry

theorem smallest_m_s_for_2_and_3 :
  (∃ (m_2 : ℕ), ∀ (n : ℕ), n ≥ m_2 → ∃ (k : ℕ), n < k^2 ∧ k^2 < 2*n) ∧
  (∃ (m_3 : ℕ), ∀ (n : ℕ), n ≥ m_3 → ∃ (k : ℕ), n < k^3 ∧ k^3 < 2*n) ∧
  (∀ (m_2' : ℕ), m_2' < 5 → ∃ (n : ℕ), n ≥ m_2' ∧ ∀ (k : ℕ), n ≥ k^2 ∨ k^2 ≥ 2*n) ∧
  (∀ (m_3' : ℕ), m_3' < 33 → ∃ (n : ℕ), n ≥ m_3' ∧ ∀ (k : ℕ), n ≥ k^3 ∨ k^3 ≥ 2*n) :=
by sorry

end NUMINAMATH_CALUDE_power_between_n_and_2n_smallest_m_s_for_2_and_3_l2607_260748


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2607_260788

theorem imaginary_part_of_z (z : ℂ) : 
  z = Complex.I * (3 - 2 * Complex.I) * Complex.I ∧ z.re = 0 → z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2607_260788


namespace NUMINAMATH_CALUDE_projection_implies_y_value_l2607_260778

/-- Given vectors v and w', if the projection of v on w' is proj_v_w, then y = -11/3 -/
theorem projection_implies_y_value (v w' proj_v_w : ℝ × ℝ) (y : ℝ) :
  v = (1, y) →
  w' = (-3, 1) →
  proj_v_w = (2, -2/3) →
  proj_v_w = (((v.1 * w'.1 + v.2 * w'.2) / (w'.1 ^ 2 + w'.2 ^ 2)) * w'.1,
              ((v.1 * w'.1 + v.2 * w'.2) / (w'.1 ^ 2 + w'.2 ^ 2)) * w'.2) →
  y = -11/3 := by
sorry

end NUMINAMATH_CALUDE_projection_implies_y_value_l2607_260778


namespace NUMINAMATH_CALUDE_profit_difference_A_C_l2607_260769

-- Define the profit-sharing ratios
def ratio_A : ℕ := 3
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6
def ratio_D : ℕ := 7

-- Define B's profit share
def profit_B : ℕ := 2000

-- Theorem statement
theorem profit_difference_A_C : 
  let part_value : ℚ := profit_B / ratio_B
  let profit_A : ℚ := part_value * ratio_A
  let profit_C : ℚ := part_value * ratio_C
  profit_C - profit_A = 1200 := by sorry

end NUMINAMATH_CALUDE_profit_difference_A_C_l2607_260769


namespace NUMINAMATH_CALUDE_sine_graph_shift_l2607_260776

theorem sine_graph_shift (x : ℝ) :
  3 * Real.sin (1/2 * (x - 4*π/5) + π/5) = 3 * Real.sin (1/2 * x - π/5) := by
  sorry

end NUMINAMATH_CALUDE_sine_graph_shift_l2607_260776


namespace NUMINAMATH_CALUDE_binomial_coefficient_x3y5_in_x_plus_y_8_l2607_260766

theorem binomial_coefficient_x3y5_in_x_plus_y_8 :
  (Finset.range 9).sum (fun k => (Nat.choose 8 k) * (1 : ℕ)^k * (1 : ℕ)^(8 - k)) = 256 ∧
  (Nat.choose 8 3) = 56 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_x3y5_in_x_plus_y_8_l2607_260766


namespace NUMINAMATH_CALUDE_basis_vectors_classification_l2607_260777

def is_basis (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 ≠ 0 ∧ v₁ ≠ (0, 0) ∧ v₂ ≠ (0, 0)

theorem basis_vectors_classification :
  let a₁ : ℝ × ℝ := (0, 0)
  let a₂ : ℝ × ℝ := (1, 2)
  let b₁ : ℝ × ℝ := (2, -1)
  let b₂ : ℝ × ℝ := (1, 2)
  let c₁ : ℝ × ℝ := (-1, -2)
  let c₂ : ℝ × ℝ := (1, 2)
  let d₁ : ℝ × ℝ := (1, 1)
  let d₂ : ℝ × ℝ := (1, 2)
  ¬(is_basis a₁ a₂) ∧
  ¬(is_basis c₁ c₂) ∧
  (is_basis b₁ b₂) ∧
  (is_basis d₁ d₂) :=
by sorry

end NUMINAMATH_CALUDE_basis_vectors_classification_l2607_260777


namespace NUMINAMATH_CALUDE_employee_y_pay_l2607_260744

/-- Represents the weekly pay of employees x, y, and z -/
structure EmployeePay where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total pay for all employees -/
def totalPay (pay : EmployeePay) : ℝ :=
  pay.x + pay.y + pay.z

/-- Theorem: Given the conditions, employee y's pay is 478.125 -/
theorem employee_y_pay :
  ∀ (pay : EmployeePay),
    totalPay pay = 1550 →
    pay.x = 1.2 * pay.y →
    pay.z = pay.y - 30 + 50 →
    pay.y = 478.125 := by
  sorry


end NUMINAMATH_CALUDE_employee_y_pay_l2607_260744


namespace NUMINAMATH_CALUDE_employee_pays_216_l2607_260785

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the store markup percentage
def store_markup : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount : ℝ := 0.10

-- Calculate the retail price
def retail_price : ℝ := wholesale_cost * (1 + store_markup)

-- Calculate the employee's final price
def employee_price : ℝ := retail_price * (1 - employee_discount)

-- Theorem to prove
theorem employee_pays_216 : employee_price = 216 := by sorry

end NUMINAMATH_CALUDE_employee_pays_216_l2607_260785


namespace NUMINAMATH_CALUDE_base5_sum_theorem_l2607_260780

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a base-5 representation to base-10 -/
def fromBase5 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two base-5 numbers represented as lists -/
def addBase5 (a b : List ℕ) : List ℕ :=
  sorry

theorem base5_sum_theorem :
  let a := toBase5 259
  let b := toBase5 63
  addBase5 a b = [2, 2, 4, 2] := by sorry

end NUMINAMATH_CALUDE_base5_sum_theorem_l2607_260780


namespace NUMINAMATH_CALUDE_bryan_total_books_l2607_260703

/-- The number of books in each of Bryan's bookshelves -/
def books_per_shelf : ℕ := 27

/-- The number of bookshelves Bryan has -/
def number_of_shelves : ℕ := 23

/-- The total number of books Bryan has -/
def total_books : ℕ := books_per_shelf * number_of_shelves

theorem bryan_total_books : total_books = 621 := by
  sorry

end NUMINAMATH_CALUDE_bryan_total_books_l2607_260703


namespace NUMINAMATH_CALUDE_odd_function_through_points_l2607_260783

/-- An odd function passing through two specific points -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

theorem odd_function_through_points :
  (∀ x, f a b c (-x) = -(f a b c x)) →
  f a b c (-Real.sqrt 2) = Real.sqrt 2 →
  f a b c (2 * Real.sqrt 2) = 10 * Real.sqrt 2 →
  ∃ g : ℝ → ℝ, (∀ x, g x = x^3 - 3*x) ∧
              (∀ x, f a b c x = g x) ∧
              (∀ m, (∃! x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ g x₁ + m = 0 ∧ g x₂ + m = 0 ∧ g x₃ + m = 0) ↔
                    -2 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_odd_function_through_points_l2607_260783


namespace NUMINAMATH_CALUDE_complex_function_property_l2607_260707

/-- A complex function f(z) = (a+bi)z with certain properties -/
def f (a b : ℝ) (z : ℂ) : ℂ := (Complex.mk a b) * z

/-- The theorem statement -/
theorem complex_function_property (a b c : ℝ) :
  (a > 0) →
  (b > 0) →
  (c > 0) →
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z - Complex.I * c)) →
  (Complex.abs (Complex.mk a b) = 9) →
  (b^2 = 323/4) := by
  sorry

end NUMINAMATH_CALUDE_complex_function_property_l2607_260707


namespace NUMINAMATH_CALUDE_total_birds_in_marsh_l2607_260720

theorem total_birds_in_marsh (geese ducks swans : ℕ) 
  (h1 : geese = 58) 
  (h2 : ducks = 37) 
  (h3 : swans = 42) : 
  geese + ducks + swans = 137 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_marsh_l2607_260720


namespace NUMINAMATH_CALUDE_angle_increase_in_equilateral_triangle_l2607_260771

/-- 
Given an equilateral triangle where each angle initially measures 60 degrees,
if one angle is increased by 40 degrees, the resulting measure of that angle is 100 degrees.
-/
theorem angle_increase_in_equilateral_triangle :
  ∀ (A B C : ℝ),
  A = 60 ∧ B = 60 ∧ C = 60 →  -- Initially equilateral triangle
  (C + 40 : ℝ) = 100 :=
by sorry

end NUMINAMATH_CALUDE_angle_increase_in_equilateral_triangle_l2607_260771


namespace NUMINAMATH_CALUDE_conditions_on_m_l2607_260746

/-- The set A defined by the quadratic equation mx² - 2x + 1 = 0 -/
def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 - 2 * x + 1 = 0}

/-- Theorem stating the conditions on m for different properties of set A -/
theorem conditions_on_m :
  (∀ m : ℝ, A m = ∅ ↔ m > 1) ∧
  (∀ m : ℝ, (∃ x : ℝ, A m = {x}) ↔ m = 0 ∨ m = 1) ∧
  (∀ m : ℝ, (∃ x : ℝ, x ∈ A m ∧ x > 1/2 ∧ x < 2) ↔ m > 0 ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_conditions_on_m_l2607_260746


namespace NUMINAMATH_CALUDE_sum_of_digits_of_sum_f_equals_8064_l2607_260724

-- Define the function f
def f (k : ℕ) : ℕ :=
  -- The smallest positive integer not written on the blackboard
  -- after the process described in the problem
  sorry

-- Define the sum of f(2k) from k=1 to 1008
def sum_f : ℕ :=
  (Finset.range 1008).sum (λ k => f (2 * (k + 1)))

-- Define a function to calculate the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

-- The main theorem
theorem sum_of_digits_of_sum_f_equals_8064 :
  sum_of_digits sum_f = 8064 :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_sum_f_equals_8064_l2607_260724


namespace NUMINAMATH_CALUDE_cricket_captain_age_l2607_260751

theorem cricket_captain_age (team_size : ℕ) (captain_age wicket_keeper_age : ℕ) 
  (team_average : ℚ) (remaining_average : ℚ) :
  team_size = 11 →
  wicket_keeper_age = captain_age + 3 →
  team_average = 25 →
  remaining_average = team_average - 1 →
  (team_size : ℚ) * team_average = 
    (team_size - 2 : ℚ) * remaining_average + captain_age + wicket_keeper_age →
  captain_age = 28 := by
  sorry

end NUMINAMATH_CALUDE_cricket_captain_age_l2607_260751


namespace NUMINAMATH_CALUDE_right_triangle_set_l2607_260722

theorem right_triangle_set : ∃! (a b c : ℝ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 6 ∧ b = 8 ∧ c = 12) ∨ 
   (a = Real.sqrt 3 ∧ b = Real.sqrt 4 ∧ c = Real.sqrt 5)) ∧
  a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2607_260722


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l2607_260756

/-- Represents the price and volume of orangeade on two consecutive days -/
structure Orangeade where
  orange_juice : ℝ  -- Amount of orange juice (same for both days)
  water_day1 : ℝ    -- Amount of water on day 1
  water_day2 : ℝ    -- Amount of water on day 2
  price_day1 : ℝ    -- Price per glass on day 1
  price_day2 : ℝ    -- Price per glass on day 2
  revenue : ℝ        -- Revenue (same for both days)

/-- The price per glass on the second day is $0.20 given the conditions -/
theorem orangeade_price_day2 (o : Orangeade)
    (h1 : o.orange_juice = o.water_day1)
    (h2 : o.water_day2 = 2 * o.water_day1)
    (h3 : o.price_day1 = 0.30)
    (h4 : o.revenue = (o.orange_juice + o.water_day1) * o.price_day1)
    (h5 : o.revenue = (o.orange_juice + o.water_day2) * o.price_day2) :
  o.price_day2 = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_orangeade_price_day2_l2607_260756


namespace NUMINAMATH_CALUDE_langsley_commute_time_l2607_260726

def pickup_time : Nat := 6 * 60  -- 6:00 a.m. in minutes since midnight
def first_station_travel_time : Nat := 40  -- 40 minutes
def work_arrival_time : Nat := 9 * 60  -- 9:00 a.m. in minutes since midnight

theorem langsley_commute_time :
  work_arrival_time - (pickup_time + first_station_travel_time) = 140 := by
  sorry

end NUMINAMATH_CALUDE_langsley_commute_time_l2607_260726


namespace NUMINAMATH_CALUDE_ratio_nature_l2607_260797

theorem ratio_nature (x : ℝ) (m n : ℝ) (hx : x > 0) (hmn : m * n ≠ 0) (hineq : m * x > n * x + n) :
  m / (m + n) = (x + 1) / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_ratio_nature_l2607_260797


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_for_gas_mixture_l2607_260770

/-- Represents different types of statistical charts -/
inductive StatChart
  | PieChart
  | LineChart
  | BarChart
  deriving Repr

/-- Represents a mixture of gases -/
structure GasMixture where
  components : List String
  proportions : List Float
  sum_to_one : proportions.sum = 1

/-- Determines if a chart type is suitable for representing a gas mixture -/
def is_suitable_chart (chart : StatChart) (mixture : GasMixture) : Prop :=
  match chart with
  | StatChart.PieChart => 
      mixture.components.length > 1 ∧ 
      mixture.proportions.all (λ p => p ≥ 0 ∧ p ≤ 1)
  | _ => False

/-- Theorem stating that a pie chart is the most suitable for representing a gas mixture -/
theorem pie_chart_most_suitable_for_gas_mixture (mixture : GasMixture) :
  ∀ (chart : StatChart), is_suitable_chart chart mixture → chart = StatChart.PieChart :=
by sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_for_gas_mixture_l2607_260770


namespace NUMINAMATH_CALUDE_pancakes_theorem_l2607_260782

/-- The number of pancakes left after Bobby and his dog eat some. -/
def pancakes_left (total : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) : ℕ :=
  total - (bobby_ate + dog_ate)

/-- Theorem: Given 21 pancakes, if Bobby eats 5 and his dog eats 7, there are 9 pancakes left. -/
theorem pancakes_theorem : pancakes_left 21 5 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pancakes_theorem_l2607_260782


namespace NUMINAMATH_CALUDE_initial_number_solution_l2607_260727

theorem initial_number_solution : ∃ x : ℤ, x - 12 * 3 * 2 = 9938 ∧ x = 10010 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_solution_l2607_260727


namespace NUMINAMATH_CALUDE_number_satisfying_condition_l2607_260764

theorem number_satisfying_condition (x : ℝ) : x = 40 ↔ 0.65 * x = 0.05 * 60 + 23 := by
  sorry

end NUMINAMATH_CALUDE_number_satisfying_condition_l2607_260764


namespace NUMINAMATH_CALUDE_system_A_is_valid_other_systems_not_valid_l2607_260711

/-- A linear equation in two variables is of the form ax + by = c, where a, b, and c are constants. -/
structure LinearEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A system of two linear equations. -/
structure LinearSystem where
  eq1 : LinearEquation
  eq2 : LinearEquation

/-- The system of equations from option A. -/
def systemA : LinearSystem := {
  eq1 := { a := 1, b := 0, c := 2 }
  eq2 := { a := 0, b := 1, c := 7 }
}

/-- Predicate to check if a given system is a valid system of two linear equations. -/
def isValidLinearSystem (s : LinearSystem) : Prop :=
  -- Additional conditions can be added here if needed
  True

theorem system_A_is_valid : isValidLinearSystem systemA := by
  sorry

/-- The other systems (B, C, D) are not valid systems of two linear equations. -/
theorem other_systems_not_valid :
  ∃ (systemB systemC systemD : LinearSystem),
    ¬ isValidLinearSystem systemB ∧
    ¬ isValidLinearSystem systemC ∧
    ¬ isValidLinearSystem systemD := by
  sorry

end NUMINAMATH_CALUDE_system_A_is_valid_other_systems_not_valid_l2607_260711


namespace NUMINAMATH_CALUDE_fake_coin_identification_l2607_260779

/-- Represents a weighing strategy for identifying a fake coin. -/
structure WeighingStrategy where
  /-- The number of weighings performed. -/
  num_weighings : ℕ
  /-- The maximum number of times any single coin is weighed. -/
  max_weighs_per_coin : ℕ

/-- Represents the problem of identifying a fake coin among a set of coins. -/
structure FakeCoinProblem where
  /-- The total number of coins. -/
  total_coins : ℕ
  /-- The number of fake coins. -/
  num_fake_coins : ℕ
  /-- Indicates whether the fake coin is lighter than the genuine coins. -/
  fake_is_lighter : Bool

/-- Theorem stating that the fake coin can be identified within the given constraints. -/
theorem fake_coin_identification
  (problem : FakeCoinProblem)
  (strategy : WeighingStrategy) :
  problem.total_coins = 99 →
  problem.num_fake_coins = 1 →
  problem.fake_is_lighter = true →
  strategy.num_weighings ≤ 7 →
  strategy.max_weighs_per_coin ≤ 2 →
  ∃ (identification_method : Unit), True :=
by
  sorry

end NUMINAMATH_CALUDE_fake_coin_identification_l2607_260779


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_in_range_l2607_260709

open Set

def A (a : ℝ) : Set ℝ := {x | |x - a| < 2}
def B : Set ℝ := {x | (2*x - 1) / (x + 2) < 1}

theorem intersection_equality_implies_a_in_range (a : ℝ) :
  A a ∩ B = A a → a ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_in_range_l2607_260709


namespace NUMINAMATH_CALUDE_sum_of_composite_function_l2607_260794

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x + 1

def xValues : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_composite_function :
  (xValues.map (λ x => q (p x))).sum = -13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_composite_function_l2607_260794


namespace NUMINAMATH_CALUDE_square_sum_equals_73_l2607_260706

theorem square_sum_equals_73 (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 24) : 
  a^2 + b^2 = 73 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_73_l2607_260706


namespace NUMINAMATH_CALUDE_marcus_pebbles_l2607_260798

theorem marcus_pebbles (initial_pebbles : ℕ) (current_pebbles : ℕ) 
  (h1 : initial_pebbles = 18)
  (h2 : current_pebbles = 39) :
  current_pebbles - (initial_pebbles - initial_pebbles / 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pebbles_l2607_260798


namespace NUMINAMATH_CALUDE_congruent_count_l2607_260792

theorem congruent_count (n : ℕ) : 
  (Finset.filter (fun x => x % 7 = 1) (Finset.range 251)).card = 36 := by
  sorry

end NUMINAMATH_CALUDE_congruent_count_l2607_260792


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2607_260761

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 9) (h_rel : y = 2 * x) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 6 :=
by sorry

theorem min_value_achievable (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x + y + z = 9) (h_rel : y = 2 * x) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ x₀ + y₀ + z₀ = 9 ∧ y₀ = 2 * x₀ ∧
    (x₀^2 + y₀^2) / (x₀ + y₀) + (x₀^2 + z₀^2) / (x₀ + z₀) + (y₀^2 + z₀^2) / (y₀ + z₀) = 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l2607_260761


namespace NUMINAMATH_CALUDE_smallest_n_with_conditions_n_satisfies_conditions_l2607_260716

def has_digit (n : ℕ) (d : ℕ) : Prop :=
  ∃ k m : ℕ, n = 10 * k + d + 10 * m

def is_terminating_decimal (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = 2^a * 5^b

theorem smallest_n_with_conditions :
  ∀ n : ℕ, n > 0 →
    (is_terminating_decimal n ∧ has_digit n 9 ∧ has_digit n 2) →
    n ≥ 524288 :=
by sorry

theorem n_satisfies_conditions :
  is_terminating_decimal 524288 ∧ has_digit 524288 9 ∧ has_digit 524288 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_with_conditions_n_satisfies_conditions_l2607_260716


namespace NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l2607_260737

theorem quadratic_trinomial_factorization (p q : ℝ) (x : ℝ) :
  x^2 + (p + q)*x + p*q = (x + p)*(x + q) := by sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_factorization_l2607_260737


namespace NUMINAMATH_CALUDE_tangent_circles_count_l2607_260705

-- Define a circle in a plane
structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

-- Define the property of two circles being tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define the property of a circle being tangent to two other circles
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

-- State the theorem
theorem tangent_circles_count 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 2) 
  (h2 : c2.radius = 2) 
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), 
    (∀ c ∈ s, c.radius = 4 ∧ is_tangent_to_both c c1 c2) ∧ 
    s.card = 6 :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l2607_260705


namespace NUMINAMATH_CALUDE_decimal_place_150_of_5_over_8_l2607_260795

theorem decimal_place_150_of_5_over_8 : 
  let decimal_expansion := (5 : ℚ) / 8
  let digit_at_n (q : ℚ) (n : ℕ) := (q * 10^n).floor % 10
  digit_at_n decimal_expansion 150 = 0 := by
  sorry

end NUMINAMATH_CALUDE_decimal_place_150_of_5_over_8_l2607_260795


namespace NUMINAMATH_CALUDE_problem_solution_l2607_260750

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a*x

theorem problem_solution (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2607_260750


namespace NUMINAMATH_CALUDE_peter_class_size_l2607_260721

/-- The number of students in Peter's class with 2 hands each -/
def students_with_two_hands : ℕ := 10

/-- The number of students in Peter's class with 1 hand each -/
def students_with_one_hand : ℕ := 3

/-- The number of students in Peter's class with 3 hands each -/
def students_with_three_hands : ℕ := 1

/-- The total number of hands in the class excluding Peter's -/
def total_hands_excluding_peter : ℕ := 20

/-- The number of hands Peter has (assumed to be typical) -/
def peter_hands : ℕ := 2

/-- The total number of students in Peter's class, including Peter -/
def total_students : ℕ := 14

theorem peter_class_size :
  (students_with_two_hands * 2 + 
   students_with_one_hand * 1 + 
   students_with_three_hands * 3 + 
   peter_hands) / 2 = total_students := by sorry

end NUMINAMATH_CALUDE_peter_class_size_l2607_260721


namespace NUMINAMATH_CALUDE_eulers_criterion_l2607_260734

theorem eulers_criterion (p : Nat) (a : Nat) (h_prime : Nat.Prime p) (h_p : p > 2) (h_a : 1 ≤ a ∧ a ≤ p - 1) :
  (∃ x : Nat, x ^ 2 % p = a % p) ↔ a ^ ((p - 1) / 2) % p = 1 := by
  sorry

end NUMINAMATH_CALUDE_eulers_criterion_l2607_260734


namespace NUMINAMATH_CALUDE_n_minus_m_range_l2607_260715

noncomputable def f (x : ℝ) : ℝ := 
  if x > 0 then Real.exp x - 1 else 3/2 * x + 1

theorem n_minus_m_range (m n : ℝ) (h1 : m < n) (h2 : f m = f n) : 
  2/3 < n - m ∧ n - m ≤ Real.log (3/2) + 1/3 := by sorry

end NUMINAMATH_CALUDE_n_minus_m_range_l2607_260715


namespace NUMINAMATH_CALUDE_sum_and_product_equality_l2607_260739

theorem sum_and_product_equality : 2357 + 3572 + 5723 + 7235 * 2 = 26122 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_product_equality_l2607_260739


namespace NUMINAMATH_CALUDE_independence_of_beta_l2607_260790

theorem independence_of_beta (α β : ℝ) : 
  ∃ (f : ℝ → ℝ), ∀ β, 
    (Real.sin (α + β))^2 + (Real.sin (β - α))^2 - 
    2 * Real.sin (α + β) * Real.sin (β - α) * Real.cos (2 * α) = f α :=
by sorry

end NUMINAMATH_CALUDE_independence_of_beta_l2607_260790


namespace NUMINAMATH_CALUDE_root_zero_implies_a_half_l2607_260747

theorem root_zero_implies_a_half (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + 2*a - 1 = 0 ∧ x = 0) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_zero_implies_a_half_l2607_260747


namespace NUMINAMATH_CALUDE_all_good_numbers_less_than_1000_l2607_260755

def isGood (n : ℕ) : Prop :=
  ∀ k p : ℕ, (10^p * k + n) % n = 0

def goodNumbersLessThan1000 : List ℕ := [1, 2, 5, 10, 20, 25, 50, 100, 125, 200]

theorem all_good_numbers_less_than_1000 :
  ∀ n ∈ goodNumbersLessThan1000, isGood n ∧ n < 1000 := by
  sorry

#check all_good_numbers_less_than_1000

end NUMINAMATH_CALUDE_all_good_numbers_less_than_1000_l2607_260755
