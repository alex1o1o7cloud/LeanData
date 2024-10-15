import Mathlib

namespace NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3477_347754

/-- A perfect square trinomial in x and y -/
def isPerfectSquareTrinomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, f x y = (x + k*y)^2 ∨ f x y = (x - k*y)^2

/-- The theorem stating that if x^2 + axy + y^2 is a perfect square trinomial, then a = 2 or a = -2 -/
theorem perfect_square_trinomial_condition (a : ℝ) :
  isPerfectSquareTrinomial (fun x y => x^2 + a*x*y + y^2) → a = 2 ∨ a = -2 := by
  sorry


end NUMINAMATH_CALUDE_perfect_square_trinomial_condition_l3477_347754


namespace NUMINAMATH_CALUDE_unique_a_value_l3477_347765

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x : ℝ | x^2 - 4*x + 3 ≥ 0}

-- Define the proposition p and q
def p (a : ℝ) : Prop := ∃ x, x ∈ A a
def q : Prop := ∃ x, x ∈ B

-- Define the negation of q
def not_q : Prop := ∃ x, x ∉ B

-- Theorem statement
theorem unique_a_value : 
  ∃! a : ℝ, (∀ x : ℝ, not_q → p a) ∧ a = 2 := by sorry

end NUMINAMATH_CALUDE_unique_a_value_l3477_347765


namespace NUMINAMATH_CALUDE_modular_equation_solution_l3477_347797

theorem modular_equation_solution : ∃ (n : ℤ), 0 ≤ n ∧ n < 144 ∧ (143 * n) % 144 = 105 % 144 ∧ n = 39 := by
  sorry

end NUMINAMATH_CALUDE_modular_equation_solution_l3477_347797


namespace NUMINAMATH_CALUDE_cone_rotation_ratio_l3477_347793

theorem cone_rotation_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  2 * π * Real.sqrt (r^2 + h^2) = 20 * π * r →
  h / r = Real.sqrt 399 := by
sorry

end NUMINAMATH_CALUDE_cone_rotation_ratio_l3477_347793


namespace NUMINAMATH_CALUDE_root_in_interval_l3477_347733

/-- The function f(x) = ln x + 3x - 7 has a root in the interval (2, 3) -/
theorem root_in_interval : ∃ x : ℝ, 2 < x ∧ x < 3 ∧ Real.log x + 3 * x - 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3477_347733


namespace NUMINAMATH_CALUDE_pi_estimation_l3477_347715

theorem pi_estimation (total_points : ℕ) (obtuse_points : ℕ) : 
  total_points = 120 → obtuse_points = 34 → 
  (obtuse_points : ℝ) / (total_points : ℝ) = π / 4 - 1 / 2 → 
  π = 47 / 15 := by
sorry

end NUMINAMATH_CALUDE_pi_estimation_l3477_347715


namespace NUMINAMATH_CALUDE_base_of_first_term_l3477_347799

theorem base_of_first_term (base x y : ℕ) : 
  base ^ x * 4 ^ y = 19683 → 
  x - y = 9 → 
  x = 9 → 
  base = 3 := by
sorry

end NUMINAMATH_CALUDE_base_of_first_term_l3477_347799


namespace NUMINAMATH_CALUDE_numerator_proof_l3477_347787

theorem numerator_proof (x y : ℝ) (h1 : x / y = 7 / 3) :
  ∃ (k N : ℝ), x = 7 * k ∧ y = 3 * k ∧ N / (x - y) = 2.5 ∧ N = 10 * k := by
  sorry

end NUMINAMATH_CALUDE_numerator_proof_l3477_347787


namespace NUMINAMATH_CALUDE_first_concert_attendance_l3477_347784

theorem first_concert_attendance (second_concert : ℕ) (difference : ℕ) : 
  second_concert = 66018 → difference = 119 → second_concert - difference = 65899 := by
  sorry

end NUMINAMATH_CALUDE_first_concert_attendance_l3477_347784


namespace NUMINAMATH_CALUDE_function_coefficient_sum_l3477_347755

/-- Given a function f : ℝ → ℝ satisfying certain conditions, prove that a + b + c = 3 -/
theorem function_coefficient_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 3) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 3 := by
sorry

end NUMINAMATH_CALUDE_function_coefficient_sum_l3477_347755


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l3477_347717

theorem quadratic_equation_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - 19 * x + k = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 - 19 * y + k = 0 ∧ y = 16/3) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l3477_347717


namespace NUMINAMATH_CALUDE_pyramid_inequality_l3477_347720

/-- A triangular pyramid with vertex O and base ABC -/
structure TriangularPyramid (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] where
  O : V
  A : V
  B : V
  C : V

/-- The area of a triangle -/
def triangleArea (A B C : V) [NormedAddCommGroup V] [InnerProductSpace ℝ V] : ℝ :=
  sorry

/-- Statement of the theorem -/
theorem pyramid_inequality (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (pyramid : TriangularPyramid V) (M : V) :
  let S_ABC := triangleArea pyramid.A pyramid.B pyramid.C
  let S_MBC := triangleArea M pyramid.B pyramid.C
  let S_MAC := triangleArea M pyramid.A pyramid.C
  let S_MAB := triangleArea M pyramid.A pyramid.B
  ‖pyramid.O - M‖ * S_ABC ≤ 
    ‖pyramid.O - pyramid.A‖ * S_MBC + 
    ‖pyramid.O - pyramid.B‖ * S_MAC + 
    ‖pyramid.O - pyramid.C‖ * S_MAB :=
by
  sorry

end NUMINAMATH_CALUDE_pyramid_inequality_l3477_347720


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3477_347776

theorem arithmetic_sequence_ratio (a b d₁ d₂ : ℝ) : 
  (a + 4 * d₁ = b) → (a + 5 * d₂ = b) → d₁ / d₂ = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3477_347776


namespace NUMINAMATH_CALUDE_intersection_of_sets_l3477_347736

theorem intersection_of_sets : 
  let M : Set ℤ := {0, 1, 2, 3}
  let P : Set ℤ := {-1, 1, -2, 2}
  M ∩ P = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l3477_347736


namespace NUMINAMATH_CALUDE_haley_zoo_pictures_l3477_347725

/-- The number of pictures Haley took at the zoo -/
def zoo_pictures : ℕ := sorry

/-- The total number of pictures Haley took before deleting any -/
def total_pictures : ℕ := zoo_pictures + 8

/-- The number of pictures Haley had after deleting some -/
def remaining_pictures : ℕ := total_pictures - 38

theorem haley_zoo_pictures :
  zoo_pictures = 50 ∧ remaining_pictures = 20 :=
sorry

end NUMINAMATH_CALUDE_haley_zoo_pictures_l3477_347725


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3477_347721

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h1 : a 1 + a 5 = 10)
  (h2 : a 4 = 7)
  (h_arith : arithmetic_sequence a) :
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3477_347721


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3477_347716

/-- Represents a right triangle with angles 30°, 60°, and 90° -/
structure Triangle30_60_90 where
  shortSide : ℝ
  longSide : ℝ
  hypotenuse : ℝ
  angle30 : Real
  angle60 : Real
  angle90 : Real

/-- Represents a circle tangent to coordinate axes and triangle hypotenuse -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given a 30-60-90 triangle with shortest side 2, 
    the radius of a circle tangent to coordinate axes and hypotenuse is 1 + 2√3 -/
theorem tangent_circle_radius 
  (t : Triangle30_60_90) 
  (c : TangentCircle) 
  (h1 : t.shortSide = 2) 
  (h2 : c.center.1 > 0 ∧ c.center.2 > 0) 
  (h3 : c.radius = c.center.1 ∧ c.radius = c.center.2) 
  (h4 : ∃ (x y : ℝ), x^2 + y^2 = t.hypotenuse^2 ∧ 
                     (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2) :
  c.radius = 1 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3477_347716


namespace NUMINAMATH_CALUDE_jaymee_is_22_l3477_347789

/-- The age of Shara -/
def shara_age : ℕ := 10

/-- The age of Jaymee -/
def jaymee_age : ℕ := 2 * shara_age + 2

/-- Theorem stating Jaymee's age is 22 -/
theorem jaymee_is_22 : jaymee_age = 22 := by
  sorry

end NUMINAMATH_CALUDE_jaymee_is_22_l3477_347789


namespace NUMINAMATH_CALUDE_system_solution_l3477_347713

theorem system_solution : 
  ∀ (x y z t : ℕ), 
    x + y = z * t ∧ z + t = x * y → 
      ((x = 1 ∧ y = 5 ∧ z = 2 ∧ t = 3) ∨ 
       (x = 5 ∧ y = 1 ∧ z = 3 ∧ t = 2) ∨ 
       (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2)) := by
  sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l3477_347713


namespace NUMINAMATH_CALUDE_john_payment_is_8000_l3477_347718

/-- Calculates John's payment for lawyer fees --/
def johnPayment (upfrontPayment : ℕ) (hourlyRate : ℕ) (courtTime : ℕ) : ℕ :=
  let totalTime := courtTime + 2 * courtTime
  let totalFee := upfrontPayment + hourlyRate * totalTime
  totalFee / 2

/-- Theorem: John's payment for lawyer fees is $8,000 --/
theorem john_payment_is_8000 :
  johnPayment 1000 100 50 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_john_payment_is_8000_l3477_347718


namespace NUMINAMATH_CALUDE_emily_small_gardens_l3477_347748

/-- Calculates the number of small gardens Emily has based on her seed distribution --/
def number_of_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_type : ℕ) (vegetable_types : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / (seeds_per_type * vegetable_types)

/-- Theorem stating that Emily has 4 small gardens --/
theorem emily_small_gardens :
  number_of_small_gardens 125 45 4 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_emily_small_gardens_l3477_347748


namespace NUMINAMATH_CALUDE_chanhee_walking_distance_l3477_347742

/-- Calculates the total distance walked given step length, duration, and pace. -/
def total_distance (step_length : Real) (duration : Real) (pace : Real) : Real :=
  step_length * duration * pace

/-- Proves that Chanhee walked 526.5 meters given the specified conditions. -/
theorem chanhee_walking_distance :
  let step_length : Real := 0.45
  let duration : Real := 13
  let pace : Real := 90
  total_distance step_length duration pace = 526.5 := by
  sorry

end NUMINAMATH_CALUDE_chanhee_walking_distance_l3477_347742


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3477_347792

theorem linear_equation_solution (a : ℝ) : 
  (a * 1 + (-2) = 3) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3477_347792


namespace NUMINAMATH_CALUDE_cos_m_eq_sin_318_l3477_347757

theorem cos_m_eq_sin_318 (m : ℤ) (h1 : -180 ≤ m) (h2 : m ≤ 180) (h3 : Real.cos (m * π / 180) = Real.sin (318 * π / 180)) :
  m = 132 ∨ m = -132 := by
sorry

end NUMINAMATH_CALUDE_cos_m_eq_sin_318_l3477_347757


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3477_347704

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmerSpeed : ℝ
  streamSpeed : ℝ

/-- Calculates the effective speed when swimming downstream. -/
def downstreamSpeed (s : SwimmerSpeed) : ℝ :=
  s.swimmerSpeed + s.streamSpeed

/-- Calculates the effective speed when swimming upstream. -/
def upstreamSpeed (s : SwimmerSpeed) : ℝ :=
  s.swimmerSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 7 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
    (downstreamSpeed s * 4 = 32) →
    (upstreamSpeed s * 4 = 24) →
    s.swimmerSpeed = 7 :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l3477_347704


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l3477_347779

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are opposite. -/
def symmetric_x_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = q.1 ∧ p.2 = -q.2

/-- Given points P(x,-3) and Q(4,y) that are symmetric with respect to the x-axis,
    prove that x + y = 7 -/
theorem symmetric_points_sum (x y : ℝ) :
  symmetric_x_axis (x, -3) (4, y) → x + y = 7 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l3477_347779


namespace NUMINAMATH_CALUDE_bus_speed_problem_l3477_347758

/-- Proves that given the conditions of the bus problem, the average speed for the 220 km distance is 40 kmph -/
theorem bus_speed_problem (total_distance : ℝ) (total_time : ℝ) (distance_at_x : ℝ) (speed_known : ℝ) :
  total_distance = 250 →
  total_time = 6 →
  distance_at_x = 220 →
  speed_known = 60 →
  ∃ x : ℝ,
    x > 0 ∧
    (distance_at_x / x) + ((total_distance - distance_at_x) / speed_known) = total_time ∧
    x = 40 :=
by sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l3477_347758


namespace NUMINAMATH_CALUDE_arrangement_count_is_36_l3477_347751

/-- The number of ways to arrange 5 students in a row with specific conditions -/
def arrangement_count : ℕ :=
  let n : ℕ := 5  -- Total number of students
  let special_pair : ℕ := 2  -- Number of students that must be adjacent (A and B)
  let non_end_student : ℕ := 1  -- Number of students that can't be at the ends (A)
  -- The actual count calculation would go here
  36

/-- Theorem stating that the number of arrangements under given conditions is 36 -/
theorem arrangement_count_is_36 : arrangement_count = 36 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_is_36_l3477_347751


namespace NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3477_347770

theorem greatest_x_quadratic_inequality :
  ∀ x : ℝ, -x^2 + 11*x - 28 ≥ 0 → x ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_quadratic_inequality_l3477_347770


namespace NUMINAMATH_CALUDE_correct_weight_calculation_l3477_347777

/-- Given a class of boys with incorrect and correct average weights, calculate the correct weight that was misread. -/
theorem correct_weight_calculation (n : ℕ) (incorrect_avg correct_avg misread_weight : ℚ) 
  (h1 : n = 20)
  (h2 : incorrect_avg = 584/10)
  (h3 : correct_avg = 59)
  (h4 : misread_weight = 56) :
  let incorrect_total := n * incorrect_avg
  let correct_total := n * correct_avg
  let weight_difference := correct_total - incorrect_total
  misread_weight + weight_difference = 68 := by sorry

end NUMINAMATH_CALUDE_correct_weight_calculation_l3477_347777


namespace NUMINAMATH_CALUDE_system_solution_l3477_347780

theorem system_solution (a b c d e : ℝ) : 
  (3 * a = (b + c + d)^3 ∧
   3 * b = (c + d + e)^3 ∧
   3 * c = (d + e + a)^3 ∧
   3 * d = (e + a + b)^3 ∧
   3 * e = (a + b + c)^3) →
  ((a = 1/3 ∧ b = 1/3 ∧ c = 1/3 ∧ d = 1/3 ∧ e = 1/3) ∨
   (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 ∧ e = 0) ∨
   (a = -1/3 ∧ b = -1/3 ∧ c = -1/3 ∧ d = -1/3 ∧ e = -1/3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3477_347780


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_1_2_l3477_347763

/-- Equation of motion for an object -/
def s (t : ℝ) : ℝ := 2 * (1 - t^2)

/-- Instantaneous velocity at time t -/
def v (t : ℝ) : ℝ := -4 * t

theorem instantaneous_velocity_at_1_2 : v 1.2 = -4.8 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_1_2_l3477_347763


namespace NUMINAMATH_CALUDE_trapezoid_point_distance_l3477_347738

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a line in 2D space -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Represents a trapezoid ABCD -/
structure Trapezoid :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Returns the intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point :=
  sorry

/-- Returns the line passing through two points -/
def lineThroughPoints (p1 p2 : Point) : Line :=
  sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

/-- Recursively defines points A_n and B_n -/
def definePoints (trap : Trapezoid) (E : Point) (n : ℕ) : Point × Point :=
  match n with
  | 0 => (trap.A, trap.B)
  | n+1 =>
    let (A_n, _) := definePoints trap E n
    let B_next := intersectionPoint (lineThroughPoints A_n trap.C) (lineThroughPoints trap.B trap.D)
    let A_next := intersectionPoint (lineThroughPoints E B_next) (lineThroughPoints trap.A trap.B)
    (A_next, B_next)

/-- The main theorem to be proved -/
theorem trapezoid_point_distance (trap : Trapezoid) (E : Point) (n : ℕ) :
  let (A_n, _) := definePoints trap E n
  distance A_n trap.B = distance trap.A trap.B / (n + 1) :=
sorry

end NUMINAMATH_CALUDE_trapezoid_point_distance_l3477_347738


namespace NUMINAMATH_CALUDE_inequality_implies_theta_range_l3477_347786

open Real

theorem inequality_implies_theta_range (θ : ℝ) :
  θ ∈ Set.Icc 0 (2 * π) →
  3 * (sin θ ^ 5 + cos (2 * θ) ^ 5) > 5 * (sin θ ^ 3 + cos (2 * θ) ^ 3) →
  θ ∈ Set.Ioo (7 * π / 6) (11 * π / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_theta_range_l3477_347786


namespace NUMINAMATH_CALUDE_expression_equality_l3477_347723

theorem expression_equality (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z - z/x ≠ 0) :
  (x^2 - 1/y^2) / (z - z/x) = x/z :=
by sorry

end NUMINAMATH_CALUDE_expression_equality_l3477_347723


namespace NUMINAMATH_CALUDE_number_of_children_l3477_347729

def total_cupcakes : ℕ := 96
def cupcakes_per_child : ℕ := 12

theorem number_of_children : 
  total_cupcakes / cupcakes_per_child = 8 := by sorry

end NUMINAMATH_CALUDE_number_of_children_l3477_347729


namespace NUMINAMATH_CALUDE_unique_students_in_musical_groups_l3477_347772

/-- The number of unique students in four musical groups -/
theorem unique_students_in_musical_groups 
  (orchestra : Nat) (band : Nat) (choir : Nat) (jazz : Nat)
  (orchestra_band : Nat) (orchestra_choir : Nat) (band_choir : Nat)
  (band_jazz : Nat) (orchestra_jazz : Nat) (choir_jazz : Nat)
  (orchestra_band_choir : Nat) (all_four : Nat)
  (h1 : orchestra = 25)
  (h2 : band = 40)
  (h3 : choir = 30)
  (h4 : jazz = 15)
  (h5 : orchestra_band = 5)
  (h6 : orchestra_choir = 6)
  (h7 : band_choir = 4)
  (h8 : band_jazz = 3)
  (h9 : orchestra_jazz = 2)
  (h10 : choir_jazz = 4)
  (h11 : orchestra_band_choir = 3)
  (h12 : all_four = 1) :
  orchestra + band + choir + jazz
  - orchestra_band - orchestra_choir - band_choir
  - band_jazz - orchestra_jazz - choir_jazz
  + orchestra_band_choir + all_four = 90 :=
by sorry

end NUMINAMATH_CALUDE_unique_students_in_musical_groups_l3477_347772


namespace NUMINAMATH_CALUDE_money_distribution_problem_l3477_347778

/-- Represents the shares of P, Q, and R in the money distribution problem. -/
structure Shares where
  p : ℕ
  q : ℕ
  r : ℕ

/-- Represents the problem constraints and solution. -/
theorem money_distribution_problem (s : Shares) : 
  -- The ratio condition
  s.p + s.q + s.r > 0 ∧ 
  3 * s.q = 7 * s.p ∧ 
  3 * s.r = 4 * s.q ∧ 
  -- The difference between P and Q's shares
  s.q - s.p = 2800 ∧ 
  -- Total amount condition
  50000 ≤ s.p + s.q + s.r ∧ 
  s.p + s.q + s.r ≤ 75000 ∧ 
  -- Minimum and maximum share conditions
  s.p ≥ 5000 ∧ 
  s.r ≤ 45000 
  -- The difference between Q and R's shares
  → s.r - s.q = 14000 := by sorry

end NUMINAMATH_CALUDE_money_distribution_problem_l3477_347778


namespace NUMINAMATH_CALUDE_no_valid_permutation_1986_l3477_347730

/-- Represents a permutation of the sequence 1,1,2,2,...,n,n -/
def Permutation (n : ℕ) := Fin (2*n) → Fin n

/-- The separation between pairs in a permutation -/
def separation (n : ℕ) (p : Permutation n) (i : Fin n) : ℕ := sorry

/-- A permutation satisfies the separation condition if for each i,
    there are exactly i numbers between the two occurrences of i -/
def satisfies_separation (n : ℕ) (p : Permutation n) : Prop :=
  ∀ i : Fin n, separation n p i = i.val

/-- The main theorem: there is no permutation of 1,1,2,2,...,1986,1986
    that satisfies the separation condition -/
theorem no_valid_permutation_1986 :
  ¬ ∃ (p : Permutation 1986), satisfies_separation 1986 p :=
sorry

end NUMINAMATH_CALUDE_no_valid_permutation_1986_l3477_347730


namespace NUMINAMATH_CALUDE_family_spent_36_dollars_l3477_347701

/-- The cost of a movie ticket in dollars -/
def ticket_cost : ℚ := 5

/-- The cost of popcorn as a fraction of the ticket cost -/
def popcorn_ratio : ℚ := 4/5

/-- The cost of soda as a fraction of the popcorn cost -/
def soda_ratio : ℚ := 1/2

/-- The number of tickets bought -/
def num_tickets : ℕ := 4

/-- The number of popcorn sets bought -/
def num_popcorn : ℕ := 2

/-- The number of soda cans bought -/
def num_soda : ℕ := 4

/-- Theorem: The total amount spent by the family is $36 -/
theorem family_spent_36_dollars :
  let popcorn_cost := ticket_cost * popcorn_ratio
  let soda_cost := popcorn_cost * soda_ratio
  let total_cost := (num_tickets : ℚ) * ticket_cost +
                    (num_popcorn : ℚ) * popcorn_cost +
                    (num_soda : ℚ) * soda_cost
  total_cost = 36 := by sorry

end NUMINAMATH_CALUDE_family_spent_36_dollars_l3477_347701


namespace NUMINAMATH_CALUDE_abs_neg_two_thirds_l3477_347796

theorem abs_neg_two_thirds : |(-2 : ℚ) / 3| = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_thirds_l3477_347796


namespace NUMINAMATH_CALUDE_marble_distribution_l3477_347700

theorem marble_distribution (total_marbles : ℕ) (group_size : ℕ) : 
  total_marbles = 364 →
  (total_marbles / group_size : ℚ) - (total_marbles / (group_size + 2) : ℚ) = 1 →
  group_size = 26 := by
  sorry

end NUMINAMATH_CALUDE_marble_distribution_l3477_347700


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l3477_347764

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 220 (-5) n = 35 ∧ n = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l3477_347764


namespace NUMINAMATH_CALUDE_range_of_a_l3477_347760

-- Define the propositions p and q
def p (x : ℝ) : Prop := 1/2 ≤ x ∧ x ≤ 1
def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) > 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (a : ℝ) : Prop :=
  (∀ x, p x → q x a) ∧ ∃ x, ¬p x ∧ q x a

-- State the theorem
theorem range_of_a : 
  ∀ a : ℝ, sufficient_not_necessary a ↔ 0 ≤ a ∧ a ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3477_347760


namespace NUMINAMATH_CALUDE_parabola_constant_term_l3477_347768

theorem parabola_constant_term (b c : ℝ) : 
  (2 = 2*(1^2) + b*1 + c) ∧ (2 = 2*(3^2) + b*3 + c) → c = 8 := by
sorry

end NUMINAMATH_CALUDE_parabola_constant_term_l3477_347768


namespace NUMINAMATH_CALUDE_overestimation_proof_l3477_347775

theorem overestimation_proof (p q k d : ℤ) 
  (p_round q_round k_round d_round : ℚ)
  (hp : p = 150) (hq : q = 50) (hk : k = 2) (hd : d = 3)
  (hp_round : p_round = 160) (hq_round : q_round = 45) 
  (hk_round : k_round = 1) (hd_round : d_round = 4) :
  (p_round / q_round - k_round + d_round) > (p / q - k + d) := by
  sorry

#check overestimation_proof

end NUMINAMATH_CALUDE_overestimation_proof_l3477_347775


namespace NUMINAMATH_CALUDE_condo_penthouse_floors_l3477_347703

/-- Represents a condo building with regular and penthouse floors -/
structure Condo where
  total_floors : ℕ
  regular_units_per_floor : ℕ
  penthouse_units_per_floor : ℕ
  total_units : ℕ

/-- Calculates the number of penthouse floors in a condo -/
def penthouse_floors (c : Condo) : ℕ :=
  c.total_floors - (c.total_units - 2 * c.total_floors) / (c.regular_units_per_floor - c.penthouse_units_per_floor)

/-- Theorem stating that the condo with given specifications has 2 penthouse floors -/
theorem condo_penthouse_floors :
  let c : Condo := {
    total_floors := 23,
    regular_units_per_floor := 12,
    penthouse_units_per_floor := 2,
    total_units := 256
  }
  penthouse_floors c = 2 := by
  sorry

end NUMINAMATH_CALUDE_condo_penthouse_floors_l3477_347703


namespace NUMINAMATH_CALUDE_fraction_comparison_l3477_347781

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l3477_347781


namespace NUMINAMATH_CALUDE_max_plus_shapes_in_square_l3477_347732

theorem max_plus_shapes_in_square (side_length : ℕ) (l_shape_area : ℕ) (plus_shape_area : ℕ) 
  (h_side : side_length = 7)
  (h_l : l_shape_area = 3)
  (h_plus : plus_shape_area = 5) :
  ∃ (num_l num_plus : ℕ),
    num_l * l_shape_area + num_plus * plus_shape_area = side_length ^ 2 ∧
    num_l ≥ 4 ∧
    ∀ (other_num_l other_num_plus : ℕ),
      other_num_l * l_shape_area + other_num_plus * plus_shape_area = side_length ^ 2 →
      other_num_l ≥ 4 →
      other_num_plus ≤ num_plus :=
by sorry

end NUMINAMATH_CALUDE_max_plus_shapes_in_square_l3477_347732


namespace NUMINAMATH_CALUDE_circle_sum_zero_l3477_347766

theorem circle_sum_zero (a : Fin 55 → ℤ) 
  (h : ∀ i : Fin 55, a i = a (i - 1) + a (i + 1)) : 
  ∀ i : Fin 55, a i = 0 := by
  sorry

end NUMINAMATH_CALUDE_circle_sum_zero_l3477_347766


namespace NUMINAMATH_CALUDE_solution_not_zero_l3477_347788

theorem solution_not_zero (a : ℝ) : ∀ x : ℝ, x = a * x + 1 → x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_not_zero_l3477_347788


namespace NUMINAMATH_CALUDE_equation_equivalence_l3477_347708

theorem equation_equivalence (x : ℝ) : x^2 - 4*x - 4 = 0 ↔ (x - 2)^2 = 8 := by sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3477_347708


namespace NUMINAMATH_CALUDE_common_chord_length_l3477_347740

theorem common_chord_length (a : ℝ) (h : a > 0) :
  (∃ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0) ∧
  (∀ x y : ℝ, x^2 + y^2 = 4 ∧ x^2 + y^2 + 2*a*y - 6 = 0 → y = 1/a) →
  a = 1 :=
sorry


end NUMINAMATH_CALUDE_common_chord_length_l3477_347740


namespace NUMINAMATH_CALUDE_jeff_cabinets_l3477_347743

/-- Calculates the total number of cabinets Jeff has after installations -/
def total_cabinets (initial : ℕ) (counters : ℕ) (additional : ℕ) : ℕ :=
  initial + counters * (2 * initial) + additional

/-- Proves that Jeff has 26 cabinets in total -/
theorem jeff_cabinets : total_cabinets 3 3 5 = 26 := by
  sorry

end NUMINAMATH_CALUDE_jeff_cabinets_l3477_347743


namespace NUMINAMATH_CALUDE_rectangle_area_l3477_347759

/-- Given a square, a circle, and two rectangles in a plane with the following properties:
    - The length of rectangle1 is two-fifths of the circle's radius
    - The circle's radius equals the square's side
    - The square's area is 900 sq. units
    - The width of rectangle1 is 10 units
    - The width of the square is thrice the width of rectangle2
    - The length of rectangle2 when tripled and added to the length of rectangle1 equals the length of rectangle1
    - The area of rectangle2 is half of the square's area
    Prove that the area of rectangle1 is 120 sq. units -/
theorem rectangle_area (square : Real) (circle : Real) (rectangle1 : Real × Real) (rectangle2 : Real × Real)
  (h1 : rectangle1.1 = (2/5) * circle)
  (h2 : circle = square)
  (h3 : square ^ 2 = 900)
  (h4 : rectangle1.2 = 10)
  (h5 : square = 3 * rectangle2.2)
  (h6 : 3 * rectangle2.1 + rectangle1.1 = rectangle1.1)
  (h7 : rectangle2.1 * rectangle2.2 = (1/2) * square ^ 2) :
  rectangle1.1 * rectangle1.2 = 120 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3477_347759


namespace NUMINAMATH_CALUDE_perp_condition_relationship_l3477_347737

/-- A structure representing a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- A structure representing a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Predicate indicating if a line is perpendicular to a plane -/
def perp_to_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Predicate indicating if a line is perpendicular to countless lines in a plane -/
def perp_to_countless_lines (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Theorem stating the relationship between the two conditions -/
theorem perp_condition_relationship :
  (∀ (l : Line3D) (α : Plane3D), perp_to_plane l α → perp_to_countless_lines l α) ∧
  (∃ (l : Line3D) (α : Plane3D), perp_to_countless_lines l α ∧ ¬perp_to_plane l α) :=
sorry

end NUMINAMATH_CALUDE_perp_condition_relationship_l3477_347737


namespace NUMINAMATH_CALUDE_sum_through_base3_l3477_347728

/-- Converts a natural number from base 10 to base 3 --/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a number from base 3 (represented as a list of digits) to base 10 --/
def fromBase3 (l : List ℕ) : ℕ :=
  sorry

/-- Adds two numbers in base 3 (represented as lists of digits) --/
def addBase3 (a b : List ℕ) : List ℕ :=
  sorry

/-- Theorem stating that the sum of 10 and 23 in base 10 is equal to 33
    when performed through base 3 conversion and addition --/
theorem sum_through_base3 :
  fromBase3 (addBase3 (toBase3 10) (toBase3 23)) = 33 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_through_base3_l3477_347728


namespace NUMINAMATH_CALUDE_square_difference_minus_sum_squares_product_l3477_347785

theorem square_difference_minus_sum_squares_product (a b : ℝ) :
  (a - b)^2 - (b^2 + a^2 - 2*a*b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_minus_sum_squares_product_l3477_347785


namespace NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l3477_347706

theorem min_value_of_expression (x : ℝ) : (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

theorem lower_bound_achievable : ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_lower_bound_achievable_l3477_347706


namespace NUMINAMATH_CALUDE_train_crossing_time_l3477_347714

/-- The time taken for a train to cross a stationary point -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) : 
  train_length = 240 → 
  train_speed_kmh = 144 → 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3477_347714


namespace NUMINAMATH_CALUDE_opposite_number_l3477_347773

theorem opposite_number (a : ℝ) : 
  -(3 * a - 2) = -3 * a + 2 := by sorry

end NUMINAMATH_CALUDE_opposite_number_l3477_347773


namespace NUMINAMATH_CALUDE_area_enclosed_by_function_and_line_l3477_347790

theorem area_enclosed_by_function_and_line (c : ℝ) : 
  30 = (1/2) * (c + 2) * (c - 2) → c = 8 := by
sorry

end NUMINAMATH_CALUDE_area_enclosed_by_function_and_line_l3477_347790


namespace NUMINAMATH_CALUDE_village_population_l3477_347791

theorem village_population (initial_population : ℕ) : 
  (initial_population : ℝ) * (1 - 0.08) * (1 - 0.15) = 3553 → 
  initial_population = 4547 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l3477_347791


namespace NUMINAMATH_CALUDE_average_height_problem_l3477_347752

theorem average_height_problem (parker daisy reese : ℕ) : 
  parker + 4 = daisy →
  daisy = reese + 8 →
  reese = 60 →
  (parker + daisy + reese) / 3 = 64 := by
sorry

end NUMINAMATH_CALUDE_average_height_problem_l3477_347752


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l3477_347719

theorem solution_set_of_inequality (x : ℝ) :
  (2 * x + 4 > 0) ↔ (x > -2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l3477_347719


namespace NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l3477_347731

theorem simplify_fraction_with_sqrt_three : 
  (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1/2 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_with_sqrt_three_l3477_347731


namespace NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l3477_347709

theorem sufficient_condition_for_f_less_than_one
  (a : ℝ) (h_a : a > 1) :
  ∃ (x : ℝ), -1 < x ∧ x < 0 ∧ a * x + 2 * x < 1 :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_f_less_than_one_l3477_347709


namespace NUMINAMATH_CALUDE_loop_structure_requirement_l3477_347710

/-- Represents a computational task that may or may not require a loop structure. -/
inductive ComputationalTask
  | SolveLinearSystem
  | CalculatePiecewiseFunction
  | CalculateFixedSum
  | FindSmallestNaturalNumber

/-- Determines if a given computational task requires a loop structure. -/
def requiresLoopStructure (task : ComputationalTask) : Prop :=
  match task with
  | ComputationalTask.FindSmallestNaturalNumber => true
  | _ => false

/-- The sum of natural numbers from 1 to n. -/
def sumUpTo (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

/-- Theorem stating that finding the smallest natural number n such that 1+2+3+...+n > 100
    requires a loop structure, while other given tasks do not. -/
theorem loop_structure_requirement :
  (∀ n : ℕ, sumUpTo n ≤ 100 → sumUpTo (n + 1) > 100) →
  (requiresLoopStructure ComputationalTask.FindSmallestNaturalNumber ∧
   ¬requiresLoopStructure ComputationalTask.SolveLinearSystem ∧
   ¬requiresLoopStructure ComputationalTask.CalculatePiecewiseFunction ∧
   ¬requiresLoopStructure ComputationalTask.CalculateFixedSum) :=
by sorry


end NUMINAMATH_CALUDE_loop_structure_requirement_l3477_347710


namespace NUMINAMATH_CALUDE_rational_function_value_l3477_347769

-- Define the property for the rational function f
def satisfies_equation (f : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 3 * f (1 / x) + 2 * f x / x = x^2

-- State the theorem
theorem rational_function_value :
  ∀ f : ℚ → ℚ, satisfies_equation f → f (-2) = 67 / 20 :=
by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3477_347769


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3477_347798

open Real

theorem quadratic_equation_solution (A : ℝ) (h1 : 0 < A) (h2 : A < π) :
  (∃ x y : ℝ, x^2 * cos A - 2*x + cos A = 0 ∧
              y^2 * cos A - 2*y + cos A = 0 ∧
              x^2 - y^2 = 3/8) →
  sin A = (sqrt 265 - 16) / 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3477_347798


namespace NUMINAMATH_CALUDE_converse_opposite_numbers_correct_l3477_347735

theorem converse_opposite_numbers_correct :
  (∀ x y : ℝ, x = -y → x + y = 0) := by sorry

end NUMINAMATH_CALUDE_converse_opposite_numbers_correct_l3477_347735


namespace NUMINAMATH_CALUDE_quadratic_curve_focal_distance_l3477_347747

theorem quadratic_curve_focal_distance (a : ℝ) (h1 : a ≠ 0) :
  (∃ (x y : ℝ), x^2 + a*y^2 + a^2 = 0) ∧
  (∃ (c : ℝ), c = 2 ∧ c^2 = a^2 + (-a)) →
  a = (1 - Real.sqrt 17) / 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_curve_focal_distance_l3477_347747


namespace NUMINAMATH_CALUDE_communication_arrangement_l3477_347782

def letter_arrangement (n : ℕ) (triple : ℕ) (double : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial 3 * (Nat.factorial 2)^double * Nat.factorial (n - triple - 2*double))

theorem communication_arrangement :
  letter_arrangement 14 1 2 = 908107825 := by
  sorry

end NUMINAMATH_CALUDE_communication_arrangement_l3477_347782


namespace NUMINAMATH_CALUDE_inequality_problem_l3477_347744

theorem inequality_problem (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / (a - 1) ≥ 1 / b) ∧
  (1 / b < 1 / a) ∧
  (|a| > -b) ∧
  (Real.sqrt (-a) > Real.sqrt (-b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l3477_347744


namespace NUMINAMATH_CALUDE_correct_observation_value_l3477_347722

theorem correct_observation_value 
  (n : ℕ) 
  (initial_mean : ℝ) 
  (wrong_value : ℝ) 
  (corrected_mean : ℝ) 
  (h1 : n = 40) 
  (h2 : initial_mean = 100) 
  (h3 : wrong_value = 75) 
  (h4 : corrected_mean = 99.075) : 
  (n : ℝ) * corrected_mean - ((n : ℝ) * initial_mean - wrong_value) = 38 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l3477_347722


namespace NUMINAMATH_CALUDE_investment_problem_l3477_347767

def first_investment_value (x : ℝ) : Prop :=
  let second_investment : ℝ := 1500
  let combined_return_rate : ℝ := 0.085
  let first_return_rate : ℝ := 0.07
  let second_return_rate : ℝ := 0.09
  (first_return_rate * x + second_return_rate * second_investment = 
   combined_return_rate * (x + second_investment)) ∧
  x = 500

theorem investment_problem : ∃ x : ℝ, first_investment_value x := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3477_347767


namespace NUMINAMATH_CALUDE_min_both_like_problem_l3477_347762

def min_both_like (total surveyed beethoven_fans chopin_fans both_and_vivaldi : ℕ) : ℕ :=
  max (beethoven_fans + chopin_fans - total) both_and_vivaldi

theorem min_both_like_problem :
  let total := 200
  let beethoven_fans := 150
  let chopin_fans := 120
  let both_and_vivaldi := 80
  min_both_like total beethoven_fans chopin_fans both_and_vivaldi = 80 := by
sorry

end NUMINAMATH_CALUDE_min_both_like_problem_l3477_347762


namespace NUMINAMATH_CALUDE_parabola_height_comparison_l3477_347707

theorem parabola_height_comparison (x₁ x₂ : ℝ) (h1 : -4 < x₁ ∧ x₁ < -2) (h2 : 0 < x₂ ∧ x₂ < 2) :
  (x₁ ^ 2 : ℝ) > x₂ ^ 2 := by sorry

end NUMINAMATH_CALUDE_parabola_height_comparison_l3477_347707


namespace NUMINAMATH_CALUDE_monomial_sum_implies_mn_twelve_l3477_347746

/-- If the sum of 2x³yⁿ and -½xᵐy⁴ is a monomial, then mn = 12 -/
theorem monomial_sum_implies_mn_twelve (x y : ℝ) (m n : ℕ) :
  (∃ (c : ℝ), ∀ x y, 2 * x^3 * y^n - 1/2 * x^m * y^4 = c * x^3 * y^4) →
  m * n = 12 := by
sorry

end NUMINAMATH_CALUDE_monomial_sum_implies_mn_twelve_l3477_347746


namespace NUMINAMATH_CALUDE_jack_minimum_cars_per_hour_l3477_347756

/-- The minimum number of cars Jack can change oil in per hour -/
def jack_cars_per_hour : ℝ := 3

/-- The number of hours worked per day -/
def hours_per_day : ℝ := 8

/-- The number of cars Paul can change oil in per hour -/
def paul_cars_per_hour : ℝ := 2

/-- The minimum number of cars both mechanics can finish per day -/
def min_cars_per_day : ℝ := 40

theorem jack_minimum_cars_per_hour :
  jack_cars_per_hour * hours_per_day + paul_cars_per_hour * hours_per_day ≥ min_cars_per_day ∧
  ∀ x : ℝ, x * hours_per_day + paul_cars_per_hour * hours_per_day ≥ min_cars_per_day → x ≥ jack_cars_per_hour :=
by sorry

end NUMINAMATH_CALUDE_jack_minimum_cars_per_hour_l3477_347756


namespace NUMINAMATH_CALUDE_problem_solution_l3477_347711

theorem problem_solution (t : ℚ) (x y : ℚ) 
  (h1 : x = 3 - 2 * t) 
  (h2 : y = 5 * t + 9) 
  (h3 : x = 0) : 
  y = 33 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3477_347711


namespace NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3477_347702

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem lines_perp_to_plane_are_parallel 
  (a b : Line) (α : Plane) 
  (h1 : perp a α) (h2 : perp b α) : 
  parallel a b :=
sorry

end NUMINAMATH_CALUDE_lines_perp_to_plane_are_parallel_l3477_347702


namespace NUMINAMATH_CALUDE_circle_radius_from_sum_of_circumference_and_area_l3477_347750

theorem circle_radius_from_sum_of_circumference_and_area :
  ∀ r : ℝ, r > 0 →
    2 * Real.pi * r + Real.pi * r^2 = 530.929158456675 →
    r = Real.sqrt 170 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_sum_of_circumference_and_area_l3477_347750


namespace NUMINAMATH_CALUDE_sunflower_count_l3477_347771

theorem sunflower_count (total_flowers : ℕ) (other_flowers : ℕ) 
  (h1 : total_flowers = 160) 
  (h2 : other_flowers = 40) : 
  total_flowers - other_flowers = 120 := by
  sorry

end NUMINAMATH_CALUDE_sunflower_count_l3477_347771


namespace NUMINAMATH_CALUDE_job_selection_probability_l3477_347761

theorem job_selection_probability 
  (carol_prob : ℚ) 
  (bernie_prob : ℚ) 
  (h1 : carol_prob = 4 / 5) 
  (h2 : bernie_prob = 3 / 5) : 
  carol_prob * bernie_prob = 12 / 25 := by
  sorry

end NUMINAMATH_CALUDE_job_selection_probability_l3477_347761


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3477_347795

theorem inequality_solution_set (x : ℝ) :
  (x^2 - 2*x - 3) * (x^2 + 1) < 0 ↔ -1 < x ∧ x < 3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3477_347795


namespace NUMINAMATH_CALUDE_water_temperature_difference_l3477_347734

theorem water_temperature_difference (n : ℕ) : 
  let T_h := (T_c : ℝ) + 64/3
  let T_n := T_h - (1/4)^n * (T_h - T_c)
  (T_h - T_n ≠ 1/2) ∧ (T_h - T_n ≠ 3) :=
by sorry

end NUMINAMATH_CALUDE_water_temperature_difference_l3477_347734


namespace NUMINAMATH_CALUDE_apples_eaten_by_keith_l3477_347745

theorem apples_eaten_by_keith (mike_apples nancy_apples apples_left : ℝ) 
  (h1 : mike_apples = 7.0)
  (h2 : nancy_apples = 3.0)
  (h3 : apples_left = 4.0) :
  mike_apples + nancy_apples - apples_left = 6.0 := by
  sorry

end NUMINAMATH_CALUDE_apples_eaten_by_keith_l3477_347745


namespace NUMINAMATH_CALUDE_badminton_medals_count_l3477_347712

/-- Proves that the number of badminton medals is 5 --/
theorem badminton_medals_count :
  ∀ (total_medals track_medals swimming_medals badminton_medals : ℕ),
  total_medals = 20 →
  track_medals = 5 →
  swimming_medals = 2 * track_medals →
  badminton_medals = total_medals - (track_medals + swimming_medals) →
  badminton_medals = 5 := by
  sorry

end NUMINAMATH_CALUDE_badminton_medals_count_l3477_347712


namespace NUMINAMATH_CALUDE_gear_rotation_problem_l3477_347727

/-- 
Given two gears p and q rotating at constant speeds:
- q makes 40 revolutions per minute
- After 4 seconds, q has made exactly 2 more revolutions than p
Prove that p makes 10 revolutions per minute
-/
theorem gear_rotation_problem (p q : ℝ) 
  (hq : q = 40) -- q makes 40 revolutions per minute
  (h_diff : q * 4 / 60 = p * 4 / 60 + 2) -- After 4 seconds, q has made 2 more revolutions than p
  : p = 10 := by sorry

end NUMINAMATH_CALUDE_gear_rotation_problem_l3477_347727


namespace NUMINAMATH_CALUDE_ashutosh_completion_time_l3477_347705

theorem ashutosh_completion_time 
  (suresh_completion_time : ℝ) 
  (suresh_work_time : ℝ) 
  (ashutosh_remaining_time : ℝ) 
  (h1 : suresh_completion_time = 15)
  (h2 : suresh_work_time = 9)
  (h3 : ashutosh_remaining_time = 8)
  : ∃ (ashutosh_alone_time : ℝ), ashutosh_alone_time = 20 := by
  sorry

end NUMINAMATH_CALUDE_ashutosh_completion_time_l3477_347705


namespace NUMINAMATH_CALUDE_probability_all_green_apples_l3477_347774

/-- The probability of selecting all green apples when choosing 3 out of 10 apples, 
    given that there are 4 green apples. -/
theorem probability_all_green_apples (total : Nat) (green : Nat) (choose : Nat) : 
  total = 10 → green = 4 → choose = 3 → 
  (Nat.choose green choose : Rat) / (Nat.choose total choose) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_green_apples_l3477_347774


namespace NUMINAMATH_CALUDE_existence_of_prime_1021_n_l3477_347753

theorem existence_of_prime_1021_n : ∃ n : ℕ, n ≥ 3 ∧ Nat.Prime (n^3 + 2*n + 1) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_1021_n_l3477_347753


namespace NUMINAMATH_CALUDE_max_factors_power_function_l3477_347739

/-- The number of positive factors of n -/
def num_factors (n : ℕ) : ℕ := sorry

/-- b^n where b and n are positive integers less than or equal to 10 -/
def power_function (b n : ℕ) : ℕ := 
  if b ≤ 10 ∧ n ≤ 10 ∧ b > 0 ∧ n > 0 then b^n else 0

theorem max_factors_power_function :
  ∃ b n : ℕ, b ≤ 10 ∧ n ≤ 10 ∧ b > 0 ∧ n > 0 ∧
    num_factors (power_function b n) = 31 ∧
    ∀ b' n' : ℕ, b' ≤ 10 → n' ≤ 10 → b' > 0 → n' > 0 →
      num_factors (power_function b' n') ≤ 31 :=
sorry

end NUMINAMATH_CALUDE_max_factors_power_function_l3477_347739


namespace NUMINAMATH_CALUDE_sum_difference_is_50_l3477_347794

def sam_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def round_to_nearest_20 (x : ℕ) : ℕ :=
  20 * ((x + 10) / 20)

def alex_sum (n : ℕ) : ℕ :=
  List.sum (List.map round_to_nearest_20 (List.range n))

theorem sum_difference_is_50 :
  sam_sum 100 - alex_sum 100 = 50 := by
  sorry

#eval sam_sum 100 - alex_sum 100

end NUMINAMATH_CALUDE_sum_difference_is_50_l3477_347794


namespace NUMINAMATH_CALUDE_total_cost_after_discounts_l3477_347749

-- Define the original costs and discount percentages
def laptop_original_cost : ℚ := 800
def accessories_original_cost : ℚ := 200
def laptop_discount_percent : ℚ := 15
def accessories_discount_percent : ℚ := 10

-- Define the function to calculate the discounted price
def discounted_price (original_cost : ℚ) (discount_percent : ℚ) : ℚ :=
  original_cost * (1 - discount_percent / 100)

-- Theorem statement
theorem total_cost_after_discounts :
  discounted_price laptop_original_cost laptop_discount_percent +
  discounted_price accessories_original_cost accessories_discount_percent = 860 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_after_discounts_l3477_347749


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3477_347741

theorem quadratic_equation_solution : 
  ∃ (a b : ℝ), 
    (a^2 - 6*a + 9 = 15) ∧ 
    (b^2 - 6*b + 9 = 15) ∧ 
    (a ≥ b) ∧ 
    (3*a - b = 6 + 4*Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3477_347741


namespace NUMINAMATH_CALUDE_b_nonempty_implies_a_geq_two_thirds_a_intersect_b_eq_b_implies_a_geq_two_l3477_347726

-- Define sets A and B
def A : Set ℝ := {x | x ≥ 1}
def B (a : ℝ) : Set ℝ := {x | (1/2) * a ≤ x ∧ x ≤ 2*a - 1}

-- Theorem 1: If B is non-empty, then a ≥ 2/3
theorem b_nonempty_implies_a_geq_two_thirds (a : ℝ) :
  (B a).Nonempty → a ≥ 2/3 := by sorry

-- Theorem 2: If A ∩ B = B, then a ≥ 2
theorem a_intersect_b_eq_b_implies_a_geq_two (a : ℝ) :
  A ∩ (B a) = B a → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_b_nonempty_implies_a_geq_two_thirds_a_intersect_b_eq_b_implies_a_geq_two_l3477_347726


namespace NUMINAMATH_CALUDE_triangle_similarity_l3477_347724

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Triangle : Type :=
  (A B C : Point)

-- Define the properties
def isAcute (t : Triangle) : Prop := sorry

def incircleTouchPoints (t : Triangle) (D E F : Point) : Prop := sorry

def isCircumcenter (P : Point) (t : Triangle) : Prop := sorry

-- Main theorem
theorem triangle_similarity (A B C D E F P Q R : Point) :
  let ABC := Triangle.mk A B C
  let AEF := Triangle.mk A E F
  let BDF := Triangle.mk B D F
  let CDE := Triangle.mk C D E
  let PQR := Triangle.mk P Q R
  isAcute ABC →
  incircleTouchPoints ABC D E F →
  isCircumcenter P AEF →
  isCircumcenter Q BDF →
  isCircumcenter R CDE →
  -- Conclusion: ABC and PQR are similar
  ∃ (k : ℝ), k > 0 ∧
    (P.x - Q.x)^2 + (P.y - Q.y)^2 = k * ((A.x - B.x)^2 + (A.y - B.y)^2) ∧
    (Q.x - R.x)^2 + (Q.y - R.y)^2 = k * ((B.x - C.x)^2 + (B.y - C.y)^2) ∧
    (R.x - P.x)^2 + (R.y - P.y)^2 = k * ((C.x - A.x)^2 + (C.y - A.y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_triangle_similarity_l3477_347724


namespace NUMINAMATH_CALUDE_cos_2x_eq_cos_2y_l3477_347783

theorem cos_2x_eq_cos_2y (x y : ℝ) 
  (h1 : Real.sin x + Real.cos y = 1) 
  (h2 : Real.cos x + Real.sin y = -1) : 
  Real.cos (2 * x) = Real.cos (2 * y) := by
  sorry

end NUMINAMATH_CALUDE_cos_2x_eq_cos_2y_l3477_347783
