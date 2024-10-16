import Mathlib

namespace NUMINAMATH_CALUDE_square_area_ratio_l2339_233981

theorem square_area_ratio (side_C side_D : ℝ) (h1 : side_C = 45) (h2 : side_D = 60) :
  (side_C^2) / (side_D^2) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l2339_233981


namespace NUMINAMATH_CALUDE_total_distance_covered_l2339_233995

/-- Calculates the total distance covered by a man rowing upstream and downstream -/
theorem total_distance_covered (upstream_speed : ℝ) (upstream_time : ℝ) 
  (downstream_speed : ℝ) (downstream_time : ℝ) : 
  upstream_speed * upstream_time + downstream_speed * downstream_time = 62 :=
by
  sorry

#check total_distance_covered 12 2 38 1

end NUMINAMATH_CALUDE_total_distance_covered_l2339_233995


namespace NUMINAMATH_CALUDE_perpendicular_slope_to_OA_l2339_233919

/-- Given point A(3, 5) and O as the origin, prove that the slope of the line perpendicular to OA is -3/5 -/
theorem perpendicular_slope_to_OA :
  let A : ℝ × ℝ := (3, 5)
  let O : ℝ × ℝ := (0, 0)
  let slope_OA : ℝ := (A.2 - O.2) / (A.1 - O.1)
  let slope_perpendicular : ℝ := -1 / slope_OA
  slope_perpendicular = -3/5 := by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_to_OA_l2339_233919


namespace NUMINAMATH_CALUDE_tangent_line_condition_range_of_a_l2339_233917

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a / x
def g (x : ℝ) : ℝ := 2 * x * Real.exp x - Real.log x - x - Real.log 2

-- Part 1: Tangent line condition
theorem tangent_line_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ ∧ (deriv (f a)) x₀ = 1) → a = Real.exp 1 / 2 :=
sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ > 0, f a x₁ ≥ g x₂) → a ≥ 1 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_condition_range_of_a_l2339_233917


namespace NUMINAMATH_CALUDE_special_rectangle_dimensions_l2339_233922

/-- A rectangle with specific properties -/
structure SpecialRectangle where
  width : ℝ
  length : ℝ
  width_pos : 0 < width
  length_pos : 0 < length
  length_twice_width : length = 2 * width
  perimeter_three_times_area : 2 * (length + width) = 3 * (length * width)

/-- The width and length of the special rectangle are 1 and 2, respectively -/
theorem special_rectangle_dimensions (r : SpecialRectangle) : r.width = 1 ∧ r.length = 2 := by
  sorry


end NUMINAMATH_CALUDE_special_rectangle_dimensions_l2339_233922


namespace NUMINAMATH_CALUDE_carl_driving_hours_l2339_233950

theorem carl_driving_hours :
  let daily_hours : ℕ := 2
  let additional_weekly_hours : ℕ := 6
  let days_in_two_weeks : ℕ := 14
  let weeks : ℕ := 2
  (daily_hours * days_in_two_weeks) + (additional_weekly_hours * weeks) = 40 :=
by sorry

end NUMINAMATH_CALUDE_carl_driving_hours_l2339_233950


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2339_233900

theorem trigonometric_identity (α : Real) : 
  Real.sin α ^ 2 + Real.cos (π / 6 - α) ^ 2 - Real.sin α * Real.cos (π / 6 - α) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2339_233900


namespace NUMINAMATH_CALUDE_all_twentynine_l2339_233994

/-- A function that represents a circular arrangement of 2017 integers. -/
def CircularArrangement := Fin 2017 → ℤ

/-- Predicate to check if five consecutive elements in the arrangement are "arrangeable". -/
def IsArrangeable (arr : CircularArrangement) : Prop :=
  ∀ i : Fin 2017, arr i - arr (i + 1) + arr (i + 2) - arr (i + 3) + arr (i + 4) = 29

/-- Theorem stating that if all consecutive five-tuples in a circular arrangement of 2017 integers
    are arrangeable, then all integers in the arrangement must be 29. -/
theorem all_twentynine (arr : CircularArrangement) (h : IsArrangeable arr) :
    ∀ i : Fin 2017, arr i = 29 := by
  sorry

end NUMINAMATH_CALUDE_all_twentynine_l2339_233994


namespace NUMINAMATH_CALUDE_farm_legs_count_l2339_233957

/-- Calculates the total number of legs in a farm with ducks and horses -/
def total_legs (total_animals : ℕ) (num_ducks : ℕ) : ℕ :=
  let num_horses := total_animals - num_ducks
  let duck_legs := 2 * num_ducks
  let horse_legs := 4 * num_horses
  duck_legs + horse_legs

/-- Proves that in a farm with 11 animals, including 7 ducks and the rest horses, 
    the total number of legs is 30 -/
theorem farm_legs_count : total_legs 11 7 = 30 := by
  sorry

end NUMINAMATH_CALUDE_farm_legs_count_l2339_233957


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l2339_233970

theorem smallest_number_with_remainders (n : ℕ) : 
  (n > 1) →
  (n % 3 = 2) →
  (n % 5 = 2) →
  (n % 7 = 2) →
  (∀ m : ℕ, m > 1 → m % 3 = 2 → m % 5 = 2 → m % 7 = 2 → n ≤ m) →
  n = 107 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l2339_233970


namespace NUMINAMATH_CALUDE_trapezoid_shorter_base_l2339_233925

/-- Represents a trapezoid -/
structure Trapezoid where
  longer_base : ℝ
  shorter_base : ℝ
  midpoint_segment : ℝ

/-- 
Theorem: In a trapezoid where the line segment joining the midpoints of the diagonals 
has length 4 and the longer base is 100, the length of the shorter base is 92.
-/
theorem trapezoid_shorter_base 
  (T : Trapezoid) 
  (h1 : T.longer_base = 100) 
  (h2 : T.midpoint_segment = 4) : 
  T.shorter_base = 92 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_shorter_base_l2339_233925


namespace NUMINAMATH_CALUDE_max_d_value_l2339_233979

theorem max_d_value : 
  let f : ℝ → ℝ := λ d => (5 + Real.sqrt 244) / 3 - d
  ∃ d : ℝ, (4 * Real.sqrt 3) ^ 2 + (d + 5) ^ 2 = (2 * d) ^ 2 ∧ 
    (∀ x : ℝ, (4 * Real.sqrt 3) ^ 2 + (x + 5) ^ 2 = (2 * x) ^ 2 → f x ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_max_d_value_l2339_233979


namespace NUMINAMATH_CALUDE_floss_per_student_l2339_233988

/-- Proves that each student needs 5 yards of floss given the problem conditions -/
theorem floss_per_student 
  (num_students : ℕ) 
  (floss_per_packet : ℕ) 
  (leftover_floss : ℕ) 
  (total_floss : ℕ) :
  num_students = 20 →
  floss_per_packet = 35 →
  leftover_floss = 5 →
  total_floss = num_students * (total_floss / num_students) →
  total_floss % floss_per_packet = 0 →
  total_floss / num_students = 5 := by
sorry

end NUMINAMATH_CALUDE_floss_per_student_l2339_233988


namespace NUMINAMATH_CALUDE_equation_solution_l2339_233946

theorem equation_solution : ∃! x : ℝ, 9 / (5 + x / 0.75) = 1 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2339_233946


namespace NUMINAMATH_CALUDE_fraction_equality_l2339_233954

theorem fraction_equality (a b : ℝ) (h : a / b = 4 / 3) :
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2339_233954


namespace NUMINAMATH_CALUDE_pipe_filling_time_l2339_233904

theorem pipe_filling_time (fill_time_A : ℝ) (fill_time_B : ℝ) (combined_time : ℝ) :
  (fill_time_B = fill_time_A / 6) →
  (combined_time = 3.75) →
  (1 / fill_time_A + 1 / fill_time_B = 1 / combined_time) →
  fill_time_A = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l2339_233904


namespace NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l2339_233968

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan 3) = 13 * Real.sqrt 10 / 50 := by sorry

end NUMINAMATH_CALUDE_sin_sum_arcsin_arctan_l2339_233968


namespace NUMINAMATH_CALUDE_hyperbola_construction_uniqueness_l2339_233963

/-- A tangent line to a hyperbola at its vertex -/
structure Tangent where
  line : Line

/-- An asymptote of a hyperbola -/
structure Asymptote where
  line : Line

/-- Linear eccentricity of a hyperbola -/
def LinearEccentricity : Type := ℝ

/-- A hyperbola -/
structure Hyperbola where
  -- Define necessary components of a hyperbola

/-- Two hyperbolas are congruent if they have the same shape and size -/
def congruent (h1 h2 : Hyperbola) : Prop := sorry

/-- Two hyperbolas are parallel translations if one can be obtained from the other by a translation -/
def parallel_translation (h1 h2 : Hyperbola) (dir : Vec) : Prop := sorry

/-- Main theorem: Given a tangent, an asymptote, and linear eccentricity, 
    there exist exactly two congruent hyperbolas satisfying these conditions -/
theorem hyperbola_construction_uniqueness 
  (t : Tangent) (a₁ : Asymptote) (c : LinearEccentricity) :
  ∃! (h1 h2 : Hyperbola), 
    (∃ (dir : Vec), parallel_translation h1 h2 dir) ∧ 
    congruent h1 h2 ∧
    -- Additional conditions to ensure h1 and h2 satisfy t, a₁, and c
    sorry := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_construction_uniqueness_l2339_233963


namespace NUMINAMATH_CALUDE_division_remainder_l2339_233966

theorem division_remainder (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 30 = 18 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l2339_233966


namespace NUMINAMATH_CALUDE_backpack_price_relationship_l2339_233938

/-- Represents the relationship between backpack purchases and prices -/
theorem backpack_price_relationship (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for division
  (h2 : 810 > 0) -- Total spent on type A is positive
  (h3 : 600 > 0) -- Total spent on type B is positive
  (h4 : x + 20 > 0) -- Ensure denominator is positive
  : 810 / (x + 20) = (600 / x) * (1 - 10 / 100) :=
by sorry

end NUMINAMATH_CALUDE_backpack_price_relationship_l2339_233938


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2339_233923

/-- The distance from a point to the y-axis is equal to the absolute value of its x-coordinate -/
theorem distance_to_y_axis (P : ℝ × ℝ) : 
  let (x, y) := P
  abs x = Real.sqrt ((x - 0)^2 + (y - y)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2339_233923


namespace NUMINAMATH_CALUDE_floor_properties_l2339_233975

theorem floor_properties (x : ℝ) : 
  (x - 1 < ⌊x⌋ ∧ ⌊x⌋ ≤ x) ∧ ⌊2*x⌋ - 2*⌊x⌋ ∈ ({0, 1} : Set ℤ) := by sorry

end NUMINAMATH_CALUDE_floor_properties_l2339_233975


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2339_233953

-- Define the solution set
def solution_set : Set ℝ := Set.Ioi 4 ∪ Set.Iic 1

-- Define the inequality
def inequality (a b x : ℝ) : Prop := (x - a) / (x - b) > 0

theorem sum_of_a_and_b (a b : ℝ) :
  (∀ x, x ∈ solution_set ↔ inequality a b x) →
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2339_233953


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2339_233921

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 8*x + 10 = 0 ↔ (x - 4)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l2339_233921


namespace NUMINAMATH_CALUDE_max_sum_under_constraints_l2339_233936

theorem max_sum_under_constraints :
  ∃ (M : ℝ), M = 32/17 ∧
  (∀ x y : ℝ, 5*x + 3*y ≤ 9 → 3*x + 5*y ≤ 11 → x + y ≤ M) ∧
  (∃ x y : ℝ, 5*x + 3*y ≤ 9 ∧ 3*x + 5*y ≤ 11 ∧ x + y = M) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_under_constraints_l2339_233936


namespace NUMINAMATH_CALUDE_square_sum_of_xy_l2339_233945

theorem square_sum_of_xy (x y : ℕ+) 
  (h1 : x * y + 2 * x + 2 * y = 152)
  (h2 : x ^ 2 * y + x * y ^ 2 = 1512) :
  x ^ 2 + y ^ 2 = 1136 ∨ x ^ 2 + y ^ 2 = 221 := by
sorry

end NUMINAMATH_CALUDE_square_sum_of_xy_l2339_233945


namespace NUMINAMATH_CALUDE_gcd_15225_20335_35475_l2339_233901

theorem gcd_15225_20335_35475 : Nat.gcd 15225 (Nat.gcd 20335 35475) = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15225_20335_35475_l2339_233901


namespace NUMINAMATH_CALUDE_perfect_squares_difference_four_digit_sqrt_difference_l2339_233912

theorem perfect_squares_difference (m n p : ℕ) 
  (h1 : m > n) 
  (h2 : Real.sqrt m - Real.sqrt n = p) : 
  ∃ (a b : ℕ), m = a^2 ∧ n = b^2 := by
  sorry

theorem four_digit_sqrt_difference : 
  ∃! (abcd : ℕ), 
    1000 ≤ abcd ∧ abcd < 10000 ∧
    ∃ (a b c d : ℕ),
      abcd = 1000 * a + 100 * b + 10 * c + d ∧
      100 * a + 10 * c + d < abcd ∧
      Real.sqrt (abcd) - Real.sqrt (100 * a + 10 * c + d) = 11 * b := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_four_digit_sqrt_difference_l2339_233912


namespace NUMINAMATH_CALUDE_adults_on_bicycles_l2339_233928

/-- The number of wheels on a bicycle -/
def bicycle_wheels : ℕ := 2

/-- The number of wheels on a tricycle -/
def tricycle_wheels : ℕ := 3

/-- The number of children riding tricycles -/
def children_on_tricycles : ℕ := 15

/-- The total number of wheels observed -/
def total_wheels : ℕ := 57

/-- Theorem: The number of adults riding bicycles is 6 -/
theorem adults_on_bicycles : 
  ∃ (a : ℕ), a * bicycle_wheels + children_on_tricycles * tricycle_wheels = total_wheels ∧ a = 6 :=
sorry

end NUMINAMATH_CALUDE_adults_on_bicycles_l2339_233928


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l2339_233905

theorem arithmetic_progression_sum (a d : ℝ) (n : ℕ) : 
  (a + 10 * d = 5.25) →
  (a + 6 * d = 3.25) →
  (n : ℝ) * (2 * a + (n - 1) * d) / 2 = 56.25 →
  n = 15 := by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l2339_233905


namespace NUMINAMATH_CALUDE_apple_distribution_exists_and_unique_l2339_233944

/-- Represents the last names of the children -/
inductive LastName
| Smith
| Brown
| Jones
| Robinson

/-- Represents a child with their name and number of apples -/
structure Child where
  firstName : String
  lastName : LastName
  apples : Nat

/-- The problem statement -/
theorem apple_distribution_exists_and_unique :
  ∃! (distribution : List Child),
    (distribution.length = 8) ∧
    (distribution.map (λ c => c.apples)).sum = 32 ∧
    (∃ ann ∈ distribution, ann.firstName = "Ann" ∧ ann.apples = 1) ∧
    (∃ mary ∈ distribution, mary.firstName = "Mary" ∧ mary.apples = 2) ∧
    (∃ jane ∈ distribution, jane.firstName = "Jane" ∧ jane.apples = 3) ∧
    (∃ kate ∈ distribution, kate.firstName = "Kate" ∧ kate.apples = 4) ∧
    (∃ ned ∈ distribution, ned.firstName = "Ned" ∧ ned.lastName = LastName.Smith ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Smith ∧ sister.apples = ned.apples) ∧
    (∃ tom ∈ distribution, tom.firstName = "Tom" ∧ tom.lastName = LastName.Brown ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Brown ∧ tom.apples = 2 * sister.apples) ∧
    (∃ bill ∈ distribution, bill.firstName = "Bill" ∧ bill.lastName = LastName.Jones ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Jones ∧ bill.apples = 3 * sister.apples) ∧
    (∃ jack ∈ distribution, jack.firstName = "Jack" ∧ jack.lastName = LastName.Robinson ∧
      ∃ sister ∈ distribution, sister.lastName = LastName.Robinson ∧ jack.apples = 4 * sister.apples) :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_exists_and_unique_l2339_233944


namespace NUMINAMATH_CALUDE_simplified_expression_doubled_l2339_233999

theorem simplified_expression_doubled (b : ℝ) : b = 5 → 2 * (15 * b^4 / (75 * b^3)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplified_expression_doubled_l2339_233999


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_power_of_two_l2339_233933

theorem negation_of_forall_positive_power_of_two (P : ℝ → Prop) :
  (¬ ∀ x > 0, 2^x > 0) ↔ (∃ x > 0, 2^x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_power_of_two_l2339_233933


namespace NUMINAMATH_CALUDE_revenue_change_l2339_233934

theorem revenue_change 
  (T : ℝ) -- Original tax
  (C : ℝ) -- Original consumption
  (tax_reduction : ℝ) -- Tax reduction percentage
  (consumption_increase : ℝ) -- Consumption increase percentage
  (h1 : tax_reduction = 0.20) -- 20% tax reduction
  (h2 : consumption_increase = 0.15) -- 15% consumption increase
  : 
  (1 - tax_reduction) * (1 + consumption_increase) * T * C - T * C = -0.08 * T * C :=
by sorry

end NUMINAMATH_CALUDE_revenue_change_l2339_233934


namespace NUMINAMATH_CALUDE_incenter_vector_sum_implies_right_angle_l2339_233965

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Define a vector from a point to another point
def vector (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

-- Define vector addition
def add_vectors (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define scalar multiplication of a vector
def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define the angle at a vertex of a triangle
def angle_at_vertex (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem incenter_vector_sum_implies_right_angle (t : Triangle) :
  let I := incenter t
  let IA := vector I t.A
  let IB := vector I t.B
  let IC := vector I t.C
  add_vectors (scalar_mult 3 IA) (add_vectors (scalar_mult 4 IB) (scalar_mult 5 IC)) = (0, 0) →
  angle_at_vertex t t.C = 90 :=
sorry

end NUMINAMATH_CALUDE_incenter_vector_sum_implies_right_angle_l2339_233965


namespace NUMINAMATH_CALUDE_amy_current_age_l2339_233939

/-- Given that Mark is 7 years older than Amy and Mark will be 27 years old in 5 years,
    prove that Amy's current age is 15 years old. -/
theorem amy_current_age :
  ∀ (mark_age amy_age : ℕ),
  mark_age = amy_age + 7 →
  mark_age + 5 = 27 →
  amy_age = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_amy_current_age_l2339_233939


namespace NUMINAMATH_CALUDE_function_composition_identity_l2339_233927

/-- Piecewise function f(x) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then a * x + b else 10 - 4 * x

/-- Theorem stating that if f(f(x)) = x for all x, then a + b = 21/4 -/
theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 21/4 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_identity_l2339_233927


namespace NUMINAMATH_CALUDE_bens_initial_marbles_l2339_233974

theorem bens_initial_marbles (B : ℕ) : 
  (17 + B / 2 = B / 2 + 17) → B = 34 := by sorry

end NUMINAMATH_CALUDE_bens_initial_marbles_l2339_233974


namespace NUMINAMATH_CALUDE_workshop_workers_l2339_233914

/-- The total number of workers in a workshop with given salary conditions -/
theorem workshop_workers (average_salary : ℕ) (technician_count : ℕ) (technician_salary : ℕ) (non_technician_salary : ℕ) 
  (h1 : average_salary = 8000)
  (h2 : technician_count = 7)
  (h3 : technician_salary = 10000)
  (h4 : non_technician_salary = 6000) :
  ∃ (total_workers : ℕ), 
    total_workers * average_salary = 
      technician_count * technician_salary + (total_workers - technician_count) * non_technician_salary ∧
    total_workers = 14 := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l2339_233914


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l2339_233980

/-- Represents a cube with stripes on its faces -/
structure StripedCube where
  faces : Fin 6 → Bool
  -- True represents one stripe orientation, False represents the other

/-- The probability of a continuous stripe encircling the cube -/
def probability_continuous_stripe : ℚ :=
  3 / 16

/-- Theorem stating the probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  probability_continuous_stripe = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l2339_233980


namespace NUMINAMATH_CALUDE_infinite_points_satisfying_condition_l2339_233940

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2}

-- Define the diameter endpoints
def DiameterEndpoints (center : ℝ × ℝ) (radius : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((center.1 - radius, center.2), (center.1 + radius, center.2))

-- Define the condition for points P
def SatisfiesCondition (p : ℝ × ℝ) (endpoints : (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  let (a, b) := endpoints
  (p.1 - a.1)^2 + (p.2 - a.2)^2 + (p.1 - b.1)^2 + (p.2 - b.2)^2 = 5

-- Theorem statement
theorem infinite_points_satisfying_condition 
  (center : ℝ × ℝ) : 
  ∃ (s : Set (ℝ × ℝ)), 
    (∀ p ∈ s, p ∈ Circle center 2 ∧ 
              SatisfiesCondition p (DiameterEndpoints center 2)) ∧
    (Set.Infinite s) := by
  sorry

end NUMINAMATH_CALUDE_infinite_points_satisfying_condition_l2339_233940


namespace NUMINAMATH_CALUDE_all_values_equal_l2339_233955

-- Define the type for coordinates
def Coord := ℤ × ℤ

-- Define the type for the value assignment function
def ValueAssignment := Coord → ℕ

-- Define the property that each value is the average of its neighbors
def IsAverageOfNeighbors (f : ValueAssignment) : Prop :=
  ∀ (x y : ℤ), f (x, y) * 4 = f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1)

-- State the theorem
theorem all_values_equal (f : ValueAssignment) (h : IsAverageOfNeighbors f) :
  ∀ (x₁ y₁ x₂ y₂ : ℤ), f (x₁, y₁) = f (x₂, y₂) := by
  sorry

end NUMINAMATH_CALUDE_all_values_equal_l2339_233955


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2339_233916

/-- The repeating decimal 0.777... -/
def repeating_decimal : ℚ := 0.7777777

/-- The fraction 7/9 -/
def fraction : ℚ := 7/9

/-- Theorem stating that the repeating decimal 0.777... is equal to the fraction 7/9 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2339_233916


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2339_233989

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let ellipse_eccentricity := Real.sqrt (1 - (b / a) ^ 2)
  let hyperbola_eccentricity := Real.sqrt (1 + (b / a) ^ 2)
  ellipse_eccentricity = Real.sqrt 3 / 2 →
  hyperbola_eccentricity = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2339_233989


namespace NUMINAMATH_CALUDE_max_baseball_hits_percentage_l2339_233961

theorem max_baseball_hits_percentage (total_hits : ℕ) (home_runs triples doubles : ℕ) 
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 4)
  (h4 : doubles = 10) :
  (total_hits - (home_runs + triples + doubles)) / total_hits * 100 = 68 := by
  sorry

end NUMINAMATH_CALUDE_max_baseball_hits_percentage_l2339_233961


namespace NUMINAMATH_CALUDE_order_of_expressions_l2339_233906

theorem order_of_expressions :
  let a := (1/3 : ℝ) ^ Real.pi
  let b := (1/3 : ℝ) ^ (1/2)
  let c := Real.pi ^ (1/2)
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_order_of_expressions_l2339_233906


namespace NUMINAMATH_CALUDE_tie_distribution_impossibility_l2339_233941

theorem tie_distribution_impossibility 
  (B : Type) -- Set of boys
  (G : Type) -- Set of girls
  (knows : B → G → Prop) -- Relation representing who knows whom
  (color : B ⊕ G → Fin 99) -- Function assigning colors to people
  : ¬ (
    -- For any boy who knows at least 2015 girls
    (∀ b : B, (∃ (girls : Finset G), girls.card ≥ 2015 ∧ ∀ g ∈ girls, knows b g) →
      -- There are two girls among them with different colored ties
      ∃ g1 g2 : G, g1 ≠ g2 ∧ knows b g1 ∧ knows b g2 ∧ color (Sum.inr g1) ≠ color (Sum.inr g2)) ∧
    -- For any girl who knows at least 2015 boys
    (∀ g : G, (∃ (boys : Finset B), boys.card ≥ 2015 ∧ ∀ b ∈ boys, knows b g) →
      -- There are two boys among them with different colored ties
      ∃ b1 b2 : B, b1 ≠ b2 ∧ knows b1 g ∧ knows b2 g ∧ color (Sum.inl b1) ≠ color (Sum.inl b2))
  ) :=
by sorry

end NUMINAMATH_CALUDE_tie_distribution_impossibility_l2339_233941


namespace NUMINAMATH_CALUDE_dice_probabilities_l2339_233951

-- Define the type for a die
def Die : Type := Fin 6

-- Define the sample space
def SampleSpace : Type := Die × Die

-- Define the probability measure
noncomputable def P : Set SampleSpace → ℝ := sorry

-- Define the event of rolling the same number on both dice
def SameNumber : Set SampleSpace :=
  {p : SampleSpace | p.1 = p.2}

-- Define the event of rolling a sum less than 7
def SumLessThan7 : Set SampleSpace :=
  {p : SampleSpace | p.1.val + p.2.val + 2 < 7}

-- Define the event of rolling a sum equal to or greater than 11
def SumGreaterEqual11 : Set SampleSpace :=
  {p : SampleSpace | p.1.val + p.2.val + 2 ≥ 11}

theorem dice_probabilities :
  P SameNumber = 1/6 ∧
  P SumLessThan7 = 5/12 ∧
  P SumGreaterEqual11 = 1/12 := by sorry

end NUMINAMATH_CALUDE_dice_probabilities_l2339_233951


namespace NUMINAMATH_CALUDE_pie_division_l2339_233926

theorem pie_division (apple_pie : ℚ) (cherry_pie : ℚ) (people : ℕ) :
  apple_pie = 3/4 ∧ cherry_pie = 5/6 ∧ people = 3 →
  apple_pie / people = 1/4 ∧ cherry_pie / people = 5/18 := by
sorry

end NUMINAMATH_CALUDE_pie_division_l2339_233926


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l2339_233971

theorem tangent_line_to_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d ∧ y^2 = 12 * x ∧ 
   ∀ x' y' : ℝ, y' = 3 * x' + d → y'^2 ≥ 12 * x') → 
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l2339_233971


namespace NUMINAMATH_CALUDE_grocery_store_costs_l2339_233987

theorem grocery_store_costs (total_costs delivery_fraction orders_cost : ℚ)
  (h1 : total_costs = 4000)
  (h2 : orders_cost = 1800)
  (h3 : delivery_fraction = 1/4) :
  let remaining_after_orders := total_costs - orders_cost
  let delivery_cost := delivery_fraction * remaining_after_orders
  let salary_cost := remaining_after_orders - delivery_cost
  salary_cost / total_costs = 33/80 := by
sorry

end NUMINAMATH_CALUDE_grocery_store_costs_l2339_233987


namespace NUMINAMATH_CALUDE_switches_in_position_a_after_process_l2339_233964

/-- Represents a switch with its label and position -/
structure Switch where
  label : Nat
  position : Fin 4

/-- The set of all switches -/
def switches : Finset Switch :=
  sorry

/-- The process of advancing switches -/
def advance_switches : Finset Switch → Finset Switch :=
  sorry

/-- The final state after 729 steps -/
def final_state : Finset Switch :=
  sorry

/-- Count switches in position A -/
def count_position_a (s : Finset Switch) : Nat :=
  sorry

theorem switches_in_position_a_after_process :
  count_position_a final_state = 409 := by
  sorry

end NUMINAMATH_CALUDE_switches_in_position_a_after_process_l2339_233964


namespace NUMINAMATH_CALUDE_positive_real_inequality_l2339_233937

theorem positive_real_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + 2*y^2 + 3*z^2 > x*y + 3*y*z + z*x := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l2339_233937


namespace NUMINAMATH_CALUDE_t_value_l2339_233977

theorem t_value (x y t : ℝ) (h1 : 2^x = t) (h2 : 5^y = t) (h3 : 1/x + 1/y = 2) (h4 : t ≠ 1) : t = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_t_value_l2339_233977


namespace NUMINAMATH_CALUDE_hiking_club_boys_count_l2339_233986

theorem hiking_club_boys_count :
  ∀ (total_members attendance boys girls : ℕ),
  total_members = 32 →
  attendance = 22 →
  boys + girls = total_members →
  boys + (2 * girls) / 3 = attendance →
  boys = 2 := by
  sorry

end NUMINAMATH_CALUDE_hiking_club_boys_count_l2339_233986


namespace NUMINAMATH_CALUDE_divisibility_by_nine_implies_divisibility_by_three_l2339_233929

theorem divisibility_by_nine_implies_divisibility_by_three (u v : ℤ) :
  (9 ∣ u^2 + u*v + v^2) → (3 ∣ u) ∧ (3 ∣ v) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_implies_divisibility_by_three_l2339_233929


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2339_233908

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  h1 : a 6 = 12
  h2 : S 3 = 12
  h3 : ∀ n : ℕ, S n = (n / 2) * (a 1 + a n)  -- Sum formula for arithmetic sequence
  h4 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1  -- Common difference property

/-- The general term of the arithmetic sequence is 2n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n : ℕ, seq.a n = 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2339_233908


namespace NUMINAMATH_CALUDE_apps_deleted_minus_added_l2339_233907

theorem apps_deleted_minus_added (initial_apps added_apps final_apps : ℕ) : 
  initial_apps + added_apps - final_apps - added_apps = 3 :=
by
  sorry

#check apps_deleted_minus_added 32 125 29

end NUMINAMATH_CALUDE_apps_deleted_minus_added_l2339_233907


namespace NUMINAMATH_CALUDE_largest_integer_is_110_l2339_233920

theorem largest_integer_is_110 (p q r s : ℤ) 
  (sum_pqr : p + q + r = 210)
  (sum_pqs : p + q + s = 230)
  (sum_prs : p + r + s = 250)
  (sum_qrs : q + r + s = 270) :
  max p (max q (max r s)) = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_is_110_l2339_233920


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_unique_solution_of_system_l2339_233924

-- Problem 1
theorem solution_set_of_inequality (x : ℝ) :
  9 * (x - 2)^2 ≤ 25 ↔ x = 11/3 ∨ x = 1/3 :=
sorry

-- Problem 2
theorem unique_solution_of_system (x y : ℝ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_unique_solution_of_system_l2339_233924


namespace NUMINAMATH_CALUDE_odd_decreasing_properties_l2339_233935

-- Define an odd, decreasing function on ℝ
def odd_decreasing_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x y, x < y → f x > f y)

-- Theorem statement
theorem odd_decreasing_properties {f : ℝ → ℝ} {a b : ℝ} 
  (h_f : odd_decreasing_function f) (h_sum : a + b ≤ 0) : 
  (f a * f (-a) ≤ 0) ∧ (f a + f b ≥ f (-a) + f (-b)) :=
by sorry

end NUMINAMATH_CALUDE_odd_decreasing_properties_l2339_233935


namespace NUMINAMATH_CALUDE_expand_polynomial_l2339_233978

theorem expand_polynomial (x : ℝ) : (x + 3) * (4 * x^2 - 8 * x + 5) = 4 * x^3 + 4 * x^2 - 19 * x + 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l2339_233978


namespace NUMINAMATH_CALUDE_total_athletes_l2339_233958

/-- Given the ratio of players and the number of basketball players, 
    calculate the total number of athletes -/
theorem total_athletes (football baseball soccer basketball : ℕ) 
  (h_ratio : football = 10 ∧ baseball = 7 ∧ soccer = 5 ∧ basketball = 4)
  (h_basketball_players : basketball * 4 = 16) : 
  football * 4 + baseball * 4 + soccer * 4 + basketball * 4 = 104 := by
  sorry

#check total_athletes

end NUMINAMATH_CALUDE_total_athletes_l2339_233958


namespace NUMINAMATH_CALUDE_coles_return_speed_l2339_233990

/-- Calculates the average speed for the return trip given the conditions of Cole's journey -/
theorem coles_return_speed (speed_to_work : ℝ) (total_time : ℝ) (time_to_work : ℝ) : 
  speed_to_work = 60 → 
  total_time = 2 → 
  time_to_work = 1.2 → 
  (speed_to_work * time_to_work) / (total_time - time_to_work) = 90 := by
sorry

end NUMINAMATH_CALUDE_coles_return_speed_l2339_233990


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2339_233972

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x < 0}
def N : Set ℝ := {x | x > 1}

-- State the theorem
theorem intersection_M_complement_N : 
  M ∩ (Set.univ \ N) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2339_233972


namespace NUMINAMATH_CALUDE_x_squared_in_set_l2339_233913

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({0, 1, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_in_set_l2339_233913


namespace NUMINAMATH_CALUDE_evaluate_expression_l2339_233947

theorem evaluate_expression : (64 : ℝ) ^ (0.125 : ℝ) * (64 : ℝ) ^ (0.375 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2339_233947


namespace NUMINAMATH_CALUDE_matthew_crackers_l2339_233903

def crackers_problem (total_crackers : ℕ) (crackers_per_friend : ℕ) : Prop :=
  total_crackers / crackers_per_friend = 4

theorem matthew_crackers : crackers_problem 8 2 := by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_l2339_233903


namespace NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2339_233962

theorem arithmetic_progression_x_value (x : ℝ) : 
  let a₁ := 2*x - 2
  let a₂ := 2*x + 2
  let a₃ := 4*x + 4
  (a₂ - a₁ = a₃ - a₂) → x = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_x_value_l2339_233962


namespace NUMINAMATH_CALUDE_exponential_function_not_in_second_quadrant_l2339_233931

/-- A function f: ℝ → ℝ does not pass through the second quadrant if for all x < 0, f(x) ≤ 0 -/
def not_in_second_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x ≤ 0

/-- The main theorem stating the condition for f(x) = 2^x + b - 1 to not pass through the second quadrant -/
theorem exponential_function_not_in_second_quadrant (b : ℝ) :
  not_in_second_quadrant (fun x ↦ 2^x + b - 1) ↔ b ≤ 0 := by
  sorry

#check exponential_function_not_in_second_quadrant

end NUMINAMATH_CALUDE_exponential_function_not_in_second_quadrant_l2339_233931


namespace NUMINAMATH_CALUDE_min_shift_sine_graph_l2339_233910

theorem min_shift_sine_graph (φ : ℝ) : 
  (φ > 0 ∧ ∀ x, Real.sin (2*x + 2*φ + π/3) = Real.sin (2*x)) → φ ≥ 5*π/6 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_sine_graph_l2339_233910


namespace NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l2339_233915

theorem tan_sum_from_sin_cos_sum (x y : ℝ) 
  (h1 : Real.sin x + Real.sin y = 116 / 85) 
  (h2 : Real.cos x + Real.cos y = 42 / 85) : 
  Real.tan x + Real.tan y = -232992832 / 5705296111 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_from_sin_cos_sum_l2339_233915


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2339_233982

def polynomial (x : ℝ) : ℝ := 10 * x^4 - 22 * x^3 + 5 * x^2 - 8 * x - 45

def divisor (x : ℝ) : ℝ := 5 * x - 10

theorem polynomial_remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * q x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2339_233982


namespace NUMINAMATH_CALUDE_otimes_h_otimes_h_l2339_233967

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 - x*y + y^2

/-- Theorem stating the result of h ⊗ (h ⊗ h) -/
theorem otimes_h_otimes_h (h : ℝ) : otimes h (otimes h h) = h^6 - h^4 + h^3 := by
  sorry

end NUMINAMATH_CALUDE_otimes_h_otimes_h_l2339_233967


namespace NUMINAMATH_CALUDE_shaded_area_is_7pi_l2339_233943

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of circles in the problem -/
structure CircleConfiguration where
  smallCircles : List Circle
  largeCircle : Circle
  allIntersectAtTangency : Bool

/-- Calculates the area of the shaded region given a circle configuration -/
def shadedArea (config : CircleConfiguration) : ℝ :=
  sorry

/-- The main theorem stating the shaded area for the given configuration -/
theorem shaded_area_is_7pi (config : CircleConfiguration)
  (h1 : config.smallCircles.length = 13)
  (h2 : ∀ c ∈ config.smallCircles, c.radius = 1)
  (h3 : config.allIntersectAtTangency = true) :
  shadedArea config = 7 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_7pi_l2339_233943


namespace NUMINAMATH_CALUDE_line_direction_vector_l2339_233969

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := (4, 2) + t • d
  let d : ℝ × ℝ := (5 / Real.sqrt 41, 4 / Real.sqrt 41)
  ∀ x y : ℝ, x ≥ 4 →
    y = (4 * x - 6) / 5 →
    ∃ t : ℝ, 
      (x, y) = line t ∧ 
      ‖(x, y) - (4, 2)‖ = t →
        direction_vector line = d :=
  sorry

end NUMINAMATH_CALUDE_line_direction_vector_l2339_233969


namespace NUMINAMATH_CALUDE_min_production_time_proof_l2339_233997

/-- Represents the production capacity of a factory --/
structure FactoryCapacity where
  typeI : ℝ  -- Units of Type I produced per day
  typeII : ℝ -- Units of Type II produced per day

/-- Represents the total order quantity --/
structure OrderQuantity where
  typeI : ℝ
  typeII : ℝ

/-- Calculates the minimum production time given factory capacities and order quantity --/
def minProductionTime (factoryA factoryB : FactoryCapacity) (order : OrderQuantity) : ℝ :=
  sorry

/-- Theorem stating the minimum production time for the given problem --/
theorem min_production_time_proof 
  (factoryA : FactoryCapacity)
  (factoryB : FactoryCapacity)
  (order : OrderQuantity)
  (h1 : factoryA.typeI = 30 ∧ factoryA.typeII = 20)
  (h2 : factoryB.typeI = 50 ∧ factoryB.typeII = 40)
  (h3 : order.typeI = 1500 ∧ order.typeII = 800) :
  minProductionTime factoryA factoryB order = 31.25 :=
sorry

end NUMINAMATH_CALUDE_min_production_time_proof_l2339_233997


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l2339_233918

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  Real.sqrt 3 * a * Real.cos C - c * Real.sin A = 0 →
  b = 6 →
  1/2 * a * b * Real.sin C = 6 * Real.sqrt 3 →
  C = π/3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l2339_233918


namespace NUMINAMATH_CALUDE_equal_area_centroid_l2339_233992

/-- Given a triangle PQR with vertices P(4,3), Q(-1,6), and R(7,-2),
    if point S(x,y) is chosen such that triangles PQS, PRS, and QRS have equal areas,
    then 8x + 3y = 101/3 -/
theorem equal_area_centroid (x y : ℚ) : 
  let P : ℚ × ℚ := (4, 3)
  let Q : ℚ × ℚ := (-1, 6)
  let R : ℚ × ℚ := (7, -2)
  let S : ℚ × ℚ := (x, y)
  let area (A B C : ℚ × ℚ) : ℚ := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  area P Q S = area P R S ∧ area P R S = area Q R S →
  8 * x + 3 * y = 101 / 3 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_centroid_l2339_233992


namespace NUMINAMATH_CALUDE_set_equality_implies_m_zero_l2339_233960

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3*m, 3}

-- State the theorem
theorem set_equality_implies_m_zero :
  ∀ m : ℝ, A m = B m → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_zero_l2339_233960


namespace NUMINAMATH_CALUDE_max_individual_score_l2339_233976

theorem max_individual_score (total_points : ℕ) (num_players : ℕ) (min_points : ℕ) 
  (h1 : total_points = 100)
  (h2 : num_players = 12)
  (h3 : min_points = 8)
  (h4 : ∀ i : ℕ, i < num_players → min_points ≤ (total_points / num_players)) :
  ∃ max_score : ℕ, max_score = 12 ∧ 
    ∀ player_score : ℕ, player_score ≤ max_score ∧
    (num_players - 1) * min_points + max_score = total_points :=
by sorry

end NUMINAMATH_CALUDE_max_individual_score_l2339_233976


namespace NUMINAMATH_CALUDE_product_division_result_l2339_233993

theorem product_division_result : (1.6 * 0.5) / 1 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_product_division_result_l2339_233993


namespace NUMINAMATH_CALUDE_triangle_pqr_area_l2339_233911

/-- Triangle PQR with given properties -/
structure Triangle where
  inradius : ℝ
  circumradius : ℝ
  angle_relation : ℝ → ℝ → ℝ → Prop

/-- The area of a triangle given its inradius and semiperimeter -/
def triangle_area (r : ℝ) (s : ℝ) : ℝ := r * s

/-- Theorem: Area of triangle PQR with given properties -/
theorem triangle_pqr_area (T : Triangle) 
  (h_inradius : T.inradius = 6)
  (h_circumradius : T.circumradius = 17)
  (h_angle : T.angle_relation = fun P Q R => 3 * Real.cos Q = Real.cos P + Real.cos R) :
  ∃ (s : ℝ), triangle_area T.inradius s = (102 * Real.sqrt 47) / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_pqr_area_l2339_233911


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_is_18_min_value_exists_l2339_233983

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8/x + 1/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 8/a + 1/b = 1 → x + 2*y ≤ a + 2*b :=
by sorry

theorem min_value_is_18 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 8/x + 1/y = 1) :
  x + 2*y ≥ 18 :=
by sorry

theorem min_value_exists :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 8/x + 1/y = 1 ∧ x + 2*y = 18 :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_min_value_is_18_min_value_exists_l2339_233983


namespace NUMINAMATH_CALUDE_power_function_through_point_l2339_233930

/-- A power function that passes through the point (-2, 4) -/
def f : ℝ → ℝ :=
  fun x => x ^ 2

theorem power_function_through_point (h : f (-2) = 4) : f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_l2339_233930


namespace NUMINAMATH_CALUDE_arithmetic_progression_sum_l2339_233948

def arithmetic_progression (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_progression_sum
  (a : ℕ → ℚ)
  (h_ap : arithmetic_progression a)
  (h_sum1 : a 1 + a 4 + a 7 = 45)
  (h_sum2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 27 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_sum_l2339_233948


namespace NUMINAMATH_CALUDE_total_bathing_suits_l2339_233991

theorem total_bathing_suits (men_suits women_suits : ℕ) 
  (h1 : men_suits = 14797) 
  (h2 : women_suits = 4969) : 
  men_suits + women_suits = 19766 := by
  sorry

end NUMINAMATH_CALUDE_total_bathing_suits_l2339_233991


namespace NUMINAMATH_CALUDE_binary_1101001101_equals_base4_12021_l2339_233952

/-- Converts a binary (base 2) number to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a decimal number to its base 4 representation -/
def decimal_to_base4 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_1101001101_equals_base4_12021 :
  let binary : List Bool := [true, true, false, true, false, false, true, true, false, true]
  let decimal := binary_to_decimal binary
  let base4 := decimal_to_base4 decimal
  base4 = [1, 2, 0, 2, 1] := by sorry

end NUMINAMATH_CALUDE_binary_1101001101_equals_base4_12021_l2339_233952


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2339_233956

theorem complex_fraction_equality : (1 / (1 + 1 / (2 + 1 / 3))) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2339_233956


namespace NUMINAMATH_CALUDE_function_range_l2339_233932

theorem function_range (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : 
    ∀ x : ℝ, f x ∈ Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2339_233932


namespace NUMINAMATH_CALUDE_equal_pairs_comparison_l2339_233949

theorem equal_pairs_comparison : 
  (-3^5 = (-3)^5) ∧ 
  (-2^2 ≠ (-2)^2) ∧ 
  (-4 * 2^3 ≠ -4^2 * 3) ∧ 
  (-(-3)^2 ≠ -(-2)^3) :=
by sorry

end NUMINAMATH_CALUDE_equal_pairs_comparison_l2339_233949


namespace NUMINAMATH_CALUDE_hair_growth_proof_l2339_233909

/-- Calculates the additional hair growth needed for donation -/
def additional_growth_needed (current_length donation_length desired_length : ℝ) : ℝ :=
  (donation_length + desired_length) - current_length

/-- Proves that the additional hair growth needed is 21 inches -/
theorem hair_growth_proof (current_length donation_length desired_length : ℝ) 
  (h1 : current_length = 14)
  (h2 : donation_length = 23)
  (h3 : desired_length = 12) :
  additional_growth_needed current_length donation_length desired_length = 21 :=
by sorry

end NUMINAMATH_CALUDE_hair_growth_proof_l2339_233909


namespace NUMINAMATH_CALUDE_parabola_intersection_l2339_233973

/-- The points of intersection between the parabolas y = 3x^2 - 4x + 2 and y = x^3 - 2x^2 + 5x - 1 -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := x^3 - 2 * x^2 + 5 * x - 1
  ∀ x y : ℝ, f x = g x ∧ y = f x ↔ (x = 1 ∧ y = 1) ∨ (x = 3 ∧ y = 17) :=
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2339_233973


namespace NUMINAMATH_CALUDE_cards_kept_away_is_seven_l2339_233902

/-- The number of cards in a standard deck -/
def standard_deck : ℕ := 52

/-- The number of cards used in the game -/
def cards_used : ℕ := 45

/-- The number of cards kept away -/
def cards_kept_away : ℕ := standard_deck - cards_used

theorem cards_kept_away_is_seven : cards_kept_away = 7 := by
  sorry

end NUMINAMATH_CALUDE_cards_kept_away_is_seven_l2339_233902


namespace NUMINAMATH_CALUDE_unique_perimeter_l2339_233959

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  AB : ℕ+
  BC : ℕ+
  CD : ℕ+
  AD : ℕ+
  right_angle_B : True
  right_angle_C : True
  AB_equals_3 : AB = 3
  CD_equals_AD : CD = AD

/-- The perimeter of a SpecialQuadrilateral -/
def perimeter (q : SpecialQuadrilateral) : ℕ :=
  q.AB + q.BC + q.CD + q.AD

/-- Theorem stating that there's exactly one valid perimeter less than 2015 -/
theorem unique_perimeter :
  ∃! p : ℕ, p < 2015 ∧ ∃ q : SpecialQuadrilateral, perimeter q = p :=
by sorry

end NUMINAMATH_CALUDE_unique_perimeter_l2339_233959


namespace NUMINAMATH_CALUDE_chloe_winter_clothing_l2339_233996

/-- The number of boxes Chloe has -/
def num_boxes : ℕ := 4

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of winter clothing pieces Chloe has -/
def total_pieces : ℕ := num_boxes * (scarves_per_box + mittens_per_box)

theorem chloe_winter_clothing :
  total_pieces = 32 :=
by sorry

end NUMINAMATH_CALUDE_chloe_winter_clothing_l2339_233996


namespace NUMINAMATH_CALUDE_tv_sets_in_shop_d_l2339_233985

theorem tv_sets_in_shop_d (total_shops : Nat) (avg_tv_sets : Nat)
  (shop_a shop_b shop_c shop_e : Nat) :
  total_shops = 5 →
  avg_tv_sets = 48 →
  shop_a = 20 →
  shop_b = 30 →
  shop_c = 60 →
  shop_e = 50 →
  ∃ shop_d : Nat, shop_d = 80 ∧
    avg_tv_sets * total_shops = shop_a + shop_b + shop_c + shop_d + shop_e :=
by sorry

end NUMINAMATH_CALUDE_tv_sets_in_shop_d_l2339_233985


namespace NUMINAMATH_CALUDE_unique_triplet_divisibility_l2339_233998

theorem unique_triplet_divisibility :
  ∃! (a b c : ℕ), 
    (∀ n : ℕ, (∀ p < 2015, Nat.Prime p → ¬(p ∣ n)) → 
      (n + c ∣ a^n + b^n + n)) ∧
    a = 1 ∧ b = 1 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_triplet_divisibility_l2339_233998


namespace NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2339_233942

/-- Represents the color of a ball -/
inductive BallColor
| Black
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first : BallColor)
  (second : BallColor)

/-- The bag containing 2 black balls and 2 white balls -/
def bag : Multiset BallColor :=
  2 • {BallColor.Black} + 2 • {BallColor.White}

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two white balls -/
def exactlyTwoWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

theorem mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∧ exactlyTwoWhite outcome)) ∧
  (∃ outcome : DrawOutcome, exactlyOneBlack outcome ∨ exactlyTwoWhite outcome) ∧
  (∃ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∨ exactlyTwoWhite outcome)) :=
sorry

end NUMINAMATH_CALUDE_mutually_exclusive_not_contradictory_l2339_233942


namespace NUMINAMATH_CALUDE_trajectory_and_intersection_l2339_233984

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + (y+1)^2 = 4

-- Define the centers of the circles
def center1 : ℝ × ℝ := (0, 1)
def center2 : ℝ × ℝ := (0, -1)

-- Define the condition for point P
def point_condition (x y : ℝ) : Prop :=
  x ≠ 0 → ((y - 1) / x) * ((y + 1) / x) = -1/2

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- State the theorem
theorem trajectory_and_intersection :
  -- Part 1: Trajectory equation
  (∀ x y : ℝ, point_condition x y → trajectory x y) ∧
  -- Part 2: Line x = 0 intersects at two points with equal distance from C₁
  (∃ C D : ℝ × ℝ,
    C.1 = 0 ∧ D.1 = 0 ∧
    C ≠ D ∧
    trajectory C.1 C.2 ∧
    trajectory D.1 D.2 ∧
    (C.1 - center1.1)^2 + (C.2 - center1.2)^2 =
    (D.1 - center1.1)^2 + (D.2 - center1.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_intersection_l2339_233984
