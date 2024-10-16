import Mathlib

namespace NUMINAMATH_CALUDE_min_value_of_u_l1648_164877

theorem min_value_of_u (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2*y = 1) :
  (x + 1/x) * (y + 1/(4*y)) ≥ 25/8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_u_l1648_164877


namespace NUMINAMATH_CALUDE_geometric_mean_of_2_and_8_l1648_164874

theorem geometric_mean_of_2_and_8 : 
  ∃ (b : ℝ), b^2 = 2 * 8 ∧ (b = 4 ∨ b = -4) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_2_and_8_l1648_164874


namespace NUMINAMATH_CALUDE_number_of_girls_in_field_trip_l1648_164858

/-- The number of girls in a field trip given the number of students in each van and the total number of boys -/
theorem number_of_girls_in_field_trip (van1 van2 van3 van4 van5 total_boys : ℕ) 
  (h1 : van1 = 24)
  (h2 : van2 = 30)
  (h3 : van3 = 20)
  (h4 : van4 = 36)
  (h5 : van5 = 29)
  (h6 : total_boys = 64) :
  van1 + van2 + van3 + van4 + van5 - total_boys = 75 := by
  sorry

end NUMINAMATH_CALUDE_number_of_girls_in_field_trip_l1648_164858


namespace NUMINAMATH_CALUDE_problem_statement_l1648_164829

theorem problem_statement (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a-b, 0} → a^2019 + b^2019 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1648_164829


namespace NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1648_164892

theorem inequality_holds_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, a * x / (x^2 + 4) < 1.5) ↔ -6 < a ∧ a < 6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_a_in_range_l1648_164892


namespace NUMINAMATH_CALUDE_sin_theta_value_l1648_164863

theorem sin_theta_value (θ : Real) 
  (h1 : 5 * Real.tan θ = 2 * Real.cos θ) 
  (h2 : 0 < θ) (h3 : θ < Real.pi) : 
  Real.sin θ = (-5 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1648_164863


namespace NUMINAMATH_CALUDE_nice_iff_in_nice_numbers_l1648_164849

/-- A natural number is nice if it satisfies the following conditions:
  1. It is a 4-digit number
  2. The first and third digits are equal
  3. The second and fourth digits are equal
  4. The product of its digits divides its square -/
def is_nice (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  ∃ a b : ℕ, a < 10 ∧ b < 10 ∧ n = 1010 * a + 101 * b ∧
  (a * b * a * b) ∣ n^2

/-- The set of all nice numbers -/
def nice_numbers : Finset ℕ :=
  {1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1212, 2424, 3636, 4848, 1515}

/-- Theorem stating that a number is nice if and only if it belongs to the set of nice numbers -/
theorem nice_iff_in_nice_numbers (n : ℕ) : is_nice n ↔ n ∈ nice_numbers := by
  sorry

end NUMINAMATH_CALUDE_nice_iff_in_nice_numbers_l1648_164849


namespace NUMINAMATH_CALUDE_f_derivative_at_one_l1648_164871

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem f_derivative_at_one :
  ∀ x : ℝ, x ≠ 0 → f (1 / x) = x / (1 + x) →
  HasDerivAt f (-1/4) 1 :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_at_one_l1648_164871


namespace NUMINAMATH_CALUDE_bus_network_property_l1648_164873

-- Define the type for bus stops
variable {V : Type}

-- Define the "can be reached from" relation
def can_reach (G : V → V → Prop) (x y : V) : Prop := G x y

-- Define the "comes after" relation
def comes_after (G : V → V → Prop) (x y : V) : Prop :=
  ∀ z, can_reach G z x → can_reach G z y ∧ ∀ w, can_reach G y w → can_reach G x w

-- State the theorem
theorem bus_network_property (G : V → V → Prop) 
  (h : ∀ x y : V, x ≠ y → (can_reach G x y ↔ comes_after G x y)) :
  ∀ a b : V, a ≠ b → (can_reach G a b ∨ can_reach G b a) ∧ ¬(can_reach G a b ∧ can_reach G b a) :=
sorry

end NUMINAMATH_CALUDE_bus_network_property_l1648_164873


namespace NUMINAMATH_CALUDE_triangle_angle_theorem_l1648_164825

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The main theorem stating that if tan(A+B)(1-tan A tan B) = (√3 sin C) / (sin A cos B), then A = π/3. -/
theorem triangle_angle_theorem (t : Triangle) :
  Real.tan (t.A + t.B) * (1 - Real.tan t.A * Real.tan t.B) = (Real.sqrt 3 * Real.sin t.C) / (Real.sin t.A * Real.cos t.B) →
  t.A = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_theorem_l1648_164825


namespace NUMINAMATH_CALUDE_circles_are_separate_l1648_164833

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Equation of the first circle: x² + y² = r₁² -/
  c1 : ℝ → ℝ → Prop
  /-- Equation of the second circle: (x-a)² + (y-b)² = r₂² -/
  c2 : ℝ → ℝ → Prop

/-- Definition of separate circles -/
def are_separate (circles : TwoCircles) : Prop :=
  ∃ (x₁ y₁ x₂ y₂ r₁ r₂ : ℝ),
    (∀ x y, circles.c1 x y ↔ (x - x₁)^2 + (y - y₁)^2 = r₁^2) ∧
    (∀ x y, circles.c2 x y ↔ (x - x₂)^2 + (y - y₂)^2 = r₂^2) ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 > (r₁ + r₂)^2

/-- The two given circles are separate -/
theorem circles_are_separate : are_separate { 
  c1 := λ x y => x^2 + y^2 = 1,
  c2 := λ x y => (x-3)^2 + (y-4)^2 = 9
} := by sorry

end NUMINAMATH_CALUDE_circles_are_separate_l1648_164833


namespace NUMINAMATH_CALUDE_range_of_a_l1648_164845

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a*x + 1 ≥ 0) → -2 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1648_164845


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1648_164865

theorem arithmetic_mean_problem (original_list : List ℝ) 
  (a b c : ℝ) : 
  (original_list.length = 20) →
  (original_list.sum / original_list.length = 45) →
  (let new_list := original_list ++ [a, b, c]
   new_list.sum / new_list.length = 50) →
  (a + b + c) / 3 = 250 / 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1648_164865


namespace NUMINAMATH_CALUDE_line_intersection_l1648_164810

def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + (a + 2) * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := a * x - y + 2 = 0

def not_parallel (a : ℝ) : Prop :=
  ¬ (∃ k : ℝ, k ≠ 0 ∧ a = k * a ∧ (a + 2) = -k ∧ 1 = k * 2)

theorem line_intersection (a : ℝ) :
  not_parallel a → a = 0 ∨ a = -3 := by sorry

end NUMINAMATH_CALUDE_line_intersection_l1648_164810


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1648_164802

theorem inscribed_circle_radius (A p r s : ℝ) : 
  A = 2 * p →  -- Area is twice the perimeter
  A = r * s →  -- Area formula using inradius and semiperimeter
  p = 2 * s →  -- Perimeter is twice the semiperimeter
  r = 4 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1648_164802


namespace NUMINAMATH_CALUDE_intersection_of_specific_lines_l1648_164868

/-- Two lines in a plane -/
structure Line where
  slope : ℚ
  intercept : ℚ

/-- The point of intersection of two lines -/
def intersection (l1 l2 : Line) : ℚ × ℚ :=
  let x := (l2.intercept - l1.intercept) / (l1.slope - l2.slope)
  let y := l1.slope * x + l1.intercept
  (x, y)

/-- Theorem: The intersection of y = -3x + 1 and y + 1 = 15x is (1/9, 2/3) -/
theorem intersection_of_specific_lines :
  let line1 : Line := { slope := -3, intercept := 1 }
  let line2 : Line := { slope := 15, intercept := -1 }
  intersection line1 line2 = (1/9, 2/3) := by
sorry

end NUMINAMATH_CALUDE_intersection_of_specific_lines_l1648_164868


namespace NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_with_ratio_l1648_164869

/-- 
Given a quadrilateral with interior angles in a 4:5:6:7 ratio,
prove that the smallest interior angle measures 720/11 degrees.
-/
theorem smallest_angle_of_quadrilateral_with_ratio (a b c d : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- All angles are positive
  b = (5/4) * a ∧ c = (6/4) * a ∧ d = (7/4) * a →  -- Angles are in 4:5:6:7 ratio
  a + b + c + d = 360 →  -- Sum of angles in a quadrilateral is 360°
  a = 720 / 11 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_with_ratio_l1648_164869


namespace NUMINAMATH_CALUDE_area_constants_sum_l1648_164808

/-- Represents a grid with squares and overlapping circles -/
structure GridWithCircles where
  grid_size : Nat
  square_size : ℝ
  circle_diameter : ℝ
  circle_center_distance : ℝ

/-- Calculates the constants C and D for the area of visible shaded region -/
def calculate_area_constants (g : GridWithCircles) : ℝ × ℝ :=
  sorry

/-- The theorem stating that C + D = 150 for the given configuration -/
theorem area_constants_sum (g : GridWithCircles) 
  (h1 : g.grid_size = 4)
  (h2 : g.square_size = 3)
  (h3 : g.circle_diameter = 6)
  (h4 : g.circle_center_distance = 3) :
  let (C, D) := calculate_area_constants g
  C + D = 150 :=
sorry

end NUMINAMATH_CALUDE_area_constants_sum_l1648_164808


namespace NUMINAMATH_CALUDE_inequality_of_four_terms_l1648_164862

theorem inequality_of_four_terms (x y z w : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) :
  Real.sqrt (x / (y + z + x)) + Real.sqrt (y / (x + z + w)) +
  Real.sqrt (z / (x + y + w)) + Real.sqrt (w / (x + y + z)) > 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_of_four_terms_l1648_164862


namespace NUMINAMATH_CALUDE_tank_fill_time_l1648_164836

/-- Proves that the time required to fill 3/4 of a 4000-gallon tank at a rate of 10 gallons per hour is 300 hours. -/
theorem tank_fill_time (tank_capacity : ℝ) (fill_rate : ℝ) (fill_fraction : ℝ) (fill_time : ℝ) :
  tank_capacity = 4000 →
  fill_rate = 10 →
  fill_fraction = 3/4 →
  fill_time = (fill_fraction * tank_capacity) / fill_rate →
  fill_time = 300 :=
by
  sorry

end NUMINAMATH_CALUDE_tank_fill_time_l1648_164836


namespace NUMINAMATH_CALUDE_monomial_simplification_l1648_164875

theorem monomial_simplification (a : ℝ) (h : a = 100) :
  a / (a + 1) - 1 / (a^2 + a) = 99 / 100 := by
  sorry

end NUMINAMATH_CALUDE_monomial_simplification_l1648_164875


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_equals_result_l1648_164899

-- Define the universe set U
def U : Set ℤ := {x : ℤ | x^2 - x - 12 ≤ 0}

-- Define set A
def A : Set ℤ := {-2, -1, 3}

-- Define set B
def B : Set ℤ := {0, 1, 3, 4}

-- Define the result set
def result : Set ℤ := {0, 1, 4}

-- Theorem statement
theorem complement_A_intersect_B_equals_result :
  (U \ A) ∩ B = result := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_equals_result_l1648_164899


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l1648_164834

theorem arithmetic_mean_of_four_numbers :
  let numbers : List ℝ := [12, 25, 39, 48]
  (numbers.sum / numbers.length : ℝ) = 31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_four_numbers_l1648_164834


namespace NUMINAMATH_CALUDE_array_sum_mod_1004_l1648_164807

/-- Represents the sum of all terms in a 1/q-array as described in the problem -/
def array_sum (q : ℕ) : ℚ :=
  (3 * q^2 : ℚ) / ((3*q - 1) * (q - 1))

/-- The theorem stating that the sum of all terms in a 1/1004-array is congruent to 1 modulo 1004 -/
theorem array_sum_mod_1004 :
  ∃ (n : ℕ), array_sum 1004 = (n * 1004 + 1 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_array_sum_mod_1004_l1648_164807


namespace NUMINAMATH_CALUDE_wilson_prime_l1648_164898

theorem wilson_prime (n : ℕ) (h : n > 1) (h_div : n ∣ (Nat.factorial (n - 1) + 1)) : Nat.Prime n := by
  sorry

end NUMINAMATH_CALUDE_wilson_prime_l1648_164898


namespace NUMINAMATH_CALUDE_distance_difference_l1648_164826

theorem distance_difference (john_distance jill_distance jim_distance : ℝ) : 
  john_distance = 15 →
  jim_distance = 0.2 * jill_distance →
  jim_distance = 2 →
  john_distance - jill_distance = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l1648_164826


namespace NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1648_164886

/-- A rhombus with side length 34 units and shorter diagonal 32 units has a longer diagonal of 60 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 34 → shorter_diagonal = 32 → longer_diagonal = 60 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_longer_diagonal_l1648_164886


namespace NUMINAMATH_CALUDE_equation_solutions_l1648_164839

theorem equation_solutions : 
  ∀ m : ℝ, 9 * m^2 - (2*m + 1)^2 = 0 ↔ m = 1 ∨ m = -1/5 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1648_164839


namespace NUMINAMATH_CALUDE_sean_total_apples_l1648_164835

def initial_apples : ℕ := 9
def apples_per_day : ℕ := 8
def days : ℕ := 5

theorem sean_total_apples :
  initial_apples + apples_per_day * days = 49 := by
  sorry

end NUMINAMATH_CALUDE_sean_total_apples_l1648_164835


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1648_164821

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (d : ℝ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l1648_164821


namespace NUMINAMATH_CALUDE_find_constant_a_l1648_164855

theorem find_constant_a (t k a : ℝ) :
  (∀ x : ℝ, x^2 + 10*x + t = (x + a)^2 + k) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_constant_a_l1648_164855


namespace NUMINAMATH_CALUDE_no_integer_root_quadratic_pair_l1648_164884

theorem no_integer_root_quadratic_pair :
  ¬ ∃ (a b c : ℤ),
    (∃ (x₁ x₂ : ℤ), a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ ≠ x₂) ∧
    (∃ (y₁ y₂ : ℤ), (a + 1) * y₁^2 + (b + 1) * y₁ + (c + 1) = 0 ∧ (a + 1) * y₂^2 + (b + 1) * y₂ + (c + 1) = 0 ∧ y₁ ≠ y₂) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_root_quadratic_pair_l1648_164884


namespace NUMINAMATH_CALUDE_factorization_a_cubed_minus_9a_l1648_164820

theorem factorization_a_cubed_minus_9a (a : ℝ) : a^3 - 9*a = a*(a+3)*(a-3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_cubed_minus_9a_l1648_164820


namespace NUMINAMATH_CALUDE_function_value_at_half_l1648_164846

def real_function_property (f : ℝ → ℝ) : Prop :=
  f 1 = -1 ∧ ∀ x y : ℝ, f (x * y + f x) = x * f y + f x

theorem function_value_at_half (f : ℝ → ℝ) (h : real_function_property f) : f (1/2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_half_l1648_164846


namespace NUMINAMATH_CALUDE_lillian_candy_count_l1648_164842

/-- The total number of candies Lillian has after receiving candies from her father and best friend. -/
def lillian_total_candies (initial : ℕ) (father_gave : ℕ) (friend_multiplier : ℕ) : ℕ :=
  initial + father_gave + friend_multiplier * father_gave

/-- Theorem stating that Lillian will have 113 candies given the initial conditions. -/
theorem lillian_candy_count :
  lillian_total_candies 88 5 4 = 113 := by
  sorry

#eval lillian_total_candies 88 5 4

end NUMINAMATH_CALUDE_lillian_candy_count_l1648_164842


namespace NUMINAMATH_CALUDE_statement_c_not_always_true_l1648_164887

theorem statement_c_not_always_true :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_statement_c_not_always_true_l1648_164887


namespace NUMINAMATH_CALUDE_room_age_distribution_l1648_164890

theorem room_age_distribution (P : ℕ) (h1 : 50 < P) (h2 : P < 100)
  (h3 : (5 : ℚ) / 12 * P = P - 36) :
  (36 : ℚ) / P = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_room_age_distribution_l1648_164890


namespace NUMINAMATH_CALUDE_randy_money_left_l1648_164830

theorem randy_money_left (initial_amount : ℝ) (lunch_cost : ℝ) (ice_cream_fraction : ℝ) : 
  initial_amount = 30 →
  lunch_cost = 10 →
  ice_cream_fraction = 1/4 →
  initial_amount - lunch_cost - (initial_amount - lunch_cost) * ice_cream_fraction = 15 := by
sorry

end NUMINAMATH_CALUDE_randy_money_left_l1648_164830


namespace NUMINAMATH_CALUDE_value_of_T_l1648_164879

theorem value_of_T : ∃ T : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * T = (1/4 : ℝ) * (1/8 : ℝ) * 120 ∧ T = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_T_l1648_164879


namespace NUMINAMATH_CALUDE_solution_to_equation_l1648_164859

theorem solution_to_equation : ∃ x : ℤ, (2010 + x)^2 = x^2 ∧ x = -1005 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1648_164859


namespace NUMINAMATH_CALUDE_second_train_crossing_time_l1648_164861

/-- Represents the time in seconds for two bullet trains to cross each other -/
def crossing_time : ℝ := 16.666666666666668

/-- Represents the length of each bullet train in meters -/
def train_length : ℝ := 120

/-- Represents the time in seconds for the first bullet train to cross a telegraph post -/
def first_train_time : ℝ := 10

theorem second_train_crossing_time :
  let first_train_speed := train_length / first_train_time
  let second_train_time := train_length / ((2 * train_length / crossing_time) - first_train_speed)
  second_train_time = 50 := by sorry

end NUMINAMATH_CALUDE_second_train_crossing_time_l1648_164861


namespace NUMINAMATH_CALUDE_am_gm_inequality_l1648_164843

theorem am_gm_inequality (a b : ℝ) (h : a * b > 0) : a / b + b / a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_am_gm_inequality_l1648_164843


namespace NUMINAMATH_CALUDE_medical_team_selection_l1648_164840

theorem medical_team_selection (male_doctors female_doctors team_size : ℕ) 
  (h1 : male_doctors = 6)
  (h2 : female_doctors = 3)
  (h3 : team_size = 5) :
  (Nat.choose (male_doctors + female_doctors) team_size) - 
  (Nat.choose male_doctors team_size) = 120 := by
  sorry

end NUMINAMATH_CALUDE_medical_team_selection_l1648_164840


namespace NUMINAMATH_CALUDE_train_crossing_problem_l1648_164852

/-- Calculates the length of a train given the length and speed of another train, 
    their crossing time, and the speed of the train we're calculating. -/
def train_length (other_length : ℝ) (other_speed : ℝ) (this_speed : ℝ) (cross_time : ℝ) : ℝ :=
  ((other_speed + this_speed) * cross_time - other_length)

theorem train_crossing_problem : 
  let first_train_length : ℝ := 290
  let first_train_speed : ℝ := 120 * 1000 / 3600  -- Convert km/h to m/s
  let second_train_speed : ℝ := 80 * 1000 / 3600  -- Convert km/h to m/s
  let crossing_time : ℝ := 9
  abs (train_length first_train_length first_train_speed second_train_speed crossing_time - 209.95) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_problem_l1648_164852


namespace NUMINAMATH_CALUDE_soup_problem_solution_l1648_164817

/-- Represents the number of people a can of soup can feed -/
structure SoupCan where
  adults : ℕ
  children : ℕ

/-- Represents the problem setup -/
structure SoupProblem where
  can : SoupCan
  totalCans : ℕ
  childrenFed : ℕ

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (problem : SoupProblem) : ℕ :=
  let cansUsedForChildren := problem.childrenFed / problem.can.children
  let remainingCans := problem.totalCans - cansUsedForChildren
  remainingCans * problem.can.adults

/-- Theorem stating that given the problem conditions, 12 adults can be fed with the remaining soup -/
theorem soup_problem_solution (problem : SoupProblem) 
  (h1 : problem.can.adults = 4)
  (h2 : problem.can.children = 6)
  (h3 : problem.totalCans = 6)
  (h4 : problem.childrenFed = 18) :
  remainingAdults problem = 12 := by
  sorry

#eval remainingAdults { can := { adults := 4, children := 6 }, totalCans := 6, childrenFed := 18 }

end NUMINAMATH_CALUDE_soup_problem_solution_l1648_164817


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l1648_164838

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, b.1 = t * a.1 ∧ b.2 = t * a.2

theorem parallel_vectors_k_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (k, 4)
  are_parallel a b → k = -2 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l1648_164838


namespace NUMINAMATH_CALUDE_spade_operation_result_l1648_164876

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result :
  spade 3 (spade 5 (spade 8 12)) = 2 := by sorry

end NUMINAMATH_CALUDE_spade_operation_result_l1648_164876


namespace NUMINAMATH_CALUDE_cookies_calculation_l1648_164857

/-- The number of people receiving cookies -/
def num_people : ℝ := 6.0

/-- The number of cookies each person should receive -/
def cookies_per_person : ℝ := 24.0

/-- The total number of cookies needed -/
def total_cookies : ℝ := num_people * cookies_per_person

theorem cookies_calculation : total_cookies = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_cookies_calculation_l1648_164857


namespace NUMINAMATH_CALUDE_cost_verification_max_purchase_l1648_164832

/-- Represents the cost of a single bat -/
def bat_cost : ℝ := 70

/-- Represents the cost of a single ball -/
def ball_cost : ℝ := 20

/-- Represents the discount rate when purchasing at least 3 bats and 3 balls -/
def discount_rate : ℝ := 0.10

/-- Represents the sales tax rate -/
def sales_tax_rate : ℝ := 0.08

/-- Represents the budget -/
def budget : ℝ := 270

/-- Verifies that the given costs satisfy the conditions -/
theorem cost_verification : 
  2 * bat_cost + 4 * ball_cost = 220 ∧ 
  bat_cost + 6 * ball_cost = 190 := by sorry

/-- Proves that the maximum number of bats and balls that can be purchased is 3 -/
theorem max_purchase : 
  ∀ n : ℕ, 
    n ≥ 3 → 
    n * (bat_cost + ball_cost) * (1 - discount_rate) * (1 + sales_tax_rate) ≤ budget → 
    n ≤ 3 := by sorry

end NUMINAMATH_CALUDE_cost_verification_max_purchase_l1648_164832


namespace NUMINAMATH_CALUDE_cos_75_degrees_l1648_164801

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l1648_164801


namespace NUMINAMATH_CALUDE_maci_school_supplies_cost_l1648_164806

/-- Calculates the total cost of school supplies with discounts applied --/
def calculate_total_cost (blue_pen_count : ℕ) (red_pen_count : ℕ) (pencil_count : ℕ) (notebook_count : ℕ) 
  (blue_pen_price : ℚ) (pen_discount_threshold : ℕ) (pen_discount_rate : ℚ) 
  (notebook_discount_threshold : ℕ) (notebook_discount_rate : ℚ) : ℚ :=
  let red_pen_price := 2 * blue_pen_price
  let pencil_price := red_pen_price / 2
  let notebook_price := 10 * blue_pen_price
  
  let total_pen_cost := blue_pen_count * blue_pen_price + red_pen_count * red_pen_price
  let pencil_cost := pencil_count * pencil_price
  let notebook_cost := notebook_count * notebook_price
  
  let pen_discount := if blue_pen_count + red_pen_count > pen_discount_threshold 
                      then pen_discount_rate * total_pen_cost 
                      else 0
  let notebook_discount := if notebook_count > notebook_discount_threshold 
                           then notebook_discount_rate * notebook_cost 
                           else 0
  
  total_pen_cost + pencil_cost + notebook_cost - pen_discount - notebook_discount

/-- Theorem stating that the total cost of Maci's school supplies is $7.10 --/
theorem maci_school_supplies_cost :
  calculate_total_cost 10 15 5 3 (10/100) 12 (10/100) 4 (20/100) = 71/10 := by
  sorry

end NUMINAMATH_CALUDE_maci_school_supplies_cost_l1648_164806


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_eq_7_l1648_164888

theorem log_sqrt10_1000sqrt10_eq_7 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_eq_7_l1648_164888


namespace NUMINAMATH_CALUDE_merchant_profit_percentage_l1648_164891

/-- Calculates the profit percentage for a merchant who marks up goods by 50%
    and then offers a 10% discount on the marked price. -/
theorem merchant_profit_percentage (cost_price : ℝ) (cost_price_pos : 0 < cost_price) :
  let markup_percentage : ℝ := 0.5
  let discount_percentage : ℝ := 0.1
  let marked_price : ℝ := cost_price * (1 + markup_percentage)
  let selling_price : ℝ := marked_price * (1 - discount_percentage)
  let profit : ℝ := selling_price - cost_price
  let profit_percentage : ℝ := profit / cost_price * 100
  profit_percentage = 35 := by
  sorry


end NUMINAMATH_CALUDE_merchant_profit_percentage_l1648_164891


namespace NUMINAMATH_CALUDE_sofa_loveseat_cost_ratio_l1648_164819

theorem sofa_loveseat_cost_ratio :
  ∀ (sofa_cost loveseat_cost : ℕ),
    loveseat_cost = 148 →
    sofa_cost + loveseat_cost = 444 →
    ∃ (n : ℕ), sofa_cost = n * loveseat_cost →
    sofa_cost / loveseat_cost = 2 :=
by sorry

end NUMINAMATH_CALUDE_sofa_loveseat_cost_ratio_l1648_164819


namespace NUMINAMATH_CALUDE_patio_tiles_l1648_164805

theorem patio_tiles (c : ℕ) (h1 : c > 2) : 
  c * 10 = (c - 2) * (10 + 4) → c * 10 = 70 := by
  sorry

#check patio_tiles

end NUMINAMATH_CALUDE_patio_tiles_l1648_164805


namespace NUMINAMATH_CALUDE_find_k_value_l1648_164860

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l1648_164860


namespace NUMINAMATH_CALUDE_olympic_medal_scenario_l1648_164815

/-- The number of ways to award medals in the Olympic 100-meter sprint -/
def olympic_medal_ways (total_athletes : ℕ) (european_athletes : ℕ) (asian_athletes : ℕ) (max_european_medals : ℕ) : ℕ :=
  -- Define the function here
  sorry

/-- Theorem: The number of ways to award medals in the given Olympic scenario is 588 -/
theorem olympic_medal_scenario : olympic_medal_ways 10 4 6 2 = 588 := by
  sorry

end NUMINAMATH_CALUDE_olympic_medal_scenario_l1648_164815


namespace NUMINAMATH_CALUDE_perp_condition_l1648_164812

/-- Two lines in the plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.slope * l2.slope = -1

/-- The first line x + y = 0 -/
def line1 : Line := { slope := -1, intercept := 0 }

/-- The second line x - ay = 0 -/
def line2 (a : ℝ) : Line := { slope := a, intercept := 0 }

/-- Theorem: a = 1 is necessary and sufficient for perpendicularity -/
theorem perp_condition (a : ℝ) :
  perpendicular line1 (line2 a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perp_condition_l1648_164812


namespace NUMINAMATH_CALUDE_chess_club_boys_count_l1648_164897

theorem chess_club_boys_count :
  ∀ (G B : ℕ),
  G + B = 30 →
  (2 * G) / 3 + (3 * B) / 4 = 18 →
  B = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_chess_club_boys_count_l1648_164897


namespace NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l1648_164889

theorem common_roots_cubic_polynomials (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    (r^3 + a*r^2 + 17*r + 10 = 0) ∧ 
    (r^3 + b*r^2 + 20*r + 12 = 0) ∧
    (s^3 + a*s^2 + 17*s + 10 = 0) ∧ 
    (s^3 + b*s^2 + 20*s + 12 = 0)) →
  a = -6 ∧ b = -7 := by
sorry

end NUMINAMATH_CALUDE_common_roots_cubic_polynomials_l1648_164889


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_three_l1648_164880

theorem subset_implies_m_equals_three (A B : Set ℝ) (m : ℝ) :
  A = {1, 3} →
  B = {1, 2, m} →
  A ⊆ B →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_three_l1648_164880


namespace NUMINAMATH_CALUDE_intersection_with_complement_l1648_164864

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l1648_164864


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1648_164896

theorem cubic_roots_sum (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 →
  b^3 - 6*b^2 + 11*b - 6 = 0 →
  c^3 - 6*c^2 + 11*c - 6 = 0 →
  a * b / c + b * c / a + c * a / b = 49 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1648_164896


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1648_164870

/-- Given a geometric sequence {aₙ} where aₙ ∈ ℝ, if a₃ and a₁₁ are the two roots of the equation 3x² - 25x + 27 = 0, then a₇ = 3. -/
theorem geometric_sequence_problem (a : ℕ → ℝ) (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_roots : 3 * (a 3)^2 - 25 * (a 3) + 27 = 0 ∧ 3 * (a 11)^2 - 25 * (a 11) + 27 = 0) :
  a 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1648_164870


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l1648_164866

theorem triangle_max_perimeter (a b y : ℕ) (ha : a = 7) (hb : b = 9) :
  (∃ (y : ℕ), a + b + y = (a + b + y).max (a + b + (a + b - 1))) :=
sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l1648_164866


namespace NUMINAMATH_CALUDE_average_book_width_l1648_164878

def book_widths : List ℝ := [6, 50, 1, 35, 3, 5, 75, 20]

theorem average_book_width :
  let total_width := book_widths.sum
  let num_books := book_widths.length
  total_width / num_books = 24.375 := by
sorry

end NUMINAMATH_CALUDE_average_book_width_l1648_164878


namespace NUMINAMATH_CALUDE_two_thousand_eighth_number_without_two_l1648_164847

/-- A function that checks if a positive integer contains the digit 2 -/
def containsTwo (n : ℕ) : Bool :=
  String.contains (toString n) '2'

/-- A function that generates the sequence of numbers without 2 -/
def sequenceWithoutTwo : ℕ → ℕ
  | 0 => 0
  | n + 1 => if containsTwo (sequenceWithoutTwo n + 1)
              then sequenceWithoutTwo n + 2
              else sequenceWithoutTwo n + 1

theorem two_thousand_eighth_number_without_two :
  sequenceWithoutTwo 2008 = 3781 := by
  sorry

end NUMINAMATH_CALUDE_two_thousand_eighth_number_without_two_l1648_164847


namespace NUMINAMATH_CALUDE_equation_system_solution_l1648_164823

theorem equation_system_solution (a b : ℝ) : 
  ((a / 4 - 1) + 2 * (b / 3 + 2) = 4 ∧ 2 * (a / 4 - 1) + (b / 3 + 2) = 5) → 
  (a = 12 ∧ b = -3) := by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l1648_164823


namespace NUMINAMATH_CALUDE_linear_dependence_implies_k₁_plus_4k₃_eq_zero_l1648_164831

def a₁ : Fin 2 → ℝ := ![1, 0]
def a₂ : Fin 2 → ℝ := ![1, -1]
def a₃ : Fin 2 → ℝ := ![2, 2]

theorem linear_dependence_implies_k₁_plus_4k₃_eq_zero :
  ∃ (k₁ k₂ k₃ : ℝ), (k₁ ≠ 0 ∨ k₂ ≠ 0 ∨ k₃ ≠ 0) ∧
    (∀ i : Fin 2, k₁ * a₁ i + k₂ * a₂ i + k₃ * a₃ i = 0) →
  ∀ (k₁ k₂ k₃ : ℝ), (∀ i : Fin 2, k₁ * a₁ i + k₂ * a₂ i + k₃ * a₃ i = 0) →
  k₁ + 4 * k₃ = 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_dependence_implies_k₁_plus_4k₃_eq_zero_l1648_164831


namespace NUMINAMATH_CALUDE_dog_feed_mixture_l1648_164850

/-- Represents the dog feed mixture problem -/
theorem dog_feed_mixture :
  let cheap_price : ℝ := 0.17
  let expensive_price : ℝ := 0.36
  let mixture_price : ℝ := 0.26
  let cheap_amount : ℝ := 14.2105263158
  let expensive_amount : ℝ := (mixture_price * (cheap_amount + expensive_amount) - cheap_price * cheap_amount) / (expensive_price - mixture_price)
  cheap_amount + expensive_amount = 27 := by
  sorry

end NUMINAMATH_CALUDE_dog_feed_mixture_l1648_164850


namespace NUMINAMATH_CALUDE_customer_satisfaction_probability_l1648_164816

/-- The probability that a dissatisfied customer leaves an angry review -/
def prob_dissatisfied_angry : ℝ := 0.80

/-- The probability that a satisfied customer leaves a positive review -/
def prob_satisfied_positive : ℝ := 0.15

/-- The number of angry reviews received -/
def num_angry_reviews : ℕ := 60

/-- The number of positive reviews received -/
def num_positive_reviews : ℕ := 20

/-- The probability that a customer is satisfied -/
def prob_satisfied : ℝ := 0.64

theorem customer_satisfaction_probability :
  prob_satisfied = 0.64 :=
sorry

end NUMINAMATH_CALUDE_customer_satisfaction_probability_l1648_164816


namespace NUMINAMATH_CALUDE_common_tangent_implies_t_value_l1648_164854

noncomputable def f (t : ℝ) (x : ℝ) : ℝ := t * Real.log x
def g (x : ℝ) : ℝ := x^2 - 1

theorem common_tangent_implies_t_value :
  ∀ t : ℝ,
  (f t 1 = g 1) →
  (deriv (f t) 1 = deriv g 1) →
  t = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_common_tangent_implies_t_value_l1648_164854


namespace NUMINAMATH_CALUDE_minimum_fencing_cost_theorem_l1648_164867

/-- Represents the cost per linear foot for different fencing materials -/
structure FencingMaterial where
  wood : ℝ
  chainLink : ℝ
  iron : ℝ

/-- Calculates the minimum fencing cost for a rectangular field -/
def minimumFencingCost (area : ℝ) (uncoveredSide : ℝ) (materials : FencingMaterial) : ℝ :=
  sorry

/-- Theorem stating the minimum fencing cost for the given problem -/
theorem minimum_fencing_cost_theorem :
  let area : ℝ := 680
  let uncoveredSide : ℝ := 34
  let materials : FencingMaterial := { wood := 5, chainLink := 7, iron := 10 }
  minimumFencingCost area uncoveredSide materials = 438 := by
  sorry

end NUMINAMATH_CALUDE_minimum_fencing_cost_theorem_l1648_164867


namespace NUMINAMATH_CALUDE_unique_divisible_by_45_l1648_164851

def is_divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

def digit (d : ℕ) : Prop := d ≥ 0 ∧ d ≤ 9

def five_digit_number (x : ℕ) : ℕ := x * 10000 + 2 * 1000 + 7 * 100 + x * 10 + 5

theorem unique_divisible_by_45 : 
  ∃! x : ℕ, digit x ∧ is_divisible_by (five_digit_number x) 45 ∧ x = 2 := by sorry

end NUMINAMATH_CALUDE_unique_divisible_by_45_l1648_164851


namespace NUMINAMATH_CALUDE_max_non_functional_segments_is_13_l1648_164837

/-- Represents a seven-segment display --/
structure SevenSegmentDisplay :=
  (segments : Fin 7 → Bool)

/-- Represents a four-digit clock display --/
structure ClockDisplay :=
  (digits : Fin 4 → SevenSegmentDisplay)

/-- The set of valid digits for each position --/
def validDigits : Fin 4 → Set ℕ
  | 0 => {0, 1, 2}
  | 1 => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
  | 2 => {0, 1, 2, 3, 4, 5}
  | 3 => {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

/-- A function that determines if a time can be unambiguously read --/
def isUnambiguous (display : ClockDisplay) : Prop := sorry

/-- The maximum number of non-functional segments --/
def maxNonFunctionalSegments : ℕ := 13

/-- The main theorem --/
theorem max_non_functional_segments_is_13 :
  ∀ (display : ClockDisplay),
    (∀ (i : Fin 4), ∃ (d : ℕ), d ∈ validDigits i ∧ isUnambiguous display) →
    (∃ (n : ℕ), n = maxNonFunctionalSegments ∧
      ∀ (m : ℕ), m > n →
        ¬(∀ (i : Fin 4), ∃ (d : ℕ), d ∈ validDigits i ∧ isUnambiguous display)) :=
by sorry

end NUMINAMATH_CALUDE_max_non_functional_segments_is_13_l1648_164837


namespace NUMINAMATH_CALUDE_sheet_width_correct_l1648_164814

/-- The width of a rectangular metallic sheet -/
def sheet_width : ℝ := 36

/-- The length of the rectangular metallic sheet -/
def sheet_length : ℝ := 48

/-- The side length of the square cut from each corner -/
def cut_square_side : ℝ := 8

/-- The volume of the resulting open box -/
def box_volume : ℝ := 5120

/-- Theorem stating that the given dimensions result in the correct volume -/
theorem sheet_width_correct : 
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = box_volume :=
by sorry

end NUMINAMATH_CALUDE_sheet_width_correct_l1648_164814


namespace NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l1648_164804

theorem cube_edge_length_from_circumscribed_sphere_volume 
  (V : ℝ) (a : ℝ) (h : V = (32 / 3) * Real.pi) :
  (V = (4 / 3) * Real.pi * (a * Real.sqrt 3 / 2)^3) → a = (4 * Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_edge_length_from_circumscribed_sphere_volume_l1648_164804


namespace NUMINAMATH_CALUDE_min_degree_of_g_l1648_164893

/-- Given polynomials f, g, and h satisfying the equation 5f + 6g = h,
    where deg(f) = 10 and deg(h) = 11, the minimum possible degree of g is 11. -/
theorem min_degree_of_g (f g h : Polynomial ℝ)
  (eq : 5 • f + 6 • g = h)
  (deg_f : Polynomial.degree f = 10)
  (deg_h : Polynomial.degree h = 11) :
  Polynomial.degree g ≥ 11 ∧ ∃ (g' : Polynomial ℝ), Polynomial.degree g' = 11 ∧ 5 • f + 6 • g' = h :=
sorry

end NUMINAMATH_CALUDE_min_degree_of_g_l1648_164893


namespace NUMINAMATH_CALUDE_quadratic_root_in_arithmetic_sequence_l1648_164881

/-- Given real numbers x, y, z forming an arithmetic sequence with x ≥ y ≥ z ≥ 0,
    if the quadratic zx^2 + yx + x has exactly one root, then this root is -3/4. -/
theorem quadratic_root_in_arithmetic_sequence (x y z : ℝ) :
  (∃ d : ℝ, y = x - d ∧ z = x - 2*d) →  -- arithmetic sequence condition
  x ≥ y →
  y ≥ z →
  z ≥ 0 →
  (∃! r : ℝ, z*r^2 + y*r + x = 0) →  -- exactly one root condition
  (∃ r : ℝ, z*r^2 + y*r + x = 0 ∧ r = -3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_in_arithmetic_sequence_l1648_164881


namespace NUMINAMATH_CALUDE_cannot_obtain_five_equal_numbers_l1648_164800

/-- Represents the set of numbers on the board -/
def BoardNumbers : Finset Int := {2, 3, 5, 7, 11}

/-- The operation of replacing two numbers with their arithmetic mean -/
def replaceWithMean (a b : Int) : Int := (a + b) / 2

/-- Predicate to check if two numbers have the same parity -/
def sameParity (a b : Int) : Prop := a % 2 = b % 2

/-- Theorem stating that it's impossible to obtain five equal numbers -/
theorem cannot_obtain_five_equal_numbers :
  ¬ ∃ (n : Int), ∃ (k : ℕ), ∃ (operations : Fin k → Int × Int),
    (∀ i, sameParity (operations i).1 (operations i).2) ∧
    (Finset.sum BoardNumbers id = 5 * n) ∧
    (∀ x ∈ BoardNumbers, x = n) :=
sorry

end NUMINAMATH_CALUDE_cannot_obtain_five_equal_numbers_l1648_164800


namespace NUMINAMATH_CALUDE_expression_calculation_l1648_164856

theorem expression_calculation : (75 * 2024 - 25 * 2024) / 2 = 50600 := by
  sorry

end NUMINAMATH_CALUDE_expression_calculation_l1648_164856


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1648_164882

theorem inequality_solution_set :
  {x : ℝ | |x - 1| + |2*x + 5| < 8} = Set.Ioo (-4 : ℝ) (4/3) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1648_164882


namespace NUMINAMATH_CALUDE_clubsuit_ratio_l1648_164824

-- Define the ♣ operation
def clubsuit (n m : ℕ) : ℕ := n^2 * m^3

-- Theorem statement
theorem clubsuit_ratio : (clubsuit 3 5) / (clubsuit 5 3) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_ratio_l1648_164824


namespace NUMINAMATH_CALUDE_backpack_pencilcase_combinations_l1648_164813

/-- The number of combinations formed by selecting one item from each of two sets -/
def combinations (set1 : ℕ) (set2 : ℕ) : ℕ := set1 * set2

/-- Theorem: The number of combinations formed by selecting one backpack from 2 styles
    and one pencil case from 2 styles is equal to 4 -/
theorem backpack_pencilcase_combinations :
  let backpacks : ℕ := 2
  let pencilcases : ℕ := 2
  combinations backpacks pencilcases = 4 := by
  sorry

end NUMINAMATH_CALUDE_backpack_pencilcase_combinations_l1648_164813


namespace NUMINAMATH_CALUDE_ellipse_intersection_product_l1648_164885

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Definition of the right focus F2 -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Definition of the left vertex A -/
def left_vertex : ℝ × ℝ := (-2, 0)

/-- Definition of a line passing through a point -/
def line_through (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

/-- Definition of intersection of a line with x = 4 -/
def intersect_x_4 (m : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (4, m * (4 - p.1) + p.2)

/-- Main theorem -/
theorem ellipse_intersection_product (l m n : ℝ) (P Q : ℝ × ℝ) :
  line_through l right_focus P.1 P.2 →
  line_through l right_focus Q.1 Q.2 →
  ellipse_C P.1 P.2 →
  ellipse_C Q.1 Q.2 →
  let M := intersect_x_4 m (left_vertex.1, left_vertex.2)
  let N := intersect_x_4 n (left_vertex.1, left_vertex.2)
  line_through m left_vertex P.1 P.2 →
  line_through n left_vertex Q.1 Q.2 →
  M.2 * N.2 = -9 :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_product_l1648_164885


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l1648_164872

theorem similar_triangles_shortest_side
  (a b c : ℝ)  -- sides of the first triangle
  (k : ℝ)      -- scaling factor
  (h1 : a^2 + b^2 = c^2)  -- Pythagorean theorem for the first triangle
  (h2 : a = 15)           -- given side length of the first triangle
  (h3 : c = 17)           -- hypotenuse of the first triangle
  (h4 : k * c = 68)       -- hypotenuse of the second triangle
  : k * min a b = 32 :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l1648_164872


namespace NUMINAMATH_CALUDE_special_function_property_l1648_164822

/-- A function satisfying g(xy) = g(x)/y for all positive real numbers x and y -/
def SpecialFunction (g : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → g (x * y) = g x / y

theorem special_function_property (g : ℝ → ℝ) 
    (h1 : SpecialFunction g) 
    (h2 : g 30 = 30) : 
    g 45 = 20 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l1648_164822


namespace NUMINAMATH_CALUDE_right_trapezoid_diagonals_l1648_164818

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- Length of the smaller base -/
  small_base : ℝ
  /-- Length of the larger base -/
  large_base : ℝ
  /-- Angle at one vertex of the smaller base (in radians) -/
  angle_at_small_base : ℝ

/-- The diagonals of the trapezoid -/
def diagonals (t : RightTrapezoid) : ℝ × ℝ :=
  sorry

theorem right_trapezoid_diagonals :
  let t : RightTrapezoid := {
    small_base := 6,
    large_base := 8,
    angle_at_small_base := 2 * Real.pi / 3  -- 120° in radians
  }
  diagonals t = (4 * Real.sqrt 3, 2 * Real.sqrt 19) := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_diagonals_l1648_164818


namespace NUMINAMATH_CALUDE_sequence_determination_l1648_164809

theorem sequence_determination (p : ℕ) (hp : p.Prime ∧ p > 5) :
  let n := (p - 1) / 2
  ∀ (a : Fin n → ℕ), 
    (∀ i : Fin n, a i ∈ Finset.range n.succ) →
    Function.Injective a →
    (∀ i j : Fin n, i ≠ j → ∃ (r : ℕ), (a i * a j) % p = r) →
    ∃! (b : Fin n → ℕ), ∀ i : Fin n, a i = b i :=
by sorry

end NUMINAMATH_CALUDE_sequence_determination_l1648_164809


namespace NUMINAMATH_CALUDE_number_of_rats_l1648_164895

/-- Given a total of 70 animals where the number of rats is 6 times the number of chihuahuas,
    prove that the number of rats is 60. -/
theorem number_of_rats (total : ℕ) (chihuahuas : ℕ) (rats : ℕ) 
    (h1 : total = 70)
    (h2 : total = chihuahuas + rats)
    (h3 : rats = 6 * chihuahuas) : 
  rats = 60 := by
  sorry

end NUMINAMATH_CALUDE_number_of_rats_l1648_164895


namespace NUMINAMATH_CALUDE_exists_non_adjacent_divisible_l1648_164853

/-- A circular arrangement of seven natural numbers -/
def CircularArrangement := Fin 7 → ℕ+

/-- Predicate to check if one number divides another -/
def divides (a b : ℕ+) : Prop := ∃ k : ℕ+, b = a * k

/-- Two positions in the circular arrangement are adjacent -/
def adjacent (i j : Fin 7) : Prop := i = j + 1 ∨ j = i + 1 ∨ (i = 0 ∧ j = 6) ∨ (j = 0 ∧ i = 6)

/-- Two positions in the circular arrangement are non-adjacent -/
def non_adjacent (i j : Fin 7) : Prop := ¬(adjacent i j) ∧ i ≠ j

/-- The main theorem -/
theorem exists_non_adjacent_divisible (arr : CircularArrangement) 
  (h : ∀ i j : Fin 7, adjacent i j → (divides (arr i) (arr j) ∨ divides (arr j) (arr i))) :
  ∃ i j : Fin 7, non_adjacent i j ∧ (divides (arr i) (arr j) ∨ divides (arr j) (arr i)) := by
  sorry

end NUMINAMATH_CALUDE_exists_non_adjacent_divisible_l1648_164853


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_9_l1648_164883

def is_multiple_of_9 (n : ℕ) : Prop := ∃ k : ℕ, n = 9 * k

theorem five_digit_multiple_of_9 (d : ℕ) (h1 : d < 10) :
  is_multiple_of_9 (63470 + d) ↔ d = 7 := by sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_9_l1648_164883


namespace NUMINAMATH_CALUDE_browser_windows_l1648_164894

theorem browser_windows (num_browsers : Nat) (tabs_per_window : Nat) (total_tabs : Nat) :
  num_browsers = 2 →
  tabs_per_window = 10 →
  total_tabs = 60 →
  ∃ (windows_per_browser : Nat),
    windows_per_browser * tabs_per_window * num_browsers = total_tabs ∧
    windows_per_browser = 3 := by
  sorry

end NUMINAMATH_CALUDE_browser_windows_l1648_164894


namespace NUMINAMATH_CALUDE_tank_capacity_l1648_164803

/-- The capacity of a tank given specific filling and draining rates and a cyclic operation pattern. -/
theorem tank_capacity 
  (fill_rate_A : ℕ) 
  (fill_rate_B : ℕ) 
  (drain_rate_C : ℕ) 
  (total_time : ℕ) 
  (h1 : fill_rate_A = 40)
  (h2 : fill_rate_B = 30)
  (h3 : drain_rate_C = 20)
  (h4 : total_time = 57) :
  fill_rate_A + fill_rate_B - drain_rate_C = 50 →
  (total_time / 3) * (fill_rate_A + fill_rate_B - drain_rate_C) + fill_rate_A = 990 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1648_164803


namespace NUMINAMATH_CALUDE_candy_count_l1648_164844

/-- The number of candy pieces caught by Tabitha and her friends -/
def total_candy (tabitha stan julie carlos veronica benjamin kelly : ℕ) : ℕ :=
  tabitha + stan + julie + carlos + veronica + benjamin + kelly

/-- Theorem stating the total number of candy pieces caught by the friends -/
theorem candy_count : ∃ (tabitha stan julie carlos veronica benjamin kelly : ℕ),
  tabitha = 22 ∧
  stan = tabitha / 3 + 4 ∧
  julie = tabitha / 2 ∧
  carlos = 2 * stan ∧
  veronica = julie + stan - 5 ∧
  benjamin = (tabitha + carlos) / 2 + 9 ∧
  kelly = stan * julie / tabitha ∧
  total_candy tabitha stan julie carlos veronica benjamin kelly = 119 := by
  sorry

#check candy_count

end NUMINAMATH_CALUDE_candy_count_l1648_164844


namespace NUMINAMATH_CALUDE_s_range_l1648_164811

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem s_range : Set.range s = {y : ℝ | y < 0 ∨ y > 0} := by sorry

end NUMINAMATH_CALUDE_s_range_l1648_164811


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l1648_164848

/-- Calculates the total amount a buyer pays for a cycle, given the initial cost,
    loss percentage, and sales tax percentage. -/
def totalCyclePrice (initialCost : ℚ) (lossPercentage : ℚ) (salesTaxPercentage : ℚ) : ℚ :=
  let sellingPrice := initialCost * (1 - lossPercentage / 100)
  let salesTax := sellingPrice * (salesTaxPercentage / 100)
  sellingPrice + salesTax

/-- Theorem stating that for a cycle bought at Rs. 1400, sold at 20% loss,
    with 5% sales tax, the total price is Rs. 1176. -/
theorem cycle_price_calculation :
  totalCyclePrice 1400 20 5 = 1176 := by
  sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l1648_164848


namespace NUMINAMATH_CALUDE_homework_problems_left_l1648_164828

theorem homework_problems_left (math_problems science_problems finished_problems : ℕ) 
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : finished_problems = 40) :
  math_problems + science_problems - finished_problems = 15 :=
by sorry

end NUMINAMATH_CALUDE_homework_problems_left_l1648_164828


namespace NUMINAMATH_CALUDE_evaluate_expression_l1648_164841

theorem evaluate_expression : ((3^5 / 3^2) * 2^10) + (1/2) = 27648.5 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1648_164841


namespace NUMINAMATH_CALUDE_sin_sum_product_l1648_164827

theorem sin_sum_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (9 * x) = 2 * Real.sin (6 * x) * Real.cos (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_product_l1648_164827
