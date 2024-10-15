import Mathlib

namespace NUMINAMATH_CALUDE_adults_fed_is_eight_l37_3771

/-- Represents the number of adults that can be fed with one can of soup -/
def adults_per_can : ℕ := 4

/-- Represents the number of children that can be fed with one can of soup -/
def children_per_can : ℕ := 6

/-- Represents the total number of cans available -/
def total_cans : ℕ := 8

/-- Represents the number of children fed -/
def children_fed : ℕ := 24

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed : ℕ :=
  let cans_used_for_children := children_fed / children_per_can
  let remaining_cans := total_cans - cans_used_for_children
  let usable_cans := remaining_cans / 2
  usable_cans * adults_per_can

theorem adults_fed_is_eight : adults_fed = 8 := by
  sorry

end NUMINAMATH_CALUDE_adults_fed_is_eight_l37_3771


namespace NUMINAMATH_CALUDE_log_equation_solution_l37_3753

theorem log_equation_solution (t : ℝ) (h : t > 0) :
  4 * Real.log t / Real.log 3 = Real.log (4 * t) / Real.log 3 + 2 → t = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l37_3753


namespace NUMINAMATH_CALUDE_bernardo_wins_l37_3728

theorem bernardo_wins (N : ℕ) : N = 78 ↔ 
  N ∈ Finset.range 1000 ∧ 
  (∀ m : ℕ, m < N → m ∉ Finset.range 1000 ∨ 
    3 * m ≥ 1000 ∨ 
    3 * m + 75 ≥ 1000 ∨ 
    9 * m + 225 ≥ 1000 ∨ 
    9 * m + 300 < 1000) ∧
  3 * N < 1000 ∧
  3 * N + 75 < 1000 ∧
  9 * N + 225 < 1000 ∧
  9 * N + 300 ≥ 1000 := by
sorry

#eval (78 / 10) + (78 % 10)  -- Sum of digits of 78

end NUMINAMATH_CALUDE_bernardo_wins_l37_3728


namespace NUMINAMATH_CALUDE_sum_in_interval_l37_3776

theorem sum_in_interval : 
  let a : ℚ := 4 + 5/9
  let b : ℚ := 5 + 3/4
  let c : ℚ := 7 + 8/17
  17.5 < a + b + c ∧ a + b + c < 18 :=
by sorry

end NUMINAMATH_CALUDE_sum_in_interval_l37_3776


namespace NUMINAMATH_CALUDE_sum_of_first_fifteen_multiples_of_eight_l37_3744

/-- The sum of the first n natural numbers -/
def sum_of_naturals (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the first n positive multiples of m -/
def sum_of_multiples (m n : ℕ) : ℕ := m * sum_of_naturals n

theorem sum_of_first_fifteen_multiples_of_eight :
  sum_of_multiples 8 15 = 960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_fifteen_multiples_of_eight_l37_3744


namespace NUMINAMATH_CALUDE_calculate_expression_l37_3783

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - 3 * x

-- Define the # operation
def hash_op (x y : ℤ) : ℤ := x * y + y

-- Theorem statement
theorem calculate_expression : (at_op 8 5) - (at_op 5 8) + (hash_op 8 5) = 36 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l37_3783


namespace NUMINAMATH_CALUDE_championship_outcomes_l37_3741

theorem championship_outcomes (num_students : ℕ) (num_events : ℕ) : 
  num_students = 4 → num_events = 3 → (num_students ^ num_events : ℕ) = 64 := by
  sorry

#check championship_outcomes

end NUMINAMATH_CALUDE_championship_outcomes_l37_3741


namespace NUMINAMATH_CALUDE_journey_start_time_l37_3764

/-- Two people moving towards each other -/
structure Journey where
  start_time : ℝ
  meet_time : ℝ
  a_finish_time : ℝ
  b_finish_time : ℝ

/-- The journey satisfies the problem conditions -/
def satisfies_conditions (j : Journey) : Prop :=
  j.meet_time = 12 ∧ 
  j.a_finish_time = 16 ∧ 
  j.b_finish_time = 21 ∧ 
  0 < j.start_time ∧ j.start_time < j.meet_time

/-- The equation representing the journey -/
def journey_equation (j : Journey) : Prop :=
  1 / (j.meet_time - j.start_time) + 
  1 / (j.a_finish_time - j.meet_time) + 
  1 / (j.b_finish_time - j.meet_time) = 1

theorem journey_start_time (j : Journey) 
  (h1 : satisfies_conditions j) 
  (h2 : journey_equation j) : 
  j.start_time = 6 := by
  sorry


end NUMINAMATH_CALUDE_journey_start_time_l37_3764


namespace NUMINAMATH_CALUDE_point_parameters_l37_3752

/-- Parametric equation of a line -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The given line -/
def givenLine : ParametricLine :=
  { x := λ t => 1 + 2 * t,
    y := λ t => 2 - 3 * t }

/-- Point A -/
def pointA : Point :=
  { x := 1,
    y := 2 }

/-- Point B -/
def pointB : Point :=
  { x := -1,
    y := 5 }

/-- Theorem stating that the parameters for points A and B are 0 and -1 respectively -/
theorem point_parameters : 
  (∃ t : ℝ, givenLine.x t = pointA.x ∧ givenLine.y t = pointA.y ∧ t = 0) ∧
  (∃ t : ℝ, givenLine.x t = pointB.x ∧ givenLine.y t = pointB.y ∧ t = -1) :=
by sorry

end NUMINAMATH_CALUDE_point_parameters_l37_3752


namespace NUMINAMATH_CALUDE_computation_proof_l37_3729

theorem computation_proof : 45 * (28 + 72) + 55 * 45 = 6975 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l37_3729


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l37_3789

-- Define the sets M and N
def M : Set ℝ := {x | |x - 1| < 1}
def N : Set ℝ := {x | x * (x - 3) < 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l37_3789


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l37_3765

theorem complex_fraction_simplification :
  ∃ (i : ℂ), i * i = -1 → (5 * i) / (1 - 2 * i) = -2 + i :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l37_3765


namespace NUMINAMATH_CALUDE_first_candle_triple_second_at_correct_time_l37_3705

/-- The time (in hours) when the first candle is three times the height of the second candle -/
def time_when_first_is_triple_second : ℚ := 40 / 11

/-- The initial height of both candles -/
def initial_height : ℚ := 1

/-- The time (in hours) it takes for the first candle to burn out completely -/
def first_candle_burnout_time : ℚ := 5

/-- The time (in hours) it takes for the second candle to burn out completely -/
def second_candle_burnout_time : ℚ := 4

/-- The height of the first candle at time t -/
def first_candle_height (t : ℚ) : ℚ := initial_height - (t / first_candle_burnout_time)

/-- The height of the second candle at time t -/
def second_candle_height (t : ℚ) : ℚ := initial_height - (t / second_candle_burnout_time)

theorem first_candle_triple_second_at_correct_time :
  first_candle_height time_when_first_is_triple_second = 
  3 * second_candle_height time_when_first_is_triple_second :=
sorry

end NUMINAMATH_CALUDE_first_candle_triple_second_at_correct_time_l37_3705


namespace NUMINAMATH_CALUDE_train_speed_l37_3724

/-- Calculates the speed of a train passing over a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) :
  train_length = 320 →
  bridge_length = 140 →
  time = 36.8 →
  (((train_length + bridge_length) / time) * 3.6) = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l37_3724


namespace NUMINAMATH_CALUDE_arctan_equation_solutions_l37_3788

theorem arctan_equation_solutions (x : ℝ) : 
  (Real.arctan (2 / x) + Real.arctan (1 / x^2) = π / 4) ↔ 
  (x = 3 ∨ x = (-3 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_CALUDE_arctan_equation_solutions_l37_3788


namespace NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l37_3701

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem smallest_integer_with_divisibility_condition : 
  ∀ n : ℕ, n > 0 →
  (∀ i ∈ Finset.range 31, i ≠ 23 ∧ i ≠ 24 → is_divisible n i) →
  ¬(is_divisible n 23) →
  ¬(is_divisible n 24) →
  n ≥ 2230928700 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_with_divisibility_condition_l37_3701


namespace NUMINAMATH_CALUDE_cube_volume_from_paper_l37_3719

theorem cube_volume_from_paper (paper_length paper_width : ℝ) 
  (h1 : paper_length = 48)
  (h2 : paper_width = 72)
  (h3 : 1 = 12) : -- 1 foot = 12 inches
  let paper_area := paper_length * paper_width
  let cube_face_area := paper_area / 6
  let cube_side_length := Real.sqrt cube_face_area
  let cube_side_length_feet := cube_side_length / 12
  cube_side_length_feet ^ 3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_from_paper_l37_3719


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l37_3790

theorem quadratic_function_bounds (a c : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^2 - c
  (-4 : ℝ) ≤ f 1 ∧ f 1 ≤ -1 ∧ (-1 : ℝ) ≤ f 2 ∧ f 2 ≤ 5 →
  (-1 : ℝ) ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l37_3790


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l37_3710

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if 2x - √3y = 0 is one of its asymptotes, then its eccentricity is √21/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y, x^2/a^2 - y^2/b^2 = 1 → 2*x - Real.sqrt 3*y = 0) →
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 21 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l37_3710


namespace NUMINAMATH_CALUDE_percentage_difference_in_gain_l37_3794

def cost_price : ℝ := 400
def selling_price1 : ℝ := 360
def selling_price2 : ℝ := 340

def gain1 : ℝ := selling_price1 - cost_price
def gain2 : ℝ := selling_price2 - cost_price

def difference_in_gain : ℝ := gain1 - gain2

theorem percentage_difference_in_gain :
  (difference_in_gain / cost_price) * 100 = 5 := by sorry

end NUMINAMATH_CALUDE_percentage_difference_in_gain_l37_3794


namespace NUMINAMATH_CALUDE_sqrt_one_third_equality_l37_3759

theorem sqrt_one_third_equality : 3 * Real.sqrt (1/3) = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_one_third_equality_l37_3759


namespace NUMINAMATH_CALUDE_remaining_income_percentage_l37_3704

-- Define the percentages as fractions
def food_percent : ℚ := 35 / 100
def education_percent : ℚ := 25 / 100
def transportation_percent : ℚ := 15 / 100
def medical_percent : ℚ := 10 / 100
def rent_percent_of_remaining : ℚ := 80 / 100

-- Theorem statement
theorem remaining_income_percentage :
  let initial_expenses := food_percent + education_percent + transportation_percent + medical_percent
  let remaining_after_initial := 1 - initial_expenses
  let rent_expense := rent_percent_of_remaining * remaining_after_initial
  1 - (initial_expenses + rent_expense) = 3 / 100 := by
  sorry

end NUMINAMATH_CALUDE_remaining_income_percentage_l37_3704


namespace NUMINAMATH_CALUDE_sum_remainder_zero_l37_3706

theorem sum_remainder_zero (n : ℤ) : (10 - 2*n + 4*n + 2) % 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_zero_l37_3706


namespace NUMINAMATH_CALUDE_parallel_angles_theorem_l37_3736

/-- Two angles with parallel sides --/
structure ParallelAngles where
  α : ℝ
  β : ℝ
  x : ℝ
  parallel : Bool
  α_eq : α = 2 * x + 10
  β_eq : β = 3 * x - 20

/-- The possible values for α in the parallel angles scenario --/
def possible_α_values (angles : ParallelAngles) : Set ℝ :=
  {70, 86}

/-- Theorem stating that the possible values for α are 70° or 86° --/
theorem parallel_angles_theorem (angles : ParallelAngles) :
  angles.α ∈ possible_α_values angles :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_angles_theorem_l37_3736


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equals_B_l37_3730

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

-- Theorem 1: Intersection of A and B when m = 1
theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

-- Theorem 2: Condition for A ∪ B = B
theorem union_equals_B (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equals_B_l37_3730


namespace NUMINAMATH_CALUDE_equal_sandwiched_segments_imply_parallel_or_intersecting_planes_l37_3780

-- Define a plane
structure Plane where
  -- Add necessary fields for a plane

-- Define a line segment
structure LineSegment where
  -- Add necessary fields for a line segment

-- Define the property of being sandwiched between two planes
def sandwichedBetween (l : LineSegment) (p1 p2 : Plane) : Prop :=
  sorry

-- Define the property of line segments being parallel
def areParallel (l1 l2 l3 : LineSegment) : Prop :=
  sorry

-- Define the property of line segments being equal
def areEqual (l1 l2 l3 : LineSegment) : Prop :=
  sorry

-- Define the property of planes being parallel
def arePlanesParallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define the property of planes intersecting
def arePlanesIntersecting (p1 p2 : Plane) : Prop :=
  sorry

-- The main theorem
theorem equal_sandwiched_segments_imply_parallel_or_intersecting_planes 
  (p1 p2 : Plane) (l1 l2 l3 : LineSegment) :
  sandwichedBetween l1 p1 p2 →
  sandwichedBetween l2 p1 p2 →
  sandwichedBetween l3 p1 p2 →
  areParallel l1 l2 l3 →
  areEqual l1 l2 l3 →
  arePlanesParallel p1 p2 ∨ arePlanesIntersecting p1 p2 :=
sorry

end NUMINAMATH_CALUDE_equal_sandwiched_segments_imply_parallel_or_intersecting_planes_l37_3780


namespace NUMINAMATH_CALUDE_last_two_digits_product_l37_3784

theorem last_two_digits_product (n : ℤ) : ∃ k : ℤ, 122 * 123 * 125 * 127 * n ≡ 50 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l37_3784


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l37_3740

/-- A quadratic polynomial function -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The derivative of a quadratic polynomial function -/
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

/-- Theorem: If the graph of a quadratic polynomial and its derivative
    divide the coordinate plane into four parts, then the polynomial has no real roots -/
theorem quadratic_no_roots (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ 
    quadratic a b c x₁ = quadratic_derivative a b x₁ ∧
    quadratic a b c x₂ = quadratic_derivative a b x₂) →
  (∀ x : ℝ, quadratic a b c x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l37_3740


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l37_3777

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - 2 * Complex.I) = Complex.I * b) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l37_3777


namespace NUMINAMATH_CALUDE_lg_equation_l37_3782

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem lg_equation : lg 2 * lg 50 + lg 25 - lg 5 * lg 20 = 1 := by sorry

end NUMINAMATH_CALUDE_lg_equation_l37_3782


namespace NUMINAMATH_CALUDE_divisibility_condition_l37_3720

def M (n : ℤ) : Finset ℤ := {n, n + 1, n + 2, n + 3, n + 4}

def S (n : ℤ) : ℤ := (M n).sum (fun x => x^2)

def P (n : ℤ) : ℤ := (M n).prod (fun x => x^2)

theorem divisibility_condition (n : ℤ) : S n ∣ P n ↔ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l37_3720


namespace NUMINAMATH_CALUDE_circle_tangent_and_secant_l37_3757

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y - 3 = 0

-- Define point M
def point_M : ℝ × ℝ := (4, -8)

-- Define the length of AB
def length_AB : ℝ := 4

theorem circle_tangent_and_secant :
  ∃ (tangent_length : ℝ) (line_DE : ℝ → ℝ → Prop) (line_AB : ℝ → ℝ → Prop),
    -- The length of the tangent from M to C is 3√5
    tangent_length = 3 * Real.sqrt 5 ∧
    -- The equation of line DE is 2x-7y-19=0
    (∀ x y, line_DE x y ↔ 2*x - 7*y - 19 = 0) ∧
    -- The equation of line AB is either 45x+28y+44=0 or x=4
    (∀ x y, line_AB x y ↔ (45*x + 28*y + 44 = 0 ∨ x = 4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_secant_l37_3757


namespace NUMINAMATH_CALUDE_sine_zeros_range_l37_3742

open Real

theorem sine_zeros_range (ω : ℝ) : 
  (ω > 0) → 
  (∃! (z₁ z₂ : ℝ), 0 ≤ z₁ ∧ z₁ < z₂ ∧ z₂ ≤ π/4 ∧ 
    sin (2*ω*z₁ - π/6) = 0 ∧ sin (2*ω*z₂ - π/6) = 0 ∧
    ∀ z, 0 ≤ z ∧ z ≤ π/4 ∧ sin (2*ω*z - π/6) = 0 → z = z₁ ∨ z = z₂) ↔ 
  (7/3 ≤ ω ∧ ω < 13/3) :=
by sorry

end NUMINAMATH_CALUDE_sine_zeros_range_l37_3742


namespace NUMINAMATH_CALUDE_solution_set_of_equations_l37_3760

theorem solution_set_of_equations (a b c d : ℝ) : 
  (a * b * c + d = 2 ∧
   b * c * d + a = 2 ∧
   c * d * a + b = 2 ∧
   d * a * b + c = 2) ↔ 
  ((a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
   (a = 3 ∧ b = -1 ∧ c = -1 ∧ d = -1) ∨
   (a = -1 ∧ b = 3 ∧ c = -1 ∧ d = -1)) :=
by sorry

#check solution_set_of_equations

end NUMINAMATH_CALUDE_solution_set_of_equations_l37_3760


namespace NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l37_3772

theorem swan_percentage_among_non_ducks (total : ℝ) (ducks swans herons geese : ℝ) :
  total = 100 →
  ducks = 35 →
  swans = 30 →
  herons = 20 →
  geese = 15 →
  (swans / (total - ducks)) * 100 = 46.15 := by
  sorry

end NUMINAMATH_CALUDE_swan_percentage_among_non_ducks_l37_3772


namespace NUMINAMATH_CALUDE_average_of_combined_results_l37_3703

theorem average_of_combined_results (n₁ n₂ : ℕ) (avg₁ avg₂ : ℚ) 
  (h₁ : n₁ = 55) (h₂ : n₂ = 28) (h₃ : avg₁ = 28) (h₄ : avg₂ = 55) :
  (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂ : ℚ) = (n₁ * avg₁ + n₂ * avg₂) / (n₁ + n₂) :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_results_l37_3703


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squared_l37_3773

theorem arithmetic_sequence_squared (x y z : ℝ) (h : y - x = z - y) :
  (x^2 + x*z + z^2) - (x^2 + x*y + y^2) = (y^2 + y*z + z^2) - (x^2 + x*z + z^2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squared_l37_3773


namespace NUMINAMATH_CALUDE_solution_of_inequality1_solution_of_inequality2_l37_3748

-- Define the solution set for the first inequality
def solutionSet1 : Set ℝ := {x | x > -1 ∧ x < 1}

-- Define the solution set for the second inequality
def solutionSet2 (a : ℝ) : Set ℝ :=
  if a = -2 then Set.univ
  else if a > -2 then {x | x ≤ -2 ∨ x ≥ a}
  else {x | x ≤ a ∨ x ≥ -2}

-- Theorem for the first inequality
theorem solution_of_inequality1 :
  {x : ℝ | (2 * x) / (x + 1) < 1} = solutionSet1 := by sorry

-- Theorem for the second inequality
theorem solution_of_inequality2 (a : ℝ) :
  {x : ℝ | x^2 + (2 - a) * x - 2 * a ≥ 0} = solutionSet2 a := by sorry

end NUMINAMATH_CALUDE_solution_of_inequality1_solution_of_inequality2_l37_3748


namespace NUMINAMATH_CALUDE_honor_roll_fraction_l37_3743

theorem honor_roll_fraction (total_students : ℝ) (female_students : ℝ) (male_students : ℝ) 
  (female_honor : ℝ) (male_honor : ℝ) :
  female_students = (2 / 5) * total_students →
  male_students = (3 / 5) * total_students →
  female_honor = (5 / 6) * female_students →
  male_honor = (2 / 3) * male_students →
  (female_honor + male_honor) / total_students = 11 / 15 := by
sorry

end NUMINAMATH_CALUDE_honor_roll_fraction_l37_3743


namespace NUMINAMATH_CALUDE_soccer_penalty_kicks_l37_3739

theorem soccer_penalty_kicks (total_players : ℕ) (goalies : ℕ) (shots : ℕ) :
  total_players = 22 →
  goalies = 4 →
  shots = goalies * (total_players - 1) →
  shots = 84 :=
by sorry

end NUMINAMATH_CALUDE_soccer_penalty_kicks_l37_3739


namespace NUMINAMATH_CALUDE_ice_cube_volume_l37_3766

theorem ice_cube_volume (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.25) →
  (original_volume = 4) := by
sorry

end NUMINAMATH_CALUDE_ice_cube_volume_l37_3766


namespace NUMINAMATH_CALUDE_rectangular_solid_properties_l37_3731

theorem rectangular_solid_properties (a b c : ℝ) 
  (h1 : a * b = Real.sqrt 6)
  (h2 : a * c = Real.sqrt 3)
  (h3 : b * c = Real.sqrt 2) :
  (a * b * c = 6) ∧ 
  (Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6) := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_properties_l37_3731


namespace NUMINAMATH_CALUDE_only_option_C_is_well_defined_set_l37_3793

-- Define the universe of discourse
def Universe : Type := String

-- Define the property of being a well-defined set
def is_well_defined_set (S : Set Universe) : Prop :=
  ∀ x, ∃ (decision : Prop), (x ∈ S ↔ decision)

-- Define the four options
def option_A : Set Universe := {x | x = "Tall students in the first grade of Fengdu Middle School in January 2013"}
def option_B : Set Universe := {x | x = "Tall trees in the campus"}
def option_C : Set Universe := {x | x = "Students in the first grade of Fengdu Middle School in January 2013"}
def option_D : Set Universe := {x | x = "Students with high basketball skills in the school"}

-- Theorem statement
theorem only_option_C_is_well_defined_set :
  is_well_defined_set option_C ∧
  ¬is_well_defined_set option_A ∧
  ¬is_well_defined_set option_B ∧
  ¬is_well_defined_set option_D :=
sorry

end NUMINAMATH_CALUDE_only_option_C_is_well_defined_set_l37_3793


namespace NUMINAMATH_CALUDE_cubic_function_properties_l37_3769

/-- A cubic function with a maximum at x = -1 and a minimum at x = 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∀ a b c : ℝ,
  (∀ x : ℝ, f a b c x ≤ f a b c (-1)) ∧
  (f a b c (-1) = 7) ∧
  (f' a b (-1) = 0) ∧
  (f' a b 3 = 0) →
  a = -3 ∧ b = -9 ∧ c = 2 ∧ f a b c 3 = -25 := by
  sorry

#check cubic_function_properties

end NUMINAMATH_CALUDE_cubic_function_properties_l37_3769


namespace NUMINAMATH_CALUDE_max_cubes_is_six_l37_3756

/-- Represents a stack of identical wooden cubes -/
structure CubeStack where
  front_view : Nat
  side_view : Nat
  top_view : Nat

/-- The maximum number of cubes in a stack given its views -/
def max_cubes (stack : CubeStack) : Nat :=
  2 * stack.top_view

/-- Theorem stating that the maximum number of cubes in the stack is 6 -/
theorem max_cubes_is_six (stack : CubeStack) 
  (h_top : stack.top_view = 3) : max_cubes stack = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_cubes_is_six_l37_3756


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l37_3774

theorem sum_of_x_and_y_on_circle (x y : ℝ) (h : x^2 + y^2 = 10*x - 6*y - 34) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_on_circle_l37_3774


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_twelve_l37_3785

theorem five_digit_divisible_by_twelve : ∃! (n : Nat), n < 10 ∧ 51470 + n ≡ 0 [MOD 12] := by
  sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_twelve_l37_3785


namespace NUMINAMATH_CALUDE_max_intersection_difference_l37_3749

/-- The first function in the problem -/
def f (x : ℝ) : ℝ := 4 - x^2 + x^3

/-- The second function in the problem -/
def g (x : ℝ) : ℝ := 2 + 2*x^2 + x^3

/-- The difference between the y-coordinates of the intersection points -/
def intersection_difference (x : ℝ) : ℝ := |f x - g x|

/-- The theorem stating the maximum difference between y-coordinates of intersection points -/
theorem max_intersection_difference : 
  ∃ (x : ℝ), f x = g x ∧ 
  ∀ (y : ℝ), f y = g y → intersection_difference x ≥ intersection_difference y ∧
  intersection_difference x = 2 * (2/3)^(3/2) :=
sorry

end NUMINAMATH_CALUDE_max_intersection_difference_l37_3749


namespace NUMINAMATH_CALUDE_remainder_divisibility_l37_3709

theorem remainder_divisibility (x y z p : ℕ) : 
  0 < x → 0 < y → 0 < z →  -- x, y, z are positive integers
  Nat.Prime p →            -- p is prime
  x < y → y < z → z < p →  -- x < y < z < p
  x^3 % p = y^3 % p →      -- x^3 and y^3 have the same remainder mod p
  y^3 % p = z^3 % p →      -- y^3 and z^3 have the same remainder mod p
  (x + y + z) ∣ (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l37_3709


namespace NUMINAMATH_CALUDE_complex_magnitude_l37_3754

theorem complex_magnitude (a b : ℝ) :
  (Complex.I + a) * (1 - b * Complex.I) = 2 * Complex.I →
  Complex.abs (a + b * Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l37_3754


namespace NUMINAMATH_CALUDE_f_inequality_range_l37_3738

def f (x : ℝ) := -x^3 + 3*x + 2

theorem f_inequality_range (m : ℝ) :
  (∀ θ : ℝ, f (3 + 2 * Real.sin θ) < m) → m > 4 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_range_l37_3738


namespace NUMINAMATH_CALUDE_rent_expense_calculation_l37_3779

def monthly_salary : ℕ := 23500
def savings_percentage : ℚ := 1/10
def savings : ℕ := 2350
def milk_expense : ℕ := 1500
def groceries_expense : ℕ := 4500
def education_expense : ℕ := 2500
def petrol_expense : ℕ := 2000
def miscellaneous_expense : ℕ := 5650

theorem rent_expense_calculation :
  let total_expenses := milk_expense + groceries_expense + education_expense + petrol_expense + miscellaneous_expense
  let rent := monthly_salary - savings - total_expenses
  rent = 4850 := by sorry

end NUMINAMATH_CALUDE_rent_expense_calculation_l37_3779


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l37_3796

theorem quadratic_equation_properties (m : ℝ) :
  -- Part 1: The equation always has real roots
  ∃ x₁ x₂ : ℝ, x₁^2 - (m+3)*x₁ + 2*(m+1) = 0 ∧ x₂^2 - (m+3)*x₂ + 2*(m+1) = 0 ∧
  -- Part 2: If x₁² + x₂² = 5, then m = 0 or m = -2
  (∀ x₁ x₂ : ℝ, x₁^2 - (m+3)*x₁ + 2*(m+1) = 0 → x₂^2 - (m+3)*x₂ + 2*(m+1) = 0 → x₁^2 + x₂^2 = 5 → m = 0 ∨ m = -2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l37_3796


namespace NUMINAMATH_CALUDE_inequality_proof_l37_3718

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l37_3718


namespace NUMINAMATH_CALUDE_total_pools_l37_3723

def arkAndAthleticPools : ℕ := 200
def poolSupplyMultiplier : ℕ := 3

theorem total_pools : arkAndAthleticPools + poolSupplyMultiplier * arkAndAthleticPools = 800 := by
  sorry

end NUMINAMATH_CALUDE_total_pools_l37_3723


namespace NUMINAMATH_CALUDE_orangeade_pricing_l37_3745

/-- Represents the amount of orange juice used each day -/
def orange_juice : ℝ := sorry

/-- Represents the amount of water used on the first day -/
def water : ℝ := sorry

/-- The price per glass on the first day -/
def price_day1 : ℝ := 0.60

/-- The price per glass on the third day -/
def price_day3 : ℝ := sorry

/-- The volume of orangeade on the first day -/
def volume_day1 : ℝ := orange_juice + water

/-- The volume of orangeade on the second day -/
def volume_day2 : ℝ := orange_juice + 2 * water

/-- The volume of orangeade on the third day -/
def volume_day3 : ℝ := orange_juice + 3 * water

theorem orangeade_pricing :
  (orange_juice > 0) →
  (water > 0) →
  (orange_juice = water) →
  (price_day1 * volume_day1 = price_day3 * volume_day3) →
  (price_day3 = price_day1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_orangeade_pricing_l37_3745


namespace NUMINAMATH_CALUDE_swim_time_proof_l37_3755

/-- Proves that the time taken to swim downstream and upstream is 6 hours each -/
theorem swim_time_proof (downstream_distance : ℝ) (upstream_distance : ℝ) 
  (still_water_speed : ℝ) (h1 : downstream_distance = 30) 
  (h2 : upstream_distance = 18) (h3 : still_water_speed = 4) :
  ∃ (t : ℝ) (current_speed : ℝ), 
    t = downstream_distance / (still_water_speed + current_speed) ∧
    t = upstream_distance / (still_water_speed - current_speed) ∧
    t = 6 := by
  sorry

#check swim_time_proof

end NUMINAMATH_CALUDE_swim_time_proof_l37_3755


namespace NUMINAMATH_CALUDE_password_letters_count_l37_3727

theorem password_letters_count : ∃ (n : ℕ), 
  (n ^ 4 : ℕ) - n * (n - 1) * (n - 2) * (n - 3) = 936 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_password_letters_count_l37_3727


namespace NUMINAMATH_CALUDE_fruit_cost_calculation_l37_3778

/-- The cost of a single orange in dollars -/
def orange_cost : ℚ := (1.27 - 2 * 0.21) / 5

/-- The total cost of six apples and three oranges in dollars -/
def total_cost : ℚ := 6 * 0.21 + 3 * orange_cost

theorem fruit_cost_calculation :
  (2 * 0.21 + 5 * orange_cost = 1.27) →
  total_cost = 1.77 := by
  sorry

end NUMINAMATH_CALUDE_fruit_cost_calculation_l37_3778


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coeff_difference_l37_3733

theorem quadratic_roots_to_coeff_difference (a b : ℝ) : 
  (∀ x, a * x^2 + b * x + 2 = 0 ↔ (x = -1/2 ∨ x = 1/3)) → 
  a - b = -10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coeff_difference_l37_3733


namespace NUMINAMATH_CALUDE_only_B_suitable_l37_3799

-- Define the structure for a sampling experiment
structure SamplingExperiment where
  totalSize : ℕ
  sampleSize : ℕ
  isWellMixed : Bool

-- Define the conditions for lottery method suitability
def isLotteryMethodSuitable (experiment : SamplingExperiment) : Prop :=
  experiment.totalSize ≤ 100 ∧ 
  experiment.sampleSize ≤ 10 ∧ 
  experiment.isWellMixed

-- Define the given sampling experiments
def experimentA : SamplingExperiment := ⟨5000, 600, true⟩
def experimentB : SamplingExperiment := ⟨36, 6, true⟩
def experimentC : SamplingExperiment := ⟨36, 6, false⟩
def experimentD : SamplingExperiment := ⟨5000, 10, true⟩

-- Theorem statement
theorem only_B_suitable : 
  ¬(isLotteryMethodSuitable experimentA) ∧
  (isLotteryMethodSuitable experimentB) ∧
  ¬(isLotteryMethodSuitable experimentC) ∧
  ¬(isLotteryMethodSuitable experimentD) :=
by sorry

end NUMINAMATH_CALUDE_only_B_suitable_l37_3799


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l37_3712

theorem smallest_k_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l37_3712


namespace NUMINAMATH_CALUDE_light_bulbs_configuration_equals_59_l37_3711

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldl (fun acc (i, b) => acc + (if b then 2^i else 0)) 0

/-- The binary representation of the light bulb configuration -/
def light_bulbs : List Bool := [true, true, true, false, true, true]

theorem light_bulbs_configuration_equals_59 :
  binary_to_decimal light_bulbs = 59 := by
  sorry

end NUMINAMATH_CALUDE_light_bulbs_configuration_equals_59_l37_3711


namespace NUMINAMATH_CALUDE_external_tangent_circle_l37_3781

/-- Given circle C with equation (x-2)^2 + (y+1)^2 = 4 and point A(4, -1) on C,
    prove that the circle with equation (x-5)^2 + (y+1)^2 = 1 is externally
    tangent to C at A and has radius 1. -/
theorem external_tangent_circle
  (C : Set (ℝ × ℝ))
  (A : ℝ × ℝ)
  (hC : C = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 1)^2 = 4})
  (hA : A = (4, -1))
  (hA_on_C : A ∈ C)
  : ∃ (M : Set (ℝ × ℝ)),
    M = {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 + 1)^2 = 1} ∧
    (∀ p ∈ M, ∃ q ∈ C, (p.1 - q.1)^2 + (p.2 - q.2)^2 = 1) ∧
    A ∈ M ∧
    (∀ p ∈ M, (p.1 - 5)^2 + (p.2 + 1)^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_external_tangent_circle_l37_3781


namespace NUMINAMATH_CALUDE_plan2_better_l37_3717

/-- The number of optional questions -/
def total_questions : ℕ := 5

/-- The number of questions Student A can solve -/
def solvable_questions : ℕ := 3

/-- Probability of participating under Plan 1 -/
def prob_plan1 : ℚ := solvable_questions / total_questions

/-- Probability of participating under Plan 2 -/
def prob_plan2 : ℚ := (Nat.choose solvable_questions 2 * Nat.choose (total_questions - solvable_questions) 1 + 
                       Nat.choose solvable_questions 3) / 
                      Nat.choose total_questions 3

/-- Theorem stating that Plan 2 gives a higher probability for Student A -/
theorem plan2_better : prob_plan2 > prob_plan1 := by
  sorry

end NUMINAMATH_CALUDE_plan2_better_l37_3717


namespace NUMINAMATH_CALUDE_fifteen_segments_two_monochromatic_triangles_fourteen_segments_no_monochromatic_triangle_possible_l37_3762

/-- Represents a segment (edge or diagonal) in a regular hexagon --/
inductive Segment
| Edge : Fin 6 → Fin 6 → Segment
| Diagonal : Fin 6 → Fin 6 → Segment

/-- Represents the color of a segment --/
inductive Color
| Red
| Blue

/-- Represents a coloring of segments in a regular hexagon --/
def Coloring := Segment → Option Color

/-- Checks if a triangle is monochromatic --/
def isMonochromatic (c : Coloring) (v1 v2 v3 : Fin 6) : Bool :=
  sorry

/-- Counts the number of colored segments in a coloring --/
def countColoredSegments (c : Coloring) : Nat :=
  sorry

/-- Counts the number of monochromatic triangles in a coloring --/
def countMonochromaticTriangles (c : Coloring) : Nat :=
  sorry

/-- Theorem: If 15 segments are colored, there are at least two monochromatic triangles --/
theorem fifteen_segments_two_monochromatic_triangles (c : Coloring) :
  countColoredSegments c = 15 → countMonochromaticTriangles c ≥ 2 :=
  sorry

/-- Theorem: It's possible to color 14 segments without forming a monochromatic triangle --/
theorem fourteen_segments_no_monochromatic_triangle_possible :
  ∃ c : Coloring, countColoredSegments c = 14 ∧ countMonochromaticTriangles c = 0 :=
  sorry

end NUMINAMATH_CALUDE_fifteen_segments_two_monochromatic_triangles_fourteen_segments_no_monochromatic_triangle_possible_l37_3762


namespace NUMINAMATH_CALUDE_leadership_diagram_is_organizational_structure_l37_3725

/-- Represents types of diagrams --/
inductive Diagram
  | ProgramFlowchart
  | ProcessFlowchart
  | KnowledgeStructureDiagram
  | OrganizationalStructureDiagram

/-- Represents a leadership relationship diagram --/
structure LeadershipDiagram where
  represents_leadership : Bool
  represents_structure : Bool

/-- Definition of an organizational structure diagram --/
def is_organizational_structure_diagram (d : LeadershipDiagram) : Prop :=
  d.represents_leadership ∧ d.represents_structure

/-- Theorem stating that a leadership relationship diagram in a governance group 
    is an organizational structure diagram --/
theorem leadership_diagram_is_organizational_structure :
  ∀ (d : LeadershipDiagram),
  d.represents_leadership ∧ d.represents_structure →
  is_organizational_structure_diagram d :=
by
  sorry

#check leadership_diagram_is_organizational_structure

end NUMINAMATH_CALUDE_leadership_diagram_is_organizational_structure_l37_3725


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l37_3732

theorem triangle_angle_sum (a b c : ℝ) : 
  b = 2 * a →
  c = a - 40 →
  a + b + c = 180 →
  a + c = 70 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l37_3732


namespace NUMINAMATH_CALUDE_age_problem_l37_3700

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →  -- a is two years older than b
  b = 2 * c →  -- b is twice as old as c
  a + b + c = 17 →  -- The total of the ages of a, b, and c is 17
  b = 6 :=  -- Prove that b is 6 years old
by
  sorry

end NUMINAMATH_CALUDE_age_problem_l37_3700


namespace NUMINAMATH_CALUDE_apple_slices_l37_3707

theorem apple_slices (S : ℕ) : 
  S > 0 ∧ 
  (S / 16 : ℚ) * S = 5 → 
  S = 16 := by
sorry

end NUMINAMATH_CALUDE_apple_slices_l37_3707


namespace NUMINAMATH_CALUDE_sequence_problem_l37_3737

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a b : ℕ → ℝ) 
  (ha : arithmetic_sequence a) 
  (hb : geometric_sequence b)
  (h_non_zero : ∀ n, a n ≠ 0)
  (h_eq : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)
  (h_b7 : b 7 = a 7) :
  b 5 * b 9 = 16 := by sorry

end NUMINAMATH_CALUDE_sequence_problem_l37_3737


namespace NUMINAMATH_CALUDE_cube_edge_length_l37_3713

/-- Given three cubes with edge lengths 6, 10, and x, when melted together to form a new cube
    with edge length 12, prove that x = 8 -/
theorem cube_edge_length (x : ℝ) : x > 0 → 6^3 + 10^3 + x^3 = 12^3 → x = 8 := by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l37_3713


namespace NUMINAMATH_CALUDE_smallest_n_for_183_div_11_l37_3735

theorem smallest_n_for_183_div_11 :
  ∃! n : ℕ, (183 + n) % 11 = 0 ∧ ∀ m : ℕ, m < n → (183 + m) % 11 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_183_div_11_l37_3735


namespace NUMINAMATH_CALUDE_sin_cos_15_ratio_l37_3734

theorem sin_cos_15_ratio : 
  (Real.sin (15 * π / 180) + Real.cos (15 * π / 180)) / 
  (Real.sin (15 * π / 180) - Real.cos (15 * π / 180)) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_15_ratio_l37_3734


namespace NUMINAMATH_CALUDE_total_bales_in_barn_l37_3798

def initial_bales : ℕ := 54
def added_bales : ℕ := 28

theorem total_bales_in_barn : initial_bales + added_bales = 82 := by
  sorry

end NUMINAMATH_CALUDE_total_bales_in_barn_l37_3798


namespace NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_integers_l37_3702

theorem largest_divisor_of_three_consecutive_integers :
  ∃ (d : ℕ), d > 0 ∧
  (∀ (n : ℤ), (n * (n + 1) * (n + 2)) % d = 0) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℤ), (m * (m + 1) * (m + 2)) % k ≠ 0) ∧
  d = 6 := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_three_consecutive_integers_l37_3702


namespace NUMINAMATH_CALUDE_root_product_expression_l37_3721

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + 3 = 0) →
  (β^2 - 2*p*β + 3 = 0) →
  (γ^2 - 3*q*γ + 4 = 0) →
  (δ^2 - 3*q*δ + 4 = 0) →
  (α - γ) * (β - δ) * (α + δ) * (β + γ) = 4 * (2*p - 3*q)^2 := by sorry

end NUMINAMATH_CALUDE_root_product_expression_l37_3721


namespace NUMINAMATH_CALUDE_problem_statement_l37_3758

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) :
  b < 0 ∧ |b| > |a| := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l37_3758


namespace NUMINAMATH_CALUDE_pentagon_perimeter_l37_3726

/-- The perimeter of pentagon ABCDE with given side lengths -/
theorem pentagon_perimeter (AB BC CD DE AE : ℝ) : 
  AB = 1 → BC = Real.sqrt 3 → CD = 2 → DE = Real.sqrt 5 → AE = Real.sqrt 13 →
  AB + BC + CD + DE + AE = 3 + Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_l37_3726


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l37_3770

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)  -- The sequence
  (S : ℕ → ℝ)  -- The sum sequence
  (h1 : ∀ n, S n = (n : ℝ) / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- Definition of sum
  (h2 : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- Definition of arithmetic sequence

/-- The main theorem -/
theorem arithmetic_sequence_fifth_term 
  (seq : ArithmeticSequence) 
  (eq1 : seq.a 3 + seq.S 3 = 22) 
  (eq2 : seq.a 4 - seq.S 4 = -15) : 
  seq.a 5 = 11 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l37_3770


namespace NUMINAMATH_CALUDE_walking_time_difference_l37_3797

/-- The speed of person A in km/h -/
def speed_A : ℝ := 4

/-- The speed of person B in km/h -/
def speed_B : ℝ := 4.555555555555555

/-- The time in hours after which B overtakes A -/
def overtake_time : ℝ := 1.8

/-- The time in hours after A started that B starts walking -/
def time_diff : ℝ := 0.25

theorem walking_time_difference :
  speed_A * (time_diff + overtake_time) = speed_B * overtake_time := by
  sorry

end NUMINAMATH_CALUDE_walking_time_difference_l37_3797


namespace NUMINAMATH_CALUDE_grade_ratio_is_two_to_one_l37_3786

/-- The ratio of students in the third grade to the second grade -/
def grade_ratio (boys_2nd : ℕ) (girls_2nd : ℕ) (total_students : ℕ) : ℚ :=
  let students_2nd := boys_2nd + girls_2nd
  let students_3rd := total_students - students_2nd
  students_3rd / students_2nd

/-- Theorem stating the ratio of students in the third grade to the second grade -/
theorem grade_ratio_is_two_to_one :
  grade_ratio 20 11 93 = 2 := by
  sorry

#eval grade_ratio 20 11 93

end NUMINAMATH_CALUDE_grade_ratio_is_two_to_one_l37_3786


namespace NUMINAMATH_CALUDE_complement_union_theorem_l37_3787

/-- The universal set I -/
def I : Set ℕ := {0, 1, 2, 3, 4}

/-- Set A -/
def A : Set ℕ := {0, 1, 2, 3}

/-- Set B -/
def B : Set ℕ := {2, 3, 4}

/-- The main theorem -/
theorem complement_union_theorem :
  (I \ A) ∪ (I \ B) = {0, 1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l37_3787


namespace NUMINAMATH_CALUDE_trivia_team_distribution_l37_3715

theorem trivia_team_distribution (total_students : ℕ) (not_picked : ℕ) (num_groups : ℕ) :
  total_students = 65 →
  not_picked = 17 →
  num_groups = 8 →
  (total_students - not_picked) / num_groups = 6 := by
  sorry

end NUMINAMATH_CALUDE_trivia_team_distribution_l37_3715


namespace NUMINAMATH_CALUDE_power_of_two_mod_nine_periodic_l37_3722

/-- The sequence of remainders when powers of 2 are divided by 9 is periodic with period 6 -/
theorem power_of_two_mod_nine_periodic :
  ∃ (p : ℕ), p > 0 ∧ ∀ (n : ℕ), (2^(n + p) : ℕ) % 9 = (2^n : ℕ) % 9 ∧ p = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_mod_nine_periodic_l37_3722


namespace NUMINAMATH_CALUDE_reflection_line_sum_l37_3791

/-- Given a line y = mx + b, if the reflection of point (2,3) across this line
    is (4,9), then m + b = 20/3 -/
theorem reflection_line_sum (m b : ℚ) : 
  (∀ (x y : ℚ), y = m * x + b →
    (2 + (2 * m * (m * 2 + b - 3) / (1 + m^2)) = 4 ∧
     3 + (2 * (m * 2 + b - 3) / (1 + m^2)) = 9)) →
  m + b = 20/3 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l37_3791


namespace NUMINAMATH_CALUDE_days_missed_by_mike_and_sarah_l37_3714

/-- Given the number of days missed by Vanessa, Mike, and Sarah, prove that Mike and Sarah missed 12 days together. -/
theorem days_missed_by_mike_and_sarah
  (total_days : ℕ)
  (vanessa_mike_days : ℕ)
  (vanessa_days : ℕ)
  (h1 : total_days = 17)
  (h2 : vanessa_mike_days = 14)
  (h3 : vanessa_days = 5)
  : ∃ (mike_days sarah_days : ℕ),
    mike_days + sarah_days = 12 ∧
    vanessa_days + mike_days + sarah_days = total_days ∧
    vanessa_days + mike_days = vanessa_mike_days :=
by
  sorry


end NUMINAMATH_CALUDE_days_missed_by_mike_and_sarah_l37_3714


namespace NUMINAMATH_CALUDE_crabapple_sequences_l37_3708

/-- The number of students in Mrs. Crabapple's British Literature class -/
def num_students : ℕ := 13

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of possible sequences of crabapple recipients in a week -/
def num_sequences : ℕ := num_students ^ meetings_per_week

theorem crabapple_sequences :
  num_sequences = 2197 :=
sorry

end NUMINAMATH_CALUDE_crabapple_sequences_l37_3708


namespace NUMINAMATH_CALUDE_special_property_implies_units_nine_l37_3775

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  units : Nat
  is_valid : 1 ≤ tens ∧ tens ≤ 9 ∧ 0 ≤ units ∧ units ≤ 9

/-- The property described in the problem -/
def has_special_property (n : TwoDigitNumber) : Prop :=
  n.tens + n.units + n.tens * n.units = 10 * n.tens + n.units

theorem special_property_implies_units_nine :
  ∀ n : TwoDigitNumber, has_special_property n → n.units = 9 := by
  sorry

end NUMINAMATH_CALUDE_special_property_implies_units_nine_l37_3775


namespace NUMINAMATH_CALUDE_remainder_seventeen_power_sixtythree_mod_seven_l37_3750

theorem remainder_seventeen_power_sixtythree_mod_seven :
  17^63 % 7 = 6 := by
sorry

end NUMINAMATH_CALUDE_remainder_seventeen_power_sixtythree_mod_seven_l37_3750


namespace NUMINAMATH_CALUDE_arithmetic_operations_l37_3746

theorem arithmetic_operations : 
  (12 - (-18) + (-7) - 20 = 3) ∧ 
  (-4 / (1/2) * 8 = -64) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_operations_l37_3746


namespace NUMINAMATH_CALUDE_auction_bid_ratio_l37_3747

/-- Auction problem statement -/
theorem auction_bid_ratio :
  -- Auction starts at $300
  let start_price : ℕ := 300
  -- Harry's first bid adds $200 to the starting value
  let harry_first_bid : ℕ := start_price + 200
  -- A third bidder adds three times Harry's bid
  let third_bid : ℕ := harry_first_bid + 3 * harry_first_bid
  -- Harry's final bid is $4,000
  let harry_final_bid : ℕ := 4000
  -- Harry's final bid exceeded the third bidder's bid by $1500
  let third_bid_final : ℕ := harry_final_bid - 1500
  -- Calculate the second bidder's bid
  let second_bid : ℕ := third_bid_final - 3 * harry_first_bid
  -- The ratio of the second bidder's bid to Harry's first bid is 2:1
  second_bid / harry_first_bid = 2 := by
  sorry

end NUMINAMATH_CALUDE_auction_bid_ratio_l37_3747


namespace NUMINAMATH_CALUDE_probability_ray_in_angle_l37_3751

/-- The probability of a randomly drawn ray falling within a 60-degree angle in a circular region is 1/6. -/
theorem probability_ray_in_angle (angle : ℝ) (total_angle : ℝ) : 
  angle = 60 → total_angle = 360 → angle / total_angle = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_ray_in_angle_l37_3751


namespace NUMINAMATH_CALUDE_min_value_of_function_l37_3716

theorem min_value_of_function (x : ℝ) (h : x > -1) :
  ∃ (y : ℝ), y = x + 1 / (x + 1) ∧
  ∀ (z : ℝ), z > -1 → z + 1 / (z + 1) ≥ x + 1 / (x + 1) ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l37_3716


namespace NUMINAMATH_CALUDE_georgia_carnation_cost_l37_3795

/-- The cost of a single carnation in dollars -/
def single_carnation_cost : ℚ := 1/2

/-- The cost of a dozen carnations in dollars -/
def dozen_carnation_cost : ℚ := 4

/-- The number of teachers Georgia sent carnations to -/
def num_teachers : ℕ := 5

/-- The number of friends Georgia bought carnations for -/
def num_friends : ℕ := 14

/-- The total cost of carnations Georgia would spend -/
def total_cost : ℚ := num_teachers * dozen_carnation_cost + num_friends * single_carnation_cost

theorem georgia_carnation_cost : total_cost = 27 := by sorry

end NUMINAMATH_CALUDE_georgia_carnation_cost_l37_3795


namespace NUMINAMATH_CALUDE_min_value_quadratic_fraction_l37_3768

theorem min_value_quadratic_fraction (x : ℝ) (h : x ≥ 0) :
  (9 * x^2 + 17 * x + 15) / (5 * (x + 2)) ≥ 18 * Real.sqrt 3 / 5 ∧
  ∃ y ≥ 0, (9 * y^2 + 17 * y + 15) / (5 * (y + 2)) = 18 * Real.sqrt 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_fraction_l37_3768


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l37_3767

/-- The number of rectangles in a n×n grid -/
def rectangles_in_grid (n : ℕ) : ℕ := (n.choose 2) * (n.choose 2)

/-- Theorem: The number of rectangles in a 5×5 grid is 100 -/
theorem rectangles_in_5x5_grid : rectangles_in_grid 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l37_3767


namespace NUMINAMATH_CALUDE_max_sum_given_constraints_l37_3763

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 98) 
  (h2 : x * y = 40) : 
  x + y ≤ Real.sqrt 178 := by
sorry

end NUMINAMATH_CALUDE_max_sum_given_constraints_l37_3763


namespace NUMINAMATH_CALUDE_jack_and_jill_speed_l37_3792

/-- Jack's speed function in miles per hour -/
def jack_speed (x : ℝ) : ℝ := x^2 - 7*x - 18

/-- Jill's distance function in miles -/
def jill_distance (x : ℝ) : ℝ := x^2 + x - 72

/-- Jill's time function in hours -/
def jill_time (x : ℝ) : ℝ := x + 8

/-- Theorem stating that under given conditions, Jack and Jill's speed is 2 miles per hour -/
theorem jack_and_jill_speed :
  ∃ x : ℝ, 
    jack_speed x = jill_distance x / jill_time x ∧ 
    jack_speed x = 2 :=
by sorry

end NUMINAMATH_CALUDE_jack_and_jill_speed_l37_3792


namespace NUMINAMATH_CALUDE_point_M_coordinates_l37_3761

/-- Given a line MN with slope 2, point N at (1, -1), and point M on the line y = x + 1,
    prove that the coordinates of point M are (4, 5). -/
theorem point_M_coordinates :
  let slope_MN : ℝ := 2
  let N : ℝ × ℝ := (1, -1)
  let M : ℝ × ℝ := (x, y)
  (y = x + 1) →  -- M lies on y = x + 1
  ((y - N.2) / (x - N.1) = slope_MN) →  -- slope formula
  M = (4, 5) := by
sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l37_3761
