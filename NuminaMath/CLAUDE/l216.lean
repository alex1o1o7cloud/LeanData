import Mathlib

namespace NUMINAMATH_CALUDE_robotics_club_age_problem_l216_21651

theorem robotics_club_age_problem (total_members : ℕ) (girls : ℕ) (boys : ℕ) (adults : ℕ)
  (overall_avg : ℚ) (girls_avg : ℚ) (boys_avg : ℚ) :
  total_members = 30 →
  girls = 10 →
  boys = 10 →
  adults = 10 →
  overall_avg = 22 →
  girls_avg = 18 →
  boys_avg = 20 →
  (total_members * overall_avg - girls * girls_avg - boys * boys_avg) / adults = 28 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_age_problem_l216_21651


namespace NUMINAMATH_CALUDE_virginia_started_with_96_eggs_l216_21686

/-- The number of eggs Virginia started with -/
def initial_eggs : ℕ := sorry

/-- The number of eggs Amy took away -/
def eggs_taken : ℕ := 3

/-- The number of eggs Virginia ended up with -/
def final_eggs : ℕ := 93

/-- Theorem stating that Virginia started with 96 eggs -/
theorem virginia_started_with_96_eggs : initial_eggs = 96 :=
by sorry

end NUMINAMATH_CALUDE_virginia_started_with_96_eggs_l216_21686


namespace NUMINAMATH_CALUDE_sum_of_circle_areas_l216_21636

/-- Represents a right triangle with mutually externally tangent circles at its vertices -/
structure TriangleWithCircles where
  /-- Side lengths of the right triangle -/
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  /-- Radii of the circles centered at the vertices -/
  r1 : ℝ
  r2 : ℝ
  r3 : ℝ
  /-- Conditions for the triangle and circles -/
  triangle_sides : side1^2 + side2^2 = hypotenuse^2
  circle_tangency1 : r1 + r2 = side1
  circle_tangency2 : r1 + r3 = side2
  circle_tangency3 : r2 + r3 = hypotenuse

/-- The sum of the areas of the three circles in a 6-8-10 right triangle with
    mutually externally tangent circles at its vertices is 56π -/
theorem sum_of_circle_areas (t : TriangleWithCircles)
    (h1 : t.side1 = 6)
    (h2 : t.side2 = 8)
    (h3 : t.hypotenuse = 10) :
  π * (t.r1^2 + t.r2^2 + t.r3^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_of_circle_areas_l216_21636


namespace NUMINAMATH_CALUDE_triangle_parallelogram_properties_l216_21658

/-- A triangle with a parallelogram inscribed in it. -/
structure TriangleWithParallelogram where
  /-- The length of the first side of the triangle -/
  side1 : ℝ
  /-- The length of the second side of the triangle -/
  side2 : ℝ
  /-- The length of the first side of the parallelogram -/
  para_side1 : ℝ
  /-- Assumption that the first side of the triangle is 9 -/
  h1 : side1 = 9
  /-- Assumption that the second side of the triangle is 15 -/
  h2 : side2 = 15
  /-- Assumption that the first side of the parallelogram is 6 -/
  h3 : para_side1 = 6
  /-- Assumption that the parallelogram is inscribed in the triangle -/
  h4 : para_side1 ≤ side1 ∧ para_side1 ≤ side2
  /-- Assumption that the diagonals of the parallelogram are parallel to the sides of the triangle -/
  h5 : True  -- This is a placeholder as we can't directly represent this geometrical property

/-- The theorem stating the properties of the triangle and parallelogram -/
theorem triangle_parallelogram_properties (tp : TriangleWithParallelogram) :
  ∃ (para_side2 triangle_side3 : ℝ),
    para_side2 = 4 * Real.sqrt 2 ∧
    triangle_side3 = 18 :=
  sorry

end NUMINAMATH_CALUDE_triangle_parallelogram_properties_l216_21658


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l216_21621

theorem smallest_n_congruence (n : ℕ) : n = 11 ↔ 
  (n > 0 ∧ 19 * n ≡ 546 [ZMOD 13] ∧ 
   ∀ m : ℕ, m > 0 ∧ m < n → ¬(19 * m ≡ 546 [ZMOD 13])) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l216_21621


namespace NUMINAMATH_CALUDE_complex_circle_equation_l216_21679

/-- The set of complex numbers z satisfying |z-i| = |3-4i| forms a circle in the complex plane -/
theorem complex_circle_equation : 
  ∃ (center : ℂ) (radius : ℝ), 
    {z : ℂ | Complex.abs (z - Complex.I) = Complex.abs (3 - 4 * Complex.I)} = 
    {z : ℂ | Complex.abs (z - center) = radius} :=
sorry

end NUMINAMATH_CALUDE_complex_circle_equation_l216_21679


namespace NUMINAMATH_CALUDE_perimeter_of_20_rectangles_l216_21655

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Creates a list of rectangles following the given pattern -/
def createRectangles (n : ℕ) : List Rectangle :=
  List.range n |>.map (fun i => ⟨i + 1, i + 2⟩)

/-- Calculates the perimeter of a polygon formed by arranging rectangles -/
def polygonPerimeter (rectangles : List Rectangle) : ℕ :=
  sorry

theorem perimeter_of_20_rectangles :
  let rectangles := createRectangles 20
  polygonPerimeter rectangles = 462 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_20_rectangles_l216_21655


namespace NUMINAMATH_CALUDE_archer_problem_l216_21637

theorem archer_problem (n m : ℕ) : 
  (10 < n) → 
  (n < 20) → 
  (5 * m = 3 * (n - m)) → 
  (n = 16 ∧ m = 6) := by
sorry

end NUMINAMATH_CALUDE_archer_problem_l216_21637


namespace NUMINAMATH_CALUDE_quadratic_minimum_change_l216_21617

/-- Given a quadratic polynomial f(x) = ax^2 + bx + c, 
    if adding x^2 to f(x) increases its minimum value by 1
    and subtracting x^2 from f(x) decreases its minimum value by 3,
    then adding 2x^2 to f(x) will increase its minimum value by 3/2. -/
theorem quadratic_minimum_change 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h_quad : ∀ x, f x = a * x^2 + b * x + c)
  (h_pos : a > 0)
  (h_add : (- b^2 / (4 * (a + 1)) + c) - (- b^2 / (4 * a) + c) = 1)
  (h_sub : (- b^2 / (4 * a) + c) - (- b^2 / (4 * (a - 1)) + c) = 3) :
  (- b^2 / (4 * a) + c) - (- b^2 / (4 * (a + 2)) + c) = 3/2 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_minimum_change_l216_21617


namespace NUMINAMATH_CALUDE_first_complete_column_coverage_l216_21600

theorem first_complete_column_coverage : ∃ n : ℕ, 
  n = 32 ∧ 
  (∀ k ≤ n, ∃ m ≤ n, m * (m + 1) / 2 % 12 = k % 12) ∧
  (∀ j < n, ¬(∀ k ≤ 11, ∃ m ≤ j, m * (m + 1) / 2 % 12 = k % 12)) := by
  sorry

end NUMINAMATH_CALUDE_first_complete_column_coverage_l216_21600


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l216_21691

theorem min_value_quadratic_form :
  ∀ x y : ℝ, x^2 + x*y + y^2 ≥ 0 ∧ (x^2 + x*y + y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l216_21691


namespace NUMINAMATH_CALUDE_minuend_value_l216_21675

theorem minuend_value (minuend subtrahend difference : ℕ) 
  (h : minuend + subtrahend + difference = 600) : minuend = 300 := by
  sorry

end NUMINAMATH_CALUDE_minuend_value_l216_21675


namespace NUMINAMATH_CALUDE_log_equality_l216_21695

theorem log_equality (a b : ℝ) (ha : a = Real.log 625 / Real.log 16) (hb : b = Real.log 25 / Real.log 4) :
  a = b := by
  sorry

end NUMINAMATH_CALUDE_log_equality_l216_21695


namespace NUMINAMATH_CALUDE_replaced_man_weight_l216_21680

theorem replaced_man_weight 
  (n : ℕ) 
  (initial_average : ℝ) 
  (weight_increase : ℝ) 
  (new_man_weight : ℝ) 
  (h1 : n = 10)
  (h2 : weight_increase = 2.5)
  (h3 : new_man_weight = 93) :
  new_man_weight - n * weight_increase = 68 := by
  sorry

end NUMINAMATH_CALUDE_replaced_man_weight_l216_21680


namespace NUMINAMATH_CALUDE_chess_swimming_percentage_l216_21601

theorem chess_swimming_percentage (total_students : ℕ) 
  (chess_percentage : ℚ) (swimming_students : ℕ) :
  total_students = 1000 →
  chess_percentage = 1/5 →
  swimming_students = 20 →
  (swimming_students : ℚ) / (chess_percentage * total_students) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_chess_swimming_percentage_l216_21601


namespace NUMINAMATH_CALUDE_fraction_equivalence_l216_21628

theorem fraction_equivalence : 
  let original_numerator : ℚ := 4
  let original_denominator : ℚ := 7
  let target_numerator : ℚ := 7
  let target_denominator : ℚ := 9
  let n : ℚ := 13/2
  (original_numerator + n) / (original_denominator + n) = target_numerator / target_denominator :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l216_21628


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l216_21684

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 = 450) →
  (a 1 + a 9 = 180) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l216_21684


namespace NUMINAMATH_CALUDE_simple_interest_rate_percent_l216_21676

/-- Given a simple interest scenario, prove that the rate percent is 10% -/
theorem simple_interest_rate_percent (P A T : ℝ) (h1 : P = 750) (h2 : A = 1125) (h3 : T = 5) :
  let SI := A - P
  let R := (SI * 100) / (P * T)
  R = 10 := by sorry

end NUMINAMATH_CALUDE_simple_interest_rate_percent_l216_21676


namespace NUMINAMATH_CALUDE_initial_solution_amount_l216_21625

theorem initial_solution_amount 
  (x : ℝ) -- initial amount of solution in ml
  (h1 : x - 200 + 1000 = 2000) -- equation representing the process
  : x = 1200 :=
by sorry

end NUMINAMATH_CALUDE_initial_solution_amount_l216_21625


namespace NUMINAMATH_CALUDE_mark_survival_days_l216_21606

/- Define the problem parameters -/
def num_astronauts : ℕ := 6
def food_days_per_astronaut : ℕ := 5
def water_per_astronaut : ℝ := 50
def potato_yield_per_sqm : ℝ := 2.5
def water_required_per_sqm : ℝ := 4
def potato_needed_per_day : ℝ := 1.875

/- Define the theorem -/
theorem mark_survival_days :
  let initial_food_days := num_astronauts * food_days_per_astronaut
  let total_water := num_astronauts * water_per_astronaut
  let irrigated_area := total_water / water_required_per_sqm
  let total_potatoes := irrigated_area * potato_yield_per_sqm
  let potato_days := total_potatoes / potato_needed_per_day
  initial_food_days + potato_days = 130 := by
  sorry


end NUMINAMATH_CALUDE_mark_survival_days_l216_21606


namespace NUMINAMATH_CALUDE_solution_set_equality_l216_21698

open Set

/-- The solution set of the inequality |x-5|+|x+3|≥10 -/
def SolutionSet : Set ℝ := {x : ℝ | |x - 5| + |x + 3| ≥ 10}

/-- The expected result set (-∞，-4]∪[6，+∞) -/
def ExpectedSet : Set ℝ := Iic (-4) ∪ Ici 6

theorem solution_set_equality : SolutionSet = ExpectedSet := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l216_21698


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_for_2023_l216_21629

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents_for_2023 :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two 2023 exponents ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two 2023 other_exponents →
      exponents.sum ≤ other_exponents.sum ∧
      exponents.sum = 48 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_for_2023_l216_21629


namespace NUMINAMATH_CALUDE_flour_already_added_l216_21609

/-- Given a cake recipe and Mary's baking progress, calculate the cups of flour already added. -/
theorem flour_already_added
  (total_flour : ℕ)  -- Total cups of flour required by the recipe
  (sugar : ℕ)        -- Cups of sugar required by the recipe
  (h1 : total_flour = 14)  -- The recipe requires 14 cups of flour
  (h2 : sugar = 9)         -- The recipe requires 9 cups of sugar
  : total_flour - (sugar + 1) = 4 := by
  sorry

end NUMINAMATH_CALUDE_flour_already_added_l216_21609


namespace NUMINAMATH_CALUDE_desired_circle_properties_l216_21648

/-- The first given circle -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y - 24 = 0

/-- The second given circle -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 8 = 0

/-- The line on which the center of the desired circle lies -/
def centerLine (x y : ℝ) : Prop := x + y = 0

/-- The equation of the desired circle -/
def desiredCircle (x y : ℝ) : Prop := (x + 3)^2 + (y - 3)^2 = 10

/-- Theorem stating that the desired circle passes through the intersection points of circle1 and circle2,
    and its center lies on the centerLine -/
theorem desired_circle_properties :
  ∀ x y : ℝ, 
    (circle1 x y ∧ circle2 x y) → 
    desiredCircle x y ∧ 
    ∃ cx cy : ℝ, centerLine cx cy ∧ desiredCircle (x - cx) (y - cy) := by
  sorry


end NUMINAMATH_CALUDE_desired_circle_properties_l216_21648


namespace NUMINAMATH_CALUDE_monkey_feeding_problem_l216_21618

theorem monkey_feeding_problem :
  ∀ (x : ℝ),
    (3/4 * x + 2 = 4/3 * (x - 2)) →
    (3/4 * x + x = 14) :=
by
  sorry

end NUMINAMATH_CALUDE_monkey_feeding_problem_l216_21618


namespace NUMINAMATH_CALUDE_four_digit_equal_digits_l216_21677

theorem four_digit_equal_digits (n : ℕ+) : 
  (∃ d : ℕ, d ∈ Finset.range 10 ∧ 12 * n.val^2 + 12 * n.val + 11 = d * 1111) → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_equal_digits_l216_21677


namespace NUMINAMATH_CALUDE_cost_increase_percentage_l216_21664

theorem cost_increase_percentage (initial_cost selling_price new_cost : ℝ) : 
  initial_cost > 0 →
  selling_price = 2.5 * initial_cost →
  new_cost > initial_cost →
  (selling_price - new_cost) / selling_price = 0.552 →
  (new_cost - initial_cost) / initial_cost = 0.12 :=
by sorry

end NUMINAMATH_CALUDE_cost_increase_percentage_l216_21664


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l216_21660

/-- If ln x - ax ≤ 2a² - 3 holds for all x > 0, then a ≥ 1 -/
theorem inequality_implies_a_range (a : ℝ) :
  (∀ x > 0, Real.log x - a * x ≤ 2 * a^2 - 3) → a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l216_21660


namespace NUMINAMATH_CALUDE_electionWaysCount_l216_21614

/-- Represents the Science Club with its election rules -/
structure ScienceClub where
  totalMembers : Nat
  aliceIndex : Nat
  bobIndex : Nat

/-- Represents the possible election outcomes -/
inductive ElectionOutcome
  | WithoutAliceAndBob (president secretary treasurer : Nat)
  | WithAliceAndBob (treasurer : Nat)

/-- Checks if an election outcome is valid according to the club's rules -/
def isValidOutcome (club : ScienceClub) (outcome : ElectionOutcome) : Prop :=
  match outcome with
  | ElectionOutcome.WithoutAliceAndBob p s t =>
      p ≠ club.aliceIndex ∧ p ≠ club.bobIndex ∧
      s ≠ club.aliceIndex ∧ s ≠ club.bobIndex ∧
      t ≠ club.aliceIndex ∧ t ≠ club.bobIndex ∧
      p ≠ s ∧ p ≠ t ∧ s ≠ t ∧
      p < club.totalMembers ∧ s < club.totalMembers ∧ t < club.totalMembers
  | ElectionOutcome.WithAliceAndBob t =>
      t ≠ club.aliceIndex ∧ t ≠ club.bobIndex ∧
      t < club.totalMembers

/-- Counts the number of valid election outcomes -/
def countValidOutcomes (club : ScienceClub) : Nat :=
  sorry

/-- The main theorem stating the number of ways to elect officers -/
theorem electionWaysCount (club : ScienceClub) 
    (h1 : club.totalMembers = 25)
    (h2 : club.aliceIndex < club.totalMembers)
    (h3 : club.bobIndex < club.totalMembers)
    (h4 : club.aliceIndex ≠ club.bobIndex) :
    countValidOutcomes club = 10649 :=
  sorry

end NUMINAMATH_CALUDE_electionWaysCount_l216_21614


namespace NUMINAMATH_CALUDE_intersection_point_on_graph_and_y_axis_l216_21619

/-- The quadratic function f(x) = (x-1)^2 + 2 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The point where f intersects the y-axis -/
def intersection_point : ℝ × ℝ := (0, 3)

/-- Theorem: The intersection_point lies on both the y-axis and the graph of f -/
theorem intersection_point_on_graph_and_y_axis :
  (intersection_point.1 = 0) ∧ 
  (intersection_point.2 = f intersection_point.1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_on_graph_and_y_axis_l216_21619


namespace NUMINAMATH_CALUDE_max_digit_occurrence_l216_21620

/-- Represents the range of apartment numbers on each floor -/
def apartment_range : Set ℕ := {n | 0 ≤ n ∧ n ≤ 35}

/-- Counts the occurrences of a digit in a given number -/
def count_digit (d : ℕ) (n : ℕ) : ℕ := sorry

/-- Counts the occurrences of a digit in a range of numbers -/
def count_digit_in_range (d : ℕ) (range : Set ℕ) : ℕ := sorry

/-- Counts the occurrences of a digit in the hundreds place for a floor -/
def count_digit_hundreds (d : ℕ) (floor : ℕ) : ℕ := sorry

/-- The main theorem stating that the maximum occurrence of any digit is 36 -/
theorem max_digit_occurrence :
  ∃ d : ℕ, d < 10 ∧
    (count_digit_in_range d apartment_range +
     count_digit_in_range d apartment_range +
     count_digit_in_range d apartment_range +
     count_digit_hundreds 1 1 +
     count_digit_hundreds 2 2 +
     count_digit_hundreds 3 3) = 36 ∧
    ∀ d' : ℕ, d' < 10 →
      (count_digit_in_range d' apartment_range +
       count_digit_in_range d' apartment_range +
       count_digit_in_range d' apartment_range +
       count_digit_hundreds 1 1 +
       count_digit_hundreds 2 2 +
       count_digit_hundreds 3 3) ≤ 36 := by
  sorry

#check max_digit_occurrence

end NUMINAMATH_CALUDE_max_digit_occurrence_l216_21620


namespace NUMINAMATH_CALUDE_sum_product_inequality_l216_21682

theorem sum_product_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_four : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l216_21682


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l216_21632

theorem quadratic_equations_solutions :
  (∀ x, x * (x - 1) - 3 * (x - 1) = 0 ↔ x = 1 ∨ x = 3) ∧
  (∀ x, x^2 + 2*x - 1 = 0 ↔ x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l216_21632


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l216_21662

theorem sum_of_squares_of_roots (x : ℝ) : 
  x^2 + 8*x - 12 = 0 → ∃ r₁ r₂ : ℝ, r₁ + r₂ = -8 ∧ r₁ * r₂ = -12 ∧ r₁^2 + r₂^2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l216_21662


namespace NUMINAMATH_CALUDE_sector_arc_length_l216_21642

/-- Given a sector with area 9 and central angle 2 radians, its arc length is 6. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) : 
  area = 9 → angle = 2 → arc_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l216_21642


namespace NUMINAMATH_CALUDE_find_y_value_l216_21623

-- Define the operation
def customOp (a b : ℤ) : ℤ := (a - 1) * (b - 1)

-- State the theorem
theorem find_y_value : ∃ y : ℤ, customOp y 12 = 110 ∧ y = 11 := by
  sorry

end NUMINAMATH_CALUDE_find_y_value_l216_21623


namespace NUMINAMATH_CALUDE_square_binomial_constant_l216_21650

theorem square_binomial_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 200*x + c = (x + a)^2) → c = 10000 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_constant_l216_21650


namespace NUMINAMATH_CALUDE_horner_v4_value_l216_21605

def horner_polynomial (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

def horner_step (v : ℝ) (x : ℝ) (a : ℝ) : ℝ := v * x + a

theorem horner_v4_value :
  let x := -4
  let v0 := 3
  let v1 := horner_step v0 x 5
  let v2 := horner_step v1 x 6
  let v3 := horner_step v2 x 79
  let v4 := horner_step v3 x (-8)
  v4 = 220 :=
by sorry

end NUMINAMATH_CALUDE_horner_v4_value_l216_21605


namespace NUMINAMATH_CALUDE_star_operation_simplification_l216_21692

/-- The star operation defined as x ★ y = 2x^2 - y -/
def star (x y : ℝ) : ℝ := 2 * x^2 - y

/-- Theorem stating that k ★ (k ★ k) = k -/
theorem star_operation_simplification (k : ℝ) : star k (star k k) = k := by
  sorry

end NUMINAMATH_CALUDE_star_operation_simplification_l216_21692


namespace NUMINAMATH_CALUDE_bruno_initial_books_l216_21613

/-- The number of books Bruno initially had -/
def initial_books : ℕ := sorry

/-- The number of books Bruno lost -/
def lost_books : ℕ := 4

/-- The number of books Bruno's dad gave him -/
def gained_books : ℕ := 10

/-- The final number of books Bruno had -/
def final_books : ℕ := 39

/-- Theorem stating that Bruno initially had 33 books -/
theorem bruno_initial_books : 
  initial_books = 33 ∧ 
  initial_books - lost_books + gained_books = final_books :=
sorry

end NUMINAMATH_CALUDE_bruno_initial_books_l216_21613


namespace NUMINAMATH_CALUDE_fixed_point_on_AB_l216_21647

-- Define the circle C
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line L
def Line (x y : ℝ) : Prop := x + y = 9

-- Define a point P on line L
def PointOnLine (P : ℝ × ℝ) : Prop := Line P.1 P.2

-- Define tangent line from P to circle C
def TangentLine (P A : ℝ × ℝ) : Prop :=
  Circle A.1 A.2 ∧ (∃ t : ℝ, A.1 = P.1 + t * (A.2 - P.2) ∧ A.2 = P.2 - t * (A.1 - P.1))

-- Theorem statement
theorem fixed_point_on_AB (P A B : ℝ × ℝ) :
  PointOnLine P →
  TangentLine P A →
  TangentLine P B →
  A ≠ B →
  ∃ t : ℝ, (4/9 : ℝ) = A.1 + t * (B.1 - A.1) ∧ (8/9 : ℝ) = A.2 + t * (B.2 - A.2) :=
sorry

end NUMINAMATH_CALUDE_fixed_point_on_AB_l216_21647


namespace NUMINAMATH_CALUDE_reading_plan_theorem_l216_21687

/-- Represents a book with a given number of pages --/
structure Book where
  pages : ℕ

/-- Represents Mrs. Hilt's reading plan --/
structure ReadingPlan where
  book1 : Book
  book2 : Book
  book3 : Book
  firstTwoDaysBook1Percent : ℝ
  firstTwoDaysBook2Percent : ℝ
  day3And4Book1Fraction : ℝ
  day3And4Book2Fraction : ℝ
  day3And4Book3Percent : ℝ
  readingRate : ℕ  -- pages per hour

def calculateRemainingPages (plan : ReadingPlan) : ℕ :=
  sorry

def calculateAverageSpeedFirstFourDays (plan : ReadingPlan) : ℝ :=
  sorry

def calculateTotalReadingHours (plan : ReadingPlan) : ℕ :=
  sorry

theorem reading_plan_theorem (plan : ReadingPlan) 
  (h1 : plan.book1.pages = 457)
  (h2 : plan.book2.pages = 336)
  (h3 : plan.book3.pages = 520)
  (h4 : plan.firstTwoDaysBook1Percent = 0.35)
  (h5 : plan.firstTwoDaysBook2Percent = 0.25)
  (h6 : plan.day3And4Book1Fraction = 1/3)
  (h7 : plan.day3And4Book2Fraction = 1/2)
  (h8 : plan.day3And4Book3Percent = 0.10)
  (h9 : plan.readingRate = 50) :
  calculateRemainingPages plan = 792 ∧
  calculateAverageSpeedFirstFourDays plan = 130.25 ∧
  calculateTotalReadingHours plan = 27 :=
sorry

end NUMINAMATH_CALUDE_reading_plan_theorem_l216_21687


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l216_21666

/-- Represents the cost function for a caterer -/
structure CatererCost where
  basicFee : ℕ
  perPersonCost : ℕ

/-- Calculates the total cost for a given number of people -/
def totalCost (c : CatererCost) (people : ℕ) : ℕ :=
  c.basicFee + c.perPersonCost * people

/-- First caterer's cost structure -/
def caterer1 : CatererCost :=
  { basicFee := 50, perPersonCost := 18 }

/-- Second caterer's cost structure -/
def caterer2 : CatererCost :=
  { basicFee := 150, perPersonCost := 15 }

/-- Theorem stating that 34 is the least number of people for which the second caterer is cheaper -/
theorem second_caterer_cheaper_at_34 :
  (∀ n : ℕ, n < 34 → totalCost caterer1 n ≤ totalCost caterer2 n) ∧
  (totalCost caterer1 34 > totalCost caterer2 34) :=
sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_at_34_l216_21666


namespace NUMINAMATH_CALUDE_crepe_myrtle_count_l216_21665

theorem crepe_myrtle_count (total : ℕ) (pink : ℕ) (red : ℕ) (white : ℕ) : 
  total = 42 →
  pink = total / 3 →
  red = 2 →
  white > pink →
  white > red →
  total = pink + red + white →
  white = 26 := by
sorry

end NUMINAMATH_CALUDE_crepe_myrtle_count_l216_21665


namespace NUMINAMATH_CALUDE_bus_bike_time_difference_l216_21626

/-- Proves that the difference between bus and bike commute times is 10 minutes -/
theorem bus_bike_time_difference :
  ∀ (bus_time : ℕ),
  (30 + 3 * bus_time + 10 = 160) →
  (bus_time - 30 = 10) := by
  sorry

end NUMINAMATH_CALUDE_bus_bike_time_difference_l216_21626


namespace NUMINAMATH_CALUDE_function_growth_l216_21611

theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  f a > Real.exp a * f 0 :=
sorry

end NUMINAMATH_CALUDE_function_growth_l216_21611


namespace NUMINAMATH_CALUDE_player_B_winning_condition_l216_21678

/-- Represents the game state -/
structure GameState where
  stones : ℕ

/-- Represents a player's move -/
structure Move where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ

/-- Checks if a move is valid according to the game rules -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  move.pile1 > 0 ∧ move.pile2 > 0 ∧ move.pile3 > 0 ∧
  move.pile1 + move.pile2 + move.pile3 = state.stones ∧
  (move.pile1 > move.pile2 ∧ move.pile1 > move.pile3) ∨
  (move.pile2 > move.pile1 ∧ move.pile2 > move.pile3) ∨
  (move.pile3 > move.pile1 ∧ move.pile3 > move.pile2)

/-- Defines a winning strategy for Player B -/
def player_B_has_winning_strategy (n : ℕ) : Prop :=
  ∃ (strategy : GameState → Move),
    ∀ (initial_move : Move),
      is_valid_move { stones := n } initial_move →
        ∃ (game_sequence : ℕ → GameState),
          game_sequence 0 = { stones := n } ∧
          (∀ i : ℕ, is_valid_move (game_sequence i) (strategy (game_sequence i))) ∧
          ∃ (end_state : ℕ), ¬is_valid_move (game_sequence end_state) (strategy (game_sequence end_state))

/-- The main theorem stating the condition for Player B's winning strategy -/
theorem player_B_winning_condition {a b : ℕ} (ha : a > 1) (hb : b > 1) :
  player_B_has_winning_strategy (a^b) ↔ ∃ k : ℕ, k > 1 ∧ (a^b = 3^k ∨ a^b = 3^k - 1) :=
sorry

end NUMINAMATH_CALUDE_player_B_winning_condition_l216_21678


namespace NUMINAMATH_CALUDE_locus_and_angle_property_l216_21681

-- Define the circle E
def circle_E (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point F
def point_F : ℝ × ℝ := (1, 0)

-- Define the locus of points Q
def locus_Q (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the line y = k(x-1)
def line_k (k x y : ℝ) : Prop := y = k * (x - 1)

-- Define the angle equality condition
def angle_equality (T R S : ℝ × ℝ) : Prop :=
  let (tx, _) := T
  let (rx, ry) := R
  let (sx, sy) := S
  (ry / (rx - tx)) + (sy / (sx - tx)) = 0

-- State the theorem
theorem locus_and_angle_property :
  -- Part 1: The locus of Q forms the given ellipse
  (∀ x y : ℝ, (∃ px py : ℝ, circle_E px py ∧ 
    (x - px)^2 + (y - py)^2 = ((x - 1) - px)^2 + (y - py)^2) 
    ↔ locus_Q x y) ∧
  -- Part 2: There exists a point T satisfying the angle property
  (∃ t : ℝ, t = 4 ∧ 
    ∀ k r s : ℝ, 
      locus_Q r s ∧ line_k k r s → 
      angle_equality (t, 0) (r, s) (s, k*(s-1))) :=
sorry

end NUMINAMATH_CALUDE_locus_and_angle_property_l216_21681


namespace NUMINAMATH_CALUDE_angle_subtraction_theorem_l216_21688

-- Define a custom type for angle measurements in degrees, minutes, and seconds
structure AngleDMS where
  degrees : Int
  minutes : Int
  seconds : Int

-- Define the subtraction operation for AngleDMS
def AngleDMS.sub (a b : AngleDMS) : AngleDMS :=
  sorry

theorem angle_subtraction_theorem :
  let a := AngleDMS.mk 108 18 25
  let b := AngleDMS.mk 56 23 32
  let result := AngleDMS.mk 51 54 53
  a.sub b = result := by sorry

end NUMINAMATH_CALUDE_angle_subtraction_theorem_l216_21688


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l216_21603

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x | x < -5 ∨ x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l216_21603


namespace NUMINAMATH_CALUDE_endpoint_sum_l216_21657

/-- Given a line segment with one endpoint at (10, -5) and its midpoint,
    when scaled by a factor of 2 along each axis, results in the point (12, -18),
    prove that the sum of the coordinates of the other endpoint is -11. -/
theorem endpoint_sum (x y : ℝ) : 
  (10 + x) / 2 = 6 ∧ (-5 + y) / 2 = -9 → x + y = -11 := by
  sorry

end NUMINAMATH_CALUDE_endpoint_sum_l216_21657


namespace NUMINAMATH_CALUDE_range_of_m_l216_21622

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_domain : ∀ x, f x ≠ 0 → x ∈ Set.Icc (-2 : ℝ) 2
axiom f_decreasing : ∀ x y, x < y ∧ x ∈ Set.Icc (-2 : ℝ) 0 → f x > f y

-- Define the inequality condition
def inequality_condition (m : ℝ) : Prop := f (1 - m) + f (1 - m^2) < 0

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, inequality_condition m ↔ m ∈ Set.Icc (-1 : ℝ) 1 ∧ m ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l216_21622


namespace NUMINAMATH_CALUDE_grapefruit_orchards_l216_21634

theorem grapefruit_orchards (total : ℕ) (lemon : ℕ) (orange : ℕ) (lime_grapefruit : ℕ) :
  total = 16 →
  lemon = 8 →
  orange = lemon / 2 →
  lime_grapefruit = total - lemon - orange →
  lime_grapefruit / 2 = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_grapefruit_orchards_l216_21634


namespace NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l216_21641

-- Define propositions P and Q
def P (x : ℝ) : Prop := |x - 1| < 4
def Q (x : ℝ) : Prop := (x - 2) * (3 - x) > 0

-- Theorem statement
theorem P_necessary_not_sufficient_for_Q :
  (∀ x, Q x → P x) ∧ (∃ x, P x ∧ ¬Q x) :=
sorry

end NUMINAMATH_CALUDE_P_necessary_not_sufficient_for_Q_l216_21641


namespace NUMINAMATH_CALUDE_angle_between_vectors_l216_21699

/-- The angle between two planar vectors satisfying given conditions -/
theorem angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 7)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 3)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 2) :
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l216_21699


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l216_21694

/-- The capacity of a water tank in gallons. -/
def tank_capacity : ℝ := 72

/-- The difference in gallons between 40% full and 10% empty. -/
def difference : ℝ := 36

/-- Proves that the tank capacity is correct given the condition. -/
theorem tank_capacity_proof : 
  tank_capacity * 0.4 = tank_capacity * 0.9 - difference :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l216_21694


namespace NUMINAMATH_CALUDE_function_equality_implies_m_zero_l216_21674

/-- Given two functions f and g, prove that m = 0 when 3f(3) = g(3) -/
theorem function_equality_implies_m_zero (m : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 3*x + m
  let g := fun (x : ℝ) => x^2 - 3*x + 5*m
  3 * f 3 = g 3 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_zero_l216_21674


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l216_21697

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₃ = 7 and a₇ = 3, prove that a₁₀ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a3 : a 3 = 7) 
  (h_a7 : a 7 = 3) : 
  a 10 = 0 := by
sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l216_21697


namespace NUMINAMATH_CALUDE_semicircle_radius_l216_21673

theorem semicircle_radius (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_area : π * a^2 / 8 = 12.5 * π) (h_arc : π * b / 2 = 11 * π) :
  c / 2 = Real.sqrt 584 / 2 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_l216_21673


namespace NUMINAMATH_CALUDE_opposite_reciprocal_problem_l216_21652

theorem opposite_reciprocal_problem (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  (a + b = 0) ∧ 
  (c * d = 1) ∧ 
  ((a + b) / 3 + m^2 - 5 * c * d = 11) := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_problem_l216_21652


namespace NUMINAMATH_CALUDE_multiple_problem_l216_21645

theorem multiple_problem (n m : ℝ) : n = 5 → m * n - 15 = 2 * n + 10 → m = 7 := by
  sorry

end NUMINAMATH_CALUDE_multiple_problem_l216_21645


namespace NUMINAMATH_CALUDE_expression_evaluation_l216_21602

theorem expression_evaluation (x : ℕ) (h : x = 3) : x^2 + x * (x^(x^2)) = 59058 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l216_21602


namespace NUMINAMATH_CALUDE_printing_press_completion_time_l216_21608

-- Define the start time (9:00 AM)
def start_time : ℕ := 9

-- Define the time when half the order is completed (12:00 PM)
def half_time : ℕ := 12

-- Define the time to complete half the order
def half_duration : ℕ := half_time - start_time

-- Theorem: The printing press will finish the entire order at 3:00 PM
theorem printing_press_completion_time :
  start_time + 2 * half_duration = 15 := by
  sorry

end NUMINAMATH_CALUDE_printing_press_completion_time_l216_21608


namespace NUMINAMATH_CALUDE_equation_solutions_l216_21633

theorem equation_solutions : 
  (∀ x : ℝ, x^2 - 10*x + 16 = 0 ↔ x = 8 ∨ x = 2) ∧
  (∀ x : ℝ, x*(x-3) = 6-2*x ↔ x = 3 ∨ x = -2) := by sorry

end NUMINAMATH_CALUDE_equation_solutions_l216_21633


namespace NUMINAMATH_CALUDE_distribute_7_4_l216_21689

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 104 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_7_4 : distribute 7 4 = 104 := by sorry

end NUMINAMATH_CALUDE_distribute_7_4_l216_21689


namespace NUMINAMATH_CALUDE_password_decryption_probability_l216_21667

theorem password_decryption_probability 
  (p : ℝ) 
  (hp : p = 1 / 4) 
  (n : ℕ) 
  (hn : n = 3) :
  (n.choose 2 : ℝ) * p^2 * (1 - p) = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_password_decryption_probability_l216_21667


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l216_21649

theorem book_arrangement_count : ℕ → ℕ → ℕ
  | 4, 6 => 17280
  | _, _ => 0

/-- The number of ways to arrange math and history books with specific constraints -/
theorem book_arrangement_proof (m h : ℕ) (hm : m = 4) (hh : h = 6) :
  book_arrangement_count m h = 4 * 3 * 2 * Nat.factorial h :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l216_21649


namespace NUMINAMATH_CALUDE_parabola_point_x_coord_l216_21670

/-- A point on a parabola with a specific distance to its focus -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_focus : (x - 1)^2 + y^2 = 25

/-- The x-coordinate of a point on a parabola with distance 5 to its focus is 4 -/
theorem parabola_point_x_coord (M : ParabolaPoint) : M.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_x_coord_l216_21670


namespace NUMINAMATH_CALUDE_colleen_pencils_colleen_pencils_proof_l216_21653

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (colleen_extra : ℕ) : ℕ :=
  let joy_total := joy_pencils * pencil_cost
  let colleen_total := joy_total + colleen_extra
  colleen_total / pencil_cost

#check colleen_pencils 30 4 80 = 50

theorem colleen_pencils_proof :
  colleen_pencils 30 4 80 = 50 := by
  sorry

end NUMINAMATH_CALUDE_colleen_pencils_colleen_pencils_proof_l216_21653


namespace NUMINAMATH_CALUDE_count_valid_pairs_l216_21638

def validPair (x y : ℕ) : Prop :=
  2 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 16 ∧ 3 * x = y

theorem count_valid_pairs :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 4 ∧
    (∀ (p : ℕ × ℕ), p ∈ pairs ↔ validPair p.1 p.2) :=
sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l216_21638


namespace NUMINAMATH_CALUDE_train_speed_calculation_l216_21654

-- Define the given constants
def train_length : ℝ := 390  -- in meters
def man_speed : ℝ := 2       -- in km/h
def crossing_time : ℝ := 52  -- in seconds

-- Define the theorem
theorem train_speed_calculation :
  ∃ (train_speed : ℝ),
    train_speed > 0 ∧
    train_speed = 25 ∧
    (train_speed + man_speed) * (crossing_time / 3600) = train_length / 1000 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_calculation_l216_21654


namespace NUMINAMATH_CALUDE_power_two_equals_four_l216_21616

theorem power_two_equals_four : 2^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_two_equals_four_l216_21616


namespace NUMINAMATH_CALUDE_system_solution_l216_21644

-- Define the system of equations
def equation1 (x y : ℚ) : Prop := (2 * x - 3) / (3 * x - y) = 3 / 5
def equation2 (x y : ℚ) : Prop := x^2 + y = 7

-- Define the solution set
def solution_set : Set (ℚ × ℚ) := {(-2/3, 47/9), (3, 4)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℚ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l216_21644


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l216_21615

/-- Jessie's weight loss calculation -/
theorem jessie_weight_loss (initial_weight current_weight : ℕ) :
  initial_weight = 69 →
  current_weight = 34 →
  initial_weight - current_weight = 35 := by
sorry

end NUMINAMATH_CALUDE_jessie_weight_loss_l216_21615


namespace NUMINAMATH_CALUDE_rational_equation_solution_l216_21624

theorem rational_equation_solution (x : ℝ) : 
  (1 / (x^2 + 8*x - 6) + 1 / (x^2 + 5*x - 6) + 1 / (x^2 - 14*x - 6) = 0) ↔ 
  (x = 3 ∨ x = -2 ∨ x = -6 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l216_21624


namespace NUMINAMATH_CALUDE_lottery_probability_l216_21646

def megaball_count : ℕ := 30
def winnerball_count : ℕ := 50
def ordered_winnerball_count : ℕ := 2
def unordered_winnerball_count : ℕ := 5

def megaball_prob : ℚ := 1 / megaball_count
def ordered_winnerball_prob : ℚ := 1 / (winnerball_count * (winnerball_count - 1))
def unordered_winnerball_prob : ℚ := 1 / (Nat.choose (winnerball_count - ordered_winnerball_count) unordered_winnerball_count)

theorem lottery_probability :
  megaball_prob * ordered_winnerball_prob * unordered_winnerball_prob = 1 / 125703480000 :=
by sorry

end NUMINAMATH_CALUDE_lottery_probability_l216_21646


namespace NUMINAMATH_CALUDE_travel_time_calculation_l216_21612

/-- Represents the travel times of a motorcyclist and cyclist meeting on a road --/
def TravelTimes (t m c : ℝ) : Prop :=
  t > 0 ∧ 
  m > t ∧ 
  c > t ∧
  m - t = 2 ∧ 
  c - t = 4.5

theorem travel_time_calculation (t m c : ℝ) (h : TravelTimes t m c) : 
  m = 5 ∧ c = 7.5 := by
  sorry

#check travel_time_calculation

end NUMINAMATH_CALUDE_travel_time_calculation_l216_21612


namespace NUMINAMATH_CALUDE_total_black_dots_l216_21669

/-- The number of butterflies -/
def num_butterflies : ℕ := 397

/-- The number of black dots per butterfly -/
def black_dots_per_butterfly : ℕ := 12

/-- Theorem: The total number of black dots is 4764 -/
theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end NUMINAMATH_CALUDE_total_black_dots_l216_21669


namespace NUMINAMATH_CALUDE_impossible_time_reduction_l216_21696

/-- Proves that it's impossible to reduce the time taken to travel 1 kilometer by 1 minute when starting from a speed of 60 km/h. -/
theorem impossible_time_reduction (initial_speed : ℝ) (distance : ℝ) (time_reduction : ℝ) : 
  initial_speed = 60 → distance = 1 → time_reduction = 1 → 
  ¬ ∃ (new_speed : ℝ), new_speed > 0 ∧ distance / new_speed = distance / initial_speed - time_reduction :=
by sorry

end NUMINAMATH_CALUDE_impossible_time_reduction_l216_21696


namespace NUMINAMATH_CALUDE_remainder_problem_l216_21640

theorem remainder_problem (N : ℕ) (h : N % 899 = 63) : N % 29 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l216_21640


namespace NUMINAMATH_CALUDE_biology_class_size_l216_21683

theorem biology_class_size :
  ∀ (S : ℕ), 
    (S : ℝ) * 0.8 * 0.25 = 8 →
    S = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_biology_class_size_l216_21683


namespace NUMINAMATH_CALUDE_sum_equals_twelve_l216_21690

theorem sum_equals_twelve 
  (a b c : ℕ) 
  (h : 28 * a + 30 * b + 31 * c = 365) : 
  a + b + c = 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_twelve_l216_21690


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_range_l216_21607

theorem quadratic_roots_difference_range (a b c x₁ x₂ : ℝ) :
  a > b →
  b > c →
  a + b + c = 0 →
  a * x₁^2 + 2 * b * x₁ + c = 0 →
  a * x₂^2 + 2 * b * x₂ + c = 0 →
  Real.sqrt 3 < |x₁ - x₂| ∧ |x₁ - x₂| < 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_range_l216_21607


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l216_21672

theorem divisibility_of_expression (a b c : ℤ) (h : 4 * b = 10 - 3 * a + c) :
  ∃ k : ℤ, 3 * b + 15 - c = 1 * k :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l216_21672


namespace NUMINAMATH_CALUDE_curve_and_circle_properties_l216_21610

-- Define the points and vectors
def E : ℝ × ℝ := (-2, 0)
def F : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 1)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the curve C
def C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the condition for point M on curve C
def M_condition (x y : ℝ) : Prop :=
  dot_product (x + 2, y) (x - 2, y) = -3

-- Define the point P and the tangent condition
def P (a b : ℝ) : ℝ × ℝ := (a, b)
def tangent_condition (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), C x y ∧ (a - x)^2 + (b - y)^2 = (a - 2)^2 + (b - 1)^2

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  (x - 6/5)^2 + (y - 3/5)^2 = (3 * Real.sqrt 5 / 5 - 1)^2

-- State the theorem
theorem curve_and_circle_properties :
  ∀ (x y a b : ℝ),
    C x y ∧
    M_condition x y ∧
    tangent_condition a b →
    (∀ (u v : ℝ), C u v ↔ (u - 1)^2 + v^2 = 1) ∧
    (∀ (r : ℝ), r > 0 → 
      (∀ (u v : ℝ), (u - a)^2 + (v - b)^2 = r^2 → ¬(C u v)) →
      r ≥ 3 * Real.sqrt 5 / 5 - 1) ∧
    circle_equation a b :=
by sorry

end NUMINAMATH_CALUDE_curve_and_circle_properties_l216_21610


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_shifted_roots_l216_21639

theorem sum_of_reciprocals_shifted_roots (a b c : ℂ) : 
  (a^3 - a - 2 = 0) → (b^3 - b - 2 = 0) → (c^3 - c - 2 = 0) →
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_shifted_roots_l216_21639


namespace NUMINAMATH_CALUDE_spencer_jump_rope_l216_21693

def initial_speed : ℕ := 4
def practice_days : List ℕ := [1, 2, 4, 5, 6]
def first_session_duration : ℕ := 10
def second_session_initial : ℕ := 10
def second_session_increase : ℕ := 5

def speed_on_day (day : ℕ) : ℕ :=
  initial_speed * (2^(day - 1))

def second_session_duration (day : ℕ) : ℕ :=
  second_session_initial + (day - 1) * second_session_increase

def jumps_on_day (day : ℕ) : ℕ :=
  speed_on_day day * (first_session_duration + second_session_duration day)

def total_jumps : ℕ :=
  practice_days.map jumps_on_day |>.sum

theorem spencer_jump_rope : total_jumps = 8600 := by
  sorry

end NUMINAMATH_CALUDE_spencer_jump_rope_l216_21693


namespace NUMINAMATH_CALUDE_total_students_is_44_l216_21661

/-- Represents the number of students who borrowed a specific number of books -/
structure BookBorrowers where
  zero : Nat
  one : Nat
  two : Nat
  threeOrMore : Nat

/-- Calculates the total number of students -/
def totalStudents (b : BookBorrowers) : Nat :=
  b.zero + b.one + b.two + b.threeOrMore

/-- Calculates the minimum number of books borrowed -/
def minBooksBorrowed (b : BookBorrowers) : Nat :=
  0 * b.zero + 1 * b.one + 2 * b.two + 3 * b.threeOrMore

/-- Theorem stating that the total number of students in the class is 44 -/
theorem total_students_is_44 (b : BookBorrowers) : 
  b.zero = 2 → 
  b.one = 12 → 
  b.two = 14 → 
  minBooksBorrowed b = 2 * totalStudents b → 
  totalStudents b = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_students_is_44_l216_21661


namespace NUMINAMATH_CALUDE_client_phones_dropped_off_kevins_phone_repair_problem_l216_21627

theorem client_phones_dropped_off (initial_phones : ℕ) (repaired_phones : ℕ) (phones_per_person : ℕ) : ℕ :=
  let remaining_phones := initial_phones - repaired_phones
  let total_phones_to_repair := 2 * phones_per_person
  total_phones_to_repair - remaining_phones

theorem kevins_phone_repair_problem :
  client_phones_dropped_off 15 3 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_client_phones_dropped_off_kevins_phone_repair_problem_l216_21627


namespace NUMINAMATH_CALUDE_inequality_proofs_l216_21659

theorem inequality_proofs :
  (∀ (a b : ℝ), a > 0 → b > 0 → (b/a) + (a/b) ≥ 2) ∧
  (∀ (x y : ℝ), x*y < 0 → (x/y) + (y/x) ≤ -2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l216_21659


namespace NUMINAMATH_CALUDE_horse_cattle_price_problem_l216_21663

theorem horse_cattle_price_problem (x y : ℚ) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) →
  x = 6 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_horse_cattle_price_problem_l216_21663


namespace NUMINAMATH_CALUDE_sufficient_lunks_for_bananas_l216_21604

/-- Represents the exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 6 / 10

/-- Represents the exchange rate between kunks and bananas -/
def kunk_to_banana_rate : ℚ := 5 / 3

/-- The number of bananas we want to purchase -/
def target_bananas : ℕ := 24

/-- The number of lunks we claim is sufficient -/
def claimed_lunks : ℕ := 25

theorem sufficient_lunks_for_bananas :
  ∃ (kunks : ℚ),
    kunks * kunk_to_banana_rate ≥ target_bananas ∧
    kunks ≤ claimed_lunks * lunk_to_kunk_rate :=
by
  sorry

#check sufficient_lunks_for_bananas

end NUMINAMATH_CALUDE_sufficient_lunks_for_bananas_l216_21604


namespace NUMINAMATH_CALUDE_final_position_on_number_line_final_position_is_28_l216_21635

/-- Given a number line where the distance from 0 to 40 is divided into 10 equal steps,
    if a person moves forward 8 steps and then back 1 step, their final position will be 28. -/
theorem final_position_on_number_line : ℝ → Prop :=
  fun final_position =>
    let total_distance : ℝ := 40
    let total_steps : ℕ := 10
    let step_size : ℝ := total_distance / total_steps
    let forward_steps : ℕ := 8
    let backward_steps : ℕ := 1
    final_position = (forward_steps - backward_steps : ℕ) * step_size

theorem final_position_is_28 : final_position_on_number_line 28 := by
  sorry

#check final_position_is_28

end NUMINAMATH_CALUDE_final_position_on_number_line_final_position_is_28_l216_21635


namespace NUMINAMATH_CALUDE_factorial_divisibility_l216_21643

theorem factorial_divisibility (n : ℕ) (M : ℕ) (h : Nat.factorial 100 = 12^n * M) 
  (h_max : ∀ k : ℕ, Nat.factorial 100 = 12^k * M → k ≤ n) : 
  (2 ∣ M) ∧ ¬(3 ∣ M) := by
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_l216_21643


namespace NUMINAMATH_CALUDE_parallelogram_base_length_l216_21685

theorem parallelogram_base_length 
  (area : ℝ) 
  (altitude_base_relation : ℝ → ℝ → Prop) :
  area = 162 →
  (∀ base height, altitude_base_relation base height → height = 2 * base) →
  ∃ base : ℝ, altitude_base_relation base (2 * base) ∧ 
    area = base * (2 * base) ∧ 
    base = 9 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_length_l216_21685


namespace NUMINAMATH_CALUDE_owen_turtle_ratio_l216_21668

def turtle_problem (owen_initial johanna_initial owen_after_month owen_final : ℕ) : Prop :=
  -- Owen initially has 21 turtles
  owen_initial = 21 ∧
  -- Johanna initially has 5 fewer turtles than Owen
  johanna_initial = owen_initial - 5 ∧
  -- After 1 month, Owen has a certain multiple of his initial number of turtles
  ∃ k : ℕ, owen_after_month = k * owen_initial ∧
  -- After 1 month, Johanna loses half of her turtles and donates the rest to Owen
  owen_final = owen_after_month + (johanna_initial / 2) ∧
  -- After all these events, Owen has 50 turtles
  owen_final = 50

theorem owen_turtle_ratio (owen_initial johanna_initial owen_after_month owen_final : ℕ)
  (h : turtle_problem owen_initial johanna_initial owen_after_month owen_final) :
  owen_after_month = 2 * owen_initial :=
by sorry

end NUMINAMATH_CALUDE_owen_turtle_ratio_l216_21668


namespace NUMINAMATH_CALUDE_cost_for_23_days_l216_21656

/-- Calculates the cost of staying in a student youth hostel for a given number of days. -/
def hostelCost (days : ℕ) : ℚ :=
  let firstWeekRate : ℚ := 18
  let additionalWeekRate : ℚ := 13
  let firstWeekDays : ℕ := min days 7
  let additionalDays : ℕ := days - firstWeekDays
  firstWeekRate * firstWeekDays + additionalWeekRate * additionalDays

/-- Proves that the cost for a 23-day stay is $334.00 -/
theorem cost_for_23_days : hostelCost 23 = 334 := by
  sorry

#eval hostelCost 23

end NUMINAMATH_CALUDE_cost_for_23_days_l216_21656


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_congruence_l216_21671

def arithmetic_sequence_sum (a : ℕ) (l : ℕ) (d : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem arithmetic_sequence_sum_congruence :
  let a := 2
  let l := 137
  let d := 5
  let S := arithmetic_sequence_sum a l d
  S % 20 = 6 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_congruence_l216_21671


namespace NUMINAMATH_CALUDE_square_root_equation_l216_21631

theorem square_root_equation (t s : ℝ) : t = 15 * s^2 → t = 3.75 → s = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l216_21631


namespace NUMINAMATH_CALUDE_rhombus_side_length_l216_21630

/-- A rhombus with perimeter 60 cm has sides of length 15 cm each. -/
theorem rhombus_side_length (perimeter : ℝ) (side_length : ℝ) : 
  perimeter = 60 → side_length * 4 = perimeter → side_length = 15 := by
  sorry

#check rhombus_side_length

end NUMINAMATH_CALUDE_rhombus_side_length_l216_21630
