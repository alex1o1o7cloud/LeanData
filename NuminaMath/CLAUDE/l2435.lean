import Mathlib

namespace NUMINAMATH_CALUDE_percentage_calculation_l2435_243506

theorem percentage_calculation (whole : ℝ) (part : ℝ) (percentage : ℝ) 
  (h1 : whole = 800)
  (h2 : part = 200)
  (h3 : percentage = (part / whole) * 100) :
  percentage = 25 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l2435_243506


namespace NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_area_l2435_243510

theorem triangle_with_consecutive_sides_and_area :
  ∃ (a b c S : ℕ), 
    (a + 1 = b) ∧ 
    (b + 1 = c) ∧ 
    (c + 1 = S) ∧
    (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (S = 6) ∧
    (2 * S = a * b) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_area_l2435_243510


namespace NUMINAMATH_CALUDE_total_seashells_eq_sum_l2435_243555

/-- The number of seashells Sam found on the beach -/
def total_seashells : ℕ := sorry

/-- The number of seashells Sam gave to Joan -/
def seashells_given : ℕ := 18

/-- The number of seashells Sam has left -/
def seashells_left : ℕ := 17

/-- Theorem stating that the total number of seashells is the sum of those given away and those left -/
theorem total_seashells_eq_sum : 
  total_seashells = seashells_given + seashells_left := by sorry

end NUMINAMATH_CALUDE_total_seashells_eq_sum_l2435_243555


namespace NUMINAMATH_CALUDE_factorization_equality_l2435_243593

-- Define the theorem
theorem factorization_equality {R : Type*} [Ring R] (a b : R) :
  2 * a^2 - a * b = a * (2 * a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2435_243593


namespace NUMINAMATH_CALUDE_triangle_property_triangle_area_l2435_243540

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  angleSum : A + B + C = π
  positiveSides : 0 < a ∧ 0 < b ∧ 0 < c
  positiveAngles : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_property (t : Triangle) 
  (h : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C) :
  t.b^2 = t.a * t.c :=
sorry

theorem triangle_area (t : Triangle) (h1 : t.a = 2 * t.c) (h2 : t.a = 2) :
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_property_triangle_area_l2435_243540


namespace NUMINAMATH_CALUDE_triangle_properties_l2435_243599

open Real

theorem triangle_properties (a b c A B C : ℝ) (h1 : 0 < A) (h2 : A < π) 
  (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) 
  (h7 : b * cos A = Real.sqrt 3 * a * sin B) (h8 : a = 1) :
  A = π / 6 ∧ 
  (∃ (S : ℝ), S = (2 + Real.sqrt 3) / 4 ∧ 
    ∀ (S' : ℝ), S' = 1 / 2 * b * c * sin A → S' ≤ S) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2435_243599


namespace NUMINAMATH_CALUDE_minimum_agreement_for_budget_constraint_l2435_243581

/-- Represents a parliament budget allocation problem -/
structure ParliamentBudget where
  members : ℕ
  items : ℕ
  limit : ℝ

/-- Defines the minimum number of members required for agreement -/
def min_agreement (pb : ParliamentBudget) : ℕ := pb.members - pb.items + 1

/-- Theorem stating the minimum agreement required for the given problem -/
theorem minimum_agreement_for_budget_constraint 
  (pb : ParliamentBudget) 
  (h_members : pb.members = 2000) 
  (h_items : pb.items = 200) :
  min_agreement pb = 1991 := by
  sorry

#eval min_agreement { members := 2000, items := 200, limit := 0 }

end NUMINAMATH_CALUDE_minimum_agreement_for_budget_constraint_l2435_243581


namespace NUMINAMATH_CALUDE_sum_w_y_l2435_243524

theorem sum_w_y (w x y z : ℚ) 
  (eq1 : w * x * y = 10)
  (eq2 : w * y * z = 5)
  (eq3 : w * x * z = 45)
  (eq4 : x * y * z = 12) :
  w + y = 19/6 := by
  sorry

end NUMINAMATH_CALUDE_sum_w_y_l2435_243524


namespace NUMINAMATH_CALUDE_max_students_distribution_l2435_243542

def number_of_pens : ℕ := 2010
def number_of_pencils : ℕ := 1050

theorem max_students_distribution (n : ℕ) :
  (n ∣ number_of_pens) ∧ 
  (n ∣ number_of_pencils) ∧ 
  (∀ m : ℕ, m > n → ¬(m ∣ number_of_pens) ∨ ¬(m ∣ number_of_pencils)) →
  n = 30 :=
by sorry

end NUMINAMATH_CALUDE_max_students_distribution_l2435_243542


namespace NUMINAMATH_CALUDE_min_people_in_group_l2435_243543

/-- Represents the number of people who like a specific fruit or combination of fruits. -/
structure FruitPreferences where
  apples : Nat
  blueberries : Nat
  cantaloupe : Nat
  dates : Nat
  blueberriesAndApples : Nat
  blueberriesAndCantaloupe : Nat
  cantaloupeAndDates : Nat

/-- The conditions given in the problem. -/
def problemConditions : FruitPreferences where
  apples := 13
  blueberries := 9
  cantaloupe := 15
  dates := 6
  blueberriesAndApples := 0  -- Derived from the solution
  blueberriesAndCantaloupe := 9  -- Derived from the solution
  cantaloupeAndDates := 6  -- Derived from the solution

/-- Theorem stating the minimum number of people in the group. -/
theorem min_people_in_group (prefs : FruitPreferences) 
  (h1 : prefs.blueberries = prefs.blueberriesAndApples + prefs.blueberriesAndCantaloupe)
  (h2 : prefs.cantaloupe = prefs.blueberriesAndCantaloupe + prefs.cantaloupeAndDates)
  (h3 : prefs = problemConditions) :
  prefs.apples + prefs.blueberriesAndCantaloupe + prefs.cantaloupeAndDates = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_people_in_group_l2435_243543


namespace NUMINAMATH_CALUDE_student_a_score_l2435_243550

/-- Calculates the score for a test based on the given grading method -/
def calculate_score (total_questions : ℕ) (correct_answers : ℕ) : ℕ :=
  let incorrect_answers := total_questions - correct_answers
  correct_answers - 2 * incorrect_answers

/-- Theorem stating that the score for the given conditions is 61 -/
theorem student_a_score :
  let total_questions : ℕ := 100
  let correct_answers : ℕ := 87
  calculate_score total_questions correct_answers = 61 := by
  sorry

end NUMINAMATH_CALUDE_student_a_score_l2435_243550


namespace NUMINAMATH_CALUDE_board_coverage_five_by_five_uncoverable_four_by_four_removed_uncoverable_four_by_five_coverable_six_by_three_coverable_l2435_243544

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : Nat
  cols : Nat
  removed_squares : Nat

/-- Checks if a board can be completely covered by non-overlapping dominoes -/
def can_be_covered (board : Checkerboard) : Prop :=
  (board.rows * board.cols - board.removed_squares) % 2 = 0

/-- Theorem: A board can be covered iff the number of squares is even -/
theorem board_coverage (board : Checkerboard) :
  can_be_covered board ↔ (board.rows * board.cols - board.removed_squares) % 2 = 0 := by
  sorry

/-- 5x5 board cannot be covered -/
theorem five_by_five_uncoverable :
  ¬(can_be_covered { rows := 5, cols := 5, removed_squares := 0 }) := by
  sorry

/-- 4x4 board with one square removed cannot be covered -/
theorem four_by_four_removed_uncoverable :
  ¬(can_be_covered { rows := 4, cols := 4, removed_squares := 1 }) := by
  sorry

/-- 4x5 board can be covered -/
theorem four_by_five_coverable :
  can_be_covered { rows := 4, cols := 5, removed_squares := 0 } := by
  sorry

/-- 6x3 board can be covered -/
theorem six_by_three_coverable :
  can_be_covered { rows := 6, cols := 3, removed_squares := 0 } := by
  sorry

end NUMINAMATH_CALUDE_board_coverage_five_by_five_uncoverable_four_by_four_removed_uncoverable_four_by_five_coverable_six_by_three_coverable_l2435_243544


namespace NUMINAMATH_CALUDE_max_value_M_l2435_243563

theorem max_value_M (x y z w : ℝ) (h : x + y + z + w = 1) :
  let M := x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ + y₀ + z₀ + w₀ = 1 ∧
    (∀ x y z w, x + y + z + w = 1 →
      x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z ≤
      x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀) ∧
    x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀ = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_M_l2435_243563


namespace NUMINAMATH_CALUDE_min_sum_a1_a5_l2435_243592

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ r > 0, a (n + 1) = r * a n ∧ a n > 0

-- State the theorem
theorem min_sum_a1_a5 (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a) 
  (h_prod : a 5 * a 4 * a 2 * a 1 = 16) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = 4 → a 1 + a 5 ≥ x + y :=
sorry

end NUMINAMATH_CALUDE_min_sum_a1_a5_l2435_243592


namespace NUMINAMATH_CALUDE_unique_digit_solution_l2435_243597

/-- Represents a six-digit number as a list of digits -/
def SixDigitNumber := List Nat

/-- Converts a three-digit number to a six-digit number -/
def toSixDigit (n : Nat) : SixDigitNumber :=
  sorry

/-- Converts a list of digits to a natural number -/
def fromDigits (digits : List Nat) : Nat :=
  sorry

/-- Checks if all digits in a list are distinct -/
def allDistinct (digits : List Nat) : Prop :=
  sorry

/-- Theorem: Unique solution for the given digit equation system -/
theorem unique_digit_solution :
  ∃! (A B C D E F : Nat),
    A ∈ Finset.range 10 ∧
    B ∈ Finset.range 10 ∧
    C ∈ Finset.range 10 ∧
    D ∈ Finset.range 10 ∧
    E ∈ Finset.range 10 ∧
    F ∈ Finset.range 10 ∧
    allDistinct [A, B, C, D, E, F] ∧
    fromDigits [A, B, C] ^ 2 = fromDigits (toSixDigit (fromDigits [D, A, E, C, F, B])) ∧
    fromDigits [C, B, A] ^ 2 = fromDigits (toSixDigit (fromDigits [E, D, C, A, B, F])) ∧
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 :=
  by sorry

end NUMINAMATH_CALUDE_unique_digit_solution_l2435_243597


namespace NUMINAMATH_CALUDE_units_digit_G_500_l2435_243520

-- Define the sequence G_n
def G (n : ℕ) : ℕ := 2^(3^n) + 1

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem units_digit_G_500 : unitsDigit (G 500) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_G_500_l2435_243520


namespace NUMINAMATH_CALUDE_canada_animal_population_l2435_243561

/-- Represents the population of different species in Canada -/
structure CanadaPopulation where
  humans : ℚ
  moose : ℚ
  beavers : ℚ
  caribou : ℚ
  wolves : ℚ
  grizzly_bears : ℚ

/-- The relationships between species in Canada -/
def population_relationships (p : CanadaPopulation) : Prop :=
  p.beavers = 2 * p.moose ∧
  p.humans = 19 * p.beavers ∧
  3 * p.caribou = 2 * p.moose ∧
  p.wolves = 4 * p.caribou ∧
  3 * p.grizzly_bears = p.wolves

/-- The theorem stating the combined population of animals given the human population -/
theorem canada_animal_population 
  (p : CanadaPopulation) 
  (h : population_relationships p) 
  (humans_pop : p.humans = 38) : 
  p.moose + p.beavers + p.caribou + p.wolves + p.grizzly_bears = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_canada_animal_population_l2435_243561


namespace NUMINAMATH_CALUDE_race_start_calculation_l2435_243507

/-- Given a kilometer race where runner A can give runner B a 200 meters start,
    and runner B can give runner C a 250 meters start,
    prove that runner A can give runner C a 400 meters start. -/
theorem race_start_calculation (Va Vb Vc : ℝ) 
  (h1 : Va / Vb = 1000 / 800)
  (h2 : Vb / Vc = 1000 / 750) :
  Va / Vc = 1000 / 600 :=
by sorry

end NUMINAMATH_CALUDE_race_start_calculation_l2435_243507


namespace NUMINAMATH_CALUDE_closest_point_on_line_l2435_243511

def v (t : ℝ) : ℝ × ℝ × ℝ := (3 + 8*t, -2 + 6*t, -4 - 2*t)

def a : ℝ × ℝ × ℝ := (5, 7, 3)

def direction : ℝ × ℝ × ℝ := (8, 6, -2)

theorem closest_point_on_line (t : ℝ) : 
  (t = 7/13) ↔ 
  (∀ s : ℝ, ‖v t - a‖ ≤ ‖v s - a‖) :=
sorry

end NUMINAMATH_CALUDE_closest_point_on_line_l2435_243511


namespace NUMINAMATH_CALUDE_circle_intersection_radius_range_l2435_243559

/-- Given two intersecting circles O and M in a Cartesian coordinate system,
    where O has center (0, 0) and radius r (r > 0),
    and M has center (3, -4) and radius 2,
    the range of possible values for r is 3 < r < 7. -/
theorem circle_intersection_radius_range (r : ℝ) : 
  r > 0 ∧ 
  (∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ (x - 3)^2 + (y + 4)^2 = 4) →
  3 < r ∧ r < 7 := by
  sorry

#check circle_intersection_radius_range

end NUMINAMATH_CALUDE_circle_intersection_radius_range_l2435_243559


namespace NUMINAMATH_CALUDE_triangle_side_length_l2435_243570

theorem triangle_side_length (a b : ℝ) (C : ℝ) (S : ℝ) :
  a = 1 →
  C = π / 4 →
  S = 2 * a →
  S = 1 / 2 * a * b * Real.sin C →
  b = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2435_243570


namespace NUMINAMATH_CALUDE_triple_product_sum_two_l2435_243531

theorem triple_product_sum_two (x y z : ℝ) :
  (x * y + z = 2) ∧ (y * z + x = 2) ∧ (z * x + y = 2) →
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -2 ∧ y = -2 ∧ z = -2)) :=
by sorry

end NUMINAMATH_CALUDE_triple_product_sum_two_l2435_243531


namespace NUMINAMATH_CALUDE_isosceles_triangles_count_l2435_243571

/-- The number of ways to choose three vertices of a regular nonagon to form an isosceles triangle -/
def isosceles_triangles_in_nonagon : ℕ := 33

/-- A regular nonagon has 9 sides -/
def nonagon_sides : ℕ := 9

/-- The number of ways to choose 2 vertices from a nonagon -/
def choose_two_vertices : ℕ := (nonagon_sides * (nonagon_sides - 1)) / 2

/-- The number of equilateral triangles in a nonagon -/
def equilateral_triangles : ℕ := 3

theorem isosceles_triangles_count :
  isosceles_triangles_in_nonagon = choose_two_vertices - equilateral_triangles :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_count_l2435_243571


namespace NUMINAMATH_CALUDE_prove_initial_person_count_l2435_243567

/-- The number of persons initially in a group, given that:
    - The average weight increases by 2.5 kg when a new person replaces someone
    - The replaced person weighs 70 kg
    - The new person weighs 90 kg
-/
def initialPersonCount : ℕ := 8

theorem prove_initial_person_count :
  let averageWeightIncrease : ℚ := 2.5
  let replacedPersonWeight : ℕ := 70
  let newPersonWeight : ℕ := 90
  averageWeightIncrease * initialPersonCount = newPersonWeight - replacedPersonWeight :=
by sorry

#eval initialPersonCount

end NUMINAMATH_CALUDE_prove_initial_person_count_l2435_243567


namespace NUMINAMATH_CALUDE_paula_four_hops_l2435_243560

def hop_distance (goal : ℚ) (remaining : ℚ) : ℚ :=
  (1 / 4) * remaining

def remaining_distance (goal : ℚ) (hopped : ℚ) : ℚ :=
  goal - hopped

theorem paula_four_hops :
  let goal : ℚ := 2
  let hop1 := hop_distance goal goal
  let hop2 := hop_distance goal (remaining_distance goal hop1)
  let hop3 := hop_distance goal (remaining_distance goal (hop1 + hop2))
  let hop4 := hop_distance goal (remaining_distance goal (hop1 + hop2 + hop3))
  hop1 + hop2 + hop3 + hop4 = 175 / 128 := by
  sorry

end NUMINAMATH_CALUDE_paula_four_hops_l2435_243560


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l2435_243577

theorem arithmetic_simplification : 180 * (180 - 12) - (180 * 180 - 12) = -2148 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l2435_243577


namespace NUMINAMATH_CALUDE_linear_function_property_l2435_243556

/-- A linear function is a function f : ℝ → ℝ such that f(ax + by) = af(x) + bf(y) for all x, y ∈ ℝ and a, b ∈ ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x y a b : ℝ), f (a * x + b * y) = a * f x + b * f y

theorem linear_function_property (f : ℝ → ℝ) (h_linear : LinearFunction f) 
    (h_given : f 10 - f 4 = 20) : f 16 - f 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_property_l2435_243556


namespace NUMINAMATH_CALUDE_correct_calculation_l2435_243562

theorem correct_calculation (x : ℕ) (h : x + 12 = 48) : x + 22 = 58 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2435_243562


namespace NUMINAMATH_CALUDE_function_properties_l2435_243504

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_properties (f : ℝ → ℝ) 
    (h1 : ∀ x, f (10 + x) = f (10 - x))
    (h2 : ∀ x, f (20 - x) = -f (20 + x)) :
    is_odd f ∧ is_periodic f 40 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l2435_243504


namespace NUMINAMATH_CALUDE_fourth_power_sum_l2435_243558

theorem fourth_power_sum (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_squares_eq : a^2 + b^2 + c^2 = 5)
  (sum_cubes_eq : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l2435_243558


namespace NUMINAMATH_CALUDE_calculator_problem_l2435_243580

theorem calculator_problem (x : ℝ) (hx : x ≠ 0) :
  (1 / (1/x - 1)) - 1 = -0.75 → x = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_calculator_problem_l2435_243580


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2435_243585

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_k_value :
  ∀ k : ℝ, 
  let a : ℝ × ℝ := (1, k)
  let b : ℝ × ℝ := (9, k - 6)
  are_parallel a b → k = -3/4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2435_243585


namespace NUMINAMATH_CALUDE_area_fraction_to_CD_l2435_243564

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is parallel to CD and AB < CD
  AB : ℝ
  CD : ℝ
  h_parallel : AB < CD
  -- ∠BAD = 45° and ∠ABC = 135°
  angle_BAD : ℝ
  angle_ABC : ℝ
  h_angles : angle_BAD = π/4 ∧ angle_ABC = 3*π/4
  -- AD = BC = 100 m
  AD : ℝ
  BC : ℝ
  h_sides : AD = 100 ∧ BC = 100
  -- AB = 80 m
  h_AB : AB = 80
  -- CD > 100 m
  h_CD : CD > 100

/-- The fraction of the area closer to CD than to AB is approximately 3/4 -/
theorem area_fraction_to_CD (t : Trapezoid) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((t.CD - t.AB) * t.AD / (2 * (t.AB + t.CD) * t.AD)) - 3/4| < ε :=
sorry

end NUMINAMATH_CALUDE_area_fraction_to_CD_l2435_243564


namespace NUMINAMATH_CALUDE_piggy_bank_problem_l2435_243553

theorem piggy_bank_problem (total_cents : ℕ) (nickel_quarter_diff : ℕ) 
  (h1 : total_cents = 625)
  (h2 : nickel_quarter_diff = 9) : 
  ∃ (nickels quarters : ℕ),
    nickels = quarters + nickel_quarter_diff ∧
    5 * nickels + 25 * quarters = total_cents ∧
    nickels = 28 := by
  sorry

end NUMINAMATH_CALUDE_piggy_bank_problem_l2435_243553


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2435_243552

def A : Set ℝ := {-1, 1, 3}
def B (a : ℝ) : Set ℝ := {a + 1, a^2 + 4}

theorem intersection_implies_a_value :
  ∀ a : ℝ, A ∩ B a = {3} → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2435_243552


namespace NUMINAMATH_CALUDE_edward_sold_games_l2435_243528

theorem edward_sold_games (initial_games : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h1 : initial_games = 35)
  (h2 : boxes = 2)
  (h3 : games_per_box = 8) :
  initial_games - (boxes * games_per_box) = 19 := by
  sorry

end NUMINAMATH_CALUDE_edward_sold_games_l2435_243528


namespace NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_non_positive_monotonicity_positive_l2435_243588

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^2 - Real.log x - 2

-- Define the derivative of f(x)
def f_deriv (a : ℝ) (x : ℝ) : ℝ := a * x - 1/x

theorem tangent_line_at_one (x : ℝ) :
  f 1 1 = -(3/2) ∧ f_deriv 1 1 = 0 :=
sorry

theorem monotonicity_non_positive (a : ℝ) (x : ℝ) (ha : a ≤ 0) (hx : x > 0) :
  f_deriv a x < 0 :=
sorry

theorem monotonicity_positive (a : ℝ) (x : ℝ) (ha : a > 0) (hx : x > 0) :
  (x < Real.sqrt a / a → f_deriv a x < 0) ∧
  (x > Real.sqrt a / a → f_deriv a x > 0) :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_at_one_monotonicity_non_positive_monotonicity_positive_l2435_243588


namespace NUMINAMATH_CALUDE_minimize_sum_distances_on_x_axis_l2435_243583

/-- The point that minimizes the sum of distances to two given points -/
def minimize_sum_distances (A B : ℝ × ℝ) : ℝ × ℝ :=
  sorry

theorem minimize_sum_distances_on_x_axis 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ) 
  (h_A : A = (-1, 2)) 
  (h_B : B = (2, 1)) :
  minimize_sum_distances A B = (1, 0) :=
sorry

end NUMINAMATH_CALUDE_minimize_sum_distances_on_x_axis_l2435_243583


namespace NUMINAMATH_CALUDE_angle_330_equals_negative_30_l2435_243508

/-- Two angles have the same terminal side if they differ by a multiple of 360° --/
def same_terminal_side (α β : Real) : Prop :=
  ∃ k : Int, α = β + 360 * k

/-- The problem statement --/
theorem angle_330_equals_negative_30 :
  same_terminal_side 330 (-30) := by
  sorry

end NUMINAMATH_CALUDE_angle_330_equals_negative_30_l2435_243508


namespace NUMINAMATH_CALUDE_product_of_roots_l2435_243598

theorem product_of_roots (a b : ℝ) 
  (ha : a^2 - 4*a + 3 = 0) 
  (hb : b^2 - 4*b + 3 = 0) 
  (hab : a ≠ b) : 
  (a + 1) * (b + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2435_243598


namespace NUMINAMATH_CALUDE_function_root_iff_a_range_l2435_243539

/-- The function f(x) = 2ax - a + 3 has a root in (-1, 1) if and only if a ∈ (-∞, -3) ∪ (1, +∞) -/
theorem function_root_iff_a_range (a : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1) 1 ∧ 2 * a * x₀ - a + 3 = 0) ↔ 
  a ∈ Set.Iic (-3) ∪ Set.Ioi 1 :=
sorry

end NUMINAMATH_CALUDE_function_root_iff_a_range_l2435_243539


namespace NUMINAMATH_CALUDE_least_xy_value_l2435_243532

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) : 
  (∀ a b : ℕ+, (1 : ℚ) / a + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 → x * y ≤ a * b) ∧ x * y = 96 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l2435_243532


namespace NUMINAMATH_CALUDE_inverse_negation_implies_contrapositive_l2435_243587

-- Define propositions as boolean variables
variable (p q r : Prop)

-- Define the inverse relation
def is_inverse (a b : Prop) : Prop :=
  (a ↔ b) ∧ (¬a ↔ ¬b)

-- Define the negation relation
def is_negation (a b : Prop) : Prop :=
  a ↔ ¬b

-- Define the contrapositive relation
def is_contrapositive (a b : Prop) : Prop :=
  (a ↔ ¬b) ∧ (b ↔ ¬a)

-- State the theorem
theorem inverse_negation_implies_contrapositive
  (h1 : is_inverse p q)
  (h2 : is_negation q r) :
  is_contrapositive p r := by
sorry

end NUMINAMATH_CALUDE_inverse_negation_implies_contrapositive_l2435_243587


namespace NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l2435_243557

theorem arccos_one_half_equals_pi_third : Real.arccos (1/2) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_half_equals_pi_third_l2435_243557


namespace NUMINAMATH_CALUDE_greatest_q_minus_r_l2435_243529

theorem greatest_q_minus_r : ∃ (q r : ℕ), 
  q > 0 ∧ r > 0 ∧ 
  1013 = 23 * q + r ∧
  ∀ (q' r' : ℕ), q' > 0 → r' > 0 → 1013 = 23 * q' + r' → q' - r' ≤ q - r ∧
  q - r = 39 := by
sorry

end NUMINAMATH_CALUDE_greatest_q_minus_r_l2435_243529


namespace NUMINAMATH_CALUDE_train_crossing_time_l2435_243515

-- Define constants
def train_length : Real := 120
def train_speed_kmh : Real := 70
def bridge_length : Real := 150

-- Define the theorem
theorem train_crossing_time :
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let crossing_time := total_distance / train_speed_ms
  ∃ ε > 0, abs (crossing_time - 13.89) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2435_243515


namespace NUMINAMATH_CALUDE_S_value_l2435_243589

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_value : S = 2 * Real.sqrt 23 - 2 := by sorry

end NUMINAMATH_CALUDE_S_value_l2435_243589


namespace NUMINAMATH_CALUDE_charlies_garden_min_cost_l2435_243537

/-- Represents a rectangular region in the garden -/
structure Region where
  length : ℝ
  width : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.length * r.width

/-- Represents the cost of fertilizer per square meter for each vegetable type -/
structure FertilizerCost where
  lettuce : ℝ
  spinach : ℝ
  carrots : ℝ
  beans : ℝ
  tomatoes : ℝ

/-- The given garden layout -/
def garden_layout : List Region := [
  ⟨3, 1⟩,  -- Upper left
  ⟨4, 2⟩,  -- Lower right
  ⟨6, 2⟩,  -- Upper right
  ⟨2, 3⟩,  -- Middle center
  ⟨5, 4⟩   -- Bottom left
]

/-- The given fertilizer costs -/
def fertilizer_costs : FertilizerCost :=
  { lettuce := 2
  , spinach := 2.5
  , carrots := 3
  , beans := 3.5
  , tomatoes := 4
  }

/-- Calculates the minimum cost of fertilizers for the garden -/
def min_fertilizer_cost (layout : List Region) (costs : FertilizerCost) : ℝ :=
  sorry  -- Proof implementation goes here

/-- Theorem stating that the minimum fertilizer cost for Charlie's garden is $127 -/
theorem charlies_garden_min_cost :
  min_fertilizer_cost garden_layout fertilizer_costs = 127 := by
  sorry  -- Proof goes here

end NUMINAMATH_CALUDE_charlies_garden_min_cost_l2435_243537


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l2435_243514

/-- The set M of real numbers less than 3 -/
def M : Set ℝ := {x : ℝ | x < 3}

/-- The set N of real numbers less than 1 -/
def N : Set ℝ := {x : ℝ | x < 1}

/-- Theorem stating that the intersection of M and the complement of N in ℝ
    is equal to the set of real numbers x where 1 ≤ x < 3 -/
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l2435_243514


namespace NUMINAMATH_CALUDE_original_number_proof_l2435_243545

theorem original_number_proof (N : ℝ) (x : ℝ) : 
  (N * 1.2 = 480) → 
  (480 * 0.85 * x^2 = 5*x^3 + 24*x - 50) → 
  N = 400 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l2435_243545


namespace NUMINAMATH_CALUDE_chess_grandmaster_time_calculation_l2435_243576

theorem chess_grandmaster_time_calculation : 
  let time_learn_rules : ℕ := 2
  let time_get_proficient : ℕ := 49 * time_learn_rules
  let time_become_master : ℕ := 100 * (time_learn_rules + time_get_proficient)
  let total_time : ℕ := time_learn_rules + time_get_proficient + time_become_master
  total_time = 10100 := by
sorry

end NUMINAMATH_CALUDE_chess_grandmaster_time_calculation_l2435_243576


namespace NUMINAMATH_CALUDE_special_rectangle_exists_l2435_243509

/-- A rectangle with the given properties --/
structure SpecialRectangle where
  length : ℝ
  width : ℝ
  perimeter_equals_area : 2 * (length + width) = length * width
  width_is_length_minus_three : width = length - 3

/-- The theorem stating that a rectangle with length 6 and width 3 satisfies the conditions --/
theorem special_rectangle_exists : ∃ (r : SpecialRectangle), r.length = 6 ∧ r.width = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_exists_l2435_243509


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2435_243500

theorem min_value_quadratic (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + 2) :
  ∃ m : ℝ, (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2435_243500


namespace NUMINAMATH_CALUDE_concentric_circles_ratio_l2435_243566

/-- Given two concentric circles x and y, if the probability of a randomly selected point 
    inside circle x being outside circle y is 0.9722222222222222, then the ratio of the 
    radius of circle x to the radius of circle y is 6. -/
theorem concentric_circles_ratio (x y : Real) (h : x > y) 
    (prob : (x^2 - y^2) / x^2 = 0.9722222222222222) : x / y = 6 := by
  sorry


end NUMINAMATH_CALUDE_concentric_circles_ratio_l2435_243566


namespace NUMINAMATH_CALUDE_equation_solution_l2435_243538

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 3) 
  (h : 3 / a + 6 / b = 2 / 3) : a = 9 * b / (2 * b - 18) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2435_243538


namespace NUMINAMATH_CALUDE_dice_sum_product_l2435_243512

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 144 →
  a + b + c + d ≠ 18 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_product_l2435_243512


namespace NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l2435_243533

/-- Two lines in the form Ax + By + C = 0 are parallel if and only if their slopes are equal -/
def parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2

/-- Two lines in the form Ax + By + C = 0 are coincident if they have the same slope and y-intercept -/
def coincident (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  A1 / B1 = A2 / B2 ∧ C1 / B1 = C2 / B2

theorem parallel_lines_a_equals_two (a : ℝ) :
  parallel a (a + 2) 2 1 a (-2) ∧
  ¬coincident a (a + 2) 2 1 a (-2) →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_a_equals_two_l2435_243533


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2435_243591

theorem min_value_of_expression (x : ℝ) (h : x ≠ -7) :
  (2 * x^2 + 98) / (x + 7)^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2435_243591


namespace NUMINAMATH_CALUDE_total_bonus_calculation_l2435_243530

def senior_bonus : ℕ := 1900
def junior_bonus : ℕ := 3100

theorem total_bonus_calculation : senior_bonus + junior_bonus = 5000 := by
  sorry

end NUMINAMATH_CALUDE_total_bonus_calculation_l2435_243530


namespace NUMINAMATH_CALUDE_complex_unit_circle_representation_l2435_243578

theorem complex_unit_circle_representation (z : ℂ) (h1 : Complex.abs z = 1) (h2 : z ≠ -1) :
  ∃ t : ℝ, z = (1 + Complex.I * t) / (1 - Complex.I * t) := by
  sorry

end NUMINAMATH_CALUDE_complex_unit_circle_representation_l2435_243578


namespace NUMINAMATH_CALUDE_logarithm_sum_simplification_l2435_243572

theorem logarithm_sum_simplification :
  (1 / (Real.log 3 / Real.log 12 + 1) +
   1 / (Real.log 2 / Real.log 8 + 1) +
   1 / (Real.log 9 / Real.log 18 + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_sum_simplification_l2435_243572


namespace NUMINAMATH_CALUDE_james_weight_vest_cost_l2435_243554

/-- The cost of James's weight vest -/
def weight_vest_cost : ℝ := 250

/-- The cost of weight plates per pound -/
def weight_plate_cost_per_pound : ℝ := 1.2

/-- The weight of the plates in pounds -/
def weight_plate_pounds : ℝ := 200

/-- The original cost of a 200-pound weight vest -/
def original_vest_cost : ℝ := 700

/-- The discount on the 200-pound weight vest -/
def vest_discount : ℝ := 100

/-- The amount James saves with his vest -/
def james_savings : ℝ := 110

/-- Theorem: The cost of James's weight vest is $250 -/
theorem james_weight_vest_cost : 
  weight_vest_cost = 
    (original_vest_cost - vest_discount) - 
    (weight_plate_cost_per_pound * weight_plate_pounds) - 
    james_savings := by
  sorry

end NUMINAMATH_CALUDE_james_weight_vest_cost_l2435_243554


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_arithmetic_relation_l2435_243501

/-- A geometric sequence with first term a and common ratio q -/
def geometricSequence (a q : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => q * geometricSequence a q n

theorem geometric_sequence_ratio (a q : ℝ) (h : q ≠ 0) (h₁ : a ≠ 0) :
  ∀ n : ℕ, geometricSequence a q (n + 1) / geometricSequence a q n = q := by sorry

/-- An arithmetic sequence with first term a and common difference d -/
def arithmeticSequence (a d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmeticSequence a d n + d

theorem geometric_arithmetic_relation (a q : ℝ) (h : q ≠ 0) (h₁ : a ≠ 0) :
  (∃ d : ℝ, arithmeticSequence 1 d 0 = 1 ∧
            arithmeticSequence 1 d 1 = geometricSequence a q 1 ∧
            arithmeticSequence 1 d 2 = geometricSequence a q 2 - 1) →
  (geometricSequence a q 2 + geometricSequence a q 3) / (geometricSequence a q 4 + geometricSequence a q 5) = 1/4 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_geometric_arithmetic_relation_l2435_243501


namespace NUMINAMATH_CALUDE_base8_addition_subtraction_l2435_243551

/-- Converts a base-8 number represented as a list of digits to its decimal (base-10) equivalent -/
def base8ToDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 8 + d) 0

/-- Converts a decimal (base-10) number to its base-8 representation as a list of digits -/
def decimalToBase8 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- The main theorem to prove -/
theorem base8_addition_subtraction :
  decimalToBase8 ((base8ToDecimal [1, 7, 6] + base8ToDecimal [4, 5]) - base8ToDecimal [6, 3]) = [1, 5, 1] := by
  sorry

end NUMINAMATH_CALUDE_base8_addition_subtraction_l2435_243551


namespace NUMINAMATH_CALUDE_chord_length_l2435_243526

/-- The length of the chord cut by a circle on a line --/
theorem chord_length (r : ℝ) (a b c : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = r^2}
  let line := {(x, y) : ℝ × ℝ | ∃ t, x = a - b*t ∧ y = c + b*t}
  let chord := circle ∩ line
  r = 2 ∧ a = 2 ∧ b = 1/2 ∧ c = -1 →
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 14 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_l2435_243526


namespace NUMINAMATH_CALUDE_two_year_increase_l2435_243534

/-- Calculates the final amount after a given number of years with a fixed annual increase rate -/
def finalAmount (initialAmount : ℝ) (annualRate : ℝ) (years : ℕ) : ℝ :=
  initialAmount * (1 + annualRate) ^ years

/-- Theorem: An initial amount of 57600, increasing by 1/8 annually, becomes 72900 after 2 years -/
theorem two_year_increase : 
  finalAmount 57600 (1/8) 2 = 72900 := by sorry

end NUMINAMATH_CALUDE_two_year_increase_l2435_243534


namespace NUMINAMATH_CALUDE_wallpaper_overlap_theorem_l2435_243535

/-- The combined area of three walls with overlapping wallpaper -/
def combined_area (two_layer_area : ℝ) (three_layer_area : ℝ) (total_covered_area : ℝ) : ℝ :=
  total_covered_area + two_layer_area + 2 * three_layer_area

/-- Theorem stating the combined area of three walls with given overlapping conditions -/
theorem wallpaper_overlap_theorem (two_layer_area : ℝ) (three_layer_area : ℝ) (total_covered_area : ℝ)
    (h1 : two_layer_area = 40)
    (h2 : three_layer_area = 40)
    (h3 : total_covered_area = 180) :
    combined_area two_layer_area three_layer_area total_covered_area = 300 := by
  sorry

end NUMINAMATH_CALUDE_wallpaper_overlap_theorem_l2435_243535


namespace NUMINAMATH_CALUDE_matrix_power_four_l2435_243518

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -2; 1, 1]

theorem matrix_power_four :
  A ^ 4 = !![(-14), -6; 3, (-17)] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l2435_243518


namespace NUMINAMATH_CALUDE_condition_relationship_l2435_243527

theorem condition_relationship (a b : ℝ) : 
  (∀ a b, a > 0 ∧ b > 0 → a * b > 0) ∧  -- A is necessary for B
  (∃ a b, a * b > 0 ∧ ¬(a > 0 ∧ b > 0)) -- A is not sufficient for B
  := by sorry

end NUMINAMATH_CALUDE_condition_relationship_l2435_243527


namespace NUMINAMATH_CALUDE_closed_set_properties_l2435_243541

-- Define a closed set
def is_closed_set (M : Set Int) : Prop :=
  ∀ a b : Int, a ∈ M ∧ b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-4, -2, 0, 2, 4}
def M : Set Int := {-4, -2, 0, 2, 4}

-- Define the set of positive integers
def positive_integers : Set Int := {n : Int | n > 0}

-- Define the set M = {n | n = 3k, k ∈ Z}
def M_3k : Set Int := {n : Int | ∃ k : Int, n = 3 * k}

theorem closed_set_properties :
  (¬ is_closed_set M) ∧
  (¬ is_closed_set positive_integers) ∧
  (is_closed_set M_3k) ∧
  (∃ A₁ A₂ : Set Int, is_closed_set A₁ ∧ is_closed_set A₂ ∧ ¬ is_closed_set (A₁ ∪ A₂)) :=
sorry

end NUMINAMATH_CALUDE_closed_set_properties_l2435_243541


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2435_243502

theorem sphere_surface_area (R : ℝ) : 
  R > 0 → 
  (∃ (x : ℝ), x > 0 ∧ x < R ∧ 
    (∀ (y : ℝ), y > 0 → y < R → 
      2 * π * x^2 * (2 * Real.sqrt (R^2 - x^2)) ≥ 2 * π * y^2 * (2 * Real.sqrt (R^2 - y^2)))) →
  2 * π * R * (2 * Real.sqrt (R^2 - (R * Real.sqrt 6 / 3)^2)) = 16 * Real.sqrt 2 * π →
  4 * π * R^2 = 48 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2435_243502


namespace NUMINAMATH_CALUDE_prob_A_third_try_prob_at_least_one_success_l2435_243590

/-- Probability of 甲 solving the cube within 30 seconds -/
def prob_A : ℝ := 0.8

/-- Probability of 乙 solving the cube within 30 seconds -/
def prob_B : ℝ := 0.6

/-- Each attempt is independent -/
axiom attempts_independent : True

/-- Probability of 甲 succeeding on the third try -/
theorem prob_A_third_try : 
  (1 - prob_A) * (1 - prob_A) * prob_A = 0.032 := by sorry

/-- Probability of at least one person succeeding on the first try -/
theorem prob_at_least_one_success : 
  1 - (1 - prob_A) * (1 - prob_B) = 0.92 := by sorry

end NUMINAMATH_CALUDE_prob_A_third_try_prob_at_least_one_success_l2435_243590


namespace NUMINAMATH_CALUDE_cube_edge_length_l2435_243548

theorem cube_edge_length (V : ℝ) (s : ℝ) (h : V = 7) (h1 : V = s^3) :
  s = (7 : ℝ)^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l2435_243548


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2435_243594

-- Define the basic types
variable (P : Type) -- Type for points
variable (α : Set P) -- Type for planes
variable (l m : Set P) -- Type for lines

-- Define the geometric relations
variable (perpendicular : Set P → Set P → Prop) -- Perpendicular relation for lines and planes
variable (parallel : Set P → Set P → Prop) -- Parallel relation for lines and planes
variable (subset : Set P → Set P → Prop) -- Subset relation for lines and planes

-- State the theorem
theorem perpendicular_necessary_not_sufficient
  (h : perpendicular m α) :
  (∀ l, parallel l α → perpendicular l m) ∧
  ¬(∀ l, perpendicular l m → parallel l α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l2435_243594


namespace NUMINAMATH_CALUDE_kennedy_house_size_l2435_243523

theorem kennedy_house_size (benedict_house_size : ℕ) (kennedy_house_size : ℕ) : 
  benedict_house_size = 2350 →
  kennedy_house_size = 4 * benedict_house_size + 600 →
  kennedy_house_size = 10000 := by
sorry

end NUMINAMATH_CALUDE_kennedy_house_size_l2435_243523


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2435_243595

-- Define the function f
def f (x : ℝ) : ℝ := x^6 + x^2 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 42 := by sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l2435_243595


namespace NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l2435_243586

/-- Conversion factor from meters to nanometers -/
def meters_to_nanometers : ℝ := 10^9

/-- The diameter of the bacterium in meters -/
def bacterium_diameter_meters : ℝ := 0.00000285

/-- Theorem stating that the diameter of the bacterium in nanometers is 2850 -/
theorem bacterium_diameter_nanometers :
  bacterium_diameter_meters * meters_to_nanometers = 2850 := by
  sorry

#eval bacterium_diameter_meters * meters_to_nanometers

end NUMINAMATH_CALUDE_bacterium_diameter_nanometers_l2435_243586


namespace NUMINAMATH_CALUDE_function_equation_implies_identity_l2435_243519

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equation_implies_identity_l2435_243519


namespace NUMINAMATH_CALUDE_expression_evaluation_l2435_243513

theorem expression_evaluation : (20 * 3 + 10) / (5 + 3) = 8.75 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2435_243513


namespace NUMINAMATH_CALUDE_street_lights_per_side_l2435_243536

/-- The number of neighborhoods in the town -/
def num_neighborhoods : ℕ := 10

/-- The number of roads in each neighborhood -/
def roads_per_neighborhood : ℕ := 4

/-- The total number of street lights in the town -/
def total_street_lights : ℕ := 20000

/-- The number of street lights on each opposite side of a road -/
def lights_per_side : ℚ := total_street_lights / (2 * num_neighborhoods * roads_per_neighborhood)

theorem street_lights_per_side :
  lights_per_side = 250 :=
sorry

end NUMINAMATH_CALUDE_street_lights_per_side_l2435_243536


namespace NUMINAMATH_CALUDE_tims_movie_marathon_duration_l2435_243565

/-- The duration of Tim's movie marathon --/
def movie_marathon_duration (first_movie : ℝ) (second_movie_factor : ℝ) (third_movie_offset : ℝ) : ℝ :=
  let second_movie := first_movie * (1 + second_movie_factor)
  let third_movie := first_movie + second_movie - third_movie_offset
  first_movie + second_movie + third_movie

/-- Theorem stating the duration of Tim's specific movie marathon --/
theorem tims_movie_marathon_duration :
  movie_marathon_duration 2 0.5 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_tims_movie_marathon_duration_l2435_243565


namespace NUMINAMATH_CALUDE_certain_amount_proof_l2435_243596

theorem certain_amount_proof (x : ℝ) (A : ℝ) (h1 : x = 840) (h2 : 0.25 * x = 0.15 * 1500 - A) : A = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l2435_243596


namespace NUMINAMATH_CALUDE_sum_of_values_l2435_243575

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  prob₁ : ℝ
  prob₂ : ℝ
  h₁ : x₁ < x₂
  h₂ : prob₁ = (1 : ℝ) / 2
  h₃ : prob₂ = (1 : ℝ) / 2
  h₄ : prob₁ + prob₂ = 1

/-- Expected value of the discrete random variable -/
def expectation (X : DiscreteRV) : ℝ :=
  X.x₁ * X.prob₁ + X.x₂ * X.prob₂

/-- Variance of the discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  (X.x₁ - expectation X)^2 * X.prob₁ + (X.x₂ - expectation X)^2 * X.prob₂

theorem sum_of_values (X : DiscreteRV) 
    (h_exp : expectation X = 2) 
    (h_var : variance X = (1 : ℝ) / 2) : 
  X.x₁ + X.x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_values_l2435_243575


namespace NUMINAMATH_CALUDE_circle_M_equation_l2435_243505

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

-- Define the axis of symmetry of the parabola
def axis_of_symmetry (y : ℝ) : Prop := y = -1

-- Define the circle ⊙M
def circle_M (x y t : ℝ) : Prop := (x - t)^2 + (y - t^2/4)^2 = t^2

-- Define the tangency condition to y-axis
def tangent_to_y_axis (t : ℝ) : Prop := t = 2 ∨ t = -2

-- Define the tangency condition to axis of symmetry
def tangent_to_axis_of_symmetry (t : ℝ) : Prop := |1 + t^2/4| = |t|

-- Theorem statement
theorem circle_M_equation (x y t : ℝ) :
  parabola x y →
  axis_of_symmetry (-1) →
  circle_M x y t →
  tangent_to_y_axis t →
  tangent_to_axis_of_symmetry t →
  ∃ (sign : ℝ), sign = 1 ∨ sign = -1 ∧ x^2 + y^2 + sign*4*x - 2*y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l2435_243505


namespace NUMINAMATH_CALUDE_rectangle_on_circle_l2435_243568

theorem rectangle_on_circle (R : ℝ) (x y : ℝ) :
  x^2 + y^2 = R^2 →
  x * y = (12 * R / 35) * (x + y) →
  ((x = 3 * R / 5 ∧ y = 4 * R / 5) ∨ (x = 4 * R / 5 ∧ y = 3 * R / 5)) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_on_circle_l2435_243568


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_711_1033_l2435_243549

theorem smallest_multiple_of_5_711_1033 :
  ∃ (n : ℕ), n > 0 ∧ 
  5 ∣ n ∧ 711 ∣ n ∧ 1033 ∣ n ∧ 
  (∀ m : ℕ, m > 0 → 5 ∣ m → 711 ∣ m → 1033 ∣ m → n ≤ m) ∧
  n = 3683445 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_711_1033_l2435_243549


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l2435_243547

theorem square_circle_area_ratio (s : ℝ) (h : s > 0) :
  let r : ℝ := s / 2
  let square_area : ℝ := s ^ 2
  let circle_area : ℝ := π * r ^ 2
  square_area / circle_area = 4 / π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l2435_243547


namespace NUMINAMATH_CALUDE_plane_equation_correct_l2435_243579

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space defined by parametric equations -/
structure Line3D where
  x : ℝ → ℝ
  y : ℝ → ℝ
  z : ℝ → ℝ

/-- A plane in 3D space defined by the equation Ax + By + Cz + D = 0 -/
structure Plane where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointOnPlane (plane : Plane) (point : Point3D) : Prop :=
  plane.A * point.x + plane.B * point.y + plane.C * point.z + plane.D = 0

/-- Check if a line is contained in a plane -/
def lineInPlane (plane : Plane) (line : Line3D) : Prop :=
  ∀ t, pointOnPlane plane ⟨line.x t, line.y t, line.z t⟩

/-- The given point that the plane passes through -/
def givenPoint : Point3D :=
  ⟨1, 4, -5⟩

/-- The given line that the plane contains -/
def givenLine : Line3D :=
  ⟨λ t => 4 * t + 2, λ t => -t + 1, λ t => 5 * t - 3⟩

/-- The plane we want to prove -/
def solutionPlane : Plane :=
  ⟨2, 7, 6, -66⟩

theorem plane_equation_correct :
  pointOnPlane solutionPlane givenPoint ∧
  lineInPlane solutionPlane givenLine ∧
  solutionPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Int.natAbs solutionPlane.A) (Int.natAbs solutionPlane.B))
          (Nat.gcd (Int.natAbs solutionPlane.C) (Int.natAbs solutionPlane.D)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l2435_243579


namespace NUMINAMATH_CALUDE_team_win_percentage_l2435_243582

theorem team_win_percentage (total_games : ℕ) (wins_first_100 : ℕ) 
  (h1 : total_games ≥ 100)
  (h2 : wins_first_100 ≤ 100)
  (h3 : (wins_first_100 : ℝ) / 100 + (0.5 * (total_games - 100) : ℝ) / total_games = 0.7) :
  wins_first_100 = 70 := by
sorry

end NUMINAMATH_CALUDE_team_win_percentage_l2435_243582


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l2435_243503

/-- Calculates the annual rent per square foot for a shop given its dimensions and monthly rent -/
theorem annual_rent_per_square_foot
  (length : ℝ) (width : ℝ) (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 20)
  (h3 : monthly_rent = 3600) :
  monthly_rent * 12 / (length * width) = 120 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l2435_243503


namespace NUMINAMATH_CALUDE_james_oranges_l2435_243569

theorem james_oranges (pieces_per_orange : ℕ) (num_people : ℕ) (calories_per_orange : ℕ) (calories_per_person : ℕ) :
  pieces_per_orange = 8 →
  num_people = 4 →
  calories_per_orange = 80 →
  calories_per_person = 100 →
  (calories_per_person * num_people) / calories_per_orange * pieces_per_orange / pieces_per_orange = 5 :=
by sorry

end NUMINAMATH_CALUDE_james_oranges_l2435_243569


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2435_243574

/-- Proves that a 25% reduction in oil price allows purchasing 5 kg more oil for Rs. 1100 --/
theorem oil_price_reduction (original_price : ℝ) : 
  (original_price * 0.75 = 55) →  -- Reduced price is 55
  (1100 / 55 - 1100 / original_price = 5) := by
sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2435_243574


namespace NUMINAMATH_CALUDE_shooting_probabilities_l2435_243573

/-- Let A and B be two individuals conducting 3 shooting trials each.
    The probability of A hitting the target in each trial is 1/2.
    The probability of B hitting the target in each trial is 2/3. -/
theorem shooting_probabilities 
  (probability_A : ℝ) 
  (probability_B : ℝ) 
  (h_prob_A : probability_A = 1/2) 
  (h_prob_B : probability_B = 2/3) :
  /- The probability that A hits the target exactly 2 times -/
  (3 : ℝ) * probability_A^2 * (1 - probability_A) = 3/8 ∧ 
  /- The probability that B hits the target at least 2 times -/
  (3 : ℝ) * probability_B^2 * (1 - probability_B) + probability_B^3 = 20/27 ∧ 
  /- The probability that B hits the target exactly 2 more times than A -/
  (3 : ℝ) * probability_B^2 * (1 - probability_B) * (1 - probability_A)^3 + 
  probability_B^3 * (3 : ℝ) * probability_A * (1 - probability_A)^2 = 1/6 :=
by sorry


end NUMINAMATH_CALUDE_shooting_probabilities_l2435_243573


namespace NUMINAMATH_CALUDE_blue_button_probability_l2435_243517

/-- Represents a jar containing buttons of different colors. -/
structure Jar where
  red : ℕ
  blue : ℕ

/-- The probability of selecting a blue button from a jar. -/
def blueProb (j : Jar) : ℚ :=
  j.blue / (j.red + j.blue)

/-- The initial state of Jar C. -/
def jarC : Jar := { red := 6, blue := 10 }

/-- The number of buttons transferred from Jar C to Jar D. -/
def transferred : ℕ := 4

/-- Jar C after the transfer. -/
def jarCAfter : Jar := { red := jarC.red - transferred / 2, blue := jarC.blue - transferred / 2 }

/-- Jar D after the transfer. -/
def jarD : Jar := { red := transferred / 2, blue := transferred / 2 }

/-- Theorem stating the probability of selecting blue buttons from both jars. -/
theorem blue_button_probability : 
  blueProb jarCAfter * blueProb jarD = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_blue_button_probability_l2435_243517


namespace NUMINAMATH_CALUDE_sum_of_roots_l2435_243525

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - a^2 + a - 5 = 0)
  (hb : b^3 - 2*b^2 + 2*b + 4 = 0) : 
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2435_243525


namespace NUMINAMATH_CALUDE_article_price_fraction_l2435_243516

/-- Proves that selling an article at 2/3 of its original price results in a 10% loss,
    given that the original price has a 35% markup from the cost price. -/
theorem article_price_fraction (original_price cost_price : ℝ) :
  original_price = cost_price * (1 + 35 / 100) →
  original_price * (2 / 3) = cost_price * (1 - 10 / 100) := by
  sorry

end NUMINAMATH_CALUDE_article_price_fraction_l2435_243516


namespace NUMINAMATH_CALUDE_range_of_m_l2435_243584

def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*(m+1)*x + m*(m+1) > 0

theorem range_of_m (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m > 2 ∨ (-2 ≤ m ∧ m < -1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2435_243584


namespace NUMINAMATH_CALUDE_polynomial_root_properties_l2435_243522

def P (x p : ℂ) : ℂ := x^4 + 3*x^3 + 3*x + p

theorem polynomial_root_properties (p : ℝ) (x₁ : ℂ) 
  (h1 : Complex.abs x₁ = 1)
  (h2 : 2 * Complex.re x₁ = (Real.sqrt 17 - 3) / 2)
  (h3 : P x₁ p = 0) :
  p = -1 - 3 * x₁^3 - 3 * x₁ ∧
  x₁ = Complex.mk ((Real.sqrt 17 - 3) / 4) (Real.sqrt ((3 * Real.sqrt 17 - 5) / 8)) ∧
  ∀ n : ℕ+, x₁^(n : ℕ) ≠ 1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_properties_l2435_243522


namespace NUMINAMATH_CALUDE_max_fraction_value_l2435_243521

def is_odd_integer (y : ℝ) : Prop := ∃ (k : ℤ), y = 2 * k + 1

theorem max_fraction_value (x y : ℝ) 
  (hx : -5 ≤ x ∧ x ≤ -3) 
  (hy : 3 ≤ y ∧ y ≤ 5) 
  (hy_odd : is_odd_integer y) : 
  (∀ z, -5 ≤ z ∧ z ≤ -3 → ∀ w, 3 ≤ w ∧ w ≤ 5 → is_odd_integer w → (x + y) / x ≥ (z + w) / z) ∧ 
  (x + y) / x ≤ 0.4 := by
  sorry

end NUMINAMATH_CALUDE_max_fraction_value_l2435_243521


namespace NUMINAMATH_CALUDE_length_of_A_l2435_243546

-- Define the points
def A : ℝ × ℝ := (0, 4)
def B : ℝ × ℝ := (0, 14)
def C : ℝ × ℝ := (3, 6)

-- Define the line y = x
def line_y_eq_x (p : ℝ × ℝ) : Prop := p.2 = p.1

-- Define the intersection of line segments
def intersect (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ (1 - t) • p + t • r = q

-- Main theorem
theorem length_of_A'B' (A' B' : ℝ × ℝ) :
  line_y_eq_x A' →
  line_y_eq_x B' →
  intersect A A' C →
  intersect B B' C →
  Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 90 * Real.sqrt 2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_length_of_A_l2435_243546
