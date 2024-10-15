import Mathlib

namespace NUMINAMATH_CALUDE_edge_enlargement_equals_graph_scale_l719_71983

/-- A graph is represented by a set of edges, where each edge has a length. -/
structure Graph where
  edges : Set (ℝ)

/-- Enlarging a graph by multiplying each edge length by a factor. -/
def enlarge (g : Graph) (factor : ℝ) : Graph :=
  { edges := g.edges.image (· * factor) }

/-- The scale factor of a transformation that multiplies each edge by 4. -/
def scale_factor : ℝ := 4

theorem edge_enlargement_equals_graph_scale (g : Graph) :
  enlarge g scale_factor = enlarge g scale_factor :=
by
  sorry

end NUMINAMATH_CALUDE_edge_enlargement_equals_graph_scale_l719_71983


namespace NUMINAMATH_CALUDE_abc_sum_l719_71994

theorem abc_sum (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →  -- A, B, C are single digits
  A ≠ B → B ≠ C → A ≠ C →     -- A, B, C are different
  (100 * A + 10 * B + C) * 4 = 1436 →  -- ABC + ABC + ABC + ABC = 1436
  A + B + C = 17 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l719_71994


namespace NUMINAMATH_CALUDE_most_likely_outcome_l719_71958

-- Define the number of children
def n : ℕ := 5

-- Define the probability of a child being a boy or a girl
def p : ℚ := 1/2

-- Define the probability of all children being the same gender
def prob_all_same : ℚ := 2 * p^n

-- Define the probability of having 2 of one gender and 3 of the other
def prob_2_3 : ℚ := (n.choose 2) * p^n

-- Define the probability of having 4 of one gender and 1 of the other
def prob_4_1 : ℚ := 2 * (n.choose 1) * p^n

-- Theorem statement
theorem most_likely_outcome :
  prob_2_3 + prob_4_1 > prob_all_same :=
by sorry

end NUMINAMATH_CALUDE_most_likely_outcome_l719_71958


namespace NUMINAMATH_CALUDE_dave_apps_left_l719_71966

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 4

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := files_left + 17

/-- Theorem: Dave has 21 apps left on his phone -/
theorem dave_apps_left : apps_left = 21 := by
  sorry

end NUMINAMATH_CALUDE_dave_apps_left_l719_71966


namespace NUMINAMATH_CALUDE_truck_distance_on_rough_terrain_truck_travel_distance_l719_71904

/-- Calculates the distance a truck can travel on rough terrain given its performance on a smooth highway and the efficiency decrease on rough terrain. -/
theorem truck_distance_on_rough_terrain 
  (highway_distance : ℝ) 
  (highway_gas : ℝ) 
  (rough_terrain_efficiency_decrease : ℝ) 
  (rough_terrain_gas : ℝ) : ℝ :=
  let highway_efficiency := highway_distance / highway_gas
  let rough_terrain_efficiency := highway_efficiency * (1 - rough_terrain_efficiency_decrease)
  rough_terrain_efficiency * rough_terrain_gas

/-- Proves that a truck traveling 300 miles on 10 gallons of gas on a smooth highway can travel 405 miles on 15 gallons of gas on rough terrain with a 10% efficiency decrease. -/
theorem truck_travel_distance : 
  truck_distance_on_rough_terrain 300 10 0.1 15 = 405 := by
  sorry

end NUMINAMATH_CALUDE_truck_distance_on_rough_terrain_truck_travel_distance_l719_71904


namespace NUMINAMATH_CALUDE_fifty_second_digit_of_1_17_l719_71944

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : ℚ := 1 / 17

-- Define the length of the repeating sequence
def repeat_length : ℕ := 16

-- Define the position we're interested in
def target_position : ℕ := 52

-- Define the function to get the nth digit after the decimal point
noncomputable def nth_digit (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem fifty_second_digit_of_1_17 : 
  nth_digit target_position = 8 := by sorry

end NUMINAMATH_CALUDE_fifty_second_digit_of_1_17_l719_71944


namespace NUMINAMATH_CALUDE_jim_travels_two_miles_l719_71962

def john_distance : ℝ := 15

def jill_distance (john_dist : ℝ) : ℝ := john_dist - 5

def jim_distance (jill_dist : ℝ) : ℝ := 0.20 * jill_dist

theorem jim_travels_two_miles :
  jim_distance (jill_distance john_distance) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_travels_two_miles_l719_71962


namespace NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l719_71957

/-- The number of students who suggested mashed potatoes -/
def mashed_potatoes : ℕ := 144

/-- The number of students who suggested bacon -/
def bacon : ℕ := 467

/-- The number of students who suggested tomatoes -/
def tomatoes : ℕ := 79

/-- The theorem stating the difference between the number of students who suggested
    mashed potatoes and those who suggested tomatoes -/
theorem mashed_potatoes_tomatoes_difference :
  mashed_potatoes - tomatoes = 65 := by sorry

end NUMINAMATH_CALUDE_mashed_potatoes_tomatoes_difference_l719_71957


namespace NUMINAMATH_CALUDE_fair_haired_women_percentage_l719_71991

/-- Given that 30% of employees are women with fair hair and 75% of employees have fair hair,
    prove that 40% of fair-haired employees are women. -/
theorem fair_haired_women_percentage
  (total_employees : ℝ)
  (women_fair_hair_percentage : ℝ)
  (fair_hair_percentage : ℝ)
  (h1 : women_fair_hair_percentage = 30 / 100)
  (h2 : fair_hair_percentage = 75 / 100) :
  (women_fair_hair_percentage * total_employees) / (fair_hair_percentage * total_employees) = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_fair_haired_women_percentage_l719_71991


namespace NUMINAMATH_CALUDE_functional_equation_solution_l719_71902

/-- The functional equation for f and g -/
def FunctionalEquation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + g y) = x * f y - y * f x + g x

/-- The solution forms for f and g -/
def SolutionForms (f g : ℝ → ℝ) : Prop :=
  ∃ t : ℝ, t ≠ -1 ∧
    (∀ x : ℝ, f x = (t * (x - t)) / (t + 1)) ∧
    (∀ x : ℝ, g x = t * (x - t))

theorem functional_equation_solution :
    ∀ f g : ℝ → ℝ, FunctionalEquation f g → SolutionForms f g :=
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l719_71902


namespace NUMINAMATH_CALUDE_expression_value_l719_71925

theorem expression_value (x y : ℝ) (hx : x = 3) (hy : y = 2) :
  3 * x^2 - 4 * y + 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l719_71925


namespace NUMINAMATH_CALUDE_p_range_q_range_p_or_q_range_l719_71980

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m - 3 > 0
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*m*x + m + 2 < 0

-- Theorem for the range of m when p is true
theorem p_range (m : ℝ) : p m ↔ m > 3/2 := by sorry

-- Theorem for the range of m when q is true
theorem q_range (m : ℝ) : q m ↔ m < -1 ∨ m > 2 := by sorry

-- Theorem for the range of m when at least one of p or q is true
theorem p_or_q_range (m : ℝ) : p m ∨ q m ↔ m < -1 ∨ m > 3/2 := by sorry

end NUMINAMATH_CALUDE_p_range_q_range_p_or_q_range_l719_71980


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l719_71989

theorem sqrt_product_plus_one : 
  Real.sqrt ((26 : ℝ) * 25 * 24 * 23 + 1) = 599 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l719_71989


namespace NUMINAMATH_CALUDE_polynomial_simplification_l719_71901

theorem polynomial_simplification (x : ℝ) : 
  x * (4 * x^2 - 2) - 5 * (x^2 - 3*x + 5) = 4 * x^3 - 5 * x^2 + 13 * x - 25 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l719_71901


namespace NUMINAMATH_CALUDE_problem_solution_l719_71982

def A : Set ℤ := {-2, 3, 4, 6}
def B (a : ℤ) : Set ℤ := {3, a, a^2}

theorem problem_solution (a : ℤ) : 
  (B a ⊆ A → a = 2) ∧ 
  (A ∩ B a = {3, 4} → a = 2 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l719_71982


namespace NUMINAMATH_CALUDE_total_shark_teeth_l719_71961

def tiger_shark_teeth : ℕ := 180

def hammerhead_shark_teeth : ℕ := tiger_shark_teeth / 6

def great_white_shark_teeth : ℕ := 2 * (tiger_shark_teeth + hammerhead_shark_teeth)

def mako_shark_teeth : ℕ := (5 * hammerhead_shark_teeth) / 3

theorem total_shark_teeth : 
  tiger_shark_teeth + hammerhead_shark_teeth + great_white_shark_teeth + mako_shark_teeth = 680 := by
  sorry

end NUMINAMATH_CALUDE_total_shark_teeth_l719_71961


namespace NUMINAMATH_CALUDE_luke_trivia_rounds_l719_71910

/-- Given that Luke gained 46 points per round and scored 8142 points in total,
    prove that he played 177 rounds. -/
theorem luke_trivia_rounds (points_per_round : ℕ) (total_points : ℕ) 
    (h1 : points_per_round = 46) 
    (h2 : total_points = 8142) : 
  total_points / points_per_round = 177 := by
  sorry

end NUMINAMATH_CALUDE_luke_trivia_rounds_l719_71910


namespace NUMINAMATH_CALUDE_no_positive_integral_solutions_l719_71976

theorem no_positive_integral_solutions : 
  ¬ ∃ (x y : ℕ+), x.val^6 * y.val^6 - 13 * x.val^3 * y.val^3 + 36 = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_positive_integral_solutions_l719_71976


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l719_71911

theorem cubic_equation_solution :
  ∃! x : ℝ, x^3 + 12*x = 6*x^2 + 35 :=
by
  -- The unique solution is x = 5
  use 5
  constructor
  · -- Prove that x = 5 satisfies the equation
    simp
    -- Additional steps to prove 5^3 + 12*5 = 6*5^2 + 35
    sorry
  · -- Prove that any solution must equal 5
    intro y hy
    -- Steps to show that if y satisfies the equation, then y = 5
    sorry


end NUMINAMATH_CALUDE_cubic_equation_solution_l719_71911


namespace NUMINAMATH_CALUDE_value_of_x_l719_71986

theorem value_of_x :
  ∀ (x y z w u : ℤ),
    x = y + 3 →
    y = z + 15 →
    z = w + 25 →
    w = u + 10 →
    u = 90 →
    x = 143 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l719_71986


namespace NUMINAMATH_CALUDE_oranges_per_box_l719_71906

/-- Given a fruit farm that packs 2650 oranges into 265 boxes,
    prove that each box contains 10 oranges. -/
theorem oranges_per_box :
  let total_oranges : ℕ := 2650
  let total_boxes : ℕ := 265
  total_oranges / total_boxes = 10 := by
sorry

end NUMINAMATH_CALUDE_oranges_per_box_l719_71906


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l719_71935

theorem quadruple_equation_solutions :
  let equation (a b c d : ℕ) := 2*a + 2*b + 2*c + 2*d = d^2 - c^2 + b^2 - a^2
  ∀ (a b c d : ℕ), a < b → b < c → c < d →
  (
    (equation 2 4 5 7) ∧
    (∀ x : ℕ, equation (2*x) (2*x+2) (2*x+4) (2*x+6))
  ) := by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l719_71935


namespace NUMINAMATH_CALUDE_total_earnings_l719_71934

/-- The total earnings of Salvadore and Santo, given Salvadore's earnings and that Santo earned half of Salvadore's earnings -/
theorem total_earnings (salvadore_earnings : ℕ) (h : salvadore_earnings = 1956) :
  salvadore_earnings + (salvadore_earnings / 2) = 2934 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_l719_71934


namespace NUMINAMATH_CALUDE_smallest_multiple_with_conditions_l719_71945

theorem smallest_multiple_with_conditions : ∃! n : ℕ, 
  n > 0 ∧ 
  47 ∣ n ∧ 
  n % 97 = 7 ∧ 
  n % 31 = 28 ∧ 
  ∀ m : ℕ, m > 0 → 47 ∣ m → m % 97 = 7 → m % 31 = 28 → n ≤ m :=
by
  use 79618
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_conditions_l719_71945


namespace NUMINAMATH_CALUDE_ages_sum_l719_71990

/-- Given the ages of Al, Bob, and Carl satisfying certain conditions, prove their sum is 80 -/
theorem ages_sum (a b c : ℕ) : 
  a = b + c + 20 → 
  a^2 = (b + c)^2 + 2000 → 
  a + b + c = 80 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l719_71990


namespace NUMINAMATH_CALUDE_smallest_area_of_P_l719_71939

/-- Represents a point on the grid --/
structure GridPoint where
  x : Nat
  y : Nat
  label : Nat
  deriving Repr

/-- Defines the properties of the grid --/
def grid : List GridPoint := sorry

/-- Checks if a label is divisible by 7 --/
def isDivisibleBySeven (n : Nat) : Bool :=
  n % 7 == 0

/-- Defines the convex polygon P --/
def P : Set GridPoint := sorry

/-- Calculates the area of a convex polygon --/
noncomputable def areaOfConvexPolygon (polygon : Set GridPoint) : Real := sorry

/-- States that P contains all points with labels divisible by 7 --/
axiom P_contains_divisible_by_seven :
  ∀ p : GridPoint, p ∈ grid → isDivisibleBySeven p.label → p ∈ P

/-- Theorem: The smallest possible area of P is 60.5 square units --/
theorem smallest_area_of_P :
  ∀ Q : Set GridPoint,
    (∀ p : GridPoint, p ∈ grid → isDivisibleBySeven p.label → p ∈ Q) →
    areaOfConvexPolygon P ≤ areaOfConvexPolygon Q ∧
    areaOfConvexPolygon P = 60.5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_area_of_P_l719_71939


namespace NUMINAMATH_CALUDE_comprehensive_investigation_is_census_l719_71905

/-- A comprehensive investigation conducted on the subject of examination for a specific purpose -/
def comprehensive_investigation : Type := Unit

/-- Census as a type -/
def census : Type := Unit

/-- Theorem stating that a comprehensive investigation is equivalent to a census -/
theorem comprehensive_investigation_is_census : 
  comprehensive_investigation ≃ census := by sorry

end NUMINAMATH_CALUDE_comprehensive_investigation_is_census_l719_71905


namespace NUMINAMATH_CALUDE_cost_reduction_over_two_years_l719_71919

theorem cost_reduction_over_two_years (total_reduction : ℝ) (annual_reduction : ℝ) :
  total_reduction = 0.19 →
  (1 - annual_reduction) * (1 - annual_reduction) = 1 - total_reduction →
  annual_reduction = 0.1 := by
sorry

end NUMINAMATH_CALUDE_cost_reduction_over_two_years_l719_71919


namespace NUMINAMATH_CALUDE_no_solution_exists_l719_71993

theorem no_solution_exists : ¬∃ (x : ℝ), 3 * (2*x)^2 - 2 * (2*x) + 5 = 2 * (6*x^2 - 3*(2*x) + 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l719_71993


namespace NUMINAMATH_CALUDE_chain_merge_time_theorem_l719_71963

/-- Represents a chain with a certain number of links -/
structure Chain where
  links : ℕ

/-- Represents the time required for chain operations -/
structure ChainOperationTime where
  openLinkTime : ℕ
  closeLinkTime : ℕ

/-- Calculates the minimum time required to merge chains -/
def minTimeMergeChains (chains : List Chain) (opTime : ChainOperationTime) : ℕ :=
  sorry

/-- Theorem statement for the chain merging problem -/
theorem chain_merge_time_theorem (chains : List Chain) (opTime : ChainOperationTime) :
  chains.length = 6 ∧ 
  chains.all (λ c => c.links = 4) ∧
  opTime.openLinkTime = 1 ∧
  opTime.closeLinkTime = 3 →
  minTimeMergeChains chains opTime = 20 :=
sorry

end NUMINAMATH_CALUDE_chain_merge_time_theorem_l719_71963


namespace NUMINAMATH_CALUDE_abacus_problem_solution_l719_71947

/-- Represents a three-digit number -/
def ThreeDigitNumber := { n : ℕ // n ≥ 100 ∧ n < 1000 }

/-- Check if a three-digit number has distinct digits -/
def has_distinct_digits (n : ThreeDigitNumber) : Prop :=
  let digits := [n.val / 100, (n.val / 10) % 10, n.val % 10]
  digits.Nodup

/-- The abacus problem solution -/
theorem abacus_problem_solution :
  ∃! (top bottom : ThreeDigitNumber),
    has_distinct_digits top ∧
    ∃ (k : ℕ), k > 1 ∧ top.val = k * bottom.val ∧
    top.val + bottom.val = 1110 ∧
    top.val = 925 := by
  sorry

end NUMINAMATH_CALUDE_abacus_problem_solution_l719_71947


namespace NUMINAMATH_CALUDE_total_wheels_four_wheelers_l719_71914

theorem total_wheels_four_wheelers (num_four_wheelers : ℕ) (wheels_per_four_wheeler : ℕ) :
  num_four_wheelers = 11 →
  wheels_per_four_wheeler = 4 →
  num_four_wheelers * wheels_per_four_wheeler = 44 :=
by sorry

end NUMINAMATH_CALUDE_total_wheels_four_wheelers_l719_71914


namespace NUMINAMATH_CALUDE_regular_polyhedra_symmetry_axes_l719_71946

-- Define the types of regular polyhedra
inductive RegularPolyhedron
  | Tetrahedron
  | Hexahedron
  | Octahedron
  | Dodecahedron
  | Icosahedron

-- Define a structure for symmetry axis information
structure SymmetryAxis where
  order : ℕ
  count : ℕ

-- Define a function that returns the symmetry axes for a given polyhedron
def symmetryAxes (p : RegularPolyhedron) : List SymmetryAxis :=
  match p with
  | RegularPolyhedron.Tetrahedron => [
      { order := 3, count := 4 },
      { order := 2, count := 3 }
    ]
  | RegularPolyhedron.Hexahedron => [
      { order := 4, count := 3 },
      { order := 3, count := 4 },
      { order := 2, count := 6 }
    ]
  | RegularPolyhedron.Octahedron => [
      { order := 4, count := 3 },
      { order := 3, count := 4 },
      { order := 2, count := 6 }
    ]
  | RegularPolyhedron.Dodecahedron => [
      { order := 5, count := 6 },
      { order := 3, count := 10 },
      { order := 2, count := 15 }
    ]
  | RegularPolyhedron.Icosahedron => [
      { order := 5, count := 6 },
      { order := 3, count := 10 },
      { order := 2, count := 15 }
    ]

-- Theorem stating that the symmetry axes for each polyhedron are correct
theorem regular_polyhedra_symmetry_axes :
  ∀ p : RegularPolyhedron, 
    (symmetryAxes p).length > 0 ∧
    (∀ axis ∈ symmetryAxes p, axis.order ≥ 2 ∧ axis.count > 0) :=
by sorry

end NUMINAMATH_CALUDE_regular_polyhedra_symmetry_axes_l719_71946


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l719_71927

theorem smallest_integer_satisfying_inequality : 
  (∃ (x : ℤ), x / 4 + 3 / 7 > 2 / 3 ∧ ∀ (y : ℤ), y < x → y / 4 + 3 / 7 ≤ 2 / 3) ∧
  (∀ (x : ℤ), x / 4 + 3 / 7 > 2 / 3 → x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l719_71927


namespace NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_13_l719_71977

theorem unique_x_with_three_prime_divisors_including_13 :
  ∀ (x n : ℕ),
    x = 9^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ x = p * q * r) →
    13 ∣ x →
    x = 728 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_with_three_prime_divisors_including_13_l719_71977


namespace NUMINAMATH_CALUDE_area_of_ω_l719_71952

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (12, 9)

-- State that A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- State that the tangent lines intersect at a point on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : circle_area ω = 306 * Real.pi := sorry

end NUMINAMATH_CALUDE_area_of_ω_l719_71952


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_18_3_l719_71964

theorem floor_plus_self_eq_18_3 :
  ∃! s : ℝ, ⌊s⌋ + s = 18.3 ∧ s = 9.3 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_18_3_l719_71964


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l719_71965

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 - 11x - 18 -/
def a : ℝ := 5
def b : ℝ := -11
def c : ℝ := -18

theorem quadratic_discriminant : discriminant a b c = 481 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l719_71965


namespace NUMINAMATH_CALUDE_cutting_tool_distance_l719_71915

-- Define the circle and points
def Circle (center : ℝ × ℝ) (radius : ℝ) := {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

def is_right_angle (A B C : ℝ × ℝ) : Prop :=
  (C.1 - B.1) * (A.1 - B.1) + (C.2 - B.2) * (A.2 - B.2) = 0

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2

-- State the theorem
theorem cutting_tool_distance (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  A ∈ Circle O (Real.sqrt 72) →
  C ∈ Circle O (Real.sqrt 72) →
  distance_squared A B = 64 →
  distance_squared B C = 9 →
  is_right_angle A B C →
  distance_squared O B = 50 := by
  sorry

end NUMINAMATH_CALUDE_cutting_tool_distance_l719_71915


namespace NUMINAMATH_CALUDE_circle_intersection_condition_l719_71926

-- Define the circles B and C
def circle_B (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + b = 0}

def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 + 8*p.2 + 16 = 0}

-- Define the condition for no common points
def no_common_points (b : ℝ) : Prop :=
  circle_B b ∩ circle_C = ∅

-- State the theorem
theorem circle_intersection_condition :
  ∀ b : ℝ, no_common_points b ↔ (-4 < b ∧ b < 0) ∨ b < -64 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_condition_l719_71926


namespace NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l719_71955

theorem product_of_three_consecutive_integers_divisibility :
  ∀ n : ℕ, n > 0 →
  ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k ∧
  ∀ m : ℕ, m > 6 → ∃ n : ℕ, n > 0 ∧ ¬(∃ k : ℕ, (n - 1) * n * (n + 1) = m * k) :=
by sorry

end NUMINAMATH_CALUDE_product_of_three_consecutive_integers_divisibility_l719_71955


namespace NUMINAMATH_CALUDE_inequality_solution_l719_71997

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 1 }
  else if a > 1 then { x | 1/a < x ∧ x < 1 }
  else if a = 1 then ∅
  else if 0 < a ∧ a < 1 then { x | 1 < x ∧ x < 1/a }
  else { x | x < 1/a ∨ x > 1 }

theorem inequality_solution (a : ℝ) :
  { x : ℝ | (a*x - 1)*(x - 1) < 0 } = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l719_71997


namespace NUMINAMATH_CALUDE_greatest_power_under_500_l719_71922

theorem greatest_power_under_500 :
  ∃ (a b : ℕ), 
    a > 0 ∧ 
    b > 1 ∧ 
    a^b < 500 ∧ 
    (∀ (c d : ℕ), c > 0 → d > 1 → c^d < 500 → c^d ≤ a^b) ∧ 
    a + b = 24 :=
by sorry

end NUMINAMATH_CALUDE_greatest_power_under_500_l719_71922


namespace NUMINAMATH_CALUDE_vector_properties_l719_71999

def a : ℝ × ℝ := (-4, 3)
def b : ℝ × ℝ := (7, 1)

theorem vector_properties :
  let angle := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let proj := ((a.1 * b.1 + a.2 * b.2) / (b.1^2 + b.2^2)) • b
  angle = 3 * Real.pi / 4 ∧ proj = (-1/2) • b := by sorry

end NUMINAMATH_CALUDE_vector_properties_l719_71999


namespace NUMINAMATH_CALUDE_sum_of_polygon_sides_l719_71908

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of sides in a triangle -/
def triangle_sides : ℕ := 3

/-- The number of sides in a quadrilateral -/
def quadrilateral_sides : ℕ := 4

/-- Theorem: The sum of the sides of a hexagon, triangle, and quadrilateral is 13 -/
theorem sum_of_polygon_sides : 
  hexagon_sides + triangle_sides + quadrilateral_sides = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polygon_sides_l719_71908


namespace NUMINAMATH_CALUDE_taxi_fare_equality_l719_71903

/-- Taxi fare calculation problem -/
theorem taxi_fare_equality (mike_base_fare annie_base_fare toll_fee : ℚ)
  (per_mile_rate : ℚ) (annie_miles : ℕ) :
  mike_base_fare = 2.5 ∧
  annie_base_fare = 2.5 ∧
  toll_fee = 5 ∧
  per_mile_rate = 0.25 ∧
  annie_miles = 16 →
  ∃ (mike_miles : ℕ),
    mike_base_fare + per_mile_rate * mike_miles =
    annie_base_fare + toll_fee + per_mile_rate * annie_miles ∧
    mike_miles = 36 := by
  sorry

end NUMINAMATH_CALUDE_taxi_fare_equality_l719_71903


namespace NUMINAMATH_CALUDE_line_intersects_circle_l719_71907

/-- Given a point (x₀, y₀) outside the circle x² + y² = r², 
    prove that the line x₀x + y₀y = r² intersects the circle. -/
theorem line_intersects_circle (x₀ y₀ r : ℝ) (h : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ x₀*x + y₀*y = r^2 := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l719_71907


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l719_71938

theorem complex_modulus_problem (z : ℂ) (h : z * (1 - Complex.I) = 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l719_71938


namespace NUMINAMATH_CALUDE_least_prime_angle_in_square_triangle_l719_71960

theorem least_prime_angle_in_square_triangle (a b : ℕ) : 
  (a > b) →
  (Nat.Prime a) →
  (Nat.Prime b) →
  (a + b = 90) →
  (∀ p, Nat.Prime p → p < b → ¬(∃ q, Nat.Prime q ∧ p + q = 90)) →
  b = 7 := by
sorry

end NUMINAMATH_CALUDE_least_prime_angle_in_square_triangle_l719_71960


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l719_71969

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  (Complex.abs a = 3) →
  (Complex.abs b = Real.sqrt 10) →
  (a * b = t - 3 * Complex.I) →
  (t > 0) →
  t = 9 := by
sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l719_71969


namespace NUMINAMATH_CALUDE_tom_found_15_seashells_l719_71932

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := 43

/-- The difference between Fred's and Tom's seashell counts -/
def difference : ℕ := 28

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := fred_seashells - difference

theorem tom_found_15_seashells : tom_seashells = 15 := by
  sorry

end NUMINAMATH_CALUDE_tom_found_15_seashells_l719_71932


namespace NUMINAMATH_CALUDE_max_obtuse_dihedral_angles_l719_71930

/-- A tetrahedron is a polyhedron with four faces. -/
structure Tetrahedron where
  -- We don't need to define the internal structure for this problem

/-- A dihedral angle is the angle between two intersecting planes. -/
structure DihedralAngle where
  -- We don't need to define the internal structure for this problem

/-- An obtuse angle is an angle greater than 90 degrees but less than 180 degrees. -/
def isObtuse (angle : DihedralAngle) : Prop :=
  sorry  -- Definition of obtuse angle

/-- A tetrahedron has exactly 6 dihedral angles. -/
axiom tetrahedron_has_six_dihedral_angles (t : Tetrahedron) :
  ∃ (angles : Finset DihedralAngle), angles.card = 6

/-- The maximum number of obtuse dihedral angles in a tetrahedron is 3. -/
theorem max_obtuse_dihedral_angles (t : Tetrahedron) :
  ∃ (angles : Finset DihedralAngle),
    (∀ a ∈ angles, isObtuse a) ∧
    angles.card = 3 ∧
    ∀ (other_angles : Finset DihedralAngle),
      (∀ a ∈ other_angles, isObtuse a) →
      other_angles.card ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_obtuse_dihedral_angles_l719_71930


namespace NUMINAMATH_CALUDE_presidency_meeting_arrangements_l719_71971

/-- Represents the number of schools --/
def num_schools : ℕ := 3

/-- Represents the number of members per school --/
def members_per_school : ℕ := 5

/-- Calculates the number of ways to choose r items from n items --/
def choose (n : ℕ) (r : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial r) * (Nat.factorial (n - r)))

/-- Represents the number of ways to choose representatives from the host school --/
def host_school_choices : ℕ := choose members_per_school 2

/-- Represents the number of ways to choose representatives from non-host schools --/
def non_host_school_choices : ℕ := (choose members_per_school 1) ^ 2

/-- Represents the total number of ways to arrange the presidency meeting --/
def total_arrangements : ℕ := num_schools * host_school_choices * non_host_school_choices

theorem presidency_meeting_arrangements :
  total_arrangements = 750 :=
sorry

end NUMINAMATH_CALUDE_presidency_meeting_arrangements_l719_71971


namespace NUMINAMATH_CALUDE_some_number_value_l719_71928

theorem some_number_value (a x : ℚ) : 
  a = 105 → 
  a^3 = x * 25 * 45 * 49 → 
  x = 7/3 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l719_71928


namespace NUMINAMATH_CALUDE_car_rental_cost_per_mile_l719_71967

/-- Proves that the cost per mile for a car rental is $0.20 given specific conditions --/
theorem car_rental_cost_per_mile 
  (daily_fee : ℝ) 
  (daily_budget : ℝ) 
  (max_distance : ℝ) 
  (h1 : daily_fee = 50) 
  (h2 : daily_budget = 88) 
  (h3 : max_distance = 190) : 
  ∃ (cost_per_mile : ℝ), 
    cost_per_mile = 0.20 ∧ 
    daily_fee + cost_per_mile * max_distance = daily_budget :=
by
  sorry

end NUMINAMATH_CALUDE_car_rental_cost_per_mile_l719_71967


namespace NUMINAMATH_CALUDE_meeting_participants_l719_71987

theorem meeting_participants :
  ∀ (F M : ℕ),
  F > 0 →
  M > 0 →
  F / 2 = 125 →
  F / 2 + M / 4 = (F + M) / 3 →
  F + M = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_meeting_participants_l719_71987


namespace NUMINAMATH_CALUDE_inequality_proof_l719_71921

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a) / (a^2 + b * c) + (2 * b) / (b^2 + c * a) + (2 * c) / (c^2 + a * b) ≤
  a / (b * c) + b / (c * a) + c / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l719_71921


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l719_71948

theorem binomial_coefficient_equality (x : ℕ) : 
  (Nat.choose 5 x = Nat.choose 5 2) → (x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l719_71948


namespace NUMINAMATH_CALUDE_triangle_is_equilateral_l719_71970

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  β : ℝ
  γ : ℝ

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  (t.a^3 + t.b^3 - t.c^3) / (t.a + t.b - t.c) = t.c^2

def condition2 (t : Triangle) : Prop :=
  Real.sin t.α * Real.sin t.β = 3/4

-- Theorem statement
theorem triangle_is_equilateral (t : Triangle) 
  (h1 : condition1 t) (h2 : condition2 t) :
  t.a = t.b ∧ t.b = t.c ∧ t.α = π/3 ∧ t.β = π/3 ∧ t.γ = π/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_is_equilateral_l719_71970


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l719_71950

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 2 * x^2 - 1 > 0)) ↔ (∃ x₀ : ℝ, 2 * x₀^2 - 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l719_71950


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l719_71942

theorem imaginary_part_of_complex (z : ℂ) (h : z = -4 * Complex.I + 3) : 
  z.im = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l719_71942


namespace NUMINAMATH_CALUDE_cubic_identity_l719_71900

theorem cubic_identity (x : ℝ) (h : x^3 + 1/x^3 = 116) : x + 1/x = 4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_l719_71900


namespace NUMINAMATH_CALUDE_fish_value_in_rice_l719_71979

/-- Represents the trade value of items in terms of bags of rice -/
structure TradeValue where
  fish : ℚ
  bread : ℚ

/-- Defines the trade rates in the distant realm -/
def trade_rates : TradeValue where
  fish := 5⁻¹ * 3 * 6  -- 5 fish = 3 bread, 1 bread = 6 rice
  bread := 6           -- 1 bread = 6 rice

/-- Theorem stating that one fish is equivalent to 3 3/5 bags of rice -/
theorem fish_value_in_rice : trade_rates.fish = 18/5 := by
  sorry

#eval trade_rates.fish

end NUMINAMATH_CALUDE_fish_value_in_rice_l719_71979


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l719_71981

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l719_71981


namespace NUMINAMATH_CALUDE_classroom_ratio_l719_71956

theorem classroom_ratio : 
  ∀ (boys girls : ℕ),
  boys = girls →
  boys + girls = 32 →
  (boys : ℚ) / (girls - 8 : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_classroom_ratio_l719_71956


namespace NUMINAMATH_CALUDE_marthas_children_l719_71996

/-- Given that Martha needs to buy a total number of cakes and each child should receive a specific number of cakes, calculate the number of children Martha has. -/
theorem marthas_children (total_cakes : ℕ) (cakes_per_child : ℚ) : 
  total_cakes = 54 → cakes_per_child = 18 → (total_cakes : ℚ) / cakes_per_child = 3 := by
  sorry

end NUMINAMATH_CALUDE_marthas_children_l719_71996


namespace NUMINAMATH_CALUDE_new_savings_approx_400_l719_71943

/-- Represents the monthly salary in rupees -/
def monthly_salary : ℝ := 7272.727272727273

/-- Represents the initial savings rate as a decimal -/
def initial_savings_rate : ℝ := 0.10

/-- Represents the expense increase rate as a decimal -/
def expense_increase_rate : ℝ := 0.05

/-- Calculates the new monthly savings after the expense increase -/
def new_monthly_savings : ℝ :=
  monthly_salary * (1 - (1 - initial_savings_rate) * (1 + expense_increase_rate))

/-- Theorem stating that the new monthly savings is approximately 400 rupees -/
theorem new_savings_approx_400 :
  ∃ ε > 0, |new_monthly_savings - 400| < ε :=
sorry

end NUMINAMATH_CALUDE_new_savings_approx_400_l719_71943


namespace NUMINAMATH_CALUDE_monotonicity_and_range_l719_71992

noncomputable def f (a b x : ℝ) : ℝ := 2 * a * x + b * x - 1 - 2 * Real.log x

theorem monotonicity_and_range :
  (∀ a ≤ 0, ∀ x > 0, (deriv (f a 0)) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioo 0 (1/a), (deriv (f a 0)) x < 0) ∧
  (∀ a > 0, ∀ x ∈ Set.Ioi (1/a), (deriv (f a 0)) x > 0) ∧
  (∀ x > 0, f 1 b x ≥ 2 * b * x - 3 → b ≤ 2 - 2 / Real.exp 2) :=
by sorry

end NUMINAMATH_CALUDE_monotonicity_and_range_l719_71992


namespace NUMINAMATH_CALUDE_mothers_age_five_times_daughters_l719_71951

/-- 
Given:
- The mother's current age is 43 years.
- The daughter's current age is 11 years.

Prove that 3 years ago, the mother's age was five times her daughter's age.
-/
theorem mothers_age_five_times_daughters (mother_age : ℕ) (daughter_age : ℕ) :
  mother_age = 43 → daughter_age = 11 → 
  ∃ (x : ℕ), x = 3 ∧ (mother_age - x) = 5 * (daughter_age - x) :=
by sorry

end NUMINAMATH_CALUDE_mothers_age_five_times_daughters_l719_71951


namespace NUMINAMATH_CALUDE_ducks_and_dogs_total_l719_71974

theorem ducks_and_dogs_total (d g : ℕ) : 
  d = g + 2 →                   -- number of ducks is 2 more than dogs
  4 * g - 2 * d = 10 →          -- dogs have 10 more legs than ducks
  d + g = 16 := by              -- total number of ducks and dogs is 16
sorry

end NUMINAMATH_CALUDE_ducks_and_dogs_total_l719_71974


namespace NUMINAMATH_CALUDE_racket_sales_total_l719_71954

/-- The total amount earned from selling rackets given the average price per pair and the number of pairs sold -/
theorem racket_sales_total (avg_price : ℝ) (num_pairs : ℕ) : 
  avg_price = 9.8 → num_pairs = 55 → avg_price * (num_pairs : ℝ) = 539 := by
  sorry

end NUMINAMATH_CALUDE_racket_sales_total_l719_71954


namespace NUMINAMATH_CALUDE_gas_price_increase_l719_71916

theorem gas_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.1 * (1 - 27.27272727272727 / 100) = 1 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_gas_price_increase_l719_71916


namespace NUMINAMATH_CALUDE_floor_of_2_7_l719_71936

theorem floor_of_2_7 :
  ⌊(2.7 : ℝ)⌋ = 2 := by sorry

end NUMINAMATH_CALUDE_floor_of_2_7_l719_71936


namespace NUMINAMATH_CALUDE_martins_berry_consumption_l719_71975

/-- Given the cost of berries and Martin's spending habits, calculate his daily berry consumption --/
theorem martins_berry_consumption
  (package_cost : ℚ)
  (total_spent : ℚ)
  (num_days : ℕ)
  (h1 : package_cost = 2)
  (h2 : total_spent = 30)
  (h3 : num_days = 30)
  : (total_spent / package_cost) / num_days = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_martins_berry_consumption_l719_71975


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l719_71931

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def four_consecutive_terms (b : ℕ → ℝ) (S : Set ℝ) : Prop :=
  ∃ k, (b k ∈ S) ∧ (b (k + 1) ∈ S) ∧ (b (k + 2) ∈ S) ∧ (b (k + 3) ∈ S)

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) (b : ℕ → ℝ) :
  is_geometric_sequence a q →
  (∀ n, b n = a n + 1) →
  |q| > 1 →
  four_consecutive_terms b {-53, -23, 19, 37, 82} →
  q = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l719_71931


namespace NUMINAMATH_CALUDE_tumbler_payment_denomination_l719_71923

/-- Proves that the denomination of bills used to pay for tumblers is $100 given the specified conditions -/
theorem tumbler_payment_denomination :
  ∀ (num_tumblers : ℕ) (cost_per_tumbler : ℕ) (num_bills : ℕ) (change : ℕ),
    num_tumblers = 10 →
    cost_per_tumbler = 45 →
    num_bills = 5 →
    change = 50 →
    (num_tumblers * cost_per_tumbler + change) / num_bills = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_tumbler_payment_denomination_l719_71923


namespace NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l719_71973

theorem rectangle_triangle_area_equality (l w h : ℝ) (l_pos : l > 0) (w_pos : w > 0) (h_pos : h > 0) :
  l * w = (1 / 2) * l * h → h = 2 * w := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_area_equality_l719_71973


namespace NUMINAMATH_CALUDE_quadratic_inequality_l719_71913

theorem quadratic_inequality (x : ℝ) : x^2 - x - 30 < 0 ↔ -5 < x ∧ x < 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l719_71913


namespace NUMINAMATH_CALUDE_ceiling_product_sqrt_l719_71912

theorem ceiling_product_sqrt : ⌈Real.sqrt 3⌉ * ⌈Real.sqrt 12⌉ * ⌈Real.sqrt 120⌉ = 88 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_product_sqrt_l719_71912


namespace NUMINAMATH_CALUDE_remainder_theorem_l719_71978

theorem remainder_theorem (d : ℚ) : 
  (∃! d, ∀ x, (3 * x^3 + d * x^2 - 6 * x + 25) % (3 * x + 5) = 3) → d = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l719_71978


namespace NUMINAMATH_CALUDE_kids_difference_l719_71937

theorem kids_difference (monday tuesday : ℕ) 
  (h1 : monday = 11) 
  (h2 : tuesday = 12) : 
  tuesday - monday = 1 := by
  sorry

end NUMINAMATH_CALUDE_kids_difference_l719_71937


namespace NUMINAMATH_CALUDE_markup_calculation_l719_71985

/-- Given a purchase price, overhead percentage, and desired net profit, 
    calculate the required markup. -/
def calculate_markup (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) : ℝ :=
  let overhead_cost := overhead_percentage * purchase_price
  let total_cost := purchase_price + overhead_cost
  let selling_price := total_cost + net_profit
  selling_price - purchase_price

/-- Theorem stating that the markup for the given conditions is $14.40 -/
theorem markup_calculation : 
  calculate_markup 48 0.05 12 = 14.40 := by
  sorry

end NUMINAMATH_CALUDE_markup_calculation_l719_71985


namespace NUMINAMATH_CALUDE_prime_factor_count_l719_71929

/-- Given an expression 4^11 * x^5 * 11^2 with a total of 29 prime factors, x must be a prime number -/
theorem prime_factor_count (x : ℕ) : 
  (∀ (p : ℕ), Prime p → (Nat.factorization (4^11 * x^5 * 11^2)).sum (λ _ e => e) = 29) → 
  Prime x := by
sorry

end NUMINAMATH_CALUDE_prime_factor_count_l719_71929


namespace NUMINAMATH_CALUDE_min_people_needed_is_30_l719_71984

/-- Represents the types of vehicles --/
inductive VehicleType
| SmallCar
| MediumCar
| LargeCar
| LightTruck
| HeavyTruck

/-- Returns the weight of a vehicle type in pounds --/
def vehicleWeight (v : VehicleType) : ℕ :=
  match v with
  | .SmallCar => 2000
  | .MediumCar => 3000
  | .LargeCar => 4000
  | .LightTruck => 10000
  | .HeavyTruck => 15000

/-- Represents the fleet of vehicles --/
def fleet : List (VehicleType × ℕ) :=
  [(VehicleType.SmallCar, 2), (VehicleType.MediumCar, 2), (VehicleType.LargeCar, 2),
   (VehicleType.LightTruck, 1), (VehicleType.HeavyTruck, 2)]

/-- The maximum lifting capacity of a person in pounds --/
def maxLiftingCapacity : ℕ := 1000

/-- Calculates the total weight of the fleet --/
def totalFleetWeight : ℕ :=
  fleet.foldl (fun acc (v, count) => acc + vehicleWeight v * count) 0

/-- Theorem: The minimum number of people needed to lift all vehicles is 30 --/
theorem min_people_needed_is_30 :
  ∃ (n : ℕ), n = 30 ∧
  n * maxLiftingCapacity ≥ totalFleetWeight ∧
  ∀ (m : ℕ), m * maxLiftingCapacity ≥ totalFleetWeight → m ≥ n :=
sorry

end NUMINAMATH_CALUDE_min_people_needed_is_30_l719_71984


namespace NUMINAMATH_CALUDE_derivative_of_odd_function_l719_71924

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (hf : Differentiable ℝ f)
variable (hodd : ∀ x, f (-x) = -f x)

-- Define x₀ and k
variable (x₀ : ℝ)
variable (k : ℝ)
variable (hk : k ≠ 0)

-- State the hypothesis about f'(-x₀)
variable (hderiv : deriv f (-x₀) = k)

-- State the theorem
theorem derivative_of_odd_function :
  deriv f x₀ = k := by sorry

end NUMINAMATH_CALUDE_derivative_of_odd_function_l719_71924


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l719_71972

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3*c)) + (b / (8*c + 4*a)) + (9*c / (3*a + 2*b)) ≥ 47/48 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 3*c)) + (b / (8*c + 4*a)) + (9*c / (3*a + 2*b)) = 47/48 ↔ 
  ∃ (k : ℝ), k > 0 ∧ a = 10*k ∧ b = 21*k ∧ c = k :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l719_71972


namespace NUMINAMATH_CALUDE_cube_sum_relation_l719_71953

theorem cube_sum_relation : 
  (2^3 + 4^3 + 6^3 + 8^3 + 10^3 + 12^3 + 14^3 + 16^3 + 18^3 = 16200) →
  (3^3 + 6^3 + 9^3 + 12^3 + 15^3 + 18^3 + 21^3 + 24^3 + 27^3 = 54675) :=
by
  sorry

end NUMINAMATH_CALUDE_cube_sum_relation_l719_71953


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l719_71988

theorem gcd_of_three_numbers : Nat.gcd 4560 (Nat.gcd 6080 16560) = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l719_71988


namespace NUMINAMATH_CALUDE_total_weight_on_scale_l719_71940

theorem total_weight_on_scale (alexa_weight katerina_weight : ℕ) 
  (h1 : alexa_weight = 46)
  (h2 : katerina_weight = 49) :
  alexa_weight + katerina_weight = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_on_scale_l719_71940


namespace NUMINAMATH_CALUDE_minimum_married_men_l719_71968

theorem minimum_married_men (total_men : ℕ) (tv_men : ℕ) (radio_men : ℕ) (ac_men : ℕ) (married_with_all : ℕ)
  (h_total : total_men = 100)
  (h_tv : tv_men = 75)
  (h_radio : radio_men = 85)
  (h_ac : ac_men = 70)
  (h_married_all : married_with_all = 11)
  (h_tv_le : tv_men ≤ total_men)
  (h_radio_le : radio_men ≤ total_men)
  (h_ac_le : ac_men ≤ total_men)
  (h_married_all_le : married_with_all ≤ tv_men ∧ married_with_all ≤ radio_men ∧ married_with_all ≤ ac_men) :
  ∃ (married_men : ℕ), married_men ≥ married_with_all ∧ married_men ≤ total_men := by
  sorry

end NUMINAMATH_CALUDE_minimum_married_men_l719_71968


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l719_71941

theorem cryptarithmetic_solution : 
  ∃! (K I S : Nat), 
    K < 10 ∧ I < 10 ∧ S < 10 ∧
    K ≠ I ∧ K ≠ S ∧ I ≠ S ∧
    100 * K + 10 * I + S + 100 * K + 10 * S + I = 100 * I + 10 * S + K ∧
    K = 4 ∧ I = 9 ∧ S = 5 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l719_71941


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l719_71909

/-- 
If the quadratic equation 2x^2 - x + c = 0 has two equal real roots, 
then c = 1/8.
-/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, 2 * x^2 - x + c = 0 ∧ 
   ∀ y : ℝ, 2 * y^2 - y + c = 0 → y = x) → 
  c = 1/8 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l719_71909


namespace NUMINAMATH_CALUDE_oil_depth_conversion_l719_71995

/-- Represents a right cylindrical tank with oil -/
structure OilTank where
  height : ℝ
  baseDiameter : ℝ
  sideOilDepth : ℝ

/-- Calculates the upright oil depth given a tank configuration -/
noncomputable def uprightOilDepth (tank : OilTank) : ℝ :=
  sorry

/-- Theorem stating the relationship between side oil depth and upright oil depth -/
theorem oil_depth_conversion (tank : OilTank) 
  (h1 : tank.height = 12)
  (h2 : tank.baseDiameter = 6)
  (h3 : tank.sideOilDepth = 2) :
  ∃ (ε : ℝ), abs (uprightOilDepth tank - 2.4) < ε ∧ ε < 0.1 :=
sorry

end NUMINAMATH_CALUDE_oil_depth_conversion_l719_71995


namespace NUMINAMATH_CALUDE_min_sum_of_sides_l719_71998

theorem min_sum_of_sides (a b c : ℝ) (A B C : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  ((a + b)^2 - c^2 = 4) →
  (C = Real.pi / 3) →
  (∃ (x : ℝ), (a + b ≥ x) ∧ (∀ y, a + b ≥ y → x ≤ y) ∧ (x = 4 * Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_sides_l719_71998


namespace NUMINAMATH_CALUDE_absolute_value_inequality_range_l719_71920

theorem absolute_value_inequality_range (a : ℝ) : 
  (∀ x : ℝ, |x - 1| + |x - 3| ≥ a^2 + a) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_range_l719_71920


namespace NUMINAMATH_CALUDE_largest_six_digit_number_l719_71917

def digit_product (n : ℕ) : ℕ :=
  if n = 0 then 1 else (n % 10) * digit_product (n / 10)

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem largest_six_digit_number : ∀ n : ℕ, 
  100000 ≤ n ∧ n ≤ 999999 ∧ digit_product n = factorial 8 → n ≤ 987744 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_six_digit_number_l719_71917


namespace NUMINAMATH_CALUDE_sports_conference_games_l719_71918

theorem sports_conference_games (n : ℕ) (d : ℕ) (intra : ℕ) (inter : ℕ) 
  (h1 : n = 16)
  (h2 : d = 2)
  (h3 : n = d * 8)
  (h4 : intra = 3)
  (h5 : inter = 2) :
  d * (Nat.choose 8 2 * intra) + (n / 2) * (n / 2) * inter = 296 := by
  sorry

end NUMINAMATH_CALUDE_sports_conference_games_l719_71918


namespace NUMINAMATH_CALUDE_planar_graph_edge_count_l719_71959

/-- A planar graph -/
structure PlanarGraph where
  V : Type* -- Vertex set
  E : Type* -- Edge set
  n : ℕ     -- Number of vertices
  m : ℕ     -- Number of edges
  is_planar : Bool
  vertex_count : n ≥ 3

/-- A planar triangulation -/
structure PlanarTriangulation extends PlanarGraph where
  is_triangulation : Bool

/-- Theorem about the number of edges in planar graphs and planar triangulations -/
theorem planar_graph_edge_count (G : PlanarGraph) :
  G.m ≤ 3 * G.n - 6 ∧
  (∀ (T : PlanarTriangulation), T.toPlanarGraph = G → T.m = 3 * T.n - 6) :=
sorry

end NUMINAMATH_CALUDE_planar_graph_edge_count_l719_71959


namespace NUMINAMATH_CALUDE_basketball_team_age_stats_l719_71933

/-- Represents the age distribution of players in a basketball team -/
structure AgeDistribution :=
  (age18 : ℕ)
  (age19 : ℕ)
  (age20 : ℕ)
  (age21 : ℕ)
  (total : ℕ)
  (sum : ℕ)
  (h_total : age18 + age19 + age20 + age21 = total)
  (h_sum : 18 * age18 + 19 * age19 + 20 * age20 + 21 * age21 = sum)

/-- The mode of a set of ages -/
def mode (d : AgeDistribution) : ℕ :=
  max (max d.age18 d.age19) (max d.age20 d.age21)

/-- The mean of a set of ages -/
def mean (d : AgeDistribution) : ℚ :=
  d.sum / d.total

/-- Theorem stating the mode and mean of the given age distribution -/
theorem basketball_team_age_stats :
  ∃ d : AgeDistribution,
    d.age18 = 5 ∧
    d.age19 = 4 ∧
    d.age20 = 1 ∧
    d.age21 = 2 ∧
    d.total = 12 ∧
    mode d = 18 ∧
    mean d = 19 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_age_stats_l719_71933


namespace NUMINAMATH_CALUDE_quadratic_solution_l719_71949

theorem quadratic_solution (m : ℝ) : 
  (2^2 - m*2 + 8 = 0) → m = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l719_71949
