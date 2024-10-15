import Mathlib

namespace NUMINAMATH_CALUDE_largest_common_term_l703_70363

theorem largest_common_term (n m : ℕ) : 
  (∃ n m : ℕ, 187 = 3 + 8 * n ∧ 187 = 5 + 9 * m) ∧ 
  (∀ k : ℕ, k > 187 → k ≤ 200 → ¬(∃ p q : ℕ, k = 3 + 8 * p ∧ k = 5 + 9 * q)) := by
  sorry

end NUMINAMATH_CALUDE_largest_common_term_l703_70363


namespace NUMINAMATH_CALUDE_subtractions_to_additions_theorem_l703_70327

-- Define the original expression
def original_expression : List ℤ := [6, -3, 7, -2]

-- Define the operation of changing subtractions to additions
def change_subtractions_to_additions (expr : List ℤ) : List ℤ :=
  expr.map (λ x => if x < 0 then -x else x)

-- Define the result of the operation
def result_expression : List ℤ := [6, -3, 7, -2]

-- State the theorem
theorem subtractions_to_additions_theorem :
  change_subtractions_to_additions original_expression = result_expression :=
sorry

end NUMINAMATH_CALUDE_subtractions_to_additions_theorem_l703_70327


namespace NUMINAMATH_CALUDE_larger_number_problem_l703_70346

theorem larger_number_problem (x y : ℕ) : 
  x + y = 64 → y = x + 12 → y = 38 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l703_70346


namespace NUMINAMATH_CALUDE_ellipse_properties_l703_70369

/-- An ellipse with minor axis length 2√3 and foci at (-1,0) and (1,0) -/
structure Ellipse where
  minor_axis : ℝ
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ
  minor_axis_eq : minor_axis = 2 * Real.sqrt 3
  foci_eq : focus1 = (-1, 0) ∧ focus2 = (1, 0)

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- The line y = x + m intersects the ellipse at two distinct points -/
def intersects_at_two_points (e : Ellipse) (m : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂, x₁ ≠ x₂ ∧ 
    standard_equation e x₁ y₁ ∧ 
    standard_equation e x₂ y₂ ∧
    y₁ = x₁ + m ∧ 
    y₂ = x₂ + m

theorem ellipse_properties (e : Ellipse) :
  (∀ x y, standard_equation e x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (∀ m, intersects_at_two_points e m ↔ -Real.sqrt 7 < m ∧ m < Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l703_70369


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_five_ninths_l703_70338

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  edge_length : ℕ
  small_cube_count : ℕ
  white_cube_count : ℕ
  black_cube_count : ℕ

/-- Calculate the fraction of white surface area for a composite cube -/
def white_surface_fraction (c : CompositeCube) : ℚ :=
  let total_surface_area := 6 * c.edge_length^2
  let black_faces := 3 * c.black_cube_count
  let white_faces := total_surface_area - black_faces
  white_faces / total_surface_area

/-- The specific cube described in the problem -/
def problem_cube : CompositeCube :=
  { edge_length := 3
  , small_cube_count := 27
  , white_cube_count := 19
  , black_cube_count := 8 }

theorem white_surface_fraction_is_five_ninths :
  white_surface_fraction problem_cube = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_five_ninths_l703_70338


namespace NUMINAMATH_CALUDE_solve_linear_equation_l703_70386

theorem solve_linear_equation (x : ℝ) (h : x - 3*x + 5*x = 150) : x = 50 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l703_70386


namespace NUMINAMATH_CALUDE_number_of_provinces_l703_70345

theorem number_of_provinces (P T : ℕ) (n : ℕ) : 
  T = (3 * P) / 4 →  -- The fraction of traditionalists is 0.75
  (∃ k : ℕ, T = k * (P / 12)) →  -- Each province has P/12 traditionalists
  n = T / (P / 12) →  -- Definition of n
  n = 9 :=
by sorry

end NUMINAMATH_CALUDE_number_of_provinces_l703_70345


namespace NUMINAMATH_CALUDE_airport_distance_proof_l703_70310

/-- The distance from Victor's home to the airport -/
def airport_distance : ℝ := 150

/-- Victor's initial speed -/
def initial_speed : ℝ := 60

/-- Victor's increased speed -/
def increased_speed : ℝ := 80

/-- Time Victor drives at initial speed -/
def initial_drive_time : ℝ := 0.5

/-- Time difference if Victor continued at initial speed -/
def late_time : ℝ := 0.25

/-- Time difference after increasing speed -/
def early_time : ℝ := 0.25

theorem airport_distance_proof :
  ∃ (planned_time : ℝ),
    -- Distance covered at initial speed
    initial_speed * initial_drive_time +
    -- Remaining distance if continued at initial speed
    initial_speed * (planned_time + late_time) =
    -- Distance covered at initial speed
    initial_speed * initial_drive_time +
    -- Remaining distance at increased speed
    increased_speed * (planned_time - early_time) ∧
    -- Total distance equals airport_distance
    airport_distance = initial_speed * initial_drive_time +
                       increased_speed * (planned_time - early_time) := by
  sorry

end NUMINAMATH_CALUDE_airport_distance_proof_l703_70310


namespace NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l703_70357

theorem greatest_two_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≤ 99 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by sorry

end NUMINAMATH_CALUDE_greatest_two_digit_multiple_of_17_l703_70357


namespace NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l703_70306

theorem ab_positive_necessary_not_sufficient (a b : ℝ) :
  (∀ a b, b / a + a / b > 2 → a * b > 0) ∧
  (∃ a b, a * b > 0 ∧ ¬(b / a + a / b > 2)) :=
sorry

end NUMINAMATH_CALUDE_ab_positive_necessary_not_sufficient_l703_70306


namespace NUMINAMATH_CALUDE_marble_jar_ratio_l703_70302

/-- Proves that the ratio of marbles in the second jar to the first jar is 2:1 --/
theorem marble_jar_ratio :
  ∀ (jar1 jar2 jar3 : ℕ),
  jar1 = 80 →
  jar3 = jar1 / 4 →
  jar1 + jar2 + jar3 = 260 →
  jar2 = 2 * jar1 :=
by
  sorry

end NUMINAMATH_CALUDE_marble_jar_ratio_l703_70302


namespace NUMINAMATH_CALUDE_total_coins_proof_l703_70311

theorem total_coins_proof (jayden_coins jasmine_coins : ℕ) 
  (h1 : jayden_coins = 300)
  (h2 : jasmine_coins = 335)
  (h3 : ∃ jason_coins : ℕ, jason_coins = jayden_coins + 60 ∧ jason_coins = jasmine_coins + 25) :
  ∃ total_coins : ℕ, total_coins = jayden_coins + jasmine_coins + (jayden_coins + 60) ∧ total_coins = 995 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_proof_l703_70311


namespace NUMINAMATH_CALUDE_range_of_a_l703_70389

open Set Real

def A (a : ℝ) : Set ℝ := {x | 3 + a ≤ x ∧ x ≤ 4 + 3*a}
def B : Set ℝ := {x | (x + 4) / (5 - x) ≥ 0}

theorem range_of_a :
  ∀ a : ℝ, (A a).Nonempty ∧ (∀ x : ℝ, x ∈ A a → x ∈ B) →
  a ∈ Icc (-1/2) (1/3) := by sorry

end NUMINAMATH_CALUDE_range_of_a_l703_70389


namespace NUMINAMATH_CALUDE_book_pages_theorem_l703_70305

theorem book_pages_theorem :
  ∀ (book1 book2 book3 : ℕ),
    (2 * book1) / 3 - (book1 / 3) = 20 →
    (3 * book2) / 5 - (2 * book2) / 5 = 15 →
    (3 * book3) / 4 - (book3 / 4) = 30 →
    book1 = 60 ∧ book2 = 75 ∧ book3 = 60 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l703_70305


namespace NUMINAMATH_CALUDE_consecutive_even_integers_sum_l703_70308

theorem consecutive_even_integers_sum (x : ℕ) (h1 : x > 4) : 
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → 
  (x - 4) + (x - 2) + x + (x + 2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_integers_sum_l703_70308


namespace NUMINAMATH_CALUDE_no_uphill_integers_divisible_by_45_l703_70342

/-- An uphill integer is a positive integer where each digit is strictly greater than the previous digit. -/
def is_uphill (n : ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < j ∧ j < (Nat.digits 10 n).length →
    (Nat.digits 10 n).get ⟨i, by sorry⟩ < (Nat.digits 10 n).get ⟨j, by sorry⟩

/-- A number is divisible by 45 if and only if it is divisible by both 9 and 5. -/
def divisible_by_45 (n : ℕ) : Prop :=
  n % 45 = 0

theorem no_uphill_integers_divisible_by_45 :
  ¬ ∃ n : ℕ, is_uphill n ∧ divisible_by_45 n :=
sorry

end NUMINAMATH_CALUDE_no_uphill_integers_divisible_by_45_l703_70342


namespace NUMINAMATH_CALUDE_bodhi_yacht_balance_l703_70322

/-- The number of sheep needed to balance a yacht -/
def sheep_needed (cows foxes : ℕ) : ℕ :=
  let zebras := 3 * foxes
  let total_needed := 100
  total_needed - (cows + foxes + zebras)

/-- Theorem stating the number of sheep needed for Mr. Bodhi's yacht -/
theorem bodhi_yacht_balance :
  sheep_needed 20 15 = 20 := by
  sorry

end NUMINAMATH_CALUDE_bodhi_yacht_balance_l703_70322


namespace NUMINAMATH_CALUDE_hyperbola_standard_equation_l703_70356

/-- The standard equation of a hyperbola with foci on the x-axis, given a and b -/
def hyperbola_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

/-- Theorem: Given a = 3 and b = 4, the standard equation of the hyperbola with foci on the x-axis is (x²/9) - (y²/16) = 1 -/
theorem hyperbola_standard_equation (x y : ℝ) :
  let a : ℝ := 3
  let b : ℝ := 4
  hyperbola_equation x y a b ↔ (x^2 / 9) - (y^2 / 16) = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_standard_equation_l703_70356


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l703_70325

theorem unique_solution_quadratic (k : ℚ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = k + 3 * x) ↔ k = 6 := by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l703_70325


namespace NUMINAMATH_CALUDE_inverse_function_point_correspondence_l703_70394

theorem inverse_function_point_correspondence
  (f : ℝ → ℝ) (h_inv : Function.Bijective f) :
  (Function.invFun f) 1 = 2 → f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_correspondence_l703_70394


namespace NUMINAMATH_CALUDE_distinct_roots_iff_k_gt_three_fourths_roots_condition_implies_k_value_l703_70362

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : Prop :=
  x^2 + (2*k + 1)*x + k^2 + 1 = 0

-- Define the condition for two distinct real roots
def has_two_distinct_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_equation k x1 ∧ quadratic_equation k x2

-- Define the condition for the sum and product of roots
def roots_condition (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, quadratic_equation k x1 ∧ quadratic_equation k x2 ∧ 
    x1 + x2 = 2 - x1 * x2

-- Theorem for part 1
theorem distinct_roots_iff_k_gt_three_fourths :
  ∀ k : ℝ, has_two_distinct_roots k ↔ k > 3/4 :=
sorry

-- Theorem for part 2
theorem roots_condition_implies_k_value :
  ∀ k : ℝ, roots_condition k → k = 1 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_iff_k_gt_three_fourths_roots_condition_implies_k_value_l703_70362


namespace NUMINAMATH_CALUDE_max_seed_weight_is_75_l703_70314

/-- Represents the weight and price of a bag of grass seed -/
structure SeedBag where
  weight : ℕ
  price : ℚ

/-- Finds the maximum weight of grass seed that can be purchased given the conditions -/
def maxSeedWeight (bags : List SeedBag) (minWeight : ℕ) (maxCost : ℚ) : ℕ :=
  sorry

/-- The theorem stating the maximum weight of grass seed that can be purchased -/
theorem max_seed_weight_is_75 (bags : List SeedBag) (h1 : bags = [
  ⟨5, 1385/100⟩, ⟨10, 2042/100⟩, ⟨25, 3225/100⟩
]) (h2 : maxSeedWeight bags 65 (9877/100) = 75) : 
  maxSeedWeight bags 65 (9877/100) = 75 :=
by sorry

end NUMINAMATH_CALUDE_max_seed_weight_is_75_l703_70314


namespace NUMINAMATH_CALUDE_davids_trip_money_l703_70393

theorem davids_trip_money (initial_amount spent_amount remaining_amount : ℕ) :
  remaining_amount = 500 →
  remaining_amount = spent_amount - 500 →
  initial_amount = spent_amount + remaining_amount →
  initial_amount = 1500 := by
sorry

end NUMINAMATH_CALUDE_davids_trip_money_l703_70393


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l703_70317

theorem division_multiplication_problem : 5 / (-1/5) * 5 = -125 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l703_70317


namespace NUMINAMATH_CALUDE_point_on_inverse_graph_and_sum_l703_70329

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
noncomputable def f_inv : ℝ → ℝ := Function.invFun f

-- State the theorem
theorem point_on_inverse_graph_and_sum (h : f 2 = 9) :
  f_inv 9 = 2 ∧ 9 + (2 / 3) = 29 / 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_inverse_graph_and_sum_l703_70329


namespace NUMINAMATH_CALUDE_winning_pair_probability_l703_70383

/-- Represents a card with a color and a label -/
structure Card where
  color : String
  label : String

/-- The set of all cards -/
def allCards : Finset Card := sorry

/-- A winning pair is defined as either two cards with the same label or two cards of the same color -/
def isWinningPair (pair : Finset Card) : Prop := sorry

/-- The probability of drawing a winning pair -/
def winningProbability : ℚ := sorry

/-- Theorem: The probability of drawing a winning pair is 3/5 -/
theorem winning_pair_probability : winningProbability = 3/5 := by sorry

end NUMINAMATH_CALUDE_winning_pair_probability_l703_70383


namespace NUMINAMATH_CALUDE_inequality_chain_l703_70335

/-- Given a > 0, b > 0, a ≠ b, prove that f((a+b)/2) < f(√(ab)) < f(2ab/(a+b)) where f(x) = (1/3)^x -/
theorem inequality_chain (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  let f : ℝ → ℝ := fun x ↦ (1/3)^x
  f ((a + b) / 2) < f (Real.sqrt (a * b)) ∧ f (Real.sqrt (a * b)) < f (2 * a * b / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_chain_l703_70335


namespace NUMINAMATH_CALUDE_simplify_expression_l703_70381

theorem simplify_expression (a b : ℝ) : (8*a - 7*b) - (4*a - 5*b) = 4*a - 2*b := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l703_70381


namespace NUMINAMATH_CALUDE_sum_of_roots_l703_70395

theorem sum_of_roots (x : ℝ) : 
  (∃ r₁ r₂ : ℝ, (3*r₁ + 4)*(r₁ - 5) + (3*r₁ + 4)*(r₁ - 7) = 0 ∧ 
                (3*r₂ + 4)*(r₂ - 5) + (3*r₂ + 4)*(r₂ - 7) = 0 ∧ 
                r₁ ≠ r₂) → 
  (∃ r₁ r₂ : ℝ, (3*r₁ + 4)*(r₁ - 5) + (3*r₁ + 4)*(r₁ - 7) = 0 ∧ 
                (3*r₂ + 4)*(r₂ - 5) + (3*r₂ + 4)*(r₂ - 7) = 0 ∧ 
                r₁ + r₂ = 14/3) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l703_70395


namespace NUMINAMATH_CALUDE_two_digit_number_transformation_l703_70307

/-- Given a two-digit integer n = 10a + b, where n = (k+1)(a + b),
    prove that 10(a+1) + (b+1) = ((k+1)(a + b) + 11) / (a + b + 2) * (a + b + 2) -/
theorem two_digit_number_transformation (a b k : ℕ) (h1 : 10*a + b = (k+1)*(a + b)) :
  10*(a+1) + (b+1) = ((k+1)*(a + b) + 11) / (a + b + 2) * (a + b + 2) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_transformation_l703_70307


namespace NUMINAMATH_CALUDE_total_turnips_is_105_l703_70377

/-- The number of turnips Keith grows per day -/
def keith_turnips_per_day : ℕ := 6

/-- The number of days Keith grows turnips -/
def keith_days : ℕ := 7

/-- The number of turnips Alyssa grows every two days -/
def alyssa_turnips_per_two_days : ℕ := 9

/-- The number of days Alyssa grows turnips -/
def alyssa_days : ℕ := 14

/-- The total number of turnips grown by Keith and Alyssa -/
def total_turnips : ℕ :=
  keith_turnips_per_day * keith_days +
  (alyssa_turnips_per_two_days * (alyssa_days / 2))

theorem total_turnips_is_105 : total_turnips = 105 := by
  sorry

end NUMINAMATH_CALUDE_total_turnips_is_105_l703_70377


namespace NUMINAMATH_CALUDE_competition_participants_l703_70385

theorem competition_participants : ∀ (initial : ℕ),
  (initial : ℚ) * (1 - 0.6) * (1 / 4) = 30 →
  initial = 300 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l703_70385


namespace NUMINAMATH_CALUDE_unbounded_sequence_l703_70319

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) = (a (n + 1) - a n) ^ (Real.sqrt n) + n ^ (-(Real.sqrt n))

theorem unbounded_sequence
  (a : ℕ → ℝ)
  (h_pos : ∀ n, 0 < a n)
  (h_incr : is_strictly_increasing a)
  (h_prop : sequence_property a) :
  ∀ C, ∃ m, C < a m :=
sorry

end NUMINAMATH_CALUDE_unbounded_sequence_l703_70319


namespace NUMINAMATH_CALUDE_square_equation_solution_l703_70344

theorem square_equation_solution : ∃! x : ℝ, 97 + x * (19 + 91 / x) = 321 ∧ x = 7 := by sorry

end NUMINAMATH_CALUDE_square_equation_solution_l703_70344


namespace NUMINAMATH_CALUDE_total_painting_time_l703_70368

/-- Given that Hadassah paints 12 paintings in 6 hours and adds 20 more paintings,
    prove that the total time to finish all paintings is 16 hours. -/
theorem total_painting_time (initial_paintings : ℕ) (initial_time : ℝ) (additional_paintings : ℕ) :
  initial_paintings = 12 →
  initial_time = 6 →
  additional_paintings = 20 →
  (initial_time + (additional_paintings * (initial_time / initial_paintings))) = 16 :=
by sorry

end NUMINAMATH_CALUDE_total_painting_time_l703_70368


namespace NUMINAMATH_CALUDE_circles_coaxial_system_l703_70392

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point2D
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle in 2D space -/
structure Triangle where
  a : Point2D
  b : Point2D
  c : Point2D

/-- Checks if three circles form a coaxial system -/
def areCoaxial (c1 c2 c3 : Circle) : Prop :=
  sorry

/-- Constructs a circle with diameter as the given line segment -/
def circleDiameterSegment (p1 p2 : Point2D) : Circle :=
  sorry

/-- Finds the intersection point of a line and a triangle side -/
def lineTriangleIntersection (l : Line) (t : Triangle) : Point2D :=
  sorry

/-- Main theorem: Given a triangle intersected by a line, 
    the circles constructed on the resulting segments form a coaxial system -/
theorem circles_coaxial_system 
  (t : Triangle) 
  (l : Line) : 
  let a1 := lineTriangleIntersection l t
  let b1 := lineTriangleIntersection l t
  let c1 := lineTriangleIntersection l t
  let circleA := circleDiameterSegment t.a a1
  let circleB := circleDiameterSegment t.b b1
  let circleC := circleDiameterSegment t.c c1
  areCoaxial circleA circleB circleC :=
by
  sorry

end NUMINAMATH_CALUDE_circles_coaxial_system_l703_70392


namespace NUMINAMATH_CALUDE_radical_product_equals_27_l703_70355

theorem radical_product_equals_27 : 
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by
  sorry

end NUMINAMATH_CALUDE_radical_product_equals_27_l703_70355


namespace NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_negative_l703_70316

theorem x_negative_necessary_not_sufficient_for_ln_negative :
  (∀ x, Real.log (x + 1) < 0 → x < 0) ∧
  (∃ x, x < 0 ∧ Real.log (x + 1) ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_x_negative_necessary_not_sufficient_for_ln_negative_l703_70316


namespace NUMINAMATH_CALUDE_monochromatic_sequence_exists_l703_70332

/-- A color can be either red or blue -/
inductive Color
| red
| blue

/-- A coloring function assigns a color to each positive integer -/
def Coloring := ℕ+ → Color

/-- An infinite sequence of positive integers -/
def InfiniteSequence := ℕ → ℕ+

theorem monochromatic_sequence_exists (c : Coloring) :
  ∃ (seq : InfiniteSequence) (color : Color),
    (∀ n : ℕ, seq n < seq (n + 1)) ∧
    (∀ n : ℕ, ∃ k : ℕ+, 2 * k = seq n + seq (n + 1)) ∧
    (∀ n : ℕ, c (seq n) = color ∧ c k = color) :=
sorry

end NUMINAMATH_CALUDE_monochromatic_sequence_exists_l703_70332


namespace NUMINAMATH_CALUDE_distance_D_to_ABC_plane_l703_70372

/-- The distance from a point to a plane in 3D space --/
def distancePointToPlane (p : ℝ × ℝ × ℝ) (a b c : ℝ × ℝ × ℝ) : ℝ := sorry

/-- Theorem: The distance from point D to plane ABC is 11 --/
theorem distance_D_to_ABC_plane : 
  let A : ℝ × ℝ × ℝ := (2, 3, 1)
  let B : ℝ × ℝ × ℝ := (4, 1, -2)
  let C : ℝ × ℝ × ℝ := (6, 3, 7)
  let D : ℝ × ℝ × ℝ := (-5, -4, 8)
  distancePointToPlane D A B C = 11 := by sorry

end NUMINAMATH_CALUDE_distance_D_to_ABC_plane_l703_70372


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_x_in_range_f_always_negative_implies_x_in_range_l703_70374

/-- The function f(x) = ax^2 - (2a+1)x + a+1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (2*a + 1) * x + a + 1

theorem f_nonnegative_iff_x_in_range (a : ℝ) (x : ℝ) :
  a = 2 → (f a x ≥ 0 ↔ x ≥ 3/2 ∨ x ≤ 1) := by sorry

theorem f_always_negative_implies_x_in_range (a : ℝ) (x : ℝ) :
  a ∈ Set.Icc (-2) 2 → (∀ y, f a y < 0) → x ∈ Set.Ioo 1 (3/2) := by sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_x_in_range_f_always_negative_implies_x_in_range_l703_70374


namespace NUMINAMATH_CALUDE_joes_money_from_mother_l703_70312

def notebook_cost : ℕ := 4
def book_cost : ℕ := 7
def notebooks_bought : ℕ := 7
def books_bought : ℕ := 2
def money_left : ℕ := 14

theorem joes_money_from_mother : 
  notebook_cost * notebooks_bought + book_cost * books_bought + money_left = 56 := by
  sorry

end NUMINAMATH_CALUDE_joes_money_from_mother_l703_70312


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_from_focus_distance_l703_70399

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The semi-focal length of a hyperbola -/
def semi_focal_length (h : Hyperbola) : ℝ := sorry

/-- The distance from a focus to an asymptote of a hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := sorry

theorem hyperbola_eccentricity_from_focus_distance (h : Hyperbola) 
  (h_dist : focus_to_asymptote_distance h = (Real.sqrt 5 / 3) * semi_focal_length h) : 
  eccentricity h = 3 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_from_focus_distance_l703_70399


namespace NUMINAMATH_CALUDE_triangle_area_angle_l703_70347

theorem triangle_area_angle (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a > 0 → b > 0 → c > 0 →
  0 < C → C < π →
  S = (1/4) * (a^2 + b^2 - c^2) →
  S = (1/2) * a * b * Real.sin C →
  C = π/4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_angle_l703_70347


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_equation_l703_70378

/-- A quadrilateral inscribed in a semicircle -/
structure InscribedQuadrilateral where
  /-- The diameter of the semicircle -/
  x : ℝ
  /-- The length of side AM -/
  a : ℝ
  /-- The length of side MN -/
  b : ℝ
  /-- The length of side NB -/
  c : ℝ
  /-- x is positive (diameter) -/
  x_pos : 0 < x
  /-- a is positive (side length) -/
  a_pos : 0 < a
  /-- b is positive (side length) -/
  b_pos : 0 < b
  /-- c is positive (side length) -/
  c_pos : 0 < c
  /-- The sum of a, b, and c is less than or equal to x (semicircle property) -/
  sum_abc_le_x : a + b + c ≤ x

/-- The theorem stating the relationship between the sides of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_equation (q : InscribedQuadrilateral) :
  q.x^3 - (q.a^2 + q.b^2 + q.c^2) * q.x - 2 * q.a * q.b * q.c = 0 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_equation_l703_70378


namespace NUMINAMATH_CALUDE_child_worker_wage_l703_70371

def num_male : ℕ := 20
def num_female : ℕ := 15
def num_child : ℕ := 5
def wage_male : ℕ := 35
def wage_female : ℕ := 20
def average_wage : ℕ := 26

theorem child_worker_wage :
  ∃ (wage_child : ℕ),
    (num_male * wage_male + num_female * wage_female + num_child * wage_child) / 
    (num_male + num_female + num_child) = average_wage ∧
    wage_child = 8 := by
  sorry

end NUMINAMATH_CALUDE_child_worker_wage_l703_70371


namespace NUMINAMATH_CALUDE_system_solution_l703_70359

theorem system_solution (x y : ℝ) (eq1 : x + 5*y = 5) (eq2 : 3*x - y = 3) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l703_70359


namespace NUMINAMATH_CALUDE_house_sale_profit_l703_70358

/-- Calculates the final profit for Mr. A after three house sales --/
theorem house_sale_profit (initial_value : ℝ) (profit1 profit2 profit3 : ℝ) : 
  initial_value = 120000 ∧ 
  profit1 = 0.2 ∧ 
  profit2 = -0.15 ∧ 
  profit3 = 0.05 → 
  let sale1 := initial_value * (1 + profit1)
  let sale2 := sale1 * (1 + profit2)
  let sale3 := sale2 * (1 + profit3)
  (sale1 - sale2) + (sale3 - sale2) = 27720 := by
  sorry

#check house_sale_profit

end NUMINAMATH_CALUDE_house_sale_profit_l703_70358


namespace NUMINAMATH_CALUDE_inequality_condition_l703_70350

theorem inequality_condition (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_condition_l703_70350


namespace NUMINAMATH_CALUDE_binomial_10_9_l703_70313

theorem binomial_10_9 : (10 : ℕ).choose 9 = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_10_9_l703_70313


namespace NUMINAMATH_CALUDE_river_current_speed_l703_70398

/-- Given a boat's travel times and distances, calculates the current's speed -/
theorem river_current_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (upstream_time : ℝ) 
  (h1 : downstream_distance = 24) 
  (h2 : upstream_distance = 24) 
  (h3 : downstream_time = 4) 
  (h4 : upstream_time = 6) :
  ∃ (boat_speed current_speed : ℝ),
    boat_speed > 0 ∧ 
    (boat_speed + current_speed) * downstream_time = downstream_distance ∧
    (boat_speed - current_speed) * upstream_time = upstream_distance ∧
    current_speed = 1 := by
  sorry

end NUMINAMATH_CALUDE_river_current_speed_l703_70398


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l703_70323

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 1 > 0) ↔ (∃ x : ℝ, x^2 + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l703_70323


namespace NUMINAMATH_CALUDE_min_distance_to_origin_l703_70352

theorem min_distance_to_origin (x y : ℝ) : 
  (3 * x + y = 10) → (x^2 + y^2 ≥ 10) := by sorry

end NUMINAMATH_CALUDE_min_distance_to_origin_l703_70352


namespace NUMINAMATH_CALUDE_hyperbola_and_intersecting_line_l703_70360

/-- Given a hyperbola C with equation x²/a² - y²/b² = 1, real axis length 2√3, and one focus at (-√5, 0),
    prove its equation and find the equation of a line intersecting it. -/
theorem hyperbola_and_intersecting_line 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (real_axis_length : ℝ) 
  (focus : ℝ × ℝ) 
  (hreal_axis : real_axis_length = 2 * Real.sqrt 3) 
  (hfocus : focus = (-Real.sqrt 5, 0)) :
  (∃ (x y : ℝ), x^2 / 3 - y^2 / 2 = 1) ∧ 
  (∃ (m : ℝ), (m = Real.sqrt 210 / 3 ∨ m = -Real.sqrt 210 / 3) ∧
    ∀ (x y : ℝ), y = 2 * x + m → 
      (∃ (A B : ℝ × ℝ), A ≠ B ∧ 
        (A.1^2 / 3 - A.2^2 / 2 = 1) ∧ 
        (B.1^2 / 3 - B.2^2 / 2 = 1) ∧
        (A.2 = 2 * A.1 + m) ∧ 
        (B.2 = 2 * B.1 + m) ∧
        (A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_and_intersecting_line_l703_70360


namespace NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l703_70309

theorem quadratic_equation_unique_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 1/b) * x + c = 0) ↔ 
  (c = (5 + Real.sqrt 21) / 2 ∨ c = (5 - Real.sqrt 21) / 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_unique_solution_l703_70309


namespace NUMINAMATH_CALUDE_equation_solution_l703_70336

theorem equation_solution : 
  ∃ x : ℚ, (1 / 7 + 7 / x = 15 / x + 1 / 15) ∧ x = 105 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l703_70336


namespace NUMINAMATH_CALUDE_august_day_occurrences_l703_70388

/-- Represents days of the week -/
inductive Weekday
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Returns the next day of the week -/
def nextDay (d : Weekday) : Weekday :=
  match d with
  | Weekday.sunday => Weekday.monday
  | Weekday.monday => Weekday.tuesday
  | Weekday.tuesday => Weekday.wednesday
  | Weekday.wednesday => Weekday.thursday
  | Weekday.thursday => Weekday.friday
  | Weekday.friday => Weekday.saturday
  | Weekday.saturday => Weekday.sunday

/-- Counts occurrences of a specific day in a month -/
def countDayOccurrences (startDay : Weekday) (days : Nat) (targetDay : Weekday) : Nat :=
  sorry

theorem august_day_occurrences
  (july_start : Weekday)
  (july_days : Nat)
  (july_sundays : Nat)
  (august_days : Nat)
  (h1 : july_start = Weekday.saturday)
  (h2 : july_days = 31)
  (h3 : july_sundays = 5)
  (h4 : august_days = 31) :
  let august_start := (List.range july_days).foldl (fun d _ => nextDay d) july_start
  (countDayOccurrences august_start august_days Weekday.tuesday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.wednesday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.thursday = 5) ∧
  (countDayOccurrences august_start august_days Weekday.friday = 5) :=
by
  sorry


end NUMINAMATH_CALUDE_august_day_occurrences_l703_70388


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l703_70366

-- Define the monomial structure
structure Monomial where
  coefficient : ℚ
  x_exponent : ℕ
  y_exponent : ℕ

-- Define our specific monomial
def our_monomial : Monomial := {
  coefficient := -2/3,
  x_exponent := 1,
  y_exponent := 2
}

-- Theorem for the coefficient
theorem coefficient_of_monomial :
  our_monomial.coefficient = -2/3 := by sorry

-- Theorem for the degree
theorem degree_of_monomial :
  our_monomial.x_exponent + our_monomial.y_exponent = 3 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_degree_of_monomial_l703_70366


namespace NUMINAMATH_CALUDE_slope_angle_30_implies_m_equals_neg_sqrt3_l703_70320

/-- Given a line with equation x + my - 2 = 0 and slope angle 30°, m equals -√3 --/
theorem slope_angle_30_implies_m_equals_neg_sqrt3 (m : ℝ) : 
  (∃ x y, x + m * y - 2 = 0) →  -- Line equation
  (Real.tan (30 * π / 180) = -1 / m) →  -- Slope angle is 30°
  m = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_slope_angle_30_implies_m_equals_neg_sqrt3_l703_70320


namespace NUMINAMATH_CALUDE_simplify_expression_l703_70376

theorem simplify_expression :
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 252 / Real.sqrt 108) + (Real.sqrt 88 / Real.sqrt 22) = (21 + 2 * Real.sqrt 21) / 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l703_70376


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l703_70343

theorem purely_imaginary_complex_number (m : ℝ) : 
  (((m^2 - 5*m + 6) : ℂ) + (m^2 - 3*m)*I = (0 : ℂ) + ((m^2 - 3*m) : ℝ)*I) → 
  (m = 2 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l703_70343


namespace NUMINAMATH_CALUDE_trig_identity_degrees_l703_70353

theorem trig_identity_degrees : 
  Real.sin ((-1200 : ℝ) * π / 180) * Real.cos ((1290 : ℝ) * π / 180) + 
  Real.cos ((-1020 : ℝ) * π / 180) * Real.sin ((-1050 : ℝ) * π / 180) = 1 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_degrees_l703_70353


namespace NUMINAMATH_CALUDE_unique_minimum_condition_l703_70367

/-- The function f(x) = ax³ + e^x has a unique minimum value if and only if a is in the range [-e²/12, 0) --/
theorem unique_minimum_condition (a : ℝ) :
  (∃ x₀ : ℝ, ∀ x : ℝ, a * x^3 + Real.exp x ≥ a * x₀^3 + Real.exp x₀ ∧
    (a * x^3 + Real.exp x = a * x₀^3 + Real.exp x₀ → x = x₀)) ↔
  a ∈ Set.Icc (-(Real.exp 2 / 12)) 0 ∧ a ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_minimum_condition_l703_70367


namespace NUMINAMATH_CALUDE_problem_solution_l703_70380

theorem problem_solution (x y a b c d : ℝ) 
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c = -d) :
  (x + y)^3 - (-a*b)^2 + 3*c + 3*d = -2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l703_70380


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l703_70364

theorem president_vice_president_selection (n : ℕ) (h : n = 5) :
  (n * (n - 1) : ℕ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_president_vice_president_selection_l703_70364


namespace NUMINAMATH_CALUDE_equation_solution_l703_70375

theorem equation_solution : ∃ (x : ℝ), 45 - (28 - (37 - (x - 19))) = 58 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l703_70375


namespace NUMINAMATH_CALUDE_color_film_fraction_l703_70348

theorem color_film_fraction (x y : ℝ) (h : x ≠ 0) :
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (total_bw / 100)
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected) = 6 / 7 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l703_70348


namespace NUMINAMATH_CALUDE_percentage_relation_l703_70354

theorem percentage_relation (a b : ℝ) (h : a = 2 * b) : 4 * b = 2 * a := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l703_70354


namespace NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l703_70387

theorem inscribed_squares_area_ratio (r : ℝ) (r_pos : r > 0) : 
  let s1 := r / Real.sqrt 2
  let s2 := r * Real.sqrt 2
  (s1 ^ 2) / (s2 ^ 2) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_inscribed_squares_area_ratio_l703_70387


namespace NUMINAMATH_CALUDE_ratio_sum_l703_70379

theorem ratio_sum (a b c d : ℝ) : 
  a / b = 2 / 3 ∧ 
  b / c = 3 / 4 ∧ 
  c / d = 4 / 5 ∧ 
  d = 672 → 
  a + b + c + d = 1881.6 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_l703_70379


namespace NUMINAMATH_CALUDE_combined_value_l703_70397

def sum_even (a b : ℕ) : ℕ := 
  (b - a + 2) / 2 * (a + b) / 2

def sum_odd (a b : ℕ) : ℕ := 
  ((b - a) / 2 + 1) * (a + b) / 2

def i : ℕ := sum_even 2 500
def k : ℕ := sum_even 8 200
def j : ℕ := sum_odd 5 133

theorem combined_value : 2 * i - k + 3 * j = 128867 := by sorry

end NUMINAMATH_CALUDE_combined_value_l703_70397


namespace NUMINAMATH_CALUDE_simplify_expression_l703_70370

theorem simplify_expression (z : ℝ) : (4 - 5 * z^2) - (2 + 7 * z^2 - z) = 2 - 12 * z^2 + z := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l703_70370


namespace NUMINAMATH_CALUDE_sixth_result_l703_70365

theorem sixth_result (total_results : ℕ) (all_average first_six_average last_six_average : ℚ) :
  total_results = 11 →
  all_average = 52 →
  first_six_average = 49 →
  last_six_average = 52 →
  ∃ (sixth_result : ℚ),
    sixth_result = 34 ∧
    (6 * first_six_average - sixth_result) + sixth_result + (6 * last_six_average - sixth_result) = total_results * all_average :=
by sorry

end NUMINAMATH_CALUDE_sixth_result_l703_70365


namespace NUMINAMATH_CALUDE_dan_makes_fifteen_tshirts_l703_70384

/-- The number of t-shirts Dan makes in two hours -/
def tshirts_made (minutes_per_hour : ℕ) (rate_hour1 : ℕ) (rate_hour2 : ℕ) : ℕ :=
  (minutes_per_hour / rate_hour1) + (minutes_per_hour / rate_hour2)

/-- Proof that Dan makes 15 t-shirts in two hours -/
theorem dan_makes_fifteen_tshirts :
  tshirts_made 60 12 6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_dan_makes_fifteen_tshirts_l703_70384


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l703_70361

/-- The line x + y = 0 is tangent to the circle (x-a)² + (y-b)² = 2 -/
def is_tangent (a b : ℝ) : Prop :=
  (a + b = 2) ∨ (a + b = -2)

/-- a + b = 2 is a sufficient condition for the line to be tangent to the circle -/
theorem sufficient_condition (a b : ℝ) :
  a + b = 2 → is_tangent a b :=
sorry

/-- a + b = 2 is not a necessary condition for the line to be tangent to the circle -/
theorem not_necessary_condition :
  ∃ a b, is_tangent a b ∧ a + b ≠ 2 :=
sorry

/-- a + b = 2 is a sufficient but not necessary condition for the line to be tangent to the circle -/
theorem sufficient_but_not_necessary :
  (∀ a b, a + b = 2 → is_tangent a b) ∧
  (∃ a b, is_tangent a b ∧ a + b ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l703_70361


namespace NUMINAMATH_CALUDE_number_puzzle_solution_l703_70321

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem number_puzzle_solution :
  ∃ (a b : ℕ),
    1 ≤ a ∧ a ≤ 60 ∧
    1 ≤ b ∧ b ≤ 60 ∧
    a ≠ b ∧
    ∀ k : ℕ, k < 5 → ¬((a + b) % k = 0) ∧
    is_prime b ∧
    b > 10 ∧
    ∃ (m : ℕ), 150 * b + a = m * m ∧
    a + b = 42 :=
by sorry

end NUMINAMATH_CALUDE_number_puzzle_solution_l703_70321


namespace NUMINAMATH_CALUDE_original_cost_price_calculation_l703_70328

/-- Represents the pricing structure of an article -/
structure ArticlePricing where
  cost_price : ℝ
  discount_rate : ℝ
  tax_rate : ℝ
  profit_rate : ℝ
  selling_price : ℝ

/-- Theorem stating the relationship between the original cost price and final selling price -/
theorem original_cost_price_calculation (a : ArticlePricing)
  (h1 : a.discount_rate = 0.10)
  (h2 : a.tax_rate = 0.05)
  (h3 : a.profit_rate = 0.20)
  (h4 : a.selling_price = 1800)
  : a.cost_price = 1500 := by
  sorry

#check original_cost_price_calculation

end NUMINAMATH_CALUDE_original_cost_price_calculation_l703_70328


namespace NUMINAMATH_CALUDE_cube_inscribed_in_sphere_surface_area_l703_70304

/-- The surface area of a cube inscribed in a sphere with radius 5 units is 200 square units. -/
theorem cube_inscribed_in_sphere_surface_area :
  let r : ℝ := 5  -- radius of the sphere
  let s : ℝ := 10 * Real.sqrt 3 / 3  -- edge length of the cube
  6 * s^2 = 200 := by sorry

end NUMINAMATH_CALUDE_cube_inscribed_in_sphere_surface_area_l703_70304


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l703_70315

theorem quadratic_form_sum (k : ℝ) : ∃ (d r s : ℝ),
  (8 * k^2 + 12 * k + 18 = d * (k + r)^2 + s) ∧ (r + s = 57 / 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l703_70315


namespace NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l703_70390

theorem short_bingo_first_column_possibilities : Fintype.card { p : Fin 8 → Fin 4 | Function.Injective p } = 1680 := by
  sorry

end NUMINAMATH_CALUDE_short_bingo_first_column_possibilities_l703_70390


namespace NUMINAMATH_CALUDE_julia_short_amount_l703_70331

def rock_price : ℚ := 7
def pop_price : ℚ := 12
def dance_price : ℚ := 5
def country_price : ℚ := 9

def discount_rate : ℚ := 0.15
def discount_threshold : ℕ := 3

def rock_desired : ℕ := 5
def pop_desired : ℕ := 3
def dance_desired : ℕ := 6
def country_desired : ℕ := 4

def rock_available : ℕ := 4
def dance_available : ℕ := 5

def julia_budget : ℚ := 80

def calculate_genre_cost (price : ℚ) (desired : ℕ) (available : ℕ) : ℚ :=
  price * (min desired available : ℚ)

def apply_discount (cost : ℚ) (quantity : ℕ) : ℚ :=
  if quantity ≥ discount_threshold then cost * (1 - discount_rate) else cost

theorem julia_short_amount : 
  let rock_cost := calculate_genre_cost rock_price rock_desired rock_available
  let pop_cost := calculate_genre_cost pop_price pop_desired pop_desired
  let dance_cost := calculate_genre_cost dance_price dance_desired dance_available
  let country_cost := calculate_genre_cost country_price country_desired country_desired
  let total_cost := rock_cost + pop_cost + dance_cost + country_cost
  let discounted_rock := apply_discount rock_cost rock_available
  let discounted_pop := apply_discount pop_cost pop_desired
  let discounted_dance := apply_discount dance_cost dance_available
  let discounted_country := apply_discount country_cost country_desired
  let total_discounted := discounted_rock + discounted_pop + discounted_dance + discounted_country
  total_discounted - julia_budget = 26.25 := by
  sorry

end NUMINAMATH_CALUDE_julia_short_amount_l703_70331


namespace NUMINAMATH_CALUDE_equation_describes_two_lines_l703_70340

theorem equation_describes_two_lines :
  ∀ x y : ℝ, (x - y)^2 = x^2 - y^2 ↔ x * y = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_describes_two_lines_l703_70340


namespace NUMINAMATH_CALUDE_counterexample_exists_l703_70351

theorem counterexample_exists : ∃ n : ℕ, 
  ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n + 1)) ∧ ¬(Nat.Prime (n - 2)) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l703_70351


namespace NUMINAMATH_CALUDE_function_passes_through_point_l703_70303

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f := fun x => a^(x - 1) + 3
  f 1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l703_70303


namespace NUMINAMATH_CALUDE_first_last_checkpoint_distance_l703_70301

-- Define the marathon parameters
def marathon_length : ℝ := 26
def num_checkpoints : ℕ := 4
def distance_between_checkpoints : ℝ := 6

-- Theorem statement
theorem first_last_checkpoint_distance :
  let total_checkpoint_distance := (num_checkpoints - 1 : ℝ) * distance_between_checkpoints
  let remaining_distance := marathon_length - total_checkpoint_distance
  let first_last_distance := remaining_distance / 2
  first_last_distance = 1 := by sorry

end NUMINAMATH_CALUDE_first_last_checkpoint_distance_l703_70301


namespace NUMINAMATH_CALUDE_ball_selection_problem_l703_70341

/-- The number of ways to select balls from a bag with red and white balls -/
def select_balls (red : ℕ) (white : ℕ) (total : ℕ) (condition : ℕ → ℕ → Bool) : ℕ :=
  sorry

/-- The total score of selected balls -/
def total_score (red : ℕ) (white : ℕ) : ℕ :=
  sorry

theorem ball_selection_problem :
  let red_balls := 4
  let white_balls := 6
  (select_balls red_balls white_balls 4 (fun r w => r ≥ w) = 115) ∧
  (select_balls red_balls white_balls 5 (fun r w => total_score r w ≥ 7) = 186) :=
by sorry

end NUMINAMATH_CALUDE_ball_selection_problem_l703_70341


namespace NUMINAMATH_CALUDE_tangent_intersection_l703_70349

/-- The function f(x) = x^3 - x^2 + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x + 1

/-- The derivative of f(x) -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*x + a

theorem tangent_intersection (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), 
    (x₁ = 1 ∧ f a x₁ = a + 1) ∧
    (x₂ = -1 ∧ f a x₂ = -a - 1) ∧
    (∀ x : ℝ, f a x = (f' a x₁) * x ↔ x = x₁ ∨ x = x₂) := by
  sorry

end NUMINAMATH_CALUDE_tangent_intersection_l703_70349


namespace NUMINAMATH_CALUDE_only_yes_allows_deduction_l703_70330

/-- Represents the three types of natives on the island --/
inductive NativeType
  | Normal
  | Zombie
  | HalfZombie

/-- Represents possible answers in the native language --/
inductive Answer
  | Yes
  | No
  | Bal

/-- Function to determine if a native tells the truth based on their type and the question number --/
def tellsTruth (t : NativeType) (questionNumber : Nat) : Bool :=
  match t with
  | NativeType.Normal => true
  | NativeType.Zombie => false
  | NativeType.HalfZombie => questionNumber % 2 = 0

/-- The complex question asked by Inspector Craig --/
def inspectorQuestion (a : Answer) : Prop :=
  ∃ (t : NativeType), tellsTruth t 1 = (a = Answer.Yes)

/-- Theorem stating that "Yes" is the only answer that allows deduction of native type --/
theorem only_yes_allows_deduction :
  ∃! (a : Answer), ∀ (t : NativeType), inspectorQuestion a ↔ t = NativeType.HalfZombie :=
sorry


end NUMINAMATH_CALUDE_only_yes_allows_deduction_l703_70330


namespace NUMINAMATH_CALUDE_chooseBoxes_eq_sixteen_l703_70339

/-- The number of ways to choose 3 out of 6 boxes with at least one of A or B chosen -/
def chooseBoxes : ℕ := sorry

/-- There are 6 boxes in total -/
def totalBoxes : ℕ := 6

/-- The number of boxes to be chosen -/
def boxesToChoose : ℕ := 3

/-- The theorem stating that the number of ways to choose 3 out of 6 boxes 
    with at least one of A or B chosen is 16 -/
theorem chooseBoxes_eq_sixteen : chooseBoxes = 16 := by sorry

end NUMINAMATH_CALUDE_chooseBoxes_eq_sixteen_l703_70339


namespace NUMINAMATH_CALUDE_vertex_of_our_parabola_l703_70334

/-- Represents a parabola in the form y = a(x-h)^2 + k -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := (p.h, p.k)

/-- Our specific parabola -/
def our_parabola : Parabola := { a := -3, h := 1, k := 2 }

/-- Theorem: The vertex of our parabola is (1,2) -/
theorem vertex_of_our_parabola : vertex our_parabola = (1, 2) := by sorry

end NUMINAMATH_CALUDE_vertex_of_our_parabola_l703_70334


namespace NUMINAMATH_CALUDE_equality_from_divisibility_l703_70300

theorem equality_from_divisibility (a b : ℕ+) (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : a = b := by
  sorry

end NUMINAMATH_CALUDE_equality_from_divisibility_l703_70300


namespace NUMINAMATH_CALUDE_quadratic_minimum_l703_70337

theorem quadratic_minimum (x : ℝ) : x^2 + 6*x + 3 ≥ -6 ∧ ∃ y : ℝ, y^2 + 6*y + 3 = -6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l703_70337


namespace NUMINAMATH_CALUDE_unique_digit_A_l703_70318

def base5ToDecimal (a : ℕ) : ℕ := 25 + 6 * a

def base6ToDecimal (a : ℕ) : ℕ := 36 + 7 * a

def isPerfectSquare (n : ℕ) : Prop := ∃ x : ℕ, x * x = n

def isPerfectCube (n : ℕ) : Prop := ∃ y : ℕ, y * y * y = n

theorem unique_digit_A : 
  ∃! a : ℕ, a ≤ 4 ∧ 
    isPerfectSquare (base5ToDecimal a) ∧ 
    isPerfectCube (base6ToDecimal a) :=
by sorry

end NUMINAMATH_CALUDE_unique_digit_A_l703_70318


namespace NUMINAMATH_CALUDE_savings_account_relationship_l703_70396

/-- The function representing the total amount in an education savings account -/
def savings_account (monthly_rate : ℝ) (initial_deposit : ℝ) (months : ℝ) : ℝ :=
  monthly_rate * initial_deposit * months + initial_deposit

/-- Theorem stating the relationship between total amount and number of months -/
theorem savings_account_relationship :
  let monthly_rate : ℝ := 0.0022  -- 0.22%
  let initial_deposit : ℝ := 1000
  ∀ x : ℝ, savings_account monthly_rate initial_deposit x = 2.2 * x + 1000 := by
  sorry

end NUMINAMATH_CALUDE_savings_account_relationship_l703_70396


namespace NUMINAMATH_CALUDE_quadratic_unique_solution_l703_70326

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →  -- exactly one solution
  (a + c = 12) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11) := by
sorry

end NUMINAMATH_CALUDE_quadratic_unique_solution_l703_70326


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l703_70382

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l703_70382


namespace NUMINAMATH_CALUDE_factorization_problems_l703_70391

theorem factorization_problems (x y : ℝ) :
  (x^2 - 4 = (x + 2) * (x - 2)) ∧
  (3 * x^2 - 6 * x * y + 3 * y^2 = 3 * (x - y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l703_70391


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocal_product_l703_70373

theorem min_value_sum_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b) * (1 / a + 1 / b) ≥ 4 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1 / x + 1 / y) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocal_product_l703_70373


namespace NUMINAMATH_CALUDE_crystal_mass_ratio_l703_70324

theorem crystal_mass_ratio : 
  ∀ (m1 m2 : ℝ), -- initial masses of crystals 1 and 2
  ∀ (r1 r2 : ℝ), -- yearly growth rates of crystals 1 and 2
  r1 > 0 ∧ r2 > 0 → -- growth rates are positive
  (3 * r1 * m1 = 7 * r2 * m2) → -- condition on 3-month and 7-month growth
  (r1 = 0.04) → -- 4% yearly growth for crystal 1
  (r2 = 0.05) → -- 5% yearly growth for crystal 2
  (m1 / m2 = 35 / 12) := by
sorry

end NUMINAMATH_CALUDE_crystal_mass_ratio_l703_70324


namespace NUMINAMATH_CALUDE_intersection_segment_length_l703_70333

/-- Line l in the Cartesian coordinate system -/
def line_l (x y : ℝ) : Prop := x + y = 3

/-- Curve C in the Cartesian coordinate system -/
def curve_C (x y : ℝ) : Prop := y = (x - 3)^2

/-- The intersection points of line l and curve C -/
def intersection_points : Set (ℝ × ℝ) :=
  {p | line_l p.1 p.2 ∧ curve_C p.1 p.2}

/-- Theorem stating that the length of the line segment between 
    the intersection points of line l and curve C is √2 -/
theorem intersection_segment_length : 
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ 
  A ≠ B ∧ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_segment_length_l703_70333
