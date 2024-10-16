import Mathlib

namespace NUMINAMATH_CALUDE_gcd_inequality_l3413_341326

theorem gcd_inequality (n : ℕ) :
  (∀ k ∈ Finset.range 34, Nat.gcd n (n + k) < Nat.gcd n (n + k + 1)) →
  Nat.gcd n (n + 35) < Nat.gcd n (n + 36) := by
sorry

end NUMINAMATH_CALUDE_gcd_inequality_l3413_341326


namespace NUMINAMATH_CALUDE_fraction_equals_44_l3413_341393

theorem fraction_equals_44 : (2450 - 2377)^2 / 121 = 44 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_44_l3413_341393


namespace NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3413_341352

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

theorem cryptarithmetic_puzzle :
  ∀ (I X E L V : ℕ),
    (I < 10) →
    (X < 10) →
    (E < 10) →
    (L < 10) →
    (V < 10) →
    (is_odd X) →
    (I ≠ 7) →
    (X ≠ 7) →
    (E ≠ 7) →
    (L ≠ 7) →
    (V ≠ 7) →
    (I ≠ X) →
    (I ≠ E) →
    (I ≠ L) →
    (I ≠ V) →
    (X ≠ E) →
    (X ≠ L) →
    (X ≠ V) →
    (E ≠ L) →
    (E ≠ V) →
    (L ≠ V) →
    (700 + 10*I + X + 700 + 10*I + X = 1000*E + 100*L + 10*E + V) →
    (I = 2) :=
by sorry

end NUMINAMATH_CALUDE_cryptarithmetic_puzzle_l3413_341352


namespace NUMINAMATH_CALUDE_trig_simplification_l3413_341369

theorem trig_simplification :
  (Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + Real.sin (60 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.sin (30 * π / 180)) =
  8 * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l3413_341369


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3413_341304

/-- Represents a circular arrangement of numbers from 1 to 60 -/
def CircularArrangement := Fin 60 → ℕ

/-- Checks if the sum of two numbers with k numbers between them is divisible by n -/
def SatisfiesDivisibilityCondition (arr : CircularArrangement) (k n : ℕ) : Prop :=
  ∀ i : Fin 60, (arr i + arr ((i + k + 1) % 60)) % n = 0

/-- Checks if the arrangement satisfies all given conditions -/
def SatisfiesAllConditions (arr : CircularArrangement) : Prop :=
  (∀ i : Fin 60, arr i ∈ Finset.range 60) ∧ 
  (Finset.card (Finset.image arr Finset.univ) = 60) ∧
  SatisfiesDivisibilityCondition arr 1 2 ∧
  SatisfiesDivisibilityCondition arr 2 3 ∧
  SatisfiesDivisibilityCondition arr 6 7

theorem no_valid_arrangement : ¬ ∃ arr : CircularArrangement, SatisfiesAllConditions arr := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3413_341304


namespace NUMINAMATH_CALUDE_systematic_sampling_most_suitable_for_C_l3413_341364

/-- Characteristics of systematic sampling -/
structure SystematicSampling where
  large_population : Bool
  regular_interval : Bool
  balanced_group : Bool

/-- Sampling scenario -/
structure SamplingScenario where
  population_size : Nat
  sample_size : Nat
  is_homogeneous : Bool

/-- Check if a scenario is suitable for systematic sampling -/
def is_suitable_for_systematic_sampling (scenario : SamplingScenario) : Bool :=
  scenario.population_size > scenario.sample_size ∧ 
  scenario.population_size ≥ 1000 ∧ 
  scenario.sample_size ≥ 100 ∧
  scenario.is_homogeneous

/-- The four sampling scenarios -/
def scenario_A : SamplingScenario := ⟨2000, 200, false⟩
def scenario_B : SamplingScenario := ⟨2000, 5, true⟩
def scenario_C : SamplingScenario := ⟨2000, 200, true⟩
def scenario_D : SamplingScenario := ⟨20, 5, true⟩

theorem systematic_sampling_most_suitable_for_C :
  is_suitable_for_systematic_sampling scenario_C ∧
  ¬is_suitable_for_systematic_sampling scenario_A ∧
  ¬is_suitable_for_systematic_sampling scenario_B ∧
  ¬is_suitable_for_systematic_sampling scenario_D :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_most_suitable_for_C_l3413_341364


namespace NUMINAMATH_CALUDE_angle_c_measure_l3413_341307

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)

-- Define the properties of the isosceles triangle
def IsIsosceles (t : Triangle) : Prop :=
  t.A = t.B

-- Define the relationship between angles A and C
def AngleCRelation (t : Triangle) : Prop :=
  t.C = t.A + 30

-- Define the sum of angles in a triangle
def AngleSum (t : Triangle) : Prop :=
  t.A + t.B + t.C = 180

-- Theorem statement
theorem angle_c_measure (t : Triangle) 
  (h1 : IsIsosceles t) 
  (h2 : AngleCRelation t) 
  (h3 : AngleSum t) : 
  t.C = 80 := by
    sorry

end NUMINAMATH_CALUDE_angle_c_measure_l3413_341307


namespace NUMINAMATH_CALUDE_intersection_M_complement_N_l3413_341350

def R : Set ℝ := Set.univ

def M : Set ℝ := {-1, 0, 1, 5}

def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_M_complement_N : M ∩ (R \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_complement_N_l3413_341350


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l3413_341333

theorem gcd_of_squares_sum : Nat.gcd (125^2 + 235^2 + 349^2) (124^2 + 234^2 + 350^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l3413_341333


namespace NUMINAMATH_CALUDE_nested_bracket_evaluation_l3413_341310

def bracket (a b c : ℚ) : ℚ := (a + b) / c

theorem nested_bracket_evaluation :
  let outer_bracket := bracket (bracket 72 36 108) (bracket 4 2 6) (bracket 12 6 18)
  outer_bracket = 2 := by sorry

end NUMINAMATH_CALUDE_nested_bracket_evaluation_l3413_341310


namespace NUMINAMATH_CALUDE_marble_probability_difference_l3413_341353

/-- The number of red marbles in the box -/
def red_marbles : ℕ := 1101

/-- The number of black marbles in the box -/
def black_marbles : ℕ := 1101

/-- The total number of marbles in the box -/
def total_marbles : ℕ := red_marbles + black_marbles

/-- The probability of drawing two marbles of the same color -/
def P_s : ℚ := (Nat.choose red_marbles 2 + Nat.choose black_marbles 2) / Nat.choose total_marbles 2

/-- The probability of drawing two marbles of different colors -/
def P_d : ℚ := (red_marbles * black_marbles) / Nat.choose total_marbles 2

/-- The theorem stating that the absolute difference between P_s and P_d is 1/2201 -/
theorem marble_probability_difference : |P_s - P_d| = 1 / 2201 := by sorry

end NUMINAMATH_CALUDE_marble_probability_difference_l3413_341353


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3413_341357

-- Define the sets M and N
def M : Set ℝ := {x | x + 1 > 0}
def N : Set ℝ := {x | 2*x - 1 < 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = Set.Ioo (-1 : ℝ) (1/2 : ℝ) := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3413_341357


namespace NUMINAMATH_CALUDE_jesse_pencils_l3413_341337

theorem jesse_pencils (initial : ℕ) (given : ℕ) (remaining : ℕ) : 
  given = 44 → remaining = 34 → initial = given + remaining :=
by
  sorry

end NUMINAMATH_CALUDE_jesse_pencils_l3413_341337


namespace NUMINAMATH_CALUDE_equation_solution_l3413_341376

theorem equation_solution : ∃ x : ℕ, 9^12 + 9^12 + 9^12 = 3^x ∧ x = 25 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3413_341376


namespace NUMINAMATH_CALUDE_sara_gave_four_limes_l3413_341367

def limes_from_sara (initial_limes final_limes : ℕ) : ℕ :=
  final_limes - initial_limes

theorem sara_gave_four_limes (initial_limes final_limes : ℕ) 
  (h1 : initial_limes = 9)
  (h2 : final_limes = 13) :
  limes_from_sara initial_limes final_limes = 4 := by
  sorry

end NUMINAMATH_CALUDE_sara_gave_four_limes_l3413_341367


namespace NUMINAMATH_CALUDE_projection_onto_yOz_plane_l3413_341356

-- Define the types for points and vectors in 3D space
def Point3D := ℝ × ℝ × ℝ
def Vector3D := ℝ × ℝ × ℝ

-- Define the projection onto the yOz plane
def projectOntoYOZ (p : Point3D) : Point3D :=
  (0, p.2.1, p.2.2)

-- Define the vector from origin to a point
def vectorFromOrigin (p : Point3D) : Vector3D := p

-- Theorem statement
theorem projection_onto_yOz_plane (A : Point3D) (h : A = (1, 6, 2)) :
  vectorFromOrigin (projectOntoYOZ A) = (0, 6, 2) := by
  sorry

end NUMINAMATH_CALUDE_projection_onto_yOz_plane_l3413_341356


namespace NUMINAMATH_CALUDE_starting_lineup_theorem_l3413_341302

def total_players : ℕ := 18
def goalie_count : ℕ := 1
def regular_players_count : ℕ := 10
def captain_count : ℕ := 1

def starting_lineup_count : ℕ :=
  total_players * (Nat.choose (total_players - goalie_count) regular_players_count) * regular_players_count

theorem starting_lineup_theorem :
  starting_lineup_count = 34928640 := by sorry

end NUMINAMATH_CALUDE_starting_lineup_theorem_l3413_341302


namespace NUMINAMATH_CALUDE_smallest_n_with_non_decimal_digit_in_g_l3413_341342

/-- Sum of digits in base-three representation -/
def f (n : ℕ) : ℕ := sorry

/-- Sum of digits in base-six representation of f(n) -/
def g (n : ℕ) : ℕ := sorry

/-- Check if a number in base-twelve contains a digit not in {0, 1, ..., 9} -/
def has_non_decimal_digit (n : ℕ) : Prop := sorry

/-- The main theorem -/
theorem smallest_n_with_non_decimal_digit_in_g : 
  (∃ n : ℕ, n > 0 ∧ has_non_decimal_digit (g n)) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < 32 → ¬has_non_decimal_digit (g m)) ∧
  has_non_decimal_digit (g 32) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_non_decimal_digit_in_g_l3413_341342


namespace NUMINAMATH_CALUDE_second_number_is_204_l3413_341383

def number_list : List ℕ := [201, 204, 205, 206, 209, 209, 210, 212, 212]

theorem second_number_is_204 : number_list[1] = 204 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_204_l3413_341383


namespace NUMINAMATH_CALUDE_economic_loss_scientific_notation_l3413_341315

-- Define the original number in millions
def original_number : ℝ := 16823

-- Define the scientific notation components
def coefficient : ℝ := 1.6823
def exponent : ℤ := 4

-- Theorem statement
theorem economic_loss_scientific_notation :
  original_number = coefficient * (10 : ℝ) ^ exponent :=
sorry

end NUMINAMATH_CALUDE_economic_loss_scientific_notation_l3413_341315


namespace NUMINAMATH_CALUDE_exists_sequence_to_1981_no_sequence_to_1982_l3413_341336

-- Define the machine operations
def multiply_by_3 (n : ℕ) : ℕ := 3 * n
def add_4 (n : ℕ) : ℕ := n + 4

-- Define a sequence of operations
inductive Operation
| Mult3 : Operation
| Add4 : Operation

def apply_operation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.Mult3 => multiply_by_3 n
  | Operation.Add4 => add_4 n

def apply_sequence (start : ℕ) (ops : List Operation) : ℕ :=
  ops.foldl apply_operation start

-- Theorem statements
theorem exists_sequence_to_1981 :
  ∃ (ops : List Operation), apply_sequence 1 ops = 1981 :=
sorry

theorem no_sequence_to_1982 :
  ¬∃ (ops : List Operation), apply_sequence 1 ops = 1982 :=
sorry

end NUMINAMATH_CALUDE_exists_sequence_to_1981_no_sequence_to_1982_l3413_341336


namespace NUMINAMATH_CALUDE_xyz_divides_product_l3413_341373

/-- A proposition stating that if x, y, and z are distinct positive integers
    such that xyz divides (xy-1)(yz-1)(zx-1), then (x, y, z) is a permutation of (2, 3, 5) -/
theorem xyz_divides_product (x y z : ℕ) : 
  x > 0 → y > 0 → z > 0 → 
  x ≠ y → y ≠ z → x ≠ z →
  (x * y * z) ∣ ((x * y - 1) * (y * z - 1) * (z * x - 1)) →
  (x = 2 ∧ y = 3 ∧ z = 5) ∨ 
  (x = 2 ∧ y = 5 ∧ z = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 5) ∨ 
  (x = 3 ∧ y = 5 ∧ z = 2) ∨ 
  (x = 5 ∧ y = 2 ∧ z = 3) ∨ 
  (x = 5 ∧ y = 3 ∧ z = 2) := by
  sorry

#check xyz_divides_product

end NUMINAMATH_CALUDE_xyz_divides_product_l3413_341373


namespace NUMINAMATH_CALUDE_orvin_max_balloons_l3413_341391

/-- Represents the maximum number of balloons Orvin can buy given his budget and the sale conditions -/
def max_balloons (regular_price_budget : ℕ) (full_price_ratio : ℕ) (discount_ratio : ℕ) : ℕ :=
  let sets := (regular_price_budget * full_price_ratio) / (full_price_ratio + discount_ratio)
  sets * 2

/-- Proves that Orvin can buy at most 52 balloons given the specified conditions -/
theorem orvin_max_balloons :
  max_balloons 40 2 1 = 52 := by
  sorry

#eval max_balloons 40 2 1

end NUMINAMATH_CALUDE_orvin_max_balloons_l3413_341391


namespace NUMINAMATH_CALUDE_angle_conversion_l3413_341370

theorem angle_conversion (angle : Real) : ∃ (α k : Real), 
  angle * (π / 180) = α + 2 * k * π ∧ 
  0 ≤ α ∧ α < 2 * π ∧ 
  α = 7 * π / 4 ∧
  k = -10 := by
  sorry

end NUMINAMATH_CALUDE_angle_conversion_l3413_341370


namespace NUMINAMATH_CALUDE_simple_interest_rate_proof_l3413_341345

/-- Given a principal amount and a simple interest rate, 
    if the amount becomes 7/6 of itself after 4 years, 
    then the rate is 1/24 -/
theorem simple_interest_rate_proof 
  (P : ℝ) (R : ℝ) (P_pos : P > 0) :
  P * (1 + 4 * R) = 7/6 * P → R = 1/24 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_proof_l3413_341345


namespace NUMINAMATH_CALUDE_angle_sum_at_point_l3413_341318

theorem angle_sum_at_point (x : ℝ) : 
  (120 : ℝ) + x + x + 2*x = 360 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_at_point_l3413_341318


namespace NUMINAMATH_CALUDE_volunteer_selection_theorem_l3413_341390

def male_students : ℕ := 5
def female_students : ℕ := 4
def total_volunteers : ℕ := 3
def schools : ℕ := 3

def selection_plans : ℕ := 420

theorem volunteer_selection_theorem :
  (male_students.choose 2 * female_students.choose 1 +
   male_students.choose 1 * female_students.choose 2) * schools.factorial = selection_plans :=
by sorry

end NUMINAMATH_CALUDE_volunteer_selection_theorem_l3413_341390


namespace NUMINAMATH_CALUDE_california_permutations_count_l3413_341347

/-- The number of distinct permutations of a word with repeated letters -/
def wordPermutations (totalLetters : ℕ) (repeatedLetters : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repeatedLetters.map Nat.factorial).prod

/-- The number of distinct permutations of CALIFORNIA -/
def californiaPermutations : ℕ := wordPermutations 10 [3, 2]

theorem california_permutations_count :
  californiaPermutations = 302400 := by
  sorry

end NUMINAMATH_CALUDE_california_permutations_count_l3413_341347


namespace NUMINAMATH_CALUDE_monotonic_function_a_range_l3413_341344

/-- Given that f(x) = ln x + a/x is monotonically increasing on [2, +∞), 
    prove that the range of values for a is (-∞, 2] -/
theorem monotonic_function_a_range (a : ℝ) :
  (∀ x ≥ 2, Monotone (fun x => Real.log x + a / x)) ↔ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_monotonic_function_a_range_l3413_341344


namespace NUMINAMATH_CALUDE_inequality_solution_l3413_341360

theorem inequality_solution (x : ℝ) : 3 * x - 6 > 5 * (x - 2) → x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3413_341360


namespace NUMINAMATH_CALUDE_scott_total_oranges_l3413_341332

/-- The number of boxes Scott has for oranges. -/
def num_boxes : ℕ := 8

/-- The number of oranges that must be in each box. -/
def oranges_per_box : ℕ := 7

/-- Theorem stating that Scott has 56 oranges in total. -/
theorem scott_total_oranges : num_boxes * oranges_per_box = 56 := by
  sorry

end NUMINAMATH_CALUDE_scott_total_oranges_l3413_341332


namespace NUMINAMATH_CALUDE_jessica_money_l3413_341385

theorem jessica_money (rodney ian jessica : ℕ) 
  (h1 : rodney = ian + 35)
  (h2 : ian = jessica / 2)
  (h3 : jessica = rodney + 15) : 
  jessica = 100 := by
sorry

end NUMINAMATH_CALUDE_jessica_money_l3413_341385


namespace NUMINAMATH_CALUDE_subjective_collection_not_set_l3413_341355

-- Define a type for objects in a textbook
structure TextbookObject where
  id : Nat

-- Define a property that determines if an object belongs to a collection
def belongsToCollection (P : TextbookObject → Prop) (obj : TextbookObject) : Prop :=
  P obj

-- Define what it means for a collection to have a clear, objective criterion
def hasClearCriterion (P : TextbookObject → Prop) : Prop :=
  ∀ (obj1 obj2 : TextbookObject), obj1 = obj2 → (P obj1 ↔ P obj2)

-- Define what it means for a collection to be subjective
def isSubjective (P : TextbookObject → Prop) : Prop :=
  ∃ (obj1 obj2 : TextbookObject), obj1 = obj2 ∧ (P obj1 ↔ ¬P obj2)

-- Theorem: A collection with subjective criteria cannot form a well-defined set
theorem subjective_collection_not_set (P : TextbookObject → Prop) :
  isSubjective P → ¬(hasClearCriterion P) :=
by
  sorry

#check subjective_collection_not_set

end NUMINAMATH_CALUDE_subjective_collection_not_set_l3413_341355


namespace NUMINAMATH_CALUDE_opposite_numbers_expression_l3413_341394

theorem opposite_numbers_expression (m n : ℝ) (h : m + n = 0) :
  3 * (m - n) - (1/2) * (2 * m - 10 * n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_expression_l3413_341394


namespace NUMINAMATH_CALUDE_benny_picked_two_apples_l3413_341306

/-- The number of apples Dan picked -/
def dan_apples : ℕ := 9

/-- The difference between the number of apples Dan and Benny picked -/
def difference : ℕ := 7

/-- The number of apples Benny picked -/
def benny_apples : ℕ := dan_apples - difference

theorem benny_picked_two_apples : benny_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_benny_picked_two_apples_l3413_341306


namespace NUMINAMATH_CALUDE_problem_statement_l3413_341389

theorem problem_statement (x : ℝ) (h : x + 1/x = 6) :
  (x - 3)^2 + 36/((x - 3)^2) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3413_341389


namespace NUMINAMATH_CALUDE_apex_at_vertex_a_l3413_341366

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangle with pillars -/
structure TriangleWithPillars where
  A : Point3D
  B : Point3D
  C : Point3D
  heightA : ℝ
  heightB : ℝ
  heightC : ℝ

/-- Check if three points form an equilateral triangle on the ground (z = 0) -/
def isEquilateral (t : TriangleWithPillars) : Prop :=
  let d1 := (t.A.x - t.B.x)^2 + (t.A.y - t.B.y)^2
  let d2 := (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2
  let d3 := (t.C.x - t.A.x)^2 + (t.C.y - t.A.y)^2
  d1 = d2 ∧ d2 = d3 ∧ t.A.z = 0 ∧ t.B.z = 0 ∧ t.C.z = 0

/-- Find the point directly below the apex of the inclined plane -/
def apexProjection (t : TriangleWithPillars) : Point3D :=
  { x := t.A.x, y := t.A.y, z := 0 }

/-- Theorem: The apex projection is at vertex A for the given triangle -/
theorem apex_at_vertex_a (t : TriangleWithPillars) :
  isEquilateral t ∧ t.heightA = 10 ∧ t.heightB = 8 ∧ t.heightC = 6 →
  apexProjection t = t.A :=
by sorry

end NUMINAMATH_CALUDE_apex_at_vertex_a_l3413_341366


namespace NUMINAMATH_CALUDE_tirzah_handbags_l3413_341397

/-- The number of handbags Tirzah has -/
def num_handbags : ℕ := 24

/-- The total number of purses Tirzah has -/
def total_purses : ℕ := 26

/-- The fraction of fake purses -/
def fake_purses_fraction : ℚ := 1/2

/-- The fraction of fake handbags -/
def fake_handbags_fraction : ℚ := 1/4

/-- The total number of authentic items (purses and handbags) -/
def total_authentic : ℕ := 31

theorem tirzah_handbags :
  num_handbags = 24 ∧
  total_purses = 26 ∧
  fake_purses_fraction = 1/2 ∧
  fake_handbags_fraction = 1/4 ∧
  total_authentic = 31 ∧
  (total_purses : ℚ) * (1 - fake_purses_fraction) + (num_handbags : ℚ) * (1 - fake_handbags_fraction) = total_authentic := by
  sorry

end NUMINAMATH_CALUDE_tirzah_handbags_l3413_341397


namespace NUMINAMATH_CALUDE_triangle_area_is_86_div_7_l3413_341329

/-- The slope of the first line -/
def m1 : ℚ := 3/4

/-- The slope of the second line -/
def m2 : ℚ := -2

/-- The x-coordinate of the intersection point of the first two lines -/
def x0 : ℚ := 1

/-- The y-coordinate of the intersection point of the first two lines -/
def y0 : ℚ := 3

/-- The equation of the third line: x + y = 8 -/
def line3 (x y : ℚ) : Prop := x + y = 8

/-- The area of the triangle formed by the three lines -/
def triangle_area : ℚ := 86/7

/-- Theorem stating that the area of the triangle is 86/7 -/
theorem triangle_area_is_86_div_7 : triangle_area = 86/7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_86_div_7_l3413_341329


namespace NUMINAMATH_CALUDE_min_horseshoed_ponies_fraction_l3413_341371

/-- A ranch with horses and ponies -/
structure Ranch where
  horses : ℕ
  ponies : ℕ
  horseshoed_ponies : ℕ
  iceland_horseshoed_ponies : ℕ

/-- The conditions of the ranch problem -/
def ranch_conditions (r : Ranch) : Prop :=
  r.horses = r.ponies + 4 ∧
  r.horses + r.ponies ≥ 40 ∧
  r.iceland_horseshoed_ponies = (2 * r.horseshoed_ponies) / 3

/-- The theorem stating the minimum fraction of ponies with horseshoes -/
theorem min_horseshoed_ponies_fraction (r : Ranch) : 
  ranch_conditions r → r.horseshoed_ponies * 12 ≤ r.ponies := by
  sorry

#check min_horseshoed_ponies_fraction

end NUMINAMATH_CALUDE_min_horseshoed_ponies_fraction_l3413_341371


namespace NUMINAMATH_CALUDE_power_output_scientific_notation_l3413_341311

/-- The power output of the photovoltaic station in kilowatt-hours -/
def power_output : ℝ := 448000

/-- The scientific notation representation of the power output -/
def scientific_notation : ℝ := 4.48 * (10 ^ 5)

/-- Theorem stating that the power output is equal to its scientific notation representation -/
theorem power_output_scientific_notation : power_output = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_power_output_scientific_notation_l3413_341311


namespace NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3413_341381

theorem a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq :
  (∀ a b : ℝ, a > |b| → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ a ≤ |b|) := by
  sorry

end NUMINAMATH_CALUDE_a_gt_abs_b_sufficient_not_necessary_for_a_sq_gt_b_sq_l3413_341381


namespace NUMINAMATH_CALUDE_stating_prob_served_last_independent_of_position_prob_served_last_2014_l3413_341378

/-- 
Represents a round table with n people, where food is passed randomly.
n is the number of people at the table.
-/
structure RoundTable where
  n : ℕ
  hn : n > 1

/-- 
The probability of a specific person (other than the head) being served last.
table: The round table setup
person: The index of the person we're interested in (2 ≤ person ≤ n)
-/
def probabilityServedLast (table : RoundTable) (person : ℕ) : ℚ :=
  1 / (table.n - 1)

/-- 
Theorem stating that the probability of any specific person (other than the head) 
being served last is 1/(n-1), regardless of their position.
-/
theorem prob_served_last_independent_of_position (table : RoundTable) 
    (person : ℕ) (h : 2 ≤ person ∧ person ≤ table.n) : 
    probabilityServedLast table person = 1 / (table.n - 1) := by
  sorry

/-- 
The specific case for the problem with 2014 people and the person of interest
seated 2 seats away from the head.
-/
def table2014 : RoundTable := ⟨2014, by norm_num⟩

theorem prob_served_last_2014 : 
    probabilityServedLast table2014 2 = 1 / 2013 := by
  sorry

end NUMINAMATH_CALUDE_stating_prob_served_last_independent_of_position_prob_served_last_2014_l3413_341378


namespace NUMINAMATH_CALUDE_increasing_sequence_bound_l3413_341320

theorem increasing_sequence_bound (a : ℝ) :
  (∀ n : ℕ+, (n.val - a)^2 < ((n + 1).val - a)^2) →
  a < 3/2 := by
sorry

end NUMINAMATH_CALUDE_increasing_sequence_bound_l3413_341320


namespace NUMINAMATH_CALUDE_log_inequality_cube_inequality_l3413_341348

theorem log_inequality_cube_inequality (a b : ℝ) :
  (∀ a b, Real.log a < Real.log b → a^3 < b^3) ∧
  (∃ a b, a^3 < b^3 ∧ ¬(Real.log a < Real.log b)) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_cube_inequality_l3413_341348


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3413_341351

theorem inequality_solution_set (a : ℝ) (h : a < 2) :
  {x : ℝ | a * x > 2 * x + a - 2} = {x : ℝ | x < 1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3413_341351


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3413_341382

theorem fractional_equation_solution (x : ℝ) (h : x ≠ 1) :
  (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3413_341382


namespace NUMINAMATH_CALUDE_executive_board_count_l3413_341392

/-- The number of ways to choose an executive board from a club -/
def choose_executive_board (total_members : ℕ) (board_size : ℕ) (specific_roles : ℕ) : ℕ :=
  Nat.choose total_members board_size * (board_size * (board_size - 1))

/-- Theorem stating the number of ways to choose the executive board -/
theorem executive_board_count :
  choose_executive_board 40 6 2 = 115151400 := by
  sorry

end NUMINAMATH_CALUDE_executive_board_count_l3413_341392


namespace NUMINAMATH_CALUDE_greater_than_negative_two_by_one_l3413_341305

theorem greater_than_negative_two_by_one : 
  ∃ x : ℝ, x = -2 + 1 ∧ x = -1 := by sorry

end NUMINAMATH_CALUDE_greater_than_negative_two_by_one_l3413_341305


namespace NUMINAMATH_CALUDE_slab_rate_calculation_l3413_341308

/-- Given a room with specific dimensions and total flooring cost, 
    prove that the rate per square meter for slabs is as calculated. -/
theorem slab_rate_calculation (length width total_cost : ℝ) 
    (h1 : length = 5.5)
    (h2 : width = 3.75)
    (h3 : total_cost = 12375) : 
  total_cost / (length * width) = 600 := by
  sorry

end NUMINAMATH_CALUDE_slab_rate_calculation_l3413_341308


namespace NUMINAMATH_CALUDE_cube_iff_greater_l3413_341317

theorem cube_iff_greater (a b : ℝ) : a > b ↔ a^3 > b^3 := by sorry

end NUMINAMATH_CALUDE_cube_iff_greater_l3413_341317


namespace NUMINAMATH_CALUDE_max_vertex_sum_l3413_341384

/-- Represents a parabola passing through specific points -/
structure Parabola where
  a : ℤ
  T : ℤ
  h : T ≠ 0

/-- Calculates the sum of vertex coordinates for a given parabola -/
def vertexSum (p : Parabola) : ℚ :=
  p.T - (36 : ℚ) * p.T^2 / (2 * p.T + 2)^2

/-- Theorem stating the maximum value of the vertex sum -/
theorem max_vertex_sum :
  ∀ p : Parabola, vertexSum p ≤ (-5 : ℚ) / 4 := by sorry

end NUMINAMATH_CALUDE_max_vertex_sum_l3413_341384


namespace NUMINAMATH_CALUDE_megan_popsicles_l3413_341362

/-- The number of Popsicles Megan can finish in a given time period --/
def popsicles_finished (total_minutes : ℕ) (popsicle_time : ℕ) (break_time : ℕ) (break_interval : ℕ) : ℕ :=
  let effective_minutes := total_minutes - (total_minutes / (break_interval * 60)) * break_time
  (effective_minutes / popsicle_time : ℕ)

/-- Theorem stating the number of Popsicles Megan can finish in 5 hours and 40 minutes --/
theorem megan_popsicles :
  popsicles_finished 340 20 5 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_megan_popsicles_l3413_341362


namespace NUMINAMATH_CALUDE_quadratic_perfect_square_constant_l3413_341324

/-- A quadratic expression can be expressed as the square of a binomial if and only if its discriminant is zero. -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b ^ 2 - 4 * a * c = 0

theorem quadratic_perfect_square_constant (b : ℝ) :
  is_perfect_square 9 (-24) b → b = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_perfect_square_constant_l3413_341324


namespace NUMINAMATH_CALUDE_bank_deposit_l3413_341328

theorem bank_deposit (n : ℕ) (x y : ℕ) (h1 : n = 100 * x + y) (h2 : 0 ≤ y ∧ y ≤ 99) 
  (h3 : (x : ℝ) + y = 0.02 * n) : n = 4950 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_l3413_341328


namespace NUMINAMATH_CALUDE_shortest_side_in_triangle_l3413_341398

theorem shortest_side_in_triangle (A B C : Real) (a b c : Real) :
  B = 45 * π / 180 →
  C = 60 * π / 180 →
  c = 1 →
  b = Real.sqrt 6 / 3 →
  b < a ∧ b < c :=
by
  sorry

end NUMINAMATH_CALUDE_shortest_side_in_triangle_l3413_341398


namespace NUMINAMATH_CALUDE_race_time_l3413_341303

/-- The time A takes to complete a 1 kilometer race, given that A can give B a start of 50 meters or 10 seconds. -/
theorem race_time : ℝ := by
  -- Define the race distance
  let race_distance : ℝ := 1000

  -- Define the head start distance
  let head_start_distance : ℝ := 50

  -- Define the head start time
  let head_start_time : ℝ := 10

  -- Define A's time to complete the race
  let time_A : ℝ := 200

  -- Prove that A's time is 200 seconds
  have h1 : race_distance / time_A * (time_A - head_start_time) = race_distance - head_start_distance := by sorry
  
  -- The final statement that proves the theorem
  exact time_A


end NUMINAMATH_CALUDE_race_time_l3413_341303


namespace NUMINAMATH_CALUDE_twentieth_common_number_l3413_341335

/-- The mth term of the first sequence -/
def a (m : ℕ) : ℕ := 4 * m - 1

/-- The nth term of the second sequence -/
def b (n : ℕ) : ℕ := 3 * n + 2

/-- The kth common number between the two sequences -/
def common_number (k : ℕ) : ℕ := 12 * k - 1

theorem twentieth_common_number :
  ∃ m n : ℕ, a m = b n ∧ a m = common_number 20 ∧ common_number 20 = 239 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_common_number_l3413_341335


namespace NUMINAMATH_CALUDE_spade_operation_result_l3413_341316

def spade (a b : ℝ) : ℝ := |a - b|

theorem spade_operation_result : spade 1.5 (spade 2.5 (spade 4.5 6)) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_spade_operation_result_l3413_341316


namespace NUMINAMATH_CALUDE_seventh_observation_l3413_341363

theorem seventh_observation (n : ℕ) (initial_avg : ℚ) (new_avg : ℚ) : 
  n = 6 → 
  initial_avg = 12 → 
  new_avg = 11 → 
  (n * initial_avg + (n + 1) * new_avg - n * initial_avg) / (n + 1) = 5 := by
  sorry

end NUMINAMATH_CALUDE_seventh_observation_l3413_341363


namespace NUMINAMATH_CALUDE_function_cuts_x_axis_l3413_341340

theorem function_cuts_x_axis : ∃ x : ℝ, x > 0 ∧ Real.log x + 2 * x = 0 := by sorry

end NUMINAMATH_CALUDE_function_cuts_x_axis_l3413_341340


namespace NUMINAMATH_CALUDE_a_6_value_l3413_341379

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem a_6_value (a : ℕ → ℝ) (h1 : arithmetic_sequence a) 
  (h2 : a 2 + a 10 = -12) (h3 : a 2 * a 10 = -8) : a 6 = -6 := by
  sorry

end NUMINAMATH_CALUDE_a_6_value_l3413_341379


namespace NUMINAMATH_CALUDE_min_value_of_f_l3413_341368

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 3| + Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ f x_min = 6 - Real.log 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3413_341368


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3413_341388

open Real

theorem trig_equation_solution (n : ℤ) : 
  let x : ℝ := π / 6 * (3 * ↑n + 1)
  tan (2 * x) * sin (2 * x) - 3 * sqrt 3 * (1 / tan (2 * x)) * cos (2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3413_341388


namespace NUMINAMATH_CALUDE_martha_guess_probability_l3413_341380

/-- Martha's guessing abilities -/
structure MarthaGuess where
  height_success : Rat
  weight_success : Rat
  child_height_success : Rat
  adult_height_success : Rat
  tight_clothes_weight_success : Rat
  loose_clothes_weight_success : Rat

/-- Represents a person Martha meets -/
inductive Person
  | Child : Bool → Person  -- Bool represents tight (true) or loose (false) clothes
  | Adult : Bool → Person

def martha : MarthaGuess :=
  { height_success := 5/6
  , weight_success := 6/8
  , child_height_success := 4/5
  , adult_height_success := 5/6
  , tight_clothes_weight_success := 3/4
  , loose_clothes_weight_success := 7/10 }

def people : List Person :=
  [Person.Child false, Person.Adult true, Person.Adult false]

/-- Calculates the probability of Martha guessing correctly for a specific person -/
def guessCorrectProb (m : MarthaGuess) (p : Person) : Rat :=
  match p with
  | Person.Child tight =>
      1 - (1 - m.child_height_success) * (1 - (if tight then m.tight_clothes_weight_success else m.loose_clothes_weight_success))
  | Person.Adult tight =>
      1 - (1 - m.adult_height_success) * (1 - (if tight then m.tight_clothes_weight_success else m.loose_clothes_weight_success))

/-- Theorem: The probability of Martha guessing correctly at least once for the given people is 7999/8000 -/
theorem martha_guess_probability :
  1 - (people.map (guessCorrectProb martha)).prod = 7999/8000 := by
  sorry


end NUMINAMATH_CALUDE_martha_guess_probability_l3413_341380


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l3413_341399

/-- A triangle inscribed in a circle with given properties -/
structure InscribedTriangle where
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The ratio of the triangle's sides -/
  side_ratio : Fin 3 → ℝ
  /-- The side ratio corresponds to a 3:4:5 triangle -/
  ratio_valid : side_ratio = ![3, 4, 5]
  /-- The radius of the circle is 5 -/
  radius_is_5 : radius = 5

/-- The area of an inscribed triangle with the given properties is 24 -/
theorem inscribed_triangle_area (t : InscribedTriangle) : Real.sqrt (
  (t.side_ratio 0 * t.side_ratio 1 * t.side_ratio 2 * (t.side_ratio 0 + t.side_ratio 1 + t.side_ratio 2)) /
  ((t.side_ratio 0 + t.side_ratio 1) * (t.side_ratio 1 + t.side_ratio 2) * (t.side_ratio 2 + t.side_ratio 0))
) * t.radius ^ 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l3413_341399


namespace NUMINAMATH_CALUDE_x_value_l3413_341327

theorem x_value (w y z x : ℕ) 
  (hw : w = 95)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 10) : 
  x = 145 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3413_341327


namespace NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l3413_341323

-- Define the parabolas
def parabola1 (x y : ℝ) : Prop := y = (x + 1)^2
def parabola2 (x y : ℝ) : Prop := x + 4 = (y - 3)^2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | parabola1 p.1 p.2 ∧ parabola2 p.1 p.2}

-- Theorem statement
theorem sum_of_intersection_coordinates :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (x₁, y₁) ∈ intersection_points ∧
    (x₂, y₂) ∈ intersection_points ∧
    (x₃, y₃) ∈ intersection_points ∧
    (x₄, y₄) ∈ intersection_points ∧
    (x₁ + x₂ + x₃ + x₄ + y₁ + y₂ + y₃ + y₄ = 8) :=
sorry

end NUMINAMATH_CALUDE_sum_of_intersection_coordinates_l3413_341323


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l3413_341343

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4) 
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l3413_341343


namespace NUMINAMATH_CALUDE_only_negative_sqrt_two_less_than_zero_l3413_341375

theorem only_negative_sqrt_two_less_than_zero :
  let numbers : List ℝ := [5, 2, 0, -Real.sqrt 2]
  (∀ x ∈ numbers, x < 0) ↔ (x = -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_sqrt_two_less_than_zero_l3413_341375


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3413_341301

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, 2 * x^2 - x - 1 > 0 ↔ x < -1/2 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3413_341301


namespace NUMINAMATH_CALUDE_jean_carter_books_l3413_341346

/-- Prove that given 12 total volumes, paperback price of $18, hardcover price of $30, 
    and total spent of $312, the number of hardcover volumes bought is 8. -/
theorem jean_carter_books 
  (total_volumes : ℕ) 
  (paperback_price hardcover_price : ℚ) 
  (total_spent : ℚ) 
  (h : total_volumes = 12)
  (hp : paperback_price = 18)
  (hh : hardcover_price = 30)
  (hs : total_spent = 312) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_price + (total_volumes - hardcover_count) * paperback_price = total_spent ∧ 
    hardcover_count = 8 :=
by sorry

end NUMINAMATH_CALUDE_jean_carter_books_l3413_341346


namespace NUMINAMATH_CALUDE_least_number_with_given_remainders_l3413_341354

theorem least_number_with_given_remainders :
  ∃ (n : ℕ), n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 ∧
  ∀ (m : ℕ), m > 1 → m % 25 = 1 → m % 7 = 1 → n ≤ m :=
by
  use 176
  sorry

end NUMINAMATH_CALUDE_least_number_with_given_remainders_l3413_341354


namespace NUMINAMATH_CALUDE_hexagon_regular_iff_equiangular_l3413_341374

/-- A hexagon is a polygon with 6 sides -/
structure Hexagon where
  sides : Fin 6 → ℝ
  angles : Fin 6 → ℝ

/-- A hexagon is equiangular if all its angles are equal -/
def is_equiangular (h : Hexagon) : Prop :=
  ∀ i j : Fin 6, h.angles i = h.angles j

/-- A hexagon is equilateral if all its sides are equal -/
def is_equilateral (h : Hexagon) : Prop :=
  ∀ i j : Fin 6, h.sides i = h.sides j

/-- A hexagon is regular if it is both equiangular and equilateral -/
def is_regular (h : Hexagon) : Prop :=
  is_equiangular h ∧ is_equilateral h

/-- Theorem: A hexagon is regular if and only if it is equiangular -/
theorem hexagon_regular_iff_equiangular (h : Hexagon) :
  is_regular h ↔ is_equiangular h :=
sorry

end NUMINAMATH_CALUDE_hexagon_regular_iff_equiangular_l3413_341374


namespace NUMINAMATH_CALUDE_square_sum_equality_l3413_341334

theorem square_sum_equality (p q r a b c : ℝ) 
  (h1 : p + q + r = 1) 
  (h2 : 1/p + 1/q + 1/r = 0) : 
  a^2 + b^2 + c^2 = (p*a + q*b + r*c)^2 + (q*a + r*b + p*c)^2 + (r*a + p*b + q*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3413_341334


namespace NUMINAMATH_CALUDE_lock_combination_solution_l3413_341386

/-- Represents a digit in base 12 --/
def Digit12 := Fin 12

/-- Represents a mapping from letters to digits --/
def LetterMapping := Char → Digit12

/-- Converts a number in base 12 to base 10 --/
def toBase10 (x : ℕ) : ℕ := x

/-- Checks if all characters in a string are distinct --/
def allDistinct (s : String) : Prop := sorry

/-- Converts a string to a number using the given mapping --/
def stringToNumber (s : String) (m : LetterMapping) : ℕ := sorry

/-- The main theorem --/
theorem lock_combination_solution :
  ∃! (m : LetterMapping),
    (allDistinct "VENUSISNEAR") ∧
    (stringToNumber "VENUS" m + stringToNumber "IS" m + stringToNumber "NEAR" m =
     stringToNumber "SUN" m) ∧
    (toBase10 (stringToNumber "SUN" m) = 655) := by sorry

end NUMINAMATH_CALUDE_lock_combination_solution_l3413_341386


namespace NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l3413_341359

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 727 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_minus_reciprocal_l3413_341359


namespace NUMINAMATH_CALUDE_danny_wrappers_l3413_341322

/-- Represents Danny's collection of bottle caps and wrappers -/
structure Collection where
  initial_caps : ℕ
  found_caps : ℕ
  found_wrappers : ℕ
  total_caps : ℕ
  initial_wrappers : ℕ

/-- The theorem states that the number of wrappers Danny has now
    is equal to his initial number of wrappers plus the number of wrappers found -/
theorem danny_wrappers (c : Collection)
  (h1 : c.initial_caps = 6)
  (h2 : c.found_caps = 22)
  (h3 : c.found_wrappers = 8)
  (h4 : c.total_caps = 28)
  (h5 : c.total_caps = c.initial_caps + c.found_caps) :
  c.initial_wrappers + c.found_wrappers = c.initial_wrappers + 8 := by
  sorry


end NUMINAMATH_CALUDE_danny_wrappers_l3413_341322


namespace NUMINAMATH_CALUDE_sin_three_pi_halves_l3413_341341

theorem sin_three_pi_halves : Real.sin (3 * Real.pi / 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_three_pi_halves_l3413_341341


namespace NUMINAMATH_CALUDE_incorrect_arrangements_count_l3413_341319

/-- The number of unique arrangements of the letters "e", "o", "h", "l", "l" -/
def total_arrangements : ℕ := 60

/-- The number of correct arrangements (spelling "hello") -/
def correct_arrangements : ℕ := 1

/-- Theorem stating the number of incorrect arrangements -/
theorem incorrect_arrangements_count :
  total_arrangements - correct_arrangements = 59 := by
  sorry

end NUMINAMATH_CALUDE_incorrect_arrangements_count_l3413_341319


namespace NUMINAMATH_CALUDE_smallest_sum_proof_l3413_341349

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/8]
  (∀ x ∈ sums, 1/3 + 1/8 ≤ x) ∧ (1/3 + 1/8 = 11/24) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_proof_l3413_341349


namespace NUMINAMATH_CALUDE_prob_two_tails_two_heads_proof_l3413_341325

/-- The probability of getting exactly two tails and two heads when four fair coins are tossed simultaneously -/
def prob_two_tails_two_heads : ℚ := 3/8

/-- The number of ways to choose 2 items from 4 items -/
def choose_two_from_four : ℕ := 6

/-- The probability of a specific sequence of two tails and two heads -/
def prob_specific_sequence : ℚ := 1/16

theorem prob_two_tails_two_heads_proof :
  prob_two_tails_two_heads = choose_two_from_four * prob_specific_sequence :=
by sorry

end NUMINAMATH_CALUDE_prob_two_tails_two_heads_proof_l3413_341325


namespace NUMINAMATH_CALUDE_sum_to_all_ones_implies_digit_five_or_greater_l3413_341339

/-- A function that checks if a natural number has no zero digits -/
def hasNoZeroDigits (n : ℕ) : Prop := sorry

/-- A function that generates all digit permutations of a natural number -/
def digitPermutations (n : ℕ) : Finset ℕ := sorry

/-- A function that checks if a natural number consists only of digit 1 -/
def isAllOnes (n : ℕ) : Prop := sorry

/-- A function that checks if a natural number has at least one digit 5 or greater -/
def hasDigitFiveOrGreater (n : ℕ) : Prop := sorry

/-- Theorem stating that if a number without zero digits and three of its permutations sum to all ones, it must have a digit 5 or greater -/
theorem sum_to_all_ones_implies_digit_five_or_greater (n : ℕ) :
  hasNoZeroDigits n →
  ∃ (p q r : ℕ), p ∈ digitPermutations n ∧ q ∈ digitPermutations n ∧ r ∈ digitPermutations n ∧
  isAllOnes (n + p + q + r) →
  hasDigitFiveOrGreater n :=
sorry

end NUMINAMATH_CALUDE_sum_to_all_ones_implies_digit_five_or_greater_l3413_341339


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3413_341314

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x - 6 < 0) ↔ (∃ x : ℝ, x^2 + x - 6 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3413_341314


namespace NUMINAMATH_CALUDE_linda_borrowed_amount_l3413_341377

-- Define the pay pattern
def payPattern : List Nat := [2, 4, 6, 8, 10]

-- Function to calculate pay for a given number of hours
def calculatePay (hours : Nat) : Nat :=
  let fullCycles := hours / payPattern.length
  let remainingHours := hours % payPattern.length
  fullCycles * payPattern.sum + (payPattern.take remainingHours).sum

-- Theorem statement
theorem linda_borrowed_amount :
  calculatePay 22 = 126 := by
  sorry

end NUMINAMATH_CALUDE_linda_borrowed_amount_l3413_341377


namespace NUMINAMATH_CALUDE_fruit_tree_count_l3413_341358

/-- Proves that given 18 streets, with every other tree being a fruit tree,
    and equal numbers of three types of fruit trees,
    the number of each type of fruit tree is 3. -/
theorem fruit_tree_count (total_streets : ℕ) (fruit_tree_types : ℕ) : 
  total_streets = 18 → 
  fruit_tree_types = 3 → 
  (total_streets / 2) / fruit_tree_types = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_tree_count_l3413_341358


namespace NUMINAMATH_CALUDE_sickness_temp_increase_l3413_341312

def normal_temp : ℝ := 95
def fever_threshold : ℝ := 100
def above_threshold : ℝ := 5

theorem sickness_temp_increase : 
  let current_temp := fever_threshold + above_threshold
  current_temp - normal_temp = 10 := by sorry

end NUMINAMATH_CALUDE_sickness_temp_increase_l3413_341312


namespace NUMINAMATH_CALUDE_bruce_eggs_l3413_341300

theorem bruce_eggs (initial_eggs lost_eggs : ℕ) : 
  initial_eggs ≥ lost_eggs → 
  initial_eggs - lost_eggs = initial_eggs - lost_eggs :=
by
  sorry

#check bruce_eggs 75 70

end NUMINAMATH_CALUDE_bruce_eggs_l3413_341300


namespace NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l3413_341372

theorem sqrt_eight_times_sqrt_two : Real.sqrt 8 * Real.sqrt 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_times_sqrt_two_l3413_341372


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l3413_341365

theorem two_digit_powers_of_three : 
  (∃! (s : Finset ℕ), ∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧ 
  (∃ (s : Finset ℕ), (∀ n : ℕ, n ∈ s ↔ (10 ≤ 3^n ∧ 3^n ≤ 99)) ∧ s.card = 2) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l3413_341365


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_exists_smallest_x_smallest_x_value_l3413_341395

theorem smallest_x_absolute_value (x : ℝ) : 
  (|5 * x - 3| = 32) → x ≥ -29/5 :=
by sorry

theorem exists_smallest_x : 
  ∃ x : ℝ, |5 * x - 3| = 32 ∧ ∀ y : ℝ, |5 * y - 3| = 32 → y ≥ x :=
by sorry

theorem smallest_x_value : 
  ∃ x : ℝ, x = -29/5 ∧ |5 * x - 3| = 32 ∧ ∀ y : ℝ, |5 * y - 3| = 32 → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_exists_smallest_x_smallest_x_value_l3413_341395


namespace NUMINAMATH_CALUDE_invalid_votes_percentage_l3413_341321

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (loser_votes : ℕ)
  (h1 : total_votes = 7500)
  (h2 : winner_percentage = 55 / 100)
  (h3 : loser_votes = 2700) :
  (total_votes - (loser_votes / (1 - winner_percentage))) / total_votes = 1 / 5 :=
sorry

end NUMINAMATH_CALUDE_invalid_votes_percentage_l3413_341321


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l3413_341313

theorem polynomial_division_quotient :
  let dividend : Polynomial ℚ := 10 * X^3 + 20 * X^2 - 9 * X + 3
  let divisor : Polynomial ℚ := 5 * X + 3
  let quotient : Polynomial ℚ := 2 * X^2 - X
  (dividend).div divisor = quotient := by sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l3413_341313


namespace NUMINAMATH_CALUDE_f_iter_has_two_roots_l3413_341387

def f (x : ℝ) : ℝ := x^2 + 2018*x + 1

def f_iter (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n+1 => f ∘ f_iter n

theorem f_iter_has_two_roots (n : ℕ+) : ∃ (x y : ℝ), x ≠ y ∧ f_iter n x = 0 ∧ f_iter n y = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_iter_has_two_roots_l3413_341387


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3413_341331

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 3*x + m > 0) ↔ m ≤ 9/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3413_341331


namespace NUMINAMATH_CALUDE_uncle_welly_roses_l3413_341338

/-- The number of roses Uncle Welly planted two days ago -/
def roses_two_days_ago : ℕ := 50

/-- The number of roses Uncle Welly planted yesterday -/
def roses_yesterday : ℕ := roses_two_days_ago + 20

/-- The number of roses Uncle Welly planted today -/
def roses_today : ℕ := 2 * roses_two_days_ago

/-- The total number of roses Uncle Welly planted in his vacant lot -/
def total_roses : ℕ := roses_two_days_ago + roses_yesterday + roses_today

theorem uncle_welly_roses : total_roses = 220 := by
  sorry

end NUMINAMATH_CALUDE_uncle_welly_roses_l3413_341338


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_5_to_1994_l3413_341361

theorem rightmost_three_digits_of_5_to_1994 : 5^1994 % 1000 = 625 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_5_to_1994_l3413_341361


namespace NUMINAMATH_CALUDE_sum_of_union_elements_l3413_341330

def A : Finset ℕ := {2, 0, 1, 9}

def B : Finset ℕ := Finset.image (· * 2) A

theorem sum_of_union_elements : Finset.sum (A ∪ B) id = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_union_elements_l3413_341330


namespace NUMINAMATH_CALUDE_emily_age_l3413_341309

/-- Given the age relationships between Alan, Bob, Carl, Donna, and Emily, 
    prove that Emily is 13 years old when Bob is 20. -/
theorem emily_age (alan bob carl donna emily : ℕ) : 
  alan = bob - 4 →
  bob = carl + 5 →
  donna = carl + 2 →
  emily = alan + donna - bob →
  bob = 20 →
  emily = 13 := by
sorry

end NUMINAMATH_CALUDE_emily_age_l3413_341309


namespace NUMINAMATH_CALUDE_factory_door_production_l3413_341396

/-- Calculates the number of doors produced by a car factory given various production changes -/
theorem factory_door_production
  (doors_per_car : ℕ)
  (initial_plan : ℕ)
  (shortage_decrease : ℕ)
  (pandemic_cut : Rat)
  (h1 : doors_per_car = 5)
  (h2 : initial_plan = 200)
  (h3 : shortage_decrease = 50)
  (h4 : pandemic_cut = 1/2) :
  (initial_plan - shortage_decrease) * pandemic_cut * doors_per_car = 375 := by
  sorry

end NUMINAMATH_CALUDE_factory_door_production_l3413_341396
