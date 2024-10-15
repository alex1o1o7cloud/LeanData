import Mathlib

namespace NUMINAMATH_CALUDE_hidden_faces_sum_l1131_113102

def standard_die := List.range 6 |>.map (· + 1)

def visible_faces : List Nat := [1, 2, 3, 4, 4, 5, 6, 6]

def total_faces : Nat := 3 * 6

theorem hidden_faces_sum :
  (3 * standard_die.sum) - visible_faces.sum = 32 := by
  sorry

end NUMINAMATH_CALUDE_hidden_faces_sum_l1131_113102


namespace NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l1131_113142

theorem negation_of_universal_nonnegative_square (P : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
sorry

end NUMINAMATH_CALUDE_negation_of_universal_nonnegative_square_l1131_113142


namespace NUMINAMATH_CALUDE_simplify_expression_l1131_113105

theorem simplify_expression (a : ℝ) : 3*a^2 - 2*a + 1 + (3*a - a^2 + 2) = 2*a^2 + a + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1131_113105


namespace NUMINAMATH_CALUDE_valid_arrangement_exists_l1131_113133

/-- A binary sequence of length 6 -/
def BinarySeq := Fin 6 → Bool

/-- The set of all possible 6-digit binary sequences -/
def AllBinarySeqs : Set BinarySeq :=
  {seq | seq ∈ Set.univ}

/-- Two binary sequences differ by exactly one digit -/
def differByOne (seq1 seq2 : BinarySeq) : Prop :=
  ∃! i : Fin 6, seq1 i ≠ seq2 i

/-- A valid arrangement of binary sequences in an 8x8 grid -/
def ValidArrangement (arrangement : Fin 8 → Fin 8 → BinarySeq) : Prop :=
  (∀ i j, arrangement i j ∈ AllBinarySeqs) ∧
  (∀ i j, i + 1 < 8 → differByOne (arrangement i j) (arrangement (i + 1) j)) ∧
  (∀ i j, j + 1 < 8 → differByOne (arrangement i j) (arrangement i (j + 1))) ∧
  (∀ i j k l, (i ≠ k ∨ j ≠ l) → arrangement i j ≠ arrangement k l)

/-- The main theorem: a valid arrangement exists -/
theorem valid_arrangement_exists : ∃ arrangement, ValidArrangement arrangement := by
  sorry


end NUMINAMATH_CALUDE_valid_arrangement_exists_l1131_113133


namespace NUMINAMATH_CALUDE_evaluate_expression_l1131_113194

theorem evaluate_expression : (3025^2 : ℝ) / (305^2 - 295^2) = 1525.10417 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1131_113194


namespace NUMINAMATH_CALUDE_second_track_has_30_checkpoints_l1131_113127

/-- The number of checkpoints on the first track -/
def first_track_checkpoints : ℕ := 6

/-- The total number of ways to form triangles -/
def total_triangles : ℕ := 420

/-- The number of checkpoints on the second track -/
def second_track_checkpoints : ℕ := 30

/-- Theorem stating that the number of checkpoints on the second track is 30 -/
theorem second_track_has_30_checkpoints :
  (first_track_checkpoints * (second_track_checkpoints.choose 2) = total_triangles) →
  second_track_checkpoints = 30 := by
  sorry

end NUMINAMATH_CALUDE_second_track_has_30_checkpoints_l1131_113127


namespace NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_l1131_113184

theorem error_percentage_division_vs_multiplication :
  ∀ x : ℝ, x ≠ 0 →
  (((5 * x - x / 5) / (5 * x)) * 100 : ℝ) = 96 := by
  sorry

end NUMINAMATH_CALUDE_error_percentage_division_vs_multiplication_l1131_113184


namespace NUMINAMATH_CALUDE_only_sphere_all_circular_l1131_113198

-- Define the geometric shapes
inductive Shape
  | Cuboid
  | Cylinder
  | Cone
  | Sphere

-- Define the views
inductive View
  | Front
  | Left
  | Top

-- Define a function to determine if a view is circular
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | _, _ => False

-- Define a function to check if all views are circular
def allViewsCircular (s : Shape) : Prop :=
  isCircularView s View.Front ∧ isCircularView s View.Left ∧ isCircularView s View.Top

-- Theorem: Only the Sphere has circular views from all perspectives
theorem only_sphere_all_circular :
  ∀ s : Shape, allViewsCircular s ↔ s = Shape.Sphere :=
sorry

end NUMINAMATH_CALUDE_only_sphere_all_circular_l1131_113198


namespace NUMINAMATH_CALUDE_vector_AB_l1131_113178

-- Define the vector type
def Vector2D := ℝ × ℝ

-- Define the vector OA
def OA : Vector2D := (2, 8)

-- Define the vector OB
def OB : Vector2D := (-7, 2)

-- Define vector subtraction
def vectorSub (v1 v2 : Vector2D) : Vector2D :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Theorem statement
theorem vector_AB (OA OB : Vector2D) (h1 : OA = (2, 8)) (h2 : OB = (-7, 2)) :
  vectorSub OB OA = (-9, -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_l1131_113178


namespace NUMINAMATH_CALUDE_total_legs_is_71_l1131_113173

/-- Represents the total number of legs in a room with various furniture items -/
def total_legs : ℝ :=
  -- 4 tables with 4 legs each
  4 * 4 +
  -- 1 sofa with 4 legs
  1 * 4 +
  -- 2 chairs with 4 legs each
  2 * 4 +
  -- 3 tables with 3 legs each
  3 * 3 +
  -- 1 table with a single leg
  1 * 1 +
  -- 1 rocking chair with 2 legs
  1 * 2 +
  -- 1 bench with 6 legs
  1 * 6 +
  -- 2 stools with 3 legs each
  2 * 3 +
  -- 2 wardrobes, one with 4 legs and one with 3 legs
  (1 * 4 + 1 * 3) +
  -- 1 three-legged ecko
  1 * 3 +
  -- 1 antique table with 3 remaining legs
  1 * 3 +
  -- 1 damaged 4-legged table with only 3.5 legs remaining
  1 * 3.5 +
  -- 1 stool that lost half a leg
  1 * 2.5

/-- Theorem stating that the total number of legs in the room is 71 -/
theorem total_legs_is_71 : total_legs = 71 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_is_71_l1131_113173


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1131_113106

theorem quadratic_rewrite (b : ℝ) (m : ℝ) : 
  (∀ x, x^2 + b*x + 49 = (x + m)^2 + 9) ∧ (b > 0) → b = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1131_113106


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l1131_113177

theorem smallest_angle_in_triangle (angle1 angle2 y : ℝ) : 
  angle1 = 60 → 
  angle2 = 65 → 
  angle1 + angle2 + y = 180 → 
  min angle1 (min angle2 y) = 55 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l1131_113177


namespace NUMINAMATH_CALUDE_dropped_student_score_l1131_113180

theorem dropped_student_score 
  (initial_count : ℕ) 
  (remaining_count : ℕ) 
  (initial_average : ℚ) 
  (new_average : ℚ) 
  (h1 : initial_count = 16)
  (h2 : remaining_count = 15)
  (h3 : initial_average = 62.5)
  (h4 : new_average = 62)
  : (initial_count : ℚ) * initial_average - (remaining_count : ℚ) * new_average = 70 :=
by sorry

end NUMINAMATH_CALUDE_dropped_student_score_l1131_113180


namespace NUMINAMATH_CALUDE_max_value_x2_l1131_113138

theorem max_value_x2 (x₁ x₂ x₃ : ℝ) 
  (h : x₁^2 + x₂^2 + x₃^2 + x₁*x₂ + x₂*x₃ = 2) : 
  |x₂| ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_x2_l1131_113138


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l1131_113199

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := fun x ↦ x^2 - 3
  ∃ x₁ x₂ : ℝ, x₁ = Real.sqrt 3 ∧ x₂ = -Real.sqrt 3 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l1131_113199


namespace NUMINAMATH_CALUDE_pattern_two_odd_one_even_l1131_113129

/-- A box containing 100 balls numbered from 1 to 100 -/
def Box := Finset (Fin 100)

/-- The set of odd-numbered balls in the box -/
def OddBalls (box : Box) : Finset (Fin 100) :=
  box.filter (fun n => n % 2 = 1)

/-- The set of even-numbered balls in the box -/
def EvenBalls (box : Box) : Finset (Fin 100) :=
  box.filter (fun n => n % 2 = 0)

/-- A selection pattern of 3 balls -/
structure SelectionPattern :=
  (first second third : Bool)

/-- The probability of selecting an odd-numbered ball first -/
def ProbFirstOdd (pattern : SelectionPattern) : ℚ :=
  if pattern.first then 2/3 else 1/3

theorem pattern_two_odd_one_even
  (box : Box)
  (h_box_size : box.card = 100)
  (h_prob_first_odd : ∃ pattern : SelectionPattern, ProbFirstOdd pattern = 2/3) :
  ∃ pattern : SelectionPattern,
    pattern.first ≠ pattern.second ∨ pattern.first ≠ pattern.third ∨ pattern.second ≠ pattern.third :=
sorry

end NUMINAMATH_CALUDE_pattern_two_odd_one_even_l1131_113129


namespace NUMINAMATH_CALUDE_grass_seed_cost_l1131_113170

/-- Represents the cost and weight of a bag of grass seed -/
structure BagInfo where
  weight : ℕ
  cost : ℚ

/-- Calculates the total cost of a given number of bags -/
def totalCost (bag : BagInfo) (count : ℕ) : ℚ :=
  bag.cost * count

/-- Calculates the total weight of a given number of bags -/
def totalWeight (bag : BagInfo) (count : ℕ) : ℕ :=
  bag.weight * count

theorem grass_seed_cost
  (bag5 : BagInfo)
  (bag10 : BagInfo)
  (bag25 : BagInfo)
  (h1 : bag5.weight = 5)
  (h2 : bag10.weight = 10)
  (h3 : bag10.cost = 20.43)
  (h4 : bag25.weight = 25)
  (h5 : bag25.cost = 32.25)
  (h6 : ∃ (c5 c10 c25 : ℕ), 
    65 ≤ totalWeight bag5 c5 + totalWeight bag10 c10 + totalWeight bag25 c25 ∧
    totalWeight bag5 c5 + totalWeight bag10 c10 + totalWeight bag25 c25 ≤ 80 ∧
    totalCost bag5 c5 + totalCost bag10 c10 + totalCost bag25 c25 = 98.75 ∧
    ∀ (d5 d10 d25 : ℕ),
      65 ≤ totalWeight bag5 d5 + totalWeight bag10 d10 + totalWeight bag25 d25 →
      totalWeight bag5 d5 + totalWeight bag10 d10 + totalWeight bag25 d25 ≤ 80 →
      totalCost bag5 d5 + totalCost bag10 d10 + totalCost bag25 d25 ≥ 98.75) :
  bag5.cost = 13.82 := by
sorry

end NUMINAMATH_CALUDE_grass_seed_cost_l1131_113170


namespace NUMINAMATH_CALUDE_rectangle_area_with_hole_l1131_113117

theorem rectangle_area_with_hole (x : ℝ) : 
  (x + 8) * (x + 6) - (2*x - 4) * (x - 3) = -x^2 + 24*x + 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_with_hole_l1131_113117


namespace NUMINAMATH_CALUDE_sum_of_fraction_and_decimal_l1131_113128

theorem sum_of_fraction_and_decimal : (1 : ℚ) / 25 + (25 : ℚ) / 100 = (29 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fraction_and_decimal_l1131_113128


namespace NUMINAMATH_CALUDE_jim_gave_away_195_cards_l1131_113144

/-- The number of cards Jim gives away -/
def cards_given_away (initial_cards : ℕ) (cards_per_set : ℕ) (sets_to_brother : ℕ) (sets_to_sister : ℕ) (sets_to_friend : ℕ) : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

/-- Proof that Jim gave away 195 cards -/
theorem jim_gave_away_195_cards :
  cards_given_away 365 13 8 5 2 = 195 := by
  sorry

end NUMINAMATH_CALUDE_jim_gave_away_195_cards_l1131_113144


namespace NUMINAMATH_CALUDE_tan_sum_product_identity_l1131_113136

open Real

theorem tan_sum_product_identity (α β γ : ℝ) : 
  0 < α ∧ α < π/2 ∧ 
  0 < β ∧ β < π/2 ∧ 
  0 < γ ∧ γ < π/2 ∧ 
  α + β + γ = π/2 ∧ 
  (∀ k : ℤ, α ≠ k * π + π/2) ∧
  (∀ k : ℤ, β ≠ k * π + π/2) ∧
  (∀ k : ℤ, γ ≠ k * π + π/2) →
  tan α * tan β + tan β * tan γ + tan γ * tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_product_identity_l1131_113136


namespace NUMINAMATH_CALUDE_farm_tax_collection_l1131_113134

theorem farm_tax_collection (william_tax : ℝ) (william_land_percentage : ℝ) 
  (h1 : william_tax = 480)
  (h2 : william_land_percentage = 0.25) : 
  william_tax / william_land_percentage = 1920 := by
  sorry

end NUMINAMATH_CALUDE_farm_tax_collection_l1131_113134


namespace NUMINAMATH_CALUDE_g_neg_two_l1131_113157

def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem g_neg_two : g (-2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_l1131_113157


namespace NUMINAMATH_CALUDE_f_expression_l1131_113168

-- Define the function f
def f : ℝ → ℝ := λ x => 2 * (x - 1) - 1

-- Theorem statement
theorem f_expression : ∀ x : ℝ, f x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_f_expression_l1131_113168


namespace NUMINAMATH_CALUDE_smallest_second_term_arithmetic_sequence_l1131_113174

theorem smallest_second_term_arithmetic_sequence :
  ∀ (a d : ℕ),
  a > 0 →
  d > 0 →
  a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d) = 95 →
  ∀ (b e : ℕ),
  b > 0 →
  e > 0 →
  b + (b + e) + (b + 2*e) + (b + 3*e) + (b + 4*e) = 95 →
  (a + d) ≤ (b + e) →
  a + d = 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_second_term_arithmetic_sequence_l1131_113174


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1131_113152

theorem line_tangent_to_circle (m n : ℝ) : 
  (∀ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 → (x - 1)^2 + (y - 1)^2 ≥ 1) ∧
  (∃ x y : ℝ, (m + 1) * x + (n + 1) * y - 2 = 0 ∧ (x - 1)^2 + (y - 1)^2 = 1) →
  m + n ≤ 2 - 2 * Real.sqrt 2 ∨ m + n ≥ 2 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1131_113152


namespace NUMINAMATH_CALUDE_cricket_game_overs_l1131_113145

/-- Proves that the number of overs played in the first part of a cricket game is 10,
    given the specified conditions. -/
theorem cricket_game_overs (total_target : ℝ) (first_run_rate : ℝ) 
  (remaining_overs : ℝ) (remaining_run_rate : ℝ) 
  (h1 : total_target = 282)
  (h2 : first_run_rate = 3.2)
  (h3 : remaining_overs = 40)
  (h4 : remaining_run_rate = 6.25) :
  (total_target - remaining_overs * remaining_run_rate) / first_run_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_cricket_game_overs_l1131_113145


namespace NUMINAMATH_CALUDE_six_million_three_hundred_ninety_thousand_scientific_notation_l1131_113103

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem six_million_three_hundred_ninety_thousand_scientific_notation :
  toScientificNotation 6390000 = ScientificNotation.mk 6.39 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_six_million_three_hundred_ninety_thousand_scientific_notation_l1131_113103


namespace NUMINAMATH_CALUDE_knife_percentage_after_trade_l1131_113113

/-- Represents a silverware set with knives, forks, and spoons -/
structure Silverware where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Represents a trade of silverware -/
structure Trade where
  knivesReceived : ℕ
  spoonsGiven : ℕ

def initialSet : Silverware :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

def trade : Trade :=
  { knivesReceived := 10
  , spoonsGiven := 6 }

def finalSet (initial : Silverware) (t : Trade) : Silverware :=
  { knives := initial.knives + t.knivesReceived
  , forks := initial.forks
  , spoons := initial.spoons - t.spoonsGiven }

def totalPieces (s : Silverware) : ℕ :=
  s.knives + s.forks + s.spoons

def knifePercentage (s : Silverware) : ℚ :=
  s.knives / totalPieces s

theorem knife_percentage_after_trade :
  knifePercentage (finalSet initialSet trade) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_knife_percentage_after_trade_l1131_113113


namespace NUMINAMATH_CALUDE_inequality_proof_l1131_113154

theorem inequality_proof (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x^2019 + y = 1) :
  x + y^2019 > 1 - 1/300 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1131_113154


namespace NUMINAMATH_CALUDE_vector_decomposition_l1131_113149

def x : ℝ × ℝ × ℝ := (-13, 2, 18)
def p : ℝ × ℝ × ℝ := (1, 1, 4)
def q : ℝ × ℝ × ℝ := (-3, 0, 2)
def r : ℝ × ℝ × ℝ := (1, 2, -1)

theorem vector_decomposition :
  x = (2 : ℝ) • p + (5 : ℝ) • q := by sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1131_113149


namespace NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_five_l1131_113191

theorem negative_sqrt_of_squared_negative_five :
  -Real.sqrt ((-5)^2) = -5 := by sorry

end NUMINAMATH_CALUDE_negative_sqrt_of_squared_negative_five_l1131_113191


namespace NUMINAMATH_CALUDE_adjacent_to_five_sum_seven_l1131_113111

/-- Represents the five corners of a pentagon -/
inductive Corner
  | a | b | c | d | e

/-- A configuration of numbers in the pentagon corners -/
def Configuration := Corner → Fin 5

/-- Two corners are adjacent if they share an edge in the pentagon -/
def adjacent (x y : Corner) : Prop :=
  match x, y with
  | Corner.a, Corner.b | Corner.b, Corner.a => true
  | Corner.b, Corner.c | Corner.c, Corner.b => true
  | Corner.c, Corner.d | Corner.d, Corner.c => true
  | Corner.d, Corner.e | Corner.e, Corner.d => true
  | Corner.e, Corner.a | Corner.a, Corner.e => true
  | _, _ => false

/-- A valid configuration satisfies the adjacency condition -/
def valid_configuration (config : Configuration) : Prop :=
  ∀ x y : Corner, adjacent x y → |config x - config y| > 1

/-- The main theorem -/
theorem adjacent_to_five_sum_seven (config : Configuration) 
  (h_valid : valid_configuration config) 
  (h_five : ∃ x : Corner, config x = 5) :
  ∃ y z : Corner, 
    adjacent x y ∧ adjacent x z ∧ y ≠ z ∧ 
    config y + config z = 7 ∧ config x = 5 := by
  sorry

end NUMINAMATH_CALUDE_adjacent_to_five_sum_seven_l1131_113111


namespace NUMINAMATH_CALUDE_unique_two_digit_sum_reverse_prime_l1131_113172

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- The main theorem -/
theorem unique_two_digit_sum_reverse_prime :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ isPrime (n + reverseDigits n) :=
sorry

end NUMINAMATH_CALUDE_unique_two_digit_sum_reverse_prime_l1131_113172


namespace NUMINAMATH_CALUDE_stadium_fee_difference_l1131_113124

theorem stadium_fee_difference (capacity : ℕ) (fee : ℕ) (h1 : capacity = 2000) (h2 : fee = 20) :
  capacity * fee - (3 * capacity / 4) * fee = 10000 := by
  sorry

end NUMINAMATH_CALUDE_stadium_fee_difference_l1131_113124


namespace NUMINAMATH_CALUDE_vector_operations_l1131_113114

def vector_a : ℝ × ℝ := (2, 0)
def vector_b : ℝ × ℝ := (-1, 3)

theorem vector_operations :
  (vector_a.1 + vector_b.1, vector_a.2 + vector_b.2) = (1, 3) ∧
  (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2) = (3, -3) := by
  sorry

end NUMINAMATH_CALUDE_vector_operations_l1131_113114


namespace NUMINAMATH_CALUDE_no_solution_iff_k_geq_two_l1131_113101

theorem no_solution_iff_k_geq_two (k : ℝ) :
  (∀ x : ℝ, ¬(1 < x ∧ x ≤ 2 ∧ x > k)) ↔ k ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_geq_two_l1131_113101


namespace NUMINAMATH_CALUDE_contractor_hourly_rate_l1131_113188

/-- Contractor's hourly rate calculation -/
theorem contractor_hourly_rate 
  (total_cost : ℝ) 
  (permit_cost : ℝ) 
  (contractor_hours : ℝ) 
  (inspector_rate_ratio : ℝ) :
  total_cost = 2950 →
  permit_cost = 250 →
  contractor_hours = 15 →
  inspector_rate_ratio = 0.2 →
  ∃ (contractor_rate : ℝ),
    contractor_rate = 150 ∧
    total_cost = permit_cost + contractor_hours * contractor_rate * (1 + inspector_rate_ratio) :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_hourly_rate_l1131_113188


namespace NUMINAMATH_CALUDE_xy_nonneg_iff_abs_sum_eq_sum_abs_l1131_113115

theorem xy_nonneg_iff_abs_sum_eq_sum_abs (x y : ℝ) : x * y ≥ 0 ↔ |x + y| = |x| + |y| := by
  sorry

end NUMINAMATH_CALUDE_xy_nonneg_iff_abs_sum_eq_sum_abs_l1131_113115


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1131_113159

theorem tan_alpha_value (α : Real) 
  (h : (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α + Real.cos α) = -1) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1131_113159


namespace NUMINAMATH_CALUDE_quinary_1234_eq_194_l1131_113176

/-- Converts a quinary (base-5) number to decimal. -/
def quinary_to_decimal (q : List Nat) : Nat :=
  q.enum.foldr (fun (i, d) acc => acc + d * (5 ^ i)) 0

/-- The quinary representation of 1234₍₅₎ -/
def quinary_1234 : List Nat := [4, 3, 2, 1]

theorem quinary_1234_eq_194 : quinary_to_decimal quinary_1234 = 194 := by
  sorry

end NUMINAMATH_CALUDE_quinary_1234_eq_194_l1131_113176


namespace NUMINAMATH_CALUDE_president_vice_president_selection_l1131_113123

/-- The number of candidates for class president and vice president -/
def num_candidates : ℕ := 4

/-- The number of positions to be filled (president and vice president) -/
def num_positions : ℕ := 2

/-- Theorem: The number of ways to choose a president and a vice president from 4 candidates is 12 -/
theorem president_vice_president_selection :
  (num_candidates * (num_candidates - 1)) = 12 := by
  sorry

#check president_vice_president_selection

end NUMINAMATH_CALUDE_president_vice_president_selection_l1131_113123


namespace NUMINAMATH_CALUDE_chessboard_cut_parts_l1131_113192

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)
  (white_squares : ℕ)
  (black_squares : ℕ)

/-- Represents the possible number of parts a chessboard can be cut into --/
def PossibleParts : Set ℕ := {2, 4, 8, 16, 32}

/-- Main theorem: The number of parts a chessboard can be cut into is a subset of PossibleParts --/
theorem chessboard_cut_parts (board : Chessboard) 
  (h1 : board.size = 8) 
  (h2 : board.white_squares = 32) 
  (h3 : board.black_squares = 32) : 
  ∃ (n : ℕ), n ∈ PossibleParts ∧ 
  (board.white_squares % n = 0) ∧ 
  (n > 1) ∧ 
  (n ≤ board.black_squares) :=
sorry

end NUMINAMATH_CALUDE_chessboard_cut_parts_l1131_113192


namespace NUMINAMATH_CALUDE_scale_length_90_inches_l1131_113135

/-- Given a scale divided into equal parts, calculates the total length of the scale. -/
def scale_length (num_parts : ℕ) (part_length : ℕ) : ℕ :=
  num_parts * part_length

/-- Theorem stating that a scale with 5 parts of 18 inches each has a total length of 90 inches. -/
theorem scale_length_90_inches :
  scale_length 5 18 = 90 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_90_inches_l1131_113135


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1131_113132

-- Define the sets M and N
def M : Set ℝ := {x | x > -1}
def N : Set ℝ := {x | x * (x + 2) ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1131_113132


namespace NUMINAMATH_CALUDE_at_op_difference_l1131_113109

-- Define the @ operation
def at_op (x y : ℤ) : ℤ := x * y - x - 2 * y

-- State the theorem
theorem at_op_difference : (at_op 7 4) - (at_op 4 7) = 3 := by sorry

end NUMINAMATH_CALUDE_at_op_difference_l1131_113109


namespace NUMINAMATH_CALUDE_prob_A_or_B_prob_A_prob_B_given_A_l1131_113153

-- Define the class composition
def total_officials : ℕ := 6
def male_officials : ℕ := 4
def female_officials : ℕ := 2
def selected_officials : ℕ := 3

-- Define the events
def event_A : Set (Fin total_officials) := sorry
def event_B : Set (Fin total_officials) := sorry

-- Define the probability measure
noncomputable def P : Set (Fin total_officials) → ℝ := sorry

-- Theorem statements
theorem prob_A_or_B : P (event_A ∪ event_B) = 4/5 := by sorry

theorem prob_A : P event_A = 1/2 := by sorry

theorem prob_B_given_A : P (event_B ∩ event_A) / P event_A = 2/5 := by sorry

end NUMINAMATH_CALUDE_prob_A_or_B_prob_A_prob_B_given_A_l1131_113153


namespace NUMINAMATH_CALUDE_max_consecutive_integers_sum_thirty_one_is_max_max_consecutive_integers_sum_is_31_l1131_113139

theorem max_consecutive_integers_sum (n : ℕ) : n ≤ 31 ↔ n * (n + 1) ≤ 1000 := by
  sorry

theorem thirty_one_is_max : ∀ m : ℕ, m > 31 → m * (m + 1) > 1000 := by
  sorry

theorem max_consecutive_integers_sum_is_31 :
  (∃ n : ℕ, n * (n + 1) ≤ 1000 ∧ ∀ m : ℕ, m > n → m * (m + 1) > 1000) ∧
  (∀ n : ℕ, n * (n + 1) ≤ 1000 ∧ (∀ m : ℕ, m > n → m * (m + 1) > 1000) → n = 31) := by
  sorry

end NUMINAMATH_CALUDE_max_consecutive_integers_sum_thirty_one_is_max_max_consecutive_integers_sum_is_31_l1131_113139


namespace NUMINAMATH_CALUDE_jerrys_books_l1131_113147

/-- Given Jerry's initial and additional books, prove the total number of books. -/
theorem jerrys_books (initial_books additional_books : ℕ) :
  initial_books = 9 → additional_books = 10 → initial_books + additional_books = 19 :=
by sorry

end NUMINAMATH_CALUDE_jerrys_books_l1131_113147


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1131_113140

theorem sum_of_numbers (A B C : ℚ) : 
  (A / B = 2 / 5) → 
  (B / C = 4 / 7) → 
  (A = 16) → 
  (A + B + C = 126) := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1131_113140


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1131_113197

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let P : Point2D := ⟨-1, 2⟩
  let Q : Point2D := symmetricXAxis P
  Q.x = -1 ∧ Q.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1131_113197


namespace NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1131_113141

/-- The number of dice being thrown -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def sides_per_die : ℕ := 4

/-- The minimum possible sum when throwing the dice -/
def min_sum : ℕ := num_dice

/-- The maximum possible sum when throwing the dice -/
def max_sum : ℕ := num_dice * sides_per_die

/-- The number of possible unique sums -/
def unique_sums : ℕ := max_sum - min_sum + 1

/-- The minimum number of throws required to ensure a repeated sum -/
def min_throws : ℕ := unique_sums + 1

theorem min_throws_for_repeated_sum :
  min_throws = 14 :=
sorry

end NUMINAMATH_CALUDE_min_throws_for_repeated_sum_l1131_113141


namespace NUMINAMATH_CALUDE_news_program_selection_methods_l1131_113126

theorem news_program_selection_methods (n : ℕ) (k : ℕ) (m : ℕ) : 
  n = 8 → k = 4 → m = 2 →
  (n.choose k) * (k.choose m) * (m.factorial) = 840 := by
  sorry

end NUMINAMATH_CALUDE_news_program_selection_methods_l1131_113126


namespace NUMINAMATH_CALUDE_orange_ribbons_l1131_113162

theorem orange_ribbons (total : ℕ) (yellow purple orange black : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 12 →
  black = 40 →
  yellow + purple + orange + black = total →
  orange = 10 := by
sorry

end NUMINAMATH_CALUDE_orange_ribbons_l1131_113162


namespace NUMINAMATH_CALUDE_all_circles_pass_through_point_l1131_113182

-- Define the parabola
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  (P.2 + 2)^2 = 4 * (P.1 - 1)

-- Define a circle with center P tangent to y-axis
def circle_tangent_y_axis (P : ℝ × ℝ) (r : ℝ) : Prop :=
  r = P.1

-- Theorem statement
theorem all_circles_pass_through_point :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    is_on_parabola P →
    circle_tangent_y_axis P r →
    (P.1 - 2)^2 + (P.2 + 2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_all_circles_pass_through_point_l1131_113182


namespace NUMINAMATH_CALUDE_gwen_birthday_money_l1131_113100

/-- Given that Gwen received 14 dollars for her birthday and spent 8 dollars,
    prove that she has 6 dollars left. -/
theorem gwen_birthday_money (received : ℕ) (spent : ℕ) (left : ℕ) : 
  received = 14 → spent = 8 → left = received - spent → left = 6 := by
  sorry

end NUMINAMATH_CALUDE_gwen_birthday_money_l1131_113100


namespace NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l1131_113130

/-- Represents the number of home runs scored by a team in each inning -/
structure HomeRuns :=
  (third : ℕ)
  (fifth : ℕ)
  (eighth : ℕ)

/-- Represents the number of home runs scored by the opposing team in each inning -/
structure OpponentHomeRuns :=
  (second : ℕ)
  (fifth : ℕ)

/-- The difference in home runs between the Cubs and the Cardinals -/
def homRunDifference (cubs : HomeRuns) (cardinals : OpponentHomeRuns) : ℕ :=
  (cubs.third + cubs.fifth + cubs.eighth) - (cardinals.second + cardinals.fifth)

theorem cubs_cardinals_home_run_difference :
  ∀ (cubs : HomeRuns) (cardinals : OpponentHomeRuns),
    cubs.third = 2 → cubs.fifth = 1 → cubs.eighth = 2 →
    cardinals.second = 1 → cardinals.fifth = 1 →
    homRunDifference cubs cardinals = 3 :=
by
  sorry

#check cubs_cardinals_home_run_difference

end NUMINAMATH_CALUDE_cubs_cardinals_home_run_difference_l1131_113130


namespace NUMINAMATH_CALUDE_average_bottle_price_l1131_113181

def large_bottles : ℕ := 1325
def small_bottles : ℕ := 750
def large_bottle_price : ℚ := 189/100
def small_bottle_price : ℚ := 138/100

theorem average_bottle_price :
  let total_cost : ℚ := large_bottles * large_bottle_price + small_bottles * small_bottle_price
  let total_bottles : ℕ := large_bottles + small_bottles
  let average_price : ℚ := total_cost / total_bottles
  ∃ ε > 0, |average_price - 17/10| < ε ∧ ε < 1/100 :=
by
  sorry

end NUMINAMATH_CALUDE_average_bottle_price_l1131_113181


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1131_113151

/-- A point in a 3D rectangular coordinate system -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The symmetric point about the z-axis -/
def symmetricAboutZAxis (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

/-- Theorem: The symmetric point about the z-axis has coordinates (-a, -b, c) -/
theorem symmetric_point_coordinates (p : Point3D) :
  symmetricAboutZAxis p = { x := -p.x, y := -p.y, z := p.z } := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1131_113151


namespace NUMINAMATH_CALUDE_min_value_of_f_l1131_113183

def f (x a : ℝ) : ℝ := |x - a| + |x - 15| + |x - (a + 15)|

theorem min_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 15) :
  ∃ Q : ℝ, Q = 15 ∧ ∀ x : ℝ, a ≤ x → x ≤ 15 → f x a ≥ Q :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1131_113183


namespace NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1131_113165

theorem sum_of_five_consecutive_even_integers (m : ℤ) (h : Even m) :
  m + (m + 2) + (m + 4) + (m + 6) + (m + 8) = 5 * m + 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_five_consecutive_even_integers_l1131_113165


namespace NUMINAMATH_CALUDE_mAssignment_is_valid_l1131_113196

/-- Represents a variable in a programming context -/
structure Variable where
  name : String

/-- Represents an expression in a programming context -/
inductive Expression where
  | Var : Variable → Expression
  | Neg : Expression → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Variable
  rhs : Expression

/-- Checks if an assignment is valid according to programming rules -/
def isValidAssignment (a : Assignment) : Prop :=
  ∃ (v : Variable), a.lhs = v ∧ 
  (a.rhs = Expression.Var v ∨ a.rhs = Expression.Neg (Expression.Var v))

/-- The specific assignment M = -M -/
def mAssignment : Assignment where
  lhs := { name := "M" }
  rhs := Expression.Neg (Expression.Var { name := "M" })

theorem mAssignment_is_valid : isValidAssignment mAssignment := by
  sorry

end NUMINAMATH_CALUDE_mAssignment_is_valid_l1131_113196


namespace NUMINAMATH_CALUDE_certain_number_value_l1131_113143

theorem certain_number_value (t b c : ℝ) (x : ℝ) :
  (t + b + c + 14 + x) / 5 = 12 →
  (t + b + c + 29) / 4 = 15 →
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l1131_113143


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l1131_113155

theorem cube_sum_divisibility (a : ℤ) (h1 : a > 1) 
  (h2 : ∃ (k : ℤ), (a - 1)^3 + a^3 + (a + 1)^3 = k^3) : 
  4 ∣ a := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l1131_113155


namespace NUMINAMATH_CALUDE_pear_sales_l1131_113104

theorem pear_sales (morning_sales afternoon_sales total_sales : ℕ) : 
  afternoon_sales = 2 * morning_sales →
  afternoon_sales = 260 →
  total_sales = morning_sales + afternoon_sales →
  total_sales = 390 :=
by sorry

end NUMINAMATH_CALUDE_pear_sales_l1131_113104


namespace NUMINAMATH_CALUDE_internal_tangent_segment_bounded_l1131_113137

/-- Two equal circles with a common internal tangent and external tangents -/
structure TwoCirclesWithTangents where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of both circles (they are equal) -/
  radius : ℝ
  /-- Point where the common internal tangent intersects the external tangent of the first circle -/
  P : ℝ × ℝ
  /-- Point where the common internal tangent intersects the external tangent of the second circle -/
  Q : ℝ × ℝ
  /-- The circles are equal -/
  equal_circles : radius > 0
  /-- P is on the external tangent of the first circle -/
  P_on_external_tangent1 : (P.1 - center1.1) * (P.1 - center1.1) + (P.2 - center1.2) * (P.2 - center1.2) = radius * radius
  /-- Q is on the external tangent of the second circle -/
  Q_on_external_tangent2 : (Q.1 - center2.1) * (Q.1 - center2.1) + (Q.2 - center2.2) * (Q.2 - center2.2) = radius * radius
  /-- PQ is perpendicular to the radii at P and Q -/
  tangent_perpendicular : 
    (P.1 - center1.1) * (Q.1 - P.1) + (P.2 - center1.2) * (Q.2 - P.2) = 0 ∧
    (Q.1 - center2.1) * (P.1 - Q.1) + (Q.2 - center2.2) * (P.2 - Q.2) = 0

/-- The theorem statement -/
theorem internal_tangent_segment_bounded (c : TwoCirclesWithTangents) :
  (c.P.1 - c.Q.1) * (c.P.1 - c.Q.1) + (c.P.2 - c.Q.2) * (c.P.2 - c.Q.2) ≤
  (c.center1.1 - c.center2.1) * (c.center1.1 - c.center2.1) + (c.center1.2 - c.center2.2) * (c.center1.2 - c.center2.2) :=
sorry

end NUMINAMATH_CALUDE_internal_tangent_segment_bounded_l1131_113137


namespace NUMINAMATH_CALUDE_carpet_length_l1131_113179

/-- Given a rectangular carpet with width 4 feet covering 75% of a 48 square feet room,
    prove that the length of the carpet is 9 feet. -/
theorem carpet_length (room_area : ℝ) (carpet_width : ℝ) (coverage_percent : ℝ) :
  room_area = 48 →
  carpet_width = 4 →
  coverage_percent = 0.75 →
  (room_area * coverage_percent) / carpet_width = 9 :=
by sorry

end NUMINAMATH_CALUDE_carpet_length_l1131_113179


namespace NUMINAMATH_CALUDE_unwatered_bushes_l1131_113122

def total_bushes : ℕ := 2006

def bushes_watered_by_vitya (n : ℕ) : ℕ := n / 2
def bushes_watered_by_anya (n : ℕ) : ℕ := n / 2
def bushes_watered_by_both : ℕ := 3

theorem unwatered_bushes :
  total_bushes - (bushes_watered_by_vitya total_bushes + bushes_watered_by_anya total_bushes - bushes_watered_by_both) = 3 := by
  sorry

end NUMINAMATH_CALUDE_unwatered_bushes_l1131_113122


namespace NUMINAMATH_CALUDE_abs_eq_neg_self_implies_nonpositive_l1131_113120

theorem abs_eq_neg_self_implies_nonpositive (a : ℝ) : |a| = -a → a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_abs_eq_neg_self_implies_nonpositive_l1131_113120


namespace NUMINAMATH_CALUDE_power_four_remainder_l1131_113150

theorem power_four_remainder (a : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) : 4^a % 10 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_four_remainder_l1131_113150


namespace NUMINAMATH_CALUDE_intersection_line_not_through_point_l1131_113163

-- Define the circles
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def circle_M (a b x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = a^2 + b^2

-- Define the condition for M being on circle C
def M_on_C (a b : ℝ) : Prop := circle_C a b

-- Define the line equation passing through intersection points
def line_AB (a b m n : ℝ) : Prop := 2*m*a + 2*n*b - (2*m + 3) = 0

-- Theorem statement
theorem intersection_line_not_through_point :
  ∀ (a b : ℝ), M_on_C a b →
  ¬(line_AB a b (1/2) (1/2)) :=
sorry

end NUMINAMATH_CALUDE_intersection_line_not_through_point_l1131_113163


namespace NUMINAMATH_CALUDE_integer_solution_condition_non_negative_condition_l1131_113119

/-- The function f(x) defined in the problem -/
def f (m : ℝ) (x : ℝ) : ℝ := x^4 - 4*x^3 + (3 + m)*x^2 - 12*x + 12

/-- Theorem for the first part of the problem -/
theorem integer_solution_condition (m : ℝ) :
  (∃ x : ℤ, f m x - f m (1 - x) + 4*x^3 = 0) ↔ (m = 8 ∨ m = 12) := by sorry

/-- Theorem for the second part of the problem -/
theorem non_negative_condition (m : ℝ) :
  (∀ x : ℝ, f m x ≥ 0) ↔ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_integer_solution_condition_non_negative_condition_l1131_113119


namespace NUMINAMATH_CALUDE_probability_of_pair_after_removal_l1131_113125

/-- Represents a deck of cards -/
structure Deck :=
  (cards : Finset (Fin 13 × Fin 4))
  (card_count : cards.card = 52)

/-- Represents the deck after removing a pair and a single card -/
def RemainingDeck (d : Deck) : Finset (Fin 13 × Fin 4) :=
  d.cards.filter (λ _ => true)  -- This is a placeholder; actual implementation would remove cards

/-- Probability of selecting a matching pair from the remaining deck -/
def ProbabilityOfPair (d : Deck) : ℚ :=
  67 / 1176

/-- Main theorem: The probability of selecting a matching pair is 67/1176 -/
theorem probability_of_pair_after_removal (d : Deck) : 
  ProbabilityOfPair d = 67 / 1176 := by
  sorry

#eval (67 : ℕ) + 1176  -- Should output 1243

end NUMINAMATH_CALUDE_probability_of_pair_after_removal_l1131_113125


namespace NUMINAMATH_CALUDE_closest_year_to_target_population_l1131_113121

-- Define the population function
def population (initial : ℕ) (year : ℕ) : ℕ :=
  initial * 2^((year - 2010) / 20)

-- Define a function to calculate the difference from target population
def diff_from_target (target : ℕ) (year : ℕ) : ℕ :=
  let pop := population 500 year
  if pop ≥ target then pop - target else target - pop

-- State the theorem
theorem closest_year_to_target_population :
  (∀ y : ℕ, y ≥ 2010 → diff_from_target 10000 2090 ≤ diff_from_target 10000 y) :=
sorry

end NUMINAMATH_CALUDE_closest_year_to_target_population_l1131_113121


namespace NUMINAMATH_CALUDE_student_selection_l1131_113169

theorem student_selection (total : ℕ) (singers : ℕ) (dancers : ℕ) (both : ℕ) :
  total = 6 ∧ singers = 3 ∧ dancers = 2 ∧ both = 1 →
  Nat.choose singers 2 * dancers = 6 :=
by sorry

end NUMINAMATH_CALUDE_student_selection_l1131_113169


namespace NUMINAMATH_CALUDE_idle_days_is_37_l1131_113156

/-- Represents the worker's payment scenario -/
structure WorkerPayment where
  totalDays : ℕ
  workPayRate : ℕ
  idleForfeitRate : ℕ
  totalReceived : ℕ

/-- Calculates the number of idle days given a WorkerPayment scenario -/
def calculateIdleDays (wp : WorkerPayment) : ℕ :=
  let totalEarning := wp.workPayRate * wp.totalDays
  let totalLoss := wp.totalReceived - totalEarning
  totalLoss / (wp.workPayRate + wp.idleForfeitRate)

/-- Theorem stating that for the given scenario, the number of idle days is 37 -/
theorem idle_days_is_37 (wp : WorkerPayment) 
    (h1 : wp.totalDays = 60)
    (h2 : wp.workPayRate = 30)
    (h3 : wp.idleForfeitRate = 5)
    (h4 : wp.totalReceived = 500) :
    calculateIdleDays wp = 37 := by
  sorry

#eval calculateIdleDays ⟨60, 30, 5, 500⟩

end NUMINAMATH_CALUDE_idle_days_is_37_l1131_113156


namespace NUMINAMATH_CALUDE_circle_line_intersection_l1131_113160

/-- Given a circle and a line, prove the value of m when the chord length is 4 -/
theorem circle_line_intersection (m : ℝ) : 
  (∃ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 2 - m ∧ x + y + 2 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + 1)^2 + (y₁ - 1)^2 = 2 - m ∧
    x₁ + y₁ + 2 = 0 ∧
    (x₂ + 1)^2 + (y₂ - 1)^2 = 2 - m ∧
    x₂ + y₂ + 2 = 0 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) →
  m = -4 :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l1131_113160


namespace NUMINAMATH_CALUDE_bridget_sarah_money_l1131_113161

/-- The amount of money Bridget and Sarah have together in dollars -/
def total_money (sarah_cents bridget_cents : ℕ) : ℚ :=
  (sarah_cents + bridget_cents : ℚ) / 100

theorem bridget_sarah_money :
  ∀ (sarah_cents : ℕ),
    sarah_cents = 125 →
    ∀ (bridget_cents : ℕ),
      bridget_cents = sarah_cents + 50 →
      total_money sarah_cents bridget_cents = 3 := by
sorry

end NUMINAMATH_CALUDE_bridget_sarah_money_l1131_113161


namespace NUMINAMATH_CALUDE_chord_length_l1131_113112

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l1131_113112


namespace NUMINAMATH_CALUDE_grouping_factoring_1_grouping_factoring_2_l1131_113107

-- Expression 1
theorem grouping_factoring_1 (a b c : ℝ) :
  a^2 + 2*a*b + b^2 + a*c + b*c = (a + b) * (a + b + c) := by sorry

-- Expression 2
theorem grouping_factoring_2 (a x y : ℝ) :
  4*a^2 - x^2 + 4*x*y - 4*y^2 = (2*a + x - 2*y) * (2*a - x + 2*y) := by sorry

end NUMINAMATH_CALUDE_grouping_factoring_1_grouping_factoring_2_l1131_113107


namespace NUMINAMATH_CALUDE_cricket_average_score_l1131_113171

theorem cricket_average_score (score1 score2 : ℝ) (n1 n2 : ℕ) (h1 : score1 = 27) (h2 : score2 = 32) (h3 : n1 = 2) (h4 : n2 = 3) :
  (score1 * n1 + score2 * n2) / (n1 + n2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_cricket_average_score_l1131_113171


namespace NUMINAMATH_CALUDE_paint_calculation_l1131_113166

/-- The total amount of paint needed for finishing touches -/
def total_paint_needed (initial : ℕ) (purchased : ℕ) (additional_needed : ℕ) : ℕ :=
  initial + purchased + additional_needed

/-- Theorem stating that the total paint needed is the sum of initial, purchased, and additional needed paint -/
theorem paint_calculation (initial : ℕ) (purchased : ℕ) (additional_needed : ℕ) :
  total_paint_needed initial purchased additional_needed =
  initial + purchased + additional_needed :=
by
  sorry

#eval total_paint_needed 36 23 11

end NUMINAMATH_CALUDE_paint_calculation_l1131_113166


namespace NUMINAMATH_CALUDE_polygon_sides_l1131_113175

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 + 360 = 1980 → n = 11 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_l1131_113175


namespace NUMINAMATH_CALUDE_fraction_value_when_y_is_three_l1131_113189

theorem fraction_value_when_y_is_three :
  let y : ℝ := 3
  (y^3 + y) / (y^2 - y) = 5 := by
sorry

end NUMINAMATH_CALUDE_fraction_value_when_y_is_three_l1131_113189


namespace NUMINAMATH_CALUDE_pradeep_failed_by_25_marks_l1131_113146

/-- Calculates the number of marks by which a student failed, given the total marks,
    passing percentage, and the student's marks. -/
def marksFailed (totalMarks passingPercentage studentMarks : ℕ) : ℕ :=
  let passingMarks := totalMarks * passingPercentage / 100
  if studentMarks ≥ passingMarks then 0
  else passingMarks - studentMarks

/-- Theorem stating that Pradeep failed by 25 marks -/
theorem pradeep_failed_by_25_marks :
  marksFailed 840 25 185 = 25 := by
  sorry

end NUMINAMATH_CALUDE_pradeep_failed_by_25_marks_l1131_113146


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1131_113195

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3 / 4 = 2 * s) → (3 * s = 8 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1131_113195


namespace NUMINAMATH_CALUDE_log_five_eighteen_l1131_113158

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_five_eighteen (a b : ℝ) 
  (h1 : log 10 2 = a) 
  (h2 : log 10 3 = b) : 
  log 5 18 = (a + 2*b) / (1 - a) := by
  sorry

end NUMINAMATH_CALUDE_log_five_eighteen_l1131_113158


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1131_113116

-- Problem 1
theorem problem_1 (a : ℝ) : (-a^2)^3 + 9*a^4*a^2 = 8*a^6 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2*a*b^2 + a^2*b + b^3 = b*(a+b)^2 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) (h1 : x ≠ y) (h2 : x ≠ 2*y) :
  (1/(x-y) - 1/(x+y)) / ((x-2*y)/((x^2)-(y^2))) = 2*y/(x-2*y) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l1131_113116


namespace NUMINAMATH_CALUDE_river_speed_calculation_l1131_113148

-- Define the swimmer's speed in still water
variable (a : ℝ) 

-- Define the speed of the river flow
def river_speed : ℝ := 0.02

-- Define the time the swimmer swam upstream before realizing the loss
def upstream_time : ℝ := 0.5

-- Define the distance downstream where the swimmer catches up to the bottle
def downstream_distance : ℝ := 1.2

-- Theorem statement
theorem river_speed_calculation (h : ∀ a > 0, 
  (downstream_distance + upstream_time * (a - 60 * river_speed)) / (a + 60 * river_speed) = 
  downstream_distance / (60 * river_speed) - upstream_time) : 
  river_speed = 0.02 := by sorry

end NUMINAMATH_CALUDE_river_speed_calculation_l1131_113148


namespace NUMINAMATH_CALUDE_rectangle_width_l1131_113167

theorem rectangle_width (w : ℝ) (h1 : w > 0) : 
  (2 * w * w = 1) → w = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_l1131_113167


namespace NUMINAMATH_CALUDE_league_games_count_l1131_113193

/-- The number of teams in the league -/
def num_teams : ℕ := 25

/-- The total number of games played in the league -/
def total_games : ℕ := num_teams * (num_teams - 1) / 2

/-- Theorem stating that the total number of games in the league is 300 -/
theorem league_games_count : total_games = 300 := by
  sorry

end NUMINAMATH_CALUDE_league_games_count_l1131_113193


namespace NUMINAMATH_CALUDE_inequality_proof_l1131_113186

theorem inequality_proof (n : ℕ+) (a b c : ℝ) 
  (ha : a ≥ 1) (hb : b ≥ 1) (hc : c > 0) : 
  ((a * b + c)^n.val - c) / ((b + c)^n.val - c) ≤ a^n.val := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1131_113186


namespace NUMINAMATH_CALUDE_go_complexity_vs_universe_atoms_l1131_113108

/-- Approximation of the upper limit of state space complexity of Go -/
def M : ℝ := 3^361

/-- Approximation of the total number of atoms in the observable universe -/
def N : ℝ := 10^80

/-- Approximation of log base 10 of 3 -/
def log10_3 : ℝ := 0.48

/-- The closest value to M/N among the given options -/
def closest_value : ℝ := 10^93

theorem go_complexity_vs_universe_atoms :
  abs (M / N - closest_value) = 
    min (abs (M / N - 10^33)) 
        (min (abs (M / N - 10^53)) 
             (min (abs (M / N - 10^73)) 
                  (abs (M / N - 10^93)))) := by
  sorry

end NUMINAMATH_CALUDE_go_complexity_vs_universe_atoms_l1131_113108


namespace NUMINAMATH_CALUDE_count_triplets_eq_30_l1131_113187

/-- Count of ordered triplets (a, b, c) of positive integers satisfying 30a + 50b + 70c ≤ 343 -/
def count_triplets : ℕ :=
  (Finset.filter (fun (t : ℕ × ℕ × ℕ) =>
    let (a, b, c) := t
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 30 * a + 50 * b + 70 * c ≤ 343)
    (Finset.product (Finset.range 12) (Finset.product (Finset.range 7) (Finset.range 5)))).card

theorem count_triplets_eq_30 : count_triplets = 30 := by
  sorry

end NUMINAMATH_CALUDE_count_triplets_eq_30_l1131_113187


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l1131_113185

theorem decimal_to_fraction (x : ℚ) : x = 224/100 → x = 56/25 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l1131_113185


namespace NUMINAMATH_CALUDE_log_inequality_l1131_113164

theorem log_inequality (x : Real) (h : x > 0) : Real.log (1 + x^2) < x^2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l1131_113164


namespace NUMINAMATH_CALUDE_equation_solution_l1131_113110

theorem equation_solution :
  ∃ x : ℚ, (3 * x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1131_113110


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1131_113131

theorem least_subtraction_for_divisibility :
  ∃! x : ℕ, x ≤ 13 ∧ (7538 - x) % 14 = 0 ∧ ∀ y : ℕ, y < x → (7538 - y) % 14 ≠ 0 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l1131_113131


namespace NUMINAMATH_CALUDE_sequence_14th_term_is_9_l1131_113118

theorem sequence_14th_term_is_9 :
  let a : ℕ → ℝ := fun n => Real.sqrt (3 * (2 * n - 1))
  a 14 = 9 := by sorry

end NUMINAMATH_CALUDE_sequence_14th_term_is_9_l1131_113118


namespace NUMINAMATH_CALUDE_student_marks_l1131_113190

theorem student_marks (total_marks passing_percentage failing_margin : ℕ) 
  (h1 : total_marks = 440)
  (h2 : passing_percentage = 50)
  (h3 : failing_margin = 20) : 
  (total_marks * passing_percentage / 100 - failing_margin : ℕ) = 200 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_l1131_113190
