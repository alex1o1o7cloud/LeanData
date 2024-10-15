import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3021_302139

theorem polynomial_division_remainder (k : ℚ) : 
  ∃! k, ∃ q : Polynomial ℚ, 
    3 * X^3 + k * X^2 - 8 * X + 52 = (3 * X + 4) * q + 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3021_302139


namespace NUMINAMATH_CALUDE_wife_weekly_contribution_l3021_302114

def husband_weekly_contribution : ℕ := 335
def savings_weeks : ℕ := 24
def children_count : ℕ := 4
def child_receives : ℕ := 1680

theorem wife_weekly_contribution (wife_contribution : ℕ) :
  (husband_weekly_contribution * savings_weeks + wife_contribution * savings_weeks) / 2 =
  children_count * child_receives →
  wife_contribution = 225 := by
  sorry

end NUMINAMATH_CALUDE_wife_weekly_contribution_l3021_302114


namespace NUMINAMATH_CALUDE_max_constant_value_l3021_302159

theorem max_constant_value (c d : ℝ) : 
  (∃ (k : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) →
  (∃ (max_k : ℝ), ∀ (k : ℝ), (∃ (c d : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) → k ≤ max_k) →
  (∃ (max_k : ℝ), ∀ (k : ℝ), (∃ (c d : ℝ), 5 * c + (d - 12)^2 = k ∧ c ≤ 47) → k ≤ max_k) ∧
  (∃ (c d : ℝ), 5 * c + (d - 12)^2 = 235 ∧ c ≤ 47) :=
by sorry

end NUMINAMATH_CALUDE_max_constant_value_l3021_302159


namespace NUMINAMATH_CALUDE_system_of_equations_l3021_302171

theorem system_of_equations (x y A : ℝ) : 
  2 * x + y = A → 
  x + 2 * y = 8 → 
  (x + y) / 3 = 1.6666666666666667 → 
  A = 7 := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l3021_302171


namespace NUMINAMATH_CALUDE_problem_solution_l3021_302138

theorem problem_solution (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3021_302138


namespace NUMINAMATH_CALUDE_arithmetic_equation_l3021_302185

theorem arithmetic_equation : 6 + 18 / 3 - 4^2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equation_l3021_302185


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l3021_302145

theorem quadratic_equation_solution_sum : ∀ c d : ℝ,
  (c^2 - 6*c + 15 = 27) →
  (d^2 - 6*d + 15 = 27) →
  c ≥ d →
  3*c + 2*d = 15 + Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l3021_302145


namespace NUMINAMATH_CALUDE_square_roots_of_625_l3021_302152

theorem square_roots_of_625 :
  (∃ x : ℝ, x > 0 ∧ x^2 = 625 ∧ x = 25) ∧
  (∀ x : ℝ, x^2 = 625 ↔ x = 25 ∨ x = -25) := by
  sorry

end NUMINAMATH_CALUDE_square_roots_of_625_l3021_302152


namespace NUMINAMATH_CALUDE_sports_camp_coach_age_l3021_302109

theorem sports_camp_coach_age (total_members : ℕ) (total_average_age : ℕ)
  (num_girls num_boys num_coaches : ℕ) (girls_average_age boys_average_age : ℕ)
  (h1 : total_members = 50)
  (h2 : total_average_age = 20)
  (h3 : num_girls = 30)
  (h4 : num_boys = 15)
  (h5 : num_coaches = 5)
  (h6 : girls_average_age = 18)
  (h7 : boys_average_age = 19)
  (h8 : total_members = num_girls + num_boys + num_coaches) :
  (total_members * total_average_age - num_girls * girls_average_age - num_boys * boys_average_age) / num_coaches = 35 := by
sorry


end NUMINAMATH_CALUDE_sports_camp_coach_age_l3021_302109


namespace NUMINAMATH_CALUDE_rectangle_ratio_theorem_l3021_302151

/-- Represents the configuration of rectangles around a square -/
structure RectangleConfiguration where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The theorem statement -/
theorem rectangle_ratio_theorem (config : RectangleConfiguration) :
  (config.inner_square_side > 0) →
  (config.rectangle_short_side > 0) →
  (config.rectangle_long_side > 0) →
  (config.inner_square_side + 2 * config.rectangle_short_side = 3 * config.inner_square_side) →
  (config.rectangle_long_side + config.rectangle_short_side = 3 * config.inner_square_side) →
  (config.rectangle_long_side / config.rectangle_short_side = 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_theorem_l3021_302151


namespace NUMINAMATH_CALUDE_unique_prime_solution_l3021_302193

theorem unique_prime_solution :
  ∃! (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ∧
    p = 5 ∧ q = 3 ∧ r = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_solution_l3021_302193


namespace NUMINAMATH_CALUDE_resulting_polygon_sides_l3021_302150

/-- Represents a regular polygon with a given number of sides -/
structure RegularPolygon where
  sides : ℕ

/-- Represents the arrangement of regular polygons -/
structure PolygonArrangement where
  polygons : List RegularPolygon

/-- Calculates the number of exposed sides in the resulting polygon -/
def exposedSides (arrangement : PolygonArrangement) : ℕ :=
  sorry

/-- The specific arrangement of polygons in our problem -/
def ourArrangement : PolygonArrangement :=
  { polygons := [
      { sides := 5 },  -- pentagon
      { sides := 4 },  -- square
      { sides := 6 },  -- hexagon
      { sides := 7 },  -- heptagon
      { sides := 9 }   -- nonagon
    ] }

/-- Theorem stating that the resulting polygon has 23 sides -/
theorem resulting_polygon_sides : exposedSides ourArrangement = 23 :=
  sorry

end NUMINAMATH_CALUDE_resulting_polygon_sides_l3021_302150


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3021_302131

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h₁ : d ≠ 0
  h₂ : ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: If a₁, a₄, and a₅ of an arithmetic sequence form a geometric sequence,
    then the common ratio of this geometric sequence is 1/3 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h : (seq.a 4) ^ 2 = (seq.a 1) * (seq.a 5)) :
  (seq.a 4) / (seq.a 1) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l3021_302131


namespace NUMINAMATH_CALUDE_evaluate_expression_l3021_302173

theorem evaluate_expression : 2 - (-3) * 2 - 4 - (-5) * 3 - 6 = 13 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3021_302173


namespace NUMINAMATH_CALUDE_system_solution_l3021_302141

theorem system_solution (a b c d x y z u : ℝ) : 
  (a^3 * x + a^2 * y + a * z + u = 0) →
  (b^3 * x + b^2 * y + b * z + u = 0) →
  (c^3 * x + c^2 * y + c * z + u = 0) →
  (d^3 * x + d^2 * y + d * z + u = 1) →
  (x = 1 / ((d-a)*(d-b)*(d-c))) →
  (y = -(a+b+c) / ((d-a)*(d-b)*(d-c))) →
  (z = (a*b + b*c + c*a) / ((d-a)*(d-b)*(d-c))) →
  (u = -(a*b*c) / ((d-a)*(d-b)*(d-c))) →
  (a ≠ d) → (b ≠ d) → (c ≠ d) →
  (a^3 * x + a^2 * y + a * z + u = 0) ∧
  (b^3 * x + b^2 * y + b * z + u = 0) ∧
  (c^3 * x + c^2 * y + c * z + u = 0) ∧
  (d^3 * x + d^2 * y + d * z + u = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3021_302141


namespace NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3021_302182

/-- The parabola y = x^2 - 4 intersects the y-axis at the point (0, -4) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, -4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_axis_intersection_l3021_302182


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l3021_302183

theorem smallest_number_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < 1572 → ¬(
    (m + 3) % 9 = 0 ∧ 
    (m + 3) % 35 = 0 ∧ 
    (m + 3) % 25 = 0 ∧ 
    (m + 3) % 21 = 0
  )) ∧
  (1572 + 3) % 9 = 0 ∧ 
  (1572 + 3) % 35 = 0 ∧ 
  (1572 + 3) % 25 = 0 ∧ 
  (1572 + 3) % 21 = 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisibility_l3021_302183


namespace NUMINAMATH_CALUDE_tan_alpha_two_expressions_l3021_302198

theorem tan_alpha_two_expressions (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -5 ∧
  Real.sin α * (Real.sin α + Real.cos α) = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_expressions_l3021_302198


namespace NUMINAMATH_CALUDE_base_conversion_addition_equality_l3021_302107

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

-- Define the given numbers in their respective bases
def n1 : List Nat := [2, 5, 3]
def b1 : Nat := 8
def d1 : List Nat := [1, 3]
def b2 : Nat := 3
def n2 : List Nat := [2, 4, 5]
def b3 : Nat := 7
def d2 : List Nat := [3, 5]
def b4 : Nat := 6

-- State the theorem
theorem base_conversion_addition_equality :
  (to_base_10 n1 b1 : ℚ) / (to_base_10 d1 b2 : ℚ) + 
  (to_base_10 n2 b3 : ℚ) / (to_base_10 d2 b4 : ℚ) = 
  171 / 6 + 131 / 23 := by sorry

end NUMINAMATH_CALUDE_base_conversion_addition_equality_l3021_302107


namespace NUMINAMATH_CALUDE_keith_score_l3021_302153

theorem keith_score (keith larry danny : ℕ) 
  (larry_score : larry = 3 * keith)
  (danny_score : danny = larry + 5)
  (total_score : keith + larry + danny = 26) :
  keith = 3 := by
sorry

end NUMINAMATH_CALUDE_keith_score_l3021_302153


namespace NUMINAMATH_CALUDE_students_walking_distance_l3021_302140

/-- The problem of finding the distance students need to walk --/
theorem students_walking_distance 
  (teacher_speed : ℝ) 
  (teacher_initial_distance : ℝ)
  (student1_initial_distance : ℝ)
  (student2_initial_distance : ℝ)
  (student3_initial_distance : ℝ)
  (h1 : teacher_speed = 1.5)
  (h2 : teacher_initial_distance = 235)
  (h3 : student1_initial_distance = 87)
  (h4 : student2_initial_distance = 59)
  (h5 : student3_initial_distance = 26) :
  ∃ x : ℝ, 
    x = 42 ∧ 
    teacher_initial_distance - teacher_speed * x = 
      (student1_initial_distance - x) + 
      (student2_initial_distance - x) + 
      (student3_initial_distance - x) := by
  sorry

end NUMINAMATH_CALUDE_students_walking_distance_l3021_302140


namespace NUMINAMATH_CALUDE_sequence_comparison_l3021_302184

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 1 ∧ ∀ n : ℕ, b (n + 1) = b n * q

/-- All terms of the sequence are positive -/
def all_positive (b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n > 0

theorem sequence_comparison
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (hpos : all_positive b)
  (h1 : a 1 = b 1)
  (h11 : a 11 = b 11) :
  a 6 > b 6 := by
  sorry

end NUMINAMATH_CALUDE_sequence_comparison_l3021_302184


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3021_302149

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (a 3)^2 - 6*(a 3) + 10 = 0 →                      -- a₃ is a root of x² - 6x + 10 = 0
  (a 15)^2 - 6*(a 15) + 10 = 0 →                    -- a₁₅ is a root of x² - 6x + 10 = 0
  (∀ n, S n = (n/2) * (2*(a 1) + (n - 1)*(a 2 - a 1))) →  -- sum formula
  S 17 = 51 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3021_302149


namespace NUMINAMATH_CALUDE_prime_square_mod_180_l3021_302130

theorem prime_square_mod_180 (p : ℕ) (h_prime : Nat.Prime p) (h_gt5 : p > 5) :
  ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ 
  (∀ (r : ℕ), p^2 % 180 = r → (r = r₁ ∨ r = r₂)) :=
sorry

end NUMINAMATH_CALUDE_prime_square_mod_180_l3021_302130


namespace NUMINAMATH_CALUDE_negation_equivalence_l3021_302192

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - x + 1 > 0) ↔ (∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3021_302192


namespace NUMINAMATH_CALUDE_plane_perp_theorem_l3021_302124

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a plane and a line
variable (perp_plane_line : Plane → Line → Prop)

-- Define the intersection operation between planes
variable (intersect : Plane → Plane → Line)

-- State the theorem
theorem plane_perp_theorem 
  (α β : Plane) (l : Line) 
  (h1 : perp_planes α β) 
  (h2 : intersect α β = l) :
  ∀ γ : Plane, perp_plane_line γ l → 
    perp_planes γ α ∧ perp_planes γ β :=
sorry

end NUMINAMATH_CALUDE_plane_perp_theorem_l3021_302124


namespace NUMINAMATH_CALUDE_power_product_result_l3021_302137

theorem power_product_result : (-8)^20 * (1/4)^31 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_power_product_result_l3021_302137


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3021_302174

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 3 ∧
  a 1 + a 2 = 0

/-- The general term of the sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n - 3

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  ArithmeticSequence a → (∀ n : ℕ, a n = GeneralTerm n) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l3021_302174


namespace NUMINAMATH_CALUDE_total_spent_l3021_302129

-- Define the amounts spent by each person
variable (A B C : ℝ)

-- Define the relationships between spending amounts
axiom alice_bella : A = (13/10) * B
axiom clara_bella : C = (4/5) * B
axiom alice_clara : A = C + 15

-- Theorem to prove
theorem total_spent : A + B + C = 93 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_l3021_302129


namespace NUMINAMATH_CALUDE_inequality_proof_l3021_302163

theorem inequality_proof (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) :
  x^2 + y^2 + z^2 + 2 * Real.sqrt (3 * x * y * z) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3021_302163


namespace NUMINAMATH_CALUDE_line_intersection_l3021_302172

theorem line_intersection : 
  let x : ℚ := 27/50
  let y : ℚ := -9/10
  let line1 (x : ℚ) : ℚ := -5/3 * x
  let line2 (x : ℚ) : ℚ := 15*x - 9
  (y = line1 x) ∧ (y = line2 x) :=
by sorry

end NUMINAMATH_CALUDE_line_intersection_l3021_302172


namespace NUMINAMATH_CALUDE_reactor_rearrangements_count_l3021_302170

/-- The number of distinguishable rearrangements of REACTOR with vowels at the end -/
def rearrangements_reactor : ℕ :=
  let consonants := 4  -- R, C, T, R
  let vowels := 3      -- E, A, O
  let consonant_arrangements := Nat.factorial consonants / Nat.factorial 2  -- 4! / 2! due to repeated R
  let vowel_arrangements := Nat.factorial vowels
  consonant_arrangements * vowel_arrangements

/-- Theorem stating that the number of rearrangements is 72 -/
theorem reactor_rearrangements_count :
  rearrangements_reactor = 72 := by
  sorry

#eval rearrangements_reactor  -- Should output 72

end NUMINAMATH_CALUDE_reactor_rearrangements_count_l3021_302170


namespace NUMINAMATH_CALUDE_cylinder_height_relation_l3021_302125

theorem cylinder_height_relation :
  ∀ (r₁ h₁ r₂ h₂ : ℝ),
  r₁ > 0 → h₁ > 0 → r₂ > 0 → h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ :=
by sorry

end NUMINAMATH_CALUDE_cylinder_height_relation_l3021_302125


namespace NUMINAMATH_CALUDE_reciprocal_comparison_reciprocal_comparison_with_condition_l3021_302196

theorem reciprocal_comparison :
  (-3/2 : ℚ) < (-2/3 : ℚ) ∧
  (2/3 : ℚ) < (3/2 : ℚ) ∧
  ¬((-1 : ℚ) < (-1 : ℚ)) ∧
  ¬((1 : ℚ) < (1 : ℚ)) ∧
  ¬((3 : ℚ) < (1/3 : ℚ)) :=
by
  sorry

-- Helper definition for the condition
def less_than_reciprocal (x : ℚ) : Prop :=
  x ≠ 0 ∧ x < 1 ∧ x < 1 / x

-- Theorem using the helper definition
theorem reciprocal_comparison_with_condition :
  less_than_reciprocal (-3/2) ∧
  less_than_reciprocal (2/3) ∧
  ¬less_than_reciprocal (-1) ∧
  ¬less_than_reciprocal 1 ∧
  ¬less_than_reciprocal 3 :=
by
  sorry

end NUMINAMATH_CALUDE_reciprocal_comparison_reciprocal_comparison_with_condition_l3021_302196


namespace NUMINAMATH_CALUDE_senior_class_size_l3021_302165

/-- The number of students in the senior class at East High School -/
def total_students : ℕ := 400

/-- The proportion of students who play sports -/
def sports_proportion : ℚ := 52 / 100

/-- The proportion of sports-playing students who play soccer -/
def soccer_proportion : ℚ := 125 / 1000

/-- The number of students who play soccer -/
def soccer_players : ℕ := 26

theorem senior_class_size :
  (total_students : ℚ) * sports_proportion * soccer_proportion = soccer_players := by
  sorry

end NUMINAMATH_CALUDE_senior_class_size_l3021_302165


namespace NUMINAMATH_CALUDE_solve_refrigerator_problem_l3021_302108

def refrigerator_problem (part_payment : ℝ) (percentage : ℝ) : Prop :=
  let total_cost := part_payment / (percentage / 100)
  let remaining_amount := total_cost - part_payment
  (part_payment = 875) ∧ 
  (percentage = 25) ∧ 
  (remaining_amount = 2625)

theorem solve_refrigerator_problem :
  ∃ (part_payment percentage : ℝ), refrigerator_problem part_payment percentage :=
sorry

end NUMINAMATH_CALUDE_solve_refrigerator_problem_l3021_302108


namespace NUMINAMATH_CALUDE_cyclic_inequality_l3021_302136

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l3021_302136


namespace NUMINAMATH_CALUDE_ticket_price_possibilities_l3021_302176

theorem ticket_price_possibilities : ∃ (n : ℕ), n = (Nat.divisors 90 ∩ Nat.divisors 150).card ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_possibilities_l3021_302176


namespace NUMINAMATH_CALUDE_shape_partition_count_l3021_302156

/-- Represents a cell in the shape -/
structure Cell :=
  (x : ℕ) (y : ℕ)

/-- Represents a rectangle in the partition -/
inductive Rectangle
  | small : Cell → Rectangle  -- 1×1 square
  | large : Cell → Cell → Rectangle  -- 1×2 rectangle

/-- A partition of the shape -/
def Partition := List Rectangle

/-- The shape with 17 cells -/
def shape : List Cell := sorry

/-- Check if a partition is valid for the given shape -/
def is_valid_partition (p : Partition) (s : List Cell) : Prop := sorry

/-- Count the number of distinct valid partitions -/
def count_valid_partitions (s : List Cell) : ℕ := sorry

/-- The main theorem -/
theorem shape_partition_count :
  count_valid_partitions shape = 10 := by sorry

end NUMINAMATH_CALUDE_shape_partition_count_l3021_302156


namespace NUMINAMATH_CALUDE_angle_value_for_point_l3021_302122

theorem angle_value_for_point (θ : Real) (P : Real × Real) :
  P.1 = Real.sin (3 * Real.pi / 4) →
  P.2 = Real.cos (3 * Real.pi / 4) →
  0 ≤ θ →
  θ < 2 * Real.pi →
  (Real.cos θ, Real.sin θ) = (P.1 / Real.sqrt (P.1^2 + P.2^2), P.2 / Real.sqrt (P.1^2 + P.2^2)) →
  θ = 7 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_angle_value_for_point_l3021_302122


namespace NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l3021_302128

theorem no_real_roots_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + a ≠ 0) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_implies_a_greater_than_one_l3021_302128


namespace NUMINAMATH_CALUDE_range_of_x_plus_y_min_distance_intersection_l3021_302146

-- Define the curve C
def on_curve_C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def on_line_l (x y t α : ℝ) : Prop := x = t * Real.cos α ∧ y = 1 + t * Real.sin α

-- Theorem 1: Range of x+y
theorem range_of_x_plus_y (x y : ℝ) (h : on_curve_C x y) : x + y ≥ -1 := by
  sorry

-- Theorem 2: Minimum distance between intersection points
theorem min_distance_intersection (α : ℝ) : 
  ∃ (A B : ℝ × ℝ), 
    (on_curve_C A.1 A.2 ∧ ∃ t, on_line_l A.1 A.2 t α) ∧ 
    (on_curve_C B.1 B.2 ∧ ∃ t, on_line_l B.1 B.2 t α) ∧
    ∀ (P Q : ℝ × ℝ), 
      (on_curve_C P.1 P.2 ∧ ∃ t, on_line_l P.1 P.2 t α) →
      (on_curve_C Q.1 Q.2 ∧ ∃ t, on_line_l Q.1 Q.2 t α) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_plus_y_min_distance_intersection_l3021_302146


namespace NUMINAMATH_CALUDE_project_completion_equivalence_l3021_302112

/-- Represents the time taken to complete a project given the number of workers -/
def project_completion_time (num_workers : ℕ) (days : ℚ) : Prop :=
  num_workers * days = 120 * 7

theorem project_completion_equivalence :
  project_completion_time 120 7 → project_completion_time 80 (21/2) := by
  sorry

end NUMINAMATH_CALUDE_project_completion_equivalence_l3021_302112


namespace NUMINAMATH_CALUDE_base_seven_sum_l3021_302190

/-- Given A, B, and C are non-zero distinct digits in base 7 satisfying the equation, prove B + C = 6 in base 7 -/
theorem base_seven_sum (A B C : ℕ) : 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A < 7 ∧ B < 7 ∧ C < 7 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B) = 7^3 * A + 7^2 * A + 7 * A →
  B + C = 6 :=
by sorry

end NUMINAMATH_CALUDE_base_seven_sum_l3021_302190


namespace NUMINAMATH_CALUDE_theater_ticket_price_l3021_302180

/-- Proves that the price of a balcony seat is $8 given the theater ticket sales conditions --/
theorem theater_ticket_price (total_tickets : ℕ) (total_revenue : ℕ) 
  (orchestra_price : ℕ) (balcony_orchestra_diff : ℕ) :
  total_tickets = 360 →
  total_revenue = 3320 →
  orchestra_price = 12 →
  balcony_orchestra_diff = 140 →
  ∃ (balcony_price : ℕ), 
    balcony_price = 8 ∧
    balcony_price * (total_tickets / 2 + balcony_orchestra_diff / 2) + 
    orchestra_price * (total_tickets / 2 - balcony_orchestra_diff / 2) = total_revenue :=
by
  sorry

#check theater_ticket_price

end NUMINAMATH_CALUDE_theater_ticket_price_l3021_302180


namespace NUMINAMATH_CALUDE_stating_standard_representation_of_point_l3021_302194

/-- 
Given a point in spherical coordinates (ρ, θ, φ), this function returns its standard representation
where 0 ≤ θ < 2π and 0 ≤ φ ≤ π.
-/
def standardSphericalRepresentation (ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- 
Theorem stating that the standard representation of the point (5, 3π/5, 9π/5) 
in spherical coordinates is (5, 8π/5, π/5).
-/
theorem standard_representation_of_point : 
  standardSphericalRepresentation 5 (3 * Real.pi / 5) (9 * Real.pi / 5) = 
    (5, 8 * Real.pi / 5, Real.pi / 5) := by
  sorry

end NUMINAMATH_CALUDE_stating_standard_representation_of_point_l3021_302194


namespace NUMINAMATH_CALUDE_function_properties_l3021_302105

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- Define the derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem function_properties :
  ∀ (a b c : ℝ),
  (f' a b 1 = 3) →  -- Tangent line condition
  (f a b c 1 = 2) →  -- Point condition
  (f' a b (-2) = 0) →  -- Extreme value condition
  (a = 2 ∧ b = -4 ∧ c = 5) ∧  -- Correct values of a, b, c
  (∀ x ∈ Set.Icc (-2 : ℝ) 1, f' 2 (-4) x ≥ 0)  -- Monotonically increasing condition
  :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3021_302105


namespace NUMINAMATH_CALUDE_divide_five_children_l3021_302119

/-- The number of ways to divide n distinguishable objects into two non-empty, 
    unordered groups, where rotations within groups and swapping of groups 
    don't create new arrangements -/
def divide_into_two_groups (n : ℕ) : ℕ :=
  sorry

/-- There are 5 children to be divided -/
def num_children : ℕ := 5

/-- The theorem stating that the number of ways to divide 5 children
    into two groups under the given conditions is 50 -/
theorem divide_five_children : 
  divide_into_two_groups num_children = 50 := by
  sorry

end NUMINAMATH_CALUDE_divide_five_children_l3021_302119


namespace NUMINAMATH_CALUDE_men_who_left_job_l3021_302155

/-- Given information about tree cutting rates, prove the number of men who left the job -/
theorem men_who_left_job (initial_men : ℕ) (initial_trees : ℕ) (initial_hours : ℕ)
  (final_trees : ℕ) (final_hours : ℕ) (h1 : initial_men = 20)
  (h2 : initial_trees = 30) (h3 : initial_hours = 4)
  (h4 : final_trees = 36) (h5 : final_hours = 6) :
  ∃ (men_left : ℕ),
    men_left = 4 ∧
    (initial_trees : ℚ) / initial_hours / initial_men =
    (final_trees : ℚ) / final_hours / (initial_men - men_left) :=
by sorry

end NUMINAMATH_CALUDE_men_who_left_job_l3021_302155


namespace NUMINAMATH_CALUDE_equal_absolute_values_imply_b_equals_two_l3021_302188

theorem equal_absolute_values_imply_b_equals_two (b : ℝ) :
  (|1 - b| = |3 - b|) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_equal_absolute_values_imply_b_equals_two_l3021_302188


namespace NUMINAMATH_CALUDE_fraction_problem_l3021_302181

theorem fraction_problem (x : ℚ) : 
  x^35 * (1/4)^18 = 1/(2*(10)^35) → x = 1/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l3021_302181


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l3021_302118

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, (a - 2) * x^2 - 4 * x - 1 = 0) ↔ (a ≥ -2 ∧ a ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l3021_302118


namespace NUMINAMATH_CALUDE_unique_k_for_prime_roots_l3021_302191

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots (a b c : ℤ) : Set ℤ :=
  {x : ℤ | a * x^2 + b * x + c = 0}

theorem unique_k_for_prime_roots : 
  ∃! k : ℤ, ∀ x ∈ quadratic_roots 1 (-76) k, is_prime (x.natAbs) :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_prime_roots_l3021_302191


namespace NUMINAMATH_CALUDE_trash_can_problem_l3021_302175

theorem trash_can_problem (x y a b : ℝ) : 
  3 * x + 4 * y = 580 →
  6 * x + 5 * y = 860 →
  a + b = 200 →
  60 * a + 100 * b ≤ 15000 →
  (x = 60 ∧ y = 100) ∧ a ≥ 125 := by sorry

end NUMINAMATH_CALUDE_trash_can_problem_l3021_302175


namespace NUMINAMATH_CALUDE_balloons_given_to_sandy_l3021_302186

def initial_red_balloons : ℕ := 31
def remaining_red_balloons : ℕ := 7

theorem balloons_given_to_sandy :
  initial_red_balloons - remaining_red_balloons = 24 :=
by sorry

end NUMINAMATH_CALUDE_balloons_given_to_sandy_l3021_302186


namespace NUMINAMATH_CALUDE_cone_height_from_cylinder_l3021_302168

/-- Given a cylinder and cones with specified dimensions, prove the height of the cones. -/
theorem cone_height_from_cylinder (cylinder_radius cylinder_height cone_radius : ℝ) 
  (num_cones : ℕ) (h_cylinder_radius : cylinder_radius = 12) 
  (h_cylinder_height : cylinder_height = 10) (h_cone_radius : cone_radius = 4) 
  (h_num_cones : num_cones = 135) : 
  ∃ (cone_height : ℝ), 
    cone_height = 2 ∧ 
    (π * cylinder_radius^2 * cylinder_height = 
     num_cones * (1/3 * π * cone_radius^2 * cone_height)) := by
  sorry


end NUMINAMATH_CALUDE_cone_height_from_cylinder_l3021_302168


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3021_302116

theorem sqrt_equation_solution :
  ∃! x : ℝ, Real.sqrt (9 + 3 * x) = 15 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3021_302116


namespace NUMINAMATH_CALUDE_natural_pairs_with_sum_and_gcd_l3021_302123

theorem natural_pairs_with_sum_and_gcd (a b : ℕ) : 
  a + b = 288 → Nat.gcd a b = 36 → 
  ((a = 36 ∧ b = 252) ∨ (a = 252 ∧ b = 36) ∨ (a = 108 ∧ b = 180) ∨ (a = 180 ∧ b = 108)) :=
by sorry

end NUMINAMATH_CALUDE_natural_pairs_with_sum_and_gcd_l3021_302123


namespace NUMINAMATH_CALUDE_m_function_inequality_l3021_302117

/-- An M-function is a function f: ℝ → ℝ defined on (0, +∞) that satisfies xf''(x) > f(x) for all x in (0, +∞) -/
def is_M_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → x * (deriv^[2] f x) > f x

/-- Theorem: For any M-function f and positive real numbers x₁ and x₂, 
    the sum f(x₁) + f(x₂) is less than f(x₁ + x₂) -/
theorem m_function_inequality (f : ℝ → ℝ) (hf : is_M_function f) 
  (x₁ x₂ : ℝ) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) :
  f x₁ + f x₂ < f (x₁ + x₂) :=
sorry

end NUMINAMATH_CALUDE_m_function_inequality_l3021_302117


namespace NUMINAMATH_CALUDE_rectangular_enclosure_fence_posts_l3021_302189

/-- Calculates the number of fence posts required for a rectangular enclosure --/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  2 * (length / postSpacing + width / postSpacing) + 4

/-- Proves that the minimum number of fence posts for the given dimensions is 30 --/
theorem rectangular_enclosure_fence_posts :
  fencePostsRequired 72 48 8 = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_enclosure_fence_posts_l3021_302189


namespace NUMINAMATH_CALUDE_logarithm_calculation_l3021_302111

theorem logarithm_calculation : (Real.log 128 / Real.log 2) / (Real.log 64 / Real.log 2) - (Real.log 256 / Real.log 2) / (Real.log 16 / Real.log 2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_calculation_l3021_302111


namespace NUMINAMATH_CALUDE_tumbler_price_l3021_302113

theorem tumbler_price (num_tumblers : ℕ) (num_bills : ℕ) (bill_value : ℕ) (change : ℕ) :
  num_tumblers = 10 →
  num_bills = 5 →
  bill_value = 100 →
  change = 50 →
  (num_bills * bill_value - change) / num_tumblers = 45 := by
sorry

end NUMINAMATH_CALUDE_tumbler_price_l3021_302113


namespace NUMINAMATH_CALUDE_age_difference_l3021_302106

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 11) : a = c + 11 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3021_302106


namespace NUMINAMATH_CALUDE_lawrence_county_summer_break_l3021_302157

/-- The number of kids who stay home during summer break in Lawrence county -/
def kids_stay_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Proof that 907,611 kids stay home during summer break in Lawrence county -/
theorem lawrence_county_summer_break :
  kids_stay_home 1363293 455682 = 907611 := by
sorry

end NUMINAMATH_CALUDE_lawrence_county_summer_break_l3021_302157


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3021_302158

/-- A geometric sequence {a_n} satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  condition1 : a 5 * a 8 = 6
  condition2 : a 3 + a 10 = 5

/-- The ratio of a_20 to a_13 in the geometric sequence is either 3/2 or 2/3 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 13 = 3/2 ∨ seq.a 20 / seq.a 13 = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3021_302158


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3021_302101

theorem cubic_roots_sum (a b : ℝ) : 
  (∃ x y z : ℕ+, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∀ t : ℝ, t^3 - 9*t^2 + a*t - b = 0 ↔ t = x ∨ t = y ∨ t = z)) →
  a + b = 38 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3021_302101


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l3021_302135

theorem quadratic_roots_relation (m n : ℝ) (r₁ r₂ : ℝ) (p q : ℝ) : 
  r₁^2 - 2*m*r₁ + n = 0 →
  r₂^2 - 2*m*r₂ + n = 0 →
  r₁^4 + p*r₁^4 + q = 0 →
  r₂^4 + p*r₂^4 + q = 0 →
  r₁ + r₂ = 2*m - 3 →
  p = -(2*m - 3)^4 + 4*n*(2*m - 3)^2 - 2*n^2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l3021_302135


namespace NUMINAMATH_CALUDE_sum_15_is_120_l3021_302100

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a₁ : ℚ
  /-- The common difference of the sequence -/
  d : ℚ
  /-- The sum of the first 5 terms is 10 -/
  sum_5 : (5 : ℚ) / 2 * (2 * a₁ + 4 * d) = 10
  /-- The sum of the first 10 terms is 50 -/
  sum_10 : (10 : ℚ) / 2 * (2 * a₁ + 9 * d) = 50

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a₁ + (n - 1 : ℚ) * seq.d)

/-- Theorem: The sum of the first 15 terms is 120 -/
theorem sum_15_is_120 (seq : ArithmeticSequence) : sum_n seq 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_15_is_120_l3021_302100


namespace NUMINAMATH_CALUDE_derivative_f_l3021_302162

noncomputable def f (x : ℝ) := x * Real.sin x + Real.cos x

theorem derivative_f :
  deriv f = fun x ↦ x * Real.cos x := by sorry

end NUMINAMATH_CALUDE_derivative_f_l3021_302162


namespace NUMINAMATH_CALUDE_cubic_of_99999_l3021_302167

theorem cubic_of_99999 :
  let N : ℕ := 99999
  N^3 = 999970000299999 := by
sorry

end NUMINAMATH_CALUDE_cubic_of_99999_l3021_302167


namespace NUMINAMATH_CALUDE_prob_non_matching_is_five_sixths_l3021_302164

/-- Represents the possible colors for shorts -/
inductive ShortsColor
| Black
| Gold
| Blue

/-- Represents the possible colors for jerseys -/
inductive JerseyColor
| White
| Gold

/-- The total number of possible color combinations -/
def total_combinations : ℕ := 6

/-- The number of non-matching color combinations -/
def non_matching_combinations : ℕ := 5

/-- Probability of selecting a non-matching color combination -/
def prob_non_matching : ℚ := non_matching_combinations / total_combinations

theorem prob_non_matching_is_five_sixths :
  prob_non_matching = 5 / 6 := by sorry

end NUMINAMATH_CALUDE_prob_non_matching_is_five_sixths_l3021_302164


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3021_302134

theorem polynomial_evaluation (f : ℝ → ℝ) :
  (∀ x, f (x^2 + 2) = x^4 + 6*x^2 + 4) →
  (∀ x, f (x^2 - 2) = x^4 - 2*x^2 - 4) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3021_302134


namespace NUMINAMATH_CALUDE_rational_expression_equals_240_l3021_302127

theorem rational_expression_equals_240 (x : ℝ) (h : x = 4) :
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end NUMINAMATH_CALUDE_rational_expression_equals_240_l3021_302127


namespace NUMINAMATH_CALUDE_mean_of_added_numbers_l3021_302120

theorem mean_of_added_numbers (original_numbers : List ℝ) (x y z : ℝ) :
  original_numbers.length = 12 →
  original_numbers.sum / original_numbers.length = 72 →
  (original_numbers.sum + x + y + z) / (original_numbers.length + 3) = 80 →
  (x + y + z) / 3 = 112 := by
sorry

end NUMINAMATH_CALUDE_mean_of_added_numbers_l3021_302120


namespace NUMINAMATH_CALUDE_remainder_problem_l3021_302143

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 97 * k + 37) → N % 19 = 1 := by sorry

end NUMINAMATH_CALUDE_remainder_problem_l3021_302143


namespace NUMINAMATH_CALUDE_complex_maximum_value_l3021_302110

theorem complex_maximum_value (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₂ = 4)
  (h2 : 4 * z₁^2 - 2 * z₁ * z₂ + z₂^2 = 0) :
  ∃ (M : ℝ), M = 6 * Real.sqrt 6 ∧ 
    ∀ (w : ℂ), w = z₁ → Complex.abs ((w + 1)^2 * (w - 2)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_complex_maximum_value_l3021_302110


namespace NUMINAMATH_CALUDE_laptop_installment_calculation_l3021_302160

/-- Calculates the monthly installment amount for a laptop purchase --/
theorem laptop_installment_calculation (laptop_cost : ℝ) (down_payment_percentage : ℝ) 
  (additional_down_payment : ℝ) (balance_after_four_months : ℝ) 
  (h1 : laptop_cost = 1000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : additional_down_payment = 20)
  (h4 : balance_after_four_months = 520) : 
  ∃ (monthly_installment : ℝ), monthly_installment = 65 := by
  sorry

#check laptop_installment_calculation

end NUMINAMATH_CALUDE_laptop_installment_calculation_l3021_302160


namespace NUMINAMATH_CALUDE_mallory_journey_expenses_l3021_302103

/-- Calculates the total expenses for Mallory's journey --/
def journey_expenses (fuel_cost : ℚ) (tank_range : ℚ) (journey_distance : ℚ) 
  (hotel_nights : ℕ) (hotel_cost : ℚ) (fuel_increase : ℚ) 
  (maintenance_cost : ℚ) (activity_cost : ℚ) : ℚ :=
  let num_refills := (journey_distance / tank_range).ceil
  let total_fuel_cost := (num_refills * (num_refills - 1) / 2 * fuel_increase) + (num_refills * fuel_cost)
  let food_cost := (3 / 5) * total_fuel_cost
  let hotel_total := hotel_nights * hotel_cost
  let extra_expenses := maintenance_cost + activity_cost
  total_fuel_cost + food_cost + hotel_total + extra_expenses

/-- Theorem stating that Mallory's journey expenses equal $746 --/
theorem mallory_journey_expenses : 
  journey_expenses 45 500 2000 3 80 5 120 50 = 746 := by
  sorry

end NUMINAMATH_CALUDE_mallory_journey_expenses_l3021_302103


namespace NUMINAMATH_CALUDE_translation_complex_plane_l3021_302199

/-- A translation in the complex plane that takes 1 + 3i to 5 + 7i also takes 2 - i to 6 + 3i -/
theorem translation_complex_plane : 
  ∀ (f : ℂ → ℂ), 
  (∀ z : ℂ, ∃ w : ℂ, f z = z + w) → -- f is a translation
  (f (1 + 3*I) = 5 + 7*I) →         -- f takes 1 + 3i to 5 + 7i
  (f (2 - I) = 6 + 3*I) :=          -- f takes 2 - i to 6 + 3i
by sorry

end NUMINAMATH_CALUDE_translation_complex_plane_l3021_302199


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3021_302144

/-- An isosceles triangle with congruent sides of 8 cm and perimeter of 25 cm has a base length of 9 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side : ℝ),
  congruent_side = 8 →
  base + 2 * congruent_side = 25 →
  base = 9 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l3021_302144


namespace NUMINAMATH_CALUDE_original_class_size_l3021_302166

theorem original_class_size (original_avg : ℝ) (new_students : ℕ) (new_avg : ℝ) (avg_decrease : ℝ) :
  original_avg = 40 →
  new_students = 12 →
  new_avg = 32 →
  avg_decrease = 4 →
  ∃ N : ℕ,
    N * original_avg + new_students * new_avg = (N + new_students) * (original_avg - avg_decrease) ∧
    N = 12 :=
by sorry

end NUMINAMATH_CALUDE_original_class_size_l3021_302166


namespace NUMINAMATH_CALUDE_prime_sqrt_sum_integer_implies_equal_l3021_302197

theorem prime_sqrt_sum_integer_implies_equal (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ (z : ℤ), (Int.sqrt (p^2 + 7*p*q + q^2) + Int.sqrt (p^2 + 14*p*q + q^2) = z) → 
  p = q :=
by sorry

end NUMINAMATH_CALUDE_prime_sqrt_sum_integer_implies_equal_l3021_302197


namespace NUMINAMATH_CALUDE_salary_comparison_l3021_302148

/-- Proves that given the salaries of A, B, and C are in the ratio of 1 : 2 : 3, 
    and the sum of B and C's salaries is 6000, 
    the percentage by which C's salary exceeds A's salary is 200%. -/
theorem salary_comparison (sa sb sc : ℝ) : 
  sa > 0 → sb > 0 → sc > 0 → 
  sb / sa = 2 → sc / sa = 3 → 
  sb + sc = 6000 → 
  (sc - sa) / sa * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_salary_comparison_l3021_302148


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3021_302154

theorem negation_of_proposition (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3021_302154


namespace NUMINAMATH_CALUDE_students_in_both_clubs_is_40_l3021_302147

/-- The number of students in both photography and science clubs -/
def students_in_both_clubs (total : ℕ) (photo : ℕ) (science : ℕ) (either : ℕ) : ℕ :=
  photo + science - either

/-- Theorem: Given the conditions from the problem, prove that there are 40 students in both clubs -/
theorem students_in_both_clubs_is_40 :
  students_in_both_clubs 300 120 140 220 = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_in_both_clubs_is_40_l3021_302147


namespace NUMINAMATH_CALUDE_sick_children_count_l3021_302142

/-- Calculates the number of children who called in sick given the initial number of jellybeans,
    normal class size, jellybeans eaten per child, and jellybeans left. -/
def children_called_sick (initial_jellybeans : ℕ) (normal_class_size : ℕ) 
                         (jellybeans_per_child : ℕ) (jellybeans_left : ℕ) : ℕ :=
  normal_class_size - (initial_jellybeans - jellybeans_left) / jellybeans_per_child

theorem sick_children_count : 
  children_called_sick 100 24 3 34 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sick_children_count_l3021_302142


namespace NUMINAMATH_CALUDE_intersection_equals_interval_l3021_302161

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt x + Real.log (1 - x)}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the half-open interval [0, 1)
def interval_zero_one : Set ℝ := {x | 0 ≤ x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_interval : M_intersect_N = interval_zero_one := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_interval_l3021_302161


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3021_302187

/-- A function g: ℝ → ℝ satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (g x - y) = g x + g (g y - g (-x)) + 2 * x

/-- Theorem stating that any function satisfying the functional equation must be g(x) = -2x -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
  ∀ x : ℝ, g x = -2 * x := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3021_302187


namespace NUMINAMATH_CALUDE_find_n_l3021_302177

theorem find_n (n : ℕ) : lcm n 16 = 48 → gcd n 16 = 4 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3021_302177


namespace NUMINAMATH_CALUDE_solution_to_equation_l3021_302104

theorem solution_to_equation (x y : ℝ) :
  4 * x^2 * y^2 = 4 * x * y + 3 ↔ y = 3 / (2 * x) ∨ y = -1 / (2 * x) :=
sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3021_302104


namespace NUMINAMATH_CALUDE_min_draws_for_red_specific_l3021_302195

/-- Given a bag with red, white, and black balls, we define the minimum number of draws
    required to guarantee drawing a red ball. -/
def min_draws_for_red (red white black : ℕ) : ℕ :=
  white + black + 1

/-- Theorem stating that for a bag with 10 red, 8 white, and 7 black balls,
    the minimum number of draws to guarantee a red ball is 16. -/
theorem min_draws_for_red_specific : min_draws_for_red 10 8 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_draws_for_red_specific_l3021_302195


namespace NUMINAMATH_CALUDE_money_lasts_four_weeks_l3021_302133

def total_earnings : ℕ := 27
def weekly_expenses : ℕ := 6

theorem money_lasts_four_weeks :
  (total_earnings / weekly_expenses : ℕ) = 4 :=
by sorry

end NUMINAMATH_CALUDE_money_lasts_four_weeks_l3021_302133


namespace NUMINAMATH_CALUDE_credit_card_balance_proof_l3021_302132

def calculate_final_balance (initial_balance : ℝ)
  (month1_interest : ℝ)
  (month2_charges : ℝ) (month2_interest : ℝ)
  (month3_charges : ℝ) (month3_payment : ℝ) (month3_interest : ℝ)
  (month4_charges : ℝ) (month4_payment : ℝ) (month4_interest : ℝ) : ℝ :=
  let balance1 := initial_balance * (1 + month1_interest)
  let balance2 := (balance1 + month2_charges) * (1 + month2_interest)
  let balance3 := ((balance2 + month3_charges) - month3_payment) * (1 + month3_interest)
  let balance4 := ((balance3 + month4_charges) - month4_payment) * (1 + month4_interest)
  balance4

theorem credit_card_balance_proof :
  calculate_final_balance 50 0.2 20 0.2 30 10 0.25 40 20 0.15 = 189.75 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_proof_l3021_302132


namespace NUMINAMATH_CALUDE_product_equals_888888_l3021_302169

theorem product_equals_888888 : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_888888_l3021_302169


namespace NUMINAMATH_CALUDE_max_salary_in_soccer_league_l3021_302126

/-- Represents a soccer team with salary constraints -/
structure SoccerTeam where
  numPlayers : ℕ
  minSalary : ℕ
  totalSalaryCap : ℕ

/-- Calculates the maximum possible salary for a single player in the team -/
def maxSinglePlayerSalary (team : SoccerTeam) : ℕ :=
  team.totalSalaryCap - (team.numPlayers - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    in a semi-professional soccer league with given constraints -/
theorem max_salary_in_soccer_league :
  let team : SoccerTeam := {
    numPlayers := 25,
    minSalary := 15000,
    totalSalaryCap := 850000
  }
  maxSinglePlayerSalary team = 490000 := by
  sorry

#eval maxSinglePlayerSalary {
  numPlayers := 25,
  minSalary := 15000,
  totalSalaryCap := 850000
}

end NUMINAMATH_CALUDE_max_salary_in_soccer_league_l3021_302126


namespace NUMINAMATH_CALUDE_standing_students_count_l3021_302178

/-- Given a school meeting with the following conditions:
  * total_attendees: The total number of attendees at the meeting
  * seated_students: The number of seated students
  * seated_teachers: The number of seated teachers

  This theorem proves that the number of standing students is equal to 25.
-/
theorem standing_students_count
  (total_attendees : Nat)
  (seated_students : Nat)
  (seated_teachers : Nat)
  (h1 : total_attendees = 355)
  (h2 : seated_students = 300)
  (h3 : seated_teachers = 30) :
  total_attendees - (seated_students + seated_teachers) = 25 := by
  sorry

#check standing_students_count

end NUMINAMATH_CALUDE_standing_students_count_l3021_302178


namespace NUMINAMATH_CALUDE_orbius_5_stay_duration_l3021_302102

/-- Calculates the number of days an astronaut stays on a planet given the total days per year, 
    number of seasons per year, and number of seasons stayed. -/
def days_stayed (total_days_per_year : ℕ) (seasons_per_year : ℕ) (seasons_stayed : ℕ) : ℕ :=
  (total_days_per_year / seasons_per_year) * seasons_stayed

/-- Theorem: An astronaut staying on Orbius-5 for 3 seasons will spend 150 days on the planet. -/
theorem orbius_5_stay_duration : 
  days_stayed 250 5 3 = 150 := by
  sorry

end NUMINAMATH_CALUDE_orbius_5_stay_duration_l3021_302102


namespace NUMINAMATH_CALUDE_average_of_combined_sets_l3021_302121

theorem average_of_combined_sets (M N : ℕ) (X Y : ℝ) :
  let sum_M := M * X
  let sum_N := N * Y
  let total_sum := sum_M + sum_N
  let total_count := M + N
  (sum_M / M = X) → (sum_N / N = Y) → (total_sum / total_count = (M * X + N * Y) / (M + N)) :=
by sorry

end NUMINAMATH_CALUDE_average_of_combined_sets_l3021_302121


namespace NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l3021_302179

-- Define the number of sides on each die
def dice_sides : ℕ := 8

-- Define the set of possible prime sums
def prime_sums : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to count favorable outcomes
def count_favorable_outcomes : ℕ := 29

-- Define the total number of possible outcomes
def total_outcomes : ℕ := dice_sides * dice_sides

-- Theorem statement
theorem probability_prime_sum_two_dice :
  (count_favorable_outcomes : ℚ) / total_outcomes = 29 / 64 :=
sorry

end NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l3021_302179


namespace NUMINAMATH_CALUDE_chris_initial_money_l3021_302115

def chris_money_problem (initial_money : ℕ) : Prop :=
  let grandmother_gift : ℕ := 25
  let aunt_uncle_gift : ℕ := 20
  let parents_gift : ℕ := 75
  let total_after_gifts : ℕ := 279
  initial_money + grandmother_gift + aunt_uncle_gift + parents_gift = total_after_gifts

theorem chris_initial_money :
  ∃ (initial_money : ℕ), chris_money_problem initial_money ∧ initial_money = 159 :=
by
  sorry

end NUMINAMATH_CALUDE_chris_initial_money_l3021_302115
