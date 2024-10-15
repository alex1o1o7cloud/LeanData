import Mathlib

namespace NUMINAMATH_CALUDE_students_answering_yes_for_R_l451_45179

theorem students_answering_yes_for_R (total : ℕ) (only_M : ℕ) (neither : ℕ) (h1 : total = 800) (h2 : only_M = 150) (h3 : neither = 250) : 
  ∃ R : ℕ, R = 400 ∧ R = total - neither - only_M :=
by sorry

end NUMINAMATH_CALUDE_students_answering_yes_for_R_l451_45179


namespace NUMINAMATH_CALUDE_inequality_proof_l451_45141

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * ((x + y + z) / 3)) ^ (1/4) ≤ (((x + y) / 2) * ((y + z) / 2) * ((z + x) / 2)) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_l451_45141


namespace NUMINAMATH_CALUDE_final_statue_count_statue_count_increases_l451_45103

/-- Represents the number of statues on Grandma Molly's lawn over four years -/
def statue_count : ℕ → ℕ
| 0 => 4  -- Initial number of statues
| 1 => 7  -- After year 2: 4 + 7 - 4
| 2 => 9  -- After year 3: 7 + 9 - 7
| 3 => 13 -- After year 4: 9 + 4
| _ => 13 -- Any year after 4

/-- The final number of statues after four years is 13 -/
theorem final_statue_count : statue_count 3 = 13 := by
  sorry

/-- The number of statues increases over the years -/
theorem statue_count_increases (n : ℕ) : n < 3 → statue_count n < statue_count (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_final_statue_count_statue_count_increases_l451_45103


namespace NUMINAMATH_CALUDE_set_operations_l451_45116

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def B : Set ℕ := {4, 7, 8, 9}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
  (A ∩ B = {4, 7, 8}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l451_45116


namespace NUMINAMATH_CALUDE_additive_function_properties_l451_45162

/-- A function f: ℝ → ℝ satisfying f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
def AdditiveFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_function_properties (f : ℝ → ℝ) (hf : AdditiveFunction f) :
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by sorry

end NUMINAMATH_CALUDE_additive_function_properties_l451_45162


namespace NUMINAMATH_CALUDE_line_through_intersection_parallel_to_given_l451_45135

-- Define the lines l1 and l2
def l1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
def l2 (x y : ℝ) : Prop := 3 * x - 4 * y + 2 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 4 * x + y - 4 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := l1 x y ∧ l2 x y

-- Define the resulting line
def result_line (x y : ℝ) : Prop := 4 * x + y - 10 = 0

-- Theorem statement
theorem line_through_intersection_parallel_to_given :
  ∃ (x₀ y₀ : ℝ), intersection_point x₀ y₀ ∧
  ∀ (x y : ℝ), (∃ (k : ℝ), y - y₀ = k * (x - x₀) ∧ 
                parallel_line (x₀ + 1) (y₀ + k)) ↔
               result_line x y := by sorry

end NUMINAMATH_CALUDE_line_through_intersection_parallel_to_given_l451_45135


namespace NUMINAMATH_CALUDE_distinct_polygons_count_l451_45169

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The total number of possible subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets that cannot form polygons (0, 1, or 2 points) -/
def non_polygon_subsets : ℕ := (n.choose 0) + (n.choose 1) + (n.choose 2)

/-- The number of distinct convex polygons with three or more sides -/
def num_polygons : ℕ := total_subsets - non_polygon_subsets

theorem distinct_polygons_count :
  num_polygons = 4017 :=
by sorry

end NUMINAMATH_CALUDE_distinct_polygons_count_l451_45169


namespace NUMINAMATH_CALUDE_third_circle_properties_l451_45147

/-- Given two concentric circles with radii 10 and 20 units, prove that a third circle
    with area equal to the shaded area between the two concentric circles has a radius
    of 10√3 and a circumference of 20√3π. -/
theorem third_circle_properties (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 20)
    (h₃ : π * r₃^2 = π * r₂^2 - π * r₁^2) :
  r₃ = 10 * Real.sqrt 3 ∧ 2 * π * r₃ = 20 * Real.sqrt 3 * π := by
  sorry

#check third_circle_properties

end NUMINAMATH_CALUDE_third_circle_properties_l451_45147


namespace NUMINAMATH_CALUDE_expression_evaluation_l451_45126

theorem expression_evaluation : 
  let b : ℚ := 4/3
  (6 * b^2 - 8 * b + 3) * (3 * b - 4) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l451_45126


namespace NUMINAMATH_CALUDE_d_value_when_x_plus_3_is_factor_l451_45189

/-- The polynomial Q(x) with parameter d -/
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 27

/-- Theorem stating that d = -27 when x+3 is a factor of Q(x) -/
theorem d_value_when_x_plus_3_is_factor :
  ∃ d : ℝ, (∀ x : ℝ, Q d x = 0 ↔ x = -3) → d = -27 := by
  sorry

end NUMINAMATH_CALUDE_d_value_when_x_plus_3_is_factor_l451_45189


namespace NUMINAMATH_CALUDE_domain_of_f_l451_45178

noncomputable def f (x : ℝ) : ℝ := (2 * x + 3) / (x + 5)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < -5 ∨ x > -5} := by sorry

end NUMINAMATH_CALUDE_domain_of_f_l451_45178


namespace NUMINAMATH_CALUDE_dividend_calculation_l451_45113

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h_divisor : divisor = 15)
  (h_quotient : quotient = 8)
  (h_remainder : remainder = 5) :
  divisor * quotient + remainder = 125 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l451_45113


namespace NUMINAMATH_CALUDE_zero_is_natural_number_zero_not_natural_is_false_l451_45196

-- Define the set of natural numbers including 0
def NaturalNumbers : Set ℕ := {n : ℕ | True}

-- State the theorem
theorem zero_is_natural_number : (0 : ℕ) ∈ NaturalNumbers := by
  sorry

-- Prove that the statement "0 is not a natural number" is false
theorem zero_not_natural_is_false : ¬(0 ∉ NaturalNumbers) := by
  sorry

end NUMINAMATH_CALUDE_zero_is_natural_number_zero_not_natural_is_false_l451_45196


namespace NUMINAMATH_CALUDE_movie_marathon_difference_l451_45104

/-- The duration of a movie marathon with three movies. -/
structure MovieMarathon where
  first_movie : ℝ
  second_movie : ℝ
  last_movie : ℝ
  total_time : ℝ

/-- The conditions of the movie marathon problem. -/
def movie_marathon_conditions (m : MovieMarathon) : Prop :=
  m.first_movie = 2 ∧
  m.second_movie = m.first_movie * 1.5 ∧
  m.total_time = 9 ∧
  m.total_time = m.first_movie + m.second_movie + m.last_movie

/-- The theorem stating the difference between the combined time of the first two movies
    and the last movie is 1 hour. -/
theorem movie_marathon_difference (m : MovieMarathon) 
  (h : movie_marathon_conditions m) : 
  m.first_movie + m.second_movie - m.last_movie = 1 := by
  sorry

end NUMINAMATH_CALUDE_movie_marathon_difference_l451_45104


namespace NUMINAMATH_CALUDE_largest_constant_inequality_l451_45159

theorem largest_constant_inequality (C : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + x*y + 1 ≥ C*(x + y)) ↔ C ≤ 2/Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_largest_constant_inequality_l451_45159


namespace NUMINAMATH_CALUDE_rice_containers_l451_45137

theorem rice_containers (total_weight : ℚ) (container_capacity : ℕ) (pound_to_ounce : ℕ) : 
  total_weight = 25 / 2 →
  container_capacity = 50 →
  pound_to_ounce = 16 →
  (total_weight * pound_to_ounce : ℚ) / container_capacity = 4 := by
  sorry

end NUMINAMATH_CALUDE_rice_containers_l451_45137


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l451_45173

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is c/a, where c² = a² + b² --/
theorem hyperbola_eccentricity (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let c := Real.sqrt (a^2 + b^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  c / a = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l451_45173


namespace NUMINAMATH_CALUDE_square_of_two_times_sqrt_three_l451_45125

theorem square_of_two_times_sqrt_three : (2 * Real.sqrt 3) ^ 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_of_two_times_sqrt_three_l451_45125


namespace NUMINAMATH_CALUDE_highest_page_number_with_19_sevens_l451_45165

/-- Counts the number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ :=
  sorry

/-- Counts the total occurrences of a digit in a range of natural numbers -/
def countDigitInRange (start finish : ℕ) (d : ℕ) : ℕ :=
  sorry

/-- The highest page number that can be reached with a given number of sevens -/
def highestPageNumber (numSevens : ℕ) : ℕ :=
  sorry

theorem highest_page_number_with_19_sevens :
  highestPageNumber 19 = 99 :=
sorry

end NUMINAMATH_CALUDE_highest_page_number_with_19_sevens_l451_45165


namespace NUMINAMATH_CALUDE_smallest_slope_tangent_line_l451_45144

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x : ℝ, f' x₀ ≤ f' x) ∧ 
    y₀ = f x₀ ∧
    (3 : ℝ) * x - y - 11 = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_slope_tangent_line_l451_45144


namespace NUMINAMATH_CALUDE_vector_parallel_k_l451_45180

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

theorem vector_parallel_k (k : ℝ) :
  parallel ((k * a.1 - b.1, k * a.2 - b.2) : ℝ × ℝ) (a.1 + 3 * b.1, a.2 + 3 * b.2) →
  k = -1/3 :=
sorry

end NUMINAMATH_CALUDE_vector_parallel_k_l451_45180


namespace NUMINAMATH_CALUDE_michelle_boutique_two_ties_probability_l451_45102

/-- Represents the probability of selecting 2 ties from a boutique with given items. -/
def probability_two_ties (shirts pants ties : ℕ) : ℚ :=
  let total := shirts + pants + ties
  (ties : ℚ) / total * ((ties - 1) : ℚ) / (total - 1)

/-- Theorem stating the probability of selecting 2 ties from Michelle's boutique. -/
theorem michelle_boutique_two_ties_probability : 
  probability_two_ties 4 8 18 = 51 / 145 := by
  sorry

end NUMINAMATH_CALUDE_michelle_boutique_two_ties_probability_l451_45102


namespace NUMINAMATH_CALUDE_count_99_is_stone_10_l451_45119

/-- Represents the number of stones in the circle -/
def num_stones : ℕ := 14

/-- Represents the length of a full counting cycle -/
def cycle_length : ℕ := 2 * num_stones - 1

/-- Maps a count to its corresponding stone number -/
def count_to_stone (count : ℕ) : ℕ :=
  let adjusted_count := (count - 1) % cycle_length + 1
  if adjusted_count ≤ num_stones
  then adjusted_count
  else 2 * num_stones - adjusted_count

/-- Theorem stating that the 99th count corresponds to the 10th stone -/
theorem count_99_is_stone_10 : count_to_stone 99 = 10 := by
  sorry

end NUMINAMATH_CALUDE_count_99_is_stone_10_l451_45119


namespace NUMINAMATH_CALUDE_two_adults_in_group_l451_45187

/-- Represents the restaurant bill problem --/
def restaurant_bill_problem (num_children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : Prop :=
  ∃ (num_adults : ℕ), 
    num_adults * meal_cost + num_children * meal_cost = total_bill

/-- Proves that there are 2 adults in the group --/
theorem two_adults_in_group : 
  restaurant_bill_problem 5 3 21 → 
  ∃ (num_adults : ℕ), num_adults = 2 ∧ restaurant_bill_problem 5 3 21 :=
by sorry

end NUMINAMATH_CALUDE_two_adults_in_group_l451_45187


namespace NUMINAMATH_CALUDE_problem_statement_l451_45105

theorem problem_statement (a b c : ℕ+) 
  (h : (18 ^ a.val) * (9 ^ (3 * a.val - 1)) * (c ^ (2 * a.val - 3)) = (2 ^ 7) * (3 ^ b.val)) :
  a = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l451_45105


namespace NUMINAMATH_CALUDE_team_selection_count_l451_45115

theorem team_selection_count (n : ℕ) (k : ℕ) (h1 : n = 17) (h2 : k = 4) :
  Nat.choose n k = 2380 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l451_45115


namespace NUMINAMATH_CALUDE_bee_swarm_count_l451_45160

theorem bee_swarm_count : ∃ x : ℕ, 
  x > 0 ∧ 
  (x / 5 : ℚ) + (x / 3 : ℚ) + 3 * ((x / 3 : ℚ) - (x / 5 : ℚ)) + 1 = x ∧ 
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bee_swarm_count_l451_45160


namespace NUMINAMATH_CALUDE_cube_root_square_root_l451_45168

theorem cube_root_square_root (x : ℝ) : (2 * x)^3 = 216 → (x + 6)^(1/2) = 3 ∨ (x + 6)^(1/2) = -3 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_square_root_l451_45168


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l451_45158

/-- A linear function in the form y = mx + b -/
structure LinearFunction where
  m : ℝ
  b : ℝ

/-- The four quadrants in a Cartesian coordinate system -/
inductive Quadrant
  | I
  | II
  | III
  | IV

/-- Determines if a point (x, y) is in a given quadrant -/
def inQuadrant (x y : ℝ) (q : Quadrant) : Prop :=
  match q with
  | Quadrant.I => x > 0 ∧ y > 0
  | Quadrant.II => x < 0 ∧ y > 0
  | Quadrant.III => x < 0 ∧ y < 0
  | Quadrant.IV => x > 0 ∧ y < 0

/-- Determines if a linear function passes through a given quadrant -/
def passesThrough (f : LinearFunction) (q : Quadrant) : Prop :=
  ∃ x y : ℝ, y = f.m * x + f.b ∧ inQuadrant x y q

/-- The main theorem: y = 2x - 3 does not pass through Quadrant II -/
theorem linear_function_not_in_quadrant_II :
  ¬ passesThrough { m := 2, b := -3 } Quadrant.II :=
  sorry

end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l451_45158


namespace NUMINAMATH_CALUDE_midpoint_chain_l451_45167

theorem midpoint_chain (A B C D E F G : ℝ) : 
  (C = (A + B) / 2) →  -- C is midpoint of AB
  (D = (A + C) / 2) →  -- D is midpoint of AC
  (E = (A + D) / 2) →  -- E is midpoint of AD
  (F = (A + E) / 2) →  -- F is midpoint of AE
  (G = (A + F) / 2) →  -- G is midpoint of AF
  (G - A = 4) →        -- AG = 4
  (B - A = 128) :=     -- AB = 128
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l451_45167


namespace NUMINAMATH_CALUDE_average_difference_l451_45191

-- Define the number of students and teachers
def num_students : ℕ := 120
def num_teachers : ℕ := 6

-- Define the class enrollments
def class_enrollments : List ℕ := [40, 30, 30, 10, 5, 5]

-- Define t (average number of students per teacher)
def t : ℚ := (num_students : ℚ) / num_teachers

-- Define s (average number of students per student)
def s : ℚ := (List.sum (List.map (λ x => x * x) class_enrollments) : ℚ) / num_students

-- Theorem to prove
theorem average_difference : t - s = -29/3 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l451_45191


namespace NUMINAMATH_CALUDE_meters_examined_l451_45139

/-- The percentage of meters rejected as defective -/
def rejection_rate : ℝ := 0.10

/-- The number of defective meters found -/
def defective_meters : ℕ := 10

/-- The total number of meters examined -/
def total_meters : ℕ := 100

/-- Theorem stating that if the rejection rate is 10% and 10 defective meters are found,
    then the total number of meters examined is 100 -/
theorem meters_examined (h : ℝ) (defective : ℕ) (total : ℕ) 
  (h_rate : h = rejection_rate)
  (h_defective : defective = defective_meters)
  (h_total : total = total_meters) :
  ↑defective = h * ↑total := by
  sorry

end NUMINAMATH_CALUDE_meters_examined_l451_45139


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l451_45118

/-- The repeating decimal 0.37246̄ expressed as a rational number -/
def repeating_decimal : ℚ := 
  37 / 100 + (246 / 999900)

/-- The target fraction -/
def target_fraction : ℚ := 3718740 / 999900

/-- Theorem stating that the repeating decimal 0.37246̄ is equal to 3718740/999900 -/
theorem repeating_decimal_equals_fraction : 
  repeating_decimal = target_fraction := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l451_45118


namespace NUMINAMATH_CALUDE_integer_operation_proof_l451_45107

theorem integer_operation_proof (n : ℤ) : 5 * (n - 2) = 85 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_integer_operation_proof_l451_45107


namespace NUMINAMATH_CALUDE_max_value_theorem_l451_45131

theorem max_value_theorem (a b c : ℝ) 
  (ha : -1 ≤ a ∧ a ≤ 1) 
  (hb : -1 ≤ b ∧ b ≤ 1) 
  (hc : -1 ≤ c ∧ c ≤ 1) : 
  Real.sqrt (a^2 * b^2 * c^2) + Real.sqrt ((1 - a^2) * (1 - b^2) * (1 - c^2)) ≤ 1 ∧ 
  ∃ (x y z : ℝ), -1 ≤ x ∧ x ≤ 1 ∧ -1 ≤ y ∧ y ≤ 1 ∧ -1 ≤ z ∧ z ≤ 1 ∧ 
    Real.sqrt (x^2 * y^2 * z^2) + Real.sqrt ((1 - x^2) * (1 - y^2) * (1 - z^2)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l451_45131


namespace NUMINAMATH_CALUDE_triangle_theorem_l451_45110

-- Define a triangle with sides a, b, c opposite to angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given equation
def satisfies_equation (t : Triangle) : Prop :=
  (t.b + t.c) / (2 * t.a * t.b * t.c) + (Real.cos t.B + Real.cos t.C - 2) / (t.b^2 + t.c^2 - t.a^2) = 0

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.b + t.c = 2 * t.a

-- Define the additional conditions
def has_specific_area_and_cosA (t : Triangle) : Prop :=
  t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 7 / 4 ∧ Real.cos t.A = 9/16

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  satisfies_equation t →
  is_arithmetic_sequence t ∧
  (has_specific_area_and_cosA t → t.a = 5 * Real.sqrt 6 / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l451_45110


namespace NUMINAMATH_CALUDE_exponential_inequality_l451_45161

theorem exponential_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  3^a + 2*a = 3^b + 3*b → a > b :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_l451_45161


namespace NUMINAMATH_CALUDE_problem_solution_l451_45132

theorem problem_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 / b = 2) (h2 : b^2 / c = 3) (h3 : c^2 / a = 4) :
  a = 576^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l451_45132


namespace NUMINAMATH_CALUDE_tangent_line_is_correct_l451_45136

/-- The circle with equation x^2 + y^2 = 20 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 20}

/-- The point M (2, -4) -/
def M : ℝ × ℝ := (2, -4)

/-- The proposed tangent line with equation x - 2y - 10 = 0 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 - 2*p.2 - 10 = 0}

theorem tangent_line_is_correct :
  ∀ p ∈ TangentLine,
    (p ∈ Circle → p = M) ∧
    (∀ q ∈ Circle, q ≠ M → (p.1 - M.1) * (q.1 - M.1) + (p.2 - M.2) * (q.2 - M.2) = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_is_correct_l451_45136


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l451_45109

theorem matrix_inverse_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, 5; 3, 2]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-2, 5; 3, -7]
  A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l451_45109


namespace NUMINAMATH_CALUDE_initial_money_calculation_l451_45157

/-- Proves that if a person spends half of their initial money, then half of the remaining money, 
    and is left with 1250 won, their initial amount was 5000 won. -/
theorem initial_money_calculation (initial_money : ℝ) : 
  (initial_money / 2) / 2 = 1250 → initial_money = 5000 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l451_45157


namespace NUMINAMATH_CALUDE_rectangular_prism_paint_l451_45199

theorem rectangular_prism_paint (m n r : ℕ) : 
  0 < m ∧ 0 < n ∧ 0 < r →
  m ≤ n ∧ n ≤ r →
  (m - 2) * (n - 2) * (r - 2) + 
  (4 * (m - 2) + 4 * (n - 2) + 4 * (r - 2)) - 
  (2 * (m - 2) * (n - 2) + 2 * (m - 2) * (r - 2) + 2 * (n - 2) * (r - 2)) = 1985 →
  ((m = 5 ∧ n = 7 ∧ r = 663) ∨
   (m = 5 ∧ n = 5 ∧ r = 1981) ∨
   (m = 3 ∧ n = 3 ∧ r = 1981) ∨
   (m = 1 ∧ n = 7 ∧ r = 399) ∨
   (m = 1 ∧ n = 3 ∧ r = 1987)) := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_paint_l451_45199


namespace NUMINAMATH_CALUDE_nut_mixture_ratio_l451_45129

/-- Given a mixture of nuts where the ratio of almonds to walnuts is x:2 by weight,
    and there are 200 pounds of almonds in 280 pounds of the mixture,
    prove that the ratio of almonds to walnuts is 2.5:1. -/
theorem nut_mixture_ratio (x : ℝ) : 
  x / 2 = 200 / 80 → x / 2 = 2.5 := by sorry

end NUMINAMATH_CALUDE_nut_mixture_ratio_l451_45129


namespace NUMINAMATH_CALUDE_complex_equation_solution_l451_45151

theorem complex_equation_solution (a : ℝ) : 
  (1 : ℂ) + a * I = I * (2 - I) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l451_45151


namespace NUMINAMATH_CALUDE_total_initial_money_l451_45123

/-- Represents the money redistribution game between three friends --/
structure MoneyRedistribution where
  amy_initial : ℝ
  jan_initial : ℝ
  toy_initial : ℝ
  amy_final : ℝ
  jan_final : ℝ
  toy_final : ℝ

/-- The theorem stating the total initial amount of money --/
theorem total_initial_money (game : MoneyRedistribution) 
  (h1 : game.amy_initial = 50)
  (h2 : game.toy_initial = 50)
  (h3 : game.amy_final = game.amy_initial)
  (h4 : game.toy_final = game.toy_initial)
  (h5 : game.amy_final = 2 * (2 * (game.amy_initial - (game.jan_initial + game.toy_initial))))
  (h6 : game.jan_final = 2 * (2 * game.jan_initial - (2 * game.toy_initial + (game.amy_initial - (game.jan_initial + game.toy_initial)))))
  (h7 : game.toy_final = 2 * game.toy_initial - (game.amy_final + game.jan_final - (2 * game.toy_initial + 2 * (game.amy_initial - (game.jan_initial + game.toy_initial)))))
  : game.amy_initial + game.jan_initial + game.toy_initial = 187.5 := by
  sorry

end NUMINAMATH_CALUDE_total_initial_money_l451_45123


namespace NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_for_subset_condition_l451_45149

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x - 1|

theorem solution_set_for_negative_one :
  {x : ℝ | f (-1) x ≤ 2} = {x : ℝ | x = 1/2 ∨ x = -1/2} := by sorry

theorem range_of_a_for_subset_condition :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2) 1, f a x ≤ |2*x + 1|) → a ∈ Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_for_negative_one_range_of_a_for_subset_condition_l451_45149


namespace NUMINAMATH_CALUDE_bill_bouquets_to_buy_l451_45193

/-- Represents the rose business scenario for Bill --/
structure RoseBusiness where
  buy_roses_per_bouquet : ℕ
  buy_price_per_bouquet : ℕ
  sell_roses_per_bouquet : ℕ
  sell_price_per_bouquet : ℕ

/-- Calculates the number of bouquets Bill needs to buy to earn a specific profit --/
def bouquets_to_buy (rb : RoseBusiness) (target_profit : ℕ) : ℕ :=
  let buy_bouquets := rb.sell_roses_per_bouquet
  let sell_bouquets := rb.buy_roses_per_bouquet
  let profit_per_operation := sell_bouquets * rb.sell_price_per_bouquet - buy_bouquets * rb.buy_price_per_bouquet
  let operations_needed := target_profit / profit_per_operation
  operations_needed * buy_bouquets

/-- Theorem stating that Bill needs to buy 125 bouquets to earn $1000 --/
theorem bill_bouquets_to_buy :
  let rb : RoseBusiness := {
    buy_roses_per_bouquet := 7,
    buy_price_per_bouquet := 20,
    sell_roses_per_bouquet := 5,
    sell_price_per_bouquet := 20
  }
  bouquets_to_buy rb 1000 = 125 := by sorry

end NUMINAMATH_CALUDE_bill_bouquets_to_buy_l451_45193


namespace NUMINAMATH_CALUDE_triangle_angle_obtuse_l451_45194

theorem triangle_angle_obtuse (α : Real) (h1 : 0 < α ∧ α < π) 
  (h2 : Real.sin α + Real.cos α = 2/3) : α > π/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_obtuse_l451_45194


namespace NUMINAMATH_CALUDE_ned_trips_theorem_l451_45124

/-- The number of trays Ned can carry in one trip -/
def trays_per_trip : ℕ := 8

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 27

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 5

/-- The total number of trays Ned needs to pick up -/
def total_trays : ℕ := trays_table1 + trays_table2

/-- The number of trips Ned will make -/
def num_trips : ℕ := (total_trays + trays_per_trip - 1) / trays_per_trip

theorem ned_trips_theorem : num_trips = 4 := by
  sorry

end NUMINAMATH_CALUDE_ned_trips_theorem_l451_45124


namespace NUMINAMATH_CALUDE_slowest_racer_time_l451_45156

/-- Represents the time taken by each person to reach the top floor -/
structure RaceTime where
  lola : ℕ
  tara : ℕ
  sam : ℕ

/-- Calculates the race times given the building parameters -/
def calculateRaceTimes (
  totalStories : ℕ
  ) (lolaTimePerStory : ℕ
  ) (samTimePerStory : ℕ
  ) (elevatorTimePerStory : ℕ
  ) (elevatorStopTime : ℕ
  ) (samSwitchFloor : ℕ
  ) (elevatorWaitTime : ℕ
  ) : RaceTime :=
  { lola := totalStories * lolaTimePerStory,
    tara := totalStories * elevatorTimePerStory + (totalStories - 1) * elevatorStopTime,
    sam := samSwitchFloor * samTimePerStory + elevatorWaitTime +
           (totalStories - samSwitchFloor) * elevatorTimePerStory +
           (totalStories - samSwitchFloor - 1) * elevatorStopTime }

/-- The main theorem to prove -/
theorem slowest_racer_time (
  totalStories : ℕ
  ) (lolaTimePerStory : ℕ
  ) (samTimePerStory : ℕ
  ) (elevatorTimePerStory : ℕ
  ) (elevatorStopTime : ℕ
  ) (samSwitchFloor : ℕ
  ) (elevatorWaitTime : ℕ
  ) (h1 : totalStories = 50
  ) (h2 : lolaTimePerStory = 12
  ) (h3 : samTimePerStory = 15
  ) (h4 : elevatorTimePerStory = 10
  ) (h5 : elevatorStopTime = 4
  ) (h6 : samSwitchFloor = 25
  ) (h7 : elevatorWaitTime = 20
  ) : (
    let times := calculateRaceTimes totalStories lolaTimePerStory samTimePerStory
                   elevatorTimePerStory elevatorStopTime samSwitchFloor elevatorWaitTime
    max times.lola (max times.tara times.sam) = 741
  ) := by
  sorry

end NUMINAMATH_CALUDE_slowest_racer_time_l451_45156


namespace NUMINAMATH_CALUDE_ben_winning_strategy_l451_45174

/-- Represents the state of the chocolate bar game -/
structure ChocolateBar where
  m : ℕ
  n : ℕ

/-- Determines if a player has a winning strategy given the current state of the game -/
def has_winning_strategy (state : ChocolateBar) : Prop :=
  ∃ (k : ℕ), (state.m + 1) = 2^k * (state.n + 1) ∨
             (state.n + 1) = 2^k * (state.m + 1) ∨
             ∃ (x : ℕ), x ≤ state.m ∧
                        ((state.m + 1 - x) = 2^k * (state.n + 1) ∨
                         (state.n + 1) = 2^k * (state.m + 1 - x))

/-- Theorem: Ben has a winning strategy if and only if the ratio can be made a power of two -/
theorem ben_winning_strategy (initial_state : ChocolateBar) :
  has_winning_strategy initial_state ↔
  ∃ (a k : ℕ), a ≥ 2 ∧ k ≥ 0 ∧
    ((initial_state.m = a - 1 ∧ initial_state.n = 2^k * a - 1) ∨
     (initial_state.m = 2^k * a - 1 ∧ initial_state.n = a - 1)) :=
sorry

end NUMINAMATH_CALUDE_ben_winning_strategy_l451_45174


namespace NUMINAMATH_CALUDE_find_d_l451_45112

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x - 3

-- State the theorem
theorem find_d (c : ℝ) (d : ℝ) : 
  (∀ x, f c (g c x) = -15 * x + d) → d = -18 := by
  sorry

end NUMINAMATH_CALUDE_find_d_l451_45112


namespace NUMINAMATH_CALUDE_triangle_properties_l451_45186

/-- Given a triangle ABC with interior angles A, B, and C, prove the magnitude of A and the maximum perimeter. -/
theorem triangle_properties (A B C : Real) (R : Real) : 
  -- Conditions
  A + B + C = π ∧ 
  (Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2) ∧
  R = 2 →
  -- Conclusions
  A = 2*π/3 ∧ 
  ∃ (L : Real), L = 2*Real.sqrt 3 + 4 ∧ 
    ∀ (a b c : Real), 
      a / Real.sin A = 2*R → 
      a^2 = b^2 + c^2 - 2*b*c*Real.cos A → 
      a + b + c ≤ L :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l451_45186


namespace NUMINAMATH_CALUDE_breakfast_cost_is_17_l451_45152

/-- The cost of breakfast for Francis and Kiera -/
def breakfast_cost (muffin_price fruit_cup_price : ℕ) 
  (francis_muffins francis_fruit_cups kiera_muffins kiera_fruit_cups : ℕ) : ℕ :=
  (francis_muffins + kiera_muffins) * muffin_price + 
  (francis_fruit_cups + kiera_fruit_cups) * fruit_cup_price

/-- Theorem stating that the total cost of breakfast for Francis and Kiera is $17 -/
theorem breakfast_cost_is_17 : 
  breakfast_cost 2 3 2 2 2 1 = 17 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_cost_is_17_l451_45152


namespace NUMINAMATH_CALUDE_petyas_run_l451_45190

theorem petyas_run (V D : ℝ) (hV : V > 0) (hD : D > 0) : 
  D / (2 * 1.25 * V) + D / (2 * 0.8 * V) > D / V := by
  sorry

end NUMINAMATH_CALUDE_petyas_run_l451_45190


namespace NUMINAMATH_CALUDE_power_equality_l451_45148

theorem power_equality (y x : ℕ) (h1 : 9^y = 3^x) (h2 : y = 6) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l451_45148


namespace NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l451_45130

/-- Given a sum of money that doubles itself in 5 years at simple interest,
    prove that the rate percent per annum is 20%. -/
theorem simple_interest_rate_for_doubling (P : ℝ) (h : P > 0) : 
  ∃ (R : ℝ), R > 0 ∧ R ≤ 100 ∧ P + (P * R * 5) / 100 = 2 * P ∧ R = 20 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_rate_for_doubling_l451_45130


namespace NUMINAMATH_CALUDE_platform_length_calculation_l451_45171

/-- Given a train of length 300 meters that crosses a platform in 51 seconds
    and a signal pole in 18 seconds, prove that the length of the platform
    is approximately 550.17 meters. -/
theorem platform_length_calculation (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 51)
  (h3 : pole_crossing_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 550.17) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_platform_length_calculation_l451_45171


namespace NUMINAMATH_CALUDE_quadratic_inequality_l451_45108

/-- The quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The solution set of ax^2 + bx + c > 0 is {x | -2 < x < 4} -/
def solution_set (a b c : ℝ) : Set ℝ := {x | -2 < x ∧ x < 4}

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x > 0) :
  f a b c 2 > f a b c (-1) ∧ f a b c (-1) > f a b c 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l451_45108


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l451_45150

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 12)
  (sum_squares_condition : a^2 / (1 - r^2) = 54) :
  a = 72 / 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l451_45150


namespace NUMINAMATH_CALUDE_coffee_milk_ratio_result_l451_45198

/-- Represents the coffee and milk consumption problem -/
def coffee_milk_ratio (thermos_capacity : ℚ) (fills_per_day : ℕ) (school_days : ℕ) 
  (coffee_reduction_factor : ℚ) (current_coffee_per_week : ℚ) : Prop :=
  let total_capacity_per_week := thermos_capacity * fills_per_day * school_days
  let previous_coffee_per_week := current_coffee_per_week / coffee_reduction_factor
  let milk_per_week := total_capacity_per_week - previous_coffee_per_week
  let milk_per_fill := milk_per_week / (fills_per_day * school_days)
  (milk_per_fill : ℚ) / thermos_capacity = 1 / 5

/-- The main theorem stating the ratio of milk to thermos capacity -/
theorem coffee_milk_ratio_result : 
  coffee_milk_ratio 20 2 5 (1/4) 40 := by sorry

end NUMINAMATH_CALUDE_coffee_milk_ratio_result_l451_45198


namespace NUMINAMATH_CALUDE_constant_function_from_surjective_injective_l451_45120

theorem constant_function_from_surjective_injective
  (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
sorry

end NUMINAMATH_CALUDE_constant_function_from_surjective_injective_l451_45120


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l451_45138

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ > 2 ∧ x₂ > 2 ∧ 
   x₁^2 + (k-2)*x₁ + 5 - k = 0 ∧ 
   x₂^2 + (k-2)*x₂ + 5 - k = 0) → 
  -5 < k ∧ k < -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l451_45138


namespace NUMINAMATH_CALUDE_crackers_distribution_l451_45101

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 36 →
  num_friends = 6 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 6 :=
by sorry

end NUMINAMATH_CALUDE_crackers_distribution_l451_45101


namespace NUMINAMATH_CALUDE_scatter_plot_regression_role_l451_45117

/-- The role of a scatter plot in regression analysis -/
def scatter_plot_role : String :=
  "to roughly judge whether variables are linearly related"

/-- The main theorem about the role of scatter plots in regression analysis -/
theorem scatter_plot_regression_role :
  scatter_plot_role = "to roughly judge whether variables are linearly related" := by
  sorry

end NUMINAMATH_CALUDE_scatter_plot_regression_role_l451_45117


namespace NUMINAMATH_CALUDE_angle_sum_in_cyclic_quad_l451_45164

-- Define the quadrilateral ABCD and point E
variable (A B C D E : Point)

-- Define the cyclic property of quadrilateral ABCD
def is_cyclic_quad (A B C D : Point) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem angle_sum_in_cyclic_quad 
  (h_cyclic : is_cyclic_quad A B C D)
  (h_angle_A : angle_measure B A C = 40)
  (h_equal_angles : angle_measure C E D = angle_measure E C D) :
  angle_measure A B C + angle_measure A D C = 160 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_cyclic_quad_l451_45164


namespace NUMINAMATH_CALUDE_delores_remaining_money_l451_45188

/-- Calculates the remaining money after purchasing a computer and printer -/
def remaining_money (initial_amount computer_cost printer_cost : ℕ) : ℕ :=
  initial_amount - (computer_cost + printer_cost)

/-- Theorem: Given the specific amounts, the remaining money is $10 -/
theorem delores_remaining_money :
  remaining_money 450 400 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_delores_remaining_money_l451_45188


namespace NUMINAMATH_CALUDE_right_triangle_with_specific_median_l451_45122

/-- A right triangle with a median to the hypotenuse -/
structure RightTriangleWithMedian where
  a : ℝ  -- First leg
  b : ℝ  -- Second leg
  c : ℝ  -- Hypotenuse
  m : ℝ  -- Median to the hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  median_property : m = c / 2  -- Property of the median to the hypotenuse
  perimeter_difference : b - a = 1  -- Difference in perimeters of smaller triangles

/-- The sides of the triangle satisfy the given conditions -/
theorem right_triangle_with_specific_median (t : RightTriangleWithMedian) 
  (h1 : t.a + t.m = 8)  -- Perimeter of one smaller triangle
  (h2 : t.b + t.m = 9)  -- Perimeter of the other smaller triangle
  : t.a = 3 ∧ t.b = 4 ∧ t.c = 5 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_with_specific_median_l451_45122


namespace NUMINAMATH_CALUDE_polar_to_rectangular_l451_45143

/-- The rectangular coordinate equation equivalent to the polar equation ρ = 4sin θ -/
theorem polar_to_rectangular (x y ρ θ : ℝ) 
  (h1 : ρ = 4 * Real.sin θ)
  (h2 : x = ρ * Real.cos θ)
  (h3 : y = ρ * Real.sin θ)
  (h4 : ρ^2 = x^2 + y^2) :
  x^2 + y^2 - 4*y = 0 := by
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_l451_45143


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l451_45121

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 5) → x^2 - 2*(a-2)*x + a > 0) → 
  a ∈ Set.Ioo 1 5 ∪ Set.singleton 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l451_45121


namespace NUMINAMATH_CALUDE_expression_value_l451_45127

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l451_45127


namespace NUMINAMATH_CALUDE_number_equation_l451_45185

theorem number_equation (x : ℝ) (n : ℝ) : x = 32 → (35 - (n - (15 - x)) = 12 * 2 / (1 / 2)) ↔ n = -30 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l451_45185


namespace NUMINAMATH_CALUDE_suitcase_electronics_weight_l451_45195

/-- Given a suitcase with books, clothes, and electronics, prove the weight of electronics. -/
theorem suitcase_electronics_weight 
  (B C E : ℝ) -- Weights of books, clothes, and electronics
  (h1 : B / C = 7 / 4) -- Initial ratio of books to clothes
  (h2 : C / E = 4 / 3) -- Initial ratio of clothes to electronics
  (h3 : B / (C - 6) = 2 * (B / C)) -- Ratio doubles after removing 6 pounds of clothes
  : E = 9 := by
  sorry

end NUMINAMATH_CALUDE_suitcase_electronics_weight_l451_45195


namespace NUMINAMATH_CALUDE_watermelon_size_ratio_l451_45175

/-- Given information about watermelons grown by Michael, Clay, and John, 
    prove the ratio of Clay's watermelon size to Michael's watermelon size. -/
theorem watermelon_size_ratio 
  (michael_weight : ℝ) 
  (john_weight : ℝ) 
  (h1 : michael_weight = 8)
  (h2 : john_weight = 12)
  (h3 : ∃ (clay_weight : ℝ), clay_weight = 2 * john_weight) :
  ∃ (clay_weight : ℝ), clay_weight / michael_weight = 3 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_size_ratio_l451_45175


namespace NUMINAMATH_CALUDE_marble_ratio_l451_45172

/-- Represents the number of marbles each person has -/
structure Marbles where
  atticus : ℕ
  jensen : ℕ
  cruz : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  3 * (m.atticus + m.jensen + m.cruz) = 60 ∧
  m.atticus = 4 ∧
  m.cruz = 8

/-- The theorem stating the ratio of Atticus's marbles to Jensen's marbles -/
theorem marble_ratio (m : Marbles) :
  marble_conditions m → m.atticus * 2 = m.jensen := by
  sorry

#check marble_ratio

end NUMINAMATH_CALUDE_marble_ratio_l451_45172


namespace NUMINAMATH_CALUDE_complex_number_problem_l451_45134

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 →
  z₂ = 4 + 2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l451_45134


namespace NUMINAMATH_CALUDE_council_arrangements_l451_45163

/-- The number of distinct arrangements of chairs and stools around a round table -/
def distinct_arrangements (chairs : ℕ) (stools : ℕ) : ℕ :=
  Nat.choose (chairs + stools - 1) (stools - 1)

/-- Theorem: There are 55 distinct arrangements of 9 chairs and 3 stools around a round table -/
theorem council_arrangements :
  distinct_arrangements 9 3 = 55 := by
  sorry

end NUMINAMATH_CALUDE_council_arrangements_l451_45163


namespace NUMINAMATH_CALUDE_joshua_skittles_l451_45146

/-- The number of friends Joshua has -/
def num_friends : ℕ := 5

/-- The number of Skittles each friend would get if Joshua shares them equally -/
def skittles_per_friend : ℕ := 8

/-- The total number of Skittles Joshua has -/
def total_skittles : ℕ := num_friends * skittles_per_friend

/-- Theorem: Joshua has 40 Skittles -/
theorem joshua_skittles : total_skittles = 40 := by
  sorry

end NUMINAMATH_CALUDE_joshua_skittles_l451_45146


namespace NUMINAMATH_CALUDE_sum_of_roots_2016_l451_45145

/-- The function f(x) = x^2 - 2016x + 2015 -/
def f (x : ℝ) : ℝ := x^2 - 2016*x + 2015

/-- Theorem: If f(a) = f(b) = c for distinct a and b, then a + b = 2016 -/
theorem sum_of_roots_2016 (a b c : ℝ) (ha : f a = c) (hb : f b = c) (hab : a ≠ b) : a + b = 2016 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_2016_l451_45145


namespace NUMINAMATH_CALUDE_sequence_a_11_l451_45192

theorem sequence_a_11 (a : ℕ+ → ℚ) (S : ℕ+ → ℚ) 
  (h : ∀ n : ℕ+, 4 * S n = 2 * a n - n.val^2 + 7 * n.val) : 
  a 11 = -2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_11_l451_45192


namespace NUMINAMATH_CALUDE_smallest_value_complex_sum_l451_45133

theorem smallest_value_complex_sum (x y z : ℕ) (θ : ℂ) 
  (hxyz : x < y ∧ y < z)
  (hθ4 : θ^4 = 1)
  (hθ_neq_1 : θ ≠ 1) :
  ∃ (w : ℕ), w > 0 ∧ ∀ (a b c : ℕ) (ϕ : ℂ),
    a < b ∧ b < c → ϕ^4 = 1 → ϕ ≠ 1 →
    Complex.abs (↑x + ↑y * θ + ↑z * θ^3) ≤ Complex.abs (↑a + ↑b * ϕ + ↑c * ϕ^3) ∧
    Complex.abs (↑x + ↑y * θ + ↑z * θ^3) = Real.sqrt (↑w) :=
sorry

end NUMINAMATH_CALUDE_smallest_value_complex_sum_l451_45133


namespace NUMINAMATH_CALUDE_expression_evaluation_l451_45114

theorem expression_evaluation :
  let x : ℚ := 1/2
  (2*x - 1)^2 - (3*x + 1)*(3*x - 1) + 5*x*(x - 1) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l451_45114


namespace NUMINAMATH_CALUDE_michaels_fish_count_l451_45140

/-- Given Michael's initial fish count and the number of fish Ben gives him,
    prove that the total number of fish Michael has now is equal to the sum of these two quantities. -/
theorem michaels_fish_count (initial : Real) (given : Real) :
  initial + given = initial + given :=
by sorry

end NUMINAMATH_CALUDE_michaels_fish_count_l451_45140


namespace NUMINAMATH_CALUDE_coupon_problem_l451_45153

/-- Calculates the total number of bottles that would be received if no coupons were lost -/
def total_bottles (bottles_per_coupon : ℕ) (lost_coupons : ℕ) (remaining_coupons : ℕ) : ℕ :=
  (remaining_coupons + lost_coupons) * bottles_per_coupon

/-- Proves that given the conditions, the total number of bottles would be 21 -/
theorem coupon_problem :
  let bottles_per_coupon : ℕ := 3
  let lost_coupons : ℕ := 3
  let remaining_coupons : ℕ := 4
  total_bottles bottles_per_coupon lost_coupons remaining_coupons = 21 := by
  sorry

end NUMINAMATH_CALUDE_coupon_problem_l451_45153


namespace NUMINAMATH_CALUDE_power_of_two_divisibility_l451_45154

theorem power_of_two_divisibility (n : ℕ) (hn : n ≥ 1) :
  (∃ k : ℕ, 2^n - 1 = 3 * k) ∧
  (∃ m : ℕ, m ≥ 1 ∧ ∃ l : ℕ, (2^n - 1) / 3 * l = 4 * m^2 + 1) →
  ∃ r : ℕ, n = 2^r :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_divisibility_l451_45154


namespace NUMINAMATH_CALUDE_network_paths_count_l451_45170

-- Define the network structure
structure Network where
  hasDirectPath : (Char × Char) → Prop
  
-- Define the number of paths between two points
def numPaths (net : Network) (start finish : Char) : ℕ := sorry

-- Theorem statement
theorem network_paths_count (net : Network) :
  (net.hasDirectPath ('E', 'B')) →
  (net.hasDirectPath ('F', 'B')) →
  (net.hasDirectPath ('F', 'A')) →
  (net.hasDirectPath ('M', 'A')) →
  (net.hasDirectPath ('M', 'B')) →
  (net.hasDirectPath ('M', 'E')) →
  (net.hasDirectPath ('M', 'F')) →
  (net.hasDirectPath ('A', 'C')) →
  (net.hasDirectPath ('A', 'D')) →
  (net.hasDirectPath ('B', 'A')) →
  (net.hasDirectPath ('B', 'C')) →
  (net.hasDirectPath ('B', 'N')) →
  (net.hasDirectPath ('C', 'N')) →
  (net.hasDirectPath ('D', 'N')) →
  (numPaths net 'M' 'N' = 16) :=
by sorry

end NUMINAMATH_CALUDE_network_paths_count_l451_45170


namespace NUMINAMATH_CALUDE_intersection_M_N_l451_45176

def M : Set ℝ := {y | ∃ x, y = |Real.cos x ^ 2 - Real.sin x ^ 2|}

def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x^2)}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l451_45176


namespace NUMINAMATH_CALUDE_common_prime_root_quadratics_l451_45111

theorem common_prime_root_quadratics (a b : ℤ) : 
  (∃ p : ℕ, Prime p ∧ 
    (p : ℤ)^2 + a * p + b = 0 ∧ 
    (p : ℤ)^2 + b * p + 1100 = 0) →
  a = 274 ∨ a = 40 :=
by sorry

end NUMINAMATH_CALUDE_common_prime_root_quadratics_l451_45111


namespace NUMINAMATH_CALUDE_part1_part2_l451_45100

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - 3*a) / (a - 2*x) ≥ 0 ∧ a > 0

def q (x : ℝ) : Prop := 2*x^2 - 7*x + 6 < 0

-- Part 1
theorem part1 (x : ℝ) : 
  p x 1 ∧ q x → 3/2 < x ∧ x < 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x a) → 
  2/3 ≤ a ∧ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_part1_part2_l451_45100


namespace NUMINAMATH_CALUDE_wire_cutting_problem_l451_45155

theorem wire_cutting_problem :
  let total_length : ℕ := 102
  let piece_length_1 : ℕ := 15
  let piece_length_2 : ℕ := 12
  ∀ x y : ℕ, piece_length_1 * x + piece_length_2 * y = total_length →
    (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_wire_cutting_problem_l451_45155


namespace NUMINAMATH_CALUDE_packages_per_box_l451_45181

/-- Given that Julie bought two boxes of standard paper, each package contains 250 sheets,
    25 sheets are used per newspaper, and 100 newspapers can be printed,
    prove that there are 5 packages in each box. -/
theorem packages_per_box (boxes : ℕ) (sheets_per_package : ℕ) (sheets_per_newspaper : ℕ) (total_newspapers : ℕ) :
  boxes = 2 →
  sheets_per_package = 250 →
  sheets_per_newspaper = 25 →
  total_newspapers = 100 →
  (boxes * sheets_per_package * (total_newspapers * sheets_per_newspaper / (boxes * sheets_per_package)) : ℚ) = total_newspapers * sheets_per_newspaper →
  total_newspapers * sheets_per_newspaper / (boxes * sheets_per_package) = 5 := by
  sorry

end NUMINAMATH_CALUDE_packages_per_box_l451_45181


namespace NUMINAMATH_CALUDE_total_legs_in_pasture_l451_45166

/-- The number of cows in the pasture -/
def num_cows : ℕ := 115

/-- The number of legs each cow has -/
def legs_per_cow : ℕ := 4

/-- Theorem: The total number of legs seen in a pasture with 115 cows, 
    where each cow has 4 legs, is equal to 460. -/
theorem total_legs_in_pasture : num_cows * legs_per_cow = 460 := by
  sorry

end NUMINAMATH_CALUDE_total_legs_in_pasture_l451_45166


namespace NUMINAMATH_CALUDE_half_percent_of_150_in_paise_l451_45106

/-- Converts rupees to paise -/
def rupees_to_paise (r : ℚ) : ℚ := 100 * r

/-- Calculates the percentage of a given value -/
def percentage_of (p : ℚ) (v : ℚ) : ℚ := (p / 100) * v

theorem half_percent_of_150_in_paise : 
  rupees_to_paise (percentage_of 0.5 150) = 75 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_of_150_in_paise_l451_45106


namespace NUMINAMATH_CALUDE_rafael_remaining_hours_l451_45142

def hours_worked : ℕ := 18
def hourly_rate : ℕ := 20
def total_earnings : ℕ := 760

theorem rafael_remaining_hours : 
  (total_earnings - hours_worked * hourly_rate) / hourly_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_rafael_remaining_hours_l451_45142


namespace NUMINAMATH_CALUDE_betta_fish_guppies_l451_45183

/-- The number of guppies eaten by each betta fish per day -/
def guppies_per_betta : ℕ := sorry

/-- The number of guppies eaten by the moray eel per day -/
def moray_guppies : ℕ := 20

/-- The number of betta fish -/
def num_bettas : ℕ := 5

/-- The total number of guppies needed per day -/
def total_guppies : ℕ := 55

theorem betta_fish_guppies :
  guppies_per_betta = 7 ∧
  moray_guppies + num_bettas * guppies_per_betta = total_guppies :=
sorry

end NUMINAMATH_CALUDE_betta_fish_guppies_l451_45183


namespace NUMINAMATH_CALUDE_trapezoid_circle_area_ratio_l451_45177

/-- Given a trapezoid inscribed in a circle, where the larger base forms an angle α 
    with a lateral side and an angle β with the diagonal, the ratio of the area of 
    the circle to the area of the trapezoid is π / (2 sin²α sin(2β)). -/
theorem trapezoid_circle_area_ratio (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) : 
  ∃ (S_circle S_trapezoid : Real),
    S_circle > 0 ∧ S_trapezoid > 0 ∧
    S_circle / S_trapezoid = π / (2 * Real.sin α ^ 2 * Real.sin (2 * β)) :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_circle_area_ratio_l451_45177


namespace NUMINAMATH_CALUDE_red_crayons_count_l451_45128

/-- Represents the number of crayons of each color in a crayon box. -/
structure CrayonBox where
  total : ℕ
  blue : ℕ
  green : ℕ
  pink : ℕ
  red : ℕ

/-- Calculates the number of red crayons in a crayon box. -/
def redCrayons (box : CrayonBox) : ℕ :=
  box.total - (box.blue + box.green + box.pink)

/-- Theorem stating the number of red crayons in the given crayon box. -/
theorem red_crayons_count (box : CrayonBox) 
  (h1 : box.total = 24)
  (h2 : box.blue = 6)
  (h3 : box.green = 2 * box.blue / 3)
  (h4 : box.pink = 6) :
  redCrayons box = 8 := by
  sorry

#eval redCrayons { total := 24, blue := 6, green := 4, pink := 6, red := 8 }

end NUMINAMATH_CALUDE_red_crayons_count_l451_45128


namespace NUMINAMATH_CALUDE_garrison_reinforcement_departure_reinforcement_left_after_27_days_l451_45182

/-- Represents the problem of determining when reinforcements left a garrison --/
theorem garrison_reinforcement_departure (initial_men : ℕ) (initial_days : ℕ) 
  (departed_men : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_men * initial_days
  let remaining_men := initial_men - departed_men
  let x := (total_provisions - remaining_men * remaining_days) / initial_men
  x

/-- Proves that the reinforcements left after 27 days given the problem conditions --/
theorem reinforcement_left_after_27_days : 
  garrison_reinforcement_departure 400 31 200 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_garrison_reinforcement_departure_reinforcement_left_after_27_days_l451_45182


namespace NUMINAMATH_CALUDE_negation_of_proposition_l451_45184

theorem negation_of_proposition (p : Prop) : 
  (¬ (∀ x : ℝ, Real.exp x ≤ 1)) ↔ (∃ x : ℝ, Real.exp x > 1) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l451_45184


namespace NUMINAMATH_CALUDE_function_satisfies_differential_equation_l451_45197

/-- Prove that the function y = x(c - ln x) satisfies the differential equation (x - y) dx + x · dy = 0 -/
theorem function_satisfies_differential_equation (x : ℝ) (c : ℝ) :
  let y := x * (c - Real.log x)
  (x - y) * 1 + x * (c - Real.log x - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_satisfies_differential_equation_l451_45197
